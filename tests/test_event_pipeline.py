"""
Tests for event_pipeline.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. EventPipeline construction
    2. dependency injection of classifier backends
    3. acquisition-session lifecycle
    4. session input validation
    5. transactional session startup
    6. robust session shutdown
    7. rejection of audio outside an active session
    8. stale-session audio rejection
    9. detector -> persistence orchestration
    10. processing-error isolation
    11. localization-window selection
    12. short-time RMS calculation
    13. channel-quality scoring
    14. best-channel selection
    15. preservation of model audio after feature-extraction failure
    16. feature-only classification
    17. waveform-only backend support
    18. classifier-result contract enforcement
    19. complete event persistence orchestration
    20. partial persistence failure isolation
    21. environmental telemetry delegation
    22. event WAV writing

Architecture
------------
AudioBlock
    ↓
MultiNodeEventDetector
    ↓
AcousticEvent
    ├── environmental association
    ├── TDOA localization
    ├── per-node DSP preprocessing
    ├── best-channel selection
    ├── feature extraction
    ├── classification
    ├── event WAV storage
    └── SQLite persistence

Scientific scope
----------------
These tests verify software orchestration.

Localization mathematics, acoustic features, classification rules,
event detection and database behavior are tested independently in their
own unit-test modules.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import wave

from dataclasses import (
    replace,
)

from pathlib import (
    Path,
)

from types import (
    SimpleNamespace,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np
import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


import event_pipeline as event_pipeline_module


from classification.base import (
    ClassificationInput,
    ClassifierBackend,
)

from classification.classifier import (
    AcousticClass,
    ClassificationResult,
)

from config import (
    CONFIG,
)

from dsp.features import (
    AcousticFeatures,
)

from event_detector import (
    AcousticEvent,
)

from event_pipeline import (
    EventPipeline,
)

from protocol import (
    EnvironmentPayload,
)

from stream_manager import (
    StreamManager,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SESSION_ID = (
    0x12345678
)


SECOND_SESSION_ID = (
    0x87654321
)


# ======================================================================
# TEST CONFIGURATION
# ======================================================================


def make_test_config(
    tmp_path: Path,
):
    """
    Derive a temporary-filesystem AppConfig from the real project
    configuration.

    Event WAV output is disabled for orchestration tests unless the WAV
    writer itself is under test.
    """

    persistence = replace(
        CONFIG.persistence,

        database_path=
            tmp_path
            / "events.db",

        events_dir=
            tmp_path
            / "events",

        save_event_wav=
            False,
    )

    return replace(
        CONFIG,
        persistence=
            persistence,
    )


# ======================================================================
# TEST DATA HELPERS
# ======================================================================


def make_event(
    *,
    event_id: int = 1,
    session_id: int = SESSION_ID,
    start_sample: int = 1000,
    end_sample: int = 5000,
    trigger_nodes: tuple[
        int,
        ...,
    ] = (
        1,
        2,
        3,
    ),
    peak_rms_dbfs: float = -15.0,
) -> AcousticEvent:
    """
    Create one deterministic completed event.
    """

    return AcousticEvent(
        event_id,
        session_id,
        start_sample,
        end_sample,
        trigger_nodes,
        peak_rms_dbfs,
    )


def make_features() -> AcousticFeatures:
    """
    Complete deterministic feature vector.
    """

    return AcousticFeatures(
        duration_s=
            1.0,

        rms=
            0.10,

        peak_amplitude=
            0.25,

        crest_factor=
            2.5,

        zero_crossing_rate=
            0.08,

        dominant_frequency_hz=
            2500.0,

        spectral_centroid_hz=
            3200.0,

        spectral_bandwidth_hz=
            1500.0,

        spectral_rolloff_hz=
            5200.0,

        spectral_flatness=
            0.12,

        spectral_flux=
            0.03,

        snr_db=
            15.0,

        mfcc_mean=
            tuple(
                0.0
                for _ in range(
                    13
                )
            ),

        mfcc_std=
            tuple(
                0.5
                for _ in range(
                    13
                )
            ),
    )


def make_classification() -> ClassificationResult:
    """
    Valid broad acoustic classification.
    """

    scores = {
        acoustic_class.value:
            0.0

        for acoustic_class
        in AcousticClass
    }

    scores[
        AcousticClass.BIRD.value
    ] = (
        0.82
    )

    scores[
        AcousticClass.INSECT.value
    ] = (
        0.25
    )

    return ClassificationResult(
        label=
            AcousticClass.BIRD,

        confidence=
            0.82,

        second_label=
            AcousticClass.INSECT,

        second_confidence=
            0.25,

        margin=
            0.57,

        scores=
            scores,

        reasons=(
            "synthetic event-pipeline test",
        ),

        classifier_name=
            "test_classifier",

        classifier_version=
            "1.0",
    )


# ======================================================================
# FAKE DETECTOR
# ======================================================================


class FakeDetector:
    """
    Minimal MultiNodeEventDetector test double.
    """

    def __init__(
        self,
    ) -> None:

        self.reset_count = (
            0
        )

        self.process_calls = []

        self.events_to_return: list[
            AcousticEvent
        ] = []

        self.process_error: (
            Exception
            | None
        ) = None

    def reset(
        self,
    ) -> None:

        self.reset_count += (
            1
        )

    def process(
        self,
        block,
    ):

        self.process_calls.append(
            block
        )

        if (
            self.process_error
            is not None
        ):

            raise self.process_error

        return list(
            self.events_to_return
        )


# ======================================================================
# FAKE LOCALIZER
# ======================================================================


class FakeLocalizer:
    """
    Minimal LocalizationEngine test double.
    """

    def __init__(
        self,
    ) -> None:

        self.calls = []

        self.result = (
            None
        )

        self.error: (
            Exception
            | None
        ) = None

    def locate_window(
        self,
        *,
        start_sample: int,
        length: int,
    ):

        self.calls.append(
            (
                start_sample,
                length,
            )
        )

        if (
            self.error
            is not None
        ):

            raise self.error

        return (
            self.result
        )


# ======================================================================
# FAKE DATABASE
# ======================================================================


class FakeDatabase:
    """
    EventDatabase test double.

    The actual SQLite behavior is covered in test_database.py.
    """

    def __init__(
        self,
    ) -> None:

        self.start_calls = []

        self.stop_calls = []

        self.environment_calls = []

        self.event_calls = []

        self.feature_calls = []

        self.classification_calls = []

        self.start_error: (
            Exception
            | None
        ) = None

        self.stop_error: (
            Exception
            | None
        ) = None

        self.event_error: (
            Exception
            | None
        ) = None

        self.feature_error: (
            Exception
            | None
        ) = None

        self.classification_error: (
            Exception
            | None
        ) = None

        self.next_event_id = (
            101
        )

    def start_session(
        self,
        session_id: int,
        label: str,
    ) -> None:

        self.start_calls.append(
            (
                session_id,
                label,
            )
        )

        if (
            self.start_error
            is not None
        ):

            raise self.start_error

    def stop_session(
        self,
        session_id,
    ) -> None:

        self.stop_calls.append(
            session_id
        )

        if (
            self.stop_error
            is not None
        ):

            raise self.stop_error

    def add_environment(
        self,
        **kwargs,
    ) -> None:

        self.environment_calls.append(
            kwargs
        )

    def add_event(
        self,
        event,
        **kwargs,
    ) -> int:

        self.event_calls.append(
            (
                event,
                kwargs,
            )
        )

        if (
            self.event_error
            is not None
        ):

            raise self.event_error

        return (
            self.next_event_id
        )

    def add_event_features(
        self,
        **kwargs,
    ) -> None:

        self.feature_calls.append(
            kwargs
        )

        if (
            self.feature_error
            is not None
        ):

            raise self.feature_error

    def add_classification(
        self,
        **kwargs,
    ) -> None:

        self.classification_calls.append(
            kwargs
        )

        if (
            self.classification_error
            is not None
        ):

            raise self.classification_error


# ======================================================================
# CLASSIFIER TEST BACKENDS
# ======================================================================


class AudioOnlyBackend(
    ClassifierBackend
):
    """
    Waveform-only backend used to prove EventPipeline is not coupled to
    handcrafted acoustic features.
    """

    def __init__(
        self,
    ) -> None:

        self.received: (
            ClassificationInput
            | None
        ) = None

    @property
    def name(
        self,
    ) -> str:

        return (
            "audio_only_test_backend"
        )

    @property
    def version(
        self,
    ) -> str:

        return (
            "1.0"
        )

    @property
    def requires_audio(
        self,
    ) -> bool:

        return (
            True
        )

    @property
    def requires_features(
        self,
    ) -> bool:

        return (
            False
        )

    def classify(
        self,
        classification_input: ClassificationInput,
    ) -> ClassificationResult:

        self.validate_input(
            classification_input
        )

        self.received = (
            classification_input
        )

        return make_classification()


class FeatureOnlyBackend(
    ClassifierBackend
):
    """
    Feature-only backend used for generic pipeline tests.
    """

    def __init__(
        self,
    ) -> None:

        self.received: (
            ClassificationInput
            | None
        ) = None

    @property
    def name(
        self,
    ) -> str:

        return (
            "feature_only_test_backend"
        )

    @property
    def version(
        self,
    ) -> str:

        return (
            "1.0"
        )

    @property
    def requires_features(
        self,
    ) -> bool:

        return (
            True
        )

    @property
    def requires_audio(
        self,
    ) -> bool:

        return (
            False
        )

    def classify(
        self,
        classification_input: ClassificationInput,
    ) -> ClassificationResult:

        self.validate_input(
            classification_input
        )

        self.received = (
            classification_input
        )

        return make_classification()


class InvalidResultBackend(
    ClassifierBackend
):
    """
    Deliberately violates the backend return contract.
    """

    @property
    def name(
        self,
    ) -> str:

        return (
            "invalid_result_backend"
        )

    @property
    def version(
        self,
    ) -> str:

        return (
            "1.0"
        )

    @property
    def requires_features(
        self,
    ) -> bool:

        return (
            True
        )

    def classify(
        self,
        classification_input: ClassificationInput,
    ):
        self.validate_input(
            classification_input
        )

        return object()


# ======================================================================
# PIPELINE FACTORY
# ======================================================================


def make_pipeline(
    monkeypatch,
    tmp_path: Path,
    *,
    classifier_backend: ClassifierBackend | None = None,
):
    """
    Construct a real EventPipeline while replacing heavyweight
    collaborators with deterministic test doubles.

    Configuration construction, DSP configuration construction and
    classifier-backend wiring still execute through the real
    EventPipeline constructor.
    """

    detector = (
        FakeDetector()
    )

    localizer = (
        FakeLocalizer()
    )

    database = (
        FakeDatabase()
    )

    monkeypatch.setattr(
        event_pipeline_module,
        "MultiNodeEventDetector",
        lambda *args, **kwargs:
            detector,
    )

    monkeypatch.setattr(
        event_pipeline_module,
        "LocalizationEngine",
        lambda *args, **kwargs:
            localizer,
    )

    monkeypatch.setattr(
        event_pipeline_module,
        "EventDatabase",
        lambda *args, **kwargs:
            database,
    )

    config = make_test_config(
        tmp_path
    )

    streams = StreamManager(
        config.audio
    )

    pipeline = EventPipeline(
        streams,
        config,
        classifier_backend=
            classifier_backend,
    )

    return (
        pipeline,
        detector,
        localizer,
        database,
    )


# ======================================================================
# CONSTRUCTION
# ======================================================================


def test_pipeline_constructs_expected_subsystems(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        detector,
        localizer,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    assert (
        pipeline.detector
        is detector
    )

    assert (
        pipeline.localizer
        is localizer
    )

    assert (
        pipeline.database
        is database
    )

    assert (
        pipeline.active_session_id
        is None
    )

    assert (
        pipeline.active_session_label
        is None
    )

    assert (
        pipeline.completed_events
        == 0
    )


def test_pipeline_uses_injected_classifier_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:

    backend = (
        FeatureOnlyBackend()
    )

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
        classifier_backend=
            backend,
    )

    assert (
        pipeline.classifier_backend
        is backend
    )


# ======================================================================
# SESSION START
# ======================================================================


def test_start_session_sets_runtime_state(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        detector,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.completed_events = (
        9
    )

    pipeline.last_event_db_id = (
        88
    )

    pipeline.last_best_node_id = (
        3
    )

    pipeline.last_features = (
        make_features()
    )

    pipeline.last_classification = (
        make_classification()
    )

    pipeline.start_session(
        SESSION_ID,
        "test_session",
    )

    assert (
        pipeline.active_session_id
        == SESSION_ID
    )

    assert (
        pipeline.active_session_label
        == "test_session"
    )

    assert (
        pipeline.completed_events
        == 0
    )

    assert (
        pipeline.last_event_db_id
        is None
    )

    assert (
        pipeline.last_best_node_id
        is None
    )

    assert (
        pipeline.last_features
        is None
    )

    assert (
        pipeline.last_classification
        is None
    )

    assert (
        database.start_calls
        == [
            (
                SESSION_ID,
                "test_session",
            )
        ]
    )

    assert (
        detector.reset_count
        == 1
    )


@pytest.mark.parametrize(
    "session_id",
    [
        0,
        -1,
    ],
)
def test_start_session_rejects_invalid_session_id(
    monkeypatch,
    tmp_path: Path,
    session_id: int,
) -> None:

    (
        pipeline,
        _,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    with pytest.raises(
        ValueError
    ):

        pipeline.start_session(
            session_id,
            "test",
        )

    assert (
        pipeline.active_session_id
        is None
    )

    assert (
        database.start_calls
        == []
    )


def test_start_session_rejects_blank_label(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    with pytest.raises(
        ValueError
    ):

        pipeline.start_session(
            SESSION_ID,
            "   ",
        )

    assert (
        pipeline.active_session_id
        is None
    )

    assert (
        database.start_calls
        == []
    )


def test_start_session_rejects_second_active_session(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.start_session(
        SESSION_ID,
        "first",
    )

    with pytest.raises(
        RuntimeError
    ):

        pipeline.start_session(
            SECOND_SESSION_ID,
            "second",
        )

    assert (
        pipeline.active_session_id
        == SESSION_ID
    )

    assert (
        len(
            database.start_calls
        )
        == 1
    )


def test_database_failure_during_start_does_not_activate_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    Regression test for transactional session startup.

    Persistent session creation must succeed before the in-memory
    pipeline advertises itself as active.
    """

    (
        pipeline,
        _,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    database.start_error = RuntimeError(
        "synthetic database failure"
    )

    with pytest.raises(
        RuntimeError
    ):

        pipeline.start_session(
            SESSION_ID,
            "test",
        )

    assert (
        pipeline.active_session_id
        is None
    )

    assert (
        pipeline.active_session_label
        is None
    )


# ======================================================================
# SESSION STOP
# ======================================================================


def test_stop_session_clears_runtime_state(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        detector,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    reset_before_stop = (
        detector.reset_count
    )

    pipeline.stop_session()

    assert (
        database.stop_calls
        == [
            SESSION_ID
        ]
    )

    assert (
        pipeline.active_session_id
        is None
    )

    assert (
        pipeline.active_session_label
        is None
    )

    assert (
        detector.reset_count
        == reset_before_stop
        + 1
    )


def test_stop_session_clears_state_even_if_database_stop_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    Regression test for try/finally session cleanup.
    """

    (
        pipeline,
        detector,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    reset_before_stop = (
        detector.reset_count
    )

    database.stop_error = RuntimeError(
        "synthetic stop failure"
    )

    with pytest.raises(
        RuntimeError
    ):

        pipeline.stop_session()

    assert (
        pipeline.active_session_id
        is None
    )

    assert (
        pipeline.active_session_label
        is None
    )

    assert (
        detector.reset_count
        == reset_before_stop
        + 1
    )


# ======================================================================
# AUDIO SESSION GATING
# ======================================================================


def test_on_audio_ignores_audio_without_active_session(
    monkeypatch,
    tmp_path: Path,
    make_audio_block,
) -> None:

    (
        pipeline,
        detector,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    block = make_audio_block(
        session_id=
            SESSION_ID
    )

    result = pipeline.on_audio(
        block
    )

    assert (
        result
        == []
    )

    assert (
        detector.process_calls
        == []
    )


def test_on_audio_ignores_stale_session_block(
    monkeypatch,
    tmp_path: Path,
    make_audio_block,
) -> None:

    (
        pipeline,
        detector,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    stale = make_audio_block(
        session_id=
            SECOND_SESSION_ID
    )

    result = pipeline.on_audio(
        stale
    )

    assert (
        result
        == []
    )

    assert (
        detector.process_calls
        == []
    )


def test_on_audio_forwards_matching_session_to_detector(
    monkeypatch,
    tmp_path: Path,
    make_audio_block,
) -> None:

    (
        pipeline,
        detector,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    block = make_audio_block(
        session_id=
            SESSION_ID
    )

    result = pipeline.on_audio(
        block
    )

    assert (
        result
        == []
    )

    assert (
        detector.process_calls
        == [
            block
        ]
    )


# ======================================================================
# DETECTOR -> EVENT PROCESSING
# ======================================================================


def test_on_audio_persists_completed_detector_events(
    monkeypatch,
    tmp_path: Path,
    make_audio_block,
) -> None:

    (
        pipeline,
        detector,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    event = make_event()

    detector.events_to_return = [
        event
    ]

    persisted = []

    monkeypatch.setattr(
        pipeline,
        "_persist_event",
        lambda received_event:
            persisted.append(
                received_event
            ),
    )

    block = make_audio_block(
        session_id=
            SESSION_ID
    )

    result = pipeline.on_audio(
        block
    )

    assert (
        result
        == [
            event
        ]
    )

    assert (
        persisted
        == [
            event
        ]
    )


def test_event_processing_failure_does_not_destroy_detector_result(
    monkeypatch,
    tmp_path: Path,
    make_audio_block,
) -> None:
    """
    One downstream processing failure should be logged/isolated rather
    than making the detector lose the fact that an event completed.
    """

    (
        pipeline,
        detector,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    event = make_event()

    detector.events_to_return = [
        event
    ]

    def fail_persistence(
        _event,
    ) -> None:

        raise RuntimeError(
            "synthetic processing failure"
        )

    monkeypatch.setattr(
        pipeline,
        "_persist_event",
        fail_persistence,
    )

    result = pipeline.on_audio(
        make_audio_block(
            session_id=
                SESSION_ID
        )
    )

    assert (
        result
        == [
            event
        ]
    )


# ======================================================================
# LOCALIZATION WINDOW
# ======================================================================


class WindowStream:
    """
    Stream test double exposing deterministic sample-indexed audio.
    """

    def __init__(
        self,
        samples_by_node,
    ) -> None:

        self.samples_by_node = (
            samples_by_node
        )

        self.window_calls = []

    def get_window(
        self,
        node_id: int,
        start_sample: int,
        length: int,
        *,
        fill_value: int = 0,
    ):

        self.window_calls.append(
            (
                node_id,
                start_sample,
                length,
            )
        )

        samples = self.samples_by_node[
            node_id
        ]

        return np.asarray(
            samples[
                :length
            ],
            dtype=np.int16,
        )


def test_localization_start_returns_whole_short_event() -> None:

    pipeline = object.__new__(
        EventPipeline
    )

    pipeline.config = SimpleNamespace(
        localization=
            SimpleNamespace(
                window_samples=
                    128,

                reference_node=
                    1,
            )
    )

    pipeline.streams = WindowStream(
        {
            1:
                np.zeros(
                    100,
                    dtype=np.int16,
                )
        }
    )

    event = make_event(
        start_sample=
            500,

        end_sample=
            600,
    )

    (
        start,
        length,
    ) = pipeline._localization_start(
        event
    )

    assert (
        start
        == 500
    )

    assert (
        length
        == 100
    )


def test_localization_start_selects_highest_energy_region() -> None:
    """
    Event has 256 samples.

    The only high-energy 64-sample region begins at relative sample 120,
    therefore the localization window should begin at:

        event.start_sample + 120
    """

    pipeline = object.__new__(
        EventPipeline
    )

    pipeline.config = SimpleNamespace(
        localization=
            SimpleNamespace(
                window_samples=
                    64,

                reference_node=
                    1,
            )
    )

    samples = np.zeros(
        256,
        dtype=np.int16,
    )

    samples[
        120:184
    ] = (
        4000
    )

    pipeline.streams = WindowStream(
        {
            1:
                samples
        }
    )

    event = make_event(
        start_sample=
            1000,

        end_sample=
            1256,
    )

    (
        start,
        length,
    ) = pipeline._localization_start(
        event
    )

    assert (
        start
        == 1120
    )

    assert (
        length
        == 64
    )


# ======================================================================
# FRAME RMS
# ======================================================================


def test_frame_rms_values(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    frame_length = int(
        pipeline.config.dsp.snr_frame_length
    )

    signal = np.concatenate(
        (
            np.full(
                frame_length,
                3.0,
                dtype=np.float32,
            ),

            np.full(
                frame_length,
                4.0,
                dtype=np.float32,
            ),
        )
    )

    result = pipeline._frame_rms_values(
        signal
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [
                3.0,
                4.0,
            ],
            dtype=np.float64,
        ),
        atol=
            1e-12,
    )


def test_frame_rms_empty_signal_returns_empty_array(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    result = pipeline._frame_rms_values(
        np.array(
            [],
            dtype=np.float32,
        )
    )

    assert (
        result.size
        == 0
    )


# ======================================================================
# CHANNEL QUALITY
# ======================================================================


def test_channel_quality_score_prefers_stronger_signal_at_same_noise(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    frame_length = int(
        pipeline.config.dsp.snr_frame_length
    )

    quiet = np.full(
        frame_length
        * 2,
        100.0,
        dtype=np.float32,
    )

    strong = np.full(
        frame_length
        * 2,
        1000.0,
        dtype=np.float32,
    )

    noise_rms = (
        10.0
    )

    quiet_score = pipeline._channel_quality_score(
        quiet,
        noise_rms,
    )

    strong_score = pipeline._channel_quality_score(
        strong,
        noise_rms,
    )

    assert (
        strong_score
        > quiet_score
    )


def test_channel_quality_zero_signal_is_negative_infinity(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    signal = np.zeros(
        2048,
        dtype=np.float32,
    )

    result = pipeline._channel_quality_score(
        signal,
        10.0,
    )

    assert (
        result
        == float(
            "-inf"
        )
    )


# ======================================================================
# BEST CHANNEL / FEATURE FAILURE REGRESSION
# ======================================================================


def test_feature_failure_preserves_selected_model_audio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    Critical regression test.

    A waveform-only classifier may still be usable when handcrafted DSP
    feature extraction fails.

    Therefore feature failure must NOT throw away best_audio.model_signal.
    """

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    event_length = max(
        4096,
        int(
            pipeline.config.dsp.minimum_event_samples
        ),
    )

    node_signals = {
        1:
            np.full(
                event_length,
                100,
                dtype=np.int16,
            ),

        2:
            np.full(
                event_length,
                1000,
                dtype=np.int16,
            ),

        3:
            np.full(
                event_length,
                500,
                dtype=np.int16,
            ),
    }

    pipeline.streams = WindowStream(
        node_signals
    )

    # --------------------------------------------------------------
    # CONTROL NOISE FLOOR SO SIGNAL MAGNITUDE DETERMINES QUALITY.
    # --------------------------------------------------------------

    monkeypatch.setattr(
        pipeline,
        "_estimate_noise_rms",
        lambda _signal:
            10.0,
    )

    # --------------------------------------------------------------
    # LIGHTWEIGHT PREPROCESSING DOUBLE
    # --------------------------------------------------------------

    def fake_preprocess(
        pcm_samples,
        _config,
    ):

        amplitude = np.asarray(
            pcm_samples,
            dtype=np.float32,
        )

        peak = float(
            np.max(
                np.abs(
                    amplitude
                )
            )
        )

        if (
            peak
            > 0.0
        ):

            model = (
                amplitude
                / peak
            ).astype(
                np.float32
            )

        else:

            model = np.zeros_like(
                amplitude,
                dtype=np.float32,
            )

        return SimpleNamespace(
            amplitude_signal=
                amplitude,

            model_signal=
                model,
        )

    monkeypatch.setattr(
        event_pipeline_module,
        "preprocess_event_audio",
        fake_preprocess,
    )

    # --------------------------------------------------------------
    # DELIBERATELY FAIL HANDCRAFTED FEATURES.
    # --------------------------------------------------------------

    def fail_features(
        *args,
        **kwargs,
    ):

        raise RuntimeError(
            "synthetic feature extraction failure"
        )

    monkeypatch.setattr(
        event_pipeline_module,
        "extract_acoustic_features",
        fail_features,
    )

    event = make_event(
        start_sample=
            0,

        end_sample=
            event_length,
    )

    (
        best_node,
        features,
        model_audio,
    ) = pipeline._extract_best_channel_data(
        event
    )

    assert (
        best_node
        == 2
    )

    assert (
        features
        is None
    )

    # --------------------------------------------------------------
    # MODEL AUDIO MUST SURVIVE FEATURE FAILURE.
    # --------------------------------------------------------------

    assert (
        model_audio
        is not None
    )

    assert (
        model_audio.ndim
        == 1
    )

    assert (
        model_audio.dtype
        == np.float32
    )

    assert np.all(
        np.isfinite(
            model_audio
        )
    )


# ======================================================================
# FEATURE-ONLY CLASSIFICATION
# ======================================================================


def test_classify_event_with_feature_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:

    backend = (
        FeatureOnlyBackend()
    )

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
        classifier_backend=
            backend,
    )

    event = make_event()

    features = (
        make_features()
    )

    result = pipeline._classify_event(
        event=
            event,

        best_node_id=
            2,

        features=
            features,

        model_audio=
            None,
    )

    assert (
        result
        == make_classification()
    )

    assert (
        backend.received
        is not None
    )

    assert (
        backend.received.features
        is features
    )

    assert (
        backend.received.source_node_id
        == 2
    )

    assert (
        backend.received.detector_event_id
        == event.event_id
    )

    assert (
        backend.received.session_id
        == SESSION_ID
    )


# ======================================================================
# WAVEFORM-ONLY CLASSIFICATION
# ======================================================================


def test_classify_event_supports_waveform_only_backend_without_features(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    Critical modular-classification regression test.

    Generic EventPipeline classification must not require handcrafted
    features when the selected backend declares:

        requires_features = False
        requires_audio = True
    """

    backend = (
        AudioOnlyBackend()
    )

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
        classifier_backend=
            backend,
    )

    waveform = np.linspace(
        -0.5,
        0.5,
        4096,
        dtype=np.float32,
    )

    result = pipeline._classify_event(
        event=
            make_event(),

        best_node_id=
            1,

        features=
            None,

        model_audio=
            waveform,
    )

    assert isinstance(
        result,
        ClassificationResult,
    )

    assert (
        backend.received
        is not None
    )

    assert (
        backend.received.features
        is None
    )

    assert (
        backend.received.model_audio
        is waveform
    )


def test_audio_backend_returns_none_when_required_audio_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:

    backend = (
        AudioOnlyBackend()
    )

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
        classifier_backend=
            backend,
    )

    result = pipeline._classify_event(
        event=
            make_event(),

        best_node_id=
            1,

        features=
            None,

        model_audio=
            None,
    )

    assert (
        result
        is None
    )


# ======================================================================
# CLASSIFIER RETURN CONTRACT
# ======================================================================


def test_classify_event_rejects_non_classification_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    Backends are required to return ClassificationResult.

    A buggy future ML adapter must not leak arbitrary model objects into
    database/dashboard code.
    """

    backend = (
        InvalidResultBackend()
    )

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
        classifier_backend=
            backend,
    )

    result = pipeline._classify_event(
        event=
            make_event(),

        best_node_id=
            1,

        features=
            make_features(),

        model_audio=
            None,
    )

    assert (
        result
        is None
    )


# ======================================================================
# COMPLETE PERSISTENCE ORCHESTRATION
# ======================================================================


class EnvironmentStream:
    """
    Stream façade for _persist_event().
    """

    def __init__(
        self,
        environment,
    ) -> None:

        self.environment = (
            environment
        )

        self.environment_queries = []

    def get_environment_near(
        self,
        sample_index: int,
    ):

        self.environment_queries.append(
            sample_index
        )

        return (
            self.environment
        )


def test_persist_event_orchestrates_all_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        localizer,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
        classifier_backend=
            FeatureOnlyBackend(),
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    environment = EnvironmentPayload(
        25.0,
        60.0,
        1008.0,
    )

    pipeline.streams = EnvironmentStream(
        environment
    )

    localization = SimpleNamespace(
        position=
            SimpleNamespace(
                x=
                    0.45,

                y=
                    0.35,

                success=
                    True,

                residual_rms_meters=
                    0.02,
            ),

        speed_of_sound_mps=
            344.0,
    )

    localizer.result = (
        localization
    )

    features = (
        make_features()
    )

    classification = (
        make_classification()
    )

    model_audio = np.zeros(
        4096,
        dtype=np.float32,
    )

    monkeypatch.setattr(
        pipeline,
        "_localization_start",
        lambda _event:
            (
                1200,
                max(
                    int(
                        pipeline.config.dsp.minimum_event_samples
                    ),
                    128,
                ),
            ),
    )

    monkeypatch.setattr(
        pipeline,
        "_extract_best_channel_data",
        lambda _event:
            (
                2,
                features,
                model_audio,
            ),
    )

    monkeypatch.setattr(
        pipeline,
        "_classify_event",
        lambda **kwargs:
            classification,
    )

    event = make_event(
        event_id=
            7,

        start_sample=
            1000,

        end_sample=
            5000,
    )

    pipeline._persist_event(
        event
    )

    # ==============================================================
    # ENVIRONMENT MIDPOINT
    # ==============================================================

    expected_midpoint = (
        event.start_sample
        + event.end_sample
    ) // 2

    assert (
        pipeline.streams.environment_queries
        == [
            expected_midpoint
        ]
    )

    # ==============================================================
    # LOCALIZATION
    # ==============================================================

    assert (
        len(
            localizer.calls
        )
        == 1
    )

    # ==============================================================
    # CORE EVENT
    # ==============================================================

    assert (
        len(
            database.event_calls
        )
        == 1
    )

    stored_event = (
        database.event_calls[
            0
        ][
            0
        ]
    )

    stored_kwargs = (
        database.event_calls[
            0
        ][
            1
        ]
    )

    assert (
        stored_event
        is event
    )

    assert (
        stored_kwargs[
            "environment"
        ]
        == environment
    )

    assert (
        stored_kwargs[
            "localization"
        ]
        is localization
    )

    assert (
        stored_kwargs[
            "best_node_id"
        ]
        == 2
    )

    assert (
        stored_kwargs[
            "event_directory"
        ]
        is None
    )

    # ==============================================================
    # FEATURES
    # ==============================================================

    assert (
        len(
            database.feature_calls
        )
        == 1
    )

    assert (
        database.feature_calls[
            0
        ][
            "event_id"
        ]
        == database.next_event_id
    )

    assert (
        database.feature_calls[
            0
        ][
            "source_node_id"
        ]
        == 2
    )

    assert (
        database.feature_calls[
            0
        ][
            "features"
        ]
        is features
    )

    # ==============================================================
    # CLASSIFICATION
    # ==============================================================

    assert (
        len(
            database.classification_calls
        )
        == 1
    )

    assert (
        database.classification_calls[
            0
        ][
            "result"
        ]
        is classification
    )

    # ==============================================================
    # RUNTIME STATUS
    # ==============================================================

    assert (
        pipeline.completed_events
        == 1
    )

    assert (
        pipeline.last_event_db_id
        == database.next_event_id
    )

    assert (
        pipeline.last_best_node_id
        == 2
    )

    assert (
        pipeline.last_features
        is features
    )

    assert (
        pipeline.last_classification
        is classification
    )


# ======================================================================
# STALE EVENT PROTECTION
# ======================================================================


def test_persist_event_ignores_stale_session_event(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    stale_event = make_event(
        session_id=
            SECOND_SESSION_ID
    )

    pipeline._persist_event(
        stale_event
    )

    assert (
        database.event_calls
        == []
    )

    assert (
        pipeline.completed_events
        == 0
    )


# ======================================================================
# OPTIONAL FEATURE / CLASSIFICATION PERSISTENCE FAILURE
# ======================================================================


def test_feature_persistence_failure_does_not_lose_core_event(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        localizer,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
        classifier_backend=
            FeatureOnlyBackend(),
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    pipeline.streams = EnvironmentStream(
        None
    )

    localizer.result = (
        None
    )

    features = (
        make_features()
    )

    classification = (
        make_classification()
    )

    monkeypatch.setattr(
        pipeline,
        "_localization_start",
        lambda _event:
            (
                0,
                0,
            ),
    )

    monkeypatch.setattr(
        pipeline,
        "_extract_best_channel_data",
        lambda _event:
            (
                1,
                features,
                np.zeros(
                    1024,
                    dtype=np.float32,
                ),
            ),
    )

    monkeypatch.setattr(
        pipeline,
        "_classify_event",
        lambda **kwargs:
            classification,
    )

    database.feature_error = RuntimeError(
        "synthetic feature DB failure"
    )

    pipeline._persist_event(
        make_event()
    )

    # Core event survives.
    assert (
        len(
            database.event_calls
        )
        == 1
    )

    # Classification should still be attempted.
    assert (
        len(
            database.classification_calls
        )
        == 1
    )

    assert (
        pipeline.completed_events
        == 1
    )


def test_classification_persistence_failure_does_not_lose_core_event(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        localizer,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
        classifier_backend=
            FeatureOnlyBackend(),
    )

    pipeline.start_session(
        SESSION_ID,
        "test",
    )

    pipeline.streams = EnvironmentStream(
        None
    )

    localizer.result = (
        None
    )

    features = (
        make_features()
    )

    classification = (
        make_classification()
    )

    monkeypatch.setattr(
        pipeline,
        "_localization_start",
        lambda _event:
            (
                0,
                0,
            ),
    )

    monkeypatch.setattr(
        pipeline,
        "_extract_best_channel_data",
        lambda _event:
            (
                1,
                features,
                None,
            ),
    )

    monkeypatch.setattr(
        pipeline,
        "_classify_event",
        lambda **kwargs:
            classification,
    )

    database.classification_error = RuntimeError(
        "synthetic classification DB failure"
    )

    pipeline._persist_event(
        make_event()
    )

    assert (
        len(
            database.event_calls
        )
        == 1
    )

    assert (
        len(
            database.feature_calls
        )
        == 1
    )

    assert (
        pipeline.completed_events
        == 1
    )


# ======================================================================
# ENVIRONMENT DELEGATION
# ======================================================================


def test_add_environment_delegates_to_database(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        database,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    environment = EnvironmentPayload(
        27.5,
        58.0,
        1007.2,
    )

    pipeline.add_environment(
        node_id=
            1,

        session_id=
            SESSION_ID,

        sample_index=
            48_000,

        environment=
            environment,
    )

    assert (
        database.environment_calls
        == [
            {
                "session_id":
                    SESSION_ID,

                "node_id":
                    1,

                "sample_index":
                    48_000,

                "environment":
                    environment,
            }
        ]
    )


# ======================================================================
# WAV WRITER
# ======================================================================


def test_write_wav_creates_valid_pcm16_file(
    monkeypatch,
    tmp_path: Path,
) -> None:

    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline(
        monkeypatch,
        tmp_path,
    )

    path = (
        tmp_path
        / "event.wav"
    )

    samples = np.array(
        [
            -32768,
            -1000,
            0,
            1000,
            32767,
        ],
        dtype=np.int16,
    )

    pipeline._write_wav(
        path,
        samples,
    )

    assert (
        path.exists()
    )

    with wave.open(
        str(
            path
        ),
        "rb",
    ) as wav:

        assert (
            wav.getnchannels()
            == pipeline.config.audio.channels
        )

        assert (
            wav.getsampwidth()
            == pipeline.config.audio.sample_width_bytes
        )

        assert (
            wav.getframerate()
            == pipeline.config.audio.sample_rate
        )

        assert (
            wav.getnframes()
            == samples.size
        )

        raw = wav.readframes(
            wav.getnframes()
        )

    recovered = np.frombuffer(
        raw,
        dtype="<i2",
    )

    np.testing.assert_array_equal(
        recovered,
        samples,
    )