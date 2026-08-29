from __future__ import annotations

import logging
import math
from pathlib import Path
import wave

import numpy as np

from classification import (
    ClassificationInput,
    ClassificationResult,
    ClassifierBackend,
)

from classification.factory import (
    create_classifier_backend,
)

from config import AppConfig
from database import EventDatabase

from dsp.features import (
    AcousticFeatures,
    FeatureConfig,
    extract_acoustic_features,
)

from dsp.preprocessing import (
    PreprocessedAudio,
    PreprocessingConfig,
    preprocess_event_audio,
)

from event_detector import (
    AcousticEvent,
    MultiNodeEventDetector,
)

from localization import (
    LocalizationEngine,
    LocalizationResult,
)

from models import AudioBlock
from stream_manager import StreamManager


logger = logging.getLogger(__name__)


# ======================================================================
# EVENT PIPELINE
# ======================================================================


class EventPipeline:
    """
    Complete automatic acoustic-event processing pipeline.

    Flow
    ----
    AudioBlock
        ↓
    Multi-node event detection
        ↓
    Synchronized event extraction
        ↓
    Environmental association
        ↓
    3-node TDOA localization
        ↓
    Per-node preprocessing
        ↓
    Best-SNR microphone selection
        ↓
    DSP feature extraction
        ↓
    ClassificationInput
        ↓
    Configured ClassifierBackend
        ↓
    ClassificationResult
        ↓
    WAV storage
        ↓
    SQLite persistence

    Classification architecture
    ---------------------------
    The pipeline depends only on ClassifierBackend.

    Default backend selection is delegated to:

        classification.factory.create_classifier_backend()

    Therefore EventPipeline does not need to know whether the configured
    classifier is:

        - heuristic
        - pretrained
        - BirdNET
        - ensemble
        - another future backend

    Classification can also be completely disabled through AppConfig.
    """

    def __init__(
        self,
        streams: StreamManager,
        config: AppConfig,
        classifier_backend: ClassifierBackend | None = None,
    ) -> None:

        self.streams = streams
        self.config = config

        # ==============================================================
        # EVENT DETECTOR
        # ==============================================================

        self.detector = MultiNodeEventDetector(
            config.audio,
            config.detection,
            tuple(
                sorted(
                    config.expected_nodes
                )
            ),
        )

        # ==============================================================
        # LOCALIZATION
        # ==============================================================

        self.localizer = LocalizationEngine(
            streams,
            config.localization,
        )

        # ==============================================================
        # DATABASE
        # ==============================================================

        self.database = EventDatabase(
            config.persistence.database_path
        )

        # ==============================================================
        # DSP PREPROCESSING CONFIGURATION
        # ==============================================================

        self.preprocessing_config = (
            PreprocessingConfig(
                sample_rate=
                    config.audio.sample_rate,

                remove_dc=
                    config.dsp.remove_dc,

                bandpass_enabled=
                    config.dsp.bandpass_enabled,

                low_cutoff_hz=
                    config.dsp.low_cutoff_hz,

                high_cutoff_hz=
                    config.dsp.high_cutoff_hz,

                filter_order=
                    config.dsp.filter_order,

                normalize_for_model=
                    config.dsp.normalize_for_model,

                model_target_peak=
                    config.dsp.model_target_peak,
            )
        )

        # ==============================================================
        # DSP FEATURE CONFIGURATION
        # ==============================================================

        self.feature_config = FeatureConfig(
            sample_rate=
                config.audio.sample_rate,

            n_fft=
                config.dsp.n_fft,

            hop_length=
                config.dsp.hop_length,

            n_mfcc=
                config.dsp.n_mfcc,

            n_mels=
                config.dsp.n_mels,

            roll_percent=
                config.dsp.roll_percent,

            mfcc_fmin_hz=
                config.dsp.mfcc_fmin_hz,

            mfcc_fmax_hz=
                config.dsp.mfcc_fmax_hz,
        )

        # ==============================================================
        # CLASSIFICATION BACKEND
        # ==============================================================
        #
        # Explicit dependency injection takes precedence over factory
        # creation, but the master `classification.enabled` switch still
        # controls whether classification is actually executed.
        # ==============================================================

        if classifier_backend is not None:

            self.classifier_backend: (
                ClassifierBackend
                | None
            ) = classifier_backend

        else:

            self.classifier_backend = (
                create_classifier_backend(
                    config.classification
                )
            )

        # ==============================================================
        # SESSION STATE
        # ==============================================================

        self.active_session_id: (
            int
            | None
        ) = None

        self.active_session_label: (
            str
            | None
        ) = None

        self.completed_events = 0

        self.last_event_db_id: (
            int
            | None
        ) = None

        self.last_best_node_id: (
            int
            | None
        ) = None

        self.last_features: (
            AcousticFeatures
            | None
        ) = None

        self.last_classification: (
            ClassificationResult
            | None
        ) = None

    # ==================================================================
    # CLASSIFICATION STATE
    # ==================================================================

    @property
    def classification_enabled(
        self,
    ) -> bool:
        """
        Whether classification is currently operational.

        Both conditions must be satisfied:

            1. classification is enabled in AppConfig
            2. a classifier backend exists
        """

        return (
            self.config.classification.enabled
            and self.classifier_backend
            is not None
        )

    # ==================================================================
    # SESSION CONTROL
    # ==================================================================

    def start_session(
        self,
        session_id: int,
        label: str,
    ) -> None:
        """
        Start a new event-processing session.
        """

        self.active_session_id = int(
            session_id
        )

        self.active_session_label = str(
            label
        )

        self.completed_events = 0

        self.last_event_db_id = None

        self.last_best_node_id = None

        self.last_features = None

        self.last_classification = None

        self.detector.reset()

        self.database.start_session(
            session_id,
            label,
        )

        # --------------------------------------------------------------
        # SESSION LOG
        # --------------------------------------------------------------

        if (
            self.classification_enabled
            and self.classifier_backend
            is not None
        ):

            logger.info(
                (
                    "Event pipeline session started "
                    "| session=0x%08X "
                    "| classifier=%s "
                    "| backend_version=%s"
                ),
                session_id,
                self.classifier_backend.name,
                self.classifier_backend.version,
            )

        else:

            logger.info(
                (
                    "Event pipeline session started "
                    "| session=0x%08X "
                    "| classification=disabled"
                ),
                session_id,
            )

    def stop_session(
        self,
    ) -> None:
        """
        Stop the current event-processing session.
        """

        self.database.stop_session(
            self.active_session_id
        )

        self.active_session_id = None

        self.active_session_label = None

        self.detector.reset()

    # ==================================================================
    # AUDIO ENTRY POINT
    # ==================================================================

    def on_audio(
        self,
        block: AudioBlock,
    ) -> list[AcousticEvent]:
        """
        Feed one audio block into the multi-node event detector.

        Completed events are automatically passed through the complete
        event-processing pipeline.
        """

        events = self.detector.process(
            block
        )

        for event in events:

            try:

                self._persist_event(
                    event
                )

            except Exception:

                logger.exception(
                    (
                        "Failed to process "
                        "event %s"
                    ),
                    event.event_id,
                )

        return events

    # ==================================================================
    # LOCALIZATION WINDOW SELECTION
    # ==================================================================

    def _localization_start(
        self,
        event: AcousticEvent,
    ) -> tuple[int, int]:
        """
        Select a high-energy portion of the event for TDOA localization.

        The complete event can contain substantial pre/post padding.

        Localization is generally more stable when focused on the
        strongest active acoustic region.
        """

        window_samples = int(
            self.config.localization.window_samples
        )

        event_length = int(
            event.sample_count
        )

        if event_length <= 0:

            return (
                int(
                    event.start_sample
                ),
                0,
            )

        # --------------------------------------------------------------
        # EVENT SHORTER THAN LOCALIZATION WINDOW
        # --------------------------------------------------------------

        if (
            event_length
            <= window_samples
        ):

            return (
                max(
                    0,
                    int(
                        event.start_sample
                    ),
                ),
                event_length,
            )

        reference_node = (
            self.config.localization.reference_node
        )

        samples = (
            self.streams.get_window(
                reference_node,
                event.start_sample,
                event_length,
            )
        )

        x = np.asarray(
            samples,
            dtype=np.float64,
        )

        if (
            x.size
            <= window_samples
        ):

            return (
                int(
                    event.start_sample
                ),
                int(
                    x.size
                ),
            )

        # --------------------------------------------------------------
        # ROLLING ENERGY SEARCH
        # --------------------------------------------------------------

        energy = (
            x
            * x
        )

        kernel = np.ones(
            window_samples,
            dtype=np.float64,
        )

        rolling_energy = np.convolve(
            energy,
            kernel,
            mode="valid",
        )

        relative_start = (
            int(
                np.argmax(
                    rolling_energy
                )
            )
            if rolling_energy.size
            else 0
        )

        return (
            int(
                event.start_sample
                + relative_start
            ),
            window_samples,
        )

    # ==================================================================
    # SHORT-TIME RMS
    # ==================================================================

    def _frame_rms_values(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate short-time RMS values.

        Frame size comes from DSPConfig.
        """

        frame_length = int(
            self.config.dsp.snr_frame_length
        )

        x = np.asarray(
            signal,
            dtype=np.float64,
        )

        if x.size == 0:

            return np.array(
                [],
                dtype=np.float64,
            )

        # --------------------------------------------------------------
        # VERY SHORT SIGNAL
        # --------------------------------------------------------------

        if (
            x.size
            < frame_length
        ):

            value = float(
                np.sqrt(
                    np.mean(
                        x
                        * x
                    )
                )
            )

            return np.array(
                [
                    value
                ],
                dtype=np.float64,
            )

        # --------------------------------------------------------------
        # COMPLETE NON-OVERLAPPING FRAMES
        # --------------------------------------------------------------

        frame_count = (
            x.size
            // frame_length
        )

        usable_samples = (
            frame_count
            * frame_length
        )

        frames = (
            x[
                :usable_samples
            ]
            .reshape(
                frame_count,
                frame_length,
            )
        )

        rms_values = np.sqrt(
            np.mean(
                frames
                * frames,
                axis=1,
            )
        )

        return np.asarray(
            rms_values,
            dtype=np.float64,
        )

    # ==================================================================
    # NOISE ESTIMATION
    # ==================================================================

    def _estimate_noise_rms(
        self,
        signal: np.ndarray,
    ) -> float | None:
        """
        Estimate background-noise RMS from the event pre-trigger region.
        """

        x = np.asarray(
            signal,
            dtype=np.float32,
        )

        if x.size == 0:

            return None

        # --------------------------------------------------------------
        # AVAILABLE PRE-TRIGGER REGION
        # --------------------------------------------------------------

        pre_pad_samples = int(
            round(
                self.config.detection.pre_pad_s
                * self.config.audio.sample_rate
            )
        )

        preferred_noise_samples = int(
            pre_pad_samples
            * self.config.dsp.noise_prepad_fraction
        )

        preferred_noise_samples = max(
            int(
                self.config.dsp.snr_frame_length
            ),
            preferred_noise_samples,
        )

        preferred_noise_samples = min(
            preferred_noise_samples,
            x.size,
        )

        noise_region = (
            x[
                :preferred_noise_samples
            ]
        )

        rms_values = (
            self._frame_rms_values(
                noise_region
            )
        )

        if rms_values.size == 0:

            return None

        finite_values = rms_values[
            np.isfinite(
                rms_values
            )
        ]

        if finite_values.size == 0:

            return None

        noise_rms = float(
            np.quantile(
                finite_values,
                self.config.dsp.noise_quantile,
            )
        )

        if not np.isfinite(
            noise_rms
        ):

            return None

        return max(
            0.0,
            noise_rms,
        )

    # ==================================================================
    # CHANNEL QUALITY
    # ==================================================================

    def _channel_quality_score(
        self,
        signal: np.ndarray,
        noise_rms: float | None,
    ) -> float:
        """
        Calculate a robust SNR-like microphone-selection score.

        This score chooses the strongest/cleanest channel for:

            - feature extraction
            - acoustic classification

        Localization continues to use all synchronized nodes.
        """

        rms_values = (
            self._frame_rms_values(
                signal
            )
        )

        if rms_values.size == 0:

            return float(
                "-inf"
            )

        finite_values = rms_values[
            np.isfinite(
                rms_values
            )
        ]

        if finite_values.size == 0:

            return float(
                "-inf"
            )

        # --------------------------------------------------------------
        # ACTIVE SIGNAL LEVEL
        # --------------------------------------------------------------

        signal_level = float(
            np.quantile(
                finite_values,
                self.config.dsp.signal_quantile,
            )
        )

        if (
            not math.isfinite(
                signal_level
            )
            or signal_level <= 0.0
        ):

            return float(
                "-inf"
            )

        # --------------------------------------------------------------
        # BACKGROUND-NOISE FLOOR
        # --------------------------------------------------------------

        digital_floor = float(
            self.config.dsp.digital_noise_floor
        )

        if noise_rms is None:

            effective_noise = (
                digital_floor
            )

        else:

            effective_noise = max(
                float(
                    noise_rms
                ),
                digital_floor,
            )

        score = (
            20.0
            * math.log10(
                signal_level
                / effective_noise
            )
        )

        if not math.isfinite(
            score
        ):

            return float(
                "-inf"
            )

        return float(
            score
        )

    # ==================================================================
    # MODEL AUDIO EXTRACTION
    # ==================================================================

    def _prepare_model_audio(
        self,
        processed: PreprocessedAudio,
        event: AcousticEvent,
    ) -> np.ndarray | None:
        """
        Prepare normalized waveform for classifier backends.

        The waveform is returned only when:

            - classification is enabled
            - model waveform delivery is enabled in config
            - the waveform is valid and finite

        The heuristic backend does not require this waveform, but future
        neural-network backends may.
        """

        if not self.classification_enabled:

            return None

        if not (
            self.config.classification
            .provide_model_audio
        ):

            return None

        model_audio = np.asarray(
            processed.model_signal,
            dtype=np.float32,
        )

        if (
            model_audio.ndim
            != 1
        ):

            logger.warning(
                (
                    "Event %s: model waveform "
                    "is not mono/1-D"
                ),
                event.event_id,
            )

            return None

        if (
            model_audio.size
            == 0
        ):

            return None

        if not np.all(
            np.isfinite(
                model_audio
            )
        ):

            logger.warning(
                (
                    "Event %s: model waveform "
                    "contains non-finite values"
                ),
                event.event_id,
            )

            return None

        # --------------------------------------------------------------
        # Return an independent contiguous float32 buffer.
        # --------------------------------------------------------------

        return np.ascontiguousarray(
            model_audio,
            dtype=np.float32,
        ).copy()

    # ==================================================================
    # BEST NODE + DSP EXTRACTION
    # ==================================================================

    def _extract_best_channel_data(
        self,
        event: AcousticEvent,
    ) -> tuple[
        int | None,
        AcousticFeatures | None,
        np.ndarray | None,
    ]:
        """
        Evaluate every available microphone and select the best channel.

        Returns
        -------
        best_node_id
            Node selected through the robust channel-quality estimate.

        features
            DSP feature vector extracted from the selected microphone.

        model_audio
            Optional normalized waveform for classifier backends that
            consume audio directly.

        Notes
        -----
        Best-node selection affects DSP/classification only.

        All synchronized microphones remain available for TDOA
        localization and event WAV storage.
        """

        best_node_id: (
            int
            | None
        ) = None

        best_score = float(
            "-inf"
        )

        best_audio: (
            PreprocessedAudio
            | None
        ) = None

        best_noise_rms: (
            float
            | None
        ) = None

        # ==============================================================
        # EVALUATE ALL NODES
        # ==============================================================

        for node_id in sorted(
            self.config.expected_nodes
        ):

            try:

                pcm_samples = (
                    self.streams.get_window(
                        node_id,
                        event.start_sample,
                        event.sample_count,
                    )
                )

            except Exception as exc:

                logger.warning(
                    (
                        "Event %s: unable to read "
                        "Node %s audio: %s"
                    ),
                    event.event_id,
                    node_id,
                    exc,
                )

                continue

            pcm_samples = np.asarray(
                pcm_samples,
                dtype=np.int16,
            )

            # ----------------------------------------------------------
            # REJECT EXTREMELY SHORT FRAGMENTS
            # ----------------------------------------------------------

            if (
                pcm_samples.size
                < self.config.dsp.minimum_event_samples
            ):

                continue

            # ----------------------------------------------------------
            # PREPROCESS
            # ----------------------------------------------------------

            try:

                processed = (
                    preprocess_event_audio(
                        pcm_samples,
                        self.preprocessing_config,
                    )
                )

            except Exception as exc:

                logger.warning(
                    (
                        "Event %s: preprocessing "
                        "failed for Node %s: %s"
                    ),
                    event.event_id,
                    node_id,
                    exc,
                )

                continue

            # ----------------------------------------------------------
            # BACKGROUND-NOISE ESTIMATION
            # ----------------------------------------------------------

            noise_rms = (
                self._estimate_noise_rms(
                    processed.amplitude_signal
                )
            )

            # ----------------------------------------------------------
            # MICROPHONE QUALITY SCORE
            # ----------------------------------------------------------

            score = (
                self._channel_quality_score(
                    processed.amplitude_signal,
                    noise_rms,
                )
            )

            logger.debug(
                (
                    "Event %s Node %s "
                    "quality score = %.2f dB"
                ),
                event.event_id,
                node_id,
                score,
            )

            # ----------------------------------------------------------
            # TRACK BEST MICROPHONE
            # ----------------------------------------------------------

            if (
                score
                > best_score
            ):

                best_score = (
                    score
                )

                best_node_id = int(
                    node_id
                )

                best_audio = (
                    processed
                )

                best_noise_rms = (
                    noise_rms
                )

        # ==============================================================
        # NO VALID MICROPHONE
        # ==============================================================

        if (
            best_node_id is None
            or best_audio is None
        ):

            return (
                None,
                None,
                None,
            )

        # ==============================================================
        # FEATURE EXTRACTION
        # ==============================================================

        try:

            features = (
                extract_acoustic_features(
                    best_audio,

                    noise_rms=
                        best_noise_rms,

                    config=
                        self.feature_config,
                )
            )

        except Exception as exc:

            logger.warning(
                (
                    "Event %s: feature extraction "
                    "failed on Node %s: %s"
                ),
                event.event_id,
                best_node_id,
                exc,
            )

            return (
                best_node_id,
                None,
                None,
            )

        # ==============================================================
        # OPTIONAL CLASSIFIER MODEL WAVEFORM
        # ==============================================================

        model_audio = (
            self._prepare_model_audio(
                best_audio,
                event,
            )
        )

        return (
            best_node_id,
            features,
            model_audio,
        )

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    def _classify_event(
        self,
        *,
        event: AcousticEvent,
        best_node_id: int | None,
        features: AcousticFeatures | None,
        model_audio: np.ndarray | None,
    ) -> ClassificationResult | None:
        """
        Build ClassificationInput and execute the configured backend.

        Returns None when classification is disabled or unavailable.
        """

        # --------------------------------------------------------------
        # MASTER SWITCH / BACKEND AVAILABILITY
        # --------------------------------------------------------------

        if not self.classification_enabled:

            return None

        backend = (
            self.classifier_backend
        )

        if backend is None:

            return None

        # --------------------------------------------------------------
        # CURRENT STANDARD INPUT REQUIRES DSP FEATURES
        # --------------------------------------------------------------

        if features is None:

            logger.warning(
                (
                    "Event %s classification skipped: "
                    "DSP features unavailable"
                ),
                event.event_id,
            )

            return None

        # --------------------------------------------------------------
        # STANDARDIZED CLASSIFIER INPUT
        # --------------------------------------------------------------

        try:

            classification_input = (
                ClassificationInput(
                    features=
                        features,

                    sample_rate=
                        self.config.audio.sample_rate,

                    model_audio=
                        model_audio,

                    source_node_id=
                        best_node_id,

                    detector_event_id=
                        event.event_id,

                    session_id=
                        event.session_id,
                )
            )

        except Exception as exc:

            logger.warning(
                (
                    "Event %s classification input "
                    "creation failed: %s"
                ),
                event.event_id,
                exc,
            )

            return None

        # --------------------------------------------------------------
        # BACKEND EXECUTION
        # --------------------------------------------------------------

        try:

            backend.validate_input(
                classification_input
            )

            return (
                backend.classify(
                    classification_input
                )
            )

        except Exception as exc:

            logger.warning(
                (
                    "Event %s classification failed "
                    "using backend '%s': %s"
                ),
                event.event_id,
                backend.name,
                exc,
            )

            return None

    # ==================================================================
    # EVENT PROCESSING
    # ==================================================================

    def _persist_event(
        self,
        event: AcousticEvent,
    ) -> None:
        """
        Execute the complete processing chain for one completed event.
        """

        # ==============================================================
        # SESSION VALIDATION
        # ==============================================================

        if (
            self.active_session_id
            is not None
            and event.session_id
            != self.active_session_id
        ):

            logger.warning(
                (
                    "Ignoring event from stale session "
                    "0x%08X "
                    "(active=0x%08X)"
                ),
                event.session_id,
                self.active_session_id,
            )

            return

        # ==============================================================
        # EVENT MIDPOINT
        # ==============================================================

        midpoint = (
            event.start_sample
            + event.end_sample
        ) // 2

        # ==============================================================
        # ENVIRONMENT
        # ==============================================================

        environment = (
            self.streams.get_environment_near(
                midpoint
            )
        )

        # ==============================================================
        # LOCALIZATION
        # ==============================================================

        localization: (
            LocalizationResult
            | None
        ) = None

        try:

            (
                localization_start,
                localization_length,
            ) = (
                self._localization_start(
                    event
                )
            )

            if (
                localization_length
                >= self.config.dsp.minimum_event_samples
            ):

                localization = (
                    self.localizer.locate_window(
                        start_sample=
                            localization_start,

                        length=
                            localization_length,
                    )
                )

        except Exception as exc:

            logger.warning(
                (
                    "Event %s localization "
                    "unavailable: %s"
                ),
                event.event_id,
                exc,
            )

        # ==============================================================
        # DSP + BEST MICROPHONE
        # ==============================================================

        best_node_id: (
            int
            | None
        ) = None

        features: (
            AcousticFeatures
            | None
        ) = None

        model_audio: (
            np.ndarray
            | None
        ) = None

        try:

            (
                best_node_id,
                features,
                model_audio,
            ) = (
                self._extract_best_channel_data(
                    event
                )
            )

        except Exception as exc:

            logger.warning(
                (
                    "Event %s DSP unavailable: %s"
                ),
                event.event_id,
                exc,
            )

        # ==============================================================
        # CLASSIFICATION
        # ==============================================================

        classification = (
            self._classify_event(
                event=
                    event,

                best_node_id=
                    best_node_id,

                features=
                    features,

                model_audio=
                    model_audio,
            )
        )

        # ==============================================================
        # EVENT WAV STORAGE
        # ==============================================================

        event_dir: (
            Path
            | None
        ) = None

        if (
            self.config.persistence.save_event_wav
        ):

            label = (
                self.active_session_label

                or (
                    f"session_"
                    f"{event.session_id:08X}"
                )
            )

            event_dir = (
                self.config.persistence.events_dir
                / label
                / (
                    f"event_"
                    f"{event.event_id:06d}"
                )
            )

            event_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ----------------------------------------------------------
            # SAVE SYNCHRONIZED WAV FROM EVERY NODE
            # ----------------------------------------------------------

            for node_id in sorted(
                self.config.expected_nodes
            ):

                try:

                    samples = (
                        self.streams.get_window(
                            node_id,
                            event.start_sample,
                            event.sample_count,
                        )
                    )

                    self._write_wav(
                        event_dir
                        / f"node_{node_id}.wav",

                        samples,
                    )

                except Exception as exc:

                    logger.warning(
                        (
                            "Event %s: failed to save "
                            "Node %s WAV: %s"
                        ),
                        event.event_id,
                        node_id,
                        exc,
                    )

        # ==============================================================
        # STORE CORE EVENT
        # ==============================================================

        database_event_id = (
            self.database.add_event(
                event,

                environment=
                    environment,

                localization=
                    localization,

                event_directory=(
                    None
                    if event_dir is None
                    else str(
                        event_dir
                    )
                ),

                best_node_id=
                    best_node_id,
            )
        )

        # ==============================================================
        # STORE DSP FEATURES
        # ==============================================================

        if (
            features is not None
            and best_node_id is not None
        ):

            try:

                self.database.add_event_features(
                    event_id=
                        database_event_id,

                    source_node_id=
                        best_node_id,

                    features=
                        features,
                )

            except Exception as exc:

                logger.warning(
                    (
                        "Event %s stored, but "
                        "DSP feature persistence "
                        "failed: %s"
                    ),
                    event.event_id,
                    exc,
                )

        # ==============================================================
        # STORE CLASSIFICATION
        # ==============================================================

        if classification is not None:

            try:

                self.database.add_classification(
                    event_id=
                        database_event_id,

                    result=
                        classification,
                )

            except Exception as exc:

                logger.warning(
                    (
                        "Event %s stored, but "
                        "classification persistence "
                        "failed: %s"
                    ),
                    event.event_id,
                    exc,
                )

        # ==============================================================
        # UPDATE RUNTIME STATE
        # ==============================================================

        self.completed_events += 1

        self.last_event_db_id = (
            database_event_id
        )

        self.last_best_node_id = (
            best_node_id
        )

        self.last_features = (
            features
        )

        self.last_classification = (
            classification
        )

        # ==============================================================
        # EVENT LOG
        # ==============================================================

        if (
            localization is not None
            and localization.position.success
        ):

            logger.info(
                (
                    "EVENT #%d DB#%d "
                    "nodes=%s "
                    "samples=%d..%d "
                    "best_node=%s "
                    "position=(%.3f, %.3f)m "
                    "c=%.2fm/s"
                ),

                event.event_id,

                database_event_id,

                event.trigger_nodes,

                event.start_sample,

                event.end_sample,

                (
                    str(
                        best_node_id
                    )
                    if best_node_id
                    is not None
                    else "N/A"
                ),

                localization.position.x,

                localization.position.y,

                localization.speed_of_sound_mps,
            )

        else:

            logger.info(
                (
                    "EVENT #%d DB#%d "
                    "nodes=%s "
                    "samples=%d..%d "
                    "best_node=%s "
                    "localization=unavailable"
                ),

                event.event_id,

                database_event_id,

                event.trigger_nodes,

                event.start_sample,

                event.end_sample,

                (
                    str(
                        best_node_id
                    )
                    if best_node_id
                    is not None
                    else "N/A"
                ),
            )

        # ==============================================================
        # DSP LOG
        # ==============================================================

        if features is not None:

            logger.info(
                (
                    "DSP event=%d "
                    "node=%s "
                    "duration=%.3fs "
                    "SNR=%s "
                    "dominant=%.1fHz "
                    "centroid=%.1fHz "
                    "bandwidth=%.1fHz "
                    "rolloff=%.1fHz"
                ),

                event.event_id,

                best_node_id,

                features.duration_s,

                (
                    f"{features.snr_db:.2f}dB"
                    if features.snr_db
                    is not None
                    else "N/A"
                ),

                features.dominant_frequency_hz,

                features.spectral_centroid_hz,

                features.spectral_bandwidth_hz,

                features.spectral_rolloff_hz,
            )

        # ==============================================================
        # CLASSIFICATION LOG
        # ==============================================================

        if (
            classification is not None
            and self.classifier_backend
            is not None
        ):

            logger.info(
                (
                    "CLASSIFICATION event=%d "
                    "backend=%s "
                    "label=%s "
                    "confidence=%.3f "
                    "second=%s "
                    "second_confidence=%.3f "
                    "margin=%.3f"
                ),

                event.event_id,

                self.classifier_backend.name,

                classification.label.value,

                classification.confidence,

                (
                    classification.second_label.value

                    if (
                        classification.second_label
                        is not None
                    )

                    else "N/A"
                ),

                classification.second_confidence,

                classification.margin,
            )

    # ==================================================================
    # ENVIRONMENT STORAGE
    # ==================================================================

    def add_environment(
        self,
        *,
        node_id: int,
        session_id: int,
        sample_index: int,
        environment,
    ) -> None:
        """
        Persist one environmental telemetry packet.
        """

        self.database.add_environment(
            session_id=
                session_id,

            node_id=
                node_id,

            sample_index=
                sample_index,

            environment=
                environment,
        )

    # ==================================================================
    # WAV WRITER
    # ==================================================================

    def _write_wav(
        self,
        path: Path,
        samples: np.ndarray,
    ) -> None:
        """
        Write one mono PCM16 event recording.
        """

        pcm = np.asarray(
            samples,
            dtype="<i2",
        )

        with wave.open(
            str(path),
            "wb",
        ) as wav:

            wav.setnchannels(
                self.config.audio.channels
            )

            wav.setsampwidth(
                self.config.audio.sample_width_bytes
            )

            wav.setframerate(
                self.config.audio.sample_rate
            )

            wav.writeframes(
                pcm.tobytes()
            )