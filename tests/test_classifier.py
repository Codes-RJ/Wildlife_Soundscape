"""
Tests for the acoustic-classification subsystem.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. AcousticClass public labels
    2. ClassificationResult invariants
    3. ClassificationInput structural validation
    4. feature-only backend input
    5. waveform-only generic input
    6. heuristic backend requirements
    7. broad bird-pattern classification
    8. broad insect-pattern classification
    9. broad amphibian-pattern classification
    10. broad mammal-pattern classification
    11. noise-pattern classification
    12. ambiguous -> UNKNOWN behaviour
    13. invalid-feature -> UNKNOWN behaviour
    14. extreme-low-SNR rejection
    15. missing-SNR handling
    16. final-score reranking after reliability adjustments
    17. score-map consistency
    18. backend adapter behaviour
    19. classifier identity/version metadata

Scientific scope
----------------
These tests validate deterministic behaviour of the transparent
heuristic baseline.

They do NOT establish biological accuracy.

A passing test means the software implements its declared heuristic
decision policy consistently; it does not mean the classifier has been
validated for species identification.
"""


from __future__ import annotations


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np
import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.classification.base import (
    ClassificationInput,
)

from wildlife_soundscape.classification.classifier import (
    AcousticClass,
    ClassificationResult,
    HeuristicClassifier,
)

from wildlife_soundscape.classification.heuristic_backend import (
    HeuristicClassifierBackend,
)

from wildlife_soundscape.dsp.features import (
    AcousticFeatures,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SAMPLE_RATE = (
    48_000
)


SESSION_ID = (
    0x12345678
)


# ======================================================================
# FEATURE FACTORY
# ======================================================================


def make_features(
    *,
    duration_s: float = 1.0,
    rms: float = 0.10,
    peak_amplitude: float = 0.20,
    crest_factor: float = 2.0,
    zero_crossing_rate: float = 0.10,
    dominant_frequency_hz: float = 3000.0,
    spectral_centroid_hz: float = 3500.0,
    spectral_bandwidth_hz: float = 1800.0,
    spectral_rolloff_hz: float = 5500.0,
    spectral_flatness: float = 0.10,
    spectral_flux: float = 0.030,
    snr_db: float | None = 15.0,
) -> AcousticFeatures:
    """
    Construct one complete deterministic AcousticFeatures object.

    MFCC values are included because they are part of the shared feature
    model even though the current heuristic classifier does not use
    them directly.
    """

    return AcousticFeatures(
        duration_s=
            duration_s,

        rms=
            rms,

        peak_amplitude=
            peak_amplitude,

        crest_factor=
            crest_factor,

        zero_crossing_rate=
            zero_crossing_rate,

        dominant_frequency_hz=
            dominant_frequency_hz,

        spectral_centroid_hz=
            spectral_centroid_hz,

        spectral_bandwidth_hz=
            spectral_bandwidth_hz,

        spectral_rolloff_hz=
            spectral_rolloff_hz,

        spectral_flatness=
            spectral_flatness,

        spectral_flux=
            spectral_flux,

        snr_db=
            snr_db,

        mfcc_mean=
            tuple(
                0.0
                for _ in range(
                    13
                )
            ),

        mfcc_std=
            tuple(
                0.0
                for _ in range(
                    13
                )
            ),
    )


# ======================================================================
# CLEAR SYNTHETIC FEATURE PATTERNS
# ======================================================================


def bird_features(
    *,
    snr_db: float | None = 15.0,
) -> AcousticFeatures:
    """
    Synthetic feature vector deliberately separated toward the heuristic
    bird-pattern region.

    This is not intended to represent one particular bird species.
    """

    return make_features(
        duration_s=
            5.7,

        dominant_frequency_hz=
            800.0,

        spectral_centroid_hz=
            8000.0,

        spectral_flatness=
            0.13,

        spectral_flux=
            0.037,

        zero_crossing_rate=
            0.037,

        spectral_bandwidth_hz=
            110.0,

        spectral_rolloff_hz=
            1730.0,

        snr_db=
            snr_db,
    )


def insect_features() -> AcousticFeatures:
    """
    Synthetic high-frequency insect-like heuristic pattern.
    """

    return make_features(
        duration_s=
            7.0,

        dominant_frequency_hz=
            12_200.0,

        spectral_centroid_hz=
            10_350.0,

        spectral_flatness=
            0.34,

        spectral_flux=
            0.0004,

        zero_crossing_rate=
            0.449,

        spectral_bandwidth_hz=
            2060.0,

        spectral_rolloff_hz=
            12_170.0,

        snr_db=
            15.0,
    )


def amphibian_features() -> AcousticFeatures:
    """
    Synthetic feature vector intentionally separated toward the
    heuristic amphibian-pattern region.
    """

    return make_features(
        duration_s=
            6.56,

        dominant_frequency_hz=
            3790.0,

        spectral_centroid_hz=
            1860.0,

        spectral_flatness=
            0.48,

        spectral_flux=
            0.0028,

        zero_crossing_rate=
            0.043,

        spectral_bandwidth_hz=
            8270.0,

        spectral_rolloff_hz=
            14_960.0,

        snr_db=
            15.0,
    )


def mammal_features() -> AcousticFeatures:
    """
    Synthetic low-frequency mammal-like heuristic pattern.
    """

    return make_features(
        duration_s=
            0.138,

        dominant_frequency_hz=
            83.0,

        spectral_centroid_hz=
            103.0,

        spectral_flatness=
            0.638,

        spectral_flux=
            0.078,

        zero_crossing_rate=
            0.028,

        spectral_bandwidth_hz=
            1495.0,

        spectral_rolloff_hz=
            1192.0,

        snr_db=
            15.0,
    )


def noise_features() -> AcousticFeatures:
    """
    Synthetic broad/flat-spectrum noise-like pattern.
    """

    return make_features(
        duration_s=
            3.23,

        dominant_frequency_hz=
            61.0,

        spectral_centroid_hz=
            9200.0,

        spectral_flatness=
            0.937,

        spectral_flux=
            0.0009,

        zero_crossing_rate=
            0.032,

        spectral_bandwidth_hz=
            8380.0,

        spectral_rolloff_hz=
            684.0,

        snr_db=
            15.0,
    )


# ======================================================================
# ACOUSTIC CLASS ENUM
# ======================================================================


def test_acoustic_class_values_are_stable() -> None:
    """
    These values are persisted in SQLite and later consumed by the
    dashboard/research layer, so accidental renaming is significant.
    """

    assert (
        AcousticClass.BIRD.value
        == "bird"
    )

    assert (
        AcousticClass.INSECT.value
        == "insect"
    )

    assert (
        AcousticClass.AMPHIBIAN.value
        == "amphibian"
    )

    assert (
        AcousticClass.MAMMAL.value
        == "mammal"
    )

    assert (
        AcousticClass.NOISE.value
        == "noise"
    )

    assert (
        AcousticClass.UNKNOWN.value
        == "unknown"
    )


# ======================================================================
# CLASSIFICATION RESULT
# ======================================================================


def test_classification_result_accepts_no_secondary_candidate() -> None:

    result = ClassificationResult(
        label=
            AcousticClass.UNKNOWN,

        confidence=
            1.0,

        second_label=
            None,

        second_confidence=
            None,

        margin=
            0.0,

        scores={
            acoustic_class.value:
                0.0
            for acoustic_class
            in AcousticClass
        },

        reasons=(
            "invalid feature vector",
        ),
    )

    assert (
        result.second_label
        is None
    )

    assert (
        result.second_confidence
        is None
    )


def test_classification_result_rejects_confidence_above_one() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationResult(
            label=
                AcousticClass.BIRD,

            confidence=
                1.1,

            second_label=
                AcousticClass.INSECT,

            second_confidence=
                0.4,

            margin=
                0.2,

            scores={
                AcousticClass.BIRD.value:
                    1.0
            },

            reasons=(
                "test",
            ),
        )


def test_classification_result_rejects_negative_confidence() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationResult(
            label=
                AcousticClass.BIRD,

            confidence=
                -0.1,

            second_label=
                AcousticClass.INSECT,

            second_confidence=
                0.2,

            margin=
                0.1,

            scores={
                AcousticClass.BIRD.value:
                    0.5
            },

            reasons=(
                "test",
            ),
        )


def test_classification_result_requires_secondary_confidence_when_label_exists() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationResult(
            label=
                AcousticClass.BIRD,

            confidence=
                0.8,

            second_label=
                AcousticClass.INSECT,

            second_confidence=
                None,

            margin=
                0.2,

            scores={
                AcousticClass.BIRD.value:
                    0.8,

                AcousticClass.INSECT.value:
                    0.6,
            },

            reasons=(
                "test",
            ),
        )


def test_classification_result_requires_none_secondary_confidence_without_label() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationResult(
            label=
                AcousticClass.UNKNOWN,

            confidence=
                1.0,

            second_label=
                None,

            second_confidence=
                0.0,

            margin=
                0.0,

            scores={
                AcousticClass.UNKNOWN.value:
                    1.0
            },

            reasons=(
                "test",
            ),
        )


def test_classification_result_rejects_same_primary_and_secondary_label() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationResult(
            label=
                AcousticClass.BIRD,

            confidence=
                0.8,

            second_label=
                AcousticClass.BIRD,

            second_confidence=
                0.6,

            margin=
                0.2,

            scores={
                AcousticClass.BIRD.value:
                    0.8
            },

            reasons=(
                "test",
            ),
        )


def test_classification_result_rejects_invalid_score_value() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationResult(
            label=
                AcousticClass.BIRD,

            confidence=
                0.8,

            second_label=
                AcousticClass.INSECT,

            second_confidence=
                0.4,

            margin=
                0.4,

            scores={
                AcousticClass.BIRD.value:
                    1.5
            },

            reasons=(
                "test",
            ),
        )


# ======================================================================
# CLASSIFICATION INPUT
# ======================================================================


def test_classification_input_accepts_feature_only_input() -> None:

    features = (
        bird_features()
    )

    classification_input = ClassificationInput(
        features=
            features,

        sample_rate=
            SAMPLE_RATE,

        source_node_id=
            1,

        detector_event_id=
            5,

        session_id=
            SESSION_ID,
    )

    assert (
        classification_input.has_features
    )

    assert not (
        classification_input.has_audio
    )

    assert (
        classification_input.features
        is features
    )


def test_classification_input_accepts_waveform_only_input() -> None:

    waveform = np.zeros(
        48_000,
        dtype=np.float32,
    )

    classification_input = ClassificationInput(
        features=
            None,

        sample_rate=
            SAMPLE_RATE,

        model_audio=
            waveform,

        source_node_id=
            2,

        session_id=
            SESSION_ID,
    )

    assert not (
        classification_input.has_features
    )

    assert (
        classification_input.has_audio
    )


@pytest.mark.parametrize(
    "sample_rate",
    [
        48_000.5,
        True,
        "48000",
    ],
)
def test_classification_input_rejects_non_integer_sample_rate(
    sample_rate,
) -> None:

    with pytest.raises(
        TypeError
    ):

        ClassificationInput(
            features=
                bird_features(),

            sample_rate=
                sample_rate,
        )


def test_classification_input_rejects_zero_sample_rate() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationInput(
            features=
                bird_features(),

            sample_rate=
                0,
        )


@pytest.mark.parametrize(
    "node_id",
    [
        0,
        256,
    ],
)
def test_classification_input_rejects_invalid_source_node_id(
    node_id: int,
) -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationInput(
            features=
                bird_features(),

            sample_rate=
                SAMPLE_RATE,

            source_node_id=
                node_id,
        )


def test_classification_input_rejects_zero_session_id() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationInput(
            features=
                bird_features(),

            sample_rate=
                SAMPLE_RATE,

            session_id=
                0,
        )


def test_classification_input_rejects_negative_detector_event_id() -> None:

    with pytest.raises(
        ValueError
    ):

        ClassificationInput(
            features=
                bird_features(),

            sample_rate=
                SAMPLE_RATE,

            detector_event_id=
                -1,
        )


def test_classification_input_rejects_model_audio_list() -> None:

    with pytest.raises(
        TypeError
    ):

        ClassificationInput(
            features=
                None,

            sample_rate=
                SAMPLE_RATE,

            model_audio=[
                0.0,
                1.0,
            ],
        )


def test_classification_input_rejects_multichannel_model_audio() -> None:

    waveform = np.zeros(
        (
            100,
            2,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        ClassificationInput(
            features=
                None,

            sample_rate=
                SAMPLE_RATE,

            model_audio=
                waveform,
        )


def test_classification_input_rejects_empty_model_audio() -> None:

    waveform = np.array(
        [],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        ClassificationInput(
            features=
                None,

            sample_rate=
                SAMPLE_RATE,

            model_audio=
                waveform,
        )


def test_classification_input_rejects_complex_model_audio() -> None:

    waveform = np.array(
        [
            1.0
            + 2.0j,
        ],
        dtype=np.complex64,
    )

    with pytest.raises(
        TypeError
    ):

        ClassificationInput(
            features=
                None,

            sample_rate=
                SAMPLE_RATE,

            model_audio=
                waveform,
        )


def test_classification_input_rejects_non_finite_model_audio() -> None:

    waveform = np.array(
        [
            0.0,
            np.nan,
            1.0,
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        ClassificationInput(
            features=
                None,

            sample_rate=
                SAMPLE_RATE,

            model_audio=
                waveform,
        )


# ======================================================================
# CLEAR BROAD-PATTERN CLASSIFICATIONS
# ======================================================================


@pytest.mark.parametrize(
    "features_factory, expected_label",
    [
        (
            bird_features,
            AcousticClass.BIRD,
        ),
        (
            insect_features,
            AcousticClass.INSECT,
        ),
        (
            amphibian_features,
            AcousticClass.AMPHIBIAN,
        ),
        (
            mammal_features,
            AcousticClass.MAMMAL,
        ),
        (
            noise_features,
            AcousticClass.NOISE,
        ),
    ],
)
def test_classifier_accepts_clear_synthetic_patterns(
    features_factory,
    expected_label: AcousticClass,
) -> None:
    """
    These are deliberately separated synthetic feature vectors designed
    to test deterministic decision branches.

    They are not biological validation samples.
    """

    classifier = (
        HeuristicClassifier()
    )

    result = classifier.classify(
        features_factory()
    )

    assert (
        result.label
        == expected_label
    )

    assert (
        result.confidence
        >= classifier.MIN_ACCEPT_CONFIDENCE
    )

    assert (
        result.margin
        >= classifier.MIN_ACCEPT_MARGIN
    )

    assert (
        result.second_label
        is not None
    )

    assert (
        result.second_confidence
        is not None
    )


# ======================================================================
# SCORE-MAP CONTRACT
# ======================================================================


def test_classifier_returns_complete_score_map() -> None:

    result = HeuristicClassifier().classify(
        bird_features()
    )

    assert (
        set(
            result.scores
        )
        == {
            acoustic_class.value
            for acoustic_class
            in AcousticClass
        }
    )


def test_classifier_scores_are_bounded() -> None:

    result = HeuristicClassifier().classify(
        insect_features()
    )

    assert all(
        0.0
        <= score
        <= 1.0
        for score
        in result.scores.values()
    )


def test_accepted_result_score_matches_primary_label() -> None:

    result = HeuristicClassifier().classify(
        bird_features()
    )

    assert (
        result.confidence
        == pytest.approx(
            result.scores[
                result.label.value
            ]
        )
    )


def test_accepted_result_second_confidence_matches_score_map() -> None:

    result = HeuristicClassifier().classify(
        insect_features()
    )

    assert (
        result.second_label
        is not None
    )

    assert (
        result.second_confidence
        is not None
    )

    assert (
        result.second_confidence
        == pytest.approx(
            result.scores[
                result.second_label.value
            ]
        )
    )


# ======================================================================
# INVALID FEATURE VECTOR
# ======================================================================


def test_invalid_zero_energy_features_return_unknown() -> None:

    features = make_features(
        rms=
            0.0,

        peak_amplitude=
            0.0,
    )

    result = HeuristicClassifier().classify(
        features
    )

    assert (
        result.label
        == AcousticClass.UNKNOWN
    )

    assert (
        result.confidence
        == 1.0
    )

    assert (
        result.second_label
        is None
    )

    assert (
        result.second_confidence
        is None
    )

    assert (
        result.margin
        == 0.0
    )

    assert (
        result.scores[
            AcousticClass.UNKNOWN.value
        ]
        == 1.0
    )


def test_non_finite_feature_vector_returns_unknown() -> None:

    features = make_features(
        spectral_centroid_hz=
            float(
                "nan"
            )
    )

    result = HeuristicClassifier().classify(
        features
    )

    assert (
        result.label
        == AcousticClass.UNKNOWN
    )

    assert (
        result.second_label
        is None
    )

    assert (
        result.second_confidence
        is None
    )


def test_classifier_rejects_wrong_feature_object_type() -> None:

    classifier = (
        HeuristicClassifier()
    )

    with pytest.raises(
        TypeError
    ):

        classifier.classify(
            object()
        )


# ======================================================================
# EXTREME LOW SNR
# ======================================================================


def test_extreme_low_snr_forces_unknown() -> None:

    features = bird_features(
        snr_db=
            -4.0
    )

    result = HeuristicClassifier().classify(
        features
    )

    assert (
        result.label
        == AcousticClass.UNKNOWN
    )

    assert (
        result.confidence
        == 1.0
    )

    assert (
        result.second_label
        == AcousticClass.NOISE
    )

    assert (
        result.second_confidence
        == pytest.approx(
            0.75
        )
    )

    assert (
        result.scores[
            AcousticClass.UNKNOWN.value
        ]
        == 1.0
    )

    assert (
        result.scores[
            AcousticClass.NOISE.value
        ]
        == pytest.approx(
            0.75
        )
    )


# ======================================================================
# MISSING SNR
# ======================================================================


def test_missing_snr_does_not_automatically_invalidate_classification() -> None:

    features = bird_features(
        snr_db=
            None
    )

    result = HeuristicClassifier().classify(
        features
    )

    assert (
        result.label
        == AcousticClass.BIRD
    )

    assert (
        result.confidence
        > 0.0
    )


def test_missing_snr_reduces_decision_confidence() -> None:

    classifier = (
        HeuristicClassifier()
    )

    known_snr = classifier.classify(
        bird_features(
            snr_db=
                15.0
        )
    )

    missing_snr = classifier.classify(
        bird_features(
            snr_db=
                None
        )
    )

    assert (
        missing_snr.confidence
        < known_snr.confidence
    )


# ======================================================================
# AMBIGUOUS EVIDENCE
# ======================================================================


def test_ambiguous_feature_vector_returns_unknown() -> None:
    """
    This feature vector intentionally produces strongly overlapping
    biological class scores.
    """

    features = make_features(
        duration_s=
            1.5,

        dominant_frequency_hz=
            1200.0,

        spectral_centroid_hz=
            1500.0,

        spectral_flatness=
            0.15,

        spectral_flux=
            0.015,

        zero_crossing_rate=
            0.08,

        spectral_bandwidth_hz=
            1000.0,

        spectral_rolloff_hz=
            3000.0,

        snr_db=
            15.0,
    )

    classifier = (
        HeuristicClassifier()
    )

    result = classifier.classify(
        features
    )

    assert (
        result.label
        == AcousticClass.UNKNOWN
    )

    # UNKNOWN retains the strongest actual acoustic candidate as its
    # secondary explanation.
    assert (
        result.second_label
        is not None
    )

    assert (
        result.second_label
        != AcousticClass.UNKNOWN
    )

    assert (
        result.second_confidence
        is not None
    )

    assert (
        result.margin
        < classifier.MIN_ACCEPT_MARGIN
    )


# ======================================================================
# FINAL-SCORE RERANKING REGRESSION
# ======================================================================


def test_flat_spectrum_adjustment_is_applied_before_final_ranking() -> None:
    """
    Regression test for classifier v1.2.

    Raw high-frequency evidence can favor INSECT for this vector, while
    the extremely flat/broad spectrum provides stronger final evidence
    for NOISE.

    The classifier must rank AFTER reliability adjustments.

    It must not:

        rank insect
        penalize insect
        still return insect

    after NOISE has become the stronger final score.
    """

    features = noise_features()

    result = HeuristicClassifier().classify(
        features
    )

    assert (
        result.label
        == AcousticClass.NOISE
    )

    assert (
        result.scores[
            AcousticClass.NOISE.value
        ]
        > result.scores[
            AcousticClass.INSECT.value
        ]
    )

    assert (
        result.confidence
        == pytest.approx(
            result.scores[
                AcousticClass.NOISE.value
            ]
        )
    )


# ======================================================================
# CONFIDENCE PROPERTY
# ======================================================================


def test_clear_classification_is_confident() -> None:

    result = HeuristicClassifier().classify(
        bird_features()
    )

    assert (
        result.is_confident
    )


def test_unknown_classification_is_not_confident() -> None:

    result = HeuristicClassifier().classify(
        make_features(
            rms=
                0.0,

            peak_amplitude=
                0.0,
        )
    )

    assert not (
        result.is_confident
    )


# ======================================================================
# SCIENTIFIC-SCOPE EXPLANATION
# ======================================================================


def test_accepted_classification_states_non_species_scope() -> None:

    result = HeuristicClassifier().classify(
        bird_features()
    )

    assert any(
        "not species identification"
        in reason.lower()
        for reason
        in result.reasons
    )


# ======================================================================
# CLASSIFIER METADATA
# ======================================================================


def test_classifier_result_contains_current_classifier_identity() -> None:

    result = HeuristicClassifier().classify(
        bird_features()
    )

    assert (
        result.classifier_name
        == "heuristic_acoustic_classifier"
    )

    assert (
        result.classifier_version
        == "1.2"
    )


# ======================================================================
# HEURISTIC BACKEND
# ======================================================================


def test_heuristic_backend_capabilities() -> None:

    backend = (
        HeuristicClassifierBackend()
    )

    assert (
        backend.requires_features
    )

    assert not (
        backend.requires_audio
    )

    assert (
        backend.name
        == "heuristic_backend"
    )

    assert (
        backend.version
        == "1.0"
    )


def test_heuristic_backend_classifies_feature_input() -> None:

    backend = (
        HeuristicClassifierBackend()
    )

    classification_input = ClassificationInput(
        features=
            bird_features(),

        sample_rate=
            SAMPLE_RATE,

        source_node_id=
            1,

        detector_event_id=
            10,

        session_id=
            SESSION_ID,
    )

    result = backend.classify(
        classification_input
    )

    assert (
        result.label
        == AcousticClass.BIRD
    )


def test_heuristic_backend_does_not_require_model_audio() -> None:

    backend = (
        HeuristicClassifierBackend()
    )

    classification_input = ClassificationInput(
        features=
            bird_features(),

        sample_rate=
            SAMPLE_RATE,

        model_audio=
            None,

        session_id=
            SESSION_ID,
    )

    result = backend.classify(
        classification_input
    )

    assert (
        result.label
        == AcousticClass.BIRD
    )


def test_heuristic_backend_rejects_missing_features() -> None:

    backend = (
        HeuristicClassifierBackend()
    )

    waveform = np.zeros(
        SAMPLE_RATE,
        dtype=np.float32,
    )

    classification_input = ClassificationInput(
        features=
            None,

        sample_rate=
            SAMPLE_RATE,

        model_audio=
            waveform,

        session_id=
            SESSION_ID,
    )

    with pytest.raises(
        ValueError
    ):

        backend.classify(
            classification_input
        )


def test_heuristic_backend_matches_direct_classifier() -> None:

    features = (
        insect_features()
    )

    direct_classifier = (
        HeuristicClassifier()
    )

    backend = HeuristicClassifierBackend(
        direct_classifier
    )

    direct_result = direct_classifier.classify(
        features
    )

    backend_result = backend.classify(
        ClassificationInput(
            features=
                features,

            sample_rate=
                SAMPLE_RATE,

            session_id=
                SESSION_ID,
        )
    )

    assert (
        backend_result
        == direct_result
    )


# ======================================================================
# MULTIPLE INPUT SOURCES
# ======================================================================


def test_heuristic_backend_ignores_optional_audio_when_features_are_present() -> None:
    """
    The heuristic backend is feature-based.

    Supplying model_audio is permitted by the generic ClassificationInput
    but should not alter the heuristic result.
    """

    features = (
        bird_features()
    )

    waveform = np.zeros(
        SAMPLE_RATE,
        dtype=np.float32,
    )

    backend = (
        HeuristicClassifierBackend()
    )

    without_audio = backend.classify(
        ClassificationInput(
            features=
                features,

            sample_rate=
                SAMPLE_RATE,

            session_id=
                SESSION_ID,
        )
    )

    with_audio = backend.classify(
        ClassificationInput(
            features=
                features,

            sample_rate=
                SAMPLE_RATE,

            model_audio=
                waveform,

            session_id=
                SESSION_ID,
        )
    )

    assert (
        with_audio
        == without_audio
    )