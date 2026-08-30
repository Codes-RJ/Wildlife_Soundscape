"""
Tests for dsp.preprocessing.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. PCM16 -> float32 conversion
    2. mono/input validation
    3. non-finite signal rejection
    4. DC-offset removal
    5. Butterworth band-pass behaviour
    6. short-signal filtering behaviour
    7. peak normalization
    8. silence / near-silence handling
    9. preprocessing configuration validation
    10. complete event preprocessing
    11. preservation of amplitude information
    12. preservation of original PCM input
    13. output dtype / shape / contiguity
    14. peak-before-normalization metadata
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


from dsp.preprocessing import (
    PreprocessingConfig,
    apply_bandpass_filter,
    peak_normalize,
    pcm16_to_float32,
    preprocess_event_audio,
    remove_dc_offset,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SAMPLE_RATE = (
    48_000
)


# ======================================================================
# TEST HELPERS
# ======================================================================


def generate_sine(
    frequency_hz: float,
    *,
    duration_s: float = 1.0,
    amplitude: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a deterministic float32 sine wave.
    """

    sample_count = int(
        duration_s
        * sample_rate
    )

    time_s = (
        np.arange(
            sample_count,
            dtype=np.float64,
        )
        / sample_rate
    )

    signal = (
        amplitude
        * np.sin(
            2.0
            * np.pi
            * frequency_hz
            * time_s
        )
    )

    return np.asarray(
        signal,
        dtype=np.float32,
    )


def float_to_pcm16(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized floating-point audio to PCM16.
    """

    clipped = np.clip(
        audio,
        -1.0,
        32767.0
        / 32768.0,
    )

    pcm = np.round(
        clipped
        * 32768.0
    )

    return np.ascontiguousarray(
        pcm.astype(
            np.int16
        )
    )


def rms(
    signal: np.ndarray,
) -> float:
    """
    Calculate RMS using a float64 accumulator.
    """

    if (
        signal.size
        == 0
    ):

        return (
            0.0
        )

    signal64 = (
        signal.astype(
            np.float64,
            copy=False,
        )
    )

    return float(
        np.sqrt(
            np.mean(
                signal64
                * signal64,
                dtype=np.float64,
            )
        )
    )


# ======================================================================
# PCM16 -> FLOAT32
# ======================================================================


def test_pcm16_to_float32_known_values() -> None:

    pcm = np.array(
        [
            -32768,
            -16384,
            0,
            16384,
            32767,
        ],
        dtype=np.int16,
    )

    result = pcm16_to_float32(
        pcm
    )

    expected = np.array(
        [
            -1.0,
            -0.5,
            0.0,
            0.5,
            32767.0
            / 32768.0,
        ],
        dtype=np.float32,
    )

    assert (
        result.dtype
        == np.float32
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=
            1e-7,
    )


def test_pcm16_conversion_preserves_shape() -> None:

    pcm = np.zeros(
        1024,
        dtype=np.int16,
    )

    result = pcm16_to_float32(
        pcm
    )

    assert (
        result.shape
        == pcm.shape
    )


def test_pcm16_conversion_is_contiguous() -> None:

    pcm = np.arange(
        2048,
        dtype=np.int16,
    )[
        ::2
    ]

    assert not (
        pcm.flags.c_contiguous
    )

    result = pcm16_to_float32(
        pcm
    )

    assert (
        result.flags.c_contiguous
    )


def test_pcm16_rejects_non_numpy_input() -> None:

    with pytest.raises(
        TypeError
    ):

        pcm16_to_float32(
            [
                1,
                2,
                3,
            ]
        )


def test_pcm16_rejects_wrong_dtype() -> None:

    wrong = np.zeros(
        100,
        dtype=np.float32,
    )

    with pytest.raises(
        TypeError
    ):

        pcm16_to_float32(
            wrong
        )


def test_pcm16_rejects_multichannel_array() -> None:

    stereo = np.zeros(
        (
            100,
            2,
        ),
        dtype=np.int16,
    )

    with pytest.raises(
        ValueError
    ):

        pcm16_to_float32(
            stereo
        )


def test_pcm16_rejects_empty_array() -> None:

    empty = np.array(
        [],
        dtype=np.int16,
    )

    with pytest.raises(
        ValueError
    ):

        pcm16_to_float32(
            empty
        )


# ======================================================================
# DC OFFSET REMOVAL
# ======================================================================


def test_remove_dc_offset() -> None:

    signal = generate_sine(
        1000.0,
        amplitude=
            0.3,
    )

    signal = (
        signal
        + np.float32(
            0.15
        )
    )

    assert (
        abs(
            float(
                np.mean(
                    signal
                )
            )
        )
        > 0.1
    )

    corrected = remove_dc_offset(
        signal
    )

    assert (
        abs(
            float(
                np.mean(
                    corrected,
                    dtype=np.float64,
                )
            )
        )
        < 1e-6
    )


def test_remove_dc_does_not_destroy_ac_component() -> None:

    original = generate_sine(
        1000.0,
        amplitude=
            0.4,
    )

    biased = (
        original
        + np.float32(
            0.2
        )
    )

    corrected = remove_dc_offset(
        biased
    )

    np.testing.assert_allclose(
        corrected,
        original,
        atol=
            1e-5,
    )


def test_remove_dc_output_is_float32() -> None:

    signal = generate_sine(
        1000.0
    )

    corrected = remove_dc_offset(
        signal
    )

    assert (
        corrected.dtype
        == np.float32
    )


def test_remove_dc_rejects_integer_audio() -> None:

    signal = np.zeros(
        100,
        dtype=np.int16,
    )

    with pytest.raises(
        TypeError
    ):

        remove_dc_offset(
            signal
        )


def test_remove_dc_rejects_non_finite_audio() -> None:

    signal = np.array(
        [
            0.0,
            1.0,
            np.nan,
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        remove_dc_offset(
            signal
        )


# ======================================================================
# BAND-PASS FILTER
# ======================================================================


def test_bandpass_preserves_1000_hz_tone() -> None:
    """
    1 kHz lies safely inside the 100 Hz - 16 kHz analysis band.
    """

    signal = generate_sine(
        1000.0,
        amplitude=
            0.5,
    )

    before = rms(
        signal
    )

    filtered = apply_bandpass_filter(
        signal,

        sample_rate=
            SAMPLE_RATE,

        low_cutoff_hz=
            100.0,

        high_cutoff_hz=
            16_000.0,

        order=
            4,
    )

    after = rms(
        filtered
    )

    ratio = (
        after
        / before
    )

    assert (
        ratio
        > 0.90
    )

    assert (
        ratio
        < 1.10
    )


def test_bandpass_rejects_low_frequency() -> None:
    """
    20 Hz lies well below the 100 Hz lower cutoff.
    """

    signal = generate_sine(
        20.0,
        amplitude=
            0.5,
    )

    before = rms(
        signal
    )

    filtered = apply_bandpass_filter(
        signal,

        sample_rate=
            SAMPLE_RATE,

        low_cutoff_hz=
            100.0,

        high_cutoff_hz=
            16_000.0,

        order=
            4,
    )

    after = rms(
        filtered
    )

    attenuation_ratio = (
        after
        / before
    )

    assert (
        attenuation_ratio
        < 0.10
    )


def test_bandpass_rejects_high_frequency() -> None:
    """
    20 kHz lies above the 16 kHz upper cutoff.
    """

    signal = generate_sine(
        20_000.0,
        amplitude=
            0.5,
    )

    before = rms(
        signal
    )

    filtered = apply_bandpass_filter(
        signal,

        sample_rate=
            SAMPLE_RATE,

        low_cutoff_hz=
            100.0,

        high_cutoff_hz=
            16_000.0,

        order=
            4,
    )

    after = rms(
        filtered
    )

    attenuation_ratio = (
        after
        / before
    )

    assert (
        attenuation_ratio
        < 0.35
    )


def test_bandpass_output_is_float32() -> None:

    signal = generate_sine(
        1000.0
    )

    filtered = apply_bandpass_filter(
        signal,

        sample_rate=
            SAMPLE_RATE,

        low_cutoff_hz=
            100.0,

        high_cutoff_hz=
            16_000.0,

        order=
            4,
    )

    assert (
        filtered.dtype
        == np.float32
    )


def test_bandpass_output_is_finite() -> None:

    signal = generate_sine(
        2400.0,
        amplitude=
            0.4,
    )

    filtered = apply_bandpass_filter(
        signal,

        sample_rate=
            SAMPLE_RATE,

        low_cutoff_hz=
            100.0,

        high_cutoff_hz=
            16_000.0,

        order=
            4,
    )

    assert np.all(
        np.isfinite(
            filtered
        )
    )


def test_bandpass_handles_very_short_signal() -> None:
    """
    Very short signals are deliberately returned safely rather than
    causing scipy.signal.sosfiltfilt to fail.
    """

    signal = np.array(
        [
            0.1,
            -0.1,
            0.2,
            -0.2,
        ],
        dtype=np.float32,
    )

    filtered = apply_bandpass_filter(
        signal,

        sample_rate=
            SAMPLE_RATE,

        low_cutoff_hz=
            100.0,

        high_cutoff_hz=
            16_000.0,

        order=
            4,
    )

    assert (
        filtered.shape
        == signal.shape
    )

    assert np.all(
        np.isfinite(
            filtered
        )
    )

    np.testing.assert_array_equal(
        filtered,
        signal,
    )

    assert not np.shares_memory(
        filtered,
        signal,
    )


def test_bandpass_rejects_non_finite_audio() -> None:

    signal = np.array(
        [
            0.0,
            np.inf,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        apply_bandpass_filter(
            signal,

            sample_rate=
                SAMPLE_RATE,

            low_cutoff_hz=
                100.0,

            high_cutoff_hz=
                16_000.0,

            order=
                4,
        )


def test_bandpass_rejects_invalid_sample_rate() -> None:

    signal = generate_sine(
        1000.0
    )

    with pytest.raises(
        ValueError
    ):

        apply_bandpass_filter(
            signal,

            sample_rate=
                0,

            low_cutoff_hz=
                100.0,

            high_cutoff_hz=
                16_000.0,

            order=
                4,
        )


def test_bandpass_rejects_reversed_cutoffs() -> None:

    signal = generate_sine(
        1000.0
    )

    with pytest.raises(
        ValueError
    ):

        apply_bandpass_filter(
            signal,

            sample_rate=
                SAMPLE_RATE,

            low_cutoff_hz=
                5000.0,

            high_cutoff_hz=
                1000.0,

            order=
                4,
        )


def test_bandpass_rejects_cutoff_at_nyquist() -> None:

    signal = generate_sine(
        1000.0
    )

    with pytest.raises(
        ValueError
    ):

        apply_bandpass_filter(
            signal,

            sample_rate=
                SAMPLE_RATE,

            low_cutoff_hz=
                100.0,

            high_cutoff_hz=
                24_000.0,

            order=
                4,
        )


def test_bandpass_rejects_zero_filter_order() -> None:

    signal = generate_sine(
        1000.0
    )

    with pytest.raises(
        ValueError
    ):

        apply_bandpass_filter(
            signal,

            sample_rate=
                SAMPLE_RATE,

            low_cutoff_hz=
                100.0,

            high_cutoff_hz=
                16_000.0,

            order=
                0,
        )


# ======================================================================
# PEAK NORMALIZATION
# ======================================================================


def test_peak_normalization_reaches_target_peak() -> None:

    signal = generate_sine(
        1000.0,
        amplitude=
            0.2,
    )

    normalized = peak_normalize(
        signal,
        target_peak=
            0.98,
    )

    peak = float(
        np.max(
            np.abs(
                normalized
            )
        )
    )

    assert (
        peak
        == pytest.approx(
            0.98,
            abs=
                1e-5,
        )
    )


def test_peak_normalization_preserves_shape() -> None:

    signal = generate_sine(
        2000.0
    )

    normalized = peak_normalize(
        signal
    )

    assert (
        normalized.shape
        == signal.shape
    )


def test_peak_normalization_output_is_float32() -> None:

    signal = generate_sine(
        2000.0
    )

    normalized = peak_normalize(
        signal
    )

    assert (
        normalized.dtype
        == np.float32
    )


def test_peak_normalization_handles_silence() -> None:

    silence = np.zeros(
        48_000,
        dtype=np.float32,
    )

    normalized = peak_normalize(
        silence
    )

    assert np.all(
        normalized
        == 0.0
    )

    assert np.all(
        np.isfinite(
            normalized
        )
    )


def test_peak_normalization_handles_near_silence() -> None:

    near_silence = np.full(
        100,
        1e-15,
        dtype=np.float32,
    )

    normalized = peak_normalize(
        near_silence
    )

    assert np.all(
        normalized
        == 0.0
    )


@pytest.mark.parametrize(
    "target_peak",
    [
        0.0,
        -0.1,
        1.5,
    ],
)
def test_peak_normalization_rejects_invalid_target(
    target_peak: float,
) -> None:

    signal = generate_sine(
        1000.0
    )

    with pytest.raises(
        ValueError
    ):

        peak_normalize(
            signal,
            target_peak=
                target_peak,
        )


def test_peak_normalization_rejects_non_finite_audio() -> None:

    signal = np.array(
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

        peak_normalize(
            signal
        )


# ======================================================================
# CONFIGURATION VALIDATION
# ======================================================================


def test_preprocessing_config_rejects_invalid_sample_rate() -> None:

    with pytest.raises(
        ValueError
    ):

        PreprocessingConfig(
            sample_rate=
                0
        )


def test_preprocessing_config_rejects_zero_filter_order() -> None:

    with pytest.raises(
        ValueError
    ):

        PreprocessingConfig(
            filter_order=
                0
        )


def test_preprocessing_config_rejects_reversed_cutoffs() -> None:

    with pytest.raises(
        ValueError
    ):

        PreprocessingConfig(
            low_cutoff_hz=
                5000.0,

            high_cutoff_hz=
                1000.0,
        )


def test_preprocessing_config_rejects_frequency_at_nyquist() -> None:
    """
    At 48 kHz sampling, Nyquist = 24 kHz.
    """

    with pytest.raises(
        ValueError
    ):

        PreprocessingConfig(
            sample_rate=
                48_000,

            low_cutoff_hz=
                100.0,

            high_cutoff_hz=
                24_000.0,
        )


@pytest.mark.parametrize(
    "target_peak",
    [
        0.0,
        -0.01,
        1.01,
    ],
)
def test_preprocessing_config_rejects_invalid_model_target_peak(
    target_peak: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        PreprocessingConfig(
            model_target_peak=
                target_peak
        )


def test_preprocessing_config_accepts_expected_project_values() -> None:

    config = PreprocessingConfig(
        sample_rate=
            48_000,

        low_cutoff_hz=
            100.0,

        high_cutoff_hz=
            16_000.0,

        filter_order=
            4,

        model_target_peak=
            0.98,
    )

    assert (
        config.sample_rate
        == 48_000
    )

    assert (
        config.low_cutoff_hz
        == 100.0
    )

    assert (
        config.high_cutoff_hz
        == 16_000.0
    )

    assert (
        config.filter_order
        == 4
    )

    assert (
        config.model_target_peak
        == 0.98
    )


# ======================================================================
# COMPLETE PREPROCESSING PIPELINE
# ======================================================================


def test_complete_preprocessing_pipeline() -> None:
    """
    Simulate a realistic captured 1 kHz event with a small DC offset.
    """

    signal = generate_sine(
        1000.0,
        amplitude=
            0.4,
    )

    signal = (
        signal
        + np.float32(
            0.05
        )
    )

    pcm = float_to_pcm16(
        signal
    )

    original_pcm = (
        pcm.copy()
    )

    config = PreprocessingConfig(
        sample_rate=
            SAMPLE_RATE,

        remove_dc=
            True,

        bandpass_enabled=
            True,

        low_cutoff_hz=
            100.0,

        high_cutoff_hz=
            16_000.0,

        filter_order=
            4,

        normalize_for_model=
            True,

        model_target_peak=
            0.98,
    )

    result = preprocess_event_audio(
        pcm,
        config,
    )

    # ==============================================================
    # ORIGINAL PCM MUST REMAIN UNCHANGED
    # ==============================================================

    np.testing.assert_array_equal(
        pcm,
        original_pcm,
    )

    # ==============================================================
    # OUTPUT SHAPES
    # ==============================================================

    assert (
        result.raw_float.shape
        == pcm.shape
    )

    assert (
        result.amplitude_signal.shape
        == pcm.shape
    )

    assert (
        result.analysis_signal.shape
        == pcm.shape
    )

    assert (
        result.model_signal.shape
        == pcm.shape
    )

    # ==============================================================
    # OUTPUT TYPES
    # ==============================================================

    assert (
        result.raw_float.dtype
        == np.float32
    )

    assert (
        result.amplitude_signal.dtype
        == np.float32
    )

    assert (
        result.analysis_signal.dtype
        == np.float32
    )

    assert (
        result.model_signal.dtype
        == np.float32
    )

    # ==============================================================
    # CONTIGUOUS BUFFERS
    # ==============================================================

    assert (
        result.raw_float.flags.c_contiguous
    )

    assert (
        result.amplitude_signal.flags.c_contiguous
    )

    assert (
        result.analysis_signal.flags.c_contiguous
    )

    assert (
        result.model_signal.flags.c_contiguous
    )

    # ==============================================================
    # SAMPLE RATE
    # ==============================================================

    assert (
        result.sample_rate
        == SAMPLE_RATE
    )

    # ==============================================================
    # DC REMOVAL
    # ==============================================================

    assert (
        abs(
            float(
                np.mean(
                    result.amplitude_signal,
                    dtype=np.float64,
                )
            )
        )
        < 1e-4
    )

    # ==============================================================
    # MODEL NORMALIZATION
    # ==============================================================

    model_peak = float(
        np.max(
            np.abs(
                result.model_signal
            )
        )
    )

    assert (
        model_peak
        == pytest.approx(
            0.98,
            abs=
                1e-4,
        )
    )

    # ==============================================================
    # PRE-NORMALIZATION PEAK METADATA
    # ==============================================================

    expected_analysis_peak = float(
        np.max(
            np.abs(
                result.analysis_signal
            )
        )
    )

    assert (
        result.peak_before_normalization
        == pytest.approx(
            expected_analysis_peak,
            rel=
                1e-6,
            abs=
                1e-8,
        )
    )

    # ==============================================================
    # FINITE DATA
    # ==============================================================

    assert np.all(
        np.isfinite(
            result.raw_float
        )
    )

    assert np.all(
        np.isfinite(
            result.amplitude_signal
        )
    )

    assert np.all(
        np.isfinite(
            result.analysis_signal
        )
    )

    assert np.all(
        np.isfinite(
            result.model_signal
        )
    )


def test_preprocessing_raw_float_is_independent_of_input_pcm() -> None:

    pcm = np.array(
        [
            1000,
            -1000,
            2000,
            -2000,
        ],
        dtype=np.int16,
    )

    result = preprocess_event_audio(
        pcm,
        PreprocessingConfig(
            bandpass_enabled=
                False,
            normalize_for_model=
                False,
        ),
    )

    saved_raw = (
        result.raw_float.copy()
    )

    pcm[
        :
    ] = (
        0
    )

    np.testing.assert_array_equal(
        result.raw_float,
        saved_raw,
    )


def test_preprocessing_keeps_amplitude_before_normalization() -> None:
    """
    amplitude_signal must not be peak-normalized because RMS/SNR
    measurements depend on the original relative recording amplitude.
    """

    signal = generate_sine(
        1000.0,
        amplitude=
            0.10,
    )

    pcm = float_to_pcm16(
        signal
    )

    result = preprocess_event_audio(
        pcm
    )

    amplitude_peak = float(
        np.max(
            np.abs(
                result.amplitude_signal
            )
        )
    )

    model_peak = float(
        np.max(
            np.abs(
                result.model_signal
            )
        )
    )

    assert (
        amplitude_peak
        < 0.2
    )

    assert (
        model_peak
        == pytest.approx(
            0.98,
            abs=
                1e-4,
        )
    )


def test_pipeline_handles_silence_without_nan() -> None:

    pcm = np.zeros(
        48_000,
        dtype=np.int16,
    )

    result = preprocess_event_audio(
        pcm
    )

    for signal in (
        result.raw_float,
        result.amplitude_signal,
        result.analysis_signal,
        result.model_signal,
    ):

        assert np.all(
            np.isfinite(
                signal
            )
        )

    assert np.all(
        result.raw_float
        == 0.0
    )

    assert np.all(
        result.amplitude_signal
        == 0.0
    )

    assert np.all(
        result.model_signal
        == 0.0
    )

    assert (
        result.peak_before_normalization
        == 0.0
    )


def test_pipeline_without_bandpass() -> None:

    signal = generate_sine(
        1000.0,
        amplitude=
            0.3,
    )

    pcm = float_to_pcm16(
        signal
    )

    config = PreprocessingConfig(
        bandpass_enabled=
            False,

        normalize_for_model=
            False,
    )

    result = preprocess_event_audio(
        pcm,
        config,
    )

    np.testing.assert_allclose(
        result.analysis_signal,
        result.amplitude_signal,
        atol=
            1e-7,
    )

    np.testing.assert_allclose(
        result.model_signal,
        result.analysis_signal,
        atol=
            1e-7,
    )


def test_pipeline_without_dc_removal() -> None:

    signal = generate_sine(
        1000.0,
        amplitude=
            0.2,
    )

    signal = (
        signal
        + np.float32(
            0.10
        )
    )

    pcm = float_to_pcm16(
        signal
    )

    config = PreprocessingConfig(
        remove_dc=
            False,

        bandpass_enabled=
            False,

        normalize_for_model=
            False,
    )

    result = preprocess_event_audio(
        pcm,
        config,
    )

    np.testing.assert_allclose(
        result.amplitude_signal,
        result.raw_float,
        atol=
            1e-7,
    )

    assert (
        abs(
            float(
                np.mean(
                    result.amplitude_signal,
                    dtype=np.float64,
                )
            )
        )
        > 0.05
    )


def test_pipeline_without_model_normalization_preserves_analysis_peak() -> None:

    signal = generate_sine(
        1800.0,
        amplitude=
            0.15,
    )

    pcm = float_to_pcm16(
        signal
    )

    config = PreprocessingConfig(
        bandpass_enabled=
            False,

        normalize_for_model=
            False,
    )

    result = preprocess_event_audio(
        pcm,
        config,
    )

    np.testing.assert_allclose(
        result.model_signal,
        result.analysis_signal,
        atol=
            1e-7,
    )

    model_peak = float(
        np.max(
            np.abs(
                result.model_signal
            )
        )
    )

    assert (
        model_peak
        == pytest.approx(
            result.peak_before_normalization,
            rel=
                1e-6,
        )
    )


def test_preprocessing_rejects_wrong_pcm_dtype() -> None:

    audio = np.zeros(
        1024,
        dtype=np.float32,
    )

    with pytest.raises(
        TypeError
    ):

        preprocess_event_audio(
            audio
        )


def test_preprocessing_rejects_empty_pcm() -> None:

    audio = np.array(
        [],
        dtype=np.int16,
    )

    with pytest.raises(
        ValueError
    ):

        preprocess_event_audio(
            audio
        )