"""
Tests for dsp.features.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. FeatureConfig validation
    2. RMS
    3. peak amplitude
    4. crest factor
    5. SNR
    6. STFT generation
    7. dominant-frequency estimation
    8. spectral flux
    9. MFCC statistics
    10. log-spectrogram generation
    11. short-event spectral padding
    12. silent-event handling
    13. complete acoustic-feature extraction
    14. amplitude-vs-analysis signal separation
    15. sample-rate consistency
    16. AcousticFeatures serialization

Scientific scope
----------------
These tests verify implementation behaviour and mathematical
consistency.

They do not validate biological classification accuracy.
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


from dsp.features import (
    AcousticFeatures,
    FeatureConfig,
    calculate_crest_factor,
    calculate_dominant_frequency,
    calculate_log_spectrogram,
    calculate_mfcc_statistics,
    calculate_peak_amplitude,
    calculate_rms,
    calculate_snr_db,
    calculate_spectral_flux,
    calculate_stft,
    extract_acoustic_features,
)

from dsp.preprocessing import (
    PreprocessingConfig,
    preprocess_event_audio,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SAMPLE_RATE = (
    48_000
)


N_FFT = (
    2048
)


HOP_LENGTH = (
    512
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
    phase_rad: float = 0.0,
) -> np.ndarray:
    """
    Generate deterministic mono float32 sinusoidal audio.
    """

    sample_count = int(
        duration_s
        * sample_rate
    )

    sample_indices = np.arange(
        sample_count,
        dtype=np.float64,
    )

    time_s = (
        sample_indices
        / float(
            sample_rate
        )
    )

    signal = (
        float(
            amplitude
        )
        * np.sin(
            (
                2.0
                * np.pi
                * float(
                    frequency_hz
                )
                * time_s
            )
            + float(
                phase_rad
            )
        )
    )

    return np.ascontiguousarray(
        signal,
        dtype=np.float32,
    )


def float_to_pcm16(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized float audio into signed PCM16.
    """

    clipped = np.clip(
        signal,
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


def make_preprocessed_sine(
    frequency_hz: float = 2000.0,
    *,
    duration_s: float = 1.0,
    amplitude: float = 0.4,
):
    """
    Produce PreprocessedAudio with filtering disabled.

    Disabling filtering makes primitive feature tests easier to reason
    about while still exercising the real preprocessing pipeline.
    """

    signal = generate_sine(
        frequency_hz,
        duration_s=
            duration_s,
        amplitude=
            amplitude,
    )

    pcm = float_to_pcm16(
        signal
    )

    return preprocess_event_audio(
        pcm,
        PreprocessingConfig(
            sample_rate=
                SAMPLE_RATE,

            remove_dc=
                True,

            bandpass_enabled=
                False,

            normalize_for_model=
                True,
        ),
    )


# ======================================================================
# FEATURE CONFIGURATION
# ======================================================================


def test_feature_config_accepts_project_defaults() -> None:

    config = FeatureConfig(
        sample_rate=
            SAMPLE_RATE,

        n_fft=
            2048,

        hop_length=
            512,

        n_mfcc=
            13,

        n_mels=
            64,

        mfcc_fmin_hz=
            50.0,

        mfcc_fmax_hz=
            16_000.0,

        roll_percent=
            0.85,
    )

    assert (
        config.sample_rate
        == SAMPLE_RATE
    )

    assert (
        config.n_fft
        == 2048
    )

    assert (
        config.hop_length
        == 512
    )

    assert (
        config.n_mfcc
        == 13
    )

    assert (
        config.n_mels
        == 64
    )


def test_feature_config_rejects_zero_sample_rate() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            sample_rate=
                0
        )


def test_feature_config_rejects_zero_fft_size() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            n_fft=
                0
        )


def test_feature_config_rejects_zero_hop_length() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            hop_length=
                0
        )


def test_feature_config_rejects_hop_larger_than_fft() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            n_fft=
                1024,

            hop_length=
                2048,
        )


def test_feature_config_rejects_zero_mfcc_count() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            n_mfcc=
                0
        )


def test_feature_config_rejects_fewer_mels_than_mfccs() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            n_mfcc=
                20,

            n_mels=
                10,
        )


@pytest.mark.parametrize(
    "roll_percent",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_feature_config_rejects_invalid_roll_percent(
    roll_percent: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            roll_percent=
                roll_percent
        )


def test_feature_config_rejects_negative_mfcc_min_frequency() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            mfcc_fmin_hz=
                -1.0
        )


def test_feature_config_rejects_mfcc_min_at_nyquist() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            sample_rate=
                SAMPLE_RATE,

            mfcc_fmin_hz=
                24_000.0,
        )


def test_feature_config_rejects_mfcc_max_below_min() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            mfcc_fmin_hz=
                1000.0,

            mfcc_fmax_hz=
                500.0,
        )


def test_feature_config_rejects_mfcc_max_above_nyquist() -> None:

    with pytest.raises(
        ValueError
    ):

        FeatureConfig(
            sample_rate=
                SAMPLE_RATE,

            mfcc_fmax_hz=
                25_000.0,
        )


# ======================================================================
# RMS
# ======================================================================


def test_calculate_rms_known_constant_signal() -> None:

    signal = np.full(
        1000,
        0.25,
        dtype=np.float32,
    )

    result = calculate_rms(
        signal
    )

    assert (
        result
        == pytest.approx(
            0.25,
            abs=
                1e-7,
        )
    )


def test_calculate_rms_sine_wave() -> None:
    """
    RMS of a sinusoid with peak amplitude A is A / sqrt(2).
    """

    amplitude = (
        0.6
    )

    signal = generate_sine(
        1000.0,
        amplitude=
            amplitude,
    )

    result = calculate_rms(
        signal
    )

    expected = (
        amplitude
        / np.sqrt(
            2.0
        )
    )

    assert (
        result
        == pytest.approx(
            expected,
            rel=
                1e-3,
        )
    )


def test_calculate_rms_silence_is_zero() -> None:

    signal = np.zeros(
        1024,
        dtype=np.float32,
    )

    assert (
        calculate_rms(
            signal
        )
        == 0.0
    )


def test_calculate_rms_rejects_integer_audio() -> None:

    signal = np.zeros(
        100,
        dtype=np.int16,
    )

    with pytest.raises(
        TypeError
    ):

        calculate_rms(
            signal
        )


def test_calculate_rms_rejects_non_finite_audio() -> None:

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

        calculate_rms(
            signal
        )


# ======================================================================
# PEAK AMPLITUDE
# ======================================================================


def test_calculate_peak_amplitude() -> None:

    signal = np.array(
        [
            -0.25,
            0.10,
            0.75,
            -0.50,
        ],
        dtype=np.float32,
    )

    result = calculate_peak_amplitude(
        signal
    )

    assert (
        result
        == pytest.approx(
            0.75
        )
    )


def test_peak_amplitude_silence_is_zero() -> None:

    signal = np.zeros(
        100,
        dtype=np.float32,
    )

    assert (
        calculate_peak_amplitude(
            signal
        )
        == 0.0
    )


# ======================================================================
# CREST FACTOR
# ======================================================================


def test_crest_factor_for_sine_is_approximately_sqrt_two() -> None:

    signal = generate_sine(
        1000.0,
        amplitude=
            0.7,
    )

    result = calculate_crest_factor(
        signal
    )

    assert (
        result
        == pytest.approx(
            np.sqrt(
                2.0
            ),
            rel=
                2e-3,
        )
    )


def test_crest_factor_for_silence_is_zero() -> None:

    signal = np.zeros(
        1024,
        dtype=np.float32,
    )

    assert (
        calculate_crest_factor(
            signal
        )
        == 0.0
    )


# ======================================================================
# SIGNAL-TO-NOISE RATIO
# ======================================================================


def test_snr_equal_signal_and_noise_is_zero_db() -> None:

    result = calculate_snr_db(
        0.25,
        0.25,
    )

    assert (
        result
        == pytest.approx(
            0.0,
            abs=
                1e-12,
        )
    )


def test_snr_ten_to_one_amplitude_ratio_is_twenty_db() -> None:

    result = calculate_snr_db(
        1.0,
        0.1,
    )

    assert (
        result
        == pytest.approx(
            20.0,
            rel=
                1e-10,
        )
    )


def test_snr_two_to_one_amplitude_ratio() -> None:

    result = calculate_snr_db(
        0.4,
        0.2,
    )

    assert (
        result
        == pytest.approx(
            20.0
            * np.log10(
                2.0
            ),
            rel=
                1e-10,
        )
    )


def test_snr_returns_none_without_noise_estimate() -> None:

    assert (
        calculate_snr_db(
            0.5,
            None,
        )
        is None
    )


@pytest.mark.parametrize(
    "signal_rms, noise_rms",
    [
        (
            0.0,
            0.1,
        ),
        (
            0.1,
            0.0,
        ),
        (
            np.nan,
            0.1,
        ),
        (
            0.1,
            np.inf,
        ),
    ],
)
def test_snr_returns_none_for_invalid_inputs(
    signal_rms: float,
    noise_rms: float,
) -> None:

    assert (
        calculate_snr_db(
            signal_rms,
            noise_rms,
        )
        is None
    )


# ======================================================================
# STFT
# ======================================================================


def test_stft_returns_expected_frequency_bins() -> None:

    signal = generate_sine(
        1000.0
    )

    (
        stft_matrix,
        magnitude,
        power,
    ) = calculate_stft(
        signal,
        n_fft=
            N_FFT,
        hop_length=
            HOP_LENGTH,
    )

    expected_bins = (
        N_FFT
        // 2
        + 1
    )

    assert (
        stft_matrix.shape[
            0
        ]
        == expected_bins
    )

    assert (
        magnitude.shape
        == stft_matrix.shape
    )

    assert (
        power.shape
        == stft_matrix.shape
    )

    assert (
        stft_matrix.dtype
        == np.complex64
    )

    assert (
        magnitude.dtype
        == np.float32
    )

    assert (
        power.dtype
        == np.float32
    )


def test_stft_power_matches_squared_magnitude() -> None:

    signal = generate_sine(
        1800.0,
        duration_s=
            0.25,
    )

    (
        _,
        magnitude,
        power,
    ) = calculate_stft(
        signal,
        n_fft=
            N_FFT,
        hop_length=
            HOP_LENGTH,
    )

    np.testing.assert_allclose(
        power,
        magnitude
        * magnitude,
        rtol=
            1e-5,
        atol=
            1e-8,
    )


def test_stft_short_signal_is_zero_padded() -> None:

    signal = generate_sine(
        1000.0,
        duration_s=
            0.005,
    )

    assert (
        signal.size
        < N_FFT
    )

    (
        _,
        magnitude,
        power,
    ) = calculate_stft(
        signal,
        n_fft=
            N_FFT,
        hop_length=
            HOP_LENGTH,
    )

    assert (
        magnitude.shape[
            0
        ]
        == N_FFT
        // 2
        + 1
    )

    assert (
        magnitude.shape[
            1
        ]
        >= 1
    )

    assert np.all(
        np.isfinite(
            power
        )
    )


def test_stft_rejects_zero_fft_size() -> None:

    signal = generate_sine(
        1000.0
    )

    with pytest.raises(
        ValueError
    ):

        calculate_stft(
            signal,
            n_fft=
                0,
            hop_length=
                512,
        )


def test_stft_rejects_hop_larger_than_fft() -> None:

    signal = generate_sine(
        1000.0
    )

    with pytest.raises(
        ValueError
    ):

        calculate_stft(
            signal,
            n_fft=
                512,
            hop_length=
                1024,
        )


# ======================================================================
# DOMINANT FREQUENCY
# ======================================================================


@pytest.mark.parametrize(
    "frequency_hz",
    [
        500.0,
        1000.0,
        2500.0,
        6000.0,
    ],
)
def test_dominant_frequency_tracks_sine_tone(
    frequency_hz: float,
) -> None:

    signal = generate_sine(
        frequency_hz,
        duration_s=
            1.0,
    )

    (
        _,
        _,
        power,
    ) = calculate_stft(
        signal,
        n_fft=
            N_FFT,
        hop_length=
            HOP_LENGTH,
    )

    result = calculate_dominant_frequency(
        power,
        sample_rate=
            SAMPLE_RATE,
        n_fft=
            N_FFT,
    )

    frequency_resolution = (
        SAMPLE_RATE
        / N_FFT
    )

    assert (
        result
        == pytest.approx(
            frequency_hz,
            abs=
                frequency_resolution,
        )
    )


def test_dominant_frequency_silence_is_zero() -> None:

    silence = np.zeros(
        SAMPLE_RATE,
        dtype=np.float32,
    )

    (
        _,
        _,
        power,
    ) = calculate_stft(
        silence,
        n_fft=
            N_FFT,
        hop_length=
            HOP_LENGTH,
    )

    assert (
        calculate_dominant_frequency(
            power,
            sample_rate=
                SAMPLE_RATE,
            n_fft=
                N_FFT,
        )
        == 0.0
    )


def test_dominant_frequency_ignores_dc_bin() -> None:

    power = np.zeros(
        (
            N_FFT
            // 2
            + 1,
            4,
        ),
        dtype=np.float32,
    )

    # Massive DC energy.
    power[
        0,
        :
    ] = (
        1000.0
    )

    # Non-DC peak at bin 10.
    power[
        10,
        :
    ] = (
        10.0
    )

    result = calculate_dominant_frequency(
        power,
        sample_rate=
            SAMPLE_RATE,
        n_fft=
            N_FFT,
    )

    expected = (
        10
        * SAMPLE_RATE
        / N_FFT
    )

    assert (
        result
        == pytest.approx(
            expected
        )
    )


# ======================================================================
# SPECTRAL FLUX
# ======================================================================


def test_spectral_flux_static_spectrum_is_zero() -> None:

    frame = np.array(
        [
            0.0,
            1.0,
            2.0,
            4.0,
        ],
        dtype=np.float32,
    )

    magnitude = np.column_stack(
        (
            frame,
            frame,
            frame,
        )
    ).astype(
        np.float32
    )

    result = calculate_spectral_flux(
        magnitude
    )

    assert (
        result
        == pytest.approx(
            0.0,
            abs=
                1e-12,
        )
    )


def test_spectral_flux_detects_changing_spectrum() -> None:

    magnitude = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float32,
    )

    result = calculate_spectral_flux(
        magnitude
    )

    assert (
        result
        > 0.0
    )


def test_spectral_flux_single_frame_is_zero() -> None:

    magnitude = np.ones(
        (
            100,
            1,
        ),
        dtype=np.float32,
    )

    assert (
        calculate_spectral_flux(
            magnitude
        )
        == 0.0
    )


def test_spectral_flux_rejects_one_dimensional_input() -> None:

    magnitude = np.ones(
        100,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        calculate_spectral_flux(
            magnitude
        )


# ======================================================================
# MFCC
# ======================================================================


def test_mfcc_statistics_have_expected_length() -> None:

    signal = generate_sine(
        2000.0,
        duration_s=
            1.0,
    )

    (
        _,
        _,
        power,
    ) = calculate_stft(
        signal,
        n_fft=
            N_FFT,
        hop_length=
            HOP_LENGTH,
    )

    (
        means,
        stds,
    ) = calculate_mfcc_statistics(
        power,

        sample_rate=
            SAMPLE_RATE,

        n_fft=
            N_FFT,

        n_mfcc=
            13,

        n_mels=
            64,

        fmin_hz=
            50.0,

        fmax_hz=
            16_000.0,
    )

    assert (
        len(
            means
        )
        == 13
    )

    assert (
        len(
            stds
        )
        == 13
    )

    assert all(
        np.isfinite(
            value
        )
        for value
        in means
    )

    assert all(
        np.isfinite(
            value
        )
        for value
        in stds
    )


def test_mfcc_statistics_for_silence_are_zero() -> None:

    power = np.zeros(
        (
            N_FFT
            // 2
            + 1,
            10,
        ),
        dtype=np.float32,
    )

    (
        means,
        stds,
    ) = calculate_mfcc_statistics(
        power,

        sample_rate=
            SAMPLE_RATE,

        n_fft=
            N_FFT,

        n_mfcc=
            13,

        n_mels=
            64,

        fmin_hz=
            50.0,

        fmax_hz=
            16_000.0,
    )

    assert (
        means
        == tuple(
            0.0
            for _ in range(
                13
            )
        )
    )

    assert (
        stds
        == tuple(
            0.0
            for _ in range(
                13
            )
        )
    )


# ======================================================================
# LOG SPECTROGRAM
# ======================================================================


def test_log_spectrogram_shapes_are_consistent() -> None:

    signal = generate_sine(
        2500.0,
        duration_s=
            0.5,
    )

    (
        spectrogram_db,
        frequencies,
        times,
    ) = calculate_log_spectrogram(
        signal,

        sample_rate=
            SAMPLE_RATE,

        n_fft=
            N_FFT,

        hop_length=
            HOP_LENGTH,
    )

    assert (
        spectrogram_db.ndim
        == 2
    )

    assert (
        frequencies.ndim
        == 1
    )

    assert (
        times.ndim
        == 1
    )

    assert (
        spectrogram_db.shape[
            0
        ]
        == frequencies.size
    )

    assert (
        spectrogram_db.shape[
            1
        ]
        == times.size
    )

    assert (
        frequencies.size
        == N_FFT
        // 2
        + 1
    )


def test_log_spectrogram_outputs_float32() -> None:

    signal = generate_sine(
        1500.0
    )

    (
        spectrogram_db,
        frequencies,
        times,
    ) = calculate_log_spectrogram(
        signal
    )

    assert (
        spectrogram_db.dtype
        == np.float32
    )

    assert (
        frequencies.dtype
        == np.float32
    )

    assert (
        times.dtype
        == np.float32
    )


def test_log_spectrogram_axes_start_at_zero() -> None:

    signal = generate_sine(
        1000.0,
        duration_s=
            0.25,
    )

    (
        _,
        frequencies,
        times,
    ) = calculate_log_spectrogram(
        signal
    )

    assert (
        frequencies[
            0
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        times[
            0
        ]
        == pytest.approx(
            0.0
        )
    )


def test_log_spectrogram_silence_is_finite() -> None:

    silence = np.zeros(
        SAMPLE_RATE,
        dtype=np.float32,
    )

    (
        spectrogram_db,
        frequencies,
        times,
    ) = calculate_log_spectrogram(
        silence
    )

    assert np.all(
        np.isfinite(
            spectrogram_db
        )
    )

    assert np.all(
        np.isfinite(
            frequencies
        )
    )

    assert np.all(
        np.isfinite(
            times
        )
    )

    assert np.all(
        spectrogram_db
        == 0.0
    )


# ======================================================================
# COMPLETE FEATURE EXTRACTION
# ======================================================================


def test_extract_features_from_sine_event() -> None:

    audio = make_preprocessed_sine(
        2000.0,
        duration_s=
            1.0,
        amplitude=
            0.4,
    )

    features = extract_acoustic_features(
        audio,
        noise_rms=
            0.02,
    )

    assert isinstance(
        features,
        AcousticFeatures,
    )

    assert (
        features.duration_s
        == pytest.approx(
            1.0,
            rel=
                1e-6,
        )
    )

    assert (
        features.rms
        > 0.0
    )

    assert (
        features.peak_amplitude
        > 0.0
    )

    assert (
        features.crest_factor
        > 1.0
    )

    assert (
        features.zero_crossing_rate
        >= 0.0
    )

    assert (
        features.dominant_frequency_hz
        == pytest.approx(
            2000.0,
            abs=
                SAMPLE_RATE
                / N_FFT,
        )
    )

    assert (
        features.spectral_centroid_hz
        > 0.0
    )

    assert (
        features.spectral_bandwidth_hz
        >= 0.0
    )

    assert (
        features.spectral_rolloff_hz
        > 0.0
    )

    assert (
        0.0
        <= features.spectral_flatness
        <= 1.0
    )

    assert (
        features.spectral_flux
        >= 0.0
    )

    assert (
        features.snr_db
        is not None
    )

    assert (
        len(
            features.mfcc_mean
        )
        == 13
    )

    assert (
        len(
            features.mfcc_std
        )
        == 13
    )


def test_extract_features_uses_original_event_duration_before_padding() -> None:
    """
    Spectral analysis may zero-pad a short event to n_fft, but duration
    must still represent the real captured event length.
    """

    sample_count = (
        240
    )

    signal = generate_sine(
        1000.0,
        duration_s=
            sample_count
            / SAMPLE_RATE,
        amplitude=
            0.3,
    )

    pcm = float_to_pcm16(
        signal
    )

    audio = preprocess_event_audio(
        pcm,
        PreprocessingConfig(
            sample_rate=
                SAMPLE_RATE,

            bandpass_enabled=
                False,
        ),
    )

    features = extract_acoustic_features(
        audio
    )

    assert (
        features.duration_s
        == pytest.approx(
            sample_count
            / SAMPLE_RATE,
            rel=
                1e-9,
        )
    )


def test_extract_features_preserves_amplitude_information() -> None:
    """
    Feature extraction must use amplitude_signal rather than the
    peak-normalized model_signal for RMS.
    """

    quiet = make_preprocessed_sine(
        1000.0,
        amplitude=
            0.10,
    )

    loud = make_preprocessed_sine(
        1000.0,
        amplitude=
            0.50,
    )

    quiet_features = extract_acoustic_features(
        quiet
    )

    loud_features = extract_acoustic_features(
        loud
    )

    # Both model signals are peak-normalized to approximately the same
    # maximum magnitude.
    assert (
        np.max(
            np.abs(
                quiet.model_signal
            )
        )
        == pytest.approx(
            np.max(
                np.abs(
                    loud.model_signal
                )
            ),
            abs=
                1e-4,
        )
    )

    # But extracted RMS must retain the actual recording-level
    # difference.
    assert (
        loud_features.rms
        > quiet_features.rms
        * 4.0
    )


def test_extract_features_without_noise_estimate_has_no_snr() -> None:

    audio = make_preprocessed_sine(
        1000.0
    )

    features = extract_acoustic_features(
        audio,
        noise_rms=
            None,
    )

    assert (
        features.snr_db
        is None
    )


def test_extract_features_snr_uses_event_rms() -> None:

    audio = make_preprocessed_sine(
        1000.0,
        amplitude=
            0.4,
    )

    preliminary = extract_acoustic_features(
        audio
    )

    noise_rms = (
        preliminary.rms
        / 10.0
    )

    features = extract_acoustic_features(
        audio,
        noise_rms=
            noise_rms,
    )

    assert (
        features.snr_db
        == pytest.approx(
            20.0,
            rel=
                1e-6,
        )
    )


def test_extract_features_rejects_sample_rate_mismatch() -> None:

    audio = make_preprocessed_sine(
        1000.0
    )

    config = FeatureConfig(
        sample_rate=
            44_100
    )

    with pytest.raises(
        ValueError
    ):

        extract_acoustic_features(
            audio,
            config=
                config,
        )


def test_extract_features_rejects_mismatched_signal_lengths() -> None:

    audio = make_preprocessed_sine(
        1000.0
    )

    # PreprocessedAudio is intentionally mutable, so deliberately corrupt
    # one representation to verify the interface guard.
    audio.analysis_signal = (
        audio.analysis_signal[
            :-1
        ]
    )

    with pytest.raises(
        ValueError
    ):

        extract_acoustic_features(
            audio
        )


# ======================================================================
# SILENT EVENT
# ======================================================================


def test_extract_features_handles_silence() -> None:

    pcm = np.zeros(
        SAMPLE_RATE,
        dtype=np.int16,
    )

    audio = preprocess_event_audio(
        pcm
    )

    features = extract_acoustic_features(
        audio
    )

    assert (
        features.duration_s
        == pytest.approx(
            1.0
        )
    )

    assert (
        features.rms
        == 0.0
    )

    assert (
        features.peak_amplitude
        == 0.0
    )

    assert (
        features.crest_factor
        == 0.0
    )

    assert (
        features.dominant_frequency_hz
        == 0.0
    )

    assert (
        features.spectral_centroid_hz
        == 0.0
    )

    assert (
        features.spectral_bandwidth_hz
        == 0.0
    )

    assert (
        features.spectral_rolloff_hz
        == 0.0
    )

    assert (
        features.spectral_flatness
        == 0.0
    )

    assert (
        features.spectral_flux
        == 0.0
    )

    assert (
        features.snr_db
        is None
    )

    assert all(
        value
        == 0.0
        for value
        in features.mfcc_mean
    )

    assert all(
        value
        == 0.0
        for value
        in features.mfcc_std
    )


# ======================================================================
# SERIALIZATION
# ======================================================================


def test_acoustic_features_to_dict() -> None:

    features = AcousticFeatures(
        duration_s=
            1.0,

        rms=
            0.1,

        peak_amplitude=
            0.2,

        crest_factor=
            2.0,

        zero_crossing_rate=
            0.05,

        dominant_frequency_hz=
            2000.0,

        spectral_centroid_hz=
            2200.0,

        spectral_bandwidth_hz=
            800.0,

        spectral_rolloff_hz=
            3500.0,

        spectral_flatness=
            0.1,

        spectral_flux=
            0.02,

        snr_db=
            12.0,

        mfcc_mean=
            tuple(
                float(
                    index
                )
                for index
                in range(
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

    result = features.to_dict()

    assert isinstance(
        result,
        dict,
    )

    assert (
        result[
            "duration_s"
        ]
        == 1.0
    )

    assert (
        result[
            "dominant_frequency_hz"
        ]
        == 2000.0
    )

    assert (
        result[
            "snr_db"
        ]
        == 12.0
    )

    assert (
        result[
            "mfcc_mean"
        ]
        == tuple(
            float(
                index
            )
            for index
            in range(
                13
            )
        )
    )


# ======================================================================
# BROAD NUMERICAL SANITY
# ======================================================================


def test_complete_feature_vector_contains_only_finite_numeric_values_except_optional_snr() -> None:

    audio = make_preprocessed_sine(
        3400.0,
        amplitude=
            0.35,
    )

    features = extract_acoustic_features(
        audio,
        noise_rms=
            0.01,
    )

    scalar_values = (
        features.duration_s,
        features.rms,
        features.peak_amplitude,
        features.crest_factor,
        features.zero_crossing_rate,
        features.dominant_frequency_hz,
        features.spectral_centroid_hz,
        features.spectral_bandwidth_hz,
        features.spectral_rolloff_hz,
        features.spectral_flatness,
        features.spectral_flux,
    )

    assert all(
        np.isfinite(
            value
        )
        for value
        in scalar_values
    )

    assert (
        features.snr_db
        is None
        or np.isfinite(
            features.snr_db
        )
    )

    assert all(
        np.isfinite(
            value
        )
        for value
        in features.mfcc_mean
    )

    assert all(
        np.isfinite(
            value
        )
        for value
        in features.mfcc_std
    )