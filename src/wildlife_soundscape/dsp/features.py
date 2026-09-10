"""
Acoustic feature extraction.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Extract compact event-level acoustic descriptors from preprocessed
audio on the laptop.

Feature groups
--------------
Time domain:
    - duration
    - RMS energy
    - peak amplitude
    - crest factor
    - zero-crossing rate

Frequency domain:
    - dominant frequency
    - spectral centroid
    - spectral bandwidth
    - spectral rolloff
    - spectral flatness
    - spectral flux

Cepstral:
    - MFCC mean
    - MFCC standard deviation

Signal quality:
    - estimated SNR

Important
---------
Amplitude-dependent features are calculated from amplitude_signal.

Spectral / cepstral features are calculated from analysis_signal.

The peak-normalized model_signal is deliberately NOT used for
feature extraction because normalization removes absolute amplitude
information.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from typing import (
    Any,
    Final,
)

import librosa
import numpy as np

from numpy.typing import (
    NDArray,
)

from .preprocessing import (
    PreprocessedAudio,
)


# ======================================================================
# TYPE ALIASES
# ======================================================================


FloatArray = NDArray[np.float32]

FloatSpectrogram = NDArray[np.float32]

ComplexSpectrogram = NDArray[np.complex64]


# ======================================================================
# CONSTANTS
# ======================================================================


EPSILON: Final[float] = 1e-12


# ======================================================================
# CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class FeatureConfig:
    """
    Configuration for acoustic feature extraction.
    """

    # ------------------------------------------------------------------
    # SAMPLE RATE
    # ------------------------------------------------------------------

    sample_rate: int = 48_000

    # ------------------------------------------------------------------
    # STFT
    # ------------------------------------------------------------------

    n_fft: int = 2048

    hop_length: int = 512

    # ------------------------------------------------------------------
    # MFCC
    # ------------------------------------------------------------------

    n_mfcc: int = 13

    n_mels: int = 64

    mfcc_fmin_hz: float = 50.0

    mfcc_fmax_hz: float | None = 16_000.0

    # ------------------------------------------------------------------
    # SPECTRAL ROLLOFF
    # ------------------------------------------------------------------

    roll_percent: float = 0.85

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate feature-extraction configuration.
        """

        if self.sample_rate <= 0:
            raise ValueError(("sample_rate must be greater than 0."))

        # --------------------------------------------------------------
        # STFT
        # --------------------------------------------------------------

        if self.n_fft <= 0:
            raise ValueError("n_fft must be greater than 0.")

        if self.hop_length <= 0:
            raise ValueError("hop_length must be greater than 0.")

        if self.hop_length > self.n_fft:
            raise ValueError(("hop_length must not exceed n_fft."))

        # --------------------------------------------------------------
        # MFCC
        # --------------------------------------------------------------

        if self.n_mfcc <= 0:
            raise ValueError("n_mfcc must be greater than 0.")

        if self.n_mels <= 0:
            raise ValueError("n_mels must be greater than 0.")

        if self.n_mels < self.n_mfcc:
            raise ValueError(("n_mels must be greater than or equal to n_mfcc."))

        # --------------------------------------------------------------
        # ROLLOFF
        # --------------------------------------------------------------

        if not (0.0 < self.roll_percent < 1.0):
            raise ValueError(("roll_percent must lie between 0 and 1."))

        # --------------------------------------------------------------
        # MFCC FREQUENCY RANGE
        # --------------------------------------------------------------

        nyquist = self.sample_rate / 2.0

        if self.mfcc_fmin_hz < 0.0:
            raise ValueError(("mfcc_fmin_hz cannot be negative."))

        if self.mfcc_fmin_hz >= nyquist:
            raise ValueError(("mfcc_fmin_hz must remain below Nyquist frequency."))

        if self.mfcc_fmax_hz is not None:
            if self.mfcc_fmax_hz <= self.mfcc_fmin_hz:
                raise ValueError(("mfcc_fmax_hz must exceed mfcc_fmin_hz."))

            if self.mfcc_fmax_hz > nyquist:
                raise ValueError(("mfcc_fmax_hz cannot exceed Nyquist frequency."))


# ======================================================================
# FEATURE MODEL
# ======================================================================


@dataclass(
    slots=True,
)
class AcousticFeatures:
    """
    Compact numerical representation of one acoustic event.

    This object is the common feature vector consumed by:

        - heuristic classification
        - database persistence
        - dashboard analytics
        - later behavioral-pattern analysis
    """

    # ------------------------------------------------------------------
    # GENERAL
    # ------------------------------------------------------------------

    duration_s: float

    # ------------------------------------------------------------------
    # TIME DOMAIN
    # ------------------------------------------------------------------

    rms: float

    peak_amplitude: float

    crest_factor: float

    zero_crossing_rate: float

    # ------------------------------------------------------------------
    # FREQUENCY DOMAIN
    # ------------------------------------------------------------------

    dominant_frequency_hz: float

    spectral_centroid_hz: float

    spectral_bandwidth_hz: float

    spectral_rolloff_hz: float

    spectral_flatness: float

    spectral_flux: float

    # ------------------------------------------------------------------
    # SIGNAL QUALITY
    # ------------------------------------------------------------------

    snr_db: float | None

    # ------------------------------------------------------------------
    # CEPSTRAL
    # ------------------------------------------------------------------

    mfcc_mean: tuple[
        float,
        ...,
    ]

    mfcc_std: tuple[
        float,
        ...,
    ]

    # ==================================================================
    # SERIALIZATION
    # ==================================================================

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Convert features to a JSON/database-friendly dictionary.
        """

        return asdict(self)


# ======================================================================
# SIGNAL VALIDATION
# ======================================================================


def _validate_signal(
    signal: np.ndarray,
) -> None:
    """
    Validate one mono floating-point signal.
    """

    if not isinstance(
        signal,
        np.ndarray,
    ):
        raise TypeError(("signal must be a NumPy ndarray."))

    if signal.ndim != 1:
        raise ValueError((f"Expected mono 1-D audio, got shape {signal.shape}."))

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.issubdtype(
        signal.dtype,
        np.floating,
    ):
        raise TypeError(("Feature extraction expects floating-point audio."))

    if not np.all(np.isfinite(signal)):
        raise ValueError(("Signal contains NaN or infinite values."))


# ======================================================================
# SPECTROGRAM VALIDATION
# ======================================================================


def _validate_spectrogram(
    spectrogram: np.ndarray,
    *,
    name: str,
) -> None:
    """
    Validate a non-empty two-dimensional spectrogram.
    """

    if not isinstance(
        spectrogram,
        np.ndarray,
    ):
        raise TypeError(f"{name} must be a NumPy ndarray.")

    if spectrogram.ndim != 2:
        raise ValueError(f"{name} must be 2-D.")

    if spectrogram.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(spectrogram)):
        raise ValueError((f"{name} contains NaN or infinite values."))


# ======================================================================
# SHORT EVENT PADDING
# ======================================================================


def _ensure_spectral_length(
    signal: FloatArray,
    n_fft: int,
) -> FloatArray:
    """
    Zero-pad events shorter than one FFT frame.

    Padding is used only for spectral calculations.

    The stored event duration continues to represent the original
    unpadded event.
    """

    _validate_signal(signal)

    if n_fft <= 0:
        raise ValueError("n_fft must be greater than 0.")

    if signal.size >= n_fft:
        return np.ascontiguousarray(
            signal,
            dtype=np.float32,
        )

    padding = n_fft - signal.size

    padded = np.pad(
        signal,
        pad_width=(
            0,
            padding,
        ),
        mode="constant",
    )

    return np.ascontiguousarray(
        padded,
        dtype=np.float32,
    )


# ======================================================================
# FINITE REDUCTION HELPERS
# ======================================================================


def _safe_mean(
    values: np.ndarray,
) -> float:
    """
    Return the arithmetic mean of finite values.

    Returns zero when no finite values exist.
    """

    values = np.asarray(values)

    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return 0.0

    value = float(
        np.mean(
            finite_values,
            dtype=np.float64,
        )
    )

    if not np.isfinite(value):
        return 0.0

    return value


# ======================================================================
# RMS
# ======================================================================


def calculate_rms(
    signal: FloatArray,
) -> float:
    """
    Calculate root-mean-square signal amplitude.

    RMS = sqrt(mean(x²))
    """

    _validate_signal(signal)

    signal64 = signal.astype(
        np.float64,
        copy=False,
    )

    value = float(
        np.sqrt(
            np.mean(
                signal64 * signal64,
                dtype=np.float64,
            )
        )
    )

    if not np.isfinite(value):
        return 0.0

    return max(
        0.0,
        value,
    )


# ======================================================================
# PEAK AMPLITUDE
# ======================================================================


def calculate_peak_amplitude(
    signal: FloatArray,
) -> float:
    """
    Calculate maximum absolute signal amplitude.
    """

    _validate_signal(signal)

    peak = float(np.max(np.abs(signal)))

    if not np.isfinite(peak):
        return 0.0

    return max(
        0.0,
        peak,
    )


# ======================================================================
# CREST FACTOR
# ======================================================================


def calculate_crest_factor(
    signal: FloatArray,
) -> float:
    """
    Calculate crest factor.

    crest factor = peak / RMS

    Higher values indicate a more impulsive/transient waveform.
    """

    peak = calculate_peak_amplitude(signal)

    rms = calculate_rms(signal)

    if rms <= EPSILON:
        return 0.0

    value = peak / rms

    if not np.isfinite(value):
        return 0.0

    return float(value)


# ======================================================================
# SNR
# ======================================================================


def calculate_snr_db(
    signal_rms: float,
    noise_rms: float | None,
) -> float | None:
    """
    Estimate event SNR from signal and background RMS.

    Formula
    -------
    SNR_dB = 20 log10(signal_rms / noise_rms)

    Notes
    -----
    `signal_rms` currently represents the event-level RMS.

    `noise_rms` is estimated independently from the pre-trigger
    background region by EventPipeline.

    None is returned when a valid noise estimate is unavailable.
    """

    if noise_rms is None:
        return None

    if not np.isfinite(signal_rms) or not np.isfinite(noise_rms):
        return None

    if signal_rms <= EPSILON:
        return None

    if noise_rms <= EPSILON:
        return None

    ratio = float(signal_rms) / float(noise_rms)

    if ratio <= 0.0 or not np.isfinite(ratio):
        return None

    snr = float(20.0 * np.log10(ratio))

    if not np.isfinite(snr):
        return None

    return snr


# ======================================================================
# STFT
# ======================================================================


def calculate_stft(
    signal: FloatArray,
    *,
    n_fft: int,
    hop_length: int,
) -> tuple[
    ComplexSpectrogram,
    FloatSpectrogram,
    FloatSpectrogram,
]:
    """
    Calculate the short-time Fourier transform.

    Returns
    -------
    stft_matrix
        Complex STFT coefficients.

    magnitude
        Absolute STFT magnitude.

    power
        Squared magnitude spectrogram.
    """

    _validate_signal(signal)

    if n_fft <= 0:
        raise ValueError("n_fft must be greater than 0.")

    if hop_length <= 0:
        raise ValueError("hop_length must be greater than 0.")

    if hop_length > n_fft:
        raise ValueError(("hop_length cannot exceed n_fft."))

    working_signal = _ensure_spectral_length(
        signal,
        n_fft,
    )

    stft_matrix = librosa.stft(
        y=working_signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=False,
    )

    stft_matrix = np.asarray(
        stft_matrix,
        dtype=np.complex64,
    )

    magnitude = np.asarray(
        np.abs(stft_matrix),
        dtype=np.float32,
    )

    power = np.asarray(
        magnitude * magnitude,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(magnitude)):
        raise ValueError(("STFT produced non-finite magnitude values."))

    if not np.all(np.isfinite(power)):
        raise ValueError(("STFT produced non-finite power values."))

    return (
        stft_matrix,
        magnitude,
        power,
    )


# ======================================================================
# DOMINANT FREQUENCY
# ======================================================================


def calculate_dominant_frequency(
    power_spectrogram: FloatSpectrogram,
    *,
    sample_rate: int,
    n_fft: int,
) -> float:
    """
    Estimate dominant frequency using mean STFT power.

    The DC bin is deliberately excluded.
    """

    _validate_spectrogram(
        power_spectrogram,
        name="power_spectrogram",
    )

    average_power = np.mean(
        power_spectrogram,
        axis=1,
        dtype=np.float64,
    )

    if average_power.size <= 1:
        return 0.0

    # --------------------------------------------------------------
    # IGNORE DC
    # --------------------------------------------------------------

    average_power[0] = 0.0

    max_power = float(np.max(average_power))

    if not np.isfinite(max_power) or max_power <= EPSILON:
        return 0.0

    dominant_bin = int(np.argmax(average_power))

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=(1.0 / sample_rate),
    )

    if dominant_bin >= frequencies.size:
        return 0.0

    value = float(frequencies[dominant_bin])

    if not np.isfinite(value):
        return 0.0

    return max(
        0.0,
        value,
    )


# ======================================================================
# SPECTRAL FLUX
# ======================================================================


def calculate_spectral_flux(
    magnitude_spectrogram: FloatSpectrogram,
) -> float:
    """
    Calculate positive spectral flux.

    Each spectral frame is normalized before comparison.

    This reduces the influence of absolute loudness and makes flux more
    representative of changing spectral shape.

    Only positive changes are retained, representing newly appearing
    spectral energy.
    """

    _validate_spectrogram(
        magnitude_spectrogram,
        name="magnitude_spectrogram",
    )

    frame_count = magnitude_spectrogram.shape[1]

    if frame_count < 2:
        return 0.0

    magnitude = magnitude_spectrogram.astype(
        np.float64,
        copy=False,
    )

    frame_energy = np.sum(
        magnitude,
        axis=0,
        keepdims=True,
        dtype=np.float64,
    )

    normalized = magnitude / np.maximum(
        frame_energy,
        EPSILON,
    )

    frame_difference = np.diff(
        normalized,
        axis=1,
    )

    positive_difference = np.maximum(
        frame_difference,
        0.0,
    )

    flux_per_frame = np.sqrt(
        np.sum(
            positive_difference * positive_difference,
            axis=0,
            dtype=np.float64,
        )
    )

    return max(
        0.0,
        _safe_mean(flux_per_frame),
    )


# ======================================================================
# MFCC
# ======================================================================


def calculate_mfcc_statistics(
    power_spectrogram: FloatSpectrogram,
    *,
    sample_rate: int,
    n_fft: int,
    n_mfcc: int,
    n_mels: int,
    fmin_hz: float,
    fmax_hz: float | None,
) -> tuple[
    tuple[
        float,
        ...,
    ],
    tuple[
        float,
        ...,
    ],
]:
    """
    Calculate event-level MFCC summary statistics.

    Only mean and standard deviation for each MFCC coefficient are
    stored, keeping the database compact while retaining cepstral
    information.
    """

    _validate_spectrogram(
        power_spectrogram,
        name="power_spectrogram",
    )

    # --------------------------------------------------------------
    # SILENT / NEAR-SILENT SIGNAL
    # --------------------------------------------------------------

    maximum_power = float(np.max(power_spectrogram))

    if not np.isfinite(maximum_power) or maximum_power <= EPSILON:
        zero_values = tuple(0.0 for _ in range(n_mfcc))

        return (
            zero_values,
            zero_values,
        )

    # --------------------------------------------------------------
    # MEL SPECTROGRAM
    # --------------------------------------------------------------

    mel_spectrogram = librosa.feature.melspectrogram(
        S=power_spectrogram,
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin_hz,
        fmax=fmax_hz,
    )

    mel_spectrogram = np.asarray(
        mel_spectrogram,
        dtype=np.float32,
    )

    # --------------------------------------------------------------
    # LOG MEL
    # --------------------------------------------------------------

    mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
    )

    # --------------------------------------------------------------
    # MFCC
    # --------------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        S=mel_db,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )

    means = np.mean(
        mfcc,
        axis=1,
        dtype=np.float64,
    )

    stds = np.std(
        mfcc,
        axis=1,
        dtype=np.float64,
    )

    mean_tuple = tuple((float(value) if np.isfinite(value) else 0.0) for value in means)

    std_tuple = tuple((float(value) if np.isfinite(value) else 0.0) for value in stds)

    return (
        mean_tuple,
        std_tuple,
    )


# ======================================================================
# LOG SPECTROGRAM
# ======================================================================


def calculate_log_spectrogram(
    signal: FloatArray,
    *,
    sample_rate: int = 48_000,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[
    FloatSpectrogram,
    FloatArray,
    FloatArray,
]:
    """
    Produce a log-power spectrogram for dashboard visualization.

    Returns
    -------
    spectrogram_db
        Frequency bins × time frames.

    frequencies_hz
        Frequency axis in Hz.

    times_s
        Frame-start time axis in seconds.
    """

    (
        _,
        _,
        power,
    ) = calculate_stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    maximum_power = float(np.max(power))

    if not np.isfinite(maximum_power) or maximum_power <= EPSILON:
        db = np.zeros_like(
            power,
            dtype=np.float32,
        )

    else:
        db = np.asarray(
            librosa.power_to_db(
                power,
                ref=np.max,
            ),
            dtype=np.float32,
        )

    frequencies = np.asarray(
        librosa.fft_frequencies(
            sr=sample_rate,
            n_fft=n_fft,
        ),
        dtype=np.float32,
    )

    frame_count = int(power.shape[1])

    times = np.asarray(
        np.arange(
            frame_count,
            dtype=np.float32,
        )
        * np.float32(hop_length / sample_rate),
        dtype=np.float32,
    )

    return (
        np.ascontiguousarray(
            db,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            frequencies,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            times,
            dtype=np.float32,
        ),
    )


# ======================================================================
# COMPLETE FEATURE EXTRACTION
# ======================================================================


def extract_acoustic_features(
    audio: PreprocessedAudio,
    *,
    noise_rms: float | None = None,
    config: FeatureConfig | None = None,
) -> AcousticFeatures:
    """
    Extract the complete event-level acoustic feature vector.

    Parameters
    ----------
    audio
        PreprocessedAudio produced by preprocess_event_audio().

    noise_rms
        Background-noise RMS estimated independently from the
        pre-trigger event region.

    config
        Feature extraction configuration.

    Returns
    -------
    AcousticFeatures
        Complete event-level feature vector.
    """

    cfg = config if config is not None else FeatureConfig(sample_rate=audio.sample_rate)

    # ==================================================================
    # SAMPLE-RATE CONSISTENCY
    # ==================================================================

    if cfg.sample_rate != audio.sample_rate:
        raise ValueError(
            ("FeatureConfig sample_rate does not match PreprocessedAudio sample_rate.")
        )

    amplitude_signal = np.asarray(
        audio.amplitude_signal,
        dtype=np.float32,
    )

    analysis_signal = np.asarray(
        audio.analysis_signal,
        dtype=np.float32,
    )

    _validate_signal(amplitude_signal)

    _validate_signal(analysis_signal)

    if amplitude_signal.size != analysis_signal.size:
        raise ValueError(
            (
                "amplitude_signal and analysis_signal "
                "must contain the same number of samples."
            )
        )

    # ==================================================================
    # EVENT DURATION
    # ==================================================================

    duration_s = float(amplitude_signal.size / cfg.sample_rate)

    # ==================================================================
    # AMPLITUDE FEATURES
    # ==================================================================

    rms = calculate_rms(amplitude_signal)

    peak = calculate_peak_amplitude(amplitude_signal)

    crest_factor = calculate_crest_factor(amplitude_signal)

    snr_db = calculate_snr_db(
        rms,
        noise_rms,
    )

    # ==================================================================
    # SPECTRAL REPRESENTATION
    # ==================================================================

    (
        _,
        magnitude,
        power,
    ) = calculate_stft(
        analysis_signal,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

    # --------------------------------------------------------------
    # The same padding policy used by calculate_stft() is needed for
    # direct time-domain frame features such as ZCR.
    # --------------------------------------------------------------

    working_signal = _ensure_spectral_length(
        analysis_signal,
        cfg.n_fft,
    )

    # ==================================================================
    # ZERO-CROSSING RATE
    # ==================================================================

    zcr_frames = librosa.feature.zero_crossing_rate(
        y=working_signal,
        frame_length=cfg.n_fft,
        hop_length=cfg.hop_length,
        center=False,
    )

    # Ratio features from quiet pre/post-trigger frames must not count as
    # much as the call itself. Broadband background can otherwise dominate
    # centroid/ZCR and make every short chirp look equally bird/insect-like.
    frame_energy = np.sum(power, axis=0, dtype=np.float64)
    energy_total = float(np.sum(frame_energy))

    def event_mean(values: np.ndarray) -> float:
        frames = np.asarray(values, dtype=np.float64).reshape(-1)
        if energy_total <= EPSILON:
            return 0.0
        return float(np.average(frames, weights=frame_energy))

    zcr = max(
        0.0,
        event_mean(zcr_frames),
    )

    # ==================================================================
    # DOMINANT FREQUENCY
    # ==================================================================

    dominant_frequency = calculate_dominant_frequency(
        power,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
    )

    # ==================================================================
    # SILENCE-AWARE SPECTRAL FEATURES
    # ==================================================================

    spectral_energy = float(np.max(power))

    if not np.isfinite(spectral_energy) or spectral_energy <= EPSILON:
        spectral_centroid = 0.0

        spectral_bandwidth = 0.0

        spectral_rolloff = 0.0

        spectral_flatness = 0.0

        spectral_flux = 0.0

    else:
        # ==============================================================
        # SPECTRAL CENTROID
        # ==============================================================

        centroid_frames = librosa.feature.spectral_centroid(
            S=magnitude,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
        )

        spectral_centroid = max(
            0.0,
            event_mean(centroid_frames),
        )

        # ==============================================================
        # SPECTRAL BANDWIDTH
        # ==============================================================

        bandwidth_frames = librosa.feature.spectral_bandwidth(
            S=magnitude,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            centroid=centroid_frames,
        )

        spectral_bandwidth = max(
            0.0,
            event_mean(bandwidth_frames),
        )

        # ==============================================================
        # SPECTRAL ROLLOFF
        # ==============================================================

        rolloff_frames = librosa.feature.spectral_rolloff(
            S=magnitude,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            roll_percent=cfg.roll_percent,
        )

        spectral_rolloff = max(
            0.0,
            event_mean(rolloff_frames),
        )

        # ==============================================================
        # SPECTRAL FLATNESS
        # ==============================================================

        flatness_frames = librosa.feature.spectral_flatness(
            S=magnitude,
            power=2.0,
        )

        spectral_flatness = float(
            np.clip(
                event_mean(flatness_frames),
                0.0,
                1.0,
            )
        )

        # ==============================================================
        # SPECTRAL FLUX
        # ==============================================================

        spectral_flux = calculate_spectral_flux(magnitude)

    # ==================================================================
    # MFCC
    # ==================================================================

    (
        mfcc_mean,
        mfcc_std,
    ) = calculate_mfcc_statistics(
        power,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mfcc=cfg.n_mfcc,
        n_mels=cfg.n_mels,
        fmin_hz=cfg.mfcc_fmin_hz,
        fmax_hz=cfg.mfcc_fmax_hz,
    )

    # ==================================================================
    # RESULT
    # ==================================================================

    return AcousticFeatures(
        duration_s=duration_s,
        rms=float(rms),
        peak_amplitude=float(peak),
        crest_factor=float(crest_factor),
        zero_crossing_rate=float(zcr),
        dominant_frequency_hz=float(dominant_frequency),
        spectral_centroid_hz=float(spectral_centroid),
        spectral_bandwidth_hz=float(spectral_bandwidth),
        spectral_rolloff_hz=float(spectral_rolloff),
        spectral_flatness=float(spectral_flatness),
        spectral_flux=float(spectral_flux),
        snr_db=(None if snr_db is None else float(snr_db)),
        mfcc_mean=tuple(mfcc_mean),
        mfcc_std=tuple(mfcc_std),
    )
