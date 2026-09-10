"""
Continuous ecoacoustic indices for soundscape analysis.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Calculate continuous soundscape metrics from multi-channel audio recordings
to assess habitat dynamics, biophonic activity, and anthropogenic noise
pressure independently from discrete event detection.

Implemented Ecoacoustic Indices
-------------------------------
1. ACI (Acoustic Complexity Index) - Pieretti, Farina, & Morri (2011)
   Measures sound intensity variability across consecutive time frames
   within frequency bins.

2. NDSI (Normalized Difference Soundscape Index) - Kasten et al. (2012)
   Measures the normalized balance between biophony and anthrophony frequency
   power bands.
   Default bands: Anthrophony [1-2 kHz], Biophony [2-8 kHz] (fully configurable).

3. Acoustic Entropy (H) - Sueur et al. (2008)
   Measures soundscape diversity combining normalized temporal entropy (Ht)
   and normalized spectral entropy (Hf): H = Ht * Hf.

4. Bioacoustic Index (BI) - Boelman et al. (2007)
   Measures the area under the dB power spectrum curve in the avian/biophonic
   band relative to baseline noise.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

EPSILON: Final[float] = 1e-12


# ======================================================================
# CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class SoundscapeIndicesConfig:
    """
    Configuration parameters for ecoacoustic index calculations.
    """

    sample_rate: int = 48000
    n_fft: int = 1024
    hop_length: int = 512
    window: str = "hann"

    # ACI parameters
    aci_freq_min_hz: float = 0.0
    aci_freq_max_hz: float = 24000.0

    # NDSI frequency bands (Hz)
    ndsi_anthrophony_min_hz: float = 1000.0
    ndsi_anthrophony_max_hz: float = 2000.0
    ndsi_biophony_min_hz: float = 2000.0
    ndsi_biophony_max_hz: float = 8000.0

    # Bioacoustic Index frequency band (Hz)
    bi_freq_min_hz: float = 2000.0
    bi_freq_max_hz: float = 8000.0
    bi_min_db_threshold: float = -50.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.n_fft <= 0 or (self.n_fft & (self.n_fft - 1)) != 0:
            raise ValueError("n_fft must be a positive power of 2.")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive.")

        nyquist = self.sample_rate / 2.0
        if not (0.0 <= self.aci_freq_min_hz < self.aci_freq_max_hz <= nyquist):
            raise ValueError(
                f"Invalid ACI band [{self.aci_freq_min_hz}, {self.aci_freq_max_hz}]."
            )

        if not (
            0.0
            <= self.ndsi_anthrophony_min_hz
            < self.ndsi_anthrophony_max_hz
            <= nyquist
        ):
            raise ValueError("Invalid NDSI anthrophony band.")
        if not (
            0.0 <= self.ndsi_biophony_min_hz < self.ndsi_biophony_max_hz <= nyquist
        ):
            raise ValueError("Invalid NDSI biophony band.")

        if not (0.0 <= self.bi_freq_min_hz < self.bi_freq_max_hz <= nyquist):
            raise ValueError("Invalid BI band.")


# ======================================================================
# RESULT DATACLASS
# ======================================================================


@dataclass(frozen=True, slots=True)
class SoundscapeIndicesResult:
    """
    Immutable container of computed soundscape metrics with parameter metadata.
    """

    aci: float
    ndsi: float
    acoustic_entropy: float
    temporal_entropy: float
    spectral_entropy: float
    bioacoustic_index: float
    anthrophony_power: float
    biophony_power: float
    parameters: SoundscapeIndicesConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "aci": self.aci,
            "ndsi": self.ndsi,
            "acoustic_entropy": self.acoustic_entropy,
            "temporal_entropy": self.temporal_entropy,
            "spectral_entropy": self.spectral_entropy,
            "bioacoustic_index": self.bioacoustic_index,
            "anthrophony_power": self.anthrophony_power,
            "biophony_power": self.biophony_power,
            "parameters": asdict(self.parameters),
        }


# ======================================================================
# SPECTROGRAM COMPUTATION
# ======================================================================


def _compute_magnitude_spectrogram(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    window_type: str = "hann",
) -> np.ndarray:
    """
    Compute 2-D magnitude spectrogram (freq_bins x time_frames) using pure NumPy.
    """
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty 1-D array.")

    # Apply windowing
    if audio.size < n_fft:
        pad_width = n_fft - audio.size
        audio = np.pad(audio, (0, pad_width), mode="constant")

    # Sliding window framing
    num_frames = 1 + (audio.size - n_fft) // hop_length
    if num_frames <= 0:
        num_frames = 1
        frames = audio[:n_fft][np.newaxis, :]
    else:
        shape = (num_frames, n_fft)
        strides = (audio.strides[0] * hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

    if window_type == "hann":
        window = np.hanning(n_fft).astype(audio.dtype)
    else:
        window = np.ones(n_fft, dtype=audio.dtype)

    windowed_frames = frames * window
    stft = np.fft.rfft(windowed_frames, n=n_fft, axis=1)  # (frames, n_fft // 2 + 1)
    mag_spec = np.abs(stft).T.astype(np.float32)  # (freq_bins, frames)
    return mag_spec


# ======================================================================
# INDIVIDUAL ECOACOUSTIC METRIC CALCULATIONS
# ======================================================================


def calculate_aci(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> float:
    """
    Calculate the Acoustic Complexity Index (ACI).
    """
    if audio.size == 0:
        return 0.0

    nyquist = sample_rate / 2.0
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    mag = _compute_magnitude_spectrogram(audio, n_fft, hop_length)
    freqs = np.linspace(0, nyquist, mag.shape[0])
    bin_mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(bin_mask) or mag.shape[1] < 2:
        return 0.0

    sub_mag = mag[bin_mask, :]
    # Absolute difference between adjacent time frames
    diffs = np.abs(np.diff(sub_mag, axis=1))  # (K, T-1)
    sum_diffs = np.sum(diffs, axis=1)  # (K,)
    sum_intensities = np.sum(sub_mag, axis=1)  # (K,)

    # Stable per-bin ACI
    valid_bins = sum_intensities > EPSILON
    if not np.any(valid_bins):
        return 0.0

    aci_per_bin = np.zeros_like(sum_diffs)
    aci_per_bin[valid_bins] = sum_diffs[valid_bins] / (
        sum_intensities[valid_bins] + EPSILON
    )

    return float(np.sum(aci_per_bin))


def calculate_ndsi(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    anthro_min: float = 1000.0,
    anthro_max: float = 2000.0,
    bio_min: float = 2000.0,
    bio_max: float = 8000.0,
) -> tuple[float, float, float]:
    """
    Calculate Normalized Difference Soundscape Index (NDSI).

    Returns:
        (ndsi, anthrophony_power, biophony_power)
    """
    if audio.size == 0:
        return 0.0, 0.0, 0.0

    nyquist = sample_rate / 2.0
    mag = _compute_magnitude_spectrogram(audio, n_fft, hop_length)
    power_spec = np.mean(mag**2, axis=1)  # average power per frequency bin
    freqs = np.linspace(0, nyquist, len(power_spec))

    anthro_mask = (freqs >= anthro_min) & (freqs < anthro_max)
    bio_mask = (freqs >= bio_min) & (freqs <= bio_max)

    anthro_power = (
        float(np.sum(power_spec[anthro_mask])) if np.any(anthro_mask) else 0.0
    )
    bio_power = float(np.sum(power_spec[bio_mask])) if np.any(bio_mask) else 0.0

    total_power = bio_power + anthro_power
    if total_power < EPSILON:
        return 0.0, anthro_power, bio_power

    ndsi = (bio_power - anthro_power) / (total_power + EPSILON)
    ndsi = float(np.clip(ndsi, -1.0, 1.0))
    return ndsi, anthro_power, bio_power


def calculate_acoustic_entropy(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> tuple[float, float, float]:
    """
    Calculate Acoustic Entropy (H = Ht * Hf).

    Returns:
        (combined_h, temporal_entropy, spectral_entropy)
    """
    if audio.size == 0:
        return 0.0, 0.0, 0.0

    # 1. Temporal Entropy (Ht) over amplitude envelope
    env = np.abs(audio)
    env_sum = np.sum(env)
    if env_sum < EPSILON or env.size <= 1:
        ht = 0.0
    else:
        p_t = env / (env_sum + EPSILON)
        # Shannon entropy normalized by ln(N)
        valid_t = p_t > EPSILON
        ht = -float(np.sum(p_t[valid_t] * np.log(p_t[valid_t]))) / math.log(p_t.size)
        ht = float(np.clip(ht, 0.0, 1.0))

    # 2. Spectral Entropy (Hf) over mean power spectrum
    mag = _compute_magnitude_spectrogram(audio, n_fft, hop_length)
    power_mean = np.mean(mag**2, axis=1)
    power_sum = np.sum(power_mean)
    if power_sum < EPSILON or power_mean.size <= 1:
        hf = 0.0
    else:
        p_f = power_mean / (power_sum + EPSILON)
        valid_f = p_f > EPSILON
        hf = -float(np.sum(p_f[valid_f] * np.log(p_f[valid_f]))) / math.log(p_f.size)
        hf = float(np.clip(hf, 0.0, 1.0))

    h_combined = float(np.clip(ht * hf, 0.0, 1.0))
    return h_combined, ht, hf


def calculate_bioacoustic_index(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    fmin: float = 2000.0,
    fmax: float = 8000.0,
) -> float:
    """
    Calculate Bioacoustic Index (BI).
    """
    if audio.size == 0:
        return 0.0

    nyquist = sample_rate / 2.0
    mag = _compute_magnitude_spectrogram(audio, n_fft, hop_length)
    mean_power = np.mean(mag**2, axis=1)
    freqs = np.linspace(0, nyquist, len(mean_power))

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0

    band_power = mean_power[mask]
    # Convert to dB relative to maximum possible or baseline
    db_spec = 10.0 * np.log10(band_power + EPSILON)
    min_db = np.min(db_spec)
    # Area under the dB curve relative to minimum
    rel_db = db_spec - min_db
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    bi = float(np.sum(rel_db) * df / 1000.0)  # Normalized by kHz
    return max(0.0, bi)


# ======================================================================
# UNIFIED CONVENIENCE CALCULATION
# ======================================================================


def calculate_soundscape_indices(
    audio: np.ndarray,
    sample_rate: int,
    config: SoundscapeIndicesConfig | None = None,
) -> SoundscapeIndicesResult:
    """
    Compute all continuous soundscape indices over an audio signal.
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a numpy.ndarray.")
    if audio.ndim != 1:
        raise ValueError("audio must be a 1-D array.")

    cfg = (
        config
        if config is not None
        else SoundscapeIndicesConfig(sample_rate=sample_rate)
    )

    # Cast to float32
    audio_f32 = audio.astype(np.float32)

    aci = calculate_aci(
        audio_f32,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.aci_freq_min_hz,
        fmax=cfg.aci_freq_max_hz,
    )

    ndsi, anthro_p, bio_p = calculate_ndsi(
        audio_f32,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        anthro_min=cfg.ndsi_anthrophony_min_hz,
        anthro_max=cfg.ndsi_anthrophony_max_hz,
        bio_min=cfg.ndsi_biophony_min_hz,
        bio_max=cfg.ndsi_biophony_max_hz,
    )

    h_comb, ht, hf = calculate_acoustic_entropy(
        audio_f32,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

    bi = calculate_bioacoustic_index(
        audio_f32,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.bi_freq_min_hz,
        fmax=cfg.bi_freq_max_hz,
    )

    return SoundscapeIndicesResult(
        aci=aci,
        ndsi=ndsi,
        acoustic_entropy=h_comb,
        temporal_entropy=ht,
        spectral_entropy=hf,
        bioacoustic_index=bi,
        anthrophony_power=anthro_p,
        biophony_power=bio_p,
        parameters=cfg,
    )
