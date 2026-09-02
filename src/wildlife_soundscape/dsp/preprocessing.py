"""
Audio preprocessing utilities.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Prepare event audio for DSP feature extraction and classification
without modifying the original captured PCM data.

Pipeline
--------
PCM16
  ↓
float32 conversion
  ↓
DC offset removal
  ↓
optional Butterworth band-pass filtering
  ↓
optional peak normalization for model input

Signal representations
----------------------
raw_float
    PCM16 converted to float32.

amplitude_signal
    DC-corrected signal preserving relative amplitude.

analysis_signal
    Filtered signal used for spectral / temporal DSP analysis.

model_signal
    Optional normalized signal intended for ML/model input.

Important
---------
Amplitude-dependent measurements such as:

    RMS
    peak amplitude
    crest factor
    SNR

must be calculated BEFORE peak normalization.

The original PCM event data must always remain preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from scipy.signal import (
    butter,
    sosfiltfilt,
)


# ======================================================================
# TYPE ALIASES
# ======================================================================


FloatArray = NDArray[np.float32]

PCM16Array = NDArray[np.int16]


# ======================================================================
# CONSTANTS
# ======================================================================


PCM16_SCALE: Final[float] = (
    32768.0
)

EPSILON: Final[float] = (
    1e-12
)


# ======================================================================
# CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PreprocessingConfig:
    """
    Configuration for event-audio preprocessing.

    Parameters
    ----------
    sample_rate
        Audio sampling frequency in Hz.

    remove_dc
        Remove mean/DC offset before further processing.

    bandpass_enabled
        Apply zero-phase Butterworth band-pass filtering.

    low_cutoff_hz
        Lower band-pass cutoff frequency.

    high_cutoff_hz
        Upper band-pass cutoff frequency.

    filter_order
        Butterworth filter order.

    normalize_for_model
        Produce a peak-normalized model-input waveform.

    model_target_peak
        Desired absolute peak amplitude of the model waveform.

        Example:
            0.98

        The normalized model waveform must NOT be used for absolute
        amplitude measurements such as RMS or SNR.
    """

    # ------------------------------------------------------------------
    # SAMPLE RATE
    # ------------------------------------------------------------------

    sample_rate: int = (
        48_000
    )

    # ------------------------------------------------------------------
    # DC REMOVAL
    # ------------------------------------------------------------------

    remove_dc: bool = (
        True
    )

    # ------------------------------------------------------------------
    # BAND-PASS FILTER
    # ------------------------------------------------------------------

    bandpass_enabled: bool = (
        True
    )

    # Broad wildlife-analysis range.
    #
    # These are engineering defaults, not biological class boundaries.

    low_cutoff_hz: float = (
        100.0
    )

    high_cutoff_hz: float = (
        16_000.0
    )

    filter_order: int = (
        4
    )

    # ------------------------------------------------------------------
    # MODEL INPUT NORMALIZATION
    # ------------------------------------------------------------------

    normalize_for_model: bool = (
        True
    )

    model_target_peak: float = (
        0.98
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate preprocessing configuration.
        """

        # --------------------------------------------------------------
        # SAMPLE RATE
        # --------------------------------------------------------------

        if (
            self.sample_rate
            <= 0
        ):

            raise ValueError(
                (
                    "sample_rate must "
                    "be greater than 0."
                )
            )

        # --------------------------------------------------------------
        # FILTER ORDER
        # --------------------------------------------------------------

        if (
            self.filter_order
            <= 0
        ):

            raise ValueError(
                (
                    "filter_order must "
                    "be greater than 0."
                )
            )

        # --------------------------------------------------------------
        # BAND-PASS RANGE
        # --------------------------------------------------------------

        nyquist = (
            self.sample_rate
            / 2.0
        )

        if self.bandpass_enabled:

            if (
                self.low_cutoff_hz
                <= 0.0
            ):

                raise ValueError(
                    (
                        "low_cutoff_hz must "
                        "be greater than 0."
                    )
                )

            if (
                self.high_cutoff_hz
                <= self.low_cutoff_hz
            ):

                raise ValueError(
                    (
                        "high_cutoff_hz must be greater "
                        "than low_cutoff_hz."
                    )
                )

            if (
                self.high_cutoff_hz
                >= nyquist
            ):

                raise ValueError(
                    (
                        "high_cutoff_hz must remain below "
                        f"Nyquist ({nyquist:.1f} Hz)."
                    )
                )

        # --------------------------------------------------------------
        # MODEL TARGET PEAK
        # --------------------------------------------------------------

        if not (
            0.0
            < self.model_target_peak
            <= 1.0
        ):

            raise ValueError(
                (
                    "model_target_peak must "
                    "be in the interval (0, 1]."
                )
            )


# ======================================================================
# OUTPUT MODEL
# ======================================================================


@dataclass(
    slots=True,
)
class PreprocessedAudio:
    """
    Audio representations produced by preprocessing.

    raw_float
        Original PCM converted to float32 approximately in [-1, 1).

    amplitude_signal
        DC-corrected signal retaining real relative amplitude.

        Intended for:
            RMS
            absolute peak
            crest factor
            SNR
            channel-quality estimation

    analysis_signal
        Filtered analysis waveform.

        Intended for:
            FFT
            STFT
            MFCC
            spectral centroid
            spectral bandwidth
            spectral rolloff
            spectral flatness
            spectral flux
            zero-crossing rate

    model_signal
        Optional peak-normalized analysis waveform.

        Intended for:
            trained/pretrained classifier input

    sample_rate
        Sampling frequency in Hz.

    peak_before_normalization
        Absolute peak of analysis_signal before model normalization.
    """

    raw_float: FloatArray

    amplitude_signal: FloatArray

    analysis_signal: FloatArray

    model_signal: FloatArray

    sample_rate: int

    peak_before_normalization: float


# ======================================================================
# GENERIC AUDIO VALIDATION
# ======================================================================


def _validate_1d_audio(
    audio: np.ndarray,
) -> None:
    """
    Ensure the supplied array represents one non-empty mono channel.
    """

    if not isinstance(
        audio,
        np.ndarray,
    ):

        raise TypeError(
            (
                "audio must be a "
                "NumPy ndarray."
            )
        )

    if (
        audio.ndim
        != 1
    ):

        raise ValueError(
            (
                "Expected mono 1-D audio, "
                f"got shape {audio.shape}."
            )
        )

    if (
        audio.size
        == 0
    ):

        raise ValueError(
            (
                "Audio array "
                "cannot be empty."
            )
        )


# ======================================================================
# FLOAT AUDIO VALIDATION
# ======================================================================


def _validate_float_audio(
    audio: np.ndarray,
) -> None:
    """
    Validate a floating-point mono audio array.
    """

    _validate_1d_audio(
        audio
    )

    if not np.issubdtype(
        audio.dtype,
        np.floating,
    ):

        raise TypeError(
            (
                "Expected floating-point "
                f"audio, received {audio.dtype}."
            )
        )

    if not np.all(
        np.isfinite(
            audio
        )
    ):

        raise ValueError(
            (
                "Audio contains NaN "
                "or infinite samples."
            )
        )


# ======================================================================
# PCM16 -> FLOAT32
# ======================================================================


def pcm16_to_float32(
    pcm: PCM16Array,
) -> FloatArray:
    """
    Convert signed PCM16 samples to float32.

    Mapping
    -------
    -32768 -> -1.0
     32767 -> approximately +0.999969

    No normalization is performed.

    Therefore the amplitude relationship of the captured recording is
    retained.
    """

    _validate_1d_audio(
        pcm
    )

    if (
        pcm.dtype
        != np.int16
    ):

        raise TypeError(
            (
                "Expected int16 PCM, "
                f"received {pcm.dtype}."
            )
        )

    result = (
        pcm.astype(
            np.float32,
            copy=False,
        )
        / np.float32(
            PCM16_SCALE
        )
    )

    return np.asarray(
        result,
        dtype=np.float32,
    )


# ======================================================================
# DC OFFSET REMOVAL
# ======================================================================


def remove_dc_offset(
    audio: FloatArray,
) -> FloatArray:
    """
    Remove the DC component from a floating-point signal.

    Formula
    -------
    x_dc[n] = x[n] - mean(x)

    The mean uses a float64 accumulator for numerical stability while
    the returned waveform remains float32.
    """

    _validate_float_audio(
        audio
    )

    mean_value = float(
        np.mean(
            audio,
            dtype=np.float64,
        )
    )

    corrected = (
        audio.astype(
            np.float32,
            copy=False,
        )
        - np.float32(
            mean_value
        )
    )

    return np.asarray(
        corrected,
        dtype=np.float32,
    )


# ======================================================================
# BAND-PASS DESIGN
# ======================================================================


def _design_bandpass_sos(
    *,
    sample_rate: int,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    order: int,
) -> NDArray[np.float64]:
    """
    Design a Butterworth band-pass filter using second-order sections.

    SOS representation is preferred over direct polynomial coefficients
    because it is numerically more stable for higher-order filters.
    """

    if (
        sample_rate
        <= 0
    ):

        raise ValueError(
            (
                "sample_rate must "
                "be greater than 0."
            )
        )

    if (
        order
        <= 0
    ):

        raise ValueError(
            (
                "Filter order must "
                "be greater than 0."
            )
        )

    nyquist = (
        sample_rate
        / 2.0
    )

    if not (
        0.0
        < low_cutoff_hz
        < high_cutoff_hz
        < nyquist
    ):

        raise ValueError(
            (
                "Band-pass cutoffs must satisfy "
                "0 < low < high < Nyquist."
            )
        )

    sos = butter(
        N=order,
        Wn=[
            low_cutoff_hz,
            high_cutoff_hz,
        ],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    return np.asarray(
        sos,
        dtype=np.float64,
    )


# ======================================================================
# SAFE FILTER PADDING
# ======================================================================


def _safe_sos_padlen(
    sos: NDArray[np.float64],
    signal_length: int,
) -> int:
    """
    Calculate a safe padding length for scipy.signal.sosfiltfilt.

    scipy's default pad length depends on the number of second-order
    sections.

    Very short acoustic events may contain fewer samples than the
    default padding requirement.

    The padding value is therefore bounded by:

        signal_length - 1
    """

    if (
        signal_length
        <= 1
    ):

        return 0

    zeros_at_origin = int(
        np.sum(
            sos[
                :,
                2,
            ]
            == 0.0
        )
    )

    poles_at_origin = int(
        np.sum(
            sos[
                :,
                5,
            ]
            == 0.0
        )
    )

    estimated_default = (
        3
        * (
            2
            * len(
                sos
            )
            + 1
            - min(
                zeros_at_origin,
                poles_at_origin,
            )
        )
    )

    return max(
        0,
        min(
            estimated_default,
            signal_length - 1,
        ),
    )


# ======================================================================
# BAND-PASS FILTER
# ======================================================================


def apply_bandpass_filter(
    audio: FloatArray,
    *,
    sample_rate: int,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    order: int = 4,
) -> FloatArray:
    """
    Apply zero-phase Butterworth band-pass filtering.

    Filtering is performed forward and backward with sosfiltfilt, which
    avoids introducing a net phase delay into the analysis signal.

    Extremely short signals are returned unchanged rather than causing
    the complete event pipeline to fail.
    """

    _validate_float_audio(
        audio
    )

    # --------------------------------------------------------------
    # FILTERING VERY SHORT AUDIO IS NOT MEANINGFUL
    # --------------------------------------------------------------

    if (
        audio.size
        < 8
    ):

        return np.asarray(
            audio,
            dtype=np.float32,
        ).copy()

    # --------------------------------------------------------------
    # FILTER DESIGN
    # --------------------------------------------------------------

    sos = (
        _design_bandpass_sos(
            sample_rate=
                sample_rate,

            low_cutoff_hz=
                low_cutoff_hz,

            high_cutoff_hz=
                high_cutoff_hz,

            order=
                order,
        )
    )

    # --------------------------------------------------------------
    # SAFE PADDING
    # --------------------------------------------------------------

    padlen = (
        _safe_sos_padlen(
            sos,
            int(
                audio.size
            ),
        )
    )

    # --------------------------------------------------------------
    # ZERO-PHASE FILTER
    # --------------------------------------------------------------

    try:

        filtered = (
            sosfiltfilt(
                sos,

                audio.astype(
                    np.float64,
                    copy=False,
                ),

                padlen=
                    padlen,
            )
        )

    except ValueError:

        # ----------------------------------------------------------
        # Pathological / extremely short signals should not crash an
        # otherwise valid event.
        # ----------------------------------------------------------

        return np.asarray(
            audio,
            dtype=np.float32,
        ).copy()

    if not np.all(
        np.isfinite(
            filtered
        )
    ):

        raise ValueError(
            (
                "Band-pass filtering "
                "produced non-finite samples."
            )
        )

    return np.asarray(
        filtered,
        dtype=np.float32,
    )


# ======================================================================
# PEAK NORMALIZATION
# ======================================================================


def peak_normalize(
    audio: FloatArray,
    *,
    target_peak: float = 0.98,
) -> FloatArray:
    """
    Peak-normalize floating-point audio.

    Parameters
    ----------
    audio
        Analysis waveform.

    target_peak
        Desired output absolute peak in the interval (0, 1].

    Important
    ---------
    Do NOT use the normalized signal for:

        RMS
        absolute peak amplitude
        SNR

    because normalization intentionally destroys absolute amplitude
    information.
    """

    _validate_float_audio(
        audio
    )

    if not (
        0.0
        < target_peak
        <= 1.0
    ):

        raise ValueError(
            (
                "target_peak must "
                "be in (0, 1]."
            )
        )

    peak = float(
        np.max(
            np.abs(
                audio
            )
        )
    )

    # --------------------------------------------------------------
    # SILENCE / NEAR-SILENCE
    # --------------------------------------------------------------

    if (
        not np.isfinite(
            peak
        )
        or peak
        <= EPSILON
    ):

        return np.zeros_like(
            audio,
            dtype=np.float32,
        )

    # --------------------------------------------------------------
    # NORMALIZATION GAIN
    # --------------------------------------------------------------

    gain = (
        float(
            target_peak
        )
        / peak
    )

    normalized = (
        audio.astype(
            np.float32,
            copy=False,
        )
        * np.float32(
            gain
        )
    )

    # --------------------------------------------------------------
    # NUMERICAL SAFETY
    # --------------------------------------------------------------

    normalized = np.clip(
        normalized,
        -1.0,
        1.0,
    )

    return np.asarray(
        normalized,
        dtype=np.float32,
    )


# ======================================================================
# COMPLETE EVENT PREPROCESSING
# ======================================================================


def preprocess_event_audio(
    pcm: PCM16Array,
    config: PreprocessingConfig | None = None,
) -> PreprocessedAudio:
    """
    Run the complete preprocessing pipeline for one mono event.

    Parameters
    ----------
    pcm
        Original signed PCM16 samples.

    config
        Optional preprocessing configuration.

    Returns
    -------
    PreprocessedAudio
        Independent representations for:

            raw audio
            amplitude analysis
            spectral analysis
            model input

    Notes
    -----
    The supplied PCM array is never modified.
    """

    cfg = (
        config
        if config is not None
        else PreprocessingConfig()
    )

    # ==================================================================
    # 1. PCM16 -> FLOAT32
    # ==================================================================

    raw_float = (
        pcm16_to_float32(
            pcm
        )
    )

    # Keep representation independent from caller memory.
    raw_float = np.asarray(
        raw_float,
        dtype=np.float32,
    ).copy()

    # ==================================================================
    # 2. DC OFFSET REMOVAL
    # ==================================================================

    if cfg.remove_dc:

        amplitude_signal = (
            remove_dc_offset(
                raw_float
            )
        )

    else:

        amplitude_signal = (
            raw_float.copy()
        )

    # ==================================================================
    # 3. OPTIONAL BAND-PASS FILTER
    # ==================================================================

    if cfg.bandpass_enabled:

        analysis_signal = (
            apply_bandpass_filter(
                amplitude_signal,

                sample_rate=
                    cfg.sample_rate,

                low_cutoff_hz=
                    cfg.low_cutoff_hz,

                high_cutoff_hz=
                    cfg.high_cutoff_hz,

                order=
                    cfg.filter_order,
            )
        )

    else:

        analysis_signal = (
            amplitude_signal.copy()
        )

    # ==================================================================
    # 4. PRE-NORMALIZATION PEAK
    # ==================================================================

    peak_before_normalization = float(
        np.max(
            np.abs(
                analysis_signal
            )
        )
    )

    if not np.isfinite(
        peak_before_normalization
    ):

        raise ValueError(
            (
                "analysis_signal contains "
                "non-finite amplitude values."
            )
        )

    # ==================================================================
    # 5. MODEL INPUT
    # ==================================================================

    if cfg.normalize_for_model:

        model_signal = (
            peak_normalize(
                analysis_signal,

                target_peak=
                    cfg.model_target_peak,
            )
        )

    else:

        model_signal = (
            analysis_signal.copy()
        )

    # ==================================================================
    # 6. CONTIGUOUS OUTPUT BUFFERS
    # ==================================================================

    raw_float = np.ascontiguousarray(
        raw_float,
        dtype=np.float32,
    )

    amplitude_signal = np.ascontiguousarray(
        amplitude_signal,
        dtype=np.float32,
    )

    analysis_signal = np.ascontiguousarray(
        analysis_signal,
        dtype=np.float32,
    )

    model_signal = np.ascontiguousarray(
        model_signal,
        dtype=np.float32,
    )

    # ==================================================================
    # RESULT
    # ==================================================================

    return PreprocessedAudio(
        raw_float=
            raw_float,

        amplitude_signal=
            amplitude_signal,

        analysis_signal=
            analysis_signal,

        model_signal=
            model_signal,

        sample_rate=
            cfg.sample_rate,

        peak_before_normalization=
            peak_before_normalization,
    )