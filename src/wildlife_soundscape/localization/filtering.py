from __future__ import annotations

import math

import numpy as np

from scipy.signal import (
    butter,
    sosfiltfilt,
)


# ======================================================================
# CONSTANTS
# ======================================================================


MIN_FILTER_SIGNAL_SAMPLES = 32


# ======================================================================
# INPUT VALIDATION
# ======================================================================


def _validate_audio(
    samples: np.ndarray,
) -> np.ndarray:
    """
    Validate one mono localization waveform.

    Localization operates on one physical microphone channel at a time.

    Multidimensional input is therefore rejected rather than flattened,
    because flattening could create an artificial waveform and corrupt
    GCC-PHAT/TDOA timing.
    """

    x = np.asarray(
        samples,
        dtype=np.float64,
    )

    if x.ndim != 1:
        raise ValueError(
            (f"Localization filtering expects mono 1-D audio, got shape {x.shape}.")
        )

    if x.size == 0:
        raise ValueError("Localization audio cannot be empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError(("Localization audio contains NaN or infinite samples."))

    return np.ascontiguousarray(
        x,
        dtype=np.float64,
    )


# ======================================================================
# BAND-PASS FILTER
# ======================================================================


def bandpass_filter(
    samples: np.ndarray,
    *,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth band-pass filtering to one localization
    waveform.

    Parameters
    ----------
    samples
        One mono audio channel.

    sample_rate
        Sampling frequency in Hz.

    low_hz
        Lower cutoff frequency.

    high_hz
        Upper cutoff frequency.

    order
        Butterworth filter order.

    Returns
    -------
    numpy.ndarray
        Filtered float64 mono waveform.

    Notes
    -----
    Zero-phase filtering is appropriate here because localization works
    on already-buffered event windows rather than a causal real-time
    embedded signal path.

    Very short waveforms are returned unchanged because forward/backward
    filtering is not meaningful or numerically reliable for such data.
    """

    # ==================================================================
    # AUDIO
    # ==================================================================

    x = _validate_audio(samples)

    if x.size == 0:
        return x

    # ==================================================================
    # SAMPLE RATE
    # ==================================================================

    try:
        sample_rate = float(sample_rate)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(("sample_rate must be a finite numeric value.")) from exc

    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(("sample_rate must be finite and greater than 0."))

    # ==================================================================
    # CUTOFF FREQUENCIES
    # ==================================================================

    try:
        low_hz = float(low_hz)

        high_hz = float(high_hz)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(("Band-pass cutoff frequencies must be numeric.")) from exc

    if not math.isfinite(low_hz) or not math.isfinite(high_hz):
        raise ValueError(("Band-pass cutoff frequencies must be finite."))

    nyquist = sample_rate / 2.0

    if not (0.0 < low_hz < high_hz < nyquist):
        raise ValueError(
            (
                "Band-pass frequencies must satisfy "
                "0 < low_hz < high_hz < Nyquist. "
                f"Received low={low_hz}, "
                f"high={high_hz}, "
                f"sample_rate={sample_rate}."
            )
        )

    # ==================================================================
    # FILTER ORDER
    # ==================================================================

    if isinstance(
        order,
        bool,
    ) or not isinstance(
        order,
        int,
    ):
        raise TypeError(("order must be an integer."))

    if order <= 0:
        raise ValueError(("order must be greater than 0."))

    # ==================================================================
    # SHORT WINDOW
    # ==================================================================
    #
    # Localization normally operates on windows containing thousands of
    # samples.
    #
    # Returning very short inputs unchanged is preferable to producing
    # unstable edge-dominated filtered data.
    # ==================================================================

    minimum_samples = max(
        MIN_FILTER_SIGNAL_SAMPLES,
        order * 12,
    )

    if x.size < minimum_samples:
        return x.copy()

    # ==================================================================
    # FILTER DESIGN
    # ==================================================================

    sos = butter(
        N=order,
        Wn=[
            low_hz,
            high_hz,
        ],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    # ==================================================================
    # ZERO-PHASE FILTERING
    # ==================================================================

    try:
        filtered = sosfiltfilt(
            sos,
            x,
        )

    except ValueError as exc:
        raise ValueError(
            ("Localization band-pass filtering failed for the supplied waveform.")
        ) from exc

    # ==================================================================
    # OUTPUT VALIDATION
    # ==================================================================

    filtered = np.asarray(
        filtered,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(filtered)):
        raise ValueError(("Localization filtering produced non-finite samples."))

    return np.ascontiguousarray(
        filtered,
        dtype=np.float64,
    )
