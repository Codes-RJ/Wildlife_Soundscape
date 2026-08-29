from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    samples: np.ndarray,
    *,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass for localization windows."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    nyquist = sample_rate / 2.0
    low = max(1.0, float(low_hz))
    high = min(float(high_hz), nyquist * 0.98)
    if not (0.0 < low < high < nyquist):
        raise ValueError(
            f"invalid bandpass {low_hz}-{high_hz} Hz for sample rate {sample_rate}"
        )
    order = max(1, int(order))
    sos = butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    # sosfiltfilt requires a modest minimum length; localization windows are
    # normally thousands of samples. Return unfiltered data only for tiny tests.
    if x.size < max(32, order * 12):
        return x.copy()
    return sosfiltfilt(sos, x)
