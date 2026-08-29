from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


@dataclass(frozen=True, slots=True)
class GCCPHATResult:
    delay_seconds: float
    delay_samples: float
    peak_value: float
    peak_ratio: float
    max_delay_samples: float
    valid: bool
    reason: str = ""


def _prepare(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return y
    # Remove DC; amplitude normalization is unnecessary for PHAT but this
    # stabilizes very large PCM values and keeps diagnostics interpretable.
    y = y - np.mean(y)
    return y


def gcc_phat(
    signal: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate: float,
    max_delay_seconds: float | None = None,
    interpolation: int = 8,
    epsilon: float = 1e-12,
    min_peak_ratio: float = 1.0,
) -> GCCPHATResult:
    """Estimate delay(signal relative to reference) using GCC-PHAT.

    Positive delay means ``signal`` arrives later than ``reference``.
    The search is physically constrained when ``max_delay_seconds`` is given.
    """
    x = _prepare(signal)
    y = _prepare(reference)
    if x.size < 8 or y.size < 8:
        return GCCPHATResult(0.0, 0.0, 0.0, 0.0, 0.0, False, "window too short")
    if not np.any(x) or not np.any(y):
        return GCCPHATResult(0.0, 0.0, 0.0, 0.0, 0.0, False, "zero-energy window")

    interpolation = max(1, int(interpolation))
    n_linear = x.size + y.size - 1
    n_fft = next_fast_len(n_linear)

    X = rfft(x, n=n_fft)
    Y = rfft(y, n=n_fft)
    cross = X * np.conj(Y)
    magnitude = np.abs(cross)
    cross /= np.maximum(magnitude, epsilon)

    n_corr = n_fft * interpolation
    corr_circular = irfft(cross, n=n_corr)

    # Rearrange circular correlation so lag zero is centered. The maximum
    # meaningful linear lag is limited by the original signal lengths.
    max_linear_lag = max(x.size, y.size) - 1
    max_linear_lag_i = max_linear_lag * interpolation
    corr = np.concatenate((corr_circular[-max_linear_lag_i:], corr_circular[: max_linear_lag_i + 1]))
    lags_i = np.arange(-max_linear_lag_i, max_linear_lag_i + 1, dtype=np.int64)

    if max_delay_seconds is None:
        max_delay_samples = float(max_linear_lag)
    else:
        max_delay_samples = min(float(max_linear_lag), abs(max_delay_seconds) * sample_rate)

    max_delay_i = int(np.floor(max_delay_samples * interpolation))
    feasible = np.abs(lags_i) <= max_delay_i
    if not np.any(feasible):
        return GCCPHATResult(0.0, 0.0, 0.0, 0.0, max_delay_samples, False, "empty physical lag window")

    fcorr = np.abs(corr[feasible])
    flags = lags_i[feasible]
    peak_local = int(np.argmax(fcorr))
    peak_value = float(fcorr[peak_local])
    lag_i = int(flags[peak_local])

    # Peak uniqueness diagnostic: compare with best peak excluding +/- one
    # native sample around the winner. This is more useful than comparing an
    # oversampled peak with its immediate interpolation neighbours.
    exclusion = interpolation
    mask = np.ones(fcorr.size, dtype=bool)
    lo = max(0, peak_local - exclusion)
    hi = min(fcorr.size, peak_local + exclusion + 1)
    mask[lo:hi] = False
    second = float(np.max(fcorr[mask])) if np.any(mask) else 0.0
    peak_ratio = peak_value / max(second, epsilon)

    delay_samples = lag_i / interpolation
    delay_seconds = delay_samples / sample_rate
    valid = peak_ratio >= min_peak_ratio
    reason = "" if valid else f"ambiguous GCC peak ratio {peak_ratio:.3f}"

    return GCCPHATResult(
        delay_seconds=delay_seconds,
        delay_samples=delay_samples,
        peak_value=peak_value,
        peak_ratio=peak_ratio,
        max_delay_samples=max_delay_samples,
        valid=valid,
        reason=reason,
    )
