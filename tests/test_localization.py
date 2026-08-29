from __future__ import annotations

import math

import numpy as np

from localization.gcc_phat import gcc_phat
from localization.solver import solve_position
from localization.tdoa import TDOAMeasurement


def fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    # FFT-domain fractional delay for synthetic test only.
    n = signal.size
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n)
    phase = np.exp(-2j * np.pi * freqs * delay_samples)
    return np.fft.irfft(X * phase, n=n)


def test_gcc_phat_known_fractional_delay() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(size=4096) * np.hanning(4096)
    delayed = fractional_delay(ref, 7.25)
    result = gcc_phat(delayed, ref, sample_rate=48_000, max_delay_seconds=0.001, interpolation=16)
    assert result.valid
    assert abs(result.delay_samples - 7.25) < 0.2


def test_solver_known_source() -> None:
    nodes = {
        1: (0.0, 0.0),
        2: (0.5, 0.8660254038),
        3: (1.0, 0.0),
    }
    source = (0.45, 0.35)
    c = 343.0
    distances = {i: math.hypot(source[0] - x, source[1] - y) for i, (x, y) in nodes.items()}
    ms = []
    for a, b in ((1, 2), (1, 3), (2, 3)):
        tau = (distances[b] - distances[a]) / c
        ms.append(TDOAMeasurement(a, b, tau, tau * 48_000, 10.0, 1.0, True, ""))
    result = solve_position(nodes, ms, speed_of_sound_mps=c)
    assert result.success
    assert math.hypot(result.x - source[0], result.y - source[1]) < 1e-4
