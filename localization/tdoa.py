from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TDOAMeasurement:
    node_a: int
    node_b: int
    delay_seconds: float  # arrival at B minus arrival at A
    delay_samples: float
    peak_ratio: float
    max_delay_seconds: float
    valid: bool
    reason: str = ""


def pair_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def physical_max_delay(
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    speed_of_sound_mps: float,
) -> float:
    if speed_of_sound_mps <= 0:
        raise ValueError("speed_of_sound_mps must be positive")
    return pair_distance(a, b) / speed_of_sound_mps
