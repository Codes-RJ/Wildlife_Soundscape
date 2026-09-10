from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from .tdoa import TDOAMeasurement


@dataclass(frozen=True, slots=True)
class PositionResult:
    x: float
    y: float
    success: bool
    residual_rms_seconds: float
    residual_rms_meters: float
    cost: float
    nfev: int
    message: str


def _centroid(positions: dict[int, tuple[float, float]]) -> np.ndarray:
    p = np.asarray(list(positions.values()), dtype=np.float64)
    return p.mean(axis=0)


def solve_position(
    node_positions: dict[int, tuple[float, float]],
    measurements: list[TDOAMeasurement],
    *,
    speed_of_sound_mps: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    initial_xy: tuple[float, float] | None = None,
) -> PositionResult:
    valid = [m for m in measurements if m.valid]
    if len(valid) < 2:
        return PositionResult(
            math.nan,
            math.nan,
            False,
            math.inf,
            math.inf,
            math.inf,
            0,
            "need at least two valid TDOAs",
        )

    used_nodes = {m.node_a for m in valid} | {m.node_b for m in valid}
    if len(used_nodes) < 3:
        return PositionResult(
            math.nan,
            math.nan,
            False,
            math.inf,
            math.inf,
            math.inf,
            0,
            "2D localization needs at least three distinct nodes",
        )

    if speed_of_sound_mps <= 0:
        raise ValueError("speed_of_sound_mps must be positive")

    x0 = (
        np.asarray(initial_xy, dtype=np.float64)
        if initial_xy
        else _centroid(node_positions)
    )

    def residuals(xy: np.ndarray) -> np.ndarray:
        x, y = float(xy[0]), float(xy[1])
        out = []
        for m in valid:
            ax, ay = node_positions[m.node_a]
            bx, by = node_positions[m.node_b]
            da = math.hypot(x - ax, y - ay)
            db = math.hypot(x - bx, y - by)
            predicted = (db - da) / speed_of_sound_mps
            out.append(predicted - m.delay_seconds)
        return np.asarray(out, dtype=np.float64)

    kwargs: dict[str, object] = {
        "fun": residuals,
        "loss": "soft_l1",
        "f_scale": 1.0 / speed_of_sound_mps,  # ~1 m residual scale
        "max_nfev": 300,
        "ftol": 1e-12,
        "xtol": 1e-12,
        "gtol": 1e-12,
    }

    if bounds is not None:
        lb = np.asarray(bounds[0], dtype=np.float64)
        ub = np.asarray(bounds[1], dtype=np.float64)
        x0 = np.clip(x0, lb, ub)
        kwargs["bounds"] = (lb, ub)

    kwargs["x0"] = x0

    result = least_squares(**kwargs)
    r = residuals(result.x)
    rms_s = float(np.sqrt(np.mean(r * r))) if r.size else math.inf
    return PositionResult(
        x=float(result.x[0]),
        y=float(result.x[1]),
        success=bool(result.success) and np.isfinite(result.x).all(),
        residual_rms_seconds=rms_s,
        residual_rms_meters=rms_s * speed_of_sound_mps,
        cost=float(result.cost),
        nfev=int(result.nfev),
        message=str(result.message),
    )
