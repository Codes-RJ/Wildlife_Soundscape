"""Deterministic synthetic positions relative to the configured microphone triangle."""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DemoPoint:
    x: float
    y: float
    region: str


def demo_points(
    node_positions: Mapping[int, tuple[float, float]],
    count: int = 144,
    seed: int = 20260909,
) -> list[DemoPoint]:
    """Half inside, half outside; negative barycentric weights ensure exterior points."""
    if count < 12 or count % 2:
        raise ValueError("Demo point count must be even and at least 12.")
    triangle = np.asarray([node_positions[n] for n in sorted(node_positions)], dtype=float)
    if triangle.shape != (3, 2) or not np.isfinite(triangle).all():
        raise ValueError("A finite three-node triangle is required.")
    if abs(np.linalg.det(triangle[1:] - triangle[0])) < 1e-10:
        raise ValueError("Microphone triangle must not be collinear.")
    rng = np.random.default_rng(seed)
    points = []
    for index in range(count):
        weights = rng.dirichlet([2.0, 2.0, 2.0])
        region = "inside" if index < count // 2 else "outside"
        if region == "outside":
            opposite = index % 3
            weights[opposite] = 0
            distance = rng.uniform(0.15, 1.8)
            weights *= (1 + distance) / weights.sum()
            weights[opposite] = -distance
        x, y = weights @ triangle
        points.append(DemoPoint(float(x), float(y), region))
    return points
