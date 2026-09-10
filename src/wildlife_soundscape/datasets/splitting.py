"""Deterministic group-level train, validation, and test splitting."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Target fractions for a leakage-safe dataset split."""

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in values
        ):
            raise ValueError("Split ratios must be finite, non-negative numbers.")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Split ratios must sum to 1.0.")


def assign_group_splits(
    group_ids: Iterable[str],
    *,
    seed: int = 2026,
    ratios: SplitRatios | None = None,
) -> dict[str, str]:
    """Assign each unique recording group to exactly one stable split."""
    ratios = ratios or SplitRatios()
    ratios.validate()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    groups = list(group_ids)
    if not groups or any(
        not isinstance(group, str) or not group.strip() for group in groups
    ):
        raise ValueError("group_ids must contain non-empty strings.")
    if len(groups) != len(set(groups)):
        raise ValueError("group_ids must be unique.")

    groups.sort()
    random.Random(seed).shuffle(groups)
    names = ("train", "validation", "test")
    targets = tuple(
        len(groups) * value for value in (ratios.train, ratios.validation, ratios.test)
    )
    counts = [math.floor(target) for target in targets]
    remainder_order = sorted(
        range(3),
        key=lambda item: (targets[item] - counts[item], -item),
        reverse=True,
    )
    for index in remainder_order[: len(groups) - sum(counts)]:
        counts[index] += 1

    assignments: dict[str, str] = {}
    cursor = 0
    for name, count in zip(names, counts, strict=True):
        for group in groups[cursor : cursor + count]:
            assignments[group] = name
        cursor += count
    return assignments
