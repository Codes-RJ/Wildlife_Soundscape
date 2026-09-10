"""
Typed research-analytics models.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module defines the data contracts shared by the higher-level
analytics modules:

    activity.py
    environmental.py
    spatial.py
    behavior.py
    service.py

Design principles
-----------------
1. Keep analytics models independent from SQLite.
2. Keep analytics models independent from Streamlit/dashboard code.
3. Keep statistical results explicit and reproducible.
4. Avoid causal language for observational environmental associations.
5. Preserve enough metadata for research-paper tables and exports.
6. Make all models directly serializable through ``to_dict()``.

No acoustic DSP, classification, localization or database processing
is performed in this module.
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from dataclasses import (
    dataclass,
)

from datetime import (
    datetime,
)

from enum import (
    Enum,
)

from typing import (
    Any,
)


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _require_finite(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require one finite numeric value.
    """

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(f"{name} must be numeric.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def _require_nonnegative_finite(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require one finite value >= 0.
    """

    result = _require_finite(
        value,
        name=name,
    )

    if result < 0.0:
        raise ValueError(f"{name} cannot be negative.")

    return result


def _require_probability(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require one finite value in [0, 1].
    """

    result = _require_finite(
        value,
        name=name,
    )

    if not (0.0 <= result <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")

    return result


def _require_nonnegative_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a Python integer >= 0.

    bool is explicitly rejected even though bool subclasses int.
    """

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        int,
    ):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")

    return int(value)


def _require_nonempty_string(
    value: str,
    *,
    name: str,
) -> str:
    """
    Require a non-empty trimmed string.
    """

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(f"{name} must be a string.")

    result = value.strip()

    if not (result):
        raise ValueError(f"{name} cannot be empty.")

    return result


def _serialize_datetime(
    value: datetime | None,
) -> str | None:
    """
    Convert datetime to ISO-8601 for JSON/export compatibility.
    """

    if value is None:
        return None

    if not isinstance(
        value,
        datetime,
    ):
        raise TypeError("Expected datetime value.")

    return value.isoformat()


# ======================================================================
# ASSOCIATION DIRECTION
# ======================================================================


class AssociationDirection(
    str,
    Enum,
):
    """
    Direction of an observed statistical association.

    These labels intentionally describe association only.

    They do not imply causation.
    """

    POSITIVE = "positive"

    NEGATIVE = "negative"

    NEUTRAL = "neutral"


# ======================================================================
# ASSOCIATION STRENGTH
# ======================================================================


class AssociationStrength(
    str,
    Enum,
):
    """
    Descriptive magnitude category for an observed association.
    """

    NEGLIGIBLE = "negligible"

    WEAK = "weak"

    MODERATE = "moderate"

    STRONG = "strong"

    VERY_STRONG = "very_strong"


# ======================================================================
# BEHAVIOR INDICATOR STATUS
# ======================================================================


class BehaviorIndicatorStatus(
    str,
    Enum,
):
    """
    Conservative state assigned to one derived behavioral indicator.

    The system deliberately avoids presenting inferred behavior as a
    biological ground-truth label.
    """

    INSUFFICIENT_DATA = "insufficient_data"

    LOW = "low"

    MODERATE = "moderate"

    HIGH = "high"


# ======================================================================
# ANALYTICS TIME WINDOW
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class AnalyticsTimeWindow:
    """
    Temporal range used for one analytics operation.

    start
        Inclusive start timestamp when available.

    end
        Exclusive or logical analysis end timestamp.

    event_count
        Number of events contained in the window.
    """

    start: datetime | None

    end: datetime | None

    event_count: int

    def __post_init__(
        self,
    ) -> None:

        _require_nonnegative_int(
            self.event_count,
            name="event_count",
        )

        if self.start is not None and not isinstance(
            self.start,
            datetime,
        ):
            raise TypeError("start must be datetime or None.")

        if self.end is not None and not isinstance(
            self.end,
            datetime,
        ):
            raise TypeError("end must be datetime or None.")

        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end cannot be earlier than start.")

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "start": _serialize_datetime(self.start),
            "end": _serialize_datetime(self.end),
            "event_count": self.event_count,
        }


# ======================================================================
# ACTIVITY BIN
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ActivityBin:
    """
    Acoustic activity accumulated into one temporal bin.

    Examples
    --------
    A bin may represent:

        one hour
        one day
        one 15-minute interval

    depending on the requested analytics resolution.
    """

    bucket_start: datetime

    bucket_end: datetime

    event_count: int

    active_duration_s: float

    mean_confidence: float | None = None

    dominant_class: str | None = None

    def __post_init__(
        self,
    ) -> None:

        if not isinstance(
            self.bucket_start,
            datetime,
        ):
            raise TypeError("bucket_start must be datetime.")

        if not isinstance(
            self.bucket_end,
            datetime,
        ):
            raise TypeError("bucket_end must be datetime.")

        if self.bucket_end <= self.bucket_start:
            raise ValueError(("bucket_end must be later than bucket_start."))

        _require_nonnegative_int(
            self.event_count,
            name="event_count",
        )

        _require_nonnegative_finite(
            self.active_duration_s,
            name="active_duration_s",
        )

        if self.mean_confidence is not None:
            _require_probability(
                self.mean_confidence,
                name="mean_confidence",
            )

        if self.dominant_class is not None:
            _require_nonempty_string(
                self.dominant_class,
                name="dominant_class",
            )

    @property
    def bucket_duration_s(
        self,
    ) -> float:
        """
        Width of this temporal bin in seconds.
        """

        return float((self.bucket_end - self.bucket_start).total_seconds())

    @property
    def activity_fraction(
        self,
    ) -> float:
        """
        Fraction of bin duration represented by detected acoustic-event
        activity.

        Clamped to 1.0 because overlapping events may otherwise make
        summed event duration exceed the wall-clock bin duration.
        """

        duration = self.bucket_duration_s

        if duration <= 0.0:
            return 0.0

        return min(
            1.0,
            self.active_duration_s / duration,
        )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "bucket_start": self.bucket_start.isoformat(),
            "bucket_end": self.bucket_end.isoformat(),
            "event_count": self.event_count,
            "active_duration_s": self.active_duration_s,
            "bucket_duration_s": self.bucket_duration_s,
            "activity_fraction": self.activity_fraction,
            "mean_confidence": self.mean_confidence,
            "dominant_class": self.dominant_class,
        }


# ======================================================================
# CLASS ACTIVITY SUMMARY
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ClassActivitySummary:
    """
    Aggregate activity statistics for one acoustic classification label.
    """

    class_label: str

    event_count: int

    total_duration_s: float

    mean_confidence: float | None

    proportion_of_events: float

    def __post_init__(
        self,
    ) -> None:

        _require_nonempty_string(
            self.class_label,
            name="class_label",
        )

        _require_nonnegative_int(
            self.event_count,
            name="event_count",
        )

        _require_nonnegative_finite(
            self.total_duration_s,
            name="total_duration_s",
        )

        if self.mean_confidence is not None:
            _require_probability(
                self.mean_confidence,
                name="mean_confidence",
            )

        _require_probability(
            self.proportion_of_events,
            name="proportion_of_events",
        )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "class_label": self.class_label,
            "event_count": self.event_count,
            "total_duration_s": self.total_duration_s,
            "mean_confidence": self.mean_confidence,
            "proportion_of_events": self.proportion_of_events,
        }


# ======================================================================
# ACTIVITY SUMMARY
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ActivitySummary:
    """
    Overall temporal activity summary.
    """

    window: AnalyticsTimeWindow

    total_events: int

    total_active_duration_s: float

    mean_event_duration_s: float

    peak_activity_hour: int | None

    class_summaries: tuple[
        ClassActivitySummary,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:

        if not isinstance(
            self.window,
            AnalyticsTimeWindow,
        ):
            raise TypeError(("window must be an AnalyticsTimeWindow."))

        _require_nonnegative_int(
            self.total_events,
            name="total_events",
        )

        _require_nonnegative_finite(
            self.total_active_duration_s,
            name="total_active_duration_s",
        )

        _require_nonnegative_finite(
            self.mean_event_duration_s,
            name="mean_event_duration_s",
        )

        if self.peak_activity_hour is not None:
            if isinstance(
                self.peak_activity_hour,
                bool,
            ) or not isinstance(
                self.peak_activity_hour,
                int,
            ):
                raise TypeError(("peak_activity_hour must be an integer or None."))

            if not (0 <= self.peak_activity_hour <= 23):
                raise ValueError(("peak_activity_hour must be between 0 and 23."))

        if not isinstance(
            self.class_summaries,
            tuple,
        ):
            raise TypeError("class_summaries must be a tuple.")

        for summary in self.class_summaries:
            if not isinstance(
                summary,
                ClassActivitySummary,
            ):
                raise TypeError(
                    ("class_summaries must contain ClassActivitySummary objects.")
                )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "window": self.window.to_dict(),
            "total_events": self.total_events,
            "total_active_duration_s": self.total_active_duration_s,
            "mean_event_duration_s": self.mean_event_duration_s,
            "peak_activity_hour": self.peak_activity_hour,
            "class_summaries": [summary.to_dict() for summary in self.class_summaries],
        }


# ======================================================================
# ENVIRONMENTAL ASSOCIATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class EnvironmentalAssociation:
    """
    Statistical association between one environmental variable and one
    acoustic/activity metric.

    Default analysis will use Spearman rank correlation.

    Important
    ---------
    This model represents association, not causation.
    """

    environmental_variable: str

    response_variable: str

    sample_count: int

    coefficient: float | None

    p_value: float | None

    direction: AssociationDirection

    strength: AssociationStrength

    statistically_significant: bool

    method: str = "spearman"

    def __post_init__(
        self,
    ) -> None:

        _require_nonempty_string(
            self.environmental_variable,
            name="environmental_variable",
        )

        _require_nonempty_string(
            self.response_variable,
            name="response_variable",
        )

        _require_nonnegative_int(
            self.sample_count,
            name="sample_count",
        )

        if self.coefficient is not None:
            coefficient = _require_finite(
                self.coefficient,
                name="coefficient",
            )

            if not (-1.0 <= coefficient <= 1.0):
                raise ValueError(("correlation coefficient must be in [-1, 1]."))

        if self.p_value is not None:
            _require_probability(
                self.p_value,
                name="p_value",
            )

        if not isinstance(
            self.direction,
            AssociationDirection,
        ):
            raise TypeError(("direction must be an AssociationDirection."))

        if not isinstance(
            self.strength,
            AssociationStrength,
        ):
            raise TypeError(("strength must be an AssociationStrength."))

        if not isinstance(
            self.statistically_significant,
            bool,
        ):
            raise TypeError(("statistically_significant must be bool."))

        _require_nonempty_string(
            self.method,
            name="method",
        )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "environmental_variable": self.environmental_variable,
            "response_variable": self.response_variable,
            "sample_count": self.sample_count,
            "coefficient": self.coefficient,
            "p_value": self.p_value,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "statistically_significant": self.statistically_significant,
            "method": self.method,
        }


# ======================================================================
# SPATIAL CELL
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SpatialCell:
    """
    One rectangular spatial-analysis cell.

    Coordinates are expressed in meters in the localization-array
    coordinate system.
    """

    cell_id: str

    x_min_m: float

    x_max_m: float

    y_min_m: float

    y_max_m: float

    event_count: int

    mean_confidence: float | None = None

    def __post_init__(
        self,
    ) -> None:

        _require_nonempty_string(
            self.cell_id,
            name="cell_id",
        )

        x_min = _require_finite(
            self.x_min_m,
            name="x_min_m",
        )

        x_max = _require_finite(
            self.x_max_m,
            name="x_max_m",
        )

        y_min = _require_finite(
            self.y_min_m,
            name="y_min_m",
        )

        y_max = _require_finite(
            self.y_max_m,
            name="y_max_m",
        )

        if x_max <= x_min:
            raise ValueError("x_max_m must exceed x_min_m.")

        if y_max <= y_min:
            raise ValueError("y_max_m must exceed y_min_m.")

        _require_nonnegative_int(
            self.event_count,
            name="event_count",
        )

        if self.mean_confidence is not None:
            _require_probability(
                self.mean_confidence,
                name="mean_confidence",
            )

    @property
    def center_x_m(
        self,
    ) -> float:

        return (self.x_min_m + self.x_max_m) / 2.0

    @property
    def center_y_m(
        self,
    ) -> float:

        return (self.y_min_m + self.y_max_m) / 2.0

    @property
    def area_m2(
        self,
    ) -> float:

        return (self.x_max_m - self.x_min_m) * (self.y_max_m - self.y_min_m)

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "cell_id": self.cell_id,
            "x_min_m": self.x_min_m,
            "x_max_m": self.x_max_m,
            "y_min_m": self.y_min_m,
            "y_max_m": self.y_max_m,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "area_m2": self.area_m2,
            "event_count": self.event_count,
            "mean_confidence": self.mean_confidence,
        }


# ======================================================================
# SPATIAL TRANSITION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SpatialTransition:
    """
    Observed transition between two spatial cells.

    This is an acoustic-location transition and should not automatically
    be interpreted as confirmed individual-animal movement.
    """

    source_cell_id: str

    destination_cell_id: str

    transition_count: int

    probability: float

    def __post_init__(
        self,
    ) -> None:

        _require_nonempty_string(
            self.source_cell_id,
            name="source_cell_id",
        )

        _require_nonempty_string(
            self.destination_cell_id,
            name="destination_cell_id",
        )

        _require_nonnegative_int(
            self.transition_count,
            name="transition_count",
        )

        _require_probability(
            self.probability,
            name="probability",
        )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "source_cell_id": self.source_cell_id,
            "destination_cell_id": self.destination_cell_id,
            "transition_count": self.transition_count,
            "probability": self.probability,
        }


# ======================================================================
# SPATIAL SUMMARY
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SpatialSummary:
    """
    Aggregate spatial-distribution result.
    """

    localized_event_count: int

    total_event_count: int

    localization_coverage: float

    cells: tuple[
        SpatialCell,
        ...,
    ]

    transitions: tuple[
        SpatialTransition,
        ...,
    ]

    hotspot_cell_id: str | None

    def __post_init__(
        self,
    ) -> None:

        _require_nonnegative_int(
            self.localized_event_count,
            name="localized_event_count",
        )

        _require_nonnegative_int(
            self.total_event_count,
            name="total_event_count",
        )

        if self.localized_event_count > self.total_event_count:
            raise ValueError(("localized_event_count cannot exceed total_event_count."))

        _require_probability(
            self.localization_coverage,
            name="localization_coverage",
        )

        if not isinstance(
            self.cells,
            tuple,
        ):
            raise TypeError("cells must be a tuple.")

        if not isinstance(
            self.transitions,
            tuple,
        ):
            raise TypeError("transitions must be a tuple.")

        for cell in self.cells:
            if not isinstance(
                cell,
                SpatialCell,
            ):
                raise TypeError(("cells must contain SpatialCell objects."))

        for transition in self.transitions:
            if not isinstance(
                transition,
                SpatialTransition,
            ):
                raise TypeError(("transitions must contain SpatialTransition objects."))

        if self.hotspot_cell_id is not None:
            _require_nonempty_string(
                self.hotspot_cell_id,
                name="hotspot_cell_id",
            )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "localized_event_count": self.localized_event_count,
            "total_event_count": self.total_event_count,
            "localization_coverage": self.localization_coverage,
            "hotspot_cell_id": self.hotspot_cell_id,
            "cells": [cell.to_dict() for cell in self.cells],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


# ======================================================================
# BEHAVIOR INDICATOR
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class BehaviorIndicator:
    """
    Conservative derived behavioral indicator.

    Important
    ---------
    This is not intended to claim verified biological behavior.

    It represents a repeatable quantitative pattern observed in acoustic
    activity, classification and/or localized event trajectories.
    """

    name: str

    status: BehaviorIndicatorStatus

    score: float | None

    supporting_event_count: int

    description: str

    evidence: tuple[
        str,
        ...,
    ] = ()

    def __post_init__(
        self,
    ) -> None:

        _require_nonempty_string(
            self.name,
            name="name",
        )

        if not isinstance(
            self.status,
            BehaviorIndicatorStatus,
        ):
            raise TypeError(("status must be a BehaviorIndicatorStatus."))

        if self.score is not None:
            _require_probability(
                self.score,
                name="score",
            )

        _require_nonnegative_int(
            self.supporting_event_count,
            name="supporting_event_count",
        )

        _require_nonempty_string(
            self.description,
            name="description",
        )

        if not isinstance(
            self.evidence,
            tuple,
        ):
            raise TypeError("evidence must be a tuple.")

        for entry in self.evidence:
            _require_nonempty_string(
                entry,
                name="evidence entry",
            )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "name": self.name,
            "status": self.status.value,
            "score": self.score,
            "supporting_event_count": self.supporting_event_count,
            "description": self.description,
            "evidence": list(self.evidence),
        }


# ======================================================================
# COMPLETE RESEARCH ANALYTICS REPORT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ResearchAnalyticsReport:
    """
    High-level container returned by analytics.service.

    Each subsystem remains independently optional so limited datasets can
    still produce useful partial analysis.
    """

    generated_at: datetime

    window: AnalyticsTimeWindow

    activity: ActivitySummary | None

    environmental_associations: tuple[
        EnvironmentalAssociation,
        ...,
    ]

    spatial: SpatialSummary | None

    behavior_indicators: tuple[
        BehaviorIndicator,
        ...,
    ]

    warnings: tuple[
        str,
        ...,
    ] = ()

    def __post_init__(
        self,
    ) -> None:

        if not isinstance(
            self.generated_at,
            datetime,
        ):
            raise TypeError("generated_at must be datetime.")

        if not isinstance(
            self.window,
            AnalyticsTimeWindow,
        ):
            raise TypeError(("window must be an AnalyticsTimeWindow."))

        if self.activity is not None and not isinstance(
            self.activity,
            ActivitySummary,
        ):
            raise TypeError(("activity must be an ActivitySummary or None."))

        if not isinstance(
            self.environmental_associations,
            tuple,
        ):
            raise TypeError(("environmental_associations must be a tuple."))

        for association in self.environmental_associations:
            if not isinstance(
                association,
                EnvironmentalAssociation,
            ):
                raise TypeError(
                    (
                        "environmental_associations "
                        "must contain "
                        "EnvironmentalAssociation "
                        "objects."
                    )
                )

        if self.spatial is not None and not isinstance(
            self.spatial,
            SpatialSummary,
        ):
            raise TypeError(("spatial must be a SpatialSummary or None."))

        if not isinstance(
            self.behavior_indicators,
            tuple,
        ):
            raise TypeError(("behavior_indicators must be a tuple."))

        for indicator in self.behavior_indicators:
            if not isinstance(
                indicator,
                BehaviorIndicator,
            ):
                raise TypeError(
                    ("behavior_indicators must contain BehaviorIndicator objects.")
                )

        if not isinstance(
            self.warnings,
            tuple,
        ):
            raise TypeError("warnings must be a tuple.")

        for warning in self.warnings:
            _require_nonempty_string(
                warning,
                name="warning",
            )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "generated_at": self.generated_at.isoformat(),
            "window": self.window.to_dict(),
            "activity": (
                self.activity.to_dict() if self.activity is not None else None
            ),
            "environmental_associations": [
                association.to_dict() for association in self.environmental_associations
            ],
            "spatial": (self.spatial.to_dict() if self.spatial is not None else None),
            "behavior_indicators": [
                indicator.to_dict() for indicator in self.behavior_indicators
            ],
            "warnings": list(self.warnings),
        }
