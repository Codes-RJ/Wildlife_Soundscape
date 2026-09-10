"""
Conservative behavior-related acoustic indicators.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module derives higher-level quantitative indicators from:

    temporal acoustic activity
    acoustic-class distribution
    spatial event concentration
    consecutive localized-event transitions

The outputs are designed for:

    research exploration
    dashboard summaries
    longitudinal comparisons
    report generation

Scientific limitation
---------------------
The system observes acoustic events.

It does NOT directly observe:

    individual animal identity
    body movement
    feeding
    mating
    nesting
    aggression
    migration
    territorial ownership
    confirmed animal trajectories

Therefore this module deliberately produces conservative indicators such
as:

    temporal activity concentration
    acoustic-class concentration
    spatial concentration
    spatial redistribution

rather than claiming:

    "the animal is feeding"
    "the animal migrated"
    "the species is territorial"
    "the same individual moved"

Interpretation
--------------
A high indicator score means that the corresponding measurable acoustic
pattern is strong in the analyzed dataset.

It does not by itself establish biological causation.

No database access is performed in this module.
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from typing import (
    Iterable,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from .models import (
    ActivityBin,
    ActivitySummary,
    BehaviorIndicator,
    BehaviorIndicatorStatus,
    SpatialCell,
    SpatialSummary,
    SpatialTransition,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_MIN_ACTIVITY_EVENTS = 5


DEFAULT_MIN_LOCALIZED_EVENTS = 5


DEFAULT_MIN_TRANSITIONS = 3


LOW_SCORE_LIMIT = 0.33


MODERATE_SCORE_LIMIT = 0.66


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _require_nonnegative_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a Python integer >= 0.
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


def _require_positive_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a Python integer > 0.
    """

    value = _require_nonnegative_int(
        value,
        name=name,
    )

    if value == 0:
        raise ValueError(f"{name} must be greater than 0.")

    return value


def _clamp_probability(
    value: float,
) -> float:
    """
    Clamp a finite numeric score into [0, 1].
    """

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError("score must be numeric.") from exc

    if not math.isfinite(value):
        raise ValueError("score must be finite.")

    return min(
        1.0,
        max(
            0.0,
            value,
        ),
    )


# ======================================================================
# SCORE -> STATUS
# ======================================================================


def score_to_behavior_status(
    score: float | None,
    *,
    sufficient_data: bool,
) -> BehaviorIndicatorStatus:
    """
    Convert a normalized indicator score into a descriptive state.

    Thresholds
    ----------
    score < 0.33
        LOW

    score < 0.66
        MODERATE

    score >= 0.66
        HIGH

    If sufficient_data is False:

        INSUFFICIENT_DATA
    """

    if not isinstance(
        sufficient_data,
        bool,
    ):
        raise TypeError("sufficient_data must be bool.")

    if not (sufficient_data):
        return BehaviorIndicatorStatus.INSUFFICIENT_DATA

    if score is None:
        return BehaviorIndicatorStatus.INSUFFICIENT_DATA

    score = _clamp_probability(score)

    if score < LOW_SCORE_LIMIT:
        return BehaviorIndicatorStatus.LOW

    if score < MODERATE_SCORE_LIMIT:
        return BehaviorIndicatorStatus.MODERATE

    return BehaviorIndicatorStatus.HIGH


# ======================================================================
# TEMPORAL CONCENTRATION
# ======================================================================


def build_temporal_concentration_indicator(
    bins: Iterable[ActivityBin],
    *,
    min_events: int = DEFAULT_MIN_ACTIVITY_EVENTS,
) -> BehaviorIndicator:
    """
    Measure how strongly detected acoustic events are concentrated into
    a small portion of the observed temporal bins.

    Score
    -----
    The score is:

        maximum event count in one bin
        --------------------------------
        total event count across all bins

    Examples
    --------
    Events evenly distributed:

        2, 2, 2, 2

        score = 0.25

    Strong temporal concentration:

        1, 1, 1, 12

        score = 0.80

    Interpretation
    --------------
    A high score indicates temporally concentrated acoustic detections.

    It may reflect:

        calling bouts
        dawn/dusk activity
        episodic environmental conditions
        detector sensitivity changes
        clustered source activity

    It is not itself proof of a biological behavioral state.
    """

    min_events = _require_positive_int(
        min_events,
        name="min_events",
    )

    materialized = tuple(bins)

    for activity_bin in materialized:
        if not isinstance(
            activity_bin,
            ActivityBin,
        ):
            raise TypeError(("bins must contain ActivityBin objects."))

    total_events = sum(activity_bin.event_count for activity_bin in materialized)

    if total_events < min_events:
        return BehaviorIndicator(
            name="temporal_activity_concentration",
            status=BehaviorIndicatorStatus.INSUFFICIENT_DATA,
            score=None,
            supporting_event_count=total_events,
            description=(
                "Insufficient detected acoustic "
                "events to estimate temporal "
                "activity concentration reliably."
            ),
            evidence=(
                (f"Observed events: {total_events}"),
                (f"Minimum required: {min_events}"),
            ),
        )

    if not (materialized):
        return BehaviorIndicator(
            name="temporal_activity_concentration",
            status=BehaviorIndicatorStatus.INSUFFICIENT_DATA,
            score=None,
            supporting_event_count=0,
            description=("No temporal activity bins were available for analysis."),
            evidence=(),
        )

    peak_bin_count = max(activity_bin.event_count for activity_bin in materialized)

    score = peak_bin_count / total_events

    score = _clamp_probability(score)

    status = score_to_behavior_status(
        score,
        sufficient_data=True,
    )

    return BehaviorIndicator(
        name="temporal_activity_concentration",
        status=status,
        score=score,
        supporting_event_count=total_events,
        description=(
            "Degree to which detected acoustic "
            "event onsets are concentrated in "
            "the most active temporal bin."
        ),
        evidence=(
            (f"Total events: {total_events}"),
            (f"Peak-bin events: {peak_bin_count}"),
            (
                "This describes acoustic-event "
                "timing concentration, not a "
                "confirmed behavioral state."
            ),
        ),
    )


# ======================================================================
# ACOUSTIC CLASS CONCENTRATION
# ======================================================================


def build_class_concentration_indicator(
    activity: ActivitySummary,
    *,
    min_events: int = DEFAULT_MIN_ACTIVITY_EVENTS,
) -> BehaviorIndicator:
    """
    Measure whether one broad acoustic class dominates detected events.

    Score
    -----
    Maximum:

        proportion_of_events

    among all class summaries.

    Interpretation
    --------------
    A high score means one broad acoustic category dominates the
    analyzed recording period.

    Example:

        Bird Song      72%
        Insect         15%
        Amphibian       8%
        Unknown         5%

    produces:

        score = 0.72

    This should not be interpreted as species abundance.
    """

    if not isinstance(
        activity,
        ActivitySummary,
    ):
        raise TypeError(("activity must be an ActivitySummary."))

    min_events = _require_positive_int(
        min_events,
        name="min_events",
    )

    total_events = activity.total_events

    if total_events < min_events or not activity.class_summaries:
        return BehaviorIndicator(
            name="acoustic_class_concentration",
            status=BehaviorIndicatorStatus.INSUFFICIENT_DATA,
            score=None,
            supporting_event_count=total_events,
            description=(
                "Insufficient classified acoustic "
                "events to estimate class "
                "concentration reliably."
            ),
            evidence=(
                (f"Observed events: {total_events}"),
                (f"Minimum required: {min_events}"),
            ),
        )

    dominant_summary = max(
        activity.class_summaries,
        key=lambda summary: (
            summary.proportion_of_events,
            summary.event_count,
        ),
    )

    score = _clamp_probability(dominant_summary.proportion_of_events)

    status = score_to_behavior_status(
        score,
        sufficient_data=True,
    )

    return BehaviorIndicator(
        name="acoustic_class_concentration",
        status=status,
        score=score,
        supporting_event_count=total_events,
        description=(
            "Concentration of detected acoustic "
            "events within the most common broad "
            "classification category."
        ),
        evidence=(
            (f"Dominant class: {dominant_summary.class_label}"),
            (f"Dominant-class events: {dominant_summary.event_count}"),
            (f"Dominant-class proportion: {dominant_summary.proportion_of_events:.3f}"),
            (
                "Class dominance represents "
                "detected acoustic composition, "
                "not population abundance."
            ),
        ),
    )


# ======================================================================
# SPATIAL ENTROPY
# ======================================================================


def _normalized_spatial_entropy(
    cells: Iterable[SpatialCell],
) -> float:
    """
    Compute normalized Shannon entropy over occupied cells.

    Returns
    -------
    0
        all localized events occupy one cell.

    1
        events are evenly distributed over occupied cells.
    """

    counts: list[int] = []

    for cell in cells:
        if not isinstance(
            cell,
            SpatialCell,
        ):
            raise TypeError(("cells must contain SpatialCell objects."))

        if cell.event_count > 0:
            counts.append(cell.event_count)

    if len(counts) <= 1:
        return 0.0

    total = float(sum(counts))

    probabilities = [count / total for count in counts]

    entropy = -math.fsum(
        probability * math.log(probability) for probability in probabilities
    )

    maximum_entropy = math.log(len(probabilities))

    if maximum_entropy <= 0.0:
        return 0.0

    return _clamp_probability(entropy / maximum_entropy)


# ======================================================================
# SPATIAL CONCENTRATION
# ======================================================================


def build_spatial_concentration_indicator(
    spatial: SpatialSummary,
    *,
    min_localized_events: int = DEFAULT_MIN_LOCALIZED_EVENTS,
) -> BehaviorIndicator:
    """
    Measure concentration of localized acoustic detections.

    Composite score
    ---------------
    Two quantities are combined:

        hotspot fraction

            =
        hotspot event count
        -------------------
        localized events


        concentration from entropy

            =
        1 - normalized spatial entropy


    Final score:

        0.5 * hotspot_fraction
        +
        0.5 * (1 - spatial_entropy)

    A larger value indicates a stronger concentration of localized
    acoustic events in a smaller part of the observed grid.
    """

    if not isinstance(
        spatial,
        SpatialSummary,
    ):
        raise TypeError(("spatial must be a SpatialSummary."))

    min_localized_events = _require_positive_int(
        min_localized_events,
        name="min_localized_events",
    )

    localized_count = spatial.localized_event_count

    if localized_count < min_localized_events or not spatial.cells:
        return BehaviorIndicator(
            name="spatial_acoustic_concentration",
            status=BehaviorIndicatorStatus.INSUFFICIENT_DATA,
            score=None,
            supporting_event_count=localized_count,
            description=(
                "Insufficient localized acoustic "
                "events to estimate spatial "
                "concentration reliably."
            ),
            evidence=(
                (f"Localized events: {localized_count}"),
                (f"Minimum required: {min_localized_events}"),
            ),
        )

    occupied_cells = [cell for cell in spatial.cells if (cell.event_count > 0)]

    if not (occupied_cells):
        return BehaviorIndicator(
            name="spatial_acoustic_concentration",
            status=BehaviorIndicatorStatus.INSUFFICIENT_DATA,
            score=None,
            supporting_event_count=localized_count,
            description=(
                "No occupied spatial cells were "
                "available despite localized "
                "event metadata."
            ),
            evidence=(),
        )

    hotspot = max(
        occupied_cells,
        key=lambda cell: (
            cell.event_count,
            cell.cell_id,
        ),
    )

    hotspot_fraction = hotspot.event_count / localized_count

    entropy = _normalized_spatial_entropy(occupied_cells)

    entropy_concentration = 1.0 - entropy

    score = 0.5 * hotspot_fraction + 0.5 * entropy_concentration

    score = _clamp_probability(score)

    status = score_to_behavior_status(
        score,
        sufficient_data=True,
    )

    return BehaviorIndicator(
        name="spatial_acoustic_concentration",
        status=status,
        score=score,
        supporting_event_count=localized_count,
        description=(
            "Concentration of localized acoustic "
            "events within the observed spatial "
            "grid."
        ),
        evidence=(
            (f"Localized events: {localized_count}"),
            (f"Occupied cells: {len(occupied_cells)}"),
            (f"Hotspot cell: {hotspot.cell_id}"),
            (f"Hotspot event fraction: {hotspot_fraction:.3f}"),
            (f"Normalized spatial entropy: {entropy:.3f}"),
            (
                "Spatial concentration describes "
                "acoustic detections and does not "
                "establish territory or home range."
            ),
        ),
    )


# ======================================================================
# TRANSITION COUNT
# ======================================================================


def _total_transition_count(
    transitions: Iterable[SpatialTransition],
) -> int:
    """
    Sum observed transition counts.
    """

    total = 0

    for transition in transitions:
        if not isinstance(
            transition,
            SpatialTransition,
        ):
            raise TypeError(("transitions must contain SpatialTransition objects."))

        total += transition.transition_count

    return total


# ======================================================================
# SPATIAL REDISTRIBUTION
# ======================================================================


def build_spatial_redistribution_indicator(
    spatial: SpatialSummary,
    *,
    min_transitions: int = DEFAULT_MIN_TRANSITIONS,
) -> BehaviorIndicator:
    """
    Estimate how often consecutive localized acoustic events occur in
    different spatial cells.

    Score
    -----

        transitions to different cells
        ------------------------------
        all accepted transitions

    Examples
    --------

        A -> A
        A -> A
        A -> B
        B -> B

        score = 1 / 4 = 0.25


        A -> B
        B -> C
        C -> A
        A -> C

        score = 4 / 4 = 1.0

    Interpretation
    --------------
    A high score means consecutive localized acoustic events frequently
    shift between spatial cells.

    It does NOT prove movement of one individual animal.
    """

    if not isinstance(
        spatial,
        SpatialSummary,
    ):
        raise TypeError(("spatial must be a SpatialSummary."))

    min_transitions = _require_positive_int(
        min_transitions,
        name="min_transitions",
    )

    total_transitions = _total_transition_count(spatial.transitions)

    if total_transitions < min_transitions:
        return BehaviorIndicator(
            name="spatial_redistribution",
            status=BehaviorIndicatorStatus.INSUFFICIENT_DATA,
            score=None,
            supporting_event_count=spatial.localized_event_count,
            description=(
                "Insufficient consecutive "
                "localized-event transitions to "
                "estimate spatial redistribution."
            ),
            evidence=(
                (f"Observed transitions: {total_transitions}"),
                (f"Minimum required: {min_transitions}"),
            ),
        )

    different_cell_transitions = sum(
        transition.transition_count
        for transition in spatial.transitions
        if (transition.source_cell_id != transition.destination_cell_id)
    )

    same_cell_transitions = total_transitions - different_cell_transitions

    score = different_cell_transitions / total_transitions

    score = _clamp_probability(score)

    status = score_to_behavior_status(
        score,
        sufficient_data=True,
    )

    return BehaviorIndicator(
        name="spatial_redistribution",
        status=status,
        score=score,
        supporting_event_count=spatial.localized_event_count,
        description=(
            "Fraction of consecutive localized "
            "acoustic-event transitions that "
            "occur between different spatial "
            "cells."
        ),
        evidence=(
            (f"Total transitions: {total_transitions}"),
            (f"Different-cell transitions: {different_cell_transitions}"),
            (f"Same-cell transitions: {same_cell_transitions}"),
            (
                "Spatial redistribution is an "
                "acoustic-location pattern and "
                "must not be interpreted as a "
                "confirmed individual trajectory."
            ),
        ),
    )


# ======================================================================
# REPEATED-AREA ACOUSTIC ACTIVITY
# ======================================================================


def build_repeated_area_indicator(
    spatial: SpatialSummary,
    *,
    min_transitions: int = DEFAULT_MIN_TRANSITIONS,
) -> BehaviorIndicator:
    """
    Complementary indicator for repeated localization in the same cell.

    Score
    -----

        same-cell transitions
        ---------------------
        all accepted transitions

    This is mathematically complementary to spatial redistribution when
    calculated over the same transition set.

    Interpretation
    --------------
    A high score indicates that consecutive acoustic detections often
    remain within the same spatial cell.

    It must not be described as confirmed territory, nesting, residence
    or individual site fidelity.
    """

    if not isinstance(
        spatial,
        SpatialSummary,
    ):
        raise TypeError(("spatial must be a SpatialSummary."))

    min_transitions = _require_positive_int(
        min_transitions,
        name="min_transitions",
    )

    total_transitions = _total_transition_count(spatial.transitions)

    if total_transitions < min_transitions:
        return BehaviorIndicator(
            name="repeated_area_acoustic_activity",
            status=BehaviorIndicatorStatus.INSUFFICIENT_DATA,
            score=None,
            supporting_event_count=spatial.localized_event_count,
            description=(
                "Insufficient consecutive "
                "localized-event transitions to "
                "estimate repeated-area acoustic "
                "activity."
            ),
            evidence=(
                (f"Observed transitions: {total_transitions}"),
                (f"Minimum required: {min_transitions}"),
            ),
        )

    same_cell_transitions = sum(
        transition.transition_count
        for transition in spatial.transitions
        if (transition.source_cell_id == transition.destination_cell_id)
    )

    score = same_cell_transitions / total_transitions

    score = _clamp_probability(score)

    status = score_to_behavior_status(
        score,
        sufficient_data=True,
    )

    return BehaviorIndicator(
        name="repeated_area_acoustic_activity",
        status=status,
        score=score,
        supporting_event_count=spatial.localized_event_count,
        description=(
            "Frequency with which consecutive "
            "localized acoustic events remain "
            "within the same analysis cell."
        ),
        evidence=(
            (f"Total transitions: {total_transitions}"),
            (f"Same-cell transitions: {same_cell_transitions}"),
            (
                "Repeated localization does not "
                "establish individual identity, "
                "territory, nesting or residence."
            ),
        ),
    )


# ======================================================================
# COMPOSITE INDICATOR SET
# ======================================================================


def build_behavior_indicators(
    *,
    activity: ActivitySummary | None = None,
    activity_bins: Iterable[ActivityBin] | None = None,
    spatial: SpatialSummary | None = None,
    min_activity_events: int = DEFAULT_MIN_ACTIVITY_EVENTS,
    min_localized_events: int = DEFAULT_MIN_LOCALIZED_EVENTS,
    min_transitions: int = DEFAULT_MIN_TRANSITIONS,
) -> tuple[
    BehaviorIndicator,
    ...,
]:
    """
    Build the complete conservative behavior-related indicator set.

    Available indicators
    --------------------
    temporal_activity_concentration
        Requires ActivityBin data.

    acoustic_class_concentration
        Requires ActivitySummary.

    spatial_acoustic_concentration
        Requires SpatialSummary.

    spatial_redistribution
        Requires spatial transitions.

    repeated_area_acoustic_activity
        Requires spatial transitions.

    Missing upstream analyses simply omit the corresponding indicators.
    """

    min_activity_events = _require_positive_int(
        min_activity_events,
        name="min_activity_events",
    )

    min_localized_events = _require_positive_int(
        min_localized_events,
        name="min_localized_events",
    )

    min_transitions = _require_positive_int(
        min_transitions,
        name="min_transitions",
    )

    indicators: list[BehaviorIndicator] = []

    # ==============================================================
    # TEMPORAL CONCENTRATION
    # ==============================================================

    if activity_bins is not None:
        indicators.append(
            build_temporal_concentration_indicator(
                activity_bins,
                min_events=min_activity_events,
            )
        )

    # ==============================================================
    # CLASS CONCENTRATION
    # ==============================================================

    if activity is not None:
        indicators.append(
            build_class_concentration_indicator(
                activity,
                min_events=min_activity_events,
            )
        )

    # ==============================================================
    # SPATIAL INDICATORS
    # ==============================================================

    if spatial is not None:
        indicators.append(
            build_spatial_concentration_indicator(
                spatial,
                min_localized_events=min_localized_events,
            )
        )

        indicators.append(
            build_spatial_redistribution_indicator(
                spatial,
                min_transitions=min_transitions,
            )
        )

        indicators.append(
            build_repeated_area_indicator(
                spatial,
                min_transitions=min_transitions,
            )
        )

    return tuple(indicators)


# ======================================================================
# VALID INDICATORS
# ======================================================================


def valid_behavior_indicators(
    indicators: Iterable[BehaviorIndicator],
) -> tuple[
    BehaviorIndicator,
    ...,
]:
    """
    Return indicators with sufficient data and a numerical score.
    """

    valid: list[BehaviorIndicator] = []

    for indicator in indicators:
        if not isinstance(
            indicator,
            BehaviorIndicator,
        ):
            raise TypeError(("indicators must contain BehaviorIndicator objects."))

        if (
            indicator.status != BehaviorIndicatorStatus.INSUFFICIENT_DATA
            and indicator.score is not None
        ):
            valid.append(indicator)

    return tuple(valid)
