"""
High-level research analytics orchestration.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module combines the independent analytics subsystems into one
coherent research-facing report.

Processing chain
----------------

    normalized event rows
            │
            ├── activity.py
            │       ├── temporal bins
            │       ├── event counts
            │       ├── durations
            │       └── class summaries
            │
            ├── spatial.py
            │       ├── localization coverage
            │       ├── occupancy grid
            │       ├── hotspot
            │       └── spatial transitions
            │
            └── behavior.py
                    └── conservative derived indicators


    regular environmental observation bins
            │
            └── environmental.py
                    └── Spearman associations

                            ↓

                ResearchAnalyticsReport


Important data distinction
--------------------------
Two different input datasets may be required.

1. Event rows
   ---------
   One row per detected acoustic event.

   Used for:

       activity analysis
       spatial analysis
       behavioral indicators


2. Environmental observation rows
   -------------------------------
   Regular temporal bins containing environmental values and activity
   metrics, including bins with zero events.

   Used for:

       environmental association analysis

Environmental association analysis should NOT normally be performed
using event-only rows because periods with zero detected activity would
be absent.

Scientific scope
----------------
The generated report describes patterns in detected acoustic events.

It does not establish:

    animal abundance
    individual identity
    causation
    verified biological behavior
    confirmed movement trajectories

No database access is performed in this module.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from collections.abc import (
    Iterable,
    Mapping,
)

from datetime import (
    datetime,
    timezone,
)

from typing import (
    Any,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from .activity import (
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TIMESTAMP_KEY,
    build_activity_bins,
    build_activity_summary,
)

from .behavior import (
    DEFAULT_MIN_ACTIVITY_EVENTS,
    DEFAULT_MIN_LOCALIZED_EVENTS,
    DEFAULT_MIN_TRANSITIONS,
    build_behavior_indicators,
)

from .environmental import (
    DEFAULT_ALPHA,
    DEFAULT_ENVIRONMENTAL_FIELDS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_NEUTRAL_THRESHOLD,
    DEFAULT_RESPONSE_FIELDS,
    build_environmental_associations,
)

from .models import (
    AnalyticsTimeWindow,
    BehaviorIndicator,
    EnvironmentalAssociation,
    ResearchAnalyticsReport,
)

from .spatial import (
    DEFAULT_CELL_SIZE_M,
    DEFAULT_MAX_GRID_CELLS,
    DEFAULT_MAX_TRANSITION_GAP_S,
    SpatialSummary,
    build_spatial_summary,
)


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _require_positive_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a Python integer > 0.
    """

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):

        raise TypeError(
            f"{name} must be an integer."
        )

    if (
        value
        <= 0
    ):

        raise ValueError(
            f"{name} must be greater than 0."
        )

    return (
        int(
            value
        )
    )


def _require_positive_finite(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require a finite numeric value > 0.
    """

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            f"{name} must be numeric."
        ) from exc

    if (
        not math.isfinite(
            result
        )
        or result
        <= 0.0
    ):

        raise ValueError(
            (
                f"{name} must be finite "
                "and greater than 0."
            )
        )

    return (
        result
    )


def _require_probability_open(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require a finite value in (0, 1).
    """

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            f"{name} must be numeric."
        ) from exc

    if (
        not math.isfinite(
            result
        )
        or not (
            0.0
            < result
            < 1.0
        )
    ):

        raise ValueError(
            f"{name} must be in (0, 1)."
        )

    return (
        result
    )


def _require_probability_closed(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require a finite value in [0, 1].
    """

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            f"{name} must be numeric."
        ) from exc

    if (
        not math.isfinite(
            result
        )
        or not (
            0.0
            <= result
            <= 1.0
        )
    ):

        raise ValueError(
            f"{name} must be in [0, 1]."
        )

    return (
        result
    )


# ======================================================================
# STRING VALIDATION
# ======================================================================


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

        raise TypeError(
            f"{name} must be a string."
        )

    result = (
        value.strip()
    )

    if not (
        result
    ):

        raise ValueError(
            f"{name} cannot be empty."
        )

    return (
        result
    )


# ======================================================================
# OPTIONAL BOUNDS VALIDATION
# ======================================================================


def _normalize_spatial_bounds(
    bounds: (
        tuple[
            tuple[
                float,
                float,
            ],
            tuple[
                float,
                float,
            ],
        ]
        | None
    ),
):
    """
    Perform lightweight service-level spatial-bounds validation.

    Detailed geometric validation remains the responsibility of
    spatial.py.
    """

    if (
        bounds
        is None
    ):

        return (
            None
        )

    if (
        not isinstance(
            bounds,
            tuple,
        )
        or len(
            bounds
        )
        != 2
    ):

        raise TypeError(
            (
                "spatial_bounds must be "
                "((x_min, y_min), "
                "(x_max, y_max)) or None."
            )
        )

    return (
        bounds
    )


# ======================================================================
# FIELD-MAPPING NORMALIZATION
# ======================================================================


def _normalize_optional_mapping(
    value: Mapping[
        str,
        str,
    ]
    | None,
    *,
    default: Mapping[
        str,
        str,
    ],
    name: str,
) -> Mapping[
    str,
    str,
]:
    """
    Resolve optional field mappings.

    Detailed mapping validation occurs inside environmental.py.
    """

    if (
        value
        is None
    ):

        return (
            default
        )

    if not isinstance(
        value,
        Mapping,
    ):

        raise TypeError(
            f"{name} must be a mapping or None."
        )

    return (
        value
    )


# ======================================================================
# REPORT WARNING HELPERS
# ======================================================================


def _activity_warnings(
    *,
    total_events: int,
    min_activity_events: int,
) -> list[
    str
]:
    """
    Generate activity-related quality warnings.
    """

    warnings: list[
        str
    ] = []

    if (
        total_events
        == 0
    ):

        warnings.append(
            (
                "No acoustic events were available "
                "for temporal activity analysis."
            )
        )

    elif (
        total_events
        < min_activity_events
    ):

        warnings.append(
            (
                "Temporal and class-distribution "
                "indicators are based on fewer than "
                f"{min_activity_events} acoustic events "
                "and should be interpreted cautiously."
            )
        )

    return (
        warnings
    )


def _spatial_warnings(
    spatial: SpatialSummary,
    *,
    min_localized_events: int,
    min_transitions: int,
) -> list[
    str
]:
    """
    Generate localization/spatial-analysis warnings.
    """

    warnings: list[
        str
    ] = []

    if (
        spatial.total_event_count
        > 0
        and spatial.localized_event_count
        == 0
    ):

        warnings.append(
            (
                "No detected acoustic events contained "
                "usable localization coordinates; "
                "spatial analytics are unavailable."
            )
        )

        return (
            warnings
        )

    if (
        spatial.localized_event_count
        < min_localized_events
        and spatial.localized_event_count
        > 0
    ):

        warnings.append(
            (
                "Spatial indicators are based on only "
                f"{spatial.localized_event_count} "
                "localized events, below the configured "
                f"minimum of {min_localized_events}."
            )
        )

    if (
        spatial.total_event_count
        > 0
        and spatial.localization_coverage
        < 0.50
    ):

        warnings.append(
            (
                "Localization coverage is below 50%; "
                "the spatial distribution may not "
                "represent the full detected acoustic "
                "event set."
            )
        )

    transition_count = sum(
        transition.transition_count

        for transition
        in spatial.transitions
    )

    if (
        spatial.localized_event_count
        >= 2
        and transition_count
        < min_transitions
    ):

        warnings.append(
            (
                "Too few accepted consecutive spatial "
                "transitions are available for robust "
                "redistribution or repeated-area "
                "indicators."
            )
        )

    return (
        warnings
    )


def _environmental_warnings(
    associations: tuple[
        EnvironmentalAssociation,
        ...,
    ],
    *,
    environmental_rows_supplied: bool,
) -> list[
    str
]:
    """
    Generate environmental-analysis warnings.
    """

    warnings: list[
        str
    ] = []

    if not (
        environmental_rows_supplied
    ):

        warnings.append(
            (
                "Environmental association analysis "
                "was not performed because regular "
                "environmental observation-bin rows "
                "were not supplied."
            )
        )

        return (
            warnings
        )

    if not (
        associations
    ):

        warnings.append(
            (
                "Environmental observation data were "
                "supplied, but no environmental "
                "associations were produced."
            )
        )

        return (
            warnings
        )

    computed = [
        association

        for association
        in associations

        if (
            association.coefficient
            is not None
        )
    ]

    if not (
        computed
    ):

        warnings.append(
            (
                "Environmental variables did not "
                "contain enough paired, varying "
                "observations to calculate any "
                "Spearman associations."
            )
        )

    return (
        warnings
    )


# ======================================================================
# BEHAVIOR WARNINGS
# ======================================================================


def _behavior_warnings(
    indicators: tuple[
        BehaviorIndicator,
        ...,
    ],
) -> list[
    str
]:
    """
    Add one scientific interpretation warning when behavior-related
    indicators are present.
    """

    if not (
        indicators
    ):

        return (
            []
        )

    return [
        (
            "Behavior-related outputs are derived "
            "acoustic indicators and must not be "
            "interpreted as confirmed ethological "
            "states or individual-animal trajectories."
        )
    ]


# ======================================================================
# DUPLICATE WARNING REMOVAL
# ======================================================================


def _deduplicate_warnings(
    warnings: Iterable[
        str
    ],
) -> tuple[
    str,
    ...,
]:
    """
    Remove duplicate warnings while preserving order.
    """

    result: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for warning in (
        warnings
    ):

        warning = (
            _require_nonempty_string(
                warning,
                name=
                    "warning",
            )
        )

        if (
            warning
            in seen
        ):

            continue

        seen.add(
            warning
        )

        result.append(
            warning
        )

    return tuple(
        result
    )


# ======================================================================
# REPORT WINDOW RESOLUTION
# ======================================================================


def _resolve_report_window(
    activity_window: AnalyticsTimeWindow,
) -> AnalyticsTimeWindow:
    """
    Preserve the activity subsystem's temporal window as the canonical
    report window.

    A separate helper is retained so this policy can later evolve
    without changing the report-building interface.
    """

    if not isinstance(
        activity_window,
        AnalyticsTimeWindow,
    ):

        raise TypeError(
            (
                "activity_window must be an "
                "AnalyticsTimeWindow."
            )
        )

    return (
        activity_window
    )


# ======================================================================
# RESEARCH REPORT BUILDER
# ======================================================================


def build_research_analytics_report(
    event_rows: Iterable[
        Any
    ],
    *,
    environmental_rows: (
        Iterable[
            Any
        ]
        | None
    ) = None,

    # ------------------------------------------------------------------
    # TEMPORAL ANALYSIS
    # ------------------------------------------------------------------

    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    start: datetime | str | None = None,
    end: datetime | str | None = None,

    # ------------------------------------------------------------------
    # SPATIAL ANALYSIS
    # ------------------------------------------------------------------

    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    spatial_bounds: (
        tuple[
            tuple[
                float,
                float,
            ],
            tuple[
                float,
                float,
            ],
        ]
        | None
    ) = None,
    max_grid_cells: int = DEFAULT_MAX_GRID_CELLS,
    max_transition_gap_s: float | None = DEFAULT_MAX_TRANSITION_GAP_S,
    same_class_transitions_only: bool = False,

    # ------------------------------------------------------------------
    # ENVIRONMENTAL ANALYSIS
    # ------------------------------------------------------------------

    environmental_fields: (
        Mapping[
            str,
            str,
        ]
        | None
    ) = None,

    response_fields: (
        Mapping[
            str,
            str,
        ]
        | None
    ) = None,

    alpha: float = DEFAULT_ALPHA,
    environmental_min_samples: int = DEFAULT_MIN_SAMPLES,
    neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,

    # ------------------------------------------------------------------
    # BEHAVIOR-INDICATOR MINIMUMS
    # ------------------------------------------------------------------

    min_activity_events: int = DEFAULT_MIN_ACTIVITY_EVENTS,
    min_localized_events: int = DEFAULT_MIN_LOCALIZED_EVENTS,
    min_transitions: int = DEFAULT_MIN_TRANSITIONS,

    # ------------------------------------------------------------------
    # REPORT METADATA
    # ------------------------------------------------------------------

    generated_at: datetime | None = None,
) -> ResearchAnalyticsReport:
    """
    Build the complete research analytics report.

    Parameters
    ----------
    event_rows
        One row per detected acoustic event.

        Required for temporal analysis:
            event_time

        Recommended:
            id
            start_sample
            end_sample
            duration_s
            classification_label
            classification_confidence
            x_m
            y_m


    environmental_rows
        Regular temporal observation bins.

        These should include periods with zero detected events.

        Expected fields normally include:

            temperature_c
            humidity_percent
            pressure_hpa
            event_count
            active_duration_s
            event_rate_per_hour


    generated_at
        Optional explicit report-generation timestamp.

        Defaults to current UTC time.


    Returns
    -------
    ResearchAnalyticsReport
        Complete typed analytics result.
    """

    # ==================================================================
    # PARAMETER VALIDATION
    # ==================================================================

    sample_rate = _require_positive_int(
        sample_rate,
        name=
            "sample_rate",
    )

    bucket_seconds = _require_positive_int(
        bucket_seconds,
        name=
            "bucket_seconds",
    )

    timestamp_key = _require_nonempty_string(
        timestamp_key,
        name=
            "timestamp_key",
    )

    cell_size_m = _require_positive_finite(
        cell_size_m,
        name=
            "cell_size_m",
    )

    max_grid_cells = _require_positive_int(
        max_grid_cells,
        name=
            "max_grid_cells",
    )

    spatial_bounds = (
        _normalize_spatial_bounds(
            spatial_bounds
        )
    )

    if (
        max_transition_gap_s
        is not None
    ):

        max_transition_gap_s = (
            _require_positive_finite(
                max_transition_gap_s,
                name=
                    "max_transition_gap_s",
            )
        )

    if not isinstance(
        same_class_transitions_only,
        bool,
    ):

        raise TypeError(
            (
                "same_class_transitions_only "
                "must be bool."
            )
        )

    alpha = (
        _require_probability_open(
            alpha,
            name=
                "alpha",
        )
    )

    neutral_threshold = (
        _require_probability_closed(
            neutral_threshold,
            name=
                "neutral_threshold",
        )
    )

    environmental_min_samples = (
        _require_positive_int(
            environmental_min_samples,
            name=
                "environmental_min_samples",
        )
    )

    if (
        environmental_min_samples
        < 3
    ):

        raise ValueError(
            (
                "environmental_min_samples "
                "must be at least 3."
            )
        )

    min_activity_events = (
        _require_positive_int(
            min_activity_events,
            name=
                "min_activity_events",
        )
    )

    min_localized_events = (
        _require_positive_int(
            min_localized_events,
            name=
                "min_localized_events",
        )
    )

    min_transitions = (
        _require_positive_int(
            min_transitions,
            name=
                "min_transitions",
        )
    )

    environmental_fields = (
        _normalize_optional_mapping(
            environmental_fields,
            default=
                DEFAULT_ENVIRONMENTAL_FIELDS,
            name=
                "environmental_fields",
        )
    )

    response_fields = (
        _normalize_optional_mapping(
            response_fields,
            default=
                DEFAULT_RESPONSE_FIELDS,
            name=
                "response_fields",
        )
    )

    # ==================================================================
    # REPORT GENERATION TIME
    # ==================================================================

    if (
        generated_at
        is None
    ):

        generated_at = datetime.now(
            timezone.utc
        )

    elif not isinstance(
        generated_at,
        datetime,
    ):

        raise TypeError(
            "generated_at must be datetime or None."
        )

    # ==================================================================
    # MATERIALIZE EVENT DATA ONCE
    # ==================================================================

    materialized_events = tuple(
        event_rows
    )

    # ==================================================================
    # TEMPORAL ACTIVITY SUMMARY
    # ==================================================================

    activity = build_activity_summary(
        materialized_events,

        sample_rate=
            sample_rate,

        timestamp_key=
            timestamp_key,

        start=
            start,

        end=
            end,
    )

    # ==================================================================
    # TEMPORAL BINS
    # ==================================================================

    activity_bins = build_activity_bins(
        materialized_events,

        bucket_seconds=
            bucket_seconds,

        sample_rate=
            sample_rate,

        timestamp_key=
            timestamp_key,

        start=
            start,

        end=
            end,
    )

    # ==================================================================
    # SPATIAL ANALYSIS
    # ==================================================================

    spatial = build_spatial_summary(
        materialized_events,

        cell_size_m=
            cell_size_m,

        bounds=
            spatial_bounds,

        max_grid_cells=
            max_grid_cells,

        max_transition_gap_s=
            max_transition_gap_s,

        same_class_transitions_only=
            same_class_transitions_only,
    )

    # ==================================================================
    # ENVIRONMENTAL ASSOCIATIONS
    # ==================================================================

    environmental_rows_supplied = (
        environmental_rows
        is not None
    )

    environmental_associations: tuple[
        EnvironmentalAssociation,
        ...,
    ]

    if (
        environmental_rows
        is None
    ):

        environmental_associations = (
            ()
        )

    else:

        materialized_environmental_rows = tuple(
            environmental_rows
        )

        environmental_associations = (
            build_environmental_associations(
                materialized_environmental_rows,

                environmental_fields=
                    environmental_fields,

                response_fields=
                    response_fields,

                alpha=
                    alpha,

                min_samples=
                    environmental_min_samples,

                neutral_threshold=
                    neutral_threshold,
            )
        )

    # ==================================================================
    # CONSERVATIVE BEHAVIOR INDICATORS
    # ==================================================================

    behavior_indicators = (
        build_behavior_indicators(
            activity=
                activity,

            activity_bins=
                activity_bins,

            spatial=
                spatial,

            min_activity_events=
                min_activity_events,

            min_localized_events=
                min_localized_events,

            min_transitions=
                min_transitions,
        )
    )

    # ==================================================================
    # WARNINGS
    # ==================================================================

    warnings: list[
        str
    ] = []

    warnings.extend(
        _activity_warnings(
            total_events=
                activity.total_events,

            min_activity_events=
                min_activity_events,
        )
    )

    warnings.extend(
        _spatial_warnings(
            spatial,

            min_localized_events=
                min_localized_events,

            min_transitions=
                min_transitions,
        )
    )

    warnings.extend(
        _environmental_warnings(
            environmental_associations,

            environmental_rows_supplied=
                environmental_rows_supplied,
        )
    )

    warnings.extend(
        _behavior_warnings(
            behavior_indicators
        )
    )

    # ==================================================================
    # REPORT WINDOW
    # ==================================================================

    report_window = (
        _resolve_report_window(
            activity.window
        )
    )

    # ==================================================================
    # FINAL REPORT
    # ==================================================================

    return ResearchAnalyticsReport(
        generated_at=
            generated_at,

        window=
            report_window,

        activity=
            activity,

        environmental_associations=
            environmental_associations,

        spatial=
            spatial,

        behavior_indicators=
            behavior_indicators,

        warnings=
            _deduplicate_warnings(
                warnings
            ),
    )


# ======================================================================
# REPORT -> DICTIONARY
# ======================================================================


def build_research_analytics_dict(
    event_rows: Iterable[
        Any
    ],
    *,
    environmental_rows: (
        Iterable[
            Any
        ]
        | None
    ) = None,
    **kwargs,
) -> dict[
    str,
    Any,
]:
    """
    Convenience wrapper returning the fully serialized report.

    Useful for:

        JSON export
        API responses
        dashboard data transfer
        research metrics export

    The typed ResearchAnalyticsReport remains the preferred internal
    representation.
    """

    report = build_research_analytics_report(
        event_rows,

        environmental_rows=
            environmental_rows,

        **kwargs,
    )

    return (
        report.to_dict()
    )