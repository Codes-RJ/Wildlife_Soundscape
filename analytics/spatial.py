"""
Spatial acoustic-activity analytics.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module analyzes the spatial distribution of localized acoustic
events.

Primary outputs
---------------
SpatialCell
    Rectangular occupancy cell containing event-count statistics.

SpatialTransition
    Observed transition between two consecutive localized acoustic
    event cells.

SpatialSummary
    Overall localization coverage, occupancy map, hotspot and
    transitions.

Scientific interpretation
-------------------------
The system localizes ACOUSTIC EVENTS.

Therefore:

    repeated locations
        ≠ confirmed animal occupancy

and:

    consecutive acoustic locations
        ≠ confirmed movement of one individual animal

A sequence such as:

    cell A -> cell B -> cell C

means:

    "successive detected acoustic events were localized in these cells"

It does NOT prove that:

    "the same animal moved from A to B to C"

unless an external individual-identification method is added.

Expected row fields
-------------------
Recommended normalized rows:

{
    "id": 123,
    "event_time": "2026-08-30T10:15:30",
    "x_m": 0.42,
    "y_m": 0.31,
    "classification_label": "bird_song",
    "classification_confidence": 0.87,
}

Required for occupancy:

    x_m
    y_m

Recommended for transitions:

    event_time

Optional:

    id
    classification_label
    classification_confidence

Rows lacking a valid position remain relevant to localization-coverage
statistics but are excluded from spatial occupancy calculations.

No database access is performed in this module.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from collections import (
    Counter,
    defaultdict,
)

from dataclasses import (
    dataclass,
)

from datetime import (
    datetime,
)

from typing import (
    Any,
    Iterable,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from .models import (
    SpatialCell,
    SpatialSummary,
    SpatialTransition,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_CELL_SIZE_M = (
    0.25
)


DEFAULT_TIMESTAMP_KEY = (
    "event_time"
)


DEFAULT_X_KEY = (
    "x_m"
)


DEFAULT_Y_KEY = (
    "y_m"
)


DEFAULT_CONFIDENCE_KEY = (
    "classification_confidence"
)


DEFAULT_CLASS_KEY = (
    "classification_label"
)


DEFAULT_MAX_GRID_CELLS = (
    10_000
)


DEFAULT_MAX_TRANSITION_GAP_S = (
    300.0
)


UTC_SUFFIX = (
    "Z"
)


# ======================================================================
# INTERNAL SPATIAL EVENT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class _SpatialEvent:
    """
    Internal normalized localized-event representation.
    """

    event_id: int | None

    timestamp: datetime | None

    x_m: float

    y_m: float

    class_label: str | None

    confidence: float | None


# ======================================================================
# ROW ACCESS
# ======================================================================


def _row_value(
    row: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read one field from dict-like or sqlite3.Row-like input.
    """

    getter = getattr(
        row,
        "get",
        None,
    )

    if callable(
        getter
    ):

        try:

            return getter(
                key,
                default,
            )

        except (
            KeyError,
            TypeError,
        ):

            pass

    try:

        return row[
            key
        ]

    except (
        KeyError,
        IndexError,
        TypeError,
    ):

        return (
            default
        )


# ======================================================================
# NUMERIC HELPERS
# ======================================================================


def _finite_float_or_none(
    value: Any,
) -> float | None:
    """
    Convert a value to finite float or return None.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            None
        )

    if not math.isfinite(
        result
    ):

        return (
            None
        )

    return (
        result
    )


def _positive_finite(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require finite numeric value > 0.
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
            f"{name} must be finite and greater than 0."
        )

    return (
        result
    )


def _nonnegative_finite_or_none(
    value: Any,
) -> float | None:
    """
    Convert optional numeric value to finite value >= 0.
    """

    result = (
        _finite_float_or_none(
            value
        )
    )

    if (
        result
        is None
        or result
        < 0.0
    ):

        return (
            None
        )

    return (
        result
    )


def _confidence_or_none(
    value: Any,
) -> float | None:
    """
    Normalize classification confidence into [0, 1].
    """

    result = (
        _nonnegative_finite_or_none(
            value
        )
    )

    if (
        result
        is None
        or result
        > 1.0
    ):

        return (
            None
        )

    return (
        result
    )


# ======================================================================
# INTEGER VALIDATION
# ======================================================================


def _positive_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a positive Python integer.
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


# ======================================================================
# DATETIME PARSING
# ======================================================================


def _parse_optional_datetime(
    value: Any,
) -> datetime | None:
    """
    Parse an optional ISO-8601 datetime.

    Missing/malformed values return None because timestamp is not
    required for occupancy analysis.

    Transition analysis can later exclude events without timestamps.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    if isinstance(
        value,
        datetime,
    ):

        return (
            value
        )

    if not isinstance(
        value,
        str,
    ):

        return (
            None
        )

    text = (
        value.strip()
    )

    if not (
        text
    ):

        return (
            None
        )

    if text.endswith(
        UTC_SUFFIX
    ):

        text = (
            text[
                :-1
            ]
            + "+00:00"
        )

    try:

        return datetime.fromisoformat(
            text
        )

    except ValueError:

        return (
            None
        )


# ======================================================================
# EVENT ID
# ======================================================================


def _normalize_event_id(
    value: Any,
) -> int | None:
    """
    Normalize optional positive database event identifier.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    try:

        result = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            None
        )

    if (
        result
        <= 0
    ):

        return (
            None
        )

    return (
        result
    )


# ======================================================================
# CLASS LABEL
# ======================================================================


def _normalize_class_label(
    value: Any,
) -> str | None:
    """
    Normalize optional classification label.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    result = str(
        value
    ).strip()

    if not (
        result
    ):

        return (
            None
        )

    return (
        result
    )


# ======================================================================
# NORMALIZE SPATIAL EVENTS
# ======================================================================


def normalize_spatial_events(
    rows: Iterable[
        Any
    ],
    *,
    x_key: str = DEFAULT_X_KEY,
    y_key: str = DEFAULT_Y_KEY,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
    confidence_key: str = DEFAULT_CONFIDENCE_KEY,
    class_key: str = DEFAULT_CLASS_KEY,
) -> tuple[
    _SpatialEvent,
    ...,
]:
    """
    Extract usable localized events.

    Rows lacking finite x/y coordinates are excluded.

    This function intentionally does NOT use rows lacking localization
    because occupancy calculations require actual positions.

    Localization coverage is calculated separately using the complete
    row set.
    """

    for name, value in (
        (
            "x_key",
            x_key,
        ),
        (
            "y_key",
            y_key,
        ),
        (
            "timestamp_key",
            timestamp_key,
        ),
        (
            "confidence_key",
            confidence_key,
        ),
        (
            "class_key",
            class_key,
        ),
    ):

        if not isinstance(
            value,
            str,
        ):

            raise TypeError(
                f"{name} must be a string."
            )

        if not (
            value.strip()
        ):

            raise ValueError(
                f"{name} cannot be empty."
            )

    normalized: list[
        _SpatialEvent
    ] = []

    for row in (
        rows
    ):

        x_m = (
            _finite_float_or_none(
                _row_value(
                    row,
                    x_key,
                )
            )
        )

        y_m = (
            _finite_float_or_none(
                _row_value(
                    row,
                    y_key,
                )
            )
        )

        if (
            x_m
            is None
            or y_m
            is None
        ):

            continue

        normalized.append(
            _SpatialEvent(
                event_id=
                    _normalize_event_id(
                        _row_value(
                            row,
                            "id",
                        )
                    ),

                timestamp=
                    _parse_optional_datetime(
                        _row_value(
                            row,
                            timestamp_key,
                        )
                    ),

                x_m=
                    x_m,

                y_m=
                    y_m,

                class_label=
                    _normalize_class_label(
                        _row_value(
                            row,
                            class_key,
                        )
                    ),

                confidence=
                    _confidence_or_none(
                        _row_value(
                            row,
                            confidence_key,
                        )
                    ),
            )
        )

    return tuple(
        normalized
    )


# ======================================================================
# BOUNDS VALIDATION
# ======================================================================


def _validate_bounds(
    bounds: tuple[
        tuple[
            float,
            float,
        ],
        tuple[
            float,
            float,
        ],
    ],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Validate:

        ((x_min, y_min), (x_max, y_max))
    """

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
                "bounds must be "
                "((x_min, y_min), "
                "(x_max, y_max))."
            )
        )

    lower = (
        bounds[
            0
        ]
    )

    upper = (
        bounds[
            1
        ]
    )

    if (
        not isinstance(
            lower,
            tuple,
        )
        or not isinstance(
            upper,
            tuple,
        )
        or len(
            lower
        )
        != 2
        or len(
            upper
        )
        != 2
    ):

        raise TypeError(
            (
                "bounds must contain two "
                "2-D coordinate tuples."
            )
        )

    x_min = (
        _finite_float_or_none(
            lower[
                0
            ]
        )
    )

    y_min = (
        _finite_float_or_none(
            lower[
                1
            ]
        )
    )

    x_max = (
        _finite_float_or_none(
            upper[
                0
            ]
        )
    )

    y_max = (
        _finite_float_or_none(
            upper[
                1
            ]
        )
    )

    if (
        x_min
        is None
        or y_min
        is None
        or x_max
        is None
        or y_max
        is None
    ):

        raise ValueError(
            "bounds must contain finite numbers."
        )

    if (
        x_max
        <= x_min
    ):

        raise ValueError(
            "x_max must exceed x_min."
        )

    if (
        y_max
        <= y_min
    ):

        raise ValueError(
            "y_max must exceed y_min."
        )

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


# ======================================================================
# DERIVE GRID BOUNDS
# ======================================================================


def _derive_grid_bounds(
    events: tuple[
        _SpatialEvent,
        ...,
    ],
    *,
    cell_size_m: float,
) -> tuple[
    float,
    float,
    float,
    float,
] | None:
    """
    Derive grid-aligned bounds covering all localized events.
    """

    if not (
        events
    ):

        return (
            None
        )

    min_x = min(
        event.x_m

        for event
        in events
    )

    max_x = max(
        event.x_m

        for event
        in events
    )

    min_y = min(
        event.y_m

        for event
        in events
    )

    max_y = max(
        event.y_m

        for event
        in events
    )

    x_min = (
        math.floor(
            min_x
            / cell_size_m
        )
        * cell_size_m
    )

    y_min = (
        math.floor(
            min_y
            / cell_size_m
        )
        * cell_size_m
    )

    x_max = (
        math.ceil(
            max_x
            / cell_size_m
        )
        * cell_size_m
    )

    y_max = (
        math.ceil(
            max_y
            / cell_size_m
        )
        * cell_size_m
    )

    # ------------------------------------------------------------------
    # A point exactly on the upper grid line would otherwise belong to
    # the next cell under the standard floor-index rule.
    # ------------------------------------------------------------------

    tolerance = (
        cell_size_m
        * 1e-12
    )

    if (
        x_max
        <= max_x
        + tolerance
    ):

        x_max += (
            cell_size_m
        )

    if (
        y_max
        <= max_y
        + tolerance
    ):

        y_max += (
            cell_size_m
        )

    # ------------------------------------------------------------------
    # Degenerate single-position dimensions still require one cell.
    # ------------------------------------------------------------------

    if (
        x_max
        <= x_min
    ):

        x_max = (
            x_min
            + cell_size_m
        )

    if (
        y_max
        <= y_min
    ):

        y_max = (
            y_min
            + cell_size_m
        )

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


# ======================================================================
# GRID DIMENSIONS
# ======================================================================


def _grid_dimensions(
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    cell_size_m: float,
) -> tuple[
    int,
    int,
]:
    """
    Calculate number of rectangular cells along each axis.
    """

    nx = int(
        math.ceil(
            (
                x_max
                - x_min
            )
            / cell_size_m
        )
    )

    ny = int(
        math.ceil(
            (
                y_max
                - y_min
            )
            / cell_size_m
        )
    )

    return (
        max(
            1,
            nx,
        ),
        max(
            1,
            ny,
        ),
    )


# ======================================================================
# CELL IDENTIFIER
# ======================================================================


def make_cell_id(
    x_index: int,
    y_index: int,
) -> str:
    """
    Produce deterministic spatial-cell identifier.
    """

    if (
        isinstance(
            x_index,
            bool,
        )
        or not isinstance(
            x_index,
            int,
        )
        or x_index
        < 0
    ):

        raise ValueError(
            "x_index must be a non-negative integer."
        )

    if (
        isinstance(
            y_index,
            bool,
        )
        or not isinstance(
            y_index,
            int,
        )
        or y_index
        < 0
    ):

        raise ValueError(
            "y_index must be a non-negative integer."
        )

    return (
        f"x{x_index}_y{y_index}"
    )


# ======================================================================
# POSITION -> CELL INDEX
# ======================================================================


def _position_to_cell_index(
    x_m: float,
    y_m: float,
    *,
    x_min: float,
    y_min: float,
    nx: int,
    ny: int,
    cell_size_m: float,
) -> tuple[
    int,
    int,
]:
    """
    Map one position into a bounded grid cell.
    """

    x_index = int(
        math.floor(
            (
                x_m
                - x_min
            )
            / cell_size_m
        )
    )

    y_index = int(
        math.floor(
            (
                y_m
                - y_min
            )
            / cell_size_m
        )
    )

    # ------------------------------------------------------------------
    # Clamp very small floating-point edge excursions.
    # ------------------------------------------------------------------

    x_index = min(
        max(
            x_index,
            0,
        ),
        nx
        - 1,
    )

    y_index = min(
        max(
            y_index,
            0,
        ),
        ny
        - 1,
    )

    return (
        x_index,
        y_index,
    )


# ======================================================================
# BUILD SPATIAL CELLS
# ======================================================================


def build_spatial_cells(
    rows: Iterable[
        Any
    ],
    *,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
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
    ) = None,
    max_grid_cells: int = DEFAULT_MAX_GRID_CELLS,
) -> tuple[
    SpatialCell,
    ...,
]:
    """
    Build a complete rectangular occupancy grid.

    Empty cells are retained with:

        event_count = 0

    which makes this output directly usable for heatmaps.

    Events outside explicitly provided bounds are excluded.
    """

    cell_size_m = _positive_finite(
        cell_size_m,
        name=
            "cell_size_m",
    )

    max_grid_cells = _positive_int(
        max_grid_cells,
        name=
            "max_grid_cells",
    )

    materialized_rows = tuple(
        rows
    )

    events = normalize_spatial_events(
        materialized_rows
    )

    # ==============================================================
    # RESOLVE BOUNDS
    # ==============================================================

    if (
        bounds
        is None
    ):

        derived = _derive_grid_bounds(
            events,
            cell_size_m=
                cell_size_m,
        )

        if (
            derived
            is None
        ):

            return (
                ()
            )

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = (
            derived
        )

    else:

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = _validate_bounds(
            bounds
        )

    # ==============================================================
    # GRID SIZE
    # ==============================================================

    (
        nx,
        ny,
    ) = _grid_dimensions(
        x_min=
            x_min,

        y_min=
            y_min,

        x_max=
            x_max,

        y_max=
            y_max,

        cell_size_m=
            cell_size_m,
    )

    total_cells = (
        nx
        * ny
    )

    if (
        total_cells
        > max_grid_cells
    ):

        raise ValueError(
            (
                "Requested spatial grid contains "
                f"{total_cells} cells, exceeding "
                f"max_grid_cells={max_grid_cells}. "
                "Increase cell_size_m or use tighter bounds."
            )
        )

    # ==============================================================
    # OCCUPANCY ACCUMULATORS
    # ==============================================================

    counts: Counter[
        tuple[
            int,
            int,
        ]
    ] = Counter()

    confidence_values: dict[
        tuple[
            int,
            int,
        ],
        list[
            float
        ],
    ] = defaultdict(
        list
    )

    for event in (
        events
    ):

        # ----------------------------------------------------------
        # Explicit half-open spatial bounds:
        #
        #     x_min <= x < x_max
        #     y_min <= y < y_max
        # ----------------------------------------------------------

        if not (
            x_min
            <= event.x_m
            < x_max
            and y_min
            <= event.y_m
            < y_max
        ):

            continue

        index = _position_to_cell_index(
            event.x_m,
            event.y_m,

            x_min=
                x_min,

            y_min=
                y_min,

            nx=
                nx,

            ny=
                ny,

            cell_size_m=
                cell_size_m,
        )

        counts[
            index
        ] += (
            1
        )

        if (
            event.confidence
            is not None
        ):

            confidence_values[
                index
            ].append(
                event.confidence
            )

    # ==============================================================
    # BUILD COMPLETE GRID
    # ==============================================================

    cells: list[
        SpatialCell
    ] = []

    for y_index in range(
        ny
    ):

        for x_index in range(
            nx
        ):

            cell_x_min = (
                x_min
                + x_index
                * cell_size_m
            )

            cell_y_min = (
                y_min
                + y_index
                * cell_size_m
            )

            cell_x_max = min(
                x_max,
                cell_x_min
                + cell_size_m,
            )

            cell_y_max = min(
                y_max,
                cell_y_min
                + cell_size_m,
            )

            index = (
                x_index,
                y_index,
            )

            confidences = (
                confidence_values.get(
                    index,
                    [],
                )
            )

            mean_confidence = (
                None

                if not confidences

                else float(
                    math.fsum(
                        confidences
                    )
                    / len(
                        confidences
                    )
                )
            )

            cells.append(
                SpatialCell(
                    cell_id=
                        make_cell_id(
                            x_index,
                            y_index,
                        ),

                    x_min_m=
                        cell_x_min,

                    x_max_m=
                        cell_x_max,

                    y_min_m=
                        cell_y_min,

                    y_max_m=
                        cell_y_max,

                    event_count=
                        int(
                            counts.get(
                                index,
                                0,
                            )
                        ),

                    mean_confidence=
                        mean_confidence,
                )
            )

    return tuple(
        cells
    )


# ======================================================================
# HOTSPOT
# ======================================================================


def find_hotspot_cell(
    cells: Iterable[
        SpatialCell
    ],
) -> str | None:
    """
    Return the cell with the highest acoustic-event count.

    Empty grids or grids containing only zero-count cells return None.

    Tie-breaking
    ------------
    Lexicographically smallest cell_id is selected for deterministic
    results.
    """

    materialized = tuple(
        cells
    )

    for cell in (
        materialized
    ):

        if not isinstance(
            cell,
            SpatialCell,
        ):

            raise TypeError(
                "cells must contain SpatialCell objects."
            )

    occupied = [
        cell

        for cell
        in materialized

        if (
            cell.event_count
            > 0
        )
    ]

    if not (
        occupied
    ):

        return (
            None
        )

    maximum_count = max(
        cell.event_count

        for cell
        in occupied
    )

    return min(
        cell.cell_id

        for cell
        in occupied

        if (
            cell.event_count
            == maximum_count
        )
    )


# ======================================================================
# TRANSITION TIME GAP VALIDATION
# ======================================================================


def _validate_max_transition_gap_s(
    value: float | None,
) -> float | None:
    """
    Validate optional maximum temporal gap between linked events.

    None disables temporal-gap filtering.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            (
                "max_transition_gap_s "
                "must be numeric or None."
            )
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
                "max_transition_gap_s must "
                "be finite and greater than 0."
            )
        )

    return (
        result
    )


# ======================================================================
# SPATIAL TRANSITIONS
# ======================================================================


def build_spatial_transitions(
    rows: Iterable[
        Any
    ],
    *,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
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
    ) = None,
    max_transition_gap_s: float | None = DEFAULT_MAX_TRANSITION_GAP_S,
    same_class_only: bool = False,
) -> tuple[
    SpatialTransition,
    ...,
]:
    """
    Build transitions between consecutive localized acoustic events.

    Only events with valid timestamps participate.

    Parameters
    ----------
    max_transition_gap_s
        Maximum time separation allowed between two consecutive events.

        This prevents linking events separated by very long periods.

        Set None to disable the temporal-gap limit.

    same_class_only
        When True, transitions are formed only when both consecutive
        events have the same non-missing classification label.

        Even then, this does NOT establish individual-animal identity.

    Transition probability
    ----------------------
    For each source cell:

        P(destination | source)

            =
        transition_count(source -> destination)
        ------------------------------------------------
        all accepted transitions leaving source
    """

    cell_size_m = _positive_finite(
        cell_size_m,
        name=
            "cell_size_m",
    )

    max_transition_gap_s = (
        _validate_max_transition_gap_s(
            max_transition_gap_s
        )
    )

    if not isinstance(
        same_class_only,
        bool,
    ):

        raise TypeError(
            "same_class_only must be bool."
        )

    materialized_rows = tuple(
        rows
    )

    events = normalize_spatial_events(
        materialized_rows
    )

    # ==============================================================
    # REQUIRE TIMESTAMPS
    # ==============================================================

    events = tuple(
        event

        for event
        in events

        if (
            event.timestamp
            is not None
        )
    )

    if (
        len(
            events
        )
        < 2
    ):

        return (
            ()
        )

    # ==============================================================
    # TIMEZONE MODE CONSISTENCY
    # ==============================================================

    awareness = {
        (
            event.timestamp.tzinfo
            is not None
            and event.timestamp.utcoffset()
            is not None
        )

        for event
        in events

        if (
            event.timestamp
            is not None
        )
    }

    if (
        len(
            awareness
        )
        > 1
    ):

        raise ValueError(
            (
                "Spatial transition timestamps "
                "cannot mix timezone-aware and "
                "timezone-naive datetime values."
            )
        )

    # ==============================================================
    # ORDER BY TIME
    # ==============================================================

    events = tuple(
        sorted(
            events,
            key=
                lambda event:
                    (
                        event.timestamp,
                        (
                            event.event_id
                            if event.event_id
                            is not None
                            else 0
                        ),
                    ),
        )
    )

    # ==============================================================
    # RESOLVE BOUNDS
    # ==============================================================

    if (
        bounds
        is None
    ):

        derived = _derive_grid_bounds(
            events,
            cell_size_m=
                cell_size_m,
        )

        if (
            derived
            is None
        ):

            return (
                ()
            )

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = (
            derived
        )

    else:

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = _validate_bounds(
            bounds
        )

    (
        nx,
        ny,
    ) = _grid_dimensions(
        x_min=
            x_min,

        y_min=
            y_min,

        x_max=
            x_max,

        y_max=
            y_max,

        cell_size_m=
            cell_size_m,
    )

    # ==============================================================
    # EVENT -> CELL
    # ==============================================================

    indexed_events: list[
        tuple[
            _SpatialEvent,
            str,
        ]
    ] = []

    for event in (
        events
    ):

        if not (
            x_min
            <= event.x_m
            < x_max
            and y_min
            <= event.y_m
            < y_max
        ):

            continue

        (
            x_index,
            y_index,
        ) = _position_to_cell_index(
            event.x_m,
            event.y_m,

            x_min=
                x_min,

            y_min=
                y_min,

            nx=
                nx,

            ny=
                ny,

            cell_size_m=
                cell_size_m,
        )

        indexed_events.append(
            (
                event,

                make_cell_id(
                    x_index,
                    y_index,
                ),
            )
        )

    if (
        len(
            indexed_events
        )
        < 2
    ):

        return (
            ()
        )

    # ==============================================================
    # TRANSITION COUNTS
    # ==============================================================

    transition_counts: Counter[
        tuple[
            str,
            str,
        ]
    ] = Counter()

    outgoing_counts: Counter[
        str
    ] = Counter()

    for index in range(
        len(
            indexed_events
        )
        - 1
    ):

        (
            current_event,
            source_cell,
        ) = (
            indexed_events[
                index
            ]
        )

        (
            next_event,
            destination_cell,
        ) = (
            indexed_events[
                index
                + 1
            ]
        )

        # ----------------------------------------------------------
        # TEMPORAL GAP
        # ----------------------------------------------------------

        time_gap_s = float(
            (
                next_event.timestamp
                - current_event.timestamp
            ).total_seconds()
        )

        if (
            time_gap_s
            < 0.0
        ):

            continue

        if (
            max_transition_gap_s
            is not None
            and time_gap_s
            > max_transition_gap_s
        ):

            continue

        # ----------------------------------------------------------
        # OPTIONAL CLASS CONSISTENCY
        # ----------------------------------------------------------

        if (
            same_class_only
        ):

            if (
                current_event.class_label
                is None
                or next_event.class_label
                is None
                or current_event.class_label
                != next_event.class_label
            ):

                continue

        transition_counts[
            (
                source_cell,
                destination_cell,
            )
        ] += (
            1
        )

        outgoing_counts[
            source_cell
        ] += (
            1
        )

    # ==============================================================
    # BUILD RESULT MODELS
    # ==============================================================

    transitions: list[
        SpatialTransition
    ] = []

    for (
        source_cell,
        destination_cell,
    ) in sorted(
        transition_counts
    ):

        count = int(
            transition_counts[
                (
                    source_cell,
                    destination_cell,
                )
            ]
        )

        outgoing = int(
            outgoing_counts[
                source_cell
            ]
        )

        probability = (
            count
            / outgoing
        )

        transitions.append(
            SpatialTransition(
                source_cell_id=
                    source_cell,

                destination_cell_id=
                    destination_cell,

                transition_count=
                    count,

                probability=
                    probability,
            )
        )

    return tuple(
        transitions
    )


# ======================================================================
# LOCALIZATION COVERAGE
# ======================================================================


def calculate_localization_coverage(
    rows: Iterable[
        Any
    ],
) -> tuple[
    int,
    int,
    float,
]:
    """
    Calculate localization success coverage.

    Returns
    -------
    localized_event_count
    total_event_count
    localization_coverage

    where:

        coverage =
            localized events / all events
    """

    materialized = tuple(
        rows
    )

    total_event_count = (
        len(
            materialized
        )
    )

    if (
        total_event_count
        == 0
    ):

        return (
            0,
            0,
            0.0,
        )

    localized_event_count = (
        0
    )

    for row in (
        materialized
    ):

        x_m = (
            _finite_float_or_none(
                _row_value(
                    row,
                    DEFAULT_X_KEY,
                )
            )
        )

        y_m = (
            _finite_float_or_none(
                _row_value(
                    row,
                    DEFAULT_Y_KEY,
                )
            )
        )

        if (
            x_m
            is not None
            and y_m
            is not None
        ):

            localized_event_count += (
                1
            )

    localization_coverage = (
        localized_event_count
        / total_event_count
    )

    return (
        localized_event_count,
        total_event_count,
        localization_coverage,
    )


# ======================================================================
# OCCUPIED CELL COUNT
# ======================================================================


def occupied_cell_count(
    cells: Iterable[
        SpatialCell
    ],
) -> int:
    """
    Count cells containing at least one localized acoustic event.
    """

    count = (
        0
    )

    for cell in (
        cells
    ):

        if not isinstance(
            cell,
            SpatialCell,
        ):

            raise TypeError(
                "cells must contain SpatialCell objects."
            )

        if (
            cell.event_count
            > 0
        ):

            count += (
                1
            )

    return (
        count
    )


# ======================================================================
# SPATIAL OCCUPANCY ENTROPY
# ======================================================================


def spatial_occupancy_entropy(
    cells: Iterable[
        SpatialCell
    ],
    *,
    normalized: bool = True,
) -> float:
    """
    Calculate Shannon entropy of localized-event spatial distribution.

    Interpretation
    --------------
    Lower entropy:
        events are concentrated into fewer cells.

    Higher entropy:
        events are distributed more evenly across occupied cells.

    normalized=True
        returns a value in [0, 1] when at least two occupied cells
        exist.

    This describes acoustic-event spatial dispersion, not biological
    home-range entropy.
    """

    if not isinstance(
        normalized,
        bool,
    ):

        raise TypeError(
            "normalized must be bool."
        )

    materialized = tuple(
        cells
    )

    counts: list[
        int
    ] = []

    for cell in (
        materialized
    ):

        if not isinstance(
            cell,
            SpatialCell,
        ):

            raise TypeError(
                "cells must contain SpatialCell objects."
            )

        if (
            cell.event_count
            > 0
        ):

            counts.append(
                cell.event_count
            )

    if (
        len(
            counts
        )
        <= 1
    ):

        return (
            0.0
        )

    total = float(
        sum(
            counts
        )
    )

    probabilities = [
        count
        / total

        for count
        in counts
    ]

    entropy = -math.fsum(
        probability
        * math.log(
            probability
        )

        for probability
        in probabilities
    )

    if not (
        normalized
    ):

        return (
            entropy
        )

    maximum_entropy = math.log(
        len(
            probabilities
        )
    )

    if (
        maximum_entropy
        <= 0.0
    ):

        return (
            0.0
        )

    return min(
        1.0,
        max(
            0.0,
            entropy
            / maximum_entropy,
        ),
    )


# ======================================================================
# BUILD COMPLETE SPATIAL SUMMARY
# ======================================================================


def build_spatial_summary(
    rows: Iterable[
        Any
    ],
    *,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
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
    ) = None,
    max_grid_cells: int = DEFAULT_MAX_GRID_CELLS,
    max_transition_gap_s: float | None = DEFAULT_MAX_TRANSITION_GAP_S,
    same_class_transitions_only: bool = False,
) -> SpatialSummary:
    """
    Build the complete spatial analytics result.

    Processing
    ----------
    complete event rows
            ↓
    localization coverage
            ↓
    valid x/y rows
            ↓
    rectangular occupancy grid
            ↓
    hotspot
            ↓
    ordered acoustic-location transitions
            ↓
    SpatialSummary
    """

    materialized_rows = tuple(
        rows
    )

    # ==============================================================
    # COVERAGE
    # ==============================================================

    (
        localized_event_count,
        total_event_count,
        localization_coverage,
    ) = calculate_localization_coverage(
        materialized_rows
    )

    # ==============================================================
    # NO LOCALIZED EVENTS
    # ==============================================================

    if (
        localized_event_count
        == 0
    ):

        return SpatialSummary(
            localized_event_count=
                0,

            total_event_count=
                total_event_count,

            localization_coverage=
                localization_coverage,

            cells=
                (),

            transitions=
                (),

            hotspot_cell_id=
                None,
        )

    # ==============================================================
    # CELLS
    # ==============================================================

    cells = build_spatial_cells(
        materialized_rows,

        cell_size_m=
            cell_size_m,

        bounds=
            bounds,

        max_grid_cells=
            max_grid_cells,
    )

    # ==============================================================
    # HOTSPOT
    # ==============================================================

    hotspot_cell_id = (
        find_hotspot_cell(
            cells
        )
    )

    # ==============================================================
    # TRANSITIONS
    # ==============================================================

    transitions = (
        build_spatial_transitions(
            materialized_rows,

            cell_size_m=
                cell_size_m,

            bounds=
                bounds,

            max_transition_gap_s=
                max_transition_gap_s,

            same_class_only=
                same_class_transitions_only,
        )
    )

    # ==============================================================
    # RESULT
    # ==============================================================

    return SpatialSummary(
        localized_event_count=
            localized_event_count,

        total_event_count=
            total_event_count,

        localization_coverage=
            localization_coverage,

        cells=
            cells,

        transitions=
            transitions,

        hotspot_cell_id=
            hotspot_cell_id,
    )