"""
Reusable Plotly visualization builders.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module converts normalized dashboard/research data into Plotly
figures.

Primary visualizations
----------------------
    recent acoustic-event timeline
    temporal activity timeline
    acoustic-class distribution
    environmental time series
    environmental-association matrix
    accepted localized-event scatter map
    spatial occupancy heatmap
    behavior-indicator scores

Design principle
----------------
This module creates figures only.

It does NOT:

    access SQLite
    call EventDatabase
    run analytics
    render Streamlit widgets
    decode WAV files
    modify source data

Localization policy
-------------------
A raw row is plotted as a successful localization only when:

    localization_success == True

and both:

    x_m
    y_m

are finite.

Numeric coordinates alone are not sufficient because a solver may retain
a candidate position even when the localization result fails acceptance
criteria.
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from collections import (
    defaultdict,
)

from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)

from datetime import (
    datetime,
)

from typing import (
    Any,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import plotly.graph_objects as go

from wildlife_soundscape.dashboard.palette import apply_chart_theme, class_color


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.analytics.models import (
    ActivityBin,
    ActivitySummary,
    BehaviorIndicator,
    EnvironmentalAssociation,
    SpatialSummary,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_MAX_PLOT_POINTS = 5000


UTC_SUFFIX = "Z"


_MISSING = object()


# ======================================================================
# ROW ACCESS
# ======================================================================


def _row_value(
    row: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read one value from dict-like or sqlite3.Row-like input.
    """

    getter = getattr(
        row,
        "get",
        None,
    )

    if callable(getter):
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
        return row[key]

    except (
        KeyError,
        IndexError,
        TypeError,
    ):
        return default


# ======================================================================
# NUMERIC NORMALIZATION
# ======================================================================


def _finite_float_or_none(
    value: Any,
) -> float | None:
    """
    Convert optional numeric input to finite float.
    """

    if value is None:
        return None

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(result):
        return None

    return result


# ======================================================================
# BOOLEAN NORMALIZATION
# ======================================================================


def _boolean_or_none(
    value: Any,
) -> bool | None:
    """
    Normalize common persisted/exported boolean representations.

    Accepted true values:

        True
        1
        1.0
        "1"
        "true"

    Accepted false values:

        False
        0
        0.0
        "0"
        "false"

    Invalid or ambiguous values return None.
    """

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        if value == 1:
            return True

        if value == 0:
            return False

        return None

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(value):
            return None

        if value == 1.0:
            return True

        if value == 0.0:
            return False

        return None

    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().lower()

        if normalized in {
            "1",
            "true",
        }:
            return True

        if normalized in {
            "0",
            "false",
        }:
            return False

    return None


# ======================================================================
# LOCALIZATION ACCEPTANCE
# ======================================================================


def _localization_is_accepted(
    row: Any,
) -> bool:
    """
    Return True only for an explicitly successful localization.

    Rows without ``localization_success`` are treated as unverified and
    are not presented as successful localization results.
    """

    raw_value = _row_value(
        row,
        "localization_success",
        _MISSING,
    )

    if raw_value is _MISSING:
        return False

    return _boolean_or_none(raw_value) is True


# ======================================================================
# EVENT IDENTIFIER
# ======================================================================


def _event_identifier(
    row: Any,
) -> Any:
    """
    Resolve event identity from database or export-style rows.

    Preferred:

        id

    Fallback:

        event_id
    """

    value = _row_value(
        row,
        "id",
        _MISSING,
    )

    if value is not _MISSING and value is not None:
        return value

    return _row_value(
        row,
        "event_id",
    )


# ======================================================================
# DATETIME NORMALIZATION
# ======================================================================


def _parse_datetime_or_none(
    value: Any,
) -> datetime | None:
    """
    Convert datetime or ISO-8601 text to datetime.

    Invalid/missing timestamps return None.
    """

    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        return None

    text = value.strip()

    if not (text):
        return None

    if text.endswith(UTC_SUFFIX):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text)

    except ValueError:
        return None


# ======================================================================
# MAX POINT VALIDATION
# ======================================================================


def _validate_max_points(
    value: int,
) -> int:
    """
    Require a positive Python integer plot-point limit.
    """

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        int,
    ):
        raise TypeError("max_points must be an integer.")

    if value <= 0:
        raise ValueError("max_points must be greater than 0.")

    return int(value)


# ======================================================================
# SIMPLE DECIMATION
# ======================================================================


def _decimate_sequence(
    values: Sequence[Any],
    *,
    max_points: int,
) -> list[Any]:
    """
    Reduce a sequence to at most max_points while preserving order.

    Scientific calculations must always use the original complete
    dataset.

    This helper is only for visualization.
    """

    max_points = _validate_max_points(max_points)

    value_count = len(values)

    if value_count <= max_points:
        return list(values)

    step = value_count / max_points

    indices = [
        min(
            value_count - 1,
            int(index * step),
        )
        for index in range(max_points)
    ]

    return [values[index] for index in indices]


# ======================================================================
# NODE POSITION NORMALIZATION
# ======================================================================


def _validated_node_positions(
    node_positions: (
        Mapping[
            int,
            tuple[
                float,
                float,
            ],
        ]
        | None
    ),
) -> tuple[
    list[int],
    list[float],
    list[float],
]:
    """
    Normalize microphone-node coordinates for plotting.
    """

    if node_positions is None:
        return (
            [],
            [],
            [],
        )

    node_ids: list[int] = []

    node_x: list[float] = []

    node_y: list[float] = []

    for node_id in sorted(node_positions):
        position = node_positions[node_id]

        try:
            raw_x_value, raw_y_value = position

        except Exception as exc:
            raise ValueError(
                ("Each node position must contain exactly two coordinates.")
            ) from exc

        x_value = _finite_float_or_none(raw_x_value)

        y_value = _finite_float_or_none(raw_y_value)

        if x_value is None or y_value is None:
            raise ValueError(("Node positions must contain finite coordinates."))

        try:
            normalized_node_id = int(node_id)

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(("Node IDs must be integer-compatible.")) from exc

        node_ids.append(normalized_node_id)

        node_x.append(x_value)

        node_y.append(y_value)

    return (
        node_ids,
        node_x,
        node_y,
    )


# ======================================================================
# MICROPHONE TRACE
# ======================================================================


def _add_microphone_trace(
    figure: go.Figure,
    *,
    node_positions: (
        Mapping[
            int,
            tuple[
                float,
                float,
            ],
        ]
        | None
    ),
) -> None:
    """
    Add microphone-array positions to a spatial figure.
    """

    (
        node_ids,
        node_x,
        node_y,
    ) = _validated_node_positions(node_positions)

    if not (node_ids):
        return

    if len(node_ids) == 3:
        figure.add_trace(go.Scatter(
            x=node_x + [node_x[0]], y=node_y + [node_y[0]],
            mode="lines", name="Array boundary",
            line={"color": "#111111", "width": 2, "dash": "dash"},
            hoverinfo="skip",
        ))

    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            name="Microphones",
            text=[f"Node {node_id}" for node_id in node_ids],
            textposition="top center",
            marker=dict(
                symbol="diamond",
                size=13,
                color="#111111",
            ),
            hovertemplate=("%{text}<br>X: %{x:.3f} m<br>Y: %{y:.3f} m<extra></extra>"),
        )
    )


# ======================================================================
# COMMON FIGURE LAYOUT
# ======================================================================


def _apply_common_layout(
    figure: go.Figure,
    *,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """
    Apply monochrome chart surfaces with semantic data colors.
    """

    apply_chart_theme(figure)
    figure.update_layout(
        title=title,
        margin=dict(
            l=40,
            r=30,
            t=60,
            b=40,
        ),
        hovermode="closest",
        legend_title_text="",
    )

    if x_title is not None:
        figure.update_xaxes(title_text=x_title)

    if y_title is not None:
        figure.update_yaxes(title_text=y_title)

    if height is not None:
        figure.update_layout(height=height)

    return figure


# ======================================================================
# EMPTY FIGURE
# ======================================================================


def empty_figure(
    *,
    title: str,
    message: str = "No data available.",
    height: int = 350,
) -> go.Figure:
    """
    Return a valid Plotly figure containing a centered status message.
    """

    if not isinstance(
        title,
        str,
    ):
        raise TypeError("title must be a string.")

    if not isinstance(
        message,
        str,
    ):
        raise TypeError("message must be a string.")

    figure = go.Figure()

    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
    )

    figure.update_xaxes(visible=False)

    figure.update_yaxes(visible=False)

    return _apply_common_layout(
        figure,
        title=title,
        height=height,
    )


# ======================================================================
# RECENT EVENT TIMELINE
# ======================================================================


def build_recent_event_timeline(
    rows: Iterable[Any],
    *,
    max_points: int = DEFAULT_MAX_PLOT_POINTS,
) -> go.Figure:
    """
    Plot persisted acoustic events over time.

    Preferred timestamp field:

        event_time

    Fallback:

        created_at

    Localization coordinates are displayed only when
    ``localization_success`` is explicitly true.
    """

    max_points = _validate_max_points(max_points)

    materialized = list(rows)

    points: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for row in materialized:
        timestamp = _parse_datetime_or_none(
            _row_value(
                row,
                "event_time",
                _row_value(
                    row,
                    "created_at",
                ),
            )
        )

        if timestamp is None:
            continue

        label = _row_value(
            row,
            "classification_label",
            "unclassified",
        )

        if label is None or not str(label).strip():
            label = "unclassified"

        confidence = _finite_float_or_none(
            _row_value(
                row,
                "classification_confidence",
            )
        )

        rms = _finite_float_or_none(
            _row_value(
                row,
                "rms",
            )
        )

        if _localization_is_accepted(row):
            x_m = _finite_float_or_none(
                _row_value(
                    row,
                    "x_m",
                )
            )

            y_m = _finite_float_or_none(
                _row_value(
                    row,
                    "y_m",
                )
            )

        else:
            x_m = None

            y_m = None

        points.append(
            {
                "time": timestamp,
                "label": str(label),
                "event_id": _event_identifier(row),
                "confidence": confidence,
                "rms": rms,
                "x_m": x_m,
                "y_m": y_m,
            }
        )

    points = _decimate_sequence(
        points,
        max_points=max_points,
    )

    if not (points):
        return empty_figure(
            title="Recent Acoustic Events",
            message=("No timestamped acoustic events are available."),
        )

    grouped: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = defaultdict(list)

    for point in points:
        grouped[point["label"]].append(point)

    figure = go.Figure()

    for label in sorted(grouped):
        label_points = grouped[label]

        figure.add_trace(
            go.Scatter(
                x=[point["time"] for point in label_points],
                y=[point["label"] for point in label_points],
                mode="markers",
                name=label,
                marker={"color": class_color(label), "size": 9},
                customdata=[
                    [
                        point["event_id"],
                        point["confidence"],
                        point["rms"],
                        point["x_m"],
                        point["y_m"],
                    ]
                    for point in label_points
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Time: %{x}<br>"
                    "Event ID: %{customdata[0]}<br>"
                    "Confidence: %{customdata[1]}<br>"
                    "RMS: %{customdata[2]}<br>"
                    "Accepted X: %{customdata[3]} m<br>"
                    "Accepted Y: %{customdata[4]} m"
                    "<extra></extra>"
                ),
            )
        )

    return _apply_common_layout(
        figure,
        title="Recent Acoustic Events",
        x_title="Time",
        y_title="Acoustic Class",
        height=400,
    )


# ======================================================================
# ACTIVITY TIMELINE
# ======================================================================


def build_activity_timeline(
    bins: Iterable[ActivityBin],
    *,
    max_points: int = DEFAULT_MAX_PLOT_POINTS,
) -> go.Figure:
    """
    Plot event count and active duration across temporal bins.
    """

    max_points = _validate_max_points(max_points)

    materialized = tuple(bins)

    for activity_bin in materialized:
        if not isinstance(
            activity_bin,
            ActivityBin,
        ):
            raise TypeError(("bins must contain ActivityBin objects."))

    if not (materialized):
        return empty_figure(
            title="Temporal Acoustic Activity",
        )

    visible_bins = _decimate_sequence(
        materialized,
        max_points=max_points,
    )

    centers = [
        (
            activity_bin.bucket_start
            + (activity_bin.bucket_end - activity_bin.bucket_start) / 2
        )
        for activity_bin in visible_bins
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=centers,
            y=[activity_bin.event_count for activity_bin in visible_bins],
            name="Event count",
            hovertemplate=("Time: %{x}<br>Events: %{y}<extra></extra>"),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=centers,
            y=[activity_bin.active_duration_s for activity_bin in visible_bins],
            mode="lines+markers",
            name="Active duration (s)",
            yaxis="y2",
            hovertemplate=("Time: %{x}<br>Active duration: %{y:.3f} s<extra></extra>"),
        )
    )

    figure.update_layout(
        yaxis2=dict(
            title="Active Duration (s)",
            overlaying="y",
            side="right",
        )
    )

    return _apply_common_layout(
        figure,
        title="Temporal Acoustic Activity",
        x_title="Time",
        y_title="Event Count",
        height=430,
    )


# ======================================================================
# CLASS DISTRIBUTION
# ======================================================================


def build_class_distribution(
    activity: ActivitySummary,
) -> go.Figure:
    """
    Plot detected acoustic-event distribution by broad class.
    """

    if not isinstance(
        activity,
        ActivitySummary,
    ):
        raise TypeError(("activity must be an ActivitySummary."))

    summaries = activity.class_summaries

    if not (summaries):
        return empty_figure(
            title="Acoustic Class Distribution",
        )

    labels = [summary.class_label for summary in summaries]

    counts = [summary.event_count for summary in summaries]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=labels,
            y=counts,
            marker={"color": [class_color(label) for label in labels]},
            customdata=[
                [
                    summary.proportion_of_events,
                    summary.total_duration_s,
                    summary.mean_confidence,
                ]
                for summary in summaries
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Events: %{y}<br>"
                "Proportion: %{customdata[0]:.1%}<br>"
                "Total duration: "
                "%{customdata[1]:.3f} s<br>"
                "Mean confidence: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    return _apply_common_layout(
        figure,
        title="Acoustic Class Distribution",
        x_title="Acoustic Class",
        y_title="Detected Events",
        height=400,
    )


# ======================================================================
# ENVIRONMENTAL TIME SERIES
# ======================================================================


def build_environmental_timeseries(
    rows: Iterable[Any],
    *,
    timestamp_key: str = "bucket_start",
    max_points: int = DEFAULT_MAX_PLOT_POINTS,
) -> go.Figure:
    """
    Plot BME280 environmental measurements over time.
    """

    max_points = _validate_max_points(max_points)

    if not isinstance(
        timestamp_key,
        str,
    ):
        raise TypeError("timestamp_key must be a string.")

    timestamp_key = timestamp_key.strip()

    if not (timestamp_key):
        raise ValueError("timestamp_key cannot be empty.")

    materialized = _decimate_sequence(
        list(rows),
        max_points=max_points,
    )

    timestamps: list[datetime] = []

    temperatures: list[float | None] = []

    humidities: list[float | None] = []

    pressures: list[float | None] = []

    for row in materialized:
        timestamp = _parse_datetime_or_none(
            _row_value(
                row,
                timestamp_key,
            )
        )

        if timestamp is None:
            continue

        timestamps.append(timestamp)

        temperatures.append(
            _finite_float_or_none(
                _row_value(
                    row,
                    "temperature_c",
                )
            )
        )

        humidities.append(
            _finite_float_or_none(
                _row_value(
                    row,
                    "humidity_percent",
                )
            )
        )

        pressures.append(
            _finite_float_or_none(
                _row_value(
                    row,
                    "pressure_hpa",
                )
            )
        )

    if not (timestamps):
        return empty_figure(
            title="Environmental Conditions",
        )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=temperatures,
            mode="lines+markers",
            name="Temperature (°C)",
            hovertemplate=("Time: %{x}<br>Temperature: %{y:.2f} °C<extra></extra>"),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=humidities,
            mode="lines+markers",
            name="Humidity (%)",
            yaxis="y2",
            hovertemplate=("Time: %{x}<br>Humidity: %{y:.2f}%<extra></extra>"),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=pressures,
            mode="lines+markers",
            name="Pressure (hPa)",
            yaxis="y3",
            hovertemplate=("Time: %{x}<br>Pressure: %{y:.2f} hPa<extra></extra>"),
        )
    )

    figure.update_layout(
        yaxis=dict(
            title="Temperature (°C)",
        ),
        yaxis2=dict(
            title="Humidity (%)",
            overlaying="y",
            side="right",
        ),
        yaxis3=dict(
            title="Pressure (hPa)",
            anchor="free",
            overlaying="y",
            side="right",
            position=0.95,
        ),
    )

    return _apply_common_layout(
        figure,
        title="Environmental Conditions",
        x_title="Time",
        height=430,
    )


# ======================================================================
# ENVIRONMENTAL ASSOCIATION MATRIX
# ======================================================================


def build_environmental_association_matrix(
    associations: Iterable[EnvironmentalAssociation],
) -> go.Figure:
    """
    Plot environmental Spearman coefficients as a heatmap.

    Undefined coefficients remain blank.
    """

    materialized = tuple(associations)

    for association in materialized:
        if not isinstance(
            association,
            EnvironmentalAssociation,
        ):
            raise TypeError(
                ("associations must contain EnvironmentalAssociation objects.")
            )

    if not (materialized):
        return empty_figure(
            title="Environmental Associations",
        )

    environmental_variables = sorted(
        {association.environmental_variable for association in materialized}
    )

    response_variables = sorted(
        {association.response_variable for association in materialized}
    )

    lookup = {
        (
            association.environmental_variable,
            association.response_variable,
        ): association
        for association in materialized
    }

    z_values: list[list[float | None]] = []

    hover_text: list[list[str]] = []

    for environmental_variable in environmental_variables:
        z_row: list[float | None] = []

        hover_row: list[str] = []

        for response_variable in response_variables:
            association = lookup.get(
                (
                    environmental_variable,
                    response_variable,
                )
            )

            if association is None or association.coefficient is None:
                z_row.append(None)

                hover_row.append(
                    (
                        f"{environmental_variable}"
                        " vs "
                        f"{response_variable}"
                        "<br>Association unavailable"
                    )
                )

                continue

            z_row.append(association.coefficient)

            p_value_text = (
                "Unavailable"
                if association.p_value is None
                else (f"{association.p_value:.4g}")
            )

            hover_row.append(
                (
                    f"{environmental_variable}"
                    " vs "
                    f"{response_variable}"
                    "<br>Spearman rho: "
                    f"{association.coefficient:.3f}"
                    "<br>p-value: "
                    f"{p_value_text}"
                    "<br>Samples: "
                    f"{association.sample_count}"
                    "<br>Strength: "
                    f"{association.strength.value}"
                    "<br>Direction: "
                    f"{association.direction.value}"
                )
            )

        z_values.append(z_row)

        hover_text.append(hover_row)

    figure = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=response_variables,
            y=environmental_variables,
            zmin=-1.0,
            zmax=1.0,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="Spearman ρ"),
        )
    )

    return _apply_common_layout(
        figure,
        title="Environmental Associations",
        x_title="Acoustic / Activity Metric",
        y_title="Environmental Variable",
        height=430,
    )


# ======================================================================
# LOCALIZED EVENT SCATTER
# ======================================================================


def build_localization_scatter(
    rows: Iterable[Any],
    *,
    node_positions: (
        Mapping[
            int,
            tuple[
                float,
                float,
            ],
        ]
        | None
    ) = None,
    max_points: int = DEFAULT_MAX_PLOT_POINTS,
) -> go.Figure:
    """
    Plot accepted localized acoustic events in array coordinates.

    A row is plotted only when:

        localization_success == True

    and both ``x_m`` and ``y_m`` are finite.
    """

    max_points = _validate_max_points(max_points)

    accepted_points: list[
        tuple[
            Any,
            float,
            float,
        ]
    ] = []

    for row in rows:
        if not (_localization_is_accepted(row)):
            continue

        x_m = _finite_float_or_none(
            _row_value(
                row,
                "x_m",
            )
        )

        y_m = _finite_float_or_none(
            _row_value(
                row,
                "y_m",
            )
        )

        if x_m is None or y_m is None:
            continue

        accepted_points.append(
            (
                row,
                x_m,
                y_m,
            )
        )

    visible_points = _decimate_sequence(
        accepted_points,
        max_points=max_points,
    )

    grouped: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = defaultdict(list)

    for (
        row,
        x_m,
        y_m,
    ) in visible_points:
        label = _row_value(
            row,
            "classification_label",
            "unclassified",
        )

        if label is None or not str(label).strip():
            label = "unclassified"

        grouped[str(label)].append(
            {
                "x": x_m,
                "y": y_m,
                "event_id": _event_identifier(row),
                "time": _row_value(
                    row,
                    "event_time",
                    _row_value(
                        row,
                        "created_at",
                    ),
                ),
                "confidence": _finite_float_or_none(
                    _row_value(
                        row,
                        "classification_confidence",
                    )
                ),
                "residual": _finite_float_or_none(
                    _row_value(
                        row,
                        "localization_residual_m",
                    )
                ),
            }
        )

    if not grouped and not node_positions:
        return empty_figure(
            title="Localized Acoustic Events",
            message=("No accepted localized acoustic events are available."),
        )

    figure = go.Figure()

    for label in sorted(grouped):
        points = grouped[label]

        figure.add_trace(
            go.Scatter(
                x=[point["x"] for point in points],
                y=[point["y"] for point in points],
                mode="markers",
                name=label,
                marker={"color": class_color(label), "size": 9,
                        "line": {"color": "#FFFFFF", "width": 1}},
                customdata=[
                    [
                        point["event_id"],
                        point["time"],
                        point["confidence"],
                        point["residual"],
                    ]
                    for point in points
                ],
                hovertemplate=(
                    "<b>" + label + "</b><br>"
                    "X: %{x:.3f} m<br>"
                    "Y: %{y:.3f} m<br>"
                    "Event: %{customdata[0]}<br>"
                    "Time: %{customdata[1]}<br>"
                    "Confidence: %{customdata[2]}<br>"
                    "Residual: %{customdata[3]} m"
                    "<extra></extra>"
                ),
            )
        )

    _add_microphone_trace(
        figure,
        node_positions=node_positions,
    )

    figure.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return _apply_common_layout(
        figure,
        title="Localized Acoustic Events",
        x_title="X Position (m)",
        y_title="Y Position (m)",
        height=520,
    )


# ======================================================================
# SPATIAL OCCUPANCY HEATMAP
# ======================================================================


def build_spatial_occupancy_heatmap(
    spatial: SpatialSummary,
    *,
    node_positions: (
        Mapping[
            int,
            tuple[
                float,
                float,
            ],
        ]
        | None
    ) = None,
) -> go.Figure:
    """
    Plot acoustic-event occupancy across the spatial analysis grid.
    """

    if not isinstance(
        spatial,
        SpatialSummary,
    ):
        raise TypeError(("spatial must be a SpatialSummary."))

    if not (spatial.cells):
        return empty_figure(
            title="Spatial Acoustic Occupancy",
            message=("No localized acoustic occupancy data are available."),
        )

    x_centers = sorted({cell.center_x_m for cell in spatial.cells})

    y_centers = sorted({cell.center_y_m for cell in spatial.cells})

    x_lookup = {
        value: index
        for (
            index,
            value,
        ) in enumerate(x_centers)
    }

    y_lookup = {
        value: index
        for (
            index,
            value,
        ) in enumerate(y_centers)
    }

    z_values = [[0 for _ in (x_centers)] for _ in (y_centers)]

    text_values = [["" for _ in (x_centers)] for _ in (y_centers)]

    for cell in spatial.cells:
        x_index = x_lookup[cell.center_x_m]

        y_index = y_lookup[cell.center_y_m]

        z_values[y_index][x_index] = cell.event_count

        text_values[y_index][x_index] = (
            "Cell: "
            f"{cell.cell_id}"
            "<br>Events: "
            f"{cell.event_count}"
            "<br>Mean confidence: "
            f"{cell.mean_confidence}"
        )

    figure = go.Figure()

    figure.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=z_values,
            text=text_values,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="Events"),
        )
    )

    _add_microphone_trace(
        figure,
        node_positions=node_positions,
    )

    figure.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return _apply_common_layout(
        figure,
        title="Spatial Acoustic Occupancy",
        x_title="X Position (m)",
        y_title="Y Position (m)",
        height=520,
    )


# ======================================================================
# BEHAVIOR INDICATORS
# ======================================================================


def build_behavior_indicator_chart(
    indicators: Iterable[BehaviorIndicator],
) -> go.Figure:
    """
    Plot conservative behavior-related acoustic indicator scores.

    Indicators with insufficient data remain visible but are represented
    with a missing bar value rather than an invented numeric zero.
    """

    materialized = tuple(indicators)

    for indicator in materialized:
        if not isinstance(
            indicator,
            BehaviorIndicator,
        ):
            raise TypeError(("indicators must contain BehaviorIndicator objects."))

    if not (materialized):
        return empty_figure(
            title="Behavior-Related Acoustic Indicators",
        )

    names: list[str] = []

    scores: list[float | None] = []

    status_text: list[str] = []

    evidence_text: list[str] = []

    for indicator in materialized:
        names.append(
            indicator.name.replace(
                "_",
                " ",
            ).title()
        )

        scores.append(indicator.score)

        status_text.append(indicator.status.value)

        evidence_text.append("<br>".join(indicator.evidence))

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            customdata=[
                [
                    status,
                    evidence,
                    indicator.supporting_event_count,
                    (indicator.score if indicator.score is not None else "Unavailable"),
                ]
                for (
                    status,
                    evidence,
                    indicator,
                ) in zip(
                    status_text,
                    evidence_text,
                    materialized,
                    strict=False,
                )
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Score: %{customdata[3]}<br>"
                "Status: %{customdata[0]}<br>"
                "Supporting events: "
                "%{customdata[2]}<br>"
                "%{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_xaxes(
        range=[
            0.0,
            1.0,
        ]
    )

    return _apply_common_layout(
        figure,
        title="Behavior-Related Acoustic Indicators",
        x_title="Normalized Indicator Score",
        y_title="Indicator",
        height=max(
            350,
            80 * len(materialized),
        ),
    )


# ======================================================================
# ACTIVITY SUMMARY METRICS
# ======================================================================


def activity_summary_metrics(
    activity: ActivitySummary,
) -> dict[
    str,
    Any,
]:
    """
    Produce dashboard-friendly scalar values from ActivitySummary.
    """

    if not isinstance(
        activity,
        ActivitySummary,
    ):
        raise TypeError(("activity must be an ActivitySummary."))

    return {
        "total_events": activity.total_events,
        "total_active_duration_s": activity.total_active_duration_s,
        "mean_event_duration_s": activity.mean_event_duration_s,
        "peak_activity_hour": activity.peak_activity_hour,
        "classified_classes": len(activity.class_summaries),
    }


# ======================================================================
# CONTINUOUS ECOACOUSTIC INDICES TIMELINE
# ======================================================================


def build_soundscape_indices_timeline(
    records: Sequence[Mapping[str, Any]], *, sample_rate: int = 48000,
) -> go.Figure:
    """Plot each microphone separately on acquisition time, not insertion time."""
    from plotly.subplots import make_subplots

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not records:
        return empty_figure(
            title="Continuous Ecoacoustic Soundscape Indices",
            message="No soundscape windows recorded for this session/node.",
        )
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Acoustic Complexity & Bioacoustic Index", "NDSI", "Acoustic Entropy"),
    )
    by_node: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_node[int(record.get("node_id", 1))].append(record)
    metrics = (
        ("aci", "ACI", 1, "#16A34A"),
        ("bioacoustic_index", "BI", 1, "#2563EB"),
        ("ndsi", "NDSI", 2, "#DC2626"),
        ("acoustic_entropy", "Entropy", 3, "#9333EA"),
    )
    styles = ("solid", "dash", "dot")
    for node_index, node in enumerate(sorted(by_node)):
        windows = sorted(by_node[node], key=lambda row: int(row.get("start_sample", 0)))
        times = [int(row.get("start_sample", 0)) / sample_rate for row in windows]
        for key, name, panel, color in metrics:
            figure.add_trace(go.Scatter(
                x=times, y=[_finite_float_or_none(row.get(key)) for row in windows],
                mode="lines+markers", name=f"{name} · Node {node}",
                line={"color": color, "width": 2, "dash": styles[node_index % len(styles)]},
                connectgaps=False,
                hovertemplate=f"{name} · Node {node}<br>Session time: %{{x:.1f}} s<br>Value: %{{y:.3f}}<extra></extra>",
            ), row=panel, col=1)
    figure.add_hline(y=0, line_dash="dash", line_color="#737373", row=2, col=1)
    figure.update_yaxes(title_text="ACI / BI", row=1, col=1)
    figure.update_yaxes(title_text="NDSI", range=[-1.05, 1.05], row=2, col=1)
    figure.update_yaxes(title_text="Entropy", range=[-0.05, 1.05], row=3, col=1)
    figure.update_xaxes(title_text="Time since session start (s)", row=3, col=1)
    figure.update_layout(
        height=700, title="Continuous Soundscape Ecoacoustic Metrics",
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )
    return apply_chart_theme(figure)
