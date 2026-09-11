"""
Database-backed live dashboard view.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module renders the lightweight live-monitoring page used by the
Streamlit dashboard.

The live view is intentionally optimized for frequent refreshes.

It displays:

    latest acquisition session
    acquisition state
    recent persisted acoustic events
    latest classification
    recent accepted localization estimates
    latest event environmental context
    recent event table
    lightweight persistence/status information


Important architecture distinction
----------------------------------
This page is DATABASE-BACKED.

It reflects information already persisted by the receiver/event
pipeline.

It does not directly inspect:

    ESP32 sockets
    FreeRTOS queues
    current I2S DMA state
    instantaneous node heartbeat packets
    network packet queues

Those can later be exposed through a dedicated runtime-status interface.


Localization-validity policy
----------------------------
A localization is considered accepted only when:

    localization_success == True

and both:

    x_m
    y_m

are finite.

Numeric coordinates alone do not imply successful localization because
the solver may retain a candidate position even when the solution fails
its acceptance criteria.


Timestamp note
--------------
The lightweight ``recent_events()`` query contains database insertion
time:

    created_at

rather than the reconstructed scientific ``event_time``.

The research analysis page uses reconstructed acquisition timestamps.

This live page intentionally remains lightweight and database-backed.


Performance policy
------------------
The live page does NOT calculate the complete:

    ResearchAnalyticsReport

on every refresh.

Research analytics such as:

    environmental correlations
    spatial occupancy grids
    behavior indicators
    historical summaries

belong in:

    analysis_view.py

This separation keeps live refresh inexpensive.
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from typing import (
    Any,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import streamlit as st


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.core.config import (
    AppConfig,
)


from wildlife_soundscape.dashboard.data_access import (
    DashboardDataAccess,
)


from wildlife_soundscape.dashboard.plots import (
    build_localization_scatter,
    build_recent_event_timeline,
)

from wildlife_soundscape.dashboard.theme import (
    render_student_explainer,
)


# ======================================================================
# CONSTANTS
# ======================================================================


UNAVAILABLE_TEXT = "—"


LIVE_EVENT_TABLE_COLUMNS = (
    "id",
    "created_at",
    "classification_label",
    "classification_confidence",
    "peak_rms_dbfs",
    "best_node_id",
    "localization_success",
    "x_m",
    "y_m",
    "localization_residual_m",
)


# ======================================================================
# SAFE VALUE HELPERS
# ======================================================================


def _finite_float_or_none(
    value: Any,
) -> float | None:
    """
    Convert optional input into a finite float.
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
# OPTIONAL INTEGER
# ======================================================================


def _int_or_none(
    value: Any,
) -> int | None:
    """
    Convert optional value to integer.
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return int(value)

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ======================================================================
# STRING DISPLAY
# ======================================================================


def _display_text(
    value: Any,
    *,
    fallback: str = UNAVAILABLE_TEXT,
) -> str:
    """
    Convert optional value to concise dashboard text.
    """

    if value is None:
        return fallback

    text = str(value).strip()

    if not (text):
        return fallback

    return text


# ======================================================================
# FLOAT DISPLAY
# ======================================================================


def _display_float(
    value: Any,
    *,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    """
    Format an optional finite numeric value.
    """

    value = _finite_float_or_none(value)

    if value is None:
        return UNAVAILABLE_TEXT

    return f"{value:.{decimals}f}{suffix}"


# ======================================================================
# CONFIDENCE DISPLAY
# ======================================================================


def _display_confidence(
    value: Any,
) -> str:
    """
    Format normalized confidence as percentage.
    """

    value = _finite_float_or_none(value)

    if value is None:
        return UNAVAILABLE_TEXT

    return f"{value * 100.0:.1f}%"


# ======================================================================
# SESSION ID DISPLAY
# ======================================================================


def _display_session_id(
    value: Any,
) -> str:
    """
    Format Protocol-v4 session ID as hexadecimal.
    """

    session_id = _int_or_none(value)

    if session_id is None:
        return UNAVAILABLE_TEXT

    if not (0 <= session_id <= 0xFFFFFFFF):
        return str(session_id)

    return f"0x{session_id:08X}"


# ======================================================================
# SESSION STATE
# ======================================================================


def _session_state(
    session: dict[
        str,
        Any,
    ]
    | None,
) -> str:
    """
    Determine persisted acquisition-session state.

    State is based only on the sessions table:

        stopped_at is None
            -> ACTIVE / not explicitly stopped

        stopped_at populated
            -> STOPPED

    This does not prove that ESP32 nodes are currently connected.
    """

    if session is None:
        return "No Session"

    if session.get("stopped_at") is None:
        return "Active"

    return "Stopped"


# ======================================================================
# LOCALIZATION SUCCESS
# ======================================================================


def _localization_succeeded(
    event: dict[
        str,
        Any,
    ],
) -> bool:
    """
    Return True only for an accepted localization result.

    A valid localization requires:

        localization_success == True
        finite x_m
        finite y_m

    Numeric coordinates alone are insufficient because an unsuccessful
    solver result may still contain a candidate position.
    """

    success_value = event.get("localization_success")

    if isinstance(
        success_value,
        bool,
    ):
        success = success_value

    else:
        success_integer = _int_or_none(success_value)

        success = success_integer == 1

    if not (success):
        return False

    x_m = _finite_float_or_none(event.get("x_m"))

    y_m = _finite_float_or_none(event.get("y_m"))

    return x_m is not None and y_m is not None


# ======================================================================
# ACCEPTED LOCALIZATIONS
# ======================================================================


def _accepted_localization_events(
    events: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Return only events containing accepted localization solutions.
    """

    return [event for event in events if _localization_succeeded(event)]


# ======================================================================
# RECENT LOCALIZATION COVERAGE
# ======================================================================


def _recent_localization_coverage(
    events: list[
        dict[
            str,
            Any,
        ]
    ],
) -> tuple[
    int,
    int,
    float,
]:
    """
    Calculate lightweight accepted-localization coverage over recent
    displayed events.

    This is NOT the full research localization-coverage metric unless
    the recent-event list happens to include the entire dataset.
    """

    total = len(events)

    if total == 0:
        return (
            0,
            0,
            0.0,
        )

    localized = sum(1 for event in events if _localization_succeeded(event))

    coverage = localized / total

    return (
        localized,
        total,
        coverage,
    )


# ======================================================================
# EVENT TABLE
# ======================================================================


def _event_table_rows(
    events: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Build a compact table representation for recent events.
    """

    result: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for event in events:
        row: dict[
            str,
            Any,
        ] = {}

        for column in LIVE_EVENT_TABLE_COLUMNS:
            row[column] = event.get(column)

        result.append(row)

    return result


# ======================================================================
# LATEST EVENT METRICS
# ======================================================================


def _render_latest_event_metrics(
    event: dict[
        str,
        Any,
    ],
) -> None:
    """
    Render detailed scalar information for the latest persisted event.
    """

    st.markdown("#### Latest Event")

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    classification_columns = st.columns(4)

    with classification_columns[0]:
        st.metric(
            "Event ID",
            _display_text(event.get("id")),
        )

    with classification_columns[1]:
        st.metric(
            "Class",
            _display_text(
                event.get("classification_label"),
                fallback="Unclassified",
            ),
        )

    with classification_columns[2]:
        st.metric(
            "Confidence",
            _display_confidence(event.get("classification_confidence")),
        )

    with classification_columns[3]:
        st.metric(
            "Peak RMS",
            _display_float(
                event.get("peak_rms_dbfs"),
                decimals=1,
                suffix=" dBFS",
            ),
        )

    # ==================================================================
    # LOCALIZATION
    # ==================================================================

    localization_columns = st.columns(4)

    localization_valid = _localization_succeeded(event)

    with localization_columns[0]:
        st.metric(
            "Localization",
            ("Accepted" if localization_valid else "Unavailable / Rejected"),
        )

    with localization_columns[1]:
        st.metric(
            "X Position",
            (
                _display_float(
                    event.get("x_m"),
                    decimals=3,
                    suffix=" m",
                )
                if localization_valid
                else UNAVAILABLE_TEXT
            ),
        )

    with localization_columns[2]:
        st.metric(
            "Y Position",
            (
                _display_float(
                    event.get("y_m"),
                    decimals=3,
                    suffix=" m",
                )
                if localization_valid
                else UNAVAILABLE_TEXT
            ),
        )

    with localization_columns[3]:
        st.metric(
            "Residual",
            (
                _display_float(
                    event.get("localization_residual_m"),
                    decimals=4,
                    suffix=" m",
                )
                if localization_valid
                else UNAVAILABLE_TEXT
            ),
        )

    # ==================================================================
    # SOURCE / ENVIRONMENT
    # ==================================================================

    source_environment_columns = st.columns(4)

    with source_environment_columns[0]:
        st.metric(
            "Best Node",
            _display_text(event.get("best_node_id")),
        )

    with source_environment_columns[1]:
        st.metric(
            "Temperature",
            _display_float(
                event.get("temperature_c"),
                decimals=2,
                suffix=" °C",
            ),
        )

    with source_environment_columns[2]:
        st.metric(
            "Humidity",
            _display_float(
                event.get("humidity_percent"),
                decimals=1,
                suffix="%",
            ),
        )

    with source_environment_columns[3]:
        st.metric(
            "Pressure",
            _display_float(
                event.get("pressure_hpa"),
                decimals=1,
                suffix=" hPa",
            ),
        )


# ======================================================================
# SESSION OVERVIEW
# ======================================================================


def _render_session_overview(
    *,
    session: dict[
        str,
        Any,
    ]
    | None,
    events: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """
    Render the top-level live dashboard metric cards.
    """

    (
        localized,
        total,
        localization_coverage,
    ) = _recent_localization_coverage(events)

    state = _session_state(session)

    # ==================================================================
    # TOP METRICS
    # ==================================================================

    columns = st.columns(5)

    with columns[0]:
        st.metric(
            "Acquisition",
            state,
        )

    with columns[1]:
        st.metric(
            "Session",
            (
                _display_session_id(session.get("session_id"))
                if session is not None
                else UNAVAILABLE_TEXT
            ),
        )

    with columns[2]:
        st.metric(
            "Recent Events",
            total,
        )

    with columns[3]:
        st.metric(
            "Recently Localized",
            localized,
        )

    with columns[4]:
        st.metric(
            "Recent Localization",
            (
                f"{localization_coverage * 100.0:.1f}%"
                if total > 0
                else UNAVAILABLE_TEXT
            ),
        )

    # ==================================================================
    # SESSION DETAILS
    # ==================================================================

    if session is not None:
        with st.expander(
            "Acquisition Session Details",
            expanded=False,
        ):
            st.write(
                {
                    "session_id": _display_session_id(session.get("session_id")),
                    "label": session.get("label"),
                    "started_at": session.get("started_at"),
                    "stopped_at": session.get("stopped_at"),
                    "state": state,
                }
            )


# ======================================================================
# EVENT VISUALIZATIONS
# ======================================================================


def _render_recent_event_visualizations(
    *,
    events: list[
        dict[
            str,
            Any,
        ]
    ],
    config: AppConfig,
) -> None:
    """
    Render recent persistence timeline and accepted localization scatter.
    """

    st.markdown("### Recent Acoustic Activity")

    st.caption(
        (
            "The live timeline uses persisted event timestamps. "
            "Scientific reconstructed acquisition timestamps are "
            "used by the Research Analysis page."
        )
    )

    left_column, right_column = st.columns(2)

    # ==================================================================
    # TIMELINE
    # ==================================================================

    with left_column:
        timeline = build_recent_event_timeline(
            events,
            max_points=config.dashboard.max_plot_points,
        )

        st.plotly_chart(
            timeline,
            theme=None,
            width="stretch",
        )

    # ==================================================================
    # ACCEPTED LOCALIZATIONS ONLY
    # ==================================================================

    accepted_localizations = _accepted_localization_events(events)

    with right_column:
        scatter = build_localization_scatter(
            accepted_localizations,
            node_positions=config.localization.node_positions,
            max_points=config.dashboard.max_plot_points,
        )

        st.plotly_chart(
            scatter,
            theme=None,
            width="stretch",
        )


# ======================================================================
# RECENT EVENT TABLE
# ======================================================================


def _render_recent_event_table(
    events: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """
    Render compact recent-event records.
    """

    st.markdown("### Recent Event Records")

    if not (events):
        st.info("No persisted acoustic events are available yet.")

        return

    rows = _event_table_rows(events)

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
    )


# ======================================================================
# DATABASE STATUS
# ======================================================================


def _render_database_status(
    data_access: DashboardDataAccess,
) -> None:
    """
    Render lightweight persistence status.
    """

    with st.expander(
        "Persistence Status",
        expanded=False,
    ):
        database_path = data_access.database_path

        exists = database_path.exists()

        try:
            file_size = (
                database_path.stat().st_size
                if (exists and database_path.is_file())
                else 0
            )

        except OSError:
            file_size = 0

        st.write(
            {
                "database_path": str(database_path),
                "database_exists": exists,
                "database_size_bytes": int(file_size),
            }
        )


# ======================================================================
# CORE LIVE CONTENT
# ======================================================================


def _render_live_content(
    *,
    data_access: DashboardDataAccess,
    config: AppConfig,
) -> None:
    """
    Perform one lightweight live-view render cycle.
    """

    # ==================================================================
    # READ SNAPSHOT
    # ==================================================================

    try:
        snapshot = data_access.snapshot(
            recent_limit=config.dashboard.recent_events_limit,
        )

    except Exception as exc:
        st.error((f"Unable to read dashboard snapshot from the event database: {exc}"))

        return

    if not isinstance(
        snapshot,
        dict,
    ):
        st.error(("Dashboard snapshot returned an unexpected data type."))

        return

    session = snapshot.get("latest_session")

    if session is not None and not isinstance(
        session,
        dict,
    ):
        try:
            session = dict(session)

        except (
            TypeError,
            ValueError,
        ):
            session = None

    events = snapshot.get(
        "recent_events",
        [],
    )

    if not isinstance(
        events,
        list,
    ):
        try:
            events = list(events)

        except TypeError:
            events = []

    normalized_events: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for event in events:
        if isinstance(
            event,
            dict,
        ):
            normalized_events.append(event)

            continue

        try:
            normalized_events.append(dict(event))

        except (
            TypeError,
            ValueError,
        ):
            continue

    events = normalized_events

    latest_event = snapshot.get("latest_event")

    if latest_event is not None and not isinstance(
        latest_event,
        dict,
    ):
        try:
            latest_event = dict(latest_event)

        except (
            TypeError,
            ValueError,
        ):
            latest_event = None

    # ==================================================================
    # OVERVIEW
    # ==================================================================

    if st.session_state.get("student_mode_active", True):
        render_student_explainer(
            topic_title="Live Bioacoustic Monitoring",
            what_is_it="This screen shows sound events streamed live from your 3 synchronized ESP32 microphone nodes.",
            why_it_matters="Allows real-time detection of animal calls and instant poaching or disturbance alerts.",
            how_to_read="Check the Node status cards, the red dot on the 2D map for recent sound locations, and the real-time event feed.",
            real_world_example="In Corbett National Park, acoustic sensors alert guards within seconds if a gunshot or chainsaw sound occurs.",
        )

    _render_session_overview(
        session=session,
        events=events,
    )

    st.divider()

    # ==================================================================
    # NO SESSION
    # ==================================================================

    if session is None:
        st.info(("No acquisition session has been persisted yet."))

    # ==================================================================
    # LATEST EVENT
    # ==================================================================

    if latest_event is not None:
        _render_latest_event_metrics(latest_event)

    else:
        st.info(("No acoustic events have been persisted yet."))

    st.divider()

    # ==================================================================
    # VISUALIZATIONS
    # ==================================================================

    _render_recent_event_visualizations(
        events=events,
        config=config,
    )

    st.divider()

    # ==================================================================
    # TABLE
    # ==================================================================

    _render_recent_event_table(events)

    # ==================================================================
    # DATABASE INFORMATION
    # ==================================================================

    _render_database_status(data_access)


# ======================================================================
# MANUAL REFRESH
# ======================================================================


def _render_manual_refresh_control(
    *,
    auto_refresh_active: bool,
) -> None:
    """
    Render fallback/manual dashboard refresh control.
    """

    control_columns = st.columns(
        [
            1,
            4,
        ]
    )

    with control_columns[0]:
        if st.button(
            "Refresh",
            key="live_view_manual_refresh",
            width="stretch",
        ):
            st.rerun()

    with control_columns[1]:
        if auto_refresh_active:
            st.caption(("Automatic database refresh is enabled."))

        else:
            st.caption(
                (
                    "Automatic refresh is unavailable "
                    "or disabled. Use Refresh to "
                    "reload persisted data."
                )
            )


# ======================================================================
# LIVE VIEW
# ======================================================================


def render_live_view(
    data_access: DashboardDataAccess,
    *,
    config: AppConfig | None = None,
) -> None:
    """
    Render the complete Streamlit live-monitoring page.

    Parameters
    ----------
    data_access
        Dashboard read service.

    config
        Optional explicit application configuration.

        When omitted, data_access.config is used.


    Auto-refresh
    ------------
    When Streamlit provides ``st.fragment`` and auto-refresh is enabled,
    the live content is placed inside a fragment using:

        run_every = refresh_interval_s

    On Streamlit versions without fragment support, the view remains
    functional through the manual Refresh button.
    """

    # ==================================================================
    # VALIDATION
    # ==================================================================

    if not isinstance(
        data_access,
        DashboardDataAccess,
    ):
        raise TypeError(("data_access must be a DashboardDataAccess instance."))

    if config is None:
        config = data_access.config

    if not isinstance(
        config,
        AppConfig,
    ):
        raise TypeError("config must be an AppConfig.")

    # ==================================================================
    # PAGE HEADER
    # ==================================================================

    st.title("Live Wildlife Soundscape Monitor")

    st.caption(
        (
            "Database-backed view of recently persisted "
            "acoustic events, classifications and "
            "accepted localization estimates."
        )
    )

    # ------------------------------------------------------------------
    # An open persisted session does not prove that every ESP32 is
    # currently connected or healthy.
    # ------------------------------------------------------------------

    st.info(
        (
            "The acquisition state shown here is derived "
            "from persisted session records. Detailed "
            "per-node TCP/I2S heartbeat health requires "
            "a separate runtime-status interface."
        )
    )

    # ==================================================================
    # AUTO REFRESH CAPABILITY
    # ==================================================================

    fragment_factory = getattr(
        st,
        "fragment",
        None,
    )

    auto_refresh_active = bool(
        config.dashboard.auto_refresh and callable(fragment_factory)
    )

    _render_manual_refresh_control(auto_refresh_active=auto_refresh_active)

    st.divider()

    # ==================================================================
    # FRAGMENT AUTO-REFRESH
    # ==================================================================

    if auto_refresh_active:
        assert callable(fragment_factory)

        @fragment_factory(run_every=config.dashboard.refresh_interval_s)
        def live_fragment() -> None:

            _render_live_content(
                data_access=data_access,
                config=config,
            )

        live_fragment()

        return

    # ==================================================================
    # NON-FRAGMENT FALLBACK
    # ==================================================================

    _render_live_content(
        data_access=data_access,
        config=config,
    )
