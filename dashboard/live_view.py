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
    recent localization estimates
    latest event environmental context
    recent event table
    lightweight health/status information


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

Those can be added later through a dedicated runtime-status interface if
required.

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

from datetime import (
    datetime,
)

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


from config import (
    AppConfig,
)

from dashboard.data_access import (
    DashboardDataAccess,
)

from dashboard.plots import (
    build_localization_scatter,
    build_recent_event_timeline,
)


# ======================================================================
# CONSTANTS
# ======================================================================


LIVE_EVENT_TABLE_COLUMNS = (
    "id",
    "created_at",
    "classification_label",
    "classification_confidence",
    "peak_rms_dbfs",
    "best_node_id",
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


# ======================================================================
# OPTIONAL INTEGER
# ======================================================================


def _int_or_none(
    value: Any,
) -> int | None:
    """
    Convert optional value to integer.
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
        bool,
    ):

        return (
            None
        )

    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            None
        )


# ======================================================================
# STRING DISPLAY
# ======================================================================


def _display_text(
    value: Any,
    *,
    fallback: str = "—",
) -> str:
    """
    Convert optional value to concise dashboard text.
    """

    if (
        value
        is None
    ):

        return (
            fallback
        )

    text = str(
        value
    ).strip()

    if not (
        text
    ):

        return (
            fallback
        )

    return (
        text
    )


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

    value = (
        _finite_float_or_none(
            value
        )
    )

    if (
        value
        is None
    ):

        return (
            "—"
        )

    return (
        f"{value:.{decimals}f}"
        f"{suffix}"
    )


# ======================================================================
# CONFIDENCE DISPLAY
# ======================================================================


def _display_confidence(
    value: Any,
) -> str:
    """
    Format normalized confidence as percentage.
    """

    value = (
        _finite_float_or_none(
            value
        )
    )

    if (
        value
        is None
    ):

        return (
            "—"
        )

    return (
        f"{value * 100.0:.1f}%"
    )


# ======================================================================
# SESSION ID DISPLAY
# ======================================================================


def _display_session_id(
    value: Any,
) -> str:
    """
    Format Protocol-v4 session ID as hexadecimal.
    """

    session_id = (
        _int_or_none(
            value
        )
    )

    if (
        session_id
        is None
    ):

        return (
            "—"
        )

    if not (
        0
        <= session_id
        <= 0xFFFFFFFF
    ):

        return str(
            session_id
        )

    return (
        f"0x{session_id:08X}"
    )


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
    """

    if (
        session
        is None
    ):

        return (
            "No Session"
        )

    if (
        session.get(
            "stopped_at"
        )
        is None
    ):

        return (
            "Active"
        )

    return (
        "Stopped"
    )


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
    Calculate lightweight localization coverage over recent displayed
    events.

    This is NOT the full research localization-coverage metric unless
    the recent-event list happens to include the entire dataset.
    """

    total = (
        len(
            events
        )
    )

    if (
        total
        == 0
    ):

        return (
            0,
            0,
            0.0,
        )

    localized = (
        0
    )

    for event in (
        events
    ):

        x_m = (
            _finite_float_or_none(
                event.get(
                    "x_m"
                )
            )
        )

        y_m = (
            _finite_float_or_none(
                event.get(
                    "y_m"
                )
            )
        )

        if (
            x_m
            is not None
            and y_m
            is not None
        ):

            localized += (
                1
            )

    coverage = (
        localized
        / total
    )

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

    for event in (
        events
    ):

        row: dict[
            str,
            Any,
        ] = {}

        for column in (
            LIVE_EVENT_TABLE_COLUMNS
        ):

            row[
                column
            ] = (
                event.get(
                    column
                )
            )

        result.append(
            row
        )

    return (
        result
    )


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

    # ==============================================================
    # CLASSIFICATION
    # ==============================================================

    st.markdown(
        "#### Latest Event"
    )

    classification_columns = st.columns(
        4
    )

    with classification_columns[
        0
    ]:

        st.metric(
            "Event ID",
            _display_text(
                event.get(
                    "id"
                )
            ),
        )

    with classification_columns[
        1
    ]:

        st.metric(
            "Class",
            _display_text(
                event.get(
                    "classification_label"
                ),
                fallback=
                    "Unclassified",
            ),
        )

    with classification_columns[
        2
    ]:

        st.metric(
            "Confidence",
            _display_confidence(
                event.get(
                    "classification_confidence"
                )
            ),
        )

    with classification_columns[
        3
    ]:

        st.metric(
            "Peak RMS",
            _display_float(
                event.get(
                    "peak_rms_dbfs"
                ),
                decimals=
                    1,
                suffix=
                    " dBFS",
            ),
        )

    # ==============================================================
    # LOCALIZATION
    # ==============================================================

    localization_columns = st.columns(
        4
    )

    with localization_columns[
        0
    ]:

        st.metric(
            "X Position",
            _display_float(
                event.get(
                    "x_m"
                ),
                decimals=
                    3,
                suffix=
                    " m",
            ),
        )

    with localization_columns[
        1
    ]:

        st.metric(
            "Y Position",
            _display_float(
                event.get(
                    "y_m"
                ),
                decimals=
                    3,
                suffix=
                    " m",
            ),
        )

    with localization_columns[
        2
    ]:

        st.metric(
            "Residual",
            _display_float(
                event.get(
                    "localization_residual_m"
                ),
                decimals=
                    4,
                suffix=
                    " m",
            ),
        )

    with localization_columns[
        3
    ]:

        st.metric(
            "Best Node",
            _display_text(
                event.get(
                    "best_node_id"
                )
            ),
        )

    # ==============================================================
    # ENVIRONMENT
    # ==============================================================

    environment_columns = st.columns(
        3
    )

    with environment_columns[
        0
    ]:

        st.metric(
            "Temperature",
            _display_float(
                event.get(
                    "temperature_c"
                ),
                decimals=
                    2,
                suffix=
                    " °C",
            ),
        )

    with environment_columns[
        1
    ]:

        st.metric(
            "Humidity",
            _display_float(
                event.get(
                    "humidity_percent"
                ),
                decimals=
                    1,
                suffix=
                    "%",
            ),
        )

    with environment_columns[
        2
    ]:

        st.metric(
            "Pressure",
            _display_float(
                event.get(
                    "pressure_hpa"
                ),
                decimals=
                    1,
                suffix=
                    " hPa",
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
    ) = _recent_localization_coverage(
        events
    )

    state = (
        _session_state(
            session
        )
    )

    # ==============================================================
    # TOP METRICS
    # ==============================================================

    columns = st.columns(
        5
    )

    with columns[
        0
    ]:

        st.metric(
            "Acquisition",
            state,
        )

    with columns[
        1
    ]:

        st.metric(
            "Session",
            (
                _display_session_id(
                    session.get(
                        "session_id"
                    )
                )

                if session
                is not None

                else "—"
            ),
        )

    with columns[
        2
    ]:

        st.metric(
            "Recent Events",
            total,
        )

    with columns[
        3
    ]:

        st.metric(
            "Recently Localized",
            localized,
        )

    with columns[
        4
    ]:

        st.metric(
            "Recent Localization",
            (
                f"{localization_coverage * 100.0:.1f}%"

                if total
                > 0

                else "—"
            ),
        )

    # ==============================================================
    # SESSION DETAILS
    # ==============================================================

    if (
        session
        is not None
    ):

        with st.expander(
            "Acquisition Session Details",
            expanded=
                False,
        ):

            st.write(
                {
                    "session_id":
                        _display_session_id(
                            session.get(
                                "session_id"
                            )
                        ),

                    "label":
                        session.get(
                            "label"
                        ),

                    "started_at":
                        session.get(
                            "started_at"
                        ),

                    "stopped_at":
                        session.get(
                            "stopped_at"
                        ),

                    "state":
                        state,
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
    Render recent event timeline and localization scatter.
    """

    st.markdown(
        "### Recent Acoustic Activity"
    )

    left_column, right_column = st.columns(
        2
    )

    # ==============================================================
    # TIMELINE
    # ==============================================================

    with left_column:

        timeline = (
            build_recent_event_timeline(
                events,

                max_points=
                    config
                    .dashboard
                    .max_plot_points,
            )
        )

        st.plotly_chart(
            timeline,
            use_container_width=
                True,
        )

    # ==============================================================
    # LOCALIZATION
    # ==============================================================

    with right_column:

        scatter = (
            build_localization_scatter(
                events,

                node_positions=
                    config
                    .localization
                    .node_positions,

                max_points=
                    config
                    .dashboard
                    .max_plot_points,
            )
        )

        st.plotly_chart(
            scatter,
            use_container_width=
                True,
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

    st.markdown(
        "### Recent Event Records"
    )

    if not (
        events
    ):

        st.info(
            "No persisted acoustic events are available yet."
        )

        return

    rows = (
        _event_table_rows(
            events
        )
    )

    st.dataframe(
        rows,
        use_container_width=
            True,
        hide_index=
            True,
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
        expanded=
            False,
    ):

        database_path = (
            data_access.database_path
        )

        exists = (
            database_path.exists()
        )

        file_size = (
            database_path.stat().st_size

            if (
                exists
                and database_path.is_file()
            )

            else 0
        )

        st.write(
            {
                "database_path":
                    str(
                        database_path
                    ),

                "database_exists":
                    exists,

                "database_size_bytes":
                    int(
                        file_size
                    ),
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

    # ==============================================================
    # READ SNAPSHOT
    # ==============================================================

    try:

        snapshot = (
            data_access.snapshot(
                recent_limit=
                    config
                    .dashboard
                    .recent_events_limit,
            )
        )

    except Exception as exc:

        st.error(
            (
                "Unable to read dashboard snapshot "
                f"from the event database: {exc}"
            )
        )

        return

    session = (
        snapshot.get(
            "latest_session"
        )
    )

    events = (
        snapshot.get(
            "recent_events",
            [],
        )
    )

    if not isinstance(
        events,
        list,
    ):

        events = list(
            events
        )

    latest_event = (
        snapshot.get(
            "latest_event"
        )
    )

    # ==============================================================
    # OVERVIEW
    # ==============================================================

    _render_session_overview(
        session=
            session,

        events=
            events,
    )

    st.divider()

    # ==============================================================
    # NO SESSION
    # ==============================================================

    if (
        session
        is None
    ):

        st.info(
            (
                "No acquisition session has been "
                "persisted yet."
            )
        )

    # ==============================================================
    # LATEST EVENT
    # ==============================================================

    if (
        latest_event
        is not None
    ):

        _render_latest_event_metrics(
            latest_event
        )

    else:

        st.info(
            (
                "No acoustic events have been "
                "persisted yet."
            )
        )

    st.divider()

    # ==============================================================
    # VISUALIZATIONS
    # ==============================================================

    _render_recent_event_visualizations(
        events=
            events,

        config=
            config,
    )

    st.divider()

    # ==============================================================
    # TABLE
    # ==============================================================

    _render_recent_event_table(
        events
    )

    # ==============================================================
    # DATABASE INFORMATION
    # ==============================================================

    _render_database_status(
        data_access
    )


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

    with control_columns[
        0
    ]:

        if st.button(
            "Refresh",
            key=
                "live_view_manual_refresh",
            use_container_width=
                True,
        ):

            st.rerun()

    with control_columns[
        1
    ]:

        if (
            auto_refresh_active
        ):

            st.caption(
                (
                    "Automatic database refresh "
                    "is enabled."
                )
            )

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

    This refreshes the live panel without intentionally running the
    research analytics page.

    On Streamlit versions without fragment support, the view remains
    functional through the manual Refresh button.
    """

    # ==============================================================
    # VALIDATION
    # ==============================================================

    if not isinstance(
        data_access,
        DashboardDataAccess,
    ):

        raise TypeError(
            (
                "data_access must be a "
                "DashboardDataAccess instance."
            )
        )

    if (
        config
        is None
    ):

        config = (
            data_access.config
        )

    if not isinstance(
        config,
        AppConfig,
    ):

        raise TypeError(
            "config must be an AppConfig."
        )

    # ==============================================================
    # PAGE HEADER
    # ==============================================================

    st.title(
        "Live Wildlife Soundscape Monitor"
    )

    st.caption(
        (
            "Database-backed view of recently persisted "
            "acoustic events, classifications and "
            "localization estimates."
        )
    )

    # --------------------------------------------------------------
    # Important wording:
    #
    # An open session does not by itself prove that every ESP32 is
    # currently healthy or connected.
    # --------------------------------------------------------------

    st.info(
        (
            "The acquisition state shown here is derived "
            "from persisted session records. Detailed "
            "per-node TCP/I2S heartbeat health requires "
            "a separate runtime-status interface."
        )
    )

    # ==============================================================
    # AUTO REFRESH CAPABILITY
    # ==============================================================

    fragment_factory = getattr(
        st,
        "fragment",
        None,
    )

    auto_refresh_active = bool(
        config.dashboard.auto_refresh
        and callable(
            fragment_factory
        )
    )

    _render_manual_refresh_control(
        auto_refresh_active=
            auto_refresh_active
    )

    st.divider()

    # ==============================================================
    # FRAGMENT AUTO-REFRESH
    # ==============================================================

    if (
        auto_refresh_active
    ):

        # ----------------------------------------------------------
        # Define the fragment locally so its refresh interval can use
        # the current validated configuration.
        # ----------------------------------------------------------

        @fragment_factory(
            run_every=
                config
                .dashboard
                .refresh_interval_s
        )
        def live_fragment() -> None:

            _render_live_content(
                data_access=
                    data_access,

                config=
                    config,
            )

        live_fragment()

        return

    # ==============================================================
    # NON-FRAGMENT FALLBACK
    # ==============================================================

    _render_live_content(
        data_access=
            data_access,

        config=
            config,
    )