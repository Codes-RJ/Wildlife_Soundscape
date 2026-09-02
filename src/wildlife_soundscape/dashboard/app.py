"""
Main Streamlit dashboard application.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Launch
------

From the project root:

    streamlit run src/wildlife_soundscape/dashboard/app.py


Purpose
-------
This module is the presentation-layer entry point for the local
Wildlife Soundscape dashboard.

Available views
---------------
Live Monitor
    Lightweight, frequently refreshed view of recently persisted:

        acquisition-session state
        acoustic events
        classifications
        localization results
        environmental context


Research Analysis
    Historical session-based analysis including:

        temporal activity
        class distribution
        environmental associations
        localization maps
        spatial occupancy
        behavior-related acoustic indicators
        event audio
        research export


Architecture
------------

    src/wildlife_soundscape/dashboard/app.py
            │
            ├── live_view.py
            │
            └── analysis_view.py
                    │
                    ↓
            data_access.py
                    │
             ┌──────┴──────┐
             ↓             ↓
        storage/database.py    analytics/
             │             │
             └──────┬──────┘
                    ↓
                 SQLite


This module does NOT perform:

    audio acquisition
    DSP
    event detection
    classification
    localization
    database persistence

Those remain responsibilities of the core backend.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import sys

from pathlib import (
    Path,
)

from typing import (
    Any,
)


# ======================================================================
# PROJECT ROOT
# ======================================================================
#
# When launched as:
#
#     streamlit run src/wildlife_soundscape/dashboard/app.py
#
# the dashboard directory can become the script import location.
#
# Explicitly make the source root importable so the
# ``wildlife_soundscape`` package remains resolvable.
# ======================================================================


SOURCE_ROOT = Path(__file__).resolve().parents[2]


if (
    str(SOURCE_ROOT)
    not in sys.path
):

    sys.path.insert(0, str(SOURCE_ROOT))


# ======================================================================
# THIRD-PARTY
# ======================================================================


import streamlit as st


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.core.config import (
    AppConfig,
    CONFIG,
)

from wildlife_soundscape.dashboard.analysis_view import (
    render_analysis_view,
)

from wildlife_soundscape.dashboard.data_access import (
    DashboardDataAccess,
    create_dashboard_data_access,
)

from wildlife_soundscape.dashboard.live_view import (
    render_live_view,
)


# ======================================================================
# NAVIGATION
# ======================================================================


LIVE_VIEW = (
    "Live Monitor"
)


ANALYSIS_VIEW = (
    "Research Analysis"
)


AVAILABLE_VIEWS = (
    LIVE_VIEW,
    ANALYSIS_VIEW,
)


# ======================================================================
# PAGE CONFIGURATION
# ======================================================================


def configure_page(
    config: AppConfig,
) -> None:
    """
    Configure the Streamlit browser page.

    This should execute before ordinary page rendering.
    """

    if not isinstance(
        config,
        AppConfig,
    ):

        raise TypeError(
            "config must be an AppConfig."
        )

    st.set_page_config(
        page_title=
            config
            .dashboard
            .page_title,

        page_icon=
            "🎙️",

        layout=
            "wide",

        initial_sidebar_state=
            "expanded",
    )


# ======================================================================
# CACHED DATA ACCESS
# ======================================================================


@st.cache_resource(
    show_spinner=
        False
)
def get_dashboard_data_access() -> DashboardDataAccess:
    """
    Create and cache the dashboard read service.

    EventDatabase itself opens short-lived SQLite connections for
    individual operations, so retaining the service object does not
    retain one permanent SQLite connection.
    """

    return create_dashboard_data_access(
        config=
            CONFIG
    )


# ======================================================================
# FILE SIZE DISPLAY
# ======================================================================


def _human_file_size(
    size_bytes: int,
) -> str:
    """
    Convert byte count into human-readable storage text.
    """

    if (
        isinstance(
            size_bytes,
            bool,
        )
        or not isinstance(
            size_bytes,
            int,
        )
    ):

        return (
            "—"
        )

    if (
        size_bytes
        < 0
    ):

        return (
            "—"
        )

    units = (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    )

    value = float(
        size_bytes
    )

    unit_index = (
        0
    )

    while (
        value
        >= 1024.0
        and unit_index
        < len(
            units
        )
        - 1
    ):

        value /= (
            1024.0
        )

        unit_index += (
            1
        )

    if (
        unit_index
        == 0
    ):

        return (
            f"{int(value)} "
            f"{units[unit_index]}"
        )

    return (
        f"{value:.2f} "
        f"{units[unit_index]}"
    )


# ======================================================================
# DATABASE SIDEBAR STATUS
# ======================================================================


def _database_status(
    data_access: DashboardDataAccess,
) -> dict[
    str,
    Any,
]:
    """
    Build lightweight database status for sidebar display.
    """

    path = (
        data_access.database_path
    )

    exists = (
        path.exists()
        and path.is_file()
    )

    size_bytes = (
        int(
            path.stat().st_size
        )

        if exists

        else 0
    )

    return {
        "path":
            path,

        "exists":
            exists,

        "size_bytes":
            size_bytes,
    }


# ======================================================================
# SIDEBAR SYSTEM INFORMATION
# ======================================================================


def _render_system_information(
    *,
    config: AppConfig,
    data_access: DashboardDataAccess,
) -> None:
    """
    Render compact architecture/configuration information.
    """

    with st.sidebar.expander(
        "System Configuration",
        expanded=
            False,
    ):

        st.markdown(
            "**Acquisition**"
        )

        st.write(
            f"{config.audio.sample_rate:,} Hz"
        )

        st.write(
            (
                f"{config.audio.frames_per_block} "
                "samples / packet"
            )
        )

        st.write(
            (
                f"{len(config.expected_nodes)} "
                "acoustic nodes"
            )
        )

        st.write(
            (
                "PCM16 mono per node"
            )
        )

        st.markdown(
            "**Localization**"
        )

        st.write(
            "GCC-PHAT / TDOA"
        )

        st.write(
            (
                f"Reference node: "
                f"{config.localization.reference_node}"
            )
        )

        st.write(
            (
                "Window: "
                f"{config.localization.window_samples} "
                "samples"
            )
        )

        st.markdown(
            "**Analytics**"
        )

        st.write(
            (
                "Activity bin: "
                f"{config.analytics.bucket_seconds} s"
            )
        )

        st.write(
            (
                "Spatial cell: "
                f"{config.analytics.cell_size_m:.3f} m"
            )
        )

        st.write(
            (
                "Environmental node: "
                f"{config.analytics.environmental_node_id}"
            )
        )

        st.write(
            (
                "Spearman α: "
                f"{config.analytics.environmental_alpha:.3f}"
            )
        )

    # ==============================================================
    # DATABASE
    # ==============================================================

    with st.sidebar.expander(
        "Database",
        expanded=
            False,
    ):

        status = (
            _database_status(
                data_access
            )
        )

        st.write(
            (
                "Status: "
                + (
                    "Available"

                    if status[
                        "exists"
                    ]

                    else "Not created"
                )
            )
        )

        st.write(
            (
                "Size: "
                + _human_file_size(
                    status[
                        "size_bytes"
                    ]
                )
            )
        )

        st.caption(
            str(
                status[
                    "path"
                ]
            )
        )


# ======================================================================
# SIDEBAR SESSION SUMMARY
# ======================================================================


def _render_session_summary(
    data_access: DashboardDataAccess,
) -> None:
    """
    Render lightweight persisted-session information.
    """

    try:

        latest_session = (
            data_access.latest_session()
        )

    except Exception as exc:

        st.sidebar.warning(
            (
                "Unable to read session "
                f"metadata: {exc}"
            )
        )

        return

    with st.sidebar.expander(
        "Latest Session",
        expanded=
            False,
    ):

        if (
            latest_session
            is None
        ):

            st.caption(
                "No acquisition session recorded."
            )

            return

        try:

            session_id = int(
                latest_session[
                    "session_id"
                ]
            )

            session_text = (
                f"0x{session_id:08X}"
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            session_text = (
                "Unknown"
            )

        state = (
            "Active"

            if latest_session.get(
                "stopped_at"
            )
            is None

            else "Stopped"
        )

        st.write(
            {
                "session":
                    session_text,

                "label":
                    latest_session.get(
                        "label"
                    ),

                "state":
                    state,

                "started_at":
                    latest_session.get(
                        "started_at"
                    ),

                "stopped_at":
                    latest_session.get(
                        "stopped_at"
                    ),
            }
        )


# ======================================================================
# SIDEBAR
# ======================================================================


def render_sidebar(
    *,
    config: AppConfig,
    data_access: DashboardDataAccess,
) -> str:
    """
    Render navigation and system information.

    Returns
    -------
    str
        Selected dashboard view.
    """

    st.sidebar.title(
        "Wildlife Soundscape"
    )

    st.sidebar.caption(
        (
            "Acoustic Monitoring & "
            "Behavior Analysis"
        )
    )

    st.sidebar.divider()

    selected_view = st.sidebar.radio(
        "Dashboard View",

        options=
            AVAILABLE_VIEWS,

        index=
            0,

        key=
            "dashboard_navigation",
    )

    st.sidebar.divider()

    _render_session_summary(
        data_access
    )

    _render_system_information(
        config=
            config,

        data_access=
            data_access,
    )

    st.sidebar.divider()

    st.sidebar.caption(
        (
            "Local research dashboard. "
            "No cloud connection is required."
        )
    )

    return (
        selected_view
    )


# ======================================================================
# APPLICATION FOOTER
# ======================================================================


def render_footer() -> None:
    """
    Render minimal scientific interpretation footer.
    """

    st.divider()

    st.caption(
        (
            "Wildlife Soundscape Mapping & Behavior Analysis System · "
            "Acoustic detections and derived indicators should be "
            "interpreted as sensor observations, not direct proof of "
            "animal abundance, identity, movement or causal behavior."
        )
    )


# ======================================================================
# DASHBOARD DISABLED STATE
# ======================================================================


def _render_disabled_dashboard(
    config: AppConfig,
) -> None:
    """
    Render configuration-disabled state.
    """

    st.title(
        config
        .dashboard
        .page_title
    )

    st.warning(
        (
            "Dashboard functionality is disabled "
            "in DashboardConfig."
        )
    )

    st.code(
        (
            "DashboardConfig(\n"
            "    enabled=True,\n"
            ")"
        ),
        language=
            "python",
    )


# ======================================================================
# APPLICATION
# ======================================================================


def main() -> None:
    """
    Run the complete Streamlit dashboard.
    """

    config = (
        CONFIG
    )

    # ==============================================================
    # PAGE CONFIG
    # ==============================================================

    configure_page(
        config
    )

    # ==============================================================
    # DASHBOARD ENABLED
    # ==============================================================

    if not (
        config
        .dashboard
        .enabled
    ):

        _render_disabled_dashboard(
            config
        )

        return

    # ==============================================================
    # DATA ACCESS
    # ==============================================================

    try:

        data_access = (
            get_dashboard_data_access()
        )

    except Exception as exc:

        st.title(
            config
            .dashboard
            .page_title
        )

        st.error(
            (
                "Dashboard initialization failed "
                "while opening the event database."
            )
        )

        st.exception(
            exc
        )

        return

    # ==============================================================
    # SIDEBAR / NAVIGATION
    # ==============================================================

    selected_view = (
        render_sidebar(
            config=
                config,

            data_access=
                data_access,
        )
    )

    # ==============================================================
    # LIVE VIEW
    # ==============================================================

    if (
        selected_view
        == LIVE_VIEW
    ):

        render_live_view(
            data_access,

            config=
                config,
        )

    # ==============================================================
    # RESEARCH VIEW
    # ==============================================================

    elif (
        selected_view
        == ANALYSIS_VIEW
    ):

        render_analysis_view(
            data_access,

            config=
                config,
        )

    # ==============================================================
    # DEFENSIVE FALLBACK
    # ==============================================================

    else:

        st.error(
            (
                "Unknown dashboard view: "
                f"{selected_view!r}"
            )
        )

    # ==============================================================
    # FOOTER
    # ==============================================================

    render_footer()


# ======================================================================
# ENTRY POINT
# ======================================================================


if (
    __name__
    == "__main__"
):

    main()
