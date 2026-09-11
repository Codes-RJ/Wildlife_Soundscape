"""Streamlit entry point for the local Wildlife Soundscape dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import streamlit as st

from wildlife_soundscape.core.config import CONFIG, AppConfig
from wildlife_soundscape.dashboard.data_access import (
    DashboardDataAccess,
    create_dashboard_data_access,
)
from wildlife_soundscape.dashboard.monitor_view import render_monitor_view
from wildlife_soundscape.dashboard.sessions_view import render_sessions_view
from wildlife_soundscape.dashboard.setup_view import render_setup_view
from wildlife_soundscape.dashboard.theme import (
    get_theme_css,
    render_student_sidebar_guide,
)


MONITOR_VIEW = "Monitor"
SESSIONS_VIEW = "Sessions"
SETUP_VIEW = "Setup"
AVAILABLE_VIEWS = (MONITOR_VIEW, SESSIONS_VIEW, SETUP_VIEW)


def configure_page(config: AppConfig) -> None:
    """Configure the browser before ordinary rendering starts."""

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    st.set_page_config(
        page_title=config.dashboard.page_title,
        page_icon=":material/graphic_eq:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_resource(show_spinner=False)
def get_dashboard_data_access() -> DashboardDataAccess:
    """Retain the lightweight database facade across Streamlit reruns."""

    return create_dashboard_data_access(config=CONFIG)


def render_sidebar(*, config: AppConfig, data_access: DashboardDataAccess) -> str:
    """Render compact navigation; detailed controls live under Setup."""

    st.sidebar.title("Wildlife Soundscape")
    st.sidebar.caption("Local acoustic observatory")
    selected = st.sidebar.radio(
        "Navigation",
        AVAILABLE_VIEWS,
        key="dashboard_navigation",
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    latest = data_access.latest_session()
    if latest is None:
        st.sidebar.caption("No recorded sessions")
    else:
        state = "open" if latest.get("stopped_at") is None else "completed"
        st.sidebar.caption(
            f"Latest: {latest.get('label', 'session')} · "
            f"0x{int(latest['session_id']):08X} · {state}"
        )
    with st.sidebar.expander("Display options"):
        student_mode = st.checkbox(
            "Learning explanations",
            value=False,
            key="student_mode_active",
            help="Show optional educational explanations beside research plots.",
        )
    if student_mode:
        render_student_sidebar_guide()
    st.sidebar.caption("Field claims require reviewed audio and calibrated hardware.")
    return selected


def render_footer() -> None:
    st.divider()
    st.caption(
        "Detections, classifier outputs, and acoustic-source estimates are sensor "
        "evidence—not direct proof of species identity, abundance, or movement."
    )


def main() -> None:
    config = CONFIG
    configure_page(config)
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    if not config.dashboard.enabled:
        st.title(config.dashboard.page_title)
        st.warning("Dashboard functionality is disabled in DashboardConfig.")
        return
    try:
        data_access = get_dashboard_data_access()
        selected = render_sidebar(config=config, data_access=data_access)
    except Exception as exc:
        st.error(f"Dashboard initialization failed: {exc}")
        return

    if selected == MONITOR_VIEW:
        render_monitor_view(data_access, config=config)
    elif selected == SESSIONS_VIEW:
        render_sessions_view(data_access, config=config)
    elif selected == SETUP_VIEW:
        render_setup_view(data_access, config=config)
    else:
        st.error(f"Unknown dashboard view: {selected!r}")
    render_footer()


if __name__ == "__main__":
    main()
