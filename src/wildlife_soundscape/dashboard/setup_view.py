"""Deployment configuration, classifier transparency, and data-health UI."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

from wildlife_soundscape.classification.factory import create_classifier_backend
from wildlife_soundscape.core.config import AppConfig
from wildlife_soundscape.core.deployment import deployment_config_path
from wildlife_soundscape.dashboard.data_access import DashboardDataAccess
from wildlife_soundscape.dashboard.monitor_view import session_state
from wildlife_soundscape.dashboard.preflight import build_preflight_checks
from wildlife_soundscape.tools.populate_demo_session import generate_demo_dataset


def _directory_size(path: Path) -> tuple[int, int]:
    count = 0
    size = 0
    if not path.exists():
        return count, size
    for item in path.rglob("*"):
        try:
            if item.is_file():
                count += 1
                size += item.stat().st_size
        except OSError:
            continue
    return count, size


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _render_classifier(config: AppConfig) -> None:
    st.subheader("Classification")
    requested = config.classification.backend
    try:
        backend = create_classifier_backend(config.classification)
        active = "disabled" if backend is None else backend.name
        version = "—" if backend is None else backend.version
    except Exception as exc:
        active = "unavailable"
        version = "—"
        st.error(f"Classifier could not be initialized: {exc}")
    columns = st.columns(3)
    columns[0].metric("Requested backend", requested)
    columns[1].metric("Active backend", active)
    columns[2].metric("Version", version)
    if requested == "heuristic":
        st.info("The heuristic backend recognizes broad acoustic patterns only. Frequent Unknown results are expected when patterns overlap.")
    if active != requested and requested not in {"heuristic", "disabled"}:
        st.warning("The requested model is not active; persisted results will identify the fallback backend that actually ran.")
    with st.expander("Enable BirdNET bird-species assistance"):
        st.code(".\\.venv\\Scripts\\python.exe -m pip install -e .[birdnet]", language="powershell")
        st.code(
            '{\n  "classification": {\n    "backend": "ensemble",\n    "latitude": 26.75,\n    "longitude": 94.20,\n    "week": 24,\n    "use_geo_filter": true\n  }\n}',
            language="json",
        )
        st.caption("Put site-specific settings in config.local.json. BirdNET specializes in birds; the ensemble retains the broad heuristic path for other sounds. Validate predictions against reviewed local recordings.")


def _render_preflight(config: AppConfig) -> None:
    st.subheader("Hardware preflight")
    for check in build_preflight_checks(config):
        message = f"**{check.title}:** {check.detail}"
        if check.status == "pass":
            st.success(message)
        elif check.status == "warning":
            st.warning(message)
        else:
            st.info(message)
    st.caption("Physical synchronization, microphone geometry, calibration, weather resistance, power, and network range still require an on-site acceptance test.")


def _render_acquisition(config: AppConfig) -> None:
    st.subheader("Acquisition and event clips")
    columns = st.columns(4)
    columns[0].metric("Sample rate", f"{config.audio.sample_rate:,} Hz")
    columns[1].metric("Microphones", len(config.expected_nodes))
    columns[2].metric("Packet", f"{config.audio.frames_per_block} samples")
    columns[3].metric("Continuous WAV", "Enabled" if config.audio.record_wav else "Disabled")
    st.write(
        {
            "event_trigger_margin_db": config.detection.trigger_margin_db,
            "event_release_margin_db": config.detection.release_margin_db,
            "minimum_event_ms": config.detection.min_event_ms,
            "maximum_event_s": config.detection.max_event_s,
            "pre_event_padding_s": config.detection.pre_pad_s,
            "post_event_padding_s": config.detection.post_pad_s,
            "required_trigger_nodes": config.detection.min_nodes,
        }
    )
    st.info("Short files in an event folder are detector evidence clips, not the full session. Sessions → Events provides surrounding continuous audio when continuous WAV recording was enabled.")


def _render_data_health(data_access: DashboardDataAccess, config: AppConfig) -> None:
    st.subheader("Data health")
    sessions = data_access.sessions()
    all_events = data_access.analytics_events()
    by_session: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in all_events:
        by_session[int(event["session_id"])].append(event)
    interrupted = [
        session
        for session in sessions
        if session_state(session, by_session[int(session["session_id"])]) == "Interrupted"
    ]
    event_count, event_size = _directory_size(Path(config.persistence.events_dir))
    recording_count, recording_size = _directory_size(Path(config.recordings_dir))
    database_size = data_access.database_path.stat().st_size if data_access.database_path.is_file() else 0
    columns = st.columns(4)
    columns[0].metric("Sessions", len(sessions))
    columns[1].metric("Interrupted", len(interrupted))
    columns[2].metric("Event evidence", f"{event_count} files · {_human_size(event_size)}")
    columns[3].metric("Continuous audio", f"{recording_count} files · {_human_size(recording_size)}")
    st.caption(f"Database: {_human_size(database_size)} · {data_access.database_path}")

    if interrupted:
        with st.expander("Recover interrupted sessions", expanded=True):
            st.warning("Only close these records after confirming the receiver process is not still acquiring them.")
            for session in interrupted:
                st.write(f"{session.get('label')} · 0x{int(session['session_id']):08X} · started {session.get('started_at')}")
            confirmed = st.checkbox("I confirm no receiver is currently writing these sessions", key="confirm_close_interrupted")
            if st.button("Mark interrupted sessions closed", disabled=not confirmed, key="close_interrupted"):
                for session in interrupted:
                    data_access.database.stop_session(int(session["session_id"]))
                st.success(f"Closed {len(interrupted)} interrupted session record(s). Raw audio and events were preserved.")
                st.rerun()
    else:
        st.success("No stale unclosed sessions were detected.")


def _render_demo(data_access: DashboardDataAccess, config: AppConfig) -> None:
    st.subheader("Demonstration data")
    st.write("Create a new, completed synthetic session with equal examples of bird, insect, amphibian, mammal, noise, and unknown. Existing data is never replaced.")
    count = st.number_input("Synthetic events", min_value=12, max_value=600, value=60, step=6)
    if st.button("Create balanced demo session", key="create_demo"):
        with st.spinner("Generating synthetic synchronized evidence…"):
            report = generate_demo_dataset(config, event_count=int(count))
        st.success(f"Created session 0x{int(report['session_id']):08X}. Open Sessions to explore all six classes.")


def render_setup_view(data_access: DashboardDataAccess, *, config: AppConfig) -> None:
    """Render setup and maintenance without crowding monitoring screens."""

    st.title("Setup")
    st.caption("Deployment settings, readiness checks, demonstration data, and storage health.")
    st.write(f"Configuration source: `{deployment_config_path()}`" if deployment_config_path().is_file() else "Configuration source: committed defaults (no config.local.json found)")
    acquisition, classifier, preflight, data_health, demo = st.tabs(["Acquisition", "Classifier", "Preflight", "Data health", "Demo data"])
    with acquisition:
        _render_acquisition(config)
    with classifier:
        _render_classifier(config)
    with preflight:
        _render_preflight(config)
    with data_health:
        _render_data_health(data_access, config)
    with demo:
        _render_demo(data_access, config)
