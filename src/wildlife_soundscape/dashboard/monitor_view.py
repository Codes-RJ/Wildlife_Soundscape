"""A focused operational dashboard for the current acquisition session."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from wildlife_soundscape.core.config import AppConfig
from wildlife_soundscape.dashboard.audio_view import (
    build_preview_wav_bytes,
    discover_event_audio_files,
)
from wildlife_soundscape.dashboard.data_access import DashboardDataAccess
from wildlife_soundscape.dashboard.plots import (
    build_localization_scatter,
    build_recent_event_timeline,
)


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def session_state(session: dict[str, Any], events: list[dict[str, Any]]) -> str:
    """Classify an unclosed old session as interrupted, not indefinitely active."""

    if session.get("stopped_at") is not None:
        return "Completed"
    timestamps = [_parse_utc(session.get("started_at"))]
    timestamps.extend(_parse_utc(event.get("created_at")) for event in events)
    latest = max((item for item in timestamps if item is not None), default=None)
    if latest is not None and (datetime.now(timezone.utc) - latest).total_seconds() > 300:
        return "Interrupted"
    return "Active"


def session_data_type(data_access: DashboardDataAccess, session_id: int) -> str:
    manifest = data_access.database.get_session_manifest(session_id)
    if manifest and manifest.get("synthetic"):
        return "Synthetic fixture"
    return "Recorded pipeline data"


def _reasons(event: dict[str, Any]) -> list[str]:
    value = event.get("classification_reasons_json")
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _render_monitor_content(
    data_access: DashboardDataAccess,
    *,
    config: AppConfig,
) -> None:
    """Render only the information needed while observing an acquisition."""

    try:
        session = data_access.latest_session()
        all_recent = data_access.recent_events(limit=config.dashboard.recent_events_limit)
    except Exception as exc:
        st.error(f"Unable to read monitoring data: {exc}")
        return

    if session is None:
        st.info("No sessions yet. Start the receiver or create a synthetic demo session.")
        return

    session_id = int(session["session_id"])
    events = [row for row in all_recent if int(row.get("session_id", -1)) == session_id]
    state = session_state(session, events)
    data_type = session_data_type(data_access, session_id)
    classifier = next((row.get("classifier_name") for row in events if row.get("classifier_name")), "Not available")

    columns = st.columns(4)
    columns[0].metric("Session", f"0x{session_id:08X}")
    columns[1].metric("State", state)
    columns[2].metric("Recent events", len(events))
    columns[3].metric("Classifier", str(classifier).replace("_", " "))
    st.caption(f"{data_type} · label: {session.get('label', 'session')}")

    if state == "Interrupted":
        st.warning("This session was never closed and has had no recent data. It is shown as interrupted; close it from Setup → Data health.")
    if data_type == "Synthetic fixture":
        st.info("Illustrative synthetic data. It demonstrates every broad class and is not a field observation.")
    elif config.classification.backend == "heuristic":
        st.info("Heuristic broad-group classifier active. Unknown is a valid uncertain result; species identity is not verified.")

    if not events:
        st.info("This session has no persisted events yet.")
        return

    latest = events[0]
    duration = max(0, int(latest.get("end_sample", 0)) - int(latest.get("start_sample", 0))) / config.audio.sample_rate
    st.subheader("Latest detection")
    details = st.columns(4)
    details[0].metric("Class", latest.get("classification_label") or "Unclassified")
    confidence = latest.get("classification_confidence")
    details[1].metric("Confidence", f"{float(confidence):.0%}" if confidence is not None else "—")
    details[2].metric("Event clip", f"{duration:.2f} s")
    details[3].metric("Location", "Accepted" if latest.get("localization_success") else "Unavailable")

    reasons = _reasons(latest)
    if reasons:
        with st.expander("Why this classification?"):
            st.write(f"Backend: {latest.get('classifier_name')} · version {latest.get('classifier_version')}")
            for reason in reasons:
                st.write(f"• {reason}")

    audio_files = discover_event_audio_files(latest, expected_nodes=config.expected_nodes)
    if audio_files:
        chosen = st.selectbox(
            "Latest-event microphone",
            audio_files,
            format_func=lambda item: f"Node {item.node_id} · {item.duration_s:.2f} s",
            key="monitor_audio_node",
        )
        if chosen is not None:
            st.audio(build_preview_wav_bytes(chosen, max_duration_s=config.dashboard.max_audio_preview_s), format="audio/wav")

    plot_left, plot_right = st.columns(2)
    with plot_left:
        st.plotly_chart(build_recent_event_timeline(events, max_points=config.dashboard.max_plot_points), theme=None, width="stretch")
    with plot_right:
        st.plotly_chart(build_localization_scatter(events, node_positions=config.localization.node_positions, max_points=config.dashboard.max_plot_points), theme=None, width="stretch")

    show_unknown = st.checkbox("Include Unknown results", value=True, key="monitor_show_unknown")
    visible = events if show_unknown else [row for row in events if row.get("classification_label") != "unknown"]
    table = [
        {
            "event": row.get("id"),
            "time": row.get("created_at"),
            "class": row.get("classification_label") or "unclassified",
            "confidence": row.get("classification_confidence"),
            "duration_s": round((int(row.get("end_sample", 0)) - int(row.get("start_sample", 0))) / config.audio.sample_rate, 3),
            "best_node": row.get("best_node_id"),
        }
        for row in visible
    ]
    st.dataframe(table, hide_index=True, width="stretch")


def render_monitor_view(
    data_access: DashboardDataAccess,
    *,
    config: AppConfig,
) -> None:
    """Render the monitor with automatic database refresh when supported."""

    st.title("Monitor")
    st.caption(
        "Current session health, latest detections, audio evidence, and "
        "location estimates."
    )
    refresh, status = st.columns([1, 4])
    if refresh.button("Refresh", key="monitor_refresh"):
        st.rerun()
    fragment_factory = getattr(st, "fragment", None)
    automatic = bool(config.dashboard.auto_refresh and callable(fragment_factory))
    status.caption(
        f"Automatic refresh every {config.dashboard.refresh_interval_s:g} seconds."
        if automatic
        else "Automatic refresh is unavailable; use Refresh."
    )
    if automatic:
        assert callable(fragment_factory)

        @fragment_factory(run_every=config.dashboard.refresh_interval_s)
        def monitor_fragment() -> None:
            _render_monitor_content(data_access, config=config)

        monitor_fragment()
    else:
        _render_monitor_content(data_access, config=config)
