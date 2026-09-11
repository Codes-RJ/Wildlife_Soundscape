"""Session-scoped evidence review and portable data downloads."""

import json
from pathlib import Path
import sqlite3

import streamlit as st

from .audio_view import discover_event_audio_files
from .data_access import DashboardDataAccess


def render_review_view(data_access: DashboardDataAccess) -> None:
    st.title("Review recordings")
    st.caption(
        "Inspect an event, check processing outcomes, and download its evidence."
    )
    database = data_access.database
    try:
        sessions = database.list_sessions(limit=100)
        if not sessions:
            st.info(
                "No sessions yet. Start the receiver or run the demo to collect recordings."
            )
            return
        labels = {int(row["session_id"]): row["label"] for row in sessions}
        session_id = st.selectbox(
            "Session",
            list(labels),
            format_func=lambda sid: f"{labels[sid]} · {sid:08X}",
        )
        manifest = database.get_session_manifest(session_id)
        if manifest is not None and manifest.get("synthetic"):
            st.info("Synthetic demonstration data; these are not field observations.")
        metrics = database.get_session_metrics(session_id)
        if metrics:
            dropped = metrics.get("dropped_samples", 0)
            if dropped:
                st.warning(
                    f"Processing overload dropped {dropped:,} audio samples in this session."
                )
            with st.expander("Acquisition processing summary"):
                st.json(metrics)
        if manifest is None:
            st.info("This older session has no recorded experiment settings.")
        else:
            st.download_button(
                "Download experiment settings",
                json.dumps(manifest, indent=2),
                file_name=f"session_{session_id:08X}.json",
                mime="application/json",
            )
        events = database.analytics_event_rows(
            session_id=session_id,
            sample_rate=(manifest or {})
            .get("config", {})
            .get("audio", {})
            .get("sample_rate", data_access.config.audio.sample_rate),
        )
        if not events:
            st.info("This session has no detected events yet.")
            return
        st.download_button(
            "Download session events",
            json.dumps([dict(row) for row in events], indent=2, default=str),
            file_name=f"events_{session_id:08X}.json",
            mime="application/json",
        )
        event_id = st.selectbox("Event", [int(row["id"]) for row in events])
        event = database.get_event(event_id)
        if event is None:
            st.warning("This event is no longer available. Refresh the session.")
            return
        statuses = database.get_event_processing_status(event_id)
        if statuses:
            st.dataframe(statuses, hide_index=True, width="stretch")
        st.json(dict(event), expanded=False)
        files = discover_event_audio_files(event, project_root=Path.cwd())
        if not files:
            st.info(
                "No audio files are available for this event. Check its audio processing status above."
            )
        for audio in files:
            st.subheader(f"Microphone {audio.node_id}")
            try:
                content = audio.path.read_bytes()
                st.audio(content, format="audio/wav")
                st.download_button(
                    f"Download microphone {audio.node_id}",
                    content,
                    file_name=audio.path.name,
                    mime="audio/wav",
                    key=f"audio_{event_id}_{audio.node_id}",
                )
            except OSError:
                st.warning(f"Microphone {audio.node_id} recording is unavailable.")
    except (OSError, sqlite3.Error) as exc:
        st.error(f"Could not read this session: {exc}")
