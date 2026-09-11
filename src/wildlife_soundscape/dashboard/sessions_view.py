"""Session-first analysis and evidence review UI."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from wildlife_soundscape.analytics.models import ResearchAnalyticsReport
from wildlife_soundscape.core.config import AppConfig
from wildlife_soundscape.dashboard.analysis_view import (
    _render_activity_section,
    _render_behavior_section,
    _render_environment_section,
    _render_report_export,
    _render_report_overview,
    _render_research_warnings,
    _render_soundscape_indices_section,
    _render_spatial_section,
    _session_option_label,
)
from wildlife_soundscape.dashboard.audio_view import (
    build_context_wav_bytes,
    build_preview_wav_bytes,
    build_spectrogram_figure,
    build_waveform_figure,
    discover_event_audio_files,
    discover_session_recordings,
)
from wildlife_soundscape.dashboard.data_access import DashboardDataAccess
from wildlife_soundscape.dashboard.monitor_view import session_data_type, session_state


def _json_list(value: Any) -> list[str]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _event_label(event: dict[str, Any], sample_rate: int) -> str:
    duration = (int(event["end_sample"]) - int(event["start_sample"])) / sample_rate
    return f"Event {event['id']} · {event.get('classification_label') or 'unclassified'} · {duration:.2f} s · {event.get('event_time', '')}"


def _select_event(key: str, event: dict[str, Any]) -> None:
    st.session_state[key] = event


def _render_classification(event: dict[str, Any]) -> None:
    st.markdown("#### Classification evidence")
    columns = st.columns(4)
    columns[0].metric("Model result", event.get("classification_label") or "Unclassified")
    confidence = event.get("classification_confidence")
    columns[1].metric("Confidence", f"{float(confidence):.1%}" if confidence is not None else "—")
    columns[2].metric("Runner-up", event.get("classification_second_label") or "—")
    margin = event.get("classification_margin")
    columns[3].metric("Decision margin", f"{float(margin):.3f}" if margin is not None else "—")
    st.caption(f"Backend: {event.get('classifier_name') or 'not recorded'} · version: {event.get('classifier_version') or 'not recorded'}")
    reasons = _json_list(event.get("classification_reasons_json"))
    if reasons:
        for reason in reasons:
            st.write(f"• {reason}")
    if event.get("classification_label") == "unknown":
        st.info("Unknown means the available acoustic evidence was not distinct enough for a reliable broad-group decision.")


def _render_review_form(data_access: DashboardDataAccess, event_id: int) -> None:
    existing = data_access.database.get_event_review(event_id)
    with st.expander("Human review", expanded=existing is not None):
        st.caption("A review is stored separately; it never overwrites the original model output.")
        choices = ["unreviewed", "bird", "insect", "amphibian", "mammal", "noise", "unknown"]
        current = str((existing or {}).get("reviewed_label", "unreviewed"))
        if current not in choices:
            choices.append(current)
        label = st.selectbox("Reviewed label", choices, index=choices.index(current), key=f"review_label_{event_id}")
        reviewer = st.text_input("Reviewer", value=str((existing or {}).get("reviewer", "")), key=f"reviewer_{event_id}")
        notes = st.text_area("Notes", value=str((existing or {}).get("notes", "")), key=f"review_notes_{event_id}")
        if st.button("Save review", key=f"save_review_{event_id}", disabled=label == "unreviewed" or not reviewer.strip()):
            data_access.database.save_event_review(event_id, reviewed_label=label, reviewer=reviewer, notes=notes)
            st.success("Review saved without changing the model prediction.")


def _render_event_audio(
    event: dict[str, Any],
    session: dict[str, Any],
    *,
    config: AppConfig,
) -> None:
    event_files = {item.node_id: item for item in discover_event_audio_files(event, expected_nodes=config.expected_nodes)}
    continuous_files = {
        item.node_id: item
        for item in discover_session_recordings(
            session,
            recordings_root=config.recordings_dir,
            expected_nodes=config.expected_nodes,
        )
    }
    node_ids = sorted(set(event_files) | set(continuous_files))
    if not node_ids:
        st.info("No WAV evidence is available for this event or session.")
        return

    st.markdown("#### Synchronized microphone evidence")
    before, after = st.columns(2)
    context_before = before.slider("Context before event (seconds)", 0, 15, 5, key=f"before_{event['id']}")
    context_after = after.slider("Context after event (seconds)", 0, 15, 5, key=f"after_{event['id']}")
    tabs = st.tabs([f"Node {node_id}" for node_id in node_ids])
    for tab, node_id in zip(tabs, node_ids, strict=True):
        with tab:
            event_audio = event_files.get(node_id)
            continuous = continuous_files.get(node_id)
            if event_audio is not None:
                st.caption(f"Detected event clip · {event_audio.duration_s:.2f} s · original evidence")
                st.audio(build_preview_wav_bytes(event_audio, max_duration_s=config.dashboard.max_audio_preview_s), format="audio/wav")
                figures = st.columns(2)
                with figures[0]:
                    st.plotly_chart(build_waveform_figure(event_audio, max_duration_s=config.dashboard.max_audio_preview_s, max_points=config.dashboard.max_plot_points), theme=None, width="stretch")
                with figures[1]:
                    st.plotly_chart(build_spectrogram_figure(event_audio, max_duration_s=config.dashboard.max_audio_preview_s, n_fft=config.dsp.n_fft, hop_length=config.dsp.hop_length), theme=None, width="stretch")
            else:
                st.warning("The event clip for this microphone is missing.")

            if continuous is not None:
                context = build_context_wav_bytes(
                    continuous,
                    event_start_sample=int(event["start_sample"]),
                    event_end_sample=int(event["end_sample"]),
                    context_before_s=context_before,
                    context_after_s=context_after,
                )
                clipping = []
                if context.clipped_at_start:
                    clipping.append("start")
                if context.clipped_at_end:
                    clipping.append("end")
                status = f" · clipped at {' and '.join(clipping)} of recording" if clipping else ""
                st.caption(f"Continuous context · starts {context.start_s:.2f} s into session · {context.duration_s:.2f} s{status}")
                st.audio(context.wav_bytes, format="audio/wav")
                with st.expander(f"Full continuous recording ({continuous.duration_s:.1f} s)"):
                    st.audio(str(continuous.path), format="audio/wav")
                    st.caption(str(continuous.path))
            else:
                st.info("No matching continuous session WAV exists, so surrounding context is unavailable.")


def _render_events_tab(
    data_access: DashboardDataAccess,
    event_rows: list[dict[str, Any]],
    session: dict[str, Any],
    *,
    config: AppConfig,
) -> None:
    if not event_rows:
        st.info("This session has no detected events.")
        return

    labels = sorted({str(row.get("classification_label") or "unclassified") for row in event_rows})
    selected_labels = st.multiselect("Class filter", labels, default=labels, key=f"class_filter_{session['session_id']}")
    filtered = [row for row in reversed(event_rows) if str(row.get("classification_label") or "unclassified") in selected_labels]
    if not filtered:
        st.info("No events match the selected classes.")
        return

    select_key = f"event_select_{session['session_id']}"
    current = st.session_state.get(select_key)
    current_id = current.get("id") if isinstance(current, dict) else None
    if current_id not in {row.get("id") for row in filtered}:
        st.session_state[select_key] = filtered[0]
    selected = st.selectbox(
        "Event",
        filtered,
        format_func=lambda row: _event_label(row, config.audio.sample_rate),
        key=select_key,
    )
    index = filtered.index(selected)
    previous, following, position = st.columns([1, 1, 3])
    previous.button(
        "← Newer",
        disabled=index == 0,
        key=f"newer_{session['session_id']}",
        on_click=_select_event,
        args=(select_key, filtered[max(0, index - 1)]),
    )
    following.button(
        "Older →",
        disabled=index >= len(filtered) - 1,
        key=f"older_{session['session_id']}",
        on_click=_select_event,
        args=(select_key, filtered[min(len(filtered) - 1, index + 1)]),
    )
    position.caption(f"{index + 1} of {len(filtered)} matching events")

    duration = (int(selected["end_sample"]) - int(selected["start_sample"])) / config.audio.sample_rate
    st.caption(f"Event clips contain the detector window and padding ({duration:.2f} s here); use continuous context below for the surrounding soundscape.")
    _render_classification(selected)
    _render_review_form(data_access, int(selected["id"]))
    _render_event_audio(selected, session, config=config)

    with st.expander("Processing and raw metadata"):
        statuses = data_access.database.get_event_processing_status(int(selected["id"]))
        if statuses:
            st.dataframe(statuses, hide_index=True, width="stretch")
        st.json(selected, expanded=False)


def render_sessions_view(
    data_access: DashboardDataAccess,
    *,
    config: AppConfig,
) -> None:
    """Render one completed acquisition session at a time."""

    st.title("Sessions")
    st.caption("Choose a recording session, inspect its evidence, then analyze or export it.")
    try:
        sessions = data_access.sessions()
    except Exception as exc:
        st.error(f"Unable to read sessions: {exc}")
        return
    if not sessions:
        st.info("No persisted sessions are available.")
        return

    default_index = next((i for i, item in enumerate(sessions) if item.get("stopped_at") is not None), 0)
    session = st.selectbox("Acquisition session", sessions, index=default_index, format_func=_session_option_label, key="sessions_selector")
    if session is None:
        return
    session_id = int(session["session_id"])
    try:
        event_rows = data_access.analytics_events(session_id=session_id)
        manifest = data_access.database.get_session_manifest(session_id)
        report = data_access.research_report(session_id=session_id, include_environment=True)
        environmental_rows = data_access.environmental_bins(session_id=session_id)
    except Exception as exc:
        st.error(f"Unable to build this session view: {exc}")
        return
    if not isinstance(report, ResearchAnalyticsReport):
        st.error("Research analytics returned an unexpected result.")
        return

    data_type = session_data_type(data_access, session_id)
    state = session_state(session, event_rows)
    st.caption(f"{data_type} · {state} · {len(event_rows)} events")
    if data_type == "Synthetic fixture":
        st.info("This balanced six-class session is an illustrative fixture, not output from a field classifier.")
    elif config.classification.backend == "heuristic":
        st.info("This session uses broad heuristic labels. Review audio before treating a label as an observation.")

    overview, events, soundscape, spatial, export = st.tabs(["Overview", "Events", "Soundscape", "Spatial", "Export"])
    with overview:
        _render_report_overview(report, session=session)
        _render_research_warnings(report)
        _render_activity_section(event_rows=event_rows, report=report, config=config)
        with st.expander("Environmental analysis"):
            _render_environment_section(environmental_rows=environmental_rows, report=report, config=config)
        with st.expander("Behavior-related acoustic indicators"):
            _render_behavior_section(report)
    with events:
        _render_events_tab(data_access, event_rows, session, config=config)
    with soundscape:
        _render_soundscape_indices_section(data_access=data_access, session_id=session_id, config=config)
    with spatial:
        _render_spatial_section(event_rows=event_rows, report=report, config=config)
    with export:
        _render_report_export(report, session_id=session_id)
        st.download_button(
            "Download session events JSON",
            json.dumps(event_rows, indent=2, default=str),
            file_name=f"events_{session_id:08X}.json",
            mime="application/json",
        )
        if manifest is not None:
            st.download_button(
                "Download experiment settings",
                json.dumps(manifest, indent=2),
                file_name=f"session_{session_id:08X}_manifest.json",
                mime="application/json",
            )
        reviews = data_access.database.list_event_reviews(session_id=session_id)
        if reviews:
            st.download_button(
                "Download human reviews JSON",
                json.dumps(reviews, indent=2, default=str),
                file_name=f"reviews_{session_id:08X}.json",
                mime="application/json",
            )
