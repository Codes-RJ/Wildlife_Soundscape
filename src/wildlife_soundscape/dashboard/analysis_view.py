"""
Historical and research analytics dashboard view.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module renders the research-oriented Streamlit analysis page.

Unlike live_view.py, this page intentionally performs the complete
analytics pipeline over a selected acquisition session.

Analysis areas
--------------
    temporal acoustic activity
    acoustic-class distribution
    environmental conditions
    environmental associations
    localization coverage
    spatial acoustic occupancy
    consecutive spatial-event transitions
    conservative behavior-related indicators
    event-audio inspection
    research-report export


Session policy
--------------
Research analysis is intentionally performed one acquisition session at
a time.

This prevents several important analytical errors:

    treating system downtime as zero wildlife activity

    linking spatial transitions across separate acquisition sessions

    averaging environmental conditions across periods when the system
    was not observing

Cross-session research aggregation can later be implemented using
explicit session-aware exposure accounting.


Partial-report policy
---------------------
ResearchAnalyticsReport deliberately permits:

    activity = None
    spatial = None

because limited datasets may still produce useful partial analytics.

The dashboard must therefore never assume that temporal or spatial
analysis is always available.

Unavailable analytical sections are displayed explicitly rather than
replaced with invented zero-valued scientific results.


Scientific interpretation
-------------------------
The dashboard describes detected acoustic-event patterns.

It does not directly establish:

    animal abundance
    individual identity
    animal trajectories
    causal environmental effects
    confirmed ethological behavior
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import json

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


from wildlife_soundscape.analytics.activity import (
    build_activity_bins,
)


from wildlife_soundscape.analytics.models import (
    ActivitySummary,
    ResearchAnalyticsReport,
    SpatialSummary,
)


from wildlife_soundscape.core.config import (
    AppConfig,
)


from wildlife_soundscape.dashboard.audio_view import (
    build_preview_wav_bytes,
    build_spectrogram_figure,
    build_waveform_figure,
    discover_event_audio_files,
)


from wildlife_soundscape.dashboard.data_access import (
    DashboardDataAccess,
)


from wildlife_soundscape.dashboard.plots import (
    activity_summary_metrics,
    build_activity_timeline,
    build_behavior_indicator_chart,
    build_class_distribution,
    build_environmental_association_matrix,
    build_environmental_timeseries,
    build_localization_scatter,
    build_soundscape_indices_timeline,
    build_spatial_occupancy_heatmap,
)

from wildlife_soundscape.dashboard.theme import (
    render_student_explainer,
)


# ======================================================================
# DISPLAY CONSTANTS
# ======================================================================


UNAVAILABLE_TEXT = "—"


# ======================================================================
# SESSION ID DISPLAY
# ======================================================================


def _display_session_id(
    value: Any,
) -> str:
    """
    Display one Protocol-v4 uint32 session identifier.
    """

    try:
        session_id = int(value)

    except (
        TypeError,
        ValueError,
    ):
        return UNAVAILABLE_TEXT

    if not (0 <= session_id <= 0xFFFFFFFF):
        return str(session_id)

    return f"0x{session_id:08X}"


# ======================================================================
# SESSION SELECTOR LABEL
# ======================================================================


def _session_option_label(
    session: dict[
        str,
        Any,
    ],
) -> str:
    """
    Build a readable Streamlit session-selector label.
    """

    session_id = _display_session_id(session.get("session_id"))

    label = str(
        session.get(
            "label",
            "Session",
        )
    ).strip()

    if not (label):
        label = "Session"

    started_at = str(
        session.get(
            "started_at",
            "",
        )
    ).strip()

    if not (started_at):
        started_at = "unknown start"

    state = "Active" if session.get("stopped_at") is None else "Stopped"

    return f"{label} | {session_id} | {started_at} | {state}"


# ======================================================================
# REPORT ACTIVITY
# ======================================================================


def _report_activity(
    report: ResearchAnalyticsReport,
) -> ActivitySummary | None:
    """
    Return validated optional activity summary.
    """

    activity = report.activity

    if activity is None:
        return None

    if not isinstance(
        activity,
        ActivitySummary,
    ):
        raise TypeError(
            ("ResearchAnalyticsReport.activity must be ActivitySummary or None.")
        )

    return activity


# ======================================================================
# REPORT SPATIAL
# ======================================================================


def _report_spatial(
    report: ResearchAnalyticsReport,
) -> SpatialSummary | None:
    """
    Return validated optional spatial summary.
    """

    spatial = report.spatial

    if spatial is None:
        return None

    if not isinstance(
        spatial,
        SpatialSummary,
    ):
        raise TypeError(
            ("ResearchAnalyticsReport.spatial must be SpatialSummary or None.")
        )

    return spatial


# ======================================================================
# REPORT OVERVIEW
# ======================================================================


def _render_report_overview(
    report: ResearchAnalyticsReport,
    *,
    session: dict[
        str,
        Any,
    ],
) -> None:
    """
    Render primary research summary metrics.

    Missing optional analytics sections are displayed as unavailable.

    They are not silently converted into scientific zero values.
    """

    if not isinstance(
        report,
        ResearchAnalyticsReport,
    ):
        raise TypeError(("report must be a ResearchAnalyticsReport."))

    activity = _report_activity(report)

    spatial = _report_spatial(report)

    # ==================================================================
    # ACTIVITY METRICS
    # ==================================================================

    if activity is not None:
        metrics = activity_summary_metrics(activity)

    else:
        metrics = None

    st.markdown("### Research Overview")

    first_row = st.columns(5)

    # ==================================================================
    # SESSION
    # ==================================================================

    with first_row[0]:
        st.metric(
            "Session",
            _display_session_id(session.get("session_id")),
        )

    # ==================================================================
    # EVENTS
    # ==================================================================

    with first_row[1]:
        # --------------------------------------------------------------
        # The report-level analytics window always carries event_count,
        # even when the optional ActivitySummary is unavailable.
        # --------------------------------------------------------------

        if metrics is not None:
            detected_events = metrics["total_events"]

        else:
            detected_events = report.window.event_count

        st.metric(
            "Detected Events",
            detected_events,
        )

    # ==================================================================
    # ACTIVE AUDIO
    # ==================================================================

    with first_row[2]:
        if metrics is None:
            active_audio_text = UNAVAILABLE_TEXT

        else:
            active_audio_text = f"{metrics['total_active_duration_s']:.2f} s"

        st.metric(
            "Event Duration",
            active_audio_text,
        )

    # ==================================================================
    # LOCALIZATION COVERAGE
    # ==================================================================

    with first_row[3]:
        if spatial is None:
            localization_coverage = UNAVAILABLE_TEXT

        else:
            localization_coverage = f"{spatial.localization_coverage * 100.0:.1f}%"

        st.metric(
            "Localization Coverage",
            localization_coverage,
        )

    # ==================================================================
    # CLASSES
    # ==================================================================

    with first_row[4]:
        if metrics is None:
            class_count = UNAVAILABLE_TEXT

        else:
            class_count = metrics["classified_classes"]

        st.metric(
            "Acoustic Classes",
            class_count,
        )

    # ==================================================================
    # SECOND ROW
    # ==================================================================

    second_row = st.columns(4)

    # ==================================================================
    # MEAN EVENT DURATION
    # ==================================================================

    with second_row[0]:
        if metrics is None:
            mean_duration = UNAVAILABLE_TEXT

        else:
            mean_duration = f"{metrics['mean_event_duration_s']:.3f} s"

        st.metric(
            "Mean Event Duration",
            mean_duration,
        )

    # ==================================================================
    # PEAK ACTIVITY HOUR
    # ==================================================================

    with second_row[1]:
        if metrics is None:
            peak_hour_text = UNAVAILABLE_TEXT

        else:
            peak_hour = metrics["peak_activity_hour"]

            peak_hour_text = (
                f"{peak_hour:02d}:00" if peak_hour is not None else UNAVAILABLE_TEXT
            )

        st.metric(
            "Peak Activity Hour",
            peak_hour_text,
        )

    # ==================================================================
    # LOCALIZED EVENTS
    # ==================================================================

    with second_row[2]:
        if spatial is None:
            localized_events_text = UNAVAILABLE_TEXT

        else:
            localized_events_text = (
                f"{spatial.localized_event_count} / {spatial.total_event_count}"
            )

        st.metric(
            "Localized Events",
            localized_events_text,
        )

    # ==================================================================
    # HOTSPOT
    # ==================================================================

    with second_row[3]:
        if spatial is None:
            hotspot_text = UNAVAILABLE_TEXT

        elif spatial.hotspot_cell_id is None:
            hotspot_text = UNAVAILABLE_TEXT

        else:
            hotspot_text = spatial.hotspot_cell_id

        st.metric(
            "Acoustic Hotspot",
            hotspot_text,
        )


# ======================================================================
# WARNINGS
# ======================================================================


def _render_research_warnings(
    report: ResearchAnalyticsReport,
) -> None:
    """
    Render scientific/data-quality warnings produced by analytics.
    """

    if not (report.warnings):
        return

    with st.expander(
        "Research Interpretation & Data Quality",
        expanded=True,
    ):
        for warning in report.warnings:
            st.warning(warning)


# ======================================================================
# TEMPORAL ACTIVITY
# ======================================================================


def _render_activity_section(
    *,
    event_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    report: ResearchAnalyticsReport,
    config: AppConfig,
) -> None:
    """
    Render temporal activity and acoustic-class analysis.

    The temporal event-bin visualization can still be generated from
    normalized event rows even when the optional aggregate
    ActivitySummary is unavailable.

    Class-distribution metrics require ActivitySummary and are therefore
    shown only when that analytics result exists.
    """

    st.markdown("## Temporal Acoustic Activity")

    if st.session_state.get("student_mode_active", True):
        render_student_explainer(
            topic_title="Temporal Activity & Calling Distribution",
            what_is_it="Graphs showing when acoustic events were detected and how broad classifier outputs are distributed.",
            why_it_matters="Repeated schedules can suggest patterns worth testing, such as a dawn chorus, while accounting for detector effort and uncertainty.",
            how_to_read="The timeline shows detections over time. The class chart summarizes model outputs, not verified species counts.",
            real_world_example="A reviewed field dataset may show more bird detections near dawn and more insect-like detections after dusk.",
        )

    activity = _report_activity(report)

    # ==================================================================
    # BUILD ACTIVITY BINS
    # ==================================================================

    try:
        activity_bins = build_activity_bins(
            event_rows,
            bucket_seconds=config.analytics.bucket_seconds,
            sample_rate=config.audio.sample_rate,
            timestamp_key="event_time",
        )

    except Exception as exc:
        st.warning((f"Temporal activity bins could not be generated: {exc}"))

        activity_bins = ()

    left_column, right_column = st.columns(2)

    # ==================================================================
    # ACTIVITY TIMELINE
    # ==================================================================

    with left_column:
        figure = build_activity_timeline(
            activity_bins,
            max_points=config.dashboard.max_plot_points,
        )

        st.plotly_chart(
            figure,
            theme=None,
            width="stretch",
        )

    # ==================================================================
    # CLASS DISTRIBUTION
    # ==================================================================

    with right_column:
        if activity is None:
            st.info(
                (
                    "Aggregate acoustic-class activity "
                    "statistics are unavailable for "
                    "this report."
                )
            )

        else:
            figure = build_class_distribution(activity)

            st.plotly_chart(
                figure,
                theme=None,
                width="stretch",
            )

    # ==================================================================
    # CLASS SUMMARY TABLE
    # ==================================================================

    if activity is not None and activity.class_summaries:
        st.markdown("#### Acoustic Class Summary")

        rows = [
            {
                "class": summary.class_label,
                "events": summary.event_count,
                "total_duration_s": summary.total_duration_s,
                "mean_confidence": summary.mean_confidence,
                "event_proportion": summary.proportion_of_events,
            }
            for summary in activity.class_summaries
        ]

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )


# ======================================================================
# ENVIRONMENTAL ANALYSIS
# ======================================================================


def _render_environment_section(
    *,
    environmental_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    report: ResearchAnalyticsReport,
    config: AppConfig,
) -> None:
    """
    Render environmental observations and correlation analysis.
    """

    st.markdown("## Environmental Analysis")

    st.caption(
        (
            "Environmental associations use regular "
            "within-session observation bins, including "
            "periods with zero detected acoustic events."
        )
    )

    if st.session_state.get("student_mode_active", True):
        render_student_explainer(
            topic_title="Environmental Correlation & Microclimate",
            what_is_it="Measures associations between recorded weather variables and acoustic-event rates using Spearman correlation (rho).",
            why_it_matters="Weather may coincide with changing acoustic activity, recorder conditions, or detection performance and is useful context for further testing.",
            how_to_read="Values near +1.0 mean calls increase as the weather factor rises. Values near -1.0 mean calls decrease.",
            real_world_example="A study could test whether reviewed amphibian calls are more frequent during humid recording periods.",
        )

    if not (environmental_rows):
        st.info(("No environmental observation bins are available for this session."))

        return

    # ==================================================================
    # ENVIRONMENTAL TIME SERIES
    # ==================================================================

    figure = build_environmental_timeseries(
        environmental_rows,
        timestamp_key="bucket_start",
        max_points=config.dashboard.max_plot_points,
    )

    st.plotly_chart(
        figure,
        theme=None,
        width="stretch",
    )

    # ==================================================================
    # CORRELATION MATRIX
    # ==================================================================

    st.markdown("#### Environment ↔ Acoustic Activity Associations")

    st.caption(
        (
            "Spearman correlation describes monotonic "
            "association. It does not establish "
            "environmental causation."
        )
    )

    figure = build_environmental_association_matrix(report.environmental_associations)

    st.plotly_chart(
        figure,
        theme=None,
        width="stretch",
    )

    # ==================================================================
    # ASSOCIATION TABLE
    # ==================================================================

    if report.environmental_associations:
        rows = [
            {
                "environmental_variable": association.environmental_variable,
                "response_variable": association.response_variable,
                "samples": association.sample_count,
                "spearman_rho": association.coefficient,
                "p_value": association.p_value,
                "direction": association.direction.value,
                "strength": association.strength.value,
                "significant_raw_alpha": association.statistically_significant,
                "method": association.method,
            }
            for association in report.environmental_associations
        ]

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            (
                "Significance flags currently describe "
                "the result carried by the analytics "
                "association object. Formal multi-test "
                "interpretation should use the project's "
                "configured statistical methodology."
            )
        )


# ======================================================================
# CONTINUOUS ECOACOUSTIC INDICES
# ======================================================================


def _render_soundscape_indices_section(
    *,
    data_access: DashboardDataAccess,
    session_id: int,
    config: AppConfig,
) -> None:
    """
    Render continuous ecoacoustic indices (ACI, NDSI, Entropy, BI) with parameter metadata.
    """
    st.markdown("## Continuous Ecoacoustic Soundscape Metrics")
    st.caption(
        "Continuous soundscape metrics (ACI, NDSI, Acoustic Entropy H, Bioacoustic Index) "
        "calculated over rolling analysis windows to evaluate biophonic and anthrophonic soundscape pressure."
    )

    if st.session_state.get("student_mode_active", True):
        render_student_explainer(
            topic_title="Continuous Ecoacoustic Indices (ACI & NDSI)",
            what_is_it="Parameter-dependent summaries of acoustic variation and energy in configured frequency bands.",
            why_it_matters="They support like-for-like comparison of recording periods after equipment, schedule, weather, and site effects are controlled.",
            how_to_read="Compare values only across compatible recordings. No universal ACI or NDSI threshold proves ecosystem health.",
            real_world_example="A study may compare NDSI across matched sites and then inspect audio to identify what drove the difference.",
        )

    indices_records = data_access.soundscape_indices(session_id=session_id)

    if not indices_records:
        st.info(
            "No continuous soundscape index records have been persisted for this acquisition session."
        )
        return

    # Node selection
    available_nodes = sorted({r.get("node_id", 1) for r in indices_records})
    node_options = ["All Nodes"] + [f"Node {n}" for n in available_nodes]
    selected_node_label = st.selectbox("Soundscape Node Filter", node_options, index=0)

    if selected_node_label != "All Nodes":
        selected_node = int(selected_node_label.replace("Node ", ""))
        display_records = [
            r for r in indices_records if r.get("node_id") == selected_node
        ]
    else:
        display_records = indices_records

    fig = build_soundscape_indices_timeline(display_records, sample_rate=config.audio.sample_rate)
    st.plotly_chart(fig, width="stretch", theme=None)

    with st.expander(
        "Ecoacoustic Parameter Traceability & Scientific Disclaimers", expanded=False
    ):
        if display_records:
            first_params = display_records[0].get("parameters", {})
            st.json(first_params)
        st.markdown(
            "- **ACI (Acoustic Complexity Index)**: Intensity variability across time/frequency. High ACI reflects dynamic frequency modulation (e.g. bird dawn choruses), not direct species richness.\n"
            "- **NDSI (Normalized Difference Soundscape Index)**: Balance between biological ([2–8 kHz] default) and anthropogenic ([1–2 kHz] default) frequency energy in range [-1, +1].\n"
            "- **Acoustic Entropy (H = Ht * Hf)**: Product of temporal and spectral entropy in range [0, 1]. High values indicate evenly distributed acoustic energy across time and frequency.\n"
            "- **Bioacoustic Index (BI)**: Area under the dB power spectrum curve in the avian/biophonic band."
        )


# ======================================================================
# SPATIAL ANALYSIS
# ======================================================================


def _render_spatial_section(
    *,
    event_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    report: ResearchAnalyticsReport,
    config: AppConfig,
) -> None:
    """
    Render localization and aggregate acoustic spatial patterns.

    Individual x/y estimates can still be plotted from event rows when a
    SpatialSummary is unavailable.

    Aggregate occupancy and transition analysis require SpatialSummary.
    """

    st.markdown("## Spatial Acoustic Analysis")

    st.caption(
        (
            "Positions represent estimated acoustic-source "
            "locations. Consecutive positions are not "
            "confirmed trajectories of individual animals."
        )
    )

    if st.session_state.get("student_mode_active", True):
        render_student_explainer(
            topic_title="2D TDOA Acoustic Multilateration & Territory Mapping",
            what_is_it="Estimates an acoustic source position from measured arrival-time differences between synchronized microphones.",
            why_it_matters="Repeated, calibrated estimates can describe where detected sounds tend to originate and guide cautious follow-up study.",
            how_to_read="The triangle vertices are the 3 microphone nodes. The dots and heatmap show where sounds occurred.",
            real_world_example="A controlled speaker test at known positions can quantify array error before field locations are interpreted.",
        )

    spatial = _report_spatial(report)

    left_column, right_column = st.columns(2)

    # ==================================================================
    # INDIVIDUAL LOCALIZATION ESTIMATES
    # ==================================================================

    with left_column:
        figure = build_localization_scatter(
            event_rows,
            node_positions=config.localization.node_positions,
            max_points=config.dashboard.max_plot_points,
        )

        st.plotly_chart(
            figure,
            theme=None,
            width="stretch",
        )

    # ==================================================================
    # OCCUPANCY MAP
    # ==================================================================

    with right_column:
        if spatial is None:
            st.info(
                ("Aggregate spatial occupancy analysis is unavailable for this report.")
            )

        else:
            figure = build_spatial_occupancy_heatmap(
                spatial,
                node_positions=config.localization.node_positions,
            )

            st.plotly_chart(
                figure,
                theme=None,
                width="stretch",
            )

    # ==================================================================
    # TRANSITIONS
    # ==================================================================

    st.markdown("#### Acoustic-Location Transitions")

    if spatial is None:
        st.info(("Spatial transition analysis is unavailable for this report."))

        return

    if not (spatial.transitions):
        st.info(("No accepted spatial transitions are available for this session."))

        return

    transition_rows = [
        {
            "source_cell": transition.source_cell_id,
            "destination_cell": transition.destination_cell_id,
            "count": transition.transition_count,
            "conditional_probability": transition.probability,
        }
        for transition in spatial.transitions
    ]

    st.dataframe(
        transition_rows,
        width="stretch",
        hide_index=True,
    )


# ======================================================================
# BEHAVIOR-RELATED INDICATORS
# ======================================================================


def _render_behavior_section(
    report: ResearchAnalyticsReport,
) -> None:
    """
    Render conservative behavior-related acoustic indicators.
    """

    st.markdown("## Behavior-Related Acoustic Indicators")

    st.caption(
        (
            "These metrics describe patterns in acoustic "
            "detections. They are not direct observations "
            "of feeding, mating, territoriality, migration "
            "or individual movement."
        )
    )

    figure = build_behavior_indicator_chart(report.behavior_indicators)

    st.plotly_chart(
        figure,
        theme=None,
        width="stretch",
    )

    if report.behavior_indicators:
        rows = [
            {
                "indicator": indicator.name,
                "status": indicator.status.value,
                "score": indicator.score,
                "supporting_events": indicator.supporting_event_count,
                "description": indicator.description,
            }
            for indicator in report.behavior_indicators
        ]

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )


# ======================================================================
# EVENT ROW HELPERS
# ======================================================================


def _event_id(
    event: dict[
        str,
        Any,
    ],
) -> Any:
    """
    Resolve event identity from current or export-style row naming.

    Dashboard database rows currently commonly use:

        id

    while research export rows may use:

        event_id
    """

    if event.get("id") is not None:
        return event.get("id")

    return event.get(
        "event_id",
        "?",
    )


def _event_time(
    event: dict[
        str,
        Any,
    ],
) -> Any:
    """
    Resolve preferred scientific event timestamp.

    Priority
    --------
    1. event_time
    2. created_at
    3. database_created_at
    """

    return (
        event.get("event_time")
        or event.get("created_at")
        or event.get("database_created_at")
        or "unknown time"
    )


# ======================================================================
# EVENT AUDIO LABEL
# ======================================================================


def _event_selector_label(
    event: dict[
        str,
        Any,
    ],
) -> str:
    """
    Produce readable event selector text.
    """

    event_id = _event_id(event)

    label = event.get("classification_label") or "unclassified"

    event_time = _event_time(event)

    return f"Event {event_id} | {label} | {event_time}"


# ======================================================================
# EVENT AUDIO INSPECTOR
# ======================================================================


def _render_audio_inspector(
    *,
    event_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    config: AppConfig,
) -> None:
    """
    Render persisted synchronized event-audio inspection controls.
    """

    st.markdown("## Event Audio Inspector")

    if st.session_state.get("student_mode_active", True):
        render_student_explainer(
            topic_title="Audio Waveform & Spectrogram Inspector",
            what_is_it="A visual picture of sound. Horizontal axis is Time (seconds); Vertical axis is Frequency/Pitch (Hz); Color is Loudness (dB).",
            why_it_matters="Allows human researchers to verify AI detections, look for call harmonics, and listen to the recorded audio clip directly.",
            how_to_read="High bright stripes indicate high-pitched whistle notes. Low thick bands indicate deep bass rumbles.",
            real_world_example="Peacock alarm calls appear as repeated chevron shapes between 2 kHz and 4 kHz.",
        )

    if not (event_rows):
        st.info("No acoustic events are available.")

        return

    # ------------------------------------------------------------------
    # Newest event first in selector.
    # ------------------------------------------------------------------

    selectable_events = list(reversed(event_rows))

    selected_event = st.selectbox(
        "Select acoustic event",
        options=selectable_events,
        format_func=_event_selector_label,
        key="analysis_audio_event_selector",
    )

    # ==================================================================
    # DISCOVER NODE WAV FILES
    # ==================================================================

    try:
        audio_files = discover_event_audio_files(
            selected_event,
            expected_nodes=config.expected_nodes,
        )

    except Exception as exc:
        st.warning((f"Unable to inspect persisted event audio files: {exc}"))

        return

    if not (audio_files):
        st.info(("No valid persisted node WAV files were found for this event."))

        return

    # ==================================================================
    # NODE SELECTOR
    # ==================================================================

    selected_audio = st.selectbox(
        "Select microphone node",
        options=audio_files,
        format_func=lambda item: (
            f"Node {item.node_id} | {item.duration_s:.3f} s | {item.sample_rate} Hz"
        ),
        key="analysis_audio_node_selector",
    )

    if selected_audio is None:
        return

    metadata_columns = st.columns(4)

    with metadata_columns[0]:
        st.metric(
            "Node",
            selected_audio.node_id,
        )

    with metadata_columns[1]:
        st.metric(
            "Sample Rate",
            f"{selected_audio.sample_rate} Hz",
        )

    with metadata_columns[2]:
        st.metric(
            "Duration",
            f"{selected_audio.duration_s:.3f} s",
        )

    with metadata_columns[3]:
        st.metric(
            "File Size",
            (f"{selected_audio.file_size_bytes / 1024.0:.1f} KiB"),
        )

    # ==================================================================
    # AUDIO PREVIEW
    # ==================================================================

    try:
        preview = build_preview_wav_bytes(
            selected_audio,
            max_duration_s=config.dashboard.max_audio_preview_s,
        )

        st.audio(
            preview,
            format="audio/wav",
        )

    except Exception as exc:
        st.warning((f"Unable to create browser audio preview: {exc}"))

    # ==================================================================
    # WAVEFORM / SPECTROGRAM
    # ==================================================================

    left_column, right_column = st.columns(2)

    with left_column:
        try:
            waveform = build_waveform_figure(
                selected_audio,
                max_duration_s=config.dashboard.max_audio_preview_s,
                max_points=config.dashboard.max_plot_points,
            )

            st.plotly_chart(
                waveform,
                theme=None,
                width="stretch",
            )

        except Exception as exc:
            st.warning((f"Unable to display waveform: {exc}"))

    with right_column:
        try:
            spectrogram = build_spectrogram_figure(
                selected_audio,
                max_duration_s=config.dashboard.max_audio_preview_s,
                n_fft=config.dsp.n_fft,
                hop_length=config.dsp.hop_length,
            )

            st.plotly_chart(
                spectrogram,
                theme=None,
                width="stretch",
            )

        except Exception as exc:
            st.warning((f"Unable to display spectrogram: {exc}"))


# ======================================================================
# REPORT EXPORT
# ======================================================================


def _render_report_export(
    report: ResearchAnalyticsReport,
    *,
    session_id: int,
) -> None:
    """
    Provide JSON export of the generated research analytics report.
    """

    st.markdown("## Research Report Export")

    report_dictionary = report.to_dict()

    serialized = json.dumps(
        report_dictionary,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    st.download_button(
        label="Download Analytics JSON",
        data=serialized,
        file_name=(f"wildlife_analytics_{session_id:08X}.json"),
        mime="application/json",
        key="analysis_report_json_download",
    )

    with st.expander(
        "View Serialized Research Report",
        expanded=False,
    ):
        st.json(report_dictionary)


# ======================================================================
# MAIN ANALYSIS VIEW
# ======================================================================


def render_analysis_view(
    data_access: DashboardDataAccess,
    *,
    config: AppConfig | None = None,
) -> None:
    """
    Render complete historical/research analytics page.

    The current research workflow intentionally analyzes one acquisition
    session at a time.
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
    # HEADER
    # ==================================================================

    st.title("Wildlife Soundscape Research Analysis")

    st.caption(
        (
            "Historical temporal, environmental, "
            "classification and localization analysis "
            "over persisted acquisition data."
        )
    )

    if not (config.analytics.enabled):
        st.warning(("Research analytics are disabled in application configuration."))

        return

    # ==================================================================
    # LOAD SESSIONS
    # ==================================================================

    try:
        sessions = data_access.sessions()

    except Exception as exc:
        st.error((f"Unable to read acquisition sessions: {exc}"))

        return

    if not (sessions):
        st.info(("No persisted acquisition sessions are available for analysis."))

        return

    # ==================================================================
    # SESSION SELECTOR
    # ==================================================================

    selected_session = st.selectbox(
        "Acquisition session",
        options=sessions,
        format_func=_session_option_label,
        key="research_session_selector",
    )

    if selected_session is None:
        return

    try:
        session_id = int(selected_session["session_id"])

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        st.error(("Selected session contains an invalid session identifier."))

        return

    if not (0 <= session_id <= 0xFFFFFFFF):
        st.error(
            ("Selected session identifier is outside the Protocol-v4 uint32 range.")
        )

        return

    # ==================================================================
    # ANALYSIS OPTIONS
    # ==================================================================

    option_columns = st.columns(2)

    with option_columns[0]:
        include_environment = st.checkbox(
            "Environmental association analysis",
            value=True,
            key="research_include_environment",
        )

    with option_columns[1]:
        st.caption(
            (
                "Activity bin width: "
                f"{config.analytics.bucket_seconds} s | "
                "Spatial cell: "
                f"{config.analytics.cell_size_m:.3f} m"
            )
        )

    # ==================================================================
    # LOAD EVENT DATA
    # ==================================================================

    try:
        event_rows = data_access.analytics_events(session_id=session_id)

    except Exception as exc:
        st.error((f"Unable to load normalized acoustic event rows: {exc}"))

        return

    # ==================================================================
    # ENVIRONMENTAL OBSERVATION BINS
    # ==================================================================

    environmental_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    if include_environment:
        try:
            environmental_rows = data_access.environmental_bins(session_id=session_id)

        except Exception as exc:
            st.warning(
                (f"Environmental observation bins could not be generated: {exc}")
            )

            environmental_rows = []

    # ==================================================================
    # COMPLETE ANALYTICS REPORT
    # ==================================================================

    try:
        with st.spinner("Calculating research analytics..."):
            report = data_access.research_report(
                session_id=session_id,
                include_environment=include_environment,
            )

    except Exception as exc:
        st.error((f"Research analytics could not be generated: {exc}"))

        return

    if not isinstance(
        report,
        ResearchAnalyticsReport,
    ):
        st.error(("Research analytics returned an unexpected report type."))

        return

    # ==================================================================
    # OVERVIEW
    # ==================================================================

    _render_report_overview(
        report,
        session=selected_session,
    )

    _render_research_warnings(report)

    st.divider()

    # ==================================================================
    # ACTIVITY
    # ==================================================================

    _render_activity_section(
        event_rows=event_rows,
        report=report,
        config=config,
    )

    st.divider()

    # ==================================================================
    # ENVIRONMENT
    # ==================================================================

    if include_environment:
        _render_environment_section(
            environmental_rows=environmental_rows,
            report=report,
            config=config,
        )

        st.divider()

    # ==================================================================
    # CONTINUOUS ECOACOUSTIC INDICES
    # ==================================================================

    _render_soundscape_indices_section(
        data_access=data_access,
        session_id=session_id,
        config=config,
    )

    st.divider()

    # ==================================================================
    # SPATIAL
    # ==================================================================

    _render_spatial_section(
        event_rows=event_rows,
        report=report,
        config=config,
    )

    st.divider()

    # ==================================================================
    # BEHAVIOR INDICATORS
    # ==================================================================

    _render_behavior_section(report)

    st.divider()

    # ==================================================================
    # AUDIO INSPECTION
    # ==================================================================

    _render_audio_inspector(
        event_rows=event_rows,
        config=config,
    )

    st.divider()

    # ==================================================================
    # EXPORT
    # ==================================================================

    _render_report_export(
        report,
        session_id=session_id,
    )
