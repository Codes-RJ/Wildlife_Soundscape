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


from analytics.activity import (
    build_activity_bins,
)

from analytics.models import (
    ResearchAnalyticsReport,
)

from config import (
    AppConfig,
)

from dashboard.audio_view import (
    build_preview_wav_bytes,
    build_spectrogram_figure,
    build_waveform_figure,
    discover_event_audio_files,
)

from dashboard.data_access import (
    DashboardDataAccess,
)

from dashboard.plots import (
    activity_summary_metrics,
    build_activity_timeline,
    build_behavior_indicator_chart,
    build_class_distribution,
    build_environmental_association_matrix,
    build_environmental_timeseries,
    build_localization_scatter,
    build_spatial_occupancy_heatmap,
)


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

        session_id = int(
            value
        )

    except (
        TypeError,
        ValueError,
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

    session_id = (
        _display_session_id(
            session.get(
                "session_id"
            )
        )
    )

    label = str(
        session.get(
            "label",
            "Session",
        )
    ).strip()

    started_at = str(
        session.get(
            "started_at",
            "",
        )
    ).strip()

    state = (
        "Active"

        if session.get(
            "stopped_at"
        )
        is None

        else "Stopped"
    )

    return (
        f"{label} | "
        f"{session_id} | "
        f"{started_at} | "
        f"{state}"
    )


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
    """

    metrics = (
        activity_summary_metrics(
            report.activity
        )
    )

    st.markdown(
        "### Research Overview"
    )

    first_row = st.columns(
        5
    )

    # ==============================================================
    # SESSION
    # ==============================================================

    with first_row[
        0
    ]:

        st.metric(
            "Session",
            _display_session_id(
                session.get(
                    "session_id"
                )
            ),
        )

    # ==============================================================
    # EVENTS
    # ==============================================================

    with first_row[
        1
    ]:

        st.metric(
            "Detected Events",
            metrics[
                "total_events"
            ],
        )

    # ==============================================================
    # ACTIVE AUDIO
    # ==============================================================

    with first_row[
        2
    ]:

        st.metric(
            "Event Duration",
            (
                f"{metrics['total_active_duration_s']:.2f} s"
            ),
        )

    # ==============================================================
    # LOCALIZATION COVERAGE
    # ==============================================================

    with first_row[
        3
    ]:

        st.metric(
            "Localization Coverage",
            (
                f"{report.spatial.localization_coverage * 100.0:.1f}%"
            ),
        )

    # ==============================================================
    # CLASSES
    # ==============================================================

    with first_row[
        4
    ]:

        st.metric(
            "Acoustic Classes",
            metrics[
                "classified_classes"
            ],
        )

    second_row = st.columns(
        4
    )

    # ==============================================================
    # MEAN EVENT DURATION
    # ==============================================================

    with second_row[
        0
    ]:

        st.metric(
            "Mean Event Duration",
            (
                f"{metrics['mean_event_duration_s']:.3f} s"
            ),
        )

    # ==============================================================
    # PEAK ACTIVITY HOUR
    # ==============================================================

    with second_row[
        1
    ]:

        peak_hour = (
            metrics[
                "peak_activity_hour"
            ]
        )

        st.metric(
            "Peak Activity Hour",
            (
                f"{peak_hour:02d}:00"

                if peak_hour
                is not None

                else "—"
            ),
        )

    # ==============================================================
    # LOCALIZED EVENTS
    # ==============================================================

    with second_row[
        2
    ]:

        st.metric(
            "Localized Events",
            (
                f"{report.spatial.localized_event_count}"
                f" / "
                f"{report.spatial.total_event_count}"
            ),
        )

    # ==============================================================
    # HOTSPOT
    # ==============================================================

    with second_row[
        3
    ]:

        st.metric(
            "Acoustic Hotspot",
            (
                report.spatial.hotspot_cell_id

                if report.spatial.hotspot_cell_id
                is not None

                else "—"
            ),
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

    if not (
        report.warnings
    ):

        return

    with st.expander(
        "Research Interpretation & Data Quality",
        expanded=
            True,
    ):

        for warning in (
            report.warnings
        ):

            st.warning(
                warning
            )


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
    """

    st.markdown(
        "## Temporal Acoustic Activity"
    )

    # ==============================================================
    # BUILD ACTIVITY BINS
    # ==============================================================

    activity_bins = (
        build_activity_bins(
            event_rows,

            bucket_seconds=
                config
                .analytics
                .bucket_seconds,

            sample_rate=
                config
                .audio
                .sample_rate,

            timestamp_key=
                "event_time",
        )
    )

    left_column, right_column = st.columns(
        2
    )

    # ==============================================================
    # ACTIVITY TIMELINE
    # ==============================================================

    with left_column:

        figure = (
            build_activity_timeline(
                activity_bins,

                max_points=
                    config
                    .dashboard
                    .max_plot_points,
            )
        )

        st.plotly_chart(
            figure,
            use_container_width=
                True,
        )

    # ==============================================================
    # CLASS DISTRIBUTION
    # ==============================================================

    with right_column:

        figure = (
            build_class_distribution(
                report.activity
            )
        )

        st.plotly_chart(
            figure,
            use_container_width=
                True,
        )

    # ==============================================================
    # CLASS SUMMARY TABLE
    # ==============================================================

    if (
        report.activity.class_summaries
    ):

        st.markdown(
            "#### Acoustic Class Summary"
        )

        rows = [
            {
                "class":
                    summary.class_label,

                "events":
                    summary.event_count,

                "total_duration_s":
                    summary.total_duration_s,

                "mean_confidence":
                    summary.mean_confidence,

                "event_proportion":
                    summary.proportion_of_events,
            }

            for summary
            in report.activity.class_summaries
        ]

        st.dataframe(
            rows,
            use_container_width=
                True,
            hide_index=
                True,
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

    st.markdown(
        "## Environmental Analysis"
    )

    st.caption(
        (
            "Environmental associations use regular "
            "within-session observation bins, including "
            "periods with zero detected acoustic events."
        )
    )

    if not (
        environmental_rows
    ):

        st.info(
            (
                "No environmental observation bins "
                "are available for this session."
            )
        )

        return

    # ==============================================================
    # ENVIRONMENTAL TIME SERIES
    # ==============================================================

    figure = (
        build_environmental_timeseries(
            environmental_rows,

            timestamp_key=
                "bucket_start",

            max_points=
                config
                .dashboard
                .max_plot_points,
        )
    )

    st.plotly_chart(
        figure,
        use_container_width=
            True,
    )

    # ==============================================================
    # CORRELATION MATRIX
    # ==============================================================

    st.markdown(
        "#### Environment ↔ Acoustic Activity Associations"
    )

    st.caption(
        (
            "Spearman correlation describes monotonic "
            "association. It does not establish "
            "environmental causation."
        )
    )

    figure = (
        build_environmental_association_matrix(
            report.environmental_associations
        )
    )

    st.plotly_chart(
        figure,
        use_container_width=
            True,
    )

    # ==============================================================
    # ASSOCIATION TABLE
    # ==============================================================

    if (
        report.environmental_associations
    ):

        rows = [
            {
                "environmental_variable":
                    association.environmental_variable,

                "response_variable":
                    association.response_variable,

                "samples":
                    association.sample_count,

                "spearman_rho":
                    association.coefficient,

                "p_value":
                    association.p_value,

                "direction":
                    association.direction.value,

                "strength":
                    association.strength.value,

                "significant_raw_alpha":
                    association.statistically_significant,

                "method":
                    association.method,
            }

            for association
            in report.environmental_associations
        ]

        st.dataframe(
            rows,
            use_container_width=
                True,
            hide_index=
                True,
        )

        st.caption(
            (
                "Significance flags currently use the "
                "configured raw alpha threshold. "
                "Multiple-comparison correction should "
                "be applied in formal hypothesis testing."
            )
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
    """

    st.markdown(
        "## Spatial Acoustic Analysis"
    )

    st.caption(
        (
            "Positions represent estimated acoustic-source "
            "locations. Consecutive positions are not "
            "confirmed trajectories of individual animals."
        )
    )

    left_column, right_column = st.columns(
        2
    )

    # ==============================================================
    # INDIVIDUAL LOCALIZATION ESTIMATES
    # ==============================================================

    with left_column:

        figure = (
            build_localization_scatter(
                event_rows,

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
            figure,
            use_container_width=
                True,
        )

    # ==============================================================
    # OCCUPANCY MAP
    # ==============================================================

    with right_column:

        figure = (
            build_spatial_occupancy_heatmap(
                report.spatial,

                node_positions=
                    config
                    .localization
                    .node_positions,
            )
        )

        st.plotly_chart(
            figure,
            use_container_width=
                True,
        )

    # ==============================================================
    # TRANSITIONS
    # ==============================================================

    st.markdown(
        "#### Acoustic-Location Transitions"
    )

    if not (
        report.spatial.transitions
    ):

        st.info(
            (
                "No accepted spatial transitions "
                "are available for this session."
            )
        )

    else:

        transition_rows = [
            {
                "source_cell":
                    transition.source_cell_id,

                "destination_cell":
                    transition.destination_cell_id,

                "count":
                    transition.transition_count,

                "conditional_probability":
                    transition.probability,
            }

            for transition
            in report.spatial.transitions
        ]

        st.dataframe(
            transition_rows,
            use_container_width=
                True,
            hide_index=
                True,
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

    st.markdown(
        "## Behavior-Related Acoustic Indicators"
    )

    st.caption(
        (
            "These metrics describe patterns in acoustic "
            "detections. They are not direct observations "
            "of feeding, mating, territoriality, migration "
            "or individual movement."
        )
    )

    figure = (
        build_behavior_indicator_chart(
            report.behavior_indicators
        )
    )

    st.plotly_chart(
        figure,
        use_container_width=
            True,
    )

    if (
        report.behavior_indicators
    ):

        rows = [
            {
                "indicator":
                    indicator.name,

                "status":
                    indicator.status.value,

                "score":
                    indicator.score,

                "supporting_events":
                    indicator.supporting_event_count,

                "description":
                    indicator.description,
            }

            for indicator
            in report.behavior_indicators
        ]

        st.dataframe(
            rows,
            use_container_width=
                True,
            hide_index=
                True,
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

    event_id = (
        event.get(
            "id",
            "?"
        )
    )

    label = (
        event.get(
            "classification_label"
        )
        or "unclassified"
    )

    event_time = (
        event.get(
            "event_time"
        )
        or event.get(
            "created_at"
        )
        or "unknown time"
    )

    return (
        f"Event {event_id} | "
        f"{label} | "
        f"{event_time}"
    )


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

    st.markdown(
        "## Event Audio Inspector"
    )

    if not (
        event_rows
    ):

        st.info(
            "No acoustic events are available."
        )

        return

    # --------------------------------------------------------------
    # Newest event first in selector.
    # --------------------------------------------------------------

    selectable_events = list(
        reversed(
            event_rows
        )
    )

    selected_event = st.selectbox(
        "Select acoustic event",

        options=
            selectable_events,

        format_func=
            _event_selector_label,

        key=
            "analysis_audio_event_selector",
    )

    if (
        selected_event
        is None
    ):

        return

    # ==============================================================
    # DISCOVER NODE WAV FILES
    # ==============================================================

    audio_files = (
        discover_event_audio_files(
            selected_event,

            expected_nodes=
                config
                .expected_nodes,
        )
    )

    if not (
        audio_files
    ):

        st.info(
            (
                "No valid persisted node WAV files "
                "were found for this event."
            )
        )

        return

    # ==============================================================
    # NODE SELECTOR
    # ==============================================================

    selected_audio = st.selectbox(
        "Select microphone node",

        options=
            audio_files,

        format_func=
            lambda item:
                (
                    f"Node {item.node_id} | "
                    f"{item.duration_s:.3f} s | "
                    f"{item.sample_rate} Hz"
                ),

        key=
            "analysis_audio_node_selector",
    )

    if (
        selected_audio
        is None
    ):

        return

    metadata_columns = st.columns(
        4
    )

    with metadata_columns[
        0
    ]:

        st.metric(
            "Node",
            selected_audio.node_id,
        )

    with metadata_columns[
        1
    ]:

        st.metric(
            "Sample Rate",
            f"{selected_audio.sample_rate} Hz",
        )

    with metadata_columns[
        2
    ]:

        st.metric(
            "Duration",
            f"{selected_audio.duration_s:.3f} s",
        )

    with metadata_columns[
        3
    ]:

        st.metric(
            "File Size",
            (
                f"{selected_audio.file_size_bytes / 1024.0:.1f} KiB"
            ),
        )

    # ==============================================================
    # AUDIO PREVIEW
    # ==============================================================

    try:

        preview = (
            build_preview_wav_bytes(
                selected_audio,

                max_duration_s=
                    config
                    .dashboard
                    .max_audio_preview_s,
            )
        )

        st.audio(
            preview,
            format=
                "audio/wav",
        )

    except Exception as exc:

        st.warning(
            (
                "Unable to create browser audio "
                f"preview: {exc}"
            )
        )

    # ==============================================================
    # WAVEFORM / SPECTROGRAM
    # ==============================================================

    left_column, right_column = st.columns(
        2
    )

    with left_column:

        try:

            waveform = (
                build_waveform_figure(
                    selected_audio,

                    max_duration_s=
                        config
                        .dashboard
                        .max_audio_preview_s,

                    max_points=
                        config
                        .dashboard
                        .max_plot_points,
                )
            )

            st.plotly_chart(
                waveform,
                use_container_width=
                    True,
            )

        except Exception as exc:

            st.warning(
                (
                    "Unable to display waveform: "
                    f"{exc}"
                )
            )

    with right_column:

        try:

            spectrogram = (
                build_spectrogram_figure(
                    selected_audio,

                    max_duration_s=
                        config
                        .dashboard
                        .max_audio_preview_s,

                    n_fft=
                        config
                        .dsp
                        .n_fft,

                    hop_length=
                        config
                        .dsp
                        .hop_length,
                )
            )

            st.plotly_chart(
                spectrogram,
                use_container_width=
                    True,
            )

        except Exception as exc:

            st.warning(
                (
                    "Unable to display spectrogram: "
                    f"{exc}"
                )
            )


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

    st.markdown(
        "## Research Report Export"
    )

    report_dictionary = (
        report.to_dict()
    )

    serialized = json.dumps(
        report_dictionary,
        indent=
            2,
        ensure_ascii=
            False,
        allow_nan=
            False,
    )

    st.download_button(
        label=
            "Download Analytics JSON",

        data=
            serialized,

        file_name=
            (
                "wildlife_analytics_"
                f"{session_id:08X}.json"
            ),

        mime=
            "application/json",

        key=
            "analysis_report_json_download",
    )

    with st.expander(
        "View Serialized Research Report",
        expanded=
            False,
    ):

        st.json(
            report_dictionary
        )


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
    # HEADER
    # ==============================================================

    st.title(
        "Wildlife Soundscape Research Analysis"
    )

    st.caption(
        (
            "Historical temporal, environmental, "
            "classification and localization analysis "
            "over persisted acquisition data."
        )
    )

    if not (
        config.analytics.enabled
    ):

        st.warning(
            (
                "Research analytics are disabled "
                "in application configuration."
            )
        )

        return

    # ==============================================================
    # LOAD SESSIONS
    # ==============================================================

    try:

        sessions = (
            data_access.sessions()
        )

    except Exception as exc:

        st.error(
            (
                "Unable to read acquisition sessions: "
                f"{exc}"
            )
        )

        return

    if not (
        sessions
    ):

        st.info(
            (
                "No persisted acquisition sessions "
                "are available for analysis."
            )
        )

        return

    # ==============================================================
    # SESSION SELECTOR
    # ==============================================================

    selected_session = st.selectbox(
        "Acquisition session",

        options=
            sessions,

        format_func=
            _session_option_label,

        key=
            "research_session_selector",
    )

    if (
        selected_session
        is None
    ):

        return

    try:

        session_id = int(
            selected_session[
                "session_id"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        st.error(
            (
                "Selected session contains an "
                "invalid session identifier."
            )
        )

        return

    # ==============================================================
    # ANALYSIS OPTIONS
    # ==============================================================

    option_columns = st.columns(
        2
    )

    with option_columns[
        0
    ]:

        include_environment = st.checkbox(
            "Environmental association analysis",

            value=
                True,

            key=
                "research_include_environment",
        )

    with option_columns[
        1
    ]:

        st.caption(
            (
                "Activity bin width: "
                f"{config.analytics.bucket_seconds} s | "
                "Spatial cell: "
                f"{config.analytics.cell_size_m:.3f} m"
            )
        )

    # ==============================================================
    # LOAD EVENT DATA
    # ==============================================================

    try:

        event_rows = (
            data_access.analytics_events(
                session_id=
                    session_id
            )
        )

    except Exception as exc:

        st.error(
            (
                "Unable to load normalized acoustic "
                f"event rows: {exc}"
            )
        )

        return

    # ==============================================================
    # ENVIRONMENTAL OBSERVATION BINS
    # ==============================================================

    environmental_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    if (
        include_environment
    ):

        try:

            environmental_rows = (
                data_access.environmental_bins(
                    session_id=
                        session_id
                )
            )

        except Exception as exc:

            st.warning(
                (
                    "Environmental observation bins "
                    f"could not be generated: {exc}"
                )
            )

            environmental_rows = (
                []
            )

    # ==============================================================
    # COMPLETE ANALYTICS REPORT
    # ==============================================================

    try:

        with st.spinner(
            "Calculating research analytics..."
        ):

            report = (
                data_access.research_report(
                    session_id=
                        session_id,

                    include_environment=
                        include_environment,
                )
            )

    except Exception as exc:

        st.error(
            (
                "Research analytics could not be "
                f"generated: {exc}"
            )
        )

        return

    # ==============================================================
    # OVERVIEW
    # ==============================================================

    _render_report_overview(
        report,

        session=
            selected_session,
    )

    _render_research_warnings(
        report
    )

    st.divider()

    # ==============================================================
    # ACTIVITY
    # ==============================================================

    _render_activity_section(
        event_rows=
            event_rows,

        report=
            report,

        config=
            config,
    )

    st.divider()

    # ==============================================================
    # ENVIRONMENT
    # ==============================================================

    if (
        include_environment
    ):

        _render_environment_section(
            environmental_rows=
                environmental_rows,

            report=
                report,

            config=
                config,
        )

        st.divider()

    # ==============================================================
    # SPATIAL
    # ==============================================================

    _render_spatial_section(
        event_rows=
            event_rows,

        report=
            report,

        config=
            config,
    )

    st.divider()

    # ==============================================================
    # BEHAVIOR INDICATORS
    # ==============================================================

    _render_behavior_section(
        report
    )

    st.divider()

    # ==============================================================
    # AUDIO INSPECTION
    # ==============================================================

    _render_audio_inspector(
        event_rows=
            event_rows,

        config=
            config,
    )

    st.divider()

    # ==============================================================
    # EXPORT
    # ==============================================================

    _render_report_export(
        report,

        session_id=
            session_id,
    )