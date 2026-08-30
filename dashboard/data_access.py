"""
Read-only dashboard data-access layer.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module provides one controlled interface between the dashboard and:

    EventDatabase
    analytics.service
    analytics models
    application configuration

Dashboard views should not execute SQL directly.

Instead:

    dashboard view
        ↓
    DashboardDataAccess
        ↓
    EventDatabase
        ↓
    analytics
        ↓
    typed / normalized results


Responsibilities
----------------
This module handles:

    session discovery
    recent event retrieval
    individual event retrieval
    normalized research event rows
    normalized environmental telemetry
    environmental observation bins
    complete ResearchAnalyticsReport generation

This module does NOT:

    write database records
    modify acquisition sessions
    process DSP
    classify audio
    localize audio
    render Streamlit widgets
    create plots
    decode WAV files

Those responsibilities remain elsewhere.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


from datetime import (
    datetime,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
    Iterable,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from analytics.models import (
    ResearchAnalyticsReport,
)

from analytics.service import (
    build_research_analytics_report,
)

from config import (
    AppConfig,
    CONFIG,
)

from database import (
    EventDatabase,
)


# ======================================================================
# ROW CONVERSION
# ======================================================================


def row_to_dict(
    row: Any,
) -> dict[
    str,
    Any,
]:
    """
    Convert one database/query row into a normal dictionary.

    Supported inputs include:

        sqlite3.Row
        dict
        mapping-like objects

    A new dictionary is always returned so dashboard code cannot mutate
    database row objects accidentally.
    """

    if isinstance(
        row,
        dict,
    ):

        return dict(
            row
        )

    try:

        return dict(
            row
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            (
                "Dashboard row must be "
                "dictionary-like."
            )
        ) from exc


# ======================================================================
# ROW LIST CONVERSION
# ======================================================================


def rows_to_dicts(
    rows: Iterable[
        Any
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Convert an iterable of database rows into plain dictionaries.
    """

    return [
        row_to_dict(
            row
        )

        for row
        in rows
    ]


# ======================================================================
# OPTIONAL SESSION ID VALIDATION
# ======================================================================


def _optional_session_id(
    value: int | None,
) -> int | None:
    """
    Validate optional Protocol-v4 session identifier.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):

        raise TypeError(
            (
                "session_id must be an "
                "integer or None."
            )
        )

    if (
        value
        <= 0
    ):

        raise ValueError(
            (
                "session_id must be greater "
                "than 0."
            )
        )

    if (
        value
        > 0xFFFFFFFF
    ):

        raise ValueError(
            (
                "session_id exceeds "
                "Protocol-v4 uint32 range."
            )
        )

    return (
        int(
            value
        )
    )


# ======================================================================
# OPTIONAL POSITIVE INTEGER
# ======================================================================


def _optional_positive_int(
    value: int | None,
    *,
    name: str,
) -> int | None:
    """
    Validate optional positive integer.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):

        raise TypeError(
            f"{name} must be an integer or None."
        )

    if (
        value
        <= 0
    ):

        raise ValueError(
            f"{name} must be greater than 0."
        )

    return (
        int(
            value
        )
    )


# ======================================================================
# DASHBOARD DATA ACCESS
# ======================================================================


class DashboardDataAccess:
    """
    Read-only data service used by dashboard views.

    Parameters
    ----------
    config
        Full application configuration.

    database
        Optional pre-created EventDatabase.

        When omitted, the database configured through:

            config.persistence.database_path

        is opened.


    Design
    ------
    The object is intentionally lightweight.

    EventDatabase itself uses short-lived SQLite connections, therefore
    this class can safely be retained by the Streamlit application
    without keeping one persistent SQLite connection open.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        *,
        config: AppConfig = CONFIG,
        database: EventDatabase | None = None,
    ) -> None:

        if not isinstance(
            config,
            AppConfig,
        ):

            raise TypeError(
                "config must be an AppConfig."
            )

        if (
            database
            is not None
            and not isinstance(
                database,
                EventDatabase,
            )
        ):

            raise TypeError(
                (
                    "database must be an "
                    "EventDatabase or None."
                )
            )

        self.config = (
            config
        )

        self.database = (
            database
            if database
            is not None
            else EventDatabase(
                config.persistence.database_path
            )
        )

    # ==================================================================
    # DATABASE PATH
    # ==================================================================

    @property
    def database_path(
        self,
    ) -> Path:
        """
        Path of the currently opened event database.
        """

        return (
            self.database.path
        )

    # ==================================================================
    # SESSION LIST
    # ==================================================================

    def sessions(
        self,
        *,
        limit: int | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """
        Return acquisition sessions newest first.

        If ``limit`` is omitted, the configured dashboard limit is used.
        """

        if (
            limit
            is None
        ):

            limit = (
                self.config
                .dashboard
                .session_list_limit
            )

        limit = _optional_positive_int(
            limit,
            name=
                "limit",
        )

        rows = (
            self.database.list_sessions(
                limit=
                    limit,
            )
        )

        return rows_to_dicts(
            rows
        )

    # ==================================================================
    # LATEST SESSION
    # ==================================================================

    def latest_session(
        self,
    ) -> dict[
        str,
        Any,
    ] | None:
        """
        Return the newest persisted acquisition session.
        """

        rows = self.sessions(
            limit=
                1
        )

        if not (
            rows
        ):

            return (
                None
            )

        return (
            rows[
                0
            ]
        )

    # ==================================================================
    # RECENT EVENTS
    # ==================================================================

    def recent_events(
        self,
        *,
        limit: int | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """
        Return newest persisted acoustic events.

        These rows use the legacy/current event query and therefore
        contain:

            core event fields
            DSP features
            classifications

        For research-time analysis use ``analytics_events()`` instead.
        """

        if (
            limit
            is None
        ):

            limit = (
                self.config
                .dashboard
                .recent_events_limit
            )

        limit = _optional_positive_int(
            limit,
            name=
                "limit",
        )

        if (
            limit
            is None
        ):

            return (
                []
            )

        rows = (
            self.database.recent_events(
                limit=
                    limit
            )
        )

        return rows_to_dicts(
            rows
        )

    # ==================================================================
    # ONE EVENT
    # ==================================================================

    def event(
        self,
        event_id: int,
    ) -> dict[
        str,
        Any,
    ] | None:
        """
        Return one event including DSP and classification fields.
        """

        if (
            isinstance(
                event_id,
                bool,
            )
            or not isinstance(
                event_id,
                int,
            )
        ):

            raise TypeError(
                "event_id must be an integer."
            )

        if (
            event_id
            <= 0
        ):

            raise ValueError(
                "event_id must be greater than 0."
            )

        row = (
            self.database.get_event(
                event_id
            )
        )

        if (
            row
            is None
        ):

            return (
                None
            )

        return row_to_dict(
            row
        )

    # ==================================================================
    # NORMALIZED ANALYTICS EVENTS
    # ==================================================================

    def analytics_events(
        self,
        *,
        session_id: int | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """
        Return events on the reconstructed acquisition timeline.

        Each result contains:

            event_time
            event_end_time

        in addition to event, DSP and classification fields.
        """

        session_id = (
            _optional_session_id(
                session_id
            )
        )

        return (
            self.database
            .analytics_event_rows(
                sample_rate=
                    self.config
                    .audio
                    .sample_rate,

                session_id=
                    session_id,

                start=
                    start,

                end=
                    end,
            )
        )

    # ==================================================================
    # NORMALIZED TELEMETRY
    # ==================================================================

    def telemetry(
        self,
        *,
        session_id: int | None = None,
        node_id: int | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """
        Return environmental telemetry on the reconstructed acquisition
        timeline.
        """

        session_id = (
            _optional_session_id(
                session_id
            )
        )

        if (
            node_id
            is not None
        ):

            if (
                isinstance(
                    node_id,
                    bool,
                )
                or not isinstance(
                    node_id,
                    int,
                )
            ):

                raise TypeError(
                    (
                        "node_id must be an "
                        "integer or None."
                    )
                )

            if not (
                1
                <= node_id
                <= 255
            ):

                raise ValueError(
                    (
                        "node_id must lie "
                        "between 1 and 255."
                    )
                )

        return (
            self.database
            .analytics_telemetry_rows(
                sample_rate=
                    self.config
                    .audio
                    .sample_rate,

                session_id=
                    session_id,

                node_id=
                    node_id,

                start=
                    start,

                end=
                    end,
            )
        )

    # ==================================================================
    # ENVIRONMENTAL OBSERVATION BINS
    # ==================================================================

    def environmental_bins(
        self,
        *,
        session_id: int,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        bucket_seconds: int | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """
        Return regular environmental/activity observation bins for one
        acquisition session.

        Zero-event bins are retained.

        These rows are suitable for:

            analytics.environmental
        """

        session_id = (
            _optional_session_id(
                session_id
            )
        )

        if (
            session_id
            is None
        ):

            raise ValueError(
                "session_id is required."
            )

        if (
            bucket_seconds
            is None
        ):

            bucket_seconds = (
                self.config
                .analytics
                .bucket_seconds
            )

        if (
            isinstance(
                bucket_seconds,
                bool,
            )
            or not isinstance(
                bucket_seconds,
                int,
            )
        ):

            raise TypeError(
                (
                    "bucket_seconds must "
                    "be an integer."
                )
            )

        if (
            bucket_seconds
            <= 0
        ):

            raise ValueError(
                (
                    "bucket_seconds must "
                    "be greater than 0."
                )
            )

        return (
            self.database
            .analytics_environmental_bins(
                session_id=
                    session_id,

                bucket_seconds=
                    bucket_seconds,

                sample_rate=
                    self.config
                    .audio
                    .sample_rate,

                node_id=
                    self.config
                    .analytics
                    .environmental_node_id,

                start=
                    start,

                end=
                    end,
            )
        )

    # ==================================================================
    # ENVIRONMENTAL BINS ACROSS SESSIONS
    # ==================================================================

    def environmental_bins_all_sessions(
        self,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        bucket_seconds: int | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """
        Build environmental observation bins independently for every
        persisted acquisition session and concatenate them.

        Important
        ---------
        Bins are NEVER created across downtime between sessions.

        Example:

            session A:
                08:00 -> 10:00

            system off:
                10:00 -> 18:00

            session B:
                18:00 -> 20:00

        The 10:00 -> 18:00 interval is not represented as wildlife
        inactivity because the system was not acquiring data.
        """

        if (
            bucket_seconds
            is None
        ):

            bucket_seconds = (
                self.config
                .analytics
                .bucket_seconds
            )

        sessions = self.database.list_sessions(
            limit=
                None
        )

        result: list[
            dict[
                str,
                Any,
            ]
        ] = []

        # --------------------------------------------------------------
        # list_sessions() returns newest first.
        #
        # Reverse here so cross-session research observations are
        # chronological.
        # --------------------------------------------------------------

        for session_row in reversed(
            sessions
        ):

            session = row_to_dict(
                session_row
            )

            session_id = int(
                session[
                    "session_id"
                ]
            )

            bins = (
                self.environmental_bins(
                    session_id=
                        session_id,

                    start=
                        start,

                    end=
                        end,

                    bucket_seconds=
                        bucket_seconds,
                )
            )

            result.extend(
                bins
            )

        return (
            result
        )

    # ==================================================================
    # COMPLETE RESEARCH REPORT
    # ==================================================================

    def research_report(
        self,
        *,
        session_id: int | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        include_environment: bool = True,
        generated_at: datetime | None = None,
    ) -> ResearchAnalyticsReport:
        """
        Build the complete typed research analytics report.

        Parameters
        ----------
        session_id
            Optional session filter.

            When omitted:
                event analytics include all sessions.

                environmental bins are generated separately within each
                acquisition session and then concatenated.


        include_environment
            When False, environmental correlation analysis is skipped.


        Important
        ---------
        Event-time ordering across sessions relies on each session's
        persisted laptop wall-clock anchor.

        TDOA itself remains based exclusively on synchronized
        sampleIndex/waveform timing and is unrelated to these dashboard
        wall-clock timestamps.
        """

        session_id = (
            _optional_session_id(
                session_id
            )
        )

        if not isinstance(
            include_environment,
            bool,
        ):

            raise TypeError(
                (
                    "include_environment "
                    "must be bool."
                )
            )

        # ==============================================================
        # EVENTS
        # ==============================================================

        event_rows = self.analytics_events(
            session_id=
                session_id,

            start=
                start,

            end=
                end,
        )

        # ==============================================================
        # ENVIRONMENTAL OBSERVATION BINS
        # ==============================================================

        environmental_rows: (
            list[
                dict[
                    str,
                    Any,
                ]
            ]
            | None
        )

        if not (
            include_environment
        ):

            environmental_rows = (
                None
            )

        elif (
            session_id
            is not None
        ):

            environmental_rows = (
                self.environmental_bins(
                    session_id=
                        session_id,

                    start=
                        start,

                    end=
                        end,

                    bucket_seconds=
                        self.config
                        .analytics
                        .bucket_seconds,
                )
            )

        else:

            environmental_rows = (
                self
                .environmental_bins_all_sessions(
                    start=
                        start,

                    end=
                        end,

                    bucket_seconds=
                        self.config
                        .analytics
                        .bucket_seconds,
                )
            )

        # ==============================================================
        # ANALYTICS SERVICE
        # ==============================================================

        return build_research_analytics_report(
            event_rows,

            environmental_rows=
                environmental_rows,

            sample_rate=
                self.config
                .audio
                .sample_rate,

            timestamp_key=
                "event_time",

            bucket_seconds=
                self.config
                .analytics
                .bucket_seconds,

            start=
                start,

            end=
                end,

            cell_size_m=
                self.config
                .analytics
                .cell_size_m,

            max_grid_cells=
                self.config
                .analytics
                .max_grid_cells,

            max_transition_gap_s=
                self.config
                .analytics
                .max_transition_gap_s,

            same_class_transitions_only=
                self.config
                .analytics
                .same_class_transitions_only,

            alpha=
                self.config
                .analytics
                .environmental_alpha,

            environmental_min_samples=
                self.config
                .analytics
                .environmental_min_samples,

            neutral_threshold=
                self.config
                .analytics
                .neutral_threshold,

            min_activity_events=
                self.config
                .analytics
                .min_activity_events,

            min_localized_events=
                self.config
                .analytics
                .min_localized_events,

            min_transitions=
                self.config
                .analytics
                .min_transitions,

            generated_at=
                generated_at,
        )

    # ==================================================================
    # SERIALIZED RESEARCH REPORT
    # ==================================================================

    def research_report_dict(
        self,
        *,
        session_id: int | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        include_environment: bool = True,
        generated_at: datetime | None = None,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Return a JSON/export-ready research report dictionary.
        """

        report = self.research_report(
            session_id=
                session_id,

            start=
                start,

            end=
                end,

            include_environment=
                include_environment,

            generated_at=
                generated_at,
        )

        return (
            report.to_dict()
        )

    # ==================================================================
    # DASHBOARD SNAPSHOT
    # ==================================================================

    def snapshot(
        self,
        *,
        recent_limit: int | None = None,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Return a lightweight dashboard overview.

        Intended for the future live view.

        This deliberately avoids running the complete research analytics
        pipeline on every live refresh.
        """

        sessions = self.sessions(
            limit=
                1
        )

        events = self.recent_events(
            limit=
                recent_limit,
        )

        latest_session = (
            sessions[
                0
            ]

            if sessions

            else None
        )

        latest_event = (
            events[
                0
            ]

            if events

            else None
        )

        return {
            "database_path":
                str(
                    self.database_path
                ),

            "latest_session":
                latest_session,

            "latest_event":
                latest_event,

            "recent_events":
                events,

            "recent_event_count":
                len(
                    events
                ),
        }


# ======================================================================
# DEFAULT DATA ACCESS FACTORY
# ======================================================================


def create_dashboard_data_access(
    *,
    config: AppConfig = CONFIG,
) -> DashboardDataAccess:
    """
    Create the standard dashboard read service.

    Keeping construction in one helper will also make later Streamlit
    resource caching straightforward.
    """

    return DashboardDataAccess(
        config=
            config
    )