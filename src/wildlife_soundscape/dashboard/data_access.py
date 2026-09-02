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
    complete single-session ResearchAnalyticsReport generation


Research-session policy
-----------------------
A complete ResearchAnalyticsReport is intentionally restricted to one
acquisition session.

This prevents:

    transitions between events from different acquisition sessions

    downtime between sessions being interpreted as observed inactivity

    environmental observations from unrelated acquisition periods being
    treated as one continuous ecological observation window

    discontinuous sample timelines being interpreted as one acquisition

Cross-session research should instead be implemented explicitly using:

    per-session reports
        ↓
    exposure-aware aggregation

rather than by concatenating event streams before behavioral/spatial
analysis.


This module does NOT
--------------------
    write event records
    start or stop acquisition sessions
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


from wildlife_soundscape.analytics.models import (
    ResearchAnalyticsReport,
)


from wildlife_soundscape.analytics.service import (
    build_research_analytics_report,
)


from wildlife_soundscape.core.config import (
    AppConfig,
    CONFIG,
)


from wildlife_soundscape.storage.database import (
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
# REQUIRED SESSION ID VALIDATION
# ======================================================================


def _required_session_id(
    value: int,
) -> int:
    """
    Validate one mandatory Protocol-v4 acquisition-session identifier.

    Complete research reports require an explicit session because the
    spatial-transition and temporal-analysis layers assume one continuous
    observation period.
    """

    result = (
        _optional_session_id(
            value
        )
    )

    if (
        result
        is None
    ):

        raise ValueError(
            (
                "session_id is required for "
                "single-session research analysis."
            )
        )

    return (
        result
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
    Read-oriented data service used by dashboard views.

    Parameters
    ----------
    config
        Full application configuration.

    database
        Optional pre-created EventDatabase.

        When omitted, the database configured through:

            config.persistence.database_path

        is opened.


    Database lifecycle note
    -----------------------
    EventDatabase uses short-lived SQLite connections.

    Constructing EventDatabase may initialize or migrate the SQLite
    schema, but this dashboard service itself does not create, modify or
    delete research event records.
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

        limit = (
            _optional_positive_int(
                limit,
                name=
                    "limit",
            )
        )

        rows = (
            self.database.list_sessions(
                limit=
                    limit,
            )
        )

        return (
            rows_to_dicts(
                rows
            )
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

        rows = (
            self.sessions(
                limit=
                    1
            )
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

        These rows use the standard event query and contain:

            core event fields
            DSP features
            classifications

        For scientific timeline analysis use ``analytics_events()``
        instead.
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

        limit = (
            _optional_positive_int(
                limit,
                name=
                    "limit",
            )
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

        return (
            rows_to_dicts(
                rows
            )
        )

    # ==================================================================
    # SOUNDSCAPE INDICES
    # ==================================================================

    def soundscape_indices(
        self,
        *,
        session_id: int | None = None,
        node_id: int | None = None,
        limit: int | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """
        Return persisted continuous ecoacoustic index records.
        """
        session_id = _optional_session_id(session_id)
        node_id = (
            _optional_positive_int(node_id, name="node_id")
            if node_id is not None
            else None
        )
        limit_val = (
            _optional_positive_int(limit, name="limit")
            or 1000
        )

        return self.database.get_soundscape_indices(
            session_id=session_id,
            node_id=node_id,
            limit=limit_val,
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

        return (
            row_to_dict(
                row
            )
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


        Low-level policy
        ----------------
        ``session_id`` remains optional here because this method is a
        normalized data-retrieval primitive.

        Callers performing behavioral/spatial research must maintain
        explicit session boundaries.
        """

        session_id = (
            _optional_session_id(
                session_id
            )
        )

        rows = (
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

        return [
            row_to_dict(
                row
            )

            for row
            in rows
        ]

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

        rows = (
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

        return [
            row_to_dict(
                row
            )

            for row
            in rows
        ]

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
            _required_session_id(
                session_id
            )
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

        rows = (
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

        return [
            row_to_dict(
                row
            )

            for row
            in rows
        ]

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

        This is a low-level research-data helper.

        It does not imply that the returned rows should be treated as one
        continuous wildlife observation session.
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

        sessions = (
            self.database.list_sessions(
                limit=
                    None
            )
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
        # Reverse here so the returned bin collection is chronological.
        # --------------------------------------------------------------

        for session_row in reversed(
            sessions
        ):

            session = (
                row_to_dict(
                    session_row
                )
            )

            session_id = (
                _required_session_id(
                    int(
                        session[
                            "session_id"
                        ]
                    )
                )
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
        session_id: int,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        include_environment: bool = True,
        generated_at: datetime | None = None,
    ) -> ResearchAnalyticsReport:
        """
        Build one complete typed research analytics report.

        A session identifier is mandatory.


        Why single-session analysis
        ---------------------------
        The analytics layer includes operations such as:

            temporal binning
            spatial transitions
            activity concentration
            behavior-related indicators

        Those operations assume one coherent acquisition period.

        Concatenating separate sessions before analytics could otherwise
        create false conclusions such as:

            last location in Session A
                ->
            first location in Session B

        being interpreted as one acoustic-location transition.


        Cross-session research
        ----------------------
        Cross-session studies should instead compute independent
        session-level reports and aggregate their outputs using explicit
        exposure/session-aware methodology.


        Wall-clock note
        ---------------
        Dashboard wall-clock timestamps are reconstructed from:

            session.started_at
                +
            sampleIndex / sample_rate

        TDOA itself remains based exclusively on synchronized waveform
        timing and is unrelated to these wall-clock timestamps.
        """

        session_id = (
            _required_session_id(
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

        if (
            generated_at
            is not None
            and not isinstance(
                generated_at,
                datetime,
            )
        ):

            raise TypeError(
                (
                    "generated_at must be "
                    "datetime or None."
                )
            )

        # ==================================================================
        # EVENTS
        # ==================================================================

        event_rows = (
            self.analytics_events(
                session_id=
                    session_id,

                start=
                    start,

                end=
                    end,
            )
        )

        # ==================================================================
        # ENVIRONMENTAL OBSERVATION BINS
        # ==================================================================

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

        else:

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

        # ==================================================================
        # ANALYTICS SERVICE
        # ==================================================================

        return (
            build_research_analytics_report(
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
        )

    # ==================================================================
    # SERIALIZED RESEARCH REPORT
    # ==================================================================

    def research_report_dict(
        self,
        *,
        session_id: int,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        include_environment: bool = True,
        generated_at: datetime | None = None,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Return one JSON/export-ready single-session research report.
        """

        report = (
            self.research_report(
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

        This deliberately avoids running the complete research analytics
        pipeline on every live refresh.
        """

        sessions = (
            self.sessions(
                limit=
                    1
            )
        )

        events = (
            self.recent_events(
                limit=
                    recent_limit,
            )
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
    Create the standard dashboard data service.

    Keeping construction in one helper also makes Streamlit resource
    caching straightforward.
    """

    return (
        DashboardDataAccess(
            config=
                config
        )
    )