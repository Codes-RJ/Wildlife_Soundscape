from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import json
import math
import sqlite3

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from pathlib import (
    Path,
)

from threading import (
    Lock,
)

from typing import (
    Any,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from classification import (
    ClassificationResult,
)

from dsp.features import (
    AcousticFeatures,
)

from event_detector import (
    AcousticEvent,
)

from localization import (
    LocalizationResult,
)

from protocol import (
    EnvironmentPayload,
)


# ======================================================================
# INTEGER LIMITS
# ======================================================================


UINT8_MAX = (
    0xFF
)


UINT32_MAX = (
    0xFFFFFFFF
)


UINT64_MAX = (
    0xFFFFFFFFFFFFFFFF
)


SQLITE_INT64_MAX = (
    0x7FFFFFFFFFFFFFFF
)


# ======================================================================
# ANALYTICS DEFAULTS
# ======================================================================


DEFAULT_ANALYTICS_SAMPLE_RATE = (
    48_000
)


DEFAULT_ANALYTICS_BUCKET_SECONDS = (
    3600
)


UTC_SUFFIX = (
    "Z"
)


# ======================================================================
# DATABASE
# ======================================================================


class EventDatabase:
    """
    SQLite persistence layer for the Wildlife Soundscape system.

    Database structure
    ------------------
    sessions
        Acquisition-session metadata.

    telemetry
        Environmental measurements received from the BME280.

    events
        Core acoustic-event metadata, environmental context and
        localization result.

    event_features
        DSP descriptors extracted from the selected best-quality
        microphone channel.

    classifications
        Broad acoustic classification associated one-to-one with an
        event.


    Storage policy
    --------------
    Raw PCM/audio is deliberately not stored as SQLite BLOB data.

        WAV/audio
            filesystem

        structured metadata
            SQLite

    This keeps the database compact and makes recordings directly
    reusable by later DSP, benchmarking and machine-learning workflows.


    Analytics policy
    ----------------
    Protocol-v4 sampleIndex is the authoritative within-session acoustic
    timeline.

    For research/dashboard queries, wall-clock event time is reconstructed
    as:

        session.started_at
            +
        sample_index / sample_rate

    The resulting timestamp is suitable for:

        temporal activity aggregation
        diurnal plots
        event ordering
        telemetry-event alignment

    Important
    ---------
    ``sessions.started_at`` is recorded by the laptop when the acquisition
    session is registered.

    It is therefore a session wall-clock anchor, not a precision hardware
    timestamp for microphone sample zero.

    The reconstructed timestamps should not be interpreted as
    microsecond-accurate absolute acoustic arrival timestamps.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = (
            Path(
                path
            )
        )

        if not (
            self.path.name
        ):

            raise ValueError(
                (
                    "Database path must contain "
                    "a database filename."
                )
            )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # SQLite connections are short-lived, but write operations within
        # this process are still serialized.
        self._lock = (
            Lock()
        )

        self._initialize()

    # ==================================================================
    # CONNECTION
    # ==================================================================

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """
        Open one configured SQLite connection.

        A fresh connection is used for every database operation.
        """

        connection = (
            sqlite3.connect(
                self.path,
                timeout=
                    5.0,
            )
        )

        # --------------------------------------------------------------
        # REFERENTIAL INTEGRITY
        # --------------------------------------------------------------

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        # --------------------------------------------------------------
        # LOCK WAIT
        # --------------------------------------------------------------

        connection.execute(
            "PRAGMA busy_timeout = 5000"
        )

        return (
            connection
        )

    # ==================================================================
    # JSON SERIALIZATION
    # ==================================================================

    @staticmethod
    def _json_dumps(
        value: Any,
        *,
        sort_keys: bool = False,
    ) -> str:
        """
        Serialize standards-compatible compact JSON.

        NaN and infinity are rejected because they are not valid JSON
        numbers according to the JSON specification.
        """

        return json.dumps(
            value,
            allow_nan=
                False,
            sort_keys=
                sort_keys,
            separators=
                (
                    ",",
                    ":",
                ),
        )

    # ==================================================================
    # FINITE FLOAT VALIDATION
    # ==================================================================

    @staticmethod
    def _finite_float(
        value: Any,
        *,
        name: str,
    ) -> float:
        """
        Convert a value to float and require a finite result.
        """

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                f"{name} must be numeric"
            ) from exc

        if not math.isfinite(
            result
        ):

            raise ValueError(
                f"{name} must be finite"
            )

        return (
            result
        )

    # ==================================================================
    # OPTIONAL FINITE FLOAT
    # ==================================================================

    @classmethod
    def _optional_finite_float(
        cls,
        value: Any | None,
        *,
        name: str,
    ) -> float | None:
        """
        Return None or one finite float.
        """

        if (
            value
            is None
        ):

            return (
                None
            )

        return cls._finite_float(
            value,
            name=
                name,
        )

    # ==================================================================
    # OPTIONAL FINITE NON-NEGATIVE FLOAT
    # ==================================================================

    @staticmethod
    def _optional_nonnegative_float(
        value: Any,
    ) -> float | None:
        """
        Convert optional input to finite float >= 0.

        Invalid values return None.

        This helper is intended for read-side analytics normalization,
        where an unavailable optional field should not invalidate an
        otherwise usable event row.
        """

        if (
            value
            is None
        ):

            return (
                None
            )

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                None
            )

        if (
            not math.isfinite(
                result
            )
            or result
            < 0.0
        ):

            return (
                None
            )

        return (
            result
        )

    # ==================================================================
    # ID VALIDATION
    # ==================================================================

    @staticmethod
    def _positive_id(
        value: int,
        *,
        name: str,
        maximum: int | None = None,
    ) -> int:
        """
        Validate a positive integer identifier.
        """

        if isinstance(
            value,
            bool,
        ):

            raise TypeError(
                f"{name} must be an integer"
            )

        if not isinstance(
            value,
            int,
        ):

            raise TypeError(
                f"{name} must be an integer"
            )

        result = int(
            value
        )

        if (
            result
            <= 0
        ):

            raise ValueError(
                f"{name} must be greater than 0"
            )

        if (
            maximum
            is not None
            and result
            > maximum
        ):

            raise ValueError(
                (
                    f"{name} cannot exceed "
                    f"{maximum}"
                )
            )

        return (
            result
        )

    # ==================================================================
    # SAMPLE INDEX VALIDATION
    # ==================================================================

    @staticmethod
    def _sample_index(
        value: int,
        *,
        name: str,
    ) -> int:
        """
        Validate one Protocol-v4 uint64 sample position.
        """

        if isinstance(
            value,
            bool,
        ):

            raise TypeError(
                f"{name} must be an integer"
            )

        if not isinstance(
            value,
            int,
        ):

            raise TypeError(
                f"{name} must be an integer"
            )

        result = int(
            value
        )

        if not (
            0
            <= result
            <= UINT64_MAX
        ):

            raise ValueError(
                (
                    f"{name} must lie in "
                    "the uint64 range"
                )
            )

        # --------------------------------------------------------------
        # SQLITE INTEGER LIMIT
        # --------------------------------------------------------------
        #
        # SQLite INTEGER is signed 64-bit.
        #
        # Protocol sampleIndex is uint64. Real acquisition sessions will
        # never approach 2^63 samples, but explicitly reject values that
        # SQLite cannot represent rather than allowing an OverflowError.
        # --------------------------------------------------------------

        if (
            result
            > SQLITE_INT64_MAX
        ):

            raise ValueError(
                (
                    f"{name} exceeds SQLite "
                    "signed 64-bit INTEGER capacity"
                )
            )

        return (
            result
        )

    # ==================================================================
    # POSITIVE SAMPLE RATE
    # ==================================================================

    @staticmethod
    def _positive_sample_rate(
        value: int,
    ) -> int:
        """
        Validate analytics sample rate.
        """

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
                "sample_rate must be an integer"
            )

        if (
            value
            <= 0
        ):

            raise ValueError(
                "sample_rate must be greater than 0"
            )

        return (
            int(
                value
            )
        )

    # ==================================================================
    # POSITIVE BUCKET SIZE
    # ==================================================================

    @staticmethod
    def _positive_bucket_seconds(
        value: int,
    ) -> int:
        """
        Validate analytics temporal-bin width.
        """

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
                "bucket_seconds must be an integer"
            )

        if (
            value
            <= 0
        ):

            raise ValueError(
                (
                    "bucket_seconds must be "
                    "greater than 0"
                )
            )

        return (
            int(
                value
            )
        )

    # ==================================================================
    # DATABASE TIMESTAMP PARSER
    # ==================================================================

    @staticmethod
    def _parse_database_datetime(
        value: str | datetime,
        *,
        name: str,
    ) -> datetime:
        """
        Convert a persisted SQLite/ISO timestamp to UTC-aware datetime.

        SQLite CURRENT_TIMESTAMP is UTC but normally stored without an
        explicit timezone suffix.

        Naive timestamps from this database are therefore interpreted as
        UTC.
        """

        if isinstance(
            value,
            datetime,
        ):

            result = (
                value
            )

        else:

            if not isinstance(
                value,
                str,
            ):

                raise TypeError(
                    (
                        f"{name} must be datetime "
                        "or timestamp string"
                    )
                )

            text = (
                value.strip()
            )

            if not (
                text
            ):

                raise ValueError(
                    f"{name} cannot be empty"
                )

            if text.endswith(
                UTC_SUFFIX
            ):

                text = (
                    text[
                        :-1
                    ]
                    + "+00:00"
                )

            try:

                result = (
                    datetime.fromisoformat(
                        text
                    )
                )

            except ValueError as exc:

                raise ValueError(
                    (
                        f"{name} is not a valid "
                        "ISO timestamp"
                    )
                ) from exc

        # --------------------------------------------------------------
        # SQLITE CURRENT_TIMESTAMP IS UTC
        # --------------------------------------------------------------

        if (
            result.tzinfo
            is None
            or result.utcoffset()
            is None
        ):

            result = result.replace(
                tzinfo=
                    timezone.utc
            )

        else:

            result = result.astimezone(
                timezone.utc
            )

        return (
            result
        )

    # ==================================================================
    # OPTIONAL QUERY DATETIME
    # ==================================================================

    @classmethod
    def _optional_query_datetime(
        cls,
        value: str | datetime | None,
        *,
        name: str,
    ) -> datetime | None:
        """
        Normalize optional analytics time boundary into UTC.
        """

        if (
            value
            is None
        ):

            return (
                None
            )

        return cls._parse_database_datetime(
            value,
            name=
                name,
        )

    # ==================================================================
    # SESSION SAMPLE -> WALL CLOCK
    # ==================================================================

    @classmethod
    def _sample_time(
        cls,
        *,
        session_started_at: str | datetime,
        sample_index: int,
        sample_rate: int,
    ) -> datetime:
        """
        Reconstruct one sampleIndex wall-clock timestamp.

        Formula
        -------

            session start
                +
            sample_index / sample_rate
        """

        sample_rate = (
            cls._positive_sample_rate(
                sample_rate
            )
        )

        sample_index = (
            cls._sample_index(
                sample_index,
                name=
                    "sample_index",
            )
        )

        session_start = (
            cls._parse_database_datetime(
                session_started_at,
                name=
                    "session_started_at",
            )
        )

        offset_seconds = (
            sample_index
            / float(
                sample_rate
            )
        )

        try:

            return (
                session_start
                + timedelta(
                    seconds=
                        offset_seconds
                )
            )

        except OverflowError as exc:

            raise ValueError(
                (
                    "sample_index produces a "
                    "datetime outside supported range"
                )
            ) from exc

    # ==================================================================
    # ISO TIMESTAMP
    # ==================================================================

    @staticmethod
    def _datetime_to_iso(
        value: datetime,
    ) -> str:
        """
        Serialize one datetime as timezone-aware ISO-8601 text.
        """

        if not isinstance(
            value,
            datetime,
        ):

            raise TypeError(
                "value must be datetime"
            )

        if (
            value.tzinfo
            is None
            or value.utcoffset()
            is None
        ):

            value = value.replace(
                tzinfo=
                    timezone.utc
            )

        else:

            value = value.astimezone(
                timezone.utc
            )

        return (
            value.isoformat()
        )

    # ==================================================================
    # DATABASE INITIALIZATION
    # ==================================================================

    def _initialize(
        self,
    ) -> None:
        """
        Create tables, perform compatible migrations and create indexes.
        """

        with self._connect() as conn:

            # ==========================================================
            # SQLITE OPERATING MODE
            # ==========================================================

            conn.execute(
                "PRAGMA journal_mode = WAL"
            )

            conn.execute(
                "PRAGMA synchronous = NORMAL"
            )

            # ==========================================================
            # CORE TABLES
            # ==========================================================

            conn.executescript(
                """
                ------------------------------------------------------------
                -- SESSIONS
                ------------------------------------------------------------

                CREATE TABLE IF NOT EXISTS sessions (

                    session_id INTEGER PRIMARY KEY,

                    label TEXT NOT NULL,

                    started_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    stopped_at TEXT
                );


                ------------------------------------------------------------
                -- ENVIRONMENTAL TELEMETRY
                ------------------------------------------------------------

                CREATE TABLE IF NOT EXISTS telemetry (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    session_id INTEGER,

                    node_id INTEGER NOT NULL,

                    sample_index INTEGER NOT NULL,

                    temperature_c REAL,

                    humidity_percent REAL,

                    pressure_hpa REAL,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                );


                ------------------------------------------------------------
                -- ACOUSTIC EVENTS
                ------------------------------------------------------------

                CREATE TABLE IF NOT EXISTS events (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    detector_event_id INTEGER NOT NULL,

                    session_id INTEGER NOT NULL,

                    start_sample INTEGER NOT NULL,

                    end_sample INTEGER NOT NULL,

                    trigger_nodes TEXT NOT NULL,

                    peak_rms_dbfs REAL NOT NULL,

                    --------------------------------------------------------
                    -- DSP SOURCE CHANNEL
                    --------------------------------------------------------

                    best_node_id INTEGER,

                    --------------------------------------------------------
                    -- ENVIRONMENT
                    --------------------------------------------------------

                    temperature_c REAL,

                    humidity_percent REAL,

                    pressure_hpa REAL,

                    --------------------------------------------------------
                    -- SPEED OF SOUND
                    --------------------------------------------------------

                    speed_of_sound_mps REAL,

                    --------------------------------------------------------
                    -- LOCALIZATION
                    --------------------------------------------------------

                    x_m REAL,

                    y_m REAL,

                    localization_success INTEGER,

                    localization_residual_m REAL,

                    --------------------------------------------------------
                    -- FILESYSTEM
                    --------------------------------------------------------

                    event_directory TEXT,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY(session_id)
                        REFERENCES sessions(session_id)
                );


                ------------------------------------------------------------
                -- DSP FEATURES
                ------------------------------------------------------------

                CREATE TABLE IF NOT EXISTS event_features (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    event_id INTEGER NOT NULL UNIQUE,

                    --------------------------------------------------------
                    -- SELECTED MICROPHONE
                    --------------------------------------------------------

                    source_node_id INTEGER NOT NULL,

                    --------------------------------------------------------
                    -- GENERAL
                    --------------------------------------------------------

                    duration_s REAL NOT NULL,

                    --------------------------------------------------------
                    -- TIME DOMAIN
                    --------------------------------------------------------

                    rms REAL NOT NULL,

                    peak_amplitude REAL NOT NULL,

                    crest_factor REAL NOT NULL,

                    zero_crossing_rate REAL NOT NULL,

                    --------------------------------------------------------
                    -- FREQUENCY DOMAIN
                    --------------------------------------------------------

                    dominant_frequency_hz REAL NOT NULL,

                    spectral_centroid_hz REAL NOT NULL,

                    spectral_bandwidth_hz REAL NOT NULL,

                    spectral_rolloff_hz REAL NOT NULL,

                    spectral_flatness REAL NOT NULL,

                    spectral_flux REAL NOT NULL,

                    --------------------------------------------------------
                    -- SIGNAL QUALITY
                    --------------------------------------------------------

                    snr_db REAL,

                    --------------------------------------------------------
                    -- MFCC
                    --------------------------------------------------------

                    mfcc_mean_json TEXT NOT NULL,

                    mfcc_std_json TEXT NOT NULL,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY(event_id)
                        REFERENCES events(id)
                        ON DELETE CASCADE
                );


                ------------------------------------------------------------
                -- CLASSIFICATIONS
                ------------------------------------------------------------

                CREATE TABLE IF NOT EXISTS classifications (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    event_id INTEGER NOT NULL UNIQUE,

                    --------------------------------------------------------
                    -- PRIMARY CLASSIFICATION
                    --------------------------------------------------------

                    label TEXT NOT NULL,

                    confidence REAL NOT NULL,

                    --------------------------------------------------------
                    -- SECONDARY CANDIDATE
                    --------------------------------------------------------

                    second_label TEXT,

                    second_confidence REAL,

                    --------------------------------------------------------
                    -- DECISION QUALITY
                    --------------------------------------------------------

                    margin REAL NOT NULL,

                    --------------------------------------------------------
                    -- EXPLAINABILITY
                    --------------------------------------------------------

                    scores_json TEXT NOT NULL,

                    reasons_json TEXT NOT NULL,

                    --------------------------------------------------------
                    -- CLASSIFIER VERSIONING
                    --------------------------------------------------------

                    classifier_name TEXT NOT NULL,

                    classifier_version TEXT NOT NULL,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY(event_id)
                        REFERENCES events(id)
                        ON DELETE CASCADE
                );
                """
            )

            # ==========================================================
            # SIMPLE EVENT MIGRATIONS
            # ==========================================================

            self._ensure_column(
                conn,
                table=
                    "events",
                column=
                    "best_node_id",
                definition=
                    "INTEGER",
            )

            # ==========================================================
            # CLASSIFICATION SCHEMA MIGRATION
            # ==========================================================

            self._ensure_nullable_second_confidence(
                conn
            )

            # ==========================================================
            # INDEXES
            # ==========================================================
            #
            # These are deliberately created after table migrations.
            #
            # Rebuilding a SQLite table drops indexes associated with the
            # old table, so this guarantees they exist in the final
            # schema.
            # ==========================================================

            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_sessions_started

                ON sessions(
                    started_at
                );


                CREATE INDEX IF NOT EXISTS
                    idx_telemetry_session_sample

                ON telemetry(
                    session_id,
                    sample_index
                );


                CREATE INDEX IF NOT EXISTS
                    idx_telemetry_node_sample

                ON telemetry(
                    node_id,
                    sample_index
                );


                CREATE INDEX IF NOT EXISTS
                    idx_events_session_start

                ON events(
                    session_id,
                    start_sample
                );


                CREATE INDEX IF NOT EXISTS
                    idx_events_created

                ON events(
                    created_at
                );


                CREATE INDEX IF NOT EXISTS
                    idx_event_features_event

                ON event_features(
                    event_id
                );


                CREATE INDEX IF NOT EXISTS
                    idx_event_features_node

                ON event_features(
                    source_node_id
                );


                CREATE INDEX IF NOT EXISTS
                    idx_classifications_event

                ON classifications(
                    event_id
                );


                CREATE INDEX IF NOT EXISTS
                    idx_classifications_label

                ON classifications(
                    label
                );
                """
            )

    # ==================================================================
    # SIMPLE COLUMN MIGRATION
    # ==================================================================

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        """
        Add one hard-coded internal column when absent.
        """

        rows = (
            conn.execute(
                f"PRAGMA table_info({table})"
            )
            .fetchall()
        )

        existing_columns = {
            str(
                row[
                    1
                ]
            )

            for row
            in rows
        }

        if (
            column
            in existing_columns
        ):

            return

        conn.execute(
            f"""
            ALTER TABLE {table}
            ADD COLUMN {column} {definition}
            """
        )

    # ==================================================================
    # CLASSIFICATION NULLABILITY MIGRATION
    # ==================================================================

    @staticmethod
    def _ensure_nullable_second_confidence(
        conn: sqlite3.Connection,
    ) -> None:
        """
        Upgrade old classifications tables where second_confidence was
        incorrectly declared NOT NULL.

        SQLite cannot directly DROP a NOT NULL constraint, therefore the
        table is rebuilt while preserving existing rows.
        """

        table_info = (
            conn.execute(
                """
                PRAGMA table_info(
                    classifications
                )
                """
            )
            .fetchall()
        )

        second_column = next(
            (
                row

                for row
                in table_info

                if str(
                    row[
                        1
                    ]
                )
                == "second_confidence"
            ),
            None,
        )

        # Fresh/current schema.
        if (
            second_column
            is None
            or int(
                second_column[
                    3
                ]
            )
            == 0
        ):

            return

        # ==============================================================
        # OLD TABLE REQUIRES REBUILD
        # ==============================================================

        conn.execute(
            """
            CREATE TABLE
                classifications_migrated (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    event_id INTEGER NOT NULL UNIQUE,

                    label TEXT NOT NULL,

                    confidence REAL NOT NULL,

                    second_label TEXT,

                    second_confidence REAL,

                    margin REAL NOT NULL,

                    scores_json TEXT NOT NULL,

                    reasons_json TEXT NOT NULL,

                    classifier_name TEXT NOT NULL,

                    classifier_version TEXT NOT NULL,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY(event_id)
                        REFERENCES events(id)
                        ON DELETE CASCADE
                )
            """
        )

        conn.execute(
            """
            INSERT INTO classifications_migrated(

                id,

                event_id,

                label,

                confidence,

                second_label,

                second_confidence,

                margin,

                scores_json,

                reasons_json,

                classifier_name,

                classifier_version,

                created_at
            )

            SELECT

                id,

                event_id,

                label,

                confidence,

                second_label,

                second_confidence,

                margin,

                scores_json,

                reasons_json,

                classifier_name,

                classifier_version,

                created_at

            FROM classifications
            """
        )

        conn.execute(
            """
            DROP TABLE classifications
            """
        )

        conn.execute(
            """
            ALTER TABLE classifications_migrated
            RENAME TO classifications
            """
        )

    # ==================================================================
    # SESSION OPERATIONS
    # ==================================================================

    def start_session(
        self,
        session_id: int,
        label: str,
    ) -> None:
        """
        Register the beginning of one acquisition session.

        A session identifier is immutable historical identity.

        If the same non-zero ID already exists, insertion fails rather
        than silently overwriting an older research session.
        """

        session_id = (
            self._positive_id(
                session_id,
                name=
                    "session_id",
                maximum=
                    UINT32_MAX,
            )
        )

        if not isinstance(
            label,
            str,
        ):

            raise TypeError(
                (
                    "session label must "
                    "be a string"
                )
            )

        label = (
            label.strip()
        )

        if not (
            label
        ):

            raise ValueError(
                (
                    "session label "
                    "cannot be empty"
                )
            )

        with (
            self._lock,
            self._connect() as conn,
        ):

            try:

                conn.execute(
                    """
                    INSERT INTO sessions(

                        session_id,

                        label,

                        started_at,

                        stopped_at
                    )

                    VALUES(
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        NULL
                    )
                    """,
                    (
                        session_id,
                        label,
                    ),
                )

            except sqlite3.IntegrityError as exc:

                raise ValueError(
                    (
                        "session_id already exists "
                        f"in database: "
                        f"0x{session_id:08X}"
                    )
                ) from exc

    # ==================================================================
    # STOP SESSION
    # ==================================================================

    def stop_session(
        self,
        session_id: int | None,
    ) -> None:
        """
        Mark one acquisition session as stopped.
        """

        if (
            session_id
            is None
        ):

            return

        try:

            session_id = (
                self._positive_id(
                    session_id,
                    name=
                        "session_id",
                    maximum=
                        UINT32_MAX,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return

        with (
            self._lock,
            self._connect() as conn,
        ):

            conn.execute(
                """
                UPDATE sessions

                SET stopped_at =
                    CURRENT_TIMESTAMP

                WHERE session_id = ?
                """,
                (
                    session_id,
                ),
            )

    # ==================================================================
    # LIST SESSIONS
    # ==================================================================

    def list_sessions(
        self,
        limit: int | None = None,
    ) -> list[
        sqlite3.Row
    ]:
        """
        Return acquisition sessions newest first.

        This read API is used by the future dashboard/session selector.

        Parameters
        ----------
        limit
            Optional maximum number of sessions.

            None:
                return every session.

            <= 0:
                return an empty list.
        """

        if (
            limit
            is not None
        ):

            if isinstance(
                limit,
                bool,
            ):

                raise TypeError(
                    "limit must be an integer or None"
                )

            if not isinstance(
                limit,
                int,
            ):

                raise TypeError(
                    "limit must be an integer or None"
                )

            if (
                limit
                <= 0
            ):

                return (
                    []
                )

        with self._connect() as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            if (
                limit
                is None
            ):

                rows = (
                    conn.execute(
                        """
                        SELECT

                            session_id,

                            label,

                            started_at,

                            stopped_at

                        FROM sessions

                        ORDER BY started_at DESC,
                                 session_id DESC
                        """
                    )
                    .fetchall()
                )

            else:

                rows = (
                    conn.execute(
                        """
                        SELECT

                            session_id,

                            label,

                            started_at,

                            stopped_at

                        FROM sessions

                        ORDER BY started_at DESC,
                                 session_id DESC

                        LIMIT ?
                        """,
                        (
                            limit,
                        ),
                    )
                    .fetchall()
                )

        return list(
            rows
        )

    # ==================================================================
    # ENVIRONMENTAL TELEMETRY
    # ==================================================================

    def add_environment(
        self,
        *,
        session_id: int,
        node_id: int,
        sample_index: int,
        environment: EnvironmentPayload,
    ) -> None:
        """
        Persist one BME280 telemetry sample.
        """

        session_id = (
            self._positive_id(
                session_id,
                name=
                    "session_id",
                maximum=
                    UINT32_MAX,
            )
        )

        node_id = (
            self._positive_id(
                node_id,
                name=
                    "node_id",
                maximum=
                    UINT8_MAX,
            )
        )

        sample_index = (
            self._sample_index(
                sample_index,
                name=
                    "sample_index",
            )
        )

        if not isinstance(
            environment,
            EnvironmentPayload,
        ):

            raise TypeError(
                (
                    "environment must be an "
                    "EnvironmentPayload instance"
                )
            )

        temperature = (
            self._finite_float(
                environment.temperature_c,
                name=
                    "temperature_c",
            )
        )

        humidity = (
            self._finite_float(
                environment.humidity_percent,
                name=
                    "humidity_percent",
            )
        )

        pressure = (
            self._finite_float(
                environment.pressure_hpa,
                name=
                    "pressure_hpa",
            )
        )

        with (
            self._lock,
            self._connect() as conn,
        ):

            conn.execute(
                """
                INSERT INTO telemetry(

                    session_id,

                    node_id,

                    sample_index,

                    temperature_c,

                    humidity_percent,

                    pressure_hpa
                )

                VALUES(
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    session_id,

                    node_id,

                    sample_index,

                    temperature,

                    humidity,

                    pressure,
                ),
            )

    # ==================================================================
    # CORE EVENT OPERATIONS
    # ==================================================================

    def add_event(
        self,
        event: AcousticEvent,
        *,
        environment: EnvironmentPayload | None,
        localization: LocalizationResult | None,
        event_directory: str | None,
        best_node_id: int | None = None,
    ) -> int:
        """
        Store core acoustic-event information.

        DSP features and classification remain in separate one-to-one
        tables.
        """

        # ==============================================================
        # EVENT TYPE
        # ==============================================================

        if not isinstance(
            event,
            AcousticEvent,
        ):

            raise TypeError(
                (
                    "event must be "
                    "an AcousticEvent instance"
                )
            )

        # ==============================================================
        # EVENT IDENTITY
        # ==============================================================

        detector_event_id = (
            self._positive_id(
                event.event_id,
                name=
                    "detector_event_id",
            )
        )

        session_id = (
            self._positive_id(
                event.session_id,
                name=
                    "event session_id",
                maximum=
                    UINT32_MAX,
            )
        )

        start_sample = (
            self._sample_index(
                event.start_sample,
                name=
                    "event start_sample",
            )
        )

        end_sample = (
            self._sample_index(
                event.end_sample,
                name=
                    "event end_sample",
            )
        )

        if (
            end_sample
            < start_sample
        ):

            raise ValueError(
                (
                    "event end_sample cannot "
                    "be smaller than start_sample"
                )
            )

        peak_rms_dbfs = (
            self._finite_float(
                event.peak_rms_dbfs,
                name=
                    "event peak_rms_dbfs",
            )
        )

        # ==============================================================
        # TRIGGER NODES
        # ==============================================================

        trigger_nodes = sorted(
            {
                self._positive_id(
                    node_id,
                    name=
                        "trigger node_id",
                    maximum=
                        UINT8_MAX,
                )

                for node_id
                in event.trigger_nodes
            }
        )

        if not (
            trigger_nodes
        ):

            raise ValueError(
                (
                    "AcousticEvent must contain "
                    "at least one trigger node"
                )
            )

        trigger_nodes_json = (
            self._json_dumps(
                trigger_nodes
            )
        )

        # ==============================================================
        # BEST NODE
        # ==============================================================

        if (
            best_node_id
            is not None
        ):

            best_node_id = (
                self._positive_id(
                    best_node_id,
                    name=
                        "best_node_id",
                    maximum=
                        UINT8_MAX,
                )
            )

        # ==============================================================
        # ENVIRONMENT
        # ==============================================================

        if (
            environment
            is None
        ):

            temperature = (
                None
            )

            humidity = (
                None
            )

            pressure = (
                None
            )

        else:

            if not isinstance(
                environment,
                EnvironmentPayload,
            ):

                raise TypeError(
                    (
                        "environment must be "
                        "EnvironmentPayload or None"
                    )
                )

            temperature = (
                self._finite_float(
                    environment.temperature_c,
                    name=
                        "event temperature_c",
                )
            )

            humidity = (
                self._finite_float(
                    environment.humidity_percent,
                    name=
                        "event humidity_percent",
                )
            )

            pressure = (
                self._finite_float(
                    environment.pressure_hpa,
                    name=
                        "event pressure_hpa",
                )
            )

        # ==============================================================
        # LOCALIZATION
        # ==============================================================

        if (
            localization
            is None
        ):

            speed_of_sound = (
                None
            )

            x_m = (
                None
            )

            y_m = (
                None
            )

            localization_success = (
                None
            )

            localization_residual = (
                None
            )

        else:

            if not isinstance(
                localization,
                LocalizationResult,
            ):

                raise TypeError(
                    (
                        "localization must be "
                        "LocalizationResult or None"
                    )
                )

            speed_of_sound = (
                self._finite_float(
                    localization.speed_of_sound_mps,
                    name=
                        "speed_of_sound_mps",
                )
            )

            position = (
                localization.position
            )

            x_m = (
                self._finite_float(
                    position.x,
                    name=
                        "localization x_m",
                )
            )

            y_m = (
                self._finite_float(
                    position.y,
                    name=
                        "localization y_m",
                )
            )

            localization_success = int(
                bool(
                    position.success
                )
            )

            localization_residual = (
                self._finite_float(
                    position.residual_rms_meters,
                    name=
                        "localization_residual_m",
                )
            )

        # ==============================================================
        # EVENT DIRECTORY
        # ==============================================================

        if (
            event_directory
            is not None
        ):

            event_directory = str(
                event_directory
            )

        # ==============================================================
        # INSERT
        # ==============================================================

        with (
            self._lock,
            self._connect() as conn,
        ):

            cursor = conn.execute(
                """
                INSERT INTO events(

                    detector_event_id,

                    session_id,

                    start_sample,

                    end_sample,

                    trigger_nodes,

                    peak_rms_dbfs,

                    best_node_id,

                    temperature_c,

                    humidity_percent,

                    pressure_hpa,

                    speed_of_sound_mps,

                    x_m,

                    y_m,

                    localization_success,

                    localization_residual_m,

                    event_directory
                )

                VALUES(
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    detector_event_id,

                    session_id,

                    start_sample,

                    end_sample,

                    trigger_nodes_json,

                    peak_rms_dbfs,

                    best_node_id,

                    temperature,

                    humidity,

                    pressure,

                    speed_of_sound,

                    x_m,

                    y_m,

                    localization_success,

                    localization_residual,

                    event_directory,
                ),
            )

            database_event_id = (
                cursor.lastrowid
            )

            if (
                database_event_id
                is None
            ):

                raise RuntimeError(
                    (
                        "SQLite did not return "
                        "an event row ID"
                    )
                )

            return int(
                database_event_id
            )

    # ==================================================================
    # DSP FEATURE OPERATIONS
    # ==================================================================

    def add_event_features(
        self,
        *,
        event_id: int,
        source_node_id: int,
        features: AcousticFeatures,
    ) -> None:
        """
        Store or update the DSP feature vector associated with an event.

        event_id refers to events.id, not detector_event_id.
        """

        event_id = (
            self._positive_id(
                event_id,
                name=
                    "event_id",
            )
        )

        source_node_id = (
            self._positive_id(
                source_node_id,
                name=
                    "source_node_id",
                maximum=
                    UINT8_MAX,
            )
        )

        if not isinstance(
            features,
            AcousticFeatures,
        ):

            raise TypeError(
                (
                    "features must be an "
                    "AcousticFeatures instance"
                )
            )

        # ==============================================================
        # MFCC
        # ==============================================================

        mfcc_mean = [
            self._finite_float(
                value,
                name=
                    "MFCC mean value",
            )

            for value
            in features.mfcc_mean
        ]

        mfcc_std = [
            self._finite_float(
                value,
                name=
                    "MFCC std value",
            )

            for value
            in features.mfcc_std
        ]

        mfcc_mean_json = (
            self._json_dumps(
                mfcc_mean
            )
        )

        mfcc_std_json = (
            self._json_dumps(
                mfcc_std
            )
        )

        # ==============================================================
        # SCALAR FEATURES
        # ==============================================================

        duration_s = (
            self._finite_float(
                features.duration_s,
                name=
                    "features.duration_s",
            )
        )

        rms = (
            self._finite_float(
                features.rms,
                name=
                    "features.rms",
            )
        )

        peak_amplitude = (
            self._finite_float(
                features.peak_amplitude,
                name=
                    "features.peak_amplitude",
            )
        )

        crest_factor = (
            self._finite_float(
                features.crest_factor,
                name=
                    "features.crest_factor",
            )
        )

        zero_crossing_rate = (
            self._finite_float(
                features.zero_crossing_rate,
                name=
                    "features.zero_crossing_rate",
            )
        )

        dominant_frequency_hz = (
            self._finite_float(
                features.dominant_frequency_hz,
                name=
                    "features.dominant_frequency_hz",
            )
        )

        spectral_centroid_hz = (
            self._finite_float(
                features.spectral_centroid_hz,
                name=
                    "features.spectral_centroid_hz",
            )
        )

        spectral_bandwidth_hz = (
            self._finite_float(
                features.spectral_bandwidth_hz,
                name=
                    "features.spectral_bandwidth_hz",
            )
        )

        spectral_rolloff_hz = (
            self._finite_float(
                features.spectral_rolloff_hz,
                name=
                    "features.spectral_rolloff_hz",
            )
        )

        spectral_flatness = (
            self._finite_float(
                features.spectral_flatness,
                name=
                    "features.spectral_flatness",
            )
        )

        spectral_flux = (
            self._finite_float(
                features.spectral_flux,
                name=
                    "features.spectral_flux",
            )
        )

        snr_db = (
            self._optional_finite_float(
                features.snr_db,
                name=
                    "features.snr_db",
            )
        )

        # ==============================================================
        # UPSERT
        # ==============================================================

        with (
            self._lock,
            self._connect() as conn,
        ):

            conn.execute(
                """
                INSERT INTO event_features(

                    event_id,

                    source_node_id,

                    duration_s,

                    rms,

                    peak_amplitude,

                    crest_factor,

                    zero_crossing_rate,

                    dominant_frequency_hz,

                    spectral_centroid_hz,

                    spectral_bandwidth_hz,

                    spectral_rolloff_hz,

                    spectral_flatness,

                    spectral_flux,

                    snr_db,

                    mfcc_mean_json,

                    mfcc_std_json
                )

                VALUES(
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )

                ON CONFLICT(event_id)
                DO UPDATE SET

                    source_node_id =
                        excluded.source_node_id,

                    duration_s =
                        excluded.duration_s,

                    rms =
                        excluded.rms,

                    peak_amplitude =
                        excluded.peak_amplitude,

                    crest_factor =
                        excluded.crest_factor,

                    zero_crossing_rate =
                        excluded.zero_crossing_rate,

                    dominant_frequency_hz =
                        excluded.dominant_frequency_hz,

                    spectral_centroid_hz =
                        excluded.spectral_centroid_hz,

                    spectral_bandwidth_hz =
                        excluded.spectral_bandwidth_hz,

                    spectral_rolloff_hz =
                        excluded.spectral_rolloff_hz,

                    spectral_flatness =
                        excluded.spectral_flatness,

                    spectral_flux =
                        excluded.spectral_flux,

                    snr_db =
                        excluded.snr_db,

                    mfcc_mean_json =
                        excluded.mfcc_mean_json,

                    mfcc_std_json =
                        excluded.mfcc_std_json
                """,
                (
                    event_id,

                    source_node_id,

                    duration_s,

                    rms,

                    peak_amplitude,

                    crest_factor,

                    zero_crossing_rate,

                    dominant_frequency_hz,

                    spectral_centroid_hz,

                    spectral_bandwidth_hz,

                    spectral_rolloff_hz,

                    spectral_flatness,

                    spectral_flux,

                    snr_db,

                    mfcc_mean_json,

                    mfcc_std_json,
                ),
            )

    # ==================================================================
    # CLASSIFICATION SCORE KEY
    # ==================================================================

    @staticmethod
    def _classification_score_key(
        key: Any,
    ) -> str:
        """
        Convert one classification-score key into stable stored text.

        Supporting `.value` also allows future enum-keyed score maps.
        """

        value = getattr(
            key,
            "value",
            key,
        )

        return str(
            value
        )

    # ==================================================================
    # CLASSIFICATION OPERATIONS
    # ==================================================================

    def add_classification(
        self,
        *,
        event_id: int,
        result: ClassificationResult,
    ) -> None:
        """
        Store or update broad acoustic classification for one event.
        """

        event_id = (
            self._positive_id(
                event_id,
                name=
                    "event_id",
            )
        )

        if not isinstance(
            result,
            ClassificationResult,
        ):

            raise TypeError(
                (
                    "result must be a "
                    "ClassificationResult instance"
                )
            )

        # ==============================================================
        # PRIMARY RESULT
        # ==============================================================

        label = str(
            result.label.value
        )

        confidence = (
            self._finite_float(
                result.confidence,
                name=
                    "classification confidence",
            )
        )

        margin = (
            self._finite_float(
                result.margin,
                name=
                    "classification margin",
            )
        )

        # ==============================================================
        # SECOND CANDIDATE
        # ==============================================================

        if (
            result.second_label
            is None
        ):

            second_label = (
                None
            )

            second_confidence = (
                None
            )

        else:

            second_label = str(
                result.second_label.value
            )

            second_confidence = (
                self._optional_finite_float(
                    result.second_confidence,
                    name=
                        (
                            "classification "
                            "second_confidence"
                        ),
                )
            )

        # ==============================================================
        # SCORES
        # ==============================================================

        scores = {
            self._classification_score_key(
                score_label
            ):
                self._finite_float(
                    score,
                    name=
                        (
                            "classification score "
                            f"{score_label!r}"
                        ),
                )

            for (
                score_label,
                score,
            )
            in result.scores.items()
        }

        scores_json = (
            self._json_dumps(
                scores,
                sort_keys=
                    True,
            )
        )

        # ==============================================================
        # REASONS
        # ==============================================================

        reasons_json = (
            self._json_dumps(
                [
                    str(
                        reason
                    )

                    for reason
                    in result.reasons
                ]
            )
        )

        # ==============================================================
        # CLASSIFIER IDENTITY
        # ==============================================================

        classifier_name = str(
            result.classifier_name
        )

        classifier_version = str(
            result.classifier_version
        )

        if not (
            classifier_name.strip()
        ):

            raise ValueError(
                (
                    "classifier_name "
                    "cannot be empty"
                )
            )

        if not (
            classifier_version.strip()
        ):

            raise ValueError(
                (
                    "classifier_version "
                    "cannot be empty"
                )
            )

        # ==============================================================
        # UPSERT
        # ==============================================================

        with (
            self._lock,
            self._connect() as conn,
        ):

            conn.execute(
                """
                INSERT INTO classifications(

                    event_id,

                    label,

                    confidence,

                    second_label,

                    second_confidence,

                    margin,

                    scores_json,

                    reasons_json,

                    classifier_name,

                    classifier_version
                )

                VALUES(
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )

                ON CONFLICT(event_id)
                DO UPDATE SET

                    label =
                        excluded.label,

                    confidence =
                        excluded.confidence,

                    second_label =
                        excluded.second_label,

                    second_confidence =
                        excluded.second_confidence,

                    margin =
                        excluded.margin,

                    scores_json =
                        excluded.scores_json,

                    reasons_json =
                        excluded.reasons_json,

                    classifier_name =
                        excluded.classifier_name,

                    classifier_version =
                        excluded.classifier_version
                """,
                (
                    event_id,

                    label,

                    confidence,

                    second_label,

                    second_confidence,

                    margin,

                    scores_json,

                    reasons_json,

                    classifier_name,

                    classifier_version,
                ),
            )

    # ==================================================================
    # SHARED EVENT QUERY
    # ==================================================================

    @staticmethod
    def _event_select_query() -> str:
        """
        Base query joining event, DSP and classification information.

        recent_events() and get_event() therefore expose the same field
        names to the CLI and future dashboard.
        """

        return """
            SELECT

                --------------------------------------------------------
                -- EVENT
                --------------------------------------------------------

                e.*,

                --------------------------------------------------------
                -- DSP FEATURES
                --------------------------------------------------------

                f.source_node_id
                    AS feature_source_node_id,

                f.duration_s,

                f.rms,

                f.peak_amplitude,

                f.crest_factor,

                f.zero_crossing_rate,

                f.dominant_frequency_hz,

                f.spectral_centroid_hz,

                f.spectral_bandwidth_hz,

                f.spectral_rolloff_hz,

                f.spectral_flatness,

                f.spectral_flux,

                f.snr_db,

                f.mfcc_mean_json,

                f.mfcc_std_json,

                --------------------------------------------------------
                -- CLASSIFICATION
                --------------------------------------------------------

                c.label
                    AS classification_label,

                c.confidence
                    AS classification_confidence,

                c.second_label
                    AS classification_second_label,

                c.second_confidence
                    AS classification_second_confidence,

                c.margin
                    AS classification_margin,

                c.scores_json
                    AS classification_scores_json,

                c.reasons_json
                    AS classification_reasons_json,

                c.classifier_name
                    AS classifier_name,

                c.classifier_version
                    AS classifier_version

            FROM events AS e

            LEFT JOIN event_features AS f

                ON f.event_id = e.id

            LEFT JOIN classifications AS c

                ON c.event_id = e.id
        """

    # ==================================================================
    # ANALYTICS EVENT QUERY
    # ==================================================================

    @staticmethod
    def _analytics_event_select_query() -> str:
        """
        Event query extended with acquisition-session metadata.

        The actual reconstructed ``event_time`` is calculated in Python
        so fractional-sample timing is not truncated by SQLite datetime
        formatting.
        """

        return """
            SELECT

                --------------------------------------------------------
                -- EVENT
                --------------------------------------------------------

                e.*,

                --------------------------------------------------------
                -- SESSION
                --------------------------------------------------------

                s.label
                    AS session_label,

                s.started_at
                    AS session_started_at,

                s.stopped_at
                    AS session_stopped_at,

                --------------------------------------------------------
                -- DSP FEATURES
                --------------------------------------------------------

                f.source_node_id
                    AS feature_source_node_id,

                f.duration_s,

                f.rms,

                f.peak_amplitude,

                f.crest_factor,

                f.zero_crossing_rate,

                f.dominant_frequency_hz,

                f.spectral_centroid_hz,

                f.spectral_bandwidth_hz,

                f.spectral_rolloff_hz,

                f.spectral_flatness,

                f.spectral_flux,

                f.snr_db,

                f.mfcc_mean_json,

                f.mfcc_std_json,

                --------------------------------------------------------
                -- CLASSIFICATION
                --------------------------------------------------------

                c.label
                    AS classification_label,

                c.confidence
                    AS classification_confidence,

                c.second_label
                    AS classification_second_label,

                c.second_confidence
                    AS classification_second_confidence,

                c.margin
                    AS classification_margin,

                c.scores_json
                    AS classification_scores_json,

                c.reasons_json
                    AS classification_reasons_json,

                c.classifier_name
                    AS classifier_name,

                c.classifier_version
                    AS classifier_version

            FROM events AS e

            INNER JOIN sessions AS s

                ON s.session_id = e.session_id

            LEFT JOIN event_features AS f

                ON f.event_id = e.id

            LEFT JOIN classifications AS c

                ON c.event_id = e.id
        """

    # ==================================================================
    # RECENT EVENTS
    # ==================================================================

    def recent_events(
        self,
        limit: int = 10,
    ) -> list[
        sqlite3.Row
    ]:
        """
        Return newest events together with DSP and classification data.
        """

        if isinstance(
            limit,
            bool,
        ):

            raise TypeError(
                "limit must be an integer"
            )

        limit = int(
            limit
        )

        if (
            limit
            <= 0
        ):

            return (
                []
            )

        with self._connect() as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            query = (
                self._event_select_query()
                + """

                ORDER BY e.id DESC

                LIMIT ?
                """
            )

            rows = (
                conn.execute(
                    query,
                    (
                        limit,
                    ),
                )
                .fetchall()
            )

            return list(
                rows
            )

    # ==================================================================
    # ONE EVENT
    # ==================================================================

    def get_event(
        self,
        event_id: int,
    ) -> sqlite3.Row | None:
        """
        Return one event together with DSP and classification data.
        """

        if isinstance(
            event_id,
            bool,
        ):

            return (
                None
            )

        try:

            event_id = int(
                event_id
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                None
            )

        if (
            event_id
            <= 0
        ):

            return (
                None
            )

        with self._connect() as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            query = (
                self._event_select_query()
                + """

                WHERE e.id = ?

                LIMIT 1
                """
            )

            return (
                conn.execute(
                    query,
                    (
                        event_id,
                    ),
                )
                .fetchone()
            )

    # ==================================================================
    # ANALYTICS EVENT ROWS
    # ==================================================================

    def analytics_event_rows(
        self,
        *,
        sample_rate: int = DEFAULT_ANALYTICS_SAMPLE_RATE,
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
        Return normalized event rows for the research analytics package.

        The key addition is:

            event_time

        derived from:

            session.started_at
                +
            event.start_sample / sample_rate


        Parameters
        ----------
        sample_rate
            Audio sampling frequency used by the acquisition session.

        session_id
            Optional acquisition-session filter.

        start
            Optional inclusive UTC time boundary.

        end
            Optional exclusive UTC time boundary.


        Returns
        -------
        list[dict]
            Chronologically ordered analytics rows.

            Existing event/DSP/classification fields are preserved and
            ``event_time`` is added.
        """

        sample_rate = (
            self._positive_sample_rate(
                sample_rate
            )
        )

        if (
            session_id
            is not None
        ):

            session_id = (
                self._positive_id(
                    session_id,
                    name=
                        "session_id",
                    maximum=
                        UINT32_MAX,
                )
            )

        start_dt = (
            self._optional_query_datetime(
                start,
                name=
                    "start",
            )
        )

        end_dt = (
            self._optional_query_datetime(
                end,
                name=
                    "end",
            )
        )

        if (
            start_dt
            is not None
            and end_dt
            is not None
            and end_dt
            < start_dt
        ):

            raise ValueError(
                "end cannot be earlier than start"
            )

        with self._connect() as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            query = (
                self._analytics_event_select_query()
            )

            parameters: list[
                Any
            ] = []

            if (
                session_id
                is not None
            ):

                query += """

                    WHERE e.session_id = ?
                """

                parameters.append(
                    session_id
                )

            query += """

                ORDER BY

                    s.started_at ASC,

                    e.start_sample ASC,

                    e.id ASC
            """

            raw_rows = (
                conn.execute(
                    query,
                    tuple(
                        parameters
                    ),
                )
                .fetchall()
            )

        result: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for raw_row in (
            raw_rows
        ):

            row = dict(
                raw_row
            )

            event_time = (
                self._sample_time(
                    session_started_at=
                        row[
                            "session_started_at"
                        ],

                    sample_index=
                        int(
                            row[
                                "start_sample"
                            ]
                        ),

                    sample_rate=
                        sample_rate,
                )
            )

            # ----------------------------------------------------------
            # HALF-OPEN QUERY INTERVAL
            #
            #     [start, end)
            # ----------------------------------------------------------

            if (
                start_dt
                is not None
                and event_time
                < start_dt
            ):

                continue

            if (
                end_dt
                is not None
                and event_time
                >= end_dt
            ):

                continue

            row[
                "event_time"
            ] = (
                self._datetime_to_iso(
                    event_time
                )
            )

            # ----------------------------------------------------------
            # NORMALIZED EVENT END TIME
            # ----------------------------------------------------------

            event_end_time = (
                self._sample_time(
                    session_started_at=
                        row[
                            "session_started_at"
                        ],

                    sample_index=
                        int(
                            row[
                                "end_sample"
                            ]
                        ),

                    sample_rate=
                        sample_rate,
                )
            )

            row[
                "event_end_time"
            ] = (
                self._datetime_to_iso(
                    event_end_time
                )
            )

            result.append(
                row
            )

        return (
            result
        )

    # ==================================================================
    # ANALYTICS TELEMETRY ROWS
    # ==================================================================

    def analytics_telemetry_rows(
        self,
        *,
        sample_rate: int = DEFAULT_ANALYTICS_SAMPLE_RATE,
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
        Return BME280 telemetry on the reconstructed session timeline.

        The returned additional key is:

            telemetry_time

        derived from:

            session.started_at
                +
            telemetry.sample_index / sample_rate
        """

        sample_rate = (
            self._positive_sample_rate(
                sample_rate
            )
        )

        if (
            session_id
            is not None
        ):

            session_id = (
                self._positive_id(
                    session_id,
                    name=
                        "session_id",
                    maximum=
                        UINT32_MAX,
                )
            )

        if (
            node_id
            is not None
        ):

            node_id = (
                self._positive_id(
                    node_id,
                    name=
                        "node_id",
                    maximum=
                        UINT8_MAX,
                )
            )

        start_dt = (
            self._optional_query_datetime(
                start,
                name=
                    "start",
            )
        )

        end_dt = (
            self._optional_query_datetime(
                end,
                name=
                    "end",
            )
        )

        if (
            start_dt
            is not None
            and end_dt
            is not None
            and end_dt
            < start_dt
        ):

            raise ValueError(
                "end cannot be earlier than start"
            )

        where_parts: list[
            str
        ] = []

        parameters: list[
            Any
        ] = []

        if (
            session_id
            is not None
        ):

            where_parts.append(
                "t.session_id = ?"
            )

            parameters.append(
                session_id
            )

        if (
            node_id
            is not None
        ):

            where_parts.append(
                "t.node_id = ?"
            )

            parameters.append(
                node_id
            )

        query = """
            SELECT

                t.id,

                t.session_id,

                t.node_id,

                t.sample_index,

                t.temperature_c,

                t.humidity_percent,

                t.pressure_hpa,

                t.created_at,

                s.label
                    AS session_label,

                s.started_at
                    AS session_started_at,

                s.stopped_at
                    AS session_stopped_at

            FROM telemetry AS t

            INNER JOIN sessions AS s

                ON s.session_id = t.session_id
        """

        if (
            where_parts
        ):

            query += (
                """

                WHERE
                """
                + " AND ".join(
                    where_parts
                )
            )

        query += """

            ORDER BY

                s.started_at ASC,

                t.sample_index ASC,

                t.id ASC
        """

        with self._connect() as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            raw_rows = (
                conn.execute(
                    query,
                    tuple(
                        parameters
                    ),
                )
                .fetchall()
            )

        result: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for raw_row in (
            raw_rows
        ):

            row = dict(
                raw_row
            )

            telemetry_time = (
                self._sample_time(
                    session_started_at=
                        row[
                            "session_started_at"
                        ],

                    sample_index=
                        int(
                            row[
                                "sample_index"
                            ]
                        ),

                    sample_rate=
                        sample_rate,
                )
            )

            if (
                start_dt
                is not None
                and telemetry_time
                < start_dt
            ):

                continue

            if (
                end_dt
                is not None
                and telemetry_time
                >= end_dt
            ):

                continue

            row[
                "telemetry_time"
            ] = (
                self._datetime_to_iso(
                    telemetry_time
                )
            )

            result.append(
                row
            )

        return (
            result
        )

    # ==================================================================
    # SESSION METADATA
    # ==================================================================

    def _session_metadata(
        self,
        session_id: int,
    ) -> dict[
        str,
        Any,
    ] | None:
        """
        Return one session as a plain dictionary.
        """

        session_id = (
            self._positive_id(
                session_id,
                name=
                    "session_id",
                maximum=
                    UINT32_MAX,
            )
        )

        with self._connect() as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            row = (
                conn.execute(
                    """
                    SELECT

                        session_id,

                        label,

                        started_at,

                        stopped_at

                    FROM sessions

                    WHERE session_id = ?

                    LIMIT 1
                    """,
                    (
                        session_id,
                    ),
                )
                .fetchone()
            )

        if (
            row
            is None
        ):

            return (
                None
            )

        return dict(
            row
        )

    # ==================================================================
    # EVENT DURATION FOR ANALYTICS
    # ==================================================================

    @classmethod
    def _analytics_event_duration_s(
        cls,
        row: dict[
            str,
            Any,
        ],
        *,
        sample_rate: int,
    ) -> float:
        """
        Resolve event duration for temporal aggregation.

        Priority
        --------
        1. DSP feature duration_s.
        2. start/end sample-index difference.
        3. 0.0.
        """

        feature_duration = (
            cls._optional_nonnegative_float(
                row.get(
                    "duration_s"
                )
            )
        )

        if (
            feature_duration
            is not None
        ):

            return (
                feature_duration
            )

        try:

            start_sample = int(
                row[
                    "start_sample"
                ]
            )

            end_sample = int(
                row[
                    "end_sample"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            return (
                0.0
            )

        if (
            start_sample
            < 0
            or end_sample
            < start_sample
        ):

            return (
                0.0
            )

        return (
            (
                end_sample
                - start_sample
            )
            / float(
                sample_rate
            )
        )

    # ==================================================================
    # ENVIRONMENTAL OBSERVATION BINS
    # ==================================================================

    def analytics_environmental_bins(
        self,
        *,
        session_id: int,
        bucket_seconds: int = DEFAULT_ANALYTICS_BUCKET_SECONDS,
        sample_rate: int = DEFAULT_ANALYTICS_SAMPLE_RATE,
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
        Build regular within-session environmental/activity observations.

        This is the database-side input expected by:

            analytics.environmental


        Why session_id is mandatory
        ---------------------------
        Zero-event bins are scientifically useful only while acquisition
        is actually active.

        Building one continuous timeline across several sessions would
        incorrectly turn downtime between sessions into apparent
        zero-activity observation periods.

        Therefore environmental bins are intentionally generated one
        acquisition session at a time.


        Event count
        -----------
        An event is counted in the bin containing its ONSET.


        Active duration
        ---------------
        Event duration is split across every temporal bin it overlaps.


        Environmental variables
        -----------------------
        Temperature, humidity and pressure are arithmetic means of valid
        BME280 telemetry samples inside each bin.


        Zero-activity periods
        ---------------------
        Bins remain present even when:

            event_count == 0

        which is essential for unbiased activity-environment
        association analysis.


        Returns
        -------
        list[dict]

        Example:

        {
            "session_id": 123,
            "bucket_start": "...",
            "bucket_end": "...",
            "bucket_duration_s": 900.0,

            "telemetry_sample_count": 30,

            "temperature_c": 26.4,
            "humidity_percent": 72.0,
            "pressure_hpa": 1007.8,

            "event_count": 4,
            "active_duration_s": 8.2,
            "event_rate_per_hour": 16.0,
        }
        """

        session_id = (
            self._positive_id(
                session_id,
                name=
                    "session_id",
                maximum=
                    UINT32_MAX,
            )
        )

        bucket_seconds = (
            self._positive_bucket_seconds(
                bucket_seconds
            )
        )

        sample_rate = (
            self._positive_sample_rate(
                sample_rate
            )
        )

        if (
            node_id
            is not None
        ):

            node_id = (
                self._positive_id(
                    node_id,
                    name=
                        "node_id",
                    maximum=
                        UINT8_MAX,
                )
            )

        # ==============================================================
        # SESSION
        # ==============================================================

        session = (
            self._session_metadata(
                session_id
            )
        )

        if (
            session
            is None
        ):

            raise ValueError(
                (
                    "Unknown session_id: "
                    f"0x{session_id:08X}"
                )
            )

        session_start = (
            self._parse_database_datetime(
                session[
                    "started_at"
                ],
                name=
                    "session.started_at",
            )
        )

        session_stop = (
            None
        )

        if (
            session[
                "stopped_at"
            ]
            is not None
        ):

            session_stop = (
                self._parse_database_datetime(
                    session[
                        "stopped_at"
                    ],
                    name=
                        "session.stopped_at",
                )
            )

        # ==============================================================
        # USER REQUESTED RANGE
        # ==============================================================

        requested_start = (
            self._optional_query_datetime(
                start,
                name=
                    "start",
            )
        )

        requested_end = (
            self._optional_query_datetime(
                end,
                name=
                    "end",
            )
        )

        if (
            requested_start
            is not None
            and requested_end
            is not None
            and requested_end
            <= requested_start
        ):

            raise ValueError(
                "end must be later than start"
            )

        # ==============================================================
        # LOAD COMPLETE SESSION DATA
        # ==============================================================
        #
        # Do not apply start/end yet.
        #
        # An event beginning just before the requested window may still
        # overlap the first bin and contribute active duration.
        # ==============================================================

        event_rows = (
            self.analytics_event_rows(
                sample_rate=
                    sample_rate,
                session_id=
                    session_id,
            )
        )

        telemetry_rows = (
            self.analytics_telemetry_rows(
                sample_rate=
                    sample_rate,
                session_id=
                    session_id,
                node_id=
                    node_id,
            )
        )

        # ==============================================================
        # RESOLVE OBSERVATION START
        # ==============================================================

        analysis_start = (
            session_start
        )

        if (
            requested_start
            is not None
        ):

            analysis_start = max(
                analysis_start,
                requested_start,
            )

        # ==============================================================
        # RESOLVE OBSERVATION END
        # ==============================================================

        if (
            session_stop
            is not None
        ):

            natural_end = (
                session_stop
            )

        else:

            # ----------------------------------------------------------
            # ACTIVE / UNSTOPPED SESSION
            #
            # Use latest known sample-derived observation rather than
            # inventing an unobserved future interval.
            # ----------------------------------------------------------

            candidates: list[
                datetime
            ] = []

            for row in (
                telemetry_rows
            ):

                candidates.append(
                    self._parse_database_datetime(
                        row[
                            "telemetry_time"
                        ],
                        name=
                            "telemetry_time",
                    )
                )

            for row in (
                event_rows
            ):

                event_end = (
                    self._parse_database_datetime(
                        row[
                            "event_end_time"
                        ],
                        name=
                            "event_end_time",
                    )
                )

                candidates.append(
                    event_end
                )

            if not (
                candidates
            ):

                return (
                    []
                )

            natural_end = max(
                candidates
            )

        analysis_end = (
            natural_end
        )

        if (
            requested_end
            is not None
        ):

            analysis_end = min(
                analysis_end,
                requested_end,
            )

        if (
            analysis_end
            <= analysis_start
        ):

            return (
                []
            )

        # ==============================================================
        # BIN COUNT
        # ==============================================================

        observation_duration_s = (
            analysis_end
            - analysis_start
        ).total_seconds()

        bin_count = int(
            math.ceil(
                observation_duration_s
                / bucket_seconds
            )
        )

        if (
            bin_count
            <= 0
        ):

            return (
                []
            )

        # ==============================================================
        # ACCUMULATORS
        # ==============================================================

        event_counts = [
            0

            for _ in range(
                bin_count
            )
        ]

        active_durations = [
            0.0

            for _ in range(
                bin_count
            )
        ]

        temperatures: list[
            list[
                float
            ]
        ] = [
            []

            for _ in range(
                bin_count
            )
        ]

        humidities: list[
            list[
                float
            ]
        ] = [
            []

            for _ in range(
                bin_count
            )
        ]

        pressures: list[
            list[
                float
            ]
        ] = [
            []

            for _ in range(
                bin_count
            )
        ]

        telemetry_counts = [
            0

            for _ in range(
                bin_count
            )
        ]

        # ==============================================================
        # TELEMETRY -> BINS
        # ==============================================================

        for row in (
            telemetry_rows
        ):

            timestamp = (
                self._parse_database_datetime(
                    row[
                        "telemetry_time"
                    ],
                    name=
                        "telemetry_time",
                )
            )

            if not (
                analysis_start
                <= timestamp
                < analysis_end
            ):

                continue

            elapsed_s = (
                timestamp
                - analysis_start
            ).total_seconds()

            bin_index = int(
                elapsed_s
                // bucket_seconds
            )

            if not (
                0
                <= bin_index
                < bin_count
            ):

                continue

            temperature = (
                self._optional_finite_float(
                    row.get(
                        "temperature_c"
                    ),
                    name=
                        "temperature_c",
                )
            )

            humidity = (
                self._optional_finite_float(
                    row.get(
                        "humidity_percent"
                    ),
                    name=
                        "humidity_percent",
                )
            )

            pressure = (
                self._optional_finite_float(
                    row.get(
                        "pressure_hpa"
                    ),
                    name=
                        "pressure_hpa",
                )
            )

            telemetry_counts[
                bin_index
            ] += (
                1
            )

            if (
                temperature
                is not None
            ):

                temperatures[
                    bin_index
                ].append(
                    temperature
                )

            if (
                humidity
                is not None
            ):

                humidities[
                    bin_index
                ].append(
                    humidity
                )

            if (
                pressure
                is not None
            ):

                pressures[
                    bin_index
                ].append(
                    pressure
                )

        # ==============================================================
        # EVENTS -> BINS
        # ==============================================================

        for row in (
            event_rows
        ):

            event_start = (
                self._parse_database_datetime(
                    row[
                        "event_time"
                    ],
                    name=
                        "event_time",
                )
            )

            duration_s = (
                self._analytics_event_duration_s(
                    row,
                    sample_rate=
                        sample_rate,
                )
            )

            event_end = (
                event_start
                + timedelta(
                    seconds=
                        duration_s
                )
            )

            # ----------------------------------------------------------
            # EVENT-ONSET COUNT
            # ----------------------------------------------------------

            if (
                analysis_start
                <= event_start
                < analysis_end
            ):

                elapsed_s = (
                    event_start
                    - analysis_start
                ).total_seconds()

                onset_bin = int(
                    elapsed_s
                    // bucket_seconds
                )

                if (
                    0
                    <= onset_bin
                    < bin_count
                ):

                    event_counts[
                        onset_bin
                    ] += (
                        1
                    )

            # ----------------------------------------------------------
            # ZERO-DURATION EVENT
            # ----------------------------------------------------------

            if (
                duration_s
                <= 0.0
            ):

                continue

            # ----------------------------------------------------------
            # EVENT DOES NOT OVERLAP ANALYSIS WINDOW
            # ----------------------------------------------------------

            if (
                event_end
                <= analysis_start
                or event_start
                >= analysis_end
            ):

                continue

            overlap_start = max(
                event_start,
                analysis_start,
            )

            overlap_end = min(
                event_end,
                analysis_end,
            )

            if (
                overlap_end
                <= overlap_start
            ):

                continue

            # ----------------------------------------------------------
            # FIRST / LAST OVERLAPPED BIN
            # ----------------------------------------------------------

            first_bin = int(
                (
                    overlap_start
                    - analysis_start
                ).total_seconds()
                // bucket_seconds
            )

            last_position_s = (
                (
                    overlap_end
                    - analysis_start
                ).total_seconds()
            )

            # ----------------------------------------------------------
            # Subtract a tiny numerical epsilon so an event ending
            # exactly on a bin boundary does not spill into the next bin.
            # ----------------------------------------------------------

            last_bin = int(
                max(
                    0.0,
                    last_position_s
                    - 1e-12,
                )
                // bucket_seconds
            )

            first_bin = max(
                0,
                min(
                    first_bin,
                    bin_count
                    - 1,
                ),
            )

            last_bin = max(
                0,
                min(
                    last_bin,
                    bin_count
                    - 1,
                ),
            )

            for bin_index in range(
                first_bin,
                last_bin
                + 1,
            ):

                bin_start = (
                    analysis_start
                    + timedelta(
                        seconds=
                            (
                                bin_index
                                * bucket_seconds
                            )
                    )
                )

                bin_end = min(
                    (
                        bin_start
                        + timedelta(
                            seconds=
                                bucket_seconds
                        )
                    ),
                    analysis_end,
                )

                segment_start = max(
                    event_start,
                    bin_start,
                )

                segment_end = min(
                    event_end,
                    bin_end,
                )

                if (
                    segment_end
                    <= segment_start
                ):

                    continue

                active_durations[
                    bin_index
                ] += (
                    (
                        segment_end
                        - segment_start
                    ).total_seconds()
                )

        # ==============================================================
        # BUILD NORMALIZED OBSERVATION ROWS
        # ==============================================================

        result: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for bin_index in range(
            bin_count
        ):

            bucket_start = (
                analysis_start
                + timedelta(
                    seconds=
                        (
                            bin_index
                            * bucket_seconds
                        )
                )
            )

            bucket_end = min(
                (
                    bucket_start
                    + timedelta(
                        seconds=
                            bucket_seconds
                    )
                ),
                analysis_end,
            )

            bucket_duration_s = (
                bucket_end
                - bucket_start
            ).total_seconds()

            if (
                bucket_duration_s
                <= 0.0
            ):

                continue

            # ----------------------------------------------------------
            # ENVIRONMENTAL MEANS
            # ----------------------------------------------------------

            temperature_c = (
                None
            )

            if (
                temperatures[
                    bin_index
                ]
            ):

                temperature_c = (
                    math.fsum(
                        temperatures[
                            bin_index
                        ]
                    )
                    / len(
                        temperatures[
                            bin_index
                        ]
                    )
                )

            humidity_percent = (
                None
            )

            if (
                humidities[
                    bin_index
                ]
            ):

                humidity_percent = (
                    math.fsum(
                        humidities[
                            bin_index
                        ]
                    )
                    / len(
                        humidities[
                            bin_index
                        ]
                    )
                )

            pressure_hpa = (
                None
            )

            if (
                pressures[
                    bin_index
                ]
            ):

                pressure_hpa = (
                    math.fsum(
                        pressures[
                            bin_index
                        ]
                    )
                    / len(
                        pressures[
                            bin_index
                        ]
                    )
                )

            event_count = int(
                event_counts[
                    bin_index
                ]
            )

            active_duration_s = float(
                active_durations[
                    bin_index
                ]
            )

            event_rate_per_hour = (
                event_count
                / (
                    bucket_duration_s
                    / 3600.0
                )
            )

            result.append(
                {
                    "session_id":
                        session_id,

                    "session_label":
                        str(
                            session[
                                "label"
                            ]
                        ),

                    "bucket_start":
                        self._datetime_to_iso(
                            bucket_start
                        ),

                    "bucket_end":
                        self._datetime_to_iso(
                            bucket_end
                        ),

                    "bucket_duration_s":
                        float(
                            bucket_duration_s
                        ),

                    "telemetry_sample_count":
                        int(
                            telemetry_counts[
                                bin_index
                            ]
                        ),

                    "temperature_c":
                        temperature_c,

                    "humidity_percent":
                        humidity_percent,

                    "pressure_hpa":
                        pressure_hpa,

                    "event_count":
                        event_count,

                    "active_duration_s":
                        active_duration_s,

                    "event_rate_per_hour":
                        float(
                            event_rate_per_hour
                        ),
                }
            )

        return (
            result
        )