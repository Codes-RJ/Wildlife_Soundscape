from __future__ import annotations

import json
import math
import sqlite3

from pathlib import Path
from threading import Lock
from typing import Any

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
                timeout=5.0,
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

        return connection

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
            allow_nan=False,
            sort_keys=sort_keys,
            separators=(
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

        result = float(
            value
        )

        if not math.isfinite(
            result
        ):

            raise ValueError(
                f"{name} must be finite"
            )

        return result

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

        if value is None:

            return None

        return cls._finite_float(
            value,
            name=name,
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

        if result <= 0:

            raise ValueError(
                f"{name} must be greater than 0"
            )

        if (
            maximum is not None
            and result > maximum
        ):

            raise ValueError(
                (
                    f"{name} cannot exceed "
                    f"{maximum}"
                )
            )

        return result

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

        if result > 0x7FFFFFFFFFFFFFFF:

            raise ValueError(
                (
                    f"{name} exceeds SQLite "
                    "signed 64-bit INTEGER capacity"
                )
            )

        return result

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
                row[1]
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
                    row[1]
                )
                == "second_confidence"
            ),
            None,
        )

        # Fresh/current schema.
        if (
            second_column is None
            or int(
                second_column[3]
            )
            == 0
        ):

            return

        # ==============================================================
        # OLD TABLE REQUIRES REBUILD
        # ==========================================================

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

        if not label:

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

        if not trigger_nodes:

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
                sort_keys=True,
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

            return []

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

            return None

        try:

            event_id = int(
                event_id
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        if (
            event_id
            <= 0
        ):

            return None

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