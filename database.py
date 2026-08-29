from __future__ import annotations

import json
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
        Core acoustic-event metadata, associated environmental
        conditions and localization results.

    event_features
        DSP/acoustic descriptors generated from the selected
        best-quality microphone channel.

    classifications
        Broad acoustic classification result associated with an event.

    Storage design
    --------------
    Raw audio is deliberately NOT stored inside SQLite.

    Audio files:
        filesystem / WAV

    Structured metadata:
        SQLite

    This keeps the database relatively compact and allows event audio
    to remain directly accessible for later research, reprocessing and
    model development.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(
            path
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # All writes are serialized inside this process.
        self._lock = Lock()

        self._initialize()

    # ==================================================================
    # CONNECTION
    # ==================================================================

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """
        Open one configured SQLite connection.

        A fresh connection is used per database operation.

        WAL mode is configured during database initialization rather
        than repeatedly for every connection.
        """

        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
        )

        # --------------------------------------------------------------
        # FOREIGN KEYS
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
        Serialize data using standards-compatible JSON.

        NaN and infinity are rejected rather than silently persisting
        non-standard JSON values.
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
    # INITIALIZATION
    # ==================================================================

    def _initialize(
        self,
    ) -> None:
        """
        Create required database tables and indexes.

        Existing databases are retained and simple compatible migrations
        are applied where possible.
        """

        with self._connect() as conn:

            # ----------------------------------------------------------
            # WAL
            # ----------------------------------------------------------
            #
            # WAL improves concurrent reads while the receiver continues
            # writing event information.
            # ----------------------------------------------------------

            conn.execute(
                "PRAGMA journal_mode = WAL"
            )

            conn.execute(
                "PRAGMA synchronous = NORMAL"
            )

            # ==========================================================
            # SCHEMA
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

                    second_confidence REAL NOT NULL,

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

            # ==========================================================
            # DATABASE MIGRATION
            # ==========================================================
            #
            # Older events tables may predate best_node_id.
            #
            # CREATE TABLE IF NOT EXISTS does not alter an already
            # existing table, therefore the column must be checked
            # separately.
            # ==========================================================

            self._ensure_column(
                conn,
                table="events",
                column="best_node_id",
                definition="INTEGER",
            )

    # ==================================================================
    # SIMPLE MIGRATION SUPPORT
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
        Add one column only when it does not already exist.

        This helper is intended only for controlled internal schema
        migrations using hard-coded table/column names.
        """

        rows = conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()

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
    # SESSION OPERATIONS
    # ==================================================================

    def start_session(
        self,
        session_id: int,
        label: str,
    ) -> None:
        """
        Register the beginning of one acquisition session.

        Session IDs are generated by the laptop and shared with all
        ESP32 nodes.
        """

        session_id = int(
            session_id
        )

        label = str(
            label
        ).strip()

        if (
            session_id
            <= 0
        ):

            raise ValueError(
                "session_id must be greater than 0"
            )

        if not label:

            raise ValueError(
                "session label cannot be empty"
            )

        with (
            self._lock,
            self._connect() as conn,
        ):

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

                ON CONFLICT(session_id)
                DO UPDATE SET

                    label =
                        excluded.label,

                    started_at =
                        CURRENT_TIMESTAMP,

                    stopped_at =
                        NULL
                """,
                (
                    session_id,
                    label,
                ),
            )

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

        session_id = int(
            session_id
        )

        if (
            session_id
            <= 0
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
        Persist one environmental telemetry sample.
        """

        session_id = int(
            session_id
        )

        node_id = int(
            node_id
        )

        sample_index = int(
            sample_index
        )

        if (
            session_id
            <= 0
        ):

            raise ValueError(
                "session_id must be greater than 0"
            )

        if (
            node_id
            <= 0
        ):

            raise ValueError(
                "node_id must be greater than 0"
            )

        if (
            sample_index
            < 0
        ):

            raise ValueError(
                "sample_index cannot be negative"
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

                    float(
                        environment.temperature_c
                    ),

                    float(
                        environment.humidity_percent
                    ),

                    float(
                        environment.pressure_hpa
                    ),
                ),
            )

    # ==================================================================
    # EVENT OPERATIONS
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

        DSP features and classification outputs are deliberately stored
        in their own one-to-one tables.
        """

        if (
            event.session_id
            <= 0
        ):

            raise ValueError(
                "event session_id must be greater than 0"
            )

        if (
            event.start_sample
            < 0
        ):

            raise ValueError(
                "event start_sample cannot be negative"
            )

        if (
            event.end_sample
            < event.start_sample
        ):

            raise ValueError(
                (
                    "event end_sample cannot be "
                    "smaller than start_sample"
                )
            )

        position = (
            localization.position
            if localization is not None
            else None
        )

        trigger_nodes = sorted(
            int(
                node_id
            )
            for node_id
            in event.trigger_nodes
        )

        trigger_nodes_json = (
            self._json_dumps(
                trigger_nodes
            )
        )

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
                    int(
                        event.event_id
                    ),

                    int(
                        event.session_id
                    ),

                    int(
                        event.start_sample
                    ),

                    int(
                        event.end_sample
                    ),

                    trigger_nodes_json,

                    float(
                        event.peak_rms_dbfs
                    ),

                    (
                        None
                        if best_node_id is None
                        else int(
                            best_node_id
                        )
                    ),

                    (
                        None
                        if environment is None
                        else float(
                            environment.temperature_c
                        )
                    ),

                    (
                        None
                        if environment is None
                        else float(
                            environment.humidity_percent
                        )
                    ),

                    (
                        None
                        if environment is None
                        else float(
                            environment.pressure_hpa
                        )
                    ),

                    (
                        None
                        if localization is None
                        else float(
                            localization.speed_of_sound_mps
                        )
                    ),

                    (
                        None
                        if position is None
                        else float(
                            position.x
                        )
                    ),

                    (
                        None
                        if position is None
                        else float(
                            position.y
                        )
                    ),

                    (
                        None
                        if position is None
                        else int(
                            bool(
                                position.success
                            )
                        )
                    ),

                    (
                        None
                        if position is None
                        else float(
                            position.residual_rms_meters
                        )
                    ),

                    (
                        None
                        if event_directory is None
                        else str(
                            event_directory
                        )
                    ),
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
                        "SQLite did not return an "
                        "event row ID"
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

        `event_id` refers to events.id, not detector_event_id.
        """

        event_id = int(
            event_id
        )

        source_node_id = int(
            source_node_id
        )

        if (
            event_id
            <= 0
        ):

            raise ValueError(
                "event_id must be greater than 0"
            )

        if (
            source_node_id
            <= 0
        ):

            raise ValueError(
                "source_node_id must be greater than 0"
            )

        mfcc_mean_json = (
            self._json_dumps(
                [
                    float(
                        value
                    )
                    for value
                    in features.mfcc_mean
                ]
            )
        )

        mfcc_std_json = (
            self._json_dumps(
                [
                    float(
                        value
                    )
                    for value
                    in features.mfcc_std
                ]
            )
        )

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

                    float(
                        features.duration_s
                    ),

                    float(
                        features.rms
                    ),

                    float(
                        features.peak_amplitude
                    ),

                    float(
                        features.crest_factor
                    ),

                    float(
                        features.zero_crossing_rate
                    ),

                    float(
                        features.dominant_frequency_hz
                    ),

                    float(
                        features.spectral_centroid_hz
                    ),

                    float(
                        features.spectral_bandwidth_hz
                    ),

                    float(
                        features.spectral_rolloff_hz
                    ),

                    float(
                        features.spectral_flatness
                    ),

                    float(
                        features.spectral_flux
                    ),

                    (
                        None
                        if features.snr_db is None
                        else float(
                            features.snr_db
                        )
                    ),

                    mfcc_mean_json,

                    mfcc_std_json,
                ),
            )

    # ==================================================================
    # CLASSIFICATION OPERATIONS
    # ==================================================================

    @staticmethod
    def _classification_score_key(
        key: Any,
    ) -> str:
        """
        Convert a classification score key into a stable string.

        Current ClassificationResult.scores uses string keys.

        Supporting `.value` here also makes persistence robust if a
        future backend supplies enum-style score keys.
        """

        value = getattr(
            key,
            "value",
            key,
        )

        return str(
            value
        )

    def add_classification(
        self,
        *,
        event_id: int,
        result: ClassificationResult,
    ) -> None:
        """
        Store or update broad acoustic classification for one event.
        """

        event_id = int(
            event_id
        )

        if (
            event_id
            <= 0
        ):

            raise ValueError(
                "event_id must be greater than 0"
            )

        scores = {
            self._classification_score_key(
                label
            ):
                float(
                    score
                )

            for (
                label,
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

        second_label = (
            None
            if result.second_label is None
            else str(
                result.second_label.value
            )
        )

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

                    str(
                        result.label.value
                    ),

                    float(
                        result.confidence
                    ),

                    second_label,

                    float(
                        result.second_confidence
                    ),

                    float(
                        result.margin
                    ),

                    scores_json,

                    reasons_json,

                    str(
                        result.classifier_name
                    ),

                    str(
                        result.classifier_version
                    ),
                ),
            )

    # ==================================================================
    # SHARED EVENT QUERY
    # ==================================================================

    @staticmethod
    def _event_select_query() -> str:
        """
        Base query joining core event, DSP and classification data.

        Keeping this query centralized ensures recent_events() and
        get_event() expose the same field names.
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
        Return recent events together with DSP and classification data.
        """

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

            rows = conn.execute(
                query,
                (
                    limit,
                ),
            ).fetchall()

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

        event_id = int(
            event_id
        )

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

            return conn.execute(
                query,
                (
                    event_id,
                ),
            ).fetchone()