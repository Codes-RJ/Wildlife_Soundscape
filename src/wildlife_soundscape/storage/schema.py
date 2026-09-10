"""Schema operations behind the EventDatabase facade."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import EventDatabase
DEFAULT_ANALYTICS_SAMPLE_RATE = 48000
DEFAULT_ANALYTICS_BUCKET_SECONDS = 3600
UTC_SUFFIX = "Z"
UINT8_MAX = 255
UINT32_MAX = 4294967295
UINT64_MAX = 18446744073709551615
SQLITE_INT64_MAX = 9223372036854775807


def _initialize(
    self: EventDatabase,
) -> None:
    """
    Create tables, perform compatible migrations and create indexes.
    """

    with self._connect() as conn:
        # ==========================================================
        # SQLITE OPERATING MODE
        # ==========================================================

        conn.execute("PRAGMA journal_mode = WAL")

        conn.execute("PRAGMA synchronous = NORMAL")

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

            CREATE TABLE IF NOT EXISTS session_manifests (
                session_id INTEGER PRIMARY KEY REFERENCES sessions(session_id),
                manifest_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_metrics (
                session_id INTEGER PRIMARY KEY REFERENCES sessions(session_id),
                metrics_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_processing (
                event_id INTEGER NOT NULL REFERENCES events(id),
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                detail TEXT,
                PRIMARY KEY(event_id, stage)
            );

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


            --------------------------------------------------------
            -- SOUNDSCAPE INDICES TABLE
            --------------------------------------------------------

            CREATE TABLE IF NOT EXISTS
                soundscape_indices (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    session_id INTEGER NOT NULL,

                    node_id INTEGER NOT NULL,

                    start_sample INTEGER NOT NULL,

                    end_sample INTEGER NOT NULL,

                    aci REAL NOT NULL,

                    ndsi REAL NOT NULL,

                    acoustic_entropy REAL NOT NULL,

                    temporal_entropy REAL NOT NULL,

                    spectral_entropy REAL NOT NULL,

                    bioacoustic_index REAL NOT NULL,

                    anthrophony_power REAL NOT NULL,

                    biophony_power REAL NOT NULL,

                    parameters_json TEXT NOT NULL,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY(session_id)
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                );
            """
        )

        # ==========================================================
        # SIMPLE EVENT MIGRATIONS
        # ==========================================================

        self._ensure_column(
            conn,
            table="events",
            column="best_node_id",
            definition="INTEGER",
        )

        # ==========================================================
        # CLASSIFICATION SCHEMA MIGRATION
        # ==========================================================

        self._ensure_nullable_second_confidence(conn)

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


            CREATE INDEX IF NOT EXISTS
                idx_soundscape_indices_session_node

            ON soundscape_indices(
                session_id,
                node_id
            );


            CREATE INDEX IF NOT EXISTS
                idx_soundscape_indices_created

            ON soundscape_indices(
                created_at
            );
            """
        )
