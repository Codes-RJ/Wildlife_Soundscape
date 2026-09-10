from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import json
import math
import sqlite3
from . import schema, migrations, queries

from contextlib import (
    contextmanager,
)

from dataclasses import (
    asdict,
)
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
    cast,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.classification import (
    ClassificationResult,
)

from wildlife_soundscape.dsp.features import (
    AcousticFeatures,
)

from wildlife_soundscape.pipeline.event_detector import (
    AcousticEvent,
)

from wildlife_soundscape.localization import (
    LocalizationResult,
)

from wildlife_soundscape.core.protocol import (
    EnvironmentPayload,
)


# ======================================================================
# INTEGER LIMITS
# ======================================================================


UINT8_MAX = 0xFF


UINT32_MAX = 0xFFFFFFFF


UINT64_MAX = 0xFFFFFFFFFFFFFFFF


SQLITE_INT64_MAX = 0x7FFFFFFFFFFFFFFF


# ======================================================================
# ANALYTICS DEFAULTS
# ======================================================================


DEFAULT_ANALYTICS_SAMPLE_RATE = 48_000


DEFAULT_ANALYTICS_BUCKET_SECONDS = 3600


UTC_SUFFIX = "Z"


class SessionAlreadyExistsError(ValueError):
    """A new acquisition attempted to reuse historical identity."""


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

        self.path = Path(path)

        if not (self.path.name):
            raise ValueError(("Database path must contain a database filename."))

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # SQLite connections are short-lived, but write operations within
        # this process are still serialized.
        self._lock = Lock()

        self._initialize()

    # ==================================================================
    # CONNECTION
    # ==================================================================

    @contextmanager
    def _connect(
        self,
    ):
        """
        Open one configured SQLite connection as a context manager.

        A fresh connection is used for every database operation and is
        properly closed when the context exits.
        """

        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
        )

        try:
            # ----------------------------------------------------------
            # REFERENTIAL INTEGRITY
            # ----------------------------------------------------------

            connection.execute("PRAGMA foreign_keys = ON")

            # ----------------------------------------------------------
            # LOCK WAIT
            # ----------------------------------------------------------

            connection.execute("PRAGMA busy_timeout = 5000")

            with connection:
                yield connection

        finally:
            connection.close()

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

        try:
            result = float(value)

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(f"{name} must be numeric") from exc

        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")

        return result

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

        if value is None:
            return None

        return cls._finite_float(
            value,
            name=name,
        )

    # ==================================================================
    # BEST-EFFORT FINITE FLOAT
    # ==================================================================

    @staticmethod
    def _finite_float_or_none(
        value: Any,
    ) -> float | None:
        """Return a finite float, or None for an unavailable estimate."""

        try:
            result = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

        if not math.isfinite(result):
            return None

        return result

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

        if value is None:
            return None

        try:
            result = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

        if not math.isfinite(result) or result < 0.0:
            return None

        return result

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
            raise TypeError(f"{name} must be an integer")

        if not isinstance(
            value,
            int,
        ):
            raise TypeError(f"{name} must be an integer")

        result = int(value)

        if result <= 0:
            raise ValueError(f"{name} must be greater than 0")

        if maximum is not None and result > maximum:
            raise ValueError((f"{name} cannot exceed {maximum}"))

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
            raise TypeError(f"{name} must be an integer")

        if not isinstance(
            value,
            int,
        ):
            raise TypeError(f"{name} must be an integer")

        result = int(value)

        if not (0 <= result <= UINT64_MAX):
            raise ValueError((f"{name} must lie in the uint64 range"))

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

        if result > SQLITE_INT64_MAX:
            raise ValueError((f"{name} exceeds SQLite signed 64-bit INTEGER capacity"))

        return result

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

        if isinstance(
            value,
            bool,
        ) or not isinstance(
            value,
            int,
        ):
            raise TypeError("sample_rate must be an integer")

        if value <= 0:
            raise ValueError("sample_rate must be greater than 0")

        return int(value)

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

        if isinstance(
            value,
            bool,
        ) or not isinstance(
            value,
            int,
        ):
            raise TypeError("bucket_seconds must be an integer")

        if value <= 0:
            raise ValueError(("bucket_seconds must be greater than 0"))

        return int(value)

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
            result = value

        else:
            if not isinstance(
                value,
                str,
            ):
                raise TypeError((f"{name} must be datetime or timestamp string"))

            text = value.strip()

            if not (text):
                raise ValueError(f"{name} cannot be empty")

            if text.endswith(UTC_SUFFIX):
                text = text[:-1] + "+00:00"

            try:
                result = datetime.fromisoformat(text)

            except ValueError as exc:
                raise ValueError((f"{name} is not a valid ISO timestamp")) from exc

        # --------------------------------------------------------------
        # SQLITE CURRENT_TIMESTAMP IS UTC
        # --------------------------------------------------------------

        if result.tzinfo is None or result.utcoffset() is None:
            result = result.replace(tzinfo=timezone.utc)

        else:
            result = result.astimezone(timezone.utc)

        return result

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

        if value is None:
            return None

        return cls._parse_database_datetime(
            value,
            name=name,
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

        sample_rate = cls._positive_sample_rate(sample_rate)

        sample_index = cls._sample_index(
            sample_index,
            name="sample_index",
        )

        session_start = cls._parse_database_datetime(
            session_started_at,
            name="session_started_at",
        )

        offset_seconds = sample_index / float(sample_rate)

        try:
            return session_start + timedelta(seconds=offset_seconds)

        except OverflowError as exc:
            raise ValueError(
                ("sample_index produces a datetime outside supported range")
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
            raise TypeError("value must be datetime")

        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)

        else:
            value = value.astimezone(timezone.utc)

        return value.isoformat()

    # ==================================================================
    # DATABASE INITIALIZATION
    # ==================================================================

    def _initialize(
        self,
    ) -> None:
        return schema._initialize(self)

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
        return migrations._ensure_column(
            conn, table=table, column=column, definition=definition
        )

    # ==================================================================
    # CLASSIFICATION NULLABILITY MIGRATION
    # ==================================================================

    @staticmethod
    def _ensure_nullable_second_confidence(
        conn: sqlite3.Connection,
    ) -> None:
        return migrations._ensure_nullable_second_confidence(conn)

    # ==================================================================
    # SESSION OPERATIONS
    # ==================================================================

    def start_session(
        self,
        session_id: int,
        label: str,
        *,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        """
        Register the beginning of one acquisition session.

        A session identifier is immutable historical identity.

        If the same non-zero ID already exists, insertion fails rather
        than silently overwriting an older research session.
        """

        session_id = self._positive_id(
            session_id,
            name="session_id",
            maximum=UINT32_MAX,
        )

        if not isinstance(
            label,
            str,
        ):
            raise TypeError(("session label must be a string"))

        label = label.strip()

        if not (label):
            raise ValueError(("session label cannot be empty"))

        with (
            self._lock,
            self._connect() as conn,
        ):
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone():
                raise SessionAlreadyExistsError(f"Session {session_id} already exists")
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
            if manifest is not None:
                conn.execute(
                    "INSERT INTO session_manifests(session_id, manifest_json) VALUES (?, ?)",
                    (session_id, self._json_dumps(manifest)),
                )

    def get_session_manifest(self, session_id: int) -> dict[str, Any] | None:
        """Return the original experiment manifest; legacy sessions have none."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest_json FROM session_manifests WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def save_session_metrics(self, session_id: int, metrics: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO session_metrics(session_id, metrics_json) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET metrics_json=excluded.metrics_json",
                (session_id, self._json_dumps(metrics)),
            )

    def get_session_metrics(self, session_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metrics_json FROM session_metrics WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def set_event_processing_status(
        self, event_id: int, stage: str, status: str, detail: str | None = None
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO event_processing(event_id, stage, status, detail) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(event_id, stage) DO UPDATE SET status=excluded.status, detail=excluded.detail",
                (event_id, stage, status, detail),
            )

    def get_event_processing_status(self, event_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [
                {"stage": row[0], "status": row[1], "detail": row[2]}
                for row in conn.execute(
                    "SELECT stage, status, detail FROM event_processing WHERE event_id = ? ORDER BY stage",
                    (event_id,),
                )
            ]

    def update_event_analysis(
        self,
        event_id: int,
        *,
        environment: EnvironmentPayload | None,
        localization: LocalizationResult | None,
        best_node_id: int | None,
    ) -> None:
        """Attach optional results after the raw event has been committed."""
        values: dict[str, Any] = {"best_node_id": best_node_id}
        if best_node_id is not None:
            self._positive_id(best_node_id, name="best_node_id", maximum=UINT8_MAX)
        if environment is not None:
            values.update(
                temperature_c=self._finite_float(
                    environment.temperature_c, name="temperature_c"
                ),
                humidity_percent=self._finite_float(
                    environment.humidity_percent, name="humidity_percent"
                ),
                pressure_hpa=self._finite_float(
                    environment.pressure_hpa, name="pressure_hpa"
                ),
            )
        if localization is not None:
            position = localization.position
            values.update(
                speed_of_sound_mps=self._finite_float(
                    localization.speed_of_sound_mps, name="speed_of_sound_mps"
                ),
                localization_success=int(bool(position.success)),
                x_m=self._finite_float_or_none(position.x),
                y_m=self._finite_float_or_none(position.y),
                localization_residual_m=self._finite_float_or_none(
                    position.residual_rms_meters
                ),
            )
            if position.success and any(
                values[key] is None for key in ("x_m", "y_m", "localization_residual_m")
            ):
                raise ValueError(
                    "Successful localization must contain finite estimates"
                )
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE events SET "
                + ", ".join(f"{key} = ?" for key in values)
                + " WHERE id = ?",
                (*values.values(), event_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Event {event_id} does not exist")

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

        if session_id is None:
            return

        try:
            session_id = self._positive_id(
                session_id,
                name="session_id",
                maximum=UINT32_MAX,
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
                (session_id,),
            )

    # ==================================================================
    # LIST SESSIONS
    # ==================================================================

    def list_sessions(
        self,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
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

        if limit is not None:
            if isinstance(
                limit,
                bool,
            ):
                raise TypeError("limit must be an integer or None")

            if not isinstance(
                limit,
                int,
            ):
                raise TypeError("limit must be an integer or None")

            if limit <= 0:
                return []

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            if limit is None:
                rows = conn.execute(
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
                ).fetchall()

            else:
                rows = conn.execute(
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
                    (limit,),
                ).fetchall()

        return list(rows)

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

        session_id = self._positive_id(
            session_id,
            name="session_id",
            maximum=UINT32_MAX,
        )

        node_id = self._positive_id(
            node_id,
            name="node_id",
            maximum=UINT8_MAX,
        )

        sample_index = self._sample_index(
            sample_index,
            name="sample_index",
        )

        if not isinstance(
            environment,
            EnvironmentPayload,
        ):
            raise TypeError(("environment must be an EnvironmentPayload instance"))

        temperature = self._finite_float(
            environment.temperature_c,
            name="temperature_c",
        )

        humidity = self._finite_float(
            environment.humidity_percent,
            name="humidity_percent",
        )

        pressure = self._finite_float(
            environment.pressure_hpa,
            name="pressure_hpa",
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
            raise TypeError(("event must be an AcousticEvent instance"))

        # ==============================================================
        # EVENT IDENTITY
        # ==============================================================

        detector_event_id = self._positive_id(
            event.event_id,
            name="detector_event_id",
        )

        session_id = self._positive_id(
            event.session_id,
            name="event session_id",
            maximum=UINT32_MAX,
        )

        start_sample = self._sample_index(
            event.start_sample,
            name="event start_sample",
        )

        end_sample = self._sample_index(
            event.end_sample,
            name="event end_sample",
        )

        if end_sample < start_sample:
            raise ValueError(("event end_sample cannot be smaller than start_sample"))

        peak_rms_dbfs = self._finite_float(
            event.peak_rms_dbfs,
            name="event peak_rms_dbfs",
        )

        # ==============================================================
        # TRIGGER NODES
        # ==============================================================

        trigger_nodes = sorted(
            {
                self._positive_id(
                    node_id,
                    name="trigger node_id",
                    maximum=UINT8_MAX,
                )
                for node_id in event.trigger_nodes
            }
        )

        if not (trigger_nodes):
            raise ValueError(("AcousticEvent must contain at least one trigger node"))

        trigger_nodes_json = self._json_dumps(trigger_nodes)

        # ==============================================================
        # BEST NODE
        # ==============================================================

        if best_node_id is not None:
            best_node_id = self._positive_id(
                best_node_id,
                name="best_node_id",
                maximum=UINT8_MAX,
            )

        # ==============================================================
        # ENVIRONMENT
        # ==============================================================

        if environment is None:
            temperature = None

            humidity = None

            pressure = None

        else:
            if not isinstance(
                environment,
                EnvironmentPayload,
            ):
                raise TypeError(("environment must be EnvironmentPayload or None"))

            temperature = self._finite_float(
                environment.temperature_c,
                name="event temperature_c",
            )

            humidity = self._finite_float(
                environment.humidity_percent,
                name="event humidity_percent",
            )

            pressure = self._finite_float(
                environment.pressure_hpa,
                name="event pressure_hpa",
            )

        # ==============================================================
        # LOCALIZATION
        # ==============================================================

        if localization is None:
            speed_of_sound = None

            x_m = None

            y_m = None

            localization_success = None

            localization_residual = None

        else:
            if not isinstance(
                localization,
                LocalizationResult,
            ) and not hasattr(
                localization,
                "position",
            ):
                raise TypeError(("localization must be LocalizationResult or None"))

            speed_of_sound = self._finite_float(
                localization.speed_of_sound_mps,
                name="speed_of_sound_mps",
            )

            position = localization.position

            localization_success = int(bool(position.success))

            if localization_success:
                x_m = self._finite_float(
                    position.x,
                    name="localization x_m",
                )
                y_m = self._finite_float(
                    position.y,
                    name="localization y_m",
                )
                localization_residual = self._finite_float(
                    position.residual_rms_meters,
                    name="localization_residual_m",
                )

            else:
                # A failed solver may intentionally expose NaN coordinates
                # and residuals. Preserve the failed-attempt flag while
                # storing unavailable numeric estimates as SQL NULL so the
                # core acoustic event is not lost.
                x_m = self._finite_float_or_none(position.x)
                y_m = self._finite_float_or_none(position.y)
                localization_residual = self._finite_float_or_none(
                    position.residual_rms_meters
                )

        # ==============================================================
        # EVENT DIRECTORY
        # ==============================================================

        if event_directory is not None:
            event_directory = str(event_directory)

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

            database_event_id = cursor.lastrowid

            if database_event_id is None:
                raise RuntimeError(("SQLite did not return an event row ID"))

            return int(database_event_id)

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

        event_id = self._positive_id(
            event_id,
            name="event_id",
        )

        source_node_id = self._positive_id(
            source_node_id,
            name="source_node_id",
            maximum=UINT8_MAX,
        )

        if not isinstance(
            features,
            AcousticFeatures,
        ):
            raise TypeError(("features must be an AcousticFeatures instance"))

        # ==============================================================
        # MFCC
        # ==============================================================

        mfcc_mean = [
            self._finite_float(
                value,
                name="MFCC mean value",
            )
            for value in features.mfcc_mean
        ]

        mfcc_std = [
            self._finite_float(
                value,
                name="MFCC std value",
            )
            for value in features.mfcc_std
        ]

        mfcc_mean_json = self._json_dumps(mfcc_mean)

        mfcc_std_json = self._json_dumps(mfcc_std)

        # ==============================================================
        # SCALAR FEATURES
        # ==============================================================

        duration_s = self._finite_float(
            features.duration_s,
            name="features.duration_s",
        )

        rms = self._finite_float(
            features.rms,
            name="features.rms",
        )

        peak_amplitude = self._finite_float(
            features.peak_amplitude,
            name="features.peak_amplitude",
        )

        crest_factor = self._finite_float(
            features.crest_factor,
            name="features.crest_factor",
        )

        zero_crossing_rate = self._finite_float(
            features.zero_crossing_rate,
            name="features.zero_crossing_rate",
        )

        dominant_frequency_hz = self._finite_float(
            features.dominant_frequency_hz,
            name="features.dominant_frequency_hz",
        )

        spectral_centroid_hz = self._finite_float(
            features.spectral_centroid_hz,
            name="features.spectral_centroid_hz",
        )

        spectral_bandwidth_hz = self._finite_float(
            features.spectral_bandwidth_hz,
            name="features.spectral_bandwidth_hz",
        )

        spectral_rolloff_hz = self._finite_float(
            features.spectral_rolloff_hz,
            name="features.spectral_rolloff_hz",
        )

        spectral_flatness = self._finite_float(
            features.spectral_flatness,
            name="features.spectral_flatness",
        )

        spectral_flux = self._finite_float(
            features.spectral_flux,
            name="features.spectral_flux",
        )

        snr_db = self._optional_finite_float(
            features.snr_db,
            name="features.snr_db",
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

        return str(value)

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

        event_id = self._positive_id(
            event_id,
            name="event_id",
        )

        if not isinstance(
            result,
            ClassificationResult,
        ):
            raise TypeError(("result must be a ClassificationResult instance"))

        # ==============================================================
        # PRIMARY RESULT
        # ==============================================================

        label = str(result.label.value)

        confidence = self._finite_float(
            result.confidence,
            name="classification confidence",
        )

        margin = self._finite_float(
            result.margin,
            name="classification margin",
        )

        # ==============================================================
        # SECOND CANDIDATE
        # ==============================================================

        if result.second_label is None:
            second_label = None

            second_confidence = None

        else:
            second_label = str(result.second_label.value)

            second_confidence = self._optional_finite_float(
                result.second_confidence,
                name=("classification second_confidence"),
            )

        # ==============================================================
        # SCORES
        # ==============================================================

        scores = {
            self._classification_score_key(score_label): self._finite_float(
                score,
                name=(f"classification score {score_label!r}"),
            )
            for (
                score_label,
                score,
            ) in result.scores.items()
        }

        scores_json = self._json_dumps(
            scores,
            sort_keys=True,
        )

        # ==============================================================
        # REASONS
        # ==============================================================

        reasons_json = self._json_dumps([str(reason) for reason in result.reasons])

        # ==============================================================
        # CLASSIFIER IDENTITY
        # ==============================================================

        classifier_name = str(result.classifier_name)

        classifier_version = str(result.classifier_version)

        if not (classifier_name.strip()):
            raise ValueError(("classifier_name cannot be empty"))

        if not (classifier_version.strip()):
            raise ValueError(("classifier_version cannot be empty"))

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
    ) -> list[sqlite3.Row]:
        """
        Return newest events together with DSP and classification data.
        """

        if isinstance(
            limit,
            bool,
        ):
            raise TypeError("limit must be an integer")

        limit = int(limit)

        if limit <= 0:
            return []

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            query = (
                self._event_select_query()
                + """

                ORDER BY e.id DESC

                LIMIT ?
                """
            )

            rows = conn.execute(
                query,
                (limit,),
            ).fetchall()

            return list(rows)

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
            event_id = int(event_id)

        except (
            TypeError,
            ValueError,
        ):
            return None

        if event_id <= 0:
            return None

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            query = (
                self._event_select_query()
                + """

                WHERE e.id = ?

                LIMIT 1
                """
            )

            return conn.execute(
                query,
                (event_id,),
            ).fetchone()

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
        return queries.analytics_event_rows(
            self, sample_rate=sample_rate, session_id=session_id, start=start, end=end
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
        return queries.analytics_telemetry_rows(
            self,
            sample_rate=sample_rate,
            session_id=session_id,
            node_id=node_id,
            start=start,
            end=end,
        )

    # ==================================================================
    # SESSION METADATA
    # ==================================================================

    def _session_metadata(
        self,
        session_id: int,
    ) -> (
        dict[
            str,
            Any,
        ]
        | None
    ):
        return queries._session_metadata(self, session_id)

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
        return queries._analytics_event_duration_s(cls, row, sample_rate=sample_rate)

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
        return queries.analytics_environmental_bins(
            self,
            session_id=session_id,
            bucket_seconds=bucket_seconds,
            sample_rate=sample_rate,
            node_id=node_id,
            start=start,
            end=end,
        )

    # ==================================================================
    # SOUNDSCAPE INDICES PERSISTENCE
    # ==================================================================

    def add_soundscape_indices(
        self,
        *,
        session_id: int,
        node_id: int,
        start_sample: int,
        end_sample: int,
        result: Any,
    ) -> int:
        """
        Persist continuous ecoacoustic indices for one node analysis window.
        """
        session_id = self._positive_id(
            session_id,
            name="soundscape_indices session_id",
            maximum=UINT32_MAX,
        )

        node_id = self._positive_id(
            node_id,
            name="soundscape_indices node_id",
            maximum=UINT8_MAX,
        )

        start_sample = self._sample_index(
            start_sample,
            name="soundscape_indices start_sample",
        )

        end_sample = self._sample_index(
            end_sample,
            name="soundscape_indices end_sample",
        )

        if end_sample < start_sample:
            raise ValueError("end_sample cannot be smaller than start_sample")

        aci = self._finite_float(
            getattr(
                result, "aci", result.get("aci") if isinstance(result, dict) else None
            ),
            name="aci",
        )

        ndsi = self._finite_float(
            getattr(
                result, "ndsi", result.get("ndsi") if isinstance(result, dict) else None
            ),
            name="ndsi",
        )

        acoustic_entropy = self._finite_float(
            getattr(
                result,
                "acoustic_entropy",
                result.get("acoustic_entropy") if isinstance(result, dict) else None,
            ),
            name="acoustic_entropy",
        )

        temporal_entropy = self._finite_float(
            getattr(
                result,
                "temporal_entropy",
                result.get("temporal_entropy") if isinstance(result, dict) else None,
            ),
            name="temporal_entropy",
        )

        spectral_entropy = self._finite_float(
            getattr(
                result,
                "spectral_entropy",
                result.get("spectral_entropy") if isinstance(result, dict) else None,
            ),
            name="spectral_entropy",
        )

        bioacoustic_index = self._finite_float(
            getattr(
                result,
                "bioacoustic_index",
                result.get("bioacoustic_index") if isinstance(result, dict) else None,
            ),
            name="bioacoustic_index",
        )

        anthrophony_power = self._finite_float(
            getattr(
                result,
                "anthrophony_power",
                result.get("anthrophony_power") if isinstance(result, dict) else None,
            ),
            name="anthrophony_power",
        )

        biophony_power = self._finite_float(
            getattr(
                result,
                "biophony_power",
                result.get("biophony_power") if isinstance(result, dict) else None,
            ),
            name="biophony_power",
        )

        params = getattr(
            result,
            "parameters",
            result.get("parameters") if isinstance(result, dict) else {},
        )
        if hasattr(params, "__dataclass_fields__"):
            params_dict = asdict(
                cast(
                    Any,
                    params,
                )
            )
        elif isinstance(params, dict):
            params_dict = params
        else:
            params_dict = {}

        parameters_json = self._json_dumps(params_dict)

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO soundscape_indices(
                    session_id,
                    node_id,
                    start_sample,
                    end_sample,
                    aci,
                    ndsi,
                    acoustic_entropy,
                    temporal_entropy,
                    spectral_entropy,
                    bioacoustic_index,
                    anthrophony_power,
                    biophony_power,
                    parameters_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    node_id,
                    start_sample,
                    end_sample,
                    aci,
                    ndsi,
                    acoustic_entropy,
                    temporal_entropy,
                    spectral_entropy,
                    bioacoustic_index,
                    anthrophony_power,
                    biophony_power,
                    parameters_json,
                ),
            )
            return int(cursor.lastrowid)

    def get_soundscape_indices(
        self,
        *,
        session_id: int | None = None,
        node_id: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return queries.get_soundscape_indices(
            self, session_id=session_id, node_id=node_id, limit=limit
        )
