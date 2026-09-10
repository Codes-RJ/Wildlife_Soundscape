"""Queries operations behind the EventDatabase facade."""

from __future__ import annotations
import json
import math
import sqlite3
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .database import EventDatabase
DEFAULT_ANALYTICS_SAMPLE_RATE = 48000
DEFAULT_ANALYTICS_BUCKET_SECONDS = 3600
UTC_SUFFIX = "Z"
UINT8_MAX = 255
UINT32_MAX = 4294967295
UINT64_MAX = 18446744073709551615
SQLITE_INT64_MAX = 9223372036854775807


def analytics_event_rows(
    self: EventDatabase,
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

    sample_rate = self._positive_sample_rate(sample_rate)

    if session_id is not None:
        session_id = self._positive_id(
            session_id,
            name="session_id",
            maximum=UINT32_MAX,
        )

    start_dt = self._optional_query_datetime(
        start,
        name="start",
    )

    end_dt = self._optional_query_datetime(
        end,
        name="end",
    )

    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise ValueError("end cannot be earlier than start")

    with self._connect() as conn:
        conn.row_factory = sqlite3.Row

        query = self._analytics_event_select_query()

        parameters: list[Any] = []

        if session_id is not None:
            query += """

                WHERE e.session_id = ?
            """

            parameters.append(session_id)

        query += """

            ORDER BY

                s.started_at ASC,

                e.start_sample ASC,

                e.id ASC
        """

        raw_rows = conn.execute(
            query,
            tuple(parameters),
        ).fetchall()

    result: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for raw_row in raw_rows:
        row = dict(raw_row)

        event_time = self._sample_time(
            session_started_at=row["session_started_at"],
            sample_index=int(row["start_sample"]),
            sample_rate=sample_rate,
        )

        # ----------------------------------------------------------
        # HALF-OPEN QUERY INTERVAL
        #
        #     [start, end)
        # ----------------------------------------------------------

        if start_dt is not None and event_time < start_dt:
            continue

        if end_dt is not None and event_time >= end_dt:
            continue

        row["event_time"] = self._datetime_to_iso(event_time)

        # ----------------------------------------------------------
        # NORMALIZED EVENT END TIME
        # ----------------------------------------------------------

        event_end_time = self._sample_time(
            session_started_at=row["session_started_at"],
            sample_index=int(row["end_sample"]),
            sample_rate=sample_rate,
        )

        row["event_end_time"] = self._datetime_to_iso(event_end_time)

        result.append(row)

    return result


def analytics_telemetry_rows(
    self: EventDatabase,
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

    sample_rate = self._positive_sample_rate(sample_rate)

    if session_id is not None:
        session_id = self._positive_id(
            session_id,
            name="session_id",
            maximum=UINT32_MAX,
        )

    if node_id is not None:
        node_id = self._positive_id(
            node_id,
            name="node_id",
            maximum=UINT8_MAX,
        )

    start_dt = self._optional_query_datetime(
        start,
        name="start",
    )

    end_dt = self._optional_query_datetime(
        end,
        name="end",
    )

    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise ValueError("end cannot be earlier than start")

    where_parts: list[str] = []

    parameters: list[Any] = []

    if session_id is not None:
        where_parts.append("t.session_id = ?")

        parameters.append(session_id)

    if node_id is not None:
        where_parts.append("t.node_id = ?")

        parameters.append(node_id)

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

    if where_parts:
        query += """

            WHERE
            """ + " AND ".join(where_parts)

    query += """

        ORDER BY

            s.started_at ASC,

            t.sample_index ASC,

            t.id ASC
    """

    with self._connect() as conn:
        conn.row_factory = sqlite3.Row

        raw_rows = conn.execute(
            query,
            tuple(parameters),
        ).fetchall()

    result: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for raw_row in raw_rows:
        row = dict(raw_row)

        telemetry_time = self._sample_time(
            session_started_at=row["session_started_at"],
            sample_index=int(row["sample_index"]),
            sample_rate=sample_rate,
        )

        if start_dt is not None and telemetry_time < start_dt:
            continue

        if end_dt is not None and telemetry_time >= end_dt:
            continue

        row["telemetry_time"] = self._datetime_to_iso(telemetry_time)

        result.append(row)

    return result


def _session_metadata(
    self: EventDatabase,
    session_id: int,
) -> (
    dict[
        str,
        Any,
    ]
    | None
):
    """
    Return one session as a plain dictionary.
    """

    session_id = self._positive_id(
        session_id,
        name="session_id",
        maximum=UINT32_MAX,
    )

    with self._connect() as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
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
            (session_id,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


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

    feature_duration = cls._optional_nonnegative_float(row.get("duration_s"))

    if feature_duration is not None:
        return feature_duration

    try:
        start_sample = int(row["start_sample"])

        end_sample = int(row["end_sample"])

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return 0.0

    if start_sample < 0 or end_sample < start_sample:
        return 0.0

    return (end_sample - start_sample) / float(sample_rate)


def analytics_environmental_bins(
    self: EventDatabase,
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

    session_id = self._positive_id(
        session_id,
        name="session_id",
        maximum=UINT32_MAX,
    )

    bucket_seconds = self._positive_bucket_seconds(bucket_seconds)

    sample_rate = self._positive_sample_rate(sample_rate)

    if node_id is not None:
        node_id = self._positive_id(
            node_id,
            name="node_id",
            maximum=UINT8_MAX,
        )

    # ==============================================================
    # SESSION
    # ==============================================================

    session = self._session_metadata(session_id)

    if session is None:
        raise ValueError((f"Unknown session_id: 0x{session_id:08X}"))

    session_start = self._parse_database_datetime(
        session["started_at"],
        name="session.started_at",
    )

    session_stop = None

    if session["stopped_at"] is not None:
        session_stop = self._parse_database_datetime(
            session["stopped_at"],
            name="session.stopped_at",
        )

    # ==============================================================
    # USER REQUESTED RANGE
    # ==============================================================

    requested_start = self._optional_query_datetime(
        start,
        name="start",
    )

    requested_end = self._optional_query_datetime(
        end,
        name="end",
    )

    if (
        requested_start is not None
        and requested_end is not None
        and requested_end <= requested_start
    ):
        raise ValueError("end must be later than start")

    # ==============================================================
    # LOAD COMPLETE SESSION DATA
    # ==============================================================
    #
    # Do not apply start/end yet.
    #
    # An event beginning just before the requested window may still
    # overlap the first bin and contribute active duration.
    # ==============================================================

    event_rows = self.analytics_event_rows(
        sample_rate=sample_rate,
        session_id=session_id,
    )

    telemetry_rows = self.analytics_telemetry_rows(
        sample_rate=sample_rate,
        session_id=session_id,
        node_id=node_id,
    )

    # ==============================================================
    # RESOLVE OBSERVATION START
    # ==============================================================

    analysis_start = session_start

    if requested_start is not None:
        analysis_start = max(
            analysis_start,
            requested_start,
        )

    # ==============================================================
    # RESOLVE OBSERVATION END
    # ==============================================================

    if session_stop is not None:
        natural_end = session_stop

    else:
        # ----------------------------------------------------------
        # ACTIVE / UNSTOPPED SESSION
        #
        # Use latest known sample-derived observation rather than
        # inventing an unobserved future interval.
        # ----------------------------------------------------------

        candidates: list[datetime] = []

        for row in telemetry_rows:
            candidates.append(
                self._parse_database_datetime(
                    row["telemetry_time"],
                    name="telemetry_time",
                )
            )

        for row in event_rows:
            event_end = self._parse_database_datetime(
                row["event_end_time"],
                name="event_end_time",
            )

            candidates.append(event_end)

        if not (candidates):
            return []

        natural_end = max(candidates)

    analysis_end = natural_end

    if requested_end is not None:
        analysis_end = min(
            analysis_end,
            requested_end,
        )

    if analysis_end <= analysis_start:
        return []

    # ==============================================================
    # BIN COUNT
    # ==============================================================

    observation_duration_s = (analysis_end - analysis_start).total_seconds()

    bin_count = int(math.ceil(observation_duration_s / bucket_seconds))

    if bin_count <= 0:
        return []

    # ==============================================================
    # ACCUMULATORS
    # ==============================================================

    event_counts = [0 for _ in range(bin_count)]

    active_durations = [0.0 for _ in range(bin_count)]

    temperatures: list[list[float]] = [[] for _ in range(bin_count)]

    humidities: list[list[float]] = [[] for _ in range(bin_count)]

    pressures: list[list[float]] = [[] for _ in range(bin_count)]

    telemetry_counts = [0 for _ in range(bin_count)]

    # ==============================================================
    # TELEMETRY -> BINS
    # ==============================================================

    for row in telemetry_rows:
        timestamp = self._parse_database_datetime(
            row["telemetry_time"],
            name="telemetry_time",
        )

        if not (analysis_start <= timestamp < analysis_end):
            continue

        elapsed_s = (timestamp - analysis_start).total_seconds()

        bin_index = int(elapsed_s // bucket_seconds)

        if not (0 <= bin_index < bin_count):
            continue

        temperature = self._optional_finite_float(
            row.get("temperature_c"),
            name="temperature_c",
        )

        humidity = self._optional_finite_float(
            row.get("humidity_percent"),
            name="humidity_percent",
        )

        pressure = self._optional_finite_float(
            row.get("pressure_hpa"),
            name="pressure_hpa",
        )

        telemetry_counts[bin_index] += 1

        if temperature is not None:
            temperatures[bin_index].append(temperature)

        if humidity is not None:
            humidities[bin_index].append(humidity)

        if pressure is not None:
            pressures[bin_index].append(pressure)

    # ==============================================================
    # EVENTS -> BINS
    # ==============================================================

    for row in event_rows:
        event_start = self._parse_database_datetime(
            row["event_time"],
            name="event_time",
        )

        duration_s = self._analytics_event_duration_s(
            row,
            sample_rate=sample_rate,
        )

        event_end = event_start + timedelta(seconds=duration_s)

        # ----------------------------------------------------------
        # EVENT-ONSET COUNT
        # ----------------------------------------------------------

        if analysis_start <= event_start < analysis_end:
            elapsed_s = (event_start - analysis_start).total_seconds()

            onset_bin = int(elapsed_s // bucket_seconds)

            if 0 <= onset_bin < bin_count:
                event_counts[onset_bin] += 1

        # ----------------------------------------------------------
        # ZERO-DURATION EVENT
        # ----------------------------------------------------------

        if duration_s <= 0.0:
            continue

        # ----------------------------------------------------------
        # EVENT DOES NOT OVERLAP ANALYSIS WINDOW
        # ----------------------------------------------------------

        if event_end <= analysis_start or event_start >= analysis_end:
            continue

        overlap_start = max(
            event_start,
            analysis_start,
        )

        overlap_end = min(
            event_end,
            analysis_end,
        )

        if overlap_end <= overlap_start:
            continue

        # ----------------------------------------------------------
        # FIRST / LAST OVERLAPPED BIN
        # ----------------------------------------------------------

        first_bin = int(
            (overlap_start - analysis_start).total_seconds() // bucket_seconds
        )

        last_position_s = (overlap_end - analysis_start).total_seconds()

        # ----------------------------------------------------------
        # Subtract a tiny numerical epsilon so an event ending
        # exactly on a bin boundary does not spill into the next bin.
        # ----------------------------------------------------------

        last_bin = int(
            max(
                0.0,
                last_position_s - 1e-12,
            )
            // bucket_seconds
        )

        first_bin = max(
            0,
            min(
                first_bin,
                bin_count - 1,
            ),
        )

        last_bin = max(
            0,
            min(
                last_bin,
                bin_count - 1,
            ),
        )

        for bin_index in range(
            first_bin,
            last_bin + 1,
        ):
            bin_start = analysis_start + timedelta(seconds=(bin_index * bucket_seconds))

            bin_end = min(
                (bin_start + timedelta(seconds=bucket_seconds)),
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

            if segment_end <= segment_start:
                continue

            active_durations[bin_index] += (segment_end - segment_start).total_seconds()

    # ==============================================================
    # BUILD NORMALIZED OBSERVATION ROWS
    # ==============================================================

    result: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for bin_index in range(bin_count):
        bucket_start = analysis_start + timedelta(seconds=(bin_index * bucket_seconds))

        bucket_end = min(
            (bucket_start + timedelta(seconds=bucket_seconds)),
            analysis_end,
        )

        bucket_duration_s = (bucket_end - bucket_start).total_seconds()

        if bucket_duration_s <= 0.0:
            continue

        # ----------------------------------------------------------
        # ENVIRONMENTAL MEANS
        # ----------------------------------------------------------

        temperature_c = None

        if temperatures[bin_index]:
            temperature_c = math.fsum(temperatures[bin_index]) / len(
                temperatures[bin_index]
            )

        humidity_percent = None

        if humidities[bin_index]:
            humidity_percent = math.fsum(humidities[bin_index]) / len(
                humidities[bin_index]
            )

        pressure_hpa = None

        if pressures[bin_index]:
            pressure_hpa = math.fsum(pressures[bin_index]) / len(pressures[bin_index])

        event_count = int(event_counts[bin_index])

        active_duration_s = float(active_durations[bin_index])

        event_rate_per_hour = event_count / (bucket_duration_s / 3600.0)

        result.append(
            {
                "session_id": session_id,
                "session_label": str(session["label"]),
                "bucket_start": self._datetime_to_iso(bucket_start),
                "bucket_end": self._datetime_to_iso(bucket_end),
                "bucket_duration_s": float(bucket_duration_s),
                "telemetry_sample_count": int(telemetry_counts[bin_index]),
                "temperature_c": temperature_c,
                "humidity_percent": humidity_percent,
                "pressure_hpa": pressure_hpa,
                "event_count": event_count,
                "active_duration_s": active_duration_s,
                "event_rate_per_hour": float(event_rate_per_hour),
            }
        )

    return result


def get_soundscape_indices(
    self: EventDatabase,
    *,
    session_id: int | None = None,
    node_id: int | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """
    Query persisted continuous ecoacoustic index records.
    """
    query_parts = ["SELECT * FROM soundscape_indices WHERE 1=1"]
    params: list[Any] = []

    if session_id is not None:
        query_parts.append("AND session_id = ?")
        params.append(
            self._positive_id(
                session_id,
                name="soundscape_indices session_id",
                maximum=UINT32_MAX,
            )
        )

    if node_id is not None:
        query_parts.append("AND node_id = ?")
        params.append(
            self._positive_id(
                node_id,
                name="soundscape_indices node_id",
                maximum=UINT8_MAX,
            )
        )

    query_parts.append("ORDER BY id ASC LIMIT ?")
    params.append(max(1, int(limit)))

    query = " ".join(query_parts)

    with self._lock, self._connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, tuple(params)).fetchall()

        return [
            {
                "id": int(row["id"]),
                "session_id": int(row["session_id"]),
                "node_id": int(row["node_id"]),
                "start_sample": int(row["start_sample"]),
                "end_sample": int(row["end_sample"]),
                "aci": float(row["aci"]),
                "ndsi": float(row["ndsi"]),
                "acoustic_entropy": float(row["acoustic_entropy"]),
                "temporal_entropy": float(row["temporal_entropy"]),
                "spectral_entropy": float(row["spectral_entropy"]),
                "bioacoustic_index": float(row["bioacoustic_index"]),
                "anthrophony_power": float(row["anthrophony_power"]),
                "biophony_power": float(row["biophony_power"]),
                "parameters": json.loads(row["parameters_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]
