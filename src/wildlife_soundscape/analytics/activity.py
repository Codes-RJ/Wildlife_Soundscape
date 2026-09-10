"""
Temporal acoustic-activity analytics.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module converts normalized acoustic-event records into temporal
activity metrics suitable for:

    research analysis
    dashboard plots
    CSV / report export
    diurnal activity studies
    class-wise activity comparisons

Primary outputs
---------------
ActivityBin
    Event activity accumulated into fixed temporal bins.

ClassActivitySummary
    Per-class event count, duration and confidence statistics.

ActivitySummary
    Overall temporal summary for one analysis window.

Scientific interpretation
-------------------------
The system measures DETECTED ACOUSTIC ACTIVITY.

It does not directly measure:

    animal abundance
    population size
    individual-animal identity
    confirmed physical presence duration
    verified behavioral state

A high number of acoustic events therefore means:

    "more detected acoustic events"

not automatically:

    "more animals were present"

Timestamp contract
------------------
For research analytics, rows should expose:

    event_time

representing the estimated acoustic-event start time.

The future database analytics query should derive this from:

    session.started_at
        +
    event.start_sample / sample_rate

rather than using database insertion time as the scientific event
timestamp.

Expected normalized row fields
------------------------------
Required:

    event_time

Recommended:

    id
    start_sample
    end_sample
    duration_s
    classification_label
    classification_confidence

If duration_s is unavailable, duration is reconstructed from:

    (end_sample - start_sample) / sample_rate
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from collections import (
    Counter,
    defaultdict,
)

from dataclasses import (
    dataclass,
)

from datetime import (
    datetime,
    timedelta,
)

from typing import (
    Any,
    Iterable,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from .models import (
    ActivityBin,
    ActivitySummary,
    AnalyticsTimeWindow,
    ClassActivitySummary,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_SAMPLE_RATE = 48_000


DEFAULT_BUCKET_SECONDS = 3600


DEFAULT_TIMESTAMP_KEY = "event_time"


UNCLASSIFIED_LABEL = "unclassified"


UTC_SUFFIX = "Z"


# ======================================================================
# INTERNAL NORMALIZED EVENT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class _ActivityEvent:
    """
    Internal normalized representation used only by this module.
    """

    event_id: int | None

    timestamp: datetime

    duration_s: float

    class_label: str

    confidence: float | None

    # ==================================================================
    # END TIME
    # ==================================================================

    @property
    def end_time(
        self,
    ) -> datetime:
        """
        Acoustic-event end time derived from timestamp + duration.
        """

        return self.timestamp + timedelta(seconds=self.duration_s)


# ======================================================================
# ROW ACCESS
# ======================================================================


def _row_value(
    row: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a field from dictionary-like or sqlite3.Row-like input.

    sqlite3.Row supports:

        row["field"]

    but does not behave exactly like a normal dict for every operation,
    so access is kept intentionally conservative.
    """

    # ==============================================================
    # DICTIONARY-LIKE GET
    # ==============================================================

    getter = getattr(
        row,
        "get",
        None,
    )

    if callable(getter):
        try:
            return getter(
                key,
                default,
            )

        except (
            KeyError,
            TypeError,
        ):
            pass

    # ==============================================================
    # SQLITE3.ROW / MAPPING ACCESS
    # ==============================================================

    try:
        return row[key]

    except (
        KeyError,
        IndexError,
        TypeError,
    ):
        return default


# ======================================================================
# DATETIME PARSING
# ======================================================================


def _parse_datetime(
    value: Any,
    *,
    name: str,
) -> datetime:
    """
    Convert datetime or ISO-8601-compatible text to datetime.

    SQLite CURRENT_TIMESTAMP values such as:

        2026-08-30 12:15:42

    are accepted.

    ISO values such as:

        2026-08-30T12:15:42
        2026-08-30T12:15:42+05:30
        2026-08-30T06:45:42Z

    are also accepted.
    """

    if isinstance(
        value,
        datetime,
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        raise TypeError((f"{name} must be datetime or ISO-8601 string."))

    text = value.strip()

    if not (text):
        raise ValueError(f"{name} cannot be empty.")

    # --------------------------------------------------------------
    # Python's fromisoformat historically accepts +00:00 more
    # consistently than a trailing Z, so normalize it explicitly.
    # --------------------------------------------------------------

    if text.endswith(UTC_SUFFIX):
        text = text[:-1] + "+00:00"

    try:
        result = datetime.fromisoformat(text)

    except ValueError as exc:
        raise ValueError((f"{name} is not a valid ISO-8601 datetime.")) from exc

    return result


# ======================================================================
# DATETIME COMPATIBILITY
# ======================================================================


def _is_timezone_aware(
    value: datetime,
) -> bool:
    """
    Return whether one datetime contains effective timezone information.
    """

    return value.tzinfo is not None and value.utcoffset() is not None


def _validate_datetime_compatibility(
    values: Iterable[datetime],
) -> None:
    """
    Reject mixtures of timezone-aware and timezone-naive datetimes.

    Mixing them would make ordering and subtraction ambiguous.
    """

    awareness = {_is_timezone_aware(value) for value in values}

    if len(awareness) > 1:
        raise ValueError(
            (
                "Activity timestamps cannot mix "
                "timezone-aware and timezone-naive "
                "datetime values."
            )
        )


# ======================================================================
# NUMERIC VALIDATION
# ======================================================================


def _require_positive_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a positive Python integer.
    """

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        int,
    ):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")

    return int(value)


def _finite_nonnegative_or_none(
    value: Any,
) -> float | None:
    """
    Convert a value to finite non-negative float.

    None or invalid values return None rather than raising.

    This is useful for optional database-derived fields.
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


def _confidence_or_none(
    value: Any,
) -> float | None:
    """
    Normalize an optional classifier confidence.

    Invalid confidence values are treated as unavailable rather than
    corrupting an otherwise usable activity event.
    """

    result = _finite_nonnegative_or_none(value)

    if result is None:
        return None

    if result > 1.0:
        return None

    return result


# ======================================================================
# EVENT DURATION
# ======================================================================


def _resolve_event_duration_s(
    row: Any,
    *,
    sample_rate: int,
) -> float:
    """
    Resolve one event's duration.

    Priority
    --------
    1. DSP ``duration_s`` when valid.
    2. ``end_sample - start_sample`` reconstruction.
    3. 0.0 when neither representation is available.

    The fallback keeps events usable for event-count analysis even when
    DSP feature extraction failed.
    """

    # ==============================================================
    # DSP DURATION
    # ==============================================================

    duration = _finite_nonnegative_or_none(
        _row_value(
            row,
            "duration_s",
        )
    )

    if duration is not None:
        return duration

    # ==============================================================
    # SAMPLE-INDEX DURATION
    # ==============================================================

    start_sample = _row_value(
        row,
        "start_sample",
    )

    end_sample = _row_value(
        row,
        "end_sample",
    )

    if start_sample is None or end_sample is None:
        return 0.0

    try:
        start_sample = int(start_sample)

        end_sample = int(end_sample)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if start_sample < 0 or end_sample < start_sample:
        return 0.0

    return (end_sample - start_sample) / float(sample_rate)


# ======================================================================
# CLASS LABEL
# ======================================================================


def _normalize_class_label(
    value: Any,
) -> str:
    """
    Normalize one optional broad acoustic class label.
    """

    if value is None:
        return UNCLASSIFIED_LABEL

    label = str(value).strip()

    if not (label):
        return UNCLASSIFIED_LABEL

    return label


# ======================================================================
# EVENT ID
# ======================================================================


def _normalize_event_id(
    value: Any,
) -> int | None:
    """
    Normalize optional database event ID.
    """

    if value is None:
        return None

    try:
        result = int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if result <= 0:
        return None

    return result


# ======================================================================
# NORMALIZE ACTIVITY EVENTS
# ======================================================================


def normalize_activity_events(
    rows: Iterable[Any],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
) -> tuple[
    _ActivityEvent,
    ...,
]:
    """
    Convert database/query rows into deterministic activity events.

    Parameters
    ----------
    rows
        Iterable of dictionary-like or sqlite3.Row-like objects.

    sample_rate
        Used only when event duration must be reconstructed from
        start_sample/end_sample.

    timestamp_key
        Field representing acoustic-event onset time.

        Recommended:
            ``event_time``

        ``created_at`` may be supplied explicitly for diagnostic or
        legacy datasets, but it should not be silently interpreted as
        precise acoustic onset time in research analysis.
    """

    sample_rate = _require_positive_int(
        sample_rate,
        name="sample_rate",
    )

    if not isinstance(
        timestamp_key,
        str,
    ):
        raise TypeError("timestamp_key must be a string.")

    timestamp_key = timestamp_key.strip()

    if not (timestamp_key):
        raise ValueError("timestamp_key cannot be empty.")

    normalized: list[_ActivityEvent] = []

    for row_index, row in enumerate(rows):
        timestamp_value = _row_value(
            row,
            timestamp_key,
        )

        if timestamp_value is None:
            raise ValueError(
                (
                    f"activity row {row_index} "
                    f"is missing required "
                    f"timestamp field "
                    f"{timestamp_key!r}."
                )
            )

        timestamp = _parse_datetime(
            timestamp_value,
            name=(f"row[{row_index}].{timestamp_key}"),
        )

        duration_s = _resolve_event_duration_s(
            row,
            sample_rate=sample_rate,
        )

        class_label = _normalize_class_label(
            _row_value(
                row,
                "classification_label",
            )
        )

        confidence = _confidence_or_none(
            _row_value(
                row,
                "classification_confidence",
            )
        )

        event_id = _normalize_event_id(
            _row_value(
                row,
                "id",
            )
        )

        normalized.append(
            _ActivityEvent(
                event_id=event_id,
                timestamp=timestamp,
                duration_s=duration_s,
                class_label=class_label,
                confidence=confidence,
            )
        )

    # ==============================================================
    # DATETIME MODE CONSISTENCY
    # ==============================================================

    _validate_datetime_compatibility(event.timestamp for event in normalized)

    # ==============================================================
    # DETERMINISTIC ORDER
    # ==============================================================

    normalized.sort(
        key=lambda event: (
            event.timestamp,
            (event.event_id if event.event_id is not None else 0),
        )
    )

    return tuple(normalized)


# ======================================================================
# FLOOR DATETIME TO BUCKET
# ======================================================================


def _floor_datetime(
    value: datetime,
    *,
    bucket_seconds: int,
) -> datetime:
    """
    Floor datetime to a fixed-width bucket relative to Unix epoch.

    Arithmetic is performed with an epoch carrying the same tzinfo as
    the source datetime, avoiding dependence on the host machine's
    local timezone.
    """

    bucket_seconds = _require_positive_int(
        bucket_seconds,
        name="bucket_seconds",
    )

    origin = datetime(
        1970,
        1,
        1,
        tzinfo=value.tzinfo,
    )

    elapsed_seconds = (value - origin).total_seconds()

    bucket_index = math.floor(elapsed_seconds / bucket_seconds)

    return origin + timedelta(seconds=bucket_index * bucket_seconds)


# ======================================================================
# CEIL DATETIME TO BUCKET
# ======================================================================


def _ceil_datetime(
    value: datetime,
    *,
    bucket_seconds: int,
) -> datetime:
    """
    Ceil datetime to the next bucket boundary.

    Values already on a bucket boundary are returned unchanged.
    """

    floored = _floor_datetime(
        value,
        bucket_seconds=bucket_seconds,
    )

    if floored == value:
        return value

    return floored + timedelta(seconds=bucket_seconds)


# ======================================================================
# ANALYSIS RANGE
# ======================================================================


def _resolve_analysis_range(
    events: tuple[
        _ActivityEvent,
        ...,
    ],
    *,
    bucket_seconds: int,
    start: datetime | str | None,
    end: datetime | str | None,
) -> tuple[
    datetime | None,
    datetime | None,
]:
    """
    Determine the activity-analysis interval.

    Explicit boundaries are preserved exactly.

    Otherwise:
        start = floor(first event)
        end   = ceil(latest event end)

    For zero-duration events on an exact boundary, at least one bucket
    is retained.
    """

    start_dt = (
        None
        if start is None
        else _parse_datetime(
            start,
            name="start",
        )
    )

    end_dt = (
        None
        if end is None
        else _parse_datetime(
            end,
            name="end",
        )
    )

    compatibility_values = [event.timestamp for event in events]

    if start_dt is not None:
        compatibility_values.append(start_dt)

    if end_dt is not None:
        compatibility_values.append(end_dt)

    _validate_datetime_compatibility(compatibility_values)

    # ==============================================================
    # EMPTY DATASET
    # ==============================================================

    if not events:
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise ValueError("end cannot be earlier than start.")

        return (
            start_dt,
            end_dt,
        )

    # ==============================================================
    # DERIVED START
    # ==============================================================

    if start_dt is None:
        start_dt = _floor_datetime(
            events[0].timestamp,
            bucket_seconds=bucket_seconds,
        )

    # ==============================================================
    # DERIVED END
    # ==============================================================

    if end_dt is None:
        latest_time = max(
            (event.end_time if event.duration_s > 0.0 else event.timestamp)
            for event in events
        )

        end_dt = _ceil_datetime(
            latest_time,
            bucket_seconds=bucket_seconds,
        )

        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(seconds=bucket_seconds)

    # ==============================================================
    # ORDER
    # ==============================================================

    if end_dt < start_dt:
        raise ValueError("end cannot be earlier than start.")

    return (
        start_dt,
        end_dt,
    )


# ======================================================================
# EVENT IN ANALYSIS WINDOW
# ======================================================================


def _event_overlaps_window(
    event: _ActivityEvent,
    *,
    start: datetime,
    end: datetime,
) -> bool:
    """
    Determine whether an event contributes to a half-open interval:

        [start, end)

    Zero-duration events are included when their timestamp lies inside
    the interval.
    """

    if event.duration_s <= 0.0:
        return start <= event.timestamp < end

    return event.end_time > start and event.timestamp < end


# ======================================================================
# EVENT DURATION OVERLAP
# ======================================================================


def _duration_overlap_s(
    event: _ActivityEvent,
    *,
    start: datetime,
    end: datetime,
) -> float:
    """
    Duration of one event falling inside [start, end).
    """

    if event.duration_s <= 0.0:
        return 0.0

    overlap_start = max(
        event.timestamp,
        start,
    )

    overlap_end = min(
        event.end_time,
        end,
    )

    if overlap_end <= overlap_start:
        return 0.0

    return float((overlap_end - overlap_start).total_seconds())


# ======================================================================
# BUILD ACTIVITY BINS
# ======================================================================


def build_activity_bins(
    rows: Iterable[Any],
    *,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> tuple[
    ActivityBin,
    ...,
]:
    """
    Aggregate acoustic events into fixed temporal bins.

    Event counts
    ------------
    An event is counted in the bin containing its START time.

    Active duration
    ---------------
    Event duration is split across every bin it overlaps.

    This distinction avoids double-counting an event merely because it
    crosses a temporal boundary while still representing acoustic
    activity duration correctly.
    """

    bucket_seconds = _require_positive_int(
        bucket_seconds,
        name="bucket_seconds",
    )

    events = normalize_activity_events(
        rows,
        sample_rate=sample_rate,
        timestamp_key=timestamp_key,
    )

    (
        analysis_start,
        analysis_end,
    ) = _resolve_analysis_range(
        events,
        bucket_seconds=bucket_seconds,
        start=start,
        end=end,
    )

    if analysis_start is None or analysis_end is None or analysis_end <= analysis_start:
        return ()

    bins: list[ActivityBin] = []

    current_start = analysis_start

    while current_start < analysis_end:
        current_end = min(
            (current_start + timedelta(seconds=bucket_seconds)),
            analysis_end,
        )

        # ==========================================================
        # EVENT START COUNT
        # ==========================================================

        starting_events = [
            event
            for event in events
            if (current_start <= event.timestamp < current_end)
        ]

        event_count = len(starting_events)

        # ==========================================================
        # ACTIVE DURATION
        # ==========================================================

        active_duration_s = math.fsum(
            _duration_overlap_s(
                event,
                start=current_start,
                end=current_end,
            )
            for event in events
            if _event_overlaps_window(
                event,
                start=current_start,
                end=current_end,
            )
        )

        # ==========================================================
        # MEAN CLASSIFICATION CONFIDENCE
        # ==========================================================

        confidences = [
            event.confidence
            for event in starting_events
            if (event.confidence is not None)
        ]

        mean_confidence = (
            None
            if not confidences
            else float(math.fsum(confidences) / len(confidences))
        )

        # ==========================================================
        # DOMINANT CLASS
        # ==========================================================

        dominant_class = None

        if starting_events:
            class_counts = Counter(event.class_label for event in starting_events)

            maximum_count = max(class_counts.values())

            # ------------------------------------------------------
            # Deterministic tie-breaking:
            #
            # alphabetical class label.
            # ------------------------------------------------------

            dominant_class = min(
                label
                for label, count in class_counts.items()
                if (count == maximum_count)
            )

        # ==========================================================
        # MODEL
        # ==========================================================

        bins.append(
            ActivityBin(
                bucket_start=current_start,
                bucket_end=current_end,
                event_count=event_count,
                active_duration_s=active_duration_s,
                mean_confidence=mean_confidence,
                dominant_class=dominant_class,
            )
        )

        current_start = current_end

    return tuple(bins)


# ======================================================================
# CLASS ACTIVITY SUMMARIES
# ======================================================================


def build_class_activity_summaries(
    rows: Iterable[Any],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> tuple[
    ClassActivitySummary,
    ...,
]:
    """
    Build per-class acoustic activity statistics.

    Events are assigned to a class according to their primary
    classification label.

    Missing classifications are retained as:

        "unclassified"

    rather than silently discarded.
    """

    events = normalize_activity_events(
        rows,
        sample_rate=sample_rate,
        timestamp_key=timestamp_key,
    )

    # ==============================================================
    # OPTIONAL ANALYSIS WINDOW
    # ==============================================================

    if start is not None or end is not None:
        start_dt = (
            None
            if start is None
            else _parse_datetime(
                start,
                name="start",
            )
        )

        end_dt = (
            None
            if end is None
            else _parse_datetime(
                end,
                name="end",
            )
        )

        compatibility = [event.timestamp for event in events]

        if start_dt is not None:
            compatibility.append(start_dt)

        if end_dt is not None:
            compatibility.append(end_dt)

        _validate_datetime_compatibility(compatibility)

        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise ValueError("end cannot be earlier than start.")

        events = tuple(
            event
            for event in events
            if (
                (start_dt is None or event.timestamp >= start_dt)
                and (end_dt is None or event.timestamp < end_dt)
            )
        )

    total_events = len(events)

    if total_events == 0:
        return ()

    grouped: dict[
        str,
        list[_ActivityEvent],
    ] = defaultdict(list)

    for event in events:
        grouped[event.class_label].append(event)

    summaries: list[ClassActivitySummary] = []

    for class_label in sorted(grouped):
        class_events = grouped[class_label]

        event_count = len(class_events)

        total_duration_s = math.fsum(event.duration_s for event in class_events)

        confidences = [
            event.confidence for event in class_events if (event.confidence is not None)
        ]

        mean_confidence = (
            None
            if not confidences
            else float(math.fsum(confidences) / len(confidences))
        )

        proportion = event_count / total_events

        summaries.append(
            ClassActivitySummary(
                class_label=class_label,
                event_count=event_count,
                total_duration_s=total_duration_s,
                mean_confidence=mean_confidence,
                proportion_of_events=proportion,
            )
        )

    # ==============================================================
    # RESEARCH-FRIENDLY ORDER
    # ==============================================================
    #
    # Highest event count first.
    #
    # Alphabetical tie-break keeps results deterministic.
    # ==============================================================

    summaries.sort(
        key=lambda summary: (
            -summary.event_count,
            summary.class_label,
        )
    )

    return tuple(summaries)


# ======================================================================
# HOURLY ACTIVITY PROFILE
# ======================================================================


def hourly_activity_profile(
    rows: Iterable[Any],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
) -> dict[
    int,
    int,
]:
    """
    Count acoustic-event onsets for each hour of day.

    Returns all 24 hours, including zero-count hours.

    Important
    ---------
    Hour-of-day interpretation follows the timezone already represented
    by ``event_time``.

    Timezone conversion should occur upstream before calling this
    function when local ecological time is required.
    """

    events = normalize_activity_events(
        rows,
        sample_rate=sample_rate,
        timestamp_key=timestamp_key,
    )

    counts = {hour: 0 for hour in range(24)}

    for event in events:
        counts[event.timestamp.hour] += 1

    return counts


# ======================================================================
# PEAK ACTIVITY HOUR
# ======================================================================


def peak_activity_hour(
    rows: Iterable[Any],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
) -> int | None:
    """
    Return the hour containing the highest number of event onsets.

    Deterministic ties
    ------------------
    The earliest hour is selected when multiple hours share the same
    maximum count.

    None is returned for an empty dataset.
    """

    events = normalize_activity_events(
        rows,
        sample_rate=sample_rate,
        timestamp_key=timestamp_key,
    )

    if not (events):
        return None

    counts = Counter(event.timestamp.hour for event in events)

    maximum = max(counts.values())

    return min(hour for hour, count in counts.items() if (count == maximum))


# ======================================================================
# BUILD OVERALL ACTIVITY SUMMARY
# ======================================================================


def build_activity_summary(
    rows: Iterable[Any],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> ActivitySummary:
    """
    Produce the overall temporal acoustic-activity summary.

    ``rows`` is materialized once so generators are safe to pass.
    """

    sample_rate = _require_positive_int(
        sample_rate,
        name="sample_rate",
    )

    materialized_rows = tuple(rows)

    events = normalize_activity_events(
        materialized_rows,
        sample_rate=sample_rate,
        timestamp_key=timestamp_key,
    )

    # ==============================================================
    # EXPLICIT WINDOW
    # ==============================================================

    start_dt = (
        None
        if start is None
        else _parse_datetime(
            start,
            name="start",
        )
    )

    end_dt = (
        None
        if end is None
        else _parse_datetime(
            end,
            name="end",
        )
    )

    compatibility = [event.timestamp for event in events]

    if start_dt is not None:
        compatibility.append(start_dt)

    if end_dt is not None:
        compatibility.append(end_dt)

    _validate_datetime_compatibility(compatibility)

    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise ValueError("end cannot be earlier than start.")

    # ==============================================================
    # FILTER EVENT ONSETS
    # ==============================================================

    selected_events = tuple(
        event
        for event in events
        if (
            (start_dt is None or event.timestamp >= start_dt)
            and (end_dt is None or event.timestamp < end_dt)
        )
    )

    # ==============================================================
    # WINDOW METADATA
    # ==============================================================

    if start_dt is None:
        window_start = selected_events[0].timestamp if selected_events else None

    else:
        window_start = start_dt

    if end_dt is None:
        if selected_events:
            window_end = max(event.end_time for event in selected_events)

        else:
            window_end = None

    else:
        window_end = end_dt

    total_events = len(selected_events)

    window = AnalyticsTimeWindow(
        start=window_start,
        end=window_end,
        event_count=total_events,
    )

    # ==============================================================
    # EMPTY DATASET
    # ==============================================================

    if total_events == 0:
        return ActivitySummary(
            window=window,
            total_events=0,
            total_active_duration_s=0.0,
            mean_event_duration_s=0.0,
            peak_activity_hour=None,
            class_summaries=(),
        )

    # ==============================================================
    # DURATION
    # ==============================================================

    total_active_duration_s = math.fsum(event.duration_s for event in selected_events)

    mean_event_duration_s = total_active_duration_s / total_events

    # ==============================================================
    # PEAK HOUR
    # ==============================================================

    hour_counts = Counter(event.timestamp.hour for event in selected_events)

    maximum_hour_count = max(hour_counts.values())

    busiest_hour = min(
        hour for hour, count in hour_counts.items() if (count == maximum_hour_count)
    )

    # ==============================================================
    # CLASS SUMMARIES
    # ==============================================================

    # Use already-selected normalized data directly rather than
    # re-parsing database rows.

    grouped: dict[
        str,
        list[_ActivityEvent],
    ] = defaultdict(list)

    for event in selected_events:
        grouped[event.class_label].append(event)

    class_summaries: list[ClassActivitySummary] = []

    for class_label in sorted(grouped):
        class_events = grouped[class_label]

        class_count = len(class_events)

        class_duration = math.fsum(event.duration_s for event in class_events)

        confidences = [
            event.confidence for event in class_events if (event.confidence is not None)
        ]

        class_mean_confidence = (
            None
            if not confidences
            else float(math.fsum(confidences) / len(confidences))
        )

        class_summaries.append(
            ClassActivitySummary(
                class_label=class_label,
                event_count=class_count,
                total_duration_s=class_duration,
                mean_confidence=class_mean_confidence,
                proportion_of_events=(class_count / total_events),
            )
        )

    class_summaries.sort(
        key=lambda summary: (
            -summary.event_count,
            summary.class_label,
        )
    )

    # ==============================================================
    # RESULT
    # ==============================================================

    return ActivitySummary(
        window=window,
        total_events=total_events,
        total_active_duration_s=total_active_duration_s,
        mean_event_duration_s=mean_event_duration_s,
        peak_activity_hour=busiest_hour,
        class_summaries=tuple(class_summaries),
    )


# ======================================================================
# ACTIVITY RATE
# ======================================================================


def calculate_event_rate_per_hour(
    event_count: int,
    *,
    window_duration_s: float,
) -> float:
    """
    Calculate acoustic-event rate in events/hour.

    Returns 0 for a zero-duration window only when event_count is also
    zero. A non-zero event count with zero observation duration is
    logically invalid.
    """

    if isinstance(
        event_count,
        bool,
    ) or not isinstance(
        event_count,
        int,
    ):
        raise TypeError("event_count must be an integer.")

    if event_count < 0:
        raise ValueError("event_count cannot be negative.")

    try:
        window_duration_s = float(window_duration_s)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(("window_duration_s must be numeric.")) from exc

    if not math.isfinite(window_duration_s) or window_duration_s < 0.0:
        raise ValueError(("window_duration_s must be finite and non-negative."))

    if window_duration_s == 0.0:
        if event_count == 0:
            return 0.0

        raise ValueError(
            ("A positive event_count cannot have zero observation duration.")
        )

    return event_count / (window_duration_s / 3600.0)
