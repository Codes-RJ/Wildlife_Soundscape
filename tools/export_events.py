"""
Acoustic-event dataset exporter.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Export persisted acoustic events and their associated research metadata
from SQLite into portable datasets.

Supported formats:

    CSV
    JSON
    JSONL


Exported information
--------------------
Each exported event can contain:

    session metadata
    scientifically reconstructed event timestamp
    detector metadata
    trigger nodes
    environmental context
    localization result
    DSP features
    MFCC vectors
    classification result
    classifier identity/version
    event-audio directory
    database insertion timestamps


Scientific timestamp policy
---------------------------
events.created_at is NOT treated as the acoustic observation time.

The event observation time is reconstructed from:

    sessions.started_at
        +
    events.start_sample / sample_rate

This preserves the shared-clock sample timeline as the timing authority.


Read-only policy
----------------
This tool never writes to the project SQLite database.

The database is opened using SQLite read-only mode whenever possible.


Recommended execution
---------------------
From the repository root:

    python -m tools.export_events

Examples:

    python -m tools.export_events --format csv

    python -m tools.export_events --format json

    python -m tools.export_events --session-id 123456

    python -m tools.export_events --output data/exports/my_events.csv
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import argparse
import csv
import json
import math
import sqlite3

from dataclasses import (
    dataclass,
)

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from enum import (
    Enum,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
)


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================


from config import (
    CONFIG,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_SAMPLE_RATE = (
    48_000
)


DEFAULT_EXPORT_DIRECTORY = (
    Path(
        "data/exports"
    )
)


DEFAULT_DATABASE_PATH = (
    Path(
        "data/database/events.db"
    )
)


# ======================================================================
# EXPORT FORMAT
# ======================================================================


class ExportFormat(
    str,
    Enum,
):
    """
    Supported event-dataset output formats.
    """

    CSV = (
        "csv"
    )

    JSON = (
        "json"
    )

    JSONL = (
        "jsonl"
    )


# ======================================================================
# EXPORT OPTIONS
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class EventExportOptions:
    """
    Event-export query and formatting options.
    """

    database_path: Path

    output_path: Path

    export_format: ExportFormat

    sample_rate: int

    session_id: int | None = (
        None
    )

    limit: int | None = (
        None
    )

    include_mfcc: bool = (
        True
    )

    pretty_json: bool = (
        True
    )

    def __post_init__(
        self,
    ) -> None:

        # ==============================================================
        # PATHS
        # ==============================================================

        if not isinstance(
            self.database_path,
            Path,
        ):

            raise TypeError(
                "database_path must be pathlib.Path."
            )

        if not isinstance(
            self.output_path,
            Path,
        ):

            raise TypeError(
                "output_path must be pathlib.Path."
            )

        # ==============================================================
        # FORMAT
        # ==============================================================

        if not isinstance(
            self.export_format,
            ExportFormat,
        ):

            raise TypeError(
                "export_format must be an ExportFormat."
            )

        # ==============================================================
        # SAMPLE RATE
        # ==============================================================

        if (
            isinstance(
                self.sample_rate,
                bool,
            )
            or not isinstance(
                self.sample_rate,
                int,
            )
        ):

            raise TypeError(
                "sample_rate must be an integer."
            )

        if (
            self.sample_rate
            <= 0
        ):

            raise ValueError(
                "sample_rate must be greater than 0."
            )

        # ==============================================================
        # SESSION
        # ==============================================================

        if (
            self.session_id
            is not None
        ):

            if (
                isinstance(
                    self.session_id,
                    bool,
                )
                or not isinstance(
                    self.session_id,
                    int,
                )
            ):

                raise TypeError(
                    "session_id must be an integer or None."
                )

            if (
                self.session_id
                <= 0
            ):

                raise ValueError(
                    "session_id must be greater than 0."
                )

        # ==============================================================
        # LIMIT
        # ==============================================================

        if (
            self.limit
            is not None
        ):

            if (
                isinstance(
                    self.limit,
                    bool,
                )
                or not isinstance(
                    self.limit,
                    int,
                )
            ):

                raise TypeError(
                    "limit must be an integer or None."
                )

            if (
                self.limit
                <= 0
            ):

                raise ValueError(
                    "limit must be greater than 0."
                )

        # ==============================================================
        # FLAGS
        # ==============================================================

        if not isinstance(
            self.include_mfcc,
            bool,
        ):

            raise TypeError(
                "include_mfcc must be bool."
            )

        if not isinstance(
            self.pretty_json,
            bool,
        ):

            raise TypeError(
                "pretty_json must be bool."
            )


# ======================================================================
# JSON FIELDS
# ======================================================================


JSON_COLUMNS = frozenset(
    {
        "trigger_nodes",
        "mfcc_mean_json",
        "mfcc_std_json",
        "classification_scores_json",
        "classification_reasons_json",
    }
)


# ======================================================================
# CSV FIELD ORDER
# ======================================================================


CSV_FIELD_ORDER = (
    # ------------------------------------------------------------------
    # DATABASE / SESSION IDENTITY
    # ------------------------------------------------------------------

    "event_id",
    "detector_event_id",
    "session_id",
    "session_label",

    # ------------------------------------------------------------------
    # SCIENTIFIC TIME
    # ------------------------------------------------------------------

    "event_time",
    "session_started_at",
    "session_stopped_at",

    # ------------------------------------------------------------------
    # SAMPLE TIMELINE
    # ------------------------------------------------------------------

    "start_sample",
    "end_sample",
    "duration_samples",
    "duration_s",

    # ------------------------------------------------------------------
    # DETECTION
    # ------------------------------------------------------------------

    "trigger_nodes",
    "trigger_node_count",
    "peak_rms_dbfs",
    "best_node_id",

    # ------------------------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------------------------

    "temperature_c",
    "humidity_percent",
    "pressure_hpa",
    "speed_of_sound_mps",

    # ------------------------------------------------------------------
    # LOCALIZATION
    # ------------------------------------------------------------------

    "x_m",
    "y_m",
    "localization_success",
    "localization_residual_m",

    # ------------------------------------------------------------------
    # DSP
    # ------------------------------------------------------------------

    "feature_source_node_id",
    "feature_duration_s",
    "rms",
    "peak_amplitude",
    "crest_factor",
    "zero_crossing_rate",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
    "spectral_flux",
    "snr_db",

    # ------------------------------------------------------------------
    # MFCC
    # ------------------------------------------------------------------

    "mfcc_mean",
    "mfcc_std",

    # ------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------

    "classification_label",
    "classification_confidence",
    "classification_second_label",
    "classification_second_confidence",
    "classification_margin",
    "classification_scores",
    "classification_reasons",
    "classifier_name",
    "classifier_version",

    # ------------------------------------------------------------------
    # TRACEABILITY
    # ------------------------------------------------------------------

    "event_directory",
    "database_created_at",
)


# ======================================================================
# READ-ONLY DATABASE CONNECTION
# ======================================================================


def open_readonly_database(
    path: Path,
) -> sqlite3.Connection:
    """
    Open SQLite database in read-only mode.

    No schema initialization or write operation is performed.
    """

    if not isinstance(
        path,
        Path,
    ):

        raise TypeError(
            "path must be pathlib.Path."
        )

    path = (
        path.expanduser()
        .resolve()
    )

    if not (
        path.exists()
    ):

        raise FileNotFoundError(
            f"Database does not exist: {path}"
        )

    if not (
        path.is_file()
    ):

        raise ValueError(
            f"Database path is not a file: {path}"
        )

    uri = (
        f"{path.as_uri()}?mode=ro"
    )

    connection = sqlite3.connect(
        uri,
        uri=True,
        timeout=5.0,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    connection.execute(
        "PRAGMA query_only = ON"
    )

    connection.execute(
        "PRAGMA busy_timeout = 5000"
    )

    return (
        connection
    )


# ======================================================================
# REQUIRED TABLE VALIDATION
# ======================================================================


def validate_database_schema(
    connection: sqlite3.Connection,
) -> None:
    """
    Validate that the database contains the tables required by the
    event exporter.
    """

    required_tables = {
        "sessions",
        "events",
        "event_features",
        "classifications",
    }

    rows = (
        connection.execute(
            """
            SELECT name

            FROM sqlite_master

            WHERE type = 'table'
            """
        )
        .fetchall()
    )

    available_tables = {
        str(
            row[
                "name"
            ]
        )
        for row
        in rows
    }

    missing = (
        required_tables
        - available_tables
    )

    if (
        missing
    ):

        raise RuntimeError(
            (
                "Database is missing required "
                f"table(s): {sorted(missing)}"
            )
        )


# ======================================================================
# EVENT SELECT QUERY
# ======================================================================


def event_export_query() -> str:
    """
    Return read-only query joining all persisted event research data.

    Field aliases deliberately avoid ambiguous names such as multiple
    `created_at` columns.
    """

    return """
        SELECT

            ------------------------------------------------------------
            -- EVENT IDENTITY
            ------------------------------------------------------------

            e.id
                AS event_id,

            e.detector_event_id
                AS detector_event_id,

            e.session_id
                AS session_id,

            s.label
                AS session_label,

            s.started_at
                AS session_started_at,

            s.stopped_at
                AS session_stopped_at,


            ------------------------------------------------------------
            -- SAMPLE TIMELINE
            ------------------------------------------------------------

            e.start_sample
                AS start_sample,

            e.end_sample
                AS end_sample,


            ------------------------------------------------------------
            -- DETECTION
            ------------------------------------------------------------

            e.trigger_nodes
                AS trigger_nodes,

            e.peak_rms_dbfs
                AS peak_rms_dbfs,

            e.best_node_id
                AS best_node_id,


            ------------------------------------------------------------
            -- ENVIRONMENT
            ------------------------------------------------------------

            e.temperature_c
                AS temperature_c,

            e.humidity_percent
                AS humidity_percent,

            e.pressure_hpa
                AS pressure_hpa,

            e.speed_of_sound_mps
                AS speed_of_sound_mps,


            ------------------------------------------------------------
            -- LOCALIZATION
            ------------------------------------------------------------

            e.x_m
                AS x_m,

            e.y_m
                AS y_m,

            e.localization_success
                AS localization_success,

            e.localization_residual_m
                AS localization_residual_m,


            ------------------------------------------------------------
            -- FILESYSTEM
            ------------------------------------------------------------

            e.event_directory
                AS event_directory,

            e.created_at
                AS database_created_at,


            ------------------------------------------------------------
            -- DSP
            ------------------------------------------------------------

            f.source_node_id
                AS feature_source_node_id,

            f.duration_s
                AS feature_duration_s,

            f.rms
                AS rms,

            f.peak_amplitude
                AS peak_amplitude,

            f.crest_factor
                AS crest_factor,

            f.zero_crossing_rate
                AS zero_crossing_rate,

            f.dominant_frequency_hz
                AS dominant_frequency_hz,

            f.spectral_centroid_hz
                AS spectral_centroid_hz,

            f.spectral_bandwidth_hz
                AS spectral_bandwidth_hz,

            f.spectral_rolloff_hz
                AS spectral_rolloff_hz,

            f.spectral_flatness
                AS spectral_flatness,

            f.spectral_flux
                AS spectral_flux,

            f.snr_db
                AS snr_db,

            f.mfcc_mean_json
                AS mfcc_mean_json,

            f.mfcc_std_json
                AS mfcc_std_json,


            ------------------------------------------------------------
            -- CLASSIFICATION
            ------------------------------------------------------------

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

            ON s.session_id =
                e.session_id


        LEFT JOIN event_features AS f

            ON f.event_id =
                e.id


        LEFT JOIN classifications AS c

            ON c.event_id =
                e.id
    """


# ======================================================================
# SQLITE TIMESTAMP PARSING
# ======================================================================


def parse_database_timestamp(
    value: Any,
) -> datetime | None:
    """
    Parse SQLite timestamp as UTC.

    SQLite CURRENT_TIMESTAMP normally produces:

        YYYY-MM-DD HH:MM:SS

    which does not contain an explicit timezone marker.

    Since SQLite CURRENT_TIMESTAMP is UTC, naive timestamps from this
    project are interpreted as UTC.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    if not isinstance(
        value,
        str,
    ):

        value = str(
            value
        )

    text = (
        value.strip()
    )

    if not (
        text
    ):

        return (
            None
        )

    if (
        text.endswith(
            "Z"
        )
    ):

        text = (
            text[
                :-1
            ]
            + "+00:00"
        )

    try:

        result = datetime.fromisoformat(
            text
        )

    except ValueError as exc:

        raise ValueError(
            (
                "Unable to parse database "
                f"timestamp: {value!r}"
            )
        ) from exc

    if (
        result.tzinfo
        is None
    ):

        result = result.replace(
            tzinfo=timezone.utc
        )

    else:

        result = result.astimezone(
            timezone.utc
        )

    return (
        result
    )


# ======================================================================
# EVENT TIMESTAMP
# ======================================================================


def reconstruct_event_time(
    session_started_at: Any,
    start_sample: int,
    *,
    sample_rate: int,
) -> str:
    """
    Reconstruct scientifically meaningful event time.

    event_time
        =
    session_started_at
        +
    start_sample / sample_rate
    """

    started = (
        parse_database_timestamp(
            session_started_at
        )
    )

    if (
        started
        is None
    ):

        raise ValueError(
            (
                "Session started_at is required "
                "to reconstruct event_time."
            )
        )

    if (
        isinstance(
            start_sample,
            bool,
        )
        or not isinstance(
            start_sample,
            int,
        )
    ):

        raise TypeError(
            "start_sample must be an integer."
        )

    if (
        start_sample
        < 0
    ):

        raise ValueError(
            "start_sample cannot be negative."
        )

    if (
        isinstance(
            sample_rate,
            bool,
        )
        or not isinstance(
            sample_rate,
            int,
        )
    ):

        raise TypeError(
            "sample_rate must be an integer."
        )

    if (
        sample_rate
        <= 0
    ):

        raise ValueError(
            "sample_rate must be greater than 0."
        )

    event_time = (
        started
        + timedelta(
            seconds=
                (
                    start_sample
                    / sample_rate
                )
        )
    )

    return (
        event_time.isoformat()
    )


# ======================================================================
# JSON DECODING
# ======================================================================


def decode_json_value(
    value: Any,
    *,
    field_name: str,
) -> Any:
    """
    Safely decode one persisted JSON field.

    None remains None.

    Malformed stored JSON raises an explicit error because silently
    exporting corrupt scientific metadata would reduce traceability.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    if not isinstance(
        value,
        str,
    ):

        raise TypeError(
            (
                f"{field_name} must contain "
                "stored JSON text."
            )
        )

    try:

        return json.loads(
            value
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            (
                "Invalid stored JSON in "
                f"{field_name}."
            )
        ) from exc


# ======================================================================
# BOOLEAN CONVERSION
# ======================================================================


def sqlite_boolean(
    value: Any,
) -> bool | None:
    """
    Convert nullable SQLite INTEGER boolean into Python bool.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    numeric = int(
        value
    )

    if (
        numeric
        not in (
            0,
            1,
        )
    ):

        raise ValueError(
            (
                "SQLite boolean value must "
                f"be 0, 1 or NULL; got {value!r}."
            )
        )

    return bool(
        numeric
    )


# ======================================================================
# FINITE NUMBER NORMALIZATION
# ======================================================================


def normalize_optional_number(
    value: Any,
    *,
    field_name: str,
) -> float | None:
    """
    Normalize optional SQLite numeric field.

    Non-finite research values are rejected.
    """

    if (
        value
        is None
    ):

        return (
            None
        )

    result = float(
        value
    )

    if not math.isfinite(
        result
    ):

        raise ValueError(
            (
                f"{field_name} contains "
                "a non-finite value."
            )
        )

    return (
        result
    )


# ======================================================================
# NORMALIZE DATABASE ROW
# ======================================================================


def normalize_event_row(
    row: Mapping[
        str,
        Any,
    ],
    *,
    sample_rate: int,
    include_mfcc: bool,
) -> dict[
    str,
    Any,
]:
    """
    Convert one joined SQLite event row into an exportable research row.
    """

    # ==================================================================
    # SAMPLE TIMELINE
    # ==================================================================

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

    if (
        start_sample
        < 0
    ):

        raise ValueError(
            "Event start_sample cannot be negative."
        )

    if (
        end_sample
        < start_sample
    ):

        raise ValueError(
            (
                "Event end_sample cannot be "
                "less than start_sample."
            )
        )

    duration_samples = (
        end_sample
        - start_sample
    )

    duration_s = (
        duration_samples
        / sample_rate
    )

    # ==================================================================
    # SCIENTIFIC TIMESTAMP
    # ==================================================================

    event_time = (
        reconstruct_event_time(
            row[
                "session_started_at"
            ],
            start_sample,
            sample_rate=
                sample_rate,
        )
    )

    # ==================================================================
    # TRIGGER NODES
    # ==================================================================

    trigger_nodes = (
        decode_json_value(
            row[
                "trigger_nodes"
            ],
            field_name=
                "trigger_nodes",
        )
    )

    if not isinstance(
        trigger_nodes,
        list,
    ):

        raise ValueError(
            (
                "Stored trigger_nodes must "
                "decode to a list."
            )
        )

    normalized_trigger_nodes = [
        int(
            node_id
        )
        for node_id
        in trigger_nodes
    ]

    # ==================================================================
    # MFCC
    # ==================================================================

    if (
        include_mfcc
    ):

        mfcc_mean = (
            decode_json_value(
                row[
                    "mfcc_mean_json"
                ],
                field_name=
                    "mfcc_mean_json",
            )
        )

        mfcc_std = (
            decode_json_value(
                row[
                    "mfcc_std_json"
                ],
                field_name=
                    "mfcc_std_json",
            )
        )

    else:

        mfcc_mean = (
            None
        )

        mfcc_std = (
            None
        )

    # ==================================================================
    # CLASSIFICATION EXPLAINABILITY
    # ==================================================================

    classification_scores = (
        decode_json_value(
            row[
                "classification_scores_json"
            ],
            field_name=
                "classification_scores_json",
        )
    )

    classification_reasons = (
        decode_json_value(
            row[
                "classification_reasons_json"
            ],
            field_name=
                "classification_reasons_json",
        )
    )

    # ==================================================================
    # OUTPUT
    # ==================================================================

    return {
        # --------------------------------------------------------------
        # IDENTITY
        # --------------------------------------------------------------

        "event_id":
            int(
                row[
                    "event_id"
                ]
            ),

        "detector_event_id":
            int(
                row[
                    "detector_event_id"
                ]
            ),

        "session_id":
            int(
                row[
                    "session_id"
                ]
            ),

        "session_label":
            row[
                "session_label"
            ],

        # --------------------------------------------------------------
        # TIME
        # --------------------------------------------------------------

        "event_time":
            event_time,

        "session_started_at":
            row[
                "session_started_at"
            ],

        "session_stopped_at":
            row[
                "session_stopped_at"
            ],

        # --------------------------------------------------------------
        # SHARED SAMPLE TIMELINE
        # --------------------------------------------------------------

        "start_sample":
            start_sample,

        "end_sample":
            end_sample,

        "duration_samples":
            duration_samples,

        "duration_s":
            duration_s,

        # --------------------------------------------------------------
        # DETECTOR
        # --------------------------------------------------------------

        "trigger_nodes":
            normalized_trigger_nodes,

        "trigger_node_count":
            len(
                normalized_trigger_nodes
            ),

        "peak_rms_dbfs":
            normalize_optional_number(
                row[
                    "peak_rms_dbfs"
                ],
                field_name=
                    "peak_rms_dbfs",
            ),

        "best_node_id":
            (
                None

                if row[
                    "best_node_id"
                ] is None

                else int(
                    row[
                        "best_node_id"
                    ]
                )
            ),

        # --------------------------------------------------------------
        # ENVIRONMENT
        # --------------------------------------------------------------

        "temperature_c":
            normalize_optional_number(
                row[
                    "temperature_c"
                ],
                field_name=
                    "temperature_c",
            ),

        "humidity_percent":
            normalize_optional_number(
                row[
                    "humidity_percent"
                ],
                field_name=
                    "humidity_percent",
            ),

        "pressure_hpa":
            normalize_optional_number(
                row[
                    "pressure_hpa"
                ],
                field_name=
                    "pressure_hpa",
            ),

        "speed_of_sound_mps":
            normalize_optional_number(
                row[
                    "speed_of_sound_mps"
                ],
                field_name=
                    "speed_of_sound_mps",
            ),

        # --------------------------------------------------------------
        # LOCALIZATION
        # --------------------------------------------------------------

        "x_m":
            normalize_optional_number(
                row[
                    "x_m"
                ],
                field_name=
                    "x_m",
            ),

        "y_m":
            normalize_optional_number(
                row[
                    "y_m"
                ],
                field_name=
                    "y_m",
            ),

        "localization_success":
            sqlite_boolean(
                row[
                    "localization_success"
                ]
            ),

        "localization_residual_m":
            normalize_optional_number(
                row[
                    "localization_residual_m"
                ],
                field_name=
                    "localization_residual_m",
            ),

        # --------------------------------------------------------------
        # DSP
        # --------------------------------------------------------------

        "feature_source_node_id":
            (
                None

                if row[
                    "feature_source_node_id"
                ] is None

                else int(
                    row[
                        "feature_source_node_id"
                    ]
                )
            ),

        "feature_duration_s":
            normalize_optional_number(
                row[
                    "feature_duration_s"
                ],
                field_name=
                    "feature_duration_s",
            ),

        "rms":
            normalize_optional_number(
                row[
                    "rms"
                ],
                field_name=
                    "rms",
            ),

        "peak_amplitude":
            normalize_optional_number(
                row[
                    "peak_amplitude"
                ],
                field_name=
                    "peak_amplitude",
            ),

        "crest_factor":
            normalize_optional_number(
                row[
                    "crest_factor"
                ],
                field_name=
                    "crest_factor",
            ),

        "zero_crossing_rate":
            normalize_optional_number(
                row[
                    "zero_crossing_rate"
                ],
                field_name=
                    "zero_crossing_rate",
            ),

        "dominant_frequency_hz":
            normalize_optional_number(
                row[
                    "dominant_frequency_hz"
                ],
                field_name=
                    "dominant_frequency_hz",
            ),

        "spectral_centroid_hz":
            normalize_optional_number(
                row[
                    "spectral_centroid_hz"
                ],
                field_name=
                    "spectral_centroid_hz",
            ),

        "spectral_bandwidth_hz":
            normalize_optional_number(
                row[
                    "spectral_bandwidth_hz"
                ],
                field_name=
                    "spectral_bandwidth_hz",
            ),

        "spectral_rolloff_hz":
            normalize_optional_number(
                row[
                    "spectral_rolloff_hz"
                ],
                field_name=
                    "spectral_rolloff_hz",
            ),

        "spectral_flatness":
            normalize_optional_number(
                row[
                    "spectral_flatness"
                ],
                field_name=
                    "spectral_flatness",
            ),

        "spectral_flux":
            normalize_optional_number(
                row[
                    "spectral_flux"
                ],
                field_name=
                    "spectral_flux",
            ),

        "snr_db":
            normalize_optional_number(
                row[
                    "snr_db"
                ],
                field_name=
                    "snr_db",
            ),

        # --------------------------------------------------------------
        # MFCC
        # --------------------------------------------------------------

        "mfcc_mean":
            mfcc_mean,

        "mfcc_std":
            mfcc_std,

        # --------------------------------------------------------------
        # CLASSIFICATION
        # --------------------------------------------------------------

        "classification_label":
            row[
                "classification_label"
            ],

        "classification_confidence":
            normalize_optional_number(
                row[
                    "classification_confidence"
                ],
                field_name=
                    "classification_confidence",
            ),

        "classification_second_label":
            row[
                "classification_second_label"
            ],

        "classification_second_confidence":
            normalize_optional_number(
                row[
                    "classification_second_confidence"
                ],
                field_name=
                    (
                        "classification_second_confidence"
                    ),
            ),

        "classification_margin":
            normalize_optional_number(
                row[
                    "classification_margin"
                ],
                field_name=
                    "classification_margin",
            ),

        "classification_scores":
            classification_scores,

        "classification_reasons":
            classification_reasons,

        "classifier_name":
            row[
                "classifier_name"
            ],

        "classifier_version":
            row[
                "classifier_version"
            ],

        # --------------------------------------------------------------
        # TRACEABILITY
        # --------------------------------------------------------------

        "event_directory":
            row[
                "event_directory"
            ],

        "database_created_at":
            row[
                "database_created_at"
            ],
    }


# ======================================================================
# LOAD EVENT DATASET
# ======================================================================


def load_event_rows(
    database_path: Path,
    *,
    sample_rate: int,
    session_id: int | None = None,
    limit: int | None = None,
    include_mfcc: bool = True,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Read and normalize persisted events.

    Results are ordered by:

        session start
        event sampleIndex
        database event ID
    """

    if (
        isinstance(
            sample_rate,
            bool,
        )
        or not isinstance(
            sample_rate,
            int,
        )
    ):

        raise TypeError(
            "sample_rate must be an integer."
        )

    if (
        sample_rate
        <= 0
    ):

        raise ValueError(
            "sample_rate must be greater than 0."
        )

    query = (
        event_export_query()
    )

    parameters: list[
        Any
    ] = []

    # ==================================================================
    # SESSION FILTER
    # ==================================================================

    if (
        session_id
        is not None
    ):

        if (
            isinstance(
                session_id,
                bool,
            )
            or not isinstance(
                session_id,
                int,
            )
        ):

            raise TypeError(
                "session_id must be an integer."
            )

        if (
            session_id
            <= 0
        ):

            raise ValueError(
                "session_id must be greater than 0."
            )

        query += (
            """

            WHERE e.session_id = ?
            """
        )

        parameters.append(
            session_id
        )

    # ==================================================================
    # ORDER
    # ==================================================================

    query += (
        """

        ORDER BY

            s.started_at ASC,

            e.start_sample ASC,

            e.id ASC
        """
    )

    # ==================================================================
    # LIMIT
    # ==================================================================

    if (
        limit
        is not None
    ):

        if (
            isinstance(
                limit,
                bool,
            )
            or not isinstance(
                limit,
                int,
            )
        ):

            raise TypeError(
                "limit must be an integer."
            )

        if (
            limit
            <= 0
        ):

            raise ValueError(
                "limit must be greater than 0."
            )

        query += (
            """

            LIMIT ?
            """
        )

        parameters.append(
            limit
        )

    # ==================================================================
    # EXECUTE READ-ONLY QUERY
    # ==================================================================

    with open_readonly_database(
        database_path
    ) as connection:

        validate_database_schema(
            connection
        )

        database_rows = (
            connection.execute(
                query,
                tuple(
                    parameters
                ),
            )
            .fetchall()
        )

    return [
        normalize_event_row(
            row,
            sample_rate=
                sample_rate,
            include_mfcc=
                include_mfcc,
        )
        for row
        in database_rows
    ]


# ======================================================================
# CSV VALUE
# ======================================================================


def csv_value(
    value: Any,
) -> Any:
    """
    Convert compound values into deterministic JSON strings for CSV.

    Scalar values remain native.
    """

    if isinstance(
        value,
        (
            list,
            tuple,
            dict,
        ),
    ):

        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(
                ",",
                ":",
            ),
            sort_keys=
                isinstance(
                    value,
                    dict,
                ),
        )

    if isinstance(
        value,
        bool,
    ):

        return (
            1
            if value
            else 0
        )

    return (
        value
    )


# ======================================================================
# WRITE CSV
# ======================================================================


def write_csv(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    output_path: Path,
) -> None:
    """
    Write event rows as UTF-8 CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=
                CSV_FIELD_ORDER,
            extrasaction=
                "ignore",
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    field:
                        csv_value(
                            row.get(
                                field
                            )
                        )

                    for field
                    in CSV_FIELD_ORDER
                }
            )


# ======================================================================
# WRITE JSON
# ======================================================================


def write_json(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    output_path: Path,
    *,
    pretty: bool = True,
) -> None:
    """
    Write event rows as one JSON array.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            list(
                rows
            ),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=
                (
                    2
                    if pretty
                    else None
                ),
            separators=
                (
                    None
                    if pretty
                    else (
                        ",",
                        ":",
                    )
                ),
        )

        handle.write(
            "\n"
        )


# ======================================================================
# WRITE JSONL
# ======================================================================


def write_jsonl(
    rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
    output_path: Path,
) -> None:
    """
    Write one JSON object per line.

    JSONL is useful for large datasets and ML/data-processing pipelines.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for row in rows:

            serialized = json.dumps(
                dict(
                    row
                ),
                ensure_ascii=False,
                allow_nan=False,
                separators=(
                    ",",
                    ":",
                ),
            )

            handle.write(
                serialized
            )

            handle.write(
                "\n"
            )


# ======================================================================
# EXPORT
# ======================================================================


def export_events(
    options: EventExportOptions,
) -> int:
    """
    Export events according to supplied options.

    Returns
    -------
    int
        Number of exported events.
    """

    if not isinstance(
        options,
        EventExportOptions,
    ):

        raise TypeError(
            "options must be EventExportOptions."
        )

    rows = load_event_rows(
        options.database_path,
        sample_rate=
            options.sample_rate,
        session_id=
            options.session_id,
        limit=
            options.limit,
        include_mfcc=
            options.include_mfcc,
    )

    if (
        options.export_format
        is ExportFormat.CSV
    ):

        write_csv(
            rows,
            options.output_path,
        )

    elif (
        options.export_format
        is ExportFormat.JSON
    ):

        write_json(
            rows,
            options.output_path,
            pretty=
                options.pretty_json,
        )

    elif (
        options.export_format
        is ExportFormat.JSONL
    ):

        write_jsonl(
            rows,
            options.output_path,
        )

    else:

        raise ValueError(
            (
                "Unsupported export format: "
                f"{options.export_format}"
            )
        )

    return (
        len(
            rows
        )
    )


# ======================================================================
# DEFAULT OUTPUT PATH
# ======================================================================


def default_output_path(
    export_format: ExportFormat,
    *,
    session_id: int | None,
) -> Path:
    """
    Build deterministic default export filename.
    """

    if (
        session_id
        is None
    ):

        stem = (
            "acoustic_events"
        )

    else:

        stem = (
            f"acoustic_events_session_{session_id}"
        )

    return (
        DEFAULT_EXPORT_DIRECTORY
        / f"{stem}.{export_format.value}"
    )


# ======================================================================
# ARGUMENT PARSER
# ======================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Export Wildlife Soundscape acoustic events "
            "from SQLite into CSV, JSON or JSONL."
        )
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=
            CONFIG.persistence.database_path,
        help=(
            "SQLite database path. "
            "Default: configured persistence database."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output dataset path. "
            "If omitted, data/exports is used."
        ),
    )

    parser.add_argument(
        "--format",
        dest="export_format",
        choices=[
            item.value
            for item
            in ExportFormat
        ],
        default=
            ExportFormat.CSV.value,
        help=(
            "Output format: csv, json or jsonl."
        ),
    )

    parser.add_argument(
        "--session-id",
        type=int,
        default=None,
        help=(
            "Export only one acquisition session."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of events."
        ),
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=
            CONFIG.audio.sample_rate,
        help=(
            "Acquisition sample rate used for "
            "event timestamp reconstruction."
        ),
    )

    parser.add_argument(
        "--exclude-mfcc",
        action="store_true",
        help=(
            "Do not include MFCC vectors in the exported dataset."
        ),
    )

    parser.add_argument(
        "--compact-json",
        action="store_true",
        help=(
            "Disable pretty indentation for JSON-array output."
        ),
    )

    return (
        parser
    )


# ======================================================================
# MAIN
# ======================================================================


def main() -> int:
    """
    Command-line entry point.
    """

    parser = (
        build_argument_parser()
    )

    arguments = (
        parser.parse_args()
    )

    export_format = ExportFormat(
        arguments.export_format
    )

    output_path = (
        arguments.output
    )

    if (
        output_path
        is None
    ):

        output_path = (
            default_output_path(
                export_format,
                session_id=
                    arguments.session_id,
            )
        )

    # ==================================================================
    # EXTENSION CHECK
    # ==================================================================

    expected_suffix = (
        f".{export_format.value}"
    )

    if (
        output_path.suffix.lower()
        != expected_suffix
    ):

        parser.error(
            (
                f"--output extension must be "
                f"{expected_suffix} for "
                f"--format {export_format.value}."
            )
        )

    # ==================================================================
    # OPTIONS
    # ==================================================================

    try:

        options = EventExportOptions(
            database_path=
                arguments.database,

            output_path=
                output_path,

            export_format=
                export_format,

            sample_rate=
                arguments.sample_rate,

            session_id=
                arguments.session_id,

            limit=
                arguments.limit,

            include_mfcc=
                not arguments.exclude_mfcc,

            pretty_json=
                not arguments.compact_json,
        )

        exported_count = (
            export_events(
                options
            )
        )

    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.DatabaseError,
    ) as exc:

        parser.exit(
            status=1,
            message=
                f"Export failed: {exc}\n",
        )

    # ==================================================================
    # STATUS
    # ==================================================================

    print(
        (
            f"Exported {exported_count} event(s) "
            f"to {options.output_path}"
        )
    )

    if (
        options.session_id
        is not None
    ):

        print(
            (
                "Session filter: "
                f"{options.session_id}"
            )
        )

    print(
        (
            "Event timestamp authority: "
            "session.started_at + start_sample/sample_rate"
        )
    )

    return (
        0
    )


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================


if (
    __name__
    == "__main__"
):

    raise SystemExit(
        main()
    )