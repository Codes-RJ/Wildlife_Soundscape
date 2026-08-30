"""
Research-analytics exporter.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Generate portable research-output files from the higher-level
``analytics`` package.

This tool differs from:

    tools/export_events.py

which exports individual persisted acoustic-event records.

This module exports derived research information such as:

    temporal activity summaries
    class activity distributions
    environmental associations
    spatial occupancy
    hotspots
    spatial transitions
    behavior-related indicators
    analysis warnings


Primary outputs
---------------
A research export contains:

    <name>.json
        Complete structured analytics report.

    <name>_summary.csv
        Flattened scalar metrics.

    <name>_tables/
        CSV tables extracted from list-based sections of the report.


Serialization policy
--------------------
The analytics model ``to_dict()`` interface is the authoritative public
serialization contract.

This is important because analytics dataclasses may expose calculated
research values through ``to_dict()`` that are not physical dataclass
fields.

Examples include calculated spatial-cell properties such as:

    center_x_m
    center_y_m
    area_m2

Therefore ``to_dict()`` is deliberately evaluated BEFORE generic
dataclass serialization.


Research-session policy
-----------------------
Database-mode research export always analyzes one acquisition session.

External event datasets are also checked for obvious multi-session
mixing whenever they contain ``session_id``.

Combining separate sessions into one behavioral/spatial sequence could:

    create false spatial transitions

    join discontinuous observation periods

    misrepresent downtime as part of one research timeline

Cross-session research requires explicit exposure-aware aggregation of
independent session-level results.


Scientific interpretation
-------------------------
Exported environmental relationships are statistical associations.

They must not automatically be interpreted as causal ecological
relationships.

Spatial transitions between acoustic events must not automatically be
interpreted as verified movement of the same individual animal.

Behavior indicators are derived acoustic-pattern indicators rather than
confirmed ethological observations.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import argparse
import csv
import json
import math

from dataclasses import (
    asdict,
    dataclass,
    is_dataclass,
)

from datetime import (
    date,
    datetime,
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
# PROJECT
# ======================================================================


from analytics.service import (
    build_research_analytics_report,
)

from config import (
    CONFIG,
    AppConfig,
)

from database import (
    EventDatabase,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_EXPORT_DIRECTORY = (
    Path(
        "data/exports"
    )
)


DEFAULT_EXPORT_STEM = (
    "research_metrics"
)


UINT32_MAX = (
    0xFFFFFFFF
)


# ======================================================================
# EXPORT OPTIONS
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ResearchMetricsExportOptions:
    """
    Research-metric export settings.
    """

    output_directory: Path

    stem: str = (
        DEFAULT_EXPORT_STEM
    )

    pretty_json: bool = (
        True
    )

    export_csv_tables: bool = (
        True
    )

    def __post_init__(
        self,
    ) -> None:

        # ==============================================================
        # OUTPUT DIRECTORY
        # ==============================================================

        if not isinstance(
            self.output_directory,
            Path,
        ):

            raise TypeError(
                (
                    "output_directory must be "
                    "pathlib.Path."
                )
            )

        # ==============================================================
        # STEM
        # ==============================================================

        if not isinstance(
            self.stem,
            str,
        ):

            raise TypeError(
                "stem must be a string."
            )

        normalized_stem = (
            self.stem.strip()
        )

        if not (
            normalized_stem
        ):

            raise ValueError(
                "stem cannot be empty."
            )

        if (
            "/" in normalized_stem
            or "\\" in normalized_stem
        ):

            raise ValueError(
                (
                    "stem must be a filename stem, "
                    "not a path."
                )
            )

        # ==============================================================
        # FLAGS
        # ==============================================================

        if not isinstance(
            self.pretty_json,
            bool,
        ):

            raise TypeError(
                "pretty_json must be bool."
            )

        if not isinstance(
            self.export_csv_tables,
            bool,
        ):

            raise TypeError(
                "export_csv_tables must be bool."
            )


# ======================================================================
# EXPORT RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ResearchMetricsExportResult:
    """
    Description of files created by one export operation.
    """

    json_path: Path

    summary_csv_path: Path | None

    table_csv_paths: tuple[
        Path,
        ...,
    ]

    def to_dict(
        self,
    ) -> dict[
        str,
        object,
    ]:

        return {
            "json_path":
                str(
                    self.json_path
                ),

            "summary_csv_path":
                (
                    str(
                        self.summary_csv_path
                    )

                    if self.summary_csv_path
                    is not None

                    else None
                ),

            "table_csv_paths":
                [
                    str(
                        path
                    )

                    for path
                    in self.table_csv_paths
                ],
        }


# ======================================================================
# SESSION-ID VALIDATION
# ======================================================================


def validate_session_id(
    value: int,
) -> int:
    """
    Validate one Protocol-v4 uint32 session identifier.
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
            "session_id must be an integer."
        )

    if (
        value
        <= 0
    ):

        raise ValueError(
            "session_id must be greater than 0."
        )

    if (
        value
        > UINT32_MAX
    ):

        raise ValueError(
            (
                "session_id exceeds the "
                "Protocol-v4 uint32 range."
            )
        )

    return (
        int(
            value
        )
    )


# ======================================================================
# JSON-SAFE VALUE CONVERSION
# ======================================================================


def to_json_safe(
    value: Any,
) -> Any:
    """
    Convert an arbitrary analytics result into strict JSON-safe values.

    Serialization priority
    ----------------------
    1. Normal JSON scalar.
    2. Finite float.
    3. Enum.
    4. Path.
    5. datetime/date.
    6. Public ``to_dict()`` contract.
    7. Generic dataclass fallback.
    8. Mapping.
    9. Sequence.
    10. NumPy-style scalar ``item()``.

    ``to_dict()`` deliberately precedes generic dataclass conversion.

    This preserves calculated/public analytics properties that are not
    represented as stored dataclass fields.
    """

    # ==================================================================
    # NONE / BASIC SCALARS
    # ==================================================================

    if (
        value
        is None
        or isinstance(
            value,
            (
                str,
                bool,
                int,
            ),
        )
    ):

        return (
            value
        )

    # ==================================================================
    # FLOAT
    # ==================================================================

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):

            raise ValueError(
                (
                    "Research report contains "
                    "a non-finite floating-point value."
                )
            )

        return (
            value
        )

    # ==================================================================
    # ENUM
    # ==================================================================

    if isinstance(
        value,
        Enum,
    ):

        return (
            to_json_safe(
                value.value
            )
        )

    # ==================================================================
    # PATH
    # ==================================================================

    if isinstance(
        value,
        Path,
    ):

        return (
            str(
                value
            )
        )

    # ==================================================================
    # DATE / DATETIME
    # ==================================================================

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):

        return (
            value.isoformat()
        )

    # ==================================================================
    # AUTHORITATIVE PUBLIC SERIALIZATION CONTRACT
    # ==================================================================

    to_dict_method = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(
        to_dict_method
    ):

        serialized = (
            to_dict_method()
        )

        if (
            serialized
            is value
        ):

            raise ValueError(
                (
                    "to_dict() returned the original "
                    "object and would recurse forever."
                )
            )

        return (
            to_json_safe(
                serialized
            )
        )

    # ==================================================================
    # GENERIC DATACLASS FALLBACK
    # ==================================================================
    #
    # This path exists for ordinary dataclasses that do NOT expose their
    # own public serialization contract.
    # ==================================================================

    if (
        is_dataclass(
            value
        )
        and not isinstance(
            value,
            type,
        )
    ):

        return (
            to_json_safe(
                asdict(
                    value
                )
            )
        )

    # ==================================================================
    # MAPPING
    # ==================================================================

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(
                key
            ):
                to_json_safe(
                    item
                )

            for key, item
            in value.items()
        }

    # ==================================================================
    # SEQUENCE
    # ==================================================================

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):

        return [
            to_json_safe(
                item
            )

            for item
            in value
        ]

    # ==================================================================
    # NUMPY-LIKE SCALAR
    # ==================================================================

    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(
        item_method
    ):

        try:

            item_value = (
                item_method()
            )

        except Exception:

            pass

        else:

            if (
                item_value
                is not value
            ):

                return (
                    to_json_safe(
                        item_value
                    )
                )

    raise TypeError(
        (
            "Unsupported research-report value "
            f"type: {type(value).__name__}"
        )
    )


# ======================================================================
# NORMALIZE REPORT
# ======================================================================


def normalize_report(
    report: Any,
) -> dict[
    str,
    Any,
]:
    """
    Convert analytics report object into a JSON-safe dictionary.

    The preferred public contract is:

        ResearchAnalyticsReport.to_dict()
    """

    normalized = (
        to_json_safe(
            report
        )
    )

    if not isinstance(
        normalized,
        dict,
    ):

        raise TypeError(
            (
                "Research analytics report "
                "must normalize to a dictionary."
            )
        )

    return (
        normalized
    )


# ======================================================================
# SCALAR CHECK
# ======================================================================


def is_scalar_export_value(
    value: Any,
) -> bool:
    """
    Return True for values suitable for one summary CSV cell.
    """

    return (
        value
        is None
        or isinstance(
            value,
            (
                str,
                bool,
                int,
                float,
            ),
        )
    )


# ======================================================================
# FLATTEN SCALAR METRICS
# ======================================================================


def flatten_scalar_metrics(
    value: Any,
    *,
    prefix: str = "",
) -> dict[
    str,
    Any,
]:
    """
    Flatten scalar fields from a nested analytics dictionary.

    Lists are deliberately omitted because tabular/list information is
    exported separately.

    Example
    -------
    Input:

        {
            "activity": {
                "total_events": 52,
                "mean_event_duration_s": 1.8,
            }
        }

    Output:

        {
            "activity.total_events": 52,
            "activity.mean_event_duration_s": 1.8,
        }
    """

    flattened: dict[
        str,
        Any,
    ] = {}

    if isinstance(
        value,
        Mapping,
    ):

        for key, item in (
            value.items()
        ):

            key_text = (
                str(
                    key
                )
            )

            path = (
                key_text

                if not prefix

                else (
                    f"{prefix}.{key_text}"
                )
            )

            if (
                is_scalar_export_value(
                    item
                )
            ):

                flattened[
                    path
                ] = (
                    item
                )

            elif isinstance(
                item,
                Mapping,
            ):

                flattened.update(
                    flatten_scalar_metrics(
                        item,
                        prefix=
                            path,
                    )
                )

    return (
        flattened
    )


# ======================================================================
# SANITIZE TABLE NAME
# ======================================================================


def sanitize_filename_component(
    value: str,
) -> str:
    """
    Convert one report path into a safe deterministic filename component.
    """

    if not isinstance(
        value,
        str,
    ):

        raise TypeError(
            "value must be a string."
        )

    result: list[
        str
    ] = []

    for character in (
        value
    ):

        if (
            character.isalnum()
            or character
            in (
                "_",
                "-",
            )
        ):

            result.append(
                character.lower()
            )

        elif character in (
            ".",
            " ",
            "/",
            "\\",
        ):

            result.append(
                "_"
            )

        else:

            result.append(
                "_"
            )

    normalized = (
        "".join(
            result
        )
        .strip(
            "_"
        )
    )

    while (
        "__"
        in normalized
    ):

        normalized = (
            normalized.replace(
                "__",
                "_",
            )
        )

    return (
        normalized
        or "table"
    )


# ======================================================================
# FLATTEN ONE TABLE ROW
# ======================================================================


def flatten_table_row(
    row: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = "",
) -> dict[
    str,
    Any,
]:
    """
    Flatten one mapping for CSV export.

    Nested mappings become dotted columns.

    Nested lists remain deterministic JSON strings so no information is
    discarded from the parent table.
    """

    output: dict[
        str,
        Any,
    ] = {}

    for key, value in (
        row.items()
    ):

        key_text = (
            str(
                key
            )
        )

        column = (
            key_text

            if not prefix

            else (
                f"{prefix}.{key_text}"
            )
        )

        if isinstance(
            value,
            Mapping,
        ):

            output.update(
                flatten_table_row(
                    value,
                    prefix=
                        column,
                )
            )

        elif isinstance(
            value,
            list,
        ):

            output[
                column
            ] = (
                json.dumps(
                    value,
                    ensure_ascii=
                        False,
                    allow_nan=
                        False,
                    separators=
                        (
                            ",",
                            ":",
                        ),
                )
            )

        elif (
            is_scalar_export_value(
                value
            )
        ):

            output[
                column
            ] = (
                value
            )

        else:

            output[
                column
            ] = (
                json.dumps(
                    to_json_safe(
                        value
                    ),
                    ensure_ascii=
                        False,
                    allow_nan=
                        False,
                    separators=
                        (
                            ",",
                            ":",
                        ),
                )
            )

    return (
        output
    )


# ======================================================================
# TABLE DISCOVERY
# ======================================================================


def discover_report_tables(
    value: Any,
    *,
    path: str = "",
) -> dict[
    str,
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    """
    Discover list-based tabular sections recursively.

    Policy
    ------
    A list of mappings becomes one CSV table.

    A list of scalar values becomes one one-column CSV table.

    Nested list values inside a mapping-row are already retained as JSON
    inside the parent CSV row. They are therefore not exploded into one
    separate file per individual parent row.

    This prevents excessive output such as:

        behavior_indicators_row_0_evidence.csv
        behavior_indicators_row_1_evidence.csv
        behavior_indicators_row_2_evidence.csv
        ...

    while still preserving all information.
    """

    tables: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    # ==================================================================
    # MAPPING
    # ==================================================================

    if isinstance(
        value,
        Mapping,
    ):

        for key, item in (
            value.items()
        ):

            child_path = (
                str(
                    key
                )

                if not path

                else (
                    f"{path}.{key}"
                )
            )

            nested_tables = (
                discover_report_tables(
                    item,
                    path=
                        child_path,
                )
            )

            tables.update(
                nested_tables
            )

        return (
            tables
        )

    # ==================================================================
    # LIST
    # ==================================================================

    if isinstance(
        value,
        list,
    ):

        if not (
            value
        ):

            return (
                tables
            )

        # --------------------------------------------------------------
        # LIST OF MAPPINGS
        # --------------------------------------------------------------

        if all(
            isinstance(
                item,
                Mapping,
            )

            for item
            in value
        ):

            tables[
                path
                or "items"
            ] = [
                flatten_table_row(
                    item
                )

                for item
                in value
            ]

            return (
                tables
            )

        # --------------------------------------------------------------
        # LIST OF SCALARS
        # --------------------------------------------------------------

        if all(
            is_scalar_export_value(
                item
            )

            for item
            in value
        ):

            tables[
                path
                or "values"
            ] = [
                {
                    "value":
                        item
                }

                for item
                in value
            ]

            return (
                tables
            )

    return (
        tables
    )


# ======================================================================
# CSV FIELD ORDER
# ======================================================================


def union_fieldnames(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> list[
    str
]:
    """
    Build deterministic union of CSV columns.
    """

    fields: set[
        str
    ] = set()

    for row in (
        rows
    ):

        fields.update(
            str(
                key
            )

            for key
            in row.keys()
        )

    return (
        sorted(
            fields
        )
    )


# ======================================================================
# WRITE CSV
# ======================================================================


def write_csv_rows(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    output_path: Path,
) -> None:
    """
    Write arbitrary flat rows to UTF-8 CSV.
    """

    if not isinstance(
        output_path,
        Path,
    ):

        raise TypeError(
            "output_path must be pathlib.Path."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = (
        union_fieldnames(
            rows
        )
    )

    with output_path.open(
        "w",
        encoding=
            "utf-8",
        newline=
            "",
    ) as handle:

        if not (
            fieldnames
        ):

            return

        writer = csv.DictWriter(
            handle,
            fieldnames=
                fieldnames,
            extrasaction=
                "ignore",
        )

        writer.writeheader()

        for row in (
            rows
        ):

            writer.writerow(
                {
                    field:
                        row.get(
                            field
                        )

                    for field
                    in fieldnames
                }
            )


# ======================================================================
# WRITE SUMMARY CSV
# ======================================================================


def write_summary_csv(
    report: Mapping[
        str,
        Any,
    ],
    output_path: Path,
) -> None:
    """
    Write flattened scalar analytics metrics.

    Format:

        metric,value
    """

    metrics = (
        flatten_scalar_metrics(
            report
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding=
            "utf-8",
        newline=
            "",
    ) as handle:

        writer = (
            csv.writer(
                handle
            )
        )

        writer.writerow(
            (
                "metric",
                "value",
            )
        )

        for metric in sorted(
            metrics
        ):

            value = (
                metrics[
                    metric
                ]
            )

            if isinstance(
                value,
                bool,
            ):

                exported_value = (
                    1
                    if value
                    else 0
                )

            else:

                exported_value = (
                    value
                )

            writer.writerow(
                (
                    metric,
                    exported_value,
                )
            )


# ======================================================================
# WRITE REPORT JSON
# ======================================================================


def write_report_json(
    report: Mapping[
        str,
        Any,
    ],
    output_path: Path,
    *,
    pretty: bool,
) -> None:
    """
    Write complete research analytics report.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding=
            "utf-8",
    ) as handle:

        json.dump(
            dict(
                report
            ),
            handle,
            ensure_ascii=
                False,
            allow_nan=
                False,
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
# EXPORT RESEARCH REPORT
# ======================================================================


def export_research_metrics(
    report: Any,
    options: ResearchMetricsExportOptions,
) -> ResearchMetricsExportResult:
    """
    Export one ResearchAnalyticsReport or equivalent mapping.

    The public analytics ``to_dict()`` contract is used when available.
    """

    if not isinstance(
        options,
        ResearchMetricsExportOptions,
    ):

        raise TypeError(
            (
                "options must be "
                "ResearchMetricsExportOptions."
            )
        )

    normalized = (
        normalize_report(
            report
        )
    )

    output_directory = (
        options.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================================
    # COMPLETE JSON
    # ==================================================================

    json_path = (
        output_directory
        / (
            f"{options.stem}.json"
        )
    )

    write_report_json(
        normalized,
        json_path,
        pretty=
            options.pretty_json,
    )

    # ==================================================================
    # JSON-ONLY MODE
    # ==================================================================

    if not (
        options.export_csv_tables
    ):

        return (
            ResearchMetricsExportResult(
                json_path=
                    json_path,

                summary_csv_path=
                    None,

                table_csv_paths=
                    (),
            )
        )

    # ==================================================================
    # SCALAR SUMMARY
    # ==================================================================

    summary_path = (
        output_directory
        / (
            f"{options.stem}_summary.csv"
        )
    )

    write_summary_csv(
        normalized,
        summary_path,
    )

    # ==================================================================
    # NESTED TABLES
    # ==================================================================

    table_directory = (
        output_directory
        / (
            f"{options.stem}_tables"
        )
    )

    tables = (
        discover_report_tables(
            normalized
        )
    )

    table_paths: list[
        Path
    ] = []

    used_names: set[
        str
    ] = set()

    for report_path, rows in sorted(
        tables.items()
    ):

        if not (
            rows
        ):

            continue

        base_name = (
            sanitize_filename_component(
                report_path
            )
        )

        filename = (
            base_name
        )

        suffix_index = (
            2
        )

        while (
            filename
            in used_names
        ):

            filename = (
                f"{base_name}_{suffix_index}"
            )

            suffix_index += (
                1
            )

        used_names.add(
            filename
        )

        output_path = (
            table_directory
            / (
                f"{filename}.csv"
            )
        )

        write_csv_rows(
            rows,
            output_path,
        )

        table_paths.append(
            output_path
        )

    return (
        ResearchMetricsExportResult(
            json_path=
                json_path,

            summary_csv_path=
                summary_path,

            table_csv_paths=
                tuple(
                    table_paths
                ),
        )
    )


# ======================================================================
# EXTERNAL SESSION COHERENCE
# ======================================================================


def validate_external_session_coherence(
    event_rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
) -> tuple[
    Mapping[
        str,
        Any,
    ],
    ...,
]:
    """
    Materialize external event rows and reject obvious multi-session
    datasets.

    If ``session_id`` is unavailable entirely, no assumption is made.

    If session IDs are present, all non-null IDs must represent one
    acquisition session.
    """

    materialized = tuple(
        event_rows
    )

    session_ids: set[
        int
    ] = set()

    for index, row in enumerate(
        materialized
    ):

        if not isinstance(
            row,
            Mapping,
        ):

            raise TypeError(
                (
                    f"event row {index} must "
                    "be mapping-like."
                )
            )

        value = (
            row.get(
                "session_id"
            )
        )

        if (
            value
            is None
        ):

            continue

        try:

            session_id = int(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                (
                    f"event row {index} contains "
                    "an invalid session_id."
                )
            ) from exc

        validate_session_id(
            session_id
        )

        session_ids.add(
            session_id
        )

    if (
        len(
            session_ids
        )
        > 1
    ):

        raise ValueError(
            (
                "External event dataset contains "
                "multiple acquisition sessions. "
                "Generate one research report per "
                "session before cross-session "
                "aggregation."
            )
        )

    return (
        materialized
    )


# ======================================================================
# BUILD ANALYTICS REPORT
# ======================================================================


def build_research_report(
    event_rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    environmental_rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ] | None = None,
    config: AppConfig = CONFIG,
):
    """
    Build research analytics using application configuration.

    Database access remains outside this function so the API can also be
    used with:

        test datasets
        exported datasets
        simulations
        research notebooks

    External rows carrying multiple distinct session identifiers are
    rejected to protect spatial/temporal interpretation.
    """

    if not isinstance(
        config,
        AppConfig,
    ):

        raise TypeError(
            "config must be an AppConfig."
        )

    event_rows_tuple = (
        validate_external_session_coherence(
            event_rows
        )
    )

    environmental_tuple = (
        None

        if environmental_rows
        is None

        else tuple(
            environmental_rows
        )
    )

    analytics = (
        config.analytics
    )

    return (
        build_research_analytics_report(
            event_rows_tuple,

            environmental_rows=
                environmental_tuple,

            sample_rate=
                config
                .audio
                .sample_rate,

            timestamp_key=
                "event_time",

            bucket_seconds=
                analytics
                .bucket_seconds,

            cell_size_m=
                analytics
                .cell_size_m,

            max_grid_cells=
                analytics
                .max_grid_cells,

            max_transition_gap_s=
                analytics
                .max_transition_gap_s,

            same_class_transitions_only=
                analytics
                .same_class_transitions_only,

            alpha=
                analytics
                .environmental_alpha,

            environmental_min_samples=
                analytics
                .environmental_min_samples,

            neutral_threshold=
                analytics
                .neutral_threshold,

            min_activity_events=
                analytics
                .min_activity_events,

            min_localized_events=
                analytics
                .min_localized_events,

            min_transitions=
                analytics
                .min_transitions,
        )
    )


# ======================================================================
# BUILD + EXPORT
# ======================================================================


def build_and_export_research_metrics(
    event_rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    environmental_rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ] | None = None,
    options: ResearchMetricsExportOptions,
    config: AppConfig = CONFIG,
) -> ResearchMetricsExportResult:
    """
    Convenience function for analytics + research export.
    """

    report = (
        build_research_report(
            event_rows,

            environmental_rows=
                environmental_rows,

            config=
                config,
        )
    )

    return (
        export_research_metrics(
            report,
            options,
        )
    )


# ======================================================================
# EVENT DATABASE CONSTRUCTION
# ======================================================================


def create_event_database(
    database_path: Path,
) -> EventDatabase:
    """
    Construct the project's current EventDatabase implementation.

    Current constructor contract:

        EventDatabase(path)
    """

    if not isinstance(
        database_path,
        Path,
    ):

        raise TypeError(
            "database_path must be pathlib.Path."
        )

    return (
        EventDatabase(
            database_path
        )
    )


# ======================================================================
# LOAD DATABASE ANALYTICS INPUTS
# ======================================================================


def load_database_analytics_inputs(
    database_path: Path,
    *,
    session_id: int,
    config: AppConfig = CONFIG,
) -> tuple[
    list[
        Mapping[
            str,
            Any,
        ]
    ],
    list[
        Mapping[
            str,
            Any,
        ]
    ],
]:
    """
    Load event and regular environmental-bin rows from EventDatabase.

    Environmental bins are deliberately used instead of merely attaching
    environmental values to detected events.

    This retains zero-event periods while the selected session was
    actually acquiring data.
    """

    session_id = (
        validate_session_id(
            session_id
        )
    )

    if not isinstance(
        database_path,
        Path,
    ):

        raise TypeError(
            "database_path must be pathlib.Path."
        )

    if not isinstance(
        config,
        AppConfig,
    ):

        raise TypeError(
            "config must be an AppConfig."
        )

    database = (
        create_event_database(
            database_path
        )
    )

    # ==================================================================
    # EVENTS
    # ==================================================================

    event_rows = (
        database.analytics_event_rows(
            session_id=
                session_id,

            sample_rate=
                config
                .audio
                .sample_rate,
        )
    )

    # ==================================================================
    # REGULAR ENVIRONMENTAL BINS
    # ==================================================================

    environmental_rows = (
        database.analytics_environmental_bins(
            session_id=
                session_id,

            sample_rate=
                config
                .audio
                .sample_rate,

            bucket_seconds=
                config
                .analytics
                .bucket_seconds,

            node_id=
                config
                .analytics
                .environmental_node_id,
        )
    )

    return (
        list(
            event_rows
        ),
        list(
            environmental_rows
        ),
    )


# ======================================================================
# EXPORT DATABASE SESSION
# ======================================================================


def export_database_session(
    database_path: Path,
    *,
    session_id: int,
    options: ResearchMetricsExportOptions,
    config: AppConfig = CONFIG,
) -> ResearchMetricsExportResult:
    """
    Generate and export research analytics for one acquisition session.

    Combining separate acquisition sessions into one continuous
    temporal/spatial sequence is deliberately prohibited here.
    """

    session_id = (
        validate_session_id(
            session_id
        )
    )

    (
        event_rows,
        environmental_rows,
    ) = (
        load_database_analytics_inputs(
            database_path,

            session_id=
                session_id,

            config=
                config,
        )
    )

    if not (
        event_rows
    ):

        raise ValueError(
            (
                "Selected session contains "
                "no persisted acoustic events."
            )
        )

    return (
        build_and_export_research_metrics(
            event_rows,

            environmental_rows=
                environmental_rows,

            options=
                options,

            config=
                config,
        )
    )


# ======================================================================
# LOAD JSON / JSONL ROWS
# ======================================================================


def load_rows_file(
    path: Path,
) -> list[
    Mapping[
        str,
        Any,
    ]
]:
    """
    Load research input rows from JSON or JSONL.

    JSON must contain a top-level list of objects.
    """

    if not isinstance(
        path,
        Path,
    ):

        raise TypeError(
            "path must be pathlib.Path."
        )

    if not (
        path.exists()
    ):

        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    if not (
        path.is_file()
    ):

        raise ValueError(
            f"Input path is not a file: {path}"
        )

    suffix = (
        path.suffix.lower()
    )

    rows: list[
        Mapping[
            str,
            Any,
        ]
    ]

    # ==================================================================
    # JSON
    # ==================================================================

    if (
        suffix
        == ".json"
    ):

        with path.open(
            "r",
            encoding=
                "utf-8",
        ) as handle:

            value = (
                json.load(
                    handle
                )
            )

        if not isinstance(
            value,
            list,
        ):

            raise ValueError(
                (
                    "JSON analytics input must "
                    "contain a top-level list."
                )
            )

        rows = (
            value
        )

    # ==================================================================
    # JSONL
    # ==================================================================

    elif (
        suffix
        == ".jsonl"
    ):

        rows = []

        with path.open(
            "r",
            encoding=
                "utf-8",
        ) as handle:

            for line_number, line in enumerate(
                handle,
                start=
                    1,
            ):

                text = (
                    line.strip()
                )

                if not (
                    text
                ):

                    continue

                try:

                    value = (
                        json.loads(
                            text
                        )
                    )

                except json.JSONDecodeError as exc:

                    raise ValueError(
                        (
                            "Invalid JSONL at "
                            f"line {line_number}."
                        )
                    ) from exc

                rows.append(
                    value
                )

    else:

        raise ValueError(
            (
                "Research input must use "
                ".json or .jsonl."
            )
        )

    # ==================================================================
    # ROW VALIDATION
    # ==================================================================

    for index, row in enumerate(
        rows
    ):

        if not isinstance(
            row,
            Mapping,
        ):

            raise ValueError(
                (
                    "Research input row "
                    f"{index} is not an object."
                )
            )

    return (
        rows
    )


# ======================================================================
# CLI PARSER
# ======================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate and export Wildlife Soundscape "
            "research analytics."
        )
    )

    source_group = (
        parser.add_mutually_exclusive_group(
            required=
                True
        )
    )

    # ==================================================================
    # DATABASE MODE
    # ==================================================================

    source_group.add_argument(
        "--session-id",
        type=
            int,
        default=
            None,
        help=(
            "Build research analytics directly "
            "from one persisted database session."
        ),
    )

    # ==================================================================
    # EXTERNAL EVENT DATASET
    # ==================================================================

    source_group.add_argument(
        "--events",
        type=
            Path,
        default=
            None,
        help=(
            "JSON/JSONL event rows used as analytics input. "
            "A dataset containing multiple session_id values "
            "is rejected."
        ),
    )

    # ==================================================================
    # DATABASE
    # ==================================================================

    parser.add_argument(
        "--database",
        type=
            Path,
        default=
            CONFIG
            .persistence
            .database_path,
        help=(
            "SQLite database path used with --session-id."
        ),
    )

    # ==================================================================
    # OPTIONAL ENVIRONMENT ROWS
    # ==================================================================

    parser.add_argument(
        "--environment",
        type=
            Path,
        default=
            None,
        help=(
            "Optional JSON/JSONL regular environmental-bin "
            "rows when using --events."
        ),
    )

    # ==================================================================
    # OUTPUT
    # ==================================================================

    parser.add_argument(
        "--output-dir",
        type=
            Path,
        default=
            DEFAULT_EXPORT_DIRECTORY,
        help=(
            "Research export directory."
        ),
    )

    parser.add_argument(
        "--stem",
        type=
            str,
        default=
            None,
        help=(
            "Output filename stem."
        ),
    )

    parser.add_argument(
        "--json-only",
        action=
            "store_true",
        help=(
            "Generate only the complete JSON report."
        ),
    )

    parser.add_argument(
        "--compact-json",
        action=
            "store_true",
        help=(
            "Disable pretty JSON indentation."
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

    # ==================================================================
    # STEM
    # ==================================================================

    if (
        arguments.stem
        is not None
    ):

        stem = (
            arguments.stem
        )

    elif (
        arguments.session_id
        is not None
    ):

        stem = (
            "research_metrics_"
            f"session_{arguments.session_id}"
        )

    else:

        stem = (
            DEFAULT_EXPORT_STEM
        )

    # ==================================================================
    # OPTIONS
    # ==================================================================

    try:

        options = (
            ResearchMetricsExportOptions(
                output_directory=
                    arguments.output_dir,

                stem=
                    stem,

                pretty_json=
                    not arguments.compact_json,

                export_csv_tables=
                    not arguments.json_only,
            )
        )

        # ==============================================================
        # DATABASE SESSION
        # ==============================================================

        if (
            arguments.session_id
            is not None
        ):

            result = (
                export_database_session(
                    arguments.database,

                    session_id=
                        arguments.session_id,

                    options=
                        options,

                    config=
                        CONFIG,
                )
            )

        # ==============================================================
        # EXTERNAL ROW DATASET
        # ==============================================================

        else:

            if (
                arguments.events
                is None
            ):

                raise ValueError(
                    (
                        "External dataset mode "
                        "requires --events."
                    )
                )

            event_rows = (
                load_rows_file(
                    arguments.events
                )
            )

            if (
                arguments.environment
                is None
            ):

                environmental_rows = (
                    None
                )

            else:

                environmental_rows = (
                    load_rows_file(
                        arguments.environment
                    )
                )

            result = (
                build_and_export_research_metrics(
                    event_rows,

                    environmental_rows=
                        environmental_rows,

                    options=
                        options,

                    config=
                        CONFIG,
                )
            )

    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:

        parser.exit(
            status=
                1,
            message=
                (
                    "Research export failed: "
                    f"{exc}\n"
                ),
        )

    # ==================================================================
    # STATUS
    # ==================================================================

    print(
        "Research analytics exported:"
    )

    print(
        (
            f"  JSON: "
            f"{result.json_path}"
        )
    )

    if (
        result.summary_csv_path
        is not None
    ):

        print(
            (
                "  Summary CSV: "
                f"{result.summary_csv_path}"
            )
        )

    if (
        result.table_csv_paths
    ):

        print(
            (
                "  Table CSV files: "
                f"{len(result.table_csv_paths)}"
            )
        )

        for path in (
            result.table_csv_paths
        ):

            print(
                (
                    f"    - {path}"
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