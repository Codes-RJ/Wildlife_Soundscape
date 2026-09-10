"""Strict validation for native three-node research datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import wave


SCHEMA_VERSION = 1
PROFILE = "wildlife_array_v1"
SAMPLE_RATE_HZ = 48_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
REQUIRED_NODES = frozenset({1, 2, 3})
FRAMES_PER_BLOCK = 1024
SYNC_TOLERANCE_SAMPLES = 10
PROTOCOL_VERSION = 4
SPLITS = frozenset({"train", "validation", "test", "calibration"})
TASKS = frozenset({"event_detection", "classification", "localization", "ecoacoustics"})
RECORDING_MODES = frozenset({"event", "continuous", "duty_cycle"})
ANNOTATION_STATES = frozenset({"unverified", "reviewed", "verified"})


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    """One actionable dataset validation result."""

    severity: str
    code: str
    message: str
    record_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Machine-readable validation result."""

    manifest: Path
    records_checked: int
    audio_files_checked: int
    issues: tuple[DatasetIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest),
            "valid": self.valid,
            "records_checked": self.records_checked,
            "audio_files_checked": self.audio_files_checked,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def _strict_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _three_nodes(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(node, int) and not isinstance(node, bool) for node in value)
        and value == [1, 2, 3]
    )


def _non_collinear(node_positions: dict[str, Any]) -> bool:
    p1, p2, p3 = (node_positions[str(node)] for node in sorted(REQUIRED_NODES))
    twice_area = abs(
        (float(p2[0]) - float(p1[0])) * (float(p3[1]) - float(p1[1]))
        - (float(p2[1]) - float(p1[1])) * (float(p3[0]) - float(p1[0]))
    )
    return twice_area > 1e-12


def _safe_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = (root / value).resolve()
    return candidate if candidate.is_relative_to(root) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _riff_declared_size(path: Path) -> int | None:
    with path.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
        return None
    return int.from_bytes(header[4:8], byteorder="little", signed=False) + 8


def validate_dataset(
    manifest_path: Path,
    *,
    scientific: bool = True,
    verify_audio: bool = True,
    verify_checksums: bool = False,
) -> ValidationReport:
    """Validate a manifest and its audio without modifying either."""
    manifest_path = Path(manifest_path).resolve()
    issues: list[DatasetIssue] = []

    def issue(
        severity: str, code: str, message: str, record_id: str | None = None
    ) -> None:
        issues.append(DatasetIssue(severity, code, message, record_id))

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issue("error", "manifest_unreadable", str(exc))
        return ValidationReport(manifest_path, 0, 0, tuple(issues))

    if not isinstance(document, dict):
        issue("error", "manifest_type", "Manifest root must be a JSON object.")
        return ValidationReport(manifest_path, 0, 0, tuple(issues))

    if not _strict_int(document.get("schema_version"), SCHEMA_VERSION):
        issue("error", "schema_version", f"schema_version must be {SCHEMA_VERSION}.")
    if document.get("profile") != PROFILE:
        issue("error", "profile", f"profile must be {PROFILE!r}.")
    if (
        not isinstance(document.get("dataset_id"), str)
        or not document["dataset_id"].strip()
    ):
        issue("error", "dataset_id", "dataset_id must be a non-empty string.")
    if not isinstance(document.get("synthetic"), bool):
        issue("error", "synthetic", "synthetic must be a JSON boolean.")
    elif scientific and document["synthetic"]:
        issue(
            "error",
            "synthetic_scientific",
            "Synthetic data cannot pass scientific mode.",
        )

    tasks_value = document.get("tasks")
    if (
        isinstance(tasks_value, list)
        and bool(tasks_value)
        and all(isinstance(task, str) and task.strip() for task in tasks_value)
    ):
        tasks = set(tasks_value)
    else:
        tasks = set()
        issue("error", "tasks", "tasks must be a non-empty string list.")
    unknown_tasks = tasks - TASKS
    if unknown_tasks:
        issue("error", "tasks_unknown", f"Unknown tasks: {sorted(unknown_tasks)}")

    audio_format = document.get("audio_format")
    expected_format = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": CHANNELS,
        "sample_width_bytes": SAMPLE_WIDTH_BYTES,
        "encoding": "PCM_S16LE",
    }
    audio_format_valid = (
        isinstance(audio_format, dict)
        and set(audio_format) == set(expected_format)
        and _strict_int(audio_format.get("sample_rate_hz"), SAMPLE_RATE_HZ)
        and _strict_int(audio_format.get("channels"), CHANNELS)
        and _strict_int(audio_format.get("sample_width_bytes"), SAMPLE_WIDTH_BYTES)
        and audio_format.get("encoding") == "PCM_S16LE"
    )
    if not audio_format_valid:
        issue("error", "audio_format", f"audio_format must equal {expected_format}.")

    required_nodes = document.get("required_nodes")
    if not _three_nodes(required_nodes):
        issue("error", "required_nodes", "required_nodes must be [1, 2, 3].")

    capture = document.get("capture_contract")
    capture_valid = (
        isinstance(capture, dict)
        and set(capture)
        == {
            "transport_protocol_version",
            "frames_per_block",
            "alignment_basis",
            "sync_tolerance_samples",
        }
        and _strict_int(capture.get("transport_protocol_version"), PROTOCOL_VERSION)
        and _strict_int(capture.get("frames_per_block"), FRAMES_PER_BLOCK)
        and capture.get("alignment_basis") == "sample_index"
        and _strict_int(capture.get("sync_tolerance_samples"), SYNC_TOLERANCE_SAMPLES)
    )
    if not capture_valid:
        issue(
            "error",
            "capture_contract",
            "capture_contract must match Protocol v4: 1024-frame blocks, "
            "sample_index alignment, and 10-sample tolerance.",
        )

    if "classification" in tasks:
        taxonomy = document.get("taxonomy")
        if not isinstance(taxonomy, dict) or not all(
            isinstance(taxonomy.get(field), str) and taxonomy[field].strip()
            for field in ("authority", "version")
        ):
            issue(
                "error",
                "taxonomy",
                "Classification datasets require taxonomy.authority and taxonomy.version.",
            )

    if "localization" in tasks:
        calibration = document.get("tdoa_calibration")
        if isinstance(calibration, dict):
            reference_node = calibration.get("reference_node")
            biases = calibration.get("node_biases_seconds")
            rms = calibration.get("rms_consistency_error_samples")
            calibration_valid = (
                calibration.get("status") == "measured"
                and isinstance(reference_node, int)
                and not isinstance(reference_node, bool)
                and reference_node in REQUIRED_NODES
                and isinstance(biases, dict)
                and set(biases) == {"1", "2", "3"}
                and all(
                    _finite_vector([biases[str(node)]], 1) for node in REQUIRED_NODES
                )
                and abs(float(biases[str(reference_node)])) <= 1e-12
                and isinstance(rms, (int, float))
                and not isinstance(rms, bool)
                and math.isfinite(float(rms))
                and 0.0 <= float(rms) <= 1.0
            )
        else:
            calibration_valid = False
        if not calibration_valid:
            issue(
                "error" if scientific else "warning",
                "tdoa_calibration",
                "Localization requires measured three-node timing biases with a zero "
                "reference bias and RMS consistency error at most 1 sample.",
            )

    license_info = document.get("license")
    if not isinstance(license_info, dict) or not all(
        isinstance(license_info.get(key), str) and license_info[key].strip()
        for key in ("name", "url")
    ):
        issue("error", "license", "license.name and license.url are required.")

    records_value = document.get("records")
    if not isinstance(records_value, list) or not records_value:
        issue("error", "records", "records must be a non-empty list.")
        records: list[Any] = []
    else:
        records = records_value

    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    group_splits: dict[str, str] = {}
    audio_files_checked = 0
    observed_splits: set[str] = set()

    for index, value in enumerate(records):
        if not isinstance(value, dict):
            issue("error", "record_type", "Record must be an object.", str(index))
            continue

        record_id_value = value.get("record_id")
        record_id = record_id_value if isinstance(record_id_value, str) else str(index)
        if not isinstance(record_id_value, str) or not record_id_value.strip():
            issue("error", "record_id", "record_id must be non-empty.", record_id)
        elif record_id in seen_ids:
            issue("error", "record_id_duplicate", "record_id is duplicated.", record_id)
        seen_ids.add(record_id)

        split = value.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            issue(
                "error", "split", f"split must be one of {sorted(SPLITS)}.", record_id
            )
        else:
            observed_splits.add(split)

        group_id = value.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            issue("error", "group_id", "group_id must be non-empty.", record_id)
        elif isinstance(split, str):
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                issue(
                    "error",
                    "group_leakage",
                    f"group_id {group_id!r} occurs in {previous!r} and {split!r}.",
                    record_id,
                )

        mode = value.get("recording_mode")
        if not isinstance(mode, str) or mode not in RECORDING_MODES:
            issue("error", "recording_mode", "Invalid recording_mode.", record_id)
        if "ecoacoustics" in tasks and mode == "event":
            issue(
                "error",
                "ecoacoustics_trigger_bias",
                "Ecoacoustics requires continuous or fixed duty-cycle audio.",
                record_id,
            )

        timestamp = value.get("started_at")
        try:
            parsed_time = (
                datetime.fromisoformat(timestamp)
                if isinstance(timestamp, str)
                else None
            )
        except ValueError:
            parsed_time = None
        if parsed_time is None or parsed_time.tzinfo is None:
            issue(
                "error",
                "started_at",
                "started_at must include a UTC offset.",
                record_id,
            )

        annotation_state = value.get("annotation_status")
        if (
            not isinstance(annotation_state, str)
            or annotation_state not in ANNOTATION_STATES
        ):
            issue("error", "annotation_status", "Invalid annotation_status.", record_id)
        elif scientific and split == "test" and annotation_state != "verified":
            issue(
                "error",
                "test_not_verified",
                "Test annotations must be verified.",
                record_id,
            )

        labels = value.get("labels")
        if not isinstance(labels, list) or not all(
            isinstance(label, str) and label.strip() for label in labels
        ):
            issue(
                "error",
                "labels",
                "labels must be a string list; [] means background.",
                record_id,
            )

        node_positions = value.get("node_positions_m")
        if not isinstance(node_positions, dict) or set(node_positions) != {
            "1",
            "2",
            "3",
        }:
            issue(
                "error",
                "node_positions",
                "node_positions_m must define nodes 1, 2, 3.",
                record_id,
            )
        elif not all(
            _finite_vector(node_positions[str(node)], 3) for node in REQUIRED_NODES
        ):
            issue(
                "error",
                "node_positions",
                "Each node position must be [x, y, z].",
                record_id,
            )
        elif len({tuple(node_positions[str(node)]) for node in REQUIRED_NODES}) != 3:
            issue(
                "error",
                "node_positions_unique",
                "Node positions must be distinct.",
                record_id,
            )
        elif "localization" in tasks and not _non_collinear(node_positions):
            issue(
                "error",
                "node_positions_collinear",
                "2-D localization requires non-collinear node coordinates.",
                record_id,
            )

        source_position = value.get("source_position_m")
        if "localization" in tasks and not _finite_vector(source_position, 3):
            issue(
                "error",
                "source_position",
                "Localization requires source_position_m.",
                record_id,
            )
        if "localization" in tasks and (
            not isinstance(value.get("source_id"), str)
            or not value["source_id"].strip()
        ):
            issue("error", "source_id", "Localization requires source_id.", record_id)

        environment = value.get("environment")
        if not isinstance(environment, dict):
            issue("error", "environment", "environment object is required.", record_id)
        else:
            environment_ranges = {
                "temperature_c": (-80.0, 70.0),
                "humidity_percent": (0.0, 100.0),
                "pressure_hpa": (300.0, 1100.0),
            }
            for field, (minimum, maximum) in environment_ranges.items():
                raw = environment.get(field)
                if (
                    not isinstance(raw, (int, float))
                    or isinstance(raw, bool)
                    or not math.isfinite(float(raw))
                    or not minimum <= float(raw) <= maximum
                ):
                    issue("error", "environment_range", f"Invalid {field}.", record_id)

        audio_paths = value.get("audio_paths")
        if not isinstance(audio_paths, dict) or set(audio_paths) != {"1", "2", "3"}:
            issue(
                "error",
                "audio_paths",
                "audio_paths must define nodes 1, 2, 3.",
                record_id,
            )
            continue

        frame_counts: dict[int, int] = {}
        for node in sorted(REQUIRED_NODES):
            path = _safe_path(manifest_path.parent, audio_paths[str(node)])
            if path is None:
                issue(
                    "error",
                    "audio_path",
                    f"Unsafe or invalid node {node} path.",
                    record_id,
                )
                continue
            if path in seen_paths:
                issue(
                    "error",
                    "audio_path_duplicate",
                    f"Audio path reused: {path.name}.",
                    record_id,
                )
            seen_paths.add(path)
            if not verify_audio:
                continue
            try:
                with wave.open(str(path), "rb") as audio:
                    params = (
                        audio.getframerate(),
                        audio.getnchannels(),
                        audio.getsampwidth(),
                        audio.getcomptype(),
                    )
                    frames = audio.getnframes()
            except (OSError, EOFError, wave.Error) as exc:
                issue("error", "audio_unreadable", f"Node {node}: {exc}", record_id)
                continue
            audio_files_checked += 1
            declared_size = _riff_declared_size(path)
            actual_size = path.stat().st_size
            if declared_size != actual_size:
                issue(
                    "error",
                    "audio_container_size",
                    f"Node {node} RIFF declares {declared_size} bytes but file has "
                    f"{actual_size}; recording may be unfinalized.",
                    record_id,
                )
            if params != (SAMPLE_RATE_HZ, CHANNELS, SAMPLE_WIDTH_BYTES, "NONE"):
                issue(
                    "error", "audio_parameters", f"Node {node} has {params}.", record_id
                )
            if frames <= 0:
                issue("error", "audio_empty", f"Node {node} has no frames.", record_id)
            frame_counts[node] = frames

            checksums = value.get("sha256")
            if verify_checksums:
                expected = (
                    checksums.get(str(node)) if isinstance(checksums, dict) else None
                )
                if not isinstance(expected, str) or len(expected) != 64:
                    issue(
                        "error",
                        "checksum_missing",
                        f"Node {node} needs SHA-256.",
                        record_id,
                    )
                elif _sha256(path) != expected.lower():
                    issue(
                        "error",
                        "checksum_mismatch",
                        f"Node {node} SHA-256 differs.",
                        record_id,
                    )

        if len(frame_counts) == 3 and len(set(frame_counts.values())) != 1:
            issue(
                "error",
                "audio_alignment",
                f"Frame counts differ: {frame_counts}.",
                record_id,
            )

    if scientific and "test" not in observed_splits:
        issue("warning", "test_split_missing", "No locked test split is present.")
    if (
        scientific
        and tasks & {"classification", "event_detection"}
        and "validation" not in observed_splits
    ):
        issue("warning", "validation_split_missing", "No validation split is present.")

    return ValidationReport(
        manifest_path,
        len(records),
        audio_files_checked,
        tuple(issues),
    )
