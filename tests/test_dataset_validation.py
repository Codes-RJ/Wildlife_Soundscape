"""Tests for the native three-node dataset contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, cast
import wave

import pytest

from wildlife_soundscape.core.config import AppConfig
from wildlife_soundscape.datasets import (
    RECOMMENDED_DATASETS,
    SplitRatios,
    assign_group_splits,
    validate_dataset,
)
from wildlife_soundscape.datasets.validation import (
    CHANNELS,
    FRAMES_PER_BLOCK,
    PROTOCOL_VERSION,
    SAMPLE_RATE_HZ,
    SAMPLE_WIDTH_BYTES,
    SYNC_TOLERANCE_SAMPLES,
    ValidationReport,
)
from wildlife_soundscape.tools.validate_dataset import main


def _write_wav(
    path: Path,
    *,
    frames: int = 2048,
    sample_rate: int = SAMPLE_RATE_HZ,
    channels: int = CHANNELS,
    sample_width: int = SAMPLE_WIDTH_BYTES,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(sample_width)
        audio.setframerate(sample_rate)
        audio.writeframes(bytes(frames * channels * sample_width))


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": "wildlife_array_v1",
        "dataset_id": "native-test",
        "synthetic": False,
        "tasks": [
            "event_detection",
            "classification",
            "localization",
            "ecoacoustics",
        ],
        "audio_format": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": CHANNELS,
            "sample_width_bytes": SAMPLE_WIDTH_BYTES,
            "encoding": "PCM_S16LE",
        },
        "required_nodes": [1, 2, 3],
        "capture_contract": {
            "transport_protocol_version": PROTOCOL_VERSION,
            "frames_per_block": FRAMES_PER_BLOCK,
            "alignment_basis": "sample_index",
            "sync_tolerance_samples": SYNC_TOLERANCE_SAMPLES,
        },
        "taxonomy": {"authority": "eBird/Clements", "version": "2024"},
        "tdoa_calibration": {
            "status": "measured",
            "reference_node": 1,
            "node_biases_seconds": {"1": 0.0, "2": 0.00001, "3": -0.00001},
            "rms_consistency_error_samples": 0.4,
        },
        "license": {"name": "Test terms", "url": "https://example.test/license"},
        "records": [
            {
                "record_id": "record-1",
                "group_id": "site-a-day-1",
                "split": "test",
                "recording_mode": "continuous",
                "started_at": "2026-09-07T06:00:00+05:30",
                "annotation_status": "verified",
                "labels": ["Buceros bicornis"],
                "audio_paths": {
                    "1": "audio/record-1/node_1.wav",
                    "2": "audio/record-1/node_2.wav",
                    "3": "audio/record-1/node_3.wav",
                },
                "node_positions_m": {
                    "1": [0.0, 0.0, 1.2],
                    "2": [0.5, 0.8660254, 1.2],
                    "3": [1.0, 0.0, 1.2],
                },
                "source_position_m": [0.4, 0.3, 1.2],
                "source_id": "source-1",
                "environment": {
                    "temperature_c": 24.0,
                    "humidity_percent": 70.0,
                    "pressure_hpa": 1008.0,
                },
            }
        ],
    }


def _write_manifest(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "dataset-manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_record_audio(
    tmp_path: Path, *, frame_counts: tuple[int, int, int] = (2048, 2048, 2048)
) -> None:
    for node, frames in enumerate(frame_counts, start=1):
        _write_wav(tmp_path / f"audio/record-1/node_{node}.wav", frames=frames)


def _error_codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.issues if issue.severity == "error"}


def test_contract_matches_runtime_defaults() -> None:
    config = AppConfig()

    assert SAMPLE_RATE_HZ == config.audio.sample_rate
    assert CHANNELS == config.audio.channels
    assert SAMPLE_WIDTH_BYTES == config.audio.sample_width_bytes
    assert FRAMES_PER_BLOCK == config.audio.frames_per_block
    assert SYNC_TOLERANCE_SAMPLES == config.audio.sync_tolerance_samples
    assert frozenset({1, 2, 3}) == config.expected_nodes


def test_valid_native_dataset_checks_all_three_audio_files(tmp_path: Path) -> None:
    _write_record_audio(tmp_path)
    report = validate_dataset(_write_manifest(tmp_path, _manifest()))

    assert report.valid
    assert report.records_checked == 1
    assert report.audio_files_checked == 3


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda data: data.update(synthetic=True), "synthetic_scientific"),
        (lambda data: data.update(synthetic=0), "synthetic"),
        (lambda data: data.update(tasks=[{}]), "tasks"),
        (lambda data: data.update(required_nodes=[True, 2, 3]), "required_nodes"),
        (lambda data: data.update(capture_contract={}), "capture_contract"),
        (lambda data: data.update(tdoa_calibration={}), "tdoa_calibration"),
    ],
)
def test_invalid_manifest_contract_is_reported(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    expected_code: str,
) -> None:
    document = _manifest()
    mutation(document)

    report = validate_dataset(_write_manifest(tmp_path, document), verify_audio=False)

    assert expected_code in _error_codes(report)


def test_wrong_wav_parameters_are_rejected(tmp_path: Path) -> None:
    _write_record_audio(tmp_path)
    _write_wav(tmp_path / "audio/record-1/node_2.wav", sample_rate=44_100)

    report = validate_dataset(_write_manifest(tmp_path, _manifest()))

    assert "audio_parameters" in _error_codes(report)


def test_cross_node_frame_mismatch_is_rejected(tmp_path: Path) -> None:
    _write_record_audio(tmp_path, frame_counts=(2048, 2047, 2048))

    report = validate_dataset(_write_manifest(tmp_path, _manifest()))

    assert "audio_alignment" in _error_codes(report)


def test_unfinalized_wav_container_is_rejected(tmp_path: Path) -> None:
    _write_record_audio(tmp_path)
    with (tmp_path / "audio/record-1/node_2.wav").open("ab") as audio:
        audio.write(bytes(4096))

    report = validate_dataset(_write_manifest(tmp_path, _manifest()))

    assert "audio_container_size" in _error_codes(report)


def test_sha256_checksums_are_verified(tmp_path: Path) -> None:
    _write_record_audio(tmp_path)
    document = _manifest()
    records = cast(list[dict[str, Any]], document["records"])
    records[0]["sha256"] = {
        str(node): hashlib.sha256(
            (tmp_path / f"audio/record-1/node_{node}.wav").read_bytes()
        ).hexdigest()
        for node in (1, 2, 3)
    }
    path = _write_manifest(tmp_path, document)

    assert validate_dataset(path, verify_checksums=True).valid
    cast(dict[str, str], records[0]["sha256"])["2"] = "0" * 64
    path = _write_manifest(tmp_path, document)
    assert "checksum_mismatch" in _error_codes(
        validate_dataset(path, verify_checksums=True)
    )


def test_group_leakage_and_collinear_geometry_are_rejected(tmp_path: Path) -> None:
    document = _manifest()
    records = cast(list[dict[str, Any]], document["records"])
    first = records[0]
    cast(dict[str, object], first["node_positions_m"])["2"] = [0.5, 0.0, 1.2]
    second = copy.deepcopy(first)
    second["record_id"] = "record-2"
    second["split"] = "train"
    second["audio_paths"] = {
        "1": "audio/record-2/node_1.wav",
        "2": "audio/record-2/node_2.wav",
        "3": "audio/record-2/node_3.wav",
    }
    records.append(second)

    report = validate_dataset(_write_manifest(tmp_path, document), verify_audio=False)

    assert {"group_leakage", "node_positions_collinear"} <= _error_codes(report)


def test_event_only_audio_cannot_pass_ecoacoustic_validation(tmp_path: Path) -> None:
    document = _manifest()
    records = cast(list[dict[str, Any]], document["records"])
    records[0]["recording_mode"] = "event"

    report = validate_dataset(_write_manifest(tmp_path, document), verify_audio=False)

    assert "ecoacoustics_trigger_bias" in _error_codes(report)


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    document = _manifest()
    records = cast(list[dict[str, Any]], document["records"])
    audio_paths = cast(dict[str, str], records[0]["audio_paths"])
    audio_paths["1"] = "../outside.wav"

    report = validate_dataset(_write_manifest(tmp_path, document), verify_audio=False)

    assert "audio_path" in _error_codes(report)


def test_group_split_is_deterministic_and_has_expected_counts() -> None:
    groups = [f"group-{index}" for index in range(20)]

    first = assign_group_splits(groups, seed=17)
    second = assign_group_splits(reversed(groups), seed=17)

    assert first == second
    assert list(first.values()).count("train") == 14
    assert list(first.values()).count("validation") == 3
    assert list(first.values()).count("test") == 3


def test_group_split_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_group_splits(["a"], ratios=SplitRatios(0.8, 0.2, 0.2))
    with pytest.raises(ValueError, match="unique"):
        assign_group_splits(["a", "a"])


def test_registry_reserves_final_validation_for_native_array() -> None:
    final_datasets = [
        item for item in RECOMMENDED_DATASETS if item.final_system_validation
    ]

    assert [item.dataset_id for item in final_datasets] == ["native-wildlife-array-v1"]
    assert any(item.dataset_id == "birdclef-2024" for item in RECOMMENDED_DATASETS)


def test_cli_returns_nonzero_and_json_for_invalid_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_manifest(tmp_path, {"schema_version": 1})

    assert main([str(path), "--skip-audio", "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is False
    assert output["issues"]
