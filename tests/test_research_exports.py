from contextlib import closing
from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import pytest

from wildlife_soundscape.core.config import AppConfig
from wildlife_soundscape.core.provenance import experiment_manifest
from wildlife_soundscape.core.protocol import EnvironmentPayload
from wildlife_soundscape.pipeline.event_detector import AcousticEvent
from wildlife_soundscape.storage.database import EventDatabase
from wildlife_soundscape.tools.export_events import (
    EventExportOptions,
    ExportFormat,
    export_events,
)
from wildlife_soundscape.tools.export_research_metrics import (
    ResearchMetricsExportOptions,
    export_database_session,
)


@pytest.fixture
def research_database(tmp_path):
    config = AppConfig()
    config = replace(
        config,
        persistence=replace(config.persistence, database_path=tmp_path / "events.db"),
    )
    database = EventDatabase(config.persistence.database_path)
    database.start_session(42, "research", manifest=experiment_manifest(config))
    for i in range(10):
        environment = EnvironmentPayload(15 + i, 50 + i, 1013)
        database.add_environment(
            session_id=42,
            node_id=1,
            sample_index=i * 60 * 48000,
            environment=environment,
        )
        if i % 2 == 0:
            event = AcousticEvent(
                event_id=i + 1,
                session_id=42,
                start_sample=i * 60 * 48000,
                end_sample=(i * 60 + 1) * 48000,
                trigger_nodes=(1, 2, 3),
                peak_rms_dbfs=-20,
            )
            database.add_event(
                event,
                environment=environment,
                localization=SimpleNamespace(
                    position=SimpleNamespace(
                        success=True, x=i / 10, y=i / 20, residual_rms_meters=0.01
                    ),
                    speed_of_sound_mps=343,
                ),
                event_directory=None,
            )
    with closing(sqlite3.connect(database.path)) as conn, conn:
        conn.execute(
            "UPDATE sessions SET started_at='2026-01-01 00:00:00', stopped_at='2026-01-01 00:10:00' WHERE session_id=42"
        )
    return database


@pytest.mark.parametrize(
    "fmt", [ExportFormat.JSON, ExportFormat.JSONL, ExportFormat.CSV]
)
def test_event_export_retains_recorded_rate_and_manifest(
    research_database, tmp_path, fmt
):
    output = tmp_path / f"events.{fmt.value}"
    count = export_events(
        EventExportOptions(
            database_path=research_database.path,
            output_path=output,
            export_format=fmt,
            sample_rate=8000,
            session_id=42,
        )
    )
    assert count == 5
    assert output.stat().st_size > 0
    manifest = json.loads(
        output.with_suffix(output.suffix + ".manifest.json").read_text()
    )
    assert manifest["sessions"]["42"]["config"]["audio"]["sample_rate"] == 48000
    if fmt is ExportFormat.JSON:
        rows = json.loads(output.read_text())
        assert rows[0]["duration_s"] == 1


def test_full_research_export_produces_json_csv_and_provenance(
    research_database, tmp_path
):
    result = export_database_session(
        research_database.path,
        session_id=42,
        options=ResearchMetricsExportOptions(output_directory=tmp_path / "export"),
    )
    payload = json.loads(result.json_path.read_text())
    assert payload
    assert result.summary_csv_path is not None and result.summary_csv_path.exists()
    assert result.table_csv_paths and all(
        path.exists() for path in result.table_csv_paths
    )
    assert result.json_path.with_suffix(
        result.json_path.suffix + ".manifest.json"
    ).exists()


def test_environmental_bins_retain_observed_zero_event_periods(research_database):
    rows = research_database.analytics_environmental_bins(
        session_id=42, sample_rate=48000, bucket_seconds=60, node_id=1
    )
    assert len(rows) == 10
    assert sum(row["event_count"] for row in rows) == 5
    assert any(row["event_count"] == 0 for row in rows)
