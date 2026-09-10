from dataclasses import replace
from pathlib import Path

from wildlife_soundscape.acquisition.stream_manager import StreamManager
from wildlife_soundscape.core.config import AppConfig
from wildlife_soundscape.pipeline.event_detector import AcousticEvent
from wildlife_soundscape.pipeline.event_pipeline import EventPipeline


def test_directory_failure_preserves_core_event_and_marks_audio_failed(
    tmp_path, monkeypatch
):
    config = AppConfig()
    config = replace(
        config,
        persistence=replace(
            config.persistence,
            database_path=tmp_path / "events.db",
            events_dir=tmp_path / "audio",
            save_event_wav=True,
        ),
    )
    pipeline = EventPipeline(StreamManager(config.audio), config)
    pipeline.start_session(123, "same-label")
    event = AcousticEvent(
        event_id=1,
        session_id=123,
        start_sample=0,
        end_sample=4800,
        trigger_nodes=(1, 2, 3),
        peak_rms_dbfs=-15,
    )

    def fail_mkdir(*args, **kwargs):
        # The core record must already be durable before any filesystem work.
        assert pipeline.database.recent_events(limit=1)
        raise OSError("audio volume unavailable")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    pipeline._persist_event(event)
    row = pipeline.database.recent_events(limit=1)[0]
    statuses = {
        item["stage"]: item
        for item in pipeline.database.get_event_processing_status(row["id"])
    }
    assert statuses["audio"]["status"] == "failed"
    assert "audio volume unavailable" in statuses["audio"]["detail"]
    assert statuses["processing"]["status"] == "complete"
    assert "session_0000007B" in row["event_directory"]


def test_core_event_survives_unexpected_analysis_exception(tmp_path, monkeypatch):
    config = AppConfig()
    config = replace(
        config,
        persistence=replace(
            config.persistence,
            database_path=tmp_path / "events.db",
            save_event_wav=False,
        ),
    )
    pipeline = EventPipeline(StreamManager(config.audio), config)
    pipeline.start_session(9, "test")
    event = AcousticEvent(
        event_id=1,
        session_id=9,
        start_sample=0,
        end_sample=4800,
        trigger_nodes=(1,),
        peak_rms_dbfs=-15,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("unexpected backend failure")

    monkeypatch.setattr(pipeline, "_classify_event", fail)
    import pytest

    with pytest.raises(RuntimeError):
        pipeline._persist_event(event)
    row = pipeline.database.recent_events(limit=1)[0]
    assert row["session_id"] == 9
    statuses = pipeline.database.get_event_processing_status(row["id"])
    assert {s["stage"]: s["status"] for s in statuses}["processing"] == "failed"
