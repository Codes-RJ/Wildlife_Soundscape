"""Regression tests for the simplified dashboard workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import wave

import numpy as np

from wildlife_soundscape.dashboard.audio_view import (
    build_context_wav_bytes,
    discover_session_recordings,
)
from wildlife_soundscape.dashboard.monitor_view import session_state
from wildlife_soundscape.storage.database import EventDatabase


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def test_continuous_recording_discovery_uses_immutable_session_suffix(tmp_path: Path) -> None:
    directory = tmp_path / "recordings" / "field label_sid_0000002A"
    _write_wav(directory / "node_1.wav", np.arange(1000, dtype=np.int16))
    _write_wav(directory / "node_2.wav", np.arange(2000, dtype=np.int16))

    recordings = discover_session_recordings(
        {"session_id": 42, "label": "field label"},
        recordings_root=tmp_path / "recordings",
        expected_nodes={1, 2, 3},
    )

    assert [item.node_id for item in recordings] == [1, 2]
    assert recordings[1].duration_s == 2.0


def test_context_clip_reports_recording_edge_truncation(tmp_path: Path) -> None:
    path = tmp_path / "node_1.wav"
    _write_wav(path, np.arange(10_000, dtype=np.int16))

    context = build_context_wav_bytes(
        path,
        event_start_sample=500,
        event_end_sample=1500,
        context_before_s=1.0,
        context_after_s=1.0,
    )

    assert context.start_s == 0.0
    assert context.duration_s == 2.5
    assert context.clipped_at_start
    assert not context.clipped_at_end
    with wave.open(str(path), "rb") as source:
        assert source.getnframes() == 10_000


def test_old_unclosed_session_is_displayed_as_interrupted() -> None:
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    session = {"session_id": 42, "started_at": old, "stopped_at": None}
    assert session_state(session, []) == "Interrupted"
    session["stopped_at"] = old
    assert session_state(session, []) == "Completed"


def test_human_review_does_not_replace_model_data(tmp_path: Path) -> None:
    database = EventDatabase(tmp_path / "events.db")
    database.start_session(42, "review")
    with database._connect() as connection:
        connection.execute(
            """
            INSERT INTO events(
                detector_event_id, session_id, start_sample, end_sample,
                trigger_nodes, peak_rms_dbfs
            ) VALUES (1, 42, 0, 1000, '[1,2,3]', -12.0)
            """
        )
        event_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])

    database.save_event_review(
        event_id,
        reviewed_label="amphibian",
        reviewer="field-team",
        notes="Confirmed chorus pattern",
    )

    review = database.get_event_review(event_id)
    assert review is not None
    assert review["reviewed_label"] == "amphibian"
    assert review["reviewer"] == "field-team"
    assert database.list_event_reviews(session_id=42) == [review]
    assert database.get_event(event_id)["classification_label"] is None
