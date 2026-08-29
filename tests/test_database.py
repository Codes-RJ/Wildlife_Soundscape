from pathlib import Path

from database import EventDatabase
from event_detector import AcousticEvent
from protocol import EnvironmentPayload


def test_database_persists_event(tmp_path: Path):
    db = EventDatabase(tmp_path / "events.db")
    db.start_session(10, "test")
    env = EnvironmentPayload(25.0, 60.0, 1008.0)
    db.add_environment(session_id=10, node_id=1, sample_index=100, environment=env)
    event = AcousticEvent(1, 10, 80, 400, (1, 2), -20.0)
    row_id = db.add_event(event, environment=env, localization=None, event_directory=None)
    assert row_id > 0
    rows = db.recent_events()
    assert len(rows) == 1
    assert rows[0]["session_id"] == 10
    db.stop_session(10)
