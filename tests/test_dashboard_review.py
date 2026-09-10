from streamlit.testing.v1 import AppTest


def review_app(path):
    from wildlife_soundscape.dashboard.data_access import DashboardDataAccess
    from wildlife_soundscape.dashboard.review_view import render_review_view
    from wildlife_soundscape.storage.database import EventDatabase

    render_review_view(DashboardDataAccess(database=EventDatabase(path)))


def test_review_empty_database(tmp_path):
    app = AppTest.from_function(review_app, args=(str(tmp_path / "events.db"),)).run()
    assert not app.exception
    assert "No sessions" in app.info[0].value


def test_review_session_switching_and_empty_session(tmp_path):
    from wildlife_soundscape.storage.database import EventDatabase

    path = tmp_path / "events.db"
    database = EventDatabase(path)
    database.start_session(1, "First")
    database.start_session(2, "Second")
    app = AppTest.from_function(review_app, args=(str(path),)).run()
    assert not app.exception
    app.selectbox[0].select(1).run()
    assert not app.exception
    assert any("no detected events" in message.value for message in app.info)
