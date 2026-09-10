"""Regression coverage for padded calls, spatial fixtures and semantic chart colors."""
from dataclasses import replace
from pathlib import Path
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from wildlife_soundscape.classification.classifier import AcousticClass, HeuristicClassifier
from wildlife_soundscape.core.config import CONFIG
from wildlife_soundscape.datasets.demo import demo_points
from wildlife_soundscape.dashboard.palette import class_color
from wildlife_soundscape.dashboard.plots import build_localization_scatter
from wildlife_soundscape.dsp.features import extract_acoustic_features
from wildlife_soundscape.dsp.preprocessing import PreprocessingConfig, preprocess_event_audio
from wildlife_soundscape.runtime.simulator import FakeNode, SharedSimulation
from wildlife_soundscape.tools.populate_demo_session import generate_demo_dataset
from wildlife_soundscape.tools.reprocess_unknown import reprocess_unknown_events


def test_quiet_padding_does_not_turn_a_chirp_into_unknown():
    rate = CONFIG.audio.sample_rate
    node = FakeNode(node_id=1, host="127.0.0.1", port=5001, shared=SharedSimulation())
    waveform = node.event_waveforms[0]
    rng = np.random.default_rng(2026)
    signals = [waveform, np.pad(waveform, (rate // 2, rate))]
    features = [extract_acoustic_features(preprocess_event_audio(
        np.round(signal + rng.normal(0, 75, signal.size)).astype(np.int16),
        PreprocessingConfig(sample_rate=rate),
    ), noise_rms=75 / 32768) for signal in signals]
    assert features[1].spectral_centroid_hz == pytest.approx(features[0].spectral_centroid_hz, rel=0.03)
    assert features[1].zero_crossing_rate == pytest.approx(features[0].zero_crossing_rate, rel=0.03)
    assert HeuristicClassifier().classify(features[1]).label == AcousticClass.BIRD
    assert features[1].duration_s > features[0].duration_s  # original event duration retained


def test_demo_points_cover_both_regions_for_nondefault_geometry():
    nodes = {1: (10.0, 8.0), 2: (13.0, 9.0), 3: (11.0, 12.0)}
    points = demo_points(nodes)
    matrix = np.vstack((np.asarray(list(nodes.values())).T, np.ones(3)))
    assert len(points) == len({(p.x, p.y) for p in points}) == 144
    assert points == demo_points(nodes)
    for point in points:
        weights = np.linalg.solve(matrix, [point.x, point.y, 1])
        assert bool(np.all(weights > 0)) == (point.region == "inside")
    assert sum(p.region == "inside" for p in points) == 72


def test_class_colors_stay_stable_when_filtering_and_triangle_is_visible():
    events = [dict(id=index, x_m=index / 10, y_m=0.2, localization_success=True,
                   classification_label=label) for index, label in enumerate(("bird", "insect", "unknown"))]
    full = build_localization_scatter(events, node_positions=CONFIG.localization.node_positions)
    filtered = build_localization_scatter(events[1:2], node_positions=CONFIG.localization.node_positions)
    for figure in (full, filtered):
        assert figure.layout.paper_bgcolor == "#FFFFFF"
        assert next(trace for trace in figure.data if trace.name == "insect").marker.color == class_color("insect")
        triangle = next(trace for trace in figure.data if trace.name == "Array boundary")
        assert triangle.x[0] == triangle.x[-1] and len(triangle.x) == 4


def test_demo_generation_is_additive_and_balanced(tmp_path: Path):
    config = replace(CONFIG, persistence=replace(CONFIG.persistence,
        database_path=tmp_path / "events.db", events_dir=tmp_path / "events"))
    first = generate_demo_dataset(config, event_count=12)
    second = generate_demo_dataset(config, event_count=12)
    assert first["session_id"] != second["session_id"]
    assert first["regions"] == {"inside": 6, "outside": 6}
    assert all(count == 2 for count in first["classes"].values())
    with closing(sqlite3.connect(config.persistence.database_path)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 24
        assert conn.execute("SELECT count(*) FROM sessions WHERE stopped_at IS NOT NULL").fetchone()[0] == 2
        elapsed = conn.execute("SELECT (julianday(stopped_at)-julianday(started_at))*86400 FROM sessions").fetchall()
        assert all(seconds == pytest.approx(60, abs=0.01) for (seconds,) in elapsed)
    # The recovery tool must never overwrite seeded or model-supplied labels.
    assert reprocess_unknown_events(config, apply=True)["selected"] == 0


def test_reprocessing_has_dry_run_backup_and_is_idempotent(tmp_path: Path):
    config = replace(CONFIG, persistence=replace(CONFIG.persistence,
        database_path=tmp_path / "events.db", events_dir=tmp_path / "events"))
    generate_demo_dataset(config, event_count=12)
    with closing(sqlite3.connect(config.persistence.database_path)) as conn:
        conn.execute("UPDATE classifications SET label='unknown', classifier_name='heuristic_acoustic_classifier' WHERE event_id=1")
        conn.commit()
    dry_run = reprocess_unknown_events(config)
    assert dry_run["selected"] == 1 and dry_run["backup"] is None
    assert not list(tmp_path.glob("*.before-feature-v2.*.db"))
    applied = reprocess_unknown_events(config, apply=True)
    assert applied["failures"] == [] and sum(applied["labels"].values()) == 1
    with closing(sqlite3.connect(applied["backup"])) as backup:
        assert backup.execute("SELECT label FROM classifications WHERE event_id=1").fetchone()[0] == "unknown"
        assert backup.execute("SELECT count(*) FROM event_processing WHERE stage='feature_reprocess_v2'").fetchone()[0] == 0
    assert reprocess_unknown_events(config, apply=True)["selected"] == 0


def test_soundscape_timeline_does_not_connect_different_microphones():
    from wildlife_soundscape.dashboard.plots import build_soundscape_indices_timeline

    rows = [dict(node_id=node, start_sample=sample, aci=node * 10,
                 bioacoustic_index=5, ndsi=0.5, acoustic_entropy=0.7,
                 created_at="2026-09-09T00:00:00Z")
            for sample in (0, 48000) for node in (1, 2, 3)]
    figure = build_soundscape_indices_timeline(rows)
    assert len(figure.data) == 12
    for trace in figure.data:
        assert list(trace.x) == [0, 1]
        assert len(set(trace.y)) == 1


def test_all_dashboard_views_render(tmp_path: Path, monkeypatch):
    import wildlife_soundscape.core.config as config_module
    from streamlit.testing.v1 import AppTest

    config = replace(CONFIG, persistence=replace(CONFIG.persistence,
        database_path=tmp_path / "events.db", events_dir=tmp_path / "events"))
    generate_demo_dataset(config, event_count=12)
    monkeypatch.setattr(config_module, "CONFIG", config)
    script = Path(__file__).resolve().parents[1] / "src/wildlife_soundscape/dashboard/app.py"
    app = AppTest.from_file(str(script), default_timeout=20).run()
    assert not app.exception
    for view in list(app.radio(key="dashboard_navigation").options):
        app.radio(key="dashboard_navigation").set_value(view).run()
        assert not app.exception, [item.message for item in app.exception]
        assert not app.error, [item.value for item in app.error]
