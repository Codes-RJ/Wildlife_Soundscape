"""
Tests for dashboard soundscape data access and visualization builders.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from wildlife_soundscape.analytics.indices import (
    SoundscapeIndicesConfig,
    SoundscapeIndicesResult,
)
from wildlife_soundscape.dashboard.data_access import DashboardDataAccess
from wildlife_soundscape.dashboard.plots import build_soundscape_indices_timeline
from wildlife_soundscape.storage.database import EventDatabase


def test_dashboard_soundscape_indices_data_access(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    db = EventDatabase(db_path)
    session_id = 123
    db.start_session(session_id, "Dashboard Test Session")

    data_access = DashboardDataAccess(database=db)

    # Empty state
    assert data_access.soundscape_indices(session_id=session_id) == []

    # Insert sample result
    cfg = SoundscapeIndicesConfig()
    result = SoundscapeIndicesResult(
        aci=15.4,
        ndsi=0.45,
        acoustic_entropy=0.72,
        temporal_entropy=0.85,
        spectral_entropy=0.847,
        bioacoustic_index=12.1,
        anthrophony_power=0.01,
        biophony_power=0.05,
        parameters=cfg,
    )
    db.add_soundscape_indices(
        session_id=session_id,
        node_id=1,
        start_sample=0,
        end_sample=48000,
        result=result,
    )

    # Query via data access
    records = data_access.soundscape_indices(session_id=session_id)
    assert len(records) == 1
    rec = records[0]
    assert rec["session_id"] == session_id
    assert rec["node_id"] == 1
    assert pytest.approx(rec["aci"], rel=1e-4) == 15.4
    assert pytest.approx(rec["ndsi"], rel=1e-4) == 0.45


def test_build_soundscape_indices_timeline_empty_and_populated() -> None:
    # Empty figure
    empty_fig = build_soundscape_indices_timeline([])
    assert isinstance(empty_fig, go.Figure)

    # Populated figure
    records = [
        {
            "id": 1,
            "session_id": 100,
            "node_id": 1,
            "start_sample": 0,
            "end_sample": 48000,
            "aci": 12.0,
            "ndsi": 0.5,
            "acoustic_entropy": 0.8,
            "temporal_entropy": 0.9,
            "spectral_entropy": 0.88,
            "bioacoustic_index": 10.0,
            "anthrophony_power": 0.02,
            "biophony_power": 0.06,
            "parameters": {},
            "created_at": "2026-08-30T12:00:00Z",
        },
        {
            "id": 2,
            "session_id": 100,
            "node_id": 2,
            "start_sample": 0,
            "end_sample": 48000,
            "aci": 14.5,
            "ndsi": -0.2,
            "acoustic_entropy": 0.65,
            "temporal_entropy": 0.75,
            "spectral_entropy": 0.86,
            "bioacoustic_index": 8.5,
            "anthrophony_power": 0.04,
            "biophony_power": 0.03,
            "parameters": {},
            "created_at": "2026-08-30T12:01:00Z",
        },
    ]

    fig = build_soundscape_indices_timeline(records)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 4  # ACI, BI, NDSI, Entropy traces
