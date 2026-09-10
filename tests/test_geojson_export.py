"""
Tests for geodetic spatial transformations and GeoJSON export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wildlife_soundscape.storage.database import EventDatabase
from wildlife_soundscape.tools.export_events import (
    EventExportOptions,
    ExportFormat,
    export_events,
    local_to_geodetic,
    write_geojson,
)


# ======================================================================
# GEODETIC TRANSFORMATION TESTS
# ======================================================================


def test_local_to_geodetic_origin_and_offsets() -> None:
    origin_lat, origin_lon = 45.0, 10.0

    # Origin point (0, 0)
    lon0, lat0 = local_to_geodetic(
        0.0,
        0.0,
        origin_latitude=origin_lat,
        origin_longitude=origin_lon,
        azimuth_deg=0.0,
    )
    assert pytest.approx(lat0, abs=1e-6) == origin_lat
    assert pytest.approx(lon0, abs=1e-6) == origin_lon

    # North offset (x=0, y=1000m, azimuth=0 deg)
    lon_n, lat_n = local_to_geodetic(
        0.0,
        1000.0,
        origin_latitude=origin_lat,
        origin_longitude=origin_lon,
        azimuth_deg=0.0,
    )
    assert lat_n > origin_lat
    assert pytest.approx(lon_n, abs=1e-5) == origin_lon

    # East offset (x=1000m, y=0, azimuth=0 deg)
    lon_e, lat_e = local_to_geodetic(
        1000.0,
        0.0,
        origin_latitude=origin_lat,
        origin_longitude=origin_lon,
        azimuth_deg=0.0,
    )
    assert lon_e > origin_lon
    assert pytest.approx(lat_e, abs=1e-5) == origin_lat

    # 90-degree azimuth rotation: forward (y=1000m) now points East
    lon_rot, lat_rot = local_to_geodetic(
        0.0,
        1000.0,
        origin_latitude=origin_lat,
        origin_longitude=origin_lon,
        azimuth_deg=90.0,
    )
    assert lon_rot > origin_lon
    assert pytest.approx(lat_rot, abs=1e-5) == origin_lat


# ======================================================================
# WRITE GEOJSON TESTS
# ======================================================================


def test_write_geojson_feature_collection(tmp_path: Path) -> None:
    out_file = tmp_path / "events.geojson"
    origin_lat, origin_lon = 52.5200, 13.4050

    rows = [
        {
            "event_id": 1,
            "has_localization": True,
            "x_m": 5.0,
            "y_m": 10.0,
            "label": "BIRD",
            "confidence": 0.92,
        },
        {
            "event_id": 2,
            "has_localization": False,
            "x_m": None,
            "y_m": None,
            "label": "INSECT",
            "confidence": 0.75,
        },
    ]

    count = write_geojson(
        rows,
        out_file,
        origin_latitude=origin_lat,
        origin_longitude=origin_lon,
        azimuth_deg=0.0,
        coarsen_decimals=4,
    )
    assert count == 1  # only event 1 localized

    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    feature = data["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Point"
    assert len(feature["geometry"]["coordinates"]) == 2
    assert feature["properties"]["event_id"] == 1
    assert feature["properties"]["label"] == "BIRD"


def test_geojson_export_options_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    out_file = tmp_path / "events.geojson"

    # Missing origin coordinates
    with pytest.raises(
        ValueError, match="origin_latitude and origin_longitude are required"
    ):
        EventExportOptions(
            database_path=db_path,
            output_path=out_file,
            export_format=ExportFormat.GEOJSON,
            sample_rate=48000,
            origin_latitude=None,
            origin_longitude=None,
        )


def test_full_export_events_geojson(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    db = EventDatabase(db_path)
    session_id = 1
    db.start_session(session_id, "GeoJSON Test Session")

    # Add localized mock event
    from wildlife_soundscape.localization.solver import PositionResult

    class MockLocalization:
        position = PositionResult(
            x=12.5,
            y=8.0,
            success=True,
            residual_rms_seconds=0.001,
            residual_rms_meters=0.34,
            cost=0.05,
            nfev=5,
            message="optimal",
        )
        tdoa_measurements: list[object] = []
        node_positions: dict[int, tuple[float, float]] = {}
        speed_of_sound_mps = 343.0

    from wildlife_soundscape.pipeline.event_detector import AcousticEvent

    event = AcousticEvent(
        event_id=1,
        session_id=session_id,
        start_sample=0,
        end_sample=48000,
        trigger_nodes=(1, 2, 3),
        peak_rms_dbfs=-15.0,
    )
    db.add_event(
        event=event,
        environment=None,
        localization=MockLocalization(),
        event_directory=None,
    )

    out_file = tmp_path / "events.geojson"
    options = EventExportOptions(
        database_path=db_path,
        output_path=out_file,
        export_format=ExportFormat.GEOJSON,
        sample_rate=48000,
        session_id=session_id,
        origin_latitude=37.7749,
        origin_longitude=-122.4194,
        azimuth_deg=45.0,
    )
    exported = export_events(options)
    assert exported == 1
    assert out_file.exists()

    with out_file.open("r", encoding="utf-8") as f:
        geojson = json.load(f)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    coords = geojson["features"][0]["geometry"]["coordinates"]
    assert len(coords) == 2
    assert -180.0 <= coords[0] <= 180.0
    assert -90.0 <= coords[1] <= 90.0
