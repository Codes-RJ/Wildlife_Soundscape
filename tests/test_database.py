"""
Tests for database.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. database initialization
    2. session creation
    3. session stopping
    4. session restart/upsert
    5. environmental telemetry persistence
    6. core acoustic-event persistence
    7. environmental metadata attached to events
    8. localization metadata persistence
    9. best-node persistence
    10. event-directory persistence
    11. DSP feature persistence
    12. DSP feature upsert behaviour
    13. classification persistence
    14. nullable secondary classification
    15. classification upsert behaviour
    16. recent-event ordering
    17. recent-event limits
    18. get_event()
    19. joined feature/classification event queries
    20. foreign-key protection
    21. important input validation
    22. classification schema compatibility

Storage architecture
--------------------
Raw event audio is not stored inside SQLite.

    WAV / PCM
        -> filesystem

    event metadata
    environmental data
    localization
    DSP features
    classification
        -> SQLite
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import json
import sqlite3

from contextlib import (
    closing,
)

from pathlib import (
    Path,
)

from types import (
    SimpleNamespace,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.classification.classifier import (
    AcousticClass,
    ClassificationResult,
)

from wildlife_soundscape.storage.database import (
    EventDatabase,
)

from wildlife_soundscape.dsp.features import (
    AcousticFeatures,
)

from wildlife_soundscape.pipeline.event_detector import (
    AcousticEvent,
)

from wildlife_soundscape.core.protocol import (
    EnvironmentPayload,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SESSION_ID = (
    10
)


SECOND_SESSION_ID = (
    20
)


# ======================================================================
# TEST HELPERS
# ======================================================================


def make_database(
    tmp_path: Path,
) -> EventDatabase:
    """
    Create one isolated SQLite database.
    """

    return EventDatabase(
        tmp_path
        / "events.db"
    )


def make_environment() -> EnvironmentPayload:
    """
    Representative BME280 telemetry.
    """

    return EnvironmentPayload(
        25.0,
        60.0,
        1008.0,
    )


def make_event(
    *,
    event_id: int = 1,
    session_id: int = SESSION_ID,
    start_sample: int = 80,
    end_sample: int = 400,
    trigger_nodes: tuple[
        int,
        ...,
    ] = (
        1,
        2,
    ),
    peak_rms_dbfs: float = -20.0,
) -> AcousticEvent:
    """
    Construct one core acoustic event.
    """

    return AcousticEvent(
        event_id,
        session_id,
        start_sample,
        end_sample,
        trigger_nodes,
        peak_rms_dbfs,
    )


def make_features(
    *,
    rms: float = 0.10,
    dominant_frequency_hz: float = 2500.0,
    snr_db: float | None = 15.0,
) -> AcousticFeatures:
    """
    Construct deterministic acoustic features.
    """

    return AcousticFeatures(
        duration_s=
            1.25,

        rms=
            rms,

        peak_amplitude=
            0.25,

        crest_factor=
            2.5,

        zero_crossing_rate=
            0.08,

        dominant_frequency_hz=
            dominant_frequency_hz,

        spectral_centroid_hz=
            3200.0,

        spectral_bandwidth_hz=
            1400.0,

        spectral_rolloff_hz=
            5200.0,

        spectral_flatness=
            0.12,

        spectral_flux=
            0.03,

        snr_db=
            snr_db,

        mfcc_mean=
            tuple(
                float(
                    index
                )
                for index
                in range(
                    13
                )
            ),

        mfcc_std=
            tuple(
                0.5
                for _ in range(
                    13
                )
            ),
    )


def make_classification(
    *,
    label: AcousticClass = AcousticClass.BIRD,
    confidence: float = 0.82,
    second_label: AcousticClass | None = AcousticClass.INSECT,
    second_confidence: float | None = 0.31,
    margin: float = 0.51,
) -> ClassificationResult:
    """
    Construct one valid classification result.
    """

    scores = {
        acoustic_class.value:
            0.0

        for acoustic_class
        in AcousticClass
    }

    scores[
        label.value
    ] = (
        confidence
    )

    if (
        second_label
        is not None
        and second_confidence
        is not None
    ):

        scores[
            second_label.value
        ] = (
            second_confidence
        )

    return ClassificationResult(
        label=
            label,

        confidence=
            confidence,

        second_label=
            second_label,

        second_confidence=
            second_confidence,

        margin=
            margin,

        scores=
            scores,

        reasons=(
            "synthetic unit-test result",
        ),

        classifier_name=
            "test_classifier",

        classifier_version=
            "1.0",
    )


def make_localization(
    *,
    x: float = 0.45,
    y: float = 0.35,
    success: bool = True,
    residual_m: float = 0.025,
    speed_of_sound_mps: float = 344.2,
):
    """
    Lightweight localization object exposing only the public fields that
    EventDatabase persists.

    The nonlinear solver itself is tested separately in test_solver.py.
    """

    position = SimpleNamespace(
        x=
            x,

        y=
            y,

        success=
            success,

        residual_rms_meters=
            residual_m,
    )

    return SimpleNamespace(
        position=
            position,

        speed_of_sound_mps=
            speed_of_sound_mps,
    )


def test_failed_nonfinite_localization_does_not_discard_event(
    tmp_path,
) -> None:
    database = make_database(
        tmp_path
    )
    database.start_session(
        SESSION_ID,
        "failed-localization",
    )

    event_id = database.add_event(
        make_event(),
        environment=None,
        localization=make_localization(
            x=float("nan"),
            y=float("nan"),
            success=False,
            residual_m=float("nan"),
        ),
        event_directory=None,
    )

    row = query_one(
        database,
        "SELECT * FROM events WHERE id = ?",
        (event_id,),
    )

    assert row["localization_success"] == 0
    assert row["x_m"] is None
    assert row["y_m"] is None
    assert row["localization_residual_m"] is None


def query_one(
    database: EventDatabase,
    sql: str,
    parameters: tuple = (),
):
    """
    Read one raw SQLite row for schema-level persistence checks.
    """

    with closing(
        sqlite3.connect(
            database.path
        )
    ) as connection:

        connection.row_factory = (
            sqlite3.Row
        )

        return connection.execute(
            sql,
            parameters,
        ).fetchone()


def query_all(
    database: EventDatabase,
    sql: str,
    parameters: tuple = (),
):
    """
    Read raw SQLite rows.
    """

    with closing(
        sqlite3.connect(
            database.path
        )
    ) as connection:

        connection.row_factory = (
            sqlite3.Row
        )

        return connection.execute(
            sql,
            parameters,
        ).fetchall()


# ======================================================================
# INITIALIZATION
# ======================================================================


def test_database_initializes_required_tables(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    rows = query_all(
        database,
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """,
    )

    table_names = {
        row[
            "name"
        ]
        for row
        in rows
    }

    assert (
        "sessions"
        in table_names
    )

    assert (
        "telemetry"
        in table_names
    )

    assert (
        "events"
        in table_names
    )

    assert (
        "event_features"
        in table_names
    )

    assert (
        "classifications"
        in table_names
    )


def test_database_file_is_created(
    tmp_path: Path,
) -> None:

    path = (
        tmp_path
        / "nested"
        / "wildlife"
        / "events.db"
    )

    EventDatabase(
        path
    )

    assert (
        path.exists()
    )


# ======================================================================
# SESSION LIFECYCLE
# ======================================================================


def test_start_session_persists_session(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test session",
    )

    row = query_one(
        database,
        """
        SELECT *
        FROM sessions
        WHERE session_id = ?
        """,
        (
            SESSION_ID,
        ),
    )

    assert (
        row
        is not None
    )

    assert (
        row[
            "session_id"
        ]
        == SESSION_ID
    )

    assert (
        row[
            "label"
        ]
        == "test session"
    )

    assert (
        row[
            "started_at"
        ]
        is not None
    )

    assert (
        row[
            "stopped_at"
        ]
        is None
    )


def test_stop_session_sets_timestamp(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    database.stop_session(
        SESSION_ID
    )

    row = query_one(
        database,
        """
        SELECT stopped_at
        FROM sessions
        WHERE session_id = ?
        """,
        (
            SESSION_ID,
        ),
    )

    assert (
        row[
            "stopped_at"
        ]
        is not None
    )


def test_stop_none_session_is_safe(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.stop_session(
        None
    )


def test_restarting_existing_session_updates_label_and_reopens_it(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "first",
    )

    database.stop_session(
        SESSION_ID
    )

    database.start_session(
        SESSION_ID,
        "second",
    )

    row = query_one(
        database,
        """
        SELECT *
        FROM sessions
        WHERE session_id = ?
        """,
        (
            SESSION_ID,
        ),
    )

    assert (
        row[
            "label"
        ]
        == "second"
    )

    assert (
        row[
            "stopped_at"
        ]
        is None
    )


def test_start_session_rejects_zero_session_id(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):

        database.start_session(
            0,
            "test",
        )


def test_start_session_rejects_empty_label(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):

        database.start_session(
            SESSION_ID,
            "   ",
        )


# ======================================================================
# ENVIRONMENTAL TELEMETRY
# ======================================================================


def test_add_environment_persists_telemetry(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    environment = (
        make_environment()
    )

    database.add_environment(
        session_id=
            SESSION_ID,

        node_id=
            1,

        sample_index=
            100,

        environment=
            environment,
    )

    row = query_one(
        database,
        """
        SELECT *
        FROM telemetry
        WHERE session_id = ?
        """,
        (
            SESSION_ID,
        ),
    )

    assert (
        row
        is not None
    )

    assert (
        row[
            "node_id"
        ]
        == 1
    )

    assert (
        row[
            "sample_index"
        ]
        == 100
    )

    assert (
        row[
            "temperature_c"
        ]
        == pytest.approx(
            25.0
        )
    )

    assert (
        row[
            "humidity_percent"
        ]
        == pytest.approx(
            60.0
        )
    )

    assert (
        row[
            "pressure_hpa"
        ]
        == pytest.approx(
            1008.0
        )
    )


def test_add_environment_rejects_negative_sample_index(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):

        database.add_environment(
            session_id=
                SESSION_ID,

            node_id=
                1,

            sample_index=
                -1,

            environment=
                make_environment(),
        )


# ======================================================================
# BASIC EVENT PERSISTENCE
# ======================================================================


def test_database_persists_event(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    environment = (
        make_environment()
    )

    database.add_environment(
        session_id=
            SESSION_ID,

        node_id=
            1,

        sample_index=
            100,

        environment=
            environment,
    )

    event = make_event()

    row_id = database.add_event(
        event,
        environment=
            environment,

        localization=
            None,

        event_directory=
            None,
    )

    assert (
        row_id
        > 0
    )

    rows = (
        database.recent_events()
    )

    assert (
        len(
            rows
        )
        == 1
    )

    assert (
        rows[
            0
        ][
            "session_id"
        ]
        == SESSION_ID
    )

    database.stop_session(
        SESSION_ID
    )


def test_event_core_fields_are_persisted(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event = make_event(
        event_id=
            17,

        start_sample=
            1234,

        end_sample=
            5678,

        trigger_nodes=(
            3,
            1,
            2,
        ),

        peak_rms_dbfs=
            -12.5,
    )

    row_id = database.add_event(
        event,
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    row = database.get_event(
        row_id
    )

    assert (
        row
        is not None
    )

    assert (
        row[
            "detector_event_id"
        ]
        == 17
    )

    assert (
        row[
            "start_sample"
        ]
        == 1234
    )

    assert (
        row[
            "end_sample"
        ]
        == 5678
    )

    assert (
        row[
            "peak_rms_dbfs"
        ]
        == pytest.approx(
            -12.5
        )
    )

    # Database canonicalizes trigger-node order.
    assert (
        json.loads(
            row[
                "trigger_nodes"
            ]
        )
        == [
            1,
            2,
            3,
        ]
    )


# ======================================================================
# EVENT ENVIRONMENT
# ======================================================================


def test_event_environment_is_persisted(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    environment = (
        make_environment()
    )

    event_id = database.add_event(
        make_event(),
        environment=
            environment,

        localization=
            None,

        event_directory=
            None,
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "temperature_c"
        ]
        == pytest.approx(
            25.0
        )
    )

    assert (
        row[
            "humidity_percent"
        ]
        == pytest.approx(
            60.0
        )
    )

    assert (
        row[
            "pressure_hpa"
        ]
        == pytest.approx(
            1008.0
        )
    )


def test_event_without_environment_stores_nulls(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "temperature_c"
        ]
        is None
    )

    assert (
        row[
            "humidity_percent"
        ]
        is None
    )

    assert (
        row[
            "pressure_hpa"
        ]
        is None
    )


# ======================================================================
# LOCALIZATION
# ======================================================================


def test_event_localization_is_persisted(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    localization = make_localization(
        x=
            0.42,

        y=
            0.31,

        success=
            True,

        residual_m=
            0.018,

        speed_of_sound_mps=
            344.5,
    )

    event_id = database.add_event(
        make_event(),
        environment=
            make_environment(),

        localization=
            localization,

        event_directory=
            None,
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "speed_of_sound_mps"
        ]
        == pytest.approx(
            344.5
        )
    )

    assert (
        row[
            "x_m"
        ]
        == pytest.approx(
            0.42
        )
    )

    assert (
        row[
            "y_m"
        ]
        == pytest.approx(
            0.31
        )
    )

    assert (
        row[
            "localization_success"
        ]
        == 1
    )

    assert (
        row[
            "localization_residual_m"
        ]
        == pytest.approx(
            0.018
        )
    )


def test_event_without_localization_stores_null_localization(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "speed_of_sound_mps"
        ]
        is None
    )

    assert (
        row[
            "x_m"
        ]
        is None
    )

    assert (
        row[
            "y_m"
        ]
        is None
    )

    assert (
        row[
            "localization_success"
        ]
        is None
    )


# ======================================================================
# BEST NODE / EVENT DIRECTORY
# ======================================================================


def test_best_node_and_event_directory_are_persisted(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            "data/events/session_10/event_1",

        best_node_id=
            3,
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "best_node_id"
        ]
        == 3
    )

    assert (
        row[
            "event_directory"
        ]
        == "data/events/session_10/event_1"
    )


# ======================================================================
# DSP FEATURES
# ======================================================================


def test_event_features_are_persisted_and_joined(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    features = make_features()

    database.add_event_features(
        event_id=
            event_id,

        source_node_id=
            2,

        features=
            features,
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "feature_source_node_id"
        ]
        == 2
    )

    assert (
        row[
            "duration_s"
        ]
        == pytest.approx(
            features.duration_s
        )
    )

    assert (
        row[
            "rms"
        ]
        == pytest.approx(
            features.rms
        )
    )

    assert (
        row[
            "dominant_frequency_hz"
        ]
        == pytest.approx(
            features.dominant_frequency_hz
        )
    )

    assert (
        json.loads(
            row[
                "mfcc_mean_json"
            ]
        )
        == list(
            features.mfcc_mean
        )
    )

    assert (
        json.loads(
            row[
                "mfcc_std_json"
            ]
        )
        == list(
            features.mfcc_std
        )
    )


def test_event_features_allow_null_snr(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    database.add_event_features(
        event_id=
            event_id,

        source_node_id=
            1,

        features=
            make_features(
                snr_db=
                    None
            ),
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "snr_db"
        ]
        is None
    )


def test_event_features_are_upserted(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    database.add_event_features(
        event_id=
            event_id,

        source_node_id=
            1,

        features=
            make_features(
                rms=
                    0.10,

                dominant_frequency_hz=
                    2000.0,
            ),
    )

    database.add_event_features(
        event_id=
            event_id,

        source_node_id=
            3,

        features=
            make_features(
                rms=
                    0.25,

                dominant_frequency_hz=
                    4200.0,
            ),
    )

    rows = query_all(
        database,
        """
        SELECT *
        FROM event_features
        WHERE event_id = ?
        """,
        (
            event_id,
        ),
    )

    assert (
        len(
            rows
        )
        == 1
    )

    assert (
        rows[
            0
        ][
            "source_node_id"
        ]
        == 3
    )

    assert (
        rows[
            0
        ][
            "rms"
        ]
        == pytest.approx(
            0.25
        )
    )

    assert (
        rows[
            0
        ][
            "dominant_frequency_hz"
        ]
        == pytest.approx(
            4200.0
        )
    )


# ======================================================================
# CLASSIFICATION
# ======================================================================


def test_classification_is_persisted_and_joined(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    result = make_classification()

    database.add_classification(
        event_id=
            event_id,

        result=
            result,
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "classification_label"
        ]
        == "bird"
    )

    assert (
        row[
            "classification_confidence"
        ]
        == pytest.approx(
            0.82
        )
    )

    assert (
        row[
            "classification_second_label"
        ]
        == "insect"
    )

    assert (
        row[
            "classification_second_confidence"
        ]
        == pytest.approx(
            0.31
        )
    )

    assert (
        row[
            "classification_margin"
        ]
        == pytest.approx(
            0.51
        )
    )

    assert (
        row[
            "classifier_name"
        ]
        == "test_classifier"
    )

    assert (
        row[
            "classifier_version"
        ]
        == "1.0"
    )


def test_classification_json_fields_are_persisted(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    result = make_classification()

    database.add_classification(
        event_id=
            event_id,

        result=
            result,
    )

    row = database.get_event(
        event_id
    )

    stored_scores = json.loads(
        row[
            "classification_scores_json"
        ]
    )

    stored_reasons = json.loads(
        row[
            "classification_reasons_json"
        ]
    )

    assert (
        stored_scores[
            "bird"
        ]
        == pytest.approx(
            0.82
        )
    )

    assert (
        stored_scores[
            "insect"
        ]
        == pytest.approx(
            0.31
        )
    )

    assert (
        stored_reasons
        == [
            "synthetic unit-test result"
        ]
    )


# ======================================================================
# NULLABLE SECONDARY CLASSIFICATION
# ======================================================================


def test_classification_allows_missing_secondary_candidate(
    tmp_path: Path,
) -> None:
    """
    Critical regression test.

    UNKNOWN/invalid classifications may legitimately contain:

        second_label = None
        second_confidence = None

    SQLite must preserve those NULL values rather than requiring a fake
    numerical secondary confidence.
    """

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    result = make_classification(
        label=
            AcousticClass.UNKNOWN,

        confidence=
            1.0,

        second_label=
            None,

        second_confidence=
            None,

        margin=
            0.0,
    )

    database.add_classification(
        event_id=
            event_id,

        result=
            result,
    )

    row = database.get_event(
        event_id
    )

    assert (
        row[
            "classification_label"
        ]
        == "unknown"
    )

    assert (
        row[
            "classification_second_label"
        ]
        is None
    )

    assert (
        row[
            "classification_second_confidence"
        ]
        is None
    )


def test_second_confidence_database_column_is_nullable(
    tmp_path: Path,
) -> None:
    """
    Schema-level protection for the ClassificationResult contract.
    """

    database = make_database(
        tmp_path
    )

    rows = query_all(
        database,
        """
        PRAGMA table_info(classifications)
        """,
    )

    column = next(
        row

        for row
        in rows

        if row[
            "name"
        ]
        == "second_confidence"
    )

    # SQLite PRAGMA table_info:
    #
    # notnull = 0  -> NULL allowed
    # notnull = 1  -> NOT NULL
    assert (
        column[
            "notnull"
        ]
        == 0
    )


def test_classification_is_upserted(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            None,

        localization=
            None,

        event_directory=
            None,
    )

    database.add_classification(
        event_id=
            event_id,

        result=
            make_classification(
                label=
                    AcousticClass.BIRD,

                confidence=
                    0.80,

                second_label=
                    AcousticClass.INSECT,

                second_confidence=
                    0.30,

                margin=
                    0.50,
            ),
    )

    database.add_classification(
        event_id=
            event_id,

        result=
            make_classification(
                label=
                    AcousticClass.NOISE,

                confidence=
                    0.90,

                second_label=
                    AcousticClass.BIRD,

                second_confidence=
                    0.20,

                margin=
                    0.70,
            ),
    )

    rows = query_all(
        database,
        """
        SELECT *
        FROM classifications
        WHERE event_id = ?
        """,
        (
            event_id,
        ),
    )

    assert (
        len(
            rows
        )
        == 1
    )

    assert (
        rows[
            0
        ][
            "label"
        ]
        == "noise"
    )

    assert (
        rows[
            0
        ][
            "confidence"
        ]
        == pytest.approx(
            0.90
        )
    )


# ======================================================================
# JOINED EVENT QUERY
# ======================================================================


def test_get_event_joins_core_features_and_classification(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event_id = database.add_event(
        make_event(),
        environment=
            make_environment(),

        localization=
            make_localization(),

        event_directory=
            "data/test/event_1",

        best_node_id=
            2,
    )

    database.add_event_features(
        event_id=
            event_id,

        source_node_id=
            2,

        features=
            make_features(),
    )

    database.add_classification(
        event_id=
            event_id,

        result=
            make_classification(),
    )

    row = database.get_event(
        event_id
    )

    assert (
        row
        is not None
    )

    # Core event.
    assert (
        row[
            "session_id"
        ]
        == SESSION_ID
    )

    # Features.
    assert (
        row[
            "feature_source_node_id"
        ]
        == 2
    )

    assert (
        row[
            "rms"
        ]
        == pytest.approx(
            0.10
        )
    )

    # Classification.
    assert (
        row[
            "classification_label"
        ]
        == "bird"
    )

    # Environment.
    assert (
        row[
            "temperature_c"
        ]
        == pytest.approx(
            25.0
        )
    )

    # Localization.
    assert (
        row[
            "x_m"
        ]
        == pytest.approx(
            0.45
        )
    )


# ======================================================================
# RECENT EVENTS
# ======================================================================


def test_recent_events_are_newest_first(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    database.add_event(
        make_event(
            event_id=
                1,
            start_sample=
                100,
            end_sample=
                200,
        ),
        environment=
            None,
        localization=
            None,
        event_directory=
            None,
    )

    database.add_event(
        make_event(
            event_id=
                2,
            start_sample=
                300,
            end_sample=
                400,
        ),
        environment=
            None,
        localization=
            None,
        event_directory=
            None,
    )

    database.add_event(
        make_event(
            event_id=
                3,
            start_sample=
                500,
            end_sample=
                600,
        ),
        environment=
            None,
        localization=
            None,
        event_directory=
            None,
    )

    rows = database.recent_events(
        limit=
            10
    )

    assert [
        row[
            "detector_event_id"
        ]
        for row
        in rows
    ] == [
        3,
        2,
        1,
    ]


def test_recent_events_respects_limit(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    for event_id in range(
        1,
        6,
    ):

        database.add_event(
            make_event(
                event_id=
                    event_id,

                start_sample=
                    event_id
                    * 1000,

                end_sample=
                    event_id
                    * 1000
                    + 100,
            ),
            environment=
                None,
            localization=
                None,
            event_directory=
                None,
        )

    rows = database.recent_events(
        limit=
            2
    )

    assert (
        len(
            rows
        )
        == 2
    )

    assert [
        row[
            "detector_event_id"
        ]
        for row
        in rows
    ] == [
        5,
        4,
    ]


def test_recent_events_nonpositive_limit_returns_empty(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    assert (
        database.recent_events(
            limit=
                0
        )
        == []
    )


# ======================================================================
# GET EVENT
# ======================================================================


def test_get_event_returns_none_for_missing_event(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    assert (
        database.get_event(
            99999
        )
        is None
    )


def test_get_event_returns_none_for_nonpositive_id(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    assert (
        database.get_event(
            0
        )
        is None
    )


# ======================================================================
# FOREIGN KEY PROTECTION
# ======================================================================


def test_event_requires_existing_session(
    tmp_path: Path,
) -> None:
    """
    events.session_id has a foreign-key relationship to sessions.
    """

    database = make_database(
        tmp_path
    )

    event = make_event(
        session_id=
            SESSION_ID
    )

    with pytest.raises(
        sqlite3.IntegrityError
    ):

        database.add_event(
            event,
            environment=
                None,
            localization=
                None,
            event_directory=
                None,
        )


def test_features_require_existing_event(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    with pytest.raises(
        sqlite3.IntegrityError
    ):

        database.add_event_features(
            event_id=
                999,

            source_node_id=
                1,

            features=
                make_features(),
        )


def test_classification_requires_existing_event(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    with pytest.raises(
        sqlite3.IntegrityError
    ):

        database.add_classification(
            event_id=
                999,

            result=
                make_classification(),
        )


# ======================================================================
# EVENT INPUT VALIDATION
# ======================================================================


def test_event_rejects_negative_start_sample(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event = make_event(
        start_sample=
            -1,

        end_sample=
            100,
    )

    with pytest.raises(
        ValueError
    ):

        database.add_event(
            event,
            environment=
                None,

            localization=
                None,

            event_directory=
                None,
        )


def test_event_rejects_end_before_start(
    tmp_path: Path,
) -> None:

    database = make_database(
        tmp_path
    )

    database.start_session(
        SESSION_ID,
        "test",
    )

    event = make_event(
        start_sample=
            500,

        end_sample=
            400,
    )

    with pytest.raises(
        ValueError
    ):

        database.add_event(
            event,
            environment=
                None,

            localization=
                None,

            event_directory=
                None,
        )


# ======================================================================
# DATABASE INSTANCE REOPEN
# ======================================================================


def test_database_data_survives_reopening(
    tmp_path: Path,
) -> None:

    path = (
        tmp_path
        / "events.db"
    )

    first = EventDatabase(
        path
    )

    first.start_session(
        SESSION_ID,
        "test",
    )

    event_id = first.add_event(
        make_event(),
        environment=
            make_environment(),

        localization=
            None,

        event_directory=
            None,
    )

    second = EventDatabase(
        path
    )

    row = second.get_event(
        event_id
    )

    assert (
        row
        is not None
    )

    assert (
        row[
            "session_id"
        ]
        == SESSION_ID
    )
