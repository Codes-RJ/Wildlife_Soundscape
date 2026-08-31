"""
End-to-end acoustic-event integration test.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Verify that the complete simulated event-processing chain works across
real project components.

Integration path
----------------

    FakeNode 1 ─┐
    FakeNode 2 ─┼── TCP / Protocol-v4
    FakeNode 3 ─┘
             ↓
       ReceiverServer
             ↓
       StreamManager
             ↓
    MultiNodeEventDetector
             ↓
        AcousticEvent
             ↓
       EventPipeline
        ├── environment
        ├── localization
        ├── DSP
        ├── classification
        ├── event WAV storage
        └── SQLite persistence
             ↓
      persisted event row

Scientific scope
----------------
This is an integration smoke test.

It verifies that a known synthetic acoustic event can travel through
the complete software architecture and become a persisted event.

It does NOT measure:

    detector precision / recall
    classification accuracy
    localization accuracy distribution
    real microphone performance

Those require dedicated benchmark and hardware-validation experiments.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import asyncio
import contextlib
import math
import socket
import sqlite3
import sys
import wave

from dataclasses import (
    replace,
)

from pathlib import (
    Path,
)


# ======================================================================
# PROJECT ROOT
# ======================================================================
#
# conftest.py already inserts the root when pytest is used.
#
# Keeping this small bootstrap also allows:
#
#     python tests/test_integration_events.py
#
# to work directly.
# ======================================================================


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)


if (
    str(
        PROJECT_ROOT
    )
    not in sys.path
):

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from config import (
    CONFIG,
)

from server import (
    ReceiverServer,
)

from simulator import (
    FakeNode,
    SharedSimulation,
)


# ======================================================================
# CONSTANTS
# ======================================================================


TEST_HOST = (
    "127.0.0.1"
)


NODE_IDS = (
    1,
    2,
    3,
)


SESSION_LABEL = (
    "integration_events"
)


# ----------------------------------------------------------------------
# The simulator emits background audio before the synthetic acoustic
# event. The detector also needs release/post-padding time before an
# AcousticEvent becomes complete.
#
# Therefore use bounded polling rather than a fixed 3.4-second sleep.
# ----------------------------------------------------------------------


EVENT_TIMEOUT_S = (
    8.0
)


POLL_INTERVAL_S = (
    0.05
)


# ======================================================================
# TCP PORT
# ======================================================================


def find_available_tcp_port() -> int:
    """
    Ask the operating system for an unused localhost TCP port.

    This avoids hard-coding 55003 and prevents collisions with another
    receiver/test process.
    """

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as temporary_socket:

        temporary_socket.bind(
            (
                TEST_HOST,
                0,
            )
        )

        port = int(
            temporary_socket
            .getsockname()[
                1
            ]
        )

    return (
        port
    )


# ======================================================================
# TEST CONFIGURATION
# ======================================================================


def make_integration_config(
    tmp_path: Path,
    *,
    port: int,
):
    """
    Derive an isolated test configuration from the complete production
    CONFIG.

    This is preferable to manually reconstructing AppConfig because new
    configuration sections such as:

        DSP
        classification
        detection
        localization
        persistence

    remain intact automatically.
    """

    # ==============================================================
    # NETWORK
    # ==============================================================

    network = replace(
        CONFIG.network,

        host=
            TEST_HOST,

        port=
            port,
    )

    # ==============================================================
    # AUDIO
    # ==============================================================
    #
    # Continuous session recording is not required here.
    #
    # Event-specific WAV storage remains enabled through persistence.
    # ==============================================================

    audio = replace(
        CONFIG.audio,

        record_wav=
            False,
    )

    # ==============================================================
    # PERSISTENCE
    # ==============================================================

    persistence = replace(
        CONFIG.persistence,

        database_path=
            tmp_path
            / "database"
            / "events.db",

        events_dir=
            tmp_path
            / "events",

        save_event_wav=
            True,
    )

    # ==============================================================
    # CONTINUOUS SOUNDSCAPE ANALYTICS
    # ==============================================================
    # A one-second window proves that receiver audio reaches the persistent
    # soundscape path without running CPU-heavy spectral analysis for every
    # 21 ms network block.

    analytics = replace(
        CONFIG.analytics,

        soundscape_enabled=
            True,

        soundscape_window_seconds=
            1.0,
    )

    # ==============================================================
    # COMPLETE APPLICATION CONFIGURATION
    # ==============================================================

    return replace(
        CONFIG,

        network=
            network,

        audio=
            audio,

        persistence=
            persistence,

        analytics=
            analytics,

        recordings_dir=
            tmp_path
            / "recordings",

        print_status_every_s=
            60.0,
    )


# ======================================================================
# NODE AUDIO STATE
# ======================================================================


def all_nodes_have_audio(
    server: ReceiverServer,
) -> bool:
    """
    True only after every expected node has delivered AUDIO packets.
    """

    for node_id in NODE_IDS:

        connection = (
            server.connections.get(
                node_id
            )
        )

        if (
            connection
            is None
        ):

            return (
                False
            )

        if (
            connection.state
            .audio_packets_received
            <= 0
        ):

            return (
                False
            )

    return (
        True
    )


async def wait_for_all_node_audio(
    server: ReceiverServer,
    *,
    timeout_s: float,
) -> None:
    """
    Wait until all three simulated microphones are streaming.
    """

    loop = (
        asyncio.get_running_loop()
    )

    deadline = (
        loop.time()
        + timeout_s
    )

    while (
        not all_nodes_have_audio(
            server
        )
    ):

        if (
            loop.time()
            >= deadline
        ):

            counts = {
                node_id:
                    (
                        server.connections[
                            node_id
                        ]
                        .state
                        .audio_packets_received

                        if node_id
                        in server.connections

                        else None
                    )

                for node_id
                in NODE_IDS
            }

            raise AssertionError(
                (
                    "Timed out waiting for "
                    "three-node audio streaming. "
                    f"Packet counts: {counts}"
                )
            )

        await asyncio.sleep(
            POLL_INTERVAL_S
        )


# ======================================================================
# EVENT POLLING
# ======================================================================


async def wait_for_persisted_event(
    server: ReceiverServer,
    *,
    timeout_s: float,
):
    """
    Wait until EventPipeline persists at least one detected event.

    Returns the newest persisted database row.
    """

    loop = (
        asyncio.get_running_loop()
    )

    deadline = (
        loop.time()
        + timeout_s
    )

    while True:

        rows = (
            server.events.database
            .recent_events(
                10
            )
        )

        if (
            rows
        ):

            return (
                rows[
                    0
                ]
            )

        if (
            loop.time()
            >= deadline
        ):

            packet_counts = {
                node_id:
                    (
                        server.connections[
                            node_id
                        ]
                        .state
                        .audio_packets_received

                        if node_id
                        in server.connections

                        else None
                    )

                for node_id
                in NODE_IDS
            }

            raise AssertionError(
                (
                    "Timed out waiting for a "
                    "detected/persisted synthetic event. "
                    f"completed_events="
                    f"{server.events.completed_events}, "
                    f"packet_counts={packet_counts}"
                )
            )

        await asyncio.sleep(
            POLL_INTERVAL_S
        )


# ======================================================================
# EVENT DIRECTORY / WAV VALIDATION
# ======================================================================


def validate_event_wav(
    path: Path,
    *,
    expected_sample_rate: int,
    expected_channels: int,
    expected_sample_width: int,
    expected_frames: int,
) -> None:
    """
    Validate one persisted event WAV.

    This verifies both file existence and essential PCM metadata.
    """

    assert (
        path.exists()
    ), (
        f"missing event WAV: {path}"
    )

    assert (
        path.is_file()
    )

    assert (
        path.stat().st_size
        > 44
    )

    with wave.open(
        str(
            path
        ),
        "rb",
    ) as wav_file:

        assert (
            wav_file.getnchannels()
            == expected_channels
        )

        assert (
            wav_file.getsampwidth()
            == expected_sample_width
        )

        assert (
            wav_file.getframerate()
            == expected_sample_rate
        )

        assert (
            wav_file.getnframes()
            == expected_frames
        )


def validate_event_directory(
    row,
    *,
    config,
) -> None:
    """
    Verify synchronized event WAV storage for all three microphones.
    """

    event_directory_text = (
        row[
            "event_directory"
        ]
    )

    assert (
        event_directory_text
        is not None
    )

    event_directory = Path(
        event_directory_text
    )

    assert (
        event_directory.exists()
    )

    assert (
        event_directory.is_dir()
    )

    expected_frames = (
        int(
            row[
                "end_sample"
            ]
        )
        - int(
            row[
                "start_sample"
            ]
        )
    )

    assert (
        expected_frames
        > 0
    )

    for node_id in NODE_IDS:

        validate_event_wav(
            event_directory
            / (
                f"node_{node_id}.wav"
            ),

            expected_sample_rate=
                config.audio.sample_rate,

            expected_channels=
                config.audio.channels,

            expected_sample_width=
                config.audio.sample_width_bytes,

            expected_frames=
                expected_frames,
        )


# ======================================================================
# DATABASE SESSION VALIDATION
# ======================================================================


def validate_session_closed(
    database_path: Path,
    *,
    session_id: int,
) -> None:
    """
    Verify that EventPipeline closed the persisted acquisition session.
    """

    assert (
        database_path.exists()
    )

    with sqlite3.connect(
        database_path
    ) as connection:

        row = connection.execute(
            """
            SELECT
                session_id,
                label,
                started_at,
                stopped_at
            FROM sessions
            WHERE session_id = ?
            """,
            (
                session_id,
            ),
        ).fetchone()

    assert (
        row
        is not None
    )

    (
        stored_session_id,
        stored_label,
        started_at,
        stopped_at,
    ) = (
        row
    )

    assert (
        stored_session_id
        == session_id
    )

    assert (
        stored_label
        == SESSION_LABEL
    )

    assert (
        started_at
        is not None
    )

    assert (
        stopped_at
        is not None
    )


# ======================================================================
# EVENT ROW VALIDATION
# ======================================================================


def validate_core_event(
    row,
    *,
    session_id: int,
) -> None:
    """
    Validate the mandatory persisted AcousticEvent fields.
    """

    assert (
        row[
            "id"
        ]
        > 0
    )

    assert (
        row[
            "session_id"
        ]
        == session_id
    )

    assert (
        row[
            "detector_event_id"
        ]
        > 0
    )

    assert (
        row[
            "start_sample"
        ]
        >= 0
    )

    assert (
        row[
            "end_sample"
        ]
        > row[
            "start_sample"
        ]
    )

    assert math.isfinite(
        row[
            "peak_rms_dbfs"
        ]
    )


def validate_optional_pipeline_outputs(
    row,
) -> None:
    """
    Sanity-check optional downstream outputs when they are available.

    Core event persistence is intentionally allowed to survive:

        localization failure
        DSP failure
        classification failure

    so this integration smoke test does not require every optional field
    to be non-NULL.

    When a field IS present, however, it must be numerically sane.
    """

    # ==============================================================
    # ENVIRONMENT
    # ==============================================================

    temperature = (
        row[
            "temperature_c"
        ]
    )

    humidity = (
        row[
            "humidity_percent"
        ]
    )

    pressure = (
        row[
            "pressure_hpa"
        ]
    )

    if (
        temperature
        is not None
    ):

        assert math.isfinite(
            temperature
        )

    if (
        humidity
        is not None
    ):

        assert math.isfinite(
            humidity
        )

    if (
        pressure
        is not None
    ):

        assert math.isfinite(
            pressure
        )

        assert (
            pressure
            > 0.0
        )

    # ==============================================================
    # LOCALIZATION
    # ==============================================================

    x_m = (
        row[
            "x_m"
        ]
    )

    y_m = (
        row[
            "y_m"
        ]
    )

    if (
        x_m
        is not None
    ):

        assert math.isfinite(
            x_m
        )

    if (
        y_m
        is not None
    ):

        assert math.isfinite(
            y_m
        )

    # ==============================================================
    # FEATURES
    # ==============================================================

    rms_value = (
        row[
            "rms"
        ]
    )

    if (
        rms_value
        is not None
    ):

        assert math.isfinite(
            rms_value
        )

        assert (
            rms_value
            >= 0.0
        )

    dominant_frequency = (
        row[
            "dominant_frequency_hz"
        ]
    )

    if (
        dominant_frequency
        is not None
    ):

        assert math.isfinite(
            dominant_frequency
        )

        assert (
            dominant_frequency
            >= 0.0
        )

    # ==============================================================
    # CLASSIFICATION
    # ==============================================================

    classification_label = (
        row[
            "classification_label"
        ]
    )

    classification_confidence = (
        row[
            "classification_confidence"
        ]
    )

    if (
        classification_label
        is not None
    ):

        assert isinstance(
            classification_label,
            str,
        )

        assert (
            len(
                classification_label
            )
            > 0
        )

    if (
        classification_confidence
        is not None
    ):

        assert math.isfinite(
            classification_confidence
        )

        assert (
            0.0
            <= classification_confidence
            <= 1.0
        )


# ======================================================================
# ASYNC INTEGRATION SCENARIO
# ======================================================================


async def run_event_integration(
    tmp_path: Path,
) -> None:
    """
    Run one complete synthetic event-detection experiment.
    """

    # ==============================================================
    # CONFIGURATION
    # ==============================================================

    port = (
        find_available_tcp_port()
    )

    config = make_integration_config(
        tmp_path,
        port=
            port,
    )

    # ==============================================================
    # RECEIVER
    # ==============================================================

    server = ReceiverServer(
        config
    )

    # ==============================================================
    # SYNTHETIC PHYSICAL WORLD
    # ==============================================================

    shared = (
        SharedSimulation()
    )

    nodes = [
        FakeNode(
            node_id,
            TEST_HOST,
            port,
            shared,
        )

        for node_id
        in NODE_IDS
    ]

    node_tasks: list[
        asyncio.Task
    ] = []

    session_id: (
        int
        | None
    ) = None

    event_row = (
        None
    )

    try:

        # ==========================================================
        # START RECEIVER
        # ==========================================================

        await server.start()

        # ==========================================================
        # START SIMULATED ESP32 NODES
        # ==========================================================

        node_tasks = [
            asyncio.create_task(
                node.run(),
                name=
                    (
                        "event-simulator-"
                        f"node-{node.node_id}"
                    ),
            )

            for node
            in nodes
        ]

        # ==========================================================
        # HELLO HANDSHAKES
        # ==========================================================

        await server.wait_for_nodes(
            timeout=
                5.0
        )

        assert (
            server.all_expected_nodes_connected()
        )

        assert (
            set(
                server.connections
            )
            == set(
                NODE_IDS
            )
        )

        # ==========================================================
        # START ACQUISITION
        # ==========================================================

        session_id = (
            await server.start_acquisition(
                SESSION_LABEL
            )
        )

        assert (
            session_id
            != 0
        )

        assert (
            server.active_session_id
            == session_id
        )

        assert (
            server.events.active_session_id
            == session_id
        )

        assert (
            server.soundscape
            is not None
        )

        assert (
            server.soundscape.active_session_id
            == session_id
        )

        # ==========================================================
        # WAIT FOR THREE-NODE AUDIO
        # ==========================================================

        await wait_for_all_node_audio(
            server,
            timeout_s=
                5.0,
        )

        # ==========================================================
        # COMMON SESSION
        # ==========================================================

        assert all(
            (
                server.connections[
                    node_id
                ]
                .state
                .session_id
                == session_id
            )

            for node_id
            in NODE_IDS
        )

        # ==========================================================
        # WAIT FOR SYNTHETIC EVENT
        # ==========================================================

        event_row = (
            await wait_for_persisted_event(
                server,

                timeout_s=
                    EVENT_TIMEOUT_S,
            )
        )

        # ==========================================================
        # CORE EVENT CONTRACT
        # ==========================================================

        validate_core_event(
            event_row,

            session_id=
                session_id,
        )

        # ==========================================================
        # CONTINUOUS SOUNDSCAPE PERSISTENCE
        # ==========================================================

        soundscape_rows = (
            server.events.database
            .get_soundscape_indices(
                session_id=session_id
            )
        )

        assert soundscape_rows

        assert set(
            row["node_id"]
            for row in soundscape_rows
        ) == set(
            NODE_IDS
        )

        assert all(
            row["end_sample"]
            - row["start_sample"]
            == config.audio.sample_rate
            for row in soundscape_rows
        )

        # ==========================================================
        # EVENT PIPELINE COUNTER
        # ==========================================================

        assert (
            server.events.completed_events
            >= 1
        )

        assert (
            server.events.last_event_db_id
            is not None
        )

        # ==========================================================
        # EVENT WAVS
        # ==========================================================

        validate_event_directory(
            event_row,

            config=
                config,
        )

        # ==========================================================
        # OPTIONAL DSP / ENVIRONMENT / LOCALIZATION /
        # CLASSIFICATION FIELDS
        # ==========================================================

        validate_optional_pipeline_outputs(
            event_row
        )

        # ==========================================================
        # NETWORK HEALTH
        # ==========================================================

        for node_id in NODE_IDS:

            state = (
                server.connections[
                    node_id
                ]
                .state
            )

            assert (
                state.audio_packets_received
                > 0
            )

            assert (
                state.sequence_gaps
                == 0
            )

            assert (
                state.sample_gaps
                == 0
            )

        # ==========================================================
        # NORMAL STOP
        # ==========================================================

        await server.stop_acquisition()

        assert (
            server.active_session_id
            is None
        )

        assert (
            server.events.active_session_id
            is None
        )

        assert (
            server.soundscape
            is not None
        )

        assert (
            server.soundscape.active_session_id
            is None
        )

    finally:

        # ==========================================================
        # BEST-EFFORT ACQUISITION STOP
        # ==========================================================

        if (
            server.active_session_id
            is not None
        ):

            with contextlib.suppress(
                Exception
            ):

                await asyncio.wait_for(
                    server.stop_acquisition(),
                    timeout=
                        2.0,
                )

        # ==========================================================
        # CANCEL SIMULATOR TASKS
        # ==========================================================

        for task in node_tasks:

            task.cancel()

        if node_tasks:

            await asyncio.gather(
                *node_tasks,
                return_exceptions=
                    True,
            )

        # ==========================================================
        # CLOSE RECEIVER
        # ==========================================================

        with contextlib.suppress(
            Exception
        ):

            await asyncio.wait_for(
                server.close(),
                timeout=
                    2.0,
            )

    # ==================================================================
    # POST-SHUTDOWN DATABASE CHECK
    # ==================================================================

    assert (
        session_id
        is not None
    )

    assert (
        event_row
        is not None
    )

    validate_session_closed(
        config.persistence.database_path,

        session_id=
            session_id,
    )


# ======================================================================
# PYTEST ENTRY POINT
# ======================================================================


def test_three_node_event_detection_and_persistence(
    tmp_path: Path,
) -> None:
    """
    Pytest-discoverable integration entry point.

    asyncio.run() avoids requiring pytest-asyncio.
    """

    asyncio.run(
        run_event_integration(
            tmp_path
        )
    )


# ======================================================================
# OPTIONAL MANUAL ENTRY POINT
# ======================================================================


if (
    __name__
    == "__main__"
):

    import tempfile

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=
                "wildlife-events-"
        )
    )

    asyncio.run(
        run_event_integration(
            temporary_root
        )
    )
