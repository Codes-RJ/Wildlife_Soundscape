"""
End-to-end integration smoke test.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Exercise the real laptop receiver and the real three-node simulator
together over localhost TCP.

This is deliberately broader than the unit tests.

The smoke path verifies:

    ReceiverServer
        ↓
    TCP connection
        ↓
    Protocol-v4 HELLO
        ↓
    common session START
        ↓
    simulated shared BCLK/WS clock
        ↓
    three-node AUDIO streaming
        ↓
    NodeState
        ↓
    StreamManager
        ↓
    coarse sampleIndex alignment
        ↓
    acquisition STOP
        ↓
    EventPipeline / SQLite session closure

This test does NOT attempt to prove:

    classification accuracy
    real microphone synchronization accuracy
    real acoustic localization accuracy
    field reliability

Those require their dedicated unit, calibration and physical-hardware
tests.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import asyncio
import contextlib
import socket
import sqlite3

from collections.abc import (
    Callable,
)

from dataclasses import (
    replace,
)

from pathlib import (
    Path,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.core.config import (
    CONFIG,
)

from wildlife_soundscape.runtime.server import (
    ReceiverServer,
)

from wildlife_soundscape.runtime.simulator import (
    FakeNode,
    SharedSimulation,
)


# ======================================================================
# CONSTANTS
# ======================================================================


TEST_HOST = (
    "127.0.0.1"
)


EXPECTED_NODE_IDS = (
    1,
    2,
    3,
)


# ======================================================================
# PORT ALLOCATION
# ======================================================================


def find_available_tcp_port() -> int:
    """
    Ask the operating system for an unused localhost TCP port.

    The temporary reservation is released immediately before the
    ReceiverServer binds to the returned port.

    There is theoretically a very small race between releasing the
    temporary socket and starting the receiver, but this is still much
    safer for a test suite than permanently hard-coding a port such as
    55001.
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
            temporary_socket.getsockname()[
                1
            ]
        )

    return (
        port
    )


# ======================================================================
# ASYNC POLLING
# ======================================================================


async def wait_until(
    predicate: Callable[
        [],
        bool,
    ],
    *,
    timeout_s: float,
    interval_s: float = 0.02,
    description: str,
) -> None:
    """
    Wait until a synchronous condition becomes true.

    Fixed sleeps are avoided because CI machines and development laptops
    may schedule asyncio/TCP tasks at different speeds.
    """

    loop = (
        asyncio.get_running_loop()
    )

    deadline = (
        loop.time()
        + float(
            timeout_s
        )
    )

    while True:

        if (
            predicate()
        ):

            return

        if (
            loop.time()
            >= deadline
        ):

            raise AssertionError(
                (
                    "Timed out waiting for "
                    f"{description}"
                )
            )

        await asyncio.sleep(
            interval_s
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
    Build an isolated version of the real application configuration.

    The test deliberately derives from CONFIG rather than manually
    reconstructing AppConfig.

    This is important because AppConfig now contains:

        network
        audio
        localization
        detection
        DSP
        classification
        persistence
        expected-node settings

    Using dataclasses.replace() preserves those production defaults while
    overriding only resources that must be isolated during testing.
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
    # Disable continuous session WAV recording.
    #
    # Event-level persistence is separately disabled below.
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
            False,
    )

    # ==============================================================
    # COMPLETE APPLICATION CONFIG
    # ==============================================================

    return replace(
        CONFIG,

        network=
            network,

        audio=
            audio,

        persistence=
            persistence,

        recordings_dir=
            tmp_path
            / "recordings",

        # Status printing is irrelevant during pytest.
        print_status_every_s=
            60.0,
    )


# ======================================================================
# CONNECTION STATE HELPERS
# ======================================================================


def every_expected_node_has_audio(
    server: ReceiverServer,
) -> bool:
    """
    Whether every expected node has delivered at least one AUDIO packet.
    """

    for node_id in EXPECTED_NODE_IDS:

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
            connection.state.audio_packets_received
            <= 0
        ):

            return (
                False
            )

    return (
        True
    )


def every_expected_node_uses_session(
    server: ReceiverServer,
    session_id: int,
) -> bool:
    """
    Whether all node states have transitioned onto the laptop-generated
    acquisition session.
    """

    for node_id in EXPECTED_NODE_IDS:

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
            connection.state.session_id
            != session_id
        ):

            return (
                False
            )

    return (
        True
    )


def simulator_is_fully_stopped(
    shared: SharedSimulation,
    nodes: list[
        FakeNode
    ],
) -> bool:
    """
    Whether the simulator has processed all STOP commands.
    """

    return (
        shared.clock_session_id
        is None

        and not (
            shared.armed_session
        )

        and all(
            not node.streaming

            for node
            in nodes
        )
    )


# ======================================================================
# DATABASE ASSERTION
# ======================================================================


def assert_session_was_closed_in_database(
    database_path: Path,
    *,
    session_id: int,
    expected_label: str,
) -> None:
    """
    Verify the real EventPipeline/EventDatabase session lifecycle.

    This is intentionally a very small schema-level check.

    Detailed database behavior is covered by test_database.py.
    """

    assert (
        database_path.exists()
    )

    with contextlib.closing(
        sqlite3.connect(
            database_path
        )
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
        == expected_label
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
# ASYNC SMOKE SCENARIO
# ======================================================================


async def run_smoke_scenario(
    tmp_path: Path,
) -> None:
    """
    Execute one complete simulated acquisition lifecycle.
    """

    # ==============================================================
    # ISOLATED CONFIGURATION
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
    # REAL RECEIVER SERVER
    # ==============================================================

    server = ReceiverServer(
        config
    )

    # ==============================================================
    # REAL THREE-NODE SIMULATOR
    # ==============================================================

    shared = (
        SharedSimulation()
    )

    nodes = [
        FakeNode(
            node_id=
                node_id,

            host=
                TEST_HOST,

            port=
                port,

            shared=
                shared,
        )

        for node_id
        in EXPECTED_NODE_IDS
    ]

    node_tasks: list[
        asyncio.Task
    ] = []

    # Session ID remains available for the persistence assertion after
    # the asynchronous network lifecycle has been shut down.
    session_id: (
        int
        | None
    ) = None

    try:

        # ==========================================================
        # START RECEIVER
        # ==========================================================

        await server.start()

        # ==========================================================
        # START VIRTUAL ESP32 NODES
        # ==========================================================

        node_tasks = [
            asyncio.create_task(
                node.run(),
                name=
                    f"simulator-node-{node.node_id}",
            )

            for node
            in nodes
        ]

        # ==========================================================
        # WAIT FOR PROTOCOL-v4 HELLO HANDSHAKES
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
                EXPECTED_NODE_IDS
            )
        )

        # ==========================================================
        # VERIFY NODE ROLES FROM REAL HELLO PAYLOADS
        # ==========================================================

        node_1_hello = (
            server.connections[
                1
            ].state.hello
        )

        node_2_hello = (
            server.connections[
                2
            ].state.hello
        )

        node_3_hello = (
            server.connections[
                3
            ].state.hello
        )

        assert (
            node_1_hello
            is not None
        )

        assert (
            node_2_hello
            is not None
        )

        assert (
            node_3_hello
            is not None
        )

        assert (
            node_1_hello.master_node
            is True
        )

        assert (
            node_2_hello.master_node
            is False
        )

        assert (
            node_3_hello.master_node
            is False
        )

        assert (
            node_1_hello.sample_rate
            == config.audio.sample_rate
        )

        assert (
            node_2_hello.sample_rate
            == config.audio.sample_rate
        )

        assert (
            node_3_hello.sample_rate
            == config.audio.sample_rate
        )

        # ==========================================================
        # START SYNCHRONIZED ACQUISITION
        # ==========================================================

        session_id = (
            await server.start_acquisition(
                "smoke"
            )
        )

        assert isinstance(
            session_id,
            int,
        )

        assert (
            1
            <= session_id
            <= 0xFFFFFFFF
        )

        assert (
            server.active_session_id
            == session_id
        )

        assert (
            server.events.active_session_id
            == session_id
        )

        # ==========================================================
        # SIMULATED MASTER CLOCK
        # ==========================================================

        await wait_until(
            lambda:
                (
                    shared.clock_session_id
                    == session_id
                ),

            timeout_s=
                2.0,

            description=
                "Node 1 simulated shared audio clock",
        )

        assert (
            shared.clock_event.is_set()
        )

        # ==========================================================
        # RECEIVE AUDIO FROM ALL NODES
        # ==========================================================

        await wait_until(
            lambda:
                every_expected_node_has_audio(
                    server
                ),

            timeout_s=
                5.0,

            description=
                "AUDIO packets from all three simulated nodes",
        )

        # ==========================================================
        # SAME SESSION ON ALL THREE NODES
        # ==========================================================

        await wait_until(
            lambda:
                every_expected_node_uses_session(
                    server,
                    session_id,
                ),

            timeout_s=
                2.0,

            description=
                "common acquisition session on all node states",
        )

        # ==========================================================
        # BASIC STREAM HEALTH
        # ==========================================================

        for node_id in EXPECTED_NODE_IDS:

            state = (
                server.connections[
                    node_id
                ].state
            )

            assert (
                state.audio_packets_received
                > 0
            )

            assert (
                state.session_id
                == session_id
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
        # COARSE THREE-NODE ALIGNMENT
        # ==========================================================

        aligned = (
            None
        )

        async def wait_for_alignment() -> None:

            nonlocal aligned

            loop = (
                asyncio.get_running_loop()
            )

            deadline = (
                loop.time()
                + 5.0
            )

            while (
                aligned
                is None
            ):

                aligned = (
                    server.streams
                    .latest_aligned_blocks()
                )

                if (
                    aligned
                    is not None
                ):

                    return

                if (
                    loop.time()
                    >= deadline
                ):

                    last_indices = {
                        node_id:
                            (
                                server.connections[
                                    node_id
                                ]
                                .state
                                .last_sample_index
                            )

                        for node_id
                        in EXPECTED_NODE_IDS
                    }

                    raise AssertionError(
                        (
                            "Timed out waiting for "
                            "three-node sampleIndex alignment. "
                            f"Last indices: {last_indices}"
                        )
                    )

                await asyncio.sleep(
                    0.02
                )

        await wait_for_alignment()

        assert (
            aligned
            is not None
        )

        assert (
            set(
                aligned.blocks
            )
            == set(
                EXPECTED_NODE_IDS
            )
        )

        assert (
            set(
                aligned.offsets
            )
            == set(
                EXPECTED_NODE_IDS
            )
        )

        # ----------------------------------------------------------
        # Every selected aligned block must belong to this one common
        # laptop-generated acquisition session.
        # ----------------------------------------------------------

        assert all(
            (
                block.session_id
                == session_id
            )

            for block
            in aligned.blocks.values()
        )

        # ----------------------------------------------------------
        # Coarse offsets must obey the configured sampleIndex tolerance.
        # Physical acoustic arrival delay remains inside the waveforms
        # and is NOT represented by these offsets.
        # ----------------------------------------------------------

        assert all(
            (
                abs(
                    offset
                )
                <= config.audio.sync_tolerance_samples
            )

            for offset
            in aligned.offsets.values()
        )

        # ==========================================================
        # STOP ACQUISITION
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

        # ==========================================================
        # VERIFY SIMULATOR PROCESSED STOP
        # ==========================================================

        await wait_until(
            lambda:
                simulator_is_fully_stopped(
                    shared,
                    nodes,
                ),

            timeout_s=
                2.0,

            description=
                "all simulator nodes to process STOP",
        )

    finally:

        # ==========================================================
        # BEST-EFFORT ACTIVE SESSION SHUTDOWN
        # ==========================================================

        if (
            server.active_session_id
            is not None
        ):

            with contextlib.suppress(
                Exception
            ):

                await server.stop_acquisition()

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
        # CLOSE TCP RECEIVER
        # ==========================================================

        await server.close()

    # ==================================================================
    # PERSISTENCE CHECK
    # ==================================================================
    #
    # Run after graceful STOP/CLOSE so the SQLite session must contain a
    # stopped_at timestamp.
    # ==================================================================

    assert (
        session_id
        is not None
    )

    assert_session_was_closed_in_database(
        config.persistence.database_path,

        session_id=
            session_id,

        expected_label=
            "smoke",
    )


# ======================================================================
# PYTEST ENTRY POINT
# ======================================================================


def test_receiver_and_three_node_simulator_smoke(
    tmp_path: Path,
) -> None:
    """
    Pytest-discoverable synchronous entry point.

    asyncio.run() is used deliberately so the project does not need an
    additional pytest-asyncio dependency just for this integration test.
    """

    asyncio.run(
        run_smoke_scenario(
            tmp_path
        )
    )
