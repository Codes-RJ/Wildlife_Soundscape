"""
End-to-end localization integration test.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Verify that the real simulated acquisition and localization stack can
produce a physically reasonable source-position estimate.

Integration path
----------------

    FakeNode 1 ─┐
    FakeNode 2 ─┼─ TCP / Protocol-v4
    FakeNode 3 ─┘
            ↓
      ReceiverServer
            ↓
       StreamManager
            ↓
    LocalizationEngine
            ↓
    localization filtering
            ↓
         GCC-PHAT
            ↓
          TDOA
            ↓
    nonlinear solver
            ↓
        estimated x,y
            ↓
    simulator ground truth

Scientific scope
----------------
This is an integration smoke test.

It answers:

    "Can the complete synthetic localization pipeline work?"

It does NOT answer:

    "What localization accuracy will the physical microphone array
    achieve?"

Real accuracy and repeatability must later be evaluated using the
dedicated calibration / localization benchmark.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import asyncio
import contextlib
import math
import socket

from dataclasses import (
    replace,
)

from pathlib import (
    Path,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================




# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.core.config import (
    CONFIG,
)

from wildlife_soundscape.localization import (
    LocalizationEngine,
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


NODE_IDS = (
    1,
    2,
    3,
)


# ----------------------------------------------------------------------
# The synthetic source event does not occur immediately after START.
#
# Give the simulator sufficient time to:
#
#     connect
#     receive START
#     establish shared simulated clock
#     generate background
#     reach its synthetic acoustic event
#     accumulate localization windows
#
# This is an integration timeout, not a fixed wait before asserting.
# ----------------------------------------------------------------------


LOCALIZATION_TIMEOUT_S = (
    8.0
)


POLL_INTERVAL_S = (
    0.05
)


# ----------------------------------------------------------------------
# This remains a smoke-test tolerance.
#
# A stricter statistical localization metric belongs in:
#
#     calibration/localization_benchmark.py
#
# rather than this integration test.
# ----------------------------------------------------------------------


MAX_SMOKE_LOCALIZATION_ERROR_M = (
    0.15
)


# ======================================================================
# AVAILABLE TCP PORT
# ======================================================================


def find_available_tcp_port() -> int:
    """
    Obtain an unused localhost TCP port from the operating system.

    This avoids hard-coding the normal application port during pytest.
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
# ISOLATED CONFIGURATION
# ======================================================================


def make_integration_config(
    tmp_path: Path,
    *,
    port: int,
):
    """
    Derive an isolated integration configuration from the real CONFIG.

    Production defaults are preserved except for:

        localhost TCP port
        continuous WAV recording
        SQLite output
        event WAV output
        recording directory
        console status interval
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
            / "localization.db",

        events_dir=
            tmp_path
            / "events",

        save_event_wav=
            False,
    )

    # ==============================================================
    # COMPLETE CONFIGURATION
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

        print_status_every_s=
            60.0,
    )


# ======================================================================
# BASIC SERVER STATE
# ======================================================================


def all_nodes_have_audio(
    server: ReceiverServer,
) -> bool:
    """
    Return True only after all three nodes have delivered AUDIO packets.
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


async def wait_for_audio(
    server: ReceiverServer,
    *,
    timeout_s: float,
) -> None:
    """
    Wait until every simulated microphone is actively streaming.
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

            packet_counts = {
                node_id:
                    (
                        server.connections[
                            node_id
                        ].state
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
                    f"Packet counts: {packet_counts}"
                )
            )

        await asyncio.sleep(
            POLL_INTERVAL_S
        )


# ======================================================================
# LOCALIZATION SEARCH
# ======================================================================


async def collect_successful_localizations(
    engine: LocalizationEngine,
    shared: SharedSimulation,
    *,
    timeout_s: float,
):
    """
    Poll LocalizationEngine until the timeout expires.

    Returns
    -------
    list[tuple[float, LocalizationResult]]
        Successful localization attempts stored as:

            (position_error_m, result)

    Notes
    -----
    Many localization attempts are expected to fail while only
    background noise is present.

    The simulated acoustic event eventually provides sufficient signal
    energy and correlation structure for valid TDOA measurements.
    """

    loop = (
        asyncio.get_running_loop()
    )

    deadline = (
        loop.time()
        + timeout_s
    )

    successful = []

    while (
        loop.time()
        < deadline
    ):

        result = (
            engine.locate_latest()
        )

        if (
            result is not None
            and result.success
        ):

            position_error_m = math.hypot(
                (
                    result.position.x
                    - shared.source_xy[
                        0
                    ]
                ),
                (
                    result.position.y
                    - shared.source_xy[
                        1
                    ]
                ),
            )

            if (
                math.isfinite(
                    position_error_m
                )
            ):

                successful.append(
                    (
                        position_error_m,
                        result,
                    )
                )

                # --------------------------------------------------
                # The purpose of this integration test is only to
                # establish that the complete pipeline can produce
                # one acceptable synthetic localization.
                #
                # Once that condition is satisfied there is no need
                # to keep the TCP simulator alive for the full
                # timeout.
                # --------------------------------------------------

                if (
                    position_error_m
                    < MAX_SMOKE_LOCALIZATION_ERROR_M
                ):

                    break

        await asyncio.sleep(
            POLL_INTERVAL_S
        )

    return (
        successful
    )


# ======================================================================
# INTEGRATION SCENARIO
# ======================================================================


async def run_localization_integration(
    tmp_path: Path,
) -> None:
    """
    Run one complete synthetic localization experiment.
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
    # SIMULATED PHYSICAL WORLD
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

    successful = []

    try:

        # ==========================================================
        # SERVER START
        # ==========================================================

        await server.start()

        # ==========================================================
        # SIMULATOR START
        # ==========================================================

        node_tasks = [
            asyncio.create_task(
                node.run(),
                name=
                    (
                        "localization-simulator-"
                        f"node-{node.node_id}"
                    ),
            )

            for node
            in nodes
        ]

        # ==========================================================
        # HELLO HANDSHAKE
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
        # ACQUISITION SESSION
        # ==========================================================

        session_id = (
            await server.start_acquisition(
                "integration_localization"
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

        # ==========================================================
        # WAIT FOR REAL AUDIO FLOW
        # ==========================================================

        await wait_for_audio(
            server,
            timeout_s=
                5.0,
        )

        # ==========================================================
        # SHARED SESSION CHECK
        # ==========================================================

        assert all(
            (
                server.connections[
                    node_id
                ].state
                .session_id
                == session_id
            )

            for node_id
            in NODE_IDS
        )

        # ==========================================================
        # LOCALIZATION ENGINE
        # ==========================================================

        engine = LocalizationEngine(
            server.streams,
            config.localization,
        )

        # ==========================================================
        # SEARCH ACROSS SYNTHETIC EVENT
        # ==========================================================

        successful = (
            await collect_successful_localizations(
                engine,
                shared,
                timeout_s=
                    LOCALIZATION_TIMEOUT_S,
            )
        )

        # ==========================================================
        # AT LEAST ONE SOLUTION
        # ==========================================================

        assert (
            successful
        ), (
            "LocalizationEngine produced no "
            "successful position estimate during "
            "the synthetic localization interval."
        )

        # ==========================================================
        # BEST SYNTHETIC SOLUTION
        # ==============================================================

        successful.sort(
            key=
                lambda item:
                    item[
                        0
                    ]
        )

        (
            best_error_m,
            best_result,
        ) = (
            successful[
                0
            ]
        )

        # ==========================================================
        # POSITION OUTPUT
        # ==========================================================

        assert math.isfinite(
            best_result.position.x
        )

        assert math.isfinite(
            best_result.position.y
        )

        assert (
            best_result.success
        )

        # ==========================================================
        # THREE MICROPHONE PAIRS
        # ==============================================================

        assert (
            len(
                best_result.measurements
            )
            == 3
        )

        assert {
            (
                measurement.node_a,
                measurement.node_b,
            )

            for measurement
            in best_result.measurements
        } == {
            (
                1,
                2,
            ),
            (
                1,
                3,
            ),
            (
                2,
                3,
            ),
        }

        # ==========================================================
        # PARTICIPATING NODE RMS
        # ==============================================================

        assert (
            set(
                best_result.node_rms
            )
            == set(
                NODE_IDS
            )
        )

        assert all(
            math.isfinite(
                value
            )
            and value
            >= 0.0

            for value
            in best_result.node_rms.values()
        )

        # ==========================================================
        # PROPAGATION SPEED
        # ==============================================================

        assert math.isfinite(
            best_result.speed_of_sound_mps
        )

        assert (
            best_result.speed_of_sound_mps
            > 0.0
        )

        # ==========================================================
        # GROUND-TRUTH POSITION ERROR
        # ==============================================================

        assert (
            best_error_m
            < MAX_SMOKE_LOCALIZATION_ERROR_M
        ), (
            "Synthetic localization error too high. "
            f"truth={shared.source_xy}, "
            "estimated="
            f"({best_result.position.x:.4f}, "
            f"{best_result.position.y:.4f}), "
            f"error={best_error_m:.4f} m, "
            "successful_estimates="
            f"{len(successful)}"
        )

        # ==========================================================
        # NORMAL STOP
        # ==========================================================

        await server.stop_acquisition()

        assert (
            server.active_session_id
            is None
        )

    finally:

        # ==========================================================
        # BEST-EFFORT STOP
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
        # STOP SIMULATED NODES
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
        # SERVER CLOSE
        # ==========================================================

        with contextlib.suppress(
            Exception
        ):

            await asyncio.wait_for(
                server.close(),
                timeout=
                    2.0,
            )


# ======================================================================
# PYTEST ENTRY POINT
# ======================================================================


def test_three_node_end_to_end_localization(
    tmp_path: Path,
) -> None:
    """
    Pytest-discoverable integration test.

    No pytest-asyncio dependency is required.
    """

    asyncio.run(
        run_localization_integration(
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
                "wildlife-localization-"
        )
    )

    asyncio.run(
        run_localization_integration(
            temporary_root
        )
    )