from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets

import numpy as np

from analytics.indices import SoundscapeIndicesConfig
from analytics.soundscape_service import SoundscapeService

from config import AppConfig

from event_pipeline import EventPipeline

from models import (
    AudioBlock,
    EnvironmentSample,
)

from node import (
    NodeConnection,
    NodeState,
)

from protocol import (
    CRCMismatch,
    HEADER_SIZE,
    ControlCommand,
    HelloPayload,
    Packet,
    PacketType,
    ProtocolError,
    parse_environment,
    parse_heartbeat,
    parse_hello,
    parse_sync,
    unpack_header,
    verify_payload_crc,
)

from stream_manager import (
    StreamManager,
    WavRecorder,
)


logger = logging.getLogger(
    __name__
)


# ======================================================================
# SERVER-LEVEL PROTOCOL CONSTANTS
# ======================================================================


MASTER_NODE_ID = 1


SESSION_BOUND_PACKET_TYPES = frozenset(
    {
        PacketType.AUDIO,
        PacketType.ENVIRONMENT,
        PacketType.SYNC,
    }
)


AUXILIARY_PACKET_TYPES = frozenset(
    {
        PacketType.HELLO,
        PacketType.HEARTBEAT,
    }
)


# ======================================================================
# SESSION-LABEL SAFETY
# ======================================================================


INVALID_SESSION_LABEL_CHARACTERS = frozenset(
    '<>:"/\\|?*'
)


WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(
            f"COM{index}"
            for index in range(
                1,
                10,
            )
        ),
        *(
            f"LPT{index}"
            for index in range(
                1,
                10,
            )
        ),
    }
)


# ======================================================================
# RECEIVER SERVER
# ======================================================================


class ReceiverServer:
    """
    Async TCP receiver for synchronized ESP32 acoustic nodes.

    Responsibilities
    ----------------
    - accept ESP32 TCP connections
    - enforce HELLO-first registration
    - validate Protocol-v4 framing
    - verify CRC32
    - track node health and connection state
    - maintain shared acquisition-session identity
    - issue START / STOP / PING control commands
    - maintain slave-before-master startup order
    - route PCM into StreamManager
    - route acoustic data into EventPipeline
    - persist environmental telemetry
    - record synchronized continuous WAV streams

    Timing model
    ------------
    sampleIndex
        authoritative shared-clock coarse audio timeline

    localMicros
        ESP32-local diagnostic timestamp only

    TCP arrival time
        transport timing only

    Neither localMicros nor TCP arrival timing is used for TDOA.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        config: AppConfig,
    ) -> None:

        if not isinstance(
            config,
            AppConfig,
        ):
            raise TypeError(
                "config must be an AppConfig instance"
            )

        if (
            MASTER_NODE_ID
            not in config.expected_nodes
        ):
            raise ValueError(
                (
                    "Current shared-clock architecture "
                    "requires Node 1 as the I2S master."
                )
            )

        self.config = config

        # ==============================================================
        # ACTIVE ESP32 CONNECTIONS
        # ==============================================================

        self.connections: dict[
            int,
            NodeConnection,
        ] = {}

        # ==============================================================
        # STREAM STORAGE
        # ==============================================================

        self.streams = StreamManager(
            config.audio
        )

        # ==============================================================
        # CONTINUOUS WAV RECORDING
        # ==============================================================

        self.recorder = WavRecorder(
            config.recordings_dir,
            config.audio,
        )

        # ==============================================================
        # EVENT PIPELINE
        # ==============================================================

        self.events = EventPipeline(
            self.streams,
            config,
        )

        # ==============================================================
        # CONTINUOUS SOUNDSCAPE ANALYTICS
        # ==============================================================

        self.soundscape: SoundscapeService | None

        if (
            config.analytics.enabled
            and config.analytics.soundscape_enabled
        ):

            self.soundscape = SoundscapeService(
                config=SoundscapeIndicesConfig(
                    sample_rate=config.audio.sample_rate,
                ),
                window_duration_seconds=
                    config.analytics.soundscape_window_seconds,
                database=self.events.database,
            )

        else:

            self.soundscape = None

        # ==============================================================
        # TCP LISTENER
        # ==============================================================

        self._server: (
            asyncio.AbstractServer
            | None
        ) = None

        # ==============================================================
        # CONNECTION REGISTRY LOCK
        # ==============================================================

        self._connection_lock = (
            asyncio.Lock()
        )

        # ==============================================================
        # SESSION-LIFECYCLE LOCK
        # ==============================================================
        #
        # START and STOP alter:
        #
        #   active_session_id
        #   database session
        #   WAV recorder state
        #   ESP32 acquisition state
        #
        # They must therefore never overlap.
        # ==============================================================

        self._session_lock = (
            asyncio.Lock()
        )

        # ==============================================================
        # ACTIVE ACQUISITION
        # ==============================================================

        self.active_session_id: (
            int
            | None
        ) = None

    # ==================================================================
    # NODE ORDER
    # ==================================================================

    @property
    def slave_node_ids(
        self,
    ) -> tuple[int, ...]:
        """
        Configured slave nodes in deterministic startup order.

        For the current 3-node deployment this returns:

            (2, 3)
        """

        return tuple(
            sorted(
                node_id
                for node_id
                in self.config.expected_nodes
                if node_id
                != MASTER_NODE_ID
            )
        )

    # ==================================================================
    # SESSION LABEL
    # ==================================================================

    def _normalize_session_label(
        self,
        value: str,
    ) -> str:
        """
        Validate a session label before it becomes part of filesystem
        paths.

        This protects both:

            data/recordings/
            data/events/

        from malformed or path-traversing labels.
        """

        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                "session_label must be a string"
            )

        label = value.strip()

        if not label:
            raise ValueError(
                "session_label cannot be empty"
            )

        if len(
            label
        ) > 96:
            raise ValueError(
                (
                    "session_label cannot exceed "
                    "96 characters"
                )
            )

        if label in {
            ".",
            "..",
        }:
            raise ValueError(
                "invalid session_label"
            )

        for character in label:

            if (
                ord(
                    character
                )
                < 32
            ):
                raise ValueError(
                    (
                        "session_label cannot contain "
                        "control characters"
                    )
                )

            if (
                character
                in INVALID_SESSION_LABEL_CHARACTERS
            ):
                raise ValueError(
                    (
                        "session_label contains "
                        f"invalid character {character!r}"
                    )
                )

        # Windows rejects names ending with a period or space.
        if label.endswith(
            (
                ".",
                " ",
            )
        ):
            raise ValueError(
                (
                    "session_label cannot end "
                    "with a period or space"
                )
            )

        # Windows also reserves names such as CON, NUL, COM1...
        stem = (
            label.split(
                ".",
                1,
            )[0]
            .upper()
        )

        if (
            stem
            in WINDOWS_RESERVED_NAMES
        ):
            raise ValueError(
                (
                    "session_label uses a reserved "
                    f"filesystem name: {label!r}"
                )
            )

        return label

    # ==================================================================
    # SERVER START
    # ==================================================================

    async def start(
        self,
    ) -> None:
        """
        Start the laptop TCP receiver.
        """

        if (
            self._server
            is not None
        ):
            return

        stream_limit = max(
            64
            * 1024,

            self.config.network.max_payload_bytes
            + HEADER_SIZE,
        )

        self._server = (
            await asyncio.start_server(
                self._handle_client,
                self.config.network.host,
                self.config.network.port,
                limit=stream_limit,
            )
        )

        sockets = (
            self._server.sockets
            or []
        )

        for sock in sockets:

            logger.info(
                "Listening on %s",
                sock.getsockname(),
            )

    # ==================================================================
    # SERVE FOREVER
    # ==================================================================

    async def serve_forever(
        self,
    ) -> None:
        """
        Run the receiver until cancellation.
        """

        if (
            self._server
            is None
        ):
            await self.start()

        assert (
            self._server
            is not None
        )

        async with (
            self._server
        ):
            await self._server.serve_forever()

    # ==================================================================
    # SERVER CLOSE
    # ==================================================================

    async def close(
        self,
    ) -> None:
        """
        Gracefully stop acquisition, listener and ESP32 sockets.

        Calling close() repeatedly is safe.
        """

        # ==============================================================
        # SESSION SHUTDOWN
        # ==============================================================

        try:

            # Calling this unconditionally is intentional.
            #
            # If START is currently in progress, _session_lock ensures
            # close() waits for that operation and subsequently stops
            # the resulting session.
            await self.stop_acquisition()

        except asyncio.CancelledError:

            raise

        except Exception:

            logger.exception(
                (
                    "Failed to stop acquisition "
                    "during server shutdown"
                )
            )

        # ==============================================================
        # CLOSE LISTENER
        # ==============================================================

        if (
            self._server
            is not None
        ):

            self._server.close()

            await self._server.wait_closed()

            self._server = (
                None
            )

        # ==============================================================
        # DETACH ACTIVE CONNECTIONS
        # ==============================================================

        async with (
            self._connection_lock
        ):

            connections = list(
                self.connections.values()
            )

            self.connections.clear()

            for connection in connections:

                connection.state.connected = (
                    False
                )

        # ==============================================================
        # CLOSE NODE SOCKETS
        # ==============================================================

        if connections:

            await asyncio.gather(
                *(
                    connection.close()
                    for connection
                    in connections
                ),
                return_exceptions=True,
            )

    # ==================================================================
    # CONNECTION STATUS
    # ==================================================================

    def all_expected_nodes_connected(
        self,
    ) -> bool:
        """
        Return True only when all expected ESP32 nodes have a usable
        connection.
        """

        for node_id in (
            self.config.expected_nodes
        ):

            connection = (
                self.connections.get(
                    node_id
                )
            )

            if (
                connection
                is None
            ):
                return False

            if not (
                connection.state.connected
            ):
                return False

            if (
                connection.writer.is_closing()
            ):
                return False

        return True

    # ==================================================================
    # WAIT FOR NODES
    # ==================================================================

    async def wait_for_nodes(
        self,
        timeout: float | None = None,
    ) -> None:
        """
        Wait until every configured acoustic node is connected.
        """

        if (
            timeout
            is not None
            and timeout <= 0
        ):
            raise ValueError(
                "timeout must be greater than 0"
            )

        async def _wait() -> None:

            while not (
                self.all_expected_nodes_connected()
            ):

                await asyncio.sleep(
                    0.05
                )

        if timeout is None:

            await _wait()

            return

        await asyncio.wait_for(
            _wait(),
            timeout=timeout,
        )

    # ==================================================================
    # COMMAND TRANSMISSION
    # ==================================================================

    async def send_command(
        self,
        node_id: int,
        command: ControlCommand,
        *,
        session_id: int = 0,
    ) -> None:
        """
        Send one control frame to the currently registered connection for
        a node.

        A bounded write prevents a dead TCP peer from blocking START or
        STOP indefinitely.
        """

        node_id = int(
            node_id
        )

        connection = (
            self.connections.get(
                node_id
            )
        )

        if connection is None:

            raise RuntimeError(
                (
                    f"node {node_id} "
                    "is not connected"
                )
            )

        try:

            await asyncio.wait_for(
                connection.send_command(
                    command,
                    session_id=session_id,
                ),
                timeout=
                    self.config.network.read_timeout_s,
            )

        except asyncio.TimeoutError as exc:

            raise asyncio.TimeoutError(
                (
                    f"control command {command.name} "
                    f"timed out for node {node_id}"
                )
            ) from exc

        # --------------------------------------------------------------
        # CONNECTION REPLACEMENT CHECK
        # --------------------------------------------------------------
        #
        # A node may reconnect while writer.drain() is in progress.
        #
        # If that happened, the command may have gone to the superseded
        # socket and cannot be considered authoritative for the current
        # node connection.
        # --------------------------------------------------------------

        if (
            self.connections.get(
                node_id
            )
            is not connection
        ):

            raise ConnectionError(
                (
                    f"node {node_id} connection "
                    "was replaced while sending "
                    f"{command.name}"
                )
            )

    # ==================================================================
    # SESSION ID
    # ==================================================================

    def _generate_session_id(
        self,
    ) -> int:
        """
        Generate a non-zero uint32 acquisition session identifier.

        One identical ID is supplied to every node.
        """

        return (
            secrets.randbits(
                32
            )
            or 1
        )

    # ==================================================================
    # START ACQUISITION
    # ==================================================================

    async def start_acquisition(
        self,
        session_label: str,
    ) -> int:
        """
        Start one synchronized acquisition session.

        Current three-node order:

            Node 2
                ↓
            Node 3
                ↓
            100 ms slave-arm period
                ↓
            Node 1 master
                ↓
            shared BCLK / WS become active

        More generally, all configured slave nodes are armed before
        Node 1.

        Startup is transactional.
        """

        async with (
            self._session_lock
        ):

            return await (
                self._start_acquisition_locked(
                    session_label
                )
            )

    async def _start_acquisition_locked(
        self,
        session_label: str,
    ) -> int:
        """
        Internal START implementation.

        Caller must hold _session_lock.
        """

        # ==============================================================
        # LABEL
        # ==============================================================

        normalized_label = (
            self._normalize_session_label(
                session_label
            )
        )

        # ==============================================================
        # PREVENT OVERLAPPING SESSION
        # ==============================================================

        if (
            self.active_session_id
            is not None
        ):

            raise RuntimeError(
                (
                    "an acquisition session "
                    "is already active"
                )
            )

        # ==============================================================
        # REQUIRE ALL NODES
        # ==============================================================

        missing = {
            node_id
            for node_id
            in self.config.expected_nodes
            if (
                node_id
                not in self.connections
                or not self.connections[
                    node_id
                ].state.connected
                or self.connections[
                    node_id
                ].writer.is_closing()
            )
        }

        if missing:

            raise RuntimeError(
                (
                    "cannot START; missing or "
                    "unavailable nodes: "
                    f"{sorted(missing)}"
                )
            )

        # ==============================================================
        # SESSION ID
        # ==============================================================

        session_id = (
            self._generate_session_id()
        )

        # ==============================================================
        # TRANSACTION STATE
        # ==============================================================

        event_session_started = (
            False
        )

        recorder_started = (
            False
        )

        # attempted_nodes includes a node before its send operation so
        # rollback can issue STOP even if START reached the wire but
        # writer.drain() subsequently failed.
        attempted_nodes: list[int] = []

        started_nodes: list[int] = []

        try:

            # ==========================================================
            # EVENT/DATABASE SESSION
            # ==============================================================

            self.events.start_session(
                session_id,
                normalized_label,
            )

            event_session_started = (
                True
            )

            # ==========================================================
            # CONTINUOUS SOUNDSCAPE SESSION
            # ==========================================================

            if (
                self.soundscape
                is not None
            ):

                self.soundscape.start_session(
                    session_id
                )

            # ==========================================================
            # CLEAR PREVIOUS PCM
            # ==========================================================
            #
            # sampleIndex restarts from zero for every START.
            # ==============================================================

            self.streams.reset_audio_buffers()

            # ==========================================================
            # ACTIVATE LAPTOP SESSION FILTER
            # ==============================================================

            self.active_session_id = (
                session_id
            )

            # ==========================================================
            # CONTINUOUS RECORDING
            # ==============================================================

            if (
                self.config.audio.record_wav
            ):

                recorder_label = (
                    f"{normalized_label}"
                    f"_sid_{session_id:08X}"
                )

                self.recorder.start(
                    recorder_label,
                    sorted(
                        self.config.expected_nodes
                    ),
                )

                recorder_started = (
                    True
                )

            # ==========================================================
            # ARM EVERY SLAVE
            # ==============================================================

            for node_id in (
                self.slave_node_ids
            ):

                attempted_nodes.append(
                    node_id
                )

                await self.send_command(
                    node_id,
                    ControlCommand.START,
                    session_id=session_id,
                )

                started_nodes.append(
                    node_id
                )

            # ==========================================================
            # SLAVE ARMING PERIOD
            # ==============================================================

            if self.slave_node_ids:

                await asyncio.sleep(
                    0.10
                )

            # ==========================================================
            # START MASTER LAST
            # ==============================================================

            attempted_nodes.append(
                MASTER_NODE_ID
            )

            await self.send_command(
                MASTER_NODE_ID,
                ControlCommand.START,
                session_id=session_id,
            )

            started_nodes.append(
                MASTER_NODE_ID
            )

        except asyncio.CancelledError:

            # Cancellation still requires transactional rollback before
            # propagating.
            await self._rollback_start(
                session_id=session_id,
                attempted_nodes=attempted_nodes,
                event_session_started=
                    event_session_started,
                recorder_started=
                    recorder_started,
            )

            raise

        except Exception as exc:

            await self._rollback_start(
                session_id=session_id,
                attempted_nodes=attempted_nodes,
                event_session_started=
                    event_session_started,
                recorder_started=
                    recorder_started,
            )

            raise RuntimeError(
                (
                    "START failed: "
                    f"{exc}"
                )
            ) from exc

        # ==============================================================
        # START COMPLETE
        # ==============================================================

        logger.info(
            (
                "Acquisition START requested "
                "| label=%s "
                "| session=0x%08X "
                "| nodes=%s"
            ),
            normalized_label,
            session_id,
            started_nodes,
        )

        return session_id

    # ==================================================================
    # START ROLLBACK
    # ==================================================================

    async def _rollback_start(
        self,
        *,
        session_id: int,
        attempted_nodes: list[int],
        event_session_started: bool,
        recorder_started: bool,
    ) -> None:
        """
        Roll back any partially completed START transaction.
        """

        # ==============================================================
        # ESP32 ROLLBACK
        # ==============================================================

        for node_id in reversed(
            attempted_nodes
        ):

            with contextlib.suppress(
                Exception
            ):

                await self.send_command(
                    node_id,
                    ControlCommand.STOP,
                    session_id=session_id,
                )

        # ==============================================================
        # RECORDER ROLLBACK
        # ==============================================================

        # WavRecorder.stop() is intentionally safe even when start()
        # failed partway.
        if (
            recorder_started
            or self.config.audio.record_wav
        ):

            with contextlib.suppress(
                Exception
            ):

                self.recorder.stop()

        # ==============================================================
        # EVENT SESSION ROLLBACK
        # ==============================================================

        if event_session_started:

            with contextlib.suppress(
                Exception
            ):

                self.events.stop_session()

        if (
            self.soundscape
            is not None
        ):

            with contextlib.suppress(
                Exception
            ):

                self.soundscape.stop_session(
                    session_id
                )

        # ==============================================================
        # SERVER SESSION STATE
        # ==============================================================

        self.active_session_id = (
            None
        )

    # ==================================================================
    # STOP ACQUISITION
    # ==================================================================

    async def stop_acquisition(
        self,
    ) -> None:
        """
        Stop the active acquisition.

        Current 3-node shutdown order:

            Node 2
                ↓
            Node 3
                ↓
            Node 1

        Master shutdown is last so shared BCLK/WS remains available while
        slaves tear down their I2S receivers.
        """

        async with (
            self._session_lock
        ):

            await self._stop_acquisition_locked()

    async def _stop_acquisition_locked(
        self,
    ) -> None:
        """
        Internal STOP implementation.

        Caller must hold _session_lock.
        """

        session_id = (
            self.active_session_id
        )

        # ==============================================================
        # NO SERVER SESSION
        # ==============================================================

        if session_id is None:

            # Defensive recovery from inconsistent partial state.
            with contextlib.suppress(
                Exception
            ):
                self.recorder.stop()

            if (
                self.soundscape
                is not None
            ):
                with contextlib.suppress(
                    Exception
                ):
                    self.soundscape.stop_session()

            if (
                self.events.active_session_id
                is not None
            ):
                with contextlib.suppress(
                    Exception
                ):
                    self.events.stop_session()

            return

        # ==============================================================
        # STOP ALL SLAVES, THEN MASTER
        # ==============================================================

        stop_order = (
            *self.slave_node_ids,
            MASTER_NODE_ID,
        )

        for node_id in stop_order:

            if (
                node_id
                not in self.connections
            ):
                continue

            try:

                await self.send_command(
                    node_id,
                    ControlCommand.STOP,
                    session_id=session_id,
                )

            except asyncio.CancelledError:

                raise

            except (
                ConnectionError,
                RuntimeError,
                OSError,
                asyncio.TimeoutError,
            ) as exc:

                logger.warning(
                    (
                        "STOP command failed "
                        "for node %d: %s"
                    ),
                    node_id,
                    exc,
                )

        # ==============================================================
        # LOCAL CLEANUP
        # ==============================================================

        cleanup_errors: list[
            Exception
        ] = []

        try:

            self.recorder.stop()

        except Exception as exc:

            cleanup_errors.append(
                exc
            )

            logger.exception(
                (
                    "Continuous recorder "
                    "shutdown failed"
                )
            )

        try:

            self.events.stop_session()

        except Exception as exc:

            cleanup_errors.append(
                exc
            )

            logger.exception(
                (
                    "Event pipeline "
                    "session shutdown failed"
                )
            )

        if (
            self.soundscape
            is not None
        ):

            try:

                self.soundscape.stop_session(
                    session_id
                )

            except Exception as exc:

                cleanup_errors.append(
                    exc
                )

                logger.exception(
                    (
                        "Soundscape analytics "
                        "session shutdown failed"
                    )
                )

        # This must happen even when local persistence encounters a
        # failure. Otherwise the server can become permanently stuck in
        # an active-session state. Every cleanup operation above records
        # its error instead of propagating immediately.
        self.active_session_id = (
            None
        )

        logger.info(
            (
                "Acquisition STOP requested "
                "| session=0x%08X"
            ),
            session_id,
        )

        if cleanup_errors:

            raise RuntimeError(
                (
                    "Acquisition stopped, but "
                    "one or more laptop-side "
                    "cleanup operations failed."
                )
            ) from cleanup_errors[0]

    # ==================================================================
    # PING
    # ==================================================================

    async def ping_all(
        self,
    ) -> None:
        """
        Send PING to every currently registered node.
        """

        node_ids = tuple(
            sorted(
                self.connections.keys()
            )
        )

        if not node_ids:
            return

        results = await asyncio.gather(
            *(
                self.send_command(
                    node_id,
                    ControlCommand.PING,
                    session_id=(
                        self.active_session_id
                        or 0
                    ),
                )
                for node_id
                in node_ids
            ),
            return_exceptions=True,
        )

        for (
            node_id,
            result,
        ) in zip(
            node_ids,
            results,
            strict=False,
        ):

            if isinstance(
                result,
                BaseException,
            ):

                logger.debug(
                    (
                        "PING failed for "
                        "node %d: %s"
                    ),
                    node_id,
                    result,
                )

    # ==================================================================
    # PACKET READER
    # ==================================================================

    async def _read_packet(
        self,
        reader: asyncio.StreamReader,
    ) -> Packet:
        """
        Read and validate one complete Protocol-v4 packet.
        """

        # ==============================================================
        # 40-BYTE HEADER
        # ==============================================================

        header_bytes = (
            await asyncio.wait_for(
                reader.readexactly(
                    HEADER_SIZE
                ),
                timeout=
                    self.config.network.read_timeout_s,
            )
        )

        header = (
            unpack_header(
                header_bytes
            )
        )

        # ==============================================================
        # PAYLOAD SIZE GUARD
        # ==============================================================

        if (
            header.payload_length
            > self.config.network.max_payload_bytes
        ):

            raise ProtocolError(
                (
                    f"node {header.node_id} "
                    f"payload {header.payload_length} "
                    "exceeds configured limit "
                    f"{self.config.network.max_payload_bytes}"
                )
            )

        # ==============================================================
        # PAYLOAD
        # ==============================================================

        if (
            header.payload_length
            == 0
        ):

            payload = b""

        else:

            payload = (
                await asyncio.wait_for(
                    reader.readexactly(
                        header.payload_length
                    ),
                    timeout=
                        self.config.network.read_timeout_s,
                )
            )

        # ==============================================================
        # CRC32
        # ==============================================================

        verify_payload_crc(
            header,
            payload,
        )

        return Packet(
            header=header,
            payload=payload,
        )

    # ==================================================================
    # ACTIVE CONNECTION IDENTITY
    # ==================================================================

    def _connection_is_current(
        self,
        connection: NodeConnection,
    ) -> bool:
        """
        Whether a connection is still the authoritative socket for its
        node.

        This prevents packets from a superseded reconnecting socket from
        entering current buffers after another handler replaced it.
        """

        return (
            self.connections.get(
                connection.node_id
            )
            is connection
        )

    # ==================================================================
    # FIRST HELLO SESSION CHECK
    # ==================================================================

    def _validate_initial_session(
        self,
        packet: Packet,
    ) -> None:
        """
        Validate the session carried by a reconnecting node's initial
        HELLO.

        During an active acquisition a node may reconnect using:

            session 0
                idle/reconnect HELLO

            active session ID
                firmware preserved acquisition state

        A different non-zero session is stale/incompatible.
        """

        active_session = (
            self.active_session_id
        )

        if (
            active_session
            is None
        ):
            return

        packet_session = (
            packet.header.session_id
        )

        if packet_session in {
            0,
            active_session,
        }:
            return

        raise ProtocolError(
            (
                "HELLO belongs to incompatible "
                f"session 0x{packet_session:08X}; "
                "active session is "
                f"0x{active_session:08X}"
            )
        )

    # ==================================================================
    # PACKET SESSION ACCEPTABILITY
    # ==================================================================

    def _packet_session_is_acceptable(
        self,
        packet: Packet,
    ) -> bool:
        """
        Decide whether a packet may alter current NodeState.

        Crucially, this check is performed BEFORE NodeState.observe_header
        so a stale non-zero session cannot reset the current node's
        sample-index buffers.
        """

        header = (
            packet.header
        )

        active_session = (
            self.active_session_id
        )

        # --------------------------------------------------------------
        # AUDIO / ENVIRONMENT / SYNC
        # --------------------------------------------------------------

        if (
            header.packet_type
            in SESSION_BOUND_PACKET_TYPES
        ):

            return (
                active_session
                is not None
                and header.session_id
                == active_session
            )

        # --------------------------------------------------------------
        # HEARTBEAT / REFRESH HELLO
        # --------------------------------------------------------------
        #
        # During an active session, auxiliary packets may use:
        #
        #   0
        #   or active session ID
        #
        # but not another non-zero session.
        # --------------------------------------------------------------

        if (
            active_session
            is not None
            and header.packet_type
            in AUXILIARY_PACKET_TYPES
        ):

            return (
                header.session_id
                in {
                    0,
                    active_session,
                }
            )

        return True

    # ==================================================================
    # CLIENT HANDLER
    # ==================================================================

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Handle one ESP32 TCP connection.
        """

        peer = writer.get_extra_info(
            "peername"
        )

        peer_text = str(
            peer
        )

        connection: (
            NodeConnection
            | None
        ) = None

        state: (
            NodeState
            | None
        ) = None

        try:

            # ==========================================================
            # FIRST PACKET MUST BE HELLO
            # ==============================================================

            first = (
                await asyncio.wait_for(
                    self._read_packet(
                        reader
                    ),
                    timeout=
                        self.config.network.hello_timeout_s,
                )
            )

            if (
                first.header.packet_type
                != PacketType.HELLO
            ):

                raise ProtocolError(
                    "first packet must be HELLO"
                )

            self._validate_initial_session(
                first
            )

            hello = (
                parse_hello(
                    first.payload
                )
            )

            node_id = int(
                first.header.node_id
            )

            # ==========================================================
            # EXPECTED NODE
            # ==============================================================

            if (
                node_id
                not in self.config.expected_nodes
            ):

                raise ProtocolError(
                    (
                        "unexpected node id "
                        f"{node_id}"
                    )
                )

            # ==========================================================
            # HELLO CONTRACT
            # ==============================================================

            self._validate_hello(
                node_id,
                hello,
            )

            # ==========================================================
            # NODE STATE
            # ==============================================================

            state = NodeState(
                node_id=node_id,
                audio_config=self.config.audio,
            )

            state.connected = (
                True
            )

            state.peer = (
                peer_text
            )

            state.hello = (
                hello
            )

            state.observe_header(
                first.header.sequence,
                first.header.session_id,
                first.header.sample_index,
                first.header.flags,
            )

            connection = NodeConnection(
                state,
                reader,
                writer,
            )

            # ==========================================================
            # REGISTER / REPLACE CONNECTION
            # ==============================================================

            old_connection: (
                NodeConnection
                | None
            ) = None

            async with (
                self._connection_lock
            ):

                old_connection = (
                    self.connections.get(
                        node_id
                    )
                )

                self.connections[
                    node_id
                ] = connection

                self.streams.register_state(
                    state
                )

            # ----------------------------------------------------------
            # CLOSE PREVIOUS SOCKET OUTSIDE LOCK
            # ----------------------------------------------------------

            if (
                old_connection
                is not None
                and old_connection
                is not connection
            ):

                logger.warning(
                    (
                        "Node %d reconnected; "
                        "closing previous socket"
                    ),
                    node_id,
                )

                with contextlib.suppress(
                    Exception
                ):

                    await old_connection.close()

            logger.info(
                (
                    "Node %d connected from %s "
                    "| master=%s "
                    "| fw=%s "
                    "| %d Hz"
                ),
                node_id,
                peer_text,
                hello.master_node,
                hello.firmware,
                hello.sample_rate,
            )

            # ==========================================================
            # PACKET LOOP
            # ==============================================================

            while True:

                # ------------------------------------------------------
                # SUPERSEDED SOCKET CHECK
                # ------------------------------------------------------

                if not (
                    self._connection_is_current(
                        connection
                    )
                ):

                    logger.debug(
                        (
                            "Stopping superseded "
                            "handler for node %d"
                        ),
                        node_id,
                    )

                    break

                try:

                    packet = (
                        await self._read_packet(
                            reader
                        )
                    )

                except CRCMismatch as exc:

                    state.crc_errors += (
                        1
                    )

                    logger.warning(
                        "%s",
                        exc,
                    )

                    continue

                # ------------------------------------------------------
                # NODE ID IMMUTABILITY
                # ------------------------------------------------------

                if (
                    packet.header.node_id
                    != node_id
                ):

                    raise ProtocolError(
                        (
                            "socket registered as node "
                            f"{node_id}, got packet for "
                            f"node {packet.header.node_id}"
                        )
                    )

                # ------------------------------------------------------
                # CONNECTION MAY HAVE BEEN REPLACED WHILE AWAITING READ
                # ------------------------------------------------------

                if not (
                    self._connection_is_current(
                        connection
                    )
                ):

                    break

                # ------------------------------------------------------
                # STALE SESSION FILTER BEFORE NODESTATE MUTATION
                # ------------------------------------------------------

                if not (
                    self._packet_session_is_acceptable(
                        packet
                    )
                ):

                    self._log_stale_packet(
                        state,
                        packet,
                    )

                    continue

                # ------------------------------------------------------
                # NODE DIAGNOSTICS
                # ------------------------------------------------------

                state.observe_header(
                    packet.header.sequence,
                    packet.header.session_id,
                    packet.header.sample_index,
                    packet.header.flags,
                )

                # ------------------------------------------------------
                # PAYLOAD DISPATCH
                # ------------------------------------------------------

                await self._dispatch_packet(
                    state,
                    packet,
                )

        except asyncio.IncompleteReadError:

            logger.info(
                (
                    "Connection closed "
                    "by %s"
                ),
                peer_text,
            )

        except asyncio.TimeoutError:

            logger.warning(
                (
                    "Connection timed out: "
                    "%s"
                ),
                peer_text,
            )

        except (
            ProtocolError,
            ValueError,
            TypeError,
        ) as exc:

            if state is not None:

                state.protocol_errors += (
                    1
                )

            logger.warning(
                (
                    "Protocol error from "
                    "%s: %s"
                ),
                peer_text,
                exc,
            )

        except (
            ConnectionError,
            OSError,
        ) as exc:

            logger.info(
                (
                    "Connection error from "
                    "%s: %s"
                ),
                peer_text,
                exc,
            )

        except asyncio.CancelledError:

            raise

        except Exception:

            logger.exception(
                (
                    "Unexpected client-handler "
                    "failure from %s"
                ),
                peer_text,
            )

        finally:

            if state is not None:

                state.connected = (
                    False
                )

            if connection is not None:

                async with (
                    self._connection_lock
                ):

                    if (
                        self.connections.get(
                            connection.node_id
                        )
                        is connection
                    ):

                        self.connections.pop(
                            connection.node_id,
                            None,
                        )

                with contextlib.suppress(
                    Exception
                ):

                    await connection.close()

    # ==================================================================
    # HELLO VALIDATION
    # ==================================================================

    def _validate_hello(
        self,
        node_id: int,
        hello: HelloPayload,
    ) -> None:
        """
        Validate firmware/audio configuration reported by HELLO.
        """

        # ==============================================================
        # SAMPLE RATE
        # ==============================================================

        if (
            hello.sample_rate
            != self.config.audio.sample_rate
        ):

            raise ProtocolError(
                (
                    f"node {node_id} sample rate "
                    f"{hello.sample_rate} != expected "
                    f"{self.config.audio.sample_rate}"
                )
            )

        # ==============================================================
        # BLOCK SIZE
        # ==============================================================

        if (
            hello.frames_per_packet
            != self.config.audio.frames_per_block
        ):

            raise ProtocolError(
                (
                    f"node {node_id} frames/packet "
                    f"{hello.frames_per_packet} "
                    "!= expected "
                    f"{self.config.audio.frames_per_block}"
                )
            )

        # ==============================================================
        # PCM FORMAT
        # ==============================================================

        if (
            hello.bits_per_sample
            != 16
        ):

            raise ProtocolError(
                (
                    f"node {node_id} reports "
                    f"{hello.bits_per_sample}-bit audio; "
                    "Protocol-v4 transport requires PCM16"
                )
            )

        if (
            hello.channels
            != self.config.audio.channels
        ):

            raise ProtocolError(
                (
                    f"node {node_id} reports "
                    f"{hello.channels} channel(s); "
                    f"expected "
                    f"{self.config.audio.channels}"
                )
            )

        # ==============================================================
        # SYNC TOLERANCE
        # ==============================================================

        if (
            hello.sync_tolerance_samples
            != self.config.audio.sync_tolerance_samples
        ):

            raise ProtocolError(
                (
                    f"node {node_id} sync tolerance "
                    f"{hello.sync_tolerance_samples} "
                    "!= laptop configuration "
                    f"{self.config.audio.sync_tolerance_samples}"
                )
            )

        # ==============================================================
        # MASTER / SLAVE ROLE
        # ==============================================================

        expected_master = (
            node_id
            == MASTER_NODE_ID
        )

        if (
            hello.master_node
            != expected_master
        ):

            expected_role = (
                "master"
                if expected_master
                else "slave"
            )

            raise ProtocolError(
                (
                    f"node {node_id} must identify "
                    f"as {expected_role}"
                )
            )

        # ==============================================================
        # FIRMWARE IDENTIFIER
        # ==============================================================

        if not (
            hello.firmware.strip()
        ):

            raise ProtocolError(
                (
                    f"node {node_id} reported "
                    "an empty firmware identifier"
                )
            )

    # ==================================================================
    # CURRENT SESSION
    # ==================================================================

    def _packet_is_current_session(
        self,
        packet: Packet,
    ) -> bool:
        """
        Whether a session-bound packet belongs to the active acquisition.
        """

        session_id = (
            self.active_session_id
        )

        if session_id is None:

            return False

        return (
            packet.header.session_id
            == session_id
        )

    # ==================================================================
    # STALE PACKET LOG
    # ==================================================================

    def _log_stale_packet(
        self,
        state: NodeState,
        packet: Packet,
    ) -> None:
        """
        Log ignored session-incompatible data.
        """

        active_session = (
            "NONE"
            if self.active_session_id
            is None
            else (
                f"0x{self.active_session_id:08X}"
            )
        )

        logger.debug(
            (
                "Ignoring stale/incompatible %s "
                "node=%d "
                "packet_session=0x%08X "
                "active_session=%s"
            ),
            packet.header.packet_type.name,
            state.node_id,
            packet.header.session_id,
            active_session,
        )

    # ==================================================================
    # PACKET DISPATCH
    # ==================================================================

    async def _dispatch_packet(
        self,
        state: NodeState,
        packet: Packet,
    ) -> None:
        """
        Route one validated Protocol-v4 packet.
        """

        header = (
            packet.header
        )

        # --------------------------------------------------------------
        # DEFENSIVE SESSION FILTER
        # --------------------------------------------------------------
        #
        # _handle_client performs this check before observe_header().
        #
        # Keeping the guard here as well ensures direct unit-test calls
        # to _dispatch_packet cannot bypass the session contract.
        # --------------------------------------------------------------

        if not (
            self._packet_session_is_acceptable(
                packet
            )
        ):

            self._log_stale_packet(
                state,
                packet,
            )

            return

        # ==============================================================
        # AUDIO
        # ==============================================================

        if (
            header.packet_type
            == PacketType.AUDIO
        ):

            # ----------------------------------------------------------
            # ACTIVE SESSION
            # ----------------------------------------------------------

            if not (
                self._packet_is_current_session(
                    packet
                )
            ):

                self._log_stale_packet(
                    state,
                    packet,
                )

                return

            # ----------------------------------------------------------
            # EXACT PCM16 PAYLOAD SIZE
            # ----------------------------------------------------------

            expected_payload_bytes = (
                self.config.audio.bytes_per_block
            )

            actual_payload_bytes = (
                len(
                    packet.payload
                )
            )

            if (
                actual_payload_bytes
                != expected_payload_bytes
            ):

                raise ProtocolError(
                    (
                        f"node {state.node_id} AUDIO "
                        f"payload is {actual_payload_bytes} bytes; "
                        f"expected exactly "
                        f"{expected_payload_bytes}"
                    )
                )

            # ----------------------------------------------------------
            # PCM16 DECODE
            # ----------------------------------------------------------

            samples = (
                np.frombuffer(
                    packet.payload,
                    dtype="<i2",
                )
                .copy()
            )

            if (
                samples.size
                != self.config.audio.frames_per_block
            ):

                raise ProtocolError(
                    (
                        f"node {state.node_id} AUDIO block "
                        f"contains {samples.size} samples; "
                        f"expected "
                        f"{self.config.audio.frames_per_block}"
                    )
                )

            # ----------------------------------------------------------
            # AUDIO BLOCK MODEL
            # ----------------------------------------------------------

            block = AudioBlock(
                node_id=
                    state.node_id,

                sequence=
                    header.sequence,

                session_id=
                    header.session_id,

                sample_index=
                    header.sample_index,

                local_micros=
                    header.local_micros,

                i2s_error_count=
                    header.i2s_error_count,

                flags=
                    header.flags,

                samples=
                    samples,
            )

            # ----------------------------------------------------------
            # STREAM BUFFER
            # ----------------------------------------------------------

            self.streams.add_audio(
                block
            )

            # ----------------------------------------------------------
            # CONTINUOUS SOUNDSCAPE INDICES
            # ----------------------------------------------------------

            if (
                self.soundscape
                is not None
            ):

                try:

                    self.soundscape.append_audio(
                        block.node_id,
                        block.samples,
                        session_id=block.session_id,
                        start_sample=block.sample_index,
                    )

                except Exception:

                    # Continuous research analytics are optional. A
                    # calculation or persistence failure must remain
                    # visible without interrupting acquisition.
                    logger.exception(
                        (
                            "Continuous soundscape "
                            "processing failed "
                            "| node=%d "
                            "| session=0x%08X "
                            "| sample=%d"
                        ),
                        block.node_id,
                        block.session_id,
                        block.sample_index,
                    )

            # ----------------------------------------------------------
            # CONTINUOUS WAV
            # ----------------------------------------------------------

            if (
                self.config.audio.record_wav
            ):

                self.recorder.write(
                    block
                )

            # ----------------------------------------------------------
            # EVENT PIPELINE
            # ----------------------------------------------------------

            self.events.on_audio(
                block
            )

            return

        # ==============================================================
        # ENVIRONMENT
        # ==============================================================

        if (
            header.packet_type
            == PacketType.ENVIRONMENT
        ):

            if not (
                self._packet_is_current_session(
                    packet
                )
            ):

                self._log_stale_packet(
                    state,
                    packet,
                )

                return

            # ----------------------------------------------------------
            # CURRENT HARDWARE CONTRACT:
            # BME280 BELONGS TO NODE 1 MASTER
            # ----------------------------------------------------------

            if (
                state.node_id
                != MASTER_NODE_ID
            ):

                raise ProtocolError(
                    (
                        "ENVIRONMENT packet received "
                        f"from slave node {state.node_id}; "
                        "current architecture expects "
                        "BME280 telemetry from Node 1"
                    )
                )

            environment = (
                parse_environment(
                    packet.payload
                )
            )

            environment_sample = (
                EnvironmentSample(
                    node_id=
                        state.node_id,

                    session_id=
                        header.session_id,

                    sample_index=
                        header.sample_index,

                    value=
                        environment,
                )
            )

            # ----------------------------------------------------------
            # NODE RUNTIME HISTORY
            # ----------------------------------------------------------

            state.add_environment(
                environment_sample
            )

            # ----------------------------------------------------------
            # DATABASE
            # ----------------------------------------------------------

            self.events.add_environment(
                node_id=
                    state.node_id,

                session_id=
                    header.session_id,

                sample_index=
                    header.sample_index,

                environment=
                    environment,
            )

            return

        # ==============================================================
        # HEARTBEAT
        # ==============================================================

        if (
            header.packet_type
            == PacketType.HEARTBEAT
        ):

            if (
                state.hello
                is None
            ):

                raise ProtocolError(
                    (
                        f"node {state.node_id} "
                        "HEARTBEAT received "
                        "before HELLO"
                    )
                )

            state.latest_heartbeat = (
                parse_heartbeat(
                    packet.payload,
                    master_node=
                        state.hello.master_node,
                )
            )

            return

        # ==============================================================
        # SYNC
        # ==============================================================

        if (
            header.packet_type
            == PacketType.SYNC
        ):

            if not (
                self._packet_is_current_session(
                    packet
                )
            ):

                self._log_stale_packet(
                    state,
                    packet,
                )

                return

            sync = (
                parse_sync(
                    packet.payload
                )
            )

            # ----------------------------------------------------------
            # SESSION CONSISTENCY
            # ----------------------------------------------------------

            if (
                sync.session_id
                != header.session_id
            ):

                raise ProtocolError(
                    (
                        f"node {state.node_id} SYNC "
                        "payload session "
                        f"0x{sync.session_id:08X} "
                        "!= header session "
                        f"0x{header.session_id:08X}"
                    )
                )

            # ----------------------------------------------------------
            # SAMPLE-INDEX CONSISTENCY
            # ----------------------------------------------------------

            if (
                sync.sample_index
                != header.sample_index
            ):

                raise ProtocolError(
                    (
                        f"node {state.node_id} SYNC "
                        "payload sampleIndex "
                        f"{sync.sample_index} "
                        "!= header sampleIndex "
                        f"{header.sample_index}"
                    )
                )

            # localMicros is deliberately NOT compared.
            #
            # Payload and header may be generated a few microseconds
            # apart, and localMicros is only a diagnostic clock.

            state.latest_sync = (
                sync
            )

            return

        # ==============================================================
        # HELLO REFRESH
        # ==============================================================

        if (
            header.packet_type
            == PacketType.HELLO
        ):

            hello = (
                parse_hello(
                    packet.payload
                )
            )

            self._validate_hello(
                state.node_id,
                hello,
            )

            state.hello = (
                hello
            )

            return

        # ==============================================================
        # UNHANDLED PACKET
        # ==============================================================

        raise ProtocolError(
            (
                "unhandled packet type "
                f"{header.packet_type}"
            )
        )
