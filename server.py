from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets

import numpy as np

from config import AppConfig

from event_pipeline import (
    EventPipeline,
)

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
# RECEIVER SERVER
# ======================================================================


class ReceiverServer:
    """
    Async TCP receiver for the three synchronized ESP32 acoustic nodes.

    Responsibilities
    ----------------
    - accept ESP32 TCP connections
    - validate node HELLO packets
    - receive binary protocol packets
    - maintain node connection state
    - manage synchronized acquisition sessions
    - route audio into StreamManager
    - route events into EventPipeline
    - persist environmental telemetry
    - control continuous WAV recording

    Classification backend selection is intentionally NOT handled here.

    EventPipeline obtains its classification backend through the
    classification factory using AppConfig.
    """

    def __init__(
        self,
        config: AppConfig,
    ) -> None:

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
        # CONTINUOUS WAV RECORDER
        # ==============================================================

        self.recorder = WavRecorder(
            config.recordings_dir,
            config.audio,
        )

        # ==============================================================
        # EVENT PIPELINE
        # ==============================================================
        #
        # EventPipeline internally obtains the configured classification
        # backend through classification.factory.
        # ==============================================================

        self.events = EventPipeline(
            self.streams,
            config,
        )

        # ==============================================================
        # TCP SERVER
        # ==============================================================

        self._server: (
            asyncio.AbstractServer
            | None
        ) = None

        # ==============================================================
        # CONNECTION SYNCHRONIZATION
        # ==============================================================

        self._connection_lock = (
            asyncio.Lock()
        )

        # ==============================================================
        # ACTIVE ACQUISITION SESSION
        # ==============================================================

        self.active_session_id: (
            int
            | None
        ) = None

    # ==================================================================
    # SERVER START
    # ==================================================================

    async def start(
        self,
    ) -> None:
        """
        Start the laptop-side TCP receiver.
        """

        if self._server is not None:

            return

        self._server = (
            await asyncio.start_server(
                self._handle_client,

                self.config.network.host,

                self.config.network.port,

                limit=max(
                    64 * 1024,
                    (
                        self.config.network.max_payload_bytes
                        + HEADER_SIZE
                    ),
                ),
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
        Run the TCP server until cancelled.
        """

        if self._server is None:

            await self.start()

        assert (
            self._server
            is not None
        )

        async with self._server:

            await self._server.serve_forever()

    # ==================================================================
    # CLOSE
    # ==================================================================

    async def close(
        self,
    ) -> None:
        """
        Gracefully close acquisition, TCP listener and node sockets.
        """

        # ==============================================================
        # STOP ACTIVE SESSION
        # ==============================================================

        if (
            self.active_session_id
            is not None
        ):

            try:

                await self.stop_acquisition()

            except Exception:

                logger.exception(
                    (
                        "Failed to stop active acquisition "
                        "during server shutdown"
                    )
                )

        else:

            with contextlib.suppress(
                Exception
            ):

                self.recorder.stop()

        # ==============================================================
        # CLOSE LISTENER
        # ==============================================================

        if (
            self._server
            is not None
        ):

            self._server.close()

            await self._server.wait_closed()

            self._server = None

        # ==============================================================
        # DETACH ACTIVE NODE CONNECTIONS
        # ==============================================================

        async with (
            self._connection_lock
        ):

            connections = list(
                self.connections.values()
            )

            self.connections.clear()

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
    # EXPECTED NODE STATE
    # ==================================================================

    def all_expected_nodes_connected(
        self,
    ) -> bool:
        """
        Whether all configured acoustic nodes currently have TCP
        connections.
        """

        return (
            self.config.expected_nodes
            .issubset(
                self.connections.keys()
            )
        )

    async def wait_for_nodes(
        self,
        timeout: float | None = None,
    ) -> None:
        """
        Wait until all configured nodes are connected.
        """

        async def _wait() -> None:

            while (
                not self.all_expected_nodes_connected()
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
        Send one control command to a connected ESP32 node.
        """

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

        await connection.send_command(
            command,
            session_id=session_id,
        )

    # ==================================================================
    # SESSION ID GENERATION
    # ==================================================================

    def _generate_session_id(
        self,
    ) -> int:
        """
        Generate a non-zero 32-bit acquisition session identifier.

        Session IDs originate exclusively on the laptop and the SAME
        value is transmitted to all three nodes.
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

        Required startup order
        ----------------------
        Node 2 slave
            ↓
        Node 3 slave
            ↓
        short arming delay
            ↓
        Node 1 master

        Node 1 starts the shared BCLK/WS audio clock last.

        All nodes receive the SAME laptop-generated session ID.

        Startup is transactional: failure at any stage rolls back local
        state and sends STOP to any nodes that were already started.
        """

        # ==============================================================
        # PREVENT OVERLAPPING SESSIONS
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

        missing = (
            self.config.expected_nodes
            .difference(
                self.connections.keys()
            )
        )

        if missing:

            raise RuntimeError(
                (
                    "cannot START; missing nodes: "
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
        # CLEAR PREVIOUS SAMPLE TIMELINE
        # ==============================================================
        #
        # ESP32 sampleIndex resets to zero on every START.
        #
        # Any previous-session PCM therefore must be removed before the
        # new sample-index timeline begins.
        # ==============================================================

        self.streams.reset_audio_buffers()

        # ==============================================================
        # TRANSACTION STATE
        # ==============================================================

        event_session_started = False

        recorder_started = False

        started_nodes: list[int] = []

        try:

            # ==========================================================
            # START LAPTOP-SIDE EVENT SESSION
            # ==========================================================

            self.events.start_session(
                session_id,
                session_label,
            )

            event_session_started = True

            # ----------------------------------------------------------
            # Mark session active BEFORE nodes begin transmitting.
            #
            # _dispatch_packet() rejects AUDIO/ENVIRONMENT/SYNC packets
            # whose session does not match active_session_id.
            # ----------------------------------------------------------

            self.active_session_id = (
                session_id
            )

            # ==========================================================
            # CONTINUOUS WAV RECORDING
            # ==========================================================

            if (
                self.config.audio.record_wav
            ):

                self.recorder.start(
                    (
                        f"{session_label}"
                        f"_sid_{session_id:08X}"
                    ),
                    sorted(
                        self.config.expected_nodes
                    ),
                )

                recorder_started = True

            # ==========================================================
            # ARM NODE 2
            # ==========================================================

            await self.send_command(
                2,
                ControlCommand.START,
                session_id=session_id,
            )

            started_nodes.append(
                2
            )

            # ==========================================================
            # ARM NODE 3
            # ==========================================================

            await self.send_command(
                3,
                ControlCommand.START,
                session_id=session_id,
            )

            started_nodes.append(
                3
            )

            # ==========================================================
            # ALLOW SLAVE I2S RECEIVERS TO ARM
            # ==========================================================

            await asyncio.sleep(
                0.10
            )

            # ==========================================================
            # START MASTER LAST
            # ==========================================================

            await self.send_command(
                1,
                ControlCommand.START,
                session_id=session_id,
            )

            started_nodes.append(
                1
            )

        except Exception as exc:

            # ==========================================================
            # ROLLBACK ESP32 NODES
            # ==========================================================

            for node_id in reversed(
                started_nodes
            ):

                with contextlib.suppress(
                    Exception
                ):

                    await self.send_command(
                        node_id,
                        ControlCommand.STOP,
                        session_id=session_id,
                    )

            # ==========================================================
            # ROLLBACK RECORDER
            # ==========================================================

            if recorder_started:

                with contextlib.suppress(
                    Exception
                ):

                    self.recorder.stop()

            else:

                # Safe even if recorder.start() failed halfway.
                with contextlib.suppress(
                    Exception
                ):

                    self.recorder.stop()

            # ==========================================================
            # ROLLBACK EVENT DATABASE SESSION
            # ==========================================================

            if event_session_started:

                with contextlib.suppress(
                    Exception
                ):

                    self.events.stop_session()

            # ==========================================================
            # CLEAR ACTIVE SESSION
            # ==========================================================

            self.active_session_id = (
                None
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
            session_label,
            session_id,
            sorted(
                started_nodes
            ),
        )

        return session_id

    # ==================================================================
    # STOP ACQUISITION
    # ==================================================================

    async def stop_acquisition(
        self,
    ) -> None:
        """
        Stop one synchronized acquisition session.

        Shutdown order
        --------------
        Node 2 slave
            ↓
        Node 3 slave
            ↓
        Node 1 master

        The master is stopped last so the shared BCLK/WS remains
        available while slave receivers are being shut down.
        """

        session_id = (
            self.active_session_id
        )

        # ==============================================================
        # NO ACTIVE SESSION
        # ==============================================================

        if session_id is None:

            with contextlib.suppress(
                Exception
            ):

                self.recorder.stop()

            return

        # ==============================================================
        # STOP SLAVES THEN MASTER
        # ==============================================================

        for node_id in (
            2,
            3,
            1,
        ):

            if (
                node_id
                not in self.connections
            ):

                continue

            with contextlib.suppress(
                ConnectionError,
                RuntimeError,
                OSError,
                asyncio.TimeoutError,
            ):

                await self.send_command(
                    node_id,
                    ControlCommand.STOP,
                    session_id=session_id,
                )

        # ==============================================================
        # CLOSE LAPTOP-SIDE SESSION
        # ==============================================================

        try:

            self.recorder.stop()

            self.events.stop_session()

        finally:

            # ----------------------------------------------------------
            # Clear this even when recorder/database cleanup encounters
            # an exception. Otherwise the server can become permanently
            # stuck believing an acquisition is still active.
            # ----------------------------------------------------------

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

    # ==================================================================
    # PING
    # ==================================================================

    async def ping_all(
        self,
    ) -> None:
        """
        Send PING to every currently connected node.
        """

        connections = list(
            self.connections.values()
        )

        if not connections:

            return

        await asyncio.gather(
            *(
                connection.send_command(
                    ControlCommand.PING,
                    session_id=(
                        self.active_session_id
                        or 0
                    ),
                )
                for connection
                in connections
            ),
            return_exceptions=True,
        )

    # ==================================================================
    # PACKET READER
    # ==================================================================

    async def _read_packet(
        self,
        reader: asyncio.StreamReader,
    ) -> Packet:
        """
        Read, validate and return one complete binary protocol packet.
        """

        # ==============================================================
        # FIXED HEADER
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

        header = unpack_header(
            header_bytes
        )

        # ==============================================================
        # PAYLOAD LIMIT
        # ==============================================================

        if (
            header.payload_length
            > self.config.network.max_payload_bytes
        ):

            raise ProtocolError(
                (
                    f"node {header.node_id} payload "
                    f"{header.payload_length} exceeds "
                    f"limit "
                    f"{self.config.network.max_payload_bytes}"
                )
            )

        # ==============================================================
        # PAYLOAD
        # ==============================================================

        payload = b""

        if (
            header.payload_length
            > 0
        ):

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
            # ==========================================================

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
                    (
                        "first packet must "
                        "be HELLO"
                    )
                )

            hello = parse_hello(
                first.payload
            )

            node_id = int(
                first.header.node_id
            )

            # ==========================================================
            # NODE ID
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
            # HELLO CONTENT
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

            state.connected = True

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

            connection = (
                NodeConnection(
                    state,
                    reader,
                    writer,
                )
            )

            # ==========================================================
            # REGISTER CONNECTION
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
            # Close old socket OUTSIDE the lock.
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
            # NORMAL PACKET LOOP
            # ==============================================================

            while True:

                try:

                    packet = (
                        await self._read_packet(
                            reader
                        )
                    )

                except CRCMismatch as exc:

                    state.crc_errors += 1

                    logger.warning(
                        "%s",
                        exc,
                    )

                    continue

                # ------------------------------------------------------
                # Node ID cannot change within one TCP connection.
                # ------------------------------------------------------

                if (
                    packet.header.node_id
                    != node_id
                ):

                    raise ProtocolError(
                        (
                            "socket registered as node "
                            f"{node_id}, got packet for node "
                            f"{packet.header.node_id}"
                        )
                    )

                # ------------------------------------------------------
                # Diagnostic counters / sequence tracking
                # ------------------------------------------------------

                state.observe_header(
                    packet.header.sequence,
                    packet.header.session_id,
                    packet.header.sample_index,
                    packet.header.flags,
                )

                await self._dispatch_packet(
                    state,
                    packet,
                )

        except asyncio.IncompleteReadError:

            logger.info(
                "Connection closed by %s",
                peer_text,
            )

        except asyncio.TimeoutError:

            logger.warning(
                "Connection timed out: %s",
                peer_text,
            )

        except (
            ProtocolError,
            ValueError,
        ) as exc:

            if state is not None:

                state.protocol_errors += 1

            logger.warning(
                (
                    "Protocol error from %s: %s"
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
                    "Connection error from %s: %s"
                ),
                peer_text,
                exc,
            )

        except asyncio.CancelledError:

            raise

        except Exception:

            logger.exception(
                (
                    "Unexpected client-handler failure "
                    "from %s"
                ),
                peer_text,
            )

        finally:

            if state is not None:

                state.connected = False

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
        hello,
    ) -> None:
        """
        Validate node firmware/audio configuration reported by HELLO.
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
                    f"node {node_id} packet frames "
                    f"{hello.frames_per_packet} != expected "
                    f"{self.config.audio.frames_per_block}"
                )
            )

        # ==============================================================
        # PCM FORMAT
        # ==============================================================

        if (
            hello.bits_per_sample
            != 16
            or hello.channels
            != 1
        ):

            raise ProtocolError(
                (
                    f"node {node_id} unsupported audio "
                    f"format: "
                    f"{hello.bits_per_sample}-bit, "
                    f"{hello.channels} channel(s)"
                )
            )

        # ==============================================================
        # MASTER / SLAVE ROLE
        # ==============================================================

        if (
            node_id == 1
            and not hello.master_node
        ):

            raise ProtocolError(
                (
                    "node 1 must identify "
                    "as master"
                )
            )

        if (
            node_id in (
                2,
                3,
            )
            and hello.master_node
        ):

            raise ProtocolError(
                (
                    f"node {node_id} must "
                    "identify as slave"
                )
            )

    # ==================================================================
    # CURRENT SESSION CHECK
    # ==================================================================

    def _packet_is_current_session(
        self,
        packet: Packet,
    ) -> bool:
        """
        Return whether a packet belongs to the active acquisition.

        AUDIO, ENVIRONMENT and SYNC packets are ignored whenever their
        session ID does not match the current laptop session.
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
        Emit a debug-level diagnostic for stale session data.
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
                "Ignoring stale %s "
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
        Route one validated protocol packet.
        """

        header = (
            packet.header
        )

        # ==============================================================
        # AUDIO
        # ==============================================================

        if (
            header.packet_type
            == PacketType.AUDIO
        ):

            # ----------------------------------------------------------
            # SESSION CHECK
            # ----------------------------------------------------------

            if (
                not self._packet_is_current_session(
                    packet
                )
            ):

                self._log_stale_packet(
                    state,
                    packet,
                )

                return

            # ----------------------------------------------------------
            # PCM16 BYTE ALIGNMENT
            # ----------------------------------------------------------

            if (
                len(
                    packet.payload
                )
                % 2
            ):

                raise ProtocolError(
                    (
                        f"node {state.node_id} "
                        "odd PCM16 payload length"
                    )
                )

            # ----------------------------------------------------------
            # ZERO-COPY DECODE + LOCAL COPY
            # ----------------------------------------------------------

            samples = np.frombuffer(
                packet.payload,
                dtype="<i2",
            ).copy()

            if (
                samples.size
                == 0
            ):

                return

            # ----------------------------------------------------------
            # FRAME COUNT
            # ----------------------------------------------------------

            if (
                samples.size
                > self.config.audio.frames_per_block
            ):

                raise ProtocolError(
                    (
                        f"node {state.node_id} audio "
                        f"block {samples.size} exceeds "
                        "configured block size"
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
            # STREAM STORAGE
            # ----------------------------------------------------------

            self.streams.add_audio(
                block
            )

            # ----------------------------------------------------------
            # CONTINUOUS WAV
            # ----------------------------------------------------------

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

            if (
                not self._packet_is_current_session(
                    packet
                )
            ):

                self._log_stale_packet(
                    state,
                    packet,
                )

                return

            env = parse_environment(
                packet.payload
            )

            # ----------------------------------------------------------
            # NODE RUNTIME STATE
            # ----------------------------------------------------------

            state.add_environment(
                EnvironmentSample(
                    node_id=
                        state.node_id,

                    session_id=
                        header.session_id,

                    sample_index=
                        header.sample_index,

                    value=
                        env,
                )
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
                    env,
            )

            return

        # ==============================================================
        # HEARTBEAT
        # ==============================================================

        if (
            header.packet_type
            == PacketType.HEARTBEAT
        ):

            if state.hello is None:

                raise ProtocolError(
                    (
                        f"node {state.node_id} "
                        "HEARTBEAT received before HELLO"
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

            if (
                not self._packet_is_current_session(
                    packet
                )
            ):

                self._log_stale_packet(
                    state,
                    packet,
                )

                return

            sync = parse_sync(
                packet.payload
            )

            # ----------------------------------------------------------
            # PAYLOAD/HEADER SESSION CONSISTENCY
            # ----------------------------------------------------------

            if (
                sync.session_id
                != header.session_id
            ):

                raise ProtocolError(
                    (
                        f"node {state.node_id} SYNC "
                        f"payload session "
                        f"0x{sync.session_id:08X} "
                        f"!= header session "
                        f"0x{header.session_id:08X}"
                    )
                )

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

            hello = parse_hello(
                packet.payload
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
        # UNKNOWN PACKET TYPE
        # ==============================================================

        raise ProtocolError(
            (
                "unhandled packet type "
                f"{header.packet_type}"
            )
        )