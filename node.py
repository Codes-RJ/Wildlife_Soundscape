from __future__ import annotations

import asyncio

from collections import deque

from dataclasses import (
    dataclass,
    field,
)

from typing import Deque

from config import (
    AudioConfig,
)

from models import (
    AudioBlock,
    EnvironmentSample,
    NodeSnapshot,
)

from protocol import (
    ControlCommand,
    ControlFrame,
    EnvironmentPayload,
    HeartbeatPayload,
    HelloPayload,
    PacketFlags,
    SyncPayload,
    pack_control,
)


# ======================================================================
# INTEGER LIMITS
# ======================================================================


UINT8_MAX = (
    0xFF
)

UINT32_MASK = (
    0xFFFFFFFF
)

UINT32_HALF = (
    0x80000000
)

UINT32_MAX = (
    0xFFFFFFFF
)

UINT64_MAX = (
    0xFFFFFFFFFFFFFFFF
)


# ======================================================================
# SEQUENCE CLASSIFICATION RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SequenceResult:
    """
    Diagnostic interpretation of one packet sequence number.

    gap
        Number of packet sequence values apparently skipped.

    duplicate_or_old
        True when the incoming sequence appears to be an older or
        duplicate value rather than a valid forward progression.

    Notes
    -----
    Sequence values are uint32 counters.

    Normal uint32 wraparound is supported.

    Sequence tracking is used only for transport diagnostics.

    Cross-node acoustic synchronization is based on sampleIndex and the
    shared I2S clock.
    """

    gap: int = (
        0
    )

    duplicate_or_old: bool = (
        False
    )


# ======================================================================
# INTEGER VALIDATION
# ======================================================================


def _validate_integer(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a genuine integer.

    bool is deliberately rejected because Python considers bool a
    subclass of int.
    """

    if isinstance(
        value,
        bool,
    ):

        raise TypeError(
            (
                f"{name} must be "
                "an integer."
            )
        )

    if not isinstance(
        value,
        int,
    ):

        raise TypeError(
            (
                f"{name} must be "
                "an integer."
            )
        )

    return int(
        value
    )


def _validate_uint8(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate one unsigned 8-bit integer.
    """

    value = (
        _validate_integer(
            value,
            name=name,
        )
    )

    if not (
        0
        <= value
        <= UINT8_MAX
    ):

        raise ValueError(
            (
                f"{name} must lie in "
                "the uint8 range."
            )
        )

    return value


def _validate_uint32(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate one unsigned 32-bit integer.
    """

    value = (
        _validate_integer(
            value,
            name=name,
        )
    )

    if not (
        0
        <= value
        <= UINT32_MAX
    ):

        raise ValueError(
            (
                f"{name} must lie in "
                "the uint32 range."
            )
        )

    return value


def _validate_uint64(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate one unsigned 64-bit integer.
    """

    value = (
        _validate_integer(
            value,
            name=name,
        )
    )

    if not (
        0
        <= value
        <= UINT64_MAX
    ):

        raise ValueError(
            (
                f"{name} must lie in "
                "the uint64 range."
            )
        )

    return value


# ======================================================================
# SEQUENCE CLASSIFICATION
# ======================================================================


def classify_sequence(
    last_sequence: int | None,
    new_sequence: int,
) -> SequenceResult:
    """
    Compare packet sequence numbers with uint32 wraparound support.

    Examples
    --------
    Normal:

        last = 10
        new  = 11

        gap = 0


    Gap:

        last = 10
        new  = 14

        gap = 3


    Wraparound:

        last = 0xFFFFFFFF
        new  = 0

        normal progression


    Duplicate / old:

        last = 100
        new  = 98

        duplicate_or_old = True


    Sequence numbers are network diagnostics only.

    They are not used for TDOA alignment.
    """

    new_sequence = (
        _validate_uint32(
            new_sequence,
            name=
                "new_sequence",
        )
    )

    # ==================================================================
    # FIRST PACKET
    # ==================================================================

    if (
        last_sequence
        is None
    ):

        return SequenceResult()

    last_sequence = (
        _validate_uint32(
            last_sequence,
            name=
                "last_sequence",
        )
    )

    # ==================================================================
    # EXPECTED NEXT VALUE
    # ==================================================================

    expected = (
        last_sequence
        + 1
    ) & UINT32_MASK

    # ==================================================================
    # NORMAL PROGRESSION
    # ==================================================================

    if (
        new_sequence
        == expected
    ):

        return SequenceResult()

    # ==================================================================
    # MODULAR DISTANCE FROM EXPECTED
    # ==================================================================

    delta = (
        new_sequence
        - expected
    ) & UINT32_MASK

    # --------------------------------------------------------------
    # FORWARD JUMP
    # --------------------------------------------------------------
    #
    # A modular displacement smaller than half the uint32 space is
    # interpreted as forward movement with packet loss.
    # --------------------------------------------------------------

    if (
        delta
        < UINT32_HALF
    ):

        return SequenceResult(
            gap=int(
                delta
            )
        )

    # --------------------------------------------------------------
    # OLD / DUPLICATE
    # --------------------------------------------------------------

    return SequenceResult(
        duplicate_or_old=True
    )


# ======================================================================
# NODE STATE
# ======================================================================


@dataclass(
    slots=True,
)
class NodeState:
    """
    Runtime state and diagnostics for one ESP32 acoustic node.

    Timing model
    ------------
    sequence
        TCP/network packet diagnostic counter.

    sampleIndex
        Authoritative shared-clock acoustic timeline.

    localMicros
        ESP32-local diagnostic timestamp.

    TCP arrival time
        Transport timing only.

    localMicros and TCP arrival timing must not be used for cross-node
    TDOA synchronization.
    """

    # ------------------------------------------------------------------
    # IDENTITY / CONFIGURATION
    # ------------------------------------------------------------------

    node_id: int

    audio_config: AudioConfig

    # ------------------------------------------------------------------
    # CONNECTION
    # ------------------------------------------------------------------

    connected: bool = (
        False
    )

    peer: str | None = (
        None
    )

    hello: HelloPayload | None = (
        None
    )

    # ------------------------------------------------------------------
    # CURRENT ESP32 SESSION
    # ------------------------------------------------------------------

    session_id: int | None = (
        None
    )

    # ------------------------------------------------------------------
    # NETWORK SEQUENCE TRACKING
    # ------------------------------------------------------------------

    last_sequence: int | None = (
        None
    )

    # ------------------------------------------------------------------
    # SAMPLE TIMELINE
    # ------------------------------------------------------------------

    last_sample_index: int | None = (
        None
    )

    expected_next_audio_sample: (
        int
        | None
    ) = None

    # ------------------------------------------------------------------
    # PACKET COUNTERS
    # ------------------------------------------------------------------

    packets_received: int = (
        0
    )

    audio_packets_received: int = (
        0
    )

    crc_errors: int = (
        0
    )

    protocol_errors: int = (
        0
    )

    sequence_gaps: int = (
        0
    )

    sequence_resets: int = (
        0
    )

    sample_gaps: int = (
        0
    )

    duplicate_or_old_packets: int = (
        0
    )

    # ------------------------------------------------------------------
    # HEALTH FLAGS
    # ------------------------------------------------------------------

    clipped_packets: int = (
        0
    )

    clock_fault_packets: int = (
        0
    )

    congested_packets: int = (
        0
    )

    # ------------------------------------------------------------------
    # LATEST TELEMETRY
    # ------------------------------------------------------------------

    latest_environment: (
        EnvironmentPayload
        | None
    ) = None

    latest_heartbeat: (
        HeartbeatPayload
        | None
    ) = None

    latest_sync: (
        SyncPayload
        | None
    ) = None

    # ------------------------------------------------------------------
    # HISTORY BUFFERS
    # ------------------------------------------------------------------

    audio_blocks: Deque[
        AudioBlock
    ] = field(
        init=False
    )

    environment_history: Deque[
        EnvironmentSample
    ] = field(
        init=False
    )

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate node identity and initialize bounded runtime buffers.
        """

        # ==============================================================
        # NODE ID
        # ==============================================================

        self.node_id = (
            _validate_uint8(
                self.node_id,
                name=
                    "node_id",
            )
        )

        if (
            self.node_id
            == 0
        ):

            raise ValueError(
                (
                    "node_id must lie between "
                    "1 and 255."
                )
            )

        # ==============================================================
        # AUDIO CONFIGURATION
        # ==============================================================

        if not isinstance(
            self.audio_config,
            AudioConfig,
        ):

            raise TypeError(
                (
                    "audio_config must be "
                    "an AudioConfig instance."
                )
            )

        # ==============================================================
        # AUDIO BUFFER
        # ==============================================================

        self.audio_blocks = deque(
            maxlen=
                self.audio_config
                .blocks_in_buffer
        )

        # ==============================================================
        # ENVIRONMENT BUFFER
        # ==============================================================
        #
        # Environment telemetry is low-rate.
        #
        # 600 retained samples therefore cover a substantial laboratory
        # recording interval without meaningful memory pressure.
        # ==============================================================

        self.environment_history = deque(
            maxlen=600
        )

    # ==================================================================
    # SESSION RESET
    # ==================================================================

    def reset_stream_tracking(
        self,
        *,
        session_id: int | None = None,
        clear_environment: bool = True,
    ) -> None:
        """
        Reset all sample-index-dependent state.

        sampleIndex restarts at zero for every acquisition session.

        Therefore PCM and normally environmental history from the
        previous session must not remain in the new timeline.

        HELLO and HEARTBEAT are retained because they describe device
        and connection health rather than sample-indexed acoustic data.
        """

        # ==============================================================
        # SESSION ID
        # ==============================================================

        if (
            session_id
            is not None
        ):

            session_id = (
                _validate_uint32(
                    session_id,
                    name=
                        "session_id",
                )
            )

        self.session_id = (
            session_id
        )

        # ==============================================================
        # NETWORK TIMELINE
        # ==============================================================

        self.last_sequence = (
            None
        )

        # ==============================================================
        # AUDIO TIMELINE
        # ==============================================================

        self.last_sample_index = (
            None
        )

        self.expected_next_audio_sample = (
            None
        )

        self.audio_blocks.clear()

        # ==============================================================
        # SYNC MARKER
        # ==============================================================

        self.latest_sync = (
            None
        )

        # ==============================================================
        # ENVIRONMENT
        # ==============================================================

        if clear_environment:

            self.latest_environment = (
                None
            )

            self.environment_history.clear()

    # ==================================================================
    # PACKET HEADER OBSERVATION
    # ==================================================================

    def observe_header(
        self,
        sequence: int,
        session_id: int,
        sample_index: int,
        flags: int = 0,
    ) -> None:
        """
        Update diagnostics from one validated Protocol-v4 packet header.

        Session handling
        ----------------
        session_id == 0
            idle / diagnostic context.

        session_id != 0
            acquisition session.

        An idle packet is not allowed to destroy an established non-zero
        acquisition timeline.

        Duplicate/old sequence handling
        -------------------------------
        A packet classified as duplicate or old is counted, but it does
        NOT replace last_sequence or last_sample_index.

        This prevents the diagnostic tracker from moving backwards and
        subsequently reporting a false packet gap.
        """

        # ==============================================================
        # FIELD VALIDATION
        # ==============================================================

        sequence = (
            _validate_uint32(
                sequence,
                name=
                    "sequence",
            )
        )

        session_id = (
            _validate_uint32(
                session_id,
                name=
                    "session_id",
            )
        )

        sample_index = (
            _validate_uint64(
                sample_index,
                name=
                    "sample_index",
            )
        )

        flags = (
            _validate_uint8(
                flags,
                name=
                    "flags",
            )
        )

        # ==============================================================
        # PACKET COUNTER
        # ==============================================================

        self.packets_received += (
            1
        )

        # ==============================================================
        # SESSION TRACKING
        # ==============================================================

        if (
            self.session_id
            is None
        ):

            self.session_id = (
                session_id
            )

        elif (
            session_id
            != 0
            and session_id
            != self.session_id
        ):

            self.sequence_resets += (
                1
            )

            self.reset_stream_tracking(
                session_id=
                    session_id,

                clear_environment=
                    True,
            )

        # ==============================================================
        # NETWORK SEQUENCE
        # ==============================================================

        result = (
            classify_sequence(
                self.last_sequence,
                sequence,
            )
        )

        self.sequence_gaps += (
            result.gap
        )

        if (
            result.duplicate_or_old
        ):

            self.duplicate_or_old_packets += (
                1
            )

        # ==============================================================
        # HEALTH FLAGS
        # ==============================================================

        packet_flags = (
            PacketFlags(
                flags
            )
        )

        if (
            packet_flags
            & PacketFlags.CLIPPED
        ):

            self.clipped_packets += (
                1
            )

        if (
            packet_flags
            & PacketFlags.CLOCK_FAULT
        ):

            self.clock_fault_packets += (
                1
            )

        if (
            packet_flags
            & PacketFlags.QUEUE_CONGESTED
        ):

            self.congested_packets += (
                1
            )

        # ==============================================================
        # LATEST ACCEPTED HEADER
        # ==============================================================
        #
        # Critical:
        #
        # Do not regress these values when an older/duplicate packet is
        # observed.
        # ==============================================================

        if not (
            result.duplicate_or_old
        ):

            self.last_sequence = (
                sequence
            )

            self.last_sample_index = (
                sample_index
            )

    # ==================================================================
    # AUDIO INGESTION
    # ==================================================================

    def add_audio(
        self,
        block: AudioBlock,
    ) -> None:
        """
        Append one validated PCM block to this node's bounded timeline.

        Timeline behavior
        -----------------
        contiguous block
            appended normally

        forward sample jump
            missing samples are counted

        duplicate / overlapping / old block
            rejected

        Missing samples are deliberately not synthesized here.

        StreamManager reconstructs explicit silence only when a requested
        absolute sample window is assembled.
        """

        # ==============================================================
        # TYPE
        # ==============================================================

        if not isinstance(
            block,
            AudioBlock,
        ):

            raise TypeError(
                (
                    "block must be "
                    "an AudioBlock instance."
                )
            )

        # ==============================================================
        # NODE ID
        # ==============================================================

        if (
            block.node_id
            != self.node_id
        ):

            raise ValueError(
                (
                    "AudioBlock node_id does "
                    "not match NodeState node_id "
                    f"({block.node_id} != "
                    f"{self.node_id})."
                )
            )

        # ==============================================================
        # SESSION CONSISTENCY
        # ==============================================================
        #
        # ReceiverServer normally calls observe_header() immediately
        # before add_audio().
        #
        # The guard remains here so StreamManager/tests cannot inject
        # another session into an established node timeline.
        # ==============================================================

        if (
            self.session_id
            not in {
                None,
                0,
            }
            and block.session_id
            != self.session_id
        ):

            raise ValueError(
                (
                    "AudioBlock session_id does "
                    "not match NodeState session_id "
                    f"({block.session_id} != "
                    f"{self.session_id})."
                )
            )

        # --------------------------------------------------------------
        # Standalone/test usage may add PCM before observe_header().
        # Establish the timeline safely in that case.
        # --------------------------------------------------------------

        if (
            self.session_id
            in {
                None,
                0,
            }
            and block.session_id
            != 0
        ):

            self.reset_stream_tracking(
                session_id=
                    block.session_id,

                clear_environment=
                    True,
            )

        # ==============================================================
        # SAMPLE BOUNDARIES
        # ==============================================================

        block_start = (
            _validate_uint64(
                block.sample_index,
                name=
                    "AudioBlock sample_index",
            )
        )

        block_end = int(
            block.end_sample
        )

        if (
            block_end
            <= block_start
        ):

            return

        # ==============================================================
        # AUDIO PACKET COUNTER
        # ==============================================================

        self.audio_packets_received += (
            1
        )

        expected = (
            self.expected_next_audio_sample
        )

        # ==============================================================
        # EXISTING TIMELINE
        # ==============================================================

        if (
            expected
            is not None
        ):

            # ----------------------------------------------------------
            # FORWARD GAP
            # ----------------------------------------------------------

            if (
                block_start
                > expected
            ):

                self.sample_gaps += (
                    block_start
                    - expected
                )

            # ----------------------------------------------------------
            # OLD / DUPLICATE / OVERLAPPING PCM
            # ----------------------------------------------------------

            elif (
                block_start
                < expected
            ):

                self.duplicate_or_old_packets += (
                    1
                )

                return

        # ==============================================================
        # STORE
        # ==============================================================

        self.audio_blocks.append(
            block
        )

        self.expected_next_audio_sample = (
            block_end
        )

    # ==================================================================
    # ENVIRONMENT INGESTION
    # ==================================================================

    def add_environment(
        self,
        sample: EnvironmentSample,
    ) -> None:
        """
        Store environmental telemetry associated with the node's current
        sample timeline.

        Stale telemetry from another established acquisition session is
        ignored.
        """

        # ==============================================================
        # TYPE
        # ==============================================================

        if not isinstance(
            sample,
            EnvironmentSample,
        ):

            raise TypeError(
                (
                    "sample must be an "
                    "EnvironmentSample instance."
                )
            )

        # ==============================================================
        # NODE
        # ==============================================================

        if (
            sample.node_id
            != self.node_id
        ):

            raise ValueError(
                (
                    "EnvironmentSample node_id "
                    "does not match NodeState."
                )
            )

        # ==============================================================
        # SESSION
        # ==============================================================

        if (
            self.session_id
            not in {
                None,
                0,
            }
            and sample.session_id
            != self.session_id
        ):

            return

        # --------------------------------------------------------------
        # Standalone/test usage may provide environment telemetry before
        # observe_header().
        # --------------------------------------------------------------

        if (
            self.session_id
            in {
                None,
                0,
            }
            and sample.session_id
            != 0
        ):

            self.session_id = (
                sample.session_id
            )

        # ==============================================================
        # STORE
        # ==============================================================

        self.latest_environment = (
            sample.value
        )

        self.environment_history.append(
            sample
        )

    # ==================================================================
    # ENVIRONMENT LOOKUP
    # ==================================================================

    def environment_near(
        self,
        sample_index: int,
    ) -> EnvironmentPayload | None:
        """
        Return environmental telemetry nearest to a sample position.

        When a non-zero acquisition session is established, only
        telemetry belonging to that exact session is eligible.
        """

        try:

            sample_index = (
                _validate_uint64(
                    sample_index,
                    name=
                        "sample_index",
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        current_session = (
            self.session_id
        )

        # ==============================================================
        # NO HISTORY
        # ==============================================================

        if not (
            self.environment_history
        ):

            # During a known acquisition session we cannot prove that an
            # unindexed latest_environment belongs to that session unless
            # its corresponding history entry still exists.
            if (
                current_session
                not in {
                    None,
                    0,
                }
            ):

                return None

            return (
                self.latest_environment
            )

        # ==============================================================
        # SESSION FILTER
        # ==============================================================

        if (
            current_session
            not in {
                None,
                0,
            }
        ):

            candidates = tuple(
                item
                for item
                in self.environment_history
                if (
                    item.session_id
                    == current_session
                )
            )

        else:

            candidates = tuple(
                self.environment_history
            )

        if not candidates:

            return None

        # ==============================================================
        # NEAREST SAMPLE INDEX
        # ==============================================================

        best = min(
            candidates,
            key=lambda item:
                abs(
                    int(
                        item.sample_index
                    )
                    - sample_index
                ),
        )

        return (
            best.value
        )

    # ==================================================================
    # SNAPSHOT
    # ==================================================================

    def snapshot(
        self,
    ) -> NodeSnapshot:
        """
        Return an immutable diagnostic snapshot for CLI/dashboard use.
        """

        return NodeSnapshot(
            node_id=
                self.node_id,

            connected=
                self.connected,

            peer=
                self.peer,

            hello=
                self.hello,

            session_id=
                self.session_id,

            last_sequence=
                self.last_sequence,

            last_sample_index=
                self.last_sample_index,

            packets_received=
                self.packets_received,

            audio_packets_received=
                self.audio_packets_received,

            crc_errors=
                self.crc_errors,

            protocol_errors=
                self.protocol_errors,

            sequence_gaps=
                self.sequence_gaps,

            sequence_resets=
                self.sequence_resets,

            sample_gaps=
                self.sample_gaps,

            duplicate_or_old_packets=
                self.duplicate_or_old_packets,

            clipped_packets=
                self.clipped_packets,

            clock_fault_packets=
                self.clock_fault_packets,

            congested_packets=
                self.congested_packets,

            latest_environment=
                self.latest_environment,

            latest_heartbeat=
                self.latest_heartbeat,

            latest_sync=
                self.latest_sync,
        )


# ======================================================================
# NODE CONNECTION
# ======================================================================


class NodeConnection:
    """
    One active TCP connection to an ESP32 acoustic node.

    ReceiverServer owns packet reads.

    NodeConnection owns serialized laptop -> ESP32 control-frame writes.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        state: NodeState,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:

        if not isinstance(
            state,
            NodeState,
        ):

            raise TypeError(
                (
                    "state must be "
                    "a NodeState instance."
                )
            )

        self.state = (
            state
        )

        self.reader = (
            reader
        )

        self.writer = (
            writer
        )

        self._write_lock = (
            asyncio.Lock()
        )

    # ==================================================================
    # NODE ID
    # ==================================================================

    @property
    def node_id(
        self,
    ) -> int:
        """
        ESP32 node ID associated with this TCP connection.
        """

        return int(
            self.state.node_id
        )

    # ==================================================================
    # CONTROL COMMAND
    # ==================================================================

    async def send_command(
        self,
        command: ControlCommand,
        *,
        session_id: int = 0,
    ) -> None:
        """
        Send one fixed 8-byte laptop -> ESP32 control frame.
        """

        session_id = (
            _validate_uint32(
                session_id,
                name=
                    "session_id",
            )
        )

        if (
            self.writer.is_closing()
        ):

            raise ConnectionError(
                (
                    f"node {self.node_id} "
                    "connection is closing"
                )
            )

        frame = (
            pack_control(
                ControlFrame(
                    command=
                        command,

                    session_id=
                        session_id,
                )
            )
        )

        async with (
            self._write_lock
        ):

            if (
                self.writer.is_closing()
            ):

                raise ConnectionError(
                    (
                        f"node {self.node_id} "
                        "connection closed "
                        "before write"
                    )
                )

            self.writer.write(
                frame
            )

            await self.writer.drain()

    # ==================================================================
    # CLOSE
    # ==================================================================

    async def close(
        self,
    ) -> None:
        """
        Gracefully close the node TCP connection.

        If another caller has already initiated StreamWriter.close(), we
        still wait for the underlying socket shutdown to complete.

        Cancellation remains observable by the caller.
        """

        if not (
            self.writer.is_closing()
        ):

            self.writer.close()

        try:

            await asyncio.wait_for(
                self.writer.wait_closed(),
                timeout=2.0,
            )

        except asyncio.CancelledError:

            raise

        except (
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        ):

            # Windows transports and disconnected ESP32 sockets may not
            # complete wait_closed() promptly. The close request has already
            # been issued, so shutdown must not block indefinitely here.
            pass
