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
# CONSTANTS
# ======================================================================


UINT32_MASK = (
    0xFFFFFFFF
)

UINT32_HALF = (
    0x80000000
)

UINT32_MAX = (
    0xFFFFFFFF
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
        True when the new sequence appears to be an older or duplicate
        value rather than a forward jump.

    Notes
    -----
    Sequence numbers are treated as unsigned 32-bit counters and normal
    wraparound is handled correctly.

    Sequence tracking is for network diagnostics only.

    Cross-node acoustic synchronization uses sampleIndex instead.
    """

    gap: int = 0

    duplicate_or_old: bool = (
        False
    )


# ======================================================================
# UINT32 VALIDATION
# ======================================================================


def _validate_uint32(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate and normalize one unsigned 32-bit integer.
    """

    value = int(
        value
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
        gap  = 0

    Gap:

        last = 10
        new  = 14
        gap  = 3

    Wraparound:

        last = 0xFFFFFFFF
        new  = 0

        normal, no gap

    Older/duplicate:

        last = 100
        new  = 98

        duplicate_or_old = True

    This counter is deliberately separate from audio sampleIndex.
    """

    new_sequence = (
        _validate_uint32(
            new_sequence,
            name="new_sequence",
        )
    )

    if (
        last_sequence
        is None
    ):

        return SequenceResult()

    last_sequence = (
        _validate_uint32(
            last_sequence,
            name="last_sequence",
        )
    )

    expected = (
        last_sequence
        + 1
    ) & UINT32_MASK

    # --------------------------------------------------------------
    # EXACT NEXT PACKET
    # --------------------------------------------------------------

    if (
        new_sequence
        == expected
    ):

        return SequenceResult()

    # --------------------------------------------------------------
    # MODULAR DIFFERENCE FROM EXPECTED
    # --------------------------------------------------------------

    delta = (
        new_sequence
        - expected
    ) & UINT32_MASK

    # A modular forward displacement less than half the uint32 space is
    # interpreted as packet loss.
    if (
        delta
        < UINT32_HALF
    ):

        return SequenceResult(
            gap=int(
                delta
            )
        )

    # Otherwise the packet is most plausibly duplicate or old.
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

    Important synchronization model
    -------------------------------
    sequence
        Network packet-loss diagnostic.

    sampleIndex
        Authoritative shared-clock audio timeline.

    localMicros
        Stored inside AudioBlock for diagnostics only.

    TCP arrival time
        Never used for acoustic synchronization.
    """

    # ------------------------------------------------------------------
    # IDENTITY / CONFIG
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
        Create bounded runtime buffers.
        """

        self.node_id = int(
            self.node_id
        )

        if (
            self.node_id
            <= 0
        ):

            raise ValueError(
                "node_id must be greater than 0"
            )

        self.audio_blocks = deque(
            maxlen=
                self.audio_config.blocks_in_buffer
        )

        # Environment packets arrive slowly, therefore 600 samples can
        # cover a large amount of laboratory recording time.
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
        Reset sample-index-dependent state for a new acquisition session.

        Why environment is normally cleared
        ------------------------------------
        sampleIndex restarts from zero on every START.

        Therefore telemetry from an older session cannot safely remain
        in the same sample-index lookup history.

        Heartbeat and HELLO data are retained because they represent
        connection/device diagnostics rather than sample-indexed event
        data.
        """

        if (
            session_id
            is not None
        ):

            session_id = (
                _validate_uint32(
                    session_id,
                    name="session_id",
                )
            )

        self.session_id = (
            session_id
        )

        self.last_sequence = (
            None
        )

        self.last_sample_index = (
            None
        )

        self.expected_next_audio_sample = (
            None
        )

        self.audio_blocks.clear()

        # SYNC is tied to a specific acquisition session.
        self.latest_sync = (
            None
        )

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
        Update node diagnostics from one validated protocol header.

        Session handling
        ----------------
        A non-zero session ID represents an acquisition session.

        Session ID zero is treated as an idle/control context and is not
        allowed to destroy an already established non-zero acquisition
        timeline.

        This makes the laptop robust if an implementation emits an idle
        HEARTBEAT or HELLO using session_id = 0.
        """

        sequence = (
            _validate_uint32(
                sequence,
                name="sequence",
            )
        )

        session_id = (
            _validate_uint32(
                session_id,
                name="session_id",
            )
        )

        sample_index = int(
            sample_index
        )

        if (
            sample_index
            < 0
        ):

            raise ValueError(
                (
                    "sample_index "
                    "cannot be negative"
                )
            )

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
            session_id != 0
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
        # NETWORK PACKET SEQUENCE
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
        # LATEST HEADER STATE
        # ==============================================================

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
        Add one audio packet to the bounded PCM buffer.

        Audio sampleIndex behavior
        --------------------------
        expected_next_audio_sample
            exact next shared-clock sample expected for this node.

        Forward jump
            counted as missing samples.

        Older/overlapping block
            rejected so old PCM cannot overwrite or reorder the
            timeline.

        Important
        ---------
        Missing samples are NOT inserted here.

        StreamManager.get_window() later reconstructs gaps using digital
        silence while preserving absolute sample positions.
        """

        if (
            block.node_id
            != self.node_id
        ):

            raise ValueError(
                (
                    "AudioBlock node_id does not "
                    f"match NodeState node_id "
                    f"({block.node_id} != {self.node_id})"
                )
            )

        block_start = int(
            block.sample_index
        )

        block_end = int(
            block.end_sample
        )

        if (
            block_start
            < 0
        ):

            raise ValueError(
                (
                    "AudioBlock sample_index "
                    "cannot be negative"
                )
            )

        if (
            block_end
            <= block_start
        ):

            # Empty or malformed audio blocks are not useful.
            return

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
            # GAP
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
            # DUPLICATE / OVERLAP / OLD BLOCK
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
        # STORE BLOCK
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
        Store one environmental sample.

        Samples from a stale non-matching session are rejected when
        NodeState already represents a non-zero acquisition session.
        """

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
        Return the nearest valid environment sample.

        Only telemetry belonging to the current non-zero acquisition
        session is considered when such a session is known.
        """

        sample_index = int(
            sample_index
        )

        if (
            sample_index
            < 0
        ):

            return None

        if not (
            self.environment_history
        ):

            return (
                self.latest_environment
            )

        # ==============================================================
        # SESSION FILTER
        # ==============================================================

        current_session = (
            self.session_id
        )

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

        return best.value

    # ==================================================================
    # SNAPSHOT
    # ==================================================================

    def snapshot(
        self,
    ) -> NodeSnapshot:
        """
        Return an immutable-style diagnostics snapshot for CLI/UI use.
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
    One active TCP connection to an ESP32 node.

    Reads are managed by ReceiverServer.

    This object owns synchronized writes of laptop → ESP32 control
    frames.
    """

    def __init__(
        self,
        state: NodeState,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:

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
        Send one binary laptop → ESP32 control frame.
        """

        session_id = (
            _validate_uint32(
                session_id,
                name="session_id",
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
                        "connection closed before write"
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

        Cancellation is propagated rather than swallowed.
        """

        if (
            self.writer.is_closing()
        ):

            return

        self.writer.close()

        try:

            await self.writer.wait_closed()

        except asyncio.CancelledError:

            raise

        except (
            ConnectionError,
            OSError,
        ):

            # Socket may already have been reset by the ESP32.
            pass