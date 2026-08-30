"""
Tests for node runtime state and packet-sequence tracking.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
This module tests:

    uint32 packet-sequence interpretation
    sequence wraparound
    packet-loss diagnostics
    duplicate/old packet handling
    session transitions
    protocol health flags
    audio sampleIndex continuity
    stale/overlapping PCM rejection
    environmental telemetry lookup
    node diagnostics snapshots
    laptop -> ESP32 control-frame transmission

Important
---------
Packet sequence numbers are network diagnostics.

Audio sampleIndex is the authoritative shared-clock timeline used by
the acoustic pipeline.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import asyncio


# ======================================================================
# THIRD-PARTY
# ======================================================================


import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from models import (
    EnvironmentSample,
)

from node import (
    NodeConnection,
    classify_sequence,
)

from protocol import (
    ControlCommand,
    EnvironmentPayload,
    PacketFlags,
    SyncPayload,
    unpack_control,
)


# ======================================================================
# SEQUENCE CLASSIFICATION
# ======================================================================


def test_sequence_first_packet() -> None:
    """
    No previous packet means no gap can yet be inferred.
    """

    result = classify_sequence(
        None,
        100,
    )

    assert (
        result.gap
        == 0
    )

    assert not (
        result.duplicate_or_old
    )


def test_sequence_normal() -> None:

    result = classify_sequence(
        10,
        11,
    )

    assert (
        result.gap
        == 0
    )

    assert not (
        result.duplicate_or_old
    )


def test_sequence_gap() -> None:
    """
    10 -> 14 means packets 11, 12 and 13 were skipped.
    """

    result = classify_sequence(
        10,
        14,
    )

    assert (
        result.gap
        == 3
    )

    assert not (
        result.duplicate_or_old
    )


def test_sequence_wrap() -> None:

    result = classify_sequence(
        0xFFFFFFFF,
        0,
    )

    assert (
        result.gap
        == 0
    )

    assert not (
        result.duplicate_or_old
    )


def test_sequence_gap_across_wraparound() -> None:
    """
    Expected:

        last     = 0xFFFFFFFE
        expected = 0xFFFFFFFF

    Received:

        new      = 1

    Missing:

        0xFFFFFFFF
        0

    Therefore gap = 2.
    """

    result = classify_sequence(
        0xFFFFFFFE,
        1,
    )

    assert (
        result.gap
        == 2
    )

    assert not (
        result.duplicate_or_old
    )


def test_sequence_duplicate() -> None:

    result = classify_sequence(
        100,
        100,
    )

    assert (
        result.gap
        == 0
    )

    assert (
        result.duplicate_or_old
    )


def test_sequence_old_packet() -> None:

    result = classify_sequence(
        100,
        98,
    )

    assert (
        result.gap
        == 0
    )

    assert (
        result.duplicate_or_old
    )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        0x1_0000_0000,
    ],
)
def test_sequence_rejects_out_of_range_new_sequence(
    value: int,
) -> None:

    with pytest.raises(
        ValueError
    ):

        classify_sequence(
            10,
            value,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        0x1_0000_0000,
    ],
)
def test_sequence_rejects_out_of_range_previous_sequence(
    value: int,
) -> None:

    with pytest.raises(
        ValueError
    ):

        classify_sequence(
            value,
            10,
        )


# ======================================================================
# NODE INITIAL STATE
# ======================================================================


def test_node_state_initializes_empty_buffers(
    make_node_state,
) -> None:

    state = make_node_state(
        node_id=
            1
    )

    assert (
        state.node_id
        == 1
    )

    assert (
        state.session_id
        is None
    )

    assert (
        state.last_sequence
        is None
    )

    assert (
        state.last_sample_index
        is None
    )

    assert (
        state.expected_next_audio_sample
        is None
    )

    assert (
        len(
            state.audio_blocks
        )
        == 0
    )

    assert (
        len(
            state.environment_history
        )
        == 0
    )


# ======================================================================
# HEADER OBSERVATION
# ======================================================================


def test_observe_first_header_establishes_session(
    make_node_state,
    session_id,
) -> None:

    state = make_node_state(
        node_id=
            1
    )

    state.observe_header(
        sequence=
            10,

        session_id=
            session_id,

        sample_index=
            4096,
    )

    assert (
        state.session_id
        == session_id
    )

    assert (
        state.last_sequence
        == 10
    )

    assert (
        state.last_sample_index
        == 4096
    )

    assert (
        state.packets_received
        == 1
    )

    assert (
        state.sequence_gaps
        == 0
    )


def test_observe_header_accumulates_sequence_gap(
    make_node_state,
    session_id,
) -> None:

    state = make_node_state()

    state.observe_header(
        sequence=
            10,

        session_id=
            session_id,

        sample_index=
            0,
    )

    state.observe_header(
        sequence=
            14,

        session_id=
            session_id,

        sample_index=
            1024,
    )

    assert (
        state.sequence_gaps
        == 3
    )

    assert (
        state.last_sequence
        == 14
    )

    assert (
        state.packets_received
        == 2
    )


def test_old_header_does_not_regress_sequence_tracker(
    make_node_state,
    session_id,
) -> None:
    """
    Critical regression test.

    Once sequence 11 has been accepted, receiving an old sequence 10
    must increment diagnostics but must NOT make 10 the new tracking
    baseline.

    Otherwise the following legitimate packet 12 would be incorrectly
    reported as containing a sequence gap.
    """

    state = make_node_state()

    state.observe_header(
        sequence=
            10,

        session_id=
            session_id,

        sample_index=
            0,
    )

    state.observe_header(
        sequence=
            11,

        session_id=
            session_id,

        sample_index=
            1024,
    )

    state.observe_header(
        sequence=
            10,

        session_id=
            session_id,

        sample_index=
            0,
    )

    assert (
        state.duplicate_or_old_packets
        == 1
    )

    assert (
        state.last_sequence
        == 11
    )

    assert (
        state.last_sample_index
        == 1024
    )

    # --------------------------------------------------------------
    # A legitimate next packet must remain normal.
    # --------------------------------------------------------------

    state.observe_header(
        sequence=
            12,

        session_id=
            session_id,

        sample_index=
            2048,
    )

    assert (
        state.sequence_gaps
        == 0
    )

    assert (
        state.last_sequence
        == 12
    )


# ======================================================================
# SESSION TRANSITION
# ======================================================================


def test_new_session_resets_sample_timeline(
    make_node_state,
    make_audio_block,
    session_id,
    second_session_id,
) -> None:

    state = make_node_state()

    state.observe_header(
        sequence=
            50,

        session_id=
            session_id,

        sample_index=
            5000,
    )

    state.add_audio(
        make_audio_block(
            node_id=
                1,

            sequence=
                50,

            session_id=
                session_id,

            sample_index=
                0,
        )
    )

    state.latest_sync = SyncPayload(
        session_id=
            session_id,

        sync_id=
            1,

        sample_index=
            0,

        local_micros=
            100,
    )

    assert (
        len(
            state.audio_blocks
        )
        == 1
    )

    # --------------------------------------------------------------
    # New non-zero acquisition session.
    # --------------------------------------------------------------

    state.observe_header(
        sequence=
            0,

        session_id=
            second_session_id,

        sample_index=
            0,
    )

    assert (
        state.session_id
        == second_session_id
    )

    assert (
        state.sequence_resets
        == 1
    )

    assert (
        len(
            state.audio_blocks
        )
        == 0
    )

    assert (
        state.expected_next_audio_sample
        is None
    )

    assert (
        state.latest_sync
        is None
    )

    assert (
        state.last_sequence
        == 0
    )

    assert (
        state.last_sample_index
        == 0
    )


def test_idle_session_zero_does_not_replace_active_session(
    make_node_state,
    session_id,
) -> None:

    state = make_node_state()

    state.observe_header(
        sequence=
            20,

        session_id=
            session_id,

        sample_index=
            1024,
    )

    state.observe_header(
        sequence=
            21,

        session_id=
            0,

        sample_index=
            1024,
    )

    assert (
        state.session_id
        == session_id
    )

    assert (
        state.sequence_resets
        == 0
    )


# ======================================================================
# PACKET HEALTH FLAGS
# ======================================================================


def test_observe_header_counts_health_flags(
    make_node_state,
    session_id,
) -> None:

    state = make_node_state()

    combined_flags = int(
        PacketFlags.CLIPPED
        | PacketFlags.CLOCK_FAULT
        | PacketFlags.QUEUE_CONGESTED
    )

    state.observe_header(
        sequence=
            1,

        session_id=
            session_id,

        sample_index=
            0,

        flags=
            combined_flags,
    )

    assert (
        state.clipped_packets
        == 1
    )

    assert (
        state.clock_fault_packets
        == 1
    )

    assert (
        state.congested_packets
        == 1
    )


def test_health_flag_counters_accumulate_independently(
    make_node_state,
    session_id,
) -> None:

    state = make_node_state()

    state.observe_header(
        sequence=
            1,

        session_id=
            session_id,

        sample_index=
            0,

        flags=
            int(
                PacketFlags.CLIPPED
            ),
    )

    state.observe_header(
        sequence=
            2,

        session_id=
            session_id,

        sample_index=
            1024,

        flags=
            int(
                PacketFlags.CLOCK_FAULT
            ),
    )

    assert (
        state.clipped_packets
        == 1
    )

    assert (
        state.clock_fault_packets
        == 1
    )

    assert (
        state.congested_packets
        == 0
    )


# ======================================================================
# AUDIO SAMPLE TIMELINE
# ======================================================================


def test_add_audio_accepts_contiguous_blocks(
    make_node_state,
    make_audio_block,
    session_id,
) -> None:

    state = make_node_state()

    state.reset_stream_tracking(
        session_id=
            session_id
    )

    first = make_audio_block(
        node_id=
            1,

        sequence=
            0,

        session_id=
            session_id,

        sample_index=
            0,
    )

    second = make_audio_block(
        node_id=
            1,

        sequence=
            1,

        session_id=
            session_id,

        sample_index=
            first.end_sample,
    )

    state.add_audio(
        first
    )

    state.add_audio(
        second
    )

    assert (
        len(
            state.audio_blocks
        )
        == 2
    )

    assert (
        state.sample_gaps
        == 0
    )

    assert (
        state.expected_next_audio_sample
        == second.end_sample
    )

    assert (
        state.audio_packets_received
        == 2
    )


def test_add_audio_counts_missing_samples(
    make_node_state,
    make_audio_block,
    session_id,
) -> None:

    state = make_node_state()

    state.reset_stream_tracking(
        session_id=
            session_id
    )

    first = make_audio_block(
        node_id=
            1,

        sequence=
            0,

        session_id=
            session_id,

        sample_index=
            0,
    )

    state.add_audio(
        first
    )

    missing_samples = (
        256
    )

    second = make_audio_block(
        node_id=
            1,

        sequence=
            1,

        session_id=
            session_id,

        sample_index=
            first.end_sample
            + missing_samples,
    )

    state.add_audio(
        second
    )

    assert (
        state.sample_gaps
        == missing_samples
    )

    assert (
        len(
            state.audio_blocks
        )
        == 2
    )


def test_add_audio_rejects_old_or_overlapping_block(
    make_node_state,
    make_audio_block,
    session_id,
) -> None:

    state = make_node_state()

    state.reset_stream_tracking(
        session_id=
            session_id
    )

    first = make_audio_block(
        node_id=
            1,

        sequence=
            0,

        session_id=
            session_id,

        sample_index=
            0,
    )

    state.add_audio(
        first
    )

    expected_after_first = (
        state.expected_next_audio_sample
    )

    overlap = make_audio_block(
        node_id=
            1,

        sequence=
            1,

        session_id=
            session_id,

        sample_index=
            512,
    )

    state.add_audio(
        overlap
    )

    assert (
        len(
            state.audio_blocks
        )
        == 1
    )

    assert (
        state.duplicate_or_old_packets
        == 1
    )

    assert (
        state.expected_next_audio_sample
        == expected_after_first
    )


def test_add_audio_rejects_wrong_node(
    make_node_state,
    make_audio_block,
    session_id,
) -> None:

    state = make_node_state(
        node_id=
            1
    )

    block = make_audio_block(
        node_id=
            2,

        session_id=
            session_id,
    )

    with pytest.raises(
        ValueError
    ):

        state.add_audio(
            block
        )


# ======================================================================
# ENVIRONMENT TELEMETRY
# ======================================================================


def test_environment_near_returns_nearest_sample(
    make_node_state,
    session_id,
) -> None:

    state = make_node_state()

    state.reset_stream_tracking(
        session_id=
            session_id
    )

    environment_1 = EnvironmentPayload(
        temperature_c=
            25.0,

        humidity_percent=
            50.0,

        pressure_hpa=
            1012.0,
    )

    environment_2 = EnvironmentPayload(
        temperature_c=
            26.0,

        humidity_percent=
            52.0,

        pressure_hpa=
            1011.5,
    )

    environment_3 = EnvironmentPayload(
        temperature_c=
            27.0,

        humidity_percent=
            54.0,

        pressure_hpa=
            1011.0,
    )

    state.add_environment(
        EnvironmentSample(
            node_id=
                1,

            session_id=
                session_id,

            sample_index=
                0,

            value=
                environment_1,
        )
    )

    state.add_environment(
        EnvironmentSample(
            node_id=
                1,

            session_id=
                session_id,

            sample_index=
                1000,

            value=
                environment_2,
        )
    )

    state.add_environment(
        EnvironmentSample(
            node_id=
                1,

            session_id=
                session_id,

            sample_index=
                2000,

            value=
                environment_3,
        )
    )

    assert (
        state.environment_near(
            1400
        )
        == environment_2
    )

    assert (
        state.latest_environment
        == environment_3
    )


def test_environment_from_stale_session_is_ignored(
    make_node_state,
    session_id,
    second_session_id,
) -> None:

    state = make_node_state()

    state.reset_stream_tracking(
        session_id=
            session_id
    )

    stale_environment = EnvironmentPayload(
        temperature_c=
            40.0,

        humidity_percent=
            10.0,

        pressure_hpa=
            900.0,
    )

    state.add_environment(
        EnvironmentSample(
            node_id=
                1,

            session_id=
                second_session_id,

            sample_index=
                1000,

            value=
                stale_environment,
        )
    )

    assert (
        state.latest_environment
        is None
    )

    assert (
        len(
            state.environment_history
        )
        == 0
    )


def test_environment_wrong_node_is_rejected(
    make_node_state,
    session_id,
) -> None:

    state = make_node_state(
        node_id=
            1
    )

    environment = EnvironmentPayload(
        temperature_c=
            25.0,

        humidity_percent=
            50.0,

        pressure_hpa=
            1013.25,
    )

    with pytest.raises(
        ValueError
    ):

        state.add_environment(
            EnvironmentSample(
                node_id=
                    2,

                session_id=
                    session_id,

                sample_index=
                    0,

                value=
                    environment,
            )
        )


def test_reset_stream_tracking_clears_environment_by_default(
    make_node_state,
    session_id,
    second_session_id,
) -> None:

    state = make_node_state()

    environment = EnvironmentPayload(
        temperature_c=
            24.0,

        humidity_percent=
            48.0,

        pressure_hpa=
            1010.0,
    )

    state.reset_stream_tracking(
        session_id=
            session_id
    )

    state.add_environment(
        EnvironmentSample(
            node_id=
                1,

            session_id=
                session_id,

            sample_index=
                0,

            value=
                environment,
        )
    )

    state.reset_stream_tracking(
        session_id=
            second_session_id
    )

    assert (
        state.latest_environment
        is None
    )

    assert (
        len(
            state.environment_history
        )
        == 0
    )


# ======================================================================
# SNAPSHOT
# ======================================================================


def test_snapshot_exposes_node_diagnostics(
    make_node_state,
    session_id,
) -> None:

    state = make_node_state(
        node_id=
            2
    )

    state.connected = (
        True
    )

    state.peer = (
        "192.168.1.42"
    )

    state.observe_header(
        sequence=
            5,

        session_id=
            session_id,

        sample_index=
            8192,

        flags=
            int(
                PacketFlags.CLIPPED
            ),
    )

    snapshot = (
        state.snapshot()
    )

    assert (
        snapshot.node_id
        == 2
    )

    assert (
        snapshot.connected
        is True
    )

    assert (
        snapshot.peer
        == "192.168.1.42"
    )

    assert (
        snapshot.session_id
        == session_id
    )

    assert (
        snapshot.last_sequence
        == 5
    )

    assert (
        snapshot.last_sample_index
        == 8192
    )

    assert (
        snapshot.packets_received
        == 1
    )

    assert (
        snapshot.clipped_packets
        == 1
    )


# ======================================================================
# NODE CONNECTION TEST DOUBLE
# ======================================================================


class _FakeWriter:
    """
    Minimal asyncio StreamWriter-compatible test double.
    """

    def __init__(
        self,
    ) -> None:

        self.data = bytearray()

        self.drain_count = (
            0
        )

        self.close_count = (
            0
        )

        self.wait_closed_count = (
            0
        )

        self._closing = (
            False
        )

    def is_closing(
        self,
    ) -> bool:

        return (
            self._closing
        )

    def write(
        self,
        data: bytes,
    ) -> None:

        self.data.extend(
            data
        )

    async def drain(
        self,
    ) -> None:

        self.drain_count += (
            1
        )

    def close(
        self,
    ) -> None:

        self.close_count += (
            1
        )

        self._closing = (
            True
        )

    async def wait_closed(
        self,
    ) -> None:

        self.wait_closed_count += (
            1
        )


# ======================================================================
# NODE CONNECTION CONTROL FRAMES
# ======================================================================


def test_node_connection_sends_control_frame(
    make_node_state,
    session_id,
) -> None:

    async def scenario() -> None:

        state = make_node_state(
            node_id=
                2
        )

        reader = (
            asyncio.StreamReader()
        )

        writer = (
            _FakeWriter()
        )

        connection = NodeConnection(
            state,
            reader,
            writer,
        )

        await connection.send_command(
            ControlCommand.START,
            session_id=
                session_id,
        )

        frame = unpack_control(
            bytes(
                writer.data
            )
        )

        assert (
            frame.command
            == ControlCommand.START
        )

        assert (
            frame.session_id
            == session_id
        )

        assert (
            writer.drain_count
            == 1
        )

        assert (
            connection.node_id
            == 2
        )

    asyncio.run(
        scenario()
    )


def test_node_connection_rejects_write_when_closing(
    make_node_state,
    session_id,
) -> None:

    async def scenario() -> None:

        state = make_node_state()

        writer = (
            _FakeWriter()
        )

        writer.close()

        connection = NodeConnection(
            state,
            asyncio.StreamReader(),
            writer,
        )

        with pytest.raises(
            ConnectionError
        ):

            await connection.send_command(
                ControlCommand.START,
                session_id=
                    session_id,
            )

    asyncio.run(
        scenario()
    )


def test_node_connection_close_waits_for_writer(
    make_node_state,
) -> None:

    async def scenario() -> None:

        state = make_node_state()

        writer = (
            _FakeWriter()
        )

        connection = NodeConnection(
            state,
            asyncio.StreamReader(),
            writer,
        )

        await connection.close()

        assert (
            writer.close_count
            == 1
        )

        assert (
            writer.wait_closed_count
            == 1
        )

        assert (
            writer.is_closing()
        )

    asyncio.run(
        scenario()
    )