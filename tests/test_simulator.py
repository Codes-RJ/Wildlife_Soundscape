"""
Tests for simulator.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. simulator environmental sound speed
    2. source-to-node geometry
    3. acoustic propagation delay
    4. distance-dependent amplitude gain
    5. shared simulated clock state
    6. node arming/disarming
    7. node streaming eligibility
    8. master/slave role assignment
    9. simulator node validation
    10. sequence-number wraparound
    11. local microsecond diagnostic range
    12. Protocol-v4 packet construction
    13. packet flags
    14. HELLO payload compatibility
    15. synthetic event waveform properties
    16. deterministic per-node audio generation
    17. independent noise between nodes
    18. event-vs-background energy
    19. PCM16 output properties
    20. clipping detection
    21. SYNC packet generation
    22. heartbeat packet generation

Scientific scope
----------------
The simulator provides controlled synthetic input for testing the
software pipeline.

It is not a physical propagation model and is not evidence of real
localization accuracy.

Real localization performance must later be measured using known-source
calibration experiments with the actual microphone hardware.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import asyncio
import math
import time


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np
import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from environment import (
    calculate_speed_of_sound_mps,
)

from protocol import (
    HEADER_SIZE,
    PacketFlags,
    PacketType,
    parse_heartbeat,
    parse_hello,
    parse_sync,
    unpack_header,
    verify_payload_crc,
)

from simulator import (
    EVENT_FIRST_SAMPLE,
    FRAMES_PER_BLOCK,
    SAMPLE_RATE,
    SIM_HUMIDITY_PERCENT,
    SIM_PRESSURE_HPA,
    SIM_TEMPERATURE_C,
    UINT32_MASK,
    FakeNode,
    SharedSimulation,
    calculate_simulated_speed_of_sound,
)


# ======================================================================
# CONSTANTS
# ======================================================================


TEST_SESSION_ID = (
    0x12345678
)


SECOND_SESSION_ID = (
    0x87654321
)


# ======================================================================
# TEST HELPERS
# ======================================================================


def rms(
    signal: np.ndarray,
) -> float:
    """
    Calculate RMS with float64 accumulation.
    """

    values = np.asarray(
        signal,
        dtype=np.float64,
    )

    if (
        values.size
        == 0
    ):

        return (
            0.0
        )

    return float(
        np.sqrt(
            np.mean(
                values
                * values,
                dtype=np.float64,
            )
        )
    )


def make_node(
    node_id: int = 1,
    *,
    shared: SharedSimulation | None = None,
) -> FakeNode:
    """
    Construct one fake ESP32 node without opening a network connection.
    """

    if (
        shared
        is None
    ):

        shared = (
            SharedSimulation()
        )

    return FakeNode(
        node_id=
            node_id,

        host=
            "127.0.0.1",

        port=
            5001,

        shared=
            shared,
    )


# ======================================================================
# FAKE ASYNC WRITER
# ======================================================================


class FakeWriter:
    """
    Minimal asyncio StreamWriter-compatible test double.

    Used to inspect packets produced by private simulator send helpers
    without opening TCP sockets.
    """

    def __init__(
        self,
    ) -> None:

        self.data = bytearray()

        self.drain_count = (
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

        self._closing = (
            True
        )


# ======================================================================
# SIMULATED SPEED OF SOUND
# ======================================================================


def test_simulated_speed_of_sound_is_finite_and_positive() -> None:

    result = (
        calculate_simulated_speed_of_sound()
    )

    assert math.isfinite(
        result
    )

    assert (
        result
        > 0.0
    )


def test_simulated_speed_of_sound_is_physically_reasonable() -> None:

    result = (
        calculate_simulated_speed_of_sound()
    )

    assert (
        330.0
        < result
        < 360.0
    )


def test_simulator_and_environment_module_use_consistent_sound_speed() -> None:
    """
    The simulator transmits the same environmental conditions that it
    uses to create the synthetic acoustic propagation delays.

    The laptop environmental model should therefore calculate the same
    sound speed.
    """

    simulator_speed = (
        calculate_simulated_speed_of_sound()
    )

    pipeline_speed = calculate_speed_of_sound_mps(
        SIM_TEMPERATURE_C,
        SIM_HUMIDITY_PERCENT,
        SIM_PRESSURE_HPA,
    )

    assert (
        simulator_speed
        == pytest.approx(
            pipeline_speed,
            rel=
                1e-12,
            abs=
                1e-12,
        )
    )


# ======================================================================
# SHARED GEOMETRY
# ======================================================================


def test_default_simulation_contains_three_project_nodes() -> None:

    shared = (
        SharedSimulation()
    )

    assert (
        set(
            shared.node_xy
        )
        == {
            1,
            2,
            3,
        }
    )


def test_source_position_is_known_and_finite() -> None:

    shared = (
        SharedSimulation()
    )

    assert (
        len(
            shared.source_xy
        )
        == 2
    )

    assert all(
        math.isfinite(
            coordinate
        )
        for coordinate
        in shared.source_xy
    )


@pytest.mark.parametrize(
    "node_id",
    [
        1,
        2,
        3,
    ],
)
def test_distance_to_source_matches_euclidean_geometry(
    node_id: int,
) -> None:

    shared = (
        SharedSimulation()
    )

    node_x, node_y = (
        shared.node_xy[
            node_id
        ]
    )

    source_x, source_y = (
        shared.source_xy
    )

    expected = math.hypot(
        source_x
        - node_x,

        source_y
        - node_y,
    )

    result = shared.distance_to_source(
        node_id
    )

    assert (
        result
        == pytest.approx(
            expected,
            rel=
                1e-12,
        )
    )


def test_distance_to_unknown_node_is_rejected() -> None:

    shared = (
        SharedSimulation()
    )

    with pytest.raises(
        KeyError
    ):

        shared.distance_to_source(
            99
        )


# ======================================================================
# PROPAGATION DELAY
# ======================================================================


@pytest.mark.parametrize(
    "node_id",
    [
        1,
        2,
        3,
    ],
)
def test_delay_samples_matches_distance_over_sound_speed(
    node_id: int,
) -> None:

    shared = (
        SharedSimulation()
    )

    distance_m = (
        shared.distance_to_source(
            node_id
        )
    )

    expected = int(
        round(
            (
                distance_m
                / shared.speed_of_sound
            )
            * SAMPLE_RATE
        )
    )

    assert (
        shared.delay_samples(
            node_id
        )
        == expected
    )


def test_pairwise_simulated_tdoa_matches_geometry() -> None:
    """
    Verify that the synthetic inter-node delay corresponds to the
    source/node geometry.

    Because individual propagation delays are rounded to integer samples,
    the pairwise difference is allowed one sample of rounding error.
    """

    shared = (
        SharedSimulation()
    )

    distance_1 = (
        shared.distance_to_source(
            1
        )
    )

    distance_2 = (
        shared.distance_to_source(
            2
        )
    )

    expected_difference_samples = (
        (
            distance_2
            - distance_1
        )
        / shared.speed_of_sound
        * SAMPLE_RATE
    )

    actual_difference_samples = (
        shared.delay_samples(
            2
        )
        - shared.delay_samples(
            1
        )
    )

    assert (
        actual_difference_samples
        == pytest.approx(
            expected_difference_samples,
            abs=
                1.0,
        )
    )


# ======================================================================
# DISTANCE ATTENUATION
# ======================================================================


@pytest.mark.parametrize(
    "node_id",
    [
        1,
        2,
        3,
    ],
)
def test_amplitude_gain_is_finite_and_positive(
    node_id: int,
) -> None:

    shared = (
        SharedSimulation()
    )

    gain = shared.amplitude_gain(
        node_id
    )

    assert math.isfinite(
        gain
    )

    assert (
        gain
        > 0.0
    )


def test_closer_node_has_higher_amplitude_gain() -> None:

    shared = (
        SharedSimulation()
    )

    nodes_by_distance = sorted(
        (
            1,
            2,
            3,
        ),
        key=
            shared.distance_to_source,
    )

    closest = (
        nodes_by_distance[
            0
        ]
    )

    farthest = (
        nodes_by_distance[
            -1
        ]
    )

    assert (
        shared.distance_to_source(
            closest
        )
        < shared.distance_to_source(
            farthest
        )
    )

    assert (
        shared.amplitude_gain(
            closest
        )
        > shared.amplitude_gain(
            farthest
        )
    )


# ======================================================================
# SHARED CLOCK — ARMING
# ======================================================================


def test_arm_node_records_session() -> None:

    shared = (
        SharedSimulation()
    )

    shared.arm_node(
        2,
        TEST_SESSION_ID,
    )

    assert (
        shared.armed_session[
            2
        ]
        == TEST_SESSION_ID
    )


def test_rearming_node_updates_session() -> None:

    shared = (
        SharedSimulation()
    )

    shared.arm_node(
        2,
        TEST_SESSION_ID,
    )

    shared.arm_node(
        2,
        SECOND_SESSION_ID,
    )

    assert (
        shared.armed_session[
            2
        ]
        == SECOND_SESSION_ID
    )


def test_disarm_matching_session_removes_node() -> None:

    shared = (
        SharedSimulation()
    )

    shared.arm_node(
        3,
        TEST_SESSION_ID,
    )

    shared.disarm_node(
        3,
        TEST_SESSION_ID,
    )

    assert (
        3
        not in shared.armed_session
    )


def test_disarm_wrong_session_does_not_remove_node() -> None:

    shared = (
        SharedSimulation()
    )

    shared.arm_node(
        3,
        TEST_SESSION_ID,
    )

    shared.disarm_node(
        3,
        SECOND_SESSION_ID,
    )

    assert (
        shared.armed_session[
            3
        ]
        == TEST_SESSION_ID
    )


# ======================================================================
# SHARED CLOCK — ACTIVATION
# ======================================================================


def test_clock_starts_when_slaves_are_armed() -> None:

    shared = (
        SharedSimulation()
    )

    shared.arm_node(
        2,
        TEST_SESSION_ID,
    )

    shared.arm_node(
        3,
        TEST_SESSION_ID,
    )

    slaves_ready = shared.activate_clock(
        TEST_SESSION_ID
    )

    assert (
        slaves_ready
        is True
    )

    assert (
        shared.clock_session_id
        == TEST_SESSION_ID
    )

    assert (
        shared.clock_event.is_set()
    )


def test_clock_reports_ordering_violation_when_slave_missing() -> None:
    """
    The simulator deliberately activates the clock even when startup
    order is wrong so the violation remains observable rather than
    deadlocking the test environment.
    """

    shared = (
        SharedSimulation()
    )

    shared.arm_node(
        2,
        TEST_SESSION_ID,
    )

    slaves_ready = shared.activate_clock(
        TEST_SESSION_ID
    )

    assert (
        slaves_ready
        is False
    )

    assert (
        shared.clock_session_id
        == TEST_SESSION_ID
    )

    assert (
        shared.clock_event.is_set()
    )


def test_deactivate_matching_clock_session() -> None:

    shared = (
        SharedSimulation()
    )

    shared.activate_clock(
        TEST_SESSION_ID
    )

    shared.deactivate_clock(
        TEST_SESSION_ID
    )

    assert (
        shared.clock_session_id
        is None
    )

    assert not (
        shared.clock_event.is_set()
    )


def test_deactivate_wrong_session_leaves_clock_running() -> None:

    shared = (
        SharedSimulation()
    )

    shared.activate_clock(
        TEST_SESSION_ID
    )

    shared.deactivate_clock(
        SECOND_SESSION_ID
    )

    assert (
        shared.clock_session_id
        == TEST_SESSION_ID
    )

    assert (
        shared.clock_event.is_set()
    )


# ======================================================================
# STREAMING ELIGIBILITY
# ======================================================================


def test_node_can_stream_only_when_armed_and_clock_matches() -> None:

    shared = (
        SharedSimulation()
    )

    shared.arm_node(
        2,
        TEST_SESSION_ID,
    )

    # Armed, but no BCLK/WS yet.
    assert not (
        shared.node_can_stream(
            2,
            TEST_SESSION_ID,
        )
    )

    shared.activate_clock(
        TEST_SESSION_ID
    )

    assert (
        shared.node_can_stream(
            2,
            TEST_SESSION_ID,
        )
    )


def test_node_cannot_stream_using_different_session_clock() -> None:

    shared = (
        SharedSimulation()
    )

    shared.arm_node(
        2,
        TEST_SESSION_ID,
    )

    shared.activate_clock(
        SECOND_SESSION_ID
    )

    assert not (
        shared.node_can_stream(
            2,
            TEST_SESSION_ID,
        )
    )


# ======================================================================
# FAKE NODE CONSTRUCTION
# ======================================================================


@pytest.mark.parametrize(
    "node_id",
    [
        1,
        2,
        3,
    ],
)
def test_valid_simulator_nodes_can_be_constructed(
    node_id: int,
) -> None:

    node = make_node(
        node_id
    )

    assert (
        node.node_id
        == node_id
    )


@pytest.mark.parametrize(
    "node_id",
    [
        0,
        -1,
        4,
        255,
    ],
)
def test_invalid_simulator_node_id_is_rejected(
    node_id: int,
) -> None:

    with pytest.raises(
        ValueError
    ):

        make_node(
            node_id
        )


# ======================================================================
# NODE ROLE
# ======================================================================


def test_node_1_is_master() -> None:

    node = make_node(
        1
    )

    assert (
        node.master_node
    )


@pytest.mark.parametrize(
    "node_id",
    [
        2,
        3,
    ],
)
def test_nodes_2_and_3_are_slaves(
    node_id: int,
) -> None:

    node = make_node(
        node_id
    )

    assert not (
        node.master_node
    )


# ======================================================================
# FAKE NODE STREAMING PROPERTY
# ======================================================================


def test_fake_node_initially_not_streaming() -> None:

    node = make_node(
        2
    )

    assert not (
        node.streaming
    )


def test_fake_node_streaming_requires_local_armed_state() -> None:

    shared = (
        SharedSimulation()
    )

    node = make_node(
        2,
        shared=
            shared,
    )

    node.session_id = (
        TEST_SESSION_ID
    )

    shared.arm_node(
        2,
        TEST_SESSION_ID,
    )

    shared.activate_clock(
        TEST_SESSION_ID
    )

    # Shared world says node can stream, but the fake ESP32 itself has
    # not yet entered its armed state.
    assert not (
        node.streaming
    )

    node.armed = (
        True
    )

    assert (
        node.streaming
    )


def test_fake_node_zero_session_never_streams() -> None:

    shared = (
        SharedSimulation()
    )

    node = make_node(
        2,
        shared=
            shared,
    )

    node.armed = (
        True
    )

    node.session_id = (
        0
    )

    shared.arm_node(
        2,
        0,
    )

    shared.activate_clock(
        0
    )

    assert not (
        node.streaming
    )


# ======================================================================
# SEQUENCE NUMBER
# ======================================================================


def test_sequence_starts_at_zero() -> None:

    node = make_node(
        1
    )

    assert (
        node.next_sequence()
        == 0
    )

    assert (
        node.next_sequence()
        == 1
    )


def test_sequence_wraps_as_uint32() -> None:

    node = make_node(
        1
    )

    node.sequence = (
        UINT32_MASK
    )

    assert (
        node.next_sequence()
        == UINT32_MASK
    )

    assert (
        node.sequence
        == 0
    )

    assert (
        node.next_sequence()
        == 0
    )


# ======================================================================
# LOCAL MICROSECOND COUNTER
# ======================================================================


def test_local_micros_is_uint32() -> None:

    node = make_node(
        1
    )

    value = (
        node.local_micros()
    )

    assert (
        0
        <= value
        <= UINT32_MASK
    )


def test_local_micros_is_diagnostic_monotonic_over_short_interval() -> None:

    node = make_node(
        1
    )

    first = (
        node.local_micros()
    )

    time.sleep(
        0.001
    )

    second = (
        node.local_micros()
    )

    # Over a tiny interval we are nowhere near uint32 wraparound.
    assert (
        second
        >= first
    )


# ======================================================================
# PACKET CONSTRUCTION
# ======================================================================


def test_fake_node_builds_protocol_packet() -> None:

    node = make_node(
        2
    )

    node.session_id = (
        TEST_SESSION_ID
    )

    node.sample_index = (
        4096
    )

    payload = (
        b"\x01\x02\x03\x04"
    )

    raw = node.packet(
        PacketType.AUDIO,
        payload,
    )

    header = unpack_header(
        raw[
            :HEADER_SIZE
        ]
    )

    stored_payload = (
        raw[
            HEADER_SIZE:
        ]
    )

    assert (
        header.node_id
        == 2
    )

    assert (
        header.packet_type
        == PacketType.AUDIO
    )

    assert (
        header.session_id
        == TEST_SESSION_ID
    )

    assert (
        header.sample_index
        == 4096
    )

    assert (
        header.sequence
        == 0
    )

    assert (
        stored_payload
        == payload
    )

    verify_payload_crc(
        header,
        stored_payload,
    )


def test_packet_advances_sequence() -> None:

    node = make_node(
        1
    )

    node.session_id = (
        TEST_SESSION_ID
    )

    first = node.packet(
        PacketType.HEARTBEAT
    )

    second = node.packet(
        PacketType.HEARTBEAT
    )

    first_header = unpack_header(
        first[
            :HEADER_SIZE
        ]
    )

    second_header = unpack_header(
        second[
            :HEADER_SIZE
        ]
    )

    assert (
        first_header.sequence
        == 0
    )

    assert (
        second_header.sequence
        == 1
    )


def test_packet_can_override_sample_index() -> None:

    node = make_node(
        1
    )

    node.session_id = (
        TEST_SESSION_ID
    )

    node.sample_index = (
        9999
    )

    raw = node.packet(
        PacketType.SYNC,
        sample_index=
            0,
    )

    header = unpack_header(
        raw[
            :HEADER_SIZE
        ]
    )

    assert (
        header.sample_index
        == 0
    )


def test_packet_preserves_health_flags() -> None:

    node = make_node(
        1
    )

    node.session_id = (
        TEST_SESSION_ID
    )

    raw = node.packet(
        PacketType.AUDIO,
        b"\x00\x00",
        flags=
            int(
                PacketFlags.CLIPPED
                | PacketFlags.QUEUE_CONGESTED
            ),
    )

    header = unpack_header(
        raw[
            :HEADER_SIZE
        ]
    )

    assert (
        header.flags
        == int(
            PacketFlags.CLIPPED
            | PacketFlags.QUEUE_CONGESTED
        )
    )


# ======================================================================
# HELLO
# ======================================================================


@pytest.mark.parametrize(
    (
        "node_id",
        "expected_master",
    ),
    [
        (
            1,
            True,
        ),
        (
            2,
            False,
        ),
        (
            3,
            False,
        ),
    ],
)
def test_hello_payload_matches_node_role(
    node_id: int,
    expected_master: bool,
) -> None:

    node = make_node(
        node_id
    )

    hello = parse_hello(
        node.hello_payload()
    )

    assert (
        hello.sample_rate
        == SAMPLE_RATE
    )

    assert (
        hello.frames_per_packet
        == FRAMES_PER_BLOCK
    )

    assert (
        hello.bits_per_sample
        == 16
    )

    assert (
        hello.channels
        == 1
    )

    assert (
        hello.master_node
        is expected_master
    )

    assert (
        hello.firmware
        == "sim-proto-v4"
    )


# ======================================================================
# SYNTHETIC EVENT WAVEFORM
# ======================================================================


def test_event_waveform_is_mono_float64() -> None:

    node = make_node(
        1
    )

    waveform = (
        node.event_waveform
    )

    assert (
        waveform.ndim
        == 1
    )

    assert (
        waveform.dtype
        == np.float64
    )

    assert (
        waveform.size
        > 0
    )

    assert np.all(
        np.isfinite(
            waveform
        )
    )


def test_event_waveform_duration_is_about_550_ms() -> None:

    node = make_node(
        1
    )

    duration_s = (
        node.event_waveform.size
        / SAMPLE_RATE
    )

    assert (
        duration_s
        == pytest.approx(
            0.55,
            abs=
                1.0
                / SAMPLE_RATE,
        )
    )


def test_event_waveform_has_smooth_zero_edges() -> None:

    waveform = (
        make_node(
            1
        ).event_waveform
    )

    assert (
        abs(
            waveform[
                0
            ]
        )
        < 1e-9
    )

    assert (
        abs(
            waveform[
                -1
            ]
        )
        < 1e-6
    )


def test_event_waveform_contains_nonzero_energy() -> None:

    waveform = (
        make_node(
            1
        ).event_waveform
    )

    assert (
        rms(
            waveform
        )
        > 100.0
    )


# ======================================================================
# AUDIO GENERATION
# ======================================================================


@pytest.mark.parametrize(
    "node_id",
    [
        1,
        2,
        3,
    ],
)
def test_generate_audio_returns_pcm16_block(
    node_id: int,
) -> None:

    node = make_node(
        node_id
    )

    (
        pcm,
        clipped,
    ) = node.generate_audio(
        0
    )

    assert (
        pcm.ndim
        == 1
    )

    assert (
        pcm.size
        == FRAMES_PER_BLOCK
    )

    assert (
        pcm.dtype
        == np.dtype(
            "<i2"
        )
    )

    assert isinstance(
        clipped,
        bool,
    )


def test_generated_pcm_stays_within_int16_range() -> None:

    node = make_node(
        1
    )

    (
        pcm,
        _,
    ) = node.generate_audio(
        EVENT_FIRST_SAMPLE
    )

    assert (
        int(
            np.min(
                pcm
            )
        )
        >= -32768
    )

    assert (
        int(
            np.max(
                pcm
            )
        )
        <= 32767
    )


# ======================================================================
# DETERMINISTIC AUDIO
# ======================================================================


def test_same_node_seed_produces_repeatable_audio() -> None:

    shared_a = (
        SharedSimulation()
    )

    shared_b = (
        SharedSimulation()
    )

    first = make_node(
        1,
        shared=
            shared_a,
    )

    second = make_node(
        1,
        shared=
            shared_b,
    )

    (
        first_pcm,
        first_clipped,
    ) = first.generate_audio(
        0
    )

    (
        second_pcm,
        second_clipped,
    ) = second.generate_audio(
        0
    )

    np.testing.assert_array_equal(
        first_pcm,
        second_pcm,
    )

    assert (
        first_clipped
        == second_clipped
    )


def test_different_nodes_have_independent_background_noise() -> None:

    shared = (
        SharedSimulation()
    )

    node_1 = make_node(
        1,
        shared=
            shared,
    )

    node_2 = make_node(
        2,
        shared=
            shared,
    )

    (
        audio_1,
        _,
    ) = node_1.generate_audio(
        0
    )

    (
        audio_2,
        _,
    ) = node_2.generate_audio(
        0
    )

    assert not np.array_equal(
        audio_1,
        audio_2,
    )


# ======================================================================
# BACKGROUND VS EVENT
# ======================================================================


def test_synthetic_event_has_substantially_more_energy_than_background() -> None:
    """
    Compare two fresh node instances so both start with the same
    deterministic RNG state.

    One generates background before the first event.

    The other generates a block from the central region of the first
    acoustically delayed event.
    """

    shared = (
        SharedSimulation()
    )

    background_node = make_node(
        1,
        shared=
            shared,
    )

    event_node = make_node(
        1,
        shared=
            shared,
    )

    (
        background,
        _,
    ) = background_node.generate_audio(
        0
    )

    propagation_delay = (
        shared.delay_samples(
            1
        )
    )

    event_middle = (
        EVENT_FIRST_SAMPLE
        + propagation_delay
        + int(
            0.20
            * SAMPLE_RATE
        )
    )

    (
        event_audio,
        _,
    ) = event_node.generate_audio(
        event_middle
    )

    assert (
        rms(
            event_audio
        )
        > rms(
            background
        )
        * 5.0
    )


# ======================================================================
# CLIPPING DETECTION
# ======================================================================


def test_clipping_flag_is_false_for_normal_background() -> None:

    node = make_node(
        1
    )

    (
        _,
        clipped,
    ) = node.generate_audio(
        0
    )

    assert (
        clipped
        is False
    )


def test_clipping_is_detected_before_pcm_saturation() -> None:
    """
    Artificially amplify the synthetic source to exercise the clipping
    diagnostic.

    PCM output must remain legal int16 while clipped=True reports that
    the pre-clipped floating waveform exceeded the representable range.
    """

    shared = (
        SharedSimulation()
    )

    node = make_node(
        1,
        shared=
            shared,
    )

    node.event_waveform *= (
        20.0
    )

    event_sample = (
        EVENT_FIRST_SAMPLE
        + shared.delay_samples(
            1
        )
        + int(
            0.20
            * SAMPLE_RATE
        )
    )

    (
        pcm,
        clipped,
    ) = node.generate_audio(
        event_sample
    )

    assert (
        clipped
        is True
    )

    assert (
        int(
            np.min(
                pcm
            )
        )
        >= -32768
    )

    assert (
        int(
            np.max(
                pcm
            )
        )
        <= 32767
    )


# ======================================================================
# SERIALIZED PACKET SENDING
# ======================================================================


def test_send_packet_writes_one_valid_frame() -> None:

    async def scenario() -> None:

        node = make_node(
            2
        )

        node.session_id = (
            TEST_SESSION_ID
        )

        writer = (
            FakeWriter()
        )

        payload = (
            b"\x10\x20\x30\x40"
        )

        await node._send_packet(
            writer,
            PacketType.AUDIO,
            payload,
            sample_index=
                2048,
        )

        raw = bytes(
            writer.data
        )

        header = unpack_header(
            raw[
                :HEADER_SIZE
            ]
        )

        stored_payload = (
            raw[
                HEADER_SIZE:
            ]
        )

        assert (
            header.node_id
            == 2
        )

        assert (
            header.sample_index
            == 2048
        )

        assert (
            stored_payload
            == payload
        )

        assert (
            writer.drain_count
            == 1
        )

        verify_payload_crc(
            header,
            stored_payload,
        )

    asyncio.run(
        scenario()
    )


def test_send_packet_rejects_closing_writer() -> None:

    async def scenario() -> None:

        node = make_node(
            1
        )

        writer = (
            FakeWriter()
        )

        writer.close()

        with pytest.raises(
            ConnectionError
        ):

            await node._send_packet(
                writer,
                PacketType.HEARTBEAT,
            )

    asyncio.run(
        scenario()
    )


# ======================================================================
# SESSION SYNC
# ======================================================================


def test_session_sync_contains_matching_session_and_zero_sample_index() -> None:

    async def scenario() -> None:

        node = make_node(
            2
        )

        node.session_id = (
            TEST_SESSION_ID
        )

        writer = (
            FakeWriter()
        )

        await node._send_session_sync(
            writer
        )

        raw = bytes(
            writer.data
        )

        header = unpack_header(
            raw[
                :HEADER_SIZE
            ]
        )

        assert (
            header.packet_type
            == PacketType.SYNC
        )

        assert (
            header.session_id
            == TEST_SESSION_ID
        )

        assert (
            header.sample_index
            == 0
        )

        sync = parse_sync(
            raw[
                HEADER_SIZE:
            ]
        )

        assert (
            sync.session_id
            == TEST_SESSION_ID
        )

        assert (
            sync.sync_id
            == 1
        )

        assert (
            sync.sample_index
            == 0
        )

        assert (
            node._sync_sent_session
            == TEST_SESSION_ID
        )

    asyncio.run(
        scenario()
    )


def test_zero_session_does_not_emit_sync() -> None:

    async def scenario() -> None:

        node = make_node(
            2
        )

        node.session_id = (
            0
        )

        writer = (
            FakeWriter()
        )

        await node._send_session_sync(
            writer
        )

        assert (
            writer.data
            == bytearray()
        )

        assert (
            node._sync_sent_session
            is None
        )

    asyncio.run(
        scenario()
    )


# ======================================================================
# MASTER HEARTBEAT
# ======================================================================


def test_master_heartbeat_reports_bme_and_stream_state() -> None:

    async def scenario() -> None:

        shared = (
            SharedSimulation()
        )

        node = make_node(
            1,
            shared=
                shared,
        )

        node.session_id = (
            TEST_SESSION_ID
        )

        node.armed = (
            True
        )

        shared.arm_node(
            1,
            TEST_SESSION_ID,
        )

        shared.arm_node(
            2,
            TEST_SESSION_ID,
        )

        shared.arm_node(
            3,
            TEST_SESSION_ID,
        )

        shared.activate_clock(
            TEST_SESSION_ID
        )

        writer = (
            FakeWriter()
        )

        await node._send_heartbeat(
            writer
        )

        raw = bytes(
            writer.data
        )

        header = unpack_header(
            raw[
                :HEADER_SIZE
            ]
        )

        heartbeat = parse_heartbeat(
            raw[
                HEADER_SIZE:
            ],
            master_node=
                True,
        )

        assert (
            header.packet_type
            == PacketType.HEARTBEAT
        )

        assert (
            heartbeat.streaming
            is True
        )

        assert (
            heartbeat.bme_available
            is True
        )

        assert (
            heartbeat.sync_received
            is None
        )

        assert (
            heartbeat.clock_healthy
            is None
        )

    asyncio.run(
        scenario()
    )


# ======================================================================
# SLAVE HEARTBEAT
# ======================================================================


def test_slave_heartbeat_reports_sync_and_clock_health() -> None:

    async def scenario() -> None:

        shared = (
            SharedSimulation()
        )

        node = make_node(
            2,
            shared=
                shared,
        )

        node.session_id = (
            TEST_SESSION_ID
        )

        node.armed = (
            True
        )

        node._sync_sent_session = (
            TEST_SESSION_ID
        )

        shared.arm_node(
            2,
            TEST_SESSION_ID,
        )

        shared.arm_node(
            3,
            TEST_SESSION_ID,
        )

        shared.activate_clock(
            TEST_SESSION_ID
        )

        writer = (
            FakeWriter()
        )

        await node._send_heartbeat(
            writer
        )

        raw = bytes(
            writer.data
        )

        heartbeat = parse_heartbeat(
            raw[
                HEADER_SIZE:
            ],
            master_node=
                False,
        )

        assert (
            heartbeat.streaming
            is True
        )

        assert (
            heartbeat.sync_received
            is True
        )

        assert (
            heartbeat.clock_healthy
            is True
        )

        assert (
            heartbeat.bme_available
            is None
        )

    asyncio.run(
        scenario()
    )