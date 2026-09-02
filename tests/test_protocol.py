"""
Tests for the binary communication protocol.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

These tests protect the binary wire contract shared by:

    ESP32 Node 1
    ESP32 Node 2
    ESP32 Node 3
          ↓
       TCP/Wi-Fi
          ↓
    Laptop Receiver

The protocol is a hardware/software boundary.

Tests therefore cover both:

    successful serialization/deserialization
    malformed/corrupted binary input
"""


from __future__ import annotations


# ======================================================================
# THIRD-PARTY
# ======================================================================


import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.core.protocol import (
    CONTROL_SIZE,
    CONTROL_VERSION,
    ENVIRONMENT_STRUCT,
    HEADER_SIZE,
    HELLO_STRUCT,
    MASTER_HEARTBEAT_STRUCT,
    PROTOCOL_VERSION,
    SLAVE_HEARTBEAT_STRUCT,
    SYNC_STRUCT,
    CRCMismatch,
    ControlCommand,
    ControlFrame,
    EnvironmentPayload,
    HeartbeatPayload,
    HelloPayload,
    InvalidControlFrame,
    InvalidHeader,
    PacketFlags,
    PacketType,
    ProtocolError,
    SyncPayload,
    build_packet,
    crc32,
    pack_control,
    pack_environment,
    pack_hello,
    pack_master_heartbeat,
    pack_slave_heartbeat,
    pack_sync,
    parse_environment,
    parse_heartbeat,
    parse_hello,
    parse_sync,
    unpack_control,
    unpack_header,
    verify_payload_crc,
)


# ======================================================================
# WIRE LAYOUT
# ======================================================================


def test_protocol_wire_sizes() -> None:
    """
    Binary layouts must never change accidentally.

    Any change here requires a coordinated firmware + laptop protocol
    version update.
    """

    assert HEADER_SIZE == 40

    assert CONTROL_SIZE == 8

    assert HELLO_STRUCT.size == 27

    assert ENVIRONMENT_STRUCT.size == 12

    assert SYNC_STRUCT.size == 20

    assert MASTER_HEARTBEAT_STRUCT.size == 28

    assert SLAVE_HEARTBEAT_STRUCT.size == 29


# ======================================================================
# CRC32
# ======================================================================


def test_crc32_known_reference_vector() -> None:
    """
    CRC32("123456789") is the standard reference vector.
    """

    assert (
        crc32(
            b"123456789"
        )
        == 0xCBF43926
    )


# ======================================================================
# COMPLETE PACKET
# ======================================================================


def test_header_and_crc_roundtrip() -> None:

    payload = (
        b"\x01\x02\x03\x04"
    )

    flags = int(
        PacketFlags.CLIPPED
        | PacketFlags.QUEUE_CONGESTED
    )

    raw = build_packet(
        node_id=
            2,

        packet_type=
            PacketType.AUDIO,

        sequence=
            10,

        session_id=
            20,

        sample_index=
            1024,

        local_micros=
            123,

        i2s_error_count=
            1,

        flags=
            flags,

        payload=
            payload,
    )

    # --------------------------------------------------------------
    # COMPLETE FRAME LENGTH
    # --------------------------------------------------------------

    assert (
        len(
            raw
        )
        == HEADER_SIZE
        + len(
            payload
        )
    )

    # --------------------------------------------------------------
    # HEADER
    # --------------------------------------------------------------

    header = unpack_header(
        raw[
            :HEADER_SIZE
        ]
    )

    assert (
        header.protocol_version
        == PROTOCOL_VERSION
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
        header.flags
        == flags
    )

    assert (
        header.sequence
        == 10
    )

    assert (
        header.session_id
        == 20
    )

    assert (
        header.sample_index
        == 1024
    )

    assert (
        header.local_micros
        == 123
    )

    assert (
        header.i2s_error_count
        == 1
    )

    assert (
        header.payload_length
        == len(
            payload
        )
    )

    # --------------------------------------------------------------
    # CRC
    # --------------------------------------------------------------

    verify_payload_crc(
        header,
        raw[
            HEADER_SIZE:
        ],
    )


def test_empty_payload_packet_has_zero_crc() -> None:

    raw = build_packet(
        node_id=
            1,

        packet_type=
            PacketType.HEARTBEAT,

        sequence=
            0,

        session_id=
            1,

        sample_index=
            0,

        payload=
            b"",
    )

    header = unpack_header(
        raw[
            :HEADER_SIZE
        ]
    )

    assert (
        header.payload_length
        == 0
    )

    assert (
        header.payload_crc32
        == 0
    )

    verify_payload_crc(
        header,
        b"",
    )


def test_packet_supports_protocol_integer_boundaries() -> None:

    payload = (
        b"\xAA"
    )

    raw = build_packet(
        node_id=
            255,

        packet_type=
            PacketType.AUDIO,

        sequence=
            0xFFFFFFFF,

        session_id=
            0xFFFFFFFF,

        sample_index=
            0xFFFFFFFFFFFFFFFF,

        local_micros=
            0xFFFFFFFF,

        i2s_error_count=
            0xFFFFFFFF,

        flags=
            0xFF,

        payload=
            payload,
    )

    header = unpack_header(
        raw[
            :HEADER_SIZE
        ]
    )

    assert (
        header.node_id
        == 255
    )

    assert (
        header.sequence
        == 0xFFFFFFFF
    )

    assert (
        header.session_id
        == 0xFFFFFFFF
    )

    assert (
        header.sample_index
        == 0xFFFFFFFFFFFFFFFF
    )

    assert (
        header.local_micros
        == 0xFFFFFFFF
    )

    assert (
        header.i2s_error_count
        == 0xFFFFFFFF
    )


# ======================================================================
# PAYLOAD CRC FAILURE
# ======================================================================


def test_crc_mismatch_is_detected() -> None:

    payload = (
        b"\x10\x20\x30\x40"
    )

    raw = build_packet(
        node_id=
            2,

        packet_type=
            PacketType.AUDIO,

        sequence=
            3,

        session_id=
            100,

        sample_index=
            2048,

        payload=
            payload,
    )

    header = unpack_header(
        raw[
            :HEADER_SIZE
        ]
    )

    corrupted_payload = bytearray(
        raw[
            HEADER_SIZE:
        ]
    )

    corrupted_payload[
        0
    ] ^= (
        0xFF
    )

    with pytest.raises(
        CRCMismatch
    ):

        verify_payload_crc(
            header,
            bytes(
                corrupted_payload
            ),
        )


def test_payload_length_mismatch_is_detected() -> None:

    payload = (
        b"\x01\x02\x03\x04"
    )

    raw = build_packet(
        node_id=
            1,

        packet_type=
            PacketType.AUDIO,

        sequence=
            1,

        session_id=
            10,

        sample_index=
            0,

        payload=
            payload,
    )

    header = unpack_header(
        raw[
            :HEADER_SIZE
        ]
    )

    with pytest.raises(
        ProtocolError
    ):

        verify_payload_crc(
            header,
            payload[
                :-1
            ],
        )


# ======================================================================
# HEADER VALIDATION
# ======================================================================


def test_rejects_incorrect_header_size() -> None:

    with pytest.raises(
        InvalidHeader
    ):

        unpack_header(
            b"\x00"
            * (
                HEADER_SIZE
                - 1
            )
        )


def test_rejects_bad_packet_magic() -> None:

    raw = build_packet(
        node_id=
            1,

        packet_type=
            PacketType.HELLO,

        sequence=
            0,

        session_id=
            1,

        sample_index=
            0,
    )

    header_bytes = bytearray(
        raw[
            :HEADER_SIZE
        ]
    )

    # First byte belongs to uint32 magic.
    header_bytes[
        0
    ] ^= (
        0xFF
    )

    with pytest.raises(
        InvalidHeader
    ):

        unpack_header(
            bytes(
                header_bytes
            )
        )


def test_rejects_wrong_protocol_version() -> None:

    raw = build_packet(
        node_id=
            1,

        packet_type=
            PacketType.HELLO,

        sequence=
            0,

        session_id=
            1,

        sample_index=
            0,
    )

    header_bytes = bytearray(
        raw[
            :HEADER_SIZE
        ]
    )

    # Header layout:
    #
    # bytes 0..3  -> magic
    # byte 4      -> protocolVersion
    header_bytes[
        4
    ] = (
        (
            PROTOCOL_VERSION
            + 1
        )
        & 0xFF
    )

    with pytest.raises(
        InvalidHeader
    ):

        unpack_header(
            bytes(
                header_bytes
            )
        )


def test_rejects_zero_node_id_in_received_header() -> None:

    raw = build_packet(
        node_id=
            1,

        packet_type=
            PacketType.AUDIO,

        sequence=
            0,

        session_id=
            1,

        sample_index=
            0,
    )

    header_bytes = bytearray(
        raw[
            :HEADER_SIZE
        ]
    )

    # byte 5 = nodeId
    header_bytes[
        5
    ] = (
        0
    )

    with pytest.raises(
        InvalidHeader
    ):

        unpack_header(
            bytes(
                header_bytes
            )
        )


def test_rejects_unknown_packet_type() -> None:

    raw = build_packet(
        node_id=
            1,

        packet_type=
            PacketType.AUDIO,

        sequence=
            0,

        session_id=
            1,

        sample_index=
            0,
    )

    header_bytes = bytearray(
        raw[
            :HEADER_SIZE
        ]
    )

    # byte 6 = packetType
    header_bytes[
        6
    ] = (
        0xFF
    )

    with pytest.raises(
        InvalidHeader
    ):

        unpack_header(
            bytes(
                header_bytes
            )
        )


# ======================================================================
# PACKET BUILD RANGE VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "node_id",
    [
        0,
        256,
    ],
)
def test_build_packet_rejects_invalid_node_id(
    node_id: int,
) -> None:

    with pytest.raises(
        ValueError
    ):

        build_packet(
            node_id=
                node_id,

            packet_type=
                PacketType.AUDIO,

            sequence=
                0,

            session_id=
                1,

            sample_index=
                0,
        )


def test_build_packet_rejects_negative_sequence() -> None:

    with pytest.raises(
        ValueError
    ):

        build_packet(
            node_id=
                1,

            packet_type=
                PacketType.AUDIO,

            sequence=
                -1,

            session_id=
                1,

            sample_index=
                0,
        )


def test_build_packet_rejects_sample_index_overflow() -> None:

    with pytest.raises(
        ValueError
    ):

        build_packet(
            node_id=
                1,

            packet_type=
                PacketType.AUDIO,

            sequence=
                0,

            session_id=
                1,

            sample_index=
                1
                << 64,
        )


# ======================================================================
# HELLO
# ======================================================================


def test_hello_roundtrip() -> None:

    hello = HelloPayload(
        sample_rate=
            48_000,

        frames_per_packet=
            1024,

        bits_per_sample=
            16,

        channels=
            1,

        master_node=
            True,

        sync_tolerance_samples=
            10,

        firmware=
            "1.2.0",
    )

    assert (
        parse_hello(
            pack_hello(
                hello
            )
        )
        == hello
    )


def test_hello_firmware_is_truncated_safely() -> None:

    firmware = (
        "abcdefghijklmnopq"
    )

    hello = HelloPayload(
        sample_rate=
            48_000,

        frames_per_packet=
            1024,

        bits_per_sample=
            16,

        channels=
            1,

        master_node=
            False,

        sync_tolerance_samples=
            10,

        firmware=
            firmware,
    )

    parsed = parse_hello(
        pack_hello(
            hello
        )
    )

    # 16-byte firmware storage reserves room for NUL termination.
    assert (
        parsed.firmware
        == firmware[
            :15
        ]
    )


def test_parse_hello_rejects_wrong_payload_size() -> None:

    with pytest.raises(
        ProtocolError
    ):

        parse_hello(
            b"\x00"
            * (
                HELLO_STRUCT.size
                - 1
            )
        )


def test_parse_hello_rejects_invalid_boolean_byte() -> None:

    firmware = (
        b"test-fw"
        .ljust(
            16,
            b"\0",
        )
    )

    raw = HELLO_STRUCT.pack(
        48_000,
        1024,
        16,
        1,

        # Invalid boolean.
        2,

        10,
        firmware,
    )

    with pytest.raises(
        ProtocolError
    ):

        parse_hello(
            raw
        )


def test_pack_hello_rejects_negative_sync_tolerance() -> None:

    hello = HelloPayload(
        sample_rate=
            48_000,

        frames_per_packet=
            1024,

        bits_per_sample=
            16,

        channels=
            1,

        master_node=
            True,

        sync_tolerance_samples=
            -1,

        firmware=
            "test",
    )

    with pytest.raises(
        ValueError
    ):

        pack_hello(
            hello
        )


# ======================================================================
# CONTROL FRAME
# ======================================================================


@pytest.mark.parametrize(
    "command",
    [
        ControlCommand.START,
        ControlCommand.STOP,
        ControlCommand.PING,
    ],
)
def test_control_frame_roundtrip(
    command: ControlCommand,
) -> None:

    frame = ControlFrame(
        command=
            command,

        session_id=
            0x12345678,
    )

    packed = pack_control(
        frame
    )

    assert (
        len(
            packed
        )
        == CONTROL_SIZE
    )

    assert (
        unpack_control(
            packed
        )
        == frame
    )


def test_unpack_control_rejects_wrong_size() -> None:

    with pytest.raises(
        InvalidControlFrame
    ):

        unpack_control(
            b"\x00"
            * (
                CONTROL_SIZE
                - 1
            )
        )


def test_unpack_control_rejects_bad_magic() -> None:

    raw = bytearray(
        pack_control(
            ControlFrame(
                command=
                    ControlCommand.START,

                session_id=
                    1,
            )
        )
    )

    raw[
        0
    ] ^= (
        0xFF
    )

    with pytest.raises(
        InvalidControlFrame
    ):

        unpack_control(
            bytes(
                raw
            )
        )


def test_unpack_control_rejects_bad_version() -> None:

    raw = bytearray(
        pack_control(
            ControlFrame(
                command=
                    ControlCommand.START,

                session_id=
                    1,
            )
        )
    )

    # <HBBI>
    #
    # bytes 0..1 -> magic
    # byte 2     -> version
    # byte 3     -> command
    raw[
        2
    ] = (
        (
            CONTROL_VERSION
            + 1
        )
        & 0xFF
    )

    with pytest.raises(
        InvalidControlFrame
    ):

        unpack_control(
            bytes(
                raw
            )
        )


def test_unpack_control_rejects_unknown_command() -> None:

    raw = bytearray(
        pack_control(
            ControlFrame(
                command=
                    ControlCommand.START,

                session_id=
                    1,
            )
        )
    )

    raw[
        3
    ] = (
        0xFF
    )

    with pytest.raises(
        InvalidControlFrame
    ):

        unpack_control(
            bytes(
                raw
            )
        )


def test_pack_control_rejects_invalid_session_id() -> None:

    frame = ControlFrame(
        command=
            ControlCommand.START,

        session_id=
            -1,
    )

    with pytest.raises(
        ValueError
    ):

        pack_control(
            frame
        )


# ======================================================================
# ENVIRONMENT
# ======================================================================


def test_environment_roundtrip() -> None:

    environment = EnvironmentPayload(
        temperature_c=
            27.5,

        humidity_percent=
            58.25,

        pressure_hpa=
            1007.2,
    )

    parsed = parse_environment(
        pack_environment(
            environment
        )
    )

    # Wire format uses float32, therefore approximate comparison is
    # appropriate after decoding.
    assert (
        parsed.temperature_c
        == pytest.approx(
            environment.temperature_c,
            rel=
                1e-6,
        )
    )

    assert (
        parsed.humidity_percent
        == pytest.approx(
            environment.humidity_percent,
            rel=
                1e-6,
        )
    )

    assert (
        parsed.pressure_hpa
        == pytest.approx(
            environment.pressure_hpa,
            rel=
                1e-6,
        )
    )


def test_pack_environment_rejects_non_finite_value() -> None:

    environment = EnvironmentPayload(
        temperature_c=
            float(
                "nan"
            ),

        humidity_percent=
            50.0,

        pressure_hpa=
            1013.25,
    )

    with pytest.raises(
        ValueError
    ):

        pack_environment(
            environment
        )


def test_parse_environment_rejects_non_finite_wire_value() -> None:

    raw = ENVIRONMENT_STRUCT.pack(
        float(
            "nan"
        ),
        50.0,
        1013.25,
    )

    with pytest.raises(
        ProtocolError
    ):

        parse_environment(
            raw
        )


# ======================================================================
# SYNC
# ======================================================================


def test_sync_roundtrip() -> None:

    sync = SyncPayload(
        session_id=
            0x12345678,

        sync_id=
            7,

        sample_index=
            987654321,

        local_micros=
            123456,
    )

    assert (
        parse_sync(
            pack_sync(
                sync
            )
        )
        == sync
    )


def test_pack_sync_rejects_negative_session_id() -> None:

    sync = SyncPayload(
        session_id=
            -1,

        sync_id=
            1,

        sample_index=
            0,

        local_micros=
            0,
    )

    with pytest.raises(
        ValueError
    ):

        pack_sync(
            sync
        )


def test_parse_sync_rejects_wrong_payload_size() -> None:

    with pytest.raises(
        ProtocolError
    ):

        parse_sync(
            b"\x00"
            * (
                SYNC_STRUCT.size
                - 1
            )
        )


# ======================================================================
# MASTER HEARTBEAT
# ======================================================================


def test_master_heartbeat_roundtrip() -> None:

    heartbeat = HeartbeatPayload(
        uptime_seconds=
            120,

        wifi_rssi=
            -61,

        free_heap=
            180_000,

        dropped_audio_blocks=
            2,

        transmitted_audio_blocks=
            500,

        i2s_errors=
            1,

        audio_queue_depth=
            3,

        streaming=
            True,

        bme_available=
            True,
    )

    packed = pack_master_heartbeat(
        heartbeat
    )

    assert (
        len(
            packed
        )
        == MASTER_HEARTBEAT_STRUCT.size
    )

    parsed = parse_heartbeat(
        packed,
        master_node=
            True,
    )

    assert (
        parsed
        == heartbeat
    )


# ======================================================================
# SLAVE HEARTBEAT
# ======================================================================


def test_slave_heartbeat_roundtrip() -> None:

    heartbeat = HeartbeatPayload(
        uptime_seconds=
            240,

        wifi_rssi=
            -66,

        free_heap=
            175_000,

        dropped_audio_blocks=
            1,

        transmitted_audio_blocks=
            900,

        i2s_errors=
            0,

        audio_queue_depth=
            2,

        streaming=
            True,

        sync_received=
            True,

        clock_healthy=
            True,
    )

    packed = pack_slave_heartbeat(
        heartbeat
    )

    assert (
        len(
            packed
        )
        == SLAVE_HEARTBEAT_STRUCT.size
    )

    parsed = parse_heartbeat(
        packed,
        master_node=
            False,
    )

    assert (
        parsed
        == heartbeat
    )


def test_heartbeat_role_mismatch_is_rejected() -> None:

    master_heartbeat = HeartbeatPayload(
        uptime_seconds=
            1,

        wifi_rssi=
            -60,

        free_heap=
            100_000,

        dropped_audio_blocks=
            0,

        transmitted_audio_blocks=
            1,

        i2s_errors=
            0,

        audio_queue_depth=
            0,

        streaming=
            False,

        bme_available=
            True,
    )

    packed = pack_master_heartbeat(
        master_heartbeat
    )

    with pytest.raises(
        ProtocolError
    ):

        parse_heartbeat(
            packed,
            master_node=
                False,
        )


def test_parse_heartbeat_rejects_invalid_boolean_byte() -> None:

    raw = MASTER_HEARTBEAT_STRUCT.pack(
        10,
        -60,
        100_000,
        0,
        100,
        0,
        2,

        # streaming -- invalid boolean byte
        2,

        # BME available
        1,
    )

    with pytest.raises(
        ProtocolError
    ):

        parse_heartbeat(
            raw,
            master_node=
                True,
        )


def test_pack_heartbeat_rejects_out_of_range_rssi() -> None:

    heartbeat = HeartbeatPayload(
        uptime_seconds=
            1,

        wifi_rssi=
            1
            << 31,

        free_heap=
            100_000,

        dropped_audio_blocks=
            0,

        transmitted_audio_blocks=
            1,

        i2s_errors=
            0,

        audio_queue_depth=
            0,

        streaming=
            False,

        bme_available=
            True,
    )

    with pytest.raises(
        ValueError
    ):

        pack_master_heartbeat(
            heartbeat
        )