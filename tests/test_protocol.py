from protocol import (
    HEADER_SIZE,
    HelloPayload,
    PacketType,
    build_packet,
    pack_hello,
    parse_hello,
    unpack_header,
    verify_payload_crc,
)


def test_header_and_crc_roundtrip():
    payload = b"\x01\x02\x03\x04"
    raw = build_packet(
        node_id=2,
        packet_type=PacketType.AUDIO,
        sequence=10,
        session_id=20,
        sample_index=1024,
        local_micros=123,
        i2s_error_count=1,
        payload=payload,
    )
    header = unpack_header(raw[:HEADER_SIZE])
    assert header.node_id == 2
    assert header.sequence == 10
    assert header.sample_index == 1024
    verify_payload_crc(header, raw[HEADER_SIZE:])


def test_hello_roundtrip():
    hello = HelloPayload(48000, 1024, 16, 1, True, 10, "1.2.0")
    assert parse_hello(pack_hello(hello)) == hello


def test_control_frame_roundtrip():
    from protocol import ControlCommand, ControlFrame, pack_control, unpack_control

    frame = ControlFrame(ControlCommand.START, 0x12345678)
    assert unpack_control(pack_control(frame)) == frame
