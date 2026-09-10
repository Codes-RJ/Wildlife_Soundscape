"""
Binary communication protocol.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

This module defines the binary wire contract shared by:

    ESP32 Node 1 — Master
    ESP32 Node 2 — Slave
    ESP32 Node 3 — Slave
            ↓
        TCP / Wi-Fi
            ↓
       Laptop Receiver

Protocol principles
-------------------
1. All multi-byte fields are little-endian.

2. ESP32 packet structures must be packed with NO compiler padding.

3. ESP32 -> laptop packets use:
       fixed 40-byte header
       variable payload
       CRC32 over payload

4. Laptop -> ESP32 commands use:
       fixed 8-byte control frame

5. sampleIndex is the authoritative cross-node audio timeline.

6. sequence is a network diagnostic counter.

7. localMicros is diagnostic only and MUST NOT be used for TDOA
   synchronization.

8. Acoustic propagation delay is estimated later from PCM waveforms
   using GCC-PHAT/TDOA.

Current protocol
----------------
Protocol version: 4

Packet types:
    HELLO
    AUDIO
    ENVIRONMENT
    HEARTBEAT
    SYNC

Control commands:
    START
    STOP
    PING
"""

from __future__ import annotations

import enum
import math
import struct
import zlib

from dataclasses import dataclass
from typing import Final


# ======================================================================
# INTEGER LIMITS
# ======================================================================


UINT8_MAX: Final[int] = 0xFF

UINT16_MAX: Final[int] = 0xFFFF

UINT32_MAX: Final[int] = 0xFFFFFFFF

UINT64_MAX: Final[int] = 0xFFFFFFFFFFFFFFFF

INT16_MIN: Final[int] = -0x8000

INT16_MAX: Final[int] = 0x7FFF

INT32_MIN: Final[int] = -0x80000000

INT32_MAX: Final[int] = 0x7FFFFFFF


# ======================================================================
# ESP32 -> LAPTOP PROTOCOL
# ======================================================================


MAGIC: Final[int] = 0x574C5343

# ASCII interpretation:
#
#     57 4C 53 43
#      W  L  S  C
#
# Wildlife Soundscape Core

PROTOCOL_VERSION: Final[int] = 4


# ======================================================================
# PACKET HEADER
# ======================================================================
#
# Equivalent packed C layout:
#
# uint32_t magic;
#
# uint8_t  protocolVersion;
# uint8_t  nodeId;
# uint8_t  packetType;
# uint8_t  flags;
#
# uint32_t sequence;
# uint32_t sessionId;
#
# uint64_t sampleIndex;
#
# uint32_t localMicros;
# uint32_t i2sErrorCount;
# uint32_t payloadLength;
# uint32_t payloadCRC32;
#
# Total = 40 bytes
#
# IMPORTANT:
# ESP32 implementation must use a packed structure or serialize fields
# explicitly so compiler padding cannot alter this layout.
# ======================================================================


HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<IBBBBIIQIIII")

HEADER_SIZE: Final[int] = HEADER_STRUCT.size


# ======================================================================
# LAPTOP -> ESP32 CONTROL FRAME
# ======================================================================
#
# uint16_t magic;
# uint8_t  version;
# uint8_t  command;
# uint32_t sessionId;
#
# Total = 8 bytes
#
# Fixed framing avoids ambiguity on the TCP byte stream.
# ======================================================================


CONTROL_MAGIC: Final[int] = 0xC0DE

CONTROL_VERSION: Final[int] = 1

CONTROL_STRUCT: Final[struct.Struct] = struct.Struct("<HBBI")

CONTROL_SIZE: Final[int] = CONTROL_STRUCT.size


# ======================================================================
# PAYLOAD STRUCTURES
# ======================================================================


# ----------------------------------------------------------------------
# HELLO
# ----------------------------------------------------------------------
#
# uint32 sampleRate
# uint16 framesPerPacket
# uint8  bitsPerSample
# uint8  channels
# uint8  masterNode
# int16  syncToleranceSamples
# char   firmware[16]
#
# Total = 27 bytes
# ----------------------------------------------------------------------


HELLO_STRUCT: Final[struct.Struct] = struct.Struct("<IHBBBh16s")


# ----------------------------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------------------------
#
# float temperatureC
# float humidityPercent
# float pressureHpa
#
# Total = 12 bytes
# ----------------------------------------------------------------------


ENVIRONMENT_STRUCT: Final[struct.Struct] = struct.Struct("<fff")


# ----------------------------------------------------------------------
# SYNC
# ----------------------------------------------------------------------
#
# uint32 sessionId
# uint32 syncId
# uint64 sampleIndex
# uint32 localMicros
#
# Total = 20 bytes
# ----------------------------------------------------------------------


SYNC_STRUCT: Final[struct.Struct] = struct.Struct("<IIQI")


# ----------------------------------------------------------------------
# MASTER HEARTBEAT
# ----------------------------------------------------------------------
#
# uint32 uptimeSeconds
# int32  wifiRSSI
# uint32 freeHeap
# uint32 droppedAudioBlocks
# uint32 transmittedAudioBlocks
# uint32 i2sErrors
# uint16 audioQueueDepth
# uint8  streaming
# uint8  bmeAvailable
#
# Total = 28 bytes
# ----------------------------------------------------------------------


MASTER_HEARTBEAT_STRUCT: Final[struct.Struct] = struct.Struct("<IiIIIIHBB")


# ----------------------------------------------------------------------
# SLAVE HEARTBEAT
# ----------------------------------------------------------------------
#
# uint32 uptimeSeconds
# int32  wifiRSSI
# uint32 freeHeap
# uint32 droppedAudioBlocks
# uint32 transmittedAudioBlocks
# uint32 i2sErrors
# uint16 audioQueueDepth
# uint8  streaming
# uint8  syncReceived
# uint8  clockHealthy
#
# Total = 29 bytes
# ----------------------------------------------------------------------


SLAVE_HEARTBEAT_STRUCT: Final[struct.Struct] = struct.Struct("<IiIIIIHBBB")


# ======================================================================
# WIRE-LAYOUT ASSERTIONS
# ======================================================================
#
# These checks deliberately fail immediately if somebody later edits a
# struct format and accidentally changes the binary protocol.
# ======================================================================


if HEADER_SIZE != 40:
    raise RuntimeError(
        (f"Protocol layout error: HEADER_SIZE={HEADER_SIZE}, expected 40")
    )


if CONTROL_SIZE != 8:
    raise RuntimeError(
        (f"Protocol layout error: CONTROL_SIZE={CONTROL_SIZE}, expected 8")
    )


if HELLO_STRUCT.size != 27:
    raise RuntimeError(
        (f"Protocol layout error: HELLO size={HELLO_STRUCT.size}, expected 27")
    )


if ENVIRONMENT_STRUCT.size != 12:
    raise RuntimeError(
        (
            "Protocol layout error: "
            f"ENVIRONMENT size={ENVIRONMENT_STRUCT.size}, expected 12"
        )
    )


if SYNC_STRUCT.size != 20:
    raise RuntimeError(
        (f"Protocol layout error: SYNC size={SYNC_STRUCT.size}, expected 20")
    )


if MASTER_HEARTBEAT_STRUCT.size != 28:
    raise RuntimeError(
        (
            "Protocol layout error: master HEARTBEAT "
            f"size={MASTER_HEARTBEAT_STRUCT.size}, expected 28"
        )
    )


if SLAVE_HEARTBEAT_STRUCT.size != 29:
    raise RuntimeError(
        (
            "Protocol layout error: slave HEARTBEAT "
            f"size={SLAVE_HEARTBEAT_STRUCT.size}, expected 29"
        )
    )


# ======================================================================
# EXCEPTIONS
# ======================================================================


class ProtocolError(Exception):
    """
    Base exception for binary protocol failures.
    """


class InvalidHeader(ProtocolError):
    """
    Packet header is malformed or incompatible.
    """


class CRCMismatch(ProtocolError):
    """
    Payload CRC32 does not match packet header.
    """


class InvalidControlFrame(ProtocolError):
    """
    Laptop -> ESP32 control frame is malformed.
    """


# ======================================================================
# FLAGS
# ======================================================================


class PacketFlags(enum.IntFlag):
    """
    ESP32 runtime health flags carried in every packet header.
    """

    NONE = 0x00

    CLIPPED = 0x01

    CLOCK_FAULT = 0x02

    QUEUE_CONGESTED = 0x04


# ======================================================================
# PACKET TYPES
# ======================================================================


class PacketType(enum.IntEnum):
    """
    ESP32 -> laptop packet type.
    """

    HELLO = 1

    AUDIO = 2

    ENVIRONMENT = 3

    HEARTBEAT = 4

    SYNC = 5


# ======================================================================
# CONTROL COMMANDS
# ======================================================================


class ControlCommand(enum.IntEnum):
    """
    Laptop -> ESP32 command.
    """

    START = 0xA1

    STOP = 0xA2

    PING = 0xA3


# ======================================================================
# DATA MODELS
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PacketHeader:
    """
    Decoded 40-byte packet header.
    """

    magic: int

    protocol_version: int

    node_id: int

    packet_type: PacketType

    flags: int

    sequence: int

    session_id: int

    sample_index: int

    local_micros: int

    i2s_error_count: int

    payload_length: int

    payload_crc32: int


@dataclass(
    frozen=True,
    slots=True,
)
class Packet:
    """
    Complete ESP32 -> laptop packet.
    """

    header: PacketHeader

    payload: bytes


@dataclass(
    frozen=True,
    slots=True,
)
class ControlFrame:
    """
    Fixed laptop -> ESP32 control frame.
    """

    command: ControlCommand

    session_id: int = 0


@dataclass(
    frozen=True,
    slots=True,
)
class HelloPayload:
    """
    Node capabilities / firmware identity.
    """

    sample_rate: int

    frames_per_packet: int

    bits_per_sample: int

    channels: int

    master_node: bool

    sync_tolerance_samples: int

    firmware: str


@dataclass(
    frozen=True,
    slots=True,
)
class EnvironmentPayload:
    """
    BME280 environmental telemetry.
    """

    temperature_c: float

    humidity_percent: float

    pressure_hpa: float


@dataclass(
    frozen=True,
    slots=True,
)
class SyncPayload:
    """
    Synchronization marker diagnostic.

    sample_index
        Shared-clock sampleIndex associated with this marker.

    local_micros
        ESP32-local diagnostic timestamp only.

    The SYNC marker is NOT the primary TDOA clock.
    """

    session_id: int

    sync_id: int

    sample_index: int

    local_micros: int


@dataclass(
    frozen=True,
    slots=True,
)
class HeartbeatPayload:
    """
    Common representation of master/slave heartbeat data.
    """

    uptime_seconds: int

    wifi_rssi: int

    free_heap: int

    dropped_audio_blocks: int

    transmitted_audio_blocks: int

    i2s_errors: int

    audio_queue_depth: int

    streaming: bool

    # Master-specific
    bme_available: bool | None = None

    # Slave-specific
    sync_received: bool | None = None

    clock_healthy: bool | None = None


# ======================================================================
# RANGE VALIDATION HELPERS
# ======================================================================


def _uint8(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate an unsigned 8-bit value.
    """

    result = int(value)

    if not (0 <= result <= UINT8_MAX):
        raise ValueError(f"{name} must fit uint8")

    return result


def _uint16(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate an unsigned 16-bit value.
    """

    result = int(value)

    if not (0 <= result <= UINT16_MAX):
        raise ValueError(f"{name} must fit uint16")

    return result


def _uint32(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate an unsigned 32-bit value.
    """

    result = int(value)

    if not (0 <= result <= UINT32_MAX):
        raise ValueError(f"{name} must fit uint32")

    return result


def _uint64(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate an unsigned 64-bit value.
    """

    result = int(value)

    if not (0 <= result <= UINT64_MAX):
        raise ValueError(f"{name} must fit uint64")

    return result


def _int16(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a signed 16-bit value.
    """

    result = int(value)

    if not (INT16_MIN <= result <= INT16_MAX):
        raise ValueError(f"{name} must fit int16")

    return result


def _int32(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a signed 32-bit value.
    """

    result = int(value)

    if not (INT32_MIN <= result <= INT32_MAX):
        raise ValueError(f"{name} must fit int32")

    return result


def _validate_boolean_byte(
    value: int,
    *,
    name: str,
) -> bool:
    """
    Decode a strict uint8 boolean.

    Only 0 and 1 are accepted.
    """

    value = int(value)

    if value not in {
        0,
        1,
    }:
        raise ProtocolError((f"{name} boolean byte must be 0 or 1, got {value}"))

    return bool(value)


def _validate_exact_payload_size(
    payload: bytes,
    *,
    expected: int,
    name: str,
) -> None:
    """
    Validate a fixed-size payload.
    """

    actual = len(payload)

    if actual != expected:
        raise ProtocolError((f"{name} size {actual} != {expected}"))


def _validate_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require a finite floating-point telemetry value.
    """

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


# ======================================================================
# CRC32
# ======================================================================


def crc32(
    data: (bytes | bytearray | memoryview),
) -> int:
    """
    Calculate protocol CRC32.

    Result is always represented as unsigned uint32.
    """

    return zlib.crc32(data) & UINT32_MAX


# ======================================================================
# HEADER — UNPACK
# ======================================================================


def unpack_header(
    data: bytes,
) -> PacketHeader:
    """
    Decode and validate one 40-byte ESP32 packet header.
    """

    if len(data) != HEADER_SIZE:
        raise InvalidHeader((f"header size {len(data)} != expected {HEADER_SIZE}"))

    values = HEADER_STRUCT.unpack(data)

    # ==============================================================
    # PACKET TYPE
    # ==============================================================

    try:
        packet_type = PacketType(values[3])

    except ValueError as exc:
        raise InvalidHeader((f"unknown packet type {values[3]}")) from exc

    # ==============================================================
    # HEADER OBJECT
    # ==============================================================

    header = PacketHeader(
        magic=values[0],
        protocol_version=values[1],
        node_id=values[2],
        packet_type=packet_type,
        flags=values[4],
        sequence=values[5],
        session_id=values[6],
        sample_index=values[7],
        local_micros=values[8],
        i2s_error_count=values[9],
        payload_length=values[10],
        payload_crc32=values[11],
    )

    # ==============================================================
    # MAGIC
    # ==============================================================

    if header.magic != MAGIC:
        raise InvalidHeader((f"bad packet magic 0x{header.magic:08X}"))

    # ==============================================================
    # PROTOCOL VERSION
    # ==============================================================

    if header.protocol_version != PROTOCOL_VERSION:
        raise InvalidHeader(
            (
                "protocol version "
                f"{header.protocol_version} "
                f"!= expected {PROTOCOL_VERSION}"
            )
        )

    # ==============================================================
    # NODE ID
    # ==============================================================

    if not (1 <= header.node_id <= UINT8_MAX):
        raise InvalidHeader((f"invalid node id {header.node_id}"))

    return header


# ======================================================================
# HEADER — PACK
# ======================================================================


def pack_header(
    header: PacketHeader,
) -> bytes:
    """
    Encode one packet header.

    This performs strict range validation instead of silently truncating
    invalid values.
    """

    magic = _uint32(
        header.magic,
        name="magic",
    )

    version = _uint8(
        header.protocol_version,
        name="protocol_version",
    )

    node_id = _uint8(
        header.node_id,
        name="node_id",
    )

    flags = _uint8(
        header.flags,
        name="flags",
    )

    sequence = _uint32(
        header.sequence,
        name="sequence",
    )

    session_id = _uint32(
        header.session_id,
        name="session_id",
    )

    sample_index = _uint64(
        header.sample_index,
        name="sample_index",
    )

    local_micros = _uint32(
        header.local_micros,
        name="local_micros",
    )

    i2s_error_count = _uint32(
        header.i2s_error_count,
        name="i2s_error_count",
    )

    payload_length = _uint32(
        header.payload_length,
        name="payload_length",
    )

    payload_crc32 = _uint32(
        header.payload_crc32,
        name="payload_crc32",
    )

    if magic != MAGIC:
        raise ValueError((f"header magic must equal 0x{MAGIC:08X}"))

    if version != PROTOCOL_VERSION:
        raise ValueError((f"header protocol_version must equal {PROTOCOL_VERSION}"))

    if node_id == 0:
        raise ValueError("node_id cannot be zero")

    try:
        packet_type = PacketType(int(header.packet_type))

    except ValueError as exc:
        raise ValueError((f"invalid packet_type {header.packet_type}")) from exc

    return HEADER_STRUCT.pack(
        magic,
        version,
        node_id,
        int(packet_type),
        flags,
        sequence,
        session_id,
        sample_index,
        local_micros,
        i2s_error_count,
        payload_length,
        payload_crc32,
    )


# ======================================================================
# COMPLETE PACKET BUILDER
# ======================================================================


def build_packet(
    *,
    node_id: int,
    packet_type: PacketType,
    sequence: int,
    session_id: int,
    sample_index: int,
    payload: (bytes | bytearray | memoryview) = b"",
    local_micros: int = 0,
    i2s_error_count: int = 0,
    flags: int = 0,
) -> bytes:
    """
    Build one complete ESP32-style packet.

    Primarily useful for:
        simulator
        protocol tests
        offline packet generation
    """

    payload_bytes = bytes(payload)

    payload_crc = crc32(payload_bytes) if payload_bytes else 0

    header = PacketHeader(
        magic=MAGIC,
        protocol_version=PROTOCOL_VERSION,
        node_id=node_id,
        packet_type=packet_type,
        flags=flags,
        sequence=sequence,
        session_id=session_id,
        sample_index=sample_index,
        local_micros=local_micros,
        i2s_error_count=i2s_error_count,
        payload_length=len(payload_bytes),
        payload_crc32=payload_crc,
    )

    return pack_header(header) + payload_bytes


# ======================================================================
# PAYLOAD CRC VERIFICATION
# ======================================================================


def verify_payload_crc(
    header: PacketHeader,
    payload: bytes,
) -> None:
    """
    Verify payload length and CRC32.
    """

    actual_length = len(payload)

    if actual_length != header.payload_length:
        raise ProtocolError(
            (f"payload size {actual_length} != header length {header.payload_length}")
        )

    expected_crc = crc32(payload) if payload else 0

    if expected_crc != header.payload_crc32:
        raise CRCMismatch(
            (
                "CRC mismatch "
                f"node={header.node_id} "
                f"seq={header.sequence}: "
                f"header=0x{header.payload_crc32:08X}, "
                f"calculated=0x{expected_crc:08X}"
            )
        )


# ======================================================================
# CONTROL — PACK
# ======================================================================


def pack_control(
    frame: ControlFrame,
) -> bytes:
    """
    Encode one 8-byte laptop -> ESP32 control frame.
    """

    try:
        command = ControlCommand(int(frame.command))

    except ValueError as exc:
        raise InvalidControlFrame((f"unknown control command {frame.command}")) from exc

    session_id = _uint32(
        frame.session_id,
        name="session_id",
    )

    return CONTROL_STRUCT.pack(
        CONTROL_MAGIC,
        CONTROL_VERSION,
        int(command),
        session_id,
    )


# ======================================================================
# CONTROL — UNPACK
# ======================================================================


def unpack_control(
    data: bytes,
) -> ControlFrame:
    """
    Decode and validate one laptop -> ESP32 control frame.
    """

    if len(data) != CONTROL_SIZE:
        raise InvalidControlFrame(
            (f"control size {len(data)} != expected {CONTROL_SIZE}")
        )

    (
        magic,
        version,
        command_raw,
        session_id,
    ) = CONTROL_STRUCT.unpack(data)

    if magic != CONTROL_MAGIC:
        raise InvalidControlFrame((f"bad control magic 0x{magic:04X}"))

    if version != CONTROL_VERSION:
        raise InvalidControlFrame(
            (f"control version {version} != expected {CONTROL_VERSION}")
        )

    try:
        command = ControlCommand(command_raw)

    except ValueError as exc:
        raise InvalidControlFrame(
            (f"unknown control command 0x{command_raw:02X}")
        ) from exc

    return ControlFrame(
        command=command,
        session_id=session_id,
    )


# ======================================================================
# HELLO — PARSE
# ======================================================================


def parse_hello(
    payload: bytes,
) -> HelloPayload:
    """
    Decode one HELLO payload.
    """

    _validate_exact_payload_size(
        payload,
        expected=HELLO_STRUCT.size,
        name="HELLO",
    )

    (
        sample_rate,
        frames,
        bits,
        channels,
        master_raw,
        tolerance,
        firmware_raw,
    ) = HELLO_STRUCT.unpack(payload)

    master_node = _validate_boolean_byte(
        master_raw,
        name="HELLO master_node",
    )

    if sample_rate <= 0:
        raise ProtocolError(("HELLO sample_rate must be greater than 0"))

    if frames <= 0:
        raise ProtocolError(("HELLO frames_per_packet must be greater than 0"))

    if bits <= 0:
        raise ProtocolError(("HELLO bits_per_sample must be greater than 0"))

    if channels <= 0:
        raise ProtocolError(("HELLO channels must be greater than 0"))

    if tolerance < 0:
        raise ProtocolError(("HELLO sync_tolerance_samples cannot be negative"))

    firmware_text = firmware_raw.split(
        b"\0",
        1,
    )[0].decode(
        "ascii",
        errors="replace",
    )

    return HelloPayload(
        sample_rate=sample_rate,
        frames_per_packet=frames,
        bits_per_sample=bits,
        channels=channels,
        master_node=master_node,
        sync_tolerance_samples=tolerance,
        firmware=firmware_text,
    )


# ======================================================================
# HELLO — PACK
# ======================================================================


def pack_hello(
    value: HelloPayload,
) -> bytes:
    """
    Encode one HELLO payload.
    """

    sample_rate = _uint32(
        value.sample_rate,
        name="sample_rate",
    )

    frames = _uint16(
        value.frames_per_packet,
        name="frames_per_packet",
    )

    bits = _uint8(
        value.bits_per_sample,
        name="bits_per_sample",
    )

    channels = _uint8(
        value.channels,
        name="channels",
    )

    tolerance = _int16(
        value.sync_tolerance_samples,
        name="sync_tolerance_samples",
    )

    if sample_rate == 0:
        raise ValueError("sample_rate cannot be zero")

    if frames == 0:
        raise ValueError("frames_per_packet cannot be zero")

    if bits == 0:
        raise ValueError("bits_per_sample cannot be zero")

    if channels == 0:
        raise ValueError("channels cannot be zero")

    if tolerance < 0:
        raise ValueError(("sync_tolerance_samples cannot be negative"))

    firmware = str(value.firmware).encode(
        "ascii",
        errors="replace",
    )[:15]

    firmware = firmware + (b"\0" * (16 - len(firmware)))

    return HELLO_STRUCT.pack(
        sample_rate,
        frames,
        bits,
        channels,
        int(bool(value.master_node)),
        tolerance,
        firmware,
    )


# ======================================================================
# ENVIRONMENT — PARSE
# ======================================================================


def parse_environment(
    payload: bytes,
) -> EnvironmentPayload:
    """
    Decode one BME280 telemetry payload.
    """

    _validate_exact_payload_size(
        payload,
        expected=ENVIRONMENT_STRUCT.size,
        name="ENVIRONMENT",
    )

    (
        temperature_c,
        humidity_percent,
        pressure_hpa,
    ) = ENVIRONMENT_STRUCT.unpack(payload)

    if not all(
        math.isfinite(value)
        for value in (
            temperature_c,
            humidity_percent,
            pressure_hpa,
        )
    ):
        raise ProtocolError(("ENVIRONMENT payload contains non-finite values"))

    return EnvironmentPayload(
        temperature_c=float(temperature_c),
        humidity_percent=float(humidity_percent),
        pressure_hpa=float(pressure_hpa),
    )


# ======================================================================
# ENVIRONMENT — PACK
# ======================================================================


def pack_environment(
    value: EnvironmentPayload,
) -> bytes:
    """
    Encode one BME280 telemetry payload.
    """

    temperature = _validate_finite_float(
        value.temperature_c,
        name="temperature_c",
    )

    humidity = _validate_finite_float(
        value.humidity_percent,
        name="humidity_percent",
    )

    pressure = _validate_finite_float(
        value.pressure_hpa,
        name="pressure_hpa",
    )

    return ENVIRONMENT_STRUCT.pack(
        temperature,
        humidity,
        pressure,
    )


# ======================================================================
# SYNC — PARSE
# ======================================================================


def parse_sync(
    payload: bytes,
) -> SyncPayload:
    """
    Decode one SYNC marker payload.
    """

    _validate_exact_payload_size(
        payload,
        expected=SYNC_STRUCT.size,
        name="SYNC",
    )

    (
        session_id,
        sync_id,
        sample_index,
        local_micros,
    ) = SYNC_STRUCT.unpack(payload)

    return SyncPayload(
        session_id=session_id,
        sync_id=sync_id,
        sample_index=sample_index,
        local_micros=local_micros,
    )


# ======================================================================
# SYNC — PACK
# ======================================================================


def pack_sync(
    value: SyncPayload,
) -> bytes:
    """
    Encode one SYNC marker payload.
    """

    return SYNC_STRUCT.pack(
        _uint32(
            value.session_id,
            name="session_id",
        ),
        _uint32(
            value.sync_id,
            name="sync_id",
        ),
        _uint64(
            value.sample_index,
            name="sample_index",
        ),
        _uint32(
            value.local_micros,
            name="local_micros",
        ),
    )


# ======================================================================
# HEARTBEAT COMMON VALIDATION
# ======================================================================


def _validated_heartbeat_common(
    value: HeartbeatPayload,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """
    Validate common heartbeat fields before packing.
    """

    return (
        _uint32(
            value.uptime_seconds,
            name="uptime_seconds",
        ),
        _int32(
            value.wifi_rssi,
            name="wifi_rssi",
        ),
        _uint32(
            value.free_heap,
            name="free_heap",
        ),
        _uint32(
            value.dropped_audio_blocks,
            name="dropped_audio_blocks",
        ),
        _uint32(
            value.transmitted_audio_blocks,
            name="transmitted_audio_blocks",
        ),
        _uint32(
            value.i2s_errors,
            name="i2s_errors",
        ),
        _uint16(
            value.audio_queue_depth,
            name="audio_queue_depth",
        ),
        int(bool(value.streaming)),
    )


# ======================================================================
# HEARTBEAT — PARSE
# ======================================================================


def parse_heartbeat(
    payload: bytes,
    *,
    master_node: bool,
) -> HeartbeatPayload:
    """
    Decode role-specific heartbeat telemetry.
    """

    # ==============================================================
    # MASTER
    # ==============================================================

    if master_node:
        _validate_exact_payload_size(
            payload,
            expected=MASTER_HEARTBEAT_STRUCT.size,
            name="master HEARTBEAT",
        )

        values = MASTER_HEARTBEAT_STRUCT.unpack(payload)

        streaming = _validate_boolean_byte(
            values[7],
            name="HEARTBEAT streaming",
        )

        bme_available = _validate_boolean_byte(
            values[8],
            name="HEARTBEAT bme_available",
        )

        return HeartbeatPayload(
            uptime_seconds=values[0],
            wifi_rssi=values[1],
            free_heap=values[2],
            dropped_audio_blocks=values[3],
            transmitted_audio_blocks=values[4],
            i2s_errors=values[5],
            audio_queue_depth=values[6],
            streaming=streaming,
            bme_available=bme_available,
        )

    # ==============================================================
    # SLAVE
    # ==============================================================

    _validate_exact_payload_size(
        payload,
        expected=SLAVE_HEARTBEAT_STRUCT.size,
        name="slave HEARTBEAT",
    )

    values = SLAVE_HEARTBEAT_STRUCT.unpack(payload)

    streaming = _validate_boolean_byte(
        values[7],
        name="HEARTBEAT streaming",
    )

    sync_received = _validate_boolean_byte(
        values[8],
        name="HEARTBEAT sync_received",
    )

    clock_healthy = _validate_boolean_byte(
        values[9],
        name="HEARTBEAT clock_healthy",
    )

    return HeartbeatPayload(
        uptime_seconds=values[0],
        wifi_rssi=values[1],
        free_heap=values[2],
        dropped_audio_blocks=values[3],
        transmitted_audio_blocks=values[4],
        i2s_errors=values[5],
        audio_queue_depth=values[6],
        streaming=streaming,
        sync_received=sync_received,
        clock_healthy=clock_healthy,
    )


# ======================================================================
# MASTER HEARTBEAT — PACK
# ======================================================================


def pack_master_heartbeat(
    value: HeartbeatPayload,
) -> bytes:
    """
    Encode one Node 1/master heartbeat.
    """

    (
        uptime,
        rssi,
        free_heap,
        dropped,
        transmitted,
        i2s_errors,
        queue_depth,
        streaming,
    ) = _validated_heartbeat_common(value)

    return MASTER_HEARTBEAT_STRUCT.pack(
        uptime,
        rssi,
        free_heap,
        dropped,
        transmitted,
        i2s_errors,
        queue_depth,
        streaming,
        int(bool(value.bme_available)),
    )


# ======================================================================
# SLAVE HEARTBEAT — PACK
# ======================================================================


def pack_slave_heartbeat(
    value: HeartbeatPayload,
) -> bytes:
    """
    Encode one Node 2/3 slave heartbeat.
    """

    (
        uptime,
        rssi,
        free_heap,
        dropped,
        transmitted,
        i2s_errors,
        queue_depth,
        streaming,
    ) = _validated_heartbeat_common(value)

    return SLAVE_HEARTBEAT_STRUCT.pack(
        uptime,
        rssi,
        free_heap,
        dropped,
        transmitted,
        i2s_errors,
        queue_depth,
        streaming,
        int(bool(value.sync_received)),
        int(bool(value.clock_healthy)),
    )
