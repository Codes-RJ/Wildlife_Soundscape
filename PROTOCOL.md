# Wire Protocol v4

Canonical interface between ESP32 firmware v1.4.0 and the Python receiver.

## ESP32 → Laptop packet header

Packed little-endian, **40 bytes**.

| Field | Type | Purpose |
|---|---|---|
| magic | uint32 | `0x574C5343` (`WLSC`) |
| protocolVersion | uint8 | `4` |
| nodeId | uint8 | 1, 2 or 3 |
| packetType | uint8 | HELLO/AUDIO/ENVIRONMENT/HEARTBEAT/SYNC |
| flags | uint8 | packet health bitmask |
| sequence | uint32 | packet/network diagnostics |
| sessionId | uint32 | laptop-owned acquisition session |
| sampleIndex | uint64 | authoritative coarse PCM timeline |
| localMicros | uint32 | local diagnostic only; never subtract across nodes |
| i2sErrorCount | uint32 | cumulative I2S error count at capture |
| payloadLength | uint32 | bytes following header |
| payloadCRC32 | uint32 | CRC32 of payload, zero when empty |

### Flags

| Bit | Value | Meaning |
|---|---:|---|
| 0 | `0x01` | `CLIPPED`: at least one PCM16 sample reached the clipping threshold |
| 1 | `0x02` | `CLOCK_FAULT`: slave clock fault was detected/latched |
| 2 | `0x04` | `QUEUE_CONGESTED`: audio queue is at least 80% occupied |

Bits may be ORed together. Unknown future bits must be ignored by older readers.

## Laptop → ESP32 control frame

Fixed **8 bytes** (`<HBBI`).

| Field | Type | Value |
|---|---|---|
| magic | uint16 | `0xC0DE` |
| version | uint8 | `1` |
| command | uint8 | START=`0xA1`, STOP=`0xA2`, PING=`0xA3` |
| sessionId | uint32 | same laptop-generated ID sent to all nodes |

The 8-byte control frame is intentionally retained for the current lab MVP. Runtime DSP/filter configuration remains laptop-side and therefore does not need ESP32 control commands yet.

## Packet types

- `1 HELLO` — role and audio format
- `2 AUDIO` — mono PCM16 little-endian
- `3 ENVIRONMENT` — Node 1 BME280 temperature/humidity/pressure
- `4 HEARTBEAT` — health/diagnostics
- `5 SYNC` — synchronization/session diagnostic

## Timing rules

- Shared **BCLK + WS** define the common acquisition sample clock.
- GPIO27 SYNC is a session/start marker, not the TDOA clock.
- `sampleIndex` defines the coarse common PCM timeline.
- `localMicros` belongs to independent ESP32 CPU clocks and is diagnostic only.
- Acoustic TDOA comes from actual PCM correlation (GCC-PHAT).
- Missing sample regions remain gaps; the laptop never compresses the timeline.
- The nominal ±10-sample tolerance is a software verification/search window, not permission to silently alter edge samples.
