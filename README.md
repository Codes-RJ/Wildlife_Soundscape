# Wildlife Soundscape Mapping & Behavior Analysis System — Core v1.1

Lab-scale 3-node synchronized acoustic monitoring prototype.

## Frozen hardware target

- **3 × ESP32 DevKit V1 / ESP32-WROOM-32**
- **3 × INMP441 I2S digital MEMS microphone modules**
- **1 × BME280 temperature/humidity/barometric-pressure module** on Node 1
- Laptop on the same Wi-Fi network
- Shared wired **BCLK + WS + SYNC + GND** from Node 1 to Nodes 2 and 3

### Pin map

| Signal | Node 1 Master | Node 2 Slave | Node 3 Slave |
|---|---|---|---|
| BCLK | GPIO26 OUT | GPIO26 IN | GPIO26 IN |
| WS/LRCLK | GPIO25 OUT | GPIO25 IN | GPIO25 IN |
| Local INMP441 SD | GPIO33 IN | GPIO33 IN | GPIO33 IN |
| SYNC | GPIO27 OUT | GPIO27 IN | GPIO27 IN |
| Common ground | GND | GND | GND |
| BME280 SDA | GPIO21 | — | — |
| BME280 SCL | GPIO22 | — | — |

All INMP441 `L/R` pins are tied to GND, so firmware extracts the LEFT I2S slot.

## Repository

```text
Wildlife_Soundscape_Core/
├── Node_1_Master.ino
├── Node_2_Slave.ino
├── Node_3_Slave.ino
├── config.py
├── environment.py
├── event_detector.py
├── event_pipeline.py
├── database.py
├── main.py
├── models.py
├── node.py
├── protocol.py
├── server.py
├── simulator.py
├── stream_manager.py
├── localization/
│   ├── __init__.py
│   ├── filtering.py
│   ├── gcc_phat.py
│   ├── tdoa.py
│   ├── solver.py
│   └── engine.py
├── tests/
│   ├── test_protocol.py
│   ├── test_node.py
│   ├── test_stream_manager.py
│   ├── test_localization.py
│   ├── test_environment.py
│   ├── test_filtering.py
│   ├── test_event_detector.py
│   ├── test_database.py
│   ├── integration_smoke.py
│   ├── integration_localization.py
│   └── integration_events.py
├── data/
│   ├── recordings/
│   ├── events/
│   ├── database/
│   └── exports/
├── PROTOCOL.md
├── RELEASE_NOTES.md
├── requirements.txt
└── .gitignore
```

## Python setup (Windows / VS Code)

Recommended Python: **3.13**.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Current unit-test result at packaging: **14 passed**.

## Test without hardware

Terminal 1:

```powershell
python main.py
```

Terminal 2:

```powershell
python simulator.py
```

Then in Terminal 1:

```text
start
status
locate
events
stop
quit
```

The simulator uses the same binary protocol and generates three delayed acoustic streams from a known virtual source.

Default simulator geometry:

- Node 1 `(0.0, 0.0)` m
- Node 2 `(0.5, 0.8660)` m
- Node 3 `(1.0, 0.0)` m
- Source `(0.45, 0.35)` m

## Core data flow

```text
3 ESP32 / simulator streams
        ↓
protocol + CRC + diagnostics
        ↓
sample-indexed PCM buffers
        ↓
adaptive multi-node acoustic event detector
        ↓
pre/post padded synchronized event window
        ↓
configurable bandpass + GCC-PHAT
        ↓
TDOA + 2-D localization
        ↓
BME280 environmental association
        ↓
SQLite metadata + per-node event WAV files
```

## Audio/event configuration

- acquisition: 48 kHz
- transport: signed mono PCM16 little-endian
- block size: 1024 samples ≈ 21.33 ms
- laptop rolling buffer: 20 s
- event minimum active duration: 30 ms
- pre-trigger preservation: 0.50 s
- post-trigger preservation: 0.75 s
- multi-node trigger: at least 2 nodes
- maximum event duration: 12 s

The detector uses a rolling low-quantile background estimate, adaptive energy thresholds, and spectral flux. It is an **acoustic event detector**, not a speech/VAD-only detector.

## Localization upgrades in v1.1

- speed of sound can be calculated from nearest BME280 temperature/humidity/pressure telemetry
- fallback remains 343 m/s when telemetry is unavailable
- configurable 4th-order Butterworth pre-filter
- default localization band: 200 Hz–12 kHz, deliberately broader than a bird-only band
- GCC-PHAT remains the baseline estimator
- physically impossible pair delays are rejected
- `localMicros` is diagnostic only and never used for cross-node TDOA

## Persistence

SQLite database:

```text
data/database/events.db
```

Tables:

- `sessions`
- `telemetry`
- `events`

Event audio is **not stored as SQLite blobs**. Each event is written as synchronized PCM WAV slices:

```text
data/events/<session>/event_000001/
├── node_1.wav
├── node_2.wav
└── node_3.wav
```

## Packet health flags

Protocol v4 retains the same 40-byte header but now defines the existing `flags` byte:

- `0x01` — audio clipping detected
- `0x02` — slave clock fault latched
- `0x04` — audio queue ≥80% full

No low-supply flag is used because the current lab nodes are USB powered and have no voltage measurement circuit.

## Firmware reliability

Current firmware baseline: **v1.4.0 / protocol v4**.

Includes:

- laptop-owned common `sessionId`
- slave-first START ordering
- CRC32 and sequence/sample diagnostics
- exponential Wi-Fi/TCP reconnect
- FreeRTOS acquisition/network separation
- I2S error counters
- queue flush and I2S reset on STOP
- BME280 retry + chip-ID check
- clock-health monitoring on slaves
- health flags on outgoing packets

### Before flashing

Edit all three `.ino` files:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
IPAddress LAPTOP_IP(192, 168, 1, 100);
```

Laptop TCP server port is **5001**.

## Integration checks

```powershell
python tests/integration_smoke.py
python tests/integration_localization.py
python tests/integration_events.py
```

Validated in software at packaging:

- 3-node TCP/PCM integration: PASS
- ideal synthetic 2-D localization: approximately 1–3 mm error depending on test window
- automatic acoustic event detection: PASS
- SQLite event persistence: PASS
- environmental telemetry attached to later simulated events: PASS

These synthetic localization numbers validate the mathematics/software only. Real lab error will be substantially larger because of room reflections, microphone tolerance, geometry error, noise, and physical clock/signal integrity.

## Still deliberately deferred

- Chan WLS initializer
- GDOP / 95% uncertainty ellipse (should use real calibration noise statistics)
- advanced SCOT/regularized GCC variants
- mDNS/UDP discovery
- periodic sample-counter “re-sync”
- unified one-file ESP32 firmware
- wildlife classification / BirdNET
- final 1–2 page dashboard

The next major feature after physical/core validation is **DSP feature extraction + wildlife classification**.
