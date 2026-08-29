# Core v1.1 Release Notes

Protocol remains **packet protocol v4 / control protocol v1**. Firmware baseline is now **v1.4.0**.

## Added

- packet health flags: clipping, clock fault, queue congestion
- BME280-aware dynamic speed-of-sound approximation
- configurable 200 Hz–12 kHz Butterworth localization pre-filter
- adaptive multi-node acoustic event detection
- 0.5 s pre-trigger + 0.75 s post-trigger event preservation
- SQLite `sessions`, `telemetry`, and `events` persistence
- synchronized per-node event WAV slices
- environmental telemetry association with detected events
- automatic event localization before database insertion
- `events` CLI command
- integration event test

## Corrected

- `integration_smoke.py` is guarded by `if __name__ == "__main__"`
- standalone integration tests add the repository root to `sys.path`
- smoke test waits for an actually aligned block instead of assuming packet arrival timing
- fixed an accidental duplicate `bool ensureWiFi()` declaration in the master sketch

## Validation

- `pytest -q`: **14 passed**
- `integration_smoke.py`: PASS
- `integration_localization.py`: PASS
- `integration_events.py`: PASS
- ideal simulated source `(0.45, 0.35)` m recovered within a few millimetres
- simulated later event stored with BME280 telemetry and environment-adjusted sound speed

## Deferred intentionally

Chan WLS, GDOP/error ellipses, SCOT variants, mDNS discovery, periodic re-sync, unified firmware, classification, and dashboard remain outside this release.

ESP32 sketches remain hardware-unvalidated until the actual DevKit V1 + INMP441 + BME280 setup is available.
