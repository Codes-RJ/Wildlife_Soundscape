# Repository Roadmap

## V1-to-V2 migration — completed

- V2 is promoted into the canonical repository root.
- Python implementation code remains under `src/wildlife_soundscape`.
- Configuration is split into focused `core/settings` modules behind the
  existing `core.config` facade.
- SQLite schema, migration, query, and provenance responsibilities are split
  behind the existing `EventDatabase` facade.
- Acquisition processing is bounded and records overload metrics.
- Experiment manifests, dataset validation, review tooling, deployment
  overrides, and a 60% coverage gate are present.
- Firmware compilation has a pinned CI workflow.
- V1's concise protocol, script guide, release notes, and dated audit remain as
  historical and compatibility references.

## Priority 0 — protect and verify existing evidence

- Back up the existing V1 SQLite database and WAV directories before running a
  migrated production session.
- Exercise the database migration on a copy and verify session, event,
  classification, localization, telemetry, and audio references afterward.
- Keep synthetic/demo sessions clearly separated from field observations.

## Dashboard workflow — completed

- Primary navigation is reduced to Monitor, Sessions, and Setup.
- Session analysis defaults to the newest completed acquisition and uses focused
  Overview, Events, Soundscape, Spatial, and Export tabs.
- Event review exposes model identity, confidence, runner-up, margin, reasons,
  synchronized microphone clips, adjustable continuous-audio context, and full
  session WAVs when available.
- Human reviews are stored and exported separately from model predictions.
- Old unclosed sessions are displayed as interrupted and can be explicitly
  closed without deleting their evidence.
- A balanced six-class synthetic fixture can be generated additively from the
  Setup page; live heuristic results remain clearly distinct.
- Storage usage, deployment settings, classifier fallback, and physical
  preflight limitations are visible under Setup.

## Priority 1 — physical array validation

- Compile all three sketches using the pinned Arduino-ESP32 toolchain.
- Bench-test the new slave `i2s_std` implementation with shared BCLK and WS.
- Test repeated START/STOP, reconnects, missing-clock timeouts, packet loss,
  overload, power loss, and long-duration capture.
- Measure microphone geometry and timing bias, then run held-out inside- and
  outside-array localization trials.
- Report success rate, median/RMSE/95th-percentile error, signed bias, SNR, and
  failed trials.

## Priority 2 — defensible classification

- Import a small licensed real-audio corpus with source, license, taxonomy,
  checksum, site/session grouping, and reviewed ground truth.
- Keep verified labels separate from model predictions.
- Lock group-level train, validation, and test splits before model tuning.
- Validate BirdNET and ensemble behavior with confusion matrices, macro-F1,
  rejection performance, and confidence calibration.
- Do not claim species coverage that has not passed the locked test set.

## Priority 3 — field hardening

- Add stronger service supervision, storage-capacity monitoring, and recovery
  drills after the measured scientific baseline is stable.
- Review deployment security, credential handling, network isolation, enclosure,
  power, and weatherproofing requirements.
- Review all educational species examples against cited biological sources
  before expanding species-specific dashboard content.

## Ongoing quality rules

- Keep Ruff and mypy at zero issues.
- Maintain or raise the 60% coverage gate with meaningful high-risk tests.
- Preserve Protocol-v4 and database compatibility deliberately.
- Treat synthetic tests as software validation, not field-accuracy evidence.
