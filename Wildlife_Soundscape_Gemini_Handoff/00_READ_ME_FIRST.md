# Wildlife Soundscape — Gemini Implementation Handoff

This folder is a controlled implementation brief for upgrading the existing
Wildlife Soundscape Mapping & Behavior Analysis System without destabilizing
the already-completed baseline.

## Use this pack with the CURRENT local repository

Do **not** treat this folder as a replacement repository. It contains:

- the implementation order;
- exact architectural constraints;
- file-change guidance;
- validation gates;
- two known-good corrected classification files;
- research-method notes for the new features.

Gemini should inspect the current local repository first, then apply the tasks
in the order defined here.

## Hard boundary

The current baseline must remain:

- 3 × ESP32-WROOM-32 / DevKit V1
- Node 1 I2S MASTER
- Nodes 2 and 3 I2S SLAVES
- INMP441 × 3
- shared BCLK/WS for sample-rate synchronization
- GPIO27 as session/start marker only
- 48 kHz acquisition
- protocol v4 unchanged
- 2D localization in V1
- laptop-side event detection, DSP, classification, localization, DB and UI

Do not add MicroSD, batteries, solar, LoRa, GPS, a fourth node, 3D localization,
or alert webhooks in this implementation cycle.

## Primary goal

Upgrade the research/software layer in a controlled way:

1. close the BirdNET taxonomy/context issue;
2. add continuous ecoacoustic indices;
3. make GCC-PHAT weighting experimentally configurable;
4. add GeoJSON export with a correct local-to-geographic transform;
5. extend persistence/dashboard only as required;
6. add tests and benchmark tooling.

## Most important instruction to Gemini

**Do not rewrite working architecture for style.**

Only change code required for the tasks in this handoff. Preserve current public
interfaces where possible. If an interface must change, update all call sites
and tests in the same phase.
