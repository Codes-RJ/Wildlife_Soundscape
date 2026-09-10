# Development, data and release guide

## Repository shape

All Python code is under `src/wildlife_soundscape/`. Domain packages are
`core`, `acquisition`, `pipeline`, `storage`, `runtime`, `dsp`,
`localization`, `calibration`, `classification`, `analytics`, `dashboard` and
`tools`. Firmware is under `firmware/`; runtime data is under `data/`.

## Development checks

```powershell
python -m compileall -q src
python -m ruff check src tests
python -m mypy src tests
python -m pytest -q --cov=wildlife_soundscape --cov-report=term-missing
python -m pip_audit
```

`scripts\setup_windows.bat` creates the Python 3.13 environment, installs the
package and runs the release checks. `scripts\validate_windows.bat` is the
broader Windows release gate. Keep the simulator as a synthetic regression
fixture; passing tests do not establish field or species accuracy.

## Dataset policy

The native dataset contract is Protocol v4, 48 kHz mono PCM16, identical
three-node frame counts, measured non-collinear coordinates, timezone-aware
timestamps, telemetry, recording group IDs, annotation status and known source
positions for localization trials. Use continuous recordings for soundscape
indices and event clips for detection/classification. Add silence, machinery,
wind, rain, overlap and hard negatives.

Split by whole site/session/day/source groups, never by nearby clips. Record
taxonomy name/version, source URL/ID, license, checksum and conversion details.
Keep verified test labels separate from model predictions.

Suitable public candidates are listed in the temporary review document and
include BirdCLEF+ 2025 (broad wildlife taxa and soundscapes), ESC-50 (small
environmental WAV benchmark), NatureLM-audio, the CSIRO Wildlife Sound Archive,
and tropical anuran/bird AudioMoth recordings. Check current terms before
redistributing audio.

## Current status and roadmap

The software prototype, simulator, SQLite/WAV evidence path, dashboard and
analytics are implemented. The default classifier produces broad acoustic
classes only; optional BirdNET requires model assets and taxonomy mapping. The
current synthetic session is not a training or field-validation dataset.

Next work should proceed in this order:

1. Import a small licensed real-audio dataset with provenance and ground truth.
2. Add prediction/ground-truth separation and confusion-matrix, F1, rejection
   and calibration reports.
3. Validate timing and localization using known source positions inside and
   outside the array.
4. Benchmark a trained backend against the locked real-audio test split.
5. Harden process supervision, disk recovery, observability and deployment
   security after the measured baseline is trustworthy.

## Historical release notes

Protocol v4 and firmware v1.4.0 remain the current compatibility baseline.
Implemented work includes packet health flags, environmental speed-of-sound
support, adaptive event detection, synchronized WAV evidence, SQLite sessions,
classification adapters, soundscape indices and dashboard analysis. Preserve
protocol and database compatibility as these areas evolve.
