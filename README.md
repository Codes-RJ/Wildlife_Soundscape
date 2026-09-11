# Wildlife Soundscape

Wildlife Soundscape is a research prototype for synchronized acoustic
monitoring. Three ESP32 microphone nodes share an I2S sampling clock and stream
sample-indexed PCM audio to a Python receiver. The laptop detects events,
extracts acoustic features, estimates TDOA locations, classifies sound,
persists evidence, computes research indicators, and serves a Streamlit
dashboard.

The system is suitable for laboratory experiments and software demonstrations.
It has not yet completed the physical-array and real-model validation required
for field accuracy claims.

## Quick start

Python 3.13 is required. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
scripts\run_demo.bat
```

The demo rotates synthetic calls through 24 source positions, alternating
12 inside and 12 outside the microphone array, every five seconds. It exercises
detection, localization, and dashboard tables, but it is not wildlife audio.

For a populated historical soundscape, run
`python -m wildlife_soundscape.tools.populate_demo_session --events 144`.
This adds a separate synthetic session with 72 points inside and 72 outside the
configured triangle, balanced illustrative class labels, and audio previews.
Existing sessions are preserved. The dashboard uses a monochrome interface
with stable class colors reserved for charts.

## Local deployment configuration

Copy `config.example.json` to `config.local.json` and edit it for a live
deployment. This ignored file supports receiver host/port, measured microphone
coordinates, selected detector thresholds, BirdNET location context, and output
paths without committing site-specific settings. Set `WILDLIFE_CONFIG_PATH` to
use a differently named configuration file.

The demo starts the receiver, simulator, and dashboard. Individual entry points
are also available:

```powershell
wildlife-receiver
wildlife-simulator
wildlife-validate-dataset docs/dataset-manifest.example.json --skip-audio
streamlit run src/wildlife_soundscape/dashboard/app.py
```

The dashboard is organized into three areas:

- **Monitor** shows the latest session, detections, event audio, and locations.
- **Sessions** defaults to the newest completed session and groups event review,
  soundscape analysis, spatial analysis, and exports into tabs. Event review
  includes the short detector clip plus adjustable surrounding context from the
  full continuous WAV when one was recorded.
- **Setup** shows acquisition/classifier configuration, hardware preflight,
  interrupted-session recovery, storage use, and an additive balanced demo-data
  generator.

The pipeline simulator exercises real detection and the heuristic classifier,
so its natural output may legitimately contain mostly `bird` and `unknown`.
Use **Setup → Demo data** when you need a clearly marked six-class interface
demonstration; those illustrative labels are never presented as model results.

Use `scripts\run_live.bat` with configured ESP32 nodes. Set deployment
credentials and the receiver address in each firmware sketch without committing
real credentials.

## Architecture

```text
ESP32 nodes -> Protocol v4/TCP -> bounded ordered processing
            -> sample-indexed streams -> event detection
            -> localization + features + classification
            -> SQLite metadata + WAV evidence
            -> analytics + dashboard + portable exports
```

Core event metadata is committed before optional analysis and WAV storage.
Failures are recorded per processing stage. Each acquisition stores an immutable
experiment manifest containing effective settings and available software,
firmware, and model identities. Research and event exports include manifest
sidecars.

## Validation

Run the complete software release gate:

```powershell
scripts\validate_windows.bat
```

It compiles Python, builds distributions, lints, type-checks, runs tests with a
60% coverage floor, and audits dependencies. Firmware compilation is pinned in
CI to Arduino CLI 1.3.1, Arduino-ESP32 3.3.0, Adafruit BME280 2.3.0,
Adafruit Unified Sensor 1.1.15, and Adafruit BusIO 1.17.2.

Automated synthetic tests do not establish real-world localization or species
accuracy. Follow the [physical and model validation](docs/OPERATIONS.md)
before reporting experimental performance.

## Documentation

- [Advanced development pathway](docs/ADVANCED_PATHWAY.md)
- [Operations guide](docs/OPERATIONS.md)
- [Development, data and release guide](docs/DEVELOPMENT_DATA.md)
- [Technical system reference](docs/TECHNICAL_REFERENCE.md)
- [Firmware setup](firmware/README.md)
- [Dataset manifest example](docs/dataset-manifest.example.json)

The system reports acoustic detections and model-derived indicators. A sound
detection is not a verified species observation, and consecutive estimated
positions do not prove an individual animal trajectory.
