# Package layout

All Python implementation code is now under `src/wildlife_soundscape/`.
Imports must use the `wildlife_soundscape` namespace; root-level Python
compatibility modules are intentionally not retained.

```text
src/wildlife_soundscape/
├── core/           public configuration, focused settings, protocol, provenance
├── acquisition/    node state and buffered audio streams
├── pipeline/       event detection and completed-event processing
├── storage/        SQLite facade, schema, migrations, queries and provenance
├── runtime/        receiver server, bounded processing, interactive CLI, simulator
├── analytics/      temporal, environmental, spatial and soundscape analysis
├── calibration/    known-source TDOA calibration and benchmarks
├── classification/ heuristic, BirdNET and ensemble classification
├── dashboard/      Streamlit UI, preflight, review, plots, audio and data access
├── datasets/       native dataset validation, safe splitting and registry
├── dsp/            preprocessing and acoustic features
├── localization/   filtering, GCC-PHAT, TDOA and position solving
├── tools/          data exporters, demo data and GCC benchmark utility
├── cli.py          installed command entry points
└── __main__.py     `python -m wildlife_soundscape` receiver entry point
```

| Previous module path | Canonical import path |
|---|---|
| `config.py` | `wildlife_soundscape.core.config` facade and `core.settings.*` implementations |
| `models.py` | `wildlife_soundscape.core.models` |
| `protocol.py` | `wildlife_soundscape.core.protocol` |
| `environment.py` | `wildlife_soundscape.core.environment` |
| `node.py`, `stream_manager.py` | `wildlife_soundscape.acquisition.*` |
| `event_detector.py`, `event_pipeline.py` | `wildlife_soundscape.pipeline.*` |
| `database.py` | `wildlife_soundscape.storage.database` facade and `storage.schema`, `migrations`, `queries` |
| `server.py`, `main.py`, `simulator.py` | `wildlife_soundscape.runtime.*`, with bounded work in `runtime.processing` |
| `analytics/`, `calibration/`, `classification/`, `dashboard/`, `dsp/`, `localization/`, `tools/` | `wildlife_soundscape.<package>` |

Run the receiver with `python -m wildlife_soundscape`; use the installed
commands `wildlife-simulator`, `wildlife-export-events`,
`wildlife-export-research`, `wildlife-benchmark-gcc`, and
`wildlife-validate-dataset` for their respective operations. The dashboard launcher runs
`src/wildlife_soundscape/dashboard/app.py`.
