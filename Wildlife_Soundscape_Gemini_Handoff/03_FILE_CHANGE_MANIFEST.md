# File Change Manifest

Gemini must inspect actual local files before editing.

## Existing files likely to modify

### Classification
- `classification/birdnet_backend.py`
  - preserve the known-good abstention contract;
  - remove unconditional non-bird -> BIRD assumption;
  - integrate taxonomy/context adapter.
- `classification/factory.py`
  - expose new optional BirdNET context settings if needed.
- `classification/__init__.py`
  - export new context types only if project style requires it.
- `event_pipeline.py`
  - change only if additional BirdNET context must be passed through;
  - preserve graceful ensemble degradation.

### Configuration
- `config.py`
  - ecoacoustic analysis configuration;
  - localization beta/band options;
  - optional BirdNET geo/taxonomy settings;
  - optional GIS georeference.

### Database
- `database.py`
  - `soundscape_indices` table and access methods;
  - migration/backward-compatible initialization.

### Analytics
- `analytics/__init__.py`
- `analytics/service.py`
  - integrate new continuous soundscape analysis only if appropriate.

### Localization
- `localization/gcc_phat.py`
  - add beta and optional band limits without changing sign convention.
- `localization/engine.py`
  - thread configuration to GCC layer.
- `localization/__init__.py`
  - only if new public APIs are exported.

### Dashboard
- `dashboard/data_access.py`
- `dashboard/plots.py`
- `dashboard/analysis_view.py`
  - add index visualizations without breaking empty-state behavior.

### Export
- `tools/export_events.py`
  - GeoJSON format.

### Dependencies
- `requirements.txt` / optional requirements file
  - add only what is actually required.
  - `pyproj` should preferably be optional GIS functionality.

## New files recommended

- `classification/birdnet_context.py`
- `analytics/indices.py`
- `analytics/soundscape_service.py`
- `tools/benchmark_gcc_variants.py`
- tests corresponding to each new subsystem

Optional:
- `tools/export_embeddings.py`
- `research/cluster_embeddings.py`

## Files that should NOT be changed for these features

Unless actual tests prove otherwise:
- `protocol.py`
- `node.py`
- `stream_manager.py`
- ESP32 firmware files
- protocol packet structures
- TDOA sign convention
