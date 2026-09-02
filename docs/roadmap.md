# Repository Roadmap

## Package migration — completed

- All Python implementation modules are under `src/wildlife_soundscape`.
- Canonical imports use the `wildlife_soundscape` namespace; root-level
  compatibility modules have been removed.
- The source package is discovered through the `src` layout in `pyproject.toml`.
- Console entry points cover the receiver, simulator, exporters, and benchmark.
- The module map is documented in [package-layout.md](package-layout.md).

## Module boundaries

- Split `core/config.py` into focused settings modules.
- Split `storage/database.py` into schema, migration, and repository layers.
- Split `runtime/server.py` into receiver lifecycle, packet dispatch, and session
  control.
- Preserve database and network protocol compatibility throughout.

## Scientific validation

- Validate BirdNET acoustic and geographic models with the optional dependency
  installed and provide a trusted structured taxonomy source.
- Compile all three ESP32 sketches with the selected Arduino-ESP32 core.
- Run physical timing calibration and localization benchmarks on the deployed
  array.
- Keep acoustic and geographic model evidence separately traceable.

## Quality targets

- Keep the enforced mypy baseline at zero.
- Raise the enforced coverage threshold above 40%, prioritizing persistence,
  receiver lifecycle, dashboard, calibration, and research-export code.
- Keep software quality checks, package builds, and dependency audits in CI;
  add a separate firmware compile job once the Arduino-ESP32 toolchain version
  is selected and recorded.
- Maintain the single canonical package namespace as new modules are added.
