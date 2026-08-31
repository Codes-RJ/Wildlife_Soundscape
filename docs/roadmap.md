# Repository Roadmap

## Package migration

- Introduce an installable `src/wildlife_soundscape` package.
- Move modules in dependency order while retaining temporary compatibility
  imports.
- Replace path bootstrapping with installed-package imports.
- Add console entry points for the receiver, simulator, exporters, and
  benchmarks.

## Module boundaries

- Split `config.py` into focused settings modules.
- Split `database.py` into schema, migration, and repository layers.
- Split `server.py` into receiver lifecycle, packet dispatch, and session
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

- Reduce the mypy baseline to zero by subsystem rather than hiding findings.
- Raise overall test coverage, prioritizing protocol, persistence,
  localization, and acquisition lifecycle code.
- Add automated software quality checks and a separate firmware compile job.
- Remove compatibility shims after every consumer uses the installable package.
