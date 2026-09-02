# Development Guide

## Environments

Create and activate a local environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install only the application runtime:

```powershell
python -m pip install -e .
```

Install the full development environment, including the runtime:

```powershell
python -m pip install -e ".[dev]"
```

The requirement files remain compatibility inputs for environments that do not
yet install from `pyproject.toml`.

`scripts\setup_windows.bat` performs the development installation and runs the
test suite automatically.

## Validation

Run commands from the repository root:

```powershell
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'wildlife-soundscape-numba-cache'
python -m compileall -q src
python -m ruff check src tests
python -m mypy src tests
python -m pytest -q --cov=wildlife_soundscape --cov-report=term-missing
python -m pip_audit
```

All five commands are required release gates. The coverage threshold is set in
`pyproject.toml`; increase it as high-risk subsystems gain meaningful tests.

Run the GCC research benchmark:

```powershell
python -m wildlife_soundscape.tools.benchmark_gcc_variants `
  --output-csv data/exports/gcc_benchmark.csv `
  --output-json data/exports/gcc_benchmark.json
```

## Application entry points

Launch the complete live stack (receiver and dashboard) from one file:

```powershell
.\scripts\run_live.bat
```

Launch the complete simulated stack (receiver, simulator, and dashboard):

```powershell
.\scripts\run_demo.bat
```

These launchers run the services concurrently in separate command windows and
start acquisition automatically after all configured nodes connect.

Receiver:

```powershell
wildlife-receiver
```

Simulator, in another terminal:

```powershell
wildlife-simulator
```

The receiver can also be started with `python -m wildlife_soundscape`.

Dashboard:

```powershell
streamlit run src/wildlife_soundscape/dashboard/app.py
```

## Repository ownership

- `src/wildlife_soundscape/analytics/`, `classification/`, `dsp/`, and
  `localization/` contain domain logic.
- `src/wildlife_soundscape/calibration/` contains physical timing calibration
  and controlled localization benchmarks.
- `src/wildlife_soundscape/dashboard/` contains the Streamlit research
  interface.
- `firmware/` contains one Arduino-compatible sketch directory per ESP32 node.
- `src/wildlife_soundscape/tools/` contains executable research and export
  utilities.
- `data/` contains runtime outputs and directory placeholders; generated data
  is not source code and should not be committed.
- `docs/` contains protocol, release, and architectural documentation.
- `scripts/` contains all Windows setup, launch, and validation scripts.
- `src/wildlife_soundscape/` is the single canonical Python package namespace.

See `docs/roadmap.md` for the staged package and module-boundary migration.
