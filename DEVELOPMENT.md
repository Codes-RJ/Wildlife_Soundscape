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
python -m pip install -r requirements.txt
```

Install the full development environment, including the runtime:

```powershell
python -m pip install -r requirements-dev.txt
```

`setup_windows.bat` performs the development installation and runs the test
suite automatically.

## Validation

Run commands from the repository root:

```powershell
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'wildlife-soundscape-numba-cache'
python -m compileall -q analytics calibration classification dashboard dsp localization tools
python -m ruff check .
python -m pytest -q
python -m mypy .
```

Ruff and pytest are required regression gates. Mypy is currently a tracked
cleanup target and may remain non-zero until the typing migration is complete.

Run the GCC research benchmark:

```powershell
python tools/benchmark_gcc_variants.py `
  --output-csv data/exports/gcc_benchmark.csv `
  --output-json data/exports/gcc_benchmark.json
```

## Application entry points

Receiver:

```powershell
python main.py
```

Simulator, in another terminal:

```powershell
python simulator.py
```

Dashboard:

```powershell
streamlit run dashboard/app.py
```

## Repository ownership

- `analytics/`, `classification/`, `dsp/`, and `localization/` contain domain
  logic.
- `calibration/` contains physical timing calibration and controlled
  localization benchmarks.
- `dashboard/` contains the Streamlit research interface.
- `firmware/` contains one Arduino-compatible sketch directory per ESP32 node.
- `tools/` contains executable research and export utilities.
- `data/` contains runtime outputs and directory placeholders; generated data
  is not source code and should not be committed.
- `docs/` contains protocol, release, and architectural documentation.

See `docs/roadmap.md` for the staged package and module-boundary migration.
