# Runtime scripts

All Windows setup, launch, and validation scripts live in this directory.
They resolve the repository root automatically, so they can be started from
any working directory.

## One-click launch

With physical ESP32 nodes:

```powershell
.\scripts\run_live.bat
```

Without hardware, using the three-node simulator:

```powershell
.\scripts\run_demo.bat
```

Both launchers start services in this order:

1. Receiver, waiting for all configured nodes before starting acquisition.
2. Simulator in demo mode, or physical nodes in live mode.
3. Streamlit dashboard, which opens in the default browser.

The services run concurrently in separate command windows. Close each window
or press `Ctrl+C` in it to stop that service.

## Individual scripts

- `setup_windows.bat` creates `.venv`, installs dependencies, and validates the
  repository.
- `run_receiver.bat` starts only the receiver and forwards additional command
  arguments.
- `run_simulator.bat` starts only the simulator and forwards additional command
  arguments.
- `run_dashboard.bat` starts only the Streamlit dashboard.
- `run_website.bat live` is the health-checked live-mode orchestrator.
- `run_website.bat demo` is the health-checked demo-mode orchestrator.
- `run_website.ps1` contains the process orchestration and startup health
  checks used by the two one-click batch files.
- `validate_windows.bat` runs compilation, Ruff, mypy, the complete test suite
  with coverage enforcement, and a dependency audit.

## Research and data utilities

- `wildlife-validate-dataset` checks the native three-node dataset contract.
- `python -m wildlife_soundscape.tools.reprocess_unknown` previews or applies
  recovery of old heuristic `unknown` classifications; apply mode creates a
  database backup first.
- `wildlife-export-events` and `wildlife-export-research` create portable
  exports with experiment-manifest sidecars.

See [`docs/OPERATIONS.md`](../docs/OPERATIONS.md) for data recovery and live
hardware checks.
