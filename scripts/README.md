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
- `run_website.bat live` is the underlying live-mode orchestrator.
- `run_website.bat demo` is the underlying demo-mode orchestrator.
- `validate_windows.bat` runs compilation, Ruff, and the complete test suite.
