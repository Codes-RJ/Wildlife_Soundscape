@echo off
setlocal

python -m venv .venv || exit /b 1
.venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python.exe -m pip install -e ".[dev]" || exit /b 1

set "NUMBA_CACHE_DIR=%TEMP%\wildlife-soundscape-numba-cache"
.venv\Scripts\python.exe -m ruff check . || exit /b 1
.venv\Scripts\python.exe -m pytest -q || exit /b 1

echo Development environment is ready.
