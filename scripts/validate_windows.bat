@echo off
setlocal

call "%~dp0_common.bat" || exit /b 1
set "NUMBA_CACHE_DIR=%TEMP%\wildlife-soundscape-numba-cache"

echo [1/5] Compiling Python sources...
"%WILDLIFE_PYTHON%" -m compileall -q src || exit /b 1

echo [2/5] Running Ruff...
"%WILDLIFE_PYTHON%" -m ruff check src tests || exit /b 1

echo [3/5] Type checking...
"%WILDLIFE_PYTHON%" -m mypy src tests || exit /b 1

echo [4/5] Running tests with coverage...
"%WILDLIFE_PYTHON%" -m pytest -q --cov=wildlife_soundscape --cov-report=term-missing || exit /b 1

echo [5/5] Auditing dependencies...
"%WILDLIFE_PYTHON%" -m pip_audit || exit /b 1

echo Validation completed successfully.
exit /b 0
