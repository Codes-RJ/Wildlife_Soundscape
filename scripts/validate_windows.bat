@echo off
setlocal

call "%~dp0_common.bat" || exit /b 1
set "NUMBA_CACHE_DIR=%TEMP%\wildlife-soundscape-numba-cache"

echo [1/6] Compiling Python sources...
"%WILDLIFE_PYTHON%" -m compileall -q src || exit /b 1

echo [2/6] Running Ruff...
"%WILDLIFE_PYTHON%" -m ruff check src tests || exit /b 1

echo [3/6] Type checking...
"%WILDLIFE_PYTHON%" -m mypy src tests || exit /b 1

echo [4/6] Running tests with coverage...
"%WILDLIFE_PYTHON%" -m pytest -q --cov=wildlife_soundscape --cov-report=term-missing || exit /b 1

echo [5/6] Building source and wheel distributions...
"%WILDLIFE_PYTHON%" -m build --no-isolation || exit /b 1

echo [6/6] Auditing dependencies...
"%WILDLIFE_PYTHON%" -m pip_audit || exit /b 1

echo Validation completed successfully.
exit /b 0
