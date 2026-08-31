@echo off
setlocal

call "%~dp0_common.bat" || exit /b 1
set "NUMBA_CACHE_DIR=%TEMP%\wildlife-soundscape-numba-cache"

echo [1/3] Compiling Python sources...
"%WILDLIFE_PYTHON%" -m compileall -q src analytics calibration classification dashboard dsp localization tools || exit /b 1

echo [2/3] Running Ruff...
"%WILDLIFE_PYTHON%" -m ruff check . || exit /b 1

echo [3/3] Running tests...
"%WILDLIFE_PYTHON%" -m pytest -q || exit /b 1

echo Validation completed successfully.
exit /b 0
