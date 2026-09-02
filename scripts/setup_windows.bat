@echo off
setlocal

for %%I in ("%~dp0..") do set "WILDLIFE_PROJECT_ROOT=%%~fI"
cd /d "%WILDLIFE_PROJECT_ROOT%" || exit /b 1

python -m venv .venv || exit /b 1
.venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python.exe -m pip install -e ".[dev]" || exit /b 1

set "NUMBA_CACHE_DIR=%TEMP%\wildlife-soundscape-numba-cache"
call scripts\validate_windows.bat || exit /b 1

echo Development environment is ready.
exit /b 0
