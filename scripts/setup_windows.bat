@echo off
setlocal

for %%I in ("%~dp0..") do set "WILDLIFE_PROJECT_ROOT=%%~fI"
cd /d "%WILDLIFE_PROJECT_ROOT%" || exit /b 1

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Removing unusable local virtual environment...
        rmdir /s /q ".venv"
    )
)

where python >nul 2>&1
if errorlevel 1 (
    echo Python 3.13 was not found on PATH.
    echo Install 64-bit Python 3.13, select "Add python.exe to PATH", then open a new terminal.
    exit /b 1
)

for /f %%V in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "WILDLIFE_PYTHON_VERSION=%%V"
if not "%WILDLIFE_PYTHON_VERSION%"=="3.13" (
    echo Python 3.13 is required; found %WILDLIFE_PYTHON_VERSION%.
    exit /b 1
)

python -m venv .venv || exit /b 1
.venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python.exe -m pip install -e ".[dev]" || exit /b 1

set "NUMBA_CACHE_DIR=%TEMP%\wildlife-soundscape-numba-cache"
call scripts\validate_windows.bat || exit /b 1

echo Development environment is ready.
exit /b 0
