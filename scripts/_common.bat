@echo off

for %%I in ("%~dp0..") do set "WILDLIFE_PROJECT_ROOT=%%~fI"
set "WILDLIFE_VENV=%WILDLIFE_PROJECT_ROOT%\.venv"
set "WILDLIFE_PYTHON=%WILDLIFE_VENV%\Scripts\python.exe"

if not exist "%WILDLIFE_PYTHON%" (
    echo Missing Python environment: "%WILDLIFE_VENV%"
    echo Run scripts\setup_windows.bat first.
    exit /b 1
)

cd /d "%WILDLIFE_PROJECT_ROOT%" || exit /b 1
exit /b 0
