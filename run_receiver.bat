@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Run setup_windows.bat first.
    exit /b 1
)

.venv\Scripts\python.exe -m wildlife_soundscape
