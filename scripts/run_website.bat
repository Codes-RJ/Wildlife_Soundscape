@echo off
setlocal

call "%~dp0_common.bat" || exit /b 1

set "WILDLIFE_MODE=%~1"
if not defined WILDLIFE_MODE set "WILDLIFE_MODE=live"

if /i not "%WILDLIFE_MODE%"=="live" if /i not "%WILDLIFE_MODE%"=="demo" (
    echo Usage: scripts\run_website.bat [live^|demo]
    exit /b 2
)

echo [1/3] Starting the receiver in automatic-acquisition mode...
start "Wildlife Receiver" /D "%WILDLIFE_PROJECT_ROOT%" "%COMSPEC%" /k call "%~dp0run_receiver.bat" --auto-start --session-label "%WILDLIFE_MODE%"

if /i "%WILDLIFE_MODE%"=="demo" (
    echo [2/3] Starting the three-node simulator...
    start "Wildlife Simulator" /D "%WILDLIFE_PROJECT_ROOT%" "%COMSPEC%" /k call "%~dp0run_simulator.bat"
) else (
    echo [2/3] Live mode selected; waiting for physical nodes.
)

echo [3/3] Starting the Streamlit website...
start "Wildlife Dashboard" /D "%WILDLIFE_PROJECT_ROOT%" "%COMSPEC%" /k call "%~dp0run_dashboard.bat"

echo.
echo Wildlife Soundscape launched in %WILDLIFE_MODE% mode.
echo Close each service window or press Ctrl+C in it to stop that service.
exit /b 0
