@echo off
setlocal

set "WILDLIFE_MODE=%~1"
if not defined WILDLIFE_MODE set "WILDLIFE_MODE=live"

if /i not "%WILDLIFE_MODE%"=="live" if /i not "%WILDLIFE_MODE%"=="demo" (
    echo Usage: scripts\run_website.bat [live^|demo]
    exit /b 2
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_website.ps1" -Mode "%WILDLIFE_MODE%"
exit /b %ERRORLEVEL%
