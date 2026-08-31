@echo off
setlocal

call "%~dp0_common.bat" || exit /b 1
"%WILDLIFE_VENV%\Scripts\wildlife-simulator.exe" %*
exit /b %ERRORLEVEL%
