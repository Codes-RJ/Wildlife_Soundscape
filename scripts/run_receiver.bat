@echo off
setlocal

call "%~dp0_common.bat" || exit /b 1
"%WILDLIFE_PYTHON%" -m wildlife_soundscape %*
exit /b %ERRORLEVEL%
