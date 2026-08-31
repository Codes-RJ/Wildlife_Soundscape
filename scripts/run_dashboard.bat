@echo off
setlocal

call "%~dp0_common.bat" || exit /b 1
"%WILDLIFE_PYTHON%" -m streamlit run dashboard\app.py %*
exit /b %ERRORLEVEL%
