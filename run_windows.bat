@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The app environment is not installed yet.
    echo Double-click install_windows.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo The app stopped with an error. Review the message above.
    pause
    exit /b 1
)

