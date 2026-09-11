@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe main.py app
if errorlevel 1 (
    echo.
    echo If .venv doesn't exist, run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -e ".[desktop,voice]"
    pause
)
