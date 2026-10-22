@echo off
title CyberSentinel AI Setup
echo ========================================
echo   CyberSentinel AI - Setup and Launch
echo ========================================
python --version
if errorlevel 1 (
  echo Python was not found. Install Python 3.10-3.12 and enable Add Python to PATH.
  pause
  exit /b 1
)
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check your internet connection and Python version.
  pause
  exit /b 1
)
echo.
echo Starting CyberSentinel AI at http://127.0.0.1:5000
python app.py
pause
