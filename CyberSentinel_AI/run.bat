@echo off
setlocal enabledelayedexpansion
title CyberSentinel AI - Threat Detection Platform

echo =====================================================================
echo    CyberSentinel AI - Intelligent Threat Detection Platform
echo =====================================================================
echo.

cd /d "%~dp0"

set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    echo [*] Using local virtual environment: .venv\Scripts\python.exe
) else if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
    echo [*] Using parent virtual environment: ..\.venv\Scripts\python.exe
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
        echo [!] Virtual environment not detected; using system Python.
    ) else (
        echo [ERROR] Python is not installed or not available in PATH!
        echo Please install Python 3.10+ from https://www.python.org/downloads/
        echo and ensure 'Add Python to PATH' is selected during installation.
        echo.
        pause
        exit /b 1
    )
)

echo [*] Checking Python dependencies...
"%PYTHON_EXE%" -c "import flask, pandas, numpy, sklearn" >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Required dependencies missing. Installing packages...
    if exist "requirements.txt" (
        "%PYTHON_EXE%" -m pip install -r requirements.txt
    ) else if exist "..\requirements.txt" (
        "%PYTHON_EXE%" -m pip install -r ..\requirements.txt
    ) else (
        "%PYTHON_EXE%" -m pip install Flask pandas numpy scikit-learn scipy
    )
    if %errorlevel% neq 0 (
        echo [ERROR] Package installation failed. Please check your internet connection.
        pause
        exit /b 1
    )
)

echo.
echo =====================================================================
echo    CyberSentinel AI Engine Active
echo    Access Dashboard at: http://127.0.0.1:5000
echo    Press Ctrl+C to terminate the server.
echo =====================================================================
echo.

"%PYTHON_EXE%" app.py

if %errorlevel% neq 0 (
    echo.
    echo [!] Server stopped.
    pause
)
