@echo off
cd /d "%~dp0"
title ION Pipeline Setup

echo ============================================================
echo   ION ORCHARD PIPELINE - WINDOWS SETUP
echo   Run this once after downloading.
echo ============================================================
echo.

:: ── Find Python ───────────────────────────────────────────────
set PYTHON=
for %%P in (
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%LOCALAPPDATA%\anaconda3\python.exe"
    "%LOCALAPPDATA%\miniconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\anaconda3\python.exe"
) do (
    if not defined PYTHON (
        if exist %%P set PYTHON=%%P
    )
)

:: Fallback to PATH python
if not defined PYTHON (
    where python >nul 2>&1
    if %ERRORLEVEL% == 0 set PYTHON=python
)

if not defined PYTHON (
    echo ❌ Python not found.
    echo.
    echo Please install Python from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during install.
    echo Then double-click this file again.
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Found Python: %PYTHON%
echo.

:: ── Create virtual environment ────────────────────────────────
echo Step 1/3: Creating virtual environment...

if exist "venv" (
    echo    Removing old venv...
    rmdir /s /q venv
)

%PYTHON% -m venv venv

if not exist "venv\Scripts\python.exe" (
    echo ❌ Failed to create virtual environment.
    pause
    exit /b 1
)

echo    Done.
echo.

:: ── Install packages ──────────────────────────────────────────
echo Step 2/3: Installing packages from requirements.txt...
echo    This takes 3-5 minutes. Please wait.
echo.

venv\Scripts\pip install --upgrade pip --quiet

venv\Scripts\pip install -r requirements.txt --quiet --no-warn-script-location

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Package installation failed.
    echo    Check your internet connection and try again.
    pause
    exit /b 1
)

echo    Done.
echo.

:: ── Verify ────────────────────────────────────────────────────
echo Step 3/3: Verifying installation...

venv\Scripts\python -c "
import streamlit, pandas, statsmodels, sklearn, plotly, openpyxl
print('   streamlit   ', streamlit.__version__, ' OK')
print('   pandas      ', pandas.__version__,    ' OK')
print('   statsmodels ', statsmodels.__version__,' OK')
print('   scikit-learn', sklearn.__version__,    ' OK')
print('   plotly      ', plotly.__version__,     ' OK')
"

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Verification failed. Please run setup again.
    pause
    exit /b 1
)

:: ── Set up config ─────────────────────────────────────────────
if not exist "config_ION.xlsx" (
    if exist "config_template.xlsx" (
        copy config_template.xlsx config_ION.xlsx >nul
        echo.
        echo ✅ Created config_ION.xlsx from template.
        echo    Open it and fill in your folder paths.
    )
)

echo.
echo ============================================================
echo   ✅ SETUP COMPLETE
echo.
echo   From now on just double-click RUN_APP.bat
echo ============================================================
echo.
pause