@echo off
setlocal enabledelayedexpansion

set "BASE=%~dp0"

title ION Pipeline Setup

echo ============================================================
echo   ION ORCHARD PIPELINE - WINDOWS SETUP
echo ============================================================
echo.

:: ── Find Python ───────────────────────────────────────────────
set "PYTHON="
for %%P in (
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%LOCALAPPDATA%\anaconda3\python.exe"
    "%LOCALAPPDATA%\miniconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\anaconda3\python.exe"
) do (
    if not defined PYTHON (
        if exist %%P set "PYTHON=%%P"
    )
)

if not defined PYTHON (
    where python >nul 2>&1
    if %ERRORLEVEL% == 0 set "PYTHON=python"
)

if not defined PYTHON (
    echo Python not found.
    echo Please install from https://www.python.org/downloads/
    echo Check "Add Python to PATH" during install.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Found Python: %PYTHON%
echo.

:: ── Create virtual environment ────────────────────────────────
echo Step 1/3: Creating virtual environment...

if exist "%BASE%venv" (
    echo    Removing old venv...
    rmdir /s /q "%BASE%venv"
)

"%PYTHON%" -m venv "%BASE%venv"

if not exist "%BASE%venv\Scripts\python.exe" (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo    Done.
echo.

:: ── Install packages directly — skip pip upgrade ──────────────
echo Step 2/3: Installing packages...
echo    This takes 3-5 minutes. Please wait.
echo.

:: Use python -m pip with fully quoted paths
:: --no-warn-script-location suppresses the OneDrive path warnings
:: --disable-pip-version-check stops the upgrade notice entirely

"%BASE%venv\Scripts\python.exe" -m pip install ^
    streamlit pandas openpyxl statsmodels scipy scikit-learn plotly ^
    --quiet ^
    --no-warn-script-location ^
    --disable-pip-version-check

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Package installation failed.
    echo Trying one package at a time...
    echo.

    for %%P in (streamlit pandas openpyxl statsmodels scipy scikit-learn plotly) do (
        echo    Installing %%P...
        "%BASE%venv\Scripts\python.exe" -m pip install %%P ^
            --quiet ^
            --no-warn-script-location ^
            --disable-pip-version-check
        if %ERRORLEVEL% NEQ 0 (
            echo    WARNING: %%P failed - continuing anyway
        ) else (
            echo    %%P OK
        )
    )
)

echo    Done.
echo.

:: ── Verify each package ───────────────────────────────────────
echo Step 3/3: Verifying installation...
echo.

set "ALLOK=1"

"%BASE%venv\Scripts\python.exe" -c "import streamlit; print('   streamlit    ' + streamlit.__version__ + '  OK')"
if %ERRORLEVEL% NEQ 0 (echo    streamlit     MISSING & set "ALLOK=0")

"%BASE%venv\Scripts\python.exe" -c "import pandas; print('   pandas        ' + pandas.__version__ + '  OK')"
if %ERRORLEVEL% NEQ 0 (echo    pandas        MISSING & set "ALLOK=0")

"%BASE%venv\Scripts\python.exe" -c "import statsmodels; print('   statsmodels   ' + statsmodels.__version__ + '  OK')"
if %ERRORLEVEL% NEQ 0 (echo    statsmodels   MISSING & set "ALLOK=0")

"%BASE%venv\Scripts\python.exe" -c "import sklearn; print('   scikit-learn  ' + sklearn.__version__ + '  OK')"
if %ERRORLEVEL% NEQ 0 (echo    scikit-learn  MISSING & set "ALLOK=0")

"%BASE%venv\Scripts\python.exe" -c "import plotly; print('   plotly         ' + plotly.__version__ + '  OK')"
if %ERRORLEVEL% NEQ 0 (echo    plotly        MISSING & set "ALLOK=0")

"%BASE%venv\Scripts\python.exe" -c "import openpyxl; print('   openpyxl      ' + openpyxl.__version__ + '  OK')"
if %ERRORLEVEL% NEQ 0 (echo    openpyxl      MISSING & set "ALLOK=0")

echo.

if "%ALLOK%"=="0" (
    echo ============================================================
    echo   WARNING: Some packages failed to install.
    echo   Try running setup_windows.bat again.
    echo   If it keeps failing, contact your support person.
    echo ============================================================
    pause
    exit /b 1
)

:: ── Copy config template ──────────────────────────────────────
if not exist "%BASE%config_ION.xlsx" (
    if exist "%BASE%config_template.xlsx" (
        copy "%BASE%config_template.xlsx" "%BASE%config_ION.xlsx" >nul
        echo Created config_ION.xlsx
        echo Open it and fill in your folder paths.
        echo.
    )
)

echo ============================================================
echo   SETUP COMPLETE
echo.
echo   Next step: Double-click RUN_APP.bat
echo ============================================================
pause