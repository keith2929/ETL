@echo off
setlocal enabledelayedexpansion

set "BASE=%~dp0"
title ION Orchard Loyalty Pipeline

:: ── Check venv exists ─────────────────────────────────────────
if not exist "%BASE%venv\Scripts\python.exe" (
    echo ============================================================
    echo   Setup not complete.
    echo   Please double-click setup_windows.bat first.
    echo ============================================================
    pause
    exit /b 1
)

:: ── Check app exists — flat structure, no app\ subfolder ──────
if not exist "%BASE%app_FINAL.py" (
    echo ============================================================
    echo   ERROR: app_FINAL.py not found in:
    echo   %BASE%
    echo.
    echo   Files found:
    dir "%BASE%" /b
    echo ============================================================
    pause
    exit /b 1
)

echo ============================================================
echo   ION ORCHARD LOYALTY PIPELINE
echo   Starting - browser opens in ~10 seconds
echo   Keep this window open while using the app
echo   If browser does not open: http://localhost:8501
echo ============================================================
echo.

:: ── Launch — all files in BASE, no cd needed ──────────────────
"%BASE%venv\Scripts\python.exe" -m streamlit run "%BASE%app_FINAL.py" ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --server.port 8501 ^
    --theme.base dark

set "ERR=%ERRORLEVEL%"
echo.
if %ERR% NEQ 0 (
    echo ============================================================
    echo   App stopped. Error code: %ERR%
    echo   Screenshot this and send to your contact.
    echo ============================================================
)
pause