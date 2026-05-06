@echo off
cd /d "%~dp0"
title ION Orchard Loyalty Pipeline

:: ── Check setup has been run ──────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo ============================================================
    echo   Setup not complete.
    echo   Please double-click setup_windows.bat first.
    echo ============================================================
    pause
    exit /b 1
)

echo ============================================================
echo   ION ORCHARD LOYALTY PIPELINE
echo   Starting — browser opens in ~10 seconds
echo   Keep this window open while using the app
echo   If browser doesn't open: http://localhost:8501
echo ============================================================
echo.

cd app
..\venv\Scripts\python.exe -m streamlit run app_FINAL.py ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --server.port 8501 ^
    --theme.base dark

echo.
echo App stopped. Screenshot this if there's an error.
pause