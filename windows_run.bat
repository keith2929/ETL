@echo off
cd /d "%~dp0"
echo ============================================================
echo   CAPSTONE PIPELINE APP
echo ============================================================
echo.

:: Try to find Python in common Anaconda/Miniconda locations
set PYTHON_EXE=
set PIP_EXE=
set STREAMLIT_EXE=

:: Check common install locations
for %%P in (
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\Anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\Miniconda3\python.exe"
    "%LOCALAPPDATA%\anaconda3\python.exe"
    "%LOCALAPPDATA%\Anaconda3\python.exe"
    "%LOCALAPPDATA%\miniconda3\python.exe"
    "C:\anaconda3\python.exe"
    "C:\Anaconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\ProgramData\Anaconda3\python.exe"
) do (
    if exist %%P (
        set PYTHON_EXE=%%P
        goto :found_python
    )
)

:: Fall back to whatever python is in PATH
where python >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON_EXE=python
    goto :found_python
)

echo Python not found. Please install Anaconda from https://www.anaconda.com
echo or Python from https://www.python.org (check "Add to PATH" during install).
pause
exit /b 1

:found_python
echo Found Python: %PYTHON_EXE%
echo.

:: Install streamlit if needed
echo Checking dependencies...
%PYTHON_EXE% -m pip install streamlit pandas openpyxl --quiet
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo Launching app — your browser will open automatically...
echo (Close this window to stop the app)
echo.
%PYTHON_EXE% -m streamlit run app.py --server.headless false

pause
