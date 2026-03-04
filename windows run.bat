@echo off
cd /d "%~dp0"

:: Find the config file in this folder
set CONFIG_FILE=
for %%f in (config_*.xlsx) do (
    set CONFIG_FILE=%%f
    goto :found
)

:found
if "%CONFIG_FILE%"=="" (
    echo ❌ No config file found. Please make sure a config_*.xlsx file is in this folder.
    pause
    exit /b 1
)

echo Running pipeline with: %CONFIG_FILE%
echo Please wait...

python main.py "%CONFIG_FILE%"

if %ERRORLEVEL% == 0 (
    echo.
    echo ✅ Pipeline completed successfully!
) else (
    echo.
    echo ❌ Pipeline failed. Please contact your support team.
)

pause
