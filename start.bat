@echo off
setlocal
title ModelFetcher
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\.deps_installed"

if not exist "%PY%" (
    echo Creating virtual environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV%"
    ) else (
        python -m venv "%VENV%"
    )
    if not exist "%PY%" (
        echo Could not create the virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

if not exist "%MARKER%" (
    echo Installing dependencies...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install --upgrade huggingface_hub
    if errorlevel 1 (
        echo Dependency install failed.
        pause
        exit /b 1
    )
    echo ok> "%MARKER%"
)

if /i "%~1"=="cli" (
    "%PY%" "%HERE%core.py"
) else (
    "%PY%" "%HERE%server.py" %*
)
echo.
pause
endlocal
