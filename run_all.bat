@echo off
setlocal

REM Determine repository root (the folder containing this script)
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

echo.
echo ============================================
echo   ModelOps Dev Launcher
echo ============================================
echo.

REM Prepare backend activation command if local venv is present
set "BACKEND_ACTIVATE="
if exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" (
    set "BACKEND_ACTIVATE=call ""%SCRIPT_DIR%.venv\Scripts\activate.bat"" && "
) else (
    echo Setting up virtual environment...
    py -m venv .venv >nul 2>&1
    if errorlevel 1 (
        echo Python launcher not available, retrying with python.exe...
        python -m venv .venv
    )
    if not exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" (
        echo Failed to create virtual environment. Please ensure Python is installed and retry.
        exit /b 1
    )
    call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
    "%SCRIPT_DIR%.venv\Scripts\python.exe" -m pip install -r "%SCRIPT_DIR%requirements.txt"
    echo Virtual environment setup complete.
    set "BACKEND_ACTIVATE=call \"%SCRIPT_DIR%.venv\Scripts\activate.bat\" && "
)

echo Launching backend API window...
start "ModelOps Backend" cmd /k "cd /d ""%SCRIPT_DIR%"" && %BACKEND_ACTIVATE%python start_api.py"

REM Frontend setup
set "FRONTEND_DIR=%SCRIPT_DIR%frontend"
if exist "%FRONTEND_DIR%\package.json" (
    if not exist "%FRONTEND_DIR%\node_modules" (
        echo Installing frontend dependencies (first run only)...
        pushd "%FRONTEND_DIR%"
        call npm install
        popd
    )
    echo Launching frontend dev server window...
    start "ModelOps Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && npm run dev -- --host"
) else (
    echo Frontend package.json not found at "%FRONTEND_DIR%\package.json".
)

echo.
echo Backend and frontend launch commands dispatched.
echo.

popd
endlocal
