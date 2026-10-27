@echo off
REM Start Local LLM Service on Windows

echo ====================================
echo Starting Local LLM Service
echo ====================================
echo.

REM Prefer the bundled KnoxnetVMS executable in production installs.
set "ROOT=%~dp0..\.."
if exist "%ROOT%\KnoxnetVMS.exe" (
    echo Using bundled runtime: %ROOT%\KnoxnetVMS.exe --run-llm-local
    "%ROOT%\KnoxnetVMS.exe" --run-llm-local
    goto :done
)

REM Check if Python is available (dev fallback)
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10 or higher, or use a packaged build.
    pause
    exit /b 1
)

REM Set environment variables (optional - can also use .env file)
REM set LLM_SERVICE_PORT=8102
REM set LLM_DEFAULT_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

echo Starting service on http://127.0.0.1:8102
echo Press Ctrl+C to stop
echo.

REM Start the service
python -m services.llm_local

:done
pause

