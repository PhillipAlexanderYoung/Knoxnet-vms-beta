@echo off
REM Start the local vision service
echo Starting Local Vision Service...
echo.
echo Models: BLIP (Salesforce) and GIT (Microsoft)
echo Device: Auto-detect (CUDA if available, else CPU)
echo Port: 8101
echo.
echo First run will download models (~500-990MB)
echo.

cd /d "%~dp0"

REM Prefer the bundled KnoxnetVMS executable in production installs.
set "ROOT=%~dp0..\.."
if exist "%ROOT%\KnoxnetVMS.exe" (
    echo Using bundled runtime: %ROOT%\KnoxnetVMS.exe --run-vision-local
    "%ROOT%\KnoxnetVMS.exe" --run-vision-local
    goto :done
)

REM Dev fallback (requires python + deps installed)
python -m services.vision_local

:done
pause

