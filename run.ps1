$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Error "Missing .venv. Run .\install.ps1 first."
}

& .\.venv\Scripts\python.exe start_desktop.py
