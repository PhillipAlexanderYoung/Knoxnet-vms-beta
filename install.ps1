$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = $env:PYTHON_BIN
if ([string]::IsNullOrWhiteSpace($python)) { $python = "python" }

if (-not (Test-Path ".venv")) {
    & $python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item "env.example" ".env"
}

Write-Host "Knoxnet VMS Beta installed."
Write-Host "Start it with: .\run.ps1 or run.bat"
