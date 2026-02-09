# Quick run on Windows — installs and starts the Grid Inference Worker
# Usage: .\run.ps1   (or right-click → Run with PowerShell)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Use venv if present, else system Python
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
}

Write-Host "Installing/updating grid-inference-worker..." -ForegroundColor Cyan
pip install -e . -q

Write-Host "Starting Grid Inference Worker (http://localhost:7861)..." -ForegroundColor Green
python -m inference_worker.cli
