# TrustLens Launch Script (Windows PowerShell)
# Launches the integrated TrustLens Streamlit application using the local .venv.
# Usage:
#   .\run.ps1
#
# If execution policy blocks this script in PowerShell, run with a one-session bypass:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1

param (
    [string]$Mode = "ui"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvStreamlit = Join-Path $VenvDir "Scripts\streamlit.exe"
$VenvUvicorn = Join-Path $VenvDir "Scripts\uvicorn.exe"

# 1. Verify virtual environment presence
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not detected at .venv. Please run .\setup.ps1 first to prepare the environment."
    exit 1
}

# 2. Check model state (without retraining)
$ModelFile = Join-Path $ProjectRoot "artifacts\model\model.joblib"
if (-not (Test-Path $ModelFile)) {
    Write-Host "[NOTICE] No trained model artifact was found in artifacts\model." -ForegroundColor Yellow
    Write-Host "         - To initialize a demo baseline model, run: & `"$VenvPython`" scripts/train_demo.py" -ForegroundColor Yellow
    Write-Host "         - To train on your raw dataset, run: & `"$VenvPython`" scripts/run_all.py data/raw/reviews.csv --label-semantics true" -ForegroundColor Yellow
    Write-Host "         Launching dashboard with model-loading notification..." -ForegroundColor DarkGray
} else {
    Write-Host "[OK] Trained model artifact found at artifacts\model\model.joblib." -ForegroundColor Green
}

# 3. Launch the requested application service
if ($Mode -eq "api") {
    Write-Host "Starting TrustLens FastAPI service on http://localhost:8000 (Swagger docs: http://localhost:8000/docs) ..." -ForegroundColor Cyan
    & $VenvUvicorn api.fastapi_app.main:app --host 127.0.0.1 --port 8000
} else {
    Write-Host "Starting TrustLens Streamlit Dashboard on http://localhost:8501 ..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server." -ForegroundColor DarkGray
    & $VenvStreamlit run streamlit_app/app.py --server.port 8501
}
