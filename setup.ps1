# TrustLens Setup Script (Windows PowerShell)
# Creates the local virtual environment, installs dependencies, and prepares the workspace.
# Usage:
#   .\setup.ps1
#
# If execution policy blocks this script in PowerShell, run with a one-session bypass:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " TRUSTLENS - ENVIRONMENT SETUP & INITIALIZATION           " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Detect Python 3.10+
$PythonExe = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
}

try {
    $PyVersionRaw = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
    Write-Host "[1/5] Detected Python version: $PyVersionRaw" -ForegroundColor Green
    $PyMajor = [int]($PyVersionRaw.Split('.')[0])
    $PyMinor = [int]($PyVersionRaw.Split('.')[1])
    if ($PyMajor -lt 3 -or ($PyMajor -eq 3 -and $PyMinor -lt 10)) {
        Write-Error "Python 3.10 or higher is required. Please install Python 3.10 or 3.11 from python.org."
        exit 1
    }
} catch {
    Write-Error "Python was not detected. Please ensure Python is installed and accessible in your PATH."
    exit 1
}

# 2. Create isolated virtual environment
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[2/5] Creating virtual environment at .venv ..." -ForegroundColor Yellow
    & $PythonExe -m venv .venv
    if (-not (Test-Path $VenvPython)) {
        Write-Error "Failed to create virtual environment at .venv."
        exit 1
    }
    Write-Host "      Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "[2/5] Virtual environment already exists at .venv." -ForegroundColor Green
}

# 3. Upgrade pip and install dependencies
Write-Host "[3/5] Upgrading pip and installing required dependencies ..." -ForegroundColor Yellow
Write-Host "      This may take 2-4 minutes depending on network speed..." -ForegroundColor DarkGray
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements-dev.txt

# 4. Initialize workspace folders and environment configuration
Write-Host "[4/5] Initializing workspace directories and .env configuration ..." -ForegroundColor Yellow
& $VenvPython scripts/setup.py

# 5. Initialize demonstration baseline model if no model artifact exists
$ModelFile = Join-Path $ProjectRoot "artifacts\model\model.joblib"
if (-not (Test-Path $ModelFile)) {
    Write-Host "[5/5] Initializing offline demonstration model artifacts ..." -ForegroundColor Yellow
    & $VenvPython scripts/train_demo.py
    Write-Host "      Baseline demonstration model ready in artifacts\model." -ForegroundColor Green
} else {
    Write-Host "[5/5] Existing trained model found in artifacts\model. Preserving existing model." -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " TrustLens setup completed successfully!                 " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "To launch the TrustLens application, simply run:" -ForegroundColor Cyan
Write-Host "    .\run.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Or if blocked by PowerShell execution policy:" -ForegroundColor DarkGray
Write-Host "    powershell -ExecutionPolicy Bypass -File .\run.ps1" -ForegroundColor DarkGray
