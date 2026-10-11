# Starts the FastAPI service on http://localhost:8000 (Swagger at /docs)
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
& (Join-Path $ProjectRoot "run.ps1") -Mode api

