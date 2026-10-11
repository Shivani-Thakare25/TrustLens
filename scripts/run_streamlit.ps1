# Starts the Streamlit UI on http://localhost:8501
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
& (Join-Path $ProjectRoot "run.ps1")

