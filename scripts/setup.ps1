# Windows PowerShell: Delegated setup script.
# Run setup.ps1 from project root: .\setup.ps1
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
& (Join-Path $ProjectRoot "setup.ps1")

