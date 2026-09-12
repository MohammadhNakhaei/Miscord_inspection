$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualEnv = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path -LiteralPath $VirtualEnv)) {
    py -3.13 -m venv $VirtualEnv
}

$PythonExe = Join-Path $VirtualEnv "Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Environment is ready."
Write-Host "Run: & '$PythonExe' 'Offline Full Pipline.py'"
