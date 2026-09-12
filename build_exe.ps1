$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualEnv = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VirtualEnv "Scripts\python.exe"
$SpecFile = Join-Path $ProjectRoot "MiscordInspector.spec"
$DistDirectory = Join-Path $ProjectRoot "dist\MiscordInspector"
$ZipPath = Join-Path $ProjectRoot "dist\MiscordInspector-windows-x64.zip"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment was not found. Run setup_env.ps1 first."
}

& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements-build.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m PyInstaller --noconfirm --clean $SpecFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Copy-Item -LiteralPath (Join-Path $ProjectRoot "config.yaml") -Destination $DistDirectory -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "README.md") -Destination $DistDirectory -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "5M25mmUnknowExp1.bmp") -Destination $DistDirectory -Force

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -LiteralPath $DistDirectory -DestinationPath $ZipPath -CompressionLevel Optimal

Write-Host "Executable: $(Join-Path $DistDirectory 'MiscordInspector.exe')"
Write-Host "Transfer package: $ZipPath"
