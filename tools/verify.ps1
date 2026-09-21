param(
    [string]$PythonExe = 'E:\CSSIM\Client\Data\AlgData\PythonEnv\python.exe'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonCode = Join-Path $repoRoot 'PythonCode'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Push-Location $pythonCode
try {
    & $PythonExe -m compileall -q .
    if ($LASTEXITCODE -ne 0) { throw "compileall failed" }
    & $PythonExe -m pytest tests -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
} finally {
    Pop-Location
}

Write-Host 'Verification passed.'
