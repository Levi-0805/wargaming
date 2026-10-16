param(
    [string]$PythonExe = ''
)

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = @(
        'E:\CSSIM\Client\Data\AlgData\PythonEnv\python.exe',
        'C:\CSSIM\Client\Data\AlgData\PythonEnv\python.exe'
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonCode = Join-Path $repoRoot 'PythonCode'
$testTemp = Join-Path $repoRoot ('logs\verify-' + [guid]::NewGuid().ToString('N'))

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Push-Location $pythonCode
try {
    # Compile source directories only; old pytest caches may belong to another
    # Windows account. Every run gets a new, ignored temporary directory.
    & $PythonExe -m compileall -q cssim TeamAlg AlgTemplate tests main.py
    if ($LASTEXITCODE -ne 0) { throw "compileall failed" }
    & $PythonExe -m pytest tests -q -p no:cacheprovider --basetemp $testTemp
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
} finally {
    Pop-Location
}

Write-Host 'Verification passed.'
