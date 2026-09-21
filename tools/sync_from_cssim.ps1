param(
    [string]$CssimRoot = 'E:\CSSIM'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $CssimRoot 'Client\Data\AlgData\PythonCode'
$destination = Join-Path $repoRoot 'PythonCode'

if (-not (Test-Path -LiteralPath $source)) {
    throw "CSSIM source directory not found: $source"
}

robocopy $source $destination /E /XD __pycache__ .idea logs .pytest_cache /XF '*.pyc' '*.log' /NFL /NDL /NJH /NJS /NP
$code = $LASTEXITCODE
if ($code -ge 8) {
    throw "robocopy failed with exit code $code"
}

Write-Host "Imported latest CSSIM client PythonCode into repository: $destination"
