param(
    [string]$CssimRoot = 'E:\CSSIM'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repoRoot 'PythonCode'

if (-not (Test-Path -LiteralPath $source)) {
    throw "Repository PythonCode directory not found: $source"
}

foreach ($target in @(
    (Join-Path $CssimRoot 'Client\Data\AlgData\PythonCode'),
    (Join-Path $CssimRoot 'Server\Data\AlgData\PythonCode')
)) {
    robocopy $source $target /E /XD __pycache__ .idea logs .pytest_cache /XF '*.pyc' '*.log' /NFL /NDL /NJH /NJS /NP
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed with exit code $code for target $target"
    }
    Write-Host "Synced repository PythonCode to: $target"
}
