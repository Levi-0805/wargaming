param(
    [string]$CssimRoot = 'E:\CSSIM'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $CssimRoot 'Client\Data\AlgData\PythonCode'
$server = Join-Path $CssimRoot 'Server\Data\AlgData\PythonCode'
$destination = Join-Path $repoRoot 'PythonCode'

function Assert-Directory {
    param([Parameter(Mandatory = $true)] [string]$Path, [Parameter(Mandatory = $true)] [string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Description directory not found: $Path"
    }
}

function Get-ManagedFiles {
    param([Parameter(Mandatory = $true)] [string]$Root)
    $excludedDirectories = @('__pycache__', '.idea', 'logs', '.pytest_cache')
    Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object {
        $relative = $_.FullName.Substring($Root.Length).TrimStart('\', '/')
        $parts = $relative -split '[\\/]'
        ($parts | Where-Object { $excludedDirectories -contains $_ }).Count -eq 0 -and
        $_.Extension -notin @('.pyc', '.log')
    }
}

function Assert-HashesMatch {
    param(
        [Parameter(Mandatory = $true)] [string]$SourceRoot,
        [Parameter(Mandatory = $true)] [string]$TargetRoot,
        [Parameter(Mandatory = $true)] [string]$Label
    )

    foreach ($file in Get-ManagedFiles -Root $SourceRoot) {
        $relative = $file.FullName.Substring($SourceRoot.Length).TrimStart('\', '/')
        $target = Join-Path $TargetRoot $relative
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            throw "$Label is missing managed file: $relative"
        }
        $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ($sourceHash -ne $targetHash) {
            throw "$Label hash mismatch for managed file: $relative"
        }
    }
}

Assert-Directory -Path $source -Description 'CSSIM client PythonCode'
Assert-Directory -Path $server -Description 'CSSIM server PythonCode'
Assert-Directory -Path $destination -Description 'repository PythonCode'

$dirty = & git -C $repoRoot status --porcelain -- PythonCode
if ($LASTEXITCODE -ne 0) { throw "git status failed with exit code $LASTEXITCODE." }
if (-not [string]::IsNullOrWhiteSpace(($dirty -join "`n"))) {
    throw 'Repository PythonCode has uncommitted changes; refusing to overwrite them during import.'
}

robocopy $source $destination /E /R:2 /W:1 /XD __pycache__ .idea logs .pytest_cache /XF '*.pyc' '*.log' /NFL /NDL /NJH /NJS /NP
$code = $LASTEXITCODE
if ($code -ge 8) {
    throw "robocopy failed with exit code $code"
}
$global:LASTEXITCODE = 0

Assert-HashesMatch -SourceRoot $source -TargetRoot $destination -Label 'Repository PythonCode'
Write-Host "Imported latest CSSIM client PythonCode into repository: $destination"
