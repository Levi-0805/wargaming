param(
    [string]$CssimRoot = 'E:\CSSIM',
    [string]$BackupRoot = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repoRoot 'PythonCode'
$client = Join-Path $CssimRoot 'Client\Data\AlgData\PythonCode'
$server = Join-Path $CssimRoot 'Server\Data\AlgData\PythonCode'
if ([string]::IsNullOrWhiteSpace($BackupRoot)) { $BackupRoot = Join-Path $CssimRoot 'Backups' }

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

function Invoke-RobocopyChecked {
    param(
        [Parameter(Mandatory = $true)] [string]$From,
        [Parameter(Mandatory = $true)] [string]$To,
        [Parameter(Mandatory = $true)] [string]$Description
    )

    robocopy $From $To /E /R:2 /W:1 /XD __pycache__ .idea logs .pytest_cache /XF '*.pyc' '*.log' /NFL /NDL /NJH /NJS /NP
    $code = $LASTEXITCODE
    if ($code -ge 8) { throw "robocopy $Description failed with exit code $code" }
    $global:LASTEXITCODE = 0
}

Assert-Directory -Path $source -Description 'repository PythonCode'
Assert-Directory -Path $client -Description 'CSSIM client PythonCode'
Assert-Directory -Path $server -Description 'CSSIM server PythonCode'
Assert-Directory -Path (Join-Path $CssimRoot 'Client\Data\AlgData') -Description 'CSSIM client AlgData'
Assert-Directory -Path (Join-Path $CssimRoot 'Server\Data\AlgData') -Description 'CSSIM server AlgData'

$branch = (& git -C $repoRoot branch --show-current)
if ($LASTEXITCODE -ne 0) { throw "git branch --show-current failed with exit code $LASTEXITCODE." }
if (($branch -join '').Trim() -ne 'main') { throw "Deployment is allowed only from branch main; current branch is '$($branch -join '')'." }

$status = & git -C $repoRoot status --porcelain
if ($LASTEXITCODE -ne 0) { throw "git status failed with exit code $LASTEXITCODE." }
if (-not [string]::IsNullOrWhiteSpace(($status -join "`n"))) {
    throw 'Repository worktree is not clean; refusing to deploy.'
}

$localHead = (& git -C $repoRoot rev-parse HEAD)
if ($LASTEXITCODE -ne 0) { throw "git rev-parse HEAD failed with exit code $LASTEXITCODE." }
$localHead = ($localHead -join '').Trim()
$remote = & git -C $repoRoot remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($remote -join ''))) {
    throw 'No origin remote configured; refusing to deploy.'
}
$remoteHeadLine = & git -C $repoRoot ls-remote origin refs/heads/main
if ($LASTEXITCODE -ne 0) { throw "git ls-remote failed with exit code $LASTEXITCODE." }
$remoteHead = ((($remoteHeadLine -join "`n") -split "`t")[0]).Trim()
if ([string]::IsNullOrWhiteSpace($remoteHead) -or $remoteHead -ne $localHead) {
    throw "origin/main ($remoteHead) does not match local HEAD ($localHead); refusing to deploy."
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$backupRun = Join-Path $BackupRoot $stamp
New-Item -ItemType Directory -Path $backupRun -Force | Out-Null
Invoke-RobocopyChecked -From $client -To (Join-Path $backupRun 'Client\Data\AlgData\PythonCode') -Description 'client backup'
Invoke-RobocopyChecked -From $server -To (Join-Path $backupRun 'Server\Data\AlgData\PythonCode') -Description 'server backup'

Invoke-RobocopyChecked -From $source -To $client -Description 'client deployment'
Invoke-RobocopyChecked -From $source -To $server -Description 'server deployment'

Assert-HashesMatch -SourceRoot $source -TargetRoot $client -Label 'CSSIM client PythonCode'
Assert-HashesMatch -SourceRoot $source -TargetRoot $server -Label 'CSSIM server PythonCode'
Write-Host "Synchronized repository PythonCode to CSSIM client and server. Backup: $backupRun"
