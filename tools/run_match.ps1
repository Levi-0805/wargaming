param(
    [string]$CssimRoot = '',
    [int]$TimeoutMinutes = 20,
    [switch]$NoLaunch
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-CssimRoot {
    param([string]$Requested)
    if (-not [string]::IsNullOrWhiteSpace($Requested) -and (Test-Path -LiteralPath (Join-Path $Requested 'Client'))) {
        return $Requested
    }
    foreach ($candidate in @('C:\CSSIM', 'E:\CSSIM')) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'Client')) {
            return $candidate
        }
    }
    throw 'CSSIM installation not found. Pass -CssimRoot.'
}

function Start-IfMissing {
    param(
        [Parameter(Mandatory = $true)][string]$ProcessName,
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string]$Work
    )
    if (Get-Process -Name $ProcessName -ErrorAction SilentlyContinue) {
        Write-Host "Already running: $ProcessName"
        return
    }
    if (-not (Test-Path -LiteralPath $Exe)) {
        throw "Executable not found: $Exe"
    }
    Start-Process -FilePath $Exe -WorkingDirectory $Work | Out-Null
    Write-Host "Started $ProcessName"
}

$CssimRoot = Resolve-CssimRoot -Requested $CssimRoot
$outputDir = Join-Path $CssimRoot 'Server\CSSIM\Saved\Output'
$python = Join-Path $CssimRoot 'Client\Data\AlgData\PythonEnv\python.exe'
$parser = Join-Path $repoRoot 'tools\parse_match_score.py'
$resultsDir = Join-Path $repoRoot 'results\matches'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

$startedAt = Get-Date
if (-not $NoLaunch) {
    Start-IfMissing -ProcessName 'Server' -Exe (Join-Path $CssimRoot 'Server\Server\Server.exe') -Work (Join-Path $CssimRoot 'Server\Server')
    Start-Sleep -Seconds 2
    Start-IfMissing -ProcessName 'CSSIMClient-Win64-Shipping' -Exe (Join-Path $CssimRoot 'Client\CSSIM\Binaries\Win64\CSSIMClient-Win64-Shipping.exe') -Work (Join-Path $CssimRoot 'Client\CSSIM\Binaries\Win64')
    Write-Host 'CSSIM client is open. In the client, start the same scenario as before (red: FsLutk7wD2RkNgbs). This script waits for the score file.'
}

$deadline = $startedAt.AddMinutes($TimeoutMinutes)
$known = @{}
Get-ChildItem -LiteralPath $outputDir -Filter 'Output_*.txt' -ErrorAction SilentlyContinue | ForEach-Object { $known[$_.FullName] = $true }

$summaryPath = $null
while ((Get-Date) -lt $deadline) {
    $candidates = Get-ChildItem -LiteralPath $outputDir -Filter 'Output_*.txt' -ErrorAction SilentlyContinue |
        Where-Object { -not $known.ContainsKey($_.FullName) -and $_.Length -gt 1000 -and $_.LastWriteTime -ge $startedAt.AddSeconds(-2) } |
        Sort-Object LastWriteTime -Descending
    foreach ($file in $candidates) {
        $stableFor = (Get-Date) - $file.LastWriteTime
        if ($stableFor.TotalSeconds -lt 8) { continue }
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $summaryPath = Join-Path $resultsDir "$stamp.json"
        & $python $parser $file.FullName --out $summaryPath
        if ($LASTEXITCODE -eq 0) {
            Copy-Item -LiteralPath $summaryPath -Destination (Join-Path $repoRoot 'results\latest.json') -Force
            Write-Host "Match recorded: $summaryPath"
            exit 0
        }
        Remove-Item -LiteralPath $summaryPath -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 5
}

Write-Host "No finished match score appeared within $TimeoutMinutes minutes."
exit 2
