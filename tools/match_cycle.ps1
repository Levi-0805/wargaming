param(
    [string]$Message = '',
    [string]$CssimRoot = '',
    [int]$TimeoutMinutes = 20
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

function Enable-GitHubProxy {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $wait = $client.BeginConnect('127.0.0.1', 7897, $null, $null)
        $open = $wait.AsyncWaitHandle.WaitOne(300) -and $client.Connected
    } finally {
        $client.Close()
    }
    if (-not $open) { return }
    $env:GIT_CONFIG_COUNT = '2'
    $env:GIT_CONFIG_KEY_0 = 'http.proxy'
    $env:GIT_CONFIG_VALUE_0 = 'http://127.0.0.1:7897'
    $env:GIT_CONFIG_KEY_1 = 'http.sslBackend'
    $env:GIT_CONFIG_VALUE_1 = 'openssl'
    $env:HTTP_PROXY = 'http://127.0.0.1:7897'
    $env:HTTPS_PROXY = 'http://127.0.0.1:7897'
}

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & git -C $repoRoot @Arguments
    if ($LASTEXITCODE -ne 0) { throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE." }
}

$CssimRoot = Resolve-CssimRoot -Requested $CssimRoot
Enable-GitHubProxy
$author = (& git -C $repoRoot log -1 --format='%an') -join ''
$email = (& git -C $repoRoot log -1 --format='%ae') -join ''
if ($LASTEXITCODE -ne 0) { throw 'git log failed while reading the existing author.' }
$env:GIT_AUTHOR_NAME = $author.Trim()
$env:GIT_AUTHOR_EMAIL = $email.Trim()
$env:GIT_COMMITTER_NAME = $author.Trim()
$env:GIT_COMMITTER_EMAIL = $email.Trim()

$dirty = (& git -C $repoRoot status --porcelain)
if ($LASTEXITCODE -ne 0) { throw "git status failed with exit code $LASTEXITCODE." }
if (-not [string]::IsNullOrWhiteSpace(($dirty -join ''))) {
    if ([string]::IsNullOrWhiteSpace($Message)) {
        throw 'The repository has uncommitted changes. Re-run with -Message "why the algorithm changed" so they are verified, pushed, and deployed before the match.'
    }
    & (Join-Path $PSScriptRoot 'publish.ps1') -Message $Message -CssimRoot $CssimRoot
    if ($LASTEXITCODE -ne 0) { throw 'Publish failed; match was not started.' }
} else {
    & (Join-Path $PSScriptRoot 'sync_to_cssim.ps1') -CssimRoot $CssimRoot
    if ($LASTEXITCODE -ne 0) { throw 'CSSIM sync failed; match was not started.' }
}

& (Join-Path $PSScriptRoot 'run_match.ps1') -CssimRoot $CssimRoot -TimeoutMinutes $TimeoutMinutes
if ($LASTEXITCODE -ne 0) { throw "Match did not produce a score (exit $LASTEXITCODE)." }

$latest = Get-Content -LiteralPath (Join-Path $repoRoot 'results\latest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$score = [int]$latest.red_score
$recordMessage = "record red match score $score"

Invoke-Git -Arguments @('add', '--', 'results')
& git -C $repoRoot diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    Invoke-Git -Arguments @('commit', '-m', $recordMessage)
    Invoke-Git -Arguments @('push', 'origin', 'main')
    $local = ((Invoke-Git -Arguments @('rev-parse', 'HEAD')) -join '').Trim()
    $remoteLine = (Invoke-Git -Arguments @('ls-remote', 'origin', 'refs/heads/main')) -join ''
    $remote = ($remoteLine -split "`t")[0].Trim()
    if ($remote -ne $local) { throw "origin/main ($remote) does not match local HEAD ($local)." }
    Write-Host "Recorded red score $score and pushed $local"
} else {
    Write-Host "Red score $score. Result file was already committed."
}
