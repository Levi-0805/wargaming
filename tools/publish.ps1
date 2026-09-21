param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [string]$CssimRoot = 'E:\CSSIM'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

$branch = git -C $repoRoot branch --show-current
if ($branch -ne 'main') {
    throw "Publishing is restricted to the main branch; current branch is '$branch'."
}

& (Join-Path $PSScriptRoot 'verify.ps1')
if (-not $?) { throw 'Verification failed; publish stopped.' }

$remote = git -C $repoRoot remote get-url origin 2>$null
if (-not $remote) {
    throw 'No origin remote configured. Add the GitHub repository URL first.'
}

git -C $repoRoot add .
git -C $repoRoot diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host 'No changes to commit.'
} else {
    git -C $repoRoot commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed.' }
}

git -C $repoRoot push -u origin main
if (-not $?) { throw 'git push failed; CSSIM files were not changed.' }

$localHead = (git -C $repoRoot rev-parse HEAD).Trim()
$remoteHead = (git -C $repoRoot ls-remote origin refs/heads/main).Split("`t")[0].Trim()
if (-not $remoteHead -or $remoteHead -ne $localHead) {
    throw "Remote main does not match local HEAD; CSSIM files were not changed."
}

& (Join-Path $PSScriptRoot 'sync_to_cssim.ps1') -CssimRoot $CssimRoot
if (-not $?) { throw 'CSSIM synchronization failed after push.' }

Write-Host 'Published to GitHub and synchronized to CSSIM client/server.'
