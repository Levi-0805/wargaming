param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [string]$CssimRoot = 'E:\CSSIM'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

& (Join-Path $PSScriptRoot 'verify.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Verification failed; publish stopped.' }

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
if ($LASTEXITCODE -ne 0) { throw 'git push failed; CSSIM files were not changed.' }

& (Join-Path $PSScriptRoot 'sync_to_cssim.ps1') -CssimRoot $CssimRoot
if ($LASTEXITCODE -ne 0) { throw 'CSSIM synchronization failed after push.' }

Write-Host 'Published to GitHub and synchronized to CSSIM client/server.'
