param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [string]$CssimRoot = 'E:\CSSIM'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $CssimRoot 'Client\Data\AlgData\PythonCode'))) {
    foreach ($candidate in @('C:\CSSIM', 'E:\CSSIM')) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'Client\Data\AlgData\PythonCode')) {
            $CssimRoot = $candidate
            break
        }
    }
}

function Invoke-GitChecked {
    param(
        [Parameter(Mandatory = $true)] [string[]]$Arguments,
        [string]$Description = ($Arguments -join ' ')
    )

    $output = & git -C $repoRoot @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "git $Description failed with exit code $code."
    }
    return @($output)
}

$branch = (Invoke-GitChecked -Arguments @('branch', '--show-current') -Description 'branch --show-current') -join ''
if ($branch.Trim() -ne 'main') {
    throw "Publishing is allowed only from branch main; current branch is '$branch'."
}

& (Join-Path $PSScriptRoot 'verify.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Verification failed; publish stopped.' }

$remote = (Invoke-GitChecked -Arguments @('remote', 'get-url', 'origin') -Description 'remote get-url origin') -join ''
if ([string]::IsNullOrWhiteSpace($remote)) {
    throw 'No origin remote configured. Add the GitHub repository URL first.'
}

& git -C $repoRoot add .
if ($LASTEXITCODE -ne 0) { throw "git add failed with exit code $LASTEXITCODE." }

& git -C $repoRoot diff --cached --quiet
$diffCode = $LASTEXITCODE
if ($diffCode -gt 1) { throw "git diff --cached failed with exit code $diffCode." }
if ($diffCode -eq 1) {
    & git -C $repoRoot commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw "git commit failed with exit code $LASTEXITCODE." }
} else {
    Write-Host 'No changes to commit.'
}

& git -C $repoRoot push -u origin main
if ($LASTEXITCODE -ne 0) { throw "git push failed with exit code $LASTEXITCODE; CSSIM files were not changed." }

$localHead = (Invoke-GitChecked -Arguments @('rev-parse', 'HEAD') -Description 'rev-parse HEAD') -join ''
$remoteHeadLine = Invoke-GitChecked -Arguments @('ls-remote', 'origin', 'refs/heads/main') -Description 'ls-remote origin refs/heads/main'
$remoteHead = (($remoteHeadLine -join "`n") -split "`t")[0].Trim()
if ([string]::IsNullOrWhiteSpace($remoteHead) -or $remoteHead -ne $localHead.Trim()) {
    throw "Remote origin/main ($remoteHead) does not match local HEAD ($localHead). CSSIM files were not changed."
}

& (Join-Path $PSScriptRoot 'sync_to_cssim.ps1') -CssimRoot $CssimRoot
if ($LASTEXITCODE -ne 0) { throw 'CSSIM synchronization failed after the verified push.' }

Write-Host 'Published to GitHub and synchronized to CSSIM client/server.'
