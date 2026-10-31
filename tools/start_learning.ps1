param([switch]$Stop)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = @('C:\CSSIM\Client\Data\AlgData\PythonEnv\python.exe', 'E:\CSSIM\Client\Data\AlgData\PythonEnv\python.exe') | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $python) { throw 'CSSIM Python environment not found.' }
Push-Location $repo
try {
    $action = if ($Stop) { 'stop' } else { 'start' }
    & $python (Join-Path $PSScriptRoot 'cssim_learning.py') $action
    if ($LASTEXITCODE -ne 0) { throw 'CSSIM learning command failed; inspect logs/learning/worker.log.' }
} catch {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show($_.Exception.Message, 'CSSIM Training') | Out-Null
    throw
} finally { Pop-Location }
