$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell
foreach ($item in @(@('CSSIM Training.lnk', ''), @('Stop CSSIM Training.lnk', ' -Stop'))) {
    $shortcut = $shell.CreateShortcut((Join-Path $desktop $item[0]))
    $shortcut.TargetPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $shortcut.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + (Join-Path $PSScriptRoot 'start_learning.ps1') + '"' + $item[1]
    $shortcut.WorkingDirectory = $repo
    $shortcut.WindowStyle = 7
    $shortcut.Description = 'CSSIM cumulative ten-game collection, offline RL and desktop reporting'
    $shortcut.Save()
}
Write-Host 'Desktop start/stop shortcuts installed.'
