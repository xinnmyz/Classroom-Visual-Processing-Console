$ErrorActionPreference = "Stop"
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutName = (-join ([char[]](0x8BFE,0x5802,0x89C6,0x89C9,0x6570,0x636E,0x5904,0x7406,0x53F0))) + ".lnk"
$shortcutPath = Join-Path $desktop $shortcutName
$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $env:WINDIR "System32\wscript.exe"
$shortcut.Arguments = "`"$(Join-Path $appRoot 'start_app.vbs')`""
$shortcut.WorkingDirectory = $appRoot
$shortcut.Description = "Classroom visual data processing tool"
$shortcut.IconLocation = $env:SystemRoot + "\System32\SHELL32.dll,13"
$shortcut.Save()
Write-Host "Shortcut created: $shortcutPath"
