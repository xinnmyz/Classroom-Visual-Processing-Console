$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$needle = (Join-Path $appRoot "app.py").Replace("\\", "\")
$targets = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($needle) }
foreach ($target in $targets) {
    Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
}
