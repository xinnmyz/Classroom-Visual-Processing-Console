$ErrorActionPreference = "Stop"

$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8502
$venvPython = Join-Path $appRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { (Get-Command python -ErrorAction SilentlyContinue).Source }

function Test-PortOpen([int]$candidatePort) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect("127.0.0.1", $candidatePort, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne(250)
        if ($connected -and $client.Connected) { return $true }
        return $false
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

if (-not $pythonCommand) {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "Python was not found. Install Python 3.10 or newer, then run install_dependencies.bat.",
        "Classroom data tool"
    ) | Out-Null
    exit 1
}

if (Test-PortOpen $port) {
    Start-Process "http://localhost:$port"
    exit 0
}

while (Test-PortOpen $port) { $port += 1 }

$logDir = Join-Path $appRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdoutLog = Join-Path $logDir "streamlit_stdout.log"
$stderrLog = Join-Path $logDir "streamlit_stderr.log"
$pythonPath = $pythonCommand
$arguments = @(
    "-m", "streamlit", "run", (Join-Path $appRoot "app.py"),
    "--server.headless", "true",
    "--server.port", "$port",
    "--browser.gatherUsageStats", "false"
)

Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $arguments `
    -WorkingDirectory $appRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog | Out-Null

$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 250
    if (Test-PortOpen $port) {
        $ready = $true
        break
    }
}

if ($ready) {
    Start-Process "http://localhost:$port"
} else {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "The app timed out while starting. Check logs\\streamlit_stderr.log.",
        "Classroom data tool"
    ) | Out-Null
}
