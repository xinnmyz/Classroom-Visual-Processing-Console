Set-Location $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = (Get-Command python).Source }
& $pythonPath -m streamlit run app.py
