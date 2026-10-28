@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>&1
    if not %errorlevel%==0 (
        echo Python 3 is not installed. Install Python 3.10 or newer first.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a private Python environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :failed
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Dependencies installed successfully.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_desktop_shortcut.ps1"
echo You can now double-click the desktop shortcut.
pause
exit /b 0

:failed
echo Installation failed. Check the message above and try again.
pause
exit /b 1
