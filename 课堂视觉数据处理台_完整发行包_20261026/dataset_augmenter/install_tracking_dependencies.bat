@echo off
setlocal
cd /d "%~dp0"
echo Installing optional multi-person tracking dependencies...
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo Could not create a Python virtual environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install "ultralytics>=8.3"
echo.
echo Tracking dependencies installed. The first run may download yolo11n.pt.
pause
