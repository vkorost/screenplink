@echo off
REM Phase 0 diagnostic tool. Run this to find out whether the camera can see the laser.
REM THE PROJECT ROOT IS ONE LEVEL UP: this moved into scripts\ on 2026-10-18.
REM Everything below is relative to the root -- the venv, src\ and the paths
REM the app resolves from its own location.
cd /d "%~dp0.."
REM 74s -> 0.4s camera open on a Logitech C910. See src/lsa/_env.py.
set OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment missing. Run setup first:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "src\lsa\tools\camera_lab.py" %*
if errorlevel 1 pause
