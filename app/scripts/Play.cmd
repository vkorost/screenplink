@echo off
rem Starts ScreenPlink on the target screen (the game).
rem Close it with Escape or the Exit tile -- never End Task: a force-killed
rem app can hang the camera until it is unplugged. Extra arguments pass through.
rem
rem ONE FILE, TWO PLACES. In a release this sits beside screenplink.exe and runs
rem it. In the repository (app\scripts) there is no exe, so it finds the virtual
rem environment (app\.venv, else .venv beside app) and runs from source, the way
rem run_camera_lab.bat does. Created with:  python -m venv app\.venv  and
rem app\.venv\Scripts\python.exe -m pip install -r app\requirements.txt
rem
rem No labels or GOTOs on purpose: a .cmd saved with bare LF line endings can
rem miss a label, and this file must work however git checks it out.
setlocal
rem 74s -> 0.4s camera open (see src/lsa/_env.py). Mandatory, and it has to be
rem set before OpenCV is imported, so it is set before anything starts.
set "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0"
cd /d "%~dp0"
if exist "%~dp0screenplink.exe" start "" "%~dp0screenplink.exe" --target %* & exit /b 0
for %%I in ("%~dp0..") do set "APP=%%~fI"
set "PYW="
if exist "%APP%\.venv\Scripts\pythonw.exe" set "PYW=%APP%\.venv\Scripts\pythonw.exe"
if not defined PYW if exist "%APP%\..\.venv\Scripts\pythonw.exe" set "PYW=%APP%\..\.venv\Scripts\pythonw.exe"
if not defined PYW echo screenplink.exe is not beside this file and no virtual environment was found.
if not defined PYW echo From the repository: python -m venv app\.venv
if not defined PYW echo then: app\.venv\Scripts\python.exe -m pip install -r app\requirements.txt
if not defined PYW pause & exit /b 1
rem pythonw, not python: a console window would sit lit on the monitor the camera watches.
start "" "%PYW%" "%APP%\src\lsa\tools\camera_lab.py" --target %*
