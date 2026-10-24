@echo off
rem Tests the cameras on this machine. CLOSE ScreenPlink FIRST: two programs on
rem one camera can leave the driver in a bad state.
rem
rem   Doctor.cmd                  test every camera           (--all)
rem   Doctor.cmd --index 2        test one camera
rem   Doctor.cmd --index 2 --frame aim.png    save one picture, for aiming
rem
rem ONE FILE, TWO PLACES. In a release it runs screenplink.exe beside it. In the
rem repository (app\scripts) it runs the doctor from source with the virtual
rem environment (app\.venv, else .venv beside app). No labels or GOTOs on purpose: a
rem .cmd saved with bare LF line endings can miss a label.
rem
rem screenplink.exe has no console of its own (a console window would sit lit on
rem the monitor the camera watches), so the report is written to doctor-report.txt
rem and then shown. That file is the report: Claude Code can read it directly.
setlocal
rem Mandatory, and before OpenCV is imported (see src/lsa/_env.py).
set "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0"
cd /d "%~dp0"
set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--all"
set "DOCTOR="
if exist "%~dp0screenplink.exe" set DOCTOR="%~dp0screenplink.exe" --doctor
for %%I in ("%~dp0..") do set "APP=%%~fI"
set "PY="
if not defined DOCTOR if exist "%APP%\.venv\Scripts\python.exe" set "PY=%APP%\.venv\Scripts\python.exe"
if not defined DOCTOR if not defined PY if exist "%APP%\..\.venv\Scripts\python.exe" set "PY=%APP%\..\.venv\Scripts\python.exe"
if not defined DOCTOR if not defined PY echo screenplink.exe is not beside this file and no virtual environment was found. & exit /b 1
if not defined DOCTOR set "PYTHONPATH=%APP%\src"
if not defined DOCTOR set DOCTOR="%PY%" -m lsa.doctor
%DOCTOR% %ARGS% > doctor-report.txt 2>&1
set "RC=%ERRORLEVEL%"
type doctor-report.txt
for %%F in (doctor-report.txt) do if %%~zF==0 echo (no report was written - the doctor printed nothing)
echo.
echo Report saved in %~dp0doctor-report.txt
exit /b %RC%
