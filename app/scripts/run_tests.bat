@echo off
REM Everything here is synthetic: no camera, no display needed.
REM
REM Measured 2026-10-15: the whole suite is about twelve seconds and never
REM touches the target screen -- it runs under QT_QPA_PLATFORM=offscreen with
REM auto_connect off. One live gate launch is about seventy seconds of his
REM monitor. So the answer to "do you need all those tests" is more of the
REM cheap kind, so fewer of the expensive kind are needed.
REM
REM Set LSA_NOPAUSE=1 to run this from a script. `pause` at the end used to
REM be unconditional, which silently hung every automated run.
REM
REM An ENVIRONMENT VARIABLE, not a /nopause switch: Git Bash rewrites any
REM argument that looks like a Unix path, so `/nopause` arrived as
REM "C:/Program Files/Git/nopause" and the comparison died with
REM "was unexpected at this time" -- after every suite had passed, so the
REM runner reported exit 255 on a completely green run. A flag that only
REM works from one shell is a check that fails for a reason it cannot name.
REM /nopause is still accepted when it survives the trip.
cd /d "%~dp0.."
set QT_QPA_PLATFORM=offscreen
set FAILED=

echo === the RULES, executing ===
REM RULES.md is 579 lines of prose and I cannot reliably re-read my own
REM prose: rewriting it, I dropped six rules that were in my own first
REM draft. These are the ones that can be checked by a machine.
".venv\Scripts\python.exe" "tests\test_rules.py"
if errorlevel 1 set FAILED=%FAILED% rules
echo.

echo === the Director: one run slot, so interleaving cannot happen ===
".venv\Scripts\python.exe" "tests\test_director.py"
if errorlevel 1 set FAILED=%FAILED% director
echo.

echo === the measurement routines, stepped branch by branch ===
REM The dot-radius ratchet lived here: its whole effect is that the app's own
REM verdict reads BETTER than the truth, so every other instrument in this
REM project goes greener as it gets worse.
".venv\Scripts\python.exe" "tests\test_routines.py"
if errorlevel 1 set FAILED=%FAILED% routines
echo.

echo === and every one of its checks can actually fail ===
REM This project has shipped four checks that could not fail the thing they
REM checked. The first run of this found three of its own.
".venv\Scripts\python.exe" "tests\mutate_director.py"
if errorlevel 1 set FAILED=%FAILED% mutation
echo.

echo === the startup splash: scaled, never trapping, never up during a measurement ===
".venv\Scripts\python.exe" "tests\test_splash.py"
if errorlevel 1 set FAILED=%FAILED% splash
echo.

echo === detector and shot tracker ===
".venv\Scripts\python.exe" "tests\test_detector.py"
if errorlevel 1 set FAILED=%FAILED% detector
echo.

echo === UI smoke test (renders the target screen to target_screen.png) ===
".venv\Scripts\python.exe" "tests\test_ui_smoke.py"
if errorlevel 1 set FAILED=%FAILED% ui_smoke
echo.

echo === arcade modes (synthetic shots, no camera) ===
".venv\Scripts\python.exe" "tests\test_modes.py"
if errorlevel 1 set FAILED=%FAILED% modes
echo.

echo === auto-alignment (recovers a drifted mapping without any shots) ===
".venv\Scripts\python.exe" "tests\test_align.py"
if errorlevel 1 set FAILED=%FAILED% align
echo.

echo === feedback: does the app say what it can and cannot see ===
".venv\Scripts\python.exe" "tests\test_feedback.py"
if errorlevel 1 set FAILED=%FAILED% feedback
echo.

echo === choosing the exposure by what it does to HIS shots ===
".venv\Scripts\python.exe" "tests\test_exposure.py"
if errorlevel 1 set FAILED=%FAILED% exposure
echo.
echo === the figure artwork: loader, scoring zones, randomness, sprite ===
".venv\Scripts\python.exe" "tests\test_artwork.py"
if errorlevel 1 set FAILED=%FAILED% artwork
echo.

echo === the brightness ceiling: every mode, every figure, measured from the artwork ===
".venv\Scripts\python.exe" "tests\test_ceiling.py"
if errorlevel 1 set FAILED=%FAILED% ceiling
echo.

echo === the figure drills (modes 7, 8, 9): size, scoring, randomness, hit feedback ===
".venv\Scripts\python.exe" "tests\test_figures.py"
if errorlevel 1 set FAILED=%FAILED% figures
echo.

echo === and every one of THEIR checks can actually fail ===
".venv\Scripts\python.exe" "tests\mutate_figures.py"
if errorlevel 1 set FAILED=%FAILED% mutate_figures
echo.

echo === and the area gate on the detector can actually fail ===
".venv\Scripts\python.exe" "tests\mutate_detector.py"
if errorlevel 1 set FAILED=%FAILED% mutate_detector
echo.

echo === the replay tool cannot silently prove nothing (no gain map, wrong log, wrong units) ===
".venv\Scripts\python.exe" "tests\test_replay.py"
if errorlevel 1 set FAILED=%FAILED% replay
echo.

echo === where a shot is reliably detected (the play-area border) ===
".venv\Scripts\python.exe" "tests\test_playarea.py"
if errorlevel 1 set FAILED=%FAILED% playarea
echo.

echo === the startup verdict: does it say what is wrong and what to do ===
".venv\Scripts\python.exe" "tests\test_selfcheck.py"
if errorlevel 1 set FAILED=%FAILED% selfcheck
echo.

echo === the ghost score: a perfect shooter at this system's pace, and the phantom guard ===
".venv\Scripts\python.exe" "tests\test_ghost.py"
if errorlevel 1 set FAILED=%FAILED% ghost
echo.

echo === and every one of THEIR checks can actually fail (the phantom guard, the score) ===
".venv\Scripts\python.exe" "tests\mutate_ghost.py"
if errorlevel 1 set FAILED=%FAILED% mutate_ghost
echo.

echo === ...and that the audio checks can fail ===
".venv\Scripts\python.exe" "tests\mutate_audio.py"
if errorlevel 1 set FAILED=%FAILED% mutate_audio
echo.

echo === the mark watchdog: a bang was heard and nothing appeared ===
".venv\Scripts\python.exe" "tests\test_watchdog.py"
if errorlevel 1 set FAILED=%FAILED% watchdog
echo.

echo === the passive screen watch: is our own picture as bright as it was ===
".venv\Scripts\python.exe" "tests\test_screenwatch.py"
if errorlevel 1 set FAILED=%FAILED% screenwatch
echo.

echo === pattern calibration against a simulated camera ===
".venv\Scripts\python.exe" "tests\test_pattern_cal.py"
if errorlevel 1 set FAILED=%FAILED% pattern_cal
echo.

echo === spoken announcements: mute, strings, staleness, no voice files ===
".venv\Scripts\python.exe" "tests\test_voice.py"
if errorlevel 1 set FAILED=%FAILED% voice
echo.

echo === no test can open a sound device ===
".venv\Scripts\python.exe" "tests\test_silent_audio.py"
if errorlevel 1 set FAILED=%FAILED% silent_audio
echo.

echo === the fall sound, and + / - for the volume ===
".venv\Scripts\python.exe" "tests\test_volume_fall.py"
if errorlevel 1 set FAILED=%FAILED% volume_fall
echo.

echo === listening THROUGH our own bang, and the one path to the room ===
".venv\Scripts\python.exe" "tests\test_audio_own.py"
if errorlevel 1 set FAILED=%FAILED% audio_own
echo.

echo === the string reader (tools\read_string.py) reads the log the app writes ===
".venv\Scripts\python.exe" "tests\test_shotreader.py"
if errorlevel 1 set FAILED=%FAILED% shotreader
echo.

echo === the shoot-to-select menu: tiles, the deferred launch, the pointer ===
REM Key M. Every tile comes from the mode REGISTRY, so a drill added next
REM year gets one with no change here. The drain -- a shot or a click on a
REM tile starting that drill through the app's own _select_mode -- is driven
REM through a real CameraLab, because a tile that silently swallows a shot
REM is indistinguishable from a dead detector at the firing line.
".venv\Scripts\python.exe" "tests\test_menu.py"
if errorlevel 1 set FAILED=%FAILED% menu
echo.

echo === the MCP face: stdlib only, a thin client of the introspection server ===
REM `screenplink.exe --mcp`. Driven as a real child process against a stub
REM HTTP app. Guards the seam with introspect/handlers.py (every route a tool,
REM descriptions verbatim), stdout carrying nothing but protocol, and the app
REM not running being answered inside two seconds with how to start it.
".venv\Scripts\python.exe" "tests\test_mcp.py"
if errorlevel 1 set FAILED=%FAILED% mcp
echo.

echo === the camera doctor, against a STUB camera (never real hardware) ===
REM Doctor.cmd (screenplink.exe --doctor --all) is the first instruction a stranger is given.
REM Driven against stand-ins whose faults are parameters: an ignored exposure, a
REM crushed gain, a sleeping display, a blind camera, a virtual device, a hung driver.
".venv\Scripts\python.exe" "tests\test_doctor.py"
if errorlevel 1 set FAILED=%FAILED% doctor
echo.

echo === the introspection server: off by default, read-only, no control writes ===
REM The in-process debug server (--introspect / LSA_INTROSPECT). Every
REM endpoint is driven headless against a stand-in lab, the real capture
REM thread's write methods are booby-trapped to prove none is reachable,
REM and section 11 measures what the server costs the capture loop --
REM the detector has 14 ms before 60 fps becomes 30.
".venv\Scripts\python.exe" "tests\test_introspect.py"
if errorlevel 1 set FAILED=%FAILED% introspect
echo.

echo === and every one of THOSE checks can actually fail ===
REM Eighteen faults put back on purpose. The first run caught 15 of 18,
REM and one of the three misses was a wrong claim in my own docstring:
REM the shot ring was queued because of the SENDER, not the receiver.
".venv\Scripts\python.exe" "tests\mutate_introspect.py"
if errorlevel 1 set FAILED=%FAILED% mutate_introspect
echo.

echo ============================================================
if "%FAILED%"=="" (
    echo ALL SUITES PASSED
    echo.
    echo NOT run here, and it is the one that finds the faults that matter:
    echo     .venv\Scripts\python.exe tests\test_coldstart.py
    echo The whole startup chain against a simulated camera, seven launches,
    echo several minutes. Unit tests pass while the app is broken, because
    echo every fault worth finding lives in the interaction between routines
    echo that share one screen, one camera and one detector.
) else (
    echo FAILED SUITES:%FAILED%
)
echo ============================================================

if defined LSA_NOPAUSE goto :done
echo %1 | findstr /i /c:"nopause" >nul && goto :done
pause
:done
if not "%FAILED%"=="" exit /b 1
exit /b 0
