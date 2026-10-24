# Start the app for a LIVE run, and refuse to if the camera is not fit.
#
# Written after a guard that worked was thrown away by a pipe. The camera
# check correctly reported WEDGED, the shell discarded its exit code through
# `| tail -1`, and the launch went ahead against a dead camera -- costing a
# replug and a slice of his evening.
#
# That was the fourth check in one day that could not actually stop anything:
# a health check that could not tell wedged from healthy, a wait loop where
# silence looked like progress, a test fixture that silently did nothing, and
# this. The common shape is a check whose verdict has no authority. So the
# check and the launch live in one script where the refusal is the default
# path, rather than in a shell line where it depends on how the pipe is
# written.
#
#   .\run_live.ps1                  calibrate normally, then self test
#   .\run_live.ps1 -TrustCal        skip calibration entirely (camera unmoved)
#   .\run_live.ps1 -Record          also record the session for analysis
#   .\run_live.ps1 -TimeoutSec 150  how long to wait for a verdict

param(
    [switch]$TrustCal,
    [switch]$Record,
    [switch]$ModeTest,
    [switch]$Introspect,
    [int]$TimeoutSec = 150
)

$ErrorActionPreference = 'Stop'
# THE PROJECT ROOT IS ONE LEVEL UP: these scripts live in scripts since
# 2026-10-18, when he asked for the working folder cleared of loose files.
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

# -- 1. is the camera fit to be used? -------------------------------------
# -- 0. ONE INSTANCE. Refuse if the app is already running ----------------
#
# Two instances on one camera is what wedges it: the older one loses the
# device mid-stream, the newer one comes up BEHIND and photographs the
# desktop, and the pair of them produce a launch that looks like a camera
# fault. That happened four times on 2026-10-18 before this guard existed,
# and each time the fault was looked for in the app.
#
# MATCHED ON THE COMMAND LINE, not on "is any python running". The hook that
# prints the rules at session start is a python process that lives for about
# a second, and a check that counted it refused four launches on an idle
# machine. Measure the thing, not something adjacent to it.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
           Where-Object { $_.CommandLine -like '*camera_lab*' }
if ($running) {
    Write-Host ""
    Write-Host ("  ALREADY RUNNING: pid " + ($running.ProcessId -join ', ')) -ForegroundColor Yellow
    Write-Host "  Close it first (Esc or F11 on the target screen, or .\stop_lab.ps1)."
    Write-Host "  Starting a second one would take the camera off this one."
    Write-Host ""
    exit 3
}

$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $alive = (& (Join-Path $root '.venv\Scripts\python.exe') `
                (Join-Path $root 'tools\camera_alive.py') 2>$null) | Out-String
    $cameraOk = ($LASTEXITCODE -eq 0)
} catch {
    $alive = "camera check failed: $_"
    $cameraOk = $false
} finally {
    $ErrorActionPreference = $prevEAP
}
Write-Host ("  " + $alive.Trim())
if (-not $cameraOk) {
    Write-Host ""
    Write-Host "NOT LAUNCHING. The camera is wedged and needs a physical replug." -ForegroundColor Yellow
    Write-Host "Launching anyway wastes the replug: every step would measure the"
    Write-Host "stall instead of the app."
    exit 2
}

# -- 2. launch ------------------------------------------------------------
# Logs live in logs\, not in the working folder: his instruction, 2026-10-18,
# after 107 of them accumulated at the top level. The app creates the
# directory when it writes; this only reads and rotates.
$logdir = Join-Path $root 'logs'
$log = Join-Path $logdir 'camera_lab.log'
# ROTATE, never delete. The logs for the evening of 2026-10-16 -- the ones the
# four firing-line faults were reported from -- were deleted here, and every
# diagnosis made from them has been unverifiable since. The renamed file
# matches the camera_lab.*.log ignore pattern and tools/status.py's *.log glob.
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if (Test-Path $log) {
    Move-Item $log (Join-Path $logdir "camera_lab.$stamp.log") -Force
}
foreach ($f in @('recordings\verdict_screen.png', 'recordings\verdict_camera.png')) {
    $p = Join-Path $root $f
    if (Test-Path $p) {
        $dest = Join-Path (Split-Path -Parent $p) `
                    ("{0}.{1}.png" -f [IO.Path]::GetFileNameWithoutExtension($p), $stamp)
        Move-Item $p $dest -Force
    }
}

$flags = @('--target', '--selftest')
if ($TrustCal) { $flags += '--trust-cal' }
if ($Record)   { $flags += '--record' }
if ($ModeTest) { $flags += '--modetest' }
# The in-process debug server. Needed whenever the app has to be DRIVEN with
# nobody at the machine -- keys and clicks posted to the window, no focus
# taken, no cursor moved. Off unless asked for, like every diagnostic here.
if ($Introspect) { $flags += '--introspect' }

$started = Get-Date
Write-Host ("  launching: " + ($flags -join ' '))
Start-Process -FilePath (Join-Path $root 'scripts\run_camera_lab.bat') `
              -ArgumentList $flags -WorkingDirectory $root -WindowStyle Minimized | Out-Null

# -- 3. wait for an ANSWER, of any kind -----------------------------------
#
# Watching only for success is how a crash came to look identical to work in
# progress: the app died two seconds in and the wait sat there until a human
# interrupted it. Silence is not success.
$deadline = $started.AddSeconds($TimeoutSec)
$outcome = ''
while ((Get-Date) -lt $deadline -and -not $outcome) {
    Start-Sleep -Seconds 2
    $text = if (Test-Path $log) { Get-Content $log -Raw -ErrorAction SilentlyContinue } else { '' }
    if ($text -match '=== VERDICT ===')          { $outcome = 'VERDICT' }
    elseif ($text -match 'stopped delivering frames') { $outcome = 'CAMERA STALLED' }
    elseif ($text -match 'Traceback')            { $outcome = 'CRASHED' }
    elseif (-not (Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" | Where-Object { $_.CommandLine -like '*camera_lab*' })) {
        $outcome = 'PROCESS GONE'
    }
}
if (-not $outcome) { $outcome = "NO ANSWER IN ${TimeoutSec}s" }
$elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)

Write-Host ""
# PowerShell 5.1 has no if-EXPRESSION: an 'if' inside an argument is a parse
# error at runtime, which crashed this script AFTER it had already launched the
# app -- so the launch looked like a failure when it had worked (C14).
$colour = if ($outcome -eq 'VERDICT') { 'Green' } else { 'Yellow' }
Write-Host ("  $outcome after ${elapsed}s") -ForegroundColor $colour

if (Test-Path $log) {
    $v = Select-String -Path $log -Pattern '^\s*\d\d:\d\d:\d\d\s+(READY|NOT READY|USABLE)' |
         Select-Object -Last 1
    if ($v) { Write-Host ("  " + $v.Line.Trim()) }
}
foreach ($f in @('recordings\verdict_screen.png', 'recordings\verdict_camera.png')) {
    $p = Join-Path $root $f
    Write-Host ("  {0}: {1}" -f $f, $(if (Test-Path $p) { 'saved' } else { 'MISSING' }))
}
Write-Host ""
Write-Host "  The app is still running. Stop it with .\stop_lab.ps1"
if ($outcome -ne 'VERDICT') { exit 1 }
exit 0
