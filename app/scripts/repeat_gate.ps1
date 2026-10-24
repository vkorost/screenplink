# THE REPEATABILITY GATE.
#
# "Every time you tell me you are done with something and waiting for a final
#  test, I show up and then things are not ready. Looks like we are running in
#  circles without improving much."
#
# He is right, and the cure is not better fixes. The chain is thirteen stages
# in series -- enumerate, open, expose, find the screen, mask, eighteen
# markers, fit, noise floor, thresholds, frame rate, group frames, map, draw.
# At 95% reliability each that is a 51% chance of an end-to-end success. So
# fixing one stage and reporting progress reliably lands him in front of the
# NEXT weakest link, which is exactly what happened a dozen times in one day.
#
# This runs the whole thing N times from COLD -- a fresh process and a fresh
# camera open every time, because that is what he does -- and every launch
# must pass the same fixed checklist. Nothing here is judged by eye.
#
# A pass needs ALL of:
#     camera opens and streams
#     pattern calibration held-out error under 2.0 mm
#     the self test detects every emulated shot it fires
#     the startup verdict says READY
#     frame rate above 25 fps, and REPORTED -- an unread check is not a pass
#     no pulses discarded for being too short
#     the app closes CLEANLY  (a force-kill wedges this camera: measured
#                              40.3s to open and 0 frames afterwards, against
#                              0.44s and 53.6 fps after a clean close)
#
# Usage:   .\repeat_gate.ps1            five launches
#          .\repeat_gate.ps1 -Runs 3    fewer, while iterating
#
# Do not report readiness on anything less than every run passing.

param(
    [int]$Runs = 5,
    [int]$TimeoutSeconds = 150
)

$ErrorActionPreference = 'Stop'
# THE PROJECT ROOT IS ONE LEVEL UP: these scripts live in scripts since
# 2026-10-18, when he asked for the working folder cleared of loose files.
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$HELD_OUT_MM_MAX = 2.0
$FPS_MIN = 25.0
$WRITE_BUDGET = 24

# The app prints one machine-readable line at the end of its self check, on
# purpose, so this script never has to regex prose. Its first version did
# exactly that and silently scored launches without a frame rate, because the
# frame rate was never written to the log at all -- it lived only in the GUI.
# FIELDS BY NAME, not by position.
#
# The first version pinned the exact order and the exact prefix:
# 'GATE verdict=(\S+) detected=...'. Adding `role=` and `screen=` in front of
# `verdict=` -- to stop a harness line ever being read as his hardware again
# -- made this regex match NOTHING, so the gate would have scored every
# launch "no calibration; no self test; frame rate not reported; control
# writes not reported" and reported 0 of 5 on a working app.
#
# Caught by an outside review before it ran, not by me. It is the fault this
# whole contract exists to prevent, in the consumer rather than the producer:
# a test of the producer that never runs the consumer. `tests/test_rules.py`
# section H now feeds a real GATE line to this parser.
#
# Named fields also mean the producer may add a column without breaking the
# gate again, which is the actual lesson.
function Get-GateField {
    param([string]$Text, [string]$Name)
    $m = [regex]::Match($Text, "\bGATE\b[^`r`n]*?\b$Name=(\S+)")
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

$GATE_REQUIRED = @('verdict', 'detected', 'held_out_mm', 'error_mm', 'fps',
                   'margin', 'floor', 'bar', 'exposure', 'rejected_short',
                   'writes', 'coupling', 'coupling_source')
# Enough to recognise a GATE line at all, for the polling loop.
$GATE_LINE = 'GATE\b.*\bverdict='

# THE JUDGEMENT, AS A FUNCTION, so something other than a live launch can run it.
#
# `tools/status.py` reports on these same logs, and it counted only
# `verdict=ready` while this applies a dozen more criteria: it printed
# "5 of 5 reached READY ... the only state in which the word 'ready' may be
# used" over a log whose self test read detected=11/12, which this gate FAILS.
# Two readers of one format disagreeing (RULES C14), and the looser one was
# the one that hands the next session permission to say "ready".
#
# The criteria used to be inline in the launch loop, where no test could
# reach them. `tests/test_rules.py` section H extracts this function by AST,
# runs it on crafted GATE lines, and requires status.py to reach the same
# verdict on every one. Change a criterion HERE and that check tells you
# which reader you forgot.
#
# Takes the log text, whether the launch reached a verdict, and whether the
# app closed cleanly -- the three things that vary between a launch and a
# crafted line. Returns the verdict and the reasons; prints nothing.
function Get-LaunchJudgement {
    param([string]$Text, [bool]$Done, [bool]$CleanClose,
          [int]$TimeoutSeconds = 150)

    $heldOut = $null; $detected = $null; $fired = $null
    $fps = $null; $verdict = 'none'; $margin = $null; $rejShort = $null
    $writes = $null
    $coupling = $null; $couplingSource = $null
    $role = $null; $screenSize = $null
    if ($text -match $GATE_LINE) {
        # BY NAME. A missing field stays $null and is reported as a FAILURE
        # below, which is the contract: an unreported check is a failure,
        # not a pass by omission.
        $verdict  = Get-GateField $text 'verdict'
        $det      = Get-GateField $text 'detected'
        if ($det -match '^(\d+)/(\d+)$') {
            $detected = [int]$Matches[1]
            $fired    = [int]$Matches[2]
        }
        $v = Get-GateField $text 'held_out_mm'
        if ($null -ne $v) { $heldOut = [double]$v }
        $v = Get-GateField $text 'fps'
        if ($null -ne $v) { $fps = [double]$v }
        $v = Get-GateField $text 'margin'
        if ($null -ne $v) { $margin = [double]$v }
        $v = Get-GateField $text 'rejected_short'
        if ($null -ne $v) { $rejShort = [int]$v }
        $v = Get-GateField $text 'writes'
        if ($null -ne $v) { $writes = [int]$v }
        $coupling = Get-GateField $text 'coupling'
        $couplingSource = Get-GateField $text 'coupling_source'
        if ($null -ne $heldOut -and $heldOut -lt 0) { $heldOut = $null }
        $role = Get-GateField $text 'role'
        $screenSize = Get-GateField $text 'screen'
    }

    $reasons = @()
    if (-not $done) { $reasons += "never reached a verdict in ${TimeoutSeconds}s" }
    # THIS MUST BE HIS HARDWARE, and the line says so itself.
    #
    # The app writes role= and screen= because a harness log line was once
    # read as a measurement of his bench and reported to him as one: the
    # giveaways, an 800x800 screen and exactly 60.0 fps, were inside the
    # line. A gate that scores a simulated launch as a live one is the same
    # mistake with a script doing it.
    if ($null -eq $role) {
        $reasons += 'the line does not say which process produced it'
    } elseif ($role -ne 'live') {
        $reasons += "this is a ${role} launch, not his hardware"
    }
    # The fast path keeps a mapping confirmed by the two-marker check and
    # does not refit, so there is no FRESH held-out error to report. That
    # is the fast path working, not a launch that failed to calibrate --
    # the first version of this check failed those launches, which would
    # have scored the three-second path as broken and sent the next
    # session to "fix" it.
    $confirmed = (Get-GateField $text 'mapping') -eq 'confirmed'
    if ($null -eq $heldOut -and -not $confirmed) { $reasons += 'no calibration' }
    elseif ($heldOut -gt $HELD_OUT_MM_MAX) { $reasons += "calibration ${heldOut}mm" }
    if ($null -eq $detected) { $reasons += 'no self test' }
    elseif ($detected -lt $fired) { $reasons += "self test $detected/$fired" }
    # An unmeasured frame rate is a FAILURE, not a pass by omission. A gate
    # that skips a check it could not read is a gate that passes anything it
    # cannot see.
    if ($null -eq $fps) { $reasons += 'frame rate not reported' }
    elseif ($fps -lt $FPS_MIN) { $reasons += "fps $fps" }
    # A too-short rejection only matters if it COST a detection, and the
    # rate check above already measures that directly. One pulse
    # fragmenting into two tracks, where the dot was still detected, is
    # noise in the instrument rather than a lost shot -- and failing a
    # launch for it would be the same mistake as the margin heuristic
    # overruling the rate it approximates. The threshold catches the
    # failure this check exists for: a frame rate so low that pulses
    # are routinely discarded, which cost 24% of his real trigger pulls.
    if ($null -ne $rejShort -and $null -ne $fired -and $fired -gt 0 -and
        $rejShort -gt [math]::Floor($fired / 4)) {
        $reasons += "$rejShort of $fired pulses discarded as too short"
    }
    # A BUDGET ON CONTROL WRITES, because the gate wedged the camera
    # three times on 2026-10-15 with its own traffic. A single exposure
    # write has hung this device for 272 seconds; five minutes of pure
    # streaming returned 17,945 frames with zero failures. The probe's
    # brightness auto-range was silently re-writing the same exposure on
    # every rung -- up to seven redundant transactions a launch, on top
    # of the five the measurement needs -- and nothing could see it,
    # because the number was never reported anywhere.
    #
    # A budget you cannot see is not a budget. 24 covers two presets, a
    # calibration swap and a three-candidate probe, with room to spare.
    if ($null -eq $writes -or $writes -lt 0) { $reasons += 'control writes not reported' }
    elseif ($writes -gt $WRITE_BUDGET) { $reasons += "$writes control writes (budget $WRITE_BUDGET)" }
    # CAN THE MICROPHONE HEAR THE APP'S OWN BANG, AS A MEASUREMENT. With the
    # coupling known the listener discounts the bang by what it measured; with
    # it unknown the flat self-mute applies and the one independent counter of
    # his trigger pulls goes deaf after every shot -- silently, because the
    # launch still reads READY. Unreported fails, like every other check here.
    # `coupling_source` is what says WHERE the number came from (a real bang
    # or the startup probe), and `known` with no source is a producer bug.
    if ($null -eq $coupling) { $reasons += 'microphone coupling not reported' }
    elseif ($coupling -ne 'known') { $reasons += 'microphone coupling not known (the flat self-mute applies)' }
    elseif ($null -eq $couplingSource -or $couplingSource -eq 'none') { $reasons += 'microphone coupling source not reported' }
    if ($verdict -ne 'ready') { $reasons += "verdict: $verdict" }
    if (-not $cleanClose) { $reasons += 'DID NOT CLOSE CLEANLY' }


    return [pscustomobject]@{
        Pass     = ($reasons.Count -eq 0)
        Reasons  = @($reasons)
        HeldOut  = $heldOut
        Detected = $detected
        Fired    = $fired
        Fps      = $fps
        Margin   = $margin
        Writes   = $writes
        Verdict  = $verdict
    }
}

$results = @()

for ($i = 1; $i -le $Runs; $i++) {
    Write-Host ""
    Write-Host "=== LAUNCH $i of $Runs ===" -ForegroundColor Cyan

    $logdir = Join-Path $root "logs"
    $log = Join-Path $logdir "camera_lab.log"
    # ROTATE, never delete (see run_live.ps1): a deleted log is evidence gone.
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

    $started = Get-Date
    Start-Process -FilePath (Join-Path $root "scripts\run_camera_lab.bat") `
                  -ArgumentList "--target", "--selftest" `
                  -WorkingDirectory $root -WindowStyle Minimized | Out-Null

    # Poll for the contract line rather than sleeping a fixed time: a launch
    # that finishes in 50s should not cost 150, and one that hangs must still
    # be caught.
    $deadline = $started.AddSeconds($TimeoutSeconds)
    $done = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        if (Test-Path $log) {
            $text = Get-Content $log -Raw -ErrorAction SilentlyContinue
            if ($text -and $text -match '=== VERDICT ===') { $done = $true; break }
        }
    }
    $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)

    # ALWAYS close it the app's own way, pass or fail. A gate that force-kills
    # on a bad run poisons the camera for the run after it, and would then be
    # measuring its own damage.
    $closed = & (Join-Path $root "scripts\stop_lab.ps1") 2>&1 | Out-String
    $cleanClose = ($closed -match 'closed cleanly')

    $text = if (Test-Path $log) { Get-Content $log -Raw } else { "" }

    $j = Get-LaunchJudgement -Text $text -Done $done -CleanClose $cleanClose `
                             -TimeoutSeconds $TimeoutSeconds
    $heldOut = $j.HeldOut; $detected = $j.Detected; $fired = $j.Fired
    $fps = $j.Fps; $margin = $j.Margin; $writes = $j.Writes
    $verdict = $j.Verdict; $reasons = $j.Reasons; $pass = $j.Pass
    $results += [pscustomobject]@{
        Run = $i; Pass = $pass; Seconds = $elapsed
        HeldOutMm = $heldOut; SelfTest = "$detected/$fired"
        Fps = $fps; Margin = $margin; Writes = $writes; Verdict = $verdict
        Why = ($reasons -join '; ')
    }

    if (Test-Path $log) { Copy-Item $log (Join-Path $logdir "gate.run$i.log") -Force }

    # MEASURE THE INSTRUMENT BETWEEN MEASUREMENTS.
    #
    # A wedged Kiyo Pro does not recover between launches -- it needs a
    # physical replug -- so every run after a stall measures the stall and
    # not the app. On 2026-10-15 that produced four identical failures
    # that looked like four faults and were one, and spent ten minutes of
    # his screen doing it. He put it plainly: "I just want to avoid you
    # running this on an infinite loop."
    #
    # The first version decided this by grepping the app's log for stall
    # messages, which is a guess about the instrument rather than a
    # measurement of it -- and it guessed wrong in BOTH directions within
    # an hour: it aborted a healthy sweep on a shutdown warning that was
    # survivable, and it carried on through four launches against a camera
    # that was already dead.
    #
    # So the gate now opens the camera itself and counts frames. No
    # control writes -- those are what hang it; streaming is safe.
    # Exit code, not text. OpenCV writes MSMF warnings to stderr, and in
    # Windows PowerShell 5.1 redirecting a native command's stderr wraps
    # each line in an ErrorRecord -- which, under ErrorActionPreference
    # 'Stop', killed the gate on launch 1 over a warning the check itself
    # had already accounted for.
    # OpenCV writes MSMF warnings to stderr, and Windows PowerShell 5.1
    # turns a native command's stderr into ErrorRecords -- which, under
    # ErrorActionPreference 'Stop', KILLS THIS SCRIPT. It did, twice,
    # mid-run, leaving the app running with no one to close it. The camera
    # was wedged by the cleanup that never happened, and the wedge was then
    # blamed on the camera.
    #
    # A harness that aborts without tidying up is worse than no harness: it
    # damages the thing it measures and then reports the damage as data.
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
        Write-Host "  THE CAMERA IS WEDGED. Stopping the gate here." -ForegroundColor Yellow
        Write-Host "  Every launch after this measures the wedge, not the app."
        Write-Host "  It needs a physical replug before anything here means anything."
        break
    }

    if ($pass) {
        Write-Host ("  PASS  {0}s  calib {1}mm  self test {2}/{3}  {4} fps  {5}x margin" -f `
                    $elapsed, $heldOut, $detected, $fired, $fps, $margin) -ForegroundColor Green
    } else {
        Write-Host ("  FAIL  {0}s  -- {1}" -f $elapsed, ($reasons -join '; ')) -ForegroundColor Yellow
    }

    # Let the camera settle between launches. It is a USB device measured to
    # hang on control traffic, and back-to-back opens are not what a person
    # does.
    if ($i -lt $Runs) { Start-Sleep -Seconds 6 }
}

Write-Host ""
Write-Host "=== THE GATE ===" -ForegroundColor Cyan
$results | Format-Table Run, Pass, Seconds, HeldOutMm, SelfTest, Fps, Margin, Writes, Why -AutoSize |
    Out-String | Write-Host

# @() so that a single passing run is a one-element array, not a scalar whose
# .Count prints nothing. It printed nothing the first time this ran.
$passed = @($results | Where-Object { $_.Pass }).Count
$attempted = @($results).Count
Write-Host ("$passed of $attempted launches passed" +
            $(if ($attempted -lt $Runs) { " ($Runs asked for; stopped early)." } else { "." }))
if ($passed -eq $Runs -and $attempted -eq $Runs) {
    Write-Host "GATE PASSED -- and only now may the word 'ready' be used." -ForegroundColor Green
    exit 0
} else {
    Write-Host "GATE NOT PASSED. Do not report readiness." -ForegroundColor Yellow
    exit 1
}
