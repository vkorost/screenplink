# Run the test suites BY AREA AFFECTED, not as a full regression every time.
#
# The owner, 2026-10-21: "work on the test suites to make sure that they are
# modular and invoked according to the areas affected by code; full regression
# is invoked after considerable amount of changes or after an invasive change".
#
#   .\run_affected.ps1                  ALWAYS + AFFECTED for the working tree
#   .\run_affected.ps1 -Staged          the same, for what is staged
#   .\run_affected.ps1 -Files a.py,b.py an explicit set of changed files
#   .\run_affected.ps1 -Full            everything, including the slow tier
#   .\run_affected.ps1 -Slow            also run the slow runners the change EARNED
#
# THE TIERS (the planner is tools\suite_plan.py; this script only executes it):
#   ALWAYS    tests\test_rules.py. About 12 s. It stops a rule being lost.
#   AFFECTED  whatever tools\suites_for.py names from the IMPORT GRAPH for the
#             changed files. Nothing is a typed file-to-suite table.
#   SLOW      earned, never automatic. COST OF EACH, MEASURED on this machine:
#               mutate_detector     245 s   (measured 2026-10-21, by the owner)
#               mutate_figures      709 s   (measured 2026-10-21, by the owner)
#               mutate_introspect    77 s   (measured 2026-10-21, by the owner)
#               mutate_ghost          4 s   (measured 2026-10-21, this session)
#               test_coldstart      ~480 s  ("about 8 minutes", the owner)
#               mutate_director     not measured (the brief forbade running it)
#               mutate_audio        not measured (written 2026-10-23 without a shell)
#             Full regression = ~270 s of fast suites + ~1500 s of the above.
#
# AN INVASIVE CHANGE ESCALATES TO -Full, AND THIS SCRIPT SAYS SO. Touching any of
#     src/lsa/detect/  calibrate/  camera/  director.py  routines.py
#     exposure.py  settings.py  selfcheck.py
# is a core change where a fault is silent and expensive, so a subset is not
# trusted. So is a LARGE change: more than 18 files or 1700 lines, the p90 of
# this repo's last 157 commits (so about one change in ten). The lists live in
# tools\suite_plan.py and tests\test_rules.py section S fails if they rot.
#
# EVERY RUN PRINTS what it ran, what it SKIPPED and WHY, and the wall time. A
# gate that silently skips what it cannot see passes anything it cannot see, so
# "skipped" is never allowed to read as "passed". Exit code is non-zero if any
# suite that RAN failed, or if the planner itself failed; the failures are named.
#
# NOT a replacement for scripts\run_tests.bat, which stays the full regression.

param(
    [switch]$Staged,
    [string[]]$Files,
    [switch]$Full,
    [switch]$Slow
)

$ErrorActionPreference = 'Stop'
# THE PROJECT ROOT IS ONE LEVEL UP: moving scripts broke the launch twice on
# 2026-10-18 for resolving it from the wrong place.
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Host "Virtual environment missing: $py"; exit 2 }

# As scripts\run_tests.bat does: no screen, no pause.
$env:QT_QPA_PLATFORM = 'offscreen'
$env:LSA_NOPAUSE = '1'
$env:PYTHONIOENCODING = 'utf-8'

$clock = [System.Diagnostics.Stopwatch]::StartNew()

# -- 1. the plan --------------------------------------------------------------
if ($Full) { $planArgs = @('--full') }
elseif ($Files) { $planArgs = @('--files', ($Files -join ',')) }
elseif ($Staged) { $planArgs = @('--staged') }
else { $planArgs = @('--changed') }

$raw = & $py (Join-Path $root 'tools\suite_plan.py') @planArgs
if ($LASTEXITCODE -ne 0 -or -not $raw) {
    Write-Host "PLANNER FAILED (exit $LASTEXITCODE). Nothing was run, so nothing passed."
    exit 2
}
$plan = ($raw -join "`n") | ConvertFrom-Json

Write-Host "=== run_affected: $($planArgs -join ' ') ==="
Write-Host "changed: $(@($plan.paths).Count) file(s), $($plan.lines) line(s)"
foreach ($n in @($plan.notes)) { if ($n) { Write-Host "note: $n" } }
if ($plan.escalated) {
    Write-Host ""
    Write-Host "ESCALATING TO THE FULL REGRESSION, because:"
    foreach ($r in @($plan.reasons)) { Write-Host "  - $r" }
    Write-Host ""
}

# -- 2. decide what runs ------------------------------------------------------
$toRun = @()
foreach ($r in @($plan.run)) { $toRun += $r }
$skipped = @()
foreach ($s in @($plan.skipped)) { $skipped += $s }
if ($Slow -and -not $plan.escalated) {
    foreach ($e in @($plan.earned_slow)) {
        $hit = $skipped | Where-Object { $_.name -eq $e.name }
        if ($hit) {
            $skipped = @($skipped | Where-Object { $_.name -ne $e.name })
            $toRun += [pscustomobject]@{ name = $e.name; tier = 'slow'; why = "-Slow, " + $e.why }
        }
    }
}

# -- 3. run them, ONE AT A TIME (parallel runs fail wall-clock checks falsely) -
$results = @()
foreach ($r in $toRun) {
    $name = $r.name
    $log = Join-Path $env:TEMP ("run_affected_" + $name + ".log")
    $err = $log + ".err"
    $t = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host ("running  {0,-24} [{1}] {2}" -f $name, $r.tier, $r.why)
    $p = Start-Process -FilePath $py -ArgumentList @((Join-Path $root "tests\$name")) `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $t.Stop()
    $rc = $p.ExitCode
    $text = @()
    if (Test-Path $log) { $text += Get-Content $log }
    $summary = ($text | Where-Object { $_ -match '(?i)\d+\s+(rule\s+|routine\s+|mcp\s+|screen-watch\s+)?checks?|ALL CHECKS|checks passed|FAILED' } |
        Select-Object -Last 1)
    $results += [pscustomobject]@{ name = $name; rc = $rc; secs = [math]::Round($t.Elapsed.TotalSeconds, 1); summary = "$summary".Trim() }
    if ($rc -ne 0) {
        Write-Host "  FAILED (exit $rc) in $([math]::Round($t.Elapsed.TotalSeconds,1)) s:"
        $text | Where-Object { $_ -match 'FAIL' } | Select-Object -First 12 | ForEach-Object { Write-Host "    $_" }
        if (Test-Path $err) { Get-Content $err | Select-Object -Last 6 | ForEach-Object { Write-Host "    stderr: $_" } }
        Write-Host "  full output: $log"
    }
}

# -- 4. the report: ran, skipped, and why -------------------------------------
$clock.Stop()
Write-Host ""
Write-Host "================ RAN ($($results.Count)) ================"
foreach ($x in $results) {
    $st = if ($x.rc -eq 0) { 'ok  ' } else { 'FAIL' }
    Write-Host ("{0} {1,-24} {2,7} s   {3}" -f $st, $x.name, $x.secs, $x.summary)
}
Write-Host ""
Write-Host "================ SKIPPED ($($skipped.Count)) -- NOT run, so NOT passed ================"
foreach ($s in $skipped) { Write-Host ("skip {0,-24} {1}" -f $s.name, $s.why) }
if (@($plan.unmapped).Count -gt 0) {
    Write-Host ""
    Write-Host "NO SUITE KNOWS ABOUT THESE (a change here is unguarded):"
    foreach ($u in @($plan.unmapped)) { Write-Host "  $u" }
}
if ($Slow -eq $false -and @($plan.earned_slow).Count -gt 0 -and -not $plan.escalated) {
    Write-Host ""
    Write-Host "EARNED BUT NOT RUN (pass -Slow, or -Full before pushing):"
    foreach ($e in @($plan.earned_slow)) { Write-Host "  $($e.name)  $($e.why)" }
}

$failed = @($results | Where-Object { $_.rc -ne 0 } | ForEach-Object { $_.name })
Write-Host ""
Write-Host ("total wall time: {0:N1} s   ran {1}   skipped {2}   failed {3}" -f `
    $clock.Elapsed.TotalSeconds, $results.Count, $skipped.Count, $failed.Count)
if ($failed.Count -gt 0) {
    Write-Host "FAILED SUITES: $($failed -join ' ')"
    exit 1
}
if ($skipped.Count -gt 0) {
    Write-Host "ALL SUITES THAT RAN PASSED. $($skipped.Count) did not run (listed above); that is not a pass."
} else {
    Write-Host "ALL SUITES PASSED"
}
exit 0
