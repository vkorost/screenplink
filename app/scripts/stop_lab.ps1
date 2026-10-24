# Close Camera Lab gracefully so the camera is released the way the app would.
#
# Stop-Process -Force is TerminateProcess: no cleanup runs, the AVI is left with
# a zero frame count and no index, and -- far worse -- the camera is cut off
# mid-stream with its driver mid-transaction. On 2026-10-14 repeated force-kills
# took the whole USB root hub down with Code 43. On 2026-10-15 a force-kill left
# the Kiyo Pro enumerated and "OK" while delivering zero frames and taking 40
# seconds to open, recoverable only by a physical replug, with NOTHING in the
# Windows event log. Forcing is not a tidy-up. It is the thing that breaks the
# hardware.
#
# WHY THE OLD VERSION ALWAYS FORCED, measured 2026-10-15:
#
#   Get-Process pythonw | Select Id, MainWindowTitle
#     2956   Laser Shooting App - Target      <- window
#    17316   (none)                           <- no window at all
#
# Two faults at once.
#
# 1. `CloseMainWindow()` sends WM_CLOSE to whatever Windows calls the process's
#    MAIN window, and that was the fullscreen TARGET window. Closing it closes
#    only itself; the operator console carries `closeEvent` and never heard a
#    thing. The app sat there. These notes already record that
#    `MainWindowTitle` reports only one window per process and that an app with
#    two windows is invisible under whichever one loses -- the same trap,
#    costing the camera this time instead of a screenshot.
#
# 2. The venv's pythonw.exe is a launcher stub that spawns the real
#    interpreter, so there are always TWO processes and the parent has no
#    window for CloseMainWindow to reach. The old loop waited for ALL of them
#    to exit, which the parent only does after its child, and then force-killed
#    both at 5 seconds.
#
# Measured after this rewrite: WM_CLOSE to the Camera Lab window closes both
# processes in about a second, and the camera is released cleanly.

$ErrorActionPreference = 'Stop'

Add-Type @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public class Win {
    public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern IntPtr PostMessage(IntPtr h, uint msg, IntPtr w, IntPtr l);

    public const uint WM_CLOSE = 0x0010;

    // Every visible top-level window, with its owning process. Enumerating is
    // the only way to see BOTH of this app's windows: MainWindowTitle shows one.
    public static List<string> Windows() {
        var found = new List<string>();
        EnumWindows(delegate(IntPtr h, IntPtr p) {
            if (!IsWindowVisible(h)) return true;
            var sb = new StringBuilder(512);
            if (GetWindowTextW(h, sb, sb.Capacity) == 0) return true;
            uint pid; GetWindowThreadProcessId(h, out pid);
            found.Add(h.ToInt64() + "|" + pid + "|" + sb.ToString());
            return true;
        }, IntPtr.Zero);
        return found;
    }

    public static void Close(IntPtr h) { PostMessage(h, WM_CLOSE, IntPtr.Zero, IntPtr.Zero); }
}
'@

# WHICH PROCESSES ARE OURS: the ones that OWN A ScreenPlink WINDOW. Never
# "every python on this machine".
#
# Measured 2026-10-24. `Get-Process python, pythonw` selects by EXECUTABLE NAME,
# which had two consequences, and the second is the one that did the damage:
#
#   * it FORCE-KILLED UNRELATED INTERPRETERS. It killed two stray python
#     processes belonging to another piece of work; a `pip install` cut off
#     mid-write leaves a broken virtual environment, and a suite cut off
#     mid-run leaves a scratch tree full of mutated source.
#   * its success condition was "no python exists anywhere", so with any other
#     python running it could NEVER report a clean close. It always fell through
#     to forcing and printed the camera-wedge warning -- while the app had in
#     fact exited cleanly two minutes earlier. A diagnostic that reports damage
#     it did not do gets believed: it was, and that report was passed on to him
#     as fact.
#
# So identify the app by what it IS, the owner of our window, not by the name of
# the interpreter it happens to run under. That is the same discriminator
# mistake as asking "which routine is running" instead of "could this be a
# trigger pull", and as choosing an exposure on mean frame brightness (C6).

# TWO INDEPENDENT PROPERTIES, never one string. An EXACT window title that the
# app actually sets, AND an owning process that is one of ours.
#
# `-like 'ScreenPlink*'` was tried first and matched, on the first run,
#     ScreenPlink - VK_Obsidian - Obsidian 1.14.4
# which is his NOTES, open on a file named after this project. A prefix match
# would have sent WM_CLOSE to an editor holding unsaved work the moment one of
# its windows mentioned "Camera Lab". Identifying anything here by a string
# alone is the mistake these notes already record for camera names and indices:
# "identify by WHAT IT DOES", and for a window that means the title the app
# itself sets plus the process it runs under.
$TITLES = @('ScreenPlink - Camera Lab', 'ScreenPlink - Target', 'ScreenPlink')
$MINE = @('python', 'pythonw', 'screenplink')      # screenplink.exe: the frozen build

$all = @([Win]::Windows())
$ours = @()
foreach ($row in $all) {
    $parts = $row -split '\|', 3
    if ($TITLES -notcontains $parts[2]) { continue }
    $owner = Get-Process -Id ([int]$parts[1]) -ErrorAction SilentlyContinue
    if (-not $owner) { continue }
    if ($MINE -notcontains $owner.ProcessName) {
        Write-Output "   ignoring '$($parts[2])' -- owned by $($owner.ProcessName), not ours"
        continue
    }
    $ours += , @([IntPtr][int64]$parts[0], [int]$parts[1], $parts[2])
}

if ($ours.Count -eq 0) {
    if (Get-Process python, pythonw -ErrorAction SilentlyContinue) {
        Write-Output 'no ScreenPlink window exists, but python IS running.'
        Write-Output '  Refusing to force an interpreter that may not be ours -- that has'
        Write-Output '  happened, and it killed work belonging to another session. If the app is still'
        Write-Output '  starting, wait for its window; if it is wedged with no window at'
        Write-Output '  all, name the process by hand, deliberately.'
        exit 2
    }
    Write-Output 'ScreenPlink is not running'
    exit 0
}

$appPids = @($ours | ForEach-Object { $_[1] } | Sort-Object -Unique)
# The venv's pythonw.exe is a launcher STUB that spawns the real interpreter, so
# the parent owns no window and exits only after its child. It is ours, and the
# wait below has to know that, or a clean close reads as a failure to close.
foreach ($one in @($appPids)) {
    try {
        $par = (Get-CimInstance Win32_Process -Filter "ProcessId=$one").ParentProcessId
        $pp = Get-Process -Id $par -ErrorAction SilentlyContinue
        if ($pp -and (@('python', 'pythonw') -contains $pp.ProcessName)) { $appPids += [int]$par }
    } catch { }
}
$appPids = @($appPids | Sort-Object -Unique)

Write-Output "ScreenPlink: $($appPids.Count) process(es): $($appPids -join ', ')"
foreach ($w in $ours) { Write-Output "   window: $($w[2])" }

# The CONSOLE is the window that owns closeEvent. Close THAT, not the target
# screen, and not 'whatever Windows calls main'.
$targets = @($ours | Where-Object { $_[2] -like '*Camera Lab*' } | ForEach-Object { $_[0] })
if ($targets.Count -eq 0) {
    Write-Output 'WARNING: only the target screen is open, and closing it does NOT shut'
    Write-Output '  the app down -- it closes itself and the console never hears it, which'
    Write-Output '  is the 2026-10-15 fault. Press Escape on the target screen: that asks'
    Write-Output '  the console to quit the way the app would.'
    exit 2
}
Write-Output "asking the Camera Lab window to close ($($targets.Count))..."
foreach ($h in $targets) { [Win]::Close($h) }

# Longer than the old 5s, because a clean exit is worth waiting for: the app's
# own disconnect allows 800ms for a capture thread that may be blocked inside
# cap.read(), a recording has to be finalised, and the parent stub exits only
# after its child.
$sw = [Diagnostics.Stopwatch]::StartNew()
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    if (-not (Get-Process -Id $appPids -ErrorAction SilentlyContinue)) {
        Write-Output "closed cleanly in $([int]$sw.Elapsed.TotalSeconds)s"
        exit 0
    }
}

Write-Output 'WARNING: did not close in 15s - forcing.'
Write-Output '  This breaks the recording AND risks wedging the camera until it'
Write-Output '  is physically replugged. If you see this, something is wrong in'
Write-Output '  closeEvent and it needs fixing rather than forcing.'
Write-Output "  Forcing ONLY ours: $($appPids -join ', ')"
Get-Process -Id $appPids -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
exit 1
