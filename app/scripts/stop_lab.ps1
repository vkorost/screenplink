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

$procs = Get-Process python, pythonw -ErrorAction SilentlyContinue
if (-not $procs) { Write-Output 'nothing running'; exit 0 }
$pids = @($procs | ForEach-Object { $_.Id })
Write-Output "$($procs.Count) process(es): $($pids -join ', ')"

# The operator console is the window that owns closeEvent. Close THAT, not the
# target screen, and not 'whatever Windows calls main'.
$targets = @()
foreach ($row in [Win]::Windows()) {
    $parts = $row -split '\|', 3
    if ($pids -notcontains [int]$parts[1]) { continue }
    Write-Output "   window: $($parts[2])"
    if ($parts[2] -like '*Camera Lab*') { $targets += [IntPtr][int64]$parts[0] }
}

if ($targets.Count -eq 0) {
    # No console window found -- fall back, but say so, because closing the
    # target screen alone has never once shut this app down.
    Write-Output 'WARNING: no Camera Lab window found; falling back to CloseMainWindow'
    foreach ($p in $procs) { $null = $p.CloseMainWindow() }
} else {
    Write-Output "asking the Camera Lab window to close ($($targets.Count))..."
    foreach ($h in $targets) { [Win]::Close($h) }
}

# Longer than the old 5s, because a clean exit is worth waiting for: the app's
# own disconnect allows 800ms for a capture thread that may be blocked inside
# cap.read(), a recording has to be finalised, and the parent stub exits only
# after its child.
$sw = [Diagnostics.Stopwatch]::StartNew()
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    if (-not (Get-Process python, pythonw -ErrorAction SilentlyContinue)) {
        Write-Output "closed cleanly in $([int]$sw.Elapsed.TotalSeconds)s"
        exit 0
    }
}

Write-Output 'WARNING: did not close in 15s - forcing.'
Write-Output '  This breaks the recording AND risks wedging the camera until it'
Write-Output '  is physically replugged. If you see this, something is wrong in'
Write-Output '  closeEvent and it needs fixing rather than forcing.'
Get-Process python, pythonw -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
exit 1
