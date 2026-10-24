"""Make a window actually come to the front on Windows, and SAY so if it did not.

`QWidget.activateWindow()` maps to SetForegroundWindow, and Windows REFUSES
that call whenever the caller is not already the foreground process. It does
not fail loudly: it flashes the taskbar button instead. His report,
2026-10-18: "the app doesn't take up the whole screen. It is blinking on the
taskbar but it's not full screen."

THIS IS NOT COSMETIC. The camera watches that monitor. A target window that
loses the foreground leaves the camera photographing the desktop, and then:

  - the noise floor is measured against editors and browsers, which this
    project has already measured at peak 90 against 0 for the target
    background -- a min_peak of 105 rejects every real shot on this screen
  - the self test photographs its own dots through whatever is on top
  - calibration markers are drawn where nothing can see them, which is the
    "12 markers within 50px of each other" signature in CLAUDE.md

On 2026-10-18 the app logged "only 0 of 5 sampled points still belong to it"
four times in one sweep, and one launch produced no self test at all.

THE FIX is the documented Win32 dance: attach to the foreground window's
input queue so we count as the foreground process for the length of one call,
then ask. Falling back to a MOMENTARY topmost, never a permanent one --
`never-trap-the-user-behind-a-window` is a rule here, paid for by a fullscreen
cover that once needed the camera unplugged to escape.
"""
from __future__ import annotations

import ctypes
import sys


def _win32_force(hwnd: int) -> bool:
    """The AttachThreadInput dance. True if we ended up in front."""
    u32 = ctypes.windll.user32
    k32 = ctypes.windll.kernel32
    if u32.GetForegroundWindow() == hwnd:
        return True

    fg = u32.GetForegroundWindow()
    ours = k32.GetCurrentThreadId()
    theirs = u32.GetWindowThreadProcessId(fg, None) if fg else 0

    attached = False
    if theirs and theirs != ours:
        attached = bool(u32.AttachThreadInput(theirs, ours, True))
    try:
        u32.BringWindowToTop(hwnd)
        u32.SetForegroundWindow(hwnd)
        u32.SetActiveWindow(hwnd)
    finally:
        if attached:
            u32.AttachThreadInput(theirs, ours, False)

    if u32.GetForegroundWindow() == hwnd:
        return True

    # MOMENTARY topmost, then straight back. A window left permanently on top
    # is how a user gets trapped, and that has happened on this project.
    HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
    SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010
    flags = SWP_NOMOVE | SWP_NOSIZE
    u32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
    u32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags | SWP_NOACTIVATE)
    u32.SetForegroundWindow(hwnd)
    return u32.GetForegroundWindow() == hwnd


def force_to_front(win) -> bool:
    """Show `win` fullscreen and get it genuinely in front.

    Returns whether it IS in front afterwards -- the caller is expected to
    report a False rather than assume it worked, because a window that is
    not in front looks exactly like a camera that cannot see the screen,
    and this project has spent whole sessions on that confusion.
    """
    win.showFullScreen()
    win.raise_()
    win.activateWindow()
    if sys.platform != "win32":
        return True
    try:
        return _win32_force(int(win.winId()))
    except Exception:
        # Never let a cosmetic raise take the app down; the caller logs the
        # False and the screen-ownership check catches it either way.
        return False
