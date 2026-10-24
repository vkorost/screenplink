"""Read-only: every visible top-level window, front to back, and what the target-screen
cover check would see at its five sample points.

    .venv\Scripts\python.exe tools\window_stack.py [screen_w screen_h]

Written 2026-10-20 to answer one question without touching the app: a 512x614 panel
covering about 4% of the screen produced "only 0 of 5 sampled points still belong to
it". Five points spread over the screen should not all land in that panel, so either
the points are clustered where the panel sits, or Qt's `topLevelAt` is answering "not
ours" for a reason other than a window visibly covering the point (for example a
large click-through layered window, which `ChildWindowFromPointEx` returns unless
CWP_SKIPTRANSPARENT is passed). This prints, for each point, what Windows returns
with each of three calls, and each window's extended-style flags, so the cause is
read off rather than guessed. It opens no camera, takes no screenshot and writes
nothing.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
user32.GetDesktopWindow.restype = wintypes.HWND
user32.WindowFromPoint.restype = wintypes.HWND
user32.WindowFromPoint.argtypes = [wintypes.POINT]
user32.ChildWindowFromPointEx.restype = wintypes.HWND
user32.ChildWindowFromPointEx.argtypes = [wintypes.HWND, wintypes.POINT, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]

GWL_EXSTYLE = -20
WS_EX_TOPMOST, WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = (
    0x8, 0x80000, 0x20, 0x80, 0x08000000)
FRACTIONS = ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))


def _title(h) -> str:
    b = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(h, b, 256)
    return b.value


def _proc(h) -> str:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
    hp = kernel32.OpenProcess(0x1000, False, pid.value)
    if not hp:
        return f"pid {pid.value}"
    try:
        n = wintypes.DWORD(520)
        p = ctypes.create_unicode_buffer(520)
        if kernel32.QueryFullProcessImageNameW(hp, 0, p, ctypes.byref(n)):
            return os.path.basename(p.value)
    finally:
        kernel32.CloseHandle(hp)
    return f"pid {pid.value}"


def _flags(h) -> str:
    ex = user32.GetWindowLongW(h, GWL_EXSTYLE)
    return ",".join(n for n, f in (("TOPMOST", WS_EX_TOPMOST), ("LAYERED", WS_EX_LAYERED),
                                   ("CLICK-THROUGH", WS_EX_TRANSPARENT), ("TOOL", WS_EX_TOOLWINDOW),
                                   ("NOACTIVATE", WS_EX_NOACTIVATE)) if ex & f) or "-"


def _rect(h):
    r = wintypes.RECT()
    user32.GetWindowRect(h, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def main() -> int:
    w = int(sys.argv[1]) if len(sys.argv) > 2 else user32.GetSystemMetrics(0)
    h = int(sys.argv[2]) if len(sys.argv) > 2 else user32.GetSystemMetrics(1)
    print(f"primary screen {w}x{h}; sample points at {FRACTIONS}\n")
    print("visible top-level windows, front to back:")
    cb = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def each(hwnd, _l):
        if user32.IsWindowVisible(hwnd):
            x, y, ww, hh = _rect(hwnd)
            if ww > 0 and hh > 0:
                print(f"  {hwnd:#010x}  {ww:5d}x{hh:<5d} at ({x},{y})  {_proc(hwnd):22s} "
                      f"{_flags(hwnd):28s} '{_title(hwnd)[:50]}'")
        return True
    user32.EnumWindows(cb(each), 0)
    print()
    desktop = user32.GetDesktopWindow()
    for fx, fy in FRACTIONS:
        pt = wintypes.POINT(int(w * fx), int(h * fy))
        wfp = user32.WindowFromPoint(pt)
        c1 = user32.ChildWindowFromPointEx(desktop, pt, 0x1)           # what Qt's topLevelAt does
        c2 = user32.ChildWindowFromPointEx(desktop, pt, 0x1 | 0x4)     # ...also skipping click-through
        def d(x):
            if not x:
                return "none"
            r = user32.GetAncestor(x, 2) or x
            return f"{r:#x} {_proc(r)} '{_title(r)[:30]}' [{_flags(r)}]"
        print(f"point ({pt.x},{pt.y}):\n   WindowFromPoint            {d(wfp)}\n"
              f"   ChildWindowFromPointEx(1)  {d(c1)}   <- the call Qt makes\n"
              f"   ChildWindowFromPointEx(5)  {d(c2)}   <- skipping click-through windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
