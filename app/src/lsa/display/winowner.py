"""Who owns the window at a screen point: its title and its process.

WHY (2026-10-20). His dictation app (Wispr Flow) keeps a small topmost panel,
"Status", on the monitor the camera watches. The app's own guard noticed within a
second ("only 0 of 5 sampled points still belong to it"), refused every detection
and told him "Something is in front of this screen" -- and he, who had not moved
the camera or the light, could do nothing with that. The guard was right and the
message was useless, because it withheld the one fact the app already had in its
hand: Windows had just told it which window was there. A conclusion with the
evidence withheld is a message nobody can act on.

Four Win32 calls turn the window handle into a name. Everything is read-only,
nothing is cached across calls, and EVERY failure returns None: a name that cannot
be read falls back to the old wording and is never invented.

Windows only (ctypes); on any other platform `window_at` returns None.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowOwner:
    hwnd: int
    title: str
    process: str          # executable name without the extension, e.g. "Wispr Flow"
    pid: int
    ours: bool            # this very process

    def describe(self) -> str:
        """"Wispr Flow's window 'Status'", or the best honest part of it."""
        who = "this app's own" if self.ours else (f"{self.process}'s" if self.process else "another program's")
        if self.title:
            return f"{who} window '{self.title}'" if not self.ours else f"the app's own window '{self.title}'"
        return f"a window of {self.process}" if self.process and not self.ours else f"{who} window"


def window_at(x: int, y: int) -> "WindowOwner | None":
    """The top-level window at screen point (x, y), as Qt's own check sees it
    (`ChildWindowFromPointEx` on the desktop, invisible windows skipped), or None
    when it cannot be read."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.GetDesktopWindow.restype = wintypes.HWND
        user32.ChildWindowFromPointEx.restype = wintypes.HWND
        user32.ChildWindowFromPointEx.argtypes = [wintypes.HWND, wintypes.POINT, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        desktop = user32.GetDesktopWindow()
        CWP_SKIPINVISIBLE = 0x0001
        hwnd = user32.ChildWindowFromPointEx(desktop, wintypes.POINT(int(x), int(y)), CWP_SKIPINVISIBLE)
        if not hwnd or hwnd == desktop:
            return None
        GA_ROOT = 2
        root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(root, buf, 256)
        title = buf.value.strip()
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(root, ctypes.byref(pid))
        proc = ""
        if pid.value:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
            if h:
                try:
                    n = wintypes.DWORD(520)
                    path = ctypes.create_unicode_buffer(520)
                    if kernel32.QueryFullProcessImageNameW(h, 0, path, ctypes.byref(n)):
                        proc = os.path.splitext(os.path.basename(path.value))[0]
                finally:
                    kernel32.CloseHandle(h)
        if not title and not proc:
            return None
        return WindowOwner(int(root), title, proc, int(pid.value), int(pid.value) == os.getpid())
    except Exception:
        return None


def most_common(owners: "list[WindowOwner | None]") -> "WindowOwner | None":
    """The window named most often among the sample points (ties: the first
    seen), ignoring points that could not be read."""
    counts: dict = {}
    order: list = []
    for o in owners:
        if o is None:
            continue
        key = (o.process, o.title, o.hwnd)
        if key not in counts:
            counts[key] = [0, o]
            order.append(key)
        counts[key][0] += 1
    if not counts:
        return None
    best = max(order, key=lambda k: counts[k][0])
    return counts[best][1]
