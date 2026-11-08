"""Is the monitor actually powered? Asked of WINDOWS, not of the camera.

WHY THIS EXISTS. A camera with a blind over its lens and a monitor that has
power-saved produce the SAME black frame, so no camera can tell them apart,
and the app has guessed wrong in both directions at a cost:

  2026-10-19  the monitor had power-saved. The app read a flash difference of
              0.0, concluded its healthy Kiyo Pro was blind, cycled to the
              C920 (blind taped over its lens), found the capture thread stuck
              while letting go of the first, and shut down. Three launches,
              each dead about 27 seconds in.

  2026-11-07  a reboot moved the name `USB Video Device` onto the BLINDED
              C920. The app opened it, saw black, and -- because a blind
              camera's own frame also reads as "the display is not lit" --
              took the 2026-10-19 branch and reported a sleeping display at a
              monitor that was plainly on. The camera-cycling code written in
              October for exactly this case was never reached, because the
              sleeping-display guard sits in front of it.

Both are the same mistake: concluding something about the DISPLAY from a
measurement taken through the one instrument that may itself be at fault.
RULES C6, measure the thing and not a proxy; the proxy here is the camera.

Windows already knows, and will say so for the price of one ctypes call and no
camera traffic at all: `GUID_CONSOLE_DISPLAY_STATE` reports ON, OFF or DIMMED
and pushes a notification whenever it changes.

THE FALLBACK IS THE OLD BEHAVIOUR, DELIBERATELY. Anything that does not work
-- not Windows, the registration refused, no notification arrived yet --
answers UNKNOWN, and the caller treats UNKNOWN exactly as it treated the world
before this module existed: report the display asleep and touch nothing. So
this can only ever turn a "stop and blame the display" into a camera switch
when the operating system has POSITIVELY said the panel is lit. It cannot
resurrect 2026-10-19.

No Qt and no OpenCV here on purpose: this must be importable by a test that
opens no window, and by the doctor.
"""

from __future__ import annotations

import sys

# What Windows reports for GUID_CONSOLE_DISPLAY_STATE.
OFF = 0
ON = 1
DIMMED = 2
UNKNOWN = -1

NAMES = {OFF: "OFF", ON: "ON", DIMMED: "DIMMED", UNKNOWN: "UNKNOWN"}

GUID_CONSOLE_DISPLAY_STATE = "{6FE69556-704A-47A0-8F24-C28D936FDA47}"

_WM_POWERBROADCAST = 0x0218
_PBT_POWERSETTINGCHANGE = 0x8013
_HWND_MESSAGE = -3
_DEVICE_NOTIFY_WINDOW_HANDLE = 0

# Module state. `_state` is the last thing Windows said; the rest is kept
# only so the callback and the window are not garbage collected, which would
# leave Windows calling into freed memory.
_state: int = UNKNOWN
_keep: list = []
_watching = False
_override: "int | None" = None       # tests only; see `force_for_test`


def force_for_test(value: "int | None") -> None:
    """Pin the answer (tests). None puts the real one back.

    A test must be able to produce OFF, ON, DIMMED and UNKNOWN on demand: the
    four branches this module exists to separate cannot otherwise be reached
    without power-saving the monitor of whoever is running the suite.
    """
    global _override
    _override = value


def state() -> int:
    """The display's power state as Windows last reported it, or UNKNOWN.

    Cheap and synchronous: it reads a value a callback has already stored.
    Never raises, because every caller is on a path that must keep working.
    """
    if _override is not None:
        return _override
    return _state


def is_lit() -> bool:
    """True ONLY when Windows positively says the panel is on.

    DIMMED and UNKNOWN are both False, on purpose. DIMMED is a panel that may
    be too dark for a camera to see change on, and UNKNOWN means nobody knows
    -- and "nobody knows" must behave like the world before this module, which
    is to leave the camera alone.
    """
    return state() == ON


def _say(log, message: str) -> None:
    """Log, and never let the LOGGER be the thing that breaks a launch.

    2026-11-07: this module was registered before the app's log widget
    existed, `self._log` raised AttributeError inside the success message,
    and the exception came back out of `watch()` and stopped the app from
    starting at all. The docstring above already promised this could not
    happen. A promise in a comment is not a guard.
    """
    if log is None:
        return
    try:
        log(message)
    except Exception:                                            # noqa: BLE001
        pass


def watch(log=None) -> bool:
    """Start listening for display power changes. Idempotent; never raises.

    Call once, from the thread that runs the message loop (the Qt main
    thread): the notification is delivered to a message-only window, and Qt's
    own event loop is what pumps it.

    Returns True if the registration took.
    """
    global _watching, _state
    if _watching:
        return True
    if not sys.platform.startswith("win"):
        return False
    try:
        _watching = _register(log)
    except Exception as exc:                                     # noqa: BLE001
        # A diagnostic convenience must never be the reason a launch fails.
        _say(log, f"display power: could not register for the display state "
                  f"({exc!r}); the display-asleep check falls back to the "
                  f"camera-only behaviour it had before.")
        _watching = False
    return _watching


def _register(log=None) -> bool:
    import ctypes
    import ctypes.wintypes as w
    import uuid

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", w.DWORD), ("Data2", w.WORD),
                    ("Data3", w.WORD), ("Data4", ctypes.c_ubyte * 8)]

        def __init__(self, text: str) -> None:
            u = uuid.UUID(text)
            super().__init__()
            self.Data1, self.Data2 = u.time_low, u.time_mid
            self.Data3 = u.time_hi_version
            self.Data4 = (ctypes.c_ubyte * 8)(*u.bytes[8:])

    class POWERBROADCAST_SETTING(ctypes.Structure):
        _fields_ = [("PowerSetting", GUID), ("DataLength", w.DWORD),
                    ("Data", ctypes.c_ubyte * 4)]

    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_longlong, w.HWND, w.UINT,
                                 w.WPARAM, w.LPARAM)

    def _proc(hwnd, msg, wparam, lparam):
        global _state
        if msg == _WM_POWERBROADCAST and wparam == _PBT_POWERSETTINGCHANGE:
            try:
                setting = ctypes.cast(
                    lparam, ctypes.POINTER(POWERBROADCAST_SETTING)).contents
                was, _state = _state, int(setting.Data[0])
                if was != _state:
                    _say(log, f"display power: Windows reports the screen is "
                              f"{NAMES.get(_state, _state)}.")
            except Exception:                                    # noqa: BLE001
                pass
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    class WNDCLASS(ctypes.Structure):
        _fields_ = [("style", w.UINT), ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                    ("hInstance", w.HINSTANCE), ("hIcon", w.HICON),
                    ("hCursor", w.HANDLE), ("hbrBackground", w.HBRUSH),
                    ("lpszMenuName", w.LPCWSTR), ("lpszClassName", w.LPCWSTR)]

    # EVERY SIGNATURE, EXPLICITLY. ctypes defaults an un-annotated integer
    # argument to a 32-bit C int, so any 64-bit HANDLE passed without an
    # argtypes entry raises OverflowError as soon as the value happens to be
    # large -- and whether it is large depends on where Windows loaded the
    # module, which is why this passed from source and failed in the frozen
    # build (2026-11-07, v1.0.1: "argument 11: OverflowError: int too long to
    # convert", argument 11 being hInstance). The result is not a crash but a
    # SILENT fallback: registration fails, the state stays UNKNOWN, and the
    # camera fix this module exists for never fires.
    #
    # Setting one signature because one call broke is the half-fix this
    # project keeps paying for (RULES E1). These are all of them.
    kernel32.GetModuleHandleW.restype = w.HMODULE
    kernel32.GetModuleHandleW.argtypes = [w.LPCWSTR]
    user32.DefWindowProcW.restype = ctypes.c_longlong
    user32.DefWindowProcW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
    user32.RegisterClassW.restype = w.ATOM
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
    user32.CreateWindowExW.restype = w.HWND
    user32.CreateWindowExW.argtypes = [
        w.DWORD, w.LPCWSTR, w.LPCWSTR, w.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        w.HWND, w.HMENU, w.HINSTANCE, ctypes.c_void_p]
    user32.RegisterPowerSettingNotification.restype = w.HANDLE
    user32.RegisterPowerSettingNotification.argtypes = [
        w.HANDLE, ctypes.POINTER(GUID), w.DWORD]

    proc = WNDPROC(_proc)
    cls = WNDCLASS()
    cls.lpfnWndProc = proc
    cls.hInstance = kernel32.GetModuleHandleW(None)
    # Unique per process: a second registration of the same class name fails,
    # and this module is imported by the app, the doctor and the suite.
    cls.lpszClassName = f"LsaDisplayPower{id(proc):x}"
    if not user32.RegisterClassW(ctypes.byref(cls)):
        return False

    hwnd = user32.CreateWindowExW(0, cls.lpszClassName, "lsa display power",
                                  0, 0, 0, 0, 0, w.HWND(_HWND_MESSAGE),
                                  None, cls.hInstance, None)
    if not hwnd:
        return False

    guid = GUID(GUID_CONSOLE_DISPLAY_STATE)
    handle = user32.RegisterPowerSettingNotification(
        w.HANDLE(hwnd), ctypes.byref(guid), _DEVICE_NOTIFY_WINDOW_HANDLE)
    if not handle:
        return False

    # HOLD EVERY ONE OF THESE. The callback especially: Windows keeps a raw
    # pointer to it, and letting Python collect it is a crash in somebody
    # else's stack frame with nothing in our log.
    _keep.extend([proc, cls, hwnd, handle, user32])
    _say(log, "display power: watching the screen's power state, so a blind "
              "camera and a sleeping monitor can be told apart without "
              "touching the camera.")
    return True
