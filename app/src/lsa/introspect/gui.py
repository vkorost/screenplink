"""The two places the introspection server has to touch Qt, and the deadlines on them.

Constraint 3 of the agreed design: **off the hot path, provably.** The detector
has about 7 ms per camera frame and crossing ~14 ms halves how many frames a
100 ms pulse lands in -- a cliff, not a slope. His Java MCP server marshals
every call with `invokeAndWait` and then sleeps 100 ms; that is fine for a
Swing form and impossible here.

So:

  * The HTTP server runs on its own thread and almost everything it answers is
    read straight off plain attributes (a dataclass field, a numpy reference, a
    counter). None of that needs Qt and none of it touches the capture loop.
  * The two things that genuinely need the Qt thread -- grabbing the target
    window, and posting a key event -- go through `GuiBridge`, which hands the
    work to the Qt thread by queued signal and **waits with a deadline**. The
    server thread blocks; the capture thread never does; and if the Qt thread is
    busy the request returns 503 saying so rather than hanging (RULES D4: no
    unbounded wait, ever).
  * Every hop is TIMED, and the time spent on the Qt thread is reported by
    `/health`, so the cost this server imposes is a number anyone can read
    rather than a claim (RULES C3).

`ShotTap` is the other Qt-shaped piece. `/shots` needs the real `Shot` objects
with their per-frame `members`, and the app keeps only `ScoredShot` summaries.
The ring must therefore be filled from `shot_detected`, which is emitted inside
the capture loop -- so the connection must be QUEUED, or the append runs on the
thread that has 14 ms to spend.

**It asks for `Qt.QueuedConnection` explicitly, and that was worth finding out
the hard way.** The first version relied on the tap being a QObject and
inferred the rest; `tests/mutate_introspect.py` then reported the fault
"ShotTap stops being a QObject" as NOT CAUGHT, and measuring it showed why.
PySide6 delivers to a plain Python callable using the SENDER's thread
affinity, and `CaptureThread` is a QThread whose QObject lives on the GUI
thread even though `run()` executes elsewhere -- so an AutoConnection was
already queued, whatever the receiver was. The claim in this docstring was
wrong about its own mechanism while being right about the outcome, which is
the shape of several expensive mistakes on this project. Asking for the
connection type says what is required instead of depending on a rule two
classes away.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication


class GuiBusy(RuntimeError):
    """The Qt thread did not take the work inside the deadline."""


class _Task:
    __slots__ = ("fn", "done", "value", "error", "gui_ms")

    def __init__(self, fn) -> None:
        self.fn = fn
        self.done = threading.Event()
        self.value = None
        self.error: "BaseException | None" = None
        self.gui_ms = 0.0


class GuiBridge(QObject):
    """Run a callable on the Qt thread, from another thread, with a deadline."""

    # How long the Qt thread may hold ONE piece of introspection work before
    # the server gives up on it. A window grab on a 4K panel measures a few
    # tens of milliseconds; a second means the GUI thread is wedged, and in
    # that case the useful answer is "it is wedged", not a hung request.
    DEFAULT_TIMEOUT_S = 1.0

    _submit = Signal(object)

    def __init__(self, parent: "QObject | None" = None) -> None:
        super().__init__(parent)
        # QueuedConnection explicitly: the default would be Auto, which is
        # Queued here anyway, but the whole correctness of this class is that
        # the slot runs on the OWNER's thread and not on the caller's. A
        # property this fragile is stated, not inferred.
        self._submit.connect(self._run, Qt.QueuedConnection)
        self.calls = 0
        self.timeouts = 0
        self.gui_ms_total = 0.0
        self.gui_ms_max = 0.0
        self.wait_ms_max = 0.0

    def run(self, fn, timeout_s: "float | None" = None):
        """Returns fn()'s value. Raises GuiBusy on the deadline, or fn's error."""
        task = _Task(fn)
        t0 = time.perf_counter()
        self._submit.emit(task)
        if not task.done.wait(self.DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s):
            self.timeouts += 1
            raise GuiBusy(
                "the Qt thread did not run this within its deadline. It is "
                "repainting, or a routine is holding it. Nothing was changed.")
        self.calls += 1
        waited = (time.perf_counter() - t0) * 1000.0
        self.wait_ms_max = max(self.wait_ms_max, waited)
        self.gui_ms_total += task.gui_ms
        self.gui_ms_max = max(self.gui_ms_max, task.gui_ms)
        if task.error is not None:
            raise task.error
        return task.value

    def _run(self, task: _Task) -> None:
        t0 = time.perf_counter()
        try:
            task.value = task.fn()
        except BaseException as exc:                     # noqa: BLE001
            task.error = exc
        finally:
            task.gui_ms = (time.perf_counter() - t0) * 1000.0
            task.done.set()

    def stats(self) -> dict:
        return {
            "gui_calls": self.calls,
            "gui_timeouts": self.timeouts,
            "gui_ms_total": round(self.gui_ms_total, 2),
            "gui_ms_max": round(self.gui_ms_max, 2),
            "gui_wait_ms_max": round(self.wait_ms_max, 2),
            "gui_deadline_s": self.DEFAULT_TIMEOUT_S,
        }


class ShotTap(QObject):
    """A ring of the last N real `Shot` objects, with their per-frame members.

    Bounded, because an unbounded buffer in a long session is a leak and this
    exists to be read a few shots at a time.
    """

    KEEP = 40

    def __init__(self, parent: "QObject | None" = None) -> None:
        super().__init__(parent)
        self._ring: deque = deque(maxlen=self.KEEP)
        self.seen = 0

    def attach(self, signal) -> None:
        """Fill the ring from a signal emitted inside the capture loop.

        `Qt.QueuedConnection` is asked for, not left to AutoConnection: the
        requirement is that `record` runs on the GUI thread, and an
        AutoConnection decides that from thread affinities two classes away.
        See the module docstring for the measurement that made this explicit.
        """
        signal.connect(self.record, Qt.QueuedConnection)

    def record(self, shot: object) -> None:
        self.seen += 1
        self._ring.append((time.perf_counter(), shot))

    def recent(self, n: int) -> list:
        rows = list(self._ring)
        return rows[-n:] if n > 0 else rows


# --- keys ------------------------------------------------------------------

#  Qt::Key_A is 0x41 and Qt::Key_0 is 0x30, so a single printable character
#  resolves by ordinal and needs no table. Everything else is named, and the
#  names are Qt's own so there is one spelling to learn.
_NAMED = (
    "Escape", "Space", "Backspace", "Return", "Enter", "Tab", "Delete",
    "Left", "Right", "Up", "Down", "BracketLeft", "BracketRight",
    "Plus", "Minus", "Equal", "Comma", "Period", "Slash",
) + tuple(f"F{i}" for i in range(1, 13))


def resolve_key(name: str) -> "int | None":
    """'g' -> Qt.Key_G, 'Escape' -> Qt.Key_Escape, 'f11' -> Qt.Key_F11."""
    if not name:
        return None
    s = name.strip()
    if len(s) == 1:
        ch = s.upper()
        if ch.isalnum():
            return ord(ch)
        attr = {"[": "BracketLeft", "]": "BracketRight", " ": "Space",
                "+": "Plus", "-": "Minus", "=": "Equal", ",": "Comma",
                ".": "Period", "/": "Slash"}.get(ch)
        if attr is None:
            return None
        s = attr
    low = s.lower()
    for candidate in _NAMED:
        if candidate.lower() == low:
            key = getattr(Qt, "Key_" + candidate, None)
            return None if key is None else int(key)
    return None


def post_key(widget: object, key: int, text: str = "") -> None:
    """Deliver a key press to `widget` WITHOUT taking the keyboard off anyone.

    Why this and not a synthetic keystroke: pressing a key used to mean
    activating the window and sending input through the OS, and Windows
    routinely refuses foreground activation to a process that is not already
    in front -- so the activation half failed, the keystroke went wherever the
    focus actually was, and on one occasion it knocked the target window behind
    Settings, which then tripped the screen-ownership guard and cost a run.

    `postEvent` is addressed to the object, so focus is irrelevant and nothing
    moves on screen. The event then travels `TargetWindow.keyPressEvent`, which
    is the real path: layers get first refusal, F11 and Escape keep their
    meanings, and anything the app binds goes through `key_pressed` to the
    app's own handler. Nothing here reaches around that, so a key that starts a
    routine starts it the way a human would -- through the one run slot
    (constraint 2).
    """
    app = QApplication.instance()
    if app is None:                                      # pragma: no cover
        raise RuntimeError("there is no QApplication to post an event to")
    app.postEvent(widget, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, text))
    app.postEvent(widget, QKeyEvent(QEvent.KeyRelease, key, Qt.NoModifier, text))


def post_click(widget: object, x: float, y: float) -> None:
    """Deliver a left-button press at widget coordinates (x, y), no focus taken.

    The pointer equivalent of `post_key`, and it exists for the same reason:
    moving the real cursor and clicking means asking Windows to put this
    window in front, and Windows routinely refuses foreground activation to a
    process that is not already there -- so the click would land wherever the
    pointer actually is, on whatever is actually in front. This project has
    already knocked the target window behind Settings that way and lost a run.

    `postEvent` is addressed to the window, so nothing moves on screen and no
    other application can receive it by accident. The event then travels
    `TargetWindow.mousePressEvent`, which is the real path a human click
    takes: `Scene.on_click` offers it to the topmost visible layer first.

    IT IS A CLICK, NOT A SHOT. Nothing here reaches the detector, the score or
    the bullet holes -- the app turns no click into a `ShotEvent`, and a tool
    that could would make every hit rate it reports untrustworthy.
    """
    app = QApplication.instance()
    if app is None:                                      # pragma: no cover
        raise RuntimeError("there is no QApplication to post an event to")
    pos = QPointF(float(x), float(y))
    app.postEvent(widget, QMouseEvent(QEvent.MouseButtonPress, pos,
                                      Qt.LeftButton, Qt.LeftButton,
                                      Qt.NoModifier))
