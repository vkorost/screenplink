"""A fullscreen panel we can paint a flat colour into, for calibration.

Used to photograph the monitor lit and unlit so the camera can work out where
the screen is. Kept separate from the target display because it must work
whether or not the target window is open, and must sit above everything while
it runs.

SAFETY, learned the hard way on 2026-10-14
------------------------------------------
This window is fullscreen, frameless and always-on-top, and the code that put
it up used to remove it only when a camera frame arrived. The camera stalled,
the frame never came, and the user was left staring at an immovable black
rectangle covering the whole screen: no title bar, no close button, no key
handler, and alt-tabbing just put the other window underneath. The only way out
was to physically unplug the camera, which made the capture thread fail and ran
the error path. That is an unacceptable thing to do to somebody's machine.

So this window now guarantees three things, regardless of what the caller does:

- **Escape always closes it.** It takes focus so it can receive the key.
- **It closes itself** after `timeout_s` whether or not anyone asks it to.
- **It says how to escape** if it has been up longer than a moment. The hint is
  delayed so a normal flash -- which lasts about a second and IS photographed --
  is never contaminated by it, and it is mid-grey, which the chroma detector
  scores at exactly zero.

A caller may rely on none of this and it still cannot trap the user.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QScreen
from PySide6.QtWidgets import QApplication, QWidget


class FlashWindow(QWidget):

    cancelled = Signal(str)          # "escape" or "timeout"

    # Long enough for any legitimate flash step (the slowest is auto exposure
    # settling at 1.6s plus a still), short enough that a stall is an
    # annoyance rather than a lockout.
    DEFAULT_TIMEOUT_S = 8.0
    HINT_AFTER_S = 2.5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Calibrating")
        self.colour = QColor("#000000")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.StrongFocus)
        # NO MOUSE CURSOR, FOR THIS WINDOW'S WHOLE LIFE (2026-10-22).
        #
        # This window exists to be photographed white then black so the camera
        # can find the monitor, and Windows composites the cursor on top of it.
        # On the target window, hiding the cursor coincided with the noise
        # floor going 17 -> 3 and the bar 23 -> 9, same exposure, thirty minutes
        # apart (`display/window.py`). `_report_screen_ink` grabs the window and
        # never sees the cursor, so no check here can; this is by construction.
        #
        # Set from construction, on THIS WIDGET: not `setOverrideCursor`, which
        # would leak to the operator console. Unlike the target window there is
        # no show-on-move: this window is up for about a second and is never
        # aimed at, so blank for its whole life is simpler and strictly better.
        #
        # The ways out (Escape/Space/Q, double click, the deadline) do not
        # depend on the pointer being visible. A double click lands anywhere in
        # a window that covers the whole screen, so there is nothing to aim at.
        self.setCursor(Qt.BlankCursor)
        self._show_hint = False
        self._done = False               # stood down already; emit exactly once

        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(lambda: self._give_up("timeout"))

        self._hint = QTimer(self)
        self._hint.setSingleShot(True)
        self._hint.timeout.connect(self._enable_hint)

    # -- lifecycle ------------------------------------------------------------

    def show_on(self, screen: QScreen | None = None,
                timeout_s: float | None = None) -> None:
        screen = screen or QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self._show_hint = False
        self._done = False
        self.showFullScreen()
        self.raise_()
        # Focus, or Escape goes to whatever had the keyboard before and this
        # window becomes exactly the trap described above.
        self.activateWindow()
        self.setFocus(Qt.OtherFocusReason)
        self._timeout.start(int((timeout_s or self.DEFAULT_TIMEOUT_S) * 1000))
        self._hint.start(int(self.HINT_AFTER_S * 1000))

    def keep_alive(self, seconds: float) -> None:
        """Extend the deadline because the step is legitimately still running."""
        self._timeout.start(int(seconds * 1000))

    def _enable_hint(self) -> None:
        self._show_hint = True
        self.update()

    def _give_up(self, why: str) -> None:
        """Stand down, and tell the owner AFTER this call has finished.

        The owner's handler sets its reference to this window to None, which in
        PySide6 drops the last Python reference and destroys the C++ object --
        while this method is still on the stack. Emitting inline therefore
        crashed Qt6Core with a 0xc0000409 fail-fast the moment anybody pressed
        Escape or the deadline expired.

        So: free the user's screen immediately, then let the event loop unwind
        before announcing anything. `_done` guards against the deadline and a
        keypress both arriving.
        """
        if self._done:
            return
        self._done = True
        self._timeout.stop()
        self._hint.stop()
        self.hide()                      # the screen is free from here on
        QTimer.singleShot(0, lambda: self.cancelled.emit(why))

    def closeEvent(self, ev) -> None:
        self._timeout.stop()
        self._hint.stop()
        super().closeEvent(ev)

    def keyPressEvent(self, ev) -> None:
        # Escape, and also Space and Q, because someone trying to get out of a
        # black fullscreen window will press whatever is nearest.
        if ev.key() in (Qt.Key_Escape, Qt.Key_Space, Qt.Key_Q):
            self._give_up("escape")
            return
        super().keyPressEvent(ev)

    def mouseDoubleClickEvent(self, _ev) -> None:
        self._give_up("escape")

    # -- painting -------------------------------------------------------------

    def set_colour(self, colour: str) -> None:
        self.colour = QColor(colour)
        self.update()
        # Force the frame out to the compositor now: the camera is about to
        # photograph this, so "eventually painted" is not good enough.
        QApplication.processEvents()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), self.colour)
        if self._show_hint:
            # Mid-grey: red minus max(green, blue) is zero, so the detector
            # cannot read this as a laser hit. Small and in a corner, so even
            # if a measurement did run with it visible it changes almost
            # nothing -- and it only appears when a measurement has already
            # taken far longer than one should.
            p.setFont(QFont("Consolas", 14, QFont.Bold))
            p.setPen(QColor("#9a9a9a"))
            p.drawText(40, self.height() - 40,
                       "Calibration is taking too long - press Esc to cancel")
        p.end()
