"""The fullscreen target window: a host for a Scene and nothing more.

Kept this thin on purpose. The window's whole job is to own a borderless
fullscreen surface on a chosen monitor, drive the scene's clock, route keys,
and get out of the way. All appearance lives in layers, so the eventual game
and today's diagnostic screen are the same window with a different stack.

The repaint loop is demand-driven: the timer ticks at the target rate but only
calls update() when some layer says it changed. A static target screen with the
clock hidden therefore costs essentially nothing, which matters because this
process is also running computer vision on every camera frame.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QPainter, QScreen
from PySide6.QtWidgets import QApplication, QWidget

from . import winowner
from .scene import Scene


class TargetWindow(QWidget):
    """Borderless fullscreen surface that renders a Scene."""

    closed = Signal()
    key_pressed = Signal(int)
    # F11 ends the SESSION, not just this window. His instruction, 2026-10-18:
    # "when I press F11 I need the app to close, that includes the other frame
    # as well". Closing this window alone left the operator console alive with
    # the camera still open, so the shooter had closed the app and the app was
    # still running -- and two of those on one camera is what wedges it.
    #
    # A SIGNAL, not `QApplication.quit()`: the console owns `closeEvent`, which
    # stands every routine down and releases the camera the way the app would.
    # Quitting from here would be the force-kill this project has paid for
    # repeatedly (A1: 40.3s to open and 90,816 read failures afterwards).
    quit_requested = Signal()
    # Emitted after every paintEvent completes, so a routine that has just
    # drawn something can wait for it to actually be on the glass before
    # photographing it.
    #
    # The alternative, used everywhere before this, was
    # `QApplication.processEvents()` -- fifteen calls inside routines, each
    # one a re-entrancy point where any other callback could run inside the
    # current step. It was also not sufficient: one processEvents() does not
    # paint a fullscreen 4K window, which is how a calibration marker came
    # to be photographed BEFORE it was drawn (strength 14 where its
    # siblings read 197-231, so the strongest noise blob in the difference
    # image won) and how an alignment check measured a 109px shift against
    # a half-painted window.
    painted = Signal()

    # Generic show/hide toggles. Layers can also claim their own keys; these
    # just spare every layer from reimplementing "am I visible".
    REPEATABLE_KEYS = (Qt.Key_BracketLeft, Qt.Key_BracketRight)

    TOGGLES: dict[int, str] = {
        Qt.Key_P: "camera",
        Qt.Key_S: "stats",
    }

    def __init__(self, scene: Scene, target_fps: int = 60, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self.setWindowTitle("ScreenPlink - Target")
        self.scene = scene
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        # WHY `in_front_of_camera` LAST SAID WHAT IT SAID. Declared in the
        # constructor and assigned unconditionally, never read through
        # getattr with a default: a guard that reads a name nobody checked
        # returned 0.0 for days while looking exactly like protection
        # (RULES C8). It was first assigned in `show_on`, so it did not
        # exist until the window was displayed -- caught by the check in
        # test_feedback section 31 rather than by me.
        self.cover_reason = ""
        self.cover_by = ""        # WHO is in front, when Windows could say (see `winowner`)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(max(1, int(1000 / max(1, target_fps))))
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # THE POINTER IS HIDDEN FROM THE MOMENT THIS WINDOW EXISTS (2026-10-22).
        # His request: "now my mouse cursor is big, could it be that the big
        # bright mouse cursor might throw off the measuring? ... hide the mouse
        # cursor unless the user moves the mouse ... hide it after 1 second.
        # And initially hide it when the app starts."
        #
        # WHY IT MATTERS HERE AND NOWHERE ELSE. Windows composites the cursor
        # on top of this window; Qt does not draw it. So it is absent from
        # `_report_screen_ink`'s window grab and from every rendering check in
        # the suite (RULES B1/B10 are enforced by rendering the scene, and the
        # scene never contains it). It is the one bright object on the screen
        # the camera watches that the app can neither see, measure nor cap, and
        # his is enlarged. Whether it moved the noise floor (17 where the same
        # exposure gave 2 an hour before) was not proven when this was written;
        # an uncontrolled bright object on the shooting screen is a liability
        # whether or not it was that time.
        #
        # BLANK AT CONSTRUCTION, not after the first timeout: startup runs the
        # screen search, the exposure probe and the self test, and all three
        # photograph this screen. A blank cursor on THIS WIDGET, not
        # `QApplication.setOverrideCursor`, which would leak to the operator
        # console, Settings and every other window of the process.
        #
        # NOTHING HERE TOUCHES THE SCENE. Showing or hiding a cursor is a
        # window-level property: no `update()`, no `mark_dirty`, no layer is
        # asked anything. The HUD tick already dirties the scene ~5x a second
        # and a slow paint starving the capture thread cost him an evening on
        # 2026-10-20, so this path must never add a repaint of its own.
        self.setMouseTracking(True)          # moves with no button held
        self.setCursor(Qt.BlankCursor)
        # A single-shot restarted on each move; no polling. Parented to self,
        # so it dies with the window and cannot fire into a destroyed one.
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(int(self.CURSOR_HIDE_S * 1000))
        self._cursor_timer.timeout.connect(self._hide_cursor)

    # How long the pointer stays after the mouse last moved.
    CURSOR_HIDE_S = 1.0

    # -- the pointer ----------------------------------------------------------

    def cursor_hidden(self) -> bool:
        """True while this window's cursor is the blank one."""
        return self.cursor().shape() == Qt.BlankCursor

    def _show_cursor(self) -> None:
        if self.cursor_hidden():
            self.unsetCursor()               # back to the ordinary arrow

    def _hide_cursor(self) -> None:
        self.setCursor(Qt.BlankCursor)

    def mouseMoveEvent(self, ev) -> None:
        """Any movement over this window shows the pointer and restarts the
        one-second countdown. Deliberately NOT forwarded to the scene: a move
        is not a click and not a shot, and no layer has a use for it."""
        self._show_cursor()
        self._cursor_timer.start()           # restart, not stack

    def showEvent(self, ev) -> None:
        # Re-asserted every time the window is shown (it is hidden and shown
        # again by fullscreen changes), so a pointer left visible by a move
        # just before a hide cannot survive into the next showing. Touches the
        # cursor only: no focus, activation or z-order, which belong to
        # `display.foreground.force_to_front`.
        self._cursor_timer.stop()
        self.setCursor(Qt.BlankCursor)
        super().showEvent(ev)

    # -- lifecycle ------------------------------------------------------------

    def show_on(self, screen: QScreen | None = None) -> None:
        screen = screen or QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def closeEvent(self, ev) -> None:
        self._timer.stop()
        self._cursor_timer.stop()
        self.closed.emit()
        super().closeEvent(ev)

    # -- frame loop -----------------------------------------------------------

    def _tick(self) -> None:
        if self.scene.advance(self.rect()):
            self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        self.scene.paint(p, self.rect())
        p.end()
        # After `p.end()`, so a listener that photographs the screen sees
        # the finished frame rather than a partly composited one.
        self.painted.emit()

    # -- is this screen what the camera is looking at? ------------------------
    #
    # Asked for after a storm of gunshots, 2026-10-16 13:29: "As soon as I
    # press Escape I hear continuous shots that just do not stop. This is
    # completely unacceptable."
    #
    # Escape LOWERS this window rather than closing it, deliberately, because
    # closing it cost an entire session (see keyPressEvent). But lowering it
    # leaves the camera pointed at a lit desktop -- editors, browsers,
    # coloured text -- while the detector is still running on a bar measured
    # against this window's near-black background a minute earlier. Measured
    # from the moment he pressed it:
    #
    #     13:29:41 -> 13:30:00   92 detections in 20s, 3-7 every second,
    #                            peak 9-29, area 2-546px
    #     one of his real shots  peak 85-146, area 300-470px
    #
    # Every one rang the gunshot, every one was offered to the sensitivity
    # learner, and the project's own notes already measured the noise floor
    # against his desktop at 90 where this window gives 0-2. So it was
    # certain to happen the moment the window dropped.
    #
    # RULES B9 says the test is "could this detection possibly BE a trigger
    # pull", and names a floor measured in this room and a mapping to place
    # the hit. Both were true here. The condition was incomplete, and this
    # is the third time B9 has been fixed by picking the discriminator in
    # front of me -- which routine is running, then is there a floor and a
    # mapping. The honest one is whether the camera is looking at the app's
    # own background at all, and only this window can answer that.
    #
    # Sampled at five points, not one: the operator console, an alert, or
    # another application can cover PART of the panel, and the centre pixel
    # is not the screen the camera watches.
    PROBE_FRACTIONS = ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25),
                       (0.25, 0.75), (0.75, 0.75))
    PROBE_MAJORITY = 3

    def _top_level_at(self, point: QPoint):
        """Which of THIS application's top-level windows is at a screen point.

        `QApplication.topLevelAt` returns None when the window there belongs
        to another process, which is exactly the "something else is in front"
        case. Separated into its own method so a test can answer it directly:
        the offscreen platform the harness runs on cannot, and a branch that
        is unreachable in every test is the C1 failure wearing a safety
        jacket.
        """
        return QApplication.topLevelAt(point)

    def _owner_at(self, point: QPoint):
        """Who owns the window at a screen point (title and process), or None.
        Its own method so a test can answer it: the offscreen platform has no
        windows of other programs, and a branch unreachable in every test is the
        C1 failure wearing a safety jacket."""
        return winowner.window_at(point.x(), point.y())

    def in_front_of_camera(self) -> bool:
        """Is this window the thing a camera aimed at the monitor sees?

        Sets `cover_reason` to what was measured, because a check that
        reports only a verdict can be re-run and not debugged (RULES C3).
        """
        if not self.isVisible() or self.isMinimized():
            self.cover_reason = ("the target screen is hidden or minimised, "
                                 "so the camera is watching the desktop")
            return False
        g = self.geometry()
        if g.isEmpty():
            self.cover_reason = "the target window has no geometry yet"
            return False
        if QGuiApplication.platformName() == "offscreen":
            # The harness runs here and the platform genuinely cannot answer.
            # Guessing "not in front" would suppress every simulated shot --
            # this guard inverted, silently, which is the worse error of the
            # two and the one this project keeps making. Fail OPEN and say so.
            self.cover_reason = ""
            return True
        ours = 0
        covering = []
        for fx, fy in self.PROBE_FRACTIONS:
            pt = QPoint(int(g.x() + g.width() * fx),
                        int(g.y() + g.height() * fy))
            if self._top_level_at(pt) is self:
                ours += 1
            else:
                covering.append(self._owner_at(pt))
        if ours >= self.PROBE_MAJORITY:
            self.cover_reason = ""
            self.cover_by = ""
            return True
        # NAME WHAT IS IN FRONT. Windows has just told us which window it is at
        # each point that is not ours (`_owner_at`); a message that withholds it
        # ("something is in front of this screen") cannot be acted on, and the
        # live case was a dictation app's topmost panel he did not know existed.
        # If the window cannot be read the wording falls back to "something" and
        # no name is invented.
        owner = winowner.most_common(covering)
        self.cover_by = owner.describe() if owner is not None else ""
        self.cover_reason = (
            f"{self.cover_by or 'something'} is in front of the target screen: "
            f"only {ours} of {len(self.PROBE_FRACTIONS)} sampled points still belong to it")
        return False

    # -- input ----------------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        """A click on the target screen, offered to the layers. NOT a shot.

        The target screen had no mouse input at all until 2026-10-18, when he
        asked for the menu to work with the pointer as well as the gun: "menu
        should also be triggered by the mouse pointer".

        It goes to `Scene.on_click`, which offers it to the topmost visible
        layer first -- the same order as a key, because what is on top is what
        the pointer is pointing at. A layer that does not want it returns
        False and the click falls through.

        IT IS NEVER TURNED INTO A `ShotEvent`. A click has no pulse, no peak
        and no camera position; if it reached the shot path it would land in
        the score, the hit rate and the sensitivity map, and the hit rate is
        the one number this app exists to measure honestly. The menu is a
        screen of buttons and takes clicks as clicks.
        """
        if ev.button() != Qt.LeftButton:
            return
        p = ev.position() if hasattr(ev, "position") else ev.localPos()
        ctx = self.scene._context(time.perf_counter(), 0.0, self.rect())
        if self.scene.on_click(p.x(), p.y(), ctx):
            self.update()

    def keyPressEvent(self, ev) -> None:
        # Layers first. An active mode or an overlay is the most specific thing
        # on screen, so it gets first refusal -- otherwise a global shortcut
        # silently shadows whatever the user is actually looking at.
        key = ev.key()
        # A HELD KEY IS NOT A SERIES OF PRESSES (2026-10-17). Windows repeats
        # a held key, and the app was at 19-30 fps with the keyboard queue
        # backed up, so one long press of 0 flipped the information group
        # nine times in two seconds (08:50:42-44, log lines alternating
        # SHOWN/HIDDEN) and it landed wherever the parity fell. He reported
        # "I cannot remove it with 0". Every key here is a discrete action
        # except the pane-size brackets, which are meant to repeat.
        if ev.isAutoRepeat() and key not in self.REPEATABLE_KEYS:
            return
        if self.scene.on_key(key):
            self.update()
            return
        if key == Qt.Key_F11:
            # Hide at once so the screen is his again on the keystroke, then
            # ask the owner to shut down on the next turn of the event loop.
            # NEVER destroy a QObject from inside its own handler: emitting
            # `cancelled` inline from a fullscreen window's own callback let
            # the owner drop the last reference, PySide6 destroyed the C++
            # object, and the method carried on calling close() on it --
            # exception 0xc0000409, every time anyone pressed Escape.
            self.hide()
            QTimer.singleShot(0, self.quit_requested.emit)
            return
        if key == Qt.Key_Escape:
            # ESCAPE CLOSES THE APP. His instruction, 2026-10-18: "Let's put
            # Escape as the exit for the app."
            #
            # THIS REVERSES A RULE THAT WAS PAID FOR, and the history belongs
            # next to the change rather than in a file nobody opens. Escape
            # used to close the target SCREEN, and it cost a session: the
            # window died, the camera was left watching a lit desktop,
            # detection collapsed, and the next report was "still not
            # working" -- four times. It was then made to merely lower the
            # window, on the reasoning that what anyone actually wants from
            # Escape is their keyboard back.
            #
            # What changed is that closing the whole app is no longer a trap:
            # it is a clean shutdown that releases the camera, and the
            # keyboard comes back with it. Alt-tab still gives him the
            # keyboard without leaving, which is what the old behaviour was
            # really for.
            #
            # Same route as F11 and the EXIT tile: hide at once so the screen
            # is his on the keystroke, then ask the owner to close on the
            # next turn of the event loop. Never destroy a QObject from
            # inside its own handler.
            self.hide()
            QTimer.singleShot(0, self.quit_requested.emit)
            return
        if key in self.TOGGLES and self.scene.get(self.TOGGLES[key]) is not None:
            self.scene.toggle(self.TOGGLES[key])
            self.update()
            return
        self.key_pressed.emit(key)
        super().keyPressEvent(ev)
