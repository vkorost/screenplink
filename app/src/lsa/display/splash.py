"""The startup splash: a logo on the monitor the camera is watching.

WHY THIS IS NOT JUST "SHOW AN IMAGE"
------------------------------------
The camera is pointed at this screen, and to the detector this picture is a
laser. Measured on the supplied artwork: red excess **229** on the dot over
the "i", 69,512 pixels above 30, 2.65% of the frame lit above luminance 60.
The house limits are 8 for anything that appears and 2% lit area. If it is
up while the noise floor is measured, the floor is taken FROM THE SPLASH and
the bar lands above every real shot (the verdict board at red excess 24, a
tenth of this, cost a whole session on 2026-10-15).

So the splash may be visible only while nothing is being measured. That is
held by CameraLab, not by anybody's ordering: `CameraLab._splash_down` is
called from every owner of the screen (see `_install_owner_hooks`), and the
check is `tests/test_splash.py`.

NEVER TRAP THE USER (RULES B6)
------------------------------
Escape, Space, Q and a double click all retire it, and so does its own
deadline, whatever the caller does or fails to do. A splash left up because
the camera never answered is the black-rectangle incident again.

NOTHING IS SIZED IN PIXELS (RULES B5)
-------------------------------------
The image is fitted to the screen by fraction and never distorted; any spare
space is the image's own background colour. The problem text is sized as a
fraction of the screen height.

COST
----
Decoding the 7680x4320 PNG is ~345 ms and the scale to screen size ~20 ms.
Both happen on a worker thread so the window is up at once and the decode
overlaps the construction of the app; the result is the screen-sized image,
which is all that is ever kept.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QScreen
from PySide6.QtWidgets import QApplication, QWidget

from lsa.artwork import assets

# His startup screen, supplied 2026-10-17 22:24. MEASURED before use, because
# the camera watches this monitor: red excess 218 at the laser dot over the
# "i", and 1.68% of the frame lit above luminance 60. Both are far over the
# limits for anything that APPEARS -- 8, and a 2% lit-area budget -- which is
# why the splash must be down before the first measurement. See _splash_down,
# which is driven off the screen-owner list rather than off call order, so a
# measurement added later is covered without anyone remembering.
SPLASH_NAME = "StartupScreen-v01.jpg"


def splash_path() -> Path:
    """The startup art, resolved when asked: LSA_IMAGES and a frozen bundle both win."""
    return assets.asset_dir("images") / SPLASH_NAME


SPLASH_PATH = splash_path()          # the default at import, kept for callers that name it
_DEFAULT_PATH = object()             # "the default, resolved at construction"

# Where the problem text sits, as fractions of the screen height. The logo
# and its tagline occupy the band 0.30-0.55 of the artwork; this is below it.
PROBLEM_Y_FRAC = 0.70
PROBLEM_FONT_FRAC = 0.028
PROBLEM_COLOUR = "#b4b4b4"          # true grey: red excess 0, nothing to read


def fit_rect(screen_w: int, screen_h: int, img_w: int, img_h: int
             ) -> tuple[int, int, int, int]:
    """Largest centred rectangle of the image's own aspect inside the screen.

    Pure and pixel-free: a function of ratios only, so it is right on a 4K
    monitor, a 75" TV and a projector alike, and cannot distort.
    """
    if min(screen_w, screen_h, img_w, img_h) <= 0:
        return (0, 0, 0, 0)
    scale = min(screen_w / img_w, screen_h / img_h)
    w = max(1, int(round(img_w * scale)))
    h = max(1, int(round(img_h * scale)))
    w = min(w, screen_w)
    h = min(h, screen_h)
    return ((screen_w - w) // 2, (screen_h - h) // 2, w, h)


class _Loader(QThread):
    """Decode once, scale once, off the GUI thread."""

    ready = Signal(QImage)

    def __init__(self, path: Path, target_w: int, target_h: int, parent=None):
        super().__init__(parent)
        self._path = path
        self._tw = target_w
        self._th = target_h

    def run(self) -> None:
        img = QImage(str(self._path))
        if img.isNull():
            return
        x, y, w, h = fit_rect(self._tw, self._th, img.width(), img.height())
        self.ready.emit(img.scaled(w, h, Qt.IgnoreAspectRatio,
                                   Qt.SmoothTransformation))


class Splash(QWidget):

    retired = Signal(str)             # why; emitted once, after the unwind

    NORMAL_DEADLINE_S = 30.0
    # Long, because it says why the app cannot start; but never unbounded.
    PROBLEM_DEADLINE_S = 120.0

    def __init__(self, image_path: Path | None = _DEFAULT_PATH,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint
                         | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("ScreenPlink")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.StrongFocus)
        # NO MOUSE CURSOR, FOR THIS WINDOW'S WHOLE LIFE (2026-10-22). Same
        # reasoning as `calibrate/flash.py` and `display/window.py`: Windows
        # composites the cursor over a window the camera is watching, and no
        # grab of the window can see it. On this widget, never
        # `setOverrideCursor`. No show-on-move: it is up briefly and never
        # aimed at. Escape, Space, Q, a double click (anywhere on a window
        # that covers the screen) and the deadline are unaffected.
        self.setCursor(Qt.BlankCursor)
        self._path = splash_path() if image_path is _DEFAULT_PATH else image_path
        self._pix: QImage | None = None
        self._bg = QColor("#0e0e0e")
        self._problem = ""
        self._done = False
        self._loader: _Loader | None = None
        self.retire_reason = ""       # declared, never read via getattr (C8)

        self._deadline = QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(lambda: self.retire("timeout"))

    # -- state ----------------------------------------------------------------

    @property
    def up(self) -> bool:
        """Is it on the screen (or about to be)? False once retired."""
        return (not self._done) and self.isVisible()

    @property
    def problem(self) -> str:
        return self._problem

    # -- lifecycle ------------------------------------------------------------

    def start(self, screen: QScreen | None = None,
              deadline_s: float | None = None) -> None:
        screen = screen or QApplication.primaryScreen()
        tw = th = 0
        if screen is not None:
            self.setGeometry(screen.geometry())
            dpr = screen.devicePixelRatio()
            tw = int(screen.geometry().width() * dpr)
            th = int(screen.geometry().height() * dpr)
        self._done = False
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.OtherFocusReason)
        self._deadline.start(int((deadline_s or self.NORMAL_DEADLINE_S) * 1000))
        if self._path is not None and Path(self._path).exists() and tw and th:
            self._loader = _Loader(Path(self._path), tw, th, self)
            self._loader.ready.connect(self._on_image)
            self._loader.start()

    def _on_image(self, img: QImage) -> None:
        if self._done:
            return
        self._pix = img
        if img.width() > 0 and img.height() > 0:
            self._bg = QColor(img.pixel(0, 0))
        self.update()

    def show_problem(self, text: str) -> None:
        """Keep the splash up and say what is wrong, under the logo.

        Only ever called when nothing is measuring (the camera did not open),
        so there is nothing for it to poison. Escape still closes it.
        """
        if self._done:
            return
        self._problem = text
        self._deadline.start(int(self.PROBLEM_DEADLINE_S * 1000))
        self.update()

    def retire(self, why: str) -> bool:
        """Take it down NOW. Returns True if this call did it.

        Hides first (the screen is free from here on) and announces on the
        next event-loop turn, because an owner's handler may drop the last
        reference to this window while this method is still on the stack:
        the same crash as `FlashWindow._give_up` (0xc0000409).
        """
        if self._done:
            return False
        self._done = True
        self.retire_reason = why
        self._deadline.stop()
        self.hide()
        if self._loader is not None:
            try:
                self._loader.ready.disconnect(self._on_image)
            except Exception:
                pass
        self._pix = None
        QTimer.singleShot(0, lambda: self.retired.emit(why))
        return True

    def dispose(self) -> None:
        """Release it. Waits for the decode thread first: destroying a
        running QThread is a fail-fast in Qt6."""
        if self._loader is not None:
            self._loader.wait(3000)
        self.deleteLater()

    def keyPressEvent(self, ev) -> None:
        if ev.key() in (Qt.Key_Escape, Qt.Key_Space, Qt.Key_Q):
            self.retire("escape")
            return
        super().keyPressEvent(ev)

    def mouseDoubleClickEvent(self, _ev) -> None:
        self.retire("escape")

    def closeEvent(self, ev) -> None:
        self._deadline.stop()
        super().closeEvent(ev)

    # -- painting -------------------------------------------------------------

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), self._bg)
        if self._pix is not None:
            x, y, w, h = fit_rect(self.width(), self.height(),
                                  self._pix.width(), self._pix.height())
            p.drawImage(QRect(x, y, w, h), self._pix)
        if self._problem:
            font = QFont("Segoe UI")
            font.setPixelSize(max(12, int(self.height() * PROBLEM_FONT_FRAC)))
            p.setFont(font)
            p.setPen(QColor(PROBLEM_COLOUR))
            box = QRect(int(self.width() * 0.15), int(self.height() * PROBLEM_Y_FRAC),
                        int(self.width() * 0.70), int(self.height() * 0.2))
            p.drawText(box, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
                       self._problem + "\n\n(Esc closes this)")
        p.end()
