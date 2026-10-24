"""Which monitor is the camera actually looking at, and does it see all of it?

Phase 1's calibration needs this answer and so does any multi-monitor setup: the
target window has to open on the screen the camera watches, not merely on the
screen the operator console happens to be on. Shooting at the wrong display
looks exactly like a detector that does not work.

Method: flash each screen white, then black, and watch what the camera sees. The
screen in view produces a large brightness swing; the others produce almost
none. Then, on the winner, flash the four quadrants one at a time to check the
whole screen is inside the frame rather than only its middle.

Run:  .venv\\Scripts\\python.exe src\\lsa\\tools\\align_check.py [camera_index] [out_dir]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Flasher(QWidget):
    """A borderless fullscreen panel we can paint a flat colour into."""

    def __init__(self) -> None:
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.colour = QColor("#000000")
        self.region = None          # (x_frac, y_frac, w_frac, h_frac) or None for the whole screen
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#000000"))
        if self.region is None:
            p.fillRect(self.rect(), self.colour)
        else:
            fx, fy, fw, fh = self.region
            p.fillRect(int(self.width() * fx), int(self.height() * fy),
                       int(self.width() * fw), int(self.height() * fh), self.colour)
        p.end()


def grab(cap, n: int = 6):
    """Read a few frames and return the last; webcams lag a frame or two."""
    frame = None
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            frame = f
    return frame


def flash_measure(app, win, cap, colour: str, region=None, settle: float = 0.45):
    win.colour = QColor(colour)
    win.region = region
    win.update()
    app.processEvents()
    time.sleep(settle)
    app.processEvents()
    frame = grab(cap)
    return frame


def main() -> int:
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("align_check")
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    screens = app.screens()
    print(f"{len(screens)} screen(s) detected:")
    for i, s in enumerate(screens):
        g = s.geometry()
        print(f"  [{i}] {s.name()}  {g.width()}x{g.height()} at ({g.x()},{g.y()})"
              f"  primary={s is app.primaryScreen()}")

    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"cannot open camera index {index}")
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Auto exposure on purpose: this test is about SEEING the screen, which is the
    # opposite of the shooting profile. Calibration and detection want different
    # cameras, effectively, and switching between them is part of the design.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    grab(cap, 20)

    win = Flasher()
    results = []

    print("\nflashing each screen white, then black...")
    for i, screen in enumerate(screens):
        win.setGeometry(screen.geometry())
        win.showFullScreen()
        app.processEvents()
        time.sleep(0.35)

        black = flash_measure(app, win, cap, "#000000")
        white = flash_measure(app, win, cap, "#ffffff")
        if black is None or white is None:
            print(f"  [{i}] no frames from the camera")
            continue

        diff = cv2.absdiff(white, black)
        gdiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        score = float(gdiff.mean())
        lit = int((gdiff > 40).sum())
        pct = 100.0 * lit / gdiff.size
        print(f"  [{i}] {screen.name():<22} brightness swing {score:6.2f}   "
              f"{pct:5.1f}% of the camera frame changed")
        results.append((score, pct, i, screen, white, black, gdiff))
        cv2.imwrite(str(out / f"screen{i}_white.png"), white)
        cv2.imwrite(str(out / f"screen{i}_diff.png"), cv2.convertScaleAbs(gdiff, alpha=2.0))

    win.hide()
    app.processEvents()

    if not results:
        print("\nno measurements taken")
        cap.release()
        return 1

    results.sort(key=lambda r: -r[0])
    best = results[0]
    score, pct, i, screen, white, black, gdiff = best
    print(f"\nBEST MATCH: screen [{i}] {screen.name()} - swing {score:.2f}, {pct:.1f}% of frame")

    if score < 3.0:
        print("VERDICT: the camera does not appear to see ANY of these screens.")
        print("         Check that it is pointed at a monitor and nothing is blocking it.")
        cap.release()
        return 2

    if len(results) > 1 and results[1][0] > score * 0.6:
        print(f"WARNING: screen [{results[1][2]}] scored nearly as high "
              f"({results[1][0]:.2f}); the camera may see both.")

    # Where is that screen in the camera frame, and is all of it inside?
    mask = (gdiff > 40).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        H, W = gdiff.shape
        print(f"screen occupies camera pixels x {x}..{x + w}, y {y}..{y + h} "
              f"({100.0 * w * h / (W * H):.1f}% of frame)")
        touching = []
        if x <= 2:
            touching.append("left")
        if y <= 2:
            touching.append("top")
        if x + w >= W - 2:
            touching.append("right")
        if y + h >= H - 2:
            touching.append("bottom")
        if touching:
            print(f"WARNING: the screen runs off the {', '.join(touching)} edge of the "
                  f"camera frame. Shots there cannot be seen. Move the camera back.")
        else:
            print("the whole screen appears to be inside the camera frame")
        vis = white.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(out / "screen_bounds.png"), vis)

    # Quadrant check on the winning screen: are the corners visible too?
    print("\nchecking the four quadrants of that screen...")
    win.setGeometry(screen.geometry())
    win.showFullScreen()
    app.processEvents()
    quads = {"top-left": (0.0, 0.0, 0.5, 0.5), "top-right": (0.5, 0.0, 0.5, 0.5),
             "bottom-left": (0.0, 0.5, 0.5, 0.5), "bottom-right": (0.5, 0.5, 0.5, 0.5)}
    dark = flash_measure(app, win, cap, "#000000")
    weak = []
    for name, region in quads.items():
        lit_frame = flash_measure(app, win, cap, "#ffffff", region)
        d = cv2.cvtColor(cv2.absdiff(lit_frame, dark), cv2.COLOR_BGR2GRAY)
        changed = 100.0 * int((d > 40).sum()) / d.size
        flag = "" if changed > 1.0 else "   <-- barely visible"
        print(f"  {name:<13} {changed:5.2f}% of frame{flag}")
        if changed <= 1.0:
            weak.append(name)
        cv2.imwrite(str(out / f"quad_{name}.png"), cv2.convertScaleAbs(d, alpha=2.0))

    win.hide()
    cap.release()
    print()
    if weak:
        print(f"VERDICT: {', '.join(weak)} of the screen is not properly visible to the "
              f"camera. Shots landing there will be missed.")
    else:
        print("VERDICT: all four quadrants are visible. Camera placement is usable.")
    print(f"images written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
