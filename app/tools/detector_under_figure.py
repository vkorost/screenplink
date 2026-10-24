"""What does a figure on the target screen cost the DETECTOR?

    python tools/detector_under_figure.py [seconds-per-scenario]

The number that matters is not the UI thread's paint time. The capture loop is
`read + detect`, and the slower half sets the pace: crossing ~14 ms per frame
halves frames-per-pulse (a 100 ms laser pulse lands in six camera frames
instead of three). This project measured exactly that once before: a HUD
repainting a 4K screen at 60 Hz cost 8.53 ms a paint and dragged the detector
from 6.96 ms to 24.15 ms.

So this runs the REAL detector (`LaserDetector.process`, a saved-style gain map
applied, 1280x720 frames) on one thread, flat out, while the main thread runs
the real scene exactly as the target window does -- a 60 Hz tick, a 3840x2160
repaint whenever any layer asks for one -- and reports the detector's
per-frame time in each scenario:

  nothing painting          the detector alone
  static targets, idle      the default screen: no repaints requested
  image target, idle        a figure standing still: no repaints requested
  advancing, 1 figure       a figure walking, 20 Hz repaints
  advancing, COUNT=2        two figures, 20 Hz repaints
  advancing + 40 holes      one figure walking while carrying 40 hit holes (the most it
                            carries), each redrawn every repaint
  bouncing balls (60 Hz)    the reference: a mode that DOES repaint every tick,
                            so the instrument is shown to see contention

THE GATE is not the mean. The hazard is a cliff, not a slope: a frame over ~14 ms
costs a camera frame, and a 100 ms pulse then lands in fewer of them. So the exit
status (0 pass, 1 fail) is decided by the figure scenarios' p95 frame time
(< 14 ms) and by the FRACTION of frames over 14 ms (<= 4.0%). The share was
measured at 0.6-2.6% for a walking figure or two, 0.2% idle, and 1.8-3.3% for the
reference (the 60 Hz bouncing-balls mode, which repaints every tick and is the
heavy case this project already knows); 4% leaves headroom over run-to-run noise
and is far from a cliff. The mean is printed and gates nothing.

Writes nothing, opens no camera and no audio device. Offscreen Qt.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402
from PySide6.QtCore import QRect                              # noqa: E402
from PySide6.QtGui import QImage, QPainter                    # noqa: E402
from PySide6.QtWidgets import QApplication                    # noqa: E402

from lsa.detect.laser import DetectorParams, LaserDetector    # noqa: E402
from lsa.display.layers import BulletHoles, SolidBackground   # noqa: E402
from lsa.display.scene import Scene                           # noqa: E402
from lsa.modes import ModeHost, available, by_name            # noqa: E402

CAM_W, CAM_H = 1280, 720
SCREEN = QRect(0, 0, 3840, 2160)
BUDGET_MS = 14.0
MAX_OVER_PCT = 4.0        # of frames over BUDGET_MS, in any figure scenario


def frames():
    rng = np.random.default_rng(11)
    out = []
    for i in range(12):
        f = rng.integers(8, 26, (CAM_H, CAM_W, 3), dtype=np.uint8)
        cv2.rectangle(f, (120, 90), (900, 560), (46, 44, 42), -1)
        cv2.rectangle(f, (1030, 40), (1270, 300), (190, 196, 200), -1)
        if i % 3 == 0:
            cv2.circle(f, (520, 330), 7, (30, 30, 230), -1)
        out.append(f)
    return out


def gain_map():
    """The shape his learned sensitivity produces: ~2x across the panel."""
    xs = np.tile(np.arange(CAM_W, dtype=np.float32), (CAM_H, 1))
    return np.clip(1.0 + 1.1 * (xs / CAM_W), 0.25, 4.0).astype(np.float32)


class DetectorLoop(threading.Thread):
    """The capture thread's detect half, flat out, timing every frame."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.det = LaserDetector(DetectorParams())
        self.det.set_gain_map(gain_map())
        self.frames = frames()
        self.times: list[float] = []
        self.stop = threading.Event()
        self.measuring = threading.Event()

    def run(self) -> None:
        i = 0
        while not self.stop.is_set():
            f = self.frames[i % len(self.frames)]
            t0 = time.perf_counter()
            self.det.process(f, t0)
            dt = (time.perf_counter() - t0) * 1000.0
            if self.measuring.is_set():
                self.times.append(dt)
            i += 1


def scenario(name: str, mode, seconds: float, loop: DetectorLoop, tick_hz: float = 60.0):
    scene = Scene()
    scene.add(SolidBackground())
    holes = BulletHoles()
    for k in range(40):
        holes.add(300.0 + 80 * (k % 40), 300.0 + 40 * (k // 8), k, 8)
    scene.add(holes)
    host = ModeHost()
    scene.add(host)
    img = QImage(SCREEN.width(), SCREEN.height(), QImage.Format_RGB32)
    if mode is not None:
        host.mode = mode              # already started by the caller
    paints, paint_ms = 0, []
    loop.times.clear()
    loop.measuring.set()
    end = time.perf_counter() + seconds
    nxt = time.perf_counter()
    while time.perf_counter() < end:
        if mode is not None and scene.advance(SCREEN):
            t0 = time.perf_counter()
            img.fill(0xFF0A0A0A)
            p = QPainter(img)
            scene.paint(p, SCREEN)
            p.end()
            paint_ms.append((time.perf_counter() - t0) * 1000.0)
            paints += 1
        nxt += 1.0 / tick_hz
        time.sleep(max(0.0, nxt - time.perf_counter()))
    loop.measuring.clear()
    t = np.array(loop.times)
    rate = paints / seconds
    return {"name": name, "frames": len(t), "mean": float(t.mean()),
            "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "max": float(t.max()), "fps": len(t) / seconds, "repaints_hz": rate,
            "over": float((t > BUDGET_MS).mean() * 100.0),
            "paint_ms": float(np.mean(paint_ms)) if paint_ms else 0.0}


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    QApplication.instance() or QApplication([])
    loop = DetectorLoop()
    loop.start()
    time.sleep(1.0)                                   # let the background model settle

    def advancing(count: int):
        m = by_name("advancing")()
        m.COUNT = count
        m._rng.seed(4)
        return m

    def walking(m):
        """Start it, then skip to the first walk so no blank wait is measured."""
        ctx = Scene()._context(time.perf_counter(), 0.0, SCREEN)
        m.start(ctx)
        for a in m._actors:
            a.live_at = ctx.now
        return m

    def carrying(m, n: int = 40):
        """Begin the walk, then put `n` hit holes on the figure (a grid, so none overlap)."""
        ctx = Scene()._context(time.perf_counter(), 0.0, SCREEN)
        m.advance(ctx)                                   # the figure appears
        a = m._actors[0]
        c, h = a.state(ctx.now)
        left, top, w, hh = a.fig.box(c, h)
        for i in range(n):
            u, v = 0.20 + (i % 8) * 0.085, 0.30 + (i // 8) * 0.08
            a.marks.add(a.fig, left + u * w, top + v * hh, c, h, ctx.now - 9.0, "10",
                        i % 3 != 0)
        return m

    # warm the figure pyramids, so loading a JPEG is not charged to the walk
    results = [
        scenario("nothing painting", None, seconds, loop),
        scenario("static targets, idle", _started("static"), seconds, loop),
        scenario("image target, idle", _image(), seconds, loop),
        scenario("advancing, 1 figure walking", walking(advancing(1)), seconds, loop),
        scenario("advancing, COUNT=2", walking(advancing(2)), seconds, loop),
        scenario("advancing, 1 figure walking + 40 holes carried",
                 carrying(walking(advancing(1))), seconds, loop),
        scenario("bouncing balls (60 Hz repaint, the reference)", _started("balls"),
                 seconds, loop),
    ]
    loop.stop.set()
    base = results[0]["mean"]
    print(f"\ndetector frame time on this machine (1280x720, gain map on), "
          f"{seconds:.0f} s per scenario; budget {BUDGET_MS:.0f} ms\n")
    print(f"{'scenario':48s} {'mean':>6s} {'p50':>6s} {'p95':>6s} {'max':>7s} "
          f"{'det fps':>8s} {'>14ms':>6s} {'repaints/s':>10s} {'paint ms':>8s}")
    for r in results:
        print(f"{r['name']:48s} {r['mean']:6.2f} {r['p50']:6.2f} {r['p95']:6.2f} "
              f"{r['max']:7.2f} {r['fps']:8.1f} {r['over']:5.1f}% {r['repaints_hz']:10.1f} "
              f"{r['paint_ms']:8.2f}")
    figs = [r for r in results if "reference" not in r["name"]]   # all but the 60 Hz one
    ref = next(r for r in results if "reference" in r["name"])
    worst_p95 = max(r["p95"] for r in figs)
    worst_over = max(r["over"] for r in figs)
    ok = worst_p95 < BUDGET_MS and worst_over <= MAX_OVER_PCT
    print(f"\nGATE: worst p95 {worst_p95:.2f} ms (must be < {BUDGET_MS:.0f}), worst share of "
          f"frames over {BUDGET_MS:.0f} ms {worst_over:.1f}% (must be <= {MAX_OVER_PCT:.1f}%) "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"for information: worst mean {max(r['mean'] for r in figs):.2f} ms, idle "
          f"baseline {base:.2f} ms; reference (60 Hz) over-budget share "
          f"{ref['over']:.1f}%")
    return 0 if ok else 1


def _started(name: str):
    m = by_name(name)()
    m.start(Scene()._context(time.perf_counter(), 0.0, SCREEN))
    return m


def _image():
    m = by_name("image")()
    m._rng.seed(4)
    m.start(Scene()._context(time.perf_counter(), 0.0, SCREEN))
    return m


if __name__ == "__main__":
    raise SystemExit(main())
