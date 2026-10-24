"""Does auto-alignment actually recover a mapping that has drifted?

The camera is simulated rather than attached, but not naively: the "camera view"
is the real rendered target screen pushed through a genuine oblique homography,
then darkened and noised to match what a short laser-hunting exposure produces.
That is the regime the aligner has to work in, and a test that skipped it would
prove nothing about the only case that matters.

Three things are checked, and the third is the one that keeps this feature safe:

  1. A drifted mapping gets measurably closer to the truth.
  2. A mapping that is already correct is not disturbed.
  3. A camera that has MOVED, or a screen with nothing on it, is refused rather
     than fitted. Silently fitting those is how an auto-correcting system
     destroys a good calibration and leaves no trace of why.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402
from PySide6.QtCore import QRect                                  # noqa: E402
from PySide6.QtGui import QImage, QPainter                        # noqa: E402
from PySide6.QtWidgets import QApplication                        # noqa: E402

from lsa.calibrate.align import AutoAligner                       # noqa: E402
from lsa.display.scene import Scene                               # noqa: E402
from lsa.display.layers import AimField, SolidBackground, StatsHud, BulletHoles  # noqa: E402
from lsa.display.session import SessionStats                      # noqa: E402

SW, SH = 1920, 1080
CAM_W, CAM_H = 1280, 720

# A real off-axis view: the same trapezoid shape the Kiyo Pro produces.
SCREEN_IN_CAMERA = np.float32([[331, 91], [975, 165], [1011, 537], [371, 667]])


def render_screen() -> np.ndarray:
    """The target screen as the app actually paints it, as a BGR array."""
    session = SessionStats()
    scene = Scene(session)
    scene.add(SolidBackground())
    scene.add(AimField())
    holes = scene.add(BulletHoles())
    scene.add(StatsHud())
    for x, y in ((600, 400), (1300, 700), (900, 300)):
        holes.add(float(x), float(y), 0, 9)
    img = QImage(SW, SH, QImage.Format_RGB32)
    p = QPainter(img)
    scene.paint(p, QRect(0, 0, SW, SH))
    p.end()
    # Round-trip through a file rather than poking at Qt's buffer. constBits()
    # hands back a view into memory Qt still owns, and reading it from Python
    # has proved to segfault here; a PNG is slower and completely reliable, and
    # a test that crashes teaches nothing.
    # Gitignored: it is rewritten on every run and nothing compares
    # against it, so tracking it meant every commit carried a
    # spurious binary change -- which hides the real ones.
    out = Path(__file__).resolve().parent / "_align_reference.png"
    assert img.save(str(out)), "could not write the reference render"
    arr = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert arr is not None, "could not read the reference render back"
    return arr


def truth_homography() -> np.ndarray:
    """Camera pixels -> screen pixels, the mapping the aligner should find."""
    dst = np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]])
    return cv2.getPerspectiveTransform(SCREEN_IN_CAMERA, dst).astype(np.float64)


def fake_camera(painted: np.ndarray, H_cam_to_screen: np.ndarray) -> np.ndarray:
    """What the camera would see: the screen, oblique, dark and noisy."""
    ok, inv = cv2.invert(H_cam_to_screen)
    assert ok
    view = cv2.warpPerspective(painted, inv, (CAM_W, CAM_H))
    # A laser-hunting exposure renders the screen as a ghost. Measured on the
    # real camera: screen content sat around a quarter of full scale.
    view = (view.astype(np.float32) * 0.28).astype(np.uint8)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 3.0, view.shape).astype(np.float32)
    return np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def corner_error(H: np.ndarray, truth: np.ndarray) -> float:
    """Worst screen-corner disagreement between two mappings, in camera px."""
    c = np.array([[[0, 0]], [[SW, 0]], [[SW, SH]], [[0, SH]]], dtype=np.float32)
    _, ia = cv2.invert(np.asarray(H, dtype=np.float64))
    _, ib = cv2.invert(np.asarray(truth, dtype=np.float64))
    pa = cv2.perspectiveTransform(c, ia).reshape(-1, 2)
    pb = cv2.perspectiveTransform(c, ib).reshape(-1, 2)
    return float(np.max(np.linalg.norm(pa - pb, axis=1)))


def drift(H: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Nudge the camera, i.e. shift where the screen appears in its frame."""
    shifted = SCREEN_IN_CAMERA + np.float32([dx, dy])
    dst = np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]])
    return cv2.getPerspectiveTransform(shifted, dst).astype(np.float64)


def main() -> int:
    # Held, not merely created: dropping the reference lets Qt collect the
    # application object out from under the widgets.
    app = QApplication.instance() or QApplication([])
    assert app is not None
    checks: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    painted = render_screen()
    truth = truth_homography()
    cam = fake_camera(painted, truth)
    aligner = AutoAligner()

    # 1. A drifted mapping should be pulled back toward the truth.
    for dx, dy in ((7.0, -5.0), (-12.0, 9.0)):
        bad = drift(truth, dx, dy)
        before = corner_error(bad, truth)
        res = aligner.refine(cam, painted, bad, (SW, SH))
        if res.accepted and res.homography is not None:
            after = corner_error(res.homography, truth)
            check(f"drift of ({dx:+.0f},{dy:+.0f})px is corrected",
                  after < before * 0.75,
                  f"{before:.1f}px -> {after:.1f}px, correlation {res.correlation:.3f}")
        else:
            check(f"drift of ({dx:+.0f},{dy:+.0f})px is corrected", False, res.reason)

    # 2. A correct mapping must survive untouched -- an auto-corrector that
    #    wanders when nothing is wrong is worse than none at all.
    res = aligner.refine(cam, painted, truth, (SW, SH))
    stay = corner_error(res.homography, truth) if res.homography is not None else 0.0
    check("a correct mapping is left alone", stay < 3.0,
          f"moved {stay:.2f}px, {'applied' if res.accepted else 'rejected: ' + res.reason}")

    # 3a. A camera that has genuinely moved is out of range, and must be
    #     reported rather than fitted.
    moved = drift(truth, 190.0, -120.0)
    res = aligner.refine(cam, painted, moved, (SW, SH))
    check("a large move is refused, not fitted", not res.accepted, res.reason)

    # 2b. Repeated passes should converge, not wander. The app runs this on a
    #     timer, so a correction that oscillates would be worse than none.
    H = drift(truth, -12.0, 9.0)
    trail = [corner_error(H, truth)]
    for _ in range(4):
        r = aligner.refine(cam, painted, H, (SW, SH))
        if r.accepted and r.homography is not None:
            H = r.homography
        trail.append(corner_error(H, truth))
    check("repeated passes converge and stay converged",
          trail[-1] <= trail[1] + 0.5 and trail[-1] < 5.0,
          " -> ".join(f"{e:.1f}" for e in trail))

    # 3b. A screen with no structure offers nothing to register against. Uniform
    #     grey rather than black, so this exercises the confidence gate itself
    #     and not merely the "frame is empty" short circuit.
    flat_painted = np.full_like(painted, 40)
    flat_cam = np.full_like(cam, 12)
    res = aligner.refine(flat_cam, flat_painted, truth, (SW, SH))
    check("a featureless screen changes nothing", not res.accepted, res.reason)

    # 3c. Garbage in must not produce a mapping.
    res = aligner.refine(None, painted, truth, (SW, SH))
    check("a missing frame is handled", not res.accepted, res.reason)

    passed = all(checks)
    print("\n" + (f"ALL {len(checks)} CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
