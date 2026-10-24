"""The whole pattern-calibration pipeline, driven by a simulated camera.

No camera, no screen, no shooting. This exists because every fault in that
pipeline tonight was found by running it against real hardware and reading the
wreckage afterwards -- and each hardware run costs a minute, risks hanging the
camera (control writes do that on this one), and needs the room left alone.
Every fault below is reproduced here in milliseconds.

The simulation renders what the camera would actually see: a screen seen
off-axis through a lens with barrel distortion, several times dimmer at its far
edge than its near one, in a room containing a lamp that never changes and a
reflective surface beside the monitor that lights up whenever the screen does.
Then it runs the real `find_marker`, the real `prune_out_of_order` and the real
`Mapping.fit_validated`, in the order the app runs them.

The faults it reproduces, each of which cost a live run:

- a reflection beside the monitor, brighter than the marker it mirrors
- a photograph taken before the screen painted, leaving only noise
- a far side so dim that a fixed threshold loses two whole columns
- a marker simply missed
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2                                                         # noqa: E402
import numpy as np                                                 # noqa: E402

from lsa.calibrate.mapping import Mapping, Sample                  # noqa: E402
from lsa.calibrate.pattern import (find_marker, marker_points,     # noqa: E402
                                   plausible_quad, prune_out_of_order,
                                   screen_mask)

SW, SH = 3840, 2160
CW, CH = 1280, 720
MM_PER_PX = 940.0 / SW


class FakeBench:
    """A camera looking at a monitor, off-axis, in a room.

    `quad` is where the screen's corners land in the camera image. Deliberately
    asymmetric: the real one is viewed from the left, so the right-hand edge is
    both further away and more oblique.
    """

    def __init__(self, quad=((300, 120), (880, 190), (880, 500), (320, 600)),
                 k1: float = 0.045, seed: int = 7):
        self.h = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]]), np.float32(quad))
        self.k1 = k1
        self.rng = np.random.default_rng(seed)

    def to_camera(self, sx, sy):
        q = cv2.perspectiveTransform(np.float32([[[sx, sy]]]), self.h)[0][0]
        ux, uy = (q[0] - CW / 2) / (CW / 2), (q[1] - CH / 2) / (CH / 2)
        r2 = ux * ux + uy * uy
        return (CW / 2 + ux * (1 + self.k1 * r2) * (CW / 2),
                CH / 2 + uy * (1 + self.k1 * r2) * (CH / 2))

    def _panel(self, img, value_near, value_far):
        """The monitor itself, fading towards its far (right) edge."""
        for i in range(24):
            f0, f1 = i / 24.0, (i + 1) / 24.0
            poly = np.int32([self.to_camera(SW * f0, 0), self.to_camera(SW * f1, 0),
                             self.to_camera(SW * f1, SH), self.to_camera(SW * f0, SH)])
            v = int(value_near + (value_far - value_near) * f0)
            cv2.fillConvexPoly(img, poly, (v, v, v))

    def frames(self, screen_pt, *, marker=True, reflection=False,
               painted=True, dim_far=True, decoy=None):
        """Return (lit, dark) as the camera would see them.

        `decoy` is a brighter, larger blob at a given CAMERA position, ON the
        panel -- the specular highlight these notes record on the right of
        this monitor. Unlike `reflection`, which sits beside the screen and the
        screen mask therefore excludes, a decoy is inside the mask and only
        geometry can reject it.
        """
        dark = self.rng.integers(0, 3, (CH, CW, 3), dtype=np.uint8)
        cv2.rectangle(dark, (1180, 60), (1270, 140), (235, 235, 235), -1)  # a lamp
        lit = dark.copy()
        if marker and painted:
            cx, cy = self.to_camera(*screen_pt)
            # Dimmer the further right it is: the real falloff is about 4x.
            fade = 1.0 - (0.72 * screen_pt[0] / SW if dim_far else 0.0)
            v = int(230 * fade)
            cv2.circle(lit, (int(round(cx)), int(round(cy))), 9, (v, v, v), -1)
            if reflection:
                # Beside the monitor, seen face-on, so brighter and larger than
                # the marker it mirrors.
                cv2.circle(lit, (1120, int(cy)), 16, (250, 250, 250), -1)
        if decoy is not None:
            cv2.circle(lit, (int(decoy[0]), int(decoy[1])), 15,
                       (252, 252, 252), -1)
        return lit, dark

    def full_flash(self):
        dark = self.rng.integers(0, 3, (CH, CW, 3), dtype=np.uint8)
        cv2.rectangle(dark, (1180, 60), (1270, 140), (235, 235, 235), -1)
        lit = dark.copy()
        self._panel(lit, 210, 58)            # near edge bright, far edge ~4x dimmer
        return lit, dark


def run_calibration(bench, *, reflection_on=(), unpainted=(), missing=(),
                    decoys=None, use_prediction=True, disc_frac=0.20,
                    widen_on_miss=True, check_corners=True, trace=None,
                    bad_mask=None, recover_mask=True):
    """Drive the real pipeline exactly as the app does. Returns (mapping, note).

    `trace`, if a dict is passed, collects what the app logs: the quad verdict
    and every prediction-versus-actual distance. That is the one thing the app
    could not say when four corners were found and the other eight were
    reported invisible.
    """
    pts = marker_points(SW, SH, cols=4, rows=3, margin=0.08)
    corners = [0, 3, 8, 11]
    order = corners + [i for i in range(len(pts)) if i not in corners]
    decoys = decoys or {}

    mask = screen_mask(*bench.full_flash())
    if bad_mask is not None:
        mask = bad_mask
    mask_retried = False
    found: dict[int, tuple] = {}

    attempt = list(order)
    for idx in attempt:
        pt = pts[idx]
        if idx in missing:
            continue
        lit, dark = bench.frames(pt, reflection=idx in reflection_on,
                                 painted=idx not in unpainted,
                                 decoy=decoys.get(idx))
        near = None
        if use_prediction and len(found) >= 4:
            trust = True
            if check_corners and len(found) == 4:
                ok, why = plausible_quad([(pts[i], found[i]) for i in found])
                trust = ok
                if trace is not None:
                    trace.setdefault("quad", (ok, why))
            if trust:
                src = np.float32([pts[i] for i in found])
                dst = np.float32([found[i] for i in found])
                if (len({round(p[0]) for p in src}) >= 2
                        and len({round(p[1]) for p in src}) >= 2):
                    h, _m = cv2.findHomography(src, dst, 0)
                    if h is not None:
                        q = cv2.perspectiveTransform(
                            np.float32([[[pt[0], pt[1]]]]), h)[0][0]
                        near = (float(q[0]), float(q[1]), CW * disc_frac)
        got = find_marker(lit, dark, mask=mask, near=near)
        widened = False
        if got is None and near is not None and widen_on_miss:
            # A wrong prediction must cost accuracy, never a marker.
            got = find_marker(lit, dark, mask=mask, near=None)
            widened = got is not None
        if trace is not None and near is not None:
            trace.setdefault("preds", []).append(
                (idx, None if got is None
                 else math.hypot(got[0] - near[0], got[1] - near[1]), widened))
        if got is not None:
            found[idx] = (got[0], got[1])

        # The app's own recovery: once the four corners are in, check that
        # they look like a screen. If they do not, the monitor outline is the
        # likeliest culprit -- it passed a generic coverage band and was still
        # wrong -- so throw it away and measure the corners again without it.
        if (recover_mask and not mask_retried and mask is not None
                and idx == corners[-1]):
            # Missing corners count as well as misplaced ones. A mask that
            # clips the panel hides them outright, so the quad test never gets
            # four points to judge -- which is the live "6 points, held-out
            # 386mm" failure: six markers in two columns.
            got4 = [i for i in corners if i in found]
            if len(got4) < 4:
                ok_q = False
            else:
                ok_q, _why = plausible_quad([(pts[i], found[i]) for i in got4])
            if not ok_q:
                mask_retried = True
                mask = None
                found = {}
                if trace is not None:
                    trace["mask_dropped"] = True
                for idx2 in corners:
                    lit2, dark2 = bench.frames(
                        pts[idx2], reflection=idx2 in reflection_on,
                        painted=idx2 not in unpainted, decoy=decoys.get(idx2))
                    g2 = find_marker(lit2, dark2, mask=None, near=None)
                    if g2 is not None:
                        found[idx2] = (g2[0], g2[1])

    if trace is not None:
        trace["found"] = sorted(found)
    ordered = [((pts[i][0], pts[i][1]), found[i]) if i in found else None
               for i in range(len(pts))]
    kept, _dropped = prune_out_of_order(ordered, 4, 3)
    samples = [Sample(screen=sc, camera=cm) for sc, cm in kept]
    if len(samples) < 4:
        return None, f"only {len(samples)} markers"
    return Mapping.fit_validated(samples, (SW, SH), (CW, CH))


def held_out_mm(bench, mapping, n=6):
    """Error at positions the fit never saw, in millimetres."""
    checks = marker_points(SW, SH, cols=3, rows=2, margin=0.22)[:n]
    errs = []
    for sx, sy in checks:
        cam = bench.to_camera(sx, sy)
        got = mapping.to_screen(*cam)
        if got is not None:
            errs.append(math.hypot(got[0] - sx, got[1] - sy) * MM_PER_PX)
    return errs


def main() -> int:
    checks: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              f"{(' -- ' + detail) if detail else ''}")

    print("\n1. a clean bench")
    bench = FakeBench()
    mapping, note = run_calibration(bench)
    check("twelve markers give a mapping", mapping is not None, note)
    if mapping is None:
        print("\ncannot continue")
        return 1
    errs = held_out_mm(bench, mapping)
    worst = max(errs)
    print(f"         held-out: median {sorted(errs)[len(errs) // 2]:.2f}mm "
          f"worst {worst:.2f}mm   ({note})")
    check("and it is accurate where it never looked", worst < 3.0,
          f"worst {worst:.2f}mm")
    check("including the far column the laser cannot reach",
          len({round(s.screen[0]) for s in mapping.samples}) == 4,
          f"{len({round(s.screen[0]) for s in mapping.samples})} columns")

    print("\n2. the far side is four times dimmer")
    # The fault: a fixed threshold found the near markers and missed the far
    # ones, so the fit fell back to two columns and reported a small error
    # while being badly wrong everywhere else.
    cols = len({round(s.screen[0]) for s in mapping.samples})
    check("a dim far side does not cost a column", cols >= 3, f"{cols} columns")

    print("\n3. a reflection beside the monitor")
    # Brighter and larger than the marker it mirrors, because it is seen
    # face-on rather than at seventy degrees.
    bench3 = FakeBench(seed=11)
    m3, n3 = run_calibration(bench3, reflection_on=(5, 6))
    check("a mapping still comes out", m3 is not None, n3)
    if m3 is not None:
        e3 = held_out_mm(bench3, m3)
        check("and the reflection does not corrupt it", max(e3) < 5.0,
              f"worst {max(e3):.2f}mm")

    print("\n4. a photograph taken before the screen painted")
    # The difference image then holds nothing but noise, and the strongest
    # noise blob wins. Live, that produced a 'marker' at strength 14 among
    # siblings reading 197-231.
    bench4 = FakeBench(seed=3)
    m4, n4 = run_calibration(bench4, unpainted=(5, 9))
    check("the pipeline survives two missed paints", m4 is not None, n4)
    if m4 is not None:
        e4 = held_out_mm(bench4, m4)
        check("without poisoning the mapping", max(e4) < 5.0,
              f"worst {max(e4):.2f}mm")

    print("\n5. markers simply missing")
    bench5 = FakeBench(seed=5)
    m5, n5 = run_calibration(bench5, missing=(1, 6, 10))
    check("nine markers are enough", m5 is not None, n5)
    if m5 is not None:
        e5 = held_out_mm(bench5, m5)
        check("and still accurate", max(e5) < 4.0, f"worst {max(e5):.2f}mm")

    print("\n6. what the prediction is worth")
    # Four correspondences define a homography, so every later marker's
    # position is predictable. Without that constraint a reflection wins.
    bench6 = FakeBench(seed=13)
    m_no, _ = run_calibration(bench6, reflection_on=(5, 6, 7),
                              use_prediction=False)
    bench6b = FakeBench(seed=13)
    m_yes, _ = run_calibration(bench6b, reflection_on=(5, 6, 7),
                               use_prediction=True)
    e_no = max(held_out_mm(bench6, m_no)) if m_no else 1e9
    e_yes = max(held_out_mm(bench6b, m_yes)) if m_yes else 1e9
    print(f"         three reflections: without prediction {e_no:7.2f}mm, "
          f"with it {e_yes:7.2f}mm")
    check("the prediction is what makes it robust", e_yes < 5.0,
          f"{e_yes:.2f}mm with prediction")

    print("\n7. a corner found in the wrong place -- the hardware failure of 18:37")
    # The signature on hardware was: four corner markers found, then every
    # one of the remaining eight reported "not visible to the camera". That is
    # not a camera that cannot see the far third of the panel; it is a
    # prediction built on a bad corner. The corners are visited first so the
    # rest can be predicted, and with exactly four correspondences the
    # homography fits them perfectly whatever they are -- so there is no
    # residual to notice and nothing downstream to catch it.
    #
    # Here corner 3 (top-right, the dimmest marker on the panel) is beaten by
    # a specular highlight low on the panel. Inside the screen mask, so only
    # geometry can reject it.
    BAD_CORNER = {3: (400, 500)}
    pts = marker_points(SW, SH, cols=4, rows=3, margin=0.08)
    corrupted = [(pts[0], bench.to_camera(*pts[0])),
                 (pts[3], (400.0, 500.0)),
                 (pts[8], bench.to_camera(*pts[8])),
                 (pts[11], bench.to_camera(*pts[11]))]
    ok, why = plausible_quad(corrupted)
    check("the corner check rejects a corner on a reflection", not ok, why)
    ok_clean, why_clean = plausible_quad(
        [(pts[i], bench.to_camera(*pts[i])) for i in (0, 3, 8, 11)])
    check("and accepts four real corners", ok_clean, why_clean)

    # The old behaviour, reproduced: a 0.10 disc, no widening, no corner check.
    old_trace: dict = {}
    b7 = FakeBench(seed=17)
    m_old, n_old = run_calibration(b7, decoys=BAD_CORNER, disc_frac=0.10,
                                   widen_on_miss=False, check_corners=False,
                                   trace=old_trace)
    lost = 12 - len(old_trace.get("found", []))
    print(f"         old behaviour: {len(old_trace.get('found', []))} markers "
          f"found, {lost} lost  ({n_old})")
    check("the old code loses markers to the bad corner", lost >= 4,
          f"{lost} markers lost")

    # And the fix. Either defence alone is enough, which is the point of
    # having both: the corner check refuses to predict from a bad quad, and
    # the widened search finds the marker anyway when a prediction is merely
    # off rather than wrong.
    new_trace: dict = {}
    b7b = FakeBench(seed=17)
    m_new, n_new = run_calibration(b7b, decoys=BAD_CORNER, trace=new_trace)
    got = len(new_trace.get("found", []))
    check("the corner check fires", new_trace.get("quad", (True, ""))[0] is False,
          new_trace.get("quad", (None, "not reached"))[1])
    check("and no good marker is lost", got >= 11, f"{got} of 12 found")
    check("a mapping comes out", m_new is not None, n_new)
    if m_new is not None:
        e7 = held_out_mm(b7b, m_new)
        check("and the bad corner does not reach it", max(e7) < 3.0,
              f"worst {max(e7):.2f}mm")
        cols7 = len({round(s.screen[0]) for s in m_new.samples})
        check("all four columns survive", cols7 == 4, f"{cols7} columns")

    print("\n8. widening alone, with the corner check deliberately off")
    # Defence in depth: if the quad check were ever fooled, a wrong prediction
    # must still cost accuracy rather than a marker.
    b8 = FakeBench(seed=19)
    t8: dict = {}
    m8, n8 = run_calibration(b8, decoys=BAD_CORNER, check_corners=False,
                             widen_on_miss=True, trace=t8)
    outside = sum(1 for _i, _d, w in t8.get("preds", []) if w)
    print(f"         {len(t8.get('found', []))} markers found, {outside} of them "
          f"outside the predicted disc  ({n8})")
    check("markers outside a wrong disc are still found", outside >= 4,
          f"{outside} recovered by widening")
    check("and the mapping is still good", m8 is not None and max(held_out_mm(b8, m8)) < 3.0,
          f"worst {max(held_out_mm(b8, m8)):.2f}mm" if m8 else n8)

    print()
    print("9. a screen mask that is wrong but plausible-looking")
    # The live failure mode, measured across seventeen runs on the bench:
    #
    #   mask 44%   corners valid     held-out 0.6 - 1.0mm
    #   no mask    corners valid     held-out 0.6 - 1.1mm
    #   mask 29%   corners INVALID   held-out 386mm
    #   mask 52%   corners INVALID   held-out 298mm
    #
    # Both disasters came from a mask that passed a generic 8-60% coverage
    # band and was still wrong. Here the mask is deliberately clipped to the
    # near half of the panel, which is what a 29% mask IS: the far markers
    # fall outside it and cannot be found at all.
    bench9 = FakeBench(seed=23)
    clipped = np.zeros((CH, CW), np.uint8)
    cv2.rectangle(clipped, (240, 80), (600, 640), 255, -1)   # near half only
    t9: dict = {}
    m9, n9 = run_calibration(bench9, bad_mask=clipped, trace=t9)
    check("the bad mask is noticed and dropped",
          t9.get("mask_dropped") is True,
          "the run carried a mask that clips the panel all the way to the end")
    check("and a mapping still comes out", m9 is not None, n9)
    if m9 is not None:
        e9 = held_out_mm(bench9, m9)
        cols9 = len({round(s.screen[0]) for s in m9.samples})
        check("covering every column, not just the masked half",
              cols9 == 4, f"{cols9} columns")
        check("and accurate", max(e9) < 3.0, f"worst {max(e9):.2f}mm")

    # Without the recovery, the same mask is what 386mm looks like.
    bench9b = FakeBench(seed=23)
    m_no, n_no = run_calibration(bench9b, bad_mask=clipped, recover_mask=False)
    if m_no is None:
        print(f"         without recovery: no mapping at all ({n_no})")
        check("the recovery is what saves the run", True, "no mapping without it")
    else:
        e_no = max(held_out_mm(bench9b, m_no))
        cols_no = len({round(s.screen[0]) for s in m_no.samples})
        print(f"         without recovery: {cols_no} columns, "
              f"worst held-out {e_no:7.2f}mm")
        check("the recovery is what saves the run",
              e_no > 5.0 or cols_no < 4,
              f"{e_no:.2f}mm, {cols_no} columns -- the bad mask did no harm, "
              f"so this case is not reproducing the live failure")

    print()
    if all(checks):
        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
