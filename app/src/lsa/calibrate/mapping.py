"""Camera-to-screen mapping measured from real shots.

The screen-outline method finds the monitor by flashing it, which is fast and
needs no cooperation, but it has two failings that showed up immediately in use:

  1. If any part of the screen falls outside the camera frame, the detected
     "corner" is really the frame boundary. Fitting to it skews the whole
     mapping -- measured 7 to 28 mm of error on a 42.5" 4K panel, which is
     exactly the "holes are half an inch off" complaint.
  2. A four-corner homography assumes a pinhole camera. Real lenses bend
     straight lines, and at an oblique angle a laser dot's bloom is asymmetric,
     biasing its centroid. Neither is modelled.

Shooting a grid of known points fixes both, because it measures the thing we
actually care about -- where a shot at screen position P lands in the camera --
rather than inferring it from the screen's outline. It needs no corners, so a
clipped screen is no obstacle, and it absorbs lens distortion and centroid bias
because they are present in the samples themselves.

The fit is a homography plus a residual correction. The homography captures the
perspective, which is genuinely projective; the residuals capture everything
else, interpolated between the measured points by inverse-distance weighting.
That is deliberately humble: it corrects what was measured and decays to the
plain homography away from the samples, rather than extrapolating a model nobody
verified.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lsa.atomic import write_json_atomic
from lsa.playarea import PlayArea

Point = tuple[float, float]


@dataclass
class Sample:
    """One correspondence: a shot aimed at a known screen point."""

    screen: Point      # where the target was drawn
    camera: Point      # where the camera saw the dot
    peak: float = 0.0  # how strong the dot scored there; 0 if not recorded


@dataclass
class Mapping:
    """Camera pixels -> screen pixels, fitted from shots."""

    homography: np.ndarray
    samples: list[Sample] = field(default_factory=list)
    residuals: list[Point] = field(default_factory=list)   # screen-space, per sample
    screen_size: tuple[int, int] = (0, 0)
    rms_before: float = 0.0        # homography alone, screen px
    rms_after: float = 0.0         # with residual correction
    max_error: float = 0.0
    k1: float = 0.0                # radial lens term, applied before the homography
    frame_size: tuple[int, int] = (1280, 720)
    correction_power: float = 2.0
    correction_radius: float = 0.0  # px; 0 = global weighting
    # How brightly a real laser scores in each region of this screen, measured
    # from actual shots. It belongs with the mapping because it is a property
    # of the same thing -- this camera, in this position, looking at this
    # panel -- and because both are invalidated by exactly the same event, the
    # camera moving. Relearning it every session is what made it unlearnable.
    sensitivity: list = field(default_factory=list)
    # The screen radius at which the self test's emulated dot images at
    # about the size a real laser does in the camera -- 400 px of blob in
    # the middle of this panel, measured from 36 real shots.
    #
    # It belongs here for the same reason `sensitivity` does: it is a
    # property of this camera, in this position, looking at this panel, and
    # it is invalidated by exactly the same event. A screen EMITS its dot
    # and blooms in the sensor where a laser REFLECTS off matte coating and
    # does not, so the size cannot be computed -- it has to be tuned in a
    # feedback loop against what the camera reports, which costs four
    # flashes and about four seconds.
    #
    # Re-tuning it from scratch on every launch was four seconds of a
    # forty-one second startup, spent rediscovering a number that had not
    # changed. 0.0 means never measured.
    dot_radius: float = 0.0

    # WHERE A SHOT IS RELIABLY DETECTED, measured from real trigger pulls.
    #
    # Asked for 2026-10-16: "draw strict borders to highlight the area that
    # is supported by the current camera placement... mark that area
    # off-limits". Targets stay inside it; holes are still drawn outside it.
    #
    # It lives HERE, with the mapping, because it is a property of the
    # camera pose and of nothing else -- the same lifetime as the
    # sensitivity map and the tuned dot radius, invalidated by the same
    # event, the camera moving. Carried across a recalibration that did not
    # move the camera by `CameraLab.POSE_FACTS`, which is one list rather
    # than a third hand-written carry: the sensitivity was dropped that way
    # once, and the dot radius after it.
    #
    # None means never measured at this camera position, which is NOT the
    # same as "the whole screen is fine" and must be reported differently.
    play_area: "PlayArea | None" = None

    # -- use ------------------------------------------------------------------

    def _undistort(self, cx: float, cy: float) -> Point:
        """Straighten the lens before the homography sees the point.

        A homography is exact for a flat screen through a PINHOLE camera. A real
        wide-angle lens bends straight lines, and no 8-parameter projective fit
        can absorb that -- it just spreads the error around. One radial term is
        enough here: fitted on the live nine points it took leave-one-out error
        from 9.7mm to 6.5mm. A second term (k2) reached 6.1mm, which is not
        worth the extra freedom on nine samples.
        """
        if not self.k1:
            return cx, cy
        w, h = self.frame_size
        f = float(w) / 2.0
        x, y = (cx - w / 2.0) / f, (cy - h / 2.0) / f
        s = 1.0 + self.k1 * (x * x + y * y)
        return w / 2.0 + x * s * f, h / 2.0 + y * s * f

    def _redistort(self, ux: float, uy: float, iters: int = 12) -> Point:
        """Inverse of `_undistort`, by fixed-point iteration.

        Needed only to express a screen-shaped region back in real camera
        pixels for the detector's mask. The forward model is a contraction for
        the magnitudes involved here, so a dozen iterations converge well below
        a pixel.
        """
        if not self.k1:
            return ux, uy
        cx, cy = ux, uy
        for _ in range(iters):
            gx, gy = self._undistort(cx, cy)
            cx += ux - gx
            cy += uy - gy
        return cx, cy

    def to_screen(self, cx: float, cy: float, corrected: bool = True) -> Point:
        cx, cy = self._undistort(cx, cy)
        pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.homography)
        sx, sy = float(out[0][0][0]), float(out[0][0][1])
        if corrected and self.residuals:
            dx, dy = self._correction(sx, sy)
            sx, sy = sx + dx, sy + dy
        return sx, sy

    def _correction(self, sx: float, sy: float) -> Point:
        """Inverse-distance-weighted blend of the measured residuals."""
        num_x = num_y = denom = 0.0
        for sample, (rx, ry) in zip(self.samples, self.residuals):
            d2 = (sx - sample.screen[0]) ** 2 + (sy - sample.screen[1]) ** 2
            if d2 < 1e-6:
                return rx, ry                      # sitting on a sample
            if self.correction_radius and d2 > self.correction_radius ** 2:
                continue
            w = 1.0 / (d2 ** (self.correction_power / 2.0))
            num_x += w * rx
            num_y += w * ry
            denom += w
        if denom <= 0:
            return 0.0, 0.0
        return num_x / denom, num_y / denom

    # -- fit ------------------------------------------------------------------

    @classmethod
    def fit_validated(cls, samples: list[Sample], screen_size: tuple[int, int],
                      frame_size: tuple[int, int] = (1280, 720)
                      ) -> "tuple[Mapping | None, str]":
        """Fit, and let leave-one-out error decide whether to keep the correction.

        The residual correction is seductive: it drives error at the sample
        points to zero, which looks like success and is actually just
        interpolation reproducing its own inputs. Measured against held-out
        points on a synthetic oblique view with barrel distortion, it was
        slightly WORSE than the plain homography (1.86 mm vs 1.62 mm) -- nine
        points is simply not enough to learn a distortion field.

        So the choice is made by evidence, not preference: hold each sample out
        in turn, predict it from the rest both ways, and keep whichever
        generalises better. With enough well-spread points the correction will
        start to win on its own.
        """
        samples, dropped = cls._reject_outliers(samples, screen_size)
        plain = cls.fit(samples, screen_size, use_residuals=False, frame_size=frame_size)
        if plain is None:
            return None, "not enough points"
        prefix = f"dropped {dropped} stray shot(s); " if dropped else ""
        if len(samples) < 6:
            return plain, prefix + "homography only (too few points to validate a correction)"

        # Choose the lens term on HELD-OUT error. Picking it on the samples it
        # was fitted to would always prefer more curvature, which is how a model
        # ends up describing the calibration instead of the camera.
        k1, k1_rms = 0.0, cls._loo_rms(samples, screen_size, 0.0, frame_size)
        for cand in [round(v * 0.05, 2) for v in range(-12, 13)]:
            if cand == 0.0:
                continue
            r = cls._loo_rms(samples, screen_size, cand, frame_size)
            if r < k1_rms:
                k1, k1_rms = cand, r
        if k1:
            plain = cls.fit(samples, screen_size, use_residuals=False,
                            k1=k1, frame_size=frame_size) or plain
            prefix += f"lens term k1={k1:+.2f}; "

        err_plain, err_corr = [], []
        for i in range(len(samples)):
            rest = samples[:i] + samples[i + 1:]
            held = samples[i]
            a = cls.fit(rest, screen_size, False, k1, frame_size)
            b = cls.fit(rest, screen_size, True, k1, frame_size)
            if a is None or b is None:
                continue
            ax, ay = a.to_screen(*held.camera, corrected=False)
            bx, by = b.to_screen(*held.camera, corrected=True)
            err_plain.append(math.hypot(ax - held.screen[0], ay - held.screen[1]))
            err_corr.append(math.hypot(bx - held.screen[0], by - held.screen[1]))

        if not err_plain:
            return plain, "homography only (validation failed)"
        rms_plain = float(np.sqrt(np.mean(np.square(err_plain))))
        rms_corr = float(np.sqrt(np.mean(np.square(err_corr))))
        note = (prefix + f"leave-one-out: homography {rms_plain:.0f}px, "
                f"with correction {rms_corr:.0f}px")
        if rms_corr < rms_plain * 0.9:
            full = cls.fit(samples, screen_size, True, k1, frame_size)
            return (full or plain), note + " -> correction kept"
        return plain, note + " -> correction discarded, it does not generalise"

    @classmethod
    def _loo_rms(cls, samples: list[Sample], screen_size: tuple[int, int],
                 k1: float, frame_size: tuple[int, int]) -> float:
        """RMS error over points the fit never saw, in screen pixels."""
        errs = []
        for i in range(len(samples)):
            rest, held = samples[:i] + samples[i + 1:], samples[i]
            fit = cls.fit(rest, screen_size, False, k1, frame_size)
            if fit is None:
                return float("inf")
            px, py = fit.to_screen(*held.camera, corrected=False)
            errs.append(math.hypot(px - held.screen[0], py - held.screen[1]))
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("inf")

    @classmethod
    def _reject_outliers(cls, samples: list[Sample], screen_size: tuple[int, int],
                         factor: float = 3.0) -> "tuple[list[Sample], int]":
        """Drop shots that disagree with the consensus of the others.

        A calibration point is only as good as the trigger pull behind it, and
        one flinched shot biases every hit on that side of the screen. Measured
        live: of nine points, six landed within 2.4mm and three were 12 to 33mm
        out, which dragged the whole fit to 13mm RMS.

        The cut is relative to the MEDIAN residual, not the mean, because the
        mean is itself moved by the outliers it is supposed to find. Points are
        only ever dropped while at least six remain -- below that each one is
        carrying real information about the geometry, and discarding a good
        point is worse than keeping a mediocre one.
        """
        if len(samples) < 7:
            return list(samples), 0
        base = cls.fit(samples, screen_size, use_residuals=False)
        if base is None:
            return list(samples), 0
        errs = base.per_sample_errors(corrected=False)
        med = float(np.median(errs))
        limit = max(med * factor, 8.0)
        keep = [s for s, e in zip(samples, errs) if e <= limit]
        if len(keep) < 6 or len(keep) == len(samples):
            return list(samples), 0
        return keep, len(samples) - len(keep)

    @classmethod
    def fit(cls, samples: list[Sample], screen_size: tuple[int, int],
            use_residuals: bool = True, k1: float = 0.0,
            frame_size: tuple[int, int] = (1280, 720)) -> "Mapping | None":
        """Least-squares homography over every sample, then residual correction.

        Four samples is the minimum and gives no redundancy at all; with more,
        findHomography averages out the aiming scatter in the shots themselves.
        """
        if len(samples) < 4:
            return None
        probe = cls(homography=np.eye(3), k1=k1, frame_size=frame_size)
        src = np.array([probe._undistort(*s.camera) for s in samples],
                       dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([s.screen for s in samples], dtype=np.float32).reshape(-1, 1, 2)
        # Least squares over every point, never RANSAC. RANSAC's threshold is in
        # SCREEN pixels, and the 25px default is 0.65% of a 4K width -- tighter
        # than the lens distortion itself. Measured on a real calibration it
        # declared 4 of 9 good shots outliers and fitted the other 5, which took
        # RMS from 5.2mm to 8.5mm and held-out error from 9.7mm to 19.7mm.
        # Gross outliers are handled by `_reject_outliers`, which judges against
        # the median residual and cannot quietly discard most of the data.
        H, _mask = cv2.findHomography(src, dst, 0)
        if H is None:
            return None

        m = cls(homography=H, samples=list(samples), screen_size=screen_size,
                k1=k1, frame_size=frame_size)
        errs = []
        residuals: list[Point] = []
        for s in samples:
            px, py = m.to_screen(s.camera[0], s.camera[1], corrected=False)
            residuals.append((s.screen[0] - px, s.screen[1] - py))
            errs.append(math.hypot(s.screen[0] - px, s.screen[1] - py))
        m.rms_before = float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0
        m.max_error = float(max(errs)) if errs else 0.0

        if use_residuals:
            m.residuals = residuals
            after = []
            for s in samples:
                px, py = m.to_screen(s.camera[0], s.camera[1], corrected=True)
                after.append(math.hypot(s.screen[0] - px, s.screen[1] - py))
            m.rms_after = float(np.sqrt(np.mean(np.square(after)))) if after else 0.0
        else:
            m.rms_after = m.rms_before
        return m

    def to_camera(self, sx: float, sy: float) -> "Point | None":
        """Where a screen point should image in the camera. The inverse of
        `to_screen`, and therefore a PREDICTION that can be tested.

        That is what makes a fast launch possible: the app can draw a
        marker at a known screen position, photograph it, and compare
        where it landed with where this says it should. Two of those
        answer "has the camera moved?" in about four seconds, against the
        twenty-four an eighteen-marker remeasure costs -- and they answer
        it with a measurement rather than by assuming the worst.
        """
        ok, inv = cv2.invert(self.homography)
        if not ok:
            return None
        pt = np.array([[[float(sx), float(sy)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, inv)
        # `inv` lands in UNdistorted camera space; a real frame is
        # distorted, so bend it back through the lens model -- the same
        # step `screen_rect_in_camera` takes, and for the same reason.
        return self._redistort(float(out[0][0][0]), float(out[0][0][1]))

    def screen_rect_in_camera(self, margin_px: float = 0.0
                              ) -> tuple[Point, Point, Point, Point] | None:
        """The screen's outline in camera coordinates, from the measured mapping.

        Better than the flash-detected outline for masking: it comes from where
        shots actually landed, so it cannot be thrown off by glare, by the screen
        running past the edge of frame, or by a contour that caught only part of
        the monitor. A mask that is too small silently rejects real hits, which
        is indistinguishable from a detector that does not work.
        """
        w, h = self.screen_size
        if not w or not h:
            return None
        ok, inv = cv2.invert(self.homography)
        if not ok:
            return None
        m = margin_px
        corners = np.array([[[-m, -m]], [[w + m, -m]], [[w + m, h + m]], [[-m, h + m]]],
                           dtype=np.float32)
        out = cv2.perspectiveTransform(corners, inv)
        # inv lands in UNdistorted camera space; the detector works on real
        # frames, so bend it back through the lens model.
        return tuple(self._redistort(float(p[0][0]), float(p[0][1])) for p in out)  # type: ignore

    # -- reporting ------------------------------------------------------------

    def per_sample_errors(self, corrected: bool = False) -> list[float]:
        out = []
        for s in self.samples:
            px, py = self.to_screen(s.camera[0], s.camera[1], corrected=corrected)
            out.append(math.hypot(s.screen[0] - px, s.screen[1] - py))
        return out

    def describe(self, mm_per_px: float | None = None) -> str:
        def fmt(v: float) -> str:
            if mm_per_px:
                return f"{v:.0f}px ({v * mm_per_px:.1f}mm)"
            return f"{v:.0f}px"

        return (f"{len(self.samples)} points, "
                f"RMS {fmt(self.rms_before)} -> {fmt(self.rms_after)} after correction, "
                f"worst {fmt(self.max_error)}")

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain Python types only, or the save fails SILENTLY.

        2026-10-16: every value here can arrive as a numpy scalar -- the
        samples come from `find_marker`, which returns numpy floats, and
        `json.dumps` cannot encode those. `save()` caught the TypeError and
        returned False, and its one caller ignored the result, so the app
        logged "saved. The mapping is verified on points it never saw."
        with nothing on disk.

        Found because the cold-start harness never had a saved mapping to
        check: six launches in a row paid for eighteen markers, the
        three-second fast path was never exercised, and a launch took 81
        seconds instead of 40 -- in the harness whose job is to measure
        exactly that. The suite was green throughout.

        `float()` and `int()` on the way out, so a numpy scalar can never
        reach the encoder, and `save()`'s result is now checked by its
        caller (RULES C1: a step that can quietly do nothing is worse than
        one that fails).
        """
        return {
            "homography": self.homography.tolist(),
            "sensitivity": [
                {"camera": [float(c) for c in (r.get("camera") or (0.0, 0.0))],
                 "peak": float(r.get("peak", 0.0))}
                if isinstance(r, dict) else r
                for r in (self.sensitivity or [])],
            "dot_radius": float(self.dot_radius or 0.0),
            "play_area": (self.play_area.to_dict()
                          if self.play_area is not None else None),
            "screen_size": [int(v) for v in self.screen_size],
            "samples": [{"screen": [float(v) for v in s.screen],
                         "camera": [float(v) for v in s.camera],
                         "peak": float(s.peak)}
                        for s in self.samples],
            "residuals": [[float(v) for v in r] for r in self.residuals],
            "k1": float(self.k1),
            "frame_size": [int(v) for v in self.frame_size],
            "rms_before": float(self.rms_before),
            "rms_after": float(self.rms_after),
            "max_error": float(self.max_error),
            # The held-out check, when one was run. A fit's error against its
            # own samples says only that it can reproduce them; this says
            # whether it is right anywhere else, and it belongs with the
            # mapping rather than in a log window that scrolls away.
            "held_out": getattr(self, "held_out", None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Mapping":
        out = cls(
            homography=np.array(d["homography"], dtype=np.float64),
            samples=[Sample(tuple(s["screen"]), tuple(s["camera"]),
                            float(s.get("peak", 0.0))) for s in d["samples"]],
            residuals=[tuple(r) for r in d.get("residuals", [])],
            screen_size=tuple(d.get("screen_size", (0, 0))),
            k1=d.get("k1", 0.0),
            frame_size=tuple(d.get("frame_size", (1280, 720))),
            rms_before=d.get("rms_before", 0.0),
            rms_after=d.get("rms_after", 0.0),
            max_error=d.get("max_error", 0.0),
            sensitivity=list(d.get("sensitivity", []) or []),
            dot_radius=float(d.get("dot_radius", 0.0) or 0.0),
            # DELIBERATELY NOT RESTORED. The border is a per-SESSION
            # measurement: 2026-10-16 18:45:57 saved a play_area of the left
            # third of the screen, measured from phantom detections, and the
            # next launch would have loaded it and fenced him into a third of
            # his monitor. `to_dict` still writes it (evidence); the loader
            # ignores it, which makes a stale value inert without editing a
            # file the app owns. (`held_out` is the opposite case: written
            # and never read back was the bug there.)
            play_area=None,
        )
        # AND THE HELD-OUT REPORT, which was written and never read back.
        #
        # It is the only error figure worth quoting (RULES C9) -- a fit's
        # error against its own samples says it can reproduce them, the
        # held-out error says whether it is right anywhere else. It was
        # being saved faithfully and dropped on load, so `--trust-cal`
        # reported "held-out -1.0mm" for a mapping measured at 0.58mm.
        held = d.get("held_out")
        if isinstance(held, dict):
            out.held_out = held
        return out

    def save(self, path: Path) -> bool:
        """Atomically -- this file holds the mapping AND the learned sensitivity.

        A bare write_text truncates the real file before writing a byte, and
        this app exits through os._exit by design and has been force-killed
        many times. A truncated calibration.json reads back as "no
        calibration", so the loss is silent and costs 18 photographs plus
        every sensitivity region his real shots have taught it.
        """
        return write_json_atomic(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "Mapping | None":
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


def grid_points(screen_w: int, screen_h: int, cols: int = 3, rows: int = 3,
                margin: float = 0.06) -> list[Point]:
    """Evenly spaced aim points, inset from the edges.

    Inset because a target at the very edge is awkward to shoot and sits exactly
    where an off-axis camera has the least resolution -- the two failure modes
    would compound and we would learn nothing useful from the point.

    But only slightly inset. Anything outside the measured region is
    EXTRAPOLATION, and a projective fit extrapolates badly; at a 12% margin the
    screen corners lay well outside the grid and that is precisely where holes
    were reported as worst. 6% keeps the corners nearly covered while still
    giving a ring that can be hit comfortably.
    """
    pts: list[Point] = []
    for r in range(rows):
        for c in range(cols):
            fx = margin + (1 - 2 * margin) * (c / max(1, cols - 1))
            fy = margin + (1 - 2 * margin) * (r / max(1, rows - 1))
            pts.append((screen_w * fx, screen_h * fy))
    return pts


def sensitivity_map(samples: list[Sample], frame_size: tuple[int, int],
                    floor: float = 0.25, power: float = 2.0,
                    region: "tuple | None" = None) -> "np.ndarray | None":
    """How strongly the laser scores at each point of the camera frame.

    A single global threshold cannot serve a screen seen from an angle. Measured
    on this setup, the same gun scored a mean peak of 62 near the centre and 34
    at the far corner, over a blob 6x smaller -- 381 camera pixels against 64.
    The far side of the screen is further away AND viewed at a steeper angle, so
    the dot is both dimmer and spread over fewer pixels. Shots in the far corner
    then fail to clear a threshold that is comfortable everywhere else, and the
    symptom is "it misses most of the shots in the upper right".

    The calibration grid already fires at known points across the whole screen,
    so it measures this falloff for free. Normalising the detector's score by it
    makes sensitivity uniform, which is the thing that was actually wanted all
    along -- a threshold should mean "brighter than noise", not "brighter than
    noise, if you are standing in the right place".

    Returns a float32 image of relative sensitivity, 1.0 where the laser scores
    strongest. `floor` caps how much a weak region may be amplified: without it
    a corner the camera barely sees would have its noise multiplied until it
    looked like signal.
    """
    usable = [s for s in samples if s.peak > 0]
    if len(usable) < 4:
        return None
    w, h = frame_size
    if w <= 0 or h <= 0:
        return None
    best = max(s.peak for s in usable)
    if best <= 0:
        return None

    # Built small and scaled up: the falloff is a smooth property of geometry,
    # so interpolating at full resolution would cost far more and say the same.
    gw, gh = 64, max(1, int(round(64 * h / float(w))))
    xs = (np.arange(gw, dtype=np.float32) + 0.5) * (w / float(gw))
    ys = (np.arange(gh, dtype=np.float32) + 0.5) * (h / float(gh))
    gx, gy = np.meshgrid(xs, ys)

    num = np.zeros((gh, gw), np.float32)
    den = np.zeros((gh, gw), np.float32)
    for s in usable:
        d2 = (gx - s.camera[0]) ** 2 + (gy - s.camera[1]) ** 2
        wgt = 1.0 / np.maximum(d2 ** (power / 2.0), 1e-3)
        num += wgt * (s.peak / best)
        den += wgt
    grid = np.where(den > 0, num / np.maximum(den, 1e-9), 1.0).astype(np.float32)
    grid = np.clip(grid, floor, 1.0)
    full = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    full = cv2.GaussianBlur(full, (0, 0), max(w, h) / 64.0).astype(np.float32)

    # Amplify ONLY where there is something to amplify. The map divides the
    # score by relative sensitivity, so a value of 0.25 multiplies by four --
    # and applied to the whole camera frame that four lands on the room as
    # well as the screen.
    #
    # Measured live 2026-10-14 21:33: a map built from five regions inside the
    # screen (camera x 280-900 of a 1280-wide frame) was applied everywhere,
    # including the blown-out window these notes record on the right of frame.
    # The noise floor went 19 -> 112 and min_peak to 127, which rejects every
    # real hit. A 1.5x correction cost 5.9x of noise.
    #
    # Outside the screen there is BY DEFINITION no laser hit to recover, so
    # the gain is pinned at 1.0 -- no change at all. Inside, the interpolation
    # still has the floor to bound it.
    if region:
        inside = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(inside, np.int32([region]).reshape(-1, 2), 255)
        # A little slack: the screen edge is not where the shots stop, and a
        # hard boundary would make the score jump across it.
        inside = cv2.GaussianBlur(inside, (0, 0), max(w, h) / 128.0)
        keep = (inside.astype(np.float32) / 255.0)[..., None].squeeze(-1)
        full = full * keep + 1.0 * (1.0 - keep)
    return full.astype(np.float32)


def confined_sensitivity_map(samples: list[Sample], frame_size: tuple[int, int],
                             region: "tuple | None") -> "np.ndarray | None":
    """`sensitivity_map`, but it REFUSES to run without the screen's outline.

    `sensitivity_map(region=None)` quietly means "the whole camera frame",
    and that is the 2026-10-14 21:33 incident: a map applied to the room as
    well as the panel took the noise floor 19 -> 112 and min_peak to 127,
    which rejects every real hit (a 1.5x correction cost 5.9x of noise).
    Measured on that map: 1.11x in the middle of the screen, 1.00x at the
    window beside the monitor when confined, 1.44x there when not.

    The learner confined it. The shot grid and the remembered-map restore
    each called `sensitivity_map` directly and did not -- three call sites,
    one rule, held by one of them. Every path that builds the detector's gain
    map from laser peaks now goes through here, so "no outline" is a None the
    caller must answer for, never a silent fall back to the whole frame.
    Detection staying uniform is how every working session ran before any of
    this existed, so None is the safe answer.
    """
    if not region:
        return None
    return sensitivity_map(samples, frame_size, region=region)


# -- is a held-out check good enough to keep the mapping? ---------------------
#
# 2026-10-18 10:04 on his bench, and it cost him the whole session. Pattern
# calibration produced twelve fit markers at RMS 3px (0.8mm), worst 5px
# (1.3mm). Of six held-out markers, FIVE measured 0.1-2.1mm and ONE, at screen
# (845,1685), landed 804px = 196.8mm out. The app printed
#
#     CALIBRATION REJECTED: held-out error 1mm (worst 197mm)
#
# and threw the whole mapping away -- a mapping that was right. A known-good
# calibration measured minutes earlier puts that same screen point at camera
# (456,517); this run "found" its marker at (600,594). It was a FALSE FIND: the
# sensor was saturated (every marker reported strength 255, and the log says
# "lit frame peaks at 255, unlit at 255"), so the finder had no contrast left to
# discriminate on and locked onto something else.
#
# He then fired 42 shots with no mapping. Every single one was detected, at
# peaks 76-109 against a bar of 8 -- margins of 9-13x -- and not one left a
# hole.
#
# THE DEFECT: the decision was driven by the WORST held-out error, which is the
# one statistic a single false find owns outright. `Mapping._reject_outliers`
# already judges the FIT against the MEDIAN residual, in its own words because
# "the mean is itself moved by the outliers it is supposed to find". The
# held-out check is the same problem on the other side of the fit and was using
# the worse statistic.
#
# THE CONSTANTS, each labelled, because a number without its provenance gets
# re-argued and then re-broken:
#
#   GOOD_MM   = 10.0  FITTED. A good run on this bench measures 0.7-1.4mm
#                     held-out median (four sessions, CLAUDE.md "MEASURED
#                     CONSTANTS TO KEEP"). The collapsed fits this check exists
#                     to catch measured 298mm, 386mm and 414mm MEDIAN. Nothing
#                     measured has ever landed between 1.4 and 298, so the bar
#                     is placed where the old worst-case bar already sat rather
#                     than invented afresh.
#   USABLE_MM = 40.0  FITTED, unchanged, and the old comment's reasoning holds:
#                     far above any plausible lens or pose error, far below a
#                     hole displacement a person can see. "Holes nobody draws
#                     are a diagnosis; holes a foot from the shot are a
#                     mystery."
#   WILD_MM   = 40.0  FITTED, the same number for the same reason. A held-out
#                     marker this far out is not lens error; it is a different
#                     point in the room.
#   the allowance     MEASURED from the failure: ONE of six. Two false finds out
#                     of six would mean the marker finder is unreliable across
#                     the panel, which is a different fault and must still be
#                     rejected -- so the allowance is one marker, or a fifth of
#                     them, whichever is more.
HELD_OUT_GOOD_MM = 10.0
HELD_OUT_USABLE_MM = 40.0
HELD_OUT_WILD_MM = 40.0
# When no mm/px is known, 0.24mm/px is this panel (42.5" 4K). The old code used
# 0.25 in the same place; kept identical so this change moves one statistic and
# not two.
HELD_OUT_FALLBACK_MM_PER_PX = 0.25


@dataclass
class HeldOut:
    """The held-out check, judged. Pure: errors in, decision out."""

    median_px: float
    worst_px: float
    worst_screen: Point
    median_mm: float
    worst_mm: float
    # Markers more than HELD_OUT_WILD_MM out, worst first: (px, mm, screen).
    wild: list[tuple[float, float, Point]]
    allowed_wild: int
    points: int
    usable: bool          # keep this mapping at all
    good: bool            # keep it AND call it good
    why: str              # why it is not usable; "" when it is


def judge_held_out(errors, mm_per_px: "float | None" = None) -> "HeldOut | None":
    """Decide a mapping's fate from its held-out errors.

    `errors` is an iterable of `(error_px, screen_point)`. Returns None when
    there is nothing to judge -- ABSENT EVIDENCE, which the caller must report
    as "unverified" rather than as either a pass or a failure (RULES C1 cuts
    both ways).
    """
    pairs = [(float(d), (float(s[0]), float(s[1]))) for d, s in errors]
    if not pairs:
        return None
    mm = float(mm_per_px) if mm_per_px else HELD_OUT_FALLBACK_MM_PER_PX
    ds = sorted(d for d, _s in pairs)
    median_px = ds[len(ds) // 2]
    worst_px, worst_screen = max(pairs, key=lambda p: p[0])
    wild = sorted(((d, d * mm, s) for d, s in pairs if d * mm >= HELD_OUT_WILD_MM),
                  key=lambda t: -t[0])
    allowed = max(1, len(pairs) // 5)
    median_mm = median_px * mm
    too_many = len(wild) > allowed
    usable = (median_mm < HELD_OUT_USABLE_MM) and not too_many
    good = (median_mm < HELD_OUT_GOOD_MM) and not too_many
    if too_many:
        why = (f"{len(wild)} of {len(pairs)} held-out markers are more than "
               f"{HELD_OUT_WILD_MM:.0f}mm out. One can be a false find; "
               f"{len(wild)} means the marker finder is not reliable across "
               f"this panel, which no fit can repair")
    elif median_mm >= HELD_OUT_USABLE_MM:
        why = (f"the MEDIAN held-out error is {median_mm:.0f}mm, so the "
               f"mapping is wrong everywhere and not merely at one marker")
    else:
        why = ""
    return HeldOut(median_px=median_px, worst_px=worst_px,
                   worst_screen=worst_screen, median_mm=median_mm,
                   worst_mm=worst_px * mm, wild=wild, allowed_wild=allowed,
                   points=len(pairs), usable=usable, good=good, why=why)
