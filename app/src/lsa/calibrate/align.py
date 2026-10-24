"""Keep the camera-to-screen mapping honest without asking anyone to shoot.

The app knows exactly what it painted. The camera sees exactly what the screen
shows. So the mapping between them is not something that has to be re-measured
by hand -- it can be recovered by registering the two images against each other,
continuously, while the game is running.

That is what this module does. Given the current mapping, it warps a camera
frame into screen space; if the mapping is right the warped frame lines up with
what was painted, and if it has drifted the residual misalignment is exactly the
correction needed. `cv2.findTransformECC` finds that residual directly, by
maximising correlation over a homography, and the result composes onto the
existing mapping.

What this can and cannot do, stated plainly because the difference matters:

  - It CORRECTS slow drift and modest knocks -- the camera settling on its mount,
    a cable pulling, a nudge of a few degrees. This is where it earns its keep,
    because such a shift is invisible to the user until holes start landing
    wrong, and by then the session is spoiled.
  - It DETECTS gross movement, by reporting a correlation that falls off a
    cliff. It cannot fix that case and does not pretend to: a camera that has
    been picked up and put somewhere else needs the shot grid again, and saying
    so is more useful than silently fitting nonsense.
  - It needs the screen to HAVE structure. On a near-black screen there is
    nothing to correlate, so it reports low confidence and changes nothing.
    Refusing to act on weak evidence is the whole point; a mapping that is
    quietly wrong is worse than one that is openly stale.

Every update is gated: the correlation must be decent, it must not get worse,
and the implied movement must be small. A registration that wants to move the
screen corners a long way has not found a better alignment, it has found a
different local optimum, and accepting it would destroy a good calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import cv2
import numpy as np


@dataclass
class AlignResult:
    """What one registration attempt concluded."""

    homography: np.ndarray | None   # camera -> screen, corrected; None if rejected
    correlation: float              # ECC score, 0..1; how well the images agree
    shift_px: float                 # how far the screen corners moved, camera px
    accepted: bool
    reason: str

    def __str__(self) -> str:
        verb = "applied" if self.accepted else "rejected"
        return (f"auto-align {verb}: correlation {self.correlation:.3f}, "
                f"shift {self.shift_px:.1f}px ({self.reason})")


class AutoAligner:
    """Registers a camera frame against the image the app painted.

    Works at a deliberately small resolution. Registration is about large-scale
    geometry, not detail; downscaling costs nothing in accuracy for a plane,
    removes sensor noise for free, and keeps a pass inside a few milliseconds so
    it can run during a game without stealing frames from the detector.
    """

    def __init__(self, work_width: int = 320, min_correlation: float = 0.25,
                 max_shift_px: float = 60.0, iterations: int = 60,
                 eps: float = 1e-5, damping: float = 0.5) -> None:
        self.work_width = work_width
        self.min_correlation = min_correlation
        self.max_shift_px = max_shift_px
        self.iterations = iterations
        self.eps = eps
        # Apply only part of each correction. This is a closed loop -- the next
        # pass sees the result of this one -- and a loop with unity gain chases
        # its own measurement noise. Measured without damping, repeated passes
        # ran 15.0 -> 4.0 -> 2.8 -> 1.8 -> 3.9 px: converging, then wandering
        # back out. Half gain still removes a real drift within a few passes
        # while a noisy estimate is mostly ignored.
        self.damping = float(damping)
        self.last: AlignResult | None = None

    # -- preparation ----------------------------------------------------------

    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _prepare(self, img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Grayscale, resize, and normalise local contrast.

        The camera sees the screen through a short exposure set for hunting a
        laser, so its version is far darker and lower contrast than the source
        image. Correlating raw intensities would mostly measure that difference
        in brightness. Equalising first makes the comparison about structure,
        which is the only thing the two images genuinely share.
        """
        g = self._gray(img)
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        g = cv2.equalizeHist(g)
        return g.astype(np.float32) / 255.0

    # -- the work -------------------------------------------------------------

    def refine(self, camera_bgr: np.ndarray, painted: np.ndarray,
               homography: np.ndarray, screen_size: tuple[int, int]) -> AlignResult:
        """Improve `homography` so the camera frame matches what was painted.

        `painted` is the image the app drew, at any resolution; `homography`
        maps camera pixels to screen pixels.
        """
        sw, sh = screen_size
        if not sw or not sh or camera_bgr is None or painted is None:
            return self._reject(0.0, 0.0, "nothing to align")

        ww = int(self.work_width)
        wh = max(1, int(round(ww * sh / float(sw))))
        scale = np.array([[ww / float(sw), 0, 0],
                          [0, wh / float(sh), 0],
                          [0, 0, 1]], dtype=np.float64)

        # Both images in the same small screen-space frame.
        ref = self._prepare(painted, (ww, wh))
        warped = cv2.warpPerspective(camera_bgr, scale @ homography, (ww, wh),
                                     flags=cv2.INTER_LINEAR)
        if not np.any(warped):
            return self._reject(0.0, 0.0, "camera frame does not cover the screen")
        mov = self._prepare(warped, (ww, wh))

        before = self._correlation(ref, mov)
        warp = np.eye(3, dtype=np.float32)
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.iterations, self.eps)
            corr, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_HOMOGRAPHY,
                                              crit, None, 1)
        except cv2.error:
            # ECC raises rather than returning when it cannot converge, which is
            # a normal outcome on a dark or featureless screen, not a fault.
            return self._reject(before, 0.0, "did not converge")

        if not np.all(np.isfinite(warp)):
            return self._reject(before, 0.0, "produced a non-finite warp")

        # ECC returns a warp that maps TEMPLATE coordinates to INPUT ones -- here
        # ref (where the pixel should be) to mov (where the current mapping put
        # it). The correction is therefore its inverse: take what the current
        # mapping produced and send it back to where it belonged.
        #
        #   mov = warp . scale . H_now . camera      (what we observed)
        #   ref = scale . H_fixed . camera           (what we want)
        #   =>  H_fixed = scale^-1 . warp^-1 . scale . H_now
        try:
            warp_inv = np.linalg.inv(np.asarray(warp, dtype=np.float64))
        except np.linalg.LinAlgError:
            return self._reject(before, 0.0, "warp was not invertible")
        full = np.linalg.inv(scale) @ warp_inv @ scale
        corrected = full @ np.asarray(homography, dtype=np.float64)
        if corrected[2, 2] == 0 or not np.all(np.isfinite(corrected)):
            return self._reject(before, 0.0, "composed to a degenerate mapping")
        corrected = corrected / corrected[2, 2]

        corrected = self._damped(homography, corrected, screen_size)
        if corrected is None:
            return self._reject(corr, 0.0, "damped mapping was degenerate")
        shift = self._corner_shift(homography, corrected, screen_size)
        if not math.isfinite(shift):
            return self._reject(corr, 0.0, "corner shift is not finite")
        if shift > self.max_shift_px:
            return self._reject(corr, shift,
                                f"wants to move the screen {shift:.0f}px - too far "
                                f"to be a drift correction; recalibrate instead")
        if corr < self.min_correlation:
            return self._reject(corr, shift, "too little structure on screen to be sure")
        if corr <= before + 1e-4:
            return self._reject(corr, shift, "no better than the current mapping")

        self.last = AlignResult(corrected, float(corr), float(shift), True,
                                f"improved correlation {before:.3f} -> {corr:.3f}")
        return self.last

    # -- helpers --------------------------------------------------------------

    def _reject(self, corr: float, shift: float, reason: str) -> AlignResult:
        self.last = AlignResult(None, float(corr), float(shift), False, reason)
        return self.last

    def _damped(self, old: np.ndarray, new: np.ndarray,
                screen_size: tuple[int, int]) -> "np.ndarray | None":
        """Move only part of the way from `old` to `new`.

        Blends the two mappings through the screen corners rather than through
        their matrix entries. Averaging homography coefficients has no geometric
        meaning -- they are a projective quantity defined up to scale -- whereas
        a point halfway between two corner positions is exactly what it sounds
        like, and the homography through those blended corners is well defined.
        """
        if self.damping >= 1.0:
            return new
        w, h = screen_size
        rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        corners = rect.reshape(-1, 1, 2)
        ok_a, inv_a = cv2.invert(np.asarray(old, dtype=np.float64))
        ok_b, inv_b = cv2.invert(np.asarray(new, dtype=np.float64))
        if not ok_a or not ok_b:
            return None
        pa = cv2.perspectiveTransform(corners, inv_a).reshape(-1, 2)
        pb = cv2.perspectiveTransform(corners, inv_b).reshape(-1, 2)
        blended = (pa + (pb - pa) * self.damping).astype(np.float32)
        try:
            return cv2.getPerspectiveTransform(blended, rect).astype(np.float64)
        except cv2.error:
            return None

    @staticmethod
    def _correlation(a: np.ndarray, b: np.ndarray) -> float:
        av, bv = a.ravel() - a.mean(), b.ravel() - b.mean()
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        return float(av.dot(bv) / denom) if denom > 0 else 0.0

    @staticmethod
    def _corner_shift(old: np.ndarray, new: np.ndarray,
                      screen_size: tuple[int, int]) -> float:
        """How far the screen's corners move, in CAMERA pixels.

        Expressed in camera pixels on purpose: that is the space the detector
        works in, so it is the space in which "a small correction" has a
        meaning a person can check against the preview.
        """
        w, h = screen_size
        corners = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
        ok_a, inv_a = cv2.invert(np.asarray(old, dtype=np.float64))
        ok_b, inv_b = cv2.invert(np.asarray(new, dtype=np.float64))
        if not ok_a or not ok_b:
            return float("inf")
        pa = cv2.perspectiveTransform(corners, inv_a).reshape(-1, 2)
        pb = cv2.perspectiveTransform(corners, inv_b).reshape(-1, 2)
        return float(np.max(np.linalg.norm(pa - pb, axis=1)))
