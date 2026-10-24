"""Find the monitor in the camera image, and map between the two.

Two jobs, from one measurement:

  1. A region of interest. A hit is by definition a laser dot on the screen, so
     anything outside the monitor is noise. Measured on this setup, a bright
     window at the edge of frame produced 1040 of 1356 detections -- 77% of
     everything. No amount of threshold tuning fixes that; geometry does.

  2. A homography, camera pixels -> screen pixels. That is what turns "a shot
     happened" into "a bullet hole exactly there", and it is exact for a flat
     screen viewed from any angle, which is why an off-axis camera is fine.

Method: photograph the screen white, photograph it black, subtract. Whatever
changed is the monitor. Then fit a quadrilateral to it -- not a bounding box,
because an off-axis view makes the screen a trapezoid and a box around it
includes a wedge of the room.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass
class ScreenQuad:
    """The monitor as seen by the camera, corners ordered TL, TR, BR, BL."""

    corners: tuple[Point, Point, Point, Point]
    coverage: float                 # fraction of the camera frame the screen fills
    confidence: float               # 0..1, how quadrilateral the found shape is
    touches_edges: tuple[str, ...]  # frame edges the screen runs off, if any
    frame_size: tuple[int, int]

    @property
    def polygon(self) -> tuple[Point, ...]:
        return tuple(self.corners)

    def homography_to_screen(self, screen_w: int, screen_h: int) -> np.ndarray:
        """Camera pixels -> screen pixels. Exact for a flat screen at any angle."""
        src = np.array(self.corners, dtype=np.float32)
        dst = np.array([[0, 0], [screen_w, 0], [screen_w, screen_h], [0, screen_h]],
                       dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def map_point(self, matrix: np.ndarray, x: float, y: float) -> Point:
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, matrix)
        return float(out[0][0][0]), float(out[0][0][1])


def _order_corners(pts: np.ndarray) -> tuple[Point, Point, Point, Point]:
    """Order four points TL, TR, BR, BL, without ever repeating one.

    The textbook trick -- smallest coordinate sum is top-left, largest is
    bottom-right, the other two split on x-y -- picks each corner independently,
    so on a strongly sheared quad two of those tests can select the SAME point.
    That happened here: a real calibration produced
    (138,202), (863,579), (863,579), (199,719), a degenerate quad which collapsed
    the homography to map every shot to x=0 and corrupted the screen mask with it.

    Sorting by angle about the centroid cannot do that: it is a permutation, so
    every input point appears exactly once. Then rotate the cycle to start at the
    top-left-most corner and fix the winding.
    """
    pts = pts.reshape(4, 2).astype(np.float64)
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    order = sorted(range(4), key=lambda i: math.atan2(pts[i][1] - cy, pts[i][0] - cx))
    ring = [pts[i] for i in order]

    # Signed shoelace area in image coordinates, where y increases downward. A
    # polygon that reads clockwise on screen gives a POSITIVE sum there (check
    # it on the unit square (0,0),(1,0),(1,1),(0,1): the sum is +2). We want
    # TL -> TR -> BR -> BL, which is clockwise, so reverse when it comes out
    # negative.
    area = 0.0
    for i in range(4):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    if area < 0:
        ring.reverse()

    start = min(range(4), key=lambda i: ring[i][0] + ring[i][1])
    ring = ring[start:] + ring[:start]
    return tuple((float(p[0]), float(p[1])) for p in ring)  # type: ignore[return-value]


# A WHITE FRAME THAT NEVER GOT BRIGHT MEANS THE DISPLAY IS NOT LIT. This is the one
# number that tells a sleeping or switched-off monitor from a camera that cannot see
# a lit one: both give a flash difference near zero and they need OPPOSITE responses.
# ONE definition, read by the app's screen search AND by `tools/camera_health.py`
# (it used to be a literal 90 in the tool alone, which the app could not consult --
# 2026-10-19 the app took a healthy camera for a blind one over a sleeping monitor,
# switched to the lens-capped C920 and shut itself down, three launches in a row).
LIT_WHITE_MAX = 90


def display_is_lit(white: "np.ndarray | None") -> bool:
    """Did the 'white' frame of a flash pair ever get bright? False means the
    display is asleep or off (or the screen search photographed nothing)."""
    return white is not None and int(white.max()) > LIT_WHITE_MAX


def find_screen(white: np.ndarray, black: np.ndarray, threshold: int | None = None,
                min_coverage: float = 0.02, max_coverage: float = 0.85) -> ScreenQuad | None:
    """Locate the monitor from a lit frame and an unlit one.

    Sweeps the threshold rather than trusting one value. A lit monitor floods the
    room, so at a low threshold the "changed" region includes everything the
    screen illuminates -- desk, wall, the operator -- and the result swallows most
    of the frame. Measured: a fixed threshold of 40 returned a quad covering 54%
    of the frame and touching three edges. The screen itself changes far more
    than anything it merely lights up, so a higher cut isolates it.

    Candidates are scored on being a convincing quadrilateral of plausible size
    that does not run off the frame, and the best one wins.
    """
    if white is None or black is None or white.shape != black.shape:
        return None

    diff = cv2.absdiff(white, black)
    if diff.ndim == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    peak = float(np.percentile(diff, 99.5))
    if threshold is not None:
        cuts = [int(threshold)]
    else:
        # Relative to how bright the flash actually was, with absolute floors so
        # a dim flash cannot produce a nonsense cut.
        cuts = sorted({max(25, int(peak * f)) for f in (0.75, 0.6, 0.5, 0.4, 0.3)} | {40})
        cuts.reverse()

    best: ScreenQuad | None = None
    best_score = -1.0
    for cut in cuts:
        quad = _quad_at_threshold(diff, cut, min_coverage, max_coverage)
        if quad is None:
            continue
        # Prefer a genuine 4-corner fit, a sane size, and no clipping at the
        # frame edge; coverage near a half-frame is typical for a usable setup.
        score = quad.confidence
        score += 0.6 if 0.10 <= quad.coverage <= 0.70 else 0.0
        score -= 0.35 * len(quad.touches_edges)
        if score > best_score:
            best_score, best = score, quad
    return best


def _quad_at_threshold(diff: np.ndarray, threshold: int, min_coverage: float,
                       max_coverage: float) -> ScreenQuad | None:
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Close enough to bridge dark UI areas and the mouse pointer on a "white"
    # screen, but not so hard that the screen merges with what it illuminates.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = mask.shape[:2]
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    coverage = area / float(w * h)
    if coverage < min_coverage or coverage > max_coverage:
        return None

    # Try for a true quadrilateral; fall back to the minimum-area rectangle,
    # which is still far better than an axis-aligned box on an oblique view.
    peri = cv2.arcLength(biggest, True)
    approx = None
    for eps in (0.02, 0.03, 0.045, 0.06, 0.08):
        cand = cv2.approxPolyDP(biggest, eps * peri, True)
        if len(cand) == 4:
            approx = cand
            break
    if approx is not None:
        corners = _order_corners(approx)
        confidence = float(min(1.0, area / max(1.0, cv2.contourArea(approx))))
    else:
        box = cv2.boxPoints(cv2.minAreaRect(biggest))
        corners = _order_corners(box)
        confidence = 0.5

    # Reject anything degenerate before it can poison a homography. A quad with
    # two coincident corners is not a rectangle seen at an angle; it is a
    # detection failure, and silently returning it is how every shot ended up
    # mapped to x=0.
    for i in range(4):
        for j in range(i + 1, 4):
            if math.dist(corners[i], corners[j]) < 20.0:
                return None
    quad_area = abs(cv2.contourArea(np.array(corners, dtype=np.float32)))
    if quad_area < 0.5 * area:
        return None          # the fitted quad does not cover the shape it came from

    touching = []
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    if min(xs) <= 2:
        touching.append("left")
    if min(ys) <= 2:
        touching.append("top")
    if max(xs) >= w - 3:
        touching.append("right")
    if max(ys) >= h - 3:
        touching.append("bottom")

    return ScreenQuad(corners=corners, coverage=coverage, confidence=confidence,
                      touches_edges=tuple(touching), frame_size=(w, h))


def draw_quad(frame: np.ndarray, quad: ScreenQuad,
              colour: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Overlay the detected screen outline, for eyeballing the result."""
    out = frame.copy()
    pts = np.array(quad.corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, colour, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(quad.corners):
        cv2.circle(out, (int(x), int(y)), 6, colour, -1, cv2.LINE_AA)
        cv2.putText(out, "TL TR BR BL".split()[i], (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out
