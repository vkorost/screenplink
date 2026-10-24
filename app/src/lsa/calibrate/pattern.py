"""Calibrate by showing the camera a pattern, instead of shooting at rings.

Why this exists
---------------
The shot grid asks the shooter to put three agreeing hits on each of nine
rings. On this setup that is twenty-seven trigger pulls, and the rings the
camera sees worst are exactly the ones that will not register -- so a session
ends with only the left two-thirds of the screen measured. A projective fit
built from two distinct screen x-values is accurate on those two columns and
badly wrong everywhere else, which is what "the holes show up nowhere near the
actual shots" looks like from the firing line.

The shooter's words, after an afternoon of it: "I'm tired of pushing that
trigger needlessly."

They are right, and the laser was never needed for this. The MONITOR is a light
source the camera can already see -- the screen flash measures a difference of
~50 even on the far right, where a laser dot is too small to register. So the
screen shows its own marker at a known position, the camera photographs it, and
that is one correspondence. Repeat over a grid and the mapping is measured
without a single shot.

This is how projector-camera systems are calibrated, and it has three
advantages here that matter:

- **No shooting.** Twelve photographs take a few seconds.
- **It works where the laser does not.** The marker is as large and as bright
  as we care to make it, so the far side of an obliquely-viewed screen is no
  harder than the near side.
- **It covers the whole screen**, so the fit is constrained everywhere rather
  than extrapolating from a couple of columns.

What it does NOT replace: the shot grid remains the ground truth for where a
HIT lands, because it measures the whole chain including the gun. This gets the
mapping right first, so that when shots are fired they land where they should.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:                                    # pragma: no cover
    cv2 = None


def marker_points(width: int, height: int, cols: int = 4, rows: int = 3,
                  margin: float = 0.08) -> list[tuple[float, float]]:
    """Where to put the markers, in screen pixels.

    A wider grid than the shot version (4x3 rather than 3x3) because photographs
    are free where trigger pulls are not, and more spread means a better
    conditioned fit. The margin keeps markers off the very edge, which an
    off-axis camera sees at the steepest angle and a lens distorts most.
    """
    xs = [width * (margin + (1 - 2 * margin) * i / max(1, cols - 1))
          for i in range(cols)]
    ys = [height * (margin + (1 - 2 * margin) * j / max(1, rows - 1))
          for j in range(rows)]
    return [(x, y) for y in ys for x in xs]


def screen_mask(lit: np.ndarray, dark: np.ndarray,
                grow: int = 25) -> "np.ndarray | None":
    """Where the monitor is in the camera image, from a full-screen flash.

    The first step of pattern calibration: light the WHOLE screen, photograph,
    darken it, photograph, subtract. Whatever changed is the monitor.

    This exists because a marker is not the only thing that lights up when the
    screen does. A bright square on a dark panel illuminates the room, and a
    glossy surface beside the monitor reflects it -- on this bench, strongly
    enough that one run found all twelve "markers" clustered at camera x of
    1050-1180 when the monitor only spans 225-920. The reflection is brighter
    and larger than a marker seen at 70 degrees, so no threshold separates
    them. Geometry does: a marker drawn on the screen appears on the screen.

    Grown by `grow` pixels, because the edge of the flash region is softer than
    the edge of the panel and a marker near the border must not be clipped.
    """
    if cv2 is None or lit is None or dark is None or lit.shape != dark.shape:
        return None
    a = cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.clip(a - b, 0, 255).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (9, 9), 0)
    peak = int(diff.max())
    if peak < 12:
        return None
    # Choose the cut by what it produces, rather than fixing a fraction.
    #
    # A fixed 15% of the peak excluded the far side of this panel, which is
    # about four times dimmer than the near side: the mask reached camera x of
    # 694 on a screen extending to 900, so the two right-hand columns of
    # markers were never searched for and the fit fell back to two columns.
    # Lowering the fraction blindly is no better -- in a lit room the screen
    # illuminates the wall, and a cut low enough to catch the far edge also
    # catches the room.
    #
    # So: try cuts from low to high and take the LOWEST whose hull is still a
    # plausible monitor rather than half the frame. In a dark room that is the
    # lowest cut offered, which is what the far edge needs; in a lit one it
    # climbs until the wall is excluded.
    mask = None
    for frac in (0.04, 0.07, 0.12, 0.20, 0.30, 0.45):
        cut = max(5, int(peak * frac))
        _, candidate = cv2.threshold(diff, cut, 255, cv2.THRESH_BINARY)
        if float((candidate > 0).mean()) <= 0.52:
            mask = candidate
            break
    if mask is None:
        # Every cut still covered too much: the screen is lighting the room, so
        # white-minus-black is large everywhere. Take the tightest offered
        # rather than a mask that spans the frame and protects nothing -- one
        # run located the "monitor" at 94% of the frame and then measured a
        # reflection as a marker.
        for frac in (0.60, 0.75, 0.88):
            _, candidate = cv2.threshold(diff, max(6, int(peak * frac)), 255,
                                         cv2.THRESH_BINARY)
            if float((candidate > 0).mean()) <= 0.52:
                mask = candidate
                break
        if mask is None:
            return None

    # The CONVEX HULL of everything substantial, not the largest blob.
    #
    # A screen is convex, and an obliquely-viewed one is several times dimmer
    # at the far edge than the near one. Thresholding therefore splits it: the
    # bright near half becomes one component and the dim far third another,
    # with a gap between them along the scalloped boundary where the panel
    # crosses the cut. Keeping only the biggest component threw the far third
    # away -- the mask ended at camera x of 686 on a panel reaching 920, so the
    # markers over there were never looked for and the fit fell back to two
    # columns. Held-out error 0.4mm when the hull was whole, 8mm when it was not.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    floor_area = diff.size * 0.004          # ignore specks, keep anything real
    pts = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < floor_area:
            continue
        ys, xs = np.nonzero(labels == i)
        pts.append(np.stack([xs, ys], axis=1))
    if not pts:
        return None
    hull = cv2.convexHull(np.concatenate(pts).astype(np.int32))
    keep = np.zeros_like(mask)
    cv2.fillConvexPoly(keep, hull, 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
    return cv2.dilate(keep, k)


def find_marker(lit: np.ndarray, dark: np.ndarray,
                min_area: int = 6,
                mask: "np.ndarray | None" = None,
                near: "tuple[float, float, float] | None" = None
                ) -> tuple[float, float, float] | None:
    """Locate the marker by subtracting an unlit frame from a lit one.

    Returns (x, y, strength) in camera pixels, or None.

    Subtraction rather than plain thresholding, for the same reason the screen
    search uses it: the room contains things brighter than a patch of monitor
    seen at 70 degrees, and the only thing that reliably distinguishes the
    marker is that it CHANGED. Strength is the peak difference, which tells the
    caller how much to trust this point.
    """
    if cv2 is None or lit is None or dark is None:
        return None
    if lit.shape != dark.shape:
        return None

    a = cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.clip(a - b, 0, 255).astype(np.uint8)
    # A light blur: MJPEG leaves blocking artefacts that otherwise fragment the
    # marker into several components.
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    diff[:3, :] = 0
    diff[-3:, :] = 0
    diff[:, :3] = 0
    diff[:, -3:] = 0
    if near is not None:
        # Once a few markers are known the rest are predictable, and a search
        # restricted to a disc around the prediction cannot be fooled by a
        # reflection, by noise, or by a photograph taken a frame too early.
        # This is far stronger than any brightness threshold, and unlike the
        # screen mask it does not need tuning against the room.
        nx, ny, nr = near
        disc = np.zeros(diff.shape, dtype=np.uint8)
        cv2.circle(disc, (int(round(nx)), int(round(ny))), int(max(8, nr)), 255, -1)
        diff = cv2.bitwise_and(diff, diff, mask=disc)
    if mask is not None and mask.shape == diff.shape:
        # Outside the monitor there is nothing a marker could be, however
        # bright. This is what stops a reflection beside the screen from being
        # measured as the marker itself.
        diff = cv2.bitwise_and(diff, diff, mask=mask)

    peak = int(diff.max())
    if peak < 12:
        return None                      # nothing changed; the camera cannot see it

    # Relative to this marker's own peak, because brightness across an
    # obliquely-viewed panel varies by a factor of four and a fixed cut would
    # find the near markers and miss the far ones -- the exact failure the shot
    # grid had.
    _, mask = cv2.threshold(diff, max(8, int(peak * 0.5)), 255, cv2.THRESH_BINARY)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    best, best_area = None, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best, best_area = i, area
    if best is None or best_area < min_area:
        return None

    # Intensity-weighted centroid over that component, for sub-pixel position.
    ys, xs = np.nonzero(labels == best)
    w = diff[ys, xs].astype(np.float32)
    total = float(w.sum())
    if total <= 0:
        return None
    return (float((w * xs).sum() / total),
            float((w * ys).sum() / total),
            float(peak))


def _row_line(row):
    """Robust (slope, intercept) for a row, per axis, or None."""
    sx, sy = [], []
    for i in range(len(row) - 1):
        dc = row[i + 1][0] - row[i][0]
        if dc == 0:
            continue
        sx.append((row[i + 1][1][1][0] - row[i][1][1][0]) / dc)
        sy.append((row[i + 1][1][1][1] - row[i][1][1][1]) / dc)
    if not sx:
        return None
    def mid(v):
        # A true median. Taking sorted[n//2] on an even count is the upper of
        # the two middle values, which after one point has been dropped leaves
        # three markers at uneven index gaps and a slope biased enough to
        # condemn an innocent neighbour.
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0

    mx, my = mid(sx), mid(sy)
    step = (mx * mx + my * my) ** 0.5
    if step < 5:
        return None
    bx = sorted(cam[0] - mx * c for c, (_s, cam) in row)[len(row) // 2]
    by = sorted(cam[1] - my * c for c, (_s, cam) in row)[len(row) // 2]
    return mx, my, bx, by, step


def prune_out_of_order(points: list, cols: int, rows: int,
                       tol: float = 0.40) -> tuple[list, list]:
    """Drop markers whose camera position contradicts the grid's own order.

    `points` is ((screen_x, screen_y), (camera_x, camera_y)) in the order the
    grid was shown, row by row, with None where a marker was not found.
    Returns (kept, dropped), dropped carrying how far out of line each was.

    A grid is a grid: along a row the screen position advances in equal steps,
    so the camera position must advance steadily too, whichever way round the
    camera is mounted. Measured on a failing run, one row gave camera x of 657
    and 1237 for screen x of 1920 and 2995 where the step should have been
    about 190px -- a reflection had been found instead of the marker, and that
    one point took the fit from 0.6mm to 215mm. Least squares cannot help: it
    splits the difference between a marker and a reflection and gets neither.

    Two details, each learned from a version that did the wrong thing:

    - The line comes from the median of CONSECUTIVE steps, not of all pairwise
      slopes. With four markers a single bad one spoils three of the six pairs,
      so that median sits on the boundary -- and the first version duly threw
      away the good point and kept the reflection.
    - Only the single WORST point is dropped, then the line is refitted and the
      test repeated. Dropping everything over tolerance at once took an
      innocent neighbour with each bad marker and left six markers of twelve,
      which is two columns, which is the degenerate fit this exists to avoid.

    Rows only. Columns converge in a perspective view, so their spacing is not
    uniform; a row's is near enough to linear to test.
    """
    kept, dropped = [], []
    for r in range(rows):
        row = [(c, points[r * cols + c]) for c in range(cols)
               if r * cols + c < len(points) and points[r * cols + c] is not None]
        while len(row) >= 3:
            line = _row_line(row)
            if line is None:
                break
            mx, my, bx, by, step = line
            worst_i, worst_off = None, 0.0
            for idx, (c, (_sc, cam)) in enumerate(row):
                ex, ey = bx + mx * c, by + my * c
                off = ((cam[0] - ex) ** 2 + (cam[1] - ey) ** 2) ** 0.5
                if off > worst_off:
                    worst_i, worst_off = idx, off
            if worst_i is None or worst_off <= step * tol:
                break
            sc, cam = row[worst_i][1]
            dropped.append((sc, cam, worst_off))
            row.pop(worst_i)
        kept.extend(pt for _c, pt in row)
    return kept, dropped


def plausible_quad(pairs: list, *, max_side_ratio: float = 3.5,
                   min_side: float = 30.0) -> "tuple[bool, str]":
    """Do four found markers actually look like the corners of a screen?

    `pairs` is a list of ((screen_x, screen_y), (camera_x, camera_y)) for the
    four markers at the corners of the grid, in any order -- they are sorted
    here by their SCREEN coordinates, which are known exactly. Returns
    (ok, why).

    This exists because the four corners are visited first so that every later
    marker's position can be predicted from them, and a prediction is only as
    good as the points behind it. With exactly four correspondences the
    homography fits them perfectly whatever they are, so there is no residual
    to notice -- one corner found on a reflection produces a confident,
    completely wrong prediction, and then every remaining marker falls outside
    the search disc and is reported "not visible to the camera". Observed on
    2026-10-14: four corners found, eight declared invisible.

    What a real screen must satisfy, and a reflection almost never does:

    - The four camera points wind the same way as the four screen points, with
      every turn in the same direction. A convex quadrilateral, in order.
    - No side shorter than `min_side` pixels, so a degenerate sliver is caught.
    - Opposite sides within `max_side_ratio` of each other. Strong perspective
      is expected here -- the far edge of this panel images about 60% the
      length of the near one -- so the bar is deliberately loose. It is there
      to catch a corner in the wrong place, not to judge the camera angle.
    """
    if len(pairs) != 4 or any(p is None for p in pairs):
        return False, "fewer than four corners"
    # Sort by screen position into TL, TR, BR, BL. The screen coordinates are
    # exact, so this ordering is never in doubt.
    by_y = sorted(pairs, key=lambda p: p[0][1])
    top = sorted(by_y[:2], key=lambda p: p[0][0])
    bottom = sorted(by_y[2:], key=lambda p: p[0][0])
    quad = [top[0][1], top[1][1], bottom[1][1], bottom[0][1]]

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    signs = [cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4])
             for i in range(4)]
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return False, "the four corners do not form a convex quadrilateral"

    sides = [((quad[(i + 1) % 4][0] - quad[i][0]) ** 2
              + (quad[(i + 1) % 4][1] - quad[i][1]) ** 2) ** 0.5
             for i in range(4)]
    if min(sides) < min_side:
        return False, f"one side is only {min(sides):.0f}px long"
    for a, b in ((0, 2), (1, 3)):
        lo, hi = sorted((sides[a], sides[b]))
        if hi > lo * max_side_ratio:
            return False, (f"opposite sides differ {hi / lo:.1f}x "
                           f"({lo:.0f}px vs {hi:.0f}px)")
    return True, (f"convex, sides {sides[0]:.0f}/{sides[1]:.0f}/"
                  f"{sides[2]:.0f}/{sides[3]:.0f}px")
