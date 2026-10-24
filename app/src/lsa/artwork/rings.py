"""Reading the printed scoring rings OUT of the artwork, as polygons.

PURE: numpy and cv2 only, no Qt.

The rings are already drawn in each image as closed grey curves. They are
bottle-shaped (a head dome joined to a wide chest), so no rectangle or ellipse
follows them (measured: best fits 0.48-0.86). Rather than guess a shape, this
module takes the ink itself:

  * `snap` moves a rough closed polyline onto the ink it is meant to follow:
    for every sample it looks along the normal for the strongest stroke and a
    dynamic programme keeps the result smooth, so a gap in the printed line is
    bridged and a neighbouring art line does not steal the curve.
  * `register` finds the similarity transform (scale + translation) that lays a
    template of those curves over another image, because the printed nest is
    the same drawing at a different size and place in each file.
  * `measure` reports how far each polygon vertex is from real ink.

Polygons are in NORMALISED coordinates (x: fraction of the image width, y:
fraction of its height), so they scale with the figure.
"""

from __future__ import annotations

import cv2
import numpy as np

INK_LEVEL = 70            # luminance above which a pixel is stroke


def resample_closed(pts: np.ndarray, step: float) -> np.ndarray:
    """Points spaced `step` apart along the closed polyline `pts` (N,2)."""
    p = np.asarray(pts, np.float64)
    q = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(q, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(round(cum[-1] / step)))
    t = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(t, cum, q[:, 0]), np.interp(t, cum, q[:, 1])], axis=1)


def normals(pts: np.ndarray) -> np.ndarray:
    d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    n = np.stack([d[:, 1], -d[:, 0]], axis=1)
    return n / np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-9)[:, None]


def stroke_score(grey: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """Float image: how stroke-like each pixel is (0..1)."""
    return cv2.GaussianBlur(np.clip(grey.astype(np.float32) / 120.0, 0.0, 1.0),
                            (0, 0), sigma)


def _sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(xy[..., 0], 0, w - 1.001)
    y = np.clip(xy[..., 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def snap(prior: np.ndarray, score: np.ndarray, reach: int = 25, step: float = 6.0,
         smooth: float = 0.35, frozen: "np.ndarray | None" = None,
         anchor: float = 0.0):
    """Snap the closed polyline `prior` (N,2 px) onto the strongest nearby
    stroke. Returns (polygon (M,2), frozen flags (M,), offsets (M,)).

    `frozen` is a boolean per PRIOR vertex: those stretches are not snapped
    (a chord across a gap where nothing is printed) and stay where the prior
    put them. `anchor` adds a pull toward the prior (cost per unit of reach),
    so a neighbouring art line must beat the printed ring by more than that to
    steal the curve.
    """
    base = resample_closed(prior, step)
    if frozen is not None:
        # carry the flag to the resampled points by nearest prior vertex
        d = np.hypot(base[:, None, 0] - prior[None, :, 0], base[:, None, 1] - prior[None, :, 1])
        fro = np.asarray(frozen)[np.argmin(d, axis=1)]
    else:
        fro = np.zeros(len(base), bool)
    nrm = normals(base)
    offs = np.arange(-reach, reach + 1, dtype=np.float64)
    cand = base[:, None, :] + nrm[:, None, :] * offs[None, :, None]       # (M,K,2)
    s = _sample(score, cand)                                              # (M,K)
    s = s - 0.0
    cost = -s + anchor * np.abs(offs)[None, :] / float(reach)
    cost[fro, :] = 0.0
    cost[fro, reach] = -1.0                                               # stay put
    M, K = cost.shape
    # unroll twice so the loop closes, run a Viterbi, keep the middle copy
    C = np.vstack([cost, cost, cost])
    pen = smooth * np.abs(offs[:, None] - offs[None, :]) / 3.0
    acc = np.zeros_like(C)
    back = np.zeros(C.shape, np.int32)
    acc[0] = C[0]
    for i in range(1, len(C)):
        tot = acc[i - 1][:, None] + pen
        back[i] = np.argmin(tot, axis=0)
        acc[i] = C[i] + tot[back[i], np.arange(K)]
    k = int(np.argmin(acc[-1]))
    path = np.zeros(len(C), np.int32)
    for i in range(len(C) - 1, -1, -1):
        path[i] = k
        k = back[i, k]
    sel = path[M:2 * M]
    o = offs[sel]
    return base + nrm * o[:, None], fro, o


def ink_distance(grey: np.ndarray, level: int = INK_LEVEL) -> np.ndarray:
    """Distance (px) from every pixel to the nearest stroke pixel."""
    inv = (grey <= level).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def measure(poly_px: np.ndarray, dist: np.ndarray, skip: "np.ndarray | None" = None):
    """(median, p95, max) distance in px from the polygon's vertices to ink,
    ignoring `skip` (vertices that are deliberately not on ink)."""
    d = _sample(dist, poly_px)
    if skip is not None:
        d = d[~skip]
    return float(np.median(d)), float(np.percentile(d, 95)), float(d.max())


def to_norm(poly_px: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.stack([poly_px[:, 0] / (w - 1), poly_px[:, 1] / (h - 1)], axis=1)


def to_px(poly_n: np.ndarray, w: int, h: int) -> np.ndarray:
    p = np.asarray(poly_n, np.float64)
    return np.stack([p[:, 0] * (w - 1), p[:, 1] * (h - 1)], axis=1)


def transform(poly_px: np.ndarray, s: float, tx: float, ty: float,
              cx: float, cy: float) -> np.ndarray:
    """Scale by `s` about (cx, cy), then translate."""
    return np.stack([(poly_px[:, 0] - cx) * s + cx + tx,
                     (poly_px[:, 1] - cy) * s + cy + ty], axis=1)


def register(template: "list[np.ndarray]", dist: np.ndarray, cap: float = 14.0,
             scales=None):
    """Find (s, tx, ty) laying the template polygons (px, in the template's own
    frame) onto the ink whose distance map is `dist`. Coarse scale/translation
    search on a downscaled map, then a fine coordinate descent. Returns
    (s, tx, ty, mean distance in px at the optimum)."""
    h, w = dist.shape
    cx, cy = w / 2.0, h / 2.0
    pts = np.vstack(template)

    def cost(s, tx, ty):
        q = transform(pts, s, tx, ty, cx, cy)
        d = _sample(dist, q)
        d = np.minimum(d, cap)
        inside = (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
        d = np.where(inside, d, cap)
        return float(d.mean())

    scales = scales if scales is not None else np.arange(0.6, 1.61, 0.05)
    best = (1e9, 1.0, 0.0, 0.0)
    for s in scales:
        for ty in range(-400, 401, 50):
            for tx in range(-400, 401, 50):
                c = cost(s, tx, ty)
                if c < best[0]:
                    best = (c, s, tx, ty)
    c, s, tx, ty = best
    for step_s, step_t in ((0.02, 20), (0.01, 8), (0.004, 3), (0.002, 1)):
        improved = True
        n = 0
        while improved and n < 60:
            improved, n = False, n + 1
            for ds, dx, dy in ((step_s, 0, 0), (-step_s, 0, 0), (0, step_t, 0),
                               (0, -step_t, 0), (0, 0, step_t), (0, 0, -step_t)):
                c2 = cost(s + ds, tx + dx, ty + dy)
                if c2 < c - 1e-9:
                    c, s, tx, ty, improved = c2, s + ds, tx + dx, ty + dy, True
    return s, tx, ty, c
