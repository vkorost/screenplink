"""A frame sequence: an animated figure, drawn and hit-tested frame by frame.

DELIBERATELY NOT IN `targets/`. That folder carries one exact rule -- every
image in it is a scoreable figure -- and `targets/fall/` was kept separate
rather than making that rule an exception with a suffix. A sequence is a
folder with a manifest under `animations/`, for the same reason.

WHAT THE MANIFEST CARRIES, per frame: the file, the crop it was tightened to
within the source video frame (x, y, w, h), the centre of the largest
connected component (cx) and the feet line (feet). The crop is why a 10s clip
is 6MB instead of 52MB; the feet line is the anchor, because it is stable
within 5px across the whole clip while the figure's height swings 17% as he
crouches and rises.

MEMORY. 120 frames of grey at full crop is about 27MB if every one is held.
Greys and silhouettes are therefore loaded on demand behind a bounded LRU --
the figure cache in `figure.py` is recorded in CLAUDE.md as 4.1MiB per image
and never evicted, which is a trap worth not repeating 120 times over.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from . import assets, ceiling, load


def animations_dir() -> Path:
    """`PROJECT_ROOT/animations`, overridable with LSA_ANIMATIONS.

    `targets/` and `menu/` have had an override since the day a test needed
    one; `animations/` arrived on 2026-10-18 and did not get one, and that
    omission cost five mutation anchors. The mutation scripts run each suite
    in a COPIED tree holding only `src`, `tests` and `tools`, so a path
    resolved from the source location points at a folder that is not there:
    `test_ui_smoke` and `test_feedback` died with FileNotFoundError before
    printing a single result, and a suite that reports no failure is read by
    the runner as a fault NOT CAUGHT. Five anchors reported MISSED while the
    checks behind them were fine.

    Read at CALL time, not captured at import, so setting it cannot be
    defeated by the order in which modules happen to load.
    """
    return assets.asset_dir("animations")      # the table, and why, are in assets.py


# The source video's frame size. The manifest's coordinates are in this space,
# so the whole clip shares one geometry and the figure's drift across it is
# preserved exactly.
SRC_W, SRC_H = 1280, 720

GREY_CACHE = 24           # frames of grey held at once (~5MB at this crop)
SIL_SCALE = 8             # the silhouette is built at 1/8, as figure.py does

# THE KILL ZONES (2026-10-20): "center mass and the head" -- the standard pairing in
# the sport and in training. Derived PER FRAME from the figure's own silhouette, never
# from a fixed screen rectangle and never tuned per frame: all fractions of the
# figure's own height `Hf` (top of the silhouette to its bottom), so they follow him
# as he crouches and rises (the clip's height swings 17%).
#
#   head         the top band, rows 0 .. HEAD_ROWS[1] of Hf, and only the columns within
#                HEAD_HALF_W of Hf of that band's own centre: a cap is narrower than a
#                pair of shoulders
#   centre mass  the upper torso, rows MASS_ROWS of Hf, within MASS_HALF_W of Hf of the
#                torso's own centre (the median column of the silhouette in those rows)
#
# Both are INTERSECTED with the frame's silhouette (a shot through a gap is not a hit)
# and the head band ends before the mass band begins, so the two never overlap: a hit
# counts once, in whichever region contains it.
HEAD_ROWS = (0.0, 0.17)
HEAD_HALF_W = 0.07
MASS_ROWS = (0.19, 0.52)
MASS_HALF_W = 0.13


def zones_from_silhouette(sil: np.ndarray):
    """(head mask, centre-mass mask), both the silhouette's own shape, derived only
    from the silhouette. Pure, so a synthetic figure can be tested exactly."""
    rows, cols = np.nonzero(sil)
    head = np.zeros_like(sil)
    mass = np.zeros_like(sil)
    if not len(rows):
        return head, mass
    r0, r1 = int(rows.min()), int(rows.max())
    hf = float(r1 - r0 + 1)

    def band(lo: float, hi: float, half_w: float):
        a = r0 + int(round(lo * hf))
        b = r0 + int(round(hi * hf))
        sub = sil[a:b]
        cs = np.nonzero(sub)[1]
        if not len(cs):
            return None
        xc = float(np.median(cs))
        c_lo, c_hi = int(round(xc - half_w * hf)), int(round(xc + half_w * hf))
        return a, b, max(0, c_lo), min(sil.shape[1], c_hi + 1)

    for out, spec in ((head, (HEAD_ROWS, HEAD_HALF_W)), (mass, (MASS_ROWS, MASS_HALF_W))):
        got = band(spec[0][0], spec[0][1], spec[1])
        if got is not None:
            a, b, c_lo, c_hi = got
            out[a:b, c_lo:c_hi] = sil[a:b, c_lo:c_hi]
    return head, mass


class Sequence:
    """One animated figure, loaded from `animations/<name>/`."""

    def __init__(self, name: str, folder: "Path | None" = None) -> None:
        self.name = name
        self.dir = (folder or animations_dir()) / name
        self.frames = json.loads((self.dir / "manifest.json").read_text())
        if not self.frames:
            raise ValueError(f"{self.dir} has no frames")
        self._grey: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._sil: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._zones: "OrderedDict[int, tuple]" = OrderedDict()
        self._baked_p99 = 0.0
        self._ceiling = 0.0
        self._gain = self._check_ceiling()

    # -- the brightness ceiling ---------------------------------------------

    def _check_ceiling(self) -> float:
        """The frames were baked to a ceiling. Is it still THE ceiling?

        The bake is what keeps this cheap -- the pixels arrive at the right
        level and nothing is rescaled per frame. But a baked value goes stale
        SILENTLY the moment he adds a darker figure, because the ceiling is
        the minimum over his artwork and it moved 71 -> 52 -> 71 in a single
        morning on 2026-10-18. So it is measured here and corrected, never
        assumed.
        """
        want = ceiling.figure_level()
        step = max(1, len(self.frames) // 12)
        got = max(ceiling.lit_p99(self.grey(i))
                  for i in range(0, len(self.frames), step))
        self._baked_p99, self._ceiling = got, want
        if got <= 0:
            return 1.0
        return min(1.0, want / got)

    @property
    def ink_gain(self) -> float:
        """Brought DOWN to the ceiling only, never up: a stroke brighter than
        the darkest of his images hides the dot that lands on it (RULES B10)."""
        return self._gain

    # -- frames --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def meta(self, i: int) -> dict:
        return self.frames[i % len(self.frames)]

    def grey(self, i: int) -> np.ndarray:
        i %= len(self.frames)
        got = self._grey.get(i)
        if got is None:
            got = load.load_grey(self.dir / self.frames[i]["file"])
            got.setflags(write=False)
            self._grey[i] = got
            while len(self._grey) > GREY_CACHE:
                self._grey.popitem(last=False)
        else:
            self._grey.move_to_end(i)
        return got

    def silhouette(self, i: int) -> np.ndarray:
        """Where the figure IS in this frame: all its ink, thickened, closed
        and filled, at 1/8 scale. A shot inside it is a shot AT the figure.

        The same construction as `modes.figure._silhouette`, because what
        hides a bullet hole and what counts as a hit must be ONE shape --
        and here it changes every frame, so it is cached per frame.
        """
        i %= len(self.frames)
        got = self._sil.get(i)
        if got is None:
            g = self.grey(i)
            lvl = max(20, int(0.55 * load.ink_level(g)))
            ink = cv2.dilate((g > lvl).astype(np.uint8), np.ones((8, 8), np.uint8))
            ink = ink[::SIL_SCALE, ::SIL_SCALE]
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
            cs, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            got = np.zeros_like(ink)
            cv2.drawContours(got, cs, -1, 1, -1)
            got.setflags(write=False)
            self._sil[i] = got
            while len(self._sil) > GREY_CACHE:
                self._sil.popitem(last=False)
        else:
            self._sil.move_to_end(i)
        return got

    def zones(self, i: int):
        """(head, centre mass) masks for frame i, cached with the silhouette."""
        i %= len(self.frames)
        got = self._zones.get(i)
        if got is None:
            got = zones_from_silhouette(self.silhouette(i))
            self._zones[i] = got
            while len(self._zones) > GREY_CACHE:
                self._zones.popitem(last=False)
        else:
            self._zones.move_to_end(i)
        return got

    def zone_at(self, i: int, x: float, y: float,
                origin: "tuple[float, float]", scale: float) -> "str | None":
        """"head", "mass" or None for a screen point, in frame i. Only the frame the
        caller names is consulted: the shot's own, never whichever is up now."""
        left, top, w, h = self.box(i, origin, scale)
        if not (left <= x < left + w and top <= y < top + h):
            return None
        head, mass = self.zones(i)
        sh, sw = head.shape
        r = min(sh - 1, int((y - top) / h * sh))
        c = min(sw - 1, int((x - left) / w * sw))
        if head[r, c]:
            return "head"
        if mass[r, c]:
            return "mass"
        return None

    # -- geometry ------------------------------------------------------------

    def box(self, i: int, origin: "tuple[float, float]", scale: float):
        """Where this frame's crop lands on screen, in pixels.

        `origin` is where the SOURCE frame's top-left sits; `scale` converts
        source pixels to screen pixels. Working in source-frame space is what
        preserves the figure's drift and crouch exactly as filmed.
        """
        m = self.meta(i)
        return (origin[0] + m["x"] * scale, origin[1] + m["y"] * scale,
                m["w"] * scale, m["h"] * scale)

    def on_figure(self, i: int, x: float, y: float,
                  origin: "tuple[float, float]", scale: float) -> bool:
        """Is (x, y) on the figure in frame i? Silhouette, not bounding box."""
        left, top, w, h = self.box(i, origin, scale)
        if not (left <= x < left + w and top <= y < top + h):
            return False
        sil = self.silhouette(i)
        sh, sw = sil.shape
        return bool(sil[min(sh - 1, int((y - top) / h * sh)),
                        min(sw - 1, int((x - left) / w * sw))])

    def to_local(self, i: int, x: float, y: float,
                 origin: "tuple[float, float]", scale: float):
        """Screen point -> this frame's own 0..1 coordinates.

        A mark is stored like this so it RIDES the figure: the crop moves and
        resizes every frame, and a mark pinned to the screen would slide off
        the body within a few frames.
        """
        left, top, w, h = self.box(i, origin, scale)
        return ((x - left) / w, (y - top) / h)

    def to_screen(self, i: int, nx: float, ny: float,
                  origin: "tuple[float, float]", scale: float):
        left, top, w, h = self.box(i, origin, scale)
        return (left + nx * w, top + ny * h)
