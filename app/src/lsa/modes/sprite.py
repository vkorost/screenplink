"""Drawing a figure without re-rasterising it. The seam between the Qt-free
`lsa.artwork` package and the screen.

A 1792x2400 JPEG must never be scaled by Qt on every repaint: a 4K repaint
costs milliseconds the detector needs (it wants ~7 ms per camera frame and holds
60 fps only under ~14 ms). So:

  * Levels are built LAZILY with `cv2.resize(INTER_AREA)` from the grey source
    and kept in an LRU bounded by `MAX_PYRAMID_BYTES` (grey, 1 byte per pixel).
  * Heights step in a GEOMETRIC series (ratio `STEP`), anchored on the largest
    height the mode will use, so a scaling figure steps imperceptibly (2% of
    1500 px is 30 px per step).
  * Only the CURRENT level exists as a pixmap, rebuilt only when the level
    changes; `paint` is a single 1:1 `drawPixmap`, no smooth transform.
  * The pixmap is built from ONE grey byte: every pixel is premultiplied
    (v, v, v, alpha=v), i.e. white at coverage v/255. R=G=B in every pixel by
    construction, so red excess is 0 whatever the file's colour cast was, and
    the figure composites over the scene background with no black rectangle.

`drawn_size(height_px)` is the size the level that WOULD be drawn actually has.
Scoring uses it, so the geometry that is drawn and the geometry that is scored
come from the same numbers.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPainter, QPixmap

STEP = 1.02                           # geometric ratio between levels
MAX_PYRAMID_BYTES = 40 * 1024 * 1024  # grey levels, 1 byte per pixel


class Sprite:
    def __init__(self, grey: np.ndarray, ink_scale: float = 1.0) -> None:
        if grey.ndim != 2 or grey.dtype != np.uint8:
            raise ValueError("Sprite needs one uint8 grey channel")
        self.grey = grey
        # 1.0 = his artwork exactly as drawn. Below 1.0 dims the ink; it is a
        # knob for step 21's measurement, never turned by default.
        self.ink_scale = float(ink_scale)
        self.aspect = grey.shape[1] / grey.shape[0]
        self._levels: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._anchor = float(grey.shape[0])       # height of level 0, px
        self._pix: "QPixmap | None" = None
        self._pix_key: "int | None" = None
        self.builds = 0                            # level builds, for tests
        self.pixmap_builds = 0

    # -- the series -------------------------------------------------------

    def set_range(self, max_height_px: float) -> None:
        """Anchor the geometric series on the largest height the mode uses. A
        new anchor invalidates the cache (the levels no longer line up)."""
        a = max(8.0, float(max_height_px))
        if abs(a - self._anchor) > 1e-6:
            self._anchor = a
            self._levels.clear()
            self._bytes = 0
            self._pix = None
            self._pix_key = None

    def level_for(self, height_px: float) -> int:
        k = int(round(math.log(self._anchor / max(8.0, height_px))
                      / math.log(STEP)))
        return max(0, k)

    def level_height(self, k: int) -> int:
        return max(8, int(round(self._anchor / (STEP ** k))))

    def drawn_size(self, height_px: float) -> tuple[int, int]:
        """(width, height) in px of what `paint` draws for this request."""
        h = self.level_height(self.level_for(height_px))
        return max(1, int(round(h * self.aspect))), h

    # -- levels -----------------------------------------------------------

    def _level(self, k: int) -> np.ndarray:
        got = self._levels.get(k)
        if got is not None:
            self._levels.move_to_end(k)
            return got
        h = self.level_height(k)
        w = max(1, int(round(h * self.aspect)))
        lvl = cv2.resize(self.grey, (w, h), interpolation=cv2.INTER_AREA)
        self.builds += 1
        self._levels[k] = lvl
        self._bytes += lvl.nbytes
        while self._bytes > MAX_PYRAMID_BYTES and len(self._levels) > 1:
            old_k, old = next(iter(self._levels.items()))
            if old_k == k:
                break
            del self._levels[old_k]
            self._bytes -= old.nbytes
        return lvl

    @property
    def pyramid_bytes(self) -> int:
        return self._bytes

    def _pixmap(self, k: int) -> QPixmap:
        if self._pix is not None and self._pix_key == k:
            return self._pix
        v = np.ascontiguousarray(self._level(k))
        if self.ink_scale != 1.0:
            v = np.clip(v.astype(np.float32) * self.ink_scale, 0, 255).astype(np.uint8)
        h, w = v.shape
        bgra = np.empty((h, w, 4), np.uint8)
        bgra[..., 0] = v
        bgra[..., 1] = v
        bgra[..., 2] = v
        bgra[..., 3] = v                      # premultiplied: colour <= alpha
        img = QImage(bgra.data, w, h, w * 4,
                     QImage.Format_ARGB32_Premultiplied).copy()
        self._pix = QPixmap.fromImage(img)
        self._pix_key = k
        self.pixmap_builds += 1
        return self._pix

    def warm(self, height_px: float) -> None:
        """Build the level and pixmap for this height now, so the first paint
        at it does not pay for them. Called while the screen is blank or the
        figure is being set up, never mid-walk."""
        self._pixmap(self.level_for(height_px))

    # -- drawing ----------------------------------------------------------

    def paint(self, p: QPainter, centre: tuple[float, float],
              height_px: float) -> tuple[int, int]:
        """Blit the figure centred on `centre`. Returns the drawn (w, h)."""
        pm = self._pixmap(self.level_for(height_px))
        w, h = pm.width(), pm.height()
        left, top = self.top_left(centre, (w, h))
        p.drawPixmap(left, top, pm)
        return w, h

    @staticmethod
    def top_left(centre: tuple[float, float], size: tuple[int, int]) -> tuple[int, int]:
        """Where the figure's top-left pixel lands: whole pixels, so the blit
        is 1:1 and scoring reads the same corner that `paint` used."""
        return (int(round(centre[0] - size[0] / 2.0)),
                int(round(centre[1] - size[1] / 2.0)))
