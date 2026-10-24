"""Load a figure image as true grey, and measure how much of the screen it lights.

PURE: cv2 and numpy only, no Qt (the seam to Qt is `lsa.modes.sprite`).

Two facts about the artwork, both measured 2026-10-16/17 and both load-bearing:

  * The files are 3-channel JPEGs that are NOT neutral: red excess (R minus
    the larger of G and B) reaches 6-12 on eleven of them and 26 on
    predator01. The saved sensitivity map multiplies the weakest region of
    the screen up to 4x, so even 8 can reach the detector as 30. So the
    conversion to grey is not optional: `load_grey` returns ONE channel, and
    everything downstream builds R=G=B from that one byte, which makes red
    excess 0 by construction rather than by checking.
  * The lit-area budget (RULES B1b: no more than ~2% of the screen above
    luminance 60 while shooting is possible) is what limits how big a figure
    may be drawn. Lit area scales as height squared and the twelve figures
    differ 3.2x at the same size, so the cap is derived per image.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np

from . import assets                                   # noqa: E402  (the asset-folder table)
PROJECT_ROOT = assets.PROJECT_ROOT

# Memory note: `lsa.modes.figure` keeps every loaded grey array for the life of the
# process (12 x 4.1 MiB = 49.2 MiB today, never evicted). See the comment there.

LIT_LEVEL = 60                 # luminance above which a pixel counts as lit
# The target background's luminance while shooting (RULES B1b measures the lit
# fraction against it). Ink is drawn as white at coverage v/255 over the
# background, so its on-screen luminance is v + BACKGROUND * (1 - v/255), not v:
# a pixel of 55 lands at 64. Measured 2026-10-17: ignoring this made the
# rendered figures light 1.6-1.9% against a 1.5% cap.
BACKGROUND = 10

MIN_HEIGHT_FRAC = 0.30         # a figure smaller than this is not a target
MAX_HEIGHT_FRAC = 0.85         # and bigger than this fills the playfield


def targets_dir() -> Path:
    """`PROJECT_ROOT/targets`, overridable with LSA_TARGETS for tests."""
    return assets.asset_dir("targets")


def load_grey(path: "str | Path") -> np.ndarray:
    """The image as one uint8 channel (HxW). Forced grey, never optional.

    cv2.imread yields BGR; BGR2GRAY collapses it to one channel, so whatever
    colour cast the JPEG carried (predator01: red excess 26) is gone before
    anything can draw it.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read artwork {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def red_excess_of_file(path: "str | Path") -> int:
    """Max of (R - max(G, B)) over the file AS STORED, before conversion.
    Reported by tests to show why the conversion is load-bearing."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR).astype(np.int16)
    return int((bgr[..., 2] - np.maximum(bgr[..., 1], bgr[..., 0])).max())


def ink_level(grey: np.ndarray) -> int:
    """99th-percentile luminance of the image (115-118 on these files)."""
    return int(np.percentile(grey, 99))


def figure_size(grey: np.ndarray, screen_h: int, height_frac: float) -> tuple[int, int]:
    """(width, height) in pixels of the figure drawn at `height_frac` of the
    screen height, keeping its aspect."""
    h = max(1, int(round(screen_h * height_frac)))
    w = max(1, int(round(h * grey.shape[1] / grey.shape[0])))
    return w, h


def scaled(grey: np.ndarray, screen_h: int, height_frac: float) -> np.ndarray:
    """The figure resampled (INTER_AREA) to its on-screen size."""
    w, h = figure_size(grey, screen_h, height_frac)
    return cv2.resize(grey, (w, h), interpolation=cv2.INTER_AREA)


def lit_fraction(grey: np.ndarray, screen_w: int, screen_h: int,
                 height_frac: float, lit: int = LIT_LEVEL,
                 background: int = 0) -> float:
    """Pixels above luminance `lit`, as a fraction of the WHOLE screen, when
    this figure is drawn at `height_frac` of the screen height.

    `background` is the luminance the figure is composited over. 0 reproduces
    the bare-artwork measurement (the coordinator's table); the size caps use
    `BACKGROUND`, because that is what the screen actually shows."""
    small = scaled(grey, screen_h, height_frac)
    if background:
        v = small.astype(np.float32)
        small = v + background * (1.0 - v / 255.0)
    return float((small > lit).sum()) / float(screen_w * screen_h)


def max_height_for_budget(grey: np.ndarray, screen_w: int, screen_h: int,
                          budget: float, background: int = BACKGROUND) -> float:
    """The largest height (fraction of screen height) at which this figure
    lights no more than `budget` of the screen, clamped to
    [MIN_HEIGHT_FRAC, MAX_HEIGHT_FRAC].

    Closed form from ONE measurement at height 1.0, since lit area scales as
    height squared (the measured table agrees to within 3%). Then VERIFIED
    against a direct measurement at the answer and shrunk if it overshoots,
    because INTER_AREA blurs thin lines below the lit level at small sizes
    and the closed form must not be trusted past what was measured.
    """
    full = lit_fraction(grey, screen_w, screen_h, 1.0, background=background)
    if full <= 0.0:
        return MAX_HEIGHT_FRAC
    h = math.sqrt(budget / full)
    h = min(MAX_HEIGHT_FRAC, max(MIN_HEIGHT_FRAC, h))
    for _ in range(6):
        got = lit_fraction(grey, screen_w, screen_h, h, background=background)
        if got <= budget or h <= MIN_HEIGHT_FRAC:
            break
        h = max(MIN_HEIGHT_FRAC, h * math.sqrt(budget / got) * 0.995)
    return h
