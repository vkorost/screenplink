"""Furniture artwork: the cartridge the magazine strip is drawn from.

Furniture (`images/bullet.png`) is NOT `targets/`. `targets/` means "a figure with verified scoring rings",
and everything in it is scoreable and feeds the brightness ceiling. A bullet
counting what is left in a magazine is furniture: nobody aims at it and it must
never move the ceiling the targets are measured against.

Treated exactly as the figures are (RULES B1, B1b, B10):

  * TRUE GREY AT SOURCE. `images/bullet.png` is a one-channel PNG converted from his
    JPEG (JPEG chroma artefacts are precisely the noise this detector hunts, and
    his gain map amplifies the weakest screen region fourfold). Red excess is 0
    by construction, and the loader returns ONE channel.
  * CAPPED AT LOAD, never rewritten on disk. The cap is the lower of the artwork
    ceiling (`ceiling.figure_level`, measured from the darkest figure) and the
    level at which a pixel stops spending the lit-area budget: ink is drawn as
    white at coverage v/255 over the background, so it lands at
    v + BACKGROUND*(1 - v/255); that is exactly LIT_LEVEL (60) at v = 52. Under
    60 the strip spends NOTHING of the 2% budget, which matters because the menu
    already sits at 1.634% of it.
  * STROKES THICKENED BEFORE THE SHRINK. The cartridge is 7px lines on 1460px;
    drawn 65-130px tall, a plain INTER_AREA shrink leaves them under one pixel
    and dim, and a count that cannot be read at a glance is no use. A dilation
    sized from the scale puts every line at about `MIN_STROKE_PX` on the screen.

PURE: cv2 and numpy only, no Qt (the seam is `lsa.modes.magazine`).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from . import assets, ceiling, load

MIN_STROKE_PX = 1.6        # on-screen width every line is brought up to
SRC_STROKE_PX = 6          # the outer outline's width in the source image
# File level at which a drawn pixel is exactly at the lit-area threshold.
LIT_CAP_FILE = int((load.LIT_LEVEL - load.BACKGROUND) / (1.0 - load.BACKGROUND / 255.0))

_RAW: "dict[str, np.ndarray]" = {}


BULLET_NAME = "bullet.png"   # read out of this source by the packaging spec: keep it a plain literal


def bullet_path() -> Path:
    """`resources/images/bullet.png`; the folder is overridable with LSA_IMAGES for tests.
    (A separate ui/ folder and its LSA_UI override were removed 2026-10-23.)"""
    return assets.asset_dir("images") / BULLET_NAME


def load_bullet() -> np.ndarray:
    """The cartridge as one uint8 channel, exactly as stored (cached)."""
    path = bullet_path()
    key = str(path)
    got = _RAW.get(key)
    if got is None:
        got = load.load_grey(path)
        _RAW[key] = got
    return got


def cap_level() -> int:
    """The brightest file level the strip may use: the lower of the measured
    artwork ceiling and the lit-area cap. Derived, so a darker figure added
    later moves it down and the strip follows."""
    return int(min(ceiling.figure_level(), LIT_CAP_FILE))


def aspect() -> float:
    g = load_bullet()
    return g.shape[1] / g.shape[0]


def bullet_for(height_px: int) -> np.ndarray:
    """The cartridge drawn `height_px` tall: strokes thickened, shrunk, then
    scaled so its brightest pixel is exactly `cap_level()`."""
    g = load_bullet()
    h = max(8, int(height_px))
    w = max(1, int(round(h * aspect())))
    s = h / g.shape[0]
    k = int(round(MIN_STROKE_PX / s - SRC_STROKE_PX))
    if k >= 2:
        k += 1 - (k % 2)                       # odd kernel
        g = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    small = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    peak = int(small.max())
    if peak <= 0:
        return small
    out = np.rint(small.astype(np.float32) * (cap_level() / float(peak)))
    return np.clip(out, 0, 255).astype(np.uint8)
