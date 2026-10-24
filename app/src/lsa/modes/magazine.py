"""The magazine strip: bullets on the screen, one per round still in the gun.

HIS REQUEST (2026-10-20): "I need 17 of those bullets to be displayed at the
lower corner of the screen (opposite of the corner where the menu button is) for
each round, then as user shoots, the number of bullets decreases ... I want to
build the user's awareness of the mag size." And, clarifying: "this is not just
the game. This is training and people should understand the exact size of their
mag and the exact number of bullets they have left in that mag."

So this strip is the answer to "how many have I got left": he should know the
magazine is spent before he presses the trigger, not learn it from the app's
reaction afterwards. (A pull on an empty magazine then draws its hole and plays
the bubble pop instead of the gunshot -- his rule of 2026-10-20; the strip is what
says the gun is dry BEFORE the pull, the pop is what says so after it.)

PLACEMENT.
  * Single player: the LOWER corner on the opposite SIDE from the MENU button,
    derived from `Mode.MENU_CORNER` (it is replaced by the app from measured
    sensitivity, so moving the button moves the magazine). "Lower" is kept
    whichever corner the button is in: his words name a lower corner, and a
    button in a bottom corner would otherwise send the strip to the top.
  * Two players: one strip per half, centred in its half, in the band directly
    under the scoreline row (`split.Layout.mag_band`), pushed sideways off the
    MENU keep-out when that is the half the button sits in. Reserved in the
    layout so no drill places a target under it. "Probably at the top" was his;
    the exact band is the orchestrator's call, not a measurement.

WHICH END IT EMPTIES FROM. The bullets nearest the anchor are slot 0 and the
remaining ones stay in their slots; the spent ones disappear from the far end.
Nothing that is left ever moves, so a glance finds the same bullets in the same
place. The count is a numeral at the fixed outer end beside slot 0, because
counting twelve outlines from several metres is slower than reading "12".

COST. Grey capped at luminance 60 by `artwork.ui` (spends none of the 2% lit-area
budget, RULES B1b), red excess 0 (B1), sized as fractions of the screen height
and fitted to the field's share of the width (`RenderContext.field_share`) --
`px()` is height-only, so a strip laid out across a half would not shrink by
itself (B5). Repaints only when a shot lands: the host's signature carries the
count, nothing here animates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QRect, QRectF
from PySide6.QtGui import QImage, QPainter, QPixmap

from ..artwork import ui as _ui
from . import split

BULLET_H_FRAC = split.BULLET_H_FRAC
GAP_FRAC = 0.30              # between bullets, of one bullet's width
NUMERAL_FRAC = 1.15          # the numeral's slot width, of one bullet's HEIGHT
EDGE_FRAC = 0.012            # of the screen height: clear of the screen edge
# How much of the field's width the whole strip may take before it is shrunk to
# fit. A corner strip leaves the other half of the bottom for the key legend.
SINGLE_MAX_W = 0.46
HALF_MAX_W = 0.90


@dataclass(frozen=True)
class Geometry:
    """Where everything is, in screen pixels. Pure data, so a test can ask it
    the same question the painter does."""

    bounds: QRect                 # the whole strip, numeral included
    slots: "tuple[QRect, ...]"    # slot i holds the bullet that is i-th from the anchor
    numeral: QRect
    bullet_h: int
    bullet_w: int


def opposite_side(menu_corner: str) -> str:
    """The side of the screen opposite the MENU button: "right" or "left"."""
    return "right" if "left" in menu_corner else "left"


def geometry(field: QRect, screen_h: int, n: int, *, share: float, side: str,
             top: "int | None" = None, bottom: "int | None" = None,
             keepout: "QRectF | None" = None) -> Geometry:
    """The strip for a magazine of `n` in `field`.

    `share` is the field's share of the screen's width (`RenderContext.
    field_share`): the strip is fitted to ITS part of the width. `side` is
    "right" or "left" (anchored at that edge, numeral at the outer end) or
    "centre" (a block centred in the field, numeral at its left). `top` (the
    strip hangs from it) or `bottom` (the strip stands on it) places it.
    """
    n = max(1, int(n))
    bh = max(8, round(screen_h * BULLET_H_FRAC))
    aspect = _ui.aspect()
    bw = bh * aspect
    pitch = bw * (1.0 + GAP_FRAC)
    num_w = bh * NUMERAL_FRAC
    total = n * pitch + num_w
    full_w = field.width() / share if share > 0 else field.width()
    cap = full_w * share * (SINGLE_MAX_W if side != "centre" else HALF_MAX_W)
    if total > cap:                                  # shrink to fit, never overflow
        k = cap / total
        bh = max(8, int(bh * k))
        bw, pitch, num_w = bw * k, pitch * k, num_w * k
        total = n * pitch + num_w
    bw_i = max(1, int(round(bh * aspect)))
    edge = round(screen_h * EDGE_FRAC)
    if bottom is not None:
        y = int(bottom - edge - bh)
    else:
        y = int(top if top is not None else field.top())
    total_i = int(round(total))
    if side == "right":
        x0 = field.right() + 1 - edge - total_i
    elif side == "left":
        x0 = field.left() + edge
    else:
        x0 = field.left() + (field.width() - total_i) // 2
    if keepout is not None and QRectF(x0, y, total_i, bh).intersects(keepout):
        if keepout.center().x() <= field.center().x() or side == "centre":
            nx = int(keepout.right()) + 1                    # slide right of it
            x0 = min(max(x0, nx), field.right() + 1 - edge - total_i)
        else:
            nx = int(keepout.left()) - total_i - 1           # slide left of it
            x0 = max(min(x0, nx), field.left() + edge)
        if QRectF(x0, y, total_i, bh).intersects(keepout):
            y = int(keepout.bottom()) + 1 if top is not None else int(keepout.top()) - bh - 1
    slots = []
    num_i = int(round(num_w))
    if side == "right":
        nx = x0 + total_i - num_i
        numeral = QRect(nx, y, num_i, bh)
        for i in range(n):
            slots.append(QRect(int(round(nx - bw * (1.0 + GAP_FRAC) - i * pitch)), y, bw_i, bh))
    else:
        numeral = QRect(x0, y, num_i, bh)
        left = x0 + num_i + bw * GAP_FRAC
        for i in range(n):
            slots.append(QRect(int(round(left + i * pitch)), y, bw_i, bh))
    return Geometry(QRect(x0, y, total_i, bh), tuple(slots), numeral, bh, bw_i)


# -- drawing -------------------------------------------------------------------

_PIX: "dict[int, QPixmap]" = {}


def bullet_pixmap(height: int) -> QPixmap:
    """One premultiplied (v, v, v, v) pixmap per height, built once: white at
    coverage v/255, so R=G=B and the ground shows through (red excess 0)."""
    got = _PIX.get(height)
    if got is None:
        v = np.ascontiguousarray(_ui.bullet_for(height))
        h, w = v.shape
        bgra = np.empty((h, w, 4), np.uint8)
        bgra[..., 0] = bgra[..., 1] = bgra[..., 2] = bgra[..., 3] = v
        img = QImage(bgra.data, w, h, w * 4, QImage.Format_ARGB32_Premultiplied).copy()
        if len(_PIX) > 8:
            _PIX.clear()
        got = _PIX[height] = QPixmap.fromImage(img)
    return got


def paint(p: QPainter, geo: Geometry, remaining: int, ctx, draw_lines) -> None:
    """Draw `remaining` bullets (slots 0..remaining-1) and the numeral.
    `draw_lines` is `ModeHost._draw_lines`, which fits text to its box from the
    font's own metrics (B5), so no new measuring code lives here."""
    from ..display.layers import GREY_MID, px

    remaining = max(0, min(int(remaining), len(geo.slots)))
    pix = bullet_pixmap(geo.bullet_h)
    for i in range(remaining):
        p.drawPixmap(geo.slots[i].topLeft(), pix)
    p.setPen(GREY_MID)
    draw_lines(p, geo.numeral, [str(remaining)], px(ctx, 44), bold=True)
