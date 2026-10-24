"""Two-player geometry: one screen rectangle in, every rectangle versus needs out.

PURE. QtCore rectangles only: no painter, no app, no mode. Four consumers ask
this module a question -- the two playfields, the divider, the scoreline and the
shot routing -- and they must all get the same answer. Two definitions of one
rectangle is the fault already recorded against the MENU button (the rectangle
a drill avoids ended up somewhere other than the rectangle he sees).

THE AXIS IS LEFT/RIGHT, and that is the point. `px()`, `top_band_height()` and
the bottom strip are all functions of HEIGHT alone, so a half-width world has the
same type sizes, the same reserved top band and the same bottom strip as the
whole screen. A top/bottom split would have halved the height and shrunk every
piece of text on a screen read from several metres. `tests/test_modes.py` pins
that equivalence so a later change to either function cannot quietly break it.

THE DIVIDER takes a few pixels from the middle; each side gets exactly half of
what is left, so the two playfields are equal by construction (the divider's
width is nudged by one pixel when the parity would make them differ).

ROUTING: a shot with x < the divider's centre belongs to the LEFT side; x equal
to the centre or beyond belongs to the RIGHT. One comparison, no dead zone. A
shot that lands on the divider strip itself is therefore routed to whichever
half its centre line is on, and misses there, because the strip is inside
neither half's rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect

LEFT = 0
RIGHT = 1

# Fractions of the screen, never pixels (RULES B5).
DIVIDER_W_FRAC = 0.0012        # of the screen width: ~5px at 4K
DIVIDER_MIN_PX = 2
SCORELINE_H_FRAC = 0.050       # of the screen height
SCORELINE_W_FRAC = 0.30        # of the screen width, centred on the divider
# THE MAGAZINE BAND: the row directly under the scoreline where each half draws
# its bullets (`modes/magazine.py`). His words: "In two-player modes we will have
# to show bullets for each player. Probably at the top of the screen." Placing it
# there is the orchestrator's call, not a measurement. It is part of the LAYOUT,
# not something a painter finds room for, because targets are placed on the
# ground below the scoreline (`below_scoreline`): a strip drawn over ground the
# drills still use would sit under a target. The height is the bullet's
# (`BULLET_H_FRAC`, a fraction of the screen height, so `px()`-style) plus a
# margin above and below; it never shrinks, so a drill's ground does not depend
# on the magazine's size.
BULLET_H_FRAC = 0.060          # of the screen height
MAG_MARGIN_FRAC = 0.008        # of the screen height, above and below the bullets
MAG_BAND_H_FRAC = BULLET_H_FRAC + 2 * MAG_MARGIN_FRAC


@dataclass(frozen=True)
class Layout:
    full: QRect
    left: QRect                # the left side's rectangle, absolute screen coords
    right: QRect
    divider: QRect             # spans the playfield height only
    scoreline: QRect           # top centre, immediately BELOW the reserved band
    left_cell: QRect           # the half of the scoreline over the left side
    right_cell: QRect
    left_note: QRect           # the rest of the scoreline's row, over the left half
    right_note: QRect          # ... and over the right half: where a side says it is done
    mag_band: QRect            # full width, directly under the scoreline: the bullets
    centre_x: float            # the routing boundary

    def side(self, which: int) -> QRect:
        return self.left if which == LEFT else self.right

    def route(self, x: float) -> int:
        """Which side a shot at screen x belongs to. One comparison."""
        return LEFT if x < self.centre_x else RIGHT

    def below_scoreline(self, rect: QRect) -> QRect:
        """`rect` with its top pushed under the scoreline, so no target is
        placed where the scoreline is drawn over it."""
        top = self.mag_band.bottom() + 1      # under the scoreline AND the bullets
        if rect.top() >= top:
            return QRect(rect)
        return QRect(rect.left(), top, rect.width(), max(0, rect.bottom() + 1 - top))

    def playable_for(self, which: int, playable: "QRect | None") -> QRect:
        """The camera-coverage rectangle for one side: `playable` INTERSECTED
        with the side (so the coverage rule is narrowed, never dropped), and
        with the ground under the scoreline removed. When nothing is known
        about coverage the side itself is the answer."""
        side = self.below_scoreline(self.side(which))
        if playable is None:
            return side
        return playable.intersected(side)


def layout(full: QRect) -> Layout:
    """Every versus rectangle, from the full screen rectangle."""
    # Imported here: `display.layers` reaches into `modes.base` lazily, and this
    # module is imported by it.
    from ..display.layers import band_height_for, bottom_strip_for

    x0, y0, w, h = full.x(), full.y(), full.width(), full.height()
    dw = max(DIVIDER_MIN_PX, round(w * DIVIDER_W_FRAC))
    if (w - dw) % 2:
        dw += 1                       # equal halves, exactly
    half = (w - dw) // 2
    left = QRect(x0, y0, half, h)
    divider_x = x0 + half
    right = QRect(divider_x + dw, y0, half, h)

    band = band_height_for(h)
    strip = bottom_strip_for(h)
    top = y0 + band
    bottom = y0 + h - strip
    divider = QRect(divider_x, top, dw, max(0, bottom - top))

    sh = max(1, round(h * SCORELINE_H_FRAC))
    sw = round(w * SCORELINE_W_FRAC)
    if (sw - dw) % 2:
        sw += 1
    sx = divider_x + dw // 2 - sw // 2
    scoreline = QRect(sx, top, sw, sh)
    cell_w = (sw - dw) // 2
    left_cell = QRect(sx, top, cell_w, sh)
    right_cell = QRect(sx + cell_w + dw, top, cell_w, sh)
    # The row beside the scoreline is already kept clear of targets (see
    # `below_scoreline`, which pushes a whole half's playfield under it), so a
    # side's own message goes there and covers nothing.
    left_note = QRect(left.left(), top, max(0, sx - left.left()), sh)
    right_note = QRect(sx + sw, top, max(0, right.right() + 1 - (sx + sw)), sh)

    mag_band = QRect(x0, scoreline.bottom() + 1, w, max(1, round(h * MAG_BAND_H_FRAC)))

    return Layout(full=QRect(full), left=left, right=right, divider=divider,
                  scoreline=scoreline, left_cell=left_cell, right_cell=right_cell,
                  left_note=left_note, right_note=right_note, mag_band=mag_band,
                  centre_x=divider_x + dw / 2.0)
