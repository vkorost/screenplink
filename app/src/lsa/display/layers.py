"""The concrete layers that make up the target display.

Phase 0 set: a background whose colour can be switched to stress the detector,
aiming furniture, camera-space hit markers, a toggleable camera pane, and a
stats HUD. Phase 1 adds real bullet holes in screen coordinates once the
homography exists; Phase 2 adds image backgrounds and animated targets, which
is what ImageBackground and the advance() hook are here for.

House rule, restated because it is the easiest mistake to make: no saturated
red anywhere on this screen. The camera is watching it.
"""

from __future__ import annotations

import math
from functools import lru_cache
import time
from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen,
                           QPixmap, QRegion)

from ..artwork import ceiling as _ceiling
from ..artwork import load as _artload
from .scene import Layer, RenderContext

# Palette, chosen under one hard constraint: for every colour here,
# red - max(green, blue) must stay below about 30. That quantity IS the chroma
# detector's score, so a colour with a high value is a colour the camera may
# read as a laser hit. A nice warm amber like #f0b429 scores 60 and is
# therefore banned, however good it looks.
INK_DARK = QColor("#3c3c3c")          # 0
INK_LIGHT = QColor(0, 0, 0, 90)       # 0
# 200,192,112 -> red excess 8. It was #d8c070, red excess 24.
#
# LEGAL UNDER THE STATIC LIMIT OF 30, AND THREE TIMES THE LIMIT OF 8 FOR
# ANYTHING THAT MOVES OR APPEARS -- and this is the accent the signal
# meter writes its live counters and its warning line in, so it does both.
#
# Measured 2026-10-16 from the screenshot the app saved while he was
# shooting: the only pixels anywhere on his 4K screen above red excess 30
# were 504 of them inside the signal meter, reaching 47. His eight
# "phantom shots at a static object" mapped to screen points inside that
# box -- local red excess 24 to 44 at every one of them. The app was
# shooting at its own meter and firing the gunshot sound at him for it.
#
# I had diagnosed those as an object in the room, from their repeating
# camera coordinates, and was about to build a position-identity rule
# that would have rejected his tight aim groups. An outside review found
# the real cause by mapping the detections back onto the screen.
# NOW GREY, red excess 0 (2026-10-17, "no yellow"). It was #c8c070 at 8, and 8
# fires wherever the saved sensitivity map amplifies (up to 4x) -- see
# GREY_HI below. Every colour on this screen that appears or changes is at 0.
ACCENT = QColor("#c8c8c8")
ACCENT_DIM = QColor(200, 200, 200, 120)

# ...AND A MUCH TIGHTER LIMIT FOR ANYTHING THAT MOVES.
#
# The house rule is red excess under 30, and it was written for STATIC
# furniture -- which is safe for a reason that has nothing to do with the
# number: the detector subtracts a running background on every frame, so
# anything that stays put is absorbed within seconds and stops being
# visible to it at all.
#
# A MOVING object defeats that by construction. It is never where the
# background model expects it, so its full red excess reaches the
# threshold on every frame -- which is the precise description of a laser
# dot travelling across the screen.
#
# Measured 2026-10-15, live, during his own string: the Escalating Ball
# target is drawn in ACCENT at red excess 24, and the app logged 272
# detections in two minutes, 231 of them below peak 15, against his real
# shots at peak 85-149. He fired perhaps fifteen times. The app was
# shooting at its own ball, and playing a gunshot for every frame of it.
#
# 8 rather than 30, because the bar a detection must clear is set from a
# measured noise floor of 2-5 and lands around 10. Anything the app draws
# must sit below that floor, not merely below the ceiling.
MOVING_RED_LIMIT = 8

# Red excess 0 (was an amber at 4).
TARGET = QColor(200, 200, 200)
TARGET_DIM = QColor(200, 200, 200, 120)
ACCENT_ON_LIGHT = QColor("#555555")   # grey: 0 (was #5a5210, 8)
COOL = QColor("#5fa8d3")              # negative
GOOD = QColor("#6bd36b")              # negative

# EVERYTHING A MODE OR A TARGET DRAWS IS NEUTRAL GREY: red excess exactly 0.
#
# Owner, 2026-10-16: "sometimes these targets disappear all by themselves
# because the camera thinks there are shots; therefore we need to draw
# everything in shades of grey; no yellow". Three recorded instances of the
# app shooting at its own drawing (console banner 80, arcade ball 24, signal
# meter 24), and the saved sensitivity map amplifies the weakest region
# fourfold, so even the house limit of 8 for a moving object can arrive at
# the detector as 16-18, above his bar. Zero cannot be amplified into
# anything. Brightness carries legibility and meaning (hit brighter than
# miss); colour carries nothing. Shape carries meaning where it matters
# (discriminate: ring = shoot, barred square = leave alone).
#
# Every one is visible against the dark target background (luminance ~10);
# none may be drawn in the background colour.
# THE BRIGHTNESS CEILING (2026-10-17). Grey keeps red excess 0, but a BRIGHT grey
# still hides the dot: the laser makes (255, 10, 10) on black (score ~245) and
# (255, 200, 200) on grey 200 (score 55, against a bar of 18). So nothing a mode
# draws may exceed the luminance of the darkest artwork he supplied, which
# already registers for him: `CEILING`, MEASURED from targets/ at import
# (lsa.artwork.ceiling), never typed in. Every grey below is a fraction of it, so
# the hit flash is the brightest thing on screen WITHOUT being brighter than a
# figure's strokes. `tests/test_ceiling.py` renders every registered mode and
# fails on any pixel above it, so a mode added later is covered without anybody
# remembering to look here.
CEILING = _ceiling.screen_level()


def grey_at(frac: float, alpha: int = 255) -> QColor:
    """A neutral grey `frac` of the way up to the ceiling (never beyond it)."""
    v = max(0, min(CEILING, int(round(CEILING * frac))))
    return QColor(v, v, v, alpha)


GREY_HI = grey_at(0.92)                # targets, outlines
GREY_HI_DIM = grey_at(0.92, 90)        # translucent strokes
GREY_MID = grey_at(0.75)               # secondary text, miss/negative flashes
GREY_HIT = grey_at(1.0)                # a hit: the brightest thing on screen
GREY_MISS = grey_at(0.50)              # a miss / false start / no-shoot
GREY_TEXT = grey_at(0.80)              # mode hints and status text
GREY_FAINT = grey_at(0.62)             # "wait for it" and other almost-nothing

# The key legend along the bottom-left, in pixels at 1080p. Reserved the same
# way as the top band, so the things that live down there cannot end up printed
# over each other on one screen size and not another.
BOTTOM_STRIP = 108


def band_height_for(height: int) -> int:
    """The instruction band's height for a screen of this height."""
    return max(56, int(height * 0.075))


def bottom_strip_for(height: int) -> int:
    """What the key legend and the signal meter occupy along the bottom."""
    return max(BOTTOM_STRIP, round(height * BOTTOM_STRIP / 1080.0))


def top_band_height(ctx: RenderContext) -> int:
    """Height of the instruction band across the top of the screen.

    Reserved unconditionally, by every layer that draws near the top, rather
    than negotiated between layers. A layout that depends on which other layers
    happen to be visible is a layout that breaks the moment someone presses a
    toggle, and the band is the one piece of furniture that should always be
    exactly where the shooter last saw it.
    """
    return band_height_for(ctx.height)


# The key legend along the bottom-left, in fixed pixels because it is drawn in a
# fixed-size font: four lines of 20 plus a 20px margin, rounded up. Reserved the
# same way as the top band, so the two things that live down there cannot end up
# printed over each other on one screen size and not another -- which is exactly
# what happened the first time the signal meter was placed.
def px(ctx: RenderContext, at_1080: int, floor: int = 9) -> int:
    """A size picked on a 1080p window, scaled to the screen it ends up on.

    Everything here was first laid out in a 1080p test window, and read
    perfectly there. On the real panel -- 42 inches of 4K, viewed from several
    metres down the range -- the same numbers are half the apparent size, and a
    shooter cannot act on an instruction they cannot read. Qt will not do this
    for us: point sizes track DPI, and a 4K 42-inch monitor is about 104 DPI,
    the same as the 1080p window it was designed in.
    """
    return max(floor, round(ctx.height * at_1080 / 1080.0))


def mono(size: int, bold: bool = False) -> QFont:
    """Monospace font with a fallback chain.

    Naming a single family and hoping is how you get a screen full of tofu
    boxes on a machine that lacks it. Qt picks the first family that resolves.
    """
    f = QFont()
    f.setFamilies(["Consolas", "Cascadia Mono", "DejaVu Sans Mono",
                   "Liberation Mono", "Courier New", "monospace"])
    f.setStyleHint(QFont.Monospace)
    f.setFixedPitch(True)
    # GREYSCALE GLYPH EDGES ONLY. On a real Windows surface Qt draws text with
    # ClearType, which writes coloured sub-pixel fringes: pure GREY text
    # rendered through the native platform measured red excess 49-86 on its
    # edges (offscreen rendering has no sub-pixel text and shows 0, which is
    # how this stayed invisible). The camera reads those edges as laser hits
    # whenever text appears -- the fake shots when 0 is pressed. This is the
    # only font factory in the display code, so it is decided here, once.
    f.setStyleStrategy(QFont.NoSubpixelAntialias)
    f.setPointSize(size)
    if bold:
        f.setBold(True)
    return f


class SolidBackground(Layer):
    """Flat colour fill, switchable to the cases that stress detection hardest.

    White is the worst case for any detector: the laser has to out-shine a fully
    lit panel. Red is the worst case specifically for the chroma detector, since
    the background then has the same hue as the thing we are looking for. If
    detection survives both, the room and the target art will not trouble it.
    """

    name = "background"
    z = -100

    PRESETS: dict[int, tuple[str, str]] = {
        Qt.Key_D: ("dark", "#0a0a0a"),
        Qt.Key_W: ("WHITE (worst case)", "#ffffff"),
        Qt.Key_R: ("RED (worst case for chroma)", "#b00000"),
        # Grey is on Y, not G: G starts calibration, and a layer key shadows
        # an app key because layers get first refusal on the target screen.
        Qt.Key_Y: ("mid grey", "#808080"),
    }

    def __init__(self, colour: str = "#0a0a0a", label: str = "dark") -> None:
        self.colour = QColor(colour)
        self.label = label

    @property
    def is_dark(self) -> bool:
        return self.colour.lightness() < 128

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        p.fillRect(ctx.rect, self.colour)

    def on_key(self, key: int) -> bool:
        if key in self.PRESETS:
            self.label, colour = self.PRESETS[key]
            self.colour = QColor(colour)
            return True
        return False


class ImageBackground(Layer):
    """A still image or a frame of a sequence, letterboxed to the screen.

    Not used in Phase 0, but present so the scene architecture is actually
    exercised by something image-based rather than being a promise. Animated
    backgrounds subclass this and swap `pixmap` inside advance().
    """

    name = "background"
    z = -100

    def __init__(self, pixmap: QPixmap | None = None, cover: bool = True,
                 dim: float = 0.0) -> None:
        self.pixmap = pixmap
        self.cover = cover      # cover crops to fill; otherwise fit inside
        self.dim = dim          # 0..1 black veil, to keep bright art from blinding the camera

    @property
    def is_dark(self) -> bool:
        return True

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        p.fillRect(ctx.rect, QColor("#000000"))
        if self.pixmap is None or self.pixmap.isNull():
            return
        mode = Qt.KeepAspectRatioByExpanding if self.cover else Qt.KeepAspectRatio
        scaled = self.pixmap.scaled(ctx.rect.size(), mode, Qt.SmoothTransformation)
        x = ctx.rect.x() + (ctx.width - scaled.width()) // 2
        y = ctx.rect.y() + (ctx.height - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        if self.dim > 0:
            p.fillRect(ctx.rect, QColor(0, 0, 0, int(255 * min(1.0, self.dim))))


# -- THE NUMBERS AND THE SHADED BANDS on the static target (2026-10-20) ------------
#
# His request: "On the static target I need you to put the numbers. Also shade the
# alternate circles. Make it look like a real target."
#
# THE NUMBERS are not typed anywhere. `ring_points` is THE arithmetic -- the ring a
# shot lands in decides its points -- and both the scorer (`score_against_marks`)
# and the painter (`AimField.ring_numbers`) call it, so the ring he SEES and the
# ring he is SCORED against are identical by construction (the rule the figure
# artwork was rebuilt around). Change the arithmetic and the drawn numbers change.
#
# THE SHADING IS THE DANGEROUS HALF. A band is lit ground his laser has to land on,
# and a dot landing on a surface at luminance L loses (L - background) units of its
# red-minus-green/blue score: the detector scores `red - max(green, blue)` and the
# dot's red saturates while the band keeps its green and blue (RULES B10). The balls
# live under "fill 20, background 10 -> a landing shot loses TEN units". So:
#
#   * a band is at most `BAND_SHADE_MAX_STEP` units above the target background, a
#     number a later "make it clearer" cannot quietly cross (tests assert it);
#   * it is far below luminance 60, so it spends NOTHING of the 2% lit-area budget;
#   * only the CENTRE mark is shaded. The corner marks are small (bands of 35-42 px
#     at 4K) and, being the corners, they are where his far-side shots land: peaks
#     of 12-15 against a bar of 9, which any shading at all would take under the
#     bar. The centre mark spans the middle of the panel where his shots measure
#     60-100. And the 10 (the aim disc) and the 8 are never shaded: the disc is
#     where he aims, and shading it would tax exactly the shot that matters.
#   * the numbers are drawn on the CENTRE mark only: a corner ring is too narrow to
#     carry a digit he can read from several metres, and a mark he cannot read is
#     worse than none.
# ROLLED BACK TO 0 ON 2026-10-20, after he found it from the firing line:
# "in mode 1 when I shoot on the shaded regions, the shots do not register...
#  obviously it did not work and it created an impression of a non-working app."
#
# He is right and the arithmetic is exact. The detector scores red MINUS the
# stronger of green and blue. On the near-black background his laser images at
# roughly (255,10,10) and scores ~245; on a band lifted 8 units, green and blue
# are 8 higher and the score is 8 LOWER. His weak-side shots arrive at about 12
# against a bar of 8, so a shot on a shaded band scores 4 and vanishes.
#
# The shading is therefore a blind spot that is worst exactly where the signal
# is thinnest -- and the note above this block already said as much about the
# corner marks ("8 units would take his dimmest real shots, peak 12, under the
# bar of 9") while shipping it on the centre one anyway.
#
# It bought appearance ("make it look like a real target") and cost detection,
# on the one drill the app uses to measure itself. The rings and the numbers
# stay: they are thin strokes, so a dot only pays where it lands on the line
# itself, not across a whole band.
#
# A cost-free version is possible -- distinguish the bands by a thin boundary
# stroke rather than a fill -- but it is a drawing change that has to be
# measured against a real laser, not an evening's edit.
BAND_SHADE_STEP = 0            # luminance units above the background
BAND_SHADE_MAX_STEP = 10       # the ceiling on the line above; a check fails past it
# EMPTY: no ring is shaded. His instruction, 2026-10-20 22:58, after shooting
# it: "In mode 1, ring 7 and ring 9 do not work. Shots do not register there.
# We need to restore mode 1 to the way it was before we shaded 7 and 9."
#
# Setting BAND_SHADE_STEP to 0 already removed the brightness, and the drawn
# screen measured red excess 0 everywhere afterwards -- but he still saw those
# two rings fail, so the step alone was not the whole rollback. This takes the
# shading out of the drawing entirely, which is what he asked for and leaves
# nothing to argue about. The rings and the numbers stay.
SHADED_RINGS = ()              # was (1, 3): the 9 and the 7 bands
SHADED_MARKS = (0,)            # the centre mark only
DIGIT_BAND_FRAC = 0.42         # a digit's height, of the width of the band it sits in
MIN_DIGIT_H_FRAC = 0.016       # of the SCREEN height: below this a digit is not drawn
NUMBER_GREY = 0.75             # of the artwork ceiling (53 on screen: under 60, red excess 0)


def ring_points(n_rings: int, ring: int) -> int:
    """Points for a shot that lands in ring `ring` (0 = innermost) of a mark with
    `n_rings` rings. THE scoring arithmetic: the scorer and the painter of the
    numbers both ask this and nothing else."""
    return n_rings - ring + 6


class AimField(Layer):
    """Something to point at, spread deliberately into the corners.

    The corner groups are not decoration. Lens distortion and the error from an
    off-axis homography are both worst at the edges, so a calibration check
    wants targets out there, not just in the middle.

    `marks()` is the single source of truth for where the rings are, used both
    for drawing them and for scoring shots against them. Two copies of that
    geometry would drift apart and the scoring would quietly stop matching what
    the shooter sees.
    """

    name = "aim"
    z = 0
    _nums_cache = None        # (geometry key, ring_numbers result); see ring_numbers

    def marks(self, ctx: RenderContext) -> list[tuple[float, float, list[float]]]:
        """Each aim mark as (x, y, ring radii from innermost outward)."""
        w, h = ctx.width, ctx.height
        x0, y0 = ctx.rect.x(), ctx.rect.y()
        # THE WHOLE MARK SET IS SIZED BY THE FIELD'S SHARE OF THE SCREEN WIDTH
        # (two-player versus, decision 4 of requirements/two_player_versus_plan_01).
        # 1.0 on a whole screen -- no division happens, so single player is
        # bit-for-bit unchanged on EVERY aspect ratio -- and 0.5 for a half.
        # Not a smaller fraction picked to fit: a half-width field keeps the
        # height, so height-only radii arrived at full size, 1814px of a 1920px
        # field, and two sides lit 2.9% of the screen against a 2% ceiling.
        f = ctx.field_share
        out: list[tuple[float, float, list[float]]] = [(
            x0 + w / 2.0, y0 + h / 2.0,
            [h * 0.05 * f, h * 0.13 * f, h * 0.26 * f, h * 0.42 * f],
        )]
        # POSITIONS keep following the field (min(w, h) of the rect it is given).
        # RADII come from the WHOLE screen's smaller side times `f`: taking
        # min(w, h) of a half field and multiplying by `f` as well would count
        # the narrowing twice (0.444 of single player instead of 0.5).
        m = min(w, h) * 0.16
        r = min(ctx.screen_width if ctx.screen_width else w, h) * 0.055 * f
        # A TWO-PLAYER HALF keeps its marks off the magazine band (the bullets
        # under the scoreline row, `modes/split.py`): the layout reserves that
        # ground for every drill that asks it for its playable region, and this
        # field does not ask, so it clears the band itself. Whole screens are
        # untouched (`screen_width` is None there).
        floor_y = None
        if ctx.screen_width is not None:
            from ..modes import split as _split
            floor_y = (y0 + band_height_for(h)
                       + round(h * (_split.SCORELINE_H_FRAC + _split.MAG_BAND_H_FRAC)) + r)
        for px, py in ((m, m), (w - m, m), (m, h - m), (w - m, h - m)):
            cx, cy = x0 + px, y0 + py
            if floor_y is not None and py < h / 2.0:
                cy = max(cy, floor_y)
            cx, cy = self._off_the_keepout(cx, cy, r, ctx)
            out.append((cx, cy, [r * 0.35, r * 0.65, r]))
        return out

    def _off_the_keepout(self, cx: float, cy: float, r: float,
                         ctx: RenderContext) -> tuple[float, float]:
        """Slide a corner mark off the keep-out, along the shorter axis.

        He photographed the bottom-left mark sitting on the MENU button on
        2026-10-18 and asked for both halves of the fix: the button lower,
        and no target placed over it. Lower alone could not have done it --
        at 4K that mark reaches y=1933 and the playfield ends at 1944, so a
        91px button does not fit beneath it at any inset. The mark moves.

        It slides rather than jumps, by exactly the overlap, so the drill
        still reads as "four marks in the corners". The direction is
        whichever axis needs the smaller move, and it stays inside the
        playfield: a mark pushed off the screen would be worse than a mark
        that overlaps a button.
        """
        from ..modes.base import Mode    # local: modes imports this module
        box = Mode.menu_keepout(ctx)
        # Nearest point of the box to the centre; if it is further than the
        # radius, there is no overlap and nothing to do.
        nx = min(max(cx, box.left()), box.right())
        ny = min(max(cy, box.top()), box.bottom())
        dx, dy = cx - nx, cy - ny
        if dx * dx + dy * dy > r * r:
            return cx, cy
        # How far to move on each axis to clear it, in the direction that is
        # already the way out.
        up = box.top() - r - cy if cy < box.center().y() else 0.0
        down = box.bottom() + r - cy if cy >= box.center().y() else 0.0
        left = box.left() - r - cx if cx < box.center().x() else 0.0
        right = box.right() + r - cx if cx >= box.center().x() else 0.0
        move_y = up if up else down
        move_x = left if left else right
        field = Mode.playfield(ctx)
        cand = []
        if abs(move_y) > 0:
            cand.append((abs(move_y), cx, cy + move_y))
        if abs(move_x) > 0:
            cand.append((abs(move_x), cx + move_x, cy))
        for _, nxx, nyy in sorted(cand):
            if (field.left() + r <= nxx <= field.right() - r
                    and field.top() + r <= nyy <= field.bottom() - r):
                return nxx, nyy
        return cx, cy

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if ctx.calibrating:
            return      # one ring to shoot at, not ten
        # Neutral grey (red excess 0) and sized from the screen, never in
        # pixels (RULES B5): these are the targets now, on a 42-inch panel, a
        # 75-inch TV or a projector alike. Brighter than the old furniture
        # grey, because this is the thing he shoots at.
        ink = grey_at(0.92) if ctx.dark_background else QColor(60, 60, 60)
        p.setRenderHint(QPainter.Antialiasing)
        marks = self.marks(ctx)
        # THE SHADED BANDS go down first, under every outline (see BAND_SHADE_STEP).
        if ctx.dark_background:
            shade = _artload.BACKGROUND + BAND_SHADE_STEP
            p.setPen(Qt.NoPen)
            for i in SHADED_MARKS:
                if i >= len(marks):
                    continue
                cx, cy, radii = marks[i]
                for k in SHADED_RINGS:
                    if k < len(radii):
                        inner = radii[k - 1] if k > 0 else 0.0
                        path = QPainterPath()
                        path.setFillRule(Qt.OddEvenFill)
                        path.addEllipse(QPointF(cx, cy), radii[k], radii[k])
                        path.addEllipse(QPointF(cx, cy), inner, inner)
                        p.fillPath(path, QColor(shade, shade, shade))
        p.setPen(QPen(ink, max(2.0, ctx.height * 0.0035)))
        for i, (cx, cy, radii) in enumerate(marks):
            for r in radii:
                p.drawEllipse(QPointF(cx, cy), r, r)
            tick = ctx.height * (0.016 if i == 0 else 0.009) * ctx.field_share
            p.drawLine(int(cx - tick), int(cy), int(cx + tick), int(cy))
            p.drawLine(int(cx), int(cy - tick), int(cx), int(cy + tick))
        # THE NUMBERS, derived from `ring_numbers` (and so from `ring_points`).
        nums = self.ring_numbers(ctx)
        if nums:
            p.setPen(QPen(grey_at(NUMBER_GREY)))
            for text, ncx, ncy, size in nums:
                p.setFont(mono(size, True))
                box = QRectF(ncx - size * 4.0, ncy - size * 2.0, size * 8.0, size * 4.0)
                p.drawText(box, Qt.AlignCenter, text)

    def ring_numbers(self, ctx: RenderContext) -> "list[tuple[str, float, float, int]]":
        """[(text, centre x, centre y, font size)] for every number drawn: the
        centre mark's rings, four of each (above, below, left and right of the
        centre, as a real target prints them), placed in the middle of their band.

        The text is `ring_points` of that ring, so a shot at the number's centre is
        scored exactly that. Digits are sized from the BAND (a fraction of the
        screen height times the field's share of the width, so a two-player half
        shrinks them with it) and a ring whose digit would fall under
        `MIN_DIGIT_H_FRAC` of the screen height is NOT drawn -- and neither is a
        digit that would touch the crosshair or leave its band. The aim point itself
        (the exact centre) is never inked.
        """
        marks = self.marks(ctx)
        if not marks or not ctx.dark_background or ctx.calibrating:
            return []
        # CACHED, because this is the most expensive thing the screen does and it
        # is a pure function of the geometry. Uncached it ran ~1000 font
        # constructions per paint (the size search below) and measured 171 ms a
        # paint under the Windows font engine (17 ms under the offscreen one,
        # which is why no suite saw it); the HUD repaints this screen five times
        # a second, so that was 0.85 s of the Qt thread per second and the camera
        # thread, starved of the interpreter lock, read at 2 fps (2026-10-20).
        key = (ctx.height, ctx.field_share, tuple(marks[0][2]), marks[0][0], marks[0][1])
        hit = self._nums_cache
        if hit is not None and hit[0] == key:
            return hit[1]
        out = self._ring_numbers(ctx, marks)
        self._nums_cache = (key, out)
        return out

    def _ring_numbers(self, ctx: RenderContext, marks) -> "list[tuple[str, float, float, int]]":
        cx, cy, radii = marks[0]
        n = len(radii)
        tick = ctx.height * 0.016 * ctx.field_share
        out: list = []
        min_h = MIN_DIGIT_H_FRAC * ctx.height
        for k in range(n):
            inner = radii[k - 1] if k > 0 else 0.0
            band = radii[k] - inner
            text = str(ring_points(n, k))
            want_h = DIGIT_BAND_FRAC * band
            rmid = (inner + radii[k]) / 2.0 if k > 0 else 0.68 * radii[0]
            size = self._fit_digits(text, want_h, band * 0.92)
            if size is None:
                continue
            h_px = QFontMetricsF(mono(size, True)).boundingRect("8").height()
            if h_px < min_h:
                continue                                   # not legible from several metres: not drawn
            if k == 0 and rmid - h_px / 2.0 < tick + 2.0:
                continue                                   # would sit on the crosshair
            for dx, dy in ((rmid, 0.0), (-rmid, 0.0), (0.0, rmid), (0.0, -rmid)):
                out.append((text, cx + dx, cy + dy, size))
        return out

    @staticmethod
    @lru_cache(maxsize=512)
    def _fit_digits(text: str, want_h: float, max_w: float) -> "int | None":
        """The largest point size whose digit height <= want_h and whose text
        width <= max_w, measured through the font (RULES B5); None if even the
        floor does not fit."""
        def fits(size: int) -> bool:
            fm = QFontMetricsF(mono(size, True))
            return fm.boundingRect("8").height() <= want_h and fm.horizontalAdvance(text) <= max_w
        # Both measures grow with the size, so the answer is the largest size that
        # fits: a binary search (8 font constructions) instead of walking down from
        # 200 one point at a time (up to 193, per ring, on the first paint).
        if not fits(8):
            return None
        lo, hi = 8, 200                       # lo fits; find the largest that does
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


def mark_name(i: int) -> str:
    """The id `score_against_marks` reports for aim mark `i` (0 is the centre).

    One definition, because a mode that tallies hits per mark has to agree with
    the scorer about which mark is which.
    """
    return "centre" if i == 0 else f"mark{i}"


def score_against_marks(x: float, y: float,
                        marks: list[tuple[float, float, list[float]]]) -> tuple[int, str | None]:
    """Score a shot against the aim rings. Returns (points, mark id or None).

    Innermost ring scores highest, outside every ring is a miss. Crude, but it
    makes the HITS and SCORE readouts mean something before real targets exist,
    and it is the same geometry the shooter is looking at.
    """
    best = (0, None)
    for i, (cx, cy, radii) in enumerate(marks):
        d = math.hypot(x - cx, y - cy)
        if d > radii[-1]:
            continue
        ring = next(k for k, r in enumerate(radii) if d <= r)
        points = ring_points(len(radii), ring)  # innermost ring is worth most
        name = mark_name(i)
        if points > best[0]:
            best = (points, name)
    return best


@dataclass
class _Mark:
    nx: float
    ny: float
    track_id: int
    born: float


class CameraSpaceHits(Layer):
    """Recent hits plotted in camera coordinates, inside a labelled frame.

    Explicitly NOT drawn as though it were the screen. Until the homography
    exists we do not know how a camera pixel maps to a screen pixel, and an
    off-axis camera makes the difference large and non-uniform. Drawing it as a
    minimap with its own border keeps that honest; Phase 1 replaces this layer
    with real bullet holes at real screen positions.
    """

    name = "hits"
    z = 20
    animated = True        # markers fade, so this needs repainting

    FADE_S = 6.0
    COLOURS = (GOOD, ACCENT, COOL, QColor("#b48adc"), QColor("#5fd3c8"))

    def __init__(self, aspect: float = 16 / 9, visible: bool = False) -> None:
        # Off by default once real bullet holes exist: this was the honest
        # stand-in for not having calibration, and is now diagnostic clutter.
        self.visible = visible
        self.marks: deque[_Mark] = deque(maxlen=60)
        self.aspect = aspect

    def add(self, nx: float, ny: float, track_id: int = 0) -> None:
        self.marks.append(_Mark(nx, ny, track_id, time.perf_counter()))

    def clear(self) -> None:
        self.marks.clear()

    def set_aspect(self, aspect: float) -> None:
        if aspect > 0:
            self.aspect = aspect

    def on_key(self, key: int) -> bool:
        # E for ERASE. This was C until 2026-10-20, when he asked for the self
        # check to move to C ("lets rewire self-check to C"). The clear needed a
        # new letter and E was the orchestrator's choice, not his: the only free
        # letters were A, E, F, U and Z, and E is the mnemonic. BOTH layers that
        # hold shot history (this one and the bullet holes) moved together, so
        # one key still clears them as one. Overrule it freely; change both.
        if key == Qt.Key_E:
            self.clear()
            return True
        return False

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        # Once the mapping is measured this pane is obsolete: the same shots
        # are drawn as real bullet holes at real screen positions, which is
        # what the class docstring always said should replace it. Left on, it
        # is a 520px box labelled "uncalibrated" sitting over the bottom-right
        # of a calibrated screen -- reported as exactly that confusion.
        if ctx.calibrated:
            return
        p.setRenderHint(QPainter.Antialiasing)
        mw = int(min(ctx.width * 0.30, 520))
        mh = int(mw / self.aspect)
        mx = ctx.rect.x() + ctx.width - mw - 40
        my = ctx.rect.y() + ctx.height - mh - 40

        p.setPen(QPen(COOL, 2))
        p.drawRect(mx, my, mw, mh)
        p.setFont(mono(px(ctx, 10)))
        p.setPen(QPen(COOL))
        p.drawText(mx, my - 8, "camera frame (uncalibrated)")

        now = time.perf_counter()
        newest = self.marks[-1] if self.marks else None
        for m in self.marks:
            age = now - m.born
            if age > self.FADE_S:
                continue
            alpha = int(255 * max(0.1, 1.0 - age / self.FADE_S))
            col = QColor(self.COLOURS[m.track_id % len(self.COLOURS)])
            col.setAlpha(alpha)
            hx = mx + m.nx * mw
            hy = my + m.ny * mh
            r = 9 if m is newest else 5
            p.setPen(QPen(col, 2))
            p.drawEllipse(QPointF(hx, hy), r, r)
            if m is newest:
                p.drawLine(QPointF(hx - 16, hy), QPointF(hx - 6, hy))
                p.drawLine(QPointF(hx + 6, hy), QPointF(hx + 16, hy))
                p.drawLine(QPointF(hx, hy - 16), QPointF(hx, hy - 6))
                p.drawLine(QPointF(hx, hy + 6), QPointF(hx, hy + 16))


@dataclass
class _Hole:
    x: float
    y: float
    born: float
    index: int
    score: int
    seed: int


class BulletHoles(Layer):
    """Impacts drawn where the shot actually landed, in screen coordinates.

    This is the whole point of calibration: the homography turns a camera pixel
    into a screen pixel, so the hole goes where you aimed rather than somewhere
    approximate. Shots that arrive before calibration never reach this layer --
    half-right hole positions would be worse than none, because they would look
    like aiming error.

    Each hole is a dark punched core with a torn rim and a few radial cracks,
    plus a brief impact flash. EVERY colour here is neutral grey (red excess
    exactly 0): the camera is watching this screen and red is what the
    detector hunts for. The palette used to be a warm off-white (rim and
    halo 235/230/220 and 210/205/195, red excess 5) and an amber flash (8) --
    legal under the house limit, and still a standing liability, because the
    saved sensitivity map amplifies the weakest region of the screen up to
    4x, which is how a legal 8 fired at him before. Holes are drawn where a
    shot just landed, i.e. exactly where the detector is looking.
    """

    name = "holes"
    z = 15
    animated = True

    FLASH_S = 0.22
    MAX_HOLES = 80
    # His words, 2026-10-17: "make the bullet holes left on screen smaller, like
    # maybe 20% smaller". A fraction of the SCREEN HEIGHT, as before: 0.8 of the
    # previous 1/110 (and of the previous 7 px floor).
    SIZE_SCALE = 0.8

    def __init__(self, fade_after_s: float = 0.0) -> None:
        self.holes: deque[_Hole] = deque(maxlen=self.MAX_HOLES)
        self.fade_after_s = fade_after_s      # 0 = holes persist until cleared
        self._seed = 0
        # Settled holes, drawn once into a pixmap and blitted thereafter.
        # See `paint` for why this is not a micro-optimisation.
        self._baked: QPixmap | None = None
        self._baked_keys: tuple = ()
        self._baked_size: tuple[int, int] = (0, 0)
        self._baked_dark = True

    @classmethod
    def hole_radius(cls, ctx: RenderContext) -> float:
        """Radius of a drawn hole in px: a fraction of the screen height."""
        return cls.SIZE_SCALE * max(7.0, ctx.height / 110.0)

    @property
    def added(self) -> int:
        """Holes ever added: monotonic, unlike `len(holes)` (capped at 80 and
        cleared by C). The mark watchdog compares it before and after."""
        return self._seed

    def add(self, x: float, y: float, index: int = 0, score: int = 0) -> None:
        self._seed += 1
        self.holes.append(_Hole(x, y, time.perf_counter(), index, score, self._seed))

    def clear(self) -> None:
        self.holes.clear()
        self._baked = None

    def move(self, index: int, x: float, y: float) -> bool:
        """Nudge an existing hole once its final position is known.

        Holes are drawn the instant a shot is credible, which is well before the
        track closes. The refined position arrives a few frames later and moves
        it by a fraction of a pixel -- but moving it is still right, and keying
        on the shot index means a late update can never create a second hole.
        """
        for h in self.holes:
            if h.index == index:
                h.x, h.y = x, y
                return True
        return False

    def on_key(self, key: int) -> bool:
        # E for ERASE. This was C until 2026-10-20, when he asked for the self
        # check to move to C ("lets rewire self-check to C"). The clear needed a
        # new letter and E was the orchestrator's choice, not his: the only free
        # letters were A, E, F, U and Z, and E is the mnemonic. BOTH layers that
        # hold shot history (this one and the bullet holes) moved together, so
        # one key still clears them as one. Overrule it freely; change both.
        if key == Qt.Key_E:
            self.clear()
            return True
        return False

    def advance(self, ctx: RenderContext) -> bool:
        # Only repaint while something is actually animating or fading.
        now = time.perf_counter()
        for h in self.holes:
            if now - h.born < self.FLASH_S:
                return True
        return bool(self.fade_after_s)

    def _draw_hole(self, p: QPainter, h, r: float, dark: bool,
                   alpha: float) -> None:
        """One hole: halo, rim, core and cracks. Eight antialiased primitives."""

        # A dark core alone is invisible on a dark screen, so the hole is
        # built from a soft halo plus a bright torn rim: light on a dark
        # background, dark on a light one, so it reads against either.
        halo = grey_at(0.88) if dark else QColor(24, 24, 24)
        halo.setAlphaF(0.16 * alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(halo)
        p.drawEllipse(QPointF(h.x, h.y), r * 1.7, r * 1.7)

        rim = grey_at(1.0) if dark else QColor(28, 28, 28)
        rim.setAlphaF(0.9 * alpha)
        p.setPen(QPen(rim, max(2.0, r * 0.26)))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(h.x, h.y), r, r)

        core = QColor(8, 8, 10)
        core.setAlphaF(0.92 * alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(core)
        p.drawEllipse(QPointF(h.x, h.y), r * 0.72, r * 0.72)

        # Cracks. Deterministic from the hole's own seed so a given hole
        # looks identical every frame instead of shimmering.
        p.setBrush(Qt.NoBrush)
        crack = QColor(rim)
        crack.setAlphaF(0.7 * alpha)
        p.setPen(QPen(crack, max(1.0, r * 0.10)))
        for k in range(5):
            ang = (h.seed * 37 + k * 72) % 360 * math.pi / 180.0
            length = r * (1.25 + ((h.seed * 13 + k * 29) % 7) / 10.0)
            p.drawLine(QPointF(h.x + math.cos(ang) * r * 0.8,
                           h.y + math.sin(ang) * r * 0.8),
                   QPointF(h.x + math.cos(ang) * length,
                           h.y + math.sin(ang) * length))


    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        """Draw the holes, leaving out whatever a layer above is drawing solid.

        A figure drill's sprite is line art on a TRANSPARENT ground, so holes
        underneath it would show through its dark areas and the figure would
        look like it floats through them (2026-10-17). Layers above announce
        their solid shapes (`Layer.occluders`); here those areas are cut out of
        the holes. Only when a hole is actually near one does this cost
        anything: the common case is the one plain pass.
        """
        if not self.holes:
            self._baked = None
            return
        reach = self.hole_radius(ctx) * 4.5       # halo / flash / cracks
        occ = [o for o in ctx.occluders if o.z > self.z and any(
            o.rect.adjusted(-int(reach), -int(reach), int(reach), int(reach))
            .contains(int(h.x), int(h.y)) for h in self.holes)]
        if not occ:
            self._paint_all(p, ctx)
            return
        full = QRegion(ctx.rect)
        for o in occ:
            full = full.subtracted(QRegion(o.rect))
        p.setClipRegion(full)
        self._paint_all(p, ctx)
        done = QRegion()
        for o in occ:
            # The holes inside this box, drawn into a scratch pixmap, with
            # every solid shape punched out of them, then blitted back.
            scratch = QPixmap(o.rect.size())
            scratch.fill(Qt.transparent)
            tp = QPainter(scratch)
            tp.translate(-o.rect.x(), -o.rect.y())
            self._paint_all(tp, ctx)
            tp.resetTransform()
            tp.setRenderHint(QPainter.SmoothPixmapTransform)
            tp.setCompositionMode(QPainter.CompositionMode_DestinationOut)
            for o2 in occ:
                if o2.rect.intersects(o.rect):
                    mk = o2.mask()
                    tp.drawPixmap(QRect(o2.rect.x() - o.rect.x(),
                                        o2.rect.y() - o.rect.y(),
                                        o2.rect.width(), o2.rect.height()),
                                  mk, mk.rect())
            tp.end()
            p.setClipRegion(QRegion(o.rect).subtracted(done))
            p.drawPixmap(o.rect.topLeft(), scratch)
            done = done.united(QRegion(o.rect))

    def _paint_all(self, p: QPainter, ctx: RenderContext) -> None:
        """Blit the settled holes, draw only the live ones.

        Every hole costs eight antialiased primitives -- a halo, a rim, a
        core and five cracks -- and they were all redrawn on every repaint
        of a 3840x2160 surface. At the 80-hole cap that is 640 primitives a
        frame, and the detector is competing for the same CPU.

        Measured live while the shooter was mid-string: the app fell from
        22 fps to 18 as holes accumulated, and the tracker rejected 32
        pulses as too short against 18 shots that got through. That is the
        loop: more hits -> slower repaint -> fewer camera frames per pulse
        -> more shots discarded. The app performed worst exactly when it was
        being used most, which is also why it looked so inconsistent.

        So a settled hole is drawn ONCE into a pixmap and blitted after
        that. Only holes still flashing, or fading, are drawn live. The cost
        of a repaint stops growing with the score.
        """
        if not self.holes:
            self._baked = None
            return
        p.setRenderHint(QPainter.Antialiasing)
        now = time.perf_counter()
        # Scale with the display so a hole looks the same on a 4K monitor and
        # a projector rather than becoming a speck on one and a blob on the
        # other.
        r = self.hole_radius(ctx)
        dark = ctx.dark_background
        w, h_px = int(ctx.width), int(ctx.height)

        live = [h for h in self.holes if now - h.born < self.FLASH_S]
        if self.fade_after_s:
            # Fading holes change every frame, so none of them can be baked.
            for hole in self.holes:
                age = now - hole.born
                alpha = 1.0
                if age > self.fade_after_s:
                    alpha = max(0.0, 1.0 - (age - self.fade_after_s) / 3.0)
                    if alpha <= 0.0:
                        continue
                self._draw_hole(p, hole, r, dark, alpha)
                self._flash(p, hole, r, now)
            return

        settled = [hole for hole in self.holes
                   if now - hole.born >= self.FLASH_S]
        # The identity of the baked set: which holes, where, and at what
        # size. A hole that MOVES when its refined position arrives changes
        # this, so the pixmap is rebuilt rather than left stale.
        key = tuple((hole.seed, round(hole.x, 1), round(hole.y, 1))
                    for hole in settled)
        if (self._baked is None or self._baked_keys != key
                or self._baked_size != (w, h_px)
                or self._baked_dark != dark):
            pm = QPixmap(w, h_px)
            pm.fill(Qt.transparent)
            bp = QPainter(pm)
            bp.setRenderHint(QPainter.Antialiasing)
            for hole in settled:
                self._draw_hole(bp, hole, r, dark, 1.0)
            bp.end()
            self._baked = pm
            self._baked_keys = key
            self._baked_size = (w, h_px)
            self._baked_dark = dark
        p.drawPixmap(0, 0, self._baked)

        for hole in live:
            self._draw_hole(p, hole, r, dark, 1.0)
            self._flash(p, hole, r, now)

    def _flash(self, p: QPainter, h, r: float, now: float) -> None:
        """The brief impact ring. Neutral grey (red excess 0), never warm."""
        age = now - h.born
        if age >= self.FLASH_S:
            return
        t = age / self.FLASH_S
        flash = QColor(GREY_HI)
        flash.setAlphaF((1.0 - t) * 0.85)
        p.setPen(QPen(flash, max(2.0, r * 0.35 * (1.0 - t))))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(h.x, h.y), r * (1.0 + t * 2.6), r * (1.0 + t * 2.6))


class CalibrationPattern(Layer):
    """One bright marker at a known screen position, for photographing.

    This is the whole display side of calibrating without shooting: put a
    square of known centre on a black screen, let the camera photograph it,
    move it, repeat. The screen is a light source the camera can already see,
    so no trigger pulls are needed and the far side of the panel -- where a
    laser dot is too small to register -- is no harder than the near side.

    White is laser-safe by construction: the chroma detector scores red minus
    the strongest of green and blue, which is zero when all three are equal.
    """

    name = "pattern"
    z = 80                      # above everything; nothing else may be visible
    animated = False

    def __init__(self) -> None:
        self.visible = False
        self.point: tuple[float, float] | None = None
        self.size_frac = 0.055  # of the screen height
        self.caption = ""
        self.full = False
        self.done = 0
        self.total = 0
        self.note = ""

    def progress(self, done: int, total: int, note: str = "") -> None:
        """How far through the sequence we are. Drawn as a bar at the bottom.

        Asked for directly: "when you're measuring the camera I need you to
        display the progress bar." Until this, the screen showed a caption and
        then eighteen silent flashes with no indication of how many remained or
        whether anything was working -- from the firing line, identical to an
        app that had hung.

        Safe to draw on the photographed screen for one reason, and it is worth
        stating: the bar is identical in the lit and unlit frames of any single
        marker, because it only advances BETWEEN markers. Subtraction therefore
        cancels it exactly, so it cannot be mistaken for a marker. It is also
        grey -- red excess zero -- so the detector cannot see it either.
        """
        self.done = max(0, int(done))
        self.total = max(0, int(total))
        self.note = note

    def show_at(self, point: tuple[float, float] | None) -> None:
        self.point = point
        self.full = False
        self.visible = True

    def show_full(self) -> None:
        """Light the entire screen, to find where the monitor is in the frame."""
        self.point = None
        self.full = True
        self.visible = True

    def stop(self) -> None:
        self.visible = False
        self.point = None

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        # Black everywhere else, so the only thing that changes between the lit
        # and unlit photographs is the marker itself.
        p.fillRect(ctx.rect, QColor("#000000"))
        if self.caption:
            p.setPen(QPen(QColor("#5a5a5a")))
            p.setFont(mono(px(ctx, 15), bold=True))
            p.drawText(ctx.rect.x() + 48, ctx.rect.y() + px(ctx, 60), self.caption)
        if self.full:
            p.fillRect(ctx.rect, QColor("#ffffff"))
            return
        self._paint_progress(p, ctx)
        if self.point is None:
            return
        side = max(18.0, ctx.height * self.size_frac)
        x, y = self.point
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#ffffff"))
        p.drawRect(QRectF(x - side / 2, y - side / 2, side, side))

    def _paint_progress(self, p: QPainter, ctx: RenderContext) -> None:
        if self.total <= 0:
            return
        # Along the BOTTOM edge, below every marker position. Markers are inset
        # 8% from each edge and are 5.5% of the height across, so the lowest
        # one reaches about 95% of the height; the bar lives under that and
        # cannot be confused with, or overlap, a marker.
        pad = px(ctx, 40)
        bar_h = px(ctx, 10)
        y = ctx.rect.y() + ctx.height - px(ctx, 30)
        x0 = ctx.rect.x() + pad
        width = ctx.width - pad * 2
        frac = min(1.0, self.done / float(self.total))

        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#242424"))                  # the trough
        p.drawRect(QRectF(x0, y, width, bar_h))
        p.setBrush(QColor("#8a8a8a"))                  # grey: invisible to the detector
        p.drawRect(QRectF(x0, y, width * frac, bar_h))

        # Tick marks, one per marker, so the count is readable at a glance from
        # several metres down the range rather than only as a length.
        p.setBrush(QColor("#000000"))
        for i in range(1, self.total):
            tx = x0 + width * (i / float(self.total))
            p.drawRect(QRectF(tx - 1, y, 2, bar_h))

        label = f"measuring the camera's view: {self.done} of {self.total}"
        if self.note:
            label += f"   {self.note}"
        p.setPen(QPen(QColor("#8a8a8a")))
        p.setFont(mono(px(ctx, 14), bold=True))
        p.drawText(int(x0), int(y - px(ctx, 8)), label)


class CalibrationTargets(Layer):
    """The shoot-this-dot sequence that measures the camera-to-screen mapping.

    One point at a time, large and unambiguous. Points already done are shown
    small so the shooter can see progress and where the grid is going; points
    that were skipped are marked, because a point that cannot be hit is itself a
    measurement -- it says the camera cannot see that part of the screen.
    """

    name = "calibration"
    z = 60
    animated = True

    def __init__(self) -> None:
        self.visible = False
        self.points: list[tuple[float, float]] = []
        self.index = 0
        self.done: list[int] = []
        self.skipped: list[int] = []
        self.message = ""
        # --- per-point progress ---------------------------------------------
        # A point is only recorded once several shots AGREE with each other, so
        # the shooter needs to know how many have landed and how many of those
        # agree. Without it, firing at a ring that will not advance is
        # indistinguishable from firing at a camera that is not working, and the
        # only available response is to keep shooting and hope.
        self.need = 3
        self.landed = 0           # shots the detector saw at this point
        self.agreeing = 0         # how many of them agree with each other
        self.signal = ""           # what happened to the last shot
        self.stall = ""            # sticky advice when nothing is arriving
        self.level = ""            # live signal against the threshold
        # What this run calls itself, so the two shooting calibrations (G
        # measures the mapping, U measures the brightness) are not the same
        # screen. `start` resets all three.
        self.title = self.DEFAULT_TITLE
        self.unit = self.DEFAULT_UNIT
        self.footer = self.DEFAULT_FOOTER

    DEFAULT_TITLE = "CALIBRATION"
    DEFAULT_UNIT = "point"
    DEFAULT_FOOTER = ("SPACE skips a point you cannot hit or the camera cannot see.",
                      "BACKSPACE redoes the last point.  ESC abandons calibration.")

    def start(self, points: list[tuple[float, float]], need: int = 3,
              title: str = DEFAULT_TITLE, unit: str = DEFAULT_UNIT,
              footer: "tuple[str, ...] | None" = None) -> None:
        self.points = list(points)
        self.index = 0
        self.done = []
        self.skipped = []
        self.need = max(1, int(need))
        self.title = title
        self.unit = unit
        self.footer = tuple(footer) if footer else self.DEFAULT_FOOTER
        self.reset_point()
        self.visible = True

    def reset_point(self) -> None:
        self.landed = 0
        self.agreeing = 0
        self.message = ""
        self.signal = ""

    def stop(self) -> None:
        self.visible = False

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if not self.points:
            return
        p.setRenderHint(QPainter.Antialiasing)
        # Dim the rest of the screen so the live target is unmistakable, and so
        # the screen stays dark for the camera while calibrating.
        p.fillRect(ctx.rect, QColor(0, 0, 0, 210))

        for i, (x, y) in enumerate(self.points):
            if i == self.index:
                continue
            col = GOOD if i in self.done else (COOL if i not in self.skipped else INK_DARK)
            c = QColor(col)
            c.setAlphaF(0.45)
            p.setPen(QPen(c, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(x, y), 14, 14)
            if i in self.skipped:
                p.drawLine(QPointF(x - 10, y - 10), QPointF(x + 10, y + 10))
                p.drawLine(QPointF(x - 10, y + 10), QPointF(x + 10, y - 10))

        # Sized from the screen, not from the 1080p window this was written in:
        # the real panel is 42 inches of 4K read from several metres away.
        title_px = max(20, ctx.height // 40)
        body_px = max(14, ctx.height // 62)
        panel_x, panel_w = self._panel_column(ctx)
        # How many characters fit, measured rather than assumed.
        p.setFont(mono(body_px))
        ch = max(1, p.fontMetrics().horizontalAdvance("M"))
        wrap_at = max(20, int((panel_w - 48) / ch))
        lines_below = (7 + len(_wrap(self.message, wrap_at))
                       + len(_wrap(self.stall, wrap_at)))
        panel_h = title_px * 2.0 + body_px * 1.75 * lines_below + 40
        top = max(ctx.rect.y() + top_band_height(ctx) + 24,
                  ctx.rect.y() + (ctx.height - panel_h) / 2)

        x0 = panel_x + 24
        y = top + title_px * 1.6
        panel = QRectF(panel_x, top, panel_w, panel_h)
        p.setBrush(QColor(8, 10, 12, 232))
        p.setPen(QPen(QColor("#2d4c5e"), 1))
        p.drawRoundedRect(panel, 10, 10)
        p.setBrush(Qt.NoBrush)

        p.setPen(QPen(QColor("#e6e6e6")))
        p.setFont(mono(title_px, bold=True))
        p.drawText(x0, y,
                   f"{self.title}   {self.unit} {min(self.index + 1, len(self.points))} "
                   f"of {len(self.points)}")

        # Shots needed at THIS ring, as pips: filled for shots that agree with
        # each other, outlined for the ones still wanted. This is the answer to
        # "do I keep shooting at this one?" and it is worth more than any text.
        y += title_px * 1.9
        pip_r = max(7.0, body_px * 0.5)
        p.setFont(mono(body_px))
        p.setPen(QPen(QColor("#b4b4b4")))
        p.drawText(x0, y + pip_r * 0.7, "agreeing shots")
        cx = x0 + body_px * 13
        for i in range(self.need):
            filled = i < self.agreeing
            p.setPen(QPen(GOOD if filled else INK_DARK, 2))
            p.setBrush(GOOD if filled else Qt.NoBrush)
            p.drawEllipse(QPointF(cx + i * pip_r * 3.2, y), pip_r, pip_r)
        p.setBrush(Qt.NoBrush)
        if self.landed:
            p.setPen(QPen(QColor("#b4b4b4")))
            p.drawText(cx + self.need * pip_r * 3.2 + body_px,
                       y + pip_r * 0.7, f"{self.landed} shot(s) seen here")

        y += body_px * 2.4
        step = body_px * 1.75
        for text, col in ((self.level, COOL), (self.signal, COOL),
                          (self.message, QColor("#e6e6e6")),
                          (self.stall, ACCENT)):
            if not text:
                continue
            p.setPen(QPen(col))
            p.setFont(mono(body_px, bold=text is self.stall))
            for line in _wrap(text, wrap_at):
                p.drawText(x0, y, line)
                y += step
            y += step * 0.35

        p.setPen(QPen(QColor("#8a8a8a")))
        p.setFont(mono(max(12, int(body_px * 0.85))))
        small = max(20, int((panel_w - 48)
                            / max(1, p.fontMetrics().horizontalAdvance("M"))))
        for line in self.footer:
            for row in _wrap(line, small):
                p.drawText(x0, y, row)
                y += step * 0.92

        self._draw_live(p, ctx)

    def _panel_column(self, ctx: RenderContext) -> tuple[float, float]:
        """A vertical corridor between grid columns, where no ring can ever be.

        Rows are 737 px apart on this panel and the block is 624 px tall, so it
        never fitted between two rows. Columns are 1690 px apart and a ring is
        about 200 px wide, so a corridor between two columns is 1474 px of
        guaranteed empty space at EVERY height -- which turns "the text covers
        a target" from unlikely into impossible.

        Reported twice before it was taken seriously enough to make structural:
        "some of those targets had been obscured by the text again".
        """
        margin = 60.0
        default = (ctx.rect.x() + margin, ctx.width * 0.56)
        cols = sorted({x for x, _y in self.points})
        if len(cols) < 2:
            return default
        clear = max(26.0, ctx.height / 32.0) * 1.6      # ring plus its ticks
        live_x = (self.points[self.index][0]
                  if 0 <= self.index < len(self.points) else cols[0])
        best = None
        for left, right in zip(cols, cols[1:]):
            lo, hi = left + clear, right - clear
            if hi - lo < 600:
                continue
            # Prefer the corridor furthest from the ring being shot at, so
            # reading and aiming do not happen in the same place.
            score = abs((lo + hi) / 2.0 - live_x)
            if best is None or score > best[0]:
                best = (score, lo, hi - lo)
        return default if best is None else (best[1], best[2])

    def _draw_live(self, p: QPainter, ctx: RenderContext) -> None:
        """The ring being shot at, drawn over everything else on this layer.

        The grid starts 6% in from the corners, and on a wide screen that puts
        the first point behind the instruction text. Of all the things that may
        overlap, the one that must not be obscured is the thing being aimed at.
        """
        if not (0 <= self.index < len(self.points)):
            return
        x, y = self.points[self.index]
        r = max(26.0, ctx.height / 32.0)
        p.setBrush(Qt.NoBrush)
        for k, rad in enumerate((r, r * 0.62, r * 0.3)):
            c = QColor(ACCENT)
            c.setAlphaF(0.9 - k * 0.15)
            p.setPen(QPen(c, 3))
            p.drawEllipse(QPointF(x, y), rad, rad)
        p.setPen(QPen(ACCENT, 2))
        p.drawLine(QPointF(x - r * 1.5, y), QPointF(x - r * 1.15, y))
        p.drawLine(QPointF(x + r * 1.15, y), QPointF(x + r * 1.5, y))
        p.drawLine(QPointF(x, y - r * 1.5), QPointF(x, y - r * 1.15))
        p.drawLine(QPointF(x, y + r * 1.15), QPointF(x, y + r * 1.5))


class CameraPane(Layer):
    """A live camera view on the target screen -- made safe by being blue.

    Showing the camera feed on the surface the camera is pointed at is an
    infinite mirror, and the laser dot appears recursively inside it. Each
    reflection is a real bright red spot as far as the detector is concerned,
    so a naive picture-in-picture manufactures phantom hits.

    The fix: composite the frame through a multiply with pure blue, which zeroes
    the red channel outright. The chroma detector scores red minus the strongest
    of green and blue, so a blue image scores zero however many times it
    recurses. Useful side effect: it is instantly obvious which pane is the
    camera and which is the game.

    Caveat worth knowing: this protects the CHROMA score mode completely and the
    RED mode substantially, but GRAY mode (used once a 650nm filter is fitted)
    responds to brightness alone -- though with the filter on, a blue image
    reaches the sensor attenuated to nothing anyway.
    """

    name = "camera"
    z = 30

    # STRUCTURAL, not a default (2026-10-17). A live camera image on the
    # surface the camera is pointed at is a bright, changing, drawn object,
    # and the detector hunts exactly that. Measured the morning of
    # 2026-10-17: the pane came up at 08:50:43 (key 0 had switched the
    # whole information group on), a speck storm began at 08:51 and ran at
    # 172 detections against 3 real shots, repeating at fixed camera cells
    # that map to the pane's own footprint, 102 of them more than two
    # seconds apart. A stand-down that runs once, and a saved setting or a
    # key that can switch it back on, is a default. So the pane draws its
    # image ONLY while it is being used to AIM the camera -- the one moment
    # no trigger pull can exist, because the app is showing a problem and
    # is not shootable. In every other state `paint` draws nothing at all,
    # whatever `visible` says, and the layer asks for no repaints. A live
    # view while shooting belongs on the operator console, which the
    # camera does not watch.
    @property
    def animated(self) -> bool:
        return bool(self.aiming and self.visible)

    def draws_camera_image(self) -> bool:
        """True only while the camera is being aimed."""
        return bool(self.aiming)

    BLUE = QColor(0, 70, 255)
    POSITIONS = ("top-right", "bottom-left", "top-left", "bottom-right")

    def __init__(self, visible: bool = False, width_frac: float = 0.22,
                 position: str = "top-right", opacity: float = 0.85) -> None:
        self.visible = visible
        self.width_frac = width_frac
        self.position = position
        self.opacity = opacity
        self._image: QImage | None = None
        self._grey_cache: QImage | None = None
        self._label = ""
        self._stale_since = 0.0
        # Greyscale rather than blue-multiplied. Both are safe -- the detector
        # scores red minus the strongest of green and blue, and in greyscale the
        # three channels are equal, so that difference is exactly zero -- but
        # only one of them is legible. The blue pane is unreadable, which makes
        # it useless for the one job it is most needed for: aiming the camera at
        # the screen in the first place.
        #
        # The exception is RED or GRAY score mode, where brightness alone counts
        # and only the blue multiply protects us. `set_score_mode` handles that.
        self.greyscale = True
        self.aiming = False                       # big, opaque, outline drawn
        self.outline: tuple | None = None         # detected screen quad, camera px
        self.frame_size: tuple[int, int] = (0, 0)
        # True when the outline is the live bright region rather than the quad
        # left over from the last flash search. The difference matters while
        # somebody is physically moving the camera.
        self.live_outline = False

    def set_score_mode(self, mode_name: str) -> None:
        """Greyscale is only safe for CHROMA; brightness modes need the blue."""
        self.greyscale = str(mode_name).upper().endswith("CHROMA")

    def set_aiming(self, on: bool, outline=None, frame_size=(0, 0)) -> None:
        self.aiming = bool(on)
        self.outline = outline
        self.frame_size = frame_size

    def _grey(self) -> QImage:
        if self._grey_cache is None and self._image is not None:
            self._grey_cache = self._image.convertToFormat(QImage.Format_Grayscale8)
        return self._grey_cache if self._grey_cache is not None else self._image

    def _draw_outline(self, p: QPainter, box: QRect) -> None:
        """Where the app thinks the monitor is, drawn over the live view.

        Without this the shooter is asked to judge "does the camera see the
        whole screen" by eye, from a small picture, while the screen in that
        picture contains a picture of itself. The outline turns that into a
        yes/no they can actually answer.
        """
        fw, fh = self.frame_size
        if not fw or not fh:
            return
        p.setBrush(Qt.NoBrush)

        # The target: where the monitor SHOULD sit in the camera's view. A
        # dashed box with room to spare all round, because an outline that
        # touches the frame edge means part of the screen is already outside it.
        if self.aiming:
            inset_x, inset_y = box.width() * 0.10, box.height() * 0.10
            want = QRectF(box.x() + inset_x, box.y() + inset_y,
                          box.width() - 2 * inset_x, box.height() - 2 * inset_y)
            pen = QPen(COOL, 2, Qt.DashLine)
            p.setPen(pen)
            p.drawRect(want)
            p.setFont(mono(max(9, box.height() // 26), bold=True))
            p.drawText(QPointF(want.x() + 6, want.y() - 6),
                       "PUT THE WHOLE MONITOR IN HERE")

        if not self.outline:
            if self.aiming:
                p.setFont(mono(max(10, box.height() // 20), bold=True))
                p.setPen(QPen(ACCENT))
                p.drawText(box, Qt.AlignCenter,
                           "no bright screen found -\npoint the camera at the monitor")
            return

        sx, sy = box.width() / float(fw), box.height() / float(fh)
        pts = [QPointF(box.x() + x * sx, box.y() + y * sy) for x, y in self.outline]
        p.setPen(QPen(GOOD, 3))
        p.drawPolygon(pts)
        p.setFont(mono(max(9, box.height() // 22), bold=True))
        p.setPen(GOOD)
        label = "SCREEN (live)" if self.live_outline else "SCREEN (last measured)"
        p.drawText(QPointF(pts[0].x() + 6, pts[0].y() + 16), label)

        # How much of the frame it fills, which is the number that decides
        # whether the calibration about to follow can work at all.
        if self.aiming:
            xs = [q.x() for q in pts]
            ys = [q.y() for q in pts]
            frac = (((max(xs) - min(xs)) * (max(ys) - min(ys)))
                    / max(1.0, box.width() * box.height()))
            good = 0.20 <= frac <= 0.85
            p.setPen(QPen(GOOD if good else ACCENT))
            p.setFont(mono(max(10, box.height() // 24), bold=True))
            p.drawText(QPointF(box.x() + 8, box.bottom() - 10),
                       f"screen fills {frac * 100:.0f}% of the view"
                       + ("  - good" if good else "  - aim for 20-85%"))

    def set_frame(self, image: QImage, label: str = "") -> None:
        self._image = image
        self._grey_cache = None          # a new frame invalidates the conversion
        self._label = label
        self._stale_since = time.perf_counter()

    def on_key(self, key: int) -> bool:
        if key == Qt.Key_BracketLeft:
            self.width_frac = max(0.10, self.width_frac - 0.03)
            return True
        if key == Qt.Key_BracketRight:
            self.width_frac = min(0.60, self.width_frac + 0.03)
            return True
        if key == Qt.Key_J:
            # J, NOT M: he asked for M to open the menu (2026-10-18) and a
            # layer claiming a key SHADOWS the app silently -- G once switched
            # the background to grey instead of calibrating for a whole
            # session. J is unused anywhere in src. This pane draws nothing on
            # the target screen unless he is aiming (`wants_paint`), so the key
            # is nearly dead weight; it is kept rather than dropped because
            # removing a capability is his call, not mine.
            i = self.POSITIONS.index(self.position) if self.position in self.POSITIONS else 0
            self.position = self.POSITIONS[(i + 1) % len(self.POSITIONS)]
            return True
        return False

    def _geometry(self, ctx: RenderContext, aspect: float) -> QRect:
        # While aiming, take over most of the screen. A thumbnail is no use for
        # deciding whether the monitor fills the camera's view.
        frac = 0.55 if self.aiming else self.width_frac
        pw = int(ctx.width * frac)
        ph = int(pw / max(0.1, aspect))
        # `px`, not 40. A 42-inch 4K panel is about 104 DPI, the same as the
        # 1080p window this was laid out in, so a pixel constant is the same
        # apparent size here and a quarter of it on the 75" TV (RULES B5).
        pad = px(ctx, 20)
        if self.aiming:
            return QRect(ctx.rect.x() + (ctx.width - pw) // 2,
                         ctx.rect.y() + int(ctx.height * 0.46), pw, ph)
        x = ctx.rect.x() + pad
        y = ctx.rect.y() + ctx.height - ph - pad
        if "top" in self.position:
            # BELOW THE BAND. `top_band_height` says in its own docstring
            # that it is "reserved unconditionally, by every layer that
            # draws near the top" -- and this was the one layer that never
            # asked. He saw it before I did, looking at the same screenshot
            # I had open: "the screen in the upper right corner is partially
            # obscured by the green panel". A rule written down in one place
            # and applied in all but one is how every expensive bug here has
            # survived (Part E).
            y = ctx.rect.y() + top_band_height(ctx) + pad
        if "right" in self.position:
            x = ctx.rect.x() + ctx.width - pw - pad
        return QRect(x, y, pw, ph)

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if not self.draws_camera_image():
            return                  # see the class comment: never while shooting
        aspect = 16 / 9
        if self._image is not None and self._image.height() > 0:
            aspect = self._image.width() / self._image.height()
        box = self._geometry(ctx, aspect)

        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(box, QColor("#000814"))

        if self._image is None:
            p.setPen(QPen(COOL))
            p.setFont(mono(px(ctx, 11)))
            p.drawText(box, Qt.AlignCenter, "no camera frame")
        else:
            p.save()
            p.setClipRect(box)
            p.setOpacity(1.0 if self.aiming else self.opacity)
            p.drawImage(box, self._grey() if self.greyscale else self._image)
            if not self.greyscale:
                # Multiply away the red channel so this pane cannot read as a
                # laser hit, no matter how the mirror recurses.
                p.setCompositionMode(QPainter.CompositionMode_Multiply)
                p.fillRect(box, self.BLUE)
            p.restore()
            if self.outline:
                self._draw_outline(p, box)

        p.setPen(QPen(COOL, 2))
        p.drawRect(box)
        p.setFont(mono(px(ctx, 10)))
        p.setPen(QPen(COOL))
        stale = (time.perf_counter() - self._stale_since) > 1.5 if self._image is not None else False
        caption = f"camera view (blue = laser-safe){'  [STALLED]' if stale else ''}"
        p.drawText(box.x(), box.y() - 8, caption)
        if self._label:
            p.drawText(box.x() + 6, box.bottom() - 8, self._label)


class StatsHud(Layer):
    """The scoreboard: big enough to read from the firing position.

    Two blocks. The headline numbers are what you care about while shooting;
    the diagnostics block is what tells you whether to trust them, and can be
    folded away with the same key once the setup is known good.
    """

    name = "stats"
    # BELOW the aim marks, the modes and the holes. Reported from the firing
    # line: "the upper targets are obscured by text" -- the scoreboard was
    # drawn over the upper-left aim ring. Furniture is informative; a target is
    # the thing being aimed at, and when they overlap the target wins. Moving
    # the text elsewhere only relocates the collision, because every part of
    # this screen is somewhere a target may legitimately be.
    z = -10
    animated = True        # the clock runs
    # Twelve times a second. These are numbers read from several
    # metres down a range; faster is a 4K repaint nobody can perceive,
    # paid for out of the same CPU the detector runs on.
    read_rate_hz = 12.0

    def __init__(self, show_diagnostics: bool = True) -> None:
        self.show_diagnostics = show_diagnostics

    def on_key(self, key: int) -> bool:
        if key == Qt.Key_I:
            self.show_diagnostics = not self.show_diagnostics
            return True
        return False

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if ctx.calibrating:
            return          # the calibration overlay owns the top-left
        s = ctx.stats
        ink = QColor("#e6e6e6") if ctx.dark_background else QColor("#101010")
        big, small, body = px(ctx, 44), px(ctx, 14), px(ctx, 15)
        col = px(ctx, 190)
        x = ctx.rect.x() + 48
        y = ctx.rect.y() + top_band_height(ctx) + px(ctx, 54)

        p.setPen(QPen(ink))
        p.setFont(mono(big, bold=True))
        p.drawText(x, y, f"{s.shots}")
        p.setFont(mono(small))
        p.drawText(x + 6, y + px(ctx, 26), "SHOTS")

        p.setFont(mono(big, bold=True))
        hits_text = str(s.hits) if s.scored_shots else "--"
        p.drawText(x + col, y, hits_text)
        p.setFont(mono(small))
        p.drawText(x + col + 6, y + px(ctx, 26), "HITS")

        p.setFont(mono(big, bold=True))
        p.drawText(x + col * 2, y, s.clock_string())
        p.setFont(mono(small))
        p.drawText(x + col * 2 + 6, y + px(ctx, 26), "ELAPSED")

        yy = y + px(ctx, 76)
        p.setFont(mono(body))
        rate = s.hit_rate
        lines = [
            f"hit rate     {rate * 100:5.1f}%" if rate is not None
            else "hit rate     -- (needs calibration)",
            f"string time  {s.string_time_s:6.2f} s",
            f"last split   {s.last_split_ms:7.0f} ms" if s.last_split_ms is not None
            else "last split      -- ms",
            f"best split   {s.best_split_ms:7.0f} ms" if s.best_split_ms is not None
            else "best split      -- ms",
            f"cadence      {s.shots_per_minute:6.1f} /min",
        ]
        if s.label:
            lines.insert(0, f"drill        {s.label}")
        for line in lines:
            p.drawText(x, yy, line)
            yy += px(ctx, 26)

        if self.show_diagnostics and ctx.diagnostics:
            yy += px(ctx, 14)
            p.setPen(QPen(ACCENT if ctx.dark_background else ACCENT_ON_LIGHT))
            p.setFont(mono(px(ctx, 13)))
            p.drawText(x, yy, "-- detector ----------------------")
            yy += px(ctx, 22)
            for line in str(ctx.diagnostics.get("text", "")).splitlines():
                p.drawText(x, yy, line)
                yy += px(ctx, 22)


class KeyHelp(Layer):
    """Bottom-of-screen key legend. Folds away once you know the keys."""

    name = "help"
    z = -12                # furniture: never over a target
    visible = True         # he asked for it to be THERE, not to be found

    # (key, what it does). Asked for 2026-10-18: "a list of keys in the lower
    # screen corner in gray letters, same color as the targets ... keys in
    # BOLD, text in normal".
    #
    # ONE TABLE, laid out in rows, and `lines` is derived from it. The keys
    # and the prose used to be a single hand-typed string per row, so the
    # rendered text and the thing a test could read were the same only by
    # the author's care.
    # THE DRILL ENTRIES (digits 1-9 and M) MUST AGREE WITH `lsa.modes.registered()`
    # key for key and title for title; `tests/test_ui_smoke.py` fails if they
    # drift. This table was hand-written once and then lied for three days after
    # the drills were renumbered on 2026-10-18. It stays hand-written HERE
    # because `display` must not import `modes` (a circular import); the test
    # is what holds the two together.
    ROWS: tuple[tuple[tuple[str, str], ...], ...] = (
        (("1", "Static Targets"), ("2", "Escalating Ball"),
         ("3", "Bouncing Balls"), ("4", "Reaction"),
         ("5", "Clear The Board")),
        (("6", "Pop-Up Target"), ("7", "Static Target"),
         ("8", "Moving Target"), ("9", "Animated Target")),
        (("M", "Main Menu"), ("K", "Keyboard Shortcuts"), ("N", "new round"), ("0", "hide/show the info"),
         ("O", "One Player (menu)"), ("T", "Two Players (menu)")),
        ((" ", "(pressing a mode's key again = a new round of that mode)"),),
        (("C", "self-calibrate"), ("A", "self-check"),
         ("G", "calibrate by shooting"),
         ("U", "calibrate brightness by shooting")),
        (("X", "measure the exposure"), ("F", "measure the noise floor"),
         ("Q", "last verdict"), ("E", "clear hits")),
        (("P", "camera pane"), ("S", "stats"), ("B", "status band"),
         ("L", "signal meter"), ("V", "aim the camera"), ("H", "this legend"),
         ("+ -", "volume")),
        (("D", "dark"), ("W", "white"), ("R", "red"), ("Y", "grey"),
         ("J", "move pane"), ("[ ]", "size")),
        (("Esc", "closes the app"), ("F11", "closes the app")),
    )

    GAP = "   "            # between entries on a row

    def __init__(self, lines: tuple[str, ...] = ()) -> None:
        self._rows = self.ROWS
        self.lines = lines or tuple(
            self.GAP.join(f"{k} {t}".strip() for k, t in row)
            for row in self._rows)

    def on_key(self, key: int) -> bool:
        if key == Qt.Key_H:
            self.visible = not self.visible
            return True
        return False

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if ctx.calibrating:
            return          # every key in this legend is inert mid-calibration
        # THE SAME GREY AS HIS ARTWORK, which is also the safe one: the
        # ceiling is the brightness every target is brought down to, so the
        # legend cannot be the brightest thing on the screen the camera
        # watches. At red excess 0 the detector cannot see it at all.
        lvl = int(_ceiling.figure_level())
        ink = (QColor(lvl, lvl, lvl) if ctx.dark_background
               else QColor(0, 0, 0, 150))
        p.setPen(QPen(ink))
        base = mono(px(ctx, 10))           # small, as asked
        bold = mono(px(ctx, 10))
        bold.setBold(True)
        step = px(ctx, 17)
        y = ctx.rect.y() + ctx.height - step - step * len(self._rows)
        for row in self._rows:
            x = ctx.rect.x() + 48
            for k, text in row:
                if k.strip():
                    p.setFont(bold)
                    p.drawText(int(x), int(y), k)
                    x += QFontMetricsF(bold).horizontalAdvance(k + " ")
                p.setFont(base)
                p.drawText(int(x), int(y), text)
                x += QFontMetricsF(base).horizontalAdvance(text + self.GAP)
            y += step


class Alert(Layer):
    """A message the shooter cannot miss, on the screen they are looking at.

    Everything diagnostic in this app used to land in the operator console's
    log -- which is behind a fullscreen target window, on the same monitor, and
    therefore invisible to the person holding the pistol. So a camera pointed at
    the wrong thing, a camera that had been knocked, or an abandoned calibration
    all presented identically: shots that simply did not register.

    Warnings are sticky, because a problem that needs the camera adjusting does
    not stop being true after five seconds. Confirmations time out, because
    telling someone something is fine is only worth a moment of their screen.
    """

    name = "alert"
    z = 70
    animated = False

    # BACKGROUND, TEXT, BORDER -- all from the measured artwork ceiling, none
    # typed. He photographed this panel on 2026-10-18 and said: "You should
    # not be outputting messages like that in that bright color. The colors
    # need to be muted so that they don't trigger shots."
    #
    # He was right and the numbers are not close. The text was `#f0f0f0`:
    # luminance 240 against a ceiling of 71, on a panel covering about 7% of
    # the screen, with a 2% budget above luminance 60 (B1b). Red excess was
    # already 0, which is why every colour check passed -- B1 is about the
    # detector mistaking something FOR a laser, and this is the other hazard
    # (B10): a dot landing on lit grey loses its red excess, because red
    # saturates while green and blue are already high. A panel that bright is
    # a hole in the screen for any shot that lands on it.
    #
    # Reading them from `grey_at` means they follow the darkest artwork he
    # ever adds, instead of being a number somebody picked in 2026.
    KINDS = {
        "warn": (grey_at(0.22), GREY_TEXT, GREY_MISS),
        "info": (grey_at(0.18), GREY_MID, GREY_MISS),
        "good": (grey_at(0.18), GREY_HI, GREY_MISS),
    }

    # NOTHING SITS ON HIS SCREEN FOR EVER. Warnings used to be sticky, on the
    # reasoning that a problem does not stop being true after five seconds --
    # true, and it is still reported continuously in the status band, which is
    # one line of text rather than a lit panel across the middle of the screen
    # the camera is watching. "the message should go away", 2026-10-18.
    MAX_DWELL_S = 12.0

    def __init__(self) -> None:
        self.visible = False
        self.title = ""
        self.detail = ""
        self.kind = "warn"
        self.until = 0.0            # 0 = no explicit deadline; the cap applies
        # When it was first PAINTED, on the scene clock. Declared here and
        # assigned unconditionally (C8), never reached through a default.
        self._first_seen = None
        # Dimming the rest of the screen focuses attention, but not when the
        # message is "look at the camera view and aim it" -- then the thing
        # being dimmed is the thing being read.
        self.dim = True

    def show(self, title: str, detail: str = "", kind: str = "warn",
             seconds: float = 0.0, now: float = 0.0, dim: bool = True) -> None:
        self.title, self.detail, self.kind = title, detail, kind
        self.until = (now + seconds) if seconds > 0 else 0.0
        # THE CAP IS TIMED FROM WHEN IT IS FIRST SEEN, not from `now`. Several
        # callers do not pass a clock at all, so `now` is 0.0 -- and computing
        # the deadline from that put it ~277,000 seconds in the PAST against
        # `perf_counter`, which cleared the alert on its first tick. He would
        # never have seen a warning again. Caught by the check that asserts
        # the alert draws something: 0 lit samples.
        self._first_seen = None
        self.dim = dim
        self.visible = True

    def clear(self) -> None:
        self.visible = False
        self.title = self.detail = ""
        self._first_seen = None

    def advance(self, ctx: RenderContext) -> bool:
        if not self.visible:
            return False
        if self._first_seen is None:
            self._first_seen = ctx.now          # the clock the deadline uses
        if self.until and ctx.now >= self.until:
            self.clear()
            return True
        # NOTHING SITS ON HIS SCREEN FOR EVER, however it was raised. A panel
        # across the middle of the screen the camera watches is not the place
        # for a standing condition; the status band is, and it keeps saying it.
        if ctx.now - self._first_seen >= self.MAX_DWELL_S:
            self.clear()
            return True
        return False

    def on_key(self, key: int) -> bool:
        # Dismissable, but only the timed ones -- a sticky warning describes
        # something still wrong, and hiding it would just hide the problem.
        if key == Qt.Key_Escape and self.visible and self.until:
            self.clear()
            return True
        return False

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if not self.title:
            return
        bg, ink, edge = self.KINDS.get(self.kind, self.KINDS["warn"])
        p.setRenderHint(QPainter.Antialiasing)

        w = min(ctx.width * 0.8, ctx.width - 80)
        title_px = max(20, ctx.height // 30)
        detail_px = max(13, ctx.height // 58)
        pad = title_px
        lines = _wrap(self.detail, 64) if self.detail else []
        h = pad * 2 + title_px * 1.4 + len(lines) * detail_px * 1.5

        box = QRectF(ctx.rect.x() + (ctx.width - w) / 2,
                     ctx.rect.y() + ctx.height * 0.12, w, h)
        if self.dim:
            # Dim everything else: this is the only thing worth reading now.
            p.fillRect(ctx.rect, QColor(0, 0, 0, 120))
        p.setBrush(QColor(bg))
        p.setPen(QPen(QColor(edge), 3))
        p.drawRoundedRect(box, 14, 14)

        p.setPen(QColor(ink))
        p.setFont(mono(title_px, bold=True))
        p.drawText(QRectF(box.x() + pad, box.y() + pad, box.width() - 2 * pad,
                          title_px * 1.4),
                   Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setFont(mono(detail_px))
        y = box.y() + pad + title_px * 1.5
        for line in lines:
            p.drawText(QRectF(box.x() + pad, y, box.width() - 2 * pad, detail_px * 1.5),
                       Qt.AlignLeft | Qt.AlignVCenter, line)
            y += detail_px * 1.5


class VolumeReadout(Layer):
    """"VOLUME 20%" for a moment after the + or - key, then gone.

    Drawn on the screen the camera watches, so it obeys the same rules as
    everything else there: neutral grey (red excess exactly 0, RULES B1), at
    most the brightness ceiling (B10), text only and no filled panel (B1b),
    sized from the screen and not in pixels (B5), and it does not paint at all
    while the app is measuring its own screen (B2) -- a readout that appeared
    during a noise measurement would be part of the floor.

    It is up for HOLD_S and asks for a repaint only when it appears and when it
    goes: a running timer on a 4K surface is the fault this project has already
    found three times (a HUD at 60 Hz, the reaction counter).
    """

    name = "volume"
    z = 72                  # above the alert, below nothing that matters
    animated = False
    HOLD_S = 1.6

    def __init__(self) -> None:
        self.visible = False
        self.text = ""
        self.until = 0.0

    def show_volume(self, text: str, now: float) -> None:
        self.text = text
        self.until = now + self.HOLD_S
        self.visible = True

    def advance(self, ctx: RenderContext) -> bool:
        if self.visible and ctx.now >= self.until:
            self.visible = False
            self.text = ""
            return True
        return False

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if not self.text or ctx.calibrating or ctx.diagnostics.get("measuring"):
            return
        p.setFont(mono(px(ctx, 30), bold=True))
        p.setPen(QPen(GREY_TEXT))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(self.text)
        x = ctx.rect.x() + (ctx.width - w) / 2
        y = ctx.rect.y() + ctx.height * 0.90
        p.drawText(QPointF(x, y), self.text)


def _wrap(text: str, width: int) -> list[str]:
    """Greedy wrap. Explicit newlines are respected as paragraph breaks."""
    out: list[str] = []
    for para in text.splitlines() or [""]:
        line = ""
        for word in para.split():
            if line and len(line) + 1 + len(word) > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return out


class BackgroundLabel(Layer):
    """Tiny caption naming the current background, for the record during tuning."""

    name = "bglabel"
    z = -11                # furniture: never over a target

    def __init__(self, background: SolidBackground) -> None:
        self.background = background

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        ink = QColor("#8a8a8a") if ctx.dark_background else QColor(0, 0, 0, 150)
        p.setPen(QPen(ink))
        p.setFont(mono(px(ctx, 12)))
        label = getattr(self.background, "label", "")
        p.drawText(ctx.rect.x() + 48,
                   ctx.rect.y() + ctx.height - px(ctx, 20) - px(ctx, 60),
                   f"background: {label}")


class PlayAreaBorder(Layer):
    """A grey outline around the part of the screen shots are detected on.

    Asked for in these words, 2026-10-16:

        "advisory but clearly visible to the user, so please use a grey line
         to outline the play area inside the monitor; and that means no
         targets (or bouncing balls, etc.) go outside of that area. However
         if the shots detected on screen outside, draw the holes outside,
         just do not draw any targets"

    GREY IS NOT A STYLE CHOICE. `red - max(green, blue)` is literally the
    detector's score, so a grey line scores ZERO and the camera cannot read
    it as a hit however long it stays on screen -- which matters more here
    than anywhere else, because this line is up during every shot he fires.
    The project has paid for this three times: a console banner at 80, the
    arcade ball at 24 (272 detections in two minutes), and the signal meter
    at 24, which fired the gunshot at him five times while he stood still.

    It is also nearly free against RULES B1b, the lit-area budget: an
    outline a few pixels wide is a rounding error on a 4K panel, where the
    filled verdict panel it replaces was not.

    Drawn only when a border has actually been MEASURED. Before that there
    is nothing honest to draw, and a box around the whole screen would read
    as a measurement that had been made (RULES C4, C7).
    """

    name = "playarea"
    z = 35            # over the background and the holes, under the HUD
    visible = True

    # Live only while the measurement changes, which is a handful of times
    # in a session. Nothing here animates, so it must never ask for a
    # repaint of a 4K surface: a running counter in one mode cost 8.53ms a
    # frame against a detector that needs 6.96ms of the same CPU.
    animated = False

    def __init__(self) -> None:
        self._last: tuple | None = None

    def advance(self, ctx: RenderContext) -> bool:
        area = getattr(ctx, "play_area", None)
        now = None if area is None else (area.x0, area.y0, area.x1, area.y1)
        if now != self._last:
            self._last = now
            return True
        return False

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        area = getattr(ctx, "play_area", None)
        if area is None or getattr(area, "whole_screen", False):
            return
        r = ctx.rect
        x0 = r.x() + area.x0 * r.width()
        y0 = r.y() + area.y0 * r.height()
        x1 = r.x() + area.x1 * r.width()
        y1 = r.y() + area.y1 * r.height()
        # Sized from the screen, never in pixels: this has to read the same
        # on a 42-inch monitor, a 75-inch TV and a projector (RULES B5).
        w = px(ctx, 3, floor=2)
        p.save()
        pen = QPen(INK_DARK)
        pen.setWidth(w)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        # And say what the line IS, once, small, in the same grey. An
        # unexplained box on the screen he is shooting at is a thing to
        # wonder about; this project has already put one black cover and one
        # unlabelled camera pane in front of him.
        f = mono(px(ctx, 13))
        p.setFont(f)
        p.setPen(INK_DARK)
        label = "detection reliable inside this border"
        fm = p.fontMetrics()
        p.drawText(int(x0 + w + px(ctx, 6)),
                   int(y0 + w + px(ctx, 6) + fm.ascent()), label)
        p.restore()


class Prompt(Layer):
    """One line across the top saying what to do RIGHT NOW.

    The complaint this exists for, in the user's words: "I don't want to be in
    the situation when I shoot the targets and absolutely nothing happens and I
    have no idea if I should continue shooting at that target or move to the
    next one or stop shooting altogether."

    Everything the app knew about its own state used to live in the operator
    console -- which sits behind a fullscreen window on the same monitor -- or
    in a modal alert that only appeared when something was already wrong. So
    the normal case, "the app is fine and is waiting for you", looked exactly
    like the broken case. This band is always populated, and says which.

    It is deliberately NOT the alert layer. An alert interrupts and must be
    read; this is a standing status line, glanceable from the firing position
    and never in the way.
    """

    name = "prompt"
    # Above the calibration overlay (which dims the screen) and below the alert
    # box, so the instruction stays legible during calibration -- that is when
    # it matters most -- while a real alert still takes precedence.
    z = 62
    animated = False

    KINDS = {
        # background, ink, edge -- every one within the red-excess rule
        "go": ("#17331c", "#7fe08a", "#2e6b38"),      # shoot
        "wait": ("#14202a", "#8fc4e0", "#2d4c5e"),    # do not shoot yet
        "warn": ("#323232", "#f0f0f0", "#a0a0a0"),    # something needs fixing
        "idle": ("#141414", "#c8c8c8", "#303030"),    # nothing to say
    }

    def __init__(self, visible: bool = True) -> None:
        self.visible = visible
        self.headline = "STARTING UP"
        self.detail = ""
        self.kind = "wait"
        self.counter = ""        # right-aligned, e.g. "3 of 9" or "7 targets left"

    def say(self, headline: str, detail: str = "", kind: str = "idle",
            counter: str = "") -> bool:
        """Returns True if anything changed, so the caller can skip a repaint."""
        new = (headline, detail, kind, counter)
        old = (self.headline, self.detail, self.kind, self.counter)
        self.headline, self.detail, self.kind, self.counter = new
        return new != old

    def on_key(self, key: int) -> bool:
        if key == Qt.Key_B:
            self.visible = not self.visible
            return True
        return False

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        bg, ink, edge = self.KINDS.get(self.kind, self.KINDS["idle"])
        h = top_band_height(ctx)
        band = QRect(ctx.rect.x(), ctx.rect.y(), ctx.width, h)
        p.fillRect(band, QColor(bg))
        p.setPen(QPen(QColor(edge), 2))
        p.drawLine(band.x(), band.bottom(), band.right(), band.bottom())

        pad = 48
        title_px = max(17, h // 4)
        p.setPen(QColor(ink))
        p.setFont(mono(title_px, bold=True))
        # The counter is drawn first so the headline can be clipped against it
        # rather than overlapping it on a narrow screen.
        right = band.right() - pad
        if self.counter:
            p.setFont(mono(max(15, h // 5), bold=True))
            w = p.fontMetrics().horizontalAdvance(self.counter)
            p.drawText(QRectF(right - w, band.y(), w, h),
                       Qt.AlignRight | Qt.AlignVCenter, self.counter)
            right -= w + 40

        width = right - (band.x() + pad)
        if self.detail:
            p.setFont(mono(title_px, bold=True))
            p.drawText(QRectF(band.x() + pad, band.y() + h * 0.12, width, h * 0.46),
                       Qt.AlignLeft | Qt.AlignVCenter, self.headline)
            p.setFont(mono(max(12, h // 7)))
            p.drawText(QRectF(band.x() + pad, band.y() + h * 0.55, width, h * 0.36),
                       Qt.AlignLeft | Qt.AlignVCenter, self.detail)
        else:
            p.drawText(QRectF(band.x() + pad, band.y(), width, h),
                       Qt.AlignLeft | Qt.AlignVCenter, self.headline)


class SignalMeter(Layer):
    """Live proof that the camera can or cannot see the laser.

    When a shot does not register there are three possibilities and they need
    opposite responses: the camera saw nothing (aim it, or the laser is not
    firing), it saw something too faint to pass the threshold (dim the room,
    move closer, lower the threshold), or it saw it somewhere the app is not
    looking (the camera has moved and the mapping is stale). The app has always
    had the numbers to tell these apart and has never shown them to the person
    holding the pistol.

    The bar holds its peak for a moment deliberately. A LaserLyte pulse lasts
    about 100 ms and this is repainted at roughly 10 Hz, so an instantaneous
    reading would miss most shots entirely -- the meter would sit flat while the
    shooter watched it, which is precisely the false reassurance it exists to
    prevent.
    """

    name = "signal"
    # Above the calibration overlay, which dims everything beneath it. During a
    # stalled calibration this is the panel that says whether anything is being
    # seen at all, so being legible then is the whole point of it.
    z = 63
    animated = True
    # Ten times a second, which is the rate this meter was DESIGNED
    # for: it holds a peak for 1.1s precisely so a 100ms pulse
    # survives until the next repaint. It was asking for 60.
    read_rate_hz = 10.0

    HOLD_S = 1.1              # how long a peak stays up before it starts falling
    DECAY_PER_S = 160.0       # how fast it falls afterwards

    def __init__(self, visible: bool = True) -> None:
        self.visible = visible
        self.live = False                 # is the camera streaming at all
        self.peak = 0                     # rolling 1 s max of the whole frame
        self.threshold = 0
        self.hold = 0.0
        self.hold_at = 0.0
        self.seen_ago: float | None = None    # seconds since any dot was seen
        self.shot_ago: float | None = None    # seconds since a shot was accepted
        self.counts: tuple[tuple[str, object], ...] = ()
        self.note = ""
        self.note_kind = "info"

    def report(self, peak: int, threshold: int, *, live: bool = True,
               seen_ago: float | None = None, shot_ago: float | None = None,
               counts: tuple = (), note: str = "", note_kind: str = "info",
               now: float = 0.0) -> None:
        self.live = live
        self.peak = int(peak)
        self.threshold = int(threshold)
        self.seen_ago, self.shot_ago = seen_ago, shot_ago
        self.counts = tuple(counts)
        self.note, self.note_kind = note, note_kind
        if self.peak >= self.hold:
            self.hold = float(self.peak)
            self.hold_at = now

    def on_key(self, key: int) -> bool:
        if key == Qt.Key_L:
            self.visible = not self.visible
            return True
        return False

    def advance(self, ctx: RenderContext) -> bool:
        # The decay still runs on every tick -- it is arithmetic, and making
        # it coarse would make the bar step rather than fall. What is throttled
        # is how often that is allowed to trigger a 4K REPAINT, which is the
        # expensive part and which no eye resolves past about 10 Hz.
        if self.hold > self.peak and ctx.now - self.hold_at > self.HOLD_S:
            self.hold = max(float(self.peak),
                            self.hold - self.DECAY_PER_S * max(0.0, ctx.dt))
        return Layer.advance(self, ctx)

    def _verdict(self) -> tuple[str, QColor]:
        """The one sentence worth reading, chosen from what was measured."""
        if not self.live:
            return "camera not streaming", ACCENT
        if self.note:
            return self.note, (GOOD if self.note_kind == "good"
                               else COOL if self.note_kind == "info" else ACCENT)
        if self.seen_ago is None:
            return "nothing seen yet - fire a shot at the screen", COOL
        if self.seen_ago > 3.0:
            return f"nothing seen for {self.seen_ago:.0f}s", COOL
        return "seeing the laser", GOOD

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        w = int(min(ctx.width * 0.34, max(420, ctx.width * 0.30)))
        pad = 16
        line = max(13, ctx.height // 76)
        # Seven rows of content plus room for the verdict to wrap onto a second
        # line, which the most useful messages do -- they name a cause and a fix.
        h = pad * 2 + line * 8.2
        # Centred along the bottom, deliberately. The camera pane lives in a
        # corner and can be moved to any of the four, and the key legend sits
        # bottom-left, so a corner here would collide with one of them
        # depending on which keys the user had pressed. The centre collides
        # with none of them, and mirrors the instruction band at the top.
        if ctx.calibrating:
            # Nothing at all. Moving this panel around to dodge the grid was a
            # losing game: sent to the bottom centre it covered ring 8, sent to
            # the top right it covered ring 3, and it was reported both times.
            # The grid owns the screen while it runs, and the calibration block
            # already carries everything this panel would say -- the last
            # shot's position and peak, how many agree, the stall advice, and
            # now the live signal level too. One place for it, no ring beneath.
            return
        else:
            strip = max(BOTTOM_STRIP, px(ctx, BOTTOM_STRIP))
            box = QRectF(ctx.rect.x() + (ctx.width - w) / 2,
                         ctx.rect.y() + ctx.height - h - strip, w, h)

        p.setRenderHint(QPainter.Antialiasing)
        # Nearly opaque during calibration, where it sits over a grid ring and
        # legibility is everything; translucent in play, where a mode's ball may
        # pass behind it and being able to see it matters more.
        p.setBrush(QColor(12, 16, 20, 238 if ctx.calibrating else 200))
        p.setPen(QPen(QColor("#2d4c5e"), 1))
        p.drawRoundedRect(box, 8, 8)

        x = box.x() + pad
        inner = box.width() - 2 * pad
        y = box.y() + pad + line

        p.setFont(mono(max(11, line - 2), bold=True))
        p.setPen(QPen(COOL))
        p.drawText(QPointF(x, y), "LASER SIGNAL")
        p.setFont(mono(max(11, line - 2)))
        p.drawText(QRectF(x, y - line, inner, line * 1.2),
                   Qt.AlignRight | Qt.AlignVCenter,
                   f"{int(self.hold):3d} / need {self.threshold}")

        # The bar. Linear in score, 0-255, with the threshold marked: the
        # question it answers is "how close was that to counting", and a log
        # scale would flatter a signal that is in fact nowhere near.
        y += line * 0.9
        bar = QRectF(x, y, inner, line * 1.5)
        p.setBrush(QColor(0, 0, 0, 120))
        p.setPen(QPen(QColor("#2d4c5e"), 1))
        p.drawRect(bar)
        frac = max(0.0, min(1.0, self.hold / 255.0))
        passing = self.threshold and self.hold >= self.threshold
        fill = GOOD if passing else (ACCENT if self.hold > 0 else COOL)
        if frac > 0:
            p.setBrush(fill)
            p.setPen(Qt.NoPen)
            p.drawRect(QRectF(bar.x() + 1, bar.y() + 1,
                              max(2.0, (bar.width() - 2) * frac), bar.height() - 2))
        if self.threshold:
            tx = bar.x() + bar.width() * min(1.0, self.threshold / 255.0)
            p.setPen(QPen(QColor("#e6e6e6"), 2))
            p.drawLine(QPointF(tx, bar.y() - 3), QPointF(tx, bar.bottom() + 3))
        p.setBrush(Qt.NoBrush)

        y = bar.bottom() + line * 1.4
        p.setFont(mono(max(10, line - 3)))
        p.setPen(QPen(QColor("#b4b4b4")))
        ago = ("never" if self.seen_ago is None else f"{self.seen_ago:4.1f}s ago")
        shot = ("never" if self.shot_ago is None else f"{self.shot_ago:4.1f}s ago")
        p.drawText(QPointF(x, y), f"dot seen {ago}    shot {shot}")
        y += line * 1.25
        if self.counts:
            p.drawText(QPointF(x, y),
                       "   ".join(f"{k} {v}" for k, v in self.counts))
            y += line * 1.25

        text, col = self._verdict()
        p.setFont(mono(max(11, line - 2), bold=True))
        p.setPen(QPen(col))
        # Measured from the font, not guessed from the point size. The guess was
        # 20% optimistic and the most useful messages -- the ones that name a
        # cause and a fix, and are therefore the longest -- ran off the panel.
        ch = max(1, p.fontMetrics().horizontalAdvance("M"))
        for row in _wrap(text, max(16, int(inner / ch)))[:2]:
            p.drawText(QPointF(x, y), row)
            y += line * 1.2


class SyntheticLaser(Layer):
    """A red dot the SCREEN draws, so the camera can be tested without a gun.

    The shooter's idea, and it is a better test than anything here before it:
    "you should just flash small red dots on screen and see how they are being
    caught by camera... then you see how the rest of the machinery catches
    those red dots and where the holes appear."

    It works because of the rule it deliberately breaks. Nothing else on the
    target screen may carry a red excess above 30 -- the camera is watching
    that screen and red minus max(green, blue) is exactly what the detector
    scores -- which is another way of saying the screen is perfectly capable of
    counterfeiting a laser hit. Every other layer avoids that; this one exists
    to do it on purpose, at a position known to the pixel.

    What that buys: the whole chain after the dot can be measured without
    anyone pulling a trigger. The chroma detector, the threshold, the
    multi-frame tracker, the camera-to-screen mapping and the hole placement
    all run exactly as they do in play, and the answer can be checked against
    the position the dot was drawn at. The pattern calibration cannot do this:
    its markers are WHITE, chroma zero, so they never exercise the detector at
    all -- which is precisely how a sensitivity map built from them came to be
    uniform while the laser varied two-fold.

    What it does NOT prove: a screen EMITS this dot while a laser REFLECTS off
    the panel, so brightness, angular spread and spectrum all differ. Detection
    RATE here is not the gun's detection rate. Geometry, plumbing and accuracy
    are the same for both, and those are where every fault so far has lived.
    """

    name = "synthetic"
    z = 90                     # above everything; nothing may occlude the dot
    animated = False

    # THE FLICKER, AND WHY IT IS NOT OPTIONAL.
    #
    # A LaserLyte's diode is PWM-driven, so a camera sampling it catches a
    # different slice of its on-time every frame. The same gun at the same
    # distance therefore produces wildly different brightnesses, which is
    # what the shooter sees: 'all of my shots are seen... it's just that
    # some of them are seen much more clearly than the other ones'.
    #
    # Measured over 575 of his logged trigger pulls:
    #
    #     peak   min 31   p10 41   median 85   p90 180   max 253
    #     area   p10 113  median 296  p90 746 px
    #
    # p90/p10 is 4.4x. That spread IS the problem: the bright half clears
    # any sane threshold and the dim half does not, so a self test that
    # fires only bright dots reports success while the shooter loses a
    # third of his string. The old one did exactly that -- 12 of 12 dots,
    # every time, while he was missing one shot in three.
    #
    # So the emulated shot draws its brightness from his measured
    # distribution. The scale is relative to the brightest dot the screen
    # can draw; the self check calibrates what that is worth in camera
    # peak, then fires the whole spread.
    # Each rung is one DECILE of his measured distribution, divided by his
    # median -- so rung i emulates the i-th decile of his real shots and
    # the detection rate over the ten is directly "what fraction of his
    # string would register".
    #
    # Why the median and not the maximum. Measured 2026-10-15 at exposure
    # -4, the brightest dot this screen can draw images at a chroma peak
    # of about 100, and his median real shot measured 85 at -5. Those are
    # the same order, so anchoring full red at his MEDIAN is a measurement
    # rather than a hope. It also means the top four rungs clamp at 1.00:
    # the screen cannot emulate his bright half, so those shots are
    # represented by the brightest thing available.
    #
    # That clamping makes the measured rate a LOWER BOUND on his, which is
    # the right direction to be wrong in -- but it must be SAID, not
    # discovered. The previous spread ran from 0.22 and reported 8 of 12
    # with the verdict claiming "a human shooting now would lose about the
    # same fraction", while his own p10 shot would have cleared the bar
    # comfortably. An app that blocks a working setup wastes exactly as
    # much of his time as one that passes a broken one.
    BRIGHTNESS_SPREAD = (0.39, 0.48, 0.60, 0.74, 0.87, 1.00, 1.00, 1.00,
                         1.00, 1.00)

    # How many of the rungs above are clamped at full red, i.e. how much of
    # his distribution the screen cannot reach. Reported, never hidden.
    CLAMPED_RUNGS = 4

    def __init__(self) -> None:
        self.visible = False
        self.point: tuple[float, float] | None = None
        self.radius = 40.0     # screen px; set from the camera's view of the screen
        self.label = ""
        # 1.0 is the brightest red a screen can show. Lower values emulate
        # a frame that caught less of the pulse.
        self.brightness = 1.0
        # The SAME dot (radius, brightness) drawn at further places in the
        # same frame. The exposure probe measures the weak and the strong end
        # of the panel in ONE photograph this way, instead of two.
        self.also: list[tuple[float, float]] = []

    def show_at(self, point: "tuple[float, float] | None",
                radius: float | None = None,
                brightness: float | None = None,
                also: "list[tuple[float, float]] | None" = None) -> None:
        self.point = point
        self.also = list(also or [])
        if radius is not None:
            self.radius = max(2.0, float(radius))
        if brightness is not None:
            self.brightness = max(0.05, min(1.0, float(brightness)))
        self.visible = True

    def stop(self) -> None:
        self.visible = False
        self.point = None
        self.also = []

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if self.point is None:
            return
        for extra in self.also:
            self._dot(p, extra)
        self._dot(p, self.point)

    def _dot(self, p: QPainter, at: "tuple[float, float]") -> None:
        x, y = at
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        # Red on black: red excess is the whole signal, and the chroma
        # detector scores exactly red minus max(green, blue). Scaling the
        # red channel scales that score almost linearly, which is how a
        # dim flicker of a real pulse is emulated -- see BRIGHTNESS_SPREAD.
        level = int(round(255 * self.brightness))
        p.setBrush(QColor(level, 0, 0))
        p.drawEllipse(QPointF(x, y), self.radius, self.radius)

class VerdictBoard(Layer):
    """The app's answer to the only question the shooter is asking.

    Asked for in these words: "when a human finally gets to shoot, he will
    see either a working app or the app giving him the exact reason why it is
    not working and what needs to be done to make it work. For example:
    lower the light in the room, or disable or enable some settings in the
    camera."

    So this is not another diagnostic panel. Every number this project
    measures -- noise floor, signal margin, flash difference, held-out error
    -- is a property of ONE stage of a thirteen-stage chain, and every one of
    them has looked healthy while trigger pulls went undetected. This layer
    shows one thing first: of N shots the app fired at ITSELF, how many came
    back. Then, at most two actions, in the order worth doing them.

    Three deliberate choices:

    - **At most two findings on screen**, however many were found. A person
      standing at the firing line with a pistol can act on one thing. The
      log keeps all of them.
    - **READY times out, a problem does not.** Telling someone things are
      fine is worth four seconds of their screen; telling them the camera
      cannot see the panel stays up until it is fixed and re-measured.
    - **No red, including for the bad verdict.** The camera is watching this
      screen and scores red minus max(green, blue): a red warning banner is
      a warning banner that fires the gunshot sound. Urgency is carried by a
      thick border and the size of the word, not by hue.
    """

    name = "verdict"
    z = 68                      # under `alert`, which interrupts
    animated = False

    # background, ink, border, heading -- every one inside the red-excess
    # rule. #d8c070 scores 24, #7fe08a is negative, #c8c070 scores 24.
    # Every colour here keeps red - max(green, blue) at or below 8, not
    # merely under the house limit of 30.
    #
    # The house limit was written for THIN TEXT, and this is a large filled
    # panel that appears all at once. Measured 2026-10-15: at red excess 24
    # the board set the measured noise floor to 16-18 where the empty
    # screen reads 0-2, which pushed min_peak to 22-24 -- above the self
    # test's own dots. The next self test then detected 1 of 12, produced a
    # worse verdict, and that verdict stayed on screen: a feedback loop in
    # which the app's report of a problem was the problem.
    #
    # It also registered as two phantom SHOTS at peak 18-20 the moment it
    # appeared, because the background model has not absorbed something
    # that is one frame old. Any panel that appears mid-session has to be
    # invisible to the detector on its FIRST frame, not its fiftieth.
    STYLES = {
        "ready":    ("#142a18", "#b8d8b8", "#3e7a46", "#7fe08a"),
        "degraded": ("#2a2a2a", "#d6d6d6", "#828282", "#d8d8d8"),
        "blocked":  ("#2a2a2a", "#e0e0e0", "#c0c0c0", "#ececec"),
    }
    HEADINGS = {
        "ready": "READY",
        "degraded": "WORKS, BUT",
        "blocked": "NOT READY",
    }
    READY_SECONDS = 4.0
    MAX_FINDINGS = 2

    def __init__(self) -> None:
        self.visible = False
        self.verdict = "ready"
        self.summary = ""
        # (headline, action, measured) triples, already trimmed.
        self.findings: list[tuple[str, str, str]] = []
        self.more = 0           # findings not shown, counted honestly
        self.until = 0.0        # 0 = sticky

    def show_report(self, report, now: float = 0.0) -> None:
        """Display a `lsa.selfcheck.Report`.

        Takes the report rather than strings so there is one place where a
        verdict becomes pixels, and `selfcheck` stays free of Qt.
        """
        self.verdict = str(getattr(report.verdict, "value", report.verdict))
        self.summary = report.summary
        found = list(report.findings)
        self.findings = [(f.headline, f.action, f.measured)
                         for f in found[:self.MAX_FINDINGS]]
        self.more = max(0, len(found) - self.MAX_FINDINGS)
        self.until = (now + self.READY_SECONDS) if self.verdict == "ready" else 0.0
        self.visible = True

    def clear(self) -> None:
        self.visible = False
        self.findings = []
        self.summary = ""
        self.more = 0

    def advance(self, ctx: RenderContext) -> bool:
        if self.visible and self.until and ctx.now >= self.until:
            self.clear()
            return True
        return False

    def on_key(self, key: int) -> bool:
        """ANY key puts it away. Q brings it back.

        It used to be sticky for a problem verdict, on the reasoning that a
        verdict saying the camera cannot see the screen must not be
        dismissable into silence. That reasoning protects the app's right
        to be read and ignores whose screen it is.

        2026-10-16, while he was shooting: "the bright message does not
        disappear if I press X so you are degrading the codebase again",
        after "this bright message on screen and you do not draw in any
        bullet holes". He had read it. It was a lit panel across the middle
        of the screen the camera watches, over the place his holes were
        landing, and nothing he pressed would move it.

        It still cannot be missed -- it appears, it is bright, and it says
        what to do. What it may not do is refuse to leave. Any key, because
        the key he reaches for is whichever one the panel told him to
        press, and that one was being handled somewhere else entirely.
        """
        if not self.visible:
            return False
        if key == Qt.Key_Q:
            # Q IS THE TOGGLE, and it must not be dismissed here.
            #
            # Layers get the key first, so clearing the board on Q meant the
            # key then reached `_on_target_key`, which found the board hidden
            # and showed it again -- Q became the one key that could not put
            # it away. Exactly the complaint it was built from ("the bright
            # message does not disappear if I press X"), in a new place,
            # inside the fix for it. Found by an outside review; the test
            # could not see it because it called the app's handler directly
            # and bypassed the layer.
            return False
        self.clear()
        return False      # not consumed: X must still re-measure, 1 still
                          # starts a mode. Dismissing is not instead of
                          # doing what he asked for.

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if not self.summary and not self.findings:
            return
        if ctx.calibrating or ctx.diagnostics.get("measuring"):
            # THE APP IS MEASURING THIS SCREEN. Get out of the way.
            #
            # Not merely obscuring: this is a large filled panel and the
            # camera is pointed at it, so it changes the brightness the
            # measurement is reading and sits over the region the test
            # dots are drawn in. It corrupts the thing it is reporting on.
            #
            # And it is dishonest besides. "READY" while the self test is
            # still flashing dots announces a conclusion the app has not
            # finished reaching. His words, watching it: "You are
            # displaying this bright message while still trying to
            # calibrate, with flashing red dots on the screen."
            #
            # `calibrating` alone was not enough -- it is raised for the
            # noise floor and the grid, not for the self test or the
            # exposure probe, which are precisely the two that draw red
            # dots. Same mistake as the race guards: one instance fixed,
            # the rest left. `measuring` is set by whatever owns the
            # screen, so a routine added later is covered by default.
            return
        bg, ink, edge, head = self.STYLES.get(self.verdict,
                                              self.STYLES["degraded"])
        p.setRenderHint(QPainter.Antialiasing)

        head_px = px(ctx, 54)
        sum_px = px(ctx, 26)
        line_px = px(ctx, 21)
        act_px = px(ctx, 23)
        pad = px(ctx, 28)

        # Measure before drawing: a box sized by a guess either clips the
        # last action or leaves a slab of empty colour over the target.
        # A LINE LENGTH, not a screen fraction. Rendered at 86% of a 4K panel
        # this wrapped at about 160 characters, which is unreadable from the
        # firing line however large the glyphs are -- the eye loses the start
        # of the next line. Typography's comfortable range is 45-75; 62 here,
        # and the BOX is then sized to fit that, rather than the other way
        # round. Same mistake in a new place as sizing a font in pixels: it
        # read perfectly in a 1080p test window and not at all on the panel.
        #
        # The character width is MEASURED from the font, not assumed. A
        # guessed 0.62-of-the-point-size put the longest line hard against
        # the border in the rendered check, and a font that falls back to a
        # different family -- which `mono()` exists to allow -- would have
        # pushed it under. Measuring costs one call and cannot be wrong.
        wrap_at = 62
        # MEASURE EVERY FONT THAT DRAWS A FULL-WIDTH LINE, not just one.
        #
        # He saw this in a screenshot and I did not: the action text ran off
        # the right-hand edge of its own box. The box was sized from
        # `mono(line_px)` -- the body font, 21 at 1080p, regular weight --
        # while the action is drawn in `mono(act_px, bold=True)`, which is
        # 23 and bold. Both bigger. 62 characters of the second does not
        # fit in a box built for 62 of the first, by about 10%, and the
        # "DO THIS: " prefix is nine characters on top of that.
        #
        # So: the box is sized on the widest font, and the action gets its
        # OWN wrap derived from the space actually left for it. Measured
        # rather than guessed, for the same reason the comment above
        # already gives -- a guessed advance put the longest line hard
        # against the border once before, and `mono()` is allowed to fall
        # back to another family, which would push it under.
        char_w = max(
            QFontMetricsF(mono(line_px)).horizontalAdvance("M"),
            QFontMetricsF(mono(line_px, bold=True)).horizontalAdvance("M"),
        )
        act_w = QFontMetricsF(mono(act_px, bold=True)).horizontalAdvance("M")
        width = min(ctx.width * 0.86,
                    max(ctx.width * 0.5,
                        wrap_at * max(char_w, act_w) + pad * 2.4))
        # BOTH WRAPS COME FROM THE WIDTH THAT RESULTED, not from the width
        # that was asked for.
        #
        # `width` is capped at 86% of the screen. When that cap bites --
        # which it does on this panel -- the box is NARROWER than the 62
        # characters it was sized for, and a wrap computed from 62 then
        # overruns it. The check in test_ui_smoke caught exactly that after
        # the first fix: a body line 1736px wide in a 1595px box, so the
        # headline was clipping as well as the action, and I had only
        # looked at the action because that is the one he could see in his
        # screenshot.
        #
        # Deriving every wrap from `usable` makes the arithmetic closed: it
        # cannot matter which constraint won.
        usable = width - 2 * pad
        body_at = max(24, min(wrap_at, int(usable / max(1.0, char_w))))
        # Less the nine-character "DO THIS: " prefix, which is drawn as
        # part of the same line and in the same wider, bolder font.
        act_at = max(16, int(usable / max(1.0, act_w)) - 9)
        blocks = []
        for headline, action, measured in self.findings:
            hl = _wrap(headline, body_at)
            me = _wrap(measured, body_at) if measured else []
            ac = _wrap(action, act_at) if action else []
            blocks.append((hl, me, ac))

        h = pad * 2 + head_px * 1.25 + sum_px * 1.9
        for hl, me, ac in blocks:
            h += len(hl) * line_px * 1.45 + len(me) * line_px * 1.35
            h += len(ac) * act_px * 1.45 + line_px * 0.9
        if self.more:
            h += line_px * 1.6

        box = QRectF(ctx.rect.x() + (ctx.width - width) / 2,
                     ctx.rect.y() + max(top_band_height(ctx) + px(ctx, 12),
                                        (ctx.height - h) * 0.32),
                     width, h)
        # DIM ONLY FOR A PROBLEM, never for good news.
        #
        # This veil is 59% black over the WHOLE screen, which is right
        # when the message is "the camera cannot see your monitor" -- the
        # rest of the screen is then not worth looking at. It is wrong
        # for READY, which is a thing to glance at and shoot past.
        #
        # Combined with the clock bug that stopped this ever clearing, it
        # is why he reported no bullet holes: they were being drawn, at
        # 41% brightness, under a permanent veil. The ball in his
        # screenshot is dark brown where the palette says bright amber,
        # which is the same veil showing.
        #
        # Worth stating because I told him modes were hiding the holes.
        # They were not -- rendering proves every mode shows them -- and
        # I offered that before measuring it.
        if self.verdict != "ready":
            p.fillRect(ctx.rect, QColor(0, 0, 0, 150))
        p.setBrush(QColor(bg))
        # A thicker border for a worse verdict: this is how urgency is
        # carried, since the obvious colour for it is the one colour this
        # screen may never show.
        p.setPen(QPen(QColor(edge), px(ctx, 3 if self.verdict == "ready" else 7)))
        p.drawRoundedRect(box, px(ctx, 16), px(ctx, 16))

        x = box.x() + pad
        w = box.width() - 2 * pad
        y = box.y() + pad

        p.setPen(QColor(head))
        p.setFont(mono(head_px, bold=True))
        p.drawText(QRectF(x, y, w, head_px * 1.25),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   self.HEADINGS.get(self.verdict, "?"))
        y += head_px * 1.25

        p.setPen(QColor(ink))
        p.setFont(mono(sum_px))
        p.drawText(QRectF(x, y, w, sum_px * 1.7),
                   Qt.AlignLeft | Qt.AlignVCenter, self.summary)
        y += sum_px * 1.9

        for hl, me, ac in blocks:
            p.setPen(QColor(head))
            p.setFont(mono(line_px, bold=True))
            for line in hl:
                p.drawText(QRectF(x, y, w, line_px * 1.45),
                           Qt.AlignLeft | Qt.AlignVCenter, line)
                y += line_px * 1.45
            p.setPen(QColor(ink))
            p.setFont(mono(line_px))
            for line in me:
                p.drawText(QRectF(x + px(ctx, 14), y, w, line_px * 1.35),
                           Qt.AlignLeft | Qt.AlignVCenter, line)
                y += line_px * 1.35
            # The action is the point of the whole panel, so it is the
            # biggest thing after the verdict itself and it is labelled.
            p.setFont(mono(act_px, bold=True))
            p.setPen(QColor(head))
            for i, line in enumerate(ac):
                prefix = "DO THIS: " if i == 0 else "         "
                p.drawText(QRectF(x, y, w, act_px * 1.45),
                           Qt.AlignLeft | Qt.AlignVCenter, prefix + line)
                y += act_px * 1.45
            y += line_px * 0.9

        if self.more:
            p.setPen(QColor(ink))
            p.setFont(mono(line_px))
            p.drawText(QRectF(x, y, w, line_px * 1.6),
                       Qt.AlignLeft | Qt.AlignVCenter,
                       f"and {self.more} more in the log -- fix these first")
