"""One loaded figure: its grey artwork, its sprite, its scoring zones and its
size cap. Shared by the three figure drills so the geometry that is DRAWN and
the geometry that is SCORED come from one place.

Loading is lazy (never at import) and the grey arrays are cached for the life
of the process, because reading a 1792x2400 JPEG costs tens of milliseconds
and a figure change must not stall the UI thread in the middle of a drill; the
drills choose their next figure while the screen is blank and load it then.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen, QPixmap

from ..artwork import catalog, ceiling, load, zones as Z
from ..display.layers import GREY_HIT, GREY_MISS, GREY_TEXT, mono, px
from ..display.scene import Occluder
from ..watchdog import record_mark
from .sprite import Sprite

# The lit-area budget (RULES B1b: ~2% of the screen above luminance 60 while
# shooting is possible) as it applies to figures. One figure alone gets 1.5%:
# it is the only lit thing besides settled holes (80 of them measure ~0.6%).
# Figures ADD, so a drill showing two at once shares COMBINED_BUDGET.
SINGLE_BUDGET = 0.015
COMBINED_BUDGET = 0.015
# 1.0 = his artwork exactly as drawn. CAP the size, do not dim the ink: he
# answered "Yes, that's big enough" to a figure at ~60% of the screen height.
INK_SCALE = 1.0

# MEMORY, WHEN THE THIRTEENTH IMAGE ARRIVES. This cache keeps every grey array for
# the life of the process and never evicts: 12 figures at 1792x2400 are 49.2 MiB
# today, on top of up to 40 MiB of sprite pyramid (`sprite.MAX_PYRAMID_BYTES`) and
# the 1/8-scale silhouettes (~0.07 MiB each). Bounded now; it grows by ~4.1 MiB for
# every file he drops into `targets/`. If that ever matters, make this an LRU (only
# the figures a drill is about to use are needed) -- nothing depends on all of them
# being resident. Recorded here, deliberately, so it is found then and not before.
_GREY_CACHE: dict[str, np.ndarray] = {}
_SILHOUETTE_CACHE: dict[str, np.ndarray] = {}

# What a shooter is told about the head, everywhere he might look (the log when
# a figure mode is chosen, the mode's summary line, the standing hint on screen).
# The artwork PRINTS a "9" in the face and a "7" above the head dome, but draws
# no ring for either, and only drawn curves score: so inside the dome is 8 and
# above it is nothing. Without this a head shot that "should" be a 9 reads as the
# app ignoring him.
HEAD_NOTE = ("what scores is what is drawn: inside the head dome is 8 (its printed 9 "
             "has no ring), above the dome nothing; a dim ring marked 'no ring' "
             "means the shot hit the figure but no ring")
HINT_LINES = ("scores what is drawn: head dome = 8, its printed 9 and the 7 above it have no ring",
              "a dim ring marked 'no ring' = on the figure, outside every ring")
NO_RING = "no ring"
# At most this many feedback marks are up at once (two pistols is the most that
# can fire together); the lit-area budget is measured with this many.
MAX_MARKS = 2
# THE MOST HOLES ONE FIGURE CARRIES. A hole is a fraction of the figure, drawn on
# top of it every frame (a ring and a core, two primitives; nothing is baked into
# the sprite). 40 is more than a full approach with every shot landing (about 5
# trigger pulls a second per gun for ~13 s); the oldest is dropped past it. The
# lit area at the largest scale with this many is measured in test_figures.
MAX_CARRIED = 40
# A hole's radius, as a fraction of the figure's drawn height. The global
# bullet hole is 1/110 of the SCREEN height; a figure fills ~60% of it, so this
# is the same size on the figure at its nearest and shrinks with it.
HOLE_R_FRAC = 0.015
# THE FIGURE DROPS AT A RUNNING TOTAL. Each scoring hit adds its ring value (7, 8,
# 9, 10, unrounded) to the figure's own total; at DROP_TOTAL the figure goes at
# once. Three centre hits or five edge hits reach it, and that gradient is the
# point of the number. 30, not 21: three hits always total at least 21, so 21
# would have measured nothing. A shot on the figure but in no ring adds nothing.
DROP_TOTAL = 30
# The running counter ("24 / 30") beside the figure: its height as a fraction of
# the FIGURE's drawn height (so it travels and shrinks with it), never below
# COUNTER_MIN_1080 on a 1080p-equivalent screen so it stays legible from the
# firing line when the figure is at its farthest.
COUNTER_H_FRAC = 0.045
COUNTER_MIN_1080 = 20
# A FIGURE THAT DROPS FALLS FIRST. At DROP_TOTAL the figure's fall pose
# (`targets/fall/<name>.jpg`) is shown for FALL_S, then it goes. One second,
# his number. Not every figure has a pose; one without simply disappears at
# once, as before the poses existed.
FALL_S = 1.0


def _silhouette(path: str, grey: np.ndarray) -> np.ndarray:
    """A coarse mask (1/8 scale) of where the printed figure is: all its ink,
    thickened, closed and filled. A shot inside it is a shot AT the figure."""
    m = _SILHOUETTE_CACHE.get(path)
    if m is None:
        lvl = max(20, int(0.55 * load.ink_level(grey)))
        ink = cv2.dilate((grey > lvl).astype(np.uint8), np.ones((8, 8), np.uint8))[::8, ::8]
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        cs, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros_like(ink)
        cv2.drawContours(m, cs, -1, 1, -1)
        m.setflags(write=False)
        _SILHOUETTE_CACHE[path] = m
    return m
_CAP_CACHE: dict[tuple, float] = {}


def _grey(path: Path) -> np.ndarray:
    key = str(path)
    got = _GREY_CACHE.get(key)
    if got is None:
        got = load.load_grey(path)
        got.setflags(write=False)
        _GREY_CACHE[key] = got
    return got


class Figure:
    """A figure ready to draw and to score."""

    def __init__(self, name: str, screen_w: int, screen_h: int,
                 budget: float, folder: "Path | None" = None) -> None:
        self.name = name
        grey = _grey(catalog.path_for(name, folder))
        # Brought DOWN to the shared brightness ceiling (never up): a stroke
        # brighter than the darkest of his images hides the dot that lands on
        # it. Only the DRAWN pixels change; the size cap, the silhouette and the
        # scoring zones are all computed from the UNSCALED art, so geometry and
        # size are exactly what they were.
        self.gain = ceiling.gain_for_path(catalog.path_for(name, folder), folder)
        self.sprite = Sprite(grey, INK_SCALE * self.gain)
        self.aspect = grey.shape[1] / grey.shape[0]
        # Per image, because the figures differ 3.2x in lit area at one size.
        # Cached: it is a pure function of the picture, the screen and the
        # budget, and a drill picks a new figure every few seconds.
        key = (str(catalog.path_for(name, folder)), screen_w, screen_h, budget)
        cap = _CAP_CACHE.get(key)
        if cap is None:
            cap = _CAP_CACHE[key] = load.max_height_for_budget(grey, screen_w,
                                                               screen_h, budget)
        self.cap = cap
        # A figure that cannot be judged is never drawn: refuse it here, so a
        # mode that is handed one (a test, a hand-typed name) fails loudly
        # instead of drawing a target that ignores good shots.
        why = catalog.reason(name, folder)
        if why:
            raise ValueError(f"figure {name!r} cannot be scored and is never "
                             f"drawn: {why}")
        info = Z.load_zones(name, folder)
        self._sil = _silhouette(str(catalog.path_for(name, folder)), grey)
        # The opaque-silhouette mask (see `occluder`): built ONCE, at the
        # silhouette's own 1/8 resolution, and stretched by the painter.
        self._mask_pm: "QPixmap | None" = None
        # The fall pose, if there is one. Same loader as the standing art, so it
        # is forced to true grey by construction (red excess 0), whatever the
        # file carries. It must share the standing canvas: the pose is drawn
        # registered on the figure it replaces, and a different canvas would
        # make it jump. A pose on another canvas is ignored (the figure then
        # just disappears), not stretched into place.
        self.fall_sprite: "Sprite | None" = None
        self.fall_cap = 0.0
        fp = catalog.fall_path(name, folder)
        if fp is not None:
            fg = _grey(fp)
            if fg.shape == grey.shape:
                self.fall_gain = ceiling.gain_for_path(fp, folder)
                self.fall_sprite = Sprite(fg, INK_SCALE * self.fall_gain)
                fkey = (str(fp), screen_w, screen_h, budget)
                fcap = _CAP_CACHE.get(fkey)
                if fcap is None:
                    fcap = _CAP_CACHE[fkey] = load.max_height_for_budget(
                        fg, screen_w, screen_h, budget)
                # The pose has its OWN cap, from its OWN brightness: they are
                # wider than the standing art and differ in lit area.
                self.fall_cap = fcap
        self.stroke = float(info["stroke"])
        self.zones = Z.compile_zones(info["zones"], self.aspect)

    def fit_height(self, field_h: float, screen_h: int, frac: float) -> float:
        """Height in px for `frac` of this figure's cap, never taller than the
        playfield can hold."""
        return min(frac * self.cap * screen_h, 0.98 * field_h)

    @property
    def has_fall(self) -> bool:
        return self.fall_sprite is not None

    def fall_height(self, height_px: float, screen_h: int) -> float:
        """The height the fall pose is drawn at: the height the figure stood at,
        unless the pose's own lit-area cap is lower (then that), so the second
        it is up cannot blow the budget the standing figure was sized to."""
        return min(height_px, self.fall_cap * screen_h)

    def prepare_fall(self, height_px: float, screen_h: int) -> None:
        """Build the fall pose's level now, while the screen is blank or the
        figure is being set up, so the drop frame does not pay for a resize."""
        if self.fall_sprite is not None:
            fh = self.fall_height(height_px, screen_h)
            self.fall_sprite.set_range(fh)
            self.fall_sprite.warm(fh)

    def on_figure(self, x: float, y: float, centre: tuple[float, float],
                  height_px: float) -> bool:
        """Is the shot on the printed figure (its silhouette), whether or not it
        is inside a ring? 42-51% of each figure's ink lies outside every ring, so
        a shot there is common, and it must not look like nothing happened."""
        left, top, w, h = self.box(centre, height_px)
        if not (left <= x < left + w and top <= y < top + h):
            return False
        mh, mw = self._sil.shape
        return bool(self._sil[min(mh - 1, int((y - top) / h * mh)),
                              min(mw - 1, int((x - left) / w * mw))])

    def occluder(self, z: int, centre: tuple[float, float],
                 height_px: float) -> Occluder:
        """The figure as a SOLID shape, for the layers beneath it to leave out.

        The sprite is line art with a TRANSPARENT ground (alpha = brightness),
        so on its own the bullet holes underneath show through every dark part
        of it and the figure looks like it floats through them (his words,
        2026-10-17: "when they go over the existing shots, they should hide
        the existing shots"). The solid shape is the same silhouette that
        decides whether a shot is ON the figure (`on_figure`), at the size
        `paint` actually draws, so what hides a hole and what counts as a hit
        on the figure are one shape. Holes the figure itself carries (Marks)
        are drawn after the sprite and are not affected.
        """
        left, top, w, h = self.box(centre, height_px)

        def make() -> QPixmap:
            # Once per figure, NOT once per size: a walking figure changes
            # size almost every frame, and rebuilding a full-size mask each
            # time measured 22 ms at 1080p (about 90 at 4K) against a 14 ms
            # frame budget. The 1/8 silhouette is stretched by the painter.
            if self._mask_pm is None:
                mh, mw = self._sil.shape
                a8 = np.ascontiguousarray(self._sil.astype(np.uint8) * 255)
                bgra = np.ascontiguousarray(np.repeat(a8[..., None], 4, axis=2))
                img = QImage(bgra.data, mw, mh, mw * 4,
                             QImage.Format_ARGB32_Premultiplied).copy()
                self._mask_pm = QPixmap.fromImage(img)
            return self._mask_pm

        return Occluder(z, QRect(left, top, w, h), make)

    def box(self, centre: tuple[float, float], height_px: float):
        """(left, top, width, height) of what is actually drawn."""
        size = self.sprite.drawn_size(height_px)
        left, top = Sprite.top_left(centre, size)
        return left, top, size[0], size[1]

    def score(self, x: float, y: float, centre: tuple[float, float],
              height_px: float) -> "tuple[int, str | None]":
        """(points, zone) for a shot at screen (x, y); (0, None) if it is
        outside every ring or the figure has none."""
        left, top, w, h = self.box(centre, height_px)
        return Z.score((x - left) / w, (y - top) / h, self.zones,
                       self.aspect, self.stroke)

    def innermost(self, centre: tuple[float, float], height_px: float):
        """(zone name, centre x, centre y, radius px) of the highest-scoring
        zone, for `Mode.nearest`. None for an unscored figure."""
        z = max(self.zones, key=lambda q: q["points"])
        left, top, w, h = self.box(centre, height_px)
        cx, cy, r = Z.centre_and_radius(z, self.aspect)
        return z["name"], left + cx * w, top + cy * h, r * h


class Marks:
    """The shots ONE figure has absorbed, stored in the FIGURE's own coordinates.

    A shot that lands on a figure becomes part of it: (u, v) are fractions of
    the figure's drawn box (0..1 across, 0..1 down), never screen pixels, so
    the mark travels, grows and shrinks with the figure and is redrawn at its
    current position and scale every frame. The global bullet-hole layer is
    told not to draw that shot (`ShotOutcome.absorbed`), so there is exactly
    one mark per shot. When the figure leaves, its Marks is cleared and
    nothing is left on the screen.

    Drawn as vector primitives on top of the blitted sprite: the sprite
    pyramid and its pixmap are untouched, so a mark costs two small ellipses a
    frame and never a re-rasterise.
    """

    def __init__(self, cap: int = MAX_CARRIED) -> None:
        self.cap = cap
        self.holes: list[tuple[float, float, bool]] = []                # u, v, good
        self.flashes: list[tuple[float, float, float, str, bool]] = []  # u, v, t, text, good
        self.total = 0                  # the figure's running score; belongs to the figure

    def tally(self, points: int) -> int:
        """Add a scoring hit's ring value to this figure's total and return it.
        Called only for a hit inside a ring: a no-ring mark never advances it."""
        self.total += int(points)
        return self.total

    @property
    def done(self) -> bool:
        """The figure has reached DROP_TOTAL and is to leave at once."""
        return self.total >= DROP_TOTAL

    def __len__(self) -> int:
        return len(self.holes)

    def clear(self) -> None:
        self.holes = []
        self.flashes = []
        self.total = 0

    def add(self, fig: "Figure", x: float, y: float, centre, height_px: float,
            t: float, text: str, good: bool) -> None:
        """Absorb a shot at screen (x, y), judged against the figure as it stood
        at (`centre`, `height_px`) -- where it WAS when the shot was fired."""
        left, top, w, h = fig.box(centre, height_px)
        u, v = (x - left) / w, (y - top) / h
        self.holes = (self.holes + [(u, v, good)])[-self.cap:]
        self.flashes = (self.flashes + [(u, v, t, text, good)])[-MAX_MARKS:]
        record_mark()          # the watchdog's ledger: a mark WAS put on the figure

    def advance(self, now: float, flash_s: float) -> bool:
        """Prune finished flashes. True while one was up at the start of this
        call (so the tick that ends a flash still repaints); a figure holding
        still with only settled holes returns False and asks for nothing."""
        had = bool(self.flashes)
        self.flashes = [f for f in self.flashes if now - f[2] < flash_s]
        return had

    def screen_position(self, fig: "Figure", centre, height_px: float,
                        u: float, v: float) -> tuple[float, float]:
        left, top, w, h = fig.box(centre, height_px)
        return left + u * w, top + v * h

    def paint(self, p: QPainter, ctx, fig: "Figure", centre, height_px: float,
              flash_s: float) -> None:
        """Draw every carried hole and live flash at the figure's CURRENT box."""
        if not self.holes and not self.flashes and self.total <= 0:
            return
        left, top, w, h = fig.box(centre, height_px)
        r = max(ctx.height * 0.002, HOLE_R_FRAC * h)
        rim_w = max(ctx.height * 0.001, 0.3 * r)
        p.setRenderHint(QPainter.Antialiasing)
        core = QColor(8, 8, 8)
        for u, v, good in self.holes:
            c = QPointF(left + u * w, top + v * h)
            p.setPen(QPen(QColor(GREY_HIT if good else GREY_MISS), rim_w))
            p.setBrush(core)
            p.drawEllipse(c, r, r)
        paint_hits(p, ctx, [(left + u * w, top + v * h, t, text, good)
                            for u, v, t, text, good in self.flashes], flash_s)
        if self.total > 0:
            self._paint_counter(p, ctx, fig, centre, height_px)

    def counter_text(self) -> str:
        return f"{self.total} / {DROP_TOTAL}"

    def counter_geometry(self, ctx, fig: "Figure", centre, height_px: float):
        """(font, x, baseline y, text width, text height) of the counter beside the
        figure at its CURRENT box: a fraction of the figure's height with a floor
        (so it stays readable when the figure is small), right of the figure or,
        if that would run off the screen, left of it. Shared by `paint` and the
        tests, so what is measured is what is drawn."""
        left, top, w, h = fig.box(centre, height_px)
        size = max(px(ctx, COUNTER_MIN_1080), round(COUNTER_H_FRAC * h))
        font = mono(size, bold=True)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(self.counter_text())
        gap = 0.02 * h
        x = left + w + gap
        if x + tw > ctx.width - 4:
            x = left - gap - tw
        x = min(max(x, 4.0), ctx.width - tw - 4.0)
        y = min(max(top + 0.30 * h, float(fm.ascent()) + 4.0), ctx.height - 4.0)
        return font, x, y, tw, fm.height()

    def _paint_counter(self, p: QPainter, ctx, fig: "Figure", centre,
                       height_px: float) -> None:
        """`24 / 30` beside the figure, riding it. Neutral grey (red excess 0).
        Nothing here animates: it is redrawn only when something else repaints,
        and its text changes only when a scoring hit lands."""
        font, x, y, _tw, _th = self.counter_geometry(ctx, fig, centre, height_px)
        p.setFont(font)
        p.setPen(QPen(QColor(GREY_HIT)))
        p.drawText(int(x), int(y), self.counter_text())


class Fall:
    """A figure lying where it was shot, for FALL_S, drawn from its fall pose.

    Frozen at the position and height the figure had when the shot landed (a
    walking figure does not carry on walking). Registered on the standing art:
    same canvas, feet on the same line. Carries no marks and no counter: the
    fall replaces the figure and they go with it. The final total's flash is not
    ours; the mode keeps it at the screen spot.
    """

    def __init__(self, fig: Figure, centre, height_px: float, t0: float,
                 screen_h: int) -> None:
        self.fig = fig
        self.t0 = float(t0)
        self.t1 = self.t0 + FALL_S
        self.height = fig.fall_height(height_px, screen_h)
        # Keep the feet where they were if the pose had to be drawn smaller.
        self.centre = (centre[0], centre[1] + (height_px - self.height) / 2.0)

    @classmethod
    def start(cls, fig: Figure, centre, height_px: float, t0: float,
              screen_h: int) -> "Fall | None":
        """None for a figure with no pose: it just disappears."""
        if not fig.has_fall or FALL_S <= 0.0:       # 0 = falls switched off
            return None
        return cls(fig, centre, height_px, t0, screen_h)

    def covers(self, t: float) -> bool:
        """Is `t` inside the fall? Open at t0: the shot that dropped the figure
        was fired at a STANDING figure, so its own clock reads not-falling,
        before and after it is reported (Mode.moving_at must not change under
        the shot that caused it)."""
        return self.t0 < t < self.t1

    def paint(self, p: QPainter) -> None:
        self.fig.fall_sprite.paint(p, self.centre, self.height)


def paint_hint(p: QPainter, ctx, field, started_at: float, extra_lines=(),
               seconds: float = 8.0) -> None:
    """The standing rule for the first few seconds of a round, bottom-left of
    the playfield (the key legend sits below it). Permanent it would be clutter
    on a screen that is also being shot at."""
    if ctx.now - started_at >= seconds:
        return
    lines = list(extra_lines) + list(HINT_LINES)
    p.setPen(QPen(QColor(GREY_TEXT)))
    p.setFont(mono(px(ctx, 14)))
    step = px(ctx, 20)
    y = int(field.bottom()) - px(ctx, 16)
    for line in reversed(lines):
        p.drawText(int(field.left()) + 8, y, line)
        y -= step


def paint_hits(p: QPainter, ctx, flashes, flash_s: float) -> None:
    """The feedback for a shot: a grey ring where it landed and, beside it, what
    it was worth (`text`). `flashes` is a list of (x, y, t, text, good).

    A hit that changes nothing the shooter can see is the third costume of one
    fault (holes switched off, aim rings switched off, a hit with no feedback),
    so every figure mode draws this. Neutral grey, red excess 0, a few hundred
    pixels of lit area against a figure budget measured separately.
    """
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(Qt.NoBrush)
    for fx, fy, ft, text, good in flashes:
        k = (ctx.now - ft) / flash_s
        if k >= 1.0:
            continue
        k = max(0.0, k)
        c = QColor(GREY_HIT if good else GREY_MISS)
        c.setAlphaF(max(0.0, 1.0 - k))
        p.setPen(QPen(c, max(2.0, ctx.height * 0.004)))
        rr = ctx.height * 0.02 * (1.0 + k * 2.0)
        p.drawEllipse(QPointF(fx, fy), rr, rr)
        if text:
            p.setFont(mono(px(ctx, 26), bold=True))
            p.setPen(QPen(c))
            p.drawText(int(fx + rr + px(ctx, 10)), int(fy + px(ctx, 9)), text)


class PlanLink:
    """What the two halves of a TWO-PLAYER round share, for the figure drills.

    The k-th figure (or cycle) of one half is the SAME figure, drawn from the same
    plan, as the k-th of the other: different figures mean different ring
    geometry and different difficulty, and the fairness of a versus round rests on
    the two halves being the same drill. Plans are generated lazily, by whichever
    half needs index k first, from one seeded stream, and handed back unchanged to
    the other. Positions are stored as FRACTIONS of the field (the right half
    mirrors them across the divider); the artwork itself is never flipped, because
    its scoring rings are registered to the ink and a mirrored figure would be a
    second geometry to verify for no gain.
    """

    def __init__(self) -> None:
        import random
        self.rng = random.Random()
        self.plans: list = []

    def get(self, k: int, make):
        """Plan number `k`, made by `make(rng, previous_plan, k)` if it does not
        exist yet. Bounded: a half cannot ask for more than one ahead of the
        other without something being wrong, and a runaway is stopped."""
        while len(self.plans) <= k:
            if len(self.plans) > 100000:
                raise RuntimeError("PlanLink: runaway plan request")
            self.plans.append(make(self.rng, self.plans[-1] if self.plans else None,
                                   len(self.plans)))
        return self.plans[k]


def figure_names(folder: "Path | None" = None) -> list[str]:
    """The figures that may be drawn (verified scoring zones only)."""
    return catalog.names(folder)


def cast_top_points(folder: "Path | None" = None) -> "int | None":
    """The most one shot can score on a figure, if EVERY figure in the cast agrees.

    `Figure.score` returns the points of the highest zone containing the shot, so
    a figure's best shot is the largest `points` among its own zones -- read from
    the same sidecar data the scorer reads. The ghost aims at the top zone of
    whichever figure comes up; if the cast disagreed (one figure topping out at 9)
    there would be no single honest ceiling, and None is returned so the drill is
    NOT SCORABLE rather than quietly wrong. None for an empty cast as well.
    """
    tops = set()
    for name in figure_names(folder):
        zones = Z.load_zones(name, folder)["zones"]
        if not zones:
            return None
        tops.add(max(int(z["points"]) for z in zones))
    return tops.pop() if len(tops) == 1 else None


def figure_ghost_ceiling(folder: "Path | None" = None):
    """The ghost ceiling for a drill that scores figures by their rings."""
    from .. import ghost
    top = cast_top_points(folder)
    return None if top is None else ghost.constant_ceiling(top)


def shots_per_figure(folder: "Path | None" = None) -> "int | None":
    """How many perfect shots drop one figure: DROP_TOTAL over the best shot's
    value, rounded up (three tens reach 30). None when the cast has no single top."""
    top = cast_top_points(folder)
    return None if top is None else -(-DROP_TOTAL // top)


def expected_wait_after_drop(gap: "tuple[float, float] | None" = None,
                             folder: "Path | None" = None) -> "float | None":
    """The mean wait before the NEXT figure after one is dropped, averaged over
    the cast that can be drawn: the ghost's expected wait for a target the round
    never drew (`Mode.ghost_expected_wait`).

    A dropped figure shows its fall pose for FALL_S before it goes (`Fall.start`:
    none for a figure with no pose, or when falls are switched off), and the drills
    that replace a figure only after the fall is over wait exactly that. With a
    `gap` (lo, hi) the next figure also waits a random gap and never appears before
    the fall ends, as the walking figure does (`live_at = max(live_at, fall.t1)`):
    the wait is E[max(gap, fall)]. Everything is read from FALL_S, the catalogue's
    fall poses and the `gap` the CALLER passes from its own constants; nothing is
    typed here. None when no figure can be drawn."""
    from .. import ghost
    names = figure_names(folder)
    if not names:
        return None
    falls = [FALL_S if FALL_S > 0.0 and catalog.fall_path(n, folder) is not None else 0.0
             for n in names]
    if gap is None:
        return sum(falls) / len(falls)
    each = [ghost.expected_max_uniform(gap[0], gap[1], f) for f in falls]
    return None if any(e is None for e in each) else sum(each) / len(each)
