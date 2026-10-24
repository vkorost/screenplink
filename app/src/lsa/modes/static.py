"""Static targets: one large target in the middle, four smaller in the corners.

Owner, 2026-10-16: "lets have option #1 to have the static target in the
middle with 4 smaller targets in the corners." And, of the blank "free aim"
screen it replaces: "Is free aim just a blank screen without anything? If
that's the case there is absolutely no need to have that. It is useless to
me."

So this is what the app lands on after setup, and what key 1 returns to. It
never moves, never escalates and a hit does not make anything vanish: he
shoots, holes appear, that is the drill. Pressing 1 again restarts it (holes
and score cleared, same layout).

A HIT IS VISIBLE (plan 02 step 11). Owner: "shots land inside the targets and
nothing happens to the targets." In this mode nothing was ever built to
happen: the score lives in the stats HUD, and the HUD is hidden while
shooting, so a hit and a miss both looked like a hole. A hit now flashes a
brief expanding ring on the mark it struck, and each mark keeps a small tally
of the hits it has taken, so a string's distribution can be read without the
HUD. Both are neutral grey (red excess 0) and sized as fractions of the
screen; the flash animates for FLASH_S and then the mode stops asking for
repaints.

ADAPTED, NOT REWRITTEN. `AimField` already drew exactly this layout (its
docstring: "something to point at, spread deliberately into the corners") and
already owns the scoring geometry, so this mode composes it rather than
carrying a second copy of the rings that could drift from what is drawn.
`AimField` is no longer a scene layer of its own: a layer that could be
switched off is how the screen came to be empty three separate times (holes
off, rings off, blank screen).

Everything is a fraction of the screen and everything is neutral grey (red
excess exactly 0): the camera is watching this screen.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from ..display.layers import (AimField, GREY_HIT, GREY_MID, mark_name, mono, px,
                              score_against_marks)
from ..display.scene import RenderContext
from .. import ghost
from .base import Mode, ShotEvent, ShotOutcome, register


@register(key=Qt.Key_1)
class StaticTargets(Mode):
    """The default screen: a large centre target and four corner targets."""

    name = "static"
    title = "Static Targets"
    versus_ready = True       # measured for two players: see tests/test_modes.py section 22
    summary = "A big target in the middle and four small ones in the corners."

    @property
    def animated(self) -> bool:
        """True only while a hit flash is live. It used to be a class constant
        False while `advance` returned True during a flash: a lie that cost
        nothing until something (a diagnostic, a future scene change) read the
        flag. Derived from the same list `advance` reads, so the two cannot
        disagree, and idle it is False, so a static screen still asks for no
        repaints."""
        return bool(self._flashes)

    FLASH_S = 0.35            # how long the ring on a struck mark lasts
    # At most this many rings at once: two pistols is the most that can fire
    # together (`modes.base`). The field alone is ~1.7% of the screen above
    # luminance 60 against a 2% budget (RULES B1b); each ring adds up to 0.1
    # percentage points, so five at once would breach it. Measured in
    # tests/test_modes.py section 17.
    MAX_FLASHES = 2

    def __init__(self) -> None:
        self.visible = True
        self._field = AimField()
        self.shots = 0
        self.hits = 0
        self.score = 0
        # Hits taken by each mark, by the id the scorer reports.
        self.hits_by_mark: dict[str, int] = {}
        # (mark id, time of the shot) for each flash still to be shown.
        self._flashes: list[tuple[str, float]] = []
        # The most one shot can score on this field, learned from the field itself
        # at `start` (see `ghost_ceiling`). None until a round has started.
        self._top_points: "int | None" = None

    def start(self, ctx: RenderContext) -> None:
        self.shots = self.hits = self.score = 0
        self.hits_by_mark = {}
        self._flashes = []
        # THROUGH THE SCORER, not retyped: shoot the exact centre of every mark with
        # `score_against_marks` -- the function `shot()` calls -- and keep the best.
        # That is the innermost ring of the biggest mark, whatever the ring
        # arithmetic or the mark layout becomes.
        marks = self._field.marks(ctx)
        self._top_points = max(score_against_marks(cx, cy, marks)[0]
                               for cx, cy, _radii in marks)

    # NO WAIT, EVER, for the ghost: every mark is on screen from the first instant
    # and a hit takes nothing off it (it flashes a ring and tallies; `shot` removes
    # nothing), so a perfect shooter can fire at the top ring again and again at
    # nothing but its own cadence. `ghost_has_waits` stays False.
    def ghost_ceiling(self):
        if self._top_points is None:
            return None
        return ghost.constant_ceiling(self._top_points)

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        self.shots += 1
        points, name = score_against_marks(ev.x, ev.y, self._field.marks(ctx))
        if points <= 0:
            return ShotOutcome.miss()
        self.hits += 1
        self.score += points
        self.hits_by_mark[name] = self.hits_by_mark.get(name, 0) + 1
        self._flashes = (self._flashes + [(name, ev.t)])[-self.MAX_FLASHES:]
        # NO EFFECT SOUND HERE, on purpose (asked for 2026-10-23). This used to return
        # `sound=VANISH_SOUND`, but nothing vanishes in this mode: the mark stays
        # standing and the field is fixed, so the pop announced an event that never
        # happened, on nearly every shot. Clear The Board (circles.py) really does
        # destroy its circle and keeps the sound. Do not restore it here.
        return ShotOutcome(hit=True, points=points, target_id=name)

    def nearest(self, ev: ShotEvent, ctx: RenderContext):
        # The distance `score_against_marks` tests: from each mark's centre to
        # its OUTERMOST ring, inside which a shot scores.
        return self.closest([
            (mark_name(i), math.hypot(ev.x - cx, ev.y - cy), radii[-1])
            for i, (cx, cy, radii) in enumerate(self._field.marks(ctx))])

    def aim_point(self, ctx: RenderContext) -> "tuple[float, float]":
        """Where the development walk-through (`--modetest`) should aim: the
        centre of the large middle target. Without this the walk-through said
        "no target" for this mode too."""
        cx, cy, _radii = self._field.marks(ctx)[0]
        return (cx, cy)

    def advance(self, ctx: RenderContext) -> bool:
        """Repaint only while a flash is showing, plus the one tick that
        removes the last of them. Idle, this asks for nothing at all: a
        static screen must not repaint a 4K surface sixty times a second."""
        if not self._flashes:
            return False
        self._flashes = [f for f in self._flashes
                         if ctx.now - f[1] < self.FLASH_S]
        return True

    def status(self) -> str:
        return f"{self.hits}/{self.shots} on target   score {self.score}"

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        self._field.paint(p, ctx)
        if ctx.calibrating:
            return          # the field shows one ring to shoot at; so do we
        marks = self._field.marks(ctx)
        p.setRenderHint(QPainter.Antialiasing)
        self._paint_flashes(p, ctx, marks)
        self._paint_tallies(p, ctx, marks)

    def _paint_flashes(self, p: QPainter, ctx: RenderContext, marks) -> None:
        """An expanding ring from the struck mark's innermost ring, fading out.
        Grey 240 (red excess 0), so the camera cannot mistake it for a dot."""
        by_name = {mark_name(i): m for i, m in enumerate(marks)}
        p.setBrush(Qt.NoBrush)
        for name, t in self._flashes:
            k = (ctx.now - t) / self.FLASH_S
            if k >= 1.0 or name not in by_name:
                continue
            k = max(0.0, k)
            cx, cy, radii = by_name[name]
            c = QColor(GREY_HIT)
            c.setAlphaF(max(0.0, 1.0 - k))
            p.setPen(QPen(c, max(2.0, ctx.height * 0.005)))
            rr = radii[0] * (1.0 + 0.8 * k)
            p.drawEllipse(QPointF(cx, cy), rr, rr)

    def _paint_tallies(self, p: QPainter, ctx: RenderContext, marks) -> None:
        """The number of hits each mark has taken, beside it, once it has any.
        A mark that has not been hit shows nothing (a column of zeros is
        clutter and lit area for no information)."""
        if not self.hits_by_mark:
            return
        field = self.playfield(ctx)
        p.setFont(mono(px(ctx, 16)))
        p.setPen(QPen(GREY_MID))
        fm = p.fontMetrics()
        gap = px(ctx, 8)
        for i, (cx, cy, radii) in enumerate(marks):
            n = self.hits_by_mark.get(mark_name(i), 0)
            if n <= 0:
                continue
            text = str(n)
            w, h = fm.horizontalAdvance(text), fm.height()
            x = cx + radii[-1] + gap                     # beside the outer ring
            if x + w > field.right():
                x = cx - radii[-1] - gap - w             # ...or on its other side
            y = min(max(cy - h / 2.0, field.top()), field.bottom() - h)
            p.drawText(QRectF(x, y, w, h), Qt.AlignCenter, text)
