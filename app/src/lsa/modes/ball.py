"""Escalation: one bouncing ball, 5% faster every time you hit it, five levels.

A difficulty curve with no settings. The ball starts at a speed anyone can hit
and multiplies by 1.05 per kill up to level 5 (top speed 1.05^4 = 1.22x the
start) and no further. It was uncapped at +10%, and the owner measured it
outrunning the detection hardware: "that ball moves too fast for our detection
hardware so let's cap that speed". The size does not change at all: the small
ball was reported unhittable.

Three things that make it fair:

  - **Speed is in screen-widths per second**, never pixels, so the drill is the
    same on a 42" monitor, a 75" TV and a projector. Pixels would make a 4K
    screen four times harder than a 1080p one for no reason.
  - **A new ball never spawns on top of the muzzle.** It appears at the far side
    of the playfield from the last hit, so nobody wins a level by holding still.
  - **A miss costs nothing but time.** Punishing misses teaches flinching; the
    clock already punishes them.

The ball is a circle, not a sprite, because the camera is watching this screen
and the palette is constrained -- see the no-red rule in `modes.base`.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from ..display.layers import (GREY_HI, GREY_MISS as COOL, GREY_HIT as GOOD,
                              grey_at, mono)
from ..display.scene import RenderContext
from ..audio.mixer import POP_SOUND
from .. import ghost
from .base import Mode, ShotEvent, ShotOutcome, register

OUTLINE = QColor(GREY_HI)         # red excess 0, under the brightness ceiling: see paint()
# THE BALL IS FILLED, very darkly. See `paint`: the fill is what a dot lands
# on, so its brightness comes off the shot score one for one. 20 against a
# background of 10 costs ten units; the fill this replaced cost fifty-five.
FILL = grey_at(0.25)


@register(key=Qt.Key_2)
class EscalatingBall(Mode):
    """Hit the ball. It comes back faster. Repeat until it beats you."""

    name = "ball"
    title = "Escalating Ball"
    versus_ready = True       # measured for two players: see tests/test_modes.py section 22
    summary = "One ball. Every hit makes the next one 5% faster, up to level 5."
    animated = True

    def moving_at(self, t: float) -> bool:
        """The ball(s) are in continuous motion for the whole round, so the
        sensitivity learner is PAUSED in this mode (logged, not silent)."""
        return True

    START_SPEED = 0.09        # screen widths per second: half the first draft's drift
    SPEED_STEP = 1.05         # +5% per level
    MAX_LEVEL = 5             # and no further
    RADIUS_FRAC = 0.07
    FLASH_S = 0.3

    # The k-th hit of a round is worth min(MAX_LEVEL, k): `shot` awards the level
    # the ball HAD reached and then raises it, capped. NO WAIT for the ghost
    # (`ghost_has_waits` stays False): the replacement ball is spawned by `_spawn`
    # inside the very `shot()` call that scored the hit, so there is no interval in
    # which nothing can be shot. If a respawn delay is ever added, this must become
    # True and `start`/`shot` must `note_target` -- tests/test_modes.py section 47
    # drives a hit and fails if a ball is not on screen straight after it.
    def ghost_ceiling(self):
        return ghost.escalating_ceiling(self.MAX_LEVEL)

    def __init__(self) -> None:
        self.visible = True
        # ITS OWN GENERATOR, seeded from the OS. A mode that draws from the
        # global `random` cannot be driven deterministically by a test, and
        # "the entries are unpredictable" is a claim about a DISTRIBUTION --
        # which needs thousands of draws to check, not one.
        self._rng = random.Random()
        self.level = 1
        self.best_level = 1
        self.shots = 0
        self.hits = 0
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.r = 10.0
        self.started_at = 0.0
        self.flash_at = -99.0
        self.flash_pos = (0.0, 0.0)

    # -- lifecycle ------------------------------------------------------------

    def start(self, ctx: RenderContext) -> None:
        self.level = 1
        self.shots = 0
        self.hits = 0
        self.started_at = ctx.now
        self._spawn(ctx, away_from=None)

    def _speed_px(self, ctx: RenderContext) -> float:
        """Current speed in pixels per second."""
        return (self.START_SPEED * (self.SPEED_STEP ** (min(self.level, self.MAX_LEVEL) - 1))
                * self.playfield(ctx).width())

    def _spawn(self, ctx: RenderContext, away_from: "tuple[float, float] | None") -> None:
        field = self.playfield(ctx)
        self.r = self.RADIUS_FRAC * min(field.width(), field.height())
        inner = field.adjusted(self.r, self.r, -self.r, -self.r)

        # IT COMES IN FROM ANYWHERE ON THE EDGE, NEVER FROM A CORNER.
        #
        # His report, 2026-10-18: "the balls very predictably show up from the
        # corner. They should float in from pretty much anywhere and never in
        # a predictable way."
        #
        # He is describing the code exactly: the old spawn picked the inner
        # rectangle's LEFT or RIGHT and TOP or BOTTOM from the side the last
        # hit was on, so there were four possible entry points in the whole
        # drill and the shooter learned them in a minute. A target whose
        # position is known before it exists is not a reaction drill.
        #
        # Now the entry point is anywhere along the perimeter, sampled by
        # length so a long edge gets its share, and the ball heads for a
        # random point in the middle of the field -- so it floats IN rather
        # than popping into the middle, and no two entries are alike.
        if away_from is None:
            self.x, self.y = self._perimeter_point(inner, self._rng.random())
        else:
            # STILL AWAY FROM THE LAST HIT. That was the point of the corner
            # rule and it is worth keeping: if the next ball can appear under
            # his muzzle, the drill stops asking him to move. Unpredictable
            # does not mean "sometimes free". Best of a handful of candidates
            # rather than a loop that could spin: bounded by construction.
            # A QUARTER OF THE SCREEN WIDTH, which is the bar this drill has
            # always been held to. Not a fraction of the DIAGONAL: that is
            # only reachable from the corners when the last hit was central,
            # so "far from the last hit" quietly meant "a corner again" --
            # 91 of 400 entries landed within a radius of one. Not the short
            # side either: on this panel it falls just under the bar and the
            # ball was handed back too close one time in six.
            far = field.width() * 0.25
            best, best_d = None, -1.0
            for _ in range(12):
                cand = self._perimeter_point(inner, self._rng.random())
                d = math.hypot(cand[0] - away_from[0], cand[1] - away_from[1])
                if d > best_d:
                    best, best_d = cand, d
                if d >= far:
                    break
            self.x, self.y = best
        self.x = min(max(self.x, inner.left()), inner.right())
        self.y = min(max(self.y, inner.top()), inner.bottom())

        # Heading for somewhere in the middle, so it always travels INTO the
        # field whichever edge it entered on, and never slides along that edge
        # -- a ball running parallel to a border is both easy and dull, which
        # is what the old four-arc angle rule was avoiding. Aiming at a point
        # keeps that property without having to know which edge we are on.
        cx, cy = inner.center().x(), inner.center().y()
        speed = self._speed_px(ctx)
        for _ in range(8):
            tx = cx + (self._rng.random() - 0.5) * inner.width() * 0.5
            ty = cy + (self._rng.random() - 0.5) * inner.height() * 0.5
            dx, dy = tx - self.x, ty - self.y
            n = math.hypot(dx, dy)
            if n < 1e-6:
                continue
            ux, uy = dx / n, dy / n
            if min(abs(ux), abs(uy)) >= 0.20:     # not near-axis-aligned
                break
        else:                                      # pragma: no cover
            ux, uy = 0.707, 0.707
        self.vx, self.vy = ux * speed, uy * speed

    @staticmethod
    def _perimeter_point(inner, u: float) -> "tuple[float, float]":
        """A point at fraction `u` around the rectangle, by LENGTH.

        By length, not one-quarter per side: sampling a side first and then a
        position along it would crowd entries onto the short edges of a 16:9
        panel. Pure and side-effect free, so the distribution is testable
        without a mode, a screen or a clock.
        """
        w, h = inner.width(), inner.height()
        per = 2.0 * (w + h)
        d = (u % 1.0) * per
        if d < w:
            return (inner.left() + d, inner.top())
        d -= w
        if d < h:
            return (inner.right(), inner.top() + d)
        d -= h
        if d < w:
            return (inner.right() - d, inner.bottom())
        d -= w
        return (inner.left(), inner.bottom() - d)

    # -- play -----------------------------------------------------------------

    def advance(self, ctx: RenderContext) -> bool:
        field = self.playfield(ctx)
        inner = field.adjusted(self.r, self.r, -self.r, -self.r)
        dt = max(0.0, min(ctx.dt, 0.05))       # clamp: a stalled frame must not teleport it
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Reflect, and clamp back inside in the same step. Reflecting without
        # clamping lets a ball that overshot stay outside and flip every frame,
        # which looks like a seizure and is unhittable.
        if self.x <= inner.left():
            self.x, self.vx = inner.left(), abs(self.vx)
        elif self.x >= inner.right():
            self.x, self.vx = inner.right(), -abs(self.vx)
        if self.y <= inner.top():
            self.y, self.vy = inner.top(), abs(self.vy)
        elif self.y >= inner.bottom():
            self.y, self.vy = inner.bottom(), -abs(self.vy)
        return True

    def _position_at(self, age: float, ctx: RenderContext) -> tuple[float, float]:
        """Where the ball was `age` seconds ago (bounces unfolded)."""
        field = self.playfield(ctx)
        inner = field.adjusted(self.r, self.r, -self.r, -self.r)
        return (self.fold(self.x - self.vx * age, inner.left(), inner.right()),
                self.fold(self.y - self.vy * age, inner.top(), inner.bottom()))

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        self.shots += 1
        # Tested against where the ball was at the moment of the shot, not
        # where it is drawn now (`Mode.shot_age`). Measured 19:57:49-58:
        # fifteen shots, all detected, fourteen scored a miss.
        bx, by = self._position_at(self.shot_age(ev, ctx), ctx)
        if math.hypot(ev.x - bx, ev.y - by) <= self.ring_hit_radius(self.r):
            self.hits += 1
            reached = self.level
            self.level = min(self.MAX_LEVEL, self.level + 1)
            self.best_level = max(self.best_level, self.level)
            self.flash_at = ev.t
            self.flash_pos = (self.x, self.y)
            self._spawn(ctx, away_from=(ev.x, ev.y))
            # HIS SOUND FOR A BURST, asked for 2026-10-18.
            return ShotOutcome(hit=True, points=reached, target_id="ball",
                               sound=POP_SOUND)
        return ShotOutcome.miss()

    def nearest(self, ev: ShotEvent, ctx: RenderContext):
        # Where `shot()` will test it: the position at the moment of the shot.
        bx, by = self._position_at(self.shot_age(ev, ctx), ctx)
        return ("ball", math.hypot(ev.x - bx, ev.y - by),
                self.ring_hit_radius(self.r))

    def status(self) -> str:
        mult = self.SPEED_STEP ** (self.level - 1)
        top = "  (top)" if self.level >= self.MAX_LEVEL else ""
        return (f"level {self.level}/{self.MAX_LEVEL}{top}   speed x{mult:.2f}   "
                f"{self.hits}/{self.shots} hits   best level {self.best_level}")

    # -- drawing --------------------------------------------------------------

    def occluders(self, ctx: RenderContext):
        """The ball is solid, so the holes it crosses do not show through.

        See `BouncingBalls.occluders`: one shared definition in `base`, so
        the escalating ball and the bouncing ones cannot drift apart.
        """
        if ctx.calibrating:
            return []
        return [self.disc_occluder(self.z, self.x, self.y, self.r)]

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        p.setRenderHint(QPainter.Antialiasing)

        # A VERY DARK FILL, so it reads as a ball rather than a ring. He asked
        # for it on 2026-10-18: "can you fill it with the shapes of dark gray?
        # So it looks a bit more like a ball, kind of."
        #
        # THE FILL IS WHAT THE DOT LANDS ON, so its brightness comes straight
        # off the shot's score: the detector sees `red - max(green, blue)`, the
        # dot's red saturates, and the fill's grey is sitting in green and blue
        # underneath it. That is why this was outline-only -- an earlier fill
        # put about 55 units of green under every hit and took his median shot
        # from 85 to ~30. The target was hiding the shots that hit it.
        #
        # So the number is derived, not chosen: FILL is 20 against a background
        # of 10, which costs a landing dot TEN units. His dimmest logged shot
        # ever is 31 and the bar runs 8-19, so ten is affordable where
        # fifty-five was not. It is also far below 60, the luminance at which a
        # pixel starts spending the lit-area budget (B1b), so a screen full of
        # balls costs nothing there. Red excess is 0, as everything here is.
        p.setBrush(FILL)
        p.setPen(QPen(OUTLINE, self.ring_stroke(self.r)))
        p.drawEllipse(QPointF(self.x, self.y), self.r, self.r)

        # The target face, so it reads as a target rather than a disc. Painted
        # only -- the ball is still one target, hit or missed as a whole. Every
        # number is in `Mode.paint_rings`, derived from B10 and B1b.
        self.paint_rings(p, self.x, self.y, self.r)

        age = ctx.now - self.flash_at
        if age < self.FLASH_S:
            k = age / self.FLASH_S
            c = QColor(GOOD)
            c.setAlphaF(max(0.0, 1.0 - k))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(c, max(2.0, self.r * 0.15)))
            rr = self.r * (1.0 + k * 1.4)
            p.drawEllipse(QPointF(*self.flash_pos), rr, rr)

        p.setFont(mono(max(14, ctx.height // 52), bold=True))
        p.setPen(COOL)
        box = QRectF(ctx.rect).adjusted(0, ctx.height * 0.04, 0, 0)
        p.drawText(box, Qt.AlignHCenter | Qt.AlignTop, f"LEVEL {self.level}" + (" (MAX)" if self.level >= self.MAX_LEVEL else ""))
