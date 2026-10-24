"""Several balls bouncing off the edges. Hit one and it comes back smaller and faster.

Asked for in these words: "several balls would just bounce off the screen edges
and would disappear when they were shot by the laser. Once they had been shot
they reappear again but this time moving 10% faster and being 10% smaller."

Revised 2026-10-16 after live play: five levels, each 5% faster and 5% smaller
than the last (top speed 1.22x, smallest ball 0.81x the start), no further.
Uncapped, the ball outran the detection hardware, and the small ball was
reported unhittable. The shrink is halved with the speed rather than kept at
10%: at 10% level 5 would be 0.66x, and the complaint was precisely about
small targets.

So each ball carries its OWN level. Hitting the easy one does not punish you on
the hard one, and after a few minutes the screen holds a spread of difficulties
at once -- which is what makes it a drill rather than a single escalating
target. The one you keep hitting becomes the one you can barely hit, while the
others wait.

Two limits that keep it playable rather than merely punishing:

- **Balls stop shrinking** at a radius the camera can still resolve. A target
  smaller than the laser's own dot is not a test of marksmanship, it is a test
  of the detector, and it fails as a miss that looks like a bug. The floor is
  set from the measured dot size on this hardware -- see MIN_RADIUS_FRAC.
- **Speed is capped** well above what anyone tracks, so a ball cannot become a
  blur that crosses the screen between two camera frames and is therefore
  unhittable even in principle. At 60fps and a 100ms pulse, a ball crossing
  faster than about two screen widths per second cannot be reliably hit.

Everything is sized as a fraction of the screen, so the same drill plays the
same on a 42" monitor, a 75" TV and a projector. And nothing is drawn in
saturated red: the camera is watching this screen -- see the no-red rule in
`modes.base`.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from ..display.layers import (GREY_HI, GREY_HIT as GOOD, GREY_TEXT, grey_at,
                              mono, px)
from ..display.scene import RenderContext
from ..audio.mixer import POP_SOUND
from .. import ghost
from .base import Mode, ShotEvent, ShotOutcome, register

OUTLINE = QColor(GREY_HI)         # red excess 0, under the brightness ceiling: see paint()
# THE BALL IS FILLED, very darkly. See `paint`: the fill is what a dot lands
# on, so its brightness comes off the shot score one for one. 20 against a
# background of 10 costs ten units; the fill this replaced cost fifty-five.
FILL = grey_at(0.25)


class _Ball:
    """One target. Carries its own difficulty, which is the whole point."""

    __slots__ = ("x", "y", "vx", "vy", "r", "level", "hits", "ident", "born")

    def __init__(self, ident: int) -> None:
        self.ident = ident
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.r = 10.0
        self.level = 1
        self.hits = 0
        self.born = 0.0


@register(key=Qt.Key_3)
class BouncingBalls(Mode):
    """Several balls at once. Each one you hit returns faster and smaller."""

    name = "balls"
    title = "Bouncing Balls"
    versus_ready = True       # measured for two players: see tests/test_modes.py section 22
    summary = "Several balls. Each one you hit comes back 5% faster and 5% smaller, up to level 5."
    animated = True

    def moving_at(self, t: float) -> bool:
        """The ball(s) are in continuous motion for the whole round, so the
        sensitivity learner is PAUSED in this mode (logged, not silent)."""
        return True

    COUNT = 4                 # enough to force target selection, few enough to read
    START_SPEED = 0.08        # screen widths per second
    SPEED_STEP = 1.05         # +5% per level
    SHRINK_STEP = 0.95        # -5% per level
    MAX_LEVEL = 5             # and no further
    START_RADIUS_FRAC = 0.080  # of the smaller playfield dimension

    # A floor on size and a ceiling on speed, both measured rather than picked.
    #
    # The laser dot images as roughly 15-25 camera pixels across on this
    # hardware, and the screen occupies about a third of the camera frame, so a
    # target below ~1% of screen height is smaller than the dot that has to hit
    # it. Past that point a miss is the detector's doing, not the shooter's.
    MIN_RADIUS_FRAC = 0.010
    # A 100ms pulse at a ball crossing faster than ~2 screen widths per second
    # moves more than its own diameter during the shot, so where it "was" when
    # the trigger broke stops being well defined.
    MAX_SPEED = 2.0

    FLASH_S = 0.35

    # The k-th hit a perfect shooter makes is worth min(MAX_LEVEL, k): it keeps
    # hitting the one ball, which `shot` awards at the level it HAD reached and
    # then raises, capped (the other balls' levels do not matter to a ghost that
    # never leaves the first). NO WAIT for the ghost (`ghost_has_waits` stays
    # False): `_place` puts the replacement ball back inside the very `shot()` call
    # that scored the hit. See `EscalatingBall.ghost_ceiling` for the guard.
    def ghost_ceiling(self):
        return ghost.escalating_ceiling(self.MAX_LEVEL)

    def __init__(self) -> None:
        self.visible = True
        self.balls: list[_Ball] = []
        self.shots = 0
        self.hits = 0
        self.started_at = 0.0
        self.flashes: list[tuple[float, float, float, int]] = []   # x, y, t, level
        self._rng = random.Random()

    # -- lifecycle ------------------------------------------------------------

    def start(self, ctx: RenderContext) -> None:
        self.shots = 0
        self.hits = 0
        self.started_at = ctx.now
        self.flashes = []
        self.balls = [_Ball(i) for i in range(self.COUNT)]
        for i, b in enumerate(self.balls):
            self._place(b, ctx, spread=i)
            b.born = ctx.now

    def _radius(self, ball: _Ball, ctx: RenderContext) -> float:
        field = self.playfield(ctx)
        base = min(field.width(), field.height())
        lvl = min(ball.level, self.MAX_LEVEL)
        frac = max(self.MIN_RADIUS_FRAC,
                   self.START_RADIUS_FRAC * (self.SHRINK_STEP ** (lvl - 1)))
        return frac * base

    def _speed_px(self, ball: _Ball, ctx: RenderContext) -> float:
        lvl = min(ball.level, self.MAX_LEVEL)
        frac = min(self.MAX_SPEED,
                   self.START_SPEED * (self.SPEED_STEP ** (lvl - 1)))
        return frac * self.playfield(ctx).width()

    def _place(self, ball: _Ball, ctx: RenderContext,
               away_from: "tuple[float, float] | None" = None,
               spread: int = 0) -> None:
        field = self.playfield(ctx)
        ball.r = self._radius(ball, ctx)
        inner = field.adjusted(ball.r, ball.r, -ball.r, -ball.r)

        if away_from is None:
            # Spread the opening positions around the field rather than
            # stacking them: four balls born on top of each other look like one.
            cols = 2
            fx = (spread % cols + 0.5) / cols
            fy = (spread // cols + 0.5) / max(1, (self.COUNT + cols - 1) // cols)
            ball.x = inner.left() + inner.width() * fx
            ball.y = inner.top() + inner.height() * fy
        else:
            # Back in from the far side of where the shot landed, so nobody
            # clears a level by holding still and waiting.
            ball.x = (inner.left() if away_from[0] > inner.center().x()
                      else inner.right())
            ball.y = (inner.top() if away_from[1] > inner.center().y()
                      else inner.bottom())
        ball.x = min(max(ball.x, inner.left()), inner.right())
        ball.y = min(max(ball.y, inner.top()), inner.bottom())

        # Never near-axis-aligned: a ball sliding along an edge is easy and dull.
        angle = self._rng.choice([
            self._rng.uniform(0.35, 1.20), self._rng.uniform(1.94, 2.79),
            self._rng.uniform(3.49, 4.33), self._rng.uniform(5.08, 5.93)])
        speed = self._speed_px(ball, ctx)
        ball.vx = math.cos(angle) * speed
        ball.vy = math.sin(angle) * speed

    # -- play -----------------------------------------------------------------

    def advance(self, ctx: RenderContext) -> bool:
        field = self.playfield(ctx)
        # Clamp dt: a stalled frame must not teleport a ball through a wall.
        dt = max(0.0, min(ctx.dt, 0.05))
        for b in self.balls:
            b.r = self._radius(b, ctx)
            inner = field.adjusted(b.r, b.r, -b.r, -b.r)
            if inner.width() <= 0 or inner.height() <= 0:
                continue
            b.x += b.vx * dt
            b.y += b.vy * dt
            # Reflect AND clamp in the same step. Reflecting alone lets a ball
            # that overshot stay outside and flip its velocity every frame,
            # which looks like a seizure and cannot be hit.
            if b.x <= inner.left():
                b.x, b.vx = inner.left(), abs(b.vx)
            elif b.x >= inner.right():
                b.x, b.vx = inner.right(), -abs(b.vx)
            if b.y <= inner.top():
                b.y, b.vy = inner.top(), abs(b.vy)
            elif b.y >= inner.bottom():
                b.y, b.vy = inner.bottom(), -abs(b.vy)
        if self.flashes:
            self.flashes = [f for f in self.flashes
                            if ctx.now - f[2] < self.FLASH_S]
        return True

    def _position_at(self, b: _Ball, age: float, ctx: RenderContext) -> tuple[float, float]:
        """Where ball `b` was `age` seconds ago (bounces unfolded). Used by
        both `shot()` and `nearest()`, so the logged distance is the tested one."""
        inner = self.playfield(ctx).adjusted(b.r, b.r, -b.r, -b.r)
        return (self.fold(b.x - b.vx * age, inner.left(), inner.right()),
                self.fold(b.y - b.vy * age, inner.top(), inner.bottom()))

    def nearest(self, ev: ShotEvent, ctx: RenderContext):
        age = self.shot_age(ev, ctx)
        out = []
        for b in self.balls:
            bx, by = self._position_at(b, age, ctx)
            out.append((f"ball{b.ident}", math.hypot(ev.x - bx, ev.y - by),
                        self.ring_hit_radius(b.r)))
        return self.closest(out)

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        self.shots += 1
        # Nearest ball whose circle contains the shot. Nearest, not first:
        # two balls can overlap, and awarding the hit to whichever happens to
        # be earlier in the list would be arbitrary from the firing line.
        #
        # Each ball is tested where it WAS at the moment of the shot
        # (`Mode.shot_age`), not where it is drawn now: the shot is confirmed
        # two camera frames later, and he had to shoot in front of the ball.
        age = self.shot_age(ev, ctx)
        best, best_d = None, 0.0
        for b in self.balls:
            bx, by = self._position_at(b, age, ctx)
            d = math.hypot(ev.x - bx, ev.y - by)
            if d <= self.ring_hit_radius(b.r) and (best is None or d < best_d):
                best, best_d = b, d
        if best is None:
            return ShotOutcome.miss()

        self.hits += 1
        best.hits += 1
        reached = best.level
        best.level = min(self.MAX_LEVEL, best.level + 1)
        self.flashes.append((best.x, best.y, ev.t, reached))
        self._place(best, ctx, away_from=(ev.x, ev.y))
        best.born = ctx.now
        # HIS SOUND FOR A BURST, asked for 2026-10-18. The ordinary
        # gunshot still plays for a miss; a ball that pops says so.
        return ShotOutcome(hit=True, points=reached,
                           target_id=f"ball{best.ident}", sound=POP_SOUND)

    def status(self) -> str:
        if not self.balls:
            return ""
        levels = sorted((b.level for b in self.balls), reverse=True)
        toughest = self.SPEED_STEP ** (levels[0] - 1)
        return (f"{self.hits}/{self.shots} hits   levels "
                + "/".join(str(v) for v in levels)
                + f"   toughest x{toughest:.2f} speed")

    # -- drawing --------------------------------------------------------------

    def occluders(self, ctx: RenderContext):
        """Each ball is solid, so the holes it crosses do not show through.

        His instruction, 2026-10-18: the figures that go over holes should
        obscure them. The ball is drawn as an outline -- a fill swallows the
        shot that hits it (B10) -- so declaring the shape is the only way to
        say it is a thing rather than a ring of empty screen. Nothing drawn
        changes.
        """
        if ctx.calibrating:
            return []
        return [self.disc_occluder(self.z, b.x, b.y, b.r) for b in self.balls]

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        p.setRenderHint(QPainter.Antialiasing)

        # The fading ring left where a ball was hit, so a kill is visible even
        # when the replacement spawns on the far side of the screen.
        for fx, fy, ft, level in self.flashes:
            age = (ctx.now - ft) / self.FLASH_S
            if age >= 1.0:
                continue
            c = QColor(GOOD)
            c.setAlphaF(max(0.0, 1.0 - age))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(c, max(2.0, ctx.height * 0.004)))
            grow = ctx.height * 0.02 * (1.0 + age * 2.0)
            p.drawEllipse(QPointF(fx, fy), grow, grow)

        for b in self.balls:
            # OUTLINE ONLY, neutral grey, red excess ZERO. A fill sits exactly
            # where the dot lands and its green lowers the dot's score (a
            # yellow fill cost ~55 of his median 85). And not ACCENT (excess
            # 8): the saved sensitivity map amplifies the weakest region up to
            # 4x, so a moving 8 reaches the detector as 16-18 -- a phantom
            # shot (RULES B1). Difficulty is carried by the level digit.
            p.setBrush(FILL)
            p.setPen(QPen(OUTLINE, self.ring_stroke(b.r)))
            p.drawEllipse(QPointF(b.x, b.y), b.r, b.r)

            # The target face, so each ball reads as a target rather than a
            # disc. Painted only; the ball is still hit or missed as a whole.
            self.paint_rings(p, b.x, b.y, b.r)

            # Its level, inside it, in the outline colour (a dark digit on the
            # now-unfilled ball would be invisible), while it is big enough.
            if b.r > ctx.height * 0.018:
                p.setPen(QPen(OUTLINE))
                p.setFont(mono(max(9, int(b.r * 0.9)), bold=True))
                p.drawText(QPointF(b.x - b.r * 0.28, b.y + b.r * 0.33),
                           str(b.level))

        # The rule, for the first few seconds of a round only. Permanent, it
        # is clutter on a screen that is also being shot at; and it sat on the
        # key legend, because the playfield's bottom edge is exactly where that
        # legend begins.
        if ctx.now - self.started_at < 8.0:
            field = self.playfield(ctx)
            p.setPen(QPen(QColor(GREY_TEXT)))
            p.setFont(mono(px(ctx, 14)))
            p.drawText(int(field.left()) + 8,
                       int(field.bottom()) - px(ctx, 16),
                       "each ball you hit returns 5% faster and 5% smaller (5 levels)")
