"""Clear the board: a field of circles that vanish as you hit them.

The simplest arcade mode, and the one worth getting exactly right, because
every other mode inherits its habits. It measures the thing a shooter actually
wants to know -- how long to clear N targets, and how many rounds it took --
rather than an abstract score.

Deliberate choices:

  - **Targets never overlap and never touch an edge.** An ambiguous hit between
    two circles, or a target half off-screen, makes the result unarguable in the
    wrong direction. The placement is rejection-sampled until it is clean, and
    falls back to a grid if the screen is too crowded for the count asked for.
  - **Everything is a fraction of the screen.** This has to look the same on a
    42" monitor, a 75" TV and a projector, so nothing is in absolute pixels.
  - **A hit is scored against the circle, not the nearest circle.** Shooting
    between two targets is a miss. Generosity here would quietly inflate every
    hit rate the app reports.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from ..display.layers import (GREY_HI as ACCENT, GREY_HI_DIM as ACCENT_DIM,
                              GREY_MISS as COOL, GREY_HIT as GOOD, mono)
from ..display.scene import RenderContext
from ..audio.mixer import VANISH_SOUND
from .. import ghost
from .base import Mode, ShotEvent, ShotOutcome, register


@dataclass
class _Target:
    x: float
    y: float
    r: float
    alive: bool = True
    killed_at: float = 0.0        # for the fade-out, so a hit is legible


@register(key=Qt.Key_5)
class ClearTheBoard(Mode):
    """N circles. Hit them all. The clock tells you how you did."""

    name = "circles"
    title = "Clear The Board"
    versus_ready = True       # measured for two players: see tests/test_modes.py section 22
    summary = "Twelve targets, one clock. Hit them all as fast as you can."
    animated = True

    FADE_S = 0.35                 # how long a hit target lingers while fading

    # HOW LONG THE BOARD STAYS EMPTY BEFORE THE NEXT ONE. His report,
    # 2026-10-18: "after all targets are cleared and the word says round
    # complete, you should populate those targets back. Otherwise the board
    # stays empty."
    #
    # It is a pause and not an instant deal, because the time he just shot is
    # the point of the drill and it is on that banner. Long enough to read
    # "CLEARED 4.21s, 12 hits / 14 shots", short enough that he does not
    # reach for a key. It must also outlast FADE_S, or the last target would
    # still be fading while the new board is dealt on top of it.
    NEXT_BOARD_S = 2.2

    # Points for a hit: one a target. The scorer (`shot`) reads this and so does
    # the ghost's ceiling, so the two cannot disagree.
    POINTS = 1
    # The board is dealt NEXT_BOARD_S after the last target of the previous one is
    # hit, so a perfect shooter waits for it too.
    ghost_has_waits = True

    def ghost_ceiling(self):
        return ghost.constant_ceiling(self.POINTS)

    def ghost_expected_wait(self, target_index):
        """The wait for a target the round never drew: every circle of a board is
        on screen at once, so only the FIRST of a new board waits, and it waits
        NEXT_BOARD_S (`advance` deals the next board that long after the last
        target of this one). A board is `count` circles, as `_note_board` records
        it. The very first target of a round waits for nothing. NOT a flat
        NEXT_BOARD_S per missing shot: that would charge the ghost 2.2 s twelve
        times per board."""
        if target_index > 0 and target_index % max(1, self.count) == 0:
            return self.NEXT_BOARD_S
        return 0.0

    def _note_board(self, appeared_at: float, released_at: "float | None") -> None:
        """Record a freshly dealt board for the ghost. Every circle is on screen at
        once, so only the FIRST carries the wait; the rest are already there."""
        for i in range(len(self.targets)):
            self.note_target(appeared_at, released_at if i == 0 else None)

    def __init__(self, count: int = 12, radius_frac: float = 0.045) -> None:
        self.count = count
        self.radius_frac = radius_frac
        self.targets: list[_Target] = []
        self.visible = True
        self.round_no = 0
        self.started_at = 0.0
        self.cleared_at: float | None = None
        self.best_s: float | None = None
        self.shots = 0
        self.hits = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self, ctx: RenderContext) -> None:
        self.round_no += 1
        self.started_at = ctx.now
        self.cleared_at = None
        self.shots = 0
        self.hits = 0
        self.targets = self._place(ctx)
        self.reset_targets()
        self._note_board(ctx.now, None)

    def _place(self, ctx: RenderContext) -> list[_Target]:
        field = self.playfield(ctx)
        r = self.radius_frac * min(field.width(), field.height())
        gap = r * 0.5                       # clear air between neighbours
        rng = random.Random()
        out: list[_Target] = []
        for _ in range(self.count):
            for _attempt in range(200):
                x = rng.uniform(field.left() + r, field.right() - r)
                y = rng.uniform(field.top() + r, field.bottom() - r)
                # Off the MENU button as well as off its neighbours. He asked
                # for it on 2026-10-18 after photographing a target sitting on
                # the button: a target there is either unhittable or opens a
                # menu, and from the firing line both read as the app
                # misbehaving. A rejected candidate costs one of 200 tries.
                if (all(math.hypot(x - t.x, y - t.y) > 2 * r + gap for t in out)
                        and self.clear_of_menu(ctx, x, y, r)):
                    out.append(_Target(x, y, r))
                    break
            else:
                break        # too crowded to place more; fall through to the grid
        if len(out) < self.count:
            out = self._grid(ctx, r)
        return out

    def _grid(self, ctx: RenderContext, r: float) -> list[_Target]:
        """Fallback when random placement cannot fit the count asked for.

        A tidy grid is a worse drill than scattered targets, but it always fits,
        and silently delivering fewer targets than requested would make the
        round-to-round times incomparable.
        """
        field = self.playfield(ctx)
        cols = max(1, int(math.ceil(math.sqrt(self.count * field.width() / max(1.0, field.height())))))
        rows = max(1, int(math.ceil(self.count / cols)))
        out: list[_Target] = []
        for i in range(self.count):
            cx = i % cols
            cy = i // cols
            x = field.left() + field.width() * (cx + 0.5) / cols
            y = field.top() + field.height() * (cy + 0.5) / rows
            out.append(_Target(x, y, r))
        return out

    # -- play -----------------------------------------------------------------

    # The ring is drawn as an outline of this width, centred on the radius.
    # `paint` and the hit test both take it from here, through the helpers in
    # `modes.base`: a shot on the outer half of the visible line is on the
    # target (same fault as the ball, plan 02 step 12).
    STROKE_FRAC = 0.09
    STROKE_MIN_PX = 2.0

    def _stroke(self, t: "_Target") -> float:
        return self.ring_stroke(t.r, self.STROKE_FRAC, self.STROKE_MIN_PX)

    def _hit_r(self, t: "_Target") -> float:
        return self.ring_hit_radius(t.r, self.STROKE_FRAC, self.STROKE_MIN_PX)

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        self.shots += 1
        for i, t in enumerate(self.targets):
            if t.alive and math.hypot(ev.x - t.x, ev.y - t.y) <= self._hit_r(t):
                t.alive = False
                t.killed_at = ev.t
                self.hits += 1
                if not any(x.alive for x in self.targets):
                    self.cleared_at = ev.t
                    taken = self.cleared_at - self.started_at
                    if self.best_s is None or taken < self.best_s:
                        self.best_s = taken
                # A RING TARGET THAT VANISHES SAYS SO, asked for 2026-10-18.
                return ShotOutcome(hit=True, points=self.POINTS,
                                   target_id=f"circle{i}", sound=VANISH_SOUND)
        return ShotOutcome.miss()

    def nearest(self, ev: ShotEvent, ctx: RenderContext):
        return self.closest([
            (f"circle{i}", math.hypot(ev.x - t.x, ev.y - t.y), self._hit_r(t))
            for i, t in enumerate(self.targets) if t.alive])

    def advance(self, ctx: RenderContext) -> bool:
        # Repaint while anything is fading, or while the clock is running.
        if self.cleared_at is None:
            return True
        # THE NEXT BOARD DEALS ITSELF. The drill used to end with an empty
        # screen and a banner, which from the firing line is the app having
        # stopped: he is standing there with a loaded pistol and nothing to
        # shoot at. A drill that has to be restarted by hand is one he cannot
        # run without walking back to the keyboard.
        if ctx.now - self.cleared_at >= self.NEXT_BOARD_S:
            self._next_board(ctx)
            return True
        return any(not t.alive and ctx.now - t.killed_at < self.FADE_S
                   for t in self.targets)

    def _next_board(self, ctx: RenderContext) -> None:
        """Deal a fresh board and restart the clock, keeping the best time.

        Not `start()`: that is the host's call when the MODE begins, and it
        resets `round_no` bookkeeping that belongs to the session. The best
        time survives on purpose -- it is the number he is shooting against,
        and losing it every twelve targets would make it meaningless.
        """
        released = self.cleared_at           # the last hit of the board that just ended
        self.round_no += 1
        self.started_at = ctx.now
        self.cleared_at = None
        self.shots = 0
        self.hits = 0
        self.targets = self._place(ctx)
        self._note_board(ctx.now, released)

    def on_key(self, key: int) -> bool:
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_N):
            return False      # the host restarts; it owns the RenderContext
        return False

    def status(self) -> str:
        left = sum(1 for t in self.targets if t.alive)
        if self.cleared_at is not None:
            taken = self.cleared_at - self.started_at
            best = f"  best {self.best_s:.2f}s" if self.best_s else ""
            return f"CLEARED in {taken:.2f}s   {self.hits}/{self.shots} hits{best}"
        return f"{left} left   {self.hits}/{self.shots} hits"

    # -- drawing --------------------------------------------------------------

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        for t in self.targets:
            if t.alive:
                self._draw_live(p, t)
            else:
                age = ctx.now - t.killed_at
                if age < self.FADE_S:
                    self._draw_dying(p, t, age / self.FADE_S)

        if self.cleared_at is not None:
            self._draw_banner(p, ctx)

    def _draw_live(self, p: QPainter, t: _Target) -> None:
        # OUTLINES, NO FILL (2026-10-17). The filled disc was the largest lit
        # area the app drew (3.2% of the screen above luminance 60, against a
        # 2% budget), and a dot landing on a lit patch loses its red excess:
        # (255, 200, 200) scores 55 where (255, 10, 10) scores 245.
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(ACCENT, self._stroke(t)))
        p.drawEllipse(QPointF(t.x, t.y), t.r, t.r)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(ACCENT, max(1.0, t.r * 0.05)))
        p.drawEllipse(QPointF(t.x, t.y), t.r * 0.45, t.r * 0.45)

    def _draw_dying(self, p: QPainter, t: _Target, k: float) -> None:
        """A ring that expands and fades, so a hit is unmistakable in peripheral
        vision -- the shooter is looking at the next target, not this one."""
        c = QColor(GOOD)
        c.setAlphaF(max(0.0, 1.0 - k))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(c, max(2.0, t.r * 0.12)))
        p.drawEllipse(QPointF(t.x, t.y), t.r * (1.0 + k * 0.8), t.r * (1.0 + k * 0.8))

    def _draw_banner(self, p: QPainter, ctx: RenderContext) -> None:
        taken = self.cleared_at - self.started_at
        p.setFont(mono(max(22, ctx.height // 26), bold=True))
        p.setPen(GOOD)
        p.drawText(ctx.rect, Qt.AlignHCenter | Qt.AlignVCenter,
                   f"CLEARED  {taken:.2f}s\n{self.hits} hits / {self.shots} shots")
        p.setFont(mono(max(13, ctx.height // 60)))
        p.setPen(COOL)
        p.drawText(ctx.rect.adjusted(0, int(ctx.height * 0.12), 0, 0),
                   Qt.AlignHCenter | Qt.AlignVCenter, "N for a new board")
