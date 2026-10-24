"""Mode 9: a figure that comes toward you and goes away, as if it were walking.

Asked for as: "make them move across the screen, like in and out, enlarging
image and making it smaller as if the target moves back and forth." The size IS
the distance: the figure grows as it approaches and shrinks as it retreats,
drifting sideways a little as it does.

EVERYTHING A CYCLE CONTROLS IS DRAWN AT RANDOM, through `self._rng` (seedable;
no module-level `random`): which figure (uniform, never the one just shown), the
wait before it appears, the side it drifts in from (left, right or straight
on), the distance it starts at, the approach, near-hold, retreat and far-hold
durations, the drift speed, and where it holds (at least 35% of the playfield
diagonal from the last hold). Nothing in a cycle is inherited from the previous
one except those two exclusions. The cast is variety, not information: nothing
is scored on which figure it is.

THE SHOT IS TESTED WHERE THE FIGURE WAS. A shot is confirmed two camera frames
after the trigger, and a scaling figure has grown since: the hit test rewinds
by the shot's age (`Mode.shot_age`, capped) and asks for the figure's centre
AND HEIGHT then, not now. The zone geometry changes under a shot in flight,
which is why the rewind covers height and not only position.

WHAT IT MEASURES. Hit rate and ring score in three distance bands (thirds of
the height range, labelled by apparent distance: 1.0x at the nearest height,
2.0x at the farthest), and the time from each hold beginning to the first hit.
Only figures whose printed rings verified are ever drawn, so every shot at a
figure can be judged. A HIT IS VISIBLE, AND RIDES THE FIGURE. A shot that lands on a figure becomes
part of it: a grey ring with its ring number for `FLASH_S`, and a grey hole that
stays, both stored in the figure's own coordinates (`figure.Marks`, fractions
of its drawn box) and redrawn at its current position and size every frame, so
they walk, grow and shrink with it. The screen's bullet-hole layer is told not
to draw an absorbed shot (`ShotOutcome.absorbed`), so there is one mark per
shot, not two drifting apart. A hit does NOT stop, drop or remove the figure:
it carries its hits and completes its approach. A shot that misses every
figure leaves the ordinary screen hole. When a figure leaves, its holes go with
it and nothing is left on the screen.

THE FIGURE DROPS AT A RUNNING TOTAL OF 30 (`figure.DROP_TOTAL`), a total that is
that figure's own. Each scoring hit adds its ring value; the flash and the
counter beside the figure (which rides it, walking and scaling) show the TOTAL.
At 30 the figure leaves at once, as at the end of a normal walk, and the next
cycle begins; the last hit's flash stays at the screen spot.

THE FALL. A figure with a fall pose shows it for `figure.FALL_S` (1 s), frozen
where and at the size it was shot, then it is gone; one without a pose leaves at
once. The blank wait before the next figure runs CONCURRENTLY with the fall (the
next one appears at the later of the two), so the fall delays nothing beyond its
own second. A shot with nothing else up during the fall is not a hit, not a miss
and not "NO TARGET": it leaves its ordinary hole and is counted in `fall_shots`.
The mode reports as moving for that second (`moving_at`).

REPAINTS. 20 Hz while a figure is moving (against the ball modes' 60): one
blit of a cached pixmap costs a few milliseconds on a 4K surface that the
detector shares. ZERO while holding and while the screen is blank: a figure
that is holding still must ask for nothing.

`COUNT = 2` (one coming while another goes) is supported and OFF until the
figure's cost to detection has been measured on the real screen. Figures ADD
in lit area, so with two each is capped at half of `COMBINED_BUDGET`.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter

from ..artwork import catalog
from ..display.scene import RenderContext
from ..watchdog import record_mark
from ..audio.fall_sounds import FallSounds
from .base import Mode, NearestTarget, ShotEvent, ShotOutcome, register
from types import SimpleNamespace

from .figure import (COMBINED_BUDGET, HEAD_NOTE, MAX_MARKS, NO_RING, SINGLE_BUDGET,
                     Fall, Figure, Marks, PlanLink, expected_wait_after_drop,
                     figure_ghost_ceiling, figure_names, paint_hint, paint_hits,
                     shots_per_figure)

# Which end of the size range is "far": the apparent-distance ratio of the
# nearest height to the farthest.
FAR_RATIO = 2.0
BAND_NAMES = ("near", "mid", "far")


class _Actor:
    """One figure's cycle: a blank wait, then a live walk, then gone."""

    def __init__(self) -> None:
        self.phase = "gap"                 # gap -> live -> gap
        self.live_at = 0.0
        self.fig: "Figure | None" = None
        self.t0 = 0.0
        # filled when the cycle starts
        self.t_a = self.t_h = self.t_r = self.t_f = 0.0
        self.h_near = self.h_far = self.h_start = 0.0
        self.hold_xy = (0.0, 0.0)
        self.drift = 0.0                   # px per second, signed
        self.x_lo = self.x_hi = 0.0
        self.hold_hit = False              # first hit after the hold begun?
        self.was_moving = False
        self.marks = Marks()               # shots this figure has absorbed
        self.plan = None                   # the shared plan (two players only)
        self.fall: "Fall | None" = None    # its fall pose, while it lies there
        self.fall_span: "tuple[float, float] | None" = None   # kept for late shots
        # When its previous figure was resolved: the instant the wait for this
        # one began. Consumed when it appears; the ghost's wait record only.
        self.released_at: "float | None" = None

    @property
    def duration(self) -> float:
        return self.t_a + self.t_h + self.t_r + self.t_f

    def state(self, t: float):
        """(centre, height px) at time `t`, or None if not on screen then."""
        age = t - self.t0
        if age < 0.0 or age > self.duration or self.fig is None:
            return None
        hx, hy = self.hold_xy
        if age < self.t_a:                                    # approaching
            k = age / self.t_a
            h = self.h_start + (self.h_near - self.h_start) * k
            x = hx - self.drift * (self.t_a - age)
        elif age < self.t_a + self.t_h:                       # holding near
            h, x = self.h_near, hx
        elif age < self.t_a + self.t_h + self.t_r:            # retreating
            k = (age - self.t_a - self.t_h) / self.t_r
            h = self.h_near + (self.h_far - self.h_near) * k
            x = hx + self.drift * (age - self.t_a - self.t_h)
        else:                                                 # holding far
            h = self.h_far
            x = hx + self.drift * self.t_r
        x = min(max(x, self.x_lo), self.x_hi)
        return (x, hy), h

    def moving(self, t: float) -> bool:
        age = t - self.t0
        if age < 0.0 or age > self.duration:
            return False
        return age < self.t_a or self.t_a + self.t_h <= age < self.t_a + self.t_h + self.t_r


@register(key=Qt.Key_8)
class AdvancingTarget(Mode):
    """A figure walks toward you, holds, and walks away."""

    name = "advancing"
    title = "Moving Target"
    summary = ("A figure comes toward you and goes away, at random. Size is distance. "
               "Head: dome = 8, its printed 9 and 7 have no ring.")
    note = HEAD_NOTE

    COUNT = 1                  # 2 = one approaching while another retreats (OFF)
    RATE_HZ = 20.0             # repaints requested while a figure is moving
    FLASH_S = 0.6

    GAP_S = (0.5, 1.5)         # blank wait before a figure appears
    APPROACH_S = (3.0, 5.0)
    HOLD_NEAR_S = (1.5, 3.0)
    RETREAT_S = (3.0, 5.0)
    HOLD_FAR_S = (1.0, 2.0)
    FAR_FRAC = 1.0 / FAR_RATIO         # h_far = this x h_near
    START_MAX_FRAC = 0.8               # starting height <= this x h_near
    DRIFT_MAX = 0.08                   # screen widths per second, during moves
    # TWO PLAYERS: the k-th cycle of each half is the SAME cycle (same figure,
    # same timings, same hold, same drift), position mirrored, at the size a
    # half allows. MEASURED by tests/test_modes.py section 35, at CLOSEST
    # APPROACH, which is the binding case.
    versus_ready = True

    # THE GHOST. A figure walks in GAP_S after the previous one leaves (dropped, or
    # walked off) and never sooner than its fall, so a perfect shooter waits for
    # each one, then needs `shots_per_figure()` tens to drop it. That chain holds
    # for ONE actor; with COUNT = 2 (one coming while another goes, OFF today) the
    # waits overlap and this cannot state a ceiling, so it says so (None) instead of
    # scoring a drill it has not modelled.
    ghost_has_waits = True

    def ghost_ceiling(self):
        return figure_ghost_ceiling() if self.COUNT == 1 else None

    def ghost_shots_per_target(self):
        return shots_per_figure() if self.COUNT == 1 else None

    def ghost_expected_wait(self, target_index):
        """The mean wait after a figure drops: a random GAP_S, but never before its
        fall is over (`advance` sets `live_at = max(live_at, fall.t1)`), averaged
        over the cast. COUNT == 2 cannot state this, as for the ceiling."""
        if self.COUNT != 1:
            return None
        return expected_wait_after_drop(self.GAP_S)

    @classmethod
    def new_link(cls) -> PlanLink:
        return PlanLink()

    def pair_with(self, link: "PlanLink | None", side: int) -> None:
        self._link = link
        self._seat = side

    def retire(self) -> None:
        """Nothing is shared that a finished half could hold up."""

    def __init__(self) -> None:
        self._link: "PlanLink | None" = None
        self._seat = 0
        self._k = 0                          # which shared cycle this half is on
        self.visible = True
        self._fall_sounds = FallSounds()      # a seedable draw, see audio/fall_sounds.py
        self._rng = random.Random()
        self._names: list[str] = []
        self._actors: list[_Actor] = []
        self._prev_name: "str | None" = None
        self._prev_hold: "tuple[float, float] | None" = None
        self._forced: "str | None" = None
        self._moving = False
        self._last_req: "float | None" = None
        self.flashes: list[tuple[float, float, float, str, bool]] = []
        # statistics
        self.shots = 0
        self.hits = 0
        self.points = 0
        self.dropped = 0                    # figures that reached the drop total
        self.no_ring = 0
        self.gap_shots = 0
        self.fall_shots = 0                 # fired while a figure lay down
        self.started_at = 0.0
        self.bands = {b: [0, 0, 0] for b in BAND_NAMES}   # shots, hits, points
        self.hold_times: list[float] = []
        self.shown: list[str] = []          # figures shown, in order (for tests)

    def moving_at(self, t: float) -> bool:
        """A figure is walking at `t`. Read from the actors' own timetables,
        so a hit (which never stops or removes a figure) cannot change it."""
        return (any(a.phase == "live" and a.moving(t) for a in self._actors)
                or any(a.fall_span is not None and a.fall_span[0] < t < a.fall_span[1]
                       for a in self._actors))

    @property
    def animated(self) -> bool:
        """True exactly when `advance` may ask for a repaint: a figure is
        moving or a hit flash is live."""
        return (self._moving or bool(self.flashes)
                or any(a.marks.flashes or a.fall is not None for a in self._actors))

    # -- lifecycle ------------------------------------------------------------

    def start(self, ctx: RenderContext, figure: "str | None" = None) -> None:
        self.reset_targets()
        self.shots = self.hits = self.points = 0
        self.dropped = 0
        self.no_ring = self.gap_shots = 0
        self.fall_shots = 0
        self.started_at = ctx.now
        self.bands = {b: [0, 0, 0] for b in BAND_NAMES}
        self.hold_times = []
        self.shown = []
        self.flashes = []
        self._moving = False
        self._last_req = None
        self._k = 0
        if figure is not None:
            self._forced = figure
        self._names = figure_names()
        self._actors = [_Actor() for _ in range(self.COUNT)]
        for i, a in enumerate(self._actors):
            self._arm(a, ctx, extra=4.0 * i)     # out of phase if there are two

    def _budget(self) -> float:
        return SINGLE_BUDGET if self.COUNT <= 1 else COMBINED_BUDGET / self.COUNT

    def _arm(self, a: _Actor, ctx: RenderContext, extra: float = 0.0) -> None:
        """Back to a blank wait: draw the next figure and load it NOW, while
        nothing is on screen, so a 60 ms JPEG read never lands mid-walk."""
        a.phase = "gap"
        a.fig = None
        a.marks.clear()                    # its holes leave with it
        if self._link is not None:
            if not self._names:
                a.live_at = ctx.now + self.GAP_S[0]
                return
            a.plan = self._plan(ctx)
            a.live_at = ctx.now + a.plan.gap + extra
            a.fig = Figure(a.plan.names[self._seat], ctx.width, ctx.height, self._budget())
            return
        a.live_at = ctx.now + self._rng.uniform(*self.GAP_S) + extra
        if not self._names:
            return
        others = {o.fig.name for o in self._actors if o.fig is not None}
        name = self._forced
        if name is None:
            prev = self._prev_name
            pool = [n for n in self._names if n not in others] or self._names
            name = catalog.pick_figure(self._rng, prev, pool)
        self._prev_name = name
        a.fig = Figure(name, ctx.width, ctx.height, self._budget())

    def _plan(self, ctx: RenderContext):
        """The shared plan for this half's next cycle, made by whichever half
        asks first: everything a cycle draws at random, with the position as
        FRACTIONS of the field in the left half's frame (the right mirrors it)."""
        k = self._k
        self._k += 1
        field = self.playfield(ctx)
        fw, fh = max(1.0, field.width()), max(1.0, field.height())

        def make(rng, prev, _idx):
            # WHEN and WHERE are shared; WHICH is per seat and never equal.
            names = catalog.pick_pair(rng, prev.names if prev else None,
                                      self._names, self._forced)
            far = catalog.MIN_MOVE_FRAC * math.hypot(fw, fh)
            u = v = 0.5
            for _ in range(40):
                u, v = rng.random(), rng.random()
                if prev is None or math.hypot((u - prev.u) * fw, (v - prev.v) * fh) >= far:
                    break
            return SimpleNamespace(
                names=names, gap=rng.uniform(*self.GAP_S),
                t_a=rng.uniform(*self.APPROACH_S), t_h=rng.uniform(*self.HOLD_NEAR_S),
                t_r=rng.uniform(*self.RETREAT_S), t_f=rng.uniform(*self.HOLD_FAR_S),
                hs=rng.random(), u=u, v=v, side=rng.choice((-1, 0, 1)),
                speed=rng.uniform(0.2, 1.0))

        return self._link.get(k, make)

    def _begin(self, a: _Actor, ctx: RenderContext) -> None:
        """Plan a whole cycle at the moment the figure appears."""
        fig = a.fig
        field = self.playfield(ctx)
        if a.plan is not None and self._link is not None:
            return self._begin_shared(a, ctx, fig, field)
        a.t_a = self._rng.uniform(*self.APPROACH_S)
        a.t_h = self._rng.uniform(*self.HOLD_NEAR_S)
        a.t_r = self._rng.uniform(*self.RETREAT_S)
        a.t_f = self._rng.uniform(*self.HOLD_FAR_S)
        a.h_near = fig.fit_height(field.height(), ctx.height, 1.0)
        a.h_far = a.h_near * self.FAR_FRAC
        a.h_start = self._rng.uniform(a.h_far, self.START_MAX_FRAC * a.h_near)
        fig.sprite.set_range(a.h_near)
        w_near, h_near = fig.sprite.drawn_size(a.h_near)
        fig.sprite.warm(a.h_start)
        fig.prepare_fall(a.h_near, ctx.height)
        a.hold_xy = catalog.pick_position(self._rng, field, w_near / 2.0,
                                          h_near / 2.0, self._prev_hold,
                                          avoid=self.menu_keepout(ctx))
        self._prev_hold = a.hold_xy
        side = self._rng.choice((-1, 0, 1))        # from left / straight / from right
        speed = self._rng.uniform(0.2, 1.0) * self.DRIFT_MAX * field.width()
        a.drift = side * speed
        a.x_lo, a.x_hi = field.left() + w_near / 2.0, field.right() - w_near / 2.0
        if a.x_hi <= a.x_lo:
            a.x_lo = a.x_hi = field.center().x()
        a.t0 = ctx.now
        a.hold_hit = False
        a.was_moving = True
        a.phase = "live"
        self.shown.append(fig.name)
        self.note_target(a.t0, a.released_at)
        a.released_at = None

    def _begin_shared(self, a: _Actor, ctx: RenderContext, fig, field) -> None:
        """`_begin` for a shared cycle: the same draws, read from the plan."""
        plan = a.plan
        a.t_a, a.t_h, a.t_r, a.t_f = plan.t_a, plan.t_h, plan.t_r, plan.t_f
        a.h_near = fig.fit_height(field.height(), ctx.height, 1.0)
        a.h_far = a.h_near * self.FAR_FRAC
        a.h_start = a.h_far + (self.START_MAX_FRAC * a.h_near - a.h_far) * plan.hs
        fig.sprite.set_range(a.h_near)
        w_near, h_near = fig.sprite.drawn_size(a.h_near)
        fig.sprite.warm(a.h_start)
        fig.prepare_fall(a.h_near, ctx.height)
        left, right = field.left() + w_near / 2.0, field.right() - w_near / 2.0
        top, bottom = field.top() + h_near / 2.0, field.bottom() - h_near / 2.0
        if right <= left or bottom <= top:
            a.hold_xy = (field.center().x(), field.center().y())
        else:
            u = 1.0 - plan.u if self._seat == 1 else plan.u        # mirrored position
            a.hold_xy = (left + (right - left) * u, top + (bottom - top) * plan.v)
        # The drift keeps its DIRECTION on both halves: the position mirrors, the
        # motion does not (same orientation, as for the artwork itself).
        a.drift = plan.side * plan.speed * self.DRIFT_MAX * field.width()
        a.x_lo, a.x_hi = field.left() + w_near / 2.0, field.right() - w_near / 2.0
        if a.x_hi <= a.x_lo:
            a.x_lo = a.x_hi = field.center().x()
        a.t0 = ctx.now
        a.hold_hit = False
        a.was_moving = True
        a.phase = "live"
        self.shown.append(fig.name)
        self.note_target(a.t0, a.released_at)
        a.released_at = None

    # -- play -----------------------------------------------------------------

    def advance(self, ctx: RenderContext) -> bool:
        changed = False
        moving = False
        for a in self._actors:
            if a.fall is not None and ctx.now >= a.fall.t1:
                a.fall = None                      # the pose goes
                changed = True
            if a.phase == "gap":
                if a.fig is not None and ctx.now >= a.live_at:
                    self._begin(a, ctx)
                    changed = True                 # it appears
            elif ctx.now >= a.t0 + a.duration:
                a.released_at = ctx.now
                self._arm(a, ctx)
                changed = True                     # it leaves
            if a.phase == "live":
                if a.marks.advance(ctx.now, self.FLASH_S):
                    changed = True             # a hit ring fading on the figure
                now_moving = a.moving(ctx.now)
                if a.was_moving and not now_moving:
                    changed = True                 # the last frame at rest
                a.was_moving = now_moving
                moving = moving or now_moving
        if self.flashes:
            self.flashes = [f for f in self.flashes
                            if ctx.now - f[2] < self.FLASH_S]
            changed = True                         # fading, or the tick that ends it
        self._moving = moving
        if moving:
            period = 1.0 / self.RATE_HZ
            if (self._last_req is None
                    or ctx.now - self._last_req >= period - 0.004):
                self._last_req = ctx.now
                changed = True
        return changed

    def _live_at(self, t: float):
        """[(actor, centre, height)] on screen at time `t`."""
        out = []
        for a in self._actors:
            if a.phase != "live":
                continue
            st = a.state(t)
            if st is not None:
                out.append((a, st[0], st[1]))
        return out

    @staticmethod
    def _band(a: _Actor, h: float) -> str:
        ratio = a.h_near / max(1.0, h)                 # 1.0 near .. FAR_RATIO far
        k = (ratio - 1.0) / (FAR_RATIO - 1.0)
        return BAND_NAMES[min(2, max(0, int(k * 3.0)))]

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
 
        t = ctx.now - self.shot_age(ev, ctx)
        if not self._live_at(t) and any(
                a.fall_span is not None and ev.t < a.fall_span[1]
                for a in self._actors):
            # Somebody is lying down and nobody is up: nothing to hit.
            self.fall_shots += 1
            return ShotOutcome(hit=False, target_id="fallen")
        if not self._live_at(t):
            # Fired into the blank between two walks. Not a miss at a figure (there
            # was none), so it does not count against hits/shots; it is SHOWN, as
            # the pop-up shows a false start, so it never looks like nothing
            # happened.
            self.gap_shots += 1
            self.flashes = (self.flashes + [(ev.x, ev.y, ev.t, "NO TARGET", False)])[-MAX_MARKS:]
            return ShotOutcome(hit=False, target_id="no-target")
        self.shots += 1
        best = None                                    # (points, -dist, ...)
        nearest = None                                 # for the band of a miss
        for a, centre, h in self._live_at(t):
            d = math.hypot(ev.x - centre[0], ev.y - centre[1])
            if nearest is None or d < nearest[0]:
                nearest = (d, a, h)
            fig = a.fig
            points, zone = fig.score(ev.x, ev.y, centre, h)
            if points > 0 and (best is None or (points, -d) > (best[0], -best[1])):
                best = (points, d, a, h, zone, centre)
        if best is not None:
            points, _d, a, h, zone, centre = best
            band = self.bands[self._band(a, h)]
            band[0] += 1
            band[1] += 1
            band[2] += points
            self.hits += 1
            self.points += points
            age = t - a.t0
            if not a.hold_hit and age >= a.t_a:
                a.hold_hit = True
                self.hold_times.append(age - a.t_a)
            name = a.fig.name
            total = a.marks.tally(points)
            dropped_now = bool(a.marks.done)
            if a.marks.done:
                # Reached the drop total: it leaves NOW, exactly as at the end of a
                # walk (marks cleared, next figure drawn and loaded while blank).
                self.dropped += 1
                self.flashes = (self.flashes + [(ev.x, ev.y, ev.t, str(total), True)])[-MAX_MARKS:]
                fig = a.fig
                record_mark()   # the flash that says why the figure dropped IS a mark
                a.released_at = ev.t
                self._arm(a, ctx)
                a.fall = Fall.start(fig, centre, h, ev.t, ctx.height)
                a.fall_span = ((a.fall.t0, a.fall.t1) if a.fall is not None else None)
                if a.fall is not None:
                    a.live_at = max(a.live_at, a.fall.t1)
            else:
                a.marks.add(a.fig, ev.x, ev.y, centre, h, ev.t, str(total), True)
            return ShotOutcome(hit=True, points=points, absorbed=True,
                               target_id=f"{name}:{zone}",
                               sound=self._fall_sounds.pick(name) if dropped_now else None)
        if nearest is not None:
            self.bands[self._band(nearest[1], nearest[2])][0] += 1
        for a, centre, h in self._live_at(t):
            if a.fig.on_figure(ev.x, ev.y, centre, h):
                self.no_ring += 1
                a.marks.add(a.fig, ev.x, ev.y, centre, h, ev.t, NO_RING, False)
                return ShotOutcome(hit=False, absorbed=True,
                                   target_id=f"{a.fig.name}:{NO_RING.replace(' ', '-')}")
        return ShotOutcome.miss()

    def aim_point(self, ctx: RenderContext) -> "tuple[float, float] | None":
        """Where `--modetest` should aim: the innermost ring of the first
        figure on screen now, None while the screen is blank."""
        live = self._live_at(ctx.now)
        if not live:
            return None
        a, centre, h = live[0]
        inner = a.fig.innermost(centre, h)
        return (inner[1], inner[2])

    def nearest(self, ev: ShotEvent, ctx: RenderContext) -> "NearestTarget | None":
        t = ctx.now - self.shot_age(ev, ctx)
        out = []
        for a, centre, h in self._live_at(t):
            inner = a.fig.innermost(centre, h)
            if inner is None:
                continue
            zone, cx, cy, r = inner
            out.append((f"{a.fig.name}:{zone}",
                        math.hypot(ev.x - cx, ev.y - cy), r))
        return self.closest(out)

    def status(self) -> str:
        if not self._names:
            return "no artwork found in targets/"
        bands = "  ".join(f"{b} {v[1]}/{v[0]}" for b, v in self.bands.items())
        out = f"{self.hits}/{self.shots} on rings   {bands}   no-ring {self.no_ring}"
        if self.gap_shots:
            out += f"   {self.gap_shots} fired with nothing up"
        if self.hold_times:
            ordered = sorted(self.hold_times)
            out += (f"   first hit after hold: median "
                    f"{ordered[len(ordered) // 2]:.1f}s")
        return out

    # -- drawing --------------------------------------------------------------

    def occluders(self, ctx: RenderContext):
        if ctx.calibrating:
            return []
        return [a.fig.occluder(self.z, centre, h)
                for a, centre, h in self._live_at(ctx.now)]

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if ctx.calibrating:
            return
        for a in self._actors:
            if a.fall is not None:
                a.fall.paint(p)                    # lying where it was shot
        for a, centre, h in self._live_at(ctx.now):
            a.fig.sprite.paint(p, centre, h)
            a.marks.paint(p, ctx, a.fig, centre, h, self.FLASH_S)
        paint_hits(p, ctx, self.flashes, self.FLASH_S)
        paint_hint(p, ctx, self.playfield(ctx), self.started_at)
