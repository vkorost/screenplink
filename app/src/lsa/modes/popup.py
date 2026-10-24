"""Mode 7: a figure pops up without warning and leaves. How fast do you hit it?

The reaction drill's shape (see `reaction.py`) with the cast as the target: a
blank screen, an unpredictable wait, then one figure at an unpredictable place,
which stays for `EXPOSURE_S` and leaves whether or not it was hit. Then the
next wait.

The reasons carried over from the reaction drill, each of which is a thing a
fixed version gets wrong:

  * THE WAIT IS RANDOM (1.2 - 3.5 s). A fixed one is trained against in a few
    repetitions and then measures counting, not reaction.
  * FIRING BEFORE IT APPEARS IS A FALSE START, reported and never scored. A
    shot is judged by its OWN time (`ev.t`), not by when it was reported: one
    confirmed after the figure appeared but fired before it is still early.
  * THE PLACE MOVES at least 35% of the playfield diagonal between figures, so
    it cannot be pre-aimed during the wait.

On top of that, what the shooter cannot predict: WHICH figure (uniform over
the twelve, never the one just shown) and HOW BIG (0.6 - 1.0 of that figure's
own size cap). All of it through `self._rng`, seedable; no module-level
`random`. The cast is variety, not information: nothing is identified and
nothing is scored on which figure it is.

The figure is chosen and its artwork loaded when the WAIT begins, while the
screen is blank, so a JPEG read never lands on the frame it appears.

WHAT IT MEASURES: time from appearance to first hit (best and median), the ring
score of those first hits, figures that left unengaged, false starts. Only
figures whose printed rings verified are ever drawn.

A HIT IS VISIBLE, and stays visible. The figure keeps standing after a hit (the
drill times the FIRST hit and the rest of the exposure is free), so the hit is
shown twice: a grey ring with the time and the ring number beside it for
`FLASH_S`, and a grey hole carried by the figure (`figure.Marks`) until it
leaves, when it goes with it. The screen's bullet-hole layer is told not to
draw a shot the figure absorbed (`ShotOutcome.absorbed`); a shot at nothing
(too early) or off the figure is not absorbed and leaves its ordinary hole.
A flash still in flight when the figure leaves goes with it.

THE FIGURE DROPS AT A RUNNING TOTAL OF 30 (`figure.DROP_TOTAL`): each scoring hit
adds its ring value, the flash and the counter beside the figure show the TOTAL,
and at 30 the exposure ends at once instead of waiting out its timer (the hit's
flash stays at the screen spot so the drop is explained). The total is the
figure's own; the next figure starts from zero.

THE FALL. A figure that has a fall pose shows it for `figure.FALL_S` (1 s) where
it stood, then it is gone; one without a pose disappears at once. The next
wait runs CONCURRENTLY with the fall (MIN_WAIT exceeds it), so the fall delays
the next figure by nothing. A shot during the fall is not a hit, not a miss and
not a false start: it leaves its ordinary screen hole and is counted only in
`fall_shots`. The mode reports as moving for that second (`moving_at`), so the
sensitivity learner refuses it.

REPAINTS. Only the two transitions and the hit flash ask for one. A figure that
is up and still, and a blank wait, ask for nothing (the reaction drill's
running-counter lesson, a third time).
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from ..artwork import catalog
from ..display.layers import GREY_FAINT, GREY_HIT, mono, px
from ..display.scene import RenderContext
from ..watchdog import record_mark
from ..audio.fall_sounds import FallSounds
from ..audio.mixer import FIGURE_APPEAR_SOUND
from .base import Mode, NearestTarget, ShotEvent, ShotOutcome, register
from .reaction import SharedReveal
from .figure import (HEAD_NOTE, MAX_MARKS, NO_RING, SINGLE_BUDGET, Fall, Figure,
                     Marks, figure_ghost_ceiling, figure_names, paint_hint,
                     paint_hits, shots_per_figure)


class PopupReveal(SharedReveal):
    """The reveal two Pop-Up halves share: ONE wait, ONE position, ONE figure and
    ONE size per appearance, so neither half can be handed an earlier figure than
    the other (the same race Reaction had, and the same ruling: "one shared
    reveal for both sides"). It is `SharedReveal` with the figure and its size
    drawn alongside the wait and the position; nothing else is copied. There is
    no timeout: progress never depends on a hit, because a figure leaves by
    itself after its exposure and each half then resolves."""

    def __init__(self, names, min_wait: float, max_wait: float, max_tries: int,
                 size_frac: "tuple[float, float]") -> None:
        super().__init__(2, min_wait, max_wait, max_tries, None)
        self.names = list(names)
        self.size_frac = size_frac
        self.pair: "tuple[str, str] | None" = None    # one figure PER SEAT
        self.frac = 1.0
        self._prev_pair: "tuple[str, str] | None" = None

    def arm(self, now: float) -> bool:
        if not super().arm(now):
            return False
        if self.names:
            # WHEN, WHERE and HOW BIG are shared; WHICH figure is per seat.
            self.pair = catalog.pick_pair(self.rng, self._prev_pair, self.names)
            self._prev_pair = self.pair
        self.frac = self.rng.uniform(*self.size_frac)
        return True


@register(key=Qt.Key_6)
class PopUpFigure(Mode):
    """A figure appears without warning. Measures time to first hit."""

    name = "popup"
    title = "Pop-Up Target"
    summary = ("A figure pops up without warning and leaves. Firing early is a false start. "
               "Head: dome = 8, its printed 9 and 7 have no ring.")
    note = HEAD_NOTE

    MIN_WAIT = 1.2
    MAX_WAIT = 3.5
    EXPOSURE_S = 2.5
    SIZE_FRAC = (0.6, 1.0)             # of the figure's own cap
    FLASH_S = 0.9
    MAX_TRIES = 40
    # TWO PLAYERS: one shared reveal (see `PopupReveal`), the same figure on both
    # halves at the size a half allows. MEASURED by tests/test_modes.py section
    # 35 and by the invariant in section 32.
    versus_ready = True

    # THE GHOST. A figure appears MIN_WAIT..MAX_WAIT after the previous one is
    # resolved (dropped, or left unengaged), and drops at a running total, so a
    # perfect shooter waits for each figure and then needs `shots_per_figure()` tens
    # to drop it. The fall runs concurrently with the next wait (MIN_WAIT exceeds
    # it), so the recorded wait already includes it.
    ghost_has_waits = True

    def ghost_ceiling(self):
        return figure_ghost_ceiling()

    def ghost_shots_per_target(self):
        return shots_per_figure()

    def ghost_expected_wait(self, target_index):
        """The mean of `_arm`'s draw from these same two constants. The fall runs
        concurrently and MIN_WAIT exceeds it, so it adds nothing (asserted by
        tests/test_modes.py section 47, not assumed)."""
        from .. import ghost
        return ghost.expected_uniform(self.MIN_WAIT, self.MAX_WAIT)

    @classmethod
    def new_link(cls) -> PopupReveal:
        return PopupReveal(figure_names(), cls.MIN_WAIT, cls.MAX_WAIT,
                           cls.MAX_TRIES, cls.SIZE_FRAC)

    def pair_with(self, link: "PopupReveal | None", side: int) -> None:
        self.seat = side
        self._reveal = link
        self._fresh = link is not None

    def retire(self) -> None:
        """This half is finished and will not be ticked again: stop being waited
        on, or the half still shooting never sees another figure."""
        if self._reveal is not None:
            self._reveal.retire(self.seat)

    @property
    def animated(self) -> bool:
        return (bool(self.flashes) or bool(self._marks.flashes)
                or self._fall is not None)

    def moving_at(self, t: float) -> bool:
        """The fall pose appearing is content the camera sees change. Answered
        from the fall's own span (kept after it ends, for late shots), never
        from what a hit did."""
        return self._fall_span is not None and self._fall_span[0] < t < self._fall_span[1]

    def __init__(self) -> None:
        self.visible = True
        self._rng = random.Random()
        self._fall_sounds = FallSounds()      # a seedable draw, see audio/fall_sounds.py
        self._names: list[str] = []
        self._forced: "str | None" = None
        self.state = "wait"                  # wait -> up -> wait
        self.next_at = 0.0
        self.shown_at = float("-inf")        # the window the figure was up
        self.left_at = float("-inf")
        self._fig: "Figure | None" = None    # the one up (or last up)
        self._centre = (0.0, 0.0)
        self._height = 0.0
        self._next_fig: "Figure | None" = None
        self._next_frac = 1.0
        self._prev_name: "str | None" = None
        self._prev_centre: "tuple[float, float] | None" = None
        self._engaged = False
        self._spent = False                  # dropped at the total: nothing left to hit
        self.dropped = 0
        self.fall_shots = 0
        self._fall: "Fall | None" = None
        self._fall_span: "tuple[float, float] | None" = None
        self.flashes: list[tuple[float, float, float, str, bool]] = []
        self._marks = Marks()                # shots the figure now up has absorbed
        self.started_at = 0.0
        self.shots = 0
        self.hits = 0
        self.false_starts = 0
        self.unengaged = 0
        self.no_ring = 0
        self.times: list[float] = []         # appearance -> first hit
        self.first_points: list[int] = []
        self.shown: list[str] = []           # figures shown, in order (tests)
        self.centres: list[tuple[float, float]] = []
        self._reveal: "PopupReveal | None" = None   # set only in a versus round
        self.seat = 0
        self._fresh = False
        self._cand_fig: "Figure | None" = None
        self._cand_h = 0.0
        # When the last figure was resolved: the instant the next wait began (the
        # dropping shot's own time, or the tick that took an unengaged figure
        # away). Consumed by the next appearance; the ghost's wait record only.
        self._released_at: "float | None" = None

    # -- lifecycle ------------------------------------------------------------

    def start(self, ctx: RenderContext, figure: "str | None" = None) -> None:
        self.reset_targets()
        self._released_at = None
        self.times = []
        self.first_points = []
        self.shots = self.hits = self.false_starts = 0
        self.unengaged = 0
        self.no_ring = 0
        self.dropped = 0
        self.fall_shots = 0
        self._fall = None
        self._fall_span = None
        self._spent = False
        self._marks = Marks()
        self.flashes = []
        self.shown = []
        self.centres = []
        self.shown_at = self.left_at = float("-inf")
        self._fig = None
        self._engaged = False
        self.started_at = ctx.now
        if figure is not None:
            self._forced = figure
        self._names = figure_names()
        if self._reveal is not None:
            self.state = "wait"
            self._cand_fig = None
            if self._fresh:
                self._fresh = False       # a new shared link: the other seat may have armed it
            else:
                self._reveal.restart()
            self._reveal.arm(ctx.now)
            return
        self._arm(ctx)

    # -- the shared reveal (two players) --------------------------------------

    def _v_centre(self, field, w: float, h: float) -> "tuple[float, float]":
        """The shared position (fractions of the field), mirrored for the right
        half, mapped into THIS half's playfield."""
        link = self._reveal
        left, right = field.left() + w / 2.0, field.right() - w / 2.0
        top, bottom = field.top() + h / 2.0, field.bottom() - h / 2.0
        if right <= left or bottom <= top:
            return (field.center().x(), field.center().y())
        nx = 1.0 - link.nx if self.seat == 1 else link.nx
        return (left + (right - left) * nx, top + (bottom - top) * link.ny)

    def _v_validate(self, ctx: RenderContext) -> None:
        """Load the shared figure now, while the wait is blank, and ACCEPT the
        shared position or veto it (it moves at least 35% of the playfield
        diagonal between figures so it cannot be pre-aimed). Bounded: after
        `max_tries` draws the link says so and this half takes what it has."""
        link = self._reveal
        field = self.playfield(ctx)
        fig = self._cand_fig
        mine = link.pair[self.seat]
        if fig is None or fig.name != mine:
            fig = Figure(mine, ctx.width, ctx.height, SINGLE_BUDGET)
        h = fig.fit_height(field.height(), ctx.height, link.frac)
        fig.sprite.set_range(h)
        fig.sprite.warm(h)
        fig.prepare_fall(h, ctx.height)
        self._cand_fig, self._cand_h = fig, h
        w, hh = fig.sprite.drawn_size(h)
        inner_w = max(1.0, field.width() - w)
        inner_h = max(1.0, field.height() - hh)
        far = catalog.MIN_MOVE_FRAC * math.hypot(inner_w, inner_h)
        for _ in range(link.max_tries + 1):
            x, y = self._v_centre(field, w, hh)
            if (self._prev_centre is None or link.exhausted
                    or math.hypot(x - self._prev_centre[0], y - self._prev_centre[1]) >= far):
                link.accept(self.seat)
                return
            link.veto()
        link.accept(self.seat)

    def _v_appear(self, ctx: RenderContext) -> None:
        link = self._reveal
        fig = self._cand_fig
        field = self.playfield(ctx)
        self._height = fig.fit_height(field.height(), ctx.height, link.frac)
        if abs(self._height - self._cand_h) > 0.5:      # geometry moved since it was loaded
            fig.sprite.set_range(self._height)
            fig.sprite.warm(self._height)
            fig.prepare_fall(self._height, ctx.height)
        w, h = fig.sprite.drawn_size(self._height)
        self._centre = self._v_centre(field, w, h)
        self._prev_centre = self._centre
        self._fig = fig
        self._cand_fig = None
        self.state = "up"
        self.shown_at = link.live_at                  # ONE origin for both halves
        self.note_target(self.shown_at, self._released_at)
        self._released_at = None
        self.left_at = link.live_at + self.EXPOSURE_S
        # BOTH halves queue the cue; the host collapses identical cues drained in
        # one tick into one (`ModeHost.take_sounds`).
        self._queued_sound = FIGURE_APPEAR_SOUND
        self._engaged = False
        self._spent = False
        self.shown.append(fig.name)
        self.centres.append(self._centre)
        self._marks = Marks()

    def _leave(self, ctx: RenderContext) -> None:
        """The figure is gone (its exposure ended, or it dropped): back to a
        blank wait. Alone: draw the next one. Shared: this half resolves, and the
        next reveal is drawn when BOTH have."""
        if self._reveal is None:
            self._arm(ctx)
            return
        self.state = "wait"
        self._reveal.resolve(self.seat)
        self._reveal.arm(ctx.now)

    def _v_advance(self, ctx: RenderContext) -> bool:
        link = self._reveal
        changed = False
        if self.state == "up" and ctx.now >= self.left_at:
            if not self._engaged:
                self.unengaged += 1
            self._marks.clear()            # shared reveal: the holes leave with it too
            self._released_at = ctx.now
            self._leave(ctx)
            changed = True
        if self._fall is not None and ctx.now >= self._fall.t1:
            self._fall = None
            changed = True
        if link.names:
            link.arm(ctx.now)
            if self.state == "wait" and link.phase == "pending":
                if self.seat not in link.accepted:
                    self._v_validate(ctx)
                if ctx.now >= link.appear_at:
                    link.try_live(ctx.now)
            if (self.state == "wait" and link.phase == "live"
                    and self.seat not in link.resolved and self._cand_fig is not None):
                self._v_appear(ctx)
                changed = True
        if self.flashes:
            self.flashes = [f for f in self.flashes if ctx.now - f[2] < self.FLASH_S]
            changed = True
        if self._marks.advance(ctx.now, self.FLASH_S):
            changed = True
        return changed

    def _arm(self, ctx: RenderContext) -> None:
        """Back to a blank wait with a fresh unpredictable delay; choose the
        next figure and its size now, and load it while nothing is on screen."""
        self.state = "wait"
        self.next_at = ctx.now + self._rng.uniform(self.MIN_WAIT, self.MAX_WAIT)
        self._next_fig = None
        if not self._names:
            return
        name = self._forced or catalog.pick_figure(self._rng, self._prev_name,
                                                   self._names)
        self._prev_name = name
        self._next_frac = self._rng.uniform(*self.SIZE_FRAC)
        self._next_fig = Figure(name, ctx.width, ctx.height, SINGLE_BUDGET)

    def _appear(self, ctx: RenderContext) -> None:
        fig = self._next_fig
        field = self.playfield(ctx)
        self._height = fig.fit_height(field.height(), ctx.height, self._next_frac)
        fig.sprite.set_range(self._height)
        fig.sprite.warm(self._height)
        fig.prepare_fall(self._height, ctx.height)
        w, h = fig.sprite.drawn_size(self._height)
        self._centre = catalog.pick_position(self._rng, field, w / 2.0, h / 2.0,
                                             self._prev_centre,
                                             avoid=self.menu_keepout(ctx))
        self._prev_centre = self._centre
        self._fig = fig
        self.state = "up"
        self.shown_at = ctx.now
        self.note_target(self.shown_at, self._released_at)
        self._released_at = None
        # THE SURPRISE IS THE DRILL, and he asked for the figure to announce
        # itself the instant it is up (2026-10-18) -- the same moment the
        # time-to-first-hit clock starts.
        self._queued_sound = FIGURE_APPEAR_SOUND
        self.left_at = ctx.now + self.EXPOSURE_S
        self._engaged = False
        self._spent = False
        self.shown.append(fig.name)
        self.centres.append(self._centre)
        self._marks = Marks()

    # -- play -----------------------------------------------------------------

    def advance(self, ctx: RenderContext) -> bool:
        if self._reveal is not None:
            return self._v_advance(ctx)
        changed = False
        if self.state == "wait" and self._next_fig is not None \
                and ctx.now >= self.next_at:
            self._appear(ctx)
            changed = True
        elif self.state == "up" and ctx.now >= self.left_at:
            if not self._engaged:
                self.unengaged += 1
            self._marks.clear()            # the holes go with the figure
            self._released_at = ctx.now
            self._arm(ctx)
            changed = True
        if self._fall is not None and ctx.now >= self._fall.t1:
            self._fall = None              # the pose goes: repaint to erase it
            changed = True
        if self.flashes:
            self.flashes = [f for f in self.flashes
                            if ctx.now - f[2] < self.FLASH_S]
            changed = True
        if self._marks.advance(ctx.now, self.FLASH_S):
            changed = True
        return changed

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        if self._fall_span is not None and ev.t < self._fall_span[1]:
            # The figure is down. Nothing to hit, nothing to be early for.
            self.fall_shots += 1
            return ShotOutcome(hit=False, target_id="fallen")
        self.shots += 1
        fig = self._fig
        if fig is None or not (self.shown_at <= ev.t <= self.left_at):
            # Fired before the figure existed (or after it had gone): a shot at
            # nothing, reported and never scored.
            self.false_starts += 1
            self.flashes = (self.flashes + [(ev.x, ev.y, ev.t, "TOO EARLY", False)])[-MAX_MARKS:]
            return ShotOutcome(hit=False, target_id="false-start")
        if self._spent:
            # An earlier shot still arriving for a figure that has already gone
            # (the second pistol, or a late confirmation): nothing is left to hit.
            return ShotOutcome.miss()
        points, zone = fig.score(ev.x, ev.y, self._centre, self._height)
        if points <= 0:
            if fig.on_figure(ev.x, ev.y, self._centre, self._height):
                self.no_ring += 1
                self._marks.add(fig, ev.x, ev.y, self._centre, self._height,
                                ev.t, NO_RING, False)
                return ShotOutcome(hit=False, absorbed=True,
                                   target_id=f"{fig.name}:{NO_RING.replace(' ', '-')}")
            return ShotOutcome.miss()
        self.hits += 1
        total = self._marks.tally(points)
        label = str(total)
        if not self._engaged:                  # the first hit is the measurement
            self._engaged = True
            took = max(0.0, ev.t - self.shown_at)
            self.times.append(took)
            self.first_points.append(points)
            label = f"{took * 1000:.0f} ms   {total}"
        dropped_now = bool(self._marks.done)
        if self._marks.done:
            # Reached the drop total: the exposure ends NOW. The marks go with the
            # figure; the flash that says why is kept at the screen spot.
            self.dropped += 1
            self._spent = True
            self.left_at = ev.t
            self.flashes = (self.flashes + [(ev.x, ev.y, ev.t, label, True)])[-MAX_MARKS:]
            record_mark()   # the flash that says why the figure dropped IS a mark
            self._marks.clear()
            self._fall = Fall.start(fig, self._centre, self._height, ev.t, ctx.height)
            self._fall_span = ((self._fall.t0, self._fall.t1)
                               if self._fall is not None else None)
            self._released_at = ev.t
            self._leave(ctx)
        else:
            self._marks.add(fig, ev.x, ev.y, self._centre, self._height,
                            ev.t, label, True)
        return ShotOutcome(hit=True, points=points, absorbed=True,
                           target_id=f"{fig.name}:{zone}",
                           sound=self._fall_sounds.pick(fig.name) if dropped_now else None)

    def aim_point(self, ctx: RenderContext) -> "tuple[float, float] | None":
        """Where `--modetest` should aim: the innermost ring's centre while
        the figure is up, None while the screen is blank."""
        if self.state != "up" or self._fig is None:
            return None
        inner = self._fig.innermost(self._centre, self._height)
        return (inner[1], inner[2])

    def nearest(self, ev: ShotEvent, ctx: RenderContext) -> "NearestTarget | None":
        fig = self._fig
        if fig is None or not (self.shown_at <= ev.t <= self.left_at):
            return None
        inner = fig.innermost(self._centre, self._height)
        if inner is None:
            return None
        zone, cx, cy, r = inner
        return (f"{fig.name}:{zone}", math.hypot(ev.x - cx, ev.y - cy), r)

    def status(self) -> str:
        if not self._names:
            return "no artwork found in targets/"
        fs = (f"   {self.false_starts} false start"
              f"{'s' if self.false_starts != 1 else ''}"
              if self.false_starts else "")
        if not self.times:
            return ("wait for it..." if self.state == "wait" else "GO") + fs
        ordered = sorted(self.times)
        out = (f"{len(self.times)} first hits   best {ordered[0] * 1000:.0f}ms"
               f"   median {ordered[len(ordered) // 2] * 1000:.0f}ms"
               f"   mean ring {sum(self.first_points) / len(self.first_points):.1f}")
        if self.unengaged:
            out += f"   {self.unengaged} left unengaged"
        return out + fs

    # -- drawing --------------------------------------------------------------

    def occluders(self, ctx: RenderContext):
        if ctx.calibrating or self.state != "up" or self._fig is None:
            return []
        return [self._fig.occluder(self.z, self._centre, self._height)]

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if ctx.calibrating:
            return
        field = self.playfield(ctx)
        if self.state == "up" and self._fig is not None:
            self._fig.sprite.paint(p, self._centre, self._height)
            self._marks.paint(p, ctx, self._fig, self._centre, self._height,
                              self.FLASH_S)
        elif self._fall is not None:
            self._fall.paint(p)            # lying where it was shot
        else:
            # Deliberately almost nothing while waiting: a countdown would be
            # something to time the shot against.
            p.setPen(QPen(QColor(GREY_FAINT)))
            p.setFont(mono(px(ctx, 20)))
            p.drawText(int(field.center().x() - px(ctx, 80)),
                       int(field.center().y()), "wait for it")
        paint_hits(p, ctx, self.flashes, self.FLASH_S)
        paint_hint(p, ctx, field, self.started_at,
                   ("a figure pops up without warning - firing early is a false start, "
                    "not a hit",))
