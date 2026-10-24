"""One target, appearing without warning. How fast can you hit it?

The three modes before this one all measure the same thing in different
clothes: how quickly you can work through targets that are already on screen.
None of them measures the interval a shooter actually trains hardest -- the
one between a stimulus appearing and the first shot breaking.

So: a blank screen, a wait of unpredictable length, then a single target.
The clock starts the instant it is drawn and stops when it is hit.

Three details that make it a drill rather than a toy, each of them standard in
the sport and each of them a thing the other modes do not do:

- **The delay is random**, 1.2 to 3.5 seconds. A fixed delay is trained
  against within a few repetitions: the shooter learns the rhythm and starts
  anticipating, which measures their counting rather than their reaction.
- **Firing before the target appears is a FALSE START** and is reported as
  one. It is not scored as a hit when the target then arrives, because a shot
  that left early would have been a shot at nothing. Counting it would reward
  exactly the habit this drill exists to break.
- **The target moves between repetitions.** Hitting the same spot repeatedly
  measures memory, not reaction, and lets the shooter pre-aim during the wait.

What it reports is a time, not a score, because a time is the thing a shooter
can compare with last week. Best and median of the session, and the median is
the honest one: a single lucky repetition is not a skill.

TWO PLAYERS: ONE REVEAL, SHARED. Left alone, two copies of this drill each draw
their own wait, so one player sees his target up to 2.3 seconds before the other
and the faster reveal wins the round: the drill would measure luck. His
decision (2026-10-19): "One shared reveal for both sides."

So the reveal -- the instant it appears and where -- is a `SharedReveal` that
the two halves read. Single player holds a one-seat one and runs the SAME code,
so the versus path cannot rot unseen. Everything geometric stays in the drill:
the object holds an RNG, a time, two numbers in 0..1 and bookkeeping, and never
sees a rectangle. Each half maps the position into its own field (the right
half mirrored in x, so the pair is symmetric about the divider) and may VETO it
-- on the MENU keep-out, or too near its own last target -- which redraws the
position for both, keeping the time, while nothing is yet on screen.

NO REVEAL MAY HANG. The next one waits for every seat to resolve the last, so a
player who never hits would block his partner for ever. His own ruling on the
round ("just count the shots that had been registered") is the same principle:
progress must not depend on a hit. With two seats a reveal times out after
`REVEAL_TIMEOUT_S`; a seat still unresolved is resolved TOO SLOW -- counted and
shown, never a hit and never a false start. With one seat there is no timeout:
single player waits for the hit, exactly as it always has.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from ..display.layers import (GREY_HI as ACCENT, GREY_HI_DIM as ACCENT_DIM,
                              GREY_MISS as COOL, GREY_HIT as GOOD, GREY_FAINT, GREY_TEXT, mono, px)
from ..display.scene import RenderContext
from ..audio.mixer import APPEAR_SOUND
from .. import ghost
from .base import Mode, ShotEvent, ShotOutcome, register


def reaction_points(took: float) -> int:
    """What a hit `took` seconds after the target appeared is worth: 10 for an
    instant one, falling by 6 a second, never under 1. ONE definition, read by the
    scorer (`ReactionDrill.shot`) and by the ghost's ceiling, which is this at a
    reaction time of zero."""
    return max(1, int(round(10.0 - took * 6.0)))


class SharedReveal:
    """One reveal, read by every seat. PURE: no Qt, no rectangle, no screen size.

    A reveal goes `idle -> pending -> live -> idle`:

    - `idle`: nothing is pending and every seat has resolved the last one.
    - `pending`: a wait and a position are drawn; the appearance time is set.
      Each seat VALIDATES the position (accept) or vetoes it (redraw).
    - `live`: every seat has accepted and the appearance time has come. The
      instant is recorded ONCE, so both seats time from the same origin.

    The random draws happen in the same order a lone drill always made them --
    wait, then x, then y, then (on a rejection) x, y again -- so a single seat
    reproduces the pre-versus sequence for a given seed.
    """

    def __init__(self, seats: int, min_wait: float, max_wait: float,
                 max_tries: int, timeout_s: "float | None") -> None:
        self.seats = int(seats)
        self.min_wait, self.max_wait = float(min_wait), float(max_wait)
        self.max_tries = int(max_tries)
        # None: never expires. Single player waits for the hit.
        self.timeout_s = timeout_s if self.seats > 1 else None
        self.rng = random.Random()
        self.phase = "idle"
        self.appear_at = 0.0
        self.nx = self.ny = 0.5
        self.attempts = 0                 # position draws made for this reveal
        self.exhausted = False            # max_tries rejected: seats fall back
        self.live_at = 0.0                # the one instant both seats time from
        self.reveal_no = 0                # reveals drawn so far
        self.accepted: set[int] = set()
        self.resolved: set[int] = set(range(self.seats))   # nothing outstanding
        self.expired: set[int] = set()    # timed out and not yet acknowledged
        self.retired: set[int] = set()    # halves that are done: never waited on

    def restart(self) -> None:
        """Forget the reveal in flight (a restart). The RNG stream continues."""
        self.phase = "idle"
        self.accepted = set()
        self.resolved = set(range(self.seats))
        self.expired = set()
        self.exhausted = False
        self.attempts = 0

    def retire(self, seat: int) -> None:
        """A seat whose half is DONE will never answer again. It counts as
        having accepted and resolved every reveal from now on, so the seat
        still shooting is not waited on forever."""
        self.retired.add(seat)
        self.expired.discard(seat)
        if self.phase == "live" and len(self.resolved | self.retired) >= self.seats:
            self.phase = "idle"

    def arm(self, now: float) -> bool:
        """Draw the next reveal. Idempotent: it draws only when nothing is
        pending AND every seat has resolved the last one."""
        if self.phase != "idle" or len(self.resolved | self.retired) < self.seats:
            return False
        self.appear_at = now + self.rng.uniform(self.min_wait, self.max_wait)
        self.nx, self.ny = self.rng.random(), self.rng.random()
        self.attempts = 1
        self.exhausted = False
        self.accepted = set()
        self.resolved = set()
        # `expired` is NOT cleared here: a seat times out in the same tick the
        # next reveal is drawn, and it has not been told yet. Looking at the
        # first render of this exposed it -- the right half's target stayed up
        # while the left half flashed TOO SLOW.
        self.reveal_no += 1
        self.phase = "pending"
        return True

    def accept(self, seat: int) -> None:
        self.accepted.add(seat)

    def veto(self) -> None:
        """A seat rejects the position. Redraw it for EVERY seat, keeping the
        appearance time; after `max_tries` draws stop redrawing and let each
        seat fall back to its own furthest free corner."""
        self.accepted = set()
        if self.attempts >= self.max_tries:
            self.exhausted = True
            return
        self.nx, self.ny = self.rng.random(), self.rng.random()
        self.attempts += 1

    def try_live(self, now: float) -> bool:
        if (self.phase == "pending" and now >= self.appear_at
                and len(self.accepted | self.retired) >= self.seats):
            self.phase = "live"
            self.live_at = now
            return True
        return False

    def resolve(self, seat: int) -> None:
        self.resolved.add(seat)
        if self.phase == "live" and len(self.resolved | self.retired) >= self.seats:
            self.phase = "idle"

    def expire(self, now: float) -> None:
        """Time out a live reveal (two seats only): every seat still unresolved
        is resolved TOO SLOW, to be acknowledged by that seat."""
        if (self.timeout_s is None or self.phase != "live"
                or now - self.live_at < self.timeout_s):
            return
        for seat in range(self.seats):
            if seat not in self.resolved and seat not in self.retired:
                self.expired.add(seat)
                self.resolved.add(seat)
        self.phase = "idle"

    def take_expired(self, seat: int) -> bool:
        if seat in self.expired:
            self.expired.discard(seat)
            return True
        return False


@register(key=Qt.Key_4)
class ReactionDrill(Mode):
    """A target appears without warning. Measures time to first hit."""

    name = "reaction"
    title = "Reaction"
    summary = "A target appears without warning. How fast can you hit it?"
    animated = True            # the countdown and the flash both need ticking

    MIN_WAIT = 1.2             # seconds; shorter is unfair at 100ms pulse rates
    MAX_WAIT = 3.5             # longer than this and attention wanders
    RADIUS_FRAC = 0.055        # of the smaller playfield side
    # Far enough that the next target cannot be pre-aimed from where the last
    # one was. Measured as a fraction of the playfield diagonal.
    MIN_MOVE_FRAC = 0.35
    FLASH_S = 0.45
    MAX_TRIES = 40             # position draws before falling back to a corner
    # Versus only. His measured cadence is about one shot a second, so three
    # seconds is three shot opportunities at a target 0.055 of the field's
    # smaller side. A player still unresolved then was too slow, and the round
    # goes on without him (see the module docstring).
    REVEAL_TIMEOUT_S = 3.0
    versus_ready = True

    # THE GHOST. A target appears MIN_WAIT..MAX_WAIT seconds after the previous
    # one is resolved, so a perfect shooter waits for it too: that wait is the
    # dominant cost in this drill and the reason its score read 47 before the
    # waits were supplied. The ceiling is `reaction_points` at zero reaction time
    # (the ghost's own one frame of seeing it is charged on the time axis, in
    # `ghost.available_from_waits`, and rounds to the same 10).
    ghost_has_waits = True

    def ghost_ceiling(self):
        return ghost.constant_ceiling(reaction_points(0.0))

    def ghost_expected_wait(self, target_index):
        """The mean of the draw `SharedReveal` makes from these same two constants:
        what the ghost waits for a target the round never drew (he missed)."""
        return ghost.expected_uniform(self.MIN_WAIT, self.MAX_WAIT)

    def __init__(self) -> None:
        self.visible = True
        self.times: list[float] = []
        self.shots = 0
        self.hits = 0
        self.false_starts = 0
        self.too_slow = 0                # versus only: timed out, never a hit
        self.state = "wait"          # wait -> live -> wait
        self.shown_at = 0.0          # when it did appear (the reveal's one instant)
        self.x = self.y = 0.0
        self.r = 10.0
        self.flashes: list[tuple[float, float, float, float]] = []
        self.started_at = 0.0
        # ALWAYS a reveal object, one seat by default, so single player and
        # versus run the SAME code and the shared path cannot rot unseen.
        self.seat = 0
        self._fresh = False              # just paired: keep the link, do not restart it
        self._cand = (0.0, 0.0)          # this seat's accepted position, not yet live
        self._cand_r = 10.0
        self._cand_key: tuple = ()       # the geometry it was accepted against
        self._late_grace = False         # the first shot after an expiry is forgiven
        self._shown_before = False       # has ANY target been shown to this seat since start
        # When this seat's last target was resolved (a hit at the shot's own time, a
        # versus timeout at the tick's): the instant its next wait began. Consumed
        # by the next appearance. For the ghost's wait record only.
        self._released_at: "float | None" = None
        self._reveal = self.new_link_for(1)

    # -- the shared reveal ----------------------------------------------------

    @classmethod
    def new_link_for(cls, seats: int) -> SharedReveal:
        return SharedReveal(seats, cls.MIN_WAIT, cls.MAX_WAIT, cls.MAX_TRIES,
                            cls.REVEAL_TIMEOUT_S)

    @classmethod
    def new_link(cls) -> SharedReveal:
        """The object the two halves of a versus round share: two seats."""
        return cls.new_link_for(2)

    def retire(self) -> None:
        """This half is done and will not be ticked again: stop being waited on."""
        self._reveal.retire(self.seat)

    def pair_with(self, link: "SharedReveal | None", side: int) -> None:
        self.seat = side
        if link is not None:
            self._reveal = link
            self._fresh = True

    @property
    def _rng(self) -> random.Random:
        return self._reveal.rng

    @property
    def next_at(self) -> float:
        """When the pending target appears."""
        return self._reveal.appear_at

    # -- lifecycle ------------------------------------------------------------

    def start(self, ctx: RenderContext) -> None:
        self.times = []
        self.shots = self.hits = self.false_starts = self.too_slow = 0
        self.flashes = []
        self.started_at = ctx.now
        self.state = "wait"
        self._shown_before = False
        self._late_grace = False
        self.reset_targets()
        self._released_at = None
        if self._fresh:
            self._fresh = False          # a new shared link: the other seat may have armed it
        else:
            self._reveal.restart()       # single player: a fresh wait, as ever
        self._reveal.arm(ctx.now)

    def _geometry(self, ctx: RenderContext):
        """(inner rect, minimum move, radius) for THIS seat. All geometry lives
        here, and it changes nothing: the radius is committed only when the
        target goes live."""
        field = self.playfield(ctx)
        r = max(10.0, self.RADIUS_FRAC * min(field.width(), field.height()))
        inner = field.adjusted(r, r, -r, -r)
        if inner.width() <= 0 or inner.height() <= 0:
            return field, None, r
        return inner, self.MIN_MOVE_FRAC * math.hypot(inner.width(), inner.height()), r

    def _geometry_key(self, ctx: RenderContext) -> tuple:
        """What a candidate position depends on, to notice it has changed."""
        inner, _far, r = self._geometry(ctx)
        k = self.menu_keepout(ctx)
        return (inner.left(), inner.top(), inner.right(), inner.bottom(), r,
                k.left(), k.top(), k.right(), k.bottom())

    def _validate(self, ctx: RenderContext) -> None:
        """Map the shared position into this seat's field and ACCEPT it, or
        veto it and look at the redrawn one. Nothing is on screen yet, so a
        redraw is invisible. Bounded: after `MAX_TRIES` draws the link says
        so and the seat takes the furthest free corner, as it always did."""
        link = self._reveal
        inner, far, r = self._geometry(ctx)
        self._cand_r = r
        self._cand_key = self._geometry_key(ctx)
        if far is None:
            self._cand = (inner.center().x(), inner.center().y())
            link.accept(self.seat)
            return
        last = (self.x, self.y)
        while True:
            if link.exhausted:
                self._cand = self._corner(ctx, inner, last, r)
                link.accept(self.seat)
                return
            nx = 1.0 - link.nx if self.seat == 1 else link.nx   # mirrored about the divider
            x = inner.left() + (inner.right() - inner.left()) * nx
            y = inner.top() + (inner.bottom() - inner.top()) * link.ny
            if (self.clear_of_menu(ctx, x, y, r)
                    and (not self._shown_before
                         or math.hypot(x - last[0], y - last[1]) >= far)):
                self._cand = (x, y)
                link.accept(self.seat)
                return
            # The move rule is keyed on whether a target was ever SHOWN to this
            # seat, not on whether one was HIT: with two seats a reveal can time
            # out, and two timeouts in a row must not put the target back where
            # it was (it could be pre-aimed). The shared position answers to the
            # STRICTEST seat, since either may veto; that asymmetry is correct.
            link.veto()

    def _corner(self, ctx: RenderContext, inner, last, r: float):
        # Nothing far enough: the playfield is small, so take the furthest
        # corner from the last target rather than giving up and letting it
        # reappear where the shooter is already aiming.
        corners = [(inner.left(), inner.top()), (inner.right(), inner.top()),
                   (inner.right(), inner.bottom()), (inner.left(), inner.bottom())]
        # One of the four IS the MENU button's corner. Drop it, unless that
        # leaves nothing -- a target somewhere beats no target at all, and the
        # button yields to the drill on a shot anyway.
        free = [c for c in corners if self.clear_of_menu(ctx, c[0], c[1], r)]
        corners = free or corners
        return max(corners, key=lambda c: math.hypot(c[0] - last[0], c[1] - last[1]))

    # -- play -----------------------------------------------------------------

    def _sync(self, ctx: RenderContext, at: "float | None" = None) -> bool:
        """Bring this seat in line with the shared reveal. True: it changed.

        `at` is the instant to judge the timeout at: the tick's clock from
        `advance`, but a shot's OWN timestamp from `shot` -- `ShotEvent.t` is
        when the pulse broke and is earlier than the tick that handles it, and
        judging at the tick would charge a hit at 2.97 s as too slow and then a
        false start (the ball drills made this mistake with position). A reveal
        that expired in an EARLIER tick cannot be un-expired; that bounds the
        error to one tick."""
        link = self._reveal
        when = ctx.now if at is None else at
        link.expire(when)
        changed = False
        if link.take_expired(self.seat):
            # Timed out with the target still up: TOO SLOW. Counted and shown;
            # it is neither a hit nor a false start (a false start is a
            # different fault and must not be charged to a man who was
            # merely slow).
            if self.state == "live":
                self.too_slow += 1
                self.flashes.append((self.x, self.y, when, -1.0))
                self._late_grace = True
                self._released_at = when
            self.state = "wait"
            changed = True
        if (self.state == "wait" and link.phase == "live"
                and self.seat not in link.resolved):
            self.x, self.y = self._cand
            self.r = self._cand_r
            self.state = "live"
            self.shown_at = link.live_at       # ONE origin for both seats
            # The ghost's record: hittable from this instant, on the clock the
            # shots use, having been waited for since `_released_at`.
            self.note_target(self.shown_at, self._released_at)
            self._released_at = None
            self._shown_before = True
            self._late_grace = False       # forgiveness never reaches the next reveal
            # THE SURPRISE IS THE DRILL. He asked for a sound the moment
            # it appears, 2026-10-18 -- the same instant the clock starts.
            # BOTH seats queue it and the HOST collapses identical cues drained
            # in one tick into one (`ModeHost.take_sounds`): when only the left
            # seat queued it, the right half's reveals were silent once the left
            # half had finished and was no longer the one being drained.
            self._queued_sound = APPEAR_SOUND
            changed = True
        return changed

    def advance(self, ctx: RenderContext) -> bool:
        link = self._reveal
        busy = self._sync(ctx)
        link.arm(ctx.now)
        if self.state == "wait" and link.phase == "pending":
            # VALIDATE WHILE WAITING, not at the instant of the reveal. Both
            # seats must have accepted BEFORE the appearance time, or the seat
            # ticked first would learn of it a frame (16.7 ms) after the
            # other while timing from the same origin -- a small, systematic
            # edge for one side. Found by a check that both halves go live in
            # the same tick. The draws still happen in the order a lone drill
            # always made them (wait, x, y, then redraws), so the random
            # stream is unchanged.
            if self.seat not in link.accepted:
                self._validate(ctx)
            if ctx.now >= link.appear_at:
                if self._geometry_key(ctx) != self._cand_key:
                    # The field or the MENU keep-out moved since this was
                    # accepted (a calibration landing mid-wait): judge again,
                    # as a lone drill judging at the instant always did.
                    link.accepted.discard(self.seat)
                    self._validate(ctx)
                link.try_live(ctx.now)
                busy = self._sync(ctx) or busy
        if self.flashes:
            before = len(self.flashes)
            self.flashes = [f for f in self.flashes
                            if ctx.now - f[2] < self.FLASH_S]
            busy = busy or len(self.flashes) != before or bool(self.flashes)
        # Only transitions and the fading flash need a repaint. Nothing moves
        # while the target is up, and nothing at all moves while waiting.
        #
        # This mode originally drew a running millisecond counter, which made
        # `advance` report a change on every tick and so repainted a 3840x2160
        # screen sixty times a second -- the exact fault found in the HUD
        # earlier today, at 8.53 ms a paint against a detector that needs 6.96
        # ms of the same CPU per camera frame. It was also the wrong drill: a
        # counter gives the shooter a number to watch instead of the target,
        # and a real shot timer reports the time AFTER the shot, not during.
        return busy

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        self._sync(ctx, at=ev.t)
        self.shots += 1
        if self.state != "live" and self._late_grace:
            # THE FIRST SHOT AFTER AN EXPIRY IS FORGIVEN. He had a shot in
            # flight when the app took his target away; charging that as a
            # false start would tell him he fired EARLY when he was late, and a
            # wrong diagnosis teaches a flinch he does not have. It still
            # counts as a registered shot and scores nothing. No constant: any
            # FURTHER shot in the same wait is anticipation of the next
            # reveal, and is a genuine false start.
            self._late_grace = False
            self.flashes.append((ev.x, ev.y, ev.t, -2.0))
            return ShotOutcome(hit=False, target_id="too-late")
        if self.state != "live":
            # Fired before the target existed. Reported, never scored: a shot
            # that leaves early is a shot at nothing, and counting it as a hit
            # when the target arrives rewards anticipation.
            self.false_starts += 1
            self.flashes.append((ev.x, ev.y, ev.t, 0.0))
            return ShotOutcome(hit=False, target_id="false-start")
        if math.hypot(ev.x - self.x, ev.y - self.y) > self.r:
            return ShotOutcome.miss()
        took = max(0.0, ev.t - self.shown_at)
        self.times.append(took)
        self.hits += 1
        self.flashes.append((self.x, self.y, ev.t, took))
        self.state = "wait"
        self._released_at = ev.t
        self._reveal.resolve(self.seat)
        # With one seat this draws the next wait at once, as it always did;
        # with two it draws only when the other seat has resolved too.
        self._reveal.arm(ctx.now)
        # Points fall with time: 10 for a quarter-second, 1 for over a second
        # and a half. The number is decoration -- the time is the measurement.
        points = reaction_points(took)
        return ShotOutcome(hit=True, points=points, target_id="reaction")

    def status(self) -> str:
        if not self.times:
            waiting = "wait for it..." if self.state == "wait" else "GO"
            extra = (f"   {self.false_starts} false start"
                     f"{'s' if self.false_starts != 1 else ''}"
                     if self.false_starts else "")
            if self.too_slow:
                extra += f"   {self.too_slow} too slow"
            return f"{waiting}{extra}"
        best = min(self.times)
        ordered = sorted(self.times)
        median = ordered[len(ordered) // 2]
        out = (f"{self.hits} hit{'s' if self.hits != 1 else ''}   "
               f"best {best * 1000:.0f}ms   median {median * 1000:.0f}ms")
        if self.false_starts:
            out += (f"   {self.false_starts} false start"
                    f"{'s' if self.false_starts != 1 else ''}")
        if self.too_slow:
            out += f"   {self.too_slow} too slow"
        return out

    # -- drawing --------------------------------------------------------------

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        field = self.playfield(ctx)

        for fx, fy, ft, took in self.flashes:
            age = (ctx.now - ft) / self.FLASH_S
            if age >= 1.0:
                continue
            # A false start carries took == 0.0 and is drawn cool rather than
            # green, so it never reads as a success from the firing line. A
            # timed-out target (took < 0) is cool for the same reason.
            c = QColor(GOOD if took > 0.0 else COOL)
            c.setAlphaF(max(0.0, 1.0 - age))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(c, max(2.0, ctx.height * 0.004)))
            grow = ctx.height * 0.02 * (1.0 + age * 2.0)
            p.drawEllipse(QPointF(fx, fy), grow, grow)
            # The time, where the shot landed, once -- the way a shot timer
            # reports it. Nothing is shown while the target is up.
            label = (f"{took * 1000:.0f} ms" if took > 0.0
                     else "TOO LATE" if took < -1.5
                     else "TOO SLOW" if took < 0.0 else "TOO EARLY")
            p.setFont(mono(px(ctx, 26), bold=True))
            p.setPen(QPen(c))
            p.drawText(int(fx + grow + px(ctx, 10)),
                       int(fy + px(ctx, 9)), label)

        if self.state == "live":
            # Two rings rather than a disc: the centre stays dark, so the
            # camera sees the laser against black rather than against a lit
            # patch, and the shooter still has an unambiguous centre to aim at.
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(ACCENT), max(3.0, self.r * 0.16)))
            p.drawEllipse(QPointF(self.x, self.y), self.r, self.r)
            p.setPen(QPen(QColor(ACCENT_DIM), max(2.0, self.r * 0.08)))
            p.drawEllipse(QPointF(self.x, self.y), self.r * 0.45, self.r * 0.45)
            tick = self.r * 0.3
            p.drawLine(QPointF(self.x - tick, self.y), QPointF(self.x + tick, self.y))
            p.drawLine(QPointF(self.x, self.y - tick), QPointF(self.x, self.y + tick))

        else:
            # Deliberately almost nothing while waiting. A countdown would be
            # something to time the shot against, which is the one thing this
            # drill must not provide.
            p.setPen(QPen(QColor(GREY_FAINT)))
            p.setFont(mono(px(ctx, 20)))
            p.drawText(int(field.center().x() - px(ctx, 80)),
                       int(field.center().y()), "wait for it")

        if ctx.now - self.started_at < 8.0:
            p.setPen(QPen(QColor(GREY_TEXT)))
            p.setFont(mono(px(ctx, 14)))
            p.drawText(int(field.left()) + 8, int(field.bottom()) - px(ctx, 16),
                       "a target appears without warning - firing early is a "
                       "false start, not a hit")
