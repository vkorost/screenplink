"""Mode 8: one figure on screen, scored by its own rings.

Asked for as: "if I give you a grayscale image of a target, could you scale it
on screen?" and "could you display these images in one of the modes". The
figures live in `targets/` (twelve of them). This is the static one: a figure
is drawn, he shoots it, the rings that could be fitted to the artwork score.

THE CAST IS VARIETY, NOT INFORMATION. He said he will shoot all of them exactly
the same way, and that they should appear at random: the figure is drawn
uniformly from the twelve EXCLUDING the one just shown (an immediate repeat
reads from the firing line as the app being stuck), placed at a random centre
at least 35% of the playfield diagonal from where the last one stood, and every
draw goes through `self._rng` so a test can seed it. Nothing is scored on which
figure it is.

SIZE IS CAPPED, NOT DIMMED. A figure is drawn at the height at which it lights
`SINGLE_BUDGET` (1.5%) of the screen above luminance 60, per image, because the
twelve differ 3.2x in lit area at the same size. His artwork is never dimmed
(`INK_SCALE` 1.0): he said a figure at ~60% of the screen height is big enough.

SCORING. What scores is what is drawn: the printed rings, extracted from the
ink as polygons (see `lsa.artwork.zones`). A figure whose rings did not verify
is never drawn at all (`lsa.artwork.catalog.excluded`). What this measures is
the zone distribution of a string, its score, and its time (first shot to
last), the standard paper-target readout.

A HIT IS VISIBLE: a grey ring where it landed with its ring number beside it,
for `FLASH_S`, and a grey hole that STAYS on the figure. A shot on the figure
is carried by it in the figure's own coordinates (`figure.Marks`) and the
screen's bullet-hole layer is told not to draw it (`ShotOutcome.absorbed`); a
shot that misses the figure leaves the ordinary screen hole. A new round is a
new figure and takes its holes with it.

THE FIGURE DROPS AT A RUNNING TOTAL OF 30 (`figure.DROP_TOTAL`). Each scoring
hit adds its ring value; a hit's flash and the counter beside the figure show the
TOTAL, not that hit's ring. At 30 the figure goes at once and the next one
appears (a new figure at a new place, the round's tally carried on); the last
hit's flash stays at the spot where it landed so the drop is explained.

THE FALL. A figure with a fall pose shows it for `figure.FALL_S` (1 s) where it
stood, and the next figure appears when that second is over (the fall IS the
delay: exactly one second, no more). A figure without a pose is replaced at
once, as before. A shot during the fall is not a hit and not a miss: it leaves
its ordinary screen hole and is counted only in `fall_shots`. The mode reports
as moving for that second (`moving_at`), so the sensitivity learner refuses it.

REPAINTS. Static, so none while idle; only the hit flash animates.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen

from ..artwork import catalog
from ..display.layers import GREY_MID, mono, px
from ..display.scene import RenderContext
from ..watchdog import record_mark
from ..audio.fall_sounds import FallSounds
from .base import Mode, NearestTarget, ShotEvent, ShotOutcome, register
from types import SimpleNamespace

from .figure import (HEAD_NOTE, MAX_MARKS, NO_RING, SINGLE_BUDGET, Fall, Figure,
                     Marks, PlanLink, expected_wait_after_drop,
                     figure_ghost_ceiling, figure_names,
                     paint_hint, paint_hits, shots_per_figure)


@register(key=Qt.Key_7)
class ImageTarget(Mode):
    """One figure, scored by ring."""

    name = "image"
    title = "Static Target"
    summary = ("A figure at a random place. Rings score; a new round is a new figure. "
               "Head: dome = 8, its printed 9 and 7 have no ring.")
    note = HEAD_NOTE

    FLASH_S = 0.6
    # TWO PLAYERS: the SAME figure on both halves, mirrored in position, at the
    # size a half of the screen allows. MEASURED by tests/test_modes.py section
    # 35 (lit area of one figure today, two at full size, two as built, and red
    # excess 0 in every state) and by the ink-coverage gate at the smaller size.
    versus_ready = True

    # THE GHOST. A figure drops at a running total (`shots_per_figure()` tens), and
    # the next one appears when its fall is over (`FALL_S`; at once for a figure
    # with no fall pose), so a perfect shooter waits for each replacement.
    ghost_has_waits = True

    def ghost_ceiling(self):
        return figure_ghost_ceiling()

    def ghost_shots_per_target(self):
        return shots_per_figure()

    def ghost_expected_wait(self, target_index):
        """The mean fall (FALL_S for a figure with a pose, nothing for one without)
        over the cast: the next figure appears when the fall is over."""
        return expected_wait_after_drop()

    @classmethod
    def new_link(cls) -> PlanLink:
        return PlanLink()

    def pair_with(self, link: "PlanLink | None", side: int) -> None:
        self._link = link
        self._seat = side

    def retire(self) -> None:
        """Nothing is shared that a finished half could hold up."""

    @property
    def animated(self) -> bool:
        """True only while a hit flash is live; derived from the list that
        `advance` reads so the two cannot disagree."""
        return (bool(self._marks.flashes) or bool(self.flashes)
                or self._fall is not None)

    def moving_at(self, t: float) -> bool:
        """The fall pose appearing is content the camera sees change; answered
        from the fall's own span, never from what a hit did."""
        return self._fall_span is not None and self._fall_span[0] < t < self._fall_span[1]

    def __init__(self) -> None:
        self.visible = True
        self._rng = random.Random()
        self._fall_sounds = FallSounds()      # a seedable draw, see audio/fall_sounds.py
        self._names: list[str] = []
        self._fig: "Figure | None" = None
        self._prev_name: "str | None" = None
        self._centre: "tuple[float, float] | None" = None
        self._prev_centre: "tuple[float, float] | None" = None
        self._height = 0.0
        self.shots = 0
        self.hits = 0
        self.score_total = 0
        self.no_ring = 0
        self.started_at = 0.0
        self.by_zone: dict[str, int] = {}
        self.first_t: "float | None" = None
        self.last_t: "float | None" = None
        self._marks = Marks()          # the shots the figure has absorbed
        self.flashes: list[tuple[float, float, float, str, bool]] = []   # screen-level
        self.dropped = 0               # figures that reached the drop total
        self.fall_shots = 0            # fired while a figure lay down
        self._fall: "Fall | None" = None
        self._fall_span: "tuple[float, float] | None" = None
        self._forced: "str | None" = None
        self._link: "PlanLink | None" = None
        self._seat = 0
        self._k = 0                    # which shared figure this half is on

    # -- lifecycle ------------------------------------------------------------

    def start(self, ctx: RenderContext, figure: "str | None" = None) -> None:
        """A new round is a new figure (never the one just shown) at a new
        place. `figure=` is for tests only: it forces the draw so a per-image
        check does not depend on luck."""
        self.shots = self.hits = self.score_total = self.no_ring = 0
        self.started_at = ctx.now
        self.by_zone = {}
        self.first_t = self.last_t = None
        self.flashes = []
        self.dropped = 0
        self.fall_shots = 0
        self._fall = None
        self._fall_span = None
        self._k = 0
        if figure is not None:
            self._forced = figure
        self._names = figure_names()
        self.reset_targets()
        self._new_figure(ctx)

    def _new_figure(self, ctx: RenderContext,
                    released_at: "float | None" = None) -> None:
        """Choose, load and place the next figure. It carries nothing over: its
        marks and its running total are its own.

        `released_at` is when the figure before it was resolved (the dropping
        shot's own time), for the ghost's wait record; None for the first."""
        self._marks = Marks()
        if not self._names:
            self._fig = None
            return
        field = self.playfield(ctx)
        plan = None
        if self._link is not None:
            plan = self._plan(ctx, field)
            name = plan.names[self._seat]
        else:
            name = self._forced or catalog.pick_figure(self._rng, self._prev_name,
                                                       self._names)
        # In a half the figure is built against the HALF's width (`ctx.width`),
        # so its size cap is the cap for a screen half as wide: a figure scaled
        # by ~0.707, which is half the area, so two cost what one costs today.
        self._fig = Figure(name, ctx.width, ctx.height, SINGLE_BUDGET)
        self._prev_name = name
        self._height = self._fig.fit_height(field.height(), ctx.height, 1.0)
        self._fig.sprite.set_range(self._height)
        self._fig.sprite.warm(self._height)
        self._fig.prepare_fall(self._height, ctx.height)
        w, h = self._fig.sprite.drawn_size(self._height)
        if plan is not None:
            left, right = field.left() + w / 2.0, field.right() - w / 2.0
            top, bottom = field.top() + h / 2.0, field.bottom() - h / 2.0
            if right <= left or bottom <= top:
                self._centre = (field.center().x(), field.center().y())
            else:
                u = 1.0 - plan.u if self._seat == 1 else plan.u     # mirrored position
                self._centre = (left + (right - left) * u, top + (bottom - top) * plan.v)
        else:
            self._centre = catalog.pick_position(self._rng, field, w / 2.0, h / 2.0,
                                                 self._prev_centre,
                                                 avoid=self.menu_keepout(ctx))
        self._prev_centre = self._centre
        self.note_target(ctx.now, released_at)

    def _plan(self, ctx: RenderContext, field):
        """The shared plan for this half's next figure: a name and a position
        (fractions of the field, in the LEFT half's frame), made by whichever
        half asks first. The menu button's keep-out is a soft preference in the
        single-player draw and is not applied here: one position serves both
        halves and the button lives in one corner."""
        k = self._k
        self._k += 1
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
            return SimpleNamespace(names=names, u=u, v=v)

        return self._link.get(k, make)

    # -- play -----------------------------------------------------------------

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        if self._fall_span is not None and ev.t < self._fall_span[1]:
            # The figure is down (or this pulse was fired before it dropped and
            # reported after). Nothing to hit; not counted as a shot at it.
            self.fall_shots += 1
            return ShotOutcome(hit=False, target_id="fallen")
        self.shots += 1
        if self.first_t is None:
            self.first_t = ev.t
        self.last_t = ev.t
        fig = self._fig
        if fig is None:
            return ShotOutcome.miss()
        points, zone = fig.score(ev.x, ev.y, self._centre, self._height)
        if points <= 0:
            if fig.on_figure(ev.x, ev.y, self._centre, self._height):
                # ON the printed figure but in no ring (42-51% of its ink is). It
                # must not look like nothing happened: a dim ring says "seen, on
                # the figure, no ring", distinct from a scored ring and from the
                # nothing a shot far off the figure draws.
                self.no_ring += 1
                self._marks.add(fig, ev.x, ev.y, self._centre, self._height,
                                ev.t, NO_RING, False)
                return ShotOutcome(hit=False, absorbed=True,
                                   target_id=f"{fig.name}:{NO_RING.replace(' ', '-')}")
            return ShotOutcome.miss()
        self.hits += 1
        self.score_total += points
        self.by_zone[zone] = self.by_zone.get(zone, 0) + 1
        total = self._marks.tally(points)
        target_id = f"{fig.name}:{zone}"
        dropped_now = bool(self._marks.done)
        if self._marks.done:
            # The figure drops NOW. Its marks go with it, so the flash that says
            # why (the total) is kept at the screen spot where the shot landed.
            self.dropped += 1
            self.flashes = (self.flashes + [(ev.x, ev.y, ev.t, str(total), True)])[-MAX_MARKS:]
            record_mark()   # the flash that says why the figure dropped IS a mark
            self._fall = Fall.start(fig, self._centre, self._height, ev.t, ctx.height)
            self._fall_span = ((self._fall.t0, self._fall.t1)
                               if self._fall is not None else None)
            if self._fall is None:
                self._new_figure(ctx, ev.t)    # no pose: gone at once
            else:
                self._marks = Marks()          # the marks go with the figure
                self._fig = None               # nothing to hit until the next one
        else:
            self._marks.add(fig, ev.x, ev.y, self._centre, self._height,
                            ev.t, str(total), True)
        return ShotOutcome(hit=True, points=points, absorbed=True,
                           target_id=target_id,
                           sound=self._fall_sounds.pick(fig.name) if dropped_now else None)

    def nearest(self, ev: ShotEvent, ctx: RenderContext) -> "NearestTarget | None":
        fig = self._fig
        if fig is None:
            return None
        inner = fig.innermost(self._centre, self._height)
        if inner is None:
            return None
        zone, cx, cy, r = inner
        return (f"{fig.name}:{zone}", math.hypot(ev.x - cx, ev.y - cy), r)

    def aim_point(self, ctx: RenderContext) -> "tuple[float, float] | None":
        """Where the development walk-through (`--modetest`) should aim: the
        centre of the innermost ring."""
        if self._fig is None:
            return None
        inner = self._fig.innermost(self._centre, self._height)
        return (inner[1], inner[2])

    def advance(self, ctx: RenderContext) -> bool:
        changed = self._marks.advance(ctx.now, self.FLASH_S)
        if self._fall is not None and ctx.now >= self._fall.t1:
            released = self._fall.t0       # the dropping shot's own time
            self._fall = None              # the second is up: next figure
            self._new_figure(ctx, released)
            changed = True
        if self.flashes:
            self.flashes = [f for f in self.flashes
                            if ctx.now - f[2] < self.FLASH_S]
            changed = True                 # fading, or the tick that ends it
        return changed

    def status(self) -> str:
        fig = self._fig
        if fig is None and self._fall is not None:
            return f"{self.dropped} down   score {self.score_total}   {self.shots} shots"
        if fig is None:
            return "no artwork found in targets/"
        counts = "  ".join(f"{z['name']}:{self.by_zone.get(z['name'], 0)}"
                           for z in sorted(fig.zones, key=lambda q: -q["points"]))
        miss = self.shots - self.hits
        out = (f"{fig.name}   {counts}  no-ring:{self.no_ring}  miss:{miss - self.no_ring}   score {self.score_total}"
               f"   {self.shots} shots")
        if self.first_t is not None and self.last_t is not None and self.shots > 1:
            out += f"   string {self.last_t - self.first_t:.1f}s"
        return out

    # -- drawing --------------------------------------------------------------

    def occluders(self, ctx: RenderContext):
        if ctx.calibrating or self._fig is None:
            return []
        return [self._fig.occluder(self.z, self._centre, self._height)]

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if ctx.calibrating:
            return          # one ring to shoot at while the grid is shot
        fig = self._fig
        if fig is None and self._fall is not None:
            self._fall.paint(p)
            paint_hits(p, ctx, self.flashes, self.FLASH_S)
            return
        if fig is None:
            p.setPen(QPen(GREY_MID))
            p.setFont(mono(px(ctx, 20)))
            p.drawText(int(ctx.width * 0.3), int(ctx.height * 0.5),
                       "no artwork found in the targets folder")
            return
        fig.sprite.paint(p, self._centre, self._height)
        self._marks.paint(p, ctx, fig, self._centre, self._height, self.FLASH_S)
        paint_hits(p, ctx, self.flashes, self.FLASH_S)
        paint_hint(p, ctx, self.playfield(ctx), self.started_at)
