"""Arcade modes: pluggable things to shoot at.

The detector's job ends when it can say "a shot landed at screen (x, y) at time
t". Everything after that -- what was there, whether it counts, what happens
next -- belongs to a mode. Modes are the part of this project that will keep
growing, so the boundary between them and the machinery has to be narrow and
dull: a mode receives shots and draws itself, and that is all it may assume.

A `Mode` is a `Layer`, so it composites with the background, the bullet holes,
the stats HUD and the camera pane without any of them knowing modes exist. It
gains one hook the other layers do not have: `shot()`.

## A shot is like a click, but it is not a click

Four differences, each of which has bitten somebody who assumed otherwise:

  - **There is no cursor.** No hover, no drag, no press-and-hold, no aiming
    feedback between shots. A mode that needs any of those cannot be driven by
    a pistol. This is the constraint that decides whether an existing
    click-driven game can be adopted.
  - **There can be two at once.** The shooter sometimes fires two pistols
    together, so shots carry a `pointer` id and arrive genuinely simultaneously.
    The touch model fits; the mouse model does not.
  - **A shot is final the instant it is reported.** It is the position of the
    FIRST frame of the laser pulse, because that is where the muzzle was at
    trigger break; everything after is follow-through. Nothing is retracted or
    refined later, so a mode may act immediately and never has to undo.
  - **The rate is bounded.** Roughly five shots a second per gun, with a
    refractory window. Nothing here can be click-spammed.

## The one rule modes may not break

Never draw saturated red: `red - max(green, blue)` must stay under about 30.
The camera is pointed at this screen hunting for a red dot, so red graphics
build a target that shoots itself. Use the palette from `lsa.display.layers`;
`tests/test_ui_smoke.py` scans every colour in the source and will fail the
build if a mode introduces one.
"""

from __future__ import annotations

import math
import textwrap
from collections import OrderedDict
from dataclasses import dataclass, field

from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainter

from .. import bests as _bests
from .. import ghost as _ghost
from .. import scorecard as _scorecard
from ..audio.mixer import DRY_SOUND, SELECT_SOUND
from ..display.scene import Layer, Occluder, RenderContext
from ..settings import DEFAULT_MAGAZINE, MAGAZINE_MAX, AppSettings, normalise_magazine
from . import efficiency, versus


# -- text fitting: a font is not free, so it is built once and a fit is found once --
#
# THE ROUND-OVER SCREEN HALVED THE CAMERA'S FRAME RATE (2026-10-22). Five of five
# round-closed lines in his logs, across three builds, read ~19 fps two seconds
# after the round closed, against 38-59 fps at the verdict minutes earlier:
#
#     15:03:49 -> 19.9   15:05:13 -> 19.3   15:43:49 -> 18.9
#     15:44:51 -> 19.3   17:12:57 -> 19.6
#
# At 19 fps a frame is 53 ms and his pulses measure 35-146 ms, so a short pulse
# lands in ONE frame and `min_frames` (2) discards it. The shots he fired at the
# NEXT ROUND box to leave the screen were thrown away BY the screen.
#
# The cause was `_draw_lines` walking its font size down one point at a time and
# building a font and a metrics object per step: from `px(ctx, 52)` at 4K the start
# is ~100, so up to ~90 of each PER BLOCK, ~6 blocks, every repaint, and the HUD
# tick asks for a repaint five times a second whatever has changed. That is the
# SAME fault as the ring numbers of Static Targets on 2026-10-20 (one paint went
# from ~12 ms to 171 ms, max 617, and starved the capture thread to 2.9 fps): fixed
# there for that caller, left standing here. The offscreen renderer does this font
# work much faster than Windows, which is why no suite saw either.
#
# So: fonts (and their metrics) are kept per (size, bold) for the process -- a few
# hundred at most, the sizes being integers -- and the SIZE a block fits at is kept
# per (box, lines, start, floor, weight), so an unchanged screen does no font work
# at all. The cache is not optional; removing it brings the 19 fps back.
_FONTS: "dict" = {}
_FONTS_MAX = 512
_FITS: "OrderedDict" = OrderedDict()
_FITS_MAX = 256


def _largest_fitting(fits, start: int, floor: int) -> int:
    """The largest size in [floor, start] for which `fits(size)` holds, else
    `floor` -- EXACTLY what walking down from `start` one point at a time until
    something fits (or the floor is reached, fit or not) returns, in ~log2
    probes instead of up to `start - floor`. Needs `fits` monotonic (if a size
    fits, every smaller one does): a width and a height only grow with the size.
    `tests/test_modes.py` proves the equality against the real font engine."""
    if start <= floor:
        return floor
    if fits(start):
        return start
    lo, hi = floor, start - 1            # `lo` is accepted untested, as the walk does
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _fit_remember(key, compute) -> int:
    """`compute()` once per `key`, least-recently-used out past `_FITS_MAX`."""
    got = _FITS.get(key)
    if got is not None:
        _FITS.move_to_end(key)
        return got
    got = _FITS[key] = compute()
    while len(_FITS) > _FITS_MAX:
        _FITS.popitem(last=False)
    return got


_DISC_MASK = None


def _disc_mask():
    """A filled circle as an alpha mask, built once for the whole process.

    A circle is the same shape at every size, so one 128x128 mask serves
    every ball in every mode and the painter does the stretching. The
    figures need one mask per frame because their outline changes; this
    does not, and building one per ball per frame would be the 176ms
    mistake the animated mode already paid for.
    """
    global _DISC_MASK
    if _DISC_MASK is None:
        from PySide6.QtCore import QRectF as _R, Qt as _Qt
        from PySide6.QtGui import QImage as _I, QPainter as _P, QPixmap as _X
        n = 128
        img = _I(n, n, _I.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = _P(img)
        p.setRenderHint(_P.Antialiasing)
        p.setPen(_Qt.NoPen)
        p.setBrush(_Qt.white)
        p.drawEllipse(_R(0, 0, n, n))
        p.end()
        _DISC_MASK = _X.fromImage(img)
    return _DISC_MASK


@dataclass(frozen=True)
class ShotEvent:
    """One shot, in screen pixels. Immutable: it already happened."""

    x: float
    y: float
    t: float                  # monotonic seconds
    index: int                # stable, 1-based, matches the bullet hole
    pointer: int = 0          # which gun/track, for two-handed shooting
    # The timestamp of the pulse's LAST frame, which is the time the microphone
    # pairing (`ShotCorroborator.add_shot`) knows this shot by. `t` is the first
    # frame (where the muzzle was); the dry-pull tally needs the other one to
    # recognise its own shot in the pairing's verdict. None: same as `t`.
    t_end: "float | None" = None
    # How bright the pulse was (the detector's chroma peak). DATA ONLY: no mode
    # reads it and it decides nothing about a shot. The host files it with the
    # round so the log can say how bright the round was (`ghost.record_fitness`
    # reports a dim median as a CAUTION). Brightness no longer decides anything:
    # since 2026-10-22 only a round's SPAN can discard it, and a dim median is how
    # his genuine far side images. None: not supplied (every test event), which is a
    # caution in the log and changes nothing else.
    peak: "float | None" = None

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class ShotOutcome:
    """What the mode made of a shot, and what the host should do about it.

    The mode decides, rather than the host inferring, because only the mode
    knows what was on screen. The feedback flags exist so a mode can take over
    presentation -- a mode where misses are punished might want silence, one
    with its own hit effect might not want a bullet hole drawn over it.
    """

    hit: bool = False
    points: int = 0
    target_id: str | None = None
    draw_hole: bool = True
    play_sound: bool = True
    # Name of a registered sample to play INSTEAD of the ordinary gunshot for
    # this shot (None = the ordinary one). The figure drills use it for the shot
    # that drops a figure. The host falls back to the ordinary gunshot if the
    # sample is not loaded, so a missing file costs the effect and never the bang.
    sound: "str | None" = None
    # "I absorbed this shot": the mode has made the shot part of something it
    # draws and moves itself (a hole carried by a walking figure), so the
    # global bullet-hole layer must NOT also put a hole on the screen at the
    # spot, or the two marks would drift apart. Everything else a shot does
    # (the sound, the log, the tally, the panels standing down) is unchanged;
    # only the screen hole is withheld. A shot that is not absorbed leaves its
    # ordinary hole.
    absorbed: bool = False
    # "This shot was SEEN and does not count": a shot the round did not want
    # (a side whose magazine is empty). Neither a hit nor a miss, so the app's
    # session records it as a shot and as nothing else. Reported as a miss it
    # inflated the one number this project exists to measure honestly: the hit
    # rate's denominator.
    unscored: bool = False
    # "A TRIGGER PULL WITH AN EMPTY MAGAZINE": counted as a dry pull (the session
    # keeps it apart from its shots, the log marks it), and the app plays the
    # bubble pop (`mixer.DRY_SOUND`) INSTEAD of the gunshot -- the gunshot says a
    # live round went off, and this was not one. The hole is drawn as for any shot.
    # It scores nothing and moves no hit rate. (His rule of 2026-10-20, replacing
    # an earlier silent, holeless dry pull.)
    dry: bool = False
    # A SELECTION CUE (`mixer.SELECT_SOUND`) played one millisecond AFTER the sound
    # above, never instead of it: the MENU button, the NEXT ROUND box and every tile
    # of the menu screen. A click, which has no gunshot, plays the cue alone.
    cue: "str | None" = None

    @classmethod
    def miss(cls) -> "ShotOutcome":
        return cls(hit=False)


# (target id, distance of the shot from the target's centre, scoring radius),
# both distances in screen pixels. See `Mode.nearest`.
NearestTarget = tuple[str, float, float]


def describe_nearest(near: "NearestTarget | None", hit: bool,
                     target_id: "str | None", mm_per_px: "float | None") -> str:
    """The part of a SHOT log line that says how far the shot was from a target.

    A hit on the target the mode named reads `(d=41px = 12.3mm from centre of
    centre, r=907px)`; anything else reads `(nearest circle3: 23px = 6.8mm
    outside, r=97px)`, with a negative number meaning inside its edge. Empty
    when the mode cannot say. Pure, so the log format is tested without an app.
    """
    if near is None:
        return ""
    ident, d, r = near

    def mm(v: float) -> str:
        return f" = {v * mm_per_px:.1f}mm" if mm_per_px else ""

    if hit and ident == target_id:
        return f"(d={d:.0f}px{mm(d)} from centre of {ident}, r={r:.0f}px)"
    gap = d - r
    word = "outside" if gap >= 0 else "inside"
    return f"(nearest {ident}: {abs(gap):.0f}px{mm(abs(gap))} {word}, r={r:.0f}px)"


def _wait_inside_clock(appeared: float, released: "float | None",
                       within: "tuple[float, float] | None") -> float:
    """The part of one recorded wait that lies inside the human's own clock.

    The wait is [released, appeared]. `within` is (first shot, last shot) of the
    side being scored. Pure, so the leak it closes is testable without a mode:

      * nothing was waited for (`released` None): 0
      * the target appeared AFTER the last shot: 0 (the human's clock never
        contained it; the mode spawns the next target whether or not anybody is
        still shooting)
      * the wait began BEFORE the first shot (a figure that timed out before he
        fired): only the part after the first shot counts, because the human's
        clock starts there
      * no bounds known: the whole recorded wait
    """
    if released is None:
        return 0.0
    if within is None:
        return max(0.0, float(appeared) - float(released))
    first, last = within
    if appeared > last:
        return 0.0
    return max(0.0, float(appeared) - max(float(released), first))


class Mode(Layer):
    """Base class for everything there is to shoot at.

    Subclasses implement `start`, `shot` and `paint`, and `advance` if they
    animate. `reset` is what a mode does between rounds; `start` is what it does
    when first selected, and by default they are the same thing.
    """

    name = "mode"
    # A one-shot sound this mode wants played, drained by the app tick.
    _queued_sound: "str | None" = None
    title = "Mode"            # shown to the user
    summary = ""              # one line describing how it is played
    note = ""                 # a limitation worth saying once when it is chosen (logged)
    z = 40                    # above the background, below holes and the HUD
    animated = False
    # CAN THIS DRILL RUN ON HALF A SCREEN BESIDE AN INDEPENDENT COPY OF ITSELF?
    # False unless someone has MEASURED that it can (its lit area with two sets
    # of targets, its paint cost, whether a shared random draw would be needed).
    # When two players are selected and the chosen drill is not ready, it runs
    # single-player across the whole screen AND SAYS SO -- silently ignoring the
    # setting is indistinguishable from a bug.
    versus_ready = False
    # A SCREEN, NOT A DRILL (Keyboard Shortcuts). It spends no rounds from the
    # magazine (`ModeHost.counts_ammo`) and is never split for two players
    # (`ModeHost.set_mode`), so a shot at a help screen cannot end a round or
    # draw a "not ready for two players" apology. The menu predates this flag
    # and is recognised by name in the same two places.
    is_screen = False

    @classmethod
    def new_link(cls) -> "object | None":
        """A fresh object the two halves of ONE versus round will SHARE.

        Almost every drill needs nothing shared: two independent copies are
        the whole design. Reaction is the exception -- whoever gets the earlier
        reveal wins the round, so the reveal has to be one draw read by both
        halves. The host does not know what the object is; it only creates it
        (here, once per `_start_all`, so a restart or a next round can never
        inherit a stale one) and hands it to both halves through `pair_with`.
        Default None: nothing is shared."""
        return None

    def pair_with(self, link: "object | None", side: int) -> None:
        """You own half `side` (0 = left, 1 = right) of a versus round, and
        `link` is what you share with the other half. Called by the host
        before `start`, in the versus branch of `ModeHost._start_all` only.
        Default: a no-op, so no existing drill changes."""

    def retire(self) -> None:
        """Your half of a versus screen is DONE: it will no longer be painted
        or ticked. Called once, by the host, before the first tick it skips.

        A drill that shares state with its other half (see `new_link`) must
        stop being WAITED ON here, or the half still shooting would hang on a
        partner that never answers again. Default: a no-op."""

    def take_sound(self) -> "str | None":
        """A sound this mode wants played NOW, once, and not for a shot.

        Almost everything audible here is a consequence of a trigger pull
        and rides home on `ShotOutcome.sound`. A target APPEARING is not:
        it is the moment the reaction drill times from, and no shot has
        happened yet. The app drains this on its own tick.

        Default None, so no existing mode changes behaviour, and nothing
        in the detection or shot path is involved at all.
        """
        got = self._queued_sound
        self._queued_sound = None
        return got

    def moving_at(self, t: float) -> bool:
        """Was content that the camera could mistake for a laser moving at `t`?

        NOT `animated`. `animated` means "may ask for a repaint", and it turns
        True because of a shot: a scoring hit appends a flash ring, so reading
        it after `shot()` reports the animation the shot itself caused. This
        is a question about the world at time `t` (the clock of
        `ShotEvent.t`, RULES B4), answered from the mode's own schedule and
        never from what a hit did to it, so it gives the same answer before
        and after `shot()` runs. A grey flash ring is not moving content.

        Default False: a mode whose targets sit still, or whose motion is
        grey at red excess 0, lets the sensitivity learner learn. A mode with
        genuinely moving content overrides it. `tests/test_feedback.py` 31e
        checks every registered mode against the "same before and after a
        hit" property.
        """
        return False

    # -- WHAT THE GHOST NEEDS FROM A DRILL (piece 2 of the ghost score) ---------
    #
    # `lsa/ghost.py` scores a round against a perfect shooter and deliberately
    # carries NO table of per-drill values: a copy of another module's rule, kept
    # in a second place, is how `Probe.bar()` drifted from the app's threshold and
    # chose the dimmest dot on the table. So each drill states, from its OWN
    # scorer, the most a shot can be worth, and records when each of its targets
    # became hittable. Nothing here draws, ticks, or touches `moving_at`; the only
    # work is appending one pair of floats when a target appears.
    #
    # Does this drill make the shooter WAIT for targets (so that the ghost must be
    # charged the same wait)? False: nothing ever has to appear (Static Targets, and
    # both ball drills, whose replacement ball is spawned inside the very call that
    # scored the hit). Declared by the drill, never guessed by the scorer.
    ghost_has_waits = False

    # (appeared_at, released_at) for each target this round, in order. A TUPLE so
    # the class-level default can never be shared and mutated between instances;
    # replaced, never appended to. Reset by every drill's `start()`.
    _targets: "tuple[tuple[float, float | None], ...]" = ()

    def ghost_ceiling(self) -> "_ghost.Ceiling | None":
        """The most the k-th shot of a fresh round can be worth (k from 1), taken
        from THIS drill's own scorer, or None when the drill cannot say -- and
        then the drill is NOT SCORABLE, which is better than a number that is
        quietly wrong. Screens (the menu, the shortcuts list) are not drills and
        keep this default."""
        return None

    def ghost_shots_per_target(self) -> "int | None":
        """How many perfect shots one target takes before the next one is needed.
        1 for a target a single hit removes; a figure that drops at a running total
        takes more. None: cannot say."""
        return 1

    def reset_targets(self) -> None:
        """Forget this round's targets. Called from `start()`, which the host runs
        at the start of every round and every restart."""
        self._targets = ()

    def note_target(self, appeared_at: float, released_at: "float | None" = None) -> None:
        """Record that a target became HITTABLE at `appeared_at`.

        `released_at` is the moment the drill began to make the shooter wait for
        it (the instant the previous target was resolved: the shot's own `ev.t`
        for a hit, the tick's `ctx.now` for a timeout), or None when nothing was
        being waited for (the first target of a round, or one of several already
        on screen). Both are on the clock `ShotEvent.t` and `ctx.now` share
        (perf_counter, RULES B4), and the difference is the wait.

        Bounded by the largest magazine: a round cannot need more targets than it
        has shots, and the first ones are the ones that matter.
        """
        if len(self._targets) >= MAGAZINE_MAX:
            return
        self._targets = self._targets + (
            (float(appeared_at), None if released_at is None else float(released_at)),)

    @property
    def target_count(self) -> int:
        """How many targets have been recorded this round."""
        return len(self._targets)

    def ghost_expected_wait(self, target_index: int) -> "float | None":
        """The wait this drill ITSELF declares for its `target_index`-th target
        (0-based, in the order `note_target` records them), in seconds.

        NOT USED BY THE GHOST SINCE 2026-10-23. It used to price a target the round
        never drew (the human missed, so the ghost needed more targets than the round
        produced) -- and that is what made the score rewardable by MISSING: a human
        who missed dropped fewer figures, endured fewer falls, and was compared with
        a ghost charged the full fall-and-rest for every figure its perfect play
        would have dropped (a stored 1146 on a scale whose ceiling is 1000). See
        `_ghost_wait_plan`. Kept, with its checks, until he decides whether the
        declaration machinery and the `declared` basis are retired.

        Each drill that makes the shooter wait overrides this, and derives the
        number from the SAME constants its runtime uses (`MIN_WAIT`/`MAX_WAIT`,
        `GAP_S`, `NEXT_BOARD_S`, `REST_S` and the fall clip, `FALL_S`). It is never
        typed into `ghost.py` or here: a copy of another module's rule kept in a
        second place is how `Probe.bar()` drifted. Default None: "I cannot say",
        and the score stays a lower bound rather than becoming a guess. Read-only;
        it draws nothing and touches no hit test."""
        return None

    def _ghost_wait_plan(self, shots: int,
                         within: "tuple[float, float] | None" = None
                         ) -> "list[float] | None":
        """The wait before each of the ghost's `shots` shots: ONLY waits this round
        actually contained, or None when the drill makes nobody wait.

        THE RULE (2026-10-23). The ghost plays the round as it unwound, so it is
        charged exactly the waits the human endured and no others. Shot i goes to
        target i // per; where the round recorded that target the ghost is charged
        its wait, and where the round NEVER DREW it -- the human missed, so he fired
        more shots than the round supplied targets for -- the ghost is charged
        NOTHING and fires at its cadence.

        WHY NOT PRICE THE MISSING TARGET AT THE DRILL'S AVERAGE WAIT (what this did
        until 2026-10-23). Because it made the score beatable by missing. On a drill
        where dropping a target triggers the wait (a figure's fall, the soldier's
        fall and rest) the human who misses drops fewer targets and endures fewer
        waits, while the ghost, who never misses, was charged a full fall for every
        figure its perfect play would have dropped. His own round: 17 shots in
        14.29 s against a ghost of 21.42 s, a speed share of 1.50 and a stored score
        of 1146 on a scale whose ghost is 1000. Charging the ghost for waits the human
        never saw is charging it for a hypothetical round, which is not "the round as
        it unwound".

        WHAT IT COSTS, STATED. A ghost charged nothing for the targets the round never
        drew is FASTER than a ghost that had to wait for them, so the speed share of a
        round with misses is understated (conservative: a score can only come out
        lower than a hypothetical better round's, never higher, which is the safe
        direction for a number that becomes a record). The round that missed almost
        everything is therefore scored, low (aim share near zero), and not refused.

        `within` is the human's own clock, (first shot, last shot) on the clock
        `note_target` shares with `ShotEvent.t`. A wait is charged only for the part
        of it that lies INSIDE that clock: the wait before the first shot is charged
        to nobody, and a target that appeared after the last shot is not charged at
        all (the mode keeps running after the magazine empties, and its next target
        arriving is not time the human's clock contains). None means the bounds are
        not known and every recorded wait is taken whole (tests that build a round
        by hand); `_score_round` always supplies them. Bounds that are not finite
        and ordered make the plan None -- the score is then a lower bound, loudly,
        rather than a plan built on a clock nobody can vouch for.

        Does NOT read `ghost_expected_wait`: a declared mean is not a measurement.
        """
        per = self.ghost_shots_per_target()
        if not self.ghost_has_waits or per is None or per < 1 or shots < 1:
            return None
        bounds = None
        if within is not None:
            try:
                bounds = (float(within[0]), float(within[1]))
            except (TypeError, ValueError, IndexError):
                return None
            if not (math.isfinite(bounds[0]) and math.isfinite(bounds[1])) \
                    or bounds[0] > bounds[1]:
                return None
        out: "list[float]" = []
        for i in range(int(shots)):
            j = i // per
            if i % per or j >= len(self._targets):
                out.append(0.0)       # the same target already up, or one never drawn
                continue
            appeared, released = self._targets[j]
            out.append(_wait_inside_clock(appeared, released, bounds))
        return out

    def ghost_waits(self, shots: int,
                    within: "tuple[float, float] | None" = None
                    ) -> "list[float] | None":
        """The wait before each of a perfect shooter's `shots` shots, in seconds,
        measured from what this round recorded (see `_ghost_wait_plan`). None when
        the drill makes nobody wait, cannot say how many shots a target takes, or
        was handed a clock that is not a clock -- then the score is a lower bound."""
        return self._ghost_wait_plan(shots, within)

    def ghost_declared_waits(self, shots: int) -> int:
        """How many of the ghost's waits came from the drill's own declared average
        and not from this round: ALWAYS 0 since 2026-10-23, because the ghost is no
        longer charged any wait the round did not contain. Kept so the host's call,
        `ghost_score(declared_waits=)`, the `declared` basis and the records already
        stored under it keep meaning what they meant; a check pins the 0."""
        return 0

    def ghost_available_at(self, shots: int, fps: float,
                           within: "tuple[float, float] | None" = None
                           ) -> "list[float] | None":
        """The `available_at` for `ghost.ghost_score`: the ghost's own timeline,
        built from this round's waits and one camera frame (`fps` is the
        configured camera rate; `within` the human's clock, see `_ghost_wait_plan`).
        None when the waits cannot be supplied -- the score is then a lower bound
        (`ghost_score(has_waits=True)`)."""
        return _ghost.available_from_waits(self.ghost_waits(shots, within), fps)

    def start(self, ctx: RenderContext) -> None:
        """Called when the mode becomes active, and on an explicit restart."""

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        """Decide what this shot hit. Must not block; it runs on the UI thread."""
        return ShotOutcome.miss()

    def stop(self) -> None:
        """Called when the mode is replaced. Release anything held."""

    def status(self) -> str:
        """A short line for the HUD: score, level, targets left."""
        return ""

    def nearest(self, ev: ShotEvent, ctx: RenderContext) -> "NearestTarget | None":
        """How close this shot is to the nearest live target. For the log only.

        Returns `(target id, distance from its centre, scoring radius)`, both
        distances in screen pixels, where the scoring radius is the one `shot()`
        tests against and the id is the one `shot()` reports in its outcome.
        The target is measured where `shot()` would find it for THIS event (a
        moving target is rewound by the shot's age, exactly as `shot()` does).

        Must be called BEFORE `shot()`: a hit removes or respawns the target,
        and the distance to what is left would describe the next target, not
        the one the shooter aimed at. Read-only; changes nothing. None means
        the mode has nothing to measure against.
        """
        return None

    @staticmethod
    def closest(candidates: "list[NearestTarget]") -> "NearestTarget | None":
        """The candidate the shot is nearest to, by the gap to its scoring
        edge (so a small target close by beats a large one far away)."""
        best = None
        for c in candidates:
            if best is None or (c[1] - c[2]) < (best[1] - best[2]):
                best = c
        return best

    # -- helpers every mode wants ---------------------------------------------

    # The longest a hit test may rewind a moving target. A shot needs two
    # camera frames to confirm (his pulses measure 31-50 ms) plus delivery,
    # so the honest age is 40-80 ms; a quarter second is the ceiling so a
    # stale or wrong timestamp cannot teleport the test.
    MAX_REWIND_S = 0.25

    @classmethod
    def shot_age(cls, ev: "ShotEvent", ctx: RenderContext) -> float:
        """How long ago the shot happened, clamped to [0, MAX_REWIND_S].

        `ev.t` is the detector's frame stamp (perf_counter) and `ctx.now` is
        the same clock (RULES B4). The shot is reported only after it has
        been confirmed, so the target has moved on since the position was
        measured: a ball at 2 screen widths a second moves several radii in
        that time. The hit test must ask where the target WAS, not where it
        is drawn now. The drawn position and the physics are untouched.
        """
        return min(cls.MAX_REWIND_S, max(0.0, ctx.now - ev.t))

    # A ball is drawn as an OUTLINE of this width, centred on its radius, so
    # the ball visibly extends half a stroke beyond `r`. The width is defined
    # once, here, and used by BOTH `paint` and the hit test: two copies of
    # `max(3.0, r * 0.08)` is how the drawn edge and the scored edge drift
    # apart, and a shot that lands on the visible rim then scores a miss.
    RING_STROKE_FRAC = 0.08
    RING_STROKE_MIN_PX = 3.0

    @classmethod
    def ring_stroke(cls, r: float, frac: "float | None" = None,
                    min_px: "float | None" = None) -> float:
        """Width in pixels of the outline drawn for a ring of radius `r`.

        `frac` and `min_px` default to the ball's. A mode that draws a heavier
        or lighter line passes ITS OWN values here -- from a constant it also
        hands to `ring_hit_radius` -- so the line that is painted and the
        line that is scored come from the same numbers (circles 0.09/2px,
        shoot rings 0.18/3px).
        """
        f = cls.RING_STROKE_FRAC if frac is None else frac
        m = cls.RING_STROKE_MIN_PX if min_px is None else min_px
        return max(m, r * f)

    @classmethod
    def ring_hit_radius(cls, r: float, frac: "float | None" = None,
                        min_px: "float | None" = None) -> float:
        """The radius within which a shot scores on a drawn ring.

        The geometric radius plus half the stroke: the outer edge of the line
        the shooter is looking at, and no further. Not generosity beyond what
        is drawn -- a shot on the outer half of the stroke is visibly on the
        ball.
        """
        return r + cls.ring_stroke(r, frac, min_px) / 2.0

    @classmethod
    def disc_occluder(cls, z: int, cx: float, cy: float, r: float):
        """A round target as a SOLID disc, for the bullet holes beneath it.

        His instruction, 2026-10-18, after the animated soldier was fixed:
        *"that should be true for all other modes - the figures (balls,
        people-like cutouts, etc.) that go over holes should obscure the
        holes"*.

        A ball is drawn as an OUTLINE (a fill sits exactly where the dot
        lands and swallows the shot -- B10), so its inside is bare
        background and every hole it crosses showed through it. Nothing
        about what is PAINTED changes here: the mode merely declares the
        shape it occupies, and `BulletHoles` leaves that area out.

        ONE definition, used by both ball modes and anything round added
        later. Two copies of a shape is exactly how the drawn edge and the
        scored edge drifted apart on this class once already.

        The mask is a single 128x128 circle built once per process and
        stretched by the painter -- a circle is scale-invariant, so every
        ball at every size shares it and the cache hit rate is perfect.
        """
        from PySide6.QtCore import QRect
        rr = cls.hit_radius_for_occluder(r)
        box = QRect(int(cx - rr), int(cy - rr), int(2 * rr), int(2 * rr))
        return Occluder(z, box, _disc_mask)

    @classmethod
    def hit_radius_for_occluder(cls, r: float) -> float:
        """How far out the solid shape reaches: the drawn outer edge.

        The same radius the hit test uses, so what hides a hole and what
        counts as a hit on the ball are one number.
        """
        return cls.ring_hit_radius(r)

    # A TARGET FACE PAINTED ON A ROUND TARGET. His request, 2026-10-18:
    # *"Can you paint an actual target on a bouncing ball?"* -- so the balls
    # read as targets rather than as discs.
    #
    # Every number here comes from the two rules that govern anything drawn on
    # the screen the camera watches, and NOT from what looks good:
    #
    #  - **B10, what the dot lands on.** The detector scores
    #    `red - max(green, blue)`. A laser dot's red saturates while the grey
    #    underneath it keeps its green and blue, so a dot landing on grey G
    #    reports roughly `255 - G` instead of `255 - background`. That is why
    #    the ball was outline-only once: a fill at 200 left 55 and took his
    #    median shot from 85 to about 30. The face is drawn at 0.55 of the
    #    artwork ceiling -- about 39 on an unchanged palette -- which leaves
    #    216 where the bare fill leaves 235. His median shot reads ~75 instead
    #    of ~85 and his dimmest ever logged pull reads ~27, against a bar that
    #    runs 8-19. Strokes are THIN (3.5% of the radius) so most of a dot that
    #    is 45-75 screen px across still lands on the dark fill either way.
    #    This is exactly how his own figure artwork works: fine printed rings
    #    over a dark body, and those measure 89% of heard trigger pulls.
    #  - **B1b, the lit-area budget.** 39 is far below 60, the luminance at
    #    which a pixel starts spending it, so a screen full of ringed balls
    #    costs nothing there. Red excess is 0, as everything here is.
    #
    # ONE definition for both ball drills. Two copies of a face is how the
    # drawn edge and the scored edge drifted apart on this class once already.
    RING_FACE_FRAC = 0.55          # of the artwork ceiling: see above
    FACE_RADII = (0.72, 0.46, 0.22)  # of the ball radius
    FACE_STROKE_FRAC = 0.035       # thin, for the reason above
    FACE_CENTRE_FRAC = 0.07        # a small aim point, not a bullseye

    # -- the MENU button's ground, which no target may be placed on ----------
    #
    # The geometry lives HERE, on Mode, rather than on the button that draws
    # it, because two different kinds of code need it and they must agree to
    # the pixel: `MenuButton` draws and hit-tests it, and every drill that
    # PLACES a target has to keep off it.
    #
    # He asked for both on 2026-10-18, from a photograph of Static Targets
    # with the button underneath the bottom-left aim mark: *"the menu button
    # has to move a bit down so that it doesn't overlap with the target; also
    # you will need to be careful not to place any targets directly over that
    # menu button in the modes that can do that"*.
    #
    # Moving it down alone could not have fixed that picture, and the
    # arithmetic says so: at 4K the bottom-left mark reaches y=1933 while the
    # playfield ends at 1944, so a 91px button cannot fit under it whatever
    # the inset. The button goes as low as it can AND the mark moves.
    #
    # Fractions of the screen, never pixels (RULES B5).
    MENU_CORNER = "top left"       # replaced by the app from measured sensitivity
    MENU_W_FRAC = 0.095
    MENU_H_FRAC = 0.042
    MENU_INSET_FRAC = 0.012        # of the smaller side
    MENU_BOTTOM_INSET_FRAC = 0.002  # lower when it sits along the bottom
    MENU_KEEPOUT_MARGIN_FRAC = 0.008

    @classmethod
    def menu_rect(cls, ctx: RenderContext) -> QRectF:
        """Where the MENU button sits, inside the playfield corner it was given.

        INSIDE THE PLAYFIELD, not the screen: that rectangle already excludes
        the instruction band, the key legend and -- when the mapping knows it
        -- any part of the panel the camera cannot see. A button drawn where a
        hit cannot be detected answers nothing and he keeps shooting at it.
        """
        field = cls.playfield(ctx)
        w = ctx.width * cls.MENU_W_FRAC
        h = ctx.height * cls.MENU_H_FRAC
        small = min(ctx.width, ctx.height)
        inset = small * cls.MENU_INSET_FRAC
        low = small * cls.MENU_BOTTOM_INSET_FRAC
        corner = cls.MENU_CORNER
        x = (field.left() + inset) if "left" in corner else (field.right() - inset - w)
        y = (field.top() + inset) if "top" in corner else (field.bottom() - low - h)
        return QRectF(x, y, w, h)

    @classmethod
    def menu_keepout(cls, ctx: RenderContext) -> QRectF:
        """The button's rectangle plus a margin: ground no target may occupy.

        A margin rather than the bare rectangle, because a target that merely
        TOUCHES the button is the same complaint -- he photographed an overlap
        of about thirty pixels and it read as one thing sitting on another.
        """
        carried = ctx.menu_keepout_rect
        if carried is not None:
            # A two-player half context: the host measured it on the FULL
            # screen, where the button is actually drawn. See RenderContext.
            return QRectF(carried)
        m = min(ctx.width, ctx.height) * cls.MENU_KEEPOUT_MARGIN_FRAC
        return cls.menu_rect(ctx).adjusted(-m, -m, m, m)

    @classmethod
    def clear_of_menu(cls, ctx: RenderContext, cx: float, cy: float,
                      r: float) -> bool:
        """Is a round target of radius `r` at (cx, cy) off the button?"""
        box = cls.menu_keepout(ctx)
        nx = min(max(cx, box.left()), box.right())
        ny = min(max(cy, box.top()), box.bottom())
        return ((cx - nx) ** 2 + (cy - ny) ** 2) > r * r

    @classmethod
    def paint_rings(cls, p: "QPainter", cx: float, cy: float, r: float) -> None:
        """Paint concentric rings inside a round target of radius `r`.

        PAINTED ONLY. The ball is still one target, hit or missed as a whole,
        and nothing here is scored -- he asked for it to be painted. Ring
        scoring on a MOVING target is a different proposition: the rewind that
        places a shot on a travelling ball carries about +/-12px of timestamp
        uncertainty, a tenth of a ball, which is the whole error budget a
        0.22r centre ring would be judged inside.
        """
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QPen
        from ..display.layers import grey_at

        face = grey_at(cls.RING_FACE_FRAC)
        w = max(1.5, r * cls.FACE_STROKE_FRAC)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(face, w))
        for frac in cls.FACE_RADII:
            rr = r * frac
            if rr > w:          # below that the ring is just a thicker blob
                p.drawEllipse(QPointF(cx, cy), rr, rr)

        # NO FILLED CENTRE, and the first version had one. The design note
        # argued for "something to aim AT rather than a hole where the aim
        # point is" -- and the centre is BY DEFINITION where the dot's
        # brightest pixel lands, so it is the one spot on the target where
        # brightness is paid for on every single shot. A 43-grey bullseye
        # measured 33 units above the background at exactly that spot, against
        # a budget of 12. The innermost ring is the bullseye; the middle of it
        # stays bare, and a dot landing where he aimed scores what it would
        # have scored on an empty ball.
        #
        # `tests/test_ui_smoke.py` caught it by sampling the ball's centre
        # pixel, which is the right place to sample and the only one that
        # would have shown it.

    @staticmethod
    def fold(v: float, lo: float, hi: float) -> float:
        """Fold v into [lo, hi] by reflection: where an elastic bouncer is
        if it travelled straight for the distance that put it at v."""
        span = hi - lo
        if span <= 0:
            return lo
        u = (v - lo) % (2.0 * span)
        return lo + (u if u <= span else 2.0 * span - u)

    @staticmethod
    def playfield(ctx: RenderContext) -> QRectF:
        """Where targets may legitimately go.

        Not simply the screen. The camera is off-axis and may not see the whole
        monitor -- in testing it could not see the bottom strip at all -- and a
        target drawn where a hit cannot be detected is worse than no target: it
        reads as the app ignoring a good shot. `RenderContext.playable` carries
        the region the mapping says is actually covered, when one is known.
        """
        from ..display.layers import BOTTOM_STRIP, px, top_band_height

        region = getattr(ctx, "playable", None)
        if region is not None and region.width() > 0 and region.height() > 0:
            area = QRectF(region)
        else:
            area = QRectF(ctx.rect)
        # The instruction band at the top and the key legend plus signal meter
        # along the bottom are always drawn over everything. A target placed
        # under either is a target that cannot be seen, which reads exactly
        # like a detector that is ignoring good shots.
        top = ctx.rect.y() + top_band_height(ctx)
        bottom = ctx.rect.y() + ctx.height - max(BOTTOM_STRIP, px(ctx, BOTTOM_STRIP))
        if area.top() < top:
            area.setTop(top)
        if area.bottom() > bottom:
            area.setBottom(bottom)
        return area


@dataclass
class _Registered:
    cls: type[Mode]
    key: int | None


_REGISTRY: dict[str, _Registered] = {}


def register(key: int | None = None):
    """Decorator: make a mode discoverable, optionally bound to a key."""
    def wrap(cls: type[Mode]) -> type[Mode]:
        _REGISTRY[cls.name] = _Registered(cls, key)
        return cls
    return wrap


def available() -> list[type[Mode]]:
    return [r.cls for r in _REGISTRY.values()]


def registered() -> "list[tuple[type[Mode], int | None]]":
    """Every registered mode with the key it is bound to (None: no key), in
    registration order. The public way to ask what `available()` cannot, so a
    consumer that needs the keys (the shoot-to-select menu) does not reach into
    the private registry."""
    return [(r.cls, r.key) for r in _REGISTRY.values()]


def by_name(name: str) -> "type[Mode] | None":
    entry = _REGISTRY.get(name)
    return entry.cls if entry else None


def by_key(key: int) -> "type[Mode] | None":
    for entry in _REGISTRY.values():
        if entry.key is not None and entry.key == key:
            return entry.cls
    return None


# HOW LONG AN ARMED PIECE OF CHROME STAYS ARMED.
#
# Anything in this app that switches the shooter out of what they are doing
# takes TWO hits, because a phantom detection doing it is the worst
# accidental action available here -- 92 phantoms arrived in twenty seconds
# once, when the window dropped behind something else. Four seconds: long
# enough to fire again deliberately, short enough that a stray in a corner
# and a stray a round later are not read as one intention.
#
# ONE definition. `modes.menu` imports it for the EXIT tile; the MENU button
# below uses the same number for the same reason, and two copies is exactly
# how two things that must agree drift apart.
CHROME_ARM_S = 4.0


class MenuButton:
    """A rectangle saying MENU, in every drill, two hits to take it.

    Asked for 2026-10-18: *"I would like to have the menu button available in
    all the modes and it should be dynamically placed into the best corner
    sensitivity-wise. It would be rectangular and it will say MENU. If it is
    hit twice it will trigger the menu mode."*

    It lives on `ModeHost` rather than in each mode for the reason the host
    exists at all: one object knows about mode lifecycle, so one object can
    carry a thing that outlives any particular mode. Nine modes each drawing
    their own would be nine chances for it to differ, and the drill written
    next month would not have one.

    THREE DECISIONS WORTH THE WORDS:

    **The mode is offered the shot first.** The button only claims a shot the
    active drill reported as a MISS. A ball that drifts under the button is
    still hittable, and a shot that would have scored never opens a menu
    instead. The opposite precedence costs a hit and reads from the firing
    line exactly like the app ignoring a good trigger pull, which is the
    failure this project has paid for more often than any other.

    **It is drawn as an outline and a word, never filled.** He has to HIT
    this, and whatever a dot lands on comes off its score: the detector reads
    `red - max(green, blue)`, the dot's red saturates, and the grey underneath
    keeps its green and blue (RULES B10). An outline leaves the middle of the
    button -- where anyone aims -- at bare background, scoring full.

    **It does not move once a drill is running.** The corner is chosen by the
    app from measured sensitivity and handed in; a button that migrated
    between rounds would be a button he shoots at where it used to be.
    """

    def __init__(self) -> None:
        self.visible = True
        self.armed_at = 0.0
        self.requested = False        # drained by the app's tick, never acted on here
        self._dirty = False

    # -- geometry -------------------------------------------------------------

    @property
    def corner(self) -> str:
        """Which corner it is in -- held on `Mode`, not here.

        ONE PLACE, because the drills have to keep their targets off it and
        they reach it through `Mode.menu_keepout`. A second copy on this
        object is how the button a drill avoids and the button he actually
        sees would come to be in different corners.
        """
        return Mode.MENU_CORNER

    @corner.setter
    def corner(self, value: str) -> None:
        Mode.MENU_CORNER = value

    def rect(self, ctx: RenderContext) -> QRectF:
        """Where it sits. The geometry is `Mode.menu_rect`; see it for why."""
        return Mode.menu_rect(ctx)

    def contains(self, x: float, y: float, ctx: RenderContext) -> bool:
        return bool(self.visible) and self.rect(ctx).contains(x, y)

    # -- being hit ------------------------------------------------------------

    def hit(self, t: float) -> bool:
        """Arm it, or on the second hit within the window, request the menu.

        It only ever RECORDS the request. Switching modes from here would run
        inside `ModeHost.shot` and stop the object that is executing -- the
        same rule the menu's own tiles follow, and the reason this app has
        exactly one route into a mode change.
        """
        self._dirty = True
        if self.armed_at and (t - self.armed_at) <= CHROME_ARM_S:
            self.armed_at = 0.0
            self.requested = True
            return True
        self.armed_at = t
        return False

    def take_request(self) -> bool:
        """The app's tick asks once; the answer is consumed."""
        if not self.requested:
            return False
        self.requested = False
        self.armed_at = 0.0
        return True

    def armed(self, now: float) -> bool:
        return bool(self.armed_at) and (now - self.armed_at) <= CHROME_ARM_S

    def advance(self, ctx: RenderContext) -> bool:
        """One repaint when the arming lapses, and none otherwise.

        A piece of chrome that reported a change every tick would repaint a
        4K surface sixty times a second for a word that does not move --
        measured at 8.53ms a paint against a detector that needs 7ms of the
        same CPU per camera frame.
        """
        lapsed = bool(self.armed_at) and (ctx.now - self.armed_at) > CHROME_ARM_S
        if lapsed:
            self.armed_at = 0.0
            self._dirty = True
        was, self._dirty = self._dirty, False
        return was

    # -- drawing --------------------------------------------------------------

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if not self.visible:
            return
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPen
        from ..display.layers import GREY_HI, GREY_TEXT, grey_at, px

        box = self.rect(ctx)
        armed = self.armed(ctx.now)
        # Armed is BRIGHTER, not coloured: the one palette this screen has is
        # grey, and red excess must stay at 0 for anything that appears.
        edge = GREY_HI if armed else grey_at(0.62)
        stroke = max(2.0, box.height() * 0.055)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(edge, stroke))
        p.drawRoundedRect(box, box.height() * 0.18, box.height() * 0.18)

        label = "MENU" if not armed else "MENU - AGAIN"
        # The word is fitted to the box rather than guessed at, so the longer
        # armed label cannot overrun the rectangle it is drawn in -- a text
        # overrun on this screen has cost a key legend its tail once already.
        # This button is painted on every in-play repaint, so its size is found
        # once per (label, box width, start) and its font is the process-wide one
        # (see `_FONTS`): the same walk-down-one-point loop that halved the frame
        # rate on the round screen stood here, ~30 fonts per paint at 4K.
        from ..display.layers import mono
        start = px(ctx, 20)
        width = box.width()

        def fits(s: int) -> bool:
            fm = ModeHost._run_fonts(s, True)[1]
            return fm.horizontalAdvance(label) <= width * 0.84

        # The walk tried sizes start..10 and kept the font of the last one tried,
        # so "nothing fits" is size 10, and a start below 10 is never walked at all.
        size = start if start < 10 else _fit_remember(
            ("button", mono, label, width, start), lambda: _largest_fitting(fits, start, 10))
        p.setFont(ModeHost._run_fonts(size, True)[0])
        p.setPen(GREY_HI if armed else GREY_TEXT)
        p.drawText(box, Qt.AlignCenter, label)




@dataclass
class _Tally:
    """What one side of a versus screen has done, counted by the HOST.

    Counted here rather than read from each drill because the drills do not
    agree on what they keep (one has `score`, the others only `hits`), and a
    scoreline built from whatever each happens to expose would compare unlike
    numbers. A shot routed to a side is one shot for that side and invisible
    to the other; a click never reaches this.
    """

    shots: int = 0
    hits: int = 0
    points: int = 0
    # Shots the camera saw AFTER this side had used its allotment. They are
    # real trigger pulls, drawn and heard, but they reach neither the drill nor
    # the score. Counted so "it stopped scoring" is a number and not a mystery.
    after_done: int = 0
    # First round to last round of this side of this magazine, from each shot's
    # own timestamp (`ShotEvent.t`). The one clock efficiency is computed from.
    clock: efficiency.Clock = field(default_factory=efficiency.Clock)
    # What the ghost score and its record guard need, filed per LIVE registered
    # shot (so at most the magazine, never more): the pulse's brightness, the
    # time the microphone pairing knows it by (`t_end`), and the pairing's two
    # verdicts about it. `heard_t` are the shots the microphone also heard;
    # `deaf_t` the ones it could not have (muted by the app's own bang), which
    # leave the denominator because a fast string legitimately outruns it.
    peaks: list = field(default_factory=list)
    seen_t: list = field(default_factory=list)
    heard_t: set = field(default_factory=set)
    deaf_t: set = field(default_factory=set)


class ModeHost(Layer):
    """Holds whichever mode is active and forwards the world to it.

    A host rather than swapping layers in the scene directly, so that exactly
    one object knows about mode lifecycle. Switching modes is then a single
    call that cannot leave a half-removed layer behind, and the rest of the app
    keeps talking to one stable layer.

    TWO PLAYERS. In versus the host holds a SECOND instance of the same drill
    and hands each the half of the screen it owns, as a derived
    `RenderContext` (`rect` and `playable` narrowed; everything else shared).
    No drill knows a split exists. What stays on the host, drawn once from the
    FULL context: the MENU button, the divider and the scoreline. See
    `modes/split.py` for the geometry and for why the axis is left/right.
    """

    name = "modes"
    z = 40
    animated = True

    def __init__(self) -> None:
        self.mode: Mode | None = None        # single-player mode, or the LEFT side
        self.mode2: Mode | None = None       # the RIGHT side, versus only
        self.versus = False
        # Non-empty when two players were requested and the drill could not
        # honour it. The app reads it once, right after `set_mode`, to say so.
        self.fallback_note = ""
        self.tallies = (_Tally(), _Tally())
        self._tally_sig: tuple = ()
        # What the app's own measurement was doing during THIS round, for the
        # honesty mark on the efficiency figure (modes/efficiency.py holds the
        # conditions). Fed by `note_measurement`; frozen when the round ends.
        self.health = efficiency.RoundHealth()
        self._round_reasons: "list[str]" = []
        # THE MATCH (versus only; modes/versus.py holds the arithmetic).
        # `seats[side]` is the PLAYER standing on that side; totals are kept by
        # PLAYER, so they follow him across the swap while the scoreline's
        # LEFT/RIGHT stay the physical halves.
        self.round_no = 1
        self.seats = (0, 1)
        self.totals = [versus.SideResult(), versus.SideResult()]
        self.round_over = False
        # THE MAGAZINE. `magazine` is the size WANTED (the app copies the saved
        # setting here); `_round_mag` is the size THIS ROUND was dealt, captured
        # when its tallies are reset. A round's length and the bullets drawn for
        # it are the same number by construction, and a setting changed mid-round
        # takes effect at the next round's start rather than ending this one
        # underneath the shooter. `settings.py` owns the default and the range;
        # nothing here keeps a number of its own.
        self.magazine = DEFAULT_MAGAZINE
        self._round_mag = DEFAULT_MAGAZINE
        # THE HOLD. His words (2026-10-20): "I want to see the hole of my last
        # shot for at least a second. As soon as I fire the number 17, that hole
        # is supposed to be there and you wait at least a second until you
        # display the end-of-round screen." `hold_s` is per host so a test can
        # set it; HOLD_S is the product's number. A side that has just fired its
        # last round is still PAINTED (the hole is seen against the target it
        # struck, not on an empty screen) until `ev.t + hold_s`; shots during
        # the hold are already dry. The round closes only when every side is
        # empty AND every hold has expired.
        self.hold_s = self.HOLD_S
        self._hold_until: list = [None, None]
        self._now = 0.0                  # the last clock reading the host was given
        self._first_live_t: "float | None" = None
        # The dry-pull summary for a round that has already been reset, waiting
        # for the corroborator's last verdicts (see `_flush_dry`).
        self._dry_pending: "dict | None" = None
        self.dry_deaf = 0                # of the camera-only: the microphone was deaf
        # DRY PULLS (see `versus.dry_total` for why this is a union and for the
        # fact that none of it is scored). `dry_camera` is every detection after
        # a magazine was empty; `dry_corroborated` is the subset of those the
        # microphone also heard (the corroborator's "both"); `dry_mic_only` is
        # clicks with no detection, counted only when the microphone has
        # calibrated and every magazine was already empty.
        self.dry_camera = 0
        self.dry_corroborated = 0
        self.dry_mic_only = 0
        self._dry_seen: list = []          # t_end of each dry detection, to recognise "both"
        # The same times, NEVER cleared by a new round: the corroborator resolves a
        # pull 0.3 s after it happened, and a shot at the START box is what starts
        # the next round, so its verdict arrives after the tally was reset.
        self._dry_times: list = []
        self._dry_open = False             # the round is over: microphone-only pulls may count
        self._dry_all_since: "float | None" = None
        self._last_live_t: "float | None" = None
        # Sides whose drill has been told it is retired (see `_retire_done`).
        self._retired: set = set()
        # THE VERDICT'S NEXT-ROUND BOX. `_round_ended` and `_next_requested` are
        # one-shot facts for the APP's tick to drain (the host never reaches
        # into the scene's layers): the first tells it to clear the bullet
        # holes that would sit under the verdict text, the second that the box
        # was shut. `_box_seen_at` is when the box was first PAINTED.
        self._round_ended = False
        self._next_requested = False
        self._box_seen_at: "float | None" = None
        # THE ROUND'S TRAIL. One line per lifecycle event, for the APP to put in
        # the log. The host is pure and must not log directly (it has no log and
        # a test builds one with none), so it records and the app's tick drains
        # `take_events`, the way `take_sound` and `take_menu_request` are
        # drained. Without this a report like "no targets appeared on the right"
        # could be neither confirmed nor denied: nothing on disk knew a round
        # had started, closed or swapped.
        self._events: list = []
        # Selection cues for CLICKS (no gunshot to follow), drained by `take_sounds`.
        self._cues: list = []
        self.verdict: "versus.Verdict | None" = None
        # The microphone's heard-but-never-seen count, fed by the app. It has
        # no position, so it cannot be given to a side; it is published beside
        # the verdict as the only check on a pull the camera missed entirely.
        self.heard_unseen = 0
        self.mic_ready = False
        # THE GHOST SCORE (lsa/scorecard.py), computed ONCE when a round ends and
        # shown on the end screen only -- his instruction: "I only want the score
        # displayed when the round ends. That's it."
        #   `fps`    the CONFIGURED camera rate, set by the app from the same
        #            setting the capture is opened with. NEVER the measured rate:
        #            the ghost's sight floor is one camera frame, and a measured
        #            rate that dipped during a round would silently revalue the
        #            ceiling. The default is the setting's own default.
        #   `bests`  the machine's best rounds (a `BestBook` over
        #            `AppSettings.bests`), or None, in which case nothing is
        #            compared and nothing is recorded.
        #   `cards`  one `RoundCard` per side, None until a round ends.
        self.fps = float(AppSettings().fps)
        self.bests: "_bests.BestBook | None" = None
        self.cards: tuple = (None, None)
        self._heard_round_base = 0
        self._heard_match_base = 0
        self.visible = True
        self.last_outcome: ShotOutcome | None = None
        # THE WAY BACK, from inside every drill. See `MenuButton`.
        self.menu_button = MenuButton()

    # About 1.2 s, not a flat 1.0: he said "at least a second", and the frame that
    # paints the hole is already a few milliseconds after the shot.
    HOLD_S = 1.2
    # How long a finished round's dry-pull summary waits for the microphone
    # pairing before it is written: the corroborator's window is 0.3 s and the
    # app polls it every 200 ms, so 0.8 s covers a pull made at the very end.
    DRY_SETTLE_S = 0.8

    # -- the round's trail ----------------------------------------------------

    def _say(self, line: str) -> None:
        # Scoped when it is SAID, not when it is drained: a line about a versus
        # round must not be labelled "round" because the mode changed before the
        # app's next tick.
        self._events.append(("versus" if self.versus else "round", line))
        if len(self._events) > 200:             # bounded: nobody drains in a test
            del self._events[:100]

    # -- the magazine ---------------------------------------------------------

    def set_magazine(self, n) -> int:
        """Set the wanted magazine size (clamped by `settings.normalise_magazine`).
        It governs the NEXT round: see `magazine`/`_round_mag` in `__init__`."""
        self.magazine = normalise_magazine(n)
        if not self.round_over and not any(t.shots for t in self.tallies):
            self._round_mag = self.magazine      # nothing fired yet: nothing to end
        return self.magazine

    @property
    def counts_ammo(self) -> bool:
        """Does the active screen spend rounds? A drill does; the menu is a
        screen, not a drill, and shooting a tile must not consume ammunition or
        end a round."""
        return (self.mode is not None and getattr(self.mode, "name", "") != "menu"
                and not self.mode.is_screen)

    def remaining(self, side: int = 0) -> int:
        """Rounds still in this side's magazine. THE one answer the bullets on
        the screen, the round's end and the tests all read."""
        return max(0, self._round_mag - self.tallies[side].shots)

    # -- dry pulls ------------------------------------------------------------

    def _note_dry_camera(self, ev: "ShotEvent") -> None:
        self.dry_camera += 1
        t = ev.t_end if ev.t_end is not None else ev.t
        self._dry_seen.append(t)
        self._dry_times.append(t)
        if len(self._dry_seen) > 256:
            del self._dry_seen[:128]
        if len(self._dry_times) > 64:
            del self._dry_times[:32]

    def is_dry_shot(self, t: float) -> bool:
        """Was the detection stamped `t` (its last frame, as `add_shot` is given
        it) a pull made while a magazine was empty? The app asks before it words
        a "seen but not heard" finding: a dry pull is the feature working, and
        calling it a phantom is a log line naming the wrong cause."""
        return any(abs(st - t) < 1e-6 for st in self._dry_times)

    def note_dry_unheard(self, t: float, deaf: bool) -> None:
        """A dry detection nobody heard. `deaf`: the microphone could not have
        (it was muted by the app's own last bang, or not delivering audio), which
        explains a camera-only count that a listening microphone would not."""
        if not deaf:
            return
        if any(abs(st - t) < 1e-6 for st in self._dry_seen):
            self.dry_deaf += 1
        elif self._dry_pending is not None and any(abs(st - t) < 1e-6 for st in self._dry_pending["seen"]):
            self._dry_pending["deaf"] += 1

    def _live_shot_at(self, t: float):
        """The tally holding a LIVE registered shot of this round that the
        microphone pairing knows by `t`, or None (a dry pull, another round's
        shot, or nothing of ours)."""
        for tally in self.tallies:
            for st in tally.seen_t:
                if abs(st - t) < 1e-6:
                    return tally
        return None

    def _note_live_heard(self, t: float) -> None:
        """The microphone also heard the live shot stamped `t`: it counts toward
        the record guard's corroboration. A set, so a repeated verdict for the
        same shot cannot count twice."""
        tally = self._live_shot_at(t)
        if tally is not None:
            tally.heard_t.add(round(float(t), 6))

    def note_live_unheard(self, t: float, deaf: bool) -> None:
        """A LIVE shot nobody heard, and whether the microphone could have: `deaf`
        means it was muted by the app's own bang or not delivering audio, so its
        silence is no evidence either way and the shot leaves the corroboration
        denominator. (`note_dry_unheard` is the same fact for a dry pull.)"""
        if not deaf:
            return
        tally = self._live_shot_at(t)
        if tally is not None:
            tally.deaf_t.add(round(float(t), 6))

    def take_bests_changed(self) -> bool:
        """One-shot: did a round just set a record that has not been written?
        The app writes the settings file when this is True -- at most once per
        round, and not at all when nothing improved."""
        return self.bests.take_changed() if self.bests is not None else False

    def _flush_dry(self, now: float, force: bool = False) -> None:
        """Write a finished round's dry-pull line once the pairing has settled."""
        rec = self._dry_pending
        if rec is not None and (force or now >= rec["due"]):
            self._dry_pending = None
            self._say(versus.dry_log_text(rec["camera"], rec["corroborated"], rec["mic_only"],
                                          rec["mic_ready"], rec["deaf"]) + "; settled")

    def note_pull_outcomes(self, outcomes) -> None:
        """Feed the corroborator's pairing decisions (`take_outcomes`).

        Only two things are taken from them, and neither ADDS a pull the camera
        already counted: a "both" for one of OUR dry detections marks it
        corroborated (a subset, not an addend), and a "heard-not-seen" click is
        a microphone-only dry pull -- if the microphone has calibrated, and
        every magazine was already empty when it was heard (a click cannot be
        given to a side, so in two players one empty half is not enough). A
        click while the microphone is uncalibrated is a guess and is never
        counted, which is what makes the on-screen figure a lower bound then.
        A mouse click never arrives here: this is fed by the microphone only.
        """
        for kind, t in outcomes:
            rec = self._dry_pending
            if kind == "both":
                self._note_live_heard(t)
                for i, st in enumerate(self._dry_seen):
                    if abs(st - t) < 1e-6:
                        del self._dry_seen[i]
                        self.dry_corroborated += 1
                        break
                else:
                    # The verdict for a pull from the round that was JUST reset
                    # (the shot at the NEXT ROUND box starts the next round, so
                    # its pairing always arrives after the reset): credit that
                    # round's summary, which is still waiting for it.
                    if rec is not None:
                        for i, st in enumerate(rec["seen"]):
                            if abs(st - t) < 1e-6:
                                del rec["seen"][i]
                                rec["corroborated"] += 1
                                break
            elif kind == "heard-not-seen":
                if (self._dry_open and self.mic_ready and self._dry_all_since is not None
                        and t > self._dry_all_since):
                    self.dry_mic_only += 1
                elif (rec is not None and rec["mic_ready"] and rec["since"] is not None
                        and rec["since"] < t <= rec["until"]):
                    rec["mic_only"] += 1

    def dry_screen_text(self) -> str:
        return versus.dry_text(self.dry_camera, self.dry_corroborated,
                               self.dry_mic_only, self.mic_ready)

    def dry_log_text(self) -> str:
        return versus.dry_log_text(self.dry_camera, self.dry_corroborated,
                                   self.dry_mic_only, self.mic_ready, self.dry_deaf)

    def take_round_ended(self) -> bool:
        """Did a round just close? One-shot. The app clears the bullet
        holes on it, through its one hole-clearing site: they were drawn under
        the verdict text and competed with it."""
        out, self._round_ended = self._round_ended, False
        return out

    def take_next_round_request(self) -> bool:
        """Was the ROUND box shut (shot or clicked)? One-shot. The app starts
        the next round through the same function the N key calls."""
        out, self._next_requested = self._next_requested, False
        return out

    def request_next_round(self, how: str) -> None:
        if self.round_over and not self._next_requested:
            self._next_requested = True
            if how == "click":
                # A click has no gunshot for the cue to follow: it plays alone. A
                # shot's cue rides on its outcome (`_round_over_shot`), so it is
                # not queued here too.
                self._cues.append(SELECT_SOUND)
            self._say(f"ROUND {self.round_no + 1} box shut ({how}): starting the next round")

    def take_sounds(self) -> "list[str]":
        """Every cue the active drill(s) queued since the last call, with
        IDENTICAL cues drained in the same tick collapsed into one. The app used
        to drain only `host.mode.take_sound()` -- the LEFT half -- so a cue
        queued by the right half was never heard. A shared appearance (both
        halves of Reaction or Pop-Up go live in the same tick) is one event and
        makes one noise; a sound carried on `ShotOutcome.sound` is routed per
        side by `shot` and never comes through here."""
        out: list = list(dict.fromkeys(self._cues))
        self._cues.clear()
        for m in (self.mode, self.mode2):
            if m is None:
                continue
            got = m.take_sound()
            if got and got not in out:
                out.append(got)
        return out

    def take_events(self) -> "list[str]":
        """The round's log lines since the last call, oldest first."""
        out, self._events = self._events, []
        return [line for _scope, line in out]

    def take_scoped_events(self) -> "list[tuple[str, str]]":
        """Like `take_events`, with each line's scope ("versus" or "round")."""
        out, self._events = self._events, []
        return out

    # -- the way back ---------------------------------------------------------

    @property
    def _button_live(self) -> bool:
        """The button is drawn and hittable in every mode but the menu itself.

        On the menu there is nothing for it to do, and a second way into the
        screen you are already looking at is clutter on a screen that is also
        being shot at.
        """
        return (self.mode is not None and self.mode.visible
                and getattr(self.mode, "name", "") != "menu")

    def take_menu_request(self) -> bool:
        """Did the MENU button get its second hit? Asked by the app's tick.

        The button never switches modes itself: it is hit from inside
        `shot()`, and changing mode there would stop the object still
        executing. This is the same deferred route a tile takes, drained
        through the app's one `_select_mode`.
        """
        return self.menu_button.take_request()

    # -- lifecycle ------------------------------------------------------------

    def set_mode(self, mode: "Mode | None", ctx: RenderContext | None = None,
                 players: int = 1) -> None:
        """Make `mode` the active drill. `players` is 1 or 2.

        Two players is honoured only for a drill that declares `versus_ready`;
        any other drill (and the menu, which is a screen and not a drill)
        runs single-player across the whole screen. For a drill, the reason is
        left in `fallback_note` for the app to say out loud.
        """
        for old in (self.mode, self.mode2):
            if old is not None:
                old.stop()
        self.mode = mode
        self.mode2 = None
        self.versus = False
        self.fallback_note = ""
        self._reset_match()
        self._reset_tallies()
        if mode is not None and players == 2 and mode.name != "menu" and not mode.is_screen:
            if mode.versus_ready:
                self.mode2 = type(mode)()
                self.versus = True
            else:
                self.fallback_note = (
                    f"{mode.title} is not ready for two players yet: "
                    f"it runs single-player across the whole screen")
        # The signature depends on whether this is versus, which was only just
        # decided: take it again, or the first tick repaints for no reason.
        self._tally_sig = self._signature()
        if ctx is not None:
            self._start_all(ctx)

    def restart(self, ctx: RenderContext) -> None:
        """N. Once a round is over it is the NEXT round (two players: sides swap
        and the totals carry); at any other time it restarts the round in play,
        keeping the seats and the banked totals."""
        if self.round_over:
            self._next_round(ctx)
            return
        self._reset_tallies()
        self._start_all(ctx)

    def _start_all(self, ctx: RenderContext) -> None:
        if self.mode is None:
            return
        if self.versus and self.mode2 is not None:
            # ONE object for the two halves, made here and nowhere else: this
            # is reached from set_mode, restart (N mid-round) and _next_round
            # (N after a round), so every round starts from a fresh shared
            # draw and no stale reveal can carry across.
            link = type(self.mode).new_link()
            self.mode.pair_with(link, 0)
            self.mode2.pair_with(link, 1)
            self._say(f"round {self.round_no} starts ({self.mode.title}): "
                      f"LEFT is player {self.seats[0] + 1}, RIGHT is player {self.seats[1] + 1}")
            self._say(f"magazine {self._round_mag} each: the bullets are drawn under the "
                      f"scoreline, and the round ends when both are empty")
            self.mode.start(self._side_ctx(ctx, 0))
            self.mode2.start(self._side_ctx(ctx, 1))
        else:
            if self.counts_ammo:
                self._say(f"round {self.round_no} starts ({self.mode.title}): "
                          f"magazine {self._round_mag}")
            self.mode.start(ctx)

    def _reset_tallies(self) -> None:
        self._flush_dry(self._now, force=True)
        if self._dry_open:
            # THE ROUND'S DRY PULLS, always in the trail (zero included): the
            # instrument is never abbreviated, and a window that closed with no
            # line would be indistinguishable from one that was never counted.
            #
            # NOT WRITTEN YET. The shot at the NEXT ROUND box is what ends the
            # round, and its microphone pairing arrives a few hundred ms later,
            # so a line written now books that last pull as camera-only every
            # time (seen live: mic_peak 0.8 on a pull counted "camera-only").
            # The numbers are kept and the line is written when they have settled
            # (`DRY_SETTLE_S`), crediting late pairings to this round.
            self._dry_pending = {
                "camera": self.dry_camera, "corroborated": self.dry_corroborated,
                "mic_only": self.dry_mic_only, "deaf": self.dry_deaf,
                "mic_ready": self.mic_ready, "seen": list(self._dry_seen),
                "since": self._dry_all_since, "until": self._now,
                "due": self._now + self.DRY_SETTLE_S}
        self.dry_camera = self.dry_corroborated = self.dry_mic_only = 0
        self.dry_deaf = 0
        self._hold_until = [None, None]
        self._first_live_t = None
        self._dry_seen = []
        self._dry_open = False
        self._dry_all_since = None
        self.tallies = (_Tally(), _Tally())
        self.cards = (None, None)
        self.health.begin(self._now)
        self._round_reasons = []
        self._round_mag = normalise_magazine(self.magazine)
        self._retired = set()
        self._box_seen_at = None
        self._next_requested = False
        self._heard_round_base = self.heard_unseen
        self._tally_sig = self._signature()

    def _reset_match(self) -> None:
        self.round_no = 1
        self.seats = (0, 1)
        self.totals = [versus.SideResult(), versus.SideResult()]
        self.round_over = False
        self.verdict = None
        self._heard_match_base = self.heard_unseen

    def _signature(self) -> tuple:
        # Everything the scoreline, the notes and the verdict are drawn from, so
        # a repaint is asked for when one CHANGES and not before.
        if not self.versus:
            # Single player repaints when the COUNT changes (a bullet leaves,
            # or the end screen comes up) and for nothing else: the microphone's
            # tally is not on this screen, and a signature that carried it would
            # repaint a 4K surface for words nobody is looking at.
            t = self.tallies[0]
            return (t.points, t.shots, self.round_no, self.round_over, self._round_mag,
                    self.dry_camera, self.dry_corroborated, self.dry_mic_only, self.mic_ready)
        return (tuple((t.points, t.shots, t.after_done) for t in self.tallies),
                self.round_no, self.seats, self._round_mag,
                self.heard_unseen, self.mic_ready,
                self.dry_camera, self.dry_corroborated, self.dry_mic_only)

    # -- the match (versus) ---------------------------------------------------

    def _side_empty(self, which: int) -> bool:
        """This side's magazine is empty (or the round was closed): what decides
        whether a shot reaches the drill. True from the instant of the last round."""
        return self.round_over or self.tallies[which].shots >= self._round_mag

    def _hold_over(self, which: int, now: float) -> bool:
        u = self._hold_until[which]
        return u is None or now >= u

    def _side_quiet(self, which: int, now: float) -> bool:
        """This side is empty AND its hold has expired: it is no longer painted
        or ticked. Empty-but-held is still on screen, so the last hole is seen."""
        return self.round_over or (self.tallies[which].shots >= self._round_mag
                                   and self._hold_over(which, now))

    def _side_done(self, which: int) -> bool:
        """`_side_quiet` at the host's last clock reading (the name the notes,
        the retire step and the tests use)."""
        return self._side_quiet(which, self._now)

    def note_heard_unseen(self, n: int = 1) -> None:
        """The app heard a trigger pull the camera never saw. Counted for the
        verdict's footnote; it never creates a shot (audio has no position)."""
        self.heard_unseen += max(0, int(n))

    def _maybe_finish(self, ctx: RenderContext) -> None:
        if self.round_over or not self.counts_ammo:
            return
        idx = (0, 1) if self.versus else (0,)
        if not all(self.tallies[i].shots >= self._round_mag for i in idx):
            return
        # ...AND every hold has run: the round closes when the last hole has been
        # on screen for the hold, not on the tick that emptied the magazine.
        if all(self._hold_over(i, ctx.now) for i in idx):
            self._finish_round(ctx)

    def _finish_round(self, ctx: RenderContext) -> None:
        """Bank both sides into their PLAYERS' totals and judge. Once per round.

        Called only from `_maybe_finish`, when both sides have registered their
        allotment, so both banked counts are exactly the magazine.

        SINGLE PLAYER ENDS THE SAME WAY (his ruling, 2026-10-20): one route for a
        round to end, the same box, the same start. It has no sides, no totals
        and no winner, so it records the result in the log and stops."""
        if self.round_over:
            return
        if not self._dry_open:
            self._dry_open = True
            self._dry_all_since = self._last_live_t
        self.health.note_unseen(self.heard_unseen - self._heard_round_base)
        self._round_reasons = self.health.reasons()
        self._say_efficiency()
        self._score_round()
        if not self.versus:
            t = self.tallies[0]
            self.round_over = True
            self._round_ended = True
            self._say(f"round {self.round_no} closed: magazine {self._round_mag} empty, "
                      f"{t.points} pts / {t.shots} shots, {t.hits} hits "
                      f"({t.after_done} after empty)")
            self._say(self.mic_log_text())
            return
        for side in (0, 1):
            who = self.seats[side]
            t, old = self.tallies[side], self.totals[who]
            self.totals[who] = versus.SideResult(
                old.points + t.points, old.shots + min(t.shots, self._round_mag))
        self.round_over = True
        self._round_ended = True
        # THE VERDICT IS THE RESULT OF THE ROUND THAT JUST ENDED, judged from the
        # two tallies, so it agrees arithmetically with the points printed
        # directly above it. It used to judge the MATCH TOTALS, which equal the
        # round's only in round 1: from round 2 the screen would have shown this
        # round's points over a cumulative winner, two different quantities on
        # one screen with nothing saying so. `totals` still accumulates per
        # PLAYER across the swap and the match goes to the LOG every round (the
        # closing event below); it is DELIBERATELY not on the screen, at his
        # request (2026-10-19): the verdict screen shows one round and is the
        # same height at round 1 and at round 9, or "by round 4 you run out of
        # screen space". Do not restore it as a "missing feature".
        self.verdict = versus.judge(
            versus.SideResult(self.tallies[0].points, min(self.tallies[0].shots, self._round_mag)),
            versus.SideResult(self.tallies[1].points, min(self.tallies[1].shots, self._round_mag)))
        match = versus.judge(self.totals[self.seats[0]], self.totals[self.seats[1]])
        t0, t1 = self.tallies
        self._say(f"round {self.round_no} closed: LEFT {t0.points} pts / {t0.shots} shots "
                  f"({t0.after_done} after done), RIGHT {t1.points} pts / {t1.shots} shots "
                  f"({t1.after_done} after done); match totals player 1 "
                  f"{self.totals[0].points} pts, player 2 {self.totals[1].points} pts; "
                  f"{versus.verdict_sentence(self.verdict, versus.SIDE_NAMES)}; "
                  f"{versus.result_words(match, versus.SIDE_NAMES, match=True)}")
        # The microphone line goes to the log EVERY time, whether or not the
        # screen shows it (see `versus.heard_on_screen`).
        self._say(self.mic_log_text())

    def _next_round(self, ctx: RenderContext) -> None:
        """The sides swap (two players), the totals carry, and the drill(s)
        start fresh. Single player has nobody to swap."""
        self.round_no += 1
        if self.versus:
            self.seats = (self.seats[1], self.seats[0])
            # SILENT ON SCREEN at his request, NEVER SILENT HERE: the swap is
            # the fairness defence and a log that cannot show it happened cannot
            # show that it did not.
            self._say(f"sides swapped for round {self.round_no}: player 1 is now on "
                      f"{versus.SIDE_NAMES[self.seats.index(0)]}, player 2 on "
                      f"{versus.SIDE_NAMES[self.seats.index(1)]}")
        self.round_over = False
        self.verdict = None
        self._reset_tallies()
        self._start_all(ctx)

    # -- the split ------------------------------------------------------------

    @staticmethod
    def _layout(ctx: RenderContext):
        from . import split
        return split.layout(ctx.rect)

    def _side_ctx(self, ctx: RenderContext, which: int) -> RenderContext:
        """The context one side of the screen is handed.

        `rect` is narrowed and `playable` INTERSECTED (never dropped, so the
        camera-coverage rule still holds per side). The MENU keep-out is
        carried in from the FULL context: `Mode.menu_keepout` would otherwise
        derive it from the half, where the playfield and the smaller side
        have both changed, and a drill would keep off a button that is not
        where it is drawn. Height-only functions (`px`, the top band, the
        bottom strip) are identical in the half, which is the whole reason the
        axis is left/right.
        """
        import dataclasses
        lay = self._layout(ctx)
        keep = Mode.menu_keepout(ctx)       # ctx here is the FULL context
        return dataclasses.replace(
            ctx, rect=lay.side(which),
            playable=lay.playable_for(which, ctx.playable),
            menu_keepout_rect=keep, screen_width=ctx.width)

    def _routed(self, x: float, ctx: RenderContext):
        """(mode, its context, side index or None) for a shot at screen x."""
        if self.versus and self.mode2 is not None:
            which = self._layout(ctx).route(x)
            return ((self.mode if which == 0 else self.mode2),
                    self._side_ctx(ctx, which), which)
        return self.mode, ctx, None

    def mode_for(self, x: float, ctx: RenderContext) -> "Mode | None":
        """The mode that owns screen x (the only one in single-player)."""
        return self._routed(x, ctx)[0]

    def nearest(self, ev: ShotEvent, ctx: RenderContext):
        """`Mode.nearest` for whichever side the shot belongs to (log only)."""
        mode, sctx, _which = self._routed(ev.x, ctx)
        return mode.nearest(ev, sctx) if mode is not None else None

    # -- the world ------------------------------------------------------------

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        if self.mode is None:
            return ShotOutcome.miss()
        # ROUTED BY X BEFORE ANY HIT TEST. The losing side never sees the
        # event, so neither player's shots, hits, score or miss count can be
        # polluted by the other's shooting.
        counts = self.counts_ammo
        self._now = ctx.now
        if counts and self.round_over:
            self.last_outcome = self._round_over_shot(ev, ctx)
            return self._after_shot(ev, ctx)
        mode, sctx, which = self._routed(ev.x, ctx)
        side = which if which is not None else 0
        if counts and self._side_empty(side):
            # THIS MAGAZINE IS EMPTY, AND A DRILL IS STILL ON SCREEN (the last
            # hole's hold, or the other half still shooting). The shot was SEEN,
            # so it is counted (`after_done`) and goes to the log with its
            # position and peak like every shot, but it reaches neither the drill
            # nor the score, and it is `unscored` (not a miss either: a miss would
            # inflate the session's hit-rate denominator).
            #
            # WHAT HE HEARS AND SEES (his rule, 2026-10-20, which REPLACED an
            # earlier one that made a dry pull silent and holeless): "use the
            # bubble pop for the sound when the bullets run out and still paint
            # the holes; that way player will understand that something is going
            # on because this is a boopping sound instead of 'bullet' sound". So
            # the hole is drawn, and the app plays `DRY_SOUND` INSTEAD of the
            # gunshot (`ShotOutcome.dry`): a gunshot would say a live round went
            # off. The reasoning that made it silent -- "a dry gun makes no noise"
            # -- is gone; a distinct non-gunshot sound tells him the app saw the
            # pull AND that it was not a live round, which silence could not.
            t = self.tallies[side]
            t.after_done += 1
            self._note_dry_camera(ev)
            name = versus.SIDE_NAMES[side] if which is not None else "magazine"
            self._say(f"{name} is empty: shot seen at ({ev.x:.0f},{ev.y:.0f}), the pop "
                      f"and its hole, not scored (after_done {t.after_done})")
            self.last_outcome = ShotOutcome(hit=False, target_id="round_done",
                                            unscored=True, draw_hole=True,
                                            play_sound=True, dry=True)
        else:
            self.last_outcome = mode.shot(ev, sctx)
            if counts:
                self._last_live_t = ev.t_end if ev.t_end is not None else ev.t
                if self._first_live_t is None:
                    self._first_live_t = self._last_live_t
                t = self.tallies[side]
                t.shots += 1
                t.clock.shot(ev.t)        # the instant the round broke, not the tick that handled it
                # Bookkeeping for the ghost score's record guard, filed beside the
                # clock and read only when the round ends: it decides nothing here.
                t.peaks.append(ev.peak)
                t.seen_t.append(ev.t_end if ev.t_end is not None else ev.t)
                if self.last_outcome.hit:
                    t.hits += 1
                    t.points += max(0, int(self.last_outcome.points))
                if t.shots == self._round_mag:
                    # The hold runs from THE SHOT'S OWN TIMESTAMP, never from the
                    # tick that handled it: the ball drills measured from the
                    # handling instant and he found it from the firing line.
                    self._hold_until[side] = (ev.t + self.hold_s) if self.hold_s > 0 else None
                    if which is not None:
                        self._say(f"{versus.SIDE_NAMES[which]} has used its allotment: "
                                  f"{t.points} pts, {t.shots} shots, {t.hits} hits; "
                                  f"magazine {self._round_mag} is empty, its half goes quiet "
                                  f"after {max(0.0, self.hold_s):g} s")
                    else:
                        self._say(f"magazine {self._round_mag} is empty: {t.points} pts, "
                                  f"{t.shots} shots, {t.hits} hits; the results follow "
                                  f"after {max(0.0, self.hold_s):g} s")
                    idx = (0, 1) if self.versus else (0,)
                    if all(self.tallies[i].shots >= self._round_mag for i in idx) and not self._dry_open:
                        self._dry_open = True
                        self._dry_all_since = self._last_live_t
                self._maybe_finish(ctx)
        return self._after_shot(ev, ctx)

    def _after_shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        # THE DRILL IS ASKED FIRST, and the MENU button only claims a shot the
        # drill called a miss. A ball that drifts under the button stays
        # hittable, and a shot that would have scored can never open a menu
        # instead. The other precedence costs a hit and reads, from the firing
        # line, exactly like the app ignoring a good trigger pull.
        if (self._button_live and not self.last_outcome.hit
                and self.menu_button.contains(ev.x, ev.y, ctx)):
            # IT ARMS THE BUTTON AND CHANGES NOTHING ELSE. The outcome the
            # drill returned stands: the shot counts exactly as it counts
            # today, it bangs, and it leaves its hole.
            #
            # Deliberately not reported as a hit. The hit rate is the one
            # number this app exists to measure honestly, and a corner button
            # is not a target -- calling it a hit would inflate the rate with
            # presses of a menu, and calling it a miss would deflate it for a
            # shot he aimed exactly where he meant to. Leaving the drill's own
            # verdict alone adds no new distortion in either direction.
            #
            # The feedback is the button itself: it visibly arms, which is
            # what "nothing happened" needed answering with.
            self.menu_button.hit(ev.t)
            # THE SELECTION CUE, on EACH of the button's two hits (the first arms it
            # and already says so; a cue confirming that it landed is the point),
            # one millisecond after the sound, never instead of it.
            self.last_outcome.cue = SELECT_SOUND
        return self.last_outcome

    # A phantom detection arriving in the same instant as the verdict would
    # otherwise skip the score before he has read it (92 phantoms once arrived
    # in twenty seconds on this bench), so the box ignores shots for this long
    # after it is first painted. ONE hit after that, not two: two is reserved
    # for EXIT, where the cost is losing the session; here the cost is only
    # starting the next round early.
    BOX_ARM_S = 0.8

    def _round_over_shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        """A shot on the ROUND-OVER SCREEN: anywhere on it, including the NEXT
        ROUND box and the MENU button.

        IT ANSWERS: the hole is drawn and a sound plays, on every shot anywhere on
        it. His words (2026-10-20): "On that screen around the results, I need to
        hear the number of shots when I shoot the round or when I shoot the menu
        and I need to see the holes on that screen. The amount of bullets in the
        magazine does not change the fact that, on the end round screen, I see the
        bullet holes and hear the sounds."

        WHICH SOUND is the same rule as everywhere: the magazine is empty, so it is
        the bubble pop (`DRY_SOUND`), not the gunshot (`ShotOutcome.dry`). That is
        the orchestrator's reading of his two rulings together (he gets holes AND
        sounds, and the sound is honest about the gun being dry); he may narrow it.
        On the NEXT ROUND box the selection cue (`SELECT_SOUND`) follows the pop
        by a millisecond, once the box is armed, i.e. when the shot actually
        starts the round: a cue for a shot that did nothing would confirm nothing.

        The reason it answers at all, so nobody "fixes" it back to silence: this
        screen is not a drill, it is a screen he navigates BY SHOOTING, the same
        way he does the menu. A control he shoots has to answer him, and "nothing
        happened" is the complaint every affordance on it exists to answer.

        What stays true: it is `unscored` (neither a hit nor a miss, so the hit
        rate does not move) and it IS counted as a dry pull and in `after_done`
        (the magazine is empty; a pull at the box is still a pull). The box is
        armed `BOX_ARM_S` after it is first painted so a phantom arriving with
        the verdict cannot skip the score unread; a shot before that still leaves
        its hole and the pop, which tells him the app saw it while the round
        deliberately does not start yet.
        """
        in_box = self.round_box(ctx).contains(int(ev.x), int(ev.y))
        armed = (in_box and self._box_seen_at is not None
                 and ev.t - self._box_seen_at >= self.BOX_ARM_S)
        if armed:
            self.request_next_round("shot")
        side = self._layout(ctx).route(ev.x) if self.versus else 0
        self.tallies[side].after_done += 1
        self._note_dry_camera(ev)
        what = ("round box" if in_box else "round-over screen")
        self._say(f"{what}: shot seen at ({ev.x:.0f},{ev.y:.0f}), "
                  f"{'armed: starting the next round' if armed else ('not armed yet: ignored' if in_box else 'drawn and heard, not scored')}; "
                  f"the pop and its hole")
        target = ("round_box" if armed else "round_box_unarmed") if in_box else "round_done"
        return ShotOutcome(hit=False, target_id=target, unscored=True, dry=True,
                           draw_hole=True, play_sound=True,
                           cue=SELECT_SOUND if armed else None)

    def _sides(self, ctx: RenderContext):
        """[(mode, context)] for everything that runs: one, or two."""
        if self.versus and self.mode2 is not None:
            return [(self.mode, self._side_ctx(ctx, 0)),
                    (self.mode2, self._side_ctx(ctx, 1))]
        return [(self.mode, ctx)]

    def _live_sides(self, ctx: RenderContext):
        """[(side index or None, mode, context)] for every half that is still
        SHOOTING. A half that has used its allotment is quiet: it is not
        painted, not ticked and has no solid shapes, because a target that
        looks live and ignores a good trigger pull is this project's most
        reported complaint. Single player goes quiet the same way, once its one
        magazine is empty (it is the whole round)."""
        if self.versus and self.mode2 is not None:
            return [(i, m, c) for i, (m, c) in enumerate(self._sides(ctx))
                    if not self._side_quiet(i, ctx.now)]
        if self.counts_ammo and self.round_over:
            return []              # the end screen stands on a clear screen
        return [(None, m, c) for m, c in self._sides(ctx)]

    def _retire_done(self) -> None:
        """Tell each finished half's drill, once, that it will not be ticked
        again (a drill sharing a reveal must release its seat, or the half
        still shooting waits on it for ever)."""
        if not (self.versus and self.mode2 is not None):
            return
        for i, m in enumerate((self.mode, self.mode2)):
            if i not in self._retired and self._side_done(i):
                self._retired.add(i)
                m.retire()

    def advance(self, ctx: RenderContext) -> bool:
        self._now = ctx.now
        self._flush_dry(ctx.now)
        if self.mode is None or not self.mode.visible:
            return False
        moved = False
        # THE HOLD EXPIRES HERE: when the last hole has been on screen for
        # `hold_s` the round closes (and the results appear, clearing the holes).
        if self.counts_ammo and not self.round_over:
            self._maybe_finish(ctx)
            if self.round_over:
                moved = True
        # A half that is done is not drawn (see `paint`), so it is not ticked
        # either: a ball bouncing on a half nobody can score on would ask for
        # sixty repaints a second of a surface nobody is looking at.
        self._retire_done()
        for _i, mode, sctx in self._live_sides(ctx):
            if mode.advance(sctx):
                moved = True            # no short-circuit: both sides must tick
        if self._button_live and self.menu_button.advance(ctx):
            moved = True
        # The scoreline repaints when a number CHANGES -- twice a round, not
        # twice a tick. A running clock here would repaint a 4K surface sixty
        # times a second (Reaction's millisecond counter once did).
        # The same in single player, where the bullets are the readout: the
        # signature carries the count, so a bullet leaving repaints once.
        sig = self._signature()
        if sig != self._tally_sig:
            self._tally_sig = sig
            moved = True
        return moved

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if self.mode is not None and self.mode.visible:
            if self.versus and self.mode2 is not None:
                lay = self._layout(ctx)
                # A finished half paints NO drill (the verdict stands on a clear
                # screen; a done half says so where its targets were).
                for which, mode, sctx in self._live_sides(ctx):
                    p.save()
                    # A flash ring or a ball mid-bounce must not cross onto
                    # the other player's ground.
                    p.setClipRect(lay.side(which))
                    mode.paint(p, sctx)
                    p.restore()
                self._paint_chrome(p, ctx, lay)
            else:
                for _i, mode, sctx in self._live_sides(ctx):
                    mode.paint(p, sctx)
                if self.counts_ammo:
                    if self.round_over:
                        self._paint_verdict(p, ctx, self._layout(ctx))
                    else:
                        self._paint_single_strip(p, ctx)
        if self._button_live:
            self.menu_button.paint(p, ctx)

    def _paint_chrome(self, p: QPainter, ctx: RenderContext, lay) -> None:
        """The divider, the scoreline, each side's note and the verdict, drawn
        once from the FULL screen.

        Everything is grey read from the artwork ceiling (RULES B10), red
        excess 0, and at or below luminance 60, so none of it spends the
        lit-area budget; `tests/test_modes.py` section 25 measures that, and
        `tests/native_text_probe.py` measures the glyph edges through the real
        platform (ClearType writes coloured fringes the offscreen renderer
        cannot see). The scoreline repaints when a number CHANGES.
        """
        from PySide6.QtCore import QRect
        from ..display.layers import GREY_FAINT, GREY_MID, px

        if not self.round_over:
            p.fillRect(lay.divider, GREY_FAINT)     # nothing to divide once the verdict is up
        # GREY_MID, not GREY_TEXT: the live furniture is on screen WHILE shooting
        # is possible, so it is kept under luminance 60 and spends none of the
        # 2% budget the drills need (at the measured ceiling GREY_TEXT is 62).
        p.setPen(GREY_MID)
        n = self._round_mag
        for side, (cell, note) in enumerate(((lay.left_cell, lay.left_note),
                                              (lay.right_cell, lay.right_note))):
            t = self.tallies[side]
            who = self.seats[side]
            name = versus.SIDE_NAMES[side]
            line1 = f"{name}  {min(t.shots, n)} of {n}"
            match = self.totals[who].points + (0 if self.round_over else t.points)
            line2 = f"{t.points} pts" + (f"  match {match}" if self.round_no > 1 else "")
            if self.round_over:
                continue                      # the verdict screen states them once, itself
            self._draw_lines(p, QRect(cell), [line1, line2], px(ctx, 24), bold=True)
            self._paint_half_strip(p, ctx, lay, side)
            p.setPen(GREY_MID)
            msg = self.note_text(side)
            if self._side_quiet(side, ctx.now):
                # DONE: the half is empty, so the words go where the targets
                # were, in the middle of the half where he is looking, and may
                # be larger because nothing is left to collide with.
                self._draw_lines(p, self.done_box(side, ctx), [msg], px(ctx, 44), bold=True)
                continue
            self._draw_lines(p, self._clear_of_button(QRect(note), ctx), [msg], px(ctx, 22))
        if self.round_over and self.verdict is not None:
            self._paint_verdict(p, ctx, lay)

    def done_box(self, side: int, ctx: RenderContext):
        """The middle of one half, between the scoreline row and the bottom
        strip: where a finished half says it is finished."""
        from PySide6.QtCore import QRect
        from ..display.layers import bottom_strip_for
        lay = self._layout(ctx)
        half = lay.side(side)
        top = lay.scoreline.bottom() + 1
        bottom = ctx.rect.y() + ctx.rect.height() - bottom_strip_for(ctx.height)
        # The MIDDLE band of that ground, not all of it: the MENU button sits in
        # a corner, and a full-height box would intersect it and be shifted off
        # the centre of the half to clear it. A band 30% tall never reaches a
        # corner, so the words are in the middle and nothing has to move them.
        h = max(1, bottom - top)
        band = max(1, round(h * 0.30))
        return QRect(half.left(), top + (h - band) // 2, half.width(), band)

    def note_text(self, side: int) -> str:
        """What one side's own half says: that it is DONE (and how many shots
        the app saw), or which round it is. A method so the words are testable
        and the painter and the tests read the same ones."""
        n = self._round_mag
        t = self.tallies[side]
        if self._side_done(side):
            return versus.done_text(versus.SIDE_NAMES[side], min(t.shots, n), n)
        # No mention of the swap on screen (his request, 2026-10-19): it is
        # silent machinery, in the log and nowhere else. Do not restore it.
        return f"round {self.round_no}"

    def _clear_of_button(self, box, ctx: RenderContext):
        """`box` with the MENU button taken out of it. The button sits in one of
        four corners and the note row runs along the top, so a side's own
        message could be drawn under it; the message keeps the larger part
        that is left."""
        from PySide6.QtCore import QRect
        if not self._button_live:
            return box
        gap = max(2, round(ctx.height * Mode.MENU_KEEPOUT_MARGIN_FRAC))
        btn = self.menu_button.rect(ctx).toAlignedRect().adjusted(-gap, -gap, gap, gap)
        if not box.intersects(btn):
            return box
        left_w = max(0, btn.left() - box.left())
        right_w = max(0, box.right() + 1 - (btn.right() + 1))
        if right_w >= left_w:
            return QRect(btn.right() + 1, box.top(), right_w, box.height())
        return QRect(box.left(), box.top(), left_w, box.height())

    @staticmethod
    def emphasised(lines) -> "list":
        """`lines` with every number bold and one grey step up (`GREY_HI`, 65 of
        the artwork ceiling of 71: no brighter than a target outline, so nothing
        new enters the screen the camera watches). ONE rule (`runs.emphasise`) for
        every block on the round screen; the words around each number keep the
        weight and colour the call gives them. The result is a list of `RunLine`s,
        which are `str`s, so anything reading the words sees the words."""
        from ..display.layers import GREY_HI
        from .runs import emphasise
        return [emphasise(t, GREY_HI) for t in lines]

    @staticmethod
    def _run_fonts(size: int, bold: bool):
        """(font, metrics) for one weight at one size, built once for the PROCESS
        (`_FONTS`): a font is not free (the Windows font engine cost this app a
        2 fps afternoon on 2026-10-20 and the round screen 19 fps on 2026-10-22),
        and the keys are integers, so this holds a few hundred entries at most."""
        from PySide6.QtGui import QFontMetricsF
        from ..display.layers import mono
        # The font factory is part of the key: a check that substitutes `mono` (the
        # run-based lines are tested with a deliberately wider bold) must get ITS
        # fonts, not whatever the real factory cached earlier, and must not leave
        # its fonts behind for the next caller once it restores the real one.
        key = (mono, int(size), bool(bold))
        got = _FONTS.get(key)
        if got is None:
            if len(_FONTS) >= _FONTS_MAX:
                _FONTS.clear()          # unreachable in practice; bounded all the same
            f = mono(key[1], bold=key[2])
            got = _FONTS[key] = (f, QFontMetricsF(f))
        return got

    @staticmethod
    def _runs_of(line):
        """The runs a line is drawn from: its own if it is a `RunLine`, else the
        whole string as one run that inherits the call's weight and the pen."""
        from .runs import Run, RunLine
        return line.runs if isinstance(line, RunLine) else (Run(str(line)),)

    @staticmethod
    def line_width(line, size: int, bold: bool = False) -> float:
        """The advance of one line at `size`: the SUM of each run's advance under
        ITS OWN font. A bold run is wider than the same characters regular, so the
        joined string measured with one font understates the line and the box is
        overrun (RULES B5, bitten three times by a size that was estimated). For a
        plain string this is exactly `QFontMetricsF.horizontalAdvance`, as before.
        Public so that the checks measure a line the way the painter does."""
        total = 0.0
        for r in ModeHost._runs_of(line):
            _f, fm = ModeHost._run_fonts(size, bold if r.bold is None else bool(r.bold))
            total += fm.horizontalAdvance(r.text)
        return total

    @staticmethod
    def line_height(line, size: int, bold: bool = False) -> float:
        """The height of one line: the tallest font among its runs. For a plain
        string this is exactly `QFontMetricsF.height()`, as before."""
        return max(ModeHost._run_fonts(size, bold if r.bold is None else bool(r.bold))[1].height()
                   for r in ModeHost._runs_of(line))

    @staticmethod
    def fit_size(box, lines, start: int, bold: bool = False, floor: int = 9) -> int:
        """The size `_draw_lines` draws `lines` at in `box`: the largest <= `start`
        at which every line fits 94% of the box's width and all of them 96% of its
        height, `floor` if none does. Found by bisection (`_largest_fitting`) and
        REMEMBERED per (box, lines, start, floor, weight), so a screen that has not
        changed does no font work on its next paint. The key holds every run's text
        and resolved weight, since a bold run is wider; colour changes no size and
        is left out. Public so the checks compare it with the one-point-at-a-time
        walk it replaced, through the real font engine."""
        from ..display.layers import mono
        start0 = max(floor, int(start))
        width, height = box.width(), box.height()
        sig = tuple(tuple((r.text, bold if r.bold is None else bool(r.bold))
                          for r in ModeHost._runs_of(t)) for t in lines)

        def fits(s: int) -> bool:
            wide = max(ModeHost.line_width(t, s, bold) for t in lines)
            high = sum(ModeHost.line_height(t, s, bold) for t in lines)
            return wide <= width * 0.94 and high <= height * 0.96

        return _fit_remember(("lines", mono, width, height, sig, start0, floor, bool(bold)),
                             lambda: _largest_fitting(fits, start0, floor))

    @staticmethod
    def _draw_lines(p: QPainter, box, lines: "list[str]", start: int, bold: bool = False,
                    floor: int = 9) -> int:
        """Centre `lines` in `box` at the largest size (<= `start`) that fits
        both ways, measured from the font and never guessed (RULES B5: a guessed
        advance once put text off the edge of its own box). Returns the size.

        A line may be a `RunLine` (modes/runs.py): a str that also carries runs,
        each with its own weight and colour. A caller that passes plain strings
        gets exactly the old behaviour -- the same fit rules, and Qt's own
        centring. Only when a line has runs does it go through the per-run path:

          * the FIT sums each run's advance under its own font (`line_width`), so
            a bold run cannot push the line past the box;
          * the PLACEMENT centres the line by that total width and then draws the
            runs left to right from there, each with its own font and pen. Qt's
            AlignCenter cannot do that and is not asked to.

        A run's weight and colour default to the call's (`bold`, the painter's
        pen), so a label keeps the weight and brightness it always had.
        """
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QPen
        from .runs import RunLine

        size = ModeHost.fit_size(box, lines, start, bold, floor)
        base_font = ModeHost._run_fonts(size, bold)[0]
        if not any(isinstance(t, RunLine) for t in lines):
            p.setFont(base_font)
            p.drawText(box, Qt.AlignCenter, "\n".join(lines))
            return size
        pen = p.pen()
        high = sum(ModeHost.line_height(t, size, bold) for t in lines)
        y = box.top() + (box.height() - high) / 2.0
        for t in lines:
            x = box.left() + (box.width() - ModeHost.line_width(t, size, bold)) / 2.0
            h = ModeHost.line_height(t, size, bold)
            ascent = max(ModeHost._run_fonts(size, bold if r.bold is None else bool(r.bold))[1].ascent()
                         for r in ModeHost._runs_of(t))
            for r in ModeHost._runs_of(t):
                f, fm = ModeHost._run_fonts(size, bold if r.bold is None else bool(r.bold))
                p.setFont(f)
                p.setPen(pen if r.color is None else QPen(r.color))
                p.drawText(QPointF(x, y + ascent), r.text)
                x += fm.horizontalAdvance(r.text)
            y += h
        p.setPen(pen)
        p.setFont(base_font)
        return size

    def side_block(self, side: int) -> "list[str]":
        """One side's column at the top of the verdict screen, units spelled out:
        its name, "20 of 20 shots" and "150 points" -- THIS ROUND's, and nothing
        else (the match is in the log, not on the screen). Single player has no
        name: "17 of 17 shots" over "240 points". A method so the tests
        read the words the painter draws."""
        n = self._round_mag
        t = self.tallies[side]
        # The points line carries its CEILING ("65 of 75 points", 2026-10-22): what
        # the ghost took on this round, the number AIM % is computed from. No card
        # (the score failed and `_score_round` deliberately set both to None) or an
        # unscorable one falls back to plain "65 points", so the round's own end is
        # never made to depend on its score. ONE rule for both screens (his ruling:
        # two players match single player), read from this SIDE's own card.
        # A DISCARDED round keeps its ceiling: "70 of 75 points" is a fact about the
        # drill and the tally he can check, not a judgement of the round, and the
        # block keeps the shape (and the type size) it has on every other round.
        card = self.cards[side]
        ceiling = (card.gs.ghost_points
                   if card is not None and (card.gs.scorable or card.gs.discarded) else None)
        if not self.versus:
            # SINGLE PLAYER, in his order and nothing else (2026-10-20): shots,
            # hits, misses, points, seconds. Misses come from the SAME tally the
            # shots and hits come from (`round_misses`), never recomputed from
            # something else; the seconds are first to last LIVE shot.
            return [versus.shots_word(t.shots), versus.hits_word(t.hits),
                    versus.misses_word(self.round_misses(side)),
                    versus.points_of_word(t.points, ceiling),
                    versus.seconds_word(self.round_seconds())]
        # TWO PLAYERS: the points AND the efficiency, both numbers for both
        # sides, the better of each marked, so the trade is visible rather than
        # inferred (points win the round; efficiency is shown beside it).
        # (Efficiency is read through `shown_efficiency`, so a DISCARDED side shows
        # `EFFICIENCY  --` and can be neither marked BEST nor compared: a figure the
        # app will not honour is not on the screen, and not set against the other
        # side's either.)
        pts_better = efficiency.better(self.tallies[0].points, self.tallies[1].points)
        eff_better = efficiency.better(self.shown_efficiency(0), self.shown_efficiency(1))
        pts = versus.points_of_word(t.points, ceiling) + ("  BEST" if pts_better == side else "")
        eff = (efficiency.headline(self.shown_efficiency(side), bool(self._round_reasons))
               + ("  BEST" if eff_better == side else ""))
        return [versus.SIDE_NAMES[side], f"{min(t.shots, n)} of {n} shots", pts,
                versus.seconds_word(self.round_seconds(side)), eff]

    # -- efficiency (modes/efficiency.py holds the arithmetic) ----------------

    def note_measurement(self, now: float, covered: bool = False, moved_count: int = 0,
                         margin: "float | None" = None, bar: "float | None" = None) -> None:
        """One reading of the app's own measurement, from numbers it already
        holds, for the honesty mark. Ignored once the round is over (the reasons
        are frozen at its end) and costs nothing in the shot path: it is called
        from the UI tick."""
        if self.round_over or not self.counts_ammo:
            return
        self.health.sample(now, covered=covered, moved_count=moved_count,
                           margin=margin, bar=bar)

    def side_efficiency(self, side: int = 0) -> "float | None":
        """Points per second for one side of this magazine, from its own clock."""
        return efficiency.efficiency(self.tallies[side].points, self.round_seconds(side))

    def side_discarded(self, side: int = 0) -> bool:
        """This side's round was DISCARDED (`ghost.record_fitness`): it has no
        score and no efficiency to show or to keep. False before a round has been
        scored and when the score failed."""
        card = self.cards[side]
        return card is not None and card.discarded

    def shown_efficiency(self, side: int = 0) -> "float | None":
        """The efficiency the SCREEN may print for one side: `side_efficiency`, or
        None for a discarded round. `side_efficiency` is the raw arithmetic and
        stays for the log; every screen reads this one, so a discarded round shows
        no figure on any of them (his ruling, 2026-10-22: the efficiency is the same
        round and is shown and recorded, or neither)."""
        return None if self.side_discarded(side) else self.side_efficiency(side)

    def efficiency_lines(self, side: int = 0) -> "list[str]":
        """The two lines under the single-player score: the figure, then its
        arithmetic. Constant shape at every round. A discarded round has no figure
        (`EFFICIENCY  --`) and its arithmetic line is left blank: the working of a
        number that is not there."""
        t = self.tallies[side]
        if self.side_discarded(side):
            return [efficiency.headline(None, bool(self._round_reasons)), ""]
        return [efficiency.headline(self.side_efficiency(side), bool(self._round_reasons)),
                efficiency.arithmetic_text(t.points, self.round_seconds(side))]

    def mark_text(self) -> str:
        return efficiency.mark_text(self._round_reasons)

    def mark_lines(self) -> "list[str]":
        """The honesty footnote as exactly two lines, blank when nothing fired,
        so the block it lives in is the same height with and without it."""
        return efficiency.wrap_mark(self.mark_text())

    def efficiency_result_line(self) -> str:
        """Two players: the sentence beside the points result about who was more
        efficient. "" when equal or undefined (the line stays, blank)."""
        v = self.verdict
        pw = v.winner if (v is not None and v.kind == "win") else None
        return efficiency.efficiency_sentence(pw, self.shown_efficiency(0),
                                              self.shown_efficiency(1), versus.SIDE_NAMES)

    def _say_efficiency(self) -> None:
        """The log carries every round's figures, marked or not: the arithmetic
        in the same words as the screen, and the reasons when there are any."""
        for side in ((0, 1) if self.versus else (0,)):
            t = self.tallies[side]
            name = versus.SIDE_NAMES[side] if self.versus else "magazine"
            self._say(f"round {self.round_no} efficiency, {name}: "
                      f"{efficiency.headline(self.side_efficiency(side)).strip()} "
                      f"({efficiency.arithmetic_text(t.points, self.round_seconds(side))})")
        self._say("round efficiency mark: " + (self.mark_text() or "none, the measurement was fine"))

    # -- the ghost score (lsa/scorecard.py holds the arithmetic and the words) ----

    def _score_round(self) -> None:
        """Score each side against the ghost, settle the machine's bests, and put
        everything in the log. Called ONLY from `_finish_round`, which runs once
        per round, so the bests are offered once per round.

        NEVER ALLOWED TO COST THE ROUND ITS END. The score is shown on the end
        screen; it is not what ends the round. A fault in it is logged and the
        screen shows `SCORE  --`, not a magazine that never closes.

        A ROUND IS SOUND OR DISCARDED (his ruling, 2026-10-22), decided once by
        `ghost.record_fitness` inside `ghost.ghost_score` and reached here through
        `scorecard.make_card`. A sound round's score and efficiency are shown and,
        if they beat the book, recorded; a discarded round has neither to show or
        to keep, and the screens read `shown_efficiency`, `score_lines` and
        `side_discarded` so none of them prints a figure the book will not honour.
        Only the round's SPAN discards; the inputs for the cautions the log carries
        come from the round itself: each shot's pulse brightness
        (`ShotEvent.peak`), the time first-to-last, and what the microphone paired.
        """
        sides = (0, 1) if self.versus else (0,)
        players = 2 if self.versus else 1
        made: "list" = [None, None]
        try:
            for side in sides:
                mode = self.mode if side == 0 else self.mode2
                t = self.tallies[side]
                n = min(t.shots, self._round_mag)
                seconds = self.round_seconds(side)
                waits = bool(mode.ghost_has_waits)
                # The human's own clock, so the ghost is charged only the waits that
                # lie inside it (`Mode._ghost_wait_plan`). None while either end is
                # unknown (no live shot), which cannot be scored anyway.
                clock = t.clock
                within = (None if clock.first is None or clock.last is None
                          else (clock.first, clock.last))
                made[side] = _scorecard.make_card(
                    key=_bests.best_key(mode.name, self._round_mag, players),
                    human_points=t.points, human_elapsed=seconds, shots=n,
                    efficiency=efficiency.efficiency(t.points, seconds),
                    ceiling=mode.ghost_ceiling(), has_waits=waits,
                    available_at=(mode.ghost_available_at(n, self.fps, within=within)
                                  if waits else None),
                    declared_waits=mode.ghost_declared_waits(n) if waits else 0,
                    peaks=list(t.peaks),
                    corroboration=_ghost.Corroboration(
                        mic_ready=bool(self.mic_ready), heard_and_seen=len(t.heard_t),
                        deaf=len(t.deaf_t)),
                    book=self.bests)
            settled = _scorecard.settle([made[i] for i in sides], self.bests, _bests.stamp())
            for i, side in enumerate(sides):
                made[side] = settled[i]
        except Exception as exc:                       # noqa: BLE001 -- see the docstring
            self._say(f"score: could not be computed ({exc!r}); the end screen shows none")
            made = [None, None]
        self.cards = (made[0], made[1])
        # THE MODEL SPEAKS WHEN IT IS WRONG. 1000 is a hard ceiling (his ruling,
        # 2026-10-23): `ghost.ghost_score` clamps the SPEED share at 1.0, so the screen
        # and the book can never carry more. But a clamp that fires means OUR MODEL is
        # off -- GHOST_INTERVAL pessimistic, a wait mischarged to the ghost
        # (`Mode._ghost_wait_plan`; a shot fired at an empty screen during a wait can
        # cost the ghost 0.25 s, anything larger is a defect), or a drill's ceiling
        # understated -- and a clamp that said nothing would hide that behind a
        # correct-looking 1000, which is exactly what the stored 1146 was. So every
        # firing is a DEFECT line carrying the RAW figures, on EVERY drill (the cadence
        # case fires on drills that make nobody wait). A DISCARDED round is not
        # reported here: a round faster than the ghost is what discarding is for.
        for side in sides:
            card = made[side]
            mode = self.mode if side == 0 else self.mode2
            if card is None or mode is None:
                continue
            if card.gs.scorable and card.gs.clamped:
                self._say(f"score: DEFECT - on {mode.name}: "
                          + _scorecard.clamp_defect_text(card.gs))
            if card.gs.ceiling_exceeded:
                self._say(f"score: DEFECT - on {mode.name}: {card.gs.not_scorable}")
        for side in sides:
            self._say(_scorecard.describe(
                self.cards[side], f", {versus.SIDE_NAMES[side]}" if self.versus else ""))

    def score_lines(self, side: int = 0) -> "list[str]":
        """The four lines of the score block for one side, exactly as drawn: the
        headline, aim and speed, the best score so far (or NEW BEST SCORE
        RECORDED), the best efficiency so far (or its NEW). For a DISCARDED round
        the four lines say so and why, with no figure (`scorecard.discarded_lines`).
        Always four, whatever the round was (the screen is the same height at round
        9 as at round 1). The caveat is NOT here (2026-10-22); it is in the log. A
        method so the painter and the tests read the same words."""
        return _scorecard.lines(self.cards[side])

    def round_misses(self, side: int = 0) -> int:
        """Shots that did not hit, from the host's own tally of the drill's
        outcomes. The one place this subtraction lives."""
        t = self.tallies[side]
        return max(0, t.shots - t.hits)

    def round_seconds(self, side: int = 0) -> float:
        """First to last LIVE registered shot of this side's magazine, from each
        shot's own timestamp (`ShotEvent.t`, the moment the round broke; see
        `efficiency.Clock`). Dry pulls are not in it. NOT the session's
        `string_time_s`, which is one session-wide figure stamped with each
        pulse's last frame."""
        return self.tallies[side].clock.seconds

    def verdict_lines(self) -> "list[str]":
        """The MIDDLE of the verdict screen, as the lines it is drawn from: the
        heading and this round's result, two lines at EVERY round. The per-side
        shots and points are NOT here (they are `side_block`, at the top, and
        are not repeated). A method so the tests can read the words, and the
        painter and the words cannot drift apart.

        Round 3:   ROUND 3 COMPLETED / LEFT WINS by 26 points"""
        v = self.verdict
        if self.counts_ammo and not self.versus and self.round_over:
            return [f"ROUND {self.round_no}"]
        if not self.versus or not self.round_over or v is None:
            return []
        names = versus.SIDE_NAMES
        # Always three lines (the efficiency sentence is blank when there is none
        # to say), so the middle block is the same height at every round.
        lines = [f"ROUND {self.round_no} COMPLETED", versus.result_words(v, names),
                 self.efficiency_result_line()]
        # (The microphone line is NOT here: it is small, at the very bottom of
        # the screen, and only when it has a count -- see `mic_screen_text`. The
        # next-round prompt is the START box, not a line of text. The swap is
        # silent: it is in the log and nowhere else.)
        return lines

    def _mic_counts(self):
        base_round = self.heard_unseen - self._heard_round_base
        base_match = self.heard_unseen - self._heard_match_base
        return self.mic_ready, base_round, base_match if self.round_no > 1 else None

    def mic_log_text(self) -> str:
        return versus.heard_text(*self._mic_counts())

    def mic_screen_text(self) -> str:
        """The microphone line for the verdict screen: empty when it has
        nothing useful to say. A method so the tests read the same words."""
        return versus.heard_on_screen(*self._mic_counts())

    # -- the verdict screen's geometry (ONE definition: paint, shot and click) --

    def _verdict_zone(self, ctx: RenderContext):
        from ..display.layers import bottom_strip_for
        lay = self._layout(ctx)
        top = lay.scoreline.bottom() + 1
        bottom = ctx.rect.y() + ctx.rect.height() - bottom_strip_for(ctx.height)
        return lay, top, bottom

    def mic_box(self, ctx: RenderContext):
        """The very bottom of the screen, above the bottom-strip furniture. Nothing
        is drawn in it any more (2026-10-22: the dry-pull and microphone lines left
        the screen); it survives as the anchor `round_box` is positioned from, so the
        NEXT ROUND button stays exactly where he has been shooting it."""
        from PySide6.QtCore import QRect
        lay, _top, bottom = self._verdict_zone(ctx)
        h = max(8, round(ctx.height * 0.035))
        w = round(ctx.rect.width() * 0.92)
        return QRect(int(lay.centre_x - w / 2), bottom - h, w, h)

    def round_box(self, ctx: RenderContext):
        """The box that starts the next round when it is shut."""
        from PySide6.QtCore import QRect
        lay, _top, _bottom = self._verdict_zone(ctx)
        w = round(ctx.rect.width() * 0.26)
        h = round(ctx.height * 0.15)
        gap = round(ctx.height * 0.02)
        mic = self.mic_box(ctx)
        return QRect(int(lay.centre_x - w / 2), mic.top() - gap - h, w, h)

    def _paint_verdict(self, p: QPainter, ctx: RenderContext, lay) -> None:
        from PySide6.QtCore import QRect
        from ..display.layers import GREY_MID, px

        lines = self.verdict_lines()
        if not lines:
            return
        p.save()
        # GREY_MID like the rest, though no drill is on this screen: a shot can
        # still be fired at it (it is seen, drawn and scores nothing), and text
        # at GREY_TEXT lit 1.08% of a 4K screen when measured. Under luminance 60
        # it spends nothing.
        p.setPen(GREY_MID)
        _lay, top, _bottom = self._verdict_zone(ctx)
        rb = self.round_box(ctx)
        gap = round(ctx.height * 0.02)
        avail = max(1, rb.top() - gap - top)
        # Five lines per side (name, shots, points, seconds, efficiency) AND the
        # side's SCORE block beneath them, so the columns take 66% of the zone and
        # the middle keeps the rest. The block was added to a screen that was
        # already full: the columns' own five lines now get 0.396 of the zone
        # where they had 0.60, so their type is smaller than it was (a measured
        # trade, not an oversight) and the middle lost a sixth of its height.
        cols_h = round(avail * 0.66)
        # THE PER-SIDE COLUMNS, one over each half, at the top. They replace the
        # scoreline cells on this screen (see `_paint_chrome`): the numbers are
        # stated once.
        if self.versus:
            # Each column holds the side's five lines (name, shots, points,
            # seconds, efficiency) and, under them, its SCORE block. Fixed
            # fractions of the column zone, so the screen is the same height at
            # round 9 as at round 1.
            blk_h = round(cols_h * 0.60)
            sc_gap = round(cols_h * 0.02)
            sc_h = max(1, cols_h - blk_h - sc_gap)
            sc_lines_h = max(1, round(sc_h * 0.74))          # SCORE, aim/speed, the two bests
            # (The remaining 0.26 of sc_h used to hold the one caveat line. It no
            # longer reaches the screen -- 2026-10-22 -- and the box is left empty
            # rather than resized, so the figures keep the size they were measured at.)
            for side in (0, 1):
                half = lay.side(side)
                cw = round(half.width() * 0.90)
                x_col = int(half.center().x() - cw / 2)
                col = QRect(x_col, top, cw, max(1, blk_h))
                self._draw_lines(p, col, self.emphasised(self.side_block(side)), px(ctx, 40))
                score = self.score_lines(side)
                y_sc = top + blk_h + sc_gap
                # Here the SCORE headline is regular weight like the rest of the
                # column, so its number is lifted too (single player draws it as a
                # bold line of its own and leaves it).
                self._draw_lines(p, QRect(x_col, y_sc, cw, sc_lines_h), self.emphasised(score),
                                 px(ctx, 30))
        w = round(ctx.rect.width() * 0.92)
        x0 = int(lay.centre_x - w / 2)
        if not self.versus:
            # SINGLE PLAYER: ROUND n, then shots / hits / misses / points /
            # seconds -- a fixed six-line block, so the screen is the same height
            # at every round. (Two players keep their own layout below.)
            # Fixed fractions of the zone, so the block is the same height at
            # every round whether or not the honesty mark or a caveat is present:
            #   heading 0.13 | block 0.30 | efficiency 0.11 | score 0.08 + 0.12
            #   | mark 0.09     (plus three 0.01 gaps between the first four)
            # The result (heading to score, 0.77 of the zone) is CENTRED in the
            # zone, not hung from its top (2026-10-22). Removing the 0.05 caveat
            # box had left the whole stack against the top with 0.22 of empty
            # ground between it and the NEXT ROUND box; the leftover is now shared
            # above and below it, and the mark box (blank unless the measurement was
            # poor) sits in the lower share. Every BOX KEEPS ITS SIZE -- the
            # fractions above are unchanged, so every type size is the one that was
            # measured -- and only the offset moved. `round_box` is positioned from
            # `mic_box`, independently of all this, so the button he shoots at has
            # not moved.
            # The SCORE block joins the efficiency figure above it; it does not
            # replace it. It is shown here and nowhere else, at the round's end
            # (his instruction: "I only want the score displayed when the round
            # ends").
            head_h = round(avail * 0.13)
            g = round(avail * 0.01)
            col_h = max(1, round(avail * 0.30))
            e_h = max(1, round(avail * 0.11))
            sh_head = max(1, round(avail * 0.08))
            sh_rest = max(1, round(avail * 0.12))
            mark_h = max(1, round(avail * 0.09))
            stack_h = max(1, head_h) + g + col_h + g + e_h + g + sh_head + sh_rest
            # Never so far down that the mark box would not fit above the button.
            lead = max(0, min((avail - stack_h) // 2, avail - stack_h - g - mark_h))
            top1 = top + lead
            self._draw_lines(p, QRect(x0, top1, w, max(1, head_h)), lines[:1], px(ctx, 56), bold=True)
            cw = round(ctx.rect.width() * 0.45)
            y = top1 + head_h + g
            col = QRect(int(lay.centre_x - cw / 2), y, cw, col_h)
            self._draw_lines(p, col, self.emphasised(self.side_block(0)), px(ctx, 48))
            y += col.height() + g
            ew = round(ctx.rect.width() * 0.60)
            ebox = QRect(int(lay.centre_x - ew / 2), y, ew, e_h)
            eff_lines = self.efficiency_lines(0)
            # The figure is lifted; the arithmetic under it ("= 20 points / 10.7
            # seconds") is deliberately NOT: it is the working, smaller and quieter,
            # and emphasising it would have it compete with the result it explains.
            self._draw_lines(p, QRect(ebox.left(), ebox.top(), ebox.width(), ebox.height() * 55 // 100),
                             self.emphasised(eff_lines[:1]), px(ctx, 52), bold=True)
            self._draw_lines(p, QRect(ebox.left(), ebox.top() + ebox.height() * 55 // 100, ebox.width(),
                                      ebox.height() - ebox.height() * 55 // 100), eff_lines[1:], px(ctx, 32))
            y += ebox.height() + g
            score = self.score_lines(0)
            sw = round(ctx.rect.width() * 0.60)
            sx = int(lay.centre_x - sw / 2)
            # SCORE n stays a plain bold line (it is already the biggest thing in
            # the block); the lines under it get the numbers lifted.
            self._draw_lines(p, QRect(sx, y, sw, sh_head), score[:1], px(ctx, 52), bold=True)
            y += sh_head
            self._draw_lines(p, QRect(sx, y, sw, sh_rest), self.emphasised(score[1:]), px(ctx, 30))
            y += sh_rest + g
            self._draw_lines(p, QRect(x0, y, w, mark_h), self.mark_lines(), px(ctx, 24))
            self._paint_round_box(p, ctx, rb)
        else:
            # THE MIDDLE: the heading larger, then the result line(s).
            mid_top = top + cols_h
            mid_h = max(1, avail - cols_h)
            head_h = round(mid_h * 0.40)
            self._draw_lines(p, QRect(x0, mid_top, w, max(1, head_h)), lines[:1], px(ctx, 56),
                             bold=True)
            if lines[1:]:
                # The result lines in the upper part of what is left, the honesty
                # mark (two reserved lines, blank when nothing fired) below them.
                rest = max(1, mid_h - head_h)
                res_h = rest * 62 // 100
                # The result line is bold end to end and the sentence under it has
                # its numbers lifted (2026-10-22): one rule, `runs.emphasise`.
                self._draw_lines(p, QRect(x0, mid_top + head_h, w, max(1, res_h)),
                                 self.emphasised(lines[1:]), px(ctx, 44))
                self._draw_lines(p, QRect(x0, mid_top + head_h + res_h, w, max(1, rest - res_h)),
                                 self.mark_lines(), px(ctx, 22))
            self._paint_round_box(p, ctx, rb)
        # NOTHING IS DRAWN BELOW THE NEXT ROUND BOX (his ruling, 2026-10-22: "no
        # need for this text"). The dry-pull and microphone lines used to sit here;
        # both still go to the log (`dry_log_text`, `mic_log_text`, and the
        # per-pull lines). `mic_box` stays because `round_box` is positioned from
        # it: removing it would move the button he shoots at.
        p.restore()

    def _paint_round_box(self, p: QPainter, ctx: RenderContext, rb) -> None:
        """"START ROUND 2" in an outlined box. Grey read from the artwork
        ceiling like everything here, at or below luminance 60 and red excess 0;
        sized from the font by `_draw_lines`, never a guess.

        THE HOW-TO LINE ("shoot it, click it or press N") WAS REMOVED AT HIS
        REQUEST (2026-10-19): "START" is the instruction. The box is still
        shootable, clickable and N-able exactly as before -- do not restore the
        sub-line as a fix for "nothing happened"; the box shutting is the
        feedback, and the first hit is armed 0.8 s after it is painted."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPen
        from ..display.layers import GREY_MID, px
        p.setPen(QPen(GREY_MID, max(2, px(ctx, 3))))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rb)
        if self._box_seen_at is None:
            self._box_seen_at = ctx.now          # armed BOX_ARM_S after this
        p.setPen(GREY_MID)
        # One box, one geometry, one start path on both screens.
        # Renamed on both screens (2026-10-20): "NEXT ROUND". The round number is
        # in the heading now, so the button need not carry it.
        self._draw_lines(p, rb, ["NEXT ROUND"], px(ctx, 48), bold=True)

    # -- the bullets ----------------------------------------------------------

    def single_strip_geometry(self, ctx: RenderContext):
        """The single-player strip: the lower corner OPPOSITE the MENU button,
        derived from `Mode.MENU_CORNER`, so moving the button moves it."""
        from . import magazine
        side = magazine.opposite_side(Mode.MENU_CORNER)
        return magazine.geometry(ctx.rect, ctx.height, self._round_mag,
                                 share=ctx.field_share, side=side,
                                 bottom=ctx.rect.y() + ctx.height)

    def half_strip_geometry(self, ctx: RenderContext, lay, side: int):
        """One half's strip: centred in its half, in the band under the
        scoreline, off the MENU keep-out where the button is in that half."""
        from . import magazine, split
        hctx = self._side_ctx(ctx, side)
        top = lay.mag_band.top() + round(ctx.height * split.MAG_MARGIN_FRAC)
        keep = Mode.menu_keepout(ctx) if self._button_live else None
        return magazine.geometry(lay.side(side), ctx.height, self._round_mag,
                                 share=hctx.field_share, side="centre",
                                 top=top, keepout=keep)

    def _paint_single_strip(self, p: QPainter, ctx: RenderContext) -> None:
        from . import magazine
        if ctx.diagnostics.get("measuring"):
            return          # furniture stands down while the app measures (RULES B2)
        magazine.paint(p, self.single_strip_geometry(ctx), self.remaining(0), ctx,
                       self._draw_lines)

    def _paint_half_strip(self, p: QPainter, ctx: RenderContext, lay, side: int) -> None:
        from . import magazine
        if ctx.diagnostics.get("measuring"):
            return
        magazine.paint(p, self.half_strip_geometry(ctx, lay, side), self.remaining(side),
                       self._side_ctx(ctx, side), self._draw_lines)

    def occluders(self, ctx: RenderContext) -> "list[Occluder]":
        """The solid shapes the active mode draws, for the layers beneath
        (the bullet holes) to leave out. A mode that draws only outlines
        returns nothing, which is the default."""
        if self.mode is not None and self.mode.visible:
            out: list = []
            for _i, mode, sctx in self._live_sides(ctx):
                out.extend(mode.occluders(sctx))
            return out
        return []

    def on_key(self, key: int) -> bool:
        # A round is a round: a key reaches both sides.
        taken = False
        for mode in (self.mode, self.mode2):
            if mode is not None and mode.on_key(key):
                taken = True
        return taken

    def on_click(self, x: float, y: float, ctx: RenderContext) -> bool:
        """A mouse click, forwarded to the active mode. Never a shot.

        A click carries none of what a shot carries -- no pulse, no peak, no
        camera position -- so it is delivered as what it is. A drill that
        scores clicks would report a hit rate he did not earn, which is the
        one number this whole app exists to measure honestly.
        """
        if (self.counts_ammo and self.round_over
                and self.round_box(ctx).contains(int(x), int(y))):
            # A click is not a phantom, so it needs no arming; it is the same
            # request a shot makes and the app drains it through one path.
            self.request_next_round("click")
            return True
        if self.mode is not None and self.mode.visible:
            mode, sctx, _which = self._routed(x, ctx)
            if mode.on_click(x, y, sctx):
                return True
        # Same precedence as a shot, and the same two clicks: the button is
        # the way back whether he is at the firing line or at the keyboard.
        # Still not a shot -- `shots` and `hits` are the gun's tallies and
        # nothing here touches them.
        if self._button_live and self.menu_button.contains(x, y, ctx):
            self.menu_button.hit(ctx.now)
            self._cues.append(SELECT_SOUND)        # a click has no gunshot: the cue plays alone
            return True
        return False

    def status(self) -> str:
        if self.mode is None:
            return ""
        if self.versus and self.mode2 is not None:
            return f"L: {self.mode.status()}   |   R: {self.mode2.status()}"
        return self.mode.status()
