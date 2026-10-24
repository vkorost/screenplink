"""The GHOST score: how a round compares with a perfect shooter at this system's pace.

PURE, like `selfcheck.py` and `exposure.py`: measurements in, a score out. No Qt,
no camera, no audio device, no file, no clock read inside the arithmetic. The
round screen (piece 3) and the modes (piece 2) feed it; nothing here reaches for
either.

THE CONCEPT
-----------
A GHOST plays the same round as the human. It hits every target in the highest
value zone available, and it shoots as fast as the round allows -- bounded by
the minimum shot interval the system accepts, and additionally charged any wait
for a target to appear. Both are scored the way the app already scores a round
(`modes/efficiency.py`): points over the clock from FIRST shot to LAST.

    SCORE = 1000 * (human points per second) / (ghost points per second)

The ghost is 1000 by construction. It factors exactly, and BOTH factors are
returned, because they are the diagnostic:

    SCORE = 1000 * (points share) * (speed share)
    points share = human points / ghost points         "how well you aimed"
    speed share  = ghost elapsed / human elapsed       "how fast you were"

1000 IS A CEILING (his ruling, 2026-10-23). The speed share is clamped at 1.0, so no
input produces a score above 1000; the raw share is kept (`GhostScore.speed_raw`) and
a clamp that fires is recorded (`clamped`) because it means the MODEL is off, not that
somebody beat the perfect shooter. The host logs it loudly as a DEFECT.

Measured over his real logs, 2026-10-20: median round 180, best 416 (165 points
of 170 in 9.34 s on Static Targets), median points share 0.82, median speed
share 0.21. He aims well and is slow, and a single number could not say so.

THE TWO THINGS THAT PROTECT A STORED RECORD
-------------------------------------------
1. A ghost's value per shot comes from the MODE ITSELF (see `Ceiling`), never
   from a table in this file. `exposure.Probe.bar()` once carried its own copy of
   the app's threshold rule, drifted, and the probe then chose the dimmest dot on
   the table (CLAUDE.md, 2026-10-16: "a seam between two pieces, tested on
   neither side"). A mode that cannot state its maximum is NOT SCORABLE, and so
   is a round in which the human out-scored the stated maximum: a number that is
   quietly wrong is worse here than a missing one.

2. A round is SOUND or it is DISCARDED, and there is no third state (his ruling,
   2026-10-22, after a round that showed `SCORE 256` over `BEST SCORE SO FAR 212`
   and recorded nothing: "how does it make sense to you to display the score and
   at the same time declare it invalid?"). A sound round's score is shown, and it
   IS the record if it beats the book: no second opinion is applied afterwards. A
   round that could not have been a human's -- three of his 26 real rounds were
   phantom storms, seventeen "shots" inside a few seconds, and one scored 535,
   which would have become an all-time best he could never legitimately beat --
   is DISCARDED WHOLE: `ghost_score` returns it with no score, no shares and no
   efficiency to show, `discarded=True`, and the reason. There is then nothing to
   record and nothing to disagree with. The figure it would have had survives in
   `withheld_score` for the LOG and for nothing else: never display a figure the
   app will not honour.

   WHAT DISCARDS A ROUND is decided by `record_fitness` and is deliberately narrow:
   the round's SPAN against the pure cadence floor, `(shots - 1) * GHOST_INTERVAL`
   (`MIN_SPAN_RATIO` of it), nothing else. A round at exactly the ghost's pace is
   SOUND and scores 1000: the ghost is reachable.
   Brightness and the microphone are measured and reported (`cautions`, which the
   log carries) and may NOT discard a round, alone or together: his real round of
   2026-10-22 (16 hits in 14.6 s) had a median peak of 28 and 16 of 17 shots deaf
   to the microphone, so a dim median is the normal state of his bench, and
   throwing a whole round away for it would be strictly worse than the silent
   refusal this replaced. See the numbers beside `MIN_SPAN_RATIO`.
"""

from __future__ import annotations

import math
import numbers
import statistics
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Sequence

# A mode's statement of its own maximum: the most the k-th shot of a fresh round
# (k counts from 1) can possibly be worth, on a perfect shot. None means "I
# cannot say", and the drill is then NOT SCORABLE. Piece 2 gives `Mode` the
# method that produces one of these, derived from the mode's OWN scorer.
Ceiling = Callable[[int], Optional[int]]


# ---------------------------------------------------------------------------
# THE FROZEN SCORING CONSTANT
# ---------------------------------------------------------------------------

# 0.25 s between two shots, as fast as the ghost may fire.
#
# MEASURED, 2026-10-20, across 3,020 consecutive-shot pairs in his logs: the
# smallest interval the detector has ever accepted between two shots at the SAME
# place on the screen is 0.254 s, and the histogram climbs sharply at 250 ms
# (58 pairs in the 250-300 ms bin against 9, 6, 10, 9 and 13 in the five bins
# below it). That is the tracker's 250 ms refractory window exactly -- so this is
# OUR OWN detector constant, not a physical limit of a trigger finger.
#
# !!! NEVER READ THIS FROM THE TRACKER'S REFRACTORY AT RUNTIME. !!!
# It is a FROZEN scoring constant. The score is directly proportional to it, so
# if anyone ever tunes the refractory and this followed, every personal best in
# history would silently change meaning. The score scales linearly with it: at
# 0.30 s his best round reads 499, at 0.35 s it reads 582 (against 416 here).
# CHANGING THIS NUMBER INVALIDATES EVERY STORED BEST. If it must change, the
# stored bests must be discarded or re-derived in the same change.
GHOST_INTERVAL = 0.25

# The ghost is 1000. A round that matches it in points and in speed scores this,
# and NOTHING scores more: the speed share is clamped at 1.0 (see `_arithmetic`).
SCORE_SCALE = 1000.0

# A raw speed share above 1.0 by more than this is a CLAMP FIRING (logged as a
# DEFECT by the host). Below it is float noise on a human who IS the ghost, whose
# share is 1.0 up to the last ulp. 1e-9 is far under any real timing difference
# (a camera frame is 1.7e-2 s on a 4 s round) and far over double rounding (~1e-16).
CLAMP_TOLERANCE = 1e-9
CLAMPED_SPEED = "speed"


# ---------------------------------------------------------------------------
# THE ROUND-VALIDITY TEST (still called `record_fitness`, `RecordFitness` and
# `Unfit`: the names are historic. Before 2026-10-22 they answered "may this round
# become a record?" and a round could be shown and refused. They now answer "is
# this round sound?", and an unsound one is discarded whole -- see the module
# docstring. `fit` means SOUND; `reasons` are what DISCARDED it.)
# ---------------------------------------------------------------------------

# THE DISCRIMINATOR, and it is the only one that discards. A round is impossible
# when its shots arrived faster than the tracker can accept them, and
# GHOST_INTERVAL is exactly that pace. So the floor is a fraction of the PURE
# CADENCE SPAN, (shots - 1) * GHOST_INTERVAL: the first shot's stamp to the last
# one's, which is the clock the human is judged on and the span the ghost itself
# takes on a drill with no waits. It is NOT a multiple of the ghost's total
# elapsed time, which on a drill that makes you wait includes the waiting and
# would put the floor far too high; and it is NOT shots * GHOST_INTERVAL, because
# there are (shots - 1) gaps between shots and no single ratio could then serve
# every magazine (the ghost itself would sit at 0.94 of that for 17 shots and 0.50
# for 2).
#
# The ratio is below 1.0 on purpose, so that a human who reaches the system's own
# limit -- the ghost, 1000 by construction -- is SCORED and not discarded for
# getting there. An earlier value of 1.2 capped the achievable score at about 833
# and threw away anyone who matched the ghost.
#
# HOW THE NUMBER WAS CHOSEN, from the real cases, as a ratio of the observed span
# to the cadence span (17 shots: 16 gaps * 0.25 s = 4.0 s):
#
#   storm (round line of 22:38:16)  17 detections   2.70 s   0.675  must be DISCARDED
#   storm (22:38:49)                17               2.20 s   0.550  must be DISCARDED
#   storm (2026-10-21 12:39:30)     17               3.30 s   0.825  must be DISCARDED
#   ----------------------- the floor goes in this gap -----------------------
#   a human AT the limit (= the ghost)               4.00 s   1.000  must be SCORED (1000)
#   2026-10-22 20:45, his real round  17 shots      14.57 s   3.64   must be SCORED (256)
#
# The nearest case to be discarded is 0.825 and the nearest to be scored is 1.000,
# so the line lies in (0.825, 1.0]. It is the MIDPOINT of that gap, 0.9125, taken
# as 0.91: 0.085 above the slowest storm (0.34 s at 17 shots) and 0.090 below the
# ghost (0.36 s). The upper margin also has to absorb the first-frame timestamp of
# a pulse being uncertain by a camera frame or two (17 ms at 60 fps, 33 ms at 30),
# which on a 17-shot span is under 1%. The lower side rests on three storms only:
# a fourth one slower than 3.3 s would be scored.
#
# WHERE IT IS THIN (stated, not hidden):
#   * Magazine 2: the floor is 0.2275 s against a ghost of 0.25 s, a margin of
#     22 ms, which one frame at 30 fps (33 ms) exceeds. Magazine 1 has no span at
#     all and is never judged (it is NOT SCORABLE, not discarded). Magazine 5:
#     0.91 s against 1.00 s. A small magazine therefore detects fewer storms: a
#     two-shot burst 0.23 s apart is indistinguishable from a person.
#   * Two pistols, and one pistol at two places. The tracker's 250 ms refractory
#     is SPATIAL (60 camera px), so the 0.254 s minimum was measured at one place;
#     two shots at different places can be accepted closer than 0.25 s, and the
#     LaserLyte is rated for about 5 pulls a second (0.2 s, ratio 0.8). The rule
#     cannot tell guns apart (a detection carries no gun id), so a single-player
#     string fired from two pistols, or alternating fast between far-apart
#     targets, can legitimately fall under this floor and be discarded. Two
#     players are judged per side, one gun each, which is the right unit. No
#     recording holds a confirmed two-gun round, so this is argued, not measured.
#
# WHY RATE AND NOT BRIGHTNESS: brightness cannot separate the storm from his real
# round -- 17 and 28 are both under 30 -- and the microphone cannot either,
# because the app's own bang had it deaf for 33 of the 34 shots between them.
# The span separates them: the storm took 0.68 of the cadence span and the real
# round 3.64. Six detections a second is faster than a LaserLyte can be fired;
# his real shooting is about one a second.
MIN_SPAN_RATIO = 0.91

# NOTED, NEVER DISCARDING: the median shot peak, against the laser/noise boundary.
# 30 is the measured boundary on this hardware: across two sessions and two
# exposures every noise single-frame detection scored 10..29 and every laser one
# 31..255 (CLAUDE.md, 2026-10-15), and his dimmest logged real shot in 575 is 31.
# The storms scored peak 11-18. But his RIGHT-HAND side images at 20-26 whenever
# the sensitivity map is withdrawn (the camera moved), so a median under 30 is
# what a genuine round on his bench looks like and it is reported in the log as a
# caution, never as a reason. It is a MEDIAN, so a few dim real shots in a good
# round do not even trigger the caution.
#
# This duplicates `CameraLab.SINGLE_FRAME_PEAK`, which lives in a module that
# imports Qt and so cannot be imported here. The duplication is guarded rather
# than trusted: tests/test_ghost.py reads that constant out of the source and
# fails if the two differ (RULES C14, a seam is tested across).
MIN_MEDIAN_PEAK = 30.0

# NOTED, NEVER DISCARDING: at least this fraction of the shots the microphone
# COULD have heard should have been heard AND seen. A JUDGEMENT, and said to be
# one: the data only brackets it. A phantom storm is 0% (nothing was fired, so
# nothing was heard), and the other direction on a healthy string is 89% of heard
# pulls detected (189 of 213, 2026-10-17; `working-v1`). Anything between would be
# a guess at a curve nobody has measured, so it is a majority. His ruling of
# 2026-10-21 -- a record may be set without a working microphone -- is why this
# cannot discard: a silent or absent microphone must never cost him a round.
MIN_CORROBORATED = 0.5


class Unfit(str, Enum):
    """Why a round was discarded (`reasons`) or merely looks doubtful (`cautions`).
    Plain strings, loggable.

    Only TOO_FAST discards a round today. NO_SCORE is a reason a round has no score
    but is not a discard (a drill with no stated maximum is simply not scorable).
    The other three are cautions."""

    NO_SCORE = "no_score"                    # nothing to record: see GhostScore.not_scorable
    PEAK_UNKNOWN = "peak_unknown"            # caution: no shot peaks supplied
    PEAK_TOO_LOW = "peak_too_low"            # caution: median under the laser/noise boundary
    TOO_FAST = "too_fast"                    # DISCARDS: span under MIN_SPAN_RATIO of the cadence floor
    MIC_UNCORROBORATED = "mic_uncorroborated"  # caution: it could, and mostly did not


# ---------------------------------------------------------------------------
# What a mode can say about itself
# ---------------------------------------------------------------------------

def constant_ceiling(value: int) -> Ceiling:
    """A drill whose every perfect shot is worth the same (a ring drill's top
    ring, Clear The Board's 1). The value must come from the mode's own scorer
    -- `ring_points(len(rings), 0)` for the ring drills -- not be typed twice."""
    def ceiling(k: int) -> Optional[int]:
        return value
    return ceiling


def escalating_ceiling(max_level: int, start_level: int = 1) -> Ceiling:
    """A drill whose reward climbs one per hit up to a cap: Escalating Ball and
    Bouncing Balls award the level a ball HAD reached, then raise it (capped at
    `MAX_LEVEL`). A ghost that keeps hitting the same ball gets 1, 2, 3, 4, 5, 5,
    5 ...: the k-th shot is worth min(max_level, start_level + k - 1). Seventeen
    shots total 75, so the ceiling is NOT constant within a round.

    `max_level` must be the mode's own `MAX_LEVEL`, passed in, never retyped."""
    def ceiling(k: int) -> Optional[int]:
        return min(int(max_level), int(start_level) + int(k) - 1)
    return ceiling


# ---------------------------------------------------------------------------
# The ghost's round
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GhostRound:
    """What the ghost earns and how long it takes."""

    shots: int
    points: int
    elapsed: float            # first shot to last, the clock the human is judged on
    cadence_elapsed: float    # (shots - 1) * GHOST_INTERVAL: the floor, waits or not
    waited: bool              # the elapsed time includes real target-availability waits


def ghost_points(shots: int, ceiling: Optional[Ceiling]) -> Optional[int]:
    """The most `shots` perfect shots can earn, or None when the mode cannot say.

    None (NOT SCORABLE) if there is no ceiling, if it answers None for any shot,
    or if it answers anything that is not a positive whole number. A zero-valued
    perfect shot would make the ghost earn nothing and divide the score by zero,
    and a negative one means the mode's statement is broken.
    """
    if ceiling is None or shots < 1:
        return None
    total = 0
    for k in range(1, int(shots) + 1):
        v = ceiling(k)
        if v is None or isinstance(v, bool) or not isinstance(v, numbers.Real):
            return None
        if not math.isfinite(v) or v <= 0 or v != int(v):
            return None
        total += int(v)
    return total


def _usable_waits(shots: int, available_at: Optional[Sequence[float]]) -> bool:
    """Can `available_at` be used? One finite number per shot, or it is ignored.

    Never raises: a malformed wait sequence is reported by the caller as an
    unusable input and the cadence-bound ghost is returned as a lower bound. A
    short sequence is not padded or extrapolated -- the module does not invent
    the waits it was not given.
    """
    if available_at is None or len(available_at) != shots:
        return False
    return all(isinstance(t, numbers.Real) and not isinstance(t, bool)
               and math.isfinite(t) for t in available_at)


def ghost_elapsed(shots: int, available_at: Optional[Sequence[float]] = None
                  ) -> "tuple[float, bool]":
    """(seconds from the ghost's first shot to its last, whether waits applied).

    `available_at[k]` is the earliest moment the k-th shot's target could be shot,
    on any one clock (only differences matter). The ghost fires the first shot the
    moment its target exists, and each later shot `GHOST_INTERVAL` after the
    previous one OR when its target appears, whichever is later. The wait before
    the FIRST shot is charged to nobody: the human's clock starts at his first
    shot too, so it is on neither side of the ratio.

    With no usable waits the answer is the cadence-bound ghost,
    `(shots - 1) * GHOST_INTERVAL`, and `waited` is False.
    """
    n = int(shots)
    cadence = max(0, n - 1) * GHOST_INTERVAL
    if n < 2 or not _usable_waits(n, available_at):
        return cadence, False
    t = float(available_at[0])
    start = t
    for k in range(1, n):
        t = max(t + GHOST_INTERVAL, float(available_at[k]))
    return t - start, True


# ---------------------------------------------------------------------------
# THE GHOST'S REACTION FLOOR, and turning a drill's waits into availability times
# ---------------------------------------------------------------------------

# The ghost cannot shoot a target the camera has not seen yet. It is "constrained
# by the speed of the system" (his words), so after a target becomes hittable it is
# charged this many CAMERA FRAMES before its pulse can begin. ONE, not two: a
# shot's position and timestamp are those of the FIRST frame of its pulse
# (`ShotEvent.t`), and the two-frame confirmation delay (`min_frames`) comes AFTER
# that stamp, so it never shows up on the clock the human is judged on.
#
# DERIVED FROM THE CONFIGURED FPS, never typed as seconds: the floor is
# GHOST_SEE_FRAMES / fps, so a 30 fps camera charges 33 ms and a 60 fps one 17 ms.
# It barely moves anything (a 2.35 s average wait becomes 2.37 s) -- it exists so
# the ghost is physically possible instead of a mathematically unreachable maximum.
GHOST_SEE_FRAMES = 1


def frame_seconds(fps) -> Optional[float]:
    """One camera frame in seconds, or None when `fps` is not a positive finite
    number (a floor that cannot be derived is not invented)."""
    if (isinstance(fps, bool) or not isinstance(fps, numbers.Real)
            or not math.isfinite(fps) or fps <= 0):
        return None
    return 1.0 / float(fps)


def _finite_range(lo, hi) -> bool:
    return all(isinstance(v, numbers.Real) and not isinstance(v, bool)
               and math.isfinite(v) and v >= 0 for v in (lo, hi)) and lo <= hi


def expected_uniform(lo, hi) -> Optional[float]:
    """The mean wait of a drill that draws it uniformly from [lo, hi]; None unless
    both ends are finite, non-negative and ordered. ARITHMETIC ONLY: the range
    itself must be handed in from the drill's own constants (`MIN_WAIT`,
    `MAX_WAIT`, ...), never typed here -- a rule copied into a second place is how
    `Probe.bar()` drifted."""
    return (float(lo) + float(hi)) / 2.0 if _finite_range(lo, hi) else None


def expected_max_uniform(lo, hi, floor) -> Optional[float]:
    """E[max(U, floor)] for U uniform on [lo, hi]: the mean wait of a drill whose
    next target appears after a random gap but never before a fixed delay (a fall
    that must finish first). `floor` at or below `lo` gives the plain mean; at or
    above `hi`, the floor itself. None unless all three are finite and >= 0."""
    if not _finite_range(lo, hi) or not _finite_range(0.0, floor):
        return None
    lo, hi, floor = float(lo), float(hi), float(floor)
    if floor <= lo:
        return (lo + hi) / 2.0
    if floor >= hi:
        return floor
    span = hi - lo
    return floor * (floor - lo) / span + (hi * hi - floor * floor) / (2.0 * span)


def available_from_waits(waits: Optional[Sequence[float]], fps
                         ) -> Optional["list[float]"]:
    """The `available_at` list for `ghost_elapsed`, built from a drill's WAITS.

    `waits[k]` is how long the drill makes the shooter wait, after the previous
    target was resolved, before the k-th shot's target can be shot (0.0: it is
    already there, as for every shot after the first on a figure that takes
    several hits, or a circle of a board already dealt). It is a property of the
    DRILL and not of the human: Reaction draws 1.2 to 3.5 s, Clear The Board waits
    2.2 s between boards.

    WHY THIS EXISTS INSTEAD OF HANDING THE ABSOLUTE TIMES THE HUMAN SAW. A target
    that appears after the previous one is resolved appears LATER for a slow
    shooter. Feeding the ghost the human's own clock makes it wait for him, so a
    slow human is compared with a ghost that is as slow as he is. The ghost has to
    be released by ITS OWN previous shot, which is what this builds: its clock
    starts at 0, and each target that has a wait becomes available that wait (plus
    the one-frame floor) after the ghost's own previous shot, exactly the recurrence
    `ghost_elapsed` uses (`max(previous + GHOST_INTERVAL, available)`).

    The first entry is ignored (the wait before the first shot is charged to
    nobody). None when `fps` is unusable or any wait is not a finite number >= 0.
    """
    frame = frame_seconds(fps)
    if frame is None or waits is None:
        return None
    if not all(isinstance(w, numbers.Real) and not isinstance(w, bool)
               and math.isfinite(w) and w >= 0 for w in waits):
        return None
    floor = GHOST_SEE_FRAMES * frame
    out: "list[float]" = []
    fired = 0.0
    for k, w in enumerate(waits):
        if k == 0:
            out.append(0.0)
            continue
        avail = fired + (float(w) + floor if w > 0 else 0.0)
        out.append(avail)
        fired = max(fired + GHOST_INTERVAL, avail)
    return out


def ghost_round(shots: int, ceiling: Optional[Ceiling],
                available_at: Optional[Sequence[float]] = None
                ) -> Optional[GhostRound]:
    """The ghost's whole round, or None when the drill is NOT SCORABLE."""
    pts = ghost_points(shots, ceiling)
    if pts is None:
        return None
    elapsed, waited = ghost_elapsed(shots, available_at)
    return GhostRound(shots=int(shots), points=pts, elapsed=elapsed,
                      cadence_elapsed=max(0, int(shots) - 1) * GHOST_INTERVAL,
                      waited=waited)


# ---------------------------------------------------------------------------
# Record fitness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Corroboration:
    """What the microphone says about a round's shots, handed in as numbers.

    The listener and the corroborator live in `lsa/audio` and own devices and
    threads; a pure module takes their conclusions, not the objects.

    mic_ready      the microphone has calibrated its threshold (it stays silent
                   until three camera-seen shots have taught it what his gun
                   sounds like, so before that it cannot vouch for anything)
    heard_and_seen shots of this round the corroborator paired as "both"
    deaf           shots of this round fired while the microphone was muted by
                   the app's own bang or not delivering audio. They are removed
                   from the denominator: a fast string legitimately outruns a
                   microphone that deafens itself for ~0.8 s after each shot,
                   and counting that as "uncorroborated" would punish speed.
    """

    mic_ready: bool
    heard_and_seen: int
    deaf: int = 0


@dataclass(frozen=True)
class RecordFitness:
    """Is this round SOUND? (The name is historic: see the section header.)

    fit        the round is sound. False means it has no score to show or keep:
               either it could not be scored at all (NO_SCORE) or it is DISCARDED
               (any other reason).
    reasons    what made it unsound, empty iff `fit`. Only TOO_FAST discards.
    text       one sentence a person can act on, built from `reasons`; "" when fit
    cautions   things that look doubtful and are NOT enough to discard anything:
               a dim median shot, no brightness recorded, a microphone that
               confirmed under half of the shots it could hear. They are present
               whether or not the round is sound, because a sound round with a
               caution is exactly what the log is for.
    caution_text   the same, as one sentence for the log; "" when none.
    """

    fit: bool
    reasons: "tuple[Unfit, ...]"
    text: str                # one sentence a person can act on; "" when fit
    cautions: "tuple[Unfit, ...]" = ()
    caution_text: str = ""


def record_fitness(*, scorable: bool, shots: int, human_elapsed: float,
                   peaks: Sequence[Optional[float]],
                   corroboration: Optional[Corroboration]) -> RecordFitness:
    """Judge whether a round could have been a human's. Every failing condition is
    reported, not just the first, so one verdict explains the whole round.

    ONLY THE SPAN DISCARDS (see `MIN_SPAN_RATIO` for the two real rounds that
    decide it). Brightness and the microphone are measured and returned as
    `cautions`; neither can make `fit` False, alone or together."""
    reasons: list[Unfit] = []
    parts: list[str] = []
    cautions: list[Unfit] = []
    notes: list[str] = []

    if not scorable:
        reasons.append(Unfit.NO_SCORE)
        parts.append("this round has no score to store")

    known = [float(p) for p in peaks
             if p is not None and not isinstance(p, bool) and math.isfinite(p)]
    if not known:
        cautions.append(Unfit.PEAK_UNKNOWN)
        notes.append("no shot brightness was recorded, so brightness cannot say "
                     "whether these were the laser")
    else:
        med = statistics.median(known)
        if med < MIN_MEDIAN_PEAK:
            cautions.append(Unfit.PEAK_TOO_LOW)
            notes.append(f"the median shot brightness was {med:.0f}, under the "
                         f"{MIN_MEDIAN_PEAK:.0f} a real laser usually reaches (noted "
                         f"only: brightness alone never discards a round, and a dim "
                         f"far side of the screen images like this)")

    floor = MIN_SPAN_RATIO * max(0, int(shots) - 1) * GHOST_INTERVAL
    if human_elapsed < floor:
        reasons.append(Unfit.TOO_FAST)
        parts.append(f"{int(shots)} shots in {human_elapsed:.1f} s is faster than a "
                     f"person can fire them (the least a real round of this many "
                     f"takes is {floor:.1f} s)")

    # THE MICROPHONE NEVER DISCARDS A ROUND (his ruling, 2026-10-21: "it is OK to
    # allow personal best w/o a working microphone"). When the microphone is not
    # ready, or the whole round fell inside the app's own self-mute, this check is
    # SKIPPED. When it IS ready and not wholly deaf and heard under half of what it
    # could, that is reported as a caution and nothing more: a flaky microphone
    # must not cost him a round he really fired. In both real rounds that decide
    # this (2026-10-21's storm, 2026-10-22's 256) the microphone was deaf for 33 of
    # the 34 shots, so it contributed nothing to either verdict and the span
    # carried both.
    if corroboration is not None and corroboration.mic_ready:
        audible = int(shots) - max(0, int(corroboration.deaf))
        if audible > 0 and corroboration.heard_and_seen < MIN_CORROBORATED * audible:
            cautions.append(Unfit.MIC_UNCORROBORATED)
            notes.append(f"the microphone confirmed only "
                         f"{max(0, corroboration.heard_and_seen)} of "
                         f"{audible} shots it could hear (noted only)")

    return RecordFitness(not reasons, tuple(reasons), "; ".join(parts),
                         tuple(cautions), "; ".join(notes))


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GhostScore:
    """Everything a screen needs, with both factors.

    `score` is None when it cannot honestly be given (`scorable` False, with the
    reason in `not_scorable`): never 0, never a guess. `shown` is the whole number
    to print. When `lower_bound` is True the drill makes the human wait for a
    target and the waits were not supplied, so the ghost is faster than a real
    one could be and the true score is AT LEAST `score` -- show it as "47+".

    `declared_waits` is a DIFFERENT, weaker state and never a lower bound: it counts
    the ghost's waits that came from the drill's OWN DECLARED EXPECTED WAIT rather
    than from this round, because the human missed and so fired shots for which no
    target ever appeared (the ghost never misses and needs one for each). The score
    is then an estimate that can be off either way, by exactly as much as those
    waits' real draws would have differed from their mean. 0 means every wait was
    measured from the round itself: a clean round is a pure measurement. Show a
    non-zero count as "about", not as "at least".

    `discarded` is a species of not-scorable, not a separate state of the score: a
    round that could not have been a human's (see `record_fitness`) comes back with
    `scorable` False, `score`, `shown` and both shares None, `discarded` True, and
    the reason in `not_scorable` and `record.text`. NOTHING on it can be drawn as a
    result, which is the point: the screen cannot show a figure that is not there.
    The figure the round would have had is kept in `withheld_score` so that the LOG
    can say what was thrown away (and whether it would have beaten a record).
    Nothing that draws may read it.
    """

    scorable: bool
    not_scorable: str            # "" when scorable
    score: Optional[float]
    shown: Optional[int]
    points_share: Optional[float]
    speed_share: Optional[float]
    human_points: int
    human_elapsed: float
    human_shots: int
    ghost_points: Optional[int]
    ghost_elapsed: Optional[float]
    lower_bound: bool
    lower_bound_why: str         # "" unless lower_bound
    record: RecordFitness
    declared_waits: int = 0      # ghost waits taken from the drill's declaration, not the round
    declared_why: str = ""       # "" when declared_waits is 0
    discarded: bool = False      # the round could not have been a human's: no result exists
    withheld_score: Optional[float] = None   # LOG ONLY: what a discarded round would have scored
                                             # (the CLAMPED figure: what the book would have been offered)
    speed_raw: Optional[float] = None        # LOG ONLY: ghost / human BEFORE the clamp. `speed_share`
                                             # is min(1.0, this). Survives a discard; None if never computed
    clamped: "tuple[str, ...]" = ()          # which factors the 1000-ceiling clamp fired on (CLAMPED_*)
    ceiling_exceeded: bool = False           # the human out-scored the drill's stated maximum: the
                                             # round is NOT SCORABLE and the host logs a DEFECT

    @property
    def has_declared_waits(self) -> bool:
        return self.declared_waits > 0


def _not_scorable(why: str, *, human_points: int, human_elapsed: float,
                  shots: int, ghost: Optional[GhostRound], peaks, corroboration,
                  lower_bound: bool = False, lower_bound_why: str = "",
                  declared_waits: int = 0, declared_why: str = "",
                  ceiling_exceeded: bool = False
                  ) -> GhostScore:
    return GhostScore(
        scorable=False, not_scorable=why, score=None, shown=None,
        points_share=None, speed_share=None, human_points=human_points,
        human_elapsed=human_elapsed, human_shots=shots,
        ghost_points=None if ghost is None else ghost.points,
        ghost_elapsed=None if ghost is None else ghost.elapsed,
        lower_bound=lower_bound, lower_bound_why=lower_bound_why,
        record=record_fitness(scorable=False, shots=shots,
                              human_elapsed=human_elapsed, peaks=peaks,
                              corroboration=corroboration),
        declared_waits=declared_waits, declared_why=declared_why,
        ceiling_exceeded=ceiling_exceeded)


def ghost_score(**kwargs) -> GhostScore:
    """Score one finished round against the ghost, or DISCARD it.

    Takes exactly `_arithmetic`'s keywords (listed and explained there; an unknown
    or missing one raises the same TypeError). The round is scored, and then judged
    by `record_fitness`: if anything but "has no score" made it unsound, the score
    is taken away and the result comes back `discarded` -- see `GhostScore`. There
    is no way to get a figure for a round this function has discarded, and none to
    get a discarded verdict for a round it scored. One decision, made here, so the
    screen, the record book and the log cannot disagree about it.
    """
    gs = _arithmetic(**kwargs)
    discarding = [r for r in gs.record.reasons if r is not Unfit.NO_SCORE]
    if not discarding:
        return gs
    why = "this round was DISCARDED: " + gs.record.text
    if gs.not_scorable:
        why += f" [it also could not be scored: {gs.not_scorable}]"
    return replace(gs, scorable=False, not_scorable=why, score=None, shown=None,
                   points_share=None, speed_share=None, discarded=True,
                   withheld_score=gs.score)


def _arithmetic(*, human_points: int, human_elapsed: float, shots: int,
                ceiling: Optional[Ceiling],
                has_waits: bool = False,
                available_at: Optional[Sequence[float]] = None,
                declared_waits: int = 0,
                peaks: Sequence[Optional[float]] = (),
                corroboration: Optional[Corroboration] = None) -> GhostScore:
    """The ghost score's arithmetic for one finished round, BEFORE it is judged.
    Call `ghost_score`, which judges it; nothing else should reach this.

    human_points / human_elapsed / shots   the round as the app already measured
        it: points earned, first-shot-to-last seconds (`efficiency.Clock`), and
        rounds spent. The ghost fires the same number of shots.
    ceiling        the mode's statement of its own maximum (None: not scorable).
    has_waits      True for a drill that makes the human wait for a target to
        appear (Reaction, Pop-Up, Static Target, Moving Target, Animated Target,
        and -- found while reading the modes -- Clear The Board, which pauses 2.2 s
        between boards). NOT the two ball drills: the replacement ball is spawned
        inside the very call that scored the hit, so nothing waits (checked in
        tests/test_modes.py section 47). Stated by the mode (`Mode.ghost_has_waits`),
        not guessed here.
    available_at   per-shot earliest-possible-shot times on the GHOST'S own clock:
        `Mode.ghost_available_at(shots, fps)` builds them, through
        `available_from_waits`, from the waits the drill recorded (NOT from the
        human's own appearance times, which would make the ghost wait for him).
        When `has_waits` and these are absent or unusable, the cadence-bound ghost
        is returned and the score is MARKED A LOWER BOUND. This module never
        estimates a wait: Reaction read 47 purely because of that gap, and the
        flag exists so the 47 is not mistaken for a measurement.
    declared_waits how many of the waits in `available_at` the DRILL declared
        (`Mode.ghost_declared_waits`) because the round had no target for that
        shot. THE APP PASSES 0 SINCE 2026-10-23: it stopped charging the ghost for
        targets the round never drew, because that let a human who MISSED outscore
        the ghost (a stored 1146 of a possible 1000); the parameter, the `declared`
        basis and the records stored under it remain so those records stay readable.
        They are already inside `available_at`; this only says how many, so
        the result can carry it. Counted only when the waits were used and the
        drill makes people wait; otherwise 0. Clamped to the shots whose wait
        matters (the first shot's wait is charged to nobody). A non-zero count
        is NOT a lower bound: it is an estimate, and is named as one.
    peaks          one brightness per shot (None where unknown) -- only reported,
        as a caution; brightness never discards a round.
    corroboration  the microphone's verdict on the round -- only reported, as a
        caution; the microphone never discards a round.

    Never raises on a degenerate round. One shot has no first-to-last time
    (human_elapsed 0 and the ghost's 0 too): there is nothing to divide by, so
    the answer is NOT SCORABLE with that reason, and nothing is recorded.
    """
    n = int(shots)
    hp = int(human_points)
    he = float(human_elapsed)
    kw = dict(human_points=hp, human_elapsed=he, shots=n, peaks=peaks,
              corroboration=corroboration)

    ghost = ghost_round(n, ceiling, available_at)
    if ghost is None:
        if ceiling is None:
            why = "this drill does not state a maximum shot value"
        elif n < 1:
            why = "no shots were fired"
        else:
            why = ("this drill's stated maximum is not a positive whole number "
                   "for every shot")
        return _not_scorable(why, ghost=None, **kw)

    lower = bool(has_waits) and not ghost.waited
    lower_why = ""
    if lower:
        supplied = available_at is not None
        lower_why = ("the target-appearance times do not match the shots"
                     if supplied else
                     "the wait for each target to appear is not known")

    declared = 0
    if (bool(has_waits) and ghost.waited and isinstance(declared_waits, numbers.Integral)
            and not isinstance(declared_waits, bool)):
        declared = max(0, min(int(declared_waits), n - 1))
    declared_why = (f"{declared} of the ghost's {n - 1} waits for a target to appear are "
                    f"this drill's own average, not measured from your round (you "
                    f"missed, so no target was drawn for those shots)"
                    if declared else "")
    kw.update(lower_bound=lower, lower_bound_why=lower_why,
              declared_waits=declared, declared_why=declared_why)

    if n < 2 or ghost.elapsed <= 0.0 or not math.isfinite(he) or he <= 0.0:
        return _not_scorable(
            "a round needs at least two shots at different moments: the clock "
            "runs from the first shot to the last",
            ghost=ghost, **kw)

    if hp > ghost.points:
        # The human beat the "maximum": the mode's statement is wrong. Refuse
        # rather than print a points share above 1 on a number that cannot be right.
        return _not_scorable(
            f"{hp} points exceeds the {ghost.points} this drill says is the most "
            f"possible, so its stated maximum is wrong",
            ghost=ghost, ceiling_exceeded=True, **kw)

    points_share = hp / ghost.points
    speed_share = ghost.elapsed / he
    # THE CEILING OF THE SCALE (his ruling, 2026-10-23: "clamp, 1000 is a hard
    # ceiling"). "1000 means you matched a perfect shooter" only means that if
    # nothing can exceed it. The SPEED share is clamped at 1.0 -- the physical
    # statement: a shooter who never misses and fires at the system's minimum
    # cadence cannot take LESS time than the ghost -- and the clamp is on the
    # factor, not on the final score, so the two percentages on the screen stay
    # honest with the headline. The RAW figure is kept (`speed_raw`) because it is
    # the evidence that the MODEL is off: GHOST_INTERVAL pessimistic, a wait
    # mismodelled, or a drill's ceiling understated. A clamp that hid that would
    # make 1000 reachable by a wrong model, so the host logs every firing as a
    # DEFECT. Float noise (1.0000000000000002 for a human who IS the ghost) is
    # clamped silently: only a raw share beyond CLAMP_TOLERANCE counts as firing.
    # AIM needs no clamp: `hp > ghost.points` was refused above, so hp / points
    # cannot exceed 1 here (a wrong ceiling is `ceiling_exceeded`, also logged loud).
    speed_raw = speed_share
    clamped: "tuple[str, ...]" = ()
    if speed_raw > 1.0 + CLAMP_TOLERANCE:
        clamped = (CLAMPED_SPEED,)
    speed_share = min(1.0, speed_share)
    # Built from the two factors, so SCORE == 1000 * share * share holds exactly
    # and a screen showing all three can never disagree with itself.
    score = SCORE_SCALE * points_share * speed_share
    fitness = record_fitness(scorable=True, shots=n, human_elapsed=he,
                             peaks=peaks, corroboration=corroboration)
    return GhostScore(
        scorable=True, not_scorable="", score=score,
        shown=int(math.floor(score + 0.5)),
        points_share=points_share, speed_share=speed_share,
        human_points=hp, human_elapsed=he, human_shots=n,
        ghost_points=ghost.points, ghost_elapsed=ghost.elapsed,
        lower_bound=lower, lower_bound_why=lower_why, record=fitness,
        declared_waits=declared, declared_why=declared_why,
        speed_raw=speed_raw, clamped=clamped)
