"""One efficiency figure per magazine: the arithmetic, the clock, the honesty
gate and the words, with no Qt and no mode.

PURE, like `versus.py`: numbers in, numbers and sentences out. The screens read
this module and nothing recomputes any of it anywhere else.

    EFFICIENCY = points earned / elapsed seconds

Specified with him on 2026-10-20 (the scoring note under requirements/); his
frame is "judged based on how you spend those 17 bullets, nothing else".

  * NO PENALTY CONSTANT. The magazine is the penalty: a round is spent whether
    it earns or not, so a miss (or a limb hit) costs its whole value by earning
    nothing. Nothing in this module subtracts anything.
  * THE CLOCK runs from the instant his FIRST round breaks to the instant his
    LAST one does, from each shot's own timestamp (`ShotEvent.t`, the first
    frame of the pulse, the moment the round breaks), never from the tick that
    handled it. `SessionStats.string_time_s` is NOT this clock: it is one
    session-wide accumulator (not per round, not per side) stamped with each
    pulse's LAST frame, so it is not used.
  * ONE FIGURE PER MAGAZINE. Nothing here averages, remembers or compares
    across rounds. A `Clock` belongs to one side of one round.
  * THE ARITHMETIC IS PRINTED, and it is computed from what is printed: the
    seconds are rounded to one decimal FIRST and the quotient taken from that,
    so "= 140 points / 24.6 seconds" always equals the figure above it to the
    digit (24.63 s would otherwise give 5.68 over a line that divides to 5.69).
  * THE HONESTY GATE (`RoundHealth`) marks a figure, and names why, when the
    app's own measurement was poor during that round. See there.

This is not USPSA or IPSC scoring and must not be described as such in any
output; see the requirements note for the arithmetic that was researched and
deliberately not adopted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Seconds are shown (and divided by) to one decimal. Below this a figure is
# undefined: a single shot has no first-to-last time, and dividing by a rounding
# of zero would print a meaningless enormous number.
MIN_SECONDS = 0.1


class Clock:
    """First round to last round of ONE side of ONE magazine, from the shots'
    own timestamps. Order-independent (it keeps the earliest and the latest), so
    a late tick or a shot handled out of order cannot move it."""

    def __init__(self) -> None:
        self.first: "float | None" = None
        self.last: "float | None" = None

    def shot(self, t: float) -> None:
        t = float(t)
        self.first = t if self.first is None else min(self.first, t)
        self.last = t if self.last is None else max(self.last, t)

    @property
    def seconds(self) -> float:
        if self.first is None or self.last is None:
            return 0.0
        return max(0.0, self.last - self.first)


def shown_seconds(seconds: float) -> float:
    """The seconds as printed: one decimal."""
    return round(max(0.0, float(seconds)), 1)


def efficiency(points: int, seconds: float) -> "float | None":
    """Points per second, from the seconds AS PRINTED. None when there is no
    time to divide by (one shot, or all of them in the same instant)."""
    s = shown_seconds(seconds)
    if s < MIN_SECONDS:
        return None
    return int(points) / s


def efficiency_word(eff: "float | None") -> str:
    """"5.69", or "--" when undefined. Two decimals, always, so the line is the
    same shape at every round."""
    return "--" if eff is None else f"{eff:.2f}"


def arithmetic_text(points: int, seconds: float) -> str:
    """The sum, one line, under the score: "= 140 points / 24.6 seconds"."""
    p = int(points)
    unit = "point" if p == 1 else "points"
    s = f"{shown_seconds(seconds):.1f}"
    tail = "" if efficiency(points, seconds) is not None else " (one shot: no time to divide by)"
    return f"= {p} {unit} / {s} seconds{tail}"


def headline(eff: "float | None", marked: bool = False) -> str:
    """"EFFICIENCY  5.69", with a star after it when the measurement was poor."""
    return f"EFFICIENCY  {efficiency_word(eff)}" + ("  *" if marked else "")


# -- two players ------------------------------------------------------------------

def better(a: "float | None", b: "float | None") -> "int | None":
    """Index of the higher efficiency, None when equal or when either is
    undefined (nothing can be said about a figure that is not there)."""
    if a is None or b is None or a == b:
        return None
    return 0 if a > b else 1


def efficiency_sentence(points_winner: "int | None", eff_a: "float | None",
                        eff_b: "float | None",
                        names: "tuple[str, str]" = ("LEFT", "RIGHT")) -> str:
    """The line about efficiency beside the points result. Points win the round;
    efficiency is shown next to it and the two may disagree (his own example).

      - the same shooter leads both: "LEFT is also more efficient, 3.10 to 2.75"
        (one line rather than two)
      - they disagree: "RIGHT is more efficient, 12.00 to 2.75"
      - equal or undefined: ""
    """
    who = better(eff_a, eff_b)
    if who is None:
        return ""
    hi, lo = (eff_a, eff_b) if who == 0 else (eff_b, eff_a)
    score = f"{hi:.2f} to {lo:.2f}"
    if points_winner is not None and points_winner == who:
        return f"{names[who]} is also more efficient, {score}"
    return f"{names[who]} is more efficient, {score}"


# -- the honesty gate -------------------------------------------------------------

# Conditions, all of them READ FROM MEASUREMENTS THE APP ALREADY HOLDS. This is
# the one list. Each names itself in the mark: one condition fires, one sentence
# says which (he lost part of a session to a message he could not act on).
#
# WHY A GATE AT ALL: an efficiency figure depressed by the app's own blindness is
# "you are pulling left" with a number on it (2026-10-17, when his holes landed
# wrong because the dot was clipping and the gain map was poisoned).
#
# The thresholds below are JUDGEMENTS, not measurements, and say so:
#   BAR_MOVE_RATIO    1.5x. The bar is max(floor*1.3, floor+6) (exposure.min_peak_for);
#                     a re-measurement that moves it by half again changes which
#                     of his dimmest shots can register (his dimmest logged shot
#                     is 31 against bars of 8-19). Fitted to nothing; a decision.
#   COVER_MIN_SECONDS 2.0. Shorter than the 1.0 s the screen search itself takes
#                     is noise; a second or two of alt-tab is not the round.
#   The signal-margin bar is NOT here: it is `selfcheck.MARGIN_READY` (3.0), the
#   app's own threshold, passed in by the caller rather than retyped.
BAR_MOVE_RATIO = 1.5
COVER_MIN_SECONDS = 2.0

COVERED = "covered"
MOVED = "moved"
THIN = "thin_margin"
BAR = "bar_moved"
UNSEEN = "unseen_pulls"


@dataclass
class RoundHealth:
    """What the app's own measurement was doing during ONE round. Fed once per
    UI tick by the app with numbers it already holds (no instrumentation of its
    own, nothing in the shot path); read once, at the end of the round.

    `begin()` at the round's start, `sample()` each tick while it runs, and
    `reasons()` at its end.
    """

    margin_floor: float = 3.0
    began: bool = False
    covered_s: float = 0.0
    _last_t: "float | None" = None
    _moved0: "int | None" = None
    _moved_now: int = 0
    _bar0: "float | None" = None
    _bar_lo: "float | None" = None
    _bar_hi: "float | None" = None
    _margin_min: "float | None" = None
    unseen: int = 0
    _notes: "list[str]" = field(default_factory=list)

    def begin(self, now: float) -> None:
        self.__init__(margin_floor=self.margin_floor)       # a new round: nothing carries
        self.began = True
        self._last_t = float(now)

    def sample(self, now: float, covered: bool = False, moved_count: int = 0,
               margin: "float | None" = None, bar: "float | None" = None) -> None:
        """One reading. `moved_count` is a running count of "the camera moved"
        findings (the first sample of a round is the baseline), `margin` the
        latest signal margin the app measured (None: none yet), `bar` the
        detector's current minimum peak."""
        if not self.began:
            return
        now = float(now)
        dt = max(0.0, now - (self._last_t if self._last_t is not None else now))
        self._last_t = now
        if covered:
            self.covered_s += dt
        if self._moved0 is None:
            self._moved0 = int(moved_count)       # the first reading is the baseline
        self._moved_now = int(moved_count)
        if margin is not None:
            self._margin_min = margin if self._margin_min is None else min(self._margin_min, margin)
        if bar is not None and bar > 0:
            if self._bar0 is None:
                self._bar0 = self._bar_lo = self._bar_hi = float(bar)
            else:
                self._bar_lo = min(self._bar_lo, float(bar))
                self._bar_hi = max(self._bar_hi, float(bar))

    def note_unseen(self, n: int) -> None:
        self.unseen = max(0, int(n))

    def reasons(self) -> "list[str]":
        """One sentence per condition that fired, each naming itself. Empty when
        the measurement was fine (then nothing is marked at all)."""
        out: "list[str]" = []
        if self.covered_s >= COVER_MIN_SECONDS:
            out.append(f"something covered the screen for {self.covered_s:.0f}s")
        if self._moved0 is not None and self._moved_now > self._moved0:
            out.append("the camera appears to have moved")
        if self._margin_min is not None and self._margin_min < self.margin_floor:
            out.append(f"the signal margin was thin ({self._margin_min:.1f}x, "
                       f"under {self.margin_floor:.1f}x)")
        if self._bar0 and self._bar_lo and self._bar_hi:
            ratio = self._bar_hi / self._bar_lo
            if ratio >= BAR_MOVE_RATIO:
                out.append(f"the detection bar moved from {self._bar_lo:.0f} to {self._bar_hi:.0f}")
        if self.unseen > 0:
            n = self.unseen
            out.append(f"the microphone heard {n} trigger pull{'' if n == 1 else 's'} "
                       f"the camera never saw")
        return out


def mark_text(reasons: "list[str]") -> str:
    """The footnote, or "" when nothing fired. One sentence: the reasons joined,
    so one condition reads as one clause naming it."""
    if not reasons:
        return ""
    return "* measurement was poor this round: " + "; ".join(reasons)


# The screen reserves a constant block for the mark whether or not it is present
# (the round screen is the same height at round 9 as at round 1). Two lines.
MARK_LINES = 2


def wrap_mark(text: str, width: int = 64) -> "list[str]":
    """`text` as exactly MARK_LINES lines (blank when empty or short), broken at
    a word boundary. Never more than MARK_LINES: a longer sentence is kept whole
    on the second line and the painter shrinks it to fit, so nothing is cut."""
    if not text:
        return [""] * MARK_LINES
    if len(text) <= width:
        return [text, ""]
    cut = text.rfind(" ", 0, width)
    if cut <= 0:
        cut = width
    return [text[:cut].rstrip(), text[cut:].lstrip()]
