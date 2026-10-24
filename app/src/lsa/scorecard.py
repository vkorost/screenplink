"""The ghost score for one finished round: what it is, whether it may be kept,
and the words the round screen prints. PURE (no Qt, no file, no clock read in the
arithmetic), like `ghost.py` and `bests.py` which it joins.

THE ONE QUESTION THIS MODULE ANSWERS FOR A ROUND

    "Is this number a measurement, an estimate, or an at-least figure -- and may
     it become the machine's best?"

and it answers it in ONE place, so the screen, the log and the settings file
cannot disagree about it.

A ROUND IS SOUND OR IT IS DISCARDED (his ruling, 2026-10-22). After a round that
showed `SCORE 256` beside `BEST SCORE SO FAR 212` and recorded nothing, he said:
"if the shot is recorded as 'hit' it goes to the score; if the shot is not
recorded, then it is not going into the score" and, of the phantom storm the old
refusal existed for, "in any other case when you detect that something funky is
going on, you should disregard the whole score". So:

  * SOUND      the score is shown, and if it beats the stored best it IS the new
               best. Nothing is decided afterwards: what the screen says is what
               the record book gets.
  * DISCARDED  no score, no aim, no speed and no efficiency are shown, because
               none exists (`GhostScore.discarded`). There is nothing to record and
               so nothing to refuse. The screen says plainly that the round was
               discarded and why (`discarded_lines`); the tally he can check for
               himself (shots, hits, points, seconds) stays. The log keeps
               everything, including what the round would have scored.

NEVER DISPLAY A FIGURE THE APP WILL NOT HONOUR. A caveat under a number the app
has already decided to ignore is the app disagreeing with itself in public.

A LOWER-BOUND ROUND IS RECORDED TOO (his ruling, 2026-10-22: "if the app shows the
score on screen, then this is the score; then the app has to honor that score and
record it as a record"). It used to be the one state that was shown and refused;
that contradiction is gone, so the rule has no exception left: a round that has a
score to show is a round that may set a record, and `RoundCard.eligible` (which
existed only to carry that exception) is deleted -- it had become exactly
`gs.scorable`, a field that decides nothing. The record is stored MARKED as a floor
(`basis` "lower_bound", see `bests.py`), never as a measured score.

  * measured     every wait the ghost was charged came from the round itself.
                 Shown plainly: `SCORE  416`.
  * declared     RETIRED 2026-10-23: the app no longer produces it (the ghost is
                 charged only the waits the round contained, and a target the
                 round never drew costs it nothing -- charging the drill's average
                 for it let a human who MISSED outscore the ghost, a stored 1146 on
                 a scale of 1000). The state, its basis and the records already
                 stored under it remain readable. What it was: part of the ghost's
                 timeline came from the drill's own average
                 wait (the human missed, so no target was ever drawn for those
                 shots). An ESTIMATE that can be off either way. Shown, since
                 2026-10-22, exactly like a measured one (`SCORE  416`); the
                 distinction lives in `card.basis`, in the stored record and in
                 the `ESTIMATE:` sentence `describe()` logs. It MAY set a record,
                 because six drills are in this state whenever the shooter misses
                 once and refusing would lock them out; the state is stored with it.
  * lower_bound  the drill makes the shooter wait and the waits were not
                 supplied, so the ghost is faster than a real one and the true
                 score is AT LEAST the figure. Shown, since 2026-10-22, as
                 plainly as the others (`SCORE  47`); the distinction lives in
                 `card.basis`, in `gs.lower_bound` / `gs.lower_bound_why` and in
                 the `AT LEAST:` sentence `describe()` logs.
                 It MAY set a record (2026-10-22), stored as a floor. That is the
                 conservative direction: the figure is lower than or equal to what
                 the round was truly worth (the ghost with the waits supplied can
                 only be SLOWER, never faster, so its speed share and the score
                 can only be higher), so a stored floor can understate a record
                 and can never inflate one. A later round beating it replaces it.
                 It cannot become unbeatable: a human did at least that well.

AN ESTIMATE, AND A FLOOR, NO LONGER WEAR A LABEL ON THE SCREEN, AND THAT IS A
DECISION, NOT AN OVERSIGHT. This module used to hold that "an estimate must not wear a measurement's
clothes", and drew `ABOUT` before a declared score, its speed and its stored best.
His ruling (2026-10-22): "I don't like the 'about' mentions. Please remove 'about'
from all the labels." His reason is measurable: the qualifier fires on any round
with a miss (a missed shot draws no target, so its wait is the drill's average),
which is most rounds, so on the screen it read as clutter rather than as
information. What was removed is what is DRAWN. What is KNOWN is untouched:
`basis_of` still says declared, `score_record` / `efficiency_record` still store
it, `bests.py` still compares by it, the record guard is unchanged, and
`describe()` still logs the basis and the full `ESTIMATE: ...` sentence. So the
statement above is now false of the screen and true of the log. The failure the
old rule was built against is recorded in CLAUDE.md: an app announcing "100% of his
shots clear the bar" beside a measurement that contradicted it.

`AT LEAST` followed the same day (his ruling, 2026-10-22: same treatment as ABOUT).
It was the one qualifier left on the screen, `SCORE  AT LEAST 47` and `SPEED AT
LEAST 21%`, on a round that makes the shooter wait with the waits not supplied. It
is no longer drawn. A lower-bound round is still `BASIS_LOWER`, still carries
`gs.lower_bound` and its reason, and `describe()` still logs `AT LEAST: <why>`.
The cost is stated: a floor now reads as an exact figure on the screen. Since the
same day it also sets a record (see above), so the stored `basis` is what tells a
floor from a measurement.

THE VALIDITY VERDICT IS `ghost.record_fitness`, AND IT IS NOT RE-DERIVED HERE. It
is applied once, inside `ghost.ghost_score`, which returns a discarded round with
no score at all; this module never asks "was it fit?" a second time. The score
best and the efficiency best therefore go through the SAME decision, and a card
that has a score to show is by construction a card that may be recorded (no
exception: a lower-bound one is recorded as a floor). Only a round's SPAN can discard it;
brightness and the microphone are logged as cautions (`describe()`).

THE SCORE BLOCK IS FOUR LINES (`lines()`), sound or discarded, so the round screen
is the same height at round 9 as at round 1. A sound round prints the score, aim
and speed, the best score, the best efficiency; a record is announced as `NEW BEST
SCORE RECORDED` in place of the `BEST SCORE SO FAR` line. A discarded round prints
`discarded_lines`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

from . import bests as _b
from . import ghost as _g

# The ghost is 1000 by construction (`ghost.SCORE_SCALE`). The screen no longer
# prints it (2026-10-22); the log line from `describe()` still does.
GHOST_SCORE = int(_g.SCORE_SCALE)

BASIS_LOWER = _b.BASIS_LOWER           # ONE definition: the book must accept what the card stores
BASIS_NONE = "none"

# Every line the score block prints, always, so the block is the same height at
# round 9 as at round 1 and whatever the round was (the round screen's own rule).
SCORE_LINES = 4

# How both record announcements begin ("NEW BEST SCORE RECORDED", "NEW BEST
# EFFICIENCY RECORDED"). They carry no number, so the round screen emphasises the
# WHOLE line, and it finds them by this prefix (`modes.runs.emphasise`). ONE
# definition: `best_line` and `efficiency_best_line` build their text from it, so
# rewording either cannot leave the screen treating a record like an ordinary line.
NEW_BEST_PREFIX = "NEW BEST "

# THE SHORT NOTE ON A CARD (`RoundCard.note`) is a LOG field and is not drawn
# anywhere (his ruling, 2026-10-22). Its size bound survives because the note is
# still built, still tested, and an unbounded string in a card is how the screen
# once ended up with a caveat at the 9 px floor: `MAX_NOTE_CHARS` is the ceiling
# on the WORST note this module can build, asserted against every state by
# test_ghost, so a longer phrase fails the build. 33 is the longest today (the
# no-maximum caption); keep it equal to that. It was 35 while the lower-bound
# caption read "..., not a best", which stopped being true on 2026-10-22.
#
# The refusal note this budget was first measured on ("NOT A BEST: ...", with
# `_SHORT`, `MAX_CAUSES_SHOWN`, `NOT_SAVED` and a fallback) is GONE with the refusal
# it described: a round is now scored or discarded, never scored-and-refused.
MAX_NOTE_CHARS = 33

# The one phrase a discarded round's note gives per reason, and the fallback for a
# reason added later that has no phrase yet.
_DISCARD_SHORT = {
    _g.Unfit.TOO_FAST: "faster than a pistol",
}
DISCARD_NOTE = "DISCARDED: "
DISCARD_FALLBACK = "not a human round"


@dataclass(frozen=True)
class RoundCard:
    """One side's finished round, scored."""

    key: str                              # where its bests live (drill|magazine|players)
    gs: _g.GhostScore
    efficiency: Optional[float]           # points per printed second; None: no time to divide by,
                                          # or the round was DISCARDED (then see withheld_efficiency)
    basis: str                            # measured | declared | lower_bound | none
    would_beat: bool                     # beat the stored best on either measure, discarded or not (LOG)
    new_score: bool = False
    new_efficiency: bool = False
    best_score: Optional[dict] = None     # what is stored for this key, once settled
    best_efficiency: Optional[dict] = None
    note: str = ""                        # the one honest caveat, or "" (log only; not a screen line)
    withheld_efficiency: Optional[float] = None   # LOG ONLY: a discarded round's efficiency

    @property
    def discarded(self) -> bool:
        """The round could not have been a human's. It has no score and no
        efficiency to show or to record; the screen says so (`discarded_lines`)."""
        return self.gs.discarded


def basis_of(gs: _g.GhostScore) -> str:
    if not gs.scorable:
        return BASIS_NONE
    if gs.lower_bound:
        return BASIS_LOWER
    return _b.BASIS_DECLARED if gs.declared_waits > 0 else _b.BASIS_MEASURED


def shown(value: float) -> int:
    """The whole number printed for a score: the same rounding `ghost_score`
    uses for its own `shown`, so a stored best and a fresh score agree."""
    return int(math.floor(float(value) + 0.5))


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _note(gs: _g.GhostScore, *, shots: int, ceiling_known: bool) -> str:
    if gs.discarded:
        why = next((_DISCARD_SHORT[r] for r in gs.record.reasons if r in _DISCARD_SHORT),
                   DISCARD_FALLBACK)
        return DISCARD_NOTE + why
    if not gs.scorable:
        if shots < 2:
            return "NO SCORE: a round needs two shots"
        if not ceiling_known:
            return "NO SCORE: drill states no maximum"
        return "NO SCORE: see the log"
    if gs.lower_bound:
        return "AT LEAST: waits unknown, a floor"
    if gs.declared_waits > 0:
        n = gs.declared_waits
        return f"ESTIMATE: {n} {_plural(n, 'wait', 'waits')} assumed"
    return ""


def make_card(*, key: str, human_points: int, human_elapsed: float, shots: int,
              efficiency: Optional[float], ceiling: Optional[_g.Ceiling],
              has_waits: bool = False,
              available_at: Optional[Sequence[float]] = None,
              declared_waits: int = 0,
              peaks: Sequence[Optional[float]] = (),
              corroboration: Optional[_g.Corroboration] = None,
              book: Optional[_b.BestBook] = None) -> RoundCard:
    """Score one finished round and decide whether it may become a record.

    NOTHING IS WRITTEN HERE. The book is only read, so two sides of a versus round
    are judged against the same prior; `settle` writes.
    """
    gs = _g.ghost_score(human_points=human_points, human_elapsed=human_elapsed,
                        shots=shots, ceiling=ceiling, has_waits=has_waits,
                        available_at=available_at, declared_waits=declared_waits,
                        peaks=peaks, corroboration=corroboration)
    basis = basis_of(gs)
    # ONE rule for both bests, and it is "has a score" (`gs.scorable`): a round
    # `ghost_score` discarded has none, so it can set nothing and there is no second
    # opinion to apply here. A lower bound has a score, so it sets a record (his
    # ruling, 2026-10-22); `basis` is what marks it a floor in the book. There used
    # to be an `eligible` field carrying the lower-bound exception; with the
    # exception gone it equalled `gs.scorable`, so it was deleted rather than kept
    # as a second name for the same fact.

    # A discarded round shows and records NO efficiency either: the same round, the
    # same verdict. The figure survives for the log only.
    shown_eff = None if gs.discarded else efficiency
    prior_s = book.best_score(key) if book is not None else None
    prior_e = book.best_efficiency(key) if book is not None else None
    # What the round would have scored, for `would_beat` (a LOG fact, and the whole
    # point of logging a discard: "this would have replaced your best of 416").
    raw_score = gs.withheld_score if gs.discarded else gs.score
    beats_s = (raw_score is not None and raw_score > 0.0
               and (prior_s is None or raw_score > prior_s["value"]))
    beats_e = (efficiency is not None and efficiency > 0.0
               and (prior_e is None or efficiency > prior_e["value"]))
    would_beat = bool((gs.scorable or gs.discarded) and (beats_s or beats_e))
    return RoundCard(
        key=key, gs=gs, efficiency=shown_eff, basis=basis,
        would_beat=would_beat,
        new_score=bool(gs.scorable and book is not None and beats_s),
        new_efficiency=bool(gs.scorable and book is not None and beats_e),
        best_score=prior_s, best_efficiency=prior_e,
        note=_note(gs, shots=int(shots), ceiling_known=ceiling is not None),
        withheld_efficiency=efficiency if gs.discarded else None)


def score_record(card: RoundCard, when: str = "") -> dict:
    gs = card.gs
    return {"value": float(gs.score), "points": int(gs.human_points),
            "seconds": round(float(gs.human_elapsed), 4), "shots": int(gs.human_shots),
            "basis": card.basis, "declared_waits": int(gs.declared_waits),
            "ghost_points": gs.ghost_points,
            "ghost_seconds": None if gs.ghost_elapsed is None else round(float(gs.ghost_elapsed), 4),
            "ghost_interval": _g.GHOST_INTERVAL, "when": when}


def efficiency_record(card: RoundCard, when: str = "") -> dict:
    gs = card.gs
    return {"value": float(card.efficiency), "points": int(gs.human_points),
            "seconds": round(float(gs.human_elapsed), 4), "shots": int(gs.human_shots),
            "basis": card.basis, "when": when}


def _same(rec: Optional[dict], value: Optional[float]) -> bool:
    return rec is not None and value is not None and abs(rec["value"] - float(value)) < 1e-9


def settle(cards: "Sequence[Optional[RoundCard]]", book: Optional[_b.BestBook],
           when: str = "") -> "list[Optional[RoundCard]]":
    """Write the records the cards earned, then return the cards as they should
    be SHOWN: with the best now stored, and a "new" flag only for a side that
    still holds it (in a versus round the other half may have beaten it in the
    same breath).

    The book itself refuses anything that is not strictly better, so the order of
    the cards cannot lose a better record. `book.take_changed()` is the caller's
    "write the file" signal: True at most once per round, and only if a record
    was actually written.
    """
    if book is None:
        return list(cards)
    for c in cards:
        if c is None:
            continue
        s = score_record(c, when) if c.new_score else None
        e = efficiency_record(c, when) if c.new_efficiency else None
        if s is not None or e is not None:
            book.offer(c.key, s, e)
    out: "list[Optional[RoundCard]]" = []
    for c in cards:
        if c is None:
            out.append(None)
            continue
        bs, be = book.best_score(c.key), book.best_efficiency(c.key)
        out.append(replace(
            c, best_score=bs, best_efficiency=be,
            new_score=c.new_score and _same(bs, c.gs.score),
            new_efficiency=c.new_efficiency and _same(be, c.efficiency)))
    return out


# ---------------------------------------------------------------------------
# The words
# ---------------------------------------------------------------------------

def headline(card: Optional[RoundCard]) -> str:
    """`SCORE  416`, or `SCORE  --`. The ` / 1000` it used to carry is gone at his
    request (2026-10-22), and so are `ABOUT` for a declared round and `AT LEAST`
    for a lower-bound one (same day, see the module docstring): every scorable
    round prints exactly as a measured one, and `card.basis` / `describe()` still
    say which it was. A lower-bound round sets a record, stored marked as a floor."""
    if card is None or not card.gs.scorable or card.gs.shown is None:
        return "SCORE  --"
    return f"SCORE  {card.gs.shown}"


def factors_line(card: Optional[RoundCard]) -> str:
    """The two factors, the diagnostic: `AIM 82%   SPEED 21%`. Aim is how much of
    the available points he took (exact: the ghost's points are known). Speed is
    the factor the ghost's waits move, and used to carry `AT LEAST` for a
    lower-bound round; neither that nor `ABOUT` is drawn any more (2026-10-22,
    his ruling). The basis is still `card.basis` and still logged."""
    if card is None or not card.gs.scorable:
        return "AIM --   SPEED --"
    aim = f"{card.gs.points_share * 100:.0f}%"
    spd = f"{card.gs.speed_share * 100:.0f}%"
    return f"AIM {aim}   SPEED {spd}"


def best_line(card: Optional[RoundCard]) -> str:
    """`BEST SCORE SO FAR  416` (or `--` with no record), and when this round set
    the record the line becomes `NEW BEST SCORE RECORDED` INSTEAD of it, not
    beside it. NEW only for a card that has been settled (a best exists to be
    new): an unsettled card must not announce a record that was never written."""
    rec = None if card is None else card.best_score
    if card is not None and card.new_score and rec is not None:
        return NEW_BEST_PREFIX + "SCORE RECORDED"
    # The stored record's basis (declared or not) is deliberately NOT printed
    # (2026-10-22); it stays in the record for `bests.py`.
    word = "--" if rec is None else f"{shown(rec['value'])}"
    return f"BEST SCORE SO FAR  {word}"


def efficiency_best_line(card: Optional[RoundCard]) -> str:
    """The same two forms for the efficiency: `BEST EFFICIENCY SO FAR  3.55`, or
    `NEW BEST EFFICIENCY RECORDED` in its place."""
    rec = None if card is None else card.best_efficiency
    if card is not None and card.new_efficiency and rec is not None:
        return NEW_BEST_PREFIX + "EFFICIENCY RECORDED"
    word = "--" if rec is None else f"{rec['value']:.2f}"
    return f"BEST EFFICIENCY SO FAR  {word}"


DISCARDED_HEADLINE = "ROUND DISCARDED"


def discarded_lines(card: RoundCard) -> "list[str]":
    """What the score block says for a discarded round: EXACTLY `SCORE_LINES` (four)
    lines, so the screen keeps its shape. NO SCORE, NO AIM, NO SPEED, no best and no
    efficiency appear -- none of them exists, and the app does not show a figure it
    will not honour.

    The reason is stated in HIS terms, with the two numbers he can check for
    himself on the same screen (the shots and the seconds), not in the detector's:
    `17 shots in 2.7 s is faster than a pistol can fire.` rather than `span under
    0.91 of the cadence floor`. Each line is no longer than the longest line of a sound
    block (36 characters), so the type is not made smaller to carry an
    explanation. The full reasoning is in the log (`describe()`).

        ROUND DISCARDED
        17 shots in 2.7 s is faster
        than a pistol can fire.
        No score, nothing saved.
    """
    gs = card.gs
    if _g.Unfit.TOO_FAST in gs.record.reasons:
        why = [f"{gs.human_shots} shots in {gs.human_elapsed:.1f} s is faster",
               "than a pistol can fire."]
    else:
        # A reason added later with no sentence of its own yet. Still true, still
        # plain, and still discards: it just does not say which test fired.
        why = ["Something about these shots", "does not look like a person."]
    return [DISCARDED_HEADLINE, why[0], why[1], "No score, nothing saved."]


def lines(card: Optional[RoundCard]) -> "list[str]":
    """The score block, EXACTLY `SCORE_LINES` (four) lines whatever the round was.

    A sound round: the headline, aim and speed, the best score, the best
    efficiency. A discarded one: `discarded_lines`, and nothing of the above.

    THE CAVEAT (`card.note`) IS NOT HERE -- his ruling, 2026-10-22: "nobody cares".
    It is built on the card and reaches the log through `describe()`."""
    if card is not None and card.discarded:
        return discarded_lines(card)
    return [headline(card), factors_line(card), best_line(card),
            efficiency_best_line(card)]


def clamp_defect_text(gs: _g.GhostScore) -> str:
    """The sentence for a SCORABLE round on which the 1000-ceiling clamp fired, with
    the numbers behind it, naming the three things that can be wrong. One definition:
    `describe()` carries it and the host logs it as a `DEFECT` line, so the two
    cannot disagree. "" when nothing was clamped."""
    if not gs.clamped or gs.speed_raw is None:
        return ""
    raw_score = (None if gs.points_share is None
                 else GHOST_SCORE * gs.points_share * gs.speed_raw)
    return (f"CLAMPED {'+'.join(gs.clamped)}: raw speed share {gs.speed_raw:.4f} is above 1.0 "
            f"(ghost {gs.ghost_elapsed:.3f}s over {gs.human_shots} shots against human "
            f"{gs.human_elapsed:.3f}s; cadence {_g.GHOST_INTERVAL}s) and was held at 1.000"
            + ("" if raw_score is None else f", the raw score was {raw_score:.1f}")
            + ". Our model is wrong somewhere: GHOST_INTERVAL is pessimistic, or a wait the "
              "ghost was charged is not one this round contained, or the drill's ceiling is "
              "understated")


def describe(card: Optional[RoundCard], name: str = "") -> str:
    """One log line carrying everything the screen compresses: both factors, the
    basis and why, and what the guard decided and why."""
    if card is None:
        return f"score{name}: none (the round could not be scored)"
    gs = card.gs
    if gs.discarded:
        # Everything the screen no longer says, in one line: the measurements, the
        # test that discarded the round and why, what it would have scored and
        # whether that would have replaced a stored best (the case the whole design
        # exists for), and the cautions that did NOT discard it.
        bits = ["ROUND DISCARDED, no score and no efficiency shown, nothing recorded: "
                + gs.record.text,
                f"human {gs.human_points} pts in {gs.human_elapsed:.2f}s over "
                f"{gs.human_shots} shots"]
        if gs.withheld_score is not None:
            bits.append(f"it would have scored {gs.withheld_score:.1f} / {GHOST_SCORE}")
        if gs.clamped and gs.speed_raw is not None:
            # Expected on a discard (a round faster than the ghost is what TOO_FAST
            # is), so this is information and not a DEFECT: the withheld figure above
            # is the clamped one, and this is what it was before the clamp.
            bits.append(f"(that figure is capped: raw speed {gs.speed_raw:.3f} was clamped to 1.000)")
        if card.withheld_efficiency is not None:
            bits.append(f"efficiency would have been {card.withheld_efficiency:.2f}")
        if card.would_beat:
            bits.append("it WOULD HAVE BEATEN the stored best")
        if gs.record.caution_text:
            bits.append("noted: " + gs.record.caution_text)
        return f"score{name}: " + "; ".join(bits)
    if not gs.scorable:
        return f"score{name}: NOT SCORABLE - {gs.not_scorable}"
    bits = [f"{gs.score:.1f} / {GHOST_SCORE} ({card.basis})",
            f"aim {gs.points_share:.3f} x speed {gs.speed_share:.3f}",
            f"human {gs.human_points} pts in {gs.human_elapsed:.2f}s over {gs.human_shots} shots",
            f"ghost {gs.ghost_points} pts in {gs.ghost_elapsed:.2f}s"]
    if gs.clamped:
        # The screen and the book carry the clamped figure; THIS line keeps the raw one,
        # because it is the evidence that the model is off (see `clamp_defect_text`).
        bits.append(clamp_defect_text(gs))
    if gs.lower_bound_why:
        bits.append("AT LEAST: " + gs.lower_bound_why)
    if gs.declared_why:
        bits.append("ESTIMATE: " + gs.declared_why)
    if card.efficiency is not None:
        bits.append(f"efficiency {card.efficiency:.2f}")
    # Every card that gets here has a score, and a score is a record candidate: there
    # is no scored-and-refused state any more (a lower bound included, 2026-10-22).
    # A floor says so, because the book will hold it as one.
    floor = " (stored as a floor: the true score was at least this)" if gs.lower_bound else ""
    bits.append(("record: new score best" + floor) if card.new_score
                else "record: score not beaten")
    # The efficiency is exact whatever the basis (points over printed seconds, no
    # ghost in it), so only the score carries the floor remark.
    bits.append("new efficiency best" if card.new_efficiency else "efficiency not beaten")
    if gs.record.caution_text:
        bits.append("noted, and did not discard the round: " + gs.record.caution_text)
    return f"score{name}: " + "; ".join(bits)
