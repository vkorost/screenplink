"""Two-player rounds: the arithmetic and the words, with no Qt and no mode.

PURE. Counts in, a judgement and its sentences out, so the whole of "who won"
is testable in microseconds and cannot drift between the painter and the rules.

A ROUND IS ONE MAGAZINE OF REGISTERED SHOTS PER SIDE (`settings.DEFAULT_MAGAZINE`,
saved as `AppSettings.magazine`, held on the host as `ModeHost.magazine`). A side
is done when the app has SEEN and scored that many; a side the app under-detects
simply pulls the trigger more often to reach its count. Both sides therefore
always reach exactly the same denominator, a round always closes by itself, and
no timer and no guess about shots the camera never saw enters the app. His
words (2026-10-19): "If the shots are missed they are missed randomly from both
sides. Just count the shots that had been registered."

THE SCORE IS JUST POINTS. With equal denominators there is nothing for a
fairness gate to protect: an earlier design refused to name a winner when the
margin was smaller than what the trailing side's unseen shots might have been
worth, which only made sense while the denominator was a count of shots fired.
It is gone, and so is the per-side "unseen" number it fed on.

THE MICROPHONE is published, not used to decide anything: a heard-but-unseen
count is a diagnostic about his setup ("the app missed 4 of your pulls") and the
only evidence of a trigger pull the camera never saw at all. It has no position,
so it cannot be given to a side.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

# THE ROUND'S LENGTH IS NOT HERE. It used to be `ROUND_SHOTS = 20`, his number
# of 2026-10-19; on 2026-10-20 he replaced it with the MAGAZINE: one number for
# the round everywhere, single player and each half of two players, owned by
# `lsa.settings` -- the number is his and lives only there -- so the bullets on the screen and the shot
# that ends the round can never disagree. Do not add a second copy here.

SIDE_NAMES = ("LEFT", "RIGHT")


@dataclass(frozen=True)
class SideResult:
    """One player's standing: points and the registered shots behind them."""

    points: int = 0
    shots: int = 0


@dataclass(frozen=True)
class Verdict:
    kind: str                 # "none" | "win" | "tie"
    winner: "int | None"      # index (0 or 1) of the argument that leads
    margin: int               # absolute points between them


def judge(a: SideResult, b: SideResult) -> Verdict:
    """Name a winner. Both sides must have registered the same number of shots:
    that equality is the whole fairness argument, so a mismatch is an error and
    not something to be smoothed over."""
    if a.shots != b.shots:
        raise ValueError(f"unequal registered shots {a.shots} vs {b.shots}")
    if a.shots == 0:
        return Verdict("none", None, 0)
    margin = abs(a.points - b.points)
    if margin == 0:
        return Verdict("tie", None, 0)
    return Verdict("win", 0 if a.points > b.points else 1, margin)


# -- the words ----------------------------------------------------------------

def _shots(n: int) -> str:
    return f"{n} shot" if n == 1 else f"{n} shots"


# The fixed words of the result line, in ONE place: `result_words` builds the line
# from them and `is_result_line` recognises it by them, so rewording a result cannot
# leave the round screen drawing it as an ordinary line (the same arrangement as
# `scorecard.NEW_BEST_PREFIX`).
_NO_SHOTS = "NO SHOTS SEEN"
_TIED = "TIED"
_MATCH = "MATCH: "
_WINS_BY = " WINS by "
_LEADS_BY = " leads by "


def is_result_line(text: str, names: "tuple[str, str]" = SIDE_NAMES) -> bool:
    """True for a line `result_words` / `verdict_sentence` produced: "RIGHT WINS by
    11 points", "MATCH: LEFT leads by 12 points", "TIED", "MATCH: TIED", "NO SHOTS
    SEEN". The round screen draws these bold end to end (his ruling, 2026-10-22:
    "'Right wins by 11 points' should be highlighted in bold"). The sentence BESIDE
    it ("LEFT is more efficient, 9.44 to 9.41") starts with a side name too but never
    with these words, so it is not one: it is the aside, and keeps numeric emphasis
    only."""
    t = str(text)
    if t in (_NO_SHOTS, _TIED, _MATCH + _TIED):
        return True
    return any(t.startswith(n + _WINS_BY) or t.startswith(_MATCH + n + _LEADS_BY) for n in names)


def verdict_sentence(v: Verdict, names: "tuple[str, str]" = SIDE_NAMES) -> str:
    """One line. Never the word 'fair'; only what was measured."""
    if v.kind == "none":
        return _NO_SHOTS
    if v.kind == "tie":
        return _TIED
    return f"{names[v.winner]}{_WINS_BY}{v.margin}"


def points_word(n: int) -> str:
    """"1 point" / "26 points": the unit is spelled out on the verdict screen."""
    return f"{n} point" if n == 1 else f"{n} points"


def points_of_word(points: int, ghost_points: "int | None") -> str:
    """"65 of 75 points": the points line of the single-player round screen with
    its ceiling, so "65 points" is no longer a number with nothing to be read
    against (his question, 2026-10-22: "65 points of how many possible points?").

    The 75 is what the ghost took on the SAME round (`GhostScore.ghost_points`):
    the very figure `AIM %` is already computed from, which the screen used to
    show only as a percentage. ONE definition, so the painter and the tests read
    the same words.

    With no usable ceiling (no card because the score failed, a round that could
    not be scored, or a ghost that took nothing) this is `points_word` unchanged,
    "65 points": a ceiling the app does not have is left off, never invented. The
    unit follows the HUMAN's number, as `points_word` does: "1 of 75 points"."""
    ok = (isinstance(ghost_points, numbers.Real) and not isinstance(ghost_points, bool)
          and math.isfinite(ghost_points) and ghost_points > 0)
    if not ok:
        return points_word(points)
    return f"{points} of {int(ghost_points)} points"


def result_words(v: Verdict, names: "tuple[str, str]" = SIDE_NAMES, match: bool = False) -> str:
    """The verdict screen's result line, with the unit spelled out. `match`
    words the running match ("MATCH: LEFT leads by 12 points") rather than a
    round. The side name stays in capitals."""
    if v.kind == "none":
        return _NO_SHOTS
    if v.kind == "tie":
        return _MATCH + _TIED if match else _TIED
    if match:
        return f"{_MATCH}{names[v.winner]}{_LEADS_BY}{points_word(v.margin)}"
    return f"{names[v.winner]}{_WINS_BY}{points_word(v.margin)}"


def done_text(name: str, shots: int, total: int) -> str:
    """What a finished side's own half says. Spelled out, because a target
    that silently stops responding is this project's most-reported complaint."""
    return f"{name}: done, {shots} of {total}"


def _pulls(n: int) -> str:
    return f"{n} trigger pull" if n == 1 else f"{n} trigger pulls"


def heard_text(ready: bool, n: int, match: "int | None" = None) -> str:
    """The microphone's tally, as the LOG states it (always written, whatever
    is on screen: an unreported check is a failure, not a pass by omission).
    It has no position, so it cannot be attributed to a side; it is the only
    check on a pull the camera missed ENTIRELY. `match` is the running total,
    given from the second round on. Never a bare zero when the microphone has
    not calibrated: that would read as "nothing was missed"."""
    if not ready:
        return "microphone not calibrated: no count available"
    tail = f"; {match} across the match" if match is not None else ""
    if n == 0 and not match:
        return "microphone: no trigger pulls heard that the camera did not register"
    return (f"microphone: {_pulls(n)} heard that the camera did not register this round"
            f"{tail} (not attributable to a side)")


def heard_on_screen(ready: bool, n: int, match: "int | None" = None) -> str:
    """What the verdict screen says about the microphone: a short line, or
    NOTHING. The channel is often not calibrated (two guns at about two shots a
    second keep the app's own gunshot muting it continuously, so it genuinely
    cannot report), and a line that says nothing useful is clutter on the screen
    the camera watches. It speaks only when it has a count to give; the log
    (`heard_text`) always speaks."""
    if not ready or (n <= 0 and not match):
        return ""
    tail = f", {match} this match" if match is not None else ""
    return f"Microphone: {_pulls(n)} heard that the camera did not register{tail}"


# -- dry pulls ------------------------------------------------------------------
#
# A DRY PULL is a trigger pull after the magazine is empty. His words
# (2026-10-20): "the app should log the dry pulls based on clicks and laser hits
# on the screen. We will later figure out how we account for that in the overall
# score."
#
# NOT SCORED. These numbers change no score, no hit rate, no points and no
# verdict, today. They are a training datum printed small on the end screen and
# written to the log. A figure on a results screen invites someone to start
# subtracting it from something; that is a decision he has deferred, so nothing
# here may be read by the scoring code.
#
# A UNION, NEVER A SUM. One pull normally produces BOTH a click (the microphone
# hearing the trigger break) and a laser detection (the camera). Adding the two
# channels would report every pull twice. `ShotCorroborator` already pairs what
# was heard with what was seen, so the count is
#
#     corroborated pairs + camera-only detections + microphone-only clicks
#
# and `camera` below is every dry detection (corroborated or not), so the total
# is camera + microphone-only and the corroborated figure is a subset of the
# camera one, never an addend.

def dry_total(camera: int, mic_only: int) -> int:
    """Every dry pull once: each camera detection, plus the clicks that had no
    detection to pair with. A corroborated pair is inside `camera`."""
    return int(camera) + int(mic_only)


def dry_text(camera: int, corroborated: int, mic_only: int, mic_ready: bool) -> str:
    """The end screen's one small line. ALWAYS a line, zero included, so the
    screen is the same height at every round. When the microphone has not
    calibrated, a pull the camera missed cannot be counted at all, so the
    number is a LOWER BOUND and says so rather than posing as a total."""
    n = dry_total(camera, mic_only)
    camera_only = max(0, camera - corroborated)
    if not mic_ready:
        return (f"Dry pulls: {n} or more (microphone not calibrated: pulls the "
                f"camera missed are not counted)")
    return (f"Dry pulls: {n} ({corroborated} seen and heard, {camera_only} seen only, "
            f"{mic_only} heard only)")


def dry_log_text(camera: int, corroborated: int, mic_only: int, mic_ready: bool,
                 deaf: int = 0) -> str:
    """The log's line: the same facts, the breakdown never abbreviated. `deaf` is
    how many of the camera-only pulls came while the microphone could not have
    heard (muted by the app's own last bang, or not delivering): those are not
    evidence of anything, whereas a camera-only pull with a LISTENING microphone
    is a detection nobody heard, which may be a very quiet pull or no pull."""
    n = dry_total(camera, mic_only)
    camera_only = max(0, camera - corroborated)
    bound = ("" if mic_ready else
             "; LOWER BOUND: the microphone has not calibrated, so heard-only "
             "pulls cannot be counted")
    return (f"dry pulls after the magazine was empty: {n} = {corroborated} corroborated "
            f"(seen and heard) + {camera_only} camera-only (of which {deaf} while the "
            f"microphone was deaf) + {mic_only} microphone-only"
            f"{bound}; not scored")


# -- the single-player result block -----------------------------------------------

def _n(n: int, one: str, many: str) -> str:
    return f"{n} {one}" if n == 1 else f"{n} {many}"


def shots_word(n: int) -> str:
    return _n(n, "shot", "shots")


def hits_word(n: int) -> str:
    return _n(n, "hit", "hits")


def misses_word(n: int) -> str:
    return _n(n, "miss", "misses")


def seconds_word(s: float) -> str:
    """"24.6 seconds": one decimal, always plural (1.0 seconds), so the line is
    the same shape at every round."""
    return f"{max(0.0, s):.1f} seconds"
