"""The ghost score, against his real rounds and the failures it exists to stop.

Every number here is a measurement from his logs of 2026-10-20 with the
reasoning beside it, or arithmetic done on the constants in `lsa/ghost.py` --
never a value typed to make a check pass. The module is pure, so this suite
needs no Qt, no camera and no screen, and runs in well under a second.

What must be able to fail, and what turns it red:

  * the exact factorisation  -> section 4 (score built any other way)
  * his best round reading 416 -> section 4 (GHOST_INTERVAL or the clock changed)
  * the ghost's escalating 75 -> section 3 (a hand-written table, or a wrong cap)
  * a round is SOUND or DISCARDED, never shown-and-refused (his ruling, 2026-10-22)
    -> sections 9 and 13: ONLY the span discards; brightness and the microphone are
    cautions that cannot discard a round alone or together, and the FIXTURE PAIR
    (the 2026-10-21 storm, discarded; his 2026-10-22 round of 256, scored) turns
    red by name if either answer flips
  * the lower-bound flag -> section 6
  * the laser/noise boundary duplicate -> section 10 reads it out of the source
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import ast
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa import ghost                                             # noqa: E402
from lsa.ghost import (GHOST_INTERVAL, GHOST_SEE_FRAMES, MIN_CORROBORATED,   # noqa: E402
                       MIN_MEDIAN_PEAK, MIN_SPAN_RATIO, SCORE_SCALE, Corroboration,
                       Unfit, available_from_waits, constant_ceiling,
                       escalating_ceiling, frame_seconds, ghost_elapsed,
                       ghost_points, ghost_round, ghost_score, record_fitness)

checks: list[bool] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    checks.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))


def close(a, b, tol=1e-9) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def fx(v, spec: str = "") -> str:
    """`v` formatted for a check's DETAIL string, "None" when it is None.

    A detail is evaluated EAGERLY, as an argument, before `check` runs: so
    `f"{r.score:.2f}"` on a round a mutation has discarded raises TypeError and
    kills the suite, and the mutation runner then counts the fault BROKEN rather
    than caught (2026-10-22: mutate_ghost faults 4, 8 and 25). Every Optional field
    of a GhostScore / RoundCard (score, shown, shares, withheld_*, efficiency) is
    read through here or behind an `is not None` in the condition."""
    return "None" if v is None else format(v, spec)


def ratio(a, b):
    """a / b, or None when either is missing or b is zero (never raises)."""
    return None if a is None or b is None or b == 0 else a / b


def rv(rec, field: str = "value"):
    """`rec[field]` for a stored-record dict, None for a missing record or field.
    A record the book failed to keep must read as a failed CHECK, not a KeyError."""
    return rec.get(field) if isinstance(rec, dict) else None


def rel_close(a, b, rel=1e-14) -> bool:
    """Equal to within `rel` of the larger magnitude.

    1e-14 is not a convenience. Two correct ways of writing the same ratio differ
    in how they associate four IEEE operations (two divisions, a multiply, and a
    division or another multiply), and each rounds by at most 2**-53 = 1.1e-16
    relative, so they can disagree by about 9e-16 in the worst case. 1e-14 is 10x
    that bound and 100x tighter than the 1e-12 / 1e-9 this suite used before.
    Where two sides run the SAME operations in the SAME order, use `==`.
    """
    return (a is not None and b is not None
            and abs(a - b) <= rel * max(abs(a), abs(b)))


# A round as a healthy one looks: bright shots, a microphone that heard them.
HEALTHY_PEAKS = [85.0] * 17
HEALTHY_MIC = Corroboration(mic_ready=True, heard_and_seen=16, deaf=0)


def S(**over):
    """ghost_score for a round with every field healthy unless overridden."""
    kw = dict(human_points=165, human_elapsed=9.34, shots=17,
              ceiling=constant_ceiling(10), peaks=HEALTHY_PEAKS,
              corroboration=HEALTHY_MIC)
    kw.update(over)
    return ghost_score(**kw)


def S_safe(**over):
    """(result, "") or (None, "ExcType: message").

    A degenerate round that RAISES must show up as a FAILED CHECK, never as a
    crashed suite: the mutation runner cannot tell a crash from a broken harness
    and counts it BROKEN, not caught (CLAUDE.md 2026-10-19).
    """
    try:
        return S(**over), ""
    except Exception as exc:                       # noqa: BLE001 - any raise is the finding
        return None, f"{type(exc).__name__}: {exc}"


def at_interval(interval, **over):
    """`S(**over)` with the module's GHOST_INTERVAL rebound for that one call.

    GHOST_INTERVAL stays a FROZEN LITERAL in ghost.py (section 10 pins it), so
    there is no parameter to pass. The module reads the name from its own globals
    at call time, which makes rebinding the attribute drive the REAL code path.
    Restored in `finally`, always: a test that leaves the constant moved would
    change every check after it.
    """
    saved = ghost.GHOST_INTERVAL
    ghost.GHOST_INTERVAL = interval
    try:
        return S(**over)
    finally:
        ghost.GHOST_INTERVAL = saved


def by_definition(points, elapsed, shots, ceiling):
    """THE DEFINITION, from raw inputs: 1000 x (human points per second) /
    (ghost points per second). Shares nothing with the module's factors: the
    ghost's points are summed here, and the 0.25 s is typed here as the literal
    the definition names, not read from the module."""
    ghost_pts = sum(ceiling(k) for k in range(1, shots + 1))
    ghost_secs = (shots - 1) * 0.25
    return 1000 * (points / elapsed) / (ghost_pts / ghost_secs)


def source_constant(path: Path, name: str):
    """The integer assigned to `name` in a source file, or None."""
    m = re.search(rf"^\s*{name}\s*=\s*(\d+)\b",
                  path.read_text(encoding="utf-8", errors="replace"), re.M)
    return int(m.group(1)) if m else None


def main() -> int:
    print("=== the ghost score ===")

    # ---------------------------------------------------------------- 1 ------
    print("\n1. the frozen constant, and what is derived from it")
    check("GHOST_INTERVAL is 0.25 s (his logs: shortest accepted interval 0.254 s)",
          GHOST_INTERVAL == 0.25)
    check("the score scale is 1000: the ghost is 1000 by construction",
          SCORE_SCALE == 1000.0)
    g17 = ghost_round(17, constant_ceiling(10))
    check("a 17-shot ghost takes (17 - 1) * 0.25 = 4.0 s",
          g17 is not None and close(g17.elapsed, 4.0) and close(g17.cadence_elapsed, 4.0),
          f"{g17.elapsed if g17 else None}")
    check("the discard floor for 17 shots is 3.64 s, DERIVED (0.91 x the 4.0 s cadence span) "
          "-- BELOW the ghost's own 4.0 s, so the ghost is reachable",
          close(MIN_SPAN_RATIO * (17 - 1) * GHOST_INTERVAL, 3.64)
          and g17 is not None and MIN_SPAN_RATIO * 16 * GHOST_INTERVAL < g17.elapsed,
          f"{MIN_SPAN_RATIO * 16 * GHOST_INTERVAL}")
    # The linearity claim is checked THROUGH THE MODULE. The first version of this
    # check was arithmetic done inside the test, never calling `ghost`, so it
    # passed whatever the module did -- a check that cannot fail (RULES C1).
    best = dict(human_points=165, human_elapsed=9.34)
    r25, r30, r35 = S(**best), at_interval(0.30, **best), at_interval(0.35, **best)
    check("the interval is rebound for one call and RESTORED after it (the "
          "frozen constant is untouched, and his best still reads 416)",
          ghost.GHOST_INTERVAL == 0.25 and S(**best).shown == 416,
          f"{ghost.GHOST_INTERVAL}")
    check("at 0.30 s the real code gives his best round 499, the ghost 4.8 s",
          r30.shown == 499 and close(r30.ghost_elapsed, 4.8),
          f"{fx(r30.score, '.2f')}, {r30.ghost_elapsed}")
    check("at 0.35 s the real code gives his best round 582, the ghost 5.6 s",
          r35.shown == 582 and close(r35.ghost_elapsed, 5.6),
          f"{fx(r35.score, '.2f')}, {r35.ghost_elapsed}")
    check("so the score is directly proportional to the interval -- why changing "
          "it voids every stored best",
          rel_close(ratio(r30.score, r25.score), 0.30 / 0.25, 1e-12)
          and rel_close(ratio(r35.score, r25.score), 0.35 / 0.25, 1e-12),
          f"{fx(ratio(r30.score, r25.score), '.6f')} / {fx(ratio(r35.score, r25.score), '.6f')}")
    check("and the span floor follows it: 4.0 s for 17 shots is SOUND at 0.25 s "
          "(floor 3.64) and DISCARDED as too fast at 0.30 s (floor 4.368), derived not typed",
          S(human_elapsed=4.0, human_points=100).record.fit
          and not S(human_elapsed=4.0, human_points=100).discarded
          and Unfit.TOO_FAST in
          at_interval(0.30, human_elapsed=4.0, human_points=100).record.reasons
          and at_interval(0.30, human_elapsed=4.0, human_points=100).discarded)

    # ---------------------------------------------------------------- 2 ------
    print("\n2. the module is pure, and says so by construction")
    src_path = ROOT / "src" / "lsa" / "ghost.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    check("imports nothing but the standard library",
          imported <= {"__future__", "math", "numbers", "statistics",
                       "dataclasses", "enum", "typing"}, f"{sorted(imported)}")
    check("so no Qt, no camera, no audio device, no clock, no tracker",
          not (imported & {"PySide6", "cv2", "sounddevice", "time", "lsa", "numpy"}))
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check("no getattr / open / print anywhere in it (RULES C8: declared fields only)",
          not (calls & {"getattr", "open", "print", "setattr"}), f"{sorted(calls)}")

    # ---------------------------------------------------------------- 3 ------
    print("\n3. what a mode can say about itself")
    check("a constant ceiling: 17 perfect shots at 10 = 170 (Static Targets)",
          ghost_points(17, constant_ceiling(10)) == 170)
    check("Clear The Board is 1 a target: 17 shots = 17",
          ghost_points(17, constant_ceiling(1)) == 17)
    esc = escalating_ceiling(5)
    check("the escalating ceiling is 1,2,3,4,5,5,5 ... -- NOT constant in a round",
          [esc(k) for k in range(1, 8)] == [1, 2, 3, 4, 5, 5, 5])
    check("the escalating ghost's 17-shot total is 75 (1+2+3+4 then 5 x 13)",
          ghost_points(17, esc) == 75, f"{ghost_points(17, esc)}")
    check("a magazine of 3 totals 1+2+3 = 6 (the cap is not applied early)",
          ghost_points(3, esc) == 6)
    check("a cap of 2 totals 1+2+2+2 = 7 over 4 shots (the cap is the mode's own)",
          ghost_points(4, escalating_ceiling(2)) == 7)

    # The caps come from the modes' own source, not from a number typed here.
    ball_cap = source_constant(ROOT / "src" / "lsa" / "modes" / "ball.py", "MAX_LEVEL")
    balls_cap = source_constant(ROOT / "src" / "lsa" / "modes" / "balls.py", "MAX_LEVEL")
    check("both ball drills' MAX_LEVEL were found in their source",
          ball_cap is not None and balls_cap is not None, f"{ball_cap}, {balls_cap}")
    check("and with THEIR cap the 17-shot ghost totals 75",
          ball_cap is not None and balls_cap is not None
          and ghost_points(17, escalating_ceiling(ball_cap)) == 75
          and ghost_points(17, escalating_ceiling(balls_cap)) == 75)

    # NOT SCORABLE: a mode that cannot state a maximum is refused, not guessed.
    check("no ceiling -> no ghost points", ghost_points(17, None) is None)
    check("a ceiling that answers None for ONE shot -> none at all",
          ghost_points(17, lambda k: None if k == 9 else 10) is None)
    check("a zero-valued perfect shot -> none (it would divide by zero)",
          ghost_points(17, lambda k: 0) is None)
    check("a negative one -> none", ghost_points(17, lambda k: -3) is None)
    check("a fractional one -> none (a score is whole points)",
          ghost_points(17, lambda k: 2.5) is None)
    check("a boolean is not a point value", ghost_points(17, lambda k: True) is None)
    check("zero shots -> none", ghost_points(0, constant_ceiling(10)) is None)
    r = S(ceiling=None)
    check("a drill with no stated maximum scores NOT SCORABLE, never a number",
          not r.scorable and r.score is None and r.shown is None
          and r.points_share is None and r.speed_share is None
          and "maximum" in r.not_scorable, r.not_scorable)
    check("... and a NOT SCORABLE round has no score to store, but is NOT a discarded round "
          "(a drill that states no maximum is not a storm)",
          not r.record.fit and Unfit.NO_SCORE in r.record.reasons and not r.discarded)
    r = S(ceiling=lambda k: None)
    check("a ceiling that cannot answer is NOT SCORABLE too",
          not r.scorable and r.score is None, r.not_scorable)

    # ---------------------------------------------------------------- 4 ------
    print("\n4. his best round: 165 of 170 points in 9.34 s -> 416, factored exactly")
    r = S()
    check("it is scorable", r.scorable and r.not_scorable == "")
    check("the ghost earned 170 points in 4.0 s",
          r.ghost_points == 170 and close(r.ghost_elapsed, 4.0))
    check("the score is 416 (his logged best)", r.shown == 416,
          f"{fx(r.score, '.3f')} -> {r.shown}")
    # EXACT, not within a tolerance: the module computes `hp / ghost.points` and
    # `ghost.elapsed / he` with ghost.elapsed = 16 * 0.25 = 4.0 (a power-of-two
    # multiple, so exact). Same operands, same operation, same bits.
    check("points share = 165 / 170, bit for bit", r.points_share == 165 / 170,
          f"{r.points_share!r}")
    check("speed share = 4.0 / 9.34, bit for bit", r.speed_share == 4.0 / 9.34,
          f"{r.speed_share!r}")
    # INTERNAL CONSISTENCY (kept): the headline can never disagree with the two
    # factors shown beside it. Same expression, same order as the module, so exact.
    check("score == 1000 x points share x speed share, EXACTLY (the headline agrees "
          "with the factors a screen prints beside it)",
          r.scorable and r.score == 1000.0 * r.points_share * r.speed_share)
    # THE DEFINITION, from raw inputs, sharing nothing with the module's factors.
    # NOT bit-exact and cannot honestly be: this associates the operations
    # differently, so IEEE rounding may differ by a few ulp (see rel_close).
    check("score == the DEFINITION: 1000 x (human points per second) / (ghost "
          "points per second), computed from the raw inputs",
          rel_close(r.score, by_definition(165, 9.34, 17, constant_ceiling(10))),
          f"{r.score!r} vs {by_definition(165, 9.34, 17, constant_ceiling(10))!r}")
    check("a lower-bound flag is NOT raised on a drill with no waits",
          not r.lower_bound and r.lower_bound_why == "")

    rows = [(165, 9.34, 17, constant_ceiling(10)), (110, 12.9, 17, constant_ceiling(10)),
            (40, 8.0, 17, escalating_ceiling(5)), (17, 30.0, 17, constant_ceiling(1)),
            (60, 3.1, 9, constant_ceiling(10)), (2, 1.0, 2, escalating_ceiling(5))]
    exact = True
    plain = True
    for hp, he, n, ceil in rows:
        q = ghost_score(human_points=hp, human_elapsed=he, shots=n, ceiling=ceil,
                        peaks=HEALTHY_PEAKS, corroboration=HEALTHY_MIC)
        exact &= bool(q.scorable and q.score == 1000.0 * q.points_share * q.speed_share)
        plain &= rel_close(q.score, by_definition(hp, he, n, ceil))
    check("the headline equals 1000 x the two factors, EXACTLY, over six different "
          "rounds and modes", exact)
    check("and equals the DEFINITION (points per second over the ghost's) for every "
          "one of them, computed from the raw inputs", plain)

    r = S(human_points=40, human_elapsed=8.0, ceiling=escalating_ceiling(5))
    check("an escalating-ball round: 40 of 75 points in 8.0 s -> 40/75 x 4/8 x 1000 = 267",
          r.ghost_points == 75 and r.shown == 267, fx(r.score, ".2f"))

    r = S(human_points=170, human_elapsed=4.0)
    check("a human who IS the ghost scores exactly 1000",
          r.score == 1000.0 and r.shown == 1000, f"{r.score}")

    # ---------------------------------------------------------------- 5 ------
    print("\n5. the ghost's wait for targets that are not there yet")
    el, used = ghost_elapsed(3, [0.0, 1.0, 1.1])
    check("target 2 appears at 1.0 s: the ghost waits for it, then 0.25 s after for #3 "
          "(its own cadence beats a target already on screen)",
          used and close(el, 1.25), f"{el}")
    el, used = ghost_elapsed(3, [5.0, 5.1, 5.2])
    check("the wait before the FIRST shot is charged to nobody (clock starts at shot 1)",
          used and close(el, 0.5), f"{el}")
    el, used = ghost_elapsed(4, [0.0, 0.0, 0.0, 0.0])
    check("every target already there -> exactly the cadence-bound ghost",
          used and close(el, 0.75))
    el, used = ghost_elapsed(3, None)
    check("no waits supplied -> cadence-bound, and it says it did not use waits",
          not used and close(el, 0.5))
    el, used = ghost_elapsed(3, [0.0, 1.0])
    check("too FEW waits are not padded or extrapolated -- they are ignored",
          not used and close(el, 0.5))
    el, used = ghost_elapsed(3, [0.0, 1.0, 2.0, 3.0])
    check("too MANY are ignored as well", not used and close(el, 0.5))
    el, used = ghost_elapsed(3, [0.0, float("nan"), 2.0])
    check("a non-finite wait makes the whole sequence unusable (no raise)",
          not used and close(el, 0.5))
    el, used = ghost_elapsed(3, [0.0, "soon", 2.0])
    check("a non-number wait does too (no raise)", not used and close(el, 0.5))
    el, used = ghost_elapsed(1, [3.0])
    check("one shot has no span with or without waits", close(el, 0.0))

    # ---------------------------------------------------------------- 6 ------
    print("\n6. waits absent on a drill that makes you wait -> a LOWER BOUND, marked")
    kw = dict(human_points=20, human_elapsed=2.0, shots=3,
              ceiling=constant_ceiling(10), peaks=[85, 85, 85],
              corroboration=Corroboration(True, 3, 0))
    bare = ghost_score(has_waits=True, **kw)
    check("has_waits and no times -> flagged as a lower bound, with the reason",
          bare.lower_bound and "not known" in bare.lower_bound_why, bare.lower_bound_why)
    check("... and the ghost used is the cadence-bound one (0.5 s)",
          close(bare.ghost_elapsed, 0.5))
    check("score = 20/30 x 0.5/2.0 x 1000 = 166.67",
          close(bare.score, 1000 * (20 / 30) * (0.5 / 2.0), 1e-9), fx(bare.score, ".2f"))
    real = ghost_score(has_waits=True, available_at=[0.0, 1.0, 2.0], **kw)
    check("with the real appearance times it is NOT a lower bound",
          not real.lower_bound and real.lower_bound_why == "")
    check("the ghost then waits for the targets: 2.0 s, and the score is 666.67",
          close(real.ghost_elapsed, 2.0) and close(real.score, 1000 * (20 / 30) * 1.0, 1e-9),
          fx(real.score, ".2f"))
    check("the lower bound really IS lower: the true score is at least the flagged one "
          "(this is why Reaction reads 47 and should read several hundred)",
          real.score is not None and bare.score is not None and real.score > bare.score)
    # WHY RECORDING A FLOOR IS SAFE (his ruling, 2026-10-22): the floor is never ABOVE
    # what the same round scores once the waits are known, whatever those waits are.
    # So a stored floor can understate the round and cannot overstate it, and it can
    # never be an unbeatable number (a human did at least that well). Asserted over
    # waits from nothing at all to far longer than the human's own span, through the
    # module (the ghost with waits is the SAME ghost, only slower).
    over_floor = []
    for w_ in (0.0, 0.05, 0.25, 0.5, 1.0, 2.0, 10.0):
        av_ = available_from_waits([0.0, w_, w_], 60.0)
        t_ = ghost_score(has_waits=True, available_at=av_, **kw)
        if not (t_.scorable and not t_.lower_bound and bare.score is not None
                and t_.score is not None and bare.score <= t_.score + 1e-9):
            over_floor.append((w_, fx(t_.score, ".2f")))
    check("a floor NEVER exceeds the score the same round gets with ANY wait profile, from none "
          "to ten seconds a target (so a recorded floor can only understate, never inflate, "
          "and can never become a record nobody can beat)",
          not over_floor and bare.lower_bound, f"{over_floor}")
    wrong = ghost_score(has_waits=True, available_at=[0.0, 1.0], **kw)
    check("times that do not match the shots stay a lower bound, with a different reason",
          wrong.lower_bound and "do not match" in wrong.lower_bound_why,
          wrong.lower_bound_why)
    quiet = ghost_score(has_waits=False, **kw)
    check("a drill with no waits is never flagged, whatever else it is given",
          not quiet.lower_bound)

    # ---------------------------------------------------------------- 7 ------
    print("\n7. degenerate rounds never raise into a round screen")
    # Every call goes through S_safe: a raise is a FAILED CHECK here, never a
    # crashed suite (which the mutation runner would count BROKEN, not caught).
    r, err = S_safe(shots=1, human_points=10, human_elapsed=0.0)
    check("ONE shot does not raise: a zero-length span is NOT SCORABLE, not an "
          "exception", r is not None, err)
    check("ONE shot: nothing to divide by -> NOT SCORABLE, not 0",
          r is not None and not r.scorable and r.score is None
          and "two shots" in r.not_scorable, r.not_scorable if r else err)
    check("... a ghost of one shot still reports its 10 points and 0 s",
          r is not None and r.ghost_points == 10 and r.ghost_elapsed == 0.0)
    check("... and a one-shot round has no score to store (and is not a discarded storm)",
          r is not None and not r.record.fit and Unfit.NO_SCORE in r.record.reasons
          and not r.discarded)
    r, err = S_safe(human_elapsed=0.0)
    check("seventeen shots in one instant (human clock 0) -> no score, no raise, and it is "
          "DISCARDED as faster than a pistol (not merely 'not scorable')",
          r is not None and not r.scorable and r.score is None and r.discarded
          and Unfit.TOO_FAST in r.record.reasons, err)
    r, err = S_safe(human_elapsed=float("nan"))
    check("a NaN clock -> NOT SCORABLE, no raise (a clock that is not a number is not a storm)",
          r is not None and not r.scorable and r.score is None and not r.discarded, err)
    r, err = S_safe(human_elapsed=-1.0)
    check("a negative clock -> no score, no raise",
          r is not None and not r.scorable and r.score is None, err)
    r, err = S_safe(shots=0, human_points=0, human_elapsed=0.0)
    check("an empty round -> NOT SCORABLE, no raise",
          r is not None and not r.scorable and "no shots" in r.not_scorable,
          r.not_scorable if r else err)
    r, err = S_safe(human_points=0)
    check("a round of nothing but misses scores 0, not a refusal",
          r is not None and r.scorable and r.score == 0.0 and r.shown == 0, err)
    r, err = S_safe(human_points=171)
    check("MORE points than the stated maximum -> NOT SCORABLE (the ceiling is wrong; "
          "a quietly wrong number is worse than none)",
          r is not None and not r.scorable and "exceeds" in r.not_scorable
          and r.points_share is None, r.not_scorable if r else err)

    # ---------------------------------------------------------------- 8 ------
    print("\n8. a healthy round is SOUND: it is scored, shown and recordable")
    r = S()
    check("bright shots, a microphone that heard them, a human pace -> sound: no reasons, no "
          "cautions, not discarded, and it has a score",
          r.record.fit and r.record.reasons == () and r.record.text == ""
          and r.record.cautions == () and r.record.caution_text == ""
          and not r.discarded and r.scorable and r.shown == 416 and r.withheld_score is None)

    # ---------------------------------------------------------------- 9 ------
    print("\n9. the validity test: ONLY the span discards a round; brightness and the "
          "microphone are NOTED (cautions), one at a time")
    r = S(peaks=[15.0] * 17)
    check("median peak 15 -> the round is SOUND and SCORED (shown 416, not discarded); the "
          "brightness is a CAUTION and nothing else (brightness alone can never discard a round)",
          r.record.fit and r.record.reasons == () and r.record.cautions == (Unfit.PEAK_TOO_LOW,)
          and r.scorable and not r.discarded and r.shown == 416 and r.score is not None,
          f"{r.record.reasons} / {r.record.cautions}")
    check("... and the caution sentence names the number it saw and says it is noted only",
          "15" in r.record.caution_text and "noted only" in r.record.caution_text
          and r.record.text == "", r.record.caution_text)
    check("median peak exactly 30 raises no caution (the boundary is inclusive)",
          S(peaks=[30.0] * 17).record.cautions == ())
    check("median 29.9 does, and still scores",
          Unfit.PEAK_TOO_LOW in S(peaks=[29.9] * 17).record.cautions
          and S(peaks=[29.9] * 17).scorable)
    check("a few dim real shots do not even raise the caution: it is the MEDIAN",
          S(peaks=[85.0] * 10 + [20.0] * 7).record.cautions == ())
    check("one bright outlier does not rescue a round from the caution: sixteen 9s and a 200 "
          "(the median decides, not the maximum)",
          Unfit.PEAK_TOO_LOW in S(peaks=[9.0] * 16 + [200.0]).record.cautions)
    r = S(peaks=())
    check("no peaks supplied -> PEAK_UNKNOWN caution only, and the round is still scored "
          "(the span alone judges it)",
          r.record.cautions == (Unfit.PEAK_UNKNOWN,) and r.record.fit and r.scorable
          and not r.discarded, f"{r.record.cautions}")
    check("peaks that are all None are the same as none",
          S(peaks=[None] * 17).record.cautions == (Unfit.PEAK_UNKNOWN,))
    check("None among real peaks is skipped, not counted as zero",
          S(peaks=[85.0] * 9 + [None] * 8).record.cautions == ())

    floor = MIN_SPAN_RATIO * (17 - 1) * GHOST_INTERVAL
    r = S(human_elapsed=floor - 0.01, human_points=100)
    check("3.63 s for 17 shots -> DISCARDED for being too fast and nothing else",
          r.discarded and r.record.reasons == (Unfit.TOO_FAST,) and not r.record.fit,
          f"{r.record.reasons}")
    check("... a discarded round has NO figure to show: no score, no whole number, no aim, no "
          "speed, and it is not scorable -- the screen cannot print what is not there",
          not r.scorable and r.score is None and r.shown is None and r.points_share is None
          and r.speed_share is None)
    # The reference round is the same round AT the floor. It is read through a name and
    # asserted SCORED before its score is used: with the floor moved (mutate_ghost fault
    # 4 counts shots instead of their gaps) the reference is itself discarded, its score
    # is None, and `None * float` killed the suite instead of failing this check by name.
    at_floor = S(human_elapsed=floor, human_points=100)
    # THE WITHHELD SCORE IS THE CLAMPED ONE (2026-10-23). A round faster than the
    # ghost is exactly what TOO_FAST means, so its raw speed share is above 1 and the
    # 1000-ceiling clamp fires on every discarded round of a no-wait drill. The
    # withheld figure is "what the book would have been offered", and the book is
    # offered the clamped score, so that is what is kept; the RAW speed survives
    # beside it for the log. Before the clamp this check asserted the unclamped
    # relation (withheld = sound score x floor / (floor - 0.01)), which no longer
    # holds: both rounds below are clamped to aim alone.
    aim_only = 1000.0 * 100 / 170
    check("... what it would have scored is kept for the LOG only (withheld_score): the "
          "CLAMPED figure (aim alone, 1000 x 100/170), the same number the same round scores "
          "when it is sound but still faster than the ghost -- and the raw speed share "
          "(4.0 / 3.63) is recoverable beside it, with the clamp recorded",
          r.withheld_score is not None and at_floor.scorable and at_floor.score is not None
          and rel_close(r.withheld_score, aim_only, 1e-12)
          and rel_close(at_floor.score, aim_only, 1e-12)
          and r.speed_raw is not None and rel_close(r.speed_raw, 4.0 / (floor - 0.01), 1e-12)
          and r.clamped == (ghost.CLAMPED_SPEED,) and r.speed_share is None,
          f"{r.withheld_score} vs {fx(at_floor.score)}; raw {fx(r.speed_raw)}")
    check("... the sentence says what the least that counts is",
          "3.6" in r.record.text and "17 shots" in r.record.text
          and "DISCARDED" in r.not_scorable, r.record.text)
    check("exactly 3.64 s is sound (inclusive)", S(human_elapsed=floor, human_points=100).record.fit
          and not S(human_elapsed=floor, human_points=100).discarded)
    floor5 = MIN_SPAN_RATIO * (5 - 1) * GHOST_INTERVAL
    r5 = ghost_score(human_points=30, human_elapsed=floor5 - 0.01, shots=5,
                     ceiling=constant_ceiling(10), peaks=[85] * 5,
                     corroboration=Corroboration(True, 5, 0))
    r5b = ghost_score(human_points=30, human_elapsed=floor5, shots=5,
                      ceiling=constant_ceiling(10), peaks=[85] * 5,
                      corroboration=Corroboration(True, 5, 0))
    check("the floor is DERIVED per magazine: 5 shots -> 0.91 s, not 3.64",
          close(floor5, 0.91) and r5.record.reasons == (Unfit.TOO_FAST,) and r5.discarded
          and r5b.record.fit and not r5b.discarded,
          f"{r5.record.reasons} / fit={r5b.record.fit}")

    # THE GHOST MUST BE REACHABLE (2026-10-22: with a floor of 1.2 x the cadence span,
    # a round at the system's own limit was DISCARDED and the score could never pass
    # about 833). A round at EXACTLY the ghost's pace is sound and scores 1000, for
    # EVERY magazine the app allows (1 to 50): the property, not one example.
    unreachable = []
    for n_ in range(2, 51):
        q_ = ghost_score(human_points=10 * n_, human_elapsed=(n_ - 1) * 0.25, shots=n_,
                         ceiling=constant_ceiling(10), peaks=[85.0] * n_, corroboration=None)
        # clamped == () AND speed_share == speed_raw: the clamp must not ALTER an
        # unclamped case (speed is exactly 1.0 here), not merely land on 1000.
        if not (q_.record.fit and not q_.discarded and q_.score == 1000.0
                and q_.clamped == () and q_.speed_share == q_.speed_raw == 1.0):
            unreachable.append(n_)
    check("a human who IS the ghost is SOUND and scores exactly 1000 at every magazine from 2 "
          "to 50 -- a floor that re-caps the score fails here, by name", not unreachable,
          f"unreachable at magazine {unreachable}")
    # ONE shot has no span to judge. It is NOT SCORABLE (nothing to divide by), and it
    # is never DISCARDED as a storm: the cadence rule cannot see a one-shot burst.
    # Through S_safe: this is the zero-span round, and the guard that stops it dividing
    # by zero is exactly what mutate_ghost fault 25 deletes. A raise is a FAILED CHECK
    # here (named, below), never a dead suite.
    q1, q1_err = S_safe(human_points=10, human_elapsed=0.0, shots=1,
                        ceiling=constant_ceiling(10), peaks=[85.0], corroboration=None)
    check("magazine 1: a zero-span round does not raise, is not scorable and is not discarded "
          "(a span of zero shots' gaps is no evidence)",
          q1 is not None and not q1.scorable and not q1.discarded
          and q1.record.reasons == (Unfit.NO_SCORE,),
          q1_err or f"{q1.record.reasons}")
    # Magazine 2: the floor is 0.2275 s against a ghost of 0.25 s. Stated as a number so
    # the thin margin is on the page and a change that thins it further turns red.
    f2 = MIN_SPAN_RATIO * GHOST_INTERVAL
    check("magazine 2 is the thinnest case: the floor sits 0.0225 s under the ghost's 0.25 s "
          "(less than one 30 fps frame of 0.033 s -- stated, not hidden)",
          close(f2, 0.2275) and close(GHOST_INTERVAL - f2, 0.0225),
          f"floor {f2}")

    # HIS RULING, 2026-10-21: "it is OK to allow personal best w/o a working
    # microphone". No microphone is not a reason; it is a skipped condition. His
    # Kiyo Pro's own microphone is stuck in a PnP state needing a reboot, so this
    # is his situation today. 2026-10-22 takes it one step further: the microphone
    # is a CAUTION and can never discard a round, however silent -- a ready
    # microphone that heard NOTHING of a round at a human pace leaves the round
    # sound, scored and recordable.
    check("the module has no 'microphone unavailable' reason left to emit",
          not hasattr(Unfit, "MIC_UNAVAILABLE"))
    r = S(corroboration=None)
    check("NO microphone verdict at all, good peaks, good span -> SOUND, no cautions",
          r.record.fit and r.record.reasons == () and r.record.text == ""
          and r.record.cautions == () and not r.discarded, f"{r.record.reasons}")
    r = S(corroboration=Corroboration(False, 0, 0))
    check("a microphone that has not calibrated is SKIPPED: sound even with 0 heard, "
          "and not even a caution",
          r.record.fit and r.record.reasons == () and r.record.cautions == (),
          f"{r.record.reasons}")
    r = S(corroboration=Corroboration(True, 0, 17))
    check("deaf for every shot (the app's own self-mute) is SKIPPED: sound, no caution",
          r.record.fit and r.record.reasons == () and r.record.cautions == (),
          f"{r.record.reasons}")
    r = S(corroboration=Corroboration(True, 0, 0))
    check("0 of 17 confirmed by a READY microphone -> MIC_UNCORROBORATED as a CAUTION only: the "
          "round is still sound, still scored (416) and still not discarded",
          r.record.cautions == (Unfit.MIC_UNCORROBORATED,) and r.record.reasons == ()
          and r.record.fit and r.scorable and not r.discarded and r.shown == 416,
          f"{r.record.reasons} / {r.record.cautions}")
    check("... and the caution sentence gives the counts", "0 of 17" in r.record.caution_text,
          r.record.caution_text)
    check("8 of 17 is under half (a caution); 9 of 17 is a majority (none)",
          Unfit.MIC_UNCORROBORATED in S(corroboration=Corroboration(True, 8, 0)).record.cautions
          and S(corroboration=Corroboration(True, 9, 0)).record.cautions == ())
    check("the fraction is the module's own constant", MIN_CORROBORATED == 0.5)
    check("shots fired while the microphone was deaf leave the denominator: "
          "4 of 17 with 8 deaf = 4 of 9, a caution; 5 of 9 is none",
          Unfit.MIC_UNCORROBORATED in S(corroboration=Corroboration(True, 4, 8)).record.cautions
          and S(corroboration=Corroboration(True, 5, 8)).record.cautions == ())
    check("brightness and the microphone TOGETHER still cannot discard a round: faint peaks AND "
          "a ready microphone that heard none, at a human pace -> two cautions, sound, scored",
          S(peaks=[15.0] * 17, corroboration=Corroboration(True, 0, 0)).record.fit
          and set(S(peaks=[15.0] * 17, corroboration=Corroboration(True, 0, 0)).record.cautions)
          == {Unfit.PEAK_TOO_LOW, Unfit.MIC_UNCORROBORATED}
          and S(peaks=[15.0] * 17, corroboration=Corroboration(True, 0, 0)).scorable)

    # THE STORM, ANCHORED. Carried verbatim from a real log, the way
    # test_ui_smoke carries his nine real calibration camera coordinates. It is a
    # COPY and the suite never reads the log: logs rotate and are untracked, and a
    # check that depends on the state of the world fails when nothing is wrong.
    #
    # SOURCE  logs\camera_lab.20261021-123712.log (the log of the launch that
    #         began 2026-10-20 17:55:59), lines 1059-1123, 22:38:16 - 22:38:20:
    #   22:38:16  round: round 1 starts (Static Targets): magazine 17
    #   22:38:17..22:38:19  SHOT 33 .. SHOT 49   (17 registered shots, 8 of them
    #             "HIT centre", at +7 +8 +8 +7 +8 +8 +8 +7 = 61 points)
    #   22:38:19  round: magazine 17 is empty: 61 pts, 17 shots, 8 hits
    #   22:38:20  round: round 1 efficiency, magazine: EFFICIENCY  22.59
    #             (= 61 points / 2.7 seconds)
    # The shots' own t= stamps run 470785.831 .. 470788.515 = 2.684 s; the app's
    # efficiency clock says 2.7 (61 / 22.59 = 2.700). The ROUND SCORE uses the
    # app's clock, 2.70, because that is the one a screen would have been given.
    # Peaks, in shot order, from the "peak=" field of those 17 SHOT lines. Every
    # one of the 17 also carries "[DEAF: the app's own gunshot had the microphone
    # muted ...]", and the "both" counter stays at 8 across the round.
    # The other two storms, same conditions (figures from their round lines):
    #   22:38:49  Static Target:  14 pts, 17 shots, 2.2 s   (same log, line 1804)
    #   2026-10-21 12:39:30  Moving Target: 0 pts, 17 shots, 3.3 s
    #                         (logs\camera_lab.log, line 286)
    STORM_POINTS = 61
    STORM_ELAPSED = 2.70
    STORM_PEAKS = [17, 16, 20, 25, 14, 20, 17, 21, 17, 16, 14, 22, 14, 15, 13, 17, 25]
    check("the anchored storm is what the log says: 17 shots, median peak 17",
          len(STORM_PEAKS) == 17 and sorted(STORM_PEAKS)[8] == 17
          and min(STORM_PEAKS) == 13 and max(STORM_PEAKS) == 25)

    def storm_round(corroboration):
        return S(human_points=STORM_POINTS, human_elapsed=STORM_ELAPSED,
                 peaks=STORM_PEAKS, corroboration=corroboration)

    # THE OTHER HALF OF THE FIXTURE PAIR (2026-10-22). Same rule as the storm: a COPY of
    # real numbers, because logs rotate and a check that reads the world fails when
    # nothing is wrong.
    #
    # SOURCE  logs\camera_lab.log, 2026-10-22, round 2 of Escalating Ball, started
    #         20:44:58, closed 20:45:13 (the log line of the round, at 20:45:14):
    #   score: 256.3 / 1000 (measured); aim 0.933 x speed 0.275; human 70 pts in
    #   14.57s over 17 shots; ghost 75 pts in 4.00s; efficiency 4.79; record REFUSED:
    #   the median shot brightness was 28 ...
    # which is the line that put `SCORE 256` over `BEST SCORE SO FAR 212` on his screen
    # and recorded nothing, and so is the reason this file's verdicts changed shape.
    # Shots 110..126 (17 registered; shot 127 was a dry pull), the "peak=" of each:
    #   110:54 111:87 112:15 113:28 114:21 115:74 116:84 117:17 118:23 119:74
    #   120:21 121:81 122:28 123:22 124:9  125:83 126:9
    # 16 hits (112 was the miss): +1 +2 miss +3 +4 then 5 on each of twelve = 70.
    # The shots' own t= stamps run 636785.900 .. 636800.465 = 14.565 s (the log's
    # 14.57). Shot 110 had the microphone listening (mic_peak 0.514 over its bar of
    # 0.303); shots 111..126 each carry "[DEAF: the app's own gunshot had the
    # microphone muted ...]", so the round's pairing is 1 heard-and-seen, 16 deaf.
    REAL_POINTS = 70
    REAL_ELAPSED = 14.565
    REAL_PEAKS = [54, 87, 15, 28, 21, 74, 84, 17, 23, 74, 21, 81, 28, 22, 9, 83, 9]
    REAL_MIC = Corroboration(True, 1, 16)
    REAL_CEILING = escalating_ceiling(5)          # 17 shots -> 75, as the ball drills' own cap
    check("(premise) the anchored real round is what the log says: 17 shots, median peak "
          "28 (under the laser/noise boundary of 30), its ghost takes 75 points",
          len(REAL_PEAKS) == 17 and sorted(REAL_PEAKS)[8] == 28 and sorted(REAL_PEAKS)[8] < 30
          and ghost_points(17, REAL_CEILING) == 75)

    def real_round(**over):
        kw = dict(human_points=REAL_POINTS, human_elapsed=REAL_ELAPSED, shots=17,
                  ceiling=REAL_CEILING, peaks=REAL_PEAKS, corroboration=REAL_MIC)
        kw.update(over)
        return ghost_score(**kw)

    span_floor = MIN_SPAN_RATIO * (17 - 1) * GHOST_INTERVAL
    check("THE LINE BETWEEN THEM, derived: the storm's 2.70 s is under the 3.64 s floor and "
          "his real 14.565 s is over it (1.35x under and 4.0x over)",
          STORM_ELAPSED < span_floor < REAL_ELAPSED
          and rel_close(span_floor / STORM_ELAPSED, 1.348, 0.01)
          and rel_close(REAL_ELAPSED / span_floor, 4.0, 0.01))

    # THE FIXTURE SET THAT FIXES THE RATIO (2026-10-22). Every case as a RATIO of its span
    # to the pure cadence span, (shots - 1) * GHOST_INTERVAL = 4.0 s for 17 shots. These
    # are COPIES of real measurements (the storms, his round) plus the one case that is
    # definitional (the ghost), not values typed to make a check pass. The floor must
    # lie strictly above the slowest storm and at or below the ghost; where in that gap
    # it was put is derived beside MIN_SPAN_RATIO in ghost.py.
    CADENCE17 = (17 - 1) * 0.25                  # typed from the definition, not read from the module
    SLOWEST_STORM = 3.3 / CADENCE17              # 0.825: Moving Target, 2026-10-21 12:39:30
    storm_cases = (("the 2.70 s storm", STORM_ELAPSED), ("the 2.20 s storm", 2.2),
                   ("the 3.30 s storm", 3.3))
    check("(premise) the three real storms are 0.675, 0.55 and 0.825 of the cadence span, and "
          "the slowest is 0.825 -- NOT the 0.78 a shots-x-interval basis gives",
          [round(e_ / CADENCE17, 3) for _n, e_ in storm_cases] == [0.675, 0.55, 0.825]
          and max(e_ for _n, e_ in storm_cases) / CADENCE17 == SLOWEST_STORM)
    check("THE RATIO sits above the slowest real storm and at or below the ghost's 1.0 "
          "(so the ghost is reachable), with at least 0.05 to spare on each side so neither "
          "neighbour is decided by rounding",
          SLOWEST_STORM + 0.05 <= MIN_SPAN_RATIO <= 1.0 - 0.05,
          f"{SLOWEST_STORM} < {MIN_SPAN_RATIO} <= 1.0")
    for label_, e_ in storm_cases:
        q_ = S(human_points=STORM_POINTS, human_elapsed=e_, peaks=STORM_PEAKS, corroboration=None)
        check(f"{label_} is DISCARDED (ratio {e_ / CADENCE17:.3f}), by the span alone",
              q_.discarded and q_.score is None and q_.record.reasons == (Unfit.TOO_FAST,),
              f"{q_.record.reasons}")
    ghost_case = S(human_points=170, human_elapsed=CADENCE17)
    check("A HUMAN WHO IS THE GHOST (ratio 1.000) is SCORED 1000, not discarded -- the case that "
          "turns red if the discard floor is ever set back above the ghost's own pace",
          ghost_case.scorable and not ghost_case.discarded and ghost_case.shown == 1000
          and ghost_case.record.fit and ghost_case.record.reasons == (), f"{ghost_case.shown}")
    check("... and so is a human a tenth SLOWER than the ghost (ratio 1.1) and one at his real "
          "round's pace (ratio 3.64): the whole range above the line is scored",
          not S(human_points=170, human_elapsed=1.1 * CADENCE17).discarded
          and not real_round().discarded)

    real = real_round()
    check("HIS REAL ROUND IS SCORED: 256 (aim 0.933 x speed 0.275), shown, not discarded -- "
          "and a round that is shown is a round that is recorded",
          real.scorable and not real.discarded and real.shown == 256
          and close(real.points_share, 70 / 75) and close(real.speed_share, 4.0 / 14.565)
          and real.withheld_score is None and real.record.fit and real.record.reasons == (),
          f"{real.score}")
    check("... its median peak of 28 (under 30) is a CAUTION in the log and NOT a reason: "
          "this is the check that would have discarded his record round",
          real.record.cautions == (Unfit.PEAK_TOO_LOW,) and "28" in real.record.caution_text
          and real.record.text == "")
    check("... with the microphone absent, uncalibrated, or wholly deaf it is scored all the "
          "same (his ruling of 2026-10-21, and the same score in every case)",
          all(real_round(corroboration=c).shown == 256 and not real_round(corroboration=c).discarded
              for c in (None, Corroboration(False, 0, 0), Corroboration(True, 0, 17))))

    # The storm: the same rule, the same arithmetic, the opposite answer.
    storm = storm_round(Corroboration(True, 0, 17))     # deaf for all 17, AS LOGGED
    check("THE PHANTOM STORM IS DISCARDED WHOLE: it has no score, no whole number, no aim "
          "and no speed to show (it would have been 359 -- 532 before the 1000 ceiling capped "
          "its speed share at 1.0 -- and that survives only in withheld_score for the log, "
          "with the raw speed 4.0 / 2.7 beside it)",
          storm.discarded and not storm.scorable and storm.score is None and storm.shown is None
          and storm.points_share is None and storm.speed_share is None
          and storm.withheld_score is not None and round(storm.withheld_score) == 359
          and storm.speed_raw is not None and close(storm.speed_raw, 4.0 / 2.7, 1e-9)
          and storm.clamped == (ghost.CLAMPED_SPEED,),
          f"{storm.withheld_score}")
    check("... for the span, and the span alone: 17 shots in 2.7 s, with the microphone "
          "deaf, so it could testify to nothing",
          storm.record.reasons == (Unfit.TOO_FAST,) and not storm.record.fit
          and "17 shots in 2.7 s" in storm.record.text, storm.record.text)
    check("... and its dim median (17) is a caution beside the verdict, not part of it",
          storm.record.cautions == (Unfit.PEAK_TOO_LOW,))

    # THE WHOLE SAFETY ARGUMENT, now on the span alone: the storm is discarded when
    # the microphone is absent, uncalibrated, wholly deaf (as logged), or ready and
    # hearing nothing -- because no microphone is consulted to discard it. This is what
    # must go red if the span test is removed or loosened.
    for label, corr in (("no microphone at all", None),
                        ("a microphone that has not calibrated",
                         Corroboration(False, 0, 0)),
                        ("deaf for all 17 shots, AS LOGGED",
                         Corroboration(True, 0, 17)),
                        ("a READY microphone that heard none (a CONSTRUCTED variant: in the "
                         "real log it was deaf)", Corroboration(True, 0, 0))):
        q = storm_round(corr)
        check(f"the storm is DISCARDED with {label}: by the span, and by nothing else",
              q.discarded and q.score is None and q.record.reasons == (Unfit.TOO_FAST,),
              f"{q.record.reasons}")
    check("neither brightness nor the microphone is what separates the two rounds: the "
          "storm's peaks at his round's pace are SCORED, and his round's peaks at the "
          "storm's pace are DISCARDED",
          not S(human_points=STORM_POINTS, human_elapsed=REAL_ELAPSED, peaks=STORM_PEAKS,
                corroboration=None).discarded
          and S(human_points=STORM_POINTS, human_elapsed=REAL_ELAPSED, peaks=STORM_PEAKS,
                corroboration=None).scorable
          and real_round(human_elapsed=STORM_ELAPSED).discarded)
    check("... and the storm's timing with laser-bright peaks and no microphone is still "
          "discarded (the span needs nothing else)",
          S(human_points=STORM_POINTS, human_elapsed=STORM_ELAPSED,
            corroboration=None).discarded)

    direct = record_fitness(scorable=True, shots=17, human_elapsed=9.34,
                            peaks=HEALTHY_PEAKS, corroboration=HEALTHY_MIC)
    check("record_fitness is callable on its own and agrees with ghost_score",
          direct.fit and direct == S().record)
    dstorm = record_fitness(scorable=True, shots=17, human_elapsed=STORM_ELAPSED,
                            peaks=STORM_PEAKS, corroboration=None)
    check("... and on the storm it returns exactly the verdict ghost_score discarded it on",
          not dstorm.fit and dstorm == storm_round(None).record)

    # --------------------------------------------------------------- 10 ------
    print("\n10. the seams: duplicated constants are checked against their source")
    camera_lab = ROOT / "src" / "lsa" / "tools" / "camera_lab.py"
    boundary = source_constant(camera_lab, "SINGLE_FRAME_PEAK")
    check("the laser/noise boundary was found in camera_lab.py", boundary is not None)
    check("MIN_MEDIAN_PEAK equals CameraLab.SINGLE_FRAME_PEAK (the one measured boundary)",
          boundary is not None and MIN_MEDIAN_PEAK == float(boundary),
          f"{MIN_MEDIAN_PEAK} vs {boundary}")
    laser_src = (ROOT / "src" / "lsa" / "detect" / "laser.py").read_text(encoding="utf-8")
    check("GHOST_INTERVAL is a literal in ghost.py, not read from the tracker's refractory",
          re.search(r"^GHOST_INTERVAL\s*=\s*0\.25\s*$",
                    src_path.read_text(encoding="utf-8"), re.M) is not None
          and "refractory_ms" in laser_src)
    check("the module exposes its public surface",
          all(hasattr(ghost, n) for n in
              ("ghost_score", "ghost_round", "ghost_points", "ghost_elapsed",
               "record_fitness", "Ceiling", "Corroboration", "Unfit")))

    # --------------------------------------------------------------- 11 ------
    print("\n11. a drill's WAITS become the ghost's own timeline, plus one camera frame")
    f60 = 1.0 / 60.0
    check("one camera frame is DERIVED from the fps: 60 -> 1/60 s, 30 -> 1/30 s",
          frame_seconds(60.0) == 1.0 / 60.0 and frame_seconds(30) == 1.0 / 30.0)
    check("the floor is exactly ONE frame (the shot's stamp is the pulse's FIRST frame; "
          "the 2-frame confirmation comes after it)", GHOST_SEE_FRAMES == 1)
    check("an fps that cannot be a rate yields no floor at all: 0, negative, NaN, "
          "inf, a bool, a string, None",
          all(frame_seconds(v) is None
              for v in (0, -30.0, float("nan"), float("inf"), True, "60", None)))
    av = available_from_waits([9.9, 2.0, 2.0], 60.0)
    check("the first wait is ignored (charged to nobody); each later one is the wait "
          "PLUS one frame after the ghost's OWN previous shot: [0, 2+f, 2+f+2+f]",
          av is not None and av[0] == 0.0 and close(av[1], 2.0 + f60)
          and close(av[2], 2.0 + f60 + 2.0 + f60), f"{av}")
    check("... so a 3-shot ghost takes 4 s of waiting + 2 frames, through the real "
          "ghost_elapsed, and the waits ARE used",
          av is not None and ghost_elapsed(3, av) == (av[2], True)
          and close(ghost_elapsed(3, av)[0], 4.0 + 2 * f60))
    av = available_from_waits([0.0, 0.1, 0.1], 60.0)
    check("waits shorter than the 0.25 s cadence change nothing: the cadence binds "
          "(0.5 s for 3 shots)", av is not None and close(ghost_elapsed(3, av)[0], 0.5))
    av = available_from_waits([0.0, 0.0, 0.0], 60.0)
    check("a zero wait is a target already there: NO frame is charged for it "
          "(a second hit on the same figure)",
          av is not None and av[0] == 0.0 and av[1] == 0.0
          and close(ghost_elapsed(3, av)[0], 0.5), f"{av}")
    av = available_from_waits([0.0, 0.0, 0.0, 2.0, 0.0, 0.0], 60.0)
    check("a figure that takes 3 hits: waits [0,0,0,W,0,0] -> the ghost fires the "
          "first three at its cadence (0.5 s), waits W + a frame after its own 3rd "
          "shot, then two more at cadence",
          av is not None and close(av[3], 0.5 + 2.0 + f60)
          and close(ghost_elapsed(6, av)[0], 0.5 + 2.0 + f60 + 0.5), f"{av}")
    check("unusable input yields None, never a raise: no waits, a bad fps, a "
          "negative, NaN, bool or string wait",
          all(available_from_waits(w, f) is None for w, f in (
              (None, 60.0), ([0.0, 2.0], 0), ([0.0, 2.0], None),
              ([0.0, -1.0], 60.0), ([0.0, float("nan")], 60.0),
              ([0.0, True], 60.0), ([0.0, "2"], 60.0))))
    check("the first target of a one-shot round is simply [0.0]",
          available_from_waits([5.0], 60.0) == [0.0])

    # REACTION, the drill that read 47. 17 shots, a target after each hit at
    # his drill's average wait (1.2..3.5 -> 2.35 s), he averages 0.55 s to hit it
    # for 7 points. With the waits supplied the ghost is charged the same waits.
    W, R = 2.35, 0.55
    waits17 = [0.0] + [W] * 16
    human_elapsed = 16 * (W + R)
    avail17 = available_from_waits(waits17, 60.0)
    kwr = dict(human_points=17 * 7, human_elapsed=human_elapsed, shots=17,
               ceiling=constant_ceiling(10), has_waits=True)
    bare = ghost_score(**kwr)
    real = ghost_score(available_at=avail17, **kwr)
    check("Reaction WITHOUT its waits is a flagged lower bound reading in the tens "
          "(the 47 he saw)",
          bare.lower_bound and bare.shown is not None and bare.shown < 100, f"{bare.shown}")
    check("Reaction WITH its waits reads in the hundreds and is NOT a lower bound",
          (not real.lower_bound) and real.shown is not None and 400 <= real.shown <= 800
          and real.ghost_elapsed is not None and real.ghost_elapsed > 16 * W,
          f"{real.shown} / ghost {fx(real.ghost_elapsed, '.2f')} s")
    check("the waits moved the ghost's time and the share, not the points: the "
          "ghost's points are 170 either way",
          real.ghost_points == bare.ghost_points == 170)

    # --------------------------------------------------------------- 12 ------
    print("\n12. a wait the DRILL declares stands in for a target the round never drew: "
          "an estimate, counted and named, never a lower bound")
    check("expected_uniform is the mean of the range it is handed: (1.2, 3.5) -> 2.35",
          close(ghost.expected_uniform(1.2, 3.5), (1.2 + 3.5) / 2.0))
    check("expected_uniform refuses what is not a range: reversed, negative, NaN, inf, "
          "bool, string, None",
          all(ghost.expected_uniform(a, b) is None
              for a, b in ((3.5, 1.2), (-1.0, 2.0), (float("nan"), 2.0), (1.0, float("inf")),
                           (True, 2.0), ("1", 2.0), (None, 2.0))))

    def mean_max(lo, hi, fl, n=20000):
        """E[max(U, fl)] by midpoint integration: shares nothing with the closed form."""
        return sum(max(lo + (hi - lo) * (k + 0.5) / n, fl) for k in range(n)) / n

    def emm_safe(lo, hi, fl):
        """expected_max_uniform, with a RAISE turned into a value the check can
        report. A crash here ended the whole suite (mutate_ghost fault 26 came
        back BROKEN, not CAUGHT): with the `floor >= hi` guard gone, a zero-width
        range divides by zero. A raise is a FAILED CHECK, never a dead suite."""
        try:
            return ghost.expected_max_uniform(lo, hi, fl), None
        except Exception as exc:                      # noqa: BLE001 -- by design
            return None, f"{type(exc).__name__}: {exc}"

    # The check that names the fault runs FIRST and on an input that cannot crash
    # (a floor above a range of real width), so the fault is recorded as a failing
    # check before anything else is attempted.
    got_over, err_over = emm_safe(0.5, 1.5, 3.0)
    check("expected_max_uniform: a floor above the whole range IS the floor, and a bad input is None",
          err_over is None and got_over is not None and close(got_over, 3.0)
          and emm_safe(0.5, 1.5, -1.0) == (None, None)
          and emm_safe(1.5, 0.5, 1.0) == (None, None), f"{got_over} {err_over}")

    ok_mm = True
    seen_mm = []
    for lo_, hi_, fl_ in ((0.5, 1.5, 1.0), (0.5, 1.5, 0.0), (0.5, 1.5, 0.5), (0.5, 1.5, 1.5),
                          (0.5, 1.5, 3.0), (1.2, 3.5, 1.0), (1.0, 1.0, 2.0), (0.5, 1.5, 0.75)):
        got_, err_ = emm_safe(lo_, hi_, fl_)
        want_ = mean_max(lo_, hi_, fl_) if hi_ > lo_ else max(lo_, fl_)
        seen_mm.append((lo_, hi_, fl_, round(got_, 4) if got_ is not None else err_))
        ok_mm = ok_mm and err_ is None and got_ is not None and abs(got_ - want_) < 1e-3
    check("expected_max_uniform (a random gap that never ends before a fixed fall) agrees with "
          "integrating the draw: below, at, inside and above the range, and a zero-width range "
          "does not raise", ok_mm, str(seen_mm))

    dec = ghost_score(available_at=avail17, declared_waits=5, **kwr)
    check("5 declared waits are COUNTED on the result, and it is a state of its own",
          dec.declared_waits == 5 and dec.has_declared_waits, f"{dec.declared_waits}")
    check("...it is an ESTIMATE, never a lower bound: lower_bound stays False",
          not dec.lower_bound and dec.lower_bound_why == "")
    check("...and it SAYS so in words a person can act on, with the count and the ghost's waits",
          "5 of the ghost's 16 waits" in dec.declared_why, dec.declared_why)
    check("the count only reports: the score is bit-for-bit what the same waits give without it",
          dec.score == real.score and dec.shown == real.shown)
    check("a clean round carries ZERO declared waits and no explanation (a pure measurement)",
          real.declared_waits == 0 and not real.has_declared_waits and real.declared_why == "")
    check("with no usable waits nothing was declared: the lower bound stands and the count is 0",
          ghost_score(declared_waits=5, **kwr).declared_waits == 0
          and ghost_score(declared_waits=5, **kwr).lower_bound)
    check("a drill that makes nobody wait never carries declared waits, whatever it is told",
          ghost_score(available_at=avail17, declared_waits=5,
                      **dict(kwr, has_waits=False)).declared_waits == 0)
    check("the count is CLAMPED to the shots whose wait matters (16 of 17; the first is "
          "charged to nobody): 99 -> 16, -4 -> 0",
          ghost_score(available_at=avail17, declared_waits=99, **kwr).declared_waits == 16
          and ghost_score(available_at=avail17, declared_waits=-4, **kwr).declared_waits == 0)
    check("a count that is not a whole number is ignored, never raised on: bool, float, string, None",
          all(ghost_score(available_at=avail17, declared_waits=v, **kwr).declared_waits == 0
              for v in (True, 3.0, "3", None)))
    over = ghost_score(available_at=avail17, declared_waits=3,
                       **dict(kwr, human_points=10_000))
    check("a round that is not scorable still carries its declared count (the screen may show why)",
          not over.scorable and over.declared_waits == 3)

    # --------------------------------------------------------------- 13 ------
    # Run behind its own crash guard. The mutation runner counts a suite that
    # never prints its summary as BROKEN, not as a catch (CLAUDE.md 2026-10-19),
    # and a mutated ghost.py can make anything in the new modules raise: a raise
    # here must be a FAILED CHECK, with the sections above still counted.
    try:
        section_13(STORM_POINTS, STORM_ELAPSED, STORM_PEAKS,
                   dict(points=REAL_POINTS, elapsed=REAL_ELAPSED, peaks=REAL_PEAKS,
                        mic=REAL_MIC, ceiling=REAL_CEILING))
    except Exception as exc:                             # noqa: BLE001 - the finding
        check("section 13 ran to its end without raising", False,
              f"{type(exc).__name__}: {exc}")

    # --------------------------------------------------------------- 14 ------
    # Behind its own crash guard, like 13 (a raise is a FAILED CHECK, not a dead suite).
    try:
        section_14()
    except Exception as exc:                             # noqa: BLE001 - the finding
        check("section 14 ran to its end without raising", False,
              f"{type(exc).__name__}: {exc}")

    print()
    if all(checks):
        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    return 1


def section_14() -> None:
    """THE CEILING OF THE SCALE, as a property over generated inputs (2026-10-23).

    `SCORE = 1000 x aim x speed` and the ghost is 1000, so what may the MODULE be
    handed, and what comes out? His stored animated-soldier record of 1146 (a speed
    share of 1.499) showed that nothing enforced it. Two cures, in two places:

      * where the waits are built (`Mode._ghost_wait_plan`, test_modes section 52):
        the speed share exceeded 1 because the app handed the ghost waits the round
        never held;
      * HERE, since his ruling of 2026-10-23 ("clamp: 1000 is a hard ceiling"): the
        speed SHARE is clamped at 1.0 inside `_arithmetic`, so NO input the module can
        take produces a score above 1000. The raw share survives in `speed_raw` and the
        clamp is recorded in `clamped`, because a clamp that fires means the MODEL is
        wrong (and the host logs it as a DEFECT; test_modes section 52 pins that).

    The first two checks are the property over generated inputs (no waits, every
    magazine, every ceiling; waits of every kind including ones longer than the whole
    round); the rest pin the clamp's own contract: it never alters an unclamped case,
    it keeps the raw figure, it keeps the headline equal to the factors beside it."""
    print("\n14. the ceiling of the scale: what the module can be handed, and the most it returns")
    tol = 1e-6
    ceilings = (("flat 10", constant_ceiling(10)), ("flat 1", constant_ceiling(1)),
                ("escalating 5", escalating_ceiling(5)))

    # -- no waits: every magazine, every ceiling, a grid of paces and of points --------
    scored = over = bad_clamp = clamp_fired = 0
    worst = 0.0
    offenders = []
    for n in range(2, 51):
        cad = (n - 1) * 0.25
        for cname, ceil in ceilings:
            gp = ghost_points(n, ceil)
            for ratio in (MIN_SPAN_RATIO, 0.95, 1.0, 1.1, 2.0, 10.0):
                for frac in (0.0, 0.5, 1.0):
                    q = ghost_score(human_points=int(frac * gp), human_elapsed=ratio * cad,
                                    shots=n, ceiling=ceil)
                    if not q.scorable:
                        continue                         # discarded at the floor's last ulp
                    scored += 1
                    worst = max(worst, q.score)
                    if q.score > SCORE_SCALE + tol:
                        over += 1
                        offenders.append((n, cname, ratio, frac, round(q.score, 2)))
                    # The clamp fires exactly when the human beat the cadence, never otherwise,
                    # and the raw share is the unclamped quotient (recoverable for the log).
                    fires = ratio < 1.0
                    clamp_fired += fires
                    if (bool(q.clamped) != fires or q.speed_raw is None
                            or abs(q.speed_raw - 1.0 / ratio) > 1e-9
                            or q.speed_share > 1.0 or (not fires and q.speed_share != q.speed_raw)):
                        bad_clamp += 1
                        offenders.append((n, cname, ratio, frac, "clamp", q.clamped, q.speed_raw))
    check(f"with no waits, over {scored} generated rounds (every magazine 2-50, three ceilings, paces from the "
          f"discard floor to ten times the ghost's, 0/50/100% of the points) NOTHING exceeds "
          f"{SCORE_SCALE:.0f} (highest seen {worst:.1f}) -- the 1099 band the discard floor used to leave "
          f"is gone -- and the clamp fires on exactly the {clamp_fired} rounds that beat the cadence, with "
          f"the raw share recoverable and an unclamped round's share left untouched",
          scored > 2300 and over == 0 and bad_clamp == 0 and clamp_fired > 600, str(offenders[:3]))

    # -- waits the round contained: the human is the ghost, or slower by a factor -------
    profiles = ([0.0] * 6, [0.0, 4.6, 0.0, 0.0, 4.6, 0.0], [0.0, 0.1, 0.1, 0.3, 2.5, 0.05],
                [0.0, 12.0, 12.0, 12.0, 12.0, 12.0])
    scored_w = over_w = exact_w = 0
    off_w = []
    for waits in profiles:
        for fps in (30.0, 60.0):
            av = available_from_waits(waits, fps)
            n = len(waits)
            gel = ghost_elapsed(n, av)[0]
            for s in (1.0, 1.0001, 1.5, 3.0, 10.0):
                for cname, ceil in ceilings:
                    gp = ghost_points(n, ceil)
                    q = ghost_score(human_points=gp, human_elapsed=s * gel, shots=n, ceiling=ceil,
                                    has_waits=True, available_at=av)
                    if not q.scorable:
                        off_w.append((waits[:3], fps, s, cname, "not scorable"))
                        continue
                    scored_w += 1
                    if q.score > SCORE_SCALE + tol:
                        over_w += 1
                        off_w.append((waits[:3], fps, s, cname, round(q.score, 3)))
                    if s == 1.0 and abs(q.score - SCORE_SCALE) < tol and q.clamped == ():
                        exact_w += 1                     # exactly 1000 WITHOUT the clamp's help
    check(f"with waits the round contained (a human s times slower than the ghost that waits the same waits, "
          f"s = 1 to 10, four wait profiles including one far longer than a cadence, two frame rates), "
          f"{scored_w} generated rounds never exceed {SCORE_SCALE:.0f}, and the human who IS the ghost reads "
          f"exactly {SCORE_SCALE:.0f} with the clamp NOT fired, in every one of the {exact_w} where s = 1",
          scored_w == 120 and over_w == 0 and exact_w == 24 and not off_w, str(off_w[:3]))

    # -- the stored record, as the module sees it: the clamp's reason to exist -------------
    # Waits longer than the human's whole round (the stored 1146 was exactly that) used to be
    # scored as handed: speed share 1.499, a score of 1146. The cure for the CAUSE is in the plan
    # that builds the waits (test_modes 52); this is the cure for the SYMPTOM, and it is loud
    # (the host logs the DEFECT) so it cannot hide the cause.
    q = ghost_score(human_points=130, human_elapsed=14.2876, shots=17, ceiling=constant_ceiling(10),
                    has_waits=True, available_at=available_from_waits([0.0] * 4 + [4.6] + [0.0] * 3
                                                                       + [4.6] + [0.0] * 3 + [4.6]
                                                                       + [0.0] * 3 + [4.6], 60.0))
    check("waits longer than the human's whole round (the stored 1146 case) are CLAMPED: the speed share "
          "is exactly 1.0, the score is aim alone (1000 x 130/170), at most 1000, the clamp is recorded, "
          "and the RAW share (above 1.4) is still there for the log",
          q.scorable and q.speed_share == 1.0 and q.score is not None
          and rel_close(q.score, SCORE_SCALE * 130 / 170, 1e-12) and q.score <= SCORE_SCALE
          and q.clamped == (ghost.CLAMPED_SPEED,) and q.speed_raw is not None and q.speed_raw > 1.4
          and q.shown == 765, f"{fx(q.speed_share, '.3f')} raw {fx(q.speed_raw, '.3f')}")
    check("... the clamp is on the FACTOR, not on the headline: score == 1000 x aim x speed still holds "
          "EXACTLY, so the two percentages on the screen can never disagree with the number above them",
          q.score == SCORE_SCALE * q.points_share * q.speed_share)

    # -- a float-noise overshoot is not a firing --------------------------------------------
    # A human who IS the ghost can read 1.0000000000000002 depending on how the two elapsed
    # times were summed. That must neither alter the score nor be reported as a model defect.
    n_ = 17
    nudged = ghost_score(human_points=170, human_elapsed=(n_ - 1) * 0.25 * (1.0 - 1e-13), shots=n_,
                         ceiling=constant_ceiling(10))
    check("a raw share above 1 by only float noise (1e-13) scores exactly 1000 and does NOT count as "
          "a clamp firing -- while 1e-6 over DOES",
          nudged.scorable and nudged.score == SCORE_SCALE and nudged.clamped == ()
          and ghost_score(human_points=170, human_elapsed=(n_ - 1) * 0.25 * (1.0 - 1e-6), shots=n_,
                          ceiling=constant_ceiling(10)).clamped == (ghost.CLAMPED_SPEED,))

    # -- THE SPACE, GENERATED: no input of any kind produces a score above 1000 -----------
    # Not picked cases. Every combination of magazine (2-50), ceiling, points (0..all, plus
    # the over-maximum ones that must be refused), human pace from 0.91x to 20x the cadence
    # AND from far below it (discarded) to far above, and waits from none to ones longer than
    # the whole round, under both a discarding and a non-discarding pace. Checked on `score`,
    # `shown`, `withheld_score` (the discarded figure the log keeps) and the two shares.
    import itertools
    paces = (1e-6, 0.3, 0.5, MIN_SPAN_RATIO, 0.95, 0.999999, 1.0, 1.0000001, 1.05, 2.0, 20.0)
    wait_profiles = (None, [0.0], [0.0, 0.0, 9.0], [0.0, 30.0], [0.0, 0.1, 0.1, 40.0, 0.0])
    space = bad_space = clamped_space = 0
    worst_space = 0.0
    off_space = []
    for n, (cname, ceil), pace, wp, frac in itertools.product(
            (2, 3, 5, 17, 33, 50), ceilings, paces, wait_profiles, (0.0, 0.3, 1.0, 1.01)):
        gp = ghost_points(n, ceil)
        waits = None
        if wp is not None:
            wl = (list(wp) + [0.0] * n)[:n]
            waits = available_from_waits(wl, 60.0)
        q = ghost_score(human_points=int(round(frac * gp)), human_elapsed=pace * (n - 1) * 0.25,
                        shots=n, ceiling=ceil, has_waits=waits is not None, available_at=waits)
        space += 1
        top = max([v for v in (q.score, q.withheld_score) if v is not None], default=0.0)
        worst_space = max(worst_space, top)
        clamped_space += bool(q.clamped)
        if (top > SCORE_SCALE + 1e-6 or any(v > 1.0 + 1e-9 for v in
                                            [x for x in (q.speed_share, q.points_share) if x is not None])
                or (q.shown is not None and q.shown > 1000)):
            bad_space += 1
            off_space.append((n, cname, pace, wp, frac, q.score, q.withheld_score))
    check(f"over {space} generated rounds (six magazines, three ceilings, eleven paces from a thousandth of the "
          f"cadence to twenty times it, five wait profiles including waits far longer than the round, four "
          f"point fractions including one OVER the maximum) no score, no withheld score, no whole number "
          f"and neither share ever exceeds 1000 / 1.0 (highest seen {worst_space:.1f}); the clamp fired "
          f"on {clamped_space} of them, so this is not a space the clamp never touches",
          space == 6 * 3 * 11 * 5 * 4 and bad_space == 0 and clamped_space > 300 and worst_space <= SCORE_SCALE,
          str(off_space[:3]))

    # -- the raw figure survives into the LOG while the card carries the clamped one -----------
    from lsa import bests as B14
    from lsa import scorecard as C14
    cc = C14.make_card(key="animated|m17|p1", human_points=130, human_elapsed=14.2876, shots=17,
                       efficiency=9.1, ceiling=constant_ceiling(10), has_waits=True,
                       book=B14.BestBook({}),
                       available_at=available_from_waits([0.0] * 4 + [4.6] + [0.0] * 3 + [4.6] + [0.0] * 3
                                                         + [4.6] + [0.0] * 3 + [4.6], 60.0))
    dtxt = C14.describe(cc, ", LEFT")
    check("a clamped round's card carries the CLAMPED figure (headline 765, `AIM 76%   SPEED 100%`) while "
          "describe() keeps the RAW share, says CLAMPED, and names the three things that can be wrong -- the "
          "screen and the record book get 765, the log gets the evidence",
          cc.gs.shown == 765 and C14.headline(cc) == "SCORE  765"
          and C14.factors_line(cc) == "AIM 76%   SPEED 100%"
          and "CLAMPED speed" in dtxt and f"raw speed share {cc.gs.speed_raw:.4f}" in dtxt
          and "GHOST_INTERVAL" in dtxt and "wait" in dtxt and "ceiling" in dtxt
          and cc.gs.speed_raw is not None and cc.gs.speed_raw > 1.4 and "1.4" not in C14.headline(cc),
          dtxt[:160])
    check("... and a clamped round STILL SETS A RECORD (it is sound and he shot it): against an empty "
          "book it is the new best, stored at the clamped 764.7 and never above 1000",
          cc.new_score and cc.gs.score is not None and cc.gs.score <= 1000.0
          and abs(C14.score_record(cc)["value"] - 1000.0 * 130 / 170) < 1e-9)

    # -- aim cannot be clamped, because it is already refused ---------------------------------
    over_max = ghost_score(human_points=171, human_elapsed=9.34, shots=17, ceiling=constant_ceiling(10))
    check("AIM above 1 never reaches a score: a human out-scoring the stated maximum is NOT SCORABLE "
          "with `ceiling_exceeded` set (the host logs it as a DEFECT), instead of an aim clamp that "
          "would hide a wrong ceiling behind a correct-looking number",
          not over_max.scorable and over_max.score is None and over_max.ceiling_exceeded
          and not ghost_score(human_points=170, human_elapsed=9.34, shots=17,
                              ceiling=constant_ceiling(10)).ceiling_exceeded)


def section_13(STORM_POINTS, STORM_ELAPSED, STORM_PEAKS, REAL) -> None:
    print("\n13. the round's score, the machine's bests, and the guard in front of them "
          "(lsa/bests.py, lsa/scorecard.py: pure, so tested here)")
    import copy
    import dataclasses
    import json
    import tempfile

    from lsa import bests as B
    from lsa import scorecard as C
    from lsa.settings import AppSettings

    KEY = B.best_key("static", 17, 1)
    VALID = {"value": 415.7, "points": 165, "seconds": 9.34, "shots": 17,
             "basis": "measured", "declared_waits": 0, "ghost_interval": GHOST_INTERVAL,
             "when": "t"}
    GOOD_EFF = {"value": 17.7, "points": 165, "seconds": 9.34, "shots": 17,
                "basis": "measured", "when": ""}
    EFF_HEALTHY = 165 / round(9.34, 1)          # efficiency is divided from the PRINTED seconds

    def card(book=None, key=KEY, **over):
        kw = dict(key=key, human_points=165, human_elapsed=9.34, shots=17,
                  efficiency=EFF_HEALTHY, ceiling=constant_ceiling(10),
                  peaks=HEALTHY_PEAKS, corroboration=HEALTHY_MIC, book=book)
        kw.update(over)
        return C.make_card(**kw)

    # -- the key, the schema and what a fresh install holds ------------------------
    check("a best is keyed per DRILL, per MAGAZINE size and per PLAYER count: all eight "
          "combinations are different keys",
          len({B.best_key(d, m, p) for d in ("static", "reaction")
               for m in (5, 17) for p in (1, 2)}) == 8)
    odd = (("a|b c", 17, 1), ("", 0, 9), (None, "x", None), ("x" * 100, 17, True), ("é", -4, 2))
    check("whatever it is handed, best_key returns a key its own validator accepts",
          all(B.normalise_bests({B.best_key(*a): {"efficiency": GOOD_EFF}}) for a in odd))
    check("a fresh install holds NO best: the dataclass default and its JSON default are both empty",
          AppSettings().bests == {}
          and dataclasses.asdict(AppSettings())["bests"] == {}
          and AppSettings().bests is not AppSettings().bests)

    # -- a corrupt or hand-edited value never crashes the loader -------------------
    nan = float("nan")
    junk = [None, 7, "x", [], [1, 2], True, 3.5,
            {KEY: "str"}, {KEY: 5}, {KEY: []}, {KEY: {"score": 7}}, {KEY: {"score": {}}},
            {KEY: {"score": dict(VALID, value="416")}},
            {KEY: {"score": dict(VALID, value=nan)}},
            {KEY: {"score": dict(VALID, value=float("inf"))}},
            {KEY: {"score": dict(VALID, value=-1)}},
            {KEY: {"score": dict(VALID, value=0)}},
            {KEY: {"score": dict(VALID, value=True)}},
            {KEY: {"score": dict(VALID, shots=0)}},
            {KEY: {"score": dict(VALID, points=-5)}},
            {KEY: {"score": dict(VALID, basis="bogus")}},
            {KEY: {"score": dict(VALID, basis=None)}},
            {KEY: {"score": {k: v for k, v in VALID.items() if k != "seconds"}}},
            {KEY: {"efficiency": dict(VALID, points=1.5)}},
            {"not a key": {"score": VALID}}, {"static|m0|p1": {"score": VALID}},
            {"static|m17|p3": {"score": VALID}}, {7: {"score": VALID}}]
    sane = []
    for j in junk:
        try:
            sane.append(B.normalise_bests(j) == {})
        except Exception as exc:                       # noqa: BLE001 - a raise is the finding
            sane.append(False)
            print(f"      normalise_bests raised on {str(j)[:60]!r}: {exc!r}")
    check(f"{len(junk)} kinds of corrupt or hand-edited value are each dropped without a raise "
          f"(non-dicts, strings, NaN, inf, negatives, booleans, a missing ingredient, a basis "
          f"nobody defined, bad keys)", all(sane), f"{sane.count(False)} did not")
    # The lower-bound basis is a KNOWN basis since 2026-10-22 (his ruling: a floor is
    # recorded), so a stored floor must be KEPT, with its mark, by the loader, and must
    # survive the round trip through the settings file. It used to be in the junk list
    # above; leaving it there would have made the loader's right answer a failure.
    floor_in = {KEY: {"score": dict(VALID, basis="lower_bound"),
                      "efficiency": dict(GOOD_EFF, basis="lower_bound")}}
    floor_out = B.normalise_bests(copy.deepcopy(floor_in))
    check("a stored floor is KEPT by the loader, still marked lower_bound on both records "
          "(a book that dropped it would silently forget a record the screen announced)",
          rv(rv(rv(floor_out, KEY), "score"), "basis") == "lower_bound"
          and rv(rv(rv(floor_out, KEY), "efficiency"), "basis") == "lower_bound"
          and B.BASIS_LOWER == "lower_bound" and B.BASIS_LOWER in B.BASES
          and set(B.BASES) == {"measured", "declared", "lower_bound"}, str(floor_out)[:120])
    check("... and the two bases that existed before are untouched, so no saved file needs "
          "migrating",
          B.BASIS_MEASURED == "measured" and B.BASIS_DECLARED == "declared"
          and rv(rv(rv(B.normalise_bests({KEY: {"score": VALID}}), KEY), "score"), "basis")
          == "measured")
    kept = B.normalise_bests({KEY: {"score": VALID, "efficiency": dict(VALID, value=-1)}})
    check("a bad record costs that record and not its neighbour: the score survives, the "
          "invalid efficiency does not",
          set(kept) == {KEY} and set(kept[KEY]) == {"score"} and kept[KEY]["score"]["value"] == 415.7)
    check("the dict is capped, so a runaway edit cannot bloat the settings file",
          len(B.normalise_bests({B.best_key(f"d{i}", 17, 1): {"efficiency": GOOD_EFF}
                                 for i in range(B.MAX_ENTRIES + 50)})) == B.MAX_ENTRIES)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "settings.local.json"
        f.write_text(json.dumps({"volume": 0.3, "bests": "garbage"}), encoding="utf-8")
        a = AppSettings.load(f)
        check("settings with a corrupt `bests` still load: the bests are empty and the rest is intact",
              a.bests == {} and a.volume == 0.3)
        f.write_text('{"volume": 0.4, "bests": {"static|m17|p1": {"score": {"value": NaN, '
                     '"points": 1, "seconds": 1, "shots": 2, "basis": "measured"}}}}',
                     encoding="utf-8")
        a = AppSettings.load(f)
        check("... including a NaN a hand edit left in the JSON", a.bests == {} and a.volume == 0.4)
        f.write_text(json.dumps({"bests": {KEY: {"score": VALID}}}), encoding="utf-8")
        a = AppSettings.load(f)
        saved = a.save(f)
        b2 = AppSettings.load(f)
        check("a valid best survives a save and a reload, through the atomic writer",
              saved and b2.bests == a.bests
              and rv(rv(rv(b2.bests, KEY), "score"), "value") == 415.7
              and not (Path(td) / "settings.local.json.tmp").exists())

    # -- the legitimate round sets both, and only on an improvement ----------------
    book = B.BestBook({})
    c1 = card(book)
    check("(premise) his best round reads 416, measured, and may set a record",
          c1.gs.shown == 416 and c1.basis == "measured" and c1.gs.scorable
          and c1.new_score and c1.new_efficiency, f"{c1.gs.shown} {c1.basis}")
    check("nothing is written until the round is SETTLED: building the card only reads the book",
          book.data == {} and not book.take_changed())
    s1 = C.settle([c1], book, "T1")[0]
    stored = book.best_score(KEY)
    check("settling a legitimate round stores BOTH bests, with the raw ingredients and the basis",
          stored is not None and book.best_efficiency(KEY) is not None
          and close(rv(stored), c1.gs.score) and rv(stored, "points") == 165
          and rv(stored, "shots") == 17 and close(rv(stored, "seconds"), 9.34)
          and rv(stored, "basis") == "measured" and rv(stored, "declared_waits") == 0
          and rv(stored, "ghost_points") == 170 and close(rv(stored, "ghost_seconds"), 4.0)
          and rv(stored, "ghost_interval") == GHOST_INTERVAL and rv(stored, "when") == "T1"
          and close(rv(book.best_efficiency(KEY)), EFF_HEALTHY)
          and rv(book.best_efficiency(KEY), "points") == 165, str(stored))
    check("... the file is to be written ONCE: take_changed is True the first time, False the next",
          book.take_changed() is True and book.take_changed() is False)
    check("... and the screen reports it as NEW",
          s1.new_score and s1.new_efficiency and stored is not None
          and rv(s1.best_score) == rv(stored))
    snap = copy.deepcopy(book.data)
    s2 = C.settle([card(book)], book, "T2")[0]
    check("the SAME round again sets nothing: not new, nothing written, the book unchanged",
          not s2.new_score and not s2.new_efficiency and not book.take_changed()
          and book.data == snap)
    s3 = C.settle([card(book, human_points=120)], book, "T3")[0]
    check("a WORSE round sets nothing, and the best on the screen is still the old one",
          not s3.new_score and not s3.new_efficiency and not book.take_changed()
          and book.data == snap and stored is not None
          and rv(s3.best_score) == rv(stored))
    s4 = C.settle([card(book, human_elapsed=8.0, efficiency=165 / 8.0)], book, "T4")[0]
    check("a BETTER round replaces both, once",
          s4.new_score and s4.new_efficiency and book.take_changed() and stored is not None
          and rv(book.best_score(KEY)) is not None and rv(book.best_score(KEY)) > rv(stored)
          and close(rv(book.best_efficiency(KEY)), 165 / 8.0)
          and close(rv(book.best_score(KEY), "seconds"), 8.0))
    only_score = B.BestBook({KEY: {"efficiency": dict(GOOD_EFF, value=999.0)}})
    c_ = card(only_score)
    check("the two bests are judged separately: a round that beats the stored score but not the "
          "stored efficiency sets only the score",
          c_.new_score and not c_.new_efficiency)
    check("a book can only be RAISED: offering a lower record changes nothing",
          B.BestBook(copy.deepcopy(book.data)).offer(KEY, C.score_record(c1, "x"), None) is False
          and book.offer("not a key", C.score_record(c1, "x"), None) is False
          and "not a key" not in book.data)
    shared: dict = {}
    sb = B.BestBook(shared)
    C.settle([card(sb)], sb)
    check("the book SHARES the dict it was given (that is how settings.bests is what gets saved)",
          sb.data is shared and KEY in shared)
    check("a round that earned nothing (every shot a miss) is never a record, even on an empty book",
          not card(B.BestBook({}), human_points=0, efficiency=0.0).new_score
          and not card(B.BestBook({}), human_points=0, efficiency=0.0).new_efficiency)

    # -- THE STORM: his real round, his real figures ------------------------------
    storm_kw = dict(human_points=STORM_POINTS, human_elapsed=STORM_ELAPSED,
                    efficiency=STORM_POINTS / 2.7, peaks=STORM_PEAKS,
                    corroboration=Corroboration(True, 0, 17))        # deaf for all 17, AS LOGGED
    real_best = {KEY: {"score": dict(VALID, value=416.0),
                       "efficiency": dict(GOOD_EFF, value=17.67)}}
    for label, bk in (("an empty book", B.BestBook({})),
                      ("his real bests, 416 and 17.67", B.BestBook(copy.deepcopy(real_best)))):
        before = copy.deepcopy(bk.data)
        cs = card(bk, **storm_kw)
        out = C.settle([cs], bk, "S")[0]
        check(f"the phantom storm (61 points, 17 shots, 2.70 s, peaks 13-25) against {label}: "
              f"DISCARDED WHOLE -- no score and no efficiency on the card, it would have "
              f"BEATEN the best (that is what the discard is for), and it sets NEITHER record",
              cs.discarded and cs.gs.shown is None and cs.gs.score is None
              and cs.efficiency is None and close(cs.withheld_efficiency, 22.59, 0.01)
              and cs.gs.withheld_score is not None and round(cs.gs.withheld_score) == 359
              and cs.would_beat
              and not cs.gs.scorable and not out.new_score and not out.new_efficiency
              and not bk.take_changed() and bk.data == before, f"{cs.gs.withheld_score}")
    cs = card(B.BestBook(copy.deepcopy(real_best)), **storm_kw)
    cs_lines = C.lines(cs)
    check("... and the SCREEN says so plainly and shows nothing the app will not honour: four lines, "
          "headed ROUND DISCARDED, the only numbers in it the shots and the seconds he can check "
          "himself (17 and 2.7), no SCORE, no AIM, no SPEED, no BEST, no 359, no 22.59",
          len(cs_lines) == C.SCORE_LINES == 4 and cs_lines[0] == "ROUND DISCARDED"
          and cs_lines[-1] == "No score, nothing saved."
          and re.findall(r"\d+(?:\.\d+)?", " ".join(cs_lines)) == ["17", "2.7"]
          and not any(w in " ".join(cs_lines) for w in ("SCORE", "AIM", "SPEED", "BEST", "359", "532", "22.59")),
          str(cs_lines))
    check("... in HIS terms, not the detector's: it says the shots came faster than a pistol can "
          "fire, and never mentions a median, a peak, a threshold or a microphone",
          "faster" in " ".join(cs_lines) and "pistol" in " ".join(cs_lines)
          and not re.search(r"median|peak|threshold|microphone|mic\b|noise", " ".join(cs_lines), re.I),
          " ".join(cs_lines))
    check("the card's own note is the short log caveat, not a refusal: DISCARDED, and under the "
          "note budget", cs.note == "DISCARDED: faster than a pistol" and len(cs.note) <= C.MAX_NOTE_CHARS,
          cs.note)

    # The prior score is 300, not his 416: the withheld score is the CLAMPED one (359 --
    # it was 532 before the 1000 ceiling), which no longer beats 416. The check must
    # still be decided by the WITHHELD score alone, so the prior sits between 0 and 359
    # and the efficiency (22.59 under 99) cannot rescue it.
    bk_hi = B.BestBook({KEY: {"score": dict(VALID, value=300.0),
                              "efficiency": dict(GOOD_EFF, value=99.0)}})
    check("would_beat reads the WITHHELD score of a discarded round: the storm beats a stored 300 "
          "on score (359) and not the efficiency (22.59 under 99), and the card still says it "
          "WOULD have beaten the best -- the fact the log exists to state",
          card(bk_hi, **storm_kw).would_beat and card(bk_hi, **storm_kw).discarded)

    # -- HIS REAL ROUND AT THE CARD: shown, and therefore recorded -------------------------
    # This is the round that put `SCORE 256` over `BEST SCORE SO FAR 212` and recorded
    # nothing. Both halves of his rule are asserted on it: what the screen shows IS what the
    # book receives.
    RKEY = B.best_key("ball", 17, 1)
    real_kw = dict(human_points=REAL["points"], human_elapsed=REAL["elapsed"],
                   efficiency=REAL["points"] / round(REAL["elapsed"], 1), peaks=REAL["peaks"],
                   corroboration=REAL["mic"], ceiling=REAL["ceiling"], key=RKEY)
    old_best = {RKEY: {"score": dict(VALID, value=212.0), "efficiency": dict(GOOD_EFF, value=3.0)}}
    bk_r = B.BestBook(copy.deepcopy(old_best))
    cr = card(bk_r, **real_kw)
    outr = C.settle([cr], bk_r, "R")[0]
    lines_r = C.lines(outr)
    check("HIS REAL ROUND (70 of 75 points in 14.57 s, median peak 28) against his old best of 212: "
          "it is scored 256, it beats the best, and it IS recorded -- the screen says "
          "NEW BEST SCORE RECORDED where it said BEST SCORE SO FAR 212",
          cr.gs.shown == 256 and not cr.discarded and cr.gs.scorable and cr.would_beat
          and outr.new_score and lines_r[0] == "SCORE  256"
          and lines_r[2] == "NEW BEST SCORE RECORDED" and lines_r[2] != "BEST SCORE SO FAR  212",
          str(lines_r))
    check("... and what the screen shows is what the book received: the stored best IS the shown "
          "256 (same rounding), the efficiency best too, and the file is to be written once",
          bk_r.best_score(RKEY) is not None and C.shown(rv(bk_r.best_score(RKEY))) == 256
          and close(rv(bk_r.best_score(RKEY)), cr.gs.score)
          and outr.new_efficiency and close(rv(bk_r.best_efficiency(RKEY)), 70 / 14.6)
          and lines_r[3] == "NEW BEST EFFICIENCY RECORDED"
          and bk_r.take_changed() is True and bk_r.take_changed() is False)
    check("... and its dim median (28) is only a note in the log: the log line says the round was "
          "recorded AND says what was noted, and never says REFUSED",
          "median shot brightness was 28" in C.describe(outr, ", LEFT")
          and "REFUSED" not in C.describe(outr, ", LEFT")
          and "new score best" in C.describe(outr, ", LEFT"))

    # -- THE RULE ITSELF, over a grid of rounds: a shown score is a recorded score ---------
    # For every round that has points, against an EMPTY book (so any scored round beats it):
    # the screen shows a number if and only if the book received that number. A round of
    # zero points is left out on purpose: SCORE 0 is not a best and nobody is told it is.
    # The lower-bound state is IN the grid since 2026-10-22: it used to be left out because
    # it was the one place a number was shown and not recorded. His ruling removed the
    # exception, so the rule is now universal and the grid is where it is proved to be.
    grid = (("a healthy round", dict()),
            ("a lower-bound round (the drill makes you wait, the waits not supplied)",
             dict(has_waits=True)),
            ("his real round", real_kw),
            ("the phantom storm", storm_kw),
            ("faint shots at a human pace", dict(peaks=STORM_PEAKS, corroboration=None)),
            ("a ready microphone that heard none, at a human pace",
             dict(corroboration=Corroboration(True, 0, 0))),
            ("bright shots at the storm's pace", dict(human_elapsed=STORM_ELAPSED,
                                                       efficiency=165 / 2.7, corroboration=None)),
            ("seventeen shots in one instant", dict(human_elapsed=0.0, efficiency=None)))
    contradictions = []
    heads = []
    for label, over in grid:
        bk_g = B.BestBook({})
        try:
            cg = card(bk_g, **over)
        except Exception as exc:                     # noqa: BLE001 - a raise is the finding
            # e.g. "seventeen shots in one instant" with the zero-span guard deleted
            # (mutate_ghost fault 25). Recorded here, by name, and the loop goes on.
            contradictions.append(f"{label}: RAISED {type(exc).__name__}: {exc}")
            continue
        og = C.settle([cg], bk_g, "G")[0]
        head = C.lines(og)[0]
        heads.append(head)
        shows_number = re.fullmatch(r"SCORE  \d+", head) is not None
        stored_g = bk_g.best_score(cg.key)
        recorded = stored_g is not None and head == f"SCORE  {C.shown(rv(stored_g))}"
        if shows_number != recorded:
            contradictions.append(f"{label}: shows {head!r}, book holds "
                                  f"{None if stored_g is None else C.shown(rv(stored_g))}")
    check("EVERY round in the grid obeys his rule: a score is on the screen if and only if the "
          "book received that same score (the storm and the instant are discarded and show "
          "none; the rest, a lower-bound round included, are shown and recorded)",
          not contradictions and len(heads) == len(grid), "; ".join(contradictions))
    check("... and the grid covers both kinds of round, or the check above is empty of one half",
          sum(1 for h_ in heads if h_.startswith("SCORE  ") and h_ != "SCORE  --") >= 5
          and sum(1 for h_ in heads if h_ == "ROUND DISCARDED") >= 3)

    # The note's budget, over EVERY state the module can build a note for (the refusal
    # note this loop was written for is gone, and with it the "+N more" counting).
    import types

    def ns(**kw):
        base = dict(scorable=True, discarded=False, lower_bound=False, declared_waits=0,
                    record=types.SimpleNamespace(fit=True, reasons=[]))
        base.update(kw)
        return types.SimpleNamespace(**base)

    longest, worst = 0, ""
    for gs_kw in (dict(scorable=False), dict(lower_bound=True), dict(declared_waits=1),
                  dict(declared_waits=17), dict(),
                  dict(scorable=False, discarded=True,
                       record=types.SimpleNamespace(fit=False, reasons=[Unfit.TOO_FAST])),
                  dict(scorable=False, discarded=True,
                       record=types.SimpleNamespace(fit=False, reasons=[]))):
        for shots_, ceil_ in ((17, True), (1, True), (17, False)):
            t_ = C._note(ns(**gs_kw), shots=shots_, ceiling_known=ceil_)
            if len(t_) > longest:
                longest, worst = len(t_), t_
    print(f"         longest note the module can build: {longest} characters {worst!r}")
    check(f"...so no note of any kind is longer than {C.MAX_NOTE_CHARS} characters",
          longest <= C.MAX_NOTE_CHARS, f"{longest}: {worst!r}")
    check("...and the budget is not slack: the longest note IS within 6 characters of it, so "
          "the constant cannot drift upward unnoticed",
          longest >= C.MAX_NOTE_CHARS - 6, f"{longest} vs {C.MAX_NOTE_CHARS}")
    check("the refusal's constants are GONE with the refusal they served (nothing may read a "
          "'NOT A BEST' phrase table that no code path can reach)",
          not any(hasattr(C, n) for n in ("_SHORT", "MAX_CAUSES_SHOWN", "NOT_SAVED",
                                          "NOT_SAVED_FALLBACK")))
    check("a discarded round's note names its reason, and a reason with no phrase of its own "
          "falls back rather than raising",
          C._note(ns(scorable=False, discarded=True,
                     record=types.SimpleNamespace(fit=False, reasons=[Unfit.TOO_FAST])),
                  shots=17, ceiling_known=True) == "DISCARDED: faster than a pistol"
          and C._note(ns(scorable=False, discarded=True,
                         record=types.SimpleNamespace(fit=False, reasons=[])),
                      shots=17, ceiling_known=True) == "DISCARDED: " + C.DISCARD_FALLBACK)
    for label, kw, want_cautions, discarded in (
            ("faint shots only (storm peaks, a human pace)",
             dict(human_points=165, human_elapsed=9.34, efficiency=EFF_HEALTHY, peaks=STORM_PEAKS,
                  corroboration=None), (Unfit.PEAK_TOO_LOW,), False),
            ("a superhuman span only (laser-bright peaks, the storm's 2.70 s)",
             dict(human_points=165, human_elapsed=STORM_ELAPSED, efficiency=165 / 2.7,
                  peaks=HEALTHY_PEAKS, corroboration=None), (), True),
            ("an unconfirmed majority only (bright, human pace, the microphone heard none)",
             dict(corroboration=Corroboration(True, 0, 0)), (Unfit.MIC_UNCORROBORATED,), False)):
        bk = B.BestBook({})
        cg = card(bk, **kw)
        out = C.settle([cg], bk)[0]
        if discarded:
            ok = (cg.discarded and cg.gs.record.reasons == (Unfit.TOO_FAST,) and not cg.gs.scorable
                  and not out.new_score and not out.new_efficiency and bk.data == {}
                  and not bk.take_changed())
        else:
            ok = (not cg.discarded and cg.gs.record.reasons == () and cg.gs.scorable
                  and cg.gs.record.cautions == want_cautions
                  and out.new_score and out.new_efficiency
                  and set(bk.data.get(cg.key, {})) == {"score", "efficiency"}
                  and bk.take_changed() is True)
        check(f"{label}: " + ("DISCARDED by the span, and BOTH bests go through the same "
                              "decision and neither is written" if discarded else
                              "a CAUTION and nothing more: scored, shown and RECORDED (both bests)"),
              ok, f"{cg.gs.record.reasons} / {cg.gs.record.cautions}")
    check("(the control) take the span away and the storm's own figures DO set both records "
          "(so the discard above is the span's doing and nothing else)",
          card(B.BestBook({}), peaks=HEALTHY_PEAKS, human_elapsed=9.34).gs.scorable
          and card(B.BestBook({}), **dict(storm_kw, human_elapsed=9.34, peaks=HEALTHY_PEAKS)).new_score)

    def card_or_none(label, **over):
        """A card built for a DEGENERATE round (one shot, a zero span), or None after
        recording the raise as a failed check named "<label> does not raise". The
        zero-span guard is what mutate_ghost fault 25 deletes; without this the card
        is never built, and every check below that reads it dies on None instead of
        failing by name."""
        try:
            return card(B.BestBook({}), **over)
        except Exception as exc:                     # noqa: BLE001 - a raise is the finding
            check(f"{label} does not raise", False, f"{type(exc).__name__}: {exc}")
            return None

    # -- the three states of the score --------------------------------------------
    book_lb = B.BestBook({})
    clb = card(book_lb, has_waits=True)                  # waits not supplied
    olb = C.settle([clb], book_lb, "LB")[0]
    stored_lb_s, stored_lb_e = book_lb.best_score(KEY), book_lb.best_efficiency(KEY)
    # HIS RULING, 2026-10-22: "if the app shows the score on screen, then this is the score;
    # then the app has to honor that score and record it as a record". A lower-bound round
    # used to be the one scored-and-refused state; the three checks that pinned the
    # refusal ("sets no record", and two mutations of it) are replaced by these, which pin
    # the opposite and cannot be passed by the old behaviour.
    check("a LOWER-BOUND round SETS a record of both kinds (his ruling, 2026-10-22): the card "
          "says so, the settled card announces both, both are in the book, and the file is to "
          "be written once",
          clb.basis == "lower_bound" and clb.gs.lower_bound and clb.gs.scorable
          and clb.new_score and clb.new_efficiency and olb.new_score and olb.new_efficiency
          and stored_lb_s is not None and stored_lb_e is not None
          and C.lines(olb)[2] == "NEW BEST SCORE RECORDED"
          and C.lines(olb)[3] == "NEW BEST EFFICIENCY RECORDED"
          and book_lb.take_changed() is True and book_lb.take_changed() is False,
          f"{clb.basis} new_score={clb.new_score} new_efficiency={clb.new_efficiency} "
          f"stored={stored_lb_s is not None}/{stored_lb_e is not None}")
    check("... and it is stored MARKED as a floor, never as a measured score: basis lower_bound on "
          "both records, the value the very score the screen showed, the raw ingredients beside "
          "it, and the mark survives a reload through the validator",
          stored_lb_s is not None and rv(stored_lb_s, "basis") == "lower_bound"
          and rv(stored_lb_e, "basis") == "lower_bound"
          and close(rv(stored_lb_s), clb.gs.score) and C.shown(rv(stored_lb_s)) == clb.gs.shown
          and rv(stored_lb_s, "points") == 165 and rv(stored_lb_s, "shots") == 17
          and close(rv(stored_lb_e), EFF_HEALTHY)
          and rv(rv(B.normalise_bests(copy.deepcopy(book_lb.data)).get(KEY), "score"), "basis")
          == "lower_bound", str(stored_lb_s))
    # A later ORDINARY round meets the stored floor and is compared by value alone.
    snap_lb = copy.deepcopy(book_lb.data)
    tie_lb = C.settle([card(book_lb)], book_lb, "LB2")[0]
    worse_lb = C.settle([card(book_lb, human_points=120)], book_lb, "LB3")[0]
    unchanged_lb = book_lb.data == snap_lb and not book_lb.take_changed()
    better_lb = C.settle([card(book_lb, human_elapsed=8.0, efficiency=165 / 8.0)], book_lb, "LB4")[0]
    check("a later ORDINARY round compares against a stored floor by value alone: an equal or a "
          "worse one changes nothing (the floor and its mark stay, nothing is written), a better "
          "one replaces it and the record then says measured",
          not tie_lb.new_score and not tie_lb.new_efficiency
          and not worse_lb.new_score and not worse_lb.new_efficiency and unchanged_lb
          and better_lb.new_score and better_lb.new_efficiency and book_lb.take_changed() is True
          and rv(book_lb.best_score(KEY), "basis") == "measured"
          and rv(book_lb.best_efficiency(KEY), "basis") == "measured"
          and rv(book_lb.best_score(KEY)) is not None and stored_lb_s is not None
          and rv(book_lb.best_score(KEY)) > rv(stored_lb_s))
    # ... and the other way round: a floor offered to a stored MEASURED best.
    book_mf = B.BestBook({})
    C.settle([card(book_mf)], book_mf, "MF1")
    book_mf.take_changed()
    snap_mf = copy.deepcopy(book_mf.data)
    flo_worse = C.settle([card(book_mf, has_waits=True, human_points=120)], book_mf, "MF2")[0]
    kept_mf = book_mf.data == snap_mf and not book_mf.take_changed()
    flo_better = C.settle([card(book_mf, has_waits=True, human_elapsed=8.0,
                                efficiency=165 / 8.0)], book_mf, "MF3")[0]
    check("a floor meets a stored MEASURED best the same way: a worse floor changes nothing, a "
          "better one replaces it and the record then says lower_bound (its true value was at "
          "least that, so it is a record)",
          not flo_worse.new_score and not flo_worse.new_efficiency and kept_mf
          and flo_better.new_score and flo_better.new_efficiency
          and rv(book_mf.best_score(KEY), "basis") == "lower_bound"
          and rv(book_mf.best_efficiency(KEY), "basis") == "lower_bound"
          and rv(book_mf.best_score(KEY)) is not None
          and rv(rv(snap_mf.get(KEY), "score")) is not None
          and rv(book_mf.best_score(KEY)) > rv(rv(snap_mf.get(KEY), "score")))
    # A FLOOR CANNOT BECOME AN UNBEATABLE RECORD: the same human round, scored once its waits
    # ARE known, is never below its own floor, so measuring it properly always at least
    # ties it, and with a real wait strictly replaces it. (The module-level proof of "never
    # above the true score, for any wait" is in section 6.)
    wk = dict(human_points=119, human_elapsed=16 * (2.35 + 0.55), shots=17,
              efficiency=119 / 46.4, has_waits=True)
    book_u = B.BestBook({})
    C.settle([card(book_u, **wk)], book_u, "U1")
    floor_u = book_u.best_score(KEY)
    avail_u = available_from_waits([0.0] + [2.35] * 16, 60.0)
    again_u = (C.settle([card(book_u, available_at=avail_u, **wk)], book_u, "U2")[0]
               if avail_u is not None else None)
    check("a stored floor is replaced by the SAME round once its waits are known (so a floor can "
          "never be an unbeatable record: measuring the round properly can only raise it)",
          floor_u is not None and rv(floor_u, "basis") == "lower_bound" and again_u is not None
          and again_u.basis == "measured" and again_u.new_score
          and rv(book_u.best_score(KEY)) is not None and rv(floor_u) is not None
          and rv(book_u.best_score(KEY)) > rv(floor_u)
          and rv(book_u.best_score(KEY), "basis") == "measured")
    # 2026-10-22, his ruling: AT LEAST leaves the SCREEN, the same treatment as ABOUT. This check
    # used to assert it was drawn ("reads as an AT LEAST figure everywhere it is shown"); it now
    # asserts it is ABSENT from every drawn line AND, in the same check, that the distinction
    # survives in the data and the log: the basis, the lower_bound flag and its reason, the
    # card's caveat and the full `AT LEAST: <why>` sentence in describe(). Pairing the two is
    # what stops a later change dropping the floor from what is known as well as from what is shown.
    desc_lb = C.describe(olb, ", LEFT")
    check("... and it is DRAWN like a measured one (no AT LEAST in the headline, the speed or any "
          "line of the block) but is still a LOWER BOUND in the data: basis, flag, reason, caveat "
          "and the AT LEAST sentence in the log",
          "AT LEAST" not in C.headline(clb) and "AT LEAST" not in C.factors_line(clb)
          and "AT LEAST" not in " ".join(C.lines(clb))
          and C.headline(clb) == f"SCORE  {clb.gs.shown}"
          and clb.gs.speed_share is not None
          and C.factors_line(clb).endswith(f"SPEED {clb.gs.speed_share * 100:.0f}%")
          and clb.basis == C.BASIS_LOWER == "lower_bound" and C.basis_of(clb.gs) == C.BASIS_LOWER
          and C.BASIS_LOWER == B.BASIS_LOWER
          and clb.gs.lower_bound and bool(clb.gs.lower_bound_why)
          and clb.note.startswith("AT LEAST")
          and "(lower_bound)" in desc_lb and ("AT LEAST: " + clb.gs.lower_bound_why) in desc_lb,
          f"{C.lines(clb)} | {desc_lb[:240]}")
    check("the log says the stored score is a floor: a lower-bound round's line names it as one "
          "('stored as a floor') beside the new-best verdict, says nothing of a refusal, and the "
          "efficiency (exact whatever the basis) carries no such remark",
          "record: new score best (stored as a floor" in desc_lb
          and "new efficiency best" in desc_lb and "new efficiency best (stored" not in desc_lb
          and "REFUSED" not in desc_lb and "cannot set a record" not in desc_lb, desc_lb[:300])
    avail17 = available_from_waits([0.0] + [2.35] * 16, 60.0)
    kd = dict(human_points=17 * 7, human_elapsed=16 * (2.35 + 0.55), efficiency=17 * 7 / 46.4,
              has_waits=True, available_at=avail17, declared_waits=3)
    book_d = B.BestBook({})
    cd = card(book_d, **kd)
    od = C.settle([cd], book_d, "D")[0]
    check("a DECLARED-waits round MAY set a record (six drills are in this state whenever the "
          "shooter misses), and the state is stored with it",
          cd.basis == "declared" and cd.gs.scorable and od.new_score
          and rv(book_d.best_score(KEY), "basis") == "declared"
          and rv(book_d.best_score(KEY), "declared_waits") == 3
          and rv(book_d.best_efficiency(KEY), "basis") == "declared")
    # 2026-10-22, his ruling: ABOUT leaves the SCREEN. These two checks used to assert it was
    # drawn; they now assert it is ABSENT from every drawn line AND, in the same check, that
    # the distinction survives in the data and the log -- so a later change cannot quietly
    # drop the estimate from what is known as well as from what is shown.
    desc_d = C.describe(od, ", LEFT")
    check("... a declared round is DRAWN like a measured one (no ABOUT in the headline, the speed or "
          "any line of the block) but is still DECLARED in the data: basis, the caveat that names "
          "the count, and the full ESTIMATE sentence in the log",
          C.headline(od) == f"SCORE  {od.gs.shown}" and od.gs.speed_share is not None
          and C.factors_line(od).endswith(f"SPEED {od.gs.speed_share * 100:.0f}%")
          and "ABOUT" not in " ".join(C.lines(od)) and "AT LEAST" not in " ".join(C.lines(od))
          and od.basis == "declared" and C.basis_of(od.gs) == "declared"
          and od.note == "ESTIMATE: 3 waits assumed"
          and "(declared)" in desc_d and "ESTIMATE: 3 of the ghost's " in desc_d and "waits for a target to appear" in desc_d
          and "not measured from your round" in desc_d, f"{C.lines(od)} | {desc_d[:200]}")
    check("... and a best set under the estimate is stored as one (so the two are never silently "
          "compared by bests.py) but is not LABELLED on the screen",
          "ABOUT" not in C.best_line(od) and "ABOUT" not in C.efficiency_best_line(od)
          and rv(book_d.best_score(KEY), "basis") == "declared"
          and rv(book_d.best_efficiency(KEY), "basis") == "declared")
    cm = card(B.BestBook({}))
    check("a measured round carries no qualifier anywhere",
          C.headline(cm) == "SCORE  416" and C.factors_line(cm) == "AIM 97%   SPEED 43%"
          and "ABOUT" not in " ".join(C.lines(cm)) and "AT LEAST" not in " ".join(C.lines(cm)))
    # `cn` is built BEFORE the check that reads it. It was built after, when the ABOUT
    # inversions were edited in on 2026-10-22, so this section died on an
    # UnboundLocalError and every check below it stopped running.
    cn = card_or_none("a one-shot zero-span round's card", shots=1, human_elapsed=0.0,
                      efficiency=None)
    check("the headline no longer carries the ghost's 1000 in ANY state (his ruling, 2026-10-22), "
          "and no qualifier is drawn for measured / estimated / a floor (the floor's is in the "
          "data: basis lower_bound)",
          all("1000" not in C.headline(x) for x in (None, cm, clb, cd, cn))
          and C.headline(clb) == f"SCORE  {clb.gs.shown}" and clb.basis == C.BASIS_LOWER
          and C.headline(cd) == f"SCORE  {cd.gs.shown}" and C.headline(None) == "SCORE  --")
    check("a round with no score reads SCORE -- and says why, and sets no record",
          cn is not None and not cn.gs.scorable and C.headline(cn) == "SCORE  --"
          and not cn.new_score and not cn.new_efficiency and cn.note.startswith("NO SCORE"))
    check("every state prints EXACTLY four lines (the screen is the same height at round 9 as at round 1)",
          all(len(C.lines(x)) == C.SCORE_LINES == 4
              for x in (None, cm, clb, cd, cn, cs, c1, s1, s3)))
    check("the caveat is NOT a line of the block in any state (his ruling, 2026-10-22), yet it is "
          "still built on the card for every state that had one",
          all(x is not None for x in (clb, cd, cn, cs))
          and all(x.note not in C.lines(x) for x in (clb, cd, cn, cs) if x is not None and x.note)
          and all(x.note for x in (clb, cd, cn, cs) if x is not None)
          and cs.note.startswith("DISCARDED") and not any("DISCARDED:" in ln for ln in C.lines(cs)))
    check("the best is `BEST SCORE SO FAR  <n>` (a dash with no record), and when THIS round set it "
          "the line becomes `NEW BEST SCORE RECORDED` IN PLACE of it, not beside it",
          C.best_line(cm) == "BEST SCORE SO FAR  --"
          and C.best_line(s1) == "NEW BEST SCORE RECORDED"
          and C.best_line(s2) == "BEST SCORE SO FAR  416"
          and C.best_line(s3) == "BEST SCORE SO FAR  416"
          and C.best_line(None) == "BEST SCORE SO FAR  --"
          and "SO FAR" not in C.best_line(s1) and "GHOST" not in " ".join(C.lines(s1))
          and C.GHOST_SCORE == int(SCORE_SCALE))
    check("the efficiency best has the same two forms",
          C.efficiency_best_line(cm) == "BEST EFFICIENCY SO FAR  --"
          and C.efficiency_best_line(s1) == "NEW BEST EFFICIENCY RECORDED"
          and C.efficiency_best_line(s2) == f"BEST EFFICIENCY SO FAR  {EFF_HEALTHY:.2f}"
          and C.efficiency_best_line(None) == "BEST EFFICIENCY SO FAR  --")
    check("an unsettled card never announces a record that was not written (NEW needs a stored best)",
          C.best_line(dataclasses.replace(cm, new_score=True, best_score=None))
          == "BEST SCORE SO FAR  --"
          and C.efficiency_best_line(dataclasses.replace(cm, new_efficiency=True, best_efficiency=None))
          == "BEST EFFICIENCY SO FAR  --")
    # `cm` is built against an EMPTY book, so its own round takes the record and
    # `new_score` is True -- which makes the line read NEW BEST SCORE RECORDED and
    # say nothing about the stored basis. The STANDING form is what carries the
    # qualifier, so these three cards are pinned to new_score=False. The first
    # version of this check omitted that and read the NEW form in both halves: it
    # agreed with itself and tested nothing about ABOUT at all.
    stored_d = dataclasses.replace(cm, new_score=False, best_score=dict(VALID, basis="declared"))
    stored_m = dataclasses.replace(cm, new_score=False, best_score=dict(VALID, basis="measured"))
    check("a stored best set under an estimate is printed WITHOUT ABOUT (2026-10-22) while the "
          "record itself still says declared; the same stored value under a measured basis prints identically",
          C.best_line(stored_d) == "BEST SCORE SO FAR  416"
          and C.best_line(stored_d) == C.best_line(stored_m)
          and stored_d.best_score["basis"] == B.BASIS_DECLARED
          and "ABOUT" not in C.best_line(stored_d),
          f"{C.best_line(stored_d)!r} / stored basis {stored_d.best_score['basis']!r}")
    check("no line the block can print is over 36 characters (the longest RECORD announcement, "
          "`NEW BEST EFFICIENCY RECORDED`, is 28, shorter than the 33 it replaced)",
          max(len(ln) for x in (None, cm, clb, cd, cn, cs, c1, s1, s2, s3) for ln in C.lines(x)) <= 36
          and len("NEW BEST EFFICIENCY RECORDED") == 28)
    desc = C.describe(cs, ", LEFT")
    check("the log line carries what the screen compresses: that the round was DISCARDED and the "
          "test that did it, what it would have scored and that it would have beaten the best "
          "(358.8 -- the clamped figure -- with the raw speed 1.481 it was capped from), the "
          "efficiency it would have had, and the dim median it "
          "was NOT discarded for; and the new-best state of a sound round",
          "ROUND DISCARDED" in desc and "17 shots in 2.7 s" in desc
          and "it would have scored 358.8" in desc
          and "raw speed 1.481 was clamped to 1.000" in desc
          and "efficiency would have been 22.59" in desc
          and "WOULD HAVE BEATEN the stored best" in desc
          and "noted: " in desc and "median shot brightness was 17" in desc
          and "REFUSED" not in desc
          and "new score best" in C.describe(s1) and "NOT SCORABLE" in C.describe(cn)
          and "none" in C.describe(None), desc[:200])
    check("what left the SCREEN is still in the LOG, in full: the discarded storm's block shows "
          "none of its figures and describe() carries all of them",
          not any(w in ln for ln in C.lines(cs) for w in ("358", "359", "1.481", "531", "532", "22.59", "median"))
          and "358.8" in desc and "22.59" in desc and "median" in desc, desc[:200])
    check("there is NO scored-and-refused state left (his ruling, 2026-10-22): no log line, for a "
          "measured, an estimated, a floor or a discarded round, says a record was REFUSED -- a "
          "round that is shown is a round that is recorded, a lower bound included",
          all("REFUSED" not in d_ for d_ in (desc, desc_lb, desc_d, C.describe(cm, ", LEFT"))),
          "REFUSED survives in a log line")

    # -- bests do not collide ------------------------------------------------------
    multi = B.BestBook({})
    combos = [("static", 17, 1), ("static", 17, 2), ("static", 5, 1), ("reaction", 17, 1)]
    cards4 = [card(multi, key=B.best_key(*a), human_points=100 + 10 * i) for i, a in enumerate(combos)]
    C.settle(cards4, multi, "M")
    check("bests do not collide across drills, magazine sizes or player counts: four combinations, "
          "four entries, each holding its own round",
          len(multi.data) == 4 and all(
              close(rv(multi.best_score(B.best_key(*a))), cards4[i].gs.score)
              for i, a in enumerate(combos)))
    multi.take_changed()
    worse = card(multi, key=B.best_key("static", 5, 1), human_points=50)
    better = card(multi, key=B.best_key("reaction", 17, 1), human_points=160)
    snap_m = copy.deepcopy(multi.data)
    C.settle([worse, better], multi, "N")
    changed = {k for k in multi.data if multi.data[k] != snap_m.get(k)}
    check("a round improves ITS OWN best and no other",
          changed == {B.best_key("reaction", 17, 1)} and multi.take_changed())

    # -- two players: both halves, against the same prior --------------------------
    duo = B.BestBook({})
    k2 = B.best_key("static", 17, 2)
    ca, cb = card(duo, key=k2, human_points=100), card(duo, key=k2, human_points=150)
    oa, ob = C.settle([ca, cb], duo, "V")
    check("two players: the better half holds the record, the other half is NOT shown as new, "
          "and the file is to be written once",
          not oa.new_score and ob.new_score and close(rv(duo.best_score(k2)), cb.gs.score)
          and rv(oa.best_score) is not None and rv(oa.best_score) == rv(ob.best_score)
          and duo.take_changed()
          and not duo.take_changed())

    # -- a stored score is only as good as the ghost it was computed against ------
    stale = B.BestBook({KEY: {"score": dict(VALID, value=900.0, ghost_interval=0.30),
                              "efficiency": dict(GOOD_EFF, value=5.0)}})
    check("a score computed against a DIFFERENT ghost interval is ignored for comparison "
          "(ghost.py: changing the interval invalidates every stored best), "
          "but a stored efficiency is a measurement and is not",
          stale.best_score(KEY) is None and rv(stale.best_efficiency(KEY)) == 5.0)
    cst = card(stale)
    C.settle([cst], stale, "R")
    check("... so the next fit round replaces it at the CURRENT interval",
          cst.new_score and rv(stale.best_score(KEY), "ghost_interval") == GHOST_INTERVAL
          and close(rv(stale.best_score(KEY)), cst.gs.score))
    check("a stored score with no interval at all cannot be vouched for either",
          B.BestBook({KEY: {"score": {k: v for k, v in VALID.items() if k != "ghost_interval"}}}
                     ).best_score(KEY) is None)

    # -- the book's own rules, asked of the book and not of the card in front of it ------
    # The card refuses to offer a record that is not strictly better, so every check
    # above that goes through `settle` would stay green with the BOOK's own comparison
    # inverted or loosened. These offer to the book directly (tests/mutate_ghost.py).
    eq_book = B.BestBook({KEY: {"score": dict(VALID), "efficiency": dict(GOOD_EFF)}})
    snap_eq = copy.deepcopy(eq_book.data)
    check("the book itself refuses an EQUAL record: a tie returns False, signals nothing and "
          "leaves the stored record (and its date) as it was",
          eq_book.offer(KEY, dict(VALID, when="later"), dict(GOOD_EFF, when="later")) is False
          and not eq_book.take_changed() and eq_book.data == snap_eq)
    check("...and a strictly higher record IS written, once (the control for the tie above)",
          eq_book.offer(KEY, dict(VALID, value=500.0), None) is True
          and rv(rv(eq_book.data.get(KEY), "score")) == 500.0
          and eq_book.take_changed() is True and eq_book.take_changed() is False)
    full_book = B.BestBook({B.best_key(f"d{i}", 17, 1): {"efficiency": dict(GOOD_EFF)}
                            for i in range(B.MAX_ENTRIES)})
    new_key, old_key = B.best_key("one_more", 17, 1), B.best_key("d0", 17, 1)
    check("a full book refuses a NEW key (it holds MAX_ENTRIES and stays there), writes nothing "
          "and signals nothing",
          len(full_book.data) == B.MAX_ENTRIES
          and full_book.offer(new_key, None, dict(GOOD_EFF)) is False
          and len(full_book.data) == B.MAX_ENTRIES and new_key not in full_book.data
          and not full_book.take_changed())
    check("...but a full book still RAISES a key it already holds (the cap bounds keys, not records)",
          full_book.offer(old_key, None, dict(GOOD_EFF, value=99.0)) is True
          and rv(rv(full_book.data.get(old_key), "efficiency")) == 99.0
          and len(full_book.data) == B.MAX_ENTRIES)
    cu = card_or_none("a one-shot zero-span round with an efficiency figure", shots=1,
                      human_elapsed=0.0, efficiency=5.0)
    check("a round with NO score that arrives with an efficiency figure above the stored one "
          "(none stored) is never 'would beat' and never new: it has nothing to beat a best "
          "with (and there is no `eligible` field to consult: a card with no score cannot "
          "record, and a card with one can)",
          cu is not None and not cu.gs.scorable and not cu.would_beat
          and not cu.new_score and not cu.new_efficiency
          and not hasattr(C.RoundCard, "eligible") and "eligible" not in C.RoundCard.__dataclass_fields__,
          "no card" if cu is None else f"scorable={cu.gs.scorable} would_beat={cu.would_beat}")

    # -- seams: what the screen shows is what is stored ----------------------------
    src_sc = (ROOT / "src" / "lsa" / "scorecard.py").read_text(encoding="utf-8")
    src_bs = (ROOT / "src" / "lsa" / "bests.py").read_text(encoding="utf-8")
    imp = set()
    for path_ in (ROOT / "src" / "lsa" / "scorecard.py", ROOT / "src" / "lsa" / "bests.py"):
        for node in ast.walk(ast.parse(path_.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imp |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imp.add(("." * node.level) + (node.module or ""))
    check("the new modules are pure: no Qt, no camera, no audio, no numpy, and bests.py does not "
          "import settings (which imports it)",
          not (imp & {"PySide6", "cv2", "sounddevice", "numpy", "lsa.settings", "..settings"})
          and "from .settings" not in src_bs and "import settings" not in src_bs, str(sorted(imp)))
    check("the validity verdict is ghost.record_fitness's, applied ONCE inside ghost_score and read "
          "from the score, never re-derived: scorecard names no peak, span or microphone "
          "threshold, asks `fit` of nothing (a card that has a score IS sound), and reads the "
          "discard from `gs.discarded`",
          "MIN_MEDIAN_PEAK" not in src_sc and "MIN_SPAN_RATIO" not in src_sc
          and "MIN_CORROBORATED" not in src_sc and "record.fit" not in src_sc
          and "gs.discarded" in src_sc and "record_fitness(" not in src_sc)
    src_gh = (ROOT / "src" / "lsa" / "ghost.py").read_text(encoding="utf-8")
    check("and ghost_score is the ONLY door to a score: the arithmetic is private, and the public "
          "function judges what it returns (no caller can get a figure for a discarded round)",
          "def _arithmetic(" in src_gh and src_gh.count("_arithmetic(") == 2
          and "discarded=True" in src_gh)


def guarded_main() -> int:
    """`main()`, but a raise ANYWHERE in sections 1-12 is a FAILED CHECK with the summary
    still printed, exactly as section 13 already is (its own try/except in `main`).

    Why (2026-10-22): three mutate_ghost faults came back BROKEN -- the suite crashed
    before it printed a summary, and the runner correctly refuses to read a crash as a
    catch. Each was a check dereferencing a field the fault had emptied; those sites are
    fixed individually (`fx`, `rv`, `ratio`, S_safe, card_or_none). This is the net
    under the ones nobody has found yet: a crash then costs the checks AFTER it, and says
    so, instead of costing the whole report. It does NOT make a crash a catch: the runner
    still needs a failing line that names the fault's own check."""
    try:
        return main()
    except Exception as exc:                          # noqa: BLE001 - the finding
        import traceback
        check("the suite ran to its end without raising", False,
              f"{type(exc).__name__}: {exc}")
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"      raised at test_ghost.py line {tb.lineno}: {tb.line}")
        print()
        print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(guarded_main())
