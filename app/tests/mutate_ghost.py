"""Reintroduce each fault the ghost score exists to prevent and watch test_ghost go red.

RULES E3 / C1: a check that has only ever been seen green proves nothing. This
does for `src/lsa/ghost.py` -- and, since 2026-10-22, for `src/lsa/bests.py` and
`src/lsa/scorecard.py`, which decide whether a permanent record is written into
his settings file -- what `mutate_figures.py` does for the figure drills:
it copies `src` and the one suite into a SCRATCH tree under tests/_out (never the
real tree, never his files), changes ONE thing, runs `test_ghost.py` there and
classifies the outcome:

    caught   the suite ran to its summary, failed, AND a failing line names the
             check written against this fault (not merely "something failed")
    MISSED   the suite stayed green, or failed only on checks unrelated to the
             fault -- the check that should notice cannot
    BROKEN   the suite CRASHED, timed out, or never reached its summary. NOT
             caught: a crash cannot be told from a broken harness, and on
             2026-10-19 twelve crashes were read as twelve caught faults
    UNPLACED the anchor did not match its target exactly once, so the fault was
             never applied (tests/test_rules.py section M also fails on this)

It REFUSES to run on a red baseline: a count measured against an already-failing
suite is fiction in both directions (2026-10-17: test_feedback was crashing on a
missing attribute and mutate_director still reported 24 of 24).

    python tests/mutate_ghost.py            all of them
    python tests/mutate_ghost.py 3 7        just these (by number)

Sequential on purpose: one tiny suite per fault, under a second each, and the one
shared mutation slot (worker_budget) so it can never be a second heavy job.

STALE BYTECODE (2026-10-21). Every fault here is applied to the SAME scratch file
and the suite runs in well under a second, so two consecutive mutants can have the
same byte length AND the same mtime second. CPython validates a cached .pyc by
(mtime in whole seconds, size) alone, so it reused the PREVIOUS mutant's compiled
code under the NEW fault's name: fault 1 came back MISSED quoting fault 0's failure,
fault 17 MISSED, fault 7 CAUGHT on a failure its own mutant cannot produce. The
false CAUGHT is the dangerous direction (a real hole reported as covered). So the
child never reads or writes bytecode: `-B` AND PYTHONDONTWRITEBYTECODE=1 (either
alone is a single point of failure), no __pycache__ is copied into the tree, any
that appears is purged before each run and fails the run if it reappears, and
`self_check()` PROVES the above on every run (see its docstring).

    python tests/mutate_ghost.py --self-check     only the instrument's self-check
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_env import scratch_env                          # noqa: E402  (one table of asset overrides)
import mutation_anchors as anchors  # noqa: E402
import mutation_verdict as verdict  # noqa: E402  (the ONE judgement of a result)
import worker_budget as budget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "_out" / "mutate_ghost"
PY = sys.executable
TEST = "test_ghost.py"
GHOST = "src/lsa/ghost.py"
BESTS = "src/lsa/bests.py"
CARD = "src/lsa/scorecard.py"
LAB ="src/lsa/tools/camera_lab.py"
TIMEOUT_S = 120

# (what is broken, file, old text, new text, test file, text a failing line must contain)
# `old` must match its file EXACTLY ONCE. The fragment is matched case-insensitively
# against the [FAIL] lines and is copied from the label of the check written for it.
M = [
    # -- 2026-10-22: a round is SOUND or DISCARDED, and only the span discards ----------
    ("the dim-median CAUTION is deleted (the log stops noting what it did not discard for)",
     GHOST, "        if med < MIN_MEDIAN_PEAK:", "        if False:",
     TEST, "median peak 15"),
    ("the span condition is deleted from record_fitness (a storm is no longer discarded)",
     GHOST, "    if human_elapsed < floor:", "    if False:",
     TEST, "DISCARDED for being too fast"),
    ("the discard floor is set back above the ghost's own pace (the ghost is discarded; the score caps near 833)",
     GHOST, "MIN_SPAN_RATIO = 0.91", "MIN_SPAN_RATIO = 1.2",
     TEST, "a human who IS the ghost"),
    ("the discard floor drops under the slowest real storm (a 3.3 s storm is scored)",
     GHOST, "MIN_SPAN_RATIO = 0.91", "MIN_SPAN_RATIO = 0.80",
     TEST, "the 3.30 s storm is DISCARDED"),
    ("the floor counts shots, not the gaps between them (a short magazine's ghost is discarded)",
     GHOST, "    floor = MIN_SPAN_RATIO * max(0, int(shots) - 1) * GHOST_INTERVAL",
     "    floor = MIN_SPAN_RATIO * max(0, int(shots)) * GHOST_INTERVAL",
     TEST, "every magazine from 2"),
    ("the microphone-majority CAUTION is deleted",
     GHOST, "        if audible > 0 and corroboration.heard_and_seen < MIN_CORROBORATED * audible:",
     "        if False:",
     TEST, "MIC_UNCORROBORATED as a CAUTION"),
    ("the microphone is judged even when it has not calibrated",
     GHOST, "    if corroboration is not None and corroboration.mic_ready:",
     "    if corroboration is not None:",
     TEST, "has not calibrated is SKIPPED"),
    ("shots fired while the microphone was deaf stay in the denominator",
     GHOST, "        audible = int(shots) - max(0, int(corroboration.deaf))",
     "        audible = int(shots)",
     TEST, "leave the denominator"),
    ("no microphone is a reason to discard a round (the lock-out he ruled out)",
     GHOST, "    if corroboration is not None and corroboration.mic_ready:",
     "    if corroboration is None or not corroboration.mic_ready:\n"
     "        reasons.append(Unfit.MIC_UNCORROBORATED)\n"
     "        parts.append(\"no microphone\")\n"
     "    if corroboration is not None and corroboration.mic_ready:",
     TEST, "NO microphone verdict at all"),
    ("a dim median is promoted to a REASON again (it would discard his own round of 256)",
     GHOST, "            cautions.append(Unfit.PEAK_TOO_LOW)",
     "            reasons.append(Unfit.PEAK_TOO_LOW)",
     TEST, "HIS REAL ROUND IS SCORED"),
    ("a silent microphone is promoted to a REASON again (a flaky one discards a real round)",
     GHOST, "            cautions.append(Unfit.MIC_UNCORROBORATED)",
     "            reasons.append(Unfit.MIC_UNCORROBORATED)",
     TEST, "MIC_UNCORROBORATED as a CAUTION"),
    ("the storm is let through when the microphone is absent: span condition gone",
     GHOST, "    if human_elapsed < floor:", "    if False:",
     TEST, "the storm is DISCARDED with no microphone"),
    ("the discard itself is deleted: ghost_score returns the storm's score (shown, then refused)",
     GHOST, "    discarding = [r for r in gs.record.reasons if r is not Unfit.NO_SCORE]",
     "    discarding = []",
     TEST, "THE PHANTOM STORM IS DISCARDED WHOLE"),
    ("a discarded round keeps its score on the result (the screen could print what it will not honour)",
     GHOST, "    return replace(gs, scorable=False, not_scorable=why, score=None, shown=None,",
     "    return replace(gs, scorable=False, not_scorable=why, score=gs.score, shown=gs.shown,",
     TEST, "has NO figure to show"),
    ("GHOST_INTERVAL is changed (every stored best silently changes meaning)",
     GHOST, "GHOST_INTERVAL = 0.25", "GHOST_INTERVAL = 0.30",
     TEST, "GHOST_INTERVAL is 0.25"),
    ("the escalating ceiling is replaced by a flat table",
     GHOST, "        return min(int(max_level), int(start_level) + int(k) - 1)",
     "        return int(max_level)",
     TEST, "escalating ceiling is"),
    ("the escalating cap is one too high",
     GHOST, "        return min(int(max_level), int(start_level) + int(k) - 1)",
     "        return min(int(max_level) + 1, int(start_level) + int(k) - 1)",
     TEST, "17-shot total is 75"),
    ("the score ignores speed (points share only)",
     GHOST, "    score = SCORE_SCALE * points_share * speed_share",
     "    score = SCORE_SCALE * points_share",
     TEST, "the score is 416"),
    ("the score is rounded before it is stored (the headline stops agreeing with its factors)",
     GHOST, "    score = SCORE_SCALE * points_share * speed_share",
     "    score = round(SCORE_SCALE * points_share * speed_share)",
     TEST, "score == 1000 x points share"),
    ("the speed share uses the cadence floor, not the ghost that waits for targets",
     GHOST, "    speed_share = ghost.elapsed / he",
     "    speed_share = ghost.cadence_elapsed / he",
     TEST, "the ghost then waits"),
    ("the shown score truncates instead of rounding",
     GHOST, "        shown=int(math.floor(score + 0.5)),", "        shown=int(score),",
     TEST, "the score is 416"),
    ("the lower_bound flag is dropped",
     GHOST, "    lower = bool(has_waits) and not ghost.waited", "    lower = False",
     TEST, "flagged as a lower bound"),
    ("the SINGLE_FRAME_PEAK seam: camera_lab's boundary moves, the ghost's copy does not",
     LAB, "    SINGLE_FRAME_PEAK = 30", "    SINGLE_FRAME_PEAK = 35",
     TEST, "equals CameraLab.SINGLE_FRAME_PEAK"),
    ("the ghost's own copy of the laser/noise boundary moves",
     GHOST, "MIN_MEDIAN_PEAK = 30.0", "MIN_MEDIAN_PEAK = 31.0",
     TEST, "median peak exactly 30"),
    ("only the FIRST caution is returned (the log would name one doubt and hide the other)",
     GHOST, 'tuple(cautions), "; ".join(notes))',
     'tuple(cautions[:1]), "; ".join(notes[:1]))',
     TEST, "TOGETHER still cannot discard"),
    ("a one-shot zero-span round raises instead of returning NOT SCORABLE",
     GHOST, "    if n < 2 or ghost.elapsed <= 0.0 or not math.isfinite(he) or he <= 0.0:",
     "    if False:",
     TEST, "does not raise"),
    ("the ghost ignores target-appearance waits",
     GHOST, "    if n < 2 or not _usable_waits(n, available_at):", "    if True:",
     TEST, "the ghost waits for it"),
    ("a zero-valued perfect shot is accepted as a ceiling",
     GHOST, "        if not math.isfinite(v) or v <= 0 or v != int(v):",
     "        if not math.isfinite(v) or v != int(v):",
     TEST, "zero-valued perfect shot"),
    ("a human who out-scores the stated maximum is scored anyway",
     GHOST, "    if hp > ghost.points:", "    if False:",
     TEST, "MORE points than the stated maximum"),
    # -- the 1000 ceiling (his ruling, 2026-10-23): clamp the SPEED share, keep the raw ------
    # NOT COVERED HERE, and said so: the host's DEFECT line (modes/base.py) and the wait
    # plan (Mode._ghost_wait_plan) are tested in test_modes, which this runner does not
    # run. The `test` column below is carried but run_suite() always runs TEST, so a
    # mutant against base.py would report MISSED for the wrong reason. See the report.
    ("the speed share is not clamped (a score above 1000 is possible again)",
     GHOST, "    speed_share = min(1.0, speed_share)", "    speed_share = speed_share",
     TEST, "NOTHING exceeds"),
    ("the clamp fires on float noise (a human who IS the ghost is reported as a model defect)",
     GHOST, "    if speed_raw > 1.0 + CLAMP_TOLERANCE:", "    if speed_raw > 1.0:",
     TEST, "float noise"),
    ("the raw speed share is lost (the log can no longer show the evidence the model is off)",
     GHOST, "        speed_raw=speed_raw, clamped=clamped)", "        speed_raw=speed_share, clamped=clamped)",
     TEST, "NOTHING exceeds"),
    ("a human who out-scores the maximum is no longer flagged as a wrong ceiling (the DEFECT never fires)",
     GHOST, "            ghost=ghost, ceiling_exceeded=True, **kw)", "            ghost=ghost, **kw)",
     TEST, "AIM above 1 never reaches"),
    ("describe() stops carrying the clamp (the screen shows 765 and the log is silent about the raw 1.4+)",
     CARD, "        bits.append(clamp_defect_text(gs))", "        pass",
     TEST, "clamped round's card carries"),
    ("a discarded round's log line forgets the raw speed behind its capped withheld figure",
     CARD, "        if gs.clamped and gs.speed_raw is not None:", "        if False:",
     TEST, "the log line carries what the screen compresses"),
    # -- the drill's declared wait: an estimate, counted and named (2026-10-21) ----
    ("the declared waits are not counted (an estimate passes as a measurement)",
     GHOST, "        declared = max(0, min(int(declared_waits), n - 1))", "        declared = 0",
     TEST, "5 declared waits are COUNTED"),
    ("the declared count is not clamped to the shots whose wait matters",
     GHOST, "        declared = max(0, min(int(declared_waits), n - 1))",
     "        declared = max(0, int(declared_waits))",
     TEST, "CLAMPED"),
    ("a declared estimate is treated as a lower bound again",
     GHOST, "    lower = bool(has_waits) and not ghost.waited",
     "    lower = bool(has_waits) and (not ghost.waited or bool(declared_waits))",
     TEST, "never a lower bound"),
    ("expected_max_uniform ignores a fall longer than the whole range",
     GHOST, "    if floor >= hi:", "    if False:",
     TEST, "a floor above the whole range"),
    # -- the card, the stored bests and their write-once signal (2026-10-22) --
    # NOT LISTED, ON PURPOSE (both are EQUIVALENT mutants, and the runner would report
    # MISSED for ever on an equivalent one):
    #   * "eligible drops the `scorable` term": ghost_score(scorable=False) is also what
    #     an unscorable round's `lower_bound`/score pair describes, and `new_score` needs
    #     `beats_s`, which needs a score; no correct check tells them apart.
    #   * "eligible stops asking record.fit": it NO LONGER ASKS. Until 2026-10-22 it
    #     did, and that was the second opinion his ruling removed ("how does it make
    #     sense to display the score and at the same time declare it invalid?"): a
    #     discarded round has no score, so `scorable` already says it. The load-bearing
    #     version of that mutation is the one in ghost_score, mutated above ("the
    #     discard itself is deleted").
    #
    # 2026-10-22, SECOND RULING: a LOWER-BOUND round IS recorded (as a floor). The
    # three faults that used to guard its exclusion ("eligible drops the not-lower_bound
    # term", "the score best bypasses eligibility", "the efficiency best bypasses
    # eligibility") are DELETED with `RoundCard.eligible` itself. Two of the three were
    # already reported MISSED (the suite stayed green with them applied). What replaces
    # them is the opposite fault, the refusal REINTRODUCED, one per record kind, plus
    # the three ways a floor could be stored or reported wrongly (below, and the
    # unknown-basis mutation further down).
    ("a lower-bound round is refused the SCORE record again (the screen shows a score the book will not keep)",
     CARD, "        new_score=bool(gs.scorable and book is not None and beats_s),",
     "        new_score=bool(gs.scorable and not gs.lower_bound and book is not None and beats_s),",
     TEST, "LOWER-BOUND round SETS a record"),
    ("a lower-bound round is refused the EFFICIENCY record again",
     CARD, "        new_efficiency=bool(gs.scorable and book is not None and beats_e),",
     "        new_efficiency=bool(gs.scorable and not gs.lower_bound and book is not None and beats_e),",
     TEST, "LOWER-BOUND round SETS a record"),
    ("a floor is stored as if it were measured (basis_of forgets the lower bound)",
     CARD, "    if gs.lower_bound:\n        return BASIS_LOWER\n    return _b.BASIS_DECLARED",
     "    return _b.BASIS_DECLARED",
     TEST, "MARKED as a floor"),
    ("the book cannot hold a floor (BASIS_LOWER is not a known basis: the record is silently dropped)",
     BESTS, "BASES = (BASIS_MEASURED, BASIS_DECLARED, BASIS_LOWER)",
     "BASES = (BASIS_MEASURED, BASIS_DECLARED)",
     TEST, "a stored floor is KEPT"),
    ("the log stops saying the stored score is a floor",
     CARD, '    floor = " (stored as a floor: the true score was at least this)" if gs.lower_bound else ""',
     '    floor = ""',
     TEST, "the log says the stored score is a floor"),
    ("would_beat drops the scorable term (a round with no score 'beats' a best)",
     CARD, "    would_beat = bool((gs.scorable or gs.discarded) and (beats_s or beats_e))",
     "    would_beat = bool(beats_s or beats_e)",
     TEST, "never 'would beat'"),
    ("would_beat ignores the WITHHELD score of a discarded round (the log stops saying the storm "
     "would have beaten the best on score)",
     CARD, "    raw_score = gs.withheld_score if gs.discarded else gs.score",
     "    raw_score = gs.score",
     TEST, "would_beat reads the WITHHELD score"),
    ("a discarded card still carries its efficiency (the screen could print a figure the book "
     "will not honour)",
     CARD, "    shown_eff = None if gs.discarded else efficiency",
     "    shown_eff = efficiency",
     TEST, "no efficiency on the card"),
    ("the discarded block is replaced by the ordinary score block (a number on a discarded round)",
     CARD, "        return discarded_lines(card)",
     "        return [headline(card), factors_line(card), best_line(card),\n"
     "                efficiency_best_line(card)]",
     TEST, "the SCREEN says so plainly"),
    ("settle keeps a side's NEW flag after the other half beat its record",
     CARD, "            new_score=c.new_score and _same(bs, c.gs.score),",
     "            new_score=c.new_score,",
     TEST, "the other half is NOT shown as new"),
    ("a worse round overwrites a better best (the comparison is inverted)",
     BESTS, '            if current is not None and not clean["value"] > current["value"]:',
     '            if current is not None and clean["value"] > current["value"]:',
     TEST, "a book can only be RAISED"),
    ("an EQUAL round overwrites the best (a tie rewrites the file)",
     BESTS, '            if current is not None and not clean["value"] > current["value"]:',
     '            if current is not None and clean["value"] < current["value"]:',
     TEST, "refuses an EQUAL record"),
    ("the key loses the drill (every drill shares one best)",
     BESTS, '    return f"{name}|m{mag}|p{2 if two else 1}"',
     '    return f"x|m{mag}|p{2 if two else 1}"',
     TEST, "keyed per DRILL"),
    ("the key loses the magazine size (a best on 5 rounds collides with 17)",
     BESTS, '    return f"{name}|m{mag}|p{2 if two else 1}"',
     '    return f"{name}|m17|p{2 if two else 1}"',
     TEST, "keyed per DRILL"),
    ("the key loses the player count (a two-player half collides with a solo round)",
     BESTS, '    return f"{name}|m{mag}|p{2 if two else 1}"',
     '    return f"{name}|m{mag}|p1"',
     TEST, "keyed per DRILL"),
    ("the stored ghost interval is ignored (a best from a different ghost is compared anyway)",
     BESTS, "    return gi is not None and abs(float(gi) - _ghost.GHOST_INTERVAL) < 1e-9",
     "    return True",
     TEST, "DIFFERENT ghost interval"),
    ("the write-once signal is never cleared (it fires on every poll)",
     BESTS, "        out, self._changed = self._changed, False",
     "        out = self._changed",
     TEST, "written ONCE"),
    ("the write-once signal never fires (a new best is never saved)",
     BESTS, "            self._changed = True",
     "            self._changed = False",
     TEST, "written ONCE"),
    ("a hand-edited entry that is not a dict RAISES instead of being dropped",
     BESTS, "        entry = raw[key]\n        if not isinstance(entry, dict):\n            continue\n"
     "        clean = {}",
     "        entry = raw[key]\n        clean = {}",
     TEST, "each dropped without a raise"),
    ("a hand-edited record that is not a dict RAISES instead of being dropped",
     BESTS, "    if kind not in KINDS or not isinstance(raw, dict):",
     "    if kind not in KINDS or raw is None:",
     TEST, "each dropped without a raise"),
    ("a stored bests value that is not a dict RAISES instead of being dropped",
     BESTS, "    if not isinstance(raw, dict):",
     "    if False:",
     TEST, "each dropped without a raise"),
    # A stored basis the book does not know is dropped. (Until 2026-10-22 this fault
    # was "a stored lower-bound record is accepted": the lower bound is a KNOWN basis
    # now, so the stand-in unknown value is a made-up one.)
    ("a stored record with an unknown basis is accepted as a best",
     BESTS, "            or shots is None or basis not in BASES:",
     "            or shots is None:",
     TEST, "each dropped without a raise"),
    ("the entry cap is removed from the loader (a runaway edit bloats the settings file)",
     BESTS, "        if len(out) >= MAX_ENTRIES:", "        if False:",
     TEST, "the dict is capped"),
    ("the entry cap is removed from the book (offer grows the file without bound)",
     BESTS, "                if len(self.data) >= MAX_ENTRIES:", "                if False:",
     TEST, "refuses a NEW key"),
]


def targets() -> list[str]:
    """Every repo-relative file a mutation touches (importable, no side effects)."""
    return sorted({m[1] for m in M})


def suite_files(test: str) -> list[str]:
    """The suite and every sibling module in tests/ it imports, transitively, as
    file names. DERIVED from the suite's own import statements (RULES C17): this
    runner used to copy `test_ghost.py` alone, so when every suite gained
    `import safe_stdio` the scratch tree could not import it and the baseline went
    red before a single fault was tried -- a hand-written copy list is a copy of
    the suite's import graph, and it drifted.

    A module is "a sibling" iff `tests/<name>.py` exists; everything else is the
    standard library or `src`. Imports anywhere in the file count (an import
    inside a function is still an import). What this cannot see is a dynamic
    import (importlib / __import__ / a path built at run time); that fails
    LOUDLY, not silently, because `_main` runs the unmutated suite in the scratch
    tree first and refuses to go on if it is red.
    """
    import ast
    tests = ROOT / "tests"
    seen: list[str] = []
    todo = [test]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.append(name)
        tree = ast.parse((tests / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if (tests / f"{m}.py").is_file():
                    todo.append(f"{m}.py")
    return sorted(seen)


def child_env() -> dict:
    """The environment every mutant's suite runs in: no bytecode, from any source."""
    env = scratch_env(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPYCACHEPREFIX", None)     # a prefix cache is shared across trees
    return env


def run_suite(tree: Path, env: dict, test: str = TEST,
              protected: bool = True) -> tuple[int, str]:
    """(returncode, output). A hang is a BROKEN run, so it is bounded.

    `protected=False` drops `-B` and exists ONLY for self_check()'s control arm,
    which must show the stale-bytecode hole is real on this interpreter.
    """
    cmd = [PY] + (["-B"] if protected else []) + [str(tree / "tests" / test)]
    try:
        r = subprocess.run(cmd, cwd=str(tree), env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return -9, f"TIMED OUT after {TIMEOUT_S}s"
    return r.returncode, r.stdout + "\n" + r.stderr


def _stray_bytecode(tree: Path) -> list[Path]:
    return sorted(tree.rglob("__pycache__"))


def _purge_bytecode(tree: Path) -> None:
    for d in _stray_bytecode(tree):
        shutil.rmtree(d, ignore_errors=True)


# -- the instrument proves it can tell two mutants apart --------------------------
_FIXED_SECOND = 1_700_000_000        # any whole second; both files are stamped with it
_PROBE_MOD = 'VALUE = "{}"\n'
_PROBE_SUITE = (
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
    "import probe_mod\n"
    'print("  [FAIL] probe value is " + probe_mod.VALUE)\n'
    'print("1 OF 1 CHECKS FAILED")\n'
    "raise SystemExit(1)\n")
_PROBE_SEQUENCE = ("AAAA", "BBBB", "AAAA")      # same length; the last is the restore


def _probe_arm(root: Path, env: dict, protected: bool) -> list[str]:
    """Run the probe suite through run_suite() once per value in _PROBE_SEQUENCE and
    return the value each child REPORTED. Every write is forced to the same byte
    length AND the same mtime second -- the exact condition that makes CPython
    trust a cached .pyc -- so the hazard is created on purpose, not hoped for."""
    tests = root / "tests"
    tests.mkdir(parents=True)
    (tests / "probe_suite.py").write_text(_PROBE_SUITE)
    mod = tests / "probe_mod.py"
    seen = []
    for v in _PROBE_SEQUENCE:
        mod.write_bytes(_PROBE_MOD.format(v).encode())
        os.utime(mod, (_FIXED_SECOND, _FIXED_SECOND))
        _rc, text = run_suite(root, env, "probe_suite.py", protected)
        m = re.search(r"probe value is (\w+)", text)
        seen.append(m.group(1) if m else "<no output: " + text.strip()[-80:] + ">")
    return seen


def self_check() -> tuple[bool, list[str]]:
    """(ok, lines). Proves, on EVERY run, that this runner's children cannot be
    served a previous mutant's compiled code -- and that the proof can fail.

    Two arms over the same sequence of same-length, same-second rewrites
    (AAAA, BBBB, AAAA) of a tiny module, each run through the real run_suite():
      CONTROL    no -B, no env var: must show STALE output. If it does not, the
                 hazard cannot be reproduced here, so the protected arm passing
                 would prove nothing -> FAIL ("a check that cannot fail").
      PROTECTED  exactly what the mutation loop uses: must report AAAA, BBBB,
                 AAAA, and must leave no __pycache__ behind.
    If the bytecode hole ever returns, the protected arm goes red here, before a
    single fault is tried, instead of the tally quietly drifting.
    """
    base = OUT / "selfcheck"
    shutil.rmtree(base, ignore_errors=True)
    lines: list[str] = []
    expected = list(_PROBE_SEQUENCE)

    control_env = child_env()
    control_env.pop("PYTHONDONTWRITEBYTECODE", None)
    control = _probe_arm(base / "control", control_env, protected=False)
    if control == expected:
        return False, [
            "SELF-CHECK CANNOT FAIL: with bytecode caching ON the control arm still "
            f"reported {control}, so the stale-pyc hazard is not reproducible on this "
            "interpreter and a green protected arm would prove nothing "
            "(SOURCE_DATE_EPOCH set? a different invalidation mode?)"]
    lines.append(f"control arm (caching on) reported {control}: the hazard is real here")

    protected = _probe_arm(base / "protected", child_env(), protected=True)
    stray = _stray_bytecode(base / "protected")
    if protected != expected or stray:
        return False, lines + [
            f"STALE BYTECODE: the protected arm reported {protected}, expected "
            f"{expected}" + (f"; and wrote {len(stray)} __pycache__ dir(s)" if stray else "")]
    lines.append(f"protected arm reported {protected}: same-length, same-second "
                 "mutants are told apart, and no bytecode was written")
    return True, lines


def judge(rc: int, text: str, frag: str) -> tuple[str, str]:
    """('caught' | 'MISSED' | 'BROKEN', one line of why).

    The judgement itself lives in `tests/mutation_verdict.py`, shared by all six
    runners. This one used to hold its own copy, which already attributed
    correctly -- and its siblings each held another that did not (2026-10-23)."""
    return verdict.classify(rc, text, frag)


classify = judge        # mutate_audio calls it by this name


def _main() -> int:
    args = sys.argv[1:]
    only_self_check = "--self-check" in args
    pick = [int(a) for a in args if not a.startswith("--")] or list(range(len(M)))
    t0 = time.perf_counter()
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    tree = OUT / "tree"
    (tree / "tests").mkdir(parents=True)
    shutil.copytree(ROOT / "src", tree / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name_ in suite_files(TEST):
        shutil.copy2(ROOT / "tests" / name_, tree / "tests" / name_)
    env = child_env()

    try:
        # THE INSTRUMENT FIRST. Before any fault is tried, prove this runner can
        # tell two same-length, same-second mutants apart (see self_check()).
        ok, notes = self_check()
        for ln in notes:
            print("  self-check: " + ln)
        if not ok:
            print("INSTRUMENT UNRELIABLE: no tally from this runner can be trusted.")
            return 2
        if only_self_check:
            print("self-check passed")
            return 0

        # PRECONDITION. "Caught" means the suite went red BECAUSE of the fault.
        rc, text = run_suite(tree, env)
        if _stray_bytecode(tree):
            print("BASELINE WROTE BYTECODE into the scratch tree despite -B and "
                  "PYTHONDONTWRITEBYTECODE: a later mutant could be served it.")
            return 2
        red = verdict.baseline_problem(rc, text)
        if red:
            print("BASELINE IS RED: test_ghost.py fails with NO fault applied, so "
                  "'caught' would mean nothing. Fix the suite first.")
            print("  " + red)
            for ln in [ln.strip() for ln in text.splitlines() if "[FAIL]" in ln][:10] \
                    or [ln for ln in text.splitlines() if ln.strip()][-5:]:
                print("  " + ln[:200])
            return 2

        pristine = {rel: (tree / rel).read_bytes() for rel in targets()}
        results = []
        for i in pick:
            name, rel, old, new, _test, frag = M[i]
            original = pristine[rel].decode()
            mutated = anchors.apply(original, old, new)
            if mutated is None:
                results.append((i, name, "UNPLACED",
                                f"{old[:60]!r}: " + anchors.not_found_reason(original, old)))
                continue
            try:
                _purge_bytecode(tree)         # belt: nothing compiled survives into this run
                (tree / rel).write_bytes(mutated.encode())
                rc, text = run_suite(tree, env)
                stray = _stray_bytecode(tree)
            finally:
                (tree / rel).write_bytes(pristine[rel])      # one fault at a time, always
            if stray:
                # -B was defeated: the NEXT mutant could be served this one's code.
                results.append((i, name, "BROKEN",
                                f"{len(stray)} __pycache__ dir(s) appeared despite -B and "
                                "PYTHONDONTWRITEBYTECODE, so this result is not trusted"))
                continue
            label, why = judge(rc, text, frag)
            results.append((i, name, label, why))

        bad = 0
        for i, name, label, why in results:
            print(f"  {label:<8} {i:2d} {name}")
            if label != "caught":
                bad += 1
            print(f"             -> {why[:220]}")
        print(f"{len(results) - bad} of {len(results)} ghost-score / bests / scorecard faults caught in "
              f"{time.perf_counter() - t0:.0f}s"
              + ("" if not bad else " -- A CHECK THAT CANNOT FAIL IS NOT A CHECK"))
        return 1 if bad else 0
    finally:
        shutil.rmtree(OUT, ignore_errors=True)


def main() -> int:
    """One mutation run at a time, and it REFUSES rather than queueing.

    See `tests/worker_budget.py` for the two incidents. The slot is shared
    by every `mutate_*` script.
    """
    try:
        with budget.exclusive():
            return _main()
    except budget.Busy as why:
        print(f"REFUSED: {why}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
