"""Which suites to run for a change, and which to SKIP and why. The planner behind
scripts/run_affected.ps1.

WHY. The owner, 2026-10-21: "work on the test suites to make sure that they are
modular and invoked according to the areas affected by code; full regression is
invoked after considerable amount of changes or after an invasive change". The
full regression is ~30 minutes (see SLOW below); a one-line change to a key
legend does not need it.

WHAT IS TYPED HERE, AND WHAT IS NOT. Almost nothing is typed:
  * which suites cover a source file    -> tools/suites_for.py (the import graph)
  * which mutation runner a change earns -> tools/pre-commit (one home), plus, for
                                           a runner the hook does not list, the
                                           file names the runner itself quotes
  * the SUITES on disk                   -> globbed
What IS typed, and why it must be:
  * ALWAYS   - one suite, chosen for its job (it stops a rule being lost)
  * SLOW     - the cost of each slow runner, MEASURED, so the cost is on record
  * INVASIVE - the core paths where a fault is silent and expensive. A policy,
               not a fact about the code, so it cannot be derived. test_rules
               section S fails if any of it names something that no longer exists.

THE ESCALATION THRESHOLD IS NOT A ROUND NUMBER. Measured over the last 157
commits of this repo (git log --shortstat, 2026-10-21):
    files changed per commit   p50 4    p75 10   p90 18   p95 31
    lines changed per commit   p50 246  p75 621  p90 1705 p95 3422
"Considerable" is taken to mean "larger than 9 commits in 10 have been": MORE
THAN 18 FILES or MORE THAN 1700 LINES. Roughly one change in ten escalates, which
is often enough to catch a sprawling edit and rare enough that the full
regression is still an event.

THE PLAN IS PRINTED AS JSON, and the runner prints every decision from it. A
check that is skipped must look skipped, never passed.

    python tools/suite_plan.py --changed
    python tools/suite_plan.py --staged
    python tools/suite_plan.py --files src/lsa/modes/shortcuts.py,src/lsa/display/layers.py
    python tools/suite_plan.py --full
"""

from __future__ import annotations

import ast
import functools
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))   # asset_env: the asset table, by path
import asset_env as _ae                   # noqa: E402
import suites_for as sf                                    # noqa: E402

# suites_for rebuilds the whole import graph on every call (seconds each). The
# graph cannot change during one planning run, so build it once.
_cov_once = {}
_real_coverage = sf.coverage


def _coverage():
    if not _cov_once:
        _cov_once['c'] = _real_coverage()
    return _cov_once['c']


sf.coverage = _coverage

# -- tiers -------------------------------------------------------------------

ALWAYS = ["test_rules.py"]            # ~12 s here; what stops a rule being lost

# Seconds of wall clock, MEASURED on this machine. None = not measured: the
# 2026-10-21 brief forbade running the mutation runners, and an unmeasured cost
# is recorded as unmeasured rather than guessed.
#   mutate_detector 245, mutate_figures 709, mutate_introspect 77: measured
#   2026-10-21 by the owner. test_coldstart ~480 ("about 8 minutes").
SLOW: dict[str, int | None] = {
    "test_coldstart.py": 480,
    "mutate_detector.py": 245,
    "mutate_figures.py": 709,
    "mutate_introspect.py": 77,
    "mutate_director.py": None,
    "mutate_ghost.py": None,
    "mutate_audio.py": None,          # the microphone / own-bang faults; not yet measured
}

# The core, where a fault is silent and expensive. Touch one of these and the
# plan escalates to the full regression and SAYS SO.
INVASIVE = [
    "src/lsa/detect/",
    "src/lsa/calibrate/",
    "src/lsa/camera/",
    "src/lsa/director.py",
    "src/lsa/routines.py",
    "src/lsa/exposure.py",
    "src/lsa/settings.py",
    "src/lsa/selfcheck.py",
]
LARGE_FILES = 18
LARGE_LINES = 1700

# Changes that are not code. Named, so skipping them is a decision on the record.
NOT_CODE = (".md", ".txt", ".log", ".png", ".jpg", ".jpeg", ".gif", ".mp3", ".wav",
            ".json", ".cache", ".ps1", ".cmd", ".bat", ".txt", ".pyc")
NOT_CODE_DIRS = ("config/", "logs/", "cache/", "reports/",
                 "agent_reports/", "calib_backups/", "recordings/")
# DERIVED, never typed. These names live in `lsa/artwork/assets.py`, and a
# second copy here is the fault that has broken this project three times in
# five days: a folder moves, one table follows and the other does not.
# `asset_env` loads that module BY PATH, so the planner still never imports
# `lsa` (which would pull in OpenCV).
#
# Both layouts are accepted: the artwork now sits under `resources/`, and a
# path naming the bare folder is what the old tree produced.
ASSET_NAMES = tuple(n.rstrip("/") for n in _ae.ASSET_DIRS)
ASSET_DIRS = (tuple(f"{_ae.ASSET_PARENT}/{n}/" for n in ASSET_NAMES)
              + tuple(n + "/" for n in ASSET_NAMES))


# -- helpers -----------------------------------------------------------------

def disk_suites() -> list[str]:
    return sorted(p.name for p in TESTS.glob("test_*.py")) + \
        sorted(p.name for p in TESTS.glob("mutate_*.py"))


@functools.lru_cache(maxsize=None)
def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


@functools.lru_cache(maxsize=None)
def _test_imports() -> dict[str, set[str]]:
    """tests/*.py module name -> the tests/*.py modules it imports (helpers)."""
    mods = {p.stem for p in TESTS.glob("*.py")}
    out: dict[str, set[str]] = {}
    for p in TESTS.glob("*.py"):
        try:
            tree = ast.parse(_read(p))
        except SyntaxError:
            out[p.stem] = set()
            continue
        got = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                got |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                got.add(n.module.split(".")[0])
        out[p.stem] = got & mods - {p.stem}
    return out


def _importers_of(stem: str) -> set[str]:
    """Every suite that reaches tests/<stem>.py through other tests/ helpers."""
    graph = _test_imports()
    seen, todo = set(), [stem]
    while todo:
        cur = todo.pop()
        for mod, deps in graph.items():
            if cur in deps and mod not in seen:
                seen.add(mod)
                todo.append(mod)
    return {m + ".py" for m in seen if m.startswith("test_")}


def _runner_quotes(runner: str) -> set[str]:
    """Basenames a mutation runner quotes ("director.py"), for runners the
    hook does not list."""
    return set(re.findall(r'"([a-z_]+\.py)"', _read(TESTS / runner)))


def _mentions(stem: str, text: str) -> bool:
    return bool(re.search(r"(tools[\\/.'\"\s]+%s\b|import %s\b|from %s\b|%s\.py)"
                          % ((re.escape(stem),) * 4), text))


def _suites_mentioning_folder(folder: str) -> set[str]:
    name = folder.rstrip("/")
    pat = re.compile(r"""["'/\\]%s[/\\"']""" % re.escape(name))
    return {p.name for p in TESTS.glob("test_*.py") if pat.search(_read(p))}


def classify(path: str):
    """(suites, mutation runners, note) for ONE changed path. note is set when
    the answer is 'nothing', and says why, so nothing vanishes unexplained."""
    p = path.replace("\\", "/")
    full = ROOT / p
    suites: set[str] = set()
    muts: list[tuple[str, str]] = []

    if p.startswith("tests/") and full.suffix == ".py":
        if full.name.startswith("test_"):
            suites.add(full.name)
        elif full.name.startswith("mutate_"):
            muts.append((full.name, "the runner itself changed"))
        else:
            suites |= _importers_of(full.stem)
            if not suites:
                return set(), [], "a test helper nothing imports"
        return suites, muts, ""

    if p.startswith("src/lsa/") and p.endswith(".py"):
        s, m, orphans = sf.suites_for([p])
        suites |= set(s)
        muts += [(Path(x).name, w) for x, w in m]
        if not suites:
            return set(), muts, "src file that NO suite imports (unguarded)"
        return suites, muts, ""

    if p.endswith(".py") and p.startswith("tools/"):
        suites |= {t.name for t in TESTS.glob("test_*.py")
                   if _mentions(full.stem, _read(t))}
        m = [(Path(x).name, w) for x, w in
             sf.suites_for([p])[1]]
        return suites, m, ("" if suites or m else "a dev tool no suite mentions")

    if p.startswith(ASSET_DIRS):
        # `resources/sounds/x.mp3` belongs to `sounds/`, not to `resources/`:
        # the suites name the asset folder, which is what is searched for.
        _parts = p.split("/")
        top = next((x for x in _parts if x in ASSET_NAMES), _parts[0]) + "/"
        suites |= _suites_mentioning_folder(top)
        m = [(Path(x).name, w) for x, w in sf.suites_for([p])[1]]
        return suites, m, ("" if suites else "an asset folder no suite names")

    if (p.startswith(("scripts/", "tools/")) or p in ("RULES.md", "CLAUDE.md")
            or p.endswith((".bat", ".ps1", ".cmd"))):
        return set(), [], "read by test_rules (always tier)"

    if p.startswith(NOT_CODE_DIRS) or p.endswith(NOT_CODE):
        return set(), [], "not code"
    return set(), [], "UNMAPPED: no rule says what covers this file"


def all_reachable() -> set[str]:
    """Every suite that SOME changed path could name. test_rules section S uses
    this: a suite no path can ever run is worse than no suite."""
    named: set[str] = set()
    for suites in sf.coverage().values():
        named |= suites
    for p in (ROOT / "tools").glob("*.py"):
        named |= classify("tools/" + p.name)[0]
    for p in TESTS.glob("*.py"):
        if not p.name.startswith(("test_", "mutate_")):
            named |= classify("tests/" + p.name)[0]
    for d in ASSET_DIRS:
        named |= classify(d + "x")[0]
    return named


# -- the plan ----------------------------------------------------------------

def _git(args):
    done = subprocess.run(["git"] + args, cwd=str(ROOT), capture_output=True,
                          text=True)
    if done.returncode != 0:
        raise SystemExit("git failed: " + " ".join(args) + ": " + done.stderr)
    return done.stdout


def _git_prefix() -> str:
    """How git names the paths that live in THIS folder.

    `git diff --name-only` answers from the REPOSITORY root, and the app is a
    subfolder of its repository (`app/`), so a changed file arrives as
    `app/src/lsa/foo.py` while everything here counts from `app/`. Joining the
    two gave `app/app/src/...`: every lookup missed and the planner reported
    "nothing imports what changed" for a file that plainly had -- silently, and
    in the one tool whose whole job is to say which suites to run.

    Empty when the app IS the repository root, which is how the old tree was.
    """
    return _git(["rev-parse", "--show-prefix"]).strip().replace("\\", "/")


def _here(raw: str) -> str:
    """One git path, re-expressed against this app's root."""
    q = raw.replace("\\", "/")
    pre = _git_prefix()
    return q[len(pre):] if pre and q.startswith(pre) else q


def changed_paths(mode: str):
    """(paths, lines_changed). Working-tree and staged both measured by git."""
    if mode == "staged":
        paths = [_here(x) for x in _git(["diff", "--cached", "--name-only"]).splitlines() if x]
        stat = _git(["diff", "--cached", "--shortstat"])
    else:
        paths = []
        for line in _git(["status", "--porcelain", "-uall"]).splitlines():
            if len(line) > 3:
                paths.append(_here(line[3:].split(" -> ")[-1].strip().strip('"')))
        stat = _git(["diff", "HEAD", "--shortstat"])
        # tests/ is gitignored here, so git cannot see a change to a suite.
        # Judge it by modification time against the HEAD commit instead, and
        # say so in the plan (see "notes").
        head = int(_git(["log", "-1", "--format=%ct"]).strip() or 0)
        for p in TESTS.glob("*.py"):
            if int(p.stat().st_mtime) > head:
                paths.append("tests/" + p.name)
    nums = [int(n) for n in re.findall(r"(\d+) (?:insertion|deletion)", stat)]
    return sorted(set(paths)), sum(nums)


def make_plan(mode: str, paths=None, lines: int = 0) -> dict:
    notes: list[str] = []
    if mode == "files":
        paths = list(paths or [])
    elif mode in ("changed", "staged"):
        paths, lines = changed_paths(mode)
        if mode == "changed":
            notes.append("tests/ is untracked, so a changed suite is detected by "
                         "modification time newer than the HEAD commit, not by git.")
    else:
        paths = []

    universe = disk_suites()
    reasons: list[str] = []
    if mode == "full":
        reasons.append("-Full was asked for")
    for p in paths:
        q = p.replace("\\", "/")
        for pref in INVASIVE:
            if q == pref or (pref.endswith("/") and q.startswith(pref)):
                reasons.append("INVASIVE PATH TOUCHED: %s (under %s)" % (q, pref))
                break
    code_paths = [p for p in paths if p.endswith(".py")]
    if len(paths) > LARGE_FILES:
        reasons.append("LARGE CHANGE: %d files > %d (p90 of this repo's last 157 "
                       "commits)" % (len(paths), LARGE_FILES))
    if lines > LARGE_LINES:
        reasons.append("LARGE CHANGE: %d lines > %d (p90 of this repo's last 157 "
                       "commits)" % (lines, LARGE_LINES))
    escalated = bool(reasons)

    run: list[dict] = []
    skipped: list[dict] = []
    earned: list[dict] = []
    unmapped: list[str] = []
    placed: set[str] = set()

    def add(name, tier, why):
        if name in placed:
            return
        placed.add(name)
        run.append({"name": name, "tier": tier, "why": why})

    for a in ALWAYS:
        add(a, "always", "always tier: stops a rule being silently lost")

    if escalated:
        for n in universe:
            if n not in SLOW:
                add(n, "full", "full regression: " + reasons[0])
        for n in SLOW:                      # slow last, coldstart the very last
            if n != "test_coldstart.py":
                add(n, "slow", "full regression: " + reasons[0])
        add("test_coldstart.py", "slow", "full regression: " + reasons[0])
    else:
        per_suite: dict[str, list[str]] = {}
        per_mut: dict[str, list[str]] = {}
        for p in paths:
            s, m, note = classify(p)
            for x in s:
                per_suite.setdefault(x, []).append(p)
            for x, _w in m:
                per_mut.setdefault(x, []).append(p)
            if note.startswith("UNMAPPED") or "NO suite" in note or "no suite" in note \
                    or note.startswith("src file"):
                unmapped.append("%s  (%s)" % (p, note))
        # runners the hook does not list: earned when a file they quote changed
        for r in (n for n in SLOW if n.startswith("mutate_")):
            quoted = _runner_quotes(r)
            for p in paths:
                if Path(p).name in quoted:
                    per_mut.setdefault(r, []).append(p)
        for n in sorted(per_suite):
            if n in SLOW:
                per_mut.setdefault(n, []).extend(per_suite[n])
            else:
                add(n, "affected", "imports " + ", ".join(sorted(set(per_suite[n]))[:3])
                    + (" ..." if len(set(per_suite[n])) > 3 else ""))
        for n in sorted(per_mut):
            earned.append({"name": n, "why": "earned by " + ", ".join(
                sorted(set(per_mut[n]))[:3])})

    ran = {r["name"] for r in run}
    earned_names = {e["name"] for e in earned}
    for n in universe:
        if n in ran:
            continue
        if n in SLOW:
            cost = SLOW[n]
            tag = ("%ds" % cost) if cost else "cost not measured"
            if n in earned_names:
                skipped.append({"name": n, "why": "SLOW tier (%s): EARNED by this "
                                "change but not run -- pass -Slow to run it" % tag})
            else:
                skipped.append({"name": n, "why": "SLOW tier (%s): not earned by "
                                "this change" % tag})
        else:
            skipped.append({"name": n, "why": "not affected: imports nothing that "
                            "changed"})
    return {"mode": mode, "paths": paths, "lines": lines,
            "escalated": escalated, "reasons": reasons, "run": run,
            "skipped": skipped, "earned_slow": earned, "unmapped": unmapped,
            "notes": notes}


def main(argv) -> int:
    a = argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if a[0] == "--full":
        plan = make_plan("full")
    elif a[0] == "--staged":
        plan = make_plan("staged")
    elif a[0] == "--changed":
        plan = make_plan("changed")
    elif a[0] == "--files":
        plan = make_plan("files", [x for x in (a[1] if len(a) > 1 else "").split(",")
                                   if x.strip()])
    else:
        print("unknown option " + a[0], file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
