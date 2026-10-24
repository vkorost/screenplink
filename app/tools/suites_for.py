"""Which suites cover these files? Answered from the import graph, not a list.

WHY THIS EXISTS. On 2026-10-19 a slice changed `_poll_audio` in
`tools/camera_lab.py`. The brief said "focused suites, test_modes is your main
one", the implementer ran five, and the suite that actually covered the change
-- `test_feedback` -- was not among them. The full harness found it fifteen
minutes later as thirteen phantom mutation failures, because the crash took a
whole mutation run down with it. The mapping was not missing: `tools/pre-commit`
has held it for days, inside its own grep patterns, where nothing could read it.

A HAND-WRITTEN TABLE WOULD DRIFT, and this project has rules about exactly that
(C7, a cache with no invalidation is a constant; and do not pin a check to the
state of the world). So nothing here is typed. A suite covers a source file when
the suite IMPORTS it, directly or through anything it imports -- which is the
real relation, and it catches the chain that was missed: `test_feedback` never
mentions `modes/reaction.py`, but it imports `lsa.tools.camera_lab`, which
imports `lsa.modes`, which registers every drill.

Most suites here import inside the function that needs the module rather than at
the top, so the whole syntax tree is walked, not the first few lines.

    python tools/suites_for.py src/lsa/modes/reaction.py src/lsa/modes/base.py
    python tools/suites_for.py --staged        # what is staged for commit
    python tools/suites_for.py --changed       # what the working tree has
    python tools/suites_for.py --self-test     # prove it answers the 2026-10-19 case

The mutation runners are not guessed either: their trigger patterns are read out
of `tools/pre-commit`, so that mapping keeps exactly one home.
"""

from __future__ import annotations

import ast
import functools
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"


# -- the import graph --------------------------------------------------------

def _module_path(dotted: str):
    """`lsa.modes.reaction` -> the file on disk, module or package."""
    if not dotted.startswith("lsa"):
        return None
    rel = Path(*dotted.split("."))
    for cand in (SRC / rel.with_suffix(".py"), SRC / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _package_of(path: Path) -> str:
    """The PACKAGE a file lives in, which is what a relative import is relative
    to: `lsa/modes/base.py` -> `lsa.modes`, `lsa/modes/__init__.py` -> `lsa.modes`.

    This used to return the MODULE's own dotted name for a plain module
    (`lsa.modes.base`), one level too deep. `from .. import scorecard` in base.py
    then resolved to `lsa.modes.scorecard`, which does not exist, and was dropped
    without a word. Only `__init__.py` files (already package-named) resolved
    correctly, so a module reachable ONLY through a plain module's relative import
    -- `scorecard.py`, via `modes/base.py` -- was reported as covered by one suite
    when four reach it (2026-10-22)."""
    rel = path.relative_to(SRC)
    return ".".join(rel.parts[:-1])


@functools.lru_cache(maxsize=None)
def _imports(path: Path, package):
    """Every `lsa.*` module this file imports, anywhere in it.

    Relative imports are resolved against `package`, which is how
    `from .base import Mode` inside `lsa/modes/reaction.py` becomes
    `lsa.modes.base`. One that cannot be resolved is dropped, never guessed at.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return frozenset()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and package:
                parts = package.split(".")
                keep = parts[:max(0, len(parts) - node.level + 1)]
                bits = keep + ([node.module] if node.module else [])
                head = ".".join(b for b in bits if b)
            else:
                head = node.module or ""
            if not head:
                continue
            out.add(head)
            for a in node.names:
                out.add(head + "." + a.name)      # `from lsa import exposure`
    return frozenset(m for m in out if m.split(".")[0] == "lsa")


def closure(path: Path):
    """Every source file reachable from `path` by imports, including itself."""
    seen = set()
    todo = [path]
    while todo:
        cur = todo.pop()
        if cur in seen or not cur.is_file():
            continue
        seen.add(cur)
        pkg = _package_of(cur) if SRC in cur.parents else None
        for dotted in _imports(cur, pkg):
            nxt = _module_path(dotted)
            if nxt is not None and nxt not in seen:
                todo.append(nxt)
    return seen


def coverage():
    """source file -> the set of suites that reach it."""
    cover = {}
    for suite in sorted(TESTS.glob("test_*.py")):
        for src in closure(suite):
            if src == suite:
                continue
            cover.setdefault(src.resolve(), set()).add(suite.name)
    return cover


# -- the mutation runners, read out of the hook ------------------------------

def mutation_triggers():
    """(script, compiled pattern, what it guards) for each `run_mutation` line
    in the pre-commit hook. Read, never retyped: one home for that mapping."""
    hook = ROOT / "tools" / "pre-commit"
    if not hook.is_file():
        return []
    text = hook.read_text(encoding="utf-8", errors="replace")
    pat = (r"run_mutation\s+(\S+)\s*\\?\s*"
           r"'([^']+)'\s*\\?\s*"
           r'"([^"]+)"')
    out = []
    for script, trigger, what in re.findall(pat, text, re.S):
        try:
            out.append((script, re.compile(trigger), what))
        except re.error:
            continue
    return out


# -- the answer --------------------------------------------------------------

def suites_for(paths):
    """(suites, mutation runners, unguarded files) for the changed `paths`."""
    cover = coverage()
    suites = set()
    orphans = []
    for raw in paths:
        p = (ROOT / raw.replace("\\", "/")).resolve()
        if p.name.startswith("test_") and p.parent == TESTS:
            suites.add(p.name)                   # a suite covers itself
            continue
        hit = cover.get(p)
        if hit:
            suites |= hit
        elif p.suffix == ".py" and SRC in p.parents:
            orphans.append(raw)
    posix = [raw.replace("\\", "/") for raw in paths]
    muts = [("tests/" + script, what)
            for script, trigger, what in mutation_triggers()
            if any(trigger.search(q) for q in posix)]
    return sorted(suites), muts, orphans


def _git(args):
    try:
        done = subprocess.run(["git"] + args, cwd=str(ROOT),
                              capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print("git failed: " + str(exc), file=sys.stderr)
        return []
    return [line for line in done.stdout.splitlines() if line.strip()]


def self_test() -> int:
    """Prove it answers the case it was built for, and that it can say NO.

    A tool whose answer has never been checked is this project's oldest
    mistake, so the claims in the docstring are asserted here instead of
    being asserted in prose.
    """
    cover = coverage()

    def who(rel):
        return cover.get((ROOT / rel).resolve(), set())

    lab = who("src/lsa/tools/camera_lab.py")
    react = who("src/lsa/modes/reaction.py")
    base = who("src/lsa/modes/base.py")
    muts = mutation_triggers()
    _s1, m1, _o1 = suites_for(["src/lsa/detect/laser.py"])
    _s2, m2, _o2 = suites_for(["src/lsa/modes/reaction.py"])
    _s3, _m3, o3 = suites_for(["src/lsa/modes/base.py"])

    base_py = SRC / "lsa" / "modes" / "base.py"
    base_imports = _imports(base_py, _package_of(base_py))
    checks = [
        # The cause of the 2026-10-22 under-coverage, asserted at its source: a
        # relative import inside a PLAIN MODULE resolves against its package.
        ("a relative import in a plain module resolves: modes/base.py imports "
         "lsa.scorecard (from .. import scorecard)",
         "lsa.scorecard" in base_imports and "lsa.modes.scorecard" not in base_imports,
         sorted(base_imports)),
        ("the 2026-10-19 miss: a camera_lab.py change names test_feedback",
         "test_feedback.py" in lab, sorted(lab)),
        ("a reaction.py change names test_modes",
         "test_modes.py" in react, sorted(react)),
        ("...and reaches test_feedback through camera_lab",
         "test_feedback.py" in react, sorted(react)),
        ("a modes/base.py change names four suites or more",
         len(base) >= 4, sorted(base)),
        ("it can say NO: an unimported file names no suites",
         who("src/lsa/no_such_module.py") == set(), "non-empty"),
        ("the hook's mutation triggers parsed (3 or more)",
         len(muts) >= 3, len(muts)),
        ("a detector change earns mutate_detector",
         any("mutate_detector" in s for s, _w in m1), m1),
        ("a reaction change earns no mutation runner",
         m2 == [], m2),
        ("a covered source file is not reported unguarded",
         o3 == [], o3),
        # The ghost score's three modules are reached by the graph alone (no list
        # names them): test_ghost imports them, and modes/base.py imports all three,
        # so every suite that builds a mode reaches them too.
        ("a ghost.py / scorecard.py / bests.py change names test_ghost, test_modes "
         "and test_ui_smoke",
         all({"test_ghost.py", "test_modes.py", "test_ui_smoke.py"} <= who(f)
             for f in ("src/lsa/ghost.py", "src/lsa/scorecard.py", "src/lsa/bests.py")),
         {f: sorted(who(f)) for f in ("src/lsa/ghost.py", "src/lsa/scorecard.py",
                                      "src/lsa/bests.py")}),
    ]
    bad = 0
    for name, ok, detail in checks:
        print(("  ok    " if ok else "  FAIL  ") + name
              + ("" if ok else "   [" + str(detail) + "]"))
        bad += 0 if ok else 1
    print("\n%d of %d checks passed" % (len(checks) - bad, len(checks)))
    return 1 if bad else 0


def main(argv) -> int:
    args = argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--self-test":
        return self_test()
    if args[0] == "--staged":
        paths = _git(["diff", "--cached", "--name-only"])
    elif args[0] == "--changed":
        paths = [line[3:] for line in _git(["status", "--porcelain"])]
    else:
        paths = args
    if not paths:
        print("nothing changed")
        return 0
    suites, muts, orphans = suites_for(paths)
    print("changed:")
    for p in paths:
        print("  " + p)
    print("\nrun these suites:")
    if suites:
        for s in suites:
            print("  tests" + chr(92) + s)
    else:
        print("  (none -- nothing imports what changed)")
    if muts:
        print("\nand these mutation runners (slow, so they are earned):")
        for script, what in muts:
            print("  %s   [%s]" % (script, what))
    if orphans:
        print("\nNO SUITE IMPORTS THESE -- a change here is unguarded:")
        for o in orphans:
            print("  " + o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
