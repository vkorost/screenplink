"""The ONE place a mutation result is judged. Every `mutate_*.py` calls this.

THE RULE (RULES C1, E3). A fault is CAUGHT only when the check written against
it FAILS: a failing line that carries the entry's own `check_substring`. Not
"the suite went red". Not "something printed [FAIL]".

WHY "RED FOR ANY REASON" MAKES THE NUMBERS FICTION IN BOTH DIRECTIONS.
2026-10-23: a new section of `test_ui_smoke.py` used a stand-in capture that
lacked one method, so the section raised halfway and the suite reported a single
failing line ("the brightness-grid section ran to its end without raising --
AttributeError"). `mutate_detector` then counted EVERY fault aimed at that suite
as caught, because the suite was red -- 22 of them, "112 of 112". Two things were
false at once:

  * it CLAIMED proof that did not exist: those 22 checks never ran, so nothing
    showed they can fail; the "caught" text was the crash line, not their own;
  * it HID a real crash: a suite dying mid-section is a defect in the harness
    that the same tally reported as a success.

Both directions are the same error, and the second is the dearer: it takes away
the only signal that the instrument itself is broken. That is also why the
fallback `named or fails` (when nothing names the fault, accept any failure) is
GONE from every runner: it is precisely the line that turned a crash into a catch.

THE FOUR OUTCOMES
    caught   a failing line contains the fragment and is not itself a crash report
    MISSED   the suite stayed green, or failed only on checks unrelated to this
             fault: the check written for it cannot fail
    BROKEN   the suite timed out, never reached its summary, or reported a
             crashed section -- and its own check did not fail. NOT caught, and
             NOT merely missed: the instrument is unreliable for this entry
    UNPLACED the anchor matched zero or several times (reported by the runners)

Every outcome but `caught` fails the run.

THE BASELINE. `baseline()` runs every distinct suite the picked entries name,
unmutated, in the same scratch tree, and returns what is red. A guard that checks
a subset is `_screen_busy()` again: one list, consulted by one of the places that
needed it. The runners derive the suite list FROM THEIR ENTRIES.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

# A suite that ran to its end prints ONE summary line, in one of these dialects:
#   ALL 40 CHECKS PASSED          ALL 12 MCP CHECKS PASSED    ALL CHECKS PASSED
#   SOME CHECKS FAILED            SOME CHECKS FAILED (3 of 40)
#   3 OF 40 CHECKS FAILED         FAILED 2 of 80 ROUTINE CHECKS:
#   FAILED 5 of 129 checks:       FAILED 1 of 5: [...]      (no word "checks" at all)
#
# 2026-10-23: this pattern was `CHECKS PASSED|CHECKS FAILED|ROUTINE CHECKS`, so
# test_director ("FAILED n of m checks:", lower case) matched none of it and 19 of
# 25 director entries were reported BROKEN ("SUITE DID NOT REPORT") when their
# checks had in fact gone red -- a judge converting real catches into fake crashes.
# It is anchored to the START of a line: a failing check's own text may say "checks
# failed" in passing, and that is not the suite reporting.
#
# THIS LIST IS NOT THE AUTHORITY. `tests/test_rules.py` section M derives every
# suite's summary prints from the suites' own source (the syntax tree, not a list)
# and requires THIS pattern to read each one. A new dialect goes red there.
SUMMARY = re.compile(
    r"^[ \t]*(?:ALL\b.*\bCHECKS PASSED"
    r"|SOME CHECKS FAILED"
    r"|\d+ OF \d+ CHECKS FAILED"
    r"|FAILED \d+ of \d+\b)",
    re.M)
GREEN = re.compile(r"^[ \t]*ALL\b.*\bCHECKS PASSED", re.M)

# A failing line that REPORTS A CRASH rather than a verdict. Suites wrap a section
# in try/except and record `check("... ran to its end without raising", False,
# "<ExcType>: <message>")`; that is a crash wearing a check's clothes.
CRASH_MARKERS = ("without raising", "traceback (most recent call last)")


def fail_lines(text: str) -> list[str]:
    """Every line a suite printed as a failing check, in any of its dialects.

    A failing line STARTS with its marker (after leading whitespace); one that merely
    CONTAINS the marker is data. 2026-10-23: test_feedback prints
    `[PASS] a hardened stream writes it ... --   [FAIL] it names <window title>` -- a
    PASS whose payload quotes the marker -- and the old `"[FAIL]" in ln` read it as red,
    so the baseline guard called test_feedback red and mutate_detector (115 entries) and
    mutate_figures (56) refused to start while the suite passed 538/538."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("[FAIL]") or re.match(r"FAIL\b", s):
            out.append(s)
    return out


def is_crash(line: str) -> bool:
    low = line.lower()
    return any(m in low for m in CRASH_MARKERS)


def classify(rc: int, text: str, frag: str) -> tuple[str, str]:
    """('caught' | 'MISSED' | 'BROKEN', one line of why).

    `rc` is the child's return code (-9 for a timeout, by convention here).
    `frag` is the entry's `check_substring`, matched case-insensitively against
    the FAILING lines only.
    """
    if not frag or not frag.strip():
        # An entry that names no check cannot be attributed. Refuse it loudly
        # rather than fall back to "any failure" -- that fallback is the fault.
        return "BROKEN", "this entry names no check (empty check_substring): nothing can be attributed"
    fails = fail_lines(text)
    if rc == -9:
        return "BROKEN", "TIMED OUT: " + (text.strip().splitlines() or [""])[-1][:120]
    if not SUMMARY.search(text):
        tail = [ln for ln in text.splitlines() if ln.strip()][-3:]
        return "BROKEN", "SUITE DID NOT REPORT (a crash is not a catch): " + " | ".join(tail)
    crashes = [ln for ln in fails if is_crash(ln)]
    named = [ln for ln in fails if not is_crash(ln) and frag.lower() in ln.lower()]
    if named:
        return "caught", named[0][:160]
    if crashes:
        return "BROKEN", ("a section CRASHED and the check written for this fault did not "
                          f"fail ({frag!r}); a crash is not a catch. First crash: "
                          + crashes[0][:140])
    if rc == 0 and not fails:
        return "MISSED", "the suite stayed GREEN with this fault in place"
    return "MISSED", ("the suite failed, but not on the check written for this fault "
                      f"({frag!r}); first failure: " + (fails[0][:120] if fails else "none"))


def crash_reason(rc: int, text: str) -> str:
    """'' when the suite ran to its summary with no crashed section, else why not.
    For an entry that names NO check (mutate_director's, still): it cannot be
    attributed, but a crash can still be refused as a catch."""
    if rc == -9:
        return "TIMED OUT"
    if not SUMMARY.search(text):
        tail = [ln for ln in text.splitlines() if ln.strip()][-3:]
        return "SUITE DID NOT REPORT: " + " | ".join(tail)
    crashes = [ln for ln in fail_lines(text) if is_crash(ln)]
    if crashes:
        return "a section CRASHED: " + crashes[0][:140]
    return ""


def baseline_problem(rc: int, text: str) -> str:
    """'' when an UNMUTATED suite is green and ran to its end, else why not."""
    if rc == -9:
        return "TIMED OUT"
    fails = fail_lines(text)
    if fails:
        return f"{len(fails)} failing line(s); first: {fails[0][:150]}"
    if rc != 0:
        tail = [ln for ln in text.splitlines() if ln.strip()][-2:]
        return f"exit code {rc}; " + " | ".join(tail)[:200]
    if not GREEN.search(text):
        return "never printed its 'ALL ... CHECKS PASSED' summary"
    return ""


def baseline(suites: Iterable[str],
             run_one: Callable[[str], tuple[int, str]]) -> list[str]:
    """Run EVERY distinct suite once, unmutated. Returns 'suite: why' for each red
    one; an empty list is a green baseline. The caller passes the suites its
    entries name, so the guard cannot cover less than the attribution uses."""
    problems = []
    for suite in sorted(set(suites)):
        rc, text = run_one(suite)
        why = baseline_problem(rc, text)
        if why:
            problems.append(f"{suite}: {why}")
    return problems
