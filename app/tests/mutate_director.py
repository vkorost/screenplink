"""Reintroduce each fault the Director exists to prevent, and check the
tests go RED.

RULES E3: "If a rule has no test, it will be broken again. Write the test,
then verify it by reintroducing the fault and watching it go red."

This project has shipped four checks that could not fail the thing they
checked. A green suite is evidence of nothing until the mutation test has
been run once.
"""
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_anchors as anchors  # noqa: E402
import mutation_verdict as verdict  # noqa: E402  (the ONE judgement of a result)
import worker_budget as budget  # noqa: E402

# DERIVED, NEVER TYPED. This was the absolute path of the OLD working tree,
# which survived the move into this repository -- and a mutation runner EDITS
# SOURCE FILES IN PLACE and restores them afterwards, so running the harness
# from here would have injected faults into a different checkout entirely, and
# a crash mid-run would have left them standing in it. `parents[1]` is `app/`,
# the folder holding both `tests/` and `src/`, so this now follows the file.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "lsa" / "director.py"
ROUTINES = ROOT / "src" / "lsa" / "routines.py"
EXPOSURE = ROOT / "src" / "lsa" / "exposure.py"
LAB = ROOT / "src" / "lsa" / "tools" / "camera_lab.py"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

# Every suite that guards these properties. Not one file: the mutation check
# itself found that three Director properties were covered only in
# `test_feedback`, which this script did not run -- so disabling the
# `world_busy` bridge, the completion callback, or the queue drain entirely
# all left "the suite" green. A property guarded by a test the mutation
# script does not run is the same shape as a rule guarded by a comment.
TESTS = [ROOT / "tests" / "test_director.py",
         ROOT / "tests" / "test_routines.py",
         # ADDED 2026-10-16 after an outside review pointed out that
         # neither was run here -- so the exposure rule and the gain-map
         # judge, both changed that day on the strength of his lost
         # string, had no mutation anchor at all. One of their checks was
         # then found to pass with the thing it tested removed.
         ROOT / "tests" / "test_exposure.py",
         ROOT / "tests" / "test_feedback.py"]

# Each mutation is a fault that actually happened on this project.
MUTATIONS = [
    (
        "two routines may interleave (the 17:29 gate failure)",
        """        if self._active is not None and not self._active.finished:
            holder = self._active.job.name""",
        """        if False:
            holder = self._active.job.name""",
    ),
    (
        "cancel abandons the routine instead of running its cleanup",
        """        try:
            run.gen.throw(Cancelled(why))
        except (Cancelled, StopIteration):
            pass""",
        """        try:
            pass
        except (Cancelled, StopIteration):
            pass""",
    ),
    (
        "a wait is armed with no deadline (the immovable black rectangle)",
        """        run.timer = self._clock.after(ms, fire)""",
        """        run.timer = None if kind != "sleep" else self._clock.after(ms, fire)""",
    ),
    (
        "control writes are not de-duplicated (the auto-range ladder)",
        """            prev = self._sink.written_value(effect.key)""",
        """            prev = None""",
    ),
    (
        "a still is matched by kind alone, not by its tag",
        """            if tag != run.still_tag:""",
        """            if False:""",
    ),
    (
        # The first attempt at this mutation only removed the guard in
        # `_deliver` and the suite stayed green -- correctly, because
        # `_apply` holds the same property independently. That is defence
        # in depth, not a gap. This version removes the real defence.
        "a delivery satisfies a request the routine has not yet made",
        """        if run.waiting_for != kind:
            self.overlaps.append(
                f"{kind} arrived while {run.job.name} was waiting for "
                f"{run.waiting_for or 'nothing'}")
            return""",
        """        if False:
            return""",
    ),
    (
        # The real one, from 2026-10-15 22:08. Every other check runs on
        # ManualClock, so QtClock was never executed and shipped unable to
        # arm a single timer: the probe and the self test both died at
        # their first Sleep on the first cold start after the migration.
        "the production clock cannot arm a timer",
        """        t.setTimerType(self._precise)""",
        """        t.setTimerType(0)""",
    ),
    (
        "the production clock does not cancel, only orphans",
        """        try:
            handle.stop()
            handle.deleteLater()""",
        """        try:
            pass""",
    ),
    (
        # Added tonight, so unmutated until now. The bridge that keeps a
        # migrated routine from starting on top of pattern calibration or
        # the legacy noise floor, which are still on timers and flags.
        # Without it the migration itself reintroduces the collision it
        # exists to end.
        "a migrated routine ignores the routines not yet migrated",
        """        outside = ""
        try:
            outside = self._world_busy() or ""
""",
        """        outside = ""
        try:
            outside = ""
""",
    ),
    (
        # Also added tonight. A completion callback that never fires leaves
        # the startup chain stopped with nothing said -- the shape of the
        # wait that made him press Escape to get an answer.
        "a routine's completion callback never fires",
        """        while self._after:
            job, value = self._after.pop(0)""",
        """        while False:
            job, value = self._after.pop(0)""",
    ),
    (
        # And the drain that lets a queued routine start once something the
        # Director does NOT run has finished. Nothing else would wake it.
        "a queued routine is never woken after an outside owner finishes",
        """        if self._active is not None and not self._active.finished:
            return
        if not self._queue:
            return
        self._next()""",
        """        return""",
    ),
    (
        # From Fable's audit of the diff, 2026-10-15. Cancelling at
        # shutdown fired the chain's completion callback, which STARTS the
        # next routine on the slot the cancel just freed, while the target
        # window is being destroyed.
        #
        # REWRITTEN 2026-10-23. This entry used to delete the `start()`
        # refusal outright, which is EXACTLY what the next entry does with
        # `if False:` -- the two produced identical failing lines, so one of
        # them was a duplicate and its "(BOTH guards removed)" claim was not
        # true: the guards in `cancel()` and `_finish()` (`and not
        # self._closing`) were still in place, so the chain never handed on
        # and the check written for that ("the chain does NOT hand on during
        # teardown") never went red. Defence in depth, not a gap -- but then
        # nothing showed that check CAN fail. `close()` raising the closing
        # state is the one line all three guards read, so leaving it down
        # removes them all at once, which is what the name always claimed.
        "shutting down still hands the chain on (BOTH guards removed)",
        """        self._closing = True
        self.cancel(why)""",
        """        self._closing = False
        self.cancel(why)""",
    ),
    (
        "anything may start after the app has begun closing",
        """        if self._closing:""",
        """        if False:""",
    ),
    (
        # The two-loop divergence. `_throw` dispatched the next effect with
        # no guard, so a routine that caught EffectFailed and then yielded
        # another failing effect let the exception escape and wedged the
        # slot for the session, silently.
        "a failing effect inside an exception handler escapes the Director",
        """                except EffectFailed as exc:
                    # Handed back INTO the routine unchanged, so its finally
                    # blocks run and it reports the real reason rather than
                    # a nested repr of one. Round the loop, not into a
                    # second function.
                    throw = exc
                    continue""",
        """                except EffectFailed as exc:
                    raise""",
    ),
    (
        # Within one run, two stills with the same routine-level tag: the
        # first, late, frame answered the second question.
        "a still is matched by the routine's tag, not by the request",
        """            self._still_seq += 1
            run.still_tag = f"{effect.tag}#{self._still_seq}\"""",
        """            run.still_tag = effect.tag""",
    ),
    (
        # A cancel reached from inside the routine's own Call finished the
        # same run twice: history said `cancelled` and then `done`.
        "a run can be finished twice",
        """        if run.finished:
            # ONCE. A `cancel()` reached from inside one of the routine's own""",
        """        if False:
            # ONCE. A `cancel()` reached from inside one of the routine's own""",
    ),
    (
        # Likewise: clearing `_active` and checking `finished` are two
        # layers over one property, so removing either alone changes
        # nothing. Removing both is the fault.
        "a stale frame revives a routine that was already stood down",
        """        run.finished = True
        self._active = None
        self.history.append((run.job.name, "cancelled",""",
        """        run.finished = False
        self.history.append((run.job.name, "cancelled",""",
    ),
    (
        # Fable's review, 2026-10-16. The whole effect of this fault is that
        # the app's own verdict reads BETTER than the truth, and it compounds.
        # Every other instrument here measures detection, so all of them go
        # GREENER as it gets worse: this is the only check that can see it.
        "an unconfirmed dot radius is persisted (the ratchet)",
        ROUTINES,
        """        if stop or not confirmed:""",
        """        if stop:""",
    ),
    (
        # 2026-10-22. The path the 2026-10-16 fix left open: SEEN, but as a
        # blob smaller than his laser makes. The tuner grew the radius toward
        # the target area and the grown value was saved (9.62 -> 14.43 in one
        # session). Removing the band leaves "seen" as the only condition.
        "a dot seen outside the laser's blob range is persisted (the ratchet's second door)",
        ROUTINES,
        """            confirmed = measured and lab.dot_area_in_band(flashed_area)""",
        """            confirmed = measured""",
    ),
    (
        "a remembered radius that fails to confirm is carried forward anyway",
        ROUTINES,
        """            if not measured and remembered and left <= 0:""",
        """            if False:""",
    ),
    (
        # Fable's review, 2026-10-16: the section-11 fault with an outside
        # caller. A run installed mid-step logs as started and never
        # advances, holding the only slot there is. Both defences removed,
        # because either alone would leave the suite correctly green.
        "a routine may take the slot while another is mid-step",
        """        if self._resuming:
            # A ROUTINE IS MID-STEP.""",
        """        if False:
            # A ROUTINE IS MID-STEP.""",
    ),
    (
        # The one he spotted before any instrument did, 2026-10-16:
        # "I see a completely black screen." A full-screen black cover left
        # up by the two-marker check, for the rest of the session. The app
        # then measured its own cover and reported a 145.7mm mapping error
        # four lines after measuring that same mapping at 2.1mm.
        "a routine leaves its full-screen cover up (the black screen)",
        ROUTINES,
        """        lab._check_hide()""",
        """        lab._check_show(None)""",
    ),
    (
        # 2026-10-16. The probe judged each candidate against its own
        # three-frame noise reading, so the one with the luckiest floor
        # won -- it picked a dot of 22 over a dot of 57, and he fired a
        # string into that exposure. Ten trigger pulls heard, eight seen.
        "a candidate is judged on its own lucky noise reading",
        EXPOSURE,
        """        worst = max(worst, float(p.floor))
        out[p.exposure] = min_peak_for(worst)""",
        """        out[p.exposure] = min_peak_for(float(p.floor))""",
    ),
    (
        # The seam: Probe.bar() is documented as mirroring what the app
        # does with a measured floor, and it had drifted -- the app used
        # max(floor*1.3, floor+6) and this used max(8, floor*1.3). At a
        # floor of 5 that is 11 against 8, which was the whole margin.
        "the probe predicts a bar the detector does not use",
        EXPOSURE,
        """    return max(MIN_PEAK_FLOOR, floor * 1.3, floor + 6.0)""",
        """    return max(MIN_PEAK_FLOOR, floor * 1.3)""",
    ),
    (
        # The gain map is the cure for the far side of his screen, and a
        # guard comparing quantised FLOORS threw it away twice in one
        # morning: 1 -> 3 and 1 -> 4 read as 3x and 4x.
        "the gain map is judged on the floor instead of the bar",
        LAB,
        """        bar_before = expo.min_peak_for(before)
        bar_after = expo.min_peak_for(floor)
        rise = bar_after / max(1.0, bar_before)""",
        """        bar_before = before
        bar_after = floor
        rise = bar_after / max(1.0, bar_before)""",
    ),
]

# The check that must fail for each fault, by the fault's name (the first element
# of its MUTATIONS entry): a text a failing line must contain, as in every other
# runner. Under the strict judge (tests/mutation_verdict.py) a fault is CAUGHT only
# when a FAILING line carries its own fragment, so an entry with none can never be
# credited -- and this table was `{}` for a week while the runner printed "24 of 24"
# and "25 of 25", a tally no entry could have earned (2026-10-23). Every entry is now
# required to have one: `unattributed()` below, called by `_main` AND by
# tests/test_rules.py section M.
#
# HOW THESE WERE CHOSEN. Each fragment was read from the output of that fault
# actually applied (a measurer ran all 25, 2026-10-23), not guessed:
#   * the stable PREFIX of a label whose tail is a varying list ("...: {events}");
#   * never a label that embeds a clock time (entry 25's first red line says
#     "(10:53)" -- a check pinned to the state of the world);
#   * a line that goes red under THIS fault and names what this fault breaks.
# Several entries redden a shared family (3 and 6 and 15 all touch the late-frame
# checks); each is given the line that is its own, which is why 15 is not "the second
# request has its own tag" (entry 3 reddens that too) and 19 is not "and it names the
# band" (entry 18 reddens that too).
#
# Where each came from (suite, section):
#   entries 1-17 and 21 test_director; 18-20, 22, 24 test_routines; 23
#   test_exposure; 25 test_feedback. The "sN" tags below are test_director's section
#   numbers and are approximate; the fragment, not the tag, is what is judged.
FRAGS: dict = {
    # test_director s2
    "two routines may interleave (the 17:29 gate failure)":
        "the second is REFUSED",
    # s3
    "cancel abandons the routine instead of running its cleanup":
        "the routine was told WHY, and then cleaned up",
    # s4
    "a wait is armed with no deadline (the immovable black rectangle)":
        "even an absurd timeout terminates",
    # s6
    "control writes are not de-duplicated (the auto-range ladder)":
        "six redundant writes never reached the camera",
    # s5
    "a still is matched by kind alone, not by its tag":
        "a frame with the wrong tag is NOT handed over",
    # s5 (the delivery-before-the-request half)
    "a delivery satisfies a request the routine has not yet made":
        "the late floor did NOT satisfy the sleep",
    # s15
    "the production clock cannot arm a timer":
        "a real QTimer fired and resumed the routine",
    "the production clock does not cancel, only orphans":
        "the real timer was STOPPED, not left to fire later",
    # s16
    "a migrated routine ignores the routines not yet migrated":
        "a routine refuses while something OUTSIDE the Director holds the screen",
    # s17
    "a routine's completion callback never fires":
        "a completed routine hands its RESULT to the callback",
    # s16 (the poke() half of the bridge)
    "a queued routine is never woken after an outside owner finishes":
        "poke() starts it once the screen is free",
    # s18. Entry 12 removes ALL the closing guards (see its comment), so the
    # check only it breaks is the chain one: the start() refusal alone leaves
    # the chain not handing on (cancel() and _finish() guard it independently).
    "shutting down still hands the chain on (BOTH guards removed)":
        "the chain does NOT hand on during teardown",
    "anything may start after the app has begun closing":
        "nothing may start after close()",
    # s10 and s19, through start_no_escape()
    "a failing effect inside an exception handler escapes the Director":
        "a failing effect does NOT escape the Director",
    # s1
    "a still is matched by the routine's tag, not by the request":
        "the still was requested, with a unique tag",
    # s20
    "a run can be finished twice":
        "the run is recorded ONCE, not twice",
    # s3 (cleanup). NOT "the slot is free": that line is generic.
    "a stale frame revives a routine that was already stood down":
        "and the app knows it is no longer drawing",
    # test_routines
    "an unconfirmed dot radius is persisted (the ratchet)":
        "a remembered radius that is never seen is NOT persisted",
    # ...and NOT "and it names the band": entry 18 reddens that too.
    "a dot seen outside the laser's blob range is persisted (the ratchet's second door)":
        "a remembered radius that images out of band is re-tuned, not nudged once",
    "a remembered radius that fails to confirm is carried forward anyway":
        "a remembered radius that fails to confirm earns a full re-tune",
    # test_director s21
    "a routine may take the slot while another is mid-step":
        "no routine is left holding the slot",
    # test_routines
    "a routine leaves its full-screen cover up (the black screen)":
        "the pattern layer is taken down when the check passes",
    # test_exposure
    "a candidate is judged on its own lucky noise reading":
        "the brightest dot wins, not the luckiest floor",
    # test_routines
    "the probe predicts a bar the detector does not use":
        "the log says plainly that the dot is under its bar",
    # test_feedback. NOT the first red line, which embeds a clock time "(10:53)".
    "the gain map is judged on the floor instead of the bar":
        "grid: and one doing its job (floor 1 -> 3) is kept",
}


def unattributed() -> list:
    """Names of entries with no usable fragment, plus fragments keyed to no entry.

    Importable and side-effect free, so tests/test_rules.py can require it empty:
    an empty-dict default is what let 25 entries ship unattributed. A fragment keyed
    to a name that is not an entry is the same fault the other way round -- it is
    credited to nothing."""
    names = [m[0] for m in MUTATIONS]
    gaps = [f"{n}: no fragment" for n in names
            if not str(FRAGS.get(n, "")).strip()]
    gaps += [f"{k}: keyed to no entry" for k in FRAGS if k not in names]
    return gaps


def normalise(m):
    """(name, old, new) targets director.py; (name, file, old, new) any file."""
    if len(m) == 4:
        return (m[0], m[1], m[2], m[3])
    return (m[0], SRC, m[1], m[2])


def targets():
    """Every file a mutation touches (importable, no side effects)."""
    return sorted({normalise(m)[1] for m in MUTATIONS}, key=str)


def read_raw(f):
    """Raw text, line endings preserved (newline='' keeps CRLF)."""
    with io.open(f, encoding="utf-8", newline="") as fh:
        return fh.read()


def write_raw(f, text):
    with io.open(f, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def run_tests():
    """Red if ANY suite goes red. Returns (returncode, stdout)."""
    out = []
    # No bytecode in or out of the child (2026-10-21). This runner mutates the REAL
    # tree in place, so a .pyc written for one mutant is validated by (mtime in whole
    # seconds, size) only: a same-length rewrite landing in the same second is served
    # the previous mutant's compiled code under the new fault's name. -B AND the env
    # var, because either alone is a single point of failure.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPYCACHEPREFIX", None)
    for t in TESTS:
        r = subprocess.run([str(PY), "-B", str(t)], capture_output=True, text=True,
                           cwd=str(ROOT), env=env)
        out.append(r.stdout)
        if r.returncode != 0:
            out.append(f"\n[FAIL] suite {t.name} exited {r.returncode}\n")
            if r.stderr.strip():
                out.append("[FAIL] stderr: "
                           + r.stderr.strip().splitlines()[-1] + "\n")
            # The synthetic lines above are the RUNNER's, not the suite's. What the
            # failing suite itself printed is returned separately so a crash in it
            # can be told from a check failing (mutation_verdict.py).
            return (r.returncode, "".join(out), r.stdout + "\n" + r.stderr)
    return (0, "".join(out), "")


def _main():
    # BEFORE anything is mutated or run: an entry that names no check can never be
    # credited by the strict judge, so running it would only print a tally nobody can
    # earn (the "24 of 24" / "25 of 25" of a week, 2026-10-23).
    gaps = unattributed()
    if gaps:
        print("REFUSED: these entries are not attributed to a check (FRAGS):")
        for g in gaps:
            print(f"  - {g}")
        sys.exit(2)
    SOURCES = {f: read_raw(f) for f in targets()}
    backups = {}
    for f in SOURCES:
        backups[f] = f.with_suffix(".py.mutbak")
        shutil.copy2(f, backups[f])

    # PRECONDITION (RULES C1). "Caught" means the suite went red BECAUSE of the
    # fault. If the unmutated suites are already red, every fault is "caught"
    # by the existing failure and the result is meaningless -- found
    # 2026-10-17, when test_feedback was crashing on a missing attribute and
    # this script still reported 24 of 24.
    base_code, base_out, base_failing = run_tests()
    # EVERY suite in TESTS ran (run_tests stops at the first red one, so a green
    # return means all of them did), and each is required to have PRINTED ITS
    # GREEN SUMMARY: an exit code of 0 alone cannot tell a suite that ran from one
    # that never reached its checks.
    base_red = ""
    if base_code != 0:
        base_red = "a suite failed"
    elif base_out.count("CHECKS PASSED") < len(TESTS):
        base_red = (f"only {base_out.count('CHECKS PASSED')} of {len(TESTS)} suites printed "
                    "an 'ALL ... CHECKS PASSED' summary")
    if base_red:
        print("BASELINE IS RED: the suites fail with NO fault applied, so "
              "'caught' would mean nothing. Fix them first.")
        print("  " + base_red)
        print("\n".join(ln for ln in base_out.splitlines()
                         if "[FAIL]" in ln)[:1500])
        for f in SOURCES:
            backups[f].unlink(missing_ok=True)
        sys.exit(2)

    print(f"{len(MUTATIONS)} faults to reintroduce\n")
    survived = []      # applied, suite stayed green (or failed elsewhere than its check)
    unplaced = []      # could not be applied at all: a DIFFERENT problem
    broken = []        # a suite CRASHED: not a catch, and not a miss either
    # NAMED `red_unnamed`, NOT `unattributed`: there is a module-level function
    # `unattributed()` which `_main` calls above, and a local assignment anywhere
    # in a function makes that name local for the WHOLE function -- so the call
    # at the top died with UnboundLocalError and the runner never mutated
    # anything. It exited 1 rather than its own refusal code 2, so it looked
    # like a failing run instead of a broken instrument (C1: a step that dies
    # before doing its work must not be mistaken for the work failing).
    red_unnamed = []   # red, no crash -- but no check is named, so "red" is all we know
    try:
        for entry in MUTATIONS:
            name, target, old, new = normalise(entry)
            original = SOURCES[target]
            # Anchors are matched independent of line endings and the
            # replacement is written in the file's own ending, so no other
            # byte in the file changes. See tests/mutation_anchors.py.
            mutated = anchors.apply(original, old, new)
            if mutated is None:
                print(f"  ??  ANCHOR NOT FOUND  {name}")
                print(f"      {anchors.not_found_reason(original, old)}")
                print("      the mutation was NEVER APPLIED, so this check has "
                      "not been shown able to fail")
                unplaced.append(name)
                continue
            # RESTORE EVERY SOURCE FIRST, then apply exactly one fault.
            #
            # Without this, a mutation to routines.py ran on top of the
            # previous iteration's mutation to director.py, so the last two
            # faults were reported "caught" on failures belonging to the one
            # before them. A mutation harness that contaminates itself
            # reports a stronger suite than exists -- the same shape as the
            # checks it exists to catch, in the instrument that is supposed
            # to be the backstop.
            for f, text in SOURCES.items():
                write_raw(f, text)
            write_raw(target, mutated)
            code, stdout, failing = run_tests()
            frag = FRAGS.get(name)
            if code == 0:
                print(f"  SURVIVED  {name}")
                print("            the suite stayed GREEN with this fault in place")
                survived.append(name)
            elif frag:
                # ATTRIBUTED: caught only when the entry's OWN check fails.
                label, why = verdict.classify(code, failing, frag)
                if label == "caught":
                    print(f"  caught    {name}")
                    print(f"              {why}")
                elif label == "BROKEN":
                    print(f"  BROKEN    {name}")
                    print(f"            {why}")
                    broken.append(name)
                else:
                    print(f"  SURVIVED  {name}")
                    print(f"            {why}")
                    survived.append(name)
            else:
                # NOT YET ATTRIBUTED (see FRAGS). A crash is still never a catch.
                why = verdict.crash_reason(code, failing)
                if why:
                    print(f"  BROKEN    {name}")
                    print(f"            {why}")
                    broken.append(name)
                else:
                    fails = [ln.strip() for ln in stdout.splitlines()
                             if ln.strip().startswith("- ") or "[FAIL]" in ln]
                    print(f"  caught?   {name}   (UNATTRIBUTED: no check is named for it)")
                    for f in fails[:2]:
                        print(f"              {f}")
                    red_unnamed.append(name)
    finally:
        for f, text in SOURCES.items():
            write_raw(f, text)
            backups[f].unlink(missing_ok=True)

    print()
    code, stdout, _failing = run_tests()
    print("restored and re-run:", " | ".join(
        ln for ln in stdout.strip().splitlines() if "PASSED" in ln or "FAIL" in ln))
    if red_unnamed:
        print(f"\n{len(red_unnamed)} of {len(MUTATIONS)} faults went red but NAME NO CHECK "
              "(add them to FRAGS): a crash was ruled out, an unrelated failure was not.")
    if broken:
        print(f"\n{len(broken)} FAULT(S) BROKEN THE SUITE INSTEAD OF FAILING A CHECK:")
        for s in broken:
            print(f"  - {s}")
    if survived or unplaced or broken:
        if unplaced:
            print(f"\n{len(unplaced)} ANCHOR(S) NOT FOUND (never applied):")
            for s in unplaced:
                print(f"  - {s}")
        if survived:
            print(f"\n{len(survived)} FAULT(S) APPLIED AND NOT CAUGHT:")
            for s in survived:
                print(f"  - {s}")
        sys.exit(1)
    print(f"\nall {len(MUTATIONS)} faults caught - every check can fail")



def main():
    """One mutation run at a time, and it REFUSES rather than queueing.

    See `tests/worker_budget.py` for the two incidents. The slot is shared
    by every `mutate_*` script, because the 2026-10-17 one was two
    DIFFERENT scripts, each politely capped on its own.
    """
    try:
        with budget.exclusive():
            return _main()
    except budget.Busy as why:
        print(f"REFUSED: {why}")
        sys.exit(2)


if __name__ == "__main__":
    main()
