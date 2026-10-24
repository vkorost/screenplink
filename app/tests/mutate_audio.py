"""Reintroduce each microphone fault and watch test_audio_own go red.

RULES E3 / C1: a check that has only ever been seen green proves nothing. This
does for `src/lsa/audio/{bang,listener,mixer}.py` -- and for the one place in
`src/lsa/tools/camera_lab.py` where a second path to the speakers once stood --
what `mutate_ghost.py` does for the ghost score: it copies `src` and the one
suite (and exactly the tests/ helpers that suite imports) into a SCRATCH tree
under tests/_out (never the real tree, never his files), changes ONE thing, runs
`test_audio_own.py` there and classifies the outcome:

    caught   the suite ran to its summary, failed, AND a failing line names the
             check written against this fault (not merely "something failed")
    MISSED   the suite stayed green, or failed only on checks unrelated to the
             fault -- the check that should notice cannot
    BROKEN   the suite CRASHED, timed out, or never reported. NOT caught: a crash
             cannot be told from a broken harness (2026-10-19: twelve crashes
             were read as twelve caught faults)
    UNPLACED the anchor did not match its target exactly once, so the fault was
             never applied (tests/test_rules.py section M also fails on this)

IT REFUSES to run on a red baseline, and it runs ONE FAULT AT A TIME, in one
process, behind the one mutation slot (`worker_budget`). The stash this was
ported from fanned out to eight workers by itself; on 2026-10-19 and 2026-10-17
that took his machine to 100% across all sixteen cores. Nothing here sizes itself
from the machine.

EACH FAULT NAMES THE SECTION(S) THAT SHOULD CATCH IT. The full suite is run once,
as the baseline; each fault then runs only its own section(s) (`LSA_SECTIONS`), so
forty faults cost seconds each and not the whole file forty times. Narrowing is
also the sharper claim: a fault is "caught" only if the check written for it
fails, in the section written for it.

A fault whose defence is TWO independent layers lists parallel lists of files,
anchors and replacements: removing either layer alone changes nothing (that is
depth, not a gap) and removing both is the fault.

    python tests/mutate_audio.py            all of them
    python tests/mutate_audio.py 3 7        just these (by number)
    python tests/mutate_audio.py --self-check
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
from asset_env import scratch_env  # noqa: E402  (one table of asset overrides, RULES section Q)
import mutation_anchors as anchors  # noqa: E402
import worker_budget as budget  # noqa: E402
# The bytecode-safe child environment, the stale-bytecode self-check and the
# classifier are mutate_ghost's, written for exactly this hazard on 2026-10-21 and
# not copied here: a copy would drift from the instrument it copies.
import mutate_ghost as ghost  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "_out" / "mutate_audio"
PY = sys.executable
TEST = "test_audio_own.py"
BASELINE_TIMEOUT_S = 600      # the whole suite, once
FAULT_TIMEOUT_S = 240         # one section or two

BANG = "src/lsa/audio/bang.py"
LIS = "src/lsa/audio/listener.py"
MIX = "src/lsa/audio/mixer.py"
LAB = "src/lsa/tools/camera_lab.py"

# (what is broken, file(s), old text(s), new text(s), sections that must catch it,
#  text a failing line must contain)
# `old` must match its file EXACTLY ONCE. The fragment is matched case-insensitively
# against the [FAIL] lines and is copied from the label of the check written for it.
M = [
    # -- 1. a bang that never reached a speaker -------------------------------------
    ("a bang is registered even when play() refused it",
     BANG, "    if not ok:\n        listener.expect_own_sound(np.zeros(0, np.float32), 0.0, 0.0,",
     "    if False:\n        listener.expect_own_sound(np.zeros(0, np.float32), 0.0, 0.0,",
     "1", "dead output stream is reported as not played"),
    ("a dead stream is not noticed: play() True is taken as heard",
     BANG, "    elif not mixer.audible:", "    elif False:",
     "1", "dead output stream is reported as not played"),
    ("Mixer.audible always says yes",
     MIX, "            return bool(self._stream.active)", "            return True",
     "1", "dead output stream is reported as not played"),
    ("a reading need not stand clear of the room's noise",
     LIS, "            if seen < self.COUPLING_SNR_MIN * floor:", "            if False:",
     "1", "three bangs into a silent endpoint teach NO coupling"),
    # -- 2. a new endpoint is a new question ---------------------------------------
    ("a reopened output keeps the old coupling",
     LIS, "            if changed:\n                self.reset_coupling(",
     "            if False:\n                self.reset_coupling(",
     "2", "next bang on a new stream finds the coupling forgotten"),
    ("the bang forgets to say which output stream it was played on",
     BANG, "        played=True, output_id=mixer.epoch, gain=mixer.master_gain,",
     "        played=True, output_id=None, gain=mixer.master_gain,",
     "2", "next bang on a new stream finds the coupling forgotten"),
    ("the epoch never moves when the output is reopened",
     MIX, "            self.available = True\n            self.epoch += 1",
     "            self.available = True",
     "2", "opening a stream moves the epoch"),
    ("starting the listener on another microphone keeps the old coupling",
     LIS, "        # A different microphone couples differently to the speakers.\n"
          "        self.reset_coupling()",
     "        # A different microphone couples differently to the speakers.",
     "2", "starting the listener forgets the coupling"),
    # -- 3. what is and is not a measurement ---------------------------------------
    ("a coupling of exactly 0.0 counts as a measurement",
     LIS, "        return c if c > 0.0 else None", "        return c",
     "3", "three zero readings are not a known coupling"),
    ("a bang neither discounted nor muted is reported as discounted",
     LIS, "                mode = OWN_UNPROTECTED\n            fresh.mode = mode",
     "                mode = OWN_DISCOUNTED\n            fresh.mode = mode",
     "3", "reported as UNPROTECTED"),
    ("the fallback mute is never applied (the app hears its own bang)",
     LIS, "            elif fallback_mute_s > 0.0:", "            elif False:",
     "3", "never unprotected"),
    ("one reading is enough to believe a coupling",
     LIS, "    COUPLING_NEEDED = 3", "    COUPLING_NEEDED = 1",
     "4", "needing three readings keeps it from teaching a coupling"),
    # -- 5. the teacher ------------------------------------------------------------
    ("the probe is a single 1 kHz tone, not a sweep",
     MIX, "def synth_probe(samplerate: int = 48000, dur: float = 0.12, f0: float = 300.0,\n"
          "                f1: float = 6000.0, level: float = 0.5,",
     "def synth_probe(samplerate: int = 48000, dur: float = 0.12, f0: float = 1000.0,\n"
     "                f1: float = 1000.0, level: float = 0.5,",
     "5b", "the sweep never reads below 0.9x"),
    ("the probe keeps going after the screen stops being the app's",
     BANG, "        if not may_play:\n            self._stop(", "        if False:\n            self._stop(",
     "5a", "stands down the instant"),
    ("the probe runs even when the coupling is already known and fresh",
     BANG, "        if self.listener.coupling is not None and not stale:", "        if False:",
     "5a", "nothing is taught when the coupling is already known"),
    ("a failed probe is retried in a loop",
     BANG, '        if self._tried_for == self._key() and self.state != "done":', "        if False:",
     "5a", "not retried in a loop"),
    ("the probe has no deadline",
     BANG, "        if now - self._started > self.DEADLINE_S:", "        if False:",
     "5a", "gives up at its deadline"),
    # -- 6. the noise floor is the room's, not the app's ---------------------------
    ("any bang level at all matters (the floor exclusion is binary)",
     LIS, "        return own > 0.0 and own >= self.OWN_SIGNIFICANCE * max(self.floor,\n"
          "                                                                 self.MIN_FLOOR)",
     "        return own > 0.0",
     "6", "the floor receives a real share of blocks"),
    ("the floor exclusion and the bar disagree: the floor uses 'any level'",
     LIS, "            ours = matters or now < self._muted_until",
     "            ours = own > 0.0 or now < self._muted_until",
     "6", "the floor receives a real share of blocks"),
    # -- 7. his own shots cannot teach the wrong thing -----------------------------
    ("a trigger click inside a bang's loud head is read as the bang",
     [LIS, LIS],
     ["    COUPLING_TOPK = 3",
      "            self._own_observe(now, level, ignore=clicked or ringing, hot=hot)"],
     ["    COUPLING_TOPK = 1",
      "            self._own_observe(now, level, ignore=False, hot=hot)"],
     "7", "keep the coupling honest"),
    ("the loud-head fraction is zero (the whole sample is its own head)",
     LIS, "    HEAD_FRAC = 0.20", "    HEAD_FRAC = 0.0",
     "7", "a string at 1.0s teaches the coupling by itself"),
    ("readings never expire",
     LIS, "    COUPLING_MAX_AGE_S = 600.0", "    COUPLING_MAX_AGE_S = 1e12",
     "7", "a reading an hour old is STALE"),
    ("a stale coupling is dropped instead of used (permanent deafness)",
     LIS, "        for pool, stale in ((fresh, False), (allr, True)):",
     "        for pool, stale in ((fresh, False),):",
     "7b", "STILL fully heard"),
    ("the teacher refuses a second probe even after its answer went stale",
     BANG, '        if self._tried_for == self._key() and self.state != "done":',
     "        if self._tried_for == self._key():",
     "7b", "probe is allowed again"),
    ("the teacher takes a stale coupling for a fresh one",
     BANG, "        if self.listener.coupling is not None and not stale:",
     "        if self.listener.coupling is not None:",
     "7b", "probe is allowed again"),
    # -- 8. the reporting ----------------------------------------------------------
    ("the reading line drops the volume",
     LIS, '        return (f"{head}, volume {vol}; filed', '        return (f"{head}; filed',
     "8", "description of the coupling carries the volume"),
    ("the listener's clock is no longer perf_counter by default",
     LIS, "refractory_ms: float = 120.0, clock=time.perf_counter) -> None:",
     "refractory_ms: float = 120.0, clock=time.monotonic) -> None:",
     "8", "by default the listener's clock is time.perf_counter"),
    # -- 9. a bang louder than predicted -------------------------------------------
    ("a bang louder than predicted is never noticed",
     LIS, "    LOUD_BLOCKS = 12", "    LOUD_BLOCKS = 10 ** 9",
     "9", "the room 2x louder: at most ONE invented pulse"),
    ("a single trigger click in a bang's head is read as a louder room",
     [LIS, LIS], ["    LOUD_BLOCKS = 12", "    LOUD_FRACTION = 0.25"],
     ["    LOUD_BLOCKS = 1", "    LOUD_FRACTION = 0.0"],
     "9", "never read as a louder room"),
    ("a louder room does not put the flat mute back",
     LIS, "        if until > self._muted_until:", "        if False:",
     "9", "the room 2x louder: at most ONE invented pulse"),
    ("the ceiling is still trusted after a bang read louder than it",
     LIS, "        self._louder = True\n        self._resets += 1",
     "        self._louder = False\n        self._resets += 1",
     "9", "a room past the CEILING is noticed on the first bang"),
    ("protection_for ignores that the room was louder than predicted",
     LIS, "        if (not self._louder\n"
          "                and out_peak * self.COUPLING_CEILING < self._click_bar()):",
     "        if (out_peak * self.COUPLING_CEILING < self._click_bar()):",
     "9", "a room past the CEILING is noticed on the first bang"),
    # -- 10. the fallback ----------------------------------------------------------
    ("the fallback mutes flatly even where the bang cannot reach the bar",
     LIS, "    COUPLING_CEILING = 2.0", "    COUPLING_CEILING = 1e9",
     "10", "volume 0.05, coupling unknown"),
    ("the fallback never mutes, even where the bang could reach the bar",
     LIS, "    COUPLING_CEILING = 2.0", "    COUPLING_CEILING = 0.0",
     "10", "a bang that could reach the bar is still muted"),
    # -- 11. a discounted bang must not make a shot look like a phantom ------------
    ("audio_at ignores that a discounted bang hid the click",
     LIS, "            deaf = max(deaf, self._masked_for(t))", "            deaf = deaf",
     "11", "a click the discounted bang hid is always reported as hidden"),
    ("_masked_for never finds the bang loud enough to matter",
     LIS, "        if level(t) < bar:", "        if True:",
     "11", "a click the discounted bang hid is always reported as hidden"),
    ("the flat mute is no longer recorded for audio_at",
     LIS, "        self._mutes.append((now, self._muted_until))", "        pass",
     "11", "flat mute is what audio_at reports"),
    # -- 12. one path to the room, and what is registered --------------------------
    ("the console's Test button calls the mixer directly again",
     LAB, "        b_test.clicked.connect(self._play_test_sound)",
     "        b_test.clicked.connect(lambda: self.mixer.play("
     "self.cmb_sound.currentText() or \"synth shot\"))",
     "12", "no second path from the app"),
    ("the gunshot path plays through the mixer as well as through the one path",
     LAB, "                              source=SOURCE_EFFECT if dry else SOURCE_BANG)",
     "                              source=SOURCE_EFFECT if dry else SOURCE_BANG)\n"
     "        self.mixer.play(sound)",
     "12", "no second path from the app"),
    ("an effect teaches the coupling like a gunshot",
     LIS, '        if o.source == "effect":', "        if False:",
     "12", "never teaches the coupling and is not a refusal either"),
    ("the fallback mute stops following the volume",
     BANG, "mixer.loud_duration(sound, gain=mixer.master_gain)", "mixer.loud_duration(sound)",
     "12", "fallback follows the volume"),
    ("the flat mute is applied but not recorded",
     LIS, "        self._mutes.append((now, self._muted_until))", "        pass",
     "12", "the mute is on record for audio_at"),
    ("a sample named with a leading underscore is offered as a gunshot",
     MIX, 'if not n.startswith((self.VOICE_PREFIX, self.RESERVED_PREFIX, "_")))',
     'if not n.startswith((self.VOICE_PREFIX, self.RESERVED_PREFIX)))',
     "12", "registered but not offered"),
]


def _edits(m) -> "list[tuple[str, str, str]]":
    """[(file, old, new)] for one fault, whether it was written as one edit or two."""
    _name, rel, old, new = m[0], m[1], m[2], m[3]
    if isinstance(rel, list):
        return list(zip(rel, old, new))
    return [(rel, old, new)]


def targets() -> list[str]:
    """Every repo-relative file a mutation touches (importable, no side effects)."""
    return sorted({rel for m in M for rel, _o, _n in _edits(m)})


def child_env() -> dict:
    """The environment every child runs in: the asset overrides (so the suite finds
    his 9mm sample in a tree that holds only `src` and `tests`), the offscreen
    platform, and NO bytecode from any source (a prefix cache is shared across
    trees, which is how a previous mutant's code gets served to the next one)."""
    env = scratch_env(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPYCACHEPREFIX", None)
    return env


def judge(rc: int, text: str, frag: str) -> tuple[str, str]:
    """The shared judgement (tests/mutation_verdict.py), via mutate_ghost."""
    return ghost.judge(rc, text, frag)


def run_suite(tree: Path, env: dict, sections: str, timeout: int) -> tuple[int, str]:
    """(returncode, output). A hang is a BROKEN run, so it is bounded. No bytecode
    is read or written (`-B` and the child environment: either alone is a single
    point of failure)."""
    env = dict(env)
    if sections:
        env["LSA_SECTIONS"] = sections
    else:
        env.pop("LSA_SECTIONS", None)
    cmd = [PY, "-B", str(tree / "tests" / TEST)]
    try:
        r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return -9, f"TIMED OUT after {timeout}s"
    return r.returncode, r.stdout + "\n" + r.stderr


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
    for name_ in ghost.suite_files(TEST):
        shutil.copy2(ROOT / "tests" / name_, tree / "tests" / name_)
    env = child_env()

    try:
        # THE INSTRUMENT FIRST: prove, on every run, that a child cannot be served
        # a previous mutant's compiled code -- and that the proof can fail.
        ok, notes = ghost.self_check()
        for ln in notes:
            print("  self-check: " + ln)
        if not ok:
            print("INSTRUMENT UNRELIABLE: no tally from this runner can be trusted.")
            return 2
        if only_self_check:
            print("self-check passed")
            return 0

        # PRECONDITION. "Caught" means the suite went red BECAUSE of the fault, so
        # the unmutated suite must be green, ALL of it.
        rc, text = run_suite(tree, env, "", BASELINE_TIMEOUT_S)
        if ghost._stray_bytecode(tree):
            print("BASELINE WROTE BYTECODE into the scratch tree despite -B and "
                  "PYTHONDONTWRITEBYTECODE: a later mutant could be served it.")
            return 2
        red = ghost.verdict.baseline_problem(rc, text)
        if red:
            print("BASELINE IS RED: test_audio_own.py fails with NO fault applied, so "
                  "'caught' would mean nothing. Fix the suite first.")
            print("  " + red)
            for ln in [ln.strip() for ln in text.splitlines() if "[FAIL]" in ln][:10] \
                    or [ln for ln in text.splitlines() if ln.strip()][-5:]:
                print("  " + ln[:200])
            return 2

        pristine = {rel: (tree / rel).read_bytes() for rel in targets()}
        results = []
        for i in pick:
            m = M[i]
            name, sections, frag = m[0], m[4], m[5]
            mutated: dict[str, str] = {}
            unplaced = ""
            for rel, old, new in _edits(m):
                base = mutated.get(rel, pristine[rel].decode())
                out = anchors.apply(base, old, new)
                if out is None:
                    unplaced = f"{old[:60]!r}: " + anchors.not_found_reason(base, old)
                    break
                mutated[rel] = out
            if unplaced:
                results.append((i, name, "UNPLACED", unplaced))
                continue
            try:
                ghost._purge_bytecode(tree)       # belt: nothing compiled survives into this run
                for rel, text_ in mutated.items():
                    (tree / rel).write_bytes(text_.encode())
                rc, text = run_suite(tree, env, sections, FAULT_TIMEOUT_S)
                stray = ghost._stray_bytecode(tree)
            finally:
                for rel in mutated:
                    (tree / rel).write_bytes(pristine[rel])    # one fault at a time, always
            if stray:
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
        print(f"{len(results) - bad} of {len(results)} microphone faults caught in "
              f"{time.perf_counter() - t0:.0f}s"
              + ("" if not bad else " -- A CHECK THAT CANNOT FAIL IS NOT A CHECK"))
        return 1 if bad else 0
    finally:
        shutil.rmtree(OUT, ignore_errors=True)
        shutil.rmtree(ghost.OUT, ignore_errors=True)    # self_check's scratch lives there


def main() -> int:
    """One mutation run at a time, and it REFUSES rather than queueing.

    See `tests/worker_budget.py` for the two incidents. The slot is shared by
    every `mutate_*` script.
    """
    try:
        with budget.exclusive():
            return _main()
    except budget.Busy as why:
        print(f"REFUSED: {why}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
