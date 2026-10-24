"""Reintroduce faults into COPIES of the tree and watch the introspection checks go red.

`tests/mutate_director.py` does this for the Director, `mutate_figures.py` for
the figure drills and `mutate_detector.py` for the detector's size gate. This
does it for the introspection server, whose checks are all new and have
therefore only ever been seen green -- which on this project is worth nothing.
Four shipped checks could not fail the thing they checked, and the most
expensive was `test_coldstart` exiting after its first launch and returning 0
for as long as it existed (RULES C1, E3).

Each mutation copies `src`, `tests` and `tools` into
`tests/_out/mutate_introspect/<n>/` -- never the real tree, never his files --
changes ONE thing, runs `test_introspect.py` there, and requires a `[FAIL]`
line that NAMES the property. Any failure will not do: a mutation "caught" by
a timing wobble in the cost section is not caught at all, it is noise counted
as evidence.

    python tests/mutate_introspect.py          all of them
    python tests/mutate_introspect.py 1 3      just these (by number)
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_env import copy_scratch_tree, scratch_env      # noqa: E402  (one table of asset overrides)
import mutation_anchors as anchors  # noqa: E402
import mutation_verdict as verdict  # noqa: E402  (the ONE judgement of a result)
import worker_budget as budget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "_out" / "mutate_introspect"
PY = sys.executable

RO = "src/lsa/introspect/readonly.py"
SRV = "src/lsa/introspect/server.py"
INIT = "src/lsa/introspect/__init__.py"
GUI = "src/lsa/introspect/gui.py"
VIEW = "src/lsa/introspect/view.py"

# (what is broken, file, old text, new text, text a failing line must contain)
M = [
    # --- constraint 1: no camera control write is reachable -------------
    ("a camera write is put back on the read-only list",
     RO, '    "config",             # CaptureConfig: index, backend, format, requested fps',
     '    "config",\n    "set_control",',
     "the Reader refuses it"),
    ("the Reader stops refusing and just forwards everything",
     RO, "        if name not in ok:",
     "        if False:",
     "the Reader refuses it"),
    ("the route to cap.set() is reopened (_post, one indirection away)",
     RO, '    "_running",',
     '    "_running",\n    "_post",',
     "the Reader refuses it"),

    # --- constraint 5: localhost only -----------------------------------
    ("the loopback guard is dropped",
     SRV, "        if host not in LOCAL_ONLY:",
     "        if False:",
     "refuses to bind"),

    # --- off by default -------------------------------------------------
    ("the server is on unless switched off",
     INIT, '    if not raw or raw in ("0", "off", "no", "false"):\n        return 0',
     '    if not raw:\n        return DEFAULT_PORT',
     "no flag and no env var"),
    ("an unparseable value is swallowed instead of raising",
     INIT, "    return int(raw)",
     "    try:\n        return int(raw)\n    except ValueError:\n        return 0",
     "RAISES rather than quietly doing nothing"),

    # --- off the hot path -----------------------------------------------
    # NOT "make ShotTap a plain object": that fault survived, because
    # PySide6 queues to a plain callable using the SENDER's affinity and the
    # capture thread's QObject lives on the GUI thread. The real fault is
    # asking for the wrong connection type, which is why the tap now asks for
    # one explicitly.
    ("the shot ring is filled by a DIRECT connection, inside the capture loop",
     GUI, "        signal.connect(self.record, Qt.QueuedConnection)",
     "        signal.connect(self.record, Qt.DirectConnection)",
     "filled on the GUI thread"),
    ("the Qt bridge runs the work on the CALLER's thread",
     GUI, "        self._submit.connect(self._run, Qt.QueuedConnection)",
     "        self._submit.connect(self._run, Qt.DirectConnection)",
     "ran on the Qt thread"),
    ("the bridge stops counting the waits it gave up on",
     GUI, "            self.timeouts += 1",
     "            pass",
     "counts the timeout"),

    # --- /score: the measurements, not a picture ------------------------
    # `{} or X` evaluates to X, so the first version of this mutation changed
    # nothing at all and was reported as a surviving fault. Drop one term
    # instead, which is the failure that would really happen.
    ("one of the four score terms stops being reported",
     VIEW, '                "raw_chroma": _num(int(raw[wy, wx])),',
     '                "raw_chroma": None,',
     "all four terms are present as numbers"),
    ("the amplification is rounded to a whole number (0.3 gain -> 3.33x)",
     VIEW, '                "amplification": _num(float(amp[wy, wx])),',
     '                "amplification": _num(int(amp[wy, wx])),',
     "amplification at a 0.3-gain pixel"),
    # The four-point mystery this endpoint exists to not have: the detector
    # subtracts the model and THEN updates it, so recomputing against the
    # model as read is wrong by about alpha x score at a bright pixel.
    ("the score is recomputed against the model AFTER its update",
     VIEW, "                bg_before = (bg - rate * blurred.astype(np.float32)) / (1.0 - rate)",
     "                bg_before = bg",
     "matches the one recomputed here"),
    ("the frame and the score are no longer checked for being the same frame",
     VIEW, "        if idx0 == idx1:\n            return frame_c, final_c, bg_c, recip_c, idx0, True",
     "        if True:\n            return frame_c, final_c, bg_c, recip_c, idx0, False",
     "the frame did not advance"),
    ("the raw blob list loses the reasons a blob would be dropped",
     VIEW, '                "would_be_dropped_because": drops,',
     '                "would_be_dropped_not_here": drops,',
     "each blob carries would_be_dropped_because"),

    # --- /settings: effective BESIDE saved ------------------------------
    ("/settings cannot find the saved file, and says nothing about it",
     VIEW, '    raw = os.environ.get("LSA_SETTINGS")',
     '    raw = None',
     "the saved value is read from disk"),

    # --- /drawn: the measurement that found the signal meter -----------
    ("red excess is measured against green alone, not the stronger channel",
     VIEW, "        red_excess = cv2.subtract(r, cv2.max(g, b))",
     "        red_excess = cv2.subtract(r, b)",
     "worst red excess on the screen is measured"),
    ("the sampled grab is not scaled back up, so boxes are in the wrong units",
     VIEW, "        sx, sy = full_w / float(w), full_h / float(h)",
     "        sx, sy = 1.0, 1.0",
     "located in SCREEN pixels"),

    # --- POST /key: the real path --------------------------------------
    ("the key is reported as sent without ever being posted",
     VIEW, "            self._bridge.run(lambda: post_key(win, key, _text_for(key)))",
     "            self._bridge.run(lambda: None)",
     "really reached the window's own handler"),

    # --- it writes nothing ---------------------------------------------
    ("something in the package starts writing a file",
     VIEW, "def _pid() -> int:",
     'def _leak(p):\n    Path(p).write_text("x")\n\n\ndef _pid() -> int:',
     "no file-writing call"),

    # --- a wrong name must be loud (RULES C8) --------------------------
    # The five names `_audio()` read wrongly for weeks, in the three shapes
    # the fault can come back in. The first is the original fault; the
    # second is the same name through a default again; the third is a
    # CORRECT name read through a default, which changes no output at all and
    # is why the check must be on the construct and not only on the values.
    ("the heard-not-seen count is read from a name that does not exist",
     VIEW, '            "heard_not_seen": _num(corr.heard_not_seen),',
     '            "heard_not_seen": _num(corr.heard_only),',
     "answers while the audio objects are real"),
    ("the heard-not-seen count goes back to a getattr default on the wrong name",
     VIEW, '            "heard_not_seen": _num(corr.heard_not_seen),',
     '            "heard_not_seen": _num(getattr(corr, "heard_only", None)),',
     "the heard-versus-seen counts that were null for weeks"),
    ("a CORRECT name is read through a getattr default (outputs unchanged)",
     VIEW, '            "mic_error": str(listener.error),',
     '            "mic_error": str(getattr(listener, "error", "")),',
     "through NO getattr default"),
    ("_call quietly returns None for a method that does not exist",
     VIEW, "    fn = getattr(obj, name)\n    try:",
     "    fn = getattr(obj, name, None)\n    if fn is None:\n        return None\n    try:",
     "through NO getattr default"),
    ("the microphone's coupling is dropped from /state",
     VIEW, '            "coupling": _jsonable(listener.coupling_report()),',
     '            "coupling_dropped": 0,',
     "the coupling is served"),
]


SUITE = "test_introspect.py"
SUITE_TIMEOUT_S = 240


def judge(rc: int, text: str, frag: str) -> tuple[str, str]:
    """('caught' | 'MISSED' | 'BROKEN', why). tests/mutation_verdict.py is the ONE
    judgement. The cost section measures real timings, so a loaded machine can
    produce an unrelated red line; it is not "caught" (it names nothing), and a
    crashed section is BROKEN, never caught."""
    return verdict.classify(rc, text, frag)


def _scratch_tree(d: Path) -> None:
    copy_scratch_tree(ROOT, d)  # the ONE copy (tests/asset_env.py), see its docstring


def _run_suite(d: Path, test: str = SUITE) -> tuple[int, str]:
    env = scratch_env(d)        # every override points at the scratch COPY, not at his folders
    try:
        r = subprocess.run([PY, str(d / "tests" / test)],
                           cwd=str(d), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=SUITE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return -9, f"TIMED OUT after {SUITE_TIMEOUT_S}s"
    return r.returncode, r.stdout + "\n" + r.stderr


def _baseline() -> list[str]:
    """The suite every entry names, unmutated (2026-10-23: none was run)."""
    d = OUT / "baseline"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _scratch_tree(d)
    try:
        return verdict.baseline([SUITE], lambda s: _run_suite(d, s))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run(i: int, m: tuple) -> tuple:
    """(name, label, why), label one of caught / MISSED / BROKEN / UNPLACED."""
    name, rel, old, new, frag = m
    d = OUT / str(i)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _scratch_tree(d)
    p = d / rel
    raw = p.read_bytes().decode()
    mutated = anchors.apply(raw, old, new)
    if mutated is None:
        shutil.rmtree(d, ignore_errors=True)
        return name, "UNPLACED", (f"MUTATION DID NOT APPLY: {old[:60]!r}: "
                                  + anchors.not_found_reason(raw, old))
    p.write_bytes(mutated.encode())
    rc, text = _run_suite(d)
    shutil.rmtree(d, ignore_errors=True)
    label, why = judge(rc, text, frag)
    return name, label, why


def _main() -> int:
    pick = [int(a) for a in sys.argv[1:]] or list(range(len(M)))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    red = _baseline()
    if red:
        print("BASELINE IS RED: the suite fails with NO fault applied, so "
              "'caught' would mean nothing. Fix it first.")
        for ln in red:
            print("  " + ln)
        shutil.rmtree(OUT, ignore_errors=True)
        return 2
    # Three at a time, not eight: section 11 of the test measures what the
    # server costs the capture loop, and eight parallel copies of that make
    # the machine the instrument.
    with cf.ThreadPoolExecutor(budget.workers(3)) as ex:
        res = list(ex.map(lambda i: (i, run(i, M[i])), pick))
    bad = 0
    for i, (name, label, why) in res:
        bad += label != "caught"
        print(f"  {label:<8} {i:2d} {name}")
        print(f"             -> {why[:220]}")
    shutil.rmtree(OUT, ignore_errors=True)
    broken = sum(1 for _i, r in res if r[1] == "BROKEN")
    print(f"{len(res) - bad} of {len(res)} introspection faults caught "
          f"in {time.perf_counter() - t0:.0f}s"
          + (f" ({broken} BROKEN: the suite crashed rather than failing the named check)"
             if broken else "")
          + ("" if not bad else " -- A CHECK THAT CANNOT FAIL IS NOT A CHECK"))
    return 1 if bad else 0



def main() -> int:
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
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
