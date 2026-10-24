"""The rules that can execute, executing.

`RULES.md` is 579 lines of prose, and prose is the wrong home for about a
third of it. Two reasons, both demonstrated on this project:

  1. I cannot reliably re-read my own prose. While rewriting RULES.md to be
     more thorough I DROPPED SIX RULES that were in my own first draft,
     three of them describing damage done that same day. It was caught by
     diffing the headings against the session, not by reading the file.
  2. A rule stated once and broken in five places is the dominant failure
     mode here. Prose cannot grep. This can.

So: every rule below was prose, is now a check, and fails the build. Each
carries the incident it came from, because a rule without its evidence gets
re-argued and then re-broken.

AST, not regex, wherever the rule is about code structure -- a regex over
source text passes when the code is renamed into equivalence and fails when
it is refactored into correctness, which is backwards from what a refactor
needs. Fable's review of this codebase made that point about the eleven
source-text greps in `test_feedback.py`; the structural ones moved here.

Run by `run_tests.bat`, and by the pre-commit hook, so "I forgot" is not
available.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import author_files as _af  # one place resolves the files he keeps private
import owned_files as _of  # the keys the isolation deliberately rewrites

import ast
import io
import os
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "lsa"
sys.path.insert(0, str(ROOT / "src"))   # section L imports the real savers

CHECKS = 0
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILED.append(what + (f"  --  {detail}" if detail else ""))
        print(f"  FAIL  {what}")
        if detail:
            print(f"        {detail}")


def section(n: str) -> None:
    print(f"\n=== {n} ===")


NOT_RUN: list[str] = []


def code_only(path: Path) -> str:
    """A file with its COMMENTARY removed, so a note recording that something
    was deleted is not read as evidence that it came back.

    Written 2026-10-23 because the `ui/`-is-gone scan below reported two hits
    and BOTH WERE PROSE: a docstring line in `artwork/ui.py` saying the folder
    and its override were removed, and the comment above the check itself. A
    plain text scan cannot tell a live override from its own obituary, and a
    check that fires on its own documentation is one somebody edits into
    silence. Five hollow checks on this project started that way.

    Single-quoted strings are KEPT, deliberately: `os.environ["LSA_UI"]` is a
    live read and must still be caught. Only comments and triple-quoted blocks
    go. An unparseable file is scanned WHOLE rather than skipped -- a scan that
    quietly covers less than it claims is the very fault this guards against.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".py":
        return "\n".join(
            ln for ln in text.splitlines()
            if not ln.lstrip().lower().startswith(("#", "rem ", "::")))
    kept = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if (tok.type == tokenize.STRING
                    and tok.string.lstrip("rbfuRBFU")[:3] in ('"""', "'''")):
                continue
            kept.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text
    return "\n".join(kept)


def author_file(rel: str, env: str) -> Path | None:
    """This suite's half of `tests/author_files.py`: resolve, or report.

    The resolver decides nothing about verdicts, so the reporting lives here --
    a wrong pointer is a FAILURE, an absent file is a VISIBLE skip, and neither
    can quietly become a pass.
    """
    found, failure, skip = _af.author_file(ROOT, env)
    if failure:
        check(False, f"{env} names a file that exists", failure)
    elif skip:
        NOT_RUN.append(f"{rel} -- {skip}")
    return found


def py_files(*under: str) -> list[Path]:
    out: list[Path] = []
    for u in under:
        base = SRC / u if u else SRC
        out.extend(sorted(base.rglob("*.py")) if base.is_dir() else [])
    return out


def read(p: Path) -> str:
    return io.open(p, encoding="utf-8", newline="").read()


def tree(p: Path) -> ast.Module:
    return ast.parse(read(p), filename=str(p))


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def enclosing_function(mod: ast.Module) -> dict[int, str]:
    """Line number -> the function it is inside. For naming offenders."""
    owner: dict[int, str] = {}
    for node in ast.walk(mod):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                line = getattr(inner, "lineno", None)
                if line is not None and line not in owner:
                    owner[line] = node.name
    return owner


ALL_PY = py_files("")
ROUTINE_FILES = [SRC / "routines.py"]
DISPLAY = py_files("display") + py_files("modes")


# ---------------------------------------------------------------------------
section("A. the Director owns sequencing: nothing may reach around it")
# ---------------------------------------------------------------------------
#
# RULES Part E, and the measured reason the Director exists. On 2026-10-15
# the central exclusion list `SCREEN_OWNERS` was consulted by ONE of the six
# routines that needed it; the other five each hand-rolled a different
# subset of the flags, and the live gate failed on exactly that.
#
# A routine that schedules its own timer, or calls the capture thread's
# measurement entry points itself, is outside the one run slot -- which is
# the entire guarantee. So it is not allowed, and this is where that is
# enforced rather than remembered.

for path in ROUTINE_FILES:
    mod = tree(path)
    owner = enclosing_function(mod)
    offenders = []
    for node in ast.walk(mod):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr in ("singleShot", "start", "stop") \
                and isinstance(node.value, ast.Name) \
                and node.value.id in ("QTimer", "QtCore"):
            offenders.append(f"{owner.get(node.lineno, '?')}:{node.lineno}")
    check(not offenders,
          f"{rel(path)} schedules nothing itself",
          f"{offenders} -- a routine that arms its own timer is outside the "
          f"run slot, and its timer cannot be cancelled at shutdown")

    banned_calls = ("request_still", "start_noise_calibration",
                    "start_room_measurement", "set_control",
                    "start_exposure_search", "processEvents")
    offenders = []
    for node in ast.walk(mod):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in banned_calls):
            offenders.append(f"{node.func.attr} at "
                             f"{owner.get(node.lineno, '?')}:{node.lineno}")
    check(not offenders,
          f"{rel(path)} touches the camera only through effects",
          f"{offenders} -- these must be yielded (Still, Floor, Room, "
          f"Control) so they are serialised, deadlined and counted")

# And the Director itself must stay free of Qt, or the clock cannot be
# swapped for a virtual one and the harness is back to jittering real
# timers and sampling a few of the orderings that exist.
d_src = read(SRC / "director.py")
# At COLUMN ZERO only. An import inside a method is how `QtClock` gets its
# QTimer without making the module depend on Qt, which is the whole point;
# the first version of this check matched the indented one too and failed a
# file that was right.
module_level_qt = [
    ln for ln in d_src.splitlines()
    if re.match(r"^(from|import)\s+PySide6", ln)
]
check(not module_level_qt,
      "the Director imports no Qt at module level",
      f"{module_level_qt} -- the injected clock is what makes a launch "
      f"deterministic in tests")


# ---------------------------------------------------------------------------
section("B. one clock")
# ---------------------------------------------------------------------------
#
# RULES B4. Three clocks were in use for the same deadlines --
# `time.time()`, `time.monotonic()` and `perf_counter()` -- and on Windows
# none of them share an epoch, so a deadline was either already past or
# unreachable. It was unreachable: the READY banner never cleared, and its
# 59% veil hid every bullet hole he shot for an evening. He reported no
# holes; I blamed the arcade modes, without measuring.
#
# The prose rule cannot distinguish a deadline from a log timestamp. This
# can, because anything that WAITS now goes through the Director, and
# everything drawn compares against `ctx.now`.

for path in DISPLAY + ROUTINE_FILES:
    src = read(path)
    bad = []
    for i, line in enumerate(src.splitlines(), 1):
        if re.search(r"\btime\.(time|monotonic)\s*\(", line):
            bad.append(f"{rel(path)}:{i}")
    check(not bad, f"{rel(path)} uses one clock",
          f"{bad} -- deadlines compared against ctx.now must be set with "
          f"perf_counter(); mixing epochs made one unreachable")


# ---------------------------------------------------------------------------
section("C. read attributes that exist")
# ---------------------------------------------------------------------------
#
# RULES C8, twice over and days apart:
#
#   * `stats.last_shot_at` never existed, so `getattr(..., 0.0)` returned
#     0.0 and a guard never fired ONCE for days while looking exactly like
#     protection.
#   * `capture.stats` does not exist -- it is `_stats` -- and reading it
#     raised inside a Qt slot, which stopped a self test mid-run with
#     nothing said at all.
#
# `getattr(self, "_private", default)` on the app's own attributes is the
# construct that hides both. An attribute an object owns should be declared
# and assigned unconditionally; if it is optional, that is a design
# decision to make explicitly, not by defaulting.

# A RATCHET, not a clean bill of health.
#
# There are two dozen of these already, written over a fortnight, and
# rewriting them all in one sitting would be a large unreviewed change to
# code that currently works -- for a fault that is latent in most of them.
# So the count is recorded here and may only go DOWN. A new one fails the
# build; removing one and lowering the number is the way through.
#
# Stating it as a budget is the honest version. Setting the threshold to
# whatever passes today and calling the rule enforced is what this project
# has done before, and it is how "every routine consults SOME flag" came to
# pass while five of six routines were unguarded.
GETATTR_BUDGET = {
    "src/lsa/tools/camera_lab.py": 23,
}
for path in ALL_PY:
    mod = tree(path)
    owner = enclosing_function(mod)
    bad = []
    for node in ast.walk(mod):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value.startswith("_")):
            bad.append(f"{owner.get(node.lineno, '?')}:{node.lineno} "
                       f"{node.args[1].value}")
    budget = GETATTR_BUDGET.get(rel(path), 0)
    check(len(bad) <= budget,
          f"{rel(path)}: getattr(self, '_x', default) at or below its budget "
          f"of {budget}",
          f"{len(bad)} found: {bad[:8]} -- each one hides a typo or a missing "
          f"assignment behind a plausible default. The budget may only go "
          f"down; lower the number in the same change that removes one.")
    if budget and len(bad) < budget:
        print(f"  (ratchet: {rel(path)} is down to {len(bad)} from a budget "
              f"of {budget} -- lower the budget in this change)")


# ---------------------------------------------------------------------------
section("D. nothing drawn may look like a laser, or compete with one")
# ---------------------------------------------------------------------------
#
# RULES B1 and B1b. The detector scores `red - max(green, blue)`:
#
#   static furniture        limit 30   the background absorbs it in seconds
#   anything that MOVES     limit  8   never absorbed
#   anything that APPEARS   limit  8   not absorbed when it matters
#
# Measured costs: the operator console's banner at red excess 80 produced
# phantom shots for an evening; the Escalating Ball at 24 produced 272
# detections in two minutes with a gunshot per frame; the verdict board at
# 24 took the noise floor from 0-2 to 16-18 and pushed the bar above the
# app's own test dots, so the self test detected 1 of 12, which produced a
# worse verdict, which stayed on screen and did it again.
#
# This scans EVERY hex colour in EVERY source file, not the ones that
# happen to get rendered: rendering tests only catch what they draw.

HEX = re.compile(r"#([0-9a-fA-F]{6})\b")
EXEMPT = ("#b00000",)      # the deliberate R-key worst-case background
worst: list[tuple[int, str, str]] = []
for path in ALL_PY:
    for i, line in enumerate(read(path).splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#"):
            continue
        # Skip comment-only lines: a colour named in a comment draws nothing.
        code = line.split("#", 1)[0] if line.strip().startswith("#") else line
        for m in HEX.finditer(code):
            hexv = "#" + m.group(1).lower()
            if hexv in EXEMPT:
                continue
            r = int(m.group(1)[0:2], 16)
            g = int(m.group(1)[2:4], 16)
            b = int(m.group(1)[4:6], 16)
            excess = r - max(g, b)
            if excess > 30:
                worst.append((excess, hexv, f"{rel(path)}:{i}"))
worst.sort(reverse=True)
check(not worst, "no colour anywhere exceeds the static red-excess limit of 30",
      f"{worst[:6]} -- the camera cannot tell the app's own UI from a laser")


# ---------------------------------------------------------------------------
section("E. nothing on the target screen is sized in pixels")
# ---------------------------------------------------------------------------
#
# RULES B5. Every font size in `layers.py` was chosen in a 1080p test window
# and read perfectly there. On the real panel -- 42 inches of 4K from
# several metres down the range -- the calibration instructions were half
# the apparent size and effectively unreadable. Qt does not help: point
# sizes track DPI, and a 42-inch 4K monitor is about 104 DPI, the same as
# the window they were designed in.

bad = []
for path in DISPLAY:
    for i, line in enumerate(read(path).splitlines(), 1):
        if re.search(r"\bmono\(\s*\d", line) or re.search(r"\bsans\(\s*\d", line):
            bad.append(f"{rel(path)}:{i}")
check(not bad, "no absolute font size on the target screen",
      f"{bad} -- use px(ctx, at_1080); this is invisible until it is on the "
      f"real hardware")


# ---------------------------------------------------------------------------
section("F. a test may never write anything the user owns")
# ---------------------------------------------------------------------------
#
# RULES C11, three times. A smoke test left `auto_connect: false` in
# `settings.local.json`, so the app stopped connecting on launch -- a worse
# bug than the one it prevented. The exposure persisted itself from the live
# control on exit, so every launch booted two stops darker than the last
# while the cause looked like a detector fault. And the cold-start harness
# saved its simulated 800x800 mapping over his real 3840x2160 one, taking
# the five sensitivity regions learned from his actual trigger pulls.

# The test is "does this test build something that COULD write", not "does
# it mention the filename". Mentioning `calibration.json` inside a
# source-text assertion writes nothing; constructing a real `CameraLab`,
# which loads settings, measures, calibrates and saves, can write both.
for path in sorted((ROOT / "tests").glob("test_*.py")):
    src = read(path)
    mod = tree(path)
    builds_lab = any(
        isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "CameraLab")
            or (isinstance(node.func, ast.Attribute)
                and node.func.attr == "CameraLab"))
        for node in ast.walk(mod))
    if not builds_lab:
        continue
    guarded = "OwnedFiles" in src or "fingerprint" in src
    check(guarded, f"{rel(path)} fingerprints the files he owns",
          "it constructs a real CameraLab, which can save a calibration and "
          "a settings file. Neutering the writer is a patch and a patch can "
          "be routed around -- the harness wrote his calibration through a "
          "path that had not been blocked. Use tests/owned_files.py.")
    # AND ITS LOG AND IMAGES MUST NOT LAND IN HIS.
    #
    # `test_ui_smoke` built a real CameraLab, published a verdict, and wrote
    # nine `GATE role=live screen=noscreen fps=0.0 writes=0` lines into
    # `camera_lab.log` -- the live log -- which `tools/status.py` reports
    # the newest of as "the last measurement on record". The role tag
    # invented to stop a measurement being attributed to the wrong process
    # was generating exactly that.
    #
    # And the harness overwrote `recordings/verdict_screen.png` with
    # pictures from a simulated 800x800 camera, destroying the evidence
    # from his last real run. The log had been separated by process months
    # -- days -- earlier; the images were the same problem and nobody went
    # looking for the second instance (RULES Part E).
    check("isolate_outputs" in src,
          f"{rel(path)} sends its log and images somewhere that is not his",
          "it constructs a real CameraLab, so it writes a log and saves "
          "verdict images. Call tests/owned_files.isolate_outputs() BEFORE "
          "importing camera_lab -- LOG_PATH and RECORDINGS are read at "
          "import time.")

# AND NO TEST THAT BUILDS A CameraLab MAY OPEN A SOUND DEVICE.
#
# `CameraLab.__init__` builds a Mixer, which opens an output stream on the
# system default endpoint: his monitor, with his speakers in it. The
# behavioural proof (a tripwire on the stream constructor) is
# tests/test_silent_audio.py; this is the structural half -- the one function
# every such test is already required to call must install the silent mixer,
# and the app must build its mixer through the seam.
_owned_src = read(ROOT / "tests" / "owned_files.py")
_iso = re.search(r"def isolate_outputs\(.*?\n(?=def )", _owned_src, re.S)
check(bool(_iso) and "_install_silent_audio()" in _iso.group(0),
      "isolate_outputs() installs the silent mixer",
      "every test that constructs a CameraLab calls it, so this is what keeps "
      "them from playing a gunshot out of the monitor")
_lab_src = read(SRC / "tools" / "camera_lab.py")
check("self.audio_factory(" in _lab_src
      and "audio_factory = staticmethod(default_mixer)" in _lab_src
      and "self.mixer = default_mixer(" not in _lab_src,
      "CameraLab builds its mixer through audio_factory",
      "a direct default_mixer() call cannot be replaced by a test")

# THE SAME FOR THE MICROPHONE. The cold-start harness opened
# `Microphone (HD Pro Webcam C920)` on every launch, so its audio behaviour
# depended on his room: a test using his hardware. Behavioural proof is
# tests/test_silent_audio.py (a tripwire on sd.InputStream); this is the
# structural half.
check(bool(_iso) and "_install_silent_audio()" in _iso.group(0)
      and "silent_listener" in _owned_src
      and "listener_mod.default_listener = silent_listener" in _owned_src,
      "isolate_outputs() installs the silent listener",
      "every test that constructs a CameraLab calls it, so this is what keeps "
      "them from opening his microphone")
check("self.listener_factory()" in _lab_src
      and "listener_factory = staticmethod(default_listener)" in _lab_src
      and "self.listener = ClickListener(" not in _lab_src,
      "CameraLab builds its listener through listener_factory",
      "a direct ClickListener() call cannot be replaced by a test")

# AND HIS SETTINGS FILE IS UNREACHABLE TO A TEST, NOT MERELY GUARDED.
#
# `LSA_LOG`, `LSA_RECORDINGS` and `LSA_CALIBRATION` were redirectable;
# `SETTINGS_PATH` was hard-coded, so the only defence was two hand-copied
# neuter lines pasted into three test files -- and a fourth test, written
# without them, wrote `settings.local.json` on the night of 2026-10-16
# (RULES C11: a patch can be routed around).
#
# Structural half: the module reads the override and `isolate_outputs()` sets
# it. Behavioural half, which is the one that matters: do what a test does --
# call `isolate_outputs()`, THEN import camera_lab -- and read back where all
# four owned paths now point. A structural check alone would pass a variable
# that was set but spelled differently from the one that is read.
check('SETTINGS_PATH = Path(os.environ.get("LSA_SETTINGS")' in _lab_src
      and "SETTINGS_PATH = PROJECT_ROOT" not in _lab_src,
      "camera_lab reads its settings path from LSA_SETTINGS",
      "a hard-coded settings path is the one a test cannot be kept away from")
check(bool(_iso) and 'os.environ["LSA_SETTINGS"]' in _iso.group(0),
      "isolate_outputs() redirects the settings file",
      "the three hand-copied 'neuter the saver' pairs must not be the only "
      "defence between a test and settings.local.json")
import json as _json
import subprocess as _subp
_probe = "\n".join([
    "import json, os, sys",
    f"sys.path.insert(0, {str(ROOT / 'tests')!r})",
    f"sys.path.insert(0, {str(ROOT / 'src')!r})",
    "import owned_files",
    "owned_files.isolate_outputs('rules_probe')",
    "from lsa.tools import camera_lab as c",
    "print('PATHS=' + json.dumps({k: str(getattr(c, k)) for k in "
    "('SETTINGS_PATH', 'CALIBRATION_PATH', 'LOG_PATH', 'RECORDINGS')}))",
])
_env = dict(__import__("os").environ, QT_QPA_PLATFORM="offscreen")
for _k in ("LSA_SETTINGS", "LSA_CALIBRATION", "LSA_LOG", "LSA_RECORDINGS"):
    _env.pop(_k, None)           # prove isolate_outputs sets them, not the caller
_r = _subp.run([sys.executable, "-c", _probe], capture_output=True, text=True,
               timeout=120, env=_env)
_paths = {}
for _ln in _r.stdout.splitlines():
    if _ln.startswith("PATHS="):
        _paths = _json.loads(_ln[len("PATHS="):])
_his = {"SETTINGS_PATH": ROOT / "config" / "settings.local.json",
        "CALIBRATION_PATH": ROOT / "config" / "calibration.json",
        "LOG_PATH": ROOT / "logs" / "camera_lab.log",
        "RECORDINGS": ROOT / "recordings"}
_reach = [k for k, v in _his.items()
          if not _paths.get(k) or Path(_paths[k]).resolve() == v.resolve()]
check(len(_paths) == 4 and not _reach,
      "after isolate_outputs(), none of the four owned paths is his",
      f"{_reach or 'no PATHS line'} still point at his file; "
      f"stderr: {_r.stderr.strip()[-300:]}")
_real = ROOT / "config" / "settings.local.json"
if _paths.get("SETTINGS_PATH") and _real.exists():
    _copy = Path(_paths["SETTINGS_PATH"])
    # CONTENT, not bytes. `isolate_outputs()` rewrites the keys named in
    # `owned_files.NEUTRALISED` and re-serialises the file, so a byte comparison
    # fails whenever his file is formatted differently from json.dumps -- which
    # is a check going red for something that is not a fault, and this one did
    # exactly that on the first run in the new repository. What C15 actually
    # requires is that his MEASURED values arrive in the copy: the harness must
    # race against his `calibrate_seconds` and `settle_seconds`, because the
    # timers around them move when those shorten, and shortened timing
    # manufactures a pass.
    _diff: list[str] = []
    _mine = _them = None
    if _copy.exists():
        try:
            _mine = _json.loads(_copy.read_text(encoding="utf-8"))
            _them = _json.loads(_real.read_text(encoding="utf-8"))
        except Exception as exc:                             # noqa: BLE001
            _diff.append(f"unreadable: {exc!r}")
    if _mine is not None and _them is not None:
        _forced = dict(_of.NEUTRALISED)
        for _k in set(_mine) | set(_them):
            if _k in _forced:
                if _mine.get(_k) != _forced[_k]:
                    _diff.append(f"{_k} should be forced to {_forced[_k]!r}, is {_mine.get(_k)!r}")
            elif _mine.get(_k, KeyError) != _them.get(_k, KeyError):
                _diff.append(f"{_k}: copy {_mine.get(_k)!r} vs his {_them.get(_k)!r}")
    check(_copy.exists() and _mine is not None and not _diff,
          "and the isolated settings are a COPY of his, not an empty file",
          f"{_diff[:6]} -- RULES C15: a harness that boots from defaults reads a "
          f"different calibrate_seconds and settle_seconds than the bench, and "
          f"the timers that race against them move")
    # And the two his timers actually race against, by name, so this cannot pass
    # on a copy that happens to match while carrying none of his measurements.
    for _timing in ("calibrate_seconds", "settle_seconds"):
        if isinstance(_them, dict) and _timing in _them:
            check(isinstance(_mine, dict) and _mine.get(_timing) == _them[_timing],
                  f"and his {_timing} reached the harness",
                  f"his {_them.get(_timing)!r} vs the copy's "
                  f"{(_mine or {}).get(_timing)!r}")

# AND A LAUNCHER MAY NOT DELETE HIS EVIDENCE.
#
# `run_live.ps1` and `repeat_gate.ps1` ran `Remove-Item` on the live log (and
# `run_live.ps1` on the verdict PNGs) before every launch. The logs of the
# evening of 2026-10-16 -- the ones four firing-line faults were reported from
# -- no longer exist, so every diagnosis made from them is unverifiable. They
# are rotated to a timestamped name now (plan 02, step 1).
#
# JUDGED BY WHAT IS DESTROYED, not by the spelling on the line. The first
# version matched `Remove-Item <target>` against a list of variable names, and
# review measured it: red for the one line it was written for, GREEN for the
# `rm`/`del`/`ri` aliases, `[IO.File]::Delete($log)`, `Clear-Content $log` and
# the same delete with the variable renamed, and a FALSE POSITIVE on an
# unrelated `Remove-Item $p` in a script that only mentioned the log in a
# comment. `tests/ps_evidence_probe.ps1` reads the PowerShell AST instead:
# comments are not nodes, a command is destructive by what it does under any
# name, and an argument is evidence if it names the files or is a variable
# assigned from something that does, transitively and through foreach.
#
# And because a check that cannot fail is worse than none (C1), the probe is
# first run on crafted scripts in a dozen spellings, each of which it must
# catch, plus the innocent ones it must not.
import json as _json3
import subprocess as _sub3
import tempfile as _tmp3


def _run_evidence_probe(paths):
    with _tmp3.TemporaryDirectory() as d:
        out = Path(d) / "out.json"
        r = _sub3.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                       "-File", str(ROOT / "tests" / "ps_evidence_probe.ps1"),
                       "-Files", "|".join(str(p) for p in paths),
                       "-Out", str(out)],
                      capture_output=True, text=True, timeout=180)
        if not out.exists():
            return None, r.stderr.strip()[-300:]
        return _json3.loads(out.read_text(encoding="utf-8-sig")), ""


_PRE = '$root = "C:/proj"\n$log = Join-Path $root "camera_lab.log"\n'
_MUST_CATCH = {
    "the original line": _PRE + "Remove-Item $log -Force\n",
    "rm alias": _PRE + "rm $log\n",
    "del alias": _PRE + "del $log -Force\n",
    "ri alias": _PRE + "ri $log\n",
    ".NET Delete": _PRE + "[IO.File]::Delete($log)\n",
    "fully-qualified .NET Delete":
        _PRE + "[System.IO.File]::Delete($log)\n",
    "Clear-Content": _PRE + "Clear-Content $log\n",
    "Set-Content truncating": _PRE + "Set-Content $log ''\n",
    "redirect truncating": _PRE + "'' > $log\n",
    "the variable renamed":
        '$x = Join-Path "C:/proj" "camera_lab.log"\nRemove-Item $x\n',
    "a variable derived from a variable":
        _PRE + '$y = $log\n$z = "$y"\nRemove-Item $z -Force\n',
    "the verdict images through foreach":
        'foreach ($f in @("recordings/verdict_screen.png")) {\n'
        '    $p = Join-Path "C:/proj" $f\n    if (Test-Path $p) { Remove-Item $p -Force }\n}\n',
    "a pipeline into Remove-Item":
        'Get-ChildItem "C:/proj/camera_lab*.log" | Remove-Item -Force\n',
    "a literal path": 'Remove-Item "C:/proj/recordings/verdict_camera.png"\n',
    "a method on a fetched item":
        _PRE + "(Get-Item $log).Delete()\n",
}
_MUST_PASS = {
    "rotating it": _PRE + "Move-Item $log (Join-Path $root 'camera_lab.1.log') -Force\n",
    "appending to it": _PRE + "'x' | Out-File $log -Append\nAdd-Content $log 'x'\n",
    "reading it": _PRE + "$t = Get-Content $log -Raw\n",
    "an unrelated delete, log only mentioned in a comment":
        "# camera_lab.log and verdict_screen.png are evidence; never delete them\n"
        '$p = Join-Path $env:TEMP "scratch.tmp"\nif (Test-Path $p) { Remove-Item $p -Force }\n',
    "an unrelated delete beside a rotation":
        _PRE + "Move-Item $log (Join-Path $root 'a.log')\n"
        '$tmp = Join-Path $env:TEMP "x.json"\nRemove-Item $tmp\n',
}
with _tmp3.TemporaryDirectory() as _d3:
    _files, _names = [], {}
    for _label, _body in list(_MUST_CATCH.items()) + list(_MUST_PASS.items()):
        _p = Path(_d3) / f"case{len(_files):02d}.ps1"
        _p.write_text(_body, encoding="utf-8")
        _files.append(_p)
        _names[str(_p)] = _label
    _found, _err = _run_evidence_probe(_files)
    check(_found is not None, "the evidence probe ran on its crafted scripts", _err)
    _hit = {_names[f["file"]] for f in (_found or []) if f["file"] in _names}
    _missed = [n for n in _MUST_CATCH if n not in _hit]
    _false = [n for n in _MUST_PASS if n in _hit]
    check(_found is not None and not _missed,
          "the evidence check catches a delete in every spelling",
          f"MISSED: {_missed}")
    check(_found is not None and not _false,
          "and does not fire on a rotation, a read, an append, or an "
          "unrelated delete",
          f"FALSE POSITIVES: {_false}")

_scripts = sorted(list((ROOT / "scripts").glob("*.ps1")) + list((ROOT / "tools").glob("*.ps1"))
                  + list((ROOT / "tests").glob("*.ps1")))
_found, _err = _run_evidence_probe(_scripts)
check(_found is not None, "the evidence probe ran on every launcher script", _err)
for ps in _scripts:
    bad = [f"line {f['line']}: {f['text']}" for f in (_found or [])
           if Path(f["file"]).resolve() == ps.resolve()]
    check(not bad, f"{rel(ps)} does not delete the live log or the verdict images",
          f"rotate them to a timestamped name instead: {bad}")


# ---------------------------------------------------------------------------
section("G. a wait must have a deadline, and a loop must have a bound")
# ---------------------------------------------------------------------------
#
# RULES B6/D4/D10. A fullscreen cover whose removal depended on a camera
# frame left the display an immovable black rectangle, escapable only by
# unplugging the camera. He had to press Escape to get an answer out of a
# wait loop with no failure branch: "you were unresponsive, you did not
# answer my questions." And `_start_noise_floor` re-armed itself every
# 700ms with no limit, so a routine that never finished meant a retry loop
# that never ended.

lab = read(SRC / "tools" / "camera_lab.py")
mod = tree(SRC / "tools" / "camera_lab.py")
owner = enclosing_function(mod)

# A RETRY LOOP is a singleShot whose callback is the ENCLOSING function
# itself. Anything else -- a timer that hands on to the next step -- is the
# normal shape of this code and not what the rule is about.
#
# The first version of this check guessed from substrings and named eight
# functions, most of which only hand on. A check that cannot tell the two
# apart is not a check (RULES C1), and an over-eager one gets switched off,
# which is worse than not having it.
def _mentions_self_call(node: ast.AST, fname: str) -> bool:
    """Does this expression call `self.<fname>` anywhere inside it?"""
    for inner in ast.walk(node):
        if (isinstance(inner, ast.Attribute) and inner.attr == fname
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"):
            return True
    return False


BOUND_TOKENS = ("_retries", "_attempt", "_tried", "_patcal_done", "range(")
selfsched = []
for fn_node in ast.walk(mod):
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    fname = fn_node.name
    body_src = ast.get_source_segment(lab, fn_node) or ""
    for node in ast.walk(fn_node):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "singleShot"):
            continue
        # args[1] is the callback. Re-entering its own function?
        if len(node.args) < 2 or not _mentions_self_call(node.args[1], fname):
            continue
        if not any(tok in body_src for tok in BOUND_TOKENS):
            selfsched.append(f"{fname}:{node.lineno}")
check(not selfsched, "no unbounded self-rescheduling timer",
      f"{selfsched} -- anything repeated needs a bound, a reason to stop, "
      f"and a report of what it found. `_start_noise_floor` re-armed itself "
      f"every 700ms with no limit for a day.")

# The Director arms a deadline for every effect, so this holds for every
# routine on it by construction. Prove the construction is still there.
check("_arm(run" in d_src and "self._clock.after(ms, fire)" in d_src,
      "the Director arms a timer for every wait",
      "if an effect can be awaited without a deadline, B6 is back")
arm_body = d_src.split("def _arm(")[1][:2600]
check("Deadline.after(self._clock" in arm_body,
      "and the deadline comes from its one clock")


# ---------------------------------------------------------------------------
section("H. the live app and the harness must be distinguishable")
# ---------------------------------------------------------------------------
#
# RULES C2. The harness built a real app and logged to the same
# `camera_lab.log` as the live one. A timing loop matched the HARNESS's
# verdict and I reported "17 seconds to READY, 12 of 12, 0.7mm" to him as a
# measurement of his bench. The giveaways were in the line I read: an
# 800x800 screen and exactly 60.0 fps.

check("LSA_LOG" in lab,
      "the log file can be redirected per process",
      "one log for two processes is how a harness result was read as his "
      "hardware")
coldstart = read(ROOT / "tests" / "test_coldstart.py")
# ONE PATH to the file that is his, and it is redirectable.
#
# It was inlined as `PROJECT_ROOT / "calibration.json"` in four places, and
# the harness saved its simulated 800x800 mapping over his real 3840x2160
# one THROUGH A PATH THAT HAD NOT BEEN BLOCKED. Neutering `Mapping.save`
# was the patch; the patch was routed around.
inlined = [f"{owner.get(n.lineno, '?')}:{n.lineno}"
           for n in ast.walk(mod)
           if isinstance(n, ast.Constant) and n.value == "calibration.json"]
check(len(inlined) <= 1,
      "camera_lab writes the calibration through ONE constant",
      f"{inlined} -- every save must go through CALIBRATION_PATH, which a "
      f"test can redirect; a second path is a second way to destroy the "
      f"mapping measured on his hardware")
check("LSA_CALIBRATION" in lab,
      "and that constant is redirectable",
      "so a test cannot reach his file at all, rather than being asked not to")

check("isolate_outputs" in coldstart,
      "and the harness redirects its log AND its images",
      "the log was separated by process and the images were not, so the "
      "harness went on overwriting recordings/verdict_screen.png with "
      "pictures from a simulated 800x800 camera")
gate_line = lab.split("GATE ")[1][:400] if "GATE " in lab else ""
check("role=" in gate_line or "role=" in lab,
      "the GATE line says which process produced it",
      "a measurement that cannot name its own process is not evidence")

# THE CONSUMER MUST PARSE WHAT THE PRODUCER WRITES.
#
# This is the check that was missing, and its absence cost the live gate.
# Adding `role=` and `screen=` in front of `verdict=` -- to stop a harness
# line ever being read as his hardware again -- made `repeat_gate.ps1`'s
# regex match NOTHING, because it pinned the prefix `GATE verdict=`. The
# gate would have scored every launch "no calibration; no self test; frame
# rate not reported; control writes not reported" and reported 0 of 5 on a
# working app.
#
# A test of the producer that never runs the consumer. Found by an outside
# review of the diff, not by me, and it is exactly the C1 shape: the gate
# could not fail for the right reason, it could only fail for all of them.
gate_ps = read(ROOT / "scripts" / "repeat_gate.ps1")
FIELDS = ("role", "screen", "mapping", "verdict", "detected", "held_out_mm",
          "error_mm", "fps", "margin", "floor", "bar", "exposure",
          "rejected_short", "writes", "coupling", "coupling_source",
          "findings")
# Reconstruct the line the app actually prints, from the source, so this
# cannot drift from the real format.
emitted = re.findall(r'f"GATE ([^"]*)"', lab) or []
produced = set()
for frag in re.findall(r'f"\s*([a-z_]+)=', lab.split("GATE role=")[0][-10:]
                       if False else ""):
    produced.add(frag)
# Simpler and exact: the names that appear as `<name>={` in the f-string
# block that starts at `GATE role=`.
block = lab[lab.find('f"GATE role='):]
block = block[:block.find("findings=") + 40] if "findings=" in block else block
produced = set(re.findall(r'\b([a-z_]+)=\{', block))
produced.update(re.findall(r'\b([a-z_]+)=([a-z]|\{)', block[:40]))
for name in ("role", "screen"):
    if f"{name}=" in block:
        produced.add(name)
missing_fields = [f for f in FIELDS if f not in produced]
check(not missing_fields,
      "the GATE line still carries every field the gate reads",
      f"{missing_fields} are no longer printed")

# And every field the gate reads must be one the app prints.
read_by_gate = set(re.findall(r"Get-GateField \$text '([a-z_]+)'", gate_ps))
orphans = sorted(read_by_gate - produced)
check(not orphans,
      "the gate reads no field the app does not print",
      f"{orphans} -- the gate would treat these as missing and fail every "
      f"launch for a reason that is about the parser, not the app")

# The gate must not pin the field ORDER, which is what broke it.
check("Get-GateField" in gate_ps,
      "the gate reads GATE fields by NAME, not by position",
      "a positional regex breaks whenever a column is added, and silently: "
      "it matches nothing and every check reports 'not reported'")
check(read_by_gate >= {"verdict", "fps", "writes", "role"},
      "and it reads at least the verdict, the frame rate, the writes and "
      "the role",
      f"it reads {sorted(read_by_gate)}")

# THE TWO READERS OF A GATE LOG MUST REACH THE SAME VERDICT.
#
# `tools/status.py` counted a launch good when its verdict read `ready` and
# nothing else, while this gate applies a dozen more criteria. It printed
# "5 of 5 reached READY ... the only state in which the word 'ready' may be
# used" over a log reading detected=11/12, which the gate FAILS -- the correct
# reading was 4 of 5. Two readers of one format disagreeing (C14), and the
# looser one hands the next session permission to say "ready".
#
# So: run the REAL judgement (tests/gate_probe.ps1 pulls Get-LaunchJudgement
# out of repeat_gate.ps1 by AST) and status.py's `judge_launch` over crafted
# launches, and require the same pass/fail AND the same reasons on every one.
import importlib.util as _ilu
import json as _json2
import subprocess as _sub2
import tempfile as _tmp

_spec = _ilu.spec_from_file_location("lsa_status", ROOT / "tools" / "status.py")
_status = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_status)

_GOOD = ("00:00:00  GATE role=live screen=3840x2160 mapping=confirmed "
         "verdict=ready detected=12/12 held_out_mm=0.82 error_mm=3.5 fps=42.3 "
         "margin=8.27 floor=9 bar=15 exposure=-4 rejected_short=0 writes=0 "
         "coupling=known coupling_source=bang findings=0")


def _gl(**over):
    line = _GOOD
    for k, v in over.items():
        line = re.sub(rf"\b{k}=\S+", f"{k}={v}", line)
    return line


_CASES = [
    # (name, log text, reached a verdict, closed cleanly)
    ("good launch, zero writes", _gl(), True, True),
    ("ready but 11/12 -- the one status.py once passed",
     _gl(detected="11/12"), True, True),
    ("ready but slow", _gl(fps="20.0"), True, True),
    ("ready but over the write budget", _gl(writes="30"), True, True),
    ("ready but writes not meaningful", _gl(writes="-1"), True, True),
    ("calibration too coarse, not confirmed",
     _gl(held_out_mm="2.5", mapping="refit"), True, True),
    ("confirmed mapping, no fresh held-out", _gl(held_out_mm="-1"), True, True),
    ("a harness line", _gl(role="harness"), True, True),
    ("too many short pulses", _gl(rejected_short="4"), True, True),
    ("a few short pulses is fine", _gl(rejected_short="3"), True, True),
    ("degraded verdict", _gl(verdict="degraded"), True, True),
    ("did not close cleanly", _gl(), True, False),
    ("never reached a verdict", "no gate line here", False, True),
    ("two lines: the first is the one the gate reads",
     _gl(detected="11/12") + "\n" + _gl(), True, True),
    ("a missing frame rate", re.sub(r"fps=\S+ ", "", _gl()), True, True),
    # The microphone's coupling to the app's own bang: the gate fails a launch
    # that is READY on every other count but whose independent counter of
    # trigger pulls would go deaf after each shot. All three outcomes of the
    # criterion, and the status.py mirror must give the same reasons.
    ("coupling unknown: the flat self-mute applies",
     _gl(coupling="none", coupling_source="none"), True, True),
    ("coupling not reported at all",
     re.sub(r"coupling=\S+ coupling_source=\S+ ", "", _gl()), True, True),
    ("coupling 'known' with no source (a producer bug)",
     _gl(coupling_source="none"), True, True),
    ("coupling from the startup probe is as good as from a real bang",
     _gl(coupling_source="probe"), True, True),
]
with _tmp.TemporaryDirectory() as _d:
    _cf, _of = Path(_d) / "cases.json", Path(_d) / "out.json"
    _cf.write_text(_json2.dumps([{"text": t, "done": d, "clean": c}
                                 for _n, t, d, c in _CASES]), encoding="utf-8")
    _pr = _sub2.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                     "-File", str(ROOT / "tests" / "gate_probe.ps1"),
                     "-Gate", str(ROOT / "scripts" / "repeat_gate.ps1"),
                     "-Cases", str(_cf), "-Out", str(_of)],
                    capture_output=True, text=True, timeout=120)
    _gate = (_json2.loads(_of.read_text(encoding="utf-8-sig"))
             if _of.exists() else [])
check(len(_gate) == len(_CASES),
      "the real gate judgement ran on every crafted launch",
      f"{len(_gate)} of {len(_CASES)}; stderr: {_pr.stderr.strip()[-300:]}")
for (_n, _t, _d, _c), _g in zip(_CASES, _gate):
    _ok, _why = _status.judge_launch(_t, _d, _c)
    check(_ok == _g["pass"] and _why == list(_g["reasons"]),
          f"status.py and the gate agree: {_n}",
          f"gate: pass={_g['pass']} {_g['reasons']}; status.py: pass={_ok} {_why}")
_by_name = {n: g for (n, *_r), g in zip(_CASES, _gate)}
check(_by_name.get("good launch, zero writes", {}).get("pass") is True
      and _by_name.get("ready but 11/12 -- the one status.py once passed",
                       {}).get("pass") is False,
      "and the two cases that matter come out as intended",
      "zero control writes must PASS; ready-but-11/12 must FAIL")
check(_by_name.get("coupling unknown: the flat self-mute applies",
                   {}).get("pass") is False
      and _by_name.get("coupling not reported at all", {}).get("pass") is False
      and _by_name.get("coupling from the startup probe is as good as from a "
                       "real bang", {}).get("pass") is True,
      "a launch whose microphone coupling is unknown or unreported FAILS the "
      "gate, and one measured by the startup probe passes",
      "READY with the microphone going deaf after every shot is the silent "
      "failure this field exists to make loud")


# ---------------------------------------------------------------------------
section("I. the camera: ration control writes, and never force-kill")
# ---------------------------------------------------------------------------
#
# RULES A1/A4/A6. One `CAP_PROP_EXPOSURE` write has hung this camera for
# 272 seconds, while five minutes of pure streaming with zero writes
# returned 17,945 frames and no failures. A force-kill leaves it opening in
# 40.3s with 0 frames, against 0.44s and 53.6 fps after a clean close.

cap = read(SRC / "camera" / "capture.py")
check("control_writes" in cap,
      "every UVC transaction is counted",
      "a budget you cannot see is not a budget")
set_ctl = cap.split("def set_control")[1][:2000]
check("_control_values" in set_ctl or "cached" in set_ctl,
      "a write whose value already matches is skipped")
check("Director" in lab and "control_writes" in d_src,
      "and the Director counts the writes its routines make")

for path in sorted((ROOT / "scripts").glob("*.ps1")):
    src = read(path)
    if "Stop-Process" in src and "-Force" in src:
        ok = "WM_CLOSE" in src or "CloseMainWindow" in src or "stop_lab" in src
        check(ok, f"{rel(path)} tries a clean close before forcing",
              "a force-kill wedges this camera's firmware")


# ---------------------------------------------------------------------------
section("J. never advise dimming a room that is already dark")
# ---------------------------------------------------------------------------
#
# RULES C4/D7. The app told a man sitting in the dark to dim his room.
# TWICE, a day apart, from two different code paths -- and a third site was
# found only by generating every verdict the module can produce.
#
# > "I already dimmed the room to the point where I cannot dim it anymore
# >  because otherwise I will be shooting in complete darkness but I need to
# >  see my gun sights. This is the max darkness for my comfort."
#
# A dark room is the BEST condition this app can have: it permits a longer
# exposure, and the laser scales with exposure time. The exhaustive version
# of this check lives in `test_selfcheck.py` section 12, which generates
# 10,080 verdicts. This one asserts that it is still there and still
# exhaustive, because that is the check that found the third site.

sc = read(ROOT / "tests" / "test_selfcheck.py")
# It sweeps with nested loops rather than itertools, and -- this is the
# part that matters -- it ASSERTS how many verdicts it generated. A sweep
# that silently shrank to three cases would otherwise still pass, which is
# the whole C1 failure: a check that can quietly do nothing.
check("room_bright=False" in sc and "swept" in sc,
      "the dark-room check sweeps generated verdicts, not a few examples",
      "example cases found two of the three sites; generation found the third")
check(re.search(r"tried\s*>\s*\d{3,}", sc) is not None,
      "and it asserts the sweep was actually large",
      "a sweep that shrinks to three cases must fail, not pass quietly")
# The mirror case too: silencing room advice everywhere would pass the
# sweep and be just as useless.
# RESTATED 2026-10-16, and the mirror still has to exist: silencing the
# room everywhere would pass the 10,080-verdict sweep and be just as
# useless. What it asserts changed. A bright VIEW is now named as the
# daylight window beside the monitor -- `room_bright` is the mean and
# chroma of the WHOLE camera frame, so the window is literally what it
# measures -- and never phrased as an instruction to dim a room he has
# already darkened as far as he can and still see his sights. It was on
# his screen again on 2026-10-16, which is the fourth time.
check("bright view is still named" in sc,
      "and a genuinely bright view is still named, as the window",
      "a sweep that only forbids advice can be satisfied by saying nothing")
advice_sites = len(re.findall(r"daylight|window|exposure",
                              read(SRC / "selfcheck.py"), re.IGNORECASE))
check(advice_sites > 0, "selfcheck.py is the one place room advice is decided",
      "advice emitted from a dozen places is how this came back the next day")


# ---------------------------------------------------------------------------
section("L. a file the user owns is written atomically")
# ---------------------------------------------------------------------------
#
# RULES C16, added 2026-10-16 after Fable's third review. `Mapping.save` was
# a bare `write_text`, which TRUNCATES the real file and only then starts
# writing. His calibration.json holds the 1mm mapping measured from 18
# photographs plus the sensitivity regions learned from his real shots, and
# this app exits through `os._exit` by design, has been force-killed many
# times, and has crashed at shutdown from timers firing into a destroyed
# scene. Any of those inside that window empties the file -- and the loader
# reads a corrupt file as "no calibration" and recalibrates, so the loss
# never announces itself.
#
# Structural half: no `save` may write the real path directly.

for f in ALL_PY:
    mod = tree(f)
    for node in ast.walk(mod):
        if not isinstance(node, ast.FunctionDef) or node.name != "save":
            continue
        bad = [
            n.lineno for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("write_text", "write_bytes")
        ]
        check(not bad, f"{rel(f)}: save() does not write the real path directly",
              f"line(s) {bad} -- truncate-then-write; use lsa.atomic")

# Behavioural half, which is the one that matters: an interrupted save must
# leave the PREVIOUS file intact. Checked by making the write fail at the
# point a crash would, then reading the file back.

from lsa.atomic import write_json_atomic as _atomic           # noqa: E402

_tmpdir = ROOT / "tests" / "_out"
_tmpdir.mkdir(parents=True, exist_ok=True)
_victim = _tmpdir / "atomic_probe.json"
_victim.write_text('{"precious": "the 1mm mapping"}', encoding="utf-8")

class _Unserialisable:
    pass

ok = _atomic(_victim, {"bad": _Unserialisable()})
check(ok is False, "a save that cannot be serialised reports failure")
check("precious" in _victim.read_text(encoding="utf-8"),
      "and leaves the previous file untouched",
      "this is what a bare write_text cannot do: it truncates first")

ok = _atomic(_victim, {"fresh": 1})
check(ok is True and "fresh" in _victim.read_text(encoding="utf-8"),
      "a good save replaces it")
check(not (_victim.parent / (_victim.name + ".tmp")).exists(),
      "and leaves no temp file behind")

# And the real consumers, not a paraphrase of them (C12/C14): the two
# dataclasses the user owns must round-trip through their own save().
from lsa.calibrate.mapping import Mapping                      # noqa: E402
from lsa.settings import AppSettings                           # noqa: E402

import numpy as _np
_m = Mapping(homography=_np.eye(3), screen_size=(3840, 2160), dot_radius=11.0)
_mp = _tmpdir / "atomic_mapping.json"
check(_m.save(_mp) and (Mapping.load(_mp) or Mapping()).dot_radius == 11.0,
      "Mapping.save round-trips through the atomic path")
# AND A MAPPING FULL OF NUMPY SCALARS STILL SAVES.
#
# 2026-10-16: every value in a real mapping can arrive as a numpy scalar --
# the samples come from `find_marker`, which returns numpy floats -- and
# `json.dumps` cannot encode those. `save()` caught the TypeError, returned
# False, and all three callers ignored the result, so the app logged
# "saved" with nothing on disk. Six consecutive harness launches then paid
# for eighteen markers because there was no saved mapping for the
# three-second check to confirm, and a launch took 81 seconds instead of
# 40. The suite was green the whole time.
from lsa.calibrate.mapping import Sample as _Sample                 # noqa: E402

_npm = Mapping(
    homography=_np.eye(3),
    screen_size=(_np.int32(800), _np.int32(800)),
    samples=[_Sample(screen=(_np.float32(1.0), _np.float32(2.0)),
                     camera=(_np.float64(3.0), _np.float64(4.0)),
                     peak=_np.float32(250.0))],
    sensitivity=[{"camera": [_np.float64(1.0), _np.float64(2.0)],
                  "peak": _np.float32(55.0)}],
)
_nmp = _tmpdir / "atomic_numpy.json"
check(_npm.save(_nmp) is True,
      "a mapping full of numpy scalars still saves",
      "json cannot encode numpy types, save() swallowed it, and the one "
      "caller ignored the False -- so the log said 'saved' with nothing "
      "on disk")
_back = Mapping.load(_nmp)
check(_back is not None and len(_back.samples) == 1,
      "and it loads back with its samples intact")
try:
    _nmp.unlink()
except Exception:
    pass

_sp = _tmpdir / "atomic_settings.json"
check(AppSettings().save(_sp) and _sp.exists(),
      "AppSettings.save round-trips through the atomic path")
for _p in (_victim, _mp, _sp):
    try:
        _p.unlink()
    except Exception:
        pass



# THE SESSION HOOK MUST PARSE AND RUN.
#
# He asked, after I made a morning of changes without re-reading the rules,
# that they be re-read after every compaction "without fail". The mechanism
# built for it -- a SessionStart hook printing status.py and RULES.md -- was
# written with SINGLE backslashes inside a JSON string, where \S and \p are
# invalid escapes. The whole file failed to parse, so the hook never ran
# once. Silently. The thing built so a rule could not be forgotten was
# itself forgotten, by construction, and an outside review found it rather
# than me.
#
# Reading the file cannot catch it: forward slashes parse and then fail to
# launch on Windows instead. So this parses it AND runs the command.
_hook = author_file("L. .claude/settings.json", "LSA_CLAUDE_SETTINGS")
if _hook is not None:
    import json as _json
    import subprocess as _sub
    _conf = None
    try:
        _conf = _json.loads(read(_hook))
    except Exception as exc:                                 # noqa: BLE001
        check(False, "the session hook is valid JSON", f"{exc}")
    if _conf is not None:
        check(True, "the session hook is valid JSON")
        _cmds = [h.get("command", "")
                 for grp in _conf.get("hooks", {}).get("SessionStart", [])
                 for h in grp.get("hooks", [])]
        check(bool(_cmds), "and it defines a SessionStart command")
        for _c in _cmds:
            try:
                # Run it where that hook LIVES -- the folder holding .claude --
                # not wherever this suite was started from. Identical in his
                # working tree, where they are the same folder; correct when the
                # settings file is reached through LSA_CLAUDE_SETTINGS, which
                # would otherwise resolve the hook's own paths against this repo
                # and fail a hook that is perfectly good.
                _r = _sub.run(_c, cwd=str(_hook.parent.parent), shell=True,
                              timeout=90, capture_output=True, text=True)
                _ok = _r.returncode == 0
                _has = "BEFORE WRITING CODE" in _r.stdout
            except Exception as exc:                         # noqa: BLE001
                _ok, _has = False, False
                _r = None
            check(_ok, f"and `{_c}` actually runs",
                  "a hook that cannot launch is a rule that is not read")
            check(_has, "and it puts RULES.md in front of the session",
                  "status alone is not the agreement: he asked for the rules")

# ---------------------------------------------------------------------------
section("K. every rule in RULES.md is accounted for")
# ---------------------------------------------------------------------------
#
# THE FAILURE THIS ONE EXISTS FOR IS MINE, AND IT IS THE WORST OF THEM.
#
# While rewriting RULES.md to be more thorough I DROPPED SIX RULES that were
# in my own first draft -- B9, C11, C12, C13, D9, D10 -- three of which
# described damage done that same day. They were restored only because I
# listed the headings and compared them against the session by hand. That
# is the exact mistake the file exists to prevent, committed inside the
# file, by someone looking straight at it.
#
# So the registry below is the file's index, and this check is what makes
# losing a rule impossible rather than unlikely: every `## Xn.` heading in
# RULES.md must appear here, and every entry here must still be in
# RULES.md. A rule can be retired, but only by deleting its line in the
# same change -- which is a visible, reviewable act, not an omission.
#
# The value is the enforcement: a test name, or "judgement" for the ones a
# machine genuinely cannot check (how to talk to him, what to measure,
# when to believe a number).
REGISTRY: dict[str, str] = {
    # A -- how not to sabotage the camera
    "A1": "test_rules I (no force-kill without a clean close first)",
    "A2": "director (cancel + bounded shutdown; test_director 3, 15)",
    "A3": "director (cancel runs cleanup; test_director 3)",
    "A4": "test_rules I + test_director 6 (counted, spaced, deduped)",
    "A5": "judgement (one instrument, not two -- review)",
    "A6": "test_director 6 (the auto-range ladder rewrote one value)",
    "A7": "test_coldstart (asserts lab.capture is None before connecting)",
    "A8": "judgement (diagnostics live in tools/ with the same rationing)",
    "A9": "judgement (a harness that aborts must still clean up)",
    "A10": "judgement (run_live.ps1 refuses against a wedged camera)",
    "A11": "judgement (identify by what it does; see CLAUDE.md)",
    "A12": "judgement (read the PnP problem code first)",
    "A13": "judgement (fps only with the exposure fixed)",
    # B -- what the app draws
    "B1": "test_rules D (every hex colour, every file)",
    "B1b": "test_ui_smoke (lit area of the shooting state, limit 2%)",
    "B10": "test_ceiling (every registered mode rendered; no pixel above the measured ceiling)",
    "B2": "test_feedback 29 (every Job declares measures_screen)",
    "B3": "test_feedback 28 (a timed overlay clears itself)",
    "B4": "test_rules B + test_director 12 (one clock, one Deadline type)",
    "B5": "test_rules E (no absolute font size on the target screen)",
    "B6": "test_director 4 + test_rules G (no wait without a deadline)",
    "B7": "judgement (one window for the shooter; --target minimises)",
    "B8": "judgement (--modetest is opt-in and off by default)",
    "B9": "test_feedback 26 + 29 (consumes_detections, declared once)",
    # C -- measurement, and claims made from it
    "C1": "mutate_director (every check reintroduced and watched go red)",
    "C2": "test_rules H (role= and screen= on the GATE line)",
    "C3": "test_feedback 29 + _check_line (report the measurement)",
    "C4": "test_selfcheck 12 (sweeps generated verdicts, room dark)",
    "C5": "test_selfcheck (a measurement beats a prediction of it)",
    "C6": "test_exposure (chosen by what it does to HIS shots)",
    "C7": "test_feedback 20 (an unmeasured exposure says so)",
    "C8": "test_rules C (ratcheted getattr(self, '_x', default) budget)",
    "C9": "test_pattern_cal (held-out error, never fit error)",
    "C10": "tools/status.py (prints N of 5; refuses the word below it)",
    "C11": "test_rules F + tests/owned_files.py (fingerprint his files)",
    "C12": "test_director 15 (the production clock, exercised for real)",
    "C13": "tools/pre-commit + test_rules N (installed hook == repo copy)",
    "C14": "test_rules H (a real GATE line fed to the real gate parser)",
    "C15": "test_coldstart (uses HIS calibrate/settle seconds, not shorter)",
    "C16": "test_rules L (no write_text in save; interrupted save, file intact)",
    "C17": "test_rules M (anchors match once) + test_feedback (stub methods read from the app source) + test_coldstart (the launch budget carries its derivation)",
    # D -- how to work with him
    "D1": "judgement (look at the screen yourself; screenshot.ps1)",
    "D2": "tools/pre-commit + test_rules N (installed hook == repo copy)",
    "D3": "judgement (ask before taking the screen)",
    "D4": "test_director 4 (no unbounded wait exists to take)",
    "D5": "judgement (no code in replies)",
    "D6": "judgement (give the structural change, not an assurance)",
    "D7": "test_selfcheck 12 (his constraints are inputs: the dark room)",
    "D8": "judgement (correct your own claims promptly and plainly)",
    "D9": "judgement (startup must be fast; two-marker check, --trust-cal)",
    "D10": "test_rules G + test_director 8 (every loop has a bound)",
    "D11": "tools/status.py (re-derives the gate from the logs on disk)",
    "D12": "judgement (split delegated work; each piece reports inside ~10 min)",
    "D13": "judgement (every burst of output ends with a short summary of what is going on)",
}

_rules_path = author_file("K. RULES.md", "LSA_RULES_MD")
rules_md = read(_rules_path) if _rules_path is not None else ""
if _rules_path is not None:
    headings = re.findall(r"^## ([A-Z]\d+b?)\.", rules_md, re.M)
    check(len(headings) >= 40, f"RULES.md still has its rules ({len(headings)})",
          "if this collapses, the file has been rewritten and lost some")

    lost = [r for r in REGISTRY if r not in headings]
    check(not lost, "no rule in the registry has vanished from RULES.md",
          f"{lost} -- a rule may be RETIRED by deleting its registry line in the "
          f"same change, which is reviewable. Silently losing one is not: six "
          f"went missing in a single rewrite of that file.")

    unregistered = [h for h in headings if h not in REGISTRY]
    check(not unregistered, "every rule in RULES.md is in the registry",
          f"{unregistered} -- add it, with the test that enforces it or the word "
          f"'judgement' if a machine genuinely cannot")

    dupes = [h for h in set(headings) if headings.count(h) > 1]
    check(not dupes, "no rule number is used twice", f"{dupes}")

enforced = [r for r, v in REGISTRY.items() if not v.startswith("judgement")]
print(f"  {len(enforced)} of {len(REGISTRY)} rules are enforced by a check; "
      f"{len(REGISTRY) - len(enforced)} are judgement")
check(len(enforced) >= 25,
      "most rules are enforced by something that runs",
      f"only {len(enforced)} -- prose I have to re-read is the part that "
      f"failed")

# C15's enforcement, checked rather than asserted: the harness must use his
# real settling times, not shorter ones. Shortened timing manufactures a
# pass, because the probe is racing the noise floor and a one-second floor
# is already over when the probe asks.
cs = read(ROOT / "tests" / "test_coldstart.py")
check(not re.search(r"calibrate_seconds\s*=\s*[01]?\.\d", cs),
      "the harness does not shorten the noise floor",
      "a floor shortened to 1s lets the probe find a free world; his is "
      "3s, so on his bench the probe is queued and refused every launch "
      "and the harness says it is fine (RULES C15)")
check("getattr(lab.settings, \"calibrate_seconds\"" in cs
      or "lab.settings.calibrate_seconds = float(" in cs,
      "and it reads the timings from his settings rather than hardcoding")

# ---------------------------------------------------------------------------
section("N. the hook git RUNS is the hook the repo SHOWS")
# ---------------------------------------------------------------------------
#
# `tools/pre-commit` is a copy for review. Git executes `.git/hooks/pre-commit`,
# a different file put there by `tools/install-hooks.ps1`. On 2026-10-17 the
# installed copy was two days old (8 suites, 2 mutation scripts) while the
# repo copy had been strengthened all day (16 suites, 3 mutation scripts) --
# so every improvement had NO EFFECT and a commit ran the October hook. RULES
# C13 and D2 are registered as enforced by that file; a rule can be
# strengthened in the repo and be inert for weeks. Same fault class as
# `.claude/settings.json` sitting in invalid JSON (section L) and the
# harness that ran one launch of seven: a mechanism that does nothing and
# looks exactly like one that works (C1, C7).
#
# Comparison is on CONTENT with line endings normalised, plus a separate
# check that the installed file contains no CR at all. Why both: this repo has
# core.autocrlf=true, so a checkout of tools/pre-commit may legitimately be
# CRLF and Copy-Item would carry that across; a byte comparison would then
# fail a hook that is semantically identical, and teach people to ignore the
# check. But a CRLF `#!/bin/sh\r` hook is BROKEN under sh, and normalising
# alone would pass it. So: equal after normalising, and the installed copy must
# itself be LF.
#
# No installed hook is a FAILURE, not a skip. An unreported check is a
# failure, not a pass by omission (C1): a skip here would pass on exactly the
# machine where nothing is enforced. The repo is local-only, so a fresh clone
# without a hook is the one state this is for.
import subprocess as _sp2

_repo_hook = ROOT / "tools" / "pre-commit"
_hook_path = None
try:
    _o = _sp2.run(["git", "rev-parse", "--git-path", "hooks/pre-commit"],
                  cwd=str(ROOT), capture_output=True, text=True, timeout=30)
    if _o.returncode == 0 and _o.stdout.strip():
        _hook_path = (ROOT / _o.stdout.strip()).resolve()
except Exception:                                            # noqa: BLE001
    _hook_path = None
if _hook_path is None:
    _hook_path = ROOT.parent / ".git" / "hooks" / "pre-commit"      # the repository root is above app/

_FIX = "run: powershell -NoProfile -File tools\\install-hooks.ps1"
check(_repo_hook.exists(), "tools/pre-commit exists in the repo")
if _repo_hook.exists():
    check(_hook_path.exists(), "a pre-commit hook is installed for git to run",
          f"{_hook_path} is missing, so NOTHING in tools/pre-commit is "
          f"enforced; {_FIX}")
    if _hook_path.exists():
        _inst = _hook_path.read_bytes()
        _want = _repo_hook.read_bytes()
        check(b"\r" not in _inst,
              "the installed hook has LF line endings",
              "CRLF breaks `#!/bin/sh` under Git for Windows; reinstall from "
              "an LF copy")
        _ni = _inst.replace(b"\r\n", b"\n")
        _nw = _want.replace(b"\r\n", b"\n")
        _why = ""
        if _ni != _nw:
            _a, _b = _ni.splitlines(), _nw.splitlines()
            _n = next((i for i in range(min(len(_a), len(_b)))
                       if _a[i] != _b[i]), min(len(_a), len(_b)))
            _why = (f"installed {len(_a)} lines vs repo {len(_b)}; first "
                    f"difference at line {_n + 1}; {_FIX}")
        check(_ni == _nw,
              "the installed hook is the repo's tools/pre-commit",
              _why)
        check(_ni.startswith(b"#!/bin/sh"),
              "and it begins with a shell shebang git can launch")


section("M. every mutation anchor must still find its target")
# 2026-10-17. mutate_director.py reported an anchor "matched 0 times" because
# camera_lab.py is CRLF and the anchors are triple-quoted literals (bare LF).
# The property it guarded was intact; nothing but a person reading mutation
# output could tell. Git hides line-ending flips (core.autocrlf), so a re-save
# could disarm any of ~180 anchors invisibly -- a check that cannot fail
# (RULES C1). The mutation scripts take over a minute and are not run by the
# commit hook; THIS runs in the unit suite and costs milliseconds.
sys.path.insert(0, str(ROOT / "tests"))
import mutation_anchors as _anc

# -- the matcher itself must be able to fail: prove it on crafted text --------
_two = "a = 1\nb = 2\n"
check(_anc.apply(_two, "a = 1\nb = 2", "a = 9\nb = 2") == "a = 9\nb = 2\n",
      "an LF anchor applies to LF text")
_crlf = _two.replace("\n", "\r\n")
check(_anc.apply(_crlf, "a = 1\nb = 2", "a = 9\nb = 2") == "a = 9\r\nb = 2\r\n",
      "an LF anchor applies to CRLF text, and the replacement keeps CRLF",
      "this is the exact case that disarmed 'the gain map is judged on the "
      "floor instead of the bar'")
check(_anc.apply(_two, "a = 1\r\nb = 2", "x\r\ny") == "x\ny\n",
      "a CRLF anchor applies to LF text")
_mixed = "a = 1\r\nb = 2\nc = 3\r\n"
check(_anc.apply(_mixed, "c = 3", "c = 4") == "a = 1\r\nb = 2\nc = 4\r\n",
      "a single-line edit in a MIXED-ending file changes no other byte")
check(_anc.apply(_two, "z = 0", "q") is None,
      "an anchor that is absent is NOT applied")
check(_anc.apply("x\nx\n", "x", "y") is None,
      "an anchor that matches twice is NOT applied")
check("NOT FOUND" in _anc.not_found_reason(_two, "z = 0"),
      "a missing anchor is reported as 'not found', not as a survivor")

# -- and the real tables -----------------------------------------------------
# IMPORTED INSIDE A GUARD, BECAUSE A CRASH IS NOT A REPORT.
#
# Found 2026-10-19 by deliberately deleting `tests/worker_budget.py` to prove
# the check that watches for its absence could fail. It could not: these four
# modules import it, so the module going missing raised ModuleNotFoundError
# HERE, twenty lines before the check that would have named the real cause.
# What a reader got was a traceback about an import, which is a log line
# naming the wrong cause -- and this project has lost a day to one of those.
#
# The suite still fails, loudly and non-zero. It now fails SAYING WHICH
# SCRIPT and WHY, and the sections after this one still run, so a second
# fault is not hidden behind the first. RULES C1.
_mutmods: dict[str, object] = {}
for _mn in ("mutate_director", "mutate_detector", "mutate_figures",
            "mutate_introspect", "mutate_ghost", "mutate_audio"):
    try:
        _mutmods[_mn] = __import__(_mn)
    except Exception as _why:          # noqa: BLE001 - any import fault reports
        check(False, f"{_mn}.py imports cleanly",
              f"{type(_why).__name__}: {_why}  --  the anchor scan below "
              f"cannot run, so its checks are NOT a pass by omission")
    else:
        check(True, f"{_mn}.py imports cleanly")

_md = _mutmods.get("mutate_director")
_mdet = _mutmods.get("mutate_detector")
_mfig = _mutmods.get("mutate_figures")
_mint = _mutmods.get("mutate_introspect"); _mghost = _mutmods.get("mutate_ghost")
_maud = _mutmods.get("mutate_audio")

_unplaced: list[str] = []
_total = 0
for _m in (_md.MUTATIONS if _md is not None else []):
    _name, _tgt, _old, _new = _md.normalise(_m)
    _total += 1
    if _anc.locate(_md.read_raw(_tgt), _old)[0] != 1:
        _unplaced.append(f"director: {_name}")
for _label, _mod in (("detector", _mdet), ("figures", _mfig),
                     ("introspect", _mint), ("ghost", _mghost),
                     ("audio", _maud)):
    if _mod is None:
        continue
    for _m in _mod.M:
        _name, _rel, _old = _m[0], _m[1], _m[2]
        if _old is None:               # a data mutation, not a text anchor
            continue
        # A fault whose defence is TWO independent layers needs both removed
        # (mutate_audio): then file and anchor are parallel lists, and EVERY
        # anchor must still match exactly once.
        _pairs = (list(zip(_rel, _old)) if isinstance(_rel, list) else [(_rel, _old)])
        for _rel1, _old1 in _pairs:
            _total += 1
            _text = (ROOT / _rel1).read_bytes().decode()
            if _anc.locate(_text, _old1)[0] != 1:
                _unplaced.append(f"{_label}: {_name}")
check(_total > 150, f"the anchor scan covers the whole set ({_total} anchors)",
      "a short count means a mutation module did not import, not that the "
      "anchors are fine")
check(not _unplaced,
      "every mutation anchor matches its target exactly once",
      "NOT FOUND (a check that cannot fail): " + "; ".join(_unplaced))

# C17, 2026-10-22: a mutation runner's scratch tree must hold everything its suite
# IMPORTS. mutate_ghost copied `test_ghost.py` alone; every suite then gained
# `import safe_stdio`, the scratch baseline died with ModuleNotFoundError, and its
# "47 of 47" could not be re-derived at all. The fix is to DERIVE the copy list from
# the suite's own imports (`suite_files`). The other runners copy the whole tests/
# directory (director mutates the real tree in place), checked here from their
# source so a runner switched back to a hand-written list goes red by name.
if _mghost is not None:
    _sf = _mghost.suite_files("test_ghost.py")
    check("test_ghost.py" in _sf and "safe_stdio.py" in _sf,
          "mutate_ghost derives its scratch copy list from the suite's imports "
          "(test_ghost.py and the safe_stdio.py it imports are both in it)", str(_sf))
_cp_gaps = []
for _rn in ("mutate_detector", "mutate_figures", "mutate_introspect"):
    _rsrc = (ROOT / "tests" / f"{_rn}.py").read_text(encoding="utf-8")
    # 2026-10-23: the copy is `asset_env.copy_scratch_tree` (code, every asset folder and
    # COPIES of the files he owns, derived from the tables); a hand-written `for sub in (...)`
    # loop here is what left config/, sounds/ and .venv/pyvenv.cfg out of three trees.
    if "copy_scratch_tree(" not in _rsrc or re.search(r'for sub in \(', _rsrc):
        _cp_gaps.append(_rn)
check(not _cp_gaps,
      "the other mutation runners build their scratch tree with copy_scratch_tree(), not a hand-written list",
      "a hand-written list drifts from what the suites read: " + ", ".join(_cp_gaps))

# -- 2026-10-23: A FAULT IS CAUGHT ONLY WHEN ITS OWN NAMED CHECK FAILS ----------
# A section of test_ui_smoke raised halfway (a stand-in lacked one method) and the
# suite went red with ONE line: "...ran to its end without raising -- AttributeError".
# mutate_detector counted every fault aimed at that suite as caught because the suite
# was red: 22 "caught" faults whose own checks never ran, "112 of 112", and a real
# crash reported as a success. `tests/mutation_verdict.py` is now the one judgement,
# and what follows proves IT can fail: fabricated outputs go in, and the verdict that
# comes out is the one that would have exposed the incident.
import mutation_verdict as _mv

_CRASH = ("  [FAIL] the brightness-grid section ran to its end without raising -- "
          "AttributeError: '_SgCap' object has no attribute 'set_aim_help'\n"
          "SOME CHECKS FAILED\n")
_OWN = "  [FAIL] brightness grid: faint shots can register - the bar the DETECTOR holds\nSOME CHECKS FAILED\n"
_OTHER = "  [FAIL] some unrelated check\nSOME CHECKS FAILED\n"
_FRAG = "faint shots can register"
check(_mv.classify(1, _CRASH, _FRAG)[0] == "BROKEN",
      "a suite that CRASHES a section instead of failing the named check is BROKEN, not caught",
      repr(_mv.classify(1, _CRASH, _FRAG)))
check(_mv.classify(1, _OWN, _FRAG)[0] == "caught",
      "...while the named check failing is caught", repr(_mv.classify(1, _OWN, _FRAG)))
check(_mv.classify(1, _OTHER, _FRAG)[0] == "MISSED",
      "...and an unrelated failure is MISSED (the check written for the fault cannot fail)")
check(_mv.classify(0, "ALL 40 CHECKS PASSED\n", _FRAG)[0] == "MISSED",
      "...and a green suite is MISSED")
check(_mv.classify(1, "Traceback (most recent call last):\n  File x\nKeyError: 1\n", _FRAG)[0] == "BROKEN",
      "a suite that dies before any summary is BROKEN")
check(_mv.classify(-9, "TIMED OUT after 900s", _FRAG)[0] == "BROKEN",
      "a timeout is BROKEN")
check(_mv.classify(1, _CRASH.replace("AttributeError", _FRAG), _FRAG)[0] == "BROKEN",
      "a CRASH line that merely quotes the fragment is not the named check failing",
      "the fragment can appear in a crash's own text")
check(_mv.classify(1, _CRASH + _OWN, _FRAG)[0] == "caught",
      "a crash elsewhere does not stop the named check, when it DID fail, being caught")
check(_mv.classify(1, "  FAIL  the named thing\nFAILED 1 of 9 ROUTINE CHECKS:\n", "named thing")[0] == "caught",
      "the 'FAIL  text' dialect (test_routines) is read as a failing line")
check(_mv.classify(1, _OWN, "")[0] == "BROKEN" and _mv.classify(1, _OWN, "  ")[0] == "BROKEN",
      "an entry that names no check is refused, never matched by 'anything'")
check(bool(_mv.crash_reason(1, _CRASH)) and _mv.crash_reason(1, _OTHER) == "",
      "the crash test for an UNNAMED entry (mutate_director) separates a crash from a failed check")

# -- A PASS LINE THAT QUOTES THE MARKER AS DATA IS NOT A FAILURE (2026-10-23) ------
# test_feedback section 43 prints, for a PASSING check, its detail: a window title the
# desktop really carried, written with the marker in front of it. The judge used to test
# `"[FAIL]" in line`, so this PASS made test_feedback "red", the baseline guard refused
# mutate_detector (115) and mutate_figures (56), and one mutate_director entry became a
# fake crash. The fixture is that line verbatim, so reverting `fail_lines` to a
# containment test turns the four checks below red.
_PASS_QUOTING = ("  [PASS] a hardened stream writes it without raising, keeping the line readable"
                 " --   [FAIL] it names \\u2733 LASER-SHOOT-BOOK is in front of the target screen\n")
_GREEN_SUITE = _PASS_QUOTING + "ALL 538 CHECKS PASSED\n"
check(_mv.fail_lines(_PASS_QUOTING) == [],
      "a PASS line carrying the failure marker in its payload is NOT read as a failing line",
      repr(_mv.fail_lines(_PASS_QUOTING)))
check(_mv.baseline_problem(0, _GREEN_SUITE) == "",
      "...and a green suite containing such a line is a GREEN baseline",
      repr(_mv.baseline_problem(0, _GREEN_SUITE)))
check(_mv.baseline(["test_feedback.py"], lambda s: (0, _GREEN_SUITE)) == [],
      "...so the baseline guard does not refuse the runners that name that suite")
check(_mv.crash_reason(0, _GREEN_SUITE) == ""
      and _mv.classify(0, _GREEN_SUITE, "faint shots can register")[0] == "MISSED",
      "...and a quoting PASS line is neither a crash nor a failure when classifying an entry")
check(_mv.fail_lines("  [FAIL] real one -- and [FAIL] quoted\n   FAIL  other dialect\n"
                     "[PASS] fine -- FAIL  not at line start\n")
      == ["[FAIL] real one -- and [FAIL] quoted", "FAIL  other dialect"],
      "...while a line that STARTS with the marker (either dialect, any indent) still is one")

# -- THE JUDGE MUST READ EVERY DIALECT THE SUITES PRINT (2026-10-23) ----------------
# `SUMMARY` was `CHECKS PASSED|CHECKS FAILED|ROUTINE CHECKS`. test_director prints
# "FAILED n of m checks:" and test_rules "FAILED n of m rule checks:" -- neither matches --
# so 19 of 25 director entries came back BROKEN ("SUITE DID NOT REPORT") when their checks
# had in fact failed: the judge turned real catches into fake crashes. Fixing the pattern
# is the instance; the RULE is that a hand-written list of dialects drifts from the suites,
# which is exactly how the pattern got here. So the dialects are DERIVED: every
# tests/test_*.py is parsed, every `print(...)` is rendered to the text it would produce
# (f-string holes become a number, both arms of a conditional are taken), and each rendered
# summary line is run through the real judge.
#
# LIMITS, stated rather than discovered: only a summary built in the print() call itself is
# visible. One composed in a variable and printed by name renders to nothing, and the suite
# is then reported as "no summary found" -- a failure, not a pass by omission -- so it
# cannot hide, it can only force its author to print the line where it can be read.
def _render(node):
    """Every string a print() argument can produce, or [] if its shape is not understood."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        outs = [""]
        for part in node.values:
            if isinstance(part, ast.Constant):
                opts = [str(part.value)]
            elif isinstance(part, ast.FormattedValue):
                opts = (_render(part.value) if isinstance(part.value, ast.IfExp) else []) or ["7"]
            else:
                opts = ["7"]
            outs = [o + p for o in outs for p in opts]
        return outs
    if isinstance(node, ast.IfExp):
        return _render(node.body) + _render(node.orelse)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _render(node.left), _render(node.right)
        return [a + b for a in left for b in right] if left and right else []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return [re.sub(r"%[-\d.]*[sdif]", "7", s) for s in _render(node.left)]
    return []


def _prints(src: str):
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print" and node.args):
            yield from _render(node.args[0])


# A SUMMARY is a printed line that says passed/failed and counts something. Defined by
# meaning, not by the judge's own pattern -- otherwise a dialect the judge cannot read
# would also be one this scan cannot see, and the check would agree with the bug.
_SUMMARY_SENSE = re.compile(r"(?i)\b(passed|failed)\b")
_COUNTS = re.compile(r"(?i)\bchecks?\b|\bof 7\b")
_FAILLINE_SENSE = re.compile(r"\bFAIL\b")


def _dialects(src: str):
    """(green lines, red summary lines, failing-check lines) the suite can print."""
    green, red, fails = [], [], []
    for s in _prints(src):
        if _SUMMARY_SENSE.search(s) and _COUNTS.search(s):
            (green if re.search(r"(?i)\bpassed\b", s) and not re.search(r"(?i)\bfailed\b", s)
             else red).append(s)
        elif _FAILLINE_SENSE.search(s):
            fails.append(s)
    return green, red, fails


_suite_files = sorted((ROOT / "tests").glob("test_*.py"))
_dial_gaps, _derived = [], 0
for _sf_ in _suite_files:
    _g, _r, _f = _dialects(_sf_.read_text(encoding="utf-8", errors="replace"))
    if not _g or not _r or not _f:
        _dial_gaps.append(f"{_sf_.name}: derived {len(_g)} green / {len(_r)} red summary lines and "
                          f"{len(_f)} failing-check lines -- a suite whose prints cannot be read here "
                          "is a suite the judge cannot be shown to read")
        continue
    for _line in _g:
        _derived += 1
        _res = _mv.classify(0, _line + "\n", "x")
        if not _mv.GREEN.search(_line + "\n") or _res[0] != "MISSED":
            _dial_gaps.append(f"{_sf_.name}: green line {_line.strip()!r} is not read as a finished, "
                              f"green suite (classify -> {_res[0]}: {_res[1][:70]})")
    for _line in _r:
        _derived += 1
        _res = _mv.classify(1, "  [FAIL] x named\n" + _line + "\n", "named")
        if _res[0] != "caught":
            _dial_gaps.append(f"{_sf_.name}: failure summary {_line.strip()!r} is not read as a finished "
                              f"suite with a failing check (classify -> {_res[0]}: {_res[1][:70]})")
    for _line in _f:
        _derived += 1
        if not _mv.fail_lines(_line + "\n"):
            _dial_gaps.append(f"{_sf_.name}: failing-check line {_line.strip()!r} is not read as a failure")
check(len(_suite_files) >= 25 and _derived >= 80 and not _dial_gaps,
      "the judge reads every summary and failing-check line each suite's own print() calls can produce",
      f"{len(_suite_files)} suites, {_derived} lines derived: " + " | ".join(_dial_gaps[:6]))
_DIRECTOR_RED = [s for s in _prints('print(f"FAILED {len(F)} of {C} checks:")')]
check(_DIRECTOR_RED == ["FAILED 7 of 7 checks:"]
      and not re.search(r"CHECKS PASSED|CHECKS FAILED|ROUTINE CHECKS", _DIRECTOR_RED[0])
      and _mv.SUMMARY.search(_DIRECTOR_RED[0]),
      "the scan CAN fail: test_director's own summary line renders, the pre-2026-10-23 pattern "
      "does not match it, and the judge now does")

_bl_ok = _mv.baseline(["a.py", "b.py"], lambda s: (0, "ALL 3 CHECKS PASSED\n"))
_bl_red = _mv.baseline(["a.py", "b.py"],
                       lambda s: (1, _OTHER) if s == "b.py" else (0, "ALL 3 CHECKS PASSED\n"))
_bl_mute = _mv.baseline(["a.py"], lambda s: (0, "nothing was ever printed\n"))
_bl_dup = []
_mv.baseline(["a.py", "a.py", "b.py"], lambda s: (_bl_dup.append(s), (0, "ALL 1 CHECKS PASSED\n"))[1])
check(_bl_ok == [] and len(_bl_red) == 1 and _bl_red[0].startswith("b.py")
      and len(_bl_mute) == 1 and _bl_dup == ["a.py", "b.py"],
      "the baseline guard names EACH red suite, passes a green set, refuses a suite that exited 0 "
      "without printing its summary, and runs each distinct suite once",
      f"{_bl_ok} / {_bl_red} / {_bl_mute} / {_bl_dup}")

# every runner uses it, none keeps a copy of the old fallback, and each runner's
# baseline covers the suites ITS entries name (not a list kept beside them)
#
# THE FALLBACK IS JUDGED ON THE CODE, NOT ON ITS TEXT (2026-10-23). This check was a
# regex over the source, and it went red on the DOCSTRING of mutate_detector.judge(),
# which quotes the old `(named or fails)` in prose to record what changed. A check that
# fails on a comment teaches the next person to reword comments, and a regex over text
# also passes when the code is renamed into equivalence. So: parse the file and look for
# the EXPRESSION -- a boolean `or` whose operands are `named` and `fails` (bare, indexed
# or sliced) -- which a docstring and a comment cannot be.
def _has_accept_any_failure(src: str) -> bool:
    def _names(n):
        return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            if {"named", "fails"} <= set().union(*(_names(v) for v in node.values)):
                return True
    return False


_FALLBACK_LIVE = "def f(named, fails):\n    return (named or fails)[0]\n"
_FALLBACK_SLICE = "def f(named, fails):\n    return named[:1] or fails[:1]\n"
_FALLBACK_PROSE = ('def f():\n    """Used to be (named or fails) -- named[:1] or fails[:1] -- before."""\n'
                   '    # the old line was `named or fails`\n    return 1\n')
check(_has_accept_any_failure(_FALLBACK_LIVE) and _has_accept_any_failure(_FALLBACK_SLICE),
      "the fallback detector sees the expression, bare and sliced (so it CAN fail)")
check(not _has_accept_any_failure(_FALLBACK_PROSE),
      "...and is NOT tripped by a docstring or a comment that quotes it (the 2026-10-23 false positive)")

_judge_gaps, _fallback, _base_gaps = [], [], []
for _rn in ("mutate_director", "mutate_detector", "mutate_figures",
            "mutate_introspect", "mutate_ghost", "mutate_audio"):
    _rsrc = (ROOT / "tests" / f"{_rn}.py").read_text(encoding="utf-8")
    if "import mutation_verdict as verdict" not in _rsrc and _rn != "mutate_audio":
        _judge_gaps.append(f"{_rn}: does not import mutation_verdict")
    if _has_accept_any_failure(_rsrc):
        _fallback.append(_rn)
    if _rn in ("mutate_detector", "mutate_figures"):
        if "{M[i][4] for i in pick}" not in _rsrc or "verdict.baseline(" not in _rsrc:
            _base_gaps.append(f"{_rn}: baseline not derived from the picked entries")
    if _rn == "mutate_introspect" and "verdict.baseline(" not in _rsrc:
        _base_gaps.append(f"{_rn}: no baseline")
    if _rn in ("mutate_ghost", "mutate_audio") and "baseline_problem(" not in _rsrc:
        _base_gaps.append(f"{_rn}: no baseline")
    if _rn == "mutate_director" and "CHECKS PASSED" not in _rsrc:
        _base_gaps.append(f"{_rn}: baseline does not require the green summaries")
check(not _judge_gaps, "every mutation runner judges through tests/mutation_verdict.py", "; ".join(_judge_gaps))
check(not _fallback,
      "no mutation runner keeps the 'no check names it, so accept any failure' fallback "
      "(the line that counted a crash as a catch)", ", ".join(_fallback))
check(not _base_gaps,
      "every mutation runner refuses a red baseline over every suite its entries name",
      "; ".join(_base_gaps))

# each runner's judge, called as the runner calls it, on the incident's output
_judge_bad = []
for _label, _mod in (("detector", _mdet), ("figures", _mfig), ("introspect", _mint),
                     ("ghost", _mghost), ("audio", _maud)):
    if _mod is None:
        continue
    _fn = _mod.judge if hasattr(_mod, "judge") else None
    if _fn is None or _fn(1, _CRASH, _FRAG)[0] != "BROKEN" or _fn(1, _OWN, _FRAG)[0] != "caught":
        _judge_bad.append(_label)
check(not _judge_bad,
      "each runner's own judge() reports the 2026-10-23 crash as BROKEN and the named check as caught",
      ", ".join(_judge_bad))
_no_check_named = [f"{_label}: entry {i}" for _label, _mod, _at in
                   (("detector", _mdet, 5), ("figures", _mfig, 5), ("introspect", _mint, 4),
                    ("ghost", _mghost, 5), ("audio", _maud, 5)) if _mod is not None
                   for i, _m in enumerate(_mod.M) if not str(_m[_at]).strip()]
check(not _no_check_named, "every mutation entry names the check that must catch it",
      "; ".join(_no_check_named[:6]))

# mutate_director keeps its fragments in a dict keyed by entry NAME (its entries are
# 3- or 4-tuples with no slot for one), so the loop above cannot see it -- and that is
# exactly how 25 entries shipped with `FRAGS = {}`: under the strict judge none could
# be credited, the runner exited 1 on every run, and the pre-commit hook (which runs it
# unconditionally) refused every commit. The loop above could not fail for it; this can.
if _md is None:
    check(False, "every mutate_director entry names the check that must catch it (FRAGS)",
          "mutate_director did not import, so its fragments could not be read")
else:
    _dir_gaps = _md.unattributed()
    check(not _dir_gaps,
          "every mutate_director entry names the check that must catch it (FRAGS)",
          "; ".join(_dir_gaps[:6]))
    # ...and prove the guard can fail: an empty table, or one holding only whitespace
    # or a key that is no entry, must each be reported.
    _saved_frags = dict(_md.FRAGS)
    try:
        _md.FRAGS.clear()
        _all_missing = _md.unattributed()
        _md.FRAGS.update({m[0]: "   " for m in _md.MUTATIONS})
        _blank = _md.unattributed()
        _md.FRAGS.clear()
        _md.FRAGS.update(_saved_frags)
        _md.FRAGS["no such entry"] = "x"
        _stray = _md.unattributed()
    finally:
        _md.FRAGS.clear()
        _md.FRAGS.update(_saved_frags)
    check(len(_all_missing) == len(_md.MUTATIONS),
          "...an EMPTY fragment table reports every entry as unattributed (so the guard CAN fail)",
          f"{len(_all_missing)} of {len(_md.MUTATIONS)}")
    check(len(_blank) == len(_md.MUTATIONS),
          "...and a whitespace-only fragment counts as none",
          f"{len(_blank)} of {len(_md.MUTATIONS)}")
    check(any("no such entry" in g for g in _stray),
          "...and a fragment keyed to a name that is no entry is reported, not silently credited to nothing")



section("O. every suite on disk is run, and every suite run exists")
# 2026-10-18. `run_tests.bat` named two suites with a literal TAB where the
# backslash-t of "tests\test_voice.py" should have been (an editor turned
# `\t` into a tab). Python answered "can't open file ... Errno 22". It was
# loud only because a non-zero exit happened to propagate to the FAILED
# marker; the suite itself -- 76 checks -- had never run since it was added.
# The mirror image is worse and silent: a suite file nobody adds to the
# runner is simply never executed, which is how the figure drills and the
# introspection server were committed without their own checks running (the
# "a check that cannot fail" family, RULES C1 and C7: a rule that is never
# executed is not a rule). Both directions are checked, in milliseconds, here.
#
# Deliberate exclusion is allowed but must be VISIBLE: it is an entry in a
# table below with a reason, and the table is itself checked for staleness
# (the entry's file must exist and must really be absent from the runner), so
# an exclusion cannot linger after the reason is gone -- the same discipline
# as the RULES registry in section K, where retiring something means deleting
# its line in the same change. An omission leaves no trace; an exclusion does.

_TESTS = ROOT / "tests"
_SUITE_RE = re.compile(r"(?:test|mutate)_\w+\.py")
_FAILED_LINE = re.compile(r"if errorlevel 1 set FAILED=%FAILED% \S+\s*$")


def runner_invocations(text: str) -> list[tuple[int, str, bool]]:
    """(line number, raw path as written, next line records a failure)."""
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s[:3].upper() == "REM" or s.startswith("::") \
                or s[:5].lower() == "echo ":
            continue      # comments and PRINTED text are not invocations
        m = re.search(r'python(?:w)?\.exe"?\s+"?([^"\r\n]*?)"?\s*$', s)
        if not m:
            continue
        nxt = next((x.strip() for x in lines[i + 1:] if x.strip()), "")
        out.append((i + 1, m.group(1), bool(_FAILED_LINE.match(nxt))))
    return out


def bad_runner_path(raw: str) -> str:
    """'' when raw is a clean tests\\name.py, else why it is not."""
    ctrl = [hex(ord(c)) for c in raw if ord(c) < 32 or ord(c) == 127]
    if ctrl:
        return f"contains control character(s) {ctrl} (a tab for backslash-t?)"
    if not re.fullmatch(r"tests\\(?:test|mutate)_\w+\.py", raw):
        return "is not of the form tests\\<name>.py"
    return ""


def hook_suites(text: str) -> set[str]:
    """Suite files tools/pre-commit runs: explicit paths, the `for t in`
    list, and run_mutation calls. Comment lines are not invocations."""
    code = "\n".join(l for l in text.splitlines()
                     if not l.lstrip().startswith("#"))
    found = set(re.findall(r"tests/((?:test|mutate)_\w+\.py)", code))
    for m in re.finditer(r"for t in ([^;]+);\s*do", code):
        found |= {n + ".py" for n in re.findall(r"test_\w+", m.group(1))}
    found |= set(re.findall(r"run_mutation ((?:test|mutate)_\w+\.py)", code))
    return found


# -- the parser itself must be able to fail: prove it on crafted text ---------
_good = ('".venv\\Scripts\\python.exe" "tests\\test_a.py"\n'
         'if errorlevel 1 set FAILED=%FAILED% a\n')
_inv = runner_invocations(_good)
check(_inv == [(1, "tests\\test_a.py", True)]
      and not bad_runner_path(_inv[0][1]),
      "a well-formed invocation is read and accepted")
_tab = '".venv\\Scripts\\python.exe" "tests\test_a.py"\n'      # a real TAB
_inv = runner_invocations(_tab)
check(len(_inv) == 1 and "control character" in bad_runner_path(_inv[0][1]),
      "a TAB where backslash-t belongs is read and REJECTED by name")
check(runner_invocations(_good.replace("if errorlevel 1", "echo"))[0][2]
      is False,
      "an invocation whose next line does not record failure is noticed")
check(runner_invocations(
          'echo     .venv\\Scripts\\python.exe tests\\test_a.py\n') == [],
      "text an echo line merely PRINTS is not an invocation")
check(runner_invocations(
          'REM ".venv\\Scripts\\python.exe" "tests\\test_a.py"\n') == [],
      "a REM line is not an invocation")
check(hook_suites("for t in test_a test_b;  do\n x\ndone\n"
                  "run_mutation mutate_c.py 'p' 'g'\n"
                  "# \"$PY\" tests/test_zz.py\n"
                  "\"$PY\" tests/test_d.py\n")
      == {"test_a.py", "test_b.py", "mutate_c.py", "test_d.py"},
      "the hook parser reads the loop, run_mutation and explicit paths, "
      "and ignores comments")

# -- the real files ----------------------------------------------------------
# Suites that exist on disk and are deliberately NOT in run_tests.bat.
RUNNER_EXCLUDED = {
    "test_coldstart.py":
        "minutes of wall clock, not 'fast'; the runner prints it as the one "
        "to run separately and the hook runs it only when the startup chain "
        "changed (RULES C13). Run by hand or by the hook, never silently.",
}
# Suites the runner runs that tools/pre-commit does not run unconditionally.
# Empty on purpose: it held five on 2026-10-18 (the hook's header claimed to run
# every camera-free, screen-free suite and did not), and they were added to the
# hook rather than excused. An entry here is a KNOWN GAP with a reason, is
# staleness-checked below, and needs tools\install-hooks.ps1 re-run if closed
# (section N compares the installed copy).
HOOK_KNOWN_GAPS: dict[str, str] = {}

_bat = ROOT / "scripts" / "run_tests.bat"
check(_bat.exists(), "run_tests.bat exists")
_on_disk = {p.name for p in _TESTS.glob("*.py") if _SUITE_RE.fullmatch(p.name)}
check(len(_on_disk) >= 25, f"the suite scan sees the suites ({len(_on_disk)})")
if _bat.exists():
    _inv = runner_invocations(read(_bat))
    check(len(_inv) >= 25, f"run_tests.bat invocations were found ({len(_inv)})")
    _named: set[str] = set()
    for _ln, _raw, _recorded in _inv:
        _why = bad_runner_path(_raw)
        check(not _why, f"run_tests.bat line {_ln}: path is well formed",
              f"{_raw!r} {_why}")
        if _why:
            continue
        _name = _raw.split("\\")[-1]
        _named.add(_name)
        check((_TESTS / _name).is_file(),
              f"run_tests.bat line {_ln}: {_name} exists on disk",
              "the runner names a suite that is not there")
        check(_recorded,
              f"run_tests.bat line {_ln}: {_name} failing is recorded",
              "the next line must be `if errorlevel 1 set FAILED=%FAILED% "
              "<label>` or this suite cannot fail the run (RULES C1)")
    _missing = sorted(_on_disk - _named - set(RUNNER_EXCLUDED))
    check(not _missing, "every suite on disk is run by run_tests.bat",
          "NOT RUN (invisible to the build): " + ", ".join(_missing) +
          "  -- add it to run_tests.bat, or to RUNNER_EXCLUDED with a reason")
    for _n in RUNNER_EXCLUDED:
        check(_n in _on_disk and _n not in _named,
              f"exclusion {_n} is still true (file exists, runner omits it)",
              "stale exclusion: delete the entry")

    _hook = ROOT / "tools" / "pre-commit"
    if _hook.exists():
        _hs = hook_suites(read(_hook))
        _gone = sorted(n for n in _hs if not (_TESTS / n).is_file())
        check(not _gone, "tools/pre-commit names only suites that exist",
              "missing on disk: " + ", ".join(_gone))
        _unhooked = sorted(_named - _hs - set(HOOK_KNOWN_GAPS))
        check(not _unhooked,
              "every suite run_tests.bat runs is also in tools/pre-commit",
              "not in the hook: " + ", ".join(_unhooked) +
              "  -- add it there, or to HOOK_KNOWN_GAPS with a reason")
        for _n in HOOK_KNOWN_GAPS:
            check(_n in _named and _n not in _hs,
                  f"known hook gap {_n} is still a gap",
                  "stale entry: delete it")



section("P. a test script may not decide for itself how much of his machine to take")
# ---------------------------------------------------------------------------
# RULES D7 (his PC is his) and A10 (an instrument that damages what it
# measures reports noise), in the one place that has broken both.
#
# 2026-10-19: `run_tests.bat` reached `mutate_detector.py`, which fans out to
# eight workers by itself, each running a whole suite in a subprocess. 36
# python processes at 100% across all 16 logical cores; he killed the tree by
# hand. 2026-10-17: two mutation scripts started by two agents, each capped
# politely on its own, took the machine to 91% and left it "almost
# nonresponsive".
#
# Twice the remedy offered was care, and care does not survive the next
# script. His instruction was a cap the runner READS. So the number lives in
# `tests/worker_budget.py` and these checks exist to make a new mutation
# script unable to compute its own.

_BUDGET = _TESTS / "worker_budget.py"
check(_BUDGET.is_file(), "tests/worker_budget.py exists",
      "the shared cap is the only thing standing between a mutation run "
      "and his whole machine")

_mutators = sorted(_TESTS.glob("mutate_*.py"))
check(bool(_mutators), "there are mutation scripts to govern")

for _m in _mutators:
    _src = read(_m)
    check("import worker_budget" in _src,
          f"{_m.name} takes its worker count from worker_budget",
          "it must not compute its own: that is how eight workers happened")
    check("cpu_count" not in _src,
          f"{_m.name} does not size itself from the core count",
          "cpu_count here means 'take the machine'; call budget.workers()")
    check(not re.search(r"ThreadPoolExecutor\(\s*\d", _src),
          f"{_m.name} has no hard-coded worker count",
          "a literal is a cap nobody can lower from outside")
    check("budget.exclusive(" in _src,
          f"{_m.name} takes the one mutation slot",
          "a per-script cap cannot see a SECOND script: that was 2026-10-17")

if _BUDGET.is_file():
    _bsrc = read(_BUDGET)
    # The cap must actually bite on HIS machine, or it is decoration.
    _cores = os.cpu_count() or 4
    sys.path.insert(0, str(_TESTS))
    import worker_budget as _wb        # noqa: E402
    check(_wb.HARD_CEILING < _cores,
          f"the worker ceiling ({_wb.HARD_CEILING}) is below this machine's "
          f"core count ({_cores})",
          "a ceiling at or above the core count permits the incident it "
          "exists to prevent")
    check(_wb.workers(18) <= _wb.HARD_CEILING,
          "a script asking for 18 workers is still held to the ceiling",
          f"got {_wb.workers(18)}")
    check(_wb.workers(1) == 1,
          "a script asking for 1 worker gets 1",
          "the budget is an upper bound, never a request for more")
    check(_wb.workers(8) <= _cores,
          "the budget never exceeds the machine",
          f"got {_wb.workers(8)} on {_cores} cores")
    # It must REFUSE, not wait. An unbounded wait is this project's most
    # expensive recurring fault (RULES A9/G).
    check("class Busy" in _bsrc and "raise Busy" in _bsrc,
          "a second mutation run is refused",
          "refused and reported, never queued behind the first")
    check("sleep" not in _bsrc,
          "the mutation slot never waits for the holder",
          "a wait here is an unbounded wait on another test run")
    # os.kill(pid, 0) TERMINATES the process on Windows.
    # The Windows path is everything from `import ctypes` on. Splitting at the
    # `!= "win32"` test instead handed the check the POSIX branch as well,
    # where `os.kill(pid, 0)` is the correct idiom -- so it went red on code
    # that was right. A check pointed at the wrong half of a function reports
    # a fault that is not there, which is the same cost as missing one.
    _win = _bsrc.split("import ctypes")[-1]
    check("os.kill" not in _win,
          "the liveness probe never calls os.kill on the Windows path",
          "python maps os.kill to TerminateProcess: the probe would KILL "
          "the run it is asking about")


# ---------------------------------------------------------------------------
section("Q. every asset folder has an override, and every runner passes it")
# Third occurrence of one fault (src/lsa/artwork/assets.py has the history): a new
# folder of the app's own pictures arrives, its loader resolves it from the source
# location, and the mutation runners -- which run each suite in a COPIED tree of
# `src`, `tests` and `tools` -- cannot see it, so the suite crashes before it can
# report. These checks make a new folder, or a runner that stops passing one, fail
# HERE instead of in a harness run.
import shutil as _shutil
import subprocess as _subprocess
import tempfile as _tempfile

sys.path.insert(0, str(ROOT / "tests"))
from asset_env import ASSET_DIRS as _AD, asset_env as _aenv, scratch_env as _senv   # noqa: E402
# THE CLEAN REPOSITORY (2026-10-23): ROOT (this suite's root) is app/; the artwork is
# <project>/resources/<name>. `_PROJ` is the repository root, `_AP` the folder name.
from asset_env import ASSET_PARENT as _AP, PROJECT as _PROJ, member_source as _msrc   # noqa: E402
from owned_files import OWNED as _OWNED_Q                                             # noqa: E402

_art = [p for p in (SRC / "artwork").glob("*.py") if p.name != "assets.py"]
_art_src = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in _art}

check(len(set(_AD.values())) == len(_AD) and all(v.startswith("LSA_") for v in _AD.values()),
      "the asset table maps each folder to its own LSA_ variable")
check(all((_PROJ / _AP / n).is_dir() for n in _AD),
      "every folder in the asset table exists on disk, under <project>/resources/",
      str([n for n in _AD if not (_PROJ / _AP / n).is_dir()]))
check(_AP == "resources" and all(o_.startswith(_AP + "/") for o_ in _OWNED_Q if o_.split("/")[0] != "config"),
      "an owned asset folder is named under the asset parent the table says (owned_files spells it out)",
      str(_OWNED_Q))
# 1. A loader names a project folder that the table does not know.
_FOLDER_USE = re.compile(r'(?:PROJECT_ROOT|parents\[[34]\])\s*/\s*"(\w+)"')
_DERIVED = {"cache"}        # written by the app, never read as input art: nothing to override
_src_all = {str(p_.relative_to(SRC)): p_.read_text(encoding="utf-8", errors="replace")
            for p_ in SRC.rglob("*.py") if p_.name != "assets.py"}     # the whole app, not just artwork/
_unknown = sorted({(f, n) for f, t in _src_all.items() for n in _FOLDER_USE.findall(t)
                   if n not in _AD and n not in _DERIVED})
check(not _unknown, "every project folder an artwork loader reads is in the asset table (assets.py)",
      f"{_unknown}: add it to ASSET_DIRS and read it with asset_dir(name)")
# 2. A loader reads its own override instead of the table (the next one to forget).
_own = sorted(f for f, t in _art_src.items() if re.search(r'environ\.get\(\s*"LSA_', t))
check(not _own, "no artwork loader reads an LSA_ override directly: they all go through asset_dir()",
      str(_own))
_called = sorted({n for p_ in (SRC).rglob("*.py") if p_.name != "assets.py"
                  for n in re.findall(r'asset_dir\(\s*"(\w+)"', p_.read_text(encoding="utf-8", errors="replace"))
                  if n not in _AD})
check(not _called, "every asset_dir(name) call anywhere in src names a folder in the table",
      f"{_called}: the table has no such folder")
_unrouted = sorted(n for n in _AD if not any(f'asset_dir("{n}")' in t for t in _src_all.values()))
check(not _unrouted, "every folder in the table is read through asset_dir() by some loader",
      str(_unrouted))
# 3. Every runner that builds a scratch tree builds its environment from the table.
_runners = sorted(p for p in list((ROOT / "tests").glob("*.py")) + list((ROOT / "tools").glob("*.py"))
                  if ("copytree" in p.read_text(encoding="utf-8", errors="replace")
                      or "copy_scratch_tree" in p.read_text(encoding="utf-8", errors="replace"))
                  and p.name not in ("test_rules.py", "asset_env.py"))
_bad_runner = [p.name for p in _runners
               if "scratch_env(" not in p.read_text(encoding="utf-8", errors="replace")
               or re.search(r'LSA_(%s)\b' % "|".join(v[4:] for v in _AD.values()),
                         p.read_text(encoding="utf-8", errors="replace"))]
check(len(_runners) >= 3 and not _bad_runner,
      "every runner that copies a tree passes the asset overrides from scratch_env(), none spells one out",
      f"{_bad_runner} (runners found: {[p.name for p in _runners]})")
# 3b. AND IT PASSES THE RIGHT ROOT. `scratch_env(X)` names `X/resources/<folder>`,
# so the argument decides whether the child reads real artwork, a scratch copy, or
# nothing at all. `ROOT` is `app/`, and the artwork is BESIDE it -- correct only
# while the app was its own repository root, which it stopped being on 2026-10-23.
# Two of five runners still passed ROOT: their children got LSA_SOUNDS pointing at
# `app/resources/sounds`, which exists nowhere, and mutate_audio reported "his 9mm
# sample does not exist" for a file sitting safely on disk.
#
# The check above could not see it, because it asked whether scratch_env was CALLED
# and the one below asks what it was called WITH -- the seam, again (RULES C14). The
# existing proof that scratch_env(d) points into the copy tests the FUNCTION; nothing
# tested the call.
_wrong_root = []
for _rp in _runners:
    try:
        _rt = ast.parse(_rp.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        continue
    for _n in ast.walk(_rt):
        if (isinstance(_n, ast.Call) and isinstance(_n.func, ast.Name)
                and _n.func.id == "scratch_env" and _n.args
                and isinstance(_n.args[0], ast.Name) and _n.args[0].id == "ROOT"):
            _wrong_root.append(f"{_rp.name}:{_n.lineno}")
check(not _wrong_root,
      "no runner passes ROOT to scratch_env(): the assets are beside app/, not in it",
      f"{_wrong_root} -- pass the scratch tree when it holds copies of the artwork, "
      f"or PROJECT when the child should read the real (read-only) folders")
# 4. The environment really carries every folder, and does not depend on where the source is.
_e = _senv(_PROJ)
check(all(v in _e for v in _AD.values()) and _e["QT_QPA_PLATFORM"] == "offscreen",
      "scratch_env() carries every override and the offscreen platform")
check(_aenv("X:/elsewhere", targets="Y:/mine")["LSA_TARGETS"] == str(Path("Y:/mine"))
      and _aenv("X:/elsewhere")["LSA_IMAGES"] == str(Path("X:/elsewhere") / _AP / "images")
      and _e["LSA_IMAGES"] == str(_PROJ / _AP / "images"),
      "asset_env() takes the root it is given and adds resources/, and a per-folder override wins over it")
# 4b. THE SCRATCH TREE HOLDS WHAT THE SUITES READ (2026-10-23).
#     Q as first written proved the OVERRIDES reach a runner. It never asked what the
#     SUITES read relative to their own location, so `config/`, `sounds/` and
#     `.venv/pyvenv.cfg` were missing from the copy and three runners refused with a red
#     baseline -- the third repeat of one fault, after `animations/` and `ui/`, with the
#     guard for it green. Overrides cover the folders the APP resolves through
#     asset_dir(); suites also join `ROOT / "name"` themselves. So: for every suite a
#     runner names, find every top-level folder it joins onto a root, and require the
#     runner's tree to hold it. Derived from the suites' own source -- no list here.
#     LIMIT: a path built dynamically (`ROOT / name` with a variable) is invisible to
#     the scan; the fall-back for that is the baseline guard, which refuses a red tree.
from asset_env import (copy_scratch_tree as _cst, scratch_members as _members,       # noqa: E402
                       NEVER_COPIED as _NEVER)
from owned_files import OWNED as _OWNED                                               # noqa: E402

_TOP = {p_.name for p_ in ROOT.iterdir() if p_.is_dir()}
# A top-level reference is `<name or ) or ]> / "folder"`. A segment after another STRING
# (`ROOT / "src" / "lsa" / "tools"`) is a subpackage that merely shares a folder's name:
# test_ghost.py reads src/lsa/tools/camera_lab.py and is not reading the project's tools/.
_REF = re.compile(r'[\w\)\]]\s*/\s*["\']([\w.\-]+)(?=[/"\'\\])')
_SUITE_LIT = re.compile(r'["\'](test_\w+\.py)["\']')
_first = lambda rel: rel.replace("\\", "/").split("/")[0]                              # noqa: E731


def _needs(suite_text):
    out = {n for n in _REF.findall(suite_text) if n in _TOP}
    if "owned_files" in suite_text or "OwnedFiles" in suite_text:
        out |= {_first(o) for o in _OWNED}              # fingerprints these, relative to ITS root
    return out


_gaps = []
_audited = 0
for _p in _runners:
    _t = _p.read_text(encoding="utf-8", errors="replace")
    if "copy_scratch_tree(" in _t:
        _held = {_first(m_) for m_ in _members()}
    else:                                               # a hand-written copy: what it names, literally
        _held = set(re.findall(r'copy(?:tree|2)\(\s*ROOT\s*/\s*"([\w.\-]+)"', _t))
    for _s in sorted({s_ for s_ in _SUITE_LIT.findall(_t) if (ROOT / "tests" / s_).exists()}):
        _audited += 1
        _need = _needs((ROOT / "tests" / _s).read_text(encoding="utf-8", errors="replace"))
        for _n in sorted(_need - _held - {"tests"}):
            _gaps.append(f"{_p.name} runs {_s}, which reads '{_n}/' -- not in its scratch tree"
                         + (f" (deliberately never copied: {_NEVER[_n]})" if _n in _NEVER else ""))
check(_audited >= 8 and not _gaps,
      "every folder a suite joins onto its root is in the scratch tree of each runner that runs it",
      f"{_gaps} (suites audited: {_audited})")
check(_needs("x = ROOT / 'nope'; z = ROOT / \"src\"; y = p / \"config/calibration.json\"") >= {"src", "config"}
      and "tools" not in _needs('c = ROOT / "src" / "lsa" / "tools" / "camera_lab.py"')
      and "tools" in _needs('c = ROOT / "tools" / "x.py"'),
      "...and the scan CAN see a folder, in both spellings, and does not mistake the src/lsa/tools "
      "subpackage for the project's tools/ (so a gap above is a finding, not a blind spot)")
check(_needs("from owned_files import OwnedFiles") >= {_first(o_) for o_ in _OWNED},
      "...and a suite that fingerprints the owned files is taken to read each of them")
# 4c. The copy is real, complete and a COPY. Without this the table above could list a
#     folder and the helper still not deliver it, and a symlink would make 'a rogue write
#     lands on the copy' false.
_td = Path(_tempfile.mkdtemp(prefix="lsa_scratch_"))
try:
    _dst = _td / "t"
    _cst(ROOT, _dst)
    _present = [m_ for m_ in _members() if _msrc(ROOT, m_).exists()]
    _lost = [m_ for m_ in _present if not (_dst / m_).exists()]
    check(len(_present) >= 8 and not _lost,
          "copy_scratch_tree() delivers every member the root has: config, ui, targets, sounds, pyvenv.cfg and the rest",
          f"missing from the copy: {_lost}")
    _env = _senv(_dst)
    check(all(str(_dst) in _env[v_] for v_ in _AD.values()),
          "scratch_env(d) points every override at the scratch COPY, none at the real folders",
          str({v_: _env[v_] for v_ in _AD.values() if str(_dst) not in _env[v_]}))
    _own = next((o_ for o_ in _OWNED if _msrc(ROOT, o_).is_file()), None)
    if _own is not None:
        _before = _msrc(ROOT, _own).read_bytes()
        (_dst / _own).write_bytes(b"ROGUE WRITE")
        check(_msrc(ROOT, _own).read_bytes() == _before and (_dst / _own).read_bytes() == b"ROGUE WRITE",
              f"a write to the scratch copy of {_own} leaves his real file byte-identical")
finally:
    _shutil.rmtree(_td, ignore_errors=True)
# 5. THE FAULT ITSELF, reproduced: a copy of the tree with only `src` cannot find the bullet
#    without LSA_UI, and finds it with it. Without this a green table proves nothing.
_tmp = Path(_tempfile.mkdtemp(prefix="lsa_assets_"))
try:
    _shutil.copytree(ROOT / "src", _tmp / "src", ignore=_shutil.ignore_patterns("__pycache__"))
    _code = "import sys; sys.path.insert(0, 'src'); from lsa.artwork import ui; ui.load_bullet()"

    def _child(env):
        return _subprocess.run([sys.executable, "-c", _code], cwd=str(_tmp), env=env,
                               capture_output=True, text=True, timeout=120)
    _with = _child(_senv(_PROJ))
    _without = _child({k: v for k, v in _senv(_PROJ).items() if k != "LSA_IMAGES"})
    check(_with.returncode == 0,
          "in a copied tree holding only src, the bullet loads when the overrides are passed",
          _with.stderr.strip().splitlines()[-1] if _with.stderr.strip() else "")
    check(_without.returncode != 0,
          "...and does NOT without LSA_IMAGES (so this check can fail, and the hazard it guards is real)")
finally:
    _shutil.rmtree(_tmp, ignore_errors=True)

# The folded-away ui/ folder must not survive anywhere: not in the table, not on disk, not as an
# override variable in any source, test or script. (Name assembled so this line is not its own hit.)
_dead = "LSA_" + "UI"
_hits = sorted(str(p_.relative_to(ROOT)) for p_ in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.ps1"))
               + list(ROOT.rglob("*.bat")) + list(ROOT.rglob("*.cmd")) + list(ROOT.rglob("*.spec"))
               if "_out" not in p_.parts and ".venv" not in p_.parts
               and _dead in code_only(p_))
check("ui" not in _AD and not (_PROJ / _AP / "ui").exists() and not _hits,
      "the ui/ asset folder is gone everywhere: not in the table, not on disk, no override named in any file",
      str(_hits))


# ---------------------------------------------------------------------------
section("R. frozen or not, read-only data and writable state resolve where they belong")
# PACKAGING STAGE 1 (2026-10-20). Nothing in the app knew it might be frozen: the root was
# `Path(__file__).resolve().parents[3]` in three places, which overshoots by one level in a
# PyInstaller bundle (there is no `src`), so every picture, sound and voice line would fail
# to load. TWO questions get two answers: the app's read-only data comes from the bundle
# (sys._MEIPASS); the user's writable state (config, logs, cache, recordings) comes from the
# folder beside the .exe -- never from _MEIPASS, which is a temp dir in one-file mode.
#
# `tests/frozen_probe.py` imports the real modules fresh in a child process with every LSA_
# override removed (so the DEFAULTS are measured) and prints where each path landed. The same
# probe is run NOT frozen against the same layout, which is what makes this able to fail: if the
# paths did not follow sys.frozen, the two runs would agree and the discrimination would break.
import json as _json


def _probe(src_dir, bundle, app, frozen):
    r = _subprocess.run([sys.executable, str(ROOT / "tests" / "frozen_probe.py"), str(src_dir),
                         str(bundle), str(app), "1" if frozen else "0"],
                        cwd=_tempfile.gettempdir(), capture_output=True, text=True, timeout=180)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("PROBE_JSON ")]
    if not line:
        return None, (r.stderr.strip().splitlines() or ["no output"])[-1]
    return _json.loads(line[0][len("PROBE_JSON "):]), ""


def _inside(path_s, folder):
    try:
        Path(path_s).resolve().relative_to(Path(folder).resolve())
        return True
    except (ValueError, OSError):
        return False


_WRITABLE = {"recordings": "recordings", "calibration": "config", "settings": "config",
             "log": "logs", "cache_file": "cache", "introspect_settings": "config"}
_lay = Path(_tempfile.mkdtemp(prefix="lsa_frozen_"))
try:
    _bundle = _lay / "dist" / "ScreenPlink" / "_internal"       # sys._MEIPASS (one-folder, PyInstaller 6)
    _app = _lay / "dist" / "ScreenPlink"                        # the folder holding the .exe
    for _n in _AD:
        (_bundle / _AP / _n).mkdir(parents=True)             # frozen: <MEIPASS>/resources/<name>
    _fz, _err = _probe(SRC.parent, _bundle, _app, True)
    check(_fz is not None, "the frozen-layout probe ran", _err)
    if _fz is not None:
        _bad = sorted(n for n in _AD if _fz.get(n, "ERROR").startswith("ERROR")
                      or not _inside(_fz[n], _bundle / _AP / n))
        check(not _bad, "frozen: every asset folder resolves inside the bundle (sys._MEIPASS)",
              str({n: _fz.get(n) for n in _bad}))
        _badw = sorted(k for k, d_ in _WRITABLE.items()
                       if _fz.get(k, "ERROR").startswith("ERROR") or not _inside(_fz[k], _app / d_))
        check(not _badw, "frozen: every writable path resolves inside the folder beside the .exe",
              str({k: _fz.get(k) for k in _badw}))
        check(not any(_inside(_fz[k], _bundle) for k in _WRITABLE if k in _fz),
              "frozen: NOTHING writable resolves inside the bundle (a temp dir in one-file mode)",
              str({k: _fz[k] for k in _WRITABLE if k in _fz and _inside(_fz[k], _bundle)}))
    _plain, _err2 = _probe(SRC.parent, _bundle, _app, False)
    check(_plain is not None, "the same probe ran NOT frozen", _err2)
    if _plain is not None:
        check(all(_inside(_plain[n], _PROJ / _AP / n) for n in _AD),
              "not frozen: every asset folder is the project's own, under <project>/resources/",
              str({n: _plain[n] for n in _AD if not _inside(_plain[n], _PROJ / _AP / n)}))
        check(all(_inside(_plain[k], ROOT / d_) for k, d_ in _WRITABLE.items()),
              "not frozen: every writable path is under app/ (config, logs, cache, recordings together)",
              str({k: _plain[k] for k, d_ in _WRITABLE.items() if not _inside(_plain[k], ROOT / d_)}))
        check(not any(_inside(_plain[n], _bundle) for n in _AD),
              "...and none is in the fake bundle, so the frozen checks above CAN fail: they pass "
              "only when the paths follow sys.frozen")
finally:
    _shutil.rmtree(_lay, ignore_errors=True)

# The call-time contract, in-process: the answer follows sys.frozen the moment it changes, and an
# override variable beats the bundle. A refactor that captures the root at import (the failure the
# docstring in assets.py is about) goes red here.
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("_lsa_assets_r", SRC / "artwork" / "assets.py")
_am = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(_am)
_had = {k: getattr(sys, k) for k in ("frozen", "_MEIPASS") if hasattr(sys, k)}
_exe, _env_voice = sys.executable, os.environ.get("LSA_VOICE")
try:
    os.environ.pop("LSA_VOICE", None)
    sys.frozen, sys._MEIPASS, sys.executable = True, str(ROOT / "b"), str(ROOT / "a" / "x.exe")
    _frozen_voice = _am.asset_dir("voice")
    _frozen_cfg = _am.user_dir("config")
    os.environ["LSA_VOICE"] = str(ROOT / "elsewhere")
    _over = _am.asset_dir("voice")
    os.environ.pop("LSA_VOICE")
    del sys.frozen
    _plain_voice = _am.asset_dir("voice")
finally:
    for _k in ("frozen", "_MEIPASS"):
        if _k in _had:
            setattr(sys, _k, _had[_k])
        else:
            sys.__dict__.pop(_k, None)
    sys.executable = _exe
    if _env_voice is None:
        os.environ.pop("LSA_VOICE", None)
    else:
        os.environ["LSA_VOICE"] = _env_voice
check(_frozen_voice == ROOT / "b" / _AP / "voice" and _frozen_cfg == ROOT / "a" / "config",
      "asset_dir follows sys._MEIPASS and user_dir follows the executable, read at CALL time",
      f"{_frozen_voice} {_frozen_cfg}")
check(_over == ROOT / "elsewhere", "an LSA_ override beats the frozen bundle", str(_over))
check(_plain_voice == _PROJ / _AP / "voice",
      "unfreezing returns the source layout at once (nothing captured at import)", str(_plain_voice))
_all_text = "".join(_src_all.values())
check(all(f'asset_dir("{n}")' in _all_text for n in ("sounds", "voice", "images")),
      "the app reads its sounds, voice lines and splash through the asset table")
check("PROJECT_ROOT =" not in "".join(t for k, t in _src_all.items() if Path(k).name != "load.py"),
      "no second definition of the project root exists outside assets.py (load.py only aliases it)")


# ---------------------------------------------------------------------------
print("\n=== S. whatever takes the screen gives it back ===")
# 2026-10-20: the app's own fullscreen always-on-top "Calibrating" cover was
# hidden four times and nothing ever put the target screen back in front;
# Windows minimised it and the user watched the app "hide itself". The cure is
# one method, `_reclaim_screen`, called from every teardown of a cover. This
# check makes a cover added later FAIL THE BUILD until it is registered and its
# teardown reclaims (RULES E2: a list nobody is forced to read is a convention).
import ast as _ast_s
_lab_src = (SRC / "tools" / "camera_lab.py").read_text(encoding="utf-8", errors="replace")
_m_cov = re.search(r"SCREEN_COVERS\s*=\s*\(([^)]*)\)", _lab_src)
_registered = set(re.findall(r"\"(\w+)\"", _m_cov.group(1))) if _m_cov else set()
# Windows that cover the screen: any class in a file that goes fullscreen or
# stays on top. TargetWindow IS the screen; Flasher (tools/align_check.py) is a
# separate tool process with its own lifetime, never run inside CameraLab.
_STANDALONE = {"TargetWindow", "Flasher"}
_found = set()
for _rel, _txt in _src_all.items():
    if "showFullScreen(" in _txt or "WindowStaysOnTopHint" in _txt:
        _found |= set(re.findall(r"^class (\w+)\((?:QWidget|QMainWindow)", _txt, re.M))
_unlisted = sorted(_found - _registered - _STANDALONE)
check(bool(_registered) and not _unlisted,
      "every window class that covers the screen is registered in CameraLab.SCREEN_COVERS",
      f"unregistered: {_unlisted}; registered: {sorted(_registered)}; found: {sorted(_found)}")
_TEARDOWN = {"FlashWindow": "_close_flash", "Splash": "_splash_down"}
_tree_s = _ast_s.parse(_lab_src)
_methods = {n.name: n for c in _tree_s.body if isinstance(c, _ast_s.ClassDef)
            and c.name == "CameraLab" for n in c.body if isinstance(n, _ast_s.FunctionDef)}
for _cover in sorted(_registered):
    _fn = _TEARDOWN.get(_cover)
    check(_fn is not None and _fn in _methods,
          f"cover {_cover}: a teardown method is registered for it in this check", str(_fn))
    if _fn in _methods:
        _calls = {n.func.attr for n in _ast_s.walk(_methods[_fn])
                  if isinstance(n, _ast_s.Call) and isinstance(n.func, _ast_s.Attribute)}
        check("_reclaim_screen" in _calls,
              f"cover {_cover}: its teardown ({_fn}) calls _reclaim_screen", str(sorted(_calls)))
_ncalls = len(re.findall(r"force_to_front\(", _lab_src))
check(_ncalls >= 3 and "def _reclaim_screen" in _lab_src,
      "force_to_front is reached from the teardowns through the one _reclaim_screen "
      "(its callers: the measurement gate, pattern calibration, and _reclaim_screen)",
      f"{_ncalls} mentions")


# ---------------------------------------------------------------------------
section("T. the tiered runner: every suite reachable, nothing skipped silently")
# ---------------------------------------------------------------------------
#
# scripts/run_affected.ps1 runs suites BY AREA AFFECTED and escalates to the full
# regression on an invasive change (2026-10-21). A tiered runner has a failure
# the full one cannot have: a suite that NO changed path can ever name is never
# run by anyone, and a skipped suite looks exactly like a passed one. So:
#   * every suite on disk is ALWAYS, SLOW, or named by some path (reachable)
#   * the slow tier is exactly the slow runners on disk, each with a cost entry
#   * the invasive list names only paths that exist
#   * a plan accounts for EVERY suite: ran or skipped-with-a-reason, none dropped
#   * the runner script exists and still reads the planner
import importlib.util as _ilu_s

_sp_path = ROOT / "tools" / "suite_plan.py"
check(_sp_path.is_file(), "tools/suite_plan.py exists")
if _sp_path.is_file():
    _sp_spec = _ilu_s.spec_from_file_location("suite_plan_under_test", _sp_path)
    _sp = _ilu_s.module_from_spec(_sp_spec)
    _sp_spec.loader.exec_module(_sp)

    _disk_s = {p.name for p in _TESTS.glob("*.py") if _SUITE_RE.fullmatch(p.name)}
    _disk_tests = {n for n in _disk_s if n.startswith("test_")}
    _disk_muts = {n for n in _disk_s if n.startswith("mutate_")}

    # reachability: a suite nothing can ever name is never run by the tiers
    _reach = _sp.all_reachable()
    _orphans_s = sorted(_disk_tests - set(_sp.ALWAYS) - set(_sp.SLOW) - _reach)
    check(not _orphans_s, "every suite is ALWAYS, SLOW, or named by some changed path",
          "UNREACHABLE by the tiered runner: " + ", ".join(_orphans_s) +
          "  -- nothing it covers is imported by any source file, tool or helper")
    check(set(_sp.ALWAYS) <= _disk_tests and "test_rules.py" in _sp.ALWAYS,
          "the always tier exists on disk and contains test_rules.py",
          str(_sp.ALWAYS))

    # the slow tier IS the slow runners on disk
    check(set(_sp.SLOW) - _disk_s == set(),
          "every suite named in the slow tier exists on disk",
          "stale: " + ", ".join(sorted(set(_sp.SLOW) - _disk_s)))
    check(_disk_muts <= set(_sp.SLOW),
          "every mutation runner on disk is in the slow tier (else it is never run)",
          "NOT TIERED: " + ", ".join(sorted(_disk_muts - set(_sp.SLOW))))
    check("test_coldstart.py" in _sp.SLOW, "test_coldstart.py is in the slow tier")
    check(all(isinstance(v, int) or v is None for v in _sp.SLOW.values()),
          "each slow runner carries a measured cost (or None = not measured)")

    # the invasive list names only things that exist
    _gone_s = [pth for pth in _sp.INVASIVE if not (ROOT / pth.rstrip("/")).exists()]
    check(not _gone_s and len(_sp.INVASIVE) >= 8,
          "the invasive-path list names only paths that exist",
          "STALE: " + (", ".join(_gone_s) or "none missing") +
          "; entries: %d (the list must keep its 8 core paths)" % len(_sp.INVASIVE))

    # the policy actually fires, and can say no
    _inv_plan = _sp.make_plan("files", ["src/lsa/detect/laser.py"])
    check(_inv_plan["escalated"] and _inv_plan["reasons"],
          "touching the detector escalates, with a stated reason")
    _quiet = _sp.make_plan("files", ["src/lsa/modes/shortcuts.py"])
    check(not _quiet["escalated"] and "test_rules.py" in
          [r["name"] for r in _quiet["run"]] and "test_coldstart.py" not in
          [r["name"] for r in _quiet["run"]],
          "a modes/shortcuts.py change does not escalate, still runs test_rules, "
          "and does not run the slow tier")
    _big = _sp.make_plan("files", ["docs%d.md" % i for i in range(_sp.LARGE_FILES + 1)])
    check(_big["escalated"], "a change touching more than LARGE_FILES files escalates")
    _doc = _sp.make_plan("files", ["RULES.md"])
    check([r["name"] for r in _doc["run"]] == list(_sp.ALWAYS),
          "a documentation-only change runs the always tier and nothing else",
          str([r["name"] for r in _doc["run"]]))

    # nothing is dropped: run + skipped is the whole universe, with no overlap
    for _lab, _pl in (("quiet", _quiet), ("invasive", _inv_plan), ("doc", _doc)):
        _r = [x["name"] for x in _pl["run"]]
        _k = [x["name"] for x in _pl["skipped"]]
        check(sorted(_r + _k) == sorted(_disk_s) and not set(_r) & set(_k),
              "plan '%s': every suite is either run or skipped with a reason" % _lab,
              "dropped: %s  doubled: %s" % (sorted(_disk_s - set(_r) - set(_k)),
                                            sorted(set(_r) & set(_k))))
        check(all(x["why"] for x in _pl["skipped"]),
              "plan '%s': every skip carries a reason" % _lab)

    # the runner script exists, reads the planner, and records the slow costs
    _ra = ROOT / "scripts" / "run_affected.ps1"
    check(_ra.is_file(), "scripts/run_affected.ps1 exists")
    if _ra.is_file():
        _ra_t = read(_ra)
        check("suite_plan.py" in _ra_t and "QT_QPA_PLATFORM" in _ra_t
              and "LSA_NOPAUSE" in _ra_t,
              "run_affected.ps1 calls the planner and sets the offscreen environment")
        _nocost = [n for n in _sp.SLOW if n.removesuffix(".py") not in _ra_t]
        check(not _nocost, "run_affected.ps1's header records every slow runner",
              "not named in the header: " + ", ".join(_nocost))
        check("exit 1" in _ra_t and "SKIPPED" in _ra_t,
              "run_affected.ps1 fails on a failed suite and prints what it skipped")


# ---------------------------------------------------------------------------
section("U. one path from the app to the room, and the microphone is told")
# ---------------------------------------------------------------------------
#
# 2026-10-16 / 2026-10-23. `play_gunshot` was "the one path from a shot to the
# listener" -- true as worded, while the operator console's Test button called
# `mixer.play(...)` directly: a real gunshot at session volume with no
# registration, no mute, no discount and no log line. A calibrated microphone
# heard it and counted it ("HEARD a shot with no detection"), so the gun test's
# N was wrong by the number of times it was pressed. The effect that follows a
# bang, the dry-pull pop, the selection cue and a target appearing each had their
# own copy of the mute: the same fault, one level down. One instance fixed and one
# standing is the dominant failure mode on this project (Part E).
#
# So: a call to `.play(` anywhere in src/ outside the module whose job is to tell
# the listener, and outside ONE named exception with its reason (speech). AST, so a
# comment or a string cannot satisfy it; the helper is shared with
# tests/test_audio_own.py, which re-runs it on the mutation runner's scratch copy.
import audio_paths as _ap

_bad_paths = _ap.violations(ROOT / "src")
check(not _bad_paths,
      "no second path from the app to the speakers or to the listener",
      "; ".join(_bad_paths) + "  -- use lsa.audio.bang.play_own_sound; the "
      "microphone must be told what the app plays")
_play_where = {rel_ for rel_, _ln, _fn in _ap.play_sites(ROOT / "src")}
for _allowed, _why in _ap.PLAY_ALLOWED.items():
    check(_allowed in _play_where,
          f"allowlist entry {_allowed} still calls .play( (else delete the entry)",
          _why)
_mute_where = {(rel_, fn_) for rel_, _ln, fn_ in _ap.mute_for_sites(ROOT / "src")}
for _rel_m, _fns in _ap.MUTE_FOR_ALLOWED.items():
    check(all((_rel_m, f_) in _mute_where for f_ in _fns),
          f"allowlist entry {_rel_m}: mute_for( is still called there (else delete the entry)",
          str(sorted(_mute_where)))
check(not (_ap.PLAY_ALLOWED.keys() - {"lsa/audio/bang.py", "lsa/audio/voice.py"}),
      "the .play( allowlist has exactly two entries: the one path, and speech",
      "a third entry is a third path; justify it in audio_paths.py AND here")
# And the rule's own detector must be able to fail, on crafted code.
import tempfile as _tf_u
with _tf_u.TemporaryDirectory() as _td_u:
    _lsa_u = Path(_td_u) / "lsa" / "audio"
    _lsa_u.mkdir(parents=True)
    (_lsa_u / "bang.py").write_text("def f(m):\n    m.play('x')\n")
    (Path(_td_u) / "lsa" / "other.py").write_text(
        "def g(m, l):\n    m.play('x')\n    l.mute_for(1.0)\n    l.expect_own_sound(1)\n")
    _crafted = _ap.violations(Path(_td_u))
    check(len(_crafted) == 3 and all("other.py" in c_ for c_ in _crafted),
          "the path detector reports a stray play, mute_for and expect_own_sound by name",
          str(_crafted))


# ---------------------------------------------------------------------------
def _report_not_run() -> None:
    """Name what did not run, so a skip can never be mistaken for a sweep.

    Deliberately worded with neither "passed" nor "failed" in it: section M
    reads every print() in every suite and treats any line carrying one of
    those words beside a count as the suite's own verdict, which the judge in
    `mutation_verdict` must then be able to parse. This is not a verdict.
    """
    if not NOT_RUN:
        return
    print(f"\n{len(NOT_RUN)} SECTION(S) DID NOT RUN -- neither green nor red.")
    print("They read files the author keeps outside this repository:")
    for _n in NOT_RUN:
        print("  - " + _n)


print(f"\n{'=' * 64}")
if FAILED:
    print(f"FAILED {len(FAILED)} of {CHECKS} rule checks:")
    for f in FAILED:
        print(f"  - {f}")
    print("\nThese are rules from RULES.md. Fix the code, or change the rule "
          "deliberately -- but not by deleting the check.")
    _report_not_run()
    sys.exit(1)
print(f"ALL {CHECKS} RULE CHECKS PASSED")
_report_not_run()
