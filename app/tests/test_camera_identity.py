"""A camera is identified by what it DOES, never by its name (RULES A11).

A11 was "judgement" in the registry until 2026-11-07, which is to say it was
not enforced by anything. It has now been broken twice at a cost, in opposite
directions, by the same confusion:

  2026-10-19  the monitor had power-saved. The app read a flash difference of
              0.0, decided its healthy camera was blind, cycled to the C920
              (blind taped over its lens), wedged the capture thread letting
              go of the first, and shut down. Three launches, each dead about
              27 seconds in. The sleeping-display guard was written to stop it.

  2026-11-07  a reboot moved the MSMF name `USB Video Device` onto the blinded
              C920 and gave the Kiyo Pro the name `HD Pro Webcam C920`.
              `resolve_device` matches on the remembered name first, so the
              app opened the camera with the blind on it, saw black, and took
              the sleeping-display branch -- announcing a sleeping monitor at a
              lit 4K panel, while the camera-cycling code written in October
              for exactly this case sat unreachable below the guard.

Both are one mistake: concluding something about the DISPLAY from a
measurement taken through the one instrument that may itself be at fault.

Each section says what goes red when the behaviour it guards is reverted.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import re
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from owned_files import isolate_outputs                          # noqa: E402
isolate_outputs("camera_identity")

from lsa.camera.devices import CameraInfo, _is_virtual           # noqa: E402
from lsa.display import power                                    # noqa: E402
from lsa.settings import AppSettings                             # noqa: E402
from owned_files import OwnedFiles                               # noqa: E402

CHECKS = 0
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  [PASS] {what}")
    else:
        FAILED.append(what)
        print(f"  [FAIL] {what}" + (f" -- {detail}" if detail else ""))


owned = OwnedFiles()

# =============================================================================
print("\n=== A. asking Windows, and what happens when it cannot be asked ===")
print("    reverting power.is_lit()'s strictness makes DIMMED or UNKNOWN look "
      "like a lit screen, which is how a camera gets cycled into a wedge")

try:
    for st, lit in ((power.ON, True), (power.OFF, False),
                    (power.DIMMED, False), (power.UNKNOWN, False)):
        power.force_for_test(st)
        check(power.is_lit() is lit,
              f"{power.NAMES[st]:>7} -> is_lit {lit}", str(power.is_lit()))

    power.force_for_test(None)
    check(power.state() in (power.ON, power.OFF, power.DIMMED, power.UNKNOWN),
          "the real state is one of the four known answers",
          power.NAMES.get(power.state(), "?"))
    check(power.watch() in (True, False), "watch() returns a bool and never raises")
    check(power.watch() is power.watch(), "watch() is idempotent",
          "it is called from the app, the doctor and this suite in one process")
finally:
    power.force_for_test(None)

check(power.UNKNOWN not in (power.ON,) and power.UNKNOWN != power.ON,
      "UNKNOWN is not ON", "a default that reads as 'lit' would cycle cameras blindly")

src_power = (ROOT / "src" / "lsa" / "display" / "power.py").read_text(encoding="utf-8")
check("import cv2" not in src_power and "PySide6" not in src_power,
      "the power module pulls in neither OpenCV nor Qt",
      "it must be importable by the doctor and by a test that opens no window")
check("_keep" in src_power and "proc" in src_power,
      "the callback is held against garbage collection",
      "Windows keeps a raw pointer; collecting it is a crash with nothing logged")


def _boom(_message):
    raise AttributeError("'CameraLab' object has no attribute 'log_view'")


# 2026-11-07, the regression this module itself caused: it was registered
# before the app's log widget existed, the success message raised inside
# `self._log`, and the exception came back out of watch() and stopped the app
# from starting. The docstring already promised that could not happen.
raised = None
try:
    power._say(_boom, "a message nobody can write down")
except Exception as exc:                                         # noqa: BLE001
    raised = exc
check(raised is None,
      "a LOGGER that raises cannot break the power watcher", repr(raised))
# The ONLY place the module may call the logger directly is inside `_say`,
# where it is wrapped. Anywhere else is a call that can escape, which is
# exactly what got out last time.
_say_body = src_power.split("def _say(", 1)[1].split("\ndef ", 1)[0]
_bare = [ln for ln in src_power.splitlines()
         if ln.strip().startswith("log(")]
check(len(_bare) == 1 and _bare[0].strip() in _say_body,
      "...and the only direct log() call in the module is the guarded one",
      f"{len(_bare)} found: {_bare}")

# EVERY WINDOWS CALL NEEDS AN EXPLICIT SIGNATURE (2026-11-07, shipped in
# v1.0.1 and inert there). ctypes defaults an un-annotated integer argument to
# a 32-bit C int, so a 64-bit HANDLE raises OverflowError as soon as its value
# happens to be large -- and how large depends on where Windows loaded the
# module. DefWindowProcW was given a signature because it broke in testing;
# CreateWindowExW was not, so registration worked from source and failed in
# the frozen build, where it fell back silently and the camera fix never
# fired. Setting one signature because one call broke is the half-fix this
# project keeps paying for (RULES E1).
_called = set(re.findall(r"(?:user32|kernel32)\.(\w+)\(", src_power))
_typed = set(re.findall(r"(?:user32|kernel32)\.(\w+)\.argtypes\s*=", src_power))
_untyped = sorted(_called - _typed)
check(not _untyped,
      "every Windows function the module CALLS has explicit argtypes",
      f"missing argtypes: {_untyped}")
check(len(_called) >= 4,
      "...and the scan actually found the calls to check",
      f"{sorted(_called)}")

if sys.platform.startswith("win"):
    check(power.watch() is True,
          "the watcher really does register on this machine",
          "a False here is the silent fallback: the fix would be inert")
    check(power.state() in (power.ON, power.OFF, power.DIMMED, power.UNKNOWN),
          "...and the state it reports is one of the four known answers",
          power.NAMES.get(power.state(), "?"))

# =============================================================================
print("\n=== B. who is blamed for a black frame ===")
print("    reverting `blame_for_a_black_frame` to a bare `return \"display\"` "
      "restores the 2026-11-07 bug")

from lsa.tools.camera_lab import CameraLab                       # noqa: E402

try:
    power.force_for_test(power.ON)
    check(CameraLab.blame_for_a_black_frame() == "camera",
          "Windows says the panel is ON -> the CAMERA is blamed",
          "this is 2026-11-07: a lit monitor and a lens with a blind on it")
    for st in (power.OFF, power.DIMMED, power.UNKNOWN):
        power.force_for_test(st)
        check(CameraLab.blame_for_a_black_frame() == "display",
              f"Windows says {power.NAMES[st]} -> the DISPLAY is blamed, camera untouched",
              "this is 2026-10-19, and cycling here cost three launches")
finally:
    power.force_for_test(None)

# =============================================================================
print("\n=== C. the guard actually consults it, and the cycling is REACHABLE ===")
print("    this is the whole bug: the cure existed and sat behind a guard that "
      "returned before reaching it")

lab_src = (ROOT / "src" / "lsa" / "tools" / "camera_lab.py").read_text(encoding="utf-8")

guard = re.search(
    r"if self\._blind and self\._display_dark and not self\._connected_device_is_virtual\(\):"
    r"(.*?)\n        if self\._blind:", lab_src, re.S)
check(guard is not None, "the blind-and-dark branch is still where it was")
body = guard.group(1) if guard else ""

check("blame_for_a_black_frame()" in body,
      "the branch ASKS who is to blame before reporting a sleeping display")
check(re.search(r'blame_for_a_black_frame\(\)\s*==\s*"display"[\s\S]{0,200}?'
                r"_report_display_asleep\(\)", body) is not None,
      "`_report_display_asleep()` is reached ONLY when the display is blamed",
      "unguarded, a blind camera is reported as a sleeping monitor again")
check(body.count("_report_display_asleep()") == 1,
      "...and there is exactly one way out of that branch into that report",
      f"found {body.count('_report_display_asleep()')}")

after = lab_src.split("if self._blind:", 1)[1][:4000] if "if self._blind:" in lab_src else ""
check("_next_camera_to_try()" in after,
      "the camera-cycling code still follows the guard and is now reachable",
      "it is the cure A11 produced in October and it was never being run")

# =============================================================================
print("\n=== D. the name that moved, and the net that catches it ===")
print("    reverting `_next_camera_to_try`'s bound lets a four-camera machine "
      "cycle for ever (RULES D10)")

# The MEASURED enumeration from his machine on 2026-11-07, after the reboot.
# "USB Video Device" is the Logitech with the blind on it; the Kiyo Pro that
# can actually see the monitor is wearing the C920's name.
# `is_virtual` is computed by the app's OWN classifier here, not asserted by
# hand: that is the thing being relied on, and a stand-in that paraphrases it
# would excuse a change to it (RULES C12).
AFTER_REBOOT = [
    CameraInfo(index=i, name=n, is_virtual=_is_virtual(n))
    for i, n in enumerate(("USB Video Device", "EOS Webcam Utility",
                           "HD Pro Webcam C920", "EOS Webcam Utility Pro"))
]
check(all(c.is_virtual == ("EOS" in c.name) for c in AFTER_REBOOT),
      "the two Canon devices are still recognised as virtual",
      "they open fine and deliver a disconnected-cable icon with a red X in it")

saved = AppSettings()
saved.device_name = "USB Video Device"       # what was in his settings.local.json
saved.device_index = 2
picked = saved.resolve_device(AFTER_REBOOT)
check(picked == 0,
      "the remembered NAME still wins, and after the reboot it picks the blind one",
      "recorded rather than 'fixed': a name is the right first guess, and the "
      "net below is what makes a wrong guess survivable")

# `_next_camera_to_try` reads exactly three attributes. The stand-in supplies
# those three and no others, and the check below proves the real class has
# them, so the stub cannot drift away from it (RULES C12).
for attr in ("_cameras", "_blind_devices", "cmb_device"):
    check(attr in lab_src, f"the real CameraLab still has `{attr}`",
          "the stand-in below mirrors it; a rename must break this, not pass quietly")


def fake_lab(current: int, cams=AFTER_REBOOT):
    lab = types.SimpleNamespace()
    lab._cameras = list(cams)
    lab._blind_devices = set()
    lab.cmb_device = types.SimpleNamespace(currentData=lambda: current)
    return lab


nxt = CameraLab._next_camera_to_try
lab = fake_lab(current=0)                    # connected to the blinded Logitech
first = nxt(lab)
check(first is not None and first[0] == 2,
      "from the blinded camera, the next one offered is the Kiyo Pro at index 2",
      str(first))
check(first is not None and "EOS" not in first[1],
      "...and never a virtual device")

tried, guard_count = [], 0
lab2 = fake_lab(current=0)
while guard_count < 12:
    guard_count += 1
    got = CameraLab._next_camera_to_try(lab2)
    if got is None:
        break
    tried.append(got[0])
    lab2.cmb_device = types.SimpleNamespace(currentData=lambda i=got[0]: i)
check(got is None, "cycling STOPS once every real camera has been offered",
      "RULES D10: anything repeated has a bound and a reason to stop")
check(len(tried) == len(set(tried)), "...and no camera is offered twice",
      str(tried))
check(len(tried) <= 2, "...so a four-camera machine makes at most two attempts",
      f"{tried} (the other two are virtual)")

# =============================================================================
print("\n=== E. the rule itself is now enforced, not 'judgement' ===")
print("    RULES E3: a rule with no test will be broken again, and this one "
      "has been broken twice")

rules_src = (ROOT / "tests" / "test_rules.py").read_text(encoding="utf-8")
a11 = re.search(r'"A11":\s*"([^"]*)"', rules_src)
check(a11 is not None, "A11 is still in the registry in test_rules.py section K")
check(a11 is not None and not a11.group(1).startswith("judgement"),
      "A11 is no longer registered as judgement", a11.group(1) if a11 else "")
check(a11 is not None and "test_camera_identity" in a11.group(1),
      "...and the registry names this file as its enforcement",
      a11.group(1) if a11 else "")

rules_md = (ROOT.parent / "RULES.md")
if rules_md.exists():
    md = rules_md.read_text(encoding="utf-8")
    check("## A11." in md, "A11 still has its heading in RULES.md",
          "section K requires the heading and the registry entry to agree")
    check("2026-11-07" in md.split("## A11.", 1)[1].split("## A12.")[0],
          "A11 carries the 2026-11-07 incident as well as the older one",
          "a rule without its evidence gets re-argued and then re-broken")

# =============================================================================
owned.check(lambda ok, what, detail="": check(ok, what, detail))
print("\n" + (f"ALL {CHECKS} CHECKS PASSED" if not FAILED else
              f"{len(FAILED)} OF {CHECKS} CHECKS FAILED"))
raise SystemExit(0 if not FAILED else 1)
