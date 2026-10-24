"""A test that constructs a CameraLab can never reach a real output stream.

Asserts on the ABSENCE OF A STREAM OBJECT, not on silence: `sounddevice`'s
OutputStream is replaced by a tripwire that counts constructions, a real
CameraLab is built the way the cold-start harness builds it, and the count must
be zero. Shown red with the install removed (see the commit message).
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from owned_files import OwnedFiles, isolate_outputs      # noqa: E402
from asset_env import res as _asset_res                   # noqa: E402  (the asset table, not a join)

owned = OwnedFiles()
isolate_outputs("silent_audio")      # BEFORE camera_lab is imported

import sounddevice                                        # noqa: E402
import lsa.audio.mixer as mixer_mod                       # noqa: E402

opened = []


class Tripwire:
    def __init__(self, *a, **k):
        opened.append((a, k))
        raise RuntimeError("TRIPWIRE: a test tried to open a real output stream")


mixer_mod.sd.OutputStream = Tripwire          # the one place a stream is made

# And the other direction: no INPUT stream either. The cold-start harness used to
# open `Microphone (HD Pro Webcam C920)` on every launch, so its audio behaviour
# depended on his room. `ClickListener.start` makes its stream in exactly one
# place, `sd.InputStream`; a construction there is the thing being counted.
import lsa.audio.listener as listener_mod                 # noqa: E402

opened_in = []


class TripwireIn:
    def __init__(self, *a, **k):
        opened_in.append((a, k))
        raise RuntimeError("TRIPWIRE: a test tried to open a real input stream")


listener_mod.sd.InputStream = TripwireIn

from PySide6.QtWidgets import QApplication                # noqa: E402

CHECKS = 0
FAILED = []


def check(ok, what, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(what)


app = QApplication.instance() or QApplication(sys.argv)

import lsa.settings as _settings                          # noqa: E402
import lsa.tools.camera_lab as cl                         # noqa: E402

_orig_load = _settings.AppSettings.load


@staticmethod
def _no_connect(path):
    st = _orig_load(path)
    st.auto_connect = False
    return st


_settings.AppSettings.load = _no_connect
cl.CameraLab.hard_exit_on_close = False
try:
    lab = cl.CameraLab()
finally:
    _settings.AppSettings.load = _orig_load
# Neutered like the cold-start harness does, AND fingerprinted below: the first
# run of this very test closed the lab, whose shutdown saves settings, and
# left auto_connect false in his settings.local.json (RULES C11, a fourth time).
lab._save_settings = lambda *a, **k: None
lab._schedule_save = lambda *a, **k: None

print("=== a CameraLab built by a test opens no sound device ===")
check(not opened, "no output stream was constructed",
      f"{len(opened)} OutputStream construction(s)")
check(lab.mixer._stream is None, "the mixer holds no stream object",
      f"{lab.mixer._stream!r}")
check(type(lab.mixer).__name__ == "SilentMixer", "it is the silent mixer",
      type(lab.mixer).__name__)
played = getattr(lab.mixer, "played", None)
check(lab.mixer.available and "synth shot" in lab.mixer.names()
      and lab.mixer.loud_duration("synth shot") > 0
      and lab.mixer.ensure_alive() == "" and lab.mixer.latency_ms == 0.0,
      "it answers the API the app uses",
      "available, registered samples, durations, ensure_alive")
asked = lab.mixer.play("synth shot", 0.5)
check(asked and getattr(lab.mixer, "played", None) == [("synth shot", 0.5)]
      and not lab.mixer._voices, "a play request is recorded, not mixed",
      f"{getattr(lab.mixer, 'played', None)}")
lab.mixer.close()
lab.mixer.start()
check(not opened and lab.mixer._stream is None, "close and restart still open nothing")

print("=== ...and opens no INPUT device either (the microphone) ===")
check(type(lab.listener).__name__ == "SilentListener", "the lab holds the silent listener",
      type(lab.listener).__name__)
# Do what connecting the camera does: start listening. The effect is asserted, not
# the attempt (RULES C1): the listener must report running, through a sentinel,
# with the tripwire never having been touched.
lab.chk_listen.blockSignals(True)
lab.chk_listen.setChecked(True)
lab.chk_listen.blockSignals(False)
lab._start_listening()
check(not opened_in, "no input stream was constructed",
      f"{len(opened_in)} InputStream construction(s)")
check(lab.listener.running and type(lab.listener._stream).__name__ == "_NoStream",
      "listening started (the app's microphone paths run) on a sentinel, not a stream",
      f"running={lab.listener.running} stream={lab.listener._stream!r}")
check(lab.listener.error == "" and lab.listener.take_clicks() == []
      and lab.listener.audio_at(0.0)[1] == 0,
      "it hears nothing: no click, no audio block, whatever the room sounds like")
lab.listener.stop()
lab.listener.start()
lab.listener.stop()
check(not opened_in and not lab.listener.running, "stop and restart still open nothing")
# The other outcome of the real start(): it failed, so every "audio
# unavailable" branch in the app is reachable in simulation.
from silent_audio import silent_mixer                     # noqa: E402
dead = silent_mixer(sounds_dir=None, available=False)
check(dead.available is False and dead.start() is False and dead.error
      and dead._stream is None and dead.play("synth shot") is False
      and not opened,
      "available=False reproduces a failed start(), still opening nothing",
      f"available={dead.available} error={dead.error!r}")
print("=== spoken announcements ride the same silent mixer ===")
import time as _time                                      # noqa: E402
from lsa.audio.voice import VoiceAnnouncer                # noqa: E402
check(isinstance(lab.voice, VoiceAnnouncer) and lab.voice.mixer is lab.mixer,
      "the announcer is built over the lab's own (silent) mixer",
      type(lab.voice.mixer).__name__)
check(lab.voice.available and "v_welcome" in [p.key for p in lab.voice._pending],
      "the welcome was asked for at startup")
deadline = _time.time() + 6.0
n_muted = len(lab.listener._mutes)
while _time.time() < deadline and not lab.voice.stats.said:
    lab.voice.poll()
    _time.sleep(0.05)
voiced = [n for n, _g in lab.mixer.played if n.startswith("voice/")]
check(voiced and lab.mixer._stream is None and not opened,
      "a line was 'played' into the recording mixer, and no stream was opened",
      f"{voiced}")
check(len(lab.listener._mutes) > n_muted,
      "speaking muted the microphone", f"{len(lab.listener._mutes) - n_muted} mute(s)")
lab.voice._speaking_until = 0.0
lab._announce_mode()
lab._signal_note("HOLD FIRE - measuring the noise floor", "warn")
keys = [p.key for p in lab.voice._pending]
check("v_hold_fire" in keys, "HOLD FIRE on the signal meter is spoken", f"{keys}")
check("v_mode_static" not in keys,
      "...and the routine line already waiting is preempted by it",
      f"{keys}")
lab.voice._pending.clear()
lab.voice._last_said.clear()
lab._announce_mode()
check("v_mode_static" in [p.key for p in lab.voice._pending],
      "choosing a mode queues its spoken name")
# THE PROPERTY, not a call site: every path into `voice.note_shot` must be
# gated by something that means "a real trigger pull" (RULES B9). A check that
# pinned one call site went stale the moment the microphone became a second
# caller -- the "names the callers its author thought of" mistake. So: find
# EVERY call by AST, and require each to sit under a guard that is either the
# camera-side predicate (`not self._app_is_drawing`) or the microphone-side one
# (a CALIBRATED listener, whose threshold is learned only from real pulls, and
# whose clicks `_poll_audio` discards while the app shoots at itself).
import ast                                                # noqa: E402

_lab_src = (Path(__file__).resolve().parents[1]
            / "src/lsa/tools/camera_lab.py").read_text(encoding="utf-8")
_tree = ast.parse(_lab_src)
_parents = {}
for _n in ast.walk(_tree):
    for _c in ast.iter_child_nodes(_n):
        _parents[_c] = _n
_sites = []
for _n in ast.walk(_tree):
    if (isinstance(_n, ast.Call) and isinstance(_n.func, ast.Attribute)
            and _n.func.attr == "note_shot"):
        guards, fn, cur = [], None, _n
        while cur in _parents:
            cur = _parents[cur]
            if isinstance(cur, ast.If):
                guards.append(ast.get_source_segment(_lab_src, cur.test) or "")
            if isinstance(cur, ast.FunctionDef):
                fn = cur
                break
        _sites.append((_n.lineno, fn.name if fn else "?", guards, fn))


def _gated(site):
    _ln, _fname, guards, fn = site
    camera_side = any("not self._app_is_drawing" in g for g in guards)
    mic_side = any("self.listener.calibrated" in g for g in guards)
    if camera_side:
        return True
    if mic_side and fn is not None:
        # ...and the function drops the clicks while the app shoots at itself,
        # BEFORE the call (so its own dots / probe / mode test never count).
        body = ast.get_source_segment(_lab_src, fn) or ""
        pre = body.split("self.voice.note_shot(")[0]
        return all(f in pre for f in ("_selftest_active", "_modetest_active",
                                      "_probe_active", "take_clicks()"))
    return False


check(len(_sites) >= 2, "there are at least two paths into note_shot (camera, microphone)",
      f"{[(a_[0], a_[1]) for a_ in _sites]}")
check(all(_gated(a_) for a_ in _sites),
      "EVERY path into note_shot is gated by a real-trigger-pull predicate",
      f"ungated: {[(a_[0], a_[1]) for a_ in _sites if not _gated(a_)]}")
# The microphone itself: nothing is heard while the app is making a noise, so
# its own speech/gunshot (and tail) cannot arrive as a "pull".
import numpy as _np                                       # noqa: E402
from silent_audio import SilentListener                   # noqa: E402
_ls = SilentListener(min_level=0.01)
_quiet = _np.full((256, 1), 0.001, dtype=_np.float32)
_loud = _np.full((256, 1), 0.6, dtype=_np.float32)
for _ in range(40):
    _ls._on_audio(_quiet, 256, None, None)
_ls.mute_for(1.0)
_ls._on_audio(_loud, 256, None, None)
check(_ls.take_clicks() == [], "a loud block while the app is muted is not a click")
_ls._muted_until = 0.0
_ls._last_click_t = 0.0
_ls._on_audio(_loud, 256, None, None)
check(len(_ls.take_clicks()) == 1, "...the same block unmuted is (the control)")

# And with NO voice files at all the lab still builds and runs.
cl.CameraLab.voice_dir = Path(__file__).resolve().parent / "_no_such_voice_dir"
_settings.AppSettings.load = _no_connect
try:
    lab2 = cl.CameraLab()
finally:
    _settings.AppSettings.load = _orig_load
    del cl.CameraLab.voice_dir
    cl.CameraLab.voice_dir = _asset_res("voice")
lab2._save_settings = lambda *a, **k: None
lab2._schedule_save = lambda *a, **k: None
lab2._signal_note("HOLD FIRE - x", "warn")
lab2._announce_mode()
lab2.voice.poll()
check(lab2.voice.available is False and not lab2.voice._pending
      and not [n for n, _g in lab2.mixer.played if n.startswith("voice/")]
      and not opened,
      "with the voice directory absent the lab builds, runs and says nothing")
try:
    lab2.close()
except Exception:
    pass

try:
    lab.close()
except Exception:
    pass
owned.check(check)

if FAILED:
    print(f"\nFAILED {len(FAILED)} of {CHECKS}: {FAILED}")
    sys.exit(1)
print(f"\nALL {CHECKS} CHECKS PASSED")
