"""Spoken announcements: the properties that keep speech from costing shots.

No camera, no screen, no sound device. A tripwire on `sd.OutputStream` proves
nothing here opens one. Each section names the behaviour whose reversal turns
it red; the mutations were tried by hand (see the commit/report).
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import lsa.audio.mixer as mixer_mod                       # noqa: E402

opened = []


class Tripwire:
    def __init__(self, *a, **k):
        opened.append(1)
        raise RuntimeError("TRIPWIRE: a test tried to open a real output stream")


if mixer_mod.sd is not None:
    mixer_mod.sd.OutputStream = Tripwire

from silent_audio import SilentMixer                      # noqa: E402
from lsa.audio import voice as V                          # noqa: E402
from lsa.audio.voice import (PHRASES, SPEC, VoiceAnnouncer,   # noqa: E402
                             mode_events, mode_snapshot)

CHECKS = 0
FAILED = []


def check(ok, what, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(what)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class StopMixer(SilentMixer):
    """SilentMixer that also records stop() requests (it tracks no voices)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stopped = []

    def stop(self, name):
        self.stopped.append(name)
        return 1


from asset_env import res as _res                           # noqa: E402  (the asset table, not a join)
VOICE = _res("voice")


def make(voice_dir=VOICE, seed=1, mixer=None, enabled=None, mute=None):
    clock = Clock()
    mx = mixer or StopMixer()
    if not mx.available:
        mx.start()
    mutes = []
    logs = []

    def do_mute(s):
        mutes.append(s)

    a = VoiceAnnouncer(mx, voice_dir, mute=mute or do_mute, log=logs.append,
                       clock=clock, seed=seed, enabled=enabled, synchronous=True)
    return a, clock, mx, mutes, logs


REAL = VOICE.is_dir()

print("=== 1. the mute covers the WHOLE spoken line, measured from real files ===")
if REAL:
    import soundfile as sf
    a, clock, mx, mutes, logs = make()
    for key in ("v_ready", "v_calibrating", "v_camera_bad", "v_false_start"):
        mutes.clear()
        a._last_said.clear()
        a._speaking_until = 0.0
        a.say(key)
        started = a.poll()
        said = a.stats.said_lines[-1]
        voice, dur, mute_s = said[2], said[3], said[4]
        path = a.sets[voice][key]
        true_dur = float(sf.info(str(path)).duration)       # independent of the mixer
        check(started == key and len(mutes) == 1
              and abs(mutes[0] - (true_dur + a.tail)) < 0.06,
              f"{key} ({voice}): mute {mutes[0] if mutes else None:.3f}s "
              f"= file {true_dur:.3f}s + tail {a.tail:g}s",
              f"mixer duration {dur:.3f}s")
        check(mutes and mutes[0] >= true_dur,
              f"{key}: the mute is never shorter than the line")
        check(mx.played[-1][0] == f"voice/{voice}/{key}",
              f"{key}: played the file the mute was sized for")
    # Every shipped file: under the 2.0 s ceiling the brief set, and mute > dur.
    longest = 0.0
    n = 0
    for sname, files in a.sets.items():
        for key, path in files.items():
            d = float(sf.info(str(path)).duration)
            longest = max(longest, d)
            n += 1
    # DERIVED, NOT PINNED. `n == 63` was three voices times the 21 lines that
    # existed when it was written, so his 22nd recording failed a correct
    # change -- the same shape as the twelve-figure counts that broke the day
    # he added artwork. The property is that every voice carries every line,
    # and that none of them outlasts the ceiling the brief set.
    _want = len(SPEC) * len(a.sets)
    check(n == _want and longest < 2.0,
          f"all {n} shipped files are under 2.0 s "
          f"({len(a.sets)} voices x {len(SPEC)} lines)",
          f"longest {longest:.3f}s, expected {_want} files")
    # The mute must come from the loaded buffer, not a constant: the same
    # announcer given a 1.9 s buffer must mute for ~2.15 s, a 0.4 s one ~0.65 s.
    import numpy as np
    a2, c2, m2, mu2, l2 = make()
    for name, secs in (("voice/x/v_ready", 1.9), ("voice/y/v_ready", 0.4)):
        m2.register(name, np.zeros(int(m2.samplerate * secs), dtype=np.float32))
    for secs, name in ((1.9, "voice/x/v_ready"), (0.4, "voice/y/v_ready")):
        a2._pending.clear()
        a2._last_said.clear()
        a2._speaking_until = 0.0
        a2._pending.append(V._Pending("v_ready", "x", name, V.ROUTINE, c2.t, c2.t + 9))
        a2._loaded.add(name)
        a2.poll()
        check(abs(mu2[-1] - (secs + a2.tail)) < 0.01,
              f"mute follows the buffer's own length ({secs}s -> {mu2[-1]:.3f}s)")
else:
    check(False, "voice/ directory exists for the duration measurement")

print("=== 2. nothing is spoken during a string of shots ===")
a, clock, mx, mutes, logs = make()
clock.t = 1000.0
a.note_shot(1000.0)
a.say("v_mode_ball")
clock.t = 1001.0
check(a.poll() is None and not mx.played, "1 s after a pull: silent")
a.note_shot(1001.0)              # the string continues
clock.t = 1003.5
check(a.poll() is None and not mx.played,
      "2.5 s after the second pull (3 s gap not reached): silent")
clock.t = 1004.2
r = a.poll()
check(r == "v_mode_ball" and len(mx.played) == 1,
      "3.2 s after the last pull: it speaks", f"{r}")
check(a.in_string(1004.2) is False and a.in_string(1002.0) is True,
      "in_string() agrees with the gap")
# A string that never pauses: the line is dropped, never spoken late.
a, clock, mx, mutes, logs = make()
a.say("v_figure_down")
t = 1000.0
for i in range(10):
    t = 1000.0 + i
    clock.t = t
    a.note_shot(t)
    a.poll()
check(not mx.played and a.stats.dropped >= 1 and any("DROPPED v_figure_down" in l
                                                       and "string" in l for l in logs),
      "a string with no pause: nothing spoken, the drop is logged",
      f"played={len(mx.played)} dropped={a.stats.dropped}")
# An urgent line that is NOT a "stop" line obeys the same gate.
a, clock, mx, mutes, logs = make()
a.note_shot(1000.0)
a.say("v_calibrating")
clock.t = 1001.0
check(a.poll() is None and not mx.played,
      "an ordinary urgent line (calibrating) is gated too")

print("=== 2b. the stop-what-you-are-doing lines are NOT gated ===")
for key in ("v_hold_fire", "v_camera_bad", "v_camera_moved"):
    a, clock, mx, mutes, logs = make()
    a.note_shot(1000.0)
    clock.t = 1000.3                    # 0.3 s after a pull: deep in a string
    a.note_shot(1000.3)
    a.say(key)
    clock.t = 1000.5
    r = a.poll()
    check(a.in_string(1000.5) and r == key and len(mx.played) == 1 and len(mutes) == 1,
          f"{key} is spoken in the middle of a string, and still mutes the mic",
          f"in_string={a.in_string(1000.5)} spoke={r}")
check(V.GATE_EXEMPT == {"v_hold_fire", "v_camera_bad", "v_camera_moved"},
      "exactly three lines are exempt", str(sorted(V.GATE_EXEMPT)))
# An exempt line arriving while an ordinary one waits: the ordinary is preempted, the stop line goes.
a, clock, mx, mutes, logs = make()
a.say("v_mode_ball")
a.note_shot(1000.0)
a.say("v_hold_fire")
clock.t = 1000.5
check(a.poll() == "v_hold_fire", "hold fire goes mid-string")
a, clock, mx, mutes, logs = make()
a.note_shot(1000.0)
a.say("v_mode_ball")
clock.t = 1000.5
check(a.poll() is None and not mx.played,
      "...while an ordinary line in the same state is still held")

print("=== 2c. a pull the microphone heard opens a string too ===")
a, clock, mx, mutes, logs = make()
a.note_shot(1000.0)                       # camera-reported
a.note_shot(1002.0)                       # delivered later: microphone only
clock.t = 1003.0
a.say("v_mode_ball")
check(a.in_string(1003.0) and a.poll() is None,
      "a heard pull 1 s ago holds ordinary speech back")
a.note_shot(1001.0)                       # an OLDER pull arriving late
check(a._last_shot_at == 1002.0, "a late, older pull never moves the gate backwards")
clock.t = 1005.2
check(a.poll() == "v_mode_ball", "...and speech resumes 3 s after the last heard pull "
      "(the gate cannot block for ever)")
src_lab = (ROOT / "src/lsa/tools/camera_lab.py").read_text(encoding="utf-8")
i = src_lab.index("for c in clicks:")
check("self.listener.calibrated" in src_lab[i - 700:i]
      and "self.voice.note_shot(c.t)" in src_lab[i:i + 200],
      "the app feeds the listener's calibrated clicks into the gate")

print("=== 3. stale speech is dropped, not played late ===")
a, clock, mx, mutes, logs = make()
a.say("v_calibrating")                 # TTL 5 s
clock.t = 1005.5
check(a.poll() is None and not mx.played and a.stats.dropped == 1
      and any("DROPPED v_calibrating" in l and "stale" in l for l in logs),
      "'calibrating' asked for 5.5 s ago is dropped as stale",
      f"TTL {SPEC['v_calibrating'][1]:g}s")
a, clock, mx, mutes, logs = make()
state = {"calibrating": True}
a.say("v_calibrating", valid=lambda: state["calibrating"])
state["calibrating"] = False
clock.t = 1000.2
check(a.poll() is None and not mx.played
      and any("no longer true" in l for l in logs),
      "a line whose predicate went false is dropped inside its TTL")
a, clock, mx, mutes, logs = make()
a.say("v_calibrating", valid=lambda: True)
clock.t = 1000.2
check(a.poll() == "v_calibrating", "...and spoken when still true")
check(all(SPEC[k][1] <= 12.0 for k in SPEC),
      "every line has a deadline of 12 s or less",
      f"max {max(v[1] for v in SPEC.values())}")

print("=== 4. a problem preempts a routine line ===")
a, clock, mx, mutes, logs = make()
a.say("v_mode_ball")
a.say("v_hold_fire")
check([p.key for p in a._pending] == ["v_hold_fire"]
      and any("DROPPED v_mode_ball" in l and "preempted" in l for l in logs),
      "a pending routine line is discarded when an urgent one arrives")
a, clock, mx, mutes, logs = make()
a.say("v_new_round")
clock.t = 1000.2
a.poll()                                   # routine line starts
a.say("v_camera_bad")
clock.t = 1000.6                           # still inside the routine line
r = a.poll()
check(r == "v_camera_bad" and mx.stopped
      and mx.stopped[0].endswith("/v_new_round"),
      "an urgent line cuts a routine one off mid-sentence",
      f"stopped={mx.stopped}")
a, clock, mx, mutes, logs = make()
a.say("v_hold_fire")
clock.t = 1000.2
a.poll()
a.say("v_camera_moved")
clock.t = 1000.5
check(a.poll() is None and not mx.stopped,
      "an urgent line does NOT cut another urgent line; it waits")
clock.t = 1004.0
check(a.poll() == "v_camera_moved", "...and goes after it")

print("=== 5. the app works with the voice directory absent ===")
with tempfile.TemporaryDirectory() as td:
    gone = Path(td) / "no_such_voice_dir"
    a, clock, mx, mutes, logs = make(voice_dir=gone)
    ok = [a.say(k) for k in SPEC] + [a.poll(), a.status()]
    check(not any(ok[:-1]) and a.available is False and not mx.played and not mutes,
          "every say() is refused, nothing plays, nothing mutes, nothing raises")
    check(sum("announcements are off" in l for l in logs) == 1,
          "it says so once", f"{len(logs)} log line(s)")
    a, *_ = make(voice_dir=None)
    check(a.say("v_ready") is False, "voice_dir=None behaves the same")
    empty = Path(td) / "empty"
    (empty / "announcer").mkdir(parents=True)
    a, clock, mx, mutes, logs = make(voice_dir=empty)
    check(a.say("v_ready") is False and any("no recognised voice files" in l
                                             for l in logs),
          "a voice dir holding no files is the same as none")

print("=== 6. a random voice per phrase, from a dedicated seeded generator ===")
if REAL:
    def seq(seed):
        a, *_ = make(seed=seed)
        out = []
        for i in range(60):
            a._pending.clear()
            a._last_said.clear()
            a.say("v_ready")
            out.append(a._pending[0].voice)
        return out

    s1, s2, s3 = seq(7), seq(7), seq(8)
    check(s1 == s2, "same seed -> same sequence of voices (testable)")
    check(s1 != s3, "a different seed -> a different sequence")
    check(set(s1) == {"announcer", "mobster", "soldier"},
          "all three voices are used", f"{sorted(set(s1))}")
    counts = {v: s1.count(v) for v in set(s1)}
    check(min(counts.values()) >= 10, "none is starved over 60 draws", f"{counts}")
src = (ROOT / "src/lsa/audio/voice.py").read_text(encoding="utf-8")
code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
check(not re.search(r"(?<![.\w])random\.(random|choice|randint|shuffle|uniform|sample)\(", code)
      and "random.Random(" in code,
      "no module-level `random` calls; a dedicated random.Random instance")
# fallback when the chosen set lacks the file
with tempfile.TemporaryDirectory() as td:
    import shutil
    for s in ("announcer", "mobster"):
        (Path(td) / s).mkdir()
    if REAL:
        shutil.copy(VOICE / "announcer/v_ready.mp3", Path(td) / "announcer/v_ready.mp3")
        shutil.copy(VOICE / "announcer/v_hold_fire.mp3", Path(td) / "announcer/v_hold_fire.mp3")
        shutil.copy(VOICE / "mobster/v_hold_fire.mp3", Path(td) / "mobster/v_hold_fire.mp3")
        a, clock, mx, mutes, logs = make(voice_dir=Path(td), seed=3)
        got = set()
        for i in range(30):
            a._pending.clear()
            a._last_said.clear()
            a.say("v_ready")
            got.add(a._pending[0].voice)
        check(got == {"announcer"}, "a set missing the file is never chosen for it "
              "(falls back to the one that has it)", f"{got}")
        check(sum("has no v_ready" in l for l in logs) == 1,
              "the substitution is logged once, not on every line")
        a._pending.clear()
        check(a.say("v_level_up") is False and any("no set has v_level_up" in l
                                                    for l in logs),
              "a line no set has stays silent, and says so")

print("=== 7. every line spoken is logged with its voice and timestamp ===")
if REAL:
    a, clock, mx, mutes, logs = make()
    clock.t = 1234.5
    a.say("v_ready")
    clock.t = 1234.7
    a.poll()
    line = next((l for l in logs if "SAID v_ready" in l), "")
    check(bool(re.search(r'SAID v_ready "Ready\. Start shooting\." voice=(announcer|mobster|soldier) '
                         r't=1234\.70 len=\d+\.\d+s mic muted \d+\.\d+s waited=0\.20s', line)),
          "SAID line carries the phrase, voice set, time, length, mute and wait",
          line)

print("=== 8. disabled / unavailable output drops, never queues for later ===")
a, clock, mx, mutes, logs = make(enabled=lambda: False)
a.say("v_ready")
clock.t = 1000.1
check(a.poll() is None and not mx.played and not a._pending,
      "the sound checkbox off: dropped, not held")
dead = StopMixer(available=False)
a, clock, mx, mutes, logs = make(mixer=dead)
a.say("v_ready")
clock.t = 1000.1
check(a.poll() is None and not a._pending and not dead.played,
      "no audio output: dropped, not held")

print("=== 9. one voice at a time, and repeats are suppressed ===")
a, clock, mx, mutes, logs = make()
a.say("v_mode_ball")
a.say("v_mode_balls")
clock.t = 1000.1
a.poll()
clock.t = 1000.3
check(a.poll() is None and len(mx.played) == 1, "the second waits while the first speaks")
clock.t = 1010.0
check(a.poll() is None, "...and is dropped if its own deadline passed meanwhile")
a, clock, mx, mutes, logs = make()
a.say("v_mode_ball")
check(a.say("v_mode_ball") is True and len(a._pending) == 1,
      "asking twice while pending refreshes, never duplicates")
clock.t = 1000.1
a.poll()
clock.t = 1000.5
check(a.say("v_mode_ball") is False, "the same line inside its cooldown is refused")
a, clock, mx, mutes, logs = make()
a.say("v_camera_moved")
check(a.say("v_calibrating") is False, "'calibrating' is covered by 'camera moved'")

print("=== 10. the wording and the files agree with the assignment ===")
md = (ROOT / "requirements" / "phrases_to_generate.md").read_text(encoding="utf-8")
rows = {}
for k_, v_ in re.findall(r"\|\s*`(v_\w+)`\s*\|\s*([^|]+?)\s*\|", md):
    rows.setdefault(k_, v_)       # first table wins: later ones list the old wording
check(set(rows) == set(SPEC), f"SPEC covers exactly the {len(rows)} lines in requirements/phrases_to_generate.md",
      f"missing {set(rows) ^ set(SPEC)}")
check(all(PHRASES.get(k) == rows[k] for k in rows),
      "the logged wording equals the assignment's wording",
      str([k for k in rows if PHRASES.get(k) != rows[k]]))
if REAL:
    for sname in ("announcer", "mobster", "soldier"):
        have = {p.stem for p in (VOICE / sname).glob("*.mp3")}
        check(have == set(SPEC), f"voice/{sname} holds exactly the 21 named files",
              str(have ^ set(SPEC)))

print("=== 11. mode events come from state the modes already have ===")
class M:   # noqa: E301
    pass


def snap(**k):
    o = M()
    for a_, b_ in k.items():
        setattr(o, a_, b_)
    return mode_snapshot(o)


check(mode_events(snap(dropped=0), snap(dropped=1)) == ["v_figure_down"], "figure down")
check(mode_events(snap(false_starts=0), snap(false_starts=1)) == ["v_false_start"], "false start")
check(mode_events(snap(level=1), snap(level=2)) == ["v_level_up"], "ball level up")
b0 = M(); b0.level = 1
b1 = M(); b1.level = 2
check(mode_events(snap(balls=[b0, b0]), snap(balls=[b0, b1])) == ["v_level_up"],
      "bouncing-balls level up (sum over balls)")
check(mode_events(snap(cleared_at=None), snap(cleared_at=3.0)) == ["v_round_done"], "round done")
check(mode_events(snap(cleared_at=3.0), snap(cleared_at=3.0)) == [], "...once, not again")
check(mode_events(snap(), snap()) == [] and mode_events({}, {}) == [],
      "nothing happened, nothing said")
# The real modes must expose those attributes, or the above is testing a fiction.
from lsa import modes as arcade                           # noqa: E402
need = {"popup": ("dropped", "false_starts"), "image": ("dropped",),
        "advancing": ("dropped",), "reaction": ("false_starts",),
        "ball": ("level",), "balls": ("balls",),
        "circles": ("cleared_at",)}
bad = []
for nm, attrs in need.items():
    cls = arcade.by_name(nm)
    inst = cls() if cls is not None else None
    for at in attrs:
        if inst is None or not hasattr(inst, at):
            bad.append(f"{nm}.{at}")
check(not bad, "every real mode exposes the counters the events read", str(bad))
# DERIVED FROM THE REGISTRY, not a list of names. The hardcoded set broke the
# moment he removed shoot/no-shoot on 2026-10-18 -- a correct change failing a
# check that was pinned to the state of the world rather than to the property.
# The property is: every line names a real drill, and every line has audio.
import lsa.modes as _modes_v
# The MENU is a screen he lands on and it announces itself, so it counts as a
# name that may be spoken -- it is simply not a drill.
_registered = {c.name for c in _modes_v.available()}
_unknown = sorted(set(V.MODE_LINES) - _registered)
check(not _unknown and all(v in SPEC for v in V.MODE_LINES.values()),
      "every spoken mode name belongs to a drill that exists, and has audio",
      f"unknown: {_unknown}")
# And say which drills arrive in silence, rather than asserting a count: it is
# a missing RECORDING, which only he can make, not a code fault.
_silent = sorted(_registered - set(V.MODE_LINES) - {"menu"})
check(True, f"drills with no spoken name: {_silent or 'none'}",
      "reported, not failed: a recording is his to make")

print("=== 12. nothing here opened a sound device ===")
check(not opened, "no OutputStream was constructed", f"{len(opened)}")
check(mixer_mod.Mixer("x") is not None and "voice/x/v" not in
      SilentMixer.names(SilentMixer()), "voice samples are hidden from the gunshot picker")
m = SilentMixer()
import numpy as _np                                      # noqa: E402
m.register("voice/a/v_ready", _np.zeros(10, dtype=_np.float32))
m.register("boom", _np.zeros(10, dtype=_np.float32))
check(m.names() == ["boom"], "Mixer.names() lists gunshots only",
      str(m.names()))

if FAILED:
    print(f"\nFAILED {len(FAILED)} of {CHECKS}: {FAILED}")
    sys.exit(1)
print(f"\nALL {CHECKS} CHECKS PASSED")
