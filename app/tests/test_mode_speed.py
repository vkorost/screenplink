"""The shooter's per-drill speed: Page Up / Page Down, remembered per drill.

Asked for 2026-11-07: "I'd like to control the speed of moving balls ... we
will need to record the speed in the settings so that next time it would pick
up whatever is configured previously", then "the same thing for the moving
targets, but lets record the speeds separately for each mode".

WHY THIS FILE MEASURES MOTION INSTEAD OF READING THE FLAG. `supports_speed` is
a claim; the thing that matters is whether the targets actually move
differently. A flag without the multiplication would make the key report a
change that never happened, which is RULES C1 -- a step that can quietly do
nothing is worse than one that fails. So section C holds a MEASUREMENT per
drill, and section C1 fails the build if a drill ever declares the flag
without one, so a drill that gains a speed next year cannot be waved through.

Each section says what goes red when the behaviour it guards is reverted.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from owned_files import isolate_outputs                          # noqa: E402
isolate_outputs("mode_speed")

from PySide6.QtCore import QRect, Qt                             # noqa: E402
from PySide6.QtWidgets import QApplication                       # noqa: E402

from lsa import modes                                            # noqa: E402
from lsa.display.scene import RenderContext                      # noqa: E402
from lsa.display.session import SessionStats                     # noqa: E402
from lsa.modes.base import LASER_SPEED_CEILING, Mode, ModeHost   # noqa: E402
from lsa.settings import (DEFAULT_SPEED, SPEED_MAX, SPEED_MIN,   # noqa: E402
                          AppSettings, normalise_mode_speed, normalise_speed)
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
_app = QApplication.instance() or QApplication(sys.argv)


def ctx(w: int = 3840, h: int = 2160, now: float = 1.0) -> RenderContext:
    return RenderContext(now=now, dt=1 / 60.0, rect=QRect(0, 0, w, h),
                         stats=SessionStats())


# =============================================================================
print("\n=== A. the setting: what survives a hand-edited or corrupt file ===")
print("    reverting `normalise_speed` turns a junk value into a crash or a "
      "drill that never moves")

check(normalise_speed(1.4) == 1.4, "a sane multiplier is kept")
check(normalise_speed(99) == SPEED_MAX, "an absurd fast value is CLAMPED, not refused",
      str(normalise_speed(99)))
check(normalise_speed(0) == SPEED_MIN, "zero is clamped to the floor, never 0",
      "a drill at speed 0 has targets that never move and reads as a frozen app")
check(normalise_speed(-5) == SPEED_MIN, "a negative speed cannot reverse a target")
check(normalise_speed("fast") == DEFAULT_SPEED, "text falls back to the default")
check(normalise_speed(None) == DEFAULT_SPEED, "a missing value falls back to the default")
check(normalise_speed(True) == DEFAULT_SPEED, "a bool is not a speed")
check(normalise_speed(float("nan")) == DEFAULT_SPEED, "NaN falls back to the default")
check(normalise_speed(float("inf")) == DEFAULT_SPEED, "infinity falls back to the default")

check(normalise_mode_speed("nonsense") == {}, "a non-mapping becomes an empty mapping")
check(normalise_mode_speed({"balls": 9.0})["balls"] == SPEED_MAX,
      "an out-of-range entry is clamped rather than dropping the whole file")
check("balls" not in normalise_mode_speed({"balls": "x"}),
      "an entry that is not a number is dropped, not defaulted to 1.0",
      "a drill never changed must stay ABSENT, so a later default change reaches it")
check(normalise_mode_speed({"": 1.5}) == {}, "an empty drill name is dropped")

s = AppSettings()
check(s.mode_speed == {}, "a FRESH INSTALL remembers no speeds at all",
      "a stranger inherits his drills as designed, not as he tuned them")
check(s.speed_for("balls") == DEFAULT_SPEED, "a drill with no entry runs at the default")
check(s.set_speed_for("balls", 1.7) == 1.7 and s.speed_for("balls") == 1.7,
      "a speed set for one drill reads back")
check(s.speed_for("ball") == DEFAULT_SPEED,
      "...and setting ONE drill's speed leaves the others alone",
      "he asked for the speeds to be recorded SEPARATELY for each mode")
check(s.set_speed_for("ball", 50.0) == SPEED_MAX,
      "a speed set past the ceiling is stored clamped, not stored raw")

# A real round trip through the file, since that is where a value actually
# has to survive (and `save` is atomic, RULES C16).
tmp = Path(os.environ.get("TEMP", ".")) / "lsa_mode_speed_probe.json"
try:
    s2 = AppSettings()
    s2.set_speed_for("advancing", 0.7)
    s2.save(tmp)
    back = AppSettings.load(tmp)
    check(back.speed_for("advancing") == 0.7, "a speed survives save + load",
          str(back.mode_speed))
    check(back.speed_for("balls") == DEFAULT_SPEED,
          "...and a drill that was never changed is still absent after the round trip")
finally:
    try:
        tmp.unlink()
    except OSError:
        pass

# =============================================================================
print("\n=== B. the ladder the two keys walk ===")
print("    reverting `next_speed` strands the shooter at one end or skips 100%")

from lsa.tools.camera_lab import CameraLab                       # noqa: E402

LADDER = CameraLab.SPEED_LADDER
check(list(LADDER) == sorted(LADDER) and len(set(LADDER)) == len(LADDER),
      "the rungs are strictly increasing with no duplicates")
check(100 in LADDER, "100% -- the drill AS DESIGNED -- is on the ladder",
      "without it there is no way back to the tuned speed")
check(LADDER[0] == int(round(SPEED_MIN * 100)),
      "the bottom rung is exactly settings.SPEED_MIN", f"{LADDER[0]} vs {SPEED_MIN}")
check(LADDER[-1] == int(round(SPEED_MAX * 100)),
      "the top rung is exactly settings.SPEED_MAX", f"{LADDER[-1]} vs {SPEED_MAX}")

walk, seen = LADDER[0], [LADDER[0]]
for _ in range(len(LADDER) + 4):
    walk = CameraLab.next_speed(walk, +1)
    seen.append(walk)
check(seen[-1] == LADDER[-1] and max(seen) == LADDER[-1],
      "walking UP reaches the top rung and stops there", str(seen[-3:]))
walk, seen = LADDER[-1], [LADDER[-1]]
for _ in range(len(LADDER) + 4):
    walk = CameraLab.next_speed(walk, -1)
    seen.append(walk)
check(seen[-1] == LADDER[0] and min(seen) == LADDER[0],
      "walking DOWN reaches the bottom rung and stops there", str(seen[-3:]))
check(set(LADDER) <= set(seen) | {LADDER[-1]},
      "every rung is reachable by walking")

# =============================================================================
print("\n=== C. the MEASURED effect, per drill ===")
print("    reverting a drill's `* self.speed_scale` leaves its motion "
      "unchanged and goes red here")


def _balls_speed(mode, c):
    mode.start(c)
    return mode._speed_px(mode.balls[0], c)


def _ball_speed(mode, c):
    mode.start(c)
    return mode._speed_px(c)


def _drift(mode, c):
    return abs(mode._drift_px(mode.playfield(c), 1.0, 1.0))


def _move_seconds(mode, c):
    return mode._moving_s(4.0)


def _reaction_speed(mode, c):
    """Pixels a second the pop-up target will travel at this multiplier.

    2026-11-09. This drill is the one whose speed may be ZERO, which is its
    default and the whole of its old behaviour, so the pair this section
    compares is 1.0x against 2.0x exactly as for the others -- zero is covered
    on its own in section G, where standing still has to really mean a target
    that does not move a pixel.
    """
    return mode._speed_px(c)


# name -> [(what is measured, how, which way it moves when the speed goes UP)]
# A DRILL THAT DECLARES A SPEED MUST APPEAR HERE. Section C1 enforces it, and
# it did: Reaction declared one on 2026-11-09 and this file went red in the
# same minute, before the drill had ever been shot at.
MEASURED: dict[str, list] = {
    "balls": [("ball speed", _balls_speed, +1)],
    "ball": [("ball speed", _ball_speed, +1)],
    "reaction": [("pop-up target speed", _reaction_speed, +1)],
    "advancing": [("sideways drift", _drift, +1),
                  ("seconds to approach", _move_seconds, -1)],
}

declared = sorted(m.name for m in
                  (cls() for cls in [type(x) for x in _instances()] if False)) \
    if False else None   # placeholder replaced below

# Build one instance of every registered drill, by the registry (never typed).
instances = {}
for cls, _key in modes.registered():
    try:
        obj = cls()
    except Exception as exc:                                     # noqa: BLE001
        print(f"  [note] {cls} could not be built: {exc}")
        continue
    instances[obj.name] = obj

check(bool(instances), "the drill registry could be read and built",
      f"{len(instances)} drills")

speedy = sorted(n for n, m in instances.items() if m.supports_speed)
check(speedy == sorted(MEASURED),
      "C1. every drill declaring a speed has a MEASUREMENT in this file",
      f"declared={speedy} measured={sorted(MEASURED)}")

for name in sorted(set(speedy) & set(MEASURED)):
    for label, fn, direction in MEASURED[name]:
        c = ctx()
        slow, fast = instances[name], type(instances[name])()
        slow.speed_scale = 1.0
        fast.speed_scale = 2.0
        try:
            a = fn(slow, c)
            b = fn(fast, ctx())
        except Exception as exc:                                 # noqa: BLE001
            check(False, f"{name}: {label} could be measured", repr(exc))
            continue
        if direction > 0:
            ok = b > a * 1.5
            why = f"1.0x -> {a:.3f}, 2.0x -> {b:.3f} (expected to GROW)"
        else:
            ok = b < a * 0.75
            why = f"1.0x -> {a:.3f}, 2.0x -> {b:.3f} (expected to SHRINK)"
        check(ok, f"{name}: doubling the speed changes the {label}", why)

# The measured ceiling still binds, at the top of the ladder.
cb = instances.get("balls")
if cb is not None:
    cb.speed_scale = SPEED_MAX
    c = ctx()
    cb.start(c)
    top = cb._speed_px(cb.balls[0], c) / cb.playfield(c).width()
    check(top <= LASER_SPEED_CEILING + 1e-9,
          "even at the FASTEST rung a ball stays under the measured laser ceiling",
          f"{top:.3f} screen widths/s vs ceiling {LASER_SPEED_CEILING}")
    check(cb.MAX_SPEED == LASER_SPEED_CEILING,
          "...and that ceiling has ONE definition, in base.py")

# =============================================================================
print("\n=== D. the host: where a saved speed is applied ===")
print("    reverting the `_apply_saved_speed()` call in `_start_all` means a "
      "saved speed is forgotten whenever a round restarts")

host = ModeHost()
host.speed_provider = lambda name: 1.8 if name == "balls" else 1.0
c = ctx()
host.set_mode(type(instances["balls"])(), c, players=1)
check(abs(host.mode.speed_scale - 1.8) < 1e-9,
      "picking a drill applies the speed its settings remember",
      str(host.mode.speed_scale))
host.restart(c)
check(abs(host.mode.speed_scale - 1.8) < 1e-9,
      "...and a NEW ROUND (N) keeps it, rather than resetting to 1.0",
      str(host.mode.speed_scale))

check(host.set_speed_scale(2.5) == 2.5 and abs(host.mode.speed_scale - 2.5) < 1e-9,
      "a live change reaches the running drill")
check(host.set_speed_scale(99) == SPEED_MAX,
      "a live change past the ceiling is clamped by the drill, not accepted")

host2 = ModeHost()
host2.speed_provider = lambda name: 1.0
quiet = next((m for m in instances.values() if not m.supports_speed), None)
if quiet is not None:
    host2.set_mode(type(quiet)(), ctx(), players=1)
    check(host2.set_speed_scale(2.0) is None,
          "a drill with NO speed reports None instead of silently doing nothing",
          "that is what lets the app say so on screen (RULES C1)")

# A broken provider must not stop a drill from starting.
host3 = ModeHost()


def _boom(_name):
    raise RuntimeError("settings exploded")


host3.speed_provider = _boom
try:
    host3.set_mode(type(instances["balls"])(), ctx(), players=1)
    started = host3.mode is not None
except Exception as exc:                                         # noqa: BLE001
    started, _ = False, exc
check(started and host3.mode.speed_scale == 1.0,
      "a provider that raises leaves the drill at 1.0 and still STARTS it",
      "the drill is the point; the speed is a preference")

# =============================================================================
print("\n=== E. the keys are discoverable, and the instrument can press them ===")
print("    reverting the Keys-screen rows leaves a key nobody can find")

from lsa.display.layers import KeyHelp                           # noqa: E402
from lsa.modes import shortcuts                                  # noqa: E402
from lsa.introspect.gui import resolve_key                       # noqa: E402

legend = " ".join(KeyHelp().lines).lower()
check("pgup" in legend and "pgdn" in legend,
      "the bottom legend lists Page Up / Page Down")
check("speed" in legend, "...and says what they do")

also = " ".join(f"{e.glyph} {e.title}" for e in shortcuts.ALSO).lower()
check("pgup" in also and "pgdn" in also,
      "the K (Keyboard Shortcuts) screen lists them too")
entry = next((e for e in shortcuts.ALSO if "PgUp" in e.glyph), None)
check(entry is not None
      and int(Qt.Key_PageUp) in entry.keys and int(Qt.Key_PageDown) in entry.keys,
      "...and the row carries the real Qt key codes, not just a glyph",
      "a glyph with the wrong codes is a legend that lies")

check(resolve_key("PageUp") == int(Qt.Key_PageUp)
      and resolve_key("PageDown") == int(Qt.Key_PageDown),
      "the introspection server can press both keys",
      "so a speed change can be driven and read back without his hands")

check(CameraLab.SPEED_UP_KEYS == (int(Qt.Key_PageUp),)
      or Qt.Key_PageUp in CameraLab.SPEED_UP_KEYS,
      "the app binds Page Up to faster")
check(Qt.Key_PageDown in CameraLab.SPEED_DOWN_KEYS,
      "the app binds Page Down to slower")
check(not (set(CameraLab.SPEED_UP_KEYS) | set(CameraLab.SPEED_DOWN_KEYS))
      & (set(CameraLab.VOLUME_UP_KEYS) | set(CameraLab.VOLUME_DOWN_KEYS)),
      "the speed keys do not collide with the volume keys")
check(Qt.Key_BracketLeft not in set(CameraLab.SPEED_UP_KEYS) | set(CameraLab.SPEED_DOWN_KEYS)
      and Qt.Key_BracketRight not in set(CameraLab.SPEED_UP_KEYS) | set(CameraLab.SPEED_DOWN_KEYS),
      "...and they left the camera pane's [ ] alone",
      "he chose Page Up / Page Down rather than take a key that was in use")

# =============================================================================
print("\n=== F. a target ALREADY MOVING changes speed, without being replaced ===")
print("    reverting any `respeed` leaves the thing on screen at its old speed "
      "until it is killed, which is what he reported minutes after it shipped")

import math                                                      # noqa: E402

# WHY THIS SECTION EXISTS. Section C measured `_speed_px`, the function that
# decides how fast a target is launched. It passed while the feature was
# broken: pressing the key changed nothing you could see until the ball you
# were looking at was replaced. Measuring the easy thing rather than the
# property that matters is RULES C6, and it cost him a round to find.

def live_host(name, scale_from=1.0):
    h = ModeHost()
    h.speed_provider = lambda _n: scale_from
    h.set_mode(type(instances[name])(), ctx(now=10.0), players=1)
    h.advance(ctx(now=10.02))
    return h


for name, read in (("balls", lambda m: math.hypot(m.balls[0].vx, m.balls[0].vy)),
                   ("ball", lambda m: math.hypot(m.vx, m.vy))):
    h = live_host(name)
    mode = h.mode
    target = mode.balls[0] if name == "balls" else mode
    before = read(mode)
    h.set_speed_scale(2.0)
    h.advance(ctx(now=10.04))
    after = read(mode)
    still_same = (mode.balls[0] is target) if name == "balls" else True
    check(before > 0 and after > before * 1.8,
          f"{name}: a ball ALREADY in flight speeds up on the next frame",
          f"{before:.1f} -> {after:.1f} px/s")
    check(still_same,
          f"{name}: ...and it is the SAME ball, not a replacement",
          "he saw the old speed persist until the ball was killed")

    h2 = live_host(name)
    b2 = h2.mode.balls[0] if name == "balls" else h2.mode
    fast = read(h2.mode)
    h2.set_speed_scale(0.5)
    h2.advance(ctx(now=10.04))
    check(read(h2.mode) < fast * 0.7,
          f"{name}: Page Down slows a ball in flight too",
          f"{fast:.1f} -> {read(h2.mode):.1f} px/s")

# The figure drill must not JUMP when it is re-speeded: the whole difficulty
# of this one is that its position and its apparent distance are both
# functions of how far through a timed phase it is.
adv = type(instances["advancing"])()
adv.speed_scale = 1.0
c = ctx()
adv.start(c)
actor = adv._actors[0]
actor.phase, actor.fig = "live", (actor.fig or getattr(adv, "_figure_for_test", None))
if actor.fig is None:                     # give it any figure the drill can draw
    import lsa.artwork.catalog as _cat
    actor.fig = adv._actors[0].fig
if actor.fig is not None:
    actor.t0, actor.t_a, actor.t_h, actor.t_r, actor.t_f = 100.0, 4.0, 2.0, 4.0, 1.5
    actor.hold_xy, actor.drift = (1900.0, 1200.0), 160.0
    actor.h_start, actor.h_near, actor.h_far = 300.0, 900.0, 300.0
    actor.x_lo, actor.x_hi = 0.0, 3840.0
    actor.speed_used = 1.0
    for label, when in (("approaching", 102.0), ("holding", 105.0),
                        ("retreating", 108.0), ("held far", 111.0)):
        a2 = type(actor)()
        a2.__dict__.update(actor.__dict__)
        adv._actors = [a2]
        adv.speed_scale = 1.0
        a2.speed_used = 1.0
        was = a2.state(when)
        adv.speed_scale = 2.0
        adv.respeed(ctx(now=when))
        now_ = a2.state(when)
        ok = (was is not None and now_ is not None
              and abs(was[0][0] - now_[0][0]) < 0.5
              and abs(was[0][1] - now_[0][1]) < 0.5
              and abs(was[1] - now_[1]) < 0.5)
        check(ok, f"advancing: doubling the speed while {label} does NOT move the figure",
              f"{was} -> {now_}")
    check(abs(a2.drift) > 160.0,
          "advancing: ...and the sideways drift really did get faster",
          f"{a2.drift:.1f} px/s from 160.0")
else:
    print("  [note] no figure artwork available; the advancing continuity "
          "checks were skipped")

# =============================================================================
print("\n=== G. Reaction: zero really is motionless, and a moving target is "
      "rewound ===")
print("    reverting the zero floor sets his pop-up target moving on a fresh "
      "install; reverting the rewind kills its trailing edge")

import math as _g_math                                          # noqa: E402
from lsa.settings import (SPEED_ZERO_MODES, speed_default,      # noqa: E402
                          speed_floor)
from lsa.tools.camera_lab import CameraLab as _GLab             # noqa: E402

check("reaction" in SPEED_ZERO_MODES,
      "G1. Reaction is one of the drills that stand still until he asks")
check(speed_floor("reaction") == 0.0 and speed_default("reaction") == 0.0,
      "...so its floor AND its default are both zero -- the drill exactly as it "
      "has always been", f"{speed_floor('reaction')} / {speed_default('reaction')}")
check(speed_floor("ball") == SPEED_MIN and speed_default("ball") == DEFAULT_SPEED,
      "...and no other drill moved: a bouncing ball at zero would read as a "
      "frozen app, which is why the general floor is not zero")
check(AppSettings().speed_for("reaction") == 0.0,
      "G2. a FRESH INSTALL reads zero for Reaction, so nobody who has never "
      "pressed the key is given a moving target")
check(AppSettings().speed_for("ball") == DEFAULT_SPEED,
      "...while every other drill still reads 1.0")

# AND THE DRILL ITSELF IS MOTIONLESS WITH NOBODY ASKED. `Mode.speed_scale`
# defaults to 1.0, which is right for drills designed to move; this one was
# designed to stand still, and leaving the class default alone meant a drill
# built DIRECTLY -- by a test, or by any path where the speed provider is
# missing or has raised -- came up moving. test_modes found it in one run:
# the positions, the repaint count and the two-player mirroring all shifted.
# The settings floor is what the KEYS can reach; this is what the drill is
# when nothing has spoken (RULES E1).
_g_bare = type(instances["reaction"])()
check(_g_bare.speed_scale == 0.0,
      "G1b. a Reaction drill built with nobody asked is MOTIONLESS: the zero "
      "lives on the drill as well as in the settings",
      f"speed_scale={_g_bare.speed_scale}")
# FRESH instances: section C sets the speed on the shared ones to measure the
# effect of doubling it, so reading those here asks what a previous section did
# rather than what the class declares.
_g_others = [n for n, m in instances.items()
             if m.supports_speed and n not in SPEED_ZERO_MODES
             and type(m)().speed_scale != DEFAULT_SPEED]
check(not _g_others,
      "...and no other drill's class default moved", str(_g_others))

_g_ladder = _GLab.speed_ladder_for("reaction")
check(_g_ladder[0] == 0 and _g_ladder[1:] == _GLab.SPEED_LADDER,
      "G3. the key's ladder gains a 0 rung at the bottom for this drill and is "
      "otherwise the ladder every drill walks", str(_g_ladder[:4]))
check(_GLab.speed_ladder_for("ball")[0] == _GLab.SPEED_LADDER[0],
      "...and the other drills' ladder is untouched")
check(_GLab.next_speed(25, -1, "reaction") == 0
      and _GLab.next_speed(0, +1, "reaction") == _GLab.SPEED_LADDER[0]
      and _GLab.next_speed(0, -1, "reaction") == 0,
      "G4. Page Down reaches 0 and stops there; Page Up sets it moving again",
      f"{_GLab.next_speed(25, -1, 'reaction')} / "
      f"{_GLab.next_speed(0, +1, 'reaction')}")
check(_GLab.next_speed(25, -1, "ball") == 25,
      "...and Page Down cannot stop a ball drill dead")


def _live_reaction(scale, seed=7):
    """A Reaction drill with its target UP, at this multiplier."""
    m = type(instances["reaction"])()
    m.speed_scale = scale
    c = ctx(now=100.0)
    m.start(c)
    # `_rng` is a read-only property onto the SHARED reveal's stream, which is
    # the point of it: both halves of a versus round draw from one. Seed that.
    m._reveal.rng.seed(seed)
    t = 100.0
    for _ in range(600):                      # past the 1.2-3.5s random wait
        t += 1 / 60.0
        m.advance(ctx(now=t))
        if m.state == "live":
            return m, t
    return m, t


_g_still, _g_t = _live_reaction(0.0)
check(_g_still.state == "live", "G5. a Reaction target appears at speed zero",
      _g_still.state)
_g_x0, _g_y0 = _g_still.x, _g_still.y
for _i in range(60):
    _g_t += 1 / 60.0
    _g_still.advance(ctx(now=_g_t))
check((_g_still.x, _g_still.y) == (_g_x0, _g_y0),
      "...and after a SECOND of ticking it has not moved one pixel: zero is "
      "motionless, not slow", f"{_g_x0:.2f},{_g_y0:.2f} -> {_g_still.x:.2f},{_g_still.y:.2f}")
check(_g_still.vx == 0.0 and _g_still.vy == 0.0,
      "...and it was never given a heading at all")

_g_fast, _g_t2 = _live_reaction(1.0)
_g_x0, _g_y0 = _g_fast.x, _g_fast.y
for _i in range(30):
    _g_t2 += 1 / 60.0
    _g_fast.advance(ctx(now=_g_t2))
_g_moved = _g_math.hypot(_g_fast.x - _g_x0, _g_fast.y - _g_y0)
check(_g_moved > _g_fast.r,
      "G6. at 100% it travels further than its own radius in half a second, so "
      "it has to be led rather than merely noticed",
      f"moved {_g_moved:.0f}px, radius {_g_fast.r:.0f}px")

# it STAYS hittable: bounces off the playfield rather than leaving ------------
_g_in = _g_fast.playfield(ctx())
_g_out = []
for _i in range(1200):                        # twenty seconds of travel
    _g_t2 += 1 / 60.0
    _g_fast.advance(ctx(now=_g_t2))
    if _g_fast.state != "live":
        break
    if not (_g_in.left() <= _g_fast.x <= _g_in.right()
            and _g_in.top() <= _g_fast.y <= _g_in.bottom()):
        _g_out.append((_g_fast.x, _g_fast.y))
check(not _g_out,
      "G7. it BOUNCES and never leaves the playfield, so the time it is "
      "available does not depend on where it popped up",
      f"{len(_g_out)} frames outside, first {_g_out[:1]}")

# THE REWIND. This drill became a moving-target drill today, so it inherits the
# fault that killed the ball's trailing edge on 2026-11-08 unless it inherits
# the cure. The shot is aimed where the target WAS; the test must still score.
#
# THE OFFSET MUST EXCEED THE RADIUS, or this check cannot fail. The first
# version aimed 106px behind a target with a 110px radius and the shot landed
# inside the circle with or without the rewind: removing the rewind from the
# drill left all 83 checks green. Measured and corrected the same minute.
# Top speed and a longer age put the aim point more than two radii behind.
_g_hit, _g_t3 = _live_reaction(3.0)
_g_age = 0.120
_g_aim_x = _g_hit.x - _g_hit.vx * _g_age
_g_aim_y = _g_hit.y - _g_hit.vy * _g_age
_g_ctx = ctx(now=_g_hit.shown_at + _g_age + 0.5)
_g_ev = modes.ShotEvent(_g_aim_x, _g_aim_y, _g_ctx.now - _g_age, 1)
_g_gap = _g_math.hypot(_g_aim_x - _g_hit.x, _g_aim_y - _g_hit.y)
_g_r = _g_hit.r
check(_g_gap > _g_r * 2.0,
      "G8a. the aim point is more than TWO RADII behind the target, so a hit "
      "test that did not rewind would have to miss it -- without this the next "
      "check passes whether the rewind exists or not",
      f"{_g_gap:.0f}px behind, radius {_g_r:.0f}px")
_g_out1 = _g_hit.shot(_g_ev, _g_ctx)
check(_g_out1.hit,
      "G8b. ...and a shot at where the target WAS when it was fired SCORES: "
      "the hit test rewinds the target by the shot's age, as the ball drills do",
      f"hit={_g_out1.hit}, {_g_gap:.0f}px behind a radius of {_g_r:.0f}px")

_g_miss, _g_t4 = _live_reaction(2.0)
_g_far = _g_math.hypot(_g_miss.vx, _g_miss.vy) * _g_age
_g_ctx2 = ctx(now=_g_miss.shown_at + _g_age + 0.5)
_g_ev2 = modes.ShotEvent(_g_miss.x + _g_miss.r * 3.0, _g_miss.y, _g_ctx2.now - _g_age, 1)
check(not _g_miss.shot(_g_ev2, _g_ctx2).hit,
      "...and the check can still fail: three radii away is still a miss",
      f"the target travels {_g_far:.0f}px in that age")

# a target ALREADY UP changes pace, and can be stopped dead -------------------
_g_live, _g_t5 = _live_reaction(1.0)
_g_was = _g_math.hypot(_g_live.vx, _g_live.vy)
_g_live.speed_scale = 2.0
_g_live.respeed(ctx())
_g_now = _g_math.hypot(_g_live.vx, _g_live.vy)
check(_g_now > _g_was * 1.8,
      "G9. Page Up speeds up the target ALREADY on screen, not just the next "
      "one", f"{_g_was:.0f} -> {_g_now:.0f} px/s")
_g_live.speed_scale = 0.0
_g_live.respeed(ctx())
check(_g_live.vx == 0.0 and _g_live.vy == 0.0,
      "...and Page Down to zero stops the one on screen dead")
_g_live.speed_scale = 1.0
_g_live.respeed(ctx())
check(_g_math.hypot(_g_live.vx, _g_live.vy) > 0.0,
      "...and starting it again gives a standing target a heading")


# =============================================================================
owned.check(lambda ok, what, detail="": check(ok, what, detail))
print("\n" + (f"ALL {CHECKS} CHECKS PASSED" if not FAILED else
              f"{len(FAILED)} OF {CHECKS} CHECKS FAILED"))
raise SystemExit(0 if not FAILED else 1)
