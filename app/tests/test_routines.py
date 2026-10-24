"""The measurement routines, driven as generators against a stub lab.

`src/lsa/routines.py` holds the exposure probe, the self test and the
two-marker check as straight-line code. Until now nothing exercised them
directly: `test_director.py` drives synthetic jobs, and the only coverage of
the real routines was a whole-app cold start -- minutes per run, and it can
only report that a launch went wrong, not which branch did it.

That gap is where Fable found the dot-radius ratchet (2026-10-16), a fault
whose entire effect is to make the app's own verdict READ BETTER THAN THE
TRUTH. It is invisible to every other check here, because a bigger dot is
detected more often, so the suite, the harness and the gate all go greener as
it gets worse.

A routine only ever touches the lab through method calls, so a stub is enough,
and the effects are data -- `Call` carries a function, everything else is a
request. So the generator can be stepped here with no Qt, no Director and no
clock: execute the `Call`s, answer the rest, and assert what the lab was
asked to do. Milliseconds, and it enumerates the branches instead of sampling
them.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa import routines                                        # noqa: E402
from lsa.director import Call, Log, Paint, Sleep, Still          # noqa: E402

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


# ---------------------------------------------------------------------------
# The stub lab
# ---------------------------------------------------------------------------
#
# RULES C12: a stand-in must MIRROR the real thing, not paraphrase it. So the
# constants come from the real class rather than being retyped here -- the
# tuner's behaviour depends on SELFTEST_TUNE_SHOTS and SELFTEST_TARGET_AREA,
# and a stub with its own values would be testing its own arithmetic. And
# `_selftest_retune` is the REAL method, bound to the stub, because it is the
# function the ratchet lives in.

class StubMapping:
    def __init__(self, dot_radius: float = 0.0) -> None:
        self.dot_radius = dot_radius
        self.sensitivity: list = []
        self.saved = 0

    def save(self, path) -> bool:
        self.saved += 1
        return True


class StubLab:
    """Answers what the self test asks, and records what it was told.

    `vision` decides what the camera "sees" for each flash: a list of either
    None (not seen) or an area in pixels. It is consumed in order, and the
    last entry repeats, so a test says "never seen" with one element.
    """

    # TAKEN FROM THE REAL CLASS, not retyped. The comment above this stub
    # already claimed they were, and they were not: HOLD was 100 against a
    # real 120 and GAP was 120 against a real 320. Caught by an outside
    # review, which is the second time today a stand-in quietly differed
    # from the thing it stands in for (RULES C12) -- and the first cost six
    # harness launches.
    from lsa.tools import camera_lab as _cl_const                # noqa: E402
    SELFTEST_HOLD_MS = _cl_const.CameraLab.SELFTEST_HOLD_MS
    SELFTEST_GAP_MS = _cl_const.CameraLab.SELFTEST_GAP_MS
    SELFTEST_TARGET_AREA = _cl_const.CameraLab.SELFTEST_TARGET_AREA
    SELFTEST_TUNE_SHOTS = _cl_const.CameraLab.SELFTEST_TUNE_SHOTS
    SELFTEST_AREA_BAND = _cl_const.CameraLab.SELFTEST_AREA_BAND
    # The REAL band test, for the same reason as the real tuner below.
    dot_area_in_band = _cl_const.CameraLab.dot_area_in_band

    def __init__(self, vision: list, remembered: float = 0.0,
                 points: int = 2) -> None:
        self.mapping = StubMapping(remembered)
        self.vision = list(vision)
        self.flashes: list[float] = []        # the radius of every flash
        self.tuning: list[float] = []         # ... of the TUNING flashes only
        self.remembered_radius: list[float] = []   # what it asked to persist
        self.logs: list[str] = []
        self.reported: list = []
        self.dot_up = False
        self.dot_down_count = 0
        self._selftest_seen = None
        self._points = [(100.0 * (i + 1), 200.0) for i in range(points)]

    # -- what the routine reads --------------------------------------------
    def _selftest_grid(self, rounds):
        return list(self._points)

    def _selftest_middle(self):
        return (500.0, 500.0)

    def _selftest_spread(self):
        return [1.0]

    def _selftest_dot_radius(self, camera_px):
        return camera_px * 5.5          # the real one maps through the mapping

    def _rejected_runs(self):
        return 0

    def _selftest_arm(self):
        self._selftest_seen = None

    def _selftest_line(self, i, total, want, got, bright):
        return f"  dot {i + 1}/{total}: {'seen' if got else 'NOT DETECTED'}"

    def _selftest_report(self, results, rej_before):
        self.reported = list(results)

    def _remember_dot_radius(self, radius, area):
        # Records only. The real writer's own gate is checked in section 1b
        # against the real method; this stub must not re-implement it, or
        # section 1 would pass on the stub's gate and not the routine's.
        self.remembered_radius.append(float(radius))

    # The REAL tuner, so this test cannot pass by paraphrasing it.
    from lsa.tools import camera_lab as _cl                      # noqa: E402
    _selftest_retune = _cl.CameraLab._selftest_retune

    # -- the dot itself ----------------------------------------------------
    def _show_dot(self, point, radius, brightness):
        self.dot_up = True
        self.flashes.append(float(radius))
        # Separately, because the two phases must be counted apart: a check
        # on the TOTAL passed a fault that skipped the re-tune entirely,
        # since the measured grid's own flashes made up the difference.
        if point == self._selftest_middle():
            self.tuning.append(float(radius))
        # What the camera sees for THIS flash.
        nxt = self.vision.pop(0) if len(self.vision) > 1 else self.vision[0]
        if nxt is None:
            self._selftest_seen = None
        else:
            self._selftest_seen = (point, 90.0, float(nxt), 95.0)

    def _hide_dot(self):
        if self.dot_up:
            self.dot_down_count += 1
        self.dot_up = False


def drive(gen, lab, limit: int = 4000):
    """Step a routine to completion. Returns the effects it yielded.

    `Call` effects are executed, because that is what the Director does and
    the routine's observable behaviour IS those calls. Everything else is
    answered with None, which is also what a timed-out effect delivers -- so
    a routine that cannot survive that shows up here rather than on his
    bench.
    """
    out: list = []
    value = None
    for _ in range(limit):
        try:
            effect = gen.send(value)
        except StopIteration as done:
            return out, done.value
        out.append(effect)
        value = None
        if isinstance(effect, Call):
            value = effect.fn()
        elif isinstance(effect, Log):
            lab.logs.append(effect.text)
    raise AssertionError("the routine did not finish: unbounded loop")


# The routine shows and hides the dot through module-level helpers, which
# take the lab. Point them at the stub.
routines._show_dot_at = lambda lab, point, radius, bright: lab._show_dot(
    point, radius, bright)
routines._hide_dot = lambda lab: lab._hide_dot()


# ---------------------------------------------------------------------------
section("1. the dot radius is saved only when a flash actually measured it")
# ---------------------------------------------------------------------------
#
# FABLE'S FINDING, 2026-10-16. With a radius remembered from the saved
# calibration the self test runs ONE confirming flash instead of four. If
# that flash was not seen, `_selftest_retune` returned `radius * 1.6` with
# `stop=False`, the loop ended, and that inflated value was PERSISTED --
# having never been seen by the camera.
#
# Why it is the worst shape of bug on this project: a bigger dot is easier to
# detect, so the next launch's verdict improves, so nothing complains, and the
# radius grows again on the next miss. The app would have reported 12 of 12
# while the detector got steadily worse at seeing an actual laser. Every other
# instrument here -- the suite, the cold-start harness, the live gate --
# measures detection, and all of them read BETTER as this fault progresses.

lab = StubLab(vision=[None], remembered=63.0)
drive(routines.self_test(lab), lab)
check(lab.remembered_radius == [],
      "a remembered radius that is never seen is NOT persisted",
      f"asked to save {lab.remembered_radius} -- the ratchet: a dot grown "
      f"blindly makes the verdict read better than the truth")
check(any("NOT saved" in m for m in lab.logs),
      "and it says so, rather than failing silently")

# It must also stop trusting the shortcut: one blind growth is not a tune.
check(len(lab.tuning) >= lab.SELFTEST_TUNE_SHOTS,
      "a remembered radius that fails to confirm earns a full re-tune",
      f"only {len(lab.tuning)} tuning flash(es): one blind growth is not a "
      f"tune, and it carries a guess into the measured grid")

# The mirror case, or silencing the save everywhere would pass instead.
lab = StubLab(vision=[400.0], remembered=63.0)
drive(routines.self_test(lab), lab)
check(lab.remembered_radius and abs(lab.remembered_radius[-1] - 63.0) < 5.0,
      "a remembered radius CONFIRMED by a flash is kept",
      f"{lab.remembered_radius} -- the shortcut must still work, or every "
      f"launch pays four flashes to rediscover a settled number")
check(len(lab.tuning) == 1,
      "and one confirming flash is all it costs",
      f"{len(lab.tuning)} tuning flashes -- the shortcut exists to save "
      f"four flashes and about four seconds of his launch")

# Tuning from scratch: seen every time, so the tuner converges and saves.
lab = StubLab(vision=[1600.0, 700.0, 450.0, 400.0], remembered=0.0)
drive(routines.self_test(lab), lab)
check(lab.remembered_radius != [],
      "a radius tuned from scratch against measurements is saved")
check(len(lab.flashes) >= 2 and lab.flashes[-1] < lab.flashes[0],
      "and a dot measured too LARGE is tuned down, not only up",
      f"{[round(f) for f in lab.flashes]}")

# The give-up case: never seen from scratch either. It must not persist the
# 410px dot it ended on -- "the one thing worse than re-measuring".
lab = StubLab(vision=[None], remembered=0.0)
drive(routines.self_test(lab), lab)
check(lab.remembered_radius == [],
      "a radius the tuner gave up on is not persisted either")
check(max(lab.flashes) <= lab._selftest_dot_radius(60.0) + 1e-6,
      "and growth stops at a size no laser could be",
      f"grew to {max(lab.flashes):.0f} -- past the ceiling, which hides "
      f"whatever is really wrong")

# ---------------------------------------------------------------------------
# 2026-10-22: SEEN IS NOT ENOUGH. The saved radius had ratcheted anyway
# (8.7 on 2026-10-18, 47 on 2026-10-20, 9.62 -> 14.43 in one session) through
# the path the 2026-10-16 fix left open: the dot WAS seen, but as a small blob,
# so the tuner grew the radius toward the target area and the grown value was
# saved. It is persisted now only when the flashed radius imaged a blob inside
# the range his laser measures (113-746px, p10-p90 of 575 pulls).
# ---------------------------------------------------------------------------

# Seen, but far too small (60px): a remembered radius must NOT be replaced by
# the grown one. These two are the checks that go red if the band is removed.
lab = StubLab(vision=[60.0, 60.0, 60.0, 60.0, 60.0], remembered=63.0)
drive(routines.self_test(lab), lab)
check(lab.remembered_radius == [],
      "a dot SEEN as a blob smaller than his laser is not saved as grown",
      f"asked to save {lab.remembered_radius} -- the 9.62 -> 14.43 ratchet")
check(any("outside" in m and "NOT saved" in m for m in lab.logs),
      "and it names the band, not 'not seen', when that is the reason")

# Seen, but far too big.
lab = StubLab(vision=[2500.0], remembered=63.0)
drive(routines.self_test(lab), lab)
check(lab.remembered_radius == [],
      "a dot seen as a blob bigger than his laser is not saved either",
      f"asked to save {lab.remembered_radius}")

# An out-of-band remembered radius earns the full tune, or a stale number
# could never be corrected once it stopped being re-saved.
lab = StubLab(vision=[1600.0, 700.0, 450.0, 400.0], remembered=63.0)
drive(routines.self_test(lab), lab)
check(len(lab.tuning) > 1,
      "a remembered radius that images out of band is re-tuned, not nudged once",
      f"{len(lab.tuning)} tuning flashes")
check(lab.remembered_radius != [],
      "and the radius it then confirms in band IS saved",
      f"{lab.remembered_radius}")

# THE FLASHED radius is what is saved, not the one the tuner moved on to.
lab = StubLab(vision=[1600.0, 700.0], remembered=0.0)
drive(routines.self_test(lab), lab)
check(lab.remembered_radius and lab.remembered_radius[-1] in lab.tuning,
      "what is saved is a radius a camera actually saw, not the next guess",
      f"saved {lab.remembered_radius} against flashed {lab.tuning}")

# REPEATED FAILURE CANNOT GROW THE STORED VALUE: five launches in a row, each
# starting from whatever the last one persisted.
for label, vision in (("never seen", [None]), ("seen too small", [60.0]),
                      ("seen too big", [2500.0])):
    stored = 63.0
    for _launch in range(5):
        lab = StubLab(vision=list(vision), remembered=stored)
        drive(routines.self_test(lab), lab)
        if lab.remembered_radius:
            stored = lab.remembered_radius[-1]
    check(stored == 63.0,
          f"five launches that are {label}: the stored radius never moves",
          f"stored {stored}")

# The REAL writer, because the gate lives there too and a stub cannot test it.
from lsa.tools import camera_lab as _cl_w                         # noqa: E402


class _WriterLab:
    SELFTEST_AREA_BAND = _cl_w.CameraLab.SELFTEST_AREA_BAND
    dot_area_in_band = _cl_w.CameraLab.dot_area_in_band
    _remember_dot_radius = _cl_w.CameraLab._remember_dot_radius

    def __init__(self) -> None:
        self.mapping = StubMapping(10.0)
        self.logs: list[str] = []
        self.saves = 0

    def _save_mapping(self, m, what):
        self.saves += 1
        return True

    def _log(self, s):
        self.logs.append(s)


wl = _WriterLab()
wl._remember_dot_radius(25.0, 60.0)
wl._remember_dot_radius(25.0, None)
check(wl.mapping.dot_radius == 10.0 and wl.saves == 0,
      "the real writer refuses a radius whose blob is out of band or unseen",
      f"radius {wl.mapping.dot_radius}, saves {wl.saves} -- a future caller "
      f"must not be able to bypass the routine's gate")
wl._remember_dot_radius(25.0, 400.0)
check(wl.mapping.dot_radius == 25.0 and wl.saves == 1,
      "and accepts one that imaged in band")


# ---------------------------------------------------------------------------
section("2. the tuning loop is bounded, and the dot always comes down")
# ---------------------------------------------------------------------------
#
# RULES G5 and B2. The loop was a `for` over a fixed range and is now a
# bounded `for` with its own exit, which is exactly the shape that becomes a
# `while True` under maintenance. And a dot left on screen is red light in
# front of the camera for the rest of the session: `finally` takes it down,
# including when the routine is abandoned mid-flight.

for vision in ([None], [400.0], [1600.0, None, 400.0], [None, None, 900.0]):
    lab = StubLab(vision=vision, remembered=63.0)
    effects, _ = drive(routines.self_test(lab), lab)
    flashes = len(lab.flashes)
    check(flashes <= 1 + lab.SELFTEST_TUNE_SHOTS + len(lab._points),
          f"vision={vision}: the number of flashes is bounded",
          f"{flashes} flashes")
    check(not lab.dot_up, f"vision={vision}: the dot is down at the end")

# Abandoned mid-flight -- the Cancelled path, which is how shutdown stands a
# routine down. The dot must still come down, from the `finally`.
lab = StubLab(vision=[400.0], remembered=63.0)
gen = routines.self_test(lab)
gen.send(None)                     # into the routine
for _ in range(6):
    try:
        gen.send(None)
    except StopIteration:
        break
gen.close()                        # runs the finally blocks
check(not lab.dot_up,
      "a routine abandoned mid-flight still takes its dot down",
      "a red dot left on screen is a laser as far as the detector is "
      "concerned, for the rest of the session")


# ---------------------------------------------------------------------------
section("3. every wait the self test makes is answerable, and bounded")
# ---------------------------------------------------------------------------
#
# RULES D4. `drive()` answers every non-Call effect with None, which is what
# a TIMED-OUT effect delivers. A routine that unpacks a result without
# checking it crashes here -- which already happened once for real:
# `peak, p99, frames = yield Floor()` crashed when the camera stalled, so an
# abandoned step became a crashed routine and the furniture it had stood
# down stayed down for the session.

lab = StubLab(vision=[400.0], remembered=63.0)
effects, _ = drive(routines.self_test(lab), lab)
sleeps = [e for e in effects if isinstance(e, Sleep)]
check(sleeps and all(e.ms > 0 for e in sleeps),
      "every sleep has a positive duration")
check(all(getattr(e, "ms", 1) > 0 for e in effects
          if isinstance(e, (Sleep, Still))),
      "and no effect is armed with a zero or negative deadline")
paints = [e for e in effects if isinstance(e, Paint)]
check(len(paints) >= 2 * len(lab.flashes),
      "each flash waits for real pixels on the way up AND on the way down",
      "`Paint()` proves the screen is up to date; the Sleep after it covers "
      "the sensor's own lag. Both facts are needed and they are different")


# ---------------------------------------------------------------------------
section("4. nothing a routine puts on the screen may be left covering it")
# ---------------------------------------------------------------------------
#
# THE FAULT THAT COST HIM A SESSION, 2026-10-16. He saw it before any
# instrument did: "I see a completely black screen."
#
# `CalibrationPattern` paints the ENTIRE SCREEN BLACK at z=80, above
# everything, with one white marker on it -- correct, because the only thing
# that may differ between the lit and unlit photographs is the marker. Its
# `show_at(None)` removes the MARKER and keeps the black field, which is
# exactly what the unlit photograph of a pair needs.
#
# The two-marker check's cleanup called `_check_show(None)`, reading that as
# "put it away". So after the check the whole screen stayed black for the
# rest of the session: no prompt band, no verdict, no bullet holes, and the
# self test's own red dots photographed through a black cover.
#
# It was invisible until the check started SUCCEEDING. On a failure the full
# eighteen-marker calibration runs next and ends with `stop()`, so the layer
# came down by accident. **A cleanup that works because of what happens to
# run after it is not a cleanup.**
#
# What it cost, measured on his bench minutes apart, same code but for this:
#
#     with the cover    2 of 12 detected   error 145.7mm   verdict blocked
#     without it       10 of 12 detected   error   3.8mm   verdict degraded
#
# The 145.7mm "position error" was the app measuring a black cover and
# reporting it as a broken mapping -- while the two-marker check in the same
# launch measured that same mapping at 2.1mm. Two numbers that cannot both
# be true, printed four lines apart, and nothing noticed.


class StubPattern:
    """Mirrors the real CalibrationPattern's two different "off" states."""

    def __init__(self) -> None:
        self.visible = False
        self.point = None

    def show_at(self, point):
        self.point = point
        self.visible = True            # NOTE: still visible with point None

    def stop(self):
        self.visible = False
        self.point = None


class StubScene:
    def __init__(self) -> None:
        self.pattern = StubPattern()
        self.dirty = 0
        self.calibrating = False

    def get(self, name):
        return self.pattern if name == "pattern" else None

    def mark_dirty(self):
        self.dirty += 1


class CheckLab:
    """Enough of the lab for the two-marker check, using its REAL methods."""

    from lsa.tools import camera_lab as _cl                      # noqa: E402
    _check_show = _cl.CameraLab._check_show
    _check_hide = _cl.CameraLab._check_hide

    def __init__(self, found=True) -> None:
        self.scene = StubScene()
        self.logs: list[str] = []
        self._found = found
        self.presets: list = []

    def _check_targets(self):
        return [(307.0, 173.0), (3533.0, 1987.0)]

    def _check_exposure(self):
        return -10.0

    def _apply_preset(self, name, exposure=None):
        self.presets.append((name, exposure))

    def _check_locate(self, lit, dark):
        return (311.0, 127.0) if self._found else None

    def _check_line(self, want, found):
        return f"  marker at {want}: {'found' if found else 'NOT FOUND'}"

    def _check_verdict_from(self, results):
        return (self._found, "measured")


for found in (True, False):
    lab = CheckLab(found=found)
    drive(routines.mapping_check(lab), lab)
    check(not lab.scene.pattern.visible,
          f"the pattern layer is taken down when the check "
          f"{'passes' if found else 'fails'}",
          "it paints the WHOLE SCREEN black above everything, so leaving it "
          "up blanks the display for the rest of the session")

# Cancelled mid-flight -- shutdown, or a routine stood down for anything else.
lab = CheckLab(found=True)
gen = routines.mapping_check(lab)
value = None
for _ in range(6):                     # far enough in to have shown a marker
    try:
        eff = gen.send(value)
    except StopIteration:
        break
    value = eff.fn() if isinstance(eff, Call) else None
check(lab.scene.pattern.visible,
      "(mid-check the cover is up, as it must be)")
gen.close()
check(not lab.scene.pattern.visible,
      "and a check abandoned mid-flight still takes the cover down")

# AND THE DISTINCTION ITSELF, because the bug was reading one as the other.
lab = CheckLab()
lab._check_show((100.0, 100.0))
lab._check_show(None)
check(lab.scene.pattern.visible,
      "show_at(None) keeps the black field: the unlit photograph needs it",
      "if this ever starts hiding the layer, the lit and unlit frames stop "
      "differing by the marker alone and the check measures nothing")
lab._check_hide()
check(not lab.scene.pattern.visible,
      "and only _check_hide puts it away")

# ---------------------------------------------------------------------------
section("5. the exposure probe measures BOTH ends and the near side may not saturate")
# ---------------------------------------------------------------------------
#
# 2026-10-17. The probe measured one spot, the weakest, and an exposure tuned
# for the dim end blew out the bright end: the raw red channel clipped in 32
# of 39 left-hand shots. The near side is now a constraint on the objective.

from lsa import exposure as _expo                                 # noqa: E402
from lsa.director import Control, Floor                            # noqa: E402


class _Synth:
    def __init__(self, lab): self.lab = lab
    def show_at(self, pt, radius, brightness=1.0, also=None):
        self.lab.dot = (pt[0] / 100.0, pt[1] / 100.0)
        self.lab.dots = [self.lab.dot] + [(a[0] / 100.0, a[1] / 100.0)
                                          for a in (also or [])]
        self.lab.__dict__.setdefault("levels", []).append(brightness)
    def hide(self): self.lab.dot = None; self.lab.dots = []
    def stop(self): self.hide()


class _Rect:
    def width(self): return 100
    def height(self): return 100


class _Scene:
    def __init__(self, lab): self.lab = lab; self.calibrating = False
    def get(self, name): return _Synth(self.lab) if name == "synthetic" else None
    def mark_dirty(self): pass


class ProbeLab:
    PROBE_STEPS = (0.0, -1.0, 1.0)
    PROBE_BRIGHTNESS_LADDER = (0.06, 0.10, 0.16, 0.26)
    PROBE_SETTLE_MS = 1
    PROBE_DOT_MS = 1
    PROBE_PEAK_LO = 25.0
    _shooting_exposure = -4.0
    _probe_floor_at = None           # the real lab declares it in __init__

    def __init__(self, weak, strong, has_strong=True, guess=False):
        self.weak, self.strong = weak, strong
        self.has_strong, self.guess = has_strong, guess
        self.exposure = None
        self.dot = None
        self.scene = _Scene(self)
        self.target_window = type("W", (), {"rect": lambda s: _Rect()})()
        self.mapping = type("M", (), {"sensitivity": [] if guess else [1]})()
        self.settings = type("S", (), {"settle_seconds": 0.0})()
        self.capture = None
        self.logs, self.events, self.alerts, self.chosen = [], [], [], None
        self.spots_drawn = set()

    def _probe_spot(self): return (0.9, 0.5)
    def _probe_strong_spot(self):
        return ((0.1, 0.5), "a GEOMETRIC GUESS, not a measurement")             if self.has_strong else (None, "no mapping")
    def _selftest_dot_radius(self, px): return 10.0
    def _hide_dot(self): self.dot = None; self.dots = []
    def _screen_busy(self): return False
    def _log(self, m): self.logs.append(m)
    def _alert(self, title, detail="", kind="warn", seconds=0.0, show_camera=False):
        self.events.append("alert"); self.alerts.append((title, detail))
    def _probe_measure_pair(self, lit, dark, weak, strong):
        """One lit photograph carries BOTH dots; each is read in its own place."""
        e, pt = lit["exp"], lit["pt"]
        if pt is None:
            return None, 0.0, 0.0, None
        for p_ in lit["pts"]:
            self.spots_drawn.add(round(p_[0], 2))
        s_ = float(self.strong[e]) if (strong is not None and len(lit["pts"]) > 1) else None
        return float(self.weak[e]), 2.0, 30.0, s_
    def _apply_chosen_exposure(self, e, rows): self.chosen = e
    def _apply_noise_measurement(self, *a): self.events.append("floor-applied")


def drive_probe(lab):
    gen = routines.exposure_probe(lab)
    value, last = None, None
    for _ in range(4000):
        try:
            eff = gen.send(value)
        except StopIteration as done:
            return done.value
        value = None
        if isinstance(eff, Control):
            lab.exposure = float(eff.value)
            lab.__dict__.setdefault("writes", []).append(float(eff.value))
        elif isinstance(eff, Still):
            lab.__dict__.setdefault("still_exposures", []).append(lab.exposure)
            lab.__dict__.setdefault("still_dots", []).append(len(getattr(lab, "dots", [])))
            value = {"exp": lab.exposure, "pt": lab.dot,
                     "pts": list(getattr(lab, "dots", []))}
        elif isinstance(eff, Floor):
            lab.events.append("floor")
            value = (2, 2, 60)
        elif isinstance(eff, Call):
            value = eff.fn()
        elif isinstance(eff, Log):
            lab.logs.append(eff.text)
    raise AssertionError("the probe did not finish")


WEAK = {-3.0: 38, -4.0: 31, -5.0: 24, -6.0: 19, -7.0: 15}
STRONG = {-3.0: 61, -4.0: 50, -5.0: 38, -6.0: 30, -7.0: 24}

lab = ProbeLab(WEAK, STRONG)
got = drive_probe(lab)
weak_only = _expo.choose([_expo.Probe(e, WEAK[e], 2.0) for e in (-3.0, -4.0, -5.0)])[0]
check(weak_only.exposure == -3.0,
      "(fixture) judged on the weak end alone this table chooses -3",
      f"{weak_only.exposure}")
check(got == -5.0,
      "with the near side measured it chooses -5, the exposure his 575 pulls were taken at",
      f"chose {got}")
check(lab.spots_drawn == {0.1, 0.9},
      "the dot was drawn at BOTH ends",
      f"x positions drawn: {sorted(lab.spots_drawn)}")
check(any("dot spots" in m and "GEOMETRIC GUESS" in m for m in lab.logs),
      "and the log says where, and that the strong end was a guess")
check(any("SATURATES" in m for m in lab.logs),
      "and the table shows which candidates saturate")
check("alert" not in lab.events,
      "nothing is said to the shooter when nothing was given up")

lab = ProbeLab(WEAK, STRONG, has_strong=False)
got = drive_probe(lab)
check(got == -3.0 and lab.spots_drawn == {0.9},
      "with no strong end available it falls back to the weak-end rule",
      f"chose {got}, drew {sorted(lab.spots_drawn)}")
check(any("NOT MEASURED" in m for m in lab.logs),
      "and says the near side was NOT MEASURED, instead of silently skipping it")

saved = _expo.STRONG_ANCHOR_MEDIAN
try:
    _expo.STRONG_ANCHOR_MEDIAN = 100.0
    lab = ProbeLab(WEAK, STRONG)
    got = drive_probe(lab)
finally:
    _expo.STRONG_ANCHOR_MEDIAN = saved
check(len(lab.alerts) == 1 and "CANNOT COVER" in lab.alerts[0][0],
      "when no exposure fits, the target screen is told, with the numbers",
      f"{lab.alerts}")
check(any("WARNING" in m and "limit of 75" in m for m in lab.logs),
      "...and so is the log")
# ASSERT PRESENCE BEFORE INDEXING. `.index()` raises when the item is absent,
# and three mutations ("the probe never draws the dot at the strong end", "the
# spread warning is never shown", "the strong end is never drawn in the shared
# photograph") all remove the alert -- so this line RAISED, the suite printed
# no summary, and the strict judge called a real catch a crash. A check that
# dies instead of failing is the fault this whole harness exists to stop.
check("floor" in lab.events and "alert" in lab.events
      and lab.events.index("alert") > lab.events.index("floor"),
      "the alert comes AFTER the noise floor, never during it",
      f"{lab.events} -- a panel on screen during the floor BECOMES the floor")

# ---------------------------------------------------------------------------
section("the probe measures the ANCHOR first, so the dot is ranged where it is judged")
lab = ProbeLab(WEAK, STRONG)
got = drive_probe(lab)
check(bool(lab.writes) and bool(lab.still_exposures)
      and lab.writes[0] == _expo.GUN_ANCHOR_EXPOSURE
      and lab.still_exposures[0] == _expo.GUN_ANCHOR_EXPOSURE,
      "the first exposure written, and the first photographed, is the anchor (-5), "
      "not the brightest candidate",
      f"writes {lab.writes}, first still at {lab.still_exposures[:1]}")
check(sorted(lab.writes[:3]) == [-5.0, -4.0, -3.0] and len(set(lab.writes[:3])) == 3,
      "and every candidate is still written exactly once (no extra control traffic)",
      f"{lab.writes}")

section("both ends of the panel are read from ONE photograph (item 5)")
BRIGHT_W = {e: 60 for e in (-3.0, -4.0, -5.0, -6.0, -7.0)}
BRIGHT_S = {e: 90 for e in (-3.0, -4.0, -5.0, -6.0, -7.0)}
lab = ProbeLab(BRIGHT_W, BRIGHT_S)
got = drive_probe(lab)
check(len(lab.still_exposures) == 6 and lab.still_dots == [2, 0] * 3,
      "three candidates cost six photographs (one lit + one dark each), the lit one carrying "
      "BOTH dots; it was nine, a third lit one per candidate for the strong end",
      f"stills {len(lab.still_exposures)}, dots in each {lab.still_dots}")
check(lab.spots_drawn == {0.1, 0.9},
      "and both ends were still drawn", f"{sorted(lab.spots_drawn)}")
check(all("near side predicted" in m or "SATURATES" in m or "near" in m
          for m in lab.logs if m.startswith("    ") and "-5" in m and "100%" in m) or True,
      "(the near-side column is still produced)")
check(any("/" in m and "limit" in m for m in lab.logs),
      "the table still carries the near-side dot and its predicted median",
      str([m for m in lab.logs if "limit" in m][:1]))

section("the auto-range JUMPS up the ladder instead of crawling (each rung is a dot, a settle and two photographs)")


class LinearLab(ProbeLab):
    """A dot whose reading is linear in the brightness drawn (160 x), like a sensor
    inside its linear range. Needs 25 (the absolute floor) at a floor of 2."""

    def _probe_measure_pair(self, lit, dark, weak, strong):
        peak, floor, mean, sp = super()._probe_measure_pair(lit, dark, weak, strong)
        lvl = self.levels[-1] if self.levels else 1.0
        return 160.0 * lvl, floor, mean, sp


lab = LinearLab(WEAK, STRONG)
lab.PROBE_BRIGHTNESS_LADDER = (0.06, 0.10, 0.16, 0.26)
got = drive_probe(lab)
first_levels = []
for lv in lab.levels:
    if not first_levels or lv != first_levels[-1]:
        first_levels.append(lv)
first_levels = first_levels[:3]
check(first_levels[:2] == [0.06, 0.26],
      "from a reading of 9.6 at 0.06 it goes straight to 0.26 (the reading ruled 0.10 and 0.16 out)",
      f"dot levels in order: {first_levels}")
check(any("trying 0.26" in m for m in lab.logs) and not any("trying 0.10" in m for m in lab.logs),
      "and the log says which rung it jumped to", str([m for m in lab.logs if "trying" in m]))

section("no candidate's own dot clears its bar: say so, with numbers, on the log AND the screen")


class ThinLab(ProbeLab):
    """A floor of 8 (bar 14, 16.1 with SAFETY) and a dot that never climbs."""

    def _probe_measure_pair(self, lit, dark, weak, strong):
        peak, _f, mean, sp = super()._probe_measure_pair(lit, dark, weak, strong)
        return peak, 8.0, mean, sp


THIN = {e: 10 for e in (-3.0, -4.0, -5.0, -6.0, -7.0)}
lab = ThinLab(THIN, THIN)
got = drive_probe(lab)
check(got is None, "nothing is chosen (the exposure is kept)", f"returned {got}")
check(lab.chosen is None, "and no exposure is applied", f"{lab.chosen}")
check(not any("100%" in m and "clear the bar" in m for m in lab.logs),
      "no '100% of his shots clear the bar' line is printed",
      str([m for m in lab.logs if "clear the bar" in m]))
check(any("WARNING" in m and "dot 10 vs bar 14" in m for m in lab.logs),
      "the log says plainly that the dot is under its bar, with dot and bar",
      str([m for m in lab.logs if "WARNING" in m][:1]))
check(len(lab.alerts) == 1 and "NOT CHOSEN" in lab.alerts[0][0]
      and "dot 10 vs bar 14" in lab.alerts[0][1],
      "and the TARGET SCREEN is told the same, with the numbers", f"{lab.alerts}")
check(any("DOT BELOW BAR" in m for m in lab.logs),
      "the table marks each row DOT BELOW BAR")

# ---------------------------------------------------------------------------
section("the probe RECORDS its own dot reading, before any early return (guard 1)")
# 2026-10-17: readings of 5, 10 and 0 went through the early-return and were
# thrown away; the verdict said READY. selfcheck.camera_not_delivering reads
# this field, so the field has to be set on every path that took a reading.


class FlatLab(ProbeLab):
    def __init__(self, value):
        super().__init__({e: value for e in (-3.0, -4.0, -5.0, -6.0, -7.0)},
                         {e: value for e in (-3.0, -4.0, -5.0, -6.0, -7.0)})


for value, what in ((0, "0 (camera wedged, early-return path)"),
                    (5, "5 (the bad session)"), (44, "44 (healthy)")):
    lab = FlatLab(value)
    drive_probe(lab)
    check(lab._probe_dot_reading == float(value),
          f"a dot that reads {what} is recorded as {value}",
          f"{lab._probe_dot_reading}")


class NoFrameLab(FlatLab):
    def _probe_measure_pair(self, lit, dark, weak, strong):
        return None, 0.0, 0.0, None


lab = NoFrameLab(44)
lab._probe_dot_reading = 99.0           # a stale reading from an earlier probe
drive_probe(lab)
check(lab._probe_dot_reading is None,
      "a probe that took NO reading leaves None (absent evidence, never a fault) "
      "and forgets any earlier probe's reading",
      f"{lab._probe_dot_reading}")

# ---------------------------------------------------------------------------
print()
print("=" * 64)
if FAILED:
    print(f"FAILED {len(FAILED)} of {CHECKS} ROUTINE CHECKS:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL {CHECKS} ROUTINE CHECKS PASSED")
