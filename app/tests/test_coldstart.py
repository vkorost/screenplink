"""Cold starts, many of them, with the timing deliberately shaken.

THE PROBLEM THIS IS FOR
-----------------------
The repeatability gate measured, on identical code in the same room:

    launch   calibration   self test
      1        1.08mm        0/12
      2        1.00mm       12/12
      3        0.93mm        0/12
      4        1.03mm        1/12
      5        1.03mm       12/12

All-or-nothing is never a tuning problem. It was the app's own routines
running at once with nothing arbitrating -- the noise floor measuring the
exposure probe's dots, the probe measuring a screen the floor measurement
had covered, the self test judged against a bar from neither.

Finding each of those cost about seventy seconds of the shooter's monitor
plus a log read, and whether one appeared at all depended on millisecond
timing. Five live launches is a terrible way to sample that. This runs the
same chain headless, in seconds, with every asynchronous reply delayed by a
random amount, so a few dozen runs cover orderings that hardware would take
weeks to produce.

WHAT IS BEING TESTED
--------------------
The real CameraLab, the real timers, the real detector and tracker, the real
calibration and the real verdict. Only the camera is a stand-in, and it
grabs the app's actual target window -- so the dots the app paints are found
by the same code that finds a laser.

The headline assertion is not "did it work". It is: NO TWO EXCLUSIVE CAMERA
OPERATIONS MAY EVER OVERLAP. A race is a statement about order, so it is
asserted as order, not inferred from an outcome -- a run can produce the
right answer and still have been wrong.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# The harness builds a real CameraLab, so without this it writes into the
# live app's log and the two streams interleave. A timing loop then read
# the HARNESS's verdict and reported it as a measurement of the bench.
# The log AND the images. The log was separated after a harness line
# was read as his hardware; the images were the same problem and were
# left writing over recordings/verdict_screen.png with pictures from
# a simulated 800x800 camera.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from owned_files import (OwnedFiles, isolate_outputs,  # noqa: E402
                         assert_silent_mixer, assert_silent_listener)
isolate_outputs("coldstart")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "src"))

from PySide6.QtCore import QRect, QTimer                      # noqa: E402
from PySide6.QtWidgets import QApplication                    # noqa: E402

import fake_camera                                            # noqa: E402

checks: list[bool] = []

# The exposure every simulated launch STARTS at (plan 02, step 2a).
#
# C15 says use HIS timings, because timers race against them. Exposure is not
# a timing: it is the fake camera's optics model (`reference_exposure = -5`,
# its noise and gain calibrated against -5/-4), not his camera. His
# settings.local.json moved from -4 to -3 when a live probe chose it, the
# harness reads that file, the probe then ended at -3 every launch, and the
# floor check -- tuned when the file said -4 -- began failing on 1 to 3 of 7
# launches. The bar was derived at one exposure and the file moved the
# harness to another. Pinned here so the harness stops tracking a value that
# means something only on his bench. Do NOT "restore" his file's value for
# consistency with the timings above.
HARNESS_START_EXPOSURE = -4.0

# How far a launch's noise floor may climb above the fake's own EMPTY-SCREEN
# floor measured in THE SAME STATE, expressed as a ratio of BARS -- what the
# detector actually thresholds on -- not as a difference of floors.
#
#     ratio = exposure.min_peak_for(floor) / exposure.min_peak_for(reference)
#
# A difference was the first form (`+4` counts) and it inherited the very
# fault the app's own gain-map judge was moved off twelve hours earlier: at
# these levels the floor is quantised to nothing, so a count is worth 1.4x at
# a floor of 4 and 1.08x at a floor of 40. The bar is the quantity that
# decides whether a shot registers, and `LEARN_MAX_FLOOR_RISE` (the app's
# judge, 2.0) is already a ratio of bars. This makes the check level- and
# exposure-independent instead of only exposure-independent.
#
# DERIVED FROM MEASUREMENT, not chosen. 42 harness launches have now been
# observed (21 in the first three runs after the fake's floor was made the
# app's statistic, 21 more on review), 2026-10-16: the maximum excess of a
# floor over its empty-screen reference was 2 counts. Two counts moves the
# bar by at most 1.25x anywhere on the scale (reference 2 -> 4: bar 8 -> 10;
# at the 4 this fake reads at -3: bar 10 -> 12, 1.2x). 1.4 leaves ~12%
# headroom over that worst case, and equals the old `+4` at the reference
# that was measured (bar 10 -> 14), so the tolerance is continuous with what
# was reviewed. It sits far below the app's own 2.0 and below the incident it
# exists to catch: the verdict board took the live floor from 0-2 to 16-18,
# a bar of 8 -> 23 (2.9x).
FLOOR_BAR_RATIO_MAX = 1.4


def check(label: str, cond: bool, detail: str = "") -> None:
    checks.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


def _lab_module():
    """Import camera_lab the way the app does, without launching it."""
    import importlib
    return importlib.import_module("lsa.tools.camera_lab")


def run_cold_start(app, jitter, seed, seconds=120.0, seed_mapping=True):
    """One launch, start to verdict, with no camera and no screen.

    Returns the lab and its fake camera so the caller can interrogate both.
    """
    mod = _lab_module()

    # Build it with auto-connect OFF, so the REAL camera is never opened.
    # The first version let the constructor connect and disconnected
    # afterwards, which touched the real device for about a second on
    # every simulated launch -- a harness whose whole purpose is to stop
    # needing the camera was reaching for it dozens of times a run.
    #
    # Restoring the setting afterwards matters as much: forcing it off is
    # a TEST concern, and leaking it into settings.local.json once left
    # the app refusing to connect on launch, which is worse than anything
    # it was guarding against.
    import lsa.settings as _settings
    _orig_load = _settings.AppSettings.load

    his_file = {}

    @staticmethod
    def _no_connect(path):
        st = _orig_load(path)
        st.auto_connect = False
        # What his file says, read BEFORE the pin overwrites it, so the line
        # that reports the pin can also say what it replaced. Twice this
        # project has reported a harness number as a measurement of his
        # bench, and both times the giveaway was inside a line nobody read.
        his_file["exposure"] = st.exposure
        # Pinned HERE, at the load, and not afterwards on `lab.settings`: the
        # constructor copies `settings.exposure` into `_boot_exposure` and
        # `_shooting_exposure`, so setting it later changes only a field
        # nothing reads.
        st.exposure = HARNESS_START_EXPOSURE
        return st

    _settings.AppSettings.load = _no_connect
    # AND IT MUST NOT TAKE THE TEST PROCESS DOWN WITH IT.
    #
    # `closeEvent` ends in `os._exit(0)`, deliberately, because a thread
    # blocked inside the camera driver cannot be joined and a shutdown that
    # overruns gets force-killed, which wedges the camera. But a real
    # CameraLab built INSIDE this process means its shutdown exits the test
    # runner -- with status 0, which every caller reads as a pass.
    #
    # Measured 2026-10-15 22:09: this file printed section 1, began section
    # 2, and vanished. It has been running one launch out of seven and
    # reporting success for as long as it has existed, and "simulated cold
    # starts: 8 of 8 READY" was quoted from it. Set on the CLASS before
    # construction, so no shutdown can race it (RULES C1).
    mod.CameraLab.hard_exit_on_close = False
    try:
        lab = mod.CameraLab()
    finally:
        _settings.AppSettings.load = _orig_load
    assert lab.capture is None, "the harness must not open the real camera"
    assert_silent_mixer(lab)
    assert_silent_listener(lab)
    lab.settings.auto_connect = True
    lab._save_settings = lambda *a, **k: None
    lab._schedule_save = lambda *a, **k: None

    # AND IT MUST NOT WRITE HIS CALIBRATION EITHER.
    #
    # This is the third time this project has let a test write something
    # the user owns, and the first two are already in CLAUDE.md: a smoke
    # test that left `auto_connect: false` in settings.local.json so the
    # app stopped connecting on launch, and the exposure that poisoned
    # itself run after run.
    #
    # Here the simulated screen is 800x800, so every harness run saved an
    # 800x800 mapping over a real one measured at 3840x2160 -- taking the
    # five sensitivity regions learned from his actual shots with it. The
    # app would have survived it (a mapping for another screen size is
    # rejected on load and remeasured) but the sensitivity would not: it
    # exists precisely to end the bootstrap deadlock where the threshold
    # rejects the shots that would teach it.
    #
    # BLOCKED FOR HIS FILES, ALLOWED FOR THE ISOLATED ONE, and the
    # difference cost this harness its most important coverage.
    #
    # The original fix blocked the save outright -- "there is then no path
    # at all from a simulated run to a real file, rather than a path that
    # depends on a variable being right". Safe, and it made the fast path
    # UNTESTABLE: no run could ever leave a saved mapping behind, so the
    # two-marker check never ran once, every launch paid for eighteen
    # markers, and a simulated launch took 81 seconds against his real 30.
    #
    # That is the path his launches actually take. It is also where the
    # 2026-10-16 black-screen bug lived -- a full-screen cover left up by
    # the two-marker check's cleanup, invisible to this harness because the
    # check never ran. Six launches of "verdict=ready, clean" and the thing
    # under test was absent (RULES C1).
    #
    # So the guard names the files rather than the operation: anything
    # under the project root that is NOT inside the isolated output
    # directory is refused LOUDLY. His calibration.json cannot be written
    # whatever an environment variable says, and the harness can still
    # persist a mapping between runs the way his machine does.
    import lsa.calibrate.mapping as _mapping
    _real_save = _mapping.Mapping.save
    _allowed = Path(os.environ["LSA_CALIBRATION"]).resolve()

    def _guarded_save(self, path):
        try:
            target = Path(path).resolve()
        except Exception:
            raise AssertionError(f"harness tried to save to {path!r}")
        if target != _allowed:
            raise AssertionError(
                f"THE HARNESS TRIED TO WRITE {target}. Only {_allowed} is "
                f"allowed. A simulated 800x800 mapping once went over his "
                f"real 3840x2160 one and took five learned sensitivity "
                f"regions with it.")
        return _real_save(self, target)

    _mapping.Mapping.save = _guarded_save

    def view():
        """What is in front of the camera right now.

        The flash panel wins when it is up, because it physically covers
        the target screen -- that is the whole reason the screen search
        works, and the reason a leaked cover once made twelve enormous
        red dots invisible.

        `isVisible()` is NOT consulted: offscreen windows report False
        for it even while they are rendering perfectly well, so asking
        made the camera blind and the screen search fail at step one.
        """
        # ONLY IF IT IS ACTUALLY VISIBLE, which is the whole of what a
        # camera can tell. `_close_flash` hides the panel and calls
        # `deleteLater`, so for a turn of the event loop -- and longer, if
        # any path leaves the reference set -- `lab._flash` is a live
        # QWidget that is not on the screen. This provider photographed it
        # anyway, so the stand-in showed the detector a hidden panel while
        # the target screen underneath carried the markers.
        #
        # Cost: "pattern calibration failed: only 0 of 12 markers were
        # visible", every launch, in a harness whose whole job is to find
        # faults before he does -- and the live app was finding 10 of 12 on
        # the same code at the same minute. A stand-in that sees something
        # no camera could see invents failures as readily as one that
        # misses real ones (RULES C12).
        win = getattr(lab, "_flash", None)
        if win is not None:
            try:
                if win.isVisible():
                    return win.grab().toImage()
            except RuntimeError:
                pass            # already deleted; fall through
        win = lab.target_window
        if win is not None:
            return win.grab().toImage()
        return None

    fake_camera.install(lab, view, jitter_ms=jitter, seed=seed)
    # HIS TIMINGS, not shortened ones. "Nothing about ORDER changes" was
    # wrong, and that comment is why it went unexamined for a day.
    #
    # After pattern calibration the app arms a noise floor at +1.2s and the
    # exposure probe at +2.5s. With the floor shortened to one second the
    # probe finds a free world and the harness asserts `refusals == []` and
    # passes. His `calibrate_seconds` is 3.0 and `settle_seconds` 1.5, so
    # on HIS bench the floor is still running when the probe asks, and the
    # probe is queued and refused -- every launch.
    #
    # Shortened timing manufacturing a pass is the review's "finds races by
    # luck" with the sign reversed, and it is worse: a race that cannot
    # happen here still happens there, and the harness says it is fine.
    #
    # Read from the settings file rather than hardcoded, so the harness
    # tracks whatever he actually runs.
    lab.settings.calibrate_seconds = float(
        getattr(lab.settings, "calibrate_seconds", 3.0) or 3.0)
    lab.settings.settle_seconds = float(
        getattr(lab.settings, "settle_seconds", 1.5) or 1.5)
    print(f"     timings: settle {lab.settings.settle_seconds:g}s, "
          f"floor {lab.settings.calibrate_seconds:g}s (his)   "
          f"start exposure {lab._shooting_exposure:g} (PINNED by the harness; "
          f"his file says {his_file.get('exposure')})")
    # ASSERT THE EFFECT, NOT THE ATTEMPT (RULES C1): the pin has to reach the
    # fields the app actually reads.
    assert lab._shooting_exposure == HARNESS_START_EXPOSURE == lab._boot_exposure, (
        "the harness's start exposure did not reach the app")
    # SEED A SENSITIVITY MAP, so the third routine runs too.
    #
    # In the failing live launches the app was applying and WITHDRAWING a
    # sensitivity map at the same moment the noise floor and the exposure
    # probe were running -- three routines, not two. The simulated run was
    # missing that one entirely: the map is carried from the saved
    # calibration, and the simulated screen is a different size, so the
    # saved mapping is rejected on load and the path never fires.
    #
    # A clean sweep with a blind spot over one of the three things that
    # were actually colliding would prove less than it appears to. These
    # rows sit inside the modelled screen quad with the 5x spread the real
    # ones measured (weakest 16, strongest 86).
    quad = fake_camera.Optics().quad
    cx = sum(p[0] for p in quad) / 4.0
    cy = sum(p[1] for p in quad) / 4.0
    lab._prev_pose["sensitivity"] = [
        {"camera": [cx - 180, cy - 90], "peak": 86.0},
        {"camera": [cx + 160, cy - 70], "peak": 31.0},
        {"camera": [cx - 150, cy + 100], "peak": 64.0},
        {"camera": [cx + 170, cy + 80], "peak": 16.0},
        {"camera": [cx, cy], "peak": 52.0},
    ]
    # SEED A SAVED MAPPING FOR THE SIMULATED SCREEN, so the fast launch
    # path is the one under test.
    #
    # Without this the saved calibration is for a 3840x2160 panel, the
    # simulated one is 800x800, the mapping is rejected on load and the
    # full eighteen-marker measurement runs every time -- so the harness
    # was timing the SLOW path and could never see whether the two-marker
    # check saves anything. It is built from the fake camera's own optics,
    # so it is exactly right, which is the case the check must confirm.
    #
    # The slow path still gets measured: run 1 deletes this and pays for
    # the full measurement, which is what a first-ever launch does.
    if seed_mapping:
        # Built from the PRIMARY SCREEN, not from the target window.
        #
        # The window does not exist yet at this point -- it opens later in
        # the chain -- so the first version read a rect of None, skipped the
        # seeding behind an `if`, and left the harness silently measuring
        # the slow path it was written to stop measuring. Every run paid for
        # eighteen markers and the two-marker check never ran once.
        #
        # A setup step that quietly does nothing is worse than one that
        # fails: the suite stayed green and the thing under test was absent.
        import cv2 as _cv2
        import numpy as _np
        from lsa.calibrate.mapping import Mapping as _Map
        geo = app.primaryScreen().geometry()
        sw, sh = geo.width(), geo.height()
        quad = _np.float32(fake_camera.Optics().quad)
        scr = _np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
        lab.mapping = _Map(homography=_cv2.getPerspectiveTransform(quad, scr),
                           screen_size=(sw, sh))
        # CARRY THE TUNED DOT RADIUS FORWARD, as his file does.
        #
        # The self test tunes the emulated dot until it images at the size a
        # real laser does, which costs four flashes and about four seconds,
        # and it is saved with the mapping so the next launch confirms it
        # with one flash. Seeding a fresh Mapping every run threw that away,
        # so the optimisation could not be exercised here at all -- and the
        # harness reported, correctly, no improvement whatsoever.
        #
        # On his bench the file persists between launches. Modelling that is
        # the point of this harness; a stand-in that cannot reach the state
        # the real thing reaches excuses every consumer of it (RULES C12).
        prior = _Map.load(Path(os.environ["LSA_CALIBRATION"]))
        if prior is not None and getattr(prior, "dot_radius", 0.0):
            lab.mapping.dot_radius = float(prior.dot_radius)
        lab._mapping_measured_now = False
        assert lab.mapping.screen_size == (sw, sh)
    else:
        # A FIRST-EVER LAUNCH HAS NO FILE, AND THAT IS WHAT THIS RUN IS.
        #
        # `seed_mapping=False` only skipped the in-memory seeding above.
        # The isolated calibration.json written by an EARLIER invocation
        # of this harness stayed on disk, so the app loaded it, took the
        # two-marker fast path, and the eighteen-marker measurement this
        # run exists to cover never ran at all.
        #
        # The call site has claimed "run 1 deletes this and pays for the
        # full measurement" since it was written. That was true only while
        # `Mapping.save` was neutered and there was never a file to
        # delete; the moment the save started working the comment became a
        # false statement about the code beside it, and the invariant
        # caught it on the very next run. A comment is not a step.
        #
        # Guarded to the isolated tree, like the save itself: no
        # environment variable may make this reach his calibration.json
        # (RULES C11).
        cal = Path(os.environ["LSA_CALIBRATION"]).resolve()
        safe = (Path(__file__).resolve().parent / "_out").resolve()
        assert safe in cal.parents, (
            f"refusing to delete {cal}: it is outside {safe}. His "
            f"calibration.json holds a mapping measured from eighteen "
            f"photographs plus the sensitivity learned from his real "
            f"trigger pulls.")
        cal.unlink(missing_ok=True)
        lab.mapping = None
        # ASSERT THE EFFECT, NOT THE ATTEMPT (RULES C1). The whole fault
        # above was a step that quietly did nothing.
        assert not cal.exists() and lab.mapping is None

    lab.auto_target = True
    lab.auto_patcal = True
    lab.auto_selftest = True
    lab._selftest_modes = False        # the modes have their own suite
    lab._connect()

    # WATCH THE THINGS THAT HAVE GONE WRONG BEFORE.
    #
    # Seven regressions were introduced in one day, every one of them
    # correct in isolation and wrong in combination, because this app is a
    # set of routines sharing one screen, one camera and one detector.
    # Each had a test for the FEATURE and none for its interaction with
    # everything already there -- which is how "all tests green" kept
    # coexisting with an app that did not work.
    #
    # So each one becomes an invariant of a whole launch, checked here,
    # where the whole launch actually happens.
    seen = {"floors": [], "probe": None, "t0": time.perf_counter()}

    _floor = lab._on_noise_measured

    seen["floor_frames"] = []
    seen["floor_states"] = []

    def floor_spy(peak, p99, frames):
        # The state is taken BEFORE the app sees the floor: handling it can
        # judge the gain map and WITHDRAW it, and a floor that was measured
        # with the map must be judged against an empty screen with the map,
        # not against the state the handler leaves behind. Judging every floor
        # against the state the launch ENDED in made a legitimate withdrawal
        # read as "something the app draws is visible to the detector" -- a
        # check naming the wrong cause.
        cam_now = getattr(lab, "_fake_camera", None)
        seen["floor_states"].append(
            cam_now.noise_state() if cam_now is not None else None)
        _floor(peak, p99, frames)
        seen["floors"].append(float(peak))
        seen["floor_frames"].append(int(frames))
    lab._on_noise_measured = floor_spy

    # The probe is a Director routine now, not a chain of methods to hook.
    # Its measurements are left on the lab as a record, and the Director
    # keeps the rest: what ran, in what order, whether anything overlapped,
    # and how many control writes it cost. That is a better instrument than
    # the method spy it replaces, because it also sees routines this harness
    # was never taught the names of.
    lab._probe_results = []

    done = {"v": False}
    _orig = lab._publish_verdict

    def spy(*a, **k):
        _orig(*a, **k)
        done["v"] = True
    lab._publish_verdict = spy

    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.start(int(seconds * 1000))
    while deadline.isActive() and not done["v"]:
        app.processEvents()
    seen["seconds"] = time.perf_counter() - seen["t0"]

    # An overlay that outlives its deadline is the fault that hid his
    # bullet holes: the verdict board paints a 59% veil over the whole
    # screen, and a clock mismatch meant it never cleared. Keep the event
    # loop turning past the board's own timeout and then look.
    board = lab.scene.get("verdict")
    if board is not None and done["v"]:
        until = time.perf_counter() + float(getattr(board, "READY_SECONDS",
                                                    4.0)) + 1.5
        while time.perf_counter() < until:
            app.processEvents()
        seen["board_stuck"] = bool(board.visible and board.verdict == "ready")
    else:
        seen["board_stuck"] = False

    # IS THERE ANYTHING ON THE SCREEN AT ALL?
    #
    # 2026-10-16: the two-marker check left a full-screen BLACK cover up,
    # so his display was blank for a whole launch -- no band, no verdict,
    # no holes -- and the self test photographed its own red dots through
    # it. HE saw it first: "I see a completely black screen."
    #
    # Every number in this harness and in the live log read healthy. The
    # chain reached a verdict, the mapping was confirmed, the floor was
    # measured. Nothing asked the one question a person asks in a second,
    # so this does: grab the window and count lit pixels.
    seen["screen_lit"] = _screen_lit(lab)

    seen["probe"] = list(getattr(lab, "_probe_results", []) or [])

    # THE REFERENCE A FLOOR IS JUDGED AGAINST: what this fake camera reports
    # for an EMPTY screen at the exposure the launch ended at, through the
    # same detector, parameters and gain map the app left set. A constant
    # cannot express "the floor did not climb because of something drawn"
    # across exposures; a reference taken at the same exposure with nothing
    # drawn can.
    #
    # PER FLOOR, in the state THAT floor was measured under (exposure, sensor
    # gain, gain map, detector parameters), recorded by `floor_spy`. Distinct
    # states are measured once. The alternative -- judging only the final floor
    # and asserting that no withdrawal happened -- was rejected: it would leave
    # a floor poisoned mid-launch (the verdict board did exactly that) unjudged
    # whenever a later, clean floor followed it.
    cam_ = getattr(lab, "_fake_camera", None)
    if cam_ is not None:
        seen["final_exposure"] = float(lab._shooting_exposure)
        cache = {}
        refs = []
        for st in seen["floor_states"]:
            if st is None:
                refs.append(None)
                continue
            key = (st["exposure"], st["sensor_gain"],
                   id(st["gain_map"]), id(st["params"]))
            if key not in cache:
                cache[key] = cam_.blank_floor(state=st)
            refs.append(cache[key])
        seen["floor_refs"] = refs
        seen["floor_with_map"] = [st is not None and st["gain_map"] is not None
                                  for st in seen["floor_states"]]
    # The Director's own books. `overlaps` is the strongest invariant in
    # this file: it records, as it happens, any delivery that reached a
    # routine which was not waiting for it, any still whose tag did not
    # match the outstanding one, and any attempt to resume a routine that
    # had been stood down. On a clean launch it must be empty.
    seen["director"] = {
        "history": list(lab.director.history),
        "overlaps": list(lab.director.overlaps),
        "refusals": list(lab.director.refusals),
        "writes": int(lab.director.control_writes),
        "skipped": int(lab.director.control_writes_skipped),
    }

    cam = getattr(lab, "_fake_camera", None)
    return lab, cam, done["v"], seen


def _screen_lit(lab) -> float:
    """What fraction of the target screen has any ink on it. -1 if no window.

    Coarse on purpose: this asks "is the shooter looking at anything",
    not "what does it say".
    """
    try:
        win = lab.target_window
        if win is None:
            return -1.0
        shot = win.grab()
        if shot.isNull():
            return -1.0
        img = shot.toImage()
        step = max(1, img.width() // 120)
        lit = total = 0
        for y in range(0, img.height(), step):
            for x in range(0, img.width(), step):
                total += 1
                c = img.pixelColor(x, y)
                if max(c.red(), c.green(), c.blue()) > 12:
                    lit += 1
        return (100.0 * lit / total) if total else -1.0
    except Exception:
        return -1.0


def _bar_ratio(r) -> tuple:
    """(worst ratio, floor, reference) for one launch's summary.

    Each floor is compared with the empty-screen floor measured in ITS OWN
    state, as a ratio of the bars the detector would use for them.
    """
    from lsa import exposure as expo
    worst = (0.0, 0.0, 0.0)
    for fl, ref in zip(r["floors"], r["floor_refs"]):
        if ref is None:
            continue
        ratio = expo.min_peak_for(fl) / expo.min_peak_for(ref)
        if ratio > worst[0]:
            worst = (ratio, fl, ref)
    return worst


def summarise(cam, lab, reached, seen=None):
    """What this launch did, in the terms the gate judges a real one by."""
    report = getattr(lab, "_last_report", None)
    held = getattr(lab.mapping, "held_out", None) or {}
    seen = seen or {}
    floors = seen.get("floors") or []
    probe = seen.get("probe")
    return {
        "seconds": seen.get("seconds"),
        "worst_floor": (max(floors) if floors else None),
        "floors": list(floors),
        "floor_frames": list(seen.get("floor_frames") or []),
        "floor_refs": list(seen.get("floor_refs") or []),
        "floor_with_map": list(seen.get("floor_with_map") or []),
        "final_exposure": seen.get("final_exposure"),
        "board_stuck": seen.get("board_stuck"),
        "screen_lit": seen.get("screen_lit"),
        "probe_blind": (bool(probe) and
                        max((pr.dot_peak for pr in probe), default=0) <= 0),
        "reached": reached,
        # Both oracles, concatenated: the camera's (two operations at once)
        # and the Director's (a routine handed something it never asked
        # for). A run is only clean if neither has anything to say.
        "overlaps": (list(cam.overlaps) if cam else ["no camera"])
                    + list((seen.get("director") or {}).get("overlaps", [])),
        "refusals": list((seen.get("director") or {}).get("refusals", [])),
        "writes": cam._stats.control_writes if cam else -1,
        "deduped": int((seen.get("director") or {}).get("skipped", 0)),
        # Whether the two-marker check confirmed the saved mapping instead
        # of remeasuring it. On that path there is no FRESH held-out error,
        # because nothing was fitted -- and that is the fast path working,
        # not a run that failed to calibrate. His constraint is the reason
        # it exists: "launch should be as fast as possible, especially if
        # the camera did not move from the previous launch."
        "check_confirmed": bool(getattr(lab, "_check_confirmed", False)),
        "has_mapping": lab.mapping is not None,
        "verdict": (report.verdict.value if report else "none"),
        "held_out_mm": held.get("median_mm"),
    }


def _floor_instrument(app) -> None:
    """The fake camera's noise floor is the app's statistic (plan 02, 2).

    Cheap -- a few seconds, before any launch -- because everything the
    launches report about the floor rests on this instrument being the real
    one. It used to take ONE raw frame, report its whole-frame chroma, and
    claim `seconds * 60` frames it never looked at: a different statistic from
    the app's (the maximum of the detector's own score over a window), so the
    gain-map judge compared a number with itself and a lie sat in the log
    (RULES C3, C12).
    """
    import numpy as np
    from PySide6.QtGui import QImage
    from lsa.camera.capture import CaptureConfig, GrouperConfig
    from lsa.detect.laser import DetectorParams

    SW = 800
    noise_rng = np.random.default_rng(7)

    def screen(noisy: bool) -> "QImage":
        """800x800 black screen; optionally a box of per-frame red noise."""
        a = np.zeros((SW, SW, 4), np.uint8)
        a[..., 3] = 255
        if noisy:
            # Fresh every call, so the detector's background model cannot
            # absorb it the way it absorbs a static grey square.
            a[300:500, 300:500, 0] = noise_rng.integers(
                0, 61, (200, 200)).astype(np.uint8)
        return QImage(a.tobytes(), SW, SW, 4 * SW, QImage.Format_RGBA8888).copy()

    state = {"noisy": False}

    def make():
        cam = fake_camera.FakeCapture(CaptureConfig(), DetectorParams(),
                                      GrouperConfig())
        cam.view_provider = lambda: screen(state["noisy"])
        cam.set_control("gain", 0.0)            # the shooting profile
        cam.set_control("exposure", -5.0)
        return cam

    def settle(cam, frames=30):
        """Let the detector's background model absorb the scene.

        The model is reset by a gain-map change and starts empty; a scene
        that has just appeared scores as a step against it. The app measures
        a floor long after both, so the instrument must too (bounded).
        """
        until = time.perf_counter() + 20.0
        target = cam._stats.frames + frames
        while cam._stats.frames < target and time.perf_counter() < until:
            app.processEvents()

    def measure(cam, seconds=1.0):
        got = {}

        def grab(p, q, n):
            got.update(p=p, q=q, n=n)
        cam.noise_measured.connect(grab)
        frames0 = cam._stats.frames
        cam.start_noise_calibration(seconds)
        until = time.perf_counter() + seconds + 10.0     # bounded (RULES D4)
        while "n" not in got and time.perf_counter() < until:
            app.processEvents()
        got["ticks"] = cam._stats.frames - frames0
        # Disconnect: a still-connected earlier receiver is overwritten by
        # the NEXT measurement, which made "before" and "after" the same
        # number the first time this was written.
        cam.noise_measured.disconnect(grab)
        return got

    print("\n0. the fake camera measures its floor the way the app does")
    cam = make()
    cam.start()
    try:
        state["noisy"] = False
        flat = measure(cam)
        check("the reported frame count is the number of frames that ran",
              "n" in flat and flat["n"] == flat["ticks"] and flat["n"] > 0,
              f"reported {flat.get('n')}, frames actually processed "
              f"{flat['ticks']} (the old body reported int(seconds*60) = 60 "
              f"having looked at one)")
        check("and it is a count of real frames, not a paraphrase of the rate",
              flat.get("n", 0) <= 63,
              f"{flat.get('n')} frames in 1s cannot exceed the 16ms tick")

        # A bright NOISE region, and a gain map that amplifies it fourfold:
        # the floor must move, because the app's floor is measured on the
        # NORMALISED score. On the old raw-chroma body it could not.
        state["noisy"] = True
        settle(cam)
        before = measure(cam)
        gmap = np.ones((fake_camera.FRAME_H, fake_camera.FRAME_W), np.float32)
        gmap[200:520, 300:800] = 0.25
        cam.set_gain_map(gmap)
        settle(cam)
        after = measure(cam)
        print(f"     floor over a noisy region: {before['p']:.0f} flat, "
              f"{after['p']:.0f} with a 0.25 gain map "
              f"({before['n']} and {after['n']} frames)")
        check("the region is visible to the floor at all", before["p"] >= 5,
              f"{before['p']:.0f} -- the check below would pass vacuously")
        check("a gain map of 0.25 under it moves the measured floor",
              after["p"] >= 2.0 * before["p"],
              f"{before['p']:.0f} -> {after['p']:.0f}; the app's floor is "
              f"taken on the normalised score, so the gain-map judge has "
              f"something to judge")
    finally:
        cam.stop()

    # The reference: same pipeline, empty screen, own random stream.
    cam = make()
    a = cam.blank_floor(-4.0)
    b = cam.blank_floor(-4.0)
    check("the empty-screen reference is repeatable and perturbs nothing",
          a == b, f"{a} then {b}")
    print(f"     empty-screen reference: -5 -> {cam.blank_floor(-5.0):.0f}, "
          f"-4 -> {a:.0f}, -3 -> {cam.blank_floor(-3.0):.0f}")

    # A FLOOR IS JUDGED IN THE STATE IT WAS MEASURED IN, not the state the
    # launch ended in. The app's gain-map judge may legitimately WITHDRAW the
    # map ("always safe"), after which the final state has no map and its
    # empty-screen floor is far lower than the with-map floor the app measured
    # a moment earlier -- so a reference taken at the end reads a legitimate
    # withdrawal as "something the app draws is visible to the detector": a
    # check naming the wrong cause. The harness's own launches never reached a
    # with-map floor in the runs that measured this, so the property is proved
    # here, on a fixture, rather than hoped for from a launch.
    cam = make()
    gmap = np.ones((fake_camera.FRAME_H, fake_camera.FRAME_W), np.float32)
    gmap[200:520, 300:800] = 0.25
    cam.set_gain_map(gmap)
    with_map = cam.noise_state()              # as measured, map in place
    cam.set_gain_map(None)                    # the judge withdraws it
    flat_now = cam.blank_floor(state=cam.noise_state())
    snapshot = cam.blank_floor(state=with_map)
    again = cam.blank_floor(state=with_map)
    print(f"     empty-screen floor: {flat_now:.0f} with the map withdrawn, "
          f"{snapshot:.0f} in the with-map state it was measured in")
    check("a with-map floor keeps its own reference after the map is withdrawn",
          snapshot >= 2.0 * flat_now and snapshot > flat_now + 2.0
          and snapshot == again,
          f"{snapshot} (snapshot) vs {flat_now} (state now): judging a floor "
          f"measured with the map against the withdrawn state would blame the "
          f"app's drawing for the map's own amplification")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    print("=== cold starts against a camera that is not there ===")
    owned_before = OwnedFiles()
    _floor_instrument(app)

    # ------------------------------------------------------------------ 1 ---
    print("\n1. one clean cold start, no jitter")
    # No seeded mapping, so this is the run that exercises the EIGHTEEN
    # MARKER measurement. Section 2 seeds one, so those runs take the
    # two-marker fast path. Both paths have to be covered, and keeping this
    # summary is how the check below can tell: the first version of it
    # looked only at section 2 and failed the moment the fast path started
    # working, reporting "nothing tests the measurement any more" while
    # section 1 was measuring it three lines earlier.
    lab, cam, reached, seen = run_cold_start(app, (0, 0), seed=1,
                                             seed_mapping=False)
    first = summarise(cam, lab, reached, seen)
    check("the chain reaches a verdict", reached,
          "without one, an unattended launch hangs to its deadline -- which "
          "is exactly what gate launch 1 did")
    if cam is not None:
        check("no two camera operations overlapped", not cam.overlaps,
              "; ".join(cam.overlaps[:3]))
        check("and it stayed inside the control-write budget",
              cam._stats.control_writes <= 24,
              f"{cam._stats.control_writes} writes")
    # THE DIRECTOR'S OWN ORACLE, which is strictly stronger than the one
    # above. The camera's oracle sees two camera operations overlapping;
    # this sees a measurement delivered to a routine that was not waiting
    # for it, a still whose tag did not match the outstanding one, and any
    # attempt to resume a routine that had been stood down. Those are the
    # faults that cost the live gate, and they are invisible at the camera.
    dr = seen.get("director") or {}
    check("no routine was handed something it had not asked for",
          not dr.get("overlaps"),
          "; ".join(dr.get("overlaps", [])[:3]))
    check("and nothing was refused for want of a free slot",
          not dr.get("refusals"),
          f"{dr.get('refusals')} -- a refusal on a clean launch means two "
          f"routines were asked for at once, which is a chain bug even "
          f"though the Director handled it")
    print(f"     director: {' '.join(f'{n}={o}' for n, o, _ms in dr.get('history', []))}"
          f"   writes={dr.get('writes')} deduped={dr.get('skipped')}")
    try:
        lab.close()
    except Exception:
        pass

    # ------------------------------------------------------------------ 2 ---
    # THE POINT OF THE WHOLE HARNESS.
    #
    # Races only appear at particular interleavings, so a fixed-timing
    # simulation would pass for ever while hardware failed one launch in
    # two -- which is precisely what the live gate measured: 12/12, 0/12,
    # 12/12, 1/12, 12/12 on identical code. Every asynchronous reply is
    # therefore delayed by a random amount, and each run uses a different
    # seed, so a handful of runs here covers orderings a real machine
    # would take weeks to produce.
    #
    # The assertion is about ORDER, not outcome. A launch can give the
    # right answer and still have been wrong -- two measurements can
    # overlap and happen to agree. Outcome is checked too, but overlap is
    # what fails the build.
    runs = int(os.environ.get("COLDSTART_RUNS", "6"))
    print(f"\n2. {runs} cold starts with the timing shaken")
    results = []
    for i in range(runs):
        lab, cam, reached, seen = run_cold_start(app, (0, 320), seed=100 + i)
        results.append(summarise(cam, lab, reached, seen))
        r = results[-1]
        print(f"     run {i + 1}: verdict={r['verdict']:9s} "
              f"writes={r['writes']:3d} (-{r['deduped']:2d}) "
              f"calib={r['held_out_mm'] if r['held_out_mm'] is not None else -1:.2f}mm "
              f"floor={r['worst_floor'] if r['worst_floor'] is not None else -1:.0f} "
              f"@{r['final_exposure'] if r['final_exposure'] is not None else 0:g} "
              f"floors={[int(f) for f in r['floors']]} "
              f"refs={[None if x is None else int(x) for x in r['floor_refs']]} "
              f"map={['Y' if m else 'n' for m in r['floor_with_map']]} "
              f"ratio={_bar_ratio(r)[0]:.2f} "
              f"frames={r['floor_frames']} "
              f"{'OVERLAP: ' + r['overlaps'][0] if r['overlaps'] else ('REFUSED: ' + r['refusals'][0][0] if r['refusals'] else 'clean')}")
        try:
            lab.close()
        except Exception:
            pass

    # A REFUSAL ON A CLEAN LAUNCH IS A CHAIN BUG.
    #
    # The Director refusing to start a routine means two were asked for at
    # once. It handles that correctly -- that is the whole point of one run
    # slot -- but on an unattended cold start with nothing else happening,
    # nobody should be asking twice. The old code's answer to this
    # situation was a 700-900ms self-rescheduling retry, which hid it: the
    # second caller simply arrived later and the log said nothing.
    #
    # This is the invariant that would have caught `auto_patcal` firing a
    # second pattern calibration on top of the repeat scheduler's -- the
    # fault that cost two sessions and read, from the firing line, as "the
    # camera cannot see the far side of the panel".
    contended = [i + 1 for i, r in enumerate(results) if r["refusals"]]
    check("no run asked for two routines at once", not contended,
          f"runs {contended}: "
          + "; ".join(f"{w} wanted it while {h} had it"
                      for w, h in results[contended[0] - 1]["refusals"][:2])
          if contended else "")

    racy = [i + 1 for i, r in enumerate(results) if r["overlaps"]]
    check("no run let two camera operations overlap", not racy,
          f"runs {racy} raced -- "
          + "; ".join(results[racy[0] - 1]["overlaps"][:2]) if racy else "")
    stalled = [i + 1 for i, r in enumerate(results) if not r["reached"]]
    check("every run reached a verdict", not stalled,
          f"runs {stalled} never got there; an unattended launch then "
          f"hangs to its deadline with nothing on screen")
    overbudget = [i + 1 for i, r in enumerate(results) if r["writes"] > 24]
    check("every run stayed inside the control-write budget", not overbudget,
          f"runs {overbudget} exceeded it -- control writes are what hang "
          f"this camera")
    # EVERY RUN ENDS WITH A MAPPING IT HAS REASON TO TRUST.
    #
    # Two ways to earn that, and the second one is the point of the fast
    # path: measure it from scratch and hold out six markers to check the
    # fit, OR confirm the saved one with two markers at opposite corners.
    #
    # The first version of this check demanded a FRESH held-out error on
    # every run, which meant it failed the moment the fast path started
    # working -- "0/2 calibrated" on two runs that both ended READY with a
    # mapping confirmed to 1.5px. A test that fails when the code starts
    # doing the right thing is worse than no test, because the obvious
    # response is to make the code go back.
    #
    # It is still an error for a run to end with no mapping at all, or
    # with a fit worse than 2mm, and it is still an error for EVERY run to
    # take the fast path -- that would mean the measuring path is never
    # exercised here at all (RULES C1: a check that cannot distinguish two
    # situations is not a check).
    fitted = [r["held_out_mm"] for r in results if r["held_out_mm"] is not None]
    confirmed = [i + 1 for i, r in enumerate(results) if r["check_confirmed"]]
    unmapped = [i + 1 for i, r in enumerate(results) if not r["has_mapping"]]
    check("every run ends with a mapping", not unmapped,
          f"runs {unmapped} have none -- a detection then has nowhere to be "
          f"drawn, which from the firing line is indistinguishable from a "
          f"detector that does not work")
    check("every fresh fit is under 2mm held out",
          all(m < 2.0 for m in fitted),
          f"worst {max(fitted) if fitted else -1:.2f}mm over "
          f"{len(fitted)} fit(s)")
    print(f"     {len(fitted)} run(s) measured the mapping, "
          f"{len(confirmed)} confirmed the saved one with two markers")
    # BOTH PATHS, across the whole harness run. Section 1 has no saved
    # mapping so it measures; section 2 has one so it confirms. Asserting
    # it over section 2 alone is what made this check fail on a green run.
    measured_somewhere = bool(fitted) or first["held_out_mm"] is not None
    check("the eighteen-marker measurement is exercised somewhere",
          measured_somewhere,
          "no run measured a mapping from scratch, so the slow path is "
          "untested -- section 1 exists to cover it and runs with no saved "
          "mapping")
    check("and the two-marker fast path is exercised somewhere",
          bool(confirmed) or first["check_confirmed"],
          "no run confirmed a saved mapping, so the three-second path that "
          "makes startup bearable is untested")
    if first["held_out_mm"] is not None:
        check("the measured mapping is under 2mm held out",
              first["held_out_mm"] < 2.0,
              f"{first['held_out_mm']:.2f}mm from the 18-marker fit")
    ready = [r for r in results if r["verdict"] == "ready"]
    check("and every run came out READY", len(ready) == len(results),
          f"{len(ready)}/{len(results)} -- the live gate managed 2 of 5")

    # -- the three invariants that each past regression violated ---------

    # 1. THE NOISE FLOOR STAYS NEAR ZERO.
    #
    # The startup verdict board was painted in amber at red excess 24 --
    # legal under the house rule -- and took the measured floor from 0-2
    # to 16-18, which pushed min_peak above the app's OWN test dots. The
    # next self test detected 1 of 12, produced a worse verdict, and that
    # verdict stayed on screen and did it again. A floor that climbs is
    # the app looking at something it drew.
    #
    # JUDGED AGAINST THE FAKE'S OWN EMPTY-SCREEN FLOOR at the exposure the
    # launch ended at, not against a constant. The old bar of 12 was derived
    # when his settings file said -4; the file moved to -3, the sensor noise
    # the fake models grows with exposure, and the bar began failing 1 to 3
    # launches in 7 for a reason that had nothing to do with anything drawn.
    # Every launch, including section 1's, is held to it.
    launches = [first] + results
    judged = [(i, r) for i, r in enumerate(launches)
              if r["floors"] and len(r["floor_refs"]) == len(r["floors"])
              and all(x is not None for x in r["floor_refs"])]
    ratios = [(i, _bar_ratio(r)) for i, r in judged]
    print(f"     bar ratio (worst floor / its own-state empty-screen floor), "
          f"per launch (section 1 first): "
          f"{[round(x[0], 2) for _i, x in ratios]}  allowed {FLOOR_BAR_RATIO_MAX:g}")
    check("every launch measured a floor and a reference to judge it by",
          len(judged) == len(launches),
          f"{len(judged)} of {len(launches)} launches had a reference for "
          f"every floor -- a check that skips what it cannot read passes "
          f"anything it cannot see")
    noisy = [(i, round(x[0], 2), x[1], x[2]) for i, x in ratios
             if x[0] > FLOOR_BAR_RATIO_MAX]
    check(f"no floor raised the detection bar more than {FLOOR_BAR_RATIO_MAX:g}x "
          f"above an empty screen's, measured in the same state", not noisy,
          f"(launch index, ratio, floor, empty-screen floor) {noisy} -- "
          f"something the app draws that MOVES or APPEARS is visible to the "
          f"detector. This check is blind to a STATIC element by design: the "
          f"background model absorbs it and it measures zero excess, as in the "
          f"app. The lit-area budget (RULES B1b) and static furniture are "
          f"owned by test_ui_smoke, not here.")

    # 2. NO OVERLAY OUTLIVES ITS DEADLINE.
    #
    # The verdict board paints a 59% black veil over the WHOLE screen. A
    # clock mismatch -- deadline set with time.time(), tested against the
    # scene's perf_counter() -- meant READY never cleared, so every
    # bullet hole he shot was drawn at 41% brightness under a permanent
    # veil. He reported no holes at all, and I blamed the arcade modes.
    stuck = [i + 1 for i, r in enumerate(results) if r["board_stuck"]]
    check("the READY banner cleared itself every time", not stuck,
          f"runs {stuck} left it up -- it veils the screen and hides the "
          f"holes underneath it")

    # 2b. THE SHOOTER CAN SEE SOMETHING.
    #
    # The cheapest question in this file and the last one to be asked.
    # On 2026-10-16 a routine left a full-screen black cover up and the
    # display stayed blank for the entire session: no prompt band, no
    # verdict, no bullet holes, and the self test's own dots photographed
    # through it -- 2 of 12 detected, a 145.7mm "mapping error" measured
    # against a cover, four lines after the two-marker check measured that
    # same mapping at 2.1mm. Two numbers that cannot both be true, and
    # nothing in six launches of this harness noticed, because every check
    # here reads the log and none of them looked at the screen.
    #
    # It is also what the harness itself was failing on, silently: its fake
    # camera photographs that same window, so "only 0 of 12 markers were
    # visible" was this.
    # THE APP'S OWN NUMBER, imported rather than retyped, so the two
    # cannot drift into disagreeing about whether he can see anything.
    from lsa.tools.camera_lab import CameraLab as _CLmin
    _min_lit = float(_CLmin.SCREEN_LIT_MIN_PCT)
    blank = [i + 1 for i, r in enumerate(results)
             if (r.get("screen_lit") or 0) < _min_lit]
    check("the shooter can see something on the target screen", not blank,
          f"runs {blank} ended with a blank screen -- he reported this one "
          f"himself before any instrument did")

    # 3. A REPORTED MEASUREMENT HAS SIGNAL IN IT.
    #
    # The exposure probe climbed all seven rungs of its brightness ladder
    # reading 0 each time, spent seven seconds doing it, then printed a
    # table of zeroes as a measurement -- "0% of his shots would register"
    # -- and chose an exposure from it. A rule that cannot see its input
    # must say so, not average it.
    blind = [i + 1 for i, r in enumerate(results) if r["probe_blind"]]
    check("no run reported an exposure measured on nothing", not blind,
          f"runs {blind} chose an exposure from a table of zeroes")

    # 4. AND IT HAS TO BE QUICK.
    #
    # "No one will have any patience for the app that takes that long to
    #  start and initialize." A launch crept to 82 seconds without anyone
    #  noticing, because nothing measured it. The simulated chain is not
    #  the same as the bench -- its settling times are shortened -- so
    #  this is a REGRESSION bar, not a promise about the real app: it
    #  catches a new step quietly adding twenty seconds.
    # The jittered runs start from a saved mapping, as every launch after
    # the first does, so they take the fast path: two markers instead of
    # eighteen. 40s in simulated time, where the slow path measured 70.
    #
    # THE BUDGET IS A SUM, AND EVERY TERM IS WRITTEN DOWN (2026-10-22).
    # This bar was a bare 40.0 and went stale without anyone able to say so:
    # nothing recorded whether 40 was a measurement or a wish. Measured over
    # 689 harness launches in the cumulative log: fast-path medians of 35-40s
    # up to 2026-10-18 and a flat 40-41s from 2026-10-19 onward. The step is
    # one log line, "holding setup 2.0s so the welcome line can finish",
    # from `_begin_after_welcome` (commit e8e2c23, 2026-10-18), added at his
    # request so the spoken "Welcome to ScreenPlink" is not cut off. It is
    # deliberate, so it belongs in the budget, not in a failure.
    #
    #   BASE   40.0  the bar as it stood; launches ran ~38s against it
    #   HOLD    2.0  the welcome-line hold, measured (speech + SPLASH_EXTRA_S,
    #                capped at SPLASH_HOLD_CAP_S; the cap is checked below)
    #   MARGIN  1.0  slack for timer jitter; NOT for new work
    #   -----  ----
    #   BAR    43.0
    #
    # The harness uses HIS settle/floor times (C15), so nothing is shortened
    # here; the old "settling times are shortened" note was untrue.
    LAUNCH_BASE_S = 40.0
    LAUNCH_WELCOME_HOLD_S = 2.0
    LAUNCH_MARGIN_S = 1.0
    SECONDS_MAX = LAUNCH_BASE_S + LAUNCH_WELCOME_HOLD_S + LAUNCH_MARGIN_S
    slow = [(i + 1, round(r["seconds"] or 0)) for i, r in enumerate(results)
            if (r["seconds"] or 0) > SECONDS_MAX]
    check(f"every run reached a verdict within {SECONDS_MAX:.0f}s", not slow,
          f"{slow} -- over the {SECONDS_MAX:.0f}s budget = {LAUNCH_BASE_S:.0f}s "
          f"base + {LAUNCH_WELCOME_HOLD_S:.0f}s welcome-line hold "
          f"(_begin_after_welcome, 2026-10-18) + {LAUNCH_MARGIN_S:.0f}s margin; "
          f"each figure is seconds beyond the known {SECONDS_MAX:.0f}, so "
          f"something NEW has been added to the launch")
    took = [r["seconds"] for r in results if r["seconds"]]
    if took:
        print(f"       launch took {min(took):.0f}-{max(took):.0f}s "
              f"(simulated, on his own settle/floor times; budget "
              f"{SECONDS_MAX:.0f}s)")

    # THE HOLD MAY NOT GROW INTO THE MARGIN. The budget above carries the
    # welcome hold as 2s; `SPLASH_HOLD_CAP_S` is the only thing bounding it.
    # Read from the app (imported, not retyped, as SCREEN_LIT_MIN_PCT is
    # below), and pinned to the values the budget was derived from: raising
    # either is a decision to make the launch slower and must raise the bar
    # in the same change, with its reason.
    from lsa.tools.camera_lab import CameraLab as _CLhold
    _cap = float(_CLhold.SPLASH_HOLD_CAP_S)
    _extra = float(_CLhold.SPLASH_EXTRA_S)
    HOLD_CAP_REVIEWED_S = 4.5
    HOLD_EXTRA_REVIEWED_S = 1.0
    check("the welcome-line hold stays within the cap the launch budget "
          "was derived from",
          _cap <= HOLD_CAP_REVIEWED_S and _extra <= HOLD_EXTRA_REVIEWED_S,
          f"SPLASH_HOLD_CAP_S={_cap:g} (reviewed {HOLD_CAP_REVIEWED_S:g}), "
          f"SPLASH_EXTRA_S={_extra:g} (reviewed {HOLD_EXTRA_REVIEWED_S:g}) "
          f"-- a longer hold eats the {LAUNCH_MARGIN_S:.0f}s margin; raise "
          f"LAUNCH_WELCOME_HOLD_S and say why")


    # ------------------------------------------------------------------ 3 ---
    print("\n3. nothing the user owns was touched")
    changed = owned_before.changed()
    check("his calibration and settings are byte-for-byte unchanged",
          not changed,
          f"{changed} were written -- the simulated screen is 800x800, so a "
          f"saved mapping would overwrite one measured at 3840x2160 and "
          f"take the sensitivity learned from his real shots with it")

    print()
    if all(checks):
        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
