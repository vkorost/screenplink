"""The measurement routines, as straight-line code.

Each of these used to be a chain of `QTimer.singleShot` callbacks with its
local variables hoisted onto `CameraLab` as `_probe_queue`, `_probe_lit`,
`_pat_index` and some sixty other attributes. Written as generators driven
by `lsa.director.Director`, they read in the order they happen, their
cleanup lives in `finally`, and two of them cannot run at once because
there is one run slot to run in.

What that bought, measured against the record (see `CLAUDE.md` 2026-10-15):

  * The 1800ms hand-off at the end of the exposure probe is GONE. It
    existed because choosing an exposure triggered a noise measurement
    through a timer, so the next routine started before `_calibrating` had
    been raised and ran against the bar from the previous exposure. "A
    guard that is armed too late is not a guard" -- but the guard was only
    needed because the chain was implicit. Here the probe measures the
    floor itself, as its next line, and the routine that follows cannot
    start until the probe returns.
  * The brightness auto-range no longer re-enters its own step function,
    which is what re-wrote the same exposure on every rung of the ladder:
    seven redundant UVC transactions a launch, on a camera where one write
    has hung for 272 seconds. It is a `while` loop.
  * Every `return` through the middle of a routine used to be a place to
    forget to take the dot down, put the exposure back, or let the
    furniture return. There is one `finally` per routine instead.

A routine takes the lab and yields effects. It must not touch the capture
thread or a QTimer directly -- `tests/test_rules.py` fails the build if it
does, because that is the unserialised path every interleaving came
through.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

from . import exposure as expo
from .director import Call, Cancelled, Control, Floor, Job, Log, Paint, Sleep, Still

__all__ = ["exposure_probe", "probe_job", "self_test", "self_test_job",
           "mapping_check", "mapping_check_job", "mark_watch_check",
           "mark_watch_job"]


# ---------------------------------------------------------------------------
# The exposure probe
# ---------------------------------------------------------------------------
#
# Chooses the camera exposure by photographing a red dot the SCREEN draws,
# at a brightness anchored to his own measured shots, and predicting what
# fraction of his 575 logged trigger pulls would clear the bar at each
# candidate. It replaced a constant (`MEAN_CEILING = 40`) that judged
# exposure on whole-frame brightness -- which included a sunlit window, and
# stopped the camera down three stops, costing half a string.


def exposure_probe(lab: Any) -> Generator:
    """Measure several exposures against the screen's own dot, and pick one.

    Yields: Control, Sleep, Still, Floor. Returns the chosen exposure, or
    None if nothing could be measured.
    """
    base = lab._shooting_exposure
    if base is None:
        base = expo.GUN_ANCHOR_EXPOSURE

    wanted: list[float] = []
    for step in lab.PROBE_STEPS:
        v = max(-11.0, min(-3.0, float(base) + step))
        if v not in wanted:
            wanted.append(v)
    # The anchor exposure must be in the list or every ratio is relative to
    # an arbitrary point: it is the one his 575 logged pulls were measured
    # at, which is what makes the prediction absolute rather than ordered.
    if expo.GUN_ANCHOR_EXPOSURE not in wanted:
        wanted.append(expo.GUN_ANCHOR_EXPOSURE)
    # THE ANCHOR IS MEASURED FIRST (2026-10-17). The dot's brightness is
    # auto-ranged on the first candidate and held for the rest, and the
    # exposure that gets chosen is the anchor on essentially every launch --
    # so it was being ranged at the STARTING exposure (the brightest
    # candidate) and judged at another. A dot ranged where the sensor is
    # generous comes out thin where it is judged: the harness line "dot 10
    # against a bar of 14" at -5. Ranging at the anchor puts the dot where
    # the judgement is made. The write count is unchanged (one per candidate;
    # a write of the value already set is skipped).
    queue = sorted(wanted, reverse=True)
    queue.sort(key=lambda v: 0 if abs(v - expo.GUN_ANCHOR_EXPOSURE) < 1e-6 else 1)

    # Forgotten at the start of every probe: a reading from an earlier launch
    # of the probe must never condemn (or excuse) this one.
    lab._probe_dot_reading = None
    radius = lab._selftest_dot_radius(11.0)
    ladder = list(lab.PROBE_BRIGHTNESS_LADDER)
    brightness: Optional[float] = None       # auto-ranged once, then held
    results: list[expo.Probe] = []
    # ONE SPOT FOR EVERY CANDIDATE, and it is the one that decides whether
    # his string registers.
    #
    # It used to walk a different spot per candidate -- centre, then
    # (0.3,0.4), then (0.7,0.6) -- so each exposure was photographed
    # somewhere else on a panel whose emitted brightness, as this camera
    # sees it, varies 3.6x between the near and far side. The dot is
    # auto-ranged OUT of saturation on purpose, so that falloff lands at
    # full strength on the measurement. The probe was comparing positions
    # and calling it exposure.
    #
    # It shows in his own tables: in four of five launches the SECOND
    # candidate read brightest whatever its exposure (113 against 27 and
    # 59; 61 against 25 and 25; 111 against 93 and 54), and no table was
    # monotonic in exposure -- at 10:46 the -3 dot read 25 and the -5 dot
    # read 25 with -4 at 61 between them, which one dot in one place
    # cannot do. Over twenty minutes in an unchanged room the chosen
    # exposure went -4, -4, -3, -5, -6. The -6 is the launch where the
    # microphone heard ten trigger pulls the camera never saw.
    #
    # The spot is the WEAKEST region the mapping knows about, because that
    # is where his shots are lost -- an exposure chosen on the bright half
    # protects the half that was never in danger. Falls back to the centre
    # when nothing has been learned yet.
    spot = lab._probe_spot()
    # THE OTHER END (2026-10-17). The weakest region is the objective; the
    # strongest is a CONSTRAINT: no exposure may saturate it. See the long
    # comment in `lsa/exposure.py` for the measurement behind that.
    strong_spot, spot_note = lab._probe_strong_spot()
    if strong_spot is not None and (abs(strong_spot[0] - spot[0]) < 0.15
                                    and abs(strong_spot[1] - spot[1]) < 0.15):
        # Same place: there is no second end to measure.
        strong_spot = None
        spot_note = "the strong and weak ends coincide, so the near side was not measured"
    _has_map = bool(getattr(getattr(lab, "mapping", None), "sensitivity", None))
    yield Log(f"dot spots: weak end at ({spot[0]:.2f},{spot[1]:.2f}) of the "
              f"screen ("
              + ("the weakest region the saved sensitivity measured"
                 if _has_map else
                 "a GEOMETRIC GUESS, no sensitivity has been measured for "
                 "this camera position") + f"); strong end "
              + (f"at ({strong_spot[0]:.2f},{strong_spot[1]:.2f}) -- "
                 if strong_spot is not None else "NOT MEASURED -- ")
              + spot_note)

    yield Log(f"measuring the exposure by photographing the screen's own red "
              f"dot at {len(queue)} settings "
              f"({', '.join(f'{v:g}' for v in queue)}). "
              f"No shooting, and one control write each.")

    try:
        while queue:
            candidate = queue.pop(0)
            yield Control("exposure", candidate)

            fx, fy = spot
            level = brightness if brightness is not None else ladder[0]
            # BOTH ENDS IN ONE PHOTOGRAPH. The two spots are far apart in the
            # same frame, so one lit still and one dark still carry both
            # readings; the strong end used to cost a dot, a settle and a
            # photograph of its own at every candidate (about a second each,
            # three a launch). `_probe_measure_pair` reads each spot in its
            # own neighbourhood, because a whole-ROI maximum would return the
            # brighter dot twice.
            yield Call(lambda level=level: _show_dots(
                lab, spot, strong_spot, radius, level))
            # The dot goes up FIRST and the still is taken after the
            # settle, so the frame photographed is one the new exposure
            # actually produced.
            yield Sleep(lab.PROBE_SETTLE_MS)
            lit = yield Still("probe_lit")

            yield Call(lambda: _hide_dot(lab))
            yield Sleep(lab.PROBE_DOT_MS)
            dark = yield Still("probe_dark")

            if lit is None or dark is None:
                yield Log(f"    exposure {candidate:g}: no frame, skipped")
                continue

            peak, floor, mean, strong_peak = lab._probe_measure_pair(
                lit, dark, spot, strong_spot)
            if peak is None:
                yield Log(f"    exposure {candidate:g}: nothing measurable, skipped")
                continue

            # AUTO-RANGE, once, on the first candidate only, then held for
            # every remaining exposure so the readings are comparable.
            #
            # This was a re-entrant call into the step function, which is
            # why it re-wrote the same exposure on every rung. As a loop it
            # writes nothing: the exposure is already set, and only the
            # dot on the screen changes.
            if brightness is None:
                # "Measurable" is judged against THIS candidate's own bar as
                # well as the absolute floor: a dot at or under its bar
                # cannot be told from the noise it must clear, and the
                # prediction built on it is a guess (`exposure.own_margin`).
                def _wanted_peak(fl):
                    return max(lab.PROBE_PEAK_LO,
                               expo.RANGE_OVER_BAR * expo.SAFETY
                               * expo.min_peak_for(fl))
                while peak < _wanted_peak(floor) and len(ladder) > 1:
                    was = ladder.pop(0)
                    # JUMP, do not crawl. The dot's reading is linear in its
                    # brightness inside the sensor's linear range, so the
                    # reading says how far up the ladder the target is.
                    # Ranging at the ANCHOR (dimmer than the brightest
                    # candidate it used to be ranged at) needs more rungs,
                    # and each rung is a dot, a settle and two photographs:
                    # crawling cost the cold start three extra seconds
                    # (probe 10s -> 15s). A rung skipped is one the reading
                    # already ruled out; a jump that undershoots simply
                    # takes one more step.
                    _need = was * _wanted_peak(floor) / max(1.0, peak) * 1.1
                    while len(ladder) > 1 and ladder[0] < _need:
                        ladder.pop(0)
                    yield Log(f"    dot at {was:.2f} brightness read "
                              f"{peak:.0f} -- too faint to measure with "
                              f"(wants {_wanted_peak(floor):.0f}: "
                              f"{expo.RANGE_OVER_BAR:g}x this exposure's bar), "
                              f"trying {ladder[0]:.2f}")
                    yield Call(lambda: _show_dots(
                        lab, spot, strong_spot, radius, ladder[0]))
                    yield Sleep(lab.PROBE_DOT_MS)
                    lit = yield Still("probe_lit")
                    yield Call(lambda: _hide_dot(lab))
                    yield Sleep(lab.PROBE_DOT_MS)
                    dark = yield Still("probe_dark")
                    if lit is None or dark is None:
                        peak = None
                        break
                    peak, floor, mean, strong_peak = lab._probe_measure_pair(
                        lit, dark, spot, strong_spot)
                    if peak is None:
                        break

                if peak is None:
                    yield Log(f"    exposure {candidate:g}: the camera stopped "
                              f"delivering during the auto-range, skipped")
                    continue

                # RECORDED BEFORE ANY EARLY RETURN. This is the reading the
                # self-check judges (`selfcheck.camera_not_delivering`): the
                # anchor candidate's dot at the brightness it ranged to. Only
                # THIS reading is comparable to the 25-97 healthy band;
                # later candidates are darker exposures and read lower
                # legitimately. Every early return below used to throw it
                # away, which is how a reading of 5 sailed through to READY.
                lab._probe_dot_reading = float(peak)

                # NOTHING AT FULL RED MEANS THE DOT IS NOT GETTING THROUGH.
                #
                # Measured live: the ladder climbed all seven rungs reading
                # 0 each time, took seven seconds, then reported a table of
                # zeroes as a measurement -- "0% of his shots would
                # register" -- and chose an exposure from it anyway. A rule
                # that cannot see its input must say so, not average it.
                if peak < max(2.0, floor + 1.0):
                    yield Log(f"    the emulated dot is not reaching the "
                              f"camera (it reads {peak:.0f} at full red, "
                              f"against a background of {floor:.0f} peaking at "
                              f"{lab._probe_floor_at}). "
                              f"Something is covering the screen, or the "
                              f"camera is not on it. Keeping the exposure "
                              f"that is already set.")
                    return None

                brightness = ladder[0]
                yield Log(f"    emulating the shot at {brightness:.2f} of "
                          f"full red: it reads {peak:.0f} here, inside the "
                          f"sensor's linear range, so a change in exposure "
                          f"will actually show up")

            # (The strong end was read from the same photograph above.)
            results.append(expo.Probe(exposure=float(candidate),
                                      dot_peak=peak, floor=floor, mean=mean,
                                      strong_peak=strong_peak,
                                      brightness=float(brightness)))

        # -- choose ------------------------------------------------------
        #
        # Kept on the lab purely as a RECORD, for the GATE line and the
        # cold-start harness to read afterwards. Not state that anything
        # steps through: that is what `_probe_queue`, `_probe_lit` and
        # `_probe_exposure` were, and hoisting a routine's locals onto a
        # shared object is how two runs of one routine came to corrupt
        # each other.
        lab._probe_results = list(results)
        # Where the dot was drawn, so the mark watchdog's self-check measures
        # the same place and its reading is comparable with this launch's.
        lab._probe_spot_used = spot
        choice = expo.choose_detailed(results)
        best, why = choice.best, choice.why
        for line in expo.table(results):
            yield Log(line)
        # ONE GREPPABLE LINE PER LAUNCH, same key=value shape as the
        # watchdog's own readings. Tonight the most useful number was this
        # one -- present in every log since noon -- and nobody had compared it
        # across runs until hours too late: 5 on the launch that was already
        # dying, 43-52 on the healthy ones.
        #   grep READING camera_lab*.log
        _ref = best
        if _ref is None and results:
            _ref = min(results, key=lambda p: abs(p.exposure
                                                  - expo.GUN_ANCHOR_EXPOSURE))
        if _ref is not None:
            yield Log(f"READING kind=probe exposure={_ref.exposure:g} "
                      f"brightness={_ref.brightness:.2f} "
                      f"dot_peak={_ref.dot_peak:.0f} floor={_ref.floor:.0f} "
                      f"floor_at={lab._probe_floor_at} "
                      f"chosen={'yes' if best is not None else 'no'}")
        # Said once the noise floor has been measured, never while it is
        # being measured: a panel on the screen would BECOME the floor.
        spread_warning = choice.warning

        if best is None and choice.unmeasurable:
            # Plainly, with the numbers, in the log AND on the target screen.
            yield Log("  " + why)
            yield Call(lambda: _say_unmeasurable(lab, choice.warning))
            yield Control("exposure", float(base))
            return None

        if best is None:
            yield Log("    nothing measured; keeping the exposure as it was")
            yield Control("exposure", float(base))
            return None

        yield Log(f"  chose exposure {best.exposure:g} - {why}")
        yield Control("exposure", float(best.exposure))
        yield Call(lambda: lab._apply_chosen_exposure(
            float(best.exposure),
            [(p.exposure, p.mean, 255, p.floor) for p in results]))

        # THE FLOOR IS MEASURED HERE, as the next line, not through a
        # timer that the following routine had to guess the length of.
        #
        # The old code ended with `QTimer.singleShot(1800, then)` and a
        # comment explaining that 300ms had been too short -- the self test
        # started before `_calibrating` was raised, so its interlock saw
        # nothing to wait for and it ran against the previous exposure's
        # bar. There is nothing to arm and nothing to guess now: whatever
        # runs next cannot start until this returns.
        yield Log(f"  settling at {best.exposure:g}, then measuring the "
                  f"noise floor it gives")
        yield Sleep(int(float(lab.settings.settle_seconds) * 1000))
        # A STALLED MEASUREMENT RESUMES WITH None, NOT A TUPLE.
        #
        # `peak, p99, frames = yield Floor()` raised TypeError on a timeout
        # -- so a camera that stopped delivering turned an abandoned step
        # into a crashed routine, and the furniture that
        # `_cover_desktop_for_noise` stood down would have stayed down for
        # the session because the thing that restores it is the consumer of
        # the result. Cleanup not in a `finally`, in the module whose whole
        # argument is `finally`. Found by an outside review.
        measured = yield Floor()
        if measured is None:
            yield Log("  the noise floor could not be measured (the camera "
                      "stopped delivering). Keeping the bar that is already "
                      "set; press N to measure it.")
            if spread_warning:
                yield Call(lambda: _say_spread(lab, spread_warning))
            return float(best.exposure)
        peak, p99, frames = measured
        yield Call(lambda: lab._apply_noise_measurement(peak, p99, frames))
        if spread_warning:
            yield Call(lambda: _say_spread(lab, spread_warning))
        return float(best.exposure)

    finally:
        # One place, reached however this ends -- returned, raised,
        # stalled, or stood down at shutdown.
        _hide_dot(lab)
        # AND THE FURNITURE COMES BACK HERE, not in the consumer of the
        # floor result. `_cover_desktop_for_noise` stands it down and
        # `_on_noise_measured` restores it, so a floor that timed out left
        # the HUD and the prompt hidden for the rest of the session -- the
        # shooter's only standing indication of what the app is doing.
        _uncover(lab)
        # THE PROBE TEACHES THE BACKGROUND MODEL ABOUT ITS OWN DOT.
        #
        # The background updates on every frame, including frames holding a
        # detection -- deliberately, and CLAUDE.md explains why. Straight
        # after a probe, the self test's own tuning dots were "not seen" at
        # four radii and scored 11 against a bar of 10 at the fifth, while
        # the probe had just measured 27 at a QUARTER of that brightness.
        # The detector was doing exactly what it was built to do, to a dot
        # the app had spent a minute teaching it to ignore.
        if lab.capture is not None:
            lab.capture.reset_background()


def _say_unmeasurable(lab: Any, text: str) -> None:
    """No candidate's own dot cleared its own bar: the probe chose nothing.
    Said in the log AND on the target screen, with the numbers."""
    lab._log("  WARNING: " + text)
    try:
        lab._alert("EXPOSURE NOT CHOSEN", text, kind="warn", seconds=20.0)
    except Exception:                                        # noqa: BLE001
        pass


def _say_spread(lab: Any, text: str) -> None:
    """The panel's brightness spread exceeds what one exposure covers: say so
    in the log AND on the target screen, with the measured numbers."""
    lab._log("  WARNING: " + text)
    try:
        lab._alert("ONE EXPOSURE CANNOT COVER THIS PANEL", text,
                   kind="warn", seconds=20.0)
    except Exception:                                        # noqa: BLE001
        pass


def probe_job(lab: Any) -> Job:
    return Job(
        name="probe",
        body=lambda: exposure_probe(lab),
        label="measuring the camera - do not fire",
        draws=True,
        draws_red=True,            # it puts a red dot on the screen
        consumes_detections=True,  # ...which must not become a gunshot
        measures_screen=True,      # ...and furniture must stand down
        # Waits for the slot rather than being dropped. All three of
        # these used to reschedule themselves on a 700-900ms loop with
        # no bound, which is the same idea without a limit or a record
        # of having done it.
        queueable=True,
    )


def _show_dot(lab: Any, fx: float, fy: float, radius: float,
              brightness: float) -> None:
    synth = lab.scene.get("synthetic")
    if synth is None or lab.target_window is None:
        return
    rect = lab.target_window.rect()
    # Moved between candidates. A dot that sits in one place is a dot the
    # background model can learn; the reset in the `finally` is the cure,
    # this is the prevention. It also samples several parts of a panel
    # whose brightness falls by half from the near side to the far one.
    synth.show_at((rect.width() * fx, rect.height() * fy), radius,
                  brightness=brightness)
    lab.scene.mark_dirty()


def _show_dots(lab: Any, spot: tuple[float, float],
               strong_spot: "tuple[float, float] | None", radius: float,
               brightness: float) -> None:
    """The probe's dot at the weak spot and, in the same frame, at the strong
    one (None = weak only)."""
    synth = lab.scene.get("synthetic")
    if synth is None or lab.target_window is None:
        return
    rect = lab.target_window.rect()
    also = ([(rect.width() * strong_spot[0], rect.height() * strong_spot[1])]
            if strong_spot is not None else None)
    synth.show_at((rect.width() * spot[0], rect.height() * spot[1]), radius,
                  brightness=brightness, also=also)
    lab.scene.mark_dirty()


def _show_dot_at(lab: Any, point: tuple[float, float], radius: float,
                 brightness: float) -> None:
    synth = lab.scene.get("synthetic")
    if synth is None:
        return
    synth.show_at(point, radius, brightness=brightness)
    lab.scene.mark_dirty()


def _uncover(lab: Any) -> None:
    """Let the informational furniture come back after a measurement.

    Paired with `_cover_desktop_for_noise`, which the sink calls before
    every Floor. Restoring it here rather than in the handler that consumes
    the floor result means it happens on every exit path -- including a
    measurement that never arrived (RULES B2 and the five
    cleanup-skipped-on-one-path incidents).

    Deliberately does NOT restore it while something else is still
    measuring: putting the HUD back in the middle of the grid draws it over
    the ring being shot at, which these notes already record as hiding the
    one thing the shooter needs to see.
    """
    try:
        if lab._screen_busy() or lab.director.measuring_screen():
            return
        if lab.scene is not None and lab.scene.calibrating:
            lab.scene.calibrating = False
            lab.scene.mark_dirty()
    except Exception:
        pass


def _hide_dot(lab: Any) -> None:
    try:
        synth = lab.scene.get("synthetic")
        if synth is not None:
            synth.stop()
            lab.scene.mark_dirty()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The self test: the screen tests the camera
# ---------------------------------------------------------------------------
#
# Asked for in these words: "you should just flash small red dots on screen
# and see how they are being caught by camera... then you see how the rest
# of the machinery catches those red dots and where the holes appear."
#
# Honest limits, so no result is over-claimed: the screen EMITS this dot, a
# laser REFLECTS off the panel. Brightness, angular spread and spectrum all
# differ, so the detection RATE measured here is not the gun's. What
# transfers exactly is geometry, plumbing and ACCURACY -- where every fault
# found so far has lived.


def _flash_dot(lab: Any, point: tuple[float, float], radius: float,
               brightness: float) -> Generator:
    """Put one dot up for a pulse, take it down, and report what was seen.

    Composed into both phases of the self test with `yield from`, which is
    the whole reason the sequence is readable: the old code expressed this
    as three methods (`_selftest_step`, `_selftest_hide`,
    `_selftest_record`) that found each other through `_selftest_index` and
    could not be called twice for two different purposes.
    """
    yield Call(lambda: lab._selftest_arm())
    yield Call(lambda: _show_dot_at(lab, point, radius, brightness))
    yield Paint()
    # A LaserLyte pulse is ~100ms, so the dot is held for about as long and
    # the tracker sees a comparable number of frames.
    yield Sleep(lab.SELFTEST_HOLD_MS)
    yield Call(lambda: _hide_dot(lab))
    yield Paint()
    # Let the tracker close the shot and the scene settle, or two dots
    # merge into one track.
    yield Sleep(lab.SELFTEST_GAP_MS)
    return lab._selftest_seen


def self_test(lab: Any, rounds: int = 1) -> Generator:
    """Flash red dots at known places and measure where they are reported.

    Returns the list of (wanted, seen) pairs.
    """
    points = lab._selftest_grid(rounds)
    # REMEMBERED FROM LAST TIME, if this camera has not moved.
    #
    # Tuning the dot costs four flashes and about four seconds, and it was
    # being paid on every launch to rediscover a number that had not
    # changed. It is a property of this camera in this position looking at
    # this panel -- the same things the mapping describes, invalidated by
    # the same event -- so it is saved with the mapping, exactly as the
    # learned sensitivity is.
    remembered = float(getattr(lab.mapping, "dot_radius", 0.0) or 0.0)
    radius = remembered or lab._selftest_dot_radius(11.0)
    tune_shots = 1 if remembered else lab.SELFTEST_TUNE_SHOTS
    results: list[tuple[tuple[float, float], Any]] = []

    # Counted as a DELTA across the test, not cumulatively since launch.
    # Two otherwise clean gate launches failed on 1 and 3 "rejected too
    # short" that had happened during the app's own calibration flashes
    # minutes earlier.
    rej_before = lab._rejected_runs()

    yield Log(f"SELF TEST: flashing {len(points)} red dots at known "
              f"positions. "
              + (f"Dot radius {radius:.0f} screen px, remembered from the "
                 f"saved calibration -- one flash to confirm it rather than "
                 f"four to rediscover it. "
                 if remembered else
                 f"Starting radius {radius:.0f} screen px, tuned over "
                 f"{tune_shots} dots until the camera sees a blob of about "
                 f"{lab.SELFTEST_TARGET_AREA:.0f}px -- what a real laser "
                 f"measures here. ")
              + "No shooting.")
    yield Log("  (this deliberately breaks the no-red rule, which is the "
              "whole point: the camera cannot tell this from a laser)")

    try:
        # -- phase 1: tune the dot's size against what the camera reports --
        #
        # A screen EMITS its dot and blooms in the sensor; a laser dot
        # REFLECTS off matte coating and does not. He spotted it before the
        # code did: "your red dots are much bigger than mine". Scaling 11
        # camera px of radius through the mapping gave 63 screen px and the
        # camera measured a blob of radius 21 -- twice the diameter asked
        # for. So the size is tuned in a feedback loop against measurement.
        middle = lab._selftest_middle()
        stop = False
        measured = False          # did the LAST tuning flash see the dot?
        # What the LAST flash proved, about the radius it was FLASHED at --
        # not the radius the tuner then moved on to, which no camera has seen.
        flashed = radius
        flashed_area: Any = None
        confirmed = False         # seen, and as big as his laser could be
        left = tune_shots - 1
        # Bounded (RULES G5): at most one confirming flash plus a full tune.
        for _ in range(lab.SELFTEST_TUNE_SHOTS + 1):
            seen = yield from _flash_dot(lab, middle, radius, 1.0)
            measured = seen is not None
            flashed = radius
            flashed_area = seen[2] if seen is not None else None
            confirmed = measured and lab.dot_area_in_band(flashed_area)
            if not measured and remembered and left <= 0:
                # THE SHORTCUT WAS WRONG, so it is spent rather than nudged.
                #
                # Fable's review, 2026-10-16. With a radius remembered only
                # ONE confirming flash runs; if it was not seen, the tuner
                # returned radius x1.6 with stop=False and that inflated
                # value was persisted -- unconfirmed. A bigger dot is easier
                # to detect, so the next verdict reads BETTER than the truth,
                # and it compounds launch by launch. The verdict is the one
                # thing on this project that must not flatter itself.
                left = lab.SELFTEST_TUNE_SHOTS - 1
                remembered = 0.0
                yield Log("  tuning: the remembered radius was not seen. "
                          "Tuning from scratch rather than carrying a "
                          "guess forward.")
            if (measured and not confirmed and remembered and left <= 0):
                # SEEN, BUT NOT A BLOB HIS LASER COULD MAKE (2026-10-22).
                # The band gate means a remembered radius that images out
                # of band is no longer re-saved after one nudge. Without
                # this it would stay out of band for ever -- one nudge a
                # launch, none of them kept -- so a stale number could
                # never be corrected. Spend the full tune instead.
                left = lab.SELFTEST_TUNE_SHOTS - 1
                remembered = 0.0
                yield Log(f"  tuning: the remembered radius imaged "
                          f"{float(flashed_area):.0f}px, outside the "
                          f"{lab.SELFTEST_AREA_BAND[0]:.0f}-"
                          f"{lab.SELFTEST_AREA_BAND[1]:.0f}px his laser "
                          f"measures. Tuning from scratch.")
            radius, stop, note = lab._selftest_retune(radius, seen, left)
            yield Log(note)
            if stop or left <= 0:
                break
            left -= 1
        # SAVE IT ONLY IF A MEASUREMENT CONFIRMED IT.
        #
        # Three ways it must not be saved, and the last two were the ratchet:
        #   `stop`          the tuner gave up -- "still not seen at 410px",
        #                   a known-bad number carried into every launch.
        #   not seen        the last flash was not seen, so the radius is the
        #                   unconfirmed result of growing it blindly.
        #   seen, no band   (2026-10-22) seen, but as a blob outside what his
        #                   laser measures (113-746px). A dot grown to reach
        #                   the target area from a tiny blob is a working
        #                   value for THIS attempt: 9.62 -> 14.43 in one
        #                   session on 2026-10-20, and 8.7 -> 47 over two days.
        # What is saved is the radius that was FLASHED and confirmed, never
        # the one the tuner moved on to afterwards. A session that confirms
        # nothing leaves the remembered radius exactly as it was.
        if stop or not confirmed:
            yield Log("  the dot radius is NOT saved: "
                      + ("the last flash was not seen, so this size was "
                         "grown rather than measured"
                         if not measured else
                         "the last flash imaged "
                         f"{float(flashed_area):.0f}px, outside the "
                         f"{lab.SELFTEST_AREA_BAND[0]:.0f}-"
                         f"{lab.SELFTEST_AREA_BAND[1]:.0f}px his laser "
                         "measures, so it is a working value for this "
                         "attempt and not a measurement")
                      + ". A dot that is too big makes the verdict read "
                      "better than the truth.")
        else:
            yield Call(lambda r=flashed, a=flashed_area:
                       lab._remember_dot_radius(r, a))

        # -- phase 2: the measured grid -----------------------------------
        spread = lab._selftest_spread()
        for i, point in enumerate(points):
            # Fire the whole brightness spread a real pulse has, not just
            # the brightest: each rung is one decile of his 575 measured
            # trigger pulls divided by his median.
            bright = spread[i % len(spread)]
            seen = yield from _flash_dot(lab, point, radius, bright)
            results.append((point, seen))
            yield Log(lab._selftest_line(i, len(points), point, seen, bright))

        yield Call(lambda: lab._selftest_report(results, rej_before))
        return results
    finally:
        _hide_dot(lab)


def self_test_job(lab: Any, rounds: int = 1) -> Job:
    return Job(
        name="selftest",
        body=lambda: self_test(lab, rounds),
        label="testing itself - do not fire",
        draws=True,
        draws_red=True,
        consumes_detections=True,
        measures_screen=True,
        queueable=True,
    )


# ---------------------------------------------------------------------------
# The two-marker mapping check
# ---------------------------------------------------------------------------
#
# His idea, and it is the right one: "You just need to flash two squares
# (for example, upper left and lower right) and see if that corresponds to
# the previously saved layout. If that corresponds, that means the camera
# did not move."
#
# A mapping is invalidated by the camera moving and by nothing else, so
# checking it beats rebuilding it: about three seconds against twenty-four,
# measured at 0.5px / 0.7mm on his bench. Diagonal corners on purpose -- a
# shift moves both markers the same way, a rotation moves them oppositely,
# and a change of distance alters the gap between them. One marker would
# miss rotation entirely.


def mapping_check(lab: Any) -> Generator:
    """Confirm the saved mapping still describes where the camera is.

    Returns (ok, detail). `detail` always names what was measured, never
    just a verdict: "one of the two markers was not visible" with no
    marker named and no distance reported is a check that can only be
    re-run, not debugged (RULES C3).
    """
    targets = lab._check_targets()
    if not targets:
        return (False, "there is no saved mapping to check")

    results: list[tuple[tuple[float, float], Any]] = []

    yield Log("checking the saved calibration with two markers "
              "(upper-left and lower-right). If the camera has not moved, "
              "the mapping measured earlier is kept and nothing is "
              "remeasured.")
    try:
        # A DARK, FIXED EXPOSURE -- the same one the screen search uses.
        #
        # `_apply_preset("calibration", with_exposure=True)` was a no-op for
        # exposure: the calibration branch passes its own `exposure=`
        # keyword, which defaults to None, and `with_exposure` only applies
        # to the laser branch. So this check ran under AUTO exposure, and
        # CLAUDE.md measures what auto does to exactly this kind of
        # photograph -- flash difference 14.9 against 43-44 at a fixed -9,
        # -10 or -11 -- because auto compensates for the very change being
        # measured.
        #
        # It found its markers anyway once gain was restored, at strength
        # 255, so this is margin rather than a live failure. Fixed because
        # the 75" TV and the projector will not have that margin.
        yield Call(lambda: lab._apply_preset(
            "calibration", exposure=lab._check_exposure()))
        yield Sleep(260)
        for point in targets:
            yield Call(lambda p=point: lab._check_show(p))
            # TWO DIFFERENT WAITS, and conflating them cost this routine
            # its first run after the migration.
            #
            # `Paint()` proves the marker is ON THE GLASS: it waits for real
            # paintEvents, where the old code called `processEvents()` once
            # and hoped -- which is how a marker came to be photographed
            # before it was drawn, reading strength 14 against siblings at
            # 197-231, so the strongest noise blob in the difference image
            # won the comparison.
            #
            # The Sleep is a separate fact about the CAMERA: a webcam lags a
            # frame or two behind a change on screen, and at these exposures
            # the integration is long. The screen being correct does not
            # mean the sensor has seen it yet.
            #
            # I replaced the original 300ms settle with `Paint()` alone, on
            # the reasoning that waiting for actual pixels beats waiting for
            # a guessed interval. That reasoning is right and it threw away
            # the other half. Measured 2026-10-15 22:25: both markers came
            # back NOT FOUND in simulation -- lit and dark were effectively
            # the same frame, so their difference was noise -- while pattern
            # calibration found the identical marker at camera
            # (356.8,145.4), which is where this check had predicted it to
            # within a pixel. The check was right about where to look and
            # wrong about when.
            yield Paint()
            yield Sleep(300)
            lit = yield Still("check_lit", timeout_ms=4000)
            yield Call(lambda: lab._check_show(None))
            yield Paint()
            yield Sleep(240)
            dark = yield Still("check_dark", timeout_ms=4000)

            found = lab._check_locate(lit, dark)
            results.append((point, found))
            yield Log(lab._check_line(point, found))

        return lab._check_verdict_from(results)
    finally:
        # HIDE THE LAYER, not just the marker. `_check_show(None)` leaves a
        # full-screen BLACK field up at z=80, above everything, for the
        # rest of the session -- which is what it did on 2026-10-16.
        lab._check_hide()


def mapping_check_job(lab: Any) -> Job:
    return Job(
        name="mapcheck",
        body=lambda: mapping_check(lab),
        label="checking the camera has not moved - do not fire",
        draws=True,
        draws_red=False,           # white markers: chroma zero
        consumes_detections=True,
        measures_screen=True,
        queueable=True,
    )


# ---------------------------------------------------------------------------
# The mark watchdog's self-check
# ---------------------------------------------------------------------------
#
# A bang was heard three times running and nothing was drawn. This is the
# app's own answer to "is the camera still seeing a known dot?": one dot at the
# spot and brightness the exposure probe measured this launch, one lit and one
# dark photograph, about a second. NO camera control write (a single exposure
# write has hung this camera for 272 s): the exposure is whatever is set.
# Judged by `lsa.watchdog.judge` against an absolute band AND this launch's own
# probe reading.


def mark_watch_job(lab: Any) -> Job:
    return Job(
        name="markwatch",
        body=lambda: mark_watch_check(lab),
        label="checking the camera - do not fire",
        draws=True,
        draws_red=True,            # it puts a red dot on the screen
        consumes_detections=True,  # ...which must not become a gunshot or a hole
        measures_screen=True,      # ...and furniture must stand down (B2)
    )


def mark_watch_check(lab: Any) -> Generator:
    """Photograph the app's own dot once. Returns a dict, or None if nothing
    could be measured.

    Yields: Call, Sleep, Still. Never Control.
    """
    spot, brightness, baseline, exposure = lab._watch_plan()
    radius = lab._selftest_dot_radius(11.0)
    try:
        yield Call(lambda: _show_dots(lab, spot, None, radius, brightness))
        yield Sleep(lab.PROBE_DOT_MS)
        lit = yield Still("watch_lit")
        yield Call(lambda: _hide_dot(lab))
        yield Sleep(lab.PROBE_DOT_MS)
        dark = yield Still("watch_dark")
        if lit is None or dark is None:
            # The camera did not deliver: that IS the finding, and it is as
            # bad as a reading gets.
            return {"peak": 0.0, "floor": 0.0, "baseline": baseline,
                    "brightness": brightness, "exposure": exposure,
                    "spot": spot, "frames": False}
        peak, floor, _mean, _s = lab._probe_measure_pair(lit, dark, spot, None)
        if peak is None:
            return None
        return {"peak": float(peak), "floor": float(floor),
                "baseline": baseline, "brightness": brightness,
                "exposure": exposure, "spot": spot, "frames": True}
    finally:
        _hide_dot(lab)
        _uncover(lab)
        # The dot is the app's own light; do not let the background model learn
        # it (see the probe's `finally`).
        if lab.capture is not None:
            lab.capture.reset_background()
