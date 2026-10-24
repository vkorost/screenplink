"""The startup verdict, against every failure this project actually produced.

Advice is only worth emitting if it is right, and this project has emitted
wrong advice more than once -- most memorably telling a shooter to dim a room
that was already as dark as it would go. So every case below is a real
measurement from a real session, with the date it happened, and the test
asserts both the verdict AND that the action named is the one that would have
helped.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa.selfcheck import SelfCheck, Verdict, assess        # noqa: E402

checks: list[bool] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    checks.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))


def words(report) -> str:
    return " ".join(report.lines()).lower()


def main() -> int:
    print("=== the startup verdict ===")

    # ---------------------------------------------------------------- 1 ------
    print("\n1. a healthy session says READY and nothing else")
    good = SelfCheck(fired=12, detected=12, error_mm=3.0, worst_error_mm=4.6,
                     peak_median=85, noise_floor=4, min_peak=10, fps=38.0,
                     exposure=-5.0, exposure_spread=1.0, flash_difference=44.0,
                     held_out_mm=0.9, audio_ok=True)
    r = assess(good)
    check("verdict is READY", r.verdict is Verdict.READY, r.summary)
    check("and it reports the number that matters",
          "12 of 12" in r.summary, r.summary)
    check("with no findings to wade through", not r.findings,
          f"{len(r.findings)} findings")

    # ---------------------------------------------------------------- 2 ------
    # 2026-10-15 15:05. Room bright, exposure forced to -8, median shot 18
    # against a bar of 15. Vladimir: "some of them are seen just like a faint
    # shadow". This is the case the whole module exists for.
    print("\n2. the afternoon: dot four times too dim, room too bright")
    dim = SelfCheck(fired=12, detected=6, error_mm=3.0, peak_median=18,
                    noise_floor=9, min_peak=15, fps=25.0, exposure=-8.0,
                    exposure_spread=1.0, flash_difference=44.0,
                    audio_ok=True, room_bright=True)
    r = assess(dim)
    check("verdict is BLOCKED", r.verdict is Verdict.BLOCKED, r.summary)
    check("it leads with the detection rate",
          "6 of 12" in r.findings[0].headline, r.findings[0].headline)
    # NAMED, NOT INSTRUCTED. Changed 2026-10-16, for the fourth time
    # (RULES C4, D7).
    #
    # "Dim the room" was on his screen again at 11:03 while he was
    # shooting, after he had closed the window and darkened the room as
    # far as he can and still see his sights. And `room_bright` is not a
    # measurement of the room he stands in: it is the mean and chroma of
    # the WHOLE camera frame, which includes the sunlit window beside the
    # monitor. There is no room state in which that advice is actionable.
    #
    # So the app still SAYS what is limiting it -- that is information he
    # is owed -- and the action it gives him is the one he can take.
    check("it names the daylight as what is limiting the exposure",
          "daylight" in words(r) or "window" in words(r), words(r)[:160])
    check("and the action is one he can actually take",
          "press x" in words(r),
          "he has already closed the window and dimmed the room; the only "
          "lever left is re-measuring the exposure against it")
    check("it does NOT tell him to dim a room again",
          "dim the room" not in words(r) and "close a blind" not in words(r))
    check("it explains the flicker, not just the threshold",
          "flicker" in words(r) or "4x" in words(r))
    check("it does NOT blame the camera", "replug" not in words(r))

    # ---------------------------------------------------------------- 3 ------
    # 2026-10-14 19:22. The room was already as dark as it would go and the
    # app told him to darken it. The floor is the discriminator.
    print("\n3. a faint dot in an ALREADY dark room must not blame the room")
    dark = SelfCheck(fired=12, detected=5, error_mm=3.0, peak_median=18,
                     noise_floor=2, min_peak=15, fps=30.0, exposure=-7.0,
                     exposure_spread=1.0, flash_difference=44.0,
                     audio_ok=True, room_bright=False)
    r = assess(dark)
    check("still BLOCKED", r.verdict is Verdict.BLOCKED)
    check("but it does NOT say to dim the room",
          "dim the room" not in words(r),
          "this exact advice was given to a man sitting in the dark")
    check("it offers exposure or camera distance instead",
          "exposure" in words(r) and "closer" in words(r))

    # ---------------------------------------------------------------- 4 ------
    # 2026-10-14. Camera accepted exposure writes and ignored them.
    print("\n4. a camera ignoring exposure blocks everything else")
    ignoring = SelfCheck(fired=12, detected=0, peak_median=None,
                         noise_floor=90, min_peak=105, fps=20.0,
                         exposure_spread=0.07, flash_difference=2.0,
                         room_bright=True)
    r = assess(ignoring)
    check("verdict is BLOCKED", r.verdict is Verdict.BLOCKED)
    check("the camera fault is reported FIRST",
          "ignoring" in r.findings[0].headline.lower(),
          r.findings[0].headline)
    check("and it does not bury it under room advice",
          "dim the room" not in words(r),
          "while the camera ignores exposure, dimming the room does nothing")
    check("it says to replug", "replug" in words(r))

    # ---------------------------------------------------------------- 5 ------
    # 2026-10-15 12:12. Canon virtual device: frames arrive, screen invisible.
    print("\n5. a camera that cannot see the screen")
    blind = SelfCheck(fired=12, detected=0, fps=20.0, exposure_spread=1.0,
                      flash_difference=0.0, noise_floor=0, min_peak=10)
    r = assess(blind)
    check("verdict is BLOCKED", r.verdict is Verdict.BLOCKED)
    check("it says the camera cannot see the screen",
          "cannot see this screen" in words(r))
    check("and sends him to check the aim, not the threshold",
          "aimed at the monitor" in words(r))

    # ---------------------------------------------------------------- 6 ------
    # 2026-10-15 13:49. 18 fps, 32 pulses discarded as too short.
    print("\n6. too few frames to confirm a shot")
    slow = SelfCheck(fired=12, detected=7, error_mm=3.0, peak_median=85,
                     noise_floor=4, min_peak=10, fps=18.0, rejected_short=32,
                     exposure_spread=1.0, flash_difference=44.0, audio_ok=True)
    r = assess(slow)
    check("verdict is at least DEGRADED", r.verdict is not Verdict.READY)
    check("it names the frame rate", "frame rate" in words(r))
    check("it explains the 100ms pulse", "100ms" in words(r))
    check("and suggests turning recording off", "recording" in words(r))

    # ---------------------------------------------------------------- 7 ------
    print("\n7. a bad mapping is BLOCKED, not merely noted")
    skewed = SelfCheck(fired=12, detected=12, error_mm=78.0, worst_error_mm=120.0,
                       peak_median=85, noise_floor=4, min_peak=10, fps=38.0,
                       exposure_spread=1.0, flash_difference=44.0, audio_ok=True)
    r = assess(skewed)
    check("verdict is BLOCKED", r.verdict is Verdict.BLOCKED,
          "78mm was the flash-outline error that caused 'holes off by an inch'")
    check("it says to recalibrate", "recalibrate" in words(r))
    check("and mentions the camera may have moved", "moved" in words(r))

    # ---------------------------------------------------------------- 8 ------
    print("\n8. small faults are DEGRADED, and still say what to do")
    # detected 10 of 12, NOT 12 of 12: the margin finding now defers to a
    # detection rate that has already answered the question directly.
    thin = SelfCheck(fired=12, detected=10, error_mm=7.0, peak_median=20,
                     noise_floor=4, min_peak=10, fps=38.0,
                     exposure_spread=1.0, flash_difference=44.0, audio_ok=True)
    r = assess(thin)
    check("verdict is DEGRADED", r.verdict is Verdict.DEGRADED, r.summary)
    check("it is honest that it is usable", "usable" in r.summary.lower())
    check("thin margin is called out", "margin" in words(r))
    check("and the few-mm error too", "millimetre" in words(r))

    # ---------------------------------------------------------------- 9 ------
    print("\n9. silent audio does not block shooting")
    quiet = SelfCheck(fired=12, detected=12, error_mm=3.0, peak_median=85,
                      noise_floor=4, min_peak=10, fps=38.0,
                      exposure_spread=1.0, flash_difference=44.0,
                      audio_ok=False)
    r = assess(quiet)
    check("verdict is DEGRADED, not BLOCKED", r.verdict is Verdict.DEGRADED)
    check("it says detection still works", "detection still works" in words(r))

    # --------------------------------------------------------------- 10 ------
    print("\n10. every finding names an action")
    for name, c in (("dim", dim), ("dark", dark), ("ignoring", ignoring),
                    ("blind", blind), ("slow", slow), ("skewed", skewed),
                    ("thin", thin)):
        r = assess(c)
        missing = [f.headline for f in r.findings if not f.action]
        check(f"{name}: all findings carry an action", not missing,
              f"{missing}")

    # --------------------------------------------------------------- 11 ------
    print("\n11. every finding shows the measurement behind it")
    for name, c in (("dim", dim), ("ignoring", ignoring), ("slow", slow),
                    ("skewed", skewed)):
        r = assess(c)
        bare = [f.headline for f in r.findings
                if not f.measured and f.verdict is not Verdict.READY]
        check(f"{name}: findings are backed by numbers", not bare, f"{bare}")

    # --------------------------------------------------------------- 12 ------
    # THE DARK-ROOM RULE, OVER EVERY VERDICT THE APP CAN PRODUCE.
    #
    # This project has now told a man sitting in the dark to dim his room
    # TWICE -- 2026-10-14 through the thin-margin alert, and 2026-10-15
    # through the startup verdict that was supposed to have learned the
    # lesson. His words: "I already dimmed the room to the point where I
    # cannot dim it anymore because otherwise I will be shooting in
    # complete darkness but I need to see my gun sights."
    #
    # It came back because advice is emitted from a dozen places and only
    # SOME of them consult the light measurement the app already holds.
    # One more example case would protect one more branch. So this sweeps
    # the whole input space instead: every combination of detection rate,
    # margin, floor, frame rate, error and camera health, with the room
    # measured DARK, and fails if any finding anywhere mentions dimming.
    #
    # A thirteenth advice path that forgets the check breaks this before
    # he ever reads it.
    print("\n12. no verdict may tell a dark room to get darker")
    banned = ("dim the room", "dimming the room", "close a blind",
              "draw a blind", "darken the room", "turn off the light",
              "turn the lights")
    offenders = []
    tried = 0
    for detected in (0, 3, 6, 9, 10, 12):
        for peak, bar in ((None, 10), (8, 15), (18, 15), (30, 10),
                          (39, 15), (85, 10), (200, 8)):
            for floor in (0, 2, 9, 26, 90):
                for fps in (10.0, 18.0, 30.0, 60.0):
                    for err in (None, 1.0, 7.0, 78.0):
                        for spread, flash in ((1.0, 44.0), (0.07, 44.0),
                                              (1.0, 0.0)):
                          # probe_dot is a sweep axis too: the dud-camera
                          # branch is advice, so it is generated like the rest.
                          for pdot in (None, 0.0, 5.0, 14.9, 15.0, 44.0):
                              tried += 1
                              c = SelfCheck(
                                  fired=12, detected=detected,
                                  error_mm=err, peak_median=peak,
                                  noise_floor=floor, min_peak=bar, fps=fps,
                                  exposure=-7.0, exposure_spread=spread,
                                  flash_difference=flash, audio_ok=True,
                                  # THE ROOM IS DARK. Measured, not assumed.
                                  room_bright=False,
                                  probe_dot=pdot)
                              said = words(assess(c))
                              for phrase in banned:
                                  if phrase in said:
                                      offenders.append(
                                          f"rate {detected}/12 peak {peak} "
                                          f"bar {bar} floor {floor} -> "
                                          f"'{phrase}'")
    check(f"swept {tried} verdicts with the room measured dark",
          tried > 2000, f"{tried}")
    check("not one of them tells him to dim the room",
          not offenders, "; ".join(sorted(set(offenders))[:3]))

    # THE SELF CHECK IS ON C (2026-10-20, "lets rewire self-check to C"). A printed
    # instruction naming a key that no longer does the job is the same lie as a log
    # line naming the wrong cause, so the WORDS the verdicts produce are swept, not
    # a constant grepped: every verdict the generator can make, both room states.
    import re as _re12k
    key_old, key_new, said_all = [], 0, 0
    for room_bright in (False, True):
        for detected in (0, 6, 10, 12):
            for peak, bar in ((None, 10), (18, 15), (85, 10)):
                for floor in (0, 9, 90):
                    for fps in (10.0, 30.0, 60.0):
                        for err in (None, 1.0, 78.0):
                            for spread, flash in ((1.0, 44.0), (0.07, 44.0), (1.0, 0.0)):
                                for pdot in (None, 5.0, 44.0):
                                    t_ = words(assess(SelfCheck(
                                        fired=12, detected=detected, error_mm=err, peak_median=peak,
                                        noise_floor=floor, min_peak=bar, fps=fps, exposure=-7.0,
                                        exposure_spread=spread, flash_difference=flash, audio_ok=True,
                                        room_bright=room_bright, probe_dot=pdot)))
                                    said_all += 1
                                    # 2026-10-21: the self check moved T -> C -> A, and C is now
                                    # the SELF-CALIBRATION. So naming C to re-run THIS check is
                                    # just as wrong as naming T: it would send him to a 25-second
                                    # measurement of the screen instead of the 12-dot check.
                                    if (_re12k.search(r"(press|then|,) t\b", t_) or "t to run" in t_
                                            or "then c to run" in t_ or "c to run this check" in t_):
                                        key_old.append(t_[:80])
                                    if ("then a to run" in t_ or "a to run this check" in t_
                                            or _re12k.search(r"press a\b", t_)):
                                        key_new += 1
    check(f"swept {said_all} verdicts: not one names T or C for the self check",
          not key_old and said_all > 1000, str(key_old[:2]))
    check("...and the ones that tell him to run it again name A", key_new > 0, str(key_new))

    # And the mirror of it: when the room really IS bright, the advice must
    # still be there. A rule that silences the room everywhere would pass
    # the sweep above and be just as useless.
    bright = SelfCheck(fired=12, detected=6, error_mm=3.0, peak_median=18,
                       noise_floor=9, min_peak=15, fps=30.0, exposure=-8.0,
                       exposure_spread=1.0, flash_difference=44.0,
                       audio_ok=True, room_bright=True)
    # The mirror, restated 2026-10-16: a bright view is still NAMED, and
    # named as what it actually is. `room_bright` is the mean and chroma
    # of the WHOLE camera frame, so the thing it measures is the daylight
    # window beside his monitor, not the room he stands in. Telling him to
    # dim that room is advice he cannot act on and has been given three
    # times; telling him daylight is limiting the exposure is a fact he is
    # owed. The sweep above still fails if "dim the room" comes back.
    check("but a bright view is still named, as the window not the room",
          ("daylight" in words(assess(bright))
           or "window" in words(assess(bright))),
          words(assess(bright))[:200])
    check("and it is never phrased as an instruction to dim anything",
          "dim the room" not in words(assess(bright))
          and "close a blind" not in words(assess(bright)))
    thin_bright = SelfCheck(fired=12, detected=10, error_mm=3.0,
                            peak_median=25, noise_floor=9, min_peak=10,
                            fps=38.0, exposure_spread=1.0,
                            flash_difference=44.0, audio_ok=True,
                            room_bright=True)
    check("including on a merely thin margin",
          ("daylight" in words(assess(thin_bright))
           or "window" in words(assess(thin_bright))),
          words(assess(thin_bright))[:200])

    # The thin-margin finding in a DARK room must still be actionable --
    # silence is not the fix, the right cure is. A dark room is what
    # PERMITS a longer exposure, which is the actual remedy.
    thin_dark = SelfCheck(fired=12, detected=10, error_mm=3.0,
                          peak_median=25, noise_floor=2, min_peak=10,
                          fps=38.0, exposure_spread=1.0,
                          flash_difference=44.0, audio_ok=True,
                          room_bright=False)
    said = words(assess(thin_dark))
    check("a thin margin in the dark offers the exposure instead",
          "press x" in said and "exposure" in said)
    check("and offers more dot pixels as the other real cure",
          "closer" in said or "zoom" in said)
    check("and says plainly that the room is not the problem",
          "not the problem" in said or "already quiet" in said)

    # --------------------------------------------------------------- 13 ------
    # A MEASUREMENT BEATS A PREDICTION OF THE SAME THING.
    #
    # Measured live 2026-10-15: a cold launch detected all twelve emulated
    # shots -- a population drawn decile by decile from his own 575 logged
    # trigger pulls -- and the verdict still came back DEGRADED because a
    # margin heuristic read 3.9x against a bar of 4.0. The heuristic exists
    # to PREDICT how many shots will be lost. The rate had just MEASURED
    # it. Acting on the prediction would have sent him to change something
    # that was working.
    print("\n13. a measured rate silences the heuristic that predicts it")
    all_seen_thin = SelfCheck(fired=12, detected=12, error_mm=3.0,
                              peak_median=25, noise_floor=4, min_peak=10,
                              fps=38.0, exposure_spread=1.0,
                              flash_difference=44.0, audio_ok=True,
                              room_bright=False)
    r = assess(all_seen_thin)
    check("every emulated shot detected means READY", r.verdict is Verdict.READY,
          r.summary)
    check("and the thin-margin prediction stays quiet",
          "margin is thin" not in words(r),
          "it contradicts a direct measurement taken seconds earlier")
    some_lost_thin = SelfCheck(fired=12, detected=9, error_mm=3.0,
                               peak_median=25, noise_floor=4, min_peak=10,
                               fps=38.0, exposure_spread=1.0,
                               flash_difference=44.0, audio_ok=True,
                               room_bright=False)
    check("but when shots ARE lost it speaks up again",
          "margin is thin" in words(assess(some_lost_thin)),
          "the prediction explains WHY they were lost")

    # The bar itself comes from his data now, not from reasoning.
    from lsa.selfcheck import MARGIN_READY
    check("the ready margin is derived from his dimmest measured shot",
          2.6 <= MARGIN_READY <= 3.5,
          f"{MARGIN_READY}: his min 31 over his median 85 is 2.6x")

    # ---------------------------------------------------------------- 14 -----
    # 2026-10-17: the camera silently degraded and the app said READY. The
    # probe's own test dot had read 5 (then 10, then 0) before he fired a shot;
    # nine healthy launches read 25-97. Everything ELSE in this SelfCheck is
    # healthy: the dot reading is the only thing wrong.
    print("\n14. a dud camera is never READY, whatever else passed")
    from lsa.selfcheck import camera_not_delivering, PROBE_DOT_BAD_BELOW
    from lsa import watchdog
    check("one threshold for one fact: the watchdog's own number",
          PROBE_DOT_BAD_BELOW is watchdog.ABS_BAD_BELOW
          or PROBE_DOT_BAD_BELOW == watchdog.ABS_BAD_BELOW,
          f"{PROBE_DOT_BAD_BELOW} vs {watchdog.ABS_BAD_BELOW}")

    def healthy(**kw):
        base = dict(fired=12, detected=12, error_mm=3.0, worst_error_mm=4.6,
                    peak_median=85, noise_floor=4, min_peak=10, fps=38.0,
                    exposure=-5.0, exposure_spread=1.0, flash_difference=44.0,
                    held_out_mm=0.9, audio_ok=True)
        base.update(kw)
        return SelfCheck(**base)

    for reading, label in ((5.0, "18:28 bad session"), (10.0, "18:46"),
                           (0.0, "19:00 wedged")):
        r = assess(healthy(probe_dot=reading))
        check(f"dot {reading:g} ({label}) is BLOCKED even with 12/12",
              r.verdict is Verdict.BLOCKED, r.summary)
        check(f"  ... and it says the camera is not delivering, with the "
              f"number", "not delivering" in words(r)
              and f"{reading:.0f}" in words(r), words(r)[:90])
        check("  ... and tells him not to shoot, with a thing he can do",
              "do not shoot" in words(r) and "unplug" in words(r))
    for reading in (25.0, 44.0, 46.0, 49.0, 52.0, 97.0, 15.0):
        check(f"dot {reading:g} (healthy band, or exactly on the line) "
              f"is READY", assess(healthy(probe_dot=reading)).verdict
              is Verdict.READY)
    r = assess(healthy(probe_dot=None))
    check("NO reading (probe bailed) is not a fault: absent evidence",
          r.verdict is Verdict.READY and not camera_not_delivering(
              healthy(probe_dot=None)), r.summary)

    # --------------------------------------------------------------- 14 ------
    print()
    print("14. NO CALIBRATION is BLOCKED, and it says so (2026-10-18)")
    # 10:04:39 on his bench. Pattern calibration was rejected, the app went
    # uncalibrated, and he fired 42 shots. Every one was DETECTED at peaks
    # 76-109 against a bar of 8 -- margins of 9 to 13x, a textbook-healthy
    # detector -- and not one left a hole, because there is nowhere to draw
    # one. He concluded the app was broken, and nothing contradicted him: no
    # verdict was published at all.
    #
    # The old `assess` with no mapping, a healthy camera and nothing fired
    # returned "READY - shoot."
    nocal = SelfCheck(noise_floor=1.0, min_peak=8.0, fps=49.0, exposure=-5.0,
                      exposure_spread=1.0, flash_difference=44.0,
                      audio_ok=True, calibrated=False)
    r = assess(nocal)
    check("no mapping is BLOCKED, not READY", r.verdict is Verdict.BLOCKED,
          r.summary)
    check("  ... and it NAMES the problem",
          "no calibration" in words(r) or "no hole can be drawn" in words(r),
          words(r)[:120])
    check("  ... and says the shots ARE being seen, so he does not chase the "
          "detector", "seen" in words(r), words(r)[:120])
    check("  ... and names a key to press",
          "press k" in words(r) or "press g" in words(r), words(r)[:160])
    # Everything else healthy must not drown it out, and a 12/12 self test
    # cannot happen without a mapping anyway.
    r2 = assess(SelfCheck(fired=12, detected=12, error_mm=3.0,
                          worst_error_mm=4.6, peak_median=85, noise_floor=4,
                          min_peak=10, fps=38.0, exposure=-5.0,
                          exposure_spread=1.0, flash_difference=44.0,
                          held_out_mm=0.9, audio_ok=True, calibrated=False))
    check("  ... and a perfect check elsewhere does not make it READY",
          r2.verdict is Verdict.BLOCKED, r2.summary)
    # RULES C1 cuts both ways: UNKNOWN is not a fault. Every existing caller
    # and all 10,080 generated verdicts leave this None.
    check("calibrated=None (unknown) is NOT read as a fault",
          assess(SelfCheck(fired=12, detected=12, error_mm=3.0,
                           worst_error_mm=4.6, peak_median=85, noise_floor=4,
                           min_peak=10, fps=38.0, exposure=-5.0,
                           exposure_spread=1.0, flash_difference=44.0,
                           held_out_mm=0.9, audio_ok=True)).verdict
          is Verdict.READY)
    check("and calibrated=True is unchanged",
          assess(SelfCheck(fired=12, detected=12, error_mm=3.0,
                           worst_error_mm=4.6, peak_median=85, noise_floor=4,
                           min_peak=10, fps=38.0, exposure=-5.0,
                           exposure_spread=1.0, flash_difference=44.0,
                           held_out_mm=0.9, audio_ok=True,
                           calibrated=True)).verdict is Verdict.READY)
    # RULES C4, for the fifth time on this project: it must not tell a man
    # sitting in the dark to dim his room.
    for r3 in (r, r2):
        low = words(r3)
        check("  ... and says nothing about dimming or blinds",
              not any(w in low for w in ("dim ", "dimming", "blind",
                                         "darken")), low[:120])

    print()
    if all(checks):
        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
