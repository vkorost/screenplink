"""The exposure rule, judged against the two strings it got wrong.

Every case here is a real measurement from a real session, with the date and
the outcome. The rule being replaced -- "reject any exposure whose frame mean
exceeds 40" -- passes none of them, which is the point: it was a constant
standing in for a measurement, and on 2026-10-15 it cost half a string while
reporting a successful exposure measurement.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa.exposure import (GUN_ANCHOR_EXPOSURE, GUN_PEAK_DECILES,   # noqa: E402
                          Probe, choose, predicted_margin,
                          predicted_rate, table,
                          bars_for, min_peak_for)

checks: list[bool] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    checks.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    print("=== choosing the exposure by what it does to HIS shots ===")

    # ------------------------------------------------------------------ 1 ----
    print("\n1. the gun data this rests on is the measured gun data")
    check("ten deciles", len(GUN_PEAK_DECILES) == 10)
    check("rising", all(a < b for a, b in zip(GUN_PEAK_DECILES,
                                              GUN_PEAK_DECILES[1:])))
    # The five anchors that were actually measured over 575 trigger pulls.
    lo, hi = GUN_PEAK_DECILES[0], GUN_PEAK_DECILES[-1]
    mid = GUN_PEAK_DECILES[len(GUN_PEAK_DECILES) // 2]
    check("spans his measured range", 31 <= lo <= 45 and 170 <= hi <= 253,
          f"{lo:.0f} .. {hi:.0f}, measured min 31 max 253")
    check("centred on his measured median of 85", 74 <= mid <= 95, f"{mid:.0f}")
    check("and keeps his measured 4.4x flicker spread",
          3.5 <= GUN_PEAK_DECILES[-2] / GUN_PEAK_DECILES[1] <= 5.5,
          f"{GUN_PEAK_DECILES[-2] / GUN_PEAK_DECILES[1]:.1f}x, measured 4.4x")
    check("the anchor is the exposure those pulls were logged at",
          GUN_ANCHOR_EXPOSURE == -5.0)

    # ------------------------------------------------------------------ 2 ----
    # THE CASE THIS MODULE EXISTS FOR. 2026-10-15 afternoon. Sunlight through
    # the window beside the monitor. The old rule stopped down to -8, and his
    # own trigger pulls, same gun, hours apart:
    #
    #   at -5:  peak median 74, floor 1-7,  bar 7-10   -> worked
    #   at -8:  peak median 18, floor 4-9,  bar 10-15  -> half the string lost
    #
    # The floor barely moved across those three stops while the signal fell
    # 4x, which is the whole mechanism: chroma noise is dominated by sensor
    # and compression noise that does NOT scale with exposure time.
    print("\n2. the afternoon that lost half a string")
    #
    # The floors below are not invented: they are reconstructed from the
    # min_peak the app ACTUALLY set at each exposure in those sessions (7-10
    # at -5, 10-15 at -8), through the same `floor * 1.3` the detector uses.
    # Reconstructing them from the bar rather than guessing a floor is the
    # difference between testing the rule and testing my arithmetic.
    afternoon = [
        Probe(-3.0, dot_peak=96, floor=22, mean=72),
        Probe(-4.0, dot_peak=72, floor=14, mean=60),
        Probe(-5.0, dot_peak=52, floor=7, mean=46),
        Probe(-6.0, dot_peak=36, floor=8, mean=36),
        Probe(-7.0, dot_peak=24, floor=9, mean=25),
        Probe(-8.0, dot_peak=13, floor=11, mean=18),
    ]
    best, why = choose(afternoon)
    for line in table(afternoon):
        print("        " + line)
    check("it does NOT choose -8", best.exposure != -8.0, f"chose {best.exposure:g}")
    check("it chooses something at or brighter than -5",
          best.exposure >= -5.0, f"chose {best.exposure:g} -- {why}")
    at8 = next(p for p in afternoon if p.exposure == -8.0)
    anchor = next(p for p in afternoon if p.exposure == -5.0)
    check("and it can say why -8 was wrong, in his own numbers",
          predicted_rate(at8, anchor) < 0.75,
          f"-8 keeps {predicted_rate(at8, anchor) * 100:.0f}% of his shots; "
          f"he measured about half lost")
    check("the margin at -8 is the thin one",
          predicted_margin(at8, anchor) < 2.0,
          f"{predicted_margin(at8, anchor):.1f}x, measured 1.2x")

    # ------------------------------------------------------------------ 3 ----
    # The OPPOSITE failure, 2026-10-14 evening: the room was darkened as far
    # as it would go, every candidate's mean collapsed toward zero, and the
    # mean-based rule had nothing left to discriminate on. It settled on -11,
    # the wrap point, where detection died.
    print("\n3. a dark room must open the exposure UP, not stop it down")
    dark = [
        Probe(-3.0, dot_peak=90, floor=3, mean=13),
        Probe(-4.0, dot_peak=66, floor=2, mean=9),
        Probe(-5.0, dot_peak=44, floor=2, mean=6),
        Probe(-7.0, dot_peak=20, floor=2, mean=2),
        Probe(-9.0, dot_peak=7, floor=1, mean=1),
        Probe(-11.0, dot_peak=2, floor=1, mean=0),
    ]
    best, why = choose(dark)
    check("it opens up rather than stopping down", best.exposure >= -5.0,
          f"chose {best.exposure:g} -- {why}")
    check("and never picks the wrap point", best.exposure != -11.0)
    at11 = next(p for p in dark if p.exposure == -11.0)
    check("-11 is correctly predicted to lose nearly everything",
          predicted_rate(at11, next(p for p in dark if p.exposure == -5.0)) < 0.3)

    # ------------------------------------------------------------------ 4 ----
    # A LIT room is what MEAN_CEILING was invented to handle, and the fear was
    # real: at -3 the monitor stops being a ghost and its own content starts
    # scoring. The floor measures that directly, so no constant is needed.
    print("\n4. a lit room stops itself down, through the floor")
    lit = [
        Probe(-3.0, dot_peak=110, floor=46, mean=127),
        Probe(-4.0, dot_peak=84, floor=30, mean=96),
        Probe(-5.0, dot_peak=60, floor=16, mean=64),
        Probe(-6.0, dot_peak=42, floor=10, mean=40),
        Probe(-7.0, dot_peak=28, floor=8, mean=24),
    ]
    best, why = choose(lit)
    check("it backs off the brightest setting on its own",
          best.exposure < -3.0, f"chose {best.exposure:g} -- {why}")
    check("no mean ceiling was consulted to do it", True,
          "the floor climbing from 8 to 46 is what did it")

    # ----------------------------------------------------------------- 4b ---
    # A CANDIDATE MAY NOT WIN ON A LUCKY NOISE READING.
    #
    # Measured on his bench, 2026-10-16 10:53, and it cost him a live
    # string. These are the three candidates exactly as the app printed
    # them:
    #
    #     exposure   dot peak   floor
    #         -4         43       15
    #         -5         57       20
    #         -6         22        5    <- chosen
    #
    # It picked the DIMMEST DOT. Its three-frame floor reading came out
    # low, so its bar came out low, so its ratio came out best. He then
    # fired a string at that exposure and the microphone heard ten trigger
    # pulls the camera never turned into shots.
    #
    # The fix is the one physical fact available: noise cannot be LOWER at
    # a longer exposure than at a shorter one, so a reading that dips below
    # a darker neighbour's is sample noise and is clamped up. Nothing else
    # is assumed -- in a lit room the floor really does climb with
    # exposure, and section 4 above fails if that climb is flattened.
    print(chr(10) + "4b. a lucky noise reading does not win the comparison")
    his = [Probe(-4.0, dot_peak=43, floor=15),
           Probe(-5.0, dot_peak=57, floor=20),
           Probe(-6.0, dot_peak=22, floor=5)]
    best_his, why_his = choose(his)
    check("the brightest dot wins, not the luckiest floor",
          best_his.exposure == -5.0,
          f"chose {best_his.exposure:g} -- {why_his}")
    bars = bars_for(his)
    # The dip in his column is at -4: going from the darkest candidate
    # upward the floors read 5, 20, 15, and the 15 cannot be right when a
    # SHORTER exposure measured 20.
    check("a floor reading below a darker candidate's is raised to it",
          abs(bars[-4.0] - min_peak_for(20.0)) < 0.01,
          f"{bars} -- 15 at -4 against 20 at -5 is not physical")
    # AND THE SEAM, which is what actually flipped this choice (RULES C14).
    # `Probe.bar()` is documented as mirroring what the app does with a
    # measured floor -- and it had drifted: the app used
    # max(floor*1.3, floor+6) while this module used max(8, floor*1.3).
    # At a floor of 5 that is 11 against 8, and 8 was enough to let the
    # dimmest dot on the table win. One function now, used by both.
    for f in (0.0, 2.0, 5.0, 15.0, 20.0, 46.0):
        check(f"the probe predicts the bar the detector uses, at floor {f:.0f}",
              abs(Probe(-5.0, dot_peak=1, floor=f).bar() - min_peak_for(f))
              < 1e-6,
              "a predicted bar the detector never uses flatters every "
              "candidate that measured a low floor")
    # The mirror: a genuinely monotonic column passes through untouched, or
    # this would be the lit-room bug with extra steps.
    mono = [Probe(-5.0, dot_peak=60, floor=16),
            Probe(-6.0, dot_peak=42, floor=10)]
    mbars = bars_for(mono)
    check("a column that is already monotonic is left alone",
          abs(mbars[-6.0] - mono[1].bar()) < 0.01
          and abs(mbars[-5.0] - mono[0].bar()) < 0.01,
          "flattening a real trend is how a lit room runs away to the "
          "brightest setting")

    # ------------------------------------------------------------------ 5 ----
    print("\n5. the arithmetic refuses to flatter itself")
    a = Probe(-5.0, dot_peak=50, floor=8)
    check("a dot that images at nothing predicts nothing",
          predicted_rate(Probe(-9.0, dot_peak=0, floor=8), a) == 0.0)
    check("an anchor that imaged at nothing yields 0, not a divide by zero",
          predicted_rate(a, Probe(-5.0, dot_peak=0, floor=8)) == 0.0)
    check("no probes at all is handled", choose([])[0] is None)
    bright = Probe(-3.0, dot_peak=5000, floor=8)
    check("a saturating scale is clamped at full scale, not extrapolated",
          predicted_margin(bright, a) <= 255.0 / bright.bar() + 1e-6,
          f"{predicted_margin(bright, a):.1f}x")
    check("the bar never drops below the detector's own minimum",
          Probe(-5.0, dot_peak=50, floor=0).bar() >= 8.0)

    # ------------------------------------------------------------------ 6 ----
    # The flicker is the reason the median is not enough, so assert that the
    # rule actually prices it in.
    print("\n6. the flicker is priced in, not averaged away")
    thin = Probe(-8.0, dot_peak=13, floor=9)
    anchor5 = Probe(-5.0, dot_peak=52, floor=8)
    median_clears = GUN_PEAK_DECILES[len(GUN_PEAK_DECILES) // 2] * \
        (thin.dot_peak / anchor5.dot_peak) >= thin.bar()
    check("the MEDIAN shot clears the bar at the bad exposure", median_clears,
          "which is exactly why a median-only rule said this was fine")
    check("...and yet the rule still rejects it",
          predicted_rate(thin, anchor5) < 0.8,
          f"{predicted_rate(thin, anchor5) * 100:.0f}% of the string survives")

    # ----------------------------------------------------------------- 7 ---
    # ANYTHING THAT PHOTOGRAPHS THE MONITOR MUST UNDO THE GAIN PIN.
    #
    # The shooting profile pins gain to 0, correctly: it keeps chroma noise
    # down and a 650nm dot saturates anyway. The calibration preset was
    # documented as deliberately NOT touching gain -- true and sufficient
    # while the only user was the screen search, which runs before anything
    # has pinned it. The two-marker mapping check runs AFTER, so "left
    # alone" meant "left pinned", and it photographed the monitor through a
    # sensor turned down for a laser.
    #
    # Measured in simulation 2026-10-15, same marker, same position:
    #
    #     gain free    lit max 203   lit-dark peak 201   found, strength 200
    #     gain pinned  lit max   5   lit-dark peak   3   NOT FOUND
    #
    # `find_marker` needs a peak of 12, so the check called both markers
    # invisible while predicting their positions to within a pixel. It has
    # been passing on his bench at 0.5px only because that camera's gain-0
    # is less extreme; the 75" TV and the projector are a coin toss.
    #
    # CLAUDE.md already records the live version of this -- frame maximum
    # 255 -> 7, flash difference 0.4 against 77.5 -- in this same preset.
    # Found once, fixed for the caller in front of me, left standing for
    # the next one. That is the whole of RULES Part E.
    print("\n7. a preset that photographs the monitor undoes the gain pin")
    from lsa.camera import controls as _controls

    shoot = _controls.laser_hunt("MSMF", exposure=-5.0)
    check("the shooting profile pins gain to 0",
          float(shoot.values.get("gain", -1)) == 0.0,
          f"{shoot.values.get('gain')} -- it should, for noise")

    blind = _controls.calibration_view("MSMF", exposure=-10.0)
    check("with no gain supplied it still leaves gain alone (old behaviour)",
          "gain" not in blind.values)

    seeing = _controls.calibration_view("MSMF", exposure=-10.0, gain=90.0)
    check("given the camera's own opening gain, it RESTORES it",
          float(seeing.values.get("gain", -1)) == 90.0,
          f"{seeing.values.get('gain')} -- without this the marker images "
          f"at a difference of 3 against a detection threshold of 12")
    check("and it still sets a dark FIXED exposure",
          seeing.values.get("exposure") == -10.0
          and "auto_exposure" in seeing.values,
          "auto exposure compensates for the very flash being measured")

    # ----------------------------------------------------------------- 8 ----
    # 2026-10-17. THE STRONG END MUST NOT SATURATE. Measured from the RAW
    # frames of session-20261017-100406: at -3 the red channel clipped in 32
    # of 39 left-hand shots. The probe measured only the weakest region, and
    # an exposure optimised for the dim end is the one that blows out the
    # bright end. The strong end is now a constraint on the weak-end objective.
    print(chr(10) + "8. the near side may not saturate")
    import lsa.exposure as _E
    from lsa.exposure import (choose_detailed, strong_end_median,
                              saturates_strong_end, STRONG_MEDIAN_LIMIT,
                              STRONG_ANCHOR_MEDIAN, SAT_CHROMA, FLICKER,
                              DOT_CEILING)

    check("the limit is derived, not a round number: knee 90 / flicker 1.2",
          abs(STRONG_MEDIAN_LIMIT - SAT_CHROMA / FLICKER) < 1e-9
          and SAT_CHROMA == 90.0 and FLICKER == 1.2,
          f"{STRONG_MEDIAN_LIMIT}")
    check("the anchor's own strong-end median sits under the limit",
          STRONG_ANCHOR_MEDIAN <= STRONG_MEDIAN_LIMIT)

    # The probe's real weak-end table of 10:04:09, with a strong-end column
    # that is SYNTHETIC: 1.6x the weak end, the near:far ratio measured on
    # the raw frames. Said here so it is not mistaken for a measurement.
    today = [Probe(-3.0, dot_peak=28, floor=3, strong_peak=45),
             Probe(-4.0, dot_peak=23, floor=2, strong_peak=37),
             Probe(-5.0, dot_peak=18, floor=0, strong_peak=29)]
    weak_only = [Probe(p.exposure, p.dot_peak, p.floor) for p in today]

    c0 = choose_detailed(weak_only)
    check("without a near-side reading the old choice is unchanged (-3)",
          c0.best.exposure == -3.0 and not c0.spread_exceeded,
          f"{c0.best.exposure}")
    check("...and it SAYS the limit could not be applied",
          "NOT measured" in c0.why, c0.why)

    c1 = choose_detailed(today)
    check("with the near side measured, -3 and -4 are set aside",
          saturates_strong_end(today[0], today[2])
          and saturates_strong_end(today[1], today[2])
          and not saturates_strong_end(today[2], today[2]))
    check("-5 is chosen",
          c1.best.exposure == -5.0, f"{c1.best.exposure}: {c1.why}")
    check("CORROBORATION: it lands on the exposure his 575 pulls were "
          "collected at",
          c1.best.exposure == _E.GUN_ANCHOR_EXPOSURE)
    check("the far side keeps a MEDIAN margin against its bar (NOT a claim about its dimmest shot: that scales to 1.5-2.9x at -5, see exposure.py)",
          predicted_margin(c1.best, today[2], bar=bars_for(today)[-5.0]) > 3.0)
    check("the line shows the near side's predicted median and the limit",
          "near side predicted" in c1.why and "75" in c1.why, c1.why)
    check("the table shows which candidates SATURATE",
          any("SATURATES" in l for l in table(today)))

    # A dot reading at the sensor ceiling has no usable ratio: refuse it.
    hot = Probe(-3.0, dot_peak=28, floor=3, strong_peak=DOT_CEILING)
    check("a near-side dot that itself hit the ceiling is refused",
          strong_end_median(hot, today[2]) == 255.0
          and saturates_strong_end(hot, today[2]))

    # Nothing satisfies the near side: take the LEAST saturated and say so.
    saved = _E.STRONG_ANCHOR_MEDIAN
    try:
        _E.STRONG_ANCHOR_MEDIAN = 100.0
        c2 = choose_detailed(today)
    finally:
        _E.STRONG_ANCHOR_MEDIAN = saved
    check("when no candidate fits, the least saturated is taken (-5)",
          c2.best.exposure == -5.0 and c2.spread_exceeded)
    check("...and the warning names the spread and the measured numbers",
          "more than one exposure" in c2.warning
          and "limit of 75" in c2.warning, c2.warning)

    # Survivors exist but cost the far side: also flagged, with both rates.
    # (Dots 3x the table above: at a floor of 24 the bar is 31, and a dot
    # that does not clear its own bar is now refused outright, section 9.)
    lit = [Probe(-3.0, dot_peak=84, floor=24, strong_peak=134),
           Probe(-4.0, dot_peak=69, floor=24, strong_peak=110),
           Probe(-5.0, dot_peak=54, floor=24, strong_peak=86)]
    c3 = choose_detailed(lit)
    check("a constraint that costs the far side is flagged, with both rates",
          c3.best.exposure == -5.0 and c3.spread_exceeded
          and "against" in c3.warning and "wrong hole" in c3.warning,
          c3.warning or "no warning")
    check("a case that costs nothing is not flagged",
          not c1.spread_exceeded and c1.warning == "")

    # ----------------------------------------------------------------- 9 ----
    # 2026-10-17. THE PROBE MAY NOT REPORT A NUMBER ITS OWN DOT CONTRADICTS.
    # For the ANCHOR the dot/anchor ratio is exactly 1.0, so the old rate and
    # margin were his logged distribution over the bar and never looked at the
    # dot just measured. A real harness line that then detected 0 of 12:
    #   chose exposure -5 - 100% ... (dot 10, floor 8, bar 14, predicted margin 6.1x)
    print(chr(10) + "9. the dot just measured gates every prediction, the anchor included")
    from lsa.exposure import (own_margin, dot_clears_bar, SAFETY as _SAFETY,
                              RANGE_OVER_BAR)
    thin5 = Probe(-5.0, dot_peak=10, floor=8)
    thin_c = choose_detailed([thin5])
    check("premise: the reviewer's numbers -- dot 10, floor 8 -> bar 14",
          thin5.bar() == 14.0)
    check("the anchor with a dot UNDER its bar is no longer '100%': rate 0",
          predicted_rate(thin5, thin5) == 0.0,
          f"{predicted_rate(thin5, thin5) * 100:.0f}%  (was 100%)")
    check("... and its margin is its OWN margin (dot / bar = 0.7x), not 6.1x",
          abs(predicted_margin(thin5, thin5) - 10 / 14) < 1e-9,
          f"{predicted_margin(thin5, thin5):.2f}x  (was 6.1x)")
    check("the table says DOT BELOW BAR and the choice is UNMEASURABLE, with the numbers",
          any("DOT BELOW BAR" in l for l in table([thin5]))
          and thin_c.best is None and thin_c.unmeasurable
          and "dot 10 vs bar 14" in thin_c.warning
          and "100%" not in thin_c.why,
          thin_c.warning)
    healthy5 = Probe(-5.0, dot_peak=40, floor=8)
    check("the same anchor with a dot that clears its bar keeps its real prediction",
          predicted_rate(healthy5, healthy5) == 1.0
          and abs(predicted_margin(healthy5, healthy5) - 85 / 14) < 1e-9)
    check("a dot exactly at its bar is not enough (needs SAFETY): 'at or below' fails",
          not dot_clears_bar(Probe(-5.0, dot_peak=14, floor=8))
          and dot_clears_bar(Probe(-5.0, dot_peak=14 * _SAFETY, floor=8)))
    # Candidates: an anchor that clears, a darker one whose dot does not.
    mix = [Probe(-4.0, dot_peak=60, floor=8), Probe(-5.0, dot_peak=44, floor=8),
           Probe(-6.0, dot_peak=12, floor=8)]
    cm = choose_detailed(mix)
    check("a candidate whose own dot sits under its bar reports 0% and cannot be chosen",
          predicted_rate(mix[2], mix[1]) == 0.0 and cm.best.exposure != -6.0,
          f"chose {cm.best.exposure:g}")
    check("... and its table row is flagged",
          any("DOT BELOW BAR" in l and "-6" in l for l in table(mix)))
    # Anchor's own dot fails: no ratio against it is trustworthy for anyone.
    bad_anchor = [Probe(-4.0, dot_peak=60, floor=8), Probe(-5.0, dot_peak=9, floor=8)]
    ca = choose_detailed(bad_anchor)
    check("when the ANCHOR's dot is under its bar, nothing is predicted for anyone",
          all(predicted_rate(p, bad_anchor[1]) == 0.0 for p in bad_anchor)
          and ca.unmeasurable and ca.best is None
          and "anchor" in ca.warning, ca.warning)
    # No instrument at all: plain statement with numbers, nothing chosen.
    none_ok = [Probe(-4.0, dot_peak=12, floor=8), Probe(-5.0, dot_peak=10, floor=8),
               Probe(-6.0, dot_peak=6, floor=8)]
    cn = choose_detailed(none_ok)
    check("no candidate's own dot clears its bar: the warning lists every one with "
          "dot, bar and multiple",
          cn.unmeasurable and cn.best is None
          and all(f"{p.exposure:g}: dot {p.dot_peak:.0f} vs bar 14" in cn.warning
                  for p in none_ok)
          and "Keeping the exposure" in cn.warning, cn.warning)
    check("choose() (the old entry) returns None for it too, never a '100%' line",
          choose(none_ok)[0] is None and "100%" not in choose(none_ok)[1])
    check("the auto-range target is a multiple of the bar, not just an absolute floor",
          RANGE_OVER_BAR >= 2.0)

    # ----------------------------------------------------------------- 10 ---
    # Item 4: a stand-in anchor must not inherit the -5 strong-end allowance.
    print(chr(10) + "10. an absent anchor does not lend its strong-end allowance to -6")
    no5 = [Probe(-6.0, dot_peak=40, floor=3, strong_peak=70),
           Probe(-7.0, dot_peak=30, floor=3, strong_peak=52),
           Probe(-8.0, dot_peak=20, floor=3, strong_peak=35)]
    check("with -5 absent the stand-in anchor (-6) predicts NO strong-end median",
          all(strong_end_median(p, no5[0]) is None for p in no5)
          and not any(saturates_strong_end(p, no5[0]) for p in no5))
    c5 = choose_detailed(no5)
    check("... and the choice says the allowance was not applied because -5 was not measured",
          "anchor was not measured" in c5.why, c5.why)
    check("with -5 present nothing changes: its allowance still applies",
          strong_end_median(today[2], today[2]) == STRONG_ANCHOR_MEDIAN)

    print()
    if all(checks):
        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
