"""Where a shot is reliably detected: the measurement, enumerated.

The module is pure -- shots and a bar in, a rectangle and sentences out --
so this file can enumerate cases instead of sampling them, which is the
whole argument for keeping it free of Qt and of the lab.

Every case here is either one of HIS measurements or one of the ways this
project has already got a region-based measurement wrong.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa.playarea import (BINS, MARGIN, MIN_PER_BIN, MIN_PER_BIN_BLOCK,  # noqa: E402
                          MIN_SHOTS, PlayArea, bins_from_shots, measure)

SW, SH = 3840, 2160


def _col(i, peak, n=MIN_PER_BIN_BLOCK, row_span=BINS):
    """n shots of `peak` in every row of screen column `i`."""
    out = []
    for j in range(row_span):
        for k in range(n):
            out.append(((i + 0.5) * SW / BINS + k,
                        (j + 0.5) * SH / BINS, peak))
    return out


def main() -> int:
    checks: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              f"{(' -- ' + detail) if detail else ''}")

    print("=== where a shot is reliably detected ===")

    # ------------------------------------------------------------------ 1 ---
    print("\n1. HIS string of 2026-10-16 13:26, which is why this exists")
    # 90 detections, bar 8. Measured by region:
    #     left + middle   median peak 88   11.0x
    #     right third     median peak 19    2.4x
    # And from the firing line: "It mostly works except for the right side."
    shots = _col(0, 88) + _col(1, 88) + _col(2, 19)
    area, lines = measure(shots, bar=8.0, screen_size=(SW, SH))
    check("the right third is marked off-limits", area is not None
          and abs(area.x1 - 2 / 3) < 1e-9,
          f"x1={area.x1:.3f}" if area else "no area at all")
    check("  and the left two thirds stay in play", area is not None
          and area.x0 == 0.0 and area.y0 == 0.0 and area.y1 == 1.0,
          f"{area}" if area else "")
    check("  and it reports the numbers, not just a verdict",
          any("median peak 19" in ln and "2.4x" in ln for ln in lines),
          " | ".join(lines))
    check("  and it names what widens the border",
          any("camera" in ln and "square" in ln for ln in lines),
          "he can move the camera; that is his half of the job, and the "
          "alternative advice -- dim the room -- is the one thing he cannot "
          "do (RULES C4)")
    check("  and it says holes still appear outside it",
          any("holes still appear" in ln for ln in lines),
          "his instruction: targets confined, holes drawn wherever a shot "
          "lands. Silently dropping a real hit is indistinguishable from a "
          "detector that does not work")

    # ------------------------------------------------------------------ 2 ---
    print("\n2. nothing is claimed without evidence")
    area, lines = measure([], bar=8.0, screen_size=(SW, SH))
    check("no shots at all claims no border", area is None)
    check("  and says so, rather than implying the screen was measured",
          any("NOT measured" in ln for ln in lines), " | ".join(lines))
    few = [(SW / 2, SH / 2, 88)] * (MIN_SHOTS - 1)
    area, lines = measure(few, bar=8.0, screen_size=(SW, SH))
    check(f"fewer than {MIN_SHOTS} shots claims no border", area is None,
          f"{len(few)} shots")
    check("  and tells him what would produce one",
          any("Shoot" in ln for ln in lines), " | ".join(lines))

    # ------------------------------------------------------------------ 3 ---
    print("\n3. a region he has not shot at is NOT off-limits")
    # THE TRAP THIS CHECK EXISTS FOR. Excluding a region for lack of
    # evidence would shrink his play area as a punishment for not shooting
    # there -- and since targets then never go there, he never shoots there,
    # so the border could never grow back. A measurement that destroys the
    # evidence it needs is the sensitivity-learner deadlock, which cost this
    # project three sessions.
    shots = _col(0, 88) + _col(1, 88)       # nothing at all on the right
    area, lines = measure(shots, bar=8.0, screen_size=(SW, SH))
    check("an unshot region leaves the border wide open", area is None,
          f"{area} -- a bin with no evidence is not evidence of weakness")
    one_each = _col(2, 19, n=1)             # present but below MIN_PER_BIN
    area, lines = measure(shots + one_each, bar=8.0, screen_size=(SW, SH))
    check(f"and under {MIN_PER_BIN} shots in a bin still does not speak",
          area is None,
          "one stray scoring 23 was once the only sample in its bin, and "
          "the map it produced took the noise floor from 11 to 70")

    # --------------------------------------------------------------- 3b ---
    print("\n3b. fencing needs more evidence than speaking (2026-10-16)")
    # The median of three shots is its middle shot. He hit everywhere and
    # watched the playfield shrink to 22% on bins of three.
    good = _col(0, 88) + _col(1, 88)
    three_weak = _col(2, 19, n=MIN_PER_BIN)
    area, lines = measure(good + three_weak, bar=8.0, screen_size=(SW, SH))
    check(f"{MIN_PER_BIN} weak shots in a bin do NOT fence it", area is None,
          f"{area}")
    check("  and it says weak so far, with n of the number needed",
          any("weak so far" in ln and f"{MIN_PER_BIN} of {MIN_PER_BIN_BLOCK}" in ln
              and "not fenced yet" in ln for ln in lines), " | ".join(lines))
    six_weak = _col(2, 19, n=MIN_PER_BIN_BLOCK)
    area, lines = measure(good + six_weak, bar=8.0, screen_size=(SW, SH))
    check(f"{MIN_PER_BIN_BLOCK} weak shots in a bin DO fence it", area is not None
          and abs(area.x1 - 2 / 3) < 1e-9, f"{area}")
    mixed = good + _col(2, 19, n=MIN_PER_BIN_BLOCK, row_span=2) + [
        ((2 + 0.5) * SW / BINS + k, (2 + 0.5) * SH / BINS, 19)
        for k in range(MIN_PER_BIN)]
    area, lines = measure(mixed, bar=8.0, screen_size=(SW, SH))
    check("a thin weak bin beside a proven one is reported, not fenced",
          area is not None and any("weak so far" in ln for ln in lines),
          f"{area}")

    # ------------------------------------------------------------------ 4 ---
    print("\n4. the border never fences him into nothing")
    # If EVERY region is weak the answer is not a tiny box: it is that the
    # camera is in the wrong place. `_update_playable` already refuses to
    # confine the game to a sliver on the strength of one shaky fit; same
    # judgement, same reason.
    shots = _col(0, 10) + _col(1, 10) + _col(2, 10)
    area, lines = measure(shots, bar=8.0, screen_size=(SW, SH))
    check("every region weak claims no border at all", area is None)
    check("  and says it is a camera placement problem",
          any("placement" in ln for ln in lines), " | ".join(lines))

    # ------------------------------------------------------------------ 5 ---
    print("\n5. a healthy screen is not fenced either")
    shots = _col(0, 88) + _col(1, 120) + _col(2, 95)
    area, lines = measure(shots, bar=8.0, screen_size=(SW, SH))
    check("all regions strong claims no border", area is None)
    check("  and says the whole screen is in play",
          any("whole screen is in play" in ln for ln in lines),
          " | ".join(lines))

    # ------------------------------------------------------------------ 6 ---
    print("\n6. the margin is the measured one, and the bar moves with it")
    # The bar is not a constant: it is 1.3x the measured noise floor, so it
    # was 8 on one of his launches and 18 on another, half an hour apart.
    # The same region must therefore change verdict with the bar, which is
    # what makes this a measurement rather than a stored opinion.
    shots = _col(0, 88) + _col(1, 88) + _col(2, 30)
    area, _ = measure(shots, bar=8.0, screen_size=(SW, SH))
    check(f"peak 30 against bar 8 is {30 / 8:.1f}x, so it stays in play",
          area is None, f"{area}")
    area, _ = measure(shots, bar=18.0, screen_size=(SW, SH))
    check(f"the same region against bar 18 is {30 / 18:.1f}x, so it is out",
          area is not None and abs(area.x1 - 2 / 3) < 1e-9, f"{area}")
    check(f"and the margin required is {MARGIN:.0f}x, from his own data",
          MARGIN == 3.0,
          "'below 3x the far side starts losing shots; below 2x they are "
          "lost everywhere' -- and selfcheck.MARGIN_READY is 3.0")

    # ------------------------------------------------------------------ 7 ---
    print("\n7. a weak BOTTOM strip, not just a weak side")
    # The first version of this module only ever cut columns, because his
    # screen happens to fail on the right. The camera could as easily miss
    # the bottom -- these notes record a bench where it saw no bottom strip
    # at all -- and a rule that only works in the direction it was written
    # in is the proxy mistake in a new coat.
    shots = []
    for i in range(BINS):
        for j in range(BINS):
            peak = 15 if j == BINS - 1 else 90
            for k in range(MIN_PER_BIN_BLOCK):
                shots.append(((i + 0.5) * SW / BINS + k,
                              (j + 0.5) * SH / BINS, peak))
    area, lines = measure(shots, bar=8.0, screen_size=(SW, SH))
    check("a weak bottom row is cut, full width", area is not None
          and abs(area.y1 - 2 / 3) < 1e-9 and area.x0 == 0.0
          and area.x1 == 1.0, f"{area}")

    # ------------------------------------------------------------------ 8 ---
    print("\n8. a hole in the middle cannot be papered over")
    # One weak bin in the CENTRE. No rectangle can contain the rest, so the
    # largest clear box is whatever survives -- and it must contain no weak
    # bin at all, because the border's whole promise is that inside it
    # detection is reliable. A box that quietly included the hole would be
    # worse than no box.
    shots = []
    for i in range(BINS):
        for j in range(BINS):
            peak = 12 if (i, j) == (1, 1) else 90
            for k in range(MIN_PER_BIN_BLOCK):
                shots.append(((i + 0.5) * SW / BINS + k,
                              (j + 0.5) * SH / BINS, peak))
    area, lines = measure(shots, bar=8.0, screen_size=(SW, SH))
    check("the weak centre bin is never inside the border", area is not None
          and not (area.x0 <= 1 / 3 < area.x1 and area.y0 <= 1 / 3 < area.y1
                   and area.x1 > 1 / 3 and area.y1 > 1 / 3
                   and area.x0 < 2 / 3 and area.y0 < 2 / 3
                   and (area.x1 - area.x0) > 1 / 3
                   and (area.y1 - area.y0) > 1 / 3), f"{area}")
    if area is not None:
        # Re-derived from the area itself rather than asserted by eye.
        inside = [(i, j) for i in range(BINS) for j in range(BINS)
                  if area.x0 <= i / BINS and (i + 1) / BINS <= area.x1
                  and area.y0 <= j / BINS and (j + 1) / BINS <= area.y1]
        check("  and every bin inside it is one that measured strong",
              (1, 1) not in inside and len(inside) >= 3, f"bins {inside}")

    # ------------------------------------------------------------------ 9 ---
    print("\n9. it survives disk, because it is a property of the pose")
    area = PlayArea(0.0, 0.0, 2 / 3, 1.0)
    back = PlayArea.from_dict(area.to_dict())
    check("round trips through a dict", back == area, f"{back}")
    check("a missing or malformed record reads as 'never measured'",
          PlayArea.from_dict(None) is None
          and PlayArea.from_dict({"x0": 0}) is None
          and PlayArea.from_dict({"x0": 0, "y0": 0, "x1": 0, "y1": 0}) is None,
          "a zero-area border would fence him into nothing, so it is "
          "rejected rather than honoured")
    check("the whole screen is reported as such", PlayArea(0, 0, 1, 1).whole_screen)
    check("and the fraction is the area, for reporting",
          abs(PlayArea(0, 0, 2 / 3, 1).fraction - 2 / 3) < 1e-9)

    # ----------------------------------------------------------------- 10 ---
    print("\n10. binning is in SCREEN space, over the screen it is given")
    cells = bins_from_shots([(10, 10, 50), (20, 20, 70), (SW - 10, SH - 10, 90)],
                            (SW, SH))
    check("shots land in the bin their screen position names",
          {(c.col, c.row) for c in cells} == {(0, 0), (BINS - 1, BINS - 1)},
          f"{[(c.col, c.row, c.shots) for c in cells]}")
    check("  and a bin reports the MEDIAN, not the brightest",
          next(c.median_peak for c in cells if (c.col, c.row) == (0, 0)) == 70,
          "one bright flyer must not speak for a ninth of the screen")
    check("a zero-sized screen is refused rather than divided by",
          bins_from_shots([(1, 1, 50)], (0, 0)) == [])

    # ----------------------------------------------------------------- 11 ---
    print("\n11. the border is a per-SESSION measurement: disk never restores it")
    # 2026-10-16 18:45:57: calibration.json was saved with the left third of
    # the screen, measured from phantom detections. Loading it would have
    # fenced him into a third of his monitor on the next launch.
    import numpy as np
    from lsa.calibrate.mapping import Mapping
    m = Mapping(homography=np.eye(3), screen_size=(SW, SH))
    m.play_area = PlayArea(0.0, 0.0, 1 / 3, 2 / 3)
    d = m.to_dict()
    check("the evidence is still written to disk", d.get("play_area") is not None)
    back = Mapping.from_dict(d)
    check("but it comes back with NO play area", back.play_area is None,
          f"{back.play_area}")

    print()
    if all(checks):
        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
