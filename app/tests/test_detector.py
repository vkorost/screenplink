"""Synthetic tests for the detector and shot tracker.

No camera needed. We fabricate what a LaserLyte pulse is supposed to look like
at 30 fps -- a saturated red dot present for three consecutive frames on top of
a dim ghost of the monitor -- and check that the pipeline turns it into exactly
one shot at the right place, and recovers the 100 ms pulse length.

The two-gun cases are the ones worth having: a detector that only ever reports
its brightest blob looks perfect in single-shot tests and silently loses half
your hits the moment you shoot with both hands.

Run:  .venv\\Scripts\\python.exe tests\\test_detector.py
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lsa.detect.laser import (  # noqa: E402
    Detection,
    DetectorParams,
    LaserDetector,
    ScoreMode,
    ShotTracker,
    amplified_min_area,
    full_frame_to_working_area,
)

FPS = 30.0
PERIOD = 1.0 / FPS
PERIOD_MS = PERIOD * 1000.0


def make_frame(dots: list[tuple[int, int]], oblique: bool = False) -> np.ndarray:
    """Dim screen ghost, a dull red decoy panel, and zero or more laser dots.

    oblique stretches each dot horizontally, which is what an off-axis camera
    actually sees: the spot images as an ellipse, not a circle.
    """
    f = np.full((480, 640, 3), 28, np.uint8)
    f[200:260, 100:200] = (40, 40, 95)          # dull red element: a decoy
    cv2.putText(f, "TARGET", (380, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
    for cx, cy in dots:
        if oblique:
            cv2.ellipse(f, (cx, cy), (11, 4), 0, 0, 360, (110, 120, 255), -1)
            cv2.ellipse(f, (cx, cy), (5, 2), 0, 0, 360, (255, 255, 255), -1)
        else:
            cv2.circle(f, (cx, cy), 5, (110, 120, 255), -1)    # bloom skirt, strongly red
            cv2.circle(f, (cx, cy), 2, (255, 255, 255), -1)    # saturated core, no chroma
    return f


def run(frames: list[list[tuple[int, int]]], params: DetectorParams,
        tracker: ShotTracker, oblique: bool = False):
    """Play a scripted sequence of dot positions through the pipeline."""
    det = LaserDetector(params)
    shots = []
    frames_with_dets = 0
    max_dets = 0
    t = 0.0
    for i, dots in enumerate(frames):
        found = det.process(make_frame(dots, oblique), timestamp=t)
        if found:
            frames_with_dets += 1
        max_dets = max(max_dets, len(found))
        # Final reports only; the provisional queue is drained and ignored so
        # these tests keep measuring one outcome per shot.
        shots.extend(tracker.feed(found, i))
        tracker.take_provisional()
        t += PERIOD
    shots.extend(tracker.flush())
    return det, shots, frames_with_dets, max_dets


def blank(n: int) -> list[list[tuple[int, int]]]:
    return [[] for _ in range(n)]


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return condition


def new_tracker(**kw) -> ShotTracker:
    opts = dict(min_frames=2, max_gap_frames=2, max_move_px=40.0,
                frame_period_ms=PERIOD_MS, max_tracks=4, max_duration_ms=400.0)
    opts.update(kw)
    return ShotTracker(**opts)


# max_area deliberately left at the production default: a laser dot measures
# 7-60 px and the false positives that got through live were 200-600 px, so the
# size limit is part of what is being tested, not something to opt out of.
BASE = DetectorParams(
    mode=ScoreMode.CHROMA, threshold=40, min_peak=50,
    blur=3, min_area=2, use_background=True, bg_alpha=0.05,
    max_blobs=4, min_separation=10.0,
)


# --- size evidence where the sensitivity map amplified the score -----------
#
# 2026-10-17: the saved map multiplies the weakest region of the screen by up
# to 4, so noise that scored 3 to 5 is reported at 10 to 19 against a bar of
# 11, and 445 "shots" were logged in one session, most of them not his. The
# map amplifies the SCORE; the blob it came from is untouched, so where the
# score was amplified the blob must be big enough to be a dot.
# `recordings/session-20261017-084337`, microphone as ground truth: shots a
# click confirmed covered 130 px at the smallest, detections nothing confirmed
# covered 12 px at the median.

GATE_BASE = dict(mode=ScoreMode.CHROMA, threshold=10, min_peak=12, blur=0,
                 close_iter=0, min_area=2, use_background=False,
                 max_blobs=4, min_separation=10.0)


def _gate_frame(blocks) -> np.ndarray:
    f = np.full((480, 640, 3), 30, np.uint8)
    for x, y, side, bgr in blocks:
        f[y:y + side, x:x + side] = bgr
    return f


FAINT = (30, 30, 36)      # raw chroma 6: noise-level, but 24 once amplified x4
BRIGHT = (30, 30, 60)     # raw chroma 30: passes with no amplification at all


def _find(params: DetectorParams, gain, frame: np.ndarray):
    det = LaserDetector(params)
    if gain is not None:
        det.set_gain_map(gain)
    return det, det.process(frame, timestamp=0.0)


def amp_area_gate_checks() -> bool:
    ok = True
    print("\n30. Size evidence where the score was amplified (the 2026-10-17 phantoms)")
    P = DetectorParams(**GATE_BASE)

    # The rule itself, as a pure function.
    ok &= check("N = 1: exactly min_area, as before",
                amplified_min_area(P, 1.0) == 2.0, f"{amplified_min_area(P, 1.0)}")
    ok &= check("N = 1.1: still min_area (the rule is silent where the map barely acts)",
                amplified_min_area(P, 1.1) == 2.0, f"{amplified_min_area(P, 1.1)}")
    ok &= check("N = 2 asks 16 px, N = 3 asks 32, N = 4 asks 48 (16 per unit above 1)",
                (amplified_min_area(P, 2.0), amplified_min_area(P, 3.0),
                 amplified_min_area(P, 4.0)) == (16.0, 32.0, 48.0),
                f"{[amplified_min_area(P, n) for n in (2.0, 3.0, 4.0)]}")
    off = DetectorParams(**{**GATE_BASE, "amp_area_per_gain": 0.0})
    ok &= check("amp_area_per_gain = 0 switches it off",
                amplified_min_area(off, 4.0) == 2.0)
    half = DetectorParams(**{**GATE_BASE, "scale": 0.5})
    ok &= check("UNITS: at scale 0.5 a blob has a quarter of the pixels, so the ask is a quarter",
                amplified_min_area(half, 4.0) == 12.0, f"{amplified_min_area(half, 4.0)}")
    ok &= check("UNITS: a larger min_area still wins where the map asks for less",
                amplified_min_area(DetectorParams(**{**GATE_BASE, "min_area": 60}), 4.0) == 60.0)

    # Through the detector.
    speck = _gate_frame([(300, 200, 3, FAINT)])          # area 9
    dot = _gate_frame([(300, 200, 10, FAINT)])           # area 100
    flat4 = np.full((480, 640), 0.25, np.float32)        # N = 4 everywhere
    d0, f0 = _find(P, None, _gate_frame([(300, 200, 3, BRIGHT)]))
    ok &= check("no map: a 9 px blob is detected exactly as before",
                len(f0) == 1, f"{len(f0)}")
    d1, f1 = _find(P, flat4, speck)
    ok &= check("N = 4: the same 9 px speck, amplified to a peak of 24, is NOT a dot",
                len(f1) == 0, f"{len(f1)}")
    ok &= check("it is counted as amp_area, not as 'dim' (it is not too faint)",
                d1.last_rejects.get("amp_area") == 1 and d1.last_rejects.get("dim") == 0,
                f"{d1.last_rejects}")
    d2, f2 = _find(P, flat4, dot)
    ok &= check("N = 4: a 100 px blob at the same brightness still is",
                len(f2) == 1 and f2[0].peak >= 20, f"{len(f2)} {[d.peak for d in f2]}")
    d3, f3 = _find(off, flat4, speck)
    ok &= check("with the rule switched off the speck is detected again",
                len(f3) == 1, f"{len(f3)}")
    flat11 = np.full((480, 640), 1 / 1.1, np.float32)
    d4, f4 = _find(P, flat11, _gate_frame([(300, 200, 3, BRIGHT)]))
    ok &= check("N = 1.1: a small bright blob is untouched",
                len(f4) == 1, f"{len(f4)}")

    # LOCAL amplification, not screen position: the left half is unmapped.
    split = np.ones((480, 640), np.float32)
    split[:, 320:] = 0.25
    two = _gate_frame([(100, 200, 3, BRIGHT), (500, 200, 3, BRIGHT)])
    d5, f5 = _find(P, split, two)
    ok &= check("same size, same frame: the unamplified side is kept, the amplified side is not",
                len(f5) == 1 and f5[0].x < 320, f"{[(round(d.x), d.peak) for d in f5]}")
    flipped = np.ones((480, 640), np.float32)
    flipped[:, :320] = 0.25
    d6, f6 = _find(P, flipped, two)
    ok &= check("move the amplification to the other side and the verdicts swap "
                "(it follows the map, not the position)",
                len(f6) == 1 and f6[0].x > 320, f"{[(round(d.x), d.peak) for d in f6]}")

    # Units through the detector, at a reduced working scale.
    hs = DetectorParams(**{**GATE_BASE, "scale": 0.5})
    d7, f7 = _find(hs, flat4, _gate_frame([(300, 200, 6, FAINT)]))     # 36 full-frame px
    ok &= check("scale 0.5, N = 4: a 36 full-frame px blob is below the 48 asked",
                len(f7) == 0, f"{len(f7)}")
    d8, f8 = _find(hs, flat4, _gate_frame([(300, 200, 8, FAINT)]))     # 64 full-frame px
    ok &= check("scale 0.5, N = 4: a 64 full-frame px blob is above it, and reports 64",
                len(f8) == 1 and abs(f8[0].area - 64.0) < 1.0,
                f"{[(d.area) for d in f8]}")

    # The blob the gate refused is REMEMBERED, because it is thrown away before
    # its peak exists and `last_dim_*` never see it: without this a shot the
    # gate refused was reported as "the camera saw nothing at any level".
    g1 = d1.last_amp_gate
    ok &= check("N = 4, 9 px speck: the gated blob is kept as (area, ask, N, x, y), full-frame units",
                g1 is not None and abs(g1[0] - 9) < 0.5 and abs(g1[1] - 48) < 0.5
                and abs(g1[2] - 4.0) < 0.01 and abs(g1[3] - 301.5) < 1.0
                and abs(g1[4] - 201.5) < 1.0, f"{g1}")
    ok &= check("... and it is not mistaken for a faint blob (no dim peak, no dim position)",
                d1.last_dim_peak == 0 and d1.last_dim_pos is None)
    ok &= check("a blob that passes leaves nothing gated", d2.last_amp_gate is None, f"{d2.last_amp_gate}")
    ok &= check("no map, no gate, nothing to remember", d0.last_amp_gate is None)
    g7 = d7.last_amp_gate
    ok &= check("scale 0.5: the report is in FULL-FRAME px (36 seen, 48 asked), not working-scale",
                g7 is not None and abs(g7[0] - 36) < 1.0 and abs(g7[1] - 48) < 1.0
                and abs(g7[3] - 303.0) < 1.5 and abs(g7[4] - 203.0) < 1.5, f"{g7}")
    d1.process(_gate_frame([]), timestamp=PERIOD)
    ok &= check("and the next frame forgets it", d1.last_amp_gate is None)
    two_g = _gate_frame([(100, 200, 3, FAINT), (500, 300, 5, FAINT)])    # 9 px and 25 px
    d9, _ = _find(P, flat4, two_g)
    ok &= check("with two gated blobs the LARGER is reported (the likelier shot)",
                d9.last_amp_gate is not None and abs(d9.last_amp_gate[0] - 25) < 0.5
                and d9.last_amp_gate[3] > 400, f"{d9.last_amp_gate}")

    # THE SEAM, tested across (RULES C14): the replay tool converts a csv area
    # with `full_frame_to_working_area`, the detector asks via
    # `amplified_min_area`; they must agree at every scale or --from-csv lies.
    for sc in (1.0, 0.75, 0.5, 0.25):
        pp = DetectorParams(**{**GATE_BASE, "scale": sc})
        for n in (1.5, 2.0, 3.0, 4.0):
            ask_full = pp.amp_area_per_gain * (n - 1.0)
            ok &= check(f"seam: scale {sc}, N {n}: full-frame ask {ask_full:.0f} px, converted, "
                        f"IS amplified_min_area",
                        abs(max(float(pp.min_area), full_frame_to_working_area(pp, ask_full))
                            - amplified_min_area(pp, n)) < 1e-6,
                        f"{full_frame_to_working_area(pp, ask_full)} vs {amplified_min_area(pp, n)}")
    return ok


def _red(chroma: int) -> tuple:
    """BGR whose chroma score (red minus the larger of green and blue) is `chroma`."""
    return (30, 30, 30 + chroma)


def oversize_checks() -> bool:
    """A blob over max_area is a question, not a rejection (2026-10-17, near side).

    Live: 7 of 13 shot-bright blobs (peak 98-109) discarded as 'too big', 1673 to
    9665 px, while specks of peak 9-24 got the holes. Over four recorded sessions
    every oversize event with peak >= 30 lasted 17-239 ms bar one steady object
    (1371 ms), and every oversize event under 30 was faint (8-16).
    """
    ok = True
    print("\n32. Oversize blobs: bright ones are shots, faint ones are not")
    P = DetectorParams(**GATE_BASE)
    ok &= check("defaults: oversize_min_peak 30 (the measured laser/noise boundary), core 0.7",
                (P.oversize_min_peak, P.oversize_core_frac) == (30, 0.7),
                f"{(P.oversize_min_peak, P.oversize_core_frac)}")

    bloom = (100, 100, 50, _red(90))                          # 2500 px > max_area 1500
    d, f = _find(P, None, _gate_frame([bloom]))
    ok &= check("a bright (peak 90) 2500 px blob is a DETECTION, not an 'area' rejection",
                len(f) == 1 and f[0].area == 2500 and d.last_rejects.get("area") == 0
                and d.last_big_rejected is None,
                f"{len(f)} detections, rejects {d.last_rejects}")
    old = DetectorParams(**{**GATE_BASE, "oversize_min_peak": 0})
    d0, f0 = _find(old, None, _gate_frame([bloom]))
    ok &= check("oversize_min_peak=0 restores the absolute ceiling (no detection, counted 'area')",
                len(f0) == 0 and d0.last_rejects.get("area") == 1, f"{len(f0)} {d0.last_rejects}")

    # --- still one illumination event: the lumps fold into the ACCEPTED bloom ---
    lumps = [(170, 120, 8, _red(14)), (60, 170, 8, _red(14)), (180, 70, 8, _red(14))]
    d3, f3 = _find(P, None, _gate_frame([bloom] + lumps))
    ok &= check("the halo rule still sees the bloom as the event: three lumps folded, ONE detection",
                len(f3) == 1 and d3.last_rejects.get("halo") == 3 and f3[0].area == 2500,
                f"{len(f3)} detections, rejects {d3.last_rejects}")

    # --- faint and big: not a pulse of this laser ---------------------------------
    patch = (100, 100, 50, _red(20))                          # peak 20: over min_peak 12, under 30
    dp, fp = _find(P, None, _gate_frame([patch]))
    ok &= check("a FAINT (peak 20) 2500 px patch is still rejected, counted 'area'",
                len(fp) == 0 and dp.last_rejects.get("area") == 1, f"{len(fp)} {dp.last_rejects}")
    ok &= check("...and remembered for the heard-but-not-seen report (peak, position, area)",
                dp.last_big_rejected is not None and dp.last_big_rejected[0] == 20
                and abs(dp.last_big_rejected[1] - 124.5) < 1.5
                and dp.last_big_rejected[3] == 2500, f"{dp.last_big_rejected}")
    dq, fq = _find(P, None, _gate_frame([(100, 100, 50, _red(8))]))   # under min_peak: noise
    ok &= check("a blob under min_peak is not 'big and faint', it is just noise: no evidence slot",
                dq.last_big_rejected is None and len(fq) == 0, f"{dq.last_big_rejected}")
    nxt = dp.process(_gate_frame([]), timestamp=PERIOD)
    ok &= check("...and the slot is forgotten on the next frame", dp.last_big_rejected is None)

    # --- the position of a bloom is its CORE, not the halo-dragged centroid ------
    halo = (100, 100, 60, _red(40))                           # 3600 px, dull
    f = _gate_frame([halo])
    f[100:110, 100:110] = _red(120)                           # a bright core in the corner
    dc, fc = _find(P, None, f)
    flat = DetectorParams(**{**GATE_BASE, "oversize_core_frac": 0.0})
    dw, fw = _find(flat, None, f)
    ok &= check("bloom position = its core (centre of the 10x10 bright patch, 104.5)",
                len(fc) == 1 and abs(fc[0].x - 104.5) < 1.0 and abs(fc[0].y - 104.5) < 1.0,
                f"{(fc[0].x, fc[0].y) if fc else None}")
    ok &= check("...which differs from the halo-dragged weighted centroid (the reason for the rule)",
                len(fw) == 1 and fw[0].x - fc[0].x > 8.0,
                f"weighted {(fw[0].x, fw[0].y) if fw else None}")
    small = _gate_frame([(100, 100, 30, _red(40))])
    small[100:106, 100:106] = _red(120)
    ds1, fs1 = _find(P, None, small)
    ds2, fs2 = _find(flat, None, small)
    ok &= check("a NORMAL-size blob is positioned exactly as before (the core rule is only for blooms)",
                len(fs1) == 1 and len(fs2) == 1 and (fs1[0].x, fs1[0].y) == (fs2[0].x, fs2[0].y),
                f"{(fs1[0].x, fs1[0].y)} vs {(fs2[0].x, fs2[0].y)}")

    # --- the report chain: a shot lost to size is not 'saw nothing' ---------------
    from lsa.camera.evidence import FrameEvidence
    from lsa.shotlog import (MISSED_NOTHING, MISSED_TOO_BIG, classify_missed_pulse,
                             describe_missed_pulse, missed_pulse_note)
    ev = FrameEvidence()
    ev.push(10.0, 20, 0, None, 0, 0, big=(20, 124.5, 124.5, 2500.0))
    summ = ev.summarise(9.9, 10.1)
    ok &= check("the evidence ring carries the oversize blob",
                summ is not None and summ["big"] == {"peak": 20, "pos": (124.5, 124.5), "area": 2500.0},
                f"{summ}")
    ok &= check("heard-but-not-seen: the branch is TOO_BIG, not 'nothing'",
                classify_missed_pulse(summ, 10.0) == MISSED_TOO_BIG,
                f"{classify_missed_pulse(summ, 10.0)}")
    ev2 = FrameEvidence()
    ev2.push(10.0, 3, 0, None, 0, 0)
    ok &= check("with no oversize blob in the window the old branch is unchanged",
                classify_missed_pulse(ev2.summarise(9.9, 10.1), 10.0) == MISSED_NOTHING)
    txt = describe_missed_pulse(summ, 8.0, 10.0) + " | " + missed_pulse_note(summ, 8.0, 10.0)
    ok &= check("the log line and the screen note both name the oversize blob and its size",
                "very large" in txt and "2500px" in txt and txt.lower().count("saw nothing") == 0,
                txt[:200])
    return ok


def halo_checks() -> bool:
    """One illumination event is one shot (2026-10-17, the left of the screen).

    The evidence: a 187x225 px halo (area 20403, peak 79) with the dot inside
    it, rejected by `max_area`; seven or twelve detached lumps of it, peak 8-10
    against a bar of 8, each of which became a shot, a hole and a gunshot.
    """
    ok = True
    print("\n31. Halo lumps: one illumination event is one shot")
    # `oversize_min_peak=0` restores the absolute ceiling: this section tests the
    # path where the bright event is an area-REJECTED blob and not in `dets`.
    # Since 2026-10-17 a bloom that bright is a detection (section 32); the
    # rejected-event path is what remains for a blob that is big, over min_peak
    # and still under oversize_min_peak.
    P = DetectorParams(**{**GATE_BASE, "oversize_min_peak": 0})
    off = DetectorParams(**{**GATE_BASE, "halo_ratio": 0.0, "oversize_min_peak": 0})
    ok &= check("defaults: ratio 0.30, radius 3.0 x the event's diagonal",
                (P.halo_ratio, P.halo_radius_diag) == (0.30, 3.0),
                f"{(P.halo_ratio, P.halo_radius_diag)}")

    # --- the failing case: the bright event is OVERSIZE and not in `dets` ----
    bloom = (100, 100, 50, _red(90))                      # 2500 px > max_area 1500
    lumps = [(170, 120, 8, _red(14)), (60, 170, 8, _red(14)), (180, 70, 8, _red(14))]
    frame = _gate_frame([bloom] + lumps)
    d_off, f_off = _find(off, None, frame)
    ok &= check("BEFORE (halo rule off): the bloom is area-rejected and the three lumps "
                "each become a detection -- the fault",
                len(f_off) == 3 and d_off.last_rejects.get("area") == 1,
                f"{len(f_off)} detections, rejects {d_off.last_rejects}")
    d_on, f_on = _find(P, None, frame)
    ok &= check("AFTER: the three lumps are folded into the bloom, none is a detection",
                len(f_on) == 0, f"{len(f_on)} detections")
    ok &= check("...counted under their own reason, 'halo', not 'dim' or 'area'",
                d_on.last_rejects.get("halo") == 3 and d_on.last_rejects.get("dim") == 0
                and d_on.last_rejects.get("area") == 1, f"{d_on.last_rejects}")
    ok &= check("...and the brightest folded blob is remembered for the heard-not-seen report",
                d_on.last_halo is not None and d_on.last_halo[0] == 14,
                f"{d_on.last_halo}")

    # --- the event's own core is not a lump (session 143752 frames 1612-1614) ---
    core_frame = _gate_frame([bloom])
    core_frame[112:138, 112:138] = 30                  # the saturated core has no chroma
    core_frame[122:126, 122:126] = _red(23)            # ...but a speck survives in it
    d_core, f_core = _find(P, None, core_frame)
    ok &= check("a dim speck AT THE CENTRE of an oversize bloom is the event's own core, "
                "not a halo lump: it survives (it is the only report of that shot)",
                len(f_core) == 1 and d_core.last_rejects.get("halo") == 0,
                f"{len(f_core)} {d_core.last_rejects}")
    off_core = _gate_frame([bloom, (160, 123, 4, _red(23))])       # ~0.5 diagonals out
    ok &= check("...but the same speck 0.5 diagonals out IS folded",
                len(_find(P, None, off_core)[1]) == 0)

    # --- distance scales with the event's own extent -------------------------
    far = _gate_frame([bloom, (600, 420, 8, _red(14))])
    d_far, f_far = _find(P, None, far)
    ok &= check("a lump FAR outside 3 diagonals of the bloom survives",
                len(f_far) == 1 and d_far.last_rejects.get("halo") == 0,
                f"{len(f_far)} {d_far.last_rejects}")
    dot = (300, 200, 10, _red(90))                        # a normal dot, diag ~14
    near = _gate_frame([dot, (330, 200, 6, _red(20))])    # 30 px away, inside 3 x 14
    d_n, f_n = _find(P, None, near)
    ok &= check("a small dot's radius is small: a dim neighbour 30 px away IS folded",
                len(f_n) == 1 and d_n.last_rejects.get("halo") == 1,
                f"{len(f_n)} {d_n.last_rejects}")
    apart = _gate_frame([dot, (390, 200, 6, _red(20))])   # 90 px away, outside it
    d_a, f_a = _find(P, None, apart)
    ok &= check("...but the same dim neighbour 90 px away is NOT: the radius follows the "
                "event's size, so no constant covers a bloom and a dot",
                len(f_a) == 2, f"{len(f_a)}")

    # --- two pistols must survive -------------------------------------------
    gun2 = _gate_frame([dot, (330, 200, 10, _red(60))])   # ratio 0.67, 30 px away
    d_g, f_g = _find(P, None, gun2)
    ok &= check("TWO GUNS: a comparable second shot (ratio 0.67) survives even 30 px away",
                len(f_g) == 2, f"{len(f_g)}")
    gun_far = _gate_frame([dot, (500, 350, 10, _red(25))])  # ratio 0.28 but far away
    d_gf, f_gf = _find(P, None, gun_far)
    ok &= check("TWO GUNS: a dim second shot far away survives (the rule is local)",
                len(f_gf) == 2, f"{len(f_gf)}")
    edge_hi = _gate_frame([dot, (330, 200, 6, _red(33))])   # ratio 0.367 > 0.30
    edge_lo = _gate_frame([dot, (330, 200, 6, _red(24))])   # ratio 0.267 < 0.30
    ok &= check("the ratio boundary sits at 0.30: 0.37 survives",
                len(_find(P, None, edge_hi)[1]) == 2)
    ok &= check("...and 0.27 is folded",
                len(_find(P, None, edge_lo)[1]) == 1)

    # --- within ONE frame: no time-based lockout ----------------------------
    lone = _gate_frame([(170, 120, 8, _red(14))])
    det = LaserDetector(P)
    det.process(frame, timestamp=0.0)
    ok &= check("frame 1: folded", det.last_rejects.get("halo") == 3 and det.last_halo is not None)
    nxt = det.process(lone, timestamp=PERIOD)
    ok &= check("NO TIME-BASED LOCKOUT: the very next frame, the same lump alone is a "
                "detection again (it is no longer near anything brighter)",
                len(nxt) == 1, f"{len(nxt)}")
    ok &= check("...and last_halo forgets the previous frame", det.last_halo is None)

    # --- the report chain: a shot lost this way never says 'saw nothing' ------
    from lsa.camera.evidence import FrameEvidence
    from lsa.shotlog import (MISSED_HALO, MISSED_NOTHING, classify_missed_pulse,
                             describe_missed_pulse, missed_pulse_note)
    from lsa.camera.capture import Stats
    ok &= check("Stats has a rejected_halo counter", hasattr(Stats(), "rejected_halo"))
    ev = FrameEvidence()
    ev.push(10.0, 70, 0, None, 0, 0, halo=(14, 170.0, 120.0))
    summ = ev.summarise(9.9, 10.1)
    ok &= check("the evidence ring carries the halo-folded blob",
                summ is not None and summ["halo"] == {"peak": 14, "pos": (170.0, 120.0)},
                f"{summ}")
    ok &= check("heard-but-not-seen: the branch is HALO, not 'nothing'",
                classify_missed_pulse(summ, 10.0) == MISSED_HALO,
                f"{classify_missed_pulse(summ, 10.0)}")
    ev2 = FrameEvidence()
    ev2.push(10.0, 70, 0, None, 0, 0)
    ok &= check("with no halo in the window the old branch is unchanged",
                classify_missed_pulse(ev2.summarise(9.9, 10.1), 10.0) == MISSED_NOTHING)
    txt = describe_missed_pulse(summ, 8.0, 10.0) + " | " + missed_pulse_note(summ, 8.0, 10.0)
    ok &= check("the log line and the screen note both name the halo",
                "halo" in txt.lower() or "glow" in txt.lower(), txt[:160])
    return ok


def rejected_record_checks() -> bool:
    """The detector can hand back what it threw away, without changing what it keeps.

    2026-10-17: the biggest fault of the day (real shots discarded by the area
    ceiling) was invisible to the only faithful replay source, because a
    rejected blob never reached detections.csv. `record_rejects` keeps one row
    per rejected blob, with the reason, for rejected.csv.
    """
    ok = True
    print("\n33. Rejected blobs are recorded with their reason (recording only)")
    P = DetectorParams(**GATE_BASE)
    flat4 = np.full((480, 640), 0.25, np.float32)
    speck = _gate_frame([(300, 200, 3, FAINT)])

    d_off, f_off = _find(P, flat4, speck)
    ok &= check("flag off (the default): nothing is recorded",
                d_off.record_rejects is False and d_off.last_rejected_blobs == [],
                f"{d_off.last_rejected_blobs}")

    d_on = LaserDetector(P)
    d_on.set_gain_map(flat4)
    d_on.record_rejects = True
    d_on.process(speck, timestamp=0.0)
    rows = d_on.last_rejected_blobs
    ok &= check("a size-gated speck is recorded as amp_area, with a peak and its area",
                len(rows) == 1 and rows[0][0] == "amp_area" and rows[0][3] > 0
                and abs(rows[0][4] - 9.0) < 1.0, f"{rows}")
    ok &= check("its position is the blob's (300+1, 200+1) in full-frame px",
                len(rows) == 1 and abs(rows[0][1] - 301) < 2 and abs(rows[0][2] - 201) < 2,
                f"{rows}")
    d_on.process(_gate_frame([]), timestamp=PERIOD)
    ok &= check("the next, empty frame starts from an empty list (no stale rows)",
                d_on.last_rejected_blobs == [], f"{d_on.last_rejected_blobs}")

    # An oversize blob under an ordinary ceiling, with the bright-oversize
    # rescue off: the old behaviour, now visible.
    BIG = DetectorParams(**{**GATE_BASE, "max_area": 500, "oversize_min_peak": 0})
    bigf = _gate_frame([(120, 300, 40, (30, 30, 50))])   # centre (139.5, 319.5): x != y on purpose
    d_big = LaserDetector(BIG)
    d_big.record_rejects = True
    got = d_big.process(bigf, timestamp=0.0)
    r_big = d_big.last_rejected_blobs
    ok &= check("an oversize blob is recorded as big, with its true peak and area",
                len(got) == 0 and len(r_big) == 1 and r_big[0][0] == "big"
                and r_big[0][3] >= 12 and r_big[0][4] > 1000, f"{r_big}")

    # Recording must not change a single decision.
    mixed = _gate_frame([(100, 200, 10, BRIGHT), (300, 200, 3, FAINT),
                         (500, 300, 3, BRIGHT)])
    outs = []
    for flag in (False, True):
        d = LaserDetector(P)
        d.set_gain_map(flat4)
        d.record_rejects = flag
        res = d.process(mixed, timestamp=0.0)
        outs.append(([(round(x.x, 3), round(x.y, 3), x.peak, round(x.area, 1)) for x in res],
                     dict(d.last_rejects)))
    ok &= check("detections and reject counters are identical with recording on and off",
                outs[0] == outs[1], f"{outs}")

    # The per-frame cap.
    from lsa.detect.laser import REJECT_ROWS_PER_FRAME
    many = _gate_frame([(20 + 50 * i, 100 + 40 * (i % 3), 3, FAINT) for i in range(12)])
    d_cap = LaserDetector(P)
    d_cap.set_gain_map(flat4)
    d_cap.record_rejects = True
    d_cap.process(many, timestamp=0.0)
    ok &= check(f"at most {REJECT_ROWS_PER_FRAME} rows a frame, however many are rejected",
                0 < len(d_cap.last_rejected_blobs) <= REJECT_ROWS_PER_FRAME
                and d_cap.last_rejects.get("amp_area", 0) > REJECT_ROWS_PER_FRAME,
                f"{len(d_cap.last_rejected_blobs)} rows, "
                f"{d_cap.last_rejects.get('amp_area')} counted")

    # --- every reason the detector writes, not only the two above ----------------
    # 2026-10-17: `dim`, `shape`, `halo` and `max_blobs` rows were produced by the
    # detector and asserted by NOTHING, so deleting any of them left this suite
    # green (tests/mutate_detector.py now reintroduces each). A reason that is
    # recorded wrongly is worse than one that is missing: a replay would then
    # A/B-test a rule against rows that do not describe what was rejected.
    def _rec(params, frame):
        d = LaserDetector(params)
        d.record_rejects = True
        out = d.process(frame, timestamp=0.0)
        return d, out, d.last_rejected_blobs

    ok &= check("the big row carries the blob's own position and area, not just a peak",
                len(r_big) == 1 and abs(r_big[0][1] - 140) < 2 and abs(r_big[0][2] - 320) < 2
                and abs(r_big[0][4] - 1600.0) < 1.0, f"{r_big}")

    _d, _o, r_dim = _rec(P, _gate_frame([(400, 300, 8, _red(11))]))
    ok &= check("a blob over the threshold but under min_peak is recorded as dim, at its "
                "own position, with its peak and area",
                len(r_dim) == 1 and r_dim[0][0] == "dim" and 6 <= r_dim[0][3] < 12
                and abs(r_dim[0][1] - 404) < 2 and abs(r_dim[0][2] - 304) < 2
                and abs(r_dim[0][4] - 64.0) < 1.0, f"{r_dim}")

    P_shape = DetectorParams(**{**GATE_BASE, "max_elongation": 3.0})
    f_shape = _gate_frame([])
    f_shape[300:304, 100:160] = BRIGHT                       # 60 x 4: a streak
    _d, o_shape, r_shape = _rec(P_shape, f_shape)
    ok &= check("a streak over max_elongation is recorded as shape, at its own position, "
                "with its area and elongation",
                len(o_shape) == 0 and len(r_shape) == 1 and r_shape[0][0] == "shape"
                and abs(r_shape[0][1] - 130) < 2 and abs(r_shape[0][2] - 302) < 2
                and abs(r_shape[0][4] - 240.0) < 1.0 and r_shape[0][7] > 10.0,
                f"{o_shape} {r_shape}")

    P_halo = DetectorParams(**{**GATE_BASE, "oversize_min_peak": 0})
    _lumps = [(170, 120, 8, _red(14)), (60, 170, 8, _red(14)), (180, 70, 8, _red(14))]
    _d, o_halo, r_halo = _rec(P_halo, _gate_frame([(100, 100, 50, _red(90))] + _lumps))
    _hal = [r for r in r_halo if r[0] == "halo"]
    _want = [(174, 124), (64, 174), (184, 74)]
    def _nearest(r):
        return min(range(3), key=lambda k: abs(r[1] - _want[k][0]) + abs(r[2] - _want[k][1]))
    ok &= check("lumps folded into a bloom are recorded as halo, one row each, at their own "
                "positions",
                len(o_halo) == 0 and len(_hal) == 3
                and {_nearest(r) for r in _hal} == {0, 1, 2}
                and all(abs(r[1] - _want[_nearest(r)][0]) < 2
                        and abs(r[2] - _want[_nearest(r)][1]) < 2 and r[3] == 14
                        for r in _hal), f"{o_halo} {r_halo}")

    _d, o_mb, r_mb = _rec(P, _gate_frame([(60 + 100 * i, 200, 8, _red(40 - 2 * i))
                                          for i in range(5)]))
    ok &= check("a fifth blob cut by max_blobs is recorded as max_blobs, at its own position",
                len(o_mb) == 4 and len(r_mb) == 1 and r_mb[0][0] == "max_blobs"
                and abs(r_mb[0][1] - 464) < 2 and abs(r_mb[0][2] - 204) < 2
                and abs(r_mb[0][3] - 32) <= 1 and abs(r_mb[0][4] - 64.0) < 1.0,
                f"{len(o_mb)} {r_mb}")

    # The per-frame cap at the sites the amp_area check above does not reach.
    _d_cd, _o, r_cd = _rec(P, _gate_frame([(20 + 50 * i, 100 + 40 * (i % 3), 4, _red(11))
                                           for i in range(12)]))
    ok &= check(f"the cap holds for dim too: 12 rejected, exactly {REJECT_ROWS_PER_FRAME} recorded",
                _d_cd.last_rejects.get("dim") == 12 and len(r_cd) == REJECT_ROWS_PER_FRAME
                and all(r[0] == "dim" for r in r_cd),
                f"{_d_cd.last_rejects.get('dim')} counted, {len(r_cd)} rows")
    _d_cb, _o, r_cb = _rec(P, _gate_frame([(20 + 50 * i, 100 + 40 * (i % 3), 8, _red(40 - 2 * i))
                                           for i in range(12)]))
    ok &= check(f"the cap holds for max_blobs too: 8 cut, exactly {REJECT_ROWS_PER_FRAME} recorded",
                len(r_cb) == REJECT_ROWS_PER_FRAME and all(r[0] == "max_blobs" for r in r_cb),
                f"{len(r_cb)} rows {[r[0] for r in r_cb]}")
    return ok


def main() -> int:
    ok = True

    print("\n1. One 100ms pulse (3 lit frames) collapses to one shot")
    seq = blank(20) + [[(400, 300)]] * 3 + blank(17)
    det, shots, hits, _ = run(seq, BASE, new_tracker())
    ok &= check("3 frames with a detection", hits == 3, f"got {hits}")
    ok &= check("exactly 1 shot", len(shots) == 1, f"got {len(shots)}")
    if shots:
        s = shots[0]
        print(f"         pos=({s.x:.2f},{s.y:.2f}) frames={s.frames} "
              f"dur={s.duration_ms:.1f}ms peak={s.peak} area={s.area:.0f}")
        ok &= check("centroid within 2px of truth (400,300)",
                    abs(s.x - 400) < 2 and abs(s.y - 300) < 2)
        ok &= check("pulse length recovered near 100ms",
                    90 <= s.duration_ms <= 110, f"{s.duration_ms:.1f}ms")
    ok &= check("under 10ms/frame at 640x480", det.last_process_ms < 10,
                f"{det.last_process_ms:.2f}ms")

    print("\n2. The dull red decoy panel never registers")
    _, shots2, hits2, _ = run(blank(40), BASE, new_tracker())
    ok &= check("no detections at all", hits2 == 0, f"got {hits2}")
    ok &= check("no phantom shots", len(shots2) == 0, f"got {len(shots2)}")

    print("\n3. TWO GUNS, perfectly simultaneous -> two shots")
    seq = blank(15) + [[(200, 240), (480, 320)]] * 3 + blank(15)
    _, shots3, _, maxd = run(seq, BASE, new_tracker())
    ok &= check("both dots seen in one frame", maxd == 2, f"max dots/frame = {maxd}")
    ok &= check("2 shots reported", len(shots3) == 2, f"got {len(shots3)}")
    if len(shots3) == 2:
        xs = sorted(s.x for s in shots3)
        ok &= check("both positions correct",
                    abs(xs[0] - 200) < 3 and abs(xs[1] - 480) < 3,
                    f"x = {xs[0]:.1f}, {xs[1]:.1f}")
        ok &= check("distinct track ids",
                    shots3[0].track_id != shots3[1].track_id,
                    f"ids {shots3[0].track_id}, {shots3[1].track_id}")

    print("\n4. TWO GUNS, overlapping but offset in time -> still two shots")
    seq = (blank(10)
           + [[(150, 150)]] * 2
           + [[(150, 150), (500, 400)]] * 2
           + [[(500, 400)]] * 2
           + blank(10))
    _, shots4, _, _ = run(seq, BASE, new_tracker())
    ok &= check("2 shots reported", len(shots4) == 2, f"got {len(shots4)}")
    if len(shots4) == 2:
        for s in shots4:
            print(f"         track {s.track_id}: pos=({s.x:.1f},{s.y:.1f}) "
                  f"frames={s.frames} dur={s.duration_ms:.1f}ms")
        ok &= check("each ran 4 frames",
                    all(s.frames == 4 for s in shots4),
                    ", ".join(str(s.frames) for s in shots4))

    print("\n5. Rapid fire from one gun at the same spot -> two shots, not one")
    seq = blank(8) + [[(300, 300)]] * 3 + blank(5) + [[(300, 300)]] * 3 + blank(8)
    _, shots5, _, _ = run(seq, BASE, new_tracker())
    ok &= check("2 shots reported", len(shots5) == 2, f"got {len(shots5)}")

    print("\n6. min_frames=2 rejects a single-frame noise blip")
    tr = new_tracker()
    _, shots6, _, _ = run(blank(15) + [[(320, 240)]] + blank(14), BASE, tr)
    ok &= check("blip rejected", len(shots6) == 0, f"got {len(shots6)}")
    ok &= check("rejection counted", tr.rejected_runs == 1, f"got {tr.rejected_runs}")

    print("\n7. Off-axis camera: an elliptical dot is still found")
    seq = blank(15) + [[(420, 260)]] * 3 + blank(12)
    _, shots7, _, _ = run(seq, BASE, new_tracker(), oblique=True)
    ok &= check("1 shot from an 11x4 ellipse", len(shots7) == 1, f"got {len(shots7)}")
    if shots7:
        print(f"         pos=({shots7[0].x:.1f},{shots7[0].y:.1f}) "
              f"shape={shots7[0].elongation:.2f}:1")
        ok &= check("centroid still within 3px",
                    abs(shots7[0].x - 420) < 3 and abs(shots7[0].y - 260) < 3)
        ok &= check("elongation reported above 1.5",
                    shots7[0].elongation > 1.5, f"{shots7[0].elongation:.2f}")

    print("\n8. Fragment suppression: a bloomy dot does not read as several shots")
    tr = new_tracker()
    seq = blank(10) + [[(300, 200), (304, 203)]] * 3 + blank(10)
    _, shots8, _, maxd8 = run(seq, BASE, tr)
    ok &= check("merged into 1 dot per frame", maxd8 == 1, f"max dots/frame = {maxd8}")
    ok &= check("1 shot reported", len(shots8) == 1, f"got {len(shots8)}")

    print("\n9. RED and GRAY score modes also work")
    for mode in (ScoreMode.RED, ScoreMode.GRAY):
        p = DetectorParams(**{**vars(BASE), "mode": mode, "threshold": 60, "min_peak": 60})
        _, sh, _, _ = run(blank(20) + [[(400, 300)]] * 3 + blank(17), p, new_tracker())
        good = len(sh) == 1 and abs(sh[0].x - 400) < 3 and abs(sh[0].y - 300) < 3
        ok &= check(f"mode {mode.value}", good, f"{len(sh)} shot(s)")

    print("\n10. Downscaled processing (scale=0.5) maps back to full-frame coords")
    p10 = DetectorParams(**{**vars(BASE), "scale": 0.5, "min_area": 1})
    _, sh10, _, _ = run(blank(20) + [[(120, 420)]] * 3 + blank(17), p10, new_tracker())
    ok &= check("1 shot at half resolution", len(sh10) == 1, f"got {len(sh10)}")
    if sh10:
        print(f"         pos=({sh10[0].x:.2f},{sh10[0].y:.2f}) vs truth (120,420)")
        ok &= check("within 4px of truth",
                    abs(sh10[0].x - 120) < 4 and abs(sh10[0].y - 420) < 4)
        ok &= check("normalized coords sane",
                    0.0 <= sh10[0].nx <= 1.0 and 0.0 <= sh10[0].ny <= 1.0,
                    f"nx={sh10[0].nx:.3f} ny={sh10[0].ny:.3f}")

    print("\n11. A steady light is NOT a shot, however bright")
    # 40 frames of a dot that never goes out: a lamp, a reflection, a window.
    # Before the duration ceiling existed, this registered as one 1.3-second
    # "shot" the moment it finally vanished. Seen for real on the C910.
    tr = new_tracker()
    _, shots11, _, _ = run(blank(5) + [[(300, 300)]] * 40 + blank(10), BASE, tr)
    ok &= check("no shot from a 40-frame steady dot", len(shots11) == 0,
                f"got {len(shots11)}")
    ok &= check("counted as too-long, not too-short", tr.rejected_too_long >= 1,
                f"too_long={tr.rejected_too_long} too_short={tr.rejected_runs}")

    print("\n12. The ceiling does not eat a legitimate 100ms pulse")
    tr = new_tracker()
    _, shots12, _, _ = run(blank(10) + [[(300, 300)]] * 3 + blank(10), BASE, tr)
    ok &= check("3-frame pulse still accepted", len(shots12) == 1, f"got {len(shots12)}")
    ok &= check("nothing rejected as too long", tr.rejected_too_long == 0)

    print("\n13. A steady light must not block a real shot elsewhere")
    # The lamp holds a track the whole time; a genuine pulse arrives mid-way.
    frames = []
    for i in range(40):
        dots = [(120, 120)]
        if i in (20, 21, 22):
            dots.append((500, 400))
        frames.append(dots)
    tr = new_tracker()
    _, shots13, _, _ = run(frames + blank(8), BASE, tr)
    real = [s for s in shots13 if abs(s.x - 500) < 10]
    ok &= check("the real shot still gets through", len(real) == 1,
                f"{len(shots13)} shot(s) total, {len(real)} at the right place")

    print("\n14. Screen mask rejects everything off the monitor")
    # Real numbers from a live session: 1040 of 1356 detections came from a
    # bright window beyond the right edge of the screen. The screen here is the
    # left two thirds of frame; the "window" sits outside it.
    screen = ((60.0, 20.0), (430.0, 20.0), (430.0, 450.0), (60.0, 450.0))
    masked = DetectorParams(**{**vars(BASE), "roi_polygon": screen, "roi_margin": 0.0})
    tr = new_tracker()
    _, on_screen, _, _ = run(blank(10) + [[(250, 240)]] * 3 + blank(10), masked, tr)
    ok &= check("a hit ON the screen still registers", len(on_screen) == 1,
                f"got {len(on_screen)}")
    tr = new_tracker()
    _, off_screen, hits_off, _ = run(blank(10) + [[(560, 240)]] * 3 + blank(10), masked, tr)
    ok &= check("a bright dot OFF the screen is ignored",
                len(off_screen) == 0 and hits_off == 0,
                f"{len(off_screen)} shot(s), {hits_off} detection(s)")

    print("\n15. Two guns, one on screen and one off it")
    tr = new_tracker()
    seq = blank(10) + [[(200, 240), (560, 300)]] * 3 + blank(10)
    _, mixed, _, maxd = run(seq, masked, tr)
    ok &= check("only the on-screen dot is reported per frame", maxd == 1,
                f"max dots/frame = {maxd}")
    ok &= check("exactly one shot", len(mixed) == 1, f"got {len(mixed)}")
    if mixed:
        ok &= check("and it is the on-screen one", abs(mixed[0].x - 200) < 5,
                    f"x={mixed[0].x:.1f}")

    print("\n16. Margin shrinks the usable area")
    inset = DetectorParams(**{**vars(BASE), "roi_polygon": screen, "roi_margin": 40.0})
    tr = new_tracker()
    _, edge, _, _ = run(blank(10) + [[(75, 240)]] * 3 + blank(10), inset, tr)
    ok &= check("a hit inside the margin band is rejected", len(edge) == 0,
                f"got {len(edge)}")

    print("\n17. A flickering pulse is one shot, not four")
    # Measured live: one trigger pull registered as four shots at (445,471)
    # within 0.29s because detection dropped out between frames.
    tr = new_tracker(max_gap_frames=4)
    flicker = (blank(8) + [[(300, 300)]] * 2 + blank(2) + [[(300, 300)]]
               + blank(1) + [[(300, 300)]] * 2 + blank(12))
    _, one, _, _ = run(flicker, BASE, tr)
    ok &= check("stitched into a single shot", len(one) == 1, f"got {len(one)}")
    if one:
        ok &= check("and it is not rejected as too long",
                    one[0].duration_ms < 400, f"{one[0].duration_ms:.0f}ms")

    print("\n18. Refractory window: a re-triggering fragment is not a second shot")
    # Measured live: one pull at (275,674) reported three times in 1.5s with
    # pulses of 43, 43 and 57 ms. Track-level gap tolerance cannot catch this,
    # because the first track has already closed by then.
    tr = new_tracker(max_gap_frames=2)
    seq = (blank(6) + [[(300, 300)]] * 2 + blank(4) + [[(301, 302)]] * 2
           + blank(4) + [[(299, 301)]] * 2 + blank(10))
    _, frag, _, _ = run(seq, BASE, tr)
    ok &= check("three fragments collapse to one shot", len(frag) == 1, f"got {len(frag)}")
    ok &= check("merges were counted", tr.merged_fragments == 2,
                f"merged={tr.merged_fragments}")

    print("\n19. But two deliberate shots at the same spot still count twice")
    # 40 frames apart at 30fps is 1.3s, far outside the 250ms refractory window.
    tr = new_tracker()
    seq = blank(5) + [[(300, 300)]] * 3 + blank(40) + [[(300, 300)]] * 3 + blank(5)
    _, two, _, _ = run(seq, BASE, tr)
    ok &= check("two separate shots", len(two) == 2, f"got {len(two)}")

    print("\n20. A hand-sized blob: size alone no longer rejects it (2026-10-17, near side)")
    # This used to assert 'no shot from a 2800px blob'. Real laser dots measured
    # 7-60 px at the far end -- and 1673-9665 px on the near side, where 7 of 13
    # shot-bright blobs (peak 98-109) were thrown away as 'too big' while specks
    # of peak 9-24 got the holes. What separates a bloom from a hand or a lamp
    # is brightness and DURATION, so those are what is asserted now. See
    # DetectorParams.oversize_min_peak.
    def _blob_run(color, lit_frames, n=60):
        det = LaserDetector(DetectorParams(**vars(BASE)))
        tr = new_tracker()
        prov, final = [], []
        for i in range(n):
            f = np.full((480, 640, 3), 28, np.uint8)
            if i in lit_frames:
                cv2.circle(f, (320, 240), 30, color, -1)          # ~2800 px
            d = det.process(f, timestamp=i * PERIOD)
            final.extend(tr.feed(d, i))
            prov.extend(tr.take_provisional())
        final.extend(tr.flush())
        return (prov, final), tr
    bright = (110, 120, 255)                                       # chroma 135
    (p20, f20), _t = _blob_run(bright, set(range(10, 15)))
    ok &= check("a BRIGHT 2800px blob lasting 5 frames IS a shot (a near-side bloom): "
                "announced once, confirmed once",
                len(p20) == 1 and len(f20) == 1, f"announced {len(p20)}, closed {len(f20)}")
    (p20s, f20s), tr_s = _blob_run(bright, set(range(10, 50)))
    # THE COST, stated: a track is announced at min_frames, before the duration
    # ceiling can know it is a lamp. One onset report per appearance, then the
    # track is poisoned and nothing more is ever reported for it.
    ok &= check("...the same blob lit for 40 frames (a lamp, a hand): the duration ceiling "
                "poisons it -- never CONFIRMED, one onset announcement only",
                len(f20s) == 0 and len(p20s) == 1 and tr_s.rejected_too_long == 1,
                f"announced {len(p20s)}, closed {len(f20s)}, too_long {tr_s.rejected_too_long}")
    (p20f, f20f), _t = _blob_run((28, 28, 60), set(range(10, 15)))   # chroma 32 < BASE min_peak 50
    ok &= check("a FAINT 2800px blob is still rejected (below min_peak, as any dim blob)",
                len(p20f) == 0 and len(f20f) == 0, f"got {len(p20f)}/{len(f20f)}")

    print("\n21. Latency: a shot is announced as soon as it is credible")
    # Reported at close, a shot waits for the pulse to end plus the whole gap
    # tolerance -- about 200 ms at 60fps with gap=6. That lag is visible as late
    # bullet holes, which is what prompted this.
    det = LaserDetector(DetectorParams(**vars(BASE)))
    tr = new_tracker(max_gap_frames=6)
    lit = {10, 11, 12, 13, 14, 15}
    announced_at = closed_at = None
    for i in range(40):
        d = det.process(make_frame([(320, 240)] if i in lit else []), timestamp=i * PERIOD)
        finals = tr.feed(d, i)
        if tr.take_provisional() and announced_at is None:
            announced_at = i
        if finals and closed_at is None:
            closed_at = i
    saved = (closed_at - announced_at) * PERIOD * 1000 if announced_at and closed_at else 0
    print(f"         announced on frame {announced_at}, finalised on frame {closed_at} "
          f"-> {saved:.0f} ms earlier")
    ok &= check("announced within 2 frames of the pulse starting",
                announced_at is not None and announced_at - min(lit) <= 1,
                f"pulse at {min(lit)}, announced {announced_at}")
    ok &= check("that is at least 100 ms sooner than the close", saved >= 100,
                f"{saved:.0f} ms")

    print("\n22. The provisional and final reports are the SAME shot")
    det = LaserDetector(DetectorParams(**vars(BASE)))
    tr = new_tracker(max_gap_frames=6)
    prov, final = [], []
    for i in range(40):
        d = det.process(make_frame([(320, 240)] if i in lit else []), timestamp=i * PERIOD)
        final.extend(tr.feed(d, i))
        prov.extend(tr.take_provisional())
    final.extend(tr.flush())
    ok &= check("one provisional and one final", len(prov) == 1 and len(final) == 1,
                f"{len(prov)} provisional, {len(final)} final")
    if prov and final:
        ok &= check("sharing an index, so the hole is moved not duplicated",
                    prov[0].index == final[0].index,
                    f"{prov[0].index} vs {final[0].index}")
        ok &= check("flagged provisional vs final",
                    prov[0].provisional and not final[0].provisional)
        drift = abs(prov[0].x - final[0].x) + abs(prov[0].y - final[0].y)
        print(f"         position moved {drift:.2f}px between the two reports")
        ok &= check("final position barely moves", drift < 6.0, f"{drift:.2f}px")
        ok &= check("shot counter did not double-count", tr.count == 1, f"count={tr.count}")

    print("\n23. The hit is the FIRST frame of the pulse, not its average")
    # A real round leaves at trigger break; the laser stays lit through the
    # follow-through. Averaging over the pulse reports a place the bullet never
    # was. Measured on a real session: up to 7.5mm of error, 11.5mm of travel.
    tr23 = new_tracker(max_move_px=60.0)
    drifting = []
    for i in range(6):
        drifting += tr23.feed([Detection(x=100.0 + i * 8.0, y=200.0, nx=0.0, ny=0.0,
                                         peak=200, area=50.0, width=8.0, height=8.0,
                                         frame_index=i, timestamp=i * PERIOD_MS / 1000.0)], i)
    drifting += tr23.flush()
    ok &= check("a drifting pulse yields one shot", len(drifting) == 1, f"{len(drifting)}")
    if drifting:
        ok &= check("reported at the first frame, not the average",
                    abs(drifting[0].x - 100.0) < 0.01,
                    f"x={drifting[0].x:.1f} (first 100.0, last 140.0, average 120.0)")

    # The one case the old averaging guarded against: a first frame that caught
    # the pulse mid-ramp. Step forward to the first solid sample, no further.
    tr23b = new_tracker(max_move_px=60.0)
    dim = []
    for i, pk in enumerate((40, 220, 215, 210)):
        dim += tr23b.feed([Detection(x=300.0 + i * 10.0, y=400.0, nx=0.0, ny=0.0,
                                     peak=pk, area=50.0, width=8.0, height=8.0,
                                     frame_index=i, timestamp=i * PERIOD_MS / 1000.0)], i)
    dim += tr23b.flush()
    if dim:
        ok &= check("a dim first frame steps forward to the first solid one",
                    abs(dim[0].x - 310.0) < 0.01,
                    f"x={dim[0].x:.1f} (dim frame 300.0, first solid 310.0)")

    print("\n24. A faint blob is reported WITH its position, and per-shot centroid evidence")
    # Plan 02 steps 9 and 10. "Heard a shot, saw nothing" is three different
    # events (too faint, too short, not on the panel) and the log could not tell
    # them apart. The detector already measured how bright the faint thing was;
    # this is where it was.
    from lsa.shotlog import shot_evidence
    # A dot that clears the mask threshold but not min_peak is a dim blob.
    dim_params = DetectorParams(**{**vars(BASE), "min_peak": 250})
    det_dim = LaserDetector(dim_params)
    for k_ in range(6):       # let the background model learn the empty screen
        det_dim.process(make_frame([]), timestamp=-(6 - k_) * PERIOD)
    found = det_dim.process(make_frame([(320, 240)]), timestamp=0.0)
    ok &= check("a dot below min_peak is rejected as too faint (nothing returned)",
                not found and det_dim.last_dim_peak > 0,
                f"dim_peak {det_dim.last_dim_peak}")
    pos = det_dim.last_dim_pos
    ok &= check("and its position is reported (centre of its bounding box)",
                pos is not None and abs(pos[0] - 320) <= 3 and abs(pos[1] - 240) <= 3,
                f"{pos}")
    det_dim.process(make_frame([]), timestamp=PERIOD)
    ok &= check("with nothing faint in the frame there is no position",
                det_dim.last_dim_peak == 0 and det_dim.last_dim_pos is None,
                f"peak {det_dim.last_dim_peak}, pos {det_dim.last_dim_pos}")
    # The dim position must not change what the detector DETECTS.
    det_same = LaserDetector(BASE)
    for k_ in range(6):
        det_same.process(make_frame([]), timestamp=-(6 - k_) * PERIOD)
    seen_same = det_same.process(make_frame([(320, 240)]), timestamp=0.0)
    ok &= check("a bright dot is still detected, and leaves no dim position",
                len(seen_same) == 1 and det_same.last_dim_pos is None,
                f"{len(seen_same)} detection(s)")

    # shot_evidence on real tracker output (sections 23's shots).
    ev_dim = shot_evidence(dim[0], 13.0, 0.30)
    ok &= check("evidence: names the frame the centroid came from (2 of 4)",
                "frame 2/4" in ev_dim, ev_dim)
    ok &= check("evidence: says how dim the first frame was (40 of 220 = 18%)",
                "first-frame peak 40 of best 220 (18%)" in ev_dim, ev_dim)
    ok &= check("evidence: reports drift first-vs-brightest (10.0 cam px, 3.0mm)",
                "drift first-vs-brightest 10.0 cam px (~3.0mm)" in ev_dim, ev_dim)
    ok &= check("evidence: reports the margin over the bar (220/13)",
                "best/bar 220/13 = 16.9x" in ev_dim, ev_dim)
    ev_even = shot_evidence(drifting[0], 13.0, None)
    ok &= check("evidence: an even pulse reports its first frame, zero drift",
                "frame 1/6" in ev_even and "drift first-vs-brightest 0.0 cam px" in ev_even,
                ev_even)

    ok &= amp_area_gate_checks()
    ok &= halo_checks()
    ok &= oversize_checks()
    ok &= rejected_record_checks()

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
