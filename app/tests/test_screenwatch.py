"""The passive screen watch: is our own picture still as bright in the camera?

Pure policy tests, plus the capture-side reduction against synthetic frames
(no camera is opened: `_cap` stays None), plus its measured cost per frame.
Each section says what goes red when the behaviour is reverted.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                               # noqa: E402

from lsa import screenwatch as SW                                # noqa: E402
from lsa import watchdog as W                                    # noqa: E402

CHECKS = 0
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILED.append(what)
        print(f"  FAIL  {what}" + (f"  --  {detail}" if detail else ""))


def section(n: str) -> None:
    print(f"\n=== {n} ===")


DT = 1.0 / SW.SAMPLE_HZ
SIG = ("StaticTargets", ("background", "holes", "modes"))


def run(sw, levels, sig=SIG, t0=0.0):
    """Feed levels at the sample rate; return [(t, text)] of every firing."""
    out = []
    t = t0
    for lv in levels:
        r = sw.feed(t, lv, sig)
        if r:
            out.append((t, r))
        t += DT
    return out


def armed():
    lines: list[str] = []
    sw = SW.ScreenWatch(log=lines.append)
    sw.arm()
    return sw, lines


# ---------------------------------------------------------------------------
section("one number for one fact: the ratio is the watchdog's own")
check(SW.RATIO_BAD_BELOW is W.REL_BAD_BELOW or SW.RATIO_BAD_BELOW == W.REL_BAD_BELOW,
      "the relative bar is imported from lsa.watchdog, not retyped",
      f"{SW.RATIO_BAD_BELOW} vs {W.REL_BAD_BELOW}")

# ---------------------------------------------------------------------------
section("it never fires on a healthy, steady screen")
sw, lines = armed()
rng = np.random.default_rng(1)
levels = list(200 * (1 + rng.uniform(-0.10, 0.10, int(600 * SW.SAMPLE_HZ))))
check(run(sw, levels) == [], "ten minutes of 200 +/-10% (his measured healthy spread): silent")
check(sw.reference(SIG) is not None and 180 <= sw.reference(SIG) <= 220,
      "and it took a reference from the first stable samples",
      f"{sw.reference(SIG)}")

section("it is silent until armed (a launch not yet judged usable)")
sw = SW.ScreenWatch()
check(run(sw, [200] * 40 + [20] * 80) == [], "unarmed: never fires")

# ---------------------------------------------------------------------------
section("a real, sustained fall fires ONCE, with the numbers, and does not nag")
sw, lines = armed()
n_ref = SW.REF_SAMPLES + 4
fall = [200] * n_ref + [60] * int(40 * SW.SAMPLE_HZ)
fired = run(sw, fall)
check(len(fired) == 1, "exactly one firing over 40 s of a screen at 30%",
      f"{len(fired)}")
if fired:
    t, text = fired[0]
    check("30%" in text and "200" in text and "60" in text,
          "it says what it read, what it is judged against and the ratio", text)
    check(SW.SUSTAIN_S <= t - n_ref * DT <= SW.SUSTAIN_S + 2 * DT,
          "and only after the fall has lasted SUSTAIN_S", f"{t - n_ref * DT:.2f}s")

section("recovery re-arms it (hysteresis), so a second fall speaks again")
sw, lines = armed()
seq = [200] * n_ref + [60] * 30 + [190] * 20 + [60] * 30
fired = run(sw, seq)
check(len(fired) == 2, "fall, recovery, second fall: two firings, not one and not many",
      f"{len(fired)}")
check(any("armed again" in l for l in lines), "and the recovery is logged")

# ---------------------------------------------------------------------------
section("legitimate dimming does not fire it")
sw, lines = armed()
# a hit flash / a mode change lasting under SUSTAIN_S
blip = [200] * n_ref + [60] * int(SW.SUSTAIN_S * SW.SAMPLE_HZ - 2) + [200] * 30
check(run(sw, blip) == [], "a fall shorter than SUSTAIN_S is ignored")
sw, lines = armed()
blank = [200] * n_ref + [None] * int(60 * SW.SAMPLE_HZ) + [200] * 10
check(run(sw, blank) == [], "a BLANK screen (no ink: a figure waiting, all targets shot) is not judged")
sw, lines = armed()
flicker = [200] * n_ref + ([60] * 8 + [None] + [60] * 8 + [None]) * 6
check(run(sw, flicker) == [], "a blank sample breaks the sustain: no ink is not 'dim ink'")
sw, lines = armed()
sw.feed(0.0, 200, SIG)
for i in range(SW.REF_SAMPLES):
    sw.feed(i * DT, 200, SIG)
for i in range(30):
    sw.feed(10 + i * DT, 60, SIG)
    if i == 12:
        sw.pause()                     # a routine owned the screen for a moment
check(sw.fired == 0, "pause() (a routine owns the screen) breaks the sustain")

# like with like: another screen state is held to its OWN reference
sw, lines = armed()
VERDICT = ("StaticTargets", ("background", "holes", "modes", "verdict"))
seq1 = run(sw, [200] * n_ref, SIG)
seq2 = run(sw, [90] * int(60 * SW.SAMPLE_HZ), VERDICT, t0=10)
check(seq1 == [] and seq2 == [] and sw.reference(VERDICT) is not None
      and abs(sw.reference(VERDICT) - 90) < 1,
      "a new screen state (verdict panel up) takes its OWN reference, at 90, and is not "
      "compared with the 200 of another state",
      f"ref {sw.reference(VERDICT)}")
fired = run(sw, [90] * 5 + [30] * int(20 * SW.SAMPLE_HZ), VERDICT, t0=100)
check(len(fired) == 1, "and a fall within THAT state is still caught", f"{len(fired)}")

# a dim reference is not a standard
sw, lines = armed()
check(run(sw, [30] * 40 + [5] * 80) == [] and sw.reference(SIG) is None,
      "a reference below REF_MIN_LEVEL is refused (a dark scene is not a standard)")

# re-arming starts afresh
sw, lines = armed()
run(sw, [200] * n_ref)
sw.arm()
check(sw.reference(SIG) is None, "arm() forgets every reference (a new verdict, a new standard)")

# ---------------------------------------------------------------------------
section("the capture-side reduction, on real frames (no camera is opened)")
import cv2                                                       # noqa: E402
from lsa.camera.capture import CaptureConfig, CaptureThread, GrouperConfig   # noqa: E402
from lsa.detect.laser import DetectorParams                      # noqa: E402

POLY = ((200.0, 100.0), (1100.0, 130.0), (1080.0, 640.0), (220.0, 600.0))
cap = CaptureThread(CaptureConfig(), DetectorParams(roi_polygon=POLY), GrouperConfig())
check(cap._cap is None, "(fixture) no device is held")
writes_before = cap._stats.control_writes


def frame_with(rings, grey=200):
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(rings):
        cv2.circle(f, (640, 360), 120 + 70 * i, (grey, grey, grey), 4)
    return f


cap._sample_screen(frame_with(3), 1.0)
lvl3 = cap.screen_level
cap._sample_screen(frame_with(1), 2.0)
lvl1 = cap.screen_level
check(lvl3 is not None and lvl1 is not None and abs(lvl3 - lvl1) <= 12,
      "a mode that draws LESS (three rings -> one) leaves the ink level where it was",
      f"{lvl3} vs {lvl1}")
cap._sample_screen(frame_with(3, grey=80), 3.0)
check(cap.screen_level is not None and abs(cap.screen_level / lvl3 - 0.4) < 0.08,
      "the SAME picture at 40% brightness reads 40% (the camera dimming is visible)",
      f"{cap.screen_level} vs {lvl3}")
cap._sample_screen(np.zeros((720, 1280, 3), dtype=np.uint8), 4.0)
check(cap.screen_level is None, "a blank frame reads None (nothing to judge), not zero")
check(cap._stats.control_writes == writes_before,
      "it made no camera control write", f"{cap._stats.control_writes}")

section("what it costs on the capture thread (measured, not hoped)")
fr = frame_with(3)
cap._sample_screen(fr, 10.0)               # builds the mask once, outside the timing
N = 2000
t0 = time.perf_counter()
for i in range(N):
    cap._sample_screen(fr, 11.0 + i)
per_call_us = (time.perf_counter() - t0) / N * 1e6
amortised_us = per_call_us * SW.SAMPLE_HZ / 55.0
print(f"  (measured: {per_call_us:.0f} us per sample; at {SW.SAMPLE_HZ:g} Hz over 55 fps "
      f"that is {amortised_us:.1f} us per frame against a 7000 us detector budget)")
# Generous on purpose: wall-clock thresholds fail under CPU contention (parallel
# agents). The MEASURED figure (about 0.12 ms, 9 us/frame) is printed above; these
# bound it at ~1.5% of the detector budget, not at the measurement.
check(per_call_us < 1500, "one sample costs under 1.5 ms", f"{per_call_us:.0f} us")
check(amortised_us < 100, "amortised over every frame it is under 100 us", f"{amortised_us:.1f} us")

# ---------------------------------------------------------------------------
section("provenance is stated beside every constant")
src = (ROOT / "src" / "lsa" / "screenwatch.py").read_text(encoding="utf-8")
for name in ("SAMPLE_HZ", "INK_MIN_LUMA", "INK_MIN_PIXELS", "REF_MIN_LEVEL",
             "REF_SAMPLES", "SUSTAIN_S", "RECOVER_RATIO"):
    i = src.index(f"\n{name} = ")
    lead = src[max(0, i - 420):i]
    check("GUESSED" in lead or "FITTED" in lead or "MEASURED" in lead,
          f"{name} carries its provenance (GUESSED / FITTED / MEASURED)")

section("it is wired in: sampled by the capture loop, judged on the HUD tick")
cap_src = (ROOT / "src" / "lsa" / "camera" / "capture.py").read_text(encoding="utf-8")
lab_src = (ROOT / "src" / "lsa" / "tools" / "camera_lab.py").read_text(encoding="utf-8")
run_body = cap_src[cap_src.index("    def run(self)"):]
check("self._sample_screen(frame, now)" in run_body,
      "the capture loop calls _sample_screen (a sampler nobody calls reads None for ever)")
tick = lab_src[lab_src.index("    def _refresh_target_diagnostics"):]
tick = tick[:tick.index("    def _check_screen_watch")]
check("self._check_screen_watch()" in tick,
      "the HUD tick calls _check_screen_watch")
check("self.screenwatch.arm()" in lab_src and "self.screenwatch.disarm()" in lab_src,
      "a verdict arms it, and a BLOCKED one disarms it")

print()
print("=" * 64)
if FAILED:
    print(f"FAILED {len(FAILED)} of {CHECKS} SCREEN-WATCH CHECKS:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL {CHECKS} SCREEN-WATCH CHECKS PASSED")
