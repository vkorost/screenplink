"""tools/read_string.py reads the log the app already writes, and only reads.

A crafted log holds one of each case a string of shots produces: a hit heard and
seen, a miss seen and not heard (a phantom), a pulse heard and not seen in each
of its branches (faint on the panel, off the panel, nothing at any level), one
trigger pull that fragmented into two detections 0.3 s apart with the second
while the app had muted its own microphone, and a shot with no calibration.

The lines are built from the app's REAL producers (`lsa.shotlog`,
`describe_nearest`, the corroborator's tally) glued the way `_on_shot` glues
them, so a change to a producer's wording that the reader no longer understands
fails here -- and `test_ui_smoke` runs the reader over lines from the real
`_on_shot`. No camera, no display, no sound device, nothing written.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa.audio.listener import ShotCorroborator                    # noqa: E402
from lsa.modes.base import describe_nearest                        # noqa: E402
from lsa.shotlog import (audio_tally, describe_missed_pulse,       # noqa: E402
                         mic_state, shot_evidence)

_spec = importlib.util.spec_from_file_location("read_string_under_test",
                                               ROOT / "tools" / "read_string.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["read_string_under_test"] = rs
_spec.loader.exec_module(rs)

results: list[bool] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


# -- build the log from the real producers --------------------------------------

MM = 940.0 / 3840.0


def run(frames: list[tuple[float, float, int]], first_peak_idx: int = 0):
    """A Shot stand-in: `members` are (x, y, peak) frames; x, y is the reported one."""
    mem = [SimpleNamespace(x=x, y=y, peak=p) for x, y, p in frames]
    rep = mem[first_peak_idx]
    return SimpleNamespace(members=mem, x=rep.x, y=rep.y, frames=len(mem),
                           peak=max(m.peak for m in mem))


def shot_line(clock, idx, cam, scr, shot, outcome, near, t, mic, co):
    sx = f"screen=({scr[0]:7.1f},{scr[1]:7.1f})" if scr else "screen=(uncalibrated)"
    hit = {"miss": "  miss", "": ""}.get(outcome, f"  {outcome}")
    extra = "  ".join(x for x in (
        describe_nearest(near[0], near[1], near[2], MM) if near else "",
        f"t={t:.3f}", f"mic_peak={mic}",
        shot_evidence(shot, 13.0, 0.3), audio_tally(co)) if x)
    return (f"{clock}  SHOT {idx:3d}  cam=({cam[0]:7.2f},{cam[1]:7.2f})  {sx}  "
            f"frames={shot.frames}  pulse~{shot.frames * 28.0:5.1f}ms  peak={shot.peak}  "
            f"area=120px{hit}  {extra}")


def mic(level, bar=0.15, learn=False, deaf=0.0, blocks=20):
    return (f"{level:.3f} (bar {bar:.3f}" + (", mic not calibrated yet" if learn else "")
            + ")" + mic_state(blocks, deaf))


def ev(dim=0, off=0, short=0, fp=None):
    fp = max(7, dim, off) if fp is None else fp          # the brightest score anywhere
    return {"frames": 30, "frame_peak": fp, "dim_peak": dim,
            "dim_pos": (321.0, 198.0) if dim else None, "off_peak": off, "short_delta": short}


co = ShotCorroborator()
LOG = []
LOG += ["20:00:00  SHOT 1  cam=( 10.00, 10.00)  screen=(  1.0,   1.0)  frames=2  pulse~ 40.0ms  peak=40  area=10px  miss",
        "=== 2026-10-17 03:00:00  camera lab started ===",
        "03:00:01  opening camera 2 on MSMF at 1280x720 MJPG @60..."]
# 1. a hit, heard and seen
co.heard, co.seen, co.both = 1, 1, 1
s1 = run([(500.0, 300.0, 90), (500.2, 300.1, 180), (500.1, 300.0, 170)], first_peak_idx=0)
LOG.append(shot_line("03:00:10", 1, (500.0, 300.0), (1920.0, 1080.0), s1, "HIT centre +9",
                     (("centre", 41.0, 907.0), True, "centre"), 100.000, mic(0.285), co))
# 2. seen, NOT heard: a phantom (mic listening, nothing there)
co.seen += 1
s2 = run([(700.0, 250.0, 60), (700.0, 250.0, 55)])
LOG.append(shot_line("03:00:14", 2, (700.0, 250.0), (2400.0, 800.0), s2, "miss",
                     (("circle3", 120.0, 97.0), False, None), 104.000, mic(0.004), co))
co.seen_not_heard += 1
LOG.append("03:00:15  SAW a shot with no gunshot sound: likely a phantom detection, not a trigger pull")
# 3. heard, not seen -- faint, on the panel
co.heard += 1
co.heard_not_seen += 1
LOG.append("03:00:20  HEARD a shot with no detection: detector evidence within 0.25 s of the click: "
           + describe_missed_pulse(ev(dim=9, off=5), 13.0, 8.0) + "  t=108.500  " + audio_tally(co))
# 4. heard, not seen -- nothing at any level (a realistic off-screen floor of 5)
co.heard += 1
co.heard_not_seen += 1
LOG.append("03:00:24  HEARD a shot with no detection: detector evidence within 0.25 s of the click: "
           + describe_missed_pulse(ev(off=5), 13.0, 8.0) + "  t=112.500  " + audio_tally(co))
# 5. one trigger pull, two detections 0.3 s apart; the second while the app was deaf
co.heard += 1
co.seen += 2
co.both += 1
s3 = run([(900.0, 400.0, 150), (900.1, 400.0, 160)])
LOG.append(shot_line("03:00:30", 3, (900.0, 400.0), (3000.0, 1500.0), s3, "HIT circle2 +1",
                     (("circle2", 30.0, 97.0), True, "circle2"), 120.000, mic(0.40), co))
s4 = run([(904.0, 402.0, 70), (904.0, 402.0, 65)])
LOG.append(shot_line("03:00:30", 4, (904.0, 402.0), (3010.0, 1505.0), s4, "miss",
                     (("circle2", 130.0, 97.0), False, None), 120.300, mic(0.003, deaf=0.48), co))
co.seen_not_heard += 1
LOG.append("03:00:31  SAW a shot with no gunshot sound, but" + mic_state(20, 0.48))
# 6. a shot with no calibration: nowhere to draw it
co.seen += 1
s5 = run([(300.0, 200.0, 50), (300.0, 200.0, 48)])
LOG.append(shot_line("03:00:40", 5, (300.0, 200.0), None, s5, "", None, 130.000,
                     mic(0.30, learn=True), co))
# 7. heard, not seen -- a dot off the panel
co.heard += 1
co.heard_not_seen += 1
LOG.append("03:00:44  HEARD a shot with no detection: detector evidence within 0.25 s of the click: "
           + describe_missed_pulse(ev(off=89, fp=91), 13.0, 8.0) + "  t=134.200  " + audio_tally(co))

print("1. the reader over a crafted log built from the real producers")
rep = rs.parse(LOG)
P = rep.pulls
shots = [p for p in P if p.kind == "shot"]
hns = [p for p in P if p.kind == "heard"]
check("only the LAST launch is read (the line above the header is not a row)",
      len(shots) == 5 and shots[0].index == 1 and len(hns) == 3,
      f"{len(shots)} shots, {len(hns)} heard-not-seen")
check("with --all the earlier launch's line is read too",
      len(rs.parse(LOG, all_launches=True).pulls) == len(P) + 1)

a = shots[0]
check("hit: heard, seen, where it was drawn and that it scored",
      a.heard.startswith("yes (0.285)") and a.seen == "yes" and a.screen == (1920.0, 1080.0)
      and a.outcome == "HIT centre +9", f"{a.heard} {a.outcome} {a.screen}")
check("hit: distance to the target in mm, from centre", a.target.startswith("10.0mm from centre of centre"),
      a.target)
check("hit: which frame the centroid came from, the drift, the margin over the bar",
      a.frame_used == "1/3" and a.drift.startswith("0.2 cam px") and a.margin.startswith("13.8x (180/13)"),
      f"{a.frame_used} | {a.drift} | {a.margin}")

b = shots[1]
check("a miss reports how far OUTSIDE which target, in mm",
      b.outcome == "miss" and b.target.startswith("5.6mm outside circle3"), b.target)
check("heard NO (the microphone was listening) and the app's SAW line marks it a phantom",
      b.heard.startswith("NO (0.004)") and b.saw_flag
      and any("SEEN, NOT HEARD: likely a phantom" in n for n in b.notes), f"{b.heard} {b.notes}")

c = hns[0]
check("heard-not-seen, faint on the panel: its own row, seen NO, branch named, bar shown",
      c.seen == "NO" and any("faint dot ON the panel" in n for n in c.notes)
      and any("faint blob 9 (bar 13)" in n for n in c.notes), str(c.notes))
d = hns[1]
check("heard-not-seen, nothing at any level (off-screen floor 5): that branch, not 'off-panel'",
      any("camera saw nothing at any level" in n for n in d.notes)
      and not any("OUTSIDE" in n for n in d.notes), str(d.notes))
e = hns[2]
check("heard-not-seen, a dot off the panel (89): the off-panel branch with its threshold",
      any("OUTSIDE the screen mask" in n for n in e.notes)
      and any("off-screen blob 89 (threshold 8)" in n for n in e.notes), str(e.notes))

f3, f4 = shots[2], shots[3]
check("the fragmented pulse: the second detection is marked a FRAGMENT of the first, with "
      "the time and the distance",
      any(n.startswith("FRAGMENT of #3: 0.30s later") for n in f4.notes)
      and not any(n.startswith("FRAGMENT") for n in f3.notes), str(f4.notes))
check("... and its microphone state is DEAF, and the SAW line is NOT read as a phantom",
      f4.deaf and f4.heard.startswith("DEAF (0.003)") and f4.saw_flag
      and any("not a phantom" in n for n in f4.notes)
      and not any("likely a phantom" in n for n in f4.notes), f"{f4.heard} {f4.notes}")

g = shots[4]
check("a shot with no calibration has no screen position and says nowhere to draw it",
      g.screen is None and any("nowhere to draw" in n for n in g.notes), str(g.notes))
check("... and a microphone still learning is not counted as heard",
      g.heard.startswith("learning (0.300)"), g.heard)

out = rs.render(rep)
print(out)
check("the heard/seen totals are the app's own, from its last tally line",
      rep.tally is not None and rep.tally["heard"] == 5 and rep.tally["seen"] == 5
      and rep.tally["hns"] == 3 and rep.tally["snh"] == 2, str(rep.tally))
check("the summary merges the fragment: 5 detections - 1 fragment + 3 heard-not-seen = 7 pulls",
      "detections 5 (1 of them fragments of an earlier pull), heard-not-seen 3  ->  trigger pulls 7" in out,
      out.splitlines()[-3])
check("holes drawn 4 (the uncalibrated shot has none), scored 2; of the 2 seen-not-heard, "
      "the microphone was deaf for 1, so ONE likely phantom, not two",
      "holes drawn 4   scored 2   seen-not-heard 2 (of which the microphone was deaf or "
      "silent-by-fault 1, so likely phantoms 1)" in out, out.splitlines()[-2])
check("the table has a column for every question asked",
      all(c_ in out.splitlines()[1] for c_ in ("heard", "seen", "drawn at", "nearest target",
                                               "frame", "drift", "margin", "note")),
      out.splitlines()[1])

# -- an older log: absent fields are "-", never invented ---------------------------
print()
print("2. an older log, and a log with no string in it")
old = ["=== 2026-10-16 20:56:31  camera lab started ===",
       "22:03:28  SHOT 3728  cam=( 261.41, 105.03)  screen=(  101.7,   96.6)  frames=2  pulse~ 47.0ms  peak=18  area=1787px  miss"]
po = rs.parse(old).pulls[0]
check("a line from before the instrumentation still parses, with '-' for what it lacks",
      po.index == 3728 and po.outcome == "miss" and po.target == "-" and po.frame_used == "-"
      and po.drift == "-" and po.margin == "-" and po.heard == "-", f"{po}")
check("a log with no shots has no rows", rs.parse(["03:00:01  opening camera"]).pulls == [])

# -- it only reads ----------------------------------------------------------------
print()
print("3. it only reads")
with tempfile.TemporaryDirectory() as td:
    lp = Path(td) / "camera_lab.log"
    lp.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    before = hashlib.sha256(lp.read_bytes()).hexdigest()
    mt = lp.stat().st_mtime_ns
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = rs.main([str(lp)])
    check("main() on a file prints the table and returns 0",
          code == 0 and "trigger pulls 7" in buf.getvalue(), f"code {code}")
    check("the log is byte-identical and untouched afterwards",
          hashlib.sha256(lp.read_bytes()).hexdigest() == before and lp.stat().st_mtime_ns == mt)
    check("nothing else appeared beside it", [p.name for p in Path(td).iterdir()] == ["camera_lab.log"])
    missing = rs.main([str(Path(td) / "nope.log")])
    check("a missing log is an error code, not a crash", missing == 2, str(missing))

print()
print(("ALL %d CHECKS PASSED" % len(results)) if all(results)
      else ("%d OF %d CHECKS FAILED" % (results.count(False), len(results))))
sys.exit(0 if all(results) else 1)
