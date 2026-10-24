"""tools/replay.py cannot silently prove nothing. No camera, no screen, no recordings.

Four things a reviewer hit or would hit on a first run (2026-10-17):

  * a calibration with fewer than 4 sensitivity rows gave no gain map, the
    replay ran to the end, and the gate-on and gate-off tables were identical,
    which reads exactly like a clean regression proof;
  * `--from-csv` compared `Detection.area` (FULL-FRAME px) against
    `amplified_min_area` (PROCESSING-scale px), which coincide only at 1.0;
  * nothing told the reader which question the source could answer;
  * `--log` defaults to the live camera_lab.log and the session it belongs to
    was read and then discarded.

Everything here is built in a temp directory. Nothing under recordings/ is
opened, and the user's calibration.json is not read.

Run:  .venv\\Scripts\\python.exe tests\\test_replay.py
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import contextlib
import dataclasses
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import replay as R  # noqa: E402
from lsa.calibrate.mapping import Mapping, Sample  # noqa: E402
from lsa.detect.laser import (DetectorParams, ScoreMode,  # noqa: E402
                              amplified_min_area, full_frame_to_working_area)

results: list[bool] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")


FRAME = (1280, 720)
SCREEN = (3840, 2160)


def make_calibration(path: Path, peaks: list[float]) -> None:
    """A real Mapping, fitted from a 3x3 grid, with one sensitivity row per peak."""
    samples = []
    for i, sx in enumerate((400, 1920, 3440)):
        for j, sy in enumerate((300, 1080, 1860)):
            samples.append(Sample(screen=(float(sx), float(sy)),
                                  camera=(sx / 3.0, sy / 3.0), peak=100.0))
    m = Mapping.fit(samples, SCREEN, use_residuals=False, frame_size=FRAME)
    assert m is not None
    spots = [(150, 100), (1100, 100), (150, 620), (1100, 620), (640, 360), (640, 150)]
    m.sensitivity = [{"camera": [float(spots[k][0]), float(spots[k][1])], "peak": float(p)}
                     for k, p in enumerate(peaks)]
    assert m.save(path)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="test_replay_"))

    print("\n1. no gain map is an ERROR, not a quiet replay")
    cal3 = tmp / "cal3.json"
    make_calibration(cal3, [100, 40, 60])                       # 3 rows: not enough
    raised = None
    try:
        R.build_gain(cal3, FRAME)
    except R.NoGainMap as exc:
        raised = str(exc)
    check("fewer than 4 sensitivity rows raises NoGainMap", raised is not None, str(raised))
    check("... and the message says how many rows there were", raised is not None and "3" in raised)
    cal0 = tmp / "cal0.json"
    make_calibration(cal0, [])
    try:
        R.build_gain(cal0, FRAME)
        got0 = False
    except R.NoGainMap:
        got0 = True
    check("no rows at all raises too (it used to return None)", got0)

    cal6 = tmp / "cal6.json"
    make_calibration(cal6, [100, 100, 100, 40, 60, 100])
    gain = R.build_gain(cal6, FRAME)
    # NEVER hand a possibly-None map to `describe_gain`. The mutation that makes
    # `build_gain` return None instead of raising already failed check 1 BY NAME
    # above -- and then this line raised `AttributeError` on `gain.min()`, so the
    # suite printed no summary and the strict judge recorded a real catch as a
    # crash. Assert presence, then describe.
    info = R.describe_gain(gain, 6) if gain is not None else {
        "shape": None, "rows": 0, "min_gain": None, "max_n": None}
    check("with 4+ rows it returns a real map of the frame's shape",
          isinstance(gain, np.ndarray) and gain.shape == (720, 1280), str(getattr(gain, "shape", gain)))
    check("describe_gain reports shape, rows, minimum gain and maximum N",
          info["shape"] == [720, 1280] and info["rows"] == 6
          and 0.25 <= info["min_gain"] < 1.0
          and abs(info["max_n"] - 1.0 / info["min_gain"]) < 0.01, str(info))

    print("\n2. the log must belong to the session")
    live_ok = {"rec_dir": r"C:\WORKSPACE\Laser.Shooting.App\recordings\session-20261017-100406"}
    sess = Path("recordings/session-20261017-100406")
    try:
        R.check_log_matches_session(live_ok, sess, Path("camera_lab.log"))
        same = True
    except R.LogMismatch:
        same = False
    check("a log whose last 'recording to' names this session is accepted", same)
    for label, live in (("another session's", {"rec_dir": r"C:\x\recordings\session-20261016-190000"}),
                        ("no 'recording to' line in the", {"rec_dir": None})):
        try:
            R.check_log_matches_session(live, sess, Path("camera_lab.log"))
            bad = False
            msg = ""
        except R.LogMismatch as exc:
            bad, msg = True, str(exc)
        check(f"{label} log fails loudly", bad, msg[:140])
    lg = tmp / "camera_lab.log"
    lg.write_text("10:04:06  recording to C:\\a\\b\\session-20261017-100406\n"
                  "10:09:00  recording to C:\\a\\b\\session-20261017-100900\n", encoding="utf-8")
    check("parse_log keeps the directory of the LAST recording",
          R.parse_log(lg)["rec_dir"].endswith("session-20261017-100900"))

    print("\n3. UNITS: the csv area is full-frame px, the gate's ask is processing-scale px")
    p10 = DetectorParams(mode=ScoreMode.CHROMA, scale=1.0)
    p05 = DetectorParams(mode=ScoreMode.CHROMA, scale=0.5)
    N = 4.0
    ask_full = p10.amp_area_per_gain * (N - 1.0)           # 48 full-frame px
    check("the seam: working-scale conversion of the full-frame ask IS amplified_min_area, "
          "at scale 1.0 and at 0.5",
          abs(full_frame_to_working_area(p10, ask_full) - amplified_min_area(p10, N)) < 1e-6
          and abs(full_frame_to_working_area(p05, ask_full) - amplified_min_area(p05, N)) < 1e-6,
          f"{full_frame_to_working_area(p05, ask_full)} vs {amplified_min_area(p05, N)}")
    check("scale 1.0: 40 px is under the 48 asked, 60 px is over",
          not R.gate_keeps(p10, N, 40.0) and R.gate_keeps(p10, N, 60.0))
    check("SCALE 0.5: the same 40 px / 60 px csv areas give the same verdicts "
          "(before the fix 40 was compared with 12 and kept)",
          not R.gate_keeps(p05, N, 40.0) and R.gate_keeps(p05, N, 60.0),
          f"40 -> {R.gate_keeps(p05, N, 40.0)}, 60 -> {R.gate_keeps(p05, N, 60.0)}")
    check("and the verdict is the same at both scales for any csv area",
          all(R.gate_keeps(p10, n, a) == R.gate_keeps(p05, n, a)
              for n in (2.0, 3.0, 3.95) for a in (20, 33, 47, 48, 49, 80, 400)))
    check("--gate off keeps everything", R.gate_keeps(p05, N, 1.0, gate="off"))

    print("\n4. end to end through replay(--from-csv) at scale 0.5, and the report")
    det = dataclasses.asdict(DetectorParams(mode=ScoreMode.CHROMA, scale=0.5,
                                            threshold=10, min_peak=12))
    det["mode"] = ScoreMode.CHROMA.value
    det["roi_polygon"] = None
    yy, xx = np.unravel_index(int(np.argmin(gain)), gain.shape)
    n_here = R.amplification_at(gain, float(xx), float(yy))
    need = 16.0 * (n_here - 1.0)                           # full-frame px the gate asks
    small, big = 0.7 * need, 1.5 * need

    def make_session(name: str, area: float) -> tuple[Path, Path]:
        """A session with ONE two-frame detection of the given full-frame area."""
        sdir = tmp / name
        sdir.mkdir()
        (sdir / "settings.json").write_text(json.dumps(dict(
            detector=det,
            grouper=dict(min_frames=2, single_frame_peak=0, max_gap_frames=2,
                         max_move_px=40.0, max_tracks=8, max_duration_ms=400.0),
            camera=dict(width=FRAME[0], height=FRAME[1], fps=60.0))), encoding="utf-8")
        rows = ["frame,t_rel,n_dets,det_i,x,y,peak,area,w,h,elongation,process_ms"]
        for fi in range(100, 130):
            t = (fi - 100) / 60.0
            if fi in (110, 111):
                rows.append(f"{fi},{t:.5f},1,0,{xx},{yy},80,{area:.1f},10,10,1.0,1.0")
            else:
                rows.append(f"{fi},{t:.5f},0,,,,,,,,,1.0")
        (sdir / "detections.csv").write_text(chr(10).join(rows) + chr(10), encoding="utf-8")
        lg_ = tmp / f"{name}.log"
        lg_.write_text("10:00:00  recording to C:/x/recordings/" + name + chr(10),
                       encoding="utf-8")
        return sdir, lg_

    s_small, l_small = make_session("session-20260101-000001", small)
    s_big, l_big = make_session("session-20260101-000002", big)
    res_small = R.replay(s_small, cal6, l_small, "on", None, None, from_csv=True)
    res_big = R.replay(s_big, cal6, l_big, "on", None, None, from_csv=True)
    res_small_off = R.replay(s_small, cal6, l_small, "off", None, None, from_csv=True)
    check(f"at scale 0.5 a blob of {small:.0f} full-frame px (asked {need:.0f} at N {n_here:.2f}) "
          f"is gated: no shot announced",
          res_small["replay_total"] == 0, f"announced {res_small['replay_total']}")
    check(f"... a blob of {big:.0f} px survives: one shot",
          res_big["replay_total"] == 1, f"announced {res_big['replay_total']}")
    check("... and with the gate off the small one is announced (the gate, not the fixture, "
          "made the difference)",
          res_small_off["replay_total"] == 1, f"announced {res_small_off['replay_total']}")
    res = res_big
    sdir, log = s_big, l_big
    check("the result records the gain map it used",
          res["gain"]["shape"] == [720, 1280] and res["gain"]["max_n"] > 1.5, str(res["gain"]))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.report(res)
    text = buf.getvalue()
    check("report() prints the map: shape, minimum gain, maximum N",
          "gain map used: shape 720x1280" in text and "min gain" in text and "max N" in text,
          text.splitlines()[1] if len(text.splitlines()) > 1 else text)
    check("report() states which question the csv source answers, beside the table",
          "SOURCE csv" in text and "which shots survive" in text and "~99%" in text)
    res_v = dict(res, source="video")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.report(res_v)
    check("... and which one the video source answers (it is ~48% faithful)",
          "SOURCE video" in buf.getvalue() and "did I change anything" in buf.getvalue()
          and "~48%" in buf.getvalue())
    # rejected.csv: the detector's own rejections, in a file of their own.
    check("a session with no rejected.csv says NONE RECORDED, not 'none rejected'",
          res_big["rejected"] is None and R.read_rejected(s_big) is None)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.report(res_big)
    check("... in the report too", "NONE RECORDED" in buf.getvalue())
    (s_big / "rejected.csv").write_text(chr(10).join([
        "frame,t_rel,reason,x,y,peak,area,w,h,elongation",
        f"105,0.08333,big,{xx},{yy},60,4000.0,70.0,70.0,1.0",
        f"105,0.08333,amp_area,10.0,10.0,24,9.0,3.0,3.0,1.0",
        f"120,0.33333,big,{xx},{yy},55,3500.0,60.0,60.0,1.0"]) + chr(10), encoding="utf-8")
    rj = R.read_rejected(s_big)
    check("read_rejected counts by reason and indexes by frame",
          rj is not None and rj["by_reason"] == {"big": 2, "amp_area": 1}
          and sorted(rj["by_frame"]) == [105, 120], str(rj and rj["by_reason"]))
    res_rj = R.replay(s_big, cal6, l_big, "on", None, None, from_csv=True)
    check("the rejected rows are NOT mistaken for detections: same single shot as before",
          res_rj["replay_total"] == res_big["replay_total"] == 1
          and res_rj["rejected"] == {"big": 2, "amp_area": 1}, str(res_rj["rejected"]))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.report(res_rj)
    check("report() prints the rejected counts beside the source line",
          "rejected blobs recorded: amp_area 1, big 2" in buf.getvalue())
    wrong_log = tmp / "other.log"
    wrong_log.write_text("09:00:00  recording to C:\\x\\recordings\\session-20250101-000000\n",
                         encoding="utf-8")
    try:
        R.replay(sdir, cal6, wrong_log, "on", None, None, from_csv=True)
        refused = False
    except R.LogMismatch:
        refused = True
    check("replay() refuses a log from another session", refused)
    try:
        R.replay(sdir, cal3, log, "on", None, None, from_csv=True)
        refused2 = False
    except R.NoGainMap:
        refused2 = True
    check("replay() refuses a calibration with no usable map", refused2)

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    print(f"{results.count(False)} OF {len(results)} CHECKS FAILED")
    return 1


def guarded_main() -> int:
    """`main()` with a net under it, so a raise costs the checks AFTER it rather
    than the whole report.

    Why (2026-10-24, the same reason test_ghost grew this on 10-22): mutation
    faults came back BROKEN -- "the suite crashed rather than failing the named
    check" -- and measuring one by hand showed the check written for it HAD gone
    red, fifteen times, before a later section raised. The attribution was sound
    and unreadable, because a suite that dies prints no summary and the runner
    refuses, correctly, to read a crash as a catch.

    It does NOT make a crash a catch: the runner still needs a failing line
    naming the fault's own check, and the crash itself is recorded as a failing
    check, so the suite is red either way.
    """
    try:
        return main()
    except Exception as exc:                          # noqa: BLE001 - the finding
        import traceback
        check("the suite ran to its end without raising", False,
              f"{type(exc).__name__}: {exc}")
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"      raised at test_replay.py line {tb.lineno}: {tb.line}")
        print()
        print(f"{results.count(False)} OF {len(results)} CHECKS FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(guarded_main())
