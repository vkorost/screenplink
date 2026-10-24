"""Replay a recorded session through the REAL detector and score it against the log.

    python tools/replay.py recordings/session-YYYYMMDD-HHMMSS
    python tools/replay.py SESSION --save-json before.json
    python tools/replay.py SESSION --save-json after.json
    python tools/replay.py --compare before.json after.json

WHY THIS EXISTS. A recording is 800 MB of frames the camera really saw while
the app really ran. The detector is deterministic, so replaying those frames
through it with the session's own parameters and the saved gain map is a
repeatable regression: change the detector, run this, and read what happened
to every shot the live app reported, instead of arguing about it.

WHAT IT RECONSTRUCTS, from files the session left behind:

  settings.json   beside the video: the detector and grouper parameters
  calibration.json  the saved mapping and its sensitivity regions -> the gain
                  map, built exactly as `CameraLab._restore_sensitivity` does
  camera_lab.log  the live app's own account: every SHOT line, with the
                  microphone level it carried

GROUND TRUTH is the microphone, which does not care what the camera thinks. A
SHOT line is

  CORROBORATED    mic_peak at or above the bar and not marked DEAF: a click
                  was heard when the camera saw a dot
  DEAF            the app's own gunshot had the microphone muted, so no click
                  could be counted. NOT evidence of a phantom, and not counted
                  as one: it is its own bucket, honestly labelled
  UNCORROBORATED  the microphone was listening, and heard nothing
  UNCONSULTED     the app had not yet decided the detection was a trigger pull

LIMITS, stated so they are not discovered later:

  * The video is MJPG re-encoded from the frames the live detector saw, so
    chroma noise differs a little. The tool reports how many live shots the
    replay reproduces; that number IS the fidelity.
  * Parameters changed mid-session (the noise floor was measured twice). The
    switch frame is placed from the log's wall clock, which has one-second
    resolution; the tool reports whether any detection falls inside that
    uncertainty.
  * Wall clock -> frame uses the first recorded frame as the anchor.

READ-ONLY. Opens capture.avi, settings.json, detections.csv, the log and the
calibration for reading, writes nothing in `recordings/`, and refuses to be
told to. Opens no camera, no screen and no audio device.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402  (after lsa._env via the lsa import below)
import numpy as np  # noqa: E402

from lsa.calibrate.mapping import (Mapping, Sample,  # noqa: E402
                                   confined_sensitivity_map)
from lsa.detect.laser import (Detection, DetectorParams, LaserDetector,  # noqa: E402
                              ScoreMode, ShotTracker, amplified_min_area,
                              full_frame_to_working_area)

MATCH_PX = 15.0     # re-encoding moves large-blob centroids by up to ~8 px
BUCKETS = ("corroborated", "deaf", "uncorroborated", "unconsulted")

_SHOT = re.compile(
    r"^(\d\d:\d\d:\d\d)\s+SHOT\s+(\d+)\s+cam=\(\s*([\d.]+),\s*([\d.]+)\)"
    r".*?frames=(\d+).*?peak=(\d+)\s+area=(\d+)px"
    r".*?\bt=([\d.]+)\s+mic_peak=([\d.]+|not consulted)(?: \(bar ([\d.]+))?")
_THRESH = re.compile(r"^(\d\d:\d\d:\d\d)\s+noise floor over .*Threshold set to (\d+), "
                     r"min peak to (\d+)")
_RECORDING = re.compile(r"^(\d\d:\d\d:\d\d)\s+recording to (.+?)\s*$")


class NoGainMap(RuntimeError):
    """The calibration cannot produce a gain map, so a replay would prove nothing.

    The size gate only acts where the map amplifies. With no map, the gate-on
    and gate-off tables are identical and read like a clean regression proof.
    A reviewer hit exactly this on their first run, so it is an error now.
    """


class LogMismatch(RuntimeError):
    """The log does not belong to the session it is being used to judge."""


def _secs(hms: str) -> int:
    h, m, s = (int(v) for v in hms.split(":"))
    return h * 3600 + m * 60 + s


def parse_log(path: Path) -> dict:
    shots, thresholds, rec_wall, rec_dir = [], [], None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _RECORDING.match(line)
        if m:
            rec_wall = _secs(m[1])          # the LAST recording start wins
            rec_dir = m[2]
            thresholds = []                 # settings of an earlier launch
            shots = []
            continue
        m = _THRESH.match(line)
        if m:
            thresholds.append((_secs(m[1]), int(m[2]), int(m[3])))
            continue
        m = _SHOT.match(line)
        if m:
            mic, bar = m[9], m[10]
            if mic == "not consulted":
                bucket = "unconsulted"
            elif "[DEAF" in line:
                bucket = "deaf"
            elif bar is not None and float(mic) >= float(bar):
                bucket = "corroborated"
            else:
                bucket = "uncorroborated"
            shots.append(dict(id=int(m[2]), wall=_secs(m[1]), x=float(m[3]),
                              y=float(m[4]), frames=int(m[5]), peak=int(m[6]),
                              area=int(m[7]), t=float(m[8]), bucket=bucket))
    return dict(shots=shots, thresholds=thresholds, rec_wall=rec_wall,
                rec_dir=rec_dir)


def check_log_matches_session(live: dict, session: Path, log: Path) -> None:
    """Fail loudly unless the log's last 'recording to' line names this session.

    `--log` defaults to the LIVE camera_lab.log, and `parse_log` keeps only the
    last recording's shots. Pointed at an older session, the tool would pair
    that session's frames with another session's SHOT lines and print a table.
    """
    rec = live.get("rec_dir")
    if rec is None:
        raise LogMismatch(
            f"{log} has no 'recording to' line, so it cannot be checked against "
            f"{session.name}. Pass the log of that session with --log.")
    logged = Path(rec.replace("\\", "/")).name
    if logged.lower() != session.name.lower():
        raise LogMismatch(
            f"{log} describes the recording {logged!r}, not {session.name!r}. "
            f"Its SHOT lines belong to another session; pass that session's "
            f"log with --log (the default is the live camera_lab.log).")


def build_params(settings: dict, gate: str,
                 overrides: dict[str, str] | None = None) -> DetectorParams:
    d = dict(settings["detector"])
    d["mode"] = ScoreMode(d["mode"])
    if d.get("roi_polygon"):
        d["roi_polygon"] = tuple(tuple(p) for p in d["roi_polygon"])
    known = {f.name for f in fields(DetectorParams)}
    d = {k: v for k, v in d.items() if k in known}
    if gate == "off" and "amp_area_per_gain" in known:
        d["amp_area_per_gain"] = 0.0
    # `--set name=value`: run the SESSION'S OWN parameters with one of them
    # moved, which is what makes a before/after an A/B rather than two runs of
    # the same thing. Typed from the dataclass field, and an unknown name is
    # REFUSED rather than ignored -- comparing two identical configs would
    # otherwise read as "the change costs nothing".
    decls = {f.name: f for f in fields(DetectorParams)}
    for name, raw in (overrides or {}).items():
        if name not in decls:
            raise SystemExit(f"replay.py: --set {name}: no such detector "
                             f"parameter. Known: {', '.join(sorted(known))}")
        kind = type(decls[name].default)
        d[name] = (raw.lower() in ("1", "true", "yes") if kind is bool
                   else kind(raw))
    return DetectorParams(**d)


def build_gain(calibration: Path, frame_size) -> np.ndarray:
    """The gain map as `_restore_sensitivity` builds it. Reads, never writes.

    RAISES `NoGainMap` rather than returning None: a replay without a map runs
    to completion and prints two identical tables.
    """
    mapping = Mapping.load(calibration)
    if mapping is None:
        raise SystemExit(f"cannot read a mapping from {calibration}")
    rows = list(mapping.sensitivity or [])
    # The gain map is built through THE SAME HELPER THE APP USES, so a replay
    # cannot build one the live detector would refuse. `sensitivity_map(region
    # =None)` quietly means "the whole camera frame", which is the 2026-10-14
    # 21:33 incident (floor 19 -> 112, min_peak 127, every real hit rejected).
    # Called directly, this instrument would silently have measured the room
    # along with the panel and reported it as a regression number -- a harness
    # modelling something live does not do, which is how `fake_camera` measured
    # its noise floor differently from the camera for days.
    if len(rows) < 4:
        raise NoGainMap(
            f"{calibration} holds {len(rows)} sensitivity rows (4 are needed), so "
            f"there is no gain map and the size gate has nothing to act on.")
    samples = [Sample(screen=(0.0, 0.0),
                      camera=(float(r["camera"][0]), float(r["camera"][1])),
                      peak=float(r["peak"])) for r in rows]
    corners = mapping.screen_rect_in_camera()
    gain = confined_sensitivity_map(samples, frame_size, corners)
    if gain is None:
        raise NoGainMap(f"{calibration}: no gain map could be built from "
                        f"{len(rows)} rows (no screen outline, or the peaks "
                        f"give no usable map), so the size gate has nothing "
                        f"to act on.")
    return gain


def describe_gain(gain: np.ndarray, rows: int) -> dict:
    """What `report` prints about the map a replay used."""
    lo = float(gain.min())
    return dict(shape=[int(gain.shape[0]), int(gain.shape[1])], rows=int(rows),
                min_gain=round(lo, 4), max_n=round(1.0 / max(lo, 1e-3), 3))


def gate_keeps(params: DetectorParams, amplification: float, area_full_frame: float,
               gate: str = "on") -> bool:
    """Would the size gate let a csv detection through?

    UNITS. `detections.csv` stores `Detection.area`, which is already FULL-FRAME
    px; `amplified_min_area` answers in PROCESSING-scale px. They coincide only
    at scale 1.0, so the csv value is put in the gate's unit before comparing.
    One function so `--from-csv` and its test share the comparison.
    """
    if gate != "on":
        return True
    return (full_frame_to_working_area(params, area_full_frame)
            >= amplified_min_area(params, amplification))


def amplification_at(gain: np.ndarray, x: float, y: float) -> float:
    h, w = gain.shape[:2]
    xi = min(w - 1, max(0, int(round(x))))
    yi = min(h - 1, max(0, int(round(y))))
    return 1.0 / max(float(gain[yi, xi]), 1e-3)


def read_csv_times(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["frame"]), float(r["t_rel"]))
    return out


def read_rejected(session: Path) -> "dict | None":
    """The detector's own rejections for a session, from `rejected.csv`.

    A SEPARATE file from detections.csv on purpose: every reader of
    detections.csv takes any row with an `x` for a detection, so rejected
    blobs mixed into it would silently become shots. Returns None for a
    session recorded before the file existed (so "none recorded" is never
    confused with "none rejected"), else {"rows": [...], "by_reason": {...},
    "by_frame": {frame: [row, ...]}}. Rows keep the csv's column names.
    """
    path = session / "rejected.csv"
    if not path.exists():
        return None
    rows, by_reason, by_frame = [], defaultdict(int), defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            by_reason[r["reason"]] += 1
            by_frame[int(r["frame"])].append(r)
    return dict(rows=rows, by_reason=dict(by_reason), by_frame=dict(by_frame))


def replay(session: Path, calibration: Path, log: Path, gate: str,
           limit: int | None, dump: Path | None, from_csv: bool = False,
           overrides: dict[str, str] | None = None) -> dict:
    settings = json.loads((session / "settings.json").read_text(encoding="utf-8"))
    frame_t = read_csv_times(session / "detections.csv")
    first = min(frame_t)
    live = parse_log(log)
    check_log_matches_session(live, session, log)
    gain = build_gain(calibration, (settings["camera"]["width"],
                                    settings["camera"]["height"]))
    gain_info = describe_gain(gain, len(Mapping.load(calibration).sensitivity or []))
    params = build_params(settings, gate, overrides)
    det = LaserDetector(params)
    det.set_gain_map(gain)
    g = settings["grouper"]
    fps = float(settings["camera"]["fps"])
    tracker = ShotTracker(min_frames=g["min_frames"],
                          single_frame_peak=g.get("single_frame_peak", 0),
                          max_gap_frames=g["max_gap_frames"],
                          max_move_px=g["max_move_px"],
                          frame_period_ms=1000.0 / max(1.0, fps),
                          max_tracks=g["max_tracks"],
                          max_duration_ms=g["max_duration_ms"])

    # Parameter timeline. The recording starts with the settings in
    # settings.json (written at start); later noise-floor results change
    # threshold and min_peak. Wall clock -> frame via the first frame.
    switches = []
    if live["rec_wall"] is not None:
        t0 = frame_t[first]
        for wall, thr, mp in live["thresholds"]:
            if wall > live["rec_wall"]:
                switches.append((t0 + (wall - live["rec_wall"]), thr, mp))
    switches.sort()

    replay_shots, dump_rows = [], []
    k = 0
    if from_csv:
        # The LIVE detector's own output, run through the area gate and the
        # real tracker. Exact where the video is not: the video is re-encoded
        # and its near-threshold noise is not the noise the live detector
        # saw. The gate is a per-blob filter, so applying it to the blobs the
        # live detector kept is the same decision it would have made, except
        # for second-order effects (a gated blob no longer suppresses a
        # dimmer neighbour or takes one of `max_blobs` slots).
        n_frames = len(frame_t)
        by_frame = defaultdict(list)
        with (session / "detections.csv").open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["x"]:
                    by_frame[int(r["frame"])].append(r)
        for fi in sorted(frame_t):
            dets = []
            for r in by_frame.get(fi, []):
                x, y, area = float(r["x"]), float(r["y"]), float(r["area"])
                amp = amplification_at(gain, x, y)
                if not gate_keeps(params, amp, area, gate):    # see its UNITS note
                    continue
                dets.append(Detection(x=x, y=y, area=area, peak=int(r["peak"]),
                                      frame_index=fi, timestamp=frame_t[fi]))
            tracker.feed(dets, fi)
            for sh in tracker.take_provisional():
                replay_shots.append(dict(
                    x=sh.x, y=sh.y, t_rel=sh.first_timestamp, frame=fi,
                    peak=sh.peak, area=sh.area, frames=sh.frames,
                    amp=amplification_at(gain, sh.x, sh.y) if gain is not None else 1.0))
            k += 1
            if limit and k >= limit:
                break
        cap = None
    else:
        cap = cv2.VideoCapture(str(session / "capture.avi"))
        if not cap.isOpened():
            raise SystemExit("cannot open capture.avi")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pending = list(switches)
    while cap is not None:
        ok, frame = cap.read()
        if not ok or (limit and k >= limit):
            break
        fi = first + k
        t = frame_t.get(fi, frame_t[first] + k / fps)
        while pending and t >= pending[0][0]:
            _, thr, mp = pending.pop(0)
            p = DetectorParams(**vars(det.params))
            p.threshold, p.min_peak = thr, mp
            det.set_params(p)
        dets = det.process(frame, timestamp=t)
        tracker.feed(dets, fi)
        for s in tracker.take_provisional():
            replay_shots.append(dict(
                x=s.x, y=s.y, t_rel=s.first_timestamp, frame=fi,
                peak=s.peak, area=s.area, frames=s.frames,
                amp=amplification_at(gain, s.x, s.y) if gain is not None else 1.0))
        if dump is not None:
            for d in dets:
                dump_rows.append((fi, t, d.x, d.y, d.peak, d.area,
                                  amplification_at(gain, d.x, d.y) if gain is not None else 1.0))
        k += 1
    if cap is not None:
        cap.release()
    if dump is not None:
        with dump.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t_rel", "x", "y", "peak", "area", "amp"])
            w.writerows(dump_rows)

    # Place each live SHOT line on a frame. The live csv holds every
    # detection with its exact position and frame, and a SHOT line carries
    # the first frame's position to two decimals, so the pair is unambiguous
    # except for static spots; those are resolved by the time offset between
    # the log's perf_counter `t` and the csv's `t_rel`, which is the mode of
    # the unambiguous pairs.
    by_pos = defaultdict(list)
    with (session / "detections.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["x"]:
                by_pos[(round(float(r["x"]), 2), round(float(r["y"]), 2))].append(
                    (int(r["frame"]), float(r["t_rel"])))
    offsets = []
    for ls in live["shots"]:
        c = by_pos.get((ls["x"], ls["y"]), [])
        if len(c) == 1:
            offsets.append(round(ls["t"] - c[0][1], 1))
    offset = statistics.mode(offsets) if offsets else None
    for ls in live["shots"]:
        ls["frame"] = None
        if offset is None:
            continue
        c = [(fr, tr) for fr, tr in by_pos.get((ls["x"], ls["y"]), [])
             if abs((ls["t"] - tr) - offset) < 0.15]
        if c:
            ls["frame"] = min(c, key=lambda p: abs((ls["t"] - p[1]) - offset))[0]
    # A replay shot reproduces a live one when it starts within a few frames
    # and a few pixels: the video is re-encoded, so centroids move by a few
    # pixels on large blobs.
    matched, used = {}, set()
    for ls in live["shots"]:
        if ls["frame"] is None:
            continue
        best, best_d = None, 1e9
        for i, rs in enumerate(replay_shots):
            if i in used or abs(rs["frame"] - ls["frame"]) > 3:
                continue
            dist = ((rs["x"] - ls["x"]) ** 2 + (rs["y"] - ls["y"]) ** 2) ** 0.5
            if dist < MATCH_PX and dist < best_d:
                best, best_d = i, dist
        if best is not None:
            used.add(best)
            matched[ls["id"]] = best
    # Only live shots whose frame was replayed are comparable.
    last = first + k - 1
    comparable = [s for s in live["shots"]
                  if s["frame"] is not None and first <= s["frame"] <= last]
    for s in comparable:
        s["amp"] = amplification_at(gain, s["x"], s["y"]) if gain is not None else 1.0
    extra = [rs for i, rs in enumerate(replay_shots) if i not in used]
    near_switch = [sw for sw in switches
                   if any(abs(r[1] - sw[0]) < 1.5 for r in dump_rows)] if dump else None
    rej = read_rejected(session)
    return dict(
        rejected=None if rej is None else rej["by_reason"],
        session=str(session), frames_read=k, frames_in_video=n_frames,
        gate=gate, source="csv" if from_csv else "video",
        params=dict(threshold=params.threshold, min_peak=params.min_peak,
                               min_area=params.min_area,
                               max_area=params.max_area,
                               oversize_min_peak=params.oversize_min_peak,
                               oversize_core_frac=params.oversize_core_frac),
        overrides=dict(overrides or {}),
        gain=gain_info,
        switches=[(round(a, 2), b, c) for a, b, c in switches],
        near_switch=near_switch, offset=offset,
        live=comparable, survived=sorted(matched), replay_total=len(replay_shots),
        extra=extra)


# One sentence each, next to the table, saying which question the source can
# answer. The numbers are the measured fidelities of the two sources.
QUESTION = {
    "video": "SOURCE video: the AVI is re-encoded, so this reproduces only ~48% of the live "
             "shots; it answers 'did I change anything', NOT 'which shots survive'.",
    "csv": "SOURCE csv: detections.csv is the live detector's own output, so this reproduces "
           "~99% of the live shots; it answers 'which shots survive'. What the live "
           "detector threw away is in rejected.csv (sessions recorded after 2026-10-17) "
           "and is COUNTED below, not yet fed back through the tracker.",
}


def _dist(vals: list[float]) -> str:
    if not vals:
        return "-"
    v = sorted(vals)
    q = lambda f: v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]  # noqa: E731
    return f"p10 {q(.1):.0f}  median {q(.5):.0f}  p90 {q(.9):.0f}"


def report(res: dict) -> None:
    live, surv = res["live"], set(res["survived"])
    print(f"session {res['session']}")
    print(f"frames replayed {res['frames_read']} of {res['frames_in_video']}; "
          f"source={res.get('source', 'video')}; gate={res['gate']}; initial threshold {res['params']['threshold']}, "
          f"min_peak {res['params']['min_peak']}, min_area {res['params']['min_area']}")
    g = res.get("gain")
    print("gain map used: " + ("NONE RECORDED (a result saved before this was printed)"
                               if not g else
                               f"shape {g['shape'][0]}x{g['shape'][1]}, {g['rows']} sensitivity rows, "
                               f"min gain {g['min_gain']}, max N {g['max_n']}"))
    print(QUESTION.get(res.get("source", "video"), ""))
    if res.get("source") == "csv":
        rj = res.get("rejected")
        print("rejected blobs recorded: " + (
            "NONE RECORDED (this session predates rejected.csv, or the result was saved "
            "before it was printed)" if rj is None else
            (", ".join(f"{k} {v}" for k, v in sorted(rj.items())) or "none rejected")))
    print(f"parameter switches (t_rel, threshold, min_peak): {res['switches']}"
          + ("" if res["near_switch"] is None else
             f"; detections within 1.5 s of a switch: {bool(res['near_switch'])}"))
    print(f"live SHOT lines inside the recording: {len(live)}; replay announced "
          f"{res['replay_total']}; reproduced {len(surv)} of {len(live)} "
          f"({100.0 * len(surv) / max(1, len(live)):.0f}%)")
    print()
    print(f"{'bucket':<16}{'live':>6}{'kept':>6}{'kept%':>7}   area (kept)               peak (kept)")
    for b in BUCKETS:
        rows = [s for s in live if s["bucket"] == b]
        kept = [s for s in rows if s["id"] in surv]
        print(f"{b:<16}{len(rows):>6}{len(kept):>6}"
              f"{(100.0 * len(kept) / len(rows) if rows else 0):>6.0f}%   "
              f"{_dist([s['area'] for s in kept]):<30}{_dist([s['peak'] for s in kept])}")
    print()
    print("the two cuts the verdict is read from (live -> kept):")
    for label, pred in (
            ("corroborated, peak >= 30 (the project's laser/noise boundary)",
             lambda s: s["bucket"] == "corroborated" and s["peak"] >= 30),
            ("corroborated, peak < 30 (a click coincided with a faint blob)",
             lambda s: s["bucket"] == "corroborated" and s["peak"] < 30),
            ("corroborated, far side (camera x >= 760)",
             lambda s: s["bucket"] == "corroborated" and s["x"] >= 760),
            ("not corroborated, far side, peak < 30 (the phantom signature)",
             lambda s: s["bucket"] != "corroborated" and s["x"] >= 760 and s["peak"] < 30),
            ("not corroborated, far side, peak >= 30 (a real shot the mic missed)",
             lambda s: s["bucket"] != "corroborated" and s["x"] >= 760 and s["peak"] >= 30)):
        rows = [s for s in live if pred(s)]
        print(f"  {label:<66}{len(rows):>5} ->{len([s for s in rows if s['id'] in surv]):>5}")
    first_live = min((s["frame"] for s in live), default=0)
    late = [e for e in res["extra"] if e["frame"] >= first_live]
    print(f"announced by the tracker but never logged live: {len(res['extra'])}, "
          f"of which {len(late)} after the first live SHOT line (before it, or with the target screen closed, the app's own "
          f"routines owned the detections or it was ignoring them, and logged no SHOT)")
    print()
    print("by local amplification (1/gain at the shot), live -> kept")
    bands = ((0, 1.25), (1.25, 2.0), (2.0, 3.0), (3.0, 99))
    print(f"{'bucket':<16}" + "".join(f"{f'N {lo}-{hi}':>16}" for lo, hi in bands))
    for b in BUCKETS:
        cells = []
        for lo, hi in bands:
            rows = [s for s in live if s["bucket"] == b and lo <= s["amp"] < hi]
            kept = [s for s in rows if s["id"] in surv]
            cells.append(f"{len(rows):>7} ->{len(kept):>4}")
        print(f"{b:<16}" + "".join(f"{c:>16}" for c in cells))


def compare(a_path: Path, b_path: Path) -> None:
    a, b = (json.loads(p.read_text()) for p in (a_path, b_path))
    ka, kb = set(a["survived"]), set(b["survived"])
    live = {s["id"]: s for s in a["live"]}
    print(f"A {a_path.name}  gate={a['gate']}   B {b_path.name}  gate={b['gate']}")
    print(f"{'bucket':<16}{'live':>6}{'A kept':>8}{'B kept':>8}{'lost A->B':>11}{'B/A %':>8}")
    for bk in BUCKETS:
        ids = [i for i, s in live.items() if s["bucket"] == bk]
        ka_b = [i for i in ids if i in ka]
        kb_b = [i for i in ids if i in kb]
        lost = [i for i in ka_b if i not in kb]
        print(f"{bk:<16}{len(ids):>6}{len(ka_b):>8}{len(kb_b):>8}{len(lost):>11}"
              f"{(100.0 * len(kb_b) / len(ka_b) if ka_b else 0):>7.0f}%")
    print("\ncorroborated shots lost A->B:")
    for i in sorted(ka - kb):
        s = live.get(i)
        if s and s["bucket"] == "corroborated":
            print(f"  SHOT {i}: cam=({s['x']:.0f},{s['y']:.0f}) peak {s['peak']} "
                  f"area {s['area']} amp {s['amp']:.2f}")
    print(f"\nreplay-only shots: A {len(a['extra'])}  B {len(b['extra'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session", nargs="?", type=Path)
    ap.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "camera_lab.log")
    ap.add_argument("--gate", choices=("on", "off"), default="on",
                    help="'off' zeroes the amplification area gate, if the detector has one")
    ap.add_argument("--from-csv", action="store_true",
                    help="skip the video: gate the live detector's own detections.csv "
                         "and run the real tracker (exact, fast)")
    ap.add_argument("--limit", type=int, default=None, help="replay only the first N frames")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override one detector parameter, e.g. "
                         "--set oversize_min_peak=0 for the old area ceiling")
    ap.add_argument("--save-json", type=Path)
    ap.add_argument("--dump", type=Path, help="write every detection to this csv")
    ap.add_argument("--report", type=Path, metavar="JSON",
                    help="print the report for a result saved with --save-json")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return 0
    if args.report:
        report(json.loads(args.report.read_text()))
        return 0
    if not args.session:
        ap.error("a session directory is required")
    rec = (ROOT / "recordings").resolve()
    for out in (args.save_json, args.dump):
        if out is not None and rec in Path(os.path.abspath(out)).parents:
            raise SystemExit(f"refusing to write inside recordings/: {out}")
    try:
        overrides = {}
        for item in args.set:
            if "=" not in item:
                ap.error(f"--set wants NAME=VALUE, got {item!r}")
            k, v = item.split("=", 1)
            overrides[k.strip()] = v.strip()
        res = replay(args.session, args.calibration, args.log, args.gate,
                     args.limit, args.dump, args.from_csv, overrides)
    except (NoGainMap, LogMismatch) as exc:
        raise SystemExit(f"replay.py: {exc}")
    report(res)
    if args.save_json:
        args.save_json.write_text(json.dumps(res), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
