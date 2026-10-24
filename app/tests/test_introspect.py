"""The in-process introspection server: off by default, read-only, and auditable.

Every section here is a property the agreed design states, or a scar it came
from. Nothing in this file opens a camera, takes the screen or plays a sound,
and it fingerprints the files the user owns (RULES C11).

The faults this server exists to answer, all from 2026-10-17 and all cheap to
query once something exposes them:

  * the exposure probe measured the CENTRE of the panel instead of the weak
    region and never logged which spot -- inferred hours later from an
    unrelated line;
  * `max_area` was 2000 in his settings against a code default of 1500,
    invisible until someone read a JSON file;
  * the sensitivity map was poisoned by a walking figure, discoverable only by
    reading `calibration.json` after shutdown;
  * pressing a key meant stealing focus with synthetic keystrokes, which
    knocked the target window behind another and tripped the screen-ownership
    guard;
  * seeing what the camera saw meant pulling frames out of an 815 MB AVI --
    whose re-encode destroys exactly the near-threshold noise being chased.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import ast
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from owned_files import OwnedFiles, isolate_outputs      # noqa: E402

OUT = isolate_outputs("introspect")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                        # noqa: E402
from PySide6.QtCore import QTimer                          # noqa: E402
from PySide6.QtWidgets import QApplication                 # noqa: E402

from lsa import introspect                                 # noqa: E402
from lsa.audio.listener import ShotCorroborator, _Reading  # noqa: E402
from silent_audio import silent_listener, silent_mixer     # noqa: E402
from lsa.calibrate.mapping import Mapping, Sample          # noqa: E402
from lsa.camera.capture import (CaptureConfig, CaptureThread,  # noqa: E402
                                GrouperConfig)
from lsa.detect.laser import (Detection, DetectorParams,    # noqa: E402
                              LaserDetector, Shot)
from lsa.director import Director, ManualClock             # noqa: E402
from lsa.display.scene import Scene                        # noqa: E402
from lsa.display.session import SessionStats               # noqa: E402
from lsa.introspect.readonly import (CAPTURE_READ, Reader,  # noqa: E402
                                     WriteAttempted, capture_reader)
from lsa.introspect.server import IntrospectServer, NotLocal  # noqa: E402
import lsa.modes as arcade                                  # noqa: E402
from lsa.modes.base import ModeHost                         # noqa: E402

PKG = ROOT / "src" / "lsa" / "introspect"

checks: list[bool] = []
FAILED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    checks.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


# ---------------------------------------------------------------------------
# a lab stand-in: only what LabView reads, and the REAL capture thread and
# detector inside it, never started. RULES C12 -- a stand-in that paraphrases
# the real thing excuses every consumer of the difference. The pieces that
# matter here (the detector, the tracker, the Stats dataclass) are the
# production classes.
# ---------------------------------------------------------------------------


class StandInLab:
    SINGLE_FRAME_PEAK = 30

    def __init__(self, params: "DetectorParams | None" = None) -> None:
        self.capture = CaptureThread(CaptureConfig(index=0),
                                     params or DetectorParams(threshold=8,
                                                              min_peak=14),
                                     GrouperConfig())
        self.session = SessionStats()
        self.scene = Scene(self.session)
        host = self.scene.add(ModeHost())
        self.layers = {"modes": host}
        host.set_mode(arcade.StaticTargets())
        self.director = Director(ManualClock())
        self.mapping = None
        self.target_window = None
        self.screen_quad = None
        # THE REAL CLASSES, never None. These three were None here, so
        # `/state` never once read a listener or a corroborator and five wrong
        # names in `_audio()` reported null for weeks behind a getattr default
        # (RULES C8, C12). `silent_audio` subclasses the real Mixer and
        # ClickListener and overrides only start()/stop()/play(): no output
        # stream and no input device is ever opened, so the properties and
        # counters the view reads are the production code's own.
        self.mixer = silent_mixer()
        self.listener = silent_listener()
        self.corroborator = ShotCorroborator()
        self.settings = None
        self._frames_ever = 0
        self._noise_peak = 1.0
        self._floor_measured = True
        self._floor_in_flight = False
        self._noise_retries = 0
        self._floor_deferred = False
        self._floor_before_gain = 1.0
        self._shooting_exposure = -5.0
        self._boot_exposure = -5.0
        self._calib_exposures = [-10.0, -9.0]
        self._sens_shots = []
        self._sens_applied = 0
        self._sens_ignored = {"faint": 0, "animating": 0}
        self._sens_admitted = 0          # mirrors CameraLab (RULES C17)
        self._sens_by_click = 0
        self._mapping_measured_now = False
        self._check_confirmed = False
        self._check_tried = False
        self._target_ever_opened = False
        self._cover_ok = True
        self._cover_suppressed = 0
        self._cover_causes = set()
        self._calibrating = False
        self._blind_devices = set()
        self._last_report = None
        self.logged: list[str] = []

    # the handful of methods LabView calls
    def _process_role(self) -> str:
        return "harness"

    def _screen_busy(self, *ignoring) -> str:
        return ""

    def _mm_per_px(self):
        return None

    def _mm_per_camera_px(self):
        return 3.0

    def _ready_for_real_shots(self) -> bool:
        return True

    def _app_is_drawing_now(self) -> bool:
        return False

    def _prompt_state(self):
        return ("a", "b", "c", "d")

    def log(self, msg: str) -> None:
        self.logged.append(msg)

    # -- helpers for the tests -------------------------------------------

    def feed(self, frame) -> list:
        """Run the REAL detector over a frame, exactly as the capture loop
        does, and leave the capture thread holding it."""
        dets = self.capture._detector.process(frame, timestamp=time.perf_counter())
        self.capture._last_frame = frame
        self.capture._stats.frames += 1
        return dets


class InlineBridge:
    """Runs the callable here and now. Used only where the property under test
    is not about threading; the REAL GuiBridge is exercised in section 8, on a
    real Qt event loop, because a suite that only ever drives the stand-in is
    testing itself (the QtClock lesson, RULES C12)."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, fn, timeout_s=None):
        self.calls += 1
        return fn()

    def stats(self) -> dict:
        return {"gui_calls": self.calls, "inline": True}


class CountingTap:
    KEEP = 40

    def __init__(self) -> None:
        self._ring: list = []
        self.seen = 0

    def record(self, shot) -> None:
        self.seen += 1
        self._ring.append((time.perf_counter(), shot))
        self._ring = self._ring[-self.KEEP:]

    def recent(self, n: int) -> list:
        return self._ring[-n:] if n > 0 else list(self._ring)


def red_dot_frame(w=320, h=240, at=(160, 120), radius=6, red=200, grey=6):
    """A dark frame with one saturated red spot: what a hit looks like."""
    import cv2
    f = np.full((h, w, 3), grey, dtype=np.uint8)
    cv2.circle(f, at, radius, (grey, grey, int(red)), -1)
    return f


def serve(lab, bridge=None, tap=None):
    ins = introspect.install(lab, port=0, bridge=bridge or InlineBridge(),
                             tap=tap or CountingTap(),
                             log=getattr(lab, "log", None))
    return ins


def get(port: int, path: str, method: str = "GET", raw: bool = False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read()
            status = r.status
            ctype = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
        ctype = exc.headers.get("Content-Type", "")
    if raw or not ctype.startswith("application/json"):
        return status, ctype, body
    return status, ctype, json.loads(body.decode("utf-8"))


def main() -> int:                                        # noqa: C901
    print("=== the introspection server: off by default, read-only, auditable ===")
    owned = OwnedFiles()
    app = QApplication.instance() or QApplication(sys.argv)

    # ------------------------------------------------------------------ 1 ---
    print("\n1. OFF BY DEFAULT, and that is a measurement rather than a promise")
    check("no flag and no env var means no server", introspect.port_from([], {}) == 0)
    check("--introspect uses the default port",
          introspect.port_from(["--introspect"], {}) == introspect.DEFAULT_PORT)
    check("--introspect=9100 is honoured",
          introspect.port_from(["--introspect=9100"], {}) == 9100)
    check("LSA_INTROSPECT=1 turns it on",
          introspect.port_from([], {"LSA_INTROSPECT": "1"}) == introspect.DEFAULT_PORT)
    check("LSA_INTROSPECT=8899 picks the port",
          introspect.port_from([], {"LSA_INTROSPECT": "8899"}) == 8899)
    for off in ("", "0", "off", "no", "false"):
        check(f"LSA_INTROSPECT={off!r} leaves it off",
              introspect.port_from([], {"LSA_INTROSPECT": off}) == 0)
    bad = False
    try:
        introspect.port_from([], {"LSA_INTROSPECT": "banana"})
    except ValueError:
        bad = True
    check("an unparseable value RAISES rather than quietly doing nothing", bad,
          "a flag that silently does nothing is the shape of four checks on "
          "this project that could not fail (RULES C1)")
    check("no introspection thread exists before anything installs one",
          not any(t.name == "lsa-introspect" for t in threading.enumerate()))

    # ------------------------------------------------------------------ 2 ---
    print("\n2. NO CAMERA CONTROL WRITE IS REACHABLE (RULES A4, constraint 1)")
    # A single CAP_PROP_EXPOSURE write has hung this camera for 272 seconds,
    # while five minutes of pure streaming returned 17,945 frames with zero
    # failures. Streaming is safe; control traffic is not.
    lab = StandInLab()
    rd = capture_reader(lab.capture)
    HAZARDS = ("set_control", "apply_preset", "start_exposure_search",
               "request_probe", "request_driver_dialog", "_post", "_cap",
               "_commands", "_mutex", "request_still",
               "start_noise_calibration", "start_room_measurement",
               "set_detector_params", "set_grouper_config", "set_gain_map",
               "reset_background", "reset_stats", "start_recording",
               "stop_recording", "save_snapshot", "stop", "set_aim_help",
               "set_show_mask", "set_draw_overlay")
    for name in HAZARDS:
        check(f"the real CaptureThread HAS {name}, and the Reader refuses it",
              hasattr(lab.capture, name) and _refuses(rd, name))
    check("and the refusal is a distinct exception, not an AttributeError",
          _refusal_type(rd, "set_control") is WriteAttempted)
    check("assigning through the Reader is refused too", _refuses_set(rd))
    check("every allowlisted name is a real attribute of the capture thread",
          all(hasattr(lab.capture, n) for n in CAPTURE_READ),
          f"missing: {sorted(n for n in CAPTURE_READ if not hasattr(lab.capture, n))}")

    # ...and behaviourally: booby-trap the real write methods and drive every
    # endpoint. A structural guard alone would pass a handler that got hold of
    # the real object some other way (RULES E4: assert the whole property).
    tripped: list[str] = []

    def trap(name):
        def boom(*_a, **_k):
            tripped.append(name)
            raise AssertionError(f"the introspection server called {name}")
        return boom

    for name in ("set_control", "apply_preset", "start_exposure_search",
                 "request_probe", "request_still", "start_noise_calibration",
                 "start_room_measurement", "set_gain_map", "reset_stats",
                 "reset_background", "stop"):
        setattr(lab.capture, name, trap(name))
    lab.feed(red_dot_frame())
    ins = serve(lab)
    paths = ("/", "/health", "/state", "/settings", "/frame", "/frame.json",
             "/frame?format=npy", "/score", "/score.png", "/score.npy",
             "/score?x=160&y=120&box=9", "/shots?n=5", "/drawn")
    statuses = {}
    for path in paths:
        st, _ct, _body = get(ins.port, path)
        statuses[path] = st
    st_key, _ct, _body = get(ins.port, "/key?key=g", method="POST")
    check("every endpoint answered", all(s in (200, 503) for s in statuses.values()),
          str(statuses))
    check("no endpoint touched a camera write", not tripped, str(tripped))
    check("Stats.control_writes did not move", lab.capture._stats.control_writes == 0)
    ins.stop()

    # ------------------------------------------------------------------ 3 ---
    print("\n3. LOOPBACK ONLY, refused before a socket exists")
    for host in ("0.0.0.0", "192.168.1.10", "", "example.com"):
        refused = False
        try:
            IntrospectServer({}, host=host, port=0)
        except NotLocal:
            refused = True
        check(f"refuses to bind to {host!r}", refused)
    ok_local = True
    try:
        s = IntrospectServer({}, host="127.0.0.1", port=0)
        ok_local = s.host == "127.0.0.1"
    except Exception as exc:                              # noqa: BLE001
        ok_local = False
        print(f"      {exc!r}")
    check("accepts 127.0.0.1", ok_local)

    # ------------------------------------------------------------------ 4 ---
    print("\n4. every endpoint returns the shape the design asks for")
    lab = StandInLab()
    lab.feed(red_dot_frame())
    ins = serve(lab)
    _st, _ct, health = get(ins.port, "/health")
    for key in ("ok", "version", "uptime_s", "camera_open", "server"):
        check(f"/health carries {key}", key in health)
    check("/health reports what the server itself has cost",
          "served_ms_max" in health["server"] and "requests" in health["server"],
          f"max {health['server'].get('served_ms_max')}ms")

    _st, _ct, state = get(ins.port, "/state")
    # FIRST, and before anything indexes into the body: the stand-in holds the
    # production listener, corroborator and mixer, so a wrong attribute name in
    # `_audio()` is an AttributeError here, i.e. HTTP 500 -- not a null. This
    # line is what a later crash on `state["detector"]` would otherwise hide.
    check("/state answers while the audio objects are real (a wrong name on "
          "them is HTTP 500 here, not a null)", _st == 200,
          f"HTTP {_st} {str(state)[:140] if _st != 200 else ''}")
    for key in ("mode", "layers", "detector", "tracker", "floor", "exposure",
                "gain_map", "mapping", "screen", "camera", "director",
                "session", "audio", "app_says"):
        check(f"/state carries {key}", key in state)
    check("/state gives the EFFECTIVE detector params, not the saved ones",
          state["detector"]["effective"]["min_peak"] == 14
          and state["detector"]["effective"]["threshold"] == 8)
    check("/state separates the measured floor from the bar it produced",
          "noise_peak" in state["floor"] and "bar_min_peak" in state["floor"])
    check("/state reports the control-write count",
          "control_writes" in state["camera"])
    check("/state reports the Director's overlaps, which must stay empty",
          state["director"]["overlaps"] == [])
    # THE SHAPE HIS REAL calibration.json CARRIES (copied from it, 2026-10-17):
    # `held_out` is a DICT, where this stand-in once gave a bare float and
    # /state crashed on `held * mm` against his real file (RULES C12: a
    # stand-in must mirror the real thing). Measured numbers, not invented.
    _REAL_HELD_OUT = {"median_px": 3.361877255761288, "worst_px": 5.290216559942521,
                      "median_mm": 0.8229595365665653, "worst_mm": 1.2950009287359294,
                      "points": [{"screen": [844.8, 475.2],
                                  "camera": [437.9796811756228, 210.38981342420794]},
                                 {"screen": [1920.0, 475.2],
                                  "camera": [655.4961731807068, 224.07889846257814]}]}
    _saved_mapping = lab.mapping
    lab.mapping = _identity_mapping()
    lab.mapping.held_out = _REAL_HELD_OUT
    try:
        _st, _ct, real_state = get(ins.port, "/state")
        check("/state survives a mapping whose held_out is the real dict", _st == 200,
              f"HTTP {_st}")
        mp = real_state.get("mapping", {}) if isinstance(real_state, dict) else {}
        check("...and reports the MEASURED held-out millimetres, median and worst",
              mp.get("held_out_mm") == 0.82296 and mp.get("held_out_worst_mm") == 1.295001,
              f"{mp.get('held_out_mm')} / {mp.get('held_out_worst_mm')}")
        check("...and the held-out pixels",
              mp.get("held_out_px") == 3.361877 and mp.get("held_out_worst_px") == 5.290217,
              f"{mp.get('held_out_px')} / {mp.get('held_out_worst_px')}")
        check("...and how many held-out points it was judged on",
              mp.get("held_out_points") == 2, f"{mp.get('held_out_points')}")
        lab.mapping.held_out = 3.5          # the legacy scalar must still work
        _st, _ct, legacy = get(ins.port, "/state")
        check("a scalar held_out (an older calibration.json) still reads",
              _st == 200 and legacy["mapping"]["held_out_px"] == 3.5,
              f"HTTP {_st}")
    finally:
        lab.mapping = _saved_mapping
    check("the app's own conclusions are quarantined under app_says",
          "note" in state["app_says"] and "verdict" not in state)
    check("and that note says why", "cross-check" in state["app_says"]["note"])

    _st, _ct, settings = get(ins.port, "/settings")
    check("/settings lists rows with effective, saved and default",
          settings["rows"] and all(
              {"effective", "saved_on_disk", "code_default"} <= set(r)
              for r in settings["rows"]))
    check("/settings names which differ from the code default",
          isinstance(settings["differs_from_default"], list))
    check("/settings covers max_area, the field that was invisible",
          any(r["key"] == "max_area" and r["group"] == "detector"
              for r in settings["rows"]))

    st, ctype, png = get(ins.port, "/frame", raw=True)
    check("/frame returns a PNG by default",
          st == 200 and ctype == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n",
          f"{st} {ctype} {len(png)} bytes")
    st, ctype, npy = get(ins.port, "/frame?format=npy", raw=True)
    check("/frame?format=npy returns the array itself",
          st == 200 and npy[:6] == b"\x93NUMPY", f"{st} {len(npy)} bytes")
    arr = np.load(io.BytesIO(npy))
    check("...and it is the frame, pixel for pixel",
          arr.shape == (240, 320, 3) and int(arr.max()) == 200,
          f"{arr.shape} max {int(arr.max())}")
    _st, _ct, fmeta = get(ins.port, "/frame.json")
    check("/frame.json gives the raw chroma spread without the image",
          "raw_chroma" in fmeta and "png_bytes" not in fmeta)

    _st, _ct, shots = get(ins.port, "/shots?n=5")
    check("/shots answers with an empty ring before anything is fired",
          shots["kept"] == 0 and shots["shots_seen_this_session"] == 0)
    ins.stop()

    # ...and with a real Shot in it. The per-frame `members` are the point:
    # the reported position is the FIRST frame of the pulse, not the brightest
    # and not the average, and a hole that disagrees with the dot is diagnosed
    # from the drift between them. Nothing else on the app keeps them, which
    # is why the tap exists at all.
    lab.mapping = _identity_mapping()
    tap = CountingTap()
    ins = serve(lab, tap=tap)
    t0 = time.perf_counter()
    members = [Detection(x=150.0 + i, y=90.0, area=300.0, peak=120 - 20 * i,
                         frame_index=40 + i, timestamp=t0 + i * 0.0167,
                         width=20.0, height=18.0, elongation=1.1)
               for i in range(3)]
    tap.record(Shot(x=150.0, y=90.0, nx=0.47, ny=0.38, frames=3, peak=120,
                    area=300.0, first_timestamp=t0, last_timestamp=t0 + 0.05,
                    duration_ms=67.0, index=7, track_id=2, elongation=1.1,
                    members=members))
    _st, _ct, shots = get(ins.port, "/shots?n=5")
    check("/shots returns the shot", shots["kept"] == 1, str(shots["kept"]))
    row = shots["rows"][0]
    check("...in camera pixels", row["camera"] == [150.0, 90.0])
    check("...mapped to SCREEN pixels through the current mapping",
          row["screen"] is not None
          and abs(row["screen"][0] - 1500.0) < 25.0,
          str(row["screen"]))
    check("...with every frame of the pulse, not just the reported one",
          len(row["members"]) == 3
          and [m["peak"] for m in row["members"]] == [120, 100, 80],
          str([m["peak"] for m in row["members"]]))
    check("...and each frame's offset from the first, in ms",
          row["members"][2]["t_offset_ms"] > 30.0,
          str(row["members"][2]["t_offset_ms"]))
    check("...and the peak as a multiple of the bar it had to clear",
          row["peak_over_bar"] is not None
          and abs(row["peak_over_bar"] - 120 / 14.0) < 0.1,
          str(row["peak_over_bar"]))
    check("...and the one-line evidence the log uses, so both agree",
          "first-frame peak 120" in (row.get("evidence_line") or ""),
          str(row.get("evidence_line"))[:110])

    st, _ct, drawn = get(ins.port, "/drawn")
    check("/drawn refuses clearly when there is no target window",
          st == 503 and "no target window" in drawn["error"],
          f"{st} {drawn.get('error')}")
    check("...and still reports the layer stack, which is what there IS to say",
          "layers" in drawn)
    ins.stop()

    # ------------------------------------------------------------------ 5 ---
    print("\n5. /score returns the detector's ARITHMETIC, as numbers")
    # THE QUESTION THIS SERVER EXISTS FOR, and the reason a PNG cannot answer
    # it: "the detector reported peak 9 at this pixel while the recorded frame
    # shows raw chroma 1 -- where did 9 come from?" An 8-bit image of the final
    # score clips, rounds, and says nothing about what a value MEANS.
    # WITH the background model on, because that is what the app runs and
    # the subtraction is one of the four terms being audited. Section 5 first
    # had it off, which made both background terms None and the check on them
    # vacuous -- a check that cannot fail (RULES C1).
    lab = StandInLab(DetectorParams(threshold=8, min_peak=14, blur=0,
                                    use_background=True))
    # A gain map that amplifies the right half, which is the real case: his
    # saved map amplifies the weakest region of the panel by 4x.
    gain = np.ones((240, 320), dtype=np.float32)
    # 0.3, NOT 0.25. A gain of 0.25 makes the amplification exactly 4.0, and
    # `int(4.0) == 4.0`, so a check on it cannot tell a float from a rounded
    # int -- which mutation 10 of `mutate_introspect.py` proved by surviving.
    # 0.3 gives 3.333..., where any rounding shows.
    gain[:, 160:] = 0.3
    lab.capture._detector.set_gain_map(gain)
    # A few empty frames so the model is a dark screen, then ONE dot frame,
    # which is what a real pulse looks like. Feeding the same dot repeatedly
    # teaches the model the dot and the score correctly falls to zero -- which
    # is the detector working, and would make this section measure nothing.
    for _ in range(4):
        lab.feed(red_dot_frame(at=(240, 120), radius=0, red=6))
    frame = red_dot_frame(at=(240, 120), red=40)
    lab.feed(frame)
    ins = serve(lab)
    _st, _ct, sc = get(ins.port, "/score?x=240&y=120&box=5")
    at = sc["at"]["at_the_centre"]
    print(f"      raw {at['raw_chroma']}  x amplification {at['amplification']}"
          f"  -> normalised {at['normalised']}  -> score {at['score_from_the_detector']}")
    check("all four terms are present as numbers",
          all(at[k] is not None for k in ("raw_chroma", "amplification",
                                          "normalised",
                                          "score_from_the_detector")))
    check("the amplification at a 0.3-gain pixel is 3.333x, reported as a "
          "float and not rounded",
          abs(at["amplification"] - 1.0 / 0.3) < 0.01, str(at["amplification"]))
    check("raw x amplification equals the normalised value, saturating at 255",
          at["normalised"] == min(255, round(at["raw_chroma"] * at["amplification"])),
          f"{at['raw_chroma']} x {at['amplification']} -> {at['normalised']}")
    check("the detector's own score matches the one recomputed here",
          at["score_from_the_detector"] == at["score_recomputed_here"],
          f"{at['score_from_the_detector']} vs {at['score_recomputed_here']}")
    check("both background terms are given, and they are DIFFERENT: the "
          "detector subtracts the model and then updates it, so the one on "
          "the object is a step ahead of the one that made this score",
          at["background_before_update"] is not None
          and at["background_after_update"] is not None,
          f"before {at['background_before_update']}, "
          f"after {at['background_after_update']}")
    check("and the response says the frame did not advance between the reads",
          sc["frame_and_score_are_the_same_frame"] is True)
    check("the score is bigger than the raw chroma, which is the gain map's "
          "whole contribution made visible by subtraction",
          at["score_from_the_detector"] > at["raw_chroma"],
          f"score {at['score_from_the_detector']} vs raw {at['raw_chroma']}")
    check("the neighbourhood is a grid of numbers, not an image",
          isinstance(sc["at"]["terms"]["raw_chroma"], list)
          and isinstance(sc["at"]["terms"]["raw_chroma"][0][0], int))
    check("every term is given over the neighbourhood",
          {"raw_chroma", "amplification", "normalised", "blurred",
           "background_before_update", "background_after_update",
           "score_from_the_detector", "score_recomputed_here"}
          <= set(sc["at"]["terms"]))
    _st, _ct, big = get(ins.port, "/score?x=160&y=120&box=9999")
    check("the box is bounded so one request cannot ask for the whole frame",
          big["at"]["box"] == 129, str(big["at"]["box"]))
    _st, _ct, auto = get(ins.port, "/score")
    check("with no coordinate it measures the brightest pixel of the score map",
          auto["at"]["chosen"].startswith("the brightest"))
    # Within the dot, not at its exact centre: a saturated blob has a FLAT
    # top, so many pixels share the maximum and cv2.minMaxLoc returns the
    # first of them. "The brightest pixel" is the honest measurement; the
    # centroid is what `blobs` reports, and both are offered.
    check("...and that is inside the dot (radius 6 at 240,120)",
          abs(auto["at"]["centre_full_frame_px"][0] - 240) <= 7
          and abs(auto["at"]["centre_full_frame_px"][1] - 120) <= 7,
          str(auto["at"]["centre_full_frame_px"]))

    check("/score reports the bars the numbers are to be read against",
          auto["bars"]["threshold"] == 8 and auto["bars"]["min_peak"] == 14)
    check("/score reports the gain map's amplification range",
          abs(auto["gain_map"]["max_amplification"] - 1.0 / 0.3) < 0.01,
          str(auto["gain_map"]["max_amplification"]))
    check("/score gives a 3x3 region grid, because a whole-frame maximum "
          "cannot say whether the far side is dimmer",
          len(auto["regions"]["cells"]) == 9)
    blobs = auto["blobs"]
    check("/score lists the RAW blobs above the threshold", blobs["rows"])
    row = blobs["rows"][0]
    for key in ("peak", "raw_chroma_peak", "local_amplification",
                "area_required_by_the_size_gate", "would_be_dropped_because",
                "would_be_reported"):
        check(f"each blob carries {key}", key in row)
    check("the blob list is explicit about why a blob would be dropped",
          isinstance(row["would_be_dropped_because"], list))
    st, ctype, spng = get(ins.port, "/score.png", raw=True)
    check("/score.png is offered as well, and is a real PNG",
          st == 200 and spng[:8] == b"\x89PNG\r\n\x1a\n")
    st, ctype, snpy = get(ins.port, "/score.npy", raw=True)
    sarr = np.load(io.BytesIO(snpy))
    check("/score.npy is the score array at its real dtype",
          st == 200 and sarr.dtype == np.uint8 and sarr.shape == (240, 320),
          f"{sarr.dtype} {sarr.shape}")
    check("...and it agrees with the number reported at the centre",
          int(sarr[120, 240]) == at["score_from_the_detector"])
    ins.stop()

    # ------------------------------------------------------------------ 6 ---
    print("\n6. /settings would have shown max_area = 2000 against a default of 1500")
    saved = OUT / "settings.maxarea.json"
    saved.write_text(json.dumps({"detector": {"max_area": 2000,
                                              "min_peak": 14},
                                 "tracker": {"min_frames": 1}}), encoding="utf-8")
    was = os.environ.get("LSA_SETTINGS")
    os.environ["LSA_SETTINGS"] = str(saved)
    try:
        lab = StandInLab(DetectorParams(max_area=2000, min_peak=14))
        lab.feed(red_dot_frame())
        ins = serve(lab)
        _st, _ct, se = get(ins.port, "/settings")
        rows = {(r["group"], r["key"]): r for r in se["rows"]}
        ma = rows[("detector", "max_area")]
        print(f"      max_area  effective {ma['effective']}  "
              f"saved {ma['saved_on_disk']}  default {ma['code_default']}")
        check("the effective value is reported", ma["effective"] == 2000)
        check("the saved value is read from disk", ma["saved_on_disk"] == 2000)
        check("the code default is reported beside it", ma["code_default"] == 1500)
        check("and it is named as differing from the default",
              "detector.max_area" in se["differs_from_default"])
        mf = rows[("tracker", "min_frames")]
        check("a SAVED value beating a code default is visible too: "
              "min_frames 1 on disk against a default of 2",
              mf["saved_on_disk"] == 1 and mf["code_default"] == 2
              and mf["effective"] == 2 and mf["differs_from_saved"],
              "min_frames sat at 1 for days because the file held the old "
              "value, and the log said frames=1")
        ins.stop()
    finally:
        if was is None:
            os.environ.pop("LSA_SETTINGS", None)
        else:
            os.environ["LSA_SETTINGS"] = was

    # ------------------------------------------------------------------ 7 ---
    print("\n7. /drawn measures what is PAINTED, in screen and camera pixels")
    # The measurement that identified the phantom detections at camera
    # (558,525) as the app's own SIGNAL METER, after I had diagnosed them as a
    # static object in his room and was about to build a position-identity
    # rule that would have rejected his real groups.
    import cv2
    lab = StandInLab()
    lab.mapping = _identity_mapping()
    view = introspect.LabView(lab, bridge=InlineBridge(), tap=CountingTap())
    panel = np.zeros((216, 384, 3), dtype=np.uint8)
    cv2.rectangle(panel, (100, 50), (180, 90), (40, 56, 80), -1)   # red excess 24
    cv2.rectangle(panel, (300, 150), (340, 180), (90, 90, 90), -1)  # grey, 0
    ink = view._measure_ink(panel, 3840, 2160, {})
    regions = sorted(ink["regions"], key=lambda r: -r["max_red_excess"])
    print(f"      worst red excess {ink['red_excess']['max']} at screen "
          f"{ink['red_excess']['max_at_screen']} -> camera "
          f"{ink['red_excess']['max_at_camera']}")
    check("the worst red excess on the screen is measured", ink["red_excess"]["max"] == 24)
    check("...and the limits it is to be judged against are stated",
          ink["red_excess"]["limit_static"] == 30
          and ink["red_excess"]["limit_moving_or_appearing"] == 8)
    check("...and it is located in SCREEN pixels",
          abs(ink["red_excess"]["max_at_screen"][0] - 100 * 10) < 60,
          str(ink["red_excess"]["max_at_screen"]))
    check("...and mapped into CAMERA pixels through the current mapping",
          ink["red_excess"]["max_at_camera"] is not None)
    check("the red region is reported and the grey one is not confused with it",
          regions[0]["max_red_excess"] == 24 and regions[-1]["max_red_excess"] == 0,
          f"{[r['max_red_excess'] for r in regions]}")
    check("each region's box is given as four camera corners",
          regions[0]["camera_box_corners"]
          and len(regions[0]["camera_box_corners"]) == 4)
    check("the lit AREA budget of RULES B1b is measured, not assumed",
          "lit_pct_above_luminance_60" in ink and ink["area_budget"]["limit_pct"] == 2.0)
    check("...and the budget verdict follows the measurement",
          ink["area_budget"]["within"] is (ink["lit_pct_above_luminance_60"] <= 2.0))
    dark = np.zeros((216, 384, 3), dtype=np.uint8)
    blank = view._measure_ink(dark, 3840, 2160, {})
    check("a BLANK screen measures as blank -- the fault he saw and no "
          "instrument did", blank["lit_pct"] == 0.0 and not blank["regions"])

    # ------------------------------------------------------------------ 8 ---
    print("\n8. the REAL GuiBridge, on a real Qt event loop")
    # A suite that only ever drives the stand-in is testing itself: 73 Director
    # checks were green while `QtClock` could not arm a single timer, because
    # every one of them ran on ManualClock (RULES C12).
    from lsa.introspect.gui import GuiBridge, ShotTap, post_key, resolve_key
    bridge = GuiBridge()
    where: list = []
    main_thread = threading.get_ident()
    done = threading.Event()
    result: dict = {}

    def worker() -> None:
        try:
            result["value"] = bridge.run(lambda: (where.append(threading.get_ident()), 42)[1])
            result["busy"] = _gui_busy(bridge)
        except Exception as exc:                          # noqa: BLE001
            result["error"] = repr(exc)
        finally:
            done.set()

    _pump(app, worker, done)
    check("the callable ran and its value came back", result.get("value") == 42,
          str(result))
    check("...and it ran on the Qt thread, not the caller's",
          where and where[0] == main_thread, f"{where} vs {main_thread}")
    busy_label, busy_stats = result.get("busy", ("?", {}))
    check("a Qt thread that never runs the work produces GuiBusy, not a hang",
          busy_label == "GuiBusy", str(busy_label))
    check("the bridge counts and reports its own cost on the Qt thread",
          bridge.stats()["gui_calls"] >= 1 and "gui_ms_max" in bridge.stats(),
          str(bridge.stats()))
    check("...and the bridge that timed out counts the timeout",
          busy_stats.get("gui_timeouts") == 1 and busy_stats.get("gui_calls") == 0,
          str(busy_stats))
    check("resolve_key handles a letter, a digit and a name",
          resolve_key("g") == 0x47 and resolve_key("0") == 0x30
          and resolve_key("escape") is not None and resolve_key("F11") is not None)
    check("and refuses what it does not know", resolve_key("hyperspace") is None)
    # MEASURED, not inferred from the base class, and the first version of
    # this check got the MECHANISM wrong while getting the outcome right.
    # It asserted `hasattr(tap, "metaObject")` on the theory that a QObject
    # receiver is what makes the connection queued. `mutate_introspect.py`
    # reported "the shot tap stops being a QObject" as NOT CAUGHT, and
    # measuring it showed that PySide6 delivers to a plain callable using the
    # SENDER's affinity -- and `CaptureThread` is a QThread whose QObject
    # lives on the GUI thread, so it was queued either way. The tap now asks
    # for `Qt.QueuedConnection` explicitly and this measures the property:
    # emit the real signal from another thread, as the capture loop does, and
    # see where `record` runs. A proxy that happens to correlate is worse than
    # no rule at all (RULES C6).
    class _WatchedTap(ShotTap):
        ran_on: list = []

        def record(self, shot) -> None:
            type(self).ran_on.append(threading.get_ident())
            ShotTap.record(self, shot)

    _WatchedTap.ran_on = []
    tap = _WatchedTap()
    cap2 = CaptureThread(CaptureConfig(index=0), DetectorParams(), GrouperConfig())
    tap.attach(cap2.shot_detected)
    emitted = threading.Event()

    def emit_from_elsewhere() -> None:
        cap2.shot_detected.emit(object())
        emitted.set()

    _pump(app, emit_from_elsewhere, emitted, extra_ms=300)
    app.processEvents()
    check("the shot ring is filled on the GUI thread, so the capture loop "
          "pays nothing for it: a QUEUED connection, not a direct one",
          _WatchedTap.ran_on == [main_thread],
          f"record() ran on {_WatchedTap.ran_on}, the Qt thread is "
          f"{main_thread}")
    check("...and the shot actually reached the ring", tap.seen == 1)

    # ------------------------------------------------------------------ 9 ---
    print("\n9. POST /key travels the real keyPressEvent path, with no focus taken")
    from lsa.display.window import TargetWindow
    lab = StandInLab()
    lab.feed(red_dot_frame())
    scene = Scene(lab.session)
    host = scene.add(ModeHost())
    host.set_mode(arcade.StaticTargets())
    lab.scene = scene
    lab.layers = {"modes": host}
    win = TargetWindow(scene)
    win.resize(640, 360)
    lab.target_window = win
    seen_keys: list[int] = []
    win.key_pressed.connect(seen_keys.append)
    bridge = GuiBridge()
    ins = serve(lab, bridge=bridge, tap=CountingTap())
    out: dict = {}
    done = threading.Event()

    def worker2() -> None:
        try:
            out["unknown"] = get(ins.port, "/key?key=hyperspace", method="POST")
            out["sent"] = get(ins.port, "/key?key=q", method="POST")
            out["drawn"] = get(ins.port, "/drawn")
        finally:
            done.set()

    _pump(app, worker2, done, extra_ms=400)
    app.processEvents()
    st, _ct, unknown = out["unknown"]
    check("an unknown key is refused with the accepted set named",
          st == 409 and "accepted" in unknown, f"{st}")
    st, _ct, sent = out["sent"]
    check("a known key is accepted", st == 200 and sent["qt_key"] == 0x51, f"{st}")
    check("...and it says it took no focus", sent["focus_taken"] is False)
    check("...and says the effect is asynchronous, so /state must be re-read",
          "ASYNCHRONOUS" in sent["note"])
    check("...and reports the state BEFORE it, so the caller can diff",
          "mode" in sent["before"] and "director_running" in sent["before"])
    check("the key really reached the window's own handler",
          0x51 in seen_keys, f"keys seen: {seen_keys}")
    check("...and it is LOGGED, so a change nobody at the machine caused is "
          "attributable", any("introspection: key" in m for m in lab.logged),
          str(lab.logged[-1:]))
    st, _ct, drawn = out["drawn"]
    check("/drawn now answers, through the real bridge and a real window",
          st == 200 and "lit_pct" in drawn, f"{st} {str(drawn)[:120]}")
    check("...and reports the layer stack with it", bool(drawn.get("layers")))
    ins.stop()
    win.close()

    # ----------------------------------------------------------------- 10 ---
    print("\n10. the server writes nothing, and nothing the user owns")
    writers = []
    for path in sorted(PKG.glob("*.py")):
        mod = ast.parse(io.open(path, encoding="utf-8", newline="").read())
        for node in ast.walk(mod):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else "")
                owner = (fn.value.id if isinstance(fn, ast.Attribute)
                         and isinstance(fn.value, ast.Name) else "")
                if name in ("write_text", "write_bytes", "unlink", "mkdir",
                            "rename", "replace", "rmtree", "remove"):
                    writers.append(f"{path.name}:{node.lineno} {name}")
                # `np.save` is allowed ONLY into a BytesIO, which is how
                # /score?format=npy returns the array itself. Flagged when it
                # is anything else, because `Mapping.save` and
                # `AppSettings.save` are the two calls that would reach his
                # files (RULES C11, C16).
                if name == "save" and owner not in ("np", "numpy"):
                    writers.append(f"{path.name}:{node.lineno} {owner}.save")
                if name == "open":
                    writers.append(f"{path.name}:{node.lineno} open()")
    check("no file-writing call anywhere in the package", not writers, str(writers))
    changed = owned.changed()
    check("the files he owns are byte-for-byte unchanged", not changed, str(changed))

    # ----------------------------------------------------------------- 11 ---
    print("\n11. WHAT IT COSTS THE CAPTURE LOOP, measured")
    # The detector has about 7 ms per camera frame and crossing ~14 ms halves
    # how many frames a 100 ms pulse lands in. It is a cliff, not a slope, so
    # the cost of being watched is a number, not a reassurance (RULES C3).
    cost = _measure_cost()
    for line in cost["lines"]:
        print("      " + line)
    # THE BAR IS DERIVED, NOT SET TO WHATEVER PASSES TODAY. The budget is the
    # 14 ms cliff: the capture loop is read + detect, and under about 14 ms of
    # detect it holds 60 fps while over it it falls to 30 -- which halves the
    # camera frames inside a 100 ms pulse and is what decides whether a
    # trigger pull registers. His real detector measures ~7.3 ms a frame at
    # 1280x720 with a gain map applied, so there are 6.7 ms of headroom and
    # this server may spend a seventh of it at the median and a third at the
    # 95th percentile. Setting a threshold to today's reading and calling the
    # rule enforced is what this project has done before.
    HEADROOM_MS = 6.7
    # A seventh of the headroom at the median, a third at the 95th. Not
    # today's reading: this machine runs the whole suite in parallel with
    # whatever else is on it, and a bar set to the best number seen is a bar
    # that fails for the wrong reason.
    check(f"an agent at ~16 requests a second over every endpoint costs the "
          f"capture loop under {HEADROOM_MS / 7:.1f} ms at the median",
          cost["median_delta_ms"] < HEADROOM_MS / 7.0,
          f"{cost['median_delta_ms']:+.2f} ms")
    check(f"...and under {HEADROOM_MS / 3:.1f} ms at the 95th percentile, a "
          f"third of the headroom before the 14 ms cliff",
          cost["p95_delta_ms"] < HEADROOM_MS / 3.0,
          f"{cost['p95_delta_ms']:+.2f} ms")
    check("...and the absolute p95 under that load stays well under 14 ms",
          cost["p95_under_load_ms"] < 12.0,
          f"{cost['p95_under_load_ms']:.2f} ms")
    check("a SATURATED client -- not a usage pattern, a stress test -- stays "
          "inside the whole headroom", cost["sat_p95_delta_ms"] < HEADROOM_MS,
          f"{cost['sat_p95_delta_ms']:+.2f} ms")
    check("and so does a two-client flood of the cheapest endpoint",
          cost["flood_p95_delta_ms"] < HEADROOM_MS,
          f"{cost['flood_p95_delta_ms']:+.2f} ms")
    check("the load was real: the paced client actually served requests",
          cost["requests"] >= 8, f"{cost['requests']} requests in ~0.9s of "
          f"detector loop, which is the point: /score is ~16ms of work, so a "
          f"60ms pause gives about 13 a second")

    # ----------------------------------------------------------------- 12 ---
    print("\n12. every lab attribute it reads EXISTS on the real CameraLab")
    # RULES C8, one object removed. `view.py` reads the app's state through
    # `getattr(lab, "_x", default)`, which is the construct that returned 0.0
    # for `stats.last_shot_at` for days while a guard never fired and looked
    # exactly like protection. Here the risk is quieter and worse: a rename in
    # `camera_lab.py` would make an endpoint report the DEFAULT, so /state
    # would answer "the floor was never measured" about an app that had
    # measured it, and the agent reading it would be confidently wrong.
    #
    # So the names are extracted from the source and checked against a real
    # CameraLab, rather than kept in a hand-written list -- a guard that names
    # a few things by hand will be incomplete (RULES E2), and this one would
    # go stale the first time an endpoint was added.
    import re
    src = io.open(ROOT / "src" / "lsa" / "introspect" / "view.py",
                  encoding="utf-8", newline="").read()
    names = set()
    for pat in (r'getattr\((?:self\._lab|lab), "([A-Za-z_][A-Za-z0-9_]*)"',
                r'_call\((?:self\._lab|lab), "([A-Za-z_][A-Za-z0-9_]*)"',
                r'(?:self\._lab|\blab)\.([A-Za-z_][A-Za-z0-9_]*)'):
        names.update(re.findall(pat, src))
    names.discard("log")               # the view's own optional hook
    check("the extraction found the attributes, so this check is not vacuous",
          len(names) >= 25, f"{len(names)} names")

    import lsa.tools.camera_lab as cl
    cl.CameraLab.hard_exit_on_close = False
    real = cl.CameraLab()
    try:
        real.settings.auto_connect = False
        real._save_settings = lambda *a, **k: None       # belt; the path is
        real._schedule_save = lambda *a, **k: None       # redirected too
        check("a CameraLab built with no flag and no env var has no server",
              real.introspect is None)
        check("...and no introspection thread was started",
              not any(t.name == "lsa-introspect" for t in threading.enumerate()))
        missing = sorted(n for n in names if not hasattr(real, n))
        check("every name view.py reads off the lab exists on the real one",
              not missing,
              f"missing: {missing} -- each one would be reported as its "
              f"DEFAULT, so an endpoint would answer confidently and wrongly")
        # And the endpoints answer against the REAL object, not only the
        # stand-in. A stand-in that paraphrases excuses every consumer of the
        # difference (RULES C12).
        ins = serve(real, bridge=InlineBridge(), tap=CountingTap())
        _st, _ct, st = get(ins.port, "/state")
        check("/state answers against a real CameraLab",
              st["camera"]["open"] is False and "floor" in st,
              str(st.get("camera", {}).get("open")))
        check("...and reports its real layer stack",
              len(st["layers"]) >= 10, f"{len(st['layers'])} layers")
        check("...and its real active mode",
              st["mode"]["active"] is not None, str(st["mode"].get("active")))
        _st, _ct, se = get(ins.port, "/settings")
        check("/settings answers against a real CameraLab",
              any(r["group"] == "app" and r["key"] == "calibrate_seconds"
                  for r in se["rows"]))
        ins.stop()
    finally:
        real.director.close("the test is done")
        real.close()
        app.processEvents()

    # ---------------------------------------------------------------- 12b ---
    print("\n12b. a wrong name on the app's OTHER objects fails loudly, "
          "and the audio path is CALLED against real ones")
    # Section 12 checks the names read off the LAB. The same fault one object
    # down was untouched: `_audio()` read five names that do not exist on the
    # real listener and corroborator (`device_name`, `threshold`, `clicks`,
    # `heard_only`, `seen_only`) through `getattr(x, "name", None)`, so
    # /state reported null for the microphone threshold, the click count and
    # BOTH heard-versus-seen mismatch counts -- the numbers the gun test is
    # built on -- for weeks, and no check could notice because the stand-in
    # held `None` where the listener should be and the code under test never
    # reached a single attribute. Null looks like an answer; absent prompts a
    # question. (RULES C8, C12; the same three-times-over fault as `stats` for
    # `_stats`, `last_shot_at` and `announcer` for `voice`.)
    #
    # Two halves, because either alone is the kind of check this project has
    # shipped hollow:
    #   (a) STRUCTURAL -- the package may not read the app's own objects
    #       through a default at all, so a SIXTH wrong name cannot hide the
    #       way the first five did, whatever it is called;
    #   (b) BEHAVIOURAL -- the real code path is called against the real
    #       classes, populated, and a wrong name is shown to come back as an
    #       error rather than as a plausible empty value.
    allowed_defaults = {
        # (what it is, why a default is the true answer there)
        "getattr(v, 'value', v)":
            "enum-or-plain duck typing on a value, not an app attribute",
        "getattr(m, 'held_out', None)":
            "`held_out` is not a declared field of Mapping: the app attaches "
            "it after the held-out fit, so 'not judged yet' is the true answer",
        "getattr(Qt, 'Key_' + candidate, None)":
            "resolves a caller-supplied key NAME against Qt; None is the "
            "documented 'unknown key' answer",
    }
    seen_defaults: dict[str, str] = {}
    offenders: list[str] = []
    bad_call: list[str] = []
    for pyf in sorted(PKG.glob("*.py")):
        tree = ast.parse(io.open(pyf, encoding="utf-8", newline="").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id == "getattr" and (len(node.args) >= 3 or node.keywords):
                text = ast.unparse(node)
                if text in allowed_defaults:
                    seen_defaults[text] = pyf.name
                else:
                    offenders.append(f"{pyf.name}:{node.lineno}: {text}")
            if node.func.id == "_call" and len(node.args) != 2:
                bad_call.append(f"{pyf.name}:{node.lineno}: {ast.unparse(node)}")
    check("the introspection package reads the app's own objects through NO "
          "getattr default (a wrong name must raise, not report a default)",
          not offenders, "; ".join(offenders[:6]))
    check("...and `_call` takes no default either: a missing method is an "
          "error, not a quiet None", not bad_call, "; ".join(bad_call[:6]))
    stale = sorted(set(allowed_defaults) - set(seen_defaults))
    check("every allowlisted default is still in the source, so the list "
          "cannot outlive the lines it excuses", not stale, f"{stale}")

    # (b) the real classes, populated with distinct nonzero values so that a
    # wrong name that happened to read a default of 0 cannot coincide.
    alab = StandInLab()
    check("the stand-in holds the REAL listener, corroborator and mixer, "
          "not None",
          type(alab.corroborator) is ShotCorroborator
          and hasattr(alab.listener, "coupling_report")
          and alab.mixer is not None)
    L, C = alab.listener, alab.corroborator
    L.start()                         # silent: a sentinel stream, no device
    L.device = 3
    L.min_level = 0.2
    L.level, L.floor, L.heard = 0.31, 0.012, 4
    C.both, C.heard_not_seen, C.seen_not_heard = 3, 5, 7
    check("no audio device is opened by any of this",
          alab.mixer._stream is None
          and type(L._stream).__name__ == "_NoStream")
    ains = serve(alab)
    _st, _ct, ast_ = get(ains.port, "/state")
    # Every read below is `.get`, never `[...]`: a wrong name now makes /state
    # answer 500 (that is the point), and a KeyError here would end the suite
    # without a [FAIL] line, which `mutate_introspect` would count as a broken
    # run rather than a caught fault.
    check("/state answers while the audio objects are real and populated",
          _st == 200, f"HTTP {_st} {str(ast_)[:160]}")
    au = (ast_.get("audio") or {}) if _st == 200 else {}
    check("/state reports the heard-versus-seen counts that were null for weeks",
          (au.get("heard_and_seen"), au.get("heard_not_seen"),
           au.get("seen_not_heard")) == (3, 5, 7),
          f"{au.get('heard_and_seen')}/{au.get('heard_not_seen')}/"
          f"{au.get('seen_not_heard')}")
    check("...and the microphone's device, click count and click level",
          (au.get("mic_device"), au.get("clicks_heard"),
           au.get("mic_min_level")) == (3, 4, 0.2),
          f"{au.get('mic_device')}/{au.get('clicks_heard')}/"
          f"{au.get('mic_min_level')}")
    check("...and its adaptive floor and margin, which set the bar in force",
          au.get("mic_floor") == 0.012 and au.get("mic_margin") == L.margin,
          f"{au.get('mic_floor')} / {au.get('mic_margin')}")
    nulls = sorted(k for k, v in au.items() if v is None)
    check("NOTHING in /state.audio is null when every object is real and "
          "populated -- structural, so a sixth wrong name is caught whatever "
          "it is called", bool(au) and not nulls, f"null: {nulls}")
    cp = au.get("coupling") or {}
    check("the coupling is served: unknown, with its (zero) evidence",
          cp.get("coupling", 1) is None and cp.get("clean") == 0
          and cp.get("source") == "", str(cp))
    check("...and its keys are the listener's own report, key for key",
          set(cp) == set(L.coupling_report()), f"{sorted(cp)}")
    for _i in range(L.COUPLING_NEEDED):
        L._readings.append(_Reading(time.perf_counter(), 0.8, "bang", 0.5, "shot"))
    _st, _ct, ast2 = get(ains.port, "/state")
    cp = ((ast2.get("audio") or {}).get("coupling") or {}) if _st == 200 else {}
    check("once the listener has measured its coupling, /state says so, with "
          "the source and the readings behind it",
          cp.get("coupling") == 0.8 and cp.get("source") == "bang"
          and cp.get("clean") == L.COUPLING_NEEDED, str(cp))

    # A wrong name is a LOUD failure: delete one real attribute and the
    # endpoint must answer 500 naming it, not 200 with a null.
    del C.heard_not_seen
    _st, _ct, broken = get(ains.port, "/state")
    check("a name that does not exist on the real object makes /state FAIL "
          "and names it, instead of answering null",
          _st == 500 and "heard_not_seen" in json.dumps(broken),
          f"HTTP {_st} {str(broken)[:120]}")
    C.heard_not_seen = 5

    # The rest of the sub-objects the view reads that a bare stand-in leaves
    # empty: the flash quad, the app's own report, the last shot's outcome.
    from lsa.calibrate.screen import ScreenQuad
    from lsa.modes.base import ShotOutcome
    from lsa.selfcheck import Finding, Report, Verdict
    alab.screen_quad = ScreenQuad(corners=((0, 0), (1, 0), (1, 1), (0, 1)),
                                  coverage=0.4, confidence=0.9,
                                  touches_edges=(), frame_size=(320, 240))
    alab._last_report = Report(Verdict.READY, [Finding(
        Verdict.DEGRADED, "headline-x", "detail-x", "action-x")], "summary-x")
    alab.layers["modes"].last_outcome = ShotOutcome(hit=True, points=9,
                                                    target_id="ring-9")
    _st, _ct, ast3 = get(ains.port, "/state")
    check("the flash quad, the app's report and the last outcome are read "
          "off their real classes",
          _st == 200
          and ast3["screen"]["flash_quad_confidence"] == 0.9
          and ast3["app_says"]["verdict"] == "ready"
          and ast3["app_says"]["findings"][0]["headline"] == "headline-x"
          and ast3["mode"]["last_outcome"]
          == {"hit": True, "points": 9, "target_id": "ring-9"},
          str(ast3.get("app_says", {}).get("findings")))
    ains.stop()

    # ----------------------------------------------------------------- 13 ---
    print(chr(10) + "13. and AFTER all of that, his files are still untouched")
    # The fingerprint in section 10 was taken before section 12 built a real
    # CameraLab -- which loads settings, builds a scene and can save both a
    # calibration and a settings file. A check placed before the thing that
    # could break it is one of this project's four that could not fail
    # (RULES C1, C11). So it is taken again, last.
    changed = owned.changed()
    check("calibration.json, settings.local.json and targets/ are byte-for-byte "
          "as they were, after a real CameraLab was built and served",
          not changed, str(changed))
    check("...and the paths were redirected, not merely the writers neutered",
          str(OUT) in os.environ.get("LSA_SETTINGS", "")
          and str(OUT) in os.environ.get("LSA_CALIBRATION", "")
          and str(OUT) in os.environ.get("LSA_LOG", ""),
          "neutering is a patch and a patch can be routed around -- the "
          "harness once wrote his calibration through a path nobody had "
          "blocked")

    print()
    if all(checks):
        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    for name in FAILED:
        print(f"  - {name}")
    return 1


# --- helpers ---------------------------------------------------------------


def _refuses(rd, name: str) -> bool:
    try:
        getattr(rd, name)
        return False
    except WriteAttempted:
        return True
    except Exception:                                     # noqa: BLE001
        return False


def _refusal_type(rd, name: str):
    try:
        getattr(rd, name)
    except Exception as exc:                              # noqa: BLE001
        return type(exc)
    return None


def _refuses_set(rd) -> bool:
    try:
        rd.anything = 1
        return False
    except WriteAttempted:
        return True
    except Exception:                                     # noqa: BLE001
        return False


def _gui_busy(bridge):
    """A bridge whose Qt thread will not run the work must raise, not hang.

    RULES D4: every wait has a deadline. A cover whose removal depended on a
    camera frame that never came once left the display an immovable black
    rectangle, escapable only by unplugging the camera.
    """
    from lsa.introspect.gui import GuiBusy
    dead = type(bridge)()
    dead._submit.disconnect()          # nothing will ever run the task
    label = "no exception"
    try:
        dead.run(lambda: 1, timeout_s=0.05)
    except GuiBusy:
        label = "GuiBusy"
    except Exception as exc:                              # noqa: BLE001
        label = repr(exc)
    return label, dead.stats()


def _pump(app, worker, done, extra_ms: int = 0) -> None:
    """Run `worker` on a thread while the Qt event loop turns, then stop."""
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    timer = QTimer()
    timer.setInterval(10)
    state = {"grace": extra_ms}

    def tick() -> None:
        if done.is_set():
            if state["grace"] <= 0:
                timer.stop()
                app.quit()
                return
            state["grace"] -= 10
    timer.timeout.connect(tick)
    timer.start()
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(app.quit)
    guard.start(15000)                 # RULES D4: never an unbounded wait
    app.exec()
    timer.stop()
    t.join(timeout=5.0)


def _identity_mapping() -> Mapping:
    """A mapping where screen (x,y) is camera (x/10, y/10). Enough to prove the
    screen-to-camera direction is wired, which is all /drawn needs from it."""
    samples = [Sample(camera=(cx, cy), screen=(cx * 10.0, cy * 10.0))
               for cx, cy in ((20.0, 20.0), (300.0, 20.0),
                              (300.0, 200.0), (20.0, 200.0),
                              (160.0, 110.0), (80.0, 60.0))]
    m = Mapping.fit(samples, (3840, 2160))
    return m


def _measure_cost() -> dict:
    """Run the REAL detector in a tight loop, with and without the server being
    used, and report what being watched costs it.

    AT HIS REAL FRAME SIZE, 1280x720 MJPG, which is what `CaptureConfig` and
    his `settings.local.json` run. Measuring at 640x480 is not a harder test,
    it is a different one: the loop iterates four times faster, so it asks for
    the GIL four times as often and collides four times as much, and the same
    code measured +4.1 ms p95 there against +1.3 ms at the size he actually
    shoots at. A harness must not make the test easier than the bench
    (RULES C15) -- and it must not measure a bench that does not exist either.

    Two loads, because they fail differently:

      * REALISTIC -- one client, the endpoint mix an agent uses, no pause.
        This is already far harsher than a human with curl.
      * A FLOOD -- two clients hammering the cheapest endpoint, which
        maximises HTTP and Python churn rather than per-request work. This is
        the pathological case, reported so the number is on record; nothing an
        agent does looks like it.
    """
    import statistics
    frames = [red_dot_frame(1280, 720, at=(600 + i, 360), red=180)
              for i in range(8)]

    def loop(n: int) -> list:
        det = LaserDetector(DetectorParams(threshold=8, min_peak=14))
        out = []
        for i in range(n):
            t0 = time.perf_counter()
            det.process(frames[i % len(frames)], timestamp=t0)
            out.append((time.perf_counter() - t0) * 1000.0)
        return out[10:]                 # drop the warm-up

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * q))]

    def under(paths, clients: int, pause: float = 0.0):
        lab = StandInLab()
        lab.feed(red_dot_frame(1280, 720))
        ins = serve(lab)
        stop = threading.Event()
        served = {"n": 0}

        def hammer() -> None:
            i = 0
            while not stop.is_set():
                try:
                    get(ins.port, paths[i % len(paths)])
                    served["n"] += 1
                except Exception:                         # noqa: BLE001
                    pass
                i += 1
                if pause:
                    time.sleep(pause)

        threads = [threading.Thread(target=hammer, daemon=True)
                   for _ in range(clients)]
        for t in threads:
            t.start()
        time.sleep(0.3)                 # let the load get going
        xs = loop(130)
        stop.set()
        for t in threads:
            t.join(timeout=3.0)
        ins.stop()
        return xs, served["n"]

    base = loop(130)
    bm, b95 = statistics.median(base), pct(base, 0.95)

    mix = ("/state", "/score", "/frame", "/score?x=600&y=360&box=33",
           "/settings", "/shots?n=10", "/health", "/drawn")
    # An agent at 16 requests a second, cycling every endpoint. A human with
    # curl manages perhaps one a second; this is already generous.
    agent, n_agent = under(mix, 1, pause=0.06)
    # The same mix with no pause at all: the server thread busy essentially
    # 100% of the time, which is not a usage pattern, it is a stress test.
    sat, n_sat = under(mix, 1)
    # Two clients on the CHEAPEST endpoint: maximum HTTP and Python churn
    # rather than maximum per-request work. The pathological case.
    flood, n_flood = under(("/state",), 2)
    am, a95 = statistics.median(agent), pct(agent, 0.95)
    sm, s95 = statistics.median(sat), pct(sat, 0.95)
    fm, f95 = statistics.median(flood), pct(flood, 0.95)

    return {
        "median_delta_ms": am - bm,
        "p95_delta_ms": a95 - b95,
        "sat_p95_delta_ms": s95 - b95,
        "flood_p95_delta_ms": f95 - b95,
        "p95_under_load_ms": a95,
        "requests": n_agent,
        "lines": [
            f"detector alone, 1280x720      median {bm:6.2f} ms   p95 {b95:6.2f} ms",
            f"+ an agent at ~16 req/s over  median {am:6.2f} ms   p95 {a95:6.2f} ms"
            f"   ({n_agent} requests)",
            f"  every endpoint: cost        median {am - bm:+6.2f} ms   "
            f"p95 {a95 - b95:+6.2f} ms",
            f"+ the same with NO pause      median {sm:6.2f} ms   p95 {s95:6.2f} ms"
            f"   ({n_sat} requests)",
            f"  saturated: cost             median {sm - bm:+6.2f} ms   "
            f"p95 {s95 - b95:+6.2f} ms",
            f"+ a /state flood, 2 clients   median {fm:6.2f} ms   p95 {f95:6.2f} ms"
            f"   ({n_flood} requests)",
            f"  cost                        median {fm - bm:+6.2f} ms   "
            f"p95 {f95 - b95:+6.2f} ms",
            "the budget: detect must stay under 14 ms, where the capture loop "
            "drops 60 fps -> 30 and halves the frames inside a 100 ms pulse",
        ],
    }


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
        print(f"      raised at test_introspect.py line {tb.lineno}: {tb.line}")
        print()
        print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(guarded_main())
