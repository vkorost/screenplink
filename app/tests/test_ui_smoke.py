"""Headless smoke test: build the whole UI, render the target screen to a PNG.

Runs with QT_QPA_PLATFORM=offscreen so it needs no display and no camera. It
constructs the real Camera Lab window (which exercises every widget, signal
connection and layer), then paints the target scene into an image with some
fake shots in it so the result can be inspected like a screenshot.

Run:  .venv\\Scripts\\python.exe tests\\test_ui_smoke.py [out.png]
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import pathlib
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# BEFORE camera_lab is imported: its LOG_PATH and RECORDINGS are
# module-level constants read at import time. This test builds a real
# CameraLab and publishes a verdict, and without this it wrote nine
# `GATE role=live` lines into camera_lab.log -- the LIVE log -- which
# tools/status.py would then offer as the last measurement on record.
from owned_files import (isolate_outputs, assert_silent_mixer,       # noqa: E402
                         assert_silent_listener)
isolate_outputs("ui_smoke")


import numpy as np                                                       # noqa: E402
from PySide6.QtCore import QRect, Qt                                     # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap              # noqa: E402
from PySide6.QtWidgets import QApplication                               # noqa: E402

from lsa.audio.mixer import synth_shot, synth_tick                       # noqa: E402
from lsa.display.layers import CameraPane, SolidBackground               # noqa: E402
from lsa.tools.camera_lab import CameraLab                               # noqa: E402
# THE HOLD (2026-10-20): see test_modes.py. These checks are about the close, not the hold.
from lsa.modes.base import ModeHost as _MH0                          # noqa: E402
REAL_HOLD_S = _MH0.HOLD_S
_MH0.HOLD_S = 0.0

from owned_files import OwnedFiles                                       # noqa: E402

WIDTH, HEIGHT = 1920, 1080


def fake_camera_frame(w: int = 640, h: int = 360) -> QImage:
    """A plausible camera view: dim screen rectangle seen at an angle."""
    buf = np.full((h, w, 3), 18, np.uint8)
    import cv2

    quad = np.array([[90, 70], [560, 110], [530, 300], [120, 265]], np.int32)
    cv2.fillPoly(buf, [quad], (46, 44, 42))
    cv2.polylines(buf, [quad], True, (90, 88, 84), 1)
    cv2.circle(buf, (330, 190), 4, (90, 100, 240), -1)
    cv2.circle(buf, (330, 190), 2, (255, 255, 255), -1)
    return QImage(buf.data, w, h, buf.strides[0], QImage.Format_BGR888).copy()


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("target_screen.png")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # FINGERPRINT HIS FILES BEFORE ANYTHING IS BUILT.
    #
    # This test already neuters `_save_settings` and `_schedule_save`, and
    # that is the defence that has failed three times: a test left
    # `auto_connect: false` in settings.local.json and stopped the app
    # connecting on launch; the exposure persisted itself from the live
    # control; the cold-start harness wrote its simulated 800x800 mapping
    # over his real one through a path nobody had blocked.
    #
    # Neutering is a patch and a patch can be routed around. The
    # fingerprint cannot: it compares bytes at the end and fails (C11 asks
    # for both defences, and this is the second one).
    owned = OwnedFiles()

    print("building Camera Lab window...")
    # Never touch the real camera from a test. The app auto-connects on launch,
    # so simply constructing it opened the device -- and on Windows two
    # processes opening the same camera leaves the driver in a state where it
    # stops honouring exposure writes. That is indistinguishable from a badly
    # aimed camera and cost most of a session to find.
    import lsa.settings as _settings
    _orig_load = _settings.AppSettings.load

    @staticmethod
    def _no_connect(path):
        st = _orig_load(path)
        st.auto_connect = False
        return st

    _settings.AppSettings.load = _no_connect
    try:
        lab = CameraLab()
    finally:
        _settings.AppSettings.load = _orig_load
    assert lab.capture is None, "the smoke test must not open the camera"
    assert_silent_mixer(lab)
    assert_silent_listener(lab)

    # THE REAL CAMERA IS UNREACHABLE FROM THIS SUITE. `auto_connect` is restored
    # to True just below (so saving is not poisoned), and several checks end in
    # something that calls `_connect`; with the default factory that opened camera
    # index 2 -- the Kiyo Pro -- five times in one run, while the live app held it
    # (2026-10-19; RULES A7: never open the camera from two processes). A factory
    # that refuses makes the open impossible rather than merely avoided.
    def _refuse_real_camera(*_a, **_k):
        raise RuntimeError("test_ui_smoke must never open a real camera")
    lab.capture_factory = _refuse_real_camera
    # Put the real value back and make saving impossible. Forcing auto_connect
    # off is a TEST concern; leaking it into the user's settings.local.json left
    # the app refusing to connect on launch, which is a far worse bug than the
    # one being prevented. A test must not be able to write the user's config.
    lab.settings.auto_connect = True
    lab._save_settings = lambda *a, **k: None
    lab._schedule_save = lambda *a, **k: None
    lab.resize(1500, 960)
    lab.show()
    app.processEvents()

    checks = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")

    check("main window constructed", lab.isVisible())
    # 2026-10-17: his settings.local.json carried max_area 2000 against the
    # code's 1500, which let a 1,899 px bloom through as a shot. The control
    # cannot accept more than the code's own limit, so no saved value can.
    from lsa.detect.laser import DetectorParams as _DP
    saved_detector = dict(lab.settings.detector)
    lab.settings.detector["max_area"] = 2000
    _logged = []
    _orig_log = lab._log
    lab._log = lambda m, _o=_orig_log: (_logged.append(m), _o(m))[1]
    lab._apply_settings_to_widgets()
    lab._log = _orig_log
    check("a saved max_area of 2000 cannot take effect (max_area clamp)",
          lab.sp_maxarea.value() == _DP.max_area == 1500,
          f"the control holds {lab.sp_maxarea.value()}")
    check("...and the clamp is logged, with the reason",
          any("max_area" in m and "clamped" in m for m in _logged),
          "; ".join(_logged)[:120])
    lab.settings.detector.clear()
    lab.settings.detector.update(saved_detector)
    check("the control's ceiling is the code's own default (max_area clamp)",
          lab.sp_maxarea.maximum() == _DP.max_area)
    check("control rows built for every camera control",
          len(lab.rows) == 14, f"{len(lab.rows)} rows")
    check("devices enumerated into the combo", lab.cmb_device.count() > 0,
          f"{lab.cmb_device.count()} devices")
    check("backends offered", lab.cmb_backend.count() > 0,
          ", ".join(lab.cmb_backend.itemText(i) for i in range(lab.cmb_backend.count())))
    check("scene has every layer", len(lab.scene.layers) == 17,
          ", ".join(la.name for la in lab.scene.layers))
    # Regression guard: recording and exposure buttons must be visibly dead
    # until a camera is streaming. They used to look pressable and then whisper
    # "not connected" into the log pane, which reads as the app being broken.
    check("recording disabled while unconnected", not lab.btn_record.isEnabled())
    check("camera-dependent widgets all disabled",
          all(not w.isEnabled() for w in lab._needs_camera),
          f"{sum(w.isEnabled() for w in lab._needs_camera)} still enabled "
          f"of {len(lab._needs_camera)}")
    check("control rows disabled too",
          all(not r.isEnabled() for r in lab.rows.values()))
    check("status banner says so", "NOT CONNECTED" in lab.lbl_status.text())
    lab._set_connected(True, "1280x720 @30 fps")
    check("everything enables on connect",
          lab.btn_record.isEnabled() and all(w.isEnabled() for w in lab._needs_camera))
    check("banner flips to CONNECTED", lab.lbl_status.text().startswith("CONNECTED"))
    lab._set_connected(False)

    check("detector params assemble", lab._detector_params().max_blobs == 4)
    check("tracker config assembles", lab._tracker_config().max_tracks == 4)

    # Audio buffers: synthesised, so verify they are sane rather than audible.
    shot = synth_shot()
    tick = synth_tick()
    check("shot sample non-silent and normalized",
          0.5 < float(np.abs(shot).max()) <= 1.0 and len(shot) > 1000,
          f"{len(shot)} samples, peak {float(np.abs(shot).max()):.2f}")
    check("tick sample non-silent", float(np.abs(tick).max()) > 0.1)

    # Drive the scene the way live shots would, including two concurrent tracks.
    print("injecting fake shots...")
    for nx, ny, track in ((0.50, 0.50, 0), (0.22, 0.30, 0), (0.78, 0.72, 1),
                          (0.30, 0.74, 1), (0.66, 0.28, 0)):
        lab.session.register_shot(camera_x=nx * 1280, camera_y=ny * 720, scored=None,
                                  track_id=track)
        lab.layers["hits"].add(nx, ny, track)
    check("session counted the shots", lab.session.shots == 5, f"{lab.session.shots}")
    check("splits recorded", len(lab.session.splits_ms) == 4,
          f"{len(lab.session.splits_ms)}")
    check("hits stay unknown without calibration", lab.session.hit_rate is None)

    # Put real holes on the screen: the mapped positions of the shots actually
    # fired in the 2026-10-13 session, so the render shows a true outcome.
    holes = lab.layers["holes"]
    for i, (sx, sy) in enumerate([(1691, 1104), (2077, 1289), (1947, 1195), (2108, 1243),
                                  (294, 1244), (495, 2098), (481, 2125), (456, 2124),
                                  (452, 2118), (426, 2041), (516, 2105)], 1):
        holes.add(sx * WIDTH / 3840, sy * HEIGHT / 2160, i, 8)
    # Backdate all but the last so the render shows settled holes rather than
    # eleven simultaneous impact flashes.
    import time as _t
    for h in list(holes.holes)[:-1]:
        h.born = _t.perf_counter() - 2.0
    check("holes layer holds every shot", len(holes.holes) == 11, f"{len(holes.holes)}")

    from lsa.display.layers import score_against_marks
    from PySide6.QtCore import QRect as _QRect
    ctx = lab.scene._context(0.0, 0.0, _QRect(0, 0, WIDTH, HEIGHT))
    from lsa.display.layers import AimField as _AimField
    marks = _AimField().marks(ctx)
    centre_pts, _ = score_against_marks(WIDTH / 2, HEIGHT / 2, marks)
    edge_pts, _ = score_against_marks(WIDTH * 0.62, HEIGHT * 0.5, marks)
    off_pts, _ = score_against_marks(WIDTH * 0.5, HEIGHT * 0.03, marks)
    check("a bullseye scores highest", centre_pts == 10, f"{centre_pts}")
    check("an outer ring scores less", 0 < edge_pts < centre_pts, f"{edge_pts}")
    check("a shot off every ring scores zero", off_pts == 0, f"{off_pts}")

    # Corner ordering. A real calibration once returned a quad with two
    # identical corners, which collapsed the homography so every shot mapped to
    # x=0 and the screen mask rejected most hits. Never again.
    from lsa.calibrate.screen import _order_corners, ScreenQuad, find_screen
    import numpy as _np0
    for label, pts in (
        ("sheared", [[138, 202], [863, 579], [199, 719], [796, 234]]),
        ("normal", [[137, 201], [796, 234], [864, 580], [197, 719]]),
        ("shuffled", [[864, 580], [197, 719], [137, 201], [796, 234]]),
        ("extreme perspective", [[100, 300], [900, 120], [950, 700], [160, 640]]),
    ):
        tl, tr, br, bl = _order_corners(_np0.array(pts, dtype=_np0.float32))
        distinct = len({(round(x), round(y)) for x, y in (tl, tr, br, bl)}) == 4
        wound = tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]
        check(f"corners ordered correctly ({label})", distinct and wound,
              f"distinct={distinct} winding={wound}")

    _q = ScreenQuad(_order_corners(_np0.array([[137, 201], [796, 234], [864, 580],
                                               [197, 719]], _np0.float32)),
                    0.35, 1.0, (), (1280, 720))
    _H = _q.homography_to_screen(3840, 2160)
    _mapped = [_q.map_point(_H, x, y) for x, y in _q.corners]
    _want = [(0, 0), (3840, 0), (3840, 2160), (0, 2160)]
    check("quad corners map onto the screen corners",
          all(abs(a[0] - b[0]) < 1 and abs(a[1] - b[1]) < 1
              for a, b in zip(_mapped, _want)),
          str([(round(x), round(y)) for x, y in _mapped]))

    # A degenerate quad must be refused outright rather than returned.
    _blank = _np0.zeros((720, 1280, 3), _np0.uint8)
    check("a featureless pair of frames yields no screen",
          find_screen(_blank, _blank) is None)

    # Grid calibration, driven exactly as a shooter would drive it.
    from lsa.calibrate.mapping import Mapping, Sample, grid_points
    import numpy as _np, cv2 as _cv2, math as _math

    SW, SH = 3840, 2160
    quad = _np.float32([[137, 201], [796, 234], [864, 580], [197, 719]])
    H_true = _cv2.getPerspectiveTransform(
        _np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]]), quad)

    def screen_to_camera(sx, sy):
        q = _cv2.perspectiveTransform(_np.float32([[[sx, sy]]]), H_true)[0][0]
        cx, cy = float(q[0]), float(q[1])
        ux, uy = (cx - 640) / 640.0, (cy - 360) / 360.0
        r2 = ux * ux + uy * uy
        return 640 + ux * (1 + 0.045 * r2) * 640, 360 + uy * (1 + 0.045 * r2) * 360

    pts = grid_points(SW, SH, 3, 3)
    samples = [Sample((sx, sy), screen_to_camera(sx, sy)) for sx, sy in pts]
    mapping, note = Mapping.fit_validated(samples, (SW, SH))
    check("grid calibration produces a mapping", mapping is not None)
    u_hooks: dict = {}          # filled by the brightness-grid section; read again at the close
    if mapping is not None:
        held = [(SW * f, SH * g) for f in (0.22, 0.5, 0.78) for g in (0.3, 0.7)]
        errs = [_math.hypot(*_np.subtract(mapping.to_screen(*screen_to_camera(sx, sy)),
                                          (sx, sy))) for sx, sy in held]
        rms = float(_np.sqrt(_np.mean(_np.square(errs))))
        mm = rms / (SW / 940.0)
        print(f"         held-out accuracy {rms:.1f}px = {mm:.2f}mm   ({note})")
        check("accurate to within 3mm on points it never saw", mm < 3.0, f"{mm:.2f}mm")
        check("survives a round trip through disk",
              Mapping.from_dict(mapping.to_dict()).to_screen(700, 400)
              == mapping.to_screen(700, 400))

        # A detected shot must become a hole, in the right place. Reported from
        # the firing line as "there are no bullet holes" and separately as
        # "the holes show up nowhere near the actual shots" -- and nothing in
        # this suite covered the path from a Shot to a drawn hole at all.
        from lsa.detect.laser import Shot as _Shot
        lab.mapping = mapping
        # A shot is only a TRIGGER PULL once the floor has been measured against
        # this room (RULES B9); a mapped detection before that no longer reaches
        # the drill (2026-10-20), so the fixture says the room has been measured.
        lab._floor_measured = True
        lab.layers["holes"].clear()
        lab.session.reset()
        before = len(lab.layers["holes"].holes)
        aimed = (SW * 0.62, SH * 0.41)
        cam = screen_to_camera(*aimed)
        lab._on_shot(_Shot(x=cam[0], y=cam[1], nx=cam[0] / 1280.0, ny=cam[1] / 720.0,
                           frames=3, duration_ms=85.0, peak=180, area=120.0,
                           first_timestamp=0.0, last_timestamp=0.085,
                           index=1, track_id=0, elongation=1.2))
        drew = len(lab.layers["holes"].holes) - before
        check("a detected shot draws exactly one hole", drew == 1, f"{drew} holes")
        if drew == 1:
            h = lab.layers["holes"].holes[-1]
            off = _math.hypot(h.x - aimed[0], h.y - aimed[1])
            check("and it lands where the shot was aimed",
                  off / (SW / 940.0) < 3.0,
                  f"{off:.1f}px = {off / (SW / 940.0):.2f}mm from the aim point")
        check("and the session counted it", lab.session.shots == 1,
              f"{lab.session.shots} shots")

        # THE SHOT LINE CARRIES THE EVIDENCE (plan 02 step 10). His two minutes
        # of shooting must answer, for each trigger pull: where it was drawn,
        # how far from the nearest target, when (on the clock the HEARD lines
        # use), what the microphone heard, how the centroid was made. Driven
        # through the real `_on_shot`, not by scanning the source for names
        # (RULES C8) -- a field computed from an attribute that does not exist
        # raises inside a Qt slot and costs the whole line.
        from lsa.detect.laser import Detection as _Det
        _log_lines: list[str] = []
        _real_log = lab._log
        lab._log = lambda m: (_log_lines.append(m), _real_log(m))[1]
        # A microphone that "is running" and heard 0.285 around t=12.4, without
        # opening a device (this suite never may).
        _ls = lab.listener
        _ls._stream = object()
        _ls._recent.append((12.40, 0.285))
        _floor_was = lab._floor_measured
        lab._floor_measured = True          # so this detection counts as a trigger pull
        # PINNED, not read from his file. The margin on the SHOT line is
        # peak / bar, and the bar is the min-peak spinbox, which `CameraLab`
        # loads from `settings.local.json`. Since `isolate_outputs()` began
        # COPYING his real settings into the isolated directory (the cold-
        # start harness has to race against his real timings, RULES C15),
        # this check followed his live value: it expected a margin starting
        # "13." and read 2.2x when it was found, 16.4x against his current
        # min_peak of 11. A test that depends on a file the user
        # owns fails when he retunes, for no fault of the code. The copy
        # stays; this check stops depending on it.
        _minpeak_was = lab.sp_minpeak.value()
        lab.sp_minpeak.setValue(13)
        try:
            lab.layers["holes"].clear()
            lab.session.reset()
            _mem = [_Det(x=cam[0], y=cam[1], area=120.0, peak=p_, frame_index=i_,
                         timestamp=0.02 * i_) for i_, p_ in enumerate((90, 180, 170))]
            lab._on_shot(_Shot(x=cam[0], y=cam[1], nx=cam[0] / 1280.0, ny=cam[1] / 720.0,
                               frames=3, duration_ms=85.0, peak=180, area=120.0,
                               first_timestamp=12.345, last_timestamp=12.43,
                               index=9, track_id=0, elongation=1.2, members=_mem))
        finally:
            lab._log = _real_log
            _ls._stream = None
            _ls._recent.clear()
            _ls._learned.clear()
            lab._floor_measured = _floor_was
            lab.sp_minpeak.setValue(_minpeak_was)
        sline = next((ln for ln in _log_lines if ln.startswith("SHOT") and "cam=(" in ln), "")
        print(f"         {sline.strip()[:400]}")
        check("the SHOT line is still there, with its original fields",
              "SHOT" in sline and "screen=(" in sline and "peak=180" in sline
              and "frames=3" in sline, sline[:120])
        check("it carries the shot's own clock reading (t=12.345)",
              "t=12.345" in sline, sline[-160:])
        check("it says what the microphone heard around the shot (0.285)",
              "mic_peak=0.285 (bar " in sline, sline[-260:])
        check("it says how the centroid was made: frame, dimness, drift, margin",
              "centroid from frame 1/3" in sline and "first-frame peak 90 of best 180 (50%)" in sline
              and "drift first-vs-brightest 0.0 cam px" in sline and "best/bar 180/" in sline,
              sline[-220:])
        check("it carries the running heard-against-seen tally",
              "[audio" in sline, sline[-120:])
        check("and the evidence did not cost a hole",
              len(lab.layers["holes"].holes) == 1, f"{len(lab.layers['holes'].holes)} holes")
        check("and did not fail: no 'evidence failed' note anywhere in the line",
              "evidence failed" not in sline and "could not measure" not in "".join(_log_lines))
        check("a microphone that WAS listening adds no tag to mic_peak",
              "DEAF" not in sline and "NO AUDIO" not in sline, sline[-300:])

        # mic_peak=0.000 USED TO MEAN THREE THINGS: a silent room, a microphone
        # that was not delivering audio, and a microphone the app had muted
        # itself (~0.8 s after its own gunshot). One trigger pull fragmenting into
        # two detections 0.3 s apart read the second as seen-not-heard -- a
        # phantom, in the channel the whole gun test rests on. Drive the real
        # `_on_shot` into each state and read what the line SAYS.
        def _shot_line(recent, mutes, idx):
            out: list[str] = []
            real_log2 = lab._log
            lab._log = lambda m: (out.append(m), real_log2(m))[1]
            _ls._stream = object()
            _ls._recent.clear()
            _ls._recent.extend(recent)
            _ls._mutes.clear()
            _ls._mutes.extend(mutes)
            fl2 = lab._floor_measured
            lab._floor_measured = True
            try:
                lab.layers["holes"].clear()
                lab.session.reset()
                lab._on_shot(_Shot(x=cam[0], y=cam[1], nx=cam[0] / 1280.0, ny=cam[1] / 720.0,
                                   frames=3, duration_ms=85.0, peak=180, area=120.0,
                                   first_timestamp=20.345, last_timestamp=20.43,
                                   index=idx, track_id=0, elongation=1.2))
            finally:
                lab._log = real_log2
                _ls._stream = None
                _ls._recent.clear()
                _ls._mutes.clear()
                _ls._learned.clear()
                lab._floor_measured = fl2
            return next((ln for ln in out if ln.startswith("SHOT") and "cam=(" in ln), "")

        l_quiet = _shot_line([(20.40, 0.004)], [], 31)
        l_deaf = _shot_line([(20.40, 0.004)], [(20.10, 20.91)], 32)
        l_none = _shot_line([], [], 33)
        print(f"         quiet room : ...{l_quiet.strip()[l_quiet.find('mic_peak'):][:150]}")
        print(f"         self-muted : ...{l_deaf.strip()[l_deaf.find('mic_peak'):][:230]}")
        print(f"         no audio   : ...{l_none.strip()[l_none.find('mic_peak'):][:200]}")
        check("a silent room reads as a level with NO excuse attached",
              "mic_peak=0.004" in l_quiet and "DEAF" not in l_quiet and "NO AUDIO" not in l_quiet,
              l_quiet[-260:])
        check("the same level while the app had muted the microphone says DEAF, with the "
              "seconds still to run, and that it is not evidence of a phantom",
              "mic_peak=0.004" in l_deaf and "DEAF" in l_deaf and "another 0.48s" in l_deaf
              and "not evidence of a phantom" in l_deaf, l_deaf[-340:])
        check("a microphone that delivered no audio around the shot says NO AUDIO, and "
              "that it is not a silent room",
              "mic_peak=0.000" in l_none and "NO AUDIO" in l_none
              and "not a silent room" in l_none, l_none[-300:])

        # THE SEAM (RULES C14): the reader in tools/read_string.py must understand
        # the lines the REAL `_on_shot` writes, not lines a test composed to match
        # it. A producer wording change the reader cannot parse fails here.
        import importlib.util as _ilu_rs
        _spec_rs = _ilu_rs.spec_from_file_location(
            "read_string_seam", Path(__file__).resolve().parents[1] / "tools" / "read_string.py")
        _rs = _ilu_rs.module_from_spec(_spec_rs)
        sys.modules["read_string_seam"] = _rs        # a dataclass module must be findable
        _spec_rs.loader.exec_module(_rs)
        _rows = _rs.parse(["03:00:01  " + ln for ln in (sline, l_quiet, l_deaf, l_none)]).pulls
        check("the reader parses all four real SHOT lines", len(_rows) == 4, f"{len(_rows)} rows")
        if len(_rows) == 4:
            r_main, r_quiet, r_deaf, r_none = _rows
            check("... reading the real line's drawn position, mic state, frame and margin",
                  r_main.screen is not None and r_main.frames == 3 and r_main.peak == 180
                  and r_main.frame_used == "1/3" and r_main.margin.startswith("13.")
                  and r_main.heard.startswith("learning (0.285)"),
                  f"{r_main.screen} {r_main.frame_used} {r_main.margin} {r_main.heard}")
            check("... telling a silent room from a deaf microphone from no audio at all",
                  r_quiet.heard.startswith("learning (0.004)") and not r_quiet.deaf
                  and r_deaf.deaf and r_deaf.heard.startswith("DEAF")
                  and r_none.no_audio and r_none.heard == "no-audio",
                  f"{r_quiet.heard!r} / {r_deaf.heard!r} / {r_none.heard!r}")

        # ==================================================================
        # THE BRIGHTNESS GRID (key U, 2026-10-23): nine spots, three agreeing
        # shots each, to measure how bright the laser looks across the screen.
        #
        # Driven through the REAL routine (`start_sensitivity_grid`, the key
        # handler, `_on_shot`, the HUD-tick stall check, `_on_target_closed`)
        # with FAINT shots (peak 22, the right-hand case the whole step exists
        # for) built by `fake_camera.shot_at_screen`, which places them through
        # the fake camera's own optics. Every check below names the fault that
        # turns it red; `tests/mutate_detector.py` reintroduces each of them.
        # What is NOT proven here, and only his gun can: that the peaks a real
        # laser makes at these spots are the ones the map is built from.
        # ==================================================================
        print("\n== the brightness grid (U) ==")
        _u_keep = dict(
            capture=lab.capture, window=lab.target_window, mapping=lab.mapping,
            noise=lab._noise_peak, floor=lab._floor_measured,
            sees=lab._camera_sees_our_screen, save_mapping=lab._save_mapping,
            set_gain=lab._set_gain_map, start_floor=lab._start_noise_floor,
            log=lab._log, thr=lab.sp_threshold.value(), mp=lab.sp_minpeak.value(),
            saved_levels=lab._grid_saved_levels, sens_applied=lab._sens_applied,
            floor_before_gain=lab._floor_before_gain)
        try:
            import dataclasses as _dc_u
            import json as _json_u
            from PySide6.QtCore import QRect as _QRect_u
            from fake_camera import (Optics as _Optics_u, shot_at_screen as _shot_u,
                                     screen_to_camera as _s2c_u)
            from lsa.detect.laser import (LaserDetector as _LD_u,
                                          ScoreMode as _SM_u)
            from lsa.director import Job as _Job_u, Sleep as _Sleep_u
            from lsa.display.layers import CalibrationTargets as _CT_u
            from PySide6.QtGui import QPainter as _QPaintU, QImage as _QImU

            # A GENTLER perspective than the bench's (sides 568/310/564/478): the
            # grid's own ORDER rule extrapolates linearly along a row and allows
            # half a step, and these checks are about the routine, not about how
            # near that rule runs to its edge on the steepest view. Same fake
            # camera, same helper; only the screen's corners differ.
            _optU = _Optics_u(quad=((230.0, 110.0), (880.0, 150.0),
                                    (860.0, 540.0), (260.0, 600.0)))
            _sizeU = (SW, SH)
            map_u, _noteU = Mapping.fit_validated(
                [Sample((sx_, sy_), _s2c_u(_optU, _sizeU, sx_, sy_))
                 for sx_, sy_ in grid_points(SW, SH, 3, 3)], _sizeU)
            if map_u is None:
                raise RuntimeError(f"no mapping fits the fake optics: {_noteU}")
            _PRIOR_U = _np.full((720, 1280), 0.5, _np.float32)   # a map already in force
            _map_before_u = _json_u.dumps(
                {k_: v_ for k_, v_ in map_u.to_dict().items() if k_ != "sensitivity"},
                sort_keys=True)

            class _SgCap:
                """Records what the app tells the capture. NOT a swallower: a method
                the routine has no business calling raises AttributeError, and the
                section then fails by name instead of passing by omission."""

                def __init__(self):
                    self.gain_maps, self.detector_params = [], []
                    self.aim_help_calls = []

                def set_gain_map(self, g):
                    self.gain_maps.append(g)

                def set_detector_params(self, p):
                    self.detector_params.append(p)

                def set_grouper_config(self, c):
                    pass

                def set_aim_help(self, on):
                    # `_clear_alert` tells the capture the aim help is off, and the
                    # grid's own alerts and any keypress that dismisses one reach it.
                    # Its absence raised AttributeError halfway through this section
                    # (2026-10-23) and ~40 of its checks never ran.
                    self.aim_help_calls.append(bool(on))

            class _SgWin:
                cover_reason = ""
                cover_by = ""

                def rect(self):
                    return _QRect_u(0, 0, SW, SH)

                def in_front_of_camera(self):
                    return True

                def close(self):
                    pass

            events_u: list = []
            saved_rows_u: list = []
            lines_u: list = []
            floors_u: list = []
            _real_set_u = lab._set_gain_map

            def _save_u(mapping_, what):
                # What is on the mapping AT THE MOMENT it is written: an ORDER,
                # not an existence (the first instance of this fault passed an
                # existence test throughout).
                events_u.append("save")
                saved_rows_u[:] = [dict(r_) for r_ in mapping_.sensitivity]
                return True

            def _set_u(gain, why):
                events_u.append("apply" if gain is not None else "withdraw")
                return _real_set_u(gain, why)

            lab._save_mapping = _save_u
            lab._set_gain_map = _set_u
            lab._start_noise_floor = lambda *a, **k: floors_u.append(1)
            lab._log = lambda m_, _o=_u_keep["log"]: (lines_u.append(str(m_)), _o(m_))[1]
            _alert_u = lab.layers["alert"]
            _cal_u = lab.layers["calibration"]
            _host_u = lab.layers["modes"]
            _tick_u = [100.0]
            _idx_u = [5000]

            def _peak_u(i_):                     # left 66, centre 44, RIGHT 22
                return (66, 44, 22)[i_ % 3]

            def _fire_u(sx_, sy_, peak_, n=1, scatter=True):
                """n shots at a SCREEN point through the fake optics, with aim scatter
                of a few camera pixels unless `scatter` is False (a steady object)."""
                jit = ((0.0, 0.0), (2.0, -1.0), (-2.0, 1.5), (1.0, 2.0))
                for k_ in range(n):
                    _idx_u[0] += 1
                    _tick_u[0] += 0.3
                    lab._sgrid_ready_at = 0.0       # the 0.35 s debounce is not under test
                    lab._on_shot(_shot_u(_optU, _sizeU, sx_, sy_, peak_,
                                         index=_idx_u[0], t=_tick_u[0],
                                         offset_px=jit[k_ % 4] if scatter else (0.0, 0.0)))

            def _prepare_u(prior=False, window=True):
                cap_ = _SgCap()
                lab.director.cancel("test: making room for the brightness grid")
                lab.capture = cap_
                lab.target_window = _SgWin() if window else None
                lab.mapping = map_u
                lab._noise_peak = 2.0
                lab._floor_measured = True
                lab._camera_sees_our_screen = lambda: True
                for flag_ in ("_calibrating", "_pat_active", "_grid_active",
                              "_sgrid_active", "_floor_in_flight", "_modetest_active"):
                    setattr(lab, flag_, False)
                lab._gain_pending = None
                lab._gain_awaiting_baseline = None
                lab._gain_in_force = None
                lab.sp_threshold.setValue(35)
                lab.sp_minpeak.setValue(40)
                lab._grid_saved_levels = None
                lab.session.reset()
                lab.layers["holes"].clear()
                lab._sens_shots = []
                lab._area_shots = []
                _alert_u.clear()
                events_u.clear()
                lines_u.clear()
                saved_rows_u.clear()
                floors_u.clear()
                if prior:
                    lab._set_gain_map(_PRIOR_U, "test: a map already in force")
                    events_u.clear()
                return cap_

            def _restored_u(cap_):
                return (lab.sp_threshold.value() == 35 and lab.sp_minpeak.value() == 40
                        and lab._grid_saved_levels is None
                        and (not cap_.detector_params
                             or cap_.detector_params[-1].min_peak == 40))

            def _dot_frame_u():
                f_ = _np.full((720, 1280, 3), 30, _np.uint8)
                f_[300:320, 600:620] = (30, 30, 52)            # raw chroma 22
                return f_

            def _sees_u(params_):
                p_ = _dc_u.replace(params_, mode=_SM_u.CHROMA, use_background=False,
                                   roi_polygon=None)
                return len(_LD_u(p_).process(_dot_frame_u(), timestamp=0.0))

            u_hooks["prepare"] = _prepare_u

            # PRESENCE BEFORE INDEXING (2026-10-23). With a fault in place nothing may
            # register: `detector_params`/`gain_maps` stay empty and `_sgrid_points` stays
            # []. A bare `[-1]` / `pts_u[4]` then raised IndexError BEFORE the check written
            # for the fault ran, so mutate_detector entries 90 and 111 came back BROKEN and
            # the properties they guard (the grid relaxes the bar; U starts it) were never
            # shown able to fail. Every read below goes through these two helpers, and each
            # check states the presence it needs, so it fails BY NAME.
            def _last_u(seq_):
                """The last element, or None when nothing was ever delivered. A caller
                that treats None as a legitimate value (`gain_maps` holds None on
                purpose) must ALSO assert `len(seq_)`."""
                return seq_[-1] if seq_ else None

            def _fallback_points_u():
                """The spots the grid WOULD have used: the same helper, the same rect."""
                return list(lab._calibration_grid_points(_SgWin().rect(), 3, 3))

            def _pos_u(seq_, item_):
                return seq_.index(item_) if item_ in seq_ else -1

            # -- the stand-in against the REAL capture (RULES C12) ----------------
            # A stand-in that lacks a name the app calls fails halfway through the
            # section, and the rest of it never runs. Two structural questions,
            # asked of the real class and of the app's real source, not of a list
            # kept beside this one:
            #   * everything _SgCap defines exists on CaptureThread with the same
            #     number of arguments (it must not invent an API the app lacks);
            #   * everything the app calls on `.capture` exists on CaptureThread at
            #     all (the `capture.stats` / `_stats` mistake, caught in the source).
            import ast as _ast_u
            import inspect as _insp_u
            from lsa.camera.capture import CaptureThread as _RealCap_u
            _src_root_u = Path(__file__).resolve().parents[1] / "src" / "lsa"
            _cap_tree_u = _ast_u.parse((_src_root_u / "camera" / "capture.py").read_text(encoding="utf-8"))
            _instance_attrs_u = {n_.attr for n_ in _ast_u.walk(_cap_tree_u)
                                 if isinstance(n_, _ast_u.Attribute) and isinstance(n_.ctx, _ast_u.Store)
                                 and isinstance(n_.value, _ast_u.Name) and n_.value.id == "self"}
            _called_u = {}
            for _mod_u in ("tools/camera_lab.py", "routines.py"):
                for n_ in _ast_u.walk(_ast_u.parse((_src_root_u / _mod_u).read_text(encoding="utf-8"))):
                    if (isinstance(n_, _ast_u.Attribute) and isinstance(n_.value, _ast_u.Attribute)
                            and n_.value.attr == "capture"):
                        _called_u.setdefault(n_.attr, _mod_u)
            _absent_u = sorted(n_ for n_ in _called_u
                               if not hasattr(_RealCap_u, n_) and n_ not in _instance_attrs_u)
            check("brightness grid: every name the app uses on `.capture` exists on the real CaptureThread "
                  f"({len(_called_u)} names, read from the source of camera_lab.py and routines.py)",
                  len(_called_u) > 10 and not _absent_u,
                  f"not on CaptureThread: {_absent_u}")
            _mine_u = [n_ for n_ in vars(_SgCap) if not n_.startswith("_")
                       and callable(vars(_SgCap)[n_])]
            _bad_u = []
            for n_ in _mine_u:
                real_ = getattr(_RealCap_u, n_, None)
                if real_ is None:
                    _bad_u.append(f"{n_}: not on CaptureThread")
                elif (len(_insp_u.signature(real_).parameters)
                        != len(_insp_u.signature(vars(_SgCap)[n_]).parameters)):
                    _bad_u.append(f"{n_}: signature differs")
            check("brightness grid: the capture stand-in defines only names the real CaptureThread has, "
                  "with the same arguments (set_aim_help among them)",
                  "set_aim_help" in _mine_u and not _bad_u, f"{_bad_u}")

            # -- 0. the control: outside the grid a shot IS scored -----------------
            cap0_u = _prepare_u(window=False)
            lab.capture = None                  # exactly the state the shot checks above ran in
            _fire_u(SW * 0.5, SH * 0.5, 80)
            ctl_shots_u = lab.session.shots
            ctl_holes_u = len(lab.layers["holes"].holes)
            check("brightness grid: CONTROL - outside the grid the same shot scores and draws a hole "
                  "(so 'scored nothing' below can fail)",
                  ctl_shots_u == 1 and ctl_holes_u == 1, f"{ctl_shots_u} shots, {ctl_holes_u} holes")
            lab.session.reset()
            lab.layers["holes"].clear()
            lab._sens_shots = []
            lab._area_shots = []

            # -- 1. it refuses, says why, and changes nothing ----------------------
            cap0_u = _prepare_u()
            _refusals = (
                ("there is no calibration", lambda: setattr(lab, "mapping", None),
                 lambda: setattr(lab, "mapping", map_u), "press G"),
                ("a measurement owns the screen", lambda: setattr(lab, "_calibrating", True),
                 lambda: setattr(lab, "_calibrating", False), "a measurement"),
                ("pattern calibration is running", lambda: setattr(lab, "_pat_active", True),
                 lambda: setattr(lab, "_pat_active", False), "pattern calibration"),
                ("the shot grid is running", lambda: setattr(lab, "_grid_active", True),
                 lambda: setattr(lab, "_grid_active", False), "grid calibration"),
                ("a Director routine holds the slot",
                 lambda: setattr(lab.director, "busy_label", lambda: "ZZ-ROUTINE"),
                 lambda: lab.director.__dict__.pop("busy_label", None), "ZZ-ROUTINE"),
                ("the noise floor was never measured",
                 lambda: setattr(lab, "_floor_measured", False),
                 lambda: setattr(lab, "_floor_measured", True), "noise floor"),
            )
            for why_u, setup_u, undo_u, frag_u in _refusals:
                lines_u.clear()
                setup_u()
                try:
                    lab.start_sensitivity_grid()
                finally:
                    undo_u()
                check(f"brightness grid: REFUSES to start while {why_u}, says why, and changes nothing",
                      not lab._sgrid_active and lab.sp_minpeak.value() == 40
                      and not _cal_u.visible
                      and any("not starting the brightness grid" in m_ and frag_u in m_
                              for m_ in lines_u), "; ".join(lines_u)[:200])
            _alert_u.clear()

            # -- 2. a full run, with a map already in force, faint on the right ----
            cap1_u = _prepare_u(prior=True)
            params_before_u = lab._detector_params()
            _box_u = _host_u.tallies[0].shots
            lab._on_target_key(Qt.Key_U)
            pts_u = list(lab._sgrid_points)
            _started_u = len(pts_u) == 9
            if not _started_u:
                # The grid did not start (a fault, or a regression): the named check just
                # below fails; the rest of the section still has nine spots to aim at.
                pts_u = _fallback_points_u()
            rp_u = _last_u(cap1_u.detector_params)
            check("brightness grid: U STARTS it - nine spots on the calibration layer, titled for what it measures",
                  lab._sgrid_active and _cal_u.visible and _started_u
                  and _cal_u.title == "BRIGHTNESS" and _cal_u.unit == "spot",
                  f"active={lab._sgrid_active} points={len(list(lab._sgrid_points))} title={_cal_u.title!r}")
            check("brightness grid: it is in the central exclusion list, so nothing else may start over it",
                  lab._screen_busy() == "the brightness grid", repr(lab._screen_busy()))
            def _gen_u():
                yield _Sleep_u(1000)         # a real generator: it takes the slot and keeps it

            _job_u = _Job_u("zz_other_routine", _gen_u)
            try:
                director_took_u = lab.director.start(_job_u)
            finally:
                lab.director.cancel("test cleanup")
            check("brightness grid: and a Director routine is REFUSED the slot while it runs (central exclusion list)",
                  director_took_u is False, f"start() returned {director_took_u!r}")
            check("brightness grid: faint shots can register - the bar the DETECTOR holds is below a peak-22 dot, "
                  "and the bar before the grid was not (real detector)",
                  rp_u is not None and rp_u.min_peak <= 22 and rp_u.threshold <= 22
                  and _sees_u(rp_u) >= 1 and _sees_u(params_before_u) == 0,
                  ("the capture was NEVER given new detector params (the grid did not relax the bar)"
                   if rp_u is None else
                   f"relaxed {rp_u.threshold}/{rp_u.min_peak} sees {_sees_u(rp_u)}; ")
                  + f"before {params_before_u.threshold}/{params_before_u.min_peak} sees {_sees_u(params_before_u)}")
            check("brightness grid: the screen MASK stays on (the mapping is trusted here; the shot grid drops it)",
                  rp_u is not None and rp_u.roi_polygon is not None,
                  "no params delivered" if rp_u is None else repr(rp_u.roi_polygon))
            check("brightness grid: the map already in force is SET ASIDE (a peak seen through a map is in the map's "
                  "units)", len(cap1_u.gain_maps) >= 1 and cap1_u.gain_maps[-1] is None
                  and lab._gain_in_force is None and lab._sgrid_prior_gain is _PRIOR_U,
                  f"{len(cap1_u.gain_maps)} gain maps delivered")
            _before2_u = (id(lab._sgrid_points), lab._sgrid_prior_gain, lab._grid_saved_levels)
            lab.start_sensitivity_grid()
            check("brightness grid: a second start changes nothing (it does not reset the run or lose the set-aside map)",
                  (id(lab._sgrid_points), lab._sgrid_prior_gain, lab._grid_saved_levels) == _before2_u
                  and any("already running; ignoring the second request" in m_ for m_ in lines_u),
                  "; ".join(lines_u[-2:])[:200])
            events_u.clear()
            lines_u.clear()

            # a STEADY OBJECT present from the first spot agrees with itself for ever
            _fire_u(*pts_u[4], 15, n=3, scatter=False)
            check("brightness grid: a steady object at the wrong place is REJECTED at the very first spot, "
                  "not recorded (three agreeing shots are not proof of a laser)",
                  lab._sgrid_samples == [] and _cal_u.index == 0
                  and any("REJECTED" in m_ for m_ in lines_u), "; ".join(lines_u[-2:])[:200])
            _fire_u(*pts_u[0], _peak_u(0), n=2)
            check("brightness grid: two agreeing shots do not record a spot - it takes three",
                  lab._sgrid_samples == [] and _cal_u.index == 0)
            _fire_u(*pts_u[0], _peak_u(0), n=1)
            check("brightness grid: the third agreeing shot records it, and the grid moves to the next spot",
                  len(lab._sgrid_samples) == 1 and _cal_u.index == 1
                  and lab._sgrid_samples[0].peak == 66, f"{len(lab._sgrid_samples)} / index {_cal_u.index}")
            _fire_u(*pts_u[1], _peak_u(1), n=3)
            # the GRID'S OWN ORDER, with the aim check switched off so only it can speak
            lab.SGRID_AIM_FRAC = 1000.0
            try:
                _fire_u(*pts_u[0], 30, n=3, scatter=False)
            finally:
                lab.__dict__.pop("SGRID_AIM_FRAC", None)
            check("brightness grid: with the aim check off, the grid's own ORDER still rejects an object sitting where "
                  "spot 1 is when spot 3 is asked for",
                  len(lab._sgrid_samples) == 2 and _cal_u.index == 2,
                  f"{len(lab._sgrid_samples)} recorded, index {_cal_u.index}")
            for i_u in range(2, 9):
                _fire_u(*pts_u[i_u], _peak_u(i_u), n=3)
                if i_u == 7:
                    check("brightness grid: mid-run, shots score NOTHING, spend no round and draw no hole "
                          "(they never reached the mode)",
                          lab.session.shots == 0 and len(lab.layers["holes"].holes) == 0
                          and _host_u.tallies[0].shots == _box_u and lab._sens_shots == [],
                          f"session {lab.session.shots}, holes {len(lab.layers['holes'].holes)}, "
                          f"rounds {_host_u.tallies[0].shots - _box_u}, learner {len(lab._sens_shots)}")
            check("brightness grid: nine spots done - the run is OVER and the thresholds are back, in the detector too",
                  not lab._sgrid_active and _restored_u(cap1_u) and not _cal_u.visible
                  and lab._screen_busy() in ("", "a measurement"),
                  f"{lab.sp_threshold.value()}/{lab.sp_minpeak.value()} busy={lab._screen_busy()!r}")
            check("brightness grid: it never touched the mapping (the fit is byte-for-byte what it was)",
                  lab.mapping is map_u and _json_u.dumps(
                      {k_: v_ for k_, v_ in map_u.to_dict().items() if k_ != "sensitivity"},
                      sort_keys=True) == _map_before_u)
            peaks_saved_u = sorted(r_["peak"] for r_ in saved_rows_u)
            check("brightness grid: the nine medians reached the file (saved rows = what the shots measured)",
                  peaks_saved_u == sorted([66.0, 44.0, 22.0] * 3), str(peaks_saved_u))
            check("brightness grid: ORDER - the rows were on the mapping when it was written, and the write came "
                  "BEFORE the map was applied",
                  events_u[:2] == ["save", "apply"], str(events_u))
            gmap_u = _last_u(cap1_u.gain_maps)
            cx_r, cy_r = _s2c_u(_optU, _sizeU, *pts_u[2])      # a right-hand (faint) spot
            cx_l, cy_l = _s2c_u(_optU, _sizeU, *pts_u[0])      # a left-hand (bright) spot
            check("brightness grid: the map was built from the SHOTS, not from the mapping's markers - it amplifies "
                  "the faint right side and leaves the bright left alone",
                  gmap_u is not None and float(gmap_u[int(cy_r), int(cx_r)]) < 0.6
                  and float(gmap_u[int(cy_l), int(cx_l)]) > 0.8,
                  "no map" if gmap_u is None else
                  f"right {float(gmap_u[int(cy_r), int(cx_r)]):.2f}, left {float(gmap_u[int(cy_l), int(cx_l)]):.2f}")
            check("brightness grid: the map is handed to the judgement (floor remembered, map pending, floor asked for)",
                  lab._gain_pending is not None and lab._calibrating,
                  f"pending {lab._gain_pending is not None}, calibrating {lab._calibrating}")
            lab._gain_pending = None
            lab._calibrating = False
            lab._gain_in_force = None
            lab.capture = None                  # the state the control above ran in
            lab.target_window = None
            _fire_u(SW * 0.5, SH * 0.5, 80)
            check("brightness grid: and once it is over a shot scores again",
                  lab.session.shots == 1, f"{lab.session.shots} shots")

            # -- 3. three spots are not a map -------------------------------------
            cap2_u = _prepare_u(prior=True)
            lab._on_target_key(Qt.Key_U)
            _pts3_u = list(lab._sgrid_points) or _fallback_points_u()
            for i_u in range(3):
                _fire_u(*_pts3_u[i_u], _peak_u(i_u), n=3)
            for _ in range(6):
                lab._on_target_key(Qt.Key_Space)
            alert_up_u = _alert_u.visible and "BRIGHTNESS NOT MEASURED" in _alert_u.title
            check("brightness grid: three spots are not a map - it needs FOUR spread over the screen, says so on the "
                  "target screen, and builds nothing",
                  not lab._sgrid_active and alert_up_u
                  and any("only 3 of 9 spots registered" in m_ for m_ in lines_u)
                  and "save" not in events_u and saved_rows_u == [] and lab._gain_pending is None,
                  f"alert={_alert_u.title!r} events={events_u}")
            check("brightness grid: ...and detection stays AS IT WAS - the thresholds and the map set aside are back",
                  _restored_u(cap2_u) and len(cap2_u.gain_maps) >= 1
                  and cap2_u.gain_maps[-1] is _PRIOR_U and lab._gain_in_force is _PRIOR_U,
                  f"{lab.sp_threshold.value()}/{lab.sp_minpeak.value()}, {len(cap2_u.gain_maps)} gain maps")
            check("brightness grid: and the wording never tells a man in a dark room to dim it (RULES C4)",
                  not any(w_ in (_alert_u.detail or "").lower() for w_ in ("dim ", "darken", "blind")),
                  (_alert_u.detail or "")[:120])
            _alert_u.clear()

            # -- 4. every other way out gives the thresholds back -------------------
            def _exit_case(label_, trigger_):
                cap_ = _prepare_u(prior=True)
                lab.start_sensitivity_grid()
                was_relaxed = lab.sp_minpeak.value() < 40 and lab._sgrid_active
                trigger_()
                check(f"brightness grid: {label_}: it was running and relaxed, and now the thresholds are back, "
                      f"the flag is down and the screen is free",
                      was_relaxed and not lab._sgrid_active and lab.sp_minpeak.value() == 40
                      and lab.sp_threshold.value() == 35 and lab._grid_saved_levels is None
                      and lab._screen_busy() == "" and not _cal_u.visible
                      and not lab.scene.calibrating,
                      f"relaxed={was_relaxed} active={lab._sgrid_active} "
                      f"{lab.sp_threshold.value()}/{lab.sp_minpeak.value()} busy={lab._screen_busy()!r}")
                return cap_

            _exit_case("pressing U again stops it", lambda: lab._on_target_key(Qt.Key_U))

            def _idle_u():
                lab._sgrid_last_seen = time.monotonic() - (lab.SGRID_IDLE_LIMIT_S + 5.0)
                lab._check_grid_stall()
            _exit_case("the idle limit gives the screen back", _idle_u)

            def _cam_gone_u():
                lab.capture = None
                lab._check_grid_stall()
            _exit_case("the camera going away ends it", _cam_gone_u)
            _exit_case("closing the target screen ends it", lab._on_target_closed)
            capm_u = _prepare_u(prior=True)
            lab.start_sensitivity_grid()
            lab._on_target_key(Qt.Key_U)
            check("brightness grid: stopped without a new map, the map it set aside is back in force",
                  len(capm_u.gain_maps) >= 1 and capm_u.gain_maps[-1] is _PRIOR_U
                  and lab._gain_in_force is _PRIOR_U)

            # -- 4b. the rule it shares with the shot grid, on the BENCH's own view --------
            # The ORDER rule extrapolates linearly along a row and allows half a step. The
            # runs above use a gentler quad on purpose; this asks the same rule about the
            # bench's steep one (sides 568/310/564/478), spot by spot in the order they are
            # shot, with exact positions. A legitimate spot refused here would be refused on
            # his bench, and nothing but this would say so before he stood at the line.
            _bench_u = _Optics_u()
            _bench_pts = list(lab._calibration_grid_points(_SgWin().rect(), 3, 3))
            _bench_done: list = []
            _bench_refused: list = []
            for i_b, (sx_b, sy_b) in enumerate(_bench_pts):
                cam_b = _s2c_u(_bench_u, _sizeU, sx_b, sy_b)
                why_b = lab._out_of_grid_order(i_b, cam_b[0], cam_b[1], points=_bench_pts,
                                               samples=_bench_done, cols=3, rows=3)
                if why_b:
                    _bench_refused.append((i_b + 1, why_b[:90]))
                _bench_done.append(Sample(screen=(sx_b, sy_b), camera=cam_b, peak=50.0))
            check("brightness grid: the grid's ORDER rule accepts all nine legitimate spots on the bench's own "
                  "steep view, shot in order", not _bench_refused, str(_bench_refused))

            # -- 4c. the stand-in itself: it can deliver a faint shot at a chosen screen point --
            # `fire_at_screen` is the fake camera's way of doing what `shot_at_screen` does,
            # through the two signals the real tracker emits. Nothing else here calls it, so
            # it is exercised once, or it is a method nobody has seen work (RULES C12).
            from fake_camera import FakeCapture as _FC_u
            from lsa.camera.capture import (CaptureConfig as _CC_u,
                                            GrouperConfig as _GC_u)
            from lsa.detect.laser import DetectorParams as _DP_u
            _fc_u = _FC_u(_CC_u(), _DP_u(), _GC_u(), optics=_optU)
            _got_p, _got_f = [], []
            _fc_u.shot_provisional.connect(_got_p.append)
            _fc_u.shot_detected.connect(_got_f.append)
            _sent_u = _fc_u.fire_at_screen(_sizeU, *pts_u[2], 22, index=77)
            _px_u, _py_u = _s2c_u(_optU, _sizeU, *pts_u[2])
            check("brightness grid: the fake camera delivers a faint (peak 22) shot at a chosen screen point, "
                  "provisional then final, at the position its own optics give",
                  len(_got_p) == 1 and len(_got_f) == 1 and _got_p[0] is _sent_u
                  and _sent_u.peak == 22 and _sent_u.index == 77
                  and abs(_sent_u.x - _px_u) < 1e-6 and abs(_sent_u.y - _py_u) < 1e-6,
                  f"{len(_got_p)} provisional, {len(_got_f)} final, at ({_sent_u.x:.1f},{_sent_u.y:.1f})")

            # -- 5. the printed instructions and the layer ---------------------------
            check("brightness grid: the key legend says U beside G",
                  "U calibrate brightness by shooting" in " | ".join(lab.layers["help"].lines))
            # The SAME action is named on the Keyboard Shortcuts screen (K). Two
            # wordings for one key is how a screen and a legend drift apart, so the
            # words must be the same ones (the screen is Title Case, the legend not).
            from lsa.modes.shortcuts import CHECKS as _SCK_u
            _scu = [e_ for e_ in _SCK_u if e_.glyph == "U"]
            check("brightness grid: the Keyboard Shortcuts screen names U in the legend's own words "
                  "(beside G)",
                  len(_scu) == 1 and _scu[0].title.lower() == "calibrate brightness by shooting"
                  and _pos_u([e_.glyph for e_ in _SCK_u], "U") >= 0
                  and _pos_u([e_.glyph for e_ in _SCK_u], "U")
                  == _pos_u([e_.glyph for e_ in _SCK_u], "G") + 1,
                  str([(e_.glyph, e_.title) for e_ in _SCK_u]))
            _l2_u = _CT_u()
            _l2_u.start(pts_u, need=3, title="BRIGHTNESS", unit="spot", footer=("x",))
            _l2_u.start(pts_u, need=3)
            check("brightness grid: the shot grid's screen is not left wearing its title (start() resets it)",
                  _l2_u.title == "CALIBRATION" and _l2_u.unit == "point"
                  and _l2_u.footer == _CT_u.DEFAULT_FOOTER)

            class _SpyU(_QPaintU):
                seen: list = []

                def drawText(self, *a):
                    if a and isinstance(a[-1], str):
                        self.seen.append(a[-1])
                    return super().drawText(*a)

            _l3_u = _CT_u()
            _l3_u.start(pts_u, need=3, title="BRIGHTNESS", unit="spot", footer=("U again stops this.",))
            _img_u = _QImU(1920, 1080, _QImU.Format_RGB32)
            _SpyU.seen = []
            _sp_u = _SpyU(_img_u)
            _l3_u.paint(_sp_u, lab.scene._context(time.perf_counter(), 0.0, _QRect_u(0, 0, 1920, 1080)))
            _sp_u.end()
            check("brightness grid: its own screen reads 'BRIGHTNESS   spot 1 of 9' and its own footer",
                  any("BRIGHTNESS   spot 1 of 9" in t_ for t_ in _SpyU.seen)
                  and any("U again stops this." in t_ for t_ in _SpyU.seen), str(_SpyU.seen[:3]))
        except Exception as exc_u:                                  # noqa: BLE001 - the finding
            check("the brightness-grid section ran to its end without raising", False,
                  f"{type(exc_u).__name__}: {exc_u}")
        finally:
            lab._log = _u_keep["log"]
            lab._save_mapping = _u_keep["save_mapping"]
            lab._set_gain_map = _u_keep["set_gain"]
            lab._start_noise_floor = _u_keep["start_floor"]
            lab._camera_sees_our_screen = _u_keep["sees"]
            lab.__dict__.pop("SGRID_AIM_FRAC", None)
            lab.director.__dict__.pop("busy_label", None)
            for flag_ in ("_sgrid_active", "_calibrating", "_grid_active"):
                setattr(lab, flag_, False)
            lab._gain_pending = None
            lab._gain_awaiting_baseline = None
            lab._gain_in_force = None
            lab._grid_saved_levels = _u_keep["saved_levels"]
            lab.sp_threshold.setValue(_u_keep["thr"])
            lab.sp_minpeak.setValue(_u_keep["mp"])
            lab._noise_peak = _u_keep["noise"]
            lab._floor_measured = _u_keep["floor"]
            lab.capture = _u_keep["capture"]
            lab.target_window = _u_keep["window"]
            lab.mapping = _u_keep["mapping"]
            lab._sens_applied = _u_keep["sens_applied"]
            lab._floor_before_gain = _u_keep["floor_before_gain"]
            lab.layers["alert"].clear()
            lab.session.reset()
            lab.layers["holes"].clear()
            lab._sens_shots = []
            lab._area_shots = []

        # NOTHING THE CAMERA SEES COUNTS WHILE SOMETHING COVERS THE SCREEN.
        #
        # 2026-10-16 13:29, from the firing line: "As soon as I press Escape
        # I hear continuous shots that just do not stop. This is completely
        # unacceptable." Escape lowers the target window -- correctly, since
        # closing it cost a whole session -- and the camera then watches a
        # lit desktop while the bar is still the one measured against a
        # near-black screen. Measured: 92 detections in 20 seconds at peak
        # 9-29, against his real shots at 85-146. Each one rang the gunshot,
        # drew a hole, moved the score and taught the sensitivity map.
        #
        # Asserted as the PROPERTY, not as "a guard exists": section 27 of
        # test_feedback once asserted every routine consulted *some* flag,
        # passed, and the app still collided (RULES E4). The predicate
        # itself -- which window is actually on top -- is exercised against
        # a real TargetWindow in test_feedback; what is pinned here is that
        # every consumer downstream of it does nothing when it says no.
        from PySide6.QtCore import QRect as _QRect

        class _Covered:
            # MIRRORS the real window, rather than paraphrasing it. The first
            # version of this stub answered `in_front_of_camera` and nothing
            # else, and the mirror case below crashed on `rect()` -- a
            # stand-in that emits fewer signals than the real thing excuses
            # every consumer of the missing ones (RULES C12).
            cover_reason = "the test is holding a window in front of it"
            cover_by = ""        # mirrors TargetWindow (2026-10-20): who is in front, when known

            def rect(self):
                return _QRect(0, 0, SW, SH)

            def in_front_of_camera(self):
                return False

        lab.layers["holes"].clear()
        lab.session.reset()
        lab._cover_ok = True            # so the transition is logged
        lab._cover_checked_at = 0.0     # and the cached answer is not reused
        lab._cover_suppressed = 0
        lab._floor_measured = True
        was_stamped = lab._last_shot_at
        real_window, lab.target_window = lab.target_window, _Covered()
        lab._on_shot(_Shot(x=cam[0], y=cam[1], nx=cam[0] / 1280.0,
                           ny=cam[1] / 720.0, frames=3, duration_ms=85.0,
                           peak=180, area=120.0, first_timestamp=0.0,
                           last_timestamp=0.085, index=3, track_id=0,
                           elongation=1.2))
        check("a detection draws NO hole while something covers the screen",
              len(lab.layers["holes"].holes) == 0,
              f"{len(lab.layers['holes'].holes)} holes -- 92 of these arrived "
              f"in 20s when he pressed Escape")
        check("  and the session does not count it", lab.session.shots == 0,
              f"{lab.session.shots} shots")
        check("  and it is not stamped as 'a shot just landed'",
              lab._last_shot_at == was_stamped,
              "that stamp discards the next room measurement as spoiled")
        check("  and the app is not fooled into thinking it is ready",
              not lab._ready_for_real_shots(),
              "which is what gates the gunshot sound")
        check("  and the suppression is COUNTED, never silent",
              lab._cover_suppressed == 1,
              f"{lab._cover_suppressed} -- a silent suppression is "
              f"indistinguishable from a detector that has stopped working")

        # THE MIRROR, so that suppressing everything cannot pass instead.
        # Four of this project's worst checks passed by never firing.
        class _InFront(_Covered):
            def in_front_of_camera(self):
                return True

        import lsa.tools.camera_lab as _CLmod

        class _Clock:
            """One clock (RULES B4): the one `_camera_sees_our_screen` reads."""
            def __init__(self):
                self.t = 5000.0

            def perf_counter(self):
                return self.t

            def __getattr__(self, name):
                return getattr(time, name)

        try:
            real_time = _CLmod.time
            clock = _Clock()
            _CLmod.time = clock
            lab.target_window = _InFront()
            # THE DRILL IS CHOSEN HERE, not inherited from startup. These
            # checks are about the SHOT LINE and the hole it leaves, and the
            # app now boots into the MENU (his instruction, 2026-10-18), whose
            # tiles are not targets and deliberately draw no hole. A check
            # pinned to whatever the app happens to start in is how seven of
            # them broke the day he added four figures.
            import lsa.modes as _m_sel
            lab.layers["modes"].set_mode(
                _m_sel.by_name("static")(),
                lab.scene._context(0.0, 0.0, QRect(0, 0, WIDTH, HEIGHT)))
            lab._cover_checked_at = 0.0
            lab._cover_true_since = None
            lab._camera_sees_our_screen()      # begins the run of True answers
            clock.t += 1.2                     # ...and it has now lasted a second
            _near_lines: list[str] = []
            _log_before = lab._log
            lab._log = lambda m: (_near_lines.append(m), _log_before(m))[1]
            try:
                lab._on_shot(_Shot(x=cam[0], y=cam[1], nx=cam[0] / 1280.0,
                                   ny=cam[1] / 720.0, frames=3, duration_ms=85.0,
                                   peak=180, area=120.0, first_timestamp=0.0,
                                   last_timestamp=0.085, index=4, track_id=0,
                                   elongation=1.2))
            finally:
                lab._log = _log_before
            _hit_line = next((ln for ln in _near_lines
                              if ln.startswith("SHOT") and "cam=(" in ln), "")
            # Plan 02 step 10: with a mode playing, the SHOT line says how far
            # the shot was from the target it hit or missed, in px and mm,
            # measured before the mode acted on it.
            check("a SHOT line says how far from the nearest target (px and mm)",
                  "(d=" in _hit_line and "px = " in _hit_line and "mm from centre of"
                  in _hit_line and ", r=" in _hit_line, _hit_line[-300:])
            check("and names the target the mode reported",
                  "HIT centre" in _hit_line and "from centre of centre" in _hit_line,
                  _hit_line[-300:])
            check("and no 'could not measure' note was logged",
                  not any("could not measure the distance" in ln for ln in _near_lines))
            check("and the same shot DOES land once the screen is in front again",
                  len(lab.layers["holes"].holes) == 1 and lab.session.shots == 1,
                  f"{len(lab.layers['holes'].holes)} holes, "
                  f"{lab.session.shots} shots")

            # A DETECTION THE APP DOES NOT TREAT AS A TRIGGER PULL NEVER REACHES THE
            # DRILL (2026-10-20, defect 2). Live 20:05:40-43: a mapping existed and
            # the noise floor was being re-measured, so `_app_is_drawing` was True
            # ("not treated as a trigger pull yet") -- and six detections at peak
            # 8-12 still went into the Two Players magazine. Through the real
            # `_on_shot` and the real host: the seam is the app deciding, and a
            # check on the mode alone cannot see the app skip the question.
            _hostp = lab.layers["modes"]
            _hostp.set_mode(_m_sel.by_name("static")(),
                            lab.scene._context(clock.t, 0.0, lab.target_window.rect()))
            lab.layers["holes"].clear()
            lab.session.reset()
            _rem0 = _hostp.remaining(0)
            _dry0 = _hostp.dry_camera
            _pidx = [900]

            def _pull_shot(peak=9):
                _pidx[0] += 1
                lab._on_shot(_Shot(x=cam[0], y=cam[1], nx=cam[0] / 1280.0,
                                   ny=cam[1] / 720.0, frames=2, duration_ms=40.0,
                                   peak=peak, area=60.0, first_timestamp=clock.t,
                                   last_timestamp=clock.t + 0.04, index=_pidx[0],
                                   track_id=0, elongation=1.2))

            lab._calibrating = True            # the noise-floor re-measurement
            try:
                check("precondition: the app says this is NOT a trigger pull",
                      lab._app_is_drawing_now() or not lab._ready_for_real_shots(),
                      "the gate this fix reads")
                for _i in range(6):
                    _pull_shot()
                check("six detections while the app is measuring spend NO round",
                      _hostp.remaining(0) == _rem0,
                      f"{_rem0} -> {_hostp.remaining(0)} rounds left")
                check("  draw no hole, and the session counts nothing",
                      len(lab.layers["holes"].holes) == 0 and lab.session.shots == 0,
                      f"{len(lab.layers['holes'].holes)} holes, {lab.session.shots} shots")
                # EMPTY the magazine for real, then show that a non-pull is not a
                # DRY pull either.
                lab._calibrating = False
                while _hostp.remaining(0) > 0 and not _hostp.round_over:
                    _pull_shot(peak=120)
                _dry_empty = _hostp.dry_camera
                lab.layers["holes"].clear()
                lab._calibrating = True
                _was_over = _hostp.round_over
                for _i in range(3):
                    _pull_shot()
                check("a non-pull on an EMPTY magazine is not a dry pull and draws no hole",
                      _hostp.dry_camera == _dry_empty
                      and len(lab.layers["holes"].holes) == 0,
                      f"dry {_dry_empty} -> {_hostp.dry_camera}, "
                      f"{len(lab.layers['holes'].holes)} holes")
                check("  and it does not end (or change) the round",
                      _hostp.round_over == _was_over)
            finally:
                lab._calibrating = False
            # THE MIRROR: once the app does treat it as a pull, the same shot on
            # the same empty magazine IS a dry pull and DOES draw its hole (his
            # instruction: the pop AND the hole). Suppressing everything cannot pass.
            _dry_b = _hostp.dry_camera
            _holes_b = len(lab.layers["holes"].holes)
            _pull_shot(peak=120)
            check("MIRROR: a real pull on the empty magazine is still a dry pull, with its hole",
                  _hostp.dry_camera == _dry_b + 1
                  and len(lab.layers["holes"].holes) == _holes_b + 1,
                  f"dry {_dry_b} -> {_hostp.dry_camera}, holes {_holes_b} -> "
                  f"{len(lab.layers['holes'].holes)}")
            _hostp.set_mode(_m_sel.by_name("static")(),
                            lab.scene._context(clock.t, 0.0, lab.target_window.rect()))
            lab.layers["holes"].clear()
            lab.session.reset()

            # A SHOT A FIGURE ABSORBS LEAVES NO SCREEN HOLE; A MISS STILL DOES.
            # Through the real `_on_shot` and the real holes layer, because the seam
            # is the app reading `ShotOutcome.absorbed`, and a unit check of the mode
            # alone cannot see the app ignore it. The figure carries the hit (it rides
            # the figure; see test_figures section 24), so a second mark on the screen
            # would drift away from it.
            from lsa.modes.image_target import ImageTarget as _IT
            _host = lab.layers["modes"]
            _saved_mode = _host.mode
            try:
                _fm = _IT()
                _fm._rng.seed(3)
                _fctx = lab.scene._context(clock.t, 0.0, lab.target_window.rect())
                _fm.start(_fctx, figure="thug01")
                _host.mode = _fm
                _inn = _fm._fig.innermost(_fm._centre, _fm._height)
                _cam_hit = screen_to_camera(_inn[1], _inn[2])
                lab.layers["holes"].clear()
                lab.session.reset()
                lab._on_shot(_Shot(x=_cam_hit[0], y=_cam_hit[1], nx=_cam_hit[0] / 1280.0,
                                   ny=_cam_hit[1] / 720.0, frames=3, duration_ms=85.0,
                                   peak=180, area=120.0, first_timestamp=clock.t,
                                   last_timestamp=clock.t + 0.085,
                                   index=21, track_id=0, elongation=1.2))
                check("a shot the figure ABSORBS counts as a hit and leaves NO hole on the "
                      "screen (the figure carries it)",
                      lab.session.shots == 1 and len(lab.layers["holes"].holes) == 0
                      and len(_fm._marks) == 1 and _host.last_outcome.absorbed
                      and _host.last_outcome.hit,
                      f"{len(lab.layers['holes'].holes)} screen holes, "
                      f"{len(_fm._marks)} carried, outcome {_host.last_outcome}")
                _off = next((q for q in ((60.0, 60.0), (SW - 60.0, 60.0), (60.0, SH - 300.0),
                                         (SW - 60.0, SH - 300.0))
                             if not _fm._fig.on_figure(q[0], q[1], _fm._centre, _fm._height)),
                            None)
                _cam_off = screen_to_camera(*_off)
                lab._on_shot(_Shot(x=_cam_off[0], y=_cam_off[1], nx=_cam_off[0] / 1280.0,
                                   ny=_cam_off[1] / 720.0, frames=3, duration_ms=85.0,
                                   peak=180, area=120.0, first_timestamp=clock.t,
                                   last_timestamp=clock.t + 0.085,
                                   index=22, track_id=0, elongation=1.2))
                check("a shot that MISSES the figure still leaves its ordinary hole on the screen, "
                      "where it landed, and the figure carries nothing for it",
                      len(lab.layers["holes"].holes) == 1 and len(_fm._marks) == 1
                      and not _host.last_outcome.absorbed
                      and _math.hypot(lab.layers["holes"].holes[-1].x - _off[0],
                                      lab.layers["holes"].holes[-1].y - _off[1])
                      / (SW / 940.0) < 3.0,
                      f"{len(lab.layers['holes'].holes)} screen holes, {len(_fm._marks)} carried")
            finally:
                _host.mode = _saved_mode
                lab.layers["holes"].clear()
                lab.session.reset()

            # S5: THE COVER CHECK IS DEBOUNCED ON THE WAY BACK TO TRUE.
            #
            # 2026-10-16 18:45:33-54: at the desktop the check flipped four times
            # in 25 seconds and each True window admitted desktop detections as
            # shots -- thirteen of them, with the microphone silent. False must
            # be immediate; True must have lasted COVER_SETTLE_S.
            def _flip_shot(i):
                lab._on_shot(_Shot(x=cam[0], y=cam[1], nx=cam[0] / 1280.0,
                                   ny=cam[1] / 720.0, frames=3, duration_ms=85.0,
                                   peak=180, area=120.0, first_timestamp=0.0,
                                   last_timestamp=0.085, index=i, track_id=0,
                                   elongation=1.2))

            class _Flip(_Covered):
                front = False

                def in_front_of_camera(self):
                    return self.front

            flip = _Flip()
            lab.target_window = flip
            lab.layers["holes"].clear()
            lab.session.reset()
            lab._cover_ok = False
            lab._cover_true_since = None
            lab._cover_suppressed = 0
            lab._cover_checked_at = 0.0
            clock.t += 10.0
            # Flips true for 0.3s, then false again: admits nothing.
            flip.front = True
            lab._camera_sees_our_screen()
            clock.t += 0.3
            _flip_shot(20)
            flip.front = False
            clock.t += 0.3
            _flip_shot(21)
            check("a window that is in front for 0.3s admits nothing",
                  len(lab.layers["holes"].holes) == 0 and lab.session.shots == 0,
                  f"{len(lab.layers['holes'].holes)} holes, "
                  f"{lab.session.shots} shots")
            check("  and false is immediate, never delayed",
                  lab._cover_ok is False and lab._cover_true_since is None)
            # In front continuously for 1.2s: admits.
            flip.front = True
            clock.t += 0.3
            lab._camera_sees_our_screen()
            clock.t += 0.7
            lab._camera_sees_our_screen()
            clock.t += 0.5
            _flip_shot(22)
            check("a window that stays in front for 1.2s admits the shot",
                  len(lab.layers["holes"].holes) == 1 and lab.session.shots == 1,
                  f"{len(lab.layers['holes'].holes)} holes, "
                  f"{lab.session.shots} shots")

            # S4: A CLOSED target window means the camera is looking at the
            # desktop. 18:45:57, three detections after the window closed rang
            # the gunshot and fed a border that collapsed to 22%.
            lab.layers["holes"].clear()
            lab.session.reset()
            lab._cover_suppressed = 0
            sounds = []
            had_play = "play" in vars(lab.mixer)
            real_play = lab.mixer.play
            lab.mixer.play = lambda *a, **k: sounds.append(a) or False
            real_ever_opened = lab._target_ever_opened
            lab.target_window = None            # closed (flag says one WAS open)
            lab._target_ever_opened = True
            _flip_shot(30)
            check("a shot after the target window CLOSED is suppressed and counted",
                  lab._cover_suppressed == 1, f"{lab._cover_suppressed}")
            check("  draws no hole, scores nothing, asks for no sound",
                  len(lab.layers["holes"].holes) == 0 and lab.session.shots == 0
                  and not sounds,
                  f"{len(lab.layers['holes'].holes)} holes, "
                  f"{lab.session.shots} shots, {len(sounds)} sound request(s)")
            # F11 CLOSES THE WINDOW AND THE APP MUST SAY SO. Before: every shot
            # after it was discarded in silence (RULES B9, closing clause).
            logged = []
            real_log = lab._log
            lab._log = lambda m, *a, **k: logged.append(m)
            try:
                lab._target_ever_opened = True
                lab.target_window = None
                lab._cover_closed_noted = False
                lab._cover_suppressed = 0
                lab._cover_causes.clear()
                lab.layers["holes"].clear()
                lab.session.reset()
                for i in (40, 41, 42):
                    _flip_shot(i)
                said = [m for m in logged if "closed" in m and "IGNORING" in m]
                check("closing the target window LOGS that detections are ignored",
                      len(said) == 1,
                      f"{len(said)} line(s) -- a silent suppression is "
                      f"indistinguishable from a detector that has stopped")
                check("  once, not once per detection (3 shots, 1 line)",
                      len(logged) == 1 and lab._cover_suppressed == 3
                      and lab.session.shots == 0
                      and len(lab.layers["holes"].holes) == 0,
                      f"{len(logged)} lines, {lab._cover_suppressed} counted, "
                      f"{lab.session.shots} shots")
                # Reopened: the resume line names what was TRUE, and the count.
                lab.target_window = flip
                flip.front = True
                lab._cover_checked_at = 0.0
                lab._camera_sees_our_screen()      # in front, but only just
                clock.t += 1.2
                lab._cover_checked_at = 0.0
                lab._camera_sees_our_screen()
                resumed = [m for m in logged if "in front of the camera again" in m]
                check("reopening reports the count and the real cause",
                      len(resumed) == 1 and "3 detection(s)" in resumed[0]
                      and "closed" in resumed[0]
                      and "the desktop" not in resumed[0],
                      f"{resumed}")
                # A suppression that was only the settle must not say 'desktop'.
                logged.clear()
                lab._cover_ok = False
                lab._cover_true_since = None
                lab._cover_causes.clear()
                lab._cover_suppressed = 0
                flip.front = True
                clock.t += 10.0
                lab._cover_checked_at = 0.0
                lab._camera_sees_our_screen()      # settling begins
                _flip_shot(43)                     # suppressed: still settling
                clock.t += 1.2
                lab._cover_checked_at = 0.0
                lab._camera_sees_our_screen()
                resumed = [m for m in logged if "in front of the camera again" in m]
                check("a suppression caused only by the settle says 'settling'",
                      len(resumed) == 1 and "settling" in resumed[0]
                      and "closed" not in resumed[0]
                      and "desktop" not in resumed[0],
                      f"{resumed}")
            finally:
                lab._log = real_log
            lab.layers["holes"].clear()
            lab.session.reset()
            # And the operator-console flow (never opened) is untouched.
            lab._target_ever_opened = False
            _flip_shot(31)
            check("  while --no-target (never opened) still counts shots",
                  lab.session.shots == 1, f"{lab.session.shots}")
            if had_play:
                lab.mixer.play = real_play
            else:
                del lab.mixer.play
            lab._target_ever_opened = real_ever_opened

        finally:
            _CLmod.time = real_time
        lab.target_window = real_window
        lab._cover_checked_at = 0.0
        # Long since settled, as it would be in a real session.
        lab._cover_true_since = time.perf_counter() - 60.0
        lab._cover_ok = True
        lab.layers["holes"].clear()
        lab.session.reset()

        # THE PLAY-AREA BORDER BINDS TARGETS AND NOT HOLES.
        #
        # His instruction, 2026-10-16: "use a grey line to outline the play
        # area inside the monitor; and that means no targets (or bouncing
        # balls, etc.) go outside of that area. However if the shots
        # detected on screen outside, draw the holes outside, just do not
        # draw any targets".
        #
        # Asserted through `scene.playable`, which is the ONE rectangle every
        # mode already places targets inside -- so this check is also what
        # proves no mode needs to know the feature exists (RULES E2).
        from lsa.playarea import PlayArea as _PA
        import lsa.playarea as _playarea_mod

        # -- THE BORDER IS OFF (owner, 2026-10-16: "disable that border
        # feature for now") and so is the camera-coverage restriction
        # ("there is no need for the border that restricts the balls").
        # A play_area still in his calibration.json must be IGNORED.
        check("the play-area border switch is off", _playarea_mod.BORDER_ENABLED is False)
        check("no border layer is visible",
              not any(getattr(l, "visible", False) and l.name == "playarea"
                      for l in lab.scene.layers))
        check("a freshly built scene has no visible border layer either",
              not lab._build_scene()[1]["playarea"].visible)
        mapping.play_area = _PA(0.0, 0.0, 2 / 3, 1.0)
        lab._update_playable(mapping)
        check("with the border off, a stored play_area narrows nothing "
              "(scene.playable is None)", lab.scene.playable is None,
              f"{lab.scene.playable}")
        check("  and the scene carries no border to draw", lab.scene.play_area is None)
        lab._area_shots = []
        lab._update_play_area(None)
        check("  and no border is measured or persisted",
              lab._area_shots == [] and mapping.play_area is not None,
              "the stored value must be left alone, not deleted")
        # With ONLY the camera-coverage switch on, targets are confined to
        # the camera rectangle again: proof the check above can fail.
        mapping.play_area = None
        lab.RESTRICT_TARGETS_TO_CAMERA_VIEW = True
        lab._update_playable(mapping)
        check("flipping the camera-coverage switch on restricts targets again",
              lab.scene.playable is not None, f"{lab.scene.playable}")
        del lab.RESTRICT_TARGETS_TO_CAMERA_VIEW
        # With the border switch flipped on, the border is back in force.
        _playarea_mod.BORDER_ENABLED = True
        mapping.play_area = _PA(0.0, 0.0, 2 / 3, 1.0)
        lab._update_playable(mapping)
        check("flipping the border switch on makes it bind again",
              lab.scene.play_area is mapping.play_area
              and lab.scene.playable is not None
              and lab._build_scene()[1]["playarea"].visible)
        # (everything below tests the border's behaviour, so it runs ON)
        lab.RESTRICT_TARGETS_TO_CAMERA_VIEW = True
        lab._update_playable(mapping)
        pl = lab.scene.playable
        check("targets are confined to the measured play area",
              pl is not None and pl.right() <= int(SW * 2 / 3) + 1,
              f"playable right edge {pl.right() if pl else None} against a "
              f"border at {int(SW * 2 / 3)}")
        check("  and the scene carries the border for the layer to draw",
              lab.scene.play_area is mapping.play_area)

        # A shot in the off-limits third STILL draws a hole. The whole
        # argument for advisory rather than hard: a refused real hit looks
        # exactly like a broken detector from the firing line, and this
        # project has twice turned a loud failure into a quiet wrong answer.
        lab.layers["holes"].clear()
        lab.session.reset()
        outside = (SW * 0.92, SH * 0.5)
        cam_out = screen_to_camera(*outside)
        lab._on_shot(_Shot(x=cam_out[0], y=cam_out[1], nx=0.5, ny=0.5,
                           frames=3, duration_ms=85.0, peak=180, area=120.0,
                           first_timestamp=0.0, last_timestamp=0.085,
                           index=5, track_id=0, elongation=1.2))
        check("a shot OUTSIDE the border still draws its hole",
              len(lab.layers["holes"].holes) == 1,
              f"{len(lab.layers['holes'].holes)} holes -- advisory, by his "
              f"instruction")
        if lab.layers["holes"].holes:
            h = lab.layers["holes"].holes[-1]
            check("  and where it actually landed, outside the border",
                  h.x > SW * 2 / 3, f"hole at x={h.x:.0f}")
        mapping.play_area = None
        lab._update_playable(mapping)
        lab.layers["holes"].clear()
        lab.session.reset()

        # AND THE BORDER MUST NOT OUTLIVE THE UNITS IT WAS MEASURED IN.
        #
        # His string of 2026-10-16, right third, bar 14:
        #
        #     up to 14:14:41   16 shots, peaks 10,12,13,14,18,26,34,38,46,...
        #     after            7 shots,  peaks 21,54,77,104,107,109,181
        #
        # The gain map was applied at 14:14:13, and a peak after it is a
        # NORMALISED score where a peak before it is raw. The border took a
        # median across both, so the pre-map readings held it down and a
        # third of his screen stayed fenced off for the rest of the session
        # -- after the very fix that cured that third had landed. He saw the
        # grey line; the numbers behind it were obsolete.
        lab.mapping = mapping
        lab.sp_minpeak.setValue(14)
        lab._floor_measured = True
        lab._area_shots = []
        mapping.play_area = None
        idx = 100

        def _fire(fx, fy, peak):
            nonlocal idx
            idx += 1
            c = screen_to_camera(SW * fx, SH * fy)
            lab._on_shot(_Shot(x=c[0], y=c[1], nx=0.5, ny=0.5, frames=3,
                               duration_ms=85.0, peak=peak, area=300.0,
                               first_timestamp=0.0, last_timestamp=0.085,
                               index=idx, track_id=0, elongation=1.2))

        from lsa.playarea import MIN_PER_BIN_BLOCK as _NEED_TO_FENCE
        for fx, peak in ((0.2, 90), (0.5, 90), (0.85, 12)):
            for fy in (0.3, 0.5, 0.8):
                # Fencing takes away part of his screen, so it needs
                # MIN_PER_BIN_BLOCK readings; three only get a bin reported.
                for _ in range(_NEED_TO_FENCE):
                    _fire(fx, fy, peak)
        check("a weak region measured in these units IS fenced off",
              mapping.play_area is not None
              and mapping.play_area.x1 < 0.9,
              f"{mapping.play_area}")

        # Now the gain map changes, which is what happened to him mid-string.
        n_before = len(lab._area_shots)
        lab._set_gain_map(None, "the test")
        check("  changing the gain map retires the evidence behind it",
              lab._area_shots == [] and n_before > 0,
              f"{n_before} shots before, {len(lab._area_shots)} after -- "
              f"keeping them is what pinned his border for a session")

        # The same region now reads strong, because the map is doing its job.
        for fx in (0.2, 0.5, 0.85):
            for fy in (0.3, 0.5, 0.8):
                for _ in range(3):
                    _fire(fx, fy, 90)
        check("  and the border widens again on the new evidence",
              mapping.play_area is None,
              f"{mapping.play_area} -- a border that can only ever shrink "
              f"would fence him in permanently on three early shots")
        lab._area_shots = []
        mapping.play_area = None
        _playarea_mod.BORDER_ENABLED = False
        del lab.RESTRICT_TARGETS_TO_CAMERA_VIEW
        lab._update_playable(mapping)
        lab.layers["holes"].clear()
        lab.session.reset()

        # THERE IS ALWAYS SOMETHING TO SHOOT AT.
        #
        # Reported twice in one session: "I'm hoping I will see some targets
        # this time", then "I still don't see the targets. Did you plan to
        # add the targets?" -- and nothing was missing. His
        # settings.local.json carried layers.aim = false, so free aim drew
        # the background, the holes and nothing else. A saved setting
        # silently beating the product, which cost him holes=false the same
        # morning and min_frames for days before that.
        #
        # And a third time, 2026-10-16: "free aim" was a blank screen, and
        # the owner said it is useless. There is no such state any more.
        host = lab.layers.get("modes")
        # SINCE 2026-10-18 IT STARTS ON THE MENU, at his instruction: "the app
        # should start with the menu mode active". The property these three
        # readings of a blank screen bought is unchanged -- something is always
        # on screen to shoot at -- and the menu is shootable by construction:
        # its tiles ARE the targets, and shooting one starts that drill.
        # Asked of a FRESHLY BUILT scene, not of the live host: this is a
        # statement about construction, and by this point the suite has been
        # driving one lab for four hundred lines and has chosen drills of its
        # own. Reading the live host made the check depend on everything that
        # ran before it, which is how it passed while meaning nothing.
        _fresh = lab._build_scene()[1]["modes"].mode
        check("the app starts on the static targets while it measures itself",
              _fresh is not None and _fresh.name == _CLmod.STARTUP_MODE_NAME,
              str(getattr(_fresh, "name", None)))
        host.mode = None
        lab._ensure_something_to_shoot_at("the test")
        check("a screen with no mode gets the static targets back",
              host.mode is not None and host.mode.name == "static")
        host.visible = False
        lab._ensure_something_to_shoot_at("the test")
        check("  and a hidden targets layer is switched back on", host.visible)
        class _FakeMode:
            name = "fake"
            title = "Fake"
        host.mode = _FakeMode()
        lab._ensure_something_to_shoot_at("the test")
        check("  but a running mode keeps its own screen",
              host.mode.name == "fake")
        host.mode = None
        lab._ensure_something_to_shoot_at("the test")

        # Without a mapping there is nothing to draw, and drawing anyway would
        # put a hole in an arbitrary place and look like an aiming error.
        lab.mapping = None
        lab.homography = None
        lab.layers["holes"].clear()
        lab._on_shot(_Shot(x=cam[0], y=cam[1], nx=0.5, ny=0.5, frames=3,
                           duration_ms=85.0, peak=180, area=120.0,
                           first_timestamp=0.0, last_timestamp=0.085,
                           index=2, track_id=0, elongation=1.2))
        check("with no calibration, no hole is invented",
              len(lab.layers["holes"].holes) == 0,
              f"{len(lab.layers['holes'].holes)} holes")
        lab.mapping = mapping

        # How much a mis-detected corner actually costs. The live run put the
        # top-right corner at x=678 instead of ~796, because a specular
        # reflection on the right of the panel made the lit and unlit frames
        # nearly identical there. This pins the consequence: it is not a small
        # error, and it is why the flash outline must never be the final word.
        bad = _np.float32([[137, 201], [678, 219], [864, 580], [197, 719]])
        H_bad = _cv2.getPerspectiveTransform(
            bad, _np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]]))
        flash_errs = []
        for sx, sy in [(SW * f, SH * g) for f in (0.2, 0.5, 0.8) for g in (0.3, 0.7)]:
            cx, cy = screen_to_camera(sx, sy)
            q = _cv2.perspectiveTransform(_np.float32([[[cx, cy]]]), H_bad)[0][0]
            flash_errs.append(_math.hypot(float(q[0]) - sx, float(q[1]) - sy))
        flash_mm = float(_np.mean(flash_errs)) / (SW / 940.0)
        print(f"         flash outline with the real mis-detected corner: {flash_mm:.0f}mm")
        check("the shot-measured grid beats the flash outline by a wide margin",
              mm < flash_mm / 10.0, f"grid {mm:.2f}mm vs flash {flash_mm:.0f}mm")

    # The nine shots from the live run of 2026-10-13 20:18, verbatim. Six were
    # good and three were flinched; this pins that one bad trigger pull can no
    # longer drag the mapping for the whole side of the screen it sits on.
    live_cam = [(247.5, 255.1), (548.8, 256.0), (769.3, 272.0),
                (312.1, 486.7), (588.4, 439.2), (792.4, 407.8),
                (312.8, 684.8), (613.0, 610.5), (814.2, 558.9)]
    live = [Sample(s, c) for s, c in zip(grid_points(SW, SH, 3, 3), live_cam)]
    raw = Mapping.fit(live, (SW, SH), use_residuals=False)
    cleaned, live_note = Mapping.fit_validated(live, (SW, SH))
    px_per_mm = SW / 940.0
    raw_mm, clean_mm = raw.rms_before / px_per_mm, cleaned.rms_before / px_per_mm
    print(f"         live 9 shots: all points {raw_mm:.1f}mm -> after rejection "
          f"{clean_mm:.1f}mm   ({live_note})")
    check("stray calibration shots are rejected", "dropped" in live_note, live_note)
    check("rejecting them improves the live fit", clean_mm < raw_mm * 0.6,
          f"{raw_mm:.1f}mm -> {clean_mm:.1f}mm")
    check("never drops so much that the fit is unconstrained",
          len(cleaned.samples) >= 6, f"{len(cleaned.samples)} kept")

    # Shot agreement, with the real spreads from 2026-10-13 20:30: points 6-9
    # grouped within 9-11px, points 2-5 contained a stray hundreds of px away.
    # Shots are (x, y, peak): the peak is what the sensitivity map is built from.
    tight = [(100.0, 100.0, 70.0), (106.0, 103.0, 66.0), (97.0, 108.0, 71.0)]
    stray = (460.0, 390.0, 55.0)
    check("three agreeing shots are accepted",
          lab._agreeing_group(tight, 3) is not None)
    check("a stray among three is not enough on its own",
          lab._agreeing_group(tight[:2] + [stray], 3) is None)
    grp = lab._agreeing_group(tight[:2] + [stray] + [tight[2]], 3)
    check("the stray is excluded once a third real shot arrives",
          grp is not None and stray not in grp, str(grp))
    check("two shots can never satisfy a three-shot requirement",
          lab._agreeing_group(tight[:2], 3) is None)

    # The sensitivity map, which is what fixes the far corner of the screen.
    from lsa.calibrate.mapping import sensitivity_map
    strong_weak = [Sample((0, 0), (200, 150), 80.0), Sample((0, 0), (1000, 150), 28.0),
                   Sample((0, 0), (200, 550), 75.0), Sample((0, 0), (1000, 550), 30.0)]
    gmap = sensitivity_map(strong_weak, (1280, 720))
    check("a sensitivity map is built from the calibration shots", gmap is not None)
    if gmap is not None:
        check("it is weaker where the laser scored weaker",
              gmap[150, 1000] < gmap[150, 200] * 0.7,
              f"strong side {gmap[150, 200]:.2f}, weak side {gmap[150, 1000]:.2f}")
        check("it never amplifies a dead region without limit", gmap.min() >= 0.25,
              f"floor {gmap.min():.2f}")
    check("no map is built when signal strength was never recorded",
          sensitivity_map([Sample((0, 0), (10, 10)) for _ in range(6)], (1280, 720)) is None)

    # And the detector must actually use it: the same faint dot should score
    # higher once the region it sits in is known to be a weak one.
    from lsa.detect.laser import LaserDetector as _LD, DetectorParams as _DP, ScoreMode as _SM
    faint = _np.zeros((720, 1280, 3), _np.uint8)
    _cv2.circle(faint, (1000, 150), 6, (0, 0, 60), -1)
    det = _LD(_DP(mode=_SM.CHROMA, threshold=10, min_peak=12, min_area=2, use_background=False))
    flat = det.process(faint)
    det.set_gain_map(gmap)
    lifted = det.process(faint)
    check("normalising lifts a dot in a weak region",
          bool(lifted) and bool(flat) and lifted[0].peak > flat[0].peak,
          f"{flat[0].peak if flat else None} -> {lifted[0].peak if lifted else None}")

    # Auto-alignment, driven through the app rather than the algorithm alone:
    # open the real target window, grab what it is showing, synthesise the view
    # a camera would have of it, and check the app accepts a correction.
    import cv2 as _cv
    lab._toggle_target_mode()
    app.processEvents()

    # WAIT FOR THE WINDOW TO ACTUALLY PAINT before registering against it.
    #
    # One pass of processEvents is not enough for a fullscreen window to
    # put pixels on a surface, and `_on_align_frame` grabs its own
    # template from that window. Against a half-painted one, ECC found a
    # 109px shift, the app correctly refused it as "the camera appears to
    # have moved", raised the alert that says so -- and the NEXT check,
    # "no alert is showing by default", failed too. One race, two red
    # lines, neither of them about alignment.
    #
    # Caught only because adding one invisible layer to the scene flipped
    # it, which is exactly the kind of result that gets blamed on the
    # change in front of you. A flaky test in a project whose standing
    # rule is "repeatability is the property, not a good run" is worse
    # than no test: it teaches you to re-run instead of to look.
    def _settled_snapshot(tries: int = 40):
        last = None
        for _ in range(tries):
            app.processEvents()
            now = lab._scene_snapshot()
            if (last is not None and now is not None
                    and now.shape == last.shape
                    and bool((now == last).all())):
                return now
            last = now
            time.sleep(0.02)
        return last

    # Auto-alignment registers the camera image against this snapshot and
    # needs TEXTURE. The clean shooting screen (aim rings only) has too
    # little for ECC (it reports an 86px "shift"), so the information is
    # shown for this check -- as it is when he presses 0. Auto-align is off
    # by default (`_auto_align = False`); if it is ever turned on it should
    # be told this.
    lab._info_full = True
    lab._apply_info_policy()
    snap = _settled_snapshot()
    check("the app can read back what the target screen is showing",
          snap is not None and snap.ndim == 3 and snap.size > 0,
          f"{None if snap is None else snap.shape}")
    if snap is not None:
        size = lab._screen_size()
        quad = _np.float32([[331, 91], [975, 165], [1011, 537], [371, 667]])
        rect = _np.float32([[0, 0], [size[0], 0], list(size), [0, size[1]]])
        H_true = _cv.getPerspectiveTransform(quad, rect).astype(float)
        full = _cv.resize(snap, size, interpolation=_cv.INTER_LINEAR)
        _ok, _inv = _cv.invert(H_true)
        cam = _cv.warpPerspective(full, _inv, (1280, 720))
        cam = (cam.astype(_np.float32) * 0.3).astype(_np.uint8)
        off = _cv.getPerspectiveTransform(quad + _np.float32([9, -7]), rect).astype(float)
        lab.mapping = Mapping(homography=off, screen_size=size)
        before = lab._align_applied
        lab._on_align_frame(cam)
        check("the app applies an auto-alignment correction",
              lab._align_applied == before + 1,
              lab.aligner.last.reason if lab.aligner.last else "no result")
        # And the safety case: a camera that has been moved must be refused.
        far = _cv.getPerspectiveTransform(quad + _np.float32([210, -150]), rect).astype(float)
        lab.mapping = Mapping(homography=far, screen_size=size)
        held = lab._align_applied
        lab._on_align_frame(cam)
        check("a moved camera is refused rather than silently fitted",
              lab._align_applied == held,
              lab.aligner.last.reason if lab.aligner.last else "no result")
    # THE CHECK ABOVE DELIBERATELY RAISES "THE CAMERA HAS MOVED", so clear
    # it here rather than leaving it for a later check to trip over.
    #
    # It did exactly that on 2026-10-16, when moving the camera pane below
    # the top band changed what ECC saw and therefore which branch ran.
    # "no alert is showing by default" then failed on an alert this file
    # had asked for ten lines earlier -- a red line about alignment
    # appearing as a red line about alerts, which is how one race produced
    # two unrelated-looking failures the last time (see the note above).
    #
    # Cleared explicitly, so that check keeps its real meaning: an alert
    # that NOTHING provoked.
    _al = lab.scene.get("alert")
    if _al is not None:
        _al.visible = False
    lab._toggle_target_mode()
    app.processEvents()

    # Arcade modes, driven the way the app drives them: keys on the target
    # screen, then a shot through the real handler.
    import lsa.modes as _arcade
    lab._toggle_target_mode()
    app.processEvents()
    host = lab.layers["modes"]
    # MODE 1 WHILE IT MEASURES, THE MENU WHEN IT HAS ANSWERED. His order,
    # 2026-10-18: "start calibration with mode 1 and display the menu when
    # calibration is complete". The reason is measurable: the screen search,
    # the exposure probe and the self test all photograph this screen, and a
    # dot landing on lit grey loses its red excess (B10) -- so eleven lit
    # tiles during the checks degrade the measurements that decide whether he
    # may shoot.
    import lsa.tools.camera_lab as _CLmod3
    import lsa.modes as _m_sel3
    check("a freshly built scene holds the startup drill, not the menu",
          lab._build_scene()[1]["modes"].mode is not None
          and lab._build_scene()[1]["modes"].mode.name
          == _CLmod3.STARTUP_MODE_NAME)
    check("...every time it is built, not once",
          all(lab._build_scene()[1]["modes"].mode.name
              == _CLmod3.STARTUP_MODE_NAME for _ in range(3)))
    lab._menu_shown_at_startup = False
    lab._select_mode(_m_sel3.by_name(_CLmod3.STARTUP_MODE_NAME)())
    lab._menu_when_setup_done()
    check("the menu goes up when setup is done",
          lab.layers["modes"].mode.name == "menu",
          lab.layers["modes"].mode.name)
    lab._select_mode(_m_sel3.by_name("balls")())
    lab._menu_when_setup_done()
    check("...and a later verdict does not yank him out of a drill he chose",
          lab.layers["modes"].mode.name == "balls",
          lab.layers["modes"].mode.name)
    lab._select_mode(None)
    check("there is no blank 'free aim': asking for no mode gives the static targets",
          host.mode is not None and host.mode.name == "static")
    lab.layers["holes"].add(100.0, 100.0, 1, 8)
    lab.session.shots = 7
    first_static = host.mode
    lab._on_target_key(Qt.Key_1)
    check("key 1 restarts the static targets: holes and score cleared",
          host.mode.name == "static" and host.mode is not first_static
          and len(lab.layers["holes"].holes) == 0 and lab.session.shots == 0)
    ctx_s = lab.scene._context(0.0, 0.0, lab.target_window.rect())
    sw_, sh_ = ctx_s.width, ctx_s.height
    out_c = host.shot(_arcade.ShotEvent(sw_ / 2.0, sh_ / 2.0, 1.0, 1), ctx_s)
    out_c2 = host.shot(_arcade.ShotEvent(sw_ / 2.0, sh_ / 2.0, 1.1, 2), ctx_s)
    check("static: the centre target scores and does not vanish when hit",
          out_c.hit and out_c2.hit and out_c.points == out_c2.points > 0)
    out_k = host.shot(_arcade.ShotEvent(sw_ * 0.5, sh_ * 0.03, 1.2, 3), ctx_s)
    check("static: off every ring is a miss", not out_k.hit)
    from lsa.display.layers import AimField as _AF
    corner = _AF().marks(ctx_s)[1]
    out_n = host.shot(_arcade.ShotEvent(corner[0], corner[1], 1.3, 4), ctx_s)
    check("static: a corner target scores", out_n.hit)
    # 5 since 2026-10-18: shoot/no-shoot was removed at his request and
    # everything after it moved up a number.
    lab._on_target_key(Qt.Key_5)
    check("key 5 is the clear-the-board drill",
          host.mode.name == "circles", host.mode.name)
    if host.mode is not None and getattr(host.mode, "targets", None):
        tgt = host.mode.targets[0]
        ctx_m = lab.scene._context(0.0, 0.0, lab.target_window.rect())
        res = host.shot(_arcade.ShotEvent(tgt.x, tgt.y, 1.0, 1), ctx_m)
        check("a shot on a target is a hit", res.hit)
        miss = host.shot(_arcade.ShotEvent(tgt.x, tgt.y, 1.1, 2), ctx_m)
        check("the same target cannot be hit twice", not miss.hit)
    lab._on_target_key(Qt.Key_2)
    check("key 2 switches to the other mode", host.mode is not None
          and host.mode.name == "ball", host.mode.name if host.mode else "none")
    # Pressing a mode's own key again RESTARTS it (a fresh round), it does
    # not exit to free aim: owner, 2026-10-16, "zero toggles the info and the
    # 1/2/3/etc. keys start and restart the shooting modes".
    first_ball = host.mode
    lab._on_target_key(Qt.Key_2)
    check("pressing a mode's key again restarts it",
          host.mode is not None and host.mode.name == "ball"
          and host.mode is not first_ball)
    # The figure modes, through the real key path (the same path the G
    # incident showed can silently swallow a key).
    for key_, name_ in ((Qt.Key_6, "popup"), (Qt.Key_7, "image"),
                        (Qt.Key_8, "advancing"), (Qt.Key_9, "animated")):
        lab._on_target_key(key_)
        mode_before = host.mode
        check(f"key {chr(key_)} selects the {name_} figure mode",
              host.mode is not None and host.mode.name == name_,
              host.mode.name if host.mode else "none")
        lab._on_target_key(key_)
        check(f"pressing key {chr(key_)} again restarts it",
              host.mode.name == name_ and host.mode is not mode_before)
    # The development walk-through (--modetest) must be able to aim at EVERY mode,
    # including the ones that have nothing up at the start. A dev tool that says
    # "no target" for a mode that has one is how a wrong cause gets believed.
    import lsa.tools.camera_lab as _cl
    _rect_m = lab.target_window.rect()
    _aimless = []
    for cls_ in _arcade.available():
        m_ = cls_()
        lab._select_mode(m_)
        ctx_m = lab.scene._context(time.perf_counter(), 0.0, _rect_m)
        if cls_.name == "popup":
            m_.next_at = 0.0
        elif cls_.name == "advancing":
            for a_ in m_._actors:
                a_.live_at = 0.0
        for _ in range(3):
            m_.advance(lab.scene._context(time.perf_counter(), 0.0, _rect_m))
        if getattr(cls_, "is_screen", False):
            # A SCREEN IS NOT A DRILL. The Keyboard Shortcuts page (K) draws no
            # target by design -- the only thing to aim at on it is the MENU
            # button, which every mode carries and which this walk-through does
            # not drive. Exempting it keeps the check meaning what it says: every
            # DRILL offers something to shoot. A screen that started drawing
            # targets would be a different bug and is caught in test_modes.
            continue
        tgt_ = lab._modetest_target(m_, _rect_m)
        if tgt_ is None or not (0 <= tgt_[0] <= _rect_m.width() and 0 <= tgt_[1] <= _rect_m.height()):
            _aimless.append((cls_.name, tgt_))
    check("the --modetest walk-through finds something to aim at in every mode",
          not _aimless, str(_aimless))
    # ...and waits for a mode that has nothing up YET instead of reporting "no target"
    class _StubTimer:
        scheduled: list = []

        @staticmethod
        def singleShot(ms, fn):
            _StubTimer.scheduled.append(ms)
    _real_timer = _cl.QTimer
    _results_before = list(lab._modetest_results)
    _mode_before, _wait_before = lab._modetest_mode, lab._modetest_wait_attempts
    try:
        _cl.QTimer = _StubTimer
        m_ = _arcade.by_name("popup")()
        lab._select_mode(m_)
        lab._modetest_mode = m_
        lab._modetest_wait_attempts = 0
        lab._modetest_fire()
        waited = (len(lab._modetest_results) == len(_results_before)
                  and lab._modetest_wait_attempts == 1 and _StubTimer.scheduled == [400])
        lab._modetest_wait_attempts = 20                         # patience used up: now it may report
        lab._modetest_fire()
        gave_up = (len(lab._modetest_results) == len(_results_before) + 1
                   and lab._modetest_results[-1][3] == "no target")
    finally:
        _cl.QTimer = _real_timer
        lab._modetest_results[:] = _results_before
        lab._modetest_mode, lab._modetest_wait_attempts = _mode_before, _wait_before
    check("a mode with nothing up yet is WAITED for (400 ms, up to 20 times), not "
          "reported as 'no target'", waited)
    check("and only after the patience is used up is 'no target' reported", gave_up)
    lab._on_target_key(Qt.Key_1)
    check("key 1 returns from any mode to the static targets",
          host.mode.name == "static")
    lab._on_target_key(Qt.Key_2)
    # The shot event a mode receives is on the SAME clock as the app's scene
    # (perf_counter) and carries the FIRST frame's stamp: the ball modes
    # rewind by `ctx.now - ev.t`. It was time.monotonic() against the
    # detector's perf_counter, which share no epoch on Windows.
    _seen = {}
    _real_hs = host.shot

    def _spy(ev, ctx_):
        _seen["ev"], _seen["ctx"] = ev, ctx_
        return _real_hs(ev, ctx_)

    host.shot = _spy
    _t_first = time.perf_counter() - 0.060
    lab._cover_checked_at = 0.0
    lab._cover_true_since = time.perf_counter() - 60.0
    lab._cover_ok = True
    _c = screen_to_camera(SW * 0.5, SH * 0.5)
    lab._on_shot(_Shot(x=_c[0], y=_c[1], nx=0.5, ny=0.5, frames=2,
                       duration_ms=40.0, peak=150, area=200.0,
                       first_timestamp=_t_first, last_timestamp=_t_first + 0.017,
                       index=900, track_id=0, elongation=1.0))
    del host.shot
    if "ev" in _seen:
        _age = _seen["ctx"].now - _seen["ev"].t
        check("a shot event's age is measurable: same clock as the scene, first frame",
              0.05 < _age < 0.5, f"age {_age:.3f}s")
    else:
        check("a shot reached the active mode", False, "host.shot never called")
    # -- key 0 is the information toggle (one group) ---------------------------
    _info = [lab.layers[n] for n in lab.INFO_LAYERS]
    lab._calibrating = False
    lab._info_full = False
    lab._apply_info_policy()
    check("0: the default while shooting is a clean screen (no information)",
          not any(l.visible for l in _info if l.name != "prompt"),
          str([(l.name, l.visible) for l in _info if l.visible]))
    lab._on_target_key(Qt.Key_0)
    check("0 shows ALL the information, the key list included",
          all(l.visible for l in _info), str([(l.name, l.visible) for l in _info]))
    check("  and the product is not part of the group",
          lab.layers["holes"].visible and host.mode is not None)
    lab._on_target_key(Qt.Key_0)
    check("0 again hides the lot",
          not any(l.visible for l in _info if l.name != "prompt"))
    # An issue speaks without being asked: not calibrated -> prompt band.
    lab.layers["prompt"].kind = "warn"
    lab._apply_info_policy()
    check("a problem brings the status band back by itself",
          lab.layers["prompt"].visible)
    lab.layers["prompt"].kind = "go"
    lab._apply_info_policy()
    check("  and it goes away again when the problem does",
          not lab.layers["prompt"].visible)
    # Measuring: nothing hidden, prompt handed back for the calibration text.
    lab._grid_active = True
    lab._apply_info_policy()
    check("while the app measures, the prompt and signal meter are handed back",
          lab.layers["prompt"].visible and lab.layers["signal"].visible)
    lab._grid_active = False
    lab._info_full = False
    lab._apply_info_policy()
    # Ready -> the camera pane is stood down once, and P can bring it back.
    cam_pane = lab.layers["camera"]
    cam_pane.visible = True
    lab._camera_pane_stood_down = False
    lab._stand_down_camera_pane("the test")
    check("the camera pane is switched off when the app is ready to be shot at",
          not cam_pane.visible)
    # ...and through the real path: the end of setup does it, unprompted.
    cam_pane.visible = True
    lab._camera_pane_stood_down = False
    lab._setup_is_over()
    check("  and the end of setup does it by itself (saved setting had it on)",
          not cam_pane.visible)
    cam_pane.visible = True
    lab._stand_down_camera_pane("the test again")
    check("  once only: the operator can turn it back on deliberately",
          cam_pane.visible)
    cam_pane.visible = False
    lab._info_last_set.clear()
    lab._on_target_key(Qt.Key_1)
    lab._info_full = False

    # -- key 0 through the REAL key path, and the camera pane never drawn ------
    # 2026-10-17 08:50: "It shows the camera screen and I cannot remove it
    # with Press + 0." The last instance of a key fault escaped because the
    # test called `_on_target_key` directly and bypassed TargetWindow, where
    # layers get the key first and auto-repeat arrives. This goes through
    # `TargetWindow.keyPressEvent` itself.
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    _tw = lab.target_window
    _logged: list = []
    _orig_log = lab._log
    lab._log = lambda m, *a, **k: (_logged.append(str(m)), _orig_log(m, *a, **k))[1]
    def _press(key, repeat=False):
        _tw.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, "", repeat, 1))
    lab._calibrating = False
    lab._info_full = False
    lab._apply_info_policy()
    _logged.clear()
    _press(Qt.Key_0)
    _s1 = lab._info_full
    _l1 = [m for m in _logged if "information on the target screen" in m]
    _press(Qt.Key_0)
    _s2 = lab._info_full
    _l2 = [m for m in _logged if "information on the target screen" in m][len(_l1):]
    check("key 0 through the real key path alternates: on, then off",
          _s1 is True and _s2 is False, f"{_s1} then {_s2}")
    check("  and the log line says SHOWN then HIDDEN, matching the state",
          len(_l1) == 1 and "SHOWN" in _l1[0] and len(_l2) == 1 and "HIDDEN" in _l2[0],
          f"{_l1} / {_l2}")
    _n = len(_logged)
    _press(Qt.Key_0, repeat=True)
    _press(Qt.Key_0, repeat=True)
    check("a HELD key (auto-repeat) does not flip it: no state change, no log line",
          lab._info_full is False and len(_logged) == _n,
          f"{lab._info_full}, {len(_logged) - _n} new line(s)")
    lab.layers["camera"].visible = True
    _bk = lab.layers["camera"].width_frac
    _press(Qt.Key_BracketRight, repeat=True)
    check("  but the pane-size brackets still repeat", lab.layers["camera"].width_frac > _bk)
    lab._log = _orig_log
    # The camera pane must never put a live image on the screen the camera
    # watches: rendered, in the shooting state, with the pane switched on.
    def _render_scene():
        im = QImage(1920, 1080, QImage.Format_RGB32)
        im.fill(QColor("#0a0a0a"))
        pp = QPainter(im)
        _r = QRect(0, 0, 1920, 1080)
        lab.scene.advance(_r)
        lab.scene.paint(pp, _r)
        pp.end()
        return im
    _cp = lab.layers["camera"]
    _cp.set_frame(fake_camera_frame(), "test")
    _cp.set_aiming(False)
    _cp.visible = False
    _without = _render_scene()
    _cp.visible = True
    _with_pane = _render_scene()
    check("camera pane switched ON in the shooting state draws nothing at all",
          _with_pane == _without, "the screen differs: a live camera image is on it")
    check("  and asks for no repaints", not _cp.animated)
    check("  and key 0's information group no longer contains the camera pane",
          "camera" not in lab.INFO_LAYERS, str(lab.INFO_LAYERS))
    _cp.set_aiming(True, None, (640, 360))
    _aimed = _render_scene()
    check("  while AIMING the camera it does draw (the check can fail)",
          _aimed != _without)
    _cp.set_aiming(False)
    _cp.visible = False
    lab._info_full = False
    lab._apply_info_policy()

    # The keys must not fire while calibration is running: they would start a
    # mode over the top of the grid the user is shooting.
    lab._grid_active = True
    lab._on_target_key(Qt.Key_2)
    check("mode keys are inert during calibration", host.mode.name == "static")
    lab._grid_active = False
    lab._toggle_target_mode()
    app.processEvents()

    # No two layers may claim the same key, and no layer may claim a key the
    # app needs. Layers get first refusal on the target screen, so a collision
    # does not error -- it silently shadows, which is how G ("calibrate") spent
    # a whole session doing nothing but switching the background to grey.
    from lsa.modes import available as _all_modes, by_key as _by_key
    from lsa.modes.base import _REGISTRY as _modereg
    APP_KEYS = {Qt.Key_G: "calibrate", Qt.Key_U: "calibrate brightness",
                Qt.Key_N: "new round",
                Qt.Key_0: "info toggle",
                Qt.Key_1: "mode 1", Qt.Key_2: "mode 2", Qt.Key_3: "mode 3",
                Qt.Key_4: "mode 4", Qt.Key_5: "mode 5", Qt.Key_6: "mode 6",
                Qt.Key_7: "mode 7", Qt.Key_8: "mode 8", Qt.Key_9: "mode 9",
                Qt.Key_Space: "skip point", Qt.Key_Backspace: "redo point"}
    # E (erase) clears hits AND bullet holes on purpose: one key, every layer that
    # holds shot history. That is a broadcast, not a collision. It was C until
    # 2026-10-20; the self check took C. The set is READ from the scene, so the check
    # and the thing it checks cannot name different keys.
    from lsa.display.scene import Scene as _Scene
    BROADCAST = set(_Scene.BROADCAST_KEYS)
    claimed: dict[int, str] = {}
    collisions: list[str] = []
    for layer in lab.scene.layers:
        for k in list(APP_KEYS) + [Qt.Key_D, Qt.Key_W, Qt.Key_R, Qt.Key_Y,
                                   Qt.Key_C, Qt.Key_K, Qt.Key_F, Qt.Key_E, Qt.Key_M, Qt.Key_H, Qt.Key_I,
                                   Qt.Key_P, Qt.Key_S, Qt.Key_A, Qt.Key_O, Qt.Key_T,
                                   Qt.Key_B, Qt.Key_L]:
            was = getattr(layer, "visible", True)
            try:
                took = bool(layer.on_key(k))
            except Exception:
                took = False
            layer.visible = was
            if not took:
                continue
            if k in APP_KEYS:
                collisions.append(f"{layer.name} takes {APP_KEYS[k]}")
            elif k in BROADCAST:
                continue          # deliberately handled by several layers at once
            elif k in claimed:
                collisions.append(f"{layer.name} and {claimed[k]} both take key {k}")
            else:
                claimed[k] = layer.name
    check("no layer shadows a key the app needs", not collisions,
          "; ".join(collisions) if collisions else "none")
    # NO KEY HAS TWO OWNERS (2026-10-20). The menu claims O and T while it is the
    # active mode (the player-count choice), and no override is declared because none
    # is needed: S stays the stats HUD, M stays the menu key, T is free (the self check
    # moved to C) and O never had a job. The walk above runs with a drill active; this
    # one runs with the menu showing and requires the keys the menu claims to equal its
    # own list EXACTLY, and requires that none of them is a key the app handles
    # anywhere else, read from the code that handles keys rather than from a list kept
    # beside it.
    import ast as _ast19
    import inspect as _inspect19
    from lsa.modes import menu as _menu_mod19
    from lsa.modes.menu import Menu as _Menu19
    from lsa.display.window import TargetWindow
    from lsa.tools import camera_lab as _cl19
    _host19 = lab.layers["modes"]
    _prev_mode19 = _host19.mode
    _ctx19 = lab.scene._context(0.0, 0.0, _QRect(0, 0, WIDTH, HEIGHT))
    _host19.set_mode(_Menu19(), _ctx19, players=1)
    try:
        _claimed19 = {k for k in list(APP_KEYS) + [Qt.Key_D, Qt.Key_W, Qt.Key_R, Qt.Key_Y, Qt.Key_C, Qt.Key_K, Qt.Key_F,
                                                   Qt.Key_M, Qt.Key_H, Qt.Key_I, Qt.Key_P, Qt.Key_S,
                                                   Qt.Key_A, Qt.Key_B, Qt.Key_L, Qt.Key_E, Qt.Key_J,
                                                   Qt.Key_O, Qt.Key_T]
                      if _host19.on_key(k)}
    finally:
        _host19.set_mode(_prev_mode19, _ctx19, players=1)
    _own19 = set(_menu_mod19.MENU_OWN_KEYS)

    def _handled_by_app19():
        """Every Qt.Key_* named in the app's own key handler and in the window's toggle table."""
        src_ = _inspect19.getsource(_cl19.CameraLab._on_target_key)
        import textwrap as _tw19
        names_ = {n_.attr for n_ in _ast19.walk(_ast19.parse(_tw19.dedent(src_)))
                  if isinstance(n_, _ast19.Attribute) and n_.attr.startswith("Key_")}
        keys_ = {getattr(Qt, n_) for n_ in names_}
        keys_ |= set(TargetWindow.TOGGLES)
        return keys_

    def _collisions19(menu_keys, app_keys, layer_claims):
        """Keys the menu claims that something else also owns."""
        return sorted(set(menu_keys) & (set(app_keys) | set(layer_claims)))

    _app19 = _handled_by_app19()
    _layer19 = {k for k in claimed} | set(BROADCAST)
    check("(the scan is not hollow) the app's key handler names a real set of keys, including the ones it must",
          len(_app19) > 10 and Qt.Key_C in _app19 and Qt.Key_A in _app19 and Qt.Key_F in _app19 and Qt.Key_K not in _app19 and Qt.Key_F6 not in _app19 and Qt.Key_S in _app19 and Qt.Key_G in _app19, str(len(_app19)))
    check("with the menu showing, the keys it claims are EXACTLY its own list (O and T): nothing undeclared, "
          "nothing listed that is not claimed",
          _claimed19 == _own19 == {Qt.Key_O, Qt.Key_T},
          f"claimed {sorted(hex(k) for k in _claimed19)}, listed {sorted(hex(k) for k in _own19)}")
    check("NO KEY HAS TWO OWNERS: neither menu key is handled by the app or claimed by any other layer, and there "
          "is no override left to declare",
          _collisions19(_own19, _app19, _layer19) == [] and _menu_mod19.DELIBERATE_KEY_OVERRIDES == {},
          str(_collisions19(_own19, _app19, _layer19)))
    check("...and the collision check CAN fail: handed a menu that also claimed S, or G, or C, it names them",
          _collisions19(_own19 | {Qt.Key_S}, _app19, _layer19) == [Qt.Key_S]
          and Qt.Key_G in _collisions19(_own19 | {Qt.Key_G}, _app19, _layer19)
          and Qt.Key_C in _collisions19(_own19 | {Qt.Key_C}, _app19, _layer19))
    check("...and the former exception is gone: C, K, A and F are claimed by NO layer (the app owns C, A and F; K is reserved), E is the one broadcast",
          not ({Qt.Key_C, Qt.Key_K, Qt.Key_A, Qt.Key_F} & _layer19) and BROADCAST == {Qt.Key_E})
    # THE PRINTED INSTRUCTIONS name the key that does the job (2026-10-21: C self-calibrates, A is the self-check, F the noise floor).
    from lsa.display.layers import KeyHelp as _KH19
    _help19 = " | ".join(_KH19().lines)
    check("the key legend the app draws says C self-calibrate, A self-check, F noise floor and E clear hits; K and F6 are not in it",
          "C self-calibrate" in _help19 and "A self-check" in _help19 and "F measure the noise floor" in _help19
          and "E clear hits" in _help19 and "C self test" not in _help19 and "F6" not in _help19
          and "K Keyboard Shortcuts" in _help19 and "K calibrate" not in _help19
          and "C clear hits" not in _help19, _help19[:200])
    _log19: list = []
    _real_log19 = lab._log
    lab._log = lambda m_: (_log19.append(m_), _real_log19(m_))[1]
    try:
        _rep19 = getattr(lab, "_last_report", None)
        lab._last_report = None
        lab._on_target_key(Qt.Key_Q)
        _map19 = lab.mapping
        _cap19, _tw19x = lab.capture, lab.target_window
        lab.mapping = None
        lab.capture = object()
        lab.target_window = object()
        lab.start_self_test()
    finally:
        lab.target_window = _tw19x
        lab.mapping, lab.capture = _map19, _cap19
        lab._last_report = _rep19
        lab._log = _real_log19
    _said19 = " ".join(_log19)
    check("the lines the app LOGS when asked for the self check without one say C then A "
          "(self-calibrate, then self-check), and Q says A",
          "press C, then A" in _said19 and "press A" in _said19
          and not __import__("re").search(r"press [KT]\b|then [KT]\b|F6", _said19),
          _said19[:200])
    # 0 and F cover every mode key too: no registered mode key may be one the
    # app uses for something else, and 0 must not be a mode key.
    # FROM THE REGISTRY, not from the digit range. This scanned Qt.Key_0 to
    # Qt.Key_9 and required one mode per digit, so the first mode bound to a
    # LETTER (A, the animated figure, 2026-10-18) made a correct registration
    # fail the build -- the same "pinned to the world as it was" shape as the
    # twelve-figure counts in test_artwork.
    _mode_keys = [e.key for e in _modereg.values() if e.key is not None]
    check("key 0 is not claimed by any mode, and every mode has a distinct key",
          Qt.Key_0 not in _mode_keys
          and len(_mode_keys) == len(set(_mode_keys)) == len(_all_modes()),
          str([(chr(k) if 32 < k < 127 else k) for k in _mode_keys]))
    # 2026-10-21: the drills were renumbered on 2026-10-18 and this check was not,
    # so it asked for "6 clear the board" while 6 is Pop-Up Target and 5 is Clear
    # The Board -- the same staleness the legend itself had. The registry-agreement
    # check below is the one that cannot go stale; this one covers 0, which is not
    # a mode and so is not in the registry.
    _legend_low = [line.lower() for line in lab.layers["help"].lines]
    check("the key legend names 0, 1 and 5 under their current meanings",
          any("0 hide/show" in line for line in _legend_low)
          and any("1 static targets" in line for line in _legend_low)
          and any("5 clear the board" in line for line in _legend_low), str(_legend_low)[:200])
    # EVERY fixed string a target-screen layer draws must fit on the screen it is drawn
    # on, at 4K and at 1080p. The verdict board got a font-metrics check after he
    # photographed its clipped text; nothing generalised it, so the key legend grew a
    # 145-character line that ran 848 px off the right edge at 4K (448 px at 1080p) and
    # lost "(press a key again = new round)". This draws every layer, and every mode,
    # through a painter that measures each string with the font it is drawn in.
    from PySide6.QtCore import QPoint as _QPt, QPointF as _QPtF, QRect as _QRc, QRectF as _QRcF
    from PySide6.QtGui import QPainter as _QPaintSpy, QImage as _QImSpy

    class _TextSpy(_QPaintSpy):
        seen: list = []

        def drawText(self, *a):
            fm = self.fontMetrics()
            if a and isinstance(a[0], int):
                self.seen.append((float(a[0]), fm.horizontalAdvance(a[2]), a[2]))
            elif a and isinstance(a[0], (_QPt, _QPtF)):
                self.seen.append((float(a[0].x()), fm.horizontalAdvance(a[1]), a[1]))
            elif len(a) >= 3 and isinstance(a[0], (_QRc, _QRcF)):
                # a rectangle form: the text must fit the rectangle it is given
                self.seen.append((float(a[0].x()), fm.horizontalAdvance(a[2]), a[2],
                                  float(a[0].width())))
            return super().drawText(*a)

    over_text = []
    n_text = 0
    all_text = set()
    for (W_, H_) in ((3840, 2160), (1920, 1080)):
        rect_ = _QRc(0, 0, W_, H_)
        img_ = _QImSpy(W_, H_, _QImSpy.Format_RGB32)
        _TextSpy.seen = []
        sp_ = _TextSpy(img_)
        ctx_t = lab.scene._context(time.perf_counter(), 0.0, rect_)
        painters = [(la.name, la, la.visible) for la in lab.scene.layers]
        for name_, la, was_ in painters:
            if name_ in ("camera", "minimap", "aimview"):
                continue                      # camera imagery, not strings
            la.visible = True
            try:
                la.paint(sp_, ctx_t)
            except Exception:
                pass
            la.visible = was_
        for cls_ in _arcade.available():
            m_ = cls_()
            m_.start(ctx_t)
            try:
                m_.paint(sp_, ctx_t)
            except Exception:
                pass
        sp_.end()
        n_text += len(_TextSpy.seen)
        all_text.update(r[2] for r in _TextSpy.seen)
        for rec in _TextSpy.seen:
            x_, w_, txt_ = rec[0], rec[1], rec[2]
            limit_ = (rec[3] if len(rec) > 3 else W_ - x_)
            if w_ > limit_ + 1:
                over_text.append(f"{W_}x{H_}: {txt_[:50]!r} needs {w_:.0f}px, has {limit_:.0f}px")
    check(f"every string any layer or mode draws fits on its screen, at 3840x2160 and "
          f"1920x1080 ({n_text} strings measured)", not over_text, "; ".join(over_text[:3]))
    # PER ENTRY, not per line. The legend draws its keys BOLD and its prose
    # normal (his request, 2026-10-18), so a row reaches the painter as
    # several strings and no full line is ever drawn as one. The property is
    # unchanged: every piece of legend text must have been measured, or the
    # width check above is measuring nothing.
    _legend_parts = [part for row in lab.layers["help"].ROWS for part in row
                     if part[1].strip()]
    check("...and the measurement saw the strings that matter: every key-legend entry "
          "and a figure mode's standing hint",
          all(any(text in t for t in all_text) for _k, text in _legend_parts)
          and any("a figure pops up without warning" in t for t in all_text),
          f"{len(all_text)} distinct strings, {len(_legend_parts)} legend entries")
    # (RULES C1: this check is shown to fail by `tests/mutate_figures.py`, which puts the
    # 145-character legend line back and requires the measurement above to go red.)
    # THE LEGEND MUST AGREE WITH THE REGISTRY (2026-10-21). KeyHelp.ROWS is
    # hand-written because `display` may not import `modes`, and it listed the
    # pre-2026-10-18 drill numbering for three days with every suite green:
    # nothing compared it with what is registered. Key for key, title for
    # title, in BOTH directions -- a registered drill missing from the legend
    # fails as surely as a legend entry naming a drill that does not exist.
    from lsa.modes import registered as _registered21
    from lsa.modes import menu as _menu_import21        # noqa: F401  (registers M)
    _expected21 = {chr(int(k_)): c_.title for c_, k_ in _registered21() if k_ is not None}
    _legend21 = {k_: t_ for row_ in lab.layers["help"].ROWS for k_, t_ in row_
                 if k_ in _expected21 or (len(k_) == 1 and k_.isdigit() and k_ != "0")}
    check("the key legend's drill entries (1-9 and M) agree with lsa.modes.registered(), key for key and title for title",
          len(_expected21) >= 10 and _legend21 == _expected21,
          f"legend {sorted(_legend21.items())} vs registry {sorted(_expected21.items())}")

    # On-screen alerts. The operator console sits behind a fullscreen target
    # window on the same monitor, so a camera problem reported only there is
    # invisible to the person holding the pistol.
    alert = lab.layers["alert"]
    check("no alert is showing by default", not alert.visible,
          f"showing: {getattr(alert, 'title', '?')!r}")
    lab._alert("THE CAMERA CANNOT SEE THIS SCREEN",
               "Point it at the monitor.\n\nPress F5 when done.", show_camera=True)
    check("a warning appears on the target screen", alert.visible)
    check("it turns the camera view on so the camera can be aimed",
          lab.scene.get("camera").visible)
    from lsa.display import layers as layers        # the cap lives there

    def context_at(lab_, t_):
        """A render context at a chosen moment on the scene clock (B4)."""
        return lab_.scene._context(t_, 0.0, _QRect(0, 0, WIDTH, HEIGHT))

    # IT LEAVES THE SCREEN. Warnings used to be sticky, on the reasoning that
    # a problem does not stop being true after five seconds. True -- and the
    # status band keeps saying so, in one line. What he photographed on
    # 2026-10-18 was a lit panel across the middle of the screen the camera
    # watches, still there long after it was read: "the message should go
    # away". It has no explicit deadline, and the cap clears it anyway.
    check("a warning has no explicit deadline of its own", alert.until == 0.0)
    _t_seen = 1000.0
    alert.advance(context_at(lab, _t_seen))
    check("...and it is still up well before the cap",
          alert.advance(context_at(lab, _t_seen + layers.Alert.MAX_DWELL_S * 0.5))
          is False and alert.visible)
    alert.advance(context_at(lab, _t_seen + layers.Alert.MAX_DWELL_S + 0.1))
    check("...and it clears itself at the cap, whatever raised it",
          not alert.visible, f"cap {layers.Alert.MAX_DWELL_S}s")
    lab._alert("THE CAMERA CANNOT SEE THIS SCREEN",
               "Point it at the monitor.\n\nPress F5 when done.", show_camera=True)
    alert.on_key(Qt.Key_Escape)
    check("Escape does not dismiss a warning: it is read, not clicked away",
          alert.visible)

    # Rendering it must not introduce a laser-coloured pixel, and must actually
    # put ink on the screen rather than silently drawing nothing.
    probe = QImage(WIDTH, HEIGHT, QImage.Format_RGB32)
    probe.fill(0)
    pp = QPainter(probe)
    alert.paint(pp, lab.scene._context(0.0, 0.0, _QRect(0, 0, WIDTH, HEIGHT)))
    pp.end()
    worst_alert, lit = 0, 0
    for x in range(0, WIDTH, 7):
        for y in range(0, HEIGHT, 7):
            c = probe.pixelColor(x, y)
            worst_alert = max(worst_alert, c.red() - max(c.green(), c.blue()))
            if c.red() + c.green() + c.blue() > 40:
                lit += 1
    check("the alert actually draws something", lit > 50, f"{lit} lit samples")
    check("the alert draws no laser-coloured pixel", worst_alert <= 30,
          f"worst red excess {worst_alert}")

    # A timed confirmation should clear itself; otherwise the screen accretes
    # stale good news nobody dismissed.
    lab._alert("CALIBRATED", "all good", kind="good", seconds=2.0)
    check("a confirmation is timed", alert.until > 0)
    alert.advance(lab.scene._context(alert.until + 1.0, 0.0, _QRect(0, 0, WIDTH, HEIGHT)))
    check("and clears itself when the time is up", not alert.visible)

    # The camera pane has to be LEGIBLE when it is being used to aim the
    # camera, and still safe. Greyscale gives both: equal channels mean a red
    # excess of exactly zero, which is the same protection the blue multiply
    # gave, except a person can actually see what the camera sees.
    pane2 = lab.layers["camera"]
    check("the pane is greyscale by default", pane2.greyscale)
    pane2.set_score_mode("ScoreMode.RED")
    check("brightness score modes fall back to the blue multiply",
          not pane2.greyscale)
    pane2.set_score_mode("ScoreMode.CHROMA")

    pane2.set_frame(fake_camera_frame(), "test")
    pane2.set_aiming(True, ((100, 60), (500, 90), (520, 300), (120, 330)), (640, 360))
    pane2.visible = True
    grey_probe = QImage(WIDTH, HEIGHT, QImage.Format_RGB32)
    grey_probe.fill(0)
    gp = QPainter(grey_probe)
    pane2.paint(gp, lab.scene._context(0.0, 0.0, _QRect(0, 0, WIDTH, HEIGHT)))
    gp.end()
    worst_pane, lit_pane = 0, 0
    for x in range(0, WIDTH, 5):
        for y in range(0, HEIGHT, 5):
            c = grey_probe.pixelColor(x, y)
            worst_pane = max(worst_pane, c.red() - max(c.green(), c.blue()))
            if c.red() + c.green() + c.blue() > 60:
                lit_pane += 1
    check("the aiming view is still laser-safe", worst_pane <= 0,
          f"worst red excess {worst_pane}")
    check("the aiming view is big enough to aim by", lit_pane > 2000,
          f"{lit_pane} lit samples")
    box_small = pane2._geometry(lab.scene._context(0.0, 0.0, _QRect(0, 0, WIDTH, HEIGHT)), 16 / 9)
    pane2.set_aiming(False)
    box_thumb = pane2._geometry(lab.scene._context(0.0, 0.0, _QRect(0, 0, WIDTH, HEIGHT)), 16 / 9)
    check("aiming makes it much larger than the corner thumbnail",
          box_small.width() > box_thumb.width() * 1.8,
          f"{box_small.width()}px vs {box_thumb.width()}px")
    pane2.visible = False

    # Camera aiming guides. This path only runs when something is wrong, which
    # is exactly when it must not itself be broken -- a missing import here was
    # live for several minutes and showed up only as a preview that stopped
    # updating.
    from lsa.calibrate.screen import ScreenQuad as _SQ
    FRAME = (1280, 720)

    def quad_at(cx, cy, half_w, half_h, edges=()):
        return _SQ(corners=((cx - half_w, cy - half_h), (cx + half_w, cy - half_h),
                            (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)),
                   coverage=(2 * half_w * 2 * half_h) / (1280 * 720),
                   confidence=1.0, touches_edges=edges, frame_size=FRAME)

    cases = [
        ("centred and well framed", quad_at(640, 360, 400, 225), "good"),
        ("screen off to the left",  quad_at(300, 360, 200, 110), "PAN LEFT"),
        ("screen off to the right", quad_at(1000, 360, 200, 110), "PAN RIGHT"),
        ("screen too high",         quad_at(640, 150, 200, 110), "TILT UP"),
        ("screen too low",          quad_at(640, 600, 200, 110), "TILT DOWN"),
        ("screen too small",        quad_at(640, 360, 120, 70), "MOVE CLOSER"),
        ("screen overflowing",      quad_at(640, 360, 639, 359, ("left", "right")),
                                                                 "MOVE BACK"),
    ]
    for label, q, expect in cases:
        lab.screen_quad = q
        advice, _state = lab._aim_advice(FRAME)
        joined = " | ".join(advice)
        ok_case = (("looks good" in joined) if expect == "good" else (expect in joined))
        check(f"aim advice: {label}", ok_case, joined)

    lab.screen_quad = None
    advice, _ = lab._aim_advice(FRAME)
    check("aim advice when the screen was not found at all",
          any("not found" in a for a in advice), " | ".join(advice))

    # And it must actually render without raising, and put ink down.
    lab.screen_quad = quad_at(400, 300, 220, 130)
    guide_pm = QPixmap(640, 360)
    guide_pm.fill(QColor(0, 0, 0))
    lab._draw_aim_guides(guide_pm, FRAME)
    gimg = guide_pm.toImage()
    lit_guide = sum(1 for x in range(0, 640, 3) for y in range(0, 360, 3)
                    if gimg.pixelColor(x, y).value() > 40)
    check("aim guides render without raising and draw something",
          lit_guide > 100, f"{lit_guide} lit samples")
    lab.screen_quad = None

    # The LIVE region takes priority over the flash-detected quad, because the
    # quad does not move while the camera does.
    live_cases = [
        ((40, 300, 200, 120),  "PAN LEFT"),
        ((1040, 300, 200, 120), "PAN RIGHT"),
        ((500, 20, 300, 160),  "TILT UP"),
        ((0, 0, 1280, 720),    "MOVE BACK"),
        ((560, 320, 100, 60),  "MOVE CLOSER"),
        ((245, 85, 790, 550),  "looks good"),
    ]
    for box, expect in live_cases:
        advice, _ = lab._aim_advice(FRAME, box)
        check(f"live aim advice for box {box}", expect in " | ".join(advice),
              " | ".join(advice))

    # And a live box must win even when a stale quad disagrees with it.
    lab.screen_quad = quad_at(640, 360, 400, 225)          # "looks good"
    advice, _ = lab._aim_advice(FRAME, (40, 300, 200, 120))
    check("a live region overrides a stale screen quad",
          "PAN LEFT" in " | ".join(advice), " | ".join(advice))
    lab.screen_quad = None

    # The bright-region finder itself, on synthetic frames.
    from lsa.camera.capture import CaptureThread as _CT
    dark = _np.zeros((720, 1280, 3), _np.uint8)
    check("no bright region in a dark frame", _CT._bright_region(dark) is None)
    lit = dark.copy()
    lit[200:520, 400:900] = 200
    found = _CT._bright_region(lit)
    ok_box = (found is not None and abs(found[0] - 400) < 40
              and abs(found[1] - 200) < 40 and abs(found[2] - 500) < 60)
    check("a bright rectangle is found roughly where it is", ok_box, str(found))

    # A static object agrees with itself: the calibration trap that produced a
    # mapping with an RMS of 1287px on a 3840px screen, and saved it. These are
    # the real recorded camera positions from that session.
    from lsa.calibrate.mapping import Sample as _S
    lab._grid_points = [(230, 316), (1920, 316), (3610, 316),
                        (230, 1053), (1920, 1053), (3610, 1053),
                        (230, 1790), (1920, 1790), (3610, 1790)]
    lab._grid_cols, lab._grid_rows = 3, 3
    lab._grid_samples = [_S(screen=(230, 316), camera=(279.6, 148.1), peak=71),
                         _S(screen=(1920, 316), camera=(658.9, 173.1), peak=24)]
    bad = lab._out_of_grid_order(2, 487.3, 387.4)
    check("a point that contradicts the grid's own order is caught", bool(bad),
          bad or "NOT CAUGHT")
    good = lab._out_of_grid_order(2, 1035.0, 198.0)
    check("and a point that continues the row is accepted", not good,
          good or "accepted")
    # Lens distortion must not be mistaken for a bad point.
    near = lab._out_of_grid_order(2, 1000.0, 215.0)
    check("modest curvature along the row is tolerated", not near,
          near or "accepted")
    # Down a column, too.
    lab._grid_samples = [_S(screen=(230, 316), camera=(279.6, 148.1), peak=71),
                         _S(screen=(230, 1053), camera=(275.0, 356.0), peak=103)]
    colbad = lab._out_of_grid_order(6, 277.0, 250.0)
    check("a column point that goes backwards is caught", bool(colbad),
          colbad or "NOT CAUGHT")
    lab._grid_samples = []
    lab._grid_points = []

    # Exposure writes being ignored: the failure that masqueraded as a badly
    # aimed camera for an entire session. A real sweep moves mean brightness
    # enormously; a camera stuck on auto barely moves at all.
    alert2 = lab.layers["alert"]
    alert2.clear()
    stuck = [(-3, 127, 161, 30), (-4, 127, 214, 30), (-5, 128, 173, 29),
             (-6, 130, 178, 30), (-7, 130, 181, 31), (-8, 130, 184, 30),
             (-9, 129, 181, 30), (-10, 136, 207, 31), (-11, 135, 215, 30)]
    lab._on_exposure_chosen(-4.0, stuck)
    check("a camera ignoring exposure is caught and reported",
          alert2.visible and "IGNORING EXPOSURE" in alert2.title, alert2.title)

    alert2.clear()
    working = [(-3, 31, 255, 22), (-4, 25, 255, 18), (-5, 22, 255, 14),
               (-6, 17, 255, 11), (-7, 17, 253, 9), (-8, 10, 224, 6),
               (-9, 5, 168, 4), (-10, 2, 71, 2), (-11, 1, 36, 1)]
    lab._on_exposure_chosen(-8.0, working)
    check("a healthy exposure sweep raises no such alarm",
          not (alert2.visible and "IGNORING EXPOSURE" in alert2.title), alert2.title)
    alert2.clear()

    # Calibration must relax the thresholds, and must put them back. Leaving
    # them relaxed would carry a calibration-only setting into play, where the
    # three-agreeing-shots protection no longer applies.
    lab._noise_peak = 25.0
    lab.sp_threshold.setValue(35)
    lab.sp_minpeak.setValue(40)
    lab._grid_saved_levels = None
    lab._relax_for_calibration()
    check("calibration lowers the detection thresholds",
          lab.sp_threshold.value() < 35 and lab.sp_minpeak.value() < 40,
          f"{lab.sp_threshold.value()}/{lab.sp_minpeak.value()}")
    check("but not below a sane floor",
          lab.sp_threshold.value() >= 6 and lab.sp_minpeak.value() >= 8,
          f"{lab.sp_threshold.value()}/{lab.sp_minpeak.value()}")
    lab._relax_for_calibration()        # must be idempotent
    check("relaxing twice does not compound",
          lab._grid_saved_levels == (35, 40), str(lab._grid_saved_levels))
    lab._restore_levels()
    check("and the originals come back afterwards",
          lab.sp_threshold.value() == 35 and lab.sp_minpeak.value() == 40,
          f"{lab.sp_threshold.value()}/{lab.sp_minpeak.value()}")

    # A sliding window: three real shots must be able to agree even when stray
    # detections arrived earlier. Without it, noise accumulated in the list
    # competes forever with the shots being fired now.
    old_strays = [(100.0 + i * 90, 500.0, 40.0) for i in range(8)]
    real = [(700.0, 300.0, 90.0), (706.0, 303.0, 88.0), (703.0, 297.0, 91.0)]
    check("strays alone never form a group",
          lab._agreeing_group(old_strays, 3) is None)
    windowed = (old_strays + real)[-lab._grid_window:]
    grp2 = lab._agreeing_group(windowed, 3)
    check("three real shots agree despite earlier strays",
          grp2 is not None and all(s in grp2 for s in real),
          str(grp2))
    check("the window is large enough to hold a trio plus noise",
          lab._grid_window >= 6, str(lab._grid_window))
    check("the shot budget is not so tight that noise exhausts it",
          lab._grid_max_shots >= 20, str(lab._grid_max_shots))

    # The signal margin report: the one number that predicts whether the next
    # session will work, which the app had every input for and never reported.
    alert3 = lab.layers["alert"]
    alert3.clear()
    lab._noise_peak = 10.0
    lab._report_signal_margin([180.0, 210.0, 165.0, 195.0])
    check("a healthy margin raises no alarm",
          not alert3.visible, alert3.title)

    alert3.clear()
    lab._noise_peak = 40.0
    lab._report_signal_margin([70.0, 95.0, 88.0])      # weakest 70 / 40 = 1.75x
    check("a thin margin is reported on screen",
          alert3.visible and "BARELY LOUDER" in alert3.title, alert3.title)
    check("the thin-margin warning clears itself", alert3.until > 0)
    alert3.clear()

    lab._report_signal_margin([])
    check("no peaks means no claim either way", not alert3.visible)

    # Every colour in the source, not just the ones that happen to get rendered.
    # The operator console lives on the SAME monitor the camera watches, and the
    # camera cannot tell it from the target screen. Measured live: the console's
    # warning banner used #f0a080 (red excess 80) and fired the gunshot sound by
    # itself while nobody was shooting.
    import re as _re
    DELIBERATE = {"#b00000"}          # the R-key worst-case background, on purpose
    bad = []
    for src in sorted(pathlib.Path("src").rglob("*.py")):
        body = src.read_text(encoding="utf-8")
        for line in body.splitlines():
            if line.lstrip().startswith("#"):
                continue              # prose, not a colour the app can draw
            for hexcol in _re.findall(r"#[0-9a-fA-F]{6}", line):
                if hexcol.lower() in DELIBERATE:
                    continue
                r = int(hexcol[1:3], 16); g = int(hexcol[3:5], 16); b = int(hexcol[5:7], 16)
                if r - max(g, b) > 30:
                    bad.append(f"{src.name}:{hexcol}({r - max(g, b)})")
    check("no colour anywhere in the app looks like a laser", not bad,
          ", ".join(bad[:6]) if bad else "all within red-excess 30")

    # A saved mapping measured on a different screen must be discarded, not
    # reused: it is silently wrong, which is the worst kind of wrong.
    lab.mapping = Mapping.fit(samples, (1920, 1080), use_residuals=False)
    lab._calibrate_if_needed()
    check("a mapping saved for another screen size is rejected", lab.mapping is None)

    pane: CameraPane = lab.layers["camera"]
    pane.set_frame(fake_camera_frame(), "index 1  1280x720")
    pane.visible = True
    lab.scene.diagnostics = {"text": (
        "fps 29.94   detect 1.85 ms\n"
        "frames with laser  9.1%\n"
        "frames/shot 3.00   pulse ~100.0 ms\n"
        "peak 212   shape 1.4:1\n"
        "dots at once 2   rejected 0"
    )}

    # ------------------------------------------------- the verdict board ----
    #
    # The app's answer to the only question the shooter asks. Tested here
    # rather than only in test_selfcheck, because that suite proves the
    # DECISION and this one proves the decision reaches the screen -- which
    # is the half that failed for this whole project: every number was
    # measured and every one of them was printed into an operator console
    # behind a fullscreen window on the same monitor.
    from lsa.selfcheck import SelfCheck, Verdict, assess

    board = lab.scene.get("verdict")
    check("the scene carries a verdict board", board is not None)

    # The afternoon of 2026-10-15: exposure forced to -8 by a bright room,
    # median shot 18 against a bar of 15, half the string lost.
    bad = SelfCheck(fired=12, detected=6, error_mm=3.0, peak_median=18,
                    noise_floor=9, min_peak=15, fps=25.0, exposure=-8.0,
                    exposure_spread=1.0, flash_difference=44.0,
                    audio_ok=True, room_bright=True)
    report = assess(bad)
    board.show_report(report, now=1000.0)
    check("a bad verdict puts the board on screen", board.visible)
    check("and it is STICKY -- a problem does not time out",
          board.until == 0.0)
    check("at most two findings reach the screen",
          len(board.findings) <= board.MAX_FINDINGS,
          f"{len(board.findings)} shown of {len(report.findings)}")
    check("the rest are counted honestly rather than dropped",
          board.more == max(0, len(report.findings) - board.MAX_FINDINGS),
          f"more={board.more}")
    check("every finding on screen names an action",
          all(a for _h, a, _m in board.findings),
          "a verdict without an action is the advice that trained him to "
          "ignore warnings")

    # NOTHING THE BOARD DRAWS MAY FALL OUTSIDE ITS OWN BOX.
    #
    # He saw this in a screenshot he sent on 2026-10-15 and I did not, and
    # it is the finding of the whole panel that was clipped -- the one line
    # that says what to do about the problem.
    #
    # The cause: the box was sized from `mono(line_px)`, the body font at
    # 21 regular, while the action is drawn in `mono(act_px, bold=True)` at
    # 23 bold. 62 characters of the second does not fit a box built for 62
    # of the first, and the "DO THIS: " prefix is nine more.
    #
    # So this renders the board and MEASURES every string it draws against
    # the box it drew it in, in the font it used. Checking that the text
    # "looks right" is what failed; measuring is one call and cannot be
    # wrong. Deliberately uses the longest real action string the selfcheck
    # module can produce, not a sample one.
    from PySide6.QtGui import QFontMetricsF                      # noqa: E402
    from lsa.display.layers import mono, px, _wrap               # noqa: E402

    # `board.findings` holds (headline, action, measured) tuples; a report's
    # `findings` holds Finding objects. Two shapes for the same idea, which
    # is its own small smell, but the point here is the longest ACTION
    # string the module can actually produce -- the one that overran.
    # Built here rather than reusing names from the surrounding scope: an
    # earlier `good` in this function is a style string, and feeding it to
    # `assess` raised AttributeError on a str. Local, explicit, and sweeping
    # the states that produce the WORDIEST actions -- a blind camera, a
    # thin margin, a noisy room -- because the longest line is the one that
    # overran the box.
    longest = max((a for _h, a, _m in board.findings), key=len, default="")
    for sc in (
        bad,
        SelfCheck(fired=12, detected=0, error_mm=None, peak_median=None,
                  noise_floor=90, min_peak=120, fps=9.0, exposure=-8.0,
                  exposure_spread=0.05, flash_difference=0.0, audio_ok=False,
                  room_bright=True),
        SelfCheck(fired=12, detected=10, error_mm=12.0, peak_median=25,
                  noise_floor=2, min_peak=10, fps=22.0, exposure=-4.0,
                  exposure_spread=1.0, flash_difference=44.0, audio_ok=True,
                  room_bright=False),
    ):
        for f in assess(sc).findings:
            if f.action and len(f.action) > len(longest):
                longest = f.action
    check(bool(longest), "there is an action string to measure")

    class _Ctx:
        now = 1000.0
        rect = QRect(0, 0, WIDTH, HEIGHT)
        width = WIDTH
        height = HEIGHT
        calibrating = False
        calibrated = True
        diagnostics: dict = {}

    cx = _Ctx()
    # The same arithmetic the layer does, so this measures the real box.
    act_px = px(cx, 23)
    line_px = px(cx, 21)
    pad = px(cx, 28)
    char_w = max(QFontMetricsF(mono(line_px)).horizontalAdvance("M"),
                 QFontMetricsF(mono(line_px, bold=True)).horizontalAdvance("M"))
    act_w = QFontMetricsF(mono(act_px, bold=True)).horizontalAdvance("M")
    width_box = min(cx.width * 0.86,
                    max(cx.width * 0.5, 62 * max(char_w, act_w) + pad * 2.4))
    usable = width_box - 2 * pad
    body_at = max(24, min(62, int(usable / max(1.0, char_w))))
    act_at = max(16, int(usable / max(1.0, act_w)) - 9)

    fm_act = QFontMetricsF(mono(act_px, bold=True))
    fm_body = QFontMetricsF(mono(line_px, bold=True))
    over = []
    for i, line in enumerate(_wrap(longest, act_at)):
        drawn = ("DO THIS: " if i == 0 else "         ") + line
        w = fm_act.horizontalAdvance(drawn)
        if w > usable:
            over.append(f"action line {i} is {w:.0f}px wide in {usable:.0f}px")
    for line in _wrap(longest, body_at):
        w = fm_body.horizontalAdvance(line)
        if w > usable:
            over.append(f"body line is {w:.0f}px wide in {usable:.0f}px")
    check("no line the board draws is wider than the box it is drawn in",
          not over,
          "; ".join(over[:3]) + "  -- the box is sized in one font and the "
          "action drawn in a larger, bolder one; he saw the clipped text "
          "before I did")

    # READY is worth a moment of his screen and no more.
    good = SelfCheck(fired=12, detected=12, error_mm=3.0, worst_error_mm=4.6,
                     peak_median=85, noise_floor=4, min_peak=10, fps=38.0,
                     exposure=-5.0, exposure_spread=1.0,
                     flash_difference=44.0, held_out_mm=0.9, audio_ok=True)
    board.show_report(assess(good), now=1000.0)
    check("READY times out", board.until > 1000.0)
    ctx = type("C", (), {"now": 1000.0 + board.READY_SECONDS + 0.1,
                         "rect": QRect(0, 0, WIDTH, HEIGHT)})()
    board.advance(ctx)
    check("and clears itself when it does", not board.visible)

    # Q recalls the last verdict from the firing line, and ANY key puts it
    # away again.
    #
    # CHANGED 2026-10-16, at his insistence and twice. It used to be sticky
    # for a problem verdict, on the reasoning that a verdict saying the
    # camera cannot see the screen must not be dismissable into the silence
    # this layer exists to end. That protects the app's right to be read
    # and ignores whose screen it is. Standing at the firing line, having
    # read it:
    #
    #   "So now you display this bright message on screen and you do not
    #    draw in any bullet holes."
    #   "The bright message does not disappear if I press X."
    #
    # It is a lit panel across the middle of the screen the camera watches,
    # over the place his holes land, and nothing he pressed would move it.
    # It still cannot be MISSED -- it appears, it is bright, it says what
    # to do. It may not refuse to leave.
    lab._last_report = report
    lab._on_target_key(Qt.Key_Q)
    check("Q recalls the last verdict", board.visible)
    board.on_key(Qt.Key_Escape)
    check("and any key puts it away, including a sticky one",
          not board.visible,
          "he had read it; it was over his holes and would not move")
    lab._on_target_key(Qt.Key_Q)
    check("Q brings it back after that", board.visible)
    # THROUGH THE SCENE, not straight to the app's handler.
    #
    # Layers get the key first (`window.on_key`), and the first version of
    # the dismiss-on-any-key change cleared the board inside the layer and
    # returned False -- so Q then reached `_on_target_key`, found it hidden
    # and showed it again. Q became the only key that could not put it
    # away, which is the complaint this change exists to fix. The check
    # above could not see it because it bypassed the layer.
    lab.scene.on_key(Qt.Key_Q)
    lab._on_target_key(Qt.Key_Q)
    check("and Q through the scene puts it away, like every other key",
          not board.visible,
          "the layer must not consume or pre-empt the toggle")
    lab._on_target_key(Qt.Key_Q)
    check("Q brings it back again", board.visible)
    consumed = board.on_key(Qt.Key_X)
    check("dismissing does not swallow the key",
          consumed is False and not board.visible,
          "the key he reaches for is the one the panel told him to press, "
          "so X must still re-measure the exposure while the panel goes")

    # THE WHOLE PATH, against a REAL capture object.
    #
    # The decision is covered by test_selfcheck and the board by the
    # checks above; what neither touches is the collection in between,
    # which reaches into the capture, the mapping, the mixer and four
    # widgets. On its first live run it read `capture.stats`, which does
    # not exist -- the capture keeps it as `_stats`, as every other
    # reader in that file already knew. It raised inside a Qt slot, so
    # the self test simply STOPPED after printing its last result line
    # and the arcade mode test that follows never ran. Nothing said so.
    #
    # These notes record the same trap once already: a noise-floor guard
    # read `stats.last_shot_at` through getattr, got 0.0, and never fired
    # once while looking exactly like protection. So this does not scan
    # the source for a plausible name -- it CALLS the thing.
    from lsa.camera.capture import CaptureConfig, CaptureThread, GrouperConfig
    from lsa.detect.laser import DetectorParams
    cap = CaptureThread(CaptureConfig(), DetectorParams(), GrouperConfig())
    lab.capture = cap
    try:
        c = lab._collect_selfcheck(res=[((0.0, 0.0), None)] * 12,
                                   seen=[((0.0, 0.0), ((0.0, 0.0), 80, 400, 90))] * 10,
                                   errs=[1.0, 2.0, 3.0], peaks=[70, 80, 90])
        collected = True
        why = ""
    except Exception as exc:                       # noqa: BLE001
        collected, why, c = False, repr(exc), None
    check("the verdict can be collected from a real capture object",
          collected, why)
    if c is not None:
        check("it reports the rate it was given", c.fired == 12 and c.detected == 10,
              f"{c.detected} of {c.fired}")
        check("and the too-short rejections, which killed 24% of his shots",
              c.rejected_short == cap._stats.rejected_runs,
              f"{c.rejected_short}")

    try:
        lab._publish_verdict(res=[((0.0, 0.0), None)] * 12,
                             seen=[((0.0, 0.0), ((0.0, 0.0), 80, 400, 90))] * 10,
                             errs=[1.0, 2.0, 3.0], peaks=[70, 80, 90])
        published, why = True, ""
    except Exception as exc:                       # noqa: BLE001
        published, why = False, repr(exc)
    check("and published all the way to the board", published, why)
    check("which remembers it for the Q key", lab._last_report is not None)

    # GUARD 1 ACROSS THE SEAM (RULES C14): the probe records its dot reading on
    # the lab, the collection hands it to the pure verdict, and the verdict is
    # spoken as a dud camera. Each half has its own test; this is the pair.
    from lsa.selfcheck import Verdict as _V
    _spoken = []
    _orig_say = lab.voice.say
    lab.voice.say = lambda key, *a, **k: _spoken.append(key)
    try:
        lab._probe_dot_reading = 5.0          # the 18:28 bad session
        c5 = lab._collect_selfcheck(res=[((0.0, 0.0), None)] * 12,
                                    seen=[((0.0, 0.0), ((0.0, 0.0), 80, 400, 90))] * 12,
                                    errs=[1.0, 2.0, 3.0], peaks=[70, 80, 90])
        check("the lab's recorded probe reading reaches the SelfCheck",
              c5.probe_dot == 5.0, f"{c5.probe_dot}")
        lab._publish_verdict(res=[((0.0, 0.0), None)] * 12,
                             seen=[((0.0, 0.0), ((0.0, 0.0), 80, 400, 90))] * 12,
                             errs=[1.0, 2.0, 3.0], peaks=[70, 80, 90])
        check("a probe dot of 5 publishes BLOCKED with 12 of 12 detected",
              lab._last_report.verdict is _V.BLOCKED,
              lab._last_report.summary)
        check("and it is SPOKEN as a bad camera, not a generic 'not ready'",
              _spoken == ["v_camera_bad"], f"{_spoken}")
        check("and a BLOCKED verdict DISARMS the screen watch (no healthy standard)",
              not lab.screenwatch.armed)
        # ...and a usable verdict ARMS it. assess() is stubbed to READY so the
        # question asked is only "does publishing a usable verdict arm it".
        import lsa.tools.camera_lab as _CL
        from lsa.selfcheck import Report as _Report
        _real_assess = _CL.assess
        _CL.assess = lambda cc: _Report(_V.READY, [], "READY - stub")
        try:
            lab._publish_verdict(res=[], seen=[], errs=[], peaks=[])
        finally:
            _CL.assess = _real_assess
        check("a usable (READY) verdict ARMS the passive screen watch",
              lab.screenwatch.armed)
    finally:
        lab._probe_dot_reading = None
        lab.voice.say = _orig_say

    # GUARD 2 ACROSS THE SEAM: the HUD tick feeds the passive screen watch from
    # the capture's sample, only while the screen is ours and nothing owns it,
    # and a sustained fall raises ONE alert.
    from lsa import screenwatch as _sw
    lab.screenwatch.arm()
    _alerts = []
    _orig_alert = lab._alert
    lab._alert = lambda title, *a, **k: _alerts.append(title)
    _orig_sees, _orig_busy = lab._camera_sees_our_screen, lab._screen_busy
    lab._camera_sees_our_screen = lambda: True
    lab._screen_busy = lambda *a: ""
    try:
        t = 100.0
        for lv in [200.0] * (_sw.REF_SAMPLES + 2) + [50.0] * 40:
            cap.screen_level, cap.screen_level_at = lv, t
            lab._check_screen_watch()
            t += 1.0 / _sw.SAMPLE_HZ
        check("a sustained fall in the capture's screen level raises exactly one alert",
              len(_alerts) == 1 and "DIMMER" in _alerts[0], f"{_alerts}")
        lab.screenwatch.arm()
        _alerts.clear()
        lab._screen_busy = lambda *a: "the exposure probe"
        t += 100.0
        for lv in [200.0] * (_sw.REF_SAMPLES + 2) + [50.0] * 40:
            cap.screen_level, cap.screen_level_at = lv, t
            lab._check_screen_watch()
            t += 1.0 / _sw.SAMPLE_HZ
        check("while a routine owns the screen the watch judges nothing",
              _alerts == [], f"{_alerts}")
    finally:
        lab._alert = _orig_alert
        lab._camera_sees_our_screen, lab._screen_busy = _orig_sees, _orig_busy
        lab.screenwatch.disarm()
    lab.capture = None

    # Left ON for the render below, in its worst-looking state: the
    # BLOCKED style is the brightest and has the thickest border, so if
    # any verdict can look like a laser to the camera, this one does.
    board.show_report(report, now=1000.0)

    print(f"rendering target screen to {out} ...")
    image = QImage(WIDTH, HEIGHT, QImage.Format_RGB32)
    image.fill(Qt.black)
    p = QPainter(image)
    p.setRenderHint(QPainter.Antialiasing)
    rect = QRect(0, 0, WIDTH, HEIGHT)
    lab.scene.advance(rect)
    lab.scene.paint(p, rect)
    p.end()
    ok_write = image.save(str(out))
    check("target screen rendered to file", ok_write and out.exists(),
          f"{out} ({out.stat().st_size // 1024} KB)" if out.exists() else "not written")

    # The blue-multiply trick is load-bearing, so assert it rather than trust it:
    # no pixel inside the camera pane may be red-dominant.
    box = pane._geometry(type("C", (), {
        "rect": rect, "width": WIDTH, "height": HEIGHT})(), 640 / 360)
    worst = 0
    for x in range(box.left() + 2, box.right() - 2, 7):
        for y in range(box.top() + 2, box.bottom() - 2, 7):
            c = image.pixelColor(x, y)
            worst = max(worst, c.red() - max(c.green(), c.blue()))
    check("camera pane emits no red-dominant pixels (laser-safe)", worst <= 0,
          f"worst red excess {worst}")

    # And the same test over the whole screen: nothing we draw may look like a laser.
    worst_all = 0
    for x in range(0, WIDTH, 11):
        for y in range(0, HEIGHT, 11):
            c = image.pixelColor(x, y)
            worst_all = max(worst_all, c.red() - max(c.green(), c.blue()))
    check("whole target screen has no strongly red pixels", worst_all <= 30,
          f"worst red excess {worst_all}")

    # ---------------- WHAT IS LIT COMPETES WITH THE SHOT --------------------
    #
    # "Check all those bright things you display on screen. That is just
    #  mind-blowing. You need to make sure that anything on screen competes
    #  with detecting the shot."
    #
    # Right, and sharper than the rule it replaces. The no-red rule guards
    # against the app drawing something the detector mistakes FOR a shot.
    # This guards against the app drawing something a shot has to out-shine.
    #
    # The whole exposure strategy is "the screen fades to a ghost, the 650nm
    # dot still saturates". A lit panel is not a ghost. Measured, against a
    # target background at luminance 10:
    #
    #     verdict board text   203   20.3x the background
    #     alert text           197   19.7x
    #     prompt text          190   19.0x
    #     key help             138   13.8x
    #
    # So the limit is on AREA: a few lines of text are a rounding error on a
    # 4K panel, a filled panel across the middle of it is not.
    # The state that matters is the one a shot can arrive in. A verdict
    # board is allowed to be bright: it is only sticky when the app has
    # just said DO NOT SHOOT, and it stands down for every measurement.
    # What must stay dark is the furniture that is there WHILE shooting --
    # the band, the key legend, the signal meter, the mode.
    LIT = 60          # luminance at which a pixel stops being a ghost
    LIT_FRACTION_MAX = 0.02
    lab.scene.get("verdict").clear()
    lab.layers["alert"].clear()
    # The camera pane is a large bright rectangle and is opt-in (key P);
    # this test turned it on earlier. Measure the SHIPPING state -- what
    # he is actually shooting at -- not the debug one. That the pane is
    # 84% lit when shown is worth knowing and is its own reason not to
    # leave it up while shooting.
    lab.layers["camera"].visible = False
    lab.layers["hits"].visible = False
    # ...and the DARK background, which is what he shoots against. The
    # key-collision walk above presses W/R/Y on every layer and leaves
    # the background on whichever it pressed last, so without this the
    # measurement is of a white screen and reads 89% lit.
    lab.layers["background"].on_key(Qt.Key_D)
    image = QImage(WIDTH, HEIGHT, QImage.Format_RGB32)
    image.fill(Qt.black)
    p2 = QPainter(image)
    p2.setRenderHint(QPainter.Antialiasing)
    lab.scene.advance(rect)
    lab.scene.paint(p2, rect)
    p2.end()
    lit = 0
    total = 0
    for x in range(0, WIDTH, 5):
        for y in range(0, HEIGHT, 5):
            c = image.pixelColor(x, y)
            total += 1
            if 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue() > LIT:
                lit += 1
    frac = lit / max(1, total)
    # WHAT THE DETECTOR WOULD SCORE ON THIS, pixel by pixel.
    #
    # The hex scan elsewhere in this file reads colours out of the source
    # and allows 30, the limit for STATIC furniture. The signal meter is
    # not static: its counters change, its bar decays and its warning line
    # appears, so the limit is 8 -- and it was drawing at red excess 24,
    # which the scan passed because 24 is under 30.
    #
    # Measured 2026-10-16 on the screenshot the app saved while he was
    # shooting: 504 pixels inside the meter above red excess 30, peaking at
    # 47, and his eight "phantom shots" mapped to screen points inside that
    # box. The app was shooting at its own meter, playing the gunshot sound
    # at him each time, and I diagnosed it as an object in the room.
    #
    # A rendered measurement catches what a source scan cannot: a legal
    # colour used by a layer that moves, and a fringe the rasteriser adds.
    worst_red = 0
    worst_at = (0, 0)
    for x in range(0, WIDTH, 3):
        for y in range(0, HEIGHT, 3):
            c = image.pixelColor(x, y)
            excess = c.red() - max(c.green(), c.blue())
            if excess > worst_red:
                worst_red, worst_at = excess, (x, y)
    check("nothing on the shooting screen looks like a laser to the detector",
          worst_red <= 8,
          f"red excess {worst_red} at {worst_at} -- the limit for anything "
          f"that moves or appears is 8 (RULES B1). Static furniture may go "
          f"to 30, and the meter is not static")

    # -- the ball modes draw an OUTLINE and nothing that can hide a hit -----
    #
    # 2026-10-16: "most of the time the shot did not register because the
    # ball was yellow". A fill under the dot raises GREEN and lowers the
    # detector's red-minus-green score (~55 of his median 85). And the
    # outline must be red excess ZERO, not the 8 allowed to things that move:
    # his saved sensitivity map amplifies the weakest region up to 4x.
    from lsa.modes.ball import EscalatingBall as _Ball1
    from lsa.modes.balls import BouncingBalls as _Balls
    BG = QColor(10, 10, 10)
    import dataclasses as _dc

    def _mctx(now: float):
        # playable=None: the scene's earlier mapping describes a larger
        # screen than this 1920x1080 image, which would put balls off it.
        return _dc.replace(lab.scene._context(now, 1 / 60.0, rect), playable=None)

    ctx_b = _mctx(0.0)
    for cls in (_Ball1, _Balls):
        mode = cls()
        mode.start(ctx_b)
        for i in range(5):
            mode.advance(_mctx((i + 1) / 60.0))
        img = QImage(WIDTH, HEIGHT, QImage.Format_RGB32)
        img.fill(BG)
        pb = QPainter(img)
        mode.paint(pb, _mctx(0.1))
        pb.end()
        worst_m = 0
        for x in range(0, WIDTH, 2):
            for y in range(0, HEIGHT, 2):
                c = img.pixelColor(x, y)
                worst_m = max(worst_m, c.red() - max(c.green(), c.blue()))
        check(f"{mode.name}: nothing it draws has any red excess",
              worst_m <= 0, f"worst red excess {worst_m}")
        balls_now = getattr(mode, "balls", None) or [mode]
        balls_now = getattr(mode, "balls", None) or [mode]
        inside = [img.pixelColor(int(b.x), int(b.y)) for b in balls_now]
        # THE BALLS ARE FILLED NOW, and these two checks asserted the opposite
        # -- that every pixel inside was the background. He asked for the fill
        # on 2026-10-18 ("so it looks a bit more like a ball, kind of"), and
        # "hollow" was never the property worth protecting. THIS is:
        #
        # The fill is what a dot LANDS ON. The detector scores
        # `red - max(green, blue)`; a hit's red saturates while the fill's
        # grey sits in green and blue underneath it, so every unit of fill
        # above the background comes straight off the shot's score. An earlier
        # fill put ~55 units under each hit and took his median shot from 85
        # to ~30 -- the target was hiding the shots that hit it.
        #
        # So the fill is allowed, and bounded: no more than 12 units above the
        # background, against a dimmest-ever logged shot of 31 and a bar of
        # 8-19. And below 60, the luminance at which a pixel starts spending
        # the lit-area budget, so a screen of balls costs nothing there.
        BALL_FILL_COST_MAX = 12
        lum = lambda c: 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        bg_lum = lum(BG)
        costs = [lum(c) - bg_lum for c in inside]
        check(f"{mode.name}: the fill costs a landing shot at most "
              f"{BALL_FILL_COST_MAX} units of its score",
              all(0.0 <= c <= BALL_FILL_COST_MAX for c in costs),
              f"{[round(c, 1) for c in costs]} units above the background")
        check(f"{mode.name}: ...and spends none of the lit-area budget",
              all(lum(c) < 60 for c in inside),
              f"{[round(lum(c)) for c in inside]} against the 60 threshold")
        # Still nothing BRIGHT inside: an interior pixel may be the fill or
        # the background and nothing else, so a stray light cannot hide in it.
        class _NoText(QPainter):
            def drawText(self, *a, **k):
                return None

        img2 = QImage(WIDTH, HEIGHT, QImage.Format_RGB32)
        img2.fill(BG)
        pb2 = _NoText(img2)
        mode.paint(pb2, _mctx(0.1))
        pb2.end()
        # THE BALLS CARRY A TARGET FACE NOW -- concentric rings, asked for on
        # 2026-10-18 ("Can you paint an actual target on a bouncing ball?").
        # So "nothing inside is brighter than the fill" has stopped being the
        # property: the rings ARE brighter than the fill, on purpose, and that
        # check would now forbid the thing he asked for.
        #
        # What still has to hold is the cost, which is what the old check was
        # really protecting, so it is stated as cost rather than as uniformity:
        #
        #   - NO PIXEL inside is brighter than the face level. A face is thin
        #     printed lines; a solid bright disc still fails here, which is the
        #     failure that mattered (a fill at 200 took his median shot from 85
        #     to ~30).
        #   - MOST OF THE BALL IS STILL BARE. A dot is 45-75 screen px across
        #     and lands somewhere inside; if the face covered the interior it
        #     would cost on every shot instead of on some.
        #   - and the AIM POINT is bare, which the centre-pixel check above
        #     measures directly. That one caught a filled bullseye at 33 units
        #     against a budget of 12.
        from lsa.artwork import ceiling as _ceiling
        from lsa.modes.base import Mode as _Mode
        # Derived from the SAME constants the painter uses, so a change to the
        # face brightness moves the check with it instead of leaving a number
        # behind that nobody notices has stopped meaning anything.
        FACE_LUM_MAX = round(_ceiling.screen_level() * _Mode.RING_FACE_FRAC) + 2
        FACE_AREA_MAX = 0.35        # of the ball's own area
        worst_lum, face_px, total_px = 0.0, 0, 0
        for b in balls_now:
            r_in = b.r * 0.9
            for x in range(max(0, int(b.x - r_in)), min(WIDTH, int(b.x + r_in) + 1)):
                for y in range(max(0, int(b.y - r_in)), min(HEIGHT, int(b.y + r_in) + 1)):
                    if (x - b.x) ** 2 + (y - b.y) ** 2 >= r_in * r_in:
                        continue
                    total_px += 1
                    v = lum(img2.pixelColor(x, y))
                    worst_lum = max(worst_lum, v)
                    if v > bg_lum + BALL_FILL_COST_MAX:
                        face_px += 1
        share = (face_px / total_px) if total_px else 0.0
        check(f"{mode.name}: nothing inside a ball is brighter than its "
              f"target face",
              worst_lum <= FACE_LUM_MAX,
              f"brightest interior pixel {worst_lum:.0f}, face level "
              f"{FACE_LUM_MAX}")
        check(f"{mode.name}: most of a ball is still bare, so a dot usually "
              f"lands on nothing",
              share <= FACE_AREA_MAX,
              f"{share * 100:.0f}% of the interior is face ink, "
              f"limit {FACE_AREA_MAX * 100:.0f}%")

    check("little enough of the target screen is lit to shoot over",
          frac <= LIT_FRACTION_MAX,
          f"{frac * 100:.1f}% of the screen is brighter than {LIT}, "
          f"limit {LIT_FRACTION_MAX * 100:.0f}% -- a lit panel is not a "
          f"ghost, and the dot has to out-shine whatever is under it")

    # -- the INFORMATIONAL layers, in their VISIBLE state, are red excess 0 ---
    #
    # 2026-10-17: "when I press zero, the app displays a few screen controls
    # I immediately hear a few fake shots". Key 0 shows stats, prompt band,
    # signal meter, key list, background label (and the uncalibrated minimap),
    # and the camera reads each as it APPEARS -- the fifth recorded instance of
    # the app shooting at its own furniture. The shooting-state check above
    # renders them HIDDEN, which is why it was green. The limit for anything
    # that appears is not "under 8": the saved sensitivity map multiplies the
    # weakest region up to 4x, so a legal 8 reaches the detector as 32 against
    # a bar of 14. It is ZERO, rendered, per pixel -- alone, composited with one
    # another, and with bullet holes in both z orders.
    import time as _time
    from lsa.display import layers as _L
    from lsa.display.scene import Scene as _Scene

    def _info_scene(holes_z=None):
        sc = _Scene()
        sc.add(_L.SolidBackground())
        lay = {}
        for nm, c in (("stats", _L.StatsHud()), ("help", _L.KeyHelp()),
                      ("prompt", _L.Prompt()), ("signal", _L.SignalMeter()),
                      ("hits", _L.CameraSpaceHits(visible=True)),
                      ("alert", _L.Alert()), ("verdict", _L.VerdictBoard()),
                      ("holes", _L.BulletHoles())):
            lay[nm] = sc.add(c)
        sc.add(_L.BackgroundLabel(_L.SolidBackground()))
        sc.diagnostics["text"] = "detect 7.2ms fps 55\nfloor 4 bar 8"
        sc.stats.shots, sc.stats.hits = 12, 7
        for i in range(30):
            lay["holes"].add(100 + i * 50, 100 + (i % 7) * 120, i, 0)
        if holes_z is not None:
            lay["holes"].z = holes_z
            sc._layers.sort(key=lambda la: la.z)
        return sc, lay

    def _worst_redx(sc):
        im = QImage(WIDTH, HEIGHT, QImage.Format_RGB32)
        im.fill(QColor("#0a0a0a"))
        pp = QPainter(im)
        pp.setRenderHint(QPainter.Antialiasing)
        sc.paint(pp, rect)
        pp.end()
        arr = np.frombuffer(im.constBits(), np.uint8).reshape(
            HEIGHT, im.bytesPerLine() // 4, 4)[:, :WIDTH].astype(int)
        ex = arr[..., 2] - np.maximum(arr[..., 1], arr[..., 0])
        yy, xx = np.unravel_index(int(ex.argmax()), ex.shape)
        return int(ex.max()), (int(xx), int(yy))

    # three whole-screen states, each with every informational layer up at
    # once, so a colour that only goes wrong in one KIND (the warn palette was
    # the yellow one) is still seen.
    for hz in (None, 90):
        for pk, nk, vk in (("warn", "warn", "blocked"), ("go", "good", "ready"),
                           ("wait", "info", "degraded")):
            sc_i, lay_i = _info_scene(hz)
            lay_i["prompt"].say("SHOOT THE RING", "detail", pk, "3 of 9")
            lay_i["signal"].report(120, 14, live=True, seen_ago=0.5, shot_ago=1.0,
                                   counts=(("rej", 3),), note="seeing the laser",
                                   note_kind=nk, now=_time.perf_counter())
            lay_i["alert"].show("ALERT", "some detail", pk if pk != "go" else "good",
                                dim=True)
            lay_i["verdict"].verdict, lay_i["verdict"].summary = vk, "11 of 12"
            lay_i["verdict"].findings = [("headline", "do this", "3.1x")]
            lay_i["verdict"].more, lay_i["verdict"].until = 0, 0.0
            lay_i["verdict"].visible = True
            for tid in range(5):
                lay_i["hits"].add(0.2 + tid * .1, 0.3, tid)
            w_e, a_e = _worst_redx(sc_i)
            check(f"every informational layer up together ({pk}/{nk}/{vk}"
                  f"{', holes over them' if hz else ''}): red excess is exactly 0",
                  w_e <= 0,
                  f"worst red excess {w_e} at {a_e}. A layer that APPEARS is read "
                  f"by the camera as a laser hit (RULES B1; 2026-10-17 'fake shots "
                  f"when I press 0'); the saved sensitivity map multiplies the "
                  f"weakest region up to 4x, so even 8 is not safe -- draw in true "
                  f"grey (R=G=B) or in blue/green, never with any red lead")

    for nm_i, cls_i in (("stats", _L.StatsHud), ("help", _L.KeyHelp),
                        ("prompt", _L.Prompt), ("signal", _L.SignalMeter),
                        ("bglabel", None)):
        sc_a = _Scene()
        sc_a.add(_L.SolidBackground())
        sc_a.add(_L.BackgroundLabel(_L.SolidBackground()) if cls_i is None else cls_i())
        sc_a.diagnostics["text"] = "detect 7.2ms fps 55"
        sc_a.stats.shots = 12
        w_a, at_a = _worst_redx(sc_a)
        check(f"layer '{nm_i}' on its own: red excess is exactly 0",
              w_a <= 0, f"worst red excess {w_a} at {at_a}")

    # ...and TEXT, through the platform a real window uses. This suite is
    # offscreen, and the offscreen platform draws no sub-pixel text: grey text
    # measured red excess 0 here and 49-86 on its glyph edges through the
    # native Windows platform (ClearType writes coloured fringes). That is what
    # made 'colours are fine' true on paper and false on his screen. So the
    # probe is a SUBPROCESS that loads the native platform exactly as the app
    # does (QT_QPA_PLATFORM unset -> lsa/_env.py decides) and paints every
    # piece of text into an image; no window opens. It goes red if the font
    # engine default in _env.py or the NoSubpixelAntialias in layers.mono() is
    # reverted -- verified by reverting _env.py: worst=86 at the stats text.
    import subprocess as _sp
    _env2 = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    probe = _sp.run([sys.executable, "-I", str(Path(__file__).resolve().parent
                                               / "native_text_probe.py")],
                    capture_output=True, text=True, timeout=120, env=_env2)
    line = next((ln for ln in probe.stdout.splitlines()
                 if ln.startswith("NATIVE_TEXT")), "")
    if os.name != "nt":
        check("native-platform text probe (Windows only)", True, "skipped off Windows")
    else:
        import re as _re
        mm = _re.search(r"platform=(\S+) worst=(\d+) at=(\d+),(\d+) layer=(\S+)", line)
        check("text through the NATIVE platform has red excess exactly 0",
              bool(mm) and int(mm.group(2)) == 0 and mm.group(1) == "windows",
              f"probe said {line!r} (stderr: {probe.stderr.strip()[-300:]!r}). "
              f"Sub-pixel (ClearType) glyph fringes are coloured; the camera "
              f"reads them as laser hits whenever text appears. lsa/_env.py "
              f"must select the GDI font engine and layers.mono() must set "
              f"NoSubpixelAntialias")

    # -- a camera is identified by what it DOES (RULES A11) ----------------
    #
    # 2026-10-16, and it cost him a morning. He moved the camera to another
    # USB port -- on my advice -- and Windows renumbered the bus. The app
    # remembers the device BY NAME; that name now belonged to the C920
    # sitting on top of the monitor with a blind over its lens. The app
    # opened it, measured a flash difference of 0.0 at every exposure,
    # correctly said "not pointed at this screen", and stopped at a black
    # screen with the right camera sitting unused one index away.
    #
    # My first fix was to write a different name and index into his
    # settings -- which is the SAME rule broken again, by hand. The app has
    # already measured the one property that identifies the right camera.
    class _Cam:
        def __init__(self, index, name, virtual=False):
            self.index, self.name, self.is_virtual = index, name, virtual
            self.label = name

    lab._blind_devices = set()
    lab._cameras = [_Cam(0, "EOS Webcam Utility", virtual=True),
                    _Cam(1, "HD Pro Webcam C920"),
                    _Cam(2, "USB Video Device")]
    lab.cmb_device.clear()
    for c in lab._cameras:
        lab.cmb_device.addItem(c.label, c.index)
    lab.cmb_device.setCurrentIndex(1)                 # the blinded C920
    nxt = lab._next_camera_to_try()
    check("a camera that cannot see the screen hands over to the next one",
          nxt is not None and nxt[0] == 2,
          f"{nxt} -- the app measured 'this device cannot see the screen' "
          f"and used it only to give up, while the right camera sat one "
          f"index away")
    lab.cmb_device.setCurrentIndex(2)
    nxt2 = lab._next_camera_to_try()
    check("and it stops once every real camera has been tried",
          nxt2 is None,
          f"{nxt2} -- each device is offered once per launch, or a machine "
          f"with two cameras cycles between them for ever (RULES D10)")
    lab._blind_devices = set()
    lab.cmb_device.setCurrentIndex(1)
    offered = []
    for _ in range(4):
        n = lab._next_camera_to_try()
        if n is None:
            break
        offered.append(n[0])
        pos = lab.cmb_device.findData(n[0])
        lab.cmb_device.setCurrentIndex(pos)
    check("and the virtual camera is never among the ones tried",
          0 not in offered,
          f"offered {offered} -- a virtual device opens fine and delivers a "
          f"picture of a disconnected-cable icon containing a saturated red "
          f"X, which is exactly what the detector hunts")

    # == a rejected calibration must be UNMISSABLE (2026-10-18) ==============
    #
    # Through the real paths, not by scanning the source. 10:04:39 on his
    # bench: pattern calibration was rejected, the alert ran for fourteen
    # seconds, and then nothing. No verdict was published; the standing prompt
    # band went on naming the arcade mode in its `go` state, which also keeps
    # it HIDDEN while shooting; and he fired 42 shots -- every one detected at
    # peaks 76-109 against a bar of 8, not one drawn. He concluded the app was
    # broken, and every channel it had agreed with him by saying nothing.
    print("")
    print("== uncalibrated: the app must say so unmissably ==")
    from lsa.detect.laser import Shot as _UShot                        # noqa: E402
    from lsa.display.scene import RenderContext as _URC                # noqa: E402
    from lsa.display.session import SessionStats as _USS               # noqa: E402
    from lsa.selfcheck import Verdict as _UV                           # noqa: E402
    from lsa.calibrate.mapping import Mapping as _UMap, Sample as _USa  # noqa: E402

    _mode_host = lab.layers.get("modes")
    _saved_mapping = lab.mapping
    lab.mapping = None
    lab._calibrating = False
    lab._grid_active = False
    lab._pat_active = False
    _uboard = lab.scene.get("verdict")
    if _uboard is not None:
        _uboard.clear()
    lab._last_report = None

    # HIS SCENARIO HAD A WORKING CAMERA, and this section did not.
    #
    # `_prompt_state` answers "CAMERA NOT CONNECTED" first, and rightly so:
    # with no camera that IS the problem. But `lab.capture` is left at None
    # earlier in this file, so these checks were asking about a state he was
    # never in, and the fix they would have driven is the wrong one.
    #
    # On 2026-10-18 at 10:04 his camera was connected and working -- it saw
    # all 42 of his shots at 9-13x the bar -- and the MAPPING had been
    # thrown away. That is the state under test. The capture is constructed
    # and never opened, exactly as at the top of this file, so no device is
    # touched.
    from lsa.camera.capture import (CaptureConfig as _UCC,                # noqa: E402
                                    CaptureThread as _UCT,
                                    GrouperConfig as _UGC)
    from lsa.detect.laser import DetectorParams as _UDP                   # noqa: E402
    _usaved_cap = lab.capture
    lab.capture = _UCT(_UCC(), _UDP(), _UGC())
    assert lab.capture._cap is None, "the test must never open a real camera"
    # ...and the camera can SEE the screen. "Something is in front of this
    # screen" is a different fault with different advice, and it outranks
    # this one correctly. His panel was in full view; the mapping was gone.
    _usaw = lab._camera_sees_our_screen
    lab._camera_sees_our_screen = lambda: True

    # 1. THE PROMPT BAND, with a mode running -- the state he was actually in.
    _head, _detail, _kind, _counter = lab._prompt_state()
    _both = (_head + " " + _detail).lower()
    check("uncalibrated: the prompt band says there is no calibration",
          "not calibrated" in _head.lower(), repr(_head))
    check("  ... in its warn state, so it returns on a clean screen",
          _kind == "warn", repr(_kind))
    check("  ... and names a key to press",
          "press c" in _both or "press g" in _both or " or g " in _both,
          repr(_head) + " / " + repr(_detail))
    check("  ... and does NOT promise holes from the screen outline",
          "rough calibration" not in _both,
          "that fallback was deleted on 2026-10-14 and _map_to_screen "
          "returns None; the text outlived it by four days")
    check("  ... and an active arcade mode does not override it",
          getattr(_mode_host, "mode", None) is None
          or "not calibrated" in _head.lower(),
          repr(_head) + " with mode " + repr(getattr(_mode_host, "mode", None))
          + " -- the mode branch used to win and say 'Shoot the targets.'")
    lab._info_full = False
    lab._update_prompt()
    lab._apply_info_policy()
    check("  ... and the band is actually VISIBLE in that state",
          lab.layers["prompt"].visible,
          "a warn band that stays hidden is the whole fault again")

    # 2. A DETECTION WITH NO MAPPING IS NOT SILENT. `_signal_note` writes to
    #    the signal meter, which is in INFO_LAYERS and hidden by default while
    #    shooting, so on 2026-10-18 the note was emitted 42 times onto a layer
    #    nobody could see and nothing at all reached the log.
    _ulines = []
    _ureal_log = lab._log

    def _ulog(m):
        _ulines.append(str(m))
        _ureal_log(m)

    lab._log = _ulog
    lab._unmapped_shots = 0
    lab._said_once.discard("unmapped-shot")
    _ualert = lab.layers.get("alert")
    if _ualert is not None:
        _ualert.visible = False
    _ut0 = time.perf_counter()
    lab._on_shot(_UShot(x=300.0, y=200.0, nx=0.5, ny=0.5, frames=2,
                        duration_ms=55.0, peak=95, area=300.0,
                        first_timestamp=_ut0, last_timestamp=_ut0 + 0.02,
                        index=901, track_id=11, elongation=1.0))
    lab._log = _ureal_log
    _usaid = " ".join(_ulines).lower()
    check("an unmapped shot is reported in the LOG, not only on a hidden layer",
          "no camera-to-screen mapping" in _usaid, _usaid[-180:])
    check("  ... and the log says the detector IS working",
          "detector is working" in _usaid, _usaid[-180:])
    check("  ... and names a key", "press c" in _usaid, _usaid[-180:])
    check("  ... and it interrupts: the alert layer is up",
          _ualert is not None and _ualert.visible,
          "the signal meter is hidden while shooting, so the note he was "
          "being sent 42 times was on a layer that was not on screen")
    check("  ... and the shot was counted, not dropped",
          lab._unmapped_shots == 1, str(lab._unmapped_shots))

    # 3. THE REJECTION PATH PUBLISHES A VERDICT. Driven with a genuinely
    #    collapsed fit (median 386mm, worst 1143mm -- measured 2026-10-14
    #    20:06), because his 2026-10-18 set is now correctly ACCEPTED; that
    #    half is in test_feedback 19d-2 with his real numbers.
    lab._last_report = None
    if _uboard is not None:
        _uboard.clear()
    _ulines.clear()
    lab._log = _ulog
    import numpy as _unp                                              # noqa: E402
    _ubad = _UMap(homography=_unp.eye(3, dtype=float),
                  screen_size=(3840, 2160),
                  samples=[_USa(screen=(x * 400.0, 400.0),
                                camera=(x * 10.0, 100.0), peak=255.0)
                           for x in range(1, 7)])
    lab._pat_reject(_ubad, 386.0 / 0.2448, 1143.0 / 0.2448, 0.2448,
                    "the MEDIAN held-out error is 386mm")
    lab._log = _ureal_log
    _urep = lab._last_report
    check("a rejected calibration PUBLISHES a verdict", _urep is not None,
          "no verdict was published at all on 2026-10-18 -- the identical "
          "fault as 'the verdict was not published when nothing was "
          "detected' (2026-10-15), in a second path")
    check("  ... and the verdict is BLOCKED, never READY",
          _urep is not None and _urep.verdict is _UV.BLOCKED,
          getattr(_urep, "summary", "<none>"))
    _uw = " ".join(_urep.lines()).lower() if _urep is not None else ""
    check("  ... and it NAMES the problem: there is nowhere to draw a hole",
          "no calibration" in _uw or "no hole can be drawn" in _uw, _uw[:140])
    check("  ... and names the key to press",
          "press k" in _uw or "press g" in _uw, _uw[:160])
    check("  ... and says nothing about dimming the room (RULES C4)",
          not any(w in _uw for w in ("dim the room", "dimming the room",
                                     "close a blind", "draw a blind",
                                     "darken the room")),
          _uw[:140])
    check("  ... and the board is on screen, where he is looking",
          _uboard is not None and _uboard.visible,
          "the log is behind a fullscreen window on the same monitor")
    check("  ... and the rejection names its reason in the log",
          "386mm" in " ".join(_ulines), " ".join(_ulines)[:200])
    # And the board must not itself look like a laser: it APPEARS, so RULES
    # B1 caps its red excess at 8. At 24 it once took the noise floor from
    # 0-2 to 16-18 and pushed the bar above the app's own test dots.
    _ubimg = QImage(1920, 1080, QImage.Format_RGB888)
    _ubimg.fill(QColor("#000000"))
    _ubp = QPainter(_ubimg)
    _uboard.paint(_ubp, _URC(now=time.perf_counter(), dt=0.016,
                            rect=QRect(0, 0, 1920, 1080), stats=_USS()))
    _ubp.end()
    _ubuf = np.frombuffer(_ubimg.constBits(), np.uint8).reshape(
        _ubimg.height(), _ubimg.bytesPerLine())
    _uarr = _ubuf[:, :1920 * 3].reshape(1080, 1920, 3).astype(np.int16)
    _uredx = int((_uarr[..., 0] - np.maximum(_uarr[..., 1],
                                            _uarr[..., 2])).max())
    check("the board for a rejected calibration keeps red excess <= 8",
          _uredx <= 8, "peak red excess " + str(_uredx))
    if _uboard is not None:
        _uboard.clear()
    lab.mapping = _saved_mapping
    lab._unmapped_shots = 0

    bg: SolidBackground = lab.layers["background"]
    bg.on_key(Qt.Key_W)
    check("background switches to white for worst-case testing",
          bg.label.startswith("WHITE") and not bg.is_dark)
    bg.on_key(Qt.Key_D)

    # ------------------------------------------------------------------
    # TWO PLAYERS, THROUGH THE REAL APP (slice 3). RULES C14: the host's own
    # checks ask the host what it recorded; these ask the APP what it did with
    # it -- its session counters, and the line the verdict publishes after the
    # microphone path has run.
    # ------------------------------------------------------------------
    print("\nTwo players through the real app: session counters and the microphone line")
    from lsa.detect.laser import Shot as _Shot3
    from lsa.audio.listener import Click as _Click3
    from lsa.settings import DEFAULT_MAGAZINE as _N3
    host3 = lab.layers["modes"]
    _opened3 = lab.target_window is None
    if _opened3:                         # the suite closed it above; shots need one
        lab._toggle_target_mode()
        app.processEvents()
    _ctx3 = lab.scene._context(time.perf_counter(), 0.0, lab.target_window.rect())
    # The smoke window is not 4K, so the mapping is fitted for THIS window the
    # same way the grid test above fits one for 3840x2160.
    _W3, _H3 = _ctx3.width, _ctx3.height
    _quad3 = _np.float32([[137, 201], [796, 234], [864, 580], [197, 719]])
    _Hw3 = _cv2.getPerspectiveTransform(
        _np.float32([[0, 0], [_W3, 0], [_W3, _H3], [0, _H3]]), _quad3)

    def _s2c3(sx_, sy_):
        q_ = _cv2.perspectiveTransform(_np.float32([[[sx_, sy_]]]), _Hw3)[0][0]
        return float(q_[0]), float(q_[1])

    _smp3 = [Sample((sx_, sy_), _s2c3(sx_, sy_)) for sx_, sy_ in grid_points(_W3, _H3, 3, 3)]
    mapping3, _note3 = Mapping.fit_validated(_smp3, (_W3, _H3))
    check("(precondition) a mapping was fitted for the smoke window", mapping3 is not None, str(_note3))
    if mapping3 is not None:
        _map_was3 = lab.mapping
        _idx3 = [100]
        lab.mapping = mapping3
        try:
            host3.set_mode(_arcade.StaticTargets(), _ctx3, players=2)
            lab.session.reset()
            lab.layers["holes"].clear()

            def _pull3(sx_, sy_):
                cx_, cy_ = _s2c3(sx_, sy_)
                _idx3[0] += 1
                lab._on_shot(_Shot3(x=cx_, y=cy_, nx=cx_ / 1280.0, ny=cy_ / 720.0,
                                    frames=3, duration_ms=85.0, peak=180, area=120.0,
                                    first_timestamp=float(_idx3[0]), last_timestamp=_idx3[0] + 0.085,
                                    index=_idx3[0], track_id=0, elongation=1.2))

            # SET the magazine this fixture asserts on; do not inherit it.
            #
            # It used to fire `DEFAULT_MAGAZINE` rounds and expect that many to
            # register, which quietly assumed the code default and the LIVE
            # value were the same number. On 2026-10-20 they stopped being: he
            # moved the default to 19, and `owned_files.isolate_outputs()` was
            # fixed the same evening so the harness finally copies his real
            # settings.local.json -- which says 17. The fixture then fired 19
            # into a 17-round magazine, two became dry pulls, and four checks
            # went red while nothing was wrong with the app.
            #
            # A check whose outcome depends on his settings file is pinned to
            # the state of the world. The property here is "a full magazine
            # registers and the next pull is a dry pull", for ANY magazine, so
            # the fixture owns the number.
            host3.set_magazine(_N3)
            _aim3 = host3.mode.aim_point(host3._side_ctx(_ctx3, 0))
            for _ in range(_N3):
                _pull3(*_aim3)
            check(f"a full magazine of {_N3} registers on the left target: the session counts "
                  f"{_N3} shots, {_N3} hits",
                  (lab.session.shots, lab.session.hits, lab.session.misses) == (_N3, _N3, 0),
                  f"{lab.session.shots} shots, {lab.session.hits} hits, {lab.session.misses} misses")
            _pull3(*_aim3)                     # one past the magazine: the side is done
            check("the host really did refuse it (the precondition for the next check)",
                  host3.tallies[0].after_done == 1 and host3.tallies[0].shots == _N3)
            check("a pull the round did not want is a DRY PULL, not a shot: the session's shots stay at the rounds fired "
                  "and the dry pull has its own counter (decided 2026-10-20: shots == hits + misses + unplaced)",
                  (lab.session.shots, lab.session.hits, lab.session.misses, lab.session.dry_pulls)
                  == (_N3, _N3, 0, 1),
                  f"{lab.session.shots} shots, {lab.session.hits} hits, {lab.session.misses} misses, "
                  f"{lab.session.dry_pulls} dry")
            check("...so the session's own numbers agree with each other, and with the host's tallies "
                  "(shots == the rounds fired, dry pulls == after_done)",
                  lab.session.consistent() and lab.session.shots == sum(t_.shots for t_ in host3.tallies)
                  and lab.session.dry_pulls == sum(t_.after_done for t_ in host3.tallies)
                  == host3.dry_camera)
            check(f"...so the hit rate's denominator IS the shot count printed beside it: {_N3} of {_N3} = 100%",
                  lab.session.hit_rate == 1.0 and lab.session.hits + lab.session.misses == lab.session.shots,
                  f"{lab.session.hit_rate}")
            _pull3(_W3 * 0.98, _H3 * 0.70)     # a genuine miss on the right
            check("...while a genuine miss on the other side still counts as a miss",
                  lab.session.misses == 1 and host3.tallies[1].shots == 1 and lab.session.consistent(),
                  f"{lab.session.misses} misses")

            # THE MICROPHONE, through `_poll_audio`: a bang heard that the camera
            # never saw. The listener is stood in for (this suite never opens a
            # device); everything from `take_clicks` onward is the real app path.
            for _ in range(_N3 - 1):
                _pull3(_W3 * 0.98, _H3 * 0.70)   # the right side empties its magazine too
            check("(precondition) the round is over and the verdict is up",
                  host3.round_over and host3.verdict is not None)
            # The microphone line left the verdict block 2026-10-19 and the SCREEN
            # altogether 2026-10-22 (his ruling: no need for it). The words are still
            # built and the log always carries them; `mic_screen_text` / `mic_log_text`
            # are what is read, as the seam between the app's count and the host.
            _words_before = host3.mic_screen_text()
            _ls3 = lab.listener
            _stream_was, _learned_was, _take_was = _ls3._stream, list(_ls3._learned), _ls3.take_clicks
            try:
                _ls3._stream = object()                          # "is running"
                _ls3._learned = []                               # NOT calibrated
                _ls3.take_clicks = lambda: [_Click3(t=time.perf_counter() - 2.0,
                                                    level=0.3, floor=0.01)]
                lab.corroborator.reset()
                lab._poll_audio()
                check("an UNCALIBRATED microphone: the app tells the host so, and the host counts nothing",
                      host3.mic_ready is False and host3.heard_unseen == 0)
                check("...and the screen stays SILENT while the log says 'not calibrated' (no zero printed)",
                      host3.mic_screen_text() == "" and "not calibrated" in host3.mic_log_text()
                      and "did not register" not in host3.mic_log_text())
                _ls3._learned = [0.1] * _ls3.LEARN_NEEDED        # calibrated
                _ls3.take_clicks = lambda: [_Click3(t=time.perf_counter() - 2.0,
                                                    level=0.3, floor=0.01)]
                lab.corroborator.reset()
                lab._poll_audio()
                check("a CALIBRATED microphone that heard a bang the camera never saw: the app counted it",
                      host3.mic_ready is True and host3.heard_unseen == 1,
                      f"mic_ready {host3.mic_ready}, heard_unseen {host3.heard_unseen}")
                _words_after = host3.mic_screen_text()
                check("...and the published microphone line CHANGED to say so",
                      _words_after != _words_before
                      and "1 trigger pull heard that the camera did not register" in _words_after,
                      _words_after[-160:])
                check("...and no shot was invented: the totals and tallies are exactly as they were",
                      sum(t_.shots for t_ in host3.tallies) == 2 * _N3
                      and host3.totals[0].shots == _N3 and host3.totals[1].shots == _N3)
            finally:
                _ls3._stream, _ls3._learned, _ls3.take_clicks = _stream_was, _learned_was, _take_was
                lab.corroborator.reset()

            # ----------------------------------------------------------------
            # THE MAGAZINE, SINGLE PLAYER, THROUGH THE REAL APP (2026-10-20).
            # His ruling: when the magazine is spent "they can continue pulling
            # the trigger but there are no sounds of shots and there are no
            # holes. Because the round is over." The host's own checks ask the
            # host what it returned; these ask the APP what it DID with it: the
            # recorded sound queue, the holes layer rendered to pixels, the
            # session's counters and the log line (RULES C14: the seam).
            # ----------------------------------------------------------------
            print("\nThe magazine through the real app: a dry pull is silent, leaves no hole, and is still counted")
            _MAG3 = 4
            host3.set_magazine(_MAG3)
            host3.set_mode(_arcade.StaticTargets(), _ctx3, players=1)
            lab.session.reset()
            lab.layers["holes"].clear()
            _holes3, _mx3 = lab.layers["holes"], lab.mixer
            _log3: list = []
            _real_log3 = lab._log
            lab._log = lambda m_: (_log3.append(m_), _real_log3(m_))[1]

            def _holes_px3():
                img_ = QImage(_ctx3.width, _ctx3.height, QImage.Format_ARGB32_Premultiplied)
                img_.fill(0)
                pn_ = QPainter(img_)
                _holes3.paint(pn_, lab.scene._context(time.perf_counter(), 0.0, lab.target_window.rect()))
                pn_.end()
                a_ = _np.frombuffer(img_.constBits(), _np.uint8).reshape(
                    _ctx3.height, img_.bytesPerLine())[:, :_ctx3.width * 4].reshape(_ctx3.height, _ctx3.width, 4)
                return int((a_[..., 3] > 0).sum())

            try:
                _aim3b = host3.mode.aim_point(_ctx3)
                check("the strip's count is the magazine configured, not a literal",
                      host3.remaining(0) == _MAG3 and host3._round_mag == _MAG3)
                _pl0, _ha0 = len(_mx3.played), _holes3.added
                _pull3(*_aim3b)
                check("a shot BEFORE the magazine is empty still bangs and still leaves its hole (unchanged)",
                      len(_mx3.played) == _pl0 + 1 and _holes3.added == _ha0 + 1
                      and _holes_px3() > 0 and host3.remaining(0) == _MAG3 - 1,
                      f"played +{len(_mx3.played) - _pl0}, holes +{_holes3.added - _ha0}")
                for _ in range(_MAG3 - 1):
                    _pull3(*_aim3b)
                check("(precondition) the Nth registered shot emptied the magazine and ended the round",
                      host3.round_over and host3.remaining(0) == 0 and host3.tallies[0].shots == _MAG3)
                lab._refresh_target_diagnostics()          # the app's tick clears the holes
                check("(precondition) the end screen's holes were cleared, so what follows is clean ground",
                      _holes_px3() == 0)
                _pl1, _ha1, _sh1 = len(_mx3.played), _holes3.added, lab.session.shots
                _ad1 = host3.tallies[0].after_done
                _log3.clear()
                lab.corroborator.reset()
                _pull3(*_aim3b)
                from lsa.audio.mixer import DRY_SOUND as _DRY3, SELECT_SOUND as _SEL3
                check("a pull on the ROUND-OVER screen with an empty magazine plays the bubble pop INSTEAD of the gunshot, "
                      "and its hole is drawn",
                      len(_mx3.played) == _pl1 + 1 and _mx3.played[-1][0] == _DRY3 and _holes3.added == _ha1 + 1
                      and _holes_px3() > 0,
                      f"sounds {[n_ for n_, _g in _mx3.played[_pl1:]]}, holes +{_holes3.added - _ha1}")
                check("...but it is still a DRY pull: counted in after_done and the session's dry counter, NOT in its shots",
                      host3.tallies[0].after_done == _ad1 + 1 and lab.session.shots == _sh1
                      and lab.session.dry_pulls == 1 and lab.session.consistent())
                _shot_ln3 = [l_ for l_ in _log3 if l_.startswith("SHOT")]
                check("...and it is in the LOG with its position and peak, marked as a dry pull",
                      len(_shot_ln3) == 1 and "cam=(" in _shot_ln3[0] and "peak=180" in _shot_ln3[0]
                      and "DRY" in _shot_ln3[0], str(_shot_ln3))
                lab._refresh_target_diagnostics()
                check("the arrival clear happened ONCE: a tick after a shot on the results screen does not wipe the new hole",
                      _holes_px3() > 0)

                # ACROSS THE SEAM (RULES C14): the app hands the host the shot's own timestamp
                # (`t_end`) so the corroborator's "both" can be recognised as ONE of our dry
                # pulls. Asked of the real `_on_shot`, not of a ShotEvent built by hand.
                lab.corroborator.add_clicks([_Click3(t=_idx3[0] + 0.085 - 0.004, level=0.3, floor=0.01)])
                host3.mic_ready = True
                host3.note_pull_outcomes(lab.corroborator.take_outcomes())
                check("the dry pull the APP detected pairs with its click: counted ONCE, corroborated (the union, through the real shot path)",
                      host3.dry_camera == 1 and host3.dry_corroborated == 1
                      and host3.dry_camera + host3.dry_mic_only == 1,
                      f"camera {host3.dry_camera}, corroborated {host3.dry_corroborated}, mic-only {host3.dry_mic_only}")
                # The START box, unarmed (it has not been painted, so it is not armed).
                _bx3 = host3.round_box(_ctx3).center()
                _pl2, _ha2 = len(_mx3.played), _holes3.added
                _pull3(float(_bx3.x()), float(_bx3.y()))
                check("a shot at the NEXT ROUND box before it arms plays the pop and leaves its hole, with NO cue, and nothing "
                      "starts (he is told the app saw it; the round deliberately does not begin yet)",
                      len(_mx3.played) == _pl2 + 1 and _mx3.played[-1][0] == _DRY3 and _holes3.added == _ha2 + 1
                      and host3.round_over and not host3.take_next_round_request())
                # A DRY PULL IS NEVER LOGGED AS A PHANTOM (2026-10-20). A detection while a
                # magazine is empty is a dry pull by definition. Driven through the real
                # `_poll_audio`, both ways: the same seen-but-not-heard detection on a
                # LIVE magazine must still read as a possible phantom exactly as before.
                def _snh3():
                    """One seen-but-not-heard pull through the real `_poll_audio`; returns
                    the log lines it wrote."""
                    _t3 = _idx3[0] + 0.085
                    keep = (_ls3._stream, list(_ls3._learned), _ls3.take_clicks, list(_ls3._recent))
                    try:
                        _ls3._stream = object()
                        _ls3._learned = [0.1] * _ls3.LEARN_NEEDED
                        _ls3.take_clicks = lambda: []
                        _ls3._recent.append((_t3, 0.001))      # the stream WAS delivering: a quiet room
                        lab.corroborator.reset()
                        lab.corroborator.add_shot(_t3)
                        _log3.clear()
                        lab._poll_audio()
                        return list(_log3)
                    finally:
                        _ls3._stream, _ls3._learned, _ls3.take_clicks = keep[0], keep[1], keep[2]
                        _ls3._recent.clear()
                        _ls3._recent.extend(keep[3])

                _pull3(*_aim3b)                                          # a dry detection
                _dry_ln3 = [l_ for l_ in _snh3() if l_.startswith("SAW a")]
                check("a seen-but-not-heard detection on an EMPTY magazine with a LISTENING microphone is never worded as "
                      "the live phantom, and says plainly that it is NOT CONFIRMED (the 12:22 log: thirty detections at "
                      "one column, mic 0.041, were counted as dry pulls)",
                      len(_dry_ln3) == 1 and "WAS listening" in _dry_ln3[0] and "not confirmed" in _dry_ln3[0]
                      and "likely a phantom" not in _dry_ln3[0] and "NOT a phantom" not in _dry_ln3[0], str(_dry_ln3))
                check("...and the note for the target screen does not tell him to treat it as a phantom either",
                      "phantom" not in lab._audio_note and "NOT heard" in lab._audio_note, lab._audio_note)
                # A DEAF microphone (muted by the app's own last bang) DOES explain a missing click.
                _pull3(*_aim3b)
                _tdeaf3 = _idx3[0] + 0.085
                _ls3._mutes.append((_tdeaf3 - 0.5, _tdeaf3 + 0.9))
                _deaf_ln3 = [l_ for l_ in _snh3() if l_.startswith("SAW a")]
                check("a dry detection while the microphone was DEAF reads as a dry pull it could not hear (the explanation "
                      "that holds), still never as the live phantom",
                      len(_deaf_ln3) == 1 and "dry pull the microphone could not hear" in _deaf_ln3[0]
                      and "DEAF" in _deaf_ln3[0] and "likely a phantom" not in _deaf_ln3[0], str(_deaf_ln3))
                _ls3._mutes.clear()
                _box3 = host3.round_box(_ctx3).center()
                _pull3(float(_box3.x()), float(_box3.y()))               # a shot AT the START box (unarmed)
                _bx_ln3 = [l_ for l_ in _snh3() if l_.startswith("SAW a")]
                check("a pull at the NEXT ROUND box is a counted pull with an empty magazine too, and is never worded as the live phantom",
                      len(_bx_ln3) == 1 and "likely a phantom" not in _bx_ln3[0] and "empty magazine" in _bx_ln3[0],
                      str(_bx_ln3))
                check("...and it counted in `after_done` AND the dry tally: the two numbers agree with their names",
                      host3.dry_camera >= 4 and host3.tallies[0].after_done == host3.dry_camera
                      and lab.session.dry_pulls == host3.dry_camera and lab.session.consistent(),
                      f"{host3.tallies[0].after_done} vs {host3.dry_camera}")

                # THE WATCHDOG MUST NOT HOLD A DRY PULL TO ACCOUNT. With the app silent
                # the microphone is awake, so it now hears dry pulls; "a bang was heard
                # and nothing appeared" is the watchdog's fault signal and is true of
                # every dry pull BY DESIGN. Driven through the real `_watch_feed`.
                _wd_calls3: list = []
                _wd_orig3 = (lab.watchdog.note_click, lab._floor_measured,
                             lab._camera_sees_our_screen, lab._app_is_drawing_now,
                             list(_ls3._learned))
                try:
                    lab.watchdog.note_click = lambda t_, counted: _wd_calls3.append(counted)
                    lab._floor_measured = True
                    lab._camera_sees_our_screen = lambda: True
                    lab._app_is_drawing_now = lambda: False
                    _ls3._learned = [0.1] * _ls3.LEARN_NEEDED            # calibrated
                    lab._watch_feed([_Click3(t=time.perf_counter(), level=0.3, floor=0.01)],
                                    time.perf_counter())
                    check("a click heard while the magazine is empty IS held to account by the mark watchdog, exactly like a "
                          "live one: a dry pull draws its hole now (2026-10-20), so 'a bang heard and nothing drawn' is a "
                          "fault there too",
                          _wd_calls3 == [True], str(_wd_calls3))
                    host3.restart(_ctx3)
                    _wd_calls3.clear()
                    lab._watch_feed([_Click3(t=time.perf_counter(), level=0.3, floor=0.01)],
                                    time.perf_counter())
                    check("...and the same click in a live round is held to account as before",
                          _wd_calls3 == [True], str(_wd_calls3))
                    for _ in range(_MAG3):
                        _pull3(*_aim3b)                                  # empty it again for what follows
                finally:
                    (lab.watchdog.note_click, lab._floor_measured, lab._camera_sees_our_screen,
                     lab._app_is_drawing_now, _ls3._learned) = _wd_orig3
                # THE SELECTION CUE, through the real app: the sounds the mixer was ASKED for, in order, after the
                # one-millisecond timer has fired. A shot plays its sound THEN the cue; a click plays the cue alone.
                import time as _tm4
                host3.restart(_ctx3)
                lab.session.reset()
                _btn3 = host3.menu_button.rect(_ctx3).center()
                _gun3 = lab.cmb_sound.currentText() or "synth shot"
                _tm4.sleep(0.03)
                app.processEvents()               # let every earlier effect timer fire first
                _pl4 = len(_mx3.played)
                _pull3(float(_btn3.x()), float(_btn3.y()))
                _tm4.sleep(0.02)
                app.processEvents()
                _names4 = [n_ for n_, _g in _mx3.played[_pl4:]]
                check("a SHOT at the MENU button plays the GUNSHOT and then the selection cue, in that order, "
                      "(the cue within milliseconds: it is in the queue after one event-loop turn)",
                      _names4 == [_gun3, _SEL3], str(_names4))
                _pl4 = len(_mx3.played)
                _pull3(_W3 * 0.98, _H3 * 0.70)           # a plain miss on the drill: no cue, no effect
                _tm4.sleep(0.02)
                app.processEvents()
                check("a shot at a DRILL (a miss) plays the gunshot and no cue",
                      [n_ for n_, _g in _mx3.played[_pl4:]] == [_gun3])
                _pl4 = len(_mx3.played)
                _before4 = (lab.session.shots, lab.session.hits, lab.session.misses, lab.session.dry_pulls,
                            host3.remaining(0), host3.tallies[0].after_done)
                host3.on_click(float(_btn3.x()), float(_btn3.y()), _ctx3)
                lab._refresh_target_diagnostics()
                _tm4.sleep(0.02)
                app.processEvents()
                check("a CLICK on the MENU button plays the cue alone (NO gunshot)",
                      [n_ for n_, _g in _mx3.played[_pl4:]] == [_SEL3],
                      str([n_ for n_, _g in _mx3.played[_pl4:]]))
                check("...and the click moved no session shot, hit, miss, dry pull, magazine count or after_done",
                      (lab.session.shots, lab.session.hits, lab.session.misses, lab.session.dry_pulls,
                       host3.remaining(0), host3.tallies[0].after_done) == _before4)
                # the pop on an empty magazine DURING a drill (the hold), through the app
                host3.restart(_ctx3)
                host3.hold_s = REAL_HOLD_S
                lab.session.reset()
                _holes3.clear()
                for _i4 in range(_MAG3):
                    _idx3[0] += 1
                    _cx4, _cy4 = _s2c3(*_aim3b)
                    _tn4 = time.perf_counter()
                    lab._on_shot(_Shot3(x=_cx4, y=_cy4, nx=_cx4 / 1280.0, ny=_cy4 / 720.0, frames=3, duration_ms=85.0,
                                        peak=180, area=120.0, first_timestamp=_tn4, last_timestamp=_tn4 + 0.085,
                                        index=_idx3[0], track_id=0, elongation=1.2))
                check("(precondition) the magazine is empty and the 1.2 s hold is running: the drill is still on screen",
                      host3.remaining(0) == 0 and not host3.round_over)
                _pl4, _ha4 = len(_mx3.played), _holes3.added
                _idx3[0] += 1
                _cx4, _cy4 = _s2c3(*_aim3b)
                _tn4 = time.perf_counter()
                lab._on_shot(_Shot3(x=_cx4, y=_cy4, nx=_cx4 / 1280.0, ny=_cy4 / 720.0, frames=3, duration_ms=85.0,
                                    peak=180, area=120.0, first_timestamp=_tn4, last_timestamp=_tn4 + 0.085,
                                    index=_idx3[0], track_id=0, elongation=1.2))
                check("a shot on an EMPTY magazine at a drill plays the bubble pop and NOT the gunshot, and DRAWS its hole "
                      "(rendered pixels)",
                      [n_ for n_, _g in _mx3.played[_pl4:]] == [_DRY3] and _holes3.added == _ha4 + 1 and _holes_px3() > 0,
                      str([n_ for n_, _g in _mx3.played[_pl4:]]))
                check("...scoring nothing: shots, hits and misses are the rounds fired, and the pull is a dry pull",
                      (lab.session.shots, lab.session.hits, lab.session.misses, lab.session.dry_pulls)
                      == (_MAG3, _MAG3, 0, 1) and lab.session.consistent())
                host3.hold_s = 0.0
                host3.restart(_ctx3)
                lab.session.reset()
                # The same shot with the round NOT over must still bang: the half that must not regress.
                host3.restart(_ctx3)
                lab.session.reset()
                _pl3 = len(_mx3.played)
                _pull3(*host3.mode.aim_point(_ctx3))
                check("after the next round starts, the gun bangs again and the bullets are full minus one",
                      len(_mx3.played) == _pl3 + 1 and host3.remaining(0) == _MAG3 - 1 and not host3.round_over)
                _live_ln3 = [l_ for l_ in _snh3() if l_.startswith("SAW a")]
                check("the SAME seen-but-not-heard detection on a LIVE magazine still reads as a possible phantom, exactly as before",
                      len(_live_ln3) == 1 and "likely a phantom detection, not a trigger pull" in _live_ln3[0]
                      and "dry pull" not in _live_ln3[0], str(_live_ln3))
            finally:
                lab._log = _real_log3
                host3.set_magazine(_N3)
                host3.set_mode(_arcade.StaticTargets(), _ctx3, players=1)
        finally:
            lab.mapping = _map_was3
            if _opened3 and lab.target_window is not None:
                lab._toggle_target_mode()
                app.processEvents()
            host3.set_mode(_arcade.StaticTargets(), _ctx3, players=1)
            lab.session.reset()
            lab.layers["holes"].clear()

    # ------------------------------------------------------------------
    # A SLEEPING DISPLAY IS NOT A BLIND CAMERA (2026-10-19, three launches lost)
    # ------------------------------------------------------------------
    # The monitor power-saved between sessions; the flash difference read 0.0; the app
    # concluded its healthy camera could not see the screen, cycled to the lens-capped
    # C920 and shut itself down. Driven through the REAL path -- the screen search's
    # 'white' handler sets the flags, `_calibrate_if_needed` reads them and either
    # reports or cycles -- because a check that asks the reporter what it would say
    # while the branch above it still cycles cameras is the hole that shipped this.
    import re as _re20
    import numpy as _np20
    from lsa.calibrate import screen as _screen20

    class _FakeCombo20:
        def __init__(self):
            self.set_to = []

        def currentData(self):
            return 2

        def findData(self, i):
            return 0

        def setCurrentIndex(self, n):
            self.set_to.append(n)

    def _drive20(white_max, label):
        """Run the white-frame handler then the setup branch; return what the app did."""
        said, cycled, closed, logs = [], [], [], []
        saved = {k: getattr(lab, k) for k in ("cmb_device", "_next_camera_to_try", "_connected_device_is_virtual",
                                              "close", "_finish_screen_search", "_log", "mapping")}
        combo = _FakeCombo20()
        say_was = lab.voice.say
        try:
            lab.cmb_device = combo
            lab._next_camera_to_try = lambda: (cycled.append(1), (1, "HD Pro Webcam C920"))[1]
            lab._connected_device_is_virtual = lambda: ""
            lab.close = lambda *a, **k: closed.append(1)
            lab._finish_screen_search = lambda q: None
            lab._log = lambda m, *a, **k: logs.append(str(m))
            lab.voice.say = lambda key, *a, **k: said.append(key)
            lab.mapping = None
            black = _np20.zeros((720, 1280, 3), _np20.uint8)
            white = _np20.zeros((720, 1280, 3), _np20.uint8)
            if white_max:
                white[:] = white_max
                black[:] = white_max            # same brightness both frames: difference 0.0
            lab._calib_black = black
            lab._blind = False
            lab._display_dark = False
            lab._on_still_ready("white", white)          # the real handler sets the flags
            flags = (lab._blind, lab._display_dark)
            lab._calibrate_if_needed()                   # the real branch reads them
        finally:
            lab.voice.say = say_was
            for k, v in saved.items():
                setattr(lab, k, v)
        return dict(said=said, cycled=cycled, closed=closed, logs=logs, set_to=combo.set_to,
                    flags=flags, alert=lab.layers["alert"].title)

    dark20 = _drive20(0, "asleep")
    check("a flash where the white frame never got bright is reported as a SLEEPING DISPLAY, not a blind camera",
          dark20["flags"] == (True, True)
          and any("ASLEEP" in l_.upper() for l_ in dark20["logs"])
          and not any("cannot see the screen, so trying" in l_ for l_ in dark20["logs"])
          and dark20["alert"] == "THE SCREEN IS OFF OR ASLEEP",
          f"{dark20['flags']} alert={dark20['alert']!r}")
    check("...and it does NOT advance to the next camera device (no lookup, no switch)",
          dark20["cycled"] == [] and dark20["set_to"] == [], str(dark20["set_to"]))
    check("...and it does NOT shut the app down", dark20["closed"] == [])
    check("...and it SPEAKS (he is downrange): the existing 'Not ready. Check the screen.' line",
          dark20["said"] == ["v_not_ready"], str(dark20["said"]))
    check("...and it tells him what to do: wake the monitor, then re-run setup (F5 toggles: twice)",
          any("wake" in l_.lower() and "F5" in l_ for l_ in dark20["logs"]))

    lit20 = _drive20(200, "lit")
    check("a flash where the white frame DID get bright but made no difference is STILL the camera not seeing "
          "the screen ...", lit20["flags"] == (True, False)
          and any("cannot see the screen, so trying the next one" in l_ for l_ in lit20["logs"]),
          f"{lit20['flags']}")
    check("...and STILL cycles to the next camera (the behaviour the cycling was built for survives)",
          lit20["cycled"] == [1] and lit20["set_to"] == [0], f"{lit20['cycled']} {lit20['set_to']}")
    check("...and does not claim the display is asleep", lit20["said"] == [] and lit20["closed"] == []
          and not any("ASLEEP" in l_.upper() for l_ in lit20["logs"]))

    # ONE threshold, read from one place: the app and the camera doctor (`lsa/doctor.py`,
    # which tools/camera_health.py now fronts) agree.
    _health20 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "doctor.py").read_text(encoding="utf-8")
    _lm20 = _screen20.LIT_WHITE_MAX
    _at = _np20.full((4, 4, 3), _lm20, _np20.uint8)
    _above = _np20.full((4, 4, 3), _lm20 + 1, _np20.uint8)
    check("the lit-white test has ONE definition: the boundary is exact (at the limit is dark, one above is lit)",
          not _screen20.display_is_lit(_at) and _screen20.display_is_lit(_above)
          and not _screen20.display_is_lit(None))
    check("...and the camera doctor uses THAT function, with no literal threshold of its own",
          "display_is_lit" in _health20 and not _re20.search(r"max\(\)\)?\s*>\s*\d+", _health20),
          "a second copy of the number would let the tool and the app disagree")

    # ------------------------------------------------------------------
    # THE GHOST SCORE'S SEAMS IN THE APP (2026-10-21, piece 3). The arithmetic is in
    # test_ghost, the screen in test_modes section 49; THIS is the app end: the
    # settings file is written when -- and only when -- a round set a record, once,
    # and the numbers the guard reads actually leave the camera path. Behind its own
    # crash guard so a raise is a failed check, not a dead suite.
    # ------------------------------------------------------------------
    try:
        import inspect as _insp21
        import re as _re21
        from lsa import bests as _b21, ghost as _g21

        host21 = lab.layers["modes"]
        check("the host's bests are a view onto settings.bests: the very dict AppSettings saves",
              host21.bests is not None and host21.bests.data is lab.settings.bests
              and isinstance(lab.settings.bests, dict))
        key21 = _b21.best_key("zz_smoke", 17, 1)            # a drill that does not exist: never his
        bests_before21 = dict(lab.settings.bests)
        saves21: list = []
        _save_was21 = lab._save_settings
        lab._save_settings = lambda *a, **k: (saves21.append(1), True)[1]
        try:
            host21.take_bests_changed()                      # drain anything pending
            for _i in range(5):
                lab._publish_measuring()
            idle21 = len(saves21)
            host21.bests.offer(key21, {
                "value": 500.0, "points": 100, "seconds": 5.0, "shots": 17, "basis": "measured",
                "declared_waits": 0, "ghost_interval": _g21.GHOST_INTERVAL, "when": ""}, None)
            for _i in range(5):
                lab._publish_measuring()
            record21 = len(saves21)
            for _i in range(5):
                lab._publish_measuring()
            later21 = len(saves21)
        finally:
            lab._save_settings = _save_was21
            lab.settings.bests.pop(key21, None)
        check("the settings file is NOT written by the app's tick when no round set a record",
              idle21 == 0, f"{idle21} writes")
        check("...is written exactly ONCE when a round does, and not again on later ticks",
              record21 == 1 and later21 == 1, f"{record21} then {later21}")
        check("...and the test left the bests as it found them", lab.settings.bests == bests_before21)
        check("_save_bests writes through the one settings writer, once (AppSettings.save is atomic: "
              "RULES C16)",
              _insp21.getsource(CameraLab._save_bests).count("self._save_settings()") == 1
              and _insp21.getsource(CameraLab._save_settings).count("st.save(SETTINGS_PATH)") == 1)
        # An earlier section leaves a CONSTRUCTED, never-opened CaptureThread on the
        # lab (section 20); that is not a live capture. A LIVE one -- a device handle
        # held (`_cap`) or the thread running -- must still stop this test, because
        # a changed FPS box would disconnect and reconnect it. The constructed one is
        # taken off the lab for the two changes (the handler's `capture is None` path
        # is then the only one it can take, so nothing is disconnected, no reconnect
        # timer is armed and no device can be opened) and put back after.
        from lsa.camera.capture import CaptureThread as _CT21
        cap21 = lab.capture
        if cap21 is not None:
            if not isinstance(cap21, _CT21):
                raise RuntimeError(f"lab.capture is a {type(cap21).__name__}, not a CaptureThread")
            if cap21._cap is not None or cap21.isRunning():
                raise RuntimeError("a changed FPS box would reconnect a LIVE capture; this test needs none")
        fps_was21 = lab.spin_fps.value()
        lab.capture = None
        try:
            lab.spin_fps.setValue(45)
            told21 = host21.fps
            lab.spin_fps.setValue(fps_was21)
        finally:
            lab.capture = cap21
        check("the ghost is told the FPS the capture is configured with when the FPS box changes, "
              "and again when it is put back",
              told21 == 45.0 and host21.fps == float(fps_was21), f"{told21} / {host21.fps}")
        lab_src21 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
                     / "camera_lab.py").read_text(encoding="utf-8")
        check("every registered shot carries its pulse's brightness into the host "
              "(the record guard's median-peak input)",
              _re21.search(r"ShotEvent\([^)]*peak=float\(shot\.peak\)", lab_src21) is not None)
        check("...and a live shot's microphone verdict (deaf or not) reaches the host, from the one "
              "place that knows it",
              "note_live_unheard(shot_t, bool(tag))" in lab_src21)
        check("the settings write is driven by the host's one-shot signal and nothing else",
              lab_src21.count("take_bests_changed()") == 1
              and "if host.take_bests_changed():" in lab_src21)
    except Exception as exc_21:                                     # noqa: BLE001 - the finding
        check("the ghost score's app seams ran to their end without raising", False,
              f"{type(exc_21).__name__}: {exc_21}")

    # == the brightness grid, CLOSED MID-RUN (Escape and F11 both end the app here) ==
    #
    # Escape closes the whole app now, so "ESC mid-grid" IS this path. The hazard
    # is specific: the relaxed thresholds are SETTINGS (`sp_threshold` and
    # `sp_minpeak` are saved), so a close that saves before it restores writes a
    # detection bar of floor+3 into settings.local.json and the next launch starts
    # permissive. Asserted by what every save SAW, not by the order of two lines:
    # `_save_settings` is replaced with a recorder (it never writes in this suite).
    shut_u: dict = {}
    shut_saves: list = []
    shut_written: list = []

    def _record_save_u(*a, **k):
        # (1) what the two spin boxes show; (2) what `_save_settings` WOULD PUT IN THE FILE:
        # it stores `vars(self._detector_params())` as `settings.detector`, so the
        # detector params at this instant ARE the min_peak / threshold written to
        # settings.local.json. Both are read at the moment of the save, never after.
        shut_saves.append((lab.sp_threshold.value(), lab.sp_minpeak.value()))
        dp_ = lab._detector_params()
        shut_written.append((dp_.threshold, dp_.min_peak))
        return True

    try:
        if "prepare" not in u_hooks:
            raise RuntimeError("the brightness-grid section did not set itself up")
        lab._save_settings = _record_save_u
        u_hooks["prepare"]()
        lab.start_sensitivity_grid()
        shut_u["running"] = bool(lab._sgrid_active)
        shut_u["relaxed"] = lab.sp_minpeak.value() < 40
        # THE SHOT GRID (G) CASE, in the same close. `_sgrid_end` already restores the
        # levels for U, so with only U running the bare `_restore_levels()` that
        # follows it in closeEvent has nothing left to do and removing it changes
        # nothing any check sees (measured 2026-10-23). The bare line exists for the
        # SHIPPED grid, which has no exit of its own. Once U's exit has run, put the
        # app in exactly the state a closing G grid is in -- thresholds relaxed and
        # remembered, `_grid_active` up, `_sgrid_active` down -- so that line is the
        # only thing standing between the relaxed bar and settings.local.json.
        _sgrid_end_real = lab._sgrid_end

        def _sgrid_end_then_g(*a, **k):
            _sgrid_end_real(*a, **k)
            lab._relax_for_calibration()          # no-op if the exit left them relaxed
            lab._grid_active = True
            shut_u["g_relaxed"] = (lab.sp_minpeak.value() < 40
                                   and lab._grid_saved_levels is not None
                                   and not lab._sgrid_active)

        lab._sgrid_end = _sgrid_end_then_g
        lab.capture = None            # closing releases the camera; the stub has no stop()
        shut_u["saves"] = shut_saves
    except Exception as exc_u2:                                     # noqa: BLE001 - the finding
        check("the brightness grid could be set running before the app was closed", False,
              f"{type(exc_u2).__name__}: {exc_u2}")

    lab.close()
    # After the close, because closing is when an app saves.
    # PRESENCE BEFORE INDEXING (2026-10-23): `shut_u` may hold only some of its keys when the
    # set-up raised halfway, so every read is `.get`, and a missing value FAILS the check.
    # The first two checks are about the grid being ended; the THIRD is about the one thing
    # that reaches his file, and is the check mutate_detector entry 97 names.
    if shut_u:
        check("brightness grid: closing the app in the middle of the grid - it WAS running with relaxed thresholds, "
              "every save made during the close saw the RESTORED 35/40, and the flag is down",
              shut_u.get("running") is True and shut_u.get("relaxed") is True
              and bool(shut_saves) and all(s_ == (35, 40) for s_ in shut_saves)
              and not lab._sgrid_active and lab._grid_saved_levels is None,
              f"running={shut_u.get('running')} relaxed={shut_u.get('relaxed')} saves={shut_saves}")
        check("shot grid (G): closing the app in the middle of it - its thresholds WERE relaxed when the "
              "bare restore in closeEvent ran, and every save made during the close saw the RESTORED 35/40 "
              "(the bar settings.local.json keeps)",
              shut_u.get("g_relaxed") is True and bool(shut_saves)
              and all(s_ == (35, 40) for s_ in shut_saves)
              and lab._grid_saved_levels is None,
              f"g_relaxed={shut_u.get('g_relaxed')} saves={shut_saves}")
    check("settings.local.json at close: the detector bar WRITTEN there is the restored 35/40, "
          "never the grid's relaxed floor+3 (at least one save happened, and every one carried the "
          "restored bar)",
          shut_u.get("relaxed") is True          # else nothing was at risk and this is vacuous
          and len(shut_written) >= 1 and all(w_ == (35, 40) for w_ in shut_written),
          (f"{len(shut_written)} save(s); detector (threshold, min_peak) at each: {shut_written}"
           + ("" if shut_u.get("relaxed") is True else
              "; and the bar was never relaxed before the close, so this proved nothing")))
    owned.check(lambda ok, what, detail="": check(what, ok, detail))
    passed = all(checks)
    print("\n" + (f"ALL {len(checks)} CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    print(f"open {out.resolve()} to see the target screen")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
