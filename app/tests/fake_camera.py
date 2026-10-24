"""A camera that is not there, watching a screen that is.

WHY THIS EXISTS
---------------
Every fault found on hardware on 2026-10-15 was a RACE between the app's own
routines: two noise measurements running at once, the exposure probe drawing
dots inside one of them, a self test judged against a bar from neither. None
of them needed real optics to reproduce. Each one cost about seventy seconds
of the shooter's monitor to provoke, plus a log read afterwards to recognise,
and whether it appeared at all depended on millisecond timing -- so five live
launches sampled the problem badly. The gate measured

    12/12, 0/12, 12/12, 1/12, 12/12

on identical code, in the same room, minutes apart.

This stands in for `CaptureThread` behind the one seam the app has for it
(`CameraLab.capture_factory`). Everything above that seam is the real thing:
the real startup chain, the real timers, the real detector, the real tracker,
the real calibration, the real verdict. Only the optics are modelled.

WHAT IS REAL AND WHAT IS NOT
----------------------------
REAL: the app draws on its actual target window (offscreen), this grabs that
window, and the genuine `LaserDetector` and `ShotTracker` run on the result.
A dot the app paints is found by the same code that finds a laser.

MODELLED: the view. The screen is seen off-axis through a homography, dimmer
at the far edge, with sensor noise, and with brightness scaling by exposure.
Numbers come from this bench -- a panel filling about 17% of the frame, a
4.6x brightness falloff across it, and the measured exposure response.

NOT MODELLED, and these are the limits to state out loud: camera firmware
behaviour, USB stalls, MJPEG chroma subsampling, and the physical difference
between a screen that EMITS a dot and a laser that REFLECTS off matte
coating. This harness cannot say anything about those. It exists to answer
one question -- does the startup chain do the right things in the right order
every single time -- and that is the question the live gate was failing.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa.camera.capture import Preview, Stats          # noqa: E402
from lsa.camera.evidence import FrameEvidence          # noqa: E402
from lsa.detect.laser import LaserDetector, Shot, ShotTracker  # noqa: E402

FRAME_W, FRAME_H = 1280, 720


@dataclass
class Optics:
    """How this fake camera sees the monitor. Numbers from the real bench."""

    # Where the screen's four corners land in the camera image. Measured on
    # 2026-10-15: a convex quad with sides 568/310/564/475px, covering about
    # 17% of the frame, viewed from the left.
    quad: tuple = ((272.0, 87.0), (877.0, 192.0), (879.0, 502.0), (341.0, 625.0))
    # The far side of an obliquely-viewed panel is dimmer. Measured along a
    # line just inside the top edge: 179 at the near edge, 39 at the far one.
    falloff: float = 4.6
    # Sensor noise, as a standard deviation in counts. The real chroma floor
    # measures 0-4 on a quiet screen.
    noise: float = 1.1
    # Exposure response. The camera's mean brightness roughly doubles per
    # stop; -5 is taken as unity so the numbers sit where the bench does.
    reference_exposure: float = -5.0
    # SENSOR GAIN, and it is the whole difference between the app's two
    # profiles. The shooting preset pins gain to 0 on purpose -- "the
    # screen fades to a ghost, the 650nm dot still saturates" -- while the
    # calibration preset deliberately does NOT touch gain, because forcing
    # it to 0 once took frame maximum from 255 to 7 and left the search
    # photographing a nearly black room, flash difference 0.4 against 77.5.
    #
    # Leaving it out of this model made the simulated camera blind in
    # exactly that way: the screen search ran at exposure -10 with no gain
    # to compensate, measured a flash difference near zero, and correctly
    # concluded the camera was not pointed at the screen. The harness was
    # reproducing a fault of its own making.
    #
    # Calibrated against the bench: at -10 with gain free, a white screen
    # saturates (flash difference 44); at -5 with gain pinned, the dark
    # target background reads about 13 and the panel is a ghost.
    gain_free: float = 90.0
    gain_pinned: float = 1.0
    # Latency, in frames, between the screen changing and the camera seeing
    # it. Webcams lag, and `request_still` exists because of it.
    lag_frames: int = 1


def screen_to_camera(optics: Optics, screen_size: tuple, sx: float, sy: float
                     ) -> tuple[float, float]:
    """Where a screen point images in this fake camera: the same perspective
    warp `FakeCapture.view` applies to the whole window, applied to one point.
    No lens distortion (the view does not model one), so a mapping fitted to
    these points inverts them exactly."""
    sw, sh = screen_size
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    warp = cv2.getPerspectiveTransform(src, np.float32(optics.quad))
    out = cv2.perspectiveTransform(np.float32([[[sx, sy]]]), warp)[0][0]
    return float(out[0]), float(out[1])


def shot_at_screen(optics: Optics, screen_size: tuple, sx: float, sy: float,
                   peak: int, *, index: int = 1, track_id: int = 1,
                   t: float = 0.0, frames: int = 3, area: float = 150.0,
                   offset_px: tuple = (0.0, 0.0)) -> Shot:
    """A `Shot` the detector would have reported for a laser landing at a
    chosen SCREEN position, at a chosen brightness.

    The view model above cannot make a FAINT shot (peak ~20, the dim side of
    the panel and the whole case the brightness grid exists for): it renders
    whatever the app paints, and the app paints dots as bright as it can. A
    real laser's brightness is the one thing a screen cannot emulate (a screen
    EMITS its dot, a laser REFLECTS off the matte coating), so the shot is
    constructed at the point where the detector hands over to the app: the
    position comes from the optics, the peak is stated by the test.
    `offset_px` is aim scatter in camera pixels.
    """
    cx, cy = screen_to_camera(optics, screen_size, sx, sy)
    cx, cy = cx + offset_px[0], cy + offset_px[1]
    return Shot(x=cx, y=cy, nx=cx / FRAME_W, ny=cy / FRAME_H, frames=frames,
                peak=int(peak), area=float(area), first_timestamp=t,
                last_timestamp=t + 0.02 * max(0, frames - 1),
                duration_ms=20.0 * frames, index=index, track_id=track_id)


class FakeCapture(QObject):
    """Stands in for CaptureThread. Same signals, same methods, no device."""

    opened = Signal(dict)
    failed = Signal(str)
    preview_ready = Signal(object)
    shot_detected = Signal(object)
    shot_provisional = Signal(object)
    probe_ready = Signal(object)
    control_echo = Signal(str, float, bool)
    log = Signal(str)
    recording_changed = Signal(bool, str)
    noise_measured = Signal(float, float, int)
    room_measured = Signal(float, float, int)
    exposure_chosen = Signal(float, object)
    still_ready = Signal(str, object)

    # What the app is allowed to see on screen, supplied by the test. Returns
    # a QImage of the monitor as it is right now, or None.
    view_provider = None

    # Mirror of CaptureThread's passive screen-watch outputs (RULES C12). The
    # stand-in does not simulate the camera's luminance, so the watch reads
    # None ("no ink to judge") and never fires in the harness.
    screen_level = None
    screen_level_at = 0.0

    # JITTER, in milliseconds, added to every asynchronous reply.
    #
    # This is the whole point of the harness. The races only appear at
    # particular interleavings, so a fixed-timing simulation would pass
    # forever while hardware kept failing one launch in two. Each run picks
    # its own delays, so running it many times samples the orderings a real
    # machine would take weeks to produce.
    jitter_ms = (0, 0)

    # The real capture throttles its preview to this; mirrored so the
    # app sees the same rate it would on hardware.
    PREVIEW_HZ = 20.0

    def __init__(self, config, det_params, grouper_cfg, optics: Optics | None = None):
        super().__init__()
        self.config = config
        self.optics = optics or Optics()
        self._detector = LaserDetector(det_params)
        self._grouper_config = grouper_cfg
        self._grouper = self._make_grouper()
        self._stats = Stats()
        self._control_values: dict = {}
        self._exposure = -5.0
        self._running = False
        self._rng = np.random.default_rng(0)
        self._pending: list = []
        # The noise-floor window, mirroring CaptureThread._noise_until and
        # _noise_samples: a deadline and one detector score per frame that
        # arrived inside it. Declared here, not conjured on first use (C8).
        self._noise_until: float = 0.0
        self._noise_samples: list[float] = []
        # Per-frame near-miss telemetry, mirroring CaptureThread._evidence, so
        # `CameraLab._poll_audio` can ask for it on either (RULES C12).
        self._evidence = FrameEvidence()
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._frame)
        # Everything the harness wants to assert about ORDER, recorded as it
        # happens. A race is a statement about order, so it has to be
        # observable as order, not inferred from an outcome.
        self.events: list[tuple[float, str]] = []
        self.overlaps: list[str] = []
        self._busy: set[str] = set()

    # -- bookkeeping the tests assert on ------------------------------------

    def _note(self, what: str) -> None:
        self.events.append((time.perf_counter(), what))

    def _begin(self, what: str) -> bool:
        """Mark an exclusive camera operation as started.

        If one is already running, that is a race, and it is recorded rather
        than smoothed over. The real camera has exactly this property: it can
        only be doing one thing at a time, and the app discovered that by
        producing garbage rather than by being told.
        """
        if self._busy:
            self.overlaps.append(f"{what} started while {sorted(self._busy)} running")
        self._busy.add(what)
        self._note("begin " + what)
        return True

    def _end(self, what: str) -> None:
        self._busy.discard(what)
        self._note("end " + what)

    def _jitter(self) -> float:
        """One draw of the asynchronous-reply jitter, in milliseconds."""
        lo, hi = self.jitter_ms
        return float(self._rng.integers(lo, hi + 1)) if hi > lo else float(lo)

    def _later(self, ms: float, fn) -> None:
        QTimer.singleShot(int(max(0.0, ms + self._jitter())), fn)

    # -- the view -----------------------------------------------------------

    def _screen_bgr(self) -> np.ndarray | None:
        """What the monitor is showing, as an array, or None."""
        provider = self.view_provider
        if provider is None:
            return None
        img = provider()
        if img is None:
            return None
        img = img.convertToFormat(QImage.Format_RGBA8888)
        h, w = img.height(), img.width()
        if h <= 0 or w <= 0:
            return None
        buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=h * w * 4)
        return cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_RGBA2BGR).copy()

    def view(self, *, blank: bool = False,
             exposure: "float | None" = None,
             sensor_gain: "float | None" = None) -> np.ndarray:
        """One camera frame of whatever is on the monitor.

        `blank`, `exposure` and `sensor_gain` exist for `blank_floor` only: a
        screen with nothing on it, at an exposure and sensor gain the caller
        names. The app never passes them.
        """
        screen = None if blank else self._screen_bgr()
        exp = self._exposure if exposure is None else float(exposure)
        frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        if screen is not None and screen.size:
            sh, sw = screen.shape[:2]
            src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
            dst = np.float32(self.optics.quad)
            warp = cv2.getPerspectiveTransform(src, dst)
            cv2.warpPerspective(screen, warp, (FRAME_W, FRAME_H), dst=frame,
                                borderMode=cv2.BORDER_TRANSPARENT)
            frame = self._shade(frame)
        gain = (2.0 ** (exp - self.optics.reference_exposure)
                * (self._sensor_gain() if sensor_gain is None
                   else float(sensor_gain)))
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255)
        # Noise grows far more slowly than signal with exposure, which is why
        # a longer exposure buys margin at all. Measured: the chroma floor
        # barely moved across three stops while the signal fell four-fold.
        # Noise grows far more slowly than signal, which is why a longer
        # exposure buys margin at all -- measured, the chroma floor barely
        # moved across three stops while the signal fell four-fold. It is
        # taken against the exposure alone, not the sensor gain, because
        # the gain-free profile is only ever used to LOOK at the monitor,
        # never to detect against a threshold.
        exposure_gain = 2.0 ** (exp - self.optics.reference_exposure)
        noise = self._rng.normal(
            0.0, self.optics.noise * (1.0 + 0.25 * exposure_gain), frame.shape)
        return np.clip(frame + noise, 0, 255).astype(np.uint8)

    def _sensor_gain(self) -> float:
        """High unless the app has pinned gain to zero for shooting."""
        g = self._control_values.get("gain")
        if g is not None and float(g) <= 0.5:
            return self.optics.gain_pinned
        return self.optics.gain_free

    def _shade(self, frame: np.ndarray) -> np.ndarray:
        """Dim the far side of the panel, as an oblique view really does."""
        xs = np.linspace(1.0, 1.0 / self.optics.falloff, FRAME_W, dtype=np.float32)
        return np.clip(frame.astype(np.float32) * xs[None, :, None],
                       0, 255).astype(np.uint8)

    # -- the CaptureThread surface ------------------------------------------

    def start(self) -> None:
        self._running = True
        self._note("open")
        self._later(40, self._announce_open)
        self._tick.start(16)

    def _announce_open(self) -> None:
        asked = {"width": self.config.width, "height": self.config.height,
                 "fourcc": self.config.fourcc, "fps": self.config.fps}
        self.opened.emit({
            "width": FRAME_W, "height": FRAME_H, "fourcc": "MJPG",
            "fps": 60.0, "backend": self.config.backend,
            "requested": asked,
            # GAIN IS REPORTED HERE, as the real camera reports it.
            #
            # A stand-in that emits fewer signals than the real thing
            # excuses every consumer of the missing ones (RULES C12), and
            # this is the field that lets the calibration preset undo the
            # shooting profile's gain pin. Without it the app would take
            # the None path -- which is exactly the behaviour that hid
            # this bug -- and the harness would keep passing while the
            # app photographed the monitor with the sensor turned down.
            "controls": {"exposure": self._exposure,
                         "gain": self.optics.gain_free},
        })

    def gain_at_open(self) -> "float | None":
        """Mirrors `CaptureThread.gain_at_open`."""
        return float(self.optics.gain_free)

    def written_value(self, key: str) -> "float | None":
        """Mirrors `CaptureThread.written_value`.

        A stand-in that emits fewer signals -- or answers fewer
        questions -- than the real thing excuses every consumer of the
        missing ones (RULES C12).
        """
        v = self._control_values.get(key)
        return None if v is None else float(v)

    def _frame(self) -> None:
        """One camera frame, driven exactly as the real loop drives it.

        The detector takes a wall-clock timestamp and the tracker takes a
        FRAME INDEX -- they are different quantities and swapping them
        silently changes what counts as a shot, since the tracker measures
        pulse length in frames.
        """
        if not self._running:
            return
        frame = self.view()
        self._stats.frames += 1
        self._stats.fps = 60.0
        now = time.perf_counter()
        dets = self._detector.process(frame, timestamp=now)
        self._grouper.feed(dets, self._stats.frames)
        for prov in self._grouper.take_provisional():
            self.shot_provisional.emit(prov)
        self._stats.rejected_runs = self._grouper.rejected_runs
        det = self._detector
        self._evidence.push(now, det.last_frame_peak, det.last_dim_peak,
                            det.last_dim_pos, det.last_off_screen_peak,
                            self._grouper.rejected_runs,
                            amp_gate=det.last_amp_gate,
                            big=det.last_big_rejected)

        # THE NOISE FLOOR, measured as CaptureThread measures it (capture.py,
        # the `if self._noise_until:` block of the read loop): the maximum of
        # the detector's own score -- after the gain map, the blur and the
        # background subtraction -- over every frame inside the window, and
        # the TRUE number of frames that arrived.
        #
        # This used to take ONE raw frame, report its whole-frame chroma, and
        # claim `seconds * 60` frames it never looked at. So the harness's
        # floor was a different statistic from the app's: the gain-map judge
        # compared a number with itself, and the floor sat at whatever the
        # simulated sensor noise maximum was at that exposure (RULES C12).
        if self._noise_until:
            score = self._detector.last_score
            if score is not None:
                self._noise_samples.append(float(score.max()))
            if now >= self._noise_until:
                self._noise_until = 0.0
                samples = np.array(self._noise_samples or [0.0], dtype=np.float32)
                self._end("noise")
                self.noise_measured.emit(
                    float(samples.max()), float(np.percentile(samples, 99)),
                    int(samples.size))

        # EMIT A PREVIEW, as the real capture does.
        #
        # Leaving it out left a whole branch of the app unexercised, and
        # it showed: the verdict saves what the CAMERA sees beside what
        # the screen shows, and that image never appeared because nothing
        # here set the preview it comes from. A stand-in that emits fewer
        # signals than the real thing silently excuses every consumer of
        # the missing ones -- the app code was fine and simply never ran.
        every = max(1, int(60.0 / self.PREVIEW_HZ))
        if self._stats.frames % every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            img = QImage(bytes(rgb.data), w, h, 3 * w,
                         QImage.Format_RGB888).copy()
            self.preview_ready.emit(Preview(
                image=img, mask=None, detections=list(dets),
                stats=self._stats, frame_size=(w, h), mask_scale=1.0))

    def fire_at_screen(self, screen_size: tuple, sx: float, sy: float, peak: int,
                       **kw) -> Shot:
        """Deliver a shot of a chosen brightness at a chosen screen position,
        through the same two signals the real tracker emits: provisional
        first (the app's `_on_shot`), then final (`_on_shot_final`)."""
        shot = shot_at_screen(self.optics, screen_size, sx, sy, peak, **kw)
        self.shot_provisional.emit(shot)
        self.shot_detected.emit(shot)
        return shot

    def stop(self) -> None:
        self._running = False
        self._tick.stop()
        self._note("close")

    def wait(self, _ms: int = 0) -> bool:
        return True

    def request_still(self, tag: str, frames: int = 2) -> None:
        self._begin("still:" + tag)

        def deliver() -> None:
            frame = self.view()
            self._end("still:" + tag)
            self.still_ready.emit(tag, frame)
        self._later(30 + 12 * self.optics.lag_frames, deliver)

    def start_noise_calibration(self, seconds: float = 3.0) -> None:
        self._begin("noise")
        # The real capture logs this line when it arms the window, and
        # `tools/launch_timing.py` keys its noise-floor phase on the text. A
        # stand-in that emits fewer signals than the real thing excuses every
        # consumer of the missing ones (RULES C12): without it a harness log
        # could not be phase-attributed by the tool that produced this
        # project's launch budgets. Same text, same moment.
        self.log.emit(
            f"measuring noise floor for {seconds:g}s - do NOT fire during this")
        # The END of the window is jittered, as every asynchronous reply is;
        # the frames inside it are whatever `_frame` actually produces.
        self._noise_until = (time.perf_counter() + float(seconds)
                             + self._jitter() / 1000.0)
        self._noise_samples = []

    def noise_state(self) -> dict:
        """Everything a floor reading depends on, as of NOW.

        Taken by the harness at the instant the app reports a floor, so that
        floor can be judged against an empty screen in ITS OWN state -- the
        exposure, sensor gain, gain map and detector parameters it was
        measured under -- instead of whatever state the launch happened to
        end in. The gain map is replaced by `set_gain_map`, never mutated, so
        holding the reference is a snapshot.
        """
        return {"exposure": float(self._exposure),
                "sensor_gain": float(self._sensor_gain()),
                "gain_map": self._detector._gain,
                "params": self._detector.params}

    def blank_floor(self, exposure: "float | None" = None, frames: int = 180,
                    warmup: int = 20, state: "dict | None" = None) -> float:
        """The floor this fake would report for an EMPTY screen.

        The reference the harness judges a launch's floor against: the same
        pipeline as the window above -- a fresh `LaserDetector` with this
        capture's own parameters and gain map, maximum of its score over
        `frames` frames -- fed an all-black screen at `exposure`, with the
        sensor gain the app has left set. If a launch's floor climbs above
        this, something that was DRAWN is visible to the detector.

        `warmup` frames are fed first and not counted, so the background
        model has settled as it has by the time the app measures a floor
        (its very first frame scores zero by construction).

        `state` is a `noise_state()` snapshot: when given, the exposure, sensor
        gain, gain map and detector parameters come from it rather than from
        this camera as it is now.

        Uses its own random stream, so calling it neither perturbs nor
        depends on the run it is a reference for.
        """
        if state is not None:
            exposure = state["exposure"]
            params, gmap, sg = (state["params"], state["gain_map"],
                                state["sensor_gain"])
        else:
            if exposure is None:
                exposure = self._exposure
            params, gmap, sg = self._detector.params, self._detector._gain, None
        det = LaserDetector(params)
        det.set_gain_map(gmap)
        saved = self._rng
        self._rng = np.random.default_rng(987654321)
        try:
            peak = 0.0
            for i in range(int(warmup) + int(frames)):
                det.process(self.view(blank=True, exposure=exposure,
                                      sensor_gain=sg))
                if i >= warmup and det.last_score is not None:
                    peak = max(peak, float(det.last_score.max()))
        finally:
            self._rng = saved
        return peak

    def start_room_measurement(self, seconds: float = 1.0) -> None:
        self._begin("room")

        def done() -> None:
            frame = self.view()
            b, g, r = cv2.split(frame)
            chroma = cv2.subtract(r, cv2.max(g, b))
            self._end("room")
            self.room_measured.emit(float(frame.mean()),
                                    float(np.percentile(chroma, 99.0)),
                                    int(seconds * 60))
        self._later(seconds * 1000.0, done)

    def start_exposure_search(self, *_a, **_k) -> None:
        self._begin("sweep")
        rows = []
        keep = self._exposure
        for value in (-3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0):
            self._exposure = value
            self._stats.control_writes += 1
            frame = self.view()
            b, g, r = cv2.split(frame)
            chroma = cv2.subtract(r, cv2.max(g, b))
            rows.append((value, float(frame.mean()), int(frame.max()),
                         float(np.percentile(chroma, 99.0))))
        self._exposure = keep

        def done() -> None:
            from lsa.camera.capture import CaptureThread
            chosen = CaptureThread._choose_exposure(rows, 25.0, 200)
            self._exposure = float(chosen[0])
            self._stats.control_writes += 1
            self._end("sweep")
            self.exposure_chosen.emit(float(chosen[0]), rows)
        self._later(600, done)

    def set_control(self, key: str, value: float) -> None:
        if key in self._control_values and \
                abs(float(self._control_values[key]) - float(value)) < 1e-6:
            self.control_echo.emit(key, float(value), True)
            return
        self._control_values[key] = float(value)
        self._stats.control_writes += 1
        if key == "exposure":
            self._exposure = float(value)
        self.control_echo.emit(key, float(value), True)

    def apply_preset(self, preset) -> None:
        written = 0
        for key, value in getattr(preset, "values", {}).items():
            known = self._control_values.get(key)
            if known is not None and abs(float(known) - float(value)) < 1e-6:
                continue
            self._control_values[key] = float(value)
            self._stats.control_writes += 1
            written += 1
            if key == "exposure":
                self._exposure = float(value)
        self.log.emit(f"applying preset: {getattr(preset, 'name', '?')}  "
                      f"({written} control writes)")

    # -- the rest of the surface, which does nothing here --------------------

    def set_detector_params(self, params) -> None:
        self._detector.set_params(params)

    def _make_grouper(self) -> ShotTracker:
        """Built exactly as the real capture builds it.

        ShotTracker takes KEYWORD arguments, not the config object. Passing
        the object made `min_frames` a GrouperConfig, so every shot died in
        `len(run) >= self.min_frames` with a TypeError swallowed inside the
        detector -- and the harness reported 0 of 12 dots detected while
        the dot was arriving at chroma 155 against a floor of 7.

        A stand-in that constructs the real thing wrongly does not test the
        real thing; it manufactures failures and invites you to go and fix
        the app. Mirror the construction, do not paraphrase it.
        """
        c = self._grouper_config
        return ShotTracker(
            min_frames=c.min_frames,
            single_frame_peak=c.single_frame_peak,
            max_gap_frames=c.max_gap_frames,
            max_move_px=c.max_move_px,
            frame_period_ms=1000.0 / max(1.0, self.config.fps),
            max_tracks=c.max_tracks,
            max_duration_ms=c.max_duration_ms,
        )

    def set_grouper_config(self, cfg) -> None:
        # The real capture rebuilds the tracker and carries its counters
        # across, so a threshold change does not reset the session.
        old = self._grouper
        self._grouper_config = cfg
        self._grouper = self._make_grouper()
        self._grouper.rejected_runs = old.rejected_runs
        self._grouper.rejected_too_long = old.rejected_too_long

    def set_gain_map(self, gmap) -> None:
        self._detector.set_gain_map(gmap)

    def reset_background(self) -> None:
        self._detector.reset_background()

    def evidence_between(self, t0: float, t1: float) -> "dict | None":
        return self._evidence.summarise(t0, t1)

    def reset_stats(self) -> None:
        writes = self._stats.control_writes
        self._stats = Stats()
        self._stats.control_writes = writes
        self._evidence.clear()

    def set_show_mask(self, _on: bool) -> None: ...
    def set_draw_overlay(self, _on: bool) -> None: ...
    def set_aim_help(self, *_a, **_k) -> None: ...
    def start_recording(self, *_a, **_k) -> None: ...
    def stop_recording(self, *_a, **_k) -> None: ...
    def save_snapshot(self, *_a, **_k) -> None: ...
    def request_probe(self, *_a, **_k) -> None: ...
    def request_driver_dialog(self, *_a, **_k) -> None: ...


def install(lab, view_provider, jitter_ms=(0, 0), seed: int = 0) -> None:
    """Point a CameraLab at a camera that is not there."""
    def factory(cfg, det, grp):
        cam = FakeCapture(cfg, det, grp)
        cam.view_provider = view_provider
        cam.jitter_ms = jitter_ms
        cam._rng = np.random.default_rng(seed)
        lab._fake_camera = cam
        return cam
    lab.capture_factory = factory
