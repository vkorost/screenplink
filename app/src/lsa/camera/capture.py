"""Threaded camera capture with inline detection.

Everything that touches cv2.VideoCapture happens on this thread and nowhere
else. The GUI never calls into the camera directly; it posts commands onto a
queue that the capture loop drains between frames. VideoCapture is not
thread-safe and a webcam that gets reconfigured mid-read tends to hang the
whole process, which on Windows means a zombie that holds the device open until
you log out.

The loop does, per frame: read -> detect -> group -> (optionally) record ->
build a scaled preview -> emit. Detection runs on every single frame, because
missing one of the three frames of a 100 ms pulse costs us accuracy; the
preview is throttled, because the GUI does not need 30 fps to be useful.
"""

from __future__ import annotations

import csv
import json
import sys
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal
from PySide6.QtGui import QImage

from . import controls
from .devices import available_backends, fourcc_to_str, str_to_fourcc
from .evidence import FrameEvidence
from ..detect.laser import Detection, DetectorParams, LaserDetector, Shot, ShotTracker
from ..screenwatch import (INK_MIN_LUMA as SCREEN_INK_MIN_LUMA,
                           INK_MIN_PIXELS as SCREEN_INK_MIN_PIXELS,
                           SAMPLE_HZ as SCREEN_SAMPLE_HZ)


@dataclass
class CaptureConfig:
    index: int = 0
    # MSMF, not DSHOW. Measured on the C920: DirectShow ignores the FOURCC
    # request, forces YUY2 and gives 10 fps at 720p; MSMF honours MJPG and gives
    # 32. Ten fps would put a 100 ms pulse inside a single frame and break the
    # multi-frame confirmation the whole detector relies on.
    backend: str = "MSMF"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"      # MJPG is what lets a C920 hit full fps above VGA


@dataclass
class GrouperConfig:
    # One frame is enough. Requiring two was costing us twice over: a pulse the
    # sensor caught only once was discarded entirely, and when it was caught
    # twice the second frame arrived anywhere from 17 to 80 ms after the first,
    # because the trainer's diode is PWM-driven and the camera samples it
    # stroboscopically. That made the delay before a hole appeared both long and
    # VARIABLE, which is far more noticeable than a constant lag.
    #
    # What makes one frame safe is everything else that now has to agree: the
    # dot must be inside the screen quad, brighter than a measured noise floor,
    # a plausible size, not a steady light, and not within the refractory window
    # of a shot already reported.
    # Two frames, not one. At 60fps a 100ms LaserLyte pulse spans six frames and
    # real shots measured 85ms on average, so requiring two is permissive even
    # with the PWM gaps inside the pulse. Requiring ONE is what let a whole class
    # of false positives through: in one session 20% of "shots" lasted a single
    # frame, fired while nobody was shooting, and repeated at identical
    # coordinates. The cost is one frame of latency -- 16.7ms, imperceptible --
    # because the provisional announcement also waits for min_frames.
    min_frames: int = 2
    # ...but one frame THIS bright is a laser and nothing else. Set by the
    # app from the measured noise floor; 0 disables it and restores the
    # strict two-frame rule. Measured worth: 24% of real trigger pulls.
    # See `LaserDetector._confirmed` for the data.
    single_frame_peak: int = 0
    max_gap_frames: int = 2
    max_move_px: float = 40.0
    max_tracks: int = 4       # two guns, plus headroom for a reflection
    max_duration_ms: float = 400.0   # a 100 ms pulse; anything far longer is a lamp


@dataclass
class Stats:
    frames: int = 0
    fps: float = 0.0
    process_ms: float = 0.0
    read_ms: float = 0.0
    frames_with_detection: int = 0
    candidates: int = 0
    peak: int = 0
    area: float = 0.0
    shots: int = 0
    rejected_runs: int = 0
    shot_frames: list[int] = field(default_factory=list)
    shot_durations: list[float] = field(default_factory=list)
    dropped: int = 0
    recording: bool = False
    recorded_frames: int = 0
    dropped_record: int = 0   # queued too fast for the disk; see _record_frame
    # EVERY UVC CONTROL TRANSACTION THIS SESSION HAS ISSUED.
    #
    # Counted because it is the one operation measured to hang this
    # camera: a single exposure write has hung for 272 seconds, while
    # five minutes of pure streaming returned 17,945 frames with zero
    # failures. The number was never visible, so nothing noticed when a
    # new feature doubled it -- and on 2026-10-15 the repeatability gate
    # wedged the camera three times with its own traffic.
    #
    # A budget you cannot see is not a budget.
    control_writes: int = 0
    rejected_too_long: int = 0     # steady lights masquerading as very long shots
    max_dets_in_frame: int = 0     # did we ever actually see two dots at once?
    max_concurrent_tracks: int = 0
    open_tracks: int = 0
    elongation: float = 1.0        # of the brightest current dot; rises with camera angle
    # --- what the camera nearly saw -------------------------------------------
    # Rolling maxima over the last second, so a 100 ms pulse cannot fall between
    # two GUI refreshes. These exist to answer one question the app could never
    # answer before: when a shot does not register, did the camera see nothing,
    # or did it see something and throw it away?
    frame_peak: int = 0            # brightest score anywhere in frame
    dim_peak: int = 0              # brightest blob rejected for being too faint
    off_screen_peak: int = 0       # brightest blob rejected for being off the screen
    rejected_dim: int = 0          # how many, cumulative
    rejected_off_screen: int = 0
    # Blobs the size gate refused (too small for how much their region is
    # amplified), cumulative. Not "too faint": a different cause, a different
    # cure, and it used to be counted nowhere.
    rejected_amp_area: int = 0
    # Candidates folded into a much brighter nearby event's scatter halo
    # (`DetectorParams.halo_ratio`), cumulative. Its own count: a shot lost
    # this way is not "too faint" and not "too small".
    rejected_halo: int = 0
    last_detection_at: float = 0.0  # perf_counter when a dot was last seen at all
    last_shot_at: float = 0.0       # perf_counter when a shot last completed

    @property
    def detection_rate(self) -> float:
        return (self.frames_with_detection / self.frames) if self.frames else 0.0

    @property
    def mean_shot_frames(self) -> float:
        return float(np.mean(self.shot_frames)) if self.shot_frames else 0.0

    @property
    def mean_pulse_ms(self) -> float:
        return float(np.mean(self.shot_durations)) if self.shot_durations else 0.0


@dataclass
class Preview:
    image: QImage
    mask: QImage | None
    detections: list[Detection]
    stats: Stats
    frame_size: tuple[int, int]
    mask_scale: float
    # The bright region the camera can see, in full-frame pixels, computed
    # only while the user is aiming the camera. None otherwise.
    bright_box: tuple[int, int, int, int] | None = None

    @property
    def brightest(self) -> Detection | None:
        return self.detections[0] if self.detections else None


class CaptureThread(QThread):
    """Owns the camera. Talk to it only through the post_* methods."""

    opened = Signal(dict)            # negotiated format + initial control values
    failed = Signal(str)
    preview_ready = Signal(object)   # Preview
    shot_detected = Signal(object)   # Shot, final position
    shot_provisional = Signal(object)  # Shot, announced early so the UI can react now
    probe_ready = Signal(object)     # list[controls.ProbeResult]
    control_echo = Signal(str, float, bool)   # key, value read back, accepted
    log = Signal(str)
    recording_changed = Signal(bool, str)
    noise_measured = Signal(float, float, int)   # peak, 99th percentile, frames
    # The room as it is NOW, at whatever exposure is already set: frame mean
    # and chroma p99, measured exactly the way the exposure sweep measures a
    # candidate -- but with no control write at all. See
    # `start_room_measurement`.
    room_measured = Signal(float, float, int)    # mean, chroma p99, frames
    exposure_chosen = Signal(float, object)      # value, [(exposure, mean, max), ...]
    still_ready = Signal(str, object)            # tag, BGR frame (or None)

    PREVIEW_HZ = 20.0

    def __init__(
        self,
        config: CaptureConfig,
        detector_params: DetectorParams | None = None,
        grouper_config: GrouperConfig | None = None,
        preview_width: int = 960,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.preview_width = preview_width
        self._detector = LaserDetector(detector_params or DetectorParams())
        self._grouper_config = grouper_config or GrouperConfig()
        self._grouper = self._make_grouper()
        self._stats = Stats()

        self._mutex = QMutex()
        self._commands: deque[Callable[[cv2.VideoCapture], None]] = deque()
        self._running = False
        self._show_mask = False
        self._draw_overlay = True

        self._cap: cv2.VideoCapture | None = None
        # Cached at open time so nothing on a hot path has to query the driver.
        self._open_info: dict | None = None
        self._control_values: dict[str, float] = {}
        self._noise_until: float = 0.0
        self._noise_samples: list[float] = []
        self._room_until: float = 0.0
        self._room_means: list[float] = []
        self._room_chroma: list[float] = []
        self._writer: cv2.VideoWriter | None = None
        # Frames waiting to be encoded, and the thread that encodes them.
        # Bounded on purpose: see `_record_frame`.
        self._write_q: queue.Queue | None = None
        self._write_thread: threading.Thread | None = None
        self._csv_file: Any = None
        self._csv: Any = None
        # The detector's own REJECTIONS, in a file beside detections.csv and
        # not inside it: every reader of detections.csv treats any row with an
        # x as a detection, so a rejected blob in that file would silently
        # become one. Rows written so far, against REJECT_ROW_CAP.
        self._rej_file: Any = None
        self._rej: Any = None
        self._rej_rows = 0
        self._session_dir: Path | None = None
        self._fps_window: deque[float] = deque(maxlen=60)
        # (timestamp, frame peak, dim peak, off-screen peak) for the last second.
        self._peak_window: deque[tuple[float, int, int, int]] = deque(maxlen=240)
        # Per-frame near-miss telemetry for ~3 s, so a click with no shot can be
        # explained from the 0.5 s around it (plan 02 step 9). Read by the GUI
        # thread through `evidence_between`.
        self._evidence = FrameEvidence()
        # Declared here, not reached for with a getattr default in
        # `_build_preview`. It was only ever assigned in `set_aim_help`, so
        # a preview built before that method was called would have raised
        # AttributeError inside the capture thread -- the same shape as
        # `capture.stats`, which raised inside a Qt slot and stopped a self
        # test mid-run with nothing said (RULES C8).
        self._aim_help = False

    # -- command plumbing -----------------------------------------------------

    def _post(self, fn: Callable[[cv2.VideoCapture], None]) -> None:
        with QMutexLocker(self._mutex):
            self._commands.append(fn)

    def _drain(self, cap: cv2.VideoCapture) -> None:
        while True:
            with QMutexLocker(self._mutex):
                if not self._commands:
                    return
                fn = self._commands.popleft()
            try:
                fn(cap)
            except Exception as exc:                      # pragma: no cover
                # Full traceback, not a one-line repr. A swallowed exception here
                # once left recording half-started: the video file was open and
                # filling, but the UI never learned about it and every later
                # click did nothing. Never hide the stack from this path again.
                detail = traceback.format_exc()
                self.log.emit(f"COMMAND FAILED: {exc!r}" + "\n" + detail)
                print("[capture] COMMAND FAILED: " + detail, file=sys.stderr, flush=True)

    # -- public API (safe to call from the GUI thread) ------------------------

    def stop(self) -> None:
        self._running = False

    def set_detector_params(self, params: DetectorParams) -> None:
        self._post(lambda _c: self._detector.set_params(params))

    def set_grouper_config(self, cfg: GrouperConfig) -> None:
        def apply(_c: cv2.VideoCapture) -> None:
            self._grouper_config = cfg
            old = self._grouper
            self._grouper = self._make_grouper()
            # Rebuilding the tracker must not reset the session's running totals,
            # or every twiddle of a spin box would invalidate a hit-rate test.
            self._grouper._count = old.count
            self._grouper.rejected_runs = old.rejected_runs
            self._grouper.rejected_too_long = old.rejected_too_long
            self._grouper.max_concurrent = old.max_concurrent

        self._post(apply)

    def set_control(self, key: str, value: float) -> None:
        """Write one control -- unless it is already set to that value.

        `apply_preset` has skipped no-op writes since the day a launch
        fired about twenty UVC transactions in two seconds and the
        camera stopped delivering frames three seconds later, with
        NOTHING in the Windows event log. This path did not, and it is
        the one the exposure probe uses.

        Measured 2026-10-15: the probe auto-ranges its dot brightness by
        re-entering its own step function, which re-wrote the SAME
        exposure on every rung of the ladder -- up to seven redundant
        writes per launch, on top of the five the measurement actually
        needs. Across a five-launch gate run that is well over a hundred
        control transactions in eight minutes, against a camera where a
        single write has hung for 272 seconds and five minutes of pure
        streaming returned 17,945 frames with zero failures.

        The gate wedged the camera three times before this was found,
        which means the instrument was damaging the thing it measured.
        """
        def apply(cap: cv2.VideoCapture) -> None:
            if key in self._control_values:
                known = self._control_values[key]
                if known is not None and abs(float(known) - float(value)) < 1e-6:
                    self.control_echo.emit(key, float(value), True)
                    return
            accepted, back = controls.apply(cap, key, value)
            self._stats.control_writes += 1
            self._control_values[key] = back      # keep the cache honest
            self.control_echo.emit(key, back, accepted)

        self._post(apply)

    # A pause between UVC control writes. The sweep already had one, for the
    # same reason, and presets did not: measured 2026-10-15, a launch fired
    # about twenty control transactions in two seconds and the camera stopped
    # delivering frames three seconds later, with NOTHING in the Windows
    # event log -- no removal, no bus reset, no driver fault. The device
    # stayed enumerated and OK while producing nothing, which is a hang
    # inside the camera's own firmware, not something Windows did.
    #
    # Against that: 17,945 frames in five minutes with zero control writes,
    # zero read failures. Streaming is not the problem and never was.
    CONTROL_WRITE_PAUSE_S = 0.06

    def apply_preset(self, preset: controls.Preset) -> None:
        def apply(cap: cv2.VideoCapture) -> None:
            # auto_exposure must be set first or the manual exposure value is
            # simply overwritten by the auto algorithm on the next frame.
            order = ["auto_exposure", "auto_wb", "autofocus"]
            keys = order + [k for k in preset.values if k not in order]
            written = 0
            skipped = 0
            for key in keys:
                if key not in preset.values:
                    continue
                want = float(preset.values[key])
                # Skip a write that cannot change anything. The cache is
                # kept honest by every path that writes a control, and a
                # preset re-applied with the same values -- which happens
                # on every screen-search retry -- is then free.
                have = self._control_values.get(key)
                if have is not None and abs(have - want) < 1e-6:
                    skipped += 1
                    continue
                if written:
                    time.sleep(self.CONTROL_WRITE_PAUSE_S)
                # No readback: the app chose these values and showing them
                # back to itself costs a second transaction per control on
                # the one interface measured to hang this camera.
                accepted, back = controls.apply(cap, key, want,
                                                readback=False)
                self._control_values[key] = back
                self.control_echo.emit(key, back, accepted)
                self._stats.control_writes += 1
                written += 1
            self.log.emit(
                f"applying preset: {preset.name}  "
                f"({written} control write{'s' if written != 1 else ''}"
                + (f", {skipped} already set" if skipped else "") + ")")
            self._detector.reset_background()

        self._post(apply)

    def request_probe(self) -> None:
        def apply(cap: cv2.VideoCapture) -> None:
            self.log.emit("probing camera controls (nudging each value and reading it back)...")
            self.probe_ready.emit(controls.probe(cap))

        self._post(apply)

    def request_driver_dialog(self) -> None:
        def apply(cap: cv2.VideoCapture) -> None:
            ok = controls.open_driver_dialog(cap)
            self.log.emit(
                "opened the driver property sheet (DirectShow only; look for a new window)"
                if ok else "driver property sheet not available on this backend"
            )

        self._post(apply)

    def reset_background(self) -> None:
        self._post(lambda _c: self._detector.reset_background())

    def start_exposure_search(
        self,
        candidates: tuple[float, ...] = (-3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0),
        target_mean: float = 25.0,
        target_max: int = 200,
        settle_frames: int = 8,
    ) -> None:
        """Pick the exposure by measuring this room, instead of hardcoding one.

        We want the darkest setting that still leaves real dynamic range, because
        darker means the monitor fades further into the background while the
        laser -- far brighter than anything else in the room -- still saturates.
        Too dark and the sensor stops responding to the laser as well.

        A hardcoded -8 was costing us every shot: measured on this camera, frame
        maximum falls 255 -> 206 -> 40 across -6, -8, -10, and at -8 the laser no
        longer stood out. The right value depends on the room, the lighting and
        the camera, so it has to be measured, not assumed.

        Judged on MEAN brightness, not the frame maximum. A bright window or lamp
        anywhere in view saturates at nearly any exposure, so the maximum stays
        pinned at 255 and tells you nothing -- selecting on it picked a blind
        setting. The mean reflects the whole scene and falls off smoothly.

        We aim for a mean near `target_mean` rather than simply the darkest
        usable frame: dark enough that the monitor is a ghost and the laser is
        the only thing that saturates, bright enough that the sensor still
        responds to it at all.

        Biased brighter than instinct suggests, because of what the recordings
        showed: at 60 fps a 100 ms pulse should occupy 6 frames, and it occupied
        2-3, with gaps INSIDE the pulse. The trainer's diode is evidently driven
        with PWM, so a short exposure samples it stroboscopically and misses
        flashes -- and a pulse that lands in only one frame is discarded. A
        longer integration window catches more of them. With the screen mask now
        excluding everything off-monitor, the extra background brightness costs
        far less than it used to.

        The search range has to be wide because different cameras have wildly
        different response curves on the same property. Measured frame means:

            exposure      -4     -6     -7     -9    -10    -11
            C910        18.9    7.3    5.8      -      -      -
            Kiyo Pro    52.9   48.0   36.0   11.9    6.9    3.0

        The right setting is -6 on one and -9 on the other. A range tuned to the
        first camera picked -4 on the second, which was far too bright and cost
        us the shots. Note also that the Kiyo Pro WRAPS below -11 (-12 reads back
        as a bright 69.5), so the range stops there.

        Runs inside the capture loop (frames pause for a second or two) so there
        is no chance of reconfiguring the device from under a read.
        """
        def apply(cap: cv2.VideoCapture) -> None:
            self.log.emit("choosing an exposure by measurement...")
            results: list[tuple[float, float, int]] = []
            for value in candidates:
                controls.apply(cap, "exposure", value)
                self._stats.control_writes += 1
                # A beat between control writes. Back-to-back UVC writes are
                # the heaviest thing asked of this device, and a stall was
                # observed starting in the middle of a sweep.
                time.sleep(0.06)
                frame = None
                for _ in range(settle_frames):
                    ok, f = cap.read()
                    if ok:
                        frame = f
                if frame is None:
                    continue
                b, g, r = cv2.split(frame)
                chroma = cv2.subtract(r, cv2.max(g, b))
                # MEASURE INSIDE THE SCREEN, NOT ACROSS THE ROOM.
                #
                # Both bars this sweep judges on are about the MONITOR: is its
                # background still a ghost the dot can out-shine, and is it
                # quiet enough in red. Measuring them over the whole frame
                # lets anything else in the room decide the exposure.
                #
                # Measured 2026-10-15 15:05, afternoon light on the window
                # beside the monitor:
                #
                #   exposure  frame mean  passes mean<=40?
                #     -5          72         no
                #     -6          74         no
                #     -7          51         no
                #     -8          25         YES  <- chosen
                #
                # So the room, not the screen, stopped the camera down three
                # stops. The cost was measured directly on the shooter's own
                # trigger pulls, same gun, same panel, hours apart:
                #
                #   at -5: shot peak median 74, area median 332px
                #   at -8: shot peak median 18, area median 172px
                #
                # Four times dimmer, against a rejection bar that had gone UP.
                # He described it exactly: 'some shots are seen just like a
                # faint shadow and some are not seen at all'. No amount of
                # software speed recovers a photon that was never collected.
                #
                # This is the project's oldest lesson in a new place: measure
                # the quantity you care about, not one that correlates with it
                # in the room you happened to tune in. A dark room hid this
                # for a week.
                roi = self._detector._roi_image(chroma.shape[:2])
                if roi is not None and float((roi > 0).mean()) > 0.02:
                    mean = float(cv2.mean(frame, mask=roi)[0:3] and
                                 np.mean(cv2.mean(frame, mask=roi)[0:3]))
                    chroma_p99 = float(np.percentile(chroma[roi > 0], 99))
                else:
                    mean = float(frame.mean())
                    chroma_p99 = float(np.percentile(chroma, 99))
                results.append((value, mean, int(frame.max()), chroma_p99))

            chosen = self._choose_exposure(results, target_mean, target_max)

            controls.apply(cap, "exposure", chosen[0])
            self._stats.control_writes += 1
            self._control_values["exposure"] = chosen[0]
            for _ in range(settle_frames):
                cap.read()
            self._detector.reset_background()
            self.exposure_chosen.emit(float(chosen[0]), results)

        self._post(apply)

    # How much chroma noise the background may have and still count as clean.
    #
    # Raised from 8 to 22 after it cost a session. At 8 the rule stopped down to
    # -9, where the dot stopped blooming and produced NO detectable chroma at
    # all -- a whole recording scanned for red excess found nothing above 20
    # anywhere on the screen. At -7, two stops brighter, the same gun gave
    # blobs of 74 to 201 px at peaks of 23-31 and the grid calibrated with a
    # 2-6px spread.
    #
    # The point is margin over the floor, not a quiet floor: the threshold is
    # measured from whatever the floor turns out to be, so a noisier background
    # costs nothing, while a dot too small to bloom cannot be recovered by any
    # threshold. Err towards more light.
    CHROMA_CLEAN = 22.0

    # ...and a ceiling on overall brightness, which is the OTHER failure mode.
    # Chroma noise alone would happily choose -4 in a lit room, where the mean
    # is 53 and the monitor is no longer a ghost but a lit panel the dot has to
    # out-shine. On the measured lit-room curve this picks -7 (mean 36); in a
    # dark room every candidate is far below it and the chroma bar decides.
    MEAN_CEILING = 40.0

    # How far inside the bars the room has to sit before a LONGER exposure is
    # worth several control writes. A stop doubles the mean, so below half of
    # a bar there is at least a stop of headroom; 0.45 leaves a little
    # hysteresis so a marginal gain does not provoke a sweep on every launch.
    ROOM_HEADROOM = 0.45

    @staticmethod
    def room_verdict(mean: float, chroma: float,
                     headroom: float | None = None) -> str:
        """Does the exposure that produced this row still fit this room?

        Returns 'bright', 'dark' or 'ok'. The row is one sweep row -- frame
        mean and chroma p99 -- measured at whatever exposure is already set,
        which `start_room_measurement` obtains without a single control
        write. The question is only whether to SWEEP; `_choose_exposure`
        still picks the value, by measurement.

        The same two bars the sweep applies, for the same reasons:

          - over either bar -> 'bright'. The background is no longer quiet
            enough for the detector, or the monitor is no longer a ghost and
            the dot has to out-shine a lit panel. Daylight arriving.
          - far inside both -> 'dark'. A longer exposure is available, and
            the laser's brightness scales with exposure time, so the unused
            headroom is lost shots. This is the 2026-10-14 evening exactly:
            the room was darkened as far as it would go, -7 was carried in
            regardless, and -3 was four stops away.

        Kept static and free of camera access so it can be tested against the
        sweeps already recorded in the logs -- as `_choose_exposure` is.
        """
        h = CaptureThread.ROOM_HEADROOM if headroom is None else float(headroom)
        if chroma > CaptureThread.CHROMA_CLEAN or mean > CaptureThread.MEAN_CEILING:
            return 'bright'
        if (chroma < CaptureThread.CHROMA_CLEAN * h
                and mean < CaptureThread.MEAN_CEILING * h):
            return 'dark'
        return 'ok'

    @staticmethod
    def _choose_exposure(results, target_mean: float, target_max: int):
        """The longest exposure whose chroma noise is still negligible.

        Returns a (value, mean, max, chroma_p99) row. Kept static and free of
        camera access so it can be tested against recorded sweeps.
        """
        if not results:
            return (-6.0, 0.0, 0, 0.0)

        # Drop anything past the wrap point. This camera's exposure reading is
        # not monotonic below -11: a darker setting suddenly reads brighter,
        # and a rule that trusts the numbers will happily choose it. Once the
        # mean stops falling as the setting gets darker, nothing beyond is
        # meaningful.
        ordered = sorted(results, key=lambda r: -r[0])        # brightest first
        clean: list = []
        for row in ordered:
            if clean and row[1] > clean[-1][1] * 1.15 + 1.0:
                break                                          # wrapped
            clean.append(row)

        # Candidates whose background the detector would see as empty AND
        # which keep the monitor dim enough to still be a ghost.
        quiet = [r for r in clean
                 if r[3] <= CaptureThread.CHROMA_CLEAN
                 and r[1] <= CaptureThread.MEAN_CEILING]
        if quiet:
            # The longest of them: the laser scales with exposure, and margin
            # over the noise floor is the only thing that decides whether a
            # shot registers.
            return max(quiet, key=lambda r: r[0])

        # Nothing was quiet: fall back to the old luminance rule, which at
        # least keeps a lit room usable.
        usable = [r for r in clean if r[2] >= target_max]
        if usable:
            return min(usable, key=lambda r: (abs(r[1] - target_mean), -r[0]))
        return max(clean, key=lambda r: r[1])

    def request_still(self, tag: str, frames: int = 2) -> None:
        """Grab one settled frame and hand it back, tagged.

        Reads a couple and keeps the last: webcams lag a frame behind a
        change on screen, and calibration depends on photographing what is
        actually being displayed right now, not what was there a moment ago.

        TWO, NOT EIGHT. These reads are BLOCKING and they happen inside the
        command drain, so each still stops the detector for the duration --
        and, far worse on this device, hammers a camera whose firmware is
        measured to hang on driver traffic.

        The arithmetic that was never done: the exposure probe's brightness
        ladder re-photographs on every rung, so one probe issues up to a
        dozen stills. At eight reads each that is ~96 blocking reads
        back-to-back. Measured on his hardware 2026-10-16 08:59, eighteen
        seconds into a launch, immediately after the ladder:
        "WARNING: no camera frames for 3s. The camera has stalled". The
        capture thread was then stuck in the driver, the app had to let go
        of it, and letting go is what leaves the device wedged.

        Eight was chosen before any routine had an explicit settle. They
        all do now -- `Paint()` waits for real pixels and a `Sleep` covers
        the sensor's own lag -- and the capture loop reads continuously, so
        the driver buffer is already drained and the next frame is at most
        one old. The extra six reads were buying nothing and costing the
        one thing this camera cannot take.
        """
        def apply(cap: cv2.VideoCapture) -> None:
            frame = None
            for _ in range(max(1, frames)):
                ok, f = cap.read()
                if ok:
                    frame = f
            self.still_ready.emit(tag, None if frame is None else frame.copy())

        self._post(apply)

    def written_value(self, key: str) -> "float | None":
        """What THIS APP last wrote for `key`, or None if it never has.

        The one honest answer to "does the camera already have this value",
        and the only cache of it. The Director briefly kept its own, which
        went stale the moment a preset was written around it -- and a stale
        cache makes `set_control` SKIP the write that would have corrected
        the camera, which is strictly worse than writing twice on a device
        where writes are the hazard.

        Never seeded from a readback: this camera reports exposure -6
        whatever you set and auto_exposure 0.0 always, and seeding from that
        once let the first write of a session -- the one that actually
        configures the camera -- be skipped on the strength of a value the
        camera invented.
        """
        v = self._control_values.get(key)
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    def gain_at_open(self) -> "float | None":
        """The gain the CAMERA reported when it was opened, or None.

        Before this app wrote anything, so it is the device's own idea of a
        sensible sensor gain -- which is the honest meaning of the
        calibration preset's "leave gain alone". Needed because the
        shooting profile pins gain to 0, correctly, and anything that
        photographs the monitor afterwards has to put it back: measured in
        simulation, a pinned sensor took the marker difference from 201 to
        3 and `find_marker` needs 12.

        Readbacks from this camera are not trustworthy in general -- it
        reports exposure -6 whatever you set, and auto_exposure 0.0 always
        -- but gain is one it answers honestly (0 -> 255 read back as
        219 -> 253), and this value is only ever used to UNDO a pin, where
        being approximately right is enough and doing nothing is not.
        """
        info = self._open_info or {}
        ctrls = info.get("controls") or {}
        value = ctrls.get("gain")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def set_gain_map(self, gain: "np.ndarray | None") -> None:
        """Hand the detector a measured sensitivity map, on its own thread.

        Posted as a command rather than written directly: the detector is owned
        by the capture thread and is in the middle of using it on every frame.
        """
        def apply(cap: cv2.VideoCapture) -> None:
            self._detector.set_gain_map(gain)
            self.log.emit("detector sensitivity normalised across the screen"
                          if gain is not None else
                          "detector sensitivity map cleared; detection is uniform")

        self._post(apply)

    def start_room_measurement(self, seconds: float = 1.5) -> None:
        """Measure the room at the CURRENT exposure, writing nothing to the camera.

        The exposure has to follow the room, and the room follows the time of
        day: a value measured in a dark room at 21:00 is the wrong value at
        10:00 with daylight on the window beside this monitor. But a sweep is
        nine UVC control writes, and control writes are the one operation
        measured to hang this camera -- a single one has hung it for 272
        seconds -- so sweeping on every launch is not an option either.

        This is the way out, and it is cheap because it writes nothing. It
        measures the two quantities `_choose_exposure` judges a candidate on
        -- frame mean and chroma p99 -- for the exposure that is already set.
        That is one row of the sweep, obtained for free, and it is enough to
        ask whether the remembered value would still be chosen in this room.

        Deliberately the SAME arithmetic as the sweep, from the raw frame:
        `r - max(g, b)`, no background subtraction and no gain map. The
        detector's own score is tempting and would be wrong -- it is
        background-subtracted and gain-mapped, so it is a different quantity
        with the same name, and comparing it against CHROMA_CLEAN would be
        the proxy mistake this project has now made four times.

        What the caller may conclude from this is only WHETHER TO MEASURE,
        never what the exposure should be. The sweep still decides that, by
        measurement. A gate in front of a measurement cannot give a silently
        wrong answer: at worst it spends a sweep, or skips one.
        """
        def apply(_c: cv2.VideoCapture) -> None:
            self._room_until = time.perf_counter() + float(seconds)
            self._room_means = []
            self._room_chroma = []

        self._post(apply)

    def start_noise_calibration(self, seconds: float = 3.0) -> None:
        """Watch for a few seconds with nothing being shot, and learn the noise floor.

        Guessing a threshold is how you end up with a tool that reports shots
        nobody fired. This measures the brightest thing the detector sees when
        no laser is present -- room lights, screen glare, sensor noise, whatever
        is actually in frame -- and the UI then sets the threshold above it.
        """
        def apply(_c: cv2.VideoCapture) -> None:
            self._noise_until = time.perf_counter() + float(seconds)
            self._noise_samples = []
            self.log.emit(
                f"measuring noise floor for {seconds:g}s - do NOT fire during this"
            )

        self._post(apply)

    def evidence_between(self, t0: float, t1: float) -> "dict | None":
        """What the detector almost saw between two `perf_counter` times (GUI thread)."""
        return self._evidence.summarise(t0, t1)

    def reset_stats(self) -> None:
        def apply(_c: cv2.VideoCapture) -> None:
            recording, recorded = self._stats.recording, self._stats.recorded_frames
            self._stats = Stats(recording=recording, recorded_frames=recorded)
            self._grouper.reset()
            self._fps_window.clear()
            self._peak_window.clear()
            self._evidence.clear()       # rejected_runs restarts from zero

        self._post(apply)

    def set_show_mask(self, on: bool) -> None:
        self._show_mask = on

    def set_draw_overlay(self, on: bool) -> None:
        self._draw_overlay = on

    def start_recording(self, root: Path) -> None:
        self._post(lambda cap: self._open_recording(cap, root))

    def stop_recording(self) -> None:
        self._post(lambda _c: self._close_recording())

    def save_snapshot(self, root: Path) -> None:
        def apply(_c: cv2.VideoCapture) -> None:
            root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if self._last_frame is not None:
                cv2.imwrite(str(root / f"snap-{stamp}-frame.png"), self._last_frame)
            if self._detector.last_mask is not None:
                cv2.imwrite(str(root / f"snap-{stamp}-mask.png"), self._detector.last_mask)
            if self._detector.last_score is not None:
                cv2.imwrite(str(root / f"snap-{stamp}-score.png"), self._detector.last_score)
            self.log.emit(f"snapshot saved: {root / ('snap-' + stamp + '-*.png')}")

        self._post(apply)

    # -- helpers --------------------------------------------------------------

    def _make_grouper(self) -> ShotTracker:
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

    def _open_camera(self) -> cv2.VideoCapture | None:
        backends = available_backends()
        api = backends.get(self.config.backend, cv2.CAP_ANY)
        cap = cv2.VideoCapture(self.config.index, api)
        if not cap.isOpened():
            cap.release()
            return None

        # Order matters: codec, then size, then rate. Several UVC drivers reset
        # the frame size when the codec changes, so doing it the other way round
        # silently leaves you at 640x480.
        if self.config.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, str_to_fourcc(self.config.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # not honoured everywhere; costs nothing to ask
        except Exception:
            pass
        return cap

    def _negotiated(self, cap: cv2.VideoCapture) -> dict:
        """Read the real format and control values. SLOW -- call only on open.

        Every cap.get() here is a round trip to the driver. On a DirectShow
        graph that is struggling for bandwidth they block for seconds each, and
        there are fifteen of them. Calling this from the recording path once
        wedged recording for minutes: the video file opened and filled while the
        UI sat waiting for a signal that came far too late. Hence the cache.
        """
        info = {
            "index": self.config.index,
            "backend": self.config.backend,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            "requested": asdict(self.config),
            "controls": controls.read_all(cap),
        }
        self._open_info = info
        # The cache starts EMPTY, deliberately, and is filled only by
        # values this app has written itself.
        #
        # It used to be seeded from `read_all`, which is what the camera
        # SAYS its controls are -- and these notes have recorded since the
        # first week that this camera lies about readbacks:
        # CAP_PROP_EXPOSURE reads back -6 whatever you set it to, and
        # CAP_PROP_AUTO_EXPOSURE always reads 0.0. Seeding from that and
        # then skipping writes that 'match' means the first write of a
        # session -- the one that actually configures the camera -- can be
        # skipped on the strength of a value the camera invented.
        #
        # Seen live the hour it was introduced: 'applying preset: Laser
        # hunt (5 control writes, 1 already set)' on a FRESH connection,
        # where nothing had been written yet and nothing could legitimately
        # be already set.
        #
        # The saving that motivated the cache is still there: a preset
        # re-applied during a session, which is what the screen search
        # does on every retry, still costs nothing.
        self._control_values = {}
        return info

    # -- recording ------------------------------------------------------------

    # About a second and a half of video at 60fps. Deep enough to absorb a
    # disk hiccup, shallow enough that giving up on a full queue loses a
    # moment of footage rather than minutes of memory.
    RECORD_QUEUE = 90
    # A hard ceiling on rejected.csv rows per recording (~60 bytes each, so
    # ~18 MB): on top of the detector's own per-frame cap, a swarm that lasts
    # an hour cannot grow the file without bound. Reached once, logged once.
    REJECT_ROW_CAP = 300_000

    def _record_frame(self, frame) -> bool:
        """Hand a frame to the writer thread. Never blocks, never waits.

        `writer.write()` encodes a 1280x720 frame and puts it on disk, and it
        used to run INLINE in the capture loop -- so every recorded session
        was paced by the encoder rather than by the camera. Measured on this
        machine, same camera, minutes apart:

            with recording     18 - 21 fps
            without recording  29.7 fps, and the banner read 47.1

        That matters far beyond convenience. Shot confirmation counts CAMERA
        FRAMES inside a ~100ms pulse, so halving the frame rate halves the
        frames per shot and pushes pulses under the minimum. Every
        measurement of 'shots that leave no hole' was taken with recording
        ON -- the instrument was causing a large part of what it measured.

        So the frame is copied and queued, and a full queue DROPS it and says
        so. A recording with a gap in it is a nuisance; a detector paced by a
        disk is a wrong answer.
        """
        q = self._write_q
        if q is None:
            return False
        try:
            q.put_nowait(frame.copy())
            return True
        except queue.Full:
            return False

    def _write_loop(self) -> None:
        """Encode queued frames until told to stop. Owns the writer."""
        while True:
            q = self._write_q
            if q is None:
                return
            try:
                frame = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if frame is None:            # the sentinel: drain complete
                return
            w = self._writer
            if w is not None:
                try:
                    w.write(frame)
                except Exception:
                    return

    def _open_recording(self, cap: cv2.VideoCapture, root: Path) -> None:
        if self._writer is not None:
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = root / f"session-{stamp}"
        d.mkdir(parents=True, exist_ok=True)
        # From the cache, not from the driver: see _negotiated's docstring.
        info = self._open_info or {}
        w = int(info.get("width") or self.config.width)
        h = int(info.get("height") or self.config.height)
        fps = float(info.get("fps") or self.config.fps)

        # MJPG rather than something lossless: 1080p30 raw is ~180 MB/s, which
        # will drop frames on any ordinary disk, and dropped frames are exactly
        # what we must not introduce while measuring a 3-frame pulse.
        writer = cv2.VideoWriter(str(d / "capture.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        if not writer.isOpened():
            # VideoWriter reports failure only through isOpened(). Skip this and
            # recording appears to start, writes nothing, and nobody finds out
            # until the folder turns out to be empty.
            writer.release()
            self.log.emit(
                f"RECORDING FAILED: could not open a video writer for {w}x{h} @{fps:g} "
                f"at {d}. Check the MJPG codec and that the path is writable."
            )
            self.recording_changed.emit(False, "")
            return
        self._writer = writer
        self._csv_file = open(d / "detections.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        # One row per detection, so a frame holding two simultaneous dots writes
        # two rows and a frame holding none still writes one with n_dets = 0.
        self._csv.writerow(
            ["frame", "t_rel", "n_dets", "det_i", "x", "y", "peak", "area",
             "w", "h", "elongation", "process_ms"]
        )
        # The rejected blobs, as a SEPARATE file (see __init__). Failing to open
        # it must not stop recording: it is a diagnostic of a diagnostic.
        try:
            self._rej_file = open(d / "rejected.csv", "w", newline="", encoding="utf-8")
            self._rej = csv.writer(self._rej_file)
            self._rej.writerow(["frame", "t_rel", "reason", "x", "y", "peak",
                                "area", "w", "h", "elongation"])
            self._rej_rows = 0
            self._detector.record_rejects = True
        except Exception as exc:
            self._rej_file = None
            self._rej = None
            self.log.emit(f"note: could not open rejected.csv ({exc!r}); recording continues")
        # Declare recording live BEFORE writing any metadata. The UI must learn
        # about this the instant frames start going to disk; a sidecar file that
        # fails to write is a nuisance, not a reason to leave the user staring at
        # a button that never changed.
        self._session_dir = d
        self._stats.recording = True
        self._stats.recorded_frames = 0
        # Start the encoder thread only once there is something to encode.
        self._write_q = queue.Queue(maxsize=self.RECORD_QUEUE)
        self._write_thread = threading.Thread(target=self._write_loop,
                                              name='lsa-record',
                                              daemon=True)
        self._write_thread.start()
        self.recording_changed.emit(True, str(d))
        self.log.emit(f"recording to {d}")

        try:
            (d / "settings.json").write_text(
                json.dumps(
                    {
                        "camera": self._open_info,
                        "controls": self._control_values,
                        "detector": {**vars(self._detector.params),
                                     "mode": self._detector.params.mode.value},
                        "grouper": asdict(self._grouper_config),
                        "started": stamp,
                    },
                    indent=2, default=str,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.log.emit(f"note: could not write settings.json ({exc!r}); recording continues")

    def _close_recording(self) -> None:
        if self._writer is None:
            return
        d = self._session_dir
        # Drain what is queued, then stop the encoder -- but on a
        # DEADLINE. A shutdown that waits on a disk is a shutdown that
        # gets force-killed, and a force-kill is what leaves this camera
        # wedged until it is physically replugged. A truncated recording
        # is the cheaper failure by a wide margin.
        t = self._write_thread
        q = self._write_q
        if q is not None:
            try:
                q.put_nowait(None)          # sentinel
            except queue.Full:
                pass
        if t is not None:
            t.join(timeout=2.0)
        self._write_thread = None
        self._write_q = None
        self._writer.release()
        self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv = None
        self._detector.record_rejects = False
        if self._rej_file is not None:
            self._rej_file.close()
            self._rej_file = None
            self._rej = None
        if d is not None:
            summary = {
                "frames": self._stats.recorded_frames,
                "shots": self._stats.shots,
                "mean_shot_frames": self._stats.mean_shot_frames,
                "mean_pulse_ms": self._stats.mean_pulse_ms,
                "detection_rate": self._stats.detection_rate,
                "max_dets_in_frame": self._stats.max_dets_in_frame,
                "max_concurrent_tracks": self._stats.max_concurrent_tracks,
                "shot_frames": self._stats.shot_frames,
                "shot_durations_ms": self._stats.shot_durations,
            }
            (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._stats.recording = False
        self.recording_changed.emit(False, str(d) if d else "")
        self.log.emit(f"recording stopped ({self._stats.recorded_frames} frames)")
        self._session_dir = None

    # -- the loop -------------------------------------------------------------

    _last_frame: np.ndarray | None = None

    # PASSIVE SCREEN WATCH (lsa/screenwatch.py). Median luminance of the lit
    # "ink" inside the screen polygon, sampled SCREEN_SAMPLE_HZ times a second
    # from a strided view of the frame; None when there is no polygon or no ink.
    # Declared on the class and mirrored by tests/fake_camera.py (RULES C8/C12).
    screen_level: float | None = None
    screen_level_at: float = 0.0
    _screen_next: float = 0.0
    _screen_mask_key: tuple | None = None
    _screen_idx: tuple | None = None
    SCREEN_STRIDE = 8

    def _sample_screen(self, frame: np.ndarray, now: float) -> None:
        """One number per sample, no control traffic, no copy of the frame.

        A ::8 slice of a 1280x720 frame is ~14k pixels; the mask is built once
        per polygon. Cost is measured in tests/test_screenwatch.py.
        """
        poly = self._detector.params.roi_polygon
        h, w = frame.shape[:2]
        st = self.SCREEN_STRIDE
        if not poly or len(poly) < 3:
            self.screen_level, self.screen_level_at = None, now
            return
        key = (poly, h, w)
        if key != self._screen_mask_key:
            sh, sw = (h + st - 1) // st, (w + st - 1) // st
            m = np.zeros((sh, sw), dtype=np.uint8)
            pts = (np.array(poly, dtype=np.float32) / st).astype(np.int32)
            cv2.fillConvexPoly(m, pts, 255)
            self._screen_idx = np.nonzero(m)
            self._screen_mask_key = key
        gray = cv2.cvtColor(frame[::st, ::st], cv2.COLOR_BGR2GRAY)
        vals = gray[self._screen_idx]
        ink = vals[vals > SCREEN_INK_MIN_LUMA]
        self.screen_level = (float(np.median(ink))
                             if ink.size >= SCREEN_INK_MIN_PIXELS else None)
        self.screen_level_at = now

    def run(self) -> None:  # noqa: C901 - a capture loop is linear by nature
        cap = self._open_camera()
        if cap is None:
            self.failed.emit(
                f"Could not open camera index {self.config.index} on {self.config.backend}. "
                "Another application may be holding it, or the index is wrong."
            )
            return

        self._cap = cap
        self._running = True
        info = self._negotiated(cap)
        self.config.fps = info["fps"] or self.config.fps
        self._grouper = self._make_grouper()
        self.opened.emit(info)

        t_start = time.perf_counter()
        last_preview = 0.0
        frame_index = -1
        consecutive_failures = 0

        while self._running:
            self._drain(cap)

            t_read = time.perf_counter()
            ok, frame = cap.read()
            read_ms = (time.perf_counter() - t_read) * 1000.0
            now = time.perf_counter()

            if not ok or frame is None:
                consecutive_failures += 1
                self._stats.dropped += 1
                if consecutive_failures >= 30:
                    self.failed.emit("Camera stopped delivering frames (30 consecutive failures).")
                    break
                continue
            consecutive_failures = 0
            frame_index += 1
            self._last_frame = frame

            dets = self._detector.process(frame, timestamp=now)
            shots = self._grouper.feed(dets, frame_index)
            # Emit provisional hits first and before anything else in the loop:
            # this is the path the user actually perceives as responsiveness.
            for prov in self._grouper.take_provisional():
                self.shot_provisional.emit(prov)

            if self._room_until:
                # Measured on the raw frame, like the sweep. Costs a split
                # and a percentile for a second or two, once.
                rb, rg, rr = cv2.split(frame)
                rc = cv2.subtract(rr, cv2.max(rg, rb))
                self._room_means.append(float(frame.mean()))
                self._room_chroma.append(float(np.percentile(rc, 99)))
                if now >= self._room_until:
                    self._room_until = 0.0
                    n = len(self._room_means)
                    self.room_measured.emit(
                        float(np.mean(self._room_means or [0.0])),
                        # The worst of the samples, not their mean: the bar
                        # the sweep applies is a ceiling, and a room that is
                        # over it half the time is over it.
                        float(max(self._room_chroma or [0.0])), n)

            if self._noise_until:
                score = self._detector.last_score
                if score is not None:
                    self._noise_samples.append(float(score.max()))
                if now >= self._noise_until:
                    self._noise_until = 0.0
                    samples = np.array(self._noise_samples or [0.0], dtype=np.float32)
                    self.noise_measured.emit(
                        float(samples.max()), float(np.percentile(samples, 99)), samples.size
                    )

            if now >= self._screen_next:
                self._screen_next = now + 1.0 / SCREEN_SAMPLE_HZ
                self._sample_screen(frame, now)

            s = self._stats
            s.frames += 1
            s.read_ms = read_ms
            s.process_ms = self._detector.last_process_ms
            s.candidates = len(dets)
            s.max_dets_in_frame = max(s.max_dets_in_frame, len(dets))
            s.open_tracks = self._grouper.open_tracks
            s.max_concurrent_tracks = self._grouper.max_concurrent
            self._fps_window.append(now)
            if len(self._fps_window) >= 2:
                span = self._fps_window[-1] - self._fps_window[0]
                s.fps = (len(self._fps_window) - 1) / span if span > 0 else 0.0
            # A rolling one-second maximum rather than this frame's value:
            # the GUI reads these at 10 Hz and a pulse lasts 100 ms, so sampling
            # the instantaneous value would miss most shots outright.
            det = self._detector
            self._peak_window.append((now, det.last_frame_peak, det.last_dim_peak,
                                      det.last_off_screen_peak))
            while self._peak_window and now - self._peak_window[0][0] > 1.0:
                self._peak_window.popleft()
            self._evidence.push(now, det.last_frame_peak, det.last_dim_peak,
                                det.last_dim_pos, det.last_off_screen_peak,
                                self._grouper.rejected_runs,
                                amp_gate=det.last_amp_gate,
                                halo=det.last_halo,
                                big=det.last_big_rejected)
            s.frame_peak = max(e[1] for e in self._peak_window)
            s.dim_peak = max(e[2] for e in self._peak_window)
            s.off_screen_peak = max(e[3] for e in self._peak_window)
            rj = det.last_rejects
            s.rejected_dim += rj.get("dim", 0)
            s.rejected_off_screen += rj.get("roi", 0)
            s.rejected_amp_area += rj.get("amp_area", 0)
            s.rejected_halo += rj.get("halo", 0)
            if dets:
                s.frames_with_detection += 1
                s.peak = dets[0].peak
                s.area = dets[0].area
                s.elongation = dets[0].elongation
                s.last_detection_at = now
            for shot in shots:
                # Counted once, at the final report, so the totals stay right
                # even though the UI already drew the hole.
                s.shots += 1
                s.shot_frames.append(shot.frames)
                s.shot_durations.append(shot.duration_ms)
                s.last_shot_at = now
                self.shot_detected.emit(shot)
            s.rejected_runs = self._grouper.rejected_runs
            s.rejected_too_long = self._grouper.rejected_too_long

            if self._writer is not None:
                if self._record_frame(frame):
                    s.recorded_frames += 1
                else:
                    s.dropped_record += 1
                if self._csv is not None:
                    t_rel = round(now - t_start, 5)
                    pms = round(self._detector.last_process_ms, 3)
                    if not dets:
                        self._csv.writerow([frame_index, t_rel, 0, "", "", "", "", "", "", "", "", pms])
                    for i, d in enumerate(dets):
                        self._csv.writerow([
                            frame_index, t_rel, len(dets), i,
                            round(d.x, 3), round(d.y, 3), d.peak, round(d.area, 1),
                            round(d.width, 1), round(d.height, 1), round(d.elongation, 3), pms,
                        ])
                if self._rej is not None:
                    rows = self._detector.last_rejected_blobs
                    if rows:
                        if self._rej_rows >= self.REJECT_ROW_CAP:
                            if self._rej_rows == self.REJECT_ROW_CAP:
                                self._rej_rows += 1
                                self.log.emit(
                                    f"rejected.csv reached {self.REJECT_ROW_CAP} rows; "
                                    f"no more rejected blobs will be recorded this session")
                        else:
                            t_rej = round(now - t_start, 5)
                            for (why, rx, ry, rpk, rar, rw, rh, rel) in rows:
                                self._rej.writerow([
                                    frame_index, t_rej, why, round(float(rx), 3),
                                    round(float(ry), 3), int(rpk), round(float(rar), 1),
                                    round(float(rw), 1), round(float(rh), 1),
                                    round(float(rel), 3)])
                            self._rej_rows += len(rows)

            if now - last_preview >= 1.0 / self.PREVIEW_HZ:
                last_preview = now
                self.preview_ready.emit(self._build_preview(frame, dets))

        self._close_recording()
        cap.release()
        self._cap = None

    # -- preview --------------------------------------------------------------

    def set_aim_help(self, on: bool) -> None:
        """Compute a live outline of the screen for the aiming guides.

        The flash-based screen search is accurate but costs a full black/white
        cycle on the monitor, so it cannot run continuously -- which makes its
        result useless as live feedback: the outline stays where the screen
        WAS while the user moves the camera. Finding the brightest large blob
        in each frame is cruder but updates every frame, and crude is exactly
        what aiming needs.
        """
        self._aim_help = bool(on)

    @staticmethod
    def _bright_region(frame: np.ndarray) -> "tuple[int, int, int, int] | None":
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA)
        # Relative to the brightest thing present, because the absolute level
        # depends entirely on an exposure chosen for hunting a laser.
        hi = float(small.max())
        if hi < 12:
            return None                       # nothing bright enough to be a screen
        _, mask = cv2.threshold(small, max(8.0, hi * 0.45), 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if bw * bh < 40:
            return None                       # a speck, not a monitor
        fh, fw = frame.shape[:2]
        sx, sy = fw / 160.0, fh / 90.0
        return (int(x * sx), int(y * sy), int(bw * sx), int(bh * sy))

    def _build_preview(self, frame: np.ndarray, dets: list[Detection]) -> Preview:
        h, w = frame.shape[:2]
        scale = min(1.0, self.preview_width / float(w))
        disp = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()

        if self._draw_overlay:
            # Brightest in green, the rest in amber: with two guns up it should be
            # obvious at a glance whether both dots are actually being seen.
            for i, det in enumerate(dets):
                colour = (0, 255, 0) if i == 0 else (0, 190, 255)
                cx, cy = int(det.x * scale), int(det.y * scale)
                cv2.circle(disp, (cx, cy), 14, colour, 1, cv2.LINE_AA)
                cv2.line(disp, (cx - 22, cy), (cx - 6, cy), colour, 1, cv2.LINE_AA)
                cv2.line(disp, (cx + 6, cy), (cx + 22, cy), colour, 1, cv2.LINE_AA)
                cv2.line(disp, (cx, cy - 22), (cx, cy - 6), colour, 1, cv2.LINE_AA)
                cv2.line(disp, (cx, cy + 6), (cx, cy + 22), colour, 1, cv2.LINE_AA)
                cv2.putText(disp, f"{det.peak}/{det.area:.0f}px/{det.elongation:.1f}:1",
                            (cx + 18, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

        image = QImage(disp.data, disp.shape[1], disp.shape[0], disp.strides[0],
                       QImage.Format_BGR888).copy()

        mask_img = None
        mask_scale = 1.0
        if self._show_mask and self._detector.last_mask is not None:
            m = self._detector.last_mask
            ms = min(1.0, 320.0 / float(m.shape[1]))
            small = cv2.resize(m, None, fx=ms, fy=ms, interpolation=cv2.INTER_NEAREST) if ms < 1.0 else m
            small = np.ascontiguousarray(small)
            mask_img = QImage(small.data, small.shape[1], small.shape[0], small.strides[0],
                              QImage.Format_Grayscale8).copy()
            mask_scale = ms

        bright = self._bright_region(frame) if self._aim_help else None
        return Preview(image, mask_img, list(dets), self._stats, (w, h), mask_scale,
                       bright_box=bright)


# --- format probing ----------------------------------------------------------

COMMON_MODES: tuple[tuple[int, int], ...] = (
    (320, 240), (640, 360), (640, 480), (800, 600), (1024, 576),
    (1280, 720), (1280, 960), (1600, 896), (1920, 1080),
)


@dataclass
class ModeResult:
    width: int
    height: int
    fourcc: str
    requested_fps: float
    actual_fps: float
    measured_fps: float
    ok: bool
    note: str = ""


class ModeScanThread(QThread):
    """Opens the camera once per candidate format and times real delivery.

    What a UVC camera reports via CAP_PROP_FPS and what it actually delivers are
    different claims, and the gap is usually the whole story: a C920 asked for
    1080p30 in YUY2 will happily say 30 and then hand you 5. We time it.
    """

    progress = Signal(int, int, str)
    result = Signal(object)      # ModeResult
    finished_scan = Signal()

    def __init__(self, index: int, backend: str, fourccs: tuple[str, ...] = ("MJPG", "YUY2"),
                 fps_targets: tuple[float, ...] = (30.0, 60.0), measure_seconds: float = 1.0,
                 parent: Any = None) -> None:
        super().__init__(parent)
        self.index = index
        self.backend = backend
        self.fourccs = fourccs
        self.fps_targets = fps_targets
        self.measure_seconds = measure_seconds
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        api = available_backends().get(self.backend, cv2.CAP_ANY)
        jobs = [(w, h, fc, f) for fc in self.fourccs for (w, h) in COMMON_MODES for f in self.fps_targets]
        for n, (w, h, fc, target) in enumerate(jobs, 1):
            if self._abort:
                break
            self.progress.emit(n, len(jobs), f"{w}x{h} {fc} @{target:g}")
            cap = cv2.VideoCapture(self.index, api)
            if not cap.isOpened():
                cap.release()
                self.result.emit(ModeResult(w, h, fc, target, 0, 0, False, "could not open"))
                continue
            cap.set(cv2.CAP_PROP_FOURCC, str_to_fourcc(fc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, target)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps = float(cap.get(cv2.CAP_PROP_FPS))
            afc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

            for _ in range(5):               # let auto-anything settle
                cap.read()
            t0 = time.perf_counter()
            count = 0
            while time.perf_counter() - t0 < self.measure_seconds:
                ok, _f = cap.read()
                if not ok:
                    break
                count += 1
            measured = count / max(1e-6, time.perf_counter() - t0)
            cap.release()

            note = ""
            if (aw, ah) != (w, h):
                note = f"got {aw}x{ah}"
            if afc != fc.strip():
                note = (note + "; " if note else "") + f"codec {afc}"
            self.result.emit(ModeResult(aw, ah, afc, target, afps, measured, count > 0, note))

        self.finished_scan.emit()
