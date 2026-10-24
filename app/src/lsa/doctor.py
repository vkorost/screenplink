"""The camera doctor: which camera on THIS machine is fit to shoot with?

    screenplink.exe --doctor --all               test every real camera, rank them
    screenplink.exe --doctor --index N           test one camera
    screenplink.exe --doctor --index N --frame aim.png
                                                 save one picture and nothing else
    python -m lsa.doctor --all                   the same, from source

WHY THIS IS A MODULE AND NOT JUST tools/camera_health.py
--------------------------------------------------------
The release ships no tools/ folder, and a stranger's first instruction is to run
this. `tools/camera_health.py` is now a thin front door onto this module: ONE
implementation, so the two cannot drift (a second copy of the lit-display test
and of the gain trap would be exactly how this project has repeated its faults).

NOTHING HERE MAY ASSUME A CAMERA. It runs on machines this project has never
seen, with cameras it has never met. No name, index, model, resolution or
exposure scale is hard-coded as a fact about the hardware; every camera is
discovered, tested by what it DOES, and a picture is saved so a person (or
Claude) can look at what it sees. Indices and names are not stable here (RULES
A11): the same physical camera has been index 1, then 2, under three names in
one morning.

THE ORDER OF THE CHECKS IS THE LESSON (CLAUDE.md, "tools/camera_health.py"):

  1. does it open, does it deliver frames
  2. frame rate -- with the exposure FIXED. On auto-exposure in a dim room the
     camera lengthens its integration until the rate falls to meet it, so the
     number tracks the room, not the device: this tool once reported 28.3,
     28.7 and 51.2 fps for one camera within minutes (RULES A13).
  3. does changing the exposure CHANGE THE IMAGE? Asked before anything that
     depends on exposure, which is everything downstream. A camera that accepts
     the write and ignores it makes every later number lie in a way that looks
     like a different fault.
  4. chroma noise floor: the bar a laser hit has to beat
  5. can the camera see the monitor flash
  6. SAVE A FRAME AND LOOK AT IT. One glance answers "is it even pointed at the
     monitor", which every indirect measurement failed to do for an hour.

TWO TRAPS THIS TOOL HAS ALREADY FALLEN INTO, kept out on purpose:

  * It pinned gain to 0 to measure the noise floor and did not put it back, so
    the flash check ran on a crushed image: difference 2.0 instead of 46.8.
    The same trap is recorded twice elsewhere in this project. Gain is now
    restored in a `finally`, and is pinned only when there is a credible value
    to restore; otherwise it is left alone and the report says so.
  * A SLEEPING DISPLAY LOOKS EXACTLY LIKE A BLIND CAMERA. Both give a flash
    difference near zero and need opposite responses. If the "white" frame never
    got bright, the display is asleep, and the report says that instead of
    blaming the camera (three launches were lost this way on 2026-10-18).

CONTROL WRITES ARE THE HAZARD (RULES A4, A8). One exposure write has hung this
camera family for 272 seconds. So every write goes through `Controls`, which
spaces them, skips ones whose value this process already wrote, counts them,
and reports the count. A per-camera watchdog exists because "a wait must have a
deadline" (RULES D4): if the driver hangs, the doctor says so and leaves, it
does not sit there looking like a slow test.

Do not run this while the app is open: two processes on one camera can leave
the Windows driver in a bad state (RULES A7).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import _env  # noqa: F401  (MSMF hw transforms off -- BEFORE cv2; 73.8s -> 0.43s)

import cv2
import numpy as np

from .camera.devices import VIRTUAL_NAME_PATTERNS, CameraInfo, list_cameras

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"

# --- the format the app asks for. A request, not a claim about any camera. ---
WIDTH, HEIGHT, FPS = 1280, 720, 60

# --- thresholds, each measured on the development hardware (CLAUDE.md) -------
OPEN_SLOW_S = 5.0            # with hw transforms off a camera opens in under 1s
FPS_WINDOW_S = 2.5
FPS_MIN_FRACTION = 0.8       # of what was asked for
EXPOSURES = (-3.0, -5.0, -7.0, -9.0, -11.0)   # never below -11: it WRAPS there
EXPOSURE_FOR_FPS = -6.0      # fixed, so the rate is the device's and not the room's
EXPOSURE_FOR_NOISE = -11.0   # the darkest of the sweep, so no new write is needed
EXPOSURE_FOR_FLASH = -10.0   # dark and FIXED: the monitor must be the only change
SPREAD_FAIL_PCT = 25.0       # a healthy camera moves 80%+
FLOOR_PASS, FLOOR_WARN = 40.0, 80.0
FLASH_FAIL, FLASH_HEALTHY = 18.0, 40.0
SETTLE_S = 0.7               # after a control write, before trusting a frame
WRITE_GAP_S = 0.06           # between control writes (RULES A4)
CAMERA_DEADLINE_S = 120.0    # one camera, start to finish; the sweep is ~15s
DARK_MEAN = 3.0              # every exposure this dark = nothing to see

STEPS = ("open", "frames", "fps", "exposure", "noise", "flash", "look")

WHY = {
    "open": "nothing else means anything if it will not open",
    "fps": "a laser pulse lasts ~100 ms; the app confirms a shot in more than one "
           "frame, so 60 fps is ~6 frames per pulse and 30 fps is ~3. Measured "
           "with the exposure FIXED, or the number tracks the room",
    "exposure": "asked BEFORE anything that depends on exposure, which is "
                "everything downstream; a camera that ignores it makes later "
                "numbers lie",
    "noise": "the bar a laser hit has to beat (a real hit measures 150-230 here)",
    "flash": "can it see the monitor change; a sleeping display and a blind "
             "camera read identically and need opposite responses",
    "look": "one glance answers 'is it pointed at the monitor', which no number can",
}


def say(text: str = "") -> None:
    print(text, flush=True)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class Check:
    state: str
    title: str
    detail: str = ""


@dataclass
class CameraReport:
    index: int
    name: str
    virtual: bool = False
    skipped: str = ""                  # why it was not tested, if it was not
    checks: list[Check] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)   # the order they RAN in
    opened: bool = False
    delivered: bool = False
    aborted: str = ""                  # an error or a hang, in words
    fps: float | None = None
    spread_pct: float | None = None
    floor: float | None = None
    flash_diff: float | None = None
    display_lit: bool | None = None
    writes: int = 0
    write_log: list[tuple[str, float]] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)  # PNG paths, absolute

    @property
    def fails(self) -> list[Check]:
        return [c for c in self.checks if c.state == FAIL]

    @property
    def warns(self) -> list[Check]:
        return [c for c in self.checks if c.state == WARN]

    @property
    def result(self) -> str:
        if self.skipped:
            return "SKIPPED"
        if self.aborted or not self.opened or not self.delivered:
            return "UNUSABLE"
        if self.fails:
            return "POOR"
        if self.warns:
            return "USABLE"
        return "HEALTHY"

    @property
    def reason(self) -> str:
        """One line a person can act on: what was measured, or what failed."""
        if self.skipped:
            return self.skipped
        if self.aborted:
            return self.aborted
        if self.fails:
            return "; ".join(f"{c.title}: {c.detail}" if c.detail else c.title
                             for c in self.fails)
        bits = []
        if self.fps is not None:
            bits.append(f"{self.fps:.0f} fps")
        if self.spread_pct is not None:
            bits.append(f"exposure moves the image {self.spread_pct:.0f}%")
        if self.floor is not None:
            bits.append(f"noise floor {self.floor:.0f}")
        if self.flash_diff is not None:
            bits.append(f"flash difference {self.flash_diff:.0f}")
        text = ", ".join(bits)
        if self.warns:
            text += "; but " + "; ".join(c.title for c in self.warns)
        return text


# ---------------------------------------------------------------------------
# the one door to the camera's controls
# ---------------------------------------------------------------------------
class Controls:
    """Every control write goes through here (RULES A4, A6).

    Spaced, counted, and skipped when THIS PROCESS already wrote that value. The
    cache is never seeded from a readback: these cameras report exposure -6.0
    and auto-exposure 0.0 while being neither (CLAUDE.md, 2026-10-13).
    """

    def __init__(self, cap, sleep: Callable[[float], None] = time.sleep,
                 gap: float = WRITE_GAP_S) -> None:
        self.cap = cap
        self._sleep = sleep
        self._gap = gap
        self._mine: dict[str, float] = {}
        self.writes = 0
        self.log: list[tuple[str, float]] = []

    def set(self, name: str, prop: int, value: float) -> bool:
        if self._mine.get(name) == value:
            return False
        self._sleep(self._gap)
        self.cap.set(prop, value)
        self._mine[name] = value
        self.writes += 1
        self.log.append((name, value))
        return True


# ---------------------------------------------------------------------------
# small measurements
# ---------------------------------------------------------------------------
def grab(cap, n: int = 10):
    """The last good frame of n reads, or None."""
    frame = None
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    return frame


def chroma_p99(frame) -> float:
    """Red minus the stronger of green and blue, 99th percentile: the quantity
    the laser detector scores, not a proxy for it (RULES C6)."""
    ch = frame[:, :, 2].astype(np.int16) - np.maximum(frame[:, :, 1],
                                                      frame[:, :, 0]).astype(np.int16)
    return float(np.percentile(ch, 99))


def open_msmf(index: int, width: int = WIDTH, height: int = HEIGHT, fps: int = FPS):
    """MSMF, MJPG requested FIRST (the order matters), then size and rate.

    MSMF because DirectShow ignores the codec request and delivers 10 fps at
    720p on the cameras this was measured on. lsa._env has already switched off
    MSMF hardware transforms; without that, some cameras take over a minute.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def virtual_pattern(name: str) -> str:
    """The first known software-camera pattern the name matches, or ''."""
    for pat in VIRTUAL_NAME_PATTERNS:
        if re.search(pat, name, re.IGNORECASE):
            return pat
    return ""


# ---------------------------------------------------------------------------
# the flash, behind a seam so a test can supply a stand-in
# ---------------------------------------------------------------------------
class QtFlasher:
    """Fullscreen black/white on a monitor, through the app's own FlashWindow.

    FlashWindow closes on Escape, on a double click, and on its own deadline, so
    this cannot trap anyone behind it (RULES B6). Qt is imported lazily: a run
    with --no-flash never needs it.
    """

    def __init__(self, screen_index: int | None = None) -> None:
        self.screen_index = screen_index
        self.cancelled = ""
        self.in_front: bool | None = None
        self._app = None
        self._win = None

    def begin(self) -> None:
        from PySide6.QtWidgets import QApplication
        from .calibrate.flash import FlashWindow
        self._app = QApplication.instance() or QApplication([])
        screens = self._app.screens()
        if self.screen_index is not None and 0 <= self.screen_index < len(screens):
            screen = screens[self.screen_index]
        else:
            screen = self._app.primaryScreen()
        self.cancelled = ""
        self._win = FlashWindow()
        self._win.cancelled.connect(self._on_cancel)
        self._win.set_colour("black")
        self._win.show_on(screen, timeout_s=20.0)
        try:
            from .display.foreground import force_to_front
            self.in_front = force_to_front(self._win)
        except Exception:                                    # noqa: BLE001
            self.in_front = None

    def _on_cancel(self, why: str) -> None:
        self.cancelled = why

    def show(self, colour: str) -> None:
        self._win.set_colour(colour)

    def settle(self, ms: int) -> None:
        end = time.time() + ms / 1000.0
        while time.time() < end:
            self._app.processEvents()
            time.sleep(0.02)

    def end(self) -> None:
        if self._win is not None:
            self._win.close()
            self._win = None
        if self._app is not None:
            self._app.processEvents()


# ---------------------------------------------------------------------------
# one camera
# ---------------------------------------------------------------------------
def examine(index: int, name: str, *,
            opener: Callable = open_msmf,
            flasher=None,
            out_dir: Path,
            sleep: Callable[[float], None] = time.sleep,
            clock: Callable[[], float] = time.perf_counter,
            width: int = WIDTH, height: int = HEIGHT, fps: int = FPS,
            out: Callable[[str], None] = say) -> CameraReport:
    """Run the six checks, in order, on one camera. Never raises."""
    rep = CameraReport(index=index, name=name, virtual=bool(virtual_pattern(name)))
    cap = None
    ctl: Controls | None = None

    def record(state: str, title: str, detail: str = "") -> None:
        rep.checks.append(Check(state, title, detail))
        out(f"  [{state}] {title}" + (f" -- {detail}" if detail else ""))

    def begin(step: str, header: str) -> None:
        rep.steps.append(step)
        out(f"\n=== {header} ===")
        if step in WHY:
            out(f"    why here: {WHY[step]}")

    try:
        # ---- 1. open, frames ------------------------------------------------
        begin("open", "1. does it open and deliver frames?")
        t0 = clock()
        cap = opener(index, width, height, fps)
        open_s = clock() - t0
        if not cap.isOpened():
            record(FAIL, "camera opens", "VideoCapture refused to open it (is the "
                   "app, or another program, already using it?)")
            return rep
        rep.opened = True
        record(PASS if open_s < OPEN_SLOW_S else WARN, "camera opens", f"{open_s:.2f}s")

        # The gain the camera reported BEFORE this process wrote anything.
        prior_gain = float(cap.get(cv2.CAP_PROP_GAIN))

        rep.steps.append("frames")
        view = grab(cap, 20)
        if view is None:
            record(FAIL, "delivers frames", "opened, but no frame ever arrived")
            return rep
        rep.delivered = True
        record(PASS, "delivers frames", f"{view.shape[1]}x{view.shape[0]}")

        ctl = Controls(cap, sleep)

        # ---- 2. frame rate, exposure FIXED ---------------------------------
        begin("fps", "2. frame rate (exposure fixed first)")
        ctl.set("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)   # manual
        ctl.set("exposure", cv2.CAP_PROP_EXPOSURE, EXPOSURE_FOR_FPS)
        sleep(SETTLE_S)
        for _ in range(15):
            cap.read()
        n, t0 = 0, clock()
        for _ in range(4000):                      # bounded even if the clock stalls
            if clock() - t0 >= FPS_WINDOW_S:
                break
            ok, _f = cap.read()
            n += int(bool(ok))
        elapsed = max(clock() - t0, 1e-6)
        rep.fps = n / elapsed
        per_pulse = rep.fps / 10.0
        record(PASS if rep.fps >= fps * FPS_MIN_FRACTION else WARN, "frame rate",
               f"{rep.fps:.1f} fps (asked for {fps:g}); a 100 ms laser pulse spans "
               f"about {per_pulse:.0f} frames. Only believe this if check 3 passes")

        # ---- 3. does exposure change the image? ----------------------------
        begin("exposure", "3. does exposure actually change the image?")
        means: list[float] = []
        for e in EXPOSURES:
            ctl.set("exposure", cv2.CAP_PROP_EXPOSURE, e)
            sleep(SETTLE_S)
            f = grab(cap, 10)
            m = float(f.mean()) if f is not None else -1.0
            means.append(m)
            out(f"       exposure {e:5.0f} -> mean {m:6.1f}")
        valid = [m for m in means if m >= 0]
        if len(valid) < 2:
            record(FAIL, "exposure control", "no frames arrived during the sweep")
        else:
            lo, hi = min(valid), max(valid)
            rep.spread_pct = 100.0 * (hi - lo) / max(hi, 1.0)
            if hi < DARK_MEAN:
                record(FAIL, "exposure control",
                       "the image is black at EVERY exposure: a lens cap or blind "
                       "over the lens, or a camera that is disabled")
            elif rep.spread_pct < SPREAD_FAIL_PCT:
                record(FAIL, "exposure control",
                       f"only {rep.spread_pct:.0f}% spread -- the camera is accepting "
                       f"exposure writes and IGNORING them, so every number after "
                       f"this (and check 2) is suspect. Look for anything automatic "
                       f"in the maker's control panel and Windows camera settings: "
                       f"low-light compensation, HDR, auto-exposure priority")
            else:
                record(PASS, "exposure control",
                       f"{rep.spread_pct:.0f}% spread, darkest mean {lo:.0f}")

        # ---- 4. chroma noise floor -----------------------------------------
        begin("noise", "4. chroma noise floor (what the laser must beat)")
        ctl.set("exposure", cv2.CAP_PROP_EXPOSURE, EXPOSURE_FOR_NOISE)
        pinned = False
        if prior_gain > 0:
            # Pin gain to 0 like the shooting profile -- and ALWAYS put it back.
            # An unrestored gain crushes the flash check (46.8 -> 2.0).
            try:
                ctl.set("gain", cv2.CAP_PROP_GAIN, 0.0)
                pinned = True
                sleep(SETTLE_S + 0.1)
                f = grab(cap, 12)
            finally:
                ctl.set("gain", cv2.CAP_PROP_GAIN, prior_gain)
        else:
            sleep(SETTLE_S + 0.1)
            f = grab(cap, 12)
        rep.floor = chroma_p99(f) if f is not None else 999.0
        brightness = float(f.mean()) if f is not None else -1.0
        if pinned:
            how = "gain pinned to 0 and then restored"
        elif prior_gain == 0:
            how = "gain was already 0"
        else:
            how = ("gain not reported by this camera, so it was LEFT ALONE; the floor "
                   "is the camera's own and may read higher than in the app")
        record(PASS if rep.floor < FLOOR_PASS else (WARN if rep.floor < FLOOR_WARN else FAIL),
               "chroma noise floor",
               f"99th percentile {rep.floor:.0f} (a real hit measures 150-230), "
               f"image mean {brightness:.0f}; {how}")

        # ---- 5. the flash --------------------------------------------------
        dark = None
        if flasher is None:
            rep.steps.append("flash")
            out("\n=== 5. can it see the monitor flash? ===")
            record(INFO, "flash check skipped", "--no-flash")
        else:
            begin("flash", "5. can it see the monitor flash?")
            from .calibrate.screen import display_is_lit, find_screen
            ctl.set("exposure", cv2.CAP_PROP_EXPOSURE, EXPOSURE_FOR_FLASH)
            black = white = None
            try:
                flasher.begin()
                flasher.settle(1000)
                flasher.show("black")
                flasher.settle(1100)
                black = grab(cap, 12)
                flasher.show("white")
                flasher.settle(1100)
                white = grab(cap, 12)
                dark = black
            except Exception as exc:                          # noqa: BLE001
                record(WARN, "flash check could not run",
                       f"{type(exc).__name__}: {exc} (no display available?)")
            finally:
                try:
                    flasher.end()
                except Exception:                             # noqa: BLE001
                    pass
            if black is not None and white is not None:
                rep.flash_diff = float(cv2.absdiff(white, black).mean())
                rep.display_lit = display_is_lit(white)
                if flasher.cancelled:
                    record(WARN, "flash was cancelled",
                           f"({flasher.cancelled}) the numbers below may be from a half flash")
                if not rep.display_lit:
                    record(WARN, "screen flash visible",
                           f"difference {rep.flash_diff:.1f}, and the 'white' frame "
                           f"never got bright (max {int(white.max())}). The DISPLAY is "
                           f"probably asleep, or this camera is looking at a different "
                           f"monitor from the one flashed -- wake it (move the mouse), "
                           f"or pass --screen N, and run again. This is NOT evidence "
                           f"against the camera")
                else:
                    record(PASS if rep.flash_diff >= FLASH_FAIL else FAIL,
                           "screen flash visible",
                           f"difference {rep.flash_diff:.1f} ({FLASH_HEALTHY:.0f}+ is "
                           f"healthy; under {FLASH_FAIL:.0f} means the camera is not "
                           f"seeing this screen change: aimed elsewhere, or pointed at "
                           f"a monitor other than the one flashed)")
                quad = find_screen(white, black)
                if quad is None:
                    record(WARN if not rep.display_lit else FAIL, "screen outline found",
                           "no quadrilateral could be fitted"
                           + ("; expected, the display was not lit" if not rep.display_lit else ""))
                else:
                    record(PASS if quad.confidence >= 0.75 else WARN, "screen outline found",
                           f"covers {quad.coverage * 100:.1f}% of the frame, confidence "
                           f"{quad.confidence:.2f}"
                           + (f", CLIPPED on {', '.join(quad.touches_edges)}"
                              if quad.touches_edges else ""))

        # ---- 6. look at it --------------------------------------------------
        begin("look", "6. look at it")
        ctl.set("exposure", cv2.CAP_PROP_EXPOSURE, EXPOSURE_FOR_FLASH)
        if dark is None:
            sleep(SETTLE_S)
            dark = grab(cap, 12)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"doctor_cam{index}"
        for label, img in (("view", view), ("dark", dark)):
            path = (out_dir / f"{tag}_{label}.png").resolve()
            if img is not None and cv2.imwrite(str(path), img):
                rep.frames.append(str(path))
        if rep.frames:
            record(PASS, "frames saved", "  ".join(rep.frames))
            out("       OPEN THEM. *_view.png is what the camera sees as it comes up;")
            out("       *_dark.png is the dark fixed exposure the app shoots at. One glance")
            out("       answers 'is it pointed at the monitor', which no number here can.")
        else:
            record(FAIL, "frames saved", "could not write the frame")
    except Exception as exc:                                  # noqa: BLE001
        rep.aborted = f"doctor error on this camera: {type(exc).__name__}: {exc}"
        record(FAIL, "doctor error", f"{type(exc).__name__}: {exc}")
    finally:
        if ctl is not None:
            rep.writes = ctl.writes
            rep.write_log = list(ctl.log)
        if cap is not None:
            try:
                cap.release()
            except Exception:                                 # noqa: BLE001
                pass
    if rep.writes:
        out(f"\n       control writes this camera: {rep.writes}. The camera was left on "
            f"a MANUAL exposure; the app sets its own on launch.")
    return rep


# ---------------------------------------------------------------------------
# many cameras, ranked
# ---------------------------------------------------------------------------
TIER = {"HEALTHY": 0, "USABLE": 1, "POOR": 2, "UNUSABLE": 3}


def rank(reports: list[CameraReport]) -> list[CameraReport]:
    """Tested cameras only, best first: result tier, then fewest failures and
    warnings, then the higher frame rate. Never decided by name or index."""
    tested = [r for r in reports if not r.skipped]
    return sorted(tested, key=lambda r: (TIER[r.result], len(r.fails), len(r.warns),
                                         -(r.fps or 0.0), r.index))


def _fmt(v: float | None, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def format_summary(reports: list[CameraReport]) -> list[str]:
    lines = ["", "=== summary, best first ===",
             f"  {'rank':<5}{'index':<7}{'result':<10}{'fps':>6}  {'exp%':>5}  "
             f"{'floor':>5}  {'flash':>6}  camera"]
    ranked = rank(reports)
    for i, r in enumerate(ranked, 1):
        lines.append(f"  {i:<5}{r.index:<7}{r.result:<10}{_fmt(r.fps, '.1f'):>6}  "
                     f"{_fmt(r.spread_pct, '.0f'):>5}  {_fmt(r.floor, '.0f'):>5}  "
                     f"{_fmt(r.flash_diff, '.1f'):>6}  {r.name}")
    for r in reports:
        if r.skipped:
            lines.append(f"  {'-':<5}{r.index:<7}{'SKIPPED':<10}  {r.name} -- {r.skipped}")
    lines.append("")
    best = ranked[0] if ranked else None
    if best is None or best.result == "UNUSABLE":
        lines.append("RECOMMENDED: none. No camera was fit to use. "
                     + (best.reason if best else "No real camera was found."))
    else:
        lines.append(f"RECOMMENDED: index {best.index} ({best.name}) -- {best.result}: "
                     f"{best.reason}")
        if best.result == "POOR":
            lines.append("  Every camera had a failure. 'POOR' can simply mean 'not aimed "
                         "at the monitor': look at the PNGs before deciding.")
    if any(r.display_lit is False for r in ranked):
        lines.append("NOTE: the display looked ASLEEP during at least one flash. Wake it "
                     "and run again; the ranking is provisional.")
    saved = [p for r in ranked for p in r.frames]
    if saved:
        lines.append("")
        lines.append("NOW LOOK AT THE PICTURES. The rank says which camera behaves; only")
        lines.append("the picture says which one is pointed at the monitor:")
        lines.extend(f"  {p}" for p in saved)
    return lines


class Watchdog:
    """A deadline on one camera, because a driver call can hang and a hang that
    looks like a slow test costs a person minutes (RULES D4). On expiry it calls
    `on_expire(label)`; the CLI's handler prints what it has and leaves."""

    def __init__(self, on_expire: Callable[[str], None]) -> None:
        self._on_expire = on_expire
        self._timer: threading.Timer | None = None

    def arm(self, seconds: float, label: str) -> None:
        self.disarm()
        self._timer = threading.Timer(seconds, self._on_expire, args=(label,))
        self._timer.daemon = True
        self._timer.start()

    def disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def run_all(cams: list[CameraInfo], *, make_flasher: Callable[[], object | None],
            out_dir: Path, opener: Callable = open_msmf, watchdog: Watchdog | None = None,
            deadline_s: float = CAMERA_DEADLINE_S, reports: list[CameraReport] | None = None,
            out: Callable[[str], None] = say, **kw) -> list[CameraReport]:
    """Every camera in turn. Virtual ones are skipped, with the reason."""
    reports = reports if reports is not None else []
    for cam in cams:
        pat = virtual_pattern(cam.name)
        if cam.is_virtual or pat:
            why = (f"flagged virtual (its name matches '{pat or 'a known pattern'}'). "
                   f"Software cameras open fine and deliver frames of nothing, "
                   f"sometimes with a saturated red X that the detector would hunt. "
                   f"Test it anyway with --index {cam.index} if you are sure")
            reports.append(CameraReport(cam.index, cam.name, virtual=True, skipped=why))
            out(f"\n--- index {cam.index}: {cam.name}: SKIPPED -- {why}")
            continue
        out(f"\n########## index {cam.index}: {cam.name} ##########")
        if watchdog is not None:
            watchdog.arm(deadline_s, f"index {cam.index} ({cam.name})")
        try:
            rep = examine(cam.index, cam.name, opener=opener, flasher=make_flasher(),
                          out_dir=out_dir, out=out, **kw)
        finally:
            if watchdog is not None:
                watchdog.disarm()
        reports.append(rep)
    return reports


# ---------------------------------------------------------------------------
# --frame: one picture, no tests, no exposure writes
# ---------------------------------------------------------------------------
def save_frame(index: int, path: Path, *, opener: Callable = open_msmf,
               warmup: int = 30, out: Callable[[str], None] = say) -> int:
    """Open, let auto-exposure settle, save ONE frame, release. No control
    writes at all, so it is safe to run repeatedly while aiming (RULES A4)."""
    cap = opener(index, WIDTH, HEIGHT, FPS)
    try:
        if not cap.isOpened():
            out(f"FAIL: camera {index} would not open (is the app, or another "
                f"program, using it?)")
            return 1
        frame = grab(cap, warmup)
        if frame is None:
            out(f"FAIL: camera {index} opened but delivered no frame")
            return 1
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            out(f"FAIL: could not write {path}")
            return 1
        out(f"FRAME SAVED: {path}")
        out(f"  {frame.shape[1]}x{frame.shape[0]}, mean brightness {frame.mean():.0f}")
        if frame.mean() < DARK_MEAN * 3:
            out("  This frame is nearly black: a lens cap or blind, the wrong camera, "
                "or a manual dark exposure left on the camera by an earlier run.")
        out("  OPEN IT AND LOOK. Is the whole monitor in frame, with margin?")
        return 0
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------
def default_out_dir() -> Path:
    env = os.environ.get("LSA_RECORDINGS")
    if env:
        return Path(env)
    from .artwork.assets import user_dir
    return user_dir("recordings")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="screenplink --doctor",
        description="Is the camera on this machine fit to shoot with? Close the app first.")
    ap.add_argument("--all", action="store_true",
                    help="test every real camera and rank them (the default)")
    ap.add_argument("--index", type=int, default=None, help="test one camera")
    ap.add_argument("--frame", metavar="PATH", default=None,
                    help="save ONE picture to PATH and nothing else (for aiming)")
    ap.add_argument("--no-flash", action="store_true",
                    help="skip the screen-flash check (it takes over the display)")
    ap.add_argument("--screen", type=int, default=None,
                    help="flash this monitor (0-based) instead of the primary one")
    ap.add_argument("--out", default=None, help="folder for the saved pictures")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.out) if args.out else default_out_dir()

    say("=== cameras ===")
    cams = list_cameras()
    if not cams:
        say("  none found. Is a camera plugged in? Check that Windows Settings > "
            "Privacy > Camera allows desktop apps.")
        return 1
    for c in cams:
        say(f"  index {c.index}: {c.name}" + ("  [virtual]" if c.is_virtual else ""))

    if args.frame:
        index = args.index
        if index is None:
            real = [c for c in cams if not c.is_virtual and not virtual_pattern(c.name)]
            if not real:
                say("FAIL: no real camera to take a picture with")
                return 1
            index = real[0].index
            say(f"no --index given: using index {index} ({real[0].name})")
        return save_frame(index, Path(args.frame))

    def make_flasher():
        return None if args.no_flash else QtFlasher(args.screen)

    reports: list[CameraReport] = []

    def expired(label: str) -> None:
        say(f"\n!!! {label} did not finish in {CAMERA_DEADLINE_S:.0f}s: the driver is "
            f"HANGING inside a camera call. Do not retry in a loop -- unplug and replug "
            f"that camera, then run again. Partial results:")
        reports.append(CameraReport(-1, label, opened=True, delivered=True,
                                    aborted="hung inside a driver call"))
        for line in format_summary(reports):
            say(line)
        sys.stdout.flush()
        os._exit(3)

    dog = Watchdog(expired)

    if args.index is not None and not args.all:
        name = next((c.name for c in cams if c.index == args.index), "?")
        if virtual_pattern(name):
            say(f"\nNOTE: index {args.index} ({name}) looks like a software camera; "
                f"testing it because you asked.")
        say(f"\ntesting index {args.index} ({name})")
        dog.arm(CAMERA_DEADLINE_S, f"index {args.index} ({name})")
        try:
            rep = examine(args.index, name, flasher=make_flasher(), out_dir=out_dir)
        finally:
            dog.disarm()
        return _finish([rep])

    run_all(cams, make_flasher=make_flasher, out_dir=out_dir,
            watchdog=dog, reports=reports)
    return _finish(reports)


def _finish(reports: list[CameraReport]) -> int:
    if len(reports) == 1 and not reports[0].skipped:
        r = reports[0]
        say("\n=== verdict ===")
        if r.fails:
            say(f"  {len(r.fails)} FAILURE(S) -- do not expect shots to register:")
            for c in r.fails:
                say(f"     - {c.title}: {c.detail}")
        elif r.warns:
            say(f"  usable, with {len(r.warns)} warning(s):")
            for c in r.warns:
                say(f"     - {c.title}: {c.detail}")
        else:
            say("  healthy. Go and shoot.")
    for line in format_summary(reports):
        say(line)
    good = [r for r in reports if r.result in ("HEALTHY", "USABLE")]
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
