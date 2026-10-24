"""The camera doctor, driven against a STUB camera. No hardware, no window.

DO NOT point this at a real camera: this camera family has hung for 272 s on a
single exposure write, and the doctor's third check is an exposure sweep. Every
camera here is a stand-in whose behaviour is a parameter, so each fault the
doctor exists to tell apart (an ignored exposure, a crushed gain, a sleeping
display, a blind camera, a virtual device, a hung driver) can be produced on
demand and the doctor's answer read.

Each section says what goes red when the behaviour it guards is reverted.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import ast
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

from lsa import doctor as D                                      # noqa: E402
from lsa.camera.devices import CameraInfo                        # noqa: E402
from owned_files import OwnedFiles                               # noqa: E402

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


owned = OwnedFiles()


# ---------------------------------------------------------------------------
# the stand-ins
# ---------------------------------------------------------------------------
class Clock:
    """Virtual time: a stub camera's read() advances it by one frame period,
    so a frame rate is measured exactly and a test takes no real time."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class Scene:
    """What is on the monitor, shared by the flasher and the camera."""

    def __init__(self, awake: bool = True) -> None:
        self.colour = "black"
        self.awake = awake


H, W = 144, 256
MON = (slice(30, 110), slice(40, 200))      # where the monitor is in frame


class StubCap:
    """A camera whose faults are parameters. Records every property write."""

    def __init__(self, clock: Clock, scene: Scene, events: list, *, fps: float = 60.0,
                 opens: bool = True, delivers: bool = True, exposure_works: bool = True,
                 gain: float = 128.0, gain_readback: float | None = None,
                 sees_flash: bool = True, slow_on_auto: bool = False,
                 speckle: int = 0, block: threading.Event | None = None) -> None:
        self.clock, self.scene, self.events = clock, scene, events
        self.fps = fps
        self._opens, self._delivers = opens, delivers
        self.exposure_works = exposure_works
        self.gain = gain
        self.gain_readback = gain if gain_readback is None else gain_readback
        self.sees_flash = sees_flash
        self.slow_on_auto = slow_on_auto
        self.speckle = speckle
        self.block = block
        self.exposure = -6.0
        self.manual = False
        self.sets: list[tuple[int, float]] = []
        self.released = False

    # -- the cv2.VideoCapture surface the doctor uses ------------------------
    def isOpened(self) -> bool:
        return self._opens

    def get(self, prop: int) -> float:
        return self.gain_readback if prop == cv2.CAP_PROP_GAIN else 0.0

    def set(self, prop: int, value: float) -> bool:
        if self.block is not None and prop in (cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_AUTO_EXPOSURE):
            self.block.wait(5.0)            # a driver call that does not return
        self.sets.append((prop, value))
        if prop == cv2.CAP_PROP_GAIN:
            self.gain = value
            self.events.append(("set", "gain", value))
        elif prop == cv2.CAP_PROP_EXPOSURE:
            self.exposure = value
            self.events.append(("set", "exposure", value))
        elif prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            self.manual = value == 1.0
            self.events.append(("set", "auto_exposure", value))
        return True

    def read(self):
        if not self._delivers:
            return False, None
        rate = self.fps
        if self.slow_on_auto and not self.manual:
            rate = 25.0           # auto-exposure drags the rate down with the room
        self.clock.t += 1.0 / rate
        level = (200.0 * 2 ** ((self.exposure + 3.0) / 1.5)) if self.exposure_works else 120.0
        crush = 0.03 if self.gain == 0 else 1.0
        base = float(np.clip(level * crush, 0, 255))
        img = np.full((H, W, 3), base, np.float32)
        if self.sees_flash:
            if self.scene.awake and self.scene.colour == "white":
                img[MON] = 230.0 * crush
        else:
            img[5:25, 5:60] = 200.0            # a window, whatever the screen does
        if self.speckle:
            img[::20, ::5, 2] += self.speckle   # ~1% of pixels, red only
        return True, np.clip(img, 0, 255).astype(np.uint8)

    def release(self) -> None:
        self.released = True


class StubFlasher:
    cancelled = ""

    def __init__(self, scene: Scene, events: list) -> None:
        self.scene, self.events, self.begun, self.ended = scene, events, 0, 0

    def begin(self) -> None:
        self.begun += 1
        self.scene.colour = "black"
        self.events.append(("flash", "begin"))

    def show(self, colour: str) -> None:
        self.scene.colour = colour
        self.events.append(("flash", colour))

    def settle(self, ms: int) -> None:
        pass

    def end(self) -> None:
        self.ended += 1
        self.events.append(("flash", "end"))


def run_one(index=1, name="Stub Camera A", *, awake=True, flash=True, quiet=True, **cap_kw):
    """Examine one stub camera; return (report, cap, events, flasher, lines)."""
    clock, scene, events = Clock(), Scene(awake), []
    cap = StubCap(clock, scene, events, **cap_kw)
    fl = StubFlasher(scene, events) if flash else None
    lines: list[str] = []
    out_dir = Path(tempfile.mkdtemp(prefix="doctor_test_"))
    rep = D.examine(index, name, opener=lambda *a: cap, flasher=fl, out_dir=out_dir,
                    sleep=clock.sleep, clock=clock, out=lines.append)
    rep.out_dir = out_dir               # for the callers that look at the files
    return rep, cap, events, fl, lines


# ---------------------------------------------------------------------------
section("1. the order of the checks is the lesson")
# Revert: reorder examine() so the exposure sweep runs after the noise floor
# and the step-order check goes red.
# ---------------------------------------------------------------------------
rep, cap, events, fl, lines = run_one()
check(rep.steps == list(D.STEPS),
      "the six checks run in the documented order",
      f"ran {rep.steps}, want {list(D.STEPS)}")
check(rep.result == "HEALTHY", "a healthy stub camera is HEALTHY",
      f"{rep.result}: {[(c.state, c.title) for c in rep.checks if c.state in ('FAIL', 'WARN')]}")
check(rep.fps is not None and 59.0 <= rep.fps <= 61.0,
      "the frame rate is measured from the camera's own cadence", f"{rep.fps}")
headers = [l.strip() for l in lines if l.strip().startswith("=== ")]
check([h[4:5] for h in headers] == ["1", "2", "3", "4", "5", "6"],
      "the printed headers are numbered 1 to 6 in that order", f"{headers}")
check(sum(1 for l in lines if l.strip().startswith("why here:")) >= 5,
      "and the output keeps saying WHY each check sits where it does")
check(all(os.path.isabs(p) and os.path.isfile(p) for p in rep.frames) and len(rep.frames) == 2,
      "two pictures are saved, by absolute path", f"{rep.frames}")
check(any(p in "\n".join(lines) for p in rep.frames),
      "and the saved paths are printed")
check(rep.writes <= 12, f"control writes are rationed ({rep.writes}, budget 12)")
check(cap.released, "the camera is always released")


# ---------------------------------------------------------------------------
section("2. frame rate is measured with the exposure FIXED")
# Revert: measure fps before the two writes in step 2 and the camera that is
# slow only on auto-exposure reads 25 instead of 60.
# ---------------------------------------------------------------------------
rep, cap, events, fl, lines = run_one(slow_on_auto=True)
check(rep.fps is not None and rep.fps > 55.0,
      "a camera slow only on auto-exposure is measured at its real rate",
      f"{rep.fps} -- the number tracked the room, which is how 28.3/28.7/51.2 happened")
first_writes = [e[1] for e in events if e[0] == "set"][:2]
check(first_writes == ["auto_exposure", "exposure"],
      "the very first writes are 'manual' and 'exposure', before the rate is read",
      f"{first_writes}")


# ---------------------------------------------------------------------------
section("3. a camera that IGNORES exposure is called out, before anything else is believed")
# Revert: delete the spread test and an ignoring camera passes.
# ---------------------------------------------------------------------------
rep, *_ = run_one(exposure_works=False)
fails = [c.title for c in rep.fails]
check("exposure control" in fails, "ignored exposure is a FAIL", f"{fails}")
check(rep.result == "POOR", "...and the camera is POOR, not healthy", rep.result)
ec = next(c for c in rep.fails if c.title == "exposure control")
check("IGNORING" in ec.detail and "check 2" in ec.detail,
      "...and it says the frame rate is suspect too")
rep, *_ = run_one(exposure_works=True)
check(rep.spread_pct is not None and rep.spread_pct > 80,
      "a working camera moves the image 80%+", f"{rep.spread_pct}")


# ---------------------------------------------------------------------------
section("4. gain is restored after the noise floor (the trap this tool fell into)")
# Revert: remove the `finally` that writes prior_gain back and the flash check
# runs on a crushed image: the 'white' frame never gets bright.
# ---------------------------------------------------------------------------
rep, cap, events, fl, lines = run_one(gain=128.0)
gain_events = [(i, e) for i, e in enumerate(events) if e[:2] == ("set", "gain")]
check([e[2] for _i, e in gain_events] == [0.0, 128.0],
      "gain is pinned to 0 and then put back to what the camera reported",
      f"{[e for _i, e in gain_events]}")
first_flash = next(i for i, e in enumerate(events) if e[0] == "flash")
check(gain_events and gain_events[-1][0] < first_flash,
      "...and it is restored BEFORE the flash check", f"{gain_events} vs flash at {first_flash}")
check(cap.gain == 128.0, "the camera finishes on its own gain", f"{cap.gain}")
check(rep.flash_diff is not None and rep.flash_diff >= D.FLASH_HEALTHY,
      "so the flash check sees a real difference, not a crushed one",
      f"{rep.flash_diff} (the crushed-gain bug measured 2.0 instead of 46.8)")

rep, cap, events, *_ = run_one(gain=128.0, gain_readback=-1.0)
check(not [e for e in events if e[:2] == ("set", "gain")],
      "a camera that does not REPORT its gain has it left alone, never pinned",
      "pinning something that cannot be restored is how it was crushed before")
noise = next(c for c in rep.checks if c.title == "chroma noise floor")
check("LEFT ALONE" in noise.detail, "...and the report says so")
check(cap.gain == 128.0, "...and the camera still has its own gain")
rep, cap, events, *_ = run_one(gain=0.0)
check(not [e for e in events if e[:2] == ("set", "gain")],
      "a camera already at gain 0 needs no write at all")

rep, *_ = run_one(speckle=60)
check(rep.floor is not None and 55 <= rep.floor <= 65,
      "the noise floor is the chroma 99th percentile, not brightness", f"{rep.floor}")


# ---------------------------------------------------------------------------
section("5. a SLEEPING DISPLAY is not a BLIND CAMERA")
# Revert: drop the display_is_lit branch and the sleeping display reads as a
# camera FAIL, sending the user to adjust hardware that is working.
# ---------------------------------------------------------------------------
asleep, *_ = run_one(awake=False)
blind, *_ = run_one(sees_flash=False)
flash_a = next(c for c in asleep.checks if c.title == "screen flash visible")
flash_b = next(c for c in blind.checks if c.title == "screen flash visible")
check(asleep.display_lit is False and blind.display_lit is True,
      "the two cases are told apart by whether the white frame got bright",
      f"asleep.lit={asleep.display_lit} blind.lit={blind.display_lit}")
check(flash_a.state == D.WARN and "asleep" in flash_a.detail.lower()
      and "NOT evidence against the camera" in flash_a.detail,
      "a sleeping display is a WARNING that says the display is asleep", flash_a.detail[:90])
check(flash_b.state == D.FAIL and "not seeing this screen" in flash_b.detail,
      "a blind camera is a FAILURE that blames the camera", flash_b.detail[:90])
check(not asleep.fails and asleep.result == "USABLE",
      "...and the sleeping display does NOT make the camera POOR", asleep.result)
check(blind.result == "POOR", "...while the blind camera does", blind.result)
check(any("ASLEEP" in l for l in D.format_summary([asleep])),
      "the summary tells the user to wake the display and run again")

nf, cap, events, fl, lines = run_one(flash=False)
check(fl is None and not [e for e in events if e[0] == "flash"],
      "--no-flash never touches the display")
check(nf.steps == list(D.STEPS) and nf.flash_diff is None,
      "...and still walks the same steps", f"{nf.steps}")


# ---------------------------------------------------------------------------
section("6. virtual cameras are skipped, with the reason, and never opened")
# Revert: remove the skip in run_all and the virtual device is opened.
# ---------------------------------------------------------------------------
opened: list[int] = []
clock = Clock()


def make_world(specs):
    """specs: index -> StubCap kwargs; the opener records what it opened."""
    scene, events = Scene(), []

    def opener(index, *_a):
        opened.append(index)
        return StubCap(clock, scene, events, **specs[index])
    return opener, scene, events


cams = [
    CameraInfo(0, "OBS Virtual Camera", True),
    CameraInfo(1, "Stub Camera A", False),
    CameraInfo(2, "Odd Name", True),            # flagged by the lister, name unremarkable
]
opener, scene, events = make_world({1: {}})
out_dir = Path(tempfile.mkdtemp(prefix="doctor_test_"))
lines = []
reps = D.run_all(cams, make_flasher=lambda: StubFlasher(scene, events), out_dir=out_dir,
                 opener=opener, sleep=clock.sleep, clock=clock, out=lines.append)
check(opened == [1], "only the real camera was opened", f"{opened}")
sk = [r for r in reps if r.skipped]
check([r.index for r in sk] == [0, 2], "both virtual devices are SKIPPED, in order")
check("virtual" in sk[0].skipped and "OBS Virtual" in sk[0].skipped
      and "--index 0" in sk[0].skipped,
      "...with the matching pattern named and the way to override it", sk[0].skipped[:100])
check(all(not r.frames for r in sk), "a skipped camera saves no picture")
summ = "\n".join(D.format_summary(reps))
check("SKIPPED" in summ and "OBS Virtual Camera" in summ, "and the summary lists them")


# ---------------------------------------------------------------------------
section("7. --all tests every real camera and RANKS them")
# Revert: break rank() (e.g. sort by index) and the order check goes red.
# ---------------------------------------------------------------------------
opened.clear()
specs = {
    0: {"exposure_works": False},            # accepts writes, ignores them
    1: {"opens": False},                     # will not open
    2: {},                                   # healthy, 60 fps
    3: {"fps": 30.0},                        # works, but slow
    4: {"fps": 45.0},                        # healthy-ish, between 3 and 2
    5: {},                                   # virtual (skipped)
}
cams = [CameraInfo(0, "Stub Cam P", False), CameraInfo(1, "Stub Cam Q", False),
        CameraInfo(2, "Stub Cam R", False), CameraInfo(3, "Stub Cam S", False),
        CameraInfo(4, "Stub Cam T", False), CameraInfo(5, "Snap Camera", False)]
opener, scene, events = make_world(specs)
out_dir = Path(tempfile.mkdtemp(prefix="doctor_test_"))
lines = []
reps = D.run_all(cams, make_flasher=lambda: StubFlasher(scene, events), out_dir=out_dir,
                 opener=opener, sleep=clock.sleep, clock=clock, out=lines.append)
ranked = D.rank(reps)
check(len(ranked) == 5 and 5 not in [r.index for r in ranked],
      "five real cameras are ranked, the virtual one is not", f"{[r.index for r in ranked]}")
check([r.index for r in ranked] == [2, 4, 3, 0, 1],
      "best first: healthy 60 fps, then the slower ones, then POOR, then UNUSABLE",
      f"{[(r.index, r.result, r.fps and round(r.fps)) for r in ranked]}")
check([r.result for r in ranked] == ["HEALTHY", "USABLE", "USABLE", "POOR", "UNUSABLE"],
      "...with honest result words", f"{[r.result for r in ranked]}")
summ = D.format_summary(reps)
rec = next(l for l in summ if l.startswith("RECOMMENDED"))
check("index 2" in rec and "Stub Cam R" in rec,
      "the summary recommends the best camera with a reason", rec)
check("fps" in rec, "...and the reason is measurements, not a name")
check(len(set(os.path.dirname(p) for r in ranked for p in r.frames)) == 1
      and sum(len(r.frames) for r in ranked) == 2 * 4,
      "a PNG pair is written for every camera that delivered frames (not the one that "
      "would not open)", f"{[(r.index, len(r.frames)) for r in ranked]}")
check(sum(1 for l in summ if l.strip().endswith(".png")) == 8,
      "...and every path is listed at the end for Claude to open")
check(5 not in opened, "the virtual camera was never opened")


# ---------------------------------------------------------------------------
section("8. one failing camera cannot sink --all")
# Revert: remove the try/except in examine() and a crash aborts the run.
# ---------------------------------------------------------------------------
class Exploding(StubCap):
    def read(self):
        raise RuntimeError("driver exploded")


boom_clock, boom_scene, boom_events = Clock(), Scene(), []
rep = D.examine(7, "Stub Cam Z", opener=lambda *a: Exploding(boom_clock, boom_scene, boom_events),
                flasher=None, out_dir=Path(tempfile.mkdtemp(prefix="doctor_test_")),
                sleep=boom_clock.sleep, clock=boom_clock, out=lambda s: None)
check(rep.result == "UNUSABLE" and "driver exploded" in rep.aborted,
      "an exception is a report, not a crash", f"{rep.result} {rep.aborted}")


# ---------------------------------------------------------------------------
section("9. a hung driver is reported, not waited on forever")
# Revert: remove the watchdog arm in run_all and nothing ever fires.
# ---------------------------------------------------------------------------
fired: list[str] = []
dog = D.Watchdog(fired.append)
release = threading.Event()
hclock, hscene, hevents = Clock(), Scene(), []
hung_cams = [CameraInfo(3, "Stub Cam H", False)]
t0 = time.time()
th = threading.Thread(target=lambda: D.run_all(
    hung_cams, make_flasher=lambda: None, out_dir=Path(tempfile.mkdtemp(prefix="doctor_test_")),
    opener=lambda *a: StubCap(hclock, hscene, hevents, block=release),
    watchdog=dog, deadline_s=0.2, sleep=hclock.sleep, clock=hclock, out=lambda s: None))
th.start()
time.sleep(0.8)
check(fired == ["index 3 (Stub Cam H)"],
      "a camera call that never returns trips the per-camera deadline, naming the camera",
      f"{fired}")
release.set()
th.join(10)
check(not th.is_alive(), "(the stub is released so the test itself cannot hang)")
dog.disarm()
fired.clear()
dog2 = D.Watchdog(fired.append)
dog2.arm(0.15, "x")
dog2.disarm()
time.sleep(0.4)
check(fired == [], "a disarmed deadline never fires (a finished camera is not a hang)")
quiet_clock, quiet_scene, quiet_events = Clock(), Scene(), []
dog3 = D.Watchdog(fired.append)
D.run_all([CameraInfo(1, "Stub Cam A", False)], make_flasher=lambda: None,
          out_dir=Path(tempfile.mkdtemp(prefix="doctor_test_")),
          opener=lambda *a: StubCap(quiet_clock, quiet_scene, quiet_events),
          watchdog=dog3, deadline_s=0.3, sleep=quiet_clock.sleep, clock=quiet_clock,
          out=lambda s: None)
time.sleep(0.6)
check(fired == [], "a camera that finishes in time never trips it")


# ---------------------------------------------------------------------------
section("10. --frame saves ONE picture and writes NO controls")
# Revert: make save_frame set an exposure and the no-write check goes red.
# ---------------------------------------------------------------------------
fc, fs, fe = Clock(), Scene(), []
fcap = StubCap(fc, fs, fe)
lines = []
target = Path(tempfile.mkdtemp(prefix="doctor_test_")) / "sub" / "aim.png"
rc = D.save_frame(1, target, opener=lambda *a: fcap, out=lines.append)
check(rc == 0 and target.is_file(), "the picture is written, creating the folder", f"{rc}")
check(fcap.sets == [], "...with no control writes at all (safe to repeat while aiming)",
      f"{fcap.sets}")
check(fcap.released, "...and the camera is released")
check(any(str(target.resolve()) in l for l in lines), "the absolute path is printed")
img = cv2.imread(str(target))
check(img is not None and img.shape[:2] == (H, W), "and it is a readable image")
bad = StubCap(fc, fs, fe, opens=False)
check(D.save_frame(1, target, opener=lambda *a: bad, out=lambda s: None) == 1,
      "a camera that will not open is exit 1")
nof = StubCap(fc, fs, fe, delivers=False)
check(D.save_frame(1, target, opener=lambda *a: nof, out=lambda s: None) == 1,
      "a camera that delivers nothing is exit 1")


# ---------------------------------------------------------------------------
section("11. control writes: spaced, counted, de-duplicated")
# ---------------------------------------------------------------------------
cc = Clock()
cap2 = StubCap(cc, Scene(), [])
ctl = D.Controls(cap2, cc.sleep)
a = ctl.set("exposure", cv2.CAP_PROP_EXPOSURE, -5.0)
b = ctl.set("exposure", cv2.CAP_PROP_EXPOSURE, -5.0)
c_ = ctl.set("exposure", cv2.CAP_PROP_EXPOSURE, -7.0)
check((a, b, c_) == (True, False, True) and ctl.writes == 2 and len(cap2.sets) == 2,
      "a value this process already wrote is not written again")
check(abs(cc.t - 2 * D.WRITE_GAP_S) < 1e-9, "each write is preceded by the gap",
      f"{cc.t}")
check(D.EXPOSURES[-1] >= -11.0, "the sweep never goes below -11, where the camera WRAPS")


# ---------------------------------------------------------------------------
section("12. nothing about a camera is assumed")
# ---------------------------------------------------------------------------
src = (ROOT / "src" / "lsa" / "doctor.py").read_text(encoding="utf-8")
code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
for word in ("razer", "kiyo", "logitech", "c920", "c910", "canon", "webcam utility", "elgato",
             "brio", "obsbot"):
    check(word not in src.lower(), f"no camera brand or model in doctor.py: {word!r}")
check(not re.search(r"index\s*==\s*\d|\bindex\s*=\s*[1-9]\b", code),
      "no camera index is hard-coded")
check(D.virtual_pattern("EOS Webcam Utility") != "" and D.virtual_pattern("Stub Camera A") == "",
      "virtual detection reuses the one list in lsa.camera.devices")
tree = ast.parse(src)
imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
order = [(ast.unparse(n)) for n in imports]
i_env = next(i for i, s in enumerate(order) if "_env" in s)
i_cv2 = next(i for i, s in enumerate(order) if s == "import cv2")
check(i_env < i_cv2, "lsa._env is imported BEFORE cv2 (73.8 s vs 0.43 s)", f"{order}")
check("display_is_lit" in src and not re.search(r"max\(\)\)?\s*>\s*\d+", src),
      "the lit-display test is the app's own function, with no number of its own")
check("getattr(self" not in src, "no getattr(self, ...) defaults")
check(all(ord(ch) < 128 for ch in src), "doctor.py is pure ASCII (a Windows console prints it)")


# ---------------------------------------------------------------------------
section("13. across the seams: docs, entry point, spec, assembler, front door")
# The producer and the consumer of every contract, tested together (RULES C14).
# ---------------------------------------------------------------------------
parser = D.build_parser()
VALUE_FLAGS = {"--index", "--frame", "--screen", "--out"}
MARKERS = ("Doctor.cmd", "screenplink.exe --doctor")


def flags_after(line: str, marker: str) -> str:
    """The flags a doc line shows after `marker`, up to the closing backtick."""
    rest = line.split(marker, 1)[1].split("`", 1)[0].split()
    toks, i = [], 0
    while i < len(rest) and rest[i].startswith("--"):
        toks.append(rest[i])
        if rest[i] in VALUE_FLAGS and i + 1 < len(rest):
            toks.append(rest[i + 1])
            i += 1
        i += 1
    return " ".join(toks)


cmds = []
for doc in ("release-CLAUDE.md", "release-README.md"):
    text = (ROOT / "packaging" / doc).read_text(encoding="utf-8")
    check("screenplink-doctor" not in text,
          f"{doc} names no screenplink-doctor.exe (there is ONE executable now)")
    check("doctor-report.txt" in text,
          f"{doc} says the doctor's report is written to doctor-report.txt")
    for line in text.splitlines():
        for marker in MARKERS:
            if marker in line:
                f = flags_after(line, marker)
                if f:
                    cmds.append(f)

# What Doctor.cmd itself runs: its default arguments and its pass-through.
# The two launchers live in scripts/ and serve BOTH the release (beside the exe) and the
# repository (from source), so each carries an exe branch and a source branch.
doctor_cmd = (ROOT / "scripts" / "Doctor.cmd").read_text(encoding="utf-8")
play_cmd = (ROOT / "scripts" / "Play.cmd").read_text(encoding="utf-8")
m_default = re.search(r'if "%ARGS%"=="" set "ARGS=([^"]*)"', doctor_cmd)
m_run = re.search(r"^%DOCTOR% %ARGS%(.*)$", doctor_cmd, re.M)
check(re.search(r'^if exist "%~dp0screenplink\.exe" set DOCTOR="%~dp0screenplink\.exe" --doctor$',
                doctor_cmd, re.M) is not None,
      "Doctor.cmd: with screenplink.exe beside it, it runs `screenplink.exe --doctor`")
check(re.search(r'set DOCTOR="%PY%" -m lsa\.doctor$', doctor_cmd, re.M) is not None
      and "PYTHONPATH=%APP%\\src" in doctor_cmd,
      "Doctor.cmd: otherwise it runs the doctor from source (`-m lsa.doctor`, src on the path)")
check(all("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0" in t for t in (doctor_cmd, play_cmd)),
      "both launchers set OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 before anything can import cv2")
check(not re.search(r"^\s*(goto|:\w+)", doctor_cmd + play_cmd, re.M),
      "neither launcher uses a label or GOTO (a .cmd with bare LF endings can miss a label)")
check(m_default is not None and m_default.group(1) == "--all",
      "Doctor.cmd defaults to --all", m_default and m_default.group(1))
check('set "ARGS=%*"' in doctor_cmd,
      "Doctor.cmd passes its own arguments through (Doctor.cmd --index 2 works)")
check(m_run is not None and m_run.group(1).strip() == "> doctor-report.txt 2>&1",
      "Doctor.cmd redirects stdout AND stderr into doctor-report.txt "
      "(the exe is windowed and has no console)", m_run and m_run.group(1))
check("type doctor-report.txt" in doctor_cmd,
      "...and then shows that file so a human sees it")
check(not re.search(r"^\s*pause\b", doctor_cmd, re.M | re.I),
      "Doctor.cmd never pauses (Claude Code would hang on it)")
check(re.search(r'screenplink\.exe"? --target %\*', play_cmd) is not None,
      "Play.cmd runs screenplink.exe --target and passes arguments through")
check(re.search(r'camera_lab\.py"? --target %\*', play_cmd) is not None
      and "pythonw.exe" in play_cmd,
      "Play.cmd without the exe runs camera_lab.py --target from the venv with pythonw "
      "(no console window on the monitor the camera watches), arguments passed through")
check("--doctor" not in play_cmd and "--mcp" not in play_cmd,
      "Play.cmd starts the game and nothing else")
if m_default:
    cmds.append(m_default.group(1))
check(len(cmds) >= 2, "the release docs and Doctor.cmd contain doctor commands to test", f"{cmds}")
for c in cmds:
    argv = [("1" if tok == "N" else tok) for tok in c.split()]
    try:
        ns = parser.parse_args(argv)
        ok = True
    except SystemExit:
        ok = False
    check(ok, f"the documented command parses: --doctor {c}")
# The check above has to be able to fail: a command the parser rejects.
try:
    parser.parse_args(["--nonsense-flag"])
    _rejects = False
except SystemExit:
    _rejects = True
check(_rejects, "the parser REJECTS an unknown flag (so 'the command parses' can fail)")
ns = parser.parse_args(["--index", "1", "--frame", "aim.png"])
check(ns.frame == "aim.png" and ns.index == 1, "--index N --frame PATH means what the docs say")
check(parser.parse_args([]).all is False and parser.parse_args([]).index is None,
      "launched bare it selects no single camera, which main() runs as --all")

mcp_json = json.loads((ROOT / "packaging" / "mcp.json").read_text(encoding="utf-8"))
_srv = mcp_json["mcpServers"]["screenplink"]
check(_srv["command"] == "./screenplink.exe" and _srv["args"] == ["--mcp"],
      ".mcp.json runs ./screenplink.exe --mcp (one exe, mode by argument)", f"{_srv}")

# --- the entry point: dispatch on ARGUMENTS ONLY ----------------------------
import types                                                    # noqa: E402

entry_src = (ROOT / "packaging" / "screenplink_entry.py").read_text(encoding="utf-8")
entry_tree = ast.parse(entry_src)
_main_fn = next(n for n in entry_tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
check("sys.executable" not in ast.unparse(_main_fn),
      "the entry point never looks at the executable's own name")
check(not any(isinstance(n, ast.FunctionDef) and n.name.startswith("_wants")
              for n in entry_tree.body),
      "no name-based dispatch helpers (_wants_*) survive")
ns_entry: dict = {"__name__": "entry_under_test"}
exec(compile(entry_src, "screenplink_entry.py", "exec"), ns_entry)

CALLS: list = []


def _fake(name):
    mod = types.ModuleType(name)
    mod.main = lambda argv=None, _n=name: CALLS.append((_n, list(argv or []))) or 0
    return mod


FAKES = ("lsa.doctor", "lsa.mcp", "lsa.__main__")
saved_mods = {k: sys.modules.get(k) for k in FAKES}
real_exe, real_frozen, real_argv = sys.executable, getattr(sys, "frozen", None), sys.argv


def dispatch(args, exe=r"C:\x\ScreenPlink\screenplink.exe", frozen=True):
    CALLS.clear()
    for k in FAKES:
        sys.modules[k] = _fake(k)
    sys.argv = ["screenplink.exe", *args]
    sys.executable = exe
    if frozen:
        sys.frozen = True
    elif hasattr(sys, "frozen"):
        del sys.frozen
    ns_entry["main"]()
    return list(CALLS)


try:
    check(dispatch(["--doctor", "--all"]) == [("lsa.doctor", ["--all"])],
          "--doctor --all runs the doctor with --all and the --doctor flag stripped")
    check(dispatch(["--doctor"]) == [("lsa.doctor", [])],
          "--doctor alone runs the doctor with no arguments (main() runs --all)")
    check(dispatch(["--doctor", "--index", "2", "--frame", "a.png"])
          == [("lsa.doctor", ["--index", "2", "--frame", "a.png"])],
          "Doctor.cmd --index 2 --frame a.png reaches the doctor intact")
    check(dispatch(["--mcp"]) == [("lsa.mcp", [])], "--mcp runs the MCP server")
    check(dispatch(["--mcp", "--port=9001"]) == [("lsa.mcp", ["--port=9001"])],
          "--mcp passes its other arguments on")
    check(dispatch(["--target", "--introspect"])
          == [("lsa.__main__", ["--target", "--introspect"])],
          "--target --introspect runs the APP, arguments intact")
    check(dispatch([]) == [("lsa.__main__", [])], "a bare screenplink.exe is the app")
    # The trap the old design had, now closed: the name decides nothing.
    for odd in (r"C:\x\screenplink-doctor.exe", r"C:\x\screenplink-mcp.exe",
                r"C:\work\mcp\screenplink.exe"):
        check(dispatch([], exe=odd) == [("lsa.__main__", [])],
              f"the executable's NAME decides nothing: {odd} with no arguments is the app")
    check(dispatch([], exe=r"C:\py\python.exe", frozen=False) == [("lsa.__main__", [])],
          "source mode: no arguments is the app")
finally:
    sys.executable, sys.argv = real_exe, real_argv
    if real_frozen is None:
        if hasattr(sys, "frozen"):
            del sys.frozen
    else:
        sys.frozen = real_frozen
    for k, v in saved_mods.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v

# The two light modes are decided before camera_lab can be imported.
_order = []
for n in _main_fn.body:
    if isinstance(n, ast.If):
        _t = ast.dump(n.test)
        if "--doctor" in _t:
            _order.append("doctor")
        elif "--mcp" in _t:
            _order.append("mcp")
    elif isinstance(n, ast.ImportFrom) and "__main__" in (n.module or ""):
        _order.append("app-import")
check(_order == ["doctor", "mcp", "app-import"],
      "--doctor and --mcp are decided BEFORE the app (and so camera_lab) is imported",
      f"{_order}")
check("lsa.doctor" in entry_src and "lsa.mcp" in entry_src,
      "the entry point imports lsa.doctor and lsa.mcp")

# --- the spec: ONE exe, windowed -------------------------------------------
spec = (ROOT / "packaging" / "screenplink.spec").read_text(encoding="utf-8")
spec_code = "\n".join(l for l in spec.splitlines() if not l.lstrip().startswith("#"))
exes = re.findall(r"\n(\w+) = EXE\(", spec_code)
check(exes == ["app"], "the spec builds exactly ONE EXE", f"{exes}")
blocks = re.split(r"\n\w+ = EXE\(", spec_code)[1:]
if blocks:
    _b = blocks[0].split("\n)\n", 1)[0]
    check('name="screenplink"' in _b, "...named screenplink")
    check("console=False" in _b and "console=True" not in _b,
          "...and WINDOWED (a console window would sit lit on the monitor the camera watches)")
check("screenplink-mcp" not in spec_code and "screenplink-doctor" not in spec_code,
      "no second or third binary is named anywhere in the spec's code")
coll = spec_code.split("COLLECT(")[1].split("\n)\n", 1)[0]
check(re.search(r"\bapp,", coll) is not None and not re.search(r"\b(mcp|doctor),", coll),
      "the COLLECT holds the one exe and nothing else")
check("upx=False" in spec_code, "upx stays off")
check('"lsa.doctor"' in spec and '"lsa.mcp"' in spec,
      "lsa.doctor and lsa.mcp are named hidden imports (reached only by dispatch)")

asm = (ROOT / "packaging" / "assemble.py").read_text(encoding="utf-8")
check('(OUT / "screenplink.exe")' in asm and '"Play.cmd", "Doctor.cmd"' in asm,
      "assemble.py REFUSES a release without screenplink.exe, Play.cmd and Doctor.cmd")
check(not re.search(r'for exe in \([^)]*screenplink-(mcp|doctor)', asm),
      "assemble.py no longer REQUIRES the two deleted exes")
check('(ROOT / "scripts" / "Play.cmd", "Play.cmd")' in asm
      and '(ROOT / "scripts" / "Doctor.cmd", "Doctor.cmd")' in asm,
      "assemble.py copies both .cmd files (the dev tree's own, from scripts/) beside the exe")
check('ASSET_PARENT / name).is_dir()' in asm and "ASSET_DIRS" in asm,
      "assemble.py REFUSES a bundle whose asset folders are not under resources/")

health = (ROOT / "tools" / "camera_health.py").read_text(encoding="utf-8")
check("lsa.doctor" in health or "from lsa import doctor" in health,
      "tools/camera_health.py is a front door onto the doctor, not a second copy")
check("cap.set(" not in health and "display_is_lit" not in health,
      "...and holds no camera code of its own")

check(D.default_out_dir().name == "recordings", "pictures default to the recordings folder")

owned.check(lambda ok, what, detail="": check(ok, what, detail))
print("\n" + (f"ALL {CHECKS} CHECKS PASSED" if not FAILED else
              f"{len(FAILED)} OF {CHECKS} CHECKS FAILED"))
raise SystemExit(0 if not FAILED else 1)
