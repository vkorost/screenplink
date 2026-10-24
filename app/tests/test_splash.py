"""The startup splash: scaled by fraction, never trapping, and NEVER visible
while anything is measuring the screen.

Why the last property matters. The splash artwork has red excess 229 (the
dot over the "i"), 69,512 pixels above 30 and 2.65% of the frame lit above
luminance 60. To the detector that is a laser. If it is on the screen while
the noise floor is measured, the floor is taken from the splash and the bar
lands above every real shot (the verdict board at red excess 24 did exactly
that on 2026-10-15). So the property is asserted from every direction a
measurement can start:

  * every flag in `CameraLab.SCREEN_OWNERS` (derived from the tuple, so an
    owner added later is covered with no edit here);
  * every Director job that declares `measures_screen`, discovered by
    calling every `*_job` factory in `lsa.routines`, and at least three
    must be found so a renamed factory cannot make this pass vacuously;
  * `_publish_measuring` when anything reports busy, and `_open_flash`.

THE CHECK THAT PROVES "NEVER VISIBLE DURING A MEASUREMENT" is section 4.

Headless (QT_QPA_PLATFORM=offscreen); no camera, no screen, no sound.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# BEFORE camera_lab is imported: LOG_PATH and RECORDINGS are read at import.
from owned_files import (isolate_outputs, assert_silent_mixer,       # noqa: E402
                         assert_silent_listener, OwnedFiles)
isolate_outputs("splash")

from PySide6.QtCore import QPoint, Qt, QEvent                         # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent               # noqa: E402
from PySide6.QtWidgets import QApplication                            # noqa: E402

from lsa import routines                                              # noqa: E402
from lsa.director import Job                                          # noqa: E402
from lsa.display import splash as splash_mod                          # noqa: E402
from lsa.display.splash import Splash, fit_rect, _Loader              # noqa: E402
from lsa.tools.camera_lab import CameraLab                            # noqa: E402


def main() -> int:
    app = QApplication.instance() or QApplication([])
    owned = OwnedFiles()
    results: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail else ""))

    def pump(ms: int = 30) -> None:
        end = time.perf_counter() + ms / 1000
        while time.perf_counter() < end:
            app.processEvents()

    def key(widget, k) -> None:
        widget.keyPressEvent(QKeyEvent(QEvent.KeyPress, k, Qt.NoModifier))

    print("\n1. Scaling is by fraction, centred and never distorted")
    screens = [(3840, 2160), (1920, 1080), (7680, 4320), (1920, 1200),
               (3440, 1440), (1280, 800), (1024, 768), (1080, 1920)]
    ok_aspect = ok_fit = ok_centre = True
    worst = 0.0
    for sw, sh in screens:
        x, y, w, h = fit_rect(sw, sh, 7680, 4320)
        worst = max(worst, abs(w / h - 7680 / 4320))
        ok_aspect &= abs(w / h - 7680 / 4320) < 0.01
        ok_fit &= (0 <= x and 0 <= y and x + w <= sw and y + h <= sh)
        ok_centre &= abs((sw - w) - 2 * x) <= 1 and abs((sh - h) - 2 * y) <= 1
        # touches the screen on at least one axis: as large as it can be
        ok_fit &= (w >= sw - 1 or h >= sh - 1)
    check("aspect ratio preserved on every screen shape", ok_aspect,
          f"worst error {worst:.4f}")
    check("always inside the screen and as large as fits", ok_fit)
    check("always centred", ok_centre)
    check("a 16:9 screen is filled exactly",
          fit_rect(3840, 2160, 7680, 4320) == (0, 0, 3840, 2160))
    check("degenerate sizes do not divide by zero",
          fit_rect(0, 0, 7680, 4320) == (0, 0, 0, 0))

    print("\n2. The real artwork loads, scales once off-thread, and is the right shape")
    check("the splash image is in the repo", splash_mod.SPLASH_PATH.exists(),
          str(splash_mod.SPLASH_PATH))
    img = QImage(str(splash_mod.SPLASH_PATH))
    # NOT a pinned size. He replaces this artwork -- he supplied three versions
    # in one evening -- and a check that names his file's dimensions fails when
    # he improves it, which is the app looking broken because he did some work.
    # What matters is that it LOADS and is big enough to scale to his panel.
    check("the splash image loads", not img.isNull())
    # A FLOOR, not a preference. Below this it is visibly soft on a 42" 4K
    # panel; above it, how much it is upscaled is his choice of artwork and
    # none of this check's business. The upscale factor is REPORTED so it is
    # visible without being a failure -- he supplied three versions of this
    # image in one evening and a check that argues with his art is noise.
    check("the splash is big enough not to look soft on a 4K panel",
          img.width() >= 1920 and img.height() >= 1080,
          f"{img.width()}x{img.height()}")
    print(f"     (splash {img.width()}x{img.height()}; upscaled "
          f"{3840 / max(1, img.width()):.2f}x to his 3840-wide panel)")
    got: list[QImage] = []
    ld = _Loader(splash_mod.SPLASH_PATH, 1280, 800)
    ld.ready.connect(got.append)
    t0 = time.perf_counter()
    ld.start()
    ld.wait(10000)
    pump(50)
    check("the loader delivered one image", len(got) == 1)
    if got:
        x, y, w, h = fit_rect(1280, 800, img.width(), img.height())
        check("scaled to the letterboxed size, not the screen size",
              (got[0].width(), got[0].height()) == (w, h),
              f"{got[0].width()}x{got[0].height()} vs {w}x{h}")
    print(f"     (decode + scale on the worker: {time.perf_counter() - t0:.2f}s)")

    print("\n3. It can never trap anyone (RULES B6)")
    s = Splash(image_path=None)
    s.start(app.primaryScreen())
    pump()
    check("it is up after start", s.up)
    key(s, Qt.Key_Escape)
    check("Escape takes it down", not s.up and s.retire_reason == "escape")
    s = Splash(image_path=None)
    s.start(app.primaryScreen())
    s.mouseDoubleClickEvent(QMouseEvent(
        QEvent.MouseButtonDblClick, QPoint(5, 5), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))
    check("a double click takes it down", not s.up)
    s = Splash(image_path=None)
    s.start(app.primaryScreen(), deadline_s=0.05)
    pump(300)
    check("its own deadline takes it down with nobody asking",
          not s.up and s.retire_reason == "timeout")
    s = Splash(image_path=None)
    s.start(app.primaryScreen())
    s.show_problem("The camera could not be opened.")
    pump()
    check("a problem keeps it up and says what it is",
          s.up and "camera" in s.problem)
    check("a problem still has a finite deadline",
          0 < s.PROBLEM_DEADLINE_S < 600)
    key(s, Qt.Key_Escape)
    check("and Escape still closes it", not s.up)
    check("retiring twice is harmless", s.retire("again") is False)

    print("\n4. NEVER VISIBLE WHILE ANYTHING MEASURES THE SCREEN")
    import lsa.settings as _settings
    _orig_load = _settings.AppSettings.load

    @staticmethod
    def _no_connect(path):
        st = _orig_load(path)
        st.auto_connect = False     # a test must never open the camera
        return st

    _settings.AppSettings.load = _no_connect
    try:
        lab = CameraLab()
    finally:
        _settings.AppSettings.load = _orig_load
    assert lab.capture is None, "the splash test must not open the camera"
    assert_silent_mixer(lab)
    assert_silent_listener(lab)
    lab._save_settings = lambda *a, **k: None
    lab._schedule_save = lambda *a, **k: None
    lab.settings.auto_connect = True          # so attach does not retire it

    def fresh() -> Splash:
        sp = Splash(image_path=None)
        sp.start(app.primaryScreen())
        pump()
        lab.splash = None
        lab.attach_splash(sp)
        assert sp.up, "fixture: the splash must be up before the trigger"
        return sp

    sp = fresh()
    check("fixture: attached with auto-connect on, it stays up", sp.up)
    lab._splash_down("test cleanup")

    owners = list(lab.SCREEN_OWNERS)
    check("SCREEN_OWNERS is not empty", len(owners) > 0)
    for flag, name in owners:
        sp = fresh()
        setattr(lab, flag, True)
        check(f"raising {flag} ({name}) takes it down",
              not sp.up and lab.splash is None)
        setattr(lab, flag, False)
        check(f"{flag} still reads back what was stored",
              getattr(lab, flag) is False)
        setattr(lab, flag, True)
        check(f"{flag} reads True when set", getattr(lab, flag) is True)
        setattr(lab, flag, False)

    def _empty():
        return 1
        yield  # noqa: unreachable -- makes this a generator

    found = 0
    for fname in sorted(dir(routines)):
        if not fname.endswith("_job"):
            continue
        job = getattr(routines, fname)(lab)
        if not isinstance(job, Job) or not job.measures_screen:
            continue
        found += 1
        sp = fresh()
        job.body = _empty
        lab.director.start(job)
        check(f"starting the Director job '{job.name}' takes it down "
              f"before its first step", not sp.up and lab.splash is None)
        lab.director.cancel("test cleanup")
    check("at least three measures_screen jobs were found and exercised",
          found >= 3, f"found {found}")

    sp = fresh()
    synthetic = Job(name="a_future_measurement", body=_empty,
                    measures_screen=True)
    lab.director.start(synthetic)
    check("a measurement ADDED LATER, declaring only measures_screen, is "
          "covered with no edit anywhere", not sp.up)
    lab.director.cancel("test cleanup")

    sp = fresh()
    real_busy = lab._screen_busy
    lab._screen_busy = lambda *a: "something"
    lab._publish_measuring()
    lab._screen_busy = real_busy
    check("_publish_measuring reporting busy takes it down", not sp.up)

    sp = fresh()
    lab._open_flash("#000000", timeout_s=0.5)
    check("flashing the whole screen takes it down", not sp.up)
    lab._close_flash()

    print("\n5. The camera is engaged, or fails")
    lab._setup_wanted = lambda: False       # open, but run no chain
    info = {"width": 1280, "height": 720, "fourcc": "MJPG", "fps": 60.0,
            "backend": "MSMF", "controls": {},
            "requested": {"width": 1280, "height": 720, "fourcc": "MJPG"}}
    # The welcome line's clock is STUBBED. The real one is set by whenever the
    # voice tick happened to say the line, so reading it made this section
    # pass or fail by how long the earlier sections took -- and the splash is
    # DELIBERATELY held while the line speaks (his instruction, 2026-10-18:
    # "give the announcer an extra second", "add another second for the
    # splash screen to stay on"), so asserting it is down at once was the old
    # behaviour. Nothing below sleeps for the hold; each case sets the clocks
    # relative to now.
    real_su = lab.voice.speaking_until
    now0 = time.perf_counter

    def line_ended(ago: float) -> None:
        lab.voice.speaking_until = lambda: now0() - ago

    # A: line finished long ago -> down at once.
    sp = fresh()
    line_ended(30.0)
    lab._on_opened(info)
    check("an opened camera takes it down at once when the welcome line "
          "has long finished", not sp.up and lab.splash is None)

    # B: line still speaking -> HELD, then comes down when the hold expires.
    sp = fresh()
    lab.voice.speaking_until = lambda: now0() - lab.SPLASH_EXTRA_S + 0.2
    lab._on_opened(info)
    check("while the welcome line is still speaking the splash is HELD",
          sp.up and lab.splash is sp)
    pump(500)
    check("an opened camera BEGINS its retirement: it is down once the "
          "hold expires, not up for ever", not sp.up and lab.splash is None,
          f"retired: {sp.retire_reason}")

    # B2: the dwell AFTER the line ends is honoured (line ended just now).
    sp = fresh()
    line_ended(0.0)
    lab._on_opened(info)
    check("the extra dwell after the line ends is honoured (still up)",
          sp.up)
    lab._splash_down("test cleanup")

    # C: the cap wins over a voice claiming a minute of speech.
    sp = fresh()
    lab.voice.speaking_until = lambda: now0() + 60.0
    lab._splash_up_at = now0() - lab.SPLASH_HOLD_CAP_S + 0.2
    lab._on_opened(info)
    check("the cap is a hold, not a wait: still up inside the cap", sp.up)
    pump(500)
    check("THE CAP WINS: a voice claiming a minute of speech cannot keep "
          "the splash past SPLASH_HOLD_CAP_S", not sp.up and lab.splash is None)
    sp = fresh()
    lab._splash_up_at = now0() - lab.SPLASH_HOLD_CAP_S - 1.0
    lab._on_opened(info)
    check("past the cap it is down at once whatever the voice says",
          not sp.up)

    # D: a measurement during the hold takes it down at once.
    sp = fresh()
    lab.voice.speaking_until = lambda: now0() + 60.0
    lab._on_opened(info)
    held = sp.up
    flag = owners[0][0]
    setattr(lab, flag, True)
    check("a measurement starting DURING the hold takes it down at once, "
          "without waiting for the hold", held and not sp.up
          and lab.splash is None)
    setattr(lab, flag, False)
    pump(50)
    lab.voice.speaking_until = real_su
    del lab.voice.speaking_until
    lab._splash_up_at = now0()

    sp = fresh()
    lab._on_failed("could not open camera 2 on MSMF")
    pump()
    check("a camera that FAILS to open leaves it up, with the reason under "
          "the logo", sp.up and "could not open camera 2" in sp.problem)
    key(sp, Qt.Key_Escape)
    check("and Escape gets the user out of that too", not sp.up)

    sp = Splash(image_path=None)
    sp.start(app.primaryScreen())
    lab.splash = None
    lab.settings.auto_connect = False
    lab.attach_splash(sp)
    check("with auto-connect off nothing would ever lower it, so attach does",
          not sp.up)

    lab._splash_down("test cleanup")
    owned.check(lambda ok, what, detail="": check(what, ok, detail))

    passed = sum(results)
    # The common dialect (tests/mutation_verdict.py reads it; test_rules section M
    # derives it): "7/7 checks passed" matched no summary pattern, and it printed the
    # same sentence whether or not anything failed.
    if passed == len(results):
        print(f"\nALL {len(results)} CHECKS PASSED")
    else:
        print(f"\n{len(results) - passed} OF {len(results)} CHECKS FAILED")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
