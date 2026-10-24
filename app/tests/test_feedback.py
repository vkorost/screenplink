"""Does the app tell the truth about what it can see? No camera, no display.

This suite exists because of one specific failure, stated by the user: "I don't
want to be in the situation when I shoot the targets and absolutely nothing
happens and I have no idea if I should continue shooting at that target or move
to the next one or stop shooting altogether."

Every check here is about a message rather than a measurement. The thing being
guarded against is not a wrong number -- it is a screen that says nothing while
the shooter keeps firing, which is indistinguishable from a screen that says
nothing because everything is fine.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                 # noqa: E402
from PySide6.QtCore import QRect                                   # noqa: E402
from PySide6.QtGui import QImage, QPainter                         # noqa: E402
from PySide6.QtWidgets import QApplication                         # noqa: E402

from lsa.audio.listener import (Click, ShotCorroborator,           # noqa: E402
                                pick_device)
from lsa.detect.laser import DetectorParams, LaserDetector, ScoreMode  # noqa: E402
from lsa.display.layers import Prompt, SignalMeter, top_band_height  # noqa: E402
from lsa.display.scene import RenderContext                        # noqa: E402
from lsa.display.session import SessionStats                       # noqa: E402

SW, SH = 3840, 2160


def context(now: float = 0.0, dt: float = 0.1) -> RenderContext:
    return RenderContext(now=now, dt=dt, rect=QRect(0, 0, SW, SH),
                         stats=SessionStats())


def frame_with_dot(peak: int, size: int = 6, where=(320, 180)) -> np.ndarray:
    """A red dot of a given brightness on a dark frame."""
    f = np.zeros((360, 640, 3), dtype=np.uint8)
    f[:, :, :] = 12
    x, y = where
    f[y:y + size, x:x + size, 2] = min(255, 12 + peak)   # BGR: red channel
    return f


def main() -> int:
    QApplication.instance() or QApplication([])
    checks: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              f"{(' -- ' + detail) if detail else ''}")

    # ---------------------------------------------------------------- 1 ------
    print("\n1. the detector reports what it ALMOST saw")
    # A shot just under the threshold and a shot that never happened look
    # identical on screen -- no hole, no sound -- and need opposite fixes.
    det = LaserDetector(DetectorParams(mode=ScoreMode.CHROMA, threshold=20,
                                       min_peak=120, use_background=False,
                                       min_area=2, blur=0))
    dets = det.process(frame_with_dot(90))
    check("a dim dot yields no detection", not dets, f"{len(dets)} detections")
    check("but it is counted as too faint", det.last_rejects.get("dim", 0) == 1,
          f"rejects {det.last_rejects}")
    check("and its brightness is reported", det.last_dim_peak >= 60,
          f"dim peak {det.last_dim_peak}")
    check("the frame peak is reported whatever happens",
          det.last_frame_peak >= 60, f"frame peak {det.last_frame_peak}")

    det.set_params(DetectorParams(mode=ScoreMode.CHROMA, threshold=20,
                                  min_peak=40, use_background=False,
                                  min_area=2, blur=0))
    dets = det.process(frame_with_dot(90))
    check("the same dot passes a lower threshold", len(dets) == 1,
          f"{len(dets)} detections")
    check("and then nothing is reported as too faint",
          det.last_rejects.get("dim", 0) == 0, f"rejects {det.last_rejects}")

    # An empty frame must not invent a signal: a meter that reads non-zero
    # with nothing in front of the camera would train the user to ignore it.
    det.process(np.full((360, 640, 3), 12, dtype=np.uint8))
    check("an empty frame reports no signal", det.last_frame_peak <= 2,
          f"frame peak {det.last_frame_peak}")

    # ---------------------------------------------------------------- 2 ------
    print("\n2. a dot OUTSIDE the screen is reported, not silently dropped")
    # This is the signature of a camera that has been knocked: the detector is
    # working perfectly and every shot is being thrown away.
    roi = DetectorParams(mode=ScoreMode.CHROMA, threshold=20, min_peak=40,
                         use_background=False, min_area=2, blur=0,
                         roi_polygon=((10, 10), (200, 10), (200, 150), (10, 150)))
    det.set_params(roi)
    dets = det.process(frame_with_dot(120, where=(400, 250)))
    check("a dot outside the screen is not a detection", not dets)
    check("it is counted as off-screen", det.last_rejects.get("roi", 0) == 1,
          f"rejects {det.last_rejects}")
    check("with its brightness, so the cause is obvious",
          det.last_off_screen_peak >= 60, f"peak {det.last_off_screen_peak}")

    # ---------------------------------------------------------------- 3 ------
    print("\n3. the signal meter holds a pulse long enough to be seen")
    # A LaserLyte pulse is ~100ms and the HUD repaints at ~10Hz. An
    # instantaneous reading would sit flat while the shooter watched it fire.
    meter = SignalMeter()
    meter.report(200, 40, now=0.0, seen_ago=0.0)
    meter.report(0, 40, now=0.1, seen_ago=0.1)
    check("the peak is held, not dropped", meter.hold == 200.0, f"{meter.hold}")
    meter.advance(context(now=0.5, dt=0.4))
    check("and still held a moment later", meter.hold == 200.0, f"{meter.hold}")
    for i in range(20):
        meter.advance(context(now=1.2 + i * 0.1, dt=0.1))
    check("then decays back to the live value", meter.hold == 0.0, f"{meter.hold}")

    meter.report(180, 40, now=0.0, seen_ago=0.2)
    text, _col = meter._verdict()
    check("a passing signal reads as working", "seeing the laser" in text, text)
    meter.report(10, 40, now=0.0, seen_ago=9.0)
    text, _col = meter._verdict()
    check("silence is reported with its age", "9s" in text, text)
    meter.report(10, 40, now=0.0, seen_ago=None)
    text, _col = meter._verdict()
    check("never having seen anything says so", "nothing seen yet" in text, text)
    meter.report(0, 0, live=False, now=0.0)
    text, _col = meter._verdict()
    check("a dead camera is distinguished from a quiet one",
          "not streaming" in text, text)
    meter.report(30, 40, now=0.0, seen_ago=0.1,
                 note="seen but too faint to count", note_kind="warn")
    text, _col = meter._verdict()
    check("an explicit note wins over the generic verdict",
          "too faint" in text, text)

    # It must render at both extremes without raising: the meter is the thing
    # that gets looked at when everything else has gone wrong.
    img = QImage(1280, 720, QImage.Format_RGB32)
    img.fill(0)
    ctx = RenderContext(now=0.0, dt=0.1, rect=QRect(0, 0, 1280, 720),
                        stats=SessionStats())
    ok = True
    for peak, thr, counts in ((0, 0, ()), (255, 40, (("seen", 9), ("shots", 3))),
                              (37, 250, (("heard", 2),))):
        try:
            meter.report(peak, thr, now=0.0, counts=counts)
            p = QPainter(img)
            meter.paint(p, ctx)
            p.end()
        except Exception as exc:
            ok = False
            print("      ", exc)
    check("the meter renders at both extremes", ok)

    # ---------------------------------------------------------------- 4 ------
    print("\n4. the instruction band always says something")
    band = Prompt()
    check("it starts populated, not blank", bool(band.headline), band.headline)
    changed = band.say("READY", "shoot anywhere", "go", "0 shots")
    check("a change is reported so a repaint can be skipped", changed)
    check("an identical update is not a change",
          not band.say("READY", "shoot anywhere", "go", "0 shots"))
    check("every kind it can be asked for exists",
          all(k in Prompt.KINDS for k in ("go", "wait", "warn", "idle")))
    check("the band reserves room at the top", top_band_height(ctx) >= 56,
          f"{top_band_height(ctx)} px at {ctx.height} px tall")
    ok = True
    for kind in Prompt.KINDS:
        try:
            band.say("A LONG HEADLINE THAT MIGHT NOT FIT", "detail line", kind, "9/9")
            p = QPainter(img)
            band.paint(p, ctx)
            p.end()
        except Exception as exc:
            ok = False
            print("      ", exc)
    check("the band renders in every kind", ok)

    # ---------------------------------------------------------------- 5 ------
    print("\n5. heard versus seen")
    # The whole point: three outcomes that used to be one.
    corr = ShotCorroborator(window_ms=300.0)
    corr.add_clicks([Click(t=1.000, level=0.4, floor=0.01)])
    corr.add_shot(1.050)
    check("a shot heard and then seen is one event",
          (corr.heard, corr.seen, corr.both) == (1, 1, 1),
          f"heard {corr.heard} seen {corr.seen} both {corr.both}")
    check("and nothing is left unresolved", not corr.resolve(2.0))

    corr = ShotCorroborator(window_ms=300.0)
    corr.add_shot(5.000)
    corr.add_clicks([Click(t=5.020, level=0.4, floor=0.01)])
    check("the other order works too", corr.both == 1, f"both {corr.both}")

    corr = ShotCorroborator(window_ms=300.0)
    corr.add_clicks([Click(t=10.0, level=0.5, floor=0.01)])
    found = corr.resolve(10.1)
    check("a click is given time to find its shot", not found, str(found))
    found = corr.resolve(10.5)
    check("an unmatched bang is reported as heard-not-seen",
          found == ["heard-not-seen"], str(found))
    check("and counted", corr.heard_not_seen == 1)
    check("it is not reported twice", not corr.resolve(11.0))

    corr = ShotCorroborator(window_ms=300.0)
    corr.add_shot(20.0)
    found = corr.resolve(20.5)
    check("a hit with no bang behind it is reported as a likely phantom",
          found == ["seen-not-heard"], str(found))
    check("and counted", corr.seen_not_heard == 1)

    # Two shots in quick succession must pair up one-to-one rather than one
    # click claiming both -- the shooter fires two guns at once.
    corr = ShotCorroborator(window_ms=300.0)
    corr.add_clicks([Click(t=30.00, level=0.4, floor=0.01),
                     Click(t=30.05, level=0.4, floor=0.01)])
    corr.add_shot(30.01)
    corr.add_shot(30.06)
    check("two bangs pair with two hits, not one",
          (corr.both, corr.heard, corr.seen) == (2, 2, 2),
          f"both {corr.both} heard {corr.heard} seen {corr.seen}")
    check("nothing is left over", not corr.resolve(31.0))

    # A click far from any shot must NOT be paired: that pairing would hide
    # exactly the problem this channel exists to expose.
    corr = ShotCorroborator(window_ms=300.0)
    corr.add_clicks([Click(t=40.0, level=0.4, floor=0.01)])
    corr.add_shot(41.0)
    corr.resolve(42.0)
    check("a bang and a hit a second apart are not the same event",
          (corr.both, corr.heard_not_seen, corr.seen_not_heard) == (0, 1, 1),
          f"both {corr.both}")

    # ---------------------------------------------------------------- 6 ------
    print("\n6. the microphone is chosen, not assumed")
    found = pick_device(preferred_name="", camera_name="USB Video Device")
    # On a machine with no input devices at all this is correctly None; the
    # point of the check is that a generic camera name must not match a
    # completely unrelated device by the word "usb" or "device" alone.
    if found is None:
        check("no microphone is handled", True, "none present")
    else:
        idx, name = found
        check("a microphone is chosen", isinstance(idx, int), f"{idx}: {name}")
        low = name.lower()
        check("and it is not a headset or a software mixer",
              not any(bad in low for bad in
                      ("savi", "plantronics", "headset", "sound mapper",
                       "stereo mix", "line in")), name)
        back = pick_device(preferred_name=name)
        check("a remembered name is honoured", back is not None and back[1] == name,
              str(back))

    # ---------------------------------------------------------------- 7 ------
    print("\n7. the microphone measures the gun instead of guessing")
    # First live run with a guessed threshold: six "a shot was HEARD" reports
    # with nobody shooting. The camera is the channel that can be trusted, so
    # it is what the microphone gets calibrated against.
    from lsa.audio.listener import ClickListener
    lis = ClickListener()
    check("it starts uncalibrated and therefore silent", not lis.calibrated)
    check("and its guessed threshold is conservative", lis.min_level >= 0.1,
          f"{lis.min_level}")

    # Feed it a history as the audio thread would, then confirm three shots.
    t0 = 1000.0
    for i in range(3):
        t = t0 + i
        lis._recent.extend([(t - 0.4, 0.004), (t, 0.62 + i * 0.02), (t + 0.4, 0.006)])
    peaks = [lis.note_seen_shot(t0 + i) for i in range(3)]
    check("each confirmed shot is measured", all(p > 0.5 for p in peaks),
          ", ".join(f"{p:.2f}" for p in peaks))
    check("three of them calibrate it", lis.calibrated)
    check("the threshold lands below a real shot", lis.min_level < lis.learned_level,
          f"{lis.min_level:.3f} < {lis.learned_level:.3f}")
    check("and far above the room", lis.min_level > 0.03 * 5,
          f"{lis.min_level:.3f} against a measured room floor of 0.005")

    # A shot the camera saw with nothing audible must not drag the threshold to
    # zero -- that would re-create the false positives this replaced.
    quiet = ClickListener()
    for i in range(4):
        quiet._recent.append((2000.0 + i, 0.0))
        quiet.note_seen_shot(2000.0 + i)
    check("silence teaches it nothing", not quiet.calibrated,
          f"{len(quiet._learned)} samples")

    # ---------------------------------------------------------------- 8 ------
    print("")
    print("8. nothing on the target screen is sized in fixed pixels")
    # Seen live: the calibration instructions were set in sizes chosen in a
    # 1080p test window. On the real 42-inch 4K panel, read from down the
    # range, they were half the apparent size and effectively unreadable --
    # and an instruction nobody can read is the same as no instruction.
    import re
    layers_src = (Path(__file__).resolve().parents[1]
                  / "src" / "lsa" / "display" / "layers.py").read_text(encoding="utf-8")
    hard = re.findall(r"mono\(\s*\d", layers_src)
    check("no layer picks a font size in raw points", not hard,
          f"{len(hard)} found" if hard else "all scale with the screen")

    from lsa.display.layers import px
    hd = RenderContext(now=0, dt=0, rect=QRect(0, 0, 1920, 1080), stats=SessionStats())
    uhd = RenderContext(now=0, dt=0, rect=QRect(0, 0, 3840, 2160), stats=SessionStats())
    check("a 1080p size is unchanged at 1080p", px(hd, 44) == 44, f"{px(hd, 44)}")
    check("and doubles on a 4K panel", px(uhd, 44) == 88, f"{px(uhd, 44)}")
    check("the top band scales too",
          top_band_height(uhd) == 2 * top_band_height(hd),
          f"{top_band_height(hd)} -> {top_band_height(uhd)}")

    # ---------------------------------------------------------------- 9 ------
    print("")
    print("9. furniture never covers a target")
    # Reported from the firing line: "the upper targets are obscured by text."
    # The scoreboard was drawn over the upper-left aim ring. Relocating the
    # text only moves the collision, because every part of this screen is
    # somewhere a target may legitimately be -- so furniture goes UNDER.
    from lsa.display.layers import (AimField, BackgroundLabel, BulletHoles,
                                    KeyHelp, StatsHud, BOTTOM_STRIP)
    from lsa.modes.base import Mode
    target_z = min(AimField.z, Mode.z, BulletHoles.z)
    for cls in (StatsHud, KeyHelp, BackgroundLabel):
        check(f"{cls.name} draws under the targets", cls.z < target_z,
              f"z {cls.z} vs lowest target z {target_z}")

    # And a mode may not place a target under the bands that are always drawn
    # over everything.
    field = Mode.playfield(uhd)
    check("a mode's playfield starts below the instruction band",
          field.top() >= top_band_height(uhd),
          f"top {field.top():.0f} vs band {top_band_height(uhd)}")
    check("and ends above the bottom strip",
          field.bottom() <= uhd.height - BOTTOM_STRIP,
          f"bottom {field.bottom():.0f} vs {uhd.height - BOTTOM_STRIP}")
    check("but still leaves most of the screen to play in",
          field.height() > uhd.height * 0.7,
          f"{field.height() / uhd.height * 100:.0f}% of the height")

    # --------------------------------------------------------------- 10 ------
    print("")
    print("10. nothing may cover the screen without a way out")
    # On 2026-10-14 the fullscreen calibration panel stayed up permanently
    # because the camera stalled and the frame it was waiting for never came.
    # No title bar, no close button, no key handler: the user had to physically
    # unplug the camera to get their desktop back.
    from PySide6.QtCore import Qt as _Qt, QEvent
    from PySide6.QtGui import QKeyEvent
    from lsa.calibrate.flash import FlashWindow

    fw = FlashWindow()
    seen = []
    fw.cancelled.connect(lambda why: seen.append(why))
    check("it has a timeout at all", fw.DEFAULT_TIMEOUT_S > 0,
          f"{fw.DEFAULT_TIMEOUT_S}s")
    check("and the timeout is not longer than a person will tolerate",
          fw.DEFAULT_TIMEOUT_S <= 15, f"{fw.DEFAULT_TIMEOUT_S}s")

    for key, name in ((_Qt.Key_Escape, "Escape"), (_Qt.Key_Space, "Space"),
                      (_Qt.Key_Q, "Q")):
        seen.clear()
        f2 = FlashWindow()
        f2.cancelled.connect(lambda why: seen.append(why))
        f2.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, _Qt.NoModifier))
        check(f"{name} hides it at once", not f2.isVisible())
        # The announcement is deferred by one event-loop turn on purpose: the
        # owner's handler destroys this window, and doing that while
        # _give_up is still on the stack is a Qt fail-fast crash.
        QApplication.processEvents()
        check(f"{name} closes it", seen == ["escape"], str(seen))

    seen.clear()
    f3 = FlashWindow()
    f3.cancelled.connect(lambda why: seen.append(why))
    f3._give_up("timeout")
    QApplication.processEvents()
    check("the deadline closes it", seen == ["timeout"], str(seen))

    f4 = FlashWindow()
    check("it takes focus, so Escape can reach it",
          f4.focusPolicy() == _Qt.StrongFocus, str(f4.focusPolicy()))
    check("a double click closes it too",
          callable(getattr(f4, "mouseDoubleClickEvent", None)))

    lab_src = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
               / "camera_lab.py").read_text(encoding="utf-8")
    check("no still is requested without a deadline",
          "self.capture.request_still(\"black\")" not in lab_src
          and "self.capture.request_still(\"white\")" not in lab_src,
          "screen-search stills go through _request_still")
    check("and the guard fails the step rather than waiting",
          "_still_timed_out" in lab_src and "_still_guard" in lab_src)

    # --------------------------------------------------------------- 11 ------
    print("")
    print("11. the calibration text can never cover a ring")
    # Reported twice: "some of those targets had been obscured by the text
    # again". Moving the block to the far end of the screen kept the LIVE ring
    # clear and still buried the markers for every other point. The gaps
    # between grid ROWS are 521px and the block is 624px tall, so it never fit;
    # the gaps between COLUMNS are 1474px, so there it always does.
    from PySide6.QtCore import QRectF
    from lsa.display.layers import (CalibrationTargets, band_height_for,
                                    bottom_strip_for)
    from lsa.calibrate.mapping import grid_points

    SW_, SH_ = 3840, 2160
    cpad = int(SH_ * 0.025)
    cband = band_height_for(SH_) + cpad
    cpts = [(gx, gy + cband) for gx, gy in
            grid_points(SW_, SH_ - cband - bottom_strip_for(SH_) - cpad, 3, 3)]
    cctx = RenderContext(now=0, dt=0, rect=QRect(0, 0, SW_, SH_),
                         stats=SessionStats(), calibrating=True)
    ring_r = max(26.0, SH_ / 32.0) * 1.5
    lay = CalibrationTargets()
    lay.start(cpts, need=3)
    covered = 0
    for i in range(len(cpts)):
        lay.index = i
        pxx, pww = lay._panel_column(cctx)
        # Worst case: assume the block fills the whole usable height.
        box = QRectF(pxx, band_height_for(SH_), pww,
                     SH_ - band_height_for(SH_) - 100)
        covered += sum(1 for (gx, gy) in cpts
                       if box.intersects(QRectF(gx - ring_r, gy - ring_r,
                                                2 * ring_r, 2 * ring_r)))
    check("no ring is covered, at any point in the sequence", covered == 0,
          f"{covered} overlaps across all {len(cpts)} points")

    # And the grid itself must stay clear of the two always-on-top bands.
    top_clear = min(gy for _gx, gy in cpts) - ring_r
    bot_clear = max(gy for _gx, gy in cpts) + ring_r
    check("the top row clears the instruction band",
          top_clear > band_height_for(SH_),
          f"ring top {top_clear:.0f} vs band {band_height_for(SH_)}")
    check("the bottom row clears the key legend",
          bot_clear < SH_ - bottom_strip_for(SH_),
          f"ring bottom {bot_clear:.0f} vs {SH_ - bottom_strip_for(SH_)}")

    # --------------------------------------------------------------- 12 ------
    print("")
    print("12. the exposure is chosen by what the detector sees")
    # The bug that cost a whole afternoon: the room was darkened, every
    # candidate's MEAN brightness collapsed, and the rule that picked the mean
    # nearest 25 started choosing on noise. It settled on -11 -- two stops
    # darker than the -9 that worked the day before -- and the recording shows
    # the cost: median detection peak 38 against the 150-230 of a good session.
    from lsa.camera.capture import CaptureThread as CT
    pick = CT._choose_exposure

    # (exposure, mean, frame max, chroma p99)
    lit = [(-4, 52.9, 255, 22.0), (-6, 48.0, 255, 14.0), (-7, 36.0, 255, 9.0),
           (-9, 11.9, 235, 4.0), (-10, 6.9, 255, 3.0), (-11, 3.0, 255, 2.0)]
    chosen = pick(lit, 25.0, 200)
    check("in a lit room it still picks a usable exposure",
          chosen[0] in (-7, -9), f"chose {chosen[0]}")

    # The same camera in a dark room: every mean collapses, so the old rule had
    # nothing left to discriminate on.
    dark = [(-4, 8.0, 255, 6.0), (-6, 5.0, 255, 4.0), (-7, 3.0, 240, 3.0),
            (-9, 1.2, 210, 1.0), (-10, 0.8, 205, 1.0), (-11, 0.5, 200, 0.0)]
    chosen = pick(dark, 25.0, 200)
    check("in a DARK room it opens up instead of stopping down",
          chosen[0] >= -6, f"chose {chosen[0]}")
    # The failure this replaced: at -9 the dot stopped blooming and produced no
    # detectable chroma anywhere in a whole recording, while -7 gave 74-201px
    # blobs from the same gun.
    check("and never stops down to where the dot stops blooming",
          chosen[0] > -9, f"chose {chosen[0]}")
    check("and never picks the darkest just because the room is dark",
          chosen[0] != -11, f"chose {chosen[0]}")

    # A bright room with real chroma noise must still be stopped down, or the
    # noise floor climbs past the signal.
    noisy = [(-4, 90.0, 255, 40.0), (-6, 60.0, 255, 22.0), (-7, 40.0, 255, 12.0),
             (-9, 20.0, 240, 6.0), (-10, 12.0, 220, 3.0), (-11, 8.0, 210, 2.0)]
    chosen = pick(noisy, 25.0, 200)
    check("a noisy room is stopped down until the background is clean",
          chosen[3] <= CT.CHROMA_CLEAN, f"chose {chosen[0]} at chroma {chosen[3]}")

    # The wrap: below -11 this camera reads BRIGHTER again, and a rule that
    # trusts the numbers will happily choose a meaningless setting.
    wrapped = dark + [(-12, 69.5, 255, 30.0)]
    chosen = pick(wrapped, 25.0, 200)
    check("a wrapped reading below the floor is discarded", chosen[0] != -12,
          f"chose {chosen[0]}")

    check("nothing at all still returns something usable",
          pick([], 25.0, 200)[0] == -6.0)

    # --------------------------------------------------------------- 13 ------
    print("")
    print("13. standing the flash panel down does not destroy it mid-call")
    # The safety fix introduced a crash: _give_up emitted `cancelled` inline,
    # the owner's handler set its reference to None, PySide6 destroyed the C++
    # widget, and _give_up carried on calling close() on the corpse. Qt6Core
    # fail-fast, 0xc0000409, every time anyone pressed Escape.
    from lsa.calibrate.flash import FlashWindow as FW

    holder = {}

    def drop(_why):
        # Exactly what the app does: release the only reference.
        holder["win"] = None

    w = FW()
    holder["win"] = w
    w.cancelled.connect(drop)
    w._give_up("escape")
    check("the panel hides itself immediately", not w.isVisible())
    check("the owner is NOT told synchronously", holder["win"] is w,
          "the emit is deferred so the stack can unwind")
    QApplication.processEvents()
    check("and is told once the stack has unwound", holder["win"] is None)

    # A deadline and a keypress arriving together must not emit twice.
    seen2 = []
    w2 = FW()
    w2.cancelled.connect(lambda why: seen2.append(why))
    w2._give_up("escape")
    w2._give_up("timeout")
    QApplication.processEvents()
    check("standing down twice announces it once", seen2 == ["escape"], str(seen2))

    # --------------------------------------------------------------- 14 ------
    print("")
    print("14. calibrating from the screen, with no shooting")
    # The shot grid needs 27 trigger pulls and still only measures the part of
    # the screen the laser registers on -- two columns out of three here, and a
    # projective fit from two columns is wrong everywhere else. The monitor is
    # a light source the camera can already see.
    import numpy as _np
    from lsa.calibrate.pattern import find_marker, marker_points

    pts = marker_points(3840, 2160, cols=4, rows=3, margin=0.08)
    check("the marker grid covers four columns", len({round(x) for x, _y in pts}) == 4,
          f"{len({round(x) for x, _y in pts})} distinct x values")
    check("including the far right the laser cannot reach",
          max(x for x, _y in pts) > 3840 * 0.85,
          f"rightmost at x={max(x for x, _y in pts):.0f}")
    check("and it stays off the extreme edge",
          min(x for x, _y in pts) > 3840 * 0.05)

    def scene_with_marker(cx, cy, bright, size=14, noise=3):
        rng = _np.random.default_rng(7)
        base = rng.integers(0, noise + 1, (720, 1280, 3), dtype=_np.uint8)
        # Something bright and STATIC, like a lamp: must not be mistaken for it.
        base[80:140, 1150:1240] = 240
        lit = base.copy()
        lit[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2] = bright
        return lit, base

    lit, dark = scene_with_marker(400, 300, 200)
    got = find_marker(lit, dark)
    check("a marker is found to sub-pixel accuracy", got is not None
          and abs(got[0] - 400) < 2 and abs(got[1] - 300) < 2,
          f"{got[0]:.1f},{got[1]:.1f}" if got else "not found")

    # The far side of an obliquely-viewed panel is four times dimmer. A fixed
    # threshold finds the near markers and misses the far ones -- exactly how
    # the shot grid ended up with only two columns.
    lit, dark = scene_with_marker(900, 500, 45)
    got = find_marker(lit, dark)
    check("a dim far-side marker is still found", got is not None
          and abs(got[0] - 900) < 3 and abs(got[1] - 500) < 3,
          f"{got[0]:.1f},{got[1]:.1f}" if got else "not found")

    # A static lamp changes nothing between the two photographs.
    same = _np.random.default_rng(1).integers(0, 4, (720, 1280, 3), dtype=_np.uint8)
    same[80:140, 1150:1240] = 240
    check("a steady bright object is not mistaken for a marker",
          find_marker(same, same.copy()) is None)
    check("and nothing at all is reported as nothing",
          find_marker(_np.zeros((720, 1280, 3), _np.uint8),
                      _np.zeros((720, 1280, 3), _np.uint8)) is None)

    # A reflection found instead of a marker took one fit from 0.6mm to 215mm.
    # Least squares cannot help: it splits the difference between a marker and
    # a reflection and gets neither. The grid's own order can.
    from lsa.calibrate.pattern import prune_out_of_order
    clean = [((307, 173), (311.9, 122.0)), ((1382, 173), (556.0, 142.8)),
             ((2458, 173), (743.0, 167.2)), ((3533, 173), (878.0, 188.8))]
    kept, dropped = prune_out_of_order(clean + [None] * 8, 4, 3)
    check("a clean row of markers loses nothing",
          len(kept) == 4 and not dropped, f"kept {len(kept)}, dropped {len(dropped)}")

    every, exactly = True, True
    for i in range(4):
        bad = list(clean)
        bad[i] = (bad[i][0], (1236.6, 266.4))     # a reflection near the frame edge
        k, d = prune_out_of_order(bad + [None] * 8, 4, 3)
        hit = [sc for sc, _c, _o in d]
        if clean[i][0] not in hit:
            every = False
        # And ONLY it. Dropping every point over tolerance at once took an
        # innocent neighbour with each bad marker, leaving six of twelve --
        # two columns, which is the degenerate fit this exists to prevent.
        if hit != [clean[i][0]] or len(k) != 3:
            exactly = False
    check("a bad marker is always dropped, wherever it is in the row", every)
    check("and nothing else is dropped with it", exactly)
    check("and enough markers survive to fit with",
          len(prune_out_of_order([((307, 173), (311.9, 122.0)),
                                  ((1382, 173), (1236.6, 266.4)),
                                  ((2458, 173), (743.0, 167.2)),
                                  ((3533, 173), (878.0, 188.8))] + [None] * 8,
                                 4, 3)[0]) >= 2)

    # A reflection of the marker on something beside the monitor is brighter
    # and larger than the marker itself, being seen face-on rather than at 70
    # degrees. One live run put all twelve "markers" at camera x of 1050-1180
    # when the panel only spans 225-920. No threshold separates them; geometry
    # does, because a marker drawn on the screen appears on the screen.
    from lsa.calibrate.pattern import screen_mask

    def bench(marker_xy, reflection_xy):
        rng = _np.random.default_rng(3)
        dark = rng.integers(0, 3, (720, 1280, 3), dtype=_np.uint8)
        panel = dark.copy()
        panel[100:600, 220:920] = 70            # the monitor, lit
        lit = dark.copy()
        mx, my = marker_xy
        lit[my - 9:my + 9, mx - 9:mx + 9] = 150   # the marker, seen at an angle
        rx, ry = reflection_xy
        lit[ry - 16:ry + 16, rx - 16:rx + 16] = 230  # its reflection, face-on
        return panel, dark, lit

    panel, dark, lit = bench((500, 300), (1100, 300))
    blind = find_marker(lit, dark)
    check("without a mask the reflection wins",
          blind is not None and blind[0] > 1000,
          f"found at x={blind[0]:.0f}" if blind else "nothing")

    mask = screen_mask(panel, dark)
    check("the monitor is located from a full-screen flash", mask is not None)
    if mask is not None:
        check("the mask covers the panel and not the reflection",
              mask[300, 500] > 0 and mask[300, 1100] == 0,
              f"panel={mask[300,500]} reflection={mask[300,1100]}")
        seen = find_marker(lit, dark, mask=mask)
        check("and with it the real marker is found",
              seen is not None and abs(seen[0] - 500) < 4 and abs(seen[1] - 300) < 4,
              f"{seen[0]:.0f},{seen[1]:.0f}" if seen else "nothing")

    # An obliquely-viewed panel is several times dimmer at the far edge, so
    # thresholding splits it into two components with a gap between. Keeping
    # only the largest threw the far third away: the mask ended at camera x 686
    # on a panel reaching 920, the markers over there were never looked for,
    # and the fit fell back to two columns. 0.4mm became 8mm.
    rng2 = _np.random.default_rng(11)
    dark2 = rng2.integers(0, 3, (720, 1280, 3), dtype=_np.uint8)
    split = dark2.copy()
    split[100:600, 220:560] = 90       # near half, bright
    split[100:600, 600:920] = 22       # far third, dim, and NOT touching
    m2 = screen_mask(split, dark2)
    check("the mask spans a panel that thresholds into two pieces",
          m2 is not None and m2[350, 900] > 0,
          "far edge included" if m2 is not None and m2[350, 900] > 0
          else "far edge LOST")
    if m2 is not None:
        ys2, xs2 = _np.nonzero(m2)
        check("and it reaches the far edge of the panel", xs2.max() >= 915,
              f"mask reaches x={xs2.max()}")
        check("without swallowing the whole frame",
              float((m2 > 0).mean()) < 0.55,
              f"{float((m2 > 0).mean()) * 100:.0f}% of frame")

    # The exposure that gets SAVED must be the one chosen for shooting, not
    # whatever the calibration preset last wrote. The screen search drives the
    # camera to -10 by design; persisting that meant each launch booted two
    # stops darker than the last, the dot stopped blooming, and the cause
    # looked like a detector fault for most of an evening.
    lab_src2 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
                / "camera_lab.py").read_text(encoding="utf-8")
    check("the saved exposure comes from the shooting profile",
          "_shooting_exposure" in lab_src2
          and 'st.exposure = float(self.rows["exposure"].spin.value())' not in lab_src2)
    check("and only the exposure decision updates it",
          lab_src2.count("self._shooting_exposure = float(value)") == 1)

    # The real failure: a fixed 15%-of-peak cut excluded the far side of the
    # panel, which is about four times dimmer. The mask reached camera x 694 on
    # a screen extending to 900, so the two right-hand columns of markers were
    # never searched and the fit fell back to two columns -- 0.7mm became 9mm.
    rng3 = _np.random.default_rng(5)
    d3 = rng3.integers(0, 3, (720, 1280, 3), dtype=_np.uint8)
    graded = d3.copy()
    for col, x0 in enumerate(range(220, 920, 100)):
        graded[100:600, x0:x0 + 100] = int(200 * (1 - 0.72 * col / 6))
    m3 = screen_mask(graded, d3)
    check("the mask reaches the dim far side of a graded panel", m3 is not None)
    if m3 is not None:
        _ys3, xs3 = _np.nonzero(m3)
        check("out to the far edge", xs3.max() > 900, f"reaches x={xs3.max()}")
        check("and is still a monitor, not half the frame",
              float((m3 > 0).mean()) < 0.55,
              f"{float((m3 > 0).mean()) * 100:.0f}% of frame")

    # Once four markers are known the rest are predictable, and a search
    # restricted to a disc around the prediction cannot be fooled. Runs that
    # went wrong found something at camera (785,371) for a marker belonging at
    # (878,189), or noise at strength 18 matching no screen position at all.
    rng4 = _np.random.default_rng(21)
    d4 = rng4.integers(0, 3, (720, 1280, 3), dtype=_np.uint8)
    l4 = d4.copy()
    l4[295:305, 495:505] = 120          # the real marker, modest
    l4[360:400, 760:800] = 240          # something far brighter elsewhere
    loose = find_marker(l4, d4)
    check("unconstrained, the brighter distractor wins",
          loose is not None and loose[0] > 700,
          f"x={loose[0]:.0f}" if loose else "nothing")
    aimed = find_marker(l4, d4, near=(500.0, 300.0, 128.0))
    check("searching near the prediction finds the real marker",
          aimed is not None and abs(aimed[0] - 500) < 5 and abs(aimed[1] - 300) < 5,
          f"{aimed[0]:.0f},{aimed[1]:.0f}" if aimed else "nothing")
    check("and reports nothing when the disc is empty",
          find_marker(l4, d4, near=(100.0, 650.0, 40.0)) is None)

    # --------------------------------------------------------------------
    # Two pattern runs at once. Source-level, because the fault is Qt timer
    # re-entrancy and reproducing it needs a real event loop and a camera.
    #
    # Measured on hardware 2026-10-14 18:50: pass 2 was started by the repeat
    # scheduler AND again by the setup-complete path, which fires every time
    # the noise floor is re-measured -- which pattern calibration itself
    # triggers through the sensitivity map. The two runs shared `_pat_index`,
    # so every photograph was taken for a marker other than the one on
    # screen, and all twelve were reported "not visible to the camera".
    # Identical wreckage to the 18:37 run that cost the evening, and it was
    # read as a camera that cannot see the far side of the panel.
    print("\n15. a second pattern run cannot start on top of the first")
    lab_src3 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
                / "camera_lab.py").read_text(encoding="utf-8")
    start = lab_src3.find("def start_pattern_calibration")
    check("start_pattern_calibration exists", start > 0)
    body = lab_src3[start:lab_src3.find(chr(10) + "    def ", start + 10)]
    guard = body.find("if self._pat_active:")
    first_work = min(i for i in (body.find("self.capture is None"),
                                 body.find("self.target_window is None"),
                                 body.find("_apply_preset")) if i > 0)
    check("it refuses while a run is active",
          guard > 0 and "already running" in body,
          "no _pat_active guard in the method")
    check("and refuses BEFORE it touches anything",
          0 < guard < first_work,
          f"guard at {guard}, first side effect at {first_work}")
    # And the launch flag must not be a second trigger.
    #
    # Searched to the END OF THE ENCLOSING METHOD rather than a fixed 1800
    # characters. The window version went red when a comment was added
    # above the guard and pushed it out of range -- a check that fails for
    # a reason it does not name is only slightly better than one that
    # cannot fail at all, and this is a file full of long comments by
    # design, so any character window here is a trap.
    autop = lab_src3.find("if self.auto_patcal:")
    block_end = lab_src3.find(chr(10) + "    def ", autop)
    block = lab_src3[autop:block_end if block_end > 0 else len(lab_src3)]
    check("the launch flag starts exactly one run",
          autop > 0 and "_patcal_done == 0" in block,
          "auto_patcal re-fires on every completed setup")

    # --------------------------------------------------------------------
    print("\n16. the calibration screen shows how far along it is")
    # Asked for directly: "when you're measuring the camera I need you to
    # display the progress bar." It used to show a caption and then eighteen
    # silent flashes, which from the firing line is identical to a hung app.
    from lsa.display.layers import CalibrationPattern         # noqa: E402

    pat = CalibrationPattern()
    pat.progress(7, 18, "5 found so far")
    pat.show_at((1920.0, 1080.0))
    img = QImage(1280, 720, QImage.Format_RGB32)
    img.fill(0)
    painter = QPainter(img)
    pat.paint(painter, RenderContext(now=0.0, dt=0.1,
                                     rect=QRect(0, 0, 1280, 720),
                                     stats=SessionStats()))
    painter.end()
    # The bar must actually mark the screen, in the bottom strip, and it must
    # be GREY -- it is drawn on the surface the detector is watching, so any
    # red excess would be a phantom shot waiting to happen.
    strip = [(x, y) for y in range(660, 719) for x in range(0, 1280, 7)]
    lit = [img.pixelColor(x, y) for x, y in strip]
    drawn = [c for c in lit if c.red() + c.green() + c.blue() > 60]
    check("the progress bar is drawn along the bottom", len(drawn) > 20,
          f"{len(drawn)} lit samples in the bottom strip")
    worst_red = max((c.red() - max(c.green(), c.blue()) for c in lit), default=0)
    check("and carries no red the detector could see", worst_red <= 0,
          f"worst red excess {worst_red}")
    # It advances only BETWEEN markers, so it is identical in the lit and
    # unlit photograph of any one marker and subtracts to exactly zero. If it
    # ever animated, it would become a marker candidate.
    check("the pattern layer is not animated", pat.animated is False)

    print("\n17. the camera-space minimap stands down once calibrated")
    # Reported as: "there's a camera frame uncalibrated on the right and I
    # don't understand why it is there." It is a pre-mapping debug pane, and
    # its own docstring says real bullet holes replace it.
    from lsa.display.layers import CameraSpaceHits             # noqa: E402

    mini = CameraSpaceHits(visible=True)
    mini.add(0.5, 0.5, 1)
    before = QImage(1280, 720, QImage.Format_RGB32)
    before.fill(0)
    q = QPainter(before)
    mini.paint(q, RenderContext(now=0.0, dt=0.1, rect=QRect(0, 0, 1280, 720),
                                stats=SessionStats(), calibrated=False))
    q.end()
    after = QImage(1280, 720, QImage.Format_RGB32)
    after.fill(0)
    q = QPainter(after)
    mini.paint(q, RenderContext(now=0.0, dt=0.1, rect=QRect(0, 0, 1280, 720),
                                stats=SessionStats(), calibrated=True))
    q.end()

    def ink(im):
        return sum(1 for y in range(0, 720, 5) for x in range(0, 1280, 5)
                   if im.pixelColor(x, y).lightness() > 12)

    check("it draws before the mapping exists", ink(before) > 20,
          f"{ink(before)} lit samples")
    check("and nothing at all once it does", ink(after) == 0,
          f"{ink(after)} lit samples")

    print("\n18. the sensitivity map is learned from shots, not from markers")
    # The regression the gun test of 2026-10-14 exposed. Markers measure
    # 227-245 across the whole panel because a white square saturates the
    # sensor wherever it is drawn; the laser's peak ran 84 in the middle
    # against 47 on the right third, over 404px of blob against 149px. Feeding
    # the flat one in produced a gain map of all ones, so the normalisation
    # that exists to save the right third did nothing -- silently, while
    # reporting success. 43% of audible trigger pulls went undetected.
    from lsa.calibrate.mapping import Sample as _S, sensitivity_map as _sm  # noqa: E402

    frame = (1280, 720)
    flat = [_S(screen=(0.0, 0.0), camera=(200.0 + 150 * i, 150.0 + 90 * (i % 4)),
               peak=230.0 + (i % 3)) for i in range(8)]
    g_flat = _sm(flat, frame)
    check("marker-like peaks produce an essentially flat map",
          g_flat is not None and float(g_flat.max() / max(1e-6, g_flat.min())) < 1.1,
          f"{float(g_flat.max() / max(1e-6, g_flat.min())):.3f}x range"
          if g_flat is not None else "no map")
    real = [_S(screen=(0.0, 0.0), camera=(250.0, 300.0), peak=84.0),
            _S(screen=(0.0, 0.0), camera=(450.0, 300.0), peak=80.0),
            _S(screen=(0.0, 0.0), camera=(650.0, 280.0), peak=66.0),
            _S(screen=(0.0, 0.0), camera=(850.0, 240.0), peak=47.0),
            _S(screen=(0.0, 0.0), camera=(300.0, 480.0), peak=76.0),
            _S(screen=(0.0, 0.0), camera=(820.0, 420.0), peak=45.0)]
    g_real = _sm(real, frame)
    check("real laser peaks produce a map that actually varies",
          g_real is not None
          and float(g_real.max() / max(1e-6, g_real.min())) > 1.4,
          f"{float(g_real.max() / max(1e-6, g_real.min())):.2f}x range"
          if g_real is not None else "no map")
    # And the pattern path must no longer build one from markers at all.
    fin = lab_src3.find("def _pattern_finish")
    fin_body = lab_src3[fin:lab_src3.find(chr(10) + "    def ", fin + 10)]
    check("pattern calibration no longer builds a gain map from markers",
          "_apply_sensitivity_map" not in fin_body,
          "it still calls _apply_sensitivity_map with marker peaks")
    learn = lab_src3.find("def _learn_sensitivity")
    check("a shot-driven learner exists instead", learn > 0)
    learn_body = lab_src3[learn:lab_src3.find(chr(10) + "    def ", learn + 10)]
    check("it refuses to fit from shots clustered in one region",
          "LEARN_MIN_BINS" in learn_body,
          "no coverage requirement: a map fitted to one region calls the "
          "others dim and amplifies their noise")
    check("it refuses to learn from the screen's own synthetic dot",
          "_selftest_active" in learn_body,
          "the synthetic dot reads 70/77/76 across the screen where the laser "
          "reads 76/84/47 -- learning from it rebuilds the flat map")
    # BEHAVIOURAL, not a grep for a method name inside one function body.
    # The grep version went red when the floor measurement moved into a
    # shared helper -- a correct refactor, and the seventh time today a
    # source-text check has failed one. What is being asserted is that a
    # map is never trusted without a fresh floor, so assert that.
    import lsa.tools.camera_lab as _CL2                           # noqa: E402
    _mapcalls: list = []

    class _FloorLab:
        _judge_map_against_a_fresh_floor = (
            _CL2.CameraLab._judge_map_against_a_fresh_floor)
        _apply_pending_map_and_remeasure = (
            _CL2.CameraLab._apply_pending_map_and_remeasure)
        # staticmethod(), or the stub passes itself as the first
        # argument -- the stand-in must mirror the real binding too.
        _map_max_amplification = staticmethod(
            _CL2.CameraLab._map_max_amplification)
        _floor_after_gain_map = _CL2.CameraLab._floor_after_gain_map
        LEARN_MAX_FLOOR_RISE = _CL2.CameraLab.LEARN_MAX_FLOOR_RISE

        def __init__(self):
            self._gain_awaiting_baseline = None
            self._gain_pending = None
            self._floor_before_gain = 0.0
            self._noise_peak = 1.0        # the last floor the app measured
            self._sens_applied = 0
            self._sens_shots = []
            # MIRRORS the real lab, which routes every gain-map write through
            # one method so the play-area evidence cannot be left behind in
            # the old units. A stand-in that is missing a signal the real
            # thing has excuses every consumer of it (RULES C12) -- and this
            # is the second stub in one session to break that way.
            self._area_shots = []
            self._calibrating = False
            self.logged = []
            self.capture = self
            self.applied = []
            self.floors_asked = 0

        def _log(self, m):
            self.logged.append(m)

        def _set_gain_map(self, gain, why):
            self.set_gain_map(gain)
            self._area_shots = []

        def set_gain_map(self, g):
            self.applied.append(g)

        def _start_noise_floor(self):
            self.floors_asked += 1

    import numpy as _npf
    from lsa import exposure as _expo_mod
    flat = _npf.full((8, 8), 0.61, dtype=_npf.float32)   # his 55/90 falloff
    fl = _FloorLab()
    fl._judge_map_against_a_fresh_floor(flat, "testing")
    check("the map is applied and then judged on a floor measured with it",
          len(fl.applied) == 1 and fl._gain_pending is not None,
          "normalising lifts noise with signal, so the floor must be "
          "re-measured with the map in place")
    check("and it remembers what the bar was before, to compare against",
          fl._floor_before_gain is not None)
    # AND THE GUARD STILL BITES when the map really is worse. This is the
    # 2026-10-14 incident: floor 11 -> 70 put min_peak at 85 while real
    # shots scored 86-107, so the normalisation meant to save the far side
    # stopped the near side working.
    #
    # Judged against a baseline measured moments earlier rather than one
    # remembered from another exposure -- which is the whole fix -- but
    # judged all the same.
    fl._floor_after_gain_map(38.0)             # bar 8 -> 44, 5.5x
    check("a map that genuinely makes the bar worse is still withdrawn",
          fl.applied[-1] is None,
          "the 2026-10-14 incident: floor 11 -> 70 put min_peak at 85 while "
          "real shots scored 86-107")
    check("and it stands setup back up to measure the floor again",
          fl._calibrating,
          "going back to a uniform score changes what the bar should be")

    # JUDGED ON THE BAR, which is what the detector consumes, and every
    # real case from 2026-10-16 comes out right. On the FLOOR these first
    # two read 3.0x and 4.0x and were withdrawn -- a map doing its job,
    # thrown away, costing him the right-hand third of his screen. The
    # floor at those levels is quantised to nothing: "1" is anywhere in
    # [0.5, 1.5).
    for before, after, keep, when in ((1.0, 3.0, True, "10:53"),
                                      (1.0, 4.0, True, "10:56"),
                                      (1.0, 19.0, False, "11:03, camera moved"),
                                      (11.0, 70.0, False, "2026-10-14")):
        fl_n = _FloorLab()
        fl_n._gain_pending = flat
        fl_n._floor_before_gain = before
        fl_n._floor_after_gain_map(after)
        kept = (fl_n.applied == [])
        check(f"floor {before:.0f} -> {after:.0f} ({when}): "
              f"{'kept' if keep else 'withdrawn'}",
              kept is keep,
              f"bar {_expo_mod.min_peak_for(before):.0f} -> "
              f"{_expo_mod.min_peak_for(after):.0f}")

    print("\n19. a measured mapping survives, and nothing else is used")
    # The cause of "the holes are off by sometimes an inch", found 2026-10-14
    # by reading the gun-test log beside the calibration log:
    #
    #   19:05:11  pattern calibration complete, held-out 0.6mm
    #   19:05:17  recalibrating for this session   <- mapping = None
    #   19:05:30  SHOT 1 ... every shot from here mapped by the FLASH OUTLINE
    #
    # `calibrate_each_session` exists to distrust a mapping saved by an
    # EARLIER session, because a camera nudged in between cannot be detected
    # after the fact. Setup completes more than once -- the noise floor is
    # re-measured after calibration -- so its second pass discarded the
    # mapping measured forty seconds earlier. `_map_to_screen` then fell back
    # to the flash outline, which these notes measure at 78mm mean error here.
    calib = lab_src3.find("def _calibrate_if_needed")
    calib_body = lab_src3[calib:lab_src3.find(chr(10) + "    def ", calib + 10)]
    check("a session's own mapping is not discarded by calibrate_each_session",
          "_mapping_measured_now" in calib_body,
          "a freshly measured mapping is thrown away on the second setup pass")
    # And it must actually be SET somewhere, or the guard is decoration.
    check("the flag is raised when a mapping is measured",
          lab_src3.count("self._mapping_measured_now = True") >= 2,
          f"set in {lab_src3.count('self._mapping_measured_now = True')} places; "
          f"both the pattern and the grid path must raise it")

    m2s = lab_src3.find("def _map_to_screen")
    m2s_body = lab_src3[m2s:lab_src3.find(chr(10) + "    def ", m2s + 10)]
    check("shots are never mapped by the flash outline",
          "screen_quad" not in m2s_body and "self.homography" not in m2s_body,
          "_map_to_screen still falls back to the flash-detected outline, "
          "which measures 78mm mean error on this bench")
    check("and an unmapped shot returns nothing rather than a guess",
          "return None" in m2s_body)

    print("\n19a. the screen can test the camera without a gun")
    # The shooter's idea, and the test this project should have had on day
    # one: "just flash small red dots on screen and see how they are being
    # caught by camera... then you see where the holes appear."
    #
    # It works because of the rule it breaks. Nothing else on the target
    # screen may carry red excess above 30, because the camera watches that
    # screen and red-minus-max(green,blue) is what the detector scores -- so
    # the screen is perfectly able to counterfeit a hit, at a position known
    # to the pixel. The white calibration markers cannot: chroma zero, so they
    # never touch the detector at all.
    from lsa.display.layers import SyntheticLaser                  # noqa: E402

    synth = SyntheticLaser()
    synth.show_at((640.0, 360.0), 30.0)
    img5 = QImage(1280, 720, QImage.Format_RGB32)
    img5.fill(0)
    q5 = QPainter(img5)
    synth.paint(q5, RenderContext(now=0.0, dt=0.1, rect=QRect(0, 0, 1280, 720),
                                  stats=SessionStats()))
    q5.end()
    mid = img5.pixelColor(640, 360)
    excess = mid.red() - max(mid.green(), mid.blue())
    check("the synthetic dot is exactly what the detector hunts", excess > 200,
          f"red excess {excess}")
    synth.stop()
    img6 = QImage(1280, 720, QImage.Format_RGB32)
    img6.fill(0)
    q6 = QPainter(img6)
    synth.paint(q6, RenderContext(now=0.0, dt=0.1, rect=QRect(0, 0, 1280, 720),
                                  stats=SessionStats()))
    q6.end()
    check("and it draws nothing at all when stopped",
          img6.pixelColor(640, 360).red() == 0)

    st = lab_src3.find("def start_self_test")
    check("the self test exists", st > 0)
    st_body = lab_src3[st:lab_src3.find(chr(10) + "    def ", st + 10)]
    check("it refuses without a mapping to check against",
          "self.mapping is None" in st_body,
          "with no mapping there is nothing to compare a hit against")
    # The interlock. Its absence is what the first run of this test exposed:
    # red dots drawn during the noise floor BECOME the noise floor --
    # measured 78 against a true 11, min_peak 93, everything after it lost.
    #
    # Asserted as BEHAVIOUR now, not as the presence of `self._calibrating`
    # in the source. The textual version passed for a day while five of six
    # routines guarded against a different subset of the flags, and it went
    # red the moment the routine was refactored into something correct.
    from lsa.director import Director as _D1, ManualClock as _MC1
    from lsa import routines as _r1
    _busy = {"who": "a noise-floor measurement"}
    _d1 = _D1(_MC1(), sink=None, log=lambda _s: None,
              world_busy=lambda: _busy["who"])

    class _Stub1:
        PROBE_STEPS = (0.0,)
        SELFTEST_TUNE_SHOTS = 1
        _shooting_exposure = -7.0
    check("it waits for a noise measurement rather than flashing into one",
          _d1.start(_r1.self_test_job(_Stub1())) is False,
          "red dots drawn during the noise floor BECOME the noise floor")
    _busy["who"] = ""
    check("and runs once the measurement is over",
          _d1.start(_r1.self_test_job(_Stub1())) is True,
          "refusing for ever is not waiting")
    _d1.cancel()
    nm = lab_src3.find("def _on_noise_measured")
    nm_body = lab_src3[nm:lab_src3.find(chr(10) + "    def ", nm + 10)]
    check("and a floor measured while the screen is lit is discarded",
          "_selftest_active" in nm_body and "_pat_active" in nm_body,
          "the app warns 'do NOT fire during this' and never enforced it")

    print(chr(10) + "19h. the laser's falloff is measured once, not every session")
    # The deadlock that made the learner impossible to satisfy, and the
    # shooter met it three times in twenty minutes: it needs 12 shots over 4
    # regions, the uncorrected threshold rejects the very edge shots that
    # would teach it, and every session threw the evidence away and asked for
    # more. Across 286 logged shots the evidence had been there since the
    # first string.
    #
    # The falloff is a property of this room, this camera pose and this
    # screen -- the same things the mapping describes, invalidated by the same
    # event -- so it belongs in the same file and is discarded with it.
    from lsa.calibrate.mapping import Mapping as _M, Sample as _S2   # noqa: E402
    import numpy as _np2, tempfile                                   # noqa: E402
    from pathlib import Path as _P                                   # noqa: E402

    mm = _M(homography=_np2.eye(3), samples=[_S2((0, 0), (1, 1), 50.0)],
            screen_size=(3840, 2160))
    mm.sensitivity = [{"camera": [300.0, 250.0], "peak": 84.0},
                      {"camera": [860.0, 280.0], "peak": 50.0}]
    tmp = _P(tempfile.gettempdir()) / "lsa_cal_roundtrip.json"
    mm.save(tmp)
    back = _M.load(tmp)
    check("a measured falloff survives save and load",
          back is not None and back.sensitivity == mm.sensitivity,
          "it is relearned from scratch every launch, which is why it could "
          "never gather enough shots")
    try:
        tmp.unlink()
    except OSError:
        pass
    # BEHAVIOURAL, not a grep for an attribute name. The grep version was
    # `"_prev_sensitivity" in lab_src3`, which passes when the field is
    # renamed into equivalence and fails when the code is refactored into
    # correctness -- backwards from what a refactor needs. It also could not
    # see the actual bug: `dot_radius`, added the next day, was captured by
    # nothing and carried by nothing, which is the sensitivity bug of the day
    # before reproduced in a second field inside a day (Fable, 2026-10-16).
    #
    # So: drive the real capture and the real carry, on a real CameraLab,
    # and assert that EVERY pose fact survives a recalibration.
    import lsa.tools.camera_lab as _CL                            # noqa: E402

    class _FakeLab:
        POSE_FACTS = _CL.CameraLab.POSE_FACTS
        _capture_pose_facts = _CL.CameraLab._capture_pose_facts
        _carry_pose_facts = _CL.CameraLab._carry_pose_facts

        def __init__(self, mapping):
            self.mapping = mapping
            self._prev_pose = {}
            self.logged = []

        def _log(self, msg):
            self.logged.append(msg)

    old = _M(homography=_np2.eye(3), screen_size=(3840, 2160))
    old.sensitivity = [{"camera": [300.0, 250.0], "peak": 84.0}]
    old.dot_radius = 63.0
    # EVERY pose fact, set from the list rather than by hand. The check
    # below asserts over `POSE_FACTS`, so a fixture that sets only the facts
    # that existed when it was written lets a new one pass by omission.
    _unset = [f for f in _CL.CameraLab.POSE_FACTS if not getattr(old, f, None)]
    assert not _unset, f"this fixture does not set {_unset}"
    fl = _FakeLab(old)
    fl._capture_pose_facts()
    fl.mapping = None                  # calibrate_each_session discards it
    fresh = _M(homography=_np2.eye(3), screen_size=(3840, 2160))
    fl._carry_pose_facts(fresh)
    missing = [f for f in _CL.CameraLab.POSE_FACTS if not getattr(fresh, f, None)]
    check("every pose fact is carried across a recalibration",
          not missing,
          f"{missing} did not survive -- a fresh calibration must not force "
          f"the shooter to reteach what the camera's position already "
          f"measured, and these are captured and applied as ONE set so a "
          f"third field cannot be half-applied")
    # And a FRESHLY measured value must win over the carried one.
    fresh2 = _M(homography=_np2.eye(3), screen_size=(3840, 2160))
    fresh2.dot_radius = 11.0
    fl._carry_pose_facts(fresh2)
    check("but a freshly measured one is not overwritten by it",
          fresh2.dot_radius == 11.0,
          "carrying forward must never beat a measurement taken just now")
    # AND NOTHING IS CARRIED WHEN THE CAMERA HAS MOVED.
    #
    # Both facts describe WHERE THE CAMERA WAS. His log, 11:02:37: "the
    # camera HAS moved: 512.5px", and 30 seconds later "re-using the
    # sensitivity measured earlier for this camera position" -- false when
    # printed. The map built from it amplified room instead of panel.
    # Today's POSE_FACTS generalisation made it worse: before, only the
    # sensitivity was carried; afterwards the dot radius went too.
    import ast as _ast_m
    _moved = _ast_m.parse(lab_src3)
    _fnm = next(n_ for n_ in _ast_m.walk(_moved)
                if isinstance(n_, _ast_m.FunctionDef)
                and n_.name == "_on_mapping_checked")
    _clears = [n_ for n_ in _ast_m.walk(_fnm)
               if isinstance(n_, _ast_m.Assign)
               and any(isinstance(t, _ast_m.Attribute)
                       and t.attr == "_prev_pose"
                       for t in n_.targets)]
    check("a measured camera move discards the old pose's facts",
          bool(_clears),
          "_on_mapping_checked clears the mapping and must clear _prev_pose "
          "with it, or the next calibration inherits a falloff measured for "
          "a view that no longer exists")
    check("there is a path that re-applies it without shooting",
          "def _restore_sensitivity" in lab_src3)
    # BY AST: the remembered map must go through the same judging as a
    # freshly learned one -- a fresh baseline floor, then the map, then the
    # comparison. Matched on the CALL rather than on method names inside a
    # body, which is what broke when the judging moved into one shared
    # helper (a correct refactor, and the seventh grep to fail one today).
    import ast as _ast2
    _fn2 = next(n_ for n_ in _ast2.walk(_ast2.parse(lab_src3))
                if isinstance(n_, _ast2.FunctionDef)
                and n_.name == "_restore_sensitivity")
    _calls2 = {n_.func.attr for n_ in _ast2.walk(_fn2)
               if isinstance(n_, _ast2.Call)
               and isinstance(n_.func, _ast2.Attribute)}
    check("a remembered map is judged exactly like a freshly learned one",
          "_judge_map_against_a_fresh_floor" in _calls2,
          f"calls {sorted(_calls2)} -- a map re-used from an earlier "
          f"session must still be measured against today's floor, and "
          f"against a baseline taken in the same conditions")

    print(chr(10) + "19g. a shot during the noise measurement is caught")
    # Three times in one evening, including twice while the shooter was
    # standing at the line waiting to be told to fire:
    #
    #   21:31:08  measuring noise floor for 3s - do NOT fire during this
    #   21:31:09  SHOT 1 ... peak=89
    #   21:31:10  SHOT 2 ... peak=89
    #   21:31:11  noise floor: peak 89 -> min peak to 104
    #
    # His words: "It worked for the first few shots and then it stopped
    # working." Those first few shots ARE why it stopped: they became the
    # noise floor, and the bar then sat above every real hit for the rest of
    # the session. The app warned and never checked, though the detector had
    # already reported the shots -- the two facts were never put side by side.
    nm2 = lab_src3.find("def _on_noise_measured")
    nm2_body = lab_src3[nm2:lab_src3.find(chr(10) + "    def ", nm2 + 10)]
    check("the measurement is checked against recent shots",
          "_last_shot_at" in nm2_body,
          "a shot during the window silently becomes the noise floor")
    # The first version of this guard read `self.capture.stats.last_shot_at`,
    # which does not exist -- the capture keeps it as `_stats`. getattr
    # returned 0.0 and the guard never fired once. So assert the attribute is
    # REAL and is written where shots arrive, not merely that the source
    # mentions a plausible name.
    import lsa.tools.camera_lab as _cl                           # noqa: E402
    check("the timestamp it reads is one the app itself records",
          "self._last_shot_at = time.perf_counter()" in lab_src3,
          "it reads an attribute nobody sets, which is worse than no guard "
          "because it looks like protection")
    onshot = lab_src3.find("def _on_shot(self, shot")
    check("and it is recorded before any early return in _on_shot",
          lab_src3.find("self._last_shot_at = time.perf_counter()", onshot)
          < lab_src3.find("if self._grid_active", onshot),
          "shots routed to the grid or the self test must still count")
    check("a spoiled measurement is discarded and retried",
          "_noise_retries" in nm2_body
          and ("start_noise_calibration" in nm2_body
               or "_start_noise_floor" in nm2_body),
          "it must re-measure, not accept a floor set by a trigger pull")
    check("and it gives up rather than looping forever",
          "_noise_retries < 5" in nm2_body,
          "someone shooting continuously would retry for ever; but two was "
          "fewer than a shooter mid-string, and giving up means accepting a "
          "floor set by a trigger pull")
    # The warning text must still exist: the guard is a backstop, not a
    # licence to stop telling the shooter what is happening.
    check("the shooter is still told to hold fire",
          "do NOT fire during this" in lab_src3 or "hold fire" in nm2_body)

    print(chr(10) + "19f. a bad screen mask is detected and discarded mid-run")
    # Correlation across seventeen runs on this bench, and it is total:
    #
    #   mask 44%   corners valid     held-out 0.6 - 1.0mm
    #   no mask    corners valid     held-out 0.6 - 1.1mm
    #   mask 29%   corners INVALID   held-out 386mm
    #   mask 52%   corners INVALID   held-out 298mm
    #
    # Both disasters had a mask that passed the generic 8-60% plausibility
    # band and was still wrong -- too small, so the panel was clipped and
    # corners landed on the clip edge; too large, so the room came in and a
    # reflection won one. Ten runs with NO mask were uniformly good.
    #
    # So the mask is a hypothesis to test, not an assumption to carry. The
    # corner check already detects the failure every time; the run now acts
    # on that instead of continuing.
    susp = lab_src3.find("def _pat_corners_suspect")
    check("there is a mid-run recovery from a bad mask", susp > 0,
          "a mask that ruins the corners is carried to the end of the run")
    susp_body = lab_src3[susp:lab_src3.find(chr(10) + "    def ", susp + 10)]
    check("it throws the mask away rather than the run",
          "_pat_mask = None" in susp_body and "_pat_index = 0" in susp_body,
          "it must re-measure the corners, not abandon calibration")
    check("and it does so at most once",
          "_pat_mask_retried" in susp_body,
          "without a latch this retries forever on a genuinely odd screen")
    # It must only fire on the CORNERS, before the grid has been walked --
    # restarting halfway would discard good markers.
    check("only at the end of the corner phase",
          "_pat_index != 4" in susp_body,
          "restarting mid-grid would throw away markers already found")
    check("and it fires on MISSING corners, not only misplaced ones",
          "len(corners) < 4" in susp_body,
          "a mask that clips the panel hides corners outright, so the quad "
          "test never gets four points: that is the 6-point, 386mm failure")
    # And it must be wired in.
    still = lab_src3.find("def _on_pattern_still")
    check("the check runs after each corner is recorded",
          "_pat_corners_suspect()" in
          lab_src3[still:lab_src3.find(chr(10) + "    def ", still + 10)],
          "_pat_corners_suspect is never called")

    print(chr(10) + "19e. furniture must not repaint a 4K screen at 60fps")
    # Measured 2026-10-14: `stats` and `signal` each claimed a change on every
    # tick, so a 3840x2160 target screen repainted 60 times a second forever,
    # at 8.53 ms a paint -- about half a CPU second per wall second, spent on
    # text nobody can read at that rate. The detector needs 6.96 ms per camera
    # frame out of the same budget and was measuring 24 ms live, which capped
    # the app near 32 fps. At 32 fps a 100 ms pulse spans 3 camera frames
    # instead of 6, halving the evidence behind every shot.
    #
    # The signal meter is the sharpest case: its own docstring says it holds a
    # peak for 1.1 s "because a 100 ms pulse against a 10 Hz repaint would
    # otherwise never be seen". It was built for 10 Hz and demanded 60.
    from lsa.display.layers import StatsHud as _SH, SignalMeter as _SM  # noqa: E402

    rect4k = QRect(0, 0, 3840, 2160)

    def dirty_rate(layer, seconds=1.0, hz=60.0):
        t, hits, n = 0.0, 0, int(seconds * hz)
        for _ in range(n):
            t += 1.0 / hz
            if layer.advance(RenderContext(now=t, dt=1.0 / hz, rect=rect4k,
                                           stats=SessionStats())):
                hits += 1
        return hits / seconds

    for name, layer, cap in (("stats", _SH(), 20.0), ("signal", _SM(), 20.0)):
        rate = dirty_rate(layer)
        check(f"{name} does not force a repaint every tick", rate <= cap,
              f"{rate:.0f} repaints/s of a 4K screen at 8.5ms each")
    # And a moving target must still animate smoothly -- the throttle applies
    # to furniture, never to the thing being shot at.
    from lsa.modes.balls import BouncingBalls as _BB                   # noqa: E402

    bb_ = _BB()
    bb_.start(RenderContext(now=0.0, dt=1 / 60, rect=rect4k, stats=SessionStats()))
    check("an animated mode is not throttled", dirty_rate(bb_) >= 55.0,
          f"{dirty_rate(bb_):.0f}/s -- a target that stutters is worse than "
          f"a HUD that lags")

    print(chr(10) + "19d. a calibration that fails its own check is thrown away")
    # Met head-on by the shooter: "I just did a few shots and holes are way,
    # way off." The run before it had logged, in order:
    #
    #   pattern calibration complete: 6 points, RMS 575px (140.8mm)
    #   HELD-OUT ERROR: median 386.1mm, worst 1143.2mm
    #   saved, but the held-out error is large.
    #
    # Six points is the degenerate two-column fit the notes warn about by
    # name. The app had PROVEN the mapping was a metre wrong, saved it, and
    # drew holes from it -- telling the shooter his aim was fifteen inches
    # out. `ok` was computed on the line above and never used.
    fin3 = lab_src3.find("def _report_pattern_check")
    fin3_body = lab_src3[fin3:lab_src3.find(chr(10) + "    def ", fin3 + 10)]
    check("the held-out verdict actually gates the mapping",
          "if not usable:" in fin3_body and "_pat_reject" in fin3_body,
          "the mapping is installed whatever the held-out check said")
    rej = lab_src3.find("def _pat_reject")
    check("there is a rejection path", rej > 0)
    rej_body = lab_src3[rej:lab_src3.find(chr(10) + "    def ", rej + 10)]
    check("rejection leaves the app uncalibrated, not on a stale mapping",
          "self.mapping = None" in rej_body,
          "falling back to the previous mapping is the same lie, older")
    check("and it says so on the target screen",
          "_alert" in rej_body and "NOT USING IT" in rej_body,
          "a silent rejection reads as a detector that stopped working")
    _map_src = (Path(__file__).resolve().parents[1] / "src" / "lsa"
                / "calibrate" / "mapping.py").read_text(encoding="utf-8")
    check("the bar is far above a good fit and far below a visible error",
          "HELD_OUT_USABLE_MM = 40.0" in _map_src,
          "a good run measures ~1mm; the degenerate one measured 386mm")

    # 19d-2. ONE BAD HELD-OUT MARKER MUST NOT DESTROY A GOOD CALIBRATION.
    #
    # 2026-10-18 10:04:39 on his bench, and it cost him the whole session.
    # These are the REAL numbers from that log, not a fixture resembling
    # them: twelve fit markers at RMS 3px (0.8mm), worst 5px (1.3mm), and six
    # held-out markers of which FIVE measured 0.1-2.1mm and one, at screen
    # (845,1685), landed 804px = 196.8mm out. The app judged on the WORST,
    # printed "CALIBRATION REJECTED: held-out error 1mm (worst 197mm)" and
    # threw a 1.3mm mapping away.
    #
    # It was a FALSE FIND, not a bad fit: a known-good mapping measured
    # minutes earlier puts that screen point at camera (456,517) and this run
    # "found" it at (600,594). The log says "lit frame peaks at 255, unlit at
    # 255" and every marker reported strength 255 -- a saturated sensor
    # leaves the finder nothing to discriminate on.
    #
    # He then fired 42 shots. All 42 detected, peaks 76-109 against a bar of
    # 8, and not one hole.
    from lsa.calibrate.mapping import (judge_held_out,                 # noqa: E402
                                       HELD_OUT_USABLE_MM,
                                       HELD_OUT_WILD_MM)
    MMPX = 0.2448                     # 196.8mm / 804px, from his own log
    his_errs = [(0.1 / MMPX, (845, 1685)), (2.1 / MMPX, (1200, 400)),
                (1.0 / MMPX, (2600, 400)), (1.3 / MMPX, (1200, 1685)),
                (0.8 / MMPX, (2600, 1685)), (804.0, (845, 1685))]
    jv = judge_held_out(his_errs, MMPX)
    check("his 2026-10-18 calibration is ACCEPTED (median 1.3mm, worst 197mm)",
          jv is not None and jv.usable and jv.good,
          f"median {jv.median_mm:.1f}mm worst {jv.worst_mm:.0f}mm "
          f"usable={jv.usable} good={jv.good}")
    check("  ... judged on the MEDIAN, which is what Mapping._reject_outliers "
          "already does to the fit", abs(jv.median_mm - 1.3) < 0.1,
          f"{jv.median_mm:.2f}mm")
    check("  ... and the outlier is REPORTED, not silently dropped",
          len(jv.wild) == 1 and abs(jv.wild[0][1] - 196.8) < 0.5,
          f"wild={[(round(a), round(b)) for a, b, _ in jv.wild]}")
    check("  ... and the app logs that marker by name, size and likely cause",
          all(t in fin3_body for t in ("OUTLIER", "verdict.wild",
                                       "FALSE FIND" if "FALSE FIND" in fin3_body
                                       else "DIFFERENT POINT")),
          "a silent drop is how a reflection once won ten of eighteen markers")
    # A GENUINELY BAD CALIBRATION MUST STILL BE REJECTED, both ways.
    #
    # The real collapsed fits, measured 2026-10-14/15/16: median 298mm,
    # 386mm and 414mm. The median catches every one of them.
    for med_mm in (298.0, 386.0, 414.0):
        bad = [(med_mm / MMPX, (x, 100)) for x in (100, 200, 300, 400, 500)]
        bv = judge_held_out(bad, MMPX)
        check(f"a collapsed fit (median {med_mm:.0f}mm) is still REJECTED",
              not bv.usable and not bv.good, bv.why[:70])
    # And several wild markers is a DIFFERENT fault -- the finder is not
    # reliable across the panel -- which no median can repair, so it is
    # rejected even when the median is perfect.
    many_wild = [(1.0 / MMPX, (100, 100)), (1.2 / MMPX, (200, 100)),
                 (0.9 / MMPX, (300, 100)), (804.0, (400, 100)),
                 (700.0, (500, 100)), (900.0, (600, 100))]
    wv = judge_held_out(many_wild, MMPX)
    check("3 of 6 held-out markers wild is REJECTED even at a 1.0mm median",
          not wv.usable, f"median {wv.median_mm:.1f}mm, {len(wv.wild)} wild, "
          f"allowance {wv.allowed_wild}")
    check("  ... and it says WHY, naming the count rather than the median",
          "marker finder" in wv.why and str(len(wv.wild)) in wv.why,
          wv.why[:90])
    # And the MEDIAN branch is the one that bites when there are too few
    # held-out markers to tell a false find from a bad fit. With two check
    # points, one of them 804px out, the allowance of one is not exceeded --
    # so only the median can refuse, and it must.
    thin = [(1.0 / MMPX, (100, 100)), (804.0, (845, 1685))]
    tv = judge_held_out(thin, MMPX)
    check("two held-out markers, one wild, is REJECTED on the median",
          not tv.usable and len(tv.wild) <= tv.allowed_wild,
          f"median {tv.median_mm:.0f}mm, {len(tv.wild)} wild of "
          f"{tv.points}, allowance {tv.allowed_wild}: {tv.why[:60]}")
    check("  ... and it says the mapping is wrong EVERYWHERE, not at one marker",
          "MEDIAN" in tv.why, tv.why[:80])
    # Absent evidence is neither a pass nor a failure (RULES C1 both ways).
    check("no held-out markers at all judges nothing",
          judge_held_out([], MMPX) is None)
    # The rejection path must carry the reason to the log, not just a number.
    check("the rejection path is given the reason and prints it",
          "why: " in rej_body and "why" in fin3_body,
          "C3: a check that reports only a verdict cannot be debugged")

    # 19d-3. A REJECTED CALIBRATION MUST PUBLISH A VERDICT.
    #
    # The defect that actually cost him the session. The rejection raised a
    # 14-second alert and then the screen went quiet: between 10:04:39 and the
    # end of his 42-shot string there is not ONE log line about the app's own
    # state, and no verdict was ever published. From the firing line, a
    # correct refusal to draw holes is indistinguishable from a broken app.
    #
    # This project has shipped this exact fault once before -- "the verdict
    # was not published when nothing was detected: an early return in the
    # self test skipped it in the one case the verdict exists for"
    # (2026-10-15). Second instance, in a second path (RULES E1).
    check("the rejection publishes a verdict",
          "_publish_verdict" in rej_body,
          "the verdict board is the app's only unmissable channel, and the "
          "state that most needs it reached it by no path at all")
    check("  ... and a failure to report is not also a crash",
          "could not publish the verdict" in rej_body,
          "he is at the line with a pistol and no holes")
    # And the verdict must have something to say: `selfcheck` has to be TOLD
    # there is no mapping, or it assesses a healthy camera with nothing fired
    # as "READY - shoot."
    coll = lab_src3.find("def _collect_selfcheck")
    coll_body = lab_src3[coll:lab_src3.find(chr(10) + "    def ", coll + 10)]
    check("and the self check is told whether there is a mapping at all",
          "calibrated=self.mapping is not None" in coll_body,
          "without it assess() returns READY for an app that can draw nothing")

    print("\n19c. calibration gives every layer back")
    # Reported as: "Currently Pressing 2 or 3 brings nothing on screen."
    # start_pattern_calibration hid twelve layers; _pattern_finish restored
    # seven. The five that fell through the gap stayed hidden for the rest of
    # the session -- `modes`, so every arcade mode started invisible, and
    # `alert`, so the app's warnings were suppressed exactly when it had most
    # to say. Two hand-written lists that had to agree, and did not.
    hide_at = lab_src3.find("def start_pattern_calibration")
    hide_body = lab_src3[hide_at:lab_src3.find(chr(10) + "    def ", hide_at + 10)]
    fin_at = lab_src3.find("def _pattern_finish")
    fin_body2 = lab_src3[fin_at:lab_src3.find(chr(10) + "    def ", fin_at + 10)]
    check("what was hidden is recorded, not retyped",
          "_pat_hidden[name] = layer.visible" in hide_body,
          "the hidden set is not captured, so it cannot be restored faithfully")
    check("and restored from that record",
          "_pat_hidden" in fin_body2 and "layer.visible = was" in fin_body2,
          "_pattern_finish still restores a hand-written list")
    # The gap is the bug, so assert it cannot reopen: every name hidden must
    # be a name the restore loop can reach.
    import re as _re
    hidden_names = set(_re.findall(r'"(\w+)"', hide_body[
        hide_body.find("self._pat_hidden = {}"):
        hide_body.find("layer.visible = False")]))
    check("modes and alert are among the layers hidden",
          {"modes", "alert"} <= hidden_names,
          f"hidden: {sorted(hidden_names)}")
    check("and the restore is driven by the same collection",
          "for name, was in" in fin_body2,
          "restoring by a literal list lets the two drift apart again")

    print("\n19b. a gain map that makes detection worse is withdrawn")
    # Measured 2026-10-14 19:43, and it is the worst kind of regression --
    # the fix for one complaint caused a bigger version of the same complaint.
    #
    #   before the map   floor 11  ->  min_peak 26
    #   after the map    floor 70  ->  min_peak 85
    #
    # Real shots in the middle of the screen were scoring 86-107, so a
    # min_peak of 85 rejected most of them: normalising to rescue the far side
    # stopped the near side working. One stray sample caused it -- a
    # calibration artefact scoring 23, alone in its bin, so the median of that
    # bin was the stray, and the map amplified that region's noise by 3.7x.
    learn2 = lab_src3.find("def _learn_sensitivity")
    learn2_body = lab_src3[learn2:lab_src3.find(chr(10) + "    def ", learn2 + 10)]
    check("a single reading cannot speak for a region",
          "LEARN_MIN_PER_BIN" in learn2_body,
          "a bin with one sample still sets that ninth of the screen")
    # Same AST treatment: the learner hands its map to the judging helper
    # rather than setting the bookkeeping itself. Covered behaviourally
    # above; this asserts the learner is wired to it at all.
    _fnL = next(n_ for n_ in _ast2.walk(_ast2.parse(lab_src3))
                if isinstance(n_, _ast2.FunctionDef)
                and n_.name == "_learn_sensitivity")
    _callsL = {n_.func.attr for n_ in _ast2.walk(_fnL)
               if isinstance(n_, _ast2.Call)
               and isinstance(n_.func, _ast2.Attribute)}
    check("the map is remembered for judging, not just applied",
          "_judge_map_against_a_fresh_floor" in _callsL,
          f"calls {sorted(_callsL)} -- a map applied and never judged is "
          f"the 2026-10-14 incident, where it put min_peak at 85 against "
          f"real shots scoring 86-107")
    verdict = lab_src3.find("def _floor_after_gain_map")
    check("there is a verdict step at all", verdict > 0,
          "nothing ever compares the floor before and after")
    v_body = lab_src3[verdict:lab_src3.find(chr(10) + "    def ", verdict + 10)]
    # BEHAVIOURAL, because the text version of this check pinned
    # `set_gain_map(None)` exactly and went red the moment the call was
    # routed through the lab's own wrapper -- a correct refactor failing a
    # check for a reason the check could not name. That is the ninth grep of
    # this shape on this project; the withdrawal itself is already asserted
    # on behaviour above (`fl.applied[-1] is None`), so what is added here is
    # the property the wrapper exists for.
    flw = _FloorLab()
    flw._area_shots = [(100.0, 200.0, 42.0)]
    flw._judge_map_against_a_fresh_floor(flat, "testing")
    flw._floor_after_gain_map(38.0)
    check("a map that worsens the floor is withdrawn",
          flw.applied[-1] is None and "WITHDRAWING" in v_body)
    check("  and withdrawing it retires the play-area evidence with it",
          flw._area_shots == [],
          "a peak measured WITH a gain map and one measured without it are "
          "not the same quantity. His border stayed fenced over a third of "
          "the screen because the median was taken across both")
    check("and the samples behind it are dropped too",
          "_sens_shots = []" in v_body,
          "keeping them rebuilds the same map at the next doubling")
    nm0 = lab_src3.find("def _on_noise_measured")
    check("the verdict is actually wired to the measurement",
          "_floor_after_gain_map(" in
          lab_src3[nm0:lab_src3.find(chr(10) + "    def ", nm0 + 10)],
          "_floor_after_gain_map is never called from _on_noise_measured")

    print("\n19c. the SHOT GRID's gain map takes the same three steps as the learner")
    # 2026-10-23. `_apply_sensitivity_map` (grid) and `_learn_sensitivity`
    # (play) both build the detector's gain map from laser peaks, and the grid
    # path had three faults the learner had already fixed:
    #   1. not confined to the screen  (floor 19 -> 112 on 2026-10-14 21:33;
    #      the same map read 1.11x mid-screen, 1.00x at the window confined,
    #      1.44x unconfined)
    #   2. never saved, so the best evidence the app gets died with the session
    #   3. never judged: it never set `_gain_pending`, so
    #      `_floor_after_gain_map` returned at its first line.
    # Each check below goes red if its fault is put back.
    import types as _t19c                                         # noqa: E402
    import re as _re19c                                           # noqa: E402
    from lsa.calibrate.mapping import (Sample as _S19, sensitivity_map as _sm19)  # noqa: E402

    _poly19 = [(300.0, 150.0), (900.0, 150.0), (900.0, 600.0), (300.0, 600.0)]
    _pts19 = [_S19(screen=(0.0, 0.0), camera=c, peak=p) for c, p in (
        ((330.0, 200.0), 84.0), ((330.0, 540.0), 80.0), ((600.0, 380.0), 66.0),
        ((860.0, 200.0), 47.0), ((860.0, 540.0), 45.0), ((600.0, 180.0), 70.0))]

    class _GridLab(_FloorLab):
        _apply_sensitivity_map = _CL2.CameraLab._apply_sensitivity_map
        _save_sensitivity = _CL2.CameraLab._save_sensitivity
        _sensitivity_rows = staticmethod(_CL2.CameraLab._sensitivity_rows)

        def __init__(self, outline=_poly19, noise=11.0):
            super().__init__()
            self.events = []
            self.saved_rows = []
            self._noise_peak = noise
            self.settings = _t19c.SimpleNamespace(width=1280, height=720)
            self.mapping = _t19c.SimpleNamespace(
                samples=list(_pts19), sensitivity=[],
                screen_rect_in_camera=lambda: outline)

        def _set_gain_map(self, gain, why):
            self.events.append("apply")
            super()._set_gain_map(gain, why)

        def _report_signal_margin(self, peaks):
            pass

        def _save_mapping(self, mapping, what):
            # What is on the mapping AT THE MOMENT it is written -- an order
            # assertion, not an existence one (the first instance of this
            # fault passed an existence test throughout).
            self.events.append("save")
            self.saved_rows = [dict(r) for r in mapping.sensitivity]
            return True

    gl = _GridLab()
    gl._apply_sensitivity_map(gl.mapping)
    g_applied = gl.applied[0] if gl.applied else None
    check("grid: a map is built and applied", g_applied is not None,
          f"applied {len(gl.applied)} maps; log {gl.logged[-2:]}")
    unconf = _sm19(_pts19, (1280, 720))
    # (x, y) = (1100, 300): outside the polygon, on the weak (right) side.
    check("grid: PREMISE - the same peaks UNconfined would amplify the room "
          "there (this is what makes the next check able to fail)",
          unconf is not None and float(unconf[300, 1100]) < 0.77,
          f"unconfined gain {float(unconf[300, 1100]):.2f} -> "
          f"{1 / max(1e-3, float(unconf[300, 1100])):.2f}x"
          if unconf is not None else "no map")
    check("grid: a point OUTSIDE the screen is not amplified (1.00x, "
          "measured 1.44x unconfined at the window beside his monitor)",
          g_applied is not None and abs(float(g_applied[300, 1100]) - 1.0) < 0.02
          and abs(float(g_applied[690, 1200]) - 1.0) < 0.02,
          f"outside gain {float(g_applied[300, 1100]):.3f}"
          if g_applied is not None else "no map")
    check("grid: and inside the screen it still corrects the weak side",
          g_applied is not None and float(g_applied[200, 860]) < 0.8,
          f"inside weak-side gain {float(g_applied[200, 860]):.3f}"
          if g_applied is not None else "no map")

    # No outline: say so and stay uniform. NOT a silent whole-frame map.
    gn = _GridLab(outline=None)
    gn._apply_sensitivity_map(gn.mapping)
    check("grid: with no screen outline nothing is applied, saved or judged",
          gn.applied == [] and gn.events == [] and gn._gain_pending is None
          and gn.floors_asked == 0,
          f"applied {len(gn.applied)}, events {gn.events}")
    check("grid: and it says why, so uniform detection is not a mystery",
          any("no screen outline" in m for m in gn.logged), str(gn.logged[-1:]))

    # Persistence, as an ORDER.
    check("grid: the sensitivity reaches the file (saved rows = the nine-point medians)",
          len(gl.saved_rows) == len(_pts19)
          and sorted(r["peak"] for r in gl.saved_rows)
          == sorted(s.peak for s in _pts19),
          f"{len(gl.saved_rows)} rows held when the mapping was written")
    check("grid: ORDER - the rows were on the mapping when it was written, "
          "and the write came before the map was applied",
          gl.events[:2] == ["save", "apply"], str(gl.events))
    _src19 = Path(_CL2.__file__).read_text(encoding="utf-8")
    _tree19 = _ast2.parse(_src19)
    _fin19 = next(n_ for n_ in _ast2.walk(_tree19)
                  if isinstance(n_, _ast2.FunctionDef) and n_.name == "_grid_finish")
    _lines19 = {}
    for n_ in _ast2.walk(_fin19):
        if isinstance(n_, _ast2.Call) and isinstance(n_.func, _ast2.Attribute):
            _lines19.setdefault(n_.func.attr, []).append(n_.lineno)
    check("grid: the final calibration save comes AFTER the sensitivity is "
          "set, so it cannot overwrite the rows with an empty list",
          "_apply_sensitivity_map" in _lines19 and "_save_mapping" in _lines19
          and min(_lines19["_apply_sensitivity_map"]) < min(_lines19["_save_mapping"]),
          str(_lines19.get("_apply_sensitivity_map")) + " vs "
          + str(_lines19.get("_save_mapping")))

    # Judged: reached, and bites on the bar.
    check("grid: the map is handed to the judgement (floor remembered, "
          "map pending, floor re-measured)",
          gl._gain_pending is not None and gl._floor_before_gain == 11.0
          and gl._calibrating,
          f"pending {gl._gain_pending is not None}, "
          f"before {gl._floor_before_gain}, calibrating {gl._calibrating}")
    gl._floor_after_gain_map(70.0)               # bar 17 -> 91: 2026-10-14
    check("grid: a map that makes the bar dramatically worse IS withdrawn",
          gl.applied[-1] is None, f"applied {[a is None for a in gl.applied]}")
    gk = _GridLab(noise=1.0)
    gk._apply_sensitivity_map(gk.mapping)
    gk._floor_after_gain_map(3.0)                # bar 8 -> 9: kept
    check("grid: and one doing its job (floor 1 -> 3) is kept",
          len(gk.applied) == 1 and gk.applied[-1] is not None)

    # Fix the rule: no code path in the lab may build a gain map WITHOUT a
    # region. The bare function is still importable for tools and tests; the
    # lab goes through the confined one.
    check("no CameraLab path calls sensitivity_map without confining it",
          _re19c.search(r"(?<![A-Za-z_])sensitivity_map\(", _src19) is None,
          "a bare sensitivity_map( call is back in camera_lab.py")
    _rest19 = next(n_ for n_ in _ast2.walk(_tree19)
                   if isinstance(n_, _ast2.FunctionDef)
                   and n_.name == "_restore_sensitivity")
    check("the remembered-map restore confines too (it passed region=None "
          "when the mapping had no outline)",
          any(isinstance(n_, _ast2.Call)
              and getattr(n_.func, "id", "") == "confined_sensitivity_map"
              for n_ in _ast2.walk(_rest19)))

    print("\n20. advice must be something the shooter can actually do")
    # Said in these words on 2026-10-14, after an evening of closing blinds:
    # "The room already is as dark as possible. It will not get any darker
    # than this." The thin-margin alert was telling him to darken it anyway.
    #
    # Advice that cannot be followed is worse than silence: it trains the
    # reader to ignore the next warning, and this app's warnings are the only
    # thing standing between a bad session and an unexplained one. The app has
    # measured BOTH halves of the ratio by this point, so it can tell a noise
    # problem (the room) from a signal problem (exposure, dot size) and must.
    marg = lab_src3.find("def _report_signal_margin")
    marg_body = lab_src3[marg:lab_src3.find(chr(10) + "    def ", marg + 10)]
    check("the thin-margin advice branches on the measured noise floor",
          "noisy" in marg_body and "floor > 25" in marg_body,
          "one fixed message regardless of what was measured")
    # Split the two branches and check each says the right thing and only that.
    cut = marg_body.find("if noisy:")
    dark_branch = marg_body[marg_body.find("else:", cut):]
    loud_branch = marg_body[cut:marg_body.find("else:", cut)]
    check("with a low floor it does NOT blame the room",
          "will not help" in dark_branch and "drawing a blind" not in dark_branch,
          "still tells a shooter in a dark room to darken it")
    check("with a low floor it names exposure and dot size instead",
          "LONGER exposure" in dark_branch and "fewer pixels" in dark_branch)
    check("with a high floor it DOES blame the room",
          "blind" in loud_branch or "Dimming" in loud_branch,
          "a genuinely noisy room gets no room advice")

    # The signal meter's own "too faint" line (fifth "dim the room", seen in
    # verdict_screen.png on 2026-10-16 with the measured floor at 9). The
    # 10,080-verdict generator cannot see strings built in the UI.
    _at = lab_src3.find("seen but too faint to count")
    _faint = lab_src3[_at:lab_src3.find("kind = ", _at)]
    check("the meter's too-faint line never says dim the room",
          _at > 0 and "dim the room" not in _faint.lower()
          and "press X" in _faint, _faint.strip()[:120])

    # And the lever it points at has to exist.
    check("the exposure can be re-measured on demand",
          "def _remeasure_exposure" in lab_src3)
    check("bound to the key the advice names",
          "Qt.Key_X" in lab_src3 and "_remeasure_exposure()" in lab_src3)
    check("and it measures against the target background, not the desktop",
          "_cover_desktop_for_noise" in
          lab_src3[lab_src3.find("def _remeasure_exposure"):
                   lab_src3.find("def _toggle_aim_view")],
          "measuring against the desktop was the difference between a floor "
          "of 2 and a floor of 90")

    # An empty measurement table must never be printed as if it were a result.
    # `_apply_chosen_exposure` since the Director split the measuring from
    # the applying: the probe now measures the floor as its own next line
    # rather than through a timer the next routine had to outlast.
    chosen = lab_src3.find("def _apply_chosen_exposure")
    chosen_body = lab_src3[chosen:lab_src3.find(chr(10) + "    def ", chosen + 10)]
    check("an unmeasured exposure says so rather than printing an empty table",
          "if rows:" in chosen_body and "not swept this run" in chosen_body,
          "prints 'exposure measured -> ' with nothing after it")

    # ---------------------------------------------------------------- 21
    # The measured falloff is saved with the calibration so the learner
    # does not have to start from nothing every session. The ORDER is the
    # whole fix: the first version carried the rows onto the new mapping
    # AFTER saving it, so the session that calibrated logged 're-using the
    # sensitivity ... 5 regions' and worked, while calibration.json was
    # written with none. Measured 2026-10-15: 0 regions on disk against 5
    # in the backup taken moments before. A test that only asserted the
    # carry-over EXISTS would have passed throughout.
    print(chr(10) + '21. the carried sensitivity reaches the file, not just the session')
    lab_src4 = (Path(__file__).resolve().parents[1] / 'src' / 'lsa' / 'tools'
                / 'camera_lab.py').read_text(encoding='utf-8')
    fin = lab_src4.find('def _report_pattern_check')
    check('the calibration-finish path exists', fin > 0)
    # BY AST, on the call graph rather than on a sentence of source. The
    # previous version matched the literal `mapping.sensitivity = prev` and
    # went red when the carry-over was generalised to apply EVERY pose fact
    # through one function -- which is the correct refactor, and the third
    # time on this project that a source-text grep has failed a change that
    # improved the code. What is being asserted is an ORDER, so assert the
    # order of the two calls and nothing about how either is spelled.
    import ast as _ast
    _fn = next(n_ for n_ in _ast.walk(_ast.parse(lab_src4))
               if isinstance(n_, _ast.FunctionDef)
               and n_.name == '_report_pattern_check')
    carry = save = -1
    for _n in _ast.walk(_fn):
        if not isinstance(_n, _ast.Call) or not isinstance(_n.func, _ast.Attribute):
            continue
        if _n.func.attr == '_carry_pose_facts' and carry < 0:
            carry = _n.lineno
        # `_save_mapping` since 2026-10-16: the bare `mapping.save(...)`
        # returned False silently on a numpy scalar and the caller ignored
        # it, so the app logged "saved" with nothing on disk.
        if _n.func.attr in ('save', '_save_mapping') and save < 0:
            save = _n.lineno
    check('the falloff is carried onto a new mapping', carry > 0,
          'nothing in the finish path calls _carry_pose_facts')
    check('the mapping is saved', save > 0)
    check('and it is carried BEFORE the save, or the file loses it',
          0 < carry < save,
          'saving first writes a mapping with no sensitivity, so the next '
          'cold launch starts from nothing -- the deadlock this persistence '
          'was written to end')

    # ---------------------------------------------------------------- 22
    # The exposure has to follow the room, and the room follows the clock.
    # A value measured at 21:00 in a blacked-out room is the wrong value at
    # 10:00 with daylight on the window beside this monitor. But a sweep is
    # the one operation measured to hang this camera, so it cannot simply be
    # run on every launch. The gate in front of it has to be free.
    print(chr(10) + '22. the remembered exposure is checked against the room')
    cap_src = (Path(__file__).resolve().parents[1] / 'src' / 'lsa' / 'camera'
               / 'capture.py').read_text(encoding='utf-8')
    check('the room can be measured without touching the controls',
          'def start_room_measurement' in cap_src)
    rm = cap_src.find('def start_room_measurement')
    rm_body = cap_src[rm:cap_src.find(chr(10) + '    def ', rm + 10)]
    check('and it writes no control value',
          'controls.apply' not in rm_body and 'CAP_PROP' not in rm_body,
          'a gate that costs a control write is no cheaper than the sweep '
          'it is gating')

    # It must measure the SAME quantity the exposure rule judges on. The
    # detector's own score is background-subtracted and gain-mapped, so
    # comparing it against CHROMA_CLEAN would be a different quantity
    # wearing the same name -- the proxy mistake this project has made four
    # times, most expensively when exposure was chosen on mean brightness.
    loop = cap_src[cap_src.find('if self._room_until:'):]
    loop = loop[:loop.find('if self._noise_until:')]
    check('measured from the raw frame, the way the sweep measures it',
          'cv2.subtract' in loop and 'cv2.max' in loop
          and 'frame.mean()' in loop)
    check('not from the detector score, which is a different quantity',
          'last_score' not in loop,
          'the detector score is background-subtracted and gain-mapped, so '
          'it cannot be compared against CHROMA_CLEAN')

    gate = lab_src4[lab_src4.find('def _on_room_measured'):]
    gate = gate[:gate.find(chr(10) + '    def ', 10)]
    check('the gate exists', len(gate) > 0)
    check('it judges against the exposure rule own bars',
          'CHROMA_CLEAN' in gate and 'MEAN_CEILING' in gate,
          'a second set of thresholds would drift away from the rule it is '
          'meant to be anticipating')
    check('it notices a room that got BRIGHTER (daylight arriving)',
          'too_bright' in gate)
    check('and a room that got DARKER, which is wasted signal',
          "verdict == 'dark'" in gate,
          'the 2026-10-14 failure: the room was darkened as far as it would '
          'go and -7 was carried in regardless, four stops short')
    check('a shot during it cannot be read as a bright room',
          '_last_shot_at' in gate,
          'a laser hit inflates chroma p99 past any room, and the conclusion '
          'would be to stop the exposure DOWN in response to a hit -- the '
          'way a trigger pull spoiled the noise floor three times')
    check('it only decides whether to measure, never what the value is',
          'start_exposure_search' in gate
          and '_shooting_exposure = ' not in gate,
          'the sweep chooses the exposure by measurement; a gate that set '
          'the value itself could be silently wrong')
    check('mid-session it reports rather than sweeping under the shooter',
          'if why != ' + chr(39) + 'setup' + chr(39) + ':' in gate.replace(chr(34), chr(39))
          and '_signal_note' in gate,
          'nine control writes and a fullscreen cover in the middle of a '
          'round reads as the app breaking')
    check('the room is re-checked as the session runs, not only at setup',
          '_room_timer' in lab_src4
          and '_check_room(' + chr(34) + 'drift' + chr(34) + ')' in lab_src4)
    check('setup consults the room instead of trusting the saved value',
          '_check_room(' + chr(34) + 'setup' + chr(34) + ')' in lab_src4
          and '_on_exposure_chosen(saved, [])' not in lab_src4,
          'the remembered value was carried into every room for days')

    # The verdict itself, against the sweeps already in the logs. Source
    # greps prove the gate is wired; these prove it decides correctly, and
    # that its answer agrees with what the sweep then finds.
    from lsa.camera.capture import CaptureThread as _CT

    # The dark room of 2026-10-14 19:26, measured when the sweep was finally
    # allowed to run. The app had been carrying -7 into this room for days.
    DARK = [(-3, 13, 203, 9), (-4, 19, 255, 11), (-5, 15, 255, 9),
            (-6, 14, 255, 9), (-7, 9, 255, 7), (-8, 6, 248, 5),
            (-9, 2, 173, 2), (-10, 2, 119, 2), (-11, 1, 44, 2)]
    at7 = next(r for r in DARK if r[0] == -7)
    check('a dark room is seen as having headroom at the remembered -7',
          _CT.room_verdict(at7[1], at7[3]) == 'dark',
          'mean %d, chroma %d' % (at7[1], at7[3]))
    picked = _CT._choose_exposure(DARK, 25.0, 200)[0]
    check('and the sweep it provokes does find a longer one',
          picked > -7,
          'the gate promised headroom and the sweep picked %g' % picked)
    check('four stops longer, which is 16x the exposure time',
          picked == -3,
          'picked %g; the laser scales with exposure time, and this is the '
          'signal the right-hand third of the panel was missing' % picked)

    # The camera ignoring exposure writes after a replug: mean pinned near
    # 130 at every setting. Over the mean bar, so the room reads as bright.
    check('a washed-out frame is seen as too bright, not as headroom',
          _CT.room_verdict(130, 31) == 'bright')

    # Either bar alone is enough; both are real failure modes.
    check('a noisy background alone triggers a sweep',
          _CT.room_verdict(10, 25) == 'bright',
          'chroma over the bar with a dim frame')
    check('a lit panel alone triggers a sweep',
          _CT.room_verdict(45, 5) == 'bright',
          'the monitor stops being a ghost and the dot must out-shine it')

    # ...and a room that fits is left alone, because the alternative is
    # several control writes on a camera a single write has hung for 272s.
    check('a room inside both bars is left alone',
          _CT.room_verdict(36, 20) == 'ok',
          'sweeping here would spend the one operation measured to hang '
          'this camera, for nothing')
    check('the verdict is free of camera access, so it can be tested',
          isinstance(_CT.__dict__.get('room_verdict'), staticmethod),
          'the same reason _choose_exposure is static')

    # The fault this gate actually shipped with, caught on hardware at
    # 10:18 on 2026-10-15 and not by any of the checks above. `_calibrating`
    # is set by the screen search and was still true when the gate ran, so a
    # guard returned early -- and because the gate sits in the MIDDLE of the
    # chain that ends in a ready detector, nothing downstream ever ran. The
    # log stopped at 'applying preset: Laser hunt' and the app sat there
    # until it was killed. Exactly the shape of the cover leak: an early
    # return that skips what the caller was depending on.
    #
    # So the invariant is structural, and worth asserting as one: at setup,
    # every way out of the gate must hand the chain on.
    import ast as _ast
    _tree = _ast.parse(lab_src4)
    _fn = next((n for n in _ast.walk(_tree)
                if isinstance(n, _ast.FunctionDef) and n.name == '_check_room'),
               None)
    check('_check_room is a function we can inspect', _fn is not None)
    if _fn is not None:
        # The branch taken at setup: find the `if why == 'setup':` test.
        _setup = None
        for _n in _ast.walk(_fn):
            if isinstance(_n, _ast.If) and 'setup' in _ast.dump(_n.test):
                _setup = _n
                break
        check('the gate treats setup as its own case', _setup is not None)
        if _setup is not None:
            _bad = []
            for _stmt in _setup.body:
                # Each guard is `if <cond>: <hand off>; return`.
                if not isinstance(_stmt, _ast.If):
                    continue
                _hands_on = any(
                    isinstance(_c, _ast.Call)
                    and getattr(_c.func, 'attr', '') in ('_room_proceed',
                                                         '_on_exposure_chosen')
                    for _c in _ast.walk(_stmt))
                _returns = any(isinstance(_c, _ast.Return)
                               for _c in _ast.walk(_stmt))
                if _returns and not _hands_on:
                    _bad.append(_stmt.lineno)
            check('no setup guard returns without handing the chain on',
                  not _bad,
                  'lines %s return out of setup and leave the app half '
                  'configured -- the detector is never readied and nothing '
                  'says why' % _bad)

    # ...and a deadline, because this step waits on frames arriving and a
    # camera that has stopped delivering them is precisely when the app must
    # still finish and report. Already written down as one of the three
    # faults behind the window that trapped the user.
    check('the room measurement has a deadline of its own',
          '_room_guard' in lab_src4 and 'setSingleShot(True)' in lab_src4
          and 'def _room_gave_up' in lab_src4,
          'a step that waits on hardware with no deadline is how the app '
          'became an immovable black rectangle once already')
    _gu = lab_src4[lab_src4.find('def _room_gave_up'):]
    _gu = _gu[:_gu.find(chr(10) + '    def ', 10)]
    check('and the deadline continues setup rather than just logging',
          '_room_proceed' in _gu)
    _h = lab_src4[lab_src4.find('def _on_room_measured'):]
    _h = _h[:_h.find(chr(10) + '    def ', 10)]
    check('a measurement that arrives disarms the deadline first',
          _h.find('_room_guard.stop()') < _h.find('if frames <'),
          'a deadline left armed fires into a step that already finished -- '
          'the reason the noise cover is taken down before any branch')

    # ---------------------------------------------------------------- 23
    # THE SOUND GOES TO THE SYSTEM DEFAULT OUTPUT. This is the user's
    # instruction, given after a change of mine broke working audio:
    # 'You should always start with the default sound-playing device.'
    #
    # What happened: the gunshot was inaudible during a live string, and I
    # concluded the app should stop playing into display audio, because a
    # monitor's endpoint disappears when the screen sleeps -- which this app
    # causes every time it flashes for calibration. The reasoning is sound
    # and the conclusion was wrong: his speakers ARE in the monitor. The
    # rule moved the sound to a Realtek output with nothing attached, and
    # made a working channel silent. Measured by ear, twice: four gunshots
    # from the display, nothing at all from the 'real' speakers.
    #
    # Windows already knows which endpoint has something connected to it.
    # That is what the default IS. Choosing against it from a device's name
    # is reasoning about hardware from a string.
    print(chr(10) + '23. the gunshot plays to the system default output')
    mixer_src = (Path(__file__).resolve().parents[1] / 'src' / 'lsa' / 'audio'
                 / 'mixer.py').read_text(encoding='utf-8')
    pick = mixer_src[mixer_src.find('def _pick_output'):]
    pick = pick[:pick.find(chr(10) + '    def ', 10)]
    check('the picker exists', len(pick) > 0)
    check('with no remembered name it returns the default',
          'not self.device_name' in pick and 'return None' in pick,
          'anything else overrides a choice the user already made in Windows')
    check('it does not rank outputs by guessing from their names',
          'DISPLAY_HINTS' not in pick and 'def score' not in pick,
          'a scoring rule here silenced a working channel once; the user then said to always start with the default')

    # The real failure mode is still real -- it just is not a reason to
    # pick a different device. It is a reason to notice the stream died.
    check('a dead output stream is noticed and reopened',
          'def ensure_alive' in mixer_src,
          'PortAudio does not raise when an endpoint goes away under a callback stream: play() keeps succeeding and the sound goes nowhere')
    alive = mixer_src[mixer_src.find('def ensure_alive'):]
    alive = alive[:alive.find(chr(10) + '    def ', 10)]
    check('by checking the stream is still active',
          '.active' in alive and 'self.start()' in alive)
    check('and saying so, rather than recovering silently',
          'had stopped' in alive,
          'a feedback channel that fails quietly is read as the app not detecting the shot')
    lab_audio = lab_src4[lab_src4.find('def _check_audio_alive'):]
    lab_audio = lab_audio[:lab_audio.find(chr(10) + '    def ', 10)]
    check('the app actually checks it on a timer',
          '_audio_timer' in lab_src4 and 'ensure_alive' in lab_audio)

    # ...and the device is NAMED at startup. 'audio ready, output latency
    # ~21.3 ms' was the whole report while playing into a device nobody
    # could hear: true, useless, and indistinguishable from success.
    check('the launch log names the output device',
          'output_name' in lab_src4,
          'a device chosen on the app behalf has to be named, or a silent wrong choice looks exactly like success')

    # ---------------------------------------------------------------- 24
    # A one-frame pulse that is unmistakably a laser now counts as a shot.
    #
    # 'About one shot in three leaves no hole' turned out not to be a
    # detection failure at all: the detector SAW those shots and the
    # tracker threw them away for lasting a single frame. Replayed over a
    # recorded session of real shooting -- 1015s, every detection logged:
    #
    #     1 frame   26 pulses   <- discarded by min_frames=2
    #     2 frames  30 pulses
    #     3 frames  30 pulses
    #     4+        23 pulses        24% of trigger pulls, gone
    #
    # The diode is PWM-driven and the camera samples it stroboscopically,
    # so a ~100ms pulse lands in a variable number of frames and sometimes
    # in one. Those discarded singles were not marginal: peak 30..255,
    # median 81, area median 560px.
    # ---------------------------------------------------------------- 23b
    # THE MICROPHONE MUST BE ABLE TO COUNT HIS SHOTS AT HIS CADENCE.
    #
    # It is the only channel that counts trigger pulls INDEPENDENTLY of the
    # camera -- heard 15, seen 12 is a real detection rate, and the agreed
    # gun test rests entirely on it.
    #
    # The app mutes it while playing its own gunshot, because the speakers
    # are in the same room: without that it hears itself and reports twice
    # the shots. The mute was the WHOLE sample -- 1.49s for his 9mm sample
    # plus a 0.25s tail, about 1.75 seconds of deafness after every shot.
    # He shoots about one a second. Measured in his log, 2026-10-16:
    # seventeen unmistakably real shots (peaks 31-134, scattered across the
    # screen) were each reported "SAW a shot with no gunshot sound: likely
    # a phantom", because the microphone was muted when he pulled the
    # trigger. Nothing tested this, in either direction.
    print(chr(10) + '23b. the self-mute is shorter than the gap between his shots')
    from lsa.audio.mixer import Mixer as _Mix                     # noqa: E402
    import numpy as _npa                                         # noqa: E402

    _mx = _Mix.__new__(_Mix)
    _mx.samplerate = 1000
    # A gunshot's shape: a loud 200ms crack, then a long quiet tail.
    _loud = _npa.ones(200, dtype=_npa.float32)
    _tail = _npa.linspace(0.08, 0.0, 1300).astype(_npa.float32)
    _mx._samples = {"shot": _npa.concatenate([_loud, _tail])}
    check("duration() still reports the whole sample",
          abs(_mx.duration("shot") - 1.5) < 0.01,
          f"{_mx.duration('shot'):.2f}s")
    _loud_s = _mx.loud_duration("shot")
    check("loud_duration() reports only the loud part",
          abs(_loud_s - 0.2) < 0.02,
          f"{_loud_s:.2f}s of a 1.50s sample -- the rest is reverb the "
          f"microphone's own threshold would never trigger on")

    # HIS CADENCE, from his own log: shots about one second apart.
    HIS_GAP_S = 1.0
    _tail_s = _CL2.CameraLab.SOUND_TAIL_S
    check("the mute fits inside the gap between his shots",
          _loud_s + _tail_s < HIS_GAP_S,
          f"mute {_loud_s + _tail_s:.2f}s against shots {HIS_GAP_S:.1f}s "
          f"apart -- longer than the gap and the channel that counts his "
          f"trigger pulls cannot count them")
    # And the mirror: it must still cover the loud part, or the app hears
    # itself and reports twice the shots fired (measured: 37 false reports
    # in one string before the mute existed).
    check("but it still covers the whole loud part of the sample",
          _loud_s + _tail_s > 0.2,
          "too short and the app hears its own gunshot as a trigger pull")

    print(chr(10) + '24. a one-frame pulse bright enough to be unmistakable')
    from lsa.detect.laser import ShotTracker as _LD
    from lsa.camera.capture import GrouperConfig as _GC

    det_src = (Path(__file__).resolve().parents[1] / 'src' / 'lsa' / 'detect'
               / 'laser.py').read_text(encoding='utf-8')
    check('the tracker has a single-frame rule', 'def _confirmed' in det_src)
    check('and both acceptance paths go through it',
          det_src.count('self._confirmed(') == 2,
          'the provisional announcement and the final shot must agree, or a hole appears and then moves')
    check('the strict rule is still the default',
          'single_frame_peak: int = 0' in det_src,
          'zero disables it; a caller has to opt in with a measured bar')

    # The populations, verbatim from that recording. These are the numbers
    # the rule is set from, so the test owns them.
    NOISE_SINGLES = [10, 11, 12, 13, 13, 14, 15, 16, 17, 19, 20, 22, 24,
                     25, 26, 27, 28]
    LASER_SINGLES = [31, 44, 50, 61, 66, 74, 81, 84, 90, 93, 94, 107, 114,
                     129, 164, 236, 255]

    class _D:
        def __init__(self, peak, x=0.0, y=0.0, t=0.0):
            self.peak, self.x, self.y, self.timestamp = peak, x, y, t

    def kept(bar, peaks):
        d = _LD.__new__(_LD)
        d.min_frames = 2
        d.single_frame_peak = bar
        d._single_spots = []
        d.rejected_single_repeat = 0
        # Spread them out so the once-per-place guard never fires: this
        # check is about the brightness bar alone.
        return sum(1 for i, p in enumerate(peaks)
                   if d._confirmed([_D(p, i * 100.0, i * 100.0, i * 5.0)]))

    check('with the rule off, every single frame is still discarded',
          kept(0, LASER_SINGLES) == 0 and kept(0, NOISE_SINGLES) == 0,
          'the old behaviour has to survive, or this is not opt-in')
    check('at the measured bar every real single is recovered',
          kept(30, LASER_SINGLES) == len(LASER_SINGLES),
          '%d of %d' % (kept(30, LASER_SINGLES), len(LASER_SINGLES)))
    check('and no noise single gets in',
          kept(30, NOISE_SINGLES) == 0,
          '%d noise singles admitted' % kept(30, NOISE_SINGLES))
    check('a lower bar does let noise in, which is why it is not lower',
          kept(25, NOISE_SINGLES) > 0,
          'if 25 were also clean the margin claim would be untested')
    check('a multi-frame run is unaffected by the rule',
          kept(0, [5]) == 0
          and _LD._confirmed(type('x', (), {'min_frames': 2,
                                            'single_frame_peak': 0,
                                            '_single_spots': []})(),
                             [_D(5), _D(5)]),
          'two frames must still count regardless of brightness')

    # The bar is a multiple of min_peak, not a number tuned to one room.
    check('the app sets the bar to the measured laser/noise boundary',
          'SINGLE_FRAME_PEAK = 30' in lab_src4
          and 'single_frame_peak=max(self.SINGLE_FRAME_PEAK,' in lab_src4,
          'two recorded sessions at different exposures and noise floors '
          'both put the boundary at 30')
    check('and it rises if the room is noisy enough to reach it',
          'self.sp_minpeak.value() + 6' in lab_src4,
          'a fixed number with no floor would be unsafe in a bright room')
    check('the failed anchors are recorded, not just replaced',
          'contaminated' in det_src and '39 is above where' in det_src,
          'tying it to the noise floor, and learning it from confirmed '
          'shots, both failed on the data; the next person needs to know')
    check('and says what it set, with what it is worth',
          'a single-frame pulse counts as a shot at peak' in lab_src4)
    check('the config carries it to the tracker',
          'single_frame_peak' in (Path(__file__).resolve().parents[1] / 'src'
           / 'lsa' / 'camera' / 'capture.py').read_text(encoding='utf-8'))

    # A bright single frame is trusted ONCE PER PLACE. Brightness alone was
    # not enough and the first live run proved it inside a minute:
    #
    #   13:37:55  SHOT 52  cam=(561.12,522.69)  frames=1  peak=45
    #   13:37:58  SHOT 54  cam=(561.13,522.76)  frames=1  peak=48
    #
    # Seven hundredths of a pixel apart, three seconds apart -- a static
    # object, which these notes already name as the signature. Those two then
    # spoiled the noise floor five times over and drove it from 4 to 48,
    # which withdrew the sensitivity map and put the rejection bar above
    # every real shot.
    class _Det:
        def __init__(self, peak, x, y, t):
            self.peak, self.x, self.y, self.timestamp = peak, x, y, t

    def _fresh(bar=30):
        tr = _LD.__new__(_LD)
        tr.min_frames = 2
        tr.single_frame_peak = bar
        tr._single_spots = []
        tr.rejected_single_repeat = 0
        return tr

    _t = _fresh()
    _first = _t._confirmed([_Det(45, 561.12, 522.69, 100.0)])
    _again = _t._confirmed([_Det(48, 561.13, 522.76, 103.0)])
    check('the first bright single is allowed through',
          _first,
          'a static object and a real first shot are the same thing at '
          'that moment; refusing both would lose a real one')
    check('a repeat at the same place is refused',
          not _again and _t.rejected_single_repeat == 1,
          'this is the live phantom that drove the noise floor from 4 to 48')
    check('and it stays refused while it keeps firing',
          not _t._confirmed([_Det(47, 561.10, 522.70, 140.0)]),
          'one phantom per object, not an unbounded stream')

    # ...and it must not cost a real shot. These are the 17 bright singles
    # from the recording, at the camera positions they actually landed at.
    _REAL = [(99,686.5,440.4),(235,662.5,475.1),(255,533.7,502.3),
             (251,455.3,515.4),(118,625.5,474.1),(143,256.4,519.0),
             (108,521.8,318.3),(107,580.4,359.7),(107,731.3,218.3),
             (78,687.4,372.5),(177,269.3,499.6),(75,709.6,343.2),
             (93,438.2,454.2),(93,437.1,483.5),(81,638.8,286.1),
             (31,836.1,410.0),(75,644.8,356.0)]
    _t2 = _fresh()
    _kept = sum(1 for i, (pk, x, y) in enumerate(_REAL)
                if _t2._confirmed([_Det(pk, x, y, i * 5.0)]))
    check('every real single-frame shot still gets through',
          _kept == len(_REAL),
          '%d of %d kept; a laser never lands twice in the same few pixels'
          % (_kept, len(_REAL)))

    # ---------------------------------------------------------------- 25 ---
    # ONE DOOR INTO THE NOISE MEASUREMENT.
    #
    # Nine call sites started a noise-floor measurement directly, and on
    # 2026-10-15 three of them fired within four seconds WHILE the exposure
    # probe was drawing red dots on the screen. The floor measured the
    # dots, the probe measured a screen the floor measurement had covered,
    # and the self test ran against a bar from neither. Five cold launches
    # came out bimodal -- 12/12 twice, 0/12 twice -- on identical code, in
    # the same room, with calibration good to 1mm every time.
    #
    # These notes already carry this exact lesson from the day two pattern
    # calibrations interleaved. Writing it down did not stop it happening
    # again in a new place; a test that fails the build might.
    print()
    print("25. a noise measurement has exactly one guarded entry point")
    lab_src = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
               / "camera_lab.py").read_text(encoding="utf-8")
    # Two entry points are allowed and no others: `_start_noise_floor` for
    # the routines still driven by timers, and `_DirectorSink`'s method for
    # everything on the Director. Counted by which function each call sits
    # in rather than by line number, so moving code cannot quietly raise
    # the allowance.
    import ast as _ast
    _tree = _ast.parse(lab_src)
    ALLOWED = {"_start_noise_floor", "start_noise_calibration"}
    offenders = []
    for _fn in _ast.walk(_tree):
        if not isinstance(_fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        if _fn.name in ALLOWED:
            continue
        for _node in _ast.walk(_fn):
            if (isinstance(_node, _ast.Call)
                    and isinstance(_node.func, _ast.Attribute)
                    and _node.func.attr == "start_noise_calibration"):
                offenders.append(f"{_fn.name}:{_node.lineno}")
    check("every caller goes through one of the two guarded entry points",
          not offenders,
          f"{offenders} start a noise measurement directly")
    check("and the helper for the unmigrated routines exists",
          "_start_noise_floor" in lab_src)
    guard = lab_src.split("def _start_noise_floor")[1][:3000]
    check("it refuses while the app is drawing red on the screen",
          "self.director.is_drawing()" in guard or "_modetest_active" in guard,
          "measuring the room while the app lights it up measures the app")

    # The reverse direction too: the probe must not start into one. Again
    # behaviourally -- the textual version checked for the word
    # `singleShot`, which is a 900ms unbounded retry loop, i.e. it was
    # asserting the presence of a thing that is itself a rule violation.
    from lsa.director import Director as _D2, ManualClock as _MC2
    _busy2 = {"who": "a noise-floor measurement"}
    _d2 = _D2(_MC2(), sink=None, log=lambda _s: None,
              world_busy=lambda: _busy2["who"])

    class _Stub2:
        PROBE_STEPS = (0.0,)
        SELFTEST_TUNE_SHOTS = 1
        _shooting_exposure = -7.0
    check("the exposure probe refuses to start mid-measurement",
          _d2.start(_r1.probe_job(_Stub2())) is False)
    _busy2["who"] = ""
    _d2.poke()
    # It ran: it appears in the history. Asserting `busy_with() == "probe"`
    # instead would be wrong here and was -- the real routine reaches for
    # `_selftest_dot_radius` on its second line, which this stub lab does
    # not have, so it starts and fails immediately. What is being checked
    # is that the queue DRAINED, not that the stub can measure an exposure.
    check("and the one it queued then runs, rather than being dropped",
          any(n == "probe" for n, _o, _ms in _d2.history),
          f"a dropped measurement leaves the exposure at whatever it was "
          f"-- history {_d2.history}")
    _d2.cancel()

    # ---------------------------------------------------------------- 26 ---
    # A GUNSHOT MEANS A TRIGGER PULL. NOTHING ELSE MAY MAKE ONE.
    #
    # Asked twice. First: "there were way too many gunshot sounds during
    # the calibration phase" -- the bang was played before anything decided
    # whose light it was, so every calibration flash and every self-test dot
    # fired one. Then, after those phases were silenced: "why do I
    # constantly hear the gunshots?"
    #
    # Because the loudest source was never a phase. Before the noise floor
    # is measured the detector runs on a threshold saved from another room,
    # and one launch logged SHOT 1 through SHOT 14 inside a single second
    # at peak 8-18 and area 3-18px -- against a real shot on this hardware
    # at peak 85 and area 296px. The app even logs them "(uncalibrated)",
    # knowing there is nowhere to draw them, and bangs anyway.
    print()
    print("26. the gunshot is reserved for things that could be shots")
    lab_src3 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
                / "camera_lab.py").read_text(encoding="utf-8")
    # The whole method, not a fixed-size window: a window breaks the day anyone adds a
    # comment above the line it is looking for (2026-10-20).
    shot_body = lab_src3.split("def _on_shot(self")[1].split("\n    def ")[0]
    # The bang lives in `_play_gunshot`, reached only through `bang()`, which
    # does nothing unless `bang_due` -- and `bang_due` is built from
    # `_app_is_drawing`. So: the guard exists, precedes the first call, and
    # `_on_shot` itself never plays anything.
    play = shot_body.find("self._play_gunshot(")
    guard = shot_body.find("_app_is_drawing")
    check("the sound is decided AFTER whose light it is",
          0 < guard < play and "mixer.play" not in shot_body
          and "if bang_due and not _banged[0]" in shot_body,
          "the bang used to be played before any branch ran, so a "
          "calibration flash was indistinguishable from a trigger pull")
    ready = lab_src3.split("def _ready_for_real_shots")[1][:1400]
    check("and a detection needs a measured floor to count",
          "_floor_measured" in ready,
          "before one exists the threshold belongs to another room")
    check("and somewhere to be drawn",
          "self.mapping" in ready,
          "a detection with nowhere to draw it is a rumour, not a shot")
    check("but the grid still speaks, because he IS shooting during it",
          "_grid_active" in ready)
    check("the readiness test is consulted by the guard",
          "_ready_for_real_shots()" in shot_body)

    # ---------------------------------------------------------------- 27 ---
    # EVERY ROUTINE THAT OWNS THE SCREEN MUST CHECK WHETHER ANOTHER DOES.
    #
    # This is the single most expensive bug shape in the project. Four
    # instances so far, each found the hard way:
    #
    #   pattern calibration re-entering itself   -- cost two sessions, and
    #       read from the firing line as "the camera cannot see the far
    #       side of the panel", which it could
    #   the noise floor racing the exposure probe -- the live gate going
    #       12/12, 0/12, 12/12, 1/12, 12/12 on identical code
    #   the noise floor racing ITSELF            -- the same guard, which I
    #       had closed to everything except the thing it was guarding
    #   the arcade mode test and the aim view    -- found by listing them
    #
    # Patching instances has not worked, because the pattern keeps arriving
    # in a new routine. So the rule is enumerated here: anything that
    # drives the camera or paints the screen for a measurement must consult
    # at least one exclusion flag before it starts. A fifth routine added
    # next month fails this until it does.
    print()
    print("27. no routine may start while another owns the screen")
    lab_src4 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
                / "camera_lab.py").read_text(encoding="utf-8")
    FLAGS = ("_calibrating", "_pat_active", "_grid_active", "_sgrid_active",
             "_probe_active", "_selftest_active", "_modetest_active",
             "_floor_in_flight")
    # Only the routines STILL driven by timers and flags belong here. The
    # three that have moved onto the Director are checked in section 29, by
    # the property rather than by the presence of a flag name: they cannot
    # start while anything else holds the slot, including each other,
    # including a second copy of themselves. This list must shrink to
    # nothing as the rest move across, and must never grow.
    OWNERS = ("start_pattern_calibration", "start_grid_calibration",
              "start_sensitivity_grid",
              "_modetest_start",
              "_start_noise_floor", "_toggle_aim_view", "_remeasure_exposure",
              "_remeasure_noise")
    unguarded = []
    missing = []
    for name in OWNERS:
        marker_def = f"def {name}(self"
        if marker_def not in lab_src4:
            missing.append(name)
            continue
        body = lab_src4.split(marker_def)[1][:2600]
        if not any(flag in body for flag in FLAGS):
            unguarded.append(name)
    check("every known owner of the screen is in the source", not missing,
          f"{missing} -- renamed? then this list is stale and protecting "
          f"nothing")
    check("and every one of them consults an exclusion flag", not unguarded,
          f"{unguarded} can start while another routine is measuring")

    # ---------------------------------------------------------------- 28 ---
    # A TIMED OVERLAY MUST ACTUALLY GO AWAY.
    #
    # Three clocks were in play: time.time() for the verdict board,
    # time.monotonic() for the alert, time.perf_counter() inside the scene
    # -- and the layers compare their deadline against ctx.now, which is
    # the scene's. On Windows none of those share an epoch, so a deadline
    # set on one and tested against another is either already past or
    # unreachably far away.
    #
    # It was unreachable. The READY banner sat across the middle of the
    # target screen through a whole live string: "you keep that text on
    # screen and that obscures the action". Nothing caught it because every
    # existing test set the deadline and the clock itself, consistently,
    # and so could never disagree with the app about which clock to use.
    #
    # This drives the overlay the way the APP does: it is shown through the
    # app's own call, and then the SCENE is advanced, with no test-chosen
    # timestamps anywhere.
    print()
    print("28. a timed overlay disappears when its time is up")
    import time as _t
    from lsa.display.scene import Scene as _Scene
    from lsa.display.layers import Alert as _Alert, VerdictBoard as _VB
    from PySide6.QtCore import QRect as _QRect

    for name, layer, show in (
        ("alert", _Alert(),
         lambda la: la.show("x", "y", "info", seconds=0.25,
                            now=_t.perf_counter())),
        ("verdict board", _VB(),
         lambda la: la.show_report(
             type("R", (), {"verdict": "ready", "summary": "s",
                            "findings": []})(), now=_t.perf_counter())),
    ):
        sc = _Scene()
        sc.add(layer)
        show(layer)
        check(f"{name}: it is up to begin with", layer.visible)
        rect = _QRect(0, 0, 400, 300)
        # Advance the scene the way the app does, until the deadline has
        # certainly passed in whatever clock the scene is using.
        if hasattr(layer, "READY_SECONDS"):
            layer.until = _t.perf_counter() + 0.25
        deadline = _t.perf_counter() + 3.0
        while layer.visible and _t.perf_counter() < deadline:
            sc.advance(rect)
        check(f"{name}: and it clears itself within its own timeout",
              not layer.visible,
              "set on one clock, tested against another: it never goes")

    # ---------------------------------------------------------------- 29 ---
    # GUARDING AGAINST SOME OF THEM IS NOT GUARDING.
    #
    # Section 27 checks that each routine consults SOME exclusion flag.
    # That passed, and the app still collided, because each routine was
    # guarding against the ones its author happened to think of. The
    # exposure probe checked for calibration; pattern calibration checked
    # for itself and not for the probe. Live, 2026-10-15:
    #
    #   20:16:33  measuring the exposure by photographing the dot
    #   20:16:34  pattern calibration: 12 markers to fit
    #
    # Both drawing on and photographing the same screen; the self test
    # afterwards detected 0 of 12, and he asked "I did not see any holes on
    # screen - was that supposed to happen?"
    #
    # AND A CENTRAL LIST WAS STILL NOT ENOUGH.
    #
    # The list was added, this test was written to prove routines consulted
    # it, and the test passed -- while FIVE of the six routines that needed
    # it went on hand-rolling a different subset of the flags. Checked on
    # 2026-10-15, `_screen_busy` had exactly two callers: pattern
    # calibration, and the function that tells the furniture to stand down.
    #
    # The reason it could pass is in the old assertion, left above for the
    # record: `asks or names_other >= 3`. "Consults the list OR names at
    # least three flags by hand" is not the property. It is the property
    # with an escape hatch, and every routine took the hatch.
    #
    # So the test is now the property itself: a routine that owns the
    # screen is a Job on the Director, and the Director has one run slot.
    # There is no subset to get wrong, because there is no list to read.
    print()
    print("29. a routine that owns the screen runs on the Director")
    lab_src5 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
                / "camera_lab.py").read_text(encoding="utf-8")

    from lsa.director import Director as _Dir, Job as _Job, ManualClock as _MC
    from lsa import routines as _routines

    # Every migrated routine declares what it does, in one place, where it
    # is defined -- rather than in four disjunctions maintained by hand.
    class _Stub:
        """Enough of a lab to build the Jobs. They are not run here."""
        PROBE_STEPS = (0.0,)
        SELFTEST_TUNE_SHOTS = 1
        _shooting_exposure = -7.0

    for factory, name, draws_red, consumes in (
        (_routines.probe_job, "probe", True, True),
        (_routines.self_test_job, "selftest", True, True),
        (_routines.mapping_check_job, "mapcheck", False, True),
    ):
        job = factory(_Stub())
        check(f"  {name} is a Job on the Director", isinstance(job, _Job))
        check(f"  {name} declares whether it draws red", job.draws_red is draws_red)
        check(f"  {name} declares whether its light is a trigger pull",
              job.consumes_detections is consumes)
        check(f"  {name} tells the shooter what is happening", bool(job.label))
        check(f"  {name} stands the furniture down (B2)", job.measures_screen)

    # The three that collided live cannot now, and the proof is structural
    # rather than textual: start the first, and the Director refuses the
    # other two and says who is holding the screen.
    clock = _MC()
    lines: list[str] = []
    d = _Dir(clock, sink=None, log=lines.append)

    from lsa.director import Sleep as _Sleep

    def _holds():
        yield _Sleep(10000)          # takes the slot and keeps it

    started = d.start(_Job("probe", _holds))
    refused = [d.start(_Job(n, _holds))
               for n in ("selftest", "mapcheck", "probe")]
    check("the first routine takes the slot", started is True)
    check("and every other routine is refused, including a second of itself",
          refused == [False, False, False], f"{refused}")
    check("each refusal names who is holding the screen",
          all("probe has the screen" in ln for ln in lines if "not starting" in ln),
          f"{lines}")

    # The old flags survive only as READ-ONLY views of the Director, so
    # nothing can set one and have it disagree with what is running.
    import inspect as _inspect
    from lsa.tools.camera_lab import CameraLab as _CL
    for flag in ("_probe_active", "_selftest_active", "_check_active"):
        attr = getattr(_CL, flag, None)
        check(f"  {flag} cannot be assigned: it is a property of the Director",
              isinstance(attr, property),
              "a flag that can be set by hand is a flag that can be stale")
        if isinstance(attr, property):
            src = _inspect.getsource(attr.fget)
            check(f"  {flag} answers from the Director, not from itself",
                  "director" in src, src)

    # ---------------------------------------------------------------- 30 ---
    # THE APP'S OWN FLASH IS NOT A SHOT, AND SETUP MUST END ON EVERY PATH.
    #
    # Both measured on his hardware, 2026-10-16 08:47, in the first honest
    # live gate run this project has had.
    #
    # Flashing the monitor white to find the screen produced SIXTEEN
    # detections at peak 11-18 and area 3-16px -- a real shot here is peak
    # 85 and area 296px -- and `_on_shot` stamped every one of them as "a
    # shot just landed". The room measurement that followed was then
    # discarded as spoiled: the app distrusted its own measurement because
    # of light it had drawn itself.
    #
    # And the 62-second stall, which was mine from an hour earlier: an
    # optimisation that skips the startup noise floor skipped the only
    # place that cleared `_calibrating`, so `_screen_busy()` answered "a
    # measurement" forever and the two-marker check -- the routine that
    # exists to make startup fast -- sat queued for a minute.
    print()
    print("30. the app's own light is not a shot, and setup ends every path")
    lab_src6 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
                / "camera_lab.py").read_text(encoding="utf-8")

    shot = lab_src6.split("def _on_shot(self")[1][:1800]
    # Both predicates, in the one condition. Written as two separate
    # `in` tests rather than as one pinned line: the first version pinned
    # `if not self._app_is_drawing_now():` exactly, and went red the moment
    # the second predicate was added beside it -- a correct refactor failing
    # a check for a reason the check could not name. Seven greps of this
    # shape have already been replaced on this project.
    _lines = shot.splitlines()
    _at = [i for i, ln in enumerate(_lines) if "self._last_shot_at = " in ln]
    stamp = [_lines[i] for i in _at]
    # The guarding `if`, which is the line ABOVE the assignment. Taking the
    # last line before the assignment gives its indentation and nothing
    # else, which is how the first version of this check failed while the
    # code was right.
    cond = _lines[_at[0] - 1] if _at else ""
    check("the shot timestamp is NOT recorded unconditionally",
          bool(stamp) and "_app_is_drawing_now()" in cond,
          "stamping the app's own calibration flash as a shot is what "
          "discarded the room measurement sixteen times over")
    check("  and not while something covers the target screen either",
          "_camera_sees_our_screen()" in cond,
          "92 desktop detections in 20s would each discard the next room "
          "measurement as spoiled by a laser")
    drawing = lab_src6.split("def _app_is_drawing_now(self")[1][:1200]
    for trace in ("_flash is not None", "_calibrating", "director.is_drawing"):
        check(f"  and it counts {trace} as the app drawing", trace in drawing,
              "the screen search is not a Director routine, so the flash "
              "panel and _calibrating are the only traces of it")

    # Setup must end from every path that finishes it -- asserted by
    # COUNTING the callers, because the bug was that there was only one.
    import ast as _ast2
    _tree2 = _ast2.parse(lab_src6)
    enders = set()
    for _fn in _ast2.walk(_tree2):
        if not isinstance(_fn, (_ast2.FunctionDef, _ast2.AsyncFunctionDef)):
            continue
        for _n in _ast2.walk(_fn):
            if (isinstance(_n, _ast2.Call)
                    and isinstance(_n.func, _ast2.Attribute)
                    and _n.func.attr == "_setup_is_over"):
                enders.add(_fn.name)
    check("more than one path declares setup finished", len(enders) >= 2,
          f"only {sorted(enders)} -- when the sole caller was "
          f"`_on_noise_measured`, skipping the noise floor left "
          f"`_calibrating` set and the two-marker check queued for 62s")
    # To the end of the METHOD, not a character window. A 1600-char window
    # missed a call that was there, because this file's methods carry long
    # comments by design -- the third time a fixed window has made a check
    # fail for a reason it could not name.
    _over_start = lab_src6.find("def _setup_is_over(self")
    _over_end = lab_src6.find(chr(10) + "    def ", _over_start + 10)
    over = lab_src6[_over_start:_over_end if _over_end > 0 else len(lab_src6)]
    check("and ending setup wakes anything that queued behind it",
          "director.poke()" in over,
          "otherwise a queued routine waits for the next HUD tick at best, "
          "and for ever at worst")

    # And shutdown stands routines down by cancelling, not by naming flags.
    # The LAST closeEvent in the file is CameraLab's; the first belongs to
    # a dialog. Splitting on the first is how this check went red while
    # the code was right -- a check must be able to fail, and it must fail
    # for the reason it claims.
    close_body = lab_src5.split("def closeEvent(self, ev)")[-1][:4000]
    # `close()`, not `cancel()`. The difference matters: cancel REPORTS to
    # the chain with None, so at shutdown it would start the next routine --
    # cancelling the probe fires `then=start_self_test` on the slot the
    # cancel just freed, while the target window is being destroyed, and a
    # timer firing into a destroyed scene is a force-kill as far as the
    # camera is concerned (RULES A1, A3).
    check("shutdown CLOSES the Director rather than listing flag names",
          "director.close" in close_body,
          "the hand-written list omitted _check_active for a whole day; and "
          "cancel() alone would hand the chain on during teardown")

    # ---------------------------------------------------------------- 31 ---
    print("\n31. the target screen knows whether the camera can see it")
    # 2026-10-16 13:29, from the firing line: "As soon as I press Escape I
    # hear continuous shots that just do not stop. This is completely
    # unacceptable."
    #
    # Escape LOWERS this window instead of closing it, which is deliberate
    # and rule-protected -- closing it cost a whole session. But the camera
    # then watches a lit desktop while the detector runs on a bar measured
    # against a near-black screen, and 92 detections arrived in 20 seconds
    # at peak 9-29 where his real shots measure 85-146.
    #
    # The CONSUMERS are pinned in test_ui_smoke, behaviourally. What is
    # pinned here is the predicate's own branches -- including the one the
    # offscreen platform cannot reach, which is why the window asks a
    # single overridable method who is on top rather than calling Qt
    # inline. A branch unreachable in every test is the C1 failure.
    from lsa.display.window import TargetWindow as _TW
    from lsa.display.scene import Scene as _Scene
    from lsa.display.layers import SolidBackground as _Bg
    import lsa.display.window as _winmod

    win = _TW(_Scene([_Bg()]), target_fps=10)
    try:
        check("a fresh window declares cover_reason, not getattr-with-default",
              isinstance(getattr(win, "cover_reason", None), str),
              "a guard that reads a name nobody checked returned 0.0 for "
              "days while looking exactly like protection (RULES C8)")

        # The harness runs on the offscreen platform, which genuinely cannot
        # answer who is on top. It must FAIL OPEN: guessing 'covered' there
        # would suppress every simulated shot -- this guard inverted,
        # silently, which is the worse of the two errors and the one this
        # project keeps making. Shown first, because a hidden window is
        # legitimately not in front and that is a different branch.
        win.setGeometry(0, 0, 800, 600)
        win.show()
        check("on a platform that cannot answer, it assumes it IS in front",
              win.in_front_of_camera() is True,
              "suppressing on an unknown would silently kill detection")

        # Now force the real branches by making the platform answerable.
        class _Plat:
            @staticmethod
            def platformName():
                return "windows"

        real_plat, _winmod.QGuiApplication = _winmod.QGuiApplication, _Plat
        try:
            win.setGeometry(0, 0, 800, 600)
            win.show()

            # THE OPERATING SYSTEM IS NOT ASKED. `_owner_at` queries Windows,
            # live, for whichever real window is at that point; unstubbed, the
            # suite's output depended on the desktop (2026-10-22: Claude Code's
            # terminal title, led by an animating spinner glyph, crashed the
            # printer on cp1252 on some runs and not others). Every check in
            # this block reads a value the test controls.
            win._owner_at = lambda pt: None

            win._top_level_at = lambda pt: win
            check("  when every sampled point belongs to it, it is in front",
                  win.in_front_of_camera() is True)

            # Escape lowers it and our own operator console comes forward.
            other = object()
            win._top_level_at = lambda pt: other
            check("  when OUR OTHER window is on top, it is not",
                  win.in_front_of_camera() is False,
                  "this is the Escape case: the console is a top-level "
                  "widget of the same application")
            check("  and it says what it measured, not just a verdict",
                  win.cover_reason == "something is in front of the target screen: "
                  "only 0 of 5 sampled points still belong to it",
                  f"got {win.cover_reason!r} -- a check that reports only a "
                  f"verdict can be re-run and not debugged (RULES C3)")

            # Another APPLICATION's window: Qt answers None.
            win._top_level_at = lambda pt: None
            check("  when another application is on top, it is not",
                  win.in_front_of_camera() is False)

            # WHO is in front is named when Windows can say (2026-10-20: a dictation app's
            # topmost panel; the old message withheld the one fact the app held).
            from lsa.display.winowner import WindowOwner as _WO
            win._top_level_at = lambda pt: None
            win._owner_at = lambda pt: _WO(1234, "Status", "Wispr Flow", 77, False)
            win.in_front_of_camera()
            check("  a covering window that Windows can name is NAMED: process and title",
                  win.cover_by == "Wispr Flow's window 'Status'"
                  and win.cover_reason.startswith("Wispr Flow's window 'Status' is in front of the target screen")
                  and "only 0 of 5 sampled points" in win.cover_reason, win.cover_reason)
            win._owner_at = lambda pt: None
            win.in_front_of_camera()
            check("  when the window cannot be read it falls back to today's wording and invents no name",
                  win.cover_by == "" and win.cover_reason.startswith("something is in front of the target screen"),
                  win.cover_reason)
            win._owner_at = lambda pt: _WO(5, "", "Wispr Flow", 77, False)
            win.in_front_of_camera()
            check("  a window with no title is still named by its program, never as an empty quote",
                  win.cover_by == "a window of Wispr Flow", win.cover_by)
            win._owner_at = lambda pt: _WO(6, "Camera Lab", "python", 1, True)
            win.in_front_of_camera()
            check("  the app's OWN other window is said to be the app's own",
                  win.cover_by == "the app's own window 'Camera Lab'", win.cover_by)
            win._owner_at = lambda pt: _WO(1234, "Status", "Wispr Flow", 77, False)
            win._top_level_at = lambda pt: win
            win.in_front_of_camera()
            check("  and the name is cleared the moment the screen is in front again",
                  win.cover_by == "" and win.cover_reason == "")
            win._top_level_at = lambda pt: None

            # Partial cover must not be enough to suppress: an alert or the
            # camera pane covers a corner, and he is still shooting.
            seen = {"n": 0}

            def _mostly_ours(pt):
                seen["n"] += 1
                return None if seen["n"] == 1 else win

            win._top_level_at = _mostly_ours
            check("  one covered sample point out of five is still in front",
                  win.in_front_of_camera() is True,
                  "an alert over a corner must not stop him being counted")

            win._top_level_at = lambda pt: win
            win.hide()
            check("  and a hidden window is never in front",
                  win.in_front_of_camera() is False)
            check("  saying so, because he cannot read the screen either",
                  "hidden or minimised" in win.cover_reason,
                  f"got {win.cover_reason!r}")
        finally:
            _winmod.QGuiApplication = real_plat
    finally:
        win.close()
        win.deleteLater()

    print()
    print("32. a shot HEARD and not seen says what the detector almost saw")
    # Plan 02 step 9. 2026-10-16 20:57: four 'HEARD a shot with no detection' in
    # one minute, and the log could not say whether each was too dim, too short,
    # or not on the panel -- three events with three different cures.
    import time as _t31
    import types as _types31
    from lsa.audio.listener import Click as _Click31
    from lsa.camera.capture import (CaptureConfig as _CC31, CaptureThread as _CT31,
                                    GrouperConfig as _GC31)
    from lsa.camera.evidence import FrameEvidence as _FE31
    from lsa.shotlog import (classify_missed_pulse as _cmp31,
                             describe_missed_pulse as _dmp31,
                             missed_pulse_note as _mpn31)
    from lsa.tools.camera_lab import CameraLab as _CL31

    import inspect as _insp31
    import re as _re31
    from lsa.modes.base import ModeHost as _MH31

    # THE MODE-HOST STAND-IN, ONE PLACE. Two fixtures hand-built this
    # SimpleNamespace and both lacked `note_live_unheard` the day piece 3 made
    # `_poll_audio` call it (2026-10-22), so test_feedback died with an
    # AttributeError that read as an app defect. Every method the real
    # `_poll_audio` calls on the host is RECORDED here, not swallowed (C1).
    def _mode_host_stub(**over):
        rec = _types31.SimpleNamespace(
            mic_ready=None, heard_unseen=[], outcomes=[], live_unheard=[],
            dry_unheard=[])
        host = _types31.SimpleNamespace(
            mic_ready=None,
            note_heard_unseen=lambda n=1: rec.heard_unseen.append(n),
            note_pull_outcomes=lambda o_: rec.outcomes.extend(o_),
            note_live_unheard=lambda t_, deaf_: rec.live_unheard.append((t_, deaf_)),
            note_dry_unheard=lambda t_, deaf_: rec.dry_unheard.append((t_, deaf_)),
            is_dry_shot=lambda t_: False)
        for k_, v_ in over.items():
            setattr(host, k_, v_)
        host.rec = rec
        return host

    # -- the buffer on its own ----------------------------------------------
    fe = _FE31()
    for i in range(60):                    # one second at 60 fps
        fe.push(10.0 + i / 60.0, 11, 9 if i == 30 else 0, (321.0, 198.0) if i == 30 else None,
                0, 5 + (1 if i >= 33 else 0))
    sm = fe.summarise(10.3, 10.8)
    check("evidence: the brightest sub-threshold blob in the window, with where",
          sm is not None and sm["dim_peak"] == 9 and sm["dim_pos"] == (321.0, 198.0), str(sm))
    check("evidence: a too-short rejection inside the window is counted once",
          sm is not None and sm["short_delta"] == 1, str(sm))
    check("evidence: a window with no frames says so (None), it does not invent zeros",
          fe.summarise(50.0, 51.0) is None)
    fe.push(100.0, 0, 0, None, 0, 6)       # 90 s later: the old rows age out
    check("evidence: rows older than the buffer are dropped",
          fe.summarise(10.0, 11.0) is None)

    # -- the words ------------------------------------------------------------
    words = _dmp31(sm, 13.0, 8.0)
    check("words: a faint blob names its brightness, its position and the bar",
          "sub-threshold blob 9 at cam (321,198) against a bar of 13" in words, words)
    check("words: a faint blob AND a too-short rejection in one window leads with the "
          "too-short branch (it cleared the bar), and still prints the faint blob",
          words.startswith("the pulse was seen but lasted too few frames to count"), words)
    words_faint = _dmp31({"frames": 30, "frame_peak": 11, "dim_peak": 9,
                          "dim_pos": (321.0, 198.0), "off_peak": 5, "short_delta": 0},
                         13.0, 8.0)
    check("words: the lead sentence for a faint on-panel dot says it was ON THE PANEL "
          "and too faint -- it does not claim a miss",
          words_faint.startswith("the dot was on the panel but too faint to count")
          and "missed" not in words_faint.split("Evidence:")[0], words_faint)
    check("words: and the too-short rejection", "too-short rejections +1" in words, words)
    nothing = _dmp31({"frames": 30, "frame_peak": 4, "dim_peak": 0, "dim_pos": None,
                      "off_peak": 0, "short_delta": 0}, 13.0, 8.0)
    check("words: nothing at any level says exactly that (the bezel / off-panel case)",
          "the camera saw nothing at any level" in nothing, nothing)
    off = _dmp31({"frames": 30, "frame_peak": 60, "dim_peak": 0, "dim_pos": None,
                  "off_peak": 44, "short_delta": 0}, 13.0, 8.0)
    check("words: a blob that was off the screen is NOT 'nothing at any level'",
          "off-screen blob 44" in off and "saw nothing" not in off, off)

    # -- the off-screen reading is judged against the BLOB THRESHOLD, not zero ---
    # Measured on this hardware's own model: a blown-out daylight window outside
    # the mask reads 4-6 after the background model absorbs it; a dim red dot on
    # the bezel reads 89. Every earlier fixture set the one value hardware never
    # produces (0), which made the 'saw nothing' branch unreachable on the bench.
    def _ev(dim=0, off=0, short=0, fp=None):
        # frame_peak is the brightest score ANYWHERE, so it is never below the
        # dim or off-screen readings (a fixture that said otherwise was not a
        # frame the detector could have produced).
        fp = max(7, dim, off) if fp is None else fp
        return {"frames": 30, "frame_peak": fp, "dim_peak": dim,
                "dim_pos": (321.0, 198.0) if dim else None,
                "off_peak": off, "short_delta": short}

    for floor in (4, 5, 6, 8):          # 8 == the threshold itself: still not a blob
        got_b = _cmp31(_ev(off=floor), 8.0)
        check(f"branch: a realistic off-screen floor of {floor} (threshold 8) is "
              f"'nothing', not an off-panel blob", got_b == "nothing", got_b)
    words_floor = _dmp31(_ev(off=5), 13.0, 8.0)
    check("words: 'saw nothing' survives a realistic floor, and prints the reading "
          "WITH its reference so 5 is never read as a blob",
          "the camera saw nothing at any level" in words_floor
          and "off-screen reading 5 (at or below the blob threshold 8" in words_floor
          and "off-screen blob" not in words_floor, words_floor)
    check("branch: a dim red dot on the bezel (89) is the off-panel branch",
          _cmp31(_ev(off=89), 8.0) == "off_panel")
    check("branch: ... and its words say the dot was seen OUTSIDE the screen, with the "
          "threshold beside the reading",
          _dmp31(_ev(off=89), 13.0, 8.0).startswith("the dot was seen OUTSIDE the screen")
          and "off-screen blob 89 (above the blob threshold 8)" in _dmp31(_ev(off=89), 13.0, 8.0))
    check("branch: a faint on-panel blob with a noise-level off-screen reading is 'faint'",
          _cmp31(_ev(dim=9, off=5), 8.0) == "faint")
    check("branch: a too-short rejection with nothing brighter is 'too_short'",
          _cmp31(_ev(short=1, off=5), 8.0) == "too_short")
    check("branch: an off-panel dot brighter than a faint on-panel blob wins",
          _cmp31(_ev(dim=9, off=89), 8.0) == "off_panel")
    check("branch: no frames is its own branch", _cmp31(None, 8.0) == "no_frames")

    # -- the three sentences side by side, so they can be READ and judged -------
    s_faint, s_off, s_none = (_ev(dim=9, off=5), _ev(off=89), _ev(off=5))
    for label, s_ in (("faint on-panel", s_faint), ("off-panel dot", s_off),
                      ("silent camera", s_none)):
        print(f"         [{label}]")
        print(f"           log : {_dmp31(s_, 13.0, 8.0)}")
        print(f"           note: {_mpn31(s_, 13.0, 8.0)}")
    notes = {_mpn31(s_, 13.0, 8.0) for s_ in (s_faint, s_off, s_none)}
    check("the on-screen note differs for the three branches (it is not one fixed string)",
          len(notes) == 3, str(notes))
    check("the on-screen notes are short enough to read at a glance (<= 56 chars)",
          all(len(n_) <= 56 for n_ in notes), str([len(n_) for n_ in notes]))
    check("no note or lead gives room advice (RULES C4: dim/blind/darken)",
          not any(w_ in (n_ + _dmp31(s_, 13.0, 8.0)).lower()
                  for n_ in notes for s_ in (s_faint, s_off, s_none)
                  for w_ in ("dim ", "blind", "darken")))
    check("the faint note says it WAS seen on the panel; the nothing note does not",
          "faint" in _mpn31(s_faint, 13.0, 8.0) and "faint" not in _mpn31(s_none, 13.0, 8.0))
    check("words: no frames at all is its own message",
          "no camera frames were recorded" in _dmp31(None, 13.0, 8.0))

    # -- the corroborator hands over WHEN ---------------------------------------
    co = ShotCorroborator()
    co.add_clicks([Click(t=20.0, level=0.3, floor=0.01)])
    got = co.resolve(20.5)
    check("a heard-not-seen finding arrives with its click's time",
          got == ["heard-not-seen"] and co.last_found_times == [20.0],
          f"{got} {co.last_found_times}")
    check("and a quiet resolve clears it", co.resolve(21.0) == [] and co.last_found_times == [])

    # -- the real path: _poll_audio -> capture.evidence_between ------------------
    class _Spin:
        def __init__(self, v): self.v = v
        def value(self): return self.v

    class _FakeListener:
        running = True
        calibrated = True
        def __init__(self, clicks): self._c = clicks
        def take_clicks(self):
            out, self._c = self._c, []
            return out

    def _poll(click_t, fill):
        """Run the REAL `_poll_audio` against a REAL (never-opened) CaptureThread."""
        cap = _CT31(_CC31(), DetectorParams(), _GC31())
        _poll.said = []
        # RECORDING both halves of the contract the real ModeHost offers
        # `_poll_audio`: whether the microphone is calibrated, and a heard pull
        # the camera never saw. A count, NEVER a shot -- audio has no position,
        # so a hit invented from it would have to be drawn somewhere, and
        # somewhere is the one thing it cannot know.
        _poll.heard_unseen = []
        _poll.outcomes = []
        # RECORDING: the dry-pull tally is fed the corroborator's pairing
        # decisions on every poll (2026-10-20); a stub that swallowed them
        # could never show that the feed happens.
        _poll.host = _mode_host_stub(
            note_heard_unseen=lambda n=1: _poll.heard_unseen.append(1),
            note_pull_outcomes=lambda o_: _poll.outcomes.extend(o_))
        fill(cap._evidence, click_t)
        lab = _types31.SimpleNamespace(
            listener=_FakeListener([_Click31(t=click_t, level=0.3, floor=0.01)]),
            corroborator=ShotCorroborator(), capture=cap,
            _selftest_active=False, _modetest_active=False, _probe_active=False,
            _audio_note="", _audio_note_until=0.0,
            # The announcer: a heard pull holds speech, so `_poll_audio` feeds
            # every calibrated click to it. RECORDING, not a no-op: a stub that
            # silently swallows calls is a check that cannot fail (RULES C1),
            # and this attribute going missing is what made the mutation
            # scripts count a CRASH as a caught fault.
            voice=_types31.SimpleNamespace(note_shot=_poll.said.append),
            sp_minpeak=_Spin(13), sp_threshold=_Spin(8),
            # THE SCENE'S LAYERS. The real lab builds these in `__init__` and
            # `_poll_audio` runs from the HUD tick, so `self.layers` ALWAYS
            # exists by the time it is read -- which is why the production read
            # is unguarded, and correct.
            #
            # A stand-in WITHOUT it is not a thinner lab, it is a lab that
            # cannot exercise the code it stands in for (RULES C12). On
            # 2026-10-19 this SimpleNamespace crashed the real `_poll_audio`
            # with AttributeError the moment the two-player verdict started
            # reading the mode host, took the whole of test_feedback down with
            # it, and made THIRTEEN mutate_detector anchors report BROKEN --
            # thirteen faults that looked like defects and were one missing
            # attribute on a test double.
            #
            # RECORDING, not a no-op, for the same reason the voice stub above
            # records: a stub that silently swallows the write makes the branch
            # untestable, and this is the branch that decides whether the
            # verdict prints a microphone count or says "not calibrated".
            layers={"modes": _poll.host})
        lines: list[str] = []
        lab._log = lines.append
        # The mark watchdog is fed every poll's clicks (side B). RECORDED, not
        # swallowed: this fixture tests the corroborator, and a stub that
        # silently does nothing is a check that cannot fail (RULES C1).
        _poll.watch_fed = []
        lab._watch_feed = lambda clicks, now: _poll.watch_fed.append(len(clicks))
        lab._missed_pulse_evidence = _types31.MethodType(_CL31._missed_pulse_evidence, lab)
        _CL31._poll_audio(lab)
        _poll.note = lab._audio_note         # what the TARGET SCREEN would show
        return lines

    t_click = _t31.perf_counter() - 0.5

    def _faint(ev, tc):
        for i in range(-20, 25):           # +-0.33 s around the click at 60 fps
            ev.push(tc + i / 60.0, 11, 9 if i == 3 else 0,
                    (321.0, 198.0) if i == 3 else None, 0, 5 + (1 if i >= 6 else 0))

    got_lines = _poll(t_click, _faint)
    heard = next((ln for ln in got_lines if ln.startswith("HEARD")), "")
    print(f"         {heard[:420]}")
    check("the HEARD line is logged", bool(heard), str(got_lines))
    # ACROSS THE SEAM (RULES C14). The two-player verdict must say "not
    # calibrated" rather than print a zero that reads as "nothing was missed",
    # and the only thing that can tell it is the real `_poll_audio`. Asserted
    # on what the HOST received, not on what the listener holds: asking the
    # producer what it stored is the shape that let the EXIT tile ship broken
    # with six green checks behind it.
    check("the versus host is told the microphone IS calibrated, through the "
          "real _poll_audio -- never left as None, which the verdict would "
          "have to guess at",
          _poll.host.mic_ready is True, repr(_poll.host.mic_ready))
    # STUB DRIFT MADE LOUD. Every attribute the real `_poll_audio` touches on the
    # mode host must exist on BOTH the stand-in and the real ModeHost. A new call
    # in the app now fails here by NAME, before any fixture raises mid-run.
    _used31 = sorted(set(_re31.findall(r"_versus_host\.(\w+)", _insp31.getsource(_CL31._poll_audio))))
    _stub31 = _mode_host_stub()
    _mh_src31 = _insp31.getsource(_MH31)
    check("every host attribute _poll_audio uses exists on the test stand-in",
          bool(_used31) and all(hasattr(_stub31, n_) for n_ in _used31),
          f"used {_used31}; missing {[n_ for n_ in _used31 if not hasattr(_stub31, n_)]}")
    check("and on the REAL ModeHost, so the stand-in cannot invent a method",
          all(hasattr(_MH31, n_) or f"self.{n_}" in _mh_src31 for n_ in _used31),
          f"not on ModeHost: {[n_ for n_ in _used31 if not (hasattr(_MH31, n_) or f'self.{n_}' in _mh_src31)]}")
    # ACROSS THE SEAM again: the dry-pull tally must be handed the corroborator's
    # pairing decisions by the real `_poll_audio` (a heard click nobody saw is a
    # "heard-not-seen" outcome stamped with the click's own time).
    check("the dry-pull tally is fed the corroborator's decisions by the real _poll_audio "
          "(a heard click with no detection arrives as ('heard-not-seen', its time))",
          len(_poll.outcomes) == 1 and _poll.outcomes[0][0] == "heard-not-seen"
          and abs(_poll.outcomes[0][1] - t_click) < 1e-6, repr(_poll.outcomes))
    check("it carries the faint blob: 9 at cam (321,198) against a bar of 13",
          "sub-threshold blob 9 at cam (321,198) against a bar of 13" in heard, heard[-300:])
    check("it carries the too-short rejection count", "too-short rejections +1" in heard,
          heard[-200:])
    check("it carries the click's time (the clock the SHOT lines use)",
          f"t={t_click:.3f}" in heard, heard[-200:])
    check("it carries the running heard-against-seen tally", "[audio so far:" in heard,
          heard[-160:])
    check("and with a rejection in the same window, the TOO-SHORT lead -- never the old "
          "fixed 'it missed the screen', which its own evidence contradicts",
          "the pulse was seen but lasted too few frames to count" in heard
          and "it missed the screen" not in heard, heard[-300:])
    check("... and the note on the target screen is the too-short branch's too",
          "too short" in _poll.note, _poll.note)

    def _faint_only(ev, tc):               # the dot on the panel, merely faint
        for i in range(-20, 25):
            ev.push(tc + i / 60.0, 11, 9 if i == 3 else 0,
                    (321.0, 198.0) if i == 3 else None, 5, 5)

    heard_f = next((ln for ln in _poll(t_click, _faint_only) if ln.startswith("HEARD")), "")
    check("through the REAL path: a faint on-panel dot (9 against a bar of 13, off-screen "
          "floor 5) leads with 'on the panel but too faint' -- not 'missed the screen'",
          "the dot was on the panel but too faint to count" in heard_f
          and "it missed the screen" not in heard_f
          and "sub-threshold blob 9 at cam (321,198) against a bar of 13" in heard_f,
          heard_f[-340:])
    check("... and the target-screen note says so, with the bar",
          _poll.note.startswith("HEARD a shot: on the panel, but too faint")
          and "9 of 13" in _poll.note, _poll.note)

    def _blank(ev, tc):
        for i in range(-20, 25):
            ev.push(tc + i / 60.0, 3, 0, None, 0, 5)

    heard2 = next((ln for ln in _poll(t_click, _blank) if ln.startswith("HEARD")), "")
    check("with nothing in the frames it says the camera saw nothing at any level",
          "the camera saw nothing at any level" in heard2, heard2[-250:])
    check("... and the target-screen note says the camera saw nothing",
          "saw nothing" in _poll.note, _poll.note)

    def _bezel_dot(ev, tc):
        for i in range(-20, 25):
            ev.push(tc + i / 60.0, 91, 0, None, 89 if i == 2 else 5, 5)

    heard_off = next((ln for ln in _poll(t_click, _bezel_dot) if ln.startswith("HEARD")), "")
    check("through the REAL path: a dim dot on the bezel (89) is reported OUTSIDE the "
          "screen, in the log and on the target screen",
          "seen OUTSIDE the screen area" in heard_off
          and "saw nothing" not in heard_off and "OUTSIDE the screen" in _poll.note,
          heard_off[-300:] + " | " + _poll.note)

    def _floor_only(ev, tc):               # a blown-out window: off-screen reads 4-6
        for i in range(-20, 25):
            ev.push(tc + i / 60.0, 7, 0, None, 4 + (i % 3), 5)

    heard_fl = next((ln for ln in _poll(t_click, _floor_only) if ln.startswith("HEARD")), "")
    check("through the REAL path: an off-screen noise floor of 4-6 still reaches "
          "'saw nothing at any level' (it was unreachable while tested against zero)",
          "the camera saw nothing at any level" in heard_fl
          and "off-screen blob" not in heard_fl and "saw nothing" in _poll.note,
          heard_fl[-300:])
    heard3 = next((ln for ln in _poll(t_click, lambda ev, tc: None) if ln.startswith("HEARD")), "")
    check("with no frames recorded it says THAT, rather than 'saw nothing'",
          "no camera frames were recorded" in heard3 and "saw nothing" not in heard3,
          heard3[-250:])

    # -- and the REAL capture loop fills the buffer (RULES C8, C12) -----------
    # The buffer is pushed from `CaptureThread.run`, one tuple per camera frame.
    # Drive that real loop with a stand-in for cv2.VideoCapture only: a faint
    # dot that clears the blob threshold but not min_peak must come back out of
    # `evidence_between` WITH its position. A push that is never made, or made
    # with the wrong attribute, shows up here and nowhere else.
    class _OneShotCam:
        def __init__(self, owner, frames):
            self.owner, self.frames, self.i = owner, frames, 0

        def read(self):
            f = self.frames[min(self.i, len(self.frames) - 1)]
            self.i += 1
            if self.i >= len(self.frames):
                self.owner._running = False          # this is the last frame
            return True, f

        def release(self):
            pass

    dim_frames = ([frame_with_dot(0)] * 12 + [frame_with_dot(70)] * 3
                  + [frame_with_dot(0)] * 6)
    real = _CT31(_CC31(), DetectorParams(), _GC31())
    fake_cam = _OneShotCam(real, dim_frames)
    real._open_camera = lambda: fake_cam
    real._negotiated = lambda cap: {"fps": 60.0, "width": 640, "height": 360,
                                    "fourcc": "MJPG", "backend": "STAND-IN",
                                    "requested": {}, "controls": {}}
    t_before = _t31.perf_counter()
    real.run()
    t_after = _t31.perf_counter()
    got = real.evidence_between(t_before, t_after)
    check("the real capture loop recorded the frames it processed",
          got is not None and got["frames"] == len(dim_frames),
          f"{got}")
    check("and a faint dot comes back out with its position (blob 70 < min_peak 80)",
          got is not None and 0 < got["dim_peak"] < 80 and got["dim_pos"] is not None
          and abs(got["dim_pos"][0] - 323) <= 4 and abs(got["dim_pos"][1] - 183) <= 4,
          f"{got}")

    print()
    print("31c. a blob the SIZE GATE refused is reported as that, never as 'saw nothing'")
    # 2026-10-17: the gate rejects a blob before its peak is computed, so
    # `last_dim_*` and rejects["dim"] stayed at zero and `last_rejects["amp_area"]`
    # was written and read by nothing. A shot the gate refused told him "the
    # camera saw nothing at any level" -- the wrong-cause message this project
    # fixed the night before, back through a new door, and it would have
    # corrupted the one measurement his trigger finger can produce.
    #
    # Driven through the REAL capture loop (detector -> evidence -> Stats), then
    # through the real words, with a gain map of 0.25 (N = 4: the gate asks 48 px).
    import re as _re34
    from lsa.shotlog import (classify_missed_pulse as _cmp34,
                             describe_missed_pulse as _dmp34,
                             missed_pulse_note as _mpn34)

    def _run_gated(side: int):
        frames34 = ([frame_with_dot(0)] * 12 + [frame_with_dot(150, size=side)] * 3
                    + [frame_with_dot(0)] * 6)
        # blur and closing off, as in test_detector's size-gate checks, so the
        # blob's area is the dot's area and 6x6 is 36 px against the 48 asked.
        gate34 = DetectorParams(mode=ScoreMode.CHROMA, threshold=10, min_peak=12,
                                blur=0, close_iter=0, min_area=2,
                                use_background=False, max_blobs=4,
                                min_separation=10.0)
        cap34 = _CT31(_CC31(), gate34, _GC31())
        cap34._detector.set_gain_map(np.full((360, 640), 0.25, np.float32))
        cam34 = _OneShotCam(cap34, frames34)
        cap34._open_camera = lambda: cam34
        cap34._negotiated = lambda cap: {"fps": 60.0, "width": 640, "height": 360,
                                         "fourcc": "MJPG", "backend": "STAND-IN",
                                         "requested": {}, "controls": {}}
        t0_ = _t31.perf_counter()
        cap34.run()
        return cap34, cap34.evidence_between(t0_, _t31.perf_counter())

    cap_g, ev_g = _run_gated(6)               # 36 px < 48
    check("the real loop counts a gated blob in Stats (rejected_amp_area), apart from 'too faint'",
          cap_g._stats.rejected_amp_area > 0 and cap_g._stats.rejected_dim == 0,
          f"amp_area {cap_g._stats.rejected_amp_area}, dim {cap_g._stats.rejected_dim}")
    check("the evidence window carries it, with area, the area it needed, N and where",
          ev_g is not None and ev_g["amp"] is not None
          and abs(ev_g["amp"]["area"] - 36) <= 6 and abs(ev_g["amp"]["need"] - 48) < 1
          and abs(ev_g["amp"]["n"] - 4.0) < 0.01
          and abs(ev_g["amp"]["pos"][0] - 323) <= 4 and abs(ev_g["amp"]["pos"][1] - 183) <= 4,
          f"{ev_g}")
    check("and it was NOT filed as 'too faint' (no peak, no position there)",
          ev_g is not None and ev_g["dim_peak"] == 0 and ev_g["dim_pos"] is None, f"{ev_g}")
    check("branch: it is its own branch, not 'nothing'",
          ev_g is not None and _cmp34(ev_g, 8.0) == "amp_area", str(ev_g and _cmp34(ev_g, 8.0)))
    log_g = _dmp34(ev_g, 13.0, 8.0)
    note_g = _mpn34(ev_g, 13.0, 8.0)
    print(f"         log : {log_g}")
    print(f"         note: {note_g}")
    check("THE GUARD: the log line does not say the camera saw nothing",
          "saw nothing" not in log_g and "size gate" in log_g, log_g)
    check("THE GUARD: the target-screen note does not say the camera saw nothing",
          "saw nothing" not in note_g and "too small" in note_g, note_g)
    check("the words carry the numbers: the blob's area, what it needed, and the gain",
          _re34.search(r"blob of (3\d|4[0-5])px at", log_g) is not None
          and "48px needed" in log_g and "amplifies 4.0x" in log_g, log_g)
    check("the note is short enough to read from the firing line (<= 56 chars)",
          len(note_g) <= 56, f"{len(note_g)}: {note_g}")
    cap_ok, ev_ok = _run_gated(10)            # 100 px >= 48: a dot, not gated
    check("control: a blob big enough is detected, and is not reported as gated",
          cap_ok._stats.rejected_amp_area == 0 and ev_ok is not None and ev_ok["amp"] is None,
          f"{cap_ok._stats.rejected_amp_area} {ev_ok}")
    check("control: with nothing gated the old branches are unchanged ('nothing' stays reachable)",
          _cmp34({"frames": 30, "frame_peak": 4, "dim_peak": 0, "dim_pos": None,
                  "off_peak": 0, "short_delta": 0, "amp": None}, 8.0) == "nothing"
          and _cmp34({"frames": 30, "frame_peak": 4, "dim_peak": 0, "dim_pos": None,
                      "off_peak": 0, "short_delta": 0}, 8.0) == "nothing")
    check("a too-short rejection still outranks a gated blob (it cleared the bar)",
          _cmp34(dict(ev_g, short_delta=1), 8.0) == "too_short")

    # -- a halo lump folded into a brighter event (2026-10-17), through the REAL
    # capture loop: detector -> evidence ring -> Stats, and out to the words.
    # The seam between capture.py and evidence.py is the one a unit test of
    # either side cannot see (RULES C14).
    def _halo_frame():
        f = np.full((360, 640, 3), 12, dtype=np.uint8)
        f[100:150, 100:150, 2] = 12 + 80          # 2500 px bloom: area-rejected
        for lx, ly in ((170, 120), (60, 170), (180, 70)):
            f[ly:ly + 8, lx:lx + 8, 2] = 12 + 14   # three lumps of its halo
        return f

    # oversize_min_peak=0: this section is about a bloom that is area-REJECTED
    # (the bright event is not in `dets`). Since 2026-10-17 a bloom that bright
    # is accepted instead; that path is exercised below, apart from this one.
    gate35 = DetectorParams(mode=ScoreMode.CHROMA, threshold=10, min_peak=12,
                            blur=0, close_iter=0, min_area=2,
                            use_background=False, max_blobs=4, min_separation=10.0,
                            oversize_min_peak=0)
    cap35 = _CT31(_CC31(), gate35, _GC31())
    cam35 = _OneShotCam(cap35, [frame_with_dot(0)] * 12 + [_halo_frame()] * 3
                        + [frame_with_dot(0)] * 6)
    cap35._open_camera = lambda: cam35
    cap35._negotiated = lambda cap: {"fps": 60.0, "width": 640, "height": 360,
                                     "fourcc": "MJPG", "backend": "STAND-IN",
                                     "requested": {}, "controls": {}}
    t35 = _t31.perf_counter()
    cap35.run()
    ev35 = cap35.evidence_between(t35, _t31.perf_counter())
    check("the real loop counts folded lumps in Stats (rejected_halo), apart from 'too faint'",
          cap35._stats.rejected_halo >= 3 and cap35._stats.rejected_dim == 0,
          f"halo {cap35._stats.rejected_halo}, dim {cap35._stats.rejected_dim}")
    check("...and NO shot, hole or detection was made of them",
          cap35._stats.shots == 0 and cap35._stats.frames_with_detection == 0,
          f"shots {cap35._stats.shots}, frames with a detection "
          f"{cap35._stats.frames_with_detection}")
    check("the evidence window carries the folded blob",
          ev35 is not None and ev35.get("halo") is not None
          and ev35["halo"]["peak"] == 14, f"{ev35}")
    check("heard-but-not-seen over that window is its own branch, never 'nothing'",
          ev35 is not None and _cmp34(ev35, 10.0) == "halo",
          str(ev35 and _cmp34(ev35, 10.0)))
    check("THE GUARD: neither the log line nor the screen note says the camera saw nothing",
          "saw nothing" not in _dmp34(ev35, 13.0, 10.0)
          and "saw nothing" not in _mpn34(ev35, 13.0, 10.0),
          _dmp34(ev35, 13.0, 10.0))

    # -- a bloom over max_area (2026-10-17, near side), through the REAL loop ------
    def _run_big(peak_add, repeats=3):
        f = np.full((360, 640, 3), 12, dtype=np.uint8)
        f[100:150, 100:150, 2] = 12 + peak_add     # 2500 px > max_area 1500
        g = DetectorParams(mode=ScoreMode.CHROMA, threshold=10, min_peak=12,
                           blur=0, close_iter=0, min_area=2,
                           use_background=False, max_blobs=4, min_separation=10.0)
        cap = _CT31(_CC31(), g, _GC31())
        cam = _OneShotCam(cap, [frame_with_dot(0)] * 12 + [f] * repeats
                          + [frame_with_dot(0)] * 6)
        cap._open_camera = lambda: cam
        cap._negotiated = lambda c: {"fps": 60.0, "width": 640, "height": 360,
                                     "fourcc": "MJPG", "backend": "STAND-IN",
                                     "requested": {}, "controls": {}}
        t0 = _t31.perf_counter()
        cap.run()
        return cap, cap.evidence_between(t0, _t31.perf_counter())

    capb, evb = _run_big(80)
    check("the real loop: a BRIGHT 2500 px bloom becomes a detection and a shot",
          capb._stats.frames_with_detection >= 2 and capb._stats.shots == 1,
          f"detections {capb._stats.frames_with_detection}, shots {capb._stats.shots}")
    capf, evf = _run_big(20)
    check("the real loop: a FAINT 2500 px patch is no shot, and is carried as evidence",
          capf._stats.shots == 0 and evf is not None and evf.get("big") is not None
          and evf["big"]["peak"] == 20, f"shots {capf._stats.shots}, {evf}")
    check("...heard-but-not-seen over it is its own branch (too_big), never 'nothing'",
          evf is not None and _cmp34(evf, 10.0) == "too_big"
          and "saw nothing" not in _dmp34(evf, 13.0, 10.0)
          and "saw nothing" not in _mpn34(evf, 13.0, 10.0),
          str(evf and _cmp34(evf, 10.0)))

    # -- the same path the app takes: _poll_audio -> _missed_pulse_evidence ------
    def _gated_fill(ev, tc):
        for i in range(-20, 25):
            ev.push(tc + i / 60.0, 11, 0, None, 0, 5,
                    amp_gate=(36.0, 48.0, 4.0, 323.0, 183.0) if i == 3 else None)

    heard_g = next((ln for ln in _poll(t_click, _gated_fill) if ln.startswith("HEARD")), "")
    check("through the REAL _poll_audio: the HEARD line names the size gate, not 'nothing'",
          "size gate" in heard_g and "saw nothing" not in heard_g
          and "size-gated blob of 36px at cam (323,183)" in heard_g, heard_g[-320:])
    check("... and so does the note on the target screen",
          "too small" in _poll.note and "saw nothing" not in _poll.note, _poll.note)

    print()
    print("31d. the sensitivity learner learns only from what can be nothing but a laser")
    # 2026-10-17 10:07:27, while a figure walked: of the four regions saved, three
    # were the figure's own edges (peak 11 over 551 px, 12 over 1130 px, 25) and
    # one a real shot (52). The map concluded the LEFT of the screen was weakest
    # where the raw sensor says it is brightest, and amplified it up to 4x.
    # The 20 SHOT lines below are verbatim from camera_lab.log (10:06:40-10:07:27).
    import types as _types35
    from lsa.tools.camera_lab import CameraLab as _CL35

    REAL20 = [(383.10, 314.43, 81), (334.67, 218.69, 11), (313.68, 325.23, 9),
              (467.23, 310.33, 25), (357.01, 255.13, 12), (436.63, 370.83, 10),
              (514.27, 219.88, 9), (522.07, 327.89, 9), (478.63, 426.35, 76),
              (809.90, 321.17, 49), (785.89, 352.01, 64), (808.81, 331.37, 32),
              (762.97, 361.31, 52), (745.96, 324.35, 67), (732.69, 302.44, 45),
              (346.62, 369.70, 62), (460.07, 344.11, 11), (426.94, 272.03, 9),
              (402.46, 372.54, 49), (411.93, 264.94, 58)]

    import time as _time35
    from silent_audio import SilentListener as _SL35

    def _listener35(calibrated=True, running=True, clock=None):
        """The REAL ClickListener logic (SilentListener only opens no device).
        Calibrated and running by default, with NO click heard: so every
        pre-existing check below also proves an uncorroborated faint shot is
        refused by a listener that is entitled to corroborate (C17)."""
        lis = _SL35() if clock is None else _SL35(clock=clock)
        if running:
            lis.start()
        if calibrated:
            lis._learned = [0.5] * lis.LEARN_NEEDED
        return lis

    def _click35(lis, t):
        """Hear one trigger click at perf_counter time t, through the real
        `_decide_click` (so the bar, mute and refractory all apply)."""
        with lis._lock:
            return lis._decide_click(t, 0.9, 0.0, False)

    def _lab35(animating=None, mode=None, listener=None):
        lab = _types35.SimpleNamespace(
            _calibrating=False, capture=object(), _selftest_active=False,
            _modetest_active=False, _sens_shots=[], _sens_applied=0,
            _sens_ignored={"faint": 0, "animating": 0},
            _sens_admitted=0, _sens_by_click=0, _sens_last_by_click=False,
            listener=listener if listener is not None else _listener35(),
            corroborator=ShotCorroborator(),
            settings=_types35.SimpleNamespace(width=1280, height=720),
            mapping=_types35.SimpleNamespace(screen_rect_in_camera=lambda: [
                (300.0, 150.0), (900.0, 150.0), (900.0, 600.0), (300.0, 600.0)]),
            layers=({"modes": _types35.SimpleNamespace(mode=mode, mode2=None)}
                    if mode is not None else {} if animating is None else {
                "modes": _types35.SimpleNamespace(
                    mode2=None,
                    mode=_types35.SimpleNamespace(
                        animated=animating,
                        moving_at=lambda t, a=animating: a))}),
            fits=[], log=[])
        lab._log = lab.log.append
        lab._save_sensitivity = lambda samples: lab.fits.append(list(samples))
        lab._set_gain_map = lambda g, why: None
        lab._report_signal_margin = lambda peaks: None
        lab._judge_map_against_a_fresh_floor = lambda g, why: None
        for n_ in ("LEARN_MIN_SHOTS", "LEARN_BINS", "LEARN_MIN_BINS", "LEARN_MIN_PER_BIN",
                   "SINGLE_FRAME_PEAK", "_SENS_REFUSAL_TEXT", "SENS_CLICK_FORWARD"):
            setattr(lab, n_, getattr(_CL35, n_))
        lab._sens_ignored_note = _types35.MethodType(_CL35._sens_ignored_note, lab)
        lab._sens_refusal = _types35.MethodType(_CL35._sens_refusal, lab)
        lab.learn = _types35.MethodType(_CL35._learn_sensitivity, lab)
        return lab

    def _shot35(x, y, peak, t=0.0):
        return _types35.SimpleNamespace(x=x, y=y, peak=peak, area=300.0,
                                        first_timestamp=t)

    lab35 = _lab35(animating=False)
    for x_, y_, pk_ in REAL20:
        lab35.learn(_shot35(x_, y_, pk_))
    check("of this morning's 20 shots the learner ignores the 9 below the laser/noise boundary",
          lab35._sens_ignored["faint"] == 9 and len(lab35._sens_shots) == 11,
          f"ignored {lab35._sens_ignored}, kept {len(lab35._sens_shots)}")
    check("and 11 is under the 12 it needs, so NO map is built from them "
          "(the old rule built one from the figure's edges)",
          lab35.fits == [], f"{len(lab35.fits)} fits")
    check("no sample below the boundary is ever kept",
          all(pk >= _CL35.SINGLE_FRAME_PEAK for _, _, pk in lab35._sens_shots),
          str([pk for _, _, pk in lab35._sens_shots]))
    check("THE LOG SAYS WHAT IT IGNORED, how many, and why",
          any("ignoring detection at peak 11" in ln and "1 below peak 30" in ln
              for ln in lab35.log)
          and "9 below peak 30" in lab35._sens_ignored_note(),
          str(lab35.log[:2]))
    # With enough real shots a map IS built, and carries only real shots.
    # Twelve more real shots, three in each corner of the screen, the near
    # corners strong and the far ones weak -- enough coverage for a fit.
    more = [(330, 200, 80), (350, 220, 84), (370, 240, 78),
            (840, 200, 38), (860, 220, 40), (820, 240, 36),
            (330, 540, 76), (350, 560, 72), (370, 520, 70),
            (840, 540, 34), (860, 560, 36), (820, 520, 33)]
    for x_, y_, pk_ in more:
        lab35.learn(_shot35(x_, y_, pk_))
    check("given 12+ real shots over 4+ regions a map IS built (the learner still learns)",
          len(lab35.fits) >= 1, f"{len(lab35.fits)} fits from {len(lab35._sens_shots)} shots")
    check("... and every region it saved scored at least the boundary "
          "(it was 11, 12 and 25 this morning)",
          all(s.peak >= _CL35.SINGLE_FRAME_PEAK for fit in lab35.fits for s in fit),
          str([[s.peak for s in fit] for fit in lab35.fits]))
    check("... and the fit's own log line reports the ignored count",
          any("sensitivity learned" in ln and "ignored so far: 9 below peak 30" in ln
              for ln in lab35.log), str(lab35.log[-1:]))

    # Never while the screen is animating, whatever the peak.
    lab_an = _lab35(animating=True)
    for x_, y_, pk_ in REAL20 + more:
        lab_an.learn(_shot35(x_, y_, max(pk_, 90)))      # every one a bright, real-looking peak
    check("while a mode is animating NOTHING is learned, even at peak 90+",
          lab_an._sens_shots == [] and lab_an.fits == []
          and lab_an._sens_ignored["animating"] == len(REAL20 + more),
          f"kept {len(lab_an._sens_shots)}, ignored {lab_an._sens_ignored}")
    check("... and the log says it was the animation",
          any("was MOVING on the screen" in ln for ln in lab_an.log), str(lab_an.log[:1]))
    lab_still = _lab35(animating=False)
    lab_still.learn(_shot35(500, 300, 90))
    check("the same bright shot with the screen still IS learned",
          len(lab_still._sens_shots) == 1)
    lab_none = _lab35(animating=None)
    lab_none.learn(_shot35(500, 300, 90))
    check("and with no mode at all it is learned too (nothing is moving)",
          len(lab_none._sens_shots) == 1)

    # One constant, not a copy (RULES C14).
    lab_c = _lab35(animating=False)
    lab_c.SINGLE_FRAME_PEAK = 50
    check("the boundary is READ from SINGLE_FRAME_PEAK: move it and the learner follows "
          "(peak 40 is refused at 50, accepted at 30)",
          lab_c._sens_refusal(_shot35(500, 300, 40)) == "faint"
          and _lab35(animating=False)._sens_refusal(_shot35(500, 300, 40)) is None)
    check("exactly at the boundary is accepted, one below is not",
          _lab35(animating=False)._sens_refusal(_shot35(500, 300, _CL35.SINGLE_FRAME_PEAK)) is None
          and _lab35(animating=False)._sens_refusal(
              _shot35(500, 300, _CL35.SINGLE_FRAME_PEAK - 1)) == "faint")
    import ast as _ast35
    _src35 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools"
              / "camera_lab.py").read_text(encoding="utf-8")
    _fn35 = next(n_ for n_ in _ast35.walk(_ast35.parse(_src35))
                 if isinstance(n_, _ast35.FunctionDef) and n_.name == "_sens_refusal")
    check("and _sens_refusal holds no numeric copy of it",
          not any(isinstance(n_, _ast35.Constant) and n_.value in (30, 29, 31)
                  for n_ in _ast35.walk(_fn35)),
          "a literal 30 in the refusal")
    _fnL35 = next(n_ for n_ in _ast35.walk(_ast35.parse(_src35))
                  if isinstance(n_, _ast35.FunctionDef) and n_.name == "_learn_sensitivity")
    check("the learner consults the refusal BEFORE it records a sample",
          any(isinstance(n_, _ast35.Attribute) and n_.attr == "_sens_refusal"
              for n_ in _ast35.walk(_fnL35)))

    print()
    print("31d2. a faint shot teaches when a trigger click was heard just before it")
    # 2026-10-23: his right-hand side images at peak 20-35 (median 28-35 vs
    # 85-90 left, 8,415 live shots in 127 logs), so the peak boundary alone
    # meant the region the map exists to recover could never teach it. The
    # microphone hears 12-16% of pulls UNIFORMLY, dim side included. Every
    # timestamp here is perf_counter-based, because the gate requires the
    # click's clock to be the camera's.
    T0 = _time35.perf_counter() + 1000.0
    FAINT_R = 25                                   # a right-hand-side peak

    check("premise: the stand-in listener is the REAL click path (inherits "
          "claim_click_near and _decide_click, overrides neither)",
          "claim_click_near" not in _SL35.__dict__
          and "_decide_click" not in _SL35.__dict__
          and "take_clicks" not in _SL35.__dict__
          and _listener35().clock_is_perf_counter)

    lis_a = _listener35()
    _click35(lis_a, T0 - 0.004)                    # the click precedes the laser
    lab_a = _lab35(animating=False, listener=lis_a)
    r_a = lab_a._sens_refusal(_shot35(900, 300, FAINT_R, t=T0))
    check("a FAINT right-hand shot with a click 4 ms before it is admitted",
          r_a is None and lab_a._sens_last_by_click is True, f"refusal {r_a!r}")
    lab_a2 = _lab35(animating=False, listener=_listener35())   # hears nothing
    check("the same faint shot with NO click is still refused as faint",
          lab_a2._sens_refusal(_shot35(900, 300, FAINT_R, t=T0)) == "faint"
          and lab_a2._sens_last_by_click is False)
    lis_a3 = _listener35()
    _click35(lis_a3, T0 - 0.004)
    lab_a3 = _lab35(animating=False, listener=lis_a3)
    lab_a3.learn(_shot35(900, 300, FAINT_R, t=T0))
    check("learn(): an admitted faint shot is KEPT and counted (admitted 1, "
          "held-on-a-click 1, none counted as ignored)",
          len(lab_a3._sens_shots) == 1 and lab_a3._sens_admitted == 1
          and lab_a3._sens_by_click == 1 and lab_a3._sens_ignored["faint"] == 0,
          f"kept {len(lab_a3._sens_shots)}, admitted {lab_a3._sens_admitted}, "
          f"ignored {lab_a3._sens_ignored}")
    check("... the FIRST admission is logged, and the note carries the count",
          any("ADMITTING detection at peak 25" in ln for ln in lab_a3.log)
          and "1 faint admitted on the microphone's word" in lab_a3._sens_ignored_note(),
          str(lab_a3.log[-1:]))

    # A bright shot never leaves the side channel set.
    lab_b = _lab35(animating=False, listener=lis_a)
    lab_b._sens_last_by_click = True
    lab_b._sens_refusal(_shot35(500, 300, 90, t=T0))
    check("a bright shot clears the side channel (it rests on brightness, not on a click)",
          lab_b._sens_last_by_click is False)

    # -- moving content is NEVER overridden, and spends no click ----------------
    lis_m = _listener35()
    _click35(lis_m, T0 - 0.004)
    lab_m = _lab35(animating=True, listener=lis_m)
    check("a faint shot in a MOVING mode is refused even with a click right before it",
          lab_m._sens_refusal(_shot35(900, 300, FAINT_R, t=T0)) == "faint"
          and lab_m._sens_last_by_click is False)
    check("... and the click was NOT spent on it (a later still-mode shot can use it)",
          lis_m.claim_click_near(T0, 0.3, 0.02) is not None)
    lab_m.learn(_shot35(900, 300, FAINT_R, t=T0))
    check("learn() in a moving mode keeps nothing from a faint corroborated shot",
          lab_m._sens_shots == [] and lab_m._sens_admitted == 0)

    # -- one click vouches for one detection ------------------------------------
    lis_x = _listener35()
    _click35(lis_x, T0 - 0.004)
    lab_x = _lab35(animating=False, listener=lis_x)
    first_x = lab_x._sens_refusal(_shot35(900, 300, FAINT_R, t=T0))
    second_x = lab_x._sens_refusal(_shot35(880, 320, FAINT_R, t=T0 + 0.01))
    check("one click corroborates exactly ONE of two nearby faint detections",
          first_x is None and second_x == "faint",
          f"first {first_x!r}, second {second_x!r}")
    lis_x2 = _listener35()
    _click35(lis_x2, T0 - 0.004)
    _click35(lis_x2, T0 + 0.2)                     # a later pull, 200 ms on
    lab_x2 = _lab35(animating=False, listener=lis_x2)
    check("... but two clicks teach two (the claim is per click, not per window)",
          lab_x2._sens_refusal(_shot35(900, 300, FAINT_R, t=T0)) is None
          and lab_x2._sens_refusal(_shot35(880, 320, FAINT_R, t=T0 + 0.2)) is None)
    lis_n = _listener35()
    _click35(lis_n, T0 - 0.100)                    # nearer in time to the SECOND
    _click35(lis_n, T0 + 0.100 - 0.004)
    taken_n = lis_n.claim_click_near(T0 + 0.100, 0.3, 0.02)
    check("the NEAREST unclaimed click is the one claimed",
          taken_n is not None and abs(taken_n.t - (T0 + 0.096)) < 1e-9)

    # -- the app's own bang lives AFTER the pulse and must not corroborate -------
    lis_p = _listener35()
    _click35(lis_p, T0 + 0.040)                    # ~ our bang: 38 ms+ after frame 1
    lab_p = _lab35(animating=False, listener=lis_p)
    check("a click arriving only AFTER the pulse (where our own bang lives) does NOT "
          "corroborate",
          lab_p._sens_refusal(_shot35(900, 300, FAINT_R, t=T0)) == "faint")
    lis_p2 = _listener35()
    _click35(lis_p2, T0 + 0.015)                   # inside the 20 ms of forward slack
    check("... while a click 15 ms after still does (clock skew, inside the slack)",
          _lab35(animating=False, listener=lis_p2)._sens_refusal(
              _shot35(900, 300, FAINT_R, t=T0)) is None)
    lis_p3 = _listener35()
    _click35(lis_p3, T0 + 0.025)
    check("... and 25 ms after does not (just outside it)",
          _lab35(animating=False, listener=lis_p3)._sens_refusal(
              _shot35(900, 300, FAINT_R, t=T0)) == "faint")
    win_s = ShotCorroborator().window
    lis_w = _listener35()
    _click35(lis_w, T0 - (win_s + 0.05))
    check("a click older than the corroborator's own window does not corroborate",
          _lab35(animating=False, listener=lis_w)._sens_refusal(
              _shot35(900, 300, FAINT_R, t=T0)) == "faint")
    lis_w2 = _listener35()
    _click35(lis_w2, T0 - (win_s - 0.05))
    check("... one inside it does",
          _lab35(animating=False, listener=lis_w2)._sens_refusal(
              _shot35(900, 300, FAINT_R, t=T0)) is None)

    # -- the listener must be entitled to speak ---------------------------------
    lis_u = _listener35(calibrated=False)
    _click35(lis_u, T0 - 0.004)
    check("an UNCALIBRATED microphone (a guessed threshold) corroborates nothing",
          _lab35(animating=False, listener=lis_u)._sens_refusal(
              _shot35(900, 300, FAINT_R, t=T0)) == "faint")
    lis_r = _listener35(running=False)
    _click35(lis_r, T0 - 0.004)
    check("a listener that is not running corroborates nothing",
          _lab35(animating=False, listener=lis_r)._sens_refusal(
              _shot35(900, 300, FAINT_R, t=T0)) == "faint")
    lis_c = _listener35(clock=_time35.monotonic)
    # THE CLICK GOES AT THE SHOT'S OWN POSITION (T0 - 0.004), not at
    # monotonic() - 0.004. Those are different epochs on Windows -- perf_counter
    # is QPC, monotonic is GetTickCount64 -- so a click filed on the monotonic
    # scale lands nowhere near T0, there is no click in the window at all, and
    # the refusal came back "faint" for a reason that has nothing to do with the
    # clock. The check therefore PASSED with the clock guard removed: measured
    # 2026-10-24, when mutate_detector entry 18 came back MISSED.
    #
    # Filed here, the ONLY thing left to refuse it for is which clock the
    # listener is on, which is the property this check is named after.
    _click35(lis_c, T0 - 0.004)
    check("a listener on a DIFFERENT clock (RULES B4) corroborates nothing",
          (not lis_c.clock_is_perf_counter)
          and _lab35(animating=False, listener=lis_c)._sens_refusal(
              _shot35(900, 300, FAINT_R, t=T0)) == "faint")

    # -- a map CAN now be earned from the dim side alone -------------------------
    lis_g = _listener35()
    lab_g = _lab35(animating=False, listener=lis_g)
    g_shots = [(330, 200, 80), (350, 220, 84), (370, 240, 78),       # near: bright
               (330, 540, 76), (350, 560, 72), (370, 520, 70),
               (840, 200, 25), (860, 220, 26), (820, 240, 24),       # far: faint
               (840, 540, 22), (860, 560, 23), (820, 520, 21)]
    for k_, (x_, y_, pk_) in enumerate(g_shots):
        t_ = T0 + 5.0 + k_ * 0.5
        if pk_ < _CL35.SINGLE_FRAME_PEAK:
            _click35(lis_g, t_ - 0.004)            # each faint pull was heard
        lab_g.learn(_shot35(x_, y_, pk_, t=t_))
    check("a fresh install's dim right-hand side (peaks 21-26, all heard) now BUILDS a map",
          len(lab_g.fits) >= 1 and len(lab_g._sens_shots) == 12,
          f"{len(lab_g.fits)} fits from {len(lab_g._sens_shots)} shots")
    check("... whose far-side regions are below the boundary: brightness alone could never "
          "have taught them",
          any(s.peak < _CL35.SINGLE_FRAME_PEAK for fit in lab_g.fits for s in fit),
          str([[s.peak for s in fit] for fit in lab_g.fits]))
    check("... and the fit's log line says SIX of its shots rest on the microphone",
          any("sensitivity learned from 12 real shots (6 of them resting on a trigger click"
              in ln for ln in lab_g.log), str(lab_g.log[-1:]))
    lab_g2 = _lab35(animating=False)
    for k_, (x_, y_, pk_) in enumerate(g_shots):
        lab_g2.learn(_shot35(x_, y_, pk_, t=T0 + 5.0 + k_ * 0.5))
    check("the same twelve with NO clicks heard build nothing (the old behaviour, preserved)",
          lab_g2.fits == [] and lab_g2._sens_ignored["faint"] == 6)

    # -- the source says what the brief says -------------------------------------
    _fnR35 = next(n_ for n_ in _ast35.walk(_ast35.parse(_src35))
                  if isinstance(n_, _ast35.FunctionDef) and n_.name == "_sens_refusal")
    _attrs35 = {n_.attr for n_ in _ast35.walk(_fnR35) if isinstance(n_, _ast35.Attribute)}
    check("_sens_refusal reads the CLICK channel, never audio_at's raw peak (which includes "
          "our own bang)",
          "claim_click_near" in _attrs35 and "audio_at" not in _attrs35
          and "note_seen_shot" not in _attrs35, str(sorted(_attrs35)))
    check("... takes its back-window from the corroborator and its forward slack from one "
          "named constant, with no numeric copy of either",
          "window" in _attrs35 and "SENS_CLICK_FORWARD" in _attrs35
          and not any(isinstance(n_, _ast35.Constant) and n_.value in (0.3, 0.02, 0.2)
                      for n_ in _ast35.walk(_fnR35)))
    check("... requires the listener running, on the camera's clock, and calibrated",
          {"running", "clock_is_perf_counter", "calibrated"} <= _attrs35, str(sorted(_attrs35)))
    check("... and holds no getattr (RULES C8)",
          not any(isinstance(n_, _ast35.Name) and n_.id == "getattr"
                  for n_ in _ast35.walk(_fnR35)))
    check("the forward slack is smaller than the quickest our own bang can arrive "
          "(one frame at 60 fps = 16.7 ms, plus ~21 ms output latency)",
          _CL35.SENS_CLICK_FORWARD < (1.0 / 60.0 + 0.021) and _CL35.SENS_CLICK_FORWARD > 0.0)

    print()
    print("31e. the learner judges the screen AS IT WAS WHEN THE SHOT WAS FIRED, with REAL modes")
    # The reviewer's measurement (2026-10-17): against the real default mode,
    # `animated` is False before a shot, True after a scoring HIT (the hit
    # appended a flash ring) and False after a miss -- so a learner reading it
    # after host.shot() refused every hit and learned only from misses. A stub
    # cannot flash, so these drive the real modes on their own clocks.
    from lsa.modes import ShotEvent as _SE35
    from lsa.modes.advancing import AdvancingTarget as _AT35
    from lsa.modes.static import StaticTargets as _ST35
    from figure_helpers import context as _ctx35, drive_to_live as _live35

    # -- static: a scoring HIT must still be learnable -----------------------
    st35 = _ST35()
    st35.start(_ctx35(now=100.0))
    cx35, cy35, _r35 = st35._field.marks(_ctx35(now=100.0))[0]
    pre_anim = st35.animated
    out35 = st35.shot(_SE35(cx35, cy35, 100.0, 1), _ctx35(now=100.0))
    check("static: premise -- `animated` was False before the hit and True after it "
          "(the very flaw being fixed)",
          (not pre_anim) and out35.hit and st35.animated is True,
          f"before {pre_anim}, hit {out35.hit}, after {st35.animated}")
    lab_st = _lab35(mode=st35)
    ref_hit = lab_st._sens_refusal(_shot35(cx35, cy35, 60, t=100.0))
    check("static: a peak-60 shot fired AT the centre mark, judged AFTER the hit "
          "flashed, is ACCEPTED (red against the old `mode.animated` read)",
          ref_hit is None, f"refusal {ref_hit!r}")
    lab_st.learn(_shot35(cx35, cy35, 60, t=100.0))
    check("... and it was kept as a sample", len(lab_st._sens_shots) == 1)
    st_miss = _ST35()
    st_miss.start(_ctx35(now=100.0))
    st_miss.shot(_SE35(5.0, 5.0, 100.0, 1), _ctx35(now=100.0))
    check("static: the same shot after a MISS is accepted too (hit and miss alike)",
          _lab35(mode=st_miss)._sens_refusal(_shot35(5, 5, 60, t=100.0)) is None)

    # -- advancing: refused while the figure MOVES, accepted at a HOLD -------
    adv = _AT35()
    adv._rng.seed(5)
    adv.start(_ctx35(now=0.0), figure="thug01")
    _live35(adv, 0.0)
    a35 = adv._actors[0]
    t_move = a35.t0 + a35.t_a * 0.5                       # mid-approach
    t_hold = a35.t0 + a35.t_a + a35.t_h * 0.5             # holding near
    adv.advance(_ctx35(now=t_move))
    lab_adv = _lab35(mode=adv)
    check("advancing: a peak-60 shot fired while the figure is MOVING is refused",
          adv.moving_at(t_move)
          and lab_adv._sens_refusal(_shot35(900, 500, 60, t=t_move)) == "animating",
          f"moving_at {adv.moving_at(t_move)}")
    adv.advance(_ctx35(now=t_hold))
    check("advancing: the same shot at a HOLD (figure standing still) is accepted",
          (not adv.moving_at(t_hold))
          and lab_adv._sens_refusal(_shot35(900, 500, 60, t=t_hold)) is None,
          f"moving_at {adv.moving_at(t_hold)}")
    # The answer is a property of the schedule at time t: a HIT must not change it.
    c35, h35 = a35.state(t_move)
    inn35 = a35.fig.innermost(c35, h35)
    before_hit = (adv.moving_at(t_move), adv.moving_at(t_hold))
    out_adv = adv.shot(_SE35(inn35[1], inn35[2], t_move, 1), _ctx35(now=t_move))
    check("advancing: a scoring HIT does not change the answer about when it was fired "
          "(and the mode is `animated` afterwards, which is what misled the old read)",
          out_adv.hit and adv.animated
          and before_hit == (adv.moving_at(t_move), adv.moving_at(t_hold)),
          f"hit {out_adv.hit}, animated {adv.animated}")
    check("... so judged after the hit, the moving-time shot is still refused",
          lab_adv._sens_refusal(_shot35(900, 500, 60, t=t_move)) == "animating")

    # -- every registered mode: the policy is stated, not accidental ---------
    import lsa.modes as _modes35
    names35 = sorted(c_.name for c_ in _modes35.available())
    always = []
    for cls_ in _modes35.available():
        inst = cls_()
        try:
            inst.start(_ctx35(now=50.0))
        except TypeError:
            pass
        if inst.moving_at(50.0):
            always.append(inst.name)
    print(f"         modes where the learner is paused while content moves at t=50: "
          f"{sorted(always)} (of {names35})")
    # THE MODES WHOSE CONTENT NEVER STOPS. A ball is always in flight, and
    # the animated figure (key A, 2026-10-18) is a filmed clip that moves in
    # every frame of its loop -- so a shot fired at any of them may never
    # teach the sensitivity map. On 2026-10-17 a WALKING figure's edges
    # taught the map that the brightest third of the screen was its weakest,
    # and the correction cost him the right-hand side of the panel.
    #
    # Everything else holds still between shots and may teach it. The list
    # is explicit rather than a count: "five with permanent animated = True"
    # was prose that went stale the moment a mode was added.
    check("the modes whose content never stops are paused for a whole round; "
          "every other mode lets the map learn at rest",
          sorted(always) == ["animated", "ball", "balls"], str(sorted(always)))
    lab_adv.learn(_shot35(900, 500, 90, t=t_move))
    check("the log line for a paused mode NAMES the mode (a silently dead learner is the fault)",
          any("in mode 'advancing'" in ln and "paused" in ln for ln in lab_adv.log),
          str(lab_adv.log[-1:]))

    print()
    print("31f. two dots in one photograph are each read in their own neighbourhood")
    # The exposure probe draws the same dot at the weak and the strong end of
    # the panel in ONE frame. A whole-ROI maximum (`_probe_measure`) would
    # return the brighter dot for both; the pair measure must not.
    import numpy as _np31f
    import cv2 as _cv31f
    from lsa.tools.camera_lab import CameraLab as _CL31f

    def _img31f(dots):
        im = _np31f.zeros((200, 400, 3), _np31f.uint8)
        for (cx_, cy_), red_ in dots:
            _cv31f.circle(im, (cx_, cy_), 8, (0, 0, red_), -1)       # BGR: red only
        return im

    dark31 = _img31f([])
    lit31 = _img31f([((60, 100), 70), ((320, 100), 150)])        # weak left, strong right
    lab31 = _types35.SimpleNamespace(
        PROBE_SPOT_HALF=_CL31f.PROBE_SPOT_HALF, capture=None,
        PROBE_FLOOR_EXCLUDE_PX=_CL31f.PROBE_FLOOR_EXCLUDE_PX,
        mapping=_types35.SimpleNamespace(to_camera=lambda sx, sy: (sx * 4.0, sy * 2.0)),
        target_window=_types35.SimpleNamespace(
            rect=lambda: _types35.SimpleNamespace(width=lambda: 100, height=lambda: 100)))
    lab31._probe_measure = _types35.MethodType(_CL31f._probe_measure, lab31)
    w31, fl31, _m31, s31 = _CL31f._probe_measure_pair(lab31, lit31, dark31,
                                                      (0.15, 0.5), (0.80, 0.5))
    old31 = _CL31f._probe_measure(lab31, lit31, dark31)[0]
    check("the weak spot reads its OWN dot (70), the strong spot its own (150)",
          abs(w31 - 70) <= 1 and abs(s31 - 150) <= 1, f"weak {w31}, strong {s31}")
    check("the whole-ROI maximum would have said 150 for both (why a window is needed)",
          abs(old31 - 150) <= 1, f"{old31}")
    check("the floor is the worst of the picture AWAY from both dots (0 on a clean frame)",
          fl31 == 0.0, f"{fl31}")
    lit31n = _img31f([((60, 100), 70), ((320, 100), 150)])
    lit31n[20, 200, 2] = 33                                    # noise away from both dots
    _w, fln31, _m, _s = _CL31f._probe_measure_pair(lab31, lit31n, dark31, (0.15, 0.5), (0.80, 0.5))
    check("and a bright speck away from both dots IS the floor", fln31 == 33.0, f"{fln31}")
    # THE BLOOM. Live 2026-10-18: the floor read 101-104 at every exposure with its
    # maximum 41 px from the strong dot -- the saturated dot's own glow -- so the bar sat
    # above the dot and the probe never chose an exposure on any launch.
    litb = _img31f([((60, 100), 70), ((320, 100), 150)])
    _cv31f.circle(litb, (320, 100), 70, (0, 0, 100), -1)       # glow reaching 70 px out
    _cv31f.circle(litb, (60, 100), 70, (0, 0, 60), -1)
    # the glow is drawn first in effect: put the cores back on top
    _cv31f.circle(litb, (60, 100), 8, (0, 0, 70), -1)
    _cv31f.circle(litb, (320, 100), 8, (0, 0, 150), -1)
    _wb, flb, _mb, _sb = _CL31f._probe_measure_pair(lab31, litb, dark31, (0.15, 0.5), (0.80, 0.5))
    check("a dot's own glow within 70 px is NOT the background floor (0, not 100)",
          flb == 0.0 and abs(_wb - 70) <= 1 and abs(_sb - 150) <= 1,
          f"floor {flb}, weak {_wb}, strong {_sb}")
    w1, f1, m1, s1 = _CL31f._probe_measure_pair(lab31, lit31, dark31, (0.15, 0.5), None)
    check("with no strong spot it reads the weak end the old way and reports no strong peak",
          s1 is None and abs(w1 - 150) <= 1, f"{w1}, {s1}")
    lab31.mapping = _types35.SimpleNamespace(to_camera=lambda sx, sy: None)
    w2, f2, m2, s2 = _CL31f._probe_measure_pair(lab31, lit31, dark31, (0.15, 0.5), (0.80, 0.5))
    check("when the spots cannot be placed it falls back, and says nothing about the strong end",
          s2 is None, f"{s2}")

    print()
    print("32b. a microphone level of 0.000 says whether the room was silent or the app was deaf")
    # The self-mute (~0.8 s after the app's own gunshot) and a stream that is not
    # delivering both read as 0.000 beside a silent room. Only the first is
    # evidence about the gun. `ClickListener.audio_at` is read-only: the muting,
    # the thresholds and the learning are untouched.
    import numpy as _np32b
    from lsa.audio.listener import ClickListener as _CL32b
    from lsa.shotlog import mic_state as _ms32b

    lis_b = _CL32b()                       # never started: no device is opened
    for _ in range(40):                    # a quiet room, delivered block by block
        lis_b._on_audio(_np32b.full((256, 1), 0.004, _np32b.float32), 256, None, None)
    now_b = _t31.perf_counter()
    pk, blocks, deaf = lis_b.audio_at(now_b)
    check("listening, in a quiet room: a small level, blocks present, not deaf",
          0.0 < pk < 0.01 and blocks > 0 and deaf == 0.0, f"{pk} {blocks} {deaf}")
    lis_b.mute_for(0.81)                   # the app plays its own gunshot
    lis_b._on_audio(_np32b.full((256, 1), 0.6, _np32b.float32), 256, None, None)
    t_second = _t31.perf_counter() + 0.30  # the second fragment, 0.3 s later
    pk2, blocks2, deaf2 = lis_b.audio_at(t_second)
    check("0.3 s after the app's own gunshot the microphone is DEAF for ~0.5 s more",
          0.45 < deaf2 < 0.55, f"deaf for {deaf2:.3f}s")
    pk3, blocks3, deaf3 = lis_b.audio_at(_t31.perf_counter() + 1.2)
    check("and after the mute has run out it is not", deaf3 == 0.0, f"{deaf3}")
    pk4, blocks4, deaf4 = lis_b.audio_at(_t31.perf_counter() - 600.0)
    check("a time with no audio blocks near it reports ZERO blocks (not a silent room)",
          pk4 == 0.0 and blocks4 == 0 and deaf4 == 0.0, f"{pk4} {blocks4} {deaf4}")
    clicks_b = lis_b.take_clicks()
    check("the mute still does its job: the loud block during it made NO click "
          "(observation did not change behaviour)", clicks_b == [], str(clicks_b))
    check("words: listening -> no tag", _ms32b(12, 0.0) == "")
    check("words: muted -> DEAF, seconds, not a phantom",
          "DEAF" in _ms32b(12, 0.48) and "0.48s" in _ms32b(12, 0.48)
          and "not evidence of a phantom" in _ms32b(12, 0.48))
    check("words: no audio -> says it is not a silent room",
          "NO AUDIO" in _ms32b(0, 0.0) and "not a silent room" in _ms32b(0, 0.0))

    # The SAW line: a visual detection with no click is only a phantom if the
    # microphone was LISTENING when it happened.
    class _DeafListener:
        running = True
        calibrated = True
        def __init__(self, blocks, deaf): self._b, self._d = blocks, deaf
        def take_clicks(self): return []
        def audio_at(self, t, window=0.25): return 0.0, self._b, self._d

    def _poll_saw(listener):
        co_ = ShotCorroborator()
        co_.add_shot(_t31.perf_counter() - 1.0)          # a detection, nothing heard
        lab_ = _types31.SimpleNamespace(
            listener=listener, corroborator=co_, capture=None,
            _selftest_active=False, _modetest_active=False, _probe_active=False,
            _audio_note="", _audio_note_until=0.0, sp_minpeak=_Spin(13),
            sp_threshold=_Spin(8),
            # The real lab ALWAYS has a modes layer by the time `_poll_audio`
            # runs; see the long note on the first stand-in above for what a
            # double missing it cost on 2026-10-19.
            layers={"modes": _mode_host_stub()})
        _poll_saw.host = lab_.layers["modes"]
        out_: list[str] = []
        lab_._log = out_.append
        lab_._watch_feed = lambda clicks, now: None   # fed by `_poll_audio`; this fixture has no clicks
        _CL31._poll_audio(lab_)
        return next((ln for ln in out_ if ln.startswith("SAW")), ""), lab_._audio_note

    saw_ok, note_ok = _poll_saw(_DeafListener(20, 0.0))
    check("SAW with a listening microphone is still reported as a likely phantom",
          "likely a phantom" in saw_ok and "phantom" in note_ok, saw_ok + " | " + note_ok)
    # ACROSS THE SEAM: the record guard's denominator is fed by the real poll,
    # with the deaf flag the microphone state implies (listening -> False).
    check("a seen shot the LISTENING microphone missed reaches the host as not-deaf",
          len(_poll_saw.host.rec.live_unheard) == 1
          and _poll_saw.host.rec.live_unheard[0][1] is False,
          repr(_poll_saw.host.rec.live_unheard))
    saw_deaf, note_deaf = _poll_saw(_DeafListener(20, 0.5))
    check("and one it was DEAF for reaches the host as deaf, leaving the denominator",
          len(_poll_saw.host.rec.live_unheard) == 1
          and _poll_saw.host.rec.live_unheard[0][1] is True,
          repr(_poll_saw.host.rec.live_unheard))
    check("SAW while the microphone was self-muted is NOT called a phantom",
          saw_deaf.startswith("SAW a shot with no gunshot sound, but")
          and "DEAF" in saw_deaf and "likely a phantom" not in saw_deaf
          and "phantom" not in note_deaf.replace("no evidence", ""), saw_deaf + " | " + note_deaf)
    saw_none, note_none = _poll_saw(_DeafListener(0, 0.0))
    check("SAW with no audio delivered is NOT called a phantom either",
          "NO AUDIO" in saw_none and "likely a phantom" not in saw_none, saw_none)

    print()
    print("33. tools/audio_meter.py reads what Windows remembers, and can change nothing")
    # Plan 02 step 3, the silent half. 2026-10-17: Windows remembers the python
    # interpreter at volume 0 on the monitor's output, and the first draft of
    # the line that COUNTS such entries treated a stored 0.0 as "missing" (0.0 is
    # falsy) and reported none. Decoded here against bytes read from the
    # registry on this machine, and fenced so the tool can never act.
    import ast as _ast33
    import importlib.util as _ilu33

    _meter_path = Path(__file__).resolve().parents[1] / "tools" / "audio_meter.py"
    _spec33 = _ilu33.spec_from_file_location("audio_meter_under_test", _meter_path)
    meter = _ilu33.module_from_spec(_spec33)
    _spec33.loader.exec_module(meter)

    zero = bytes.fromhex("040000000000000000000000000000000000000000000000")
    one = bytes.fromhex("04000000000000000000803f000000000000000000000000")
    false_ = bytes.fromhex("0b0000000000000000000000000000000000000000000000")
    true_ = bytes.fromhex("0b00000000000000ffff00000000000000000000000000")
    chans = bytes.fromhex("0420000000000000180000000000000000000000000000000000803f0000803f")
    check("a stored volume of 1.0 decodes to 1.0", meter.decode_float(one) == 1.0)
    check("a stored volume of 0.0 decodes to 0.0 (and not to 'nothing')",
          meter.decode_float(zero) == 0.0 and meter.decode_float(zero) is not None)
    check("an unknown type decodes to None rather than to a guess",
          meter.decode_float(false_) is None and meter.decode_bool(one) is None)
    check("the boolean decodes both ways", meter.decode_bool(false_) is False
          and meter.decode_bool(true_) is True)
    check("the channel array decodes", meter.decode_floats(chans) == [0.0, 0.0, 1.0, 1.0],
          str(meter.decode_floats(chans)))

    real = (r"{2}.\\?\hdaudio#func_01&ven_10de&dev_00a7&subsys_103c8c3a&rev_1001#"
            r"{6994ad04-93ef-11d0-a3cc-00a0c9223196}\topo00/00010001|\Device"
            r"\HarddiskVolume3\Python313\python.exe%b{00000000-0000-0000-0000-000000000000}")
    pe = meter.parse_entry(real)
    check("a remembered-application key splits into hardware and exe",
          pe is not None and meter.exe_tail(pe["exe"]) == r"\Python313\python.exe"
          and meter.hardware_key(pe["dev"])
          == "HDAUDIO\\FUNC_01&VEN_10DE&DEV_00A7&SUBSYS_103C8C3A&REV_1001", str(pe))
    sysnote = (r"{2}.\\?\hdaudio#func_01&ven_10de&dev_00a7&subsys_103c8c3a&rev_1001#"
               r"{6994ad04-93ef-11d0-a3cc-00a0c9223196}\topo00/00010001|#%b{A9EF3FD9-4240-455E-A4D5-F2B3301887B2}")
    check("the system-sounds entry (no exe) parses and is not 'ours'",
          meter.parse_entry(sysnote) is not None
          and not meter.is_ours(meter.parse_entry(sysnote)["exe"]))
    check("python, pythonw and the project path are ours; a browser is not",
          all(meter.is_ours(x) for x in (r"\Python313\python.exe", r"\Python313\pythonw.exe",
                                         r"\WORKSPACE\Laser.Shooting.App\x.exe",
                                         r"\github\screenplink\screenplink.exe"))
          and not meter.is_ours(r"\Program Files\Google\Chrome\Application\chrome.exe"))

    entries = [
        {"parsed": {"exe": r"\Device\HarddiskVolume3\Python313\python.exe"},
         "values": {"3": zero}},
        {"parsed": {"exe": r"\Device\HarddiskVolume3\Program Files\VideoLAN\VLC\vlc.exe"},
         "values": {"3": one}},
        {"parsed": {"exe": r"\Device\HarddiskVolume3\Python313\pythonw.exe"}, "values": {}},
    ]
    check("an entry stored at volume 0.0 IS counted as quiet (the `or 1.0` bug)",
          meter.quiet_entries(entries) == [r"\Python313\python.exe"],
          str(meter.quiet_entries(entries)))

    # -- which interpreter is OURS: from pyvenv.cfg, not from the word "python" ----
    import tempfile as _tf33
    root33 = Path(__file__).resolve().parents[1]
    real_info = meter.venv_interpreters(str(root33 / ".venv"))
    check("this repo's .venv/pyvenv.cfg is read: home and executable come from the file",
          real_info["confirmed"] and real_info["cfg_home"]
          and real_info["cfg_home"].lower().rstrip("\\") in
          (real_info["cfg_executable"] or "").lower(), str(real_info))
    home_tail = meter._tail_key(str(Path(real_info["cfg_home"]) / "python.exe"))
    check("the base interpreter pyvenv.cfg names is labelled CONFIRMED, citing the file",
          "confirmed by pyvenv.cfg" in meter.interpreter_label(home_tail, real_info),
          meter.interpreter_label(home_tail, real_info))
    check("another python.exe elsewhere is `is_ours` by the word but is NOT labelled ours",
          meter.is_ours(r"\Other\python.exe")
          and meter.interpreter_label(r"\other\python.exe", real_info) == "")
    with _tf33.TemporaryDirectory() as td33:
        none_info = meter.venv_interpreters(td33, base_exe=r"C:\Python313\python.exe")
        lab_none = meter.interpreter_label(r"\python313\python.exe", none_info)
        check("with no pyvenv.cfg it says PROBABLY, and prints what it rests on",
              not none_info["confirmed"] and "PROBABLY" in lab_none
              and "sys._base_executable" in lab_none and r"\python313\python.exe" in lab_none,
              lab_none)
        (Path(td33) / "pyvenv.cfg").write_text(
            "home = D:\\Elsewhere\ninclude-system-site-packages = false\n", encoding="utf-8")
        dis = meter.venv_interpreters(td33, base_exe=r"C:\Python313\python.exe")
        lab_dis = meter.interpreter_label(r"\python313\python.exe", dis)
        check("when pyvenv.cfg and sys._base_executable disagree it says NOT confirmed",
              dis["confirmed"] and "NOT confirmed" in lab_dis
              and "confirmed by pyvenv.cfg" not in lab_dis, lab_dis)

    # -- the probable cause leads; the asleep-endpoint note is context ------------
    f_ctx = ("Context, not the probable cause: the system default output is now 'Speaker', "
             "but the last launch played on 'DELL U4320Q'. A display's audio endpoint disappears")
    f_mix = "a live python session on 'DELL U4320Q' (pid 12) is at 0%: report only, never touched"
    f_reg = ("Windows remembers '\\Python313\\python.exe' as volume 0% on "
             "HDAUDIO\\FUNC_01&VEN_10DE&DEV_00A7 (decoded by pattern, unconfirmed)")
    f_ep = "the default output 'DELL U4320Q' master volume is 1%: report only, never touched"
    ranked = meter.rank_findings([f_ctx, f_ep, f_reg, f_mix])
    check("per-application volume 0 (remembered or live) leads; endpoint volume next; the "
          "asleep-endpoint note is last, as context",
          ranked[0] in (f_reg, f_mix) and ranked[1] in (f_reg, f_mix) and ranked[2] == f_ep
          and ranked[3] == f_ctx, str([r[:24] for r in ranked]))
    check("the app's own finding strings really do start with the words ranking relies on",
          'f"Context, not the probable cause: the system default output is now' in
          _meter_path.read_text(encoding="utf-8")
          and 'f"Windows remembers' in _meter_path.read_text(encoding="utf-8")
          and "live python session" in _meter_path.read_text(encoding="utf-8"))

    # -- the fence: this tool reads, and only reads --------------------------
    src33 = _meter_path.read_text(encoding="utf-8")
    tree33 = _ast33.parse(src33)
    banned_calls = {"SetDefaultDevice", "SetMute", "SetMasterVolume", "SetMasterVolumeLevel",
                    "SetMasterVolumeLevelScalar", "SetChannelVolume", "SetValue",
                    "SetValueEx", "CreateKey", "CreateKeyEx", "DeleteKey", "DeleteKeyEx",
                    "DeleteValue", "OutputStream", "InputStream", "RawOutputStream",
                    "play", "default_mixer", "SetSubscription", "SaveKey"}
    banned_attrs = {"KEY_WRITE", "KEY_SET_VALUE", "KEY_ALL_ACCESS", "KEY_CREATE_SUB_KEY"}
    called, attrs, imports = set(), set(), set()
    for node in _ast33.walk(tree33):
        if isinstance(node, _ast33.Call):
            f = node.func
            called.add(f.attr if isinstance(f, _ast33.Attribute) else getattr(f, "id", ""))
        elif isinstance(node, _ast33.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, _ast33.Import):
            imports |= {a.name for a in node.names}
        elif isinstance(node, _ast33.ImportFrom):
            imports.add(node.module or "")
    check("it calls no setter, opens no stream, plays nothing, writes no registry value",
          not (called & banned_calls), f"calls: {sorted(called & banned_calls)}")
    check("it asks the registry for read access only",
          not (attrs & banned_attrs) and "KEY_READ" in attrs, f"{sorted(attrs & banned_attrs)}")
    check("it imports neither the app's mixer nor sounddevice (no output device, ever)",
          not any(m.startswith(("sounddevice", "lsa.audio")) for m in imports), str(imports))
    check("it never touches the device-picking rule (RULES D7)",
          "_pick_output" not in src33 and "pick_device" not in src33)

    # ---------------------------------------------------------------------- 40 -
    print("\n40. the corroborator publishes one decision per PULL (the dry-pull tally is a union of these)")
    co40 = ShotCorroborator(window_ms=300.0)
    co40.add_shot(10.0)
    co40.add_clicks([Click(t=9.995, level=0.3, floor=0.01)])
    check("a pull seen AND heard is ONE outcome, 'both', stamped with the shot's own time",
          co40.take_outcomes() == [("both", 10.0)])
    co40.add_clicks([Click(t=20.0, level=0.3, floor=0.01)])
    co40.add_shot(20.004)
    out40 = co40.take_outcomes()
    check("...the same when the click was heard first", out40 == [("both", 20.004)], str(out40))
    co40.add_shot(30.0)
    co40.add_clicks([Click(t=40.0, level=0.3, floor=0.01)])
    co40.resolve(41.0)
    out40 = co40.take_outcomes()
    check("a detection nobody heard and a click nobody saw each resolve to their own outcome",
          sorted(out40) == [("heard-not-seen", 40.0), ("seen-not-heard", 30.0)], str(out40))
    check("draining empties it, and reset() clears anything left",
          co40.take_outcomes() == [] and (co40.add_shot(1.0), co40.resolve(9.0), co40.reset(),
                                          co40.take_outcomes())[-1] == [])
    for i40 in range(600):
        co40.add_shot(100.0 + i40)
    co40.resolve(10_000.0)
    check("nobody has to drain it: it is bounded", len(co40.outcomes) <= 256)

    # ---------------------------------------------------------------------- 41 -
    print("\n41. the cover is reported once per transition, named, and the suppression is unchanged")
    import types as _types41
    from lsa.tools.camera_lab import CameraLab as _CL41
    lines41: list = []
    win41 = _types41.SimpleNamespace(cover_reason="", cover_by="", raw=False)
    win41.in_front_of_camera = lambda: win41.raw
    lab41 = _types41.SimpleNamespace(
        target_window=win41, _target_ever_opened=True, _cover_closed_noted=False,
        _cover_checked_at=0.0, COVER_CACHE_S=0.0, _cover_true_since=None, _cover_causes=set(),
        _cover_ok=True, COVER_SETTLE_S=0.0, _cover_suppressed=0, _cover_since=None,
        _cover_reported=("", 0.0), COVER_REPORT_S=_CL41.COVER_REPORT_S, _log=lines41.append,
        _screen_busy=lambda: "", director=_types41.SimpleNamespace(busy_label=lambda: ""),
        _reclaim_screen=lambda why: None)
    check("the repeat interval is the app's own constant, long (>= 30 s), never zero", _CL41.COVER_REPORT_S >= 30.0)
    win41.cover_by = "Wispr Flow's window 'Status'"
    win41.cover_reason = f"{win41.cover_by} is in front of the target screen: only 0 of 5 sampled points still belong to it"
    res41 = [_CL41._camera_sees_our_screen(lab41) for _ in range(50)]
    ign41 = [l_ for l_ in lines41 if "IGNORING" in l_ or "still ignoring" in l_]
    check("while covered every check says NO (the suppression is unchanged) ...", not any(res41))
    check("...and the log says so ONCE for 50 consecutive checks, naming the window (not once per tick)",
          len(ign41) == 1 and "Wispr Flow's window 'Status'" in ign41[0], str(lines41))
    win41.cover_by = "Zoom's window 'Meeting'"
    win41.cover_reason = f"{win41.cover_by} is in front of the target screen: only 0 of 5 sampled points still belong to it"
    _CL41._camera_sees_our_screen(lab41)
    ch41 = [l_ for l_ in lines41 if "still ignoring" in l_]
    check("it speaks again when WHAT is covering the screen CHANGES", len(ch41) == 1 and "Zoom" in ch41[0], str(lines41))
    for _ in range(20):
        _CL41._camera_sees_our_screen(lab41)
    check("...and not again while that stays the same", len([l_ for l_ in lines41 if "still ignoring" in l_]) == 1)
    lab41._cover_reported = (win41.cover_by, lab41._cover_reported[1] - 200.0)
    _CL41._camera_sees_our_screen(lab41)
    check("...and once more after the long interval, for an unchanged cover",
          len([l_ for l_ in lines41 if "still ignoring" in l_]) == 2)
    win41.raw = True
    back41 = [_CL41._camera_sees_our_screen(lab41) for _ in range(3)]
    check("when the screen comes back the suppression ends and it says so once",
          back41[-1] is True and len([l_ for l_ in lines41 if "in front of the camera again" in l_]) == 1)
    # the prompt band: shown only after the cover has lasted, named, standing down when it clears
    lab41p = _types41.SimpleNamespace(
        capture=object(), _grid_active=False, _sgrid_active=False,
        target_window=win41, _cover_since=None,
        COVER_BANNER_AFTER_S=2.5, _camera_sees_our_screen=lambda: False)
    import time as _time41
    lab41p._cover_since = _time41.perf_counter()
    try:
        head41 = _CL41._prompt_state(lab41p)[0]
    except Exception as exc41:                       # the rest of the state machine needs a full lab
        head41 = f"(prompt needs more of a lab: {exc41!r})"
    check("a cover that has lasted < 2.5 s shows NO banner (the suppression is immediate; the banner is not)",
          head41 != "SHOTS ARE NOT BEING COUNTED", str(head41))
    lab41p._cover_since = _time41.perf_counter() - 5.0
    win41.cover_by = "Wispr Flow's window 'Status'"
    got41 = _CL41._prompt_state(lab41p)
    check("after that the banner names the window, and keeps its advice",
          got41[0] == "SHOTS ARE NOT BEING COUNTED" and "Wispr Flow's window 'Status' is in front of this screen" in got41[1]
          and "Bring it back to the front" in got41[1], str(got41))
    win41.cover_by = ""
    got41 = _CL41._prompt_state(lab41p)
    check("an unreadable cover falls back to 'Something' (no name invented)",
          got41[1].startswith("Something is in front of this screen"), str(got41))

    # ---------------------------------------------------------------------- 42 -
    print("\n42. whatever covers the screen gives it back (2026-10-20: the cover came down and nothing restored the target)")
    import lsa.tools.camera_lab as _clmod42
    from lsa.tools.camera_lab import CameraLab as _CL42
    fronted42: list = []
    real_ftf42 = _clmod42.force_to_front

    class _Win42:
        cover_reason = "BitTorrent's window 'BitTorrent 7.9.5' is in front of the target screen"
        cover_by = "BitTorrent's window 'BitTorrent 7.9.5'"
        front = False
        refuse = False

        def in_front_of_camera(self):
            return self.front

    def _fake_ftf42(win):
        fronted42.append(win)
        if not win.refuse:
            win.front = True
        return win.front

    def _lab42(win):
        lines: list = []
        lab = _types41.SimpleNamespace(
            target_window=win, _target_ever_opened=True, _reclaim_at=-1e9, _reclaim_tries=0,
            _reclaim_refused_noted=False, RECLAIM_MIN_GAP_S=0.0, RECLAIM_MAX_TRIES=_CL42.RECLAIM_MAX_TRIES,
            _log=lines.append, _flash=None)
        lab._reclaim_screen = lambda why: _CL42._reclaim_screen(lab, why)
        lab.lines = lines
        return lab

    class _Flash42:
        hidden = False
        deleted = False

        def hide(self):
            self.hidden = True

        def deleteLater(self):
            self.deleted = True

    _clmod42.force_to_front = _fake_ftf42
    try:
        # the cover comes down while the target is NOT in front -> it is put back
        w42 = _Win42()
        l42 = _lab42(w42)
        l42._flash = _Flash42()
        _CL42._close_flash(l42)
        check("the cover coming down puts the target screen back in front",
              fronted42 == [w42] and w42.front and l42._flash is None,
              f"force_to_front called {len(fronted42)}x, in front {w42.front}")
        check("  and it says so in the log", any("put back in front" in x for x in l42.lines), str(l42.lines))
        # already in front -> nothing is grabbed (no needless fight)
        fronted42.clear()
        _CL42._close_flash(l42)                      # no flash left: nothing to give back
        l42._flash = _Flash42()
        _CL42._close_flash(l42)
        check("a target already in front is left alone", fronted42 == [], f"{len(fronted42)} calls")
        # Escape on the cover is HIM leaving: do not drag the target over where he goes
        w42b = _Win42()
        l42b = _lab42(w42b)
        l42b._flash = _Flash42()
        _CL42._close_flash(l42b, reclaim=False)
        check("closing the cover with reclaim=False (the user pressed Escape) does not grab the screen",
              not w42b.front and l42b._flash is None)
        calls42: list = []
        l42c = _types41.SimpleNamespace(
            _close_flash=lambda reclaim=True: calls42.append(reclaim),
            _still_guard=_types41.SimpleNamespace(stop=lambda: None), _still_tag="x",
            _log=lambda m: None, _calibrating=True, _set_status=lambda *a, **k: None)
        _CL42._on_flash_cancelled(l42c, "escape")
        _CL42._on_flash_cancelled(l42c, "timeout")
        check("the real handler: Escape does NOT reclaim, the app's own timeout DOES",
              calls42 == [False, True], str(calls42))
        # Windows refuses: said once per episode, bounded, with what is in front
        fronted42.clear()
        w42d = _Win42()
        w42d.refuse = True
        l42d = _lab42(w42d)
        results42 = [l42d._reclaim_screen("test") for _ in range(10)]
        check("a refusing Windows is retried a bounded number of times, never forever",
              len(fronted42) == _CL42.RECLAIM_MAX_TRIES and not any(results42),
              f"{len(fronted42)} attempts of {_CL42.RECLAIM_MAX_TRIES} allowed")
        refused42 = [x for x in l42d.lines if "could NOT put the target screen back" in x]
        check("  and the refusal is logged ONCE, naming what is in front",
              len(refused42) == 1 and "BitTorrent" in refused42[0], str(l42d.lines))
        w42d.refuse = False
        w42d.front = True
        l42d._reclaim_screen("test")
        w42d.front = False
        l42d._reclaim_tries = 0
        check("  and a new episode (it was in front, then lost) may be refused-and-reported again",
              l42d._reclaim_refused_noted is False)
        # the minimum gap between attempts is a real number
        w42e = _Win42()
        w42e.refuse = True
        l42e = _lab42(w42e)
        l42e.RECLAIM_MIN_GAP_S = _CL42.RECLAIM_MIN_GAP_S
        fronted42.clear()
        for _ in range(5):
            l42e._reclaim_screen("test")
        check("attempts are spaced by RECLAIM_MIN_GAP_S (five calls in one instant = one attempt)",
              len(fronted42) == 1 and _CL42.RECLAIM_MIN_GAP_S >= 1.0, f"{len(fronted42)} attempts")
        # no target window / never opened: nothing to give back, nothing raised
        l42f = _lab42(None)
        check("with no target screen there is nothing to give back (and nothing raises)",
              l42f._reclaim_screen("test") is True)
        # the splash teardown
        spl_calls: list = []

        class _Splash42:
            def retire(self, why):
                spl_calls.append(why)

            def dispose(self):
                pass

        l42g = _types41.SimpleNamespace(splash=_Splash42(), _reclaim_screen=lambda why: spl_calls.append("reclaim"))
        _CL42._splash_down(l42g, "a measurement is starting")
        check("the splash coming down gives the screen back too", spl_calls == ["a measurement is starting", "reclaim"],
              str(spl_calls))
        # mid-measurement cover -> reclaim; idle / playing cover -> NOT (do not fight him)
        for busy, expect in (("a measurement", 1), ("", 0)):
            rec42: list = []
            w42h = _types41.SimpleNamespace(cover_reason="x is in front", cover_by="x", raw=False)
            w42h.in_front_of_camera = lambda w=w42h: w.raw
            lab42h = _types41.SimpleNamespace(
                target_window=w42h, _target_ever_opened=True, _cover_closed_noted=False,
                _cover_checked_at=0.0, COVER_CACHE_S=0.0, _cover_true_since=None, _cover_causes=set(),
                _cover_ok=True, COVER_SETTLE_S=0.0, _cover_suppressed=0, _cover_since=None,
                _cover_reported=("", 0.0), COVER_REPORT_S=_CL42.COVER_REPORT_S, _log=lambda m: None,
                _screen_busy=lambda b=busy: b, director=_types41.SimpleNamespace(busy_label=lambda: ""),
                _reclaim_screen=lambda why, r=rec42: r.append(why))
            _CL42._camera_sees_our_screen(lab42h)
            _CL42._camera_sees_our_screen(lab42h)
            check(f"a foreign window covering the screen {'WHILE MEASURING' if busy else 'while he is just playing'} "
                  f"{'is reclaimed once' if expect else 'is NOT fought over (he may have alt-tabbed)'}",
                  len(rec42) == expect, f"{len(rec42)} reclaim attempts")
    finally:
        _clmod42.force_to_front = real_ftf42

    print()
    print("43. a suite's own printer cannot crash it, and no suite reads the live desktop")
    # 2026-10-22: `check()` printed `win.cover_reason`, which carried a REAL window
    # title (Claude Code's terminal, led by an animating spinner glyph); the
    # console is cp1252 and the print raised UnicodeEncodeError on some runs.
    import io as _io43
    import re as _re43
    import safe_stdio as _ss43
    _title43 = "✳ LASER-SHOOT-BOOK is in front of the target screen"
    _strict43 = _io43.TextIOWrapper(_io43.BytesIO(), encoding="cp1252", errors="strict")
    try:
        _strict43.write(_title43)
        _strict43.flush()
        _premise43 = False
    except UnicodeEncodeError:
        _premise43 = True
    check("premise: a strict cp1252 stream DOES crash on that title (so the guard is not vacuous)",
          _premise43)
    _safe43_raw = _io43.BytesIO()
    _safe43 = _ss43.harden(_io43.TextIOWrapper(_safe43_raw, encoding="cp1252", errors="strict"))
    _safe43.write("  [FAIL] it names " + _title43)
    _safe43.flush()
    _out43 = _safe43_raw.getvalue().decode("cp1252")
    check("a hardened stream writes it without raising, keeping the line readable",
          "[FAIL] it names" in _out43 and "\\u2733" in _out43 and "LASER-SHOOT-BOOK" in _out43,
          _out43)
    check("this suite's own stdout is hardened",
          getattr(sys.stdout, "errors", "backslashreplace") != "strict",
          repr(getattr(sys.stdout, "errors", None)))
    _printers43 = [p_ for p_ in sorted((Path(__file__).resolve().parent).glob("test_*.py"))
                   if _re43.search(r"def check\(|'PASS' if|\"PASS\" if",
                                   p_.read_text(encoding="utf-8", errors="replace"))]
    _unsafe43 = [p_.name for p_ in _printers43
                 if not _re43.search(r"^import safe_stdio\b", p_.read_text(encoding="utf-8", errors="replace"),
                                     _re43.M)]
    check("every suite with its own printer imports safe_stdio",
          len(_printers43) >= 20 and not _unsafe43,
          f"{len(_printers43)} printers; unprotected: {_unsafe43}")
    # THE OTHER HALF: no suite may ask the operating system which window is where.
    # `winowner.window_at` is the live-desktop query; the only legitimate use in a
    # test is behind a stubbed `_owner_at`, so a bare call is the fault.
    _desk43 = []
    for p_ in sorted(Path(__file__).resolve().parent.glob("*.py")):
        t_ = p_.read_text(encoding="utf-8", errors="replace")
        if p_.name == "test_feedback.py":
            continue
        if _re43.search(r"window_at\(|EnumWindows|GetForegroundWindow|WindowFromPoint", t_):
            _desk43.append(p_.name)
    check("no other test file queries the live desktop (window_at / EnumWindows / WindowFromPoint)",
          not _desk43, str(_desk43))

    print()
    if all(checks):





        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
