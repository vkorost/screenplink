"""Camera Lab - Phase 0 diagnostic tool for the laser shooting app.

This is not the game. It is the instrument that tells us whether the game is
possible on this hardware, and it answers six questions:

  1. Which of the attached cameras is the one pointed at the screen?
  2. Which camera, backend, resolution and codec actually deliver full frame
     rate? (What a webcam reports and what it delivers are different claims.)
  3. Which exposure/gain controls does the driver really honour, in what units?
  4. With the screen underexposed to a ghost, can we see the laser at all?
  5. How many frames does one trigger pull occupy -- i.e. is the LaserLyte
     100 ms pulse figure true for these particular guns?
  6. Out of N shots fired, how many register? That number is the whole project.

Target mode (F11) is the part you use while standing back from the monitor: a
fullscreen aiming field with a large scoreboard, an optional camera pane, and a
click on every hit. It is built from the same Scene/Layer stack the real game
will use, so none of it is throwaway.

Run:  run_camera_lab.bat
  or: .venv\\Scripts\\python.exe src\\lsa\\tools\\camera_lab.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Before cv2, not after: opening a C910 through Media Foundation takes 74 seconds
# without this and 0.4 with it. See lsa/_env.py for the measurements.
import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
from PySide6.QtCore import QPointF, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                           QPixmap, QShortcut)
from PySide6.QtWidgets import (
    QApplication,
    QSlider,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if __package__ in (None, ""):                       # allow running the file directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsa.audio.bang import (CouplingTeacher, SOURCE_BANG, SOURCE_EFFECT,      # noqa: E402
                            fallback_mute_s, play_own_sound)
from lsa.audio.mixer import DRY_SOUND, default_mixer                           # noqa: E402
from lsa.audio.voice import (MODE_LINES, VoiceAnnouncer, mode_events,         # noqa: E402
                             mode_snapshot)
from lsa.camera import controls                                               # noqa: E402
from lsa.camera.capture import (                                              # noqa: E402
    COMMON_MODES,
    CaptureConfig,
    CaptureThread,
    GrouperConfig,
    ModeScanThread,
    Preview,
)
from lsa.camera.devices import available_backends, list_cameras               # noqa: E402
from lsa.detect.laser import DetectorParams, ScoreMode, Shot                  # noqa: E402
from lsa.display.layers import (                                              # noqa: E402
    BulletHoles,
    CalibrationTargets,
    BackgroundLabel,
    CameraPane,
    CameraSpaceHits,
    Alert,
    CalibrationPattern,
    KeyHelp,
    PlayAreaBorder,
    Prompt,
    SignalMeter,
    SyntheticLaser,
    SolidBackground,
    StatsHud,
    VerdictBoard,
    VolumeReadout,
)
from lsa.audio.listener import (OWN_UNPROTECTED, ClickListener,               # noqa: E402
                                ShotCorroborator, default_listener,
                                input_devices, pick_device)
from lsa.selfcheck import MARGIN_READY, SelfCheck, Verdict, assess, camera_not_delivering                          # noqa: E402
from lsa import exposure as expo                                              # noqa: E402
from lsa import playarea                                                     # noqa: E402
from lsa.calibrate.flash import FlashWindow                                   # noqa: E402
import lsa.modes as arcade                                                    # noqa: E402
from lsa.modes import ModeHost, ShotEvent                                     # noqa: E402
from lsa.artwork import catalog as figure_catalog                              # noqa: E402
from lsa.modes.base import describe_nearest                                   # noqa: E402
from lsa.modes.menu import Menu as MenuMode                                   # noqa: E402

# WHAT IS ON SCREEN WHILE THE APP MEASURES ITSELF. Mode 1: a few thin rings on
# black, which is what the screen search, the exposure probe and the self test
# all photograph. Named once rather than spelled in three places, because the
# construction site and the "has he moved on?" test must mean the same mode.
STARTUP_MODE_NAME = "static"
from lsa.shotlog import (audio_tally, describe_missed_pulse,                 # noqa: E402
                         mic_state, missed_pulse_note, shot_evidence)
from lsa.calibrate.align import AutoAligner                                   # noqa: E402
from lsa.calibrate.mapping import (Mapping, Sample, grid_points,              # noqa: E402
                                   judge_held_out, confined_sensitivity_map)
from lsa.calibrate.screen import (ScreenQuad, LIT_WHITE_MAX, display_is_lit,
                                  find_screen, draw_quad)           # noqa: E402
from lsa.calibrate.pattern import (find_marker, marker_points,                # noqa: E402
                                   plausible_quad, prune_out_of_order,
                                   screen_mask)
from lsa.display.layers import (band_height_for, bottom_strip_for,            # noqa: E402
                                score_against_marks)
from lsa.display.scene import Scene                                           # noqa: E402
from lsa.display.session import SessionStats                                  # noqa: E402
from lsa.display.foreground import force_to_front  # noqa: E402
from lsa.display.splash import Splash  # noqa: E402
from lsa.display.window import TargetWindow                                   # noqa: E402
from lsa.settings import AppSettings, normalise_magazine, normalise_players  # noqa: E402
from lsa.bests import BestBook                                                # noqa: E402
from lsa.director import Director, QtClock                                     # noqa: E402
from lsa import routines                                                       # noqa: E402
from lsa.screenwatch import ScreenWatch                                      # noqa: E402
from lsa import watchdog as watch                                              # noqa: E402
from lsa import introspect as introspection                                    # noqa: E402

# THE ROOT IS DEFINED ONCE, in lsa/artwork/assets.py, which is stdlib-only and
# knows whether the app is frozen. This module used to carry its own
# `Path(__file__).resolve().parents[3]`, a second definition that could drift and
# that overshoots by one level inside a PyInstaller bundle. Read-only data (sounds,
# voice) comes from the bundle; everything WRITABLE comes from the folder beside
# the executable. Run from source those are the same folder.
from lsa.artwork import assets as _assets                                      # noqa: E402
USER_ROOT = _assets.user_root()
# Redirectable, for the same reason the log is.
#
# The log was separated by process on 2026-10-15 after a harness line was
# reported to him as a measurement of his bench. THE IMAGES WERE NOT, and
# the harness went on overwriting `recordings/verdict_screen.png`,
# `verdict_camera.png` and `calib_white.png` with pictures from a simulated
# 800x800 camera -- measured at 22:41 and 22:48 tonight, destroying the
# evidence from his last real run.
#
# Those images exist precisely so the app's own verdict can be looked at
# afterwards instead of him being asked what he saw. A simulation that
# overwrites them is worse than one that writes nothing, because the file
# is still there and still looks like evidence.
#
# Half a lesson applied is how every multi-instance bug on this project has
# survived (RULES Part E): the log was fixed, the images were the same
# problem, and nothing went looking for the second instance.
RECORDINGS = Path(os.environ.get("LSA_RECORDINGS")
                  or _assets.user_dir("recordings"))

# THE FILE THAT IS HIS. One path, and redirectable like the other two.
#
# It was inlined as `PROJECT_ROOT / "calibration.json"` in three places,
# which is both a duplication and the reason a test could write it: the
# cold-start harness saved its simulated 800x800 mapping over his real
# 3840x2160 one, taking the five sensitivity regions learned from his
# actual trigger pulls with it, THROUGH A PATH THAT HAD NOT BEEN BLOCKED.
#
# Neutering `Mapping.save` is a patch and a patch can be routed around --
# which is exactly what happened. One redirectable constant means a test
# cannot reach his file at all, which is the difference between a defence
# and a reminder (RULES C11).
# THE FILES HE OWNS LIVE IN config/, not at the top level. He asked for the
# working folder cleared of loose json on 2026-10-18. This one is the
# expensive one: the mapping measured from eighteen photographs AND the
# sensitivity learned from his own trigger pulls, which is what saves the
# right-hand third of his screen. It was copied and verified byte-identical
# before anything was moved, because a missing calibration does not announce
# itself -- the loader returns None and the app quietly recalibrates.
CONFIG_DIR = _assets.user_dir("config")
CALIBRATION_PATH = Path(os.environ.get("LSA_CALIBRATION")
                        or (CONFIG_DIR / "calibration.json"))

# WHERE THE LOG GOES, and it must not be one shared file.
#
# The headless cold-start harness builds a real CameraLab, so it logged to
# the same camera_lab.log as the live app. With both running, a reader --
# human or script -- sees one interleaved stream with no way to tell which
# process wrote which line.
#
# That is not untidiness, it is a corrupted instrument. On 2026-10-15 a
# timing loop watching for "=== VERDICT ===" matched the HARNESS's verdict
# and reported "17 seconds to READY, 12 of 12, 0.7mm" as a measurement of
# the real bench. The giveaways were in the same line and went unread: an
# 800x800 screen and exactly 60.0 fps, which are simulation numbers.
# Measuring the wrong process is worse than not measuring.
# LOGS LIVE IN logs/, NOT IN THE WORKING FOLDER. He made the directory and
# asked for it on 2026-10-18: "clean up the folder from the log files ... so
# all the logs need to move there and be created there". A hundred and seven
# of them had accumulated at the top level of a repo whose README-level
# contents are supposed to be what a new user needs to run the app.
#
# The directory is created where the log is about to be written rather than at
# import: a test redirected through LSA_LOG must not have this one made for it
# somewhere it did not ask for.
LOG_DIR = _assets.user_dir("logs")
LOG_PATH = Path(os.environ.get("LSA_LOG") or (LOG_DIR / "camera_lab.log"))

# THE FOURTH FILE THAT IS HIS, and the only one that had no redirect.
#
# `LSA_LOG`, `LSA_RECORDINGS` and `LSA_CALIBRATION` each let a test be sent
# somewhere that is not his; this one was hard-coded, so the only thing
# standing between a test and `settings.local.json` was two hand-copied lines
# (`lab._save_settings = ...`) pasted into three test files -- and a fourth
# test, written without them, wrote his file (2026-10-16). A neutered writer
# is a patch and a patch can be routed around (RULES C11). With the path
# redirectable, `tests/owned_files.isolate_outputs()` can point every test at
# a COPY, and a test cannot reach his file at all.
SETTINGS_PATH = Path(os.environ.get("LSA_SETTINGS")
                     or (CONFIG_DIR / "settings.local.json"))


class _AssetDir:
    """A class attribute that is an asset folder, looked up when it is READ.

    `voice_dir` is a class attribute so a test can point it at a temporary or a
    missing folder (`CameraLab.voice_dir = ...`); assigning over this descriptor
    does exactly that. Left alone it asks the asset table every time, so the
    override variable and the frozen bundle are honoured at call time, never
    frozen in at import.
    """

    def __init__(self, resolve) -> None:
        self._resolve = resolve

    def __get__(self, obj, owner=None):
        return self._resolve()


# --- small widgets -----------------------------------------------------------

# The alert raised when two players were asked for and the drill cannot do it.
ONE_PLAYER_ONLY_TITLE = "ONE PLAYER ONLY"
ONE_PLAYER_ONLY_SECONDS = 1.0


class ControlRow(QWidget):
    """One camera control: slider + spin box + what the camera said back.

    The readback column is the point of this widget. OpenCV will cheerfully
    accept a value the driver ignores, so the only trustworthy number is the
    one we read after writing.
    """

    changed = Signal(str, float)

    def __init__(self, spec: controls.ControlSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._steps = max(1, int(round((spec.hi - spec.lo) / spec.step)))

        from PySide6.QtWidgets import QSlider

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setMinimumWidth(100)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(spec.lo, spec.hi)
        self.spin.setSingleStep(spec.step)
        self.spin.setDecimals(spec.decimals)
        self.spin.setFixedWidth(78)

        self.echo = QLabel("--")
        self.echo.setFixedWidth(60)
        self.echo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.echo.setStyleSheet("color:#8ab4ff;")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        name = QLabel(spec.label)
        name.setFixedWidth(116)
        lay.addWidget(name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        lay.addWidget(self.echo)

        if spec.hint:
            self.setToolTip(spec.hint)
            name.setToolTip(spec.hint)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._emit)

        self._guard = False
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _to_steps(self, value: float) -> int:
        return int(round((value - self.spec.lo) / self.spec.step))

    def _from_steps(self, steps: int) -> float:
        return self.spec.lo + steps * self.spec.step

    def _from_slider(self, steps: int) -> None:
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(self._from_steps(steps))
        self._guard = False
        self._debounce.start()

    def _from_spin(self, value: float) -> None:
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(self._to_steps(value))
        self._guard = False
        self._debounce.start()

    def _emit(self) -> None:
        self.changed.emit(self.spec.key, float(self.spin.value()))

    def set_value_silently(self, value: "float | None") -> None:
        # None is a real state, not a caller mistake: it means "this has never
        # been measured on this camera", which is exactly what makes the app
        # run a full exposure sweep instead of trusting a remembered number.
        # `_calibrate_if_needed` already tests `if saved is not None`, so the
        # value legitimately arrives here as None -- and the widget crashed on
        # it before reaching any of that, taking the whole app down at startup.
        if value is None:
            self._guard = False
            return
        self._guard = True
        self.spin.setValue(max(self.spec.lo, min(self.spec.hi, value)))
        self.slider.setValue(self._to_steps(value))
        self._guard = False

    def set_echo(self, value: float, accepted: bool) -> None:
        self.echo.setText(f"{value:g}")
        self.echo.setStyleSheet("color:#8ab4ff;" if accepted else "color:#b8a850;")
        self.echo.setToolTip("" if accepted else "driver returned a different value than requested")

    def mark_probe(self, res: controls.ProbeResult) -> None:
        if res.writable:
            self.setStyleSheet("")
            self.echo.setToolTip("writable")
        else:
            self.setStyleSheet("color:#777;")
            self.echo.setToolTip("driver ignored a write to this control")


class IdentifyThread(QThread):
    """Grabs one frame from every candidate camera so you can see which is which."""

    shot = Signal(int, str, object)      # index, name, QImage | None
    done = Signal()

    def __init__(self, devices: list[tuple[int, str]], backend: str, parent=None) -> None:
        super().__init__(parent)
        self.devices = devices
        self.backend = backend

    def run(self) -> None:
        api = available_backends().get(self.backend, cv2.CAP_ANY)
        for index, name in self.devices:
            image = None
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                for _ in range(6):               # let auto-exposure settle
                    ok, frame = cap.read()
                if ok and frame is not None:
                    frame = cv2.resize(frame, (384, int(384 * frame.shape[0] / frame.shape[1])))
                    image = QImage(frame.data, frame.shape[1], frame.shape[0],
                                   frame.strides[0], QImage.Format_BGR888).copy()
            cap.release()
            self.shot.emit(index, name, image)
        self.done.emit()


class IdentifyDialog(QDialog):
    """Side-by-side stills from each camera: the quickest way to pick the right one."""

    def __init__(self, devices: list[tuple[int, str]], backend: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Which camera sees the screen?")
        self.resize(880, 460)
        self._lay = QHBoxLayout()
        self.status = QLabel("grabbing a frame from each camera...")

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(
            "<b>Pick the camera showing your monitor.</b> Its index is in the caption. "
            "Point it at the screen first if you have not already."
        ))
        outer.addLayout(self._lay, 1)
        outer.addWidget(self.status)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        outer.addWidget(btn)

        self.thread = IdentifyThread(devices, backend, self)
        self.thread.shot.connect(self._add)
        self.thread.done.connect(lambda: self.status.setText("done"))
        self.thread.start()

    def _add(self, index: int, name: str, image) -> None:
        box = QVBoxLayout()
        pic = QLabel()
        pic.setFixedSize(384, 240)
        pic.setAlignment(Qt.AlignCenter)
        pic.setStyleSheet("background:#111; border:1px solid #444; color:#888;")
        if image is None:
            pic.setText("no frame")
        else:
            pic.setPixmap(QPixmap.fromImage(image).scaled(
                pic.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        cap = QLabel(f"<b>index {index}</b><br>{name}")
        cap.setAlignment(Qt.AlignCenter)
        box.addWidget(pic)
        box.addWidget(cap)
        w = QWidget()
        w.setLayout(box)
        self._lay.addWidget(w)

    def closeEvent(self, ev) -> None:
        self.thread.wait(4000)
        super().closeEvent(ev)


class ModeScanDialog(QDialog):
    def __init__(self, index: int, backend: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Format scan - camera {index} on {backend}")
        self.resize(660, 520)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["W", "H", "Codec", "Asked", "Reported", "MEASURED"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.bar = QProgressBar()
        self.status = QLabel("Opening the camera once per format. This takes a while.")
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "<b>MEASURED</b> is the only column that matters: frames actually delivered "
            "per second. A camera that reports 30 and measures 5 is lying to you."
        ))
        lay.addWidget(self.table, 1)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.close_btn)

        self.thread = ModeScanThread(index, backend, parent=self)
        self.thread.progress.connect(self._progress)
        self.thread.result.connect(self._row)
        self.thread.finished_scan.connect(self._done)
        self.thread.start()

    def _progress(self, n: int, total: int, label: str) -> None:
        self.bar.setMaximum(total)
        self.bar.setValue(n)
        self.status.setText(f"testing {label}  ({n}/{total})")

    def _row(self, r) -> None:
        if not r.ok:
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        vals = [str(r.width), str(r.height), r.fourcc, f"{r.requested_fps:g}",
                f"{r.actual_fps:g}", f"{r.measured_fps:.1f}"]
        for col, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if col == 5:
                good = r.measured_fps >= r.requested_fps * 0.9
                item.setForeground(QColor("#5fd35f") if good else QColor("#b8a850"))
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.table.setItem(row, col, item)
        if r.note:
            self.table.item(row, 2).setToolTip(r.note)

    def _done(self) -> None:
        self.status.setText(f"done - {self.table.rowCount()} working formats")
        self.bar.setValue(self.bar.maximum())

    def reject(self) -> None:
        self.thread.abort()
        self.thread.wait(3000)
        super().reject()

    def accept(self) -> None:
        self.thread.abort()
        self.thread.wait(3000)
        super().accept()


# --- main window -------------------------------------------------------------

class _DirectorSink:
    """Everything the Director is allowed to do to the world.

    Deliberately small and deliberately the only route. A routine cannot
    reach the capture thread or a QTimer itself -- `tests/test_rules.py`
    fails the build if `lsa/routines.py` tries -- so there is one place
    where the camera is touched and one place where the screen is forced to
    repaint, instead of the fifteen `processEvents()` calls and the direct
    `capture.*` calls that were scattered through the routines before.
    """

    def __init__(self, lab: "CameraLab") -> None:
        self._lab = lab

    # -- camera ----------------------------------------------------------

    def request_still(self, tag: str, frames: int = 8) -> None:
        cap = self._lab.capture
        if cap is None:
            raise RuntimeError("no camera")
        cap.request_still(tag, frames)

    def start_noise_calibration(self, seconds: "float | None" = None) -> None:
        cap = self._lab.capture
        if cap is None:
            raise RuntimeError("no camera")
        # Covered FIRST, every time, because the floor sets the detection
        # threshold for the session and was once measured against a lit
        # desktop: peak 15 against a quiet one, peak 90 against a window
        # full of coloured text, and a min_peak of 105 rejects every shot
        # on the far side of this screen outright.
        self._lab._cover_desktop_for_noise()
        if seconds is None:
            cap.start_noise_calibration()
        else:
            cap.start_noise_calibration(seconds)

    def start_room_measurement(self, seconds: float = 1.5) -> None:
        cap = self._lab.capture
        if cap is None:
            raise RuntimeError("no camera")
        cap.start_room_measurement(seconds)

    def set_control(self, key: str, value: float) -> None:
        cap = self._lab.capture
        if cap is None:
            raise RuntimeError("no camera")
        cap.set_control(key, value)

    def written_value(self, key: str) -> "float | None":
        """The ONE control cache: what the capture was last told.

        The Director kept its own copy and it went stale the moment a
        preset was written around it -- which these routines do, via
        `Call(lab._apply_preset(...))`. A stale cache makes the next
        identical write be SKIPPED while the camera sits elsewhere,
        which is worse than writing twice on a device where writes are
        the hazard.
        """
        cap = self._lab.capture
        return None if cap is None else cap.written_value(key)

    # -- screen ----------------------------------------------------------

    def request_paint(self, frames: int = 2) -> None:
        """Force a repaint and report when it has actually happened.

        `TargetWindow.painted` fires after `paintEvent` completes, so this
        waits for pixels rather than for a guessed number of milliseconds.
        One `processEvents()` does not paint a fullscreen 4K window, which
        is how a calibration marker came to be photographed before it was
        drawn -- strength 14 where its siblings read 197-231 -- and how an
        alignment check measured a 109px shift against a half-painted
        window and correctly refused a mapping that was fine.
        """
        win = self._lab.target_window
        if win is None:
            raise RuntimeError("no target window")
        remaining = {"n": max(1, int(frames))}

        def one_frame() -> None:
            remaining["n"] -= 1
            if remaining["n"] > 0:
                win.update()
                return
            try:
                win.painted.disconnect(one_frame)
            except Exception:
                pass
            self._lab.director.deliver_painted()

        win.painted.connect(one_frame)
        self._lab.scene.mark_dirty()
        win.update()


def pose_fact_note(name: str, value) -> str:
    """Describe one carried pose fact for the log.

    Generic on purpose. The first version hard-coded the two facts it knew
    about -- `len(value)` for the sensitivity, `float(value)` for the dot
    radius -- and the moment a THIRD joined `POSE_FACTS`, carrying it raised
    a TypeError from outside the try that guards the assignment. That list
    exists so a new fact is covered without anyone remembering, and a report
    that knows each fact by name defeats it (RULES E2).
    """
    if name == "sensitivity":
        return f"{len(value)} sensitivity regions"
    if name == "dot_radius":
        return f"dot radius {float(value):.0f}px"
    return str(name)


class CameraLab(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # The product is ScreenPlink; "Camera Lab" is this window, the operator
        # console. KEEP THE WORDS "Camera Lab" IN IT: `scripts/stop_lab.ps1`
        # enumerates top-level windows and posts WM_CLOSE to the one matching
        # `*Camera Lab*`. If that match fails the script falls back to a force
        # kill, and force-killing this app leaves the camera wedged until it is
        # physically replugged.
        self.setWindowTitle("ScreenPlink - Camera Lab")
        self.resize(1760, 1000)

        self.capture: CaptureThread | None = None
        self.target_window: TargetWindow | None = None
        # THE STARTUP SPLASH, or None. Declared FIRST: the screen-owner
        # flags below are properties that consult it the moment they are
        # assigned (`_install_owner_hooks`), and `__init__` assigns them.
        self.splash: Splash | None = None
        # When the splash went up, so the welcome line can finish before
        # it comes down. 0.0 means it never went up.
        self._splash_up_at: float = 0.0
        # The in-process introspection server, or None. OFF unless
        # `--introspect` or LSA_INTROSPECT asked for it, and `main()` is the
        # only thing that installs one. Declared here and assigned
        # unconditionally, never read through getattr with a default
        # (RULES C8).
        self.introspect = None
        # Has the menu been raised after setup? Declared here and assigned
        # unconditionally, never read through a getattr default: three times
        # on this project a plausible-but-wrong attribute name turned a
        # feature off silently, and every one was hidden by a default (C8).
        self._menu_shown_at_startup = False
        # The last startup verdict. Declared here because it was NOT: it was
        # only ever assigned inside `_publish_verdict`, so every reader before
        # the first verdict depended on a getattr default. Found by
        # `tests/test_introspect.py` section 12, which extracts every lab
        # attribute the introspection server reads and checks it exists on a
        # real CameraLab -- the same fault shape as `stats.last_shot_at`
        # returning 0.0 for days while a guard never fired (RULES C8).
        self._last_report = None
        self.auto_record = False      # set from the command line by main()
        self.auto_target = False
        # Start in the camera-placement view rather than running setup. See
        # _enter_aim_on_start.
        self.auto_aim = False
        # Calibrate by photographing the screen's own markers, with no shooting.
        self.auto_patcal = False
        # Repeat the pattern calibration N times IN ONE PROCESS. Verifying
        # repeatability by restarting the app is self-defeating: every start
        # writes camera controls, and control writes are the one operation
        # measured to hang this camera. The harness was destroying the
        # instrument it was measuring with.
        self.patcal_repeats = 1
        self._patcal_done = 0
        self._patcal_results: list[str] = []
        self.screen_quad: ScreenQuad | None = None
        self.homography = None
        self._flash: FlashWindow | None = None
        # Every still request is a promise the camera may not keep. If it does
        # not arrive, this fires and the step is failed deliberately rather
        # than waiting forever behind a fullscreen window.
        self._still_guard = QTimer(self)
        self._still_guard.setSingleShot(True)
        self._still_guard.timeout.connect(self._still_timed_out)
        self._still_tag = ""
        self._calib_black = None
        # Measured from shots, and always preferred over the flash outline.
        # Reloaded from disk because it is the only mapping worth trusting and
        # asking for nine shots on every launch would be a tax, not a feature.
        self.mapping: Mapping | None = Mapping.load(CALIBRATION_PATH)
        # Grab the measured falloff NOW, at load, before anything can discard
        # it. `calibrate_each_session` sets `self.mapping` to None during
        # setup, which happens BEFORE pattern calibration starts -- so reading
        # it when calibration begins, as the first version did, always found
        # nothing. The mapping is replaced every session by design; the
        # falloff is not, because the camera has not moved.
        self._prev_pose: dict[str, object] = {}
        self._capture_pose_facts()
        self._grid_points: list[tuple[float, float]] = []
        self._grid_samples: list[Sample] = []
        self.aligner = AutoAligner()
        # OFF by default. It was applying 13-45px shifts at correlations of
        # 0.55-0.60 on top of a mapping that had just been measured to 0.6mm --
        # moving a good answer around based on a weak match. It is for drift
        # over a long session, not for second-guessing a fresh measurement, and
        # it earns its place back only by being shown to help.
        self._auto_align = False
        # True only once the user picks a camera from the combo themselves.
        self._device_was_chosen = False
        # While true the console preview carries aiming guides: a target
        # zone, the detected screen, and which way to move the camera.
        self._aim_help = False
        self._align_pending = False
        self._align_passes = 0
        self._align_applied = 0
        self._grid_shots: list[tuple[float, float, float]] = []
        self._grid_repeats = 3
        # Generous, and a SLIDING window below. Noise counts toward this cap
        # too, so a tight limit means a few stray detections can exhaust the
        # budget before three real shots have had a chance to agree.
        self._grid_max_shots = 30
        self._grid_window = 12
        self._grid_ready_at = 0.0
        self._noise_peak = 0.0
        # Nothing measured against THIS room yet. Until it has been, a
        # detection is noise against a stale threshold -- see
        # `_ready_for_real_shots`.
        self._floor_measured = False
        # A measurement is in flight. See `_start_noise_floor`.
        self._floor_in_flight = False
        self._grid_saved_levels: tuple[int, int] | None = None
        self._grid_active = False
        # -- the brightness grid (key U), `start_sensitivity_grid` -------------
        # Declared here and read directly everywhere: a getattr default turned
        # a wrong name into a silently disabled guard three times (RULES C8).
        # `_sgrid_active` is also a SCREEN_OWNERS flag, so it becomes a
        # property at the bottom of this module; assigning it here is correct.
        self._sgrid_active = False
        self._sgrid_points: list[tuple[float, float]] = []
        self._sgrid_cols = 3
        self._sgrid_rows = 3
        self._sgrid_samples: list = []        # one Sample per spot that agreed
        self._sgrid_shots: list[tuple[float, float, float]] = []
        self._sgrid_last_seen = 0.0
        self._sgrid_ready_at = 0.0
        # The gain map that was in force when the run began and was set aside
        # (peaks measured THROUGH a map are in the map's units, not the
        # laser's), to be put back on every exit that does not replace it.
        self._sgrid_prior_gain = None
        # The last map handed to `_set_gain_map`, which is the ONE funnel every
        # map goes through, so this cannot disagree with the detector's.
        self._gain_in_force = None
        # Which corner the MENU button was last placed in, and whether that
        # placement has been said in the log. DECLARED, never read through a
        # getattr default: three guards on this project have been silently
        # disabled by a default standing in for a name nobody checked.
        self._menu_corner_said = False
        # DARK and FIXED first. The search subtracts a black frame from a white
        # one, and auto exposure compensates for precisely that change, which
        # collapses the difference it is trying to measure. Measured on one
        # scene: auto gave a flash difference of 14.9 at confidence 0.72, while
        # -10 gave 44.3 at confidence 1.00. Auto is kept last, as a fallback for
        # a camera that will not honour manual exposure at all.
        self._calib_exposures = [-10.0, -9.0, -11.0, None]
        self._calib_attempt = 0
        self._calibrating = False
        # --- calibration from the screen itself (no shooting) ---------------
        self._pat_points: list[tuple[float, float]] = []
        self._pat_check: list[tuple[float, float]] = []
        self._pat_samples: list[Sample] = []
        self._pat_checks: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._pat_index = 0
        self._pat_lit = None
        self._pat_active = False
        self._pat_misses: list[tuple[float, float]] = []
        self._pat_mask = None          # where the monitor is, in camera pixels
        # -1, not 0: at 0 the comparison against the marker index denied the
        # FIRST marker its one retry, and marker 1 is a corner the prediction
        # for everything else is built on.
        self._pat_retried = -1         # markers re-photographed after a miss
        self._pat_quad_warned = False  # the corner sanity check reports once
        self._pat_hidden: dict = {}    # layer visibility to restore afterwards
        self._pat_mask_retried = False  # corners re-measured without the mask
        self._noise_retries = 0        # noise measurements spoiled by a shot
        self._last_shot_at = 0.0       # perf_counter of the last detection
        # IS THE CAMERA LOOKING AT OUR SCREEN? Declared and assigned here,
        # never read through getattr with a default: that returned 0.0 for
        # days once and the guard never fired while looking like protection
        # (RULES C8). Starts "not ours", because before the target window
        # exists the camera is watching his desktop.
        self._cover_ok = False
        self._cover_checked_at = 0.0
        self._cover_since = None          # when the current cover began
        # Readings the round's honesty mark is built from (modes/efficiency.py).
        # All are state the app already holds; none is new instrumentation.
        self._cover_active = False        # detections are being ignored right now
        self._moved_findings = 0          # times the app concluded the camera moved
        self._last_margin: "float | None" = None   # latest signal margin it measured
        self._cover_reported = ("", 0.0)  # (who, when) of the last cover line
        self._cover_suppressed = 0     # detections ignored while it was not
        self._not_pull_ignored = 0     # detections the mode was never shown
        # Reclaiming the screen (see `_reclaim_screen`).
        self._reclaim_at = -1e9
        self._reclaim_tries = 0
        self._reclaim_refused_noted = False
        # A target window has been created at some point this session. Once
        # one has, "no window" means it was CLOSED and the camera is looking
        # at the desktop; before one ever has, it is the operator-console
        # flow (--no-target) and nothing is suppressed.
        self._target_ever_opened = False
        self._cover_true_since: float | None = None   # start of the current run of True answers
        # WHY detections were ignored since the last time they counted:
        # a subset of {"closed", "covered", "settling"}. Reported when they
        # count again, so the resume line says what was true instead of
        # assuming "the desktop" (RULES C3).
        self._cover_causes: set[str] = set()
        self._cover_closed_noted = False   # the close has been logged once
        self._room_pending = False     # a zero-write room measurement is out
        self._room_why = ""            # 'setup' or 'drift'
        self._exposure_gate_done = False  # the room has been consulted once
        self._room_warned = ""         # 'bright'/'dark', said once each
        self._room_unmeasured = False  # the gate ran but got no frames
        # Real laser shots, (camera_x, camera_y, peak), for learning how the
        # dot's strength falls off across the screen. The markers cannot
        # measure that -- see `_learn_sensitivity`.
        self._sens_shots: list[tuple[float, float, float]] = []
        self._sens_applied = 0         # shot count at the last fit; 0 = never
        # Detections the learner refused to learn from, by reason. Declared
        # here and read directly (RULES C8). See `_sens_refusal`.
        self._sens_ignored: dict[str, int] = {"faint": 0, "animating": 0}
        # Faint detections ADMITTED because a trigger click was heard just
        # before them (never reset, like the counters above), and how many of
        # the samples now in `_sens_shots` are such (reset with that list).
        # `_sens_last_by_click` is `_sens_refusal`'s side channel to its one
        # caller: did the last None it returned rest on the microphone?
        self._sens_admitted = 0
        self._sens_by_click = 0
        self._sens_last_by_click = False
        # The play area's own evidence, in whatever units the detector is
        # reporting right now. Retired by `_set_gain_map` (RULES C8: declared
        # here, never read through getattr with a default).
        self._area_shots: list[tuple[float, float, float]] = []
        self._said_once: set[str] = set()   # keys logged once per session
        # Shots detected while there is no mapping, so nothing could be drawn.
        # Declared here, not reached for with getattr: `stats.last_shot_at`
        # returned 0.0 through getattr for days and a guard never fired once
        # while looking exactly like protection (RULES C8).
        self._unmapped_shots: int = 0
        # The camera preview is stood down once per session when the app
        # becomes shootable; after that it belongs to the operator.
        self._camera_pane_stood_down = False
        # THE INFORMATION GROUP (key 0). False = clean screen: holes and
        # targets, and the prompt band only while something is wrong.
        self._info_full = False
        self._info_last_set: dict[str, bool] = {}
        self._info_hidden_by_policy: set[str] = set()
        self._gain_pending = None      # a map awaiting its noise-floor verdict
        # A map waiting for its BASELINE floor -- the reading taken with
        # no map on, in the same conditions, so the verdict compares like
        # with like. Declared, not conjured with getattr (RULES C8).
        self._gain_awaiting_baseline = None
        # Checking a saved mapping with two markers instead of remeasuring
        # with eighteen. See `start_mapping_check`.
        #
        # `_check_active`, `_probe_active` and `_selftest_active` USED to be
        # assigned here. They are read-only properties now, answered by the
        # Director, so these lines would raise AttributeError -- which is
        # the point. A routine's "am I running" cannot be set by hand from
        # somewhere else, so it cannot disagree with reality, and a
        # shutdown path cannot stand a routine down by clearing a flag it
        # happens to remember the name of.
        #
        # One attempt per session: if the check says the camera moved, the
        # answer is to measure, not to check again.
        self._check_tried = False

        # The screen-draws-a-red-dot self test. See `start_self_test`.
        self._selftest_results: list = []
        self._selftest_seen = None
        # Declared and assigned unconditionally rather than reached for with
        # `getattr(self, "_floor_retries", 0)`. That construct is how
        # `stats.last_shot_at` returned 0.0 through a default for days while
        # a guard never fired once and looked exactly like protection
        # (RULES C8).
        self._floor_retries = 0
        self._probe_results: list = []
        # The anchor dot's reading at its ranged brightness, None until a probe
        # has actually taken one (routines.exposure_probe).
        self._probe_dot_reading: float | None = None
        # Setup finishes once per LAUNCH, but `_on_noise_measured` runs on
        # every accepted floor -- five times on his bench 2026-10-18. Each
        # pass used to reset the session and clear the holes layer, so his
        # bullet holes and score were wiped mid-string by a measurement.
        self._setup_completed_once = False
        self._probe_floor_at: tuple | None = None   # where the probe's floor peaked
        # Where the exposure probe drew its dot this launch (screen fractions),
        # so the mark watchdog's self-check measures the same place.
        self._probe_spot_used: "tuple[float, float] | None" = None
        # Heard-and-nothing-drawn watchdog (lsa/watchdog.py). `voice` is the
        # announcer, hung on by the merge; None means it is not there and the
        # on-screen line is all there is.
        self.watchdog = watch.MarkWatchdog(log=lambda s: self._log(s))
        # Passive watch on our own screen as the camera sees it
        # (lsa/screenwatch.py). Armed by a usable verdict, fed from the HUD tick.
        self.screenwatch = ScreenWatch(log=lambda s: self._log(s))
        self._screenwatch_last_at = 0.0
        self._camera_stalled = False
        self.voice = None
        self._grid_attempt = 0
        self._modetest_attempt = 0
        # How many camera frames this app has EVER received. Deliberately
        # not read from `Stats.frames`, which `reset_stats()` zeroes --
        # and `_on_noise_measured` calls `reset_stats()`, so a count taken
        # from there reads 0 in the middle of a perfectly real launch. It
        # answers exactly one question, "did a camera deliver anything at
        # all", and that is what decides whether a GATE line may claim
        # `role=live`. A number answering that must not be resettable by
        # the thing it reports on.
        self._frames_ever = 0
        # What to start once the next noise floor is accepted. Replaces
        # a guessed 2500ms delay that put the exposure probe inside the
        # floor measurement on every launch with his real timings.
        self._probe_after_floor = None
        # True while the startup noise floor has been deliberately
        # skipped because the exposure probe will measure one. Cleared
        # by the fallback, which is what bounds the bet.
        self._floor_deferred = False
        # Devices shown BY MEASUREMENT not to see this screen, and whether
        # the connected one is among them. Declared here, not reached for
        # with getattr: a plausible attribute name read through a default
        # returned 0.0 for days while a guard never fired once (C8).
        self._blind_devices: set[int] = set()
        # One attempt at opening the target screen for the two-marker
        # check, so a failure to open cannot loop (RULES D10).
        self._check_opened_screen = False
        self._blind = False
        # Did the screen search's WHITE frame never get bright? Then the display is
        # not lit, which is NOT the camera's fault (see `_report_display_asleep`).
        self._display_dark = False
        # Walking every arcade mode is a DEVELOPMENT check, not something
        # the shooter needs before firing. It costs about twenty seconds,
        # it runs AFTER the verdict has already said READY, and it leaves
        # whichever mode it tested last on screen -- which is how he ended
        # up shooting at Escalating Ball without choosing it.
        #
        # "No one will have any patience for the app that takes that long
        #  to start and initialize." Off by default; --modetest turns it on.
        self._selftest_modes = False
        # Set from --trust-cal. Declared here so it is a real attribute
        # rather than something read hopefully through getattr, which has
        # cost this project twice.
        self.trust_calibration = False
        self._modetest_queue: list = []
        self._modetest_results: list = []
        self._modetest_mode = None
        self._modetest_aim = (0.0, 0.0)
        # How many times the walk-through has waited for a mode that has
        # nothing to aim at YET (a pop-up figure is blank for 1.2-3.5 s).
        self._modetest_wait_attempts = 0
        # Shots must keep routing to the harness during the mode phase;
        # _selftest_active is already False by then, so without this the
        # camera's detections went down the normal path and the harness
        # reported 'NOT DETECTED' for dots the log shows it detecting.
        self._modetest_active = False
        self._modetest_hits_before = 0
        self._modetest_shots_before = 0
        self.auto_selftest = False
        self._floor_before_gain = 0.0  # the floor it has to beat
        # True once a mapping has been MEASURED in this run, as opposed to
        # loaded from calibration.json. `calibrate_each_session` exists to
        # distrust the saved one; it must never discard a fresh one.
        self._mapping_measured_now = False
        # Whether the mapping is trusted because the TWO-MARKER CHECK
        # confirmed it, as opposed to because it was measured from
        # scratch this session. Two different facts that were sharing
        # one flag, so the log announced the wrong one.
        self._check_confirmed = False
        # --- the microphone channel -----------------------------------------
        # The camera says WHERE a shot landed. It cannot say whether a shot
        # happened, and that is the difference between "the detector is broken"
        # and "you missed the screen". The camera has a microphone on it, a
        # trigger break is a sharp transient, and the measured room floor here
        # is -46 dBFS, so there is plenty of margin. Audio never creates a
        # shot -- it has no position -- it only corroborates and explains.
        # Built through `listener_factory`, beside `audio_factory`: a test can
        # hand the app a listener that opens no input device (silent_audio.py).
        self.listener = self.listener_factory()
        # Click.t is compared against the camera's shot timestamps, which are
        # time.perf_counter: two clocks with different epochs is RULES B4's
        # exact failure, and nothing would raise -- every pairing would just
        # miss. The clock is injectable for tests of the listener on its own;
        # an app must not be handed one.
        assert self.listener.clock_is_perf_counter, \
            "the listener's clock must be time.perf_counter (RULES B4)"
        self.corroborator = ShotCorroborator()
        self._audio_note = ""
        self._audio_note_until = 0.0
        self._note: tuple[str, str, float] | None = None
        self._grid_last_seen = 0.0
        self.rows: dict[str, ControlRow] = {}
        # Widgets that do nothing without a live camera. Collected here so they
        # can be greyed out rather than silently swallowing clicks: a button that
        # looks pressable and then only whispers into a log pane is a bug.
        self._needs_camera: list[QWidget] = []

        self.settings = AppSettings.load(SETTINGS_PATH)
        # The exposure as it was ON DISK at startup. settings.exposure is
        # overwritten the moment a preset is applied -- the screen search runs
        # at -10 by design -- so by the time setup asks "what did we measure
        # last time?", the answer has already been clobbered with -10. That is
        # two stops darker than the value that works, and it produced a whole
        # session in which the dot never bloomed at all.
        self._boot_exposure = self.settings.exposure
        # The value the SHOOTING profile settled on. Only
        # _on_exposure_chosen may change it, so a calibration preset
        # cannot leak into what gets saved.
        self._shooting_exposure = self.settings.exposure
        self.session = SessionStats()
        self.scene, self.layers = self._build_scene()
        # The player-count SETTING, mirrored onto the scene. It is consumed in
        # `_select_mode` ONLY, and only once setup is over
        # (`_menu_shown_at_startup`): a split screen is different furniture, and
        # furniture changes what the app measures about itself, so startup stays
        # single-player mode 1 whatever is saved. The startup mode is set in
        # `_build_scene` with no player count at all, which is the default 1.
        self.settings.players = normalise_players(self.settings.players)
        self.scene.players = self.settings.players
        # THE MAGAZINE: the one saved number that is both the length of a round
        # and the count of bullets drawn. Read here, held by the host, and
        # nowhere else; a missing key is the default (`settings.py`) and reading
        # it writes nothing. No UI yet, by his instruction.
        self.settings.magazine = normalise_magazine(self.settings.magazine)
        self.layers["modes"].set_magazine(self.settings.magazine)
        self.layers["modes"].health.margin_floor = MARGIN_READY
        # THE GHOST SCORE'S TWO INPUTS FROM SETTINGS. `bests` is a view onto the
        # very dict `settings.bests` holds, so what a round writes is what
        # `settings.save` persists. `fps` is the CONFIGURED rate (never the
        # measured one: a rate that dipped mid-round would revalue the ceiling);
        # `_publish_measuring` refreshes it from the FPS box on every tick.
        self.layers["modes"].bests = BestBook(self.settings.bests)
        self.layers["modes"].fps = float(self.settings.fps)

        # THE DIRECTOR: one owner of the screen, the camera and the
        # detector's routing. See `lsa/director.py` for why it exists; the
        # short version is that ~15 measurement routines were chains of
        # QTimer.singleShot callbacks sharing ~60 flags on this object, and
        # every race that cost a live run came out of that.
        #
        # The clock is injected so the headless harness can drive a whole
        # launch deterministically in milliseconds instead of jittering real
        # timers and sampling a few of the orderings that exist.
        self.director = Director(
            QtClock(self),
            sink=_DirectorSink(self),
            log=self._log,
            on_change=self._on_director_change,
            # What else owns the screen that the Director does not run yet.
            # One run slot is the whole answer only once everything is on
            # it; until then a Director routine could start on top of
            # pattern calibration or a legacy noise floor, which is the
            # very collision being fixed, reintroduced by the fixing.
            world_busy=self._screen_busy,
            screen_ready=self._screen_is_ours,
        )

        # Built through `audio_factory`, beside `capture_factory`, so a test
        # can hand the app a mixer that cannot open a device. Without the
        # seam every test that constructs a CameraLab opened a real output
        # stream on the system default endpoint -- which is his monitor.
        self.mixer = self.audio_factory(sounds_dir=_assets.asset_dir("sounds"),
                                        master_gain=self.settings.volume,
                                        device_name=self.settings.sound_device)

        # Spoken announcements. The microphone is muted for each line's own
        # duration (derived from the loaded file) plus the room tail, and
        # nothing is spoken within `QUIET_GAP_S` of a real trigger pull.
        self.voice = self.voice_factory(
            self.mixer, self.voice_dir,
            mute=lambda secs: self.listener.mute_for(secs),
            log=lambda text: self._log(text),
            enabled=lambda: self.chk_sound.isChecked(),
            tail=self.SOUND_TAIL_S)
        # Teaches the microphone how loud the speakers are BEFORE he shoots.
        # No timer of its own: it is ticked from the HUD tick
        # (`_audio_housekeeping`), so there is nothing to stand down at
        # shutdown (RULES A3). Built over the mixer `audio_factory` produced,
        # so a silent test mixer makes its sweeps silent too.
        self.teacher = CouplingTeacher(self.mixer, self.listener, self._log)

        # Timers before the widgets: restoring saved settings fires the same
        # change signals a user would, and those schedule a save.
        #
        # Saving on every keystroke would thrash the disk; saving only on exit
        # loses everything if the app is killed, which happens a lot during tuning.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1200)
        self._save_timer.timeout.connect(self._save_settings)

        # Startup sequence runs after the window is up, so the user sees progress
        # rather than a frozen window during camera negotiation.
        self._calibrate_timer = QTimer(self)
        self._calibrate_timer.setSingleShot(True)
        self._calibrate_timer.timeout.connect(self._begin_auto_calibration)
        # Every few seconds, not every frame: the mapping drifts over minutes,
        # and a registration pass costs more than a detection pass.
        self._align_timer = QTimer(self)
        self._align_timer.setInterval(4000)
        self._align_timer.timeout.connect(self._maybe_align)
        self._align_timer.start()

        # The room keeps changing after setup, too. A session that starts at
        # 09:00 and runs to 11:00 is not shooting in the same room it
        # calibrated in -- the sun moves, a blind goes up, a lamp goes on.
        # Asking is free (no control write), so ask regularly; only the
        # sweep it may provoke costs the camera anything, and that happens
        # only when the room has left the bars.
        # The sound output can go away underneath a running session. Check
        # it on a slow timer rather than at the moment of a shot, so the
        # recovery never sits in the path the shooter perceives as latency.
        self._audio_timer = QTimer(self)
        self._audio_timer.setInterval(5000)
        self._audio_timer.timeout.connect(self._check_audio_alive)
        self._audio_timer.start()

        # Speech is only ever STARTED from here, on the Qt thread, ten times
        # a second: a poll is a few comparisons; decoding happens elsewhere.
        self._voice_timer = QTimer(self)
        self._voice_timer.setInterval(100)
        self._voice_timer.timeout.connect(self.voice.poll)
        self._voice_timer.start()

        self._room_guard = QTimer(self)
        self._room_guard.setSingleShot(True)
        self._room_guard.timeout.connect(self._room_gave_up)
        self._room_timer = QTimer(self)
        self._room_timer.setInterval(int(self.ROOM_RECHECK_S * 1000))
        self._room_timer.timeout.connect(lambda: self._check_room("drift"))
        self._room_timer.start()

        self._build_ui()
        self._apply_settings_to_widgets()
        self._populate_devices()

        self._hud_timer = QTimer(self)
        self._hud_timer.setInterval(200)
        self._hud_timer.timeout.connect(self._refresh_target_diagnostics)
        self._hud_timer.start()

        QShortcut(QKeySequence("F11"), self, self._toggle_target_mode)
        QShortcut(QKeySequence("F5"), self, self._toggle_connect)

        if not self.mixer.available:
            self._log(f"audio unavailable: {self.mixer.error} (detection still works, just silent)")
        else:
            self._log(f"audio ready on {self.mixer.output_name}, "
                      f"output latency ~{self.mixer.latency_ms:.1f} ms")
        self._log_figure_cast()
        self.voice.say("v_welcome")
        self._set_connected(False)
        self._log("Target mode is F11. In it: P camera view (aiming only), S stats, H key help.")
        # SAID AT EVERY LAUNCH, so a feature that is switched off is never
        # mistaken for one that is broken (RULES C1). Nothing is measured
        # for it either: `_update_play_area` returns before it collects a
        # single shot, so there is no number to report.
        if playarea.BORDER_ENABLED:
            self._log("play-area border: ON (playarea.BORDER_ENABLED)")
        else:
            self._log("play-area border: OFF (playarea.BORDER_ENABLED = False), "
                      "not measured, not drawn, not limiting targets. It fenced "
                      "off the bottom third on a PREDICTED margin of 1.7x while "
                      "all six shots there were detected (19:52:55); targets "
                      "stay inside the camera-coverage rectangle only.")
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write("\n=== " + time.strftime("%Y-%m-%d %H:%M:%S")
                         + "  camera lab started ===\n")
        except Exception:
            pass
        if self.settings.auto_connect:
            QTimer.singleShot(250, self._autostart)
        else:
            self._log("auto-connect is off; press Connect (F5) when ready.")

    # -- scene ----------------------------------------------------------------

    def _build_scene(self) -> tuple[Scene, dict]:
        """The target-screen layer stack. Phase 1 swaps hits/aim for real targets."""
        scene = Scene(self.session)
        background = SolidBackground()
        layers = {
            "background": scene.add(background),
            "modes": scene.add(ModeHost()),
            "holes": scene.add(BulletHoles()),
            "calibration": scene.add(CalibrationTargets()),
            "pattern": scene.add(CalibrationPattern()),
            "synthetic": scene.add(SyntheticLaser()),
            "hits": scene.add(CameraSpaceHits()),
            "camera": scene.add(CameraPane(visible=False)),
            "stats": scene.add(StatsHud()),
            "bglabel": scene.add(BackgroundLabel(background)),
            "alert": scene.add(Alert()),
            "volume": scene.add(VolumeReadout()),
            # Visible only while the switch is on (`playarea.BORDER_ENABLED`);
            # it is OFF, so this layer exists and draws nothing.
            "playarea": scene.add(PlayAreaBorder()),
            "prompt": scene.add(Prompt()),
            "signal": scene.add(SignalMeter()),
            "verdict": scene.add(VerdictBoard()),
            "help": scene.add(KeyHelp()),
        }
        layers["playarea"].visible = bool(playarea.BORDER_ENABLED)
        # THE SCREEN IS NEVER EMPTY, and what is on it while the app MEASURES
        # ITSELF is the static targets -- a handful of thin rings on black.
        #
        # His instruction, 2026-10-18, after watching a launch land on the
        # menu: "previously the app would start with mode 1 for calibration
        # and in my mind mode 1 works better for calibration. Let's just start
        # calibration with mode 1 and display the menu when calibration is
        # complete."
        #
        # The reason is measurable, not aesthetic: the screen search, the
        # exposure probe and the self test all PHOTOGRAPH this screen, and the
        # menu lights eleven pictures across it. A dot landing on lit grey
        # loses its red excess, because red saturates while green and blue are
        # already high (B10) -- so a bright startup screen degrades the very
        # measurements that decide whether he may shoot.
        #
        # The menu goes up when those checks have answered, in
        # `_menu_when_setup_done`, and only over this mode.
        layers["modes"].set_mode(arcade.by_name(STARTUP_MODE_NAME)())
        return scene, layers

    # -- construction ---------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        # A DIVIDER HE CAN ACTUALLY DRAG. It was nominally a splitter already
        # and could not move: the settings pane was pinned between a minimum
        # of 440 and a maximum of 480, so the whole travel was forty pixels.
        # Asked for directly, 2026-10-18, together with "the settings should
        # have enough space on the right so that I could see everything on
        # that panel without scrolling".
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)   # neither pane can vanish
        splitter.setStyleSheet(
            "QSplitter::handle:horizontal { background:#2a2a2a; "
            "border-left:1px solid #3c3c3c; border-right:1px solid #3c3c3c; }"
            "QSplitter::handle:horizontal:hover { background:#3a3a3a; }")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)

        self.preview = QLabel("not connected")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 400)
        self.preview.setStyleSheet("background:#111; color:#777; border:1px solid #333;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lv.addWidget(self.preview, 1)

        # The preview shows the camera feed on the very screen the camera is
        # watching, which is an infinite mirror: the laser dot recurses inside it
        # and each reflection is a real red spot to the detector. Multiplying the
        # preview by blue zeroes the red channel so it cannot register as a hit.
        self.chk_safe_preview = QCheckBox(
            "Laser-safe preview - prevents the camera seeing its own feed as hits")
        self.chk_safe_preview.setChecked(True)
        self.chk_safe_preview.setToolTip(
            "Leave this on whenever the camera is pointed at this monitor.\n"
            "Turn it off only to judge true colour, and expect phantom detections if you do."
        )
        lv.addWidget(self.chk_safe_preview)

        strip = QHBoxLayout()
        self.mask_label = QLabel("mask off")
        self.mask_label.setFixedSize(320, 180)
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setStyleSheet("background:#000; border:1px solid #333; color:#555;")
        strip.addWidget(self.mask_label)

        self.readout = QLabel("")
        self.readout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.readout.setFont(QFont("Consolas", 10))
        self.readout.setStyleSheet("color:#bbb;")
        strip.addWidget(self.readout, 1)
        lv.addLayout(strip)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.setFont(QFont("Consolas", 9))
        lv.addWidget(self.log_view)

        splitter.addWidget(left)

        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(8, 8, 8, 8)
        self.lbl_status = QLabel()
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        pv.addWidget(self.lbl_status)
        pv.addWidget(self._group_camera())
        pv.addWidget(self._group_exposure())
        pv.addWidget(self._group_detector())
        pv.addWidget(self._group_tracking())
        pv.addWidget(self._group_target())
        pv.addWidget(self._group_hittest())
        pv.addWidget(self._group_recording())
        pv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        # WIDE ENOUGH THAT NOTHING SCROLLS. The panel's own sizeHint is what its
        # widest row genuinely needs -- the control rows are a 116px name, a
        # slider, a 78px spin box and a 60px echo, and 480 was not enough for
        # them, which is why a horizontal bar appeared along the bottom and
        # the right-hand edge of every slider was cut off. So the width is
        # ASKED FOR rather than guessed: the content's hint, plus room for the
        # vertical scrollbar and the frame. Measured on this layout it comes
        # out around 520-560; it will follow the panel if a wider row is ever
        # added, which a constant would not (RULES C7).
        #
        # The horizontal bar is left on AS NEEDED rather than switched off.
        # Off, a row that one day wants more width is CLIPPED instead of
        # scrolled -- hidden content where there used to be a visible bar,
        # which is the worse failure. With the minimum below it should never
        # appear; if it does, it is telling the truth about something new.
        bar_w = scroll.verticalScrollBar().sizeHint().width()
        chrome = bar_w + 2 * scroll.frameWidth() + 4
        # The floor is what the old cap was plus the slack it was short by, so
        # a sizeHint reported before the window is shown -- which can
        # under-report -- cannot land him back where he started.
        need = max(540, panel.sizeHint().width() + chrome)
        # The MINIMUM is what the panel genuinely cannot shrink below, so a
        # drag can never clip a control; the INITIAL width is the wider
        # "everything visible" number above.
        scroll.setMinimumWidth(min(need, panel.minimumSizeHint().width() + chrome))
        # NO MAXIMUM. That cap is what made the divider immovable; he can now
        # give the settings as much of the window as he likes, and the preview
        # keeps the rest.
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([max(640, self.width() - need - 24), need])
        self.setCentralWidget(splitter)

    def _group_camera(self) -> QGroupBox:
        g = QGroupBox("1. Camera")
        form = QFormLayout(g)

        self.cmb_device = QComboBox()
        self.cmb_backend = QComboBox()
        self.cmb_res = QComboBox()
        self.cmb_fourcc = QComboBox()
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 240)
        self.spin_fps.setValue(30)

        for w, h in COMMON_MODES:
            self.cmb_res.addItem(f"{w} x {h}", (w, h))
        self.cmb_res.setCurrentText("1280 x 720")
        self.cmb_fourcc.addItems(["MJPG", "YUY2", "NV12", "(leave alone)"])

        form.addRow("Device", self.cmb_device)
        form.addRow("Backend", self.cmb_backend)
        form.addRow("Resolution", self.cmb_res)
        form.addRow("Codec", self.cmb_fourcc)
        form.addRow("FPS", self.spin_fps)

        row0 = QHBoxLayout()
        b_id = QPushButton("Which camera?")
        b_id.setToolTip("Grab one still from every camera so you can see which one faces the screen.")
        b_id.clicked.connect(self._identify_cameras)
        b_re = QPushButton("Rescan")
        b_re.clicked.connect(self._populate_devices)
        row0.addWidget(b_id)
        row0.addWidget(b_re)
        form.addRow(row0)

        btns = QHBoxLayout()
        self.btn_connect = QPushButton("Connect  (F5)")
        self.btn_connect.clicked.connect(self._toggle_connect)
        self.btn_scan = QPushButton("Scan formats...")
        self.btn_scan.clicked.connect(self._scan_formats)
        btns.addWidget(self.btn_connect)
        btns.addWidget(self.btn_scan)
        form.addRow(btns)

        self.lbl_negotiated = QLabel("-")
        self.lbl_negotiated.setWordWrap(True)
        self.lbl_negotiated.setStyleSheet("color:#8ab4ff;")
        form.addRow("Actual", self.lbl_negotiated)

        self.chk_autoconnect = QCheckBox("Connect automatically on launch")
        self.chk_autoconnect.setChecked(True)
        self.chk_autocal = QCheckBox("Darken and calibrate automatically after connecting")
        self.chk_autocal.setChecked(True)
        self.chk_autocal.setToolTip(
            "Applies the Laser hunt exposure, waits for it to settle, measures the\n"
            "noise floor and sets the detection threshold from it. No fiddling needed."
        )
        form.addRow(self.chk_autoconnect)
        form.addRow(self.chk_autocal)

        # Selecting a different camera should just switch to it. Making the user
        # then find and press Connect is a step the app can take by itself.
        self.cmb_device.currentIndexChanged.connect(self._on_device_changed)
        for w in (self.cmb_backend, self.cmb_res, self.cmb_fourcc):
            w.currentIndexChanged.connect(self._on_format_changed)
        self.spin_fps.valueChanged.connect(self._on_format_changed)
        return g

    def _group_exposure(self) -> QGroupBox:
        g = QGroupBox("2. Exposure and camera controls")
        v = QVBoxLayout(g)
        v.addWidget(QLabel(
            "<small>Blue numbers are what the camera reported back after the write. "
            "If a value will not stick, use the driver's own dialog.</small>"
        ))

        row = QHBoxLayout()
        b1 = QPushButton("Laser hunt (dark)")
        b1.setToolTip("Underexpose hard: the screen fades to a ghost, the laser still saturates.")
        b1.clicked.connect(lambda: self._apply_preset("laser"))
        b2 = QPushButton("Calibration view")
        b2.setToolTip("Normal exposure so the camera can see the monitor.")
        b2.clicked.connect(lambda: self._apply_preset("calib"))
        row.addWidget(b1)
        row.addWidget(b2)
        v.addLayout(row)
        self._needs_camera += [b1, b2]

        row2 = QHBoxLayout()
        b3 = QPushButton("Probe controls")
        b3.setToolTip("Nudge every control and read it back to find which ones the driver honours.")
        b3.clicked.connect(lambda: self.capture and self.capture.request_probe())
        b4 = QPushButton("Driver dialog...")
        b4.setToolTip("Open the camera driver's own property sheet (DirectShow only).")
        b4.clicked.connect(lambda: self.capture and self.capture.request_driver_dialog())
        row2.addWidget(b3)
        row2.addWidget(b4)
        v.addLayout(row2)
        self._needs_camera += [b3, b4]

        for spec in controls.CONTROLS:
            r = ControlRow(spec)
            r.changed.connect(self._on_control_changed)
            self.rows[spec.key] = r
            v.addWidget(r)
        return g

    def _group_detector(self) -> QGroupBox:
        g = QGroupBox("3. Detector")
        form = QFormLayout(g)

        self.cmb_mode = QComboBox()
        for m in ScoreMode:
            self.cmb_mode.addItem(m.value, m)
        self.cmb_mode.setCurrentText(ScoreMode.CHROMA.value)
        self.cmb_mode.currentIndexChanged.connect(self._push_detector)
        form.addRow("Score mode", self.cmb_mode)

        self.lbl_mode = QLabel(ScoreMode.CHROMA.description)
        self.lbl_mode.setWordWrap(True)
        self.lbl_mode.setStyleSheet("color:#999;")
        form.addRow("", self.lbl_mode)

        def spin(lo, hi, val, step=1, tip=""):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setValue(val)
            if tip:
                s.setToolTip(tip)
            s.valueChanged.connect(self._push_detector)
            return s

        self.sp_threshold = spin(0, 255, 60)
        self.sp_minpeak = spin(0, 255, 80)
        self.sp_blur = spin(0, 15, 3, 2)
        self.sp_minarea = spin(1, 500, 2)
        # CAPPED AT THE CODE'S OWN LIMIT, so a larger value in
        # settings.local.json cannot take effect. His file carried 2000
        # against the code's 1500, which is how a 1,899 px bloom -- the
        # figure's edge, or a sensor-saturated core -- was accepted as a
        # shot on 2026-10-17 and drew a wrong hole. Same lesson as
        # `min_frames` (a saved setting silently beating a code default):
        # when a default exists to prevent a failure, put the bad value out
        # of reach rather than just changing the default.
        #
        # THE COST, MEASURED (2026-10-17, tools/ over the detections.csv of
        # session-20261017-100406, exposure -3, 155 real shots = a run of
        # consecutive-frame detections with a peak >= 30): 9 have their
        # brightest frame over 1500 px and 8 have EVERY frame over 1500, so
        # they vanish -- 5.2%, about one near-side shot in twenty. Nothing
        # exceeded 1978, so his saved 2000 admitted all of them. By ceiling:
        # 1500 loses 8, 1700 loses 6, 1800 loses 3, 2000 loses 0.
        # ESTIMATE, not measurement (nothing was recorded at a second
        # exposure): at -5, scaling each frame's peak by the probe's
        # 0.64-0.84 per two stops and the area by ln(peak/threshold), 4 to 7
        # of the 155 are still lost (2.6-4.5%). Blobs shrink less than the
        # peak does, because the bloom is wider than the core.
        # KEPT AT 1500 rather than made conditional on the exposure: the shots
        # lost are the sensor-saturated cores (peak 220-255, raw red channel
        # clipped), whose centroid is the least trustworthy of all -- the
        # wrong-hole failure this clamp was added for, a 1,899 px blob. A
        # ceiling that rises where the sensor clips (a brighter exposure)
        # raises it exactly where it is least safe, and 2000 was never a
        # measured limit (it was the old spinbox default). The honest
        # remedy for those shots is the exposure, which the probe's strong-end
        # constraint steps down to -5; this ceiling is revisited with a
        # recording AT -5, which does not exist yet.
        self.sp_maxarea = spin(10, self.MAX_AREA_CEILING, self.MAX_AREA_CEILING, 50)
        self.sp_close = spin(0, 5, 1)
        self.sp_maxblobs = spin(1, 8, 4, 1,
                                "Dots reported per frame. Must be 2 or more to see both guns.")
        self.sp_minsep = spin(0, 200, 10, 1,
                              "Blobs closer than this count as fragments of one dot.\n"
                              "Too large and two close simultaneous hits merge.")
        self.sp_elong = QDoubleSpinBox()
        self.sp_elong.setRange(0.0, 20.0)
        self.sp_elong.setSingleStep(0.5)
        self.sp_elong.setDecimals(1)
        self.sp_elong.setValue(0.0)
        self.sp_elong.setToolTip(
            "0 = off. An off-axis camera images the dot as an ellipse, so keep this\n"
            "generous or disabled. It is for killing long specular streaks."
        )
        self.sp_elong.valueChanged.connect(self._push_detector)

        form.addRow("Threshold", self.sp_threshold)
        form.addRow("Min peak", self.sp_minpeak)
        form.addRow("Blur (odd)", self.sp_blur)
        form.addRow("Min area px", self.sp_minarea)
        form.addRow("Max area px", self.sp_maxarea)
        form.addRow("Close iters", self.sp_close)
        form.addRow("Max dots/frame", self.sp_maxblobs)
        form.addRow("Min separation px", self.sp_minsep)
        self.sp_roimargin = QSpinBox()
        self.sp_roimargin.setRange(-50, 200)
        self.sp_roimargin.setValue(0)
        self.sp_roimargin.setToolTip(
            "Shrink the detected screen outline by this many pixels.\n"
            "A few pixels of margin trims glare along the bezel without\n"
            "losing shots near the edge of the screen.")
        self.sp_roimargin.valueChanged.connect(self._push_detector)
        form.addRow("Max elongation", self.sp_elong)
        form.addRow("Screen margin px", self.sp_roimargin)

        self.chk_bg = QCheckBox("Subtract running background")
        self.chk_bg.setChecked(True)
        self.chk_bg.setToolTip(
            "Models the slow-changing screen content and looks for a sudden spike.\n"
            "The model freezes while a laser is visible so it cannot chase the dot."
        )
        self.chk_bg.stateChanged.connect(self._push_detector)
        form.addRow(self.chk_bg)

        self.sp_alpha = QDoubleSpinBox()
        self.sp_alpha.setRange(0.001, 1.0)
        self.sp_alpha.setSingleStep(0.01)
        self.sp_alpha.setDecimals(3)
        self.sp_alpha.setValue(0.05)
        self.sp_alpha.valueChanged.connect(self._push_detector)
        form.addRow("Background alpha", self.sp_alpha)

        self.cmb_scale = QComboBox()
        for s in ("1.0", "0.75", "0.5", "0.25"):
            self.cmb_scale.addItem(s, float(s))
        self.cmb_scale.currentIndexChanged.connect(self._push_detector)
        form.addRow("Process scale", self.cmb_scale)

        row_cal = QHBoxLayout()
        b_noise = QPushButton("Measure noise floor (3 s)")
        b_noise.setToolTip(
            "Watches for three seconds with nothing being fired, finds the brightest\n"
            "thing the detector sees, and sets the threshold above it.\n"
            "Do not shoot while it runs. This is the cure for phantom shots."
        )
        b_noise.clicked.connect(
            lambda: self._start_noise_floor(3.0))
        row_cal.addWidget(b_noise)
        form.addRow(row_cal)
        self._needs_camera.append(b_noise)

        row = QHBoxLayout()
        self.chk_mask = QCheckBox("Show mask")
        self.chk_mask.stateChanged.connect(
            lambda: self.capture and self.capture.set_show_mask(self.chk_mask.isChecked()))
        self.chk_overlay = QCheckBox("Crosshair")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(
            lambda: self.capture and self.capture.set_draw_overlay(self.chk_overlay.isChecked()))
        bbg = QPushButton("Reset background")
        bbg.clicked.connect(lambda: self.capture and self.capture.reset_background())
        row.addWidget(self.chk_mask)
        row.addWidget(self.chk_overlay)
        row.addWidget(bbg)
        form.addRow(row)
        return g

    def _group_tracking(self) -> QGroupBox:
        g = QGroupBox("4. Shot tracking")
        form = QFormLayout(g)
        form.addRow(QLabel(
            "<small>A 100 ms pulse spans about 3 frames at 30 fps, so requiring 2+ frames "
            "discards single-frame noise almost for free. One track per dot, so two guns "
            "fired together give two shots.</small>"
        ))
        self.sp_minframes = QSpinBox()
        # Floor of 2, not 1. One frame is not a shot: a 100ms pulse spans six
        # frames at 60fps, and allowing one admitted a whole class of false
        # positives that fired the gunshot sound with nobody shooting. Kept out
        # of reach rather than merely defaulted away, because a saved setting
        # silently overrides a default and this one is not worth re-learning.
        self.sp_minframes.setRange(2, 10)
        self.sp_minframes.setValue(2)
        self.sp_gap = QSpinBox()
        self.sp_gap.setRange(0, 15)
        self.sp_gap.setToolTip(
            "Frames a dot may vanish for and still be the same shot.\n"
            "Measured: one trigger pull at (445,471) was counted four times\n"
            "in 0.29s because detection flickered. 4 frames absorbs that.")
        self.sp_gap.setValue(4)
        self.sp_move = QSpinBox()
        self.sp_move.setRange(1, 500)
        self.sp_move.setValue(40)
        self.sp_move.setToolTip(
            "How far a dot may move between frames and still be the same shot.\n"
            "Too large and two nearby simultaneous hits get stitched into one track.")
        self.sp_tracks = QSpinBox()
        self.sp_tracks.setRange(1, 8)
        self.sp_tracks.setValue(4)
        self.sp_tracks.setToolTip("Simultaneous shots to follow. Two guns plus headroom = 4.")
        for s in (self.sp_minframes, self.sp_gap, self.sp_move, self.sp_tracks):
            s.valueChanged.connect(self._push_tracker)
        form.addRow("Min frames per shot", self.sp_minframes)
        form.addRow("Max gap frames", self.sp_gap)
        form.addRow("Max move px", self.sp_move)
        self.sp_maxdur = QSpinBox()
        self.sp_maxdur.setRange(50, 5000)
        self.sp_maxdur.setSingleStep(50)
        self.sp_maxdur.setValue(400)
        self.sp_maxdur.setToolTip(
            "Longest a dot may persist and still count as a shot.\n"
            "A LaserLyte pulse is ~100 ms. A lamp or a reflection stays lit for\n"
            "seconds, and without this ceiling it eventually registers as one\n"
            "enormous shot. 400 ms leaves plenty of margin.")
        self.sp_maxdur.valueChanged.connect(self._push_tracker)
        form.addRow("Max simultaneous", self.sp_tracks)
        form.addRow("Max shot length ms", self.sp_maxdur)
        return g

    def _group_target(self) -> QGroupBox:
        g = QGroupBox("5. Target screen  (F11)")
        v = QVBoxLayout(g)
        v.addWidget(QLabel(
            "<small>Fullscreen aiming field with the scoreboard. Panels toggle from here or "
            "with keys on the target screen itself.</small>"
        ))

        row = QHBoxLayout()
        self.btn_target = QPushButton("Open target screen  (F11)")
        self.btn_target.clicked.connect(self._toggle_target_mode)
        row.addWidget(self.btn_target)
        v.addLayout(row)

        grid = QGridLayout()
        self.chk_pane = QCheckBox("Camera pane  (P)")
        self.chk_pane.setToolTip(
            "Live camera view on the target screen, rendered through a blue multiply so\n"
            "it cannot register as a hit. Without that, the camera would see its own\n"
            "feed and the laser dot inside it, producing phantom shots."
        )
        self.chk_pane.stateChanged.connect(
            lambda: self._set_layer("camera", self.chk_pane.isChecked()))
        self.chk_stats = QCheckBox("Stats panel  (S)")
        self.chk_stats.setChecked(True)
        self.chk_stats.stateChanged.connect(
            lambda: self._set_layer("stats", self.chk_stats.isChecked()))
        self.chk_minimap = QCheckBox("Camera minimap")
        self.chk_minimap.setChecked(False)
        self.chk_minimap.stateChanged.connect(
            lambda: self._set_layer("hits", self.chk_minimap.isChecked()))
        self.chk_help = QCheckBox("Key help  (H)")
        self.chk_help.setChecked(True)
        self.chk_help.stateChanged.connect(
            lambda: self._set_layer("help", self.chk_help.isChecked()))
        grid.addWidget(self.chk_pane, 0, 0)
        grid.addWidget(self.chk_stats, 0, 1)
        grid.addWidget(self.chk_minimap, 1, 0)
        self.chk_holes = QCheckBox("Bullet holes")
        self.chk_holes.setChecked(True)
        self.chk_holes.stateChanged.connect(
            lambda: self._set_layer("holes", self.chk_holes.isChecked()))
        grid.addWidget(self.chk_help, 2, 0)
        grid.addWidget(self.chk_holes, 2, 1)
        v.addLayout(grid)
        return g

    def _group_hittest(self) -> QGroupBox:
        g = QGroupBox("6. Hit-rate test  <-- the number that decides the project")
        v = QVBoxLayout(g)
        v.addWidget(QLabel(
            "<small>Press Reset, open the target screen, fire a known number of shots, "
            "come back and enter how many you fired.</small>"
        ))
        grid = QGridLayout()
        self.sp_fired = QSpinBox()
        self.sp_fired.setRange(0, 1000)
        self.sp_fired.valueChanged.connect(self._refresh_hitrate)
        grid.addWidget(QLabel("Shots I fired:"), 0, 0)
        grid.addWidget(self.sp_fired, 0, 1)
        self.lbl_hitrate = QLabel("-")
        self.lbl_hitrate.setFont(QFont("Consolas", 14, QFont.Bold))
        grid.addWidget(self.lbl_hitrate, 0, 2)
        v.addLayout(grid)

        row = QHBoxLayout()
        b = QPushButton("Reset counters")
        b.clicked.connect(self._reset_counters)
        self.chk_sound = QCheckBox("Sound on hit")
        self.chk_sound.setChecked(True)
        self.chk_sound.stateChanged.connect(self._schedule_save)
        row.addWidget(b)
        row.addWidget(self.chk_sound)
        v.addLayout(row)

        srow = QHBoxLayout()
        self.cmb_sound = QComboBox()
        for name in self.mixer.names():
            self.cmb_sound.addItem(name)
        self.cmb_sound.currentIndexChanged.connect(self._schedule_save)
        b_test = QPushButton("Test")
        # NOT mixer.play(): that was a second path from the app to the room,
        # a real gunshot at session volume registered with nobody, which a
        # calibrated microphone counted as a trigger pull. Through
        # `play_own_sound` it is registered like any bang (muted or discounted,
        # logged) and it is never a shot (RULES C17, test_rules section U).
        b_test.clicked.connect(self._play_test_sound)
        srow.addWidget(QLabel("Shot sound"))
        srow.addWidget(self.cmb_sound, 1)
        srow.addWidget(b_test)
        v.addLayout(srow)

        # Listening for the gun is a second, independent answer to "did a shot
        # happen". The camera says where a hit landed; it cannot distinguish a
        # shot it failed to see from a shot that missed the screen from no shot
        # at all, and those three need different responses.
        mrow = QHBoxLayout()
        self.chk_listen = QCheckBox("Listen for shots")
        self.chk_listen.setToolTip(
            "Use the camera's microphone to hear the gun.\n"
            "A shot that is heard but not seen means it missed the screen, or the\n"
            "camera cannot see where it landed. A hit seen with no bang behind it\n"
            "is a phantom. Sound never creates a hit: it has no position.")
        self.chk_listen.setChecked(self.settings.listen_for_shots)
        self.chk_listen.stateChanged.connect(self._on_listen_changed)
        self.cmb_mic = QComboBox()
        self.cmb_mic.addItem("auto", "")
        for idx, name, api in input_devices():
            self.cmb_mic.addItem(f"{name}  [{api}]", name)
        self._select_mic_in_combo(self.settings.mic_name)
        self.cmb_mic.currentIndexChanged.connect(self._on_listen_changed)
        mrow.addWidget(self.chk_listen)
        mrow.addWidget(self.cmb_mic, 1)
        v.addLayout(mrow)

        vrow = QHBoxLayout()
        self.sld_volume = QSlider(Qt.Horizontal)
        self.sld_volume.setRange(0, 100)
        self.sld_volume.setValue(int(self.settings.volume * 100))
        self.sld_volume.setToolTip(
            "Deliberately low by default. A gunshot sample at full volume through\n"
            "desktop speakers is startling the first time it fires.")
        self.lbl_volume = QLabel(f"{self.sld_volume.value()}%")
        self.lbl_volume.setFixedWidth(44)
        self.sld_volume.valueChanged.connect(self._on_volume_changed)
        vrow.addWidget(QLabel("Volume"))
        vrow.addWidget(self.sld_volume, 1)
        vrow.addWidget(self.lbl_volume)
        v.addLayout(vrow)
        return g

    def _group_recording(self) -> QGroupBox:
        g = QGroupBox("7. Recording (for offline tuning)")
        v = QVBoxLayout(g)
        v.addWidget(QLabel(
            "<small>Saves video, a per-frame detection CSV and all settings, so the "
            "detector can be retuned against your real room without you standing there.</small>"
        ))
        row = QHBoxLayout()
        self.btn_record = QPushButton("Start recording")
        self.btn_record.clicked.connect(self._toggle_record)
        bsnap = QPushButton("Snapshot")
        bsnap.clicked.connect(
            lambda: self.capture and self.capture.save_snapshot(RECORDINGS / "snapshots"))
        row.addWidget(self.btn_record)
        row.addWidget(bsnap)
        v.addLayout(row)
        self._needs_camera += [self.btn_record, bsnap]
        self.lbl_record = QLabel(str(RECORDINGS))
        self.lbl_record.setWordWrap(True)
        self.lbl_record.setStyleSheet("color:#888;")
        v.addWidget(self.lbl_record)
        return g

    # -- settings persistence -------------------------------------------------

    # The code's own max_area (laser.py), the most a saved setting may ask for.
    MAX_AREA_CEILING = DetectorParams.max_area

    def _apply_settings_to_widgets(self) -> None:
        """Restore last session's choices. Device selection happens separately."""
        st = self.settings
        if self.cmb_backend.count() == 0:
            for name in available_backends():
                if name in ("MSMF", "DSHOW"):
                    self.cmb_backend.addItem(name)
        i = self.cmb_backend.findText(st.backend)
        if i >= 0:
            self.cmb_backend.setCurrentIndex(i)
        for idx in range(self.cmb_res.count()):
            if self.cmb_res.itemData(idx) == (st.width, st.height):
                self.cmb_res.setCurrentIndex(idx)
                break
        j = self.cmb_fourcc.findText(st.fourcc)
        if j >= 0:
            self.cmb_fourcc.setCurrentIndex(j)
        self.spin_fps.setValue(int(st.fps))
        self.chk_safe_preview.setChecked(st.safe_preview)
        self.chk_sound.setChecked(st.sound)
        self.sld_volume.setValue(int(round(st.volume * 100)))
        if st.shot_sound:
            k = self.cmb_sound.findText(st.shot_sound)
            if k >= 0:
                self.cmb_sound.setCurrentIndex(k)
        self.chk_autoconnect.setChecked(st.auto_connect)
        self.chk_autocal.setChecked(st.auto_calibrate)

        spins = {
            "threshold": self.sp_threshold, "min_peak": self.sp_minpeak,
            "blur": self.sp_blur, "min_area": self.sp_minarea,
            "max_area": self.sp_maxarea, "close_iter": self.sp_close,
            "max_blobs": self.sp_maxblobs, "min_separation": self.sp_minsep,
        }
        for key, widget in spins.items():
            if key in st.detector:
                if int(st.detector[key]) > widget.maximum():
                    # Once, because this runs once per launch.
                    self._log(
                        f"saved {key} of {int(st.detector[key])} is above the "
                        f"limit of {widget.maximum()} and is clamped to it: "
                        f"anything larger is a bloom or a figure's edge, not "
                        f"a laser dot (a 1,899 px blob was accepted as a "
                        f"shot on 2026-10-17). THE COST: about one real "
                        f"shot in twenty is bigger than that in every frame "
                        f"and is NOT reported (8 of 155 on 2026-10-17 at "
                        f"exposure -3, sensor-saturated near-side shots; an "
                        f"ESTIMATED 4-7 of 155 at -5). Those are shots you "
                        f"will fire and see no hole for.")
                widget.setValue(int(st.detector[key]))
        if "max_elongation" in st.detector:
            self.sp_elong.setValue(float(st.detector["max_elongation"]))
        if "bg_alpha" in st.detector:
            self.sp_alpha.setValue(float(st.detector["bg_alpha"]))
        if "use_background" in st.detector:
            self.chk_bg.setChecked(bool(st.detector["use_background"]))
        if "mode" in st.detector:
            k = self.cmb_mode.findText(str(st.detector["mode"]))
            if k >= 0:
                self.cmb_mode.setCurrentIndex(k)

        for key, widget in (("min_frames", self.sp_minframes), ("max_gap_frames", self.sp_gap),
                            ("max_move_px", self.sp_move), ("max_tracks", self.sp_tracks)):
            if key in st.tracker:
                widget.setValue(int(st.tracker[key]))

        for name, box in self._layer_boxes().items():
            if name in st.layers:
                box.setChecked(bool(st.layers[name]))
                self._set_layer(name, bool(st.layers[name]))

        if "exposure" in self.rows:
            self.rows["exposure"].set_value_silently(st.exposure)

    def _layer_boxes(self) -> dict:
        return {"camera": self.chk_pane, "stats": self.chk_stats,
                "hits": self.chk_minimap, "help": self.chk_help, "holes": self.chk_holes}

    def _schedule_save(self) -> None:
        self._save_timer.start()

    def _save_bests(self) -> None:
        """Persist a new personal best. The write is `AppSettings.save`, which is
        atomic (`lsa/atomic.py`): a kill mid-write leaves the old file or the new
        one, never a truncated one that would lose the camera chosen by name and
        the measured exposure. Never allowed to break the tick that drains it."""
        try:
            ok = self._save_settings()
        except Exception as exc:                      # noqa: BLE001
            self._log(f"personal best could NOT be saved: {exc!r}")
            return
        if ok is False:
            from lsa import atomic as _atomic
            self._log(f"personal best could NOT be saved: "
                      f"{_atomic.last_error or 'no reason recorded'}")
        else:
            self._log("personal best saved to the settings file")

    def _save_settings(self) -> "bool":
        st = self.settings
        st.backend = self.cmb_backend.currentText()
        st.width, st.height = self.cmb_res.currentData()
        st.fourcc = self.cmb_fourcc.currentText()
        st.fps = self.spin_fps.value()
        st.safe_preview = self.chk_safe_preview.isChecked()
        st.sound = self.chk_sound.isChecked()
        st.volume = self.sld_volume.value() / 100.0
        st.shot_sound = self.cmb_sound.currentText()
        st.listen_for_shots = self.chk_listen.isChecked()
        st.mic_name = self.cmb_mic.currentData() or ""
        st.auto_connect = self.chk_autoconnect.isChecked()
        st.auto_calibrate = self.chk_autocal.isChecked()
        # The exposure chosen for SHOOTING, not whatever the spinbox happens to
        # show. The calibration preset drives the camera to -10 by design, and
        # saving that on exit meant every launch booted two stops darker than
        # the last -- the dot stopped blooming, nothing registered, and the
        # cause looked like a detector fault for most of an evening.
        st.exposure = float(getattr(self, "_shooting_exposure", None)
                            or self.rows["exposure"].spin.value())

        params = self._detector_params()
        st.detector = {**vars(params), "mode": ScoreMode(params.mode).value}
        st.tracker = vars(self._tracker_config())
        st.layers = {name: box.isChecked() for name, box in self._layer_boxes().items()}

        idx = self.cmb_device.currentData()
        if idx is not None:
            st.device_index = int(idx)
            for c in getattr(self, "_cameras", []):
                if c.index == idx:
                    # Only remember a camera the user actually chose. When the
                    # remembered one is unplugged the app falls back to whatever
                    # is present, and saving THAT makes the mistake permanent --
                    # the next launch "remembers" the wrong camera and stops
                    # warning. Observed for real: the fallback was the webcam
                    # pointed at the user, and it overwrote the shooting camera.
                    if self._device_was_chosen or not st.device_name:
                        st.device_name = c.name
                    break
        return st.save(SETTINGS_PATH)

    # -- automatic startup ----------------------------------------------------

    def _autostart(self) -> None:
        """Launch straight into a working state: connect, darken, calibrate."""
        if self.cmb_device.currentData() is None:
            self._log("no usable camera found - plug one in and press Rescan")
            self._splash_problem("No usable camera was found. "
                                 "Plug one in, then restart (or press Rescan).")
            return
        if self.auto_aim:
            # Aiming only. Setup would flash the screen, sweep the exposure and
            # ask for nine shots -- none of which helps someone holding the
            # camera, and all of which is control traffic this device dislikes.
            self._log("aim mode: connecting and showing the camera brightly so "
                      "the monitor can be framed. No flash, no exposure sweep, "
                      "no grid. Press V on the target screen to leave it, then "
                      "F5 to run setup for real.")
            self._connect()
            QTimer.singleShot(2500, self._enter_aim_on_start)
            return
        self._connect()
        # The target screen goes up FIRST, before any measurement.
        #
        # It used to open only after setup finished, which meant the screen
        # search, the exposure and the noise floor were all measured against
        # whatever happened to be on the DESKTOP -- editors, browsers, this
        # app's own console. On 2026-10-14 that was the difference between a
        # noise floor of 2 and a noise floor of 90, and when setup failed for
        # any reason the target screen never opened at all, so the camera spent
        # the whole session staring at a lit desktop and detecting nothing.
        #
        # Everything downstream assumes a dark screen. Put one there first.
        if self.auto_target and self.target_window is None:
            QTimer.singleShot(600, self._toggle_target_mode)

    def _begin_auto_calibration(self) -> None:
        if self.capture is None or not self._setup_wanted():
            return
        self._set_status(
            "CALIBRATING - do not fire",
            f"Measuring the noise floor for {self.settings.calibrate_seconds:g}s.",
            "#3a2f16", "#e8d48a", "#7a6a2a",
        )
        self._start_noise_floor(self.settings.calibrate_seconds)

    def _set_status(self, title: str, detail: str, bg: str, fg: str, border: str) -> None:
        self.lbl_status.setText(title + ("\n" + detail if detail else ""))
        self.lbl_status.setStyleSheet(
            f"background:{bg}; color:{fg}; border:1px solid {border};"
            "padding:6px; font-weight:bold;")

    def _set_connected(self, connected: bool, detail: str = "") -> None:
        """Single place that decides what is usable, so the UI cannot lie."""
        for w in self._needs_camera:
            w.setEnabled(connected)
        for row in self.rows.values():
            row.setEnabled(connected)
        if connected:
            self.lbl_status.setText("CONNECTED" + (f"  -  {detail}" if detail else ""))
            self.lbl_status.setStyleSheet(
                "background:#17331c; color:#7fe08a; border:1px solid #2e6b38;"
                "padding:6px; font-weight:bold;")
        else:
            self.lbl_status.setText(
                "NOT CONNECTED" + (f"  -  {detail}" if detail else "")
                + "\nPick a camera above and press Connect (F5)."
                + "\nExposure, recording and snapshots stay disabled until then.")
            self.lbl_status.setStyleSheet(
                "background:#322e16; color:#e0cc9a; border:1px solid #5e5a2a;"
                "padding:6px; font-weight:bold;")

    # -- devices --------------------------------------------------------------

    def _populate_devices(self) -> None:
        cams = list_cameras()
        self._cameras = cams

        # Repopulating fires currentIndexChanged, which would reconnect mid-scan.
        self.cmb_device.blockSignals(True)
        self.cmb_device.clear()
        for c in cams:
            self.cmb_device.addItem(c.label, c.index)

        wanted = self.settings.resolve_device(cams)
        pick = 0
        for i in range(self.cmb_device.count()):
            if self.cmb_device.itemData(i) == wanted:
                pick = i
                break
        self.cmb_device.setCurrentIndex(pick)
        self.cmb_device.blockSignals(False)

        self._log(f"{len(cams)} capture device(s): " + ", ".join(c.label for c in cams))
        if wanted is not None:
            name = next((c.name for c in cams if c.index == wanted), "?")
            saved = self.settings.device_name
            if saved and name != saved:
                # Say this loudly. The remembered camera being absent and a
                # different one silently taking its place is indistinguishable
                # from a detector that does not work: the fallback here was the
                # webcam on top of the monitor, pointed at the user, and it
                # cost a whole session before anyone noticed.
                self._log(f"*** '{saved}' IS NOT CONNECTED. Falling back to "
                          f"index {wanted} ({name}), which is probably pointed "
                          f"at the wrong thing. Plug the shooting camera in. ***")
                self._set_status(
                    f"WRONG CAMERA  -  '{saved}' is not connected",
                    f"Using {name} instead. Plug in the shooting camera and press F5.",
                    "#322e16", "#c8c070", "#5e5a2a",
                )
                self._alert(
                    "WRONG CAMERA",
                    f"The shooting camera ('{saved}') is not connected, so the app "
                    f"fell back to '{name}' - which is probably pointed somewhere "
                    f"else entirely.\n\n"
                    f"Plug the shooting camera in, then press F5 to reconnect.")
            else:
                how = "remembered from last time" if saved else "first real camera"
                self._log(f"selected index {wanted} ({name}) - {how}")

    def _on_device_changed(self, _idx: int) -> None:
        """Switching camera reconnects straight away rather than waiting to be told."""
        self._device_was_chosen = True
        self._schedule_save()
        if self.cmb_device.currentData() is None:
            return
        name = self.cmb_device.currentText()
        if self.capture is not None:
            self._log(f"switching to {name}...")
            self._disconnect()
        if self.chk_autoconnect.isChecked():
            QTimer.singleShot(150, self._connect)

    def _on_format_changed(self, *_args) -> None:
        """A format change only takes effect on reopen, so reopen."""
        # The ghost score is told the CONFIGURED rate the capture is opened with
        # (`_current_config`), at the one place that rate can change.
        self.layers["modes"].fps = float(self.spin_fps.value())
        self._schedule_save()
        if self.capture is None:
            return
        self._log("format changed - reconnecting")
        self._disconnect()
        QTimer.singleShot(150, self._connect)

    def _identify_cameras(self) -> None:
        if self.capture is not None:
            self._log("disconnecting first - a camera can only be open once")
            self._disconnect()
        devices = [(c.index, c.name) for c in getattr(self, "_cameras", []) if not c.is_virtual]
        if not devices:
            self._log("no non-virtual cameras found")
            return
        IdentifyDialog(devices, self.cmb_backend.currentText(), self).exec()

    # -- connect / disconnect -------------------------------------------------

    def _current_config(self) -> CaptureConfig:
        w, h = self.cmb_res.currentData()
        fourcc = self.cmb_fourcc.currentText()
        return CaptureConfig(
            index=int(self.cmb_device.currentData()),
            backend=self.cmb_backend.currentText(),
            width=w, height=h,
            fps=float(self.spin_fps.value()),
            fourcc="" if fourcc.startswith("(") else fourcc,
        )

    # THE ONE SEAM THE CAMERA CAN BE REPLACED AT.
    #
    # The capture is a thread that talks to the app through thirteen
    # signals and a dozen methods, and nothing else. That is a clean
    # boundary, so a stand-in can sit behind it and the whole startup
    # chain above it runs for real.
    #
    # Why that is worth a line of indirection: every fault found on
    # hardware on 2026-10-15 was a RACE between the app's own routines --
    # two noise measurements at once, a probe inside a measurement, a self
    # test against a bar from neither. None of them needed real optics to
    # reproduce, and each one cost about seventy seconds of his monitor to
    # find, plus a log read afterwards. Whether they appear at all depends
    # on millisecond timing, so a handful of live launches samples the
    # problem badly: the gate measured 12/12, 0/12, 12/12, 1/12, 12/12 on
    # identical code.
    #
    # With a stand-in the same chain runs headless, in seconds, as many
    # times as it takes, with the timing deliberately jittered -- and it
    # cannot wedge a camera that is not there. See tests/fake_camera.py.
    capture_factory = CaptureThread
    # The same seam for sound. `staticmethod` so the function is not bound as
    # a method; evaluated when this class is defined, so a test that installs
    # a silent mixer before importing this module (tests/owned_files.py) is
    # honoured, and one that sets the attribute afterwards is too.
    audio_factory = staticmethod(default_mixer)
    # And for the microphone. Same shape, same reason: bound when the class is
    # defined, so a silent listener installed on the listener module BEFORE this
    # module is imported is honoured. Without it every harness launch opened the
    # real microphone and its behaviour depended on his room.
    listener_factory = staticmethod(default_listener)
    # Spoken announcements (`lsa/audio/voice.py`). Built over whatever mixer
    # `audio_factory` produced, so a silent test mixer makes this silent too
    # and no second path to a sound device exists. `voice_dir` is a class
    # attribute so a test can point it at a temporary folder, or at a
    # missing one to prove the app runs without any voice files.
    voice_factory = staticmethod(VoiceAnnouncer)
    voice_dir = _AssetDir(lambda: _assets.asset_dir("voice"))      # resolved when read, so LSA_VOICE and a frozen bundle both win

    def _toggle_connect(self) -> None:
        self._disconnect() if self.capture is not None else self._connect()

    def _connect(self) -> None:
        cfg = self._current_config()
        self._log(f"opening camera {cfg.index} on {cfg.backend} "
                  f"at {cfg.width}x{cfg.height} {cfg.fourcc or 'default'} @{cfg.fps:g}...")
        t = self.capture_factory(cfg, self._detector_params(),
                                 self._tracker_config())
        t.opened.connect(self._on_opened)
        t.failed.connect(self._on_failed)
        t.preview_ready.connect(self._on_preview)
        t.shot_detected.connect(self._on_shot_final)
        t.shot_provisional.connect(self._on_shot)
        t.probe_ready.connect(self._on_probe)
        t.control_echo.connect(self._on_echo)
        t.log.connect(self._log)
        t.recording_changed.connect(self._on_recording_changed)
        if self.introspect is not None:
            # The connection is made INSIDE the package, which asks for
            # Qt.QueuedConnection explicitly so the ring is filled on the GUI
            # thread and the capture loop pays nothing. See
            # lsa/introspect/gui.py.
            self.introspect.attach_capture(t)
        # Each of these three asks the Director first. If a routine was
        # waiting for it, the routine gets it and the old handler does not
        # run; if nothing was waiting, this is a measurement belonging to
        # code that has not been migrated yet. That is the whole migration
        # seam, and the reason `deliver_*` returns a bool: without it both
        # paths would run, which is a worse bug than the one being fixed.
        t.noise_measured.connect(self._route_noise)
        t.exposure_chosen.connect(self._on_exposure_chosen)
        t.room_measured.connect(self._route_room)
        t.still_ready.connect(self._on_still_ready)
        t.set_show_mask(self.chk_mask.isChecked())
        t.set_draw_overlay(self.chk_overlay.isChecked())
        self.capture = t
        t.start()
        self.btn_connect.setText("Disconnect  (F5)")
        self._start_listening()

    def _disconnect(self) -> None:
        if self.capture is None:
            return
        self.capture.stop()
        # A short wait, and then let go. The thread may be blocked inside
        # cap.read() on a camera that has stopped answering, in which case it
        # will never return -- and waiting three seconds for it is what turned
        # a stalled camera into an app that would not close, which is what made
        # a force-kill necessary, which is what left the device worse than it
        # started. Qt will reap the thread when the process exits.
        if not self.capture.wait(800):
            self._log("the capture thread is stuck on an unresponsive camera; "
                      "letting go of it rather than freezing the app")
        self.capture = None
        # Bracketed by log lines because the shutdown dies somewhere in
        # here without an exception, without stderr, and without reaching
        # the next line -- which is the signature of a hard crash in
        # native code rather than a Python fault. A crash at shutdown is
        # indistinguishable from a force-kill to the camera, and a
        # force-kill is measured to wedge it: 40.3s to open and 0 frames
        # afterwards, against 0.44s and 53.6 fps after a clean close.
        self._log("shutdown: stopping the microphone")
        self.listener.stop()
        self._log("shutdown: microphone stopped")
        self.btn_connect.setText("Connect  (F5)")
        self.preview.setText("not connected")
        self.preview.setPixmap(QPixmap())
        self._set_connected(False)
        self._log("disconnected")

    def _start_listening(self) -> None:
        """Open the camera's own microphone, not whatever is default.

        The point is a microphone in the room with the gun and the screen. A
        headset on the shooter would work too, but the camera's is the one that
        is certainly present whenever the rest of this app is usable, and it
        needs no extra hardware.
        """
        self.listener.stop()
        if not self.chk_listen.isChecked():
            return
        want = self.cmb_mic.currentData() or ""
        found = pick_device(preferred_name=want,
                            camera_name=self.cmb_device.currentText())
        if found is None:
            self._log("no usable microphone found - corroboration by sound is off")
            return
        dev, name = found
        self.listener.device = dev
        self.corroborator.reset()
        if self.listener.start():
            self._log(f"listening for gunshots on '{name}' (input {dev}). "
                      f"Sound corroborates a shot; it never creates a hit.")
            self._select_mic_in_combo(name)
        else:
            self._log(f"could not open '{name}' for listening: {self.listener.error}")

    def _select_mic_in_combo(self, name: str) -> None:
        for i in range(self.cmb_mic.count()):
            if self.cmb_mic.itemData(i) == name:
                self.cmb_mic.blockSignals(True)
                self.cmb_mic.setCurrentIndex(i)
                self.cmb_mic.blockSignals(False)
                return

    def _play_test_sound(self) -> None:
        """The console's Test button: a gunshot nobody fired, said so."""
        sound = self.cmb_sound.currentText() or "synth shot"
        bang = play_own_sound(self.mixer, self.listener, sound,
                              tail_s=self.SOUND_TAIL_S, source=SOURCE_EFFECT)
        if bang.played:
            self._log(f"  test sound {sound!r} played on "
                      f"{self.mixer.output_name} (not a shot: the microphone "
                      f"was told, so it is {bang.mode} and not counted).")
        else:
            self._log(f"  test sound {sound!r} did not reach a speaker: "
                      f"{bang.why}.")

    def _audio_housekeeping(self) -> None:
        """What the HUD tick does for the microphone besides pairing clicks:
        say what the listener has to say, and tick the teacher. Kept out of
        `_poll_audio` so the stand-ins that run THAT method keep working."""
        if not self.listener.running:
            return
        self._drain_audio_notes()
        self._tick_coupling_teacher()

    def _drain_audio_notes(self) -> None:
        """Say what the microphone side has to say, with its evidence.

        Both failure modes of the coupling are invisible from the firing
        line (RULES C3): measured too HIGH it is silent deafness, which reads
        as "SAW a shot with no gunshot sound: likely a phantom"; measured too
        LOW it is invented trigger pulls. So every reading, every refusal and
        every reset is logged with the value, how many clean readings it
        rests on, what was refused and why, the volume and the sample.
        """
        for kind, text in self.listener.take_notes():
            self._log(f"  microphone: {text}")
            if kind == "implausible":
                # The second line, naming the device: the usual cause of a
                # reading that never clears the room's noise is that the
                # sound is not coming out of anything this microphone can
                # hear -- and the output is chosen by Windows, not by us.
                out = self.mixer.output_name
                vol = self.mixer.master_gain
                self._say_once(
                    f"coupling-implausible|{out}|{vol:.2f}",
                    f"    the gunshot is not audible to the microphone: "
                    f"output {out!r} at volume {vol:.2f}, microphone input "
                    f"{self.listener.device}. Either the sound is not coming "
                    f"out of that device, it is too quiet to reach the "
                    f"microphone, or the stream is dead. Until a clean "
                    f"reading arrives the flat mute applies, which is deaf "
                    f"for 0.8s after each shot.")

    def _tick_coupling_teacher(self) -> None:
        """Teach the microphone how loud the speakers are, while the app owns
        the screen and every detection is its own -- never while he could be
        shooting (RULES B9). Hosted by the self test: it already holds the
        slot for ~10s, so the three sweeps add nothing to the launch.
        """
        mine = bool(self.director.owns_detections()) and not self._grid_active
        if mine and self.director.busy_with() == "selftest":
            self.teacher.begin()
        self.teacher.tick(may_play=mine)

    def _poll_audio(self) -> None:
        """Pair up what was heard with what was seen. Called from the HUD tick.

        The two useful findings are asymmetric and both worth saying out loud:
        heard-but-not-seen means the shot missed the screen or the camera cannot
        see that part of it, and seen-but-not-heard means a visual detection
        with no bang behind it, which is what a phantom hit looks like.
        """
        if not self.listener.running:
            return
        if self._selftest_active or self._modetest_active or self._probe_active:
            # The app is shooting at itself. Every dot is a detection with
            # no bang behind it, which is exactly the phantom signature --
            # true, useless, and sixteen lines long in the first live run.
            # Worse, it reaches the TARGET SCREEN as an audio note, where
            # it reads as the app seeing phantoms while it is testing.
            # Drop what was heard too, so nothing straddles the boundary.
            self.listener.take_clicks()
            self.corroborator.resolve(time.perf_counter())
            return
        clicks = self.listener.take_clicks()
        if self.listener.calibrated:
            # A pull the MICROPHONE heard opens a string for the announcer even
            # when the camera missed it -- the shots he complains about are
            # exactly those. Only once calibrated: before that the threshold
            # is a guess and room noise would hold speech off for no reason.
            # The listener records nothing while muted (our own speech and
            # its tail), so this can never be our own voice.
            for c in clicks:
                self.voice.note_shot(c.t)
        self.corroborator.add_clicks(clicks)
        now = time.perf_counter()
        self._watch_feed(clicks, now)
        findings = self.corroborator.resolve(now)
        _versus_host = self.layers.get("modes")
        if _versus_host is not None:
            # The two-player verdict publishes the microphone's tally beside the
            # score, and must say when the microphone is not calibrated rather
            # than print a zero that reads as "nothing was missed".
            _versus_host.mic_ready = bool(self.listener.calibrated)
            # THE DRY-PULL TALLY reads the corroborator's pairing decisions: the
            # UNION of the camera and the microphone, never their sum. Drained
            # on every poll, before the early return below, so an uncalibrated
            # microphone still empties the queue (and the host then says the
            # figure is a lower bound). Never scored.
            _versus_host.note_pull_outcomes(self.corroborator.take_outcomes())
        if not self.listener.calibrated:
            # The threshold is still a guess, so its findings are worthless and
            # worse than silence: "a shot was heard but not seen" is alarming,
            # and being told it while nobody is shooting teaches the shooter to
            # ignore the one channel that could have explained a real problem.
            return
        times = self.corroborator.last_found_times       # aligned with `findings`
        for n, finding in enumerate(findings):
            if finding == "heard-not-seen":
                if _versus_host is not None:
                    _versus_host.note_heard_unseen()      # a count, never a shot
                click_t = times[n] if n < len(times) else None
                # BOTH the log line and the note on the target screen are
                # worded from the branch the detector's evidence selects. A
                # fixed "it missed the screen, or the camera cannot see that
                # part of it" contradicted its own evidence whenever the dot
                # was on the panel and merely faint (RULES C4).
                log_text, self._audio_note = self._missed_pulse_evidence(click_t)
                self._log("HEARD a shot with no detection: " + log_text
                          + (f"  t={click_t:.3f}" if click_t is not None else "")
                          + "  " + audio_tally(self.corroborator))
            else:
                # A silence only counts against a detection if the microphone
                # was LISTENING: muted by the app's own gunshot, or not
                # delivering audio, it says nothing about whether the pulse
                # was real.
                shot_t = times[n] if n < len(times) else None
                tag = ""
                if shot_t is not None:
                    _pk, _blocks, _deaf = self.listener.audio_at(shot_t)
                    tag = mic_state(_blocks, _deaf)
                    if _versus_host is not None:
                        # The record guard's denominator: a live shot the
                        # microphone could not have heard is no evidence either way.
                        # (A no-op for a dry pull, which is not a live shot.)
                        _versus_host.note_live_unheard(shot_t, bool(tag))
                if (shot_t is not None and _versus_host is not None
                        and _versus_host.is_dry_shot(shot_t)):
                    # A DETECTION ON AN EMPTY MAGAZINE: not the live phantom
                    # wording either way, but the two cases are NOT the same, and
                    # the live log proved it (2026-10-20 12:22): thirty detections
                    # at one screen column with the microphone LISTENING and
                    # silent (mic peak 0.041 against a bar of 0.239) were counted
                    # as dry pulls and, by my first wording, announced as "NOT a
                    # phantom". Only a DEAF microphone explains a missing click.
                    _versus_host.note_dry_unheard(shot_t, bool(tag))
                    if tag:
                        self._audio_note = ("a dry pull was seen (the magazine is empty); "
                                            "the microphone could not have heard it")
                        self._log("SAW a dry pull the microphone could not hear: the "
                                  "magazine was empty and" + tag)
                    else:
                        self._audio_note = ("a detection on an empty magazine was NOT "
                                            "heard by a listening microphone")
                        self._log("SAW a detection on an empty magazine that the "
                                  "microphone, which WAS listening, did not hear: a "
                                  "very quiet dry pull, or not a trigger pull at all "
                                  "(not confirmed)")
                elif tag:
                    self._audio_note = ("a hit was seen; the microphone was not "
                                        "listening then, so no bang is no evidence")
                    self._log("SAW a shot with no gunshot sound, but" + tag)
                else:
                    self._audio_note = ("a hit was seen with NO bang behind it - "
                                        "treat it as a phantom")
                    self._log("SAW a shot with no gunshot sound: likely a phantom "
                              "detection, not a trigger pull")
            self._audio_note_until = time.monotonic() + 6.0
        if self._audio_note and time.monotonic() > self._audio_note_until:
            self._audio_note = ""

    # -- the mark watchdog: a bang was heard and NOTHING appeared --------------
    #
    # See lsa/watchdog.py for the reasoning and every number. Wired here and
    # nowhere else: clicks come from `_poll_audio`, marks from the end of
    # `_on_shot`, and the check is a Director job like every other routine
    # that owns the screen.

    def _watch_feed(self, clicks, now: float) -> None:
        """Hand this poll's microphone clicks to the watchdog, then step it.

        A click is HELD TO ACCOUNT only when it could have been a trigger pull
        the app can draw: the microphone has calibrated (its threshold is a
        guess until three shots corroborate it), the app is not drawing its own
        light, the grid is not being shot (no hole is drawn there by design),
        the floor has been measured against this room, and the camera is
        looking at the app's screen. It deliberately does NOT require a
        mapping: no mapping is one of the failures this exists to name.
        """
        wd = self.watchdog
        if clicks:
            # A DRY pull draws its hole like any shot (2026-10-20), so "a bang heard
            # and nothing appeared" is a fault there too: the watchdog holds every
            # click to account, empty magazine or not. (It briefly did not, while a
            # dry pull was silent and holeless by design.)
            counted = bool(self.listener.calibrated
                           and not self._app_is_drawing_now()
                           and not self._grid_active
                           and not self._sgrid_active
                           and self._floor_measured
                           and self._camera_sees_our_screen())
            for c in clicks:
                wd.note_click(c.t, counted)
        self._watch_step(now)

    def _watch_step(self, now: float) -> None:
        wd = self.watchdog
        wd.resolve(now)
        if self.capture is None or self.target_window is None:
            return
        busy = bool(self.director.busy_with() or self._screen_busy()
                    or self._app_is_drawing_now())
        if not wd.tick(now, busy):
            return
        wd.begin_check(now)
        if self.mapping is None:
            # Nowhere to put a dot, and nowhere to have put a hole: that IS the
            # answer, and it needs no photograph.
            self._on_watch_done({"no_mapping": True})
            return
        job = routines.mark_watch_job(self)
        job.on_done = self._on_watch_done
        if not self.director.start(job):
            wd.abort_check()

    def _watch_plan(self):
        """(spot, brightness, baseline, exposure) for the self-check.

        The probe's own record for the exposure now set: the same spot, the
        brightness it auto-ranged to, and the reading it got there. With no
        probe this launch, the centre spot at FULL brightness, where a healthy
        camera reads far above the absolute bar, and no baseline.
        """
        cur = self._shooting_exposure
        base = None
        for p in self._probe_results:
            if cur is not None and abs(p.exposure - cur) <= 0.5:
                if base is None or abs(p.exposure - cur) < abs(base.exposure - cur):
                    base = p
        spot = self._probe_spot_used or self._probe_spot()
        if base is None:
            return spot, float(self.PROBE_BRIGHTNESS_LADDER[-1]), None, None
        return spot, float(base.brightness), float(base.dot_peak), float(base.exposure)

    def _on_watch_done(self, result) -> None:
        wd = self.watchdog
        now = time.perf_counter()
        if isinstance(result, dict) and result.get("no_mapping"):
            v = watch.Verdict("bad", "there is no calibration (no mapping), so "
                                     "a shot has nowhere to be drawn", True, False)
            wd.end_check(now, v)
            self._log("READING kind=watch no_mapping=yes verdict=bad")
            self._watch_alert(v, "There is no calibration, so nothing can be "
                                 "drawn. Press G.", speak=False)
            return
        if not isinstance(result, dict):
            v = watch.judge(None, None)
            wd.end_check(now, v)
            self._log("READING kind=watch verdict=unmeasurable")
            return
        v = watch.judge(result["peak"], result["baseline"])
        wd.end_check(now, v)
        base = result["baseline"]
        self._log(f"READING kind=watch exposure={result['exposure']} "
                  f"brightness={result['brightness']:.2f} "
                  f"dot_peak={result['peak']:.0f} floor={result['floor']:.0f} "
                  f"baseline={'none' if base is None else format(base, '.0f')} "
                  f"frames={'yes' if result['frames'] else 'NO'} "
                  f"verdict={v.kind}")
        if v.kind == "bad":
            self._watch_alert(
                v, "The camera is not seeing the app's own test dot (it reads "
                   f"{result['peak']:.0f}; "
                   + ("a healthy camera reads far more). " if base is None else
                      f"it read {base:.0f} earlier this session). ")
                   + "Stop firing, check it has not moved or lost its cable, "
                     "and restart the app by closing its window.")

    def _watch_alert(self, v, text: str, speak: bool = True) -> None:
        """Bang heard, nothing drawn, and the self-check says the camera is bad.
        Spoken as well, when an announcer is attached."""
        self._log("  WARNING: mark watchdog: " + v.why)
        try:
            self._alert("NOTHING IS BEING DRAWN FOR YOUR SHOTS", text,
                        kind="warn", seconds=20.0)
        except Exception:                                    # noqa: BLE001
            pass
        voice = self.voice
        if speak and voice is not None:
            try:
                voice.say("v_camera_bad")
            except Exception as exc:                         # noqa: BLE001
                self._log(f"  the spoken line v_camera_bad failed: {exc!r}")

    def _missed_pulse_evidence(self, click_t: "float | None") -> "tuple[str, str]":
        """What the detector almost saw within 0.25 s either side of a click.

        Separates "too faint" (a sub-threshold blob, with its position and the
        bar it fell short of), "too short" (rejected runs) and "not on the
        panel at all" (the camera saw nothing at any level), which need three
        different answers and were one indistinguishable line. Read from the
        capture's per-frame buffer rather than from `Stats`, whose one-second
        maxima smear everything near the click together.

        Returns `(log text, on-screen note)`, both worded from the same branch.
        When there is no evidence to choose a branch from, both say so and
        neither claims a cause.
        """
        plain = "a shot was HEARD but not seen"
        if click_t is None:
            return "(no click time for this finding)", plain
        try:
            cap = self.capture
            if cap is None:
                return "(no camera, so no detector evidence)", plain
            summary = cap.evidence_between(click_t - 0.25, click_t + 0.25)
            bar, thr = float(self.sp_minpeak.value()), float(self.sp_threshold.value())
            return ("detector evidence within 0.25 s of the click: "
                    + describe_missed_pulse(summary, bar, thr),
                    missed_pulse_note(summary, bar, thr))
        except Exception as exc:
            return f"(detector evidence unavailable: {exc!r})", plain

    def _setup_wanted(self) -> bool:
        """Should the automatic darken-and-calibrate chain run?

        Gated on a session flag as well as the checkbox, and NOT by unticking
        the checkbox for the duration -- that gets written to settings.json on
        exit, so an aim-mode launch silently turned off calibration for every
        launch afterwards. The same trap as a test writing the user's config:
        a temporary UI change is still a change.
        """
        return self.chk_autocal.isChecked() and not self.auto_aim

    def _enter_aim_on_start(self) -> None:
        """Open straight into the view used to place the camera physically."""
        if self.target_window is None:
            self._toggle_target_mode()
        QTimer.singleShot(500, self._toggle_aim_view)

    def _on_opened(self, info: dict) -> None:
        # The camera is engaged: the splash comes down BEFORE the setup
        # chain below begins any measurement.
        self._splash_down("the camera opened", urgent=False)
        self.lbl_negotiated.setText(
            f"{info['width']}x{info['height']}  {info['fourcc']}  @{info['fps']:g} fps  "
            f"(backend {info['backend']})"
        )
        asked = info["requested"]
        if (info["width"], info["height"]) != (asked["width"], asked["height"]):
            self._log(f"NOTE: asked for {asked['width']}x{asked['height']}, "
                      f"got {info['width']}x{info['height']}")
        # MSMF genuinely does not report a FOURCC, so a blank answer there is
        # not a mismatch and should not be announced as one.
        got_codec = str(info["fourcc"])
        if asked["fourcc"] and got_codec != asked["fourcc"] and "not reported" not in got_codec:
            self._log(f"NOTE: asked for codec {asked['fourcc']}, got {got_codec}")
        for key, value in info["controls"].items():
            if key in self.rows:
                self.rows[key].set_value_silently(value)
                self.rows[key].set_echo(value, True)
        if info["height"]:
            self.layers["hits"].set_aspect(info["width"] / info["height"])
        self._set_connected(True, f"{info['width']}x{info['height']} @{info['fps']:g} fps")
        self._schedule_save()

        if not self._setup_wanted():
            self._log("camera open. Click 'Laser hunt (dark)', then 'Measure noise floor'.")
            return

        # Hand over a working detector rather than a pile of sliders. Three steps,
        # all measured: stabilise the camera, find the exposure that suits this
        # room, then set the threshold from what it actually sees.
        self._set_status(
            "SETTING UP - do not fire",
            "Stabilising the camera, measuring exposure, then the noise floor.",
            "#3a2f16", "#e8d48a", "#7a6a2a",
        )
        # THE SCREEN SEARCH STARTS OVER FOR EACH CAMERA, so its attempt counter
        # belongs to the camera and must be reset here.
        #
        # Found on 2026-10-20 by running the frozen build cold on his machine.
        # `_calib_exposures` holds four candidates and `_calib_attempt` walks
        # them; when all four fail the app concludes this camera cannot see the
        # screen and TRIES THE NEXT ONE -- which re-enters the search with the
        # counter still sitting at 4, indexes past the end of the list, and
        # kills setup with an IndexError. The app survives and keeps detecting,
        # so it looks alive while being permanently unusable: no screen search,
        # no mapping, no holes.
        #
        # It needs two cameras to reproduce, the first of which cannot see the
        # screen -- which is the ORDINARY case for a new user (a laptop webcam,
        # or a virtual camera sitting at index 0 that opens fine and delivers
        # pictures of nothing). It never fires here, because this bench has a
        # remembered camera that works.
        self._calib_attempt = 0
        self._calib_black = None
        self._begin_after_welcome()

    def _begin_after_welcome(self) -> None:
        """Let the welcome line finish before ANYTHING owns the screen.

        He asked twice: "Give one extra second to the flash screen so that
        the guy can say Welcome to Screenplink. Currently he is cut off at
        the last syllable."

        HOLDING THE SPLASH LONGER CANNOT DO IT, and that was my first
        attempt. The splash is his startup art -- red excess 218 at the
        laser dot over the "i" -- so any measurement must take the screen
        from it, and the log said exactly that: "splash down (a measurement
        owns the screen)", one second into a 1.25s line. The thing to delay
        is the MEASUREMENT, not the splash.

        One bounded timer into a DIFFERENT method, never a self-rescheduling
        one: `_start_noise_floor` re-armed itself every 700ms for a day.
        """
        left = 0.0
        if self.voice is not None:
            # The measurement waits for the line AND for the extra second the
            # splash is meant to dwell -- otherwise the measurement takes the
            # screen the moment he stops speaking and pulls the splash down
            # urgently, which is the hold being granted and then cancelled.
            left = min(self.voice.speaking_until() + self.SPLASH_EXTRA_S
                       - time.perf_counter(),
                       self.SPLASH_HOLD_CAP_S)
        if left > 0.0:
            self._log(f"  holding setup {left:.1f}s so the welcome line can "
                      f"finish; the splash stays up meanwhile.")
            QTimer.singleShot(int(left * 1000) + 1, self._begin_screen_search)
            return
        self._begin_screen_search()

    def _on_failed(self, msg: str) -> None:
        self._log("ERROR: " + msg)
        # Nothing is measuring when the camera fails to open, so there is
        # nothing for the splash to poison: keep it up and say why.
        self._splash_problem("The camera could not be opened. " + str(msg))
        self.capture = None
        self.btn_connect.setText("Connect  (F5)")
        self._set_connected(False, "last attempt failed")

    # -- live updates ---------------------------------------------------------

    def _on_preview(self, pv: Preview) -> None:
        self._frames_ever += 1
        # Keep the most recent preview image, so the verdict can save what
        # the camera was actually looking at when it decided. Read through
        # getattr elsewhere would have returned None silently -- the exact
        # trap this project has hit twice with `stats.last_shot_at` and
        # `capture.stats` -- so it is assigned here, unconditionally.
        self._last_preview = pv

        if self.chk_safe_preview.isChecked():
            # GREYSCALE, not blue. Both are safe: the detector scores
            # red - max(green, blue), and in greyscale all three channels are
            # equal, so that difference is exactly zero -- the same protection
            # the blue multiply gave against the camera seeing a laser dot in
            # its own reflected feed.
            #
            # But only one of them is legible, and this preview is how a person
            # aims the camera at the screen in the first place. A blue-washed
            # image makes that impossible, which is the opposite of helping.
            # The blue multiply survives for RED and GRAY score modes, where
            # brightness alone counts and greyscale would not protect us.
            if str(self._current_mode()).upper().endswith("CHROMA"):
                pm = QPixmap.fromImage(
                    pv.image.convertToFormat(QImage.Format_Grayscale8))
            else:
                pm = QPixmap.fromImage(pv.image)
                painter = QPainter(pm)
                painter.setCompositionMode(QPainter.CompositionMode_Multiply)
                painter.fillRect(pm.rect(), QColor(0, 70, 255))
                painter.end()
        else:
            pm = QPixmap.fromImage(pv.image)
        if self._aim_help:
            self._draw_aim_guides(pm, pv.frame_size, pv.bright_box)
        self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation))
        if pv.mask is not None:
            self.mask_label.setPixmap(QPixmap.fromImage(pv.mask).scaled(
                self.mask_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        elif self.chk_mask.isChecked():
            self.mask_label.setText("mask pending")
        else:
            self.mask_label.setPixmap(QPixmap())
            self.mask_label.setText("mask off")

        # Feed the target screen's camera pane whether or not it is visible, so
        # switching it on shows a live frame immediately rather than a stall.
        pane: CameraPane = self.layers["camera"]
        pane.set_frame(pv.image, f"index {self.cmb_device.currentData()}  "
                                 f"{pv.frame_size[0]}x{pv.frame_size[1]}")
        if getattr(pane, "aiming", False):
            # The LIVE bright region, not the quad from the last flash search.
            # That quad is a photograph taken before the camera was touched, so
            # the instant it is picked up the outline is wrong while still
            # looking authoritative. This one follows the monitor.
            box = pv.bright_box
            poly = None
            if box:
                bx, by, bw, bh = box
                poly = ((bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh))
            elif self.screen_quad is not None:
                poly = self.screen_quad.polygon
            pane.set_aiming(True, poly, pv.frame_size)
            pane.live_outline = box is not None
            self.scene.mark_dirty()

        s = pv.stats
        pos = ("  ".join(f"({d.x:7.2f},{d.y:7.2f})" for d in pv.detections[:3])
               if pv.detections else "none")
        self.readout.setText(
            f"frame      {pv.frame_size[0]}x{pv.frame_size[1]}\n"
            f"fps        {s.fps:6.2f}   read {s.read_ms:5.1f} ms   detect {s.process_ms:5.2f} ms\n"
            f"frames     {s.frames}   dropped {s.dropped}\n"
            f"detected   {s.frames_with_detection}  ({s.detection_rate * 100:5.1f}% of frames)"
            f"   dots now {s.candidates}\n"
            f"dots       {pos}\n"
            f"brightest  peak {s.peak:3d}   area {s.area:7.1f} px   shape {s.elongation:4.1f}:1\n"
            f"SHOTS      {s.shots}   mean {s.mean_shot_frames:4.2f} frames/shot"
            f"   pulse ~{s.mean_pulse_ms:6.1f} ms\n"
            f"tracking   {s.open_tracks} open   most at once {s.max_concurrent_tracks}"
            f"   most dots in a frame {s.max_dets_in_frame}\n"
            f"rejected   {s.rejected_runs} too short, {s.rejected_too_long} too long\n"
            f"recording  {'YES ' + str(s.recorded_frames) + ' frames' if s.recording else 'no'}"
        )
        self._warn_if_slow(s)
        self._refresh_hitrate()

    def _warn_if_slow(self, s) -> None:
        """A low frame rate silently breaks multi-frame shot confirmation."""
        if s.frames < 60 or self.capture is None or self._calibrating:
            return
        slow = s.fps < 20.0
        if slow == getattr(self, "_slow_warned", None):
            return
        self._slow_warned = slow
        if slow:
            self.lbl_status.setText(
                f"CONNECTED but only {s.fps:.1f} FPS"
                "\nToo slow: a 100 ms pulse needs 25+ FPS to span the 2-3 frames"
                "\nthat shot confirmation relies on. Try the MSMF backend with MJPG."
            )
            self.lbl_status.setStyleSheet(
                "background:#322e16; color:#c8c070; border:1px solid #5e5a2a;"
                "padding:6px; font-weight:bold;")
            self._log(f"WARNING: only {s.fps:.1f} fps. Switch backend to MSMF with MJPG.")
        else:
            self._set_connected(True, f"{s.fps:.1f} FPS")

    def _on_shot_final(self, shot: Shot) -> None:
        """The refined position, a few frames after the hole was already drawn.

        Everything the user perceives already happened on the provisional
        report. All that is left is to nudge the hole to its final position.
        """
        mapped = self._map_to_screen(shot.x, shot.y)
        if mapped is None:
            return
        sx, sy = mapped
        if self.layers["holes"].move(shot.index, sx, sy):
            self.scene.mark_dirty()

    # How long a cover check is reused. The question is "what is on top of
    # the monitor", which changes at human speed, and the answer is wanted
    # on every detection -- during the desktop storm that was seven times a
    # second.
    COVER_CACHE_S = 0.25
    # How long a cover must last before the PRINTED banner is shown, and how often
    # the log repeats an unchanged cover. A dictation panel that pops up while he
    # speaks is gone in a second or two; the guard still suppresses at once (that
    # is the point of it), but a banner flashing on and off mid-string "messes up
    # gameplay". The voice line follows the same delay.
    COVER_BANNER_AFTER_S = 2.5
    COVER_REPORT_S = 120.0

    # How long the window must have answered "in front" before detections
    # count again. WHAT THIS GUARANTEES, exactly: no False answer has been
    # OBSERVED in the last COVER_SETTLE_S. It does NOT guarantee the window
    # was in front continuously, for two reasons: `_cover_true_since` only
    # advances when something calls `_camera_sees_our_screen`, and the only
    # polling caller returns early while the grid is active or the capture
    # is None, so a False between two calls can go unseen. Both gaps fail
    # OPEN (a real shot is never lost to them), which is why the mechanism
    # stands and only the claim needed correcting.
    # Going the other way is immediate, to within COVER_CACHE_S: a True
    # cached just before the window went behind can admit one or two
    # detections before the next real check. 2026-10-16 18:45:33-54, at the desktop with the
    # microphone silent throughout, the check flipped four times in 25
    # seconds and each True window admitted desktop detections as shots --
    # thirteen of them. Cost, accepted: his first shot within a second of
    # alt-tabbing back is suppressed, and the log line reports the count.
    COVER_SETTLE_S = 1.0

    def _camera_sees_our_screen(self) -> bool:
        """Is the camera looking at the app's own background, or the desktop?

        THE THIRD CONDITION OF RULES B9, added 2026-10-16 after a storm of
        gunshots he described as "continuous shots that just do not stop.
        This is completely unacceptable."

        Escape lowers the target window rather than closing it, which is
        right and rule-protected. But then the camera watches a lit desktop
        while the detector runs on a bar measured against a near-black
        screen, and 92 detections arrived in twenty seconds at peak 9-29
        where his real shots measure 85-146. Every one rang the gunshot and
        was offered to the sensitivity learner.

        One function, consulted by every consumer, rather than a condition
        added to each: a guard that names the callers its author thought of
        is how `SCREEN_OWNERS` ended up read by one routine in six
        (RULES E2).

        This also covers two cases nobody would have listed: the fullscreen
        flash panel that the screen search puts over this window -- whose
        sixteen detections once discarded a room measurement -- and the
        desktop before the target window has opened at all, which is where
        fourteen bangs in one second came from.
        """
        win = self.target_window
        if win is None:
            # NEVER OPENED is the operator-console flow (`--no-target`, hits
            # read off the camera pane): not suppressed, and the dangerous
            # case -- light the app cannot vouch for arriving before setup is
            # done -- is held by the floor-and-mapping half of
            # `_ready_for_real_shots`, where the fourteen bangs were stopped.
            #
            # CLOSED is the opposite: the camera is then looking at his
            # desktop. At 18:45:57 on 2026-10-16 three detections arrived
            # after the window closed, were treated as real shots, rang the
            # gunshot and fed the border that collapsed to 22%.
            if self._target_ever_opened:
                # So that a re-opened window starts from "not yet settled".
                self._cover_ok = False
                self._cover_true_since = None
                self._cover_causes.add("closed")
                if not self._cover_closed_noted:
                    # Once per close, never per detection (the project has
                    # had a four-hundred-identical-lines incident). F11
                    # closes the window, and without this line every shot
                    # after it vanished in silence (RULES B9).
                    self._cover_closed_noted = True
                    self._log("  IGNORING detections: the target screen is "
                              "closed, so the camera is watching the "
                              "desktop. No gunshot, no hole, nothing "
                              "learned. Press F11 to open it again; the "
                              "count ignored is reported when it counts "
                              "again.")
                return False
            return True
        self._cover_closed_noted = False
        now = time.perf_counter()
        if now - self._cover_checked_at < self.COVER_CACHE_S:
            return self._cover_ok
        self._cover_checked_at = now
        raw = bool(win.in_front_of_camera())
        if not raw:
            # Immediately, and the run of True answers starts again.
            self._cover_true_since = None
            self._cover_causes.add("covered")
            ok = False
        else:
            if self._cover_true_since is None:
                self._cover_true_since = now
            ok = now - self._cover_true_since >= self.COVER_SETTLE_S
            if not ok:
                self._cover_causes.add("settling")
        if ok != self._cover_ok:
            # Say it once per transition, with the measurement and the count
            # of what was thrown away, so the silence is explained rather
            # than mysterious (RULES C3). A suppression nobody can see is
            # indistinguishable from a detector that has stopped working.
            self._cover_active = not ok
            if ok:
                why = {"closed": "the screen was closed",
                       "covered": "the screen was covered",
                       "settling": "the screen was still settling in "
                                   "front of the camera"}
                reasons = ", ".join(why[c] for c in
                                    ("closed", "covered", "settling")
                                    if c in self._cover_causes) or "unknown"
                self._log(f"  the target screen is in front of the camera "
                          f"again. {self._cover_suppressed} detection(s) were "
                          f"ignored while it was not ({reasons}); none of "
                          f"them counted as trigger pulls.")
                self._cover_suppressed = 0
                self._cover_causes.clear()
            else:
                self._log(f"  IGNORING detections: {win.cover_reason}. "
                          f"Nothing it sees now can be a trigger pull, so no "
                          f"gunshot, no hole and nothing learned from it. "
                          f"Alt-tab or click the taskbar to bring it back.")
                self._cover_since = now
                self._cover_reported = (win.cover_by, now)
                if self._screen_busy() or self.director.busy_label():
                    # Mid-measurement a cover makes every number meaningless,
                    # and the Director's own jobs only check before they
                    # start. Bounded, and never when he is just playing.
                    self._reclaim_screen("something covered it mid-measurement")
        elif not ok and not raw:
            # STILL COVERED. Said once per transition above; again only when WHAT
            # is covering the screen changes, or after COVER_REPORT_S -- never on
            # every check for as long as the condition lasts (he reported the
            # message "showing for several seconds in a row").
            by, at = self._cover_reported
            if win.cover_by != by or now - at >= self.COVER_REPORT_S:
                self._log(f"  still ignoring detections: {win.cover_reason}.")
                self._cover_reported = (win.cover_by, now)
        self._cover_ok = ok
        return ok

    def _ready_for_real_shots(self) -> bool:
        """Could this detection possibly BE a trigger pull?

        THREE things have to be true before the answer is yes, and none of
        them is about which routine happens to be running:

        - the noise floor has been measured against THIS room, so the
          threshold is a measurement rather than a value left over from
          another day and another light level;
        - there is a mapping, so the hit has somewhere to go. A detection
          with nowhere to draw it is not a shot, it is a rumour;
        - the camera is looking at the app's own background. The first two
          were both TRUE during the desktop storm of 2026-10-16, which is
          precisely why it fired 92 times. See `_camera_sees_our_screen`.

        The grid is the exception: it is shot with no mapping on purpose,
        because it is the thing that makes one.
        """
        if self._grid_active:
            return True
        return bool(self._floor_measured and self.mapping is not None
                    and self._camera_sees_our_screen())

    def _play_gunshot(self, override: "str | None" = None, dry: bool = False,
                      cue: "str | None" = None) -> None:
        """Play the sound for one real trigger pull and mute the microphone for it.

        `override` is a mode's own sample for this shot (the figure that drops).
        It is used only if it is loaded; otherwise the ordinary gunshot plays, so
        a missing file costs the effect and never the bang.

        `dry`: the magazine is empty, so the bubble pop (`DRY_SOUND`) plays INSTEAD
        of the gunshot (his rule, 2026-10-20). `cue`: a selection cue that follows
        the sound by a millisecond, in addition to it (the MENU button, the NEXT
        ROUND box, the menu's tiles).
        """
        sound = DRY_SOUND if dry else (self.cmb_sound.currentText() or "synth shot")
        if dry and not self.mixer.has(DRY_SOUND):
            # A missing dry-pop sample is not allowed to turn a dry pull into a
            # gunshot (that is the lie the pop exists to avoid): say so once and
            # play nothing.
            self._say_once("dry-sound-missing",
                           "  the dry-pull sample is not loaded, so dry pulls are "
                           "silent. They are NOT played as a gunshot.")
            return
        if override is not None and not self.mixer.has(override):
            self._say_once(
                "effect-sound-missing",
                f"  the effect sample {override!r} is not loaded, so only the "
                f"ordinary gunshot plays. A missing file costs the effect and "
                f"never the bang.")
            override = None
        # PLAY IT AND TELL THE MICROPHONE, in one place (audio/bang.py).
        #
        # The microphone is in the same room as the speakers, so without this
        # the app hears its own bang and reports twice the shots that were
        # fired. What matters at THIS call site: a bang that did not reach a
        # speaker is not registered with the listener. The previous version
        # logged "play() refused it" and then muted the microphone anyway,
        # which filed noise as a measurement of the room. A dry pull's pop is
        # an effect, never a reading of the gunshot's coupling.
        bang = play_own_sound(self.mixer, self.listener, sound,
                              tail_s=self.SOUND_TAIL_S,
                              source=SOURCE_EFFECT if dry else SOURCE_BANG)
        if not bang.played:
            self._say_once(
                "nosound-play",
                f"  the gunshot {sound!r} was asked for but did not reach a "
                f"speaker: {bang.why}. PortAudio does not raise when a "
                f"callback stream's endpoint has gone, so this is the only "
                f"place it can show. The microphone was told nothing was "
                f"played, so this is not a measurement.")
        else:
            self._say_once(
                "sound-ok",
                f"  gunshot {sound!r} played on "
                f"{self.mixer.output_name}. Said once, "
                f"so a silent channel and a working one are "
                f"distinguishable in this log.")
            if bang.mode == OWN_UNPROTECTED:
                self._say_once(
                    "bang-unprotected",
                    "  the gunshot was neither discounted nor muted: "
                    "the microphone will hear it as a trigger pull.")
        # HOW THE MICROPHONE IS KEPT FROM HEARING THIS BANG is decided inside
        # `play_own_sound` / `ClickListener.expect_own_sound`, and the history
        # lives with the code that does it (audio/bang.py, audio/listener.py):
        # the whole-sample mute (1.75s of deafness per shot, 87% of the
        # 'HEARD a shot with no detection' reports within three seconds of a
        # detection on 2026-10-15), the loud-part mute scaled by the volume
        # (0.82s, still longer than the 0.4-0.8s between his shots: 55-65% of
        # every shot he fires was never heard, measured 2026-10-23), and now
        # discounting the bang by a MEASURED coupling. The flat mute remains
        # only as the fallback until that coupling is known.

        # THE EFFECT FOLLOWS THE BANG. IT DOES NOT REPLACE IT.
        #
        # His correction, 2026-10-18: "That sound should not be instead of
        # the gun sound. That sound should be directly after the gun sound.
        # In addition." I had built every one of them as a substitution, so
        # a figure dropping or a ball bursting swallowed the gunshot that
        # caused it -- and the gunshot is the shooter's confirmation that
        # the app saw the trigger pull at all.
        #
        # Delayed by the bang's own LOUD part, measured rather than guessed,
        # so the two read as two events instead of one muddy overlap.
        extras = [e for e in (override, cue) if e is not None and self.mixer.has(e)]
        if extras:
            # THE NEXT MILLISECOND, not after the bang has finished.
            #
            # His correction, 2026-10-18: "insert the sound of the
            # disappearing figure as soon as the last shot plays, like the
            # next millisecond. Otherwise the figure disappears and then I
            # hear the sound." I delayed by the gunshot's whole LOUD part,
            # about half a second, so the effect arrived after the figure had
            # already gone and read as an echo rather than as the event.
            #
            # They overlap now, which is what a gunshot and the thing it
            # destroys actually do.
            gap = 0.001
            for extra in extras:
                # The effect registers ITSELF with the microphone when it plays
                # (`_play_effect`), muted or discounted like the bang: the app
                # must not hear its own effect and count it as a trigger pull
                # on the one channel that is independent of the camera.
                QTimer.singleShot(int(gap * 1000) + 1,
                                  lambda s_=extra: self._play_effect(s_))

    def _play_effect(self, sample: str) -> None:
        """A sound that is NOT a trigger pull.

        Two callers: the effect that follows a bang (above), and a target
        APPEARING, which no shot caused at all -- so this must never play a
        gunshot of its own. Muted on its own terms, because the microphone
        cannot tell the app's speaker from his pistol.
        """
        if not self.mixer.has(sample):
            return
        bang = play_own_sound(self.mixer, self.listener, sample,
                              tail_s=self.SOUND_TAIL_S, source=SOURCE_EFFECT)
        if not bang.played:
            self._say_once(
                f"effect-not-played|{sample}",
                f"  the effect {sample!r} was asked for but did not reach a "
                f"speaker: {bang.why}. The microphone was told nothing was "
                f"played.")

    def _gunshot_mute_s(self, sound: str) -> float:
        """How long the microphone is deaf after playing `sound` at the current
        volume WHILE THE COUPLING IS NOT KNOWN: the fallback, and the number the
        log quotes. One definition, in audio/bang.py, shared with the play path."""
        return fallback_mute_s(self.mixer, sound, self.SOUND_TAIL_S)

    def _on_shot(self, shot: Shot) -> None:
        # Recorded first -- the noise-floor guard needs to know a shot
        # arrived even for shots later discarded, routed to the grid, or
        # swallowed by the self test -- but NOT when the light was the
        # app's own.
        #
        # It used to be unconditional, and the cost was measured on his
        # hardware at 08:47:41. Flashing the monitor white to find the
        # screen produced SIXTEEN detections at peak 11-18 and area 3-16px
        # (a real shot here is peak 85, area 296px), every one of them
        # stamped as "a shot just landed" -- so the room measurement that
        # followed was discarded as spoiled:
        #
        #   discarding that room measurement: a shot landed during it, so
        #   it measured the laser and not the room
        #
        # The app told itself to distrust a measurement because of light it
        # had drawn itself. The guard is right; its input was wrong. This
        # is the same discriminator error as the gunshot sound -- "did a
        # detection arrive" instead of "could this have been a trigger
        # pull" (RULES B9).
        if not self._app_is_drawing_now() and self._camera_sees_our_screen():
            self._last_shot_at = time.perf_counter()

        # IS THIS A TRIGGER PULL, OR THE APP'S OWN LIGHT?
        #
        # Asked directly, 2026-10-15: "there were way too many gunshot
        # sounds. I don't understand why there were that many gunshot
        # sounds during the calibration phase."
        #
        # Because the sound was played BEFORE anything decided whose shot
        # it was. Everything below this point routes a detection -- to the
        # grid, to the self test, to the exposure probe -- but the bang,
        # the microphone learning and the audio corroboration all happened
        # first, unconditionally. So every time the app flashed the monitor
        # white to find the screen, or drew a red dot to test itself, it
        # fired a gunshot at him. Measured on one run: 94 detections, of
        # which about four were his.
        #
        # Worse than the noise: those detections also taught the microphone
        # what 'a shot' sounds like, from recordings of the app's own
        # speaker. The grid is deliberately NOT in this list -- he is
        # really shooting during it, and the feedback is the point.
        self._app_is_drawing = bool(
            # ONE FIELD, from the running routine's own declaration.
            #
            # This was an OR of six flags, and it had to be kept in step by
            # hand in three other places as well. Every routine added later
            # needed four edits in four files and never got all four: the
            # self test did not check the probe, the probe did not check the
            # self test, and `closeEvent`'s list of eight omitted the
            # two-marker check entirely. A routine declares
            # `consumes_detections` once, where it is defined.
            self.director.owns_detections()
            # Not yet migrated onto the Director; these come out as each one
            # moves. The list shrinks, and cannot silently grow -- a new
            # routine has nowhere to put a flag.
            or self._pat_active or self._modetest_active
            or self._calibrating
            # ...OR THE APP IS NOT SET UP YET.
            #
            # "Why do I constantly hear the gunshots?" -- still, after the
            # phases above were silenced. Because the loudest source was
            # never a phase at all. Before the noise floor has been
            # measured, the detector runs on whatever threshold was saved
            # last time, and against a different room that threshold is a
            # guess. Measured on one launch:
            #
            #     17:34:17  SHOT 1 .. SHOT 14   peak 8-18, area 3-18px
            #
            # Fourteen bangs in a single second, before calibration began,
            # every one of them noise -- a real shot on this hardware
            # measures peak 85 and area 296px. The app logs them as
            # "(uncalibrated)" because it knows there is nowhere to draw
            # them, and then fires a gunshot anyway.
            #
            # So the rule is not "which phase is running" but "does this
            # detection mean anything yet". It does not until the floor
            # has been measured against this room AND there is a mapping
            # to place it with -- except during the grid, where he is
            # really shooting and the feedback is the whole point.
            or not self._ready_for_real_shots())
        if self._app_is_drawing:
            # Routed below (grid, probe, self test) or dropped there; either
            # way silently, and with no audio bookkeeping.
            pass
        else:
            if self._not_pull_ignored:
                self._log(f"  {self._not_pull_ignored} detection(s) were ignored "
                          f"while the app was not treating detections as "
                          f"trigger pulls; none spent a round or scored. "
                          f"Shots count again.")
                self._not_pull_ignored = 0
            self.corroborator.add_shot(shot.last_timestamp)
        mic_txt = ("off" if not self.listener.running else
                   "not consulted (this detection is not treated as a trigger pull yet)")
        if self.listener.running and not self._app_is_drawing:
            was = self.listener.calibrated
            peak = self.listener.note_seen_shot(shot.last_timestamp)
            _pk, _blocks, _deaf = self.listener.audio_at(shot.last_timestamp)
            mic_txt = (f"{peak:.3f} (bar {self.listener.min_level:.3f}"
                       + ("" if was else ", mic not calibrated yet") + ")"
                       + mic_state(_blocks, _deaf))
            if not was and self.listener.calibrated:
                self._log(f"microphone calibrated from {self.listener.LEARN_NEEDED} "
                          f"confirmed shots: a shot measures {self.listener.learned_level:.3f}, "
                          f"so anything above {self.listener.min_level:.3f} counts as one. "
                          f"Heard-but-not-seen reporting is now on.")
            elif not was and peak:
                self._log(f"microphone: that shot measured {peak:.3f} "
                          f"({len(self.listener._learned)}/{self.listener.LEARN_NEEDED} "
                          f"needed before sound is trusted)")
        # WHY THERE WAS NO BANG, SAID ONCE PER REASON.
        #
        # 2026-10-16, after a string in which every shot was detected and
        # drawn: "I did not hear any sounds. Did that part get broken
        # again?" The log held the answer to every other question about that
        # string and not to this one -- audio initialised on his monitor,
        # `sound` was true, the corroborator was treating his shots as real,
        # and there was nothing anywhere saying whether a gunshot had been
        # asked for, played, or refused.
        #
        # Four things can silence it and they need different fixes: the
        # checkbox, a mixer that never opened, the trigger-pull gate
        # upstream, and a play() that fails. A feedback channel that fails
        # silently is the worst kind on this project -- from the firing line
        # it reads as the shot not being detected at all.
        if not self.chk_sound.isChecked():
            self._say_once("nosound-off",
                           "  no gunshot: the sound checkbox is off.")
        elif not self.mixer.available:
            self._say_once("nosound-mixer",
                           "  no gunshot: no audio output was opened.")
        elif self._app_is_drawing:
            self._say_once(
                "nosound-notashot",
                "  no gunshot: this detection is not treated as a trigger "
                "pull yet (the floor, the mapping, the screen, or a routine "
                "owning the detections). See `_ready_for_real_shots`.")
        if not self._app_is_drawing:
            # A REAL trigger pull: the announcer must not speak into a string.
            # The same predicate that decides whether to bang (RULES B9), so
            # the app's own dots never open or extend a "string".
            self.voice.note_shot(time.perf_counter())
        # THE BANG IS DECIDED AFTER THE MODE HAS SEEN THE SHOT. The shot that
        # drops a figure plays a sample of its own (`ShotOutcome.sound`), and
        # only the mode knows that. Every path that returns before the mode is
        # asked (the grid, a covered screen) calls `bang()`
        # itself, so none of them loses its gunshot; the main path calls it
        # with the mode's answer. `bang` is idempotent: one shot, one sound.
        bang_due = bool(self.chk_sound.isChecked() and self.mixer.available
                        and not self._app_is_drawing)
        _banged = [False]

        def bang(override: "str | None" = None, dry: bool = False,
                 cue: "str | None" = None) -> None:
            if bang_due and not _banged[0]:
                _banged[0] = True
                self._play_gunshot(override, dry=dry, cue=cue)

        if self._grid_active:
            bang()
            self._grid_record(shot)
            return

        if self._sgrid_active and self._camera_sees_our_screen():
            # The brightness grid (U). Routed HERE, ahead of the mode, which is
            # the whole of "its shots score nothing, spend no magazine round,
            # draw no hole and teach the play-time learner nothing": none of
            # those consumers is reached below this return. The gunshot is
            # kept, as for the shot grid -- he really is shooting, and the
            # bang is his confirmation that the app saw the pull. Only while
            # the camera is looking at OUR screen: a detection of his desktop
            # must not become a spot (it falls through to the cover branch).
            bang()
            self._sgrid_record(shot)
            return

        if self._probe_active:
            # The exposure probe is painting a red dot on purpose and
            # photographing it directly. Its dot is not a shot: letting
            # it through drew holes, played gunshots, moved the score,
            # and spoiled the noise-floor measurement that follows by
            # making the DOT the floor -- measured at peak 179 on the
            # first live run, against a room that scores 2.
            return

        if self._selftest_active:
            # The dot the screen is drawing right now. Keep the BRIGHTEST
            # report for this position: the tracker can split one steady dot
            # into more than one short track, and the strongest is the one
            # that actually saw it.
            mapped = self._map_to_screen(shot.x, shot.y)
            if mapped is not None:
                prev = self._selftest_seen
                if prev is None or shot.peak > prev[1]:
                    # The pulse duration is the DETECTOR's answer to "how long
                    # was it actually on screen", which is not the same as how
                    # long we asked for. Reported, not assumed: the shooter
                    # said the dots "disappear even quicker than mine" and the
                    # code had no idea whether that was true.
                    self._selftest_seen = (mapped, float(shot.peak),
                                           float(shot.area),
                                           float(shot.duration_ms))
            return

        if not self._camera_sees_our_screen():
            # The camera is not looking at this app's screen, so this blob is
            # something on his desktop. Everything below draws a hole, moves
            # the score, counts a shot and teaches the sensitivity map -- and
            # during the storm of 2026-10-16 all four happened 92 times in
            # twenty seconds. Counted and reported by
            # `_camera_sees_our_screen`, never silently dropped.
            #
            # Placed AFTER the grid, probe and self-test routing above on
            # purpose: those three own the screen while they run, and a
            # routine's own dots must still reach it.
            self._cover_suppressed += 1
            bang()
            return

        sx = sy = None
        dry = False
        silent = False
        scored: bool | None = None
        points = 0
        target_id = None
        near = None
        # THE END OF THE CHAIN (lsa/watchdog.py). Did anything actually get put
        # on the scene for this shot? Decided from what the drawing code did,
        # never from what it meant to do: a hole counts only if the holes layer
        # is visible and its own monotonic count advanced; a figure mode's
        # absorbed shot counts only if the mode recorded a mark in the ledger.
        mark_drawn = False
        _ledger_before = watch.LEDGER.count
        mapped = self._map_to_screen(shot.x, shot.y)
        if (mapped is not None and self._app_is_drawing
                and not self._modetest_active):
            # NOT A TRIGGER PULL, SO THE DRILL NEVER HEARS OF IT (2026-10-20).
            #
            # `_app_is_drawing` has been deciding "could this possibly be a
            # trigger pull" since 2026-10-15 -- and until now it gated the
            # BANG and nothing else. Everything below (the mode's ammunition,
            # the round, the dry-pull counter, the hole, the session score,
            # the sensitivity learner) ran for every detection regardless.
            # Live, 20:05:40-43: the grid had just finished, so a mapping
            # existed, and the noise floor was being re-measured, so the app
            # said "not treated as a trigger pull yet". Six detections at peak
            # 8-12 (his real shots: 85) went straight into the Two Players
            # magazine; he had fired nothing and saw 11 of 17 rounds.
            #
            # The same gate for every consumer, here, and not a condition in
            # each of them (RULES B9, E2). The arcade mode test is the one
            # exception: it exists to drive a mode with the app's own dots.
            #
            # Only a MAPPED detection is dropped here. With no mapping the
            # detection never reaches the mode (there is nowhere to put it)
            # and the unmapped branch below says so out loud, which is the
            # diagnostic a shooter without a calibration needs.
            self._not_pull_ignored += 1
            self._say_once(
                "not-a-pull-ignored",
                "  a detection arrived while the app is not treating "
                "detections as trigger pulls (a measurement, no measured "
                "floor, no mapping, or the screen covered). It spends no "
                "round, scores nothing, draws no hole and teaches nothing. "
                "The count is reported when real shots count again.")
            return

        if mapped is not None:
            sx, sy = mapped
            rect = self.target_window.rect() if self.target_window is not None else None
            host = self.layers.get("modes")
            draw_hole = True
            absorbed = False
            dry = False
            silent = False
            if host is not None and host.mode is None:
                # Cannot happen in the app (the host starts on the static
                # targets), but a screen with nothing on it is the fault he
                # has reported three times, so it is repaired, not tolerated.
                host.set_mode(arcade.StaticTargets())
            if host is not None and host.mode is not None and rect is not None:
                # An arcade mode is playing, so it owns what a shot means. Only
                # it knows what was on screen; the app must not second-guess it.
                # perf_counter, the clock the detector stamps frames with
                # (RULES B4). This was time.monotonic(), which on Windows
                # shares no epoch with it, so `ctx.now - ev.t` -- what the
                # ball modes rewind by -- was meaningless. The FIRST frame's
                # stamp, because the position reported is the first frame's.
                ctx = self.scene._context(time.perf_counter(), 0.0, rect)
                ev = ShotEvent(x=sx, y=sy, t=shot.first_timestamp,
                               index=shot.index, pointer=shot.track_id,
                               t_end=shot.last_timestamp,
                               # Data only: the pulse's brightness, filed by the host
                               # with the round for the ghost score's record guard.
                               peak=float(shot.peak))
                # How close was it to the nearest target? Asked BEFORE the mode
                # acts: a hit removes or respawns its target, and the distance
                # to what is left would describe the next one. Log-only, and
                # never allowed to cost a hole.
                try:
                    near = host.nearest(ev, ctx)
                except Exception as exc:
                    self._say_once("nearest-failed",
                                   f"  could not measure the distance to the "
                                   f"nearest target: {exc!r}")
                # The side the shot belongs to (the only mode in single-player).
                routed = host.mode_for(ev.x, ctx)
                before_mode = mode_snapshot(routed)
                outcome = host.shot(ev, ctx)
                # A DRY PULL is a pull with an empty magazine (`outcome.dry`): the
                # bubble pop plays INSTEAD of the gunshot and its hole is drawn (his
                # rule, 2026-10-20, replacing a silent holeless dry pull; see
                # `ModeHost.shot`). A selection cue (`outcome.cue`: the MENU button,
                # the NEXT ROUND box, a menu tile) follows the sound by a
                # millisecond. Either way it is counted, logged and learned from
                # below; `play_sound` False would still silence a shot.
                dry = bool(outcome.dry)
                silent = not outcome.play_sound
                if silent:
                    _banged[0] = True
                bang(outcome.sound, dry=dry, cue=outcome.cue)
                if not self._app_is_drawing:
                    for line in mode_events(before_mode, mode_snapshot(routed)):
                        self.voice.say(line)
                # An `unscored` shot (a versus side already at its allotment) was
                # SEEN and is neither a hit nor a miss: scored=None makes the
                # session count the shot and nothing else.
                scored = None if outcome.unscored else outcome.hit
                points, target_id = outcome.points, outcome.target_id
                draw_hole = outcome.draw_hole
                # A mode that carries the hole on its own figure says so; the
                # screen hole is then withheld (everything else still runs).
                absorbed = outcome.absorbed
                self.scene.mark_dirty()
            if draw_hole:
                # THE VERDICT PANEL GETS OUT OF THE WAY WHEN HE STARTS
                # SHOOTING. He said it, looking at the screen, while I had
                # the same picture open and had not: "So now you display
                # this bright message on screen and you do not draw in any
                # bullet holes."
                #
                # A READY verdict clears itself after four seconds. A
                # problem verdict deliberately does not, so it cannot be
                # missed -- and that was written for a shooter who is NOT
                # shooting. Once a trigger pull arrives the panel has been
                # read or it has not, and either way it is now a bright
                # filled panel across the middle of the screen the camera
                # is watching, which is both in his way and against the
                # house rule on lit area (RULES B1b/B2). Q brings it back.
                board = self.scene.get("verdict")
                if board is not None and getattr(board, "visible", False):
                    board.visible = False
                    self.scene.mark_dirty()
                    self._log("  a shot arrived, so the verdict panel stood "
                              "down: it is a lit panel on the screen the "
                              "camera watches, and he is shooting now. "
                              "Press Q to read it again.")
                # AND THE HOLES ARE VISIBLE, because this is the product.
                #
                # 2026-10-16: `settings.local.json` carried holes=false, so
                # every shot he fired was detected, mapped, and drawn into
                # a layer that was switched off. The app reported READY,
                # 12 of 12, 55 fps -- and he saw a black screen. A saved
                # setting silently beating a code default is already in
                # these notes, from `min_frames`; this is the same bug
                # applied to the one thing he is there to see.
                # AND SOMETHING TO AIM AT, for the same reason as the
                # holes below: he is shooting, so the screen must show him
                # what he is shooting at.
                self._ensure_something_to_shoot_at("a shot arrived")
                self._stand_down_camera_pane("a shot arrived")
                holes = self.layers["holes"]
                if not holes.visible:
                    holes.visible = True
                    if "holes" in self._layer_boxes():
                        self._layer_boxes()["holes"].setChecked(True)
                    self._log("  bullet holes were switched off in the saved "
                              "settings, and a shot has just landed. Turning "
                              "them on: a detected shot with nothing drawn is "
                              "indistinguishable from a detector that does "
                              "not work.")
                if not absorbed:
                    _holes_before = holes.added
                    holes.add(sx, sy, shot.index, points)
                    mark_drawn = bool(holes.visible
                                      and holes.added > _holes_before)
            if rect is not None and not rect.contains(int(sx), int(sy)):
                # Seen, mapped, and landed outside the screen. Silently dropping
                # this is indistinguishable from not detecting it at all.
                self._signal_note(
                    "that shot mapped OFF the screen - the camera has probably "
                    "moved. Press G to recalibrate.", "warn")
        else:
            self._signal_note("shot seen, but there is no calibration yet, so "
                              "there is nowhere to draw it. Press C.", "warn")
            # AND SAY IT WHERE HE WILL ACTUALLY SEE IT.
            #
            # `_signal_note` writes to the SIGNAL METER, which is in
            # INFO_LAYERS and therefore hidden by default while shooting --
            # "the default while shooting is a clean screen". So on
            # 2026-10-18 the note was emitted 42 times onto a layer that was
            # not on screen, and nothing was written to the log either. Every
            # channel the app had was pointed somewhere nobody was looking,
            # which is the shape of every expensive fault here: it did not
            # fail, it reported into silence.
            #
            # Three channels, each with a different job: the log, so an
            # unattended run can be diagnosed afterwards; the alert, which
            # interrupts and must be read; and the running count, so the
            # second and the fortieth shot are not identical lines.
            self._unmapped_shots += 1
            if self._unmapped_shots == 1:
                self._alert(
                    "SEEN - BUT NOT CALIBRATED",
                    "That shot WAS detected.\n"
                    "It is not drawn because the app does not know\n"
                    "where the camera is looking, and a hole in the\n"
                    "wrong place reads as bad shooting.\n\n"
                    "Press C to measure the screen (nothing to shoot).\n"
                    "If that keeps failing, press G and shoot the grid.",
                    seconds=20.0)
            self._say_once(
                "unmapped-shot",
                "  that shot was SEEN and not drawn: there is no "
                "camera-to-screen mapping, so there is nowhere to put a "
                "hole. The detector is working. Press C to measure the "
                "screen (nothing to shoot) or G to shoot the grid. Every "
                "shot from here will report `screen=(uncalibrated)` until "
                "then.")
            if self._unmapped_shots in (5, 10) or self._unmapped_shots % 25 == 0:
                self._log(f"  {self._unmapped_shots} shots have now been "
                          f"detected with no calibration, so none of them "
                          f"has been drawn. Press C.")
        bang()      # no mode answered (unmapped, or no host): the ordinary gunshot
        # Without calibration we know a shot happened but not what it hit, and
        # a hole in the wrong place reads as aiming error. Better none at all.

        # A figure mode's absorbed shot (the mark rides the figure and the
        # screen hole is withheld) is a mark if, and only if, the mode said so.
        if watch.LEDGER.count > _ledger_before:
            mark_drawn = True
        if mark_drawn:
            self.watchdog.note_mark(time.perf_counter())

        if dry:
            # A dry pull is not a shot the session fired: its own counter, and
            # nothing else it keeps (shots, rate, timing) moves. See SessionStats.
            self.session.register_dry()
        else:
            self.session.register_shot(
                camera_x=shot.x, camera_y=shot.y, screen_x=sx, screen_y=sy,
                scored=scored, score=points, target_id=target_id,
                track_id=shot.track_id, timestamp=shot.last_timestamp,
            )
        self.layers["hits"].add(shot.nx, shot.ny, shot.track_id)
        self.scene.mark_dirty()

        where = f"screen=({sx:7.1f},{sy:7.1f})" if sx is not None else "screen=(uncalibrated)"
        hit = ("" if scored is None else
               f"  HIT {target_id} +{points}" if scored else
               f"  miss [{target_id}]" if target_id else "  miss")
        if dry:
            hit += "  [DRY: magazine empty; the bubble pop instead of the gunshot, hole drawn, not scored]"
        # Evidence for the morning's string (plan 02 step 10): how far from
        # the nearest target, when (the clock the HEARD lines use), what the
        # microphone heard, how the centroid was made, and the running count
        # of heard against seen. Built defensively: it must never cost a line.
        try:
            extra = "  ".join(x for x in (
                describe_nearest(near, bool(scored), target_id, self._mm_per_px()),
                f"t={shot.first_timestamp:.3f}",
                f"mic_peak={mic_txt}",
                shot_evidence(shot, float(self.sp_minpeak.value()),
                              self._mm_per_camera_px()),
                audio_tally(self.corroborator) if self.listener.running
                else "[audio: microphone off]") if x)
        except Exception as exc:
            extra = f"(shot evidence failed: {exc!r})"
        self._log(
            f"SHOT {shot.index:3d}  cam=({shot.x:7.2f},{shot.y:7.2f})  {where}  "
            f"frames={shot.frames}  pulse~{shot.duration_ms:5.1f}ms  peak={shot.peak}  "
            f"area={shot.area:.0f}px{hit}  {extra}"
        )
        self._learn_sensitivity(shot)
        # Beside the sensitivity learner, from the same evidence and under
        # the same conditions -- `_learn_sensitivity` refuses the self test's
        # own dots and so does this, because a screen EMITS its dot and a
        # laser reflects (RULES C6).
        if not (self._calibrating or self._selftest_active
                or self._modetest_active):
            self._update_play_area(shot)

    # -- learning the screen's sensitivity from real shots --------------------
    #
    # Measured in the gun test of 2026-10-14: the laser's median peak was 84 in
    # the middle of this screen and 47 on the right third, over a blob of 149
    # camera pixels against 404. A single threshold is therefore too strict on
    # one side and too loose on the other, and 43% of audible trigger pulls were
    # never detected -- almost all of them on the right.
    #
    # The screen markers CANNOT measure this. They saturate the sensor wherever
    # they are drawn (227-245 across the whole panel), so they measure how
    # brightly the camera sees the screen, not how a point source smears out as
    # the surface turns away. Only the laser measures the laser.
    #
    # So it is learned while playing, from the shots that do get through, and
    # applied once there is enough spread to be worth trusting. Nothing has to
    # be fired on purpose.

    LEARN_MIN_SHOTS = 12          # below this, one flyer moves the whole map
    LEARN_BINS = 3                # 3x3 over the camera's view of the screen
    LEARN_MIN_BINS = 4            # must span the screen, not cluster in it
    # A bin speaks for a ninth of the screen, so it must not speak on the
    # strength of one reading. Measured 2026-10-14 19:43: a single stray
    # scoring 23 -- a calibration artefact, not a trigger pull -- was the only
    # sample in its bin, so the median of that bin WAS the stray. The map
    # concluded that region was 3.7x insensitive, amplified its noise by the
    # same factor, and the re-measured floor went 11 -> 70, taking min_peak
    # from 26 to 85. More than half of all real shots were then rejected: the
    # normalisation meant to recover the weak regions destroyed detection
    # everywhere. A median only protects against a flyer when it has company.
    LEARN_MIN_PER_BIN = 3
    # And the map is judged on its own terms afterwards. It exists to make a
    # threshold mean the same everywhere; if applying it makes the measured
    # noise floor dramatically worse, it has failed, whatever its inputs
    # looked like. Rejecting it is always safe -- the detector simply stays
    # uniform, which is how it ran for every working session before this.
    LEARN_MAX_FLOOR_RISE = 2.0

    def _say_once(self, key: str, message: str) -> None:
        """Log `message` the first time `key` comes up in this session.

        For facts that are true of every shot. The gunshot reasons below
        would otherwise print sixty times a string and bury the log that is
        the only account of an unattended run -- which happened once already,
        with roughly four hundred identical refusal lines from `poke()`.
        """
        if key in self._said_once:
            return
        self._said_once.add(key)
        self._log(message)

    # -- the information group (key 0) and the clean screen -------------------
    #
    # Purely informational furniture: the camera preview, the stats HUD, the
    # standing prompt band, the signal meter, the key legend, the background
    # label and the uncalibrated camera-space minimap. NOT in it, because
    # they are the product or must interrupt: bullet holes, targets and
    # anything a mode draws, the alert, the verdict board.
    #
    # The camera preview is NOT in this group any more, and cannot be: it
    # draws a live image only while the camera is being aimed
    # (`CameraPane.draws_camera_image`). Key 0 used to switch it on with
    # the rest and the feedback loop that followed produced 172 detections
    # against 3 real shots (2026-10-17 08:50).
    INFO_LAYERS = ("stats", "prompt", "signal", "help", "bglabel", "hits")

    def _info_shootable(self) -> bool:
        """True when nothing is measuring and the screen is his to shoot at."""
        if (self._grid_active or self._sgrid_active
                or self._pat_active or self._calibrating
                or self._flash is not None or self._aim_help
                or self._selftest_active or self._modetest_active
                or self._probe_active):
            return False
        return not self.director.busy_label()

    def _toggle_info(self) -> None:
        self._info_full = not self._info_full
        self._info_last_set.clear()          # a deliberate toggle ends manual overrides
        self._log("information on the target screen: "
                  + ("SHOWN (stats, status band, signal meter, "
                     "key list). Press 0 to hide it." if self._info_full else
                     "HIDDEN (shown again by 0, and the status band returns "
                     "by itself when something is wrong)."))
        self._apply_info_policy()

    def _apply_info_policy(self) -> None:
        """Clean while shooting; speak only when something is wrong.

        WHAT COUNTS AS AN ISSUE: the standing prompt band is in its `warn`
        state -- camera not connected, shots not being counted (screen
        covered), not calibrated, rough calibration only, or a verdict that
        is not READY. That list is the app's own existing definition of
        "something needs fixing", so there is one, not a second one here.
        An issue brings back the prompt band only. The alert and verdict
        board are outside the group and speak for themselves. A working but
        idle app is deliberately silent -- he can press 0 to see it.

        While the app is MEASURING (calibration, probes, self test) this
        hides nothing, and hands back the prompt and signal meter it hid,
        because calibration instructions live there.

        Keys that toggle one layer (P, S, B, L, H) are respected: a layer
        whose visibility differs from what this last set is left alone
        until the next 0.
        """
        if self.scene is None:
            return
        if not self._info_shootable():
            for name in list(self._info_hidden_by_policy):
                lay = self.layers.get(name)
                if lay is not None:
                    lay.visible = True
            if self._info_hidden_by_policy:
                self.scene.mark_dirty()
            self._info_hidden_by_policy.clear()
            self._info_last_set.clear()
            return
        prompt = self.layers.get("prompt")
        issue = prompt is not None and prompt.kind == "warn"
        changed = False
        for name in self.INFO_LAYERS:
            lay = self.layers.get(name)
            if lay is None:
                continue
            want = True if self._info_full else (issue and name == "prompt")
            last = self._info_last_set.get(name)
            if last is not None and lay.visible != last:
                continue                     # operator toggled it by hand
            if lay.visible != want:
                if not want and name in ("prompt", "signal"):
                    self._info_hidden_by_policy.add(name)
                lay.visible = want
                changed = True
            self._info_last_set[name] = want
        if changed:
            self._sync_layer_checkboxes()
            self.scene.mark_dirty()
            self._say_once("clean-screen",
                           "  clean screen while shooting: holes and targets "
                           "only; the status band returns by itself when "
                           "something is wrong. 0 shows or hides all the info.")

    def _stand_down_camera_pane(self, why: str) -> None:
        """The camera preview must not be up while anyone is shooting.

        2026-10-16 evening: the pane sat in the top-right corner of his
        screen all evening, obscuring that corner. RULES B1b says it is
        opt-in because it lights 84% of the screen and "has no business
        being up while anyone is shooting" -- but `settings.local.json`
        carried camera=true and nothing ever turned it off. A saved
        setting silently beating the product, the same shape as the holes
        and the aim rings, so the shooting state owns it the same way.

        Once per session, at the first moment the app is shootable (setup
        over, or a shot has arrived). After that it belongs to the
        operator: P or the checkbox turns it back on and it stays on.
        Never touched while the camera is being AIMED -- the aiming view
        is the one legitimate reason for it to be up, and it ends itself.
        """
        if self._camera_pane_stood_down:
            return
        pane = self.scene.get("camera")
        if pane is None:
            return
        if getattr(pane, "aiming", False):
            return                  # aiming the camera: try again later
        self._camera_pane_stood_down = True
        if not pane.visible:
            return
        pane.visible = False
        self._sync_layer_checkboxes()
        self.scene.mark_dirty()
        self._log(f"  the camera preview pane was switched on in the saved "
                  f"settings and the app is now ready to be shot at "
                  f"({why}): switching it off, because it covers the "
                  f"top-right of the screen he is shooting at. P turns it "
                  f"back on.")

    def _ensure_something_to_shoot_at(self, why: str) -> None:
        """Never present a screen with nothing on it to aim at.

        Reported twice in one session, 2026-10-16:

            "I'm hoping I will see some targets this time."
            "However I still don't see the targets. Did you plan to add the
             targets?"

        And the cause was not missing targets. His `settings.local.json`
        carried `layers.aim = false`, so free aim drew the background, the
        holes and nothing else -- while the arcade modes sat behind keys he
        had no reason to know about.

        That is a SAVED SETTING SILENTLY BEATING THE PRODUCT, which these
        notes already carry twice: `min_frames` sat wrong for days, and this
        same morning `holes = false` meant every shot he fired was detected,
        mapped, and drawn into a layer that was switched off while the app
        reported READY. The fix there was to turn the layer on when a shot
        arrives and say so; this is that fix for the other half of the thing
        he is there for.

        A mode owns the screen when one is running, so this defers to it.
        """
        host = self.layers.get("modes")
        if host is None:
            return
        # 2026-10-16, the third report: "free aim" was a blank screen and
        # the owner said it is useless. There is no such state any more:
        # the screen always has a mode, by default the static targets. This
        # repairs the two ways it could still be empty -- no mode, or the
        # mode layer switched off.
        fixed = []
        if getattr(host, "mode", None) is None:
            host.set_mode(arcade.StaticTargets())
            fixed.append("no mode was running")
        if not host.visible:
            host.visible = True
            fixed.append("the targets layer was hidden")
        if fixed:
            self.scene.mark_dirty()
            self._log(f"  there was nothing on the target screen to shoot at "
                      f"({why}): {' and '.join(fixed)}. Restored the static "
                      f"targets -- an empty screen is not a product, and he "
                      f"has reported it three times. 1-6 start a drill.")

    def _screen_is_ours(self) -> str:
        """Does the app own the display? Empty string if it does.

        Asked for after a gate in which all five launches failed and I
        blamed the room:

            'the app did not acquire the full screen. That is the issue you
             need to check. Before you take any screen measures you need to
             see if the app is maximized on screen.'

        The evidence was in the log and I read past it. A marker is found by
        photographing the screen with it and without it and subtracting; the
        target screen is near-black, so this:

            lit frame peaks at 255, unlit at 255

        is not a photograph of the target screen at all. Both frames were
        the desktop, so they were the same picture, and all eighteen markers
        came back invisible on all five launches -- 150 seconds each.

        It RAISES the window first and only then reports, because the fix
        for 'something is in front of it' is usually to put it in front,
        and a check that only complains costs a launch to act on.
        """
        # Before ANYTHING else: this is the gate every measures_screen job
        # passes at start, and the splash must be gone before its first step.
        self._splash_down("a measurement is starting")
        win = self.target_window
        if win is None:
            # No target screen yet. The routines that need one open it
            # themselves (the two-marker check does, bounded to one
            # attempt); refusing here would deadlock that.
            return ""
        if win.in_front_of_camera():
            return ""
        self._log(f"  the target screen is not in front ({win.cover_reason}); "
                  f"raising it before measuring anything on it.")
        if not force_to_front(win):
            self._log("  the target window could NOT be brought to the front. "
                      "Windows refuses SetForegroundWindow to a process that "
                      "is not already in front, and flashes the taskbar "
                      "instead. The camera is looking at whatever IS in "
                      "front, so nothing measured now means anything: click "
                      "the target window, or alt-tab to it.")
        if win.in_front_of_camera():
            self._log("  it came to the front. Measuring.")
            return ""
        # ASSERT THE EFFECT, NOT THE ATTEMPT (RULES C1). Windows refuses
        # foreground activation to a process that is not already in front,
        # which these notes record for screenshots and which is the most
        # likely reason a launch finds itself behind.
        return (f"the app does not own the display: {win.cover_reason}. Click "
                f"the target window or alt-tab to it, then press G.")

    def _set_gain_map(self, gain, why: str) -> None:
        """Change the detector's gain map, and retire what it makes stale.

        THE EVIDENCE IS IN DIFFERENT UNITS EITHER SIDE OF THIS CALL. A shot's
        `peak` is the score the detector reports, and the gain map divides
        that score by the region's relative sensitivity -- so a peak measured
        before the map and one measured after are not the same quantity.

        Measured on his string of 2026-10-16, right third, bar 14:

            up to the map   16 shots, peaks 10,12,13,14,18,26,34,38,46,...
            after it         7 shots, peaks 21,54,77,104,107,109,181

        The play-area border took a median over both sets together, so the
        pre-map readings held it down for the rest of the session and a third
        of his screen stayed fenced off after the very fix that cured it had
        landed. He saw the border; the numbers behind it were obsolete.

        That is the same error as the gain-map withdrawal guard comparing a
        floor measured at one preset against one measured at another, fixed
        this morning: COMPARE MEASUREMENTS TAKEN THE SAME WAY. Fixed here by
        retiring the evidence rather than by trying to convert it, because
        the conversion is exactly the map that just changed.

        Every caller goes through here. There were five direct calls to
        `capture.set_gain_map`, and a sixth added later would have silently
        kept stale evidence -- which is the `SCREEN_OWNERS` failure shape.
        """
        # Remembered HERE and nowhere else, so the brightness grid can put back
        # the map it set aside. A copy of a fact the capture holds is a second
        # cache (RULES C7); it is safe only because this is the one funnel.
        self._gain_in_force = gain
        if self.capture is not None:
            self.capture.set_gain_map(gain)
        if self._area_shots:
            self._log(f"  the play-area evidence ({len(self._area_shots)} "
                      f"shots) is retired: {why} changed what a peak means, "
                      f"so those readings are in the old units.")
        self._area_shots = []

    def _update_play_area(self, shot: "Shot | None" = None) -> None:
        """Measure where a shot is reliably detected, and draw the border.

        Asked for 2026-10-16, after a string that worked everywhere but the
        right-hand side: "every time the app starts, you will calibrate and
        then you draw strict borders to highlight the area that is supported
        by the current camera placement... You will invite the user to
        concentrate his shots on the area of the monitor where you can
        reliably detect the shots."

        Measured from REAL SHOTS and from nothing else -- see
        `lsa/playarea.py` for why the twelve-dot self test cannot answer
        this. It shares its evidence with the sensitivity learner, which is
        why it lives next to it: both are measurements of how this panel
        images a laser at this camera position, and both die when the camera
        moves.

        Advisory, by his instruction: targets stay inside, holes are drawn
        wherever a shot lands.
        """
        if not playarea.BORDER_ENABLED:
            return          # switched off, and said so at startup
        if self.mapping is None or self._sens_shots is None:
            return
        # THE BAR THE DETECTOR IS ACTUALLY USING, read from the control that
        # holds it. The first version of this line was
        # `getattr(self.detector_params, "min_peak", 0)` -- and
        # `detector_params` is a METHOD, so it would have returned 0 for ever
        # and switched this whole feature off silently. That is RULES C8
        # exactly: `stats.last_shot_at` returned 0.0 through getattr and a
        # guard never fired for days while looking like protection. No
        # getattr-with-default on our own state.
        bar = float(self.sp_minpeak.value())
        if bar <= 0:
            return
        # The shots are kept in CAMERA space, for the gain map that divides a
        # camera-space score. The border is a thing on the SCREEN, so they
        # are mapped before binning rather than the grid being guessed.
        if shot is not None and shot.peak > 0:
            # ITS OWN LIST, not the sensitivity learner's. That one is kept
            # in camera space for a camera-space score and is deliberately
            # NOT cleared when the map is applied -- the map is fitted from
            # it. This one must be, because the border compares peaks
            # against a bar and the map changes what a peak is.
            self._area_shots.append((float(shot.x), float(shot.y),
                                     float(shot.peak)))
        on_screen = []
        for cx, cy, pk in self._area_shots:
            mapped = self._map_to_screen(cx, cy)
            if mapped is not None:
                on_screen.append((mapped[0], mapped[1], pk))
        area, lines = playarea.measure(on_screen, bar, self.mapping.screen_size)
        before = self.mapping.play_area
        self.mapping.play_area = area
        if area != before:
            # Said once per change, with the numbers. A border that moves
            # between launches with no explanation is the thing this project
            # has paid most for.
            for line in lines:
                self._log(line)
            self._update_playable(self.mapping)
            self.scene.mark_dirty()
            # PERSISTED ON CHANGE. It was written to disk only when
            # something else happened to save the mapping, so his
            # calibration.json kept the 67% border measured from his first
            # twelve shots while the live value had moved on -- and the next
            # launch would have started from the obsolete one. A measured
            # value that reaches the file only by luck does not persist.
            self._save_mapping(self.mapping, "the play area")

    # Why `_sens_refusal` says no, in words for the log.
    _SENS_REFUSAL_TEXT = {
        "faint": "below the laser/noise boundary, so it cannot be told from "
                 "something that is not a laser",
        "animating": "content was MOVING on the screen when the shot was "
                     "fired, and a moving figure's edges are not a laser",
    }

    def _sens_ignored_note(self) -> str:
        ig = self._sens_ignored
        return (f"ignored so far: {ig['faint']} below peak "
                f"{self.SINGLE_FRAME_PEAK}, {ig['animating']} while the "
                f"screen was animating; {self._sens_admitted} faint admitted "
                f"on the microphone's word")

    # How far AFTER a detection's first frame a click may arrive and still
    # corroborate it. The trigger click precedes the laser by a few ms, so
    # nothing real needs more than a hair of forward slack for clock skew. The
    # app's OWN bang for this shot can only be announced after `min_frames`
    # (2) camera frames, which at 60 fps is >= 16.7 ms after the first frame
    # (33 ms at 30 fps), and then needs the output latency (~21 ms measured)
    # to reach the microphone: >= ~38 ms at 60 fps. 20 ms sits under that by
    # construction, so the bang is outside the window; a symmetric window
    # would admit it. If `min_frames` or the frame rate ever move, redo this
    # sum -- `_decide_click`'s own-sound discount is the second line of
    # defence, not the first. (The brief for this change counted the pipeline
    # as 33/67 ms; the first-frame-to-announce part alone is half of that, and
    # the output latency makes up the rest, so the margin is the same size.)
    SENS_CLICK_FORWARD = 0.02

    def _sens_refusal(self, shot: "Shot") -> "str | None":
        """Why this detection must not teach the sensitivity map, or None.

        Measured 2026-10-17 10:07:27, while a figure walked: of the four
        regions the learner saved, three were the figure's own edges
        (peak 11 over 551px, peak 12 over 1130px, peak 25) and only one was
        a shot. It concluded the LEFT of the screen was weakest, where the
        raw sensor says it is the brightest, and amplified it up to 4x.

        A detection teaches when it cannot be anything but a laser: at or
        above the project's measured laser/noise boundary
        (`SINGLE_FRAME_PEAK`, 30 -- one constant, not a copy), and not while
        content is moving, whatever its peak.

        THE ONE WAY A FAINT DETECTION GETS IN (2026-10-23): a trigger click
        heard just before it. His right-hand side images at peak 20-35
        (median 28-35 against 85-90 on the left, measured over 8,415 live
        shots in 127 logs), so the peak boundary alone means the region the
        map exists to recover can never teach it, and a fresh install can
        never earn a map. The microphone does not share that falloff: the
        click is heard 12-16% of the time UNIFORMLY across the screen, and
        each right-hand region already held ~72 such shots against the 3 the
        learner needs. A screen's own dot, a figure's edge and sensor noise
        make no click. The click must be (see `claim_click_near`):
          - from a listener whose clock IS the camera's (RULES B4),
          - from a CALIBRATED listener -- an uncalibrated threshold is a
            guess, and a guessed 0.035 reported six shots in thirty seconds
            with nobody shooting,
          - unclaimed (one click vouches for one detection), and
          - BEFORE the pulse: back by the corroborator's own window, forward
            by `SENS_CLICK_FORWARD` only, so the app's own bang is outside it.

        "ANIMATING" IS NEVER OVERRIDDEN by a click. On 2026-10-17 10:07:27,
        while a figure walked AND he was shooting, three of the four regions
        saved were the figure's own edges (peak 11 over 551 px, 12 over
        1130 px). A click near a faint blob in a moving mode cannot be
        attributed to the blob rather than to his real shot elsewhere on the
        screen. So "moving" is decided BEFORE a click is claimed: claiming a
        click for a shot that is refused anyway would spend it and deny it to
        a shot that could have used it.

        No area band is applied to a corroborated shot: his right-side blobs
        measure 64-208 px median against 404 in the middle, so a band fitted
        to the whole-screen distribution (p10 113 px) would refuse exactly
        the shots this exists to admit.

        Side channel: `_sens_last_by_click` says whether a None returned here
        rests on the microphone rather than on brightness.

        "Moving" is `Mode.moving_at(first frame of the pulse)`: the state of
        the screen when the trigger broke, from the mode's own schedule. It
        was `mode.animated`, read after `host.shot()`, and a scoring HIT
        appends a flash ring and turns that True -- so the learner refused
        every hit and could learn only from misses, which land off-target and
        are biased toward the edges of this panel (an inversion, worse than no
        gate). Of the nine modes, only `advancing` (while a figure walks) and
        the two ball modes (always) answer True: the learner is PAUSED in
        those, by design, and the learner's log line names the mode.
        """
        self._sens_last_by_click = False
        host = self.layers.get("modes")
        mode = host.mode if host is not None else None
        moving = mode is not None and mode.moving_at(shot.first_timestamp)
        # EITHER SIDE MOVING REFUSES: in versus the camera sees both halves, and
        # the learner cannot tell whose animation a faint blob belongs to.
        other = host.mode2 if host is not None else None
        if other is not None and other.moving_at(shot.first_timestamp):
            moving = True
        if shot.peak < self.SINGLE_FRAME_PEAK:
            if moving:
                return "faint"            # no click is claimed for a refused shot
            lis = self.listener
            if (lis.running and lis.clock_is_perf_counter and lis.calibrated
                    and lis.claim_click_near(
                        shot.first_timestamp, self.corroborator.window,
                        self.SENS_CLICK_FORWARD) is not None):
                self._sens_last_by_click = True
                return None
            return "faint"
        if moving:
            return "animating"
        return None

    def _learn_sensitivity(self, shot: "Shot") -> None:
        if self._calibrating or self.capture is None or self.mapping is None:
            return
        if shot.peak <= 0:
            return
        if self._selftest_active or self._modetest_active:
            # Never learn sensitivity from the SCREEN's own red dot. Measured
            # 2026-10-14: the synthetic dot reads peak 70 / 77 / 76 across
            # left / middle / right where the laser reads 76 / 84 / 47. A
            # screen emits its dot; that does not fall off the way light
            # reflected off matte coating does, so a map learned from it would
            # be flat -- the same mistake as learning from the white markers,
            # in new clothes.
            #
            # This was already true by accident, because `_on_shot` returns
            # early during the self test. Saying it here makes it survive a
            # reordering of that function.
            return
        why = self._sens_refusal(shot)
        if why is not None:
            # Say so, because a map that silently learns from less than it
            # claims is the fault being fixed. First of each kind, then every
            # tenth, so the log carries the count without a line per frame.
            self._sens_ignored[why] += 1
            k = self._sens_ignored[why]
            if k == 1 or k % 10 == 0:
                _h = self.layers.get("modes")
                _m = getattr(getattr(_h, "mode", None), "name", "")
                self._log(f"  sensitivity learner: ignoring detection at peak "
                          f"{shot.peak} (area {shot.area:.0f}px)"
                          + (f" in mode '{_m}' (the learner is paused while "
                             f"this mode's content moves)"
                             if why == "animating" and _m else "")
                          + f" - {self._SENS_REFUSAL_TEXT[why]} "
                          f"[{self._sens_ignored_note()}]")
            return
        if self._sens_last_by_click:
            # Admitted below the laser/noise boundary on a click's word. Said
            # as loudly as a refusal: a map built partly on the microphone is
            # a different claim from one built on bright shots.
            self._sens_admitted += 1
            self._sens_by_click += 1
            k = self._sens_admitted
            if k == 1 or k % 10 == 0:
                self._log(f"  sensitivity learner: ADMITTING detection at "
                          f"peak {shot.peak} (area {shot.area:.0f}px), below "
                          f"the laser/noise boundary of "
                          f"{self.SINGLE_FRAME_PEAK}, because a trigger click "
                          f"was heard just before it "
                          f"[{self._sens_ignored_note()}]")
        self._sens_shots.append((float(shot.x), float(shot.y), float(shot.peak)))
        n = len(self._sens_shots)
        if n < self.LEARN_MIN_SHOTS:
            return
        # Re-fit on the first qualifying shot, then only as the evidence
        # doubles. A map that moves after every shot churns the noise floor
        # measurement with it, and the floor is what the map is judged against.
        if self._sens_applied and n < self._sens_applied * 2:
            return

        # Bin over the camera's view of the SCREEN, and take the median peak in
        # each bin rather than individual shots. One bright flyer must not
        # convince the map that its corner is the strong one -- the same reason
        # the shot grid keeps a median of three rather than an average.
        corners = self.mapping.screen_rect_in_camera()
        if not corners:
            return
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        b = self.LEARN_BINS
        buckets: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        for cx, cy, pk in self._sens_shots:
            i = min(b - 1, max(0, int((cx - x0) / (x1 - x0) * b)))
            j = min(b - 1, max(0, int((cy - y0) / (y1 - y0) * b)))
            buckets.setdefault((i, j), []).append((cx, cy, pk))
        if len(buckets) < self.LEARN_MIN_BINS:
            # Clustered shooting teaches nothing about the rest of the screen,
            # and a map fitted to one region claims the others are dim -- which
            # would amplify their noise until it looked like signal.
            return

        samples = []
        for (i, j), group in buckets.items():
            if len(group) < self.LEARN_MIN_PER_BIN:
                continue          # one reading may not speak for a ninth of the screen
            group.sort(key=lambda g: g[2])
            mid = group[len(group) // 2]
            samples.append(Sample(screen=(0.0, 0.0), camera=(mid[0], mid[1]),
                                  peak=mid[2]))
        if len(samples) < self.LEARN_MIN_BINS:
            return
        frame = (self.settings.width, self.settings.height)
        # Confine it to the screen. Outside the panel there is no laser hit to
        # recover, and amplifying the room is how a 1.5x correction cost 5.9x
        # of noise floor at 21:33 on 2026-10-14.
        gain = confined_sensitivity_map(samples, frame, corners)
        if gain is None:
            return
        peaks = sorted(s.peak for s in samples)
        spread = peaks[-1] / max(1.0, peaks[0])
        if spread < 1.4 and not self._sens_applied:
            # Already uniform enough. Saying so is worth a line, because
            # "normalised" and "did not need normalising" look identical from
            # the firing line and only one of them means the right third is fine.
            self._sens_applied = n
            self._log(f"  sensitivity learned from {n} shots "
                      f"({self._sens_by_click} of them on the microphone's "
                      f"word) across "
                      f"{len(buckets)} regions: only {spread:.1f}x between "
                      f"weakest and strongest, so detection is already even. "
                      f"No gain map applied. [{self._sens_ignored_note()}]")
            return
        self._sens_applied = n
        # Keep them WITH the mapping. The falloff is a property of this room,
        # this camera pose and this screen -- exactly what the mapping is for
        # -- so it has no business being relearned from scratch every launch.
        #
        # That reset is what made the learner impossible to satisfy: it needs
        # 12 shots over 4 regions, and the uncorrected threshold rejects the
        # very edge shots that would teach it, so each session threw away the
        # evidence and asked for more. Measured over 286 logged shots, the
        # evidence has been there since the first string.
        self._save_sensitivity(samples)
        self._set_gain_map(gain, "the sensitivity map")
        self._log(f"  sensitivity learned from {n} real shots "
                  f"({self._sens_by_click} of them resting on a trigger click "
                  f"heard just before, not on brightness) across "
                  f"{len(buckets)} regions of the screen: weakest region scored "
                  f"{peaks[0]:.0f}, strongest {peaks[-1]:.0f} ({spread:.1f}x). "
                  f"Detection normalised, so one threshold now means the same "
                  f"everywhere. [{self._sens_ignored_note()}]")
        self._report_signal_margin([s.peak for s in samples])
        # Normalising amplifies noise in the weak regions along with the
        # signal, so a floor measured before it is an underestimate. Not
        # optional -- and it is why the map is not re-fitted every shot.
        #
        # Remember the floor it has to beat. If the normalised floor comes
        # back far worse, the map is withdrawn: see `_floor_after_gain_map`.
        self._judge_map_against_a_fresh_floor(
            gain, "this map was just learned from real shots")

    @staticmethod
    def _sensitivity_rows(samples: list) -> list:
        """The on-disk form of a falloff: camera position and peak per region.

        ONE definition for both writers (the learner and the shot grid), so
        the file cannot hold two shapes of the same fact.
        """
        return [{"camera": [float(s.camera[0]), float(s.camera[1])],
                 "peak": float(s.peak)} for s in samples]

    def _save_sensitivity(self, samples: list) -> None:
        """Store the learned falloff beside the calibration it belongs to."""
        if self.mapping is None:
            return
        try:
            self.mapping.sensitivity = self._sensitivity_rows(samples)
            if self._save_mapping(self.mapping, "the sensitivity measurement"):
                self._log(f"  saved the sensitivity measurement "
                          f"({len(samples)} regions) with the calibration, so "
                          f"it does not have to be reshot next time.")
        except Exception as exc:
            self._log(f"  could not save the sensitivity measurement: {exc}")

    # -- what survives a recalibration of the same camera pose -------------
    #
    # A mapping is replaced every session by design (`calibrate_each_session`
    # -- a camera that was nudged cannot be detected after the fact). But some
    # of what the file holds is a property of the CAMERA POSE and the panel,
    # not of the fit: it is invalidated by the camera moving, which is the one
    # event that also invalidates the mapping. Measuring it again costs the
    # shooter shots or seconds for a number that had not changed.
    #
    # There are two such facts now and there will be a third, so they are
    # captured and applied as ONE SET rather than one field at a time:
    #
    #   sensitivity   the falloff learned from his real trigger pulls. Carried
    #                 since 2026-10-14, when the bootstrap deadlock was found:
    #                 the far side loses shots, the gain map cures it, the map
    #                 needs 12 shots across 4 regions, and the uncorrected
    #                 threshold rejects exactly the edge shots that teach it.
    #   dot_radius    the self test's tuned dot size. Four flashes, ~4s a
    #                 launch, rediscovering a settled number.
    #
    # `dot_radius` was added on 2026-10-15 and NOT carried, which is the
    # sensitivity bug of the day before reproduced in a second field inside a
    # day. Fable's review found it. Hence one set, one applier.
    # `play_area` is NOT a pose fact any more: the border is measured afresh
    # every session, and carrying it across a recalibration re-imported a
    # fence drawn from phantom detections (2026-10-16).
    POSE_FACTS = ("sensitivity", "dot_radius")

    def _capture_pose_facts(self) -> None:
        """Remember the pose facts the CURRENT mapping holds. Merges.

        Called at load -- before `calibrate_each_session` can set
        `self.mapping` to None -- and again when a calibration starts. It
        merges rather than replaces precisely because of that: by the second
        call there may be no mapping to read, and the first version read it
        there only, so it always found nothing.
        """
        m = self.mapping
        if m is None:
            return
        for name in self.POSE_FACTS:
            value = getattr(m, name, None)
            if isinstance(value, list):
                if value:
                    self._prev_pose[name] = list(value)
            elif value:
                self._prev_pose[name] = value

    def _carry_pose_facts(self, mapping) -> None:
        """Put them onto a freshly measured mapping. BEFORE it is saved.

        The first version of this ran AFTER the save. The carried rows reached
        the live mapping, so the session that calibrated logged "re-using the
        sensitivity ... 5 regions" and worked -- while the file on disk was
        written with none, so every pattern calibration silently emptied the
        measurement that exists to end the bootstrap deadlock. Measured
        2026-10-15: calibration.json held 0 regions where the backup held 5.
        """
        carried = []
        for name, value in getattr(self, "_prev_pose", {}).items():
            if getattr(mapping, name, None):
                continue                      # freshly measured; leave it
            try:
                setattr(mapping, name, value)
            except Exception:
                continue
            carried.append(pose_fact_note(name, value))
        if carried:
            self._log("    carrying forward what the camera's position "
                      "already measured (" + ", ".join(carried)
                      + "), and saving it with this mapping.")

    def _save_mapping(self, mapping, what: str) -> bool:
        """Write the calibration and SAY WHETHER IT LANDED.

        `Mapping.save` returns False on any failure and all three callers
        ignored it, so the app logged "saved" with nothing on disk. Found
        2026-10-16: numpy scalars from `find_marker` cannot be JSON
        encoded, the save returned False every time in the harness, and
        six consecutive launches paid for eighteen markers because there
        was never a saved mapping for the three-second check to confirm.
        The suite was green throughout, and the log said "saved."

        A failure here is not fatal -- the mapping is live in memory and
        this session works -- but it costs the NEXT launch twenty-four
        seconds and the shooter has a right to know the difference.
        """
        try:
            ok = bool(mapping.save(CALIBRATION_PATH))
        except Exception as exc:                             # noqa: BLE001
            ok = False
            self._log(f"  could not save {what}: {exc!r}")
        if not ok:
            from lsa import atomic as _atomic
            why = _atomic.last_error or "no reason recorded"
            self._log(f"  WARNING: {what} was NOT saved to "
                      f"{CALIBRATION_PATH} -- {why}. It is in use for this "
                      f"session, but the next launch will have to measure "
                      f"it again.")
        return ok

    def _restore_sensitivity(self) -> None:
        """Re-apply a falloff measured in an earlier session, if one fits.

        Only valid for the mapping it was measured with -- a camera that moved
        invalidates both together, which is why they live in the same file and
        are discarded together.
        """
        if self.capture is None or self.mapping is None:
            return
        rows = list(getattr(self.mapping, "sensitivity", None) or [])
        if len(rows) < self.LEARN_MIN_BINS:
            return
        samples = [Sample(screen=(0.0, 0.0),
                          camera=(float(r["camera"][0]), float(r["camera"][1])),
                          peak=float(r["peak"])) for r in rows]
        corners = self.mapping.screen_rect_in_camera()
        if not corners:
            # Without the outline the only map available is a whole-frame one,
            # which amplifies the room (see `confined_sensitivity_map`).
            # Uniform detection is the safe answer, and it is said.
            self._log("  the remembered sensitivity cannot be re-applied: the "
                      "mapping gives no screen outline to confine it to, and "
                      "an unconfined map amplifies the room. Detection stays "
                      "uniform.")
            return
        gain = confined_sensitivity_map(samples, (self.settings.width,
                                                  self.settings.height), corners)
        if gain is None:
            return
        peaks = sorted(s.peak for s in samples)
        spread = peaks[-1] / max(1.0, peaks[0])
        if spread < 1.4:
            return
        self._log(f"  re-using the sensitivity measured earlier for this "
                  f"camera position: {len(samples)} regions, weakest "
                  f"{peaks[0]:.0f}, strongest {peaks[-1]:.0f} ({spread:.1f}x). "
                  f"No need to reshoot it.")
        self._judge_map_against_a_fresh_floor(
            gain, "this map was measured in an earlier session")

    def _judge_map_against_a_fresh_floor(self, gain, why: str) -> None:
        """Apply the map, measure the floor it gives, and judge it ON THE BAR.

        THE GUARD WAS COMPARING TWO DIFFERENT ROOMS, and it cost him the
        right-hand third of his screen on 2026-10-16.

        `_floor_before_gain` used to be whatever floor was last measured --
        in his launch, one taken minutes earlier, before eighteen markers,
        at the CALIBRATION preset with gain free. The "after" reading was
        taken three seconds after the camera was switched back to the dark
        shooting profile. Measured in that one launch:

            11:03:10   floor 19   with the map
            11:03:18   floor 12   without it, 8s later
            11:04:00   floor  3   without it, 50s later

        The floor was DECAYING as the sensor settled after the preset
        switch. The map was simply holding the parcel when the music
        stopped, and the guard threw away the one thing that makes a single
        threshold mean the same on both sides of an obliquely-viewed panel.
        His far-side shots then imaged at peak 10-13 against a bar of 18
        and three of them left no hole.

        AN ARITHMETIC ARGUMENT LOOKS AVAILABLE HERE AND IS FALSE. It reads:
        the map is a RELATIVE sensitivity floored at 0.25, so it amplifies by
        at most 1/0.25 = 4x, his spans 55 to 90 which is 1.64x, therefore a
        19x rise cannot be the map's doing. DO NOT USE IT. The floor is a
        MAXIMUM over the screen region, so a non-uniform map can move WHERE
        that maximum is -- amplifying a noisier region by 1.5x took the floor
        from 11 to 30 with nothing multiplied by more than 1.5. That is the
        2026-10-14 incident this guard exists for, so the bound would have
        disarmed the guard for its own founding case. The full account, and
        the test that caught me writing it, are beside the `rise` comparison
        below. This paragraph used to assert the opposite and was corrected
        on 2026-10-23: a comment contradicting the code twenty lines under it
        is read as evidence, and this project has lost time to exactly that.

        I first fixed this by measuring a fresh baseline with no map and
        then another with it -- correct, and four to five seconds of his
        launch against a bar of twenty-five. It is also not what was
        wrong. The comparison was on the FLOOR, and at these levels the
        floor is quantised to nothing: a reading of "1" is really anywhere
        in [0.5, 1.5), so a map doing exactly its job reads as 3x.

        The floor only ever matters THROUGH THE BAR, which is what the
        detector consumes. Judged there, every real case today comes out
        right with no extra measurement at all:

            floor  1 -> 3    bar  8 -> 9     1.12x   kept   (was withdrawn)
            floor  1 -> 4    bar  8 -> 10    1.25x   kept   (was withdrawn)
            floor  1 -> 19   bar  8 -> 25    3.12x   withdrawn, correctly:
                                                     the camera had moved
            floor 11 -> 70   bar 17 -> 91    5.35x   withdrawn -- the
                                                     2026-10-14 incident
                                                     this guard exists for

        So: one measurement, judged on the quantity that decides whether a
        shot registers. The seconds go back to him.
        """
        self._gain_awaiting_baseline = gain
        self._log(f"  {why}.")
        self._apply_pending_map_and_remeasure(float(self._noise_peak))

    def _apply_pending_map_and_remeasure(self, baseline: float) -> None:
        """The baseline is in. Put the map on and measure again."""
        gain = self._gain_awaiting_baseline
        self._gain_awaiting_baseline = None
        if gain is None or self.capture is None:
            return
        self._floor_before_gain = float(baseline)
        self._gain_pending = gain
        self._sens_applied = 10 ** 6          # measured already; do not refit
        self._set_gain_map(gain, "the sensitivity map")
        self._log(f"  the bar before it is "
                  f"{expo.min_peak_for(baseline):.0f} (floor "
                  f"{baseline:.0f}). Applying the map and measuring the "
                  f"floor it gives.")
        self._calibrating = True
        QTimer.singleShot(400, lambda: self._start_noise_floor())

    @staticmethod
    def _map_max_amplification(gain) -> float:
        """The most this map can multiply any score by. 1.0 if it cannot.

        The detector DIVIDES by the map, so the largest amplification is
        one over its smallest value inside the screen region.
        """
        try:
            import numpy as _np
            arr = _np.asarray(gain, dtype=_np.float32)
            inside = arr[arr > 0]
            if inside.size == 0:
                return 1.0
            return float(1.0 / max(1e-3, float(inside.min())))
        except Exception:                                    # noqa: BLE001
            return 1.0

    def _floor_after_gain_map(self, floor: float) -> None:
        """Judge the gain map by the floor it produced, and undo it if worse.

        A sensitivity map exists so that one threshold means the same thing
        across a screen the camera sees unevenly. It pays for that by
        amplifying the weak regions -- noise included. Usually the trade is
        worth it; when the map is wrong it is catastrophic, and in exactly the
        direction nobody looks, because the map reports success and the damage
        shows up later as "more than half my shots do not register".

        Measured 2026-10-14 19:43 on this hardware:

            before the map   floor 11  ->  min_peak 26
            after the map    floor 70  ->  min_peak 85

        Real shots in the middle of the screen scored 86-107. A min_peak of 85
        rejects most of them, so the normalisation intended to rescue the far
        side of the screen stopped the near side working as well. The app had
        every number needed to notice and simply never compared them.
        """
        if self._gain_pending is None:
            return
        before = max(0.0, float(self._floor_before_gain))
        self._gain_pending = None
        # ON THE BAR, not on the floor. See `_judge_map_against_a_fresh_floor`.
        bar_before = expo.min_peak_for(before)
        bar_after = expo.min_peak_for(floor)
        rise = bar_after / max(1.0, bar_before)
        if rise <= self.LEARN_MAX_FLOOR_RISE:
            return
        # NO ARITHMETIC SHORTCUT HERE, and the attempt is worth recording.
        #
        # I first added a bound: the detector divides by the map, so the
        # most it can lift anything is one over the map's smallest value --
        # 1.64x for his 55-to-90 falloff -- and a 19x rise is therefore not
        # the map's doing. That reasoning is WRONG, and the test for it
        # caught me: the floor is a MAXIMUM over the screen region, so a
        # non-uniform map can move WHERE the maximum is. Amplifying a
        # noisier region by 1.5x can take the floor from 11 to 30 even
        # though nothing was multiplied by more than 1.5.
        #
        # That is exactly the 2026-10-14 incident this guard exists for
        # (floor 11 -> 70, min_peak 85, real shots 86-107), so the bound
        # would have disarmed it. The honest fix is the baseline, not a
        # clever sum: measure both ways, back to back, in the same
        # conditions -- which is what `_judge_map_against_a_fresh_floor`
        # now does.
        self._log(f"  WITHDRAWING the sensitivity map: it took the bar a "
                  f"shot has to clear from {bar_before:.0f} to "
                  f"{bar_after:.0f} ({rise:.1f}x, floor {before:.0f} -> "
                  f"{floor:.0f}), which rejects more real shots than the "
                  f"normalisation recovers. Detection goes back to uniform "
                  f"and the floor is measured again.")
        self._sens_applied = 0
        # The samples that produced it are suspect too -- keeping them would
        # rebuild the same map at the next doubling.
        self._sens_shots = []
        self._sens_by_click = 0
        if self.capture is not None:
            self._set_gain_map(None, "withdrawing the sensitivity map")
            self._calibrating = True
            QTimer.singleShot(400, lambda: self._start_noise_floor())

    def _on_probe(self, results: list) -> None:
        self._log("--- control probe ---")
        for r in results:
            if r.key in self.rows:
                self.rows[r.key].mark_probe(r)
            verdict = "writable" if r.writable else "IGNORED by driver"
            extra = f"  {r.note}" if r.note else ""
            self._log(f"  {r.key:16s} initial={r.initial:<10g} {verdict}{extra}")
        self._log("--- end probe (greyed rows are not settable on this backend) ---")

    def _on_echo(self, key: str, value: float, accepted: bool) -> None:
        # Show the readback, but never overwrite what the user asked for with it.
        # Measured on the C910 over MSMF: CAP_PROP_EXPOSURE always reads back -6
        # however it is set, while the writes genuinely work (mean frame
        # brightness sweeps 95 -> 6.4 across the range). Snapping the control back
        # to a bogus readback both loses the real setting and makes a working
        # driver look broken.
        if key in self.rows:
            self.rows[key].set_echo(value, accepted)

    # -- calibration by shooting ---------------------------------------------

    def _on_mapping_checked(self, unmoved: bool) -> None:
        """Act on the two-marker check: keep the mapping, or measure again."""
        if unmoved:
            # Treat it as measured NOW, because it has just been tested
            # against this camera in this position -- which is the only
            # thing `calibrate_each_session` was ever trying to establish.
            self._mapping_measured_now = True
            self._check_confirmed = True
            self._push_detector()
            self._update_playable(self.mapping)
            self._restore_sensitivity()
            self._apply_preset("laser", with_exposure=True)
            self._set_status(
                "CALIBRATION CONFIRMED  -  the camera has not moved",
                "Two markers landed where the saved mapping predicted, so "
                "it was kept. Press C to measure it again from scratch.",
                "#17331c", "#7fe08a", "#2e6b38")
            self._after_calibration_ready()
            return
        # THE CAMERA MOVED, SO EVERYTHING ABOUT THE OLD POSE IS VOID.
        #
        # `_prev_pose` holds the sensitivity learned from his real shots and
        # the tuned dot radius, and both are properties of WHERE THE CAMERA
        # WAS. Clearing the mapping and keeping them carries the old pose's
        # facts onto the new one -- a falloff measured for a view that no
        # longer exists, applied over points up to 512 px away.
        #
        # Measured in his log, 2026-10-16 11:02:37: "the camera HAS moved:
        # 512.5 px", then at 11:03:07 "re-using the sensitivity measured
        # earlier for this camera position" -- a sentence that was false
        # when it was printed. The map built from it amplified room rather
        # than panel, took the noise floor to 19, and was withdrawn. Had it
        # lifted the floor only slightly it would have stayed on and
        # amplified the wrong third of the room for the whole session.
        #
        # Today's generalisation to POSE_FACTS made this worse rather than
        # better: before it, only the sensitivity was carried; now the dot
        # radius goes with it. Found by an outside review.
        self._prev_pose = {}
        self.voice.say("v_camera_moved")
        self._log("  the camera has moved, so the sensitivity and dot size "
                  "measured at the old position are discarded with the "
                  "mapping: they describe a view that no longer exists.")
        self.mapping = None
        self._apply_preset("laser", with_exposure=True)
        QTimer.singleShot(400, self._calibrate_if_needed)

    def _after_calibration_ready(self) -> None:
        """The chain that normally follows a successful calibration.

        The probe is asked for DIRECTLY, with no delay to guess. It is
        queueable, so if a measurement is somehow still in flight the
        Director holds it and starts it the moment the slot frees -- which
        is what the 1200ms here was trying to approximate.

        Fixed in both places at once. The same guessed-delay-into-a-race
        shape was in `_pattern_finish` (1200ms floor, 2500ms probe), and
        fixing the instance in front of me and leaving the other standing
        is the single most expensive habit on this project (RULES Part E).
        """
        if self.auto_selftest:
            self.start_exposure_probe(then=self.start_self_test)

    def _calibrate_if_needed(self) -> None:
        """Run the grid unless a saved mapping is still applicable.

        A mapping is only meaningful for the screen it was measured on, so a
        different resolution invalidates it outright. Camera movement
        invalidates it too and cannot be detected from here -- that is what the
        G key is for.
        """
        want = self._screen_size()
        # TRUST THE SAVED CALIBRATION AND GO STRAIGHT TO THE TESTS.
        #
        # "When I'm telling you that the screen is yours, that means I'm
        #  not using the screen and I'm not touching the camera. You can
        #  just run the tests one after one after one and not worry about
        #  recalibration."
        #
        # Exactly so, and it is the difference between a seventy-second
        # iteration and a fifteen-second one. A mapping is invalidated by
        # the camera moving and by nothing else; when a person states that
        # it has not moved, re-measuring it is pure tax on every run.
        #
        # Deliberately a LAUNCH FLAG and never a setting. A saved setting
        # that skips calibration would outlive the afternoon it was true
        # for, and these notes already record what a stale saved value
        # costs: `min_frames` sat at 1 for days because settings.local.json
        # quietly beat the code default, and the exposure poisoned itself
        # run after run the same way. This is true for one launch or not
        # at all.
        if (self.trust_calibration and self.mapping is not None
                and want and self.mapping.screen_size == want):
            self._mapping_measured_now = True
            held = getattr(self.mapping, "held_out", None) or {}
            self._log(f"--trust-cal: using the saved mapping as it stands "
                      f"(held-out {held.get('median_mm', -1):.1f}mm when it "
                      f"was measured), neither remeasuring nor checking it. "
                      f"Valid only while the camera has not moved.")
            self._push_detector()
            self._update_playable(self.mapping)
            self._restore_sensitivity()
            self._after_calibration_ready()
            return
        if self.mapping is not None and want and self.mapping.screen_size != want:
            saved = self.mapping.screen_size
            self._log(f"saved calibration was for a {saved[0]}x{saved[1]} screen, "
                      f"this one is {want[0]}x{want[1]} - remeasuring.")
            self.mapping = None
        elif (self.mapping is not None and self.settings.calibrate_each_session
                and not self._mapping_measured_now):
            # CHECK IT BEFORE THROWING IT AWAY.
            #
            # "Even a 25-second launch is too long... you just need to
            #  flash two squares and see if that corresponds to the
            #  previously saved layout. If that corresponds, that means
            #  the camera did not move."
            #
            # This branch used to delete the mapping outright and pay 24
            # seconds of markers on every launch. The setting exists
            # because a camera nudged between sessions cannot be detected
            # AFTER THE FACT -- which was true when remeasuring was the
            # only way to find out, and is not true now that the mapping
            # can predict where a marker will image and be tested on it.
            if self._check_tried:
                self._log("the two-marker check has already run this "
                          "session; remeasuring properly.")
                self.mapping = None
            else:
                # THE CHECK NEEDS THE TARGET SCREEN, so open it first.
                #
                # `start_mapping_check` refuses without a target window and
                # reports False, and False here means "the camera moved" --
                # so a mapping measured to 0.58mm was thrown away and
                # eighteen markers were shot, every launch where the window
                # had not opened yet. `start_mapping_check`'s own comment
                # says "'I could not check' must never be read as 'nothing
                # moved'"; this is the inverse of that error and it costs
                # twenty-four seconds and a good calibration.
                #
                # Measured 2026-10-16 by spying on this decision: call 1 saw
                # a valid (800,800) mapping with the target window absent,
                # and call 2 saw the mapping already cleared. Six harness
                # launches in a row, each taking 78 seconds against the 30
                # his bench takes when the window happens to be open.
                #
                # Pattern calibration already opens the window for itself
                # ("pattern calibration needs the target screen: opening
                # it"); doing it a second earlier costs nothing and saves
                # the measurement. Bounded: one attempt, then the honest
                # fallback of measuring properly.
                if self.target_window is None and not self._check_opened_screen:
                    self._check_opened_screen = True
                    self._log("the two-marker check needs the target "
                              "screen: opening it, then checking the saved "
                              "calibration rather than remeasuring it.")
                    self._toggle_target_mode()
                    QTimer.singleShot(600, self._calibrate_if_needed)
                    return
                self._check_tried = True
                self.start_mapping_check(self._on_mapping_checked)
                return
        elif self.mapping is not None and self._mapping_measured_now:
            # The whole point of `calibrate_each_session` is to distrust a
            # mapping SAVED BY AN EARLIER SESSION, because a camera that was
            # nudged in between cannot be detected after the fact. It was
            # throwing away the mapping measured forty seconds ago, in this
            # session, by this run.
            #
            # Setup completes more than once -- the noise floor is re-measured
            # after calibration -- so the second pass found a mapping present,
            # saw the setting, and discarded it. `_map_to_screen` then fell
            # back to the flash outline, measured on this bench at 78mm mean
            # error, and every shot landed about an inch out with no warning.
            # That is exactly what "the holes are off by sometimes an inch"
            # was, and the 0.6mm mapping had already been measured and
            # verified before a single trigger was pulled.
            self._log("  keeping the mapping measured in this session "
                      "(held-out error was checked when it was made).")
        elif self.mapping is not None and self._camera_moved():
            self.voice.say("v_camera_moved")
            self._log("the screen is not where the saved mapping says it is - "
                      "the camera has moved. Remeasuring.")
            self.mapping = None
        # One place that tells the display whether a shot can be drawn where it
        # landed. The camera-space minimap keys off this.
        self.scene.calibrated = self.mapping is not None
        if self._blind and self._display_dark and not self._connected_device_is_virtual():
            # THE DISPLAY WAS NOT LIT. The camera is not the suspect, so nothing
            # below -- the blind-camera message, the camera cycling -- applies.
            self._report_display_asleep()
            return
        if self._blind:
            self._log("NOT starting calibration: the screen flash made no "
                      "difference to the camera image, so it is not pointed at "
                      "this screen. Fix that first, then press F5.")
            virtual = self._connected_device_is_virtual()
            if virtual:
                # Worth naming, because the symptom is identical to a camera
                # that is merely aimed wrongly and the fix is not. Cost three
                # runs during development: the real cameras dropped off the USB
                # bus mid-session, the indices shifted underneath, and the app
                # cheerfully opened Canon's placeholder device -- which returns
                # a perfectly valid frame of a picture of a disconnected cable.
                self._log(f"  the connected device is '{virtual}', a VIRTUAL "
                          f"camera. It is software, not a lens: it will never "
                          f"see this or any other screen.")
                self._alert(
                    "THAT IS NOT A REAL CAMERA",
                    f"The app is connected to '{virtual}', which is a virtual "
                    f"device -- software pretending to be a webcam -- so it can "
                    f"never see the screen.\n\n"
                    f"Pick the real camera in the Camera Lab window. If it is "
                    f"not in the list, unplug it and plug it back in: Windows "
                    f"drops USB cameras and renumbers the rest, which is how "
                    f"this one got selected.",
                    show_camera=True)
                return
            # TRY THE OTHER CAMERA, because the app has just MEASURED the
            # one property that identifies the right one (RULES A11).
            #
            # 2026-10-16, and it cost him a morning. He moved the camera to
            # a different USB port -- on my advice, after a wedge -- and
            # Windows renumbered the bus. The app remembers the device BY
            # NAME, that name now belonged to the C920 sitting on top of
            # the monitor with a blind over its lens, and the app opened
            # it, measured a flash difference of 0.0 at every exposure,
            # correctly concluded "not pointed at this screen" and stopped
            # dead at a black screen.
            #
            # A name is not an identity here: in one morning the same
            # physical camera appeared at index 1 then 2, named `USB Video
            # Device`, then `HD Pro Webcam C920`, then `USB Video Device`.
            # What IS an identity is "flashing the monitor changes what
            # this device sees", and that is exactly what has just been
            # measured -- and was being used only to give up.
            #
            # Bounded: each non-virtual device is tried at most once per
            # launch, so this cannot loop (D10), and virtual devices are
            # skipped because they open fine and deliver a picture of a
            # disconnected-cable icon for ever.
            nxt = self._next_camera_to_try()
            if nxt is not None:
                index, label = nxt
                self._log(f"  this camera cannot see the screen, so trying "
                          f"the next one: {label}. Identity here is what a "
                          f"device DOES -- a remembered name follows the USB "
                          f"bus, not the lens.")
                self._set_status(
                    "TRYING ANOTHER CAMERA",
                    f"The connected camera cannot see this screen. "
                    f"Switching to {label} and measuring again.",
                    "#322e16", "#c8c070", "#5e5a2a",
                )
                pos = self.cmb_device.findData(index)
                if pos >= 0:
                    self.cmb_device.setCurrentIndex(pos)   # reconnects itself
                return
            self.voice.say("v_camera_bad")
            self._set_status(
                "THE CAMERA CANNOT SEE THIS SCREEN",
                "Flashing the whole monitor changed nothing in the camera image. "
                "Check it is the right camera, pointed at this screen, unobstructed.",
                "#322e16", "#c8c070", "#5e5a2a",
            )
            self._alert(
                "THE CAMERA CANNOT SEE THIS SCREEN",
                "The whole monitor was flashed black then white and the camera "
                "image did not change at all.\n\n"
                "Point it at this screen, check nothing is covering the lens, and "
                "check it is the right camera. The live view is now showing in the "
                "corner so you can aim it.\n\n"
                "Press F5 when it is looking at the screen.",
                show_camera=True)
            return
        if self.auto_patcal:
            # Once only. Setup completes again every time the noise floor is
            # re-measured -- which pattern calibration itself triggers, via
            # the sensitivity map -- so without this the launch flag starts a
            # second, overlapping run. Later passes come from the repeat
            # scheduler in `_pattern_finish`, which knows how many are left.
            # ...and NOT AT ALL if the two-marker check has just confirmed
            # the saved mapping. That check exists precisely to make this
            # unnecessary: it measured 0.5px against the prediction, which
            # is 0.7mm on the panel, and then eighteen markers ran anyway
            # for another twenty-four seconds. Worse, they ran ON TOP of
            # the exposure probe, and the self test afterwards saw 0 of 12.
            #
            # "Launch should be as fast as possible, especially if the
            #  camera did not move from the previous launch."
            if self._mapping_measured_now and self.mapping is not None:
                # SAY WHICH OF THE TWO REASONS IT IS.
                #
                # This used to print "confirmed by the two-marker check"
                # whenever a mapping existed and was measured this session
                # -- including right after eighteen markers had measured
                # one, because the check had FAILED. Observed in the
                # cold-start log at 22:25: both markers reported NOT FOUND
                # at 15s, pattern calibration ran from 16s to 53s, and at
                # 57s the app announced that the check had confirmed the
                # mapping and the measurement was skipped.
                #
                # Nothing downstream was harmed, which is exactly why it
                # could survive: a log line that describes the wrong cause
                # is read as evidence, and the fastest way to waste an hour
                # is to debug the thing it points at (RULES C3).
                if self._check_confirmed:
                    self._log("the saved calibration was confirmed by the "
                              "two-marker check, so the full measurement is "
                              "skipped. Press C to force it.")
                else:
                    self._log("the mapping was measured in this session "
                              "already, so it is not measured again. "
                              "Press C to force it.")
            elif self._patcal_done == 0 and not self._pat_active:
                QTimer.singleShot(400, self.start_pattern_calibration)
            return
        if self.mapping is None:
            self.start_grid_calibration()

    def _report_display_asleep(self) -> None:
        """The screen search's white frame never got bright: the display is asleep
        or switched off. SAY SO, and WAIT -- do not touch the camera.

        2026-10-19, three launches in a row, each dead about 27 seconds in: the
        monitor had power-saved between sessions, the flash difference read 0.0, and
        the app concluded its healthy Kiyo Pro was blind, cycled to the C920 (blind
        taped over its lens, so it can never see anything), found the capture thread
        stuck while letting go of the first, and shut down. `tools/camera_health.py`
        made the right call in one step on the same camera minutes later: the white
        frame never got bright, so the DISPLAY is the fault.

        Cycling cameras cannot light a screen; it only takes a good camera off the
        bus. Nothing here changes the device, closes the app or starts anything:
        wake the monitor and press F5 twice (it toggles connect: the second press
        reconnects and runs setup again), the existing re-check route. The spoken line is
        the existing recording `v_not_ready` ("Not ready. Check the screen.") -- he
        is downrange, not reading the console, and a dedicated line needs a
        recording, which is his call.
        """
        self._log("NOT starting calibration: the display appears to be ASLEEP or "
                  "switched off -- the 'white' flash frame never got bright (max "
                  f"{LIT_WHITE_MAX} or below), so the screen was not lit. The camera "
                  "has NOT been judged: wake the monitor, then press F5 twice (it "
                  "toggles: disconnect, then connect and run setup again). Not "
                  "trying another camera, and not shutting down.")
        self.voice.say("v_not_ready")
        self._set_status(
            "THE DISPLAY LOOKS ASLEEP",
            "The screen did not light up when it was flashed. Wake the monitor "
            "(move the mouse), then press F5 twice (disconnect, then connect again) "
            "to run setup. The camera has not been judged.",
            "#322e16", "#c8c070", "#5e5a2a",
        )
        self._alert(
            "THE SCREEN IS OFF OR ASLEEP",
            "The monitor did not light when it was flashed, so nothing could be "
            "measured. Wake it, then press F5 twice (disconnect, then connect "
            "again) to run setup.",
            show_camera=False)

    def _next_camera_to_try(self) -> "tuple[int, str] | None":
        """The next real camera that has not already failed to see the screen.

        Each device is offered once per launch, so a machine with four
        cameras makes at most four attempts and then reports honestly
        rather than cycling (RULES D10). Virtual devices are never offered:
        they open, deliver frames, and show a picture of a disconnected
        cable -- with a saturated red X in it, which is what the detector
        hunts.
        """
        here = self.cmb_device.currentData()
        if here is not None:
            self._blind_devices.add(int(here))
        try:
            cams = list(self._cameras or [])
        except Exception:                                    # noqa: BLE001
            return None
        for cam in cams:
            if cam.is_virtual or int(cam.index) in self._blind_devices:
                continue
            return (int(cam.index), cam.name)
        return None

    def _connected_device_is_virtual(self) -> str:
        """The name of the connected device if it is a virtual one, else ""."""
        idx = self.cmb_device.currentData()
        if idx is None:
            return ""
        try:
            for cam in list_cameras():
                if cam.index == idx:
                    return cam.name if cam.is_virtual else ""
        except Exception:
            return ""
        return ""

    # -- calibration from the screen, with no shooting ------------------------

    def start_pattern_calibration(self) -> None:
        """Measure the mapping by photographing the screen's own markers.

        The shot grid asks for twenty-seven trigger pulls and still ends up
        measuring only the part of the screen the laser registers on -- which
        here is two columns out of three, and a projective fit from two columns
        is wrong everywhere else. The monitor is a light source the camera can
        already see, so it can show its own markers and be photographed.

        Twelve markers are used for the fit and six MORE, at positions the fit
        never saw, are kept back to measure it. That held-out check is the thing
        that has been missing: every fit today reported a small error while
        being badly wrong, because it only ever checked itself against its own
        points.
        """
        busy = self._screen_busy("_pat_active", "_calibrating")
        if busy:
            self._log(f"not starting pattern calibration while {busy} is "
                      f"running; waiting for it to finish")
            QTimer.singleShot(900, lambda: self.start_pattern_calibration())
            return
        if self._pat_active:
            # Two runs at once share `_pat_index`, `_pat_samples` and the
            # still-request tag, so the second resets the counter while the
            # first's timers are still in flight: every photograph is then
            # taken for a marker position other than the one on screen, and
            # every marker is reported "not visible to the camera".
            #
            # That is exactly the 18:37 failure -- four corners found, eight
            # invisible -- and it was read as a camera that cannot see the far
            # side of the panel. Two callers can arrive here: the repeat
            # scheduler, and the setup-complete path firing again after the
            # sensitivity map re-measures the noise floor.
            self._log("pattern calibration is already running; ignoring the "
                      "second request")
            return
        if self.capture is None:
            self._log("pattern calibration needs the camera connected")
            return
        if self.target_window is None:
            # Open it and come back, rather than refusing. The markers are
            # drawn ON the target screen, so needing one is not a reason to
            # make the user find the right key -- especially after a setup
            # that has already cost forty seconds and a camera open.
            self._log("pattern calibration needs the target screen: opening it")
            self._toggle_target_mode()
            QTimer.singleShot(700, self.start_pattern_calibration)
            return
        # AND THE APP MUST ACTUALLY OWN THE DISPLAY.
        #
        # This routine is not on the Director yet, so it does not get the
        # `measures_screen` gate for free -- and it is the one that failed
        # all five launches of the 15:37 gate with every marker invisible
        # and 'lit frame peaks at 255, unlit at 255'. Eighteen photographs
        # of a desktop, subtracted from eighteen photographs of the same
        # desktop. Fixing it only where the Director can reach would leave
        # the instance that actually cost the launches standing (RULES E1).
        not_ours = self._screen_is_ours()
        if not_ours:
            self._log(f"pattern calibration will not run: {not_ours}")
            self._alert("THE APP IS NOT ON SCREEN",
                        f"The markers are drawn on the target screen, and "
                        f"something else is in front of it, so every "
                        f"photograph would be of that instead.\n\n"
                        f"{not_ours}")
            return
        self.voice.say("v_calibrating",
                       valid=lambda: bool(self._pat_active or self._grid_active))
        rect = self.target_window.rect()
        w, h = rect.width(), rect.height()
        self._pat_points = marker_points(w, h, cols=4, rows=3, margin=0.08)
        # Visit the four CORNERS first. Four correspondences give a homography
        # and every later marker's position can then be predicted -- but only
        # if those four are in general position, and the natural row-major
        # order starts with a whole row, which is collinear and defines
        # nothing. Pruning still works in grid order; only the display order
        # changes, and samples carry their screen coordinates.
        corners = [0, 3, 8, 11]                       # 4x3 grid, row-major
        self._pat_order = corners + [i for i in range(len(self._pat_points))
                                     if i not in corners]
        # Held-out points, deliberately offset from the fit grid so they fall
        # BETWEEN fitted positions rather than near them.
        self._pat_check = marker_points(w, h, cols=3, rows=2, margin=0.22)
        self._pat_samples = []
        self._pat_checks = []
        self._pat_misses = []
        self._pat_retried = -1
        self._pat_quad_warned = False
        self._pat_mask_retried = False
        # NOT re-read here: by this point `calibrate_each_session` has already
        # set self.mapping to None. It is captured at load instead.
        self._capture_pose_facts()
        self._pat_index = 0
        self._pat_active = True
        self._calibrating = True
        self.scene.calibrating = False

        # The shooting exposure renders the screen as a ghost on purpose. For
        # photographing it, use a view that can actually see a monitor.
        #
        # Brighter than the screen SEARCH uses (-10). That value exists because
        # the flash method subtracts a white frame from a black one and wants
        # the room to vanish; here the markers are white squares on black and
        # the job is simply to see them, including on the far side of the panel
        # where everything is several times dimmer. At -10 the full-screen
        # flash found only the bright near half -- the mask ended at camera x
        # 686 on a panel reaching 920 -- and the far column was never searched.
        self._apply_preset("calib", exposure=-6.0)
        # Remember what each layer's visibility WAS, and restore exactly that
        # when the run ends. This used to be two hand-written lists -- twelve
        # layers hidden here, seven restored in `_pattern_finish` -- and the
        # five that fell through the gap stayed hidden for the rest of the
        # session. `modes` was one of them, so after any calibration pressing
        # 1, 2 or 3 started a mode that was never drawn: reported as "pressing
        # 2 or 3 brings nothing on screen". `alert` was another, which means
        # the app's own warnings were invisible precisely when it had most to
        # say. A list of things to undo must be built by the code that does
        # them, never maintained in parallel by hand.
        self._pat_hidden = {}
        for name in ("calibration", "signal", "stats", "help", "holes",
                     "hits", "camera", "modes", "bglabel", "alert", "prompt"):
            layer = self.scene.get(name)
            if layer is not None:
                self._pat_hidden[name] = layer.visible
                layer.visible = False
        # Bring the target screen to the FRONT first. It can be open but
        # lowered -- Escape puts it behind other windows on purpose -- and a
        # pattern drawn on a window nobody can see produces twelve markers that
        # all land in the same place, which is exactly what happened.
        if not force_to_front(self.target_window):
            self._log("  the target window could NOT be brought to the front "
                      "(Windows refused it). The camera is watching whatever "
                      "is in front of it; click it or alt-tab to it.")
        QApplication.processEvents()

        pattern = self.scene.get("pattern")
        pattern.caption = "measuring the camera's view of this screen - no shooting needed"
        pattern.progress(0, len(self._pat_points) + len(self._pat_check),
                         "locating the monitor")
        pattern.show_at(None)
        self.scene.mark_dirty()
        self._pat_mask = None
        self._log(f"pattern calibration: {len(self._pat_points)} markers to fit, "
                  f"{len(self._pat_check)} held back to check it. No trigger pulls.")
        # First light the WHOLE screen and photograph it, to learn where the
        # monitor is in the camera image. Without that, a reflection of the
        # marker on something beside the screen is indistinguishable from the
        # marker -- it is brighter and larger, being seen face-on rather than
        # at 70 degrees. One run put all twelve markers at camera x of
        # 1050-1180 when the panel only spans 225-920.
        pattern.show_full()
        self.scene.mark_dirty()
        QApplication.processEvents()
        # A full second and a half before the first photograph. The preset
        # above moves the camera from the shooting exposure, where the screen
        # is deliberately a ghost, to one that can see a monitor -- and that
        # change has to reach the sensor. Photographed too early, the flash
        # difference is tiny and the monitor is "located" at 3% of the frame;
        # the markers are then looked for inside a sliver and none is found.
        QTimer.singleShot(1500, lambda: self._request_still("pat_full_lit", timeout_s=5.0))

    def _pattern_all(self) -> list[tuple[float, float]]:
        order = getattr(self, "_pat_order", None) or range(len(self._pat_points))
        return [self._pat_points[i] for i in order] + self._pat_check

    def _pattern_step(self) -> None:
        if not self._pat_active:
            return
        points = self._pattern_all()
        if self._pat_index >= len(points):
            self._pattern_finish()
            return
        pattern = self.scene.get("pattern")
        kind = ("fit" if self._pat_index < len(self._pat_points)
                else "checking the fit")
        pattern.progress(self._pat_index, len(points),
                         f"{len(self._pat_samples)} found so far   ({kind})")
        pattern.show_at(points[self._pat_index])
        self.scene.mark_dirty()
        QApplication.processEvents()      # get it onto the glass before the photo
        QTimer.singleShot(420, lambda: self._request_still("pat_lit", timeout_s=4.0))

    def _on_pattern_still(self, tag: str, frame: object) -> None:
        if not self._pat_active:
            return
        pattern = self.scene.get("pattern")
        if tag == "pat_full_lit":
            self._pat_lit = frame
            pattern.show_at(None)                 # all black
            self.scene.mark_dirty()
            QApplication.processEvents()
            QTimer.singleShot(700,
                              lambda: self._request_still("pat_full_dark", timeout_s=5.0))
            return
        if tag == "pat_full_dark":
            self._pat_mask = screen_mask(self._pat_lit, frame)
            self._pat_lit = None
            if self._pat_mask is not None:
                # Sanity-check it. Measured on this bench the monitor fills
                # 25-35% of the frame; runs that went wrong produced 3% (the
                # exposure had not settled) and 94% (the screen was lighting
                # the whole room). A mask that wrong is worse than none: at 3%
                # no marker is inside it, and at 94% it protects nothing. The
                # strength filter, the grid-order check and the predicted
                # position all still apply without it.
                frac = float((self._pat_mask > 0).mean())
                if not (0.08 <= frac <= 0.60):
                    self._log(f"  the monitor could not be located plausibly "
                              f"({frac * 100:.0f}% of the frame): ignoring that "
                              f"and relying on marker strength, grid order and "
                              f"predicted position instead")
                    self._pat_mask = None
            if self._pat_mask is None and self.screen_quad is not None:
                # Fall back to the quad the SCREEN SEARCH found, rather
                # than to nothing.
                #
                # The search photographs white-minus-black at a dark
                # fixed exposure (-10), chosen precisely so the room
                # vanishes and the monitor is the only thing that
                # changes. This mask is derived at -6, which is brighter
                # on purpose so the far column of markers is still
                # visible -- and in a bright room that same brightness
                # makes the whole room light up with the screen.
                #
                # Measured 14:51 on 2026-10-15, one run, seconds apart:
                #
                #   screen search at -10:  17.1% of frame, confidence 1.00
                #   this mask at -6:       85% of frame, rejected
                #
                # With no mask the search ran over the whole frame and a
                # bright object at camera (1000,670) -- well outside the
                # panel, which ends at x=627 -- won TEN of the eighteen
                # markers. Held-out error came out at 414mm and the
                # calibration was rejected outright.
                #
                # A mask measured in the conditions designed for it
                # beats no mask. It is still only advisory: strength,
                # grid order and predicted position all apply on top.
                try:
                    import numpy as _np
                    import cv2 as _cv2
                    poly = self.screen_quad.polygon
                    h, w = frame.shape[:2]
                    m = _np.zeros((h, w), dtype=_np.uint8)
                    pts = _np.array(poly, dtype=_np.float32)
                    # A little outward margin, as the detector ROI uses:
                    # the quad is fitted and its corners sit slightly
                    # inside the real panel.
                    c = pts.mean(axis=0)
                    pts = (c + (pts - c) * 1.06).astype(_np.int32)
                    _cv2.fillConvexPoly(m, pts, 255)
                    frac2 = float((m > 0).mean())
                    if 0.05 <= frac2 <= 0.60:
                        self._pat_mask = m
                        self._log(f"  using the monitor outline the screen "
                                  f"search found instead ({frac2 * 100:.0f}% "
                                  f"of the frame): it was measured at a dark "
                                  f"exposure where the room goes black.")
                except Exception as exc:
                    self._log(f"  could not reuse the screen outline: {exc}")
            if self._pat_mask is None:
                self._log("  could not locate the monitor in the camera image; "
                          "markers will be searched for across the whole frame")
            else:
                import numpy as _np
                frac = float((self._pat_mask > 0).mean())
                self._log(f"  monitor located: it fills {frac * 100:.0f}% of the "
                          f"camera frame. Markers will only be looked for inside it.")
                try:
                    import cv2 as _c
                    RECORDINGS.mkdir(parents=True, exist_ok=True)
                    _c.imwrite(str(RECORDINGS / "pat_screen_mask.png"), self._pat_mask)
                except Exception:
                    pass
                _ = _np
            QTimer.singleShot(200, self._pattern_step)
            return
        if tag == "pat_lit":
            self._pat_lit = frame
            pattern.show_at(None)                     # same screen, marker off
            self.scene.mark_dirty()
            QApplication.processEvents()
            QTimer.singleShot(420, lambda: self._request_still("pat_dark", timeout_s=4.0))
            return

        points = self._pattern_all()
        i = self._pat_index
        screen = points[i] if i < len(points) else (0.0, 0.0)
        pred = self._pattern_expected(screen)
        found = find_marker(self._pat_lit, frame, mask=self._pat_mask,
                            near=pred)
        widened = False
        if found is None and pred is not None:
            # A wrong prediction must cost accuracy, never a marker. Search
            # the whole frame -- the screen mask, the strength filter and the
            # grid-order check all still apply -- and say how far out the
            # prediction was. Before this, a bad prediction and a camera that
            # cannot see the far third of the panel produced the identical
            # message, and the wrong one was believed for a whole session.
            found = find_marker(self._pat_lit, frame, mask=self._pat_mask,
                                near=None)
            widened = found is not None
        if pred is not None:
            if found is None:
                self._log(f"      predicted camera ({pred[0]:.0f},{pred[1]:.0f}) "
                          f"+-{pred[2]:.0f}px: nothing found there OR anywhere "
                          f"in the frame")
            else:
                off = math.hypot(found[0] - pred[0], found[1] - pred[1])
                self._log(f"      predicted camera ({pred[0]:.0f},{pred[1]:.0f}) "
                          f"+-{pred[2]:.0f}px, found at ({found[0]:.0f},"
                          f"{found[1]:.0f}): {off:.0f}px out"
                          + ("  <-- OUTSIDE the disc, found by widening the "
                             "search" if widened else ""))
        if found is None and self._pat_retried != i:
            # One more photograph before giving up on this position. A miss is
            # usually a timing slip rather than a marker the camera cannot see,
            # and a lost marker costs a column of the fit -- far more than the
            # half second a retry takes.
            self._pat_retried = i
            self._pat_lit = None
            self._log(f"  marker {i + 1}/{len(points)}: not seen, photographing "
                      f"it once more")
            QTimer.singleShot(250, self._pattern_step)
            return
        if i < 3 or found is None:
            # The first few, and every failure. Twelve markers reported at the
            # same camera position looked like a detection bug and was actually
            # a window that was never on screen; one saved frame says which.
            try:
                import cv2 as _c
                RECORDINGS.mkdir(parents=True, exist_ok=True)
                if self._pat_lit is not None:
                    _c.imwrite(str(RECORDINGS / f"pat_{i:02d}_lit.png"), self._pat_lit)
                if frame is not None:
                    _c.imwrite(str(RECORDINGS / f"pat_{i:02d}_dark.png"), frame)
            except Exception:
                pass
        if found is None:
            self._pat_misses.append(screen)
            # SAY WHAT WAS MEASURED, not just that it failed (RULES C3).
            #
            # "not visible to the camera" with no number cannot be
            # debugged, only re-run -- and it reads as a statement about
            # the CAMERA'S VIEW when it is usually a statement about
            # brightness. The two have opposite fixes: one sends you to
            # move a camera that is aimed correctly, the other to the
            # exposure or the gain. The brightest thing in the difference
            # image tells them apart in one line, and `find_marker` needs
            # a peak of 12.
            self._log(f"  marker {i + 1}/{len(points)} at screen "
                      f"({screen[0]:.0f},{screen[1]:.0f}): not visible to "
                      f"the camera -- {self._pat_miss_detail(frame)}")
        else:
            cx, cy, strength = found
            if i < len(self._pat_points):     # order is corners-first, then the rest
                self._pat_samples.append(Sample(screen=screen, camera=(cx, cy),
                                                peak=strength))
                kind = "fit"
            else:
                self._pat_checks.append((screen, (cx, cy)))
                kind = "check"
            self._log(f"  marker {i + 1}/{len(points)} ({kind}) screen "
                      f"({screen[0]:.0f},{screen[1]:.0f}) -> camera "
                      f"({cx:.1f},{cy:.1f}), strength {strength:.0f}")
        self._pat_index += 1
        self._pat_lit = None
        if self._pat_corners_suspect():
            return
        QTimer.singleShot(60, self._pattern_step)

    def _pat_miss_detail(self, dark) -> str:
        """The numbers behind a marker that was not found.

        The brightest point of the lit frame, of the difference, and what
        the threshold is -- so "the camera is not pointed at the screen",
        "the sensor is turned down" and "the photograph was taken before
        the marker was drawn" are three different lines rather than one.
        """
        try:
            import cv2 as _c
            lit = self._pat_lit
            if lit is None or dark is None:
                return ("no pair to compare: the lit photograph never "
                        "arrived")
            diff = _c.absdiff(lit, dark)
            covered = ""
            flash = self._flash
            if flash is not None:
                try:
                    if flash.isVisible():
                        covered = ("; THE APP'S OWN FULLSCREEN COVER IS "
                                   "OVER THE SCREEN, so this photographed "
                                   "the cover and not the marker")
                except Exception:                            # noqa: BLE001
                    pass
            return (f"brightest point of the difference {int(diff.max())} "
                    f"(needs 12); lit frame peaks at {int(lit.max())}, "
                    f"unlit at {int(dark.max())}{covered}")
        except Exception as exc:                             # noqa: BLE001
            return f"could not measure it ({exc!r})"

    def _pat_corners_suspect(self) -> bool:
        """If the four corners are implausible, blame the mask and start over.

        Measured across seventeen runs on this bench, and the correlation is
        total:

            mask 44%    corners valid     held-out 0.6 - 1.0mm
            no mask     corners valid     held-out 0.6 - 1.1mm
            mask 29%    corners INVALID   held-out 386mm
            mask 52%    corners INVALID   held-out 298mm

        Both disasters had a screen mask that passed the generic 8-60%
        plausibility band and was still wrong -- too small at 29%, so the
        panel was clipped and corners were found at the clip edge; too large
        at 52%, so the room came in and a reflection won one. And ten runs
        that proceeded with NO mask at all were uniformly good, because the
        corner check, the prediction disc, the strength filter and the
        grid-order prune already do this job.

        So the mask is now a hypothesis the run tests rather than an
        assumption it carries: find the corners with it, and if they do not
        look like a screen, throw the mask away and find them again. Costs
        four photographs, about two seconds, and only when something is
        already wrong. Deleting the mask outright is not the answer -- it
        exists because a reflection beside the monitor once supplied all
        twelve 'markers' -- but it must not be able to silently ruin a run.
        """
        if self._pat_mask is None or self._pat_mask_retried:
            return False
        if self._pat_index != 4:
            return False          # only at the end of the corner phase
        corners = list(self._pat_samples)
        if len(corners) < 4:
            # Corners MISSING, not merely misplaced. A mask that clips the
            # panel hides them completely, so the quad test never gets four
            # points to judge and the run limps on with whatever is left --
            # which is the "6 points, held-out 386mm" failure exactly: six
            # markers in two columns, the degenerate fit.
            why = (f"only {len(corners)} of the 4 corners were found inside "
                   f"the monitor outline")
        else:
            ok, why = plausible_quad([(s.screen, s.camera) for s in corners])
            if ok:
                return False
        self._pat_mask_retried = True
        self._pat_mask = None
        self._pat_samples = []
        self._pat_checks = []
        self._pat_misses = []
        self._pat_index = 0
        self._pat_retried = -1
        self._pat_quad_warned = False
        self._log(f"  the four corners are not a plausible screen ({why}). "
                  f"The monitor outline is the most likely culprit, so it is "
                  f"being discarded and the corners measured again without it.")
        QTimer.singleShot(120, self._pattern_step)
        return True

    def _pat_reject(self, mapping, med: float, worst: float,
                    mm_per_px: "float | None", why: str = "") -> None:
        """Throw away a mapping that failed its own held-out check.

        Keeping the previous mapping is not an option either: it was measured
        at a camera pose that may no longer be current, and a stale mapping is
        the same lie with an older date on it. The app goes uncalibrated and
        says so, which makes the next shot report "shot seen, but there is
        nowhere to draw it" -- a diagnosis a person can act on, instead of a
        hole a foot from where they aimed.
        """
        mm = (lambda v: v * mm_per_px) if mm_per_px else (lambda v: v * 0.25)
        n = len(getattr(mapping, "samples", []) or [])
        cols = len({round(s.screen[0]) for s in (mapping.samples or [])})
        self._log(f"CALIBRATION REJECTED: held-out error {mm(med):.0f}mm "
                  f"(worst {mm(worst):.0f}mm) from {n} markers in {cols} "
                  f"column(s). That is far too wrong to draw holes from, so "
                  f"it is being discarded rather than used.")
        if why:
            self._log(f"  why: {why}")
        if cols < 3:
            self._log(f"  {n} markers in {cols} columns is the degenerate fit: "
                      f"a projective mapping from that few distinct screen "
                      f"x-values is accurate on those columns and wrong "
                      f"everywhere else. Usually the far side of the panel was "
                      f"not found.")
        self.mapping = None
        self.scene.calibrated = False
        # Re-arm the once-per-state announcements: a rejection later in a
        # session is a new event, not a repeat of the first one.
        self._unmapped_shots = 0
        self._said_once.discard("unmapped-shot")
        self._set_status(
            "NOT CALIBRATED  -  press C to measure the screen again",
            f"The measurement came out {mm(med):.0f}mm wrong on points it had "
            f"not seen, so it was thrown away. No holes will be drawn until a "
            f"good one is measured.",
            "#322e16", "#c8c070", "#5e5a2a")
        self._alert(
            "CALIBRATION FAILED - NOT USING IT",
            f"The screen measurement was checked against markers it had never "
            f"seen and came out {mm(med):.0f}mm out, worst {mm(worst):.0f}mm. "
            f"A good one measures about 1mm.\n\n"
            f"It has been discarded. Holes will NOT be drawn until this "
            f"succeeds -- a hole in the wrong place is worse than no hole, "
            f"because it looks like bad shooting.\n\n"
            f"Press C to try again.",
            seconds=14.0)
        # AND PUBLISH A VERDICT, which this path did not.
        #
        # 2026-10-18, and it is the fault that cost him the session rather
        # than the rejection itself. The alert above runs for 14 seconds and
        # then the screen goes quiet; nothing else said anything. Between the
        # rejection at 10:04:39 and the end of his 42-shot string there is not
        # one line in the log about the app's own state, and no verdict was
        # ever published, so from the firing line a correct refusal to draw
        # holes was indistinguishable from the whole app being broken.
        #
        # This project has published exactly this fault once before: "the
        # verdict was not published when nothing was detected -- an early
        # return in the self test skipped it in the one case the verdict
        # exists for" (2026-10-15). The verdict board is the app's one
        # unmissable channel and the one state that most needs it was the one
        # state that reached it by no path at all.
        #
        # The verdict is what persists: it stays on screen until dismissed
        # when it is not READY, Q recalls it, and `selfcheck` now carries a
        # BLOCKED finding for "no mapping" so the board NAMES the problem and
        # the key. Nothing here decides anything -- the decision is in
        # `selfcheck.assess`, which is pure and tested.
        try:
            self._publish_verdict()
        except Exception as exc:                         # noqa: BLE001
            # A failure to report must never also be a crash: the shooter is
            # standing at the line with a pistol and no holes.
            self._log(f"  could not publish the verdict after the rejection "
                      f"({exc!r}) -- the alert above is all he will see")

    def _pattern_expected(self, screen) -> "tuple[float, float, float] | None":
        """Where this marker should appear, from the markers already found.

        Four correspondences define a homography, so from the fifth marker
        onwards its camera position is predictable to within the lens error.
        Searching a disc around that prediction is a far stronger constraint
        than any brightness threshold, and unlike the screen mask it needs no
        tuning against the room: a reflection, a noise blob, or a photograph
        taken a frame too early all fall outside it.

        Earned the hard way. Runs that went wrong did so by finding something
        at camera (785,371) for a marker that belonged at (878,189), or at
        strength 18 in a position matching no screen location at all.
        """
        if len(self._pat_samples) < 4:
            return None
        # Four points define a homography only if they are in general
        # position. The grid is shown corners-first precisely so this is true
        # by the fifth marker, but check rather than assume.
        if (len({round(s.screen[0]) for s in self._pat_samples}) < 2
                or len({round(s.screen[1]) for s in self._pat_samples}) < 2):
            return None
        # And they must look like the corners of a screen. With exactly four
        # correspondences the homography fits them perfectly whatever they
        # are, so a corner found on a reflection yields a confident, wholly
        # wrong prediction -- and then every later marker falls outside the
        # disc and is reported invisible. Four found, eight "invisible" is
        # precisely that, and it was seen on hardware.
        if len(self._pat_samples) == 4:
            ok, why = plausible_quad([(s.screen, s.camera)
                                      for s in self._pat_samples])
            if not ok:
                if not self._pat_quad_warned:
                    self._pat_quad_warned = True
                    self._log(f"  the four corner markers are not a plausible "
                              f"screen ({why}): not predicting positions from "
                              f"them. Each marker will be searched for across "
                              f"the frame and the grid-order check will drop "
                              f"whichever one is wrong.")
                return None
            if not self._pat_quad_warned:
                self._pat_quad_warned = True
                self._log(f"  the four corner markers form a plausible screen "
                          f"({why}): predicting the rest from them")
        try:
            import cv2 as _cv2
            import numpy as _np
            src = _np.float32([s.screen for s in self._pat_samples])
            dst = _np.float32([s.camera for s in self._pat_samples])
            h, _mask = _cv2.findHomography(src, dst, 0)
            if h is None:
                return None
            q = _cv2.perspectiveTransform(
                _np.float32([[[float(screen[0]), float(screen[1])]]]), h)[0][0]
        except Exception:
            return None
        w = float(self.settings.width or 1280)
        # Generous: wide enough for lens distortion and a rough early fit,
        # narrow enough that nothing else in the room is inside it.
        #
        # Widened from 0.10 to 0.20 of the frame width. At 0.10 (128px here) a
        # prediction only slightly off cost the marker outright, and the app
        # reported it as "not visible to the camera" -- indistinguishable from
        # a camera that genuinely cannot see that part of the panel, which is
        # what sent an afternoon after the camera position. The caller now also
        # falls back to a full search, so a wrong prediction costs accuracy and
        # never a marker.
        return (float(q[0]), float(q[1]), w * 0.20)

    def _pattern_finish(self) -> None:
        self._pat_active = False
        self._calibrating = False
        pattern = self.scene.get("pattern")
        pattern.stop()
        for name, was in (getattr(self, "_pat_hidden", None) or {}).items():
            layer = self.scene.get(name)
            if layer is not None:
                layer.visible = was
        self._pat_hidden = {}
        self.scene.mark_dirty()
        self._apply_preset("laser", with_exposure=False)

        rect = self.target_window.rect() if self.target_window else None
        size = (rect.width(), rect.height()) if rect else (0, 0)
        mm = 940.0 / size[0] if size[0] else None
        if len(self._pat_samples) < 6:
            self._log(f"pattern calibration failed: only {len(self._pat_samples)} "
                      f"of {len(self._pat_points)} markers were visible.")
            self.voice.say("v_not_ready")
            self._alert("THE CAMERA CANNOT SEE THE SCREEN WELL ENOUGH",
                        f"Only {len(self._pat_samples)} of {len(self._pat_points)} "
                        f"markers were found. The camera needs a clearer view of "
                        f"the whole monitor.", show_camera=True)
            return

        # Reject markers far weaker than their siblings before anything else.
        #
        # A marker measures 176-231 on this screen. One run recorded a "marker"
        # at strength 14, in a position that matched no screen location -- the
        # photograph had been taken before the square was painted, so the
        # difference image held nothing but noise and the strongest noise blob
        # won. An absolute threshold cannot catch that, because the far side of
        # the panel is legitimately several times dimmer than the near side;
        # the comparison that works is against the other markers in the same
        # session.
        if len(self._pat_samples) >= 4:
            strengths = sorted(s.peak for s in self._pat_samples)
            med = strengths[len(strengths) // 2]
            floor_strength = max(20.0, med * 0.25)
            weak = [s for s in self._pat_samples if s.peak < floor_strength]
            for s in weak:
                self._log(f"  REJECTED marker at screen ({s.screen[0]:.0f},"
                          f"{s.screen[1]:.0f}): strength {s.peak:.0f} against a "
                          f"median of {med:.0f} - that is noise, not the marker")
            if len(self._pat_samples) - len(weak) >= 6:
                self._pat_samples = [s for s in self._pat_samples
                                     if s.peak >= floor_strength]

        # Drop markers that contradict the grid's own order before fitting.
        # One reflection found near the frame edge took a fit from 0.6mm to
        # 215mm, and a least-squares fit cannot tell a reflection from a
        # marker: it just splits the difference between them.
        found = [((s.screen[0], s.screen[1]), (s.camera[0], s.camera[1]))
                 for s in self._pat_samples]
        by_screen = {(round(a), round(b)): i for i, (a, b) in
                     enumerate(self._pat_points)}
        ordered: list = [None] * len(self._pat_points)
        for sc, cam in found:
            idx = by_screen.get((round(sc[0]), round(sc[1])))
            if idx is not None:
                ordered[idx] = (sc, cam)
        kept, dropped = prune_out_of_order(ordered, 4, 3)
        for sc, cam, off in dropped:
            self._log(f"  REJECTED marker at screen ({sc[0]:.0f},{sc[1]:.0f}): "
                      f"camera ({cam[0]:.1f},{cam[1]:.1f}) is {off:+.0f}px out of "
                      f"line along its row - a reflection, not the marker")
        if len(kept) >= 6:
            peaks = {(round(s.screen[0]), round(s.screen[1])): s.peak
                     for s in self._pat_samples}
            self._pat_samples = [
                Sample(screen=sc, camera=cam,
                       peak=peaks.get((round(sc[0]), round(sc[1])), 0.0))
                for sc, cam in kept]
        elif dropped:
            self._log(f"  too many markers were out of line ({len(dropped)}); "
                      f"keeping them all and letting the held-out check judge")

        frame = (self.settings.width, self.settings.height)
        mapping, note = Mapping.fit_validated(self._pat_samples, size, frame)
        if mapping is None:
            self._log("pattern calibration failed: the markers do not form a mapping")
            return

        self._log("pattern calibration complete: " + mapping.describe(mm_per_px=mm))
        self._log("  " + note)
        self._report_pattern_check(mapping, mm)

    def _report_pattern_check(self, mapping, mm_per_px) -> None:
        """Measure the mapping on points it was never fitted to.

        This is the number that matters and the one nothing has reported all
        day. A fit's error against its own samples says only that it can
        reproduce them.
        """
        if not self._pat_checks:
            self._log("  no held-out markers were visible, so this mapping is "
                      "UNVERIFIED - treat it with suspicion")
            return
        errs = []
        self._log("  held-out check (markers the fit never saw):")
        for screen, camera in self._pat_checks:
            got = mapping.to_screen(camera[0], camera[1])
            if got is None:
                continue
            d = math.hypot(got[0] - screen[0], got[1] - screen[1])
            errs.append((d, screen))
            unit = f" = {d * mm_per_px:5.1f}mm" if mm_per_px else ""
            self._log(f"    screen ({screen[0]:6.0f},{screen[1]:6.0f}) -> "
                      f"({got[0]:6.0f},{got[1]:6.0f})   off by {d:5.1f}px{unit}")
        if not errs:
            return
        ds = sorted(d for d, _s in errs)
        worst_d, worst_s = max(errs, key=lambda e: e[0])
        med = ds[len(ds) // 2]
        u = (lambda v: f"{v:.1f}px = {v * mm_per_px:.1f}mm") if mm_per_px else (
            lambda v: f"{v:.1f}px")
        self._log(f"  HELD-OUT ERROR: median {u(med)}, worst {u(worst_d)} "
                  f"at screen ({worst_s[0]:.0f},{worst_s[1]:.0f})")
        # THE DECISION IS MADE ON THE MEDIAN, NOT THE WORST.
        #
        # 2026-10-18: five held-out markers at 0.1-2.1mm and one at 196.8mm
        # threw away a 1.3mm mapping, and he then fired 42 shots that were all
        # detected and left no hole. The one bad marker was a false find from a
        # saturated sensor, not a bad fit. `Mapping._reject_outliers` has
        # judged the FIT against the median since the shot grid, for exactly
        # this reason; the held-out check was using the one statistic a single
        # false find owns outright. Provenance of every constant is at
        # `judge_held_out`.
        verdict = judge_held_out(errs, mm_per_px)
        # REPORT THE OUTLIER LOUDLY. A silent drop is how a reflection once
        # won ten of eighteen markers, and how 215mm of error was announced as
        # a success. One line per wild marker, naming the marker, the size and
        # the most likely cause -- never "an outlier was ignored".
        for d, d_mm, sc in verdict.wild:
            self._log(
                f"  OUTLIER among the held-out markers: screen "
                f"({sc[0]:.0f},{sc[1]:.0f}) is {d:.0f}px = {d_mm:.0f}mm out "
                f"against a median of {verdict.median_mm:.1f}mm. That is not "
                f"a fit error, it is a DIFFERENT POINT: almost certainly the "
                f"marker was never found and something else in the frame was "
                f"measured instead (a saturated sensor leaves the finder no "
                f"contrast to discriminate on). The fit is judged on the "
                f"median, so this one marker does not decide it -- but if "
                f"holes land wrong near screen ({sc[0]:.0f},{sc[1]:.0f}), "
                f"this line is why.")
        if verdict.wild and verdict.usable:
            self._log(f"  {len(verdict.wild)} of {verdict.points} held-out "
                      f"markers are wild; up to {verdict.allowed_wild} is "
                      f"read as a false find rather than a bad mapping. More "
                      f"than that and this calibration would be rejected.")
        ok = verdict.good
        # A mapping that fails its own held-out check is REJECTED, not adopted
        # with a warning. `ok` was computed here and then ignored: the mapping
        # was installed, saved to disk and used to draw holes whatever it said.
        #
        # Measured 2026-10-14 20:06, and the shooter met it directly ("I just
        # did a few shots and holes are way, way off"):
        #
        #   pattern calibration complete: 6 points, RMS 575px (140.8mm)
        #   HELD-OUT ERROR: median 386.1mm, worst 1143.2mm
        #   saved, but the held-out error is large.
        #
        # Six points is the degenerate two-column fit these notes warn about
        # by name. The app had already PROVEN the mapping was a metre wrong
        # and used it regardless, so the shooter was told, in effect, that his
        # aim was off by fifteen inches.
        #
        # 40mm is the bar for keeping one. Well above the 1mm a good run
        # measures and well above any plausible lens or pose error, so it only
        # ever catches a fit that has genuinely collapsed -- but far below the
        # size of hole displacement a person can see. Holes nobody draws are a
        # diagnosis; holes a foot from the shot are a mystery.
        #
        # It is applied to the MEDIAN now, plus a cap on how many held-out
        # markers may be wild (see `judge_held_out`): a median that bad means
        # the mapping is wrong everywhere, and several wild markers mean the
        # finder cannot be trusted on this panel. Either still rejects.
        usable = verdict.usable
        if not usable:
            self._pat_reject(mapping, med, worst_d, mm_per_px, verdict.why)
            return
        self.mapping = mapping
        self._mapping_measured_now = True   # measured here, not loaded
        self._check_confirmed = False        # measured, not confirmed
        try:
            # Keep the verification WITH the mapping. A fit's own error means
            # little; this is the number that says whether to trust it, and it
            # should outlive the log window it was printed to.
            mapping.held_out = {
                "median_px": med, "worst_px": worst_d,
                "median_mm": med * (mm_per_px or 0),
                "worst_mm": worst_d * (mm_per_px or 0),
                "points": [{"screen": list(sc), "camera": list(cm)}
                           for sc, cm in self._pat_checks],
            }
        except Exception:
            pass
        # A falloff measured EARLIER for a camera in this same position is
        # worth keeping: it is a property of the room, the pose and the
        # panel, which is what this mapping describes. Carried across so the
        # shooter does not have to teach the same lesson twice.
        #
        # This MUST happen before the save, and the first version did it
        # after. The carried rows reached the live mapping, so the session
        # that calibrated logged "re-using the sensitivity ... 5 regions"
        # and worked -- while the file on disk was written with none. Every
        # pattern calibration silently emptied the very measurement that was
        # persisted to end the bootstrap deadlock, so the next cold launch
        # started from nothing again. Measured 2026-10-15: calibration.json
        # held 0 regions where the pre-run backup held 5.
        self._carry_pose_facts(mapping)
        self._save_mapping(mapping, "the calibration")
        self._push_detector()
        self._update_playable(mapping)
        QTimer.singleShot(1500, self._restore_sensitivity)
        # NO sensitivity map from the markers. This used to build one here, on
        # the reasoning that the markers measure how brightly the camera sees
        # each part of the screen and that this is the same falloff that kills
        # the laser dot on the far side. It is not, and the live numbers say so
        # plainly (2026-10-14 gun test, 63 trigger pulls):
        #
        #   marker strength across the whole screen   227 .. 245   (8% spread)
        #   laser peak, median, middle of the screen         84
        #   laser peak, median, right third                  47   (2x down)
        #   laser blob area, middle / right third      404px / 149px
        #
        # A 9px white square saturates the sensor wherever it is drawn, so its
        # peak measures panel illumination and is nearly flat. A laser dot's
        # peak measures its POINT SPREAD -- the same energy over three times
        # fewer pixels on the far side, because of foreshortening -- and that
        # is what the detector actually thresholds. Feeding the flat one in
        # produced a gain map of all ones: the normalisation that exists
        # precisely to save the right third did nothing, silently, while
        # reporting success. 43% of audible trigger pulls went undetected.
        #
        # The falloff that matters can only be measured from real shots, so it
        # is learned from them as they arrive -- see `_learn_sensitivity`.
        self._log("  sensitivity: marker brightness is nearly uniform across "
                  "the screen (it measures illumination, not the laser's point "
                  "spread), so no gain map is built from it. It is learned from "
                  "real shots instead.")
        # A new mapping means new camera geometry, so peaks recorded against the
        # old one describe a screen that is no longer where it was.
        self._sens_shots = []
        self._sens_by_click = 0
        self._sens_applied = 0
        if self.capture is not None:
            self._set_gain_map(None, "withdrawing the sensitivity map")
        self._patcal_done += 1
        if self.patcal_repeats > 1:
            h = getattr(mapping, "held_out", None) or {}
            self._patcal_results.append(
                f"pass {self._patcal_done}: markers {len(mapping.samples):2d}  "
                f"fit {mapping.rms_after * (mm_per_px or 0):5.1f}mm  |  held-out "
                f"median {h.get('median_mm', -1):5.1f}mm worst {h.get('worst_mm', -1):5.1f}mm")
            self._log("REPEAT " + self._patcal_results[-1])
            if self._patcal_done < self.patcal_repeats:
                # Straight into the next pass. No reconnect, no preset churn:
                # the camera is already in the right state.
                QTimer.singleShot(1500, self.start_pattern_calibration)
                return
            self._log("=== repeatability ===")
            for line in self._patcal_results:
                self._log("  " + line)

        # Re-measure the noise floor. Calibration has just driven the camera
        # through a bright preset and back, and the threshold that matters is
        # the one measured in the state the shooting actually happens in. A
        # floor left over from before leaves min_peak at 42 when real hits on
        # this hardware measure 20-43, which rejects most of them.
        QTimer.singleShot(1200, lambda: self._start_noise_floor(self.settings.calibrate_seconds))
        if self.auto_selftest and ok:
            # Exposure first, THEN the self test, because the self test
            # measures the detector and the exposure is what the detector
            # is measuring through.
            #
            # STARTED WHEN THE FLOOR ABOVE FINISHES, not 2500ms later.
            #
            # The comment that used to be here said the re-measure was
            # "waited for rather than raced" because the probe hands the
            # chain on itself and the self test refuses to flash into a
            # measurement. That is two guessed delays (1200 and 2500) and
            # two downstream guards standing in for a sequence, and it was
            # wrong in the way those always are: the harness shortened his
            # 3s floor to 1s, so the probe arrived after it and everything
            # looked fine. Measured with HIS timings, the probe is asked
            # for mid-floor on every launch and queued -- and the only
            # reason nothing breaks is the Director refusing it.
            #
            # An outside review predicted this before it was measured, from
            # reading the two delays. There is nothing to guess now: the
            # floor's completion starts the probe.
            self._probe_after_floor = self.start_self_test
        if ok:
            self._log("  saved. The mapping is verified on points it never saw.")
            self._set_status(
                f"CALIBRATED FROM THE SCREEN  -  held-out error "
                f"{(med * (mm_per_px or 0)):.1f}mm",
                "Measured by photographing the screen, with no shooting, and "
                "checked against markers the fit never saw.",
                "#17331c", "#7fe08a", "#2e6b38")
        else:
            self._log("  saved, but the held-out error is large. Expect holes to "
                      "land visibly off, worst away from the marker positions.")

    def _check_view_quality(self, quad: object) -> None:
        """Is the camera seeing the screen WELL, not merely seeing it?

        "Sees it at all" and "sees it well enough to shoot at" are different
        questions, and only the second one matters. Thresholds are from measured
        sessions on this hardware: a good view gave a flash difference of 65-84
        and filled 27-42% of the frame at confidence 1.00. A session that had
        been knocked gave 15.3, 11.8% and confidence 0.50 -- still "found", but
        the calibration that followed would have been built on sand.

        Confidence 0.50 specifically means the quadrilateral fit failed and the
        code fell back to a bounding rectangle, which for an off-axis camera is
        never the real shape of the screen.
        """
        problems: list[str] = []
        flash = getattr(self, "_flash_strength", None)
        if flash is not None and flash < 18.0:
            problems.append(f"the screen flash barely registered ({flash:.0f}; "
                            f"a good view measures 40 or more)")
        if quad is not None:
            if quad.coverage < 0.08:
                problems.append(f"the screen fills only {quad.coverage * 100:.0f}% of the "
                                f"camera's view, so each camera pixel covers a lot of "
                                f"screen and precision suffers (12% or more is comfortable)")
            if quad.confidence < 0.75:
                problems.append("its outline could not be fitted as a quadrilateral, "
                                "which usually means it is partly out of view or obscured")
        if not problems:
            self._clear_alert()      # whatever was wrong has been fixed
            return
        self._log("WARNING: the camera's view of the screen is poor:")
        for why in problems:
            self._log(f"  - {why}")
        self._alert(
            "THE CAMERA IS NOT SEEING THE SCREEN WELL",
            "Calibration can still run, but accuracy will suffer and shots may be "
            "missed.\n\n"
            + "\n".join(f"- {w}" for w in problems)
            + "\n\nUse the preview in the Camera Lab window. It now shows a dashed "
              "box marked PUT THE SCREEN IN HERE, the detected screen in green, "
              "and which way to move the camera.\n\n"
              "Press F5 to re-check, or just carry on and shoot the grid.",
            show_camera=True)

    def _camera_moved(self, tolerance_px: float = 60.0) -> bool:
        """Has the camera moved since the saved mapping was measured?

        Compares the quadrilateral the saved mapping implies against the one the
        flash just found. The flash outline is far too crude to calibrate with --
        that is the whole reason for the shot grid -- but it is perfectly good
        for "has this shifted by 200 pixels", which is the question here.

        Measured: the mapping from 20:18 implied a top-left corner at (102,178);
        after the camera was nudged the flash found (84,372). 194 px apart. The
        consequences were holes in the wrong place AND a screen mask sitting over
        part of the room, which fired the gunshot sound at a red object in it.
        """
        if self.mapping is None or self.screen_quad is None:
            return False
        implied = self.mapping.screen_rect_in_camera()
        if implied is None:
            return False
        found = self.screen_quad.corners
        offsets = sorted(math.dist(a, b) for a, b in zip(implied, found))
        median = offsets[len(offsets) // 2]      # robust to one badly-found corner
        if median > tolerance_px:
            self._log(f"  screen outline sits {median:.0f}px from where the saved "
                      f"mapping expects it (tolerance {tolerance_px:.0f}px)")
            return True
        return False

    def _screen_size(self) -> tuple[int, int] | None:
        """The pixel size of the surface targets are drawn on."""
        if self.target_window is not None:
            r = self.target_window.rect()
            return (r.width(), r.height())
        scr = self.screen() or QApplication.primaryScreen()
        if scr is None:
            return None
        g = scr.geometry()
        return (g.width(), g.height())

    def _calibration_grid_points(self, rect, cols: int, rows: int) -> list:
        """Where the rings of a shooting calibration go, on a screen `rect`.

        ONE definition for both shooting calibrations (G measures the mapping,
        U measures the brightness): two copies of an inset rule is how one of
        them ends up with its top row under the instruction band.

        Inset by the two bands that are drawn OVER everything, or the top
        row of rings lands underneath the instruction band and cannot be seen
        at all -- which reads as the app asking you to shoot at nothing.
        The pad covers the ring's own tick marks, which reach 1.5 radii out
        from its centre: clearing the band by the centre alone still clips
        the top tick.
        """
        pad = int(rect.height() * 0.025)
        band = band_height_for(rect.height()) + pad
        strip = bottom_strip_for(rect.height()) + pad
        usable_h = max(100, rect.height() - band - strip)
        return [(x, y + band) for x, y in
                grid_points(rect.width(), usable_h, cols, rows)]

    def start_grid_calibration(self, cols: int = 3, rows: int = 3) -> None:
        """Measure the mapping from shots instead of from the screen's outline.

        This is the accurate route and the one that survives a camera which
        cannot see the whole screen: it needs no corners, only correspondences.
        """
        if self._sgrid_active:
            # Two shooting calibrations at once would share one calibration
            # layer, one set of relaxed thresholds (`_grid_saved_levels`) and
            # one pair of shot routes. The brightness grid refuses to start
            # over this one (`_sgrid_refusal`); this is the other direction.
            self._log("not starting grid calibration while the brightness grid "
                      "is running; press U to stop it first")
            return
        if self.capture is None:
            self._log("connect a camera first")
            return
        if self.target_window is None:
            # Opening the target screen is a one-shot: either it exists on
            # the next pass or something is wrong that waiting will not
            # cure. Bounded, because the version without `_grid_attempt`
            # would retry for ever if the window never opened, saying
            # nothing while it did (RULES D10).
            if self._grid_attempt >= 3:
                self._grid_attempt = 0
                self._log("cannot start grid calibration: the target screen "
                          "did not open after three attempts. Press F11.")
                return
            self._grid_attempt += 1
            self._toggle_target_mode()
            QTimer.singleShot(600, lambda: self.start_grid_calibration(cols, rows))
            return
        self._grid_attempt = 0
        self.voice.say("v_calibrating",
                       valid=lambda: bool(self._pat_active or self._grid_active))
        rect = self.target_window.rect()
        self.scene.calibrating = True
        self._grid_points = self._calibration_grid_points(rect, cols, rows)
        self._grid_cols, self._grid_rows = cols, rows
        self._grid_samples = []
        self._grid_shots = []
        self._grid_active = True
        # Not zero: the stall check measures against this, and an unset clock
        # reads as "nothing seen for 1304090s" the instant the grid appears.
        self._grid_last_seen = time.monotonic()
        self._relax_for_calibration()
        self._push_detector()      # drop the mask while we measure
        layer = self.layers["calibration"]
        layer.start(self._grid_points, need=self._grid_repeats)
        layer.message = ""
        self._clear_holes("calibration grid started")
        self.scene.mark_dirty()
        self._log(f"grid calibration: fire {self._grid_repeats} shots at each of the "
                  f"{len(self._grid_points)} rings. SPACE skips, BACKSPACE redoes, "
                  f"ESC abandons.")

    def _grid_record(self, shot: Shot) -> None:
        """Take several shots per point and keep the median.

        One shot per point makes the fit only as good as that trigger pull. In
        the first live run six of nine points landed inside 2.4mm and three were
        12 to 33mm out -- pure aim scatter, and the fit has no way to tell a
        sloppy shot from a true one. The median of three is robust: one flyer
        cannot move it at all, which is exactly the failure being guarded
        against, and it costs about twenty extra seconds once.
        """
        layer = self.layers["calibration"]
        i = layer.index
        if i >= len(self._grid_points):
            return
        # Ignore anything arriving in the instant after the ring moved. One
        # trigger pull can produce a late extra detection, and once the point
        # has advanced that detection is attributed to the NEW ring while the
        # gun was still pointed at the old one. Observed live as shots 164 to
        # 361px adrift -- almost exactly the grid spacing, which is the
        # fingerprint of this and not of a bad shot.
        if time.monotonic() < self._grid_ready_at:
            return
        self._grid_shots.append((shot.x, shot.y, float(shot.peak)))
        n = len(self._grid_shots)
        self._grid_last_seen = time.monotonic()
        layer.stall = ""
        layer.landed = n
        # Visible, because a calibration that will not advance is otherwise
        # impossible to diagnose: these never reached the normal shot log.
        self._log(f"    shot {n} at ({shot.x:.0f},{shot.y:.0f}) peak {shot.peak} "
                  f"area {shot.area:.0f}px")
        want = self._grid_repeats
        # Only the most RECENT shots may form a group. Without a window, a
        # stray from thirty seconds ago still sits in the list competing with
        # the shots being fired now.
        recent = self._grid_shots[-self._grid_window:]
        group = self._agreeing_group(recent, want)
        if group is None:
            if n >= self._grid_max_shots:
                self._log(f"  point {i + 1}/{len(self._grid_points)}: {n} shots never "
                          f"agreed - skipping it rather than poisoning the fit")
                self._grid_shots = []
                layer.message = ""
                self._grid_skip()
                return
            # How many of the recent shots DO agree with each other, so the
            # pips show progress rather than jumping from nothing to done.
            layer.agreeing = self._best_cluster(recent)
            if n < want:
                layer.message = f"shot {n} seen - fire again at the same ring"
            else:
                # This is the case that used to read as a dead app: shots are
                # arriving and being thrown away for disagreeing. Say the
                # number, because it tells the shooter whether they are
                # flinching or the detector is seeing something else entirely.
                spread = self._spread_px(recent[-want:])
                layer.message = (
                    f"{n} shots seen here, but they are {spread:.0f}px apart and "
                    f"must agree within 25px. Keep shooting at this ring."
                )
            last = self._grid_shots[-1]
            layer.signal = (f"last shot seen at camera ({last[0]:.0f},{last[1]:.0f}) "
                            f"peak {last[2]:.0f}")
            self._update_prompt()
            self.scene.mark_dirty()
            return

        xs = sorted(pt[0] for pt in group)
        ys = sorted(pt[1] for pt in group)
        ps = sorted(pt[2] for pt in group)
        mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
        peak = ps[len(ps) // 2]
        spread = max(math.hypot(x - mx, y - my) for x, y, _ in group)
        wrong = self._out_of_grid_order(i, mx, my)
        if wrong:
            # Three shots agreeing is not proof of a laser: a static object
            # agrees with itself perfectly. The grid's own geometry is the
            # check that catches it.
            self._log(f"  point {i + 1}/{len(self._grid_points)}: REJECTED - {wrong}")
            layer.message = (f"those shots landed somewhere impossible: {wrong} "
                             f"Something steady and dim is being mistaken for the "
                             f"laser. Fire again, or press SPACE to skip this ring.")
            layer.signal = ""
            layer.agreeing = 0
            layer.landed = 0
            self._grid_shots = []
            self._signal_note("a steady object is being read as a laser hit", "warn")
            self.scene.mark_dirty()
            return
        self._grid_samples.append(Sample(screen=self._grid_points[i], camera=(mx, my),
                                        peak=peak))
        layer.done.append(i)
        extra = f" ({n - len(group)} stray ignored)" if n > len(group) else ""
        self._log(f"  point {i + 1}/{len(self._grid_points)}: median of {len(group)} "
                  f"agreeing shots at camera ({mx:.1f},{my:.1f}), "
                  f"spread {spread:.0f}px{extra}")
        self._grid_shots = []
        layer.reset_point()
        self._grid_advance()

    def _out_of_grid_order(self, index: int, cx: float, cy: float, *,
                           points: "list | None" = None,
                           samples: "list | None" = None,
                           cols: "int | None" = None,
                           rows: "int | None" = None) -> str:
        """Does this camera position contradict the points already recorded?

        `points` / `samples` / `cols` / `rows` default to the shot grid's own
        (`_grid_*`); the brightness grid passes its own so the two runs share
        this one rule and never one run's state.

        The grid is a grid: along a row, screen x only increases, so the camera
        position must advance steadily in whatever direction that row runs in
        the camera image -- whichever way round the camera is mounted. The same
        holds down a column. Two already-recorded points in a row define that
        direction, and the third either continues it or is not a laser.

        Returns a human sentence describing the violation, or "" if it is fine.
        """
        cols = getattr(self, "_grid_cols", 3) if cols is None else cols
        rows = getattr(self, "_grid_rows", 3) if rows is None else rows
        points = self._grid_points if points is None else points
        samples = self._grid_samples if samples is None else samples
        if cols < 3 and rows < 3:
            return ""
        recorded = {points.index(tuple(s.screen)): s
                    for s in samples
                    if tuple(s.screen) in points}

        for axis, stride, length, name in (("row", 1, cols, "left to right"),
                                           ("col", cols, rows, "top to bottom")):
            if axis == "row":
                base = (index // cols) * cols
                pos = index % cols
            else:
                base = index % cols
                pos = index // cols
            line = [(k, recorded[base + k * stride])
                    for k in range(length)
                    if base + k * stride in recorded and k != pos]
            if len(line) < 2:
                continue
            (k0, s0), (k1, s1) = line[0], line[-1]
            dx = s1.camera[0] - s0.camera[0]
            dy = s1.camera[1] - s0.camera[1]
            span = (dx * dx + dy * dy) ** 0.5
            if span < 20 or k1 == k0:
                continue
            # Project onto the direction this line runs in the camera image.
            def proj(px_, py_):
                return ((px_ - s0.camera[0]) * dx + (py_ - s0.camera[1]) * dy) / span
            per_step = proj(*s1.camera) / (k1 - k0)
            want = per_step * (pos - k0)
            got = proj(cx, cy)
            # Half a step of slack: generous for lens distortion, nowhere near
            # enough to let a point land on the wrong side of its neighbours.
            if abs(got - want) > abs(per_step) * 0.5:
                return (f"along this {axis} the camera sees the screen advance "
                        f"{name}; this point should sit about {want:.0f}px along "
                        f"that line and it measured {got:.0f}px.")
        return ""

    def _best_cluster(self, shots: list[tuple[float, float, float]],
                      tol: float = 25.0) -> int:
        """The largest number of these shots that agree with each other.

        Purely for feedback: the fit itself uses _agreeing_group. Showing a
        count of 2 of 3 is the difference between "it is working, one more" and
        a screen that looks identical whether or not anything is being heard.
        """
        best = 0
        for x, y, _p in shots:
            n = sum(1 for (ox, oy, _q) in shots if math.hypot(ox - x, oy - y) <= tol)
            best = max(best, n)
        return best

    def _spread_px(self, shots: list[tuple[float, float, float]]) -> float:
        if len(shots) < 2:
            return 0.0
        return max(math.hypot(a[0] - b[0], a[1] - b[1])
                   for a in shots for b in shots)

    def _check_grid_stall(self) -> None:
        """Nothing is arriving. Say so, and say what to do about it.

        A calibration that will not advance is the single most demoralising
        state this app has: the ring sits there, shots go into it, and the
        screen says nothing at all. After a few seconds of silence the shooter
        deserves to be told whether to keep shooting, skip the point, or stop.
        """
        if self._sgrid_active:
            # The brightness grid shares this tick, its stall message and its
            # silence-to-seconds rule, and adds the two exits a flag-driven
            # routine needs and the Director gives for free: a camera that
            # went away and an overall deadline (RULES D4).
            if self.capture is None:
                self._sgrid_end("the camera was disconnected")
                return
            quiet = time.monotonic() - self._sgrid_last_seen
            if quiet >= self.SGRID_IDLE_LIMIT_S:
                self._log(f"the brightness grid saw no shot for {quiet:.0f}s, so it "
                          f"is giving the screen back. Press U to start it again.")
                self._sgrid_end(f"nothing was shot for {quiet:.0f}s")
                return
            stop_key = "U stops it"
        elif self._grid_active:
            quiet = time.monotonic() - getattr(self, "_grid_last_seen", 0.0)
            stop_key = "ESC stops"
        else:
            return
        layer = self.layers["calibration"]
        if quiet < 6.0:
            return
        heard = self.corroborator.heard_not_seen if self.listener.running else 0
        if heard:
            msg = (f"{heard} shot(s) were HEARD but not seen. The laser is "
                   f"firing - it is landing where the camera cannot see it. "
                   f"Try the next ring with SPACE, or press "
                   f"{'U to stop' if self._sgrid_active else 'ESC'} and re-aim the camera.")
        else:
            msg = (f"nothing seen for {quiet:.0f}s. If you are firing: SPACE skips "
                   f"this ring, {stop_key}. If not, carry on.")
        if layer.stall != msg:
            layer.stall = msg
            self.scene.mark_dirty()

    def _relax_for_calibration(self) -> None:
        """Drop the threshold while the grid is being shot, and say so.

        The thresholds that protect a game are too strict for calibration. A dot
        on the far side of an obliquely-viewed screen is dimmer and spread over
        fewer pixels -- measured mean peak 34 there against 62 near the centre --
        so a min_peak of 40 rejects the very shots needed to measure that
        falloff. The sensitivity map fixes it afterwards, but it is BUILT from a
        completed calibration, so it cannot help during one.

        Being permissive here is safe in a way it would not be during play:
        a point is only recorded when three shots agree within 25px, and noise
        does not land in the same place three times.
        """
        if self._grid_saved_levels is not None:
            return
        self._grid_saved_levels = (self.sp_threshold.value(), self.sp_minpeak.value())
        floor = self._noise_peak or float(self.sp_threshold.value())
        self.sp_threshold.setValue(int(max(6, floor + 3)))
        self.sp_minpeak.setValue(int(max(8, floor + 6)))
        self._log(f"  calibration thresholds relaxed to {self.sp_threshold.value()}/"
                  f"{self.sp_minpeak.value()} (from {self._grid_saved_levels[0]}/"
                  f"{self._grid_saved_levels[1]}) so dim far-side shots still register")
        self._push_detector()

    def _restore_levels(self) -> None:
        if self._grid_saved_levels is None:
            return
        thr, peak = self._grid_saved_levels
        self._grid_saved_levels = None
        self.sp_threshold.setValue(thr)
        self.sp_minpeak.setValue(peak)
        self._push_detector()

    def _agreeing_group(self, shots: list[tuple[float, float, float]], want: int,
                        radius: float = 25.0) -> "list[tuple[float, float, float]] | None":
        """The largest set of shots that landed on the same spot, or None.

        Taking the median of the first three shots assumes at least two of them
        are real. The screen mask is deliberately off during calibration -- it
        is derived from the mapping being replaced, so trusting it could reject
        the very shots meant to fix it -- which leaves the detector watching the
        whole room. Measured live: four of nine points had shots up to 361px
        apart, because something red in the room fired while the shooter was
        aiming, and those points then dragged the fit.

        Requiring agreement instead of assuming it costs nothing when shooting
        is clean (three shots, done) and recovers automatically when it is not.

        The radius doubles as an accuracy control, because once strays are gone
        the limiting error is the shooter's own group. One camera pixel is about
        1.4mm on a 4K screen at this geometry, so a trio spread over 39px -- seen
        live -- carries ~53mm of scatter into a point meant to be exact. At 25px
        a tight group still finishes in three shots (measured spreads were 4 to
        19px for six of nine points) while a loose one simply asks for another,
        spending shots only where they are actually needed.
        """
        if len(shots) < want:
            return None
        best: list[tuple[float, float, float]] = []
        for cx, cy, _p in shots:
            near = [s for s in shots if math.hypot(s[0] - cx, s[1] - cy) <= radius]
            if len(near) > len(best):
                best = near
        return best if len(best) >= want else None

    def _grid_skip(self) -> None:
        layer = self.layers["calibration"]
        self._grid_shots = []
        layer.message = ""
        if layer.index < len(self._grid_points):
            layer.skipped.append(layer.index)
            x, y = self._grid_points[layer.index]
            self._log(f"  point {layer.index + 1} skipped - screen ({x:.0f},{y:.0f}) "
                      f"is not usable")
        self._grid_advance()

    def _grid_back(self) -> None:
        layer = self.layers["calibration"]
        self._grid_shots = []
        layer.message = ""
        if layer.index <= 0:
            return
        layer.index -= 1
        i = layer.index
        if i in layer.done:
            layer.done.remove(i)
            self._grid_samples = [sm for sm in self._grid_samples
                                  if sm.screen != self._grid_points[i]]
        if i in layer.skipped:
            layer.skipped.remove(i)
        self.scene.mark_dirty()

    def _grid_advance(self) -> None:
        layer = self.layers["calibration"]
        layer.index += 1
        layer.reset_point()
        layer.stall = ""
        self._grid_last_seen = time.monotonic()
        self._grid_ready_at = time.monotonic() + 0.35
        self.scene.mark_dirty()
        if layer.index >= len(self._grid_points):
            self._grid_finish()

    def _grid_abort(self) -> None:
        self._grid_active = False
        self.scene.calibrating = False
        self._restore_levels()
        self.layers["calibration"].stop()
        self.scene.mark_dirty()
        self._log("grid calibration abandoned; the previous mapping is still in use")

    # -- the BRIGHTNESS grid (key U): nine spots, three agreeing shots each ------
    #
    # WHY IT EXISTS. A fresh install has no sensitivity gain map, so the dim
    # (right-hand) side of an obliquely viewed screen loses shots, and a
    # seasoned install stays permanently better than a new one on identical
    # hardware: the map is learned only from REAL shots and the learner refuses
    # anything under the laser/noise boundary of 30, which is every shot on the
    # side that needs the map. The shot grid (G) measures exactly what the map
    # needs but pattern calibration, which supersedes it, never fires a shot.
    # This is a grid whose ONLY job is the falloff: 27 shots, about two minutes.
    #
    # WHAT IS DIFFERENT FROM G, on purpose:
    #   * It never touches the mapping. It CONSUMES the saved one, to know where
    #     the screen is in camera space and where each spot must image.
    #   * The screen MASK STAYS ON. `start_grid_calibration` drops it (via
    #     `_roi_polygon` returning None while `_grid_active`) because the
    #     mapping it is about to replace cannot be trusted to reject anything.
    #     Here the mapping IS the trusted input, so the mask stays and rejects
    #     the blown-out window beside his monitor, where a reflection once
    #     supplied ten of eighteen "markers". `_roi_polygon` is therefore
    #     deliberately NOT told about `_sgrid_active`.
    #   * It sets any gain map aside while it runs. A peak measured through a
    #     map is in the map's units, and a map built from flattened peaks is a
    #     flat map. `_sgrid_prior_gain` puts it back on every exit that does not
    #     replace it.
    #   * Its shots never reach the mode, so they score nothing, spend no
    #     magazine round, draw no hole, and the play-time learner never sees
    #     them: `_on_shot` routes them to `_sgrid_record` before `host.shot`,
    #     exactly where `_grid_record` is routed.
    #
    # THE ONE EXIT. Every way out -- the nine spots done, U pressed again, the
    # idle limit, the camera gone, the target screen closed, the app closing --
    # goes through `_sgrid_end`, which restores the thresholds. Cleanup written
    # per exit path was missed five times on this project; the Director's
    # `finally` blocks are the whole argument for one place.
    #
    # DEBT, STATED: this is the THIRD routine (after pattern calibration and the
    # shot grid) still driven by flags and the HUD tick instead of a Director
    # Job. It owns the screen by raising a SCREEN_OWNERS flag, which is the
    # convention the Director replaces; it is owed the same migration.

    # Nothing shot at all for this long: give the screen back. A forgotten
    # grid would otherwise hold `_screen_busy()` for ever and starve every
    # Director routine (RULES D4: every wait has a deadline).
    SGRID_IDLE_LIMIT_S = 90.0
    # How far from where the saved mapping says a spot must image a group may
    # sit, as a fraction of the screen's WIDEST extent in the camera image.
    # Aim scatter measured on this bench is 2-33 mm of a 940 mm panel (under
    # 4%); the neighbouring spot is ~44% away. 12% is three times the worst
    # scatter and a quarter of the spacing.
    SGRID_AIM_FRAC = 0.12

    def _sgrid_refusal(self) -> str:
        """Why the brightness grid cannot start now, or "" if it can.

        Its OWN flag is ignored here on purpose: "am I already running" is
        answered by the guard at the top of `start_sensitivity_grid`, and the
        central list answers it for everyone else. Answering it in both places
        would make each look protected by the other.
        """
        busy = self._screen_busy("_sgrid_active") or self.director.busy_label()
        if busy:
            return f"{busy} is running"
        if self.capture is None:
            return "no camera is connected"
        if self.target_window is None:
            return "the target screen is not open (press F11)"
        if self.mapping is None:
            return ("there is no calibration to place the spots with: "
                    "press G (or C) first")
        if not self.mapping.screen_rect_in_camera():
            return ("the calibration gives no screen outline to confine the "
                    "map to, and an unconfined map amplifies the room")
        if not self._floor_measured:
            return "the noise floor has not been measured in this room yet"
        return ""

    def start_sensitivity_grid(self, cols: int = 3, rows: int = 3) -> None:
        """Shoot nine spots to measure how bright the laser looks across the screen."""
        # EXCLUSIVITY FIRST, before any side effect. Two runs sharing one
        # index photographed every marker for the wrong position and cost two
        # sessions (pattern calibration, 2026-10-14).
        if self._sgrid_active:
            self._log("brightness grid is already running; ignoring the second request")
            return
        why = self._sgrid_refusal()
        if why:
            self._log(f"not starting the brightness grid: {why}")
            if self._screen_busy("_sgrid_active") or self.director.busy_label():
                # A measurement owns the screen: a panel now would be part of
                # what it measures (B2), so the signal meter's note only.
                self._signal_note(f"brightness grid not started: {why}", "warn")
            else:
                self._alert("THE BRIGHTNESS GRID DID NOT START", why,
                            seconds=8.0, dim=False)
            return
        not_ours = self._screen_is_ours()
        if not_ours:
            self._log(f"the brightness grid will not run: {not_ours}")
            self._alert("THE APP IS NOT ON SCREEN", not_ours, seconds=8.0, dim=False)
            return
        # No voice line: `v_calibrating` says "Do not shoot yet", which is the
        # opposite of this routine, and a new recording is not made here.
        rect = self.target_window.rect()
        self._sgrid_cols, self._sgrid_rows = cols, rows
        self._sgrid_points = self._calibration_grid_points(rect, cols, rows)
        self._sgrid_samples = []
        self._sgrid_shots = []
        self._sgrid_prior_gain = self._gain_in_force
        self._sgrid_ready_at = 0.0
        # Not zero: the stall check measures against this.
        self._sgrid_last_seen = time.monotonic()
        # Raised BEFORE anything else changes, so that `_sgrid_end` covers
        # every step below if one of them raises.
        self._sgrid_active = True
        try:
            self.scene.calibrating = True
            self._relax_for_calibration()
            if self._sgrid_prior_gain is not None:
                self._set_gain_map(
                    None, "the brightness grid is measuring the RAW laser, "
                          "and a peak seen through a map is in the map's units")
            layer = self.layers["calibration"]
            layer.start(self._sgrid_points, need=self._grid_repeats,
                        title="BRIGHTNESS", unit="spot", footer=(
                            "SPACE skips a spot you cannot hit or the camera cannot see.",
                            "BACKSPACE redoes the last spot.  U again stops this."))
            layer.message = ""
            layer.stall = ""        # the last run's advice is not this run's
            self._clear_holes("brightness grid started")
            board = self.scene.get("verdict")
            if board is not None and board.visible:
                board.clear()        # a lit panel is part of what this measures
            self.scene.mark_dirty()
        except Exception as exc:                          # noqa: BLE001
            self._log(f"the brightness grid could not start ({exc!r}); "
                      f"nothing was changed")
            self._sgrid_end("it could not start")
            return
        self._log(f"brightness grid: fire {self._grid_repeats} shots at each of the "
                  f"{len(self._sgrid_points)} spots. SPACE skips, BACKSPACE redoes, "
                  f"U again stops. Shots here score nothing and spend no round; "
                  f"the screen mask stays on.")

    def _sgrid_record(self, shot: Shot) -> None:
        """Take shots at this spot until three agree, and keep the median."""
        layer = self.layers["calibration"]
        i = layer.index
        if i >= len(self._sgrid_points):
            return
        # The instant after the spot moved: a late extra detection of the last
        # pull would be filed against the NEW spot (see `_grid_record`).
        if time.monotonic() < self._sgrid_ready_at:
            return
        self._sgrid_shots.append((shot.x, shot.y, float(shot.peak)))
        n = len(self._sgrid_shots)
        self._sgrid_last_seen = time.monotonic()
        layer.stall = ""
        layer.landed = n
        self._log(f"    shot {n} at ({shot.x:.0f},{shot.y:.0f}) peak {shot.peak} "
                  f"area {shot.area:.0f}px")
        want = self._grid_repeats
        recent = self._sgrid_shots[-self._grid_window:]
        group = self._agreeing_group(recent, want)
        if group is None:
            if n >= self._grid_max_shots:
                self._log(f"  spot {i + 1}/{len(self._sgrid_points)}: {n} shots never "
                          f"agreed - skipping it rather than teaching the map from noise")
                self._sgrid_skip()
                return
            layer.agreeing = self._best_cluster(recent)
            if n < want:
                layer.message = f"shot {n} seen - fire again at the same spot"
            else:
                # The case that reads as a dead app: shots arriving and being
                # thrown away for disagreeing. Say the number.
                spread = self._spread_px(recent[-want:])
                layer.message = (
                    f"{n} shots seen here, but they are {spread:.0f}px apart and "
                    f"must agree within 25px. Keep shooting at this spot.")
            last = self._sgrid_shots[-1]
            layer.signal = (f"last shot seen at camera ({last[0]:.0f},{last[1]:.0f}) "
                            f"peak {last[2]:.0f}")
            self._update_prompt()
            self.scene.mark_dirty()
            return
        xs = sorted(pt[0] for pt in group)
        ys = sorted(pt[1] for pt in group)
        ps = sorted(pt[2] for pt in group)
        mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
        peak = ps[len(ps) // 2]
        spread = max(math.hypot(x - mx, y - my) for x, y, _ in group)
        # THREE SHOTS AGREEING IS NOT PROOF OF A LASER: a steady object agrees
        # with itself perfectly. Two independent checks, because each alone has
        # a blind spot. The grid's own order catches an object that appears
        # AFTER real spots were recorded, but needs two recorded spots in a
        # line to judge, so an object present from the first spot is
        # "consistent" with itself for ever. The saved mapping says where THIS
        # spot must image, which needs no earlier spot at all.
        wrong = (self._sgrid_off_aim(i, mx, my)
                 or self._out_of_grid_order(
                     i, mx, my, points=self._sgrid_points,
                     samples=self._sgrid_samples,
                     cols=self._sgrid_cols, rows=self._sgrid_rows))
        if wrong:
            self._log(f"  spot {i + 1}/{len(self._sgrid_points)}: REJECTED - {wrong}")
            layer.message = (f"those shots landed somewhere impossible: {wrong} "
                             f"Fire again, or press SPACE to skip this spot.")
            layer.signal = ""
            layer.agreeing = 0
            layer.landed = 0
            self._sgrid_shots = []
            self._signal_note("a steady object is being read as a laser hit", "warn")
            self.scene.mark_dirty()
            return
        self._sgrid_samples.append(Sample(screen=self._sgrid_points[i],
                                          camera=(mx, my), peak=peak))
        layer.done.append(i)
        extra = f" ({n - len(group)} stray ignored)" if n > len(group) else ""
        self._log(f"  spot {i + 1}/{len(self._sgrid_points)}: median of {len(group)} "
                  f"agreeing shots at camera ({mx:.1f},{my:.1f}), peak {peak:.0f}, "
                  f"spread {spread:.0f}px{extra}")
        self._sgrid_shots = []
        layer.reset_point()
        self._sgrid_advance()

    def _sgrid_off_aim(self, index: int, cx: float, cy: float) -> str:
        """Is this group nowhere near where the saved mapping puts this spot?

        "" if it is fine (or cannot be judged), else a sentence. The mapping is
        the trusted input to this routine, so a group far from its prediction
        is either a steady object or a camera that moved since the calibration
        -- and in both cases the map must not be taught from it.
        """
        mp = self.mapping
        corners = mp.screen_rect_in_camera() if mp is not None else None
        if mp is None or not corners:
            return ""
        sx, sy = self._sgrid_points[index]
        predicted = mp.to_camera(sx, sy)
        if predicted is None:
            return ""
        extent = max(max(c[0] for c in corners) - min(c[0] for c in corners),
                     max(c[1] for c in corners) - min(c[1] for c in corners))
        limit = self.SGRID_AIM_FRAC * extent
        off = math.hypot(cx - predicted[0], cy - predicted[1])
        if off <= limit:
            return ""
        return (f"the saved calibration says this spot images at camera "
                f"({predicted[0]:.0f},{predicted[1]:.0f}) and these shots landed "
                f"{off:.0f}px away (limit {limit:.0f}px). Either something steady "
                f"is being read as the laser, or the camera has moved since the "
                f"calibration (press C to remeasure).")

    def _sgrid_skip(self) -> None:
        layer = self.layers["calibration"]
        self._sgrid_shots = []
        layer.message = ""
        if layer.index < len(self._sgrid_points):
            layer.skipped.append(layer.index)
            x, y = self._sgrid_points[layer.index]
            self._log(f"  spot {layer.index + 1} skipped - screen ({x:.0f},{y:.0f}) "
                      f"is not usable")
        self._sgrid_advance()

    def _sgrid_back(self) -> None:
        layer = self.layers["calibration"]
        self._sgrid_shots = []
        layer.message = ""
        if layer.index <= 0:
            return
        layer.index -= 1
        i = layer.index
        if i in layer.done:
            layer.done.remove(i)
            self._sgrid_samples = [sm for sm in self._sgrid_samples
                                   if tuple(sm.screen) != self._sgrid_points[i]]
        if i in layer.skipped:
            layer.skipped.remove(i)
        self.scene.mark_dirty()

    def _sgrid_advance(self) -> None:
        layer = self.layers["calibration"]
        layer.index += 1
        layer.reset_point()
        layer.stall = ""
        self._sgrid_last_seen = time.monotonic()
        self._sgrid_ready_at = time.monotonic() + 0.35
        self.scene.mark_dirty()
        if layer.index >= len(self._sgrid_points):
            self._sgrid_finish()

    def _sgrid_finish(self) -> None:
        """The last spot is done: build the map from what agreed, or say why not."""
        layer = self.layers["calibration"]
        samples = list(self._sgrid_samples)
        total = len(self._sgrid_points)
        skipped = len(layer.skipped)
        order = [self._sgrid_points.index(tuple(s.screen)) for s in samples]
        cols = self._sgrid_cols
        # Four spots in a single row are three regions, not a screen: the
        # learner demands the same coverage (a map fitted to one strip calls
        # the rest dim and amplifies its noise until it looks like signal).
        spans = (len({i % cols for i in order}) >= 2
                 and len({i // cols for i in order}) >= 2)
        if len(samples) < self.LEARN_MIN_BINS or not spans:
            self._sgrid_end("too few spots registered")
            peaks = ", ".join(f"{s.peak:.0f}" for s in samples) or "none"
            self._log(f"brightness grid: only {len(samples)} of {total} spots "
                      f"registered (peaks: {peaks}); {self.LEARN_MIN_BINS} spread "
                      f"over the screen are needed. Detection stays as it was.")
            self._alert(
                "BRIGHTNESS NOT MEASURED",
                f"Only {len(samples)} of {total} spots registered a shot, and "
                f"{self.LEARN_MIN_BINS} spread over the screen are needed to "
                f"say how the laser's brightness changes across it. Detection "
                f"is left as it was.\n\nPress U to try again; SPACE skips a spot "
                f"the camera cannot see.", seconds=10.0, dim=False)
            return
        prior = self._sgrid_prior_gain
        self._sgrid_end("finished", restore_prior_gain=False)
        self._log(f"brightness grid complete: {len(samples)} of {total} spots "
                  f"registered" + (f", {skipped} skipped" if skipped else ""))
        try:
            applied = self._apply_sensitivity_map(
                self.mapping, samples=samples,
                why="this map was measured by the brightness grid (U)")
        except Exception as exc:                          # noqa: BLE001
            # The run is already over and its levels are back; what must not
            # happen is losing the map that was set aside because building
            # the new one raised.
            self._log(f"brightness grid: building the map failed ({exc!r})")
            applied = False
        if not applied and prior is not None:
            self._set_gain_map(prior, "the brightness grid built no new map, so "
                                      "the one it set aside is back")
        self._set_status(
            f"BRIGHTNESS MEASURED  -  {len(samples)} of {total} spots",
            "The sensitivity map was built from your shots, saved with the "
            "calibration, and is being checked against the noise floor.",
            "#17331c", "#7fe08a", "#2e6b38",
        )

    def _sgrid_end(self, why: str, restore_prior_gain: bool = True) -> None:
        """THE ONE EXIT of the brightness grid. Idempotent.

        Every way out comes here, so the thresholds are restored once, in one
        place, whatever ended the run: the nine spots done, U pressed again,
        the idle limit, the camera going, the target screen closing or the app
        closing. `restore_prior_gain` is False only for a run that is about to
        hand over a NEW map (and which puts the old one back itself if it
        turns out there is none).
        """
        if not self._sgrid_active:
            return
        self._sgrid_active = False
        # The consequential step first, before anything below can raise.
        self._restore_levels()
        prior, self._sgrid_prior_gain = self._sgrid_prior_gain, None
        self.scene.calibrating = False
        self.layers["calibration"].stop()
        if restore_prior_gain and prior is not None:
            self._set_gain_map(prior, "the brightness grid ended without a new "
                                      "map, so the one it set aside is back")
        self.scene.mark_dirty()
        self._log(f"brightness grid ended: {why}")

    # Targets limited to the camera's view? OFF by owner's instruction; see
    # `_update_playable` for the reason and the risk.
    RESTRICT_TARGETS_TO_CAMERA_VIEW = False

    def _update_playable(self, mapping: Mapping) -> None:
        """Tell the scene which part of the screen the camera can actually see.

        The camera is off-axis and may not cover the whole monitor -- in testing
        it could not see the bottom strip at all. A target drawn there cannot be
        hit, and reads to the shooter as the app ignoring a good shot, which is
        the worst possible failure because it is indistinguishable from a bug.

        The region is the largest axis-aligned rectangle that is certainly
        inside the camera's view: taking the inner edge of each mapped frame
        boundary rather than the bounding box, because a bounding box of a
        rotated quadrilateral includes corners the camera never saw.
        """
        # A mapping has been measured, so shots can be drawn as real holes at
        # real screen positions. The camera-space minimap -- a box labelled
        # "camera frame (uncalibrated)" over the bottom-right of the screen --
        # exists only for before this moment, and now stands down.
        self.scene.calibrated = True
        w, h = self.settings.width, self.settings.height
        sw, sh = mapping.screen_size
        if not (w and h and sw and sh):
            return
        corners = [mapping.to_screen(0, 0), mapping.to_screen(w, 0),
                   mapping.to_screen(w, h), mapping.to_screen(0, h)]
        (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = corners
        left = max(0.0, tlx, blx)
        right = min(float(sw), trx, brx)
        top = max(0.0, tly, try_)
        bottom = min(float(sh), bly, bry)
        if right - left < sw * 0.3 or bottom - top < sh * 0.3:
            # Implausibly small: trust the whole screen rather than confine the
            # game to a sliver on the strength of one shaky fit.
            self._log("  playable area came out implausibly small; using the whole screen")
            self.scene.playable = None
            return
        rect = QRect(int(left), int(top), int(right - left), int(bottom - top))
        # AND NARROWED TO WHERE A SHOT IS RELIABLY DETECTED, not merely seen.
        #
        # Asked for 2026-10-16: "no targets (or bouncing balls, etc.) go
        # outside of that area". This rectangle is what every mode already
        # places targets inside, so narrowing it here is the whole of that
        # instruction -- no mode needs to know the feature exists. A check
        # added to each mode instead is how collisions survived here for
        # days (RULES E2).
        #
        # The two measurements answer different questions and both matter:
        # the rectangle above is where the camera can SEE, measured from the
        # mapping with no shooting; `play_area` is where a shot is strong
        # enough to be DETECTED, measured from real trigger pulls. The
        # right-hand third of this panel is inside the first and outside the
        # second -- geometry good to 0.75mm, signal at 2.4x instead of 11x.
        # Governed by `playarea.BORDER_ENABLED`: a play_area still sitting in
        # his calibration.json is IGNORED while the switch is off, not
        # deleted, so turning it back on needs no remeasuring.
        area = (getattr(mapping, "play_area", None)
                if playarea.BORDER_ENABLED else None)
        if area is not None:
            rect = rect.intersected(QRect(
                int(area.x0 * sw), int(area.y0 * sh),
                int((area.x1 - area.x0) * sw), int((area.y1 - area.y0) * sh)))
        self.scene.play_area = area
        frac = (rect.width() * rect.height()) / float(sw * sh)
        if self.RESTRICT_TARGETS_TO_CAMERA_VIEW:
            self.scene.playable = rect
            self._log(f"  camera covers {frac * 100:.0f}% of the screen; targets will stay "
                      f"inside {rect.width()}x{rect.height()} at ({rect.x()},{rect.y()})")
        else:
            # SWITCHED OFF, owner, 2026-10-16: "there is no need for the border
            # that restricts the balls, comment that code out for now". Set
            # RESTRICT_TARGETS_TO_CAMERA_VIEW = True to restore it (one line).
            #
            # THE RISK, stated rather than accepted silently: this rectangle
            # exists because a target drawn where the camera cannot see is
            # unhittable, which reads as the app ignoring a good shot. That
            # evening he hit shots across the whole panel including the far
            # right, so it was not earning its keep -- but if a corner truly
            # is invisible to the camera, a ball there will look like a bug.
            # The measurement still runs so the log keeps saying what the
            # camera covers.
            #
            # The play-area border (its own switch) still narrows targets if
            # it is turned back on.
            self.scene.playable = (rect.intersected(QRect(
                int(area.x0 * sw), int(area.y0 * sh),
                int((area.x1 - area.x0) * sw), int((area.y1 - area.y0) * sh)))
                if area is not None else None)
            self._log(f"  camera covers {frac * 100:.0f}% of the screen; targets are "
                      f"NOT restricted to it (RESTRICT_TARGETS_TO_CAMERA_VIEW is off)")

    def _apply_sensitivity_map(self, mapping: Mapping, samples: "list | None" = None,
                               why: "str | None" = None) -> bool:
        """Flatten the detector's sensitivity using what the grid just measured.

        `samples` / `why` exist for the brightness grid (key U), which measures
        ONLY the falloff and must not touch the mapping: it hands in its own
        nine agreeing-group medians instead of `mapping.samples` (which, on a
        mapping loaded from disk, are the calibration markers -- 227-245
        across the whole panel, so a map built from them is all ones). Returns
        True when a map was handed to the judgement, False when detection was
        left as it was, so a caller that set an older map aside knows whether
        to put it back.

        The same gun scored a mean peak of 62 near the centre of this screen and
        34 at the far corner, over a blob 6x smaller, so one threshold was
        simultaneously too strict there and too loose here -- and the far corner
        registered nothing at all. Replayed against a real session, normalising
        took right-half detections from 4 to 12 and upper-right from 0 to 4.

        The noise floor must then be measured AGAIN, because normalising
        amplifies noise in the weak regions along with the signal, and a floor
        measured before this would now be an underestimate. That is 3 more
        seconds and it is not optional.

        THIS PATH HAD THREE FAULTS THE LEARNER (`_learn_sensitivity`) HAD
        ALREADY FIXED, and the two had diverged (2026-10-23):

        1. It was not confined to the screen, so it amplified the room. The
           same map measured 1.11x mid-screen and 1.00x at the window beside
           the monitor confined, 1.44x there unconfined; unconfined, a 1.5x
           correction took the floor 19 -> 112 (2026-10-14 21:33).
        2. It was never saved. The nine agreeing-group medians are the best
           evidence of the falloff this app ever gets -- real laser peaks over
           the whole screen -- and they died with the session.
        3. It was never judged. It armed a floor measurement but never set
           `_gain_pending`, so `_floor_after_gain_map` returned at its first
           line and a map that made the bar dramatically worse stayed on.

        It now goes through the same three steps as the learner, with the
        same helpers: `confined_sensitivity_map`, `_save_sensitivity`,
        `_judge_map_against_a_fresh_floor`.
        """
        if self.capture is None:
            return False
        frame = (self.settings.width, self.settings.height)
        corners = mapping.screen_rect_in_camera()
        if not corners:
            self._log("  sensitivity map: the mapping gives no screen outline "
                      "to confine it to, and an unconfined map amplifies the "
                      "room; detection stays uniform")
            return False
        usable = [s for s in (mapping.samples if samples is None else samples)
                  if s.peak > 0]
        gain = confined_sensitivity_map(usable, frame, corners)
        if gain is None:
            self._log("  sensitivity map: not enough points with a signal reading; "
                      "detection stays uniform")
            return False
        peaks = [s.peak for s in usable]
        self._report_signal_margin(peaks)
        # Written BEFORE the map is applied and judged, as the learner does:
        # the evidence is the measurement, whatever the judgement later says
        # about this session's floor. `_restore_sensitivity` re-judges it at
        # the next launch against that launch's floor.
        self._save_sensitivity(usable)
        self._log(f"  sensitivity across the screen measured: weakest point scored "
                  f"{min(peaks):.0f}, strongest {max(peaks):.0f} "
                  f"({max(peaks) / max(1.0, min(peaks)):.1f}x). Detection normalised.")
        # Applies it, re-measures the floor ON the normalised score, and
        # withdraws it if the bar a shot must clear comes back far worse.
        self._judge_map_against_a_fresh_floor(
            gain, why or "this map was measured by the shot grid")
        return True

    def _report_signal_margin(self, peaks: list[float]) -> None:
        """How much louder than the noise the shots actually were.

        The recurring complaint all through development was "shots are not
        registering", and the app could never say how close to the edge it was
        running -- it had every number needed and reported none of them. A
        margin is the one figure that predicts whether the next session will
        work, and it is cheap: the grid has just measured both halves of it.

        Thresholds come from measurement. A healthy session on this hardware put
        hits at 150-230 against a floor around 10, a margin of 15x or more.
        Below 3x, shots on the far side of the screen start being lost; below
        2x, they are lost everywhere.
        """
        if not peaks:
            return
        floor = max(1.0, self._noise_peak)
        weakest, median = min(peaks), sorted(peaks)[len(peaks) // 2]
        margin = weakest / floor
        self._last_margin = margin
        self._log(f"  signal margin: weakest point scored {weakest:.0f}, median "
                  f"{median:.0f}, against a noise floor of {floor:.0f} "
                  f"-> {margin:.1f}x at the worst point")
        if margin >= 3.0:
            return
        self._log("  WARNING: that is a thin margin. Expect shots to be missed, "
                  "worst on the far side of the screen.")
        # Which advice is right depends on which half of the ratio is wrong,
        # and the app has just measured both. Telling a shooter to darken a
        # room that is already as dark as it goes is worse than saying
        # nothing: it is advice that cannot be followed, and it trains them to
        # ignore the next warning. Said in exactly those words on 2026-10-14:
        # "The room already is as dark as possible. It will not get any darker
        # than this."
        #
        # A thin margin with a LOW floor is a signal problem, not a noise
        # problem, and the room cannot fix it. The levers are exposure time
        # (the dot's brightness scales with it, and a dark room is precisely
        # the condition that allows a longer one) and the dot's size in camera
        # pixels, which is distance and zoom.
        noisy = floor > 25.0
        if noisy:
            why = (f"The noise floor is {floor:.0f}, which is high -- the camera "
                   f"is seeing something bright that is not your laser.\n\n"
                   f"Light falling on the screen, a lamp or window in frame, or "
                   f"a bright window on the desktop behind the target screen. "
                   f"Dimming the room or drawing a blind helps more than any "
                   f"setting.")
        else:
            why = (f"The noise floor is only {floor:.0f}, so the room is NOT the "
                   f"problem and darkening it further will not help. The laser "
                   f"itself is simply faint where it lands weakest.\n\n"
                   f"What does help, in order:\n"
                   f"  - a LONGER exposure. The dot's brightness scales with it, "
                   f"and a dark room is exactly the condition that allows one. "
                   f"Press X to measure the best exposure for the room as it is "
                   f"now.\n"
                   f"  - moving the camera closer, or zooming in, so the screen "
                   f"fills more of the frame. The dot spreads over three times "
                   f"fewer pixels on the far side of this screen than in the "
                   f"middle, and that is what decides whether it registers.")
        self._alert(
            "THE LASER IS BARELY LOUDER THAN THE NOISE",
            f"At the weakest point of the screen your shots scored {weakest:.0f} "
            f"against a noise floor of {floor:.0f} -- only {margin:.1f} times "
            f"louder. A healthy setup manages 15 times.\n\n" + why,
            seconds=12.0)

    def _grid_failed(self, why: str) -> None:
        """Say so loudly, and leave the app honest about being uncalibrated.

        Falling back to the flash outline here would be the worst outcome: it
        still draws holes, just in the wrong place, and the user has no way to
        tell that from a detector that is merely imprecise.
        """
        self._log(f"grid calibration failed: {why}")
        self._set_status(
            "NOT CALIBRATED  -  press G to try again",
            f"Calibration did not complete ({why}). Hits will be drawn roughly "
            f"until the grid is measured.",
            "#322e16", "#c8c070", "#5e5a2a",
        )

    def _grid_finish(self) -> None:
        self._grid_active = False
        self.scene.calibrating = False
        self._restore_levels()
        layer = self.layers["calibration"]
        layer.stop()
        self.scene.mark_dirty()

        if len(self._grid_samples) < 4:
            self._grid_failed(f"only {len(self._grid_samples)} of "
                              f"{len(self._grid_points)} points registered; 4 is the minimum")
            return
        rect = self.target_window.rect() if self.target_window else None
        size = (rect.width(), rect.height()) if rect else (0, 0)
        frame = (self.settings.width, self.settings.height)
        mapping, note = Mapping.fit_validated(self._grid_samples, size, frame)
        if mapping is None:
            self._grid_failed("the recorded points do not form a usable mapping")
            return

        # 42.5" 16:9 panel is about 940mm wide; good enough to report millimetres.
        mm_per_px = 940.0 / size[0] if size[0] else None
        # A mapping whose own error is bigger than a target is not a mapping.
        # Saving one is worse than failing: every hole afterwards lands
        # somewhere arbitrary, and nothing on screen says why. The first live
        # calibration to complete had an RMS of 1287px on a 3840px screen and
        # was saved and announced as a success.
        worst = max(mapping.per_sample_errors(), default=0.0)
        limit = max(60.0, size[0] * 0.02) if size[0] else 60.0
        if worst > limit:
            mm = f" ({worst * mm_per_px:.0f}mm)" if mm_per_px else ""
            self._grid_failed(
                f"the points do not agree: worst error {worst:.0f}px{mm}, "
                f"and anything over {limit:.0f}px means at least one recorded "
                f"point was not a real hit")
            return

        self.mapping = mapping
        self._mapping_measured_now = True   # measured here, not loaded
        self._check_confirmed = False        # measured, not confirmed
        self._log("grid calibration complete: " + mapping.describe(mm_per_px=mm_per_px))
        self._log("  " + note)
        for i, err in enumerate(mapping.per_sample_errors(corrected=bool(mapping.residuals)), 1):
            mm = f" = {err * mm_per_px:.1f}mm" if mm_per_px else ""
            self._log(f"    point {i}: {err:5.1f}px{mm}")
        if layer.skipped:
            self._log(f"  {len(layer.skipped)} point(s) could not be registered - that part "
                      f"of the screen is outside what the camera can see.")
        # The mask now comes from this mapping rather than the flash outline.
        self._push_detector()
        poly = mapping.screen_rect_in_camera()
        if poly:
            self._log("  screen mask rebuilt from the measured mapping: "
                      + ", ".join(f"({x:.0f},{y:.0f})" for x, y in poly))
        self._apply_sensitivity_map(mapping)
        self._update_playable(mapping)
        if self._save_mapping(mapping, "the calibration"):
            self._log("  saved to calibration.json and in use now")
        self._set_status(
            f"CALIBRATED BY SHOOTING  -  {mapping.describe(mm_per_px=mm_per_px)}",
            "This mapping is measured, not inferred from the screen outline.",
            "#17331c", "#7fe08a", "#2e6b38",
        )

    # -- keeping the mapping honest while playing ----------------------------

    def _maybe_align(self) -> None:
        """Ask for one frame to check the mapping against what is on screen.

        One frame, not the eight a calibration still takes: that call reads and
        discards frames to be sure it photographs the current screen, and doing
        that every few seconds would blind the detector for 133ms at a time.
        Alignment does not need a settled frame -- it needs a representative one.
        """
        if not self._auto_align or self.capture is None or self._align_pending:
            return
        if (self._grid_active or self._sgrid_active or self._calibrating
                or self.mapping is None):
            return
        if self.target_window is None:
            return
        self._align_pending = True
        self.capture.request_still("align", frames=1)

    def _scene_snapshot(self, width: int = 480) -> "object | None":
        """What the target screen is showing right now, as a small BGR array.

        Grabbed from the window rather than re-rendered at a smaller size, so it
        is literally the image the camera is looking at -- a re-render would
        differ wherever a layer uses absolute pixel sizes, and those differences
        would be read as misalignment.
        """
        w = self.target_window
        if w is None:
            return None
        try:
            img = (w.grab().toImage()
                   .scaledToWidth(int(width), Qt.SmoothTransformation)
                   .convertToFormat(QImage.Format_RGBA8888))
            h, ww = img.height(), img.width()
            if h <= 0 or ww <= 0:
                return None
            # RGBA8888 has no row padding, so the buffer maps straight onto an
            # array. Copy before the QImage goes out of scope.
            import numpy as _np
            arr = _np.frombuffer(img.constBits(), dtype=_np.uint8, count=h * ww * 4)
            rgba = arr.reshape(h, ww, 4)
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR).copy()
        except Exception as exc:
            self._log(f"auto-align: could not read the target screen ({exc!r})")
            return None

    def _on_align_frame(self, frame: object) -> None:
        self._align_pending = False
        if self.mapping is None or frame is None:
            return
        painted = self._scene_snapshot()
        if painted is None:
            return
        size = self._screen_size()
        if not size:
            return
        self._align_passes += 1
        res = self.aligner.refine(frame, painted, self.mapping.homography, size)
        if res.accepted and res.homography is not None:
            self.mapping.homography = res.homography
            self._align_applied += 1
            self._push_detector()          # the mask follows the mapping
            self._log(f"  {res}")
        elif res.shift_px > self.aligner.max_shift_px:
            # Worth saying out loud: this is the signature of a camera that has
            # been moved, and no amount of refinement will fix that.
            self._log(f"  auto-align: {res.reason}")
            self._moved_findings += 1
            self._set_status(
                "THE CAMERA APPEARS TO HAVE MOVED  -  press G to recalibrate",
                "Holes will be in the wrong place until the grid is measured again.",
                "#322e16", "#c8c070", "#5e5a2a",
            )
            self._alert(
                "THE CAMERA HAS MOVED",
                "The screen is no longer where the calibration says it is, so "
                "bullet holes will land in the wrong place.\n\n"
                "Press G to shoot the grid again.")

    def _alert(self, title: str, detail: str = "", kind: str = "warn",
               seconds: float = 0.0, show_camera: bool = False,
               dim: "bool | None" = None) -> None:
        """Put a message where the shooter is actually looking.

        `dim` defaults to dimming the screen behind the message, except while
        aiming the camera. A note shown WHILE SHOOTING IS POSSIBLE must pass
        dim=False: the veil lands on the targets (2026-10-16: a 59% veil hid
        every bullet hole he fired).

        The operator console is behind a fullscreen target window on the same
        monitor, so anything logged there is invisible to the person holding the
        pistol. A camera pointed at the wrong thing and a detector that is
        simply not working look identical from the firing line; this is what
        tells them apart.
        """
        layer = self.layers.get("alert")
        if layer is None:
            return
        # perf_counter, because that is what `Scene.advance` stamps the
        # render context with, and the layer compares its deadline against
        # `ctx.now`. Three different clocks were in play here --
        # time.time() for the verdict board, time.monotonic() for this,
        # perf_counter() in the scene -- and on Windows none of them share
        # an epoch. A deadline set on one and tested against another is
        # either already past or unreachably far away, so a timed overlay
        # vanishes instantly or never goes at all.
        #
        # It never went. The READY banner sat across the middle of the
        # target screen through an entire live string: "you keep that text
        # on screen and that obscures the action".
        layer.show(title, detail, kind, seconds, now=time.perf_counter(),
                   dim=(not show_camera) if dim is None else bool(dim))
        if show_camera:
            # So they can aim the camera while reading why they need to. Shown
            # LARGE and in GREYSCALE, with the detected screen outlined on it.
            # Greyscale is as safe as the blue multiply for the chroma detector
            # -- equal channels mean zero red excess -- and unlike the blue pane
            # it is actually legible, which is the entire point here.
            pane = self.scene.get("camera")
            if pane is not None:
                pane.visible = True
                pane.set_score_mode(str(self._current_mode()))
                quad = self.screen_quad.corners if self.screen_quad else None
                pane.set_aiming(True, quad,
                                (self.settings.width, self.settings.height))
                self._sync_layer_checkboxes()
        self._aim_help = bool(show_camera)
        if self.capture is not None:
            self.capture.set_aim_help(self._aim_help)
        self.scene.mark_dirty()

    # -- aiming the camera ----------------------------------------------------

    IDEAL_FILL = 0.62          # of the frame's width and height

    def _aim_advice(self, frame_size, live_box=None) -> "tuple[list[str], str]":
        """Which way to move the camera, in words a person can act on.

        "The camera is not seeing the screen well" is a diagnosis, not an
        instruction. Someone standing at the monitor with a camera in their hand
        needs to know which way to move it, and a greyscale blob does not tell
        them that however legible it is.
        """
        fw, fh = frame_size
        if not fw or not fh:
            return ([], "")
        quad = self.screen_quad
        if live_box is not None:
            bx, by, bw, bh = live_box
            xs = [bx, bx + bw]
            ys = [by, by + bh]
            edges = []
            if bx <= 2:
                edges.append("left")
            if by <= 2:
                edges.append("top")
            if bx + bw >= fw - 3:
                edges.append("right")
            if by + bh >= fh - 3:
                edges.append("bottom")
        elif quad is None:
            return (["The monitor was not found in the camera's view at all.",
                     "Point the camera at the screen."], "none")
        else:
            xs = [c[0] for c in quad.corners]
            ys = [c[1] for c in quad.corners]
            edges = list(quad.touches_edges)
        # len(xs), not 4: a live region contributes two edges, a quad four.
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        span_x = (max(xs) - min(xs)) / float(fw)
        span_y = (max(ys) - min(ys)) / float(fh)
        say: list[str] = []

        # Direction first: it is the thing they can act on immediately.
        if cx < fw * 0.38:
            say.append("PAN LEFT - the screen sits to the left of centre")
        elif cx > fw * 0.62:
            say.append("PAN RIGHT - the screen sits to the right of centre")
        if cy < fh * 0.38:
            say.append("TILT UP - the screen sits high in the view")
        elif cy > fh * 0.62:
            say.append("TILT DOWN - the screen sits low in the view")

        fill = max(span_x, span_y)
        if fill < 0.45:
            say.append(f"MOVE CLOSER - the screen fills only {fill * 100:.0f}% "
                       f"of the view, aim for about 60%")
        elif fill > 0.95 or edges:
            where = ", ".join(edges) if edges else "the frame"
            say.append(f"MOVE BACK - the screen runs off {where}")

        if not say:
            say.append("The framing looks good. Press F5 to re-check.")
        return (say, "ok" if len(say) == 1 else "move")

    def _draw_aim_guides(self, pm: QPixmap, frame_size, live_box=None) -> None:
        """Overlay a target zone, the detected screen, and what to do about it."""
        fw, fh = frame_size
        if not fw or not fh:
            return
        s = pm.width() / float(fw)
        g = QPainter(pm)
        g.setRenderHint(QPainter.Antialiasing)

        # Where the screen SHOULD sit. Without something to aim at, "fills most
        # of the view" is a judgement call made against nothing.
        tw, th = pm.width() * self.IDEAL_FILL, pm.height() * self.IDEAL_FILL
        target = QRect(int((pm.width() - tw) / 2), int((pm.height() - th) / 2),
                       int(tw), int(th))
        pen = QPen(QColor("#c8c070"), 2, Qt.DashLine)
        g.setPen(pen)
        g.setBrush(Qt.NoBrush)
        g.drawRect(target)
        g.setFont(QFont("Consolas", 9, QFont.Bold))
        g.drawText(target.x() + 6, target.y() - 6, "PUT THE SCREEN IN HERE")

        # The LIVE bright region, not the flash-detected quad. The quad is
        # accurate but only updates when the screen is flashed, so while
        # someone is physically moving the camera it shows where the screen
        # used to be -- worse than nothing, because it looks authoritative.
        if live_box is not None:
            bx, by, bw, bh = live_box
            g.setPen(QPen(QColor("#6bd36b"), 3))
            g.drawRect(int(bx * s), int(by * s), int(bw * s), int(bh * s))
            g.setPen(QColor("#6bd36b"))
            g.drawText(int(bx * s) + 6, int(by * s) + 16, "SCREEN")
        elif self.screen_quad is not None:
            pts = [QPointF(x * s, y * s) for x, y in self.screen_quad.corners]
            g.setPen(QPen(QColor("#6bd36b"), 3))
            g.drawPolygon(pts)
            g.setPen(QColor("#6bd36b"))
            g.drawText(int(pts[0].x()) + 6, int(pts[0].y()) + 16, "SCREEN")

        lines, _ = self._aim_advice(frame_size, live_box)
        g.setFont(QFont("Consolas", 11, QFont.Bold))
        y = 20
        for line in lines:
            g.setPen(QColor(0, 0, 0))
            g.drawText(11, y + 1, line)          # a shadow, so it reads on any image
            g.setPen(QColor("#c8c070"))
            g.drawText(10, y, line)
            y += 20
        g.end()

    def _clear_alert(self) -> None:
        layer = self.layers.get("alert")
        if layer is not None and layer.visible and not layer.until:
            layer.clear()
            self.scene.mark_dirty()
        self._aim_help = False
        if self.capture is not None:
            self.capture.set_aim_help(False)
        pane = self.scene.get("camera")
        if pane is not None and getattr(pane, "aiming", False):
            # Back to a corner thumbnail; the setup view has done its job.
            pane.set_aiming(False)
            self.scene.mark_dirty()

    def _mm_per_px(self) -> float | None:
        """Screen millimetres per pixel, so errors can be reported in real units.

        940 mm is the active width of the Dell U4320Q. Pixels are meaningless to
        a shooter; "4 mm" is a number they can judge against a target ring.
        """
        if self.mapping is not None and self.mapping.screen_size[0]:
            w = self.mapping.screen_size[0]
        else:
            size = self._screen_size()
            w = size[0] if size else 0
        return 940.0 / w if w else None

    def _map_to_screen(self, cx: float, cy: float):
        """Camera pixel -> screen pixel, or None. There is no second choice.

        This used to fall back to the flash-detected screen outline when no
        measured mapping existed. That fallback is why holes landed about an
        inch from the shot: on this bench the flash outline measures **78mm
        mean error** -- the notes already said so, and already said the app
        "does not quietly fall back to the flash outline", because a mapping
        that is wrong still draws holes, just in the wrong place, which the
        shooter cannot tell from a detector that needs tuning.

        The code did it anyway, and on 2026-10-14 a verified 0.6mm mapping was
        discarded a few seconds after being measured, so every shot of the gun
        test was mapped by the outline instead. Drawing nothing and saying so
        is strictly better: "there is nowhere to draw it, press G" is a
        diagnosis, an inch-wrong hole is a mystery.
        """
        if self.mapping is not None:
            return self.mapping.to_screen(cx, cy)
        return None

    # -- step 1: where is the monitor in the camera image? --------------------

    def _begin_screen_search(self) -> None:
        """Photograph the screen lit and unlit, and work out where it is.

        Everything downstream depends on this. Without it the detector hunts the
        whole camera frame, and on this setup a bright window at the edge
        produced 77% of all detections. It also yields the homography, which is
        what will put a bullet hole where the shot actually landed.
        """
        if self.capture is None:
            return
        self._set_status(
            "FINDING THE SCREEN - do not fire",
            "Flashing the monitor so the camera can locate it.",
            "#3a2f16", "#e8d48a", "#7a6a2a",
        )
        self._calibrating = True
        expo = self._calib_exposures[self._calib_attempt]
        how = "auto exposure" if expo is None else f"exposure {expo:g}"
        self._log(f"auto setup: locating the screen using {how} "
                  f"(attempt {self._calib_attempt + 1}/{len(self._calib_exposures)}); "
                  f"the display will flash...")
        # Fixed exposure, not auto: see controls.calibration_view for why.
        self._apply_preset("calib", exposure=expo)
        # Auto exposure needs time to adapt away from the dark shooting profile.
        settle = 1600 if expo is None else 1000
        self._open_flash("#000000", timeout_s=10.0)
        # A full second: the exposure change has to reach the sensor and the
        # compositor has to actually put black on the glass.
        QTimer.singleShot(settle, lambda: self._request_still("black"))

    def _request_still(self, tag: str, timeout_s: float = 5.0) -> None:
        """Ask for a frame, and decide in advance what happens if none comes.

        The screen search blocks on this while a fullscreen panel covers the
        monitor. Without a deadline, a camera that stops delivering leaves that
        panel up permanently -- which happened, and the only escape was
        unplugging the camera.
        """
        if self.capture is None:
            self._on_still_ready(tag, None)
            return
        self._still_tag = tag
        self._still_guard.start(int(timeout_s * 1000))
        if self._flash is not None:
            # The window's own deadline must outlast the request it is waiting
            # for, or it vanishes mid-measurement.
            self._flash.keep_alive(timeout_s + 2.0)
        self.capture.request_still(tag)

    def _still_timed_out(self) -> None:
        tag, self._still_tag = self._still_tag, ""
        if not tag:
            return
        self._log(f"camera did not deliver the '{tag}' frame within the time "
                  f"allowed - it has stalled. Abandoning this step rather than "
                  f"waiting behind a fullscreen window.")
        self._on_still_ready(tag, None)

    def _route_noise(self, peak: float, p99: float, frames: int) -> None:
        if self.director.deliver_floor(peak, p99, frames):
            return
        self._on_noise_measured(peak, p99, frames)

    def _route_room(self, mean: float, chroma: float, frames: int) -> None:
        if self.director.deliver_room(mean, chroma, frames):
            return
        self._on_room_measured(mean, chroma, frames)

    def _on_still_ready(self, tag: str, frame: object) -> None:
        # A routine running on the Director owns its own stills, including
        # the deadline for them. Nothing below this line applies to it.
        if self.director.deliver_still(tag, frame):
            return
        if tag == self._still_tag:
            self._still_guard.stop()
            self._still_tag = ""
        if tag in ("pat_lit", "pat_dark", "pat_full_lit", "pat_full_dark"):
            self._on_pattern_still(tag, frame)
            return
        # Director tags carry a "#<n>" suffix so a frame can only answer
        # the request it was asked for, so match on the PREFIX here.
        if tag.split("#")[0] in ("probe_lit", "probe_dark",
                                 "check_lit", "check_dark"):
            # These belong to routines on the Director, which took them
            # above if one was waiting. Arriving HERE means the frame is
            # late or orphaned -- the routine was stood down, or its
            # deadline already expired and it moved on. Dropped and said
            # so, never applied to whatever is running now: a measurement
            # that belongs to something else is the C2 fault, and it cost
            # an hour when a harness log line was read as his bench.
            self._log(f"a late '{tag}' frame arrived with no routine waiting "
                      f"for it; discarded")
            return
        if tag == "align":
            self._on_align_frame(frame)
            return
        if frame is None:
            self._log("calibration: camera returned no frame; skipping screen search")
            self._finish_screen_search(None)
            return
        if tag == "black":
            self._calib_black = frame
            if self._flash is not None:
                self._flash.set_colour("#ffffff")
            QTimer.singleShot(1200, lambda: self._request_still("white"))
            return
        if tag == "white":
            import cv2 as _cv2
            black = self._calib_black
            quad = find_screen(frame, black)
            # Always keep the evidence. When calibration fails, these two frames
            # are the only way to tell whether the screen failed to flash, the
            # exposure was wrong, or the camera is pointed somewhere else.
            RECORDINGS.mkdir(parents=True, exist_ok=True)
            _cv2.imwrite(str(RECORDINGS / "calib_white.png"), frame)
            if black is not None:
                _cv2.imwrite(str(RECORDINGS / "calib_black.png"), black)
                diff = _cv2.absdiff(frame, black)
                _cv2.imwrite(str(RECORDINGS / "calib_diff.png"),
                             _cv2.convertScaleAbs(diff, alpha=2.0))
                flash = float(diff.mean())
                self._log(f"  flash difference: mean {flash:.1f} "
                          f"(needs to be well above zero)")
                # Near-zero means the camera is not looking at this screen at
                # all -- not a dim room, not a bad exposure. Flashing the whole
                # monitor black then white is about as loud as a signal gets.
                # Measured on the wrong camera: 0.0 at every exposure. Say so
                # instead of proceeding to ask for 27 calibration shots that
                # cannot possibly register.
                self._blind = flash < 1.0
                self._flash_strength = flash
                self._display_dark = not display_is_lit(frame)
            if quad is not None:
                _cv2.imwrite(str(RECORDINGS / "calib_found.png"), draw_quad(frame, quad))
                self._finish_screen_search(quad)
                return
            self._calib_attempt += 1
            if self._calib_attempt < len(self._calib_exposures):
                self._log("  no screen found at this exposure; trying another")
                self._calib_black = None
                QTimer.singleShot(200, self._begin_screen_search)
                return
            self._finish_screen_search(None)

    def _finish_screen_search(self, quad: object) -> None:
        self._close_flash()
        self._still_guard.stop()
        self._still_tag = ""
        self._calib_black = None

        if quad is None:
            self._log("WARNING: could not find the screen. The detector will search the "
                      "whole camera frame, so expect false hits from anything bright.")
        else:
            self.screen_quad = quad
            corners = ", ".join(f"({x:.0f},{y:.0f})" for x, y in quad.corners)
            self._log(f"screen found: {corners}")
            self._log(f"  covers {quad.coverage * 100:.1f}% of the camera frame, "
                      f"confidence {quad.confidence:.2f}")
            if quad.touches_edges:
                # A corner sitting exactly on the frame boundary is not the
                # screen's corner -- it is where the screen leaves the camera's
                # view. Fitting a homography to it skews the whole mapping, and
                # that is what put bullet holes half an inch out. Better to say
                # so loudly than to hand back confident, wrong coordinates.
                self._log(f"  SCREEN IS CLIPPED on the {', '.join(quad.touches_edges)} "
                          f"edge(s) of the camera frame.")
                self._log("  The clipped corner is NOT the real one, so hole positions "
                          "will be systematically off and shots there cannot be seen at all.")
                self._set_status(
                    "CALIBRATED, BUT THE CAMERA CANNOT SEE THE WHOLE SCREEN",
                    f"Clipped on the {', '.join(quad.touches_edges)}. Move the camera back "
                    f"until the whole monitor fits with a margin, then press Connect again.",
                    "#322e16", "#c8c070", "#5e5a2a",
                )
                self._alert(
                    "THE CAMERA CANNOT SEE THE WHOLE SCREEN",
                    f"The monitor runs off the {', '.join(quad.touches_edges)} edge of the "
                    f"camera's view.\n\n"
                    f"Shots in that part of the screen cannot be detected at all, and the "
                    f"clipped corner skews the aim everywhere else.\n\n"
                    f"Move the camera back until the whole monitor fits with room to spare, "
                    f"then press F5.",
                    show_camera=True)
            self._check_view_quality(quad)
            screen = self.screen() or QApplication.primaryScreen()
            geo = screen.geometry()
            self.homography = quad.homography_to_screen(geo.width(), geo.height())
            self._log(f"  homography ready for {geo.width()}x{geo.height()}")

        self._apply_preset("laser", with_exposure=False)
        self._push_detector()                 # hands the new ROI to the detector
        # Let auto-exposure actually disengage before measuring. Without this the
        # sweep reads nonsense: adjacent settings measured 104, 34, 19, 255.
        saved = self._boot_exposure
        # NOT compared against the combo's label: that reads "2: USB Video
        # Device" while the saved name is "USB Video Device", so the test
        # always failed and the full nine-write sweep ran on every launch --
        # which is the one operation measured to hang this camera.
        if saved is not None:
            # Three probes around the remembered value instead of nine across
            # the range. A full sweep is nine UVC control writes in a few
            # seconds, which is the heaviest thing this app asks of the camera
            # -- and at 14:56 on 2026-10-14 a stall began in the middle of one.
            # No search at all. Measured on this camera: streaming is
            # flawless for five minutes with zero control writes, and a single
            # exposure write can hang it for good. Three writes still did.
            # The value is already known and already applied by the preset, so
            # the cheapest sweep is the one that does not happen.
            self._log(f"auto setup: the remembered exposure is {saved:g}. "
                      f"Checking it against the room before trusting it - "
                      f"this costs no control write, and this camera hangs "
                      f"on those. A sweep follows only if the room has "
                      f"moved outside the bars the rule uses.")
            self.rows["exposure"].set_value_silently(saved)
            # ...but do not simply trust it. Ask what the room is doing at
            # that exposure first -- which costs no control write at all --
            # and sweep only if the answer says the remembered value no
            # longer fits. The room changes with the time of day; a value
            # measured at 21:00 in the dark is the wrong value at 10:00 with
            # daylight on the window beside this monitor. A cache with no
            # invalidation is a constant.
            QTimer.singleShot(900, lambda: self._check_room("setup"))
        else:
            self._log("auto setup: settling, then measuring exposure across the range...")
            QTimer.singleShot(900, lambda: self.capture and
                              self.capture.start_exposure_search())

    # The headroom fraction lives on CaptureThread, beside the bars it is a
    # fraction of. Zero-write, so the room can be asked often; only the sweep
    # it may provoke costs the camera anything.
    ROOM_RECHECK_S = 300.0
    # How long the room keeps ringing after a gunshot sample stops. Added to
    # the sample's own length to size the self-mute; a quarter second is
    # generous for a domestic room and costs nothing, because the trigger
    # click that matters precedes the detection.
    SOUND_TAIL_S = 0.25

    def _check_audio_alive(self) -> None:
        """Reopen the gunshot output if its device went away."""
        if self.mixer is None or not self.mixer.available:
            return
        note = self.mixer.ensure_alive()
        if note:
            self._log(note)

    def _check_room(self, why: str) -> None:
        """Ask what the room is doing now, at the exposure already set.

        Measured against the TARGET BACKGROUND, never the desktop. The noise
        floor learned this the expensive way: measured against editors and
        browsers it came out 90, against the target screen's own background
        it came out 0, and a min_peak set from the first rejects every shot
        on the far side of this panel. The same mistake here would read a lit
        desktop as a bright room and sweep on every single launch.

        At setup this sits in the middle of the chain that ends in a ready
        detector, so EVERY path out of here has to continue that chain. The
        first version returned early on a guard and the whole setup died
        silently -- the log stopped at 'applying preset: Laser hunt' and the
        app sat there -- because `_calibrating` is set by the screen search
        and was still true when this ran. The same shape as the cover leak:
        an early return that skips what the caller depended on.
        """
        if why == "setup":
            # Nothing may abandon setup. A guard here means 'do not trust a
            # measurement taken now', which is a reason to skip the CHECK,
            # never a reason to leave the app half configured.
            if self.capture is None or self._room_pending:
                self._room_proceed("no camera to measure the room with")
                return
            if (self._selftest_active or self._modetest_active
                    or self._pat_active):
                self._room_proceed("the screen is lit on purpose just now")
                return
        else:
            if self.capture is None or self._room_pending:
                return
            if (self._calibrating or self._pat_active or self._grid_active
                    or self._sgrid_active):
                return
            if self._selftest_active or self._modetest_active:
                return
            # An arcade mode puts its own bright targets on screen, so the
            # frame mean would be the mode's and not the room's, and the
            # shooter would be told the room had brightened because a ball
            # happened to be large. Free aim is the dark background this is
            # meant to measure, so wait for it.
            host = self.scene.get("modes")
            if host is not None and getattr(host, "mode", None) is not None:
                return

        self._room_why = why
        self._room_pending = True
        # A deadline, whatever the camera does. This waits on frames arriving,
        # and a camera that has stopped delivering them is exactly the
        # condition under which the app must still finish setting itself up
        # and say what is wrong. 'A request that waits on hardware with no
        # deadline' is already written down in these notes as one of the
        # three faults behind the window that trapped the user.
        self._room_guard.start(6000)
        if self.target_window is None:
            # No target screen, so the camera is looking at the desktop.
            # Stand the flash panel in for it, exactly as the noise floor
            # does. The path this hands off to closes it: the noise
            # measurement always runs after this, on both branches.
            self._cover_desktop_for_noise()
            QTimer.singleShot(500, lambda: self.capture
                              and self.capture.start_room_measurement(1.5))
            return
        self.capture.start_room_measurement(1.5)

    def _room_proceed(self, why: str) -> None:
        """Continue setup with the remembered exposure, having not measured.

        The one job of this function is that setup cannot die here. Called
        from every refusal and from the deadline.
        """
        self._room_guard.stop()
        self._room_pending = False
        self._room_why = ""
        # So the line that follows does not claim a measurement that
        # did not happen. Seen live 2026-10-15: "the room was not
        # measured (no camera frames arrived within 6s)" followed
        # immediately by "the room was measured at the remembered -3
        # and still sits inside the rule's bars". Both from the same
        # path, two seconds apart, and the second one was a lie.
        self._room_unmeasured = True
        value = self._shooting_exposure
        if value is None:
            value = self._boot_exposure
        self._log(f"  the room was not measured ({why}); continuing with the "
                  f"remembered exposure. Press X to measure it against the "
                  f"room as it is now.")
        self._on_exposure_chosen(
            float(value if value is not None else -7.0), [])

    def _room_gave_up(self) -> None:
        """No room measurement arrived in time."""
        if not self._room_pending:
            return
        why, self._room_why = self._room_why, ""
        if why == "setup":
            self._room_proceed("no camera frames arrived within 6s")
        else:
            self._room_pending = False
            self._log("  room measurement timed out; the camera is not "
                      "delivering frames.")

    def _on_room_measured(self, mean: float, chroma: float,
                          frames: int) -> None:
        """Decide whether the remembered exposure still fits this room.

        The rule is the exposure rule, unchanged and applied to the present
        room: a candidate is good if its background is quiet enough for the
        detector (chroma p99 <= CHROMA_CLEAN) and the monitor is still a
        ghost (mean <= MEAN_CEILING). Two ways the remembered value can be
        wrong, and both happen with the clock:

          - over a bar: the room is brighter than when the value was
            measured, so that exposure is now too long. Daylight arriving.
          - far inside both bars: the room is darker, so a LONGER exposure
            is available and the laser is dimmer than it needs to be. This
            is the 2026-10-14 evening failure exactly -- the room was
            darkened as far as it would go, -7 was carried in regardless,
            and -3 was four stops and sixteen times the exposure time away.

        Either way this only decides whether to MEASURE. The sweep chooses
        the value, by measurement, as it always did.
        """
        # Before any branch, as with the cover the noise measurement puts
        # up: a deadline left armed fires into a step that already finished.
        self._room_guard.stop()
        self._room_pending = False
        self._room_unmeasured = False
        why, self._room_why = self._room_why, ""
        if frames < 5:
            self._log("  room measurement saw too few frames; keeping the "
                      "remembered exposure.")
            if why == "setup":
                self._exposure_gate_done = True
                self._room_proceed("the measurement was not usable")
            return
        # A shot inflates chroma p99 far past any room, and the conclusion
        # would be 'the room got bright' -- stopping the exposure DOWN in
        # response to a laser hit, which is the opposite of right. Same
        # lesson as the noise floor, which was spoiled by a trigger pull
        # three times in one evening before it was guarded.
        last_shot = float(self._last_shot_at or 0.0)
        if last_shot and (time.perf_counter() - last_shot) < 2.5:
            self._log("  discarding that room measurement: a shot landed "
                      "during it, so it measured the laser and not the room.")
            if why == "setup":
                self._exposure_gate_done = True
                self._room_proceed("the measurement was not usable")
            return

        chroma_bar = float(CaptureThread.CHROMA_CLEAN)
        mean_bar = float(CaptureThread.MEAN_CEILING)
        current = self._shooting_exposure
        if current is None:
            current = self._boot_exposure
        shown = f"{current:g}" if current is not None else "?"
        room = (f"room at exposure {shown}: mean {mean:.0f} (bar "
                f"{mean_bar:.0f}), chroma noise {chroma:.0f} (bar "
                f"{chroma_bar:.0f})")

        # The verdict is the exposure rule's own, applied to this room. It
        # lives beside that rule and its bars so the two cannot drift apart,
        # and so it can be tested against the sweeps in the logs.
        verdict = CaptureThread.room_verdict(mean, chroma)
        too_bright = verdict == 'bright'
        # The discriminator the advice turns on. A faint dot in a BRIGHT
        # room is cured by dimming the room; a faint dot in a room that is
        # already dark is not, and this app has told a man sitting in the
        # dark to dim the room. Never again from this measurement.
        self._room_too_bright = bool(too_bright)

        if too_bright:
            reason = ("the room is brighter than this exposure allows, so "
                      "the monitor is no longer a ghost and the dot has to "
                      "out-shine a lit panel")
        elif verdict == 'dark':
            # A DARK ROOM IS NOT A REASON TO THROW AWAY A WORKING EXPOSURE.
            #
            # Measured on his machine, 2026-10-20, over five launches: this
            # branch fires EVERY TIME, because the shooting profile makes the
            # screen a ghost and the room mean is 0 against a bar of 40. So
            # the sweep ran on every launch, and the sweep takes the longest
            # exposure whose background still reads empty -- which on this
            # camera is -3, where the image CLIPS: red saturates along with
            # green and blue, the red-excess score collapses, and his shots
            # came back either 11x over the bar or at frame peak ZERO with
            # nothing in between. His proven value, in the working folder and
            # unchanged for days, is -4. He found it by diffing the two
            # configs after an hour of my chasing symptoms:
            #
            #     "Can you please compare the bad config with the good config
            #      and then see why we can't arrive at the good config?"
            #
            # It was circular: remembered -4 -> room reads dark -> sweep ->
            # -3 -> room still reads dark -> sweep -> -3, for ever. Writing
            # -4 into the settings file achieved nothing.
            #
            # These notes already say a dark room is the BEST condition this
            # app can have: it is what PERMITS a long exposure, not a fault to
            # correct. So when a value is already remembered FOR THIS CAMERA,
            # it stands. A room that has gone BRIGHT still sweeps, because
            # that genuinely harms detection, and a camera with nothing
            # remembered still gets its full sweep. X sweeps on demand.
            if current is not None and why == "setup":
                self._log(f"  {room} - a dark room is the condition this app "
                          f"wants, not a fault to correct, so the remembered "
                          f"{shown} stands and the camera is not written to. "
                          f"Press X to sweep it anyway.")
                self._exposure_gate_done = True
                self._on_exposure_chosen(float(current), [])
                return
            reason = ("the room is darker than this exposure needs, so a "
                      "longer one is available and the laser is dimmer than "
                      "it has to be")
        else:
            self._log(f"  {room} - still inside the bars, so the remembered "
                      f"exposure is kept and the camera is not written to.")
            if why == "setup":
                self._exposure_gate_done = True
                self._on_exposure_chosen(float(current if current is not None
                                               else -7.0), [])
            return

        self._log(f"  {room}")
        direction = "bright" if too_bright else "dark"

        if why != "setup":
            # Mid-session: SAY it, do not do it. A sweep puts a cover on the
            # screen and writes to the camera nine times, and doing that in
            # the middle of a round would read as the app breaking. The
            # shooter decides when to spend it -- which is the same reason
            # key X exists, now with the app noticing instead of the shooter
            # having to guess that the room has moved.
            if self._room_warned == direction:
                return                       # said once; do not nag
            self._room_warned = direction
            self._log(f"  the room has changed since the exposure was set: "
                      f"{reason}. Press X when you are not mid-string to "
                      f"measure it against the room as it is now.")
            self._signal_note("the room has changed - press X to re-measure "
                              "the exposure", "warn")
            return

        # At setup nothing is being shot yet, so act rather than ask. This is
        # the one moment in a session when several control writes cost the
        # shooter nothing but a few seconds.
        self._room_warned = direction
        self._exposure_gate_done = True

        # ONE EXPOSURE MEASUREMENT PER LAUNCH, NOT TWO.
        #
        # This swept nine exposures here AND ran the dot probe after
        # calibration -- about fourteen control writes per launch, on the
        # one operation measured to hang this camera (a single write has
        # hung it for 272 seconds; 17,945 frames came back clean over five
        # minutes with no writes at all).
        #
        # It wedged on the fifth gate launch of 2026-10-15, mid-sweep. That
        # launch could not then close cleanly, and every launch after it
        # inherited a dead camera: four identical failures that look like
        # four faults and are one. Adding a better instrument without
        # removing the worse one made the app more fragile, not less.
        #
        # The probe is the better instrument -- it asks what fraction of
        # his 575 logged trigger pulls would clear the bar, where the sweep
        # asks how dark the background is -- and it costs five writes
        # rather than nine. So when the probe is going to run, the room
        # gate defers to it and writes nothing at all here.
        #
        # A room that is too DARK is not urgent: it means signal is being
        # left on the table, and the probe will collect it a minute later.
        # A room too BRIGHT is not urgent either, for the same reason. The
        # sweep stays for the launch where no probe is coming.
        if self.auto_selftest or self.auto_patcal:
            self._log(f"  the room has moved ({reason}), but the exposure "
                      f"probe runs after calibration and measures this "
                      f"properly -- against how many of his shots would "
                      f"register, not against how dark the background is. "
                      f"Deferring to it rather than sweeping twice.")
            if why == "setup":
                current = self._shooting_exposure
                if current is None:
                    current = self._boot_exposure
                self._on_exposure_chosen(float(current if current is not None
                                               else -7.0), [])
            return

        self._log(f"  measuring the exposure against the room as it is now: "
                  f"{reason}.")
        # The sweep re-measures the noise floor when it finishes, so setup
        # continues through the normal path from here.
        self._cover_desktop_for_noise()
        QTimer.singleShot(600, lambda: self.capture
                          and self.capture.start_exposure_search())

    def _on_exposure_chosen(self, value: float, results: object) -> None:
        """The LEGACY entry: apply the exposure, then trigger the floor.

        Kept for the camera's own sweep (`start_exposure_search`), which
        still reports through a signal. The Director's probe calls
        `_apply_chosen_exposure` and then measures the floor as its own next
        line, which is what removed the 1800ms hand-off this path needs.
        """
        self._apply_chosen_exposure(value, results)
        # SKIP THIS FLOOR IF THE PROBE IS GOING TO REPLACE THE EXPOSURE.
        #
        # Measured with his real settings: settle 1.5s + measurement 3.0s +
        # hand-on 1.2s is about six seconds of a forty-one second launch,
        # and it measures the floor at the REMEMBERED exposure -- which the
        # exposure probe then changes, and measures the floor again. The
        # first number is never used to judge a shot: nothing is fired
        # between here and the probe, and the two-marker check photographs
        # rather than detects.
        #
        # Not a gamble. If the probe never runs -- calibration failed, no
        # target screen, the camera stalled -- the fallback below measures
        # it, so the session cannot end up with no floor at all, which
        # would leave the detector on a threshold saved in another room.
        # That is the state that produced fourteen gunshots in one second.
        if self._probe_will_measure_the_floor():
            self._log("skipping the startup noise floor: the exposure probe "
                      "will change the exposure and measure the floor that "
                      "exposure actually gives. Saves about six seconds, and "
                      "the first measurement was never used to judge a shot.")
            self._floor_deferred = True
            # SETUP IS OVER ON THIS PATH TOO. Skipping the floor skipped the
            # only place that said so, and the two-marker check then sat
            # queued for 62 seconds behind a measurement that had already
            # finished. See `_setup_is_over`.
            self._setup_is_over()
            QTimer.singleShot(int(self.settings.settle_seconds * 1000),
                              self._calibrate_if_needed)
            # A BOUND on the deferral, with a reason to stop and a report.
            QTimer.singleShot(60000, self._floor_fallback)
            return
        self._cover_desktop_for_noise()
        self._calibrate_timer.start(int(self.settings.settle_seconds * 1000))

    def _app_is_drawing_now(self) -> bool:
        """Is the light the camera is seeing the APP'S, not a trigger pull?

        Needed before `_on_shot` has computed `_app_is_drawing`, because the
        first thing that method does is stamp "a shot landed" -- and during
        the screen search that stamp came from the app's own flash sixteen
        times in two seconds, which discarded the room measurement that
        followed it.

        The screen search is the case the existing flag misses: it is not a
        Director routine, so it declares nothing, and `_calibrating` is the
        only trace of it. That is the argument for finishing the migration
        rather than for another flag.
        """
        if self.director.is_drawing():
            return True
        if self._flash is not None:          # the fullscreen calibration panel
            return True
        return bool(self._calibrating or self._pat_active
                    or self._modetest_active)

    def _setup_is_over(self) -> None:
        """Setup has finished. Say so once, from every path that finishes it.

        THIS EXISTS BECAUSE I BROKE IT. Measured on his hardware, 08:47-08:48:

            08:47:46  mapcheck: queued behind a measurement
            08:48:48  mapcheck: checking the saved calibration ...

        Sixty-two seconds. The two-marker check -- the thing that exists to
        make startup fast -- sat in the queue because `_screen_busy()` kept
        answering "a measurement", because `_calibrating` was still set,
        because the ONLY place that cleared it was `_on_noise_measured`,
        and an hour earlier I had added a path that skips the noise floor.
        A six-second saving that cost sixty-two.

        `_calibrating` is written in a dozen places and means at least four
        things -- the screen search is running, a floor is in flight,
        pattern calibration is running, tell the shooter not to fire. The
        real fix is for the remaining routines to move onto the Director,
        where "am I running" is not a flag anyone can leave set. Until then
        this is at least ONE function that ends setup, callable from every
        path that ends it, instead of the end of one handler.
        """
        # AND NOTHING THE APP PUT OVER THE SCREEN IS STILL THERE.
        #
        # The FOURTH leaked fullscreen cover on this project, each for a
        # different good reason. This one is mine, from the same
        # optimisation that cost the sixty-two seconds above: the room
        # measurement raises the cover, `_on_noise_measured` lowers it, and
        # "skip the startup noise floor" skipped the lowering too.
        #
        # Measured in the harness, 2026-10-16: every one of eighteen
        # calibration markers was photographed through it, so the lit and
        # unlit frames were the same saturated picture -- difference 8
        # against a threshold of 12 -- and the app reported "not visible to
        # the camera" eighteen times, which reads as a statement about
        # where the camera is pointed.
        #
        # Cleanup belongs at the end of the THING, not at the end of one of
        # its paths. This is that end.
        self._close_flash()
        # The frame-rate warning may fire again from here on.
        self._calibrating = False
        # AND THERE IS SOMETHING TO SHOOT AT. Setup is over, so this is the
        # moment the screen becomes his rather than the app's.
        self._ensure_something_to_shoot_at("setup finished")
        self._stand_down_camera_pane("setup finished")
        # AND HE ARRIVES AT THE MENU. His words, 2026-10-20 21:43: "the app
        # should arrive at the menu but it gets stuck in mode 1."
        #
        # `_menu_when_setup_done` had exactly ONE caller -- `_publish_verdict`
        # -- so the menu was a side effect of publishing a verdict. A verdict
        # comes only from the self test, and the self test runs only under
        # `--selftest`. So a normal launch finished setup and sat in Static
        # Targets for ever, with no menu and nothing saying why. Reproduced
        # twice on his machine tonight, on the path EVERY launch takes once a
        # calibration has been saved.
        #
        # The menu belongs to "setup is over", which is what this function is,
        # not to "a verdict happened". `_menu_when_setup_done` is idempotent
        # and already refuses if he has moved off the startup mode himself, so
        # the verdict path can still call it and the second call does nothing.
        #
        # NOT when `--selftest` is coming: that routine PHOTOGRAPHS the screen,
        # and the menu is the worst screen to measure against -- measured on
        # this bench, 8 of 12 detected with the menu up against 12 of 12 on
        # Static Targets, floor 4 against floor 1. There the verdict shows the
        # menu afterwards, which is the right order.
        if not self.auto_selftest:
            self._menu_when_setup_done()
        self._floor_deferred = self._floor_deferred and not self._floor_measured
        # And the furniture comes back, unless a calibration that hid it
        # for its own reasons is still running. Restoring unconditionally
        # would put the HUD back up in the middle of the grid, over the
        # ring being shot at -- which these notes already record as hiding
        # the one thing the shooter needs to see.
        if not (self._pat_active or self._grid_active or self._sgrid_active):
            if self.scene is not None:
                self.scene.calibrating = False
                self.scene.mark_dirty()
        # EVERYTHING DETECTED DURING SETUP IS GARBAGE. Clear it.
        #
        # CLAUDE.md has said since the first week that "counters are cleared
        # after calibration -- anything detected while the exposure was
        # still settling is garbage and must not appear as the user's first
        # shot". It was never wired up: `_reset_counters` had exactly one
        # caller, a button. A rule that is never executed is not a rule
        # (RULES C7).
        #
        # The cost is in the screenshot from the passing live launch: the
        # signal meter reads "shots 16" over a READY verdict, and he had
        # fired none. All sixteen were the app's own calibration flash.
        try:
            self._reset_counters()
        except Exception:
            pass
        # And give anything that queued behind the measurement its chance
        # immediately, rather than waiting for the next HUD tick.
        try:
            self.director.poke()
        except Exception:
            pass

    def _probe_will_measure_the_floor(self) -> bool:
        """Is the exposure probe certain to run, and measure a floor?

        Both chains that reach it -- `_pattern_finish` and
        `_after_calibration_ready` -- start it only when `auto_selftest` is
        set. The probe measures the floor as its own last step, so when it
        runs, the startup floor is superseded rather than duplicated.

        DELIBERATELY DOES NOT REQUIRE THE TARGET WINDOW. The probe needs
        one to draw its dot on, but this question is asked earlier than the
        window exists: measured in the harness log, the floor runs at
        23:11:16-21 and "target screen open" is logged at 23:11:21. The
        first version of this predicate checked for the window and
        therefore never fired once -- which the harness reported honestly
        as no improvement at all, 41s before and 41s after.

        So the bet is "the probe will run", and `_floor_fallback` is what
        bounds losing it.
        """
        return bool(self.auto_selftest and self.capture is not None)

    def _floor_fallback(self) -> None:
        """Measure the floor if the probe never got round to it.

        The deferral above is a bet that the probe will run. This is what
        makes losing the bet cost a minute rather than the session: a
        session with no measured floor runs the detector on a threshold
        saved in another room, which is where fourteen gunshots in one
        second came from.
        """
        if not self._floor_deferred:
            return
        self._floor_deferred = False
        if self._floor_measured:
            return
        self._log("the exposure probe never measured a noise floor, so "
                  "measuring one now -- the detector must not run on a "
                  "threshold saved in another room.")
        self._start_noise_floor(self.settings.calibrate_seconds)

    def _apply_noise_measurement(self, peak: float, p99: float,
                                 frames: int) -> None:
        """A floor the Director's routine measured, applied the same way.

        An alias rather than a copy, deliberately: the thresholds, the
        spoiled-measurement guard, the gain map and the margin report are
        one body of logic and must not fork. `_on_noise_measured` clears
        `_floor_in_flight` on entry, and the Director's path never sets it,
        so this is safe from either side.
        """
        self._on_noise_measured(peak, p99, frames)

    def _apply_chosen_exposure(self, value: float, results: object) -> None:
        """Record a measured exposure and say what it means. Measures nothing."""
        rows = list(results or [])
        # Rows gained a chroma column; tolerate the older shape so a replayed
        # sweep from a recording still prints.
        table = "  ".join(
            f"{r[0]:g}:{r[1]:.0f}/{r[2]}" + (f"c{r[3]:.0f}" if len(r) > 3 else "")
            for r in rows)
        if rows:
            self._log(f"exposure measured (value:mean/peak) -> {table}")
        else:
            # Do not print an empty measurement table. It reads as "I measured
            # and this is what I found", when in fact nothing was measured --
            # the sweep is skipped when a value is remembered, because control
            # writes hang this camera. Saying so plainly matters: on
            # 2026-10-14 the remembered -7 was carried into a room that had
            # since been darkened as far as it could go, and the laser was
            # left two stops dimmer than the conditions allowed.
            if getattr(self, "_room_unmeasured", False):
                self._log(f"exposure not swept this run, and the room "
                          f"could not be measured either: keeping the "
                          f"remembered {value:g} blind. Press X once the "
                          f"camera is delivering frames again.")
            else:
                self._log(f"exposure not swept this run: the room was "
                          f"measured at the remembered {value:g} and still "
                          f"sits inside the rule's bars, so the camera was "
                          f"not written to. Press X to sweep it anyway.")
        picked = next((r for r in rows if abs(r[0] - value) < 1e-6), None)
        if picked is not None and len(picked) > 3:
            self._log(f"chose exposure {value:g}  (mean {picked[1]:.0f}, "
                      f"frame max {picked[2]}, chroma noise {picked[3]:.0f} - "
                      f"the longest exposure whose background the detector "
                      f"still reads as empty)")
        else:
            self._log(f"chose exposure {value:g}")
        self.rows["exposure"].set_value_silently(value)
        self.settings.exposure = float(value)
        self._shooting_exposure = float(value)
        # A fresh measurement means the room and the exposure agree again,
        # so a later drift in either direction is news and may speak.
        self._room_warned = ""
        self._schedule_save()

        if rows and max(r[2] for r in rows) < 200:
            self._log(
                "WARNING: even the brightest exposure peaked below 200. The scene is "
                "very dark, or the camera is not seeing much. Detection may struggle."
            )

        # Is the camera honouring exposure at all? Sweeping -3 to -11 is a
        # factor of 256 in exposure time; a working camera's mean brightness
        # moves enormously. If it barely moves, the writes are being ignored and
        # auto-exposure is still in charge -- which quietly breaks everything
        # downstream: the screen stops being a ghost, the noise floor climbs,
        # the flash difference collapses, and the shooter is told to move a
        # camera that was pointed correctly all along.
        #
        # Observed after a USB replug: mean 127,127,128,130,130,130,129,136,135
        # across the whole range. Spread of 7%, where a working camera gave 31
        # down to 5.
        means = [r[1] for r in rows]
        if len(means) >= 4 and max(means) > 1:
            spread = (max(means) - min(means)) / max(means)
            # Kept, not just logged: the startup verdict has to be able to
            # say "this is the camera, not the room" without re-measuring,
            # and a camera ignoring exposure must suppress every piece of
            # room advice downstream of it.
            self._exposure_spread = float(spread)
            if spread < 0.25:
                self._log(
                    f"WARNING: exposure writes are being IGNORED. Mean brightness "
                    f"moved only {spread * 100:.0f}% across the whole range "
                    f"({min(means):.0f} to {max(means):.0f}); a working camera "
                    f"changes by 80% or more."
                )
                self._log("  Auto-exposure is still in control. This usually follows a "
                          "USB replug. Unplug and replug the camera, or turn auto "
                          "exposure off in the driver's own property sheet.")
                self._alert(
                    "THE CAMERA IS IGNORING EXPOSURE",
                    f"Brightness barely changed ({spread * 100:.0f}%) across the whole "
                    f"exposure range, so auto-exposure is still in control.\n\n"
                    f"Everything downstream depends on a dark, fixed exposure: without "
                    f"it the screen does not fade to a ghost, the noise floor climbs, "
                    f"and real shots get lost.\n\n"
                    f"Unplug and replug the camera, then press F5. If that does not "
                    f"help, turn auto exposure off in the camera driver's own "
                    f"property sheet.")

        settle = self.settings.settle_seconds
        self._set_status(
            "SETTING UP - do not fire",
            f"Exposure {value:g}. Measuring the noise floor "
            f"({settle:g}s + {self.settings.calibrate_seconds:g}s).",
            "#3a2f16", "#e8d48a", "#7a6a2a",
        )

    def _open_flash(self, colour: str, timeout_s: float) -> None:
        """Show the fullscreen panel, with a guaranteed way out of it.

        The window closes itself on Escape, on a double click, and on its own
        deadline, whatever the caller does or fails to do. Covering somebody's
        entire screen with something they cannot dismiss is not an acceptable
        failure mode for any reason, and it took unplugging the camera to
        escape the last time it happened.
        """
        self._splash_down("the screen is about to be flashed")
        if self._flash is None:
            self._flash = FlashWindow()
            self._flash.cancelled.connect(self._on_flash_cancelled)
        self._flash.show_on(self.screen() or QApplication.primaryScreen(),
                            timeout_s=timeout_s)
        self._flash.set_colour(colour)

    def _close_flash(self, reclaim: bool = True) -> None:
        # deleteLater, not a dropped reference: the window may still be inside
        # one of its own handlers, and destroying it from under itself is a
        # 0xc0000409 fail-fast in Qt6Core rather than an exception.
        flash, self._flash = self._flash, None
        if flash is not None:
            flash.hide()
            flash.deleteLater()
            if reclaim:
                self._reclaim_screen("the calibration cover came down")

    # -- giving the screen back ------------------------------------------------
    #
    # 2026-10-20, a frozen build on his hardware: the app's own fullscreen,
    # always-on-top "Calibrating" cover went over the target screen four times
    # in eleven seconds (the first camera was blind, so the screen search
    # retried at four exposures) and each time it was hidden and nothing put
    # the target screen back in front. Windows minimises a fullscreen window
    # that has lost the foreground, so he watched the app "hide itself behind
    # the other apps". 43 seconds passed, BitTorrent took the screen, and the
    # app noticed only the consequence (a line a second saying detections were
    # being ignored). It came back because HE clicked the taskbar.
    #
    # `force_to_front` existed for exactly this since 2026-10-18 and had two
    # callers, neither of them the one place that provably steals the screen.
    #
    # THE RULE: whatever takes the screen gives it back, through THIS method,
    # and a cover that is added later is covered by construction -- the
    # registry test in test_rules fails the build for any window class that
    # covers the screen and is not listed in `SCREEN_COVERS` and torn down by
    # code that calls this.
    #
    # WHAT IT DOES NOT DO: fight him. It is called when the APP'S OWN cover
    # comes down, and when something covers the screen WHILE THE APP IS
    # MEASURING (nothing measured under a cover means anything). It is not
    # called because a foreign window is in front while he is playing: he may
    # have alt-tabbed deliberately, and an app that keeps grabbing the screen
    # back is worse than the fault. That case is reported (the existing
    # "IGNORING detections" line names what is in front) and left to him.
    # Bounded: at most RECLAIM_MAX_TRIES per episode, RECLAIM_MIN_GAP_S apart.
    SCREEN_COVERS = ("FlashWindow", "Splash")
    RECLAIM_MIN_GAP_S = 1.5
    RECLAIM_MAX_TRIES = 3

    def _reclaim_screen(self, why: str) -> bool:
        """Put the target screen back in front if it is not. True if it is."""
        win = self.target_window
        if win is None or not self._target_ever_opened:
            return True            # nothing to give back
        if win.in_front_of_camera():
            self._reclaim_tries = 0
            self._reclaim_refused_noted = False
            return True
        now = time.perf_counter()
        if (now - self._reclaim_at < self.RECLAIM_MIN_GAP_S
                or self._reclaim_tries >= self.RECLAIM_MAX_TRIES):
            return False
        self._reclaim_at = now
        self._reclaim_tries += 1
        ok = force_to_front(win) and win.in_front_of_camera()
        if ok:
            self._log(f"  the target screen was put back in front ({why}).")
            self._reclaim_tries = 0
            self._reclaim_refused_noted = False
            return True
        # Windows refuses SetForegroundWindow to a process that is not in
        # front, and the symptom is identical to a camera that cannot see the
        # screen -- so a refusal is said, once per episode, with what is there.
        if not self._reclaim_refused_noted:
            self._reclaim_refused_noted = True
            self._log(f"  could NOT put the target screen back in front "
                      f"({why}): {win.cover_reason or 'Windows refused'}. "
                      f"Click it or alt-tab to it.")
        return False

    def _on_flash_cancelled(self, why: str) -> None:
        """The panel removed itself. Abandon whatever it was covering the screen for."""
        # Escape / double click is HIM leaving a cover: do not drag the target
        # screen back over whatever he is going to. A timeout is the app's.
        self._close_flash(reclaim=(why != "escape"))
        self._still_guard.stop()
        self._still_tag = ""
        word = "cancelled" if why == "escape" else "timed out"
        self._log(f"screen calibration {word}. Nothing is covering the display "
                  f"any more. Press G to try again when ready.")
        self._calibrating = False
        self._set_status(
            f"CALIBRATION {word.upper()}  -  press G to retry",
            "The display flash was stopped. Detection still works, but hits "
            "cannot be placed on the screen until the grid is measured.",
            "#322e16", "#c8c070", "#5e5a2a",
        )

    # WHO OWNS THE SCREEN RIGHT NOW. One answer, one place.
    #
    # Each routine used to guard against the ones its author happened to
    # think of, which is not the same as guarding against all of them. The
    # exposure probe checked for calibration; pattern calibration checked
    # for itself and not for the probe. Measured live 2026-10-15:
    #
    #   20:16:33  measuring the exposure by photographing the dot
    #   20:16:34  pattern calibration: 12 markers to fit
    #
    # Both drawing on the same screen, both photographing it, and the self
    # test that followed detected 0 of 12.
    #
    # Five instances of this bug were fixed one at a time, and then this
    # list was added to end it -- and ended up consulted by ONE of the six
    # routines that needed it, while the other five each hand-rolled a
    # different subset of the flags. That is the measured state of it on
    # 2026-10-15, checked rather than assumed:
    #
    #     start_self_test      _selftest_active _pat_active _grid_active
    #                          _calibrating          (not the probe, not
    #                                                 the check, not the
    #                                                 floor)
    #     start_exposure_probe _probe_active _calibrating _pat_active
    #                          _grid_active          (not the self test)
    #     start_mapping_check  five of the eight
    #     start_grid_...       none
    #
    # A central list is still a convention, and the next routine will
    # forget it exactly as five of six did. The Director is the version
    # that cannot be forgotten: there is one run slot, so a second routine
    # has nowhere to run rather than a list it ought to have read.
    #
    # What remains here is only what has not been migrated yet. Entries
    # come OUT as routines move onto the Director; nothing is ever added.
    #
    # ONE EXCEPTION, 2026-10-23, and it is stated rather than slipped in:
    # `_sgrid_active` (the brightness grid, key U) was added by instruction
    # ("it must be in the central exclusion list"), because it is a new
    # flag-driven screen owner and the Director's `world_busy` bridge is the
    # only thing that stops a Director routine starting on top of it. It is
    # the third routine owed the migration (see `start_sensitivity_grid`), and
    # leaves this tuple when it moves.
    SCREEN_OWNERS = (
        ("_pat_active", "pattern calibration"),
        ("_grid_active", "grid calibration"),
        ("_sgrid_active", "the brightness grid"),
        ("_modetest_active", "the arcade mode test"),
        ("_calibrating", "a measurement"),
        ("_floor_in_flight", "a noise-floor measurement"),
    )

    # -- the startup splash: down before ANY measurement ---------------------
    #
    # The splash is a lit logo with a red dot (red excess 229) on the screen
    # the camera watches, so it is a laser to the detector and must never be
    # up while anything is measured. That is held HERE, from the same lists
    # every exclusion check uses, rather than by ordering:
    #
    #   * every flag in SCREEN_OWNERS becomes a property (below) whose setter
    #     takes the splash down the instant it is raised, so a legacy routine
    #     added to that tuple later is covered with no further edit;
    #   * every Director job declaring `measures_screen` passes through
    #     `_screen_is_ours`, which takes it down first;
    #   * `_publish_measuring` (every Director change and the HUD tick) takes
    #     it down whenever anything reports busy;
    #   * `_open_flash`, the one thing that covers the whole screen.
    #
    # A cleanup that works because of what happens to run next is not a
    # cleanup: a black cover once stayed up all session for that reason.

    def attach_splash(self, splash: Splash | None) -> None:
        """Adopt the splash `main()` put up before this window existed."""
        self.splash = splash
        self._splash_up_at = time.perf_counter()
        if splash is None:
            return
        splash.retired.connect(self._on_splash_retired)
        if not self.settings.auto_connect:
            # Nothing is going to open the camera, so nothing will ever
            # take it down for us.
            self._splash_down("auto-connect is off")

    def _on_splash_retired(self, why: str) -> None:
        self._log(f"splash down ({why})")

    # How much longer than the camera takes the splash may stay up, so the
    # welcome line is not cut off mid-word. His instruction, 2026-10-18:
    # "give the announcer an extra second ... it is OK to take 2 more
    # seconds at startup". A CAP, not a wait: the camera opens in ~0.4s and
    # the line runs ~1.25s, so this is normally under a second of hold.
    SPLASH_HOLD_CAP_S = 4.5

    # How long the splash stays up AFTER the welcome line has finished.
    # Asked for on 2026-10-18 having watched it: "lets add another second
    # for the splash screen to stay on". The line ends and the art is still
    # there for a beat, instead of the screen changing on the last syllable.
    # It is dwell, not slack: the cap above moved with it, or the cap would
    # have eaten the second he asked for.
    SPLASH_EXTRA_S = 1.0

    def _splash_down(self, why: str, urgent: bool = True) -> None:
        """Take the splash down if it is up. Cheap and idempotent.

        URGENT BY DEFAULT, and that default is the safe one: the splash is
        structurally forbidden from being on screen while anything measures
        -- it is bright art with red excess 218 at the laser dot over the
        "i", against a limit of 8 -- so every caller that is about to own
        the screen takes it down at once. Only the "setup is finished"
        callers pass urgent=False, and even they are capped.
        """
        sp = self.splash
        if sp is None:
            return
        if not urgent:
            # `self.voice`, NOT `self.announcer`. This read
            # getattr(self, "announcer", None), which returned None on every
            # launch, so the hold below never ran once and the splash came
            # down in the same second the welcome line started -- he lost the
            # last syllable and said so twice.
            #
            # THIRD TIME on this project: `stats` for `_stats`, then
            # `last_shot_at` through getattr, now this. In every case a
            # getattr DEFAULT turned a wrong name into a silently disabled
            # feature that still looked present. Read it directly, so a wrong
            # name raises in the first second instead of costing a day.
            says_until = (self.voice.speaking_until()
                          if self.voice is not None else 0.0)
            now = time.perf_counter()
            left = min(says_until + self.SPLASH_EXTRA_S - now,
                       self._splash_up_at + self.SPLASH_HOLD_CAP_S - now)
            if left > 0.0:
                # ONE timer for the whole remaining wait, and it comes back
                # URGENT so it cannot arm another. A self-rescheduling timer
                # is what `_start_noise_floor` did every 700ms for a day with
                # no limit, and test_rules section G fails the build for it.
                # A measurement that wants the screen meanwhile still calls
                # this urgently and wins outright.
                QTimer.singleShot(int(left * 1000) + 1, self._splash_deadline)
                return
        self.splash = None
        sp.retire(why)
        self._reclaim_screen("the splash came down")
        # Released after the unwind and after its decode thread is done;
        # never destroyed from inside one of its own handlers.
        QTimer.singleShot(0, sp.dispose)

    def _splash_deadline(self) -> None:
        """The welcome line has had its hold. Down it comes, once.

        A SEPARATE METHOD on purpose. `_splash_down` arming a timer that
        calls `_splash_down` is a function that schedules itself, which
        test_rules section G fails whatever the author believes about the
        bound -- and it is right to: `_start_noise_floor` re-armed every
        700ms for a day, with a bound its author also believed in.
        """
        self._splash_down("the welcome line has finished")

    def _splash_problem(self, text: str) -> None:
        """The camera did not open: nothing is measuring, so say why."""
        if self.splash is not None:
            self.splash.show_problem(text)

    def _publish_measuring(self) -> None:
        """Tell the SCENE whether the app is measuring its own screen.

        Layers cannot see the app's flags, so anything that must stand
        down during a measurement has to be told. Set from the same list
        every exclusion check uses, so a routine added later is covered
        without anyone remembering to add it here.
        """
        if self.scene is None:
            return
        # The app's own tick is also where a routine queued behind
        # something the Director does not run gets its chance. Nothing
        # else would wake it, and a queue that never drains is a step that
        # quietly does nothing (RULES C1).
        self.director.poke()

        # ...and where a mode's own one-shot sound is drained. A target
        # APPEARING is not a consequence of a trigger pull, so it cannot
        # ride home on ShotOutcome.sound the way every other sound here
        # does. Nothing in the detection or shot path is involved: this
        # asks the active mode whether it wants a noise and plays it
        # through the same path the gunshot uses (`_play_effect` ->
        # `play_own_sound`), so the microphone that counts his trigger pulls
        # is told about it and does not hear it as one.
        host = self.layers.get("modes") if self.layers else None
        if host is not None:
            # BOTH halves, collapsed (`ModeHost.take_sounds`): draining only
            # `host.mode` meant a cue queued by the right half was never heard.
            for want in host.take_sounds():
                self._play_effect(want)
        # ...and the versus round's trail (the host is pure and cannot log).
        if host is not None:
            for scope, line in host.take_scoped_events():
                self._log(f"  {scope}: {line}")
            # A round just closed: the holes would sit under the verdict text.
            # Round 2 then starts on a clean screen, DELIBERATELY -- "where did
            # my holes go" is otherwise a bug report waiting to happen.
            if host.take_round_ended():
                self._clear_holes("a versus round closed: the verdict is up"
                                  if host.versus else
                                  "a round closed: the magazine is empty and the end screen is up")
            # A round set a personal best: write the settings file, ONCE. The host
            # raises this only when a record was actually written, and a round
            # ends once, so the file is written at most once per round and not at
            # all when nothing improved.
            if host.take_bests_changed():
                self._save_bests()
            # The ROUND box was shot or clicked: the same function N calls.
            if host.take_next_round_request():
                self._restart_mode("the ROUND box")
        busy = bool(self._screen_busy()) or self.director.measuring_screen()
        if busy:
            self._splash_down("a measurement owns the screen")
        if self.scene.diagnostics.get("measuring") != busy:
            self.scene.diagnostics["measuring"] = busy
            self.scene.mark_dirty()

    def _on_director_change(self) -> None:
        """A routine started, finished or was stood down.

        Everything that used to be recomputed from six flags in four places
        is refreshed from one place here: whether furniture stands down
        (B2), and what the top band tells the shooter. One of the three
        faults still open in his screenshot was the band saying "FINDING
        THE SCREEN - DO NOT FIRE" while a verdict was on screen, because
        nothing told it the routine had ended.
        """
        try:
            self._publish_measuring()
            self._update_prompt()
        except Exception:
            # Never let a display refresh take down a routine transition.
            pass

    def _screen_busy(self, *ignoring: str) -> str:
        """The name of whatever owns the screen, or "" if it is free."""
        for flag, name in self.SCREEN_OWNERS:
            if flag in ignoring:
                continue
            if getattr(self, flag, False):
                return name
        return ""

    def _start_noise_floor(self, seconds: float | None = None) -> None:
        """The ONE place a noise measurement may begin.

        Nine call sites used to start one directly, and on 2026-10-15 three
        of them fired within four seconds of each other WHILE the exposure
        probe was drawing red dots on the screen:

            17:29:28  measuring noise floor for 3s
            17:29:28  measuring noise floor for 3s
            17:29:29  measuring the exposure by photographing the dot...
            17:29:31  noise floor over 97 frames: peak 4
            17:29:32  measuring noise floor for 3s

        The floor then measured the DOTS, the probe measured a screen that
        the floor measurement had covered, and the self test that followed
        ran against a bar from neither. Across five cold launches the
        result was bimodal -- 12/12 twice, 0/12 twice -- with the same
        code, the same room and a calibration good to 1mm every time.

        These notes already carry this lesson, from the day two pattern
        calibrations ran at once and every photograph was taken for a
        marker other than the one on screen: AN ASYNCHRONOUS ROUTINE
        DRIVEN BY TIMERS MUST REFUSE TO START WHILE IT IS ALREADY RUNNING.
        The guard is three lines and finding its absence costs sessions.
        """
        if self.capture is None:
            return
        if self.director.is_drawing() or self._modetest_active:
            # The app is deliberately putting red on the screen. Measuring
            # the room now would measure that.
            #
            # BOUNDED, unlike the version this replaces. That one re-armed
            # itself every 700ms with no limit, so a routine that never
            # finished meant a retry loop that never ended -- and it would
            # not even say it was looping (RULES D10).
            if self._floor_retries >= 12:
                self._floor_retries = 0
                self._log("gave up waiting to measure the noise floor: "
                          f"{self.director.busy_label() or 'something'} has "
                          f"had the screen for over eight seconds. The bar "
                          f"stays where it is; press N to measure it.")
                return
            self._floor_retries += 1
            QTimer.singleShot(700, lambda: self._start_noise_floor(seconds))
            return
        self._floor_retries = 0
        if self._floor_in_flight:
            # ...AND NOT A SECOND ONE WHILE THE FIRST IS RUNNING.
            #
            # The first version of this guard closed the door on the probe
            # and the self test and left it open to ITSELF -- which is the
            # commonest shape of this bug and the one the live gate was
            # actually failing on. Caught in seconds by the headless cold
            # start harness, on its first successful run:
            #
            #     noise started while ['noise'] running
            #
            # Dropped rather than queued: a second measurement of the same
            # unchanged screen tells nobody anything, and re-arming it
            # would just move the overlap later.
            return
        self._floor_in_flight = True
        if seconds is None:
            self.capture.start_noise_calibration()
        else:
            self.capture.start_noise_calibration(seconds)

    def _cover_desktop_for_noise(self) -> None:
        """Put the target screen's own background up before measuring the floor.

        The noise floor sets the detection threshold for the whole session, and
        it was being measured against whatever happened to be on the DESKTOP --
        editors, browsers, this app's own console. Measured twice, four minutes
        apart, on the same camera in the same room: peak 15 with a quiet desktop
        and peak 90 with a window full of coloured text on it. The second gives
        a min_peak of 105, which on this screen rejects every shot on the far
        side outright, where a real dot measures about 34.

        The screen that matters is the one that will be shot at, so that is what
        gets measured. The target window may not be open yet, so the flash panel
        stands in for it at the same background colour.
        """
        if self.target_window is not None:
            # Showing the real thing -- but the real thing has FURNITURE on
            # it, and furniture that appeared a moment ago has not been
            # absorbed by the background model yet. Measured 2026-10-15:
            # the startup verdict board, drawn seconds earlier, took the
            # measured floor from 0-2 to 16-18 and min_peak to 22-24 --
            # above the self test's own dots, so the next self test
            # detected 1 of 12 and produced a worse verdict, which stayed
            # on screen and did it again.
            #
            # `calibrating` is the existing signal for "informational
            # furniture, get out of the way", and this is the same request
            # for a stronger reason: during a flash the furniture obscures
            # the measurement, during this one it BECOMES the measurement.
            self.scene.calibrating = True
            self.scene.mark_dirty()
            QApplication.processEvents()
            return
        bg = self.scene.get("background")
        colour = getattr(bg, "colour", "#0a0a0a") if bg is not None else "#0a0a0a"
        # Deliberately short: this one covers the screen for a measurement
        # that takes about four seconds, and nothing downstream depends on it
        # beyond the noise floor being honest.
        self._open_flash(colour, timeout_s=float(self.settings.settle_seconds)
                         + float(self.settings.calibrate_seconds) + 4.0)
        self._log("covering the desktop with the target background so the noise "
                  "floor is measured against what will actually be on screen")

    def _on_noise_measured(self, peak: float, p99: float, frames: int) -> None:
        """Set the threshold from what the camera actually sees, not from a guess."""
        # Cleared on EVERY path out, including the early returns below, or
        # one abandoned measurement locks the door for the session.
        self._floor_in_flight = False
        if frames < 5:
            self._log("noise measurement saw too few frames - is the camera streaming?")
            return
        # Take the cover down FIRST, on every path out of here. The noise
        # measurement puts a black panel over the desktop so the floor is
        # measured against what will be on screen -- and an early return that
        # skips this leaves a fullscreen black window on top of everything.
        #
        # That is how the second self-test run scored 0 on twelve enormous red
        # dots: they were drawn behind the cover, so the tuner kept doubling a
        # dot nothing could see until it reached a 410px radius. This project
        # has now leaked a fullscreen cover three separate times, each for a
        # different good reason. Cleanup belongs before the branch, never
        # inside it.
        if self.target_window is None:
            self._close_flash()
        # A shot during the measurement makes the SHOT the noise floor, and
        # the bar then sits above every real hit for the whole session. The
        # app has warned "do NOT fire during this" since the beginning and
        # never checked, and it has now happened three times in one evening:
        #
        #   21:31:08  measuring noise floor for 3s - do NOT fire during this
        #   21:31:09  SHOT 1 ... peak=89
        #   21:31:10  SHOT 2 ... peak=89
        #   21:31:11  noise floor: peak 89 -> min peak to 104
        #
        # The detector already reported those shots. Nothing had to be guessed
        # -- the two facts simply were never put side by side.
        window = float(self.settings.calibrate_seconds or 3.0) + 1.0
        # Our OWN record of when a shot last arrived. The first version of
        # this guard read `self.capture.stats.last_shot_at`, which does not
        # exist -- the capture object keeps it as `_stats`, and it is replaced
        # wholesale on reconnect. getattr quietly returned 0.0, so the guard
        # never fired once, and the floor it was written to catch sailed
        # straight through at 21:33 on 2026-10-14. A guard that reads a name
        # nobody checked is worse than none, because it looks like protection.
        last_shot = float(self._last_shot_at or 0.0)
        if last_shot and (time.perf_counter() - last_shot) < window:
            # Five, not two. Two is fewer than a shooter mid-string, and
            # giving up means accepting a floor set by a trigger pull -- the
            # exact outcome this exists to prevent. Measured 21:37: two
            # retries were both spoiled and the third was accepted at peak 58
            # against a true 19, putting min_peak at 73.
            if self._noise_retries < 5:
                self._noise_retries += 1
                self._log(f"  a shot landed during that noise measurement, so "
                          f"it measured the SHOT (peak {peak:.0f}) and not the "
                          f"room. Discarding it and measuring again in 2s -- "
                          f"please hold fire.")
                # And say it where the shooter is looking, not only in a log
                # behind a fullscreen window.
                self._signal_note("HOLD FIRE - measuring the noise floor",
                                  "warn")
                self._calibrating = True
                QTimer.singleShot(2000, lambda: self.capture
                                  and self._start_noise_floor(self.settings.calibrate_seconds))
                return
            self._log(f"  shots keep arriving during the noise measurement. "
                      f"Using peak {peak:.0f}, which may be too high; stop "
                      f"firing and press F to measure it cleanly.")
            self._alert(
                "THE NOISE FLOOR WAS MEASURED WHILE YOU WERE SHOOTING",
                f"Five measurements in a row caught a shot, so the rejection "
                f"bar is set from your laser (peak {peak:.0f}) instead of the "
                f"room." + chr(10) * 2
                + "Shots will be missed until this is redone." + chr(10) * 2
                + "Stop firing for five seconds, then press F.",
                seconds=12.0)
        self._noise_retries = 0
        if self._selftest_active or self._modetest_active or self._pat_active:
            # Something is deliberately putting light on the screen, so this
            # is not a measurement of the noise -- it is a measurement of the
            # thing we are drawing. Accepting it sets the rejection bar above
            # a real hit for the rest of the session.
            #
            # Seen on the first self-test run: the floor came back 78 against
            # a true 11, min_peak went to 93, and every red dot after it
            # (peak ~73) was reported NOT DETECTED. Identical to what happens
            # when a shot is fired during the three-second measurement, which
            # the app warns about and never guarded.
            self._log(f"  discarding that noise measurement (peak {peak:.0f}): "
                      f"the screen was lit on purpose at the time, so it "
                      f"measured our own pattern, not the room.")
            return
        # Margins kept tight. A dot on the far side of the screen is dimmer,
        # more oblique and spread over fewer camera pixels, so it is the first
        # thing a generous margin throws away -- which is exactly the "shots on
        # the right do not register" symptom.
        self._noise_peak = float(peak)
        # A floor of zero is a legitimate measurement on a dark screen, so
        # "has it been measured" cannot be inferred from the value.
        self._floor_measured = True
        # Relative to the noise that was actually measured, not a fixed step
        # above it. `peak` is the LOUDEST noise seen in three seconds, so a bar
        # a little above it already means "louder than anything the room did".
        #
        # The old rule was peak + 10 and peak + 15. On a quiet screen that is
        # punitive: measured 2026-10-14 21:40, floor peak 22 put min_peak at
        # 37 while real shots on the far side of the panel scored 42, 44 and
        # 50 and the ones that vanished scored below 37. A tenth of the blob
        # area over there (32-124 px against 542-1780 in the middle) is the
        # physics; the threshold does not have to add to it.
        #
        # Multiplicative keeps the same meaning at every floor: 30% above the
        # loudest measured noise, with a small absolute minimum so a near-zero
        # floor does not put the bar on top of the noise itself. At floor 22
        # this gives 29 instead of 37; at floor 90 it gives 117, still high,
        # which is correct because something is wrong at floor 90.
        # AND NEVER BELOW THE LASER/NOISE BOUNDARY, which is a property of
        # this gun against this panel and NOT of the room.
        #
        # 2026-10-16, live, with him standing at the firing line: the floor
        # measured 2, so the bar came out 8 -- and a static object on the
        # panel at camera (558,525) scored 8 to 16 and fired the gunshot
        # sound at him five times while he was not shooting. Identical
        # coordinates seconds apart; his notes record the same spot doing
        # it on a previous day. His real shots in that same session
        # measured 74 to 127.
        #
        # The multiplicative rule is right and it has nothing to hold on to
        # when the floor is near zero: 30% above almost nothing is almost
        # nothing. The number that does hold is measured twice over --
        # across two sessions at two exposures the laser/noise boundary sat
        # at 30, and over 575 of his logged trigger pulls the DIMMEST shot
        # he has ever fired measured 31, p10 41, median 85. So a bar of 30
        # costs him nothing he has ever shot and silences a whole class of
        # phantom.
        #
        # Deliberately the same constant the single-frame rule already
        # uses, because it is the same fact about the same laser.
        threshold = int(min(250, max(peak * 1.15, peak + 3)))
        min_peak = int(min(255, expo.min_peak_for(peak)))
        self.sp_threshold.setValue(threshold)
        self.sp_minpeak.setValue(min_peak)
        self._log(
            f"noise floor over {frames} frames: peak {peak:.0f}, 99th pct {p99:.0f}. "
            f"Threshold set to {threshold}, min peak to {min_peak}."
        )
        # The measured laser/noise boundary, raised only if the room is
        # noisy enough for min_peak to approach it.
        single = max(self.SINGLE_FRAME_PEAK, min_peak + 6)
        self._log(f"  a single-frame pulse counts as a shot at peak "
                  f"{single} or above. Measured on a recorded session, "
                  f"that recovers the 24% of trigger pulls the two-frame "
                  f"rule threw away, and admits no noise.")
        self._schedule_save()
        # If this measurement was the BASELINE for a map waiting to be
        # judged, put the map on and measure again. Both readings are then
        # taken in the same conditions, which is the whole point.
        if self._gain_awaiting_baseline is not None:
            self._apply_pending_map_and_remeasure(float(peak))
            return
        # If this measurement was ordered to judge a freshly applied gain map,
        # judge it. A map that makes the floor much worse rejects more real
        # shots than its normalisation recovers.
        self._floor_after_gain_map(float(peak))

        if peak > 120:
            self._set_status(
                f"CALIBRATED but the room is bright  (noise floor {peak:.0f})",
                "The camera still sees something bright and red. Detection may be "
                "unreliable; try a darker exposure or dim the room.",
                "#322e16", "#c8c070", "#5e5a2a",
            )
            self._log(
                "WARNING: high noise floor. The camera is seeing something bright and red. "
                "Lower the exposure value further, or dim the room, then recalibrate."
            )
        else:
            # SAY WHAT WILL HAPPEN, NOT WHAT THE SETTING SAYS.
            #
            # This used to announce "starting grid calibration" whenever
            # calibrate_each_session was set -- which is its default -- and
            # `_calibrate_if_needed` then confirmed the saved mapping with
            # two markers and skipped the grid a second later. Measured on
            # his bench 2026-10-18: announced four times, skipped four
            # times, while the mapping was confirmed at 1.2px. He read that
            # as "it could not calibrate", and he was right to.
            if self.mapping is None:
                self._set_status(
                    f"CALIBRATION NEEDED  -  threshold {threshold}, noise floor {peak:.0f}",
                    "Shoot the nine rings as they appear. This takes about a minute, once.",
                    "#17331c", "#7fe08a", "#2e6b38",
                )
                self._log("READY - starting grid calibration.")
            elif (self.settings.calibrate_each_session
                    and not self._mapping_measured_now):
                self._set_status(
                    f"READY  -  threshold {threshold}, noise floor {peak:.0f}",
                    "Checking the saved calibration. Do not fire yet.",
                    "#17331c", "#7fe08a", "#2e6b38",
                )
                self._log("READY - checking the saved calibration with two "
                          "markers; the grid runs only if that check fails.")
            else:
                self._set_status(
                    f"READY  -  threshold {threshold}, noise floor {peak:.0f}",
                    "Fire at the screen. The crosshair should land on your dot.",
                    "#17331c", "#7fe08a", "#2e6b38",
                )
                self._log("READY - using saved calibration: "
                          + self.mapping.describe(mm_per_px=self._mm_per_px()))
            # Anything detected while the exposure was still settling is garbage
            # by construction, and must not greet the user as their first shot.
            #
            # ONCE PER LAUNCH. This handler runs on every accepted noise
            # floor, and the floor is re-measured whenever the exposure
            # moves -- five times on his bench 2026-10-18. Clearing here on
            # every pass wiped the holes he had just shot and reset his
            # score mid-string, which is the opposite of the garbage this
            # is meant to discard.
            first_setup = not self._setup_completed_once
            self._setup_completed_once = True
            if first_setup:
                if self.capture is not None:
                    self.capture.reset_stats()
                self.session.reset()
                self.layers["hits"].clear()
                self._clear_holes("first setup completed")
                self.sp_fired.setValue(0)
            if self.auto_record and not self.capture._stats.recording:
                self.capture.start_recording(RECORDINGS)
            if self.auto_target and self.target_window is None:
                # Normally already open from _autostart; this is the fallback
                # for a connection made by hand.
                QTimer.singleShot(150, self._toggle_target_mode)
            # Without a measured mapping the only thing left is the flash
            # outline, and on this setup that is demonstrably wrong: a specular
            # reflection on the right of the panel makes the lit and unlit
            # frames nearly identical there, so the detected corner lands well
            # inside the screen and every shot maps askew. Rather than start in
            # a state that is known to be inaccurate and hope the user presses
            # a key, ask for the nine shots up front.
            if first_setup:
                QTimer.singleShot(1200, self._calibrate_if_needed)
        self._setup_is_over()

        # AND NOW, if something was waiting for this floor, start it.
        #
        # Reached only on a floor that was actually accepted -- the spoiled
        # and re-measured paths return above, leaving the pending callable
        # set, so it is consumed by whichever measurement finally succeeds.
        # That is the whole point: no delay to guess, and no routine asked
        # for while the thing it depends on is still running.
        self._floor_deferred = False
        then, self._probe_after_floor = self._probe_after_floor, None
        if then is not None:
            self.start_exposure_probe(then=then)

    def _on_recording_changed(self, active: bool, path: str) -> None:
        self.btn_record.setText("Stop recording" if active else "Start recording")
        self.lbl_record.setText(path or str(RECORDINGS))

    # -- parameter plumbing ---------------------------------------------------

    def _current_mode(self) -> ScoreMode:
        """Coerce the combo back to a ScoreMode.

        ScoreMode subclasses str, and PySide6 unwraps a str subclass when it
        round-trips through QVariant, so currentData() hands back a bare str.
        Rebuilding from the text is the only reliable route.
        """
        try:
            return ScoreMode(self.cmb_mode.currentText())
        except ValueError:
            return ScoreMode.CHROMA

    def _roi_polygon(self):
        """Where on the camera image a hit can legitimately land.

        Order of preference: the outline implied by a shot-measured mapping,
        then the flash-detected screen outline, then nothing. And NOTHING while
        grid calibration is running -- the mask is derived from a calibration we
        are in the middle of replacing, and if it is wrong it would reject the
        very shots meant to fix it.
        """
        if self._grid_active:
            return None
        # `_sgrid_active` (the brightness grid) is DELIBERATELY not tested
        # here. The shot grid drops the mask because the mapping it replaces
        # cannot be trusted to reject anything; the brightness grid consumes
        # a mapping that IS trusted, and keeps the mask on so that the
        # blown-out window beside the monitor cannot supply spots.
        if self.mapping is not None:
            # Deliberately generous. The mask exists to exclude the ROOM, not to
            # define the screen edge precisely, and the two errors are wildly
            # asymmetric: a mask a little too large admits some dark wall, while
            # a mask a little too small silently discards real hits and looks
            # exactly like a detector that does not work.
            #
            # It is too small by construction near the corners: the calibration
            # grid is inset 6% from the edges, so every corner of this rectangle
            # is an extrapolation, and a projective fit extrapolates short. That
            # is why shots in the far upper-right corner were being dropped.
            w = self.mapping.screen_size[0] or 3840
            poly = self.mapping.screen_rect_in_camera(margin_px=w * 0.04)
            if poly is not None:
                return poly
        return self.screen_quad.polygon if self.screen_quad else None

    def _detector_params(self) -> DetectorParams:
        return DetectorParams(
            mode=self._current_mode(),
            threshold=self.sp_threshold.value(),
            min_peak=self.sp_minpeak.value(),
            blur=self.sp_blur.value(),
            min_area=self.sp_minarea.value(),
            max_area=self.sp_maxarea.value(),
            close_iter=self.sp_close.value(),
            use_background=self.chk_bg.isChecked(),
            bg_alpha=self.sp_alpha.value(),
            scale=self.cmb_scale.currentData(),
            max_blobs=self.sp_maxblobs.value(),
            min_separation=float(self.sp_minsep.value()),
            max_elongation=float(self.sp_elong.value()),
            roi_polygon=self._roi_polygon(),
            roi_margin=float(self.sp_roimargin.value()),
        )

    # Where a one-frame pulse has to sit to count as a shot. Two recorded
    # sessions at different exposures and noise floors put the boundary
    # at 30 in both -- see `ShotTracker` for the measurements and for the
    # two anchors that were tried and failed. The + 6 lets it rise if the
    # room is noisy enough for min_peak to approach it.
    # THE BAR FOR TRUSTING A ONE-FRAME DETECTION, and it is NOT the same
    # number as the detector's general minimum, which is 8.
    #
    # I merged them on 2026-10-16 and it cost him half a string. Two
    # recorded sessions at different exposures and different noise floors
    # both put the single-frame laser/noise boundary at 30: that is a
    # property of this gun against this panel. The general minimum exists
    # for a different reason -- to stop a near-zero floor putting the bar
    # on top of the noise itself -- and belongs where the floor rule is.
    SINGLE_FRAME_PEAK = 30

    def _tracker_config(self) -> GrouperConfig:
        return GrouperConfig(
            min_frames=self.sp_minframes.value(),
            single_frame_peak=max(self.SINGLE_FRAME_PEAK,
                                  self.sp_minpeak.value() + 6),
            max_gap_frames=self.sp_gap.value(),
            max_move_px=float(self.sp_move.value()),
            max_tracks=self.sp_tracks.value(),
            max_duration_ms=float(self.sp_maxdur.value()),
        )

    def _push_detector(self) -> None:
        self._schedule_save()
        self.lbl_mode.setText(self._current_mode().description)
        if self.capture is not None:
            self.capture.set_detector_params(self._detector_params())

    def _push_tracker(self) -> None:
        self._schedule_save()
        if self.capture is not None:
            self.capture.set_grouper_config(self._tracker_config())

    def _on_control_changed(self, key: str, value: float) -> None:
        if self.capture is not None:
            self.capture.set_control(key, value)
        else:
            self._log("CANNOT SET CONTROLS: no camera connected.")

    def _apply_preset(self, which: str, with_exposure: bool = True,
                      exposure: float | None = None) -> None:
        if self.capture is None:
            self._log("not connected")
            return
        backend = self.cmb_backend.currentText()
        # with_exposure=False leaves exposure alone so the measurement step can
        # choose it, rather than the preset stamping a guess over the result.
        expo = float(self.rows["exposure"].spin.value()) if with_exposure else None
        if which == "laser":
            preset = controls.laser_hunt(backend, exposure=expo)
        else:
            # The gain the CAMERA reported at open, so the calibration
            # preset UNDOES the shooting profile's gain pin rather than
            # inheriting it. Anything that photographs the monitor after
            # shooting has started was looking at a sensor turned down for
            # a laser dot: measured in simulation, marker difference 201
            # with gain free and 3 with it pinned, against a detection
            # threshold of 12.
            # No `hasattr` guard: the stand-in mirrors this method, because
            # a stand-in that emits fewer signals than the real thing
            # excuses every consumer of the missing ones -- and taking the
            # None path here is precisely the behaviour that hid the bug
            # (RULES C12).
            preset = controls.calibration_view(
                backend, exposure=exposure,
                gain=self.capture.gain_at_open())
        self.capture.apply_preset(preset)
        for k, v in preset.values.items():
            if k in self.rows:
                self.rows[k].set_value_silently(v)
        self._log(preset.description)

    # -- target screen --------------------------------------------------------

    def _set_layer(self, name: str, on: bool) -> None:
        self._schedule_save()
        layer = self.scene.get(name)
        if layer is not None:
            layer.visible = on
            self.scene.mark_dirty()

    def _toggle_target_mode(self) -> None:
        if self.target_window is not None:
            self.target_window.close()
            return
        w = TargetWindow(self.scene, target_fps=60)
        self._target_ever_opened = True
        w.closed.connect(self._on_target_closed)
        # Without this the target screen's keys went nowhere: G, SPACE and
        # BACKSPACE were documented on screen and silently did nothing,
        # because the signal was emitted and never connected.
        w.key_pressed.connect(self._on_target_key)
        w.quit_requested.connect(self._quit_from_target)
        w.show_on(self.screen() or QApplication.primaryScreen())
        self.target_window = w
        self.btn_target.setText("Close target screen  (F11)")
        self._log("target screen open. Esc returns here. Try W (white) and R (red) backgrounds.")
        # ...BUT NOT WHEN THE SELF TEST IS STILL TO COME. Same gate as in
        # `_setup_is_over`, and I missed it here on the first pass: with
        # `--selftest` the window opened, this put the MENU up, and the twelve
        # red dots were then photographed against eleven lit tiles. He caught
        # it from the firing line -- "You used to check the startup screen in
        # mode 1" -- and his 10/18 instruction says exactly that. Measured:
        # menu 8 of 12 at floor 4, Static Targets 12 of 12 at floor 1. The
        # verdict puts the menu up afterwards, which is the right order.
        if self.auto_selftest:
            return
        # IF SETUP ALREADY FINISHED, HE ARRIVES AT THE MENU NOW.
        #
        # On the fast path the window is opened LATE and lazily -- by the
        # two-marker check, a second after setup reports itself over ("the
        # two-marker check needs the target screen: opening it"). So the menu
        # was being selected into a scene that did not exist yet, and opening
        # the window installed the startup mode over the top of it. Measured
        # on his machine at 21:48:30: the log says "setup is done: showing the
        # menu", the window opens at 21:48:31, and he is left in Static
        # Targets. The call latched, so nothing tried again. He said: "You
        # didn't fix anything."
        self._menu_when_setup_done()

    def _menu_when_setup_done(self, force: bool = False) -> None:
        """Put the MENU up once the startup checks have answered.

        His instruction, 2026-10-18, after watching a launch: *"previously the
        app would start with mode 1 for calibration and in my mind mode 1
        works better for calibration. Let's just start calibration with mode 1
        and display the menu when calibration is complete."*

        He is right about mode 1 and the reason is measurable: the static
        targets are a handful of thin rings on black, while the menu lights
        eleven pictures across the screen -- and the screen search, the
        exposure probe and the self test all photograph that screen. A dot
        landing on lit grey loses its red excess (B10), so a bright startup
        screen makes the app's own measurements worse at exactly the moment
        they decide whether he can shoot.

        ONCE, and only over the mode the app itself started in. If he has
        chosen a drill while the checks ran, that is his choice and it stands;
        and a verdict re-run later (A, or a camera move) must not yank him out
        of whatever he is shooting.
        """
        if self._menu_shown_at_startup:
            return
        # NO WINDOW, NO MENU, AND DO NOT LATCH.
        #
        # Selecting a mode into a scene that does not exist yet does nothing
        # visible, and the window -- when it is finally opened -- installs the
        # startup mode over it. Latching on the way past made that permanent:
        # setup said "showing the menu", the window opened a second later, and
        # he sat in Static Targets with nothing willing to try again.
        #
        # Returning WITHOUT setting the flag is the whole fix: the window's own
        # open path calls this again, and by then there is something to put the
        # menu into.
        if self.target_window is None or self.scene is None:
            return
        # AND NOT WHILE ANYTHING IS STILL MEASURING. His instruction,
        # 2026-10-20 22:50: "Do not show the menu before the calibration is
        # completed."
        #
        # Every startup routine PHOTOGRAPHS this screen -- the screen search,
        # the marker check, the exposure probe, the self test -- and the menu
        # lights eleven pictures across it. Measured on this bench: with the
        # menu up, 8 of 12 dots detected at floor 4; on Static Targets, 12 of
        # 12 at floor 1. So the menu waits for the measuring to finish, and
        # mode 1 stays up for it, which is what he asked for on 2026-10-18 and
        # had to ask for again tonight after I broke it.
        # `force` is the VERDICT calling: the checks have answered, so this is
        # the moment he should arrive at the menu -- "This is how the app
        # should present itself to the user when it's ready for action"
        # (2026-10-20 23:12). Without it the guard below refuses, because the
        # verdict is published from INSIDE the self test while that routine
        # still holds the run slot. Measured minutes after I added the guard:
        # READY published at 23:12:30 and the app sat in Static Targets.
        if not force and (self._pat_active or self._grid_active
                          or self._sgrid_active
                          or self._calibrating or self.director.busy_with()):
            return
        host = self.layers.get("modes") if self.layers else None
        mode = getattr(host, "mode", None) if host is not None else None
        if mode is None or mode.name != STARTUP_MODE_NAME:
            self._menu_shown_at_startup = True      # he moved; leave him there
            return
        self._menu_shown_at_startup = True
        self._log("setup is done: showing the menu.")
        self._select_mode(MenuMode())

    def _quit_from_target(self) -> None:
        """F11 on the target screen ends the whole session.

        His instruction, 2026-10-18: *"when I press F11 I need the app to
        close, that includes the other frame as well"*. Before this, F11 shut
        the target screen and left this console running with the camera open
        -- so from the firing line the app was closed while it was still
        holding the device, and the next launch was a second process on one
        camera, which is what wedges it (A7).

        Closing THIS window, not quitting the application: `closeEvent` here
        stands every routine down, stops the microphone and releases the
        camera the way the app would, which is the whole difference between
        a 0.44s open next time and a 40.3s one (A1).
        """
        self._log("F11 on the target screen: closing the app.")
        self.close()

    def _on_target_key(self, key: int) -> None:
        """Keys pressed on the fullscreen target screen."""
        if self._grid_active:
            if key == Qt.Key_Space:
                self._grid_skip()
            elif key == Qt.Key_Backspace:
                self._grid_back()
            return          # nothing else may run mid-calibration
        if self._sgrid_active:
            # The brightness grid has the same three keys, plus U again to
            # stop it: Escape closes the whole app now, so it is no longer a
            # way out of a calibration, and a run needs one that is.
            if key == Qt.Key_Space:
                self._sgrid_skip()
            elif key == Qt.Key_Backspace:
                self._sgrid_back()
            elif key == Qt.Key_U:
                self._sgrid_end("stopped from the keyboard (U)")
                self._log("the brightness grid was stopped; detection is as it was.")
            return          # nothing else may run mid-calibration
        if key in self.VOLUME_UP_KEYS:
            self._step_volume(+1)
            return
        if key in self.VOLUME_DOWN_KEYS:
            self._step_volume(-1)
            return
        if key == Qt.Key_C:
            # SELF-CALIBRATE is on C (2026-10-21). It was K. K is reserved for
            # the Keyboard Shortcuts mode and is deliberately bound to NOTHING
            # here.
            self.start_pattern_calibration()
            return
        if key == Qt.Key_V:
            self._toggle_aim_view()
            return
        if key == Qt.Key_X:
            self._remeasure_exposure()
            return
        if key == Qt.Key_A:
            # THE SELF CHECK IS ON A (2026-10-21). It was C (and T before
            # that); every printed instruction that named C for this names A.
            self.start_self_test()
            return
        if key == Qt.Key_Q:
            # Recall the last verdict. Not a toggle: a problem the app has
            # measured should be one key away from the firing line, and
            # re-running the whole self check to re-read its answer costs
            # twenty seconds of flashing dots.
            board = self.scene.get("verdict")
            report = getattr(self, "_last_report", None)
            if board is not None and board.visible:
                board.clear()
            elif board is not None and report is not None:
                board.show_report(report, now=time.perf_counter())
            else:
                self._log("no self check has run yet: press A")
            self.scene.mark_dirty()
            return
        if key == Qt.Key_F:
            # F6 retired 2026-10-21; the noise floor is on F.
            self._remeasure_noise()
            return
        if key == Qt.Key_G:
            self.start_grid_calibration()
            return
        if key == Qt.Key_U:
            # U measures the BRIGHTNESS by shooting; G measures the MAPPING by
            # shooting. Siblings, so they sit together in the legend too.
            self.start_sensitivity_grid()
            return
        if key == Qt.Key_N:
            self._restart_mode()
            self.voice.say("v_new_round")
            return
        if key == Qt.Key_0:
            # ZERO TOGGLES THE INFORMATION, as one group. Owner, 2026-10-16:
            # "zero toggles the info and the 1/2/3/etc. keys start and
            # restart the shooting modes."
            self._toggle_info()
            return
        cls = arcade.by_key(key)
        if cls is not None:
            self._select_mode(cls())
            self._announce_mode()

    # -- is the saved mapping still true? two markers, four seconds ----------
    #
    # "Even a 25-second launch is too long. You just need to flash two
    #  squares (for example, upper left and lower right) and see if that
    #  corresponds to the previously saved layout. If that corresponds, that
    #  means the camera did not move. And then you can initialize the app
    #  with the previously measured data."
    #
    # Exactly right, and it fixes a cost I introduced by following an
    # earlier instruction too literally. `calibrate_each_session` exists
    # because a camera that was nudged between sessions cannot be detected
    # after the fact -- which was true when a full remeasure was the only
    # way to find out. It is not true any more: the saved mapping PREDICTS
    # where a marker will image, so photographing two of them tests the
    # prediction directly.
    #
    # Measured: eighteen markers cost 24 seconds plus a second noise floor
    # afterwards. Two cost about four, and they answer the only question
    # the eighteen were being asked at launch -- has anything moved?
    #
    # The corners are chosen opposite each other on purpose. A camera can
    # rotate, shift or be knocked closer, and two diagonal points separate
    # all three: a shift moves both the same way, a rotation moves them
    # oppositely, and a change of distance changes the gap between them.
    # One marker would miss rotation entirely.

    # 2.5, from measurement (2026-10-18). A good mapping puts both markers within
    # 0.8-1.1 px of the prediction (8 launches); a mapping from a camera that had
    # drifted ~4 px read 3.4-4.0 px, was ACCEPTED at the old 5.0, and cost every
    # self test ~2 mm (3.3 -> 5.4 mm: READY became 'degraded'). A fresh calibration
    # took it straight back to 3.6 mm and READY.
    CHECK_TOLERANCE_PX = 2.5
    CHECK_MARGIN = 0.08          # in from the screen edge, as the grid is

    def start_mapping_check(self, then=None) -> None:
        """Photograph two markers and compare them with the saved mapping.

        The sequence is `routines.mapping_check`. This one had the most
        thorough hand-rolled guard of the six -- five of the eight flags --
        and it was still not all of them, which is the argument against
        hand-rolled guards rather than against this one.
        """
        if (self.capture is None or self.target_window is None
                or self.mapping is None):
            if then is not None:
                then(False)
            return
        job = routines.mapping_check_job(self)
        if then is not None:
            # A stood-down routine reports None, which is not a pass. "I
            # could not check" must never be read as "nothing moved": that
            # is how a bad mapping would survive into a session and draw
            # holes an inch out.
            job.on_done = lambda result: then(bool(result and result[0]))
        job.body = lambda: self._mapping_check_reporting()
        self.director.start(job)

    def _mapping_check_reporting(self):
        """The check, with its verdict logged. Thin wrapper, on purpose."""
        ok, why = yield from routines.mapping_check(self)
        self._log("  " + why)
        return (ok, why)

    def _mm_per_camera_px(self) -> float:
        """Roughly how far one camera pixel is on the panel, in mm."""
        corners = self.mapping.screen_rect_in_camera() if self.mapping else None
        if not corners:
            return 1.5
        xs = [c[0] for c in corners]
        cam_w = max(xs) - min(xs)
        if cam_w <= 1:
            return 1.5
        return 940.0 / cam_w

    # ======================================================================
    # WHAT THE DIRECTOR'S ROUTINES CALL
    # ======================================================================
    #
    # `lsa/routines.py` holds the SEQUENCES -- straight-line generators that
    # yield the thing they are waiting for. Everything below is the part
    # that needs the app: a drawing, a number, a line of report. Nothing
    # here waits, schedules, or touches the capture thread's measurement
    # entry points; that is the Director's job, and `tests/test_rules.py`
    # fails the build if a routine reaches around it.
    #
    # The split is Fable's, reviewing this file on 2026-10-15: "pull the
    # exclusion/routing policy out as the Director -- this is where the gate
    # failures live -- then move each sequence into its own module."

    # -- "is this routine running?", answered in one place ---------------
    #
    # These were plain attributes, set and cleared by hand at every exit
    # path of every routine, and read in twenty-odd places. The failures
    # that came out of that are the whole reason the Director exists:
    #
    #   * `closeEvent` stood routines down by clearing eight flags by name
    #     and omitted `_check_active`, so the two-marker check was never
    #     stood down at shutdown at all.
    #   * `_probe_active` was cleared at the top of `_probe_finish`, before
    #     the dot came down and before the background was reset, so for
    #     those few milliseconds the app believed nothing was drawing.
    #   * Five routines each tested a different subset of the others.
    #
    # As read-only properties they cannot be set, cannot be stale, and
    # cannot disagree with what is actually running. An assignment is now
    # an AttributeError at the moment it is written rather than a wrong
    # answer three screens later.
    @property
    def _probe_active(self) -> bool:
        return self.director.busy_with() == "probe"

    @property
    def _selftest_active(self) -> bool:
        return self.director.busy_with() == "selftest"

    @property
    def _check_active(self) -> bool:
        return self.director.busy_with() == "mapcheck"

    def _process_role(self) -> str:
        """`live`, `harness:<class>`, or `nocamera`. Who produced a number.

        DERIVED FROM FRAMES ACTUALLY RECEIVED, not from the capture's class
        name. The first version read the class name and called anything
        named `CaptureThread` live -- so `test_ui_smoke`, which constructs a
        real CaptureThread, never opens it, and publishes a verdict, wrote
        nine `GATE role=live` lines into `camera_lab.log`. The live log. The
        tag invented to stop a measurement being attributed to the wrong
        process was itself attributing test data to his hardware, and
        `tools/status.py` would have reported one of those as "the last
        measurement on record".

        Found by an outside review, not by me, and it is the third time on
        this project that a plausible discriminator has been chosen over a
        true one: "which routine is running" instead of "could this be a
        trigger pull", mean frame brightness instead of what the detector
        consumes, and now a class name instead of whether a camera
        delivered anything.
        """
        cap = self.capture
        if cap is None:
            return "nocamera"
        name = type(cap).__name__
        if name != "CaptureThread":
            return f"harness:{name}"
        if self._frames_ever <= 0:
            # A real capture class that never delivered a frame is not a
            # live measurement of anything.
            #
            # Counted here rather than from `Stats.frames`, which
            # `reset_stats()` zeroes -- and `_on_noise_measured` calls
            # `reset_stats()`, so a counter read from there can be 0 in the
            # middle of a perfectly real launch. A number that answers
            # "has this camera ever delivered anything" must not be
            # resettable by the thing it is reporting on.
            return "noframes"
        return "live"

    def _screen_label(self) -> str:
        """The target screen's size, as a sanity check on the role.

        800x800 is the harness; 3840x2160 is his monitor. This is the
        number that was sitting in the line I misread.
        """
        win = self.target_window
        if win is None:
            return "noscreen"
        r = win.rect()
        return f"{r.width()}x{r.height()}"

    def _rejected_runs(self) -> int:
        """The too-short counter, read from the attribute that exists.

        `capture.stats` does NOT exist -- it is `_stats` -- and reading the
        plausible name raised inside a Qt slot, stopping a self test
        mid-run with nothing said at all (RULES C8).
        """
        return int(self.capture._stats.rejected_runs) if self.capture else 0

    # -- the self test ---------------------------------------------------

    def _selftest_grid(self, rounds: int) -> list[tuple[float, float]]:
        """The points to flash: the same grid the fit was NOT measured on.

        Held out on purpose, so this checks the mapping as well as the
        detector. Reporting fit error instead of held-out error is how a
        mapping with an RMS of 1287px on a 3840px screen was once saved and
        announced as a success (RULES C9).
        """
        rect = self.target_window.rect()
        return [(x, y) for _r in range(max(1, rounds))
                for (x, y) in marker_points(rect.width(), rect.height(),
                                            cols=4, rows=3, margin=0.14)]

    def _selftest_middle(self) -> tuple[float, float]:
        rect = self.target_window.rect()
        return (rect.width() * 0.5, rect.height() * 0.5)

    def _selftest_spread(self) -> list[float]:
        """The brightness rungs: deciles of his 575 pulls over his median."""
        synth = self.scene.get("synthetic")
        if synth is None:
            return [1.0]
        return list(type(synth).BRIGHTNESS_SPREAD)

    def _selftest_arm(self) -> None:
        self._selftest_seen = None

    def _selftest_retune(self, radius: float, seen: object,
                         left: int) -> "tuple[float, bool, str]":
        """New radius, whether to stop tuning, and what to say about it.

        Area goes as radius squared, so the correction is the square root of
        the ratio, damped to half because the relationship is only
        approximately that -- bloom is a property of the sensor, not of
        geometry, and a small bright dot blooms proportionally more than a
        large one. Three or four steps settle it.
        """
        if seen is None:
            # Too small to see at all: open up rather than vanish -- but
            # never past a size a laser could never be. A dot bigger than
            # that is not testing the detector, it is testing nothing, and
            # growing it hides whatever is really wrong. One run grew to a
            # 410px radius against a screen cover it could not have been
            # seen through, and reported the dot as the problem.
            ceiling = self._selftest_dot_radius(60.0)
            if radius >= ceiling:
                return (radius, True,
                        f"  tuning: STILL not seen at radius {radius:.0f}px, "
                        f"which images about 60 camera px across -- far "
                        f"larger than any laser dot. The dot is not the "
                        f"problem: check that the target screen is on top, "
                        f"that nothing is covering it, and the threshold.")
            grown = min(ceiling, radius * 1.6)
            return (grown, False,
                    f"  tuning: not seen, growing to {grown:.0f}px "
                    f"({left} left)")
        (_pt, peak, area, pulse) = seen
        ratio = math.sqrt(max(0.05, self.SELFTEST_TARGET_AREA / max(1.0, area)))
        tuned = max(3.0, radius * (1.0 + (ratio - 1.0) * 0.5))
        note = (f"  tuning: camera saw {area:.0f}px (want "
                f"{self.SELFTEST_TARGET_AREA:.0f}), peak {peak:.0f}, on "
                f"screen {pulse:.0f}ms -> radius {tuned:.0f}px ({left} left)")
        if left <= 0:
            note += (f"\n  dot tuned to radius {tuned:.0f} screen px. "
                     f"Now measuring the grid.")
        return (tuned, False, note)

    def _selftest_line(self, i: int, total: int, want: tuple[float, float],
                       got: object, bright: float) -> str:
        if got is None:
            return (f"  dot {i + 1}/{total} at ({want[0]:.0f},{want[1]:.0f}) "
                    f"at {bright * 100:.0f}% brightness: NOT DETECTED")
        (gx, gy), peak, area, pulse = got
        err = math.hypot(gx - want[0], gy - want[1])
        mm = err * (940.0 / max(1, self.target_window.rect().width()))
        return (f"  dot {i + 1}/{total} at ({want[0]:.0f},{want[1]:.0f}) -> "
                f"({gx:.0f},{gy:.0f})  off by {err:.0f}px = {mm:.1f}mm  "
                f"peak {peak:.0f} area {area:.0f}px  at "
                f"{bright * 100:.0f}% brightness")

    def _remember_dot_radius(self, radius: float, area: object) -> None:
        """Keep the tuned dot size with the mapping it belongs to.

        `area` is the blob the camera measured when the dot was flashed AT
        `radius`, and the radius is refused unless that blob is one his laser
        could have made (`dot_area_in_band`). The gate lives HERE, at the one
        writer, as well as in the routine: the ratchet (8.7 -> 47 over two
        days; 9.62 -> 14.43 in one session on 2026-10-20) was growth logic and
        persistence in different places, and a caller added later must not be
        able to route around the gate by forgetting it.

        Saved only when it has changed meaningfully -- 2% -- because
        `Mapping.save` writes the whole calibration file and there is no
        reason to rewrite it on every launch for a number that settled.
        """
        m = self.mapping
        if m is None or radius <= 0:
            return
        if not self.dot_area_in_band(area):
            self._log(f"  dot radius {radius:.0f}px NOT saved: it imaged "
                      f"{'nothing' if area is None else f'{float(area):.0f}px'}"
                      f", outside the {self.SELFTEST_AREA_BAND[0]:.0f}-"
                      f"{self.SELFTEST_AREA_BAND[1]:.0f}px his laser measures")
            return
        was = float(getattr(m, "dot_radius", 0.0) or 0.0)
        if was and abs(radius - was) / max(1.0, was) < 0.02:
            return
        m.dot_radius = float(radius)
        try:
            # THE FOURTH CALLER. "All three callers" were fixed this
            # morning and this one was left standing -- which is Part E of
            # RULES in a single line: the instance in front of me, not the
            # rule. Found by an outside review.
            if self._save_mapping(m, "the dot radius"):
                self._log(f"  dot radius {radius:.0f}px saved with the "
                          f"calibration, so the next launch confirms it with "
                          f"one flash instead of finding it with four")
        except Exception as exc:
            self._log(f"  could not save the dot radius: {exc!r}")

    def _selftest_report(self, results: list, rej_before: int) -> None:
        """Hand the measured results to the existing reporting and verdict."""
        self._selftest_results = list(results)
        self._selftest_rej_before = int(rej_before)
        self._selftest_finish()

    # -- the two-marker mapping check ------------------------------------

    def _check_targets(self) -> list[tuple[float, float]]:
        """Two markers at OPPOSITE corners, which is the whole design.

        His idea: "flash two squares (for example, upper left and lower
        right) and see if that corresponds to the previously saved layout."
        Diagonal on purpose -- a shift moves both the same way, a rotation
        moves them oppositely, a change of distance alters the gap. One
        marker would miss rotation entirely.
        """
        if self.target_window is None or self.mapping is None:
            return []
        rect = self.target_window.rect()
        w, h = rect.width(), rect.height()
        m = self.CHECK_MARGIN
        return [(w * m, h * m), (w * (1 - m), h * (1 - m))]

    def _check_exposure(self) -> float:
        """The dark, FIXED exposure to photograph the two markers at.

        The first of the same list the screen search works through, for the
        same reason: the room goes black while the lit screen does not, so
        the monitor is the only thing in frame that changes. Measured on
        this bench -- auto 14.9, -9 43.2, -10 44.3, -11 43.9.

        A fixed value rather than whatever is currently set, because the
        value currently set is the one chosen for seeing a laser dot
        against a dark screen, which is a different job.
        """
        return float(self._calib_exposures[0] or -10.0)

    def _check_show(self, point: "tuple[float, float] | None") -> None:
        """Put the marker up, or take just the MARKER down.

        `show_at(None)` keeps the black field and removes the marker, which
        is exactly what the unlit photograph of a pair needs. It does NOT
        hide the layer -- see `_check_hide`, and the incident there.
        """
        pattern = self.scene.get("pattern")
        if pattern is None:
            return
        pattern.show_at(point)
        self.scene.mark_dirty()

    def _check_hide(self) -> None:
        """Take the whole pattern layer down. Not the same as show_at(None).

        2026-10-16, and it cost him a session. The two-marker check's
        cleanup called `_check_show(None)`, reading it as "take the marker
        away". It takes the MARKER away and leaves the layer visible -- and
        that layer paints the ENTIRE SCREEN BLACK at z=80, above
        everything, because the only thing that may differ between the lit
        and unlit photographs is the marker itself.

        So the screen stayed black for the whole session: no prompt band,
        no verdict, no bullet holes, and the self test's own red dots
        photographed through a black cover (10 of 12 "not detected"). He
        saw it before I did -- "I see a completely black screen" -- while
        every number in the log read healthy.

        It surfaced only when the check started SUCCEEDING. On a failure
        the full pattern calibration runs next and ends with `stop()`, so
        the layer came down by accident; on the fast path nothing called
        it. A cleanup that works because of what runs after it is not a
        cleanup (RULES B2).
        """
        pattern = self.scene.get("pattern") if self.scene is not None else None
        if pattern is None:
            return
        pattern.stop()
        self.scene.mark_dirty()

    def _check_locate(self, lit: object, dark: object) -> object:
        if lit is None or dark is None:
            return None
        try:
            return find_marker(lit, dark)
        except Exception:
            return None

    def _check_line(self, want: tuple[float, float], found: object) -> str:
        """SAY WHAT WAS MEASURED, not just whether it passed.

        The first version reported "one of the two markers was not visible"
        and nothing else, so a failure gave no way to tell a marker that was
        dim from one in the wrong place from one that was never drawn. A
        check that reports only a verdict cannot be debugged, only re-run
        (RULES C3).
        """
        pred = (self.mapping.to_camera(want[0], want[1])
                if self.mapping is not None else None)
        if found is None:
            return (f"    marker at screen ({want[0]:.0f},{want[1]:.0f}): "
                    f"NOT FOUND"
                    + (f", expected near camera ({pred[0]:.0f},{pred[1]:.0f})"
                       if pred else ""))
        off = (math.hypot(found[0] - pred[0], found[1] - pred[1])
               if pred else -1.0)
        return (f"    marker at screen ({want[0]:.0f},{want[1]:.0f}): found "
                f"at camera ({found[0]:.0f},{found[1]:.0f}) strength "
                f"{found[2]:.0f}, {off:.1f}px from the prediction")

    def _check_verdict_from(self, results: list) -> "tuple[bool, str]":
        """Did both markers land where the saved mapping says?

        A marker that was not found at all FAILS rather than being ignored:
        "I could not see it" is not evidence that nothing moved, and
        treating it as such is how a bad mapping would survive into a
        session and draw holes an inch out.
        """
        if self.mapping is None or len(results) < 2:
            return False, "the check did not complete; measuring properly instead"
        offsets = []
        for want, got in results:
            if got is None:
                pred = self.mapping.to_camera(want[0], want[1])
                where = (f" (expected near camera ({pred[0]:.0f},"
                         f"{pred[1]:.0f}))" if pred else "")
                return False, (f"the marker at screen ({want[0]:.0f},"
                               f"{want[1]:.0f}) was not visible{where}, so "
                               f"the saved mapping cannot be confirmed; "
                               f"measuring properly instead")
            pred = self.mapping.to_camera(want[0], want[1])
            if pred is None:
                return False, ("the saved mapping cannot predict where a "
                               "marker should be; measuring properly instead")
            offsets.append(math.hypot(got[0] - pred[0], got[1] - pred[1]))
        worst = max(offsets)
        mm = self._mm_per_camera_px() * worst
        if worst <= self.CHECK_TOLERANCE_PX:
            return True, (f"the camera has not moved: both markers landed "
                          f"within {worst:.1f}px of where the saved mapping "
                          f"predicts ({mm:.1f}mm). Keeping it -- no "
                          f"recalibration needed.")
        return False, (f"the camera HAS moved: a marker is {worst:.1f}px from "
                       f"where the saved mapping predicts ({mm:.1f}mm on "
                       f"screen), past the {self.CHECK_TOLERANCE_PX:.0f}px "
                       f"bar. Remeasuring.")

    # -- measuring the exposure with the emulated shot ------------------------
    #
    # "You should be able to emulate the exact thing that my gun does and
    #  find the right combination of threshold and brightness so that your
    #  code is auto-calibrated every single time the program starts."
    #
    # What this replaces: a constant. `MEAN_CEILING = 40` rejected any
    # exposure whose frame mean exceeded it, and on 2026-10-15 afternoon
    # light through the window beside the monitor pushed that mean up and
    # stopped the camera down three stops. His own trigger pulls went from
    # a 10x margin to 1.2x, and half of every string vanished -- while the
    # app reported a successful exposure measurement.
    #
    # What this measures instead, at each candidate exposure, by
    # photographing the screen twice:
    #
    #     dot_peak   how brightly a known red dot images
    #     floor      what the background scores, which sets the bar
    #
    # Both inside the screen polygon, because the window beside the
    # monitor is not where anyone is shooting. `lsa/exposure.py` turns
    # those into "what fraction of his 575 logged trigger pulls would
    # clear the bar here", and that fraction picks the exposure.
    #
    # It photographs rather than detects on purpose: no tracker, no
    # thresholds, no shot plumbing, nothing that could reject the very
    # measurement being taken. Two stills per candidate, one control
    # write per candidate -- strictly less camera traffic than the sweep
    # it replaces, which writes nine times.

    # THREE CANDIDATES, NOT FIVE.
    #
    # Every candidate is a control write, and control writes are what
    # hang this camera -- the gate wedged it three times in an
    # afternoon before that was traced to its own traffic. The outer
    # pair earned nothing: across every live run the predicted rate was
    # already 100% at the base and its neighbours, and the choice came
    # from the margin between them. A fourth and fifth reading bought
    # two more writes and no decision.
    #
    # The exposure also moves by at most a stop per launch now, which
    # is the right behaviour for a room that changes slowly: it tracks
    # the day instead of jumping across it.
    PROBE_STEPS = (0.0, -1.0, 1.0)
    PROBE_SETTLE_MS = 420         # after a control write, before photographing
    PROBE_DOT_MS = 160            # the dot is up while the still is taken

    # THE DOT MUST NOT SATURATE, or the whole measurement is worthless.
    #
    # Measured on the first live run: at full brightness the dot read
    # 105 / 106 / 109 / 111 across -3 to -6, which is EIGHT TIMES the
    # integration time for a two percent change. A saturated dot reports
    # the sensor's ceiling, not the light that reached it, so the
    # exposure response -- the only thing this probe exists to measure --
    # was invisible, and the choice came down to a 4.0-against-4.4
    # difference that is pure noise.
    #
    # So the brightness is auto-ranged ONCE, at the brightest candidate,
    # and then held fixed for every exposure. Fixed is what makes the
    # readings comparable; the ladder is coarse because only the order
    # of magnitude matters.
    PROBE_BRIGHTNESS_LADDER = (0.06, 0.10, 0.16, 0.26, 0.42, 0.68, 1.00)
    PROBE_PEAK_LO = 25.0          # below this the reading is noise, not signal

    def start_exposure_probe(self, then=None) -> None:
        """Measure several exposures by photographing the screen's own dot.

        The sequence itself is `routines.exposure_probe`. This is only the
        decision to run it, and the Director decides whether it may: there
        is one run slot, so the three hand-rolled flag checks that used to
        be here -- which did not include the self test -- are gone, along
        with the 900ms self-rescheduling loop that had no bound.
        """
        if (self.capture is None or self.target_window is None
                or self.mapping is None):
            self._log("measuring the exposure this way needs the camera, the "
                      "target screen and a mapping; keeping what is set")
            if then is not None:
                then()
            return
        job = routines.probe_job(self)
        if then is not None:
            job.on_done = lambda _value: then()
        self.director.start(job)

    def _probe_measure(self, lit, dark):
        """Chroma peak of the dot, and the background floor, inside the ROI.

        Scored exactly the way the detector scores -- red minus the stronger
        of green and blue -- because that is the number a shot is thresholded
        on. Anything else here would be a proxy, and this project has paid
        for proxies four times.
        """
        import numpy as _np
        if lit is None or dark is None:
            return None, 0.0, 0.0
        det = getattr(self.capture, "_detector", None)
        mask = None
        if det is not None:
            try:
                mask = det._roi_image(lit.shape[:2])
            except Exception:
                mask = None
        def chroma(img):
            b, g, r = cv2.split(img)
            return cv2.subtract(r, cv2.max(g, b))
        c_lit, c_dark = chroma(lit), chroma(dark)
        if mask is not None and mask.shape == c_lit.shape:
            sel = mask > 0
        else:
            sel = _np.ones(c_lit.shape, dtype=bool)
        if not sel.any():
            return None, 0.0, 0.0
        # The dot against the SAME scene without it. Subtracting the dark
        # frame removes anything red that is simply always there, so a
        # reflection cannot be measured as the dot -- the mistake that cost
        # a pattern calibration ten of its eighteen markers.
        diff = cv2.subtract(c_lit, c_dark)
        flat = diff[sel]
        peak = float(flat.max())

        # THE FLOOR IS MEASURED THE WAY THE DETECTOR MEASURES IT.
        #
        # The first version took the raw chroma max of the dark frame.
        # That is not what the detector thresholds: it subtracts a
        # running background on every frame, so anything red and STATIC
        # is absorbed within seconds and what is left is the
        # frame-to-frame variation. The two disagree badly -- the raw
        # version reported 16, 16, 16, 14, 10 across five exposures
        # while the app's own noise measurement, running minutes apart
        # in the same room, reported 9, 5 and 4.
        #
        # A floor that barely moves makes the bar barely move, which
        # leaves dot brightness alone to decide -- and dot brightness
        # alone always prefers more exposure. The one thing that must
        # stop it in a lit room is the floor climbing, so the floor has
        # to be the real one.
        #
        # The difference between these two frames IS a background
        # subtraction, so the floor is simply the worst of it away from
        # the dot. Ten pixels either side of the brightest point covers
        # a dot imaging at the 15-25 camera px these notes measure.
        import numpy as _np2
        idx = int(_np2.argmax(_np2.where(sel, diff, 0)))
        cy, cx = divmod(idx, diff.shape[1])
        away = sel.copy()
        r = self.PROBE_FLOOR_EXCLUDE_PX
        y0, y1 = max(0, cy - r), min(diff.shape[0], cy + r + 1)
        x0, x1 = max(0, cx - r), min(diff.shape[1], cx + r + 1)
        away[y0:y1, x0:x1] = False
        floor = float(diff[away].max()) if away.any() else 0.0
        mean = float(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)[sel].mean())
        return peak, floor, mean

    # Half-width, in camera px, of the window read around each probe spot. The
    # dot images at 15-25 px and the mapping places it to about 2 px.
    PROBE_SPOT_HALF = 40
    # Radius, camera px, of the box blanked round each dot before the FLOOR is
    # read. It was 26 and the floor then read 101-104 at every exposure, with its
    # maximum 41 px from the strong dot (live 2026-10-18, floor_at in the READING
    # line): the saturated dot's own bloom, not the room. Floor ~ dot -> bar above
    # dot -> the probe never chose an exposure on any launch. The bloom of a full-red
    # dot reaches well past the 2*PROBE_SPOT_HALF window, so blank 2.5x it.
    PROBE_FLOOR_EXCLUDE_PX = 100

    def _save_probe_evidence(self, lit, dark) -> None:
        """Keep the two photographs the probe gave up on.

        The probe's "the emulated dot is not reaching the camera" is the one
        conclusion in this app that cannot be checked from the log, because the
        log holds only the number it computed. On 2026-10-24 it fired on a
        healthy bench -- camera opening in 0.42s, monitor awake, the target
        screen in front of it by the app's own window check, and the very next
        launch reading 108 at the same exposure and reaching READY 12/12 -- and
        the advice it produced was to unplug a camera that had nothing wrong.

        One pair of PNGs turns the next occurrence into a one-second question:
        is the dot on the screen in `probe_lit`, or is it not? Writing them
        costs nothing on a path that is already ending the probe, and nothing
        downstream reads them -- they are for a person.
        """
        try:
            import cv2 as _c
            RECORDINGS.mkdir(parents=True, exist_ok=True)
            for name, frame in (("probe_lit", lit), ("probe_dark", dark)):
                if frame is not None:
                    _c.imwrite(str(RECORDINGS / f"{name}.png"), frame)
            self._log(f"    kept what the probe judged: "
                      f"{RECORDINGS / 'probe_lit.png'} and probe_dark.png -- "
                      f"if the dot IS on the screen in probe_lit, the camera "
                      f"missed a frame and the advice below is wrong")
        except Exception as exc:                             # noqa: BLE001
            self._log(f"    (could not save the probe's frames: {exc!r})")

    def _probe_measure_pair(self, lit, dark, weak, strong):
        """(weak_peak, floor, mean, strong_peak) from ONE lit/dark pair that
        carries the dot at two places.

        `weak` and `strong` are screen fractions (as `_show_dots` draws them);
        `strong` may be None, in which case this is `_probe_measure` and
        strong_peak is None. Each spot is read inside its own window around
        where the mapping says it is, because the whole-ROI maximum of
        `_probe_measure` would return the brighter dot for both. The floor is
        the worst of the difference image outside both dots, as before.
        """
        if strong is None or lit is None or dark is None:
            peak, floor, mean = self._probe_measure(lit, dark)
            return peak, floor, mean, None
        import numpy as _np
        m = self.mapping
        rect = self.target_window.rect() if self.target_window is not None else None
        pts = []
        if m is not None and rect is not None:
            for fx, fy in (weak, strong):
                try:
                    c = m.to_camera(rect.width() * fx, rect.height() * fy)
                except Exception:                              # noqa: BLE001
                    c = None
                pts.append(c)
        if len(pts) != 2 or pts[0] is None or pts[1] is None:
            # No way to place the spots: read the weak end the old way and say
            # nothing about the strong one (the log reports it NOT MEASURED).
            peak, floor, mean = self._probe_measure(lit, dark)
            return peak, floor, mean, None
        det = getattr(self.capture, "_detector", None)
        mask = None
        if det is not None:
            try:
                mask = det._roi_image(lit.shape[:2])
            except Exception:
                mask = None

        def chroma(img):
            b, g, r = cv2.split(img)
            return cv2.subtract(r, cv2.max(g, b))
        diff = cv2.subtract(chroma(lit), chroma(dark))
        h, w = diff.shape[:2]
        sel = (mask > 0) if (mask is not None and mask.shape == diff.shape) \
            else _np.ones(diff.shape, dtype=bool)
        if not sel.any():
            return None, 0.0, 0.0, None
        R = self.PROBE_SPOT_HALF
        peaks, centres = [], []
        for cx, cy in pts:
            x0, x1 = max(0, int(cx) - R), min(w, int(cx) + R + 1)
            y0, y1 = max(0, int(cy) - R), min(h, int(cy) + R + 1)
            if x1 <= x0 or y1 <= y0:
                peaks.append(0.0)
                centres.append((int(min(max(cx, 0), w - 1)), int(min(max(cy, 0), h - 1))))
                continue
            win = _np.where(sel[y0:y1, x0:x1], diff[y0:y1, x0:x1], 0)
            k = int(_np.argmax(win))
            iy, ix = divmod(k, win.shape[1])
            peaks.append(float(win[iy, ix]))
            centres.append((x0 + ix, y0 + iy))
        away = sel.copy()
        r = self.PROBE_FLOOR_EXCLUDE_PX
        for ccx, ccy in centres:
            away[max(0, ccy - r):min(h, ccy + r + 1),
                 max(0, ccx - r):min(w, ccx + r + 1)] = False
        floor = float(diff[away].max()) if away.any() else 0.0
        # WHERE the floor's maximum sits: a record only. On his bench the probe
        # read floors of 83-106 against a dot of 46-97 and never chose; if the
        # maximum sits just outside a dot's 26px box it is the dot's own bloom,
        # not the room. (x, y, px from each dot)
        self._probe_floor_at = None
        if away.any():
            _fy, _fx = divmod(int(_np.argmax(_np.where(away, diff, 0))), w)
            self._probe_floor_at = (_fx, _fy, [round(float(((_fx - a) ** 2 + (_fy - b) ** 2) ** 0.5))
                                               for a, b in centres])
        mean = float(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)[sel].mean())
        return peaks[0], floor, mean, peaks[1]

    # -- the screen tests the camera: synthetic red dots ----------------------
    #
    # Asked for in these words: "you should just flash small red dots on screen
    # and see how they are being caught by camera... then you see how the rest
    # of the machinery catches those red dots and where the holes appear."
    #
    # It is the test this project should have had from the first evening. The
    # screen draws a saturated red dot at a position known to the pixel, the
    # real camera photographs it, the real detector scores it, the real tracker
    # confirms it and the real mapping places it -- and the answer is compared
    # with where it was drawn. Nobody pulls a trigger, so it can run as often
    # as it likes, unattended, after every change.
    #
    # The white markers of pattern calibration cannot do this. Their chroma is
    # zero, so they never touch the detector; that is exactly how a sensitivity
    # map built from them came out uniform while the laser varied two-fold.
    #
    # Honest limits, stated once here so no result is over-claimed: the screen
    # EMITS this dot, a laser REFLECTS off the panel. Brightness, angular
    # spread and spectrum all differ, so the detection RATE measured here is
    # not the gun's. What transfers exactly is geometry, plumbing and
    # ACCURACY -- where every fault found so far has lived.

    SELFTEST_HOLD_MS = 120        # a LaserLyte pulse is ~100ms; match it
    SELFTEST_GAP_MS = 320         # let the tracker close the shot and settle
    # What a REAL laser dot measures on this hardware, in camera pixels of
    # thresholded blob: about 400 in the middle of the screen and 150 on the
    # right third (gun test, 2026-10-14, 36 shots). The synthetic dot is sized
    # to match that, by measurement rather than by arithmetic.
    #
    # Arithmetic is not enough, and the shooter spotted it before the code
    # did: "your red dots are much bigger than mine". Scaling 11 camera px of
    # radius through the mapping gave 63 screen px, and the camera then
    # measured a blob of radius 21 -- twice the diameter asked for. A screen
    # EMITS its dot and blooms in the sensor; a laser dot REFLECTS off matte
    # coating and does not. So the size is tuned in a feedback loop against
    # what the camera actually reports.
    SELFTEST_TARGET_AREA = 400.0
    SELFTEST_TUNE_SHOTS = 4
    # THE BAND A DOT MUST IMAGE IN BEFORE ITS RADIUS MAY BE SAVED (2026-10-22).
    #
    # Taken from his real shots, not typed: p10 113px and p90 746px over 575
    # logged trigger pulls (CLAUDE.md, 2026-10-15). A dot whose blob falls in
    # the range his laser actually occupies is a fair stand-in for it; one
    # outside that range is a dot sized to get a result rather than to be
    # a laser. `SELFTEST_TARGET_AREA` (400) sits inside it, so a tuner that
    # has converged is always in band.
    #
    # This is the bound on the radius too, and the better one: it limits the
    # quantity the detector consumes (the blob) and not the geometry that
    # produces it, which depends on the mapping and on sensor bloom. A radius
    # cap would have passed 47 screen px (about 8.6 camera px of radius on
    # this bench, a perfectly ordinary dot) and could not tell a good 47 from
    # a bad 14.
    SELFTEST_AREA_BAND = (113.0, 746.0)

    @classmethod
    def dot_area_in_band(cls, area: object) -> bool:
        """True when `area` (camera px) is a blob his laser could have made.

        None -- the dot was not seen -- is never in band.
        """
        if area is None:
            return False
        lo, hi = cls.SELFTEST_AREA_BAND
        return lo <= float(area) <= hi  # type: ignore[arg-type]

    def start_self_test(self, rounds: int = 1) -> None:
        """Flash red dots at known places and measure where they are reported.

        The sequence is `routines.self_test`. What used to be here -- four
        flag checks that did not include the exposure probe, and a 900ms
        self-rescheduling loop with no bound -- is the Director's one run
        slot now. The interlock that "does wait properly" and never fired
        once, because the probe handed off before `_calibrating` was
        raised, cannot be armed too late when there is nothing to arm.
        """
        if self.capture is None or self.target_window is None:
            self._log("the self test needs the camera and the target screen")
            return
        if self.mapping is None:
            self._log("the self test needs a mapping first: press C, then A. "
                      "Without one there is nothing to compare a hit against.")
            return
        self.director.start(routines.self_test_job(self, rounds))

    def _probe_spot(self) -> "tuple[float, float]":
        """Where the exposure probe should put its dot, as screen fractions.

        THE WEAKEST REGION THE MAPPING KNOWS ABOUT, because that is where
        his shots are lost. An exposure chosen on the bright half of the
        panel protects the half that was never in danger: measured here,
        the middle of his screen images a shot at peak 68 while the far
        right reads 10-13 against a bar of 18.

        The sensitivity rows are camera points with the peak a real shot
        made there, so the weakest row is the hardest place on the screen,
        already measured from his own trigger pulls. Mapped back through
        the homography to the screen fractions the probe draws in.

        Falls back to the centre when nothing has been learned -- a first
        launch on a new camera has no opinion, and the centre is the
        honest default rather than a guess at which side is dim.
        """
        default = (0.5, 0.5)
        m = self.mapping
        rows = list(getattr(m, "sensitivity", None) or []) if m else []
        if not rows:
            # No learned falloff: the FAR edge by geometry (a guess, labelled
            # as one in the probe's log) rather than the centre, because the
            # probe now measures both ends and the centre is neither.
            spot, _note = self._probe_strong_spot()
            if spot is not None and m is not None:
                return (1.0 - spot[0], 0.5)
            return default
        try:
            weakest = min(rows, key=lambda r: float(r.get("peak", 0.0)))
            cam = weakest.get("camera") or ()
            if len(cam) < 2:
                return default
            sx, sy = m.to_screen(float(cam[0]), float(cam[1]))
            w, h = m.screen_size
            if not (w and h):
                return default
            fx = min(0.9, max(0.1, float(sx) / float(w)))
            fy = min(0.9, max(0.1, float(sy) / float(h)))
            return (fx, fy)
        except Exception:                                    # noqa: BLE001
            return default

    def _probe_strong_spot(self) -> "tuple[tuple[float, float] | None, str]":
        """Where the probe puts its dot at the STRONG end, and how it knows.

        From the saved sensitivity when there is one (the strongest row), else
        a GEOMETRIC GUESS: the screen edge that images longer is nearer the
        camera, hence brighter. The guess is labelled as one, because it is
        not a measurement and the log should not let it read as one.
        """
        m = self.mapping
        if m is None:
            return None, "no mapping"
        rows = list(getattr(m, "sensitivity", None) or [])
        w, h = m.screen_size
        if rows and w and h:
            try:
                top = max(rows, key=lambda r: float(r.get("peak", 0.0)))
                cam = top.get("camera") or ()
                if len(cam) >= 2:
                    sx, sy = m.to_screen(float(cam[0]), float(cam[1]))
                    return ((min(0.9, max(0.1, float(sx) / float(w))),
                             min(0.9, max(0.1, float(sy) / float(h)))),
                            "from the strongest region the saved sensitivity "
                            "measured")
            except Exception:                                # noqa: BLE001
                pass
        try:
            c = m.screen_rect_in_camera()
            if c:
                import math as _m
                left = _m.hypot(c[0][0] - c[3][0], c[0][1] - c[3][1])
                right = _m.hypot(c[1][0] - c[2][0], c[1][1] - c[2][1])
                fx = 0.1 if left >= right else 0.9
                return ((fx, 0.5),
                        f"a GEOMETRIC GUESS, not a measurement: the "
                        f"{'left' if fx < 0.5 else 'right'} edge images "
                        f"longer ({max(left, right):.0f} against "
                        f"{min(left, right):.0f} camera px), so it is nearer "
                        f"the camera")
        except Exception:                                    # noqa: BLE001
            pass
        return None, "no geometry available"

    def _selftest_dot_radius(self, camera_px: float) -> float:
        """A screen radius that images at about `camera_px` in the camera."""
        corners = self.mapping.screen_rect_in_camera() if self.mapping else None
        rect = self.target_window.rect() if self.target_window else None
        if not corners or rect is None or not rect.width():
            return 40.0
        xs = [c[0] for c in corners]
        cam_w = max(xs) - min(xs)
        if cam_w <= 1:
            return 40.0
        return max(6.0, camera_px * (rect.width() / cam_w))

    def _selftest_finish(self) -> None:
        # No flag to clear: the routine is still running when this is
        # called, and stops being active when it returns. The old code
        # cleared `_selftest_active` here, at the TOP of the reporting,
        # which meant the app believed nothing was drawing while the
        # verdict board went up -- and the verdict board is what poisoned
        # the noise floor from 0-2 to 16-18.
        res = self._selftest_results
        seen = [(w, g) for w, g in res if g is not None]
        mm_per_px = 940.0 / max(1, self.target_window.rect().width())
        self._log("=== SELF TEST RESULT ===")
        self._log(f"  detected {len(seen)} of {len(res)} dots")
        if not seen:
            self._log("  nothing was detected at all. The detector is not seeing "
                      "the screen's own red dot, so it cannot see a laser either: "
                      "check the exposure, the threshold and that the camera is "
                      "looking at this screen.")
            # AND STILL PUBLISH. This return used to end the method, which
            # skipped the verdict in the one case it exists for: an app that
            # can see nothing at all then says nothing at all, and the shooter
            # is left with a blank screen to interpret. Measured 2026-10-15,
            # gate launch 1: 0 of 12 detected, no verdict, and the unattended
            # run hung to its deadline waiting for one that was never coming.
            self._publish_verdict(res, seen, [], [])
            return
        errs = sorted(math.hypot(g[0][0] - w[0], g[0][1] - w[1]) * mm_per_px
                      for w, g in seen)
        peaks = sorted(g[1] for _w, g in seen)
        areas = sorted(g[2] for _w, g in seen)
        pulses = sorted(g[3] for _w, g in seen)
        self._log(f"  dot was on screen for {pulses[0]:.0f}..{pulses[-1]:.0f}ms "
                  f"(median {pulses[len(pulses) // 2]:.0f}ms), asked for "
                  f"{self.SELFTEST_HOLD_MS}ms; a LaserLyte pulse is ~100ms")
        self._log(f"  position error: median {errs[len(errs) // 2]:.1f}mm, "
                  f"worst {errs[-1]:.1f}mm")
        self._log(f"  signal: peak {peaks[0]:.0f}..{peaks[-1]:.0f} "
                  f"(median {peaks[len(peaks) // 2]:.0f}), "
                  f"area {areas[0]:.0f}..{areas[-1]:.0f}px "
                  f"(median {areas[len(areas) // 2]:.0f}px)")
        # Split by third of the screen: the far side is the whole argument.
        rect = self.target_window.rect()
        for name, lo, hi in (("left  ", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3),
                             ("right ", 2 / 3, 1.01)):
            band = [(w, g) for w, g in seen
                    if lo <= w[0] / rect.width() < hi]
            total = [w for w, _g in res if lo <= w[0] / rect.width() < hi]
            if not band:
                self._log(f"  {name}: 0 of {len(total)} detected")
                continue
            be = sorted(math.hypot(g[0][0] - w[0], g[0][1] - w[1]) * mm_per_px
                        for w, g in band)
            bp = sorted(g[1] for _w, g in band)
            ba = sorted(g[2] for _w, g in band)
            self._log(f"  {name}: {len(band)}/{len(total)} detected, "
                      f"error median {be[len(be) // 2]:.1f}mm, "
                      f"peak median {bp[len(bp) // 2]:.0f}, "
                      f"area median {ba[len(ba) // 2]:.0f}px")
        self._log("  NOTE: a screen EMITS this dot, a laser REFLECTS off the "
                  "panel, so the detection RATE here is not the gun's. The "
                  "position error IS, because it is the same mapping.")
        self._publish_verdict(res, seen, errs, peaks)
        if self._selftest_modes:
            QTimer.singleShot(600, self._modetest_start)

    # -- the verdict ----------------------------------------------------------
    #
    # "So that when a human finally gets to shoot, he will see either a
    #  working app or the app giving him the exact reason why it is not
    #  working and what needs to be done to make it work."
    #
    # That sentence is the whole specification. The decision itself lives in
    # `lsa/selfcheck.py`, which is pure -- numbers in, verdict and actions
    # out -- so it can be tested against every failure this project has
    # produced, with the date and the measurements of each. This method does
    # only the two things that need the app: COLLECT the measurements, and
    # put the answer where the person holding the pistol will see it.

    def _collect_selfcheck(self, res=None, seen=None, errs=None,
                           peaks=None) -> SelfCheck:
        """Everything measured this launch, in one object.

        Each number is read from where it was MEASURED, never re-derived.
        A verdict assembled from fresh guesses would be the fourth proxy
        this project has mistaken for a measurement.
        """
        # `_stats`, not `stats`: the capture has no public accessor, and
        # every other reader in this file already uses the underscore.
        # Written as `stats` here, it raised inside a Qt slot and the
        # whole self-test chain stopped silently after its last line.
        st = self.capture._stats if self.capture is not None else None
        held = getattr(self.mapping, "held_out", None) or {}
        c = SelfCheck(
            fired=len(res) if res is not None else 0,
            detected=len(seen) if seen is not None else 0,
            error_mm=(errs[len(errs) // 2] if errs else None),
            worst_error_mm=(errs[-1] if errs else None),
            peak_median=(peaks[len(peaks) // 2] if peaks else None),
            noise_floor=float(self._noise_peak),
            min_peak=float(self.sp_minpeak.value()),
            fps=(float(st.fps) if st is not None else None),
            exposure=self._shooting_exposure,
            exposure_spread=getattr(self, "_exposure_spread", None),
            flash_difference=getattr(self, "_flash_strength", None),
            held_out_mm=(held.get("median_mm") if held else None),
            # THE ONE FACT THAT DECIDES WHETHER A HOLE CAN EXIST, and the
            # verdict never carried it. A launch with no mapping, no shots
            # fired and a healthy camera used to assess as "READY - shoot."
            # `_map_to_screen` returns None without a mapping and there is
            # deliberately no fallback, so this is exactly "can anything be
            # drawn at all".
            calibrated=self.mapping is not None,
            audio_ok=self._audio_verdict(),
            camera_name=str(self.settings.device_name or ""),
            # `rejected_runs` is the too-short counter: a pulse the
            # detector SAW and the tracker discarded for not lasting
            # min_frames. 24% of his real trigger pulls died here and it
            # read as a detection failure for a day, so it is part of the
            # verdict rather than a line in the stats pane.
            rejected_short=max(0, int(st.rejected_runs if st is not None else 0)
                               - int(getattr(self, "_selftest_rej_before", 0))),
            room_bright=getattr(self, "_room_too_bright", None),
            probe_dot=self._probe_dot_reading,
        )
        return c

    def _audio_verdict(self):
        """True if a gunshot would actually be heard, None if not knowable."""
        mixer = getattr(self, "mixer", None)
        if mixer is None:
            return None
        if not getattr(mixer, "available", False):
            return False
        # A dead output stream still accepts writes -- PortAudio does not
        # raise when a callback stream's endpoint disappears -- so
        # `available` alone has already reported success into silence.
        #
        # Note `ensure_alive()` returns a LOG LINE when it had to reopen
        # and "" when nothing was wrong, so its truthiness is the exact
        # opposite of health. Call it to REPAIR, then judge on the flag
        # the repair sets. The first version of this returned True only
        # when the stream had just died.
        try:
            note = mixer.ensure_alive()
        except Exception:
            return None
        if note:
            self._log("  " + note)
        return bool(getattr(mixer, "available", False))

    def _control_write_count(self) -> int:
        """UVC transactions this session has issued, or -1 if unknown.

        Reported in the GATE line so a budget can be enforced on it.
        Control writes are the one operation measured to hang this
        camera, and the repeatability gate wedged it three times with
        its own traffic before anyone could see the number.
        """
        cap = self.capture
        if cap is None:
            return -1
        return int(getattr(cap._stats, "control_writes", -1))

    def _save_verdict_screenshot(self) -> None:
        """Grab the target screen and the camera's view of it, to disk.

        Two images, because they answer different questions. The screen
        grab says what the shooter sees -- a banner in the way, a veil over
        the holes, a mode left running. The camera frame says what the
        detector sees, which is the only thing that decides whether a shot
        registers. Neither substitutes for the other.
        """
        try:
            RECORDINGS.mkdir(parents=True, exist_ok=True)
            win = self.target_window
            if win is not None:
                shot = win.grab()
                if not shot.isNull():
                    shot.save(str(RECORDINGS / "verdict_screen.png"))
            pv = getattr(self, "_last_preview", None)
            img = getattr(pv, "image", None) if pv is not None else None
            if img is not None and not img.isNull():
                img.save(str(RECORDINGS / "verdict_camera.png"))
            self._log(f"  saved what this looks like: "
                      f"{RECORDINGS / 'verdict_screen.png'} (the screen) "
                      f"and verdict_camera.png (what the camera sees)")
            self._report_screen_ink("at the verdict")
        except Exception as exc:                         # noqa: BLE001
            self._log(f"  could not save the verdict screenshot ({exc!r})")

    # What counts as "there is something on the screen", as a percentage of
    # sampled pixels. Shared with `tests/test_coldstart.py`, which fails a
    # run below it -- one number, so the app and the harness cannot
    # disagree about whether he can see anything.
    SCREEN_LIT_MIN_PCT = 1.0

    def _report_screen_ink(self, when: str) -> None:
        """Say how much of the target screen is actually lit, and by what.

        2026-10-16: his screen was ENTIRELY BLACK for a whole launch, and
        the app saved a uniformly black `verdict_screen.png` as its record
        of "what this looks like" -- evidence that looked like evidence.
        Every number in the log was healthy: the screen was found at
        confidence 1.00, the mapping was confirmed to 2.1mm, the floor was
        1 and the bar 7. Nothing in it mentioned the one fact that made the
        whole launch meaningless.

        A screenshot nobody looks at is not a check (RULES C1/D1). This
        measures it: an unlit screen is reported as a problem, in the log,
        with the state that produced it, so the next launch can be
        diagnosed from the record rather than from a photograph of the
        monitor.
        """
        try:
            win = self.target_window
            if win is None or self.scene is None:
                return
            shot = win.grab()
            if shot.isNull():
                return
            img = shot.toImage()
            # A coarse sample is enough and costs nothing: this asks "is
            # there ink", not "what does it say".
            step = max(1, img.width() // 160)
            lit = total = 0
            for y in range(0, img.height(), step):
                for x in range(0, img.width(), step):
                    total += 1
                    c = img.pixelColor(x, y)
                    if max(c.red(), c.green(), c.blue()) > 12:
                        lit += 1
            pct = (100.0 * lit / total) if total else 0.0
            shown = [la.name for la in self.scene.layers if la.visible]
            # THE SAME NUMBER THE HARNESS USES. This said "blank" only at
            # exactly zero lit pixels while `test_coldstart` fails a run
            # under 1%, so a screen showing almost nothing -- a 59% veil
            # with a few pixels through it, which is a fault this project
            # has actually shipped -- passed both. Two thresholds for one
            # question is how a check comes to mean nothing.
            if pct < self.SCREEN_LIT_MIN_PCT:
                self._log(f"  THE TARGET SCREEN IS BLANK {when}: only "
                          f"{pct:.2f}% of it is lit. The shooter sees a "
                          f"black screen and every measurement taken "
                          f"against it is meaningless.")
                self._log(f"    visible layers: {shown}; "
                          f"calibrating={self.scene.calibrating}; "
                          f"calibrated={self.scene.calibrated}")
            else:
                self._log(f"  target screen {when}: {pct:.1f}% of it is lit, "
                          f"layers {shown}")
        except Exception as exc:                         # noqa: BLE001
            self._log(f"  could not measure the target screen ({exc!r})")

    def _publish_verdict(self, res=None, seen=None, errs=None, peaks=None) -> None:
        c = self._collect_selfcheck(res, seen, errs, peaks)
        report = assess(c)
        self._last_report = report
        # The passive screen watch holds the launch to what it looked like when
        # it was last judged usable; a BLOCKED launch has no healthy standard.
        if report.verdict is Verdict.BLOCKED:
            self.screenwatch.disarm()
        else:
            self.screenwatch.arm()
        # READY is the only word for READY (RULES C10): anything else, a
        # blocked or a degraded start, is "not ready".
        # A dud camera is spoken as that, not as a generic "not ready": the
        # shooter hears WHY before his trigger finger moves.
        self.voice.say("v_ready" if report.verdict is Verdict.READY
                       else ("v_camera_bad" if camera_not_delivering(c)
                             else "v_not_ready"),
                       valid=lambda: self._last_report is report)
        # ONE MACHINE-READABLE LINE, as a deliberate contract.
        #
        # The repeatability gate has to judge a launch without a human
        # reading it, and its first version regex-matched the prose above
        # -- which failed on the frame rate, because the frame rate was
        # never written to the log at all. It lived only in the GUI stats
        # pane, so an unattended run could not see the one number that
        # decides how many camera frames a 100ms pulse lands in.
        #
        # Prose is for him; this line is for the gate. Changing it breaks
        # `repeat_gate.ps1`, which is the point of writing it down.
        #
        # CAN THE MICROPHONE HEAR THE APP'S OWN BANG: known or none, and from
        # what. One locked read of the listener's own report -- the same one
        # /state serves -- so the gate and the instrument cannot disagree.
        # `coupling_source` is "" exactly when the coupling is unknown.
        _cr = self.listener.coupling_report()
        coupling = "known" if _cr["coupling"] is not None else "none"
        coupling_source = _cr["source"] or "none"
        self._log(
            # role= AND screen= FIRST, because the whole point of them is
            # to be impossible to miss when the line is read.
            #
            # 2026-10-15: the harness and the live app both wrote
            # `camera_lab.log`, a timing loop matched the HARNESS's verdict,
            # and I reported "17 seconds to READY, 12 of 12, 0.7mm" to him
            # as a measurement of his bench. The giveaways were inside the
            # line I had just read -- an 800x800 screen and exactly 60.0
            # fps -- and I read past both. A measurement that cannot name
            # its own process is not evidence (RULES C2).
            f"GATE role={self._process_role()} "
            f"screen={self._screen_label()} "
            # HOW THE MAPPING WAS EARNED, because the fast path has no
            # fresh held-out error to report and a gate that reads a
            # missing number as "no calibration" scores the three-second
            # path as broken -- and would send the next session to fix the
            # thing that just saved 33 seconds.
            f"mapping={'confirmed' if self._check_confirmed else ('measured' if self.mapping is not None else 'none')} "
            f"verdict={report.verdict.value} "
            f"detected={c.detected}/{c.fired} "
            f"held_out_mm={(c.held_out_mm if c.held_out_mm is not None else -1):.2f} "
            f"error_mm={(c.error_mm if c.error_mm is not None else -1):.2f} "
            f"fps={(c.fps or 0):.1f} "
            f"margin={(c.margin or 0):.2f} "
            f"floor={(c.noise_floor if c.noise_floor is not None else -1):.0f} "
            f"bar={(c.min_peak or 0):.0f} "
            f"exposure={(c.exposure if c.exposure is not None else 0):g} "
            f"rejected_short={c.rejected_short} "
            f"writes={self._control_write_count()} "
            f"coupling={coupling} "
            f"coupling_source={coupling_source} "
            f"findings={len(report.findings)}")
        self._log("=== VERDICT ===")
        for line in report.lines():
            self._log("  " + line)
        if report.verdict is Verdict.READY:
            self._log("  (this is the FIRST time the app has answered the "
                      "shooter's question instead of printing a stage's "
                      "numbers. Press Q on the target screen to see it again.)")
        # AND SAVE WHAT THE SHOOTER IS LOOKING AT.
        #
        # Asked, fairly: "I gave you the screen to test. You can capture
        #  the screenshots. You can capture what the camera sees. What
        #  prevents you from doing that?"
        #
        # Nothing did. `tools/screenshot.ps1` was written for this, these
        # notes describe it as existing "so the assistant can check its own
        # UI work instead of asking the user what they see", and it was
        # not used once during a day of live runs. Every diagnosis came
        # from reading the log and inferring, and the three faults that
        # wasted his evening -- a banner that never cleared, the 59% veil
        # under it, and bullet holes invisible beneath that veil -- were
        # all plainly visible in a screenshot HE took and I did not.
        #
        # So the app saves it itself, at the one moment it matters: the
        # instant it declares a verdict. No one has to remember to ask.
        board = self.scene.get("verdict") if self.scene is not None else None
        if board is not None:
            board.show_report(report, now=time.perf_counter())
            self.scene.mark_dirty()
        # THE CHECKS HAVE ANSWERED, SO HE ARRIVES AT THE MENU. `force`,
        # because this runs while the self test still owns the run slot and
        # the "nothing may be measuring" guard would otherwise refuse.
        self._menu_when_setup_done(force=True)
        # The console banner too, so the operator side agrees with the
        # target side. Three channels have now reported success into
        # silence in this project by disagreeing with each other.
        # AND NOW photograph it, with the verdict actually on screen.
        #
        # The first version captured before the board was shown, so the
        # image was of the instant BEFORE the thing it exists to record --
        # found by opening the file and looking at it, which is the whole
        # point of saving it. A screenshot taken at the wrong moment is a
        # confident picture of nothing.
        QApplication.processEvents()      # let the board paint first
        self._save_verdict_screenshot()
        if report.verdict is Verdict.READY:
            self._set_status("READY  -  " + report.summary, "",
                             "#17331c", "#7fe08a", "#2e6b38")
        else:
            first = report.findings[0] if report.findings else None
            self._set_status(
                ("NOT READY" if report.verdict is Verdict.BLOCKED
                 else "WORKS, BUT") + "  -  " + report.summary,
                (first.action if first else ""),
                "#322e16", "#c8c070", "#5e5a2a")

    # -- the same trick, pointed at the arcade modes --------------------------
    #
    # "This is how you can debug all the other arcade modes, like Escalating
    # Ball, Bouncing Balls, and everything else."
    #
    # Exactly right, and it closes the longest-standing gap in this project:
    # the modes have unit tests driven by synthetic ShotEvents, which proves
    # the scoring arithmetic and proves nothing about whether a mode is drawn,
    # whether a real detection reaches it, or whether what it draws can be hit.
    # The bug that prompted this -- "pressing 2 or 3 brings nothing on screen"
    # -- was invisible to 36 passing mode tests, because every one of them
    # called the mode directly instead of through a key, a layer and a camera.
    #
    # Here the mode is started for real, the app asks it where its target is,
    # the SCREEN draws a red dot on that target, the CAMERA sees it, the
    # detector and mapping convert it back to a screen position, and the mode
    # is asked whether it scored. End to end, no trigger pulls.

    def _log_figure_cast(self) -> None:
        """Say, once per launch and by name, which figures the figure drills
        will draw and which they never will. A figure whose printed rings did
        not verify is EXCLUDED from every random draw rather than drawn and
        reported unscored: he reported "sometimes the shots land inside the
        targets and nothing happens", and a figure that cannot be judged
        reproduces exactly that. It is logged so an absent figure is never
        mistaken for a bug."""
        try:
            names = figure_catalog.all_names()
            bad = figure_catalog.excluded()
        except OSError as exc:
            self._log(f"figure drills: the targets folder cannot be read: {exc}")
            return
        self._log(f"figure drills (keys 7, 8, 9): {len(names) - len(bad)} of "
                  f"{len(names)} figures in the random draw")
        for n, why in sorted(bad.items()):
            self._log(f"  EXCLUDED from every random draw: {n} -- {why}")
        try:
            falls = [n for n in figure_catalog.names() if figure_catalog.fall_path(n)]
            stray = figure_catalog.unpaired_falls()
        except OSError as exc:
            self._log(f"figure drills: the fall folder cannot be read: {exc}")
            return
        self._log(f"  fall poses: {len(falls)} of {len(names) - len(bad)} figures "
                  f"have one (the rest simply disappear when they drop)")
        for fn, why in sorted(stray.items()):
            self._log(f"  UNREACHABLE fall pose, never shown: targets/fall/{fn} -- {why}")

    def _modetest_start(self) -> None:
        if (self._modetest_active or self._selftest_active
                or self._probe_active or self._pat_active
                or self._grid_active or self._sgrid_active
                or self._calibrating):
            # Same rule as everything else that owns the screen. Only one
            # caller reaches here today, which is exactly what was true of
            # pattern calibration before a second one appeared and two runs
            # interleaved for two sessions.
            #
            # Bounded. This is a DEVELOPMENT walk-through behind --modetest,
            # so giving up on it costs nothing the shooter needs; retrying
            # for ever costs twenty seconds of his monitor every time,
            # silently (RULES D10).
            if self._modetest_attempt >= 10:
                self._modetest_attempt = 0
                self._log("skipping the arcade mode walk-through: the screen "
                          "has been busy for nine seconds.")
                return
            self._modetest_attempt += 1
            QTimer.singleShot(900, self._modetest_start)
            return
        self._modetest_attempt = 0
        self._modetest_active = True
        self._modetest_queue = list(arcade.available())
        self._modetest_results = []
        self._log("=== ARCADE MODE SELF TEST === each mode is started, drawn, "
                  "and shot at with a red dot the camera has to find")
        QTimer.singleShot(200, self._modetest_next)

    def _modetest_next(self) -> None:
        if not self._modetest_queue:
            self._modetest_finish()
            return
        cls = self._modetest_queue.pop(0)
        self._modetest_mode = cls()
        self._select_mode(self._modetest_mode)
        QApplication.processEvents()
        # Let an animated mode actually move, so a target caught mid-flight is
        # the realistic case rather than the frozen opening frame.
        QTimer.singleShot(700, self._modetest_fire)

    def _modetest_fire(self) -> None:
        mode = self._modetest_mode
        rect = self.target_window.rect() if self.target_window else None
        if mode is None or rect is None:
            QTimer.singleShot(100, self._modetest_next)
            return
        host = self.layers.get("modes")
        drawn = bool(host is not None and host.visible and mode.visible)
        target = self._modetest_target(mode, rect)
        if (target is None and callable(getattr(mode, "aim_point", None))
                and self._modetest_wait_attempts < 20):
            # The mode knows how to be aimed at, it just has nothing up yet
            # (a pop-up figure is blank for 1.2-3.5 s; an advancing one waits
            # up to 1.5 s before it appears). Waiting is not "no target".
            self._modetest_wait_attempts += 1
            QTimer.singleShot(400, self._modetest_fire)
            return
        self._modetest_wait_attempts = 0
        if target is None:
            self._modetest_results.append((mode.name, drawn, None, "no target"))
            self._log(f"  {mode.name}: drawn={drawn}, but it exposes no target "
                      f"to aim at, so this cannot test it")
            QTimer.singleShot(100, self._modetest_next)
            return
        self._selftest_seen = None
        self._modetest_aim = target
        self._modetest_hits_before = int(getattr(mode, "hits", 0) or 0)
        self._modetest_shots_before = int(getattr(mode, "shots", 0) or 0)
        synth = self.scene.get("synthetic")
        synth.show_at(target, self._selftest_dot_radius(11.0))
        self.scene.mark_dirty()
        QApplication.processEvents()
        QTimer.singleShot(self.SELFTEST_HOLD_MS, self._modetest_hide)

    def _modetest_hide(self) -> None:
        synth = self.scene.get("synthetic")
        synth.stop()
        self.scene.mark_dirty()
        QApplication.processEvents()
        QTimer.singleShot(self.SELFTEST_GAP_MS, self._modetest_score)

    def _modetest_score(self) -> None:
        """Did the mode itself count the hit?

        Deliberately NOT scored by calling `host.shot()` from here. The first
        version did, several hundred milliseconds after the dot was drawn, and
        reported every animated mode as a MISS -- at 0.16 screen widths per
        second a ball travels about 270px in that time, so it had simply moved
        on. The harness was measuring its own latency and blaming the mode.

        So the detection takes the ordinary path through `_on_shot`, which
        scores against the mode at the moment the camera reports, exactly as a
        live shot does. All this has to do is read the mode's own counter
        before and after. That also means this exercises the real code rather
        than a parallel copy of it, which is the entire point of the exercise.
        """
        mode = self._modetest_mode
        if mode is None:
            QTimer.singleShot(100, self._modetest_next)
            return
        host = self.layers.get("modes")
        drawn = bool(host is not None and host.visible and mode.visible)
        after = int(getattr(mode, "hits", 0) or 0)
        scored = after > self._modetest_hits_before
        ax, ay = self._modetest_aim
        shots = int(getattr(mode, "shots", 0) or 0)
        # How far did the thing we aimed at travel while the shot was in
        # flight? A 100ms pulse plus the tracker's two-frame confirmation is
        # real latency, and an animated target does not wait for it. If this
        # exceeds the target's own radius, a perfectly aimed shot MUST miss,
        # and no amount of accuracy in the mapping can save it.
        moved = None
        now_at = self._modetest_target(mode, self.target_window.rect())
        if now_at is not None:
            moved = math.hypot(now_at[0] - ax, now_at[1] - ay)
            radius = getattr(getattr(mode, "balls", [None])[0], "r", None)                 if getattr(mode, "balls", None) else getattr(mode, "r", None)
            self._log(f"    target moved {moved:.0f}px between aiming and "
                      f"scoring" + (f", its radius is {radius:.0f}px"
                                    if radius else ""))
        if shots <= self._modetest_shots_before:
            self._modetest_results.append((mode.name, drawn, False, "not detected"))
            self._log(f"  {mode.name}: drawn={drawn}, the dot on its target was "
                      f"NOT DETECTED by the camera (the mode was never told "
                      f"about a shot at all)")
        else:
            self._modetest_results.append(
                (mode.name, drawn, scored,
                 f"aimed ({ax:.0f},{ay:.0f})"))
            self._log(f"  {mode.name}: drawn={drawn}, aimed at "
                      f"({ax:.0f},{ay:.0f}), the mode received the shot and "
                      f"says {'HIT' if scored else 'MISS'}")
        QTimer.singleShot(150, self._modetest_next)

    def _modetest_target(self, mode, rect) -> "tuple[float, float] | None":
        """Ask a mode where something shootable currently is.

        Deliberately duck-typed rather than a new abstract method: a mode that
        cannot answer is reported as untestable, which is itself a finding
        worth printing, and adding a required method would break any mode
        written against the published interface.
        """
        # Modes that need more than an attribute to be aimed at (the figure
        # drills: a ring centre on a scaled, moving drawing) answer through an
        # optional `aim_point(ctx)`. It is asked first; None means "nothing
        # up yet", which the walk-through waits out instead of reporting.
        aim = getattr(mode, "aim_point", None)
        if callable(aim):
            ctx = self.scene._context(time.perf_counter(), 0.0, rect)
            got = aim(ctx)
            return None if got is None else (float(got[0]), float(got[1]))
        for attr in ("balls", "targets"):
            items = getattr(mode, attr, None)
            if items:
                live = [t for t in items if getattr(t, "alive", True)]
                if live:
                    t = live[0]
                    return (float(t.x), float(t.y))
        if hasattr(mode, "x") and hasattr(mode, "y"):
            return (float(mode.x), float(mode.y))
        return None

    def _modetest_finish(self) -> None:
        self._modetest_active = False
        self._select_mode(None)
        self._log("=== ARCADE MODE RESULT ===")
        for name, drawn, hit, note in self._modetest_results:
            verdict = ("OK" if (drawn and hit) else
                       "NOT DRAWN" if not drawn else
                       "DRAWN BUT THE SHOT DID NOT SCORE")
            self._log(f"  {name:10s} {verdict:32s} {note}")
        bad = [r for r in self._modetest_results if not (r[1] and r[2])]
        if not bad:
            self._log("  every mode drew itself and scored a real camera hit.")
        else:
            self._log(f"  {len(bad)} mode(s) failed. A mode that is not drawn "
                      f"is the 'pressing 2 or 3 brings nothing on screen' bug; "
                      f"a mode drawn but not scoring means the shot reached it "
                      f"in the wrong place.")

    def _remeasure_noise(self) -> None:
        """Measure the noise floor again, from the firing line. Key F.

        The floor sets the rejection bar for the whole session, and the one
        thing that ruins it is a shot arriving during the three seconds it
        takes. Until now the only way to redo it was a checkbox and a button
        in the operator console -- behind a fullscreen window, at the other
        end of the room from the person holding the gun.
        """
        if self.capture is None:
            self._log("measuring the noise floor needs the camera connected")
            return
        if (self._pat_active or self._grid_active or self._sgrid_active
                or self._selftest_active):
            self._log("not measuring the noise floor during calibration")
            return
        self._noise_retries = 0
        self._log("measuring the noise floor again - hold fire for 3 seconds")
        self._signal_note("HOLD FIRE - measuring the noise floor", "warn")
        self._calibrating = True
        QTimer.singleShot(300, lambda: self.capture
                          and self._start_noise_floor(self.settings.calibrate_seconds))

    def _remeasure_exposure(self) -> None:
        """Measure the exposure against the room as it is NOW. Key X.

        The sweep is skipped on a normal launch because control writes are the
        one thing measured to hang this camera, so a remembered value is used
        instead. That is the right default and the wrong permanent answer: the
        remembered value was measured in whatever room existed when it was
        measured, and nothing ever revisits it.

        On 2026-10-14 the shooter darkened the room as far as it would go and
        the app carried on at -7 regardless, reporting a thin signal margin and
        advising him to darken a room that had no darkness left in it. The
        laser's brightness scales with exposure time, and a dark room is
        exactly the condition that permits a longer one -- so the conditions
        that should have helped most were the ones being ignored.

        Deliberately a key rather than automatic. It costs several control
        writes on a camera that has hung on them, so it happens when a person
        decides it is worth the risk, not on every launch.
        """
        if self.capture is None:
            self._log("exposure measurement needs the camera connected")
            return
        if (self._calibrating or self._pat_active or self._grid_active
                or self._sgrid_active):
            self._log("not measuring exposure during calibration")
            return
        # Prefer the measurement that asks the right question. The sweep
        # below judges a candidate on how dark the background is, which
        # is a proxy; the probe judges it on how many of his 575 logged
        # trigger pulls would clear the bar at it, which is the thing.
        # The sweep stays for the case the probe cannot run in -- no
        # mapping yet, so no screen polygon to measure inside.
        if self.mapping is not None and self.target_window is not None:
            self.start_exposure_probe()
            return
        self._log("measuring exposure against the room as it is now. This is "
                  "several control writes -- the operation this camera has "
                  "hung on -- so it is only done when asked for.")
        # Against the target background, for the same reason the noise floor
        # is: measure the conditions you will shoot under, not whatever
        # happens to be on the desktop. That was worth 90 against 2.
        self._cover_desktop_for_noise()
        QTimer.singleShot(600, lambda: self.capture
                          and self.capture.start_exposure_search())

    def _toggle_aim_view(self) -> None:
        if (self._calibrating or self._pat_active or self._grid_active
                or self._sgrid_active
                or self._probe_active or self._selftest_active):
            # This swaps camera PRESETS, which is control traffic, and
            # control traffic is the one operation measured to hang this
            # camera. Pressing V in the middle of a measurement would both
            # spoil the measurement and add writes to the budget.
            self._log("not switching the camera view while the app is "
                      "measuring -- wait for READY, then press V")
            return
        """Show the camera brightly enough to aim it, from the firing line.

        The shooting exposure deliberately renders the screen as a ghost, so
        the preview is black and useless for placing the camera. This swaps to
        the calibration exposure, puts the live view up large and in greyscale
        with the detected screen outlined on it, and swaps back. It is the
        answer to "the window was exactly black".
        """
        pane = self.scene.get("camera")
        if pane is None or self.capture is None:
            return
        on = not getattr(pane, "aiming", False)
        if on:
            self._apply_preset("calib")
            self.capture.set_aim_help(True)
            self._aim_help = True
            pane.visible = True
            pane.set_aiming(True, getattr(self.screen_quad, "polygon", None),
                            self._frame_size())
            self._alert(
                "AIMING THE CAMERA",
                "The camera is now shown bright, with the screen outlined on it "
                "if the app can find it.\n\n"
                "Aim so the WHOLE monitor is inside the view with room to spare, "
                "and as square-on as you can manage.\n\n"
                "Press V again when you are done, then G to calibrate.",
                kind="info", show_camera=True)
        else:
            self._apply_preset("laser", with_exposure=False)
            self.capture.set_aim_help(False)
            self._aim_help = False
            pane.set_aiming(False)
            self._clear_alert()
        self.scene.mark_dirty()
        self._log(f"aim view {'ON - camera brightened' if on else 'OFF'}")

    def _frame_size(self) -> tuple[int, int]:
        info = getattr(self, "_open_info", None) or {}
        return (int(info.get("width", 1280) or 1280),
                int(info.get("height", 720) or 720))

    def _select_mode(self, mode) -> None:
        host = self.layers.get("modes")
        if host is None or self.target_window is None:
            return
        ctx = self.scene._context(time.perf_counter(), 0.0, self.target_window.rect())
        if mode is None:
            mode = arcade.StaticTargets()      # there is no blank "free aim"
        # THE SAVED PLAYER COUNT TAKES EFFECT ONLY AFTER SETUP. The condition is
        # the one the app already has for "the checks have answered":
        # `_menu_when_setup_done` raises `_menu_shown_at_startup` as it puts the
        # menu up, so the first drill chosen after that is the first to split.
        wanted = self.settings.players
        players = wanted if self._menu_shown_at_startup else 1
        host.set_mode(mode, ctx, players=players)
        self._place_menu_button(host)
        if wanted == 2 and players == 1 and mode.name != "menu":
            self._log(f"  two players is saved but setup is not over: "
                      f"{mode.title} runs single-player until the menu has been shown")
        if host.versus:
            self._log(f"  two players: {mode.title} runs on each half of the screen")
        elif host.fallback_note:
            # SAID, NEVER SILENT: ignoring the setting would be
            # indistinguishable from a bug, on the screen and in the log.
            self._log(f"  {host.fallback_note}")
            # ONE SECOND, FOUR WORDS, NO VEIL, no detail line (the full
            # sentence is in the log above). His words: "it shows up for
            # longer and that interferes with the play". Anything put on this
            # screen while shooting is possible competes with the targets for
            # his attention and for the lit-area budget, and the detector is
            # watching it too. Pinned by tests/test_ui_smoke.py.
            self._alert(ONE_PLAYER_ONLY_TITLE, "", "info",
                        seconds=ONE_PLAYER_ONLY_SECONDS, dim=False)
        # A fresh round: holes and score cleared, so pressing a mode's own
        # key again restarts it.
        self._clear_holes("a new drill")
        self.session.reset()
        self.scene.mark_dirty()
        self._log(f"mode: {mode.title}")
        if mode.note:
            self._log(f"  note: {mode.note}")

    def _place_menu_button(self, host) -> None:
        """Put the MENU button in the corner his shots register best in.

        Asked for 2026-10-18: "it should be dynamically placed into the best
        corner sensitivity-wise".

        THE ONLY HONEST SOURCE IS REAL SHOTS, and the app already has them:
        `Mapping.sensitivity` holds the regions the learner built from his own
        trigger pulls, each backed by at least three that cleared the
        laser/noise boundary, each with the peak it measured. Everything
        cheaper has been tried here and was wrong -- the white calibration
        markers spread 8% where the laser spreads 2x, the self test's emitted
        dot is flat because a screen emits where a laser reflects, and the
        mapping's Jacobian is backwards (RULES C6, four times).

        They are stored in CAMERA space, because that is where the gain map
        divides, so they are mapped through to the screen here.

        CHOSEN WHEN A DRILL STARTS, not every tick: a button that migrated
        mid-round is a button he shoots at where it used to be. It dies with
        the camera pose, like the sensitivity it is derived from -- nothing
        extra is needed for that, because the sensitivity is cleared on a
        measured move and this is recomputed from it at the next switch.
        """
        button = getattr(host, "menu_button", None)
        if button is None:
            return
        points = []
        mapping = self.mapping
        if mapping is not None:
            for r in (getattr(mapping, "sensitivity", None) or []):
                if not isinstance(r, dict):
                    continue
                cam = r.get("camera") or ()
                peak = r.get("peak", 0.0)
                if len(cam) != 2 or not peak:
                    continue
                try:
                    sx, sy = mapping.to_screen(float(cam[0]), float(cam[1]))
                except Exception:
                    continue
                points.append((float(sx), float(sy), float(peak)))
        size = (0, 0)
        if self.target_window is not None:
            rect = self.target_window.rect()
            size = (rect.width(), rect.height())
        corner, measured, why = playarea.best_corner(points, size)
        if corner != button.corner or not self._menu_corner_said:
            # Said once per change, not per drill: the reason is worth having
            # in the log when a button turns up somewhere new, and 200ms of
            # repetition is what makes a log unreadable.
            self._log(("menu button: " if measured else "menu button (NOT MEASURED): ") + why)
            self._menu_corner_said = True
        button.corner = corner

    def _drain_menu_button(self) -> bool:
        """The MENU button inside a drill, hit twice: go to the menu.

        A SEPARATE DRAIN from `_drain_menu_choice`, which returns at its first
        line unless the MENU is the active mode -- and this request only ever
        arrives from the other nine. Draining both from one function is how
        the EXIT fault happened: an early return for one case stepping over
        another that shares the tick.
        """
        host = self.layers.get("modes") if self.layers else None
        if host is None or not hasattr(host, "take_menu_request"):
            return False
        if not host.take_menu_request():
            return False
        if self._grid_active or self._sgrid_active:
            # The same rule every other route obeys: nothing starts a mode
            # over the grid being shot. The request is consumed rather than
            # kept, so it cannot fire minutes later when the grid ends.
            self._log("menu button: dropped, the calibration grid is running")
            return False
        if self.target_window is None:
            return False
        self._log("menu button: confirmed on the second hit.")
        self._select_mode(MenuMode())
        self._announce_mode()
        return True

    def _announce_mode(self) -> None:
        """Speak the name of the mode just chosen with a key, if it still is."""
        host = self.layers.get("modes")
        mode = getattr(host, "mode", None) if host is not None else None
        line = MODE_LINES.get(getattr(mode, "name", ""))
        if line is None:
            return
        name = mode.name

        def still_current() -> bool:
            h = self.layers.get("modes")
            m = getattr(h, "mode", None) if h is not None else None
            return getattr(m, "name", None) == name

        self.voice.say(line, valid=still_current)

    def _mode_title(self) -> str:
        host = self.layers.get("modes")
        mode = getattr(host, "mode", None) if host is not None else None
        return mode.title.upper() if mode is not None else ""

    def _clear_holes(self, why: str) -> None:
        """THE ONE place bullet holes are cleared, so a clear is never a second
        mechanism and every one says why in the log."""
        self.layers["holes"].clear()
        self._log(f"  holes cleared: {why}")

    def _restart_mode(self, via: str = "the N key") -> None:
        """Restart the drill, or start the NEXT versus round if one is over.
        The N key, a shot at the ROUND box and a click on it all come here:
        two routes into a mode change is a defect this project already shipped."""
        host = self.layers.get("modes")
        if host is None or host.mode is None or self.target_window is None:
            return
        ctx = self.scene._context(time.perf_counter(), 0.0, self.target_window.rect())
        host.restart(ctx)
        self.session.reset()
        self._clear_holes(f"restart via {via}")
        self.scene.mark_dirty()

    def _on_target_closed(self) -> None:
        # Closing the window mid-calibration must not leave the grid running:
        # _grid_active suppresses the screen mask, so a stuck flag means the
        # detector watches the whole room until the app is restarted.
        if self._grid_active:
            self._grid_abort()
        # The brightness grid draws on this window and is aimed at it; with the
        # window gone it holds the screen-owner flag and relaxed thresholds for
        # a screen that no longer exists.
        self._sgrid_end("the target screen closed")
        self.target_window = None
        # Say so now, once, rather than at the first lost shot: with the
        # window gone every detection is ignored, and that must be audible.
        self._camera_sees_our_screen()
        self.btn_target.setText("Open target screen  (F11)")
        self._sync_layer_checkboxes()

    def _sync_layer_checkboxes(self) -> None:
        """Keys pressed on the target screen change layer state; mirror it here."""
        for name, box in (("camera", self.chk_pane), ("stats", self.chk_stats),
                          ("hits", self.chk_minimap),
                          ("help", self.chk_help), ("holes", self.chk_holes)):
            layer = self.scene.get(name)
            if layer is not None and box.isChecked() != layer.visible:
                box.blockSignals(True)
                box.setChecked(layer.visible)
                box.blockSignals(False)

    def _drain_menu_choice(self) -> "str | None":
        """Start the drill a shot (or an earlier key) chose on the menu.

        Runs on the HUD tick (`_refresh_target_diagnostics`, 200 ms), NOT from
        `Menu.shot`: that runs inside `ModeHost.shot`, and `set_mode` would stop
        the object still executing. The tick is the first point after the stack
        has unwound. 200 ms is below anything he can perceive between pulling the
        trigger and the screen changing, so no timer of its own is needed.

        Returns the name of the mode started, or None. None covers three cases
        that are told apart in the log, not here: the menu is not the active
        mode, nothing is pending, or the choice was refused (grid running, no
        target window). A refusal never consumes a choice that was not acted on
        unless it is stale or unsafe to keep (see below).

        THE HOLES ARE CLEARED BY `_select_mode`, DELIBERATELY. It already clears
        the holes layer, resets the session and marks the scene dirty on every
        switch, so the previous string's holes vanish with no code here. Do not
        add a second mechanism.
        """
        host = self.layers.get("modes") if self.layers else None
        mode = getattr(host, "mode", None) if host is not None else None
        if not isinstance(mode, MenuMode):
            return None             # a choice from a replaced menu is gone with it
        if self._grid_active or self._sgrid_active:
            # The keys obey the same rule (`_on_target_key`): nothing may start a
            # drill over the grid being shot. The menu is not drawn then, so a
            # shot that lands on an invisible tile must not become a choice that
            # fires when the grid ends -- and that applies to EXIT above all.
            dropped = mode.take_pending()
            quit_dropped, mode.quit_requested = mode.quit_requested, False
            if dropped is not None or quit_dropped:
                self._log(f"menu: {'EXIT' if quit_dropped else getattr(dropped, 'title', dropped)}"
                          f" dropped, the calibration grid is running")
            # A player-count request is dropped for the same reason: the menu
            # is not drawn while the grid is shot, so a shot on an invisible
            # box must not take effect when the grid ends.
            players_dropped = mode.take_pending_players()
            if players_dropped is not None:
                self._log(f"menu: player-count request ({players_dropped}) dropped, "
                          f"the calibration grid is running")
            return None
        # EXIT, SHOT OR CLICKED TWICE -- AND IT IS CHECKED BEFORE `pending`.
        #
        # It was checked after, and the second hit did nothing: EXIT sets
        # `quit_requested` and never sets `pending`, so the "nothing is
        # chosen, nothing to do" early return fired every time and the quit
        # was never reached. He found it in one try -- "shooting exit twice
        # doesn't work, clicking it twice doesn't work either" -- while six
        # checks passed, because every one of them asked the MENU whether it
        # had recorded the request and none asked the APP what it did with
        # it. A test of the producer that never runs the consumer (C14).
        #
        # It needs no target window: closing works whether or not one is up,
        # and `close()` is the clean shutdown that releases the camera.
        if mode.quit_requested:
            mode.quit_requested = False
            self._log("menu: EXIT confirmed on the second hit: closing the app.")
            self.close()
            return None
        # A PLAYER-COUNT REQUEST IS APPLIED BEFORE THE `pending` EARLY RETURN
        # BELOW -- that return is exactly how the EXIT bug got through -- and IT
        # DOES NOT RETURN: it falls through, so a shot on 2P and a shot on a
        # drill tile landing in the same 200 ms tick both take effect. It needs
        # no target window: it only sets a setting.
        requested = mode.take_pending_players()
        if requested is not None:
            self._apply_players(requested)
        if mode.pending is None:
            return None             # nothing chosen: not an error, nothing consumed
        if self.target_window is None:
            return None             # `_select_mode` would do nothing; keep the choice
        cls = mode.take_pending()
        if cls is None:
            return None
        self._log(f"menu: shot on the {cls.title} tile")
        self._select_mode(cls())
        self._announce_mode()       # the same pair `_on_target_key` runs for a key
        return cls.name

    def _apply_players(self, n: int) -> None:
        """Make a player-count request the app's setting: save it (debounced),
        mirror it to the scene so the underline moves, and say so in the log.
        It is consumed by `_select_mode` when the next drill is chosen."""
        n = normalise_players(n)
        changed = n != self.settings.players
        self.settings.players = n
        self.scene.players = n
        self._schedule_save()
        self.scene.mark_dirty()
        word = "one player" if n == 1 else "two players"
        self._log(f"menu: {word} selected" + ("" if changed else " (already)"))

    def _refresh_target_diagnostics(self) -> None:
        # Published on the HUD tick, so it tracks whatever is running
        # without every routine having to remember to announce itself.
        try:
            self._drain_menu_button()
            self._drain_menu_choice()
        except Exception as exc:      # said, never swallowed: a dead menu looks like a dead detector
            self._say_once("menu-drain-failed", f"menu: starting the chosen drill failed: {exc!r}")
        self._publish_measuring()
        self._apply_info_policy()
        bg = self.scene.get("background")
        if bg is not None:
            self.scene.dark_background = getattr(bg, "is_dark", True)
        self._sync_layer_checkboxes()
        self._audio_housekeeping()
        self._poll_audio()
        self._feed_round_health()
        self._check_grid_stall()
        if self.capture is None:
            self.scene.diagnostics = {"text": "camera not connected"}
            self._update_prompt()
            meter = self.layers.get("signal")
            if meter is not None:
                meter.report(0, 0, live=False, now=time.monotonic())
            self.scene.mark_dirty()
            return
        s = self.capture._stats
        self._check_camera_alive(s)
        self._check_screen_watch()
        self.scene.diagnostics = {
            "text": (
                f"fps {s.fps:5.2f}   detect {s.process_ms:4.2f} ms\n"
                f"frames with laser {s.detection_rate * 100:5.1f}%\n"
                f"frames/shot {s.mean_shot_frames:4.2f}   pulse ~{s.mean_pulse_ms:5.1f} ms\n"
                f"peak {s.peak:3d}   shape {s.elongation:4.1f}:1\n"
                f"dots at once {s.max_dets_in_frame}   rejected {s.rejected_runs}"
            )
        }
        self._update_signal_meter(s)
        self._update_prompt()
        self.scene.mark_dirty()

    def _feed_round_health(self) -> None:
        """Give the round's honesty mark one reading of what the app's own
        measurement is doing now (modes/efficiency.py names the conditions).
        Reads state only; called from the UI tick, never the shot path."""
        try:
            self.layers["modes"].note_measurement(
                time.perf_counter(), covered=self._cover_active,
                moved_count=self._moved_findings, margin=self._last_margin,
                bar=float(self.sp_minpeak.value()))
        except Exception as exc:              # said, never swallowed
            self._say_once("round-health-failed", f"  round health reading failed: {exc!r}")

    def _check_screen_watch(self) -> None:
        """Feed the passive screen watch one new sample, if there is one.

        Read-only: no control write, no dot, no photograph. The capture thread
        reduces a frame to one number ~4 times a second; this only judges it.
        """
        sw = self.screenwatch
        cap = self.capture
        if not sw.armed or cap is None:
            return
        # Moments that cannot be judged: a routine owns the screen (its dots,
        # covers and panels are ours), the camera is not on our screen, or it
        # has stalled (its own alert says so).
        if (self._screen_busy() or not self._camera_sees_our_screen()
                or self._camera_stalled):
            sw.pause()
            return
        at = cap.screen_level_at
        if at == self._screenwatch_last_at:
            return
        self._screenwatch_last_at = at
        mode = self.layers["modes"].mode
        signature = (type(mode).__name__ if mode is not None else "",
                     tuple(la.name for la in self.scene.layers if la.visible))
        text = sw.feed(at, cap.screen_level, signature)
        if text:
            self._log("  WARNING: screen watch: " + text)
            try:
                self._alert(
                    "THE CAMERA SEES THE SCREEN DIMMER THAN IT DID",
                    text + ". Check the camera has not moved, lost its cable, "
                           "or had its settings changed. If it persists, close "
                           "the app by closing its window and restart it.",
                    kind="warn", seconds=20.0)
            except Exception:                                # noqa: BLE001
                pass

    def _check_camera_alive(self, s) -> None:
        """Say so when frames stop arriving, instead of looking merely idle.

        A stalled camera and a quiet range produce the same screen: no hits, no
        sound, nothing moving. The difference matters enormously -- one is you
        not shooting, the other is the app being deaf -- and the app has the
        frame counter to tell them apart.
        """
        now = time.monotonic()
        last_n, last_t = getattr(self, "_frame_watch", (None, now))
        if s.frames != last_n:
            self._frame_watch = (s.frames, now)
            if getattr(self, "_camera_stalled", False):
                self._camera_stalled = False
                self._clear_alert()
                self._log("camera is delivering frames again")
            return
        if now - last_t < 3.0 or getattr(self, "_camera_stalled", False):
            return
        self._camera_stalled = True
        self.voice.say("v_camera_bad", valid=lambda: self._camera_stalled)
        self._log(f"WARNING: no camera frames for {now - last_t:.0f}s. The camera "
                  f"has stalled - unplug and replug it, or press F5 to reconnect.")
        self._alert(
            "THE CAMERA HAS STOPPED SENDING FRAMES",
            "Nothing has arrived for several seconds, so no shot can be "
            "detected however well you shoot.\n\n"
            "Press F5 in the Camera Lab window to reconnect. If that does not "
            "help, unplug the camera and plug it back in.")

    def _signal_note(self, text: str, kind: str = "info",
                     seconds: float = 6.0) -> None:
        """Say something on the signal meter for a few seconds."""
        self._note = (text, kind, time.monotonic() + seconds)
        if text.startswith("HOLD FIRE"):
            # Spoken too, from the one funnel every HOLD FIRE goes through.
            # True for as long as the note is on the meter.
            note = self._note
            self.voice.say("v_hold_fire",
                           valid=lambda: self._note is note
                           and time.monotonic() < note[2])

    def _update_signal_meter(self, s) -> None:
        """Everything needed to tell 'saw nothing' from 'saw it and refused it'.

        This is the panel that answers the question the app could never answer:
        a shot that does not appear on screen looks identical whether the camera
        saw nothing, saw something too faint, or saw it outside the area it is
        watching -- and those need completely different responses.
        """
        meter = self.layers.get("signal")
        if meter is None:
            return
        now_p = time.perf_counter()
        seen_ago = (now_p - s.last_detection_at) if s.last_detection_at else None
        shot_ago = (now_p - s.last_shot_at) if s.last_shot_at else None

        counts = [("seen", s.frames_with_detection), ("shots", s.shots)]
        if s.rejected_runs:
            counts.append(("too short", s.rejected_runs))
        if s.rejected_dim:
            counts.append(("too faint", s.rejected_dim))
        if s.rejected_off_screen:
            counts.append(("off-screen", s.rejected_off_screen))
        if s.rejected_amp_area:
            counts.append(("too small", s.rejected_amp_area))
        if s.rejected_halo:
            counts.append(("in a halo", s.rejected_halo))
        if self.listener.running:
            if self.listener.calibrated:
                counts.append(("heard", self.corroborator.heard))
            else:
                # Saying "heard 9" while the threshold is still a guess invites
                # the shooter to believe a number that means nothing yet.
                counts.append(("mic", f"learning {len(self.listener._learned)}"
                                      f"/{self.listener.LEARN_NEEDED}"))

        # Priority order, most actionable first. Each of these was a real
        # session that ended in "it is just not registering".
        note, kind = "", "info"
        held = getattr(self, "_note", None)
        threshold = self.sp_minpeak.value()
        if self._audio_note:
            note, kind = self._audio_note, "warn"
        elif held is not None and time.monotonic() < held[2]:
            note, kind = held[0], held[1]
        elif s.off_screen_peak >= threshold:
            note = (f"something bright (peak {s.off_screen_peak}) is being seen "
                    f"OUTSIDE the screen area - the camera may have moved")
            kind = "warn"
        elif s.dim_peak and s.dim_peak >= threshold * 0.5:
            note = (f"seen but too faint to count: peak {s.dim_peak} against a "
                    f"threshold of {threshold}. Shoot closer, or press X for a "
                    f"longer exposure.")
            kind = "warn"
        meter.report(s.frame_peak, threshold, live=True, seen_ago=seen_ago,
                     shot_ago=shot_ago, counts=counts, note=note, note_kind=kind,
                     now=time.monotonic())
        if self._grid_active or self._sgrid_active:
            # The meter hides while the grid runs, so the one number it had
            # that the calibration block lacks goes into the block instead.
            cal = self.layers.get("calibration")
            if cal is not None:
                ago = "" if seen_ago is None else f", last seen {seen_ago:.1f}s ago"
                cal.level = (f"signal {s.frame_peak} against a threshold of "
                             f"{threshold}{ago}")

    def _update_prompt(self) -> None:
        """One line saying what to do right now, on the screen being shot at.

        The user's complaint, verbatim: "I don't want to be in the situation
        when I shoot the targets and absolutely nothing happens and I have no
        idea if I should continue shooting at that target or move to the next
        one or stop shooting altogether." Everything the app knew about its own
        state lived in a console behind a fullscreen window, so the normal case
        -- fine, waiting for you -- looked exactly like the broken case.
        """
        band = self.layers.get("prompt")
        if band is None:
            return
        head, detail, kind, counter = self._prompt_state()
        if band.say(head, detail, kind, counter):
            self.scene.mark_dirty()

    def _prompt_state(self) -> tuple[str, str, str, str]:
        if self.capture is None:
            return ("CAMERA NOT CONNECTED", "Press F5 in the Camera Lab window.",
                    "warn", "")
        if self._grid_active:
            layer = self.layers["calibration"]
            n = len(self._grid_points)
            return ("SHOOT THE HIGHLIGHTED RING",
                    f"{layer.need} shots that agree. SPACE skips it, "
                    f"BACKSPACE redoes the last one.",
                    "go", f"point {min(layer.index + 1, n)} of {n}")
        if self._sgrid_active:
            layer = self.layers["calibration"]
            n = len(self._sgrid_points)
            return ("SHOOT THE HIGHLIGHTED SPOT",
                    f"{layer.need} shots that agree. This measures how bright "
                    f"your laser looks across the screen. SPACE skips it, "
                    f"BACKSPACE redoes the last one, U stops.",
                    "go", f"spot {min(layer.index + 1, n)} of {n}")
        # SOMETHING IS IN FRONT OF THIS SCREEN, so nothing the camera sees
        # can be a trigger pull. Said here for the case he CAN read it --
        # another window covering part of the panel. When this window is
        # lowered outright he sees none of it, which is why the fix is
        # silence and a log line rather than a message he cannot reach.
        if not self._camera_sees_our_screen():
            win_ = self.target_window
            by_ = getattr(win_, "cover_by", "") if win_ is not None else ""
            since_ = self._cover_since
            long_enough = (since_ is None or win_ is None
                           or time.perf_counter() - since_ >= self.COVER_BANNER_AFTER_S)
            if long_enough:
                # Said WITH ITS NAME when Windows could give one: "Something is in
                # front" withheld the one fact the app already had, and he could
                # do nothing with it. A cover shorter than COVER_BANNER_AFTER_S
                # shows no banner at all (the suppression itself is immediate).
                what = by_ if by_ else "Something"
                return ("SHOTS ARE NOT BEING COUNTED",
                        f"{what} is in front of this screen, so the camera "
                        "cannot see it. Bring it back to the front.", "warn", "")
        # WHAT IS ACTUALLY RUNNING, from the routine's own declaration.
        #
        # One of the three faults visible in the screenshot he sent on
        # 2026-10-15: the top band said "FINDING THE SCREEN - DO NOT FIRE"
        # while a finished verdict was on screen. The screen search was
        # long over. The band was reading `_calibrating`, which is written
        # in 12 places and read in 24 and means at least four different
        # things -- "the screen search is running", "a noise floor is in
        # flight", "pattern calibration is running", and "tell the shooter
        # not to fire". It was raised for a measurement and never came back
        # down in the way the band assumed.
        #
        # A routine on the Director carries one line of its own, written
        # where the routine is defined, so the band cannot describe the
        # wrong thing.
        # A VERDICT ON SCREEN OUTRANKS THE ROUTINE THAT PRODUCED IT.
        #
        # Third variant of the same fault, and he caught the first two. The
        # band now correctly names the running routine -- but the verdict is
        # published from INSIDE the self test, while that routine still
        # holds the slot, so the screenshot from the passing launch shows
        # "TESTING ITSELF - DO NOT FIRE" above a green READY panel saying
        # 12 of 12. Two instructions at once, and the louder one is wrong.
        #
        # The structural fix is to publish the verdict after the routine
        # RETURNS, on its completion callback, so the slot is free when the
        # answer appears. That is a chain change and this is not the hour
        # for it; this makes the band tell the truth meanwhile.
        board = self.scene.get("verdict") if self.scene is not None else None
        if board is not None and getattr(board, "visible", False):
            if getattr(board, "verdict", "") == "ready":
                return ("READY  -  " + (self._mode_title() or "SHOOT"), "0 shows or "
                        "hides this info and the key list. 1-6 start a drill.",
                        "go", f"{self.session.shots} shots")
            return ("NOT READY  -  read the panel", "The app has told you "
                    "what is wrong and what to do about it.", "warn", "")
        running = self.director.busy_label()
        if running:
            if self.teacher.active:
                return (running.upper(), "Three short quiet beeps: the app is "
                        "measuring how loud its speakers are to the "
                        "microphone. They are not shots.", "wait", "")
            return (running.upper(), "The app is measuring itself. This takes "
                    "a few seconds.", "wait", "")
        if self._calibrating:
            # Still ambiguous, but no longer silently so: say which of the
            # two it is, from whether the flash panel is actually up.
            if self._flash is not None:
                return ("FINDING THE SCREEN  -  DO NOT FIRE",
                        "The monitor is being flashed so the camera can "
                        "locate it.", "wait", "")
            return ("MEASURING  -  DO NOT FIRE",
                    "Measuring what the camera sees with nothing on screen, "
                    "to set the detection threshold.", "wait", "")
        # NO MAPPING OUTRANKS THE MODE. Two faults here, both live on
        # 2026-10-18, and together they are why he shot 42 rounds into an
        # app that told him nothing:
        #
        # 1. This was TWO branches with a mode between them. The first fired
        #    only when `self.homography` was also None -- and the screen
        #    search sets the homography on every launch, so after a rejected
        #    calibration it never fired. Control then reached the mode branch,
        #    which says the mode's name and "Shoot the targets.", in the `go`
        #    state, which also keeps the band HIDDEN while shooting. The app's
        #    single standing channel was telling him to carry on.
        # 2. The second branch said "ROUGH CALIBRATION ONLY -- holes will be
        #    placed from the screen outline". That fallback was deleted on
        #    2026-10-14 (`_map_to_screen` returns None and nothing else) and
        #    the text outlived it by four days. It promised holes that could
        #    not be drawn -- the mirror of RULES C4, advice the measurement
        #    contradicts.
        #
        # One branch, before the mode, keyed on the only thing that decides
        # whether a hole can exist: `self.mapping`.
        if self.mapping is None:
            return ("NOT CALIBRATED  -  NO HOLES WILL BE DRAWN",
                    "Your shots ARE being seen. The app does not know where "
                    "on this screen the camera is looking, so it will not "
                    "guess a hole. Press C to measure the screen (nothing to "
                    "shoot), or G to shoot the grid.", "warn", "")
        host = self.layers.get("modes")
        mode = getattr(host, "mode", None) if host is not None else None
        if mode is not None:
            return (mode.title.upper(), mode.status() or "Shoot the targets.",
                    "go", f"{self.session.hits}/{self.session.shots}")
        return ("READY  -  " + (self._mode_title() or "SHOOT"), "0 shows or hides this info "
                "and the key list. 1-6 start a drill.", "go",
                f"{self.session.shots} shots")

    # -- hit test -------------------------------------------------------------

    def _reset_counters(self) -> None:
        if self.capture is not None:
            self.capture.reset_stats()
        self.sp_fired.setValue(0)
        self.session.reset()
        self.layers["hits"].clear()
        self._clear_holes("counters reset")
        self.scene.mark_dirty()
        self._log("counters reset")

    def _on_listen_changed(self, *_args) -> None:
        self._schedule_save()
        if self.capture is not None:
            self._start_listening()
        else:
            self.listener.stop()

    # + and - change the volume of the gunshot AND the voice (one master gain).
    # Unshifted "=" is accepted because he is holding a pistol, not typing
    # carefully; the numpad's + and - arrive as Key_Plus / Key_Minus.
    VOLUME_UP_KEYS = (Qt.Key_Plus, Qt.Key_Equal)
    VOLUME_DOWN_KEYS = (Qt.Key_Minus, Qt.Key_Underscore)
    # Percent rungs, about 3 dB (x1.4) apart: the ear hears ratios, so a linear
    # +5 is a doubling at the quiet end and nothing at the loud one. 5 is the
    # default and is on the ladder. The floor is 1, not 0: a key that can mute
    # the app by accident recreates the three days of "the gunshot is silent".
    VOLUME_LADDER = (1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80, 100)

    @classmethod
    def next_volume(cls, current_pct: int, direction: int) -> int:
        """The next rung up (+1) or down (-1) from `current_pct`, clamped."""
        ladder = cls.VOLUME_LADDER
        if direction > 0:
            for v in ladder:
                if v > current_pct:
                    return v
            return ladder[-1]
        for v in reversed(ladder):
            if v < current_pct:
                return v
        return ladder[0]

    def _step_volume(self, direction: int) -> None:
        """One press of + or -: move the slider (which sets the gain and saves
        the setting through the existing path) and show the readout."""
        before = self.sld_volume.value()
        target = self.next_volume(before, direction)
        if target != before:
            self.sld_volume.setValue(target)       # -> _on_volume_changed
        end = ""
        if target == self.VOLUME_LADDER[-1] and direction > 0:
            end = "  (maximum)"
        elif target == self.VOLUME_LADDER[0] and direction < 0:
            end = "  (minimum)"
        self._log(f"volume {before}% -> {self.sld_volume.value()}% "
                  f"(gunshot and voice; the microphone's flat-mute fallback is now "
                  f"{self._gunshot_mute_s(self.cmb_sound.currentText() or 'synth shot'):.2f}s, "
                  f"used only until the coupling is measured)")
        ro = self.layers.get("volume")
        if ro is not None and self._info_shootable():
            ro.show_volume(f"VOLUME {self.sld_volume.value()}%{end}", time.perf_counter())
            self.scene.mark_dirty()

    def _on_volume_changed(self, value: int) -> None:
        self.lbl_volume.setText(f"{value}%")
        self.mixer.set_master_gain(value / 100.0)
        self._schedule_save()

    def _refresh_hitrate(self) -> None:
        detected = self.capture._stats.shots if self.capture is not None else 0
        fired = self.sp_fired.value()
        if fired <= 0:
            self.lbl_hitrate.setText(f"detected {detected}")
            self.lbl_hitrate.setStyleSheet("color:#bbb;")
            return
        rate = detected / fired * 100.0
        colour = "#5fd35f" if rate >= 95 else ("#d0c030" if rate >= 80 else "#c0c040")
        self.lbl_hitrate.setText(f"{detected}/{fired} = {rate:.0f}%")
        self.lbl_hitrate.setStyleSheet(f"color:{colour};")

    # -- misc -----------------------------------------------------------------

    def _scan_formats(self) -> None:
        if self.capture is not None:
            self._log("disconnecting for the format scan - the camera can only be open once")
            self._disconnect()
        cfg = self._current_config()
        ModeScanDialog(cfg.index, cfg.backend, self).exec()

    def _toggle_record(self) -> None:
        if self.capture is None:
            self._log("CANNOT RECORD: no camera connected. Press Connect (F5) first.")
            return
        if self.capture._stats.recording:
            self.capture.stop_recording()
        else:
            self.capture.start_recording(RECORDINGS)

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"{stamp}  {msg}"
        print(line, flush=True)                  # so a backgrounded run is inspectable
        # ...and to a file, always. Launched with pythonw there is no console,
        # and the log pane scrolls: diagnosing a failed calibration meant
        # screenshotting the window and reading the last ten lines, repeatedly,
        # for an entire evening. A file costs nothing and holds the whole run.
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        self.log_view.appendPlainText(line)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def closeEvent(self, ev) -> None:
        """Shut down, and TIME each step.

        A force-kill wedges this camera -- measured at 40.3s to open and 0
        frames afterwards, against 0.44s and 53.6 fps after a clean close
        -- so a shutdown that overruns the stop script's patience is not a
        cosmetic problem, it is the thing that breaks the next launch.

        One gate launch in five overran, and the app's log stopped at the
        same line on the launches that closed cleanly AND the one that did
        not, so the log said nothing about where the time went. Hence the
        timings: the next overrun names its own cause instead of being
        diagnosed by elimination.
        """
        t0 = time.perf_counter()

        # STAND EVERY ROUTINE DOWN BEFORE TEARING ANYTHING DOWN.
        #
        # This whole app is driven by about sixty QTimer callbacks. When
        # the window closes mid-routine, those are still queued, and the
        # next one to fire reaches into a scene whose target window has
        # just been destroyed. Qt6 is fail-fast about that: it does not
        # raise, it takes the process down -- no traceback, nothing on
        # stderr, the log simply stopping mid-shutdown.
        #
        # Which is indistinguishable from a force-kill to the camera, and
        # a force-kill is measured to wedge this one: 40.3s to open and 0
        # frames afterwards, against 0.44s and 53.6 fps after a clean
        # close. Every gate launch today was closed in the middle of the
        # self test or the arcade mode test, because that is where the
        # chain ends -- so this was being provoked on nearly every run,
        # and the replugs it cost were blamed on the camera.
        #
        # ONE CANCEL for everything on the Director. Its timers DO have
        # handles, so they are stopped rather than neutralised, and the
        # running routine is resumed with Cancelled at its current yield --
        # which runs its `finally` blocks, so the dot comes down, the
        # levels go back and the cover is removed on the way out.
        #
        # What this replaces is the list below, and the comment that used
        # to be here: "the timers themselves cannot be cancelled --
        # QTimer.singleShot keeps no handle -- which is exactly why the
        # flags have to be the defence." They were the defence, and the
        # list omitted `_check_active`, so the two-marker check was never
        # stood down at all. A guard that names flags by hand will be
        # incomplete (RULES E2) -- including this one, written to be
        # complete, by someone looking straight at the problem.
        self._splash_down("the app is closing")
        self.director.close("the app is closing")

        # THE SHOOTING CALIBRATIONS' RELAXED THRESHOLDS ARE SETTINGS, so they
        # must be put back BEFORE `_save_settings` below, or a close in the
        # middle of a grid writes a detection bar of floor+3 to
        # settings.local.json and the next launch starts permissive. The
        # brightness grid goes through its one exit; the shot grid has no
        # exit of its own, so its levels are restored here as well (the
        # flag loop below only clears the flag).
        self._sgrid_end("the app is closing")
        self._restore_levels()

        # Still by hand for the routines not yet migrated. This list
        # shrinks as they move; it must never grow.
        for flag in ("_modetest_active",
                     "_pat_active", "_grid_active", "_calibrating",
                     "_floor_in_flight", "_align_pending"):
            if hasattr(self, flag):
                setattr(self, flag, False)
        for timer in ("_audio_timer", "_voice_timer", "_room_timer", "_still_guard",
                      "_room_guard", "_hud_timer", "_align_timer"):
            t = getattr(self, timer, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
        QApplication.processEvents()

        def mark(step: str) -> float:
            nonlocal t0
            dt = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            if dt > 250.0:
                self._log(f"shutdown: {step} took {dt:.0f}ms")
            return dt

        try:
            self._save_settings()
        except Exception as exc:                     # noqa: BLE001
            self._log(f"shutdown: could not save settings ({exc!r})")
        mark("saving settings")
        if self.target_window is not None:
            self.target_window.close()
        mark("closing the target screen")
        self._disconnect()
        mark("releasing the camera")
        # Audio teardown is the one step that can block on something this
        # app does not control: PortAudio closes a callback stream by
        # waiting for its callback to return, and the endpoint here is a
        # monitor over DisplayPort which this app has just finished
        # flashing black and white. It must never hold the process.
        try:
            self.mixer.close()
        except Exception as exc:                     # noqa: BLE001
            self._log(f"shutdown: the mixer would not close ({exc!r})")
        mark("closing audio")
        self._log("shutdown complete")
        super().closeEvent(ev)

        # AND ACTUALLY LEAVE.
        #
        # `_disconnect` deliberately does not wait for the capture thread:
        # waiting three seconds on a thread blocked inside cap.read() is an
        # older bug, already recorded here. But a QThread that is still
        # alive keeps the PROCESS alive, so the window closed, the
        # shutdown logged "complete", and nothing exited. stop_lab waited
        # its fifteen seconds and force-killed -- and a force-kill is
        # measured to wedge this camera: 40.3s to open and 0 frames after,
        # against 0.44s and 53.6 fps after a clean close.
        #
        # So the camera stalling produced: blocked thread -> process
        # cannot exit -> force-kill -> camera wedged -> next launch stalls.
        # A loop that feeds itself, and every link in it was something
        # this project had already fixed once in isolation.
        #
        # Everything that must be flushed has been flushed by this line:
        # settings saved, recording closed, camera released as far as it
        # can be, audio shut. There is nothing left for a graceful
        # interpreter shutdown to do except wait for a thread that is
        # never coming back.
        try:
            app = QApplication.instance()
            if app is not None:
                app.quit()
        except Exception:
            pass
        # ONLY when this process exists to be the app.
        #
        # `os._exit` takes the whole process down, and the cold-start
        # harness builds a real CameraLab INSIDE the test process. So the
        # first launch's shutdown was killing the test runner -- with
        # status 0, which every caller read as a pass. Measured
        # 2026-10-15 22:09: the harness printed section 1, began section
        # 2, and vanished. It has been exercising one launch out of seven
        # and reporting success, and I quoted "8 of 8 READY" from it.
        #
        # The fourth unfailable check on this project (RULES C1), and the
        # most expensive, because it is the instrument the others rely on.
        if self.hard_exit_on_close:
            QTimer.singleShot(1200, self._leave_now)
        else:
            self._log("shutdown complete (leaving the process alive: this "
                      "CameraLab does not own it)")

    # Set False by anything that builds a CameraLab inside its own process.
    # A class attribute, so the harness sets it before constructing and
    # there is no window in which a shutdown could race it.
    hard_exit_on_close = True

    @staticmethod
    def _leave_now() -> None:
        """Exit even if a thread is wedged inside the camera driver.

        Deliberately os._exit: a normal exit joins non-daemon threads, and
        the one in question is blocked in a driver call that may never
        return. Reached only after the full shutdown has run, so this
        skips nothing -- it is the difference between leaving cleanly a
        second late and being force-killed fifteen seconds late.
        """
        os._exit(0)


def _install_owner_hooks(cls) -> None:
    """Make raising any SCREEN_OWNERS flag take the startup splash down.

    Derived from the tuple itself, so an owner added there is covered
    without anyone remembering (RULES E2). Storage keeps the plain
    attribute's semantics: unset reads False, assignment stores.
    """
    for flag, _name in cls.SCREEN_OWNERS:
        def _get(self, _k="_owner_" + flag):
            return self.__dict__.get(_k, False)

        def _set(self, value, _k="_owner_" + flag):
            if value:
                self._splash_down("a measurement owns the screen")
            self.__dict__[_k] = value

        setattr(cls, flag, property(_get, _set))


_install_owner_hooks(CameraLab)


def main() -> int:
    """Flags exist so a diagnostic session needs no clicking:

      --record   start recording as soon as calibration finishes
      --target   open the fullscreen target screen once it is ready
      --aim      go straight to the camera-placement view: bright picture,
                 monitor outlined, nothing else running
      --trust-cal  do not calibrate at all: use the saved mapping as it
                 stands and go straight to the self test. Valid only while
                 the camera has not moved, so it is a flag and never a
                 setting. Turns a seventy-second iteration into fifteen.
      --modetest  also walk every arcade mode after the verdict (slow)
    """
    args = set(sys.argv[1:])
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # THE SPLASH GOES UP BEFORE THE WINDOW IS BUILT, and well before the
    # camera is opened. Its decode runs on a worker thread so it overlaps
    # the construction below. It comes down in `CameraLab._splash_down`.
    splash = Splash()
    splash.start(app.primaryScreen())
    app.processEvents()
    win = CameraLab()
    win.attach_splash(splash)
    win.auto_record = "--record" in args
    win.auto_target = "--target" in args
    win.auto_aim = "--aim" in args
    win.auto_patcal = "--patcal" in args
    # Calibrate, then immediately test the result by flashing red dots at
    # known positions. One command, no trigger pulls, and it answers the only
    # question that matters: does a hit land where it happened?
    # ON BY DEFAULT, from 2026-10-20. His instruction, 23:10:
    #
    #   "We will have to do a self-test on every launch because the light in
    #    the room is not guaranteed to be the same over the hours."
    #
    # He is right, and the evening proved it twice over. It used to be opt-in,
    # which meant `run_live.ps1` passed it and the `Play.cmd` I shipped did
    # not -- so the packaged app never once checked whether it could see its
    # own shots before he started firing. He noticed the red dots were missing
    # and it took me twenty minutes to work out why.
    #
    # Everything the check depends on drifts: the room light over the hours,
    # the exposure the camera settles at, the noise floor, and the detection
    # bar derived from it. A health check that runs only when someone
    # remembers a flag is not a health check (RULES C7: a rule that is never
    # executed is not a rule).
    #
    # `--no-selftest` remains for the rare case of wanting the app up without
    # spending twelve seconds on dots -- an explicit opt-OUT, which is the
    # right direction for something that protects the user.
    win.auto_selftest = "--no-selftest" not in args
    # Skip calibration entirely and go straight to the tests. For running
    # one change after another while nobody is touching the camera.
    win.trust_calibration = "--trust-cal" in args
    # The arcade walk-through, for development. It runs after READY and
    # costs about twenty seconds, so it is opt-in.
    win._selftest_modes = "--modetest" in args
    # THE INTROSPECTION SERVER, off unless asked for. `port_from` returns 0
    # when neither `--introspect[=PORT]` nor LSA_INTROSPECT is present, and
    # nothing is created in that case -- no thread, no socket, no tap -- so
    # the app behaves identically without it. Read-only except POST /key, and
    # it cannot write a camera control: see lsa/introspect/readonly.py.
    port = introspection.port_from(args, os.environ)
    if port:
        win.introspect = introspection.install(win, port=port, log=win._log)
    if win.auto_selftest:
        win.auto_patcal = True
    for a in args:
        if a.startswith("--patcal-repeat="):
            try:
                win.patcal_repeats = max(1, int(a.split("=", 1)[1]))
                win.auto_patcal = True
            except ValueError:
                pass
    # Pattern calibration photographs the TARGET SCREEN's own markers, so it
    # cannot run without that screen. Asking for it and getting "pattern
    # calibration needs the camera and the target screen" after a full
    # forty-second setup is a wasted run and a wasted camera open -- which on
    # this camera is the expensive part. A flag that implies another should say
    # so itself.
    if win.auto_patcal:
        win.auto_target = True
    # ONE WINDOW FOR THE SHOOTER, not two.
    #
    # "This two-window launch doesn't make any sense for me. Do you need
    #  to have those two windows up at all times? Can you just start one
    #  full screen window and use just that one window?"
    #
    # No, and the second one has done real harm. The operator console is a
    # DEVELOPMENT tool: it sits on the same monitor the camera watches,
    # and its own warning banner at red excess 80 produced phantom shots
    # for an evening. Two top-level windows are also why shutdown was
    # fragile -- Windows names the wrong one as the process's main window,
    # so WM_CLOSE went to the fullscreen target screen, which closes only
    # itself, while the console that owns closeEvent never heard it.
    #
    # With --target the console starts MINIMISED. It is still there, one
    # click away in the taskbar, and every control and the whole log stay
    # reachable -- but the shooter sees one fullscreen window, which is
    # what the app is.
    if win.auto_target:
        win.showMinimized()
    else:
        win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
