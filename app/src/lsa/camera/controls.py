"""UVC camera control definitions, probing, and presets.

Everything about camera controls on Windows is folklore, because OpenCV passes
values straight through to two backends that interpret them differently and
neither documents its units. Rather than pretend we know, this module defines
generous ranges, always reads the value back after writing it, and surfaces
both numbers in the UI. What the camera actually honours is then an observation
rather than an assumption.

What we believe about exposure, to be confirmed by the lab:

  DSHOW  CAP_PROP_EXPOSURE      log2(seconds).  -6 = 1/64s, -10 = ~1/1000s
         CAP_PROP_AUTO_EXPOSURE 0.25 = manual, 0.75 = auto   (DirectShow idiom)
  MSMF   CAP_PROP_EXPOSURE      also log2(seconds) in most builds
         CAP_PROP_AUTO_EXPOSURE 0 = auto, 1 = manual         (inverted vs DSHOW!)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2

# CAP_PROP_SETTINGS opens the driver's own property sheet on DirectShow. It is
# the reliable escape hatch when OpenCV's pass-through refuses to stick.
PROP_SETTINGS = 37


@dataclass(frozen=True)
class ControlSpec:
    key: str
    prop: int
    label: str
    lo: float
    hi: float
    step: float = 1.0
    decimals: int = 0
    hint: str = ""


CONTROLS: tuple[ControlSpec, ...] = (
    ControlSpec("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, "Auto exposure", 0.0, 3.0, 0.25, 2,
                "DSHOW: 0.25=manual 0.75=auto. MSMF: 1=manual 0=auto. Set this FIRST."),
    ControlSpec("exposure", cv2.CAP_PROP_EXPOSURE, "Exposure (log2 sec)", -13.0, 0.0, 1.0, 0,
                "Lower = darker & shorter. -6 is 1/64s, -10 is ~1/1000s."),
    ControlSpec("gain", cv2.CAP_PROP_GAIN, "Gain", 0.0, 255.0, 1.0, 0,
                "Keep low. Gain amplifies sensor noise along with the laser."),
    ControlSpec("brightness", cv2.CAP_PROP_BRIGHTNESS, "Brightness", 0.0, 255.0, 1.0, 0, ""),
    ControlSpec("contrast", cv2.CAP_PROP_CONTRAST, "Contrast", 0.0, 255.0, 1.0, 0, ""),
    ControlSpec("saturation", cv2.CAP_PROP_SATURATION, "Saturation", 0.0, 255.0, 1.0, 0,
                "Higher helps the red-chroma detector separate laser from grey."),
    ControlSpec("sharpness", cv2.CAP_PROP_SHARPNESS, "Sharpness", 0.0, 255.0, 1.0, 0,
                "Low is better: sharpening creates ringing around a saturated dot."),
    ControlSpec("gamma", cv2.CAP_PROP_GAMMA, "Gamma", 0.0, 500.0, 1.0, 0, ""),
    ControlSpec("backlight", cv2.CAP_PROP_BACKLIGHT, "Backlight comp.", 0.0, 5.0, 1.0, 0,
                "Set to 0. Backlight compensation fights our deliberate underexposure."),
    ControlSpec("auto_wb", cv2.CAP_PROP_AUTO_WB, "Auto white balance", 0.0, 1.0, 1.0, 0,
                "Turn OFF. Auto WB will drift the red channel shot to shot."),
    ControlSpec("wb_temperature", cv2.CAP_PROP_WB_TEMPERATURE, "WB temperature (K)", 2000.0, 10000.0, 100.0, 0, ""),
    ControlSpec("autofocus", cv2.CAP_PROP_AUTOFOCUS, "Autofocus", 0.0, 1.0, 1.0, 0,
                "Turn OFF, then set focus manually. Hunting autofocus blurs the dot."),
    ControlSpec("focus", cv2.CAP_PROP_FOCUS, "Focus", 0.0, 255.0, 5.0, 0, ""),
    # Measured on the Razer Kiyo Pro over MSMF: writes are accepted and the
    # readback stays pinned at 100.0 whatever you set. Its field of view
    # (103/90/80 degrees) is a vendor setting, not a UVC one, and is reachable
    # from Windows Settings > Bluetooth & devices > Cameras -- where it does
    # work. Say so rather than offering a slider that does nothing.
    ControlSpec("zoom", cv2.CAP_PROP_ZOOM, "Zoom", 0.0, 500.0, 5.0, 0,
                "Often ignored by the driver. If nothing moves, set it in "
                "Windows Settings > Bluetooth & devices > Cameras."),
)

CONTROLS_BY_KEY = {c.key: c for c in CONTROLS}


@dataclass
class ProbeResult:
    key: str
    initial: float
    supported: bool
    writable: bool
    readback: float | None = None
    note: str = ""


def read_all(cap: cv2.VideoCapture) -> dict[str, float]:
    return {c.key: float(cap.get(c.prop)) for c in CONTROLS}


def apply(cap: cv2.VideoCapture, key: str, value: float,
          readback: bool = True) -> tuple[bool, float]:
    """Write a control. With `readback`, read it straight back as well.

    The readback is a SECOND UVC transaction on a device whose control
    interface is the fragile part. It is worth paying for when a human is
    watching a value they just typed -- these notes record that this camera
    accepts exposure writes and lies about them, so the UI shows both -- and
    it is not worth paying for when the app is applying a preset it already
    knows the contents of. `readback=False` halves the traffic.
    """
    spec = CONTROLS_BY_KEY[key]
    cap.set(spec.prop, float(value))
    if not readback:
        return True, float(value)
    back = float(cap.get(spec.prop))
    return (abs(back - float(value)) < max(1e-3, abs(value) * 0.02)), back


def probe(cap: cv2.VideoCapture) -> list[ProbeResult]:
    """Find out which controls this camera + backend combination honours.

    Intrusive by necessity: for each control we nudge the value, read it back,
    then restore the original. Do not run this mid-session while shooting.
    """
    results: list[ProbeResult] = []
    for spec in CONTROLS:
        initial = float(cap.get(spec.prop))
        # OpenCV returns 0 for props the backend does not implement, but 0 is
        # also a legitimate value, so 0 alone proves nothing either way.
        supported = initial != 0.0
        target = initial - spec.step if initial - spec.step >= spec.lo else initial + spec.step
        cap.set(spec.prop, target)
        moved = float(cap.get(spec.prop))
        writable = abs(moved - initial) > 1e-6
        cap.set(spec.prop, initial)
        restored = float(cap.get(spec.prop))
        note = ""
        if writable and abs(restored - initial) > 1e-6:
            note = f"did not restore cleanly (now {restored:g})"
        if not supported and writable:
            supported = True
        results.append(ProbeResult(spec.key, initial, supported, writable, moved, note))
    return results


def open_driver_dialog(cap: cv2.VideoCapture) -> bool:
    """Open the camera driver's own property sheet (DirectShow only).

    Logitech's dialog can set things OpenCV's pass-through sometimes cannot.
    """
    try:
        cap.set(PROP_SETTINGS, 1)
        return True
    except Exception:
        return False


# --- Presets -----------------------------------------------------------------
#
# Two profiles, because the two jobs want opposite things from the sensor:
# finding the screen needs to SEE the screen; finding the laser needs to be
# blind to everything except the laser.

@dataclass
class Preset:
    name: str
    description: str
    values: dict[str, float] = field(default_factory=dict)


def laser_hunt(backend: str, exposure: float | None = -6.0) -> Preset:
    """The shooting profile: stable colour, manual everything, underexposed.

    `exposure` is only a starting point. The right value depends on the room and
    the camera, and getting it wrong costs every shot -- a hardcoded -8 measured
    206 peak brightness on this C910 where -6 measured 255, and the laser stopped
    registering. CaptureThread.start_exposure_search() picks it by measurement;
    pass exposure=None to leave it alone for that.

    Measured on this hardware: saturation, sharpness and backlight make no
    detectable difference to the image. They are set anyway for repeatability --
    a profile that drifts between sessions is worse than one that is slightly
    redundant.
    """
    manual_ae = 0.25 if backend == "DSHOW" else 1.0
    values = {
        "auto_exposure": manual_ae,
        "gain": 0.0,
        "backlight": 0.0,
        "auto_wb": 0.0,
        "autofocus": 0.0,
        # Saturation DOWN, not up. The instinct is to boost it so the red dot
        # stands out, and saturation=200 was in here for exactly that reason --
        # but both cameras clamp it to their default (128 on the Kiyo Pro, 32 on
        # the C910), so it never did anything, and raising it would have been
        # wrong anyway: it amplifies the colour noise the detector has to beat.
        # Lowering it measurably helps. On the Kiyo Pro at exposure -9, chroma
        # p99 fell 16 -> 6 and chroma max 55 -> 20 going from 128 to 64, while a
        # 650nm laser dot is saturated enough to survive it easily.
        "saturation": 64.0,
    }
    if exposure is not None:
        values["exposure"] = exposure
    return Preset(
        "Laser hunt",
        "Underexposed and colour-stable: the screen fades to a ghost, the 650nm "
        "dot still saturates. This is the shooting profile.",
        values,
    )


def calibration_view(backend: str, exposure: float | None = None,
                     gain: float | None = None) -> Preset:
    """See the monitor. The opposite job to shooting, so the opposite settings.

    Deliberately minimal, and in particular it does NOT touch gain. Forcing gain
    to 0 — which the shooting profile does, correctly, to keep noise down — took
    frame maximum from 255 to 7 on this camera and left calibration photographing
    a nearly black room. The flash difference measured 0.4 against the 77.5 the
    same method gets with gain left alone.

    PREFER A DARK, FIXED EXPOSURE. Auto exposure actively fights this method:
    the search photographs the monitor white then black and subtracts, and auto
    exposure compensates for exactly that change, so the two frames come out
    similar and the difference collapses. It was the default here because it
    worked in a dark room; in a lit one it does not. Measured on the same scene,
    same camera, minutes apart:

    | exposure | flash difference | confidence |
    |---|---|---|
    | auto | 14.9 | 0.72 |
    | -9 | 43.2 | 1.00 |
    | **-10** | **44.3** | **1.00** |
    | -11 | 43.9 | 0.99 |

    Dark wins because the ROOM goes black while the lit screen does not, so the
    monitor is the only thing that changes. exposure=None still selects auto, as
    a fallback for a camera that ignores manual exposure.
    """
    values: dict[str, float] = {"autofocus": 0.0}
    if exposure is None:
        values["auto_exposure"] = 0.75 if backend == "DSHOW" else 0.0
    else:
        values["auto_exposure"] = 0.25 if backend == "DSHOW" else 1.0
        values["exposure"] = exposure
    # AND IT MUST RESTORE GAIN, not merely refrain from pinning it.
    #
    # The paragraph above says this preset "does NOT touch gain", and that
    # was right when the only thing using it was the screen search, which
    # runs before the shooting profile has pinned gain to 0. It is wrong
    # now: the two-marker mapping check runs AFTER, so it photographed the
    # monitor with the sensor turned down for shooting, and "left alone"
    # meant "left pinned".
    #
    # Measured in simulation, 2026-10-15 22:36, same marker and position:
    #
    #     gain free    lit max 203   diff peak 201   found, strength 200
    #     gain pinned  lit max   5   diff peak   3   NOT FOUND
    #
    # `find_marker` needs a peak of 12, so the check reported both markers
    # invisible while correctly predicting where they were to within a
    # pixel. It has been working on his bench at 0.5px only because that
    # camera's gain-0 is less extreme than this; the same code on the 75"
    # TV or the projector is a coin toss. CLAUDE.md already records the
    # live version of this failure -- frame maximum 255 -> 7, flash
    # difference 0.4 against 77.5 -- which is the same mistake in the same
    # preset, found once and fixed only for the caller in front of me.
    #
    # `gain` is the value the camera reported at OPEN, before this app
    # wrote anything, which is the honest meaning of "left alone". None
    # keeps the old behaviour for a camera that did not report one.
    if gain is not None:
        values["gain"] = float(gain)
    return Preset(
        "Calibration view",
        "Camera set to see the monitor so its four corners can be found. "
        "Switch back to Laser hunt before shooting.",
        values,
    )
