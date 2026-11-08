"""Persisted application settings.

The app should remember what you told it last time. Re-picking a camera, a
backend and a resolution on every launch is friction that adds up fast when
you are iterating on detector tuning.

Stored as JSON next to the project (gitignored, since it is per-machine). The
loader is deliberately forgiving: unknown keys are ignored and missing ones fall
back to defaults, so an older settings file never stops the app from starting.

Cameras are remembered by NAME first and index second. USB enumeration order
changes when you plug something in, so an index alone will eventually point at
the wrong device — or at a virtual one that opens successfully and then delivers
nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from lsa.atomic import write_json_atomic
from lsa.bests import normalise_bests


PLAYER_COUNTS = (1, 2)


def normalise_players(value: Any) -> int:
    """1 or 2, whatever was stored: a hand-edited or corrupt settings file must
    never leave the menu with no box selected."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 1
    return n if (n in PLAYER_COUNTS and not isinstance(value, bool)) else 1


# THE MAGAZINE: how many registered shots end a round, everywhere (single player
# and each half of two-player), and how many bullets the screen draws. ONE number,
# owned HERE and nowhere else -- an earlier round length (`versus.ROUND_SHOTS`, 20)
# and a magazine drawn on the screen would be two copies of one fact, and the
# bullets must always mean exactly what is left. He will expose it as a setting
# later ("the user will be able to specify the exact mag size"); today it is read
# from `settings.local.json` and defaults here when the key is absent.
DEFAULT_MAGAZINE = 17      # his instruction, 2026-10-21 12:58, for every NEW
                           # deployment. It went 17 -> 19 on 2026-10-20 and back
                           # to 17 here; both numbers were his, and this one is
                           # what a fresh install should start with. The count is
                           # his call and nothing is derived from it -- the strip
                           # shrinks to fit rather than overflowing, and the round
                           # ends when the magazine does, whatever it holds.
                           #
                           # It changes NOTHING on an install that has already
                           # run: `settings.local.json` carries an explicit
                           # magazine once the app has saved, and a saved value
                           # beats a code default (the min_frames lesson). His
                           # own config says 17 already, so nothing he shoots
                           # with moves.
MAGAZINE_MIN = 1
MAGAZINE_MAX = 50          # a rifle magazine; beyond it the strip is unreadable

# -- how fast a drill's targets move -----------------------------------------
# Asked for 2026-11-07: Page Up / Page Down, remembered separately per drill.
# A multiplier on each drill's OWN speed curve, never an absolute speed: the
# curves were tuned against his hardware and a shared absolute would flatten
# the difference between a bouncing ball and an advancing figure.
#
# The range lives HERE, with the magazine's, for the reason stated beside the
# magazine: settings.py owns the default and the range, and nothing in modes/
# keeps a number of its own. The FLOOR is not zero -- a drill at speed zero
# has targets that never move, which is indistinguishable from a frozen app
# and is the same trap as a volume key that can reach mute.
# HOW LATE A SHOT'S TIMESTAMP IS, in seconds, beyond what the app can see.
#
# `ShotEvent.t` is stamped when `cap.read()` returns. The display latency and
# the camera's own pipeline both happened BEFORE that, to a ball the shooter
# had already aimed at, so a moving target's hit test must rewind by this much
# again or the back of the target is dead (`Mode.shot_age`).
#
# The default is a typical webcam-and-LCD pipeline and is REPLACED by the first
# measurement on this machine. The ceiling is deliberately well under
# `Mode.MAX_REWIND_S`: beyond ~150ms this is not a pipeline latency any more,
# it is a broken clock, and the hit test must not follow it there.
PIPELINE_LAG_DEFAULT = 0.060
PIPELINE_LAG_MIN = 0.0
PIPELINE_LAG_MAX = 0.150


def normalise_pipeline_lag(value: object) -> float:
    """Clamp to a believable pipeline latency. Anything unreadable -> default."""
    try:
        v = float(value)      # type: ignore[arg-type]
    except (TypeError, ValueError):
        return PIPELINE_LAG_DEFAULT
    if v != v:                # NaN
        return PIPELINE_LAG_DEFAULT
    return max(PIPELINE_LAG_MIN, min(PIPELINE_LAG_MAX, v))


DEFAULT_SPEED = 1.0
SPEED_MIN = 0.25
SPEED_MAX = 3.0


def normalise_speed(value: Any) -> float:
    """A usable speed multiplier, whatever was stored: clamped into
    [SPEED_MIN, SPEED_MAX]; anything that is not a real number (missing, text,
    a bool, NaN, an infinity) is the default. A hand-edited or corrupt value
    costs that one preference, never the launch -- the same contract the
    magazine and the bests are held to."""
    if isinstance(value, bool):
        return DEFAULT_SPEED
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_SPEED
    if f != f or f in (float("inf"), float("-inf")):
        return DEFAULT_SPEED
    return max(SPEED_MIN, min(SPEED_MAX, f))


def normalise_mode_speed(raw: Any) -> "dict[str, float]":
    """The saved per-drill speeds, with every entry validated. Anything that is
    not a mapping becomes an empty one; a drill whose value will not survive
    `normalise_speed` is dropped rather than defaulted, so a drill that has
    never been changed stays absent instead of being written at 1.0."""
    if not isinstance(raw, dict):
        return {}
    out: "dict[str, float]" = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        speed = normalise_speed(value)
        if speed != DEFAULT_SPEED or float(value) == DEFAULT_SPEED:
            out[name] = speed
    return out


def normalise_magazine(value: Any) -> int:
    """A usable magazine size, whatever was stored: clamped into
    [MAGAZINE_MIN, MAGAZINE_MAX]; anything that is not a whole number (missing,
    text, a bool, NaN) is the default, never zero (a zero magazine would end
    every round before it began)."""
    if isinstance(value, bool):
        return DEFAULT_MAGAZINE
    try:
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")) or f != int(f):
            return DEFAULT_MAGAZINE
        n = int(f)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_MAGAZINE
    return max(MAGAZINE_MIN, min(MAGAZINE_MAX, n))


@dataclass
class AppSettings:
    # -- camera ---------------------------------------------------------------
    device_name: str = ""          # matched first; survives index reshuffles
    device_index: int = -1         # fallback when the name is gone
    backend: str = "MSMF"
    width: int = 1280
    height: int = 720
    fourcc: str = "MJPG"
    # ASK FOR 60, not 30 (his decision, 2026-10-20, for the shipped release).
    # "100 ms pulse at 30 fps = about 3 frames per shot" is one of the two
    # numbers this project rests on, and at 60 fps it is about six -- so a
    # default of 30 halves the margin that shot confirmation lives on, and
    # `min_frames` of 2 goes from permissive to marginal. A camera that cannot
    # sustain 60 is not harmed: the request is simply not honoured and the app
    # measures and reports what actually arrives. The reverse is silent -- a
    # 60-capable camera held at half rate reads as "some shots leave no hole".
    # He shoots at 60; 30 had never been exercised on hardware here.
    fps: int = 60

    # -- behaviour ------------------------------------------------------------
    auto_connect: bool = True      # connect on launch without being asked
    auto_laser_hunt: bool = True   # drop exposure as soon as the camera opens
    auto_calibrate: bool = True    # measure the noise floor and set the threshold
    # Shoot the grid at the start of every session. The mapping is only valid
    # for the exact camera pose it was measured at, and a camera that gets
    # nudged -- by a cable, by the desk, by being aimed better -- produces holes
    # in the wrong place and a screen mask over the wrong part of the room,
    # which shows up as phantom shots. Nothing in software can see that the
    # camera moved after the fact, so the cheap, certain answer is to remeasure.
    calibrate_each_session: bool = True
    calibrate_seconds: float = 3.0
    settle_seconds: float = 1.5    # let exposure settle before measuring
    safe_preview: bool = True      # blue-multiply the preview; see CLAUDE.md
    sound: bool = True
    # Audible on a first launch (his decision, 2026-10-20). It was 0.05 --
    # deliberately quiet, raised with `+` -- which is right for a bench where
    # someone already knows the app works, and wrong for the first thirty
    # seconds of a stranger's. The gunshot is the user's only confirmation
    # that the app saw their trigger pull at all, and a silent one is the exact
    # fault that was chased for three days here before it turned out to be a
    # Windows per-app volume slider at zero. His own machine is unaffected:
    # a saved value wins over this.
    # 0.70, his instruction ("make the volume louder by default", 2026-10-20
    # 20:31). It went 0.05 -> 0.35 -> 0.70 tonight.
    #
    # AND THE COST OF LOUDER IS ALREADY PAID AT 0.35, which is why this is
    # cheap. The app mutes its own gunshot listener while it bangs, and that
    # mute scales with the gain, so a louder shot blinds for longer the one
    # channel that counts his trigger pulls independently of the camera.
    # Measured against his own 9mm sample (1.49s), mute plus room tail:
    #
    #     0.05 -> 0.82s      0.50 -> 1.51s
    #     0.35 -> 1.42s      0.70 -> 1.52s
    #                        1.00 -> 1.57s
    #
    # He shoots about one a second, so anything at or above 0.35 is already
    # deaf between shots; 0.35 -> 0.70 buys twice the loudness for 0.10s more.
    # Full scale is left alone deliberately: a gunshot and a figure dropping
    # overlap, and at 1.0 they sum past full scale and distort.
    volume: float = 0.70           # `+` and `-` adjust it; floor is 1%, never 0
    # How many people shoot (1 or 2), chosen on the menu. Saved so a family
    # evening survives a relaunch. Applied by the app only after setup is over;
    # `normalise_players` turns anything else into 1.
    players: int = 1
    # Rounds per magazine; see DEFAULT_MAGAZINE. No UI yet, by his instruction.
    magazine: int = DEFAULT_MAGAZINE
    shot_sound: str = ""           # empty = first real sample found in sounds/
    # The SPEAKER the gunshot comes out of, by name. Chosen deliberately
    # because the Windows default here is the monitor over DisplayPort, and
    # a display's audio endpoint vanishes when the display sleeps -- which
    # this app causes, every time it flashes the screen for calibration.
    sound_device: str = ""         # empty = pick the best non-display output
    # The microphone that listens for the gun, remembered BY NAME for the same
    # reason the camera is: USB enumeration order shifts when anything is
    # plugged in, and an index alone eventually selects the wrong device.
    listen_for_shots: bool = True
    mic_name: str = ""             # empty = pick one automatically
    # None means "never measured on this camera", which is what makes the app
    # run a full sweep rather than trust a number from some other room. A
    # remembered value is never revisited otherwise, so one measured in a lit
    # room follows the camera into a dark one and leaves the laser dimmer than
    # the conditions allow. Key X re-measures on demand.
    #
    # AND THE DEFAULT IS None FOR EXACTLY THAT REASON, which it was not until
    # 2026-10-20: it was -8.0, contradicting the five lines above it. The rule
    # was written down and the code did the opposite, which is a shape this
    # project has met before.
    #
    # Two things made it worth finding. A remembered value is NOT revisited --
    # the sweep is skipped whenever `saved is not None`, deliberately, because
    # control writes are the one operation measured to hang this camera -- so a
    # non-None default means a machine that has measured nothing still never
    # sweeps. And -8 is not a neutral number: on this hardware it measured shot
    # peak median 18 against a bar of 10-15, about 1.2x margin, and cost half a
    # string on 2026-10-15. Exposure response curves differ wildly between
    # cameras on the same property, so on somebody else's camera it is not even
    # wrong in a known direction.
    #
    # Caught by the release gate in packaging/assemble.py, which refuses to
    # build an installer carrying anything measured on another machine.
    exposure: "float | None" = None   # the laser-hunt exposure value; None =
                                      # never measured HERE, so measure it

    # -- detector / tracker ---------------------------------------------------
    detector: dict[str, Any] = field(default_factory=dict)
    tracker: dict[str, Any] = field(default_factory=dict)
    layers: dict[str, bool] = field(default_factory=dict)

    # -- the machine's best rounds ---------------------------------------------
    # The best SCORE and the best EFFICIENCY this machine has seen, keyed per
    # drill, per magazine size and per player count, each with the raw ingredients
    # it was derived from (see lsa/bests.py for the schema and the reasons).
    #
    # A FRESH INSTALL HOLDS NO BEST: the default is the empty dict, and
    # `packaging/assemble.py` refuses to build a release whose default is
    # anything else -- a best is something this machine measured, and a stranger
    # must not inherit his. A saved value beats a code default (the `min_frames`
    # lesson), which is exactly why the loader validates whatever it finds here
    # (`__post_init__`): a hand-edited or corrupt value costs that record, never
    # the file and never the launch.
    bests: dict[str, Any] = field(default_factory=dict)

    # -- how fast each drill's targets move ------------------------------------
    # `{mode name: multiplier}`, keyed by the drill's `Mode.name` and NOT by its
    # digit. The digits were renumbered on 2026-10-18 and a hand-written table
    # keyed on them lied for three days afterwards; a name is the drill's own
    # identity and moves with it.
    #
    # A drill absent from this dict runs at DEFAULT_SPEED, so a fresh install
    # and a drill added next year both behave without an entry existing.
    mode_speed: dict[str, float] = field(default_factory=dict)
    # Measured on this machine from real shots at moving targets. See
    # PIPELINE_LAG_DEFAULT above.
    pipeline_lag_s: float = PIPELINE_LAG_DEFAULT

    def __post_init__(self) -> None:
        self.bests = normalise_bests(self.bests)
        self.pipeline_lag_s = normalise_pipeline_lag(self.pipeline_lag_s)
        self.mode_speed = normalise_mode_speed(self.mode_speed)

    def speed_for(self, name: str) -> float:
        """The remembered multiplier for drill `name`, or the default.

        THE ONE READ PATH, handed to `ModeHost.speed_provider`, so a drill
        started any of the four ways gets the same answer (RULES E2).
        """
        return normalise_speed(self.mode_speed.get(name, DEFAULT_SPEED))

    def set_speed_for(self, name: str, value: float) -> float:
        """Remember `value` for drill `name` and return what was stored."""
        speed = normalise_speed(value)
        self.mode_speed[name] = speed
        return speed

    # -- plumbing -------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()                      # corrupt file must not block startup
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path) -> bool:
        """Atomically: see lsa/atomic.py. A truncated settings.local.json
        loses the camera chosen by name and the measured exposure, and the
        loader swallows a corrupt file and returns defaults -- silently."""
        return write_json_atomic(path, asdict(self))

    def resolve_device(self, cameras: list) -> int | None:
        """Pick the device to use, preferring the remembered name.

        Order: exact remembered name, then the remembered index if it is still a
        real camera, then the first non-virtual device. Returns None if there is
        nothing usable at all.
        """
        if self.device_name:
            for c in cameras:
                if c.name == self.device_name and not c.is_virtual:
                    return c.index
        for c in cameras:
            if c.index == self.device_index and not c.is_virtual:
                return c.index
        for c in cameras:
            if not c.is_virtual:
                return c.index
        return None
