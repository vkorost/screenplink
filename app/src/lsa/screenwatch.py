"""A passive watch on the app's OWN screen as the camera sees it.

Written 2026-10-18, from the night the camera silently degraded and the app
said READY. Guard 1 (`selfcheck.camera_not_delivering`) refuses READY from the
probe's dot reading. This is the other half: once he is shooting, is the thing
the camera is looking at -- the app's own picture -- still arriving as bright
as it was when the launch was judged healthy?

It costs the shooter nothing and the camera nothing: no dot, no interruption,
no control write. A frame arrives ~55 times a second and the app knows what it
painted; the capture thread reduces one frame in a dozen to a single number
(the median luminance of the screen's INK -- the lit pixels inside the screen
polygon) and this module judges that number. Pure on purpose: no Qt, no clock
of its own, so the whole policy runs in a unit test.

WHAT IT DOES NOT PROMISE (measured, not hoped)
----------------------------------------------
The 2026-10-17 bad session (18:29) left `recordings/verdict_camera*.png` behind.
Its whole-frame mean luminance was 10.6 against 10.4-10.9 on every healthy
launch, and the lit-pixel p99 195 against 185-226: the picture did NOT dim. What
fell was the camera's chroma (the red-excess maximum in the frame was 0-1,
against 8-14 on healthy frames before the ClearType fix), which the app's
grey-only drawing cannot show. So THIS watch would NOT have caught that night;
Guard 1 does. What it catches is the other family: exposure ignored,
auto-exposure fighting, a lens covered, the screen asleep or the window gone --
a real fall in how bright our own ink is.

HOW IT AVOIDS FIRING ON LEGITIMATE DIMMING
------------------------------------------
- LIKE WITH LIKE. The reference is per SIGNATURE (which layers are visible and
  which mode is running), taken from the first stable samples of that signature
  after the verdict. A verdict panel coming or going, key 0 hiding the info
  layers, a new mode, all change the signature and so are compared with their
  own reference, never with another state's.
- INK, NOT AREA. The metric is the median luminance of lit pixels, not the mean
  of the region, so a mode that draws LESS (balls shot, a figure gone) leaves it
  where it was. A blank screen has no ink at all and is simply not judged.
- A RATIO, SUSTAINED. Below the watchdog's own relative bar for SUSTAIN_S
  seconds without a break, once, with hysteresis before it can speak again.
- Not while a routine owns the screen, not while the camera is not looking at
  our screen: those are the caller's predicates, passed in as `pause()`.

KNOWN WEAKNESS, STATED
----------------------
A signature seen for the first time AFTER the camera has degraded takes its
reference from the degraded camera, so it cannot condemn it. The reference
for the launch's own first state was taken right after the verdict, which now
requires a healthy probe dot.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, Hashable, Optional

# The same relative bar as the mark watchdog's mid-session test, from the same
# place (one fact, one number). FITTED there: healthy launches spread about
# +/-10% apart, so half is five times that spread away from normal.
from lsa.watchdog import REL_BAD_BELOW as RATIO_BAD_BELOW

# -- constants, each with its provenance --------------------------------------
#
# Capture-side reduction rate. GUESSED (cost-driven; the measured cost is
# reported by tests/test_screenwatch.py).
SAMPLE_HZ = 4.0
# Pixels at or below this luminance are black level, not ink. GUESSED: the
# panel's black reads under 20 in every recorded camera frame and the app's
# dimmest strokes sit far above.
INK_MIN_LUMA = 40
# Fewer ink pixels than this (of ~14k sampled) and there is nothing to judge.
# GUESSED.
INK_MIN_PIXELS = 40
# A reference quieter than this is not a standard to hold anything to. GUESSED
# (his healthy verdict frames: lit pixels read ~200).
REF_MIN_LEVEL = 60.0
# Stable samples that make a reference (2 s at SAMPLE_HZ). GUESSED.
REF_SAMPLES = 8
# The fall must last this long without a break before it speaks. GUESSED: long
# enough that a hit flash or a mode change cannot be mistaken for a fault,
# short enough to precede a string.
SUSTAIN_S = 5.0
# Back above this fraction of the reference for RECOVER_S re-arms the watch.
# GUESSED hysteresis (midway between the bar and 1).
RECOVER_RATIO = 0.75
RECOVER_S = 2.0


class ScreenWatch:
    def __init__(self, log: Callable[[str], None] = lambda s: None) -> None:
        self._log = log
        self.armed = False
        self._refs: Dict[Hashable, float] = {}
        self._building: Dict[Hashable, Deque[float]] = {}
        self._below_since: Optional[float] = None
        self._ok_since: Optional[float] = None
        self.latched = False
        self.fired = 0
        self.skipped_unjudgeable = 0

    # -- control -------------------------------------------------------------

    def arm(self) -> None:
        """A launch has just been judged usable: references start afresh, in
        the state the camera was last PROVEN healthy in."""
        self.armed = True
        self._refs.clear()
        self._building.clear()
        self._below_since = None
        self._ok_since = None
        self.latched = False

    def disarm(self) -> None:
        self.armed = False
        self._below_since = None

    def pause(self) -> None:
        """The caller says this moment cannot be judged (a routine owns the
        screen, the camera is not on our screen). A break in the sustain."""
        self._below_since = None

    # -- the judgement -------------------------------------------------------

    def reference(self, signature: Hashable) -> Optional[float]:
        return self._refs.get(signature)

    def feed(self, now: float, level: Optional[float],
             signature: Hashable) -> Optional[str]:
        """One sample. Returns a sentence, once, when the screen has dimmed.

        `level` None means there was no ink to measure (blank screen): not a
        fault, not evidence -- it breaks the sustain and nothing else.
        """
        if not self.armed:
            return None
        if level is None:
            self.skipped_unjudgeable += 1
            self._below_since = None
            return None
        ref = self._refs.get(signature)
        if ref is None:
            buf = self._building.setdefault(signature,
                                            deque(maxlen=REF_SAMPLES))
            buf.append(float(level))
            if len(buf) >= REF_SAMPLES:
                ordered = sorted(buf)
                med = ordered[len(ordered) // 2]
                if med >= REF_MIN_LEVEL:
                    self._refs[signature] = med
                    self._log(f"  screen watch: reference for this screen "
                              f"state is {med:.0f} (median ink luminance)")
                buf.clear()
            self._below_since = None
            return None
        ratio = level / ref
        if ratio >= RECOVER_RATIO:
            self._below_since = None
            if self.latched:
                if self._ok_since is None:
                    self._ok_since = now
                elif now - self._ok_since >= RECOVER_S:
                    self.latched = False
                    self._ok_since = None
                    self._log("  screen watch: the screen is as bright as it "
                              "was; armed again")
            return None
        self._ok_since = None
        if ratio >= RATIO_BAD_BELOW:
            self._below_since = None
            return None
        if self.latched:
            return None
        if self._below_since is None:
            self._below_since = now
            return None
        if now - self._below_since < SUSTAIN_S:
            return None
        self.latched = True
        self.fired += 1
        return (f"the app's own screen reads {level:.0f} in the camera "
                f"against {ref:.0f} when this launch was judged healthy "
                f"({ratio * 100:.0f}%, bar {RATIO_BAD_BELOW * 100:.0f}%) for "
                f"{now - self._below_since:.0f} s")
