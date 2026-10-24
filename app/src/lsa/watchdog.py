"""The mark watchdog: a bang was heard, and NOTHING appeared on screen.

Written 2026-10-17 night, from his design: "a background checker that detects
shots by sound, but sees that there are no holes painted."

He shot a string into a camera that had silently degraded. The app said READY,
he fired, more than half his shots left no hole, and an hour went before anyone
noticed -- with the evidence in the app the whole time.

## It measures the END of the chain, not the middle

A shot can be detected perfectly and still leave nothing on screen: no mapping
to place it, the holes layer switched off (2026-10-16: every shot detected and
mapped and drawn into a layer his settings had turned off, READY and 12 of 12
over a black screen), a mode swallowing it, the camera gone deaf. So the signal
is not "heard and not detected" (the corroborator already counts that) but
"heard, and no mark appeared anywhere". The ledger is fed by the code that
actually puts the mark on the scene, never by the detector.

## Why a RUN and not a count

A healthy session has plenty of these. His good string: 24 heard-without-
detection out of 213 pulls, 11%, on a perfectly working camera, because the
microphone is deaf for ~0.8 s after each of the app's own bangs and he shoots
about once a second. A cumulative count would fire during the best session this
project has recorded. What separates healthy from broken is the pattern:
scattered, against runs of four to twelve consecutive.

So: THREE consecutive heard-with-nothing-drawn triggers a SELF-CHECK -- never a
prompt. The check is the app's own dot measurement (about a second, no
shooting, no camera control write). Its reading decides whether to speak.

## Pure on purpose

No Qt, no clock of its own (the caller passes `time.perf_counter()`), so the
whole policy runs in a unit test with a counter and a list of times.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

# -- the run ----------------------------------------------------------------

RUN_TRIGGER = 3
# A mark follows its click: the trigger click precedes the laser by a few ms,
# then the tracker needs min_frames (two) at ~50 fps plus the pulse. The
# corroborator's own window is 0.3 s each way; the mark side is narrower BEFORE
# the click on purpose, so the previous shot's late mark cannot claim this one.
MARK_PRE_S = 0.10
MARK_POST_S = 0.50
RESOLVE_AFTER_S = MARK_POST_S + 0.10
LEDGER_KEEP_S = 30.0

# -- the self-check's limits ------------------------------------------------

# Not mid-string: the check draws a red dot and the detections it causes are
# the app's, so a real shot fired during it is swallowed. Wait for a gap.
QUIET_S = 2.0
# Give up on a trigger that never finds a gap. The run is dropped, and logged.
DEFER_MAX_S = 15.0
# A run whose last empty click is older than this is stale (he stopped
# shooting); it must not fire a check minutes later.
STALE_S = 30.0
# Rate limits. A healthy answer ("he was missing the panel") can be repeated
# at most once a minute; a bad answer is the alert, and repeating it faster
# than every two minutes is nagging. Two bad answers in a row and the watchdog
# goes quiet until a mark appears: the camera is known to be bad, and a check
# every three shots for ever is exactly the nuisance the brief forbids.
GAP_AFTER_HEALTHY_S = 60.0
GAP_AFTER_BAD_S = 120.0
BAD_LATCH = 2
MAX_CHECKS = 8          # per session; the hard ceiling

# -- the verdict ------------------------------------------------------------
#
# The probe's dot readings on healthy launches, 2026-10-17:
#     44, 46, 49, 52, 43, 25, 47, 52, 46
# and on broken ones: 5, then 0 once the camera wedged.
# Those readings are taken at the brightness the probe AUTO-RANGED to, and the
# ranging stops climbing the moment a reading reaches PROBE_PEAK_LO = 25 -- so
# a healthy camera cannot read below 25 at its ranged brightness (the lowest
# healthy launch, 25, is exactly that floor). The absolute bar is FITTED to
# those readings, not reasoned: 15 sits 0.6x below the lowest healthy reading
# and 3x above the failing 5. It is applied at the brightness the probe ranged
# to this launch; with no probe this launch, at full brightness, where a
# healthy camera reads far above it.
ABS_BAD_BELOW = 15.0
# A camera that degrades MID-SESSION, having started healthy: half of this
# session's own probe reading. His healthy launches spread 43-52 (about +/-10%)
# apart from the one at the ranging floor, so 0.5 is five times that spread
# away from normal. Used ONLY when the session's baseline is itself healthy
# (>= ABS_BAD_BELOW): a baseline of 5 is not a standard to be held to -- that
# is the night this was written for, a launch whose probe read 5 before he
# fired a shot. Either test firing is enough.
REL_BAD_BELOW = 0.5


class MarkLedger:
    """A monotonic count of marks the scene has been given, with timestamps.

    Module-level (`LEDGER`) so a figure mode, which draws its own mark on the
    figure and withholds the screen hole, can declare the mark without knowing
    a watchdog exists.
    """

    def __init__(self) -> None:
        self.count = 0
        self.last_t: Optional[float] = None

    def note(self, t: Optional[float] = None) -> None:
        self.count += 1
        self.last_t = time.perf_counter() if t is None else t


LEDGER = MarkLedger()


def record_mark(t: Optional[float] = None) -> None:
    """A mode drew something for a shot it absorbed."""
    LEDGER.note(t)


@dataclass
class Verdict:
    kind: str                 # "healthy" | "bad" | "unmeasurable"
    why: str                  # the numbers, in words
    absolute_fired: bool = False
    relative_fired: bool = False


def judge(reading: Optional[float], baseline: Optional[float]) -> Verdict:
    """Read the self-check's dot against BOTH standards. Either is enough.

    `reading` None means the check could not measure at all (no mask, no frame
    pair resolved); 0.0 is a measurement -- the camera delivered and the dot
    was not in it -- and is as bad as it gets.
    """
    if reading is None:
        return Verdict("unmeasurable", "the check could not measure the dot")
    bits = [f"dot reads {reading:.0f}"]
    absolute = reading < ABS_BAD_BELOW
    bits.append(f"absolute bar {ABS_BAD_BELOW:.0f}"
                + (" (BELOW)" if absolute else " (ok)"))
    relative = False
    if baseline is None:
        bits.append("no probe baseline this session")
    elif baseline < ABS_BAD_BELOW:
        bits.append(f"this session's probe read {baseline:.0f}, itself under "
                    f"the bar, so it is not used as a standard")
    else:
        relative = reading < REL_BAD_BELOW * baseline
        bits.append(f"session baseline {baseline:.0f}, {REL_BAD_BELOW:g}x = "
                    f"{REL_BAD_BELOW * baseline:.0f}"
                    + (" (BELOW)" if relative else " (ok)"))
    kind = "bad" if (absolute or relative) else "healthy"
    return Verdict(kind, "; ".join(bits), absolute, relative)


@dataclass
class _Click:
    t: float
    outcome: str = ""         # "" until resolved, then "ok" | "empty"


class MarkWatchdog:
    """Pairs counted trigger clicks with marks; owns the run and the rate limits."""

    def __init__(self, log: Callable[[str], None] = lambda s: None) -> None:
        self._log = log
        self._marks: List[List[float]] = []       # [t, claimed(0/1)]
        self._pending: List[_Click] = []
        self.run = 0
        self.run_last_t: Optional[float] = None
        self.history: Deque[Tuple[float, str]] = deque(maxlen=64)
        self.ok_total = 0
        self.empty_total = 0
        self.last_click_t: Optional[float] = None
        self.in_check = False
        self.swallowed = 0
        self.checks = 0
        self.bad_streak = 0
        self.latched = False
        self.last_check_end: Optional[float] = None
        self.last_gap = 0.0
        self._eligible_since: Optional[float] = None
        self._said: set = set()

    # -- inputs -----------------------------------------------------------

    def note_mark(self, t: float) -> None:
        """A mark was put on the scene (the end of the chain)."""
        self._marks.append([t, 0.0])
        # A mark means the chain works again: lift the latch.
        if self.latched or self.bad_streak:
            self._log("  mark watchdog: a mark appeared, so it is armed again")
        self.latched = False
        self.bad_streak = 0

    def note_click(self, t: float, counted: bool) -> None:
        """A microphone click. `counted` False = heard but not a trigger pull
        the app can be held to (the app's own drawing, an uncalibrated
        microphone, the grid, a covered screen)."""
        self.last_click_t = t
        if self.in_check:
            self.swallowed += 1
            return
        if counted:
            self._pending.append(_Click(t))

    # -- resolution -------------------------------------------------------

    def resolve(self, now: float) -> List[Tuple[float, str]]:
        """Settle every click old enough that its mark would have arrived."""
        out: List[Tuple[float, str]] = []
        keep: List[_Click] = []
        for c in self._pending:
            if now - c.t <= RESOLVE_AFTER_S:
                keep.append(c)
                continue
            hit = None
            for m in self._marks:
                if not m[1] and c.t - MARK_PRE_S <= m[0] <= c.t + MARK_POST_S:
                    hit = m
                    break
            if hit is not None:
                hit[1] = 1.0
                c.outcome = "ok"
                self.ok_total += 1
                self.run = 0
            else:
                c.outcome = "empty"
                self.empty_total += 1
                self.run += 1
                self.run_last_t = c.t
                self._log(f"  mark watchdog: bang heard at t={c.t:.3f}, NOTHING "
                          f"appeared on screen within {MARK_POST_S:g} s "
                          f"(run of {self.run} consecutive; "
                          f"{self.empty_total} empty / {self.ok_total} drawn "
                          f"this session)")
            self.history.append((c.t, c.outcome))
            out.append((c.t, c.outcome))
        self._pending = keep
        self._marks = [m for m in self._marks if now - m[0] <= LEDGER_KEEP_S]
        return out

    # -- the decision to check -------------------------------------------

    def _say_once(self, key: str, text: str) -> None:
        if key not in self._said:
            self._said.add(key)
            self._log(text)

    def tick(self, now: float, busy: bool) -> bool:
        """True when the self-check should be started NOW."""
        if self.in_check:
            return False
        if self.run < RUN_TRIGGER:
            self._eligible_since = None
            self._said.clear()
            return False
        if self.run_last_t is not None and now - self.run_last_t > STALE_S:
            self._log(f"  mark watchdog: a run of {self.run} is "
                      f"{now - self.run_last_t:.0f} s old (more than "
                      f"{STALE_S:.0f} s); he stopped shooting, dropping it "
                      f"without a check")
            self.run = 0
            return False
        if self.latched:
            self._say_once("latched",
                           f"  mark watchdog: run of {self.run}, but the last "
                           f"{self.bad_streak} checks already said the camera is "
                           f"bad; staying quiet until a mark appears")
            return False
        if self.checks >= MAX_CHECKS:
            self._say_once("max", f"  mark watchdog: run of {self.run}, but "
                                  f"{MAX_CHECKS} checks have run this session "
                                  f"(the ceiling); not checking again")
            return False
        if (self.last_check_end is not None
                and now - self.last_check_end < self.last_gap):
            self._say_once("gap", f"  mark watchdog: run of {self.run}, rate "
                                  f"limited: the last check ended "
                                  f"{now - self.last_check_end:.0f} s ago, the "
                                  f"minimum gap is {self.last_gap:.0f} s")
            return False
        if self._eligible_since is None:
            self._eligible_since = now
            self._log(f"  mark watchdog: {self.run} consecutive bangs with "
                      f"nothing drawn -- will run the app's own dot check "
                      f"(not a prompt) at the next gap of {QUIET_S:g} s")
        if now - self._eligible_since > DEFER_MAX_S:
            self._log(f"  mark watchdog: no {QUIET_S:g} s gap in "
                      f"{DEFER_MAX_S:.0f} s, so the check is ABANDONED and the "
                      f"run of {self.run} dropped (a check mid-string would "
                      f"swallow his shots)")
            self.run = 0
            self._eligible_since = None
            return False
        if busy or self._pending:
            return False
        if self.last_click_t is not None and now - self.last_click_t < QUIET_S:
            return False
        return True

    def begin_check(self, now: float) -> None:
        self.checks += 1
        self.in_check = True
        self.swallowed = 0
        self._log(f"  mark watchdog: CHECK {self.checks}/{MAX_CHECKS} starting "
                  f"after a run of {self.run}")

    def abort_check(self) -> None:
        """The check could not start (the slot was busy). Not counted."""
        self.in_check = False
        self.checks = max(0, self.checks - 1)

    def end_check(self, now: float, verdict: Verdict) -> None:
        self.in_check = False
        run = self.run
        if verdict.kind == "bad":
            self.bad_streak += 1
            self.last_gap = GAP_AFTER_BAD_S
            if self.bad_streak >= BAD_LATCH:
                self.latched = True
        else:
            self.bad_streak = 0
            self.last_gap = GAP_AFTER_HEALTHY_S
        self.last_check_end = now
        self.run = 0
        self._eligible_since = None
        self._said.clear()
        self._log(f"  mark watchdog: CHECK result {verdict.kind.upper()} after "
                  f"a run of {run}: {verdict.why}"
                  + (f"; {self.swallowed} bang(s) heard during the check were "
                     f"not counted" if self.swallowed else "")
                  + ("; healthy, so nothing is said to him: the dot is seen, he "
                     "was missing the panel" if verdict.kind == "healthy" else ""))
