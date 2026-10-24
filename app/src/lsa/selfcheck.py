"""Decide whether this app can see a laser right now -- and if not, say why.

THE POINT OF THIS MODULE, in the shooter's own words:

    "when a human finally gets to shoot, he will see either a working app or
    the app giving him the exact reason why it is not working and what needs
    to be done to make it work."

Everything else in this project measures one stage. This decides the question
the person at the firing line is actually asking, from the measurements the
startup self-check collects, and it names an action rather than a symptom.

It is deliberately pure: it takes numbers and returns a verdict. No camera, no
Qt, no I/O. That is what makes it testable against every failure this project
has actually produced, which is the only way to trust advice.

WHY A DETECTION RATE AND NOT A THRESHOLD CHECK
----------------------------------------------
Every number the app used to report -- noise floor, signal margin, flash
difference, held-out error -- is a property of one stage. All of them can look
healthy while the shooter's trigger pulls go undetected, and all of them did,
repeatedly, on 2026-10-15. The only honest summary is: of N shots the app
FIRED AT ITSELF, how many came back as hits in the right place?

The emulated shots must therefore resemble the real ones, including their
flicker. Measured over 575 of Vladimir's logged trigger pulls:

    peak    min 31   p10 41   median 85   p90 180   max 253
    area    p10 113  median 296   p90 746 px
    pulse   p10 40   median 59    p90 85 ms

The p90/p10 brightness ratio is 4.4x. That spread is not noise in the
measurement, it IS the measurement: a PWM-driven diode sampled by a camera
gives a different slice of its on-time every frame, which is why he sees some
shots vividly and some as a faint shadow. A self-check that fires only bright
dots proves nothing about the dim half, and the old one did exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# THE SAME NUMBER AS THE MARK WATCHDOG, FROM THE SAME PLACE. One fact (how
# faint may the app's own full-red test dot read before the camera is judged
# not to be delivering) has exactly one threshold. Its provenance is stated
# at its definition in `lsa/watchdog.py`: FITTED to the 2026-10-17 readings
# (healthy 25-97, bad 5 / 10 / 0), not reasoned.
from lsa.watchdog import ABS_BAD_BELOW as PROBE_DOT_BAD_BELOW


class Verdict(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"        # usable, but the shooter should know
    BLOCKED = "blocked"          # do not waste his trigger finger


# What fraction of emulated shots must come back. Below the first number the
# app is not usable; between them it works but loses shots noticeably.
#
# 0.90 rather than 1.00 because the emulated population deliberately includes
# the dim tail, and a few of those legitimately sit under any sane threshold.
# Vladimir's complaint was "one in three", i.e. a rate of 0.67, so the bar has
# to sit well above that to mean anything.
RATE_READY = 0.90
RATE_DEGRADED = 0.70

# Position error that makes a hole look wrong rather than merely imprecise. A
# 4K 42" panel is about 0.24mm per pixel, and these notes measure a good
# calibration at under 1.5mm held-out.
ERROR_READY_MM = 5.0
ERROR_DEGRADED_MM = 12.0

# Below this the detector is working on margin it does not have: a real pulse
# flickers 4.4x, so a median shot only 2x over the bar means the dim half
# lands under it. Measured directly: at exposure -8 the median shot peak was
# 18 against a bar of 15 and half the string vanished; at -5 it was 74 against
# a bar of 7 and the same gun registered.
# Derived from his own distribution rather than chosen. The dimmest shot
# in 575 logged trigger pulls measured 31 against a median of 85, so a
# median sitting 2.6x over the bar puts EVERY shot he has ever fired above
# it. 3.0 adds a little headroom for a floor that drifts during a session.
#
# It was 4.0, picked by reasoning from the 4.4x flicker ratio, and that was
# wrong twice over: the ratio is p90/p10, not max/median, so it overstates
# the gap that matters by nearly two. Measured live, it failed a launch at
# 3.9x in which all twelve emulated shots were detected -- a heuristic
# contradicting the measurement it exists to approximate.
MARGIN_READY = 3.0


def _probe_answered(c: "SelfCheck") -> bool:
    """Has the exposure probe already answered the question, better?

    THE SELF TEST CANNOT MEASURE DETECTION RATE. These notes concluded that
    on 2026-10-14 and the verdict never honoured it: "A screen EMITS its
    dot; the panel's own emission does not fall off the way light REFLECTED
    off its matte coating does. So the self test measures geometry, plumbing
    and accuracy exactly, and cannot measure detection rate or sensitivity
    at all."

    Measured on his bench 2026-10-18, inside ONE launch: five emulated dots
    at the SAME 100% brightness read peak 53, 17, 27, 137 and 13 -- a
    ninefold spread out of a 10px emitted dot. His real trigger pulls that
    same morning measured peak 86-103 at 10-13x the bar. In the same
    seconds, the probe's own dot -- auto-ranged into the sensor's linear
    range, which is the whole point of it -- read 114 against a bar of 8.

    So when the probe's dot clears the bar comfortably, a middling self-test
    rate is evidence about the TEST DOT and not about his gun. Saying "Not
    ready, check the screen" from it sent him to fix a setup that was
    working, which is the fourth time this project has emitted advice its
    own measurements contradicted.

    ABSENT EVIDENCE IS NOT EVIDENCE (RULES C1 cuts both ways): with no probe
    reading this returns False and nothing below changes.
    """
    if c.probe_dot is None or not c.min_peak:
        return False
    return (c.probe_dot / max(1.0, c.min_peak)) >= MARGIN_READY
MARGIN_DEGRADED = 2.0


@dataclass
class SelfCheck:
    """Everything the startup check measured. Any field may be None if unknown."""

    fired: int = 0
    detected: int = 0
    error_mm: float | None = None          # median position error
    worst_error_mm: float | None = None
    peak_median: float | None = None       # of the detected emulated shots
    noise_floor: float | None = None
    min_peak: float | None = None
    fps: float | None = None
    exposure: float | None = None
    exposure_spread: float | None = None   # 0..1, how much the sweep moved
    flash_difference: float | None = None
    held_out_mm: float | None = None
    # Is there a camera-to-screen mapping AT ALL? False means every shot will
    # be seen and counted and NOTHING will be drawn, which from the firing
    # line is indistinguishable from a detector that has stopped working.
    # None means unknown, and is never read as a fault (RULES C1 cuts both
    # ways) -- so every existing caller and every generated verdict is
    # unchanged until it is set.
    calibrated: bool | None = None
    audio_ok: bool | None = None
    camera_name: str = ""
    rejected_short: int = 0
    room_bright: bool | None = None        # background too bright to stop down
    # What the exposure probe's own full-red dot read, at the brightness it
    # ranged to this launch. None means the probe never took a reading (it
    # bailed, or the camera delivered no frame): ABSENT EVIDENCE, never read
    # as a fault (RULES C1 cuts both ways).
    probe_dot: float | None = None

    @property
    def rate(self) -> float:
        return self.detected / self.fired if self.fired else 0.0

    @property
    def margin(self) -> float | None:
        if self.peak_median is None or not self.min_peak:
            return None
        return self.peak_median / max(1.0, self.min_peak)


@dataclass
class Finding:
    """One thing that is wrong, what it means, and what to DO about it."""

    verdict: Verdict
    headline: str
    detail: str
    action: str = ""
    measured: str = ""

    def lines(self) -> list[str]:
        out = [self.headline]
        if self.measured:
            out.append(f"    measured: {self.measured}")
        if self.detail:
            out.append(f"    {self.detail}")
        if self.action:
            out.append(f"    DO THIS: {self.action}")
        return out


@dataclass
class Report:
    verdict: Verdict
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""

    def lines(self) -> list[str]:
        out = [self.summary] if self.summary else []
        for f in self.findings:
            out.extend(f.lines())
        return out


def camera_not_delivering(c: SelfCheck) -> bool:
    """True when the probe measured its own dot and it was far too faint.

    2026-10-17: the camera silently degraded, the app reported READY, he fired
    a string and more than half of it left no hole. The probe had already
    printed the evidence before he fired a shot (it read 5, then 10, then 0;
    nine healthy launches read 25-97) and nothing acted on it.
    """
    return c.probe_dot is not None and c.probe_dot < PROBE_DOT_BAD_BELOW


def _camera_findings(c: SelfCheck) -> list[Finding]:
    """Faults in the camera itself, which no software setting can work around."""
    out: list[Finding] = []

    if camera_not_delivering(c):
        out.append(Finding(
            Verdict.BLOCKED,
            "The camera is not delivering the app's own test dot.",
            "The app drew a full-red dot on its own screen and the camera "
            "barely registered it. On every healthy launch that dot reads "
            "25 or more. Any shot you fire now would go unseen, whatever "
            "the rest of this check says.",
            "Do not shoot. Close the app by closing its window (never "
            "kill it), unplug the camera and plug it back in, then start "
            "the app again. If it persists, try another USB port.",
            f"the test dot read {c.probe_dot:.0f} at full red; "
            f"{PROBE_DOT_BAD_BELOW:.0f} is the line, healthy is 25-97",
        ))

    if c.exposure_spread is not None and c.exposure_spread < 0.25:
        out.append(Finding(
            Verdict.BLOCKED,
            "The camera is IGNORING the exposure setting.",
            "Everything downstream depends on a dark, fixed exposure: without "
            "it the screen never fades to a ghost, the noise floor climbs and "
            "real shots are lost. This is the camera, not the app.",
            "Unplug and replug the camera. If that does not help, open the "
            "camera's own driver dialog and untick 'Low Light Compensation' "
            "and any auto exposure.",
            f"brightness moved only {c.exposure_spread * 100:.0f}% across the "
            f"whole exposure range; a working camera moves 80%+",
        ))

    if c.flash_difference is not None and c.flash_difference < 18.0:
        out.append(Finding(
            Verdict.BLOCKED,
            "The camera cannot see this screen.",
            "The monitor was flashed white then black and the camera barely "
            "noticed. Either it is not pointed at the screen, something is "
            "covering it, or the display is asleep.",
            "Check the camera is aimed at the monitor with the whole panel in "
            "view, and that nothing is over the lens.",
            f"flash difference {c.flash_difference:.1f}, healthy is 40+",
        ))

    return out


def _signal_findings(c: SelfCheck) -> list[Finding]:
    """The dot is seen, but not well enough. These have real, specific cures."""
    out: list[Finding] = []
    margin = c.margin

    if margin is not None and margin < MARGIN_DEGRADED:
        if c.room_bright:
            out.append(Finding(
                Verdict.BLOCKED,
                "The dot is barely brighter than the background.",
                # NAMES THE WINDOW, not the room. What this is measured
                # from is the mean and chroma of the WHOLE camera frame,
                # and the brightest thing in it is the daylight window
                # beside the monitor -- not the room he stands in, which
                # he has already darkened as far as he can and still see
                # his sights.
                "Daylight in the camera's view -- the window beside the "
                "monitor -- means the exposure had to be stopped down, and "
                "a shorter exposure collects less of the laser. A pulse "
                "flickers about 4x between its dimmest and brightest, so at "
                "this margin the dim half of every string is lost.",
                # The third site, and the one that was on his screen.
                # See the note in `_rate_findings`: "dim the room" is not
                # an action he has left, and `room_bright` measures the
                # whole camera frame, window included.
                "Press X to re-measure the exposure against the room as it "
                "is now, then A to run this check again.",
                f"median shot {c.peak_median:.0f} against a bar of "
                f"{c.min_peak:.0f} -- {margin:.1f}x, and 4x is needed",
            ))
        else:
            out.append(Finding(
                Verdict.BLOCKED,
                "The dot is too faint for the detector.",
                "The room is not the problem -- the floor is low. The dot "
                "itself is arriving weak, which means too little light is "
                "reaching the sensor from it.",
                "Press X to re-measure the exposure, or move the camera closer "
                "to the screen so the dot covers more pixels.",
                f"median shot {c.peak_median:.0f} against a bar of "
                f"{c.min_peak:.0f} -- {margin:.1f}x, and 4x is needed",
            ))
    elif (margin is not None and margin < MARGIN_READY
          and c.rate < RATE_READY
          and not _probe_answered(c)):
        # ONLY when the rate has not already answered the question.
        #
        # The margin is a PREDICTION of how many shots will be lost. The
        # detection rate is a MEASUREMENT of it, taken seconds earlier,
        # over a population drawn from his own 575 logged pulls. When the
        # measurement says every one of them registered, a prediction that
        # says otherwise is not a second opinion -- it is a worse
        # instrument disagreeing with a better one, and acting on it told
        # him to go and change something that was working.
        # THE SAME SPLIT AS ABOVE, AND IT WAS MISSING HERE.
        #
        # The BLOCKED branch asks whether the room is actually bright before
        # blaming it. This one did not, and said "dimming the room would widen
        # it" to a man who had already dimmed his room as far as he can and
        # still see his sights. His words, 2026-10-15: "This is the max
        # darkness for my comfort."
        #
        # That is the SECOND time this project has given that advice to
        # someone sitting in the dark, and the first time is written up in
        # CLAUDE.md under "advice that cannot be followed is worse than
        # silence" -- because the warnings are the only thing standing between
        # a bad session and an unexplained one, and an impossible instruction
        # teaches him to ignore the next one.
        #
        # A dark room is not a problem to be fixed. It is the BEST condition
        # this app can have: it is what allows a longer exposure, and the
        # laser's brightness scales with exposure time. When the floor is low
        # and the margin is still thin, the dot is arriving faint, and only
        # three things change that -- more exposure, more dot pixels, or a
        # shorter range.
        if c.room_bright:
            # NOT "DIM THE ROOM". For the fourth time (RULES C4, D7).
            #
            # It was on his screen again on 2026-10-16 at 11:03, twice,
            # while he was shooting -- after he had already closed the
            # window and darkened the room as far as he can and still see
            # his sights. The 10,080-verdict generator in test_selfcheck
            # cannot see this branch, because it holds the room measured
            # DARK and this is the bright branch.
            #
            # And `room_bright` is not a measurement of his room: it comes
            # from the mean and chroma of the WHOLE camera frame, which
            # includes the sunlit window beside the monitor. He cannot act
            # on it, and the one thing he could do about that window he has
            # already done.
            #
            # So it becomes information, never an instruction, and the
            # action is the one he can actually take.
            action = ("Press X to re-measure the exposure against the room "
                      "as it is now.")
            detail = ("Shots will register, but the dimmest flickers of a "
                      "pulse may be missed. Daylight in the camera's view -- "
                      "the window beside the monitor, not the room you are "
                      "standing in -- is limiting how long an exposure the "
                      "app can use.")
        else:
            action = ("Press X: the app will re-measure the exposure against "
                      "the room as it is now. If that does not widen it, move "
                      "the camera closer to the screen or zoom in, so the dot "
                      "lands on more pixels.")
            detail = ("Shots will register, but the dimmest flickers of a "
                      "pulse may be missed, most likely on the far side of "
                      "the screen. The room is NOT the problem -- the "
                      "background is already quiet. The dot itself is "
                      "arriving faint.")
        out.append(Finding(
            Verdict.DEGRADED,
            "The signal margin is thin.",
            detail,
            action,
            f"median shot {c.peak_median:.0f} against a bar of "
            f"{c.min_peak:.0f} -- {margin:.1f}x"
            + ("" if c.room_bright else
               f", with a quiet background (floor {c.noise_floor:.0f})"
               if c.noise_floor is not None else ""),
        ))

    if c.noise_floor is not None and c.noise_floor > 25:
        out.append(Finding(
            Verdict.DEGRADED,
            "Something bright and red is in the camera's view.",
            "The rejection bar has to sit above it, and that bar then also "
            "rejects real shots on the dim side of the screen.",
            "Look at the camera pane: find what is red and bright, and move "
            "it, cover it, or re-aim the camera so it is out of frame.",
            f"noise floor {c.noise_floor:.0f}, quiet is under 10",
        ))

    return out


def _speed_findings(c: SelfCheck) -> list[Finding]:
    """Frame rate decides how many camera frames a 100ms pulse lands in."""
    out: list[Finding] = []
    if c.fps is None:
        return out
    if c.fps < 15:
        out.append(Finding(
            Verdict.BLOCKED,
            "The camera is delivering too few frames to confirm a shot.",
            "A trigger pull lights the laser for about 100ms. Confirming a "
            "shot needs it to land in at least two camera frames, and at this "
            "rate many pulses land in one or none.",
            "Close other programs using the camera or the CPU. If it persists "
            "the capture format is wrong -- it should be MJPG at 1280x720.",
            f"{c.fps:.1f} fps; 25+ is needed, 60 is what this camera can do",
        ))
    elif c.fps < 25:
        out.append(Finding(
            Verdict.DEGRADED,
            "The frame rate is low enough to cost shots.",
            "A 100ms pulse lands in fewer camera frames than the confirmation "
            "rule expects, so the shortest pulses will be discarded.",
            "Turning off recording, if it is on, usually recovers most of it.",
            f"{c.fps:.1f} fps, {c.rejected_short} pulses already discarded as "
            f"too short",
        ))
    return out


def _accuracy_findings(c: SelfCheck) -> list[Finding]:
    out: list[Finding] = []
    # NO MAPPING AT ALL comes first, and it is BLOCKED.
    #
    # 2026-10-18 10:04: pattern calibration was rejected (correctly or not --
    # see `judge_held_out`), the app went uncalibrated, and he fired 42 shots.
    # Every one was DETECTED, at peaks 76-109 against a bar of 8 -- margins of
    # 9 to 13x, a textbook-healthy detector -- and not one left a hole,
    # because there is nowhere to draw it. That is the correct behaviour and
    # the app never said so, so he concluded it was broken.
    #
    # There is no finding here to flatter: every other number in the check can
    # be perfect while this one is false, and when it is false the verdict must
    # be BLOCKED rather than READY. The old `assess` with no mapping, no shots
    # fired and a healthy camera returned "READY - shoot."
    if c.calibrated is False:
        out.append(Finding(
            Verdict.BLOCKED,
            "There is no calibration, so no hole can be drawn.",
            "Your shots are still being SEEN and counted -- the detector is "
            "working. What is missing is where on this screen the camera is "
            "looking, so the app cannot put a hole where you hit, and it "
            "draws nothing rather than guess. Nothing appearing when you fire "
            "is this, not a detection fault.",
            "Press C on the target screen: the app photographs its own "
            "markers and measures the screen again, with nothing to shoot. "
            "If that keeps failing, press G and shoot the grid instead.",
            "no camera-to-screen mapping",
        ))
        # And nothing below is meaningful: a position error is measured
        # THROUGH the mapping that does not exist.
        return out
    if c.error_mm is None:
        return out
    if c.error_mm > ERROR_DEGRADED_MM:
        out.append(Finding(
            Verdict.BLOCKED,
            "Holes would land nowhere near the shot.",
            "The mapping from camera to screen is wrong. Drawing holes from it "
            "would look like aiming error and teach the wrong lesson.",
            "Press G to recalibrate. If it fails again the camera has probably "
            "been moved since it was last calibrated.",
            f"{c.error_mm:.1f}mm median error, worst "
            f"{c.worst_error_mm or c.error_mm:.1f}mm",
        ))
    elif c.error_mm > ERROR_READY_MM:
        out.append(Finding(
            Verdict.DEGRADED,
            "Holes will be a few millimetres out.",
            "Good enough to train with, not good enough to judge a tight group "
            "by.",
            "Press G to recalibrate when convenient.",
            f"{c.error_mm:.1f}mm median error",
        ))
    return out


def _rate_finding(c: SelfCheck) -> Finding | None:
    """The headline number: of the shots it fired at itself, how many landed."""
    if not c.fired:
        return None
    rate = c.rate
    if rate >= RATE_READY:
        return None
    missed = c.fired - c.detected
    if rate < RATE_DEGRADED:
        return Finding(
            Verdict.BLOCKED,
            f"Only {c.detected} of {c.fired} test shots were detected.",
            "These were fired by the app at known positions, with the same "
            "brightness spread as a real trigger pull. A human shooting now "
            "would lose about the same fraction.",
            "See the reasons listed above; fix those first, then press A to "
            "run this check again.",
            f"{rate * 100:.0f}% detected, {missed} missed",
        )
    # A MIDDLING RATE DEFERS TO THE PROBE. See `_probe_answered`. A LOW
    # rate (below RATE_DEGRADED) still BLOCKS above: that is the plumbing
    # failing, not a dim test dot, and no probe reading excuses it.
    if _probe_answered(c):
        return None

    # THE THIRD PLACE THIS ADVICE LIVED, and the one that proved the sweep
    # was worth writing. Two branches above were fixed by hand after he said
    # "this is the max darkness for my comfort"; this one was not, because
    # nobody thinks of the headline rate as room advice. The sweep found it
    # in its first run, across 10,080 generated verdicts.
    return Finding(
        Verdict.DEGRADED,
        f"{c.detected} of {c.fired} test shots were detected.",
        "Usable, but some real shots will be missed -- most likely the dim "
        "ones, and most likely on the far side of the screen.",
        # ONE ACTION, THE SAME EITHER WAY. The bright-room variant said
        # "after dimming the room", which he cannot do -- see above.
        "Press X to re-measure the exposure, then A to run this check again.",
        f"{rate * 100:.0f}% detected",
    )


def assess(c: SelfCheck) -> Report:
    """Turn the self-check's measurements into a verdict and a list of actions.

    Order matters: the camera faults come first, because nothing downstream
    means anything while the camera is ignoring its exposure or cannot see the
    screen, and advice about dimming a room is actively misleading in that
    state. This project has emitted exactly that misleading advice before.
    """
    findings: list[Finding] = []
    findings += _camera_findings(c)
    # A camera that cannot see the screen makes every other number meaningless.
    if any(f.verdict is Verdict.BLOCKED for f in findings):
        return Report(Verdict.BLOCKED, findings,
                      "NOT READY - the camera itself has to be fixed first.")

    findings += _speed_findings(c)
    findings += _signal_findings(c)
    findings += _accuracy_findings(c)
    rate = _rate_finding(c)
    if rate is not None:
        findings.insert(0, rate)

    if c.audio_ok is False:
        findings.append(Finding(
            Verdict.DEGRADED,
            "The gunshot sound did not play.",
            "Detection still works; you simply will not hear a hit.",
            "Check the system's default playback device, and that the sound "
            "checkbox is ticked.",
        ))

    if any(f.verdict is Verdict.BLOCKED for f in findings):
        verdict = Verdict.BLOCKED
        summary = "NOT READY - do not start shooting yet."
    elif findings:
        verdict = Verdict.DEGRADED
        summary = "USABLE, with the reservations below."
    else:
        verdict = Verdict.READY
        summary = "READY - shoot."

    if verdict is Verdict.READY and c.fired:
        summary = (f"READY - {c.detected} of {c.fired} test shots detected, "
                   f"{c.error_mm:.1f}mm median error."
                   if c.error_mm is not None else
                   f"READY - {c.detected} of {c.fired} test shots detected.")
    return Report(verdict, findings, summary)
