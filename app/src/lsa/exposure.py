"""Choose the exposure by how many of HIS shots would register at it.

WHY THIS MODULE EXISTS
----------------------
The exposure is the single most consequential setting in this app, and until
now it was decided by a constant:

    MEAN_CEILING = 40.0     # reject an exposure whose frame mean exceeds this

Measured on 2026-10-15, that constant cost half a string. Afternoon light
through the window beside the monitor pushed the frame mean up, so the rule
stopped the camera down three stops, and his own trigger pulls went from

    at -5:  peak median 74, floor 1-7,  bar 7-10   -> about 10x margin, worked
    at -8:  peak median 18, floor 4-9,  bar 10-15  -> about 1.2x, half lost

A pulse flickers 4.4x between its dimmest and brightest frame, so at 1.2x
margin the dim half of every string lands under the bar. That is exactly what
he reported: "some of them are seen much more clearly than the other ones".

The mean is a PROXY. It correlates with "is the background dark enough" in
the room it was tuned in, and these notes already record what proxies do in
other rooms -- choosing an exposure on mean brightness is what broke
detection the night the room was darkened. This project's standing rule is
measure the quantity the detector actually consumes, and the detector
consumes one thing: a chroma peak, against a bar.

WHAT REPLACES IT
----------------
He asked for it directly:

    "You should be able to emulate the exact thing that my gun does and find
     the right combination of threshold and brightness so that your code is
     auto-calibrated every single time the program starts."

So: the screen draws its own red dot at each candidate exposure, the camera
photographs it, and the two numbers that decide everything are measured
rather than assumed --

    dot_peak(E)        how brightly a known red dot images at exposure E
    floor(E)           what the background scores at exposure E, which sets
                       the bar the detector will use

The emulated dot is not a laser, and this module never pretends it is. A
screen EMITS its dot; a laser REFLECTS off matte coating. What transfers is
the RESPONSE TO EXPOSURE -- both are light arriving through the same lens
onto the same sensor during the same integration window, so doubling the
integration time doubles both. The emulated dot measures the shape of that
curve; his 575 logged trigger pulls supply where on it a real shot sits.

That is the whole trick, and it is why this can be done with nobody at the
firing line.
"""

from __future__ import annotations

from dataclasses import dataclass

# HIS GUN, AS MEASURED. 575 logged trigger pulls, 2026-10-15, two LaserLyte
# LT-TTL at varying distances on this panel:
#
#     peak   min 31   p10 41   median 85   p90 180   max 253
#
# The deciles below interpolate between those measured percentiles, which is
# an assumption about the SHAPE of the distribution and nothing more -- the
# five anchors are real and the points between them are a smooth fill. That
# matters because the thing being computed is "what fraction clears the bar",
# and near the bar the shape is what decides it.
#
# Interpolated geometrically rather than linearly: brightness here is the
# product of several multiplicative effects (distance, angle, the slice of
# the PWM cycle the frame caught), and such things are log-normal, not normal.
GUN_PEAK_DECILES = (33.0, 41.0, 51.0, 63.0, 74.0, 85.0, 103.0, 132.0,
                    180.0, 231.0)

# The exposure those shots were fired at. Everything below scales relative to
# this one anchor, so it must be the exposure the measurement was taken at --
# not a round number and not today's choice.
GUN_ANCHOR_EXPOSURE = -5.0

# A pulse must clear the bar by this much before it is worth calling safe.
# Not 1.0: a shot exactly at the bar is a coin toss, and the flicker means
# the same trigger pull can land either side of it on consecutive frames.
SAFETY = 1.15

# The probe auto-ranges its dot until it reads at least this many times its
# own bar (times SAFETY), so the dot it judges with is not noise on top of
# noise. The ladder only climbs, in steps of at most 1.6x.
RANGE_OVER_BAR = 2.0


@dataclass
class Probe:
    """One candidate exposure, as measured by photographing the screen's dot."""

    exposure: float
    dot_peak: float          # chroma peak of the emulated dot, inside the ROI
    # The WORST chroma anywhere in the ROI with no dot on screen -- a max, not
    # a percentile. It has to be the max, because that is what the app's own
    # noise-floor measurement reports ("noise floor over 143 frames: peak 2")
    # and what min_peak is then derived from. A p99 reads lower, which would
    # predict a bar the detector never uses and quietly flatter every long
    # exposure.
    floor: float
    mean: float = 0.0        # ROI brightness, kept for the log and nothing else
    # The same emulated dot, at the same brightness, photographed at the
    # STRONGEST end of the panel. None when the probe did not measure it, in
    # which case the saturation constraint below cannot be applied and says so.
    strong_peak: "float | None" = None
    # The fraction of full red the probe's dot was drawn at (log only).
    brightness: float = 0.0

    def bar(self) -> float:
        """The min_peak the detector would end up using at this exposure."""
        return min_peak_for(self.floor)


# THE ABSOLUTE MINIMUM FOR THE BAR, so a near-zero floor does not put
# it on top of the noise itself. NOT the single-frame laser/noise
# boundary, which is 30 and lives on CameraLab -- merging the two cost
# him half a string on 2026-10-16.
#
# Across two sessions at two different exposures the point where this gun
# stops looking like this room sat at 30, both times -- it is a property of
# a 650nm dot against this panel, not of the room. And over 575 of his
# logged trigger pulls the DIMMEST shot he has ever fired measured 31,
# with p10 at 41 and a median of 85. So a bar of 30 costs him nothing he
# has ever shot.
# REVERTED to 8.0 on 2026-10-16, within the hour, because raising it live
# cost him half a string.
#
# The reasoning for 30 was sound and the evidence was real: a static object
# on the panel scored 8-16 and fired the gunshot sound at him five times
# while he was not shooting, and his dimmest logged shot ever measured 31.
# But raising the bar ALSO moved what the exposure probe chooses -- a
# higher bar makes every candidate look thinner, the probe picked a darker
# exposure, and his real shots went from 106-127 down to 77-88 with the dim
# half falling under. Measured by the microphone, which does not care what
# the camera thinks: 10 trigger pulls heard with no detection, 8 detected.
#
# The lesson is not about the number. A phantom is a STATIC OBJECT -- the
# detections sat within 0.07 px of each other, seconds apart, which no
# human hand can do -- and the honest discriminator is that, not
# brightness. Raising a brightness bar to reject it is the proxy mistake
# this project has now made five times: it cost real shots, and it would
# cost them again on the 75" TV where everything images dimmer.
MIN_PEAK_FLOOR = 8.0


def min_peak_for(floor: float) -> float:
    """The bar the detector will use, given a measured noise floor.

    ONE FUNCTION, because this rule lived in two places and they drifted
    apart the moment one changed -- which is Part E of RULES, and it cost
    him a live string on 2026-10-16.

    The app floored the bar at the laser/noise boundary to stop a static
    object on the panel firing the gunshot sound at him while he was not
    shooting. The exposure probe kept its own copy of the old formula, so
    it went on choosing an exposure against a bar of 8, picked the darkest
    candidate on the table, and reported "100% of his shots clear the bar,
    4.1x margin" -- while the detector it was choosing for used 30, and the
    emulated dot measured 22. The real margin came out 1.97.

    Multiplicative above the boundary, because "floor + 15" is punitive on
    a quiet screen: at a measured floor of 22 it sat at 37 while real
    far-side shots scored 42, 44 and 50 and everything that vanished scored
    under 37.
    """
    return max(MIN_PEAK_FLOOR, floor * 1.3, floor + 6.0)


# THE PROBE MUST NOT REPORT A NUMBER ITS OWN DOT CONTRADICTS (2026-10-17).
#
# `scale = probe.dot / anchor.dot` is exactly 1.0 for the anchor, so for the
# anchor the dot cancels and "100% of his shots clear the bar, 6.1x" was
# nothing but his logged median (85) over the bar (14) -- it never looked at
# the dot just measured. A real line from a harness launch that then detected
# 0 of 12:
#
#     probe: chose exposure -5 - 100% of his measured shots clear the bar here
#            (dot 10, floor 8, bar 14, predicted margin 6.1x ...)
#
# A dot of 10 against a bar of 14 is the app's own emulated shot BELOW ITS OWN
# THRESHOLD: that reading predicted the failure and was discarded. So the dot
# is a gate on every prediction, anchor included: a candidate whose measured
# dot does not clear its own bar (with SAFETY) is not "100%" of anything. Its
# rate is 0 and its margin is its OWN margin, dot over bar, and the table
# says so. A dot at or under the bar also means the ratio is noise on top of
# noise, so if the ANCHOR's dot fails, no scaled prediction is trustworthy
# and every candidate is reported as unmeasurable. The direction is the
# conservative one: it can only make a candidate look worse than it is.
def own_margin(probe: Probe, bar: float | None = None) -> float:
    """The dot just measured over the bar it must clear. No distribution, no
    anchor: the one number the measurement itself supports."""
    use = probe.bar() if bar is None else bar
    return probe.dot_peak / max(1.0, use)


def dot_clears_bar(probe: Probe, bar: float | None = None) -> bool:
    """Does the dot photographed at this exposure clear this exposure's bar?"""
    use = probe.bar() if bar is None else bar
    return probe.dot_peak >= use * SAFETY


def trusted(probe: Probe, anchor: Probe, bar: float | None = None,
            anchor_bar: float | None = None) -> bool:
    """True when both the candidate's own dot and the anchor's clear their
    own bars, so the ratio between them is a measurement and not noise."""
    return dot_clears_bar(probe, bar) and dot_clears_bar(anchor, anchor_bar)


def predicted_rate(probe: Probe, anchor: Probe,
                   deciles: tuple[float, ...] = GUN_PEAK_DECILES,
                   bar: float | None = None,
                   anchor_bar: float | None = None) -> float:
    """What fraction of his measured shots would clear the bar at `probe`.

    0.0 when the probe's own dot (or the anchor's) does not clear its own bar
    -- see the block above `own_margin`.

    `anchor` is the probe taken at GUN_ANCHOR_EXPOSURE -- the exposure his
    575 pulls were logged at. The emulated dot is measured at both, and the
    ratio between them says how much brighter or dimmer a real shot would be
    at this candidate. The dot's absolute brightness cancels, which is the
    point: it never has to equal a laser's.
    """
    if anchor.dot_peak <= 0:
        return 0.0
    if not trusted(probe, anchor, bar, anchor_bar):
        return 0.0
    scale = probe.dot_peak / anchor.dot_peak
    bar = (probe.bar() if bar is None else bar) * SAFETY
    # A real dot saturates the sensor, an emulated one need not, so the
    # scaling cannot be trusted past full scale. Clamping is the conservative
    # direction: it can only make a long exposure look worse than it is.
    clears = sum(1 for p in deciles if min(255.0, p * scale) >= bar)
    return clears / len(deciles)


def predicted_margin(probe: Probe, anchor: Probe,
                     deciles: tuple[float, ...] = GUN_PEAK_DECILES,
                     bar: float | None = None,
                     anchor_bar: float | None = None) -> float:
    """Median predicted shot peak over the bar. The number worth reporting.

    `bar` overrides the probe's own, so several candidates can be compared
    against ONE bar. See `common_bar` for why that matters.

    When the dot does not clear its own bar (or the anchor's does not) the
    scaled prediction is not trustworthy, so this returns the probe's OWN
    margin, dot over bar -- a number below SAFETY that cannot be mistaken
    for a healthy 6x.
    """
    if anchor.dot_peak <= 0 or not deciles:
        return 0.0
    use = probe.bar() if bar is None else bar
    if not trusted(probe, anchor, bar, anchor_bar):
        return own_margin(probe, use)
    scale = probe.dot_peak / anchor.dot_peak
    mid = sorted(deciles)[len(deciles) // 2]
    return min(255.0, mid * scale) / max(1.0, use)


# ---------------------------------------------------------------------------
# THE STRONG END MUST NOT SATURATE (2026-10-17)
# ---------------------------------------------------------------------------
#
# Until now the probe measured ONE spot, deliberately the weakest region,
# because the far side of the panel was losing shots all week. Optimising an
# exposure for the dim end is exactly what blows out the bright end, and the
# bright end is the near side where he stands next to the camera.
#
# MEASURED on session-20261017-100406, 147 shots read back out of the RAW
# recording (capture.avi), not out of the detector's score, at exposure -3:
#
#     band    n   raw chroma p50/p90/max   raw RED channel clipped (>=250)
#     LEFT    39   97 / 105 / 111           ~19 of 39 (49%)   <- CORRECTED
#     MIDDLE  74   85 /  99 / 110            7 of 74  ( 9%)
#     RIGHT   34   62 /  78 / 107            0 of 34
#
#     clip rate by raw chroma:  <90 -> 6..29%   90-100 -> 34%   >=100 -> 56%
#
# CORRECTION (2026-10-17, review): the first pass of this table said the LEFT
# band clipped 32 of 39 (82%). The reviewer re-derived it independently from
# the same recording and got 49%; the 82% is withdrawn. I (the implementer
# of this round) did NOT re-read the 487 MB raw AVI, so 49% is the reviewer's
# figure, and "~19 of 39" is just 49% of 39. What did reproduce independently
# is the knee at chroma 90, which is the only number the rule below uses from
# this table. The other bands are unchanged.
#
# A clipped core has lost where its centre was, and its bloom clears the bar
# on its own: a wrong hole, which unlike a missed shot looks exactly like bad
# aim. Raw chroma plateaus near 110 because green and blue rise as red clips,
# so the LEFT median of 97 is a LOWER BOUND on what the sensor was asked for.
#
# ** A STOP IS NOT A FACTOR OF TWO HERE. ** Above about -6 this camera is
# frame-time limited, and the probe's own tables show it:
#
#     10:04:09  -3: dot 28   -4: dot 23   -5: dot 18      (0.64x over two stops)
#     08:43:41  -3: dot 38   -4: dot 37   -5: dot 32      (0.84x over two stops)
#
# i.e. 0.74x over two stops on average, about 1.16x per stop. "Drop one stop
# and it halves" is the assumption everyone brings and it is wrong on this
# camera: getting out of saturation takes MORE stops than intuition says,
# which is also why the probe's +-1 stop window may need several launches.
#
# The rule, every number from the table above:
#
#   SAT_CHROMA     90   the knee of the clip-rate table.
#   FLICKER        1.2  p90/p50 of the two UNCLIPPED bands: 99/85 = 1.16 and
#                       78/62 = 1.26. (LEFT's 105/97 is compressed by the
#                       ceiling and is not used.)
#   so the predicted strong-end MEDIAN must be <= 90 / 1.2 = 75, which puts
#   its p90 at the knee.
#
#   STRONG_ANCHOR_MEDIAN  72  the strong-end gun median at the ANCHOR
#       exposure (-5): 97 measured at -3, scaled back two stops by the mean
#       response above (0.74). MEASURED: the 97 and the two tables.
#       EXTRAPOLATED: that the strong end responds like the centre where the
#       tables were taken, and that the gun scales like the emulated dot. No
#       recording holds a second exposure, so there is nothing to check it
#       against. 97 is a lower bound on what the sensor was asked for, so the
#       true anchor median may be higher than 72.
#
# ** THE PAIR (72 against 75) IS FITTED, NOT DERIVED. **
# Each factor above has more than 10% uncertainty (the 0.74 response alone
# spans 0.64-0.84, i.e. 62-82 for the anchor median), and the decision is
# 72 against 75: a 4% gap between two quantities each carrying more than that
# on their own. The numbers are those that make -5 come out; they were not
# computed and then found to give -5. The two tables of the response block
# above, run through `choose_detailed` (strong column synthetic, 1.6x the weak
# end, as in the tests), give the chosen exposure for each anchor median A and
# limit L ('*' = A > L, so even the anchor is flagged "spread exceeded"):
#
#     10:04:09 table      L= 68  70  72  75  78  80  83  85  90
#       A=65                 -5  -5  -5  -5  -5  -5  -4  -4  -4
#       A=72                 -5* -5* -5  -5  -5  -5  -5  -5  -5
#       A=80                 -5* -5* -5* -5* -5* -5  -5  -5  -5
#     08:43:41 table
#       A=65                 -5  -5  -5  -5  -3  -3  -3  -3  -3
#       A=72                 -5* -5* -5  -5  -5  -5  -5  -4  -3
#       A=80                 -5* -5* -5* -5* -5* -5  -5  -5  -5
#
# Read it as: with A = 72, -5 is chosen without a warning for L in [72, 83) on
# both tables; the reviewer's own analysis puts the clean band at [75, 80],
# and 75 is inside both. If the true anchor median is 65, any L from 78 up
# lets the table of 08:43:41 pick -3 (the clipping exposure); if it is 80, L
# must be at least 80 or the anchor itself is flagged. L sits at the LOW edge
# of the band on purpose: an error in either number then tips towards the
# DARKER exposure, i.e. away from clipping, at the price of a spurious
# "spread exceeded" warning. That is the only sense in which it "errs safe".
#
# The earlier text here called the agreement with the 575-pull anchor
# exposure "corroboration, not coincidence". It is withdrawn: -5 was the
# target when the constants were chosen, so that -5 comes out is not evidence
# that the rule is right. What WOULD be: one recording at -5 (the clip rate
# of the LEFT band there, and the strong-end median itself).
#
# THE FAR SIDE IS NOT SAFE AT -5 EITHER. The tables above show the far side
# keeping a median margin of several times its bar at -5 and the p10 shot of
# his 575 at about 4.0x. The DIMMEST far-side shot on record scales to only
# 1.5-2.9x its bar, under this project's own 3x standard, and on the 155 real
# shots of session-20261017-100406 (peak >= 30 at -3) scaling by 0.64 / 0.74 /
# 0.84 puts 12 / 9 / 7 of them below SINGLE_FRAME_PEAK (30) at -5. Those are
# lower bounds: shots that never reached 30 at -3 are not in the 155.
SAT_CHROMA = 90.0
FLICKER = 1.2
STRONG_MEDIAN_LIMIT = SAT_CHROMA / FLICKER            # 75.0
STRONG_ANCHOR_MEDIAN = 72.0
# A dot reading at or above this has itself hit the ceiling: the ratio
# against the anchor is then meaningless and the candidate is refused.
DOT_CEILING = 250.0


def strong_end_median(probe: Probe, anchor: Probe) -> "float | None":
    """Predicted median gun chroma at the strong end of the panel, or None
    when the probe did not measure the strong end."""
    if probe.strong_peak is None or anchor.strong_peak is None:
        return None
    if anchor.strong_peak <= 0:
        return None
    # STRONG_ANCHOR_MEDIAN is the strong-end gun median AT -5. A stand-in
    # anchor at any other exposure (the window (-6,-7,-8) leaves -5 absent
    # and the brightest probe, -6, stands in) would inherit it although it
    # is darker, silently inflating the allowance. Unreachable from the
    # routine today -- it always adds -5 -- except when -5's photograph
    # fails, so it is refused here rather than assumed away.
    if abs(anchor.exposure - GUN_ANCHOR_EXPOSURE) > 1e-6:
        return None
    if probe.strong_peak >= DOT_CEILING:
        return 255.0
    return STRONG_ANCHOR_MEDIAN * probe.strong_peak / anchor.strong_peak


def saturates_strong_end(probe: Probe, anchor: Probe) -> bool:
    m = strong_end_median(probe, anchor)
    return m is not None and m > STRONG_MEDIAN_LIMIT


@dataclass
class Choice:
    best: "Probe | None"
    why: str
    # True when no exposure satisfies both ends, so something was given up.
    spread_exceeded: bool = False
    # The sentence for the shooter / the log when spread_exceeded is True.
    warning: str = ""
    strong_median: "float | None" = None
    # True when NO candidate's own measured dot cleared its own bar, so the
    # probe has no instrument to choose with. `best` is None then, `why` and
    # `warning` carry the numbers, and the caller keeps the exposure it has.
    unmeasurable: bool = False


def bars_for(probes: list[Probe]) -> dict[float, float]:
    """A bar per candidate, with the floor readings clamped MONOTONIC.

    MEASURED ON HIS BENCH, 2026-10-16, and this is why the rule exists:

        exposure   dot peak   floor   bar
            -4         43       15     20
            -5         57       20     26
            -6         22        5      8   <- chosen

    It chose the candidate with the DIMMEST DOT, because that candidate's
    three-frame noise reading came out low, which made its bar low, which
    made its ratio the best. The dot at -5 was two and a half times
    brighter and lost. He then fired a string into the exposure this picked
    and the microphone heard ten trigger pulls the camera never turned into
    shots.

    THE FIX IS NOT ONE SHARED BAR. I tried that first and a test caught it:
    in a LIT room the floor genuinely climbs with exposure -- measured 46,
    30, 16, 10, 8 from -3 to -7 -- and that climb is the entire mechanism
    that makes a bright room stop itself down without a constant. Flatten
    it and the app runs away to the brightest setting, which is the bug
    `MEAN_CEILING` was invented for and this module exists to replace.

    What is true, and is all that is needed: **noise cannot be LOWER at a
    longer exposure than at a shorter one.** More integration time cannot
    collect less noise. So a reading that dips below a darker neighbour's
    is measurement noise in a very short sample, and it is clamped up. The
    lit-room column above is already monotonic and passes through
    untouched.

    BE PRECISE ABOUT HIS COLUMN, because the first version of this comment
    was not and an outside review caught it. Sorted darkest-first his
    readings are 5 (at -6), 20 (at -5), 15 (at -4). The dip is the 15, and
    it is raised to 20. The 5 at -6 is NOT touched: nothing is darker than
    the darkest candidate, so there is nothing to clamp it against.

    What actually moved his decision from -6 to -5 was the seam fix above
    -- `bar()` now predicts the bar the detector really uses, and the
    `floor + 6` term took -6's bar from 8 to 11. The clamp matters for
    tables where the dip is in the middle, and it is cheap insurance; it
    is not what saved this one, and saying otherwise would leave the next
    person believing a fix that was not the fix.

    What then decides the winner is how brightly a known dot images, which
    is the thing actually being measured.
    """
    if not probes:
        return {}
    # Darkest (most negative) first: that is increasing exposure time.
    order = sorted(probes, key=lambda p: p.exposure)
    out: dict[float, float] = {}
    worst = 0.0
    for p in order:
        worst = max(worst, float(p.floor))
        out[p.exposure] = min_peak_for(worst)
    return out


def choose(probes: list[Probe],
           anchor_exposure: float = GUN_ANCHOR_EXPOSURE) -> tuple[Probe | None, str]:
    """`choose_detailed` without the extras; kept for existing callers."""
    c = choose_detailed(probes, anchor_exposure)
    return c.best, c.why


def choose_detailed(probes: list[Probe],
                    anchor_exposure: float = GUN_ANCHOR_EXPOSURE) -> Choice:
    """Pick the exposure, and say in one line why.

    The strong end of the panel is a CONSTRAINT, the weak end the objective
    subject to it: candidates whose predicted strong-end median exceeds
    STRONG_MEDIAN_LIMIT are set aside before the objective below runs. If
    that leaves nothing, or leaves only candidates that cost the far side,
    the least-saturating is taken and the result says so. A wrong hole is
    worse than a missed shot: a miss is visible as nothing, a wrong hole is a
    lie the shooter cannot tell from bad aim.

    The anchor is whichever probe sits at `anchor_exposure`; if it was not
    measured, the brightest probe stands in and the result is relative rather
    than absolute -- still the right ORDERING, which is all choose() needs.
    """
    if not probes:
        return Choice(None, "nothing was measured")
    anchor = next((p for p in probes if abs(p.exposure - anchor_exposure) < 1e-6),
                  None)
    if anchor is None:
        anchor = max(probes, key=lambda p: p.exposure)

    bars = bars_for(probes)
    abar = bars[anchor.exposure]
    anchor_true = abs(anchor.exposure - anchor_exposure) < 1e-6
    scored_all = [(predicted_rate(p, anchor, bar=bars[p.exposure],
                                  anchor_bar=abar), p) for p in probes]

    # NO INSTRUMENT: no candidate's own dot clears its own bar (or the
    # anchor's does not, so no ratio is trustworthy). Say so with the numbers
    # and choose nothing -- a "100%" built from the logged distribution alone
    # is the number the dot contradicted.
    ok = [p for p in probes if trusted(p, anchor, bars[p.exposure], abar)]
    if not ok:
        cells = "; ".join(
            f"{p.exposure:g}: dot {p.dot_peak:.0f} vs bar {bars[p.exposure]:.0f} "
            f"({own_margin(p, bars[p.exposure]):.1f}x, needs {SAFETY:.2f}x)"
            for p in sorted(probes, key=lambda q: -q.exposure))
        what = ("the anchor's own dot is not above its bar, so no ratio "
                "against it means anything" if not dot_clears_bar(anchor, abar)
                else "no candidate's own dot is above its bar")
        warn = (f"The exposure probe cannot choose: {what}. Measured dot "
                f"against the bar it must clear: {cells}. The emulated shot is "
                f"below the app's own threshold, so a prediction built on it "
                f"would be a guess. Keeping the exposure as it is.")
        return Choice(None, "UNMEASURABLE - " + warn, False, warn, None, True)
    free_best_rate = max(r for r, _p in scored_all)

    # THE CONSTRAINT: set aside every candidate that saturates the strong end.
    measured_strong = any(strong_end_median(p, anchor) is not None
                          for p in probes)
    survivors = [(r, p) for r, p in scored_all
                 if p in ok and not saturates_strong_end(p, anchor)]
    if not survivors:
        # Nothing keeps the bright end clean. Take the one that saturates
        # LEAST: a wrong hole is worse than a missed shot.
        least = min(ok, key=lambda p: (strong_end_median(p, anchor) or 0.0))
        m = strong_end_median(least, anchor)
        r = predicted_rate(least, anchor, bar=bars[least.exposure],
                           anchor_bar=abar)
        warn = (f"The panel's brightness spread is more than one exposure "
                f"can cover. Even the darkest setting tried ({least.exposure:g}) "
                f"images the near side at a predicted {m:.0f} against a limit "
                f"of {STRONG_MEDIAN_LIMIT:.0f} (the point where the sensor "
                f"starts to clip and holes land in the wrong place). Using "
                f"it anyway, because a wrong hole is worse than a missed "
                f"shot; it will step one stop darker at the next launch.")
        why = (f"NO candidate keeps the near side below the clipping point; "
               f"{least.exposure:g} is the least saturated (predicted "
               f"strong-end median {m:.0f}, limit {STRONG_MEDIAN_LIMIT:.0f}); "
               f"{r * 100:.0f}% of his measured shots clear the bar here "
               f"(dot {least.dot_peak:.0f}, floor {least.floor:.0f}, bar "
               f"{bars[least.exposure]:.0f}, own dot "
               f"{own_margin(least, bars[least.exposure]):.1f}x its bar)")
        return Choice(least, why, True, warn, m)
    scored = survivors
    best_rate = max(r for r, _p in scored)

    # Among everything that scores the best rate, take the widest MARGIN.
    #
    # The first version of this took the darkest instead, reasoning that once
    # every shot clears the bar, more exposure only risks the monitor ceasing
    # to be a ghost. Checked against what actually happened, that rule picks
    # -8 -- the exact exposure that lost him half a string on 2026-10-15. It
    # is wrong because a pulse is not one brightness: it flickers 4.4x
    # between its dimmest and brightest frame, so "the median clears the bar"
    # and "the string registers" are different claims, and the gap between
    # them is exactly the margin.
    #
    # Nothing is needed to stop this running away to the brightest setting,
    # because the floor already does it. In a lit room the background chroma
    # climbs at long exposures -- measured 22 at -3 against 6 at -8 -- the
    # bar climbs with it, and the margin falls. That is the MEAN_CEILING's
    # job being done by a measurement instead of a constant.
    # A MARGIN DIFFERENCE INSIDE THE MEASUREMENT NOISE IS NOT A DIFFERENCE.
    #
    # Measured on his bench, one launch, 2026-10-18:
    #
    #     -5   dot 113   floor 3   bar 9   margin  9.4x
    #     -6   dot 114   floor 2   bar 8   margin 10.7x   <- chosen
    #
    # The dot reads ONE COUNT apart across a full stop, because it is
    # auto-ranged into the sensor's linear range before the sweep -- so it
    # carries almost no information about the exposure, and the whole
    # decision falls onto the floor: 3 against 2. That floor is one frame
    # pair, where the app's own noise floor is a maximum over 180 frames,
    # and the same exposure has read 0, 2 and 3 on consecutive launches.
    #
    # It RATCHETS, because each launch centres its search on the last
    # choice. Across 2026-10-18 it walked -4, -4, -4, -5, -6, -6 with
    # nothing pulling it back, and at -6 his real shots scored 6 against a
    # background that scored 9, while the same gun measured 76-109 at -4/-5
    # that morning. The self test lost 5 of 12 dots, all on the right-hand
    # side, and the log named the reason: blobs "too small". The dot images
    # 269px at -6 against 564px at -5.
    #
    # So candidates within MARGIN_NOISE of the best are TIED, and the
    # existing tie-break -- prefer the larger exposure, i.e. the brighter --
    # decides. A genuinely better exposure still wins; one count of floor
    # noise no longer moves him a stop into the dark.
    #
    # 0.15 is FITTED to that noise, not guessed: bar = max(8, floor*1.3,
    # floor+6), so one count of floor moves the bar 8 -> 9 (12.5%) and the
    # margin about 11% the other way. 15% covers one count and little else.
    MARGIN_NOISE = 0.15
    _cands = [(predicted_margin(p, anchor, bar=bars[p.exposure],
                                anchor_bar=abar), p)
              for r, p in scored if r >= best_rate]
    _top = max(m for m, _ in _cands)
    best = max((p for m, p in _cands if m >= _top * (1.0 - MARGIN_NOISE)),
               key=lambda p: p.exposure)
    bar = bars[best.exposure]
    margin = predicted_margin(best, anchor, bar=bar, anchor_bar=abar)
    note = ("" if abs(bar - best.bar()) < 0.5 else
            f" -- its own floor reading of {best.floor:.0f} was below a "
            f"darker candidate's, which cannot happen physically, so it was "
            f"clamped up")
    sm = strong_end_median(best, anchor)
    strong_note = (f", near side predicted {sm:.0f} of {STRONG_MEDIAN_LIMIT:.0f} "
                   f"allowed" if sm is not None else
                   ", near side NOT measured so the clipping limit could not "
                   "be applied" if not measured_strong else "")
    if not anchor_true:
        strong_note += (f" (the {anchor_exposure:g} anchor was not measured, so "
                        f"the strong-end allowance, which is pinned to it, "
                        f"was not applied)")
    why = (f"{best_rate * 100:.0f}% of his measured shots clear the bar here "
           f"(dot {best.dot_peak:.0f} = {own_margin(best, bar):.1f}x its own "
           f"bar, floor {best.floor:.0f}, bar "
           f"{bar:.0f}{note}, predicted margin {margin:.1f}x{strong_note})")
    exceeded = best_rate < free_best_rate
    warn = ""
    if exceeded:
        warn = (f"The panel's brightness spread is more than one exposure "
                f"can cover: keeping the near side below clipping costs the "
                f"far side, {best_rate * 100:.0f}% of his shots clear the bar "
                f"against {free_best_rate * 100:.0f}% at the brighter "
                f"setting. Using {best.exposure:g}, because a wrong hole is "
                f"worse than a missed shot.")
    return Choice(best, why, exceeded, warn, sm)


def _strong_cell(p: Probe, anchor: Probe) -> str:
    m = strong_end_median(p, anchor)
    if m is None:
        return "   near side not measured"
    flag = "  SATURATES" if m > STRONG_MEDIAN_LIMIT else ""
    return f"   {p.strong_peak:>4.0f} / {m:>4.0f} (limit {STRONG_MEDIAN_LIMIT:.0f}){flag}"


def table(probes: list[Probe],
          anchor_exposure: float = GUN_ANCHOR_EXPOSURE) -> list[str]:
    """The measurement, laid out so a human can check the decision."""
    if not probes:
        return ["    nothing measured"]
    anchor = next((p for p in probes if abs(p.exposure - anchor_exposure) < 1e-6),
                  max(probes, key=lambda p: p.exposure))
    out = ["    exposure   dot peak   floor   bar   predicted shots seen"
           "   near-side dot / predicted gun median"]
    bars = bars_for(probes)
    abar = bars[anchor.exposure]
    for p in sorted(probes, key=lambda q: -q.exposure):
        b = bars[p.exposure]
        rate = predicted_rate(p, anchor, bar=b, anchor_bar=abar)
        if not trusted(p, anchor, b, abar):
            who = "its dot" if not dot_clears_bar(p, b) else "the anchor's dot"
            out.append(f"      {p.exposure:>5.0f}      {p.dot_peak:>6.0f}   "
                       f"{p.floor:>5.0f}  {b:>4.0f}     0%  "
                       f"(DOT BELOW BAR: {who} is {own_margin(p, b):.1f}x the "
                       f"bar, needs {SAFETY:.2f}x; no prediction)")
            continue
        out.append(f"      {p.exposure:>5.0f}      {p.dot_peak:>6.0f}   "
                   f"{p.floor:>5.0f}  {b:>4.0f}   {rate * 100:>3.0f}%  "
                   f"({predicted_margin(p, anchor, bar=b, anchor_bar=abar):.1f}x margin)"
                   + _strong_cell(p, anchor))
    return out
