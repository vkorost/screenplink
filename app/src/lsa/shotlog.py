"""The words in the log about a shot: what the detector saw, what it almost saw.

Pure functions, no Qt and no app, so the format is tested by feeding them real
`Shot` objects. Plan 02 steps 9 and 10.

His two minutes of shooting are the scarce resource, and the question they must
answer completely is, for each trigger pull: was it heard, was it seen, where
was it drawn, how far from the nearest target, and -- when it was heard and not
seen -- what did the detector almost see. Every line here exists so that the
answer is in the log and does not need a second string fired to find out.
"""

from __future__ import annotations

import math


def shot_evidence(shot, bar: float, mm_per_cam_px: "float | None") -> str:
    """Per-shot evidence about the CENTROID, for a hole that disagrees with the dot.

    Asked of the owner: when the dot he sees and the hole disagree, is it one
    direction or scattered? He said "scattered, no pattern". Scatter points away
    from the mapping and toward the first-frame centroid of a dim pulse, so what
    matters per shot is how dim the pulse was, how many frames it lasted, and how
    far the first frame sat from the brightest one.
    """
    run = shot.members
    if not run:
        return f"(no per-frame record; {shot.frames} frames, peak {shot.peak})"
    n = len(run)
    first = run[0]
    best = max(run, key=lambda d: d.peak)
    reported = next((i for i, d in enumerate(run)
                     if d.x == shot.x and d.y == shot.y), None)
    drift = math.hypot(first.x - best.x, first.y - best.y)
    mm = f" (~{drift * mm_per_cam_px:.1f}mm)" if mm_per_cam_px else ""
    pct = 100.0 * first.peak / best.peak if best.peak > 0 else 0.0
    which = f"{reported + 1}/{n}" if reported is not None else f"?/{n}"
    ratio = f"{best.peak / bar:.1f}x" if bar > 0 else "n/a"
    return (f"centroid from frame {which}; first-frame peak {first.peak} of "
            f"best {best.peak} ({pct:.0f}%); drift first-vs-brightest "
            f"{drift:.1f} cam px{mm}; best/bar {best.peak}/{bar:.0f} = {ratio}")


def mic_state(blocks: int, deaf_for: float) -> str:
    """Why a microphone level means what it means, as a bracketed tag (or "").

    A level of 0.000 used to mean three different things and read as one: a
    silent room, a microphone that was not delivering audio, and a microphone the
    app had muted itself (~0.8 s after its own gunshot). Only the first is
    evidence about the gun. The other two make a trigger pull look
    seen-and-not-heard, which reads as a phantom in the one channel the gun
    test rests on, so each is named.
    """
    if blocks <= 0:
        return " [NO AUDIO within 0.25 s: the stream was not delivering - this is not a silent room]"
    if deaf_for > 0.0:
        return (f" [DEAF: the app's own gunshot had the microphone muted for another "
                f"{deaf_for:.2f}s, so no click could be counted here - not evidence of a phantom]")
    return ""


def audio_tally(corroborator) -> str:
    """Running totals of the microphone against the camera since launch."""
    c = corroborator
    return (f"[audio so far: heard {c.heard}, seen {c.seen}, both {c.both}, "
            f"heard-not-seen {c.heard_not_seen}, seen-not-heard {c.seen_not_heard}]")


# The branches a heard-but-not-seen pulse can fall into, and what each means.
# The lead sentence of the log line AND the note on the target screen are both
# chosen FROM the branch, so neither can assert a conclusion the evidence next
# to it contradicts (RULES C4: advice the measurement contradicts).
MISSED_NO_FRAMES = "no_frames"      # the camera delivered nothing around the click
MISSED_OFF_PANEL = "off_panel"      # a real blob, outside the screen mask
MISSED_TOO_SHORT = "too_short"      # cleared the bar, lasted too few frames
MISSED_AMP_AREA = "amp_area"        # a blob on the panel, too small for its region's gain
MISSED_HALO = "halo"                # seen, and folded into a brighter nearby event's halo
MISSED_TOO_BIG = "too_big"          # a blob over max_area that was too faint to be a pulse
MISSED_FAINT = "faint"              # on the panel, under min_peak
MISSED_NOTHING = "nothing"          # no blob of any kind, anywhere


def classify_missed_pulse(summary: "dict | None", threshold: float) -> str:
    """Which of the eight branches the evidence selects.

    `summary` is `FrameEvidence.summarise`; `threshold` is the detector's BLOB
    threshold (a pixel is part of a blob only when its score is ABOVE it).

    The off-screen reading is a MAXIMUM over every pixel outside the screen
    mask, so sensor noise alone keeps it off zero: a blown-out window beside
    the monitor reads 4-6 once the background model has absorbed it, while a
    dim red dot sitting on the bezel reads 89. Testing it against zero made
    the "saw nothing at any level" branch unreachable on real hardware, so it
    is judged against the blob threshold instead: at or below it, it is noise.

    The branches are tested in this order, and the first that matches wins:

      1. no_frames   there is no summary at all (the camera delivered nothing).
      2. off_panel   an off-screen blob (above the blob threshold) at least as
                     bright as the brightest faint on-panel one: the brightest
                     thing in view is the pulse.
      3. too_short   it cleared the bar, so a real detection existed, and was
                     discarded for lasting too few frames.
      4. halo        a candidate folded into a brighter nearby event's halo.
      5. amp_area    a blob the size gate threw away where the sensitivity map
                     amplifies the score.
      6. too_big     a blob over `max_area` that was too faint to be a pulse.
      7. faint       an on-panel blob that stayed under min_peak.
      8. nothing     no blob of any kind, anywhere.

    halo, amp_area and too_big all sit AHEAD of faint on purpose: each rejects
    a blob before its peak reaches `dim_peak`, or reports on a blob the
    detector chose to fold in, so left unnamed they fell through to "faint" or
    "saw nothing" and blamed the camera for something the detector decided.
    They also sit behind too_short, which is the only one of the three that
    means a full detection existed.

    A second off_panel return follows the faint test (an off-screen blob
    dimmer than a faint on-panel one). As the code stands it cannot be reached:
    reaching it needs dp == 0, and then the first off_panel test has already
    matched. It is a defensive fallback, not a ninth branch.
    """
    if summary is None:
        return MISSED_NO_FRAMES
    dp, op, sd = summary["dim_peak"], summary["off_peak"], summary["short_delta"]
    off_is_blob = op > threshold
    if off_is_blob and op >= dp:
        return MISSED_OFF_PANEL
    if sd > 0:
        return MISSED_TOO_SHORT
    # A candidate folded into a brighter event's halo was SEEN. Unnamed it
    # fell through to "faint" or "saw nothing", both of which blame the
    # camera for something the detector chose.
    if summary.get("halo"):
        return MISSED_HALO
    # The size gate throws a blob away BEFORE its peak exists, so dp stays 0
    # and, unnamed, this fell through to "saw nothing at any level".
    if summary.get("amp"):
        return MISSED_AMP_AREA
    # A blob over `max_area` and under `oversize_min_peak` was thrown away
    # before its peak entered `dim_peak`, so unnamed it fell through to "saw
    # nothing at any level".
    if summary.get("big"):
        return MISSED_TOO_BIG
    if dp > 0:
        return MISSED_FAINT
    if off_is_blob:
        return MISSED_OFF_PANEL
    return MISSED_NOTHING


_LEADS = {
    MISSED_NO_FRAMES: "no camera frames were recorded around the click",
    MISSED_OFF_PANEL: "the dot was seen OUTSIDE the screen area: the shot missed "
                      "the panel, or the camera has moved",
    MISSED_TOO_SHORT: "the pulse was seen but lasted too few frames to count",
    MISSED_AMP_AREA: "a blob was seen where the shot would be, but it was too small "
                     "for how much the sensitivity map amplifies that part of the "
                     "screen, so the size gate rejected it",
    MISSED_TOO_BIG: "a very large blob was seen where the shot would be, but it was "
                    "too faint to be a laser pulse of this size, so the size ceiling "
                    "rejected it",
    MISSED_HALO: "a blob was seen, but it sat in the glow around a much brighter "
                 "flash and was counted as part of that flash",
    MISSED_FAINT: "the dot was on the panel but too faint to count",
    MISSED_NOTHING: "the camera saw nothing at any level, on or off the screen: "
                    "the shot missed beyond the camera's view, or the camera "
                    "cannot see that part of the panel",
}


def describe_missed_pulse(summary: "dict | None", bar: float, threshold: float) -> str:
    """What the detector almost saw around a click that produced no shot.

    The sentence it opens with is CHOSEN BY the branch the evidence selects
    (`classify_missed_pulse`), and every reading is printed beside the
    reference it is judged against, so "off-screen blob 5" can never again
    assert a blob where 5 is the noise floor.
    """
    branch = classify_missed_pulse(summary, threshold)
    lead = _LEADS[branch]
    if summary is None:
        return (f"{lead} (the camera had stalled, or the click is older than "
                f"the 3 s buffer)")
    fp, dp = summary["frame_peak"], summary["dim_peak"]
    op, sd = summary["off_peak"], summary["short_delta"]
    bits = []
    amp = summary.get("amp")
    if amp:
        x, y = amp["pos"]
        bits.append(f"size-gated blob of {amp['area']:.0f}px at cam ({x:.0f},{y:.0f}); "
                    f"{amp['need']:.0f}px needed where the map amplifies "
                    f"{amp['n']:.1f}x")
    big = summary.get("big")
    if big:
        x, y = big["pos"]
        bits.append(f"oversize blob {big['peak']} of {big['area']:.0f}px at cam "
                    f"({x:.0f},{y:.0f}), too faint for its size")
    halo = summary.get("halo")
    if halo:
        x, y = halo["pos"]
        bits.append(f"halo-folded blob {halo['peak']} at cam ({x:.0f},{y:.0f})")
    if dp > 0:
        x, y = summary["dim_pos"]
        bits.append(f"brightest sub-threshold blob {dp} at cam ({x:.0f},{y:.0f}) "
                    f"against a bar of {bar:.0f}")
    else:
        bits.append("no blob was rejected as too faint")
    if op > threshold:
        bits.append(f"off-screen blob {op} (above the blob threshold {threshold:.0f})")
    else:
        bits.append(f"off-screen reading {op} (at or below the blob threshold "
                    f"{threshold:.0f}: noise, not a blob)")
    bits.append(f"too-short rejections +{sd}")
    bits.append(f"frame peak {fp} (blob threshold {threshold:.0f})")
    return f"{lead}. Evidence: " + "; ".join(bits)


def missed_pulse_note(summary: "dict | None", bar: float, threshold: float) -> str:
    """The same finding for the target screen, read from several metres away.

    Short, plain, and chosen by the same branch as the log line. No advice
    about the room: the app already holds the light measurement, and a dark
    room is the best condition it can have (RULES C4).
    """
    branch = classify_missed_pulse(summary, threshold)
    if branch == MISSED_FAINT:
        return (f"HEARD a shot: on the panel, but too faint "
                f"({summary['dim_peak']} of {bar:.0f})")
    if branch == MISSED_HALO:
        return (f"HEARD a shot: seen, but folded into the glow of a brighter "
                f"flash ({summary['halo']['peak']})")
    if branch == MISSED_AMP_AREA:
        a = summary["amp"]
        return (f"HEARD a shot: blob too small here "
                f"({a['area']:.0f}px of {a['need']:.0f}px)")
    if branch == MISSED_TOO_BIG:
        b = summary["big"]
        return (f"HEARD a shot: a very large faint patch ({b['area']:.0f}px, "
                f"peak {b['peak']}), not counted")
    if branch == MISSED_OFF_PANEL:
        return "HEARD a shot: the dot was seen OUTSIDE the screen"
    if branch == MISSED_TOO_SHORT:
        return "HEARD a shot: seen, but the flash was too short to count"
    if branch == MISSED_NO_FRAMES:
        return "HEARD a shot: the camera was not sending frames"
    return "HEARD a shot: the camera saw nothing. Off the screen?"
