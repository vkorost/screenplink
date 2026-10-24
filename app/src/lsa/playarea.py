"""Which part of the screen a shot is RELIABLY detected on.

Asked for directly, 2026-10-16, after a live string that worked everywhere
except the right-hand side:

    "every time the app starts, you will calibrate and then you draw strict
     borders to highlight the area that is supported by the current camera
     placement... where you cannot reliably place the shot on the right
     corner, you will mark that area off-limits. You will invite the user to
     concentrate his shots on the area of the monitor where you can reliably
     detect the shots."

    "advisory but clearly visible to the user, so please use a grey line to
     outline the play area inside the monitor; and that means no targets (or
     bouncing balls, etc.) go outside of that area. However if the shots
     detected on screen outside, draw the holes outside, just do not draw any
     targets"

So: a border, measured, drawn, and binding on TARGETS only. A detection
outside it is still a shot, still drawn, still counted -- silently discarding
a real hit is indistinguishable from a detector that does not work, and this
project has already paid for that twice.

WHAT IT IS MEASURED FROM, AND WHY NOT FROM THE SELF TEST
--------------------------------------------------------
The twelve-dot self test is the tempting source: it runs every launch, needs
no trigger pulls, and reports per-position detection. It is the WRONG source.
A screen EMITS its dot; a laser REFLECTS off the matte coating. Measured
2026-10-14 across left / middle / right: the emulated dot read peak 70/77/76
while his laser read 76/84/47. A border drawn from emitted light would put
the fence around nearly the whole panel and call the right third fine.

That is RULES C6 -- measure the thing, not a proxy -- and this project has
made that mistake with the white calibration markers (8% spread against the
laser's 2x), with frame mean brightness instead of chroma, and with the
mapping's Jacobian, which is also backwards. Four times is enough.

So the border is measured from REAL SHOTS, which is the only thing that
measures the laser. Consequences, stated rather than discovered:

- It cannot exist on a camera pose nobody has shot at yet. The caller must
  say so rather than invent one (RULES C4, C7).
- It is a property of the CAMERA POSE, like the mapping and the sensitivity
  map, and dies with them when the camera moves.
- Detected shots are survivors: a region that loses shots outright reports
  only the ones that got through, so its median peak OVERSTATES it. Hence a
  margin of 3x rather than 1x -- see MARGIN below.

Pure on purpose: measurements in, rectangle and sentences out, no Qt, no
camera, no lab. `tests/test_playarea.py` can therefore enumerate cases
instead of sampling them.
"""

from __future__ import annotations

from dataclasses import dataclass

# THE SWITCH. False = the app neither measures, persists, draws nor honours a
# play-area border; True restores all four. One line to bring it back.
#
# Turned OFF by the owner, 2026-10-16 20:00: "we need to address this border.
# Let's disable that border feature for now. It doesn't make a lot of sense
# as it is right now anyway." The evidence: it fenced off the bottom third on
#
#     19:52:55  bottom centre: 6 shots, median peak 24 against a bar of 14
#               = 1.7x, under the 3x a shot needs
#
# while every one of those six shots WAS detected and drew a hole. A margin
# PREDICTED that shots would be lost while the app watched them all arrive:
# a measurement beats a prediction of the same thing (RULES C5). If this
# returns it should be driven by measured LOSS (heard but not seen), not by a
# predicted margin.
#
# The camera-coverage rectangle that keeps targets where the camera can see
# is a different measurement (from the mapping, no shooting) and is NOT
# governed by this switch.
BORDER_ENABLED = False


# A 3x3 grid over the screen. The same grid the sensitivity map learns on,
# for the same reason: a ninth of a 42-inch panel is a region a shooter can
# aim at deliberately, and finer bins need more shots than one string gives.
BINS = 3

# A bin speaks for a ninth of the screen, so it must not speak on the
# strength of one reading. Measured 2026-10-14 19:43: a single stray scoring
# 23 was the only sample in its bin, the map concluded that region was 3.7x
# insensitive, amplified its noise, and the floor went 11 -> 70.
MIN_PER_BIN = 3

# A bin may be REPORTED weak on MIN_PER_BIN readings but is only FENCED on
# this many. The median of three shots is its middle shot: on 2026-10-16
# the border fenced "bottom left: 3 shots, median 18 = 1.2x" and "top
# right: 3 shots, median 42 = 2.8x" and shrank to 22% while he was hitting
# everywhere. Fencing takes away part of his screen, so it needs more
# evidence than speaking does.
MIN_PER_BIN_BLOCK = 6

# Below this many shots in total, nothing is claimed at all.
MIN_SHOTS = 9

# HOW MUCH MARGIN MAKES A REGION "RELIABLE".
#
# 3.0, and it is measured rather than chosen. This project's own records:
# "below 3x the far side of the screen starts losing shots; below 2x they
# are lost everywhere", and `selfcheck.MARGIN_READY` is 3.0 for the same
# reason. His pulse flickers over a measured 4.4x range (p90/p10), and his
# dimmest logged shot of 575 is 31 against a median of 85 -- so a region
# whose MEDIAN sits at 3x the bar still has its dim tail above it, while a
# region at 1.5x loses half of what he fires there.
#
# Checked against the string of 2026-10-16 13:26, 90 detections, bar 8:
#
#     left + middle third   median peak 88   11.0x   reliable
#     right third           median peak 19    2.4x   NOT reliable
#
# which is exactly what he reported from the firing line: "It mostly works
# except for the right side." A rule that reproduces his experience on his
# data, rather than one that sounds principled.
MARGIN = 3.0


@dataclass(frozen=True)
class Bin:
    """One cell of the grid, and what the shots landing in it measured."""

    col: int
    row: int
    shots: int
    median_peak: float

    def margin(self, bar: float) -> float:
        return self.median_peak / bar if bar > 0 else 0.0


@dataclass(frozen=True)
class PlayArea:
    """The reliable rectangle, in FRACTIONS of the screen.

    Fractions rather than pixels so the same measurement survives being
    drawn on a 42-inch monitor, a 75-inch TV and a projector -- RULES B5,
    the same reason nothing on the target screen is sized in pixels.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def whole_screen(self) -> bool:
        return (self.x0 <= 0.0 and self.y0 <= 0.0
                and self.x1 >= 1.0 and self.y1 >= 1.0)

    @property
    def fraction(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def to_dict(self) -> dict:
        return {"x0": float(self.x0), "y0": float(self.y0),
                "x1": float(self.x1), "y1": float(self.y1)}

    @staticmethod
    def from_dict(d) -> "PlayArea | None":
        if not isinstance(d, dict):
            return None
        try:
            area = PlayArea(float(d["x0"]), float(d["y0"]),
                            float(d["x1"]), float(d["y1"]))
        except (KeyError, TypeError, ValueError):
            return None
        if area.fraction <= 0.0:
            return None
        return area


def bins_from_shots(shots, screen_size, bins: int = BINS) -> list[Bin]:
    """Group real shots into the grid. `shots` are (screen_x, screen_y, peak).

    Binned in SCREEN space, not camera space, because the border has to be
    drawn on the screen and obeyed by targets placed on it. The sensitivity
    map bins the same shots in CAMERA space, which is right for its own job
    (it divides a camera-space score) and wrong for this one.
    """
    sw, sh = screen_size
    if not sw or not sh:
        return []
    cells: dict[tuple[int, int], list[float]] = {}
    for sx, sy, peak in shots:
        if peak <= 0:
            continue
        i = min(bins - 1, max(0, int(sx / sw * bins)))
        j = min(bins - 1, max(0, int(sy / sh * bins)))
        cells.setdefault((i, j), []).append(float(peak))
    out = []
    for (i, j), peaks in sorted(cells.items()):
        peaks.sort()
        out.append(Bin(col=i, row=j, shots=len(peaks),
                       median_peak=peaks[len(peaks) // 2]))
    return out


def _largest_clear_rect(blocked: set, bins: int) -> tuple[int, int, int, int] | None:
    """The biggest box of grid cells containing none of `blocked`.

    Brute force over every rectangle, which is 36 of them on a 3x3 grid.
    Ties are broken towards the box containing the centre cell, because the
    centre is where he is standing and where the signal is strongest; a tie
    resolved by iteration order would move the fence for no reason between
    two launches, and this project has already learned what an unexplained
    change between launches costs.
    """
    best = None
    best_key = None
    mid = bins // 2
    for i0 in range(bins):
        for i1 in range(i0, bins):
            for j0 in range(bins):
                for j1 in range(j0, bins):
                    if any((i, j) in blocked
                           for i in range(i0, i1 + 1)
                           for j in range(j0, j1 + 1)):
                        continue
                    area = (i1 - i0 + 1) * (j1 - j0 + 1)
                    has_mid = i0 <= mid <= i1 and j0 <= mid <= j1
                    key = (area, has_mid, -(i0 + j0))
                    if best_key is None or key > best_key:
                        best_key, best = key, (i0, j0, i1, j1)
    return best


def measure(shots, bar: float, screen_size, bins: int = BINS,
            min_per_bin: int = MIN_PER_BIN, min_shots: int = MIN_SHOTS,
            margin: float = MARGIN, min_per_block: int = MIN_PER_BIN_BLOCK) -> tuple[PlayArea | None, list[str]]:
    """Measure the reliable area. Returns (area or None, lines to report).

    None means "no border is claimed", which is NOT the same as "the whole
    screen is fine" -- the caller must say which. Three distinct cases
    return None, and each returns a different sentence, because a check that
    reports only a verdict can be re-run and not debugged (RULES C3):

    - too few shots to know anything;
    - evidence everywhere and every region reliable;
    - evidence everywhere and NO region reliable, which is a camera
      placement problem and must not be answered by fencing the shooter
      into nothing.
    """
    sw, sh = screen_size
    lines: list[str] = []
    cells = bins_from_shots(shots, screen_size, bins)
    total = sum(c.shots for c in cells)
    if total < min_shots:
        lines.append(
            f"the reliable area is NOT measured yet: {total} shot(s) on "
            f"record for this camera position, {min_shots} needed. Shoot a "
            f"few groups across the screen and it will draw itself.")
        return None, lines

    need = margin * bar
    blocked = set()
    weak = []
    pending = []        # weak on thin evidence: reported, never fenced
    for c in cells:
        if c.shots < min_per_bin:
            # NOT ENOUGH EVIDENCE IS NOT EVIDENCE OF WEAKNESS. Excluding a
            # region he simply has not shot at yet would shrink his play
            # area as a punishment for not shooting there, and the border
            # would then never grow back. Left in, and said.
            continue
        if c.median_peak < need:
            if c.shots < min_per_block:
                pending.append(c)
                continue
            blocked.add((c.col, c.row))
            weak.append(c)

    for c in sorted(pending, key=lambda c: (c.row, c.col)):
        lines.append(
            f"  {_name(c.col, c.row)}: weak so far, {c.shots} of "
            f"{min_per_block} shots needed, median peak {c.median_peak:.0f} "
            f"against a bar of {bar:.0f} = {c.margin(bar):.1f}x -- not "
            f"fenced yet.")

    if not blocked:
        lines.append(
            f"every region with evidence detects shots with margin to spare "
            f"(bar {bar:.0f}, weakest median peak "
            f"{min((c.median_peak for c in cells if c.shots >= min_per_bin),
                   default=0.0):.0f}): the whole screen is in play.")
        return None, lines

    box = _largest_clear_rect(blocked, bins)
    if box is None:
        lines.append(
            f"NO region of this screen detects shots reliably (bar "
            f"{bar:.0f}, every region under {need:.0f}). That is a camera "
            f"placement problem, not a reason to fence you in: the whole "
            f"screen stays in play. Move the camera more square to the "
            f"panel, or closer.")
        return None, lines

    i0, j0, i1, j1 = box
    area = PlayArea(x0=i0 / bins, y0=j0 / bins,
                    x1=(i1 + 1) / bins, y1=(j1 + 1) / bins)
    for c in sorted(weak, key=lambda c: (c.row, c.col)):
        lines.append(
            f"  {_name(c.col, c.row)}: {c.shots} shots, median peak "
            f"{c.median_peak:.0f} against a bar of {bar:.0f} = "
            f"{c.margin(bar):.1f}x, under the {margin:.0f}x a shot needs. "
            f"Marked off-limits for TARGETS; holes still appear there.")
    lines.append(
        f"play area: {area.fraction * 100:.0f}% of the screen, "
        f"{int(area.x0 * sw)},{int(area.y0 * sh)} to "
        f"{int(area.x1 * sw)},{int(area.y1 * sh)}. Concentrate your shots "
        f"inside the grey border; moving the camera more square to the "
        f"panel is what widens it.")
    return area, lines


def _name(col: int, row: int) -> str:
    cols = ("left", "centre", "right")
    rows = ("top", "middle", "bottom")
    c = cols[col] if 0 <= col < len(cols) else f"col {col}"
    r = rows[row] if 0 <= row < len(rows) else f"row {row}"
    return f"{r} {c}"


# ---------------------------------------------------------------------------
# WHICH CORNER OF THE SCREEN A SHOT REGISTERS BEST IN
#
# Asked for 2026-10-18: a MENU button in every drill, "dynamically placed
# into the best corner sensitivity-wise". So the question is which corner a
# deliberate trigger pull is most likely to be SEEN in, and on this bench the
# answer is not uniform: 36 real shots measured peak 76 on the left, 84 in the
# middle and 47 on the right, with blob area 2.7x smaller on the far side.
#
# IT CAN ONLY BE MEASURED FROM REAL SHOTS, and that is the whole point of
# putting it here rather than deriving it from something cheaper. This project
# has tried three cheaper proxies and all three were wrong (RULES C6):
#
#   white calibration markers   227-245 across the panel -- an 8% spread,
#                               because a white square saturates the sensor
#                               wherever it is drawn
#   the self test's own dot     70/77/76 left/middle/right -- flat, because a
#                               screen EMITS its dot where a laser REFLECTS
#   the mapping's Jacobian      predicted 1.65x on the left against a measured
#                               0.98x, and BACKWARDS: foreshortening raises a
#                               peak, while the laser's falls
#
# The honest source is the LEARNED SENSITIVITY MAP -- regions this gun has
# actually been fired into, each already backed by at least three shots that
# cleared the laser/noise boundary. It is a property of the camera POSE and
# dies with it, so a caller must clear its answer when the camera moves, the
# same as the sensitivity itself and the tuned dot radius.
#
# Pure, like the rest of this module: points in, a corner and a sentence out.

CORNERS = ("top left", "top right", "bottom left", "bottom right")


def best_corner(points, screen_size, fallback: str = "top left"):
    """Pick the corner a shot is most likely to be detected in.

    `points` are (screen_x, screen_y, peak) -- the learned sensitivity
    regions mapped into screen space, or raw shots. Returns
    `(corner, measured, why)`: the corner name, whether it was MEASURED or is
    merely the fallback, and one sentence saying which.

    A corner is only eligible if a point actually fell in its quadrant. With
    nothing eligible the fallback is returned and `measured` is False, which
    the caller must report rather than paper over (RULES C4): "I could not
    measure this" and "this corner is best" are different claims, and this
    project has already paid for reading one as the other.
    """
    sw, sh = screen_size
    if not sw or not sh:
        return fallback, False, "no screen size: using the default corner."
    buckets: dict[str, list[float]] = {}
    for sx, sy, peak in points or ():
        if peak is None or peak <= 0:
            continue
        name = ("top " if sy < sh / 2.0 else "bottom ") + \
               ("left" if sx < sw / 2.0 else "right")
        buckets.setdefault(name, []).append(float(peak))
    if not buckets:
        return (fallback, False,
                "no shots have been measured at this camera position, so the "
                "MENU button sits in the default corner.")
    scored = []
    for name, peaks in buckets.items():
        peaks.sort()
        scored.append((peaks[len(peaks) // 2], len(peaks), name))
    # Highest median first; more readings breaks a tie, then a stable name
    # order so the button cannot wander between launches on equal evidence.
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    best, n, name = scored[0]
    rest = ", ".join(f"{m:.0f} {nm}" for m, _, nm in scored[1:])
    why = (f"MENU button in the {name}: real shots there measured a median "
           f"peak of {best:.0f} over {n} region(s)"
           + (f", against {rest}." if rest else ", the only region measured."))
    return name, True, why
