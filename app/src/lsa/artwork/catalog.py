"""Which figure, and where: the randomness rules for the figure drills, as pure
functions of a caller-supplied random generator.

His words (2026-10-17): "they should appear at random. there should be nothing
predictable about the order these targets show up", and "no need to tell them
apart - I will shoot all of them exactly the same way". So the cast is variety,
never information: nothing is scored on which figure it is.

Rules, and why:

  * `pick_figure` is UNIFORM over every figure except the one just shown. Never
    the same figure twice in a row, because an immediate repeat reads from the
    firing line as the app being stuck, and this project has lost days to
    faults that were really the app looking broken. NOT a shuffled bag: a bag
    makes the last draw of a cycle certain, which is predictability in the
    dimension he named. Uniform over eleven gives the shooter nothing to
    pre-aim on.
  * `pick_position` is `ReactionDrill._place` carried across: a centre drawn
    uniformly inside the playfield inset by the figure's half-size, at least
    35% of the playfield diagonal from the previous one, so it cannot be
    pre-aimed from where the last one stood.
  * EVERY draw goes through the generator passed in (a mode's own seeded
    `random.Random`). Nothing here touches the module-level `random`, so a
    test can seed a mode and name the sequence it expects. A drill that can
    only be checked by luck cannot be checked.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .load import targets_dir
from .zones import zones_path

MIN_MOVE_FRAC = 0.35


def all_names(folder: "Path | None" = None) -> list[str]:
    """Every image in the targets folder, SORTED, whether or not it can be
    scored. A deterministic order is the precondition for a seeded draw being
    reproducible."""
    root = folder if folder is not None else targets_dir()
    return sorted(p.stem for p in root.glob("*.jpg"))


def excluded(folder: "Path | None" = None) -> dict[str, str]:
    """Figures that may NEVER be drawn, with the reason: no sidecar, an
    unreadable one, or one whose extracted rings did not verify.

    A figure that cannot be judged must not be drawn at all. His complaint
    from the firing line was "sometimes the shots land inside the targets and
    nothing happens", and a figure whose shots cannot be scored reproduces it
    exactly. A new image dropped into the folder without a verified sidecar is
    therefore excluded, by name, rather than drawn and reported unscored."""
    out: dict[str, str] = {}
    for n in all_names(folder):
        why = reason(n, folder)
        if why:
            out[n] = why
    return out


def reason(name: str, folder: "Path | None" = None) -> "str | None":
    """Why this one figure may not be drawn, or None if it may."""
    zp = zones_path(name, folder)
    if not zp.exists():
        return "no zones file (run tools/zones_author.py --author)"
    try:
        info = json.loads(zp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"zones file unreadable: {exc}"
    if not info.get("scored") or not info.get("zones"):
        return "its printed rings did not verify: " + "; ".join(
            info.get("why_not") or ["no zones"])
    return None


def names(folder: "Path | None" = None) -> list[str]:
    """The figures that may be DRAWN: sorted, and only those with verified
    scoring zones. Everything the random draw uses comes from here."""
    bad = excluded(folder)
    return [n for n in all_names(folder) if n not in bad]


def path_for(name: str, folder: "Path | None" = None) -> Path:
    return (folder if folder is not None else targets_dir()) / f"{name}.jpg"


# FALL POSES. A figure that is shot down shows a second picture for a second
# (his words, 2026-10-17: "for each of the target files I've added another one
# when the targets are falling"). They live in their OWN folder, `targets/fall/`,
# matched to their figure by basename: `targets/fall/alien01.jpg` is the fall of
# `targets/alien01.jpg`. A folder rather than a filename suffix because the
# suffix made the rule "everything in targets/ is a figure EXCEPT names matching
# a pattern", and a rule with an exception is what failed the night they
# arrived: eleven fall poses were audited as unscoreable figures. Now everything
# directly in targets/ is a scoreable figure, with no exception to parse.
#
# `<name>-fall.jpg` inside the folder is tolerated and the suffix stripped, since
# that is how he names them as he generates them.
FALL_DIR = "fall"
FALL_SUFFIX = "-fall"


def fall_dir(folder: "Path | None" = None) -> Path:
    return (folder if folder is not None else targets_dir()) / FALL_DIR


def _fall_base(stem: str) -> str:
    return stem[:-len(FALL_SUFFIX)] if stem.endswith(FALL_SUFFIX) else stem


def _fall_files(folder: "Path | None" = None) -> list[Path]:
    d = fall_dir(folder)
    return sorted(d.glob("*.jpg")) if d.is_dir() else []


def fall_path(name: str, folder: "Path | None" = None) -> "Path | None":
    """The fall pose of figure `name`, or None if it has none (it then simply
    disappears when it drops). `<name>.jpg` wins over `<name>-fall.jpg` if both
    exist; the loser is reported by `unpaired_falls`."""
    d = fall_dir(folder)
    for stem in (name, name + FALL_SUFFIX):
        p = d / f"{stem}.jpg"
        if p.is_file():
            return p
    return None


def unpaired_falls(folder: "Path | None" = None) -> dict[str, str]:
    """Files in the fall folder that no figure will ever show, with why: no
    figure of that name, or a second file for a figure that already has one. A
    fall pose nobody can reach is a file he thinks is working."""
    figures = set(all_names(folder))
    out: dict[str, str] = {}
    for f in _fall_files(folder):
        base = _fall_base(f.stem)
        if base not in figures:
            out[f.name] = f"no figure called {base!r} in targets/"
        elif fall_path(base, folder) != f:
            out[f.name] = f"{base!r} already has {fall_path(base, folder).name}"
    return out


def pick_figure(rng: random.Random, previous: "str | None",
                all_names: "list[str]") -> str:
    """Uniform over `all_names` excluding `previous`."""
    pool = [n for n in all_names if n != previous]
    if not pool:                       # a one-figure cast: nothing to exclude
        pool = list(all_names)
    return pool[rng.randrange(len(pool))]


def pick_pair(rng: random.Random, previous: "tuple[str, str] | None",
              all_names: "list[str]", forced: "str | None" = None
              ) -> "tuple[str, str]":
    """One figure per SEAT for a two-player cycle: each seat is uniform over the
    cast except the figure IT showed last cycle (so a seat never sees the same
    figure twice running), and the two seats never get the same figure in the
    same cycle. Seat 1 is drawn after seat 0 and excludes it as well, which
    keeps both marginals uniform by symmetry (measured, tests/test_modes.py).
    A cast too small to keep the two apart (one figure) falls back to sharing,
    rather than raising mid-round. `forced` (a test or debug choice) is both."""
    if forced is not None:
        return (forced, forced)
    prev0, prev1 = previous if previous else (None, None)
    first = pick_figure(rng, prev0, all_names)
    pool = [n for n in all_names if n not in (first, prev1)]
    if not pool:
        pool = [n for n in all_names if n != first] or list(all_names)
    return (first, pool[rng.randrange(len(pool))])


def pick_position(rng: random.Random, field, half_w: float, half_h: float,
                  previous: "tuple[float, float] | None",
                  min_move_frac: float = MIN_MOVE_FRAC,
                  avoid=None) -> tuple[float, float]:
    """A centre inside `field` (a QRectF-like: left/right/top/bottom/center),
    inset by the figure's half-size, at least `min_move_frac` of the
    playfield diagonal from `previous` (40 tries, then the farthest corner,
    as `ReactionDrill._place`).

    `avoid` is ground the figure should not stand on -- today the MENU
    button's rectangle, after he photographed a target sitting on it on
    2026-10-18. It is a PREFERENCE here, not a rule, and that is deliberate:
    a figure is more than half the height of the screen, so forbidding every
    position whose box touches a corner strip would push every figure to one
    side and quietly turn a random drill into a predictable one. A candidate
    that clears it wins; if forty tries never find one, the figure stands
    where the drill wants it. Nothing is lost when that happens -- the drill
    is offered each shot first, so a hit on the figure is still a hit.
    """
    left, right = field.left() + half_w, field.right() - half_w
    top, bottom = field.top() + half_h, field.bottom() - half_h
    if right <= left or bottom <= top:
        return (field.center().x(), field.center().y())

    def clears(x: float, y: float) -> bool:
        if avoid is None:
            return True
        return not (x - half_w < avoid.right() and x + half_w > avoid.left()
                    and y - half_h < avoid.bottom() and y + half_h > avoid.top())

    if previous is None:
        first = None
        for _ in range(40):
            x, y = rng.uniform(left, right), rng.uniform(top, bottom)
            if first is None:
                first = (x, y)
            if clears(x, y):
                return (x, y)
        return first
    far = min_move_frac * math.hypot(right - left, bottom - top)
    fallback = None
    for _ in range(40):
        x, y = rng.uniform(left, right), rng.uniform(top, bottom)
        if math.hypot(x - previous[0], y - previous[1]) >= far:
            if clears(x, y):
                return (x, y)
            if fallback is None:
                fallback = (x, y)
    if fallback is not None:
        return fallback
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    return max(corners, key=lambda c: math.hypot(c[0] - previous[0],
                                                 c[1] - previous[1]))
