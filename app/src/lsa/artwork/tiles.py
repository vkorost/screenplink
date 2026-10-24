"""The menu's tile pictures: one 512x512 grey per drill, loaded and corrected.

They live in `menu/` (NOT in `targets/`, whose one exact rule is "every image
in it is a scoreable figure", and NOT in `animations/`). A tile is named
`<key glyph>_<anything>.png`, so `7_popup.png` is the picture for the drill on
key 7 and `A_animated.png` the one for key A. The glyph is the join, not the
mode's name: `6_clear.png` belongs to `circles`, whose registered name is not
`clear`, and a new drill on key `B` picks up `B_whatever.png` with no code
change. `menu.png` (the word MENU in a box) has no glyph and is fetched by
stem with `named("menu")`.

THE BAKED BRIGHTNESS IS RE-CHECKED AT LOAD, NEVER TRUSTED. The files were baked
to the brightness ceiling (71 on 2026-10-18), and the ceiling is the smallest
p99 over every figure he has supplied: it moved 71 -> 52 -> 71 in a single
morning when new figures landed, and a baked value goes stale SILENTLY the
moment he adds a darker one. So each tile's own lit p99 is measured here and
the tile is brought DOWN to `ceiling.figure_level()` -- never up, a stroke
brighter than the darkest of his images hides the dot that lands on it
(RULES B10). This is `sequence.Sequence._check_ceiling` for a single picture.

PURE: cv2 and numpy only, no Qt. The seam to the screen is `modes/menu.py`.

THE CACHE IS A PLAIN DICT AND THAT IS SAFE HERE, where `modes.figure`'s cache is
a recorded trap (CLAUDE.md: 4.1 MiB per image, never evicted, 12 images and
growing with every figure he adds). A tile is 512x512 single-channel, 0.26 MiB;
the folder is a dozen files written by this project, not a directory he drops
artwork into, so twelve tiles are ~3 MiB for the life of the process and the
count is bounded by the number of drills, not by anything the shooter adds.
`MAX_TILES` is there anyway: a plain dict whose bound is a comment is the same
trap waiting for a larger folder, so past it the cache is dropped and refilled.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import assets, ceiling, load

MENU_DIR = load.PROJECT_ROOT / assets.ASSET_PARENT / "menu"      # unread; menu_dir() is the answer
MAX_TILES = 64            # see the module docstring: a bound, not a working size
TILE_SIDE = 512           # the pictures are baked square at this size


def menu_dir() -> Path:
    """`PROJECT_ROOT/menu`, overridable with LSA_MENU for tests (as LSA_TARGETS)."""
    return assets.asset_dir("menu")


@dataclass(frozen=True)
class Tile:
    """One tile picture, already brought down to the ceiling."""

    stem: str
    path: Path
    ink: np.ndarray       # uint8, 2-D, read-only; ALREADY corrected by `gain`
    baked_p99: float      # what the file measured at load
    ceiling: float        # `ceiling.figure_level()` at load
    gain: float           # <= 1.0; 1.0 when the file is already at or under it


_CACHE: "dict[str, tuple[tuple, Tile]]" = {}
_PROBLEMS: "dict[str, str]" = {}


def glyph_of(path: "str | Path") -> str:
    """The key glyph a tile file is for: the text before the first underscore.
    `menu.png` has no underscore and therefore no glyph ("")."""
    stem = Path(path).stem
    return stem.split("_", 1)[0] if "_" in stem else ""


def files(folder: "Path | None" = None) -> list[Path]:
    root = Path(folder) if folder is not None else menu_dir()
    return sorted(root.glob("*.png")) if root.is_dir() else []


def _stamp(path: Path, want: float) -> tuple:
    st = path.stat()
    return (st.st_size, st.st_mtime_ns, round(want, 3))


def load_tile(path: "str | Path") -> "Tile | None":
    """Load one tile, corrected against the CURRENT ceiling. None if the file is
    missing or unreadable; an unreadable one is recorded in `problems()` so a
    corrupt picture is not mistaken for a drill that simply has none."""
    path = Path(path)
    key = str(path)
    want = ceiling.figure_level()
    try:
        stamp = _stamp(path, want)
    except OSError:
        _CACHE.pop(key, None)
        return None
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        grey = load.load_grey(path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        _PROBLEMS[key] = str(exc)
        return None
    _PROBLEMS.pop(key, None)
    baked = ceiling.lit_p99(grey)
    gain = 1.0 if baked <= 0.0 else min(1.0, want / baked)
    if gain < 1.0:
        # FLOOR, not round: rounding up could land a pixel one level above the
        # ceiling, and the ceiling check is "no pixel above it".
        ink = np.floor(grey.astype(np.float32) * gain).astype(np.uint8)
    else:
        ink = grey.copy()
    ink.setflags(write=False)
    tile = Tile(path.stem, path, ink, float(baked), float(want), float(gain))
    if len(_CACHE) >= MAX_TILES:
        _CACHE.clear()
    _CACHE[key] = (stamp, tile)
    return tile


def for_glyph(glyph: str, folder: "Path | None" = None) -> "Tile | None":
    """The tile for the drill on this key, or None when it has no picture (the
    menu then draws its glyph and name alone)."""
    if not glyph:
        return None
    for f in files(folder):
        if glyph_of(f).upper() == glyph.upper():
            return load_tile(f)
    return None


def named(stem: str, folder: "Path | None" = None) -> "Tile | None":
    """A tile by file stem, e.g. `named("menu")` for the MENU corner picture."""
    root = Path(folder) if folder is not None else menu_dir()
    return load_tile(root / f"{stem}.png")


def problems() -> "dict[str, str]":
    """Files that exist and could not be read, path -> reason. A copy."""
    return dict(_PROBLEMS)


def forget() -> None:
    """Drop every cached tile and recorded problem (tests, and a new folder)."""
    _CACHE.clear()
    _PROBLEMS.clear()
