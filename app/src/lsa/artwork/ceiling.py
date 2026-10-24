"""One brightness ceiling for everything the app draws, taken from the darkest
artwork he supplied.

Why it exists (his finding, 2026-10-17, after shooting the app's own targets):
the detector scores `red - max(green, blue)`. On near-black his laser makes about
(255, 10, 10), a score near 245. Landing on a grey stroke at luminance 200 it
makes (255, 200, 200): a score of 55, and with the floor at 12 and the bar at 18
the dot is nearly erased. "The shots do not penetrate that brightness." His own
figures register because their strokes are dark (p99 of the lit pixels 77-218);
the app's circles, rings and balls were drawn at 190-240 and as filled discs
(3.3% of the screen above luminance 60, against a 2% budget).

His instruction: "lets take the lowest light level among all the images I gave
you and then align everything else to that level" -- and scale his brighter
images down to meet it too, so that no target is harder to register on than
another for a reason unrelated to his shooting.

THE NUMBER IS MEASURED, NEVER HARDCODED. `figure_level()` is the smallest, over
every image in `targets/` and `targets/fall/`, of the 99th percentile of that
image's LIT pixels (above `LIT_FLOOR`). The p99 and not the median, because what
competes with the dot is the brightest stroke it can land on, not the average
one. Adding a darker image moves the ceiling down to it, honestly, and every
other target follows; the one guard is `MIN_LEVEL`, below which a target would
no longer be legible from the firing line, so the ceiling stops there and the
darker image is drawn as it is (never raised).

Levels here are FILE luminance (what is stored). `screen_level()` is what the
same pixel shows on the target background: ink is drawn as white at coverage
v/255 over the background, so it lands at v + BACKGROUND * (1 - v/255). The app's
own drawing colours are opaque, so they are set to the on-screen figure.

PURE: cv2 and numpy only, no Qt. Cached in-process and on disk (decoding two
dozen 1792x2400 JPEGs costs about a second), keyed on every file's size and
mtime, so editing or adding an image invalidates it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import assets, load

LIT_FLOOR = 30            # a pixel above this is ink; the black field and JPEG fuzz sit below
PERCENTILE = 99.0
MIN_LEVEL = 40            # a ceiling below this would not be legible from the firing line
FALLBACK_LEVEL = 78       # only if there is no artwork at all to measure (never in practice)

# DERIVED, so it lives in cache/ rather than with the files he owns: deleting
# it costs a few seconds of re-measuring the artwork and nothing else.
# DERIVED and WRITABLE, so it lives with the user's state (beside the .exe when
# frozen), never in the read-only bundle.
_CACHE_FILE = assets.user_dir("cache") / "target_ceiling.cache.json"
_MEM: dict[str, dict] = {}


def _images(folder: Path) -> list[Path]:
    out = sorted(folder.glob("*.jpg"))
    fall = folder / "fall"
    if fall.is_dir():
        out += sorted(fall.glob("*.jpg"))
    return out


def lit_p99(grey: np.ndarray) -> float:
    """The 99th percentile of the lit pixels (above LIT_FLOOR); 0 if none."""
    lit = grey[grey > LIT_FLOOR]
    return float(np.percentile(lit, PERCENTILE)) if lit.size else 0.0


def _stamp(files: list[Path], folder: Path) -> dict[str, list]:
    out = {}
    for f in files:
        st = f.stat()
        out[str(f.relative_to(folder)).replace("\\", "/")] = [st.st_size, st.st_mtime_ns]
    return out


def measure(folder: "Path | None" = None) -> dict:
    """{'p99': {relative path: p99 of lit pixels}, 'level': the ceiling}."""
    root = Path(folder) if folder is not None else load.targets_dir()
    files = _images(root)
    stamp = _stamp(files, root)
    key = str(root)
    got = _MEM.get(key)
    if got is not None and got["stamp"] == stamp:
        return got
    real = root == assets.bundle_root() / assets.ASSET_PARENT / "targets"      # only the real folder is cached on disk
    try:
        disk = json.loads(_CACHE_FILE.read_text(encoding="utf-8")) if real else {}
        if disk.get("folder") == key and disk.get("stamp") == stamp:
            _MEM[key] = disk
            return disk
    except (OSError, ValueError):
        pass
    p99 = {}
    for f in files:
        v = lit_p99(load.load_grey(f))
        if v > 0.0:
            p99[str(f.relative_to(root)).replace("\\", "/")] = round(v, 2)
    darkest = min(p99.values()) if p99 else float(FALLBACK_LEVEL)
    got = {"folder": key, "stamp": stamp, "p99": p99,
           "level": round(max(float(MIN_LEVEL), darkest), 2),
           "darkest": round(darkest, 2)}
    _MEM[key] = got
    if real:
        try:                                 # best effort; never fatal
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps(got), encoding="utf-8")
        except OSError:
            pass
    return got


def figure_level(folder: "Path | None" = None) -> float:
    """The ceiling in FILE luminance: the p99 of the darkest artwork, floored."""
    return float(measure(folder)["level"])


def screen_level(folder: "Path | None" = None) -> int:
    """The same ceiling as it appears on the target background."""
    v = figure_level(folder)
    return int(round(v + load.BACKGROUND * (1.0 - v / 255.0)))


def gain_for_path(path: "str | Path", folder: "Path | None" = None) -> float:
    """`gain_for` for an image on disk, using the p99 already measured (so
    building a figure does not re-scan 4 million pixels on the UI thread)."""
    root = Path(folder) if folder is not None else load.targets_dir()
    m = measure(root)
    try:
        rel = str(Path(path).relative_to(root)).replace("\\", "/")
    except ValueError:
        return gain_for(load.load_grey(path), folder)
    p = m["p99"].get(rel)
    if p is None:
        return gain_for(load.load_grey(path), folder)
    return 1.0 if p <= m["level"] else float(m["level"]) / p


def gain_for(grey: np.ndarray, folder: "Path | None" = None) -> float:
    """The factor (<= 1) that brings this image's p99 down to the ceiling. 1.0
    for an image already at or under it: images are scaled DOWN, never up."""
    p = lit_p99(grey)
    level = figure_level(folder)
    return 1.0 if p <= level or p <= 0.0 else level / p
