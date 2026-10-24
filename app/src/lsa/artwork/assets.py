"""The asset folders, and the one place that says how a test finds them.

THIS HAS GONE WRONG THREE TIMES, and every time the shape was the same: a new
folder of the app's own pictures arrived, its loader resolved the folder from the
source file's location, and somebody forgot that the mutation runners run each
suite in a COPIED tree holding only `src`, `tests` and `tools`.

  * 2026-10-18  `animations/` -- five mutation anchors came back "not caught"
                because the suite crashed before it could report.
  * 2026-10-20  `ui/` (the magazine bullet) -- the figure and introspection
                runners reported 12 faults BROKEN, because neither passed its
                override. (The folder was folded into images/ on 2026-10-23.)
  * `mutate_detector.py` had the same hole in the same place and nothing said so.

Each time the instance was fixed (one more env var in one more runner) and the
next folder was left to be found the same way. So the rule is now one table:

  * `ASSET_DIRS` maps every asset folder to the environment variable that
    overrides it. Add a folder HERE and its loader reads `asset_dir(name)`.
  * `asset_env(root)` is that table turned into an environment for a child
    process; `scratch_env(root)` adds the offscreen platform. Every runner that
    spawns a suite in a scratch tree builds its environment from these, so a new
    folder reaches every runner by construction.
  * `tests/test_rules.py` fails when a loader reads an override that is not in the
    table, names a project folder that is not in it, or a runner stops using
    `scratch_env`.

No imports beyond the standard library, so a runner can load this file by path
without pulling in OpenCV or Qt.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# TWO QUESTIONS, TWO ANSWERS (2026-10-20, packaging stage 1).
#
# Where is the app's READ-ONLY data (pictures, sounds, voice lines)?  `bundle_root()`
# Where is the USER'S WRITABLE state (config, logs, cache, recordings)?  `user_root()`
#
# THE REPOSITORY LAYOUT (2026-10-23, the clean repository). From source:
#
#     <project>/                 the repository root: README.md, .gitignore, nothing else
#         app/                   everything the app IS and everything it WRITES
#             src/lsa/artwork/assets.py     <- this file
#             config/ logs/ cache/ recordings/   the user's writable state
#         resources/             the app's own read-only artwork, sound and voice
#             targets/ animations/ menu/ sounds/ voice/ images/
#             (images/ also holds the magazine bullet; the old ui/ folder is gone)
#         docs/  manual/
#
#   * bundle_root() is the PROJECT root, and `asset_dir(name)` adds `ASSET_PARENT`
#     ("resources") itself: <project>/resources/<name>.
#   * user_root() is the `app/` folder.
#
# From source those are now DIFFERENT folders (they used to be the same one).
# Frozen (PyInstaller) they differ as well, and they must:
#
#   * bundle_root() = sys._MEIPASS. In a one-folder build that is the folder the
#     data was unpacked to; in one-file mode it is a TEMP directory deleted on exit.
#     Read-only either way. The packaging step puts the asset folders under
#     `resources/` inside it, so the SAME `asset_dir` rule serves both layouts.
#   * user_root()   = the folder holding the .exe (PORTABLE: his chosen layout).
#     NOT _MEIPASS: writing a calibration into a temp directory loses it at exit,
#     and writing it into the bundle mixes his measured state with the app's files.
#
# Both are functions that read `sys.frozen` at CALL time, for the reason `asset_dir`
# is: a test (or a launcher) that changes the answer must not be defeated by the
# order modules happen to import in.
#
# THE OVERSHOOT THIS REPLACES, and the arithmetic that must not be repeated on the
# frozen path: a parent count from `__file__` is right for exactly one layout. This
# file sits at app/src/lsa/artwork/assets.py, so counting up from it:
#
#     parents[0] artwork   parents[1] lsa   parents[2] src
#     parents[3] app       <- user_root() from source
#     parents[4] <project> <- bundle_root() from source
#
# A frozen bundle has no `src` and no `app` level, so the same count lands one folder
# ABOVE the app and every picture, sound and voice line fails to load. The frozen
# branches therefore NEVER count parents: they read sys._MEIPASS and sys.executable.
# Moving this file again means recounting the two source lines, and nothing else.


# The folder, under bundle_root(), that holds every asset folder. The ONE place that
# says so: `asset_dir`, `asset_env` and the packaging spec all read it.
ASSET_PARENT = "resources"


def frozen() -> bool:
    """True inside a PyInstaller bundle. Read at call time."""
    return bool(getattr(sys, "frozen", False))


def _app_root() -> Path:
    """<project>/app -- the folder holding src/, config/, logs/ ... (source mode only)."""
    return Path(__file__).resolve().parents[3]


def _source_root() -> Path:
    """<project> -- the repository root, which holds resources/ (source mode only)."""
    return Path(__file__).resolve().parents[4]


def bundle_root() -> Path:
    """The folder under which the app's own read-only data lives (`resources/<name>`):
    the project root from source, sys._MEIPASS when frozen."""
    if frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        return Path(meipass) if meipass else Path(sys.executable).resolve().parent
    return _source_root()


def user_root() -> Path:
    """The folder holding the user's writable state: beside the .exe when frozen,
    the `app/` folder from source (so config, logs, cache and recordings sit together
    as app/config, app/logs, app/cache, app/recordings)."""
    if frozen():
        return Path(sys.executable).resolve().parent
    return _app_root()


# Kept under its old name: `load.PROJECT_ROOT`, `camera_lab.PROJECT_ROOT` and the
# tests all read it. It is the BUNDLE root, fixed at import (equal to the project
# root when not frozen); code that needs the answer at call time uses the functions.
PROJECT_ROOT = bundle_root()

# folder name under the project root -> the variable that points a test at another one
ASSET_DIRS: "dict[str, str]" = {
    "targets": "LSA_TARGETS",         # the figure artwork and its scoring sidecars
    "animations": "LSA_ANIMATIONS",   # the filmed clips
    "menu": "LSA_MENU",               # the menu tile pictures
    "sounds": "LSA_SOUNDS",           # gunshots and effects
    "voice": "LSA_VOICE",             # the spoken announcements
    "images": "LSA_IMAGES",           # the startup splash and the magazine bullet
}

# The user's writable folders. NOT assets: they have their own overrides, which
# live with the files they redirect (LSA_SETTINGS, LSA_CALIBRATION, LSA_LOG,
# LSA_RECORDINGS) because a test redirects a FILE, not a folder.
USER_DIRS = ("config", "logs", "cache", "recordings")


def asset_dir(name: str) -> Path:
    """The folder for asset `name`: its override if set, else
    `bundle_root()/resources/<name>`. Read at CALL time, never captured at import,
    so setting the variable cannot be defeated by the order modules happen to load in."""
    env = os.environ.get(ASSET_DIRS[name])
    return Path(env) if env else bundle_root() / ASSET_PARENT / name


def user_dir(name: str) -> Path:
    """A writable folder of the user's: `user_root()/name`. Call time, like asset_dir."""
    if name not in USER_DIRS:
        raise KeyError(f"not a user folder: {name!r} (known: {USER_DIRS})")
    return user_root() / name


def asset_env(root: "Path | str" = PROJECT_ROOT, **folders: "Path | str") -> "dict[str, str]":
    """Every override variable, pointing at `root/resources/<folder>` unless `folders`
    gives that folder a different path (keyword = folder name, e.g. `targets=...`).
    `root` is a PROJECT root, or a scratch tree laid out like one."""
    unknown = set(folders) - set(ASSET_DIRS)
    if unknown:
        raise KeyError(f"not asset folders: {sorted(unknown)}")
    return {var: str(Path(folders.get(name, Path(root) / ASSET_PARENT / name))) for name, var in ASSET_DIRS.items()}


def scratch_env(root: "Path | str" = PROJECT_ROOT, **folders: "Path | str") -> "dict[str, str]":
    """The environment for a suite run in a scratch tree: this process's, the
    offscreen platform, and every asset override."""
    return dict(os.environ, QT_QPA_PLATFORM="offscreen", **asset_env(root, **folders))
