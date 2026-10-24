"""Load `lsa/artwork/assets.py` by path, for the runners that spawn suites -- and
build the scratch tree they run those suites in.

A mutation runner must not import the `lsa` package (that pulls in OpenCV and the
app's environment hooks), but it must build its child's environment from the same
table the app reads, or a new asset folder reaches the app and not the runner.

THE SCRATCH TREE (2026-10-23). `ASSET_DIRS` answers "which folders does the APP
read through an override". It never answered "which paths do the SUITES read
relative to their own location", and a suite run in a copied tree resolves
`ROOT / "sounds"`, `ROOT / "config" / "calibration.json"` and `ROOT / ".venv"`
against the COPY. Three runners each copied `src tests tools targets` by hand and
three suites went red for a reason that had nothing to do with any fault under
test. `copy_scratch_tree()` is now the one copy, built from the same tables:

    src, tests, tools      the code
    every ASSET_DIRS key   the app's own pictures, sounds and voice
    owned_files.OWNED      the files he owns (config/*.json, ui, targets) -- as
                           COPIES, so a rogue write lands on the copy (RULES C11)
    .venv/pyvenv.cfg       one file, not the environment: a suite reads it

NEVER copied: recordings/, logs/, cache/ (13 GB, or evidence, or disposable).
`scratch_env(d)` points every override at the COPY it was just given.

THE CLEAN REPOSITORY (2026-10-23). The code is `<project>/app/` and the artwork is
`<project>/resources/<name>`. `ROOT` here is `app/` (what every suite calls ROOT) and
`PROJECT` is the folder above it. The scratch tree `d` is laid out like `app/` PLUS a
`resources/` folder inside it:

    d/src  d/tests  d/tools  d/config  d/.venv/pyvenv.cfg      from ROOT
    d/resources/<each ASSET_DIRS name>                          from PROJECT/resources

so `scratch_env(d)` (which points at `d/resources/<name>`) and `copy_scratch_tree`
agree, and a suite finds artwork through `asset_dir()` / `res()` -- never by joining a
folder name onto its own root, which is wrong in the real tree and in the scratch one.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from owned_files import OWNED

_PATH = Path(__file__).resolve().parents[1] / "src" / "lsa" / "artwork" / "assets.py"
_spec = importlib.util.spec_from_file_location("_lsa_assets", _PATH)
assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assets)

ASSET_DIRS = assets.ASSET_DIRS
ASSET_PARENT = assets.ASSET_PARENT
asset_env = assets.asset_env
scratch_env = assets.scratch_env

ROOT = Path(__file__).resolve().parents[1]          # app/
PROJECT = ROOT.parent                               # the repository root


def res(name: str) -> Path:
    """Where a SUITE finds an asset folder: the override if a runner set one (a scratch
    tree), else the project's own `resources/<name>`. The one answer for tests; it is
    `asset_dir` itself, so a suite and the app can never disagree about it."""
    return assets.asset_dir(name)


def member_source(root: "Path | str", rel: str) -> Path:
    """Where a scratch-tree member (`scratch_members()` entry) lives under `root`
    (the app folder): `resources/...` is beside it, everything else inside it."""
    root = Path(root)
    return root.parent / rel if rel.replace("\\", "/").split("/")[0] == ASSET_PARENT else root / rel

CODE_DIRS = ("src", "tests", "tools")
# Single files a suite reads beside the tree, outside every folder above.
EXTRA_FILES = (".venv/pyvenv.cfg",)
_IGNORE = shutil.ignore_patterns("__pycache__", "_out")

# Top-level folders that exist on disk and are DELIBERATELY not in the scratch tree.
# test_rules Q fails when a suite a runner executes reads one of these (or any folder
# that is in neither list), so the decision is made by a person, not by a crash.
NEVER_COPIED = {
    "recordings": "gigabytes of his evidence; a suite writes to tests/_out, never here",
    "logs": "his run logs, the evidence tools/status.py reads",
    "cache": "derived and disposable; the app rebuilds it",
    "calib_backups": "his backups of the files he owns: the one copy that must stay untouched",
    ".git": "history",
}


def scratch_members() -> tuple[str, ...]:
    """Every path (relative to the root) the scratch tree holds, in copy order.
    Derived from the tables, so a folder added to ASSET_DIRS or a file added to
    OWNED reaches every runner that uses `copy_scratch_tree`."""
    out: list[str] = []
    for n in (*CODE_DIRS, *(f"{ASSET_PARENT}/{a}" for a in ASSET_DIRS), *OWNED, *EXTRA_FILES):
        if n not in out:
            out.append(n)
    return tuple(out)


def copy_scratch_tree(root: "Path | str", dest: "Path | str") -> None:
    """Copy `scratch_members()` from `root` into the new folder `dest`.
    A member missing from `root` is skipped (a fresh clone has no config/); the
    suites that need it say so by name (owned_files reports NOT FOUND)."""
    root, dest = Path(root), Path(dest)
    dest.mkdir(parents=True)
    for rel in scratch_members():
        src = member_source(root, rel)
        if src.is_dir():
            shutil.copytree(src, dest / rel, ignore=_IGNORE)
        elif src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
