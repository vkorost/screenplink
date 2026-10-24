"""Turn the PyInstaller output into the folder a stranger downloads.

    (from app\\)  .venv\\Scripts\\python.exe packaging\\assemble.py

Run AFTER the spec has built `build/ScreenPlink`. This adds the things that
have to sit BESIDE the executable rather than inside `_internal/`, where
neither the user nor Claude Code would ever look:

    .mcp.json                 so Claude Code finds the tools by itself
    Play.cmd                  screenplink.exe --target
    Doctor.cmd                the camera doctor; output redirected to
                              doctor-report.txt, because the exe is windowed
    CLAUDE.md                 the first-run decision tree, written for Claude
    README.md                 the same thing for a person
    stop_screenplink.ps1      a graceful close, because force-killing this app
                              wedges the camera until it is physically replugged
    config/settings.default.json
                              GENERATED from the dataclass, never hand-written

WHY THE DEFAULTS ARE GENERATED AND NOT SHIPPED AS A COPY
--------------------------------------------------------
There is deliberately NO `settings.local.json` in the release. The app's own
defaults are already right for a machine nothing has been measured on -- no
device name, index -1, and `exposure` None, which is what forces a full sweep
instead of trusting a number measured in another room. Shipping a settings file
would be a saved setting silently beating a code default, which is a scar on
this project (`min_frames` was fixed to 2 in code and the app ran with 1 for
days because the saved file said so).

`settings.default.json` is therefore a REFERENCE the app does not load: it
shows what is assumed before anything is measured, and gives the user a thing
to look at when they want to reset. Generated from `AppSettings` so it cannot
drift from the code the way a hand-written copy would.

This script also REFUSES to produce a release that leaks anything, rather than
trusting the spec's exclude list. `scripts/gen_voice_lines.py` carries an
ElevenLabs API key in plaintext; it is gitignored, PyInstaller does not read
gitignore, and a check beats a comment.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "packaging"
OUT = ROOT / "build" / "ScreenPlink"

sys.path.insert(0, str(ROOT / "src"))

# The asset table, loaded by path (stdlib-only) so a build step needs no Qt/OpenCV.
import importlib.util as _ilu                                          # noqa: E402

_aspec = _ilu.spec_from_file_location("_lsa_assets", ROOT / "src" / "lsa" / "artwork" / "assets.py")
_assets = _ilu.module_from_spec(_aspec)
_aspec.loader.exec_module(_assets)
ASSET_DIRS, ASSET_PARENT = _assets.ASSET_DIRS, _assets.ASSET_PARENT


def refuse(why: str) -> None:
    print(f"\nREFUSING TO ASSEMBLE: {why}")
    raise SystemExit(2)


# --------------------------------------------------------------------------
# 1. Nothing secret, and nothing of his, may be in the release.
# --------------------------------------------------------------------------
def check_no_leaks() -> list[str]:
    checked = []

    bad = list(OUT.rglob("gen_voice_lines*"))
    if bad:
        refuse(f"the voice generator carries an API key in plaintext: {bad}")
    checked.append("no gen_voice_lines.py (it holds an ElevenLabs key)")

    # Any file that looks like a credential, wherever it came from.
    for pat in ("*.key", "*.pem", "*_secret*", "*.env", ".env"):
        hit = list(OUT.rglob(pat))
        if hit:
            refuse(f"something credential-shaped is in the release: {hit}")
    checked.append("no *.key / *.pem / *.env / *_secret*")

    # His measured state. A shipped calibration is wrong for every other
    # camera and would be silently believed.
    for name in ("settings.local.json", "calibration.json"):
        hit = list(OUT.rglob(name))
        if hit:
            refuse(f"{name} is HIS, measured on his camera in his room, and "
                   f"would be trusted on a machine it means nothing about: {hit}")
    checked.append("no settings.local.json or calibration.json")

    if list(OUT.rglob("test_*.py")) or (OUT / "_internal" / "tests").exists():
        refuse("the test suite is in the release")
    checked.append("no test suite")

    # Staging material: 19 MB referenced nowhere in src/.
    if list(OUT.rglob("fall-originals")):
        refuse("images/fall-originals is staging material, 19 MB, unused")
    checked.append("no images/fall-originals (19 MB of staging material)")

    # The asset folders must sit where asset_dir() looks for them frozen:
    # <bundle>/resources/<name>. A bundle with them at the bare name would start,
    # draw nothing and play nothing, which reads as a broken install.
    for name in ASSET_DIRS:
        if not (OUT / "_internal" / ASSET_PARENT / name).is_dir():
            refuse(f"_internal/{ASSET_PARENT}/{name} is missing: asset_dir('{name}') "
                   f"would find nothing in this bundle. Rebuild from the current spec.")
    checked.append(f"every asset folder is under _internal/{ASSET_PARENT}/")

    return checked


# --------------------------------------------------------------------------
# 2. The default settings, generated from the dataclass.
# --------------------------------------------------------------------------
def write_defaults() -> Path:
    from lsa.settings import AppSettings                       # noqa: E402

    fresh = AppSettings()
    data = {f.name: getattr(fresh, f.name) for f in dataclasses.fields(fresh)}

    # These must be absent-or-neutral or the app would trust a measurement it
    # never made. Assert rather than assume: this is the whole point of the file.
    # `bests` joins the three: the best rounds on THIS machine are a measurement
    # of this shooter, and a stranger's first round must not be compared with them.
    for name, want in (("device_name", ""), ("device_index", -1), ("exposure", None),
                       ("bests", {})):
        if name in data and data[name] != want:
            refuse(f"AppSettings.{name} defaults to {data[name]!r}, not {want!r}. "
                   f"A release must carry NOTHING measured on another machine: "
                   f"a remembered camera or exposure is never revisited, so a "
                   f"number from this room would silently govern someone else's.")

    cfg = OUT / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    path = cfg / "settings.default.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 3. The files that go beside the exe.
# --------------------------------------------------------------------------
BESIDE = (
    (HERE / "mcp.json", ".mcp.json"),
    # The SAME two files the dev tree uses (app/scripts): each runs screenplink.exe
    # when it sits beside it and the virtual environment otherwise.
    (ROOT / "scripts" / "Play.cmd", "Play.cmd"),
    (ROOT / "scripts" / "Doctor.cmd", "Doctor.cmd"),
    (HERE / "release-CLAUDE.md", "CLAUDE.md"),
    (HERE / "release-README.md", "README.md"),
    (ROOT / "scripts" / "stop_lab.ps1", "stop_screenplink.ps1"),
)


def main() -> int:
    if not OUT.is_dir():
        refuse(f"{OUT} does not exist -- build the spec first")
    # ONE executable; the modes are chosen by argument. Both release documents
    # tell a stranger to run Doctor.cmd first, and Doctor.cmd and Play.cmd are
    # the only things that start the exe in a mode, so a release without either
    # makes the first instruction fail. Absence is a refusal, not a warning.
    if not (OUT / "screenplink.exe").is_file():
        refuse(f"screenplink.exe is missing from {OUT}")
    for cmd in ("Play.cmd", "Doctor.cmd"):
        if not (ROOT / "scripts" / cmd).is_file():
            refuse(f"{cmd} is missing from {ROOT / 'scripts'}; the release cannot start "
                   f"the game or the doctor without it")
    # The old three-exe build left these behind in a reused build folder. They
    # would ship a stale second binary that nothing documents.
    for stale in ("screenplink-mcp.exe", "screenplink-doctor.exe"):
        if (OUT / stale).exists():
            refuse(f"{stale} is in {OUT}: the release is ONE executable now. "
                   f"Rebuild from a clean build folder.")

    print("=== gate: nothing secret, nothing of his ===")
    for line in check_no_leaks():
        print(f"  [ok] {line}")

    print("\n=== files beside the exe ===")
    for src, dest in BESIDE:
        if not src.is_file():
            refuse(f"missing {src}")
        shutil.copyfile(src, OUT / dest)
        print(f"  {dest:28} <- {src.relative_to(ROOT)}")

    path = write_defaults()
    print(f"  {path.relative_to(OUT).as_posix():28} <- generated from AppSettings")

    # The app creates these itself, but an empty folder tells the user where
    # things will land before anything has happened.
    for d in ("logs", "cache"):
        (OUT / d).mkdir(exist_ok=True)
        print(f"  {d + '/':28}   created empty")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    files = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"\n=== release: {files} files, {total / 1e6:.0f} MB unzipped ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
