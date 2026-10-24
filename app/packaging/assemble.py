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
    source/                   THE WHOLE REPOSITORY, tests included, taken from
                              `git archive HEAD` so it is exactly the tracked
                              tree and cannot carry anything gitignored

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

WHY THE SOURCE COMES FROM git archive AND NOT FROM A COPY OF THE FOLDER
-----------------------------------------------------------------------
He asked for the release to carry everything: the installer, the source and all
the tests. The honest way to do that is to ship the TRACKED tree, because the
set of things that must never ship is exactly the set git already ignores --
the ElevenLabs key, his measured `config/`, the working notes, the calibration
backups. A hand-rolled copy with an exclude list would be a second definition
of that set, free to drift from the first. `git archive HEAD` cannot.

The source is taken AS IT IS ON DISK, not from the commit, because that is what
the executable beside it was built from. `source/BUILT-FROM.txt` names HEAD and
lists anything that differs from it, so a reader can tell the two cases apart
without this script refusing to build until someone commits.

This script also REFUSES to produce a release that leaks anything, rather than
trusting the spec's exclude list. `scripts/gen_voice_lines.py` carries an
ElevenLabs API key in plaintext; it is gitignored, PyInstaller does not read
gitignore, and a check beats a comment. That one check is NOT relaxed by
shipping the source: the key file is untracked, so `git archive` never sees it.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
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

    # The tests SHIP (he asked for them), but they belong in source/, not
    # inside the frozen bundle: _internal is what the exe imports from, and a
    # suite in there would be dead weight the app could import by accident.
    if list((OUT / "_internal").rglob("test_*.py")) or (OUT / "_internal" / "tests").exists():
        refuse("the test suite is INSIDE the frozen bundle (_internal); it ships in source/")
    checked.append("no test suite inside _internal/")

    suite = OUT / "source" / "app" / "tests"
    found = len(list(suite.glob("test_*.py"))) if suite.is_dir() else 0
    if found < 20:
        refuse(f"source/app/tests holds {found} suites: the release is supposed to "
               f"carry all of them, and this tree has 26")
    checked.append(f"source/app/tests carries {found} suites")

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

    # ...AND EVERY FILE IN IT. A folder-level check cannot see a missing file,
    # which is how bullet.png came to be absent from a release while images/
    # was present and this gate said "every asset folder is under resources/".
    # The magazine draws that file on every repaint of every drill, so the
    # FileNotFoundError was raised inside paintEvent, `painted` never emitted,
    # every Paint() timed out, and the app concluded its camera was blind --
    # four seconds after measuring a 0.91mm mapping by photographing the same
    # screen. Read as an intermittent camera fault for days.
    #
    # The invariant is equality, not presence: a bundle that trims an asset
    # folder has to say so here, where it is a visible, reviewable act.
    short = []
    for name in ASSET_DIRS:
        src = REPO / ASSET_PARENT / name
        dst = OUT / "_internal" / ASSET_PARENT / name
        if not src.is_dir():
            continue
        have = {q.relative_to(dst).as_posix() for q in dst.rglob("*") if q.is_file()}
        want = {q.relative_to(src).as_posix() for q in src.rglob("*") if q.is_file()}
        for rel in sorted(want - have):
            short.append(f"{name}/{rel}")
    if short:
        refuse("the bundle is MISSING asset files the repository has. Anything "
               "the app draws from one of these would raise inside paintEvent, "
               "which stops the whole screen painting, not just that item:"
               + "".join(chr(10) + "    " + m for m in short[:20])
               + (chr(10) + f"    ... and {len(short) - 20} more" if len(short) > 20 else ""))
    checked.append(f"every asset FILE is in the bundle ("
                   + str(sum(1 for n in ASSET_DIRS
                             for _ in (REPO / ASSET_PARENT / n).rglob("*")
                             if _.is_file())) + " checked)")

    return checked


# --------------------------------------------------------------------------
# 1b. The source, straight from the tracked tree.
# --------------------------------------------------------------------------
REPO = ROOT.parent


def _git(*args: str) -> str:
    out = subprocess.run(("git", "-C", str(REPO)) + args,
                         capture_output=True, text=True)
    if out.returncode:
        refuse(f"git {' '.join(args)} failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def copy_source() -> tuple[str, int]:
    """Copy every TRACKED file into OUT/source, as it is on disk right now.

    Tracked, because the set of things that must never ship is exactly the set
    git already ignores: the ElevenLabs key, his measured config/, the working
    notes, the calibration backups. An exclude list written here would be a
    second definition of that set, free to drift from the first.

    As it is on disk, because that is what the executable beside it was built
    from. BUILT-FROM.txt names HEAD and lists anything that differs from it, so
    the reader can tell a release built from a commit from one built from a
    working tree, without this script having to forbid the second.
    """
    tracked = [f for f in _git("ls-files", "-z").split(chr(0)) if f]
    if len(tracked) < 400:
        refuse("git ls-files reports only " + str(len(tracked)) + " tracked files")

    dest = OUT / "source"
    if dest.exists():
        shutil.rmtree(dest)

    missing, files = [], 0
    for rel in tracked:
        src = REPO / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files += 1
    if missing:
        refuse("tracked but not on disk, so the source would be incomplete:"
               + "".join(chr(10) + "    " + m for m in missing))

    suite = dest / "app" / "tests"
    if not suite.is_dir():
        refuse("app/tests did not reach source/: the release is supposed to "
               "carry the tests")

    commit = _git("rev-parse", "HEAD").strip()
    when = _git("log", "-1", "--format=%cI").strip()
    changed = sorted(ln[3:].strip().strip(chr(34))
                     for ln in _git("status", "--porcelain").splitlines()
                     if ln and not ln.startswith("??"))
    changed = [c for c in changed if c in set(tracked)]

    venv = chr(92).join(("app", ".venv", "Scripts", "python.exe"))
    lines = [
        "ScreenPlink source: every tracked file, exactly as it was on disk when",
        "the executable beside it was built. Nothing is generated or trimmed.",
        "",
        "commit  " + commit,
        "date    " + when,
    ]
    if changed:
        lines += ["",
                  "These files differ from that commit: the release was built from a",
                  "working tree, and what is here is what it was built from.", ""]
        lines += ["    " + c for c in changed]
    lines += [
        "",
        "To run it from source you need Python and the dependencies:",
        "",
        "    python -m venv " + chr(92).join(("app", ".venv")),
        "    " + venv + " -m pip install -r "
        + chr(92).join(("app", "requirements.txt")),
        "    " + chr(92).join(("app", "scripts", "Play.cmd")),
        "",
        "The tests need nothing further, and no camera:",
        "",
        "    " + chr(92).join(("app", "scripts", "run_tests.bat")),
        "",
    ]
    (dest / "BUILT-FROM.txt").write_text(chr(10).join(lines), encoding="utf-8")
    files += 1
    return commit, files


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

    print("=== source: every tracked file, as it is on disk ===")
    commit, src_files = copy_source()
    print(f"  source/{'':21} <- {src_files} files, at or past commit {commit[:8]}")

    print(chr(10) + "=== gate: nothing secret, nothing of his ===")
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
