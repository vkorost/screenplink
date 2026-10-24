"""Where a launch spends its seconds, from the log it already writes.

RULES D9 is his constraint, in his words:

    "Even a 25-second launch is too long... Launch should be as fast as
     possible, especially if the camera did not move from the previous
     launch."

    "No one will have any patience for the app that takes that long to
     start and initialize."

A launch had crept to 82 seconds BECAUSE NOTHING MEASURED IT. That is the
whole reason this file exists: the log has carried the timestamps all along,
and reconstructing the phases from them by hand -- which is what I did on
2026-10-15 -- is both slow and easy to get wrong.

    .venv\\Scripts\\python.exe tools\\launch_timing.py                 # newest log
    .venv\\Scripts\\python.exe tools\\launch_timing.py coldstart.log   # a specific one
    .venv\\Scripts\\python.exe tools\\launch_timing.py camera_lab.log --all

Prints every phase with its duration, biggest first, and names the process
the log came from -- because reading a harness measurement as his hardware
is a mistake this project has already made and reported to him (RULES C2).
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (first line that marks the phase STARTING, label). Order matters only for
# presentation; each phase is timed from its own marker to the next one that
# occurs after it, so a phase that does not happen costs nothing and is
# reported as absent rather than silently folded into its neighbour.
PHASES: list[tuple[str, str]] = [
    ("opening camera", "open the camera"),
    ("locating the screen", "find the screen (flash black/white)"),
    ("Checking it against the room", "check the room at the saved exposure"),
    ("measuring noise floor", "measure the noise floor"),
    ("checking the saved calibration", "two-marker check (his idea)"),
    ("pattern calibration:", "pattern calibration (18 markers)"),
    ("measuring the exposure by photographing", "exposure probe"),
    ("SELF TEST: flashing", "self test (12 dots)"),
    ("=== ARCADE MODE SELF TEST", "arcade walk-through (--modetest)"),
    ("=== VERDICT ===", "VERDICT ON SCREEN"),
]

TS = re.compile(r"^(\d\d):(\d\d):(\d\d)\s")


def secs(line: str) -> float | None:
    m = TS.match(line)
    if not m:
        return None
    h, mi, s = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s


def newest_log() -> Path | None:
    best, best_at = None, -1.0
    for p in (ROOT / "logs").glob("*.log"):
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt > best_at:
            best, best_at = p, mt
    return best


def role_of(block: str) -> str:
    m = re.search(r"GATE role=(\S+)", block)
    if m:
        screen = re.search(r"screen=(\S+)", block)
        return f"{m.group(1)}  screen {screen.group(1) if screen else '?'}"
    # Pre-dates the role tag. Guess from the one giveaway that was in the
    # line I misread: a simulated screen is 800x800.
    if "800x800" in block:
        return "harness (guessed from an 800x800 screen; no role= tag)"
    return "unknown -- this log predates role=, do not quote it as his bench"


def report(block: str, title: str) -> None:
    lines = [ln for ln in block.splitlines() if secs(ln) is not None]
    if len(lines) < 3:
        print(f"  {title}: too few timestamped lines")
        return
    t0 = secs(lines[0])
    hits: list[tuple[float, str]] = []
    searched_from = 0
    for needle, label in PHASES:
        found = None
        for i in range(searched_from, len(lines)):
            if needle in lines[i]:
                found = i
                break
        if found is None:
            continue
        hits.append((secs(lines[found]) - t0, label))
        searched_from = found           # strictly forward, so a repeated
                                        # phrase later cannot shadow a phase
    end = secs(lines[-1]) - t0

    print(f"\n  {title}")
    print(f"  process: {role_of(block)}")
    print(f"  {'phase':<42}{'start':>7}{'took':>7}")
    print("  " + "-" * 56)
    durations: list[tuple[float, str]] = []
    for i, (at, label) in enumerate(hits):
        nxt = hits[i + 1][0] if i + 1 < len(hits) else end
        took = max(0.0, nxt - at)
        durations.append((took, label))
        print(f"  {label:<42}{at:>6.0f}s{took:>6.0f}s")
    verdict_at = next((at for at, l in hits if l.startswith("VERDICT")), None)
    print("  " + "-" * 56)
    if verdict_at is not None:
        print(f"  {'TO THE VERDICT':<42}{'':>7}{verdict_at:>6.0f}s")
        if verdict_at > 25:
            print(f"  His bar is 25s. This is {verdict_at:.0f}s.")
    print(f"  {'whole launch in the log':<42}{'':>7}{end:>6.0f}s")

    durations.sort(reverse=True)
    worst = [d for d in durations if not d[1].startswith("VERDICT")][:3]
    if worst:
        print("\n  the three longest phases:")
        for took, label in worst:
            print(f"    {took:>5.0f}s  {label}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    show_all = "--all" in sys.argv[1:]
    path = Path(args[0]) if args else newest_log()
    if path is None or not path.exists():
        print("no log found. Run the app, or pass a path.")
        return 1
    if not path.is_absolute():
        path = ROOT / path

    text = io.open(path, encoding="utf-8", errors="replace").read()
    # Each launch writes a dated header, so the file holds several.
    blocks = re.split(r"=== .*camera lab started.*===", text)
    blocks = [b for b in blocks if len([1 for ln in b.splitlines()
                                        if secs(ln) is not None]) > 5]
    if not blocks:
        print(f"{path.name}: no launches with enough timestamps in it")
        return 1

    print("=" * 60)
    print(f"LAUNCH TIMING -- {path.name}  ({len(blocks)} launch(es) in it)")
    print("=" * 60)
    chosen = blocks if show_all else blocks[-1:]
    for i, b in enumerate(chosen, start=len(blocks) - len(chosen) + 1):
        report(b, f"launch {i} of {len(blocks)}")
    if not show_all and len(blocks) > 1:
        print(f"\n  ({len(blocks) - 1} earlier launch(es) in this file; "
              f"pass --all)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
