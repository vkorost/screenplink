"""Every way a sound can leave the app, found by AST. One definition, two readers.

WHY IT IS A HELPER. The rule it enforces ("a sound the app makes goes through
`lsa.audio.bang`, so the microphone is told") is wanted in two places that run
in different tiers: `test_rules.py` (always run, cheap, reads the real tree) and
`test_audio_own.py` (which the mutation runner re-runs in a scratch copy of
`src/`, so a reintroduced second path can be shown to turn it red). Two copies
of an AST walk would drift; this is the one.

THE FAULT (2026-10-16). `play_gunshot` was "the one path from a shot to the
listener" -- true as worded, while the operator console's Test button called
`mixer.play(...)` directly: a real gunshot at session volume with no
registration, no mute, no discount and no log line. A calibrated microphone
heard it and counted it ("HEARD a shot with no detection"), so the gun test's N
was wrong by the number of times it was pressed. One instance fixed, one
standing. The effect that follows a bang, the selection cue and a target
appearing had the same shape, each with its own copy of the mute.

SPEECH IS THE ONE LEGITIMATE OTHER CALLER, and the rule says so as an allowlist
entry with a reason, not as a blanket ban. A spoken line runs for seconds; it is
muted flatly for its own duration plus the room tail BEFORE it plays, which is
what a line that long needs, and it is not a bang whose level could be
discounted block by block.
"""

from __future__ import annotations

import ast
from pathlib import Path

# file (relative to src/) -> why it may call `.play(` itself
PLAY_ALLOWED = {
    "lsa/audio/bang.py": "the one path from the app to the speakers",
    "lsa/audio/voice.py": ("speech: muted flat for the line's own length plus the "
                           "tail BEFORE it plays (`self._mute(mute_s)`), and a "
                           "spoken line is seconds long, not a bang to discount"),
}

# file -> the enclosing functions that may call `.mute_for(`. The one caller is the
# lambda the app hands the announcer, built in `CameraLab.__init__`.
MUTE_FOR_ALLOWED = {
    "lsa/tools/camera_lab.py": {"__init__"},
}

# only the module whose job it is may register a sound with the listener
EXPECT_ALLOWED = {"lsa/audio/bang.py"}


def _calls(src_root: Path, attr: str) -> list[tuple[str, int, str]]:
    """(path under src/, line, enclosing function) of every `<x>.<attr>(` call."""
    out: list[tuple[str, int, str]] = []
    for path in sorted(Path(src_root).rglob("*.py")):
        rel = path.relative_to(src_root).as_posix()
        text = path.read_bytes().decode("utf-8", errors="replace")
        tree = ast.parse(text, filename=str(path))
        owner: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    ln = getattr(inner, "lineno", None)
                    if ln is not None and ln not in owner:
                        owner[ln] = node.name
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == attr):
                out.append((rel, node.lineno, owner.get(node.lineno, "?")))
    return out


def play_sites(src_root: Path) -> list[tuple[str, int, str]]:
    return _calls(src_root, "play")


def mute_for_sites(src_root: Path) -> list[tuple[str, int, str]]:
    return _calls(src_root, "mute_for")


def expect_sites(src_root: Path) -> list[tuple[str, int, str]]:
    return _calls(src_root, "expect_own_sound")


def violations(src_root: Path) -> list[str]:
    """Every call that reaches the room or the listener by a second path."""
    bad: list[str] = []
    for rel, ln, fn in play_sites(src_root):
        if rel not in PLAY_ALLOWED:
            bad.append(f"{rel}:{ln} ({fn}) calls .play(")
    for rel, ln, fn in mute_for_sites(src_root):
        if fn not in MUTE_FOR_ALLOWED.get(rel, set()):
            bad.append(f"{rel}:{ln} ({fn}) calls .mute_for(")
    for rel, ln, fn in expect_sites(src_root):
        if rel not in EXPECT_ALLOWED:
            bad.append(f"{rel}:{ln} ({fn}) calls .expect_own_sound(")
    return bad
