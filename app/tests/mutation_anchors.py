"""Anchor matching for the mutation scripts, independent of line endings.

WHY THIS EXISTS (2026-10-17). `mutate_director.py` read sources with
`newline=""` (raw, CRLF kept) while its anchors are Python triple-quoted
literals, which always contain bare "\n". On the CRLF `camera_lab.py` every
multi-line anchor matched ZERO times, and the property it guarded was intact.
Git hides line-ending flips (`core.autocrlf` is true here), so an editor
re-saving a file could silently disarm any of ~180 anchors with nothing in
review to show it. That is RULES C1: a check that cannot fail.

THE CHOICE: translate the anchor (and its replacement) to the file's own
ending; never normalise the file. Normalise-on-read then restore-on-write
rewrites every line of a file whose endings are mixed, so the mutation would
itself flip endings -- a new instance of the bug. Translating touches only the
bytes the anchor covers, and leaves every other byte exactly as found.

A zero-match is a FAILURE everywhere it is used, reported as "anchor not
found", which is a different problem from a fault that was applied and not
caught.
"""
from __future__ import annotations

CRLF = "\r\n"


def file_ending(text: str) -> str:
    """'\r\n', '\n', or 'mixed' for a file's text read with newline=''."""
    crlf = text.count(CRLF)
    lf = text.count("\n") - crlf
    if crlf and lf:
        return "mixed"
    return CRLF if crlf else "\n"


def _dominant(text: str) -> str:
    crlf = text.count(CRLF)
    return CRLF if crlf > text.count("\n") - crlf else "\n"


def translate(s: str, ending: str) -> str:
    """Rewrite every line ending in `s` as `ending`."""
    return s.replace(CRLF, "\n").replace("\n", ending)


def locate(text: str, old: str) -> tuple[int, str, str]:
    """(match count, ending used, file's ending) for `old` in raw `text`."""
    kind = file_ending(text)
    ending = _dominant(text) if kind == "mixed" else kind
    return text.count(translate(old, ending)), ending, kind


def not_found_reason(text: str, old: str) -> str:
    n, _, kind = locate(text, old)
    why = f"ANCHOR NOT FOUND: matched {n} times, needs exactly 1"
    if kind == "mixed" and "\n" in old:
        why += (" (the file has MIXED line endings, so a multi-line anchor "
                "cannot match it reliably)")
    return why


def apply(text: str, old: str, new: str) -> str | None:
    """`text` with `old` -> `new`, or None unless `old` matches exactly once.

    Both sides are translated to the file's own ending, so no byte outside the
    anchor changes.
    """
    n, ending, _ = locate(text, old)
    if n != 1:
        return None
    return text.replace(translate(old, ending), translate(new, ending))
