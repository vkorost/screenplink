"""A line of text made of RUNS, each with its own weight and colour -- and the one
rule that decides which runs a drawn line has.

WHY THIS EXISTS (his ruling, 2026-10-22): on the round screen the NUMBERS are
bold and one step brighter than their labels ("lets do one step up as GREY_HI;
that should be enough"), on every number and not only the two BEST lines. Qt will
centre a string with one font and one pen, so a line with a bold run in the middle
of it cannot go through `drawText(rect, AlignCenter, ...)`; `ModeHost._draw_lines`
lays such a line out run by run (see there). This module holds the pure half:
the type, and the rule.

PURE: no Qt, no font, no painter. Colour is whatever object the caller's painter
accepts (a QColor); this module only carries it. Widths are NOT computed here: a
bold run is wider than the same characters regular, and only the font knows by
how much (RULES B5), so measuring lives where the fonts are.

THE TYPE IS A `str`. A `RunLine` is the line's plain text with the runs attached,
so a caller that does not care (the log, a test comparing the words, `startswith`,
`re.fullmatch`) sees an ordinary string, and `_draw_lines` is the one place that
looks for `.runs`. The price, stated: a check that measures a `RunLine` as a
joined string with ONE font gets a plausible, too-small answer instead of an
error. Measure it with `ModeHost.line_width`, which sums each run under its own
font, never with a font's `horizontalAdvance` on the string.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple, Optional


class Run(NamedTuple):
    text: str
    bold: Optional[bool] = None       # None: inherit the call's weight (a label stays as it was)
    color: Any = None                 # None: inherit the painter's pen (a label stays as it was)


class RunLine(str):
    """The text of a line, with the runs it is drawn from."""

    runs: "tuple[Run, ...]"

    def __new__(cls, runs):
        runs = tuple(runs)
        obj = super().__new__(cls, "".join(r.text for r in runs))
        obj.runs = runs
        return obj


# A number as it appears on the round screen: digits, an optional decimal part,
# an optional trailing percent sign ("17", "10.7", "27%", "1000"). Not preceded by
# a letter, a digit or a point, so a digit inside a word is not picked out, and
# "--" (no figure) and "ABOUT" / "AT LEAST" (qualifiers) are never numeric.
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?%?")

# The prefix of the two announcements that a record was set. They have no number
# inside to pick out, so the WHOLE line is the emphasis ("NEW BEST SCORE RECORDED").
# Imported, not retyped: `scorecard` builds both announcements from it.
from ..scorecard import NEW_BEST_PREFIX as WHOLE_LINE_PREFIX  # noqa: E402
# The two-player result line ("RIGHT WINS by 11 points", "MATCH: LEFT leads by 12
# points", "TIED") is the OTHER whole-line case (2026-10-22): the same lift as a
# record announcement, recognised by `versus.is_result_line` because its start is a
# side name, not one fixed prefix. One rule with two ways to qualify, not a second
# mechanism.
from .versus import is_result_line as _is_result_line  # noqa: E402


def emphasise(line: str, color: Any, whole_prefix: str = WHOLE_LINE_PREFIX) -> RunLine:
    """`line` split into runs: every numeric span bold and in `color`, the words
    around them left exactly as the call draws them. ONE rule for every block
    that is emphasised, so a number that changes shape ("1000", "203",
    "10.7 seconds") needs no new case, and a line with no number is one plain run.

    A line that starts with `whole_prefix`, or is a two-player result line, is
    bold and coloured end to end."""
    text = str(line)
    if (whole_prefix and text.startswith(whole_prefix)) or _is_result_line(text):
        return RunLine([Run(text, True, color)])
    runs: "list[Run]" = []
    at = 0
    for m in _NUMBER.finditer(text):
        if m.start() > at:
            runs.append(Run(text[at:m.start()]))
        runs.append(Run(m.group(0), True, color))
        at = m.end()
    if at < len(text) or not runs:
        runs.append(Run(text[at:]))
    return RunLine(runs)
