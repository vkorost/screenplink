"""Make a suite's own output unable to crash it.

`import safe_stdio` at the top of a suite, for its side effect.

Why (2026-10-22): `test_feedback` died inside its own `print` with
`UnicodeEncodeError: 'charmap' codec can't encode character '\\u2733'`. The
check printed a detail that carried a live desktop window title, and the
console encoding on this machine is cp1252. A suite that crashes on its own
printer reports nothing about the code under test, and the exit status looks
like a defect in the app.

`errors="backslashreplace"` rather than switching to UTF-8: the reader of this
output (a pipe, the pre-commit hook, a runner) decodes with ITS OWN locale, and
a UTF-8 stream into a cp1252 reader is mojibake. Backslash-replace keeps the
stream in the encoding the reader already expects, never raises, and leaves the
offending character readable as its escape (`\\u2733`), so a `[FAIL]` line still
says what it measured.
"""

from __future__ import annotations

import sys


def harden(stream):
    """Make one text stream unable to raise on an unencodable character."""
    reconf = getattr(stream, "reconfigure", None)
    if reconf is not None:                     # a captured stream may not have it
        try:
            reconf(errors="backslashreplace")
        except (ValueError, OSError):
            pass
    return stream


def install() -> None:
    for stream in (sys.stdout, sys.stderr):
        harden(stream)


install()
