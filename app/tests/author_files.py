"""Files the AUTHOR maintains, which this repository deliberately does not ship.

Three of them exist: `RULES.md` and `.claude/settings.json` -- the agreement
between him and me, ruled out of this repo in as many words ("those are between
me and you, people will not understand what that file is for") -- and
`requirements/phrases_to_generate.md`, the assignment the spoken lines were
generated from.

The checks that read them are worth running wherever they DO exist, so the path
is resolved rather than assumed. ONE module does it, because the alternative is
each suite inventing its own convention for the same question: that is
`_screen_busy()` again, a central rule consulted by one of the six places that
needed it, which cost this project two sessions.

THREE BEHAVIOURS, and the difference between them is the whole point:

  * found          -> the check runs at full strength, exactly as before.
  * absent         -> a VISIBLE skip, named in the suite's summary, which then
                      refuses to read as a clean sweep. Not a pass. A
                      contributor cannot satisfy a check about a file they do
                      not have, and a warning nobody can satisfy teaches people
                      to ignore warnings -- paid for twice here, telling him to
                      dim a room that was already as dark as it gets.
  * named but gone -> a FAILURE. If the variable is set and the file is not
                      there, the pointer is wrong. That is a real fault, and
                      skipping it quietly is how a check goes hollow on the one
                      machine where anything is enforced at all.
"""

from __future__ import annotations

import os
from pathlib import Path

#: env var -> (path relative to a tree, what the check it feeds is for)
KNOWN = {
    "LSA_RULES_MD": ("RULES.md", "the rule registry"),
    "LSA_CLAUDE_SETTINGS": (".claude/settings.json", "the session hook"),
    "LSA_VOICE_SPEC": ("requirements/phrases_to_generate.md",
                       "the spoken wording against its assignment"),
}


def author_file(root: Path, env: str) -> tuple[Path | None, str, str]:
    """Resolve one author file.

    Returns `(path, failure, skip)`. Exactly one of the three is meaningful:
    a path when it was found, a `failure` message when the variable names
    something that is not there, and a `skip` note when nobody said where it is.
    The caller reports -- this module decides nothing about verdicts, so a suite
    cannot accidentally be made green by it.
    """
    rel, what = KNOWN[env]
    named = os.environ.get(env, "").strip()
    if named:
        at = Path(named)
        if at.is_file():
            return at, "", ""
        return (None,
                f"{env}={named!r} does not exist. Unset it to skip {rel} "
                f"deliberately, or point it at the real file.",
                "")
    for cand in (root / rel, root.parent / rel):    # app/, then the repo root
        if cand.is_file():
            return cand, "", ""
    return (None, "",
            f"needs {rel} ({what}) -- the author's, not the app's. "
            f"{env} runs it.")
