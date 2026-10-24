"""Writing a file the user owns, so a kill cannot leave it half-written.

Two files on this machine are expensive to lose and cheap to protect:

  calibration.json    the 1mm mapping, measured from 18 photographs, plus
                      the sensitivity regions learned from his real shots
  settings.local.json the camera chosen by name, the measured exposure

Both were written with a bare `Path.write_text`, which opens the real file,
truncates it to zero, and only then starts writing. The window between those
two is small and this project spends it badly: these notes record force-kills
at shutdown (which is how the camera got wedged for days), a crash from timers
firing into a destroyed scene, and `os._exit(0)` in `closeEvent` by design.
Any of them landing inside that window leaves a truncated or empty file, and
the app's own loader treats a corrupt file as "no calibration" and silently
recalibrates -- so the loss does not even announce itself.

`os.replace` is atomic on Windows as well as POSIX: a reader sees the old file
or the new one, never a partial one. The temp file is the only thing a crash
can damage, and it is named so a stray one is identifiable rather than
mysterious.

Not used for the recording sidecars (`settings.json`, `summary.json` beside an
AVI). Those are disposable by design and the notes already require that a
failed sidecar never interrupts anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# WHY THE LAST SAVE FAILED. A bare False cannot be debugged, only
# re-run (RULES C3) -- and this one was: the save returned False, the
# caller ignored it, the log said "saved", and six harness launches paid
# for eighteen markers each because there was nothing on disk to check.
# Even once the result was checked, "it was NOT saved" without a reason
# sent me looking at the wrong half of the function.
last_error: str = ""


def _plain(obj):
    """Last resort for a value `json` cannot encode: make it a Python one.

    EVERY FIELD OF A REAL MAPPING CAN ARRIVE AS A NUMPY SCALAR. The samples
    come from `find_marker`, the residuals from the fit, the held-out
    report from arithmetic on both -- and `json.dumps` raises TypeError on
    all of them. `Mapping.save` caught that, returned False, and its
    callers ignored the result, so the app logged "saved" with nothing on
    disk. Six consecutive harness launches then had no saved mapping for
    the three-second check to confirm, paid for eighteen markers each, and
    took 81 seconds instead of 40 -- with twelve suites green.

    Coercing the known fields by hand fixed three of them and missed
    `held_out`, which is exactly how this project loses an afternoon: the
    instance in front of me, not the rule. So the conversion lives HERE,
    at the one gate every save passes through, and a field added later is
    covered without anyone remembering.

    `.item()` turns a numpy scalar into a float or int; `.tolist()` turns
    an array into nested lists. Anything else still raises, because
    silently writing a repr of an object nobody meant to persist is worse
    than refusing.
    """
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:                                    # noqa: BLE001
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:                                    # noqa: BLE001
            pass
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def write_json_atomic(path: Path, payload: Any, indent: int = 2) -> bool:
    """Serialise `payload` to `path` as JSON, atomically. True if it landed.

    Serialising BEFORE touching the filesystem matters: an object that cannot
    be encoded must not cost the existing file, and `json.dump` to an open
    handle would already have truncated it.
    """
    global last_error
    last_error = ""
    try:
        text = json.dumps(payload, indent=indent, default=_plain)
    except Exception as exc:
        last_error = f"could not encode it as JSON: {exc}"
        return False

    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with io_open(tmp) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())      # the bytes, not just the handle
        os.replace(tmp, path)          # atomic on Windows and POSIX
        return True
    except Exception as exc:
        last_error = f"could not write {tmp}: {exc!r}"
        try:
            tmp.unlink()
        except Exception:
            pass
        return False


def io_open(path: Path):
    """UTF-8, newline-agnostic. Separate so the write path stays readable."""
    return open(path, "w", encoding="utf-8", newline="\n")
