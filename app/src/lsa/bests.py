"""The best round this machine has seen: stored, validated, and never regressed.

PURE, like `ghost.py` (which it imports) and for the same reason: no Qt, no
camera, no file. `AppSettings` holds the dict and writes it; this module decides
what may go into it and reads it back safely. `settings.py` imports it (to
normalise whatever the JSON held), so it must not import anything that imports
`settings`: that is why efficiency is handed in as a number rather than computed
from `modes/efficiency.py`.

WHAT IS STORED, AND WHY AS INGREDIENTS
--------------------------------------
Keyed per DRILL, per MAGAZINE SIZE and per PLAYER COUNT:

    "static|m17|p1": {
        "score":      {"value": 415.7, "points": 165, "seconds": 9.34, "shots": 17,
                       "basis": "measured", "declared_waits": 0,
                       "ghost_points": 170, "ghost_seconds": 4.0,
                       "ghost_interval": 0.25, "when": "2026-10-21T20:12:03"},
        "efficiency": {"value": 17.74, "points": 165, "seconds": 9.34, "shots": 17,
                       "basis": "measured", "when": "2026-10-21T20:12:03"}}

  * per drill, because drills hand the ghost very different amounts of dead time,
    so one global best would make most drills unbeatable for ever;
  * per magazine size, because a best set on a 5-round magazine cannot be beaten
    on 17 and the magazine is meant to become a setting;
  * per player count, because a two-player half is a different test from a solo
    round.

The value is a DERIVATION (the ghost model has changed twice in one day) and goes
stale when the model moves; the points, the seconds and the shot count are
MEASUREMENTS and do not. They are stored beside it so an old best can be
re-scored instead of orphaned, and a score record carries the `GHOST_INTERVAL` it
was computed against: `ghost.py` says CHANGING THAT NUMBER INVALIDATES EVERY
STORED BEST, and a record whose interval no longer matches is therefore IGNORED
for comparison (and overwritten by the next fit round), never compared as though
it still meant the same thing. An efficiency record is a pure measurement and is
never stale.

`basis` is the state the round was set under: "measured" (every one of the
ghost's waits came from the round itself), "declared" (some came from the
drill's own average) or "lower_bound" (the drill makes the shooter wait and the
waits were not supplied, so the ghost was faster than a real one could be and the
true score was AT LEAST the stored value).

A LOWER-BOUND ROUND IS RECORDED (his ruling, 2026-10-22: "if the app shows the
score on screen, then this is the score; then the app has to honor that score and
record it as a record"). It used to be refused, which left the screen showing a
number the book would not keep. It is stored WITH ITS MARK, never as a measured
score: the value is a floor, so a stored floor can only UNDERSTATE what that round
was worth, never inflate it, and a later round that beats the floor replaces it
(and carries its own basis). Whether the same holds for the efficiency record: an
efficiency is points over the printed seconds and involves no ghost at all, so it
is exact whatever the basis; the basis stored beside it is the ROUND's, and says
only which state the round was in.

No stored file needs migrating: the two older values are unchanged and the new one
is purely additive (a file written now and read by an older build would drop the
floor record, never misread it).

WHAT THIS MODULE REFUSES
------------------------
Anything it cannot vouch for. A hand-edited or corrupt value (a string where a
number goes, a NaN, a negative, a boolean, a key that is not one of ours, a
record missing an ingredient) is dropped, never raised on: the loader swallowing
a corrupt file and the app silently starting from nothing is already the shape of
bug this project fears (`atomic.py`), so a bad record costs that record and not
the file. The dict is capped so a runaway edit cannot bloat the settings file.
"""

from __future__ import annotations

import math
import numbers
import re
import time
from typing import Any, Optional

from . import ghost as _ghost

BASIS_MEASURED = "measured"
BASIS_DECLARED = "declared"
BASIS_LOWER = "lower_bound"        # a FLOOR: the true score was at least the stored value
BASES = (BASIS_MEASURED, BASIS_DECLARED, BASIS_LOWER)

KIND_SCORE = "score"
KIND_EFFICIENCY = "efficiency"
KINDS = (KIND_SCORE, KIND_EFFICIENCY)

# More keys than there are drills x magazine sizes x player counts he will ever
# use; a bound, not a target.
MAX_ENTRIES = 256
VALUE_MAX = 1.0e6                  # a score is 0..~1000 and an efficiency a few tens
_KEY = re.compile(r"^[A-Za-z0-9_\-]{1,40}\|m[1-9][0-9]?\|p[12]$")


def best_key(drill: Any, magazine: Any, players: Any) -> str:
    """The key a best is stored under: drill, magazine size, player count.

    Always a key `normalise_bests` accepts: the drill is reduced to its safe
    characters, the magazine to a whole number >= 1, the player count to 1 or 2.
    """
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", str(drill))[:40] or "unknown"
    try:
        mag = max(1, min(99, int(magazine)))
    except (TypeError, ValueError, OverflowError):
        mag = 1
    two = False
    try:
        two = int(players) == 2 and not isinstance(players, bool)
    except (TypeError, ValueError):
        pass
    return f"{name}|m{mag}|p{2 if two else 1}"


def stamp() -> str:
    """Wall-clock time for a record's `when`. Informational only: nothing
    compares it, so it is not on the perf_counter clock the deadlines share."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _num(v: Any, lo: float, hi: float, whole: bool = False):
    """`v` as a number in [lo, hi], or None. Booleans are not numbers here (True
    would read as 1), strings are not, NaN and infinity are not."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return None
    f = float(v)
    if not math.isfinite(f) or f < lo or f > hi:
        return None
    if whole:
        return int(f) if f == int(f) else None
    return f


def normalise_record(kind: str, raw: Any) -> Optional[dict]:
    """One stored record as a clean dict, or None when it cannot be vouched for.

    Required: a positive `value`, and the three measured ingredients (`points`,
    `seconds`, `shots`) and a known `basis`. Without the ingredients an old best
    could not be re-scored, so it is not kept.
    """
    if kind not in KINDS or not isinstance(raw, dict):
        return None
    value = _num(raw.get("value"), 0.0, VALUE_MAX)
    points = _num(raw.get("points"), 0, VALUE_MAX, whole=True)
    seconds = _num(raw.get("seconds"), 0.0, 86400.0)
    shots = _num(raw.get("shots"), 1, 1000, whole=True)
    basis = raw.get("basis")
    if value is None or value <= 0.0 or points is None or seconds is None \
            or shots is None or basis not in BASES:
        return None
    out: dict = {"value": value, "points": points, "seconds": seconds,
                 "shots": shots, "basis": basis}
    if kind == KIND_SCORE:
        dw = _num(raw.get("declared_waits"), 0, 1000, whole=True)
        out["declared_waits"] = 0 if dw is None else dw
        gp = _num(raw.get("ghost_points"), 1, VALUE_MAX, whole=True)
        if gp is not None:
            out["ghost_points"] = gp
        gs = _num(raw.get("ghost_seconds"), 0.0, 86400.0)
        if gs is not None:
            out["ghost_seconds"] = gs
        gi = _num(raw.get("ghost_interval"), 0.0, 100.0)
        if gi is not None:
            out["ghost_interval"] = gi
    when = raw.get("when")
    out["when"] = when[:40] if isinstance(when, str) else ""
    return out


def normalise_bests(raw: Any) -> dict:
    """Whatever the settings file held, as a dict that is safe to read and write.

    Never raises. Unknown keys, malformed keys, non-dict entries and invalid
    records are dropped; what survives is a fresh dict (never the caller's), in a
    deterministic order, capped at `MAX_ENTRIES`.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key in sorted(k for k in raw if isinstance(k, str)):
        if len(out) >= MAX_ENTRIES:
            break
        if not _KEY.match(key):
            continue
        entry = raw[key]
        if not isinstance(entry, dict):
            continue
        clean = {}
        for kind in KINDS:
            rec = normalise_record(kind, entry.get(kind))
            if rec is not None:
                clean[kind] = rec
        if clean:
            out[key] = clean
    return out


def score_is_current(rec: Optional[dict]) -> bool:
    """A stored score is comparable only if it was computed against the ghost
    interval in force now (`ghost.GHOST_INTERVAL`). A record with no interval is
    one nobody can vouch for."""
    if rec is None:
        return False
    gi = rec.get("ghost_interval")
    return gi is not None and abs(float(gi) - _ghost.GHOST_INTERVAL) < 1e-9


# ---------------------------------------------------------------------------
# The book
# ---------------------------------------------------------------------------

class BestBook:
    """A view onto the dict `AppSettings.bests` holds. It SHARES that dict, so
    what it writes is what `AppSettings.save` persists, and it never keeps a
    second copy of a fact.

    Reads re-validate (a live edit of the file cannot hand the screen a string);
    `offer` only ever RAISES a record, so no ordering of two offers can lose a
    better one; `take_changed` says, once, whether anything was written since it
    was last asked, which is how the app writes the file at most once a round and
    not at all when nothing improved.
    """

    def __init__(self, data: Optional[dict] = None) -> None:
        self.data: dict = data if isinstance(data, dict) else {}
        self._changed = False

    # -- reads ---------------------------------------------------------------

    def record(self, key: str, kind: str) -> Optional[dict]:
        entry = self.data.get(key)
        if not isinstance(entry, dict):
            return None
        return normalise_record(kind, entry.get(kind))

    def best_score(self, key: str) -> Optional[dict]:
        """The stored score for `key`, or None when there is none or it was
        computed against a different ghost (see `score_is_current`)."""
        rec = self.record(key, KIND_SCORE)
        return rec if score_is_current(rec) else None

    def best_efficiency(self, key: str) -> Optional[dict]:
        return self.record(key, KIND_EFFICIENCY)

    # -- writes --------------------------------------------------------------

    def offer(self, key: str, score_rec: Optional[dict] = None,
              eff_rec: Optional[dict] = None) -> bool:
        """Store whichever of the two records BEATS what is stored. True if
        anything was written. A record that is not strictly better, or that does
        not validate, changes nothing."""
        if not _KEY.match(str(key)):
            return False
        wrote = False
        for kind, rec, current in (
                (KIND_SCORE, score_rec, self.best_score(key)),
                (KIND_EFFICIENCY, eff_rec, self.best_efficiency(key))):
            if rec is None:
                continue
            clean = normalise_record(kind, rec)
            if clean is None:
                continue
            if current is not None and not clean["value"] > current["value"]:
                continue
            entry = self.data.get(key)
            if not isinstance(entry, dict):
                if len(self.data) >= MAX_ENTRIES:
                    continue
                entry = {}
                self.data[key] = entry
            entry[kind] = clean
            wrote = True
        if wrote:
            self._changed = True
        return wrote

    def take_changed(self) -> bool:
        """One-shot: was anything written since the last call?"""
        out, self._changed = self._changed, False
        return out
