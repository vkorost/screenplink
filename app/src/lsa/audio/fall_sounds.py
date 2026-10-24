"""What a figure sounds like when it goes down.

HIS RULE (2026-10-20): "on the zombie02 target I need the sound to be exactly
'Banshee-02.mp3' when the target goes down; on the other targets the sounds should
be 'Male-Grunt-??.mp3' -- just use the random sound from the 6 grunt sounds there;
that includes the animated shooter target when it goes down."

BUILT AS DATA, NOT AS A LIST OF FIFTEEN. `FALL_SOUND_BY_FIGURE` names the figures
that have a sound of their own; every other name -- including one he drops into
`targets/` tomorrow, and the animated soldier, which is not a figure at all -- gets a
random grunt, and nothing crashes or falls silent for a name nobody listed. Checks
test that behaviour (the named figure gets its sound, an unnamed one gets a grunt,
an unknown one still gets a grunt), never today's filenames.

IT REPLACES `mixer.FALL_SOUND` (the cinematic gun he shortened to 0.81 s), which is
still registered and still hidden from the gunshot picker, because every dropped or
repurposed sample here stays reserved so it can never be drawn as an ordinary bang.

RANDOMNESS goes through a seedable `random.Random` (testable), and a grunt never
repeats the one immediately before it, the way the figure draw avoids immediate
repeats.

As with every effect, the sound FOLLOWS the gunshot by a millisecond and never
replaces it: that is `ShotOutcome.sound`'s job and is not decided here.
"""

from __future__ import annotations

import random

from .mixer import BANSHEE_01_SOUND, BANSHEE_SOUND, GRUNT_SOUNDS

# Figure name -> its own fall sound. Everything else gets a grunt.
FALL_SOUND_BY_FIGURE: "dict[str, str]" = {
    "zombie02": BANSHEE_SOUND,
    "alien01": BANSHEE_01_SOUND,
}


class FallSounds:
    """Picks the sound for one figure falling."""

    def __init__(self, rng: "random.Random | None" = None) -> None:
        self.rng = rng if rng is not None else random.Random()
        self._prev_grunt: "str | None" = None

    def pick(self, figure: "str | None") -> str:
        own = FALL_SOUND_BY_FIGURE.get(figure) if figure else None
        if own is not None:
            return own
        pool = [g for g in GRUNT_SOUNDS if g != self._prev_grunt] or list(GRUNT_SOUNDS)
        choice = pool[self.rng.randrange(len(pool))]
        self._prev_grunt = choice
        return choice
