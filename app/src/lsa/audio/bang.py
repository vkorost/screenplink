"""Telling the microphone what the app has just played.

Two things live here, and neither needs Qt, a camera or a screen, so both are
tested by driving them with a fake clock against the real listener:

`play_own_sound`
    Plays a sound AND tells the listener about it, in one place. The call
    site used to do both inline, log "play() refused it" when it did, and then
    register the bang with the listener anyway -- so a bang that reached no
    speaker was filed as a measurement of the room. This is the only path from
    "the app is making a noise" to the listener, so the trust decision is made
    once.

    It is also the only path from the APP to the ROOM: `.play(` may appear
    nowhere else in `src/` except the announcer's speech
    (tests/test_rules.py, section U). That covers the gunshot, the effect that
    follows it, the dry-pull pop, the selection cue, a target appearing, the
    operator console's "Test" button and the teacher's sweeps. The console's
    button used to call the mixer directly: a real gunshot at session volume,
    registered with nobody, which a calibrated microphone counted as a trigger
    pull ("HEARD a shot with no detection"), so the gun test's N was wrong by
    the number of times it was pressed. The effect and cue sounds had their own
    copies of the mute, which is the same fault one level down.

    SPEECH IS THE ONE LEGITIMATE OTHER CALLER (`audio/voice.py`). A spoken line
    runs for seconds, not 0.8s; it is muted flatly for its own duration plus the
    room tail, which is what a line that long needs, and it is not a bang whose
    level could be discounted block by block. The rule is an allowlist with one
    entry and a reason, not a blanket ban.

`CouplingTeacher`
    Teaches the listener how loud the speakers are at the microphone BEFORE
    he shoots, instead of from his own string. The coupling used to be learned
    only from bangs his shooting produced, and a bang inside the previous
    bang's loud head is discarded as overlapped. Measured from a cold start:

        gap 1.00s  heard 12 of 12   coupling learned
        gap 0.70s  heard  7 of 12   coupling learned
        gap 0.55s  heard  1 of 12   coupling NEVER learned
        gap 0.50s  heard  1 of 12   coupling NEVER learned

    Below about a 0.6s cadence it could not learn at all and the old deafness
    applied for the whole session with nothing said -- and two pistols at once,
    a stated design driver, is well inside that region.

    It plays three short quiet sweeps while the app already owns the screen and
    its detections are already the app's own (so nothing can be mistaken for a
    shot, and no bang reads as one -- RULES B9). It is announced in the log and
    on the target screen as a measurement, it is bounded in time, and it
    stands down the instant anything that could be a trigger pull is allowed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lsa.audio.listener import OWN_NOT_PLAYED, ClickListener
from lsa.audio.mixer import Mixer, synth_probe


PROBE_NAME = "_coupling probe"        # "_": not offered as a gunshot

# What the listener is told a sound IS, which decides whether it may teach the
# coupling: only the gunshot the shooter chose and the teacher's own sweeps
# are readings. An effect or a pop is short and overlaps the bang that causes
# it, so it is registered (it must still be discounted or muted) and never read.
SOURCE_BANG = "bang"
SOURCE_PROBE = "probe"
SOURCE_EFFECT = "effect"


@dataclass
class Bang:
    played: bool          # did it reach a speaker, as far as we can tell
    mode: str             # one of the listener's OWN_* outcomes
    why: str = ""         # when not played: the reason, for the log


def fallback_mute_s(mixer: Mixer, sound: str, tail_s: float,
                    min_mute_s: float = 0.35) -> float:
    """How long the microphone is deaf for `sound` while the coupling is unknown.

    The sample's own LOUD part at the volume it is playing at (a louder app
    rings above the microphone's absolute threshold for longer, so the
    fraction scales with the gain -- `Mixer.mute_fraction`), plus a tail for
    the room. Only the FALLBACK: once the coupling is measured nothing is
    muted for a bang, its level is discounted instead.
    """
    return max(min_mute_s, mixer.loud_duration(sound, gain=mixer.master_gain) + tail_s)


def play_own_sound(mixer: Mixer, listener: ClickListener, sound: str,
                   tail_s: float, min_mute_s: float = 0.35, *,
                   fallback_s: "float | None" = None,
                   source: str = SOURCE_BANG, label: str = "") -> Bang:
    """Play `sound` and tell the listener, or tell it nothing was played.

    "Played" needs BOTH `mixer.play()` and a live stream. PortAudio keeps
    accepting writes to a callback stream whose endpoint has gone, so `play()`
    alone says True into silence until the 5s keepalive notices.

    `fallback_s` replaces the computed flat mute (the teacher's sweeps are not
    a gunshot sample). `label` is what the listener records the sound as.
    """
    ok = mixer.play(sound)
    if not ok:
        why = "play() refused it"
    elif not mixer.audible:
        ok, why = False, ("the output stream is not running (the endpoint "
                          "went away and PortAudio did not say so)")
    else:
        why = ""
    if not ok:
        listener.expect_own_sound(np.zeros(0, np.float32), 0.0, 0.0,
                                  played=False, output_id=mixer.epoch,
                                  sample=label or sound, source=source)
        return Bang(False, OWN_NOT_PLAYED, why)
    dt, env = mixer.own_envelope(sound)
    # The flat mute is only the fallback until the coupling is known: the
    # sample's own loud part plus a tail for the room.
    fallback = (fallback_s if fallback_s is not None
                else fallback_mute_s(mixer, sound, tail_s, min_mute_s))
    mode = listener.expect_own_sound(
        np.minimum(1.0, env * mixer.master_gain), dt, fallback,
        played=True, output_id=mixer.epoch, gain=mixer.master_gain,
        sample=label or sound, source=source)
    return Bang(True, mode)


class CouplingTeacher:
    """Three quiet sweeps, one at a time, while nothing could be a shot.

    `tick(may_play)` is called from an existing periodic tick (the HUD's, five
    times a second); it owns no timer, so there is nothing to stand down at
    shutdown and nothing for a force-quit to leave running (RULES A3).
    `may_play` is the caller's statement that the app owns the screen and its
    own detections right now. If it turns false mid-way the probe stops there.
    """

    BEEPS = 3
    SPACING_S = 0.7
    # The whole thing, begin to verdict. Three sweeps and their ring-down fit
    # in about 2.5s; the rest is the deadline for the HUD's tick granularity.
    DEADLINE_S = 4.5
    SWEEP_S = 0.12
    # How loud the sweep is, decided by what the reading needs and not by
    # taste. A reading must clear 3x the room's floor to count, and the floor
    # on this bench is ~0.0127 (-48 dBFS noise), so the sweep has to put at
    # least ~0.1 of output level into the room to read at twice the bar when
    # the coupling is 0.8. His saved volume is 0.05, where even a full-scale
    # sweep only reaches 0.05; so: scale the sweep up as the volume goes down
    # to reach TARGET_OUT, never below LEVEL (so at normal volumes it is a
    # soft half-scale whoop, not a bang) and never above 1.0 (so it is never
    # louder than the gunshot's own 1.17 peak). Measured in the fake room, a
    # half-scale sweep FAILED at volume 0.05 and a full one passed:
    #   volume    0.05   0.10   0.20   0.40   (true coupling 0.8)
    #   level 0.5 fail   pass   pass   pass
    #   adaptive  pass   pass   pass   pass   output peaks 0.05 0.10 0.10 0.20
    LEVEL = 0.5
    TARGET_OUT = 0.1

    def __init__(self, mixer: Mixer, listener: ClickListener, log,
                 clock=time.perf_counter) -> None:
        self.mixer, self.listener, self._log = mixer, listener, log
        self._clock = clock
        self.state = "idle"   # idle | running | done | failed | aborted
        self.reason = ""
        self._played = 0
        self._next = 0.0
        self._started = 0.0
        self._last_end = 0.0
        # One attempt per (microphone session, output stream). A probe that
        # fails is not retried in a loop (RULES D10); a new stream or a new
        # microphone is a new question.
        self._tried_for: tuple | None = None

    @property
    def active(self) -> bool:
        return self.state == "running"

    def _key(self) -> tuple:
        return (self.listener.device, self.mixer.epoch)

    def begin(self) -> bool:
        """Start, if it is worth it. Says why not, once, when it is not."""
        if self.state == "running":
            return False
        stale = self.listener.coupling_stale
        if self.listener.coupling is not None and not stale:
            return False                      # known and fresh: nothing to teach
        if not self.listener.running:
            return False                      # no microphone: nothing to hear it
        if not self.mixer.audible:
            self._log("  microphone probe skipped: the sound output is not "
                      "running.")
            return False
        # One attempt per key, EXCEPT after a success whose result has since
        # gone: gone stale, or forgotten because a bang read louder than
        # predicted. That is a new question, asked once per loss and only at a
        # moment the app owns the detections. The key does not move when a
        # coupling expires, so without this the probe's answer was permanent
        # for the key and its expiry was permanent deafness.
        if self._tried_for == self._key() and self.state != "done":
            return False
        again = self._tried_for is not None and self.state == "done"
        self._tried_for = self._key()
        level = min(1.0, max(self.LEVEL,
                             self.TARGET_OUT / max(self.mixer.master_gain, 1e-3)))
        self.mixer.register(PROBE_NAME, synth_probe(
            self.mixer.samplerate, dur=self.SWEEP_S, level=level))
        now = self._clock()
        self.state, self.reason = "running", ""
        self._played = 0
        self._started = now
        self._next = now
        why = (" (the earlier measurement is stale or was withdrawn, so it is "
               "being redone)" if again else "")
        self._log(f"  MEASURING THE MICROPHONE (not shooting){why}: {self.BEEPS} "
                  f"short quiet sweeps, about {self.BEEPS * self.SPACING_S:.0f}s, "
                  f"so the app can tell its own gunshot from yours when you "
                  f"shoot fast. They are not shots. The volume is "
                  f"{self.mixer.master_gain:.2f}, the output "
                  f"{self.mixer.output_name}.")
        return True

    def tick(self, may_play: bool) -> None:
        if self.state != "running":
            return
        now = self._clock()
        if not may_play:
            self._stop("aborted", "something that could be a trigger pull was "
                                  "allowed again")
            return
        if now - self._started > self.DEADLINE_S:
            self._stop("failed", f"did not finish in {self.DEADLINE_S:g}s")
            return
        if self._played < self.BEEPS:
            if now >= self._next:
                self._beep(now)
            return
        # All played: wait for the last one to ring down and be filed.
        if now >= self._last_end:
            self._finish()

    def _beep(self, now: float) -> None:
        # Flat-muted by the listener while the coupling is unknown, so the
        # sweep is not itself heard as a click. Through the one path to the
        # speakers, like every other sound the app makes.
        tail = self.listener.OWN_SLACK_S + 0.25
        bang = play_own_sound(self.mixer, self.listener, PROBE_NAME, tail,
                              fallback_s=self.SWEEP_S + tail,
                              source=SOURCE_PROBE, label="coupling probe")
        if not bang.played:
            self._stop("failed", "the sweep could not be played")
            return
        self._played += 1
        self._next = now + self.SPACING_S
        # The reading is filed once the sweep's envelope has run out; give
        # it a few audio blocks beyond that.
        self._last_end = now + self.SWEEP_S + self.listener.OWN_SLACK_S + 0.1

    def _stop(self, state: str, why: str) -> None:
        self.state, self.reason = state, why
        self._log(f"  microphone measurement {state}: {why}. Until the "
                  f"coupling is known the listener stays open if the app's "
                  f"own gunshot cannot reach the click bar, and mutes flatly "
                  f"if it can; it will be learned from your shots instead "
                  f"(which needs them more than 0.6s apart).")

    def _finish(self) -> None:
        c = self.listener.coupling
        self.reason = self.listener.describe_coupling()
        if c is not None:
            self.state = "done"
            self._log(f"  microphone measured: {self.reason}.")
        else:
            self.state = "failed"
            # The second line the owner asked for: when the number is not
            # believable, name the device, because the usual cause is the
            # wrong one -- speakers that are not audible to this microphone.
            self._log(f"  microphone measurement FAILED: {self.reason}.")
            self._log(f"    the speakers ({self.mixer.output_name}, volume "
                      f"{self.mixer.master_gain:.2f}) are not audible to the "
                      f"microphone, or the output is not where the sound "
                      f"actually comes out. Until it is known the listener "
                      f"stays open if the gunshot is too quiet to reach the "
                      f"click bar, and mutes flatly if it is not.")
