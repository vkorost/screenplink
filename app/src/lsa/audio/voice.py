"""Spoken announcements: recorded lines, played through the existing Mixer.

The shooter is downrange looking at the target, not at the operator console,
and the console is hidden behind the fullscreen target window anyway. A short
spoken line is the one channel that reaches him in that position.

THE RULES THIS MODULE EXISTS TO KEEP (each one learned on this project):

1. SPEECH MUST NOT DEAFEN THE SHOT COUNTER. The microphone is the only
   independent count of his trigger pulls, and the app mutes it while it makes
   a noise itself (otherwise it hears its own speaker). A spoken line is 1-2 s,
   so the mute is derived from THE LOADED FILE'S OWN DURATION plus the room's
   ring -- never a constant. A hardcoded 0.35 s against a 1.49 s gunshot sample
   once produced 37 false "heard but not seen" reports in a single string.

2. NEVER IN A STRING. A string is a run of trigger pulls; the mute would blind
   the count exactly while it matters. The rule is `QUIET_GAP_S`: a line may
   start only when no real trigger pull has been seen for that long. His
   cadence in a string is about one pull a second, so three seconds is three
   missed beats -- the string is over, or he is reloading, or reading the
   screen. The gate sees every pull the camera reported AND every pull the
   microphone heard (so a pull the camera misses still opens a string). It
   applies to every line except `GATE_EXEMPT` (hold fire, camera failed,
   camera moved): "stop what you are doing" is worth more than the count.

3. STALE SPEECH IS WORSE THAN SILENCE. Every request carries a deadline
   (`TTL_S`) and, where it has one, a predicate that says whether the thing
   it announces is still true ("calibrating" is only true while calibrating).
   A line whose moment has passed is DROPPED and the drop is logged, never
   played late.

4. A PROBLEM PREEMPTS A ROUTINE LINE. An urgent request discards pending
   routine lines and cuts off a routine line that is mid-sentence.

5. NEVER BLOCK. Nothing here runs on the capture thread. Playing is a
   `Mixer.play` (a lock and a list append; the mixer's own callback thread
   does the work). Decoding an mp3 is the only slow thing, and it happens on
   a worker thread, ahead of need, so the Qt thread only ever polls.

6. WORKS WITH NO VOICE FILES. No `voice/` directory means no sets, `say()`
   returns False, and the app carries on silent. The product is complete
   without it.

7. EVERYTHING SAID IS LOGGED, with the voice set and the mute applied, so a
   later reader can tell a remark from a measurement.

Which voice: each announcement independently picks one of the sets found on
disk, with a dedicated seeded `random.Random` (never the module-level
generator -- `tests/test_artwork.py` section 8 forbids that for modes and the
same reasoning applies: a test must be able to make the choice deterministic).
If the chosen set lacks the file, another set that has it is used, and the
substitution is logged once.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

URGENT = 0          # a problem the shooter must know about NOW
ROUTINE = 1         # a mode name, a round event, good news

# line key -> (priority, time-to-live in seconds, cooldown in seconds)
#
# TTL: how long after being asked a line may still be spoken. Measured against
# the quiet-gap rule: a line asked for mid-string can start 3 s after the last
# pull at the earliest, so a TTL under 3 s would make mid-string lines
# unspeakable, and a long one would let them arrive after the world moved on.
#   state lines (problems, calibration) 5 s -- and most also have a predicate
#   round events                        6 s -- "figure down" is only meaningful
#                                              until the next figure
#   mode names / verdicts               8 s -- predicate-guarded (mode still
#                                              active, verdict still current)
#   welcome                            12 s -- the first file may still be
#                                              decoding at launch
# COOLDOWN: the same line is not repeated inside this window (several call
# sites can announce one state; the mode key can be mashed).
SPEC: dict[str, tuple[int, float, float]] = {
    "v_welcome":          (ROUTINE, 12.0, 30.0),
    "v_calibrating":      (URGENT,   5.0,  8.0),
    "v_hold_fire":        (URGENT,   5.0,  5.0),
    "v_ready":            (ROUTINE,  8.0,  3.0),
    "v_not_ready":        (URGENT,   8.0,  3.0),
    "v_camera_bad":       (URGENT,   5.0, 15.0),
    "v_camera_moved":     (URGENT,   5.0, 10.0),
    # The menu announces itself like any other screen he can land on. His
    # 22nd line, recorded 2026-10-18; the suite pinned the count at 21 in
    # five places and went red the moment the files existed.
    "v_main_menu":        (ROUTINE,  8.0,  2.0),
    "v_mode_static":      (ROUTINE,  8.0,  2.0),
    "v_mode_ball":        (ROUTINE,  8.0,  2.0),
    "v_mode_balls":       (ROUTINE,  8.0,  2.0),
    "v_mode_reaction":    (ROUTINE,  8.0,  2.0),
    "v_mode_clear":       (ROUTINE,  8.0,  2.0),
    "v_mode_popup":       (ROUTINE,  8.0,  2.0),
    "v_mode_image":       (ROUTINE,  8.0,  2.0),
    "v_mode_advancing":   (ROUTINE,  8.0,  2.0),
    "v_mode_animated":    (ROUTINE,  8.0,  2.0),
    "v_new_round":        (ROUTINE,  8.0,  2.0),
    "v_round_done":       (ROUTINE,  6.0,  2.0),
    "v_figure_down":      (ROUTINE,  6.0,  2.0),
    "v_level_up":         (ROUTINE,  6.0,  2.0),
    "v_false_start":      (ROUTINE,  6.0,  2.0),
}

# What each line says -- for the log only. `phrases_to_generate.md` is the
# source of truth for the recordings; a test compares the two.
PHRASES: dict[str, str] = {
    "v_welcome": "Welcome to ScreenPlink.",
    "v_calibrating": "Calibrating. Do not shoot yet.",
    "v_hold_fire": "Hold fire. Measuring.",
    "v_ready": "Ready. Start shooting.",
    "v_not_ready": "Not ready. Check the screen.",
    "v_camera_bad": "Camera failed. Do not shoot.",
    "v_camera_moved": "Camera moved. Recalibrating.",
    "v_main_menu": "The Main Menu.",
    "v_mode_static": "Static targets.",
    "v_mode_ball": "Escalating ball.",
    "v_mode_balls": "Bouncing balls.",
    "v_mode_reaction": "Reaction.",
    "v_mode_clear": "Clear the board.",
    "v_mode_popup": "Pop-up target.",
    "v_mode_image": "Static target.",
    "v_mode_advancing": "Moving target.",
    "v_mode_animated": "Animated target.",
    "v_new_round": "New round.",
    "v_round_done": "Round complete.",
    "v_figure_down": "Figure down.",
    "v_level_up": "Level up.",
    "v_false_start": "False start.",
}

# A line that is pointless when another has just been spoken or is pending:
# {line: (covered_by, window_s)}. "Calibrating" right after "Camera moved.
# Recalibrating." says the same thing twice.
COVERED_BY: dict[str, tuple[str, float]] = {
    "v_calibrating": ("v_camera_moved", 10.0),
}

# Mode class `name` -> the line spoken when the shooter switches to it.
MODE_LINES: dict[str, str] = {
    "menu": "v_main_menu",
    "static": "v_mode_static", "ball": "v_mode_ball", "balls": "v_mode_balls",
    "reaction": "v_mode_reaction",
    "circles": "v_mode_clear", "popup": "v_mode_popup", "image": "v_mode_image",
    "animated": "v_mode_animated",
    "advancing": "v_mode_advancing",
}

# "STOP WHAT YOU ARE DOING" lines. They speak even in the middle of a string.
# Hold fire exists to stop a trigger pull landing inside a noise measurement
# (a shot there becomes the noise floor and lifts the bar above every real hit
# for the session -- three times on this project); deferring it three seconds
# defeats it. Camera failed and camera moved are the same kind of line: the
# shooter is firing into something that is not counting, and each costs nothing
# to say now, because while the camera is down or the mapping is void no
# pull is being counted for the mute to blind. Everything else keeps the gate.
GATE_EXEMPT = frozenset({"v_hold_fire", "v_camera_bad", "v_camera_moved"})

QUIET_GAP_S = 3.0           # no real trigger pull for this long before speaking
ROOM_TAIL_S = 0.25          # the room rings after the file ends (= SOUND_TAIL_S)
MAX_PENDING = 8
AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac")


@dataclass
class _Pending:
    key: str
    voice: str
    sample: str
    priority: int
    asked_at: float
    expires_at: float
    valid: Optional[Callable[[], bool]] = None
    seq: int = 0


@dataclass
class VoiceStats:
    said: int = 0
    dropped: int = 0
    refused: int = 0
    said_lines: list = field(default_factory=list)       # (t, key, voice, dur, mute)


class VoiceAnnouncer:
    def __init__(self, mixer, voice_dir: "Path | None",
                 mute: "Callable[[float], None] | None" = None,
                 log: "Callable[[str], None] | None" = None,
                 clock: Callable[[], float] = time.perf_counter,
                 seed: "int | None" = None,
                 enabled: "Callable[[], bool] | None" = None,
                 synchronous: bool = False,
                 quiet_gap: float = QUIET_GAP_S,
                 tail: float = ROOM_TAIL_S) -> None:
        self.mixer = mixer
        self._mute = mute
        self._log_fn = log
        self._clock = clock
        self._enabled = enabled
        self._sync = synchronous
        self.quiet_gap = float(quiet_gap)
        self.tail = float(tail)
        # A dedicated generator -- never the module-level `random`.
        self._rng = random.Random(seed)

        self.stats = VoiceStats()
        self._lock = threading.RLock()
        self._pending: list[_Pending] = []
        self._seq = 0
        self._last_shot_at: Optional[float] = None
        self._speaking_until = 0.0
        self._speaking: Optional[tuple[str, str, int]] = None   # sample, key, priority
        self._last_said: dict[str, float] = {}
        self._said_log: dict[str, float] = {}
        self._once: set[str] = set()
        self._loaded: set[str] = set()
        self._loading: set[str] = set()
        self._failed: set[str] = set()

        # voice sets found on disk: {set name: {line key: path}}
        self.sets: dict[str, dict[str, Path]] = {}
        self._scan(Path(voice_dir) if voice_dir is not None else None)

    # -- discovery -----------------------------------------------------------

    def _scan(self, root: "Path | None") -> None:
        if root is None or not root.is_dir():
            self._say_once("none", "voice: no voice directory "
                           f"({root}); announcements are off. The app is "
                           "complete without them.")
            return
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            files = {p.stem: p for p in sorted(d.iterdir())
                     if p.is_file() and p.suffix.lower() in AUDIO_EXTS
                     and p.stem in SPEC}
            if files:
                self.sets[d.name] = files
        if not self.sets:
            self._say_once("none", f"voice: {root} holds no recognised voice "
                           "files; announcements are off.")
            return
        self._log(f"voice: sets {', '.join(sorted(self.sets))}  "
                  f"({sum(len(v) for v in self.sets.values())} files); "
                  f"quiet gap {self.quiet_gap:g}s, one random set per line")

    def speaking_until(self) -> float:
        """When the current line finishes, on the same clock as everything
        else here. 0.0 when nothing is being said.

        PUBLIC ON PURPOSE. The startup splash has to know whether the
        welcome line is still running, and this project has twice read a
        plausible-looking private name that did not exist -- `stats` for
        `_stats`, `last_shot_at` through getattr -- and got a guard that
        looked like protection and never fired.
        """
        return self._speaking_until

    @property
    def available(self) -> bool:
        return bool(self.sets)

    # -- logging --------------------------------------------------------------

    def _log(self, text: str) -> None:
        if self._log_fn is not None:
            try:
                self._log_fn(text)
            except Exception:                                # noqa: BLE001
                pass

    def _say_once(self, tag: str, text: str) -> None:
        if tag in self._once:
            return
        self._once.add(tag)
        self._log(text)

    # -- the world tells us ---------------------------------------------------

    def note_shot(self, t: "float | None" = None) -> None:
        """A REAL trigger pull was seen (not one of the app's own dots)."""
        with self._lock:
            t = self._clock() if t is None else float(t)
            # Never move backwards: a heard pull is delivered late and may be
            # older than a pull the camera has already reported.
            if self._last_shot_at is None or t > self._last_shot_at:
                self._last_shot_at = t

    def in_string(self, now: "float | None" = None) -> bool:
        now = self._clock() if now is None else now
        last = self._last_shot_at
        return last is not None and (now - last) < self.quiet_gap

    # -- asking ---------------------------------------------------------------

    def say(self, key: str, valid: "Callable[[], bool] | None" = None,
            ttl: "float | None" = None) -> bool:
        """Ask for a line. Returns whether it was accepted (not whether it
        has been spoken: that happens in `poll`, when it is safe to)."""
        spec = SPEC.get(key)
        if spec is None:
            self._say_once("unknown:" + key, f"voice: unknown line {key!r} ignored")
            return False
        if not self.sets:
            return False
        priority, default_ttl, cooldown = spec
        now = self._clock()
        with self._lock:
            # Already waiting: refresh it rather than queue a second.
            for p in self._pending:
                if p.key == key:
                    p.expires_at = now + (default_ttl if ttl is None else ttl)
                    p.valid = valid or p.valid
                    return True
            last = self._last_said.get(key)
            if last is not None and now - last < cooldown:
                self._drop_note(key, "repeat", f"said {now - last:.1f}s ago, "
                                f"cooldown {cooldown:g}s", quiet=True)
                return False
            cov = COVERED_BY.get(key)
            if cov is not None:
                other, window = cov
                lt = self._last_said.get(other)
                if (any(p.key == other for p in self._pending)
                        or (lt is not None and now - lt < window)):
                    self._drop_note(key, "covered", f"{other} covers it",
                                    quiet=True)
                    return False
            voice = self._pick_voice(key)
            if voice is None:
                self._say_once("missing:" + key,
                               f"voice: no set has {key}; staying silent for it")
                return False
            sample = f"voice/{voice}/{key}"
            self._seq += 1
            self._pending.append(_Pending(
                key, voice, sample, priority, now,
                now + (default_ttl if ttl is None else ttl), valid, self._seq))
            if priority == URGENT:
                # A problem preempts routine speech: waiting routine lines
                # describe a world that is no longer the headline.
                keep = []
                for p in self._pending:
                    if p.priority == ROUTINE:
                        self._drop(p, "preempted by " + key)
                    else:
                        keep.append(p)
                self._pending = keep
            if len(self._pending) > MAX_PENDING:
                old = min(self._pending, key=lambda p: (-p.priority, p.seq))
                self._pending.remove(old)
                self._drop(old, "queue full")
        self._prefetch(sample, self.sets[voice][key])
        return True

    def _pick_voice(self, key: str) -> "str | None":
        names = sorted(self.sets)
        if not names:
            return None
        choice = self._rng.choice(names)
        if key in self.sets[choice]:
            return choice
        have = [n for n in names if key in self.sets[n]]
        if not have:
            return None
        sub = self._rng.choice(have)
        self._say_once(f"fallback:{choice}:{key}",
                       f"voice: set {choice!r} has no {key}; using {sub!r} "
                       f"instead (said once)")
        return sub

    # -- loading (never on the caller's thread unless synchronous) ----------

    def _prefetch(self, sample: str, path: Path) -> None:
        with self._lock:
            if sample in self._loaded or sample in self._loading or sample in self._failed:
                return
            self._loading.add(sample)
        if self._sync:
            self._load(sample, path)
        else:
            threading.Thread(target=self._load, args=(sample, path),
                             name="voice-load", daemon=True).start()

    def _load(self, sample: str, path: Path) -> None:
        # `Mixer.load` writes `error` on failure; that field belongs to the
        # gunshot path's diagnostics, so put it back.
        prev = getattr(self.mixer, "error", "")
        try:
            ok = bool(self.mixer.load(sample, path))
            err = getattr(self.mixer, "error", "")
        except Exception as exc:                             # noqa: BLE001
            ok, err = False, repr(exc)
        try:
            self.mixer.error = prev
        except Exception:                                    # noqa: BLE001
            pass
        with self._lock:
            self._loading.discard(sample)
            if ok:
                self._loaded.add(sample)
            else:
                self._failed.add(sample)
        if not ok:
            self._log(f"voice: could not load {path.name} ({err})")

    # -- speaking --------------------------------------------------------------

    def _drop(self, p: _Pending, why: str) -> None:
        self.stats.dropped += 1
        self._log(f"voice: DROPPED {p.key} (voice={p.voice}) -- {why}; "
                  f"asked {self._clock() - p.asked_at:.1f}s ago")

    def _drop_note(self, key: str, tag: str, why: str, quiet: bool = False) -> None:
        self.stats.dropped += 1
        # Repeats and covered lines are ordinary; say each reason at most
        # once a few seconds so a mashed key cannot flood the log.
        now = self._clock()
        k = f"{key}:{tag}"
        if now - self._said_log.get(k, -99.0) > 5.0:
            self._said_log[k] = now
            self._log(f"voice: skipped {key} -- {tag}: {why}")

    def poll(self, now: "float | None" = None) -> Optional[str]:
        """Speak the next line if it is safe to. Cheap; call ~10 times a second.

        Returns the key it started, or None."""
        now = self._clock() if now is None else now
        with self._lock:
            if not self._pending:
                return None
            # 1. stale ones go, always, whatever else is happening.
            keep = []
            for p in self._pending:
                if now > p.expires_at:
                    self._drop(p, "stale: its moment passed"
                               + (" while a string was in progress"
                                  if self.in_string(now) else ""))
                elif p.valid is not None and not self._still_true(p):
                    self._drop(p, "no longer true")
                else:
                    keep.append(p)
            self._pending = keep
            if not self._pending:
                return None
            # 2. is speech allowed at all right now?
            if self._enabled is not None and not self._enabled():
                for p in self._pending:
                    self._drop(p, "sound is switched off")
                self._pending = []
                return None
            if not getattr(self.mixer, "available", False):
                for p in self._pending:
                    self._drop(p, "no audio output")
                self._pending = []
                return None
            # 3. NEVER IN A STRING -- except the "stop" lines (GATE_EXEMPT).
            pool = self._pending
            if self.in_string(now):
                pool = [p for p in pool if p.key in GATE_EXEMPT]
                if not pool:
                    return None
            best = min(pool, key=lambda p: (p.priority, p.seq))
            # 4. one voice at a time; a problem may cut a routine line off.
            if now < self._speaking_until:
                cut = (best.priority == URGENT and self._speaking is not None
                       and self._speaking[2] == ROUTINE)
                if not cut:
                    return None
                self._cut()
            # 5. the file must be decoded (it is being, on a worker).
            if best.sample in self._failed:
                self._pending.remove(best)
                self._drop(best, "its file could not be loaded")
                return None
            if best.sample not in self._loaded:
                return None
            dur = float(self.mixer.duration(best.sample))
            if dur <= 0.0:
                self._pending.remove(best)
                self._drop(best, "zero-length sample")
                return None
            self._pending.remove(best)
            mute_s = dur + self.tail
            # Mute FIRST: the microphone must never be live while our own
            # speaker is, even for the output latency.
            if self._mute is not None:
                try:
                    self._mute(mute_s)
                except Exception as exc:                     # noqa: BLE001
                    self._log(f"voice: could not mute the microphone ({exc!r}); "
                              f"NOT speaking {best.key}")
                    self.stats.refused += 1
                    return None
            ok = self.mixer.play(best.sample, 1.0)
            if not ok:
                self.stats.refused += 1
                self._log(f"voice: play() refused {best.key}")
                return None
            self._speaking = (best.sample, best.key, best.priority)
            self._speaking_until = now + mute_s
            self._last_said[best.key] = now
            self.stats.said += 1
            self.stats.said_lines.append((now, best.key, best.voice, dur, mute_s))
            self._log(f"voice: SAID {best.key} \"{PHRASES.get(best.key, '')}\" "
                      f"voice={best.voice} t={now:.2f} len={dur:.2f}s "
                      f"mic muted {mute_s:.2f}s waited={now - best.asked_at:.2f}s")
            return best.key

    def _still_true(self, p: _Pending) -> bool:
        try:
            return bool(p.valid())
        except Exception:                                    # noqa: BLE001
            return False

    def _cut(self) -> None:
        sample, key, _pri = self._speaking
        stop = getattr(self.mixer, "stop", None)
        if stop is not None:
            try:
                stop(sample)
            except Exception:                                # noqa: BLE001
                pass
        self._log(f"voice: CUT OFF {key} for a problem line")
        self._speaking = None
        self._speaking_until = 0.0

    def status(self) -> str:
        with self._lock:
            return (f"voice: {self.stats.said} said, {self.stats.dropped} "
                    f"dropped, {len(self._pending)} waiting")


# ---------------------------------------------------------------------------
# Events derived from mode state that already exists. The modes know nothing
# about audio and are not changed: a snapshot before and after a shot says
# what the shot did.

def mode_snapshot(mode) -> dict:
    if mode is None:
        return {}
    level = 0
    balls = getattr(mode, "balls", None)
    if balls:
        level = sum(int(getattr(b, "level", 0)) for b in balls)
    else:
        level = int(getattr(mode, "level", 0) or 0)
    return {
        "dropped": int(getattr(mode, "dropped", 0) or 0),
        "false_starts": int(getattr(mode, "false_starts", 0) or 0),
        "level": level,
        "cleared": getattr(mode, "cleared_at", None) is not None,
    }


def mode_events(before: dict, after: dict) -> list[str]:
    """The lines a shot earned, from two snapshots of the active mode."""
    if not before or not after:
        return []
    out: list[str] = []
    if after["false_starts"] > before["false_starts"]:
        out.append("v_false_start")
    if after["dropped"] > before["dropped"]:
        out.append("v_figure_down")
    if after["level"] > before["level"]:
        out.append("v_level_up")
    if after["cleared"] and not before["cleared"]:
        out.append("v_round_done")
    return out
