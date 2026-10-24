"""A tiny polyphonic sample mixer.

Built rather than borrowed because the thing we need is unusual: a shot sound
must start within a few milliseconds of a hit, and several must be able to
overlap when you shoot fast. The obvious one-liner (sounddevice.play) opens a
fresh stream per call and cancels the previous sound, which gives you both high
latency and a machine gun that clicks once.

So: one persistent output stream with a small block size, and a callback that
sums whatever voices are currently alive. Nothing allocates in the callback.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except Exception:                                     # pragma: no cover
    sd = None                                         # audio becomes a no-op


# The cinematic gun was DROPPED at his request 2026-10-18 after its 5.20s
# length crashed the mixer mid-drill and left the microphone deaf for 4.48s
# -- about four of his trigger pulls uncorroborated after every figure. It
# stays registered and hidden so it can never be drawn as an ordinary bang.
_CINEMATIC_SAMPLE = "reserved/cinematic"

# ...and REPLACED, same day, by a short one he chose himself: "that sound is
# short and should not create issues like the 5 second sound did". He is
# right -- 33KB against 52KB, and the mute is derived from the file's own
# loud part, so a short sample costs a short deafness.
_VANISH_SAMPLE = "reserved/vanish"

# A RING TARGET being cleared: CLEAR THE BOARD (key 5) and nothing else.
# Figures use their own sound below -- he asked for them to differ.
# One sound for one meaning -- asked for separately on 2026-10-18 and
# then extended to the ring targets, so it is ONE constant rather than
# the same filename typed in four modes.
#
# THE CALLER LIST WAS WRONG IN BOTH DIRECTIONS and said "modes 1, 6".
# Mode 6 is Pop-Up, which has always used the figure-drop sound. Mode 1
# used this until 2026-10-23, when he asked for it to go: "it doesn't
# make sense for mode 1 because most of the shots land inside the target
# anyway". The better reason is that NOTHING VANISHES in mode 1 -- its
# marks stand through the round -- so the sound announced an event that
# never happened, on nearly every shot. A comment that names callers
# goes stale silently; `tests/test_volume_fall.py` now pins which mode
# returns this sound and which returns none.
VANISH_SOUND = _VANISH_SAMPLE

# A FIGURE WITH ENOUGH HITS, DROPPING (modes 7, 8, 9). His cinematic gun
# is BACK, because he shortened the file himself: 5.20s -> 0.81s, loud
# until 0.75s, and the second boom at 4.0-4.25s is gone. Measured here
# rather than taken on trust, because its LENGTH is the whole reason it
# was dropped this morning -- at 5.20s the self-mute ran 4.48s after
# every figure, about four of his trigger pulls uncorroborated on the
# only channel that counts them independently of the camera. At 0.81s
# the mute is ~1.0s, the same order as the 9mm gunshot.
#
# The ring targets keep the bubble pop (VANISH_SOUND): a figure going
# down and a ring clearing should not sound alike.
#
# SUPERSEDED 2026-10-20 as the sound a figure makes when it goes down: that is now
# per figure (`audio/fall_sounds.py`: the banshee for zombie02, a random grunt for
# every other figure and for the soldier). `FALL_SOUND` stays defined and its file
# stays registered and hidden, so the cinematic gun can never be drawn as an
# ordinary bang, and nothing that still names it breaks.
FALL_SOUND = _CINEMATIC_SAMPLE

# THE FALL SOUNDS (2026-10-20). Banshee-02 is zombie02's, exactly; the six
# Male-Grunt samples are the random default for every other figure. Banshee-01 is
# registered hidden too: unregistered, the picker would offer it as a gunshot.
BANSHEE_SOUND = "reserved/banshee-02"
# Banshee-01 is the alien's (he named it the same day: "lets use Banshee-01.mp3 for
# the Alien target sound").
BANSHEE_01_SOUND = "reserved/banshee-01"
GRUNT_SOUNDS = tuple(f"reserved/grunt-{i}" for i in range(1, 7))

# A BALL BURSTING, modes 2 and 3. Glass breaking, which he picked after
# hearing the first one: "this is the better effect for the balls".
# 1.54s, so the self-mute costs about what the 9mm gunshot does (1.49s)
# and the microphone is not deafened the way the 5.20s sample did it.
POP_SOUND = "reserved/break-ball"

# A TARGET APPEARING WITHOUT WARNING (mode 4, the reaction drill). Not a
# shot sound: it plays when the target is REVEALED, which is the moment
# the drill is timing from. His choice, 2026-10-18.
APPEAR_SOUND = "reserved/appear"

# A HUMAN FIGURE popping up without warning (mode 7). A different sound
# from the ring target appearing, so from the firing line he can tell
# what arrived before he has finished turning his head. 0.83s.
FIGURE_APPEAR_SOUND = "reserved/figure-appear"

# A TRIGGER PULL WITH AN EMPTY MAGAZINE (2026-10-20). His words: "use
# 'universfield-bubble-pop-06-351337.mp3' for the sound when the bullets run out
# and still paint the holes; that way player will understand that something is
# going on because this is a boopping sound instead of 'bullet' sound." It is
# played INSTEAD of the gunshot: the gunshot says a live round went off, and a
# dry pull is not one. 0.099 s long (measured), so its self-mute is the 0.35 s
# floor, shorter than the gunshot's 0.82 s. NOTE it is NOT `VANISH_SOUND`: that is
# `universfield-bubble-pop-07-487896` (1.06 s), the ring target being destroyed.
# Two files from one artist with similar names, and a hit and a dry pull sound
# alike in character (same family of pop) but differ 10x in length.
DRY_SOUND = "reserved/dry-pop"

# A SELECTION CUE (2026-10-20): "use 'freesound_community-080997_bullet-39735.mp3'
# when the player hits 'menu' button or 'next round' button; the same thing for
# hitting anything in the 'menu' mode." Played IN ADDITION to the gunshot (one
# millisecond after it, like every other effect), never instead of it; a click,
# which has no gunshot, plays it alone. 0.405 s long (measured).
SELECT_SOUND = "reserved/select"


@dataclass(eq=False)
class _Voice:
    buffer: np.ndarray
    position: int
    gain: float


class Mixer:
    # Deliberately quiet. A gunshot sample at unity gain through desktop speakers
    # is startling, and this is a tool someone runs for the first time without
    # knowing how loud it will be. Easy to turn up, unpleasant to be surprised by.
    DEFAULT_GAIN = 0.05

    def __init__(self, samplerate: int = 48000, blocksize: int = 256, max_voices: int = 16,
                 master_gain: float | None = None,
                 device_name: str = "") -> None:
        self.samplerate = samplerate
        # Remembered BY NAME, as the camera and the microphone are, because
        # sounddevice indices shift whenever anything is plugged in.
        self.device_name = device_name
        self.device_index: int | None = None
        self.device_label = ""
        self.blocksize = blocksize
        self.max_voices = max_voices
        self.master_gain = self.DEFAULT_GAIN if master_gain is None else float(master_gain)
        self._samples: dict[str, np.ndarray] = {}
        self._voices: list[_Voice] = []
        self._lock = threading.Lock()
        self._stream = None
        self.available = False
        self.error = ""
        # Incremented every time an output stream is opened. The microphone's
        # idea of how loud this mixer is at the room is only valid for the
        # stream it was measured on: a reopened stream is a new endpoint,
        # possibly a different one, and `ensure_alive` reopens silently.
        self.epoch = 0

    # Words that mark an output as a display reached over HDMI or
    # DisplayPort. Kept for the log and for the health check, NOT for
    # choosing: see `_pick_output`.
    DISPLAY_HINTS = ("nvidia", "hdmi", "displayport", "amd high definition")

    def is_display_output(self) -> bool:
        """Is the open stream playing into a monitor or TV?

        Worth knowing because such an endpoint disappears when the display
        sleeps or renegotiates its link -- and this app flashes the monitor
        white and black throughout calibration, so it provokes exactly that.
        Not a reason to avoid the device: on this bench the monitor IS the
        only audible speaker. It is a reason to check the stream is alive.
        """
        low = (self.device_label or "").lower()
        return any(h in low for h in self.DISPLAY_HINTS)

    def _pick_output(self):
        """(index, label) for the output to use, or (None, label) for the default.

        By remembered name if there is one. Otherwise None, which means the
        system default -- and that is deliberate.

        A first version scored real speakers ABOVE any display endpoint, on
        the reasoning that a monitor's audio vanishes when the screen sleeps.
        Measured on this bench, that rule was simply wrong: the Realtek
        output it preferred has nothing audible attached, and the monitor is
        where the sound actually comes out. The shooter heard four gunshots
        from the display and none at all from the 'real' speakers.

        Windows already knows which endpoint has something connected to it,
        and that is what the default means. Guessing against it from a
        device's name is reasoning about hardware from a string.
        """
        if sd is None or not self.device_name:
            return None, ""
        try:
            apis = sd.query_hostapis()
            devices = list(enumerate(sd.query_devices()))
        except Exception:
            return None, ""
        want = self.device_name.lower()
        for i, d in devices:
            if d.get("max_output_channels", 0) <= 0:
                continue
            name = str(d.get("name", ""))
            if name.lower() == want or want in name.lower():
                return i, f"{name}  [{apis[d['hostapi']]['name']}]"
        return None, ""

    def set_output_device(self, name: str) -> bool:
        """Switch output. Restarts the stream, so it can be done live."""
        self.device_name = name or ""
        was = self._stream is not None
        self.close()
        return self.start() if was else True

    def set_master_gain(self, gain: float) -> None:
        self.master_gain = max(0.0, min(1.0, float(gain)))

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> bool:
        if sd is None:
            self.error = "sounddevice unavailable"
            return False
        if self._stream is not None:
            return True
        try:
            idx, label = self._pick_output()
            if idx is None and not label:
                # Using the system default. Resolve its NAME anyway, or
                # `is_display_output` reads False for a monitor and the
                # health check never knows it is watching a fragile
                # endpoint -- which is the whole case it exists for.
                try:
                    d = sd.query_devices(sd.default.device[1])
                    label = f"{d['name']}  [system default]"
                except Exception:
                    label = "system default"
            self.device_index, self.device_label = idx, label
            self._stream = sd.OutputStream(
                samplerate=self.samplerate, channels=1, dtype="float32",
                blocksize=self.blocksize, latency=0.02, callback=self._callback,
                device=idx,
            )
            self._stream.start()
            self.available = True
            self.epoch += 1
            return True
        except Exception as exc:
            self.error = repr(exc)
            self._stream = None
            return False

    def ensure_alive(self) -> str:
        """Reopen the output if it has stopped. Returns a line to log, or "".

        The stream is opened once at startup and then played into for the
        rest of the session, and nothing ever checked it was still there.
        An endpoint can go away underneath it -- a display sleeping or
        renegotiating its link is the case here, and this app causes that
        every time it flashes the screen for calibration. PortAudio does not
        raise for a callback stream when that happens: `play()` keeps
        succeeding, `available` stays True, and the gunshots go nowhere.
        Silently, which is the worst way for a feedback channel to fail,
        because the shooter reads it as the app not detecting the shot.
        """
        if sd is None or self._stream is None:
            return ""
        try:
            if self._stream.active:
                return ""
        except Exception:
            pass                      # a closed stream raises; treat as dead
        was = self.device_label or "the system default"
        self.close()
        if self.start():
            return (f"sound output had stopped ({was}); reopened on "
                    f"{self.output_name}")
        return f"sound output stopped ({was}) and will not reopen: {self.error}"

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.available = False

    @property
    def audible(self) -> bool:
        """Is there a live stream a bang written now would actually reach?

        `available` only says a stream was opened once. PortAudio does not
        raise for a callback stream whose endpoint has gone, so `play()`
        keeps returning True into silence until `ensure_alive` (every 5s)
        notices. Anything that REGISTERS what it believes it played has to
        ask this as well, or a bang that reached no speaker is filed as a
        measurement of the room.
        """
        if not self.available or self._stream is None:
            return False
        try:
            return bool(self._stream.active)
        except Exception:
            return False

    @property
    def output_name(self) -> str:
        """What the sound is actually coming out of, for the log.

        'audio ready, output latency ~21.3 ms' was the whole report, and it
        is true of a stream playing into a monitor nobody can hear. A device
        chosen on the app's behalf has to be named, or a silent wrong choice
        looks exactly like success -- the same lesson as the camera.
        """
        if self.device_label:
            return self.device_label
        if sd is None:
            return "unavailable"
        try:
            d = sd.query_devices(sd.default.device[1])
            return f"{d['name']}  [system default]"
        except Exception:
            return "system default"

    @property
    def latency_ms(self) -> float:
        if self._stream is None:
            return 0.0
        try:
            return float(self._stream.latency) * 1000.0
        except Exception:
            return self.blocksize / self.samplerate * 1000.0

    # -- samples --------------------------------------------------------------

    def register(self, name: str, mono: np.ndarray) -> None:
        self._samples[name] = np.asarray(mono, dtype=np.float32).reshape(-1)

    def load(self, name: str, path: Path) -> bool:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            if sr != self.samplerate:
                mono = _resample_linear(mono, sr, self.samplerate)
            self.register(name, mono)
            return True
        except Exception as exc:
            self.error = f"load {path.name}: {exc!r}"
            return False

    def load_folder(self, folder: Path) -> list[str]:
        """Load every audio file in a folder, named by a tidied-up filename.

        Downloaded samples arrive with names like
        'freesound_community-9mm-pistol-shoot-short-reverb-7152.mp3'; the user
        has to pick from these in a combo box, so trim the source prefix and the
        trailing id and leave something readable.
        """
        loaded: list[str] = []
        if not folder.exists():
            return loaded
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (
                    ".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aif"):
                continue
            stem = path.stem
            for prefix in ("freesound_community-", "freesound-", "zapsplat-"):
                if stem.lower().startswith(prefix):
                    stem = stem[len(prefix):]
            parts = stem.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                stem = parts[0]
            reserved = self.RESERVED_FILES.get(path.stem)
            if reserved is not None:
                self.load(reserved, path)       # registered, never listed
                continue
            name = stem.replace("-", " ").replace("_", " ").strip() or path.stem
            if self.load(name, path):
                loaded.append(name)
        return loaded

    def duration(self, name: str) -> float:
        """How long a loaded sample runs, in seconds.

        The caller that needs this is the self-mute: the microphone is in
        the same room as the speakers, so the app has to stop listening for
        exactly as long as it is making a noise. That was a hardcoded 0.35s
        chosen when the only sample was a short synthetic click, and the
        samples in use are longer -- measured 2026-10-15, 87% of the
        'heard a shot with no detection' reports landed within three seconds
        of a detection, which is the app listening to its own gunshot.
        """
        buf = self._samples.get(name)
        if buf is None or not len(buf):
            return 0.0
        return float(len(buf)) / float(self.samplerate)

    # The master gain at which `frac = 0.10` was judged safe (the default,
    # his own setting). The microphone's threshold is an ABSOLUTE level, so a
    # louder app rings above it for longer: the sample is only "quiet enough"
    # once gain * level falls below what it was at this gain with `frac` of the
    # peak left. See `mute_fraction`.
    MUTE_REF_GAIN = DEFAULT_GAIN
    MUTE_MIN_FRAC = 0.005          # 46 dB under the peak: past this, the whole file

    @classmethod
    def mute_fraction(cls, gain: "float | None", frac: float = 0.10) -> float:
        """The fraction of the sample's peak below which the app cannot hear itself.

        At the reference gain it is `frac`. Louder, it shrinks in proportion
        (twice the gain, half the fraction: the same absolute level reaches the
        microphone at half the point on the envelope), so the mute grows with
        the volume. QUIETER than the reference it is NOT relaxed: the 10% figure
        is the one measured to work, a shorter mute at low volume is a saving
        nobody has verified, and the volume keys must not be able to open a gap.
        Floored, because a mute that reached the end of every file is the old
        behaviour with extra steps. None means "unscaled" (the old behaviour).
        """
        if gain is None:
            return float(frac)
        g = max(1e-6, float(gain))
        scaled = float(frac) * cls.MUTE_REF_GAIN / g
        return max(cls.MUTE_MIN_FRAC, min(float(frac), scaled))

    def loud_duration(self, name: str, frac: float = 0.10,
                      gain: "float | None" = None) -> float:
        """How long the sample stays ABOVE `frac` of its own peak.

        With `gain` (the master gain the sample is about to play at) the
        fraction is scaled by `mute_fraction`, so a louder gunshot is muted for
        longer. Without it, the unscaled figure, as before.

        The self-mute is the microphone's blind spot, and it was the whole
        sample: 1.49s for the 9mm sample he uses, plus a 0.25s tail, so the
        app stopped listening for about 1.75 seconds after every shot.

        He shoots about one a second. Measured in his own log, 2026-10-16:
        seventeen unmistakably real shots -- peaks 31 to 134, scattered
        across the screen -- were each reported "SAW a shot with no gunshot
        sound: likely a phantom", because the microphone was muted when the
        trigger was pulled. The channel that exists to count his trigger
        pulls INDEPENDENTLY of the camera cannot count them at his cadence,
        and the agreed gun test rests on exactly that count.

        Most of a gunshot sample is reverb tail. Measured on his:

            above 50% of peak until 0.26s
            above 20%           until 0.48s
            above 10%           until 0.56s
            above  2%           until 1.12s
            total                     1.49s

        So the app is only LOUD for about a third of it, and the mute needs
        to cover the loud part plus the room's own ring -- not the whole
        file. 10% is the point where the sample is well under the level the
        microphone's own threshold is set from (40% of his median heard
        shot), with margin.
        """
        buf = self._samples.get(name)
        if buf is None or not len(buf):
            return 0.0
        frac = self.mute_fraction(gain, frac)
        try:
            import numpy as _np
            x = _np.abs(_np.asarray(buf, dtype=_np.float32))
            if x.ndim > 1:
                x = x.max(axis=1)
            peak = float(x.max())
            if peak <= 0.0:
                return 0.0
            above = _np.nonzero(x > peak * float(frac))[0]
            if not len(above):
                return 0.0
            return float(above[-1] + 1) / float(self.samplerate)
        except Exception:                                    # noqa: BLE001
            # Never worse than the old behaviour.
            return self.duration(name)

    def own_envelope(self, name: str, step_s: float = 0.005) -> tuple[float, np.ndarray]:
        """(step, block-peak envelope) of a sample, for the listener to discount.

        The listener used to go deaf for `loud_duration + tail` = 0.82s after
        every bang the app played, and he shoots about one a second: about 18
        of roughly 40 real shots in his 2026-10-16 14:14 string arrived while
        it was deaf and were reported as phantoms. The app knows exactly what
        it played and when, so it can say how LOUD it is at each instant
        instead of ignoring everything -- see `ClickListener.expect_own_sound`.

        Peak per `step_s` block, at unity gain; the caller scales by the master
        gain, which can change live.
        """
        buf = self._samples.get(name)
        if buf is None or not len(buf):
            return step_s, np.zeros(0, dtype=np.float32)
        step = max(1, int(round(step_s * self.samplerate)))
        x = np.abs(np.asarray(buf, dtype=np.float32).reshape(-1))
        n = -(-len(x) // step)
        x = np.pad(x, (0, n * step - len(x)))
        return step / float(self.samplerate), x.reshape(n, step).max(axis=1)

    # Spoken announcements are registered here too (so they share the one
    # stream, the one lock and the one set of durations), under this prefix.
    # They are NOT gunshots: the sound picker lists `names()` and must not
    # offer "voice/mobster/v_ready" as a gunshot.
    VOICE_PREFIX = "voice/"
    # Samples with a job of their own, never offered as THE gunshot. Keyed by
    # file stem; registered under `reserved/<role>` so `names()` (the picker)
    # cannot list them, exactly as it cannot list a voice line.
    RESERVED_PREFIX = "reserved/"
    # Registered under a reserved name and NEVER offered by the gunshot
    # picker: a 5-second cinematic boom drawn at random as an ordinary shot
    # would deafen the only channel that counts his trigger pulls.
    RESERVED_FILES = {
        "lordsonny-cinematic-gun-163702": _CINEMATIC_SAMPLE,
        "universfield-bubble-pop-07-487896": _VANISH_SAMPLE,
        "11325622-glass-breaking-sound-effect-240679": POP_SOUND,
        # Kept reserved though nothing plays it: unregistered it would
        # be offered as an ordinary gunshot by the picker.
        "creatorshome-sharp-pop-328170": "reserved/pop-unused",
        "dragon-studio-notification-sound-effect-372475": APPEAR_SOUND,
        "youssefmizani-a-shrill-thin-horn-sound-510622": FIGURE_APPEAR_SOUND,
        # Both new on 2026-10-20 and both reserved at once: unregistered, the
        # picker offered them as ordinary gunshots (a 0.099 s pop and a 0.4 s
        # click drawn at random as THE bang), which is how a 5-second boom was
        # once drawn as one.
        "Banshee-02": BANSHEE_SOUND,
        "Banshee-01": BANSHEE_01_SOUND,
        **{f"Male-Grunt-0{i}": f"reserved/grunt-{i}" for i in range(1, 7)},
        "universfield-bubble-pop-06-351337": DRY_SOUND,
        "freesound_community-080997_bullet-39735": SELECT_SOUND,
    }

    def names(self) -> list[str]:
        # A name starting with "_" is the app's own (the microphone probe): it
        # is not a gunshot the shooter can choose.
        return sorted(n for n in self._samples
                      if not n.startswith((self.VOICE_PREFIX, self.RESERVED_PREFIX, "_")))

    def has(self, name: str) -> bool:
        return name in self._samples

    def stop(self, name: str) -> int:
        """Cut off every playing voice of this sample. Returns how many."""
        buf = self._samples.get(name)
        if buf is None:
            return 0
        with self._lock:
            keep = [v for v in self._voices if v.buffer is not buf]
            n = len(self._voices) - len(keep)
            self._voices[:] = keep
        return n

    # -- playback -------------------------------------------------------------

    def play(self, name: str, gain: float = 1.0) -> bool:
        buf = self._samples.get(name)
        if buf is None or not self.available:
            return False
        with self._lock:
            if len(self._voices) >= self.max_voices:
                self._voices.pop(0)
            self._voices.append(_Voice(buf, 0, float(gain)))
        return True

    def _callback(self, outdata, frames, _time, _status) -> None:
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            # Keep the survivors rather than REMOVE the finished. list.remove
            # finds its target with ==, and a dataclass __eq__ compares the
            # numpy buffers elementwise: with his 5.20s fall sound (249,522
            # samples) and a 1.49s gunshot (71,471) playing together, that
            # raises ValueError INSIDE the realtime callback. Seen on his
            # screen 2026-10-18 at 29/30 of a figure drill, as a modal
            # Python-CFFI dialog on the monitor the camera watches.
            keep = []
            for v in self._voices:
                remaining = len(v.buffer) - v.position
                n = min(frames, remaining)
                if n > 0:
                    out[:n] += v.buffer[v.position:v.position + n] * v.gain
                    v.position += n
                if v.position < len(v.buffer):
                    keep.append(v)
            self._voices[:] = keep
        out *= self.master_gain
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:, 0] = out


def _resample_linear(mono: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    n_dst = int(round(len(mono) * dst_sr / src_sr))
    if n_dst <= 1:
        return mono
    x_src = np.linspace(0.0, 1.0, len(mono), endpoint=True)
    x_dst = np.linspace(0.0, 1.0, n_dst, endpoint=True)
    return np.interp(x_dst, x_src, mono).astype(np.float32)


# --- procedural placeholder sounds -------------------------------------------
#
# Real recorded samples come later. These exist so the tool is audibly useful on
# day one: when you are standing three metres away looking at the screen, a
# click that fires the instant a hit registers tells you more about whether
# detection is working than any number on a panel you cannot read from there.

def synth_shot(samplerate: int = 48000, seed: int = 7) -> np.ndarray:
    """A dry, short report: noise crack over a low-frequency thump."""
    rng = np.random.default_rng(seed)
    dur = 0.18
    n = int(samplerate * dur)
    t = np.arange(n) / samplerate

    crack = rng.standard_normal(n).astype(np.float32)
    crack *= np.exp(-t * 90.0)                                   # very fast decay

    thump = np.sin(2 * np.pi * 70.0 * t).astype(np.float32)
    thump *= np.exp(-t * 28.0)

    body = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    body *= np.exp(-t * 55.0)

    out = 0.55 * crack + 0.6 * thump + 0.18 * body
    attack = np.minimum(1.0, t / 0.0008)                         # de-click the onset
    out *= attack
    out /= max(1e-6, float(np.abs(out).max()))
    return (out * 0.85).astype(np.float32)


def synth_probe(samplerate: int = 48000, dur: float = 0.12, f0: float = 300.0,
                f1: float = 6000.0, level: float = 0.5,
                fade: float = 0.012) -> np.ndarray:
    """A short quiet sweep, for measuring how loud the speakers are at the mic.

    A sweep and not a tone, and that is a measurement. Each candidate was
    pushed through 120 random speaker-and-room responses (2nd-order high-pass
    80-1200 Hz, low-pass 4-12 kHz, half with a resonance, one to three
    echoes) and read the way the listener reads it, as microphone level per
    unit of output level, then divided by what the real gunshot sample gives.
    The WORST room for each:

        single 1 kHz tone      0.49x      three tones         0.61x
        noise burst            0.56x      three noise bursts  0.62x
        this sweep             0.93x      (median 1.55x, 95th percentile 2.33x)

    A probe that reads LOW teaches too small a coupling, and the app then hears
    its own bang as trigger pulls; one that reads high only makes the app a
    little deaf for the first few shots, until real bangs replace it. The
    sweep errs the safe way in every one of the 120 rooms because it parks its
    energy wherever the room responds best. tests/test_audio_own.py section 5b
    repeats the measurement over 40 rooms and fails if the probe is changed to
    something that reads low.

    Nothing about it can be mistaken for a shot: it is a rising whoop with
    soft edges, not a transient.
    """
    n = int(dur * samplerate)
    t = np.arange(n) / samplerate
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    env = np.clip(np.minimum(t / fade, (dur - t) / fade), 0.0, 1.0)
    return (level * np.sin(phase) * env).astype(np.float32)


def synth_tick(samplerate: int = 48000, freq: float = 1800.0) -> np.ndarray:
    """A clean high blip: used for 'detector saw something' feedback."""
    dur = 0.04
    n = int(samplerate * dur)
    t = np.arange(n) / samplerate
    out = np.sin(2 * np.pi * freq * t).astype(np.float32) * np.exp(-t * 120.0)
    out *= np.minimum(1.0, t / 0.001)
    return (out * 0.5).astype(np.float32)


def default_mixer(samplerate: int = 48000, sounds_dir: Path | None = None,
                  master_gain: float | None = None,
                  device_name: str = "") -> Mixer:
    """Procedural fallbacks plus whatever real samples are on disk.

    The synthesised ones stay registered regardless, so audio feedback still
    works on a machine with no sound files and the app never falls silent
    because a folder is missing.
    """
    m = Mixer(samplerate=samplerate, master_gain=master_gain,
              device_name=device_name)
    m.register("synth shot", synth_shot(samplerate))
    m.register("synth tick", synth_tick(samplerate))
    if sounds_dir is not None:
        m.load_folder(sounds_dir)
    m.start()
    return m
