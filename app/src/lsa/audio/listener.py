"""Listen to the camera's microphone for the gun going off.

Why this exists
---------------
The camera answers "where did the laser land". It cannot answer "did a shot
happen at all", and that is the question behind every frustrating session: a
shot that does not appear on screen is either a shot the camera could not see,
or a shot that missed the screen entirely, or no shot at all. Those three need
completely different responses and they are indistinguishable from the firing
line.

A dry-fire trigger break is a loud, extremely sharp transient, and the camera
has a microphone on it. So the app gets an INDEPENDENT channel that says a shot
happened, and can then say which of the three cases it is:

    heard and seen      - working
    heard, not seen     - the shot missed the screen, or the camera cannot see
                          that part of it. Either way, not a detector problem.
    seen, not heard     - suspicious. A visual detection with no bang is the
                          signature of a phantom hit, which this project has
                          had before (the app's own warning banner once fired
                          the gunshot sound by itself).

Measured on this hardware (Razer Kiyo Pro microphone, WASAPI, 2026-10-14):
room noise over 3 s sat at -46 dBFS median, -35 dBFS at the 99th percentile and
-31 dBFS at its worst. A trigger click arrives far above that, so a threshold
set from the measured floor has a wide margin to work in.

One hard rule
-------------
**Audio never creates a shot.** It has no position, so a hit it invented would
have to be drawn somewhere, and somewhere is the one thing it cannot know. It
corroborates and it explains; it does not score. Everything in this module
feeds diagnostics and on-screen messages only.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    import sounddevice as sd
except Exception:                                 # pragma: no cover - optional dep
    sd = None


# What happened to a bang the app played, as `expect_own_sound` reports it.
# Four outcomes and no boolean, because the boolean it used to return was
# True for "discounted by measurement" and False for "muted instead" -- and
# nothing at all for the third case, a bang that was neither (RULES C1).
OWN_DISCOUNTED = "discounted"     # its expected level raised the bar, no deafness
OWN_MUTED = "muted"               # coupling unknown: the flat mute covered it
OWN_UNPROTECTED = "unprotected"   # neither -- the app WILL hear itself
OWN_NOT_PLAYED = "not-played"     # it never reached a speaker; nothing filed
OWN_BELOW_BAR = "below-bar"       # coupling unknown, but the bang cannot reach
                                  # the click bar even at the ceiling: no mute


@dataclass
class _Own:
    """One bang the app itself played, as the microphone should hear it."""

    t0: float             # clock() when play() was called
    env: np.ndarray       # digital output level per `dt`, dilated for latency
    dt: float
    peak: float           # env.max(): the loudest this bang gets
    end: float            # seconds after t0 when it is over
    head_end: float       # ...and when it drops under HEAD_FRAC of its peak
    # The loudest few microphone blocks in the head, descending. NOT just the
    # loudest: a real trigger click landing inside the head is one or two
    # blocks and would otherwise be read as the bang (see COUPLING_TOPK).
    top: list = field(default_factory=list)
    overlapped: bool = False   # another bang was playing too: no clean reading
    source: str = "bang"       # "bang" (the gunshot) or "probe" (the whoop)
    mode: str = ""             # one of the OWN_* outcomes
    gain: float = 0.0          # master volume it was played at
    sample: str = ""
    fallback_s: float = 0.0    # the flat mute this bang would get if it must
    # Blocks in the loud head that WOULD have been taken for a trigger click
    # (`hot`), out of the blocks seen there (`blocks`). A click is one block
    # and its ring; a bang louder than predicted is most of the head.
    hot: int = 0
    blocks: int = 0
    tainted: bool = False      # read louder than predicted: no clean reading


@dataclass
class _Reading:
    """One clean measurement of microphone level per unit of output level."""

    t: float
    value: float
    source: str
    gain: float
    sample: str


@dataclass
class Click:
    """One audible transient, loud enough and sharp enough to be a trigger."""

    # clock() of the listener that heard it. In the app that IS
    # time.perf_counter, because the corroborator compares it against the
    # camera's shot timestamps, which are perf_counter: two clocks with
    # different epochs is RULES B4's exact failure, and on Windows
    # perf_counter and monotonic do not share one. The clock is injectable
    # for tests only; `ClickListener.clock_is_perf_counter` says which it is.
    t: float
    level: float          # peak sample magnitude, 0..1
    floor: float          # the noise floor it beat
    # A visual detection has been CLAIMED against this click by
    # `ClickListener.claim_click_near`, so it cannot corroborate a second one.
    # Unused anywhere else in src/ (checked 2026-10-23); the corroborator keeps
    # its own pairing state and never reads it.
    matched: bool = False

    @property
    def margin(self) -> float:
        return self.level / max(self.floor, 1e-5)


def input_devices() -> list[tuple[int, str, str]]:
    """Every input device, as (index, name, host api). Empty if no sounddevice."""
    if sd is None:
        return []
    try:
        apis = sd.query_hostapis()
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append((i, str(d.get("name", "?")),
                            str(apis[d["hostapi"]]["name"])))
        return out
    except Exception:
        return []


# Words that identify nothing. The camera on this bench enumerates as "USB
# Video Device" while its microphone is "Microphone (Razer Kiyo Pro)", so
# matching the camera name against microphone names is worse than useless --
# "usb" and "device" alone will happily match a completely different device.
GENERIC = {"usb", "video", "audio", "device", "microphone", "mic", "input",
           "camera", "webcam", "capture", "the", "and", "pro", "hd", "cam"}

# Things that are a microphone but not the right microphone: a headset moves
# with the shooter, and the rest are software mixers that may hear the app's own
# output rather than the room.
UNWANTED = ("savi", "plantronics", "headset", "sound mapper", "primary sound",
            "stereo mix", "line in", "wave", "what you hear", "virtual")


def pick_device(preferred_name: str = "", camera_name: str = "") -> tuple[int, str] | None:
    """Choose an input device, by remembered name, then by camera, then by guess.

    Windows lists the same microphone three or four times, once per host API.
    WASAPI is the low-latency path and reports a sane sample rate, so it wins.

    This is a heuristic and says so in the log, because the one thing it must
    not do is listen to something silently wrong: a microphone on the shooter's
    headset would still hear the shot, but one that hears the app's own audio
    output would manufacture a click for every hit.
    """
    devices = input_devices()
    if not devices:
        return None

    def api_bonus(api: str) -> int:
        return 5 if api == "Windows WASAPI" else 0

    def distinctive(text: str) -> list[str]:
        cleaned = text.lower().replace("(", " ").replace(")", " ")
        return [w for w in cleaned.split() if len(w) > 2 and w not in GENERIC]

    # 1. The remembered choice, matched by NAME rather than index: USB
    #    enumeration order moves when anything is plugged in, which is the same
    #    reason the camera itself is remembered by name.
    if preferred_name:
        want = preferred_name.lower()
        best = None
        for idx, name, api in devices:
            if name.lower() == want or want in name.lower():
                score = 100 + api_bonus(api)
                if best is None or score > best[0]:
                    best = (score, idx, name)
        if best is not None:
            return best[1], best[2]

    # 2. Something distinctive shared with the camera's name.
    words = distinctive(camera_name)
    if words:
        best = None
        for idx, name, api in devices:
            low = name.lower()
            hits = sum(1 for w in words if w in low)
            if hits:
                score = hits * 10 + api_bonus(api)
                if best is None or score > best[0]:
                    best = (score, idx, name)
        if best is not None:
            return best[1], best[2]

    # 3. A guess: any real microphone that is not a headset or a software mixer.
    best = None
    for idx, name, api in devices:
        low = name.lower()
        if any(bad in low for bad in UNWANTED):
            continue
        score = api_bonus(api) + (3 if "microphone" in low else 0)
        if best is None or score > best[0]:
            best = (score, idx, name)
    return None if best is None else (best[1], best[2])


def best_device_for(camera_name: str) -> int | None:
    """Back-compatible helper: just the index, or None."""
    found = pick_device(camera_name=camera_name)
    return None if found is None else found[0]


class ClickListener:
    """Watches one input device for sharp transients, on its own thread.

    Detection is deliberately crude and deliberately adaptive: block peak
    against a rolling percentile of recent block peaks. A trigger break is 20 dB
    or more above room noise, so there is no need for anything cleverer, and
    anything cleverer would have parameters nobody can tune from the firing
    line. The adaptive floor is what makes it survive a noisy room, a fan, or a
    different microphone without being re-tuned.

    Two refinements that are not optional:

    - **A refractory period.** One click rings for tens of milliseconds and
      would otherwise be counted several times over.
    - **Not hearing the app's own sounds.** The app plays a bang through the
      speakers when it detects a hit, and the microphone is in the same room.
      Without this the app hears itself and reports twice as many shots as
      were fired. That is `expect_own_sound()`, called by `lsa.audio.bang` --
      the only path from the app to the speakers (speech excepted, below). It
      discounts the bang by a MEASURED coupling, mutes flatly while the
      coupling is not known, and does neither when the loudest the bang could
      possibly be at the microphone is under the click bar. `mute_for()` is a
      plain "ignore audio for a moment" and its one caller in `src/` is the
      announcer's speech (`audio/voice.py`, via the app): speech runs for
      seconds and is not registered as a bang.
    """

    BLOCK = 256               # 5.3 ms at 48 kHz: fine enough for a transient
    FLOOR_WINDOW = 400        # blocks of history for the noise floor (~2 s)

    LEARN_NEEDED = 3          # shots that must be seen before audio may speak
    # Of the median measured click. Raised from 0.40 after typing at the
    # keyboard was reported as gunfire: a keystroke is the same shape of
    # transient as a trigger break and only differs in level, so the bar has to
    # sit much closer to a real shot than to the room.
    LEARN_FRACTION = 0.60
    LEARN_FLOOR = 0.08        # and never below this, whatever was measured

    # DISCOUNTING THE APP'S OWN BANG, rather than going deaf for it.
    #
    # The mute was a flat 0.82s after every shot (0.57s loud + 0.25s tail)
    # and he shoots about one a second, so in his 2026-10-16 14:14 string
    # about 18 of roughly 40 real shots arrived while the microphone was deaf
    # and were logged "SAW a shot with no gunshot sound: likely a phantom".
    # The pattern was exact: the first shot after a pause was heard, every
    # shot within 0.8s of the previous one was not. The channel the gun test
    # counts on was blind in precisely the strings that matter. Measured again
    # 2026-10-23 over 8,415 logged shots: 55-65% of every shot he fires is
    # never heard, and that deafness is the ceiling on everything downstream.
    #
    # The app knows what it played and when, so it raises the bar by the
    # bang's own expected level at each instant instead. That level is the
    # bang's digital amplitude times how strongly THIS room couples speaker
    # to microphone, and the coupling is MEASURED, never assumed (the house
    # rule, learned in this very channel when a guessed threshold reported
    # six shots that never happened): it is the loudest the microphone got
    # while each bang played, over the loudest the bang was. Until
    # COUPLING_NEEDED bangs have been measured the old flat mute applies, so
    # the first shots of a session are as safe as they ever were.
    #
    # Measured on his 9mm sample (5 ms blocks, against its own peak): above
    # 50% until 0.27s, 20% until 0.48s, 10% until 0.57s, 5% until 0.71s. A
    # bar of coupling x envelope x OWN_SAFETY therefore falls under the
    # click threshold well inside the old mute whenever the playback volume
    # is low, and is exactly as long as it has to be when it is not.
    COUPLING_NEEDED = 3
    OWN_SAFETY = 1.5          # the bang's level varies block to block
    # The coupling is read from the bang's LOUD HEAD only (its first stretch
    # above this fraction of its peak, 0.48s on his sample). Reading the whole
    # 1.49s would let his next trigger click, about a second later, inflate
    # it; and a bang is only 'overlapped' while another starts inside its head.
    HEAD_FRAC = 0.20
    # The sound reaches the microphone some time after play(): output buffer
    # (~21 ms measured), the input block (5 ms), the air, and the GUI thread
    # calling play() a little before or after the audio callback. Covered by
    # taking the envelope's maximum over this much of its recent past.
    OWN_SLACK_S = 0.10

    # WHAT COUNTS AS A TRUSTWORTHY READING. Each constant here was decided from
    # a measurement (2026-10-16, tests/test_audio_own.py reproduces all of
    # them), because the coupling decides whether the app hears itself and the
    # opposite error is silent deafness: the two failures need opposite fixes
    # and look identical from the firing line.
    #
    # A reading is the bang's loudest microphone blocks over its own loud
    # head, divided by the loudest it was sent. It must stand clear of the
    # room, or it is the room: a bang that never reached the speakers (an
    # endpoint PortAudio did not report dead) reads noise-over-peak, 0.028
    # against a true 0.8, and three of those taught a coupling that made the
    # app hear its next real bang as 2 to 4 trigger pulls. The null case,
    # measured: the third-loudest of ~90 noise blocks over the 70th
    # percentile of 400 more, 600 trials of Gaussian noise: median 1.20,
    # 99th percentile 1.34, WORST 1.42. 3.0 is twice the worst. Heavy-tailed
    # noise (t with 3 degrees of freedom, a fan or a keyboard) reaches 5.3,
    # so this is a filter and not a proof: the median of several readings and
    # the reset on a reopened output are the other two layers.
    COUPLING_SNR_MIN = 3.0
    # How many of the head's loudest blocks to look at, and which of them is
    # the reading. A sample's loud head is many blocks near its peak (a 0.27s
    # stretch above half of it); a trigger click landing inside the head is
    # one or two. Taking the THIRD loudest means a click cannot be read as
    # the bang. Measured: at a sustained 0.62s cadence the click that arrives
    # 0.54s after each bang is inside the head, and the plain maximum
    # inflated the coupling from 0.80 to 1.57 and kept it there.
    COUPLING_TOPK = 3
    # Readings go STALE. Volume, the microphone's position and the room all
    # move over a session. A stale coupling is still the best estimate there
    # is, and it is used -- flagged, and re-measured at the next moment the
    # app owns the detections (`CouplingTeacher.begin`) -- rather than dropped.
    #
    # It used to be dropped, which was permanent deafness: the probe ran once
    # per (microphone, output stream), nothing re-measured, and the flat mute
    # came back for the rest of the session. Measured in the fake room, the
    # same fast string (0.5s gap) after the probe and again after 700s of not
    # shooting -- his sessions are launch, calibrate, talk, THEN shoot:
    #
    #     fresh probe, coupling 0.84      heard 12 of 12
    #     700s later, coupling dropped    heard  1 of 12
    #
    # Using a stale value is safe in one direction by construction (too high
    # only discounts a little more of the bang) and guarded in the other
    # (`LOUD_BLOCKS`: a bang louder than predicted forgets the coupling).
    COUPLING_MAX_AGE_S = 600.0
    # THE CEILING ON HOW LOUD THE APP'S OWN BANG CAN BE AT THE MICROPHONE,
    # per unit of digital output, used only while the coupling is NOT known.
    #
    # The flat mute (0.82s on his sample) exists to stop the app hearing its
    # own gunshot as a trigger pull. A bang can only be mistaken for one if
    # it can reach the click bar, and the app knows every term: the sample's
    # peak, the master volume, and the bar (the larger of `min_level` and
    # floor x margin). At a volume of 0.05 his sample (peak 1.166) leaves the
    # speaker at 0.058 and reaches the microphone at about 0.047, against a
    # bar of 0.10 from the measured floor and 0.3 once learned: two to six
    # times BELOW it. The mute that cost him ~18 of 40 shots was guarding
    # against something that could not happen.
    #
    # 2.0 is a bound, not a measurement of his room (the coupling has not yet
    # been measured on his hardware; the fake room's 0.8 is an assumption and
    # a loud one -- a speaker half a metre from a microphone at a normal
    # system volume is nowhere near unity). It is 2.5x that 0.8, and it
    # covers the largest probe-to-gunshot ratio of the 40 random rooms in
    # test_audio_own 5b (2.4x of a linear room's). Being wrong costs one
    # invented trigger pull, not a session: LOUD_BLOCKS notices a bang that
    # reaches the bar anyway and puts the flat mute back.
    COUPLING_CEILING = 2.0
    # A bang is LOUDER THAN PREDICTED when, inside its loud head, at least
    # this many microphone blocks would have been taken for a trigger click
    # AND they are at least this fraction of the blocks seen there. One
    # trigger click is one block and a few of ring (tens of ms: well under 12
    # blocks of 5.3 ms); a sustained bang is most of a 0.5 s head. Measured
    # in the fake room (test_audio_own section 9), probe first, then the
    # room's coupling changed, eight bangs with nobody shooting:
    #
    #                 before: invented trigger pulls     coupling afterwards
    #     unchanged           0                          0.79
    #     2x louder          29                          drifted 0.84 -> 0.62
    #     4x louder          32                          drifted 0.84 -> 0.31
    #
    # The drift fed itself: a block louder than predicted was classified as
    # a click, and clicked blocks are excluded from the next reading, so
    # exactly the evidence that the coupling had risen was thrown away.
    LOUD_BLOCKS = 12
    LOUD_FRACTION = 0.25
    # THE ONE SIGNIFICANCE THRESHOLD, used both to decide whether the bar is
    # raised for the bang and whether the microphone's blocks stay out of
    # the noise-floor history: the bang matters when its expected level at
    # the microphone, safety factor included, reaches this multiple of the
    # room's floor. Below it the bang is inside the room's own noise: raising
    # the bar there changes nothing (a click must clear floor x margin) and
    # excluding the block from the floor starves the adaptive floor of the
    # one thing it exists to track. It was a test for "> 0.0", which stayed
    # true for 1.59s after every bang, decided at one point by an envelope
    # value of 1.4e-8.
    #
    # The value is a trade between two measured errors (simulated string of 24
    # shots 0.7s apart, his 9mm sample, coupling 0.8, room noise as measured;
    # truth = the floor with no bangs at all). Excluding more starves the
    # floor; including more lets the bang's tail lift it:
    #
    #   significance      1       2       3       4       6
    #   volume 0.05  bias x1.05  x1.06  x1.39  x2.59  x2.59   floor tracks a
    #                tracked in 3.3s  2.5s   1.1s   0.7s   0.0s   3x louder room
    #   volume 0.2   bias x1.01  x1.15  x1.52  x1.57  x1.67   (bias is median
    #                tracked in 14.8s 11.8s  3.3s   3.2s   2.0s   floor / truth)
    #
    # 2.0 keeps the floor within 25% of the truth at every volume tried, which
    # is harmless because the click bar is the larger of floor x margin and the
    # learned min_level, and the latter dominates. At volume 0.5 the sample's
    # reverb is above the room's noise for the whole 0.7s gap, so the floor
    # genuinely cannot see the room during a string at his cadence -- that is
    # the physics of a loud speaker, not a threshold; it adapts again within
    # about a second of the last shot.
    OWN_SIGNIFICANCE = 2.0
    MIN_FLOOR = 1e-3          # a digital-silence floor must not make every bang matter

    def __init__(self, device: int | None = None, samplerate: int = 48000,
                 margin: float = 8.0, min_level: float = 0.15,
                 refractory_ms: float = 120.0, clock=time.perf_counter) -> None:
        # Injected so the self-hearing and cadence behaviour can be driven
        # block by block in a test, instead of sampled against a real clock.
        self._clock = clock
        self.device = device
        self.samplerate = samplerate
        # How far above the rolling floor a block must sit. 8x is about 18 dB.
        self.margin = margin
        # An absolute floor as well, so that in a silent room the adaptive
        # threshold cannot chase the noise down until a cough counts. This
        # starts deliberately high and is REPLACED by measurement: the first
        # live run with a guessed 0.035 reported six shots that never happened.
        self.min_level = min_level
        self.refractory_ms = refractory_ms
        # What this gun actually sounds like, learned from shots the camera
        # saw. Until there are a few of these the listener still detects, but
        # the app does not report its findings -- see `calibrated`.
        self._learned: list[float] = []
        self._recent: deque[tuple[float, float]] = deque(maxlen=800)  # ~4 s

        self._stream = None
        self._lock = threading.Lock()
        self._history: deque[float] = deque(maxlen=self.FLOOR_WINDOW)
        self._clicks: deque[Click] = deque(maxlen=256)
        # The same Click objects, kept AFTER `take_clicks` drains `_clicks` into
        # the corroborator. 64 is ample at his ~1/s cadence (a minute of
        # history) and the claim window is a third of a second.
        self._recent_clicks: deque[Click] = deque(maxlen=64)
        self._last_click_t = 0.0
        self._muted_until = 0.0
        # (start, until) of every self-mute, newest last. OBSERVATION ONLY:
        # nothing in the detection path reads it. It exists so the log can say
        # whether the microphone was DEAF at the moment of a shot, which a level
        # of 0.000 cannot tell apart from a silent room (`audio_at`).
        self._mutes: deque[tuple[float, float]] = deque(maxlen=64)
        self._own: list[_Own] = []
        # Bangs that have finished, kept briefly so `audio_at` can say whether
        # one of them was hiding a click at a moment a shot is asked about
        # (the shot is reported ~50-100 ms after it happened).
        self._own_done: deque[_Own] = deque(maxlen=16)
        self._readings: deque[_Reading] = deque(maxlen=16)
        self._filed = 0
        self._refused: dict[str, int] = {}
        self._resets = 0
        self._output_id = None
        # A bang read louder than predicted: whatever was believed about the
        # room is wrong, so the flat mute applies until a clean coupling is
        # learned again, and the ceiling is not trusted meanwhile.
        self._louder = False
        self.bangs = 0
        self.last_mode = ""
        # (kind, text) for the app to log. Written on the audio thread.
        self._notes: deque[tuple[str, str]] = deque(maxlen=64)
        self.heard = 0
        self.level = 0.0          # most recent block peak, for a live meter
        self.floor = 0.0          # current adaptive floor
        self.error = ""

    # -- lifecycle ------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def clock_is_perf_counter(self) -> bool:
        """Do `Click.t` and the camera's shot timestamps share an epoch?"""
        return self._clock is time.perf_counter

    def start(self) -> bool:
        if sd is None:
            self.error = "sounddevice is not available"
            return False
        if self._stream is not None:
            return True
        # A different microphone couples differently to the speakers.
        self.reset_coupling()
        try:
            self._stream = sd.InputStream(
                device=self.device, channels=1, samplerate=self.samplerate,
                blocksize=self.BLOCK, dtype="float32", callback=self._on_audio,
            )
            self._stream.start()
            self.error = ""
            return True
        except Exception as exc:                  # a mic in use, or no mic at all
            self._stream = None
            self.error = str(exc).strip().splitlines()[0] if str(exc) else repr(exc)
            return False

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    # -- the audio thread -----------------------------------------------------

    def _on_audio(self, indata, _frames, _time_info, _status) -> None:
        """Runs on PortAudio's thread. Keep it short and allocation-free."""
        now = self._clock()
        try:
            level = float(np.abs(indata[:, 0]).max())
        except Exception:
            return

        with self._lock:
            self.level = level
            self._recent.append((now, level))
            # What our own bangs should read at the microphone right now. Pure:
            # nothing is filed until the click decision below has been made,
            # because a block that IS a trigger click must not be read as the
            # bang (see COUPLING_TOPK).
            own = self._own_expected(now)
            matters = self._own_matters(own)
            # THE NOISE FLOOR IS THE ROOM'S, NOT THE APP'S OWN BANG. A 70th
            # percentile over ~2s sees a 0.5s bang every 0.7s as the
            # background and lifts the bar x8 above any trigger click, so
            # discounting the bang by level alone still left the channel
            # deaf at his cadence (heard 2 of 10 in simulation). Blocks the
            # app is itself responsible for stay out of the history -- but
            # only while it is a significant part of what the microphone
            # hears (`_own_matters`, the same test the bar uses). Excluding
            # on "any bang level at all" starved the floor for 1.59s after
            # every bang, i.e. for the whole of a string.
            ours = matters or now < self._muted_until
            if not ours:
                self._history.append(level)
            if len(self._history) >= 20:
                # A percentile, not a mean: the mean is dragged up by the very
                # transients being detected, which raises the bar every time a
                # shot is fired and makes the next one harder to hear.
                self.floor = float(np.percentile(np.fromiter(self._history,
                                                             dtype=np.float32), 70))
            elif not ours:
                self.floor = max(self.floor, level)

            clicked = self._decide_click(now, level, own, matters)
            # A click rings for tens of milliseconds; none of it is the bang.
            ringing = (now - self._last_click_t) * 1000.0 < self.refractory_ms
            # Would this block have passed for a trigger click, whatever the
            # mute or the refractory period decided? Counted separately from
            # `clicked`, because the readings leave clicked blocks out and a
            # bang louder than predicted is exactly what that discards.
            hot = level >= self._click_bar() and (not matters or level >= own)
            self._own_observe(now, level, ignore=clicked or ringing, hot=hot)

    def _click_bar(self) -> float:
        """The level a block must reach to be a trigger click, the bang's own
        discount aside. Reads only; callable under the lock."""
        return max(self.min_level, self.floor * self.margin)

    def _decide_click(self, now: float, level: float, own: float,
                      matters: bool) -> bool:
        """Is this block a trigger click? Called under the lock."""
        if now < self._muted_until:
            return False
        if (now - self._last_click_t) * 1000.0 < self.refractory_ms:
            return False
        if level < self.min_level or level < self.floor * self.margin:
            return False
        if matters and level < own:
            return False              # that is our own bang, not a trigger
        self._last_click_t = now
        self.heard += 1
        click = Click(t=now, level=level, floor=self.floor)
        self._clicks.append(click)
        self._recent_clicks.append(click)
        return True

    # -- the GUI thread -------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        """Has this microphone heard enough confirmed shots to be believed?"""
        return len(self._learned) >= self.LEARN_NEEDED

    @property
    def learned_level(self) -> float:
        if not self._learned:
            return 0.0
        s = sorted(self._learned)
        return s[len(s) // 2]

    def note_seen_shot(self, t: float, window: float = 0.25) -> float:
        """A shot the CAMERA saw. Record how loud it was, and learn from it.

        The camera is the reliable channel, so it is what the microphone is
        calibrated against. This is the same discipline as the detector's noise
        floor: measure the thing in front of you rather than carrying a number
        over from somebody else's room.
        """
        with self._lock:
            near = [lv for (ts, lv) in self._recent if abs(ts - t) <= window]
            peak = max(near) if near else 0.0
            if peak > 0.0:
                self._learned.append(peak)
                if len(self._learned) >= self.LEARN_NEEDED:
                    s = sorted(self._learned)
                    median = s[len(s) // 2]
                    # Well below the median so a quiet pull still counts, well
                    # above the room so nothing else does.
                    self.min_level = max(self.LEARN_FLOOR,
                                         median * self.LEARN_FRACTION)
            return peak

    def mute_for(self, seconds: float) -> None:
        """Ignore audio for a moment -- the app is about to make a noise itself.

        The announcer's speech is the one caller in `src/`: a line runs for
        seconds and is not registered as a bang. The app's own gunshot and
        effects do NOT come through here; they are `expect_own_sound`.
        """
        with self._lock:
            self._flat_mute(self._clock(), seconds)

    def _flat_mute(self, now: float, seconds: float) -> None:
        """Deaf from `now` for `seconds`, recorded for `audio_at`. Under the lock."""
        self._muted_until = max(self._muted_until, now + seconds)
        self._mutes.append((now, self._muted_until))

    def audio_at(self, t: float, window: float = 0.25) -> "tuple[float, int, float]":
        """What this microphone could say about time `t` (the listener's clock).

        Returns `(peak, blocks, deaf_for)`:
          peak      loudest block within `window` of t (0.0 if there was none)
          blocks    how many audio blocks fell in that window -- 0 means the
                    stream was not delivering, which is NOT a silent room
          deaf_for  seconds the app's own sound would still have hidden a
                    click at t (0.0 if the microphone was listening): either a
                    flat self-mute, or a bang loud enough at the microphone
                    that a typical trigger click was below the raised bar.
                    While it is > 0 no click could have been counted, whatever
                    the level -- and "seen, not heard" is no evidence then.

        Read-only. `note_seen_shot` is unchanged and still does the learning.
        """
        with self._lock:
            near = [lv for (ts, lv) in self._recent if abs(ts - t) <= window]
            deaf = 0.0
            for a, b in reversed(self._mutes):
                if a <= t <= b:
                    deaf = b - t
                    break
            deaf = max(deaf, self._masked_for(t))
            return (max(near) if near else 0.0), len(near), deaf

    # -- the app's own bang ---------------------------------------------------

    def _fresh_readings(self) -> list[_Reading]:
        now = self._clock()
        # list() first: the GUI thread reads this without the lock while the
        # audio thread appends, and iterating a deque that is being mutated
        # raises. list(deque) is one C call and cannot be interleaved.
        return [r for r in list(self._readings)
                if now - r.t <= self.COUPLING_MAX_AGE_S]

    def _coupling_from(self, readings: list[_Reading]) -> float | None:
        if len(readings) < self.COUPLING_NEEDED:
            return None
        s = sorted(r.value for r in readings)
        # The median of several bangs: a real click that landed inside one
        # bang's window inflates that bang's reading, never the others'.
        c = s[len(s) // 2]
        # A coupling of exactly 0.0 is not None, and it silently discounts
        # nothing while still reporting success. Not a measurement.
        return c if c > 0.0 else None

    def _pool_in_use(self) -> tuple[list[_Reading], bool]:
        """The readings the coupling is taken from, and whether they are stale.

        Fresh readings first; if none of the pools is large enough, the same
        choice over ALL readings, flagged stale. The real gunshot supersedes
        the probe within either.
        """
        fresh = self._fresh_readings()
        allr = list(self._readings)
        for pool, stale in ((fresh, False), (allr, True)):
            bangs = [r for r in pool if r.source == "bang"]
            if len(bangs) >= self.COUPLING_NEEDED:
                return bangs, stale
            probes = [r for r in pool if r.source == "probe"]
            if len(probes) >= self.COUPLING_NEEDED:
                return probes, stale
        return [r for r in fresh if r.source == "probe"], False

    def _readings_in_use(self) -> list[_Reading]:
        return self._pool_in_use()[0]

    @property
    def coupling_stale(self) -> bool:
        """Is the coupling in use older than COUPLING_MAX_AGE_S?

        True means it is still USED, and is due to be measured again at the
        next moment the app owns the detections.
        """
        rs, stale = self._pool_in_use()
        return bool(stale and self._coupling_from(rs) is not None)

    @property
    def coupling_distrusted(self) -> bool:
        """Has a bang read louder than predicted since the coupling was last
        learned? Then the ceiling is not trusted either."""
        return self._louder

    def protection_for(self, out_peak: float) -> str:
        """What a bang whose digital peak is `out_peak` would get right now:
        OWN_DISCOUNTED, OWN_BELOW_BAR or OWN_MUTED.

        ONE function, used by `expect_own_sound` to decide and by anything
        that reports, so what is reported is what would happen.
        """
        if self.coupling is not None:
            return OWN_DISCOUNTED
        if (not self._louder
                and out_peak * self.COUPLING_CEILING < self._click_bar()):
            return OWN_BELOW_BAR
        return OWN_MUTED

    @property
    def coupling(self) -> float | None:
        """Microphone level per unit of digital output level, or None if unknown.

        None covers: too few clean readings, readings too old, and a median
        that is not positive.
        """
        return self._coupling_from(self._readings_in_use())

    @property
    def coupling_source(self) -> str:
        """Where `coupling` comes from: "bang", "probe", or "" if unknown."""
        rs = self._readings_in_use()
        return rs[0].source if self._coupling_from(rs) is not None else ""

    def reset_coupling(self, why: str = "") -> None:
        with self._lock:
            self._readings.clear()
            self._own.clear()
            self._own_done.clear()
            self._louder = False          # a new endpoint is a new question
            self._resets += 1
            if why:
                self._notes.append(("reset", f"coupling forgotten: {why}"))

    def expect_own_sound(self, env: np.ndarray, dt: float,
                         fallback_mute_s: float = 0.0, *, played: bool = True,
                         output_id=None, gain: float = 0.0, sample: str = "",
                         source: str = "bang") -> str:
        """The app has just called play(): `env` is its digital level per `dt`.

        Returns what was done about it, one of the OWN_* outcomes, so the
        caller can SAY so: discounted by measurement, muted flatly because
        the coupling is not known, unprotected (neither), or not played at
        all. It used to return True for the first and False for the second,
        and said nothing for the third.

        `played=False` means the bang never reached a speaker. Then nothing
        is registered and nothing is muted -- there is no sound to hear --
        and the refusal is counted. Filing it as a reading was the one
        measured path to INVENTED trigger pulls: three bangs into a dead
        endpoint taught a coupling of 0.028 against a true 0.8, and when the
        endpoint recovered the app heard its own bang as 2 to 6 shots.

        `output_id` identifies the output stream. When it changes, whatever
        was measured on the old one is forgotten.
        """
        if output_id is not None:
            with self._lock:
                changed = self._output_id is not None and output_id != self._output_id
                self._output_id = output_id
            if changed:
                self.reset_coupling("the sound output was reopened, so the "
                                    "old measurement belongs to another "
                                    "endpoint")
        if not played:
            with self._lock:
                self._refuse("not played")
                self.last_mode = OWN_NOT_PLAYED
            return OWN_NOT_PLAYED
        env = np.asarray(env, dtype=np.float32).reshape(-1)
        if not len(env) or dt <= 0.0 or float(env.max()) <= 1e-4:
            # Empty or silent: nothing will reach the microphone either.
            with self._lock:
                self._refuse("silent")
                self.last_mode = OWN_NOT_PLAYED
            return OWN_NOT_PLAYED
        # Dilate backwards: at `tau` seconds after play() the sound at the
        # microphone is somewhere in the sample's last OWN_SLACK_S.
        w = int(np.ceil(self.OWN_SLACK_S / dt))
        padded = np.concatenate([env, np.zeros(w, dtype=np.float32)])
        dil = padded.copy()
        for k in range(1, w + 1):
            np.maximum(dil[k:], padded[:-k], out=dil[k:])
        with self._lock:
            now = self._clock()
            live = []
            for o in self._own:
                if now - o.t0 <= o.end:
                    live.append(o)
                else:
                    self._own_done.append(o)      # over, never filed: keep for audio_at
            self._own = live
            peak = float(dil.max())
            loud = np.nonzero(dil >= peak * self.HEAD_FRAC)[0]
            head = (int(loud[-1]) + 1) * dt if len(loud) else 0.0
            fresh = _Own(t0=now, env=dil, dt=dt, peak=peak,
                         end=len(dil) * dt, head_end=head, source=source,
                         gain=gain, sample=sample, fallback_s=fallback_mute_s)
            for o in self._own:
                if now - o.t0 <= o.head_end:      # still in its loud head
                    o.overlapped = fresh.overlapped = True
            how = self.protection_for(peak)
            if how == OWN_DISCOUNTED or how == OWN_BELOW_BAR:
                mode = how
            elif fallback_mute_s > 0.0:
                self._flat_mute(now, fallback_mute_s)
                mode = OWN_MUTED
            else:
                mode = OWN_UNPROTECTED
            fresh.mode = mode
            self.last_mode = mode
            self.bangs += 1
            self._own.append(fresh)
            if mode == OWN_UNPROTECTED:
                self._notes.append((
                    "unprotected",
                    "a bang was played that is neither discounted (the "
                    "coupling is not known) nor muted (no fallback was "
                    "given): the microphone WILL hear it as a trigger pull"))
        return mode

    def _refuse(self, reason: str) -> None:
        self._refused[reason] = self._refused.get(reason, 0) + 1

    @staticmethod
    def _env_at(o: _Own, t: float) -> float:
        """Digital level of bang `o` at clock time `t` (0.0 outside it)."""
        tau = t - o.t0
        if tau < 0.0 or tau > o.end:
            return 0.0
        i = int(tau / o.dt)
        return float(o.env[i]) if i < len(o.env) else 0.0

    def _own_expected(self, now: float) -> float:
        """What our own bangs should currently read at the microphone.

        0.0 when none is playing or the coupling is not yet measured (the flat
        mute covers that case). Pure: it changes nothing. Called under the lock.
        """
        if not self._own:
            return 0.0
        c = self.coupling
        if c is None:
            return 0.0
        total = 0.0
        for o in self._own:
            total += self._env_at(o, now)
        return c * total * self.OWN_SAFETY if total > 0.0 else 0.0

    # How finely `_masked_for` looks along a bang's envelope. The envelope
    # itself is in 5 ms steps; 10 ms is the resolution a log line is read at.
    MASK_STEP_S = 0.010

    def _masked_for(self, t: float) -> float:
        """Seconds from `t` for which the app's own bang would have hidden a
        trigger click at the click bar (0.0 if it would not at `t`).

        The discount raises the bar instead of closing the ears, so a click
        weaker than the bang's expected level at that instant is not counted
        -- and `audio_at` must say so, or a shot fired in the loud head of the
        previous bang would be reported "seen, no gunshot sound: likely a
        phantom" by a microphone that could not have heard it. Judged at the
        click BAR (the weakest click that could ever count), not at a typical
        click: that errs toward not accusing a shot of being a phantom, which
        is the direction the flat mute always erred. Read-only; under the lock.
        """
        c = self.coupling
        if c is None:
            return 0.0                # the flat mute, if any, is in `_mutes`
        owns = [o for o in (*self._own, *self._own_done) if t <= o.t0 + o.end]
        if not owns:
            return 0.0
        bar = self._click_bar()

        def level(x: float) -> float:
            return c * sum(self._env_at(o, x) for o in owns) * self.OWN_SAFETY

        if level(t) < bar:
            return 0.0
        last, x = t, t
        end = max(o.t0 + o.end for o in owns)
        while x <= end:
            if level(x) >= bar:
                last = x
            x += self.MASK_STEP_S
        return last + self.MASK_STEP_S - t

    def _own_matters(self, own: float) -> bool:
        """Is the bang a significant part of what the microphone hears?

        ONE test for both of its consequences -- the bar is raised for it, and
        its blocks stay out of the noise-floor history -- so they cannot
        disagree about when it is over. See OWN_SIGNIFICANCE.
        """
        return own > 0.0 and own >= self.OWN_SIGNIFICANCE * max(self.floor,
                                                                 self.MIN_FLOOR)

    def _own_observe(self, now: float, level: float, ignore: bool,
                     hot: bool = False) -> None:
        """Record what the microphone heard while each bang played, and file
        the readings of those that have finished. Called under the lock."""
        if not self._own:
            return
        alive = []
        for o in self._own:
            tau = now - o.t0
            if tau > o.end:
                self._file(o, now)
                continue
            alive.append(o)
            if 0.0 <= tau <= o.head_end:
                if not ignore:
                    o.top.append(level)
                    o.top.sort(reverse=True)
                    del o.top[self.COUPLING_TOPK:]
                # Only a bang that was NOT muted can be heard as a click, so
                # only those can be louder than the protection allowed for.
                if o.mode in (OWN_DISCOUNTED, OWN_BELOW_BAR):
                    o.blocks += 1
                    o.hot += 1 if hot else 0
                    if (o.hot >= self.LOUD_BLOCKS
                            and o.hot >= self.LOUD_FRACTION * o.blocks):
                        self._louder_than_predicted(o, now)
        self._own = alive

    def _louder_than_predicted(self, o: _Own, now: float) -> None:
        """The app's own bang is reaching the click bar. Under the lock.

        Whatever the app believed about the room -- a measured coupling, or
        the ceiling -- is wrong. Forget the coupling, stop trusting the
        ceiling, and put the flat mute back from this instant, for this bang
        and any other still playing. The bang's own reading is withheld: its
        loudest blocks were classified as clicks and left out, so it would
        teach a coupling LOWER than the truth, which is the drift that fed
        itself. The next bangs are muted flatly and so read cleanly.
        """
        had = self.coupling
        self._readings.clear()
        self._louder = True
        self._resets += 1
        until = now + 0.25
        for p in self._own:
            if p.mode in (OWN_DISCOUNTED, OWN_BELOW_BAR):
                p.mode = OWN_MUTED
                p.tainted = True
                until = max(until, p.t0 + p.fallback_s)
        if until > self._muted_until:
            self._muted_until = until
            self._mutes.append((now, until))
        self.last_mode = OWN_MUTED
        what = (f"the coupling it had measured ({had:.3f})" if had is not None
                else f"the ceiling it assumed ({self.COUPLING_CEILING:g})")
        self._notes.append((
            "reset",
            f"coupling forgotten: the app's own {o.source} was louder at the "
            f"microphone than {what} allows ({o.hot} of {o.blocks} blocks in "
            f"its loud head would have passed for a trigger click, volume "
            f"{o.gain:.2f}, sample {o.sample!r}). The room or the system "
            f"volume has changed. The flat mute applies until it is learned "
            f"again."))

    def _file(self, o: _Own, now: float) -> None:
        """A bang is over: keep its reading, or say why not. Under the lock."""
        self._own_done.append(o)
        if o.source == "effect":
            # An effect (a pop, a cue, a figure dropping) is registered so it is
            # muted or discounted like any bang, and is never a reading: it is
            # short, and it starts a millisecond after the gunshot that caused
            # it. Not a refusal either -- nothing was expected of it, and a log
            # line per target destroyed would bury the ones that matter.
            return
        floor = max(self.floor, self.MIN_FLOOR)
        reason = ""
        seen = 0.0
        if o.tainted:
            # Louder than predicted: its loudest blocks were taken for clicks
            # and left out, so what is left would read LOW (see above).
            reason = "louder than predicted"
        elif o.overlapped:
            # Another bang was playing: cannot say which was which.
            reason = "overlapped"
        elif len(o.top) < self.COUPLING_TOPK or o.peak <= 1e-4:
            reason = "too short to read"
        else:
            seen = o.top[self.COUPLING_TOPK - 1]
            if seen < self.COUPLING_SNR_MIN * floor:
                reason = "not above the room's noise"
        before = self.coupling
        if reason:
            self._refuse(reason)
            self._notes.append((
                "implausible" if reason.startswith("not above") else "refused",
                f"{o.source} {o.mode}: reading REFUSED ({reason}"
                + (f": {seen:.4f} against a floor of {floor:.4f}, needs "
                   f"{self.COUPLING_SNR_MIN:g}x" if seen else "")
                + f"). {self.describe_coupling()}"))
            return
        value = seen / o.peak
        self._readings.append(_Reading(now, value, o.source, o.gain, o.sample))
        self._filed += 1
        after = self.coupling
        if after is not None:
            self._louder = False      # learned again: the ceiling may be used
        switched = ""
        if o.source == "bang" and before is not None and self.coupling_source == "bang":
            probes = [r.value for r in self._fresh_readings() if r.source == "probe"]
            if probes and len(self._readings_in_use()) == self.COUPLING_NEEDED:
                p = sorted(probes)[len(probes) // 2]
                switched = (f" The probe had said {p:.3f}; your own shots say "
                            f"{after:.3f} (x{p / after:.2f}).") if after else ""
        self._notes.append((
            "reading",
            f"{o.source} {o.mode} (sample {o.sample!r}, volume {o.gain:.2f}): "
            f"reading {value:.3f}. {self.describe_coupling()}{switched}"))

    def describe_coupling(self) -> str:
        """One line a person can read: the value, its evidence, its refusals."""
        c = self.coupling
        rs = self._readings_in_use()
        refused = ", ".join(f"{k} {v}" for k, v in sorted(self._refused.items())) or "none"
        vol = f"{rs[-1].gain:.2f}" if rs else "n/a"
        stale = " [STALE: to be re-measured]" if self.coupling_stale else ""
        head = (f"coupling {c:.3f}{stale} from {len(rs)} clean "
                f"{self.coupling_source} reading(s)" if c is not None else
                f"coupling NOT KNOWN ({len(rs)} of {self.COUPLING_NEEDED} clean "
                f"readings; flat mute applies)")
        return (f"{head}, volume {vol}; filed {self._filed}, refused: "
                f"{refused}; resets {self._resets}")

    def coupling_report(self) -> dict:
        """The coupling and its evidence, for the log and the introspection server."""
        with self._lock:
            rs = self._readings_in_use()
            return {
                "coupling": self.coupling,
                "source": self.coupling_source,
                "clean": len(rs),
                "filed": self._filed,
                "refused": sum(self._refused.values()),
                "refused_by": dict(self._refused),
                "bangs": self.bangs,
                "mode": self.last_mode,
                "resets": self._resets,
                "stale": self.coupling_stale,
                "distrusted": self._louder,
            }

    def take_notes(self) -> list[tuple[str, str]]:
        """Drain what the audio side has to say, as (kind, text)."""
        with self._lock:
            out = list(self._notes)
            self._notes.clear()
        return out

    def take_clicks(self) -> list[Click]:
        """Drain everything heard since the last call."""
        with self._lock:
            out = list(self._clicks)
            self._clicks.clear()
        return out

    def claim_click_near(self, t: float, back: float,
                         forward: float) -> "Click | None":
        """Was a trigger click heard just before `t`? Claim it, or say no.

        Returns the UNCLAIMED click nearest in time to `t` within
        [t - back, t + forward] and marks it claimed, so one click can vouch
        for at most one detection. Nothing is drained: the corroborator still
        gets every click from `take_clicks`.

        Why a click and not `audio_at`'s peak: `audio_at` returns the loudest
        block in its window, and that INCLUDES THE APP'S OWN GUNSHOT. The app
        bangs for every detection it believes, so reading that as
        corroboration closes a loop (faint artefact -> bang -> the bang read
        as a click -> the artefact teaches the map) in which the instrument
        measures its own output. `_decide_click` already refuses our own bang
        (the own-sound discount, and a flat mute while the coupling is not
        yet known), so the click channel is the only safe one.

        Why exclusive: when a real shot and a faint artefact both land near
        one pull, only one of them may be taught. That bounds the damage to
        one bad sample per trigger pull rather than an unbounded number.
        """
        with self._lock:
            best: "Click | None" = None
            for c in self._recent_clicks:
                if c.matched:
                    continue
                dt = c.t - t
                if dt < -back or dt > forward:
                    continue
                if best is None or abs(dt) < abs(best.t - t):
                    best = c
            if best is not None:
                best.matched = True
            return best

    def snapshot(self) -> tuple[float, float, int]:
        with self._lock:
            return self.level, self.floor, self.heard


def default_listener(**kwargs) -> ClickListener:
    """The listener the app builds, behind a seam like `default_mixer`.

    `CameraLab.listener_factory` is bound to this when the class is defined, so
    a test that builds a CameraLab can hand it a listener that opens no input
    device (tests/silent_audio.py). Without the seam every such test opened the
    real microphone, so its behaviour depended on his room. Nothing about how a
    ClickListener works is decided here.
    """
    return ClickListener(**kwargs)


class ShotCorroborator:
    """Pairs up what was heard with what was seen, and says what it means.

    Kept separate from the listener so it can be tested without a microphone,
    which is the only way it will ever be tested at all.
    """

    def __init__(self, window_ms: float = 300.0) -> None:
        # Generous on purpose. The sound takes ~10 ms to cross the room, the
        # audio block adds ~5 ms, the camera adds up to two frames, and the
        # trigger click comes a few milliseconds before the laser lights. All of
        # that is far inside 300 ms, and nothing else is: two aimed shots that
        # close together would be exceptional.
        self.window = window_ms / 1000.0
        self._pending_clicks: list[Click] = []
        self._pending_shots: list[float] = []
        self.heard = 0
        self.seen = 0
        self.both = 0
        self.heard_not_seen = 0
        self.seen_not_heard = 0
        self.last_verdict = ""
        # The perf_counter time of the click (heard-not-seen) or of the shot
        # (seen-not-heard) behind each finding the LAST `resolve()` returned,
        # in the same order, so a caller can ask what the camera saw around it.
        self.last_found_times: list[float] = []
        # EVERY pairing decision, as (kind, t), for the one consumer that has to
        # count PULLS rather than findings: the magazine's dry-pull tally
        # (`ModeHost.note_pull_outcomes`). kind is "both" (t = the shot's time,
        # exactly as `add_shot` was given it), "heard-not-seen" (t = the click)
        # or "seen-not-heard" (t = the shot). A pull seen AND heard is ONE
        # outcome here, which is what lets a caller take the UNION of the two
        # channels instead of adding them. Bounded: nobody may drain it.
        self.outcomes: deque = deque(maxlen=256)

    def take_outcomes(self) -> list:
        """Drain the pairing decisions made since the last call."""
        out = list(self.outcomes)
        self.outcomes.clear()
        return out

    def reset(self) -> None:
        self.outcomes.clear()
        self._pending_clicks.clear()
        self._pending_shots.clear()
        self.heard = self.seen = self.both = 0
        self.heard_not_seen = self.seen_not_heard = 0
        self.last_verdict = ""
        self.last_found_times = []

    def add_clicks(self, clicks: list[Click]) -> None:
        for c in clicks:
            self.heard += 1
            matched = self._claim(self._pending_shots, c.t)
            if matched is not None:
                self.both += 1
                self.last_verdict = "heard and seen"
                self.outcomes.append(("both", matched))
            else:
                self._pending_clicks.append(c)

    def add_shot(self, t: float) -> None:
        self.seen += 1
        hit = None
        for i, c in enumerate(self._pending_clicks):
            if abs(c.t - t) <= self.window:
                hit = i
                break
        if hit is None:
            self._pending_shots.append(t)
        else:
            self._pending_clicks.pop(hit)
            self.both += 1
            self.last_verdict = "heard and seen"
            self.outcomes.append(("both", t))

    def _claim(self, times: list[float], t: float) -> "float | None":
        """Remove and return the time of the entry within the window of `t`."""
        for i, other in enumerate(times):
            if abs(other - t) <= self.window:
                times.pop(i)
                return other
        return None

    def resolve(self, now: float) -> list[str]:
        """Retire anything too old to find a partner. Returns new findings."""
        found: list[str] = []
        times: list[float] = []
        keep_c = []
        for c in self._pending_clicks:
            if now - c.t > self.window:
                self.heard_not_seen += 1
                found.append("heard-not-seen")
                times.append(c.t)
                self.outcomes.append(("heard-not-seen", c.t))
            else:
                keep_c.append(c)
        self._pending_clicks = keep_c

        keep_s = []
        for t in self._pending_shots:
            if now - t > self.window:
                self.seen_not_heard += 1
                found.append("seen-not-heard")
                times.append(t)
                self.outcomes.append(("seen-not-heard", t))
            else:
                keep_s.append(t)
        self._pending_shots = keep_s
        self.last_found_times = times
        return found
