"""The microphone and the app's own gunshot: does it hear itself, and does it
hear HIM?

No camera, no display, no sound card. A fake room: a speaker that plays his 9mm
sample, a microphone that hears it through a measured coupling plus room noise,
and a shooter whose trigger click precedes each bang by 80ms. The REAL
`ClickListener._on_audio` callback is driven block by block on a fake clock, and
the REAL `play_own_sound` and `CouplingTeacher` talk to a real `Mixer` whose
output stream is a stand-in.

NOTHING HERE CAN OPEN A SOUND DEVICE, and a check says so. `sounddevice`'s
OutputStream and InputStream are replaced by a tripwire before anything is built
(the way tests/test_silent_audio.py does it), and the suite fails if either was
ever constructed. The clock is injected; the room is seeded; no timer, no thread,
no wall clock is read. It gives the same answer on every machine in every room.

THE ONE THING IT READS FROM DISK is his 9mm sample, `sounds/freesound_community-
9mm-pistol-shoot-short-reverb-7152.mp3`, which is a tracked file of the repo and
the thing whose envelope every number below was measured on. If it cannot be
decoded the suite FAILS and says so; it does not fall back to a synthetic one,
because a different sample is different physics and would pass or fail for
reasons unrelated to the code.

Why this file exists (2026-10-16). The microphone is the only channel that
counts his trigger pulls independently of the camera, so the gun test --
"heard N, seen M" -- rests on it, and an outside review measured four ways it
could lie about N before he spent his trigger finger:

  1. a bang that never reached a speaker was filed as a reading of the room,
     teaching a coupling of 0.028 against a true 0.8; the app then heard its
     own bang as 2 to 6 trigger pulls
  2. the coupling could only be learned from his OWN string, and below a 0.6s
     cadence it never was: heard 1 of 12 at 0.5s, for the whole session
  3. the noise floor was shut out for 1.59s after every bang, i.e. for the
     whole of a string, by a test of "greater than zero" decided at an
     envelope value of 1.4e-8
  4. nothing anywhere reported the coupling, so both failures were invisible

and on 2026-10-23, over 8,415 logged shots, 55-65% of every shot he fires was
never heard because the flat self-mute covers most of his cadence.

SECTIONS ARE INDEPENDENT. `LSA_SECTIONS=1,5a,9` runs just those, which is how
tests/mutate_audio.py proves each check can fail in seconds instead of running
the whole file once per fault.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC))

import numpy as np                                                # noqa: E402

import audio_paths                                                # noqa: E402
import lsa.audio.bang as bang_mod                                 # noqa: E402
import lsa.audio.listener as listener_mod                         # noqa: E402
import lsa.audio.mixer as mixer_mod                               # noqa: E402
from lsa.audio.bang import (PROBE_NAME, SOURCE_EFFECT,            # noqa: E402
                            CouplingTeacher, fallback_mute_s,
                            play_own_sound)
from lsa.audio.listener import (OWN_BELOW_BAR, OWN_DISCOUNTED,    # noqa: E402
                                OWN_MUTED, OWN_NOT_PLAYED,
                                OWN_UNPROTECTED, ClickListener,
                                _Reading)
from lsa.artwork.assets import asset_dir                          # noqa: E402
from lsa.audio.mixer import Mixer, synth_probe                    # noqa: E402

# ---- the tripwire: no real stream, in either direction ----------------------
OPENED: list = []


class _Tripwire:
    def __init__(self, *a, **k):
        OPENED.append((a, k))
        raise RuntimeError("TRIPWIRE: a test tried to open a real audio stream")


for _m in (mixer_mod, listener_mod):
    if _m.sd is not None:
        _m.sd.OutputStream = _Tripwire
        _m.sd.InputStream = _Tripwire

SR, BLK = 48000, 256
TAIL_S = 0.25                       # CameraLab.SOUND_TAIL_S
NOISE = 0.004                       # per-sample sigma: ~ -48 dBFS, as measured

SAMPLE_FILE = "freesound_community-9mm-pistol-shoot-short-reverb-7152.mp3"

SHOT: "np.ndarray | None" = None


def load_sample() -> "tuple[np.ndarray | None, str]":
    """His 9mm sample decoded by the app's own loader, or (None, why not).

    Found through `asset_dir("sounds")`, the one place that says where the app's
    sounds are, so the mutation runner's scratch tree (which holds only `src` and
    `tests`) finds it through the same override the app does."""
    path = asset_dir("sounds") / SAMPLE_FILE
    m = Mixer.__new__(Mixer)
    m.samplerate, m._samples, m.error = SR, {}, ""
    if not path.exists():
        return None, f"{path} does not exist"
    if not Mixer.load(m, "shot", path):
        return None, m.error or "Mixer.load failed"
    return m._samples["shot"], ""


class _Stream:
    active = True


class Room:
    """A speaker, a microphone and a shooter, on a fake clock.

    `reaches=False` is the dead-endpoint case: PortAudio keeps the stream
    "active" and accepts writes, and nothing comes out. `stream.active=False`
    is the case the keepalive has already noticed.
    """

    def __init__(self, gain=0.5, coupling=0.8, calibrated=True, reaches=True,
                 noise=NOISE, seed=3, ring=False, warm=1.0, heavy=False):
        self.clk = 0.0
        self.c, self.reaches, self.noise, self.ring = coupling, reaches, noise, ring
        # A fan or a keyboard: Student's t with 3 degrees of freedom, scaled
        # to the same standard deviation as the Gaussian room.
        self.heavy = heavy
        self.events: list[tuple[float, str]] = []
        room = self

        class RoomMixer(Mixer):
            def play(self, name, gain=1.0):
                ok = super().play(name, gain)
                if ok:
                    room.events.append((room.clk, name))
                return ok

        self.mixer = RoomMixer(samplerate=SR, master_gain=gain)
        self.mixer.register("shot", SHOT)
        self.mixer.available = True
        self.mixer.device_label = "test speakers"   # not the real audio devices
        self.stream = _Stream()
        self.mixer._stream = self.stream
        self.lis = ClickListener(clock=lambda: self.clk)
        self.lis.device = 1
        self.lis._stream = object()     # 'running': a microphone is open
        if calibrated:          # as if three confirmed shots had been heard
            self.lis._learned = [0.9, 0.9, 0.9]
            self.lis.min_level = 0.3
        self.rng = np.random.default_rng(seed)
        self.logs: list[str] = []
        self.run(warm)          # the stream has been running; the floor is warm

    # -- the app side --------------------------------------------------------
    def bang(self):
        return play_own_sound(self.mixer, self.lis, "shot", TAIL_S)

    def effect(self):
        return play_own_sound(self.mixer, self.lis, "shot", TAIL_S,
                              source=SOURCE_EFFECT)

    # -- the room ------------------------------------------------------------
    def run(self, seconds, clicks=()):
        """Advance `seconds`. `clicks` = [(time, amplitude)] real trigger
        pulls. Returns how many clicks the audio thread heard."""
        heard0 = self.lis.heard
        end = self.clk + seconds
        while self.clk < end:
            self.clk += BLK / SR
            if self.heavy:
                blk = (self.rng.standard_t(3, BLK) * self.noise
                       * 0.577).astype(np.float32)
            else:
                blk = self.rng.normal(0.0, self.noise, BLK).astype(np.float32)
            if self.reaches:
                for t0, name in self.events:         # arrives ~30ms after play()
                    buf = self.mixer._samples[name]
                    i0 = int((self.clk - t0 - 0.030) * SR)
                    if 0 <= i0 < len(buf):
                        seg = buf[i0:i0 + BLK]
                        blk[:len(seg)] += self.c * np.minimum(
                            1.0, self.mixer.master_gain * seg)
            prev = self.clk - BLK / SR
            for tc, amp in clicks:
                if prev <= tc < self.clk:
                    blk[0] += amp
                elif self.ring and tc < prev < tc + 0.04:   # a real click rings
                    blk += amp * 0.6 * np.exp(-(prev - tc) / 0.012) * np.sin(
                        np.arange(BLK) * 0.4).astype(np.float32)
            self.lis._on_audio(blk.reshape(-1, 1), BLK, None, None)
        return self.lis.heard - heard0

    def teach_with_shots(self, n=4):
        """Bangs with nobody shooting, as the old bootstrap needed."""
        for _ in range(n):
            self.bang()
            self.run(2.0)

    def string(self, n=12, gap=0.7, amp=0.9):
        """n real shots `gap` apart; each bang plays 80ms after its click."""
        t0 = self.clk + 0.1
        shots = [t0 + i * gap for i in range(n)]
        end = t0 + n * gap + 1.0
        banged = set()
        heard = 0
        while self.clk < end:
            for s in shots:
                if s + 0.08 <= self.clk and s not in banged:
                    banged.add(s)
                    self.bang()
            heard += self.run(BLK / SR, clicks=[(s, amp) for s in shots])
        return heard, n

    def teacher(self) -> CouplingTeacher:
        return CouplingTeacher(self.mixer, self.lis, self.logs.append,
                               clock=lambda: self.clk)

    def teach_with_probe(self, may_play=lambda: True, limit=8.0):
        """Run the teacher the way the HUD tick does: every 0.2s."""
        tc = self.teacher()
        t0 = self.clk
        began = tc.begin()
        while tc.active and self.clk - t0 < limit:
            tc.tick(may_play())
            self.run(0.2)
        return tc, began, self.clk - t0


def sec_1(check) -> None:
    print("\n1. a bang that never reached a speaker teaches nothing")
    # The shot path logged "play() refused it" and then registered the bang
    # with the listener anyway. And PortAudio keeps `play()` returning True
    # into a dead endpoint, so even a True is not proof.
    r = Room()
    r.stream.active = False                 # the keepalive's view: stream dead
    b = r.bang()
    check("a dead output stream is reported as not played",
          not b.played and b.mode == OWN_NOT_PLAYED, f"{b}")
    check("and nothing was registered with the listener",
          r.lis.bangs == 0 and not r.lis._own, f"bangs={r.lis.bangs}")
    check("and nothing was muted for a sound that never happened",
          r.lis._muted_until == 0.0, f"muted until {r.lis._muted_until}")
    check("and the refusal is counted, with its reason",
          r.lis.coupling_report()["refused_by"].get("not played", 0) == 1,
          f"{r.lis.coupling_report()['refused_by']}")
    r2 = Room()
    r2.mixer.available = False              # play() itself refuses
    b2 = r2.bang()
    check("play() refusing is reported the same way",
          not b2.played and r2.lis.bangs == 0, f"{b2}")
    # ...and on its own, with a LIVE stream, so the two defences cannot cover
    # for each other: an unknown sample makes play() return False.
    r3 = Room()
    b3 = play_own_sound(r3.mixer, r3.lis, "no such sound", TAIL_S)
    check("play() refusing on a live stream is enough by itself",
          r3.mixer.audible and not b3.played and r3.lis.bangs == 0
          and r3.lis._muted_until == 0.0, f"{b3}, bangs {r3.lis.bangs}")

    # The harder case: the stream says it is alive and nothing comes out.
    r = Room(reaches=False)
    for _ in range(3):
        r.bang()
        r.run(2.0)
    check("three bangs into a silent endpoint teach NO coupling",
          r.lis.coupling is None,
          f"coupling {r.lis.coupling} (was 0.028 against a true 0.8)")
    refused = r.lis.coupling_report()["refused_by"]
    check("they are refused as 'not above the room's noise'",
          refused.get("not above the room's noise", 0) == 3, f"{refused}")
    kinds = [k for k, _ in r.lis.take_notes()]
    check("and the listener says the reading was implausible",
          kinds.count("implausible") == 3, f"{kinds}")
    r.reaches = True                        # the endpoint recovers
    r.events.clear()
    invented = 0
    for _ in range(4):
        r.bang()
        invented += r.run(2.0)
    check("when it recovers the app does not hear its own bang as shots",
          invented == 0, f"{invented} invented trigger pulls over 4 bangs "
          f"(6 before the fix)")
    check("and the coupling it then learns is the true one",
          r.lis.coupling is not None and abs(r.lis.coupling - 0.8) < 0.1,
          f"{r.lis.coupling}")


def sec_2(check) -> None:
    print("\n2. a reopened output is a different endpoint: forget the coupling")
    r = Room()
    r.teach_with_shots(4)
    check("PRECONDITION: the coupling was learned",
          r.lis.coupling is not None, f"{r.lis.coupling}")
    r.mixer.epoch += 1                      # what Mixer.start() does on reopen
    b = r.bang()
    check("the next bang on a new stream finds the coupling forgotten",
          r.lis.coupling is None and b.mode == OWN_MUTED,
          f"coupling {r.lis.coupling}, mode {b.mode}")
    check("and says why", any("reopened" in t for _, t in r.lis.take_notes()))
    # The real Mixer, with a stand-in for the sound card library: every time
    # the output stream is opened -- at start, and again when the keepalive
    # finds it dead -- the epoch must move, or the listener cannot know.
    import types

    class _OutStream:
        def __init__(self, **kw):
            self.active, self.latency = True, 0.02

        def start(self): pass
        def stop(self): self.active = False
        def close(self): pass

    _fake_sd = types.SimpleNamespace(
        OutputStream=_OutStream, default=types.SimpleNamespace(device=(0, 0)),
        query_devices=lambda *a, **k: {"name": "fake speakers"})
    _real_sd, mixer_mod.sd = mixer_mod.sd, _fake_sd
    try:
        m = Mixer(samplerate=SR)
        check("a real Mixer starts at epoch 0", m.epoch == 0, f"{m.epoch}")
        m.start()
        e1 = m.epoch
        live = m.audible
        m._stream.active = False             # the endpoint went away
        dead = m.audible
        note = m.ensure_alive()
        check("opening a stream moves the epoch, and so does reopening a dead one",
              e1 == 1 and m.epoch == 2 and "reopened" in note,
              f"after start {e1}, after ensure_alive {m.epoch}: {note!r}")
        check("and a dead stream is not 'audible'; a live one is",
              live and not dead, f"live {live}, dead {dead}")
    finally:
        mixer_mod.sd = _real_sd
    # Starting the listener on a microphone is a new question too: a different
    # microphone couples differently to the same speakers. The tripwire turns
    # the attempt to open it into an error `start()` reports, so no device is
    # touched; the coupling must be gone either way.
    r = Room()
    r.teach_with_shots(4)
    r.lis._stream = None
    hits = len(OPENED)
    ok = r.lis.start()
    del OPENED[hits:]                       # the one expected, harmless construction
    check("starting the listener forgets the coupling measured on another microphone",
          r.lis.coupling is None and not ok, f"coupling {r.lis.coupling}, start() {ok}")


def sec_3(check) -> None:
    print("\n3. a coupling of exactly 0.0 is not a measurement")
    r = Room()
    for _ in range(3):
        r.lis._readings.append(_Reading(r.clk, 0.0, "bang", 0.5, "shot"))
    check("three zero readings are not a known coupling",
          r.lis.coupling is None, f"{r.lis.coupling}")
    b = r.bang()
    check("the bang is muted flatly, not 'discounted' by nothing",
          b.mode == OWN_MUTED, f"{b.mode}")
    check("so the app does not hear its own bang",
          r.run(2.0) == 0)
    # And success is never reported for a bang that was neither.
    r = Room()
    mode = r.lis.expect_own_sound(np.ones(100, np.float32) * 0.5, 0.005, 0.0,
                                  gain=0.5, sample="shot")
    check("a bang with no coupling and no fallback mute is reported as UNPROTECTED",
          mode == OWN_UNPROTECTED, f"{mode}")
    check("and the listener says so", any(k == "unprotected"
                                          for k, _ in r.lis.take_notes()))
    r = Room()
    b = r.bang()
    check("play_own_sound always supplies a fallback, so it is never unprotected",
          b.mode in (OWN_MUTED, OWN_DISCOUNTED), f"{b.mode}")


def sec_4(check) -> None:
    print("\n4. 'implausibly low' is decided from measurement")
    # The null: how big does the third-loudest of ~90 noise blocks get against
    # the 70th percentile of 400, with no bang at all? Worst of 300 trials.
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        x = np.abs(rng.normal(0, 1, (490, BLK))).max(axis=1)
        floor = np.percentile(x[:400], 70)
        third = np.sort(x[400:])[::-1][ClickListener.COUPLING_TOPK - 1]
        worst = max(worst, third / floor)
    check("the SNR bar is at least twice the worst pure-noise reading",
          ClickListener.COUPLING_SNR_MIN >= 2.0 * worst,
          f"bar {ClickListener.COUPLING_SNR_MIN:g}, worst noise {worst:.2f} "
          f"over 300 trials (1.42 over 600)")
    for g in (0.5, 0.2):
        r = Room(gain=g)
        r.teach_with_shots(3)
        check(f"but a real bang at volume {g} clears it",
              r.lis.coupling is not None, f"coupling {r.lis.coupling}")
    r = Room(gain=0.02)
    r.teach_with_shots(3)
    check("while a bang too quiet to hear (volume 0.02) is refused",
          r.lis.coupling is None, f"coupling {r.lis.coupling}")
    # HEAVY-TAILED NOISE (a fan, a keyboard). The bar above is a filter and
    # not a proof: Student's t with 3 degrees of freedom reaches 5.3, so a
    # single reading from a bang that never arrived clears it now and then.
    # What stops that teaching a coupling is that THREE readings must clear it
    # and the median is taken. Asserted by driving the real listener, not by
    # the arithmetic: 40 rooms of heavy-tailed noise, three bangs each into a
    # dead endpoint (nothing arrives), counting how many single readings were
    # filed and how many rooms ended up believing a coupling.
    filed_total = learned = 0
    for seed in range(40):
        rh = Room(reaches=False, heavy=True, seed=100 + seed, warm=6.0)
        for _ in range(3):
            rh.bang()
            rh.run(2.0)
        filed_total += rh.lis.coupling_report()["filed"]
        learned += rh.lis.coupling is not None
    check("TEETH: heavy-tailed noise DOES clear the single-reading bar sometimes",
          filed_total >= 1,
          f"{filed_total} of 120 single readings were filed from pure noise; "
          f"if this is 0 the case is not heavy enough to test anything")
    check("but needing three readings keeps it from teaching a coupling",
          learned == 0,
          f"{learned} of 40 heavy-tailed rooms believed a coupling from "
          f"bangs that never arrived (~0.5% per room expected from 17% per "
          f"reading; the filter is the first layer, three-of-three the "
          f"second, and section 9's louder-than-predicted guard the third)")


def sec_5(check) -> None:
    print("\n5. the coupling is taught before he shoots, so fast strings are heard")
    for gap in (1.0, 0.7, 0.55, 0.5):
        cold = Room()
        h_cold, n = cold.string(12, gap)
        r = Room()
        tc, began, took = r.teach_with_probe()
        h, n = r.string(12, gap)
        if gap <= 0.55:
            check(f"TEETH: from a cold start at {gap}s he is mostly unheard",
                  h_cold <= n // 2, f"heard {h_cold} of {n}")
        check(f"after the probe every shot {gap}s apart is heard",
              h == n and tc.state == "done", f"heard {h} of {n}; teacher {tc.state}")
    r = Room(coupling=0.8)
    tc, began, took = r.teach_with_probe()
    c = r.lis.coupling
    check("the probe's coupling errs high, never low (the safe direction)",
          c is not None and 0.93 * 0.8 <= c <= 2.4 * 0.8,
          f"{c} against a true 0.8 (a linear room: the frequency-dependent "
          f"transfer is measured in 5b)")
    check("and it is labelled as the probe's", r.lis.coupling_source == "probe")


def sec_5a(check) -> None:
    print("\n5a. the probe is a measurement, announced, quiet, and bounded")
    r = Room()
    tc, began, took = r.teach_with_probe()
    check("it says it is a measurement, in the log, with the volume and the output",
          any("MEASURING THE MICROPHONE" in l and "not shooting" in l
              and "volume" in l for l in r.logs), f"{r.logs[:1]}")
    check("it does not add perceptible time: three sweeps in under 3s",
          took <= 3.0, f"{took:.1f}s of a launch that is already 30s")
    check("it plays exactly three sweeps",
          len([e for e in r.events if e[1] == PROBE_NAME]) == 3,
          f"{[e for e in r.events if e[1] == PROBE_NAME]}")
    shot_peak = float(np.abs(SHOT).max())
    for g in (0.05, 0.1, 0.2, 0.4):
        rg = Room(gain=g)
        tcg, _, _ = rg.teach_with_probe()
        reg = rg.mixer._samples[PROBE_NAME]
        out_peak = float(np.abs(reg).max()) * g
        check(f"at volume {g} the sweep is short, never louder than his gunshot, "
              f"and the measurement succeeds",
              len(reg) / SR <= 0.15 and float(np.abs(reg).max()) <= shot_peak
              and out_peak <= 0.2 and tcg.state == "done",
              f"{len(reg) / SR * 1000:.0f}ms, output peak {out_peak:.2f} "
              f"(gunshot {shot_peak * g:.2f}), {tcg.state}")
    check("and is not offered to the shooter as a gunshot to choose",
          PROBE_NAME not in r.mixer.names(), f"{r.mixer.names()}")
    # It must not read as a trigger pull to the MICROPHONE either.
    r = Room(calibrated=False)
    r.lis.min_level = 0.02           # as sensitive as it can be: the worst case
    h0 = r.lis.heard
    tc, _, _ = r.teach_with_probe()
    check("the microphone never mistakes a sweep for a click",
          r.lis.heard == h0, f"{r.lis.heard - h0} clicks heard during the probe")
    # Never while he could be shooting.
    r = Room()
    tc = r.teacher()
    tc.begin()
    tc.tick(True)
    r.run(0.2)
    n_before = len([e for e in r.events if e[1] == PROBE_NAME])
    tc.tick(False)                   # something that could be a shot is allowed
    for _ in range(10):
        r.run(0.2)
        tc.tick(False)
    n_after = len([e for e in r.events if e[1] == PROBE_NAME])
    check("it stands down the instant the app stops owning the detections",
          tc.state == "aborted" and n_after == n_before,
          f"state {tc.state}; sweeps {n_before} -> {n_after}")
    r = Room()
    tc, began, took = r.teach_with_probe(may_play=lambda: False)
    check("and never starts a sweep if the screen is not the app's",
          len([e for e in r.events if e[1] == PROBE_NAME]) <= 1
          and tc.state != "done", f"state {tc.state}")
    # Bounded, and not repeated.
    r = Room(reaches=False)
    tc, began, took = r.teach_with_probe()
    check("a probe that cannot be heard fails in bounded time",
          tc.state == "failed" and took <= tc.DEADLINE_S, f"{tc.state}, {took:.1f}s")
    check("and names the output device when the number is not believable",
          any("NOT audible" in l.upper() or "not audible" in l for l in r.logs)
          and any("volume" in l for l in r.logs), f"{r.logs[-2:]}")
    # A HUD tick that stalls must not leave the probe open for ever.
    r4 = Room()
    tc4 = r4.teacher()
    tc4.begin()
    tc4.tick(True)
    r4.run(tc4.DEADLINE_S + 5.0)             # nobody ticked for a long time
    played4 = len([e for e in r4.events if e[1] == PROBE_NAME])
    tc4.tick(True)
    check("a probe whose ticks stall gives up at its deadline, playing nothing more",
          tc4.state == "failed"
          and len([e for e in r4.events if e[1] == PROBE_NAME]) == played4,
          f"state {tc4.state}")
    again = tc.begin()
    check("and is not retried in a loop on the same output (RULES D10)",
          not again, "begin() returned True a second time")
    r.mixer.epoch += 1
    check("but a new output stream is a new question",
          tc.begin(), "begin() refused after the output changed")
    r = Room()
    r.teach_with_shots(4)
    tc = r.teacher()
    check("nothing is taught when the coupling is already known",
          not tc.begin() and tc.state == "idle")
    r = Room()
    r.lis._stream = None
    tc = r.teacher()
    check("or when there is no microphone to hear it",
          not tc.begin())


def sec_5b(check) -> None:
    print("\n5b. does the probe's coupling predict the gunshot's?")
    # Through 40 random speaker-and-room responses: a 2nd-order high-pass
    # 80-1200 Hz, a low-pass, sometimes a resonance, one to three echoes.
    # The ratio is (probe coupling)/(gunshot coupling), each read the way the
    # listener reads it (third-loudest 256-sample block). Under 1 means the
    # probe would teach too small a coupling and the app would hear itself.
    def through(x, hp, lp, bump, echoes):
        n = len(x) + int(0.2 * SR)
        X = np.fft.rfft(x, n)
        f = np.fft.rfftfreq(n, 1 / SR)
        f[0] = 1e-3
        h = (1j * f / hp) ** 2 / (1 + 1.414 * 1j * f / hp + (1j * f / hp) ** 2)
        h = h / (1 + 1j * f / lp)
        if bump:
            h = h * (1 + bump[1] * np.exp(-((f - bump[0]) / bump[2]) ** 2))
        y = np.fft.irfft(X * h, n)
        out = y.copy()
        for d, a in echoes:
            k = int(d * SR)
            out[k:] += a * y[:-k]
        return out

    def coupling_of(x, hp, lp, bump, echoes):
        y = through(x.astype(np.float64), hp, lp, bump, echoes)
        nb = len(y) // BLK
        s = np.sort(np.abs(y[:nb * BLK]).reshape(nb, BLK).max(axis=1))[::-1]
        return s[ClickListener.COUPLING_TOPK - 1] / np.abs(x).max()

    t = np.arange(int(0.12 * SR)) / SR
    tone = (np.sin(2 * np.pi * 1000.0 * t)
            * np.clip(np.minimum(t / 0.012, (0.12 - t) / 0.012), 0, 1))
    rng = np.random.default_rng(5)
    sweep_r, tone_r = [], []
    sweep = synth_probe(SR)
    for _ in range(40):
        hp = float(np.exp(rng.uniform(np.log(80), np.log(1200))))
        lp = float(rng.uniform(4000, 12000))
        bump = None
        if rng.random() < 0.5:
            bump = (float(rng.uniform(800, 4000)), float(rng.uniform(0.3, 2.5)),
                    float(rng.uniform(200, 800)))
        ech = tuple((float(rng.uniform(0.003, 0.04)), float(rng.uniform(0.05, 0.6)))
                    for _ in range(int(rng.integers(1, 4))))
        cs = coupling_of(SHOT, hp, lp, bump, ech)
        sweep_r.append(coupling_of(sweep, hp, lp, bump, ech) / cs)
        tone_r.append(coupling_of(tone, hp, lp, bump, ech) / cs)
    check("the sweep never reads below 0.9x of the gunshot, in any of 40 rooms",
          min(sweep_r) >= 0.9, f"min {min(sweep_r):.2f}, median "
          f"{np.median(sweep_r):.2f}, max {max(sweep_r):.2f}")
    check("TEETH: a single 1 kHz tone WOULD have read low in some of them",
          min(tone_r) < 0.7, f"tone min {min(tone_r):.2f} -- if this passes "
          f"the test cannot tell a good probe from a bad one")
    # the safety factor in use covers the sweep's worst room as well:
    check("and the bar the listener actually uses still covers the worst room",
          min(sweep_r) * ClickListener.OWN_SAFETY >= 1.0,
          f"{min(sweep_r) * ClickListener.OWN_SAFETY:.2f}")


def sec_6(check) -> None:
    print("\n6. the noise floor is graded, not shut out for a whole string")
    truth_room = Room()
    truth_room.run(6.0)
    truth = truth_room.lis.floor
    loud_room = Room(noise=3 * NOISE)
    loud_room.run(6.0)
    truth_loud = loud_room.lis.floor

    import collections

    class Counting(collections.deque):
        n = 0

        def append(self, x):
            Counting.n += 1
            super().append(x)

    def floor_trial(gain, gap=0.7, n=24, step=None):
        r = Room(gain=gain)
        r.teach_with_shots(4)
        r.lis._history = Counting(r.lis._history, maxlen=r.lis.FLOOR_WINDOW)
        Counting.n = 0
        t0 = r.clk + 0.1
        shots = [t0 + i * gap for i in range(n)]
        banged, end, blocks, trace, tstep = set(), t0 + n * gap + 1.0, 0, [], None
        while r.clk < end:
            for s in shots:
                if s + 0.08 <= r.clk and s not in banged:
                    banged.add(s)
                    r.bang()
            if step and tstep is None and r.clk > t0 + step[0]:
                r.noise, tstep = step[1], r.clk
            r.run(BLK / SR, clicks=[(s, 0.9) for s in shots])
            trace.append((r.clk, r.lis.floor))
            blocks += 1
        return Counting.n / blocks, np.array(trace), tstep, r

    frac, tr, _, r = floor_trial(0.05)
    check("at volume 0.05 and his cadence the floor receives a real share of blocks",
          frac >= 0.15, f"{frac * 100:.1f}% of blocks (1.2% when any bang level "
          f"at all excluded a block)")
    check("without lifting the floor",
          np.median(tr[len(tr) // 3:, 1]) <= 1.25 * truth,
          f"{np.median(tr[len(tr) // 3:, 1]) / truth:.2f}x the true floor")
    frac, tr, tstep, r = floor_trial(0.05, step=(3.0, 3 * NOISE))
    after = tr[tr[:, 0] > tstep]
    ok = np.nonzero(after[:, 1] >= 0.7 * truth_loud)[0]
    took = after[ok[0], 0] - tstep if len(ok) else 99.0
    check("and when the room gets 3x louder mid-string the floor follows it",
          took <= 5.0, f"reached 70% of the new floor after {took:.1f}s "
          f"(never, before)")
    check("hearing every shot meanwhile", r.lis.heard == 24, f"heard {r.lis.heard}/24")
    frac, tr, _, r = floor_trial(0.2)
    check("at volume 0.2 it stays within 25% of the truth",
          tr[len(tr) // 3:, 1].max() <= 1.30 * truth,
          f"max {tr[len(tr) // 3:, 1].max() / truth:.2f}x")


def sec_7(check) -> None:
    print("\n7. the coupling cannot be inflated by his own shots")
    # At a sustained 0.62s cadence the click arriving 0.54s after each bang
    # lies inside its loud head. The plain maximum read it as the bang and the
    # coupling went 0.80 -> 1.57 and stayed there.
    for ring in (False, True):
        r = Room(ring=ring)
        r.teach_with_shots(4)
        h, n = r.string(40, 0.62)
        c = r.lis.coupling
        check(f"40 shots at 0.62s keep the coupling honest"
              f"{' (clicks ring)' if ring else ''}",
              c is not None and abs(c - 0.8) < 0.08 and h == n,
              f"coupling {c} against a true 0.8 (1.57 before); heard {h}/{n}")
    # And with no probe at all: learning from a slow string must work, which
    # is what the loud-head fraction is for.
    r = Room()
    h, n = r.string(10, 1.0)
    c = r.lis.coupling
    check("a string at 1.0s teaches the coupling by itself (head fraction)",
          c is not None and abs(c - 0.8) < 0.1,
          f"coupling {c}; heard {h}/{n}. With the whole 1.49s sample as its "
          f"'head' every bang overlaps the next and nothing is ever learned")
    # HOW MANY ARE HEARD is a different question, and it depends on the VOLUME.
    # Room() plays at 0.5, ten times his 0.05. Until the coupling is known the
    # flat mute applies, and it is volume-scaled (`Mixer.mute_fraction`): at 0.5
    # it is the sample's part above 1% of its peak plus the tail, longer than
    # the 0.92s from a bang to the next click at a 1.0s cadence. So the first
    # four clicks after the first are deaf by design: 6 of 10 heard, which is
    # the whole of the failure this check once reported (clicks 1 and 6..10).
    # The reference measured "12 of 12" with an UNSCALED mute (0.81s), i.e.
    # the conditions of his own volume. At that volume nothing needs muting at
    # all (the bang cannot reach the click bar), so every click is heard.
    gap_to_next_click = 1.0 - 0.08
    check("at volume 0.5 the flat mute outlasts a 1.0s cadence (why 6 of 10 is "
          "correct there, and the scaling is what is being exercised)",
          fallback_mute_s(r.mixer, "shot", TAIL_S) > gap_to_next_click,
          f"{fallback_mute_s(r.mixer, 'shot', TAIL_S):.2f}s against "
          f"{gap_to_next_click:.2f}s")
    rq = Room(gain=0.05)
    hq, nq = rq.string(10, 1.0)
    check("at his volume (0.05) a string at 1.0s is heard in full, cold",
          hq == nq, f"heard {hq}/{nq}, coupling {rq.lis.coupling}")
    # Readings go stale.
    r = Room()
    r.teach_with_shots(4)
    r.clk += 60.0
    check("a reading a minute old is fresh",
          r.lis.coupling is not None and not r.lis.coupling_stale,
          f"{r.lis.coupling}")
    r.clk += 3600.0
    check("a reading an hour old is STALE and is still used (the best "
          "estimate there is; the guards in section 9 cover it being wrong)",
          r.lis.coupling is not None and r.lis.coupling_stale,
          f"{r.lis.coupling}, stale {r.lis.coupling_stale}")
    b = r.bang()
    check("so a bang is still discounted, not muted flatly",
          b.mode == OWN_DISCOUNTED, f"{b.mode}")
    check("and the report says it is stale",
          r.lis.coupling_report()["stale"] and "STALE" in r.lis.describe_coupling())


def sec_7b(check) -> None:
    print("\n7b. the probe's answer goes stale, and something re-measures it")
    # Measured (the review): after the probe, coupling 0.84, a fast string
    # heard 12 of 12; after 700s of not shooting the coupling EXPIRED, the
    # teacher refused a second attempt for the same (device, stream) because
    # that key does not move when a coupling expires, and the same fast
    # string heard 1 of 12. His sessions are launch, calibrate, talk, then
    # shoot.
    r = Room()
    tc, _, _ = r.teach_with_probe()
    h_fresh, n = r.string(12, 0.5)
    check("PRECONDITION: right after the probe a 0.5s string is fully heard",
          h_fresh == n and tc.state == "done", f"heard {h_fresh} of {n}")
    r.clk += 700.0                       # eleven minutes of not shooting
    check("PRECONDITION: the probe's reading is now older than the limit",
          r.lis.coupling_stale, f"stale {r.lis.coupling_stale}")
    h_late, n = r.string(12, 0.5)
    check("a 0.5s string eleven minutes later is STILL fully heard "
          "(it was 1 of 12 when expiry dropped the coupling)",
          h_late == n, f"heard {h_late} of {n}")
    again = tc.begin()
    check("and the next moment the app owns the detections the probe is "
          "allowed again, because its answer has gone stale",
          again and tc.state == "running", f"begin() returned {again}")
    while tc.active and tc.state == "running":
        tc.tick(True)
        r.run(0.2)
    check("and it re-measures: fresh coupling, no longer stale",
          tc.state == "done" and r.lis.coupling is not None
          and not r.lis.coupling_stale,
          f"{tc.state}, coupling {r.lis.coupling}, stale {r.lis.coupling_stale}")
    check("and says it is redoing an earlier measurement",
          any("redone" in l for l in r.logs), f"{r.logs[-3:]}")
    check("but a fresh one is not redone (no loop)", not tc.begin())
    # A failed probe is still not retried: only a SUCCESS that has since gone.
    rf = Room(reaches=False)
    tcf, _, _ = rf.teach_with_probe()
    rf.clk += 700.0
    check("a probe that FAILED is not retried because time passed (RULES D10)",
          tcf.state == "failed" and not tcf.begin())


def sec_8(check) -> None:
    print("\n8. the coupling is reported, with its evidence")
    r = Room()
    r.teach_with_shots(4)
    notes = r.lis.take_notes()
    text = " | ".join(t for _, t in notes)
    check("every reading is logged", len([k for k, _ in notes if k == "reading"]) >= 3,
          f"{[k for k, _ in notes]}")
    for what, needle in (("the value", "coupling 0."),
                         ("how many clean readings", "clean"),
                         ("what was refused", "refused:"),
                         ("the volume", "volume 0.50"),
                         ("the sample", "'shot'"),
                         ("whether the bang was muted or discounted", "muted")):
        check(f"a reading line carries {what}", needle in text, f"needle {needle!r}")
    check("the one-line description of the coupling carries the volume too",
          "volume 0.50" in r.lis.describe_coupling(), r.lis.describe_coupling())
    r.bang()
    r.run(2.0)
    text2 = " | ".join(t for _, t in r.lis.take_notes())
    check("and a later bang says it was DISCOUNTED", "discounted" in text2, text2[:120])
    rep = r.lis.coupling_report()
    check("the report carries the fields a reader needs",
          {"coupling", "source", "clean", "refused", "mode"} <= set(rep), f"{sorted(rep)}")
    check("with sensible values", rep["coupling"] and rep["clean"] >= 3
          and rep["mode"] == OWN_DISCOUNTED, f"{rep}")
    # The listener's clock is the camera's clock in the app.
    check("by default the listener's clock is time.perf_counter (RULES B4)",
          ClickListener().clock_is_perf_counter)
    check("and one built with another clock says so",
          not ClickListener(clock=lambda: 0.0).clock_is_perf_counter)


def sec_9(check) -> None:
    print("\n9. a bang LOUDER than predicted is not him: forget, mute, relearn")
    # Coupling is microphone level per unit of DIGITAL output. The app's own
    # volume is handled; the system or monitor volume is not -- and his
    # speakers are inside the monitor. Probe first, then the room's coupling
    # changes, eight bangs with nobody shooting. Before the fix:
    #     room unchanged  0 invented, coupling 0.79
    #     room 2x louder 29 invented, coupling drifted 0.84 -> 0.62
    #     room 4x louder 32 invented, coupling drifted 0.84 -> 0.31
    for mult in (1.0, 2.0, 4.0):
        r = Room()
        tc, _, _ = r.teach_with_probe()
        c0 = r.lis.coupling
        resets0 = r.lis.coupling_report()["resets"]
        r.c = 0.8 * mult
        invented = 0
        for _ in range(8):
            r.bang()
            invented += r.run(2.0)
        c = r.lis.coupling
        resets = r.lis.coupling_report()["resets"] - resets0
        text = " | ".join(t for _, t in r.lis.take_notes())
        if mult == 1.0:
            check("the room unchanged: nothing invented, nothing forgotten",
                  invented == 0 and resets == 0 and c is not None
                  and abs(c - c0) < 0.1 * c0,
                  f"{invented} invented, {resets} resets, {c0:.2f} -> {c:.2f}")
            continue
        check(f"the room {mult:g}x louder: at most ONE invented pulse "
              f"(29-32 before), not one per bang",
              invented <= 1, f"{invented} invented over 8 bangs")
        check(f"and the coupling is not left drifting low: withdrawn, or "
              f"relearned near the true {0.8 * mult:.1f}",
              c is None or abs(c - 0.8 * mult) < 0.25 * 0.8 * mult,
              f"{c} (0.31 / 0.62 before)")
        check("and it relearns it from the muted bangs that follow",
              c is not None, f"coupling {c}")
        check("and says why it forgot",
              "louder" in text and "forgot" in text, text[:160])
        if mult == 2.0:
            r2 = Room()
            tc2, _, _ = r2.teach_with_probe()
            r2.c = 1.6
            r2.bang()
            r2.run(1.0)
            check("a withdrawn coupling lets the probe measure again "
                  "(its answer is gone; the key did not move)",
                  r2.lis.coupling is None and tc2.begin(),
                  f"coupling {r2.lis.coupling}, state {tc2.state}")
    # The ceiling used while the coupling is unknown is guarded the same way.
    r = Room(gain=0.05, coupling=6.0)       # three times past the ceiling
    r.bang()
    first = r.run(2.0)
    check("a room past the CEILING is noticed on the first bang: the ceiling "
          "is no longer trusted and the bang's own reading is withheld",
          r.lis.coupling_distrusted and r.lis.coupling is None
          and r.lis.protection_for(0.058) == OWN_MUTED,
          f"distrusted {r.lis.coupling_distrusted}, coupling {r.lis.coupling}, "
          f"protection {r.lis.protection_for(0.058)}")
    invented = first
    for _ in range(5):
        r.bang()
        invented += r.run(2.0)
    check("and at most one pulse was invented over 6 bangs",
          invented <= 1, f"{invented} over 6 bangs")
    check("and the coupling is then LEARNED (true 6.0) and the ceiling "
          "trusted again",
          r.lis.coupling is not None and abs(r.lis.coupling - 6.0) < 1.5
          and not r.lis.coupling_distrusted,
          f"coupling {r.lis.coupling}, distrusted {r.lis.coupling_distrusted}")
    rr = Room(gain=0.05, coupling=2.0)      # exactly the ceiling: still fine
    inv = 0
    for _ in range(6):
        rr.bang()
        inv += rr.run(2.0)
    check("a room AT the ceiling invents nothing", inv == 0, f"{inv}")
    # A real trigger click is not mistaken for a loud bang, however it rings.
    for ring in (False, True):
        r = Room(ring=ring)
        tc, _, _ = r.teach_with_probe()
        h, n = r.string(30, 0.62)
        check(f"30 real shots at 0.62s are never read as a louder room"
              f"{' (clicks ring)' if ring else ''}",
              r.lis.coupling_report()["resets"] == 0 and h == n
              and not r.lis.coupling_distrusted,
              f"resets {r.lis.coupling_report()['resets']}, heard {h}/{n}")


def sec_10(check) -> None:
    print("\n10. the fallback does not go deaf for a bang that cannot be heard")
    # His sample peaks at 1.166 and is loud for 0.57s. At a volume of 0.05 it
    # reaches the microphone at about 0.047, against a click bar of 0.10 from
    # the measured floor and 0.3 once learned: 2 to 6 times BELOW it. The 0.82s
    # flat mute that cost ~18 of 40 shots guarded against something that could
    # not happen -- and was applied exactly when the probe fails, which is
    # when the bang is too quiet for it to read.
    peak = float(np.abs(SHOT).max())
    for calibrated in (True, False):
        r = Room(gain=0.05, calibrated=calibrated)
        bar = r.lis._click_bar()
        b = r.bang()
        check(f"volume 0.05, coupling unknown, bar {bar:.2f}: the bang is "
              f"neither muted nor discounted",
              b.mode == OWN_BELOW_BAR and r.lis._muted_until == 0.0,
              f"{b.mode}, muted until {r.lis._muted_until}")
        h, n = r.string(12, 0.5)
        check(f"and a 0.5s string with NO probe at all is fully heard "
              f"({'learned' if calibrated else 'default'} bar)",
              h == n, f"heard {h} of {n} (1 of 12 under the flat mute)")
        check("and the app did not hear itself", r.lis.coupling_report()["resets"] == 0)
    # Teeth: where the bang CAN reach the bar the protection stays.
    for g in (0.2, 0.5):
        r = Room(gain=g)
        b = r.bang()
        check(f"volume {g}: a bang that could reach the bar is still muted",
              b.mode == OWN_MUTED and r.lis._muted_until > 0.0, f"{b.mode}")
    # What is REPORTED is what happens: one function decides both.
    for g in (0.05, 0.5):
        r = Room(gain=g)
        out_peak = min(1.0, peak * g)
        said = r.lis.protection_for(out_peak)
        did = r.bang().mode
        check(f"volume {g}: protection_for() says what the bang then gets",
              said == did, f"said {said}, got {did}")
    r = Room(gain=0.5)
    r.teach_with_shots(4)
    check("and once the coupling is known it says discounted",
          r.lis.protection_for(0.5) == OWN_DISCOUNTED)
    # The probe's own sweeps at his volume do not need the mute either, and
    # the measurement still succeeds.
    r = Room(gain=0.05)
    tc, _, took = r.teach_with_probe()
    check("the probe at volume 0.05 still measures, with no mute in its way",
          tc.state == "done" and r.lis.coupling is not None,
          f"{tc.state}, coupling {r.lis.coupling}")


def sec_11(check) -> None:
    print("\n11. a shot the discounted bang could have hidden is NOT evidence of a phantom")
    # THE DISCOUNT RAISES THE BAR INSTEAD OF CLOSING THE EARS. The flat mute was
    # recorded for `audio_at`, so a shot fired inside it was reported "the
    # microphone was DEAF - not evidence of a phantom". A discounted bang is
    # not a mute and left nothing there: a click weaker than the bang's
    # expected level was dropped, and the shot it belonged to was then logged
    # "SAW a shot with no gunshot sound: likely a phantom" by a microphone that
    # could not have heard it. Same channel, same honesty (RULES C3, C4).
    amps = (0.05, 0.1, 0.2, 0.35, 0.5, 0.9, 1.2)
    unheard = 0
    broken = []
    for amp in amps:
        r = Room()
        r.teach_with_probe()
        r.bang()
        r.run(0.10)                         # inside the loud head, past its attack
        tc = r.clk + 0.002
        heard = r.run(2 * BLK / SR, clicks=[(tc, amp)])
        deaf = r.lis.audio_at(tc)[2]
        unheard += 0 if heard else 1
        if not heard and not deaf > 0.0:
            broken.append((amp, round(deaf, 3)))
    check("a click the discounted bang hid is always reported as hidden "
          "(never accusable as a phantom)",
          not broken, f"unheard and not deaf: {broken}")
    check("TEETH: some of those clicks WERE hidden (the case is not vacuous)",
          unheard >= 1, f"{unheard} of {len(amps)} unheard")
    r = Room()
    r.teach_with_probe()
    r.bang()
    r.run(0.10)
    deaf_in = r.lis.audio_at(r.clk)[2]
    check("inside the loud head the microphone reports a bounded deafness",
          0.0 < deaf_in <= 1.6, f"{deaf_in:.3f}s")
    r.run(3.0)
    check("and once the bang is over it reports none",
          r.lis.audio_at(r.clk)[2] == 0.0, f"{r.lis.audio_at(r.clk)[2]}")
    r.run(0.5)
    tc = r.clk + 0.002
    heard = r.run(2 * BLK / SR, clicks=[(tc, 0.35)])
    check("and a weak click after the bang is heard",
          heard == 1 and r.lis.audio_at(tc)[2] == 0.0, f"heard {heard}")
    # A bang that cannot reach the bar hides nothing, so nothing is claimed.
    r = Room(gain=0.05)
    r.teach_with_probe()
    r.bang()
    r.run(0.10)
    check("a quiet bang (volume 0.05) hides no click, so none is excused",
          r.lis.audio_at(r.clk)[2] == 0.0, f"{r.lis.audio_at(r.clk)[2]}")
    # And the flat mute, while the coupling is unknown, is still recorded.
    r = Room()
    r.bang()
    r.run(0.10)
    deaf_flat = r.lis.audio_at(r.clk)[2]
    check("while the coupling is unknown the flat mute is what audio_at reports",
          deaf_flat > 0.5, f"{deaf_flat:.3f}s")


def sec_12(check) -> None:
    print("\n12. one path from the app to the room (and the registered sounds behave)")
    # The Test button, the effect that follows a bang, the dry-pull pop, the
    # selection cue and a target appearing are all sounds the microphone can
    # mistake for a trigger pull. Found by AST over the tree under test, so a
    # reintroduced second path turns this red in the mutation runner's scratch
    # copy as well as in the real one.
    bad = audio_paths.violations(SRC)
    check("no second path from the app to the speakers or the listener",
          not bad, "; ".join(bad))
    plays = {rel for rel, _ln, _fn in audio_paths.play_sites(SRC)}
    check("and the two allowed callers are still callers (the allowlist is not stale)",
          {"lsa/audio/bang.py", "lsa/audio/voice.py"} <= plays, str(sorted(plays)))
    rm = Room()
    rm.mixer.register(PROBE_NAME, synth_probe(SR))
    check("a sample named with a leading underscore is registered but not offered "
          "as a gunshot to choose",
          PROBE_NAME in rm.mixer._samples and PROBE_NAME not in rm.mixer.names(),
          str(rm.mixer.names()))
    # The flat mute is recorded, volume-scaled, and ONE definition.
    r = Room(gain=0.5)
    t_play = r.clk
    fb = fallback_mute_s(r.mixer, "shot", TAIL_S)
    b = r.bang()
    check("coupling unknown and the bang can reach the bar: muted, for the fallback length",
          b.mode == OWN_MUTED and abs((r.lis._muted_until - t_play) - fb) < 1e-6,
          f"{b.mode}, {r.lis._muted_until - t_play:.3f}s against {fb:.3f}s")
    check("and the mute is on record for audio_at",
          len(r.lis._mutes) == 1 and abs(r.lis._mutes[-1][0] - t_play) < 1e-9
          and abs(r.lis._mutes[-1][1] - (t_play + fb)) < 1e-6, str(list(r.lis._mutes)))
    lo = Room(gain=0.05)
    hi = Room(gain=1.0)
    f_lo = fallback_mute_s(lo.mixer, "shot", TAIL_S)
    f_hi = fallback_mute_s(hi.mixer, "shot", TAIL_S)
    check("the fallback follows the volume the app plays at: louder, longer",
          f_hi > f_lo + 0.3, f"{f_lo:.2f}s at 0.05 -> {f_hi:.2f}s at 1.0")
    # An EFFECT is registered (so it is muted or discounted like a bang) and
    # never taught from.
    r = Room(gain=0.5)
    modes = []
    invented = 0
    for _ in range(3):
        modes.append(r.effect().mode)
        invented += r.run(2.0)
    rep = r.lis.coupling_report()
    check("an effect is registered with the listener like any bang",
          rep["bangs"] == 3 and set(modes) == {OWN_MUTED}, f"{rep['bangs']} {modes}")
    check("and the microphone does not hear it as a trigger pull", invented == 0,
          f"{invented} invented")
    check("but it never teaches the coupling and is not a refusal either",
          rep["filed"] == 0 and rep["refused"] == 0 and r.lis.coupling is None
          and not any(k in ("reading", "refused", "implausible")
                      for k, _ in r.lis.take_notes()), str(rep))


def sec_13(check) -> None:
    print("\n13. no real audio device was opened by this suite")
    check("no OutputStream or InputStream was ever constructed",
          not OPENED, f"{len(OPENED)} construction(s)")
    check("and the tripwire is still the thing standing where a stream would be made",
          all(getattr(m.sd, n, None) is _Tripwire
              for m in (mixer_mod, listener_mod) if m.sd is not None
              for n in ("OutputStream", "InputStream")))


SECTIONS = [("1", sec_1), ("2", sec_2), ("3", sec_3), ("4", sec_4), ("5", sec_5),
            ("5a", sec_5a), ("5b", sec_5b), ("6", sec_6), ("7", sec_7),
            ("7b", sec_7b), ("8", sec_8), ("9", sec_9), ("10", sec_10),
            ("11", sec_11), ("12", sec_12), ("13", sec_13)]


def main() -> int:
    global SHOT
    checks: list[bool] = []
    verbose = bool(os.environ.get("LSA_VERBOSE"))
    only = {s.strip() for s in os.environ.get("LSA_SECTIONS", "").split(",") if s.strip()}

    def check(label: str, cond: bool, detail: str = "") -> None:
        # The detail is printed for a FAILURE, or on request. Most details
        # read "(6 before the fix)" or "was 0.028 against a true 0.8", which
        # is exactly what a failure looks like; on a pass they made every
        # green line read as a red one. LSA_VERBOSE=1 prints the numbers.
        checks.append(bool(cond))
        show = detail and (verbose or not cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              f"{(' -- ' + detail) if show else ''}")

    print(f"(source under test: {Path(bang_mod.__file__).parents[2]})")

    SHOT, why = load_sample()
    check("his 9mm sample decodes (the envelope every number here was measured on)",
          SHOT is not None, why)
    if SHOT is None:
        print(f"\n{checks.count(False)} OF {len(checks)} CHECKS FAILED")
        return 1
    unknown = only - {n for n, _ in SECTIONS}
    check("every section named in LSA_SECTIONS exists", not unknown, str(sorted(unknown)))

    for name, fn in SECTIONS:
        if not only or name in only:
            fn(check)

    print()
    if all(checks):
        print(f"ALL {len(checks)} CHECKS PASSED")
        return 0
    print(f"{checks.count(False)} OF {len(checks)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
