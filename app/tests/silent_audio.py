"""A mixer that cannot open a sound device, and remembers what it was asked.

WHY: `CameraLab.__init__` builds a real `Mixer`, which opens an output stream
on the system default endpoint -- on this machine the DELL monitor, with his
speakers in it. Every test that constructs a `CameraLab` (the cold-start
harness, the UI smoke test) therefore opened a real stream, and any gunshot
the app decided to play came out of his monitor. Asked for directly on the
night of 2026-10-16: no sounds while he sleeps, and structurally so.

It subclasses the real `Mixer`, so registration, sample loading, durations and
the whole API are the real code. Only `start()` differs: it never imports or
touches a stream, and `play()` records instead of mixing. `_stream` stays None
for ever, which is what the tests assert on -- the absence of a stream object,
not silence nobody can hear.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa.audio.listener import ClickListener                # noqa: E402
from lsa.audio.mixer import Mixer, synth_shot, synth_tick   # noqa: E402


class SilentMixer(Mixer):
    def __init__(self, *a, available: bool = True, **k) -> None:
        super().__init__(*a, **k)
        # Which of the real `start()`'s two outcomes this mixer reproduces.
        # True: the stream opened. False: it raised, so `error` is set, no
        # stream exists, `available` stays False and start() returns False --
        # which is the state every "audio unavailable" branch in the app is
        # written for and which, with this hard-coded True, no simulation
        # could ever reach.
        self._start_ok = bool(available)
        # (sample name, gain) for every play() the app asked for.
        self.played: list[tuple[str, float]] = []

    def start(self) -> bool:
        if not self._start_ok:
            self.error = "SILENT TEST MIXER: simulated start failure (no device opened)"
            return False
        self.device_label = "SILENT TEST MIXER (no device opened)"
        self.available = True        # so the app's play path runs and is recorded
        self.epoch += 1              # as the real start() does on every open
        return True

    @property
    def audible(self) -> bool:
        """The real property asks the stream whether it is alive. This mixer has
        no stream (that is the point), so it answers from the one thing it has:
        whether its simulated start succeeded. Without this the app's play path
        (`audio/bang.py`) would report every harness bang as 'did not reach a
        speaker', which is a fault the real path has only when an endpoint dies."""
        return bool(self.available)

    def play(self, name: str, gain: float = 1.0) -> bool:
        if name not in self._samples or not self.available:
            return False
        self.played.append((name, float(gain)))
        return True


def silent_mixer(samplerate: int = 48000, sounds_dir: Path | None = None,
                 master_gain: float | None = None,
                 device_name: str = "", available: bool = True) -> SilentMixer:
    """Same signature and registrations as `default_mixer`, plus `available`:
    pass False to get a mixer whose start() fails the way a real one does."""
    m = SilentMixer(samplerate=samplerate, master_gain=master_gain,
                    device_name=device_name, available=available)
    m.register("synth shot", synth_shot(samplerate))
    m.register("synth tick", synth_tick(samplerate))
    if sounds_dir is not None:
        m.load_folder(sounds_dir)
    m.start()
    return m


class _NoStream:
    """Stands where a PortAudio input stream would. Opens nothing."""

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class SilentListener(ClickListener):
    """A listener that opens no input device and hears nothing, ever.

    The real `ClickListener` logic is untouched (thresholds, the self-mute, the
    learning, `audio_at`); only `start()` and `stop()` differ. `start()` reports
    success with a sentinel in `_stream`, so the app's "microphone running"
    paths run (SHOT lines carry `mic_peak`, `_poll_audio` consults it) and they
    run the same way every time: no audio block ever arrives, so nothing is ever
    heard, whatever the room sounds like. `_stream` is never a real stream,
    which is what the tests assert on.
    """

    def start(self) -> bool:
        self.error = ""
        self._stream = _NoStream()
        return True

    def stop(self) -> None:
        self._stream = None


def silent_listener(**kwargs) -> SilentListener:
    """Same signature as `default_listener`."""
    return SilentListener(**kwargs)


def silent_pick_device(preferred_name: str = "", camera_name: str = ""):
    """Stands in for `pick_device`: the harness must not depend on which
    microphones HIS machine happens to list, any more than on his room."""
    return 0, "SILENT TEST MICROPHONE (no device opened)"
