"""Is the gunshot audible? Answers it in ten seconds, with no camera.

Written 2026-10-16 after a live string in which every shot was detected and
drawn and he heard NOTHING, while the log said:

    14:12:58  audio ready on DELL U4320Q (NVIDIA High Defini  [system
              default], output latency ~21.3 ms

So audio initialised, on the right device -- his speakers are inside that
monitor -- and the gunshots still went nowhere. This project has three
recorded ways for that to happen and they need opposite fixes:

1. The app never asked for a sound (a gate upstream decided the detection
   was not a trigger pull). Diagnosed from the app's log, not here.
2. The stream is dead and does not say so. PortAudio keeps accepting writes
   to a callback stream whose endpoint has gone: `play()` returns True,
   `available` stays True, and nothing comes out. The endpoint here is a
   DISPLAY audio endpoint, and this app flashes that display white and black
   during every calibration, which is exactly the event that renegotiates
   the link.
3. The sample itself is silent or missing.

This tool settles 2 and 3 by playing the real sample through the real mixer
with nothing else running, and printing what the stream says before and
after. The one thing it cannot do is hear, so it asks.

    .venv\\Scripts\\python.exe tools\\audio_check.py

NO CAMERA, NO CONTROL WRITES. A diagnostic that touches the camera to answer
an audio question is how an afternoon went into reading a wedged device as
data (RULES A8).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa.artwork import assets                     # noqa: E402  (where the sounds and the settings live)
from lsa.audio.mixer import default_mixer          # noqa: E402
from lsa.settings import AppSettings               # noqa: E402


def main() -> int:
    settings = AppSettings.load(assets.user_dir("config") / "settings.local.json")
    want = getattr(settings, "shot_sound", "") or "synth shot"
    print("=== is the gunshot audible? ===")
    print(f"  settings say:  sound={getattr(settings, 'sound', None)!r}  "
          f"sample={want!r}  device={getattr(settings, 'sound_device', '')!r}")

    # BUILT EXACTLY AS THE APP BUILDS IT.
    #
    # The first version called `default_mixer()` bare. That registers only
    # the two synthesised fallbacks, so his configured sample -- '9mm pistol
    # shoot short reverb' -- was never loaded, all three `play()` calls
    # returned False, and the answer key below would have told him "the mixer
    # is at fault" while the tool played nothing. A diagnostic that cannot
    # produce the thing it tests is RULES C1, and a stand-in that does not
    # mirror the real construction is C12. Caught in review before he ran it.
    mixer = default_mixer(sounds_dir=assets.asset_dir("sounds"),
                          master_gain=settings.volume,
                          device_name=settings.sound_device)
    print(f"  mixer available: {mixer.available}")
    print(f"  master gain:     {settings.volume!r}  (0.05 is the default; at "
          f"a low gain a working channel is easy to mistake for a dead one)")
    print(f"  device:          {mixer.output_name}")
    names = list(getattr(mixer, "names", lambda: [])())
    print(f"  samples loaded:  {len(names)} -> {names[:6]}")
    if want not in names:
        print(f"  !! the configured sample {want!r} is NOT among them. The "
              f"app would fall back, or play nothing.")

    stream = getattr(mixer, "stream", None)
    print(f"  stream active before: {getattr(stream, 'active', None)}")

    loud = mixer.loud_duration(want)
    # THIS USED TO SAY "the microphone is muted for that plus a tail" as if it
    # were what the app does. It is only the FALLBACK, and a diagnostic that
    # describes a superseded mechanism is how a wrong cause gets read as
    # evidence (RULES E1). Until the app has MEASURED how loud its own gunshot
    # is at the microphone (three clean readings: three short sweeps during the
    # launch self test, or his own shots) the microphone is muted flatly for the
    # sample's loud part plus a 0.25s tail -- unless the bang cannot reach the
    # click bar even at the listener's coupling ceiling, when it stays open.
    # After that nothing is muted: the click bar is raised by the bang's
    # expected level at each instant. This tool plays no probe and listens to
    # nothing; the measured coupling and every refused reading are the
    # "microphone:" lines in the app's log.
    print(f"  the sample is loud for {loud:.2f}s: that, plus a 0.25s tail, is the "
          f"FLAT mute the app falls back on until it has measured how loud its "
          f"own gunshot is at the microphone (see the 'microphone:' lines in "
          f"the log); after that it discounts the bang instead of going deaf "
          f"for it")

    print("\n  playing it three times, one second apart. LISTEN.")
    for i in range(3):
        ok = mixer.play(want)
        print(f"    play {i + 1}: play() returned {ok!r}, "
              f"stream.active={getattr(getattr(mixer, 'stream', None), 'active', None)}")
        time.sleep(1.0)

    # `ensure_alive` returns a LOG LINE when it had to reopen a dead stream
    # and "" when all was well -- so bool() of it is the exact opposite of
    # health, which is a trap this project has already fallen into once.
    note = mixer.ensure_alive()
    print(f"\n  ensure_alive(): {note!r}")
    if note:
        print("  ^^ it found the stream DEAD and reopened it. That is cause 2:"
              " the endpoint went away and PortAudio never said so.")
        print("     Playing three more times on the reopened stream. LISTEN.")
        for i in range(3):
            mixer.play(want)
            time.sleep(1.0)
    else:
        print("  ^^ the stream reports healthy.")

    print("\n  ANSWER THE QUESTION THAT MATTERS: did you hear three bangs?")
    print("    heard them  -> the mixer is fine, so the app never asked for")
    print("                   a sound. Read camera_lab.log for the reason it")
    print("                   now prints when it suppresses one.")
    print("    heard none  -> the mixer is at fault even with nothing else")
    print("                   running, and the endpoint is the first suspect.")
    print("    heard them only after ensure_alive reopened the stream ->")
    print("                   cause 2 confirmed, and the 5s keepalive is not")
    print("                   catching it during a session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
