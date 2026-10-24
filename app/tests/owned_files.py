"""Fingerprint the files the USER owns, and fail if a test changed them.

RULES C11, and the reason it needs two defences rather than one:

  * A smoke test left `auto_connect: false` in `settings.local.json`, so
    the app stopped connecting on launch -- a worse bug than the one that
    test was preventing.
  * The exposure persisted itself from the live control on exit, so every
    launch booted two stops darker than the one before, while the cause
    looked like a detector fault.
  * The cold-start harness saved its simulated 800x800 mapping over his
    real 3840x2160 one, taking the five sensitivity regions learned from
    his actual trigger pulls with it.

Every one of those was a WRITER that had been neutered somewhere. Neutering
is a patch, and a patch can be moved, renamed, or routed around -- the third
case got through because the harness wrote through a different path than the
one that had been blocked. So: neuter the writer AND take a fingerprint, and
let the fingerprint be the thing that fails.

    from owned_files import OwnedFiles
    owned = OwnedFiles()
    ...                      # run whatever might write
    owned.check(check_fn)    # fails if a byte moved
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

# Everything here is his, is gitignored, and means nothing on another
# machine -- which is exactly why nothing notices when a test rewrites one.
#
# THESE ARE PATHS RELATIVE TO THE PROJECT ROOT, AND THE `config/` PREFIX IS THE
# POINT. The two JSON files moved into `config/` on 2026-10-18 and these names
# were not updated, so for two days `_digest(ROOT / "calibration.json")`
# answered None, None compared equal to None, and the check that exists to
# prove a test has not destroyed his measured state PASSED WITHOUT LOOKING AT
# ANYTHING. A hollow check is worse than no check, and this one was quoted as
# evidence. `_absent()` below is the structural half of the fix: a named path
# that is not there now FAILS, so the next move cannot go unnoticed.
OWNED = (
    "config/calibration.json",  # a mapping valid for one camera in one place,
                               # plus the sensitivity learned from real shots
    "config/settings.local.json",  # the chosen camera, its measured exposure
                               # and the thresholds measured against this room
    # `resources/` is a sibling of ROOT (app/), not inside it: `path_of()` resolves it,
    # through the asset table's override when a runner has pointed one at a scratch copy.
    "resources/images/bullet.png",   # furniture artwork (the magazine bullet): same protection as targets
    "resources/targets",       # his artwork (12 figures) and the scoring-zone
                               # sidecars derived from it. A DIRECTORY: every
                               # file in it is fingerprinted, so a test that
                               # rewrites one image, or drops a preview beside
                               # it, fails here (plan 02 trap 7).
)


def path_of(rel: str) -> Path:
    """Where an OWNED entry lives. `config/...` is under ROOT (app/); `resources/<name>`
    is the asset folder `<name>`, answered by the asset table so that a scratch copy
    (LSA_<NAME> set by a runner) is fingerprinted rather than his real folder.
    `asset_env` is imported here, not at the top, because it imports OWNED from us."""
    rel = rel.replace("\\", "/")
    head, _, rest = rel.partition("/")
    if head == "resources":
        import asset_env
        name, _, tail = rest.partition("/")
        base = asset_env.res(name)
        return base / tail if tail else base
    return ROOT / rel


def isolate_outputs(tag: str) -> Path:
    """Send this test's log and images somewhere that is not his.

    Must be called BEFORE importing camera_lab, because `LOG_PATH` and
    `RECORDINGS` are module-level constants read at import.

    Two contamination paths, both measured tonight:

      * `test_ui_smoke` constructs a real `CameraLab`, publishes a verdict,
        and wrote NINE `GATE role=live screen=noscreen fps=0.0` lines into
        `camera_lab.log` -- the live log. `tools/status.py` reports the
        newest log holding a GATE line as "the last measurement on record",
        so one of those would have been offered as his hardware. The role
        tag invented to stop exactly that was generating it.
      * the cold-start harness overwrote `recordings/verdict_screen.png`
        and `verdict_camera.png` with pictures from a simulated 800x800
        camera, destroying the evidence from his last real run. The log had
        been separated by process; the images had not, because the fix was
        applied to the instance in front of me and not to the rule
        (RULES Part E).

    A file that is still there and still looks like evidence is worse than
    a missing one.
    """
    import os

    out = ROOT / "tests" / "_out" / tag
    (out / "recordings").mkdir(parents=True, exist_ok=True)
    os.environ["LSA_LOG"] = str(out / "camera_lab.log")
    os.environ["LSA_RECORDINGS"] = str(out / "recordings")
    # AND HIS CALIBRATION. The harness saved a simulated 800x800 mapping
    # over his real 3840x2160 one through a path that had not been
    # blocked -- neutering `Mapping.save` was the patch, and the patch
    # was routed around. Redirecting the path means a test cannot reach
    # his file at all, which is a defence rather than a reminder.
    os.environ["LSA_CALIBRATION"] = str(out / "calibration.json")
    # AND HIS SETTINGS -- the one that had no redirect. A test written without
    # the hand-copied "neuter the saver" lines wrote `settings.local.json`
    # (2026-10-16), because that path was hard-coded. It is redirectable now.
    #
    # Redirected to a COPY, never to an empty path: RULES C15 and the
    # cold-start harness require it to read his real `calibrate_seconds` and
    # `settle_seconds`, since other timers race against them, and a harness
    # that boots from defaults is a different harness. The copy is refreshed
    # on every call, so a stale one from an earlier run cannot stand in for
    # what is on his disk now; if he has no file there is none to copy, and
    # a leftover is removed rather than served.
    import shutil
    # His real settings, from where they actually live. This read the project
    # root until 2026-10-20 and the file has been in `config/` since 10-18, so
    # it copied NOTHING and every harness launch since ran on the code defaults
    # instead of his measured exposure and his 3.0s calibrate_seconds. A
    # harness that races against shortened timing manufactures a pass
    # (RULES C15).
    real = ROOT / "config" / "settings.local.json"
    iso = out / "settings.local.json"
    if real.exists():
        shutil.copyfile(real, iso)
        # HIS TIMING AND HIS THRESHOLDS, BUT NOT THE MODE HE LAST CHOSE.
        #
        # The point of copying is C15: the harness must race against his real
        # `calibrate_seconds` and his measured exposure, not against defaults
        # that make the timing easy. But the same file also records UI choices,
        # and a test that inherits those is pinned to the state of the world.
        #
        # On 2026-10-20 a phantom detection at peak 10 landed on the Two
        # Players tile, the app saved `players: 2`, and two checks in
        # test_ui_smoke started failing -- a corner target and the pop-up
        # drill -- because the suite was suddenly running split-screen. The
        # app was right, the checks were right, and the only wrong thing was
        # that a test could see which mode he happened to leave the app in.
        try:
            import json
            cfg = json.loads(iso.read_text(encoding="utf-8"))
            for key, neutral in (("players", 1),):
                if key in cfg:
                    cfg[key] = neutral
            iso.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        except Exception:
            # A settings file we cannot parse is not worth failing a run over;
            # the copy above already gave the harness his timing.
            pass
    elif iso.exists():
        iso.unlink()
    os.environ["LSA_SETTINGS"] = str(iso)
    _install_silent_audio()
    return out


def _install_silent_audio() -> None:
    """No test that builds a CameraLab may open a sound device.

    `CameraLab.__init__` builds a real Mixer, which opens an output stream on
    the system default endpoint -- on this machine the DELL monitor with his
    speakers in it. Asked for directly on the night of 2026-10-16: nothing
    that runs while he sleeps may be able to play audio.

    `CameraLab.audio_factory` is bound to `default_mixer` when the class is
    defined, so the replacement is installed on the mixer module BEFORE
    camera_lab is imported (this function is called before that, like the
    path redirects above), and on the class too if it is already loaded.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tests"))
    import lsa.audio.mixer as mixer_mod
    import lsa.audio.listener as listener_mod
    from silent_audio import silent_listener, silent_mixer, silent_pick_device
    mixer_mod.default_mixer = silent_mixer
    # The microphone, by the same route: a listener that opens no input device,
    # and a device choice that does not depend on which microphones HIS machine
    # lists. `pick_device` itself is not changed, only substituted here.
    listener_mod.default_listener = silent_listener
    listener_mod.pick_device = silent_pick_device
    cl = sys.modules.get("lsa.tools.camera_lab")
    if cl is not None:
        cl.CameraLab.audio_factory = staticmethod(silent_mixer)
        cl.CameraLab.listener_factory = staticmethod(silent_listener)
        cl.pick_device = silent_pick_device


def assert_silent_mixer(lab) -> None:
    """The mixer this lab ACTUALLY holds is the silent one.

    `isolate_outputs()` installs it, and `test_silent_audio.py` proves the
    install with its own tripwire -- but a tripwire in one file says nothing
    about the lab built in another. If the install silently became a no-op
    for the harness, test_silent_audio would still pass while the harness
    opened a real stream on his monitor speakers with nothing saying so.
    Assert the effect on the object in hand (RULES C1), not the attempt.
    """
    m = lab.mixer
    assert type(m).__name__ == "SilentMixer" and m._stream is None, (
        f"this test's CameraLab holds {type(m).__name__} with stream "
        f"{m._stream!r}: a real mixer would play out of his monitor")


def assert_silent_listener(lab) -> None:
    """The listener this lab ACTUALLY holds opens no input device.

    The effect on the object in hand (RULES C1), as `assert_silent_mixer` does
    for output: `isolate_outputs()` installs it and `test_silent_audio.py`
    proves the install with a tripwire, but neither says anything about the lab
    built in another file.
    """
    lis = lab.listener
    assert type(lis).__name__ == "SilentListener", (
        f"this test's CameraLab holds {type(lis).__name__}: a real listener opens "
        f"his microphone, so the run depends on his room")
    assert lis._stream is None or type(lis._stream).__name__ == "_NoStream", (
        f"the listener holds a real stream: {lis._stream!r}")


def _digest(path: Path) -> str | None:
    try:
        if path.is_dir():
            h = hashlib.sha256()
            for f in sorted(p for p in path.rglob("*") if p.is_file()):
                h.update(str(f.relative_to(path)).replace("\\", "/").encode())
                h.update(hashlib.sha256(f.read_bytes()).digest())
            return h.hexdigest()
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None
    except OSError as exc:                                 # pragma: no cover
        return f"unreadable:{exc!r}"


def _members(path: Path) -> dict[str, str]:
    """Per-file digests of a directory, so a change can be NAMED. A single
    digest says only that something moved; on 2026-10-17 it reported
    'targets was MODIFIED by this test' while the user was saving new images
    into the folder minute by minute, and the message could not tell a test
    writing there from a file arriving."""
    out: dict[str, str] = {}
    if path.is_dir():
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            key = str(f.relative_to(path)).replace("\\", "/")
            try:
                out[key] = hashlib.sha256(f.read_bytes()).hexdigest()
            except OSError:
                out[key] = "unreadable"
    return out


def _what_moved(before: dict[str, str], now: dict[str, str]) -> str:
    added = sorted(set(now) - set(before))
    gone = sorted(set(before) - set(now))
    edited = sorted(k for k in set(before) & set(now) if before[k] != now[k])
    parts = []
    for word, names in (("ADDED", added), ("REMOVED", gone), ("CHANGED", edited)):
        if names:
            parts.append(f"{word}: " + ", ".join(names[:6]) + (" ..." if len(names) > 6 else ""))
    return "; ".join(parts) or "contents differ"


class OwnedFiles:
    """A before-and-after fingerprint of the files the user owns."""

    def __init__(self, names: tuple[str, ...] = OWNED) -> None:
        self.names = names
        self.before = {n: _digest(path_of(n)) for n in names}
        self._members_before = {n: _members(path_of(n)) for n in names
                                if path_of(n).is_dir()}
        # A path that is not there digests to None, and None equals None, so
        # every later comparison agrees and the check passes having examined
        # nothing. That is exactly how this module protected nothing for two
        # days after the files moved into `config/`. Record the absences at
        # construction so `check()` can refuse rather than agree with itself.
        self.absent = tuple(n for n in names if not path_of(n).exists())

    def changed(self) -> list[str]:
        """Which of them differ from when this object was made.

        An absence is reported HERE, not only in `check()`, because
        `test_introspect` and `test_coldstart` call this method directly and
        would otherwise bypass the guard -- which is the same mistake one level
        up from the one being fixed. A name that was not there when this object
        was made means nothing was watched, and that must never read as "his
        files are unchanged".
        """
        out = [f"{n} was NOT FOUND, so nothing was watched for it -- the path "
               f"is probably stale" for n in self.absent]
        for name in self.names:
            now = _digest(path_of(name))
            was = self.before[name]
            if now != was:
                if was is None:
                    out.append(f"{name} was CREATED by this test")
                elif now is None:
                    out.append(f"{name} was DELETED by this test")
                elif name in self._members_before:
                    out.append(f"{name} was MODIFIED ("
                               + _what_moved(self._members_before[name], _members(path_of(name)))
                               + "). Files ADDED by the user mid-run are an arrival, not a "
                               "test writing; art or zones CHANGED is a test writing (C11)")
                else:
                    out.append(f"{name} was MODIFIED by this test")
        return out

    def check(self, check: Callable[..., None]) -> None:
        """Report through the caller's own check function."""
        # Before asking whether anything changed, prove there was something to
        # watch. Either a name here is stale -- which is what happened on
        # 2026-10-18 and went unnoticed because absent compares equal to absent
        # -- or the file genuinely is not on this machine. Both need a person,
        # and neither may be reported as "his files are unchanged".
        check(not self.absent,
              "every file this module claims to protect is actually there",
              f"NOT FOUND: {list(self.absent)} -- so the fingerprint of each "
              f"is None and this check would agree with itself whatever a "
              f"test did. Either the path is stale (the usual cause: a file "
              f"moved and this list did not) or the file is genuinely absent "
              f"on this machine. Fix the path; do not delete the name.")
        bad = self.changed()
        check(not bad,
              "the files the user owns are byte-for-byte unchanged",
              f"{bad} -- calibration.json carries a mapping measured on his "
              f"hardware and the sensitivity learned from his own trigger "
              f"pulls; settings.local.json carries the exposure and "
              f"thresholds measured against his room. A test has destroyed "
              f"both before (RULES C11).")
