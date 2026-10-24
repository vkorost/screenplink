"""Where did Windows put the app's sound, and has it been muted or turned down?

READ-ONLY, AND SILENT. This tool plays nothing, opens no audio stream, changes
no volume, no mute, no default device and no registry value. It only reads.

    .venv\\Scripts\\python.exe tools\\audio_meter.py

Plan 02 step 3 (agent_reports/02-fable-plan.md). The symptom: the app logs
"gunshot ... played on <default endpoint>" and nothing is heard. Everything the
app can see is healthy, so the question is what Windows did with the sound
afterwards. This reports, without anybody's ear:

  1. the default render endpoint (console, multimedia, communications): its
     name, its master mute and volume, and whether it is the device the last
     launch said it played on;
  2. every audio session on every active output: process, volume, mute --
     python first. Only LIVE sessions exist, so with the app closed there is
     nothing to see here, and the tool says so rather than saying "fine";
  3. what Windows REMEMBERS per application, read from the registry
     (HKCU\\Software\\Microsoft\\Internet Explorer\\LowRegistry\\Audio\\
     PolicyConfig\\PropertyStore). That store is the only place the
     per-application volume and mute of an app that is NOT running can be seen,
     and it is keyed by the executable's path -- the venv's python.exe is a
     launcher stub, so the audio session belongs to the BASE interpreter that
     the venv's pyvenv.cfg names in `home` (read from that file at run time,
     not assumed: see `venv_interpreters`).

WHAT IS DELIBERATELY NOT HERE. The playback half of plan step 3 (play the real
sample while polling the session and endpoint meters, and the microphone) plays
a gunshot, so it needs somebody present and awake. It is not implemented, and
this tool refuses to grow it by accident: tests/test_feedback.py section 33
fails if this file ever calls a setter, opens a stream or writes the registry.
Nor does it change anything it finds -- no un-muting, no volume change, no
re-picking a device (RULES D7). It reports.

WHAT THE REGISTRY DECODE IS AND IS NOT. Each remembered application has a
property set {219ED5A0-9CBF-4F3A-B927-37C9E5C5F14F} with three values. By
pattern they are: '3' a float (VT_R4), '4' an array of floats, '5' a boolean
(VT_BOOL). The floats range over 0.0 .. 1.0 across this machine's 54 entries
(1.0 for 47 of them), which is what an application volume looks like, so '3' is
read as the application's volume and '4' as its per-channel levels. '5' is the
only boolean, so it is read as the mute flag -- but on this machine NO entry has
ever held TRUE there, so that reading has never been confirmed by a positive
case. Microsoft does not document this store. Everything decoded from it is
labelled (inferred) in the output, and the raw bytes are always printed beside
it, so nobody has to trust the label.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import winreg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STORE = (r"Software\Microsoft\Internet Explorer\LowRegistry\Audio"
         r"\PolicyConfig\PropertyStore")
POLICY_SET = "{219ED5A0-9CBF-4F3A-B927-37C9E5C5F14F}"
PID_VOLUME, PID_CHANNELS, PID_MUTE = "3", "4", "5"

# {2}.<endpoint hardware path>/<flags>|<NT path of the exe>%b{<guid>}
_ENTRY = re.compile(
    r"^\{(?P<n>\d+)\}\.(?P<dev>.*?)/(?P<flags>[0-9a-fA-F]+)\|(?P<exe>.*?)%b\{(?P<guid>[^}]*)\}$")

IID_KEY = "{B3F8FA53-0004-438E-9003-51A46E139BFC} 2"      # the endpoint's hardware id


# ---------------------------------------------------------------------------
# Pure decoding. Importable, and exercised by tests/test_feedback.py section 33
# against bytes read from this machine.
# ---------------------------------------------------------------------------

def parse_entry(text: str) -> "dict | None":
    """Split a remembered-application key into endpoint hardware and exe path."""
    m = _ENTRY.match(text or "")
    if not m:
        return None
    return {"dev": m["dev"], "flags": m["flags"], "exe": m["exe"], "guid": m["guid"]}


def exe_tail(nt_path: str) -> str:
    """`\\Device\\HarddiskVolume3\\Python313\\python.exe` -> `\\Python313\\python.exe`.

    The volume number is not stable and a drive letter is not stored, so
    comparison is on the path below the volume.
    """
    m = re.match(r"^\\Device\\HarddiskVolume\d+(\\.*)$", nt_path or "", re.I)
    return m[1] if m else (nt_path or "")


def hardware_key(dev: str) -> str:
    """The endpoint hardware id out of a registry device path.

    `\\\\?\\hdaudio#func_01&ven_10de&dev_00a7&subsys_103c8c3a&rev_1001#{...}\\topo00`
    -> `HDAUDIO\\FUNC_01&VEN_10DE&DEV_00A7&SUBSYS_103C8C3A&REV_1001`
    """
    s = re.sub(r"^\\\\\?\\", "", dev or "")
    s = s.split("#{", 1)[0]
    return s.replace("#", "\\").upper()


def _vt(b: bytes) -> "int | None":
    return struct.unpack_from("<H", b, 0)[0] if len(b) >= 2 else None


def decode_float(b: bytes) -> "float | None":
    """A serialised VT_R4 (type 4, then padding, then the float at offset 8)."""
    if len(b) >= 12 and _vt(b) == 0x0004:
        return struct.unpack_from("<f", b, 8)[0]
    return None


def decode_bool(b: bytes) -> "bool | None":
    """A serialised VT_BOOL (type 0x0B; VARIANT_TRUE is -1, FALSE is 0)."""
    if len(b) >= 10 and _vt(b) == 0x000B:
        return struct.unpack_from("<h", b, 8)[0] != 0
    return None


def decode_floats(b: bytes) -> "list[float] | None":
    """An array of floats: 16 bytes of header, then the values."""
    if len(b) >= 20 and _vt(b) == 0x2004:
        body = b[16:]
        return list(struct.unpack_from(f"<{len(body) // 4}f", body))
    return None


def decode_policy(values: "dict[str, bytes]") -> dict:
    """The three remembered values, decoded by pattern (inferred, see the top)."""
    return {
        "volume": decode_float(values.get(PID_VOLUME, b"")),
        "channels": decode_floats(values.get(PID_CHANNELS, b"")),
        "mute": decode_bool(values.get(PID_MUTE, b"")),
        "raw": {k: v.hex() for k, v in sorted(values.items())},
    }


def quiet_entries(entries: "list[dict]", below: float = 0.02) -> "list[str]":
    """The exe of every remembered application whose stored volume is under `below`.

    `is not None`, never `or`: a stored volume of 0.0 is falsy, and the first
    draft of this counted the very entries it exists to find as 1.0 and
    reported none.
    """
    out = []
    for e in entries:
        v = decode_float(e["values"].get(PID_VOLUME, b""))
        if v is not None and v < below and e["parsed"]:
            out.append(exe_tail(e["parsed"]["exe"]))
    return out


def _tail_key(path: str) -> str:
    """`C:\\Python313\\python.exe` -> `\\python313\\python.exe` (no drive, lower case),
    the same shape `exe_tail` gives a remembered NT path, so the two compare."""
    return re.sub(r"^[a-z]:", "", (path or "").lower())


def venv_interpreters(prefix: "str | None" = None, base_exe: "str | None" = None) -> dict:
    """Which interpreter does THIS venv launch? Read from pyvenv.cfg, not guessed.

    `is_ours` selects registry entries by the word "python", which says nothing
    about whether an entry is the interpreter this venv runs. pyvenv.cfg does:
    `home` is the directory of the base interpreter and `executable` is the
    base python.exe the venv was made from. Returns

        cfg_home, cfg_executable   what pyvenv.cfg says (None if unreadable)
        cfg_tails                  tail keys derived from those (python/pythonw)
        runtime_tails              tail keys from sys._base_executable, as a
                                   second opinion
        confirmed                  True iff pyvenv.cfg named a home or executable
    """
    cfg = Path(prefix or sys.prefix) / "pyvenv.cfg"
    home = exe = None
    try:
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            k, sep, v = line.partition("=")
            if not sep:
                continue
            key, val = k.strip().lower(), v.strip()
            if key == "home" and val:
                home = val
            elif key == "executable" and val:
                exe = val
    except OSError:
        pass
    cfg_tails: set[str] = set()
    if home:
        for name in ("python.exe", "pythonw.exe"):
            cfg_tails.add(_tail_key(str(Path(home) / name)))
    if exe:
        cfg_tails.add(_tail_key(exe))
    runtime: set[str] = set()
    base = base_exe if base_exe is not None else getattr(sys, "_base_executable", None)
    if base:
        runtime.add(_tail_key(base))
        runtime.add(_tail_key(str(Path(base).with_name("pythonw.exe"))))
    return {"cfg_home": home, "cfg_executable": exe, "cfg_tails": cfg_tails,
            "runtime_tails": runtime, "confirmed": bool(cfg_tails)}


def interpreter_label(tail: str, info: dict) -> str:
    """The note beside a remembered entry that says whether it is OUR interpreter."""
    t = (tail or "").lower()
    if t in info["cfg_tails"]:
        return (f"   <-- the interpreter THIS venv launches (confirmed by pyvenv.cfg: "
                f"home = {info['cfg_home']})")
    if not info["confirmed"] and t in info["runtime_tails"]:
        return ("   <-- PROBABLY the interpreter this venv launches (pyvenv.cfg could not be "
                "read, so this rests on sys._base_executable alone: "
                + ", ".join(sorted(info["runtime_tails"])) + ")")
    if info["confirmed"] and t in info["runtime_tails"]:
        return ("   <-- sys._base_executable says this is ours but pyvenv.cfg does not "
                f"(home = {info['cfg_home']}): the two disagree, so it is NOT confirmed")
    return ""


def rank_findings(findings: "list[str]") -> "list[str]":
    """Most probable cause first. Stable within a rank.

    0  a per-application volume or mute (a live session, or what Windows
       remembers) -- the probable cause of "the app plays and nothing is heard"
    1  the endpoint's own master mute / volume
    2  the monitor-asleep / default-device note, which is CONTEXT: it explains
       why the default may have moved, and does not say the app is silent
    3  anything else
    """
    def rank(f: str) -> int:
        if f.startswith("Windows remembers") or ("live python session" in f):
            return 0
        if "master volume" in f or "MUTED at the endpoint" in f:
            return 1
        if f.startswith("Context"):
            return 2
        return 3
    return sorted(findings, key=rank)


def is_ours(exe: str) -> bool:
    """python, pythonw, the packaged exe, or anything under this project.

    `screenplink` is here because the pattern only knew the OLD project folder
    (`laser.shooting`), so a PACKAGED install -- screenplink.exe, which is what
    a stranger runs -- would not have been recognised as ours. This is the tool
    that found the silent gunshot he reported for three days: python.exe sitting
    at volume 0.0 on the monitor endpoint. Blind to the frozen build, it would
    have found nothing for the one person who cannot read the registry himself.
    """
    return bool(re.search(r"python|laser\.shooting|screenplink", exe or "", re.I))


# ---------------------------------------------------------------------------
# Reading the registry (read-only: KEY_READ, enumerate and query only).
# ---------------------------------------------------------------------------

def read_store() -> "list[dict]":
    entries: list[dict] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, STORE, 0, winreg.KEY_READ)
    except OSError as exc:
        print(f"  could not open the policy store: {exc}")
        return entries
    with root:
        i = 0
        while True:
            try:
                name = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            with winreg.OpenKey(root, name, 0, winreg.KEY_READ) as k:
                try:
                    default = winreg.QueryValueEx(k, "")[0]
                except OSError:
                    default = ""
                parsed = parse_entry(default)
                values: dict[str, bytes] = {}
                try:
                    with winreg.OpenKey(k, POLICY_SET, 0, winreg.KEY_READ) as ps:
                        j = 0
                        while True:
                            try:
                                vn, vd, _t = winreg.EnumValue(ps, j)
                            except OSError:
                                break
                            j += 1
                            if isinstance(vd, (bytes, bytearray)):
                                values[vn] = bytes(vd)
                except OSError:
                    pass                      # no remembered values for this one
            entries.append({"key": name, "text": default, "parsed": parsed,
                            "values": values})
    return entries


# ---------------------------------------------------------------------------
# Core Audio, through pycaw. Getters only.
# ---------------------------------------------------------------------------

def _endpoints(pycaw_utils, consts):
    return pycaw_utils.AudioUtilities.GetAllDevices(
        consts.EDataFlow.eRender.value, consts.DEVICE_STATE.MASK_ALL.value)


def _sessions_of(device):
    """Live sessions on one endpoint, as pycaw AudioSession objects."""
    from pycaw.api.audiopolicy import IAudioSessionControl2
    from pycaw.utils import AudioSession
    out = []
    enum = device.AudioSessionManager.GetSessionEnumerator()
    for i in range(enum.GetCount()):
        ctl = enum.GetSession(i)
        if ctl is None:
            continue
        ctl2 = ctl.QueryInterface(IAudioSessionControl2)
        if ctl2 is not None:
            out.append(AudioSession(ctl2))
    return out


def _last_logged_output() -> "str | None":
    """The device the last launch said it played on, from camera_lab.log."""
    log = ROOT / "logs" / "camera_lab.log"
    if not log.exists():
        return None
    found = None
    try:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"audio ready on (.+?)\s{1,2}\[(system default|chosen)\]", line)
            if m:
                found = m[1].strip()
    except OSError:
        return None
    return found


def _settings() -> dict:
    p = ROOT / "settings.local.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args()

    try:
        import comtypes
        comtypes.CoInitialize()
        from pycaw import utils as pycaw_utils
        from pycaw import constants as consts
    except Exception as exc:                      # no guessing without the instrument
        print(f"pycaw/comtypes could not be imported ({exc!r}). Stopping: this tool "
              f"does not guess. Install with:  .venv\\Scripts\\python.exe -m pip install pycaw")
        return 2

    print("audio_meter: READ-ONLY and SILENT. Plays nothing, opens no stream, "
          "changes nothing.\n")
    findings: list[str] = []

    # -- 1. the default endpoints --------------------------------------------
    print("1. Default output (what the app uses unless the user chose another)")
    enum = pycaw_utils.AudioUtilities.GetDeviceEnumerator()
    default_ids: dict[str, str] = {}
    for role in (consts.ERole.eConsole, consts.ERole.eMultimedia, consts.ERole.eCommunications):
        try:
            dev = enum.GetDefaultAudioEndpoint(consts.EDataFlow.eRender.value, role.value)
            default_ids[role.name] = dev.GetId()
        except Exception as exc:
            default_ids[role.name] = ""
            print(f"   {role.name:15s} none ({exc!r})")
    devices = _endpoints(pycaw_utils, consts)
    by_id = {d.id: d for d in devices}
    default_name = None
    for role, did in default_ids.items():
        d = by_id.get(did)
        if d is None:
            continue
        default_name = default_name or d.FriendlyName
        print(f"   {role:15s} {d.FriendlyName}   [{d.state.name}]")
        if role == "eConsole":
            try:
                ev = d.EndpointVolume
                mute, vol = bool(ev.GetMute()), float(ev.GetMasterVolumeLevelScalar())
                print(f"                   master mute: {mute}   master volume: {vol * 100:.0f}%")
                if mute:
                    findings.append(f"the default output '{d.FriendlyName}' is MUTED at the "
                                    f"endpoint (system volume): report only, never touched")
                elif vol < 0.02:
                    findings.append(f"the default output '{d.FriendlyName}' master volume is "
                                    f"{vol * 100:.0f}%: report only, never touched")
            except Exception as exc:
                print(f"                   could not read mute/volume: {exc!r}")
                findings.append(f"could not read the default output's mute/volume ({exc!r})")

    logged = _last_logged_output()
    s = _settings()
    print(f"   settings.local.json: sound_device={s.get('sound_device', '?')!r} "
          f"(empty = system default), volume={s.get('volume', '?')}")
    if logged:
        same = bool(default_name) and default_name.startswith(logged.rstrip("( "))
        print(f"   last launch's log said it played on: {logged!r}   "
              f"{'= the default right now' if same else '!= the default right now'}")
        named = [d for d in devices if d.FriendlyName.startswith(logged.rstrip("( "))]
        print(f"   outputs with that name: {len(named)}, active right now: "
              f"{sum(1 for d in named if d.state.name == 'Active')} "
              f"(a monitor's audio output is NotPresent while the monitor sleeps)")
        if not same and default_name:
            findings.append(
                f"Context, not the probable cause: the system default output is now "
                f"'{default_name}', but the last launch "
                f"played on '{logged}'. A display's audio endpoint disappears while the "
                f"monitor sleeps and Windows falls back to another device; whether it moves "
                f"back when the monitor wakes is not something this tool can predict. The "
                f"launch line 'audio ready on ...' names the device it actually used.")
    else:
        print("   camera_lab.log has no 'audio ready on' line to compare with.")

    # -- 2. live sessions --------------------------------------------------------
    print("\n2. Live audio sessions on every active output (python first)")
    live_python = 0
    rows = []
    for d in devices:
        if d.state.name != "Active":
            continue
        try:
            for sess in _sessions_of(d):
                proc = sess.Process
                try:
                    exe = proc.exe() if proc else ""
                except Exception:
                    exe = ""
                try:
                    sv = sess.SimpleAudioVolume
                    vol, mute = float(sv.GetMasterVolume()), bool(sv.GetMute())
                except Exception:
                    vol, mute = None, None
                rows.append((0 if is_ours(exe) else 1, d.FriendlyName, sess.ProcessId,
                             proc.name() if proc else "(system sounds)", exe, vol, mute,
                             sess.State))
        except Exception as exc:
            print(f"   {d.FriendlyName}: could not list sessions ({exc!r})")
    for _o, dn, pid, pname, exe, vol, mute, state in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
        tag = "   <-- python" if _o == 0 else ""
        v = "?" if vol is None else f"{vol * 100:.0f}%"
        print(f"   {dn[:34]:34s} pid {pid:<6d} {pname[:22]:22s} volume {v:>4s}  "
              f"muted {mute}  state {state}{tag}")
        if _o == 0:
            live_python += 1
            if mute:
                findings.append(f"a live python session on '{dn}' (pid {pid}) is MUTED in the "
                                f"Windows mixer: report only, never touched")
            elif vol is not None and vol < 0.02:
                findings.append(f"a live python session on '{dn}' (pid {pid}) is at {v}: "
                                f"report only, never touched")
    if live_python == 0:
        print("   no python session is live. With the app closed that is expected, and it "
              "means the LIVE mute state of the app is UNKNOWN here -- not 'fine'.")

    # -- 3. what Windows remembers -------------------------------------------------
    print("\n3. What Windows remembers per application (registry, read-only; decode inferred)")
    entries = read_store()
    ours = [e for e in entries if e["parsed"] and is_ours(e["parsed"]["exe"])]
    print(f"   {len(entries)} remembered entries; {len(ours)} are python/pythonw or under this project.")
    interp = venv_interpreters()
    print(f"   pyvenv.cfg: home = {interp['cfg_home']}   executable = {interp['cfg_executable']}"
          if interp["confirmed"] else
          "   pyvenv.cfg could not be read: which interpreter this venv launches is NOT "
          "confirmed (labels below say 'probably')")
    for e in ours:
        p = e["parsed"]
        dec = decode_policy(e["values"])
        tail = exe_tail(p["exe"])
        cand = sorted({d.FriendlyName for d in devices
                       if hardware_key(p["dev"]) and (d.properties.get(IID_KEY) or "")
                       .upper().replace("{1}.", "").startswith(hardware_key(p["dev"]) + "\\")})
        mine = interpreter_label(tail, interp)
        print(f"   {tail}{mine}")
        print(f"      endpoint hardware: {hardware_key(p['dev'])}  ({p['dev'].rsplit(chr(92), 1)[-1]})")
        if cand:
            print(f"      endpoints on that hardware: {', '.join(cand)}")
        if not e["values"]:
            print("      nothing remembered (no volume or mute stored: Windows defaults apply)")
            continue
        vol, mute = dec["volume"], dec["mute"]
        print(f"      volume (inferred): {'?' if vol is None else f'{vol * 100:.0f}%'}   "
              f"mute (inferred): {mute}   channels (inferred): {dec['channels']}")
        print(f"      raw: " + "  ".join(f"[{k}] {v}" for k, v in dec["raw"].items()))
        if mute or (vol is not None and vol < 0.02):
            findings.append(
                f"Windows remembers '{tail}' as "
                f"{'MUTED' if mute else f'volume {vol * 100:.0f}%'} on {hardware_key(p['dev'])} "
                f"(decoded by pattern, unconfirmed) -- it would follow every launch of that "
                f"interpreter. Report only; the user decides.")
    quiet = quiet_entries(entries)
    others_quiet = len(quiet)
    trues = sum(1 for e in entries if decode_bool(e["values"].get(PID_MUTE, b"")))
    print(f"   across all {len(entries)} entries: {trues} with the (inferred) mute flag TRUE, "
          f"{others_quiet} with volume under 2%"
          + (f" ({', '.join(sorted(set(quiet)))})." if quiet else "."))
    if trues == 0:
        print("   NOTE: no entry on this machine has ever held TRUE in value '5', so that it "
              "is the mute flag has never been confirmed by a positive case.")

    # -- verdict -----------------------------------------------------------------------
    print("\nWhat this found (most probable cause first)")
    if findings:
        for f in rank_findings(findings):
            print("   - " + f)
    else:
        print("   - Nothing muted or turned down that this tool can see.")
    print("   - This does NOT show that the sound is audible. Only the playback half can "
          "(play the sample while reading the session/endpoint meters and the microphone); "
          "it is deliberately NOT implemented here, because it plays a gunshot and needs a "
          "person present.")
    print("   - Nothing was changed. No default device, volume, mute or registry value was "
          "written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
