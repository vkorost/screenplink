r"""Read a string of shots out of the log the app already writes.

    .venv\Scripts\python.exe tools\read_string.py                # last launch in camera_lab.log
    .venv\Scripts\python.exe tools\read_string.py path\to.log    # another log
    .venv\Scripts\python.exe tools\read_string.py --all          # every launch in the file

One row per detection (SHOT), and one per pulse the microphone heard that the
camera did not (HEARD), in the order they happened:

    heard      what the microphone says about THIS pull: yes / NO / DEAF / no-audio
    seen       whether the camera detected it (a HEARD row is "NO")
    drawn at   the screen position the hole went to, and whether it scored
    target     the distance to the nearest target, in mm, in the target's own shape
    frame      which frame of the pulse the reported position came from
    drift      first-frame vs brightest-frame position, in camera px (and mm)
    margin     brightest frame over the detection bar
    note       a fragment of an earlier pull, a phantom, a faint dot, an off-panel dot ...

and the heard/seen totals the app itself kept.

This READS. It opens the log read-only, writes nothing, and uses no new log
format: every pattern below is a field the app already prints (see
`_on_shot`, `_poll_audio` and `lsa/shotlog.py` in camera_lab). A line that does
not match is skipped, and a field that is absent (an older log, a mode with no
distance) is shown as "-" rather than guessed.

Built because the alternative was reading 25 lines of ~380 characters by eye
after the one string that exists to answer "of N trigger pulls, how many became
a hole in the right place".
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A fragmented pulse: two detections this close in time and place are one
# trigger pull seen twice. The tracker's own refractory window is 250 ms and
# 60 camera px; a fragment is reported when the second falls inside both, with
# a little room in time for a pulse that flickers through the PWM gaps.
FRAGMENT_S = 0.60
FRAGMENT_PX = 60.0

_STAMP = re.compile(r"^(\d\d:\d\d:\d\d)  (.*)$")
_HEADER = re.compile(r"^=== .*camera lab started ===")
_SHOT = re.compile(
    r"^SHOT\s+(?P<idx>\d+)\s+cam=\(\s*(?P<cx>-?[\d.]+),\s*(?P<cy>-?[\d.]+)\)\s+"
    r"(?:screen=\(\s*(?P<sx>-?[\d.]+),\s*(?P<sy>-?[\d.]+)\)|screen=\(uncalibrated\))\s+"
    r"frames=(?P<frames>\d+)\s+pulse~\s*(?P<pulse>[\d.]+)ms\s+peak=(?P<peak>\d+)\s+"
    r"area=(?P<area>[\d.]+)px(?P<rest>.*)$")
_HIT = re.compile(r"HIT (?P<id>\S+) (?P<pts>[+-]\d+)")
_NEAR_HIT = re.compile(
    r"\(d=(?P<px>[\d.]+)px(?: = (?P<mm>[\d.]+)mm)? from centre of (?P<id>[^,]+), "
    r"r=(?P<r>[\d.]+)px\)")
_NEAR_MISS = re.compile(
    r"\(nearest (?P<id>[^:]+): (?P<px>[\d.]+)px(?: = (?P<mm>[\d.]+)mm)? "
    r"(?P<side>outside|inside), r=(?P<r>[\d.]+)px\)")
_T = re.compile(r"(?:^|\s)t=(?P<t>-?[\d.]+)")
_MIC = re.compile(r"mic_peak=(?P<lvl>[\d.]+) \(bar (?P<bar>[\d.]+)(?P<learn>, mic not calibrated yet)?\)")
_FRAME = re.compile(r"centroid from frame (?P<k>\d+|\?)/(?P<n>\d+)")
_DRIFT = re.compile(r"drift first-vs-brightest (?P<px>[\d.]+) cam px(?: \(~(?P<mm>[\d.]+)mm\))?")
_MARGIN = re.compile(r"best/bar (?P<best>\d+)/(?P<bar>[\d.]+) = (?P<x>[\d.]+)x")
_TALLY = re.compile(
    r"\[audio so far: heard (?P<heard>\d+), seen (?P<seen>\d+), both (?P<both>\d+), "
    r"heard-not-seen (?P<hns>\d+), seen-not-heard (?P<snh>\d+)\]")


@dataclass
class Pull:
    """One row: a detection, or a pulse heard and not seen."""
    kind: str                       # "shot" | "heard"
    clock: str                      # HH:MM:SS from the log
    index: int | None = None
    t: float | None = None          # the app's own clock (first frame / click)
    cam: tuple[float, float] | None = None
    screen: tuple[float, float] | None = None
    frames: int | None = None
    peak: int | None = None
    outcome: str = "-"              # "HIT centre +9" | "miss" | "-"
    target: str = "-"               # distance to the nearest target
    frame_used: str = "-"
    drift: str = "-"
    margin: str = "-"
    heard: str = "-"                # yes (0.285) | NO (0.004) | DEAF | no-audio | learning | off
    seen: str = "yes"
    notes: list[str] = field(default_factory=list)
    mic_level: float | None = None
    mic_bar: float | None = None
    deaf: bool = False
    no_audio: bool = False
    learning: bool = False
    mic_on: bool = True
    saw_flag: bool = False          # the app itself reported seen-not-heard


@dataclass
class Report:
    pulls: list[Pull] = field(default_factory=list)
    tally: dict[str, int] | None = None
    launch: str = ""


def _heard_state(p: Pull) -> str:
    if not p.mic_on:
        return "mic off"
    if p.no_audio:
        return "no-audio"
    if p.mic_level is None:
        return "-"
    lvl = f"{p.mic_level:.3f}"
    if p.deaf:
        return f"DEAF ({lvl})"
    if p.learning:
        return f"learning ({lvl})"
    if p.mic_bar is not None and p.mic_level >= p.mic_bar:
        return f"yes ({lvl})"
    return f"NO ({lvl})"


def _branch_of(lead: str) -> str:
    """The heard-but-not-seen branch the app's own lead sentence names."""
    table = (
        ("too faint to count", "faint dot ON the panel (below the bar)"),
        ("seen OUTSIDE the screen", "dot OUTSIDE the screen mask"),
        ("too few frames", "rejected as too short"),
        ("saw nothing at any level", "camera saw nothing at any level"),
        ("no camera frames were recorded", "camera delivered no frames"),
    )
    for needle, name in table:
        if needle in lead:
            return name
    if "it missed the screen" in lead:
        return "(old wording: unqualified 'missed the screen')"
    return "branch not stated"


def parse(lines: list[str], all_launches: bool = False) -> Report:
    """Turn log lines into rows. Pure: no file access, so it is testable."""
    # Keep only the last launch unless asked for everything.
    start = 0
    if not all_launches:
        for i, ln in enumerate(lines):
            if _HEADER.match(ln):
                start = i
    rep = Report()
    if start < len(lines) and _HEADER.match(lines[start]):
        rep.launch = lines[start].strip("= ").strip()
    shots: list[Pull] = []
    saw_unassigned: list[Pull] = []
    for raw in lines[start:]:
        m = _STAMP.match(raw.rstrip("\n"))
        if not m:
            continue
        clock, body = m.group(1), m.group(2)

        s = _SHOT.match(body)
        if s:
            rest = s.group("rest")
            p = Pull(kind="shot", clock=clock, index=int(s.group("idx")),
                     cam=(float(s.group("cx")), float(s.group("cy"))),
                     frames=int(s.group("frames")), peak=int(s.group("peak")))
            if s.group("sx") is not None:
                p.screen = (float(s.group("sx")), float(s.group("sy")))
            h = _HIT.search(rest)
            if h:
                p.outcome = f"HIT {h.group('id')} {h.group('pts')}"
            elif re.search(r"(?:^|\s)miss(?:\s|$)", rest):
                p.outcome = "miss"
            nh, nm = _NEAR_HIT.search(rest), _NEAR_MISS.search(rest)
            if nh:
                d = (f"{nh.group('mm')}mm" if nh.group("mm") else f"{nh.group('px')}px")
                p.target = f"{d} from centre of {nh.group('id')}"
            elif nm:
                d = (f"{nm.group('mm')}mm" if nm.group("mm") else f"{nm.group('px')}px")
                p.target = f"{d} {nm.group('side')} {nm.group('id')}"
            t = _T.search(rest)
            if t:
                p.t = float(t.group("t"))
            mic = _MIC.search(rest)
            if "mic_peak=off" in rest or "[audio: microphone off]" in rest:
                p.mic_on = False
            if "mic_peak=not consulted" in rest:
                p.mic_on = False
                p.notes.append("not treated as a trigger pull yet (no sound consulted)")
            if mic:
                p.mic_level = float(mic.group("lvl"))
                p.mic_bar = float(mic.group("bar"))
                p.learning = bool(mic.group("learn"))
            if "[DEAF:" in rest:
                p.deaf = True
            if "[NO AUDIO" in rest:
                p.no_audio = True
            fr = _FRAME.search(rest)
            if fr:
                p.frame_used = f"{fr.group('k')}/{fr.group('n')}"
            dr = _DRIFT.search(rest)
            if dr:
                p.drift = (f"{dr.group('px')} cam px"
                           + (f" (~{dr.group('mm')}mm)" if dr.group("mm") else ""))
            mg = _MARGIN.search(rest)
            if mg:
                p.margin = f"{mg.group('x')}x ({mg.group('best')}/{mg.group('bar')})"
            tl = _TALLY.search(rest)
            if tl:
                rep.tally = {k: int(v) for k, v in tl.groupdict().items()}
            p.heard = _heard_state(p)
            if p.screen is None:
                p.notes.append("no calibration: nowhere to draw it")
            rep.pulls.append(p)
            shots.append(p)
            saw_unassigned.append(p)
            continue

        if body.startswith("HEARD a shot with no detection"):
            lead = body.split("detector evidence within 0.25 s of the click:", 1)[-1].strip()
            p = Pull(kind="heard", clock=clock, seen="NO", heard="yes",
                     outcome="no hole", frame_used="-")
            t = _T.search(body)
            if t:
                p.t = float(t.group("t"))
            p.notes.append(_branch_of(lead))
            bar = re.search(r"sub-threshold blob (\d+) at cam \((\d+),(\d+)\) against a bar of (\d+)", body)
            if bar:
                p.notes.append(f"faint blob {bar.group(1)} (bar {bar.group(4)}) at cam "
                               f"({bar.group(2)},{bar.group(3)})")
                p.cam = (float(bar.group(2)), float(bar.group(3)))
            off = re.search(r"off-screen blob (\d+) \(above the blob threshold (\d+)\)", body)
            if off:
                p.notes.append(f"off-screen blob {off.group(1)} (threshold {off.group(2)})")
            tl = _TALLY.search(body)
            if tl:
                rep.tally = {k: int(v) for k, v in tl.groupdict().items()}
            rep.pulls.append(p)
            continue

        if body.startswith("SAW a shot with no gunshot sound"):
            # The corroborator retires its OLDEST pending shot first, so the n-th
            # SAW belongs to the oldest detection not yet accounted for as heard.
            target = None
            for cand in saw_unassigned:
                if cand.heard.startswith("yes") or cand.heard.startswith("learning"):
                    continue
                target = cand
                break
            if target is not None:
                saw_unassigned.remove(target)
                target.saw_flag = True
                if "[DEAF:" in body:
                    target.deaf = True
                    target.heard = _heard_state(target)
                    target.notes.append("seen-not-heard, but the microphone was DEAF "
                                        "(self-muted): not a phantom")
                elif "[NO AUDIO" in body:
                    target.no_audio = True
                    target.heard = _heard_state(target)
                    target.notes.append("seen-not-heard, but the microphone delivered "
                                        "no audio: not a phantom")
                else:
                    target.notes.append("SEEN, NOT HEARD: likely a phantom")
            continue

    # Fragments: one trigger pull seen as several detections.
    last: Pull | None = None
    for p in shots:
        if last is not None and p.t is not None and last.t is not None and p.cam and last.cam:
            dt = p.t - last.t
            dpx = ((p.cam[0] - last.cam[0]) ** 2 + (p.cam[1] - last.cam[1]) ** 2) ** 0.5
            if 0.0 <= dt <= FRAGMENT_S and dpx <= FRAGMENT_PX:
                p.notes.append(f"FRAGMENT of #{last.index}: {dt:.2f}s later, {dpx:.0f} cam px "
                               f"away (one trigger pull seen twice)")
        last = p
    return rep


def render(rep: Report) -> str:
    cols = ("clock", "#", "heard", "seen", "drawn at (screen px)", "outcome", "nearest target",
            "frame", "drift first-vs-brightest", "margin over bar", "note")
    rows = []
    for p in rep.pulls:
        drawn = f"({p.screen[0]:.0f},{p.screen[1]:.0f})" if p.screen else "-"
        rows.append((p.clock, str(p.index) if p.index is not None else "HEARD", p.heard,
                     p.seen, drawn, p.outcome, p.target, p.frame_used, p.drift, p.margin,
                     "; ".join(p.notes) if p.notes else ""))
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
              for i in range(len(cols))]
    widths[-1] = 0                              # the note is last and may be long
    def fmt(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    out = []
    if rep.launch:
        out.append(f"launch: {rep.launch}")
    out.append(fmt(cols))
    out.append(fmt(tuple("-" * min(w, 18) for w in widths)))
    out.extend(fmt(r) for r in rows)
    out.append("")

    shots = [p for p in rep.pulls if p.kind == "shot"]
    heard_only = [p for p in rep.pulls if p.kind == "heard"]
    frags = [p for p in shots if any(n.startswith("FRAGMENT") for n in p.notes)]
    pulls = len(shots) - len(frags) + len(heard_only)
    holes = [p for p in shots if p.screen is not None]
    hits = [p for p in shots if p.outcome.startswith("HIT")]
    out.append(f"detections {len(shots)} ({len(frags)} of them fragments of an earlier pull), "
               f"heard-not-seen {len(heard_only)}  ->  trigger pulls {pulls}")
    saw = [p for p in shots if p.saw_flag]
    excused = [p for p in saw if p.deaf or p.no_audio]
    out.append(f"holes drawn {len(holes)}   scored {len(hits)}   "
               f"seen-not-heard {len(saw)} (of which the microphone was deaf or silent-by-fault "
               f"{len(excused)}, so likely phantoms {len(saw) - len(excused)})")
    if rep.tally:
        t = rep.tally
        out.append(f"the app's own running totals at the last line: heard {t['heard']}, "
                   f"seen {t['seen']}, both {t['both']}, heard-not-seen {t['hns']}, "
                   f"seen-not-heard {t['snh']}")
    else:
        out.append("the app's own heard/seen totals: none in this log (no audio tally printed)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", nargs="?", default=str(ROOT / "logs" / "camera_lab.log"))
    ap.add_argument("--all", action="store_true", help="every launch in the file, not just the last")
    args = ap.parse_args(argv)
    path = Path(args.log)
    if not path.exists():
        print(f"no such log: {path}", file=sys.stderr)
        return 2
    with open(path, "r", encoding="utf-8", errors="replace") as fh:      # read-only
        lines = fh.read().splitlines()
    rep = parse(lines, all_launches=args.all)
    if not rep.pulls:
        print("no SHOT or HEARD lines in " + str(path)
              + ("" if args.all else " after the last launch header (try --all)"))
        return 1
    print(render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
