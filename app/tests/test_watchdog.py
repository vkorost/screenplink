"""The mark watchdog: a bang was heard and nothing appeared.

Pure policy tests on a counter and a list of times, plus structural checks on
the wiring. No Qt, no camera, no screen, no audio. Each section names the
behaviour it pins, and what goes red when that behaviour is reverted.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa import watchdog as W                                   # noqa: E402

CHECKS = 0
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILED.append(what)
        print(f"  FAIL  {what}" + (f"  --  {detail}" if detail else ""))


def section(n: str) -> None:
    print(f"\n=== {n} ===")


def fresh():
    lines: list[str] = []
    return W.MarkWatchdog(log=lines.append), lines


def pull(wd, t, drawn, counted=True):
    """One trigger pull at time t; a mark follows at t+0.1 when `drawn`."""
    wd.note_click(t, counted)
    if drawn:
        wd.note_mark(t + 0.1)
    wd.resolve(t + 1.0)


section("1. the pattern, not the count (his good string: 24 of 213, 11%)")
wd, _ = fresh()
t = 0.0
empties_in_a_row_max = 0
for i in range(213):
    t += 1.0
    pull(wd, t, drawn=(i % 9 != 4))          # ~11% empty, scattered
    empties_in_a_row_max = max(empties_in_a_row_max, wd.run)
check(wd.empty_total >= 20, "the healthy string really has ~11% empties",
      str(wd.empty_total))
check(empties_in_a_row_max < W.RUN_TRIGGER and not wd.tick(t + 5, False),
      "scattered empties never trigger a check, however many there are")

section("2. a run of three in a row triggers; two does not; a mark resets it")
wd, _ = fresh()
pull(wd, 1.0, False); pull(wd, 2.0, False)
check(not wd.tick(10.0, False), "a run of two does not trigger")
pull(wd, 3.0, True)
check(wd.run == 0, "a drawn mark resets the run")
for k in (4.0, 5.0, 6.0):
    pull(wd, k, False)
check(wd.run == 3, "three consecutive empties make a run of three")
check(wd.tick(20.0, False), "a run of three, after a quiet gap, starts the check")

section("3. a mark pairs with ONE click only")
wd, _ = fresh()
wd.note_click(1.0, True); wd.note_click(1.2, True)
wd.note_mark(1.15)
wd.resolve(5.0)
check((wd.ok_total, wd.empty_total) == (1, 1),
      "two bangs and one mark is one drawn and one empty", str((wd.ok_total, wd.empty_total)))

section("4. a mark BEFORE the click does not count for it")
wd, _ = fresh()
wd.note_mark(0.5)                    # the previous shot's late mark
wd.note_click(1.0, True)
wd.resolve(5.0)
check(wd.empty_total == 1, "a mark 0.5 s earlier is not this click's mark")

section("5. uncounted clicks (own drawing, grid, uncalibrated mic) are not held to account")
wd, _ = fresh()
for k in (1.0, 2.0, 3.0, 4.0):
    wd.note_click(k, False)
wd.resolve(10.0)
check(wd.run == 0 and wd.empty_total == 0, "uncounted clicks never build a run")

section("6. never mid-string; abandon after the deferral deadline")
wd, lines = fresh()
for k in (1.0, 2.0, 3.0):
    pull(wd, k, False)
check(not wd.tick(3.5, False), "clicked 0.5 s ago: not quiet, no check")
check(not wd.tick(10.0, True), "the slot or screen is busy: no check")
check(wd.tick(10.5, False), "quiet and free: check")
wd2, lines2 = fresh()
for k in (1.0, 2.0, 3.0):
    pull(wd2, k, False)
wd2.tick(3.5, False)
wd2.note_click(18.0, False)           # still shooting at the deadline
started = wd2.tick(3.5 + W.DEFER_MAX_S + 0.5, False)
check(not started and wd2.run == 0, "no gap in DEFER_MAX_S: abandoned, run dropped")
check(any("ABANDONED" in s for s in lines2), "the abandonment is logged")

section("7. stale runs do not fire minutes later")
wd, _ = fresh()
for k in (1.0, 2.0, 3.0):
    pull(wd, k, False)
check(not wd.tick(3.0 + W.STALE_S + 5, False) and wd.run == 0,
      "a run older than STALE_S is dropped without a check")

section("8. rate limits: healthy 60 s, bad 120 s, latch after two bad, ceiling")
wd, _ = fresh()
for k in (1.0, 2.0, 3.0):
    pull(wd, k, False)
wd.begin_check(10.0)
wd.end_check(11.0, W.Verdict("healthy", "dot reads 48"))
for k in (20.0, 21.0, 22.0):
    pull(wd, k, False)
check(not wd.tick(30.0, False), "inside the 60 s healthy gap: no second check")
wd.run = 0
for k in (60.0, 61.0, 62.0):
    pull(wd, k, False)
check(not wd.tick(65.0, False), "still inside the gap at 54 s after the last check")
check(wd.tick(11.0 + W.GAP_AFTER_HEALTHY_S + 1, False),
      "after the 60 s gap, a fresh run does start a check")
wd, _ = fresh()
bad = W.Verdict("bad", "dot reads 5")
for n in range(2):
    base = 1000.0 * (n + 1)
    for k in (1.0, 2.0, 3.0):
        pull(wd, base + k, False)
    ok = wd.tick(base + 20, False)
    check(ok, f"bad check {n + 1} is allowed")
    wd.begin_check(base + 20)
    wd.end_check(base + 21, bad)
check(wd.latched, "two bad answers in a row latch the watchdog")
for k in (5000.0, 5001.0, 5002.0):
    pull(wd, k, False)
check(not wd.tick(5100.0, False), "latched: no further check however long it runs")
wd.note_mark(6000.0)
check(not wd.latched and wd.bad_streak == 0, "a mark appearing lifts the latch")
wd, _ = fresh()
wd.checks = W.MAX_CHECKS
for k in (1.0, 2.0, 3.0):
    pull(wd, k, False)
check(not wd.tick(30.0, False), "the per-session ceiling stops all checks")

section("9. the verdict reads BOTH standards, either is enough")
v = W.judge(5.0, 5.0)
check(v.kind == "bad" and v.absolute_fired and not v.relative_fired,
      "a poisoned baseline (5 against 5) is NOT a standard: absolute fires",
      v.why)
v = W.judge(0.0, 48.0)
check(v.kind == "bad", "reading 0 against a healthy baseline is bad")
v = W.judge(20.0, 48.0)
check(v.kind == "bad" and v.relative_fired and not v.absolute_fired,
      "mid-session degradation: 20 passes the absolute bar, fails half of 48",
      v.why)
for r in (44, 46, 49, 52, 43, 25, 47, 52, 46):
    check(W.judge(float(r), float(r)).kind == "healthy",
          f"his healthy launch reading {r} judged against itself is healthy")
check(W.judge(30.0, None).kind == "healthy", "no baseline: absolute bar alone")
check(W.judge(None, 48.0).kind == "unmeasurable", "unreadable is not 'bad'")

section("10. structure: the wiring is where the brief says")
lab = (ROOT / "src/lsa/tools/camera_lab.py").read_text(encoding="utf-8")
rout = (ROOT / "src/lsa/routines.py").read_text(encoding="utf-8")
check("self._watch_feed(clicks, now)" in lab, "_poll_audio feeds the watchdog")
check("self.watchdog.note_mark(" in lab, "_on_shot declares the mark at the end of the chain")
check("holes.added > _holes_before" in lab,
      "a hole counts only when the holes layer's own count advanced")
check("watch.LEDGER.count > _ledger_before" in lab,
      "an absorbed shot counts only when the mode recorded a mark")
check('voice.say("v_camera_bad")' in lab, "the spoken hook is named v_camera_bad")
body = rout[rout.index("def mark_watch_check"):]
check("Control(" not in body and "set_control" not in body,
      "the self-check issues no camera control write")
check("consumes_detections=True" in rout[rout.index("def mark_watch_job"):],
      "the self-check's own dot is not a gunshot or a hole")
for f in ("figure.py",):
    check("record_mark()" in (ROOT / "src/lsa/modes" / f).read_text(encoding="utf-8"),
          "Marks.add records into the ledger")
for f in ("advancing.py", "image_target.py", "popup.py"):
    src = (ROOT / "src/lsa/modes" / f).read_text(encoding="utf-8")
    check("record_mark()" in src,
          f"{f}: the drop flash (which replaces Marks.add) records a mark")

section("11. BEHAVIOUR at the drop: the ledger moves in modes 7, 8 and 9, and the fall second")
# The checks above read source text. This DRIVES each figure mode to the drop and
# reads the ledger the camera-lab consults (RULES C1: assert the effect, not the
# attempt). Merge of the watchdog (ledger) with the fall poses (Fall.start).
sys.path.insert(0, str(ROOT / "tests"))
from PySide6.QtWidgets import QApplication                      # noqa: E402
_APP = QApplication.instance() or QApplication([])
from figure_helpers import context, drive_to_live                # noqa: E402
import lsa.modes as modes                                         # noqa: E402
from lsa.modes.advancing import AdvancingTarget                  # noqa: E402
from lsa.modes.image_target import ImageTarget                  # noqa: E402
from lsa.modes.popup import PopUpFigure                         # noqa: E402

CENTRE = (800, 1200)                                              # ring 10 on the artwork
for cls in (ImageTarget, PopUpFigure, AdvancingTarget):
    m = cls()
    m._rng.seed(21)
    m.start(context(now=0.0), figure="thug01")
    if cls is ImageTarget:
        t = 10.0
    else:
        drive_to_live(m, 0.0)
        if cls is PopUpFigure:
            t = m.shown_at + 0.2
        else:
            a0 = m._actors[0]
            t = a0.t0 + a0.t_a + 0.1

    def geom(tt):
        if cls is AdvancingTarget:
            a = m._actors[0]
            c, h = a.state(tt)
            return a.fig, c, h
        return m._fig, m._centre, m._height

    def fire(tt, n=[0]):
        n[0] += 1
        fig, c, h = geom(tt)
        l, top, w, hh = fig.box(c, h)
        return m.shot(modes.ShotEvent(l + CENTRE[0] / 1791.0 * w, top + CENTRE[1] / 2399.0 * hh,
                                      tt, 700 + n[0]), context(now=tt))

    nm = cls.name
    before = W.LEDGER.count
    o1 = fire(t)
    o2 = fire(t + 0.02)
    check(W.LEDGER.count == before + 2 and o1.hit and o2.hit,
          f"{nm}: two ordinary ring hits each record a mark ({W.LEDGER.count - before} of 2)")
    before = W.LEDGER.count
    t_drop = t + 0.04
    o3 = fire(t_drop)
    check(o3.hit and o3.absorbed and W.LEDGER.count == before + 1,
          f"{nm}: the DROPPING shot records exactly one ledger mark (the total flash)",
          f"+{W.LEDGER.count - before}")
    span = (m._actors[0].fall_span if cls is AdvancingTarget else m._fall_span)
    check(span is not None, f"{nm}: thug01 has a fall pose, so the fall started")
    if span is None:
        continue
    check(not m.moving_at(t_drop),
          f"{nm}: moving_at is False at the dropping shot's own instant (it is a real hit)")
    mid = (span[0] + span[1]) / 2
    check(m.moving_at(mid), f"{nm}: moving_at is True across the fall second")
    before = W.LEDGER.count
    fo = m.shot(modes.ShotEvent(1900.0, 1100.0, mid, 799), context(now=mid))   # where the body lies
    check((not fo.hit) and (not fo.absorbed) and fo.target_id == "fallen",
          f"{nm}: a shot in the fall second scores nothing and is NOT absorbed, so the "
          f"screen draws its ordinary hole (which is the mark the watchdog counts)",
          f"hit={fo.hit} absorbed={fo.absorbed} id={fo.target_id}")
    check(W.LEDGER.count == before,
          f"{nm}: ...and the ledger is untouched by it (the hole counts through holes.added)")
    check(m.hits == 3, f"{nm}: the fall-second shot did not change the tally ({m.hits} hits)")

print()
if FAILED:
    print(f"{len(FAILED)} OF {CHECKS} CHECKS FAILED")
    raise SystemExit(1)
print(f"ALL {CHECKS} CHECKS PASSED")
