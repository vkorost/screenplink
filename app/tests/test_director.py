"""The Director: the properties the old flag-and-timer code could not hold.

Every section here is a fault that actually happened on this project, with
its date. The point of each is not that the Director handles it, but that
the situation is UNREPRESENTABLE: there is one run slot, so there is
nowhere for a second routine to stand.

No Qt, no camera, no screen. Runs in milliseconds on a virtual clock, which
is the other half of the argument -- the headless cold-start harness could
previously only jitter real timers and sample a few of the orderings that
exist.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa.director import (                                      # noqa: E402
    Call, Cancelled, Control, Deadline, Director, EffectFailed, Floor, Job,
    Log, ManualClock, Paint, Room, Sleep, Still,
)

CHECKS = 0
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILED.append(what + (f"  --  {detail}" if detail else ""))
        print(f"  FAIL  {what}")
        if detail:
            print(f"        {detail}")


def section(n: str) -> None:
    print(f"\n=== {n} ===")


class Sink:
    """Stands in for the capture thread. Records everything, answers nothing.

    Deliberately does NOT answer by itself: the test decides when a frame
    arrives and in what order, which is the whole point of owning the clock.
    """

    def __init__(self) -> None:
        self.stills: list[tuple[str, int]] = []
        self.floors: list[float | None] = []
        self.rooms: list[float] = []
        self.controls: list[tuple[str, float]] = []
        self.paints: list[int] = []
        self.written: dict[str, float] = {}

    def request_still(self, tag: str, frames: int = 8) -> None:
        self.stills.append((tag, frames))

    def start_noise_calibration(self, seconds: float | None = None) -> None:
        self.floors.append(seconds)

    def start_room_measurement(self, seconds: float = 1.5) -> None:
        self.rooms.append(seconds)

    def set_control(self, key: str, value: float) -> None:
        self.controls.append((key, float(value)))
        self.written[key] = float(value)

    def written_value(self, key: str) -> "float | None":
        """Mirrors `CaptureThread.written_value` -- the ONE control cache.

        The Director asks the camera object what it was last told rather
        than keeping its own copy, because a second cache goes stale the
        moment a preset is written around it, and a stale cache makes the
        next identical write be SKIPPED while the camera sits elsewhere.
        """
        return self.written.get(key)

    def request_paint(self, frames: int = 2) -> None:
        self.paints.append(frames)


def answer_still(d: Director, sink: Sink, frame: object) -> None:
    """Hand back a frame for the OUTSTANDING request, as a camera does.

    The Director gives each Still a unique tag ("probe_lit#3"), so a
    frame can only answer the request it was asked for; the real
    capture echoes back whatever tag it was handed. Tests that
    hardcoded the bare tag were not modelling the round-trip.
    """
    # Presence is asserted BEFORE indexing, so a fault that stops a still being
    # requested fails this check by name instead of killing the suite with an
    # IndexError after its other checks have already gone red (2026-10-23: the
    # same unguarded-index shape that crashed test_ui_smoke's brightness grid).
    check(bool(sink.stills), "a still had been requested before the test answered it")
    if not sink.stills:
        return
    tag = sink.stills[-1][0]
    d.deliver_still(tag, frame)


def start_no_escape(d: Director, job: Job) -> bool:
    """`d.start(job)`, where an exception escaping it FAILS A CHECK BY NAME.

    2026-10-23: `tests/mutate_director.py` entry 14 ("a failing effect inside an
    exception handler escapes the Director") printed ZERO failing lines. The
    property had a section (19), but both it and section 10 called `d.start(...)`
    bare, so the escaping `EffectFailed` killed the suite at the first of them with
    a traceback and no summary -- the mutation judged BROKEN, and the property that
    CLAUDE.md records as having wedged the run slot for a whole session had no
    check able to name it. The exception must not escape, and a test that lets it
    kill the process cannot say so."""
    try:
        started = d.start(job)
    except Exception as exc:                          # the very thing being tested
        check(False, "a failing effect does NOT escape the Director",
              f"{type(exc).__name__}: {exc} -- escaped start(), leaving the slot "
              "held with no timer and nothing logged")
        return False
    check(True, "a failing effect does NOT escape the Director")
    return started


def last(seq, default=None):
    """The final element, or `default` when there is none.

    A check's arguments are evaluated BEFORE the check runs, so `seq[-1] == x`
    on an empty list raises and kills the suite -- after the checks that
    already went red have printed, so a mutation that empties it is judged
    BROKEN instead of caught. This lets the comparison fail by name."""
    return seq[-1] if seq else default


def make(sink: Sink | None = None, log: list[str] | None = None):
    clock = ManualClock()
    lines = log if log is not None else []
    d = Director(clock, sink=sink or Sink(), log=lines.append)
    return clock, d, lines


# ---------------------------------------------------------------------------
section("1. a routine is straight-line code again")
# ---------------------------------------------------------------------------

sink = Sink()
clock, d, log = make(sink)
trace: list[str] = []


def simple():
    trace.append("start")
    yield Control("exposure", -7.0)
    trace.append("exposed")
    yield Sleep(400)
    trace.append("settled")
    frame = yield Still("probe_lit")
    trace.append(f"got {frame}")
    return "finished"


d.start(Job("probe", simple))
check(trace == ["start"], f"stops at the first effect, not before: {trace}")
check(sink.controls == [("exposure", -7.0)], "the control write went out")

clock.advance(60)                      # the inter-write spacing
check(trace == ["start", "exposed"], f"resumes after the write: {trace}")

clock.advance(400)
check(last(trace) == "settled", f"resumes after the sleep: {trace}")
check(last(sink.stills) == ("probe_lit#1", 2),
      f"the still was requested, with a unique tag: {sink.stills}")

answer_still(d, sink, "FRAME")
check(last(trace) == "got FRAME", f"the frame comes back INTO the routine: {trace}")
check(d.active is None, "the slot is free when the routine returns")
check(last(d.history, (None, None))[1] == "done", f"recorded as done: {d.history}")


# ---------------------------------------------------------------------------
section("2. two routines cannot interleave -- there is one slot")
# ---------------------------------------------------------------------------
#
# 2026-10-15 17:29, the live gate. Three routines owned the screen at once:
#
#   17:29:28  measuring noise floor for 3s
#   17:29:28  measuring noise floor for 3s
#   17:29:29  measuring the exposure by photographing the screen's own dot
#
# The floor measured the probe's red dots; the probe measured a screen the
# floor measurement had covered. Five cold launches, identical code, same
# room: 12/12 twice and 0/12 twice.

clock, d, log = make()
order: list[str] = []


def floor_job():
    order.append("floor-begin")
    yield Floor(3.0)
    order.append("floor-end")


def probe_job():
    order.append("probe-begin")
    yield Sleep(100)
    order.append("probe-end")


check(d.start(Job("noise", floor_job)) is True, "the first routine starts")
check(d.start(Job("probe", probe_job)) is False, "the second is REFUSED")
check(d.start(Job("noise", floor_job)) is False, "and so is a second of ITSELF")
check(order == ["floor-begin"], f"the refused routines never ran: {order}")
check(d.refusals == [("probe", "noise"), ("noise", "noise")],
      f"both refusals recorded with who was holding it: {d.refusals}")
check(any("noise has the screen" in line for line in log),
      "the refusal names the holder in the log, not just a verdict")

d.deliver_floor(4.0, 2.0, 97)
check(order == ["floor-begin", "floor-end"], f"the first finished: {order}")
check(d.start(Job("probe", probe_job)) is True, "and the slot is free after")


# ---------------------------------------------------------------------------
section("3. cleanup cannot be skipped -- cancel runs the finally blocks")
# ---------------------------------------------------------------------------
#
# Five incidents, all the same missing cleanup on one exit path: three
# flash covers left up, `_restore_levels` not reached, `_pat_hidden` not
# restored. And `closeEvent` stood routines down by a hand list of eight
# flag names that omitted `_check_active`.

clock, d, log = make()
state = {"cover": False, "levels": "play", "furniture": True}


def covers_the_screen():
    state["cover"] = True
    state["levels"] = "relaxed"
    state["furniture"] = False
    try:
        yield Still("pat_lit")
        yield Sleep(10000)
    finally:
        state["cover"] = False
        state["levels"] = "play"
        state["furniture"] = True


d.start(Job("pattern", covers_the_screen, draws=True, measures_screen=True))
check(state["cover"] is True, "the cover went up")
check(d.measuring_screen() is True, "furniture is told to stand down (B2)")
check(d.is_drawing() is True, "the app knows it is drawing")

d.cancel("shutting down")
check(state == {"cover": False, "levels": "play", "furniture": True},
      f"EVERY cleanup ran on cancel: {state}")
check(d.active is None, "the slot is free")
check(d.is_drawing() is False, "and the app knows it is no longer drawing")
check(d.measuring_screen() is False, "furniture may come back")


# a routine that never yields again after cleanup still terminates
clock, d, log = make()
done = {"cleanup": 0}


def stubborn():
    try:
        yield Sleep(1000)
    finally:
        done["cleanup"] += 1


d.start(Job("stubborn", stubborn))
d.cancel()
check(done["cleanup"] == 1, "cleanup ran exactly once")
check(d.cancel() is False, "cancelling an idle Director is a no-op, not an error")


# A routine is TOLD it is being stood down, not merely abandoned.
#
# This check exists because the mutation test caught the first version of
# section 3 being decorative: replacing `gen.throw(Cancelled)` with nothing
# left the suite GREEN, because the `gen.close()` that follows raises
# GeneratorExit and runs the finally blocks anyway. Cleanup was never the
# distinguishing property. Being able to REPORT is: a routine that gets
# GeneratorExit cannot say how far it had got, and "one of the two markers
# was not visible", with nothing named, is a check that can only be re-run
# rather than debugged (RULES C3).
clock, d, log = make()
told: list[str] = []


def reports_cancellation():
    try:
        yield Sleep(5000)
    except Cancelled as why:
        told.append(f"stood down: {why}")
        raise
    finally:
        told.append("cleaned up")


d.start(Job("pattern", reports_cancellation))
d.cancel("the camera stalled")
check(told == ["stood down: the camera stalled", "cleaned up"],
      f"the routine was told WHY, and then cleaned up: {told}")


# ---------------------------------------------------------------------------
section("4. no wait can be unbounded")
# ---------------------------------------------------------------------------
#
# RULES B6/D4. A fullscreen cover whose removal depended on a camera frame
# that never came left the display an immovable black rectangle -- the only
# escape was unplugging the camera. And he had to press Escape to get an
# answer out of a wait loop with no deadline.

clock, d, log = make()
got: list[object] = []
cleaned = {"n": 0}


def waits_forever():
    try:
        frame = yield Still("pat_lit", timeout_ms=5000)
        got.append(frame)
    finally:
        cleaned["n"] += 1


d.start(Job("pattern", waits_forever))
clock.advance(4999)
check(got == [], "still waiting just before the deadline")
clock.advance(2)
check(got == [None], f"resumed with None at the deadline: {got}")
check(cleaned["n"] == 1, "and the cover came down")
check(any("did not arrive within" in line for line in log),
      "the stall is reported, with the wait named")

# A routine cannot opt out: the Director arms the timer, not the caller.
clock, d, log = make()
seen: list[object] = []


def asks_for_forever():
    seen.append((yield Still("x", timeout_ms=10 ** 9)))


d.start(Job("greedy", asks_for_forever))
ms = clock.run_until_idle(limit_ms=10 ** 9 + 1000)
check(seen == [None], "even an absurd timeout terminates")
check(ms > 0, "and it was the clock that ended it")


# ---------------------------------------------------------------------------
section("5. only one still outstanding; a stale frame is dropped by identity")
# ---------------------------------------------------------------------------
#
# `_request_still` held one tag in a string and dispatched by prefix. A
# second requester overwrote `_still_tag`, so the first routine's timeout
# silently never fired. And a frame belonging to a cancelled run used to be
# indistinguishable from a live one.

sink5 = Sink()
clock, d, log = make(sink5)
frames: list[object] = []


def one_still():
    frames.append((yield Still("pat_lit")))


d.start(Job("pattern", one_still))
d.deliver_still("probe_lit", "WRONG")        # a different routine's tag
check(frames == [], "a frame with the wrong tag is NOT handed over")
check(len(d.overlaps) == 1, f"and it is recorded as an overlap: {d.overlaps}")
answer_still(d, sink5, "RIGHT")
check(frames == ["RIGHT"], f"the matching frame is: {frames}")


# SAME TAG, DIFFERENT REQUEST, WITHIN ONE RUN.
#
# Matching on the routine's tag alone is enough only across runs. A routine
# that asks for `Still("probe_lit")`, has it time out, moves on and asks for
# `Still("probe_lit")` again would accept the FIRST frame as the answer to
# the SECOND question -- and `overlaps` stayed empty, so the oracle could
# not see it. That is `_request_still`'s bare string tag reproduced inside
# the class written to replace it. Found by an outside review of the diff,
# not by any of the checks above.
sink5b = Sink()
clock, d, log = make(sink5b)
got5: list[object] = []


def asks_twice():
    got5.append((yield Still("probe_lit", timeout_ms=1000)))
    got5.append((yield Still("probe_lit", timeout_ms=1000)))


d.start(Job("probe", asks_twice))
check(len(sink5b.stills) == 1, f"the first request went out: {sink5b.stills}")
first_tag = last(sink5b.stills, ("<none>",))[0]
clock.advance(1001)                       # the first request times out
check(got5 == [None], f"it gives up on the first: {got5}")
check(len(sink5b.stills) == 2, f"and the second went out: {sink5b.stills}")
second_tag = last(sink5b.stills, ("<none>",))[0]
check(first_tag != second_tag,
      f"the second request has its own tag: {first_tag} vs {second_tag}")
d.deliver_still(first_tag, "THE OLD FRAME")   # ...and the first one arrives
check(got5 == [None],
      f"the late frame does NOT answer the second question: {got5}")
check(any("does not match" in o or "stale" in o for o in d.overlaps),
      f"and the oracle sees it: {d.overlaps}")
d.deliver_still(second_tag, "THE RIGHT FRAME")
check(got5 == [None, "THE RIGHT FRAME"], f"the real answer lands: {got5}")

# a frame arriving after the routine was stood down
clock, d, log = make()
late: list[object] = []


def cancelled_mid_still():
    try:
        late.append((yield Still("pat_lit")))
    finally:
        pass


d.start(Job("pattern", cancelled_mid_still))
d.cancel()
d.deliver_still("pat_lit", "TOO LATE")
check(late == [], "a frame for a stood-down routine is dropped")
check(d.active is None, "and does not revive it")


# A measurement that arrives AFTER its own deadline, while the routine has
# moved on to something else.
#
# This is the realistic version: the capture thread is mid-measurement when
# the deadline expires, the routine is resumed with None and carries on,
# and the result then turns up. Without the guard it satisfies whatever the
# routine is waiting for NOW -- so a Sleep returns a floor tuple, and the
# routine reads a measurement it never asked for as the answer to a
# different question. Exactly the shape of C2: the number is real, and it
# belongs to something else.
#
# Added because the mutation test found nothing covering it: removing the
# guard in `_apply` left all 68 other checks green.
clock, d, log = make()
steps: list[object] = []


def floor_then_sleep():
    steps.append((yield Floor(3.0, timeout_ms=1000)))
    steps.append((yield Sleep(5000)))
    steps.append("end")


d.start(Job("noise", floor_then_sleep))
clock.advance(1001)                       # the floor measurement times out
check(steps == [None], f"resumed with None at its deadline: {steps}")
d.deliver_floor(4.0, 2.0, 97)             # ...and the result arrives late
check(steps == [None], f"the late floor did NOT satisfy the sleep: {steps}")
check(any("waiting for sleep" in o for o in d.overlaps),
      f"and it is recorded, saying what it interrupted: {d.overlaps}")
clock.advance(5001)
check(steps == [None, None, "end"], f"the sleep ran its own course: {steps}")


# ---------------------------------------------------------------------------
section("6. control writes are counted, spaced and de-duplicated in one place")
# ---------------------------------------------------------------------------
#
# RULES A4/A5/A6. One CAP_PROP_EXPOSURE write has hung this camera for 272
# seconds. The probe's auto-range re-entered its own step function and
# re-wrote the SAME exposure on every rung of the ladder.

sink = Sink()
clock, d, log = make(sink)


def ladder():
    for _ in range(7):
        yield Control("exposure", -7.0)      # the same value, seven times
    yield Control("exposure", -8.0)
    yield Control("brightness", 128.0)


d.start(Job("probe", ladder))
clock.run_until_idle()
check(sink.controls == [("exposure", -7.0), ("exposure", -8.0),
                        ("brightness", 128.0)],
      f"six redundant writes never reached the camera: {sink.controls}")
check(d.control_writes == 3, f"three counted: {d.control_writes}")
check(d.control_writes_skipped == 6,
      f"six skipped, and the number is visible: {d.control_writes_skipped}")

# the 60ms gap is enforced by the Director, not by each routine
sink = Sink()
clock, d, log = make(sink)


def two_writes():
    yield Control("exposure", -5.0)
    yield Control("exposure", -6.0)


d.start(Job("probe", two_writes))
check(len(sink.controls) == 1, "the second write waits")
clock.advance(59)
check(len(sink.controls) == 1, "still waiting at 59ms")
clock.advance(2)
check(len(sink.controls) == 2, "and goes out after the spacing")


# ---------------------------------------------------------------------------
section("7. a delivery during a resume is queued, never re-entrant")
# ---------------------------------------------------------------------------
#
# There were fifteen `QApplication.processEvents()` calls inside routines.
# Each was a point where any other callback -- including the same routine's
# next step -- could run INSIDE the current step.

clock, d, log = make()
seq: list[str] = []


def reentrant():
    seq.append("a")
    # Deliver a floor result from inside the routine's own execution, which
    # is exactly what processEvents() used to permit.
    yield Call(lambda: d.deliver_floor(1.0, 1.0, 10))
    seq.append("b")
    res = yield Floor(3.0)
    seq.append(f"floor {res}")


d.start(Job("noise", reentrant))
check(seq == ["a", "b"], f"the stray delivery did not jump the queue: {seq}")
d.deliver_floor(4.0, 2.0, 97)
check(last(seq) == "floor (4.0, 2.0, 97)", f"the real one arrived: {seq}")


# ---------------------------------------------------------------------------
section("8. a routine that loops forever fails the test instead of hanging it")
# ---------------------------------------------------------------------------
#
# RULES D10, and a real one: `_start_noise_floor` re-armed itself every
# 700ms with no bound whenever the probe was drawing, so a probe that never
# finished meant a floor retry loop that never ended.

clock, d, log = make()


def forever():
    while True:
        yield Sleep(700)


d.start(Job("noise", forever))
try:
    clock.run_until_idle(limit_ms=30000)
    check(False, "an endless routine should have been caught")
except AssertionError as exc:
    check("looping" in str(exc), f"caught and named: {exc}")
d.cancel()


# ---------------------------------------------------------------------------
section("9. the declaration replaces the six-flag disjunctions")
# ---------------------------------------------------------------------------
#
# `_app_is_drawing` was an OR of six flags, maintained by hand in three
# places. RULES B9: the gunshot's test is not "which routine is running"
# but "could this detection possibly BE a trigger pull".

clock, d, log = make()


def quiet():
    yield Sleep(100)


check(d.is_drawing() is False, "idle: the app is not drawing")
check(d.owns_detections() is False, "idle: detections belong to the shooter")

d.start(Job("selftest", quiet, draws=True, draws_red=True,
            consumes_detections=True, measures_screen=True,
            label="testing itself - do not fire"))
check(d.is_drawing() is True, "the self test draws")
check(d.owns_detections() is True, "and its dots are not trigger pulls")
check(d.busy_label() == "testing itself - do not fire",
      "and the shooter is told, from the same declaration")
clock.run_until_idle()
check(d.owns_detections() is False, "which lapses when it finishes")

# The grid is the documented exception: he really is shooting during it.
d.start(Job("grid", quiet, draws=True, consumes_detections=False,
            label="shoot the highlighted ring"))
check(d.owns_detections() is False,
      "grid calibration does NOT consume detections -- he is shooting it")
clock.run_until_idle()


# ---------------------------------------------------------------------------
section("10. a failing effect goes back into the routine, which can clean up")
# ---------------------------------------------------------------------------

clock = ManualClock()
d = Director(clock, sink=None, log=[].append)       # no camera at all
outcome: list[str] = []


def needs_a_camera():
    try:
        yield Still("pat_lit")
        outcome.append("got a frame")
    except EffectFailed as exc:
        outcome.append(f"refused: {exc}")
    finally:
        outcome.append("cleaned up")


start_no_escape(d, Job("pattern", needs_a_camera))
check(outcome == ["refused: no camera to photograph with", "cleaned up"],
      f"the routine was told, and still cleaned up: {outcome}")
check(d.active is None, "and the slot is free")


# ---------------------------------------------------------------------------
section("11. a queued routine runs when the slot frees, bounded and announced")
# ---------------------------------------------------------------------------
#
# The old code DROPPED a noise measurement requested while one was running,
# or re-armed it on an unbounded loop. Both are defensible; neither says so.

clock, d, log = make()
ran: list[str] = []


def holder():
    ran.append("holder")
    yield Floor(3.0)


def later():
    ran.append("later")
    yield Sleep(10)


d.start(Job("pattern", holder))
check(d.start(Job("noise", later, queueable=True)) is False,
      "a queueable job does not start while busy")
check(any("queued behind pattern" in line for line in log),
      f"and says so: {last(log, '(nothing was logged)')}")
check(d.start(Job("noise", later, queueable=True)) is False,
      "a second copy is refused, not stacked")
check(sum(1 for line in log if "already queued" in line) == 1,
      "and that refusal is explained")
d.deliver_floor(4.0, 2.0, 97)
check(ran == ["holder", "later"], f"the queued job then ran: {ran}")
clock.run_until_idle()


# ---------------------------------------------------------------------------
section("12. one clock: a Deadline can only come from the Director's clock")
# ---------------------------------------------------------------------------
#
# RULES B4. Three clocks were in use -- time.time(), time.monotonic(),
# perf_counter() -- and on Windows none share an epoch, so the READY
# banner's deadline was unreachable and its 59% veil hid every bullet hole
# he shot for an evening.

clock = ManualClock(start=1000.0)
dl = Deadline.after(clock, 250)
check(dl.expired(clock.now()) is False, "not expired at once")
check(abs(dl.remaining_ms(clock.now()) - 250.0) < 1e-6, "remaining is right")
clock.advance(250)
check(dl.expired(clock.now()) is True, "expired on the Director's clock")
check(not hasattr(Deadline, "now"), "Deadline cannot read a clock of its own")


# ---------------------------------------------------------------------------
section("13. the same launch, same order, every time (determinism)")
# ---------------------------------------------------------------------------
#
# The point of the virtual clock. The cold-start harness could previously
# only jitter real timers, which samples a handful of the orderings that
# exist; three seeds found one race by luck.

def launch_trace() -> list[str]:
    sink2 = Sink()
    clock2 = ManualClock()
    tr: list[str] = []
    dd = Director(clock2, sink=sink2, log=lambda s: tr.append(f"log:{s}"))

    def chain():
        tr.append("expose")
        yield Control("exposure", -7.0)
        tr.append("floor")
        yield Floor(3.0)
        tr.append("still")
        yield Still("pat_lit")
        tr.append("end")

    dd.start(Job("setup", chain))
    for _ in range(40):
        clock2.advance(50)
        if dd.active is None:
            break
        if dd._active is not None and dd._active.waiting_for == "floor":
            dd.deliver_floor(4.0, 2.0, 97)
        elif dd._active is not None and dd._active.waiting_for == "still":
            answer_still(dd, sink2, "F")
    return [t for t in tr if not t.startswith("log:")]


runs = [launch_trace() for _ in range(5)]
check(all(r == runs[0] for r in runs),
      f"five launches, one order: {set(tuple(r) for r in runs)}")
check(runs[0] == ["expose", "floor", "still", "end"], f"and it is right: {runs[0]}")


# ---------------------------------------------------------------------------
section("14. the overlap oracle stays empty on a clean run")
# ---------------------------------------------------------------------------

sink14 = Sink()
clock, d, log = make(sink14)


def clean():
    yield Control("exposure", -7.0)
    yield Floor(3.0)
    yield Still("pat_lit")


d.start(Job("setup", clean))
clock.advance(60)
d.deliver_floor(4.0, 2.0, 97)
answer_still(d, sink14, "F")
check(d.overlaps == [], f"no overlaps on a clean run: {d.overlaps}")
check(d.refusals == [], "and no refusals")
check(d.ran("setup") is True, "and the run is recorded as having completed")


# ---------------------------------------------------------------------------
section("18. shutting down does not start anything")
# ---------------------------------------------------------------------------
#
# The startup chain hands on with `then=` callbacks, so cancelling the probe
# at shutdown would have STARTED THE SELF TEST -- on the slot the cancel
# just freed, while the target window was being destroyed. And the mapping
# check's `then(False)` runs `_on_mapping_checked`, which sets
# `self.mapping = None`, writes a camera control, and arms a 400ms timer
# into the calibration chain.
#
# A Qt timer firing into a destroyed scene takes the process down with no
# traceback and nothing on stderr, which is indistinguishable from a
# force-kill to the camera -- and a force-kill leaves it opening in 40.3s
# with 0 frames, against 0.44s and 53.6 fps after a clean close (A1, A3).
#
# There was no "closing" state anywhere in this app. Found by an outside
# review of the diff, not by any test here.

clock, d, log = make()
events: list[str] = []


def long_routine():
    events.append("probe started")
    try:
        yield Sleep(10000)
    finally:
        events.append("probe cleaned up")


def next_routine():
    events.append("SELF TEST STARTED DURING SHUTDOWN")
    yield Sleep(10)


d.start(Job("probe", long_routine,
            on_done=lambda _v: d.start(Job("selftest", next_routine))))
check(events == ["probe started"], f"the probe is running: {events}")

d.close("the app is closing")
check("probe cleaned up" in events, "its cleanup still runs on close")
check("SELF TEST STARTED DURING SHUTDOWN" not in events,
      f"and the chain does NOT hand on during teardown: {events}",
      "a routine starting while the window is destroyed is a crash, which "
      "to the camera is a force-kill")
check(d.start(Job("selftest", next_routine)) is False,
      "nothing may start after close()")
# The REFUSAL's own line, not any line mentioning the words: `cancel()` already logs
# "standing down (the app is closing)", so the looser test passed with the start()
# refusal removed entirely (mutate_director entry 13 left it green).
check(any("not starting -- the app is closing" in line for line in log),
      "and the refusal says why", f"{log[-2:]}")
clock.run_until_idle()
check("SELF TEST STARTED DURING SHUTDOWN" not in events,
      "not even on a later timer tick")

# cancel(), by contrast, MUST tell the chain -- otherwise a stood-down
# routine leaves whatever was waiting for it waiting for ever (C1, D4).
clock, d2, log2 = make()
told2: list[object] = []
d2.start(Job("probe", long_routine, on_done=told2.append))
d2.cancel("the camera stalled")
check(told2 == [None],
      f"cancel still reports to the chain, with None: {told2}",
      "a chain told nothing stops silently, which is the wait he had to "
      "press Escape to get out of")


# ---------------------------------------------------------------------------
section("20. a run finishes exactly once, even cancelled from inside itself")
# ---------------------------------------------------------------------------
#
# `cancel()` reached from inside one of the routine's own `Call` effects
# finished the same run TWICE: history recorded `cancelled` and then `done`,
# and `ran()` returned True for a routine that had been stood down -- so a
# cancelled run could be reported as a completed one, which is the kind of
# number this project has already handed him wrongly.
#
# Reachable today: `_publish_verdict` calls `processEvents()` inside a
# `Call`, so any queued close can land in the middle of a routine's step.
# Found by an outside review of the diff.

clock, d, log = make()
steps20: list[str] = []


def cancels_itself():
    steps20.append("start")
    # Exactly what a Call that pumps the event loop can do.
    yield Call(lambda: d.cancel("close arrived mid-step"))
    steps20.append("carried on after being cancelled")
    yield Sleep(10)


d.start(Job("selftest", cancels_itself))
clock.run_until_idle()
outcomes = [o for n, o, _ms in d.history if n == "selftest"]
check(len(outcomes) == 1,
      f"the run is recorded ONCE, not twice: {d.history}",
      "a cancelled run recorded as 'done' as well is a cancelled run that "
      "can be reported as a success")
check(outcomes == ["cancelled"],
      f"and it is recorded as cancelled, not done: {outcomes}")
check(d.ran("selftest") is False,
      "ran() is False for a routine that was stood down",
      "this is the function a verdict would ask")
check(d.active is None, "and the slot is free")


# ---------------------------------------------------------------------------
section("19. a failing effect inside an exception handler cannot wedge it")
# ---------------------------------------------------------------------------
#
# `_throw` dispatched the routine's NEXT effect with no guard. A routine
# that catches EffectFailed -- the pattern section 10 teaches -- and then
# yields another effect that also fails let the exception escape the
# Director entirely: `_active` stayed set, `finished` stayed False, no timer
# was armed, nothing was logged, and every later `start()` was refused.
#
# The Director deadlocks for the session, silently, in the class whose one
# job is to be the thing that cannot deadlock. Found by an outside review.

clock = ManualClock()
d3 = Director(clock, sink=None, log=lambda _s: None)   # no camera: Still fails
trace3: list[str] = []


def catches_then_fails_again():
    try:
        yield Still("one")
    except EffectFailed:
        trace3.append("caught the first")
        # And now ask for something that cannot work either.
        yield Still("two")
        trace3.append("should not get here")
    finally:
        trace3.append("cleaned up")


start_no_escape(d3, Job("probe", catches_then_fails_again))
check(trace3[:1] == ["caught the first"], f"the first failure is caught: {trace3}")
check("cleaned up" in trace3, f"and cleanup ran: {trace3}")
check(d3.active is None,
      "the slot is FREE after a failure inside an exception handler",
      "it stayed held, with no timer and nothing logged, and every later "
      "routine was refused for the rest of the session")
check(d3.start(Job("selftest", lambda: iter(()))) is False
      or d3.busy_with() != "probe",
      "and the Director still works afterwards")
check(any("probe" == n for n, _o, _ms in d3.history),
      f"the run is recorded rather than vanishing: {d3.history}")


# ---------------------------------------------------------------------------
section("16. the bridge to the routines not yet migrated")
# ---------------------------------------------------------------------------
#
# One run slot is the whole answer only once everything is on it. Until
# then the app still has pattern calibration, the grid and a legacy noise
# floor driven by timers and flags, and a Director routine starting on top
# of one of those is the exact collision this class exists to prevent,
# reintroduced by the act of fixing it.
#
# Written here because the MUTATION CHECK found it untested: disabling the
# `world_busy` hook entirely left all 85 checks green. It was covered only
# in test_feedback, which the mutation script does not run -- so a Director
# property was being guarded by a test in another file, which is the same
# shape as a rule guarded by a comment.

clock, d, log = make()
outside = {"who": "pattern calibration"}
d = Director(clock, sink=Sink(), log=log.append,
             world_busy=lambda: outside["who"])
ran: list[str] = []


def wants_the_screen():
    ran.append("ran")
    yield Sleep(10)


check(d.start(Job("probe", wants_the_screen)) is False,
      "a routine refuses while something OUTSIDE the Director holds the screen")
check(ran == [], f"and it did not run: {ran}")
check(any("pattern calibration" in line for line in log),
      "the refusal names what is holding it, not just that it refused",
      f"{log}")
outside["who"] = ""
check(d.start(Job("probe", wants_the_screen)) is True,
      "and it runs once that routine has finished")
clock.run_until_idle()

# A routine QUEUED behind an outside owner has nothing to wake it: `_next`
# only runs when a Director routine finishes. `poke()` is the explicit
# drain, called from the app's own tick. Without it the queue just sat
# there -- a step that quietly does nothing (RULES C1).
clock, _unused, log2 = make()
outside2 = {"who": "a noise-floor measurement"}
d2 = Director(clock, sink=Sink(), log=log2.append,
              world_busy=lambda: outside2["who"])
ran2: list[str] = []


def queued_one():
    ran2.append("ran")
    yield Sleep(10)


check(d2.start(Job("noise", queued_one, queueable=True)) is False,
      "a queueable routine waits rather than being dropped")
check(any("queued behind" in line for line in log2), f"and says so: {log2}")
outside2["who"] = ""
d2.poke()
check(ran2 == ["ran"], f"poke() starts it once the screen is free: {ran2}")
clock.run_until_idle()

# And poke() must not spin: a job it cannot start keeps its place rather
# than being popped and re-queued for ever.
outside2["who"] = "pattern calibration again"
d2.start(Job("noise", queued_one, queueable=True))
before = len(log2)
for _ in range(5):
    d2.poke()
check(len(log2) - before <= 2,
      "poking repeatedly while still blocked does not spam or spin",
      f"{len(log2) - before} log lines from five pokes")


# ---------------------------------------------------------------------------
section("17. a routine's completion is reported, success or not")
# ---------------------------------------------------------------------------
#
# The startup chain hands on with `then=` callbacks. A callback that fires
# only on success stops the chain silently when a routine is stood down or
# fails -- which is the shape of the wait that made him press Escape to get
# an answer out of this app (RULES D4).
#
# Also found untested by the mutation check: disabling the callback drain
# entirely left all 85 checks green.

clock, d, log = make()
told: list[object] = []


def succeeds():
    yield Sleep(10)
    return "the answer"


d.start(Job("probe", succeeds, on_done=told.append))
clock.run_until_idle()
check(told == ["the answer"],
      f"a completed routine hands its RESULT to the callback: {told}")

told.clear()


def fails():
    yield Sleep(10)
    raise RuntimeError("the camera fell off")


d.start(Job("probe", fails, on_done=told.append))
clock.run_until_idle()
check(told == [None], f"a failed routine reports None, not nothing: {told}")

told.clear()


def gets_cancelled():
    yield Sleep(10000)


d.start(Job("probe", gets_cancelled, on_done=told.append))
d.cancel("shutting down")
check(told == [None],
      f"and so does one that is stood down -- the chain is never left "
      f"waiting for a callback that is not coming: {told}")

# A callback may start the next routine, and must find a free slot.
clock, d, log = make()
order: list[str] = []


def second():
    order.append("second")
    yield Sleep(10)


def first_one():
    order.append("first")
    yield Sleep(10)


d.start(Job("probe", first_one,
            on_done=lambda _v: d.start(Job("selftest", second))))
clock.run_until_idle()
check(order == ["first", "second"],
      f"a callback can chain the next routine: {order}")
check(d.ran("selftest") is True, f"and it actually completed: {d.history}")


# ---------------------------------------------------------------------------
section("15. the PRODUCTION clock, exercised for real")
# ---------------------------------------------------------------------------
#
# Every section above runs on ManualClock, and that is the point of
# ManualClock -- but it meant `QtClock` was never executed by a test at
# all. It had `t.setTimerType(0)` in it, where PySide6 requires the enum
# and raises TypeError, so on the first real cold start after the migration
# the exposure probe and the self test both died at their first Sleep:
#
#   22:08:05  probe FAILED: EffectFailed('TypeError(...setTimerType...)')
#
# 73 checks green, and the production clock could not arm a single timer.
# A stand-in that is never checked against the real thing excuses every
# consumer of the difference (RULES C12), and a suite that exercises only
# the stand-in is testing itself.
#
# This section is deliberately the slowest thing in the file -- it needs a
# real Qt event loop and real elapsed milliseconds -- and still costs well
# under a second.
import os                                                       # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
    from lsa.director import QtClock

    _app = QCoreApplication.instance() or QCoreApplication([])
    qclock = QtClock()
    marks: list[str] = []
    sink2 = Sink()
    qd = Director(qclock, sink=sink2, log=lambda s: marks.append(f"log:{s}"))

    def real_routine():
        marks.append("start")
        yield Sleep(20)
        marks.append("slept")
        yield Control("exposure", -7.0)      # Sleep for the write spacing
        marks.append("wrote")
        return "ok"

    check(qd.start(Job("real", real_routine)) is True,
          "a routine starts on the production clock")

    # Spin a real event loop until the routine finishes or a real deadline
    # passes. Watches for failure as well as success (RULES D4).
    loop = QEventLoop()
    bail = QTimer()
    bail.setSingleShot(True)
    bail.timeout.connect(loop.quit)
    bail.start(3000)
    while qd.active is not None and bail.isActive():
        loop.processEvents(QEventLoop.AllEvents, 10)
    bail.stop()

    check(marks[:1] == ["start"], f"it ran: {[m for m in marks if not m.startswith('log:')]}")
    check("slept" in marks, "a real QTimer fired and resumed the routine",
          f"{[m for m in marks if not m.startswith('log:')]}")
    check("wrote" in marks, "and so did the one after the control write")
    check(sink2.controls == [("exposure", -7.0)],
          f"the write reached the camera: {sink2.controls}")
    check(qd.ran("real") is True, f"and it completed: {qd.history}")
    check(not any("FAILED" in m for m in marks),
          f"with nothing failing: {[m for m in marks if 'FAILED' in m]}")

    # Cancellation must also work on the real clock: the timer is stopped,
    # not merely orphaned, because an orphaned timer firing into a
    # destroyed scene is indistinguishable from a force-kill to the camera.
    cleaned = {"n": 0}

    def long_one():
        try:
            yield Sleep(5000)
        finally:
            cleaned["n"] += 1

    qd.start(Job("long", long_one))
    check(qd.active is not None, "a long routine is running")
    # `_active` is None when the routine above never started (the production
    # clock failing to arm a timer is exactly what makes it so): the check
    # below then fails BY NAME rather than this line killing the suite.
    armed = qd._active.timer if qd._active is not None else None   # the real QTimer
    check(armed is not None and armed.isActive(),
          "and its timer is armed on the real clock")
    qd.cancel("test")
    check(cleaned["n"] == 1, "cancel on the real clock runs the cleanup")
    check(qd.active is None, "and frees the slot")
    # THE TIMER IS STOPPED, not merely orphaned.
    #
    # An orphaned single-shot timer still fires. Today a sequence check
    # inside `fire()` makes that harmless, but "harmless because something
    # else catches it" is not the property: a Qt timer firing into a scene
    # whose window has been destroyed takes the process down with no
    # traceback and nothing on stderr -- which is indistinguishable from a
    # force-kill to the camera, and a force-kill leaves it opening in 40.3s
    # with 0 frames (RULES A1, A3).
    #
    # Added because the mutation check found nothing covering it: replacing
    # the body of QtClock.cancel with `pass` left all 83 checks green.
    check(armed is not None and not armed.isActive(),
          "and the real timer was STOPPED, not left to fire later",
          "a timer firing after teardown is how shutdown became a crash")
except ImportError as exc:                                      # pragma: no cover
    check(False, "PySide6 is available to exercise the production clock",
          f"{exc}")


# ---------------------------------------------------------------------------
section("21. nothing may take the slot while a routine is mid-step")
# ---------------------------------------------------------------------------
#
# FOUND BY FABLE'S REVIEW OF THE DIFF, 2026-10-16, and it is the fault of
# section 11 with an outside caller instead of an inside one.
#
# `_resume()` returns immediately while `_resuming` is True. So a `start()`
# that lands mid-step created the routine's generator, set `_active`, and
# then NEVER ADVANCED IT: a routine that logs as started, does nothing for
# ever, and holds the only slot there is. `_finish` was fixed for this in
# section 11 by setting `_want_next` instead of starting anything -- and the
# same hole stayed open to every other caller.
#
# It is reachable, not theoretical: `_finish` clears `_active` and calls
# `_on_change()` from INSIDE the resume, `_on_change` is the app's own
# callback, and `poke()` from there finds a slot that looks free.

# THE WINDOW IS WHERE `_active` IS ALREADY CLEARED. That matters: while a
# routine is still installed, the ordinary "something is running" guard
# covers this, so a test written against THAT window passes with the fault
# in place -- which the mutation check demonstrated on the first attempt at
# this section. The real hole is inside `_finish`, which sets `_active` to
# None and then calls `_on_change()` without having left the resume.
sink21 = Sink()
clock21 = ManualClock()
log21: list[str] = []
intruder_ran: list[str] = []
intruder_done: list[str] = []
tried: list[int] = []


def intruder():
    intruder_ran.append("started")
    yield Log("the intruder is running")
    intruder_done.append("finished")


def on_change_starts_something():
    # Exactly what the app does here is repaint the prompt band; what
    # matters is that it is APP code, running inside the Director's resume.
    #
    # Fired only in the window that matters: AFTER a routine has finished,
    # which is when `_finish` has cleared `_active` without having left the
    # resume. The first version of this fired on the host's own `start()`,
    # where `_active` is set and the ordinary "something is running" guard
    # covers it -- so it spent its one attempt outside the hole and passed
    # with the fault in place. The mutation check is the only reason that
    # was noticed.
    if d21.history and not tried:
        tried.append(1)
        d21.start(Job("intruder", intruder, queueable=False))


d21 = Director(clock21, sink=sink21, log=log21.append,
               on_change=on_change_starts_something)


def brief_host():
    yield Log("the host does one thing and finishes")


d21.start(Job("host", brief_host))
check(d21.active is None,
      "no routine is left holding the slot",
      f"busy_with() == {d21.busy_with()!r} -- a run installed mid-step never "
      f"advances: it logs as started and then sits there for ever, holding "
      f"the only slot there is")
check(intruder_ran == intruder_done,
      "a routine that started is a routine that ran to completion",
      f"started {intruder_ran}, finished {intruder_done} -- started and "
      f"stuck is the shape of this fault (RULES C1)")
clock21.advance(50)
check(d21.active is None, "and nothing wakes up later holding it either")

# A QUEUEABLE one waits instead of being thrown away -- the slot is about to
# be free, so refusing outright would lose a routine the launch needs.
sink21b = Sink()
clock21b, d21b, log21b = make(sink21b)
later_ran: list[str] = []


def later():
    later_ran.append("started")
    yield Log("the queued routine ran after all")


def finishes_and_pokes():
    yield Call(lambda: d21b.start(Job("later", later, queueable=True)))
    yield Log("done")


d21b.start(Job("host2", finishes_and_pokes))
check(later_ran == ["started"],
      "a queueable routine started mid-step runs once the step is over",
      f"{later_ran} -- queued but never drained is a step that quietly does "
      f"nothing (RULES C1), which is the failure this class exists to end")
check(d21b.active is None, "and the slot is free afterwards")

# And poke() itself, which is the real route in: the app calls it from the
# HUD tick about twelve times a second.
sink21c = Sink()
clock21c, d21c, log21c = make(sink21c)
poke_ran: list[str] = []


def poked():
    poke_ran.append("started")
    yield Log("ran")


def pokes_the_director():
    yield Call(lambda: (d21c._queue.append(Job("poked", poked, queueable=True)),
                        d21c.poke()))
    yield Sleep(5)


d21c.start(Job("host3", pokes_the_director))
check(poke_ran == [],
      "poke() mid-step does not start a queued routine either",
      "two independent defences, because this one is called twelve times a "
      "second from the HUD tick")
clock21c.advance(10)
check(poke_ran == ["started"],
      "and it starts as soon as the step is over")



# ---------------------------------------------------------------------------
# A SCREEN MEASUREMENT WILL NOT START UNLESS THE APP OWNS THE SCREEN.
#
# 2026-10-16: five gate launches, every one failing with all eighteen markers
# invisible and 'lit frame peaks at 255, unlit at 255' -- both photographs
# were of his desktop, because the target window was not in front. He had to
# tell me: 'before you take any screen measures you need to see if the app is
# maximized on screen.' 150 seconds a launch, five launches, and the app
# never reported the one thing it was in a position to measure.
print("\n22. a screen measurement needs the screen")
started22: list[int] = []


def _measures_screen():
    started22.append(1)
    yield Log('should never get here')


d22 = Director(clock=ManualClock(), log=lambda m: None,
               screen_ready=lambda: 'something is in front of it')
ok22 = d22.start(Job('selftest', _measures_screen, measures_screen=True))
check(not ok22 and not started22,
      'a routine that photographs the screen is refused when it is covered',
      'eighteen markers were photographed against a desktop, five times')

# THE MIRROR, so refusing everything cannot pass instead. Four checks on this
# project have passed by never firing.
ran22: list[int] = []


def _measures_screen_ok():
    ran22.append(1)
    yield Log('measuring')


d22b = Director(clock=ManualClock(), log=lambda m: None, screen_ready=lambda: '')
ok22b = d22b.start(Job('selftest', _measures_screen_ok, measures_screen=True))
check(bool(ok22b and ran22), 'and it runs when the app does own the screen')

# A routine that does not photograph the screen is unaffected.
ran22c: list[int] = []


def _not_a_screen_job():
    ran22c.append(1)
    yield Log('nothing to do with the screen')


d22c = Director(clock=ManualClock(), log=lambda m: None,
                screen_ready=lambda: 'something is in front of it')
ok22c = d22c.start(Job('other', _not_a_screen_job))
check(bool(ok22c and ran22c),
      'and a routine that does not touch the screen still runs')

# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
if FAILED:
    print(f"FAILED {len(FAILED)} of {CHECKS} checks:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL {CHECKS} CHECKS PASSED - the Director holds its properties")
