"""The Director: one owner of the screen, the camera and the detector's routing.

WHY THIS EXISTS
---------------
This app is about fifteen linear procedures -- set an exposure, draw
something, wait for the panel to paint, photograph it, compute, put
everything back -- that each need the monitor, the camera and the detector
to themselves for a few seconds.

Every one of them was written as a chain of `QTimer.singleShot` callbacks
with its local variables hoisted onto one 7,000-line window object as
flags. Measured on 2026-10-15, in that code: 74 timer hand-offs, 15 places
that re-enter the Qt event loop mid-step, and ~60 `_*_active` /
`_*_in_flight` / `_*_pending` attributes. The cost, all of it on the record:

  * Two pattern calibrations ran at once and every photograph was taken for
    a marker other than the one on screen. Two sessions lost.
  * The noise floor measured the exposure probe's red dots; the probe
    measured a screen the floor measurement had covered. Five cold launches
    came out bimodal -- 12/12 twice, 0/12 twice -- on identical code.
  * Five race guards were written one at a time, each naming a different
    subset of the flags. `SCREEN_OWNERS`, the central list added to end
    that, ended up consulted by ONE of the six routines that needed it.
  * `closeEvent` stood routines down by a hand-written list of eight flag
    names. It omitted `_check_active`. The comment above it says, in as many
    words, that the timers cannot be cancelled because
    `QTimer.singleShot` keeps no handle -- so the flags "have to be" the
    defence. They were the defence, and they were incomplete.
  * Cleanup was skipped on an exit path five separate times: three flash
    covers left up, `_restore_levels` not reached, `_pat_hidden` not
    restored.
  * `_request_still` held ONE outstanding tag in a string and dispatched by
    prefix. A second requester overwrote it, and the first routine's
    timeout then silently never fired.

Each of those was fixed once, correctly, and left standing somewhere else.
That is not carelessness; it is the shape of the code. A property that has
to hold in N places gets fixed in one. **The cure is to make N equal 1.**

WHAT IT IS
----------
A routine becomes a generator -- straight-line code again -- that YIELDS
the thing it wants to wait for:

    def measure(self):
        yield Control("exposure", -7.0)
        yield Sleep(400)
        frame = yield Still("probe_lit")
        ...

The Director holds exactly ONE run slot. It resumes the generator when the
awaited effect completes, and nothing else can take the screen meanwhile
because there is nowhere else to stand. The properties that used to need a
guard per routine now hold by construction:

  * two routines cannot interleave -- there is one slot
  * cleanup cannot be skipped -- it goes in `finally`, and cancelling a
    routine closes the generator, which RUNS the finally blocks
  * shutdown is `cancel()`, not a list of flag names
  * a wait cannot be unbounded -- the Director arms a deadline for every
    effect, whatever the caller passes
  * only one still can be outstanding, and a late frame from a cancelled
    run is dropped by run IDENTITY, not by comparing tag strings
  * control writes are counted, spaced and de-duplicated in one place
    (a single UVC write has hung this camera for 272 seconds)
  * "is the app drawing on the screen right now, and does it own the
    detections" is ONE field -- the active job -- instead of a six-flag
    disjunction repeated in four places

NO QT IN THIS FILE, DELIBERATELY
--------------------------------
The clock is injected. Production passes `QtClock`; tests pass
`ManualClock`, which makes a whole app launch run in milliseconds and makes
the ORDER of events something a test chooses rather than something it hopes
for. The headless cold-start harness could previously only jitter real
timers and sample a handful of the orderings that exist.

See `RULES.md` Part E, and `CLAUDE.md` 2026-10-15.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Tuple

__all__ = [
    "Sleep", "Still", "Floor", "Room", "Control", "Paint", "Call", "Log",
    "Job", "Director", "Clock", "ManualClock", "QtClock", "Deadline",
    "Cancelled", "EffectFailed",
]


# ---------------------------------------------------------------------------
# Effects: the complete list of things a routine may wait for.
# ---------------------------------------------------------------------------
#
# This list being short and closed is the point. A routine cannot invent a
# new way to wait, so there is no way to wait without a deadline, and no way
# to touch the camera or the screen except through the one object that
# serialises it.


@dataclass(frozen=True)
class Sleep:
    """Let the world turn for a while. The ONLY way a routine may pause.

    Replaces `QTimer.singleShot(ms, self._next_step)`. The difference is not
    stylistic: a singleShot keeps no handle, so it cannot be cancelled, so
    shutdown had to neutralise it by clearing a flag the callback would
    check -- which is why there was a hand-written list of eight flag names
    in `closeEvent`, and why it was missing one.
    """
    ms: float


@dataclass(frozen=True)
class Still:
    """Photograph the screen. Resumes with a BGR frame, or None on a stall.

    `timeout_ms` is a maximum, not a request: the Director arms a deadline
    whatever is passed. A fullscreen cover once stayed up permanently
    because the frame it was waiting for never arrived, and the only way out
    was to unplug the camera (RULES B6).
    """
    tag: str
    timeout_ms: float = 5000.0
    # TWO, not eight. These are BLOCKING reads inside the capture
    # thread command drain, and one probe issues up to a dozen stills --
    # ~96 back-to-back reads at the old default, on a camera whose
    # firmware is measured to hang on driver traffic. It stalled his
    # camera eighteen seconds into a launch, immediately after the
    # probe's brightness ladder. Every routine now has an explicit
    # Paint() and settle, so the extra six bought nothing.
    frames: int = 2


@dataclass(frozen=True)
class Floor:
    """Measure the noise floor. Resumes with (peak, p99, frames).

    Nine call sites used to start this directly and three once fired within
    four seconds of each other while the probe was drawing red dots. As an
    effect it is a STEP INSIDE the routine that needs it, so "measure the
    floor, then use it" is one function and cannot be interleaved with the
    thing that would spoil it.
    """
    seconds: Optional[float] = None
    timeout_ms: float = 20000.0


@dataclass(frozen=True)
class Room:
    """Measure the room at the current exposure. Resumes with (mean, chroma, frames).

    Writes nothing to the camera, which is why it is safe mid-session.
    """
    seconds: float = 1.5
    timeout_ms: float = 10000.0


@dataclass(frozen=True)
class Control:
    """Write one camera control, then wait out the inter-write spacing.

    Centralised because control traffic is the one thing measured to hang
    this camera: a single `CAP_PROP_EXPOSURE` write once blocked for 272
    seconds, while five minutes of pure streaming with no writes returned
    17,945 frames and zero failures.

    The Director de-duplicates against what it has itself written (never
    against a readback -- this camera reports exposure -6.0 and
    auto_exposure 0.0 while being neither), counts every write, and enforces
    the gap. An auto-range ladder that re-entered its own step function once
    re-wrote the SAME exposure on every rung: seven redundant transactions a
    launch, on top of the five that were needed.
    """
    key: str
    value: float
    spacing_ms: float = 60.0


@dataclass(frozen=True)
class Paint:
    """Wait until what was just drawn is actually on the glass.

    Replaces `QApplication.processEvents()`, of which there were fifteen
    inside routines. Each one was a re-entrancy point: any other callback --
    including the same routine's next step -- could run INSIDE the current
    step. A photograph taken before the screen painted once produced a
    marker at strength 14 where its siblings read 197-231, and the strongest
    noise blob in the difference image won.
    """
    frames: int = 2
    timeout_ms: float = 2000.0


@dataclass(frozen=True)
class Call:
    """Run a plain callable and resume with its value.

    For the small synchronous steps -- redraw this layer, read that number.
    It exists so a routine never has to reach around the Director to do
    something immediate, which would reintroduce the unserialised path.
    """
    fn: Callable[[], Any]
    what: str = ""


@dataclass(frozen=True)
class Log:
    """Say something, attributed to the running routine."""
    text: str


Effect = Any   # one of the above


class Cancelled(Exception):
    """Thrown into a routine that is being stood down.

    Thrown rather than simply abandoned so the routine's `finally` blocks
    run: that is what makes "put the screen back" structural instead of
    something every exit path has to remember. Five cleanup-skipped
    incidents on this project were all the same missing `finally`.
    """


class EffectFailed(Exception):
    """An effect could not be carried out at all (no camera, no painter)."""


# ---------------------------------------------------------------------------
# One clock
# ---------------------------------------------------------------------------
#
# RULES B4. Three clocks were in use for the same deadlines -- time.time(),
# time.monotonic() and perf_counter() -- and on Windows none of them share
# an epoch, so a deadline was either already past or unreachable. It was
# unreachable: the READY banner never cleared, and its 59% veil hid every
# bullet hole he shot for an evening.
#
# A prose rule cannot fix that, because a prose rule cannot tell a
# legitimate wall-clock log timestamp from a deadline. A TYPE can: a
# Deadline is constructible only from a Clock, and everything that waits
# goes through the Director, which has exactly one.


class Clock:
    """Interface: monotonic now() in seconds, and a cancellable timer."""

    def now(self) -> float:                                # pragma: no cover
        raise NotImplementedError

    def after(self, ms: float, fn: Callable[[], None]) -> Any:   # pragma: no cover
        raise NotImplementedError

    def cancel(self, handle: Any) -> None:                 # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class Deadline:
    """A moment on the Director's clock. The only kind of deadline."""
    at: float

    @staticmethod
    def after(clock: Clock, ms: float) -> "Deadline":
        return Deadline(clock.now() + max(0.0, ms) / 1000.0)

    def expired(self, now: float) -> bool:
        return now >= self.at

    def remaining_ms(self, now: float) -> float:
        return max(0.0, (self.at - now) * 1000.0)


class ManualClock(Clock):
    """A clock a test drives by hand. Deterministic, and instant.

    The cold-start harness could previously only add random jitter to real
    Qt timers, which samples a handful of the orderings that exist. With
    time under the test's control a whole launch runs in milliseconds and
    the order of events is CHOSEN, so a race either exists or it does not --
    rather than showing up on the third seed.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self._seq = 0
        # (due, seq, fn, cancelled-box). A list kept sorted on insert: the
        # queue is tiny and insertion order must break ties deterministically.
        self._timers: List[Tuple[float, int, Callable[[], None], List[bool]]] = []

    def now(self) -> float:
        return self._now

    def after(self, ms: float, fn: Callable[[], None]) -> Any:
        self._seq += 1
        box = [False]
        entry = (self._now + max(0.0, ms) / 1000.0, self._seq, fn, box)
        self._timers.append(entry)
        self._timers.sort(key=lambda e: (e[0], e[1]))
        return entry

    def cancel(self, handle: Any) -> None:
        if handle is None:
            return
        try:
            handle[3][0] = True
        except Exception:
            pass

    # -- test driving ------------------------------------------------------

    def pending(self) -> int:
        return sum(1 for e in self._timers if not e[3][0])

    def advance(self, ms: float) -> int:
        """Run every timer due within `ms`. Returns how many fired."""
        target = self._now + max(0.0, ms) / 1000.0
        fired = 0
        while True:
            due = [e for e in self._timers if not e[3][0] and e[0] <= target]
            if not due:
                break
            due.sort(key=lambda e: (e[0], e[1]))
            entry = due[0]
            self._timers.remove(entry)
            self._now = max(self._now, entry[0])
            entry[2]()
            fired += 1
        self._now = target
        return fired

    def run_until_idle(self, limit_ms: float = 600000.0,
                       max_steps: int = 100000) -> float:
        """Fire timers in order until none are left. Returns ms elapsed.

        Bounded on purpose: a routine that re-arms itself forever would
        otherwise hang the test instead of failing it, and this project has
        shipped exactly that -- a noise-floor retry on a 700ms loop with no
        limit (RULES D10).
        """
        t0 = self._now
        steps = 0
        while True:
            live = [e for e in self._timers if not e[3][0]]
            if not live:
                break
            live.sort(key=lambda e: (e[0], e[1]))
            entry = live[0]
            if (entry[0] - t0) * 1000.0 > limit_ms:
                raise AssertionError(
                    f"virtual clock ran past {limit_ms:.0f}ms of simulated "
                    f"time with timers still queued -- something is looping")
            steps += 1
            if steps > max_steps:
                raise AssertionError(
                    f"virtual clock fired {steps} timers without going idle "
                    f"-- something is re-arming itself")
            self._timers.remove(entry)
            self._now = max(self._now, entry[0])
            entry[2]()
        return (self._now - t0) * 1000.0


class QtClock(Clock):
    """The production clock: `perf_counter` plus real, CANCELLABLE QTimers.

    Single-shot QTimers are created as objects with a handle rather than via
    `QTimer.singleShot`, precisely so they can be stopped. That is what lets
    shutdown be a cancel instead of a list of flag names.
    """

    def __init__(self, parent: Any = None) -> None:
        from PySide6.QtCore import Qt, QTimer      # local: keep this file Qt-free
        import time as _time
        self._QTimer = QTimer
        # The ENUM, not 0. PySide6 type-checks this and raises TypeError,
        # which the Director correctly turned into an EffectFailed handed
        # back to the routine -- so the probe and the self test each
        # cleaned up and reported "FAILED: setTimerType called with wrong
        # argument types" instead of vanishing. Loud and named, which is
        # the design working; but it is still a bug of mine, and it cost
        # the first cold start after the migration.
        self._precise = Qt.PreciseTimer
        self._perf = _time.perf_counter
        self._parent = parent

    def now(self) -> float:
        return self._perf()

    def after(self, ms: float, fn: Callable[[], None]) -> Any:
        t = self._QTimer(self._parent)
        t.setSingleShot(True)
        t.setTimerType(self._precise)

        def fire() -> None:
            try:
                t.deleteLater()
            except Exception:
                pass
            fn()

        t.timeout.connect(fire)
        t.start(max(0, int(ms)))
        return t

    def cancel(self, handle: Any) -> None:
        if handle is None:
            return
        try:
            handle.stop()
            handle.deleteLater()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# A job: a routine plus what it needs the app to know about it
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One measurement routine, and its declaration of what it does.

    The declaration replaces several six-flag disjunctions that were
    maintained by hand in `_on_shot`, `_check_room`, `_publish_measuring`
    and `closeEvent`. Those read "is the probe active or the self test
    active or the mode test active or...", so every routine added later
    needed four edits in four places and never got all four.

    `draws_red` and `consumes_detections` are separate because they answer
    different questions, and conflating them is a bug this project shipped:
    the gunshot was first gated on "which calibration phase is running",
    which silenced the phases and left fourteen bangs a second coming from
    uncalibrated noise before any phase began (RULES B9).
    """

    name: str                             # short id, for logs and tests
    body: Callable[[], Generator]         # the routine, as a generator fn
    label: str = ""                       # what to tell the shooter
    draws: bool = False                   # puts anything on the target screen
    draws_red: bool = False               # puts laser-coloured light on it
    consumes_detections: bool = False     # detections are the app's, not his
    measures_screen: bool = False         # furniture must stand down (B2)
    queueable: bool = False               # may wait for the slot (bounded)

    # Called with the routine's return value when it COMPLETES, and with
    # None if it was stood down or failed. A transitional convenience for
    # callers that still chain with `then=`; the end state is one routine
    # composing another with `yield from`, which is what makes a whole
    # launch readable as a single function.
    #
    # It is called from outside the resume, so an `on_done` that starts
    # another job gets a genuinely free slot rather than one that is about
    # to be freed.
    on_done: Optional[Callable[[Any], None]] = None


@dataclass
class _Run:
    """The live state of the one routine that is running."""
    job: Job
    gen: Generator
    started_at: float
    seq: int
    timer: Any = None
    deadline: Optional[Deadline] = None
    waiting_for: str = ""
    still_tag: str = ""
    control_writes: int = 0
    cancelled: bool = False
    finished: bool = False


# ---------------------------------------------------------------------------
# The Director
# ---------------------------------------------------------------------------


class Director:
    """One run slot. Everything that needs the screen goes through here.

    The app wires its capture signals into `deliver_*`. It must not call the
    capture's measurement entry points itself -- `tests/test_rules.py`
    fails the build if it does, because that is the path that produced every
    interleaving on record.
    """

    MAX_QUEUE = 4

    def __init__(
        self,
        clock: Clock,
        *,
        sink: Any = None,
        log: Optional[Callable[[str], None]] = None,
        on_change: Optional[Callable[[], None]] = None,
        world_busy: Optional[Callable[[], str]] = None,
        screen_ready: Optional[Callable[[], str]] = None,
    ) -> None:
        self._clock = clock
        self._sink = sink                 # the capture thread, or a stub
        self._log_fn = log or (lambda _s: None)
        self._on_change = on_change or (lambda: None)
        # WHAT ELSE OWNS THE SCREEN, that the Director does not run.
        #
        # One run slot is only the whole answer once everything is on it.
        # Until then the app has routines still driven by timers and flags
        # -- pattern calibration, the grid, the legacy noise floor -- and a
        # Director routine starting on top of one of those is the exact
        # collision this class exists to prevent, reintroduced by the move
        # itself.
        #
        # Caught by test_feedback section 27 going red: the exposure probe,
        # newly migrated, no longer consulted anything. So the Director
        # asks the app what else is busy, and refuses if the answer is not
        # empty. The hook disappears when the last routine moves across.
        self._world_busy = world_busy or (lambda: "")
        # "" when the app owns the display, otherwise why it does not.
        # Asked before any routine that PHOTOGRAPHS the screen.
        self._screen_ready = screen_ready or (lambda: "")

        self._active: Optional[_Run] = None
        self._queue: Deque[Job] = deque()
        self._seq = 0
        self._resuming = False
        self._want_next = False
        self._dequeuing = False
        self._closing = False
        # Every Still request gets a unique tag; see `_dispatch`.
        self._still_seq = 0
        self._after: List[Tuple[Job, Any]] = []

        self.control_writes = 0                   # total, all runs
        self.control_writes_skipped = 0
        self.history: List[Tuple[str, str, float]] = []   # (name, outcome, ms)
        self.refusals: List[Tuple[str, str]] = []         # (wanted, busy-with)
        self.overlaps: List[str] = []                     # must stay empty

    # -- what the rest of the app asks about ------------------------------

    @property
    def active(self) -> Optional[Job]:
        run = self._active
        return None if run is None or run.finished else run.job

    def busy_with(self) -> str:
        """The name of the routine holding the slot, or ""."""
        job = self.active
        return "" if job is None else job.name

    def busy_label(self) -> str:
        job = self.active
        return "" if job is None else (job.label or job.name)

    def is_drawing(self) -> bool:
        """Is the app itself putting light on the target screen right now?

        The one question `_app_is_drawing` used to answer by OR-ing six
        flags in three different places.
        """
        job = self.active
        return bool(job is not None and (job.draws or job.draws_red))

    def owns_detections(self) -> bool:
        """Do detections belong to the app rather than to the shooter?

        While true, a detection must not make a gunshot, must not teach the
        microphone, and must not be drawn or scored: it is the app's own
        light. 94 detections in one run, about four of them his.
        """
        job = self.active
        return bool(job is not None and job.consumes_detections)

    def measuring_screen(self) -> bool:
        """Must informational furniture stand down? (RULES B2)"""
        job = self.active
        return bool(job is not None and job.measures_screen)

    # -- starting and stopping --------------------------------------------

    def close(self, why: str = "the app is closing") -> None:
        """Stand everything down FOR GOOD. Nothing may start after this.

        Distinct from `cancel()`, and the distinction is the difference
        between a clean shutdown and a wedged camera.

        `cancel()` ends the running routine and reports None to its
        completion callback, so the chain that was waiting is told rather
        than left waiting (RULES C1/D4). At shutdown that is exactly wrong:
        the startup chain hands on with `then=` callbacks, so
        `start_exposure_probe(then=start_self_test)` means cancelling the
        probe during `closeEvent` STARTS THE SELF TEST -- on the slot the
        cancel just freed, while the target window is being destroyed. And
        `start_mapping_check`'s `then(False)` runs `_on_mapping_checked`,
        which sets `self.mapping = None`, writes a camera control, and arms
        a 400ms timer into the calibration chain.

        A Qt timer firing into a destroyed scene takes the process down
        with no traceback and nothing on stderr, which is
        indistinguishable from a force-kill to the camera -- and a
        force-kill leaves it opening in 40.3s with 0 frames against 0.44s
        and 53.6 fps after a clean close (RULES A1, A3).

        So: after `close()`, the slot is shut, callbacks are dropped, and
        `start()` refuses. Found by an outside review of the diff; there
        was no "closing" state anywhere in this app.
        """
        self._closing = True
        self.cancel(why)
        self._queue.clear()
        self._after.clear()

    def start(self, job: Job) -> bool:
        """Take the slot, or refuse and say who has it.

        Refusing LOUDLY matters: "one of the two markers was not visible",
        with no marker named and no distance reported, is a check that can
        only be re-run, not debugged (RULES C3).
        """
        if self._closing:
            # Nothing starts after the app has begun shutting down. Said in
            # the log rather than dropped silently, because a chain that
            # stops at shutdown should be visible in the record of it.
            self._log(f"{job.name}: not starting -- the app is closing")
            return False
        if self._resuming:
            # A ROUTINE IS MID-STEP. Installing a run here would create its
            # generator and then never advance it, because `_resume()`
            # returns at once while `_resuming` is True -- a routine that
            # logs as started and does nothing, for ever, holding the slot.
            #
            # That is exactly the fault test_director section 11 caught in
            # `_finish`, which is why `_finish` sets `_want_next` instead of
            # starting anything. Fable's review (2026-10-16) found the same
            # hole still open to an OUTSIDE caller, and it is reachable:
            # `_finish` clears `_active` and calls `_on_change()` while
            # still inside the resume, `_on_change` is the app's own
            # callback, and `poke()` from there finds a free slot.
            #
            # Queue it if it can wait, refuse it loudly if it cannot. What
            # it must never do is install.
            if job.queueable and len(self._queue) < self.MAX_QUEUE                     and not any(q.name == job.name for q in self._queue):
                self._queue.append(job)
                self._log(f"{job.name}: queued -- another routine is "
                          f"mid-step and the slot cannot be handed over "
                          f"from inside one")
                return False
            self._refuse(job, self.busy_with() or "a routine mid-step",
                         "a routine is mid-step")
            return False
        if self._active is not None and not self._active.finished:
            holder = self._active.job.name
            if job.name == holder:
                # The commonest shape of this bug, and the one the live gate
                # was actually failing on: a routine re-entering ITSELF. The
                # first version of the noise-floor guard closed the door on
                # the probe and the self test and left it open to itself.
                self._refuse(job, holder, "it is already running")
                return False
            if job.queueable and not self._dequeuing \
                    and len(self._queue) < self.MAX_QUEUE:
                if any(q.name == job.name for q in self._queue):
                    self._refuse(job, holder, "one is already queued")
                    return False
                self._queue.append(job)
                self._log(f"{job.name}: queued behind {holder}")
                return False
            self._refuse(job, holder, "the screen is busy")
            return False

        # Nothing of ours is running -- but something that is not ours
        # might be. Asked before any side effect, as the first thing.
        outside = ""
        try:
            outside = self._world_busy() or ""
        except Exception as exc:
            self._log(f"{job.name}: could not find out what else is running "
                      f"({exc!r}); refusing rather than guessing")
            outside = "an unknown routine"
        # DOES THE APP ACTUALLY OWN THE DISPLAY?
        #
        # Asked for directly, 2026-10-16, after a five-launch gate in which
        # every launch failed and I blamed the room for it:
        #
        #   'the app did not acquire the full screen. That is the issue you
        #    need to check. Before you take any screen measures you need to
        #    see if the app is maximized on screen.'
        #
        # He was right and my diagnosis was wrong. The log said it exactly:
        #
        #     marker 15/18: not visible -- brightest point of the
        #     difference 27 (needs 12); lit frame peaks 255, unlit 255
        #
        # The target screen is near-black, so an unlit frame peaking at 255
        # is not a photograph of it. The camera was looking at the desktop:
        # the lit and unlit photographs were the same picture and the marker
        # had nothing to stand out from. Eighteen markers, five launches,
        # 150 seconds each, and the app never said the one thing it was in a
        # position to measure.
        #
        # `measures_screen` already names the routines this applies to, so
        # the check belongs here rather than at the top of each one: a guard
        # repeated per routine is the failure this class exists to end.
        if job.measures_screen:
            try:
                why_not = self._screen_ready() or ""
            except Exception as exc:
                why_not = f"could not tell ({exc!r})"
            if why_not:
                self._refuse(job, why_not, "the app does not own the screen")
                return False

        if outside:
            if job.queueable and not self._dequeuing \
                    and len(self._queue) < self.MAX_QUEUE \
                    and not any(q.name == job.name for q in self._queue):
                self._queue.append(job)
                self._log(f"{job.name}: queued behind {outside}")
                return False
            self._refuse(job, outside, "the screen is busy")
            return False

        self._seq += 1
        try:
            gen = job.body()
        except Exception as exc:
            self._log(f"{job.name}: could not start: {exc!r}")
            self.history.append((job.name, f"start-failed:{exc!r}", 0.0))
            return False
        # A body that is not a generator must be refused HERE, with a clear
        # reason. Found by test_feedback section 29, which passed a plain
        # iterator: the Director took the slot, died on `.send()` with an
        # AttributeError, freed the slot, and every routine that should
        # have been refused then started. One mistake at the edge undid
        # the single property this class exists to hold -- and it reported
        # it as a traceback about list_iterator, which names neither the
        # cause nor the consequence.
        if not (hasattr(gen, "send") and hasattr(gen, "throw")
                and hasattr(gen, "close")):
            self._log(f"{job.name}: not starting -- its body returned "
                      f"{type(gen).__name__}, not a generator. A routine must "
                      f"be a generator function so it can be resumed, "
                      f"cancelled, and have its cleanup run.")
            self.history.append((job.name, "start-failed:not-a-generator", 0.0))
            return False
        self._active = _Run(job=job, gen=gen, started_at=self._clock.now(),
                            seq=self._seq)
        self._on_change()
        self._resume(None)
        return True

    def cancel(self, why: str = "shutting down") -> bool:
        """Stand the running routine down, RUNNING ITS CLEANUP.

        This is the whole replacement for `closeEvent`'s hand-written list
        of eight flag names -- which omitted `_check_active`, so the
        two-marker check was never stood down at all. `gen.close()` raises
        Cancelled inside the routine at its current yield, so every
        `finally` runs: covers come down, levels go back, furniture returns.
        """
        run = self._active
        if run is None or run.finished:
            self._queue.clear()
            return False
        run.cancelled = True
        self._disarm(run)
        self._queue.clear()
        self._log(f"{run.job.name}: standing down ({why})")
        try:
            run.gen.throw(Cancelled(why))
        except (Cancelled, StopIteration):
            pass
        except Exception as exc:
            self._log(f"{run.job.name}: cleanup raised {exc!r}")
        finally:
            try:
                run.gen.close()
            except Exception:
                pass
        run.finished = True
        self._active = None
        self.history.append((run.job.name, "cancelled",
                             (self._clock.now() - run.started_at) * 1000.0))
        self._on_change()
        # The caller is TOLD, with None, rather than left waiting for a
        # callback that will never come. A chain whose next link is only
        # ever called on success stops silently at shutdown, and a wait
        # with no failure branch is how he came to press Escape to get an
        # answer out of this app (RULES C1, D4).
        #
        # UNLESS the app is closing, in which case telling the chain is how
        # the next routine gets started during teardown. See `close()`.
        if run.job.on_done is not None and not self._closing:
            self._after.append((run.job, None))
            self._drain_next()
        return True

    # -- deliveries from the world ----------------------------------------

    # Each returns True if a running routine was waiting for this and has
    # been handed it. The app uses that to tell migrated routines from
    # unmigrated ones while the move is in progress: if the Director did
    # not want it, the old handler runs. Without the return value both
    # would run, which is a worse bug than the one being fixed.
    def deliver_still(self, tag: str, frame: Any) -> bool:
        return self._deliver("still", (tag, frame))

    def deliver_floor(self, peak: float, p99: float, frames: int) -> bool:
        return self._deliver("floor", (peak, p99, frames))

    def deliver_room(self, mean: float, chroma: float, frames: int) -> bool:
        return self._deliver("room", (mean, chroma, frames))

    def deliver_painted(self) -> bool:
        return self._deliver("paint", ())

    def waiting_for(self) -> str:
        """What the running routine is blocked on, or "". For the GATE line."""
        run = self._active
        return "" if run is None or run.finished else run.waiting_for

    def _deliver(self, kind: str, payload: tuple) -> bool:
        if self._resuming:
            # A delivery arriving while a generator is mid-step is
            # UNSOLICITED by construction: a run that is executing is not
            # waiting for anything. Record it and drop it.
            #
            # An earlier version of this method QUEUED it and applied it to
            # the next request of the same kind. test_director section 7
            # caught that at once: a floor result that arrived before the
            # routine asked for a floor satisfied the ask that came
            # afterwards -- the stale-still fault wearing new clothes,
            # inside the class written to make it impossible.
            self.overlaps.append(
                f"unsolicited {kind} arrived while "
                f"{self.busy_with() or 'nothing'} was mid-step")
            return True        # consumed, in the sense of discarded here
        return self._apply(kind, payload)

    def _apply(self, kind: str, payload: tuple) -> bool:
        run = self._active
        if run is None or run.finished:
            # Nothing is running, so this belongs to whatever has not been
            # migrated onto the Director yet. Say so, rather than dropping
            # it: during the move, an unclaimed delivery is the old code's.
            return False
        if run.waiting_for != kind:
            self.overlaps.append(
                f"{kind} arrived while {run.job.name} was waiting for "
                f"{run.waiting_for or 'nothing'}")
            return True
        if kind == "still":
            tag, frame = payload
            if tag != run.still_tag:
                self.overlaps.append(
                    f"still tag '{tag}' does not match the outstanding "
                    f"'{run.still_tag}' in {run.job.name}")
                return True
            self._disarm(run)
            self._resume(frame)
            return True
        self._disarm(run)
        self._resume(payload if len(payload) != 1 else payload[0])
        return True

    # -- the engine --------------------------------------------------------

    def _resume(self, value: Any, throw: Optional[Exception] = None) -> None:
        """Drive the running routine until it waits, returns or raises.

        ONE LOOP, which handles both resuming with a value and throwing an
        exception in. There used to be two -- `_resume` and `_throw` -- and
        they diverged exactly where it mattered: `_throw` dispatched the
        routine's NEXT effect with no guard around it, so a routine that
        caught `EffectFailed` and then yielded another effect that also
        failed let the exception escape the Director completely. `_active`
        stayed set, `finished` stayed False, no timer was armed, nothing was
        logged, and every later `start()` was refused for the rest of the
        session: a silent deadlock in the class whose entire purpose is to
        be the thing that cannot deadlock.

        Found by an outside review of the diff. Two copies of a loop is how
        that happens, so there is one.
        """
        run = self._active
        if run is None or run.finished or self._resuming:
            return
        self._resuming = True
        try:
            while True:
                try:
                    if throw is not None:
                        effect, throw = run.gen.throw(throw), None
                    else:
                        effect = run.gen.send(value)
                except StopIteration as stop:
                    self._finish(run, "done", getattr(stop, "value", None))
                    return
                except Cancelled:
                    self._finish(run, "cancelled", None)
                    return
                except Exception as exc:
                    import traceback
                    self._log(f"{run.job.name} FAILED: {exc!r}")
                    self._log(traceback.format_exc())
                    self._finish(run, f"failed:{exc!r}", None)
                    return
                value = None
                try:
                    done, value = self._dispatch(run, effect)
                except EffectFailed as exc:
                    # Handed back INTO the routine unchanged, so its finally
                    # blocks run and it reports the real reason rather than
                    # a nested repr of one. Round the loop, not into a
                    # second function.
                    throw = exc
                    continue
                except Exception as exc:
                    throw = EffectFailed(repr(exc))
                    continue
                if not done:
                    return
        finally:
            self._resuming = False
            self._drain_next()

    def _dispatch(self, run: _Run, effect: Effect) -> Tuple[bool, Any]:
        """Carry out one effect. Returns (resume-immediately, value)."""

        if isinstance(effect, Log):
            self._log(f"{run.job.name}: {effect.text}")
            return True, None

        if isinstance(effect, Call):
            return True, effect.fn()

        if isinstance(effect, Sleep):
            self._arm(run, "sleep", max(0.0, effect.ms))
            return False, None

        if isinstance(effect, Paint):
            if self._sink is None or not hasattr(self._sink, "request_paint"):
                # No painter: fall back to a short sleep rather than
                # pretending the screen is up to date. A step that quietly
                # does nothing is worse than one that fails (RULES C1).
                self._arm(run, "sleep", 60.0 * max(1, effect.frames))
                return False, None
            self._arm(run, "paint", effect.timeout_ms)
            self._sink.request_paint(effect.frames)
            return False, None

        if isinstance(effect, Still):
            if self._sink is None:
                raise EffectFailed("no camera to photograph with")
            # A UNIQUE TAG PER REQUEST, so a frame can only answer the
            # question it was asked for.
            #
            # Matching on the routine's tag alone is enough only ACROSS
            # runs. Within one run, a `Still("probe_lit")` that timed out,
            # followed by another `Still("probe_lit")`, would accept the
            # first frame as the answer to the second question -- and
            # `overlaps` stayed empty, so the oracle could not see it. That
            # is `_request_still`'s bare string tag, reproduced inside the
            # class written to replace it. Found by an outside review.
            #
            # Done by making the tag unique rather than by threading a
            # sequence number through the `still_ready` signal: the signal
            # is the real camera's, the fake camera's and the app's routing
            # all at once, and a tag suffix needs none of them to change.
            self._still_seq += 1
            run.still_tag = f"{effect.tag}#{self._still_seq}"
            self._arm(run, "still", effect.timeout_ms)
            self._sink.request_still(run.still_tag, effect.frames)
            return False, None

        if isinstance(effect, Floor):
            if self._sink is None:
                raise EffectFailed("no camera to measure with")
            self._arm(run, "floor", effect.timeout_ms)
            if effect.seconds is None:
                self._sink.start_noise_calibration()
            else:
                self._sink.start_noise_calibration(effect.seconds)
            return False, None

        if isinstance(effect, Room):
            if self._sink is None:
                raise EffectFailed("no camera to measure with")
            self._arm(run, "room", effect.timeout_ms)
            self._sink.start_room_measurement(effect.seconds)
            return False, None

        if isinstance(effect, Control):
            if self._sink is None:
                raise EffectFailed("no camera to write to")
            # ASK THE CAMERA OBJECT what it was last told, rather than
            # keeping a second cache here.
            #
            # The Director used to keep `_written`, and it went stale the
            # moment a preset was written around it -- which happens on
            # every launch, including from inside these routines via
            # `Call(lab._apply_preset(...))`. A stale cache makes the next
            # identical write be SKIPPED while the camera sits somewhere
            # else entirely, which is strictly worse than writing twice on
            # a device where writes are the hazard. Found by an outside
            # review; two caches for one fact is the bug, not the drift.
            # No hasattr guard: every sink answers this, including the
            # test stub and the fake camera. A guard here would silently
            # disable de-duplication for any sink that forgot it, which is
            # the behaviour that hides the bug (RULES C12).
            prev = self._sink.written_value(effect.key)
            if prev is not None and abs(prev - float(effect.value)) < 1e-9:
                # Measured: an auto-range ladder re-wrote the same exposure
                # on every rung, seven redundant UVC transactions a launch
                # on a device where one write has hung for 272 seconds.
                self.control_writes_skipped += 1
                return True, None
            self.control_writes += 1
            run.control_writes += 1
            self._sink.set_control(effect.key, float(effect.value))
            self._arm(run, "sleep", max(0.0, effect.spacing_ms))
            return False, None

        raise EffectFailed(f"unknown effect {effect!r}")

    def _arm(self, run: _Run, kind: str, ms: float) -> None:
        """Every wait gets a deadline, whatever the routine asked for.

        He had to press Escape to get an answer out of an unbounded wait
        once (RULES D4), and a fullscreen cover whose removal depended on a
        camera frame left the display an immovable black rectangle (B6).
        Neither is possible if there is no way to wait without a timer.
        """
        self._disarm(run)
        run.waiting_for = kind
        run.deadline = Deadline.after(self._clock, ms)
        seq = run.seq

        def fire() -> None:
            cur = self._active
            if cur is None or cur.seq != seq or cur.finished:
                return
            cur.timer = None
            if kind == "sleep":
                cur.waiting_for = ""
                self._resume(None)
                return
            what = cur.waiting_for or kind
            cur.waiting_for = ""
            self._log(f"{cur.job.name}: the {what} did not arrive within "
                      f"{ms:.0f}ms -- abandoning this step rather than "
                      f"waiting behind a cover that cannot be dismissed")
            # Resumed with None, not cancelled: the routine decides what a
            # stall means for it, and its finally blocks still run.
            self._resume(None)

        run.timer = self._clock.after(ms, fire)

    def _disarm(self, run: _Run) -> None:
        if run.timer is not None:
            self._clock.cancel(run.timer)
            run.timer = None
        run.waiting_for = ""
        run.deadline = None

    def _finish(self, run: _Run, outcome: str, value: Any) -> None:
        if run.finished:
            # ONCE. A `cancel()` reached from inside one of the routine's own
            # `Call` effects finished the same run twice: history recorded
            # `cancelled` and then `done`, and `ran()` returned True for a
            # routine that had been stood down -- so a cancelled run could
            # be reported as a completed one. Reachable today because
            # `_publish_verdict` calls `processEvents()` inside a `Call`.
            return
        self._disarm(run)
        run.finished = True
        ms = (self._clock.now() - run.started_at) * 1000.0
        self.history.append((run.job.name, outcome, ms))
        if self._active is run:
            self._active = None
        self._on_change()
        if outcome == "done":
            self._log(f"{run.job.name}: finished in {ms / 1000.0:.1f}s"
                      + (f" ({run.control_writes} control writes)"
                         if run.control_writes else ""))
        # Deliberately NOT started from here: `_finish` runs inside
        # `_resume`, and starting the next routine from in here created its
        # generator and then never advanced it, because `start()` calls
        # `_resume()`, which returns at once while `_resuming` is True.
        # test_director section 11 caught it: the queued job logged as
        # started and did nothing -- a step that quietly does nothing
        # (RULES C1), produced by the code written to stop that.
        self._want_next = True
        # Not while closing: see `close()`. A completion callback during
        # teardown is how cancelling the probe would start the self test.
        if run.job.on_done is not None and not self._closing:
            self._after.append((run.job, value if outcome == "done" else None))

    def poke(self) -> None:
        """Try to start a queued routine. Called from the app's own tick.

        A routine queued behind something the Director does NOT run has
        nothing to wake it: `_next()` only runs when a Director routine
        finishes, so when the outside owner finished the queue just sat
        there. That is a step that quietly does nothing, which is the
        failure mode this project has shipped four times (RULES C1), so it
        gets an explicit, visible drain rather than a hope.
        """
        if self._resuming:
            # Mid-step. `_drain_next` already refuses here and runs on the
            # way out of the resume, so the queue is not forgotten -- and
            # `start()` now refuses too, so this is the second of two
            # independent defences rather than the only one.
            return
        if self._active is not None and not self._active.finished:
            return
        if not self._queue:
            return
        self._next()

    def _drain_next(self) -> None:
        """Run the completion callbacks, then start whatever was queued.

        Both happen OUTSIDE the resume, so a callback that starts another
        routine finds a slot that is genuinely free rather than one that is
        about to be. Starting a job from inside a resume is exactly the bug
        test_director section 11 caught.
        """
        if self._resuming:
            return
        while self._after:
            job, value = self._after.pop(0)
            try:
                job.on_done(value)              # type: ignore[misc]
            except Exception as exc:
                self._log(f"{job.name}: its completion handler raised {exc!r}")
        if not self._want_next:
            return
        self._want_next = False
        self._next()

    def _next(self) -> None:
        """Start the first queued routine that can run. Never re-queues.

        `_dequeuing` matters: without it, `start()` refusing a queued job
        appends it to the queue again, and the `while` below pops it
        straight back -- an infinite loop inside the class whose job is to
        stop unbounded retries (RULES D10). Caught while adding `poke()`,
        by reading the two together rather than by running them.
        """
        if self._dequeuing:
            return
        self._dequeuing = True
        try:
            held: List[Job] = []
            while self._queue:
                job = self._queue.popleft()
                if self.start(job):
                    break
                held.append(job)
            # Whatever could not start keeps its place, in order.
            for job in reversed(held):
                self._queue.appendleft(job)
        finally:
            self._dequeuing = False

    def _refuse(self, job: Job, holder: str, why: str) -> None:
        self.refusals.append((job.name, holder))
        if self._dequeuing:
            # A queued job that still cannot start keeps its place and says
            # nothing. The app calls `poke()` from its own HUD tick, about
            # 12 times a second, so logging here put roughly four hundred
            # identical refusals into the diagnostic record while a routine
            # waited out 33 seconds of pattern calibration -- and that log
            # is the only account of what happened during an unattended
            # launch. It was already announced once, when it was queued.
            return
        self._log(f"{job.name}: not starting -- {why} ({holder} has the screen)")

    def _log(self, text: str) -> None:
        try:
            self._log_fn(text)
        except Exception:
            pass

    # -- for tests and the GATE line --------------------------------------

    def summary(self) -> str:
        parts = [f"{n}={o.split(':')[0]}" for n, o, _ms in self.history]
        return " ".join(parts)

    def ran(self, name: str) -> bool:
        return any(n == name and o == "done" for n, o, _ms in self.history)
