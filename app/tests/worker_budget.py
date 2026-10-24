"""One place the mutation scripts get their worker count and their run slot.

THIS EXISTS BECAUSE IT HAPPENED. 2026-10-19, mid-morning: `run_tests.bat`
reached `tests/mutate_detector.py`, which fans out to eight workers by
itself, each worker running a whole suite in a subprocess. Measured on his
machine: **36 python processes at 100% CPU across all 16 logical cores.**
He had to kill the process tree by hand, and his words afterwards were "I
had to kill your python process, it completely hosed my PC".

It had happened once before, on 2026-10-17: two agents each spawning
parallel mutation workers took his PC to 91% across 16 cores and left it
"almost nonresponsive".

Both times the fix offered was care. Care does not survive the next script.
His instruction was specific -- a cap the runner READS, not one anybody
promises to respect -- so this module is the only place the number lives,
and `tests/test_rules.py` fails the build if a `mutate_*.py` computes its
own.

TWO SEPARATE FAILURES, AND ONE CAP DOES NOT COVER BOTH:

- **Fan-out inside one script.** `mutate_figures.py` asked for up to
  EIGHTEEN workers, `mutate_detector.py` for eight. `workers()` answers
  that question now, and answers it low.
- **Two scripts at once.** The 2026-10-17 incident was two scripts, each
  politely capped, run simultaneously by two agents. A per-script cap
  cannot see that, so `exclusive()` is a lock across PROCESSES.

WHY IT REFUSES RATHER THAN WAITS. An unbounded wait is the thing this
project has paid for most often -- a cover with no deadline needed a
camera unplugged to escape it, and `Director` exists because fifteen
routines each waited on something with no bound. A second mutation run
says so and exits non-zero. RULES A10: an instrument that damages what it
measures reports noise, and a queued one reports it later.

NEVER `os.kill(pid, 0)` ON WINDOWS. Python maps `os.kill` to
TerminateProcess for any signal that is not a CTRL_* event, so the
idiomatic liveness probe KILLS THE PROCESS IT IS ASKING ABOUT. The
staleness check below opens a read-only handle instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "tests" / "_out" / "locks"

#: Never more than this, whatever a script asks for. His machine has 16
#: logical cores and the point is that it stays usable while a suite runs.
HARD_CEILING = 4

#: The share of the machine a test run may take when nobody has said
#: otherwise. A quarter of 16 cores is 4, which the ceiling then allows;
#: on a 4-core machine it is 1, which is the right answer there.
CORE_SHARE = 4

ENV = "LSA_TEST_WORKERS"


def workers(ceiling: int = HARD_CEILING) -> int:
    """How many parallel workers a test script may use.

    `ceiling` is what the CALLER believes it can use; it is honoured only
    as an upper bound, never as a request. An explicit `LSA_TEST_WORKERS`
    is a human decision and overrides the share, but is still clamped to
    the machine -- more workers than cores is slower, not faster, and it
    is how a machine stops responding.
    """
    cores = os.cpu_count() or 4
    want = max(1, cores // CORE_SHARE)
    override = os.environ.get(ENV, "").strip()
    if override:
        try:
            want = max(1, min(cores, int(override)))
        except ValueError:
            pass                 # a typo must not silently uncap the run
        else:
            return max(1, min(int(ceiling), want, cores))
    return max(1, min(int(ceiling), want, HARD_CEILING))


def _alive(pid: int) -> bool:
    """Is that process still running? Read-only, and on Windows it must be.

    `os.kill(pid, 0)` TERMINATES the process on Windows (see the module
    docstring), so this opens a query handle and asks for an exit code.
    A handle that cannot be opened is read as dead, which is the safe
    direction: the worst case is that a live run's lock is taken over,
    where the other direction blocks every future run for ever.
    """
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True
    import ctypes
    from ctypes import wintypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(wintypes.HANDLE(h), ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(wintypes.HANDLE(h))


class Busy(RuntimeError):
    """Another run holds the slot. Reported, never waited out."""


class exclusive:
    """One mutation run at a time, across processes.

    Used as a context manager. Raises `Busy` immediately if a LIVE process
    holds the slot; takes the slot over if the holder is gone, because this
    project force-kills things and a lock nobody can clear would block
    every run after the first kill.
    """

    def __init__(self, name: str = "mutation") -> None:
        self.name = name
        self.path = LOCKS / f"{name}.lock"
        self._taken = False

    def holder(self) -> "int | None":
        """The live PID holding the slot, or None."""
        try:
            pid = int(self.path.read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            return None
        return pid if _alive(pid) else None

    def __enter__(self) -> "exclusive":
        LOCKS.mkdir(parents=True, exist_ok=True)
        pid = self.holder()
        if pid is not None:
            raise Busy(
                f"another {self.name} run is already going (pid {pid}). "
                f"These spawn a suite per worker; two at once took this "
                f"machine to 91% across 16 cores on 2026-10-17. Wait for it, "
                f"or stop it, and run this one after.")
        self.path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        self._taken = True
        return self

    def __exit__(self, *exc) -> bool:
        if self._taken:
            try:
                if self.holder() in (None, os.getpid()):
                    self.path.unlink(missing_ok=True)
            except OSError:
                pass            # a lock we cannot clear goes stale by PID
        return False
