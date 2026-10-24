"""Arcade modes, driven by synthetic shots. No camera, no display.

A mode is the part of the app a person actually plays, so the things worth
testing are the ones that would make it feel broken or unfair rather than the
ones that would make it crash: targets that overlap so a hit is ambiguous, a
ball that escapes the screen or teleports after a stalled frame, a difficulty
curve that does not actually curve, and targets placed where the camera cannot
see them.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2                                                         # noqa: E402
from dataclasses import replace as dataclass_replace              # noqa: E402
from PySide6.QtCore import QRect                                   # noqa: E402
from PySide6.QtWidgets import QApplication                         # noqa: E402

import lsa.modes as modes                                          # noqa: E402
from lsa.settings import DEFAULT_MAGAZINE as _DEFAULT_MAG           # noqa: E402
# THE HOLD (2026-10-20): a side that has just fired its last round stays painted for
# `ModeHost.HOLD_S` before the round closes. The checks below are about what happens AT
# the close (verdicts, swaps, quiet halves), not about the hold, so they run with it off;
# section 39 and the hold checks in test_ui_smoke use the REAL value, kept here.
from lsa.modes.base import ModeHost as _MH0                          # noqa: E402
REAL_HOLD_S = _MH0.HOLD_S
_MH0.HOLD_S = 0.0

from lsa.display.scene import RenderContext                        # noqa: E402
from lsa.display.session import SessionStats                       # noqa: E402
from lsa.modes.ball import EscalatingBall                          # noqa: E402
from lsa.modes.circles import ClearTheBoard                        # noqa: E402

ROOT_SRC = Path(__file__).resolve().parents[1] / "src"

from figure_helpers import SW, SH, context                         # noqa: E402


from figure_helpers import (FIGURE_MODES, drive_to_live as _drive_to_live,   # noqa: E402
                           figure_hit_case as _figure_hit_case,
                           figure_states as _figure_states, full as _fh_full,
                           zone_point as _zone_point)


def main() -> int:
    QApplication.instance() or QApplication([])
    checks: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    print("\n1. Modes are discoverable without being imported by name")
    # THE PROPERTY IS THAT IMPORTING THE PACKAGE REGISTERS THEM, not which
    # ones exist. This pinned a list of nine names, so adding the animated
    # figure (key A, 2026-10-18) failed the build for a correct change --
    # the same shape as the twelve-figure counts in test_artwork. Every
    # module in the package must register exactly one mode, and the names
    # must be unique; the count follows from the folder.
    _names = sorted(m.name for m in modes.available())
    check("every built-in mode registers, one per module, with unique names",
          len(_names) == len(set(_names)) and len(_names) >= 9
          and all(isinstance(n, str) and n for n in _names),
          str(_names))
    check("an unknown name returns nothing", modes.by_name("nope") is None)

    print("\n2. Clear the board: placement must be unambiguous")
    ctx = context()
    board = ClearTheBoard(count=12)
    board.start(ctx)
    check("asked for 12 targets, got 12", len(board.targets) == 12, f"{len(board.targets)}")
    inside = all(t.r <= t.x <= SW - t.r and t.r <= t.y <= SH - t.r for t in board.targets)
    check("none of them hangs off the screen", inside)
    worst = min((math.hypot(a.x - b.x, a.y - b.y) - a.r - b.r)
                for i, a in enumerate(board.targets) for b in board.targets[i + 1:])
    check("none of them overlaps another", worst > 0, f"closest gap {worst:.0f}px")

    print("\n3. Clear the board: hits, misses and the clock")
    t0 = board.targets[0]
    out = board.shot(modes.ShotEvent(t0.x, t0.y, 1.0, 1), ctx)
    check("a shot in the middle of a target hits", out.hit and out.points == 1)
    check("that target is now dead", not board.targets[0].alive)
    again = board.shot(modes.ShotEvent(t0.x, t0.y, 1.1, 2), ctx)
    check("shooting the corpse does not score twice", not again.hit)
    # Deliberately just outside the ring: generosity here inflates every hit
    # rate the app reports, so it must be a miss.
    edge = board.shot(modes.ShotEvent(t0.x + t0.r * 1.3, t0.y, 1.2, 3), ctx)
    check("a shot just outside the ring is a miss", not edge.hit)
    check("misses are counted as shots", board.shots == 3 and board.hits == 1,
          f"{board.hits}/{board.shots}")

    for i, t in enumerate(board.targets):
        if t.alive:
            board.shot(modes.ShotEvent(t.x, t.y, 2.0 + i * 0.1, 10 + i), ctx)
    check("clearing every target ends the round", board.cleared_at is not None)
    check("and records how long it took", board.best_s is not None and board.best_s > 0,
          f"{board.best_s:.2f}s" if board.best_s else "none")

    print("\n4. Targets never go where the camera cannot see")
    # The real case: an off-axis camera that cannot see the bottom strip.
    visible = QRect(0, 0, SW, int(SH * 0.7))
    ctx_clipped = context(playable=visible)
    clipped = ClearTheBoard(count=12)
    clipped.start(ctx_clipped)
    below = [t for t in clipped.targets if t.y + t.r > visible.bottom() + 1]
    check("no target is placed in the blind strip", not below,
          f"{len(below)} of {len(clipped.targets)} were")

    print("\n5. Escalating ball: the curve actually escalates")
    ctx = context()
    ball = EscalatingBall()
    ball.start(ctx)
    speeds = [math.hypot(ball.vx, ball.vy)]
    for n in range(4):
        ball.shot(modes.ShotEvent(ball.x, ball.y, float(n), n + 1), ctx)
        speeds.append(math.hypot(ball.vx, ball.vy))
    ratios = [b / a for a, b in zip(speeds, speeds[1:])]
    check("each hit makes the next ball 5% faster",
          all(abs(r - 1.05) < 0.01 for r in ratios),
          " ".join(f"{r:.3f}" for r in ratios))
    # THE CAP: five levels, +5% each, and no further (owner, 2026-10-16: the
    # ball moved too fast for the detection hardware).
    for n in range(10, 16):
        ball.shot(modes.ShotEvent(ball.x, ball.y, float(n), n + 1), ctx)
    v_top = math.hypot(ball.vx, ball.vy)
    check("escalation stops at level 5", ball.level == 5 == ball.MAX_LEVEL,
          f"level {ball.level}")
    check("and the top speed is 1.05^4 of the start, not more",
          abs(v_top / speeds[0] - 1.05 ** 4) < 0.01,
          f"{v_top / speeds[0]:.4f} vs {1.05 ** 4:.4f}")
    check("the ball's size does not change with level (small ball was unhittable)",
          abs(ball.r - EscalatingBall.RADIUS_FRAC * min(ball.playfield(ctx).width(),
                                       ball.playfield(ctx).height())) < 1.0,
          f"r={ball.r:.1f}")
    check("level tracks the number of hits (to the cap)", ball.level == 5, f"level {ball.level}")

    print("\n6. Escalating ball: it stays on the screen")
    ctx = context()
    ball = EscalatingBall()
    ball.start(ctx)
    ball.level = 20          # ~6x start speed: the regime where bugs appear
    ball._spawn(ctx, away_from=None)
    escaped = 0
    for i in range(4000):
        ball.advance(context(now=i / 60.0))
        if not (0 <= ball.x <= SW and 0 <= ball.y <= SH):
            escaped += 1
    check("a fast ball never leaves the screen", escaped == 0,
          f"{escaped} frames outside")

    # A stalled frame (GC pause, a slow repaint) must not teleport it across the
    # screen, which would make a level unwinnable through no fault of the shooter.
    before = (ball.x, ball.y)
    ball.advance(context(now=99.0, dt=2.0))
    moved = math.hypot(ball.x - before[0], ball.y - before[1])
    check("a stalled frame does not teleport it", moved < SW * 0.1, f"moved {moved:.0f}px")

    print("\n7. Escalating ball: a new ball is not handed to you")
    ctx = context()
    ball = EscalatingBall()
    ball.start(ctx)
    far = 0
    for n in range(6):
        hit_at = (ball.x, ball.y)
        ball.shot(modes.ShotEvent(ball.x, ball.y, float(n), n + 1), ctx)
        if math.hypot(ball.x - hit_at[0], ball.y - hit_at[1]) > SW * 0.25:
            far += 1
    check("it respawns well away from the last hit", far == 6, f"{far}/6")

    print("\n8. The host owns the lifecycle")
    ctx = context()
    host = modes.ModeHost()
    first = ClearTheBoard(count=6)
    host.set_mode(first, ctx)
    check("the host forwards shots to the active mode",
          host.shot(modes.ShotEvent(first.targets[0].x, first.targets[0].y, 1.0, 1), ctx).hit)
    second = EscalatingBall()
    host.set_mode(second, ctx)
    check("switching modes replaces the old one", host.mode is second)
    check("the host reports the active mode's status",
          "level" in host.status(), host.status())
    host.set_mode(None, ctx)
    check("with no mode, a shot is simply a miss",
          not host.shot(modes.ShotEvent(1.0, 1.0, 1.0, 1), ctx).hit)

    # ---------------------------------------------------------------------
    # Bouncing balls: several at once, each carrying its own difficulty.
    print("\n9. Bouncing balls: several targets, each with its own level")
    from lsa.modes.balls import BouncingBalls

    bb = BouncingBalls()
    bctx = context()
    bb.start(bctx)
    check("it starts with several balls", len(bb.balls) == bb.COUNT,
          f"{len(bb.balls)} balls")
    check("and they do not all begin in the same place",
          len({(round(b.x), round(b.y)) for b in bb.balls}) == bb.COUNT)

    # Each ball keeps its OWN level: hitting the easy one must not punish you
    # on the hard one. That is what makes this a drill rather than one
    # escalating target.
    target = bb.balls[0]
    other_level = bb.balls[1].level
    r0 = target.r
    speed0 = math.hypot(target.vx, target.vy)
    out = bb.shot(modes.ShotEvent(target.x, target.y, 1.0, 1), bctx)
    check("a shot inside a ball is a hit", out.hit)
    check("that ball levels up", target.level == 2, f"level {target.level}")
    check("the others are untouched", bb.balls[1].level == other_level)
    bb.advance(context(now=0.1))
    check("it comes back faster",
          math.hypot(target.vx, target.vy) > speed0 * 1.03,
          f"{speed0:.0f} -> {math.hypot(target.vx, target.vy):.0f} px/s")
    check("and smaller", target.r < r0 * 0.97, f"{r0:.1f} -> {target.r:.1f}px")

    check("a shot nowhere near is a miss",
          not bb.shot(modes.ShotEvent(-5000.0, -5000.0, 2.0, 2), bctx).hit)

    # Overlapping balls: the nearest takes the hit, not whichever happens to be
    # first in the list, which from the firing line would look arbitrary.
    bb.balls[0].x, bb.balls[0].y, bb.balls[0].r = 1000.0, 1000.0, 120.0
    bb.balls[1].x, bb.balls[1].y, bb.balls[1].r = 1060.0, 1000.0, 120.0
    near = bb.shot(modes.ShotEvent(1010.0, 1000.0, 3.0, 3), bctx)
    check("overlapping balls give the hit to the nearest",
          near.target_id == "ball0", str(near.target_id))

    # The two limits that keep it playable. A target smaller than the laser's
    # own dot, or faster than a 100ms pulse can be localised, is not a test of
    # marksmanship: it reads as the app ignoring good shots.
    print("\n10. Bouncing balls: it stays winnable")
    bb2 = BouncingBalls()
    bb2.start(bctx)
    hard = bb2.balls[0]
    hard.level += 60
    bb2._place(hard, bctx)          # so speed and size reflect the new level
    bb2.advance(context(now=1.0))
    pf = bb2.playfield(bctx)
    smallest = hard.r / min(pf.width(), pf.height())
    fastest = math.hypot(hard.vx, hard.vy) / pf.width()
    check("balls stop shrinking before the camera loses them",
          smallest >= bb2.MIN_RADIUS_FRAC * 0.99,
          f"{smallest * 100:.2f}% of the screen after 60 hits")
    check("and stop accelerating before a pulse cannot be placed",
          fastest <= bb2.MAX_SPEED * 1.01,
          f"{fastest:.2f} screen widths/sec")

    print("\n11. Bouncing balls: nothing escapes")
    bb3 = BouncingBalls()
    bb3.start(bctx)
    escaped = 0
    for step in range(3600):
        dt = 1 / 60.0 if step % 97 else 0.9      # an occasional stalled frame
        bb3.advance(context(now=step / 60.0, dt=dt))
        pf3 = bb3.playfield(bctx)
        for ball in bb3.balls:
            if not (pf3.left() - 1 <= ball.x <= pf3.right() + 1
                    and pf3.top() - 1 <= ball.y <= pf3.bottom() + 1):
                escaped += 1
    check("no ball escapes, even after stalled frames", escaped == 0,
          f"{escaped} escapes in 60s")

    seen = QRect(400, 300, 2600, 1400)
    bb4 = BouncingBalls()
    bb4.start(context(playable=seen))
    for step in range(600):
        bb4.advance(context(now=step / 60.0, playable=seen))
    check("they stay where the camera can see them",
          all(seen.left() - 1 <= b.x <= seen.right() + 1
              and seen.top() - 1 <= b.y <= seen.bottom() + 1 for b in bb4.balls))
    check("the status line reports every ball", "levels" in bb.status(), bb.status())

    # ---------------------------------------------------------------------
    # Reaction: the only mode that measures the interval a shooter actually
    # trains hardest -- stimulus to first shot. Everything else here measures
    # working through targets that are already up.
    print(chr(10) + "12. Reaction: the clock starts when the target does")
    from lsa.modes.reaction import ReactionDrill

    r = ReactionDrill()
    r._rng.seed(5)
    rctx = context(now=0.0)
    r.start(rctx)
    check("it starts in the waiting state, with nothing to shoot at",
          r.state == "wait")

    # Firing before the target appears must never be scored as a hit, however
    # quickly the target then arrives: a shot that leaves early was a shot at
    # nothing, and counting it would reward anticipation -- the exact habit
    # this drill exists to break.
    early = r.shot(modes.ShotEvent(1000.0, 1000.0, 0.5, 1), rctx)
    check("firing before the target is a false start, not a hit",
          not early.hit and early.target_id == "false-start")
    check("and it is counted", r.false_starts == 1 and r.shots == 1)

    r.advance(context(now=r.next_at + 0.01))
    check("the target appears once the wait elapses", r.state == "live")
    appeared = (r.x, r.y)
    hit = r.shot(modes.ShotEvent(r.x, r.y, r.shown_at + 0.25, 2), rctx)
    check("a hit on the live target scores", hit.hit)
    check("and the time is measured from when it appeared",
          abs(r.times[-1] - 0.25) < 1e-6, f"{r.times[-1]:.3f}s")
    check("hitting it re-arms the wait", r.state == "wait")

    # The target must move, or the drill measures memory instead of reaction
    # and lets the shooter pre-aim during the wait.
    r.advance(context(now=r.next_at + 0.01))
    moved = math.hypot(r.x - appeared[0], r.y - appeared[1])
    field = r.playfield(rctx)
    check("the next target is well away from the last",
          moved >= r.MIN_MOVE_FRAC * math.hypot(field.width(),
                                                field.height()) * 0.9,
          f"moved {moved:.0f}px")

    slow = r.shot(modes.ShotEvent(r.x, r.y, r.shown_at + 1.5, 3), rctx)
    check("a slow hit scores less than a fast one",
          slow.hit and slow.points < hit.points,
          f"{slow.points} vs {hit.points}")
    check("the status reports a median, not just a best",
          "median" in r.status(), r.status())

    # A shot nowhere near the live target is an ordinary miss, not a false
    # start. One is a timing fault and the other an aiming fault, and telling
    # the shooter the wrong one is worse than telling them nothing.
    r.advance(context(now=r.next_at + 0.01))
    stray = r.shot(modes.ShotEvent(r.x + r.r * 4, r.y, r.shown_at + 0.2, 4), rctx)
    check("a miss while live is a miss, not a false start",
          not stray.hit and stray.target_id is None and r.false_starts == 1)

    # A static target must not repaint a 4K screen sixty times a second. This
    # mode first drew a running millisecond counter, which did exactly that --
    # the same fault found in the HUD today, 8.53 ms a paint against a
    # detector needing 6.96 ms of the same CPU per camera frame. It was also
    # the wrong drill: a counter is a number to watch instead of the target,
    # and a real shot timer reports the time after the shot.
    r3 = ReactionDrill()
    r3._rng.seed(9)
    r3.start(context(now=0.0))
    asked, t = 0, 0.0
    for _ in range(180):               # three seconds of waiting, then live
        t += 1 / 60.0
        if r3.advance(context(now=t)):
            asked += 1
    check("a waiting or static target asks for almost no repaints",
          asked <= 5, f"{asked} repaints in 180 ticks")
    check("but the time is still reported after the shot",
          "ms" in open(ROOT_SRC / "lsa" / "modes" / "reaction.py",
                       encoding="utf-8").read(),
          "the shooter has to be told how fast they were")


    # Section 13 was "Shoot / no-shoot: the penalty has to outweigh the
    # reward". The drill was REMOVED on 2026-10-18 at his request -- "I do not
    # really understand what that is" -- and the modes after it moved up a
    # number. The checks went with it rather than being left to fail; the
    # module is in git history if it is ever wanted back.

    print(chr(10) + "14. Neither new mode puts a target where the camera is blind")
    seen = QRect(400, 300, 2600, 1400)
    clipped_ctx = context(playable=seen)
    r2 = ReactionDrill()
    r2._rng.seed(3)
    r2.start(clipped_ctx)
    r2.advance(context(now=r2.next_at + 0.01, playable=seen))
    check("a reaction target stays where the camera can see it",
          seen.contains(int(r2.x), int(r2.y)), f"({r2.x:.0f},{r2.y:.0f})")
    # ----------------------------------------------------------------------
    # WHAT MOVES MUST BE INVISIBLE TO THE DETECTOR.
    #
    # The house rule is red excess under 30, written for STATIC furniture --
    # which is safe for a reason that is not the number: the detector
    # subtracts a running background every frame, so anything that stays put
    # is absorbed within seconds and stops existing for it.
    #
    # A moving object defeats that by construction. It is never where the
    # background expects it, so its full red excess reaches the threshold on
    # every frame -- the exact description of a laser dot crossing a screen.
    #
    # Measured live on 2026-10-15: the Escalating Ball target was drawn at
    # red excess 24, legal under the house rule, and the app logged 272
    # detections in two minutes, 231 below peak 15, against his real shots
    # at 85-149. He fired about fifteen times. The app was shooting at its
    # own ball and playing a gunshot for every frame of it.
    # ----------------------------------------------------------------------
    # 2026-10-16: the hit test uses the shot's own moment, not "now".
    print()
    print("15a. the ball hit test rewinds to the moment of the shot")
    from lsa.modes.balls import BouncingBalls as _BB

    def _fast(mode_cls):
        m = mode_cls()
        c0 = context(now=100.0)
        m.start(c0)
        return m, c0

    for cls_name, cls_ in (("ball", EscalatingBall), ("balls", _BB)):
        m, c0 = _fast(cls_)
        b = m if cls_ is EscalatingBall else m.balls[0]
        # A ball mid-field heading right at a speed where 80 ms is more than
        # a radius (the limit case the owner met after many escalations).
        b.x, b.y = SW / 2.0, SH / 2.0
        b.vx, b.vy = b.r * 1.5 / 0.08, 0.0
        was = (b.x - b.vx * 0.08, b.y)
        shot_t = 100.0
        now_ctx = context(now=100.08)
        # Shot lands where the ball WAS 80 ms ago: more than a radius behind it.
        ev = modes.ShotEvent(was[0], was[1], shot_t, 1)
        gap = math.hypot(ev.x - b.x, ev.y - b.y)
        out = (m.shot(ev, now_ctx))
        check(f"{cls_name}: a shot where the ball was 80 ms ago is a HIT "
              f"(it has since moved {gap / b.r:.2f} radii)", out.hit and gap > b.r)
        m, c0 = _fast(cls_)
        b = m if cls_ is EscalatingBall else m.balls[0]
        b.x, b.y, b.vx, b.vy = SW / 2.0, SH / 2.0, b.r * 1.5 / 0.08, 0.0
        far = modes.ShotEvent(b.x + b.r * 3.0, b.y, 100.0, 2)
        check(f"{cls_name}: a shot nowhere near the ball's past is still a miss",
              not m.shot(far, context(now=100.08)).hit)
        m, c0 = _fast(cls_)
        b = m if cls_ is EscalatingBall else m.balls[0]
        b.x, b.y, b.vx, b.vy = SW / 2.0, SH / 2.0, b.r * 1.5 / 0.08, 0.0
        stale = modes.ShotEvent(b.x - b.vx * 5.0, b.y, 95.0, 3)
        check(f"{cls_name}: a stale timestamp (5 s old) cannot teleport the test",
              not m.shot(stale, context(now=100.0)).hit)
    # The rewind folds through a wall: a ball that bounced since the shot.
    mfold, _ = _fast(EscalatingBall)
    fld = mfold.playfield(context())
    inner = fld.adjusted(mfold.r, mfold.r, -mfold.r, -mfold.r)
    mfold.x, mfold.y = inner.left() + 10.0, SH / 2.0
    mfold.vx, mfold.vy = 400.0, 0.0          # just bounced off the left wall
    px_then = mfold._position_at(0.1, context())[0]
    check("a bounce between the shot and now is unfolded correctly",
          abs(px_then - (inner.left() + 30.0)) < 1.0, f"{px_then - inner.left():.1f}")

    # The multi-ball mode has the same cap: five levels, +5% speed, -5% size.
    mcap, cc = _fast(_BB)
    speed1 = math.hypot(mcap.balls[0].vx, mcap.balls[0].vy)
    r1 = mcap.balls[0].r
    for n in range(12):
        bb0 = mcap.balls[0]
        mcap.shot(modes.ShotEvent(bb0.x, bb0.y, 100.0, n + 1), context(now=100.0))
    top = mcap.balls[0]
    check("balls: escalation stops at level 5", top.level == 5 == _BB.MAX_LEVEL,
          f"level {top.level}")
    check("balls: top speed is 1.05^4 of the start, size 0.95^4",
          abs(math.hypot(top.vx, top.vy) / speed1 - 1.05 ** 4) < 0.02
          and abs(top.r / r1 - 0.95 ** 4) < 0.02,
          f"speed x{math.hypot(top.vx, top.vy) / speed1:.3f}, size x{top.r / r1:.3f}")

    print()
    print("15b. level 5 is hittable: the arithmetic")
    for cls_name, cls_ in (("ball", EscalatingBall), ("balls", _BB)):
        m, _c = _fast(cls_)
        b = m if cls_ is EscalatingBall else m.balls[0]
        if cls_ is EscalatingBall:
            m.level = 5
            m._spawn(context(), None)
        else:
            b.level = 5
            m._place(b, context(), None)
        speed = math.hypot(b.vx, b.vy)
        pulse_100 = speed * 0.100
        check(f"{cls_name}: at level 5 a full 100 ms pulse moves the ball under half its radius",
              pulse_100 < 0.5 * b.r,
              f"{speed:.0f} px/s -> {pulse_100:.0f}px in 100 ms, r={b.r:.0f}px")

    print()
    print("15c. the ball is scored where it is DRAWN: the outer half of the outline counts")
    # Plan 02 step 12. The ball is drawn as an outline centred on r, so its
    # visible edge is r + stroke/2; the hit test used to stop at r, and a shot
    # on the outer half of the line the shooter was looking at scored a miss.
    from PySide6.QtGui import QImage as _QIc, QPainter as _QPc

    class _PenLog(_QPc):
        widths: list = []

        def setPen(self, *a, **k):
            for x in a:
                if hasattr(x, "widthF"):
                    _PenLog.widths.append(x.widthF())
            return super().setPen(*a, **k)

    def _pinned(cls_):
        m, _c = _fast(cls_)
        b = m if cls_ is EscalatingBall else m.balls[0]
        b.x, b.y, b.vx, b.vy = SW / 2.0, SH / 2.0, 0.0, 0.0
        return m, b

    for cls_name, cls_ in (("ball", EscalatingBall), ("balls", _BB)):
        m, b = _pinned(cls_)
        stroke = m.ring_stroke(b.r)
        check(f"{cls_name}: stroke is {stroke / b.r:.3f} of the radius "
              f"({stroke:.1f}px on r={b.r:.0f}px)",
              abs(stroke / b.r - 0.08) < 1e-9)
        # The width that is DRAWN is the width that is SCORED.
        _PenLog.widths = []
        im = _QIc(SW, SH, _QIc.Format_RGB32)
        im.fill(0xFF0A0A0A)
        pr = _PenLog(im)
        m.paint(pr, context(now=100.0))
        pr.end()
        drawn = [m.ring_stroke(x.r) for x in ([m] if cls_ is EscalatingBall else m.balls)]
        check(f"{cls_name}: the outline is drawn at the width the hit test uses",
              all(any(abs(w - d) < 1e-9 for w in _PenLog.widths) for d in drawn),
              f"drawn widths {sorted(set(round(w, 2) for w in _PenLog.widths))}, "
              f"scored widths {sorted(set(round(d, 2) for d in drawn))}")
        edge = b.r + stroke / 2.0
        check(f"{cls_name}: hit radius is r + stroke/2 = {edge:.1f}px",
              abs(m.ring_hit_radius(b.r) - edge) < 1e-9)
        for label, dx, want in (("on the outer half of the stroke", edge - 1.0, True),
                                ("1px past the visible edge", edge + 1.0, False)):
            m, b = _pinned(cls_)
            r_before = b.r              # a hit respawns the ball at another size
            out = m.shot(modes.ShotEvent(b.x + dx, b.y, 100.0, 1), context(now=100.0))
            old_rule = dx <= r_before
            check(f"{cls_name}: a shot {label} ({dx:.1f}px from centre, r={r_before:.1f}) "
                  f"is a {'hit' if want else 'miss'}  [old rule: "
                  f"{'hit' if old_rule else 'miss'}]", out.hit is want)
        # And the stroke floor: a small ball (pen at its 3px minimum) still
        # scores to the edge it draws.
        m, b = _pinned(cls_)
        b.r = 20.0
        check(f"{cls_name}: a small ball's stroke floors at {m.RING_STROKE_MIN_PX:g}px "
              f"and the hit radius follows",
              abs(m.ring_hit_radius(20.0) - 21.5) < 1e-9, f"{m.ring_hit_radius(20.0):.2f}")

    print()
    print("15d. circles and shoot/no-shoot are scored where they are DRAWN, in their own shape")
    # The ball's fault (15c) stood in two more modes. circles: ring drawn with a
    # 0.09 r stroke, hit-tested at r. discriminate: a shoot ring drawn with a
    # 0.18 r stroke (rim overhanging the test by 9% of r), and -- pointing the
    # WRONG way -- a no-shoot SQUARE drawn with half-width 0.75 r + stroke and
    # hit-tested as a CIRCLE of radius r: a shot visibly outside the square took
    # the -30, and one visibly on its corner did not. The no-shoot test must
    # match the shape that is drawn, neither wider (a penalty pointed at the
    # shooter) nor narrower.
    from PySide6.QtCore import QRectF as _QRF15d

    class _ShapeLog(_PenLog):
        rects: list = []

        def drawRect(self, *a, **k):
            if a and isinstance(a[0], _QRF15d):
                _ShapeLog.rects.append((a[0].width(), self.pen().widthF()))
            return super().drawRect(*a, **k)

    def _drawn(mode_):
        _PenLog.widths, _ShapeLog.rects = [], []
        im_ = _QIc(SW, SH, _QIc.Format_RGB32)
        im_.fill(0xFF0A0A0A)
        pr_ = _ShapeLog(im_)
        mode_.paint(pr_, context(now=100.0))
        pr_.end()
        return list(_PenLog.widths), list(_ShapeLog.rects)

    # ---- circles ----------------------------------------------------------
    from lsa.modes.circles import _Target as _CT15d
    cbd = ClearTheBoard(count=1)
    cbd.start(context(now=100.0))
    cbd.targets = [_CT15d(1500.0, 1000.0, 100.0)]
    tgt = cbd.targets[0]
    stroke_c = cbd._stroke(tgt)
    widths_c, _ = _drawn(cbd)
    check(f"circles: the ring is DRAWN at the width the hit test uses "
          f"({stroke_c:.1f}px = {stroke_c / tgt.r:.2f} r)",
          any(abs(w - stroke_c) < 1e-9 for w in widths_c)
          and abs(stroke_c / tgt.r - 0.09) < 1e-9, f"drawn {sorted(set(round(w, 2) for w in widths_c))}")
    edge_c = tgt.r + stroke_c / 2.0
    check(f"circles: hit radius is r + stroke/2 = {edge_c:.1f}px",
          abs(cbd._hit_r(tgt) - edge_c) < 1e-9)
    for label, dx, want in (("on the outer half of the stroke", edge_c - 1.0, True),
                            ("1px past the visible edge", edge_c + 1.0, False)):
        cbd.targets = [_CT15d(1500.0, 1000.0, 100.0)]
        out = cbd.shot(modes.ShotEvent(1500.0 + dx, 1000.0, 100.0, 1), context(now=100.0))
        check(f"circles: a shot {label} ({dx:.1f}px from centre) is a "
              f"{'hit' if want else 'miss'}  [old rule at r: {'hit' if dx <= 100.0 else 'miss'}]",
              out.hit is want)
    cbd.targets = [_CT15d(1500.0, 1000.0, 100.0)]
    near_c = cbd.nearest(modes.ShotEvent(1500.0 + edge_c + 10.0, 1000.0, 100.0, 2), context(now=100.0))
    check("circles: nearest() reports the radius shot() tests (so 'outside' means outside)",
          near_c is not None and abs(near_c[2] - edge_c) < 1e-9
          and abs((near_c[1] - near_c[2]) - 10.0) < 1e-6, str(near_c))

    # ----------------------------------------------------------------------
    # Plan 02 step 10: every SHOT line says how far the shot was from the
    # nearest target. `nearest()` is log-only and read-only.
    # ----------------------------------------------------------------------
    print()

    print()
    print("15e. every mode COMPOSITED with the holes layer: red excess of what the camera sees")
    # The step-11 flash measures red excess 0 on its own. The camera sees the
    # COMPOSITE: the mode's hit/flash state and a fresh bullet hole drawn at the
    # same point at the same instant. This project's last phantom was found by
    # mapping detections back onto the screen, not by rendering one layer at a
    # time, and a legal colour fired twice because the check could not see
    # composition. So: render each mode in its hit state with a hole drawn at the
    # hit (flash still live), at full pixel resolution but over a 1400x1400 CROP of the
    # 3840x2160 surface centred on the hit (see `CROP` below for why a crop is enough),
    # and measure red excess per pixel.
    #
    # WHY THE NUMBER MATTERS: the saved sensitivity map amplifies the weakest
    # region of the screen up to 4x (floored at 0.25), so a composite red excess
    # of E can arrive at the detector as 4E there, against a bar of 8-17.
    # The hole palette WAS rim 5, halo 5, flash 8 and parked; since 2026-10-17 it is
    # exactly grey (red excess 0), like everything else the app draws, because a
    # legal 8 has already fired at him once through the 4x gain map. The numbers
    # are printed and ASSERTED to be 0, alone and composited with every mode.
    import numpy as _np15e
    from lsa.display.layers import BulletHoles as _BH15e
    GAIN_MAX = 4.0
    # Compositing two alpha-blended layers in 8 bits rounds R and G separately, so
    # a mode drawn over (or under) the hole can read ONE level above the hole's
    # own palette on a handful of pixels (measured: 2 px at 9 against 8). That is
    # quantisation, not colour. A warm mode colour adds many levels, which is
    # what this must catch -- shown below by painting one on purpose.
    COMPOSITE_ROUNDING = 1

    def _excess_stats(img):
        """(max red excess, pixels above 0, pixels above 4) over every pixel.
        uint8 saturating arithmetic in cv2 instead of an int16 copy of the
        whole 4-channel image: this ran ~200 times and was most of the suite."""
        bpl = img.bytesPerLine() // 4
        arr = _np15e.frombuffer(img.constBits(), _np15e.uint8).reshape(img.height(), bpl, 4)
        arr = arr[:, :img.width()]
        ex = cv2.subtract(arr[..., 2], cv2.max(arr[..., 1], arr[..., 0]))     # BGRA
        return int(ex.max()), int(cv2.countNonZero(ex)), int((ex > 4).sum())

    # WHAT THIS MEASURES NOW: a 1400x1400 CROP of the 3840x2160 surface (36% of its
    # area) centred on the hit, at full pixel resolution (the painter is translated,
    # nothing is scaled). It used to be the whole canvas; it is ~4x cheaper cropped.
    #
    # WHY A CROP IS SUFFICIENT: this check exists to see COMPOSITION -- red excess that
    # appears only where the hole's palette and the mode's own marks overlap -- and
    # both are drawn within a few hundred pixels of the hit (the hole's rim and halo,
    # the mode's flash ring and its label). Everything the mode draws FAR from the hit
    # (the whole figure, targets, hints) does not composite with the hole at all, so
    # its colour is measured where it matters: over the full 3840x2160 surface in the
    # state checks of section 15 and in sections 18-23 of `test_figures.py` (the
    # reviewer also verified full-screen that the figure adds nothing to the composite).
    # What the crop would NOT catch is a mode that draws something hole-coloured beyond
    # 700 px of its own hit; those sections would.
    CROP = 1400

    def _canvas():
        im_ = _QIc(CROP, CROP, _QIc.Format_RGB32)
        im_.fill(0xFF0A0A0A)
        return im_

    def _pc(img_, pt_):
        x0_ = min(max(int(pt_[0] - CROP / 2), 0), SW - CROP)
        y0_ = min(max(int(pt_[1] - CROP / 2), 0), SH - CROP)
        p_ = _QPc(img_)
        p_.translate(-x0_, -y0_)
        return p_

    def _hit_case(cls_):
        """(mode, time of the hit, hit point): the mode just struck at that point."""
        if cls_.name in FIGURE_MODES:
            return _figure_hit_case(cls_)
        m = cls_()
        m.start(context(now=40.0))
        t0 = 50.0
        if hasattr(m, "targets"):
            t = m.targets[0]
            pt = (t.x, t.y)
            m.shot(modes.ShotEvent(pt[0], pt[1], t0, 1), context(now=t0))
            return m, t0, pt
        if hasattr(m, "balls"):
            b = m.balls[0]
            pt = (b.x, b.y)
            m.shot(modes.ShotEvent(pt[0], pt[1], t0, 1), context(now=t0))
            return m, t0, pt
        if cls_.name == "reaction":
            m.advance(context(now=m.next_at + 0.01))
            pt = (m.x, m.y)
            t_hit = m.shown_at + 0.3
            m.shot(modes.ShotEvent(pt[0], pt[1], t_hit, 2), context(now=t_hit))
            return m, t_hit, pt
        if hasattr(m, "x") and hasattr(m, "vx"):                 # escalating ball
            pt = (m.x, m.y)
            m.shot(modes.ShotEvent(pt[0], pt[1], t0, 1), context(now=t0))
            return m, t0, pt
        pt = (SW / 2.0, SH / 2.0)                                # static targets
        m.shot(modes.ShotEvent(pt[0], pt[1], t0, 1), context(now=t0))
        return m, t0, pt

    # The hole's flash and the mode's flash both ANIMATE, so one instant proves
    # little: sweep the age of the hit across the whole flash, mode and hole
    # moving together (the same shot, so the same instant), and take the worst.
    AGES = (0.0, 0.03, 0.07, 0.11, 0.15, 0.19, 0.21)
    import types as _types15e
    from lsa.display import layers as _layers15e

    # BulletHoles reads the wall clock when it paints, so a hole `age` s old would
    # drift by milliseconds between the two renders being COMPARED and move the
    # flash's alpha. Freeze the layer module's clock for this section so the hole
    # alone and the composite see the identical instant (restored afterwards).
    _FROZEN_NOW = 1_000_000.0
    _light_ctx15e = context(now=50.0)
    _light_ctx15e.dark_background = False
    _real_time15e = _layers15e.time
    _layers15e.time = _types15e.SimpleNamespace(perf_counter=lambda: _FROZEN_NOW)

    def _hole_layer(pt, age):
        hl_ = _BH15e()
        hl_.add(pt[0], pt[1], 1, 8)
        hl_.holes[-1].born = _FROZEN_NOW - age                  # a hole `age` s old
        return hl_

    # the hole on its own: the reference every composite is read against
    hole_by_age = {}
    for age_ in AGES:
        hole_only = _canvas()
        hp = _pc(hole_only, (SW / 2.0, SH / 2.0))
        _hole_layer((SW / 2.0, SH / 2.0), age_).paint(hp, context(now=50.0 + age_))
        hp.end()
        hole_by_age[age_] = _excess_stats(hole_only)[0]
    hole_worst = max(hole_by_age.values())
    print(f"         hole alone, by age {dict((k, v) for k, v in hole_by_age.items())}: "
          f"worst red excess {hole_worst}")
    check("the bullet hole ALONE, at every age of its flash, is exactly grey (red excess 0)",
          hole_worst == 0, f"worst red excess {hole_worst}, by age {hole_by_age}")
    # ...and on a LIGHT background, where the palette switches to its dark variant
    _light_worst15e = 0
    for age_ in AGES:
        _lt = _QIc(CROP, CROP, _QIc.Format_RGB32)
        _lt.fill(0xFFC8C8C8)
        _hp = _pc(_lt, (SW / 2.0, SH / 2.0))
        _hp.setRenderHint(_QPc.Antialiasing)
        _hole_layer((SW / 2.0, SH / 2.0), age_).paint(_hp, _light_ctx15e)
        _hp.end()
        _light_worst15e = max(_light_worst15e, _excess_stats(_lt)[0])
    check("the bullet hole on a LIGHT background is exactly grey too (red excess 0)",
          _light_worst15e == 0, f"worst red excess {_light_worst15e}")
    worst_by_mode = {}
    for cls_ in modes.available():
        m_, t_hit, pt_ = _hit_case(cls_)
        mode_worst = comp_worst = comp_px = comp_over4 = 0
        adds = -99                     # most red the MODE adds to the hole, at any age
        for age_ in AGES:
            ctx_ = context(now=t_hit + age_)
            only = _canvas()
            pr_ = _pc(only, pt_)
            pr_.setRenderHint(_QPc.Antialiasing)
            m_.paint(pr_, ctx_)
            pr_.end()
            mode_worst = max(mode_worst, _excess_stats(only)[0])
            # The Scene paints layers in ascending z, so the order is DERIVED from
            # the layers (holes z=15 under the mode z=40), not assumed -- and the
            # reverse order is measured too, since a z change must not open a gap.
            hole_ = _hole_layer(pt_, age_)
            age_worst = 0
            for order in (sorted([hole_, m_], key=lambda la: la.z),
                          sorted([hole_, m_], key=lambda la: -la.z)):
                both = _canvas()
                pr_ = _pc(both, pt_)
                pr_.setRenderHint(_QPc.Antialiasing)
                for la_ in order:
                    la_.paint(pr_, ctx_)
                pr_.end()
                w_, px_, o4_ = _excess_stats(both)
                age_worst = max(age_worst, w_)
                if w_ > comp_worst or (w_ == comp_worst and px_ > comp_px):
                    comp_worst, comp_px, comp_over4 = w_, px_, o4_
            adds = max(adds, age_worst - hole_by_age[age_])
        worst_by_mode[cls_.name] = comp_worst
        check(f"{cls_.name}: the COMPOSITE of that mode and a hole at its hit is exactly grey "
              f"(red excess 0)", comp_worst == 0,
              f"worst composite red excess {comp_worst}, {comp_px} px above 0")
        print(f"         {cls_.name:13s} mode alone {mode_worst}, composite worst red excess "
              f"{comp_worst} ({comp_px} px above 0, {comp_over4} px above 4); "
              f"the mode adds {adds:+d} to the hole; "
              f"x{GAIN_MAX:g} gain map -> {comp_worst * GAIN_MAX:g}")
        check(f"{cls_.name}: the MODE alone, in its hit state, is exactly grey (red excess 0)",
              mode_worst == 0, f"{mode_worst}")
        check(f"{cls_.name}: composited with a hole at the hit, the mode adds no red to the "
              f"hole's own palette beyond {COMPOSITE_ROUNDING} level of 8-bit blend rounding, "
              f"at any age of the flash, in either z order (the weakest region amplifies "
              f"by up to {GAIN_MAX:g}x, so a composite of {comp_worst} could arrive "
              f"as {comp_worst * GAIN_MAX:g})",
              adds <= COMPOSITE_ROUNDING,
              f"the mode adds {adds:+d}: composite {comp_worst} vs hole alone {hole_worst}; "
              f"x{GAIN_MAX:g} gain map -> {comp_worst * GAIN_MAX:g}")
    print(f"         worst composite per mode: {worst_by_mode}")

    # A check must be able to fail (RULES C1): a mode that draws its hit ring in the
    # old warm accent (#d8c070, red excess 24) must be caught by this instrument.
    class _WarmMode:
        z = 40

        def paint(self, p_, c_):
            from PySide6.QtCore import QPointF as _QPF
            from PySide6.QtGui import QColor as _QCol, QPen as _QPen
            p_.setPen(_QPen(_QCol("#d8c070"), 6.0))
            p_.setBrush(_QCol(0, 0, 0, 0))
            p_.drawEllipse(_QPF(SW / 2.0, SH / 2.0), 60.0, 60.0)

    warm_adds = -99
    for age_ in AGES:
        both = _canvas()
        pr_ = _pc(both, (SW / 2.0, SH / 2.0))
        pr_.setRenderHint(_QPc.Antialiasing)
        for la_ in sorted([_hole_layer((SW / 2.0, SH / 2.0), age_), _WarmMode()],
                          key=lambda la: la.z):
            la_.paint(pr_, context(now=50.0 + age_))
        pr_.end()
        warm_adds = max(warm_adds, _excess_stats(both)[0] - hole_by_age[age_])
    check(f"the composition check can fail: a warm mode colour adds {warm_adds:+d} levels "
          f"(limit {COMPOSITE_ROUNDING})", warm_adds > COMPOSITE_ROUNDING, f"{warm_adds}")
    _layers15e.time = _real_time15e

    print()

    print("15. a moving target must be invisible to the detector")
    from lsa.display import layers as _L
    from PySide6.QtGui import QImage as _QI, QPainter as _QP
    check(f"there is a limit for moving things ({_L.MOVING_RED_LIMIT})",
          _L.MOVING_RED_LIMIT <= 10,
          "the bar sits near 10 on a measured floor of 2-5, so anything "
          "drawn must be below the FLOOR, not below the ceiling")
    W2, H2 = SW, SH
    # GREY, NOT MERELY LOW: owner, 2026-10-16, "draw everything in shades of
    # grey; no yellow". The saved sensitivity map amplifies the weakest
    # region fourfold, so 8 can arrive at the detector as 16-18. Red excess
    # must be exactly ZERO, in every STATE a mode can be in -- live,
    # a hit flash, a dying target, a cleared banner, a false start -- and
    # every element must still be visible against the dark target background.
    from lsa.display.layers import AimField as _Aim

    def _render(paint_fn, ctx_):
        """(worst red excess, sampled pixels with red > 40) over every third
        pixel of a 3840x2160 render. The per-pixel Python loop this replaced
        cost ~1 s a render; the numbers it reports are the same."""
        img = _QI(W2, H2, _QI.Format_RGB32)
        img.fill(0xFF0A0A0A)
        pnt = _QP(img)
        pnt.setRenderHint(_QP.Antialiasing)
        paint_fn(pnt, ctx_)
        pnt.end()
        a = _np15e.frombuffer(img.constBits(), _np15e.uint8).reshape(
            H2, img.bytesPerLine())[::3, :W2 * 4].reshape(H2 // 3 + (H2 % 3 > 0), W2, 4)
        a = a[:, ::3].astype(_np15e.int16)
        ex = a[..., 2] - _np15e.maximum(a[..., 1], a[..., 0])
        return int(ex.max()), int((a[..., 2] > 40).sum())

    for cls in modes.available():
        mode = cls()
        ctx2 = context()
        mode.start(ctx2)
        for _ in range(30):
            mode.advance(ctx2)
        states = [("live", ctx2)]
        # drive it into its other states
        t0 = 50.0
        if cls.name in FIGURE_MODES:
            states = _figure_states(mode, cls)
        elif hasattr(mode, "targets"):
            for t in list(mode.targets)[:3]:
                mode.shot(modes.ShotEvent(t.x, t.y, t0, 1), context(now=t0))
            states.append(("after hits", context(now=t0 + 0.05)))
            for t in list(mode.targets):
                mode.shot(modes.ShotEvent(t.x, t.y, t0 + 0.1, 2), context(now=t0 + 0.1))
            states.append(("cleared", context(now=t0 + 0.2)))
        elif hasattr(mode, "balls"):
            b0 = mode.balls[0]
            mode.shot(modes.ShotEvent(b0.x, b0.y, t0, 1), context(now=t0))
            states.append(("after hit", context(now=t0 + 0.05)))
        elif cls.name == "reaction":
            mode.shot(modes.ShotEvent(1.0, 1.0, t0, 1), context(now=t0))   # false start
            mode.advance(context(now=mode.next_at + 0.01))
            mode.shot(modes.ShotEvent(mode.x, mode.y, mode.shown_at + 0.3, 2),
                      context(now=mode.shown_at + 0.3))
            states.append(("after hit", context(now=mode.shown_at + 0.31)))
            mode.advance(context(now=mode.shown_at + 0.5))
        else:
            # static targets: a shot on the centre target
            mode.shot(modes.ShotEvent(SW / 2.0, SH / 2.0, t0, 1), context(now=t0))
            states.append(("after hit", context(now=t0 + 0.05)))
        for label, cx in states:
            worst, lit = _render(mode.paint, cx)
            check(f"{cls.name} ({label}): everything it draws is grey (red excess 0)",
                  worst <= 0, f"worst red excess {worst}")
            check(f"{cls.name} ({label}): and something is actually visible",
                  lit > 20, f"{lit} sampled pixels brighter than the background")
    # Sized as fractions of the screen, never pixels (RULES B5): the pen and
    # the ticks must scale with the screen height (42" monitor, 75" TV, projector).
    class _Rec(_QP):
        widths: list = []

        def setPen(self, *a, **k):
            for x in a:
                if hasattr(x, "widthF"):
                    _Rec.widths.append(x.widthF())
            return super().setPen(*a, **k)

    def _pen_for(h):
        _Rec.widths = []
        im = _QI(int(h * 16 / 9), h, _QI.Format_RGB32)
        pr = _Rec(im)
        _Aim().paint(pr, RenderContext(now=0.0, dt=0.0, rect=QRect(0, 0, int(h * 16 / 9), h),
                                       stats=SessionStats()))
        pr.end()
        return max(_Rec.widths) if _Rec.widths else 0.0

    w1080, w2160 = _pen_for(1080), _pen_for(2160)
    check("the static targets' line width scales with the screen, not pixels",
          w2160 >= 1.8 * w1080 > 0, f"{w1080:.1f}px at 1080p, {w2160:.1f}px at 2160p")
    aim = _Aim()
    worst, lit = _render(aim.paint, context())
    check("static targets: the marks are grey (red excess 0)", worst <= 0,
          f"worst red excess {worst}")
    check("static targets: and visible against the dark background", lit > 20,
          f"{lit} sampled pixels brighter than the background")

    # ----------------------------------------------------------------------
    # Plan 02 step 10: every SHOT line says how far the shot was from the
    # nearest target. `nearest()` is log-only and read-only.
    # ----------------------------------------------------------------------
    print()

    print("16. each mode with targets says how far a shot was from the nearest one")
    from lsa.modes.base import describe_nearest
    from lsa.modes.static import StaticTargets
    from lsa.modes.circles import _Target as _CircleTarget

    n_ctx = context(now=100.0)

    # static: the centre mark and four corner marks
    stt = StaticTargets()
    stt.start(n_ctx)
    cx0, cy0, radii0 = stt._field.marks(n_ctx)[0]
    R0 = radii0[-1]
    near = stt.nearest(modes.ShotEvent(cx0, cy0 + R0 + 40.0, 100.0, 1), n_ctx)
    check("static: 40px beyond the outer ring is the centre mark, 40px outside",
          near is not None and near[0] == "centre"
          and abs(near[1] - near[2] - 40.0) < 1e-6 and abs(near[2] - R0) < 1e-6,
          str(near))
    near = stt.nearest(modes.ShotEvent(cx0 + 41.0, cy0, 100.0, 2), n_ctx)
    check("static: a shot 41px from the centre is inside it (distance 41 < radius)",
          near is not None and near[0] == "centre" and abs(near[1] - 41.0) < 1e-6
          and near[1] < near[2], str(near))
    shots_before = stt.shots
    stt.nearest(modes.ShotEvent(cx0, cy0, 100.0, 3), n_ctx)
    check("static: asking changes nothing (read-only)",
          stt.shots == shots_before and not stt.hits_by_mark and not stt._flashes)

    # circles
    cb = ClearTheBoard(count=2)
    cb.start(n_ctx)
    cb.targets = [_CircleTarget(1000.0, 1000.0, 100.0), _CircleTarget(2500.0, 1500.0, 100.0)]
    near = cb.nearest(modes.ShotEvent(1123.0, 1000.0, 100.0, 1), n_ctx)
    want_r = cb._hit_r(cb.targets[0])               # r + half the drawn stroke
    check("circles: 23px past circle0's centre-line is (23 - stroke/2) outside its DRAWN edge",
          near is not None and near[0] == "circle0"
          and abs(near[2] - want_r) < 1e-9
          and abs((near[1] - near[2]) - (23.0 - (want_r - 100.0))) < 1e-6, str(near))
    cb.targets[0].alive = False
    near = cb.nearest(modes.ShotEvent(1123.0, 1000.0, 100.0, 2), n_ctx)
    check("circles: a dead target is not the nearest one", near[0] == "circle1", str(near))
    cb.targets[1].alive = False
    check("circles: with nothing alive there is nothing to measure",
          cb.nearest(modes.ShotEvent(1.0, 1.0, 100.0, 3), n_ctx) is None)

    # ball / balls: measured where `shot()` will test them (rewound)
    for cls_name, cls_ in (("ball", EscalatingBall), ("balls", _BB)):
        m, _c = _fast(cls_)
        b = m if cls_ is EscalatingBall else m.balls[0]
        for other in ([] if cls_ is EscalatingBall else m.balls[1:]):
            other.x, other.y = 100.0, 100.0         # out of the way
        b.x, b.y, b.vx, b.vy = SW / 2.0, SH / 2.0, 1000.0, 0.0
        was_x = b.x - 1000.0 * 0.08
        near = m.nearest(modes.ShotEvent(was_x, b.y, 100.0, 1), context(now=100.08))
        check(f"{cls_name}: a shot where the ball was 80 ms ago is ~0px from it, "
              f"not 80px", near is not None and near[1] < 1.0, str(near))
        check(f"{cls_name}: the radius reported is the one the hit test uses "
              f"(r + stroke/2 = {m.ring_hit_radius(b.r):.1f})",
              abs(near[2] - m.ring_hit_radius(b.r)) < 1e-6, f"{near[2]:.1f}")
        want_id = "ball" if cls_ is EscalatingBall else f"ball{b.ident}"
        check(f"{cls_name}: names the ball as the mode does", near[0] == want_id, near[0])
        # nearest() agrees with shot(): inside the reported radius is a hit
        out = m.shot(modes.ShotEvent(was_x, b.y, 100.0, 2), context(now=100.08))
        check(f"{cls_name}: and agrees with shot() about hit or miss",
              out.hit == (near[1] <= near[2]))

    # reaction has one transient target and the plan names no distance for it
    check("reaction: no distance is claimed (default is None)",
          ReactionDrill().nearest(modes.ShotEvent(1.0, 1.0, 1.0, 1), n_ctx) is None)

    # the words in the log
    mm = 940.0 / SW
    check("log words, a hit: d, mm, id and radius",
          describe_nearest(("centre", 41.0, 907.0), True, "centre", mm)
          == "(d=41px = 10.0mm from centre of centre, r=907px)",
          describe_nearest(("centre", 41.0, 907.0), True, "centre", mm))
    check("log words, a miss: how far OUTSIDE which target",
          describe_nearest(("circle3", 120.0, 97.0), False, None, mm)
          == "(nearest circle3: 23px = 5.6mm outside, r=97px)",
          describe_nearest(("circle3", 120.0, 97.0), False, None, mm))
    check("log words, a miss that is inside some target's edge says inside",
          "inside" in describe_nearest(("x", 10.0, 50.0), False, None, None))
    check("log words, a mode that cannot say adds nothing",
          describe_nearest(None, False, None, mm) == "")

    # ----------------------------------------------------------------------
    # Plan 02 step 11: a hit on the static targets is visible.
    # ----------------------------------------------------------------------
    print()
    print("17. static targets: a hit flashes the mark and is tallied beside it")
    import numpy as np
    from PySide6.QtGui import QImage as _QI3, QPainter as _QP3

    _full = _fh_full          # one definition, shared with test_figures.py

    LIT_MAX = 0.02            # RULES B1b: the lit-area budget while shooting
    sm = StaticTargets()
    sm.start(context(now=100.0))
    r_idle, lit_idle, img_idle = _full(sm.paint, context(now=100.0))
    check("idle: grey only (red excess 0)", r_idle == 0, f"worst {r_idle}")
    print(f"         idle lit fraction {lit_idle * 100:.3f}% (budget {LIT_MAX * 100:.0f}%)")

    # --- the repaint rule: idle asks for (almost) nothing --------------------
    asked = sum(1 for i in range(180) if sm.advance(context(now=100.0 + i / 60.0)))
    check("idle: at most 5 repaints requested in 180 ticks", asked <= 5,
          f"{asked} repaints in 180 ticks")
    check("idle: the animated flag says False, as advance() does", sm.animated is False)

    # --- a hit on the centre mark ---------------------------------------------
    marks_ = sm._field.marks(context())
    out = sm.shot(modes.ShotEvent(marks_[0][0], marks_[0][1], 200.0, 1), context(now=200.05))
    check("a shot on the centre scores and names the mark",
          out.hit and out.target_id == "centre" and out.points > 0)
    check("after a hit: the animated flag says True while the flash is live (it was a "
          "constant False while advance() returned True)", sm.animated is True)
    r_hit, lit_hit, img_hit = _full(sm.paint, context(now=200.05))
    changed = int((img_hit != img_idle).any(axis=2).sum())
    check("after a hit: still grey only (red excess 0)", r_hit == 0, f"worst {r_hit}")
    check("after a hit: something visibly changed on screen", changed > 2000,
          f"{changed} pixels differ from the idle render")
    print(f"         flash lit fraction {lit_hit * 100:.3f}%  "
          f"(+{(lit_hit - lit_idle) * 100:.3f} percentage points)")
    check("after a hit: the lit area stays inside the budget", lit_hit <= LIT_MAX,
          f"{lit_hit * 100:.3f}% against {LIT_MAX * 100:.0f}%")

    # --- the flash ends, the tally stays --------------------------------------
    asked = 0
    t_ = 200.05
    flag_lies = 0
    while True:
        flag_ = sm.animated
        more_ = sm.advance(context(now=t_))
        if more_ and not flag_:
            flag_lies += 1            # asked for a repaint while the flag said nothing animates
        if not more_:
            break
        asked += 1
        t_ += 1 / 60.0
        if asked > 120:
            break
    check("through the whole flash, advance() never asks for a repaint the flag denies",
          flag_lies == 0 and sm.animated is False, f"{flag_lies} disagreements")
    check("the flash stops asking for repaints by itself (at most FLASH_S + 1 tick)",
          asked <= int(sm.FLASH_S * 60) + 2 and not sm._flashes,
          f"{asked} repaints after the hit")
    asked_after = sum(1 for i in range(180) if sm.advance(context(now=t_ + 1 + i / 60.0)))
    check("afterwards it is idle again: 0 repaints in 180 ticks", asked_after == 0,
          f"{asked_after}")
    r_tal, lit_tal, img_tal = _full(sm.paint, context(now=t_ + 5.0))
    tally_px = int((img_tal != img_idle).any(axis=2).sum())
    check("the tally remains once the flash is gone", 50 < tally_px < 20000,
          f"{tally_px} pixels differ from idle")
    check("the tally is grey only (red excess 0)", r_tal == 0, f"worst {r_tal}")

    # --- every mark tallies, two flashes at most, all inside the budget -------
    sm2 = StaticTargets()
    sm2.start(context(now=100.0))
    for i, (mx, my, _r) in enumerate(marks_):
        sm2.shot(modes.ShotEvent(mx, my, 300.0, i + 1), context(now=300.0))
    check("each of the five marks keeps its own count",
          sorted(sm2.hits_by_mark) == ["centre", "mark1", "mark2", "mark3", "mark4"]
          and all(v == 1 for v in sm2.hits_by_mark.values()), str(sm2.hits_by_mark))
    check("no more than two rings flash at once (two pistols is the most)",
          len(sm2._flashes) == StaticTargets.MAX_FLASHES == 2, f"{len(sm2._flashes)}")
    # worst case: every mark tallied, and the centre ring (the biggest) flashing
    # together with one other, at several points in their expansion
    sm3 = StaticTargets()
    sm3.start(context(now=100.0))
    for i, (mx, my, _r) in enumerate(marks_[1:]):
        sm3.shot(modes.ShotEvent(mx, my, 300.0, i + 1), context(now=300.0))
    sm3.shot(modes.ShotEvent(marks_[0][0], marks_[0][1], 300.0, 9), context(now=300.0))
    worst_lit, worst_red = 0.0, 0
    for dt_ in (0.0, 0.05, 0.1, 0.2, 0.3):
        r_, l_, _ = _full(sm3.paint, context(now=300.0 + dt_))
        worst_lit, worst_red = max(worst_lit, l_), max(worst_red, r_)
    print(f"         worst case (two rings + five tallies) lit {worst_lit * 100:.3f}%"
          f" = +{(worst_lit - lit_idle) * 100:.3f} percentage points over idle")
    check("worst case, every state: grey only (red excess 0)", worst_red == 0,
          f"worst {worst_red}")
    check("worst case: still inside the lit-area budget", worst_lit <= LIT_MAX,
          f"{worst_lit * 100:.3f}% against {LIT_MAX * 100:.0f}%")

    # --- start() clears them; calibration hides them --------------------------
    sm3.start(context(now=400.0))
    check("a restart clears the tallies and the flashes",
          not sm3.hits_by_mark and not sm3._flashes)

    cal = RenderContext(now=300.0, dt=0.0, rect=QRect(0, 0, SW, SH),
                        stats=SessionStats(), calibrating=True)
    sm4 = StaticTargets()
    sm4.start(cal)
    sm4.shot(modes.ShotEvent(marks_[0][0], marks_[0][1], 300.0, 1), cal)
    _r4, lit4, _i4 = _full(sm4.paint, cal)
    check("while calibrating it draws nothing extra (one ring to shoot at)",
          lit4 < 0.0005, f"{lit4 * 100:.4f}% lit")

    # --- sized as fractions of the screen, not pixels (RULES B5) --------------
    def _tally_height(h):
        w_ = int(h * 16 / 9)
        c_ = RenderContext(now=300.0, dt=0.0, rect=QRect(0, 0, w_, h), stats=SessionStats())
        s_ = StaticTargets()
        s_.start(c_)
        mk = s_._field.marks(c_)
        s_.shot(modes.ShotEvent(mk[0][0], mk[0][1], 100.0, 1), c_)
        s_._flashes = []                                  # tally only
        im_ = _QI3(w_, h, _QI3.Format_RGB32)
        im_.fill(0xFF0A0A0A)
        pn = _QP3(im_)
        s_._paint_tallies(pn, c_, mk)
        pn.end()
        a_ = np.frombuffer(im_.constBits(), np.uint8).reshape(h, im_.bytesPerLine())
        a_ = a_[:, :w_ * 4].reshape(h, w_, 4)
        rows = np.where((a_[..., 1] > 40).any(axis=1))[0]
        return int(rows.max() - rows.min() + 1) if rows.size else 0

    t1080, t2160 = _tally_height(1080), _tally_height(2160)
    check("the tally scales with the screen (1.6x or more from 1080p to 2160p)",
          t1080 > 0 and t2160 >= 1.6 * t1080, f"{t1080}px at 1080p, {t2160}px at 2160p")

    # ----------------------------------------------------------------------
    print()
    print("18. a mode that draws ARTWORK must announce it as solid")
    # His report, 2026-10-18, with two photographs: "you are drawing the shots
    # on the screen and the soldier moves BEHIND those shots".
    #
    # The figure drills met this on 2026-10-17 and gained `Occluder` for it.
    # The animated soldier, written the next day, drew the same transparent
    # line art and published nothing, so every hole showed through him. The
    # behavioural check that proves the figures are solid (test_figures 27)
    # could not catch it: it loops over a HARDCODED tuple of three mode
    # classes, which is the world-pinned shape that broke seven checks the
    # same day he added artwork.
    #
    # So this one derives its list from the REGISTRY. Any mode that loads
    # sprite or sequence art is drawing something whose dark ground is
    # transparent, and must override `occluders`; a mode that draws only
    # outlines inherits the default and is left alone. A mode written next
    # year is covered without anybody remembering this happened.
    import inspect as _i18
    from lsa.artwork.sequence import Sequence as _Seq18
    from lsa.modes.figure import Figure as _Fig18

    # Judged on what the module HOLDS, never on its text. The first version
    # of this check scanned source for "sprite" and failed the ball mode,
    # whose comment says *"The ball is a circle, not a sprite"* -- a check
    # that fails when nothing is wrong is how he learns to ignore checks.
    #
    # AND NOT ON THE IDENTITY OF TWO CLASSES EITHER. The first version asked
    # only "does this module hold `Sequence` or `Figure`", which is a list of
    # the loaders that existed the day it was written. A mode loading art
    # through a THIRD loader passes it while drawing exactly the same
    # transparent-ground pictures -- the menu mode on the other session's
    # branch loads tiles through `artwork.tiles` and would have shipped
    # drawing his bullet holes straight through its pictures. A check that
    # silently does not cover a new loader is worse than no check (C1).
    #
    # So: a mode draws artwork if it reaches into `lsa.artwork` for anything
    # but `ceiling`, which is the brightness palette and not a picture.
    def _draws_art18(mod18) -> bool:
        for n18 in dir(mod18):
            v18 = getattr(mod18, n18, None)
            if v18 in (_Seq18, _Fig18):
                return True
            nm18 = getattr(v18, "__name__", "") or ""
            if _i18.ismodule(v18) and nm18.startswith("lsa.artwork."):
                if not nm18.endswith(".ceiling"):
                    return True
            # A name imported FROM an artwork module (`from ..artwork import
            # catalog` binds a module; `from ..artwork.sequence import
            # Sequence` binds a class) -- catch the class form by where it
            # was defined.
            if getattr(v18, "__module__", "").startswith("lsa.artwork."):
                if not getattr(v18, "__module__", "").endswith(".ceiling"):
                    return True
        return False

    _need18, _bad18 = [], []
    for cls18 in modes.available():
        mod18 = _i18.getmodule(cls18)
        if not _draws_art18(mod18):
            continue
        _need18.append(cls18.name)
        if "def occluders" not in _i18.getsource(cls18):
            _bad18.append(cls18.name)
    check("every mode that draws artwork publishes its solid shape",
          _need18 and not _bad18,
          f"artwork modes: {sorted(_need18)}"
          + (f"   MISSING occluders: {sorted(_bad18)}" if _bad18 else ""))

    # And the shape it publishes must be the one it SCORES against, or what
    # hides a hole and what counts as a hit drift apart silently.
    from lsa.modes.animated import AnimatedFigure as _AF18
    m18 = _AF18()
    m18.start(context(now=10.0))
    m18.advance(context(now=10.3))
    occ18 = m18.occluders(context(now=10.3))
    seq18 = m18.active()
    l18, t18, w18, h18 = seq18.box(m18._frame, m18._origin, m18._scale)
    check("the soldier announces exactly one solid shape, his own box",
          len(occ18) == 1
          and abs(occ18[0].rect.width() - int(w18)) <= 1
          and abs(occ18[0].rect.height() - int(h18)) <= 1,
          f"{len(occ18)} occluder(s), {occ18[0].rect.width()}x{occ18[0].rect.height()}"
          f" against a drawn {int(w18)}x{int(h18)}" if occ18 else "none")

    _sil18 = seq18.silhouette(m18._frame)
    _mask18 = occ18[0].mask() if occ18 else None
    check("his solid shape IS his silhouette -- the one a hit is scored on",
          _mask18 is not None
          and (_mask18.width(), _mask18.height()) == (_sil18.shape[1], _sil18.shape[0]),
          f"mask {_mask18.width()}x{_mask18.height()} against silhouette "
          f"{_sil18.shape[1]}x{_sil18.shape[0]}" if _mask18 else "no mask")

    # WHICH FRAME IS HELD THROUGH THE BREAK. He watched it and said: "I meant
    # the frame when the guy is lying down (the last frame of the lying-down
    # sequence), not the first frame of the shooting sequence". It was holding
    # frame 0 of the fall -- him still standing -- because `_go` reset the
    # frame that the line above it had just set to the last one. The existing
    # check measured how LONG the break lasts and could not see which frame
    # was in it.
    m18f = _AF18()
    m18f.start(context(now=20.0))
    fall18 = m18f.fall_sequence()
    m18f._go("fall", 20.0)
    for k18 in range(1, 60):                      # past the end of the clip
        m18f.advance(context(now=20.0 + k18 * 0.1))
        if m18f._state == "rest":
            break
    check("he lies there through the break: the LAST frame of the fall, "
          "not the first", fall18 is not None and m18f._state == "rest"
          and m18f.active() is fall18 and m18f._frame == len(fall18) - 1,
          f"state {m18f._state}, frame {m18f._frame} of "
          f"{len(fall18) if fall18 else 0}")

    # And it stays there for the whole break rather than ticking on.
    m18f.advance(context(now=m18f._state_at + _AF18.REST_S * 0.5))
    check("and he does not move while he is down",
          m18f._state == "rest" and fall18 is not None
          and m18f._frame == len(fall18) - 1,
          f"state {m18f._state}, frame {m18f._frame}")
    m18f.advance(context(now=m18f._state_at + _AF18.REST_S + 0.05))
    check("then he gets up, from the start of the walk, with a clean counter",
          m18f._state == "walk" and m18f._frame == 0 and m18f._hits == 0,
          f"state {m18f._state}, frame {m18f._frame}, hits {m18f._hits}")

    # The travelling targets, which he asked for next: "the figures (balls,
    # people-like cutouts, etc.) that go over holes should obscure the holes".
    # A ball is drawn as an outline, so without this its inside is bare
    # screen and every hole it crosses shows through it.
    from lsa.modes.ball import EscalatingBall as _EB18
    from lsa.modes.balls import BouncingBalls as _BB18
    for cls18b, n18 in ((_EB18, 1), (_BB18, None)):
        m18b = cls18b()
        m18b.start(context(now=5.0))
        occ18b = m18b.occluders(context(now=5.0))
        want18 = n18 if n18 is not None else len(m18b.balls)
        check(f"{cls18b.name}: every ball on screen is solid",
              len(occ18b) == want18 and want18 > 0,
              f"{len(occ18b)} of {want18}")
        if cls18b is _EB18:
            r18 = m18b.ring_hit_radius(m18b.r)
            check("the ball's solid disc is exactly the circle a hit scores on",
                  abs(occ18b[0].rect.width() - int(2 * r18)) <= 1,
                  f"{occ18b[0].rect.width()}px across against a hit circle of "
                  f"{int(2 * r18)}px")
            # A DISC, not the bounding box: opaque in the middle, clear in
            # the corner. A box would hide holes beside the ball as well.
            mimg18 = occ18b[0].mask().toImage()
            w18b = mimg18.width()
            # pixelColor, NOT pixel: QImage.pixel returns a QRgb whose alpha
            # QColor reads back as 255 whatever the image holds, so the first
            # version of this check reported a square and the mask was a disc
            # all along -- the instrument, not the finding.
            mid18 = mimg18.pixelColor(w18b // 2, w18b // 2).alpha()
            cor18 = mimg18.pixelColor(1, 1).alpha()
            check("it is a disc and not a square -- it hides what it covers, "
                  "no more", mid18 > 200 and cor18 < 40,
                  f"alpha {mid18} at the centre, {cor18} in the corner")

    # The mask cache is bounded: this mode is the only content here with no
    # cache hit rate (a 120-frame clip reuses nothing), and an un-evicted
    # cache is already recorded in CLAUDE.md as costing 4.1MiB per figure.
    for _k18 in range(80):
        m18._frame = _k18 % len(seq18)
        m18.occluders(context(now=10.3 + _k18 * 0.01))[0].mask()
    check("the mask cache stays bounded while he walks",
          len(m18._masks) <= _AF18.PIXMAP_CACHE,
          f"{len(m18._masks)} held after 80 frames, cap {_AF18.PIXMAP_CACHE}")


    # ----------------------------------------------------------------------
    print()
    print("19. clear the board deals the next board by itself")
    # His report, 2026-10-18: "after all targets are cleared and the word says
    # round complete, you should populate those targets back. Otherwise the
    # board stays empty." An empty screen with a loaded pistol in his hand
    # reads as the app having stopped.
    from lsa.modes.circles import ClearTheBoard as _CTB
    cb = _CTB()
    c19 = context(now=100.0)
    cb.start(c19)
    n19 = len(cb.targets)
    check("a board is dealt to start with", n19 > 0, f"{n19} targets")
    for t in cb.targets:                       # clear it the way a shot does
        cb.shot(modes.ShotEvent(t.x, t.y, 100.5, 1), c19)
    check("clearing them all stops the clock and says so",
          cb.cleared_at is not None and "CLEARED" in cb.status(),
          cb.status())
    check("...and the board is empty for a moment, so the time can be read",
          not any(t.alive for t in cb.targets)
          and cb.advance(context(now=100.5 + _CTB.NEXT_BOARD_S * 0.5)) is not None
          and not any(t.alive for t in cb.targets))
    cb.advance(context(now=100.5 + _CTB.NEXT_BOARD_S + 0.05))
    alive19 = sum(1 for t in cb.targets if t.alive)
    check("...then a full board deals itself, with the clock running again",
          alive19 == n19 and cb.cleared_at is None,
          f"{alive19} of {n19} back, cleared_at {cb.cleared_at}")
    check("the best time survives the new board (it is what he shoots against)",
          cb.best_s is not None, str(cb.best_s))
    check("the pause outlasts the fade, so a new target is never dealt onto a "
          "dying one", _CTB.NEXT_BOARD_S > _CTB.FADE_S,
          f"{_CTB.NEXT_BOARD_S}s vs {_CTB.FADE_S}s")

    # ----------------------------------------------------------------------
    print()
    print("20. the escalating ball enters from anywhere, never a corner")
    # "the balls very predictably show up from the corner. They should float
    # in from pretty much anywhere and never in a predictable way." The old
    # spawn had FOUR possible entry points in the whole drill.
    from lsa.modes.ball import EscalatingBall as _EB
    eb = _EB()
    eb._rng.seed(11)
    c20 = context(now=200.0)
    eb.start(c20)
    field20 = eb.playfield(c20)
    inner20 = field20.adjusted(eb.r, eb.r, -eb.r, -eb.r)
    corners20 = [(inner20.left(), inner20.top()), (inner20.right(), inner20.top()),
                 (inner20.left(), inner20.bottom()), (inner20.right(), inner20.bottom())]
    pts20, near_corner20, edges20 = [], 0, set()
    for i in range(400):
        eb._spawn(c20, (inner20.center().x(), inner20.center().y()))
        pts20.append((eb.x, eb.y))
        if min(math.hypot(eb.x - cx, eb.y - cy) for cx, cy in corners20) < eb.r:
            near_corner20 += 1
        on = []
        if abs(eb.x - inner20.left()) < 1.0:   on.append("L")
        if abs(eb.x - inner20.right()) < 1.0:  on.append("R")
        if abs(eb.y - inner20.top()) < 1.0:    on.append("T")
        if abs(eb.y - inner20.bottom()) < 1.0: on.append("B")
        edges20.update(on)
    check("it enters on the edge of the playfield, from all four sides",
          edges20 == {"L", "R", "T", "B"}, "".join(sorted(edges20)))
    # JUDGED AGAINST WHAT UNIFORM ENTRY PREDICTS, not against a number I
    # liked. A corner is a real part of the perimeter, so entries near one
    # are not a fault -- entries ONLY near one were. Eight ball-radii of
    # perimeter sit within a radius of the four corners, so uniform sampling
    # puts that share there by construction; demanding fewer would be asking
    # for a bias in the other direction.
    _per20 = 2.0 * (inner20.width() + inner20.height())
    _expect20 = 400.0 * (8.0 * eb.r) / _per20
    check("corners get no more than their share of the perimeter "
          "(the old spawn gave them ALL of it)",
          near_corner20 <= _expect20 * 1.6,
          f"{near_corner20} of 400 within a radius of a corner, "
          f"uniform predicts {_expect20:.0f}")
    check("the entry points are spread, not a handful of places",
          len({(round(x / 40), round(y / 40)) for x, y in pts20}) >= 40,
          f"{len({(round(x / 40), round(y / 40)) for x, y in pts20})} distinct cells")

    # It still makes him MOVE: the point of the old corner rule.
    far20 = 0
    for i in range(200):
        last = (inner20.left() + 30, inner20.top() + 30)
        eb._spawn(c20, last)
        if math.hypot(eb.x - last[0], eb.y - last[1]) >= \
           field20.width() * 0.25:
            far20 += 1
    # Measured against the SHORT side. On a 16:9 panel that is about a
    # quarter of the screen width -- a real move. A fraction of the DIAGONAL
    # is only reachable from the corners, which is the bias being removed:
    # the bar and the fault would have been the same thing.
    check("and it still lands well away from the last hit", far20 >= 198,
          f"{far20} of 200 at least a quarter of the screen width away "
          f"({field20.width() * 0.25:.0f}px)")

    # It must travel INTO the field, or it would hug the edge it arrived on.
    inward20, parallel20 = 0, 0
    for i in range(200):
        eb._spawn(c20, None)
        to_cx, to_cy = inner20.center().x() - eb.x, inner20.center().y() - eb.y
        if eb.vx * to_cx + eb.vy * to_cy > 0:
            inward20 += 1
        sp = math.hypot(eb.vx, eb.vy)
        if sp > 0 and min(abs(eb.vx), abs(eb.vy)) / sp < 0.18:
            parallel20 += 1
    check("every entry heads into the field, not along the edge",
          inward20 == 200, f"{inward20} of 200 inward")
    check("and none of them is near-axis-aligned", parallel20 == 0,
          f"{parallel20} of 200 within 10 degrees of an axis")
    # ------------------------------------------------------------------
    # 21. versus geometry (pure): modes/split.py
    # ------------------------------------------------------------------
    print()
    print("21. versus geometry: one pure module answers every rectangle")
    from lsa.modes import split as _split
    from lsa.display.layers import px as _px, top_band_height as _tbh
    for (_w, _h) in ((3840, 2160), (1920, 1080), (1921, 1080), (2561, 1441)):
        _full = QRect(0, 0, _w, _h)
        _lay = _split.layout(_full)
        _tag = f"{_w}x{_h}"
        check(f"{_tag}: the two sides are exactly equal",
              _lay.left.width() == _lay.right.width() and _lay.left.height() == _h)
        check(f"{_tag}: sides and divider tile the width with no gap or overlap",
              _lay.left.right() + 1 == _lay.divider.left()
              and _lay.divider.right() + 1 == _lay.right.left()
              and _lay.right.right() + 1 == _w and _lay.left.left() == 0)
        check(f"{_tag}: the divider is centred on the screen",
              abs((_lay.divider.left() + _lay.divider.right() + 1) / 2.0 - _w / 2.0) < 1.0
              and abs(_lay.centre_x - _w / 2.0) < 1.0)
        check(f"{_tag}: the divider is inside neither playfield",
              not _lay.divider.intersects(_lay.left) and not _lay.divider.intersects(_lay.right))
        from lsa.display.layers import band_height_for as _bhf, bottom_strip_for as _bsf
        check(f"{_tag}: the divider spans the playfield height only",
              _lay.divider.top() == _bhf(_h) and _lay.divider.bottom() + 1 == _h - _bsf(_h))
        check(f"{_tag}: the scoreline sits just BELOW the reserved band, centred",
              _lay.scoreline.top() == _bhf(_h)
              and abs(_lay.scoreline.center().x() - _lay.centre_x) <= 1.0
              and _lay.left_cell.right() < _lay.divider.left()
              and _lay.right_cell.left() > _lay.divider.right())
        check(f"{_tag}: the divider width is a fraction of the screen, not a pixel count",
              _lay.divider.width() <= max(4, round(_w * 0.003)))
    _L = _split.layout(QRect(0, 0, 3840, 2160))
    check("PREMISE: px() is identical in a half-width derived context",
          all(_px(context(), n) == _px(RenderContext(now=0.0, dt=0.0, rect=_L.right,
                                                       stats=SessionStats()), n)
              for n in (8, 16, 20, 24, 40, 108)))
    check("PREMISE: top_band_height() is identical in a half-width derived context",
          _tbh(context()) == _tbh(RenderContext(now=0.0, dt=0.0, rect=_L.left,
                                                 stats=SessionStats())))
    # Routing boundary, stated and checked AT the divider.
    check("routing: well left goes left, well right goes right",
          _L.route(10) == _split.LEFT and _L.route(3830) == _split.RIGHT)
    check("routing: one pixel either side of the centre line",
          _L.route(_L.centre_x - 0.01) == _split.LEFT and _L.route(_L.centre_x + 0.01) == _split.RIGHT)
    check("routing: exactly ON the centre line goes right (one comparison, no dead zone)",
          _L.route(_L.centre_x) == _split.RIGHT)
    check("routing: a shot on the divider strip's left half is routed left",
          _L.route(_L.divider.left() + 0.5) == _split.LEFT)
    # playable is narrowed by INTERSECTION per side.
    _cov = QRect(500, 300, 2600, 1500)
    _pl, _pr = _L.playable_for(_split.LEFT, _cov), _L.playable_for(_split.RIGHT, _cov)
    check("playable is intersected per side, not replaced",
          _cov.contains(_pl) and _cov.contains(_pr) and _L.left.contains(_pl)
          and _L.right.contains(_pr) and not _pl.intersects(_pr))
    check("an unknown coverage yields the side itself below the scoreline",
          _L.playable_for(_split.LEFT, None).left() == _L.left.left()
          and _L.playable_for(_split.LEFT, None).top() > _L.scoreline.bottom())
    check("no target ground under the scoreline on either side",
          _pl.top() > _L.scoreline.bottom() and _pr.top() > _L.scoreline.bottom())
    check("coverage that excludes a side leaves that side empty",
          _L.playable_for(_split.RIGHT, QRect(0, 0, 1000, 2160)).isEmpty())

    # ------------------------------------------------------------------
    # 22. versus: ModeHost holds two modes and routes by x
    # ------------------------------------------------------------------
    print()
    print("22. versus: the host runs one copy of the drill per half and routes shots by x")
    from lsa.modes.base import Mode as _Mode, ModeHost as _Host
    from lsa.modes.static import StaticTargets as _Static
    from lsa.modes.balls import BouncingBalls as _Balls
    from lsa.modes.reaction import ReactionDrill as _Reaction
    _full22 = context()
    _lay22 = _split.layout(_full22.rect)

    def _versus_host(cls):
        h = _Host()
        h.set_mode(cls(), _full22, players=2)
        return h

    # -- routing isolation ----------------------------------------------------
    h22 = _versus_host(_Static)
    check("two players on a ready drill builds a second instance of the same drill",
          h22.versus and type(h22.mode2) is _Static and h22.mode2 is not h22.mode)
    lctx22 = h22._side_ctx(_full22, 0)
    rctx22 = h22._side_ctx(_full22, 1)
    lx, ly = h22.mode.aim_point(lctx22)
    rx, ry = h22.mode2.aim_point(rctx22)
    check("the two aim points are in their own halves",
          lx < _lay22.centre_x < rx and _lay22.left.contains(int(lx), int(ly))
          and _lay22.right.contains(int(rx), int(ry)), f"left {lx:.0f}, right {rx:.0f}")
    out22 = h22.shot(modes.ShotEvent(lx, ly, 1.0, 1), _full22)
    check("a shot on the left centre mark hits there",
          out22.hit and h22.mode.hits == 1 and h22.mode.shots == 1 and h22.mode.score > 0)
    check("...and the RIGHT mode is completely untouched (shots, hits, score, marks)",
          h22.mode2.shots == 0 and h22.mode2.hits == 0 and h22.mode2.score == 0
          and h22.mode2.hits_by_mark == {} and h22.mode2._flashes == [])
    check("...and the host tallies agree: left 1 shot 1 hit, right all zero",
          (h22.tallies[0].shots, h22.tallies[0].hits) == (1, 1)
          and (h22.tallies[1].shots, h22.tallies[1].hits, h22.tallies[1].points) == (0, 0, 0))
    h22.shot(modes.ShotEvent(rx, ry, 2.0, 2), _full22)
    h22.shot(modes.ShotEvent(rx + 200, ry + 500, 3.0, 3), _full22)
    check("shots on the right reach only the right mode",
          h22.mode2.shots == 2 and h22.mode.shots == 1 and h22.mode.hits == 1)
    check("...and the host tallies follow the side: left 1 shot, right 2 shots, no cross-count",
          (h22.tallies[0].shots, h22.tallies[1].shots) == (1, 2)
          and h22.tallies[0].hits == 1)
    h22b = _versus_host(_Static)
    h22b.shot(modes.ShotEvent(20.0, 20.0, 1.0, 1), _full22)       # a left miss
    check("a MISS on one side leaves the other side every counter at zero",
          h22b.mode.shots == 1 and h22b.mode.hits == 0
          and h22b.mode2.shots == 0 and h22b.mode2.hits == 0 and h22b.mode2.score == 0
          and h22b.tallies[1].shots == 0)
    h22c = _versus_host(_Static)
    h22c.shot(modes.ShotEvent(_lay22.centre_x, 1000.0, 1.0, 1), _full22)
    check("a shot exactly on the centre line is routed right, once",
          h22c.mode2.shots == 1 and h22c.mode.shots == 0)
    _before22 = (h22c.tallies[0].shots, h22c.tallies[1].shots, h22c.mode.shots, h22c.mode2.shots)
    h22c.on_click(lx, ly, _full22)
    h22c.on_click(rx, ry, _full22)
    check("a click is never a shot: two clicks on the aim marks change no shot count",
          _before22 == (h22c.tallies[0].shots, h22c.tallies[1].shots,
                        h22c.mode.shots, h22c.mode2.shots))

    # -- each side keeps to its own half --------------------------------------
    inside22 = True
    for cls22 in (_Static, EscalatingBall, _Balls, ClearTheBoard):
        hh = _versus_host(cls22)
        for which22 in (0, 1):
            fld = _Mode.playfield(hh._side_ctx(_full22, which22))
            side = _lay22.side(which22)
            if not (fld.left() >= side.left() and fld.right() <= side.right() + 1):
                inside22 = False
    check("every versus-ready drill playfield lies inside its own half", inside22)
    hc22 = _versus_host(ClearTheBoard)
    ok_c = all(_lay22.left.left() <= t.x - t.r and t.x + t.r <= _lay22.left.right() + 1
               for t in hc22.mode.targets) and all(
        _lay22.right.left() <= t.x - t.r and t.x + t.r <= _lay22.right.right() + 1
        for t in hc22.mode2.targets)
    check("Clear The Board deals each side targets wholly inside that side", ok_c,
          f"{len(hc22.mode.targets)} + {len(hc22.mode2.targets)} targets")
    _cov22 = QRect(600, 400, 2400, 1300)
    hcov = _Host()
    hcov.set_mode(ClearTheBoard(), context(playable=_cov22), players=2)
    check("camera coverage still binds each side (playable intersected, not dropped)",
          all(_cov22.left() <= t.x - t.r and t.x + t.r <= _cov22.right() + 1
              and _cov22.top() <= t.y - t.r and t.y + t.r <= _cov22.bottom() + 1
              for t in hcov.mode.targets + hcov.mode2.targets))

    # -- the MENU keep-out comes from the FULL screen -------------------------
    _saved_corner = _Mode.MENU_CORNER
    try:
        for corner22 in ("top left", "bottom right"):
            _Mode.MENU_CORNER = corner22
            hk = _versus_host(ClearTheBoard)
            full_keep = _Mode.menu_keepout(_full22)
            lk = _Mode.menu_keepout(hk._side_ctx(_full22, 0))
            rk = _Mode.menu_keepout(hk._side_ctx(_full22, 1))
            half_only = _Mode.menu_keepout(dataclass_replace(_full22, rect=_lay22.right))
            check(f"[{corner22}] a drill handed a half asks for the FULL-screen keep-out",
                  lk == full_keep and rk == full_keep)
            check(f"[{corner22}] (premise: derived from the half alone it would be elsewhere)",
                  half_only != full_keep)
            btn = _Mode.menu_rect(_full22)
            hk.shot(modes.ShotEvent(btn.center().x(), btn.center().y(), 5.0, 1), _full22)
            check(f"[{corner22}] a shot at the button the shooter sees arms it",
                  hk.menu_button.armed_at == 5.0)
            side_of_btn = 0 if btn.center().x() < _lay22.centre_x else 1
            tg = (hk.mode if side_of_btn == 0 else hk.mode2).targets
            check(f"[{corner22}] no target stands on the button, on its own side",
                  all(_Mode.clear_of_menu(hk._side_ctx(_full22, side_of_btn), t.x, t.y, t.r)
                      for t in tg))
    finally:
        _Mode.MENU_CORNER = _saved_corner

    # -- not versus-ready: single-player, with a reason -----------------------
    # EVERY REGISTERED DRILL IS NOW READY (2026-10-19: drills 6-9 joined), so the
    # fallback path is exercised with a stand-in that has not been measured.
    class _NotReady22(_Mode):
        name = "notready"
        title = "Not Ready"

        def start(self, ctx):
            pass

        def shot(self, ev, ctx):
            return modes.ShotOutcome.miss()

        def paint(self, p, ctx):
            pass

    hr = _Host()
    hr.set_mode(_NotReady22(), _full22, players=2)
    check("a drill that is not versus-ready runs single-player and keeps the reason",
          hr.versus is False and hr.mode2 is None and "not ready" in hr.fallback_note)
    hm = _Host()
    hm.set_mode(modes.by_name("menu")(), _full22, players=2)
    check("the menu never splits and carries no apology",
          hm.versus is False and hm.fallback_note == "")
    hs = _Host()
    hs.set_mode(_Static(), _full22, players=1)
    check("one player: no second instance, no note",
          hs.versus is False and hs.mode2 is None and hs.fallback_note == "")
    check("the flag is declared on drills 1-9 and nowhere else (each measured: sections 22, 26 and 35)",
          sorted(c.name for c, _k in modes.registered() if c.versus_ready)
          == ["advancing", "animated", "ball", "balls", "circles", "image", "popup",
              "reaction", "static"])

    # -- both advance, repaints are bounded, occluders concatenate ------------
    ha = _versus_host(EscalatingBall)
    p1 = (ha.mode.x, ha.mode.y)
    p2 = (ha.mode2.x, ha.mode2.y)
    for i in range(30):
        ha.advance(context(now=0.0166 * (i + 1)))
    check("both sides advance every tick",
          (ha.mode.x, ha.mode.y) != p1 and (ha.mode2.x, ha.mode2.y) != p2)
    check("each ball occluder is reported: two sides, two discs",
          len(ha.occluders(context(now=1.0))) == 2)
    hi = _versus_host(_Static)
    asked = sum(1 for i in range(180) if hi.advance(context(now=i / 60.0)))
    check("a static split asks for no repaints while idle", asked <= 1, f"{asked} of 180 ticks")
    hi.shot(modes.ShotEvent(lx, ly, 3.0, 1), _full22)
    n_after = sum(1 for i in range(180) if hi.advance(context(now=3.0 + i / 60.0)))
    check("a scoring shot costs the hit flash plus one repaint for the scoreline, nothing more",
          n_after <= 1 + int(_Static.FLASH_S * 60) + 2, f"{n_after} of 180 ticks")
    hi2 = _versus_host(_Static)
    hi2.advance(context(now=0.0))
    hi2.shot(modes.ShotEvent(20.0, 20.0, 1.0, 1), _full22)       # a miss: no flash at all
    n_miss = sum(1 for i in range(120) if hi2.advance(context(now=1.0 + i / 60.0)))
    check("a missed shot is exactly ONE repaint (the shots count), not a stream",
          n_miss == 1, f"{n_miss} of 120 ticks")

    # -- restart / stop / status ----------------------------------------------
    hr2 = _versus_host(_Static)
    hr2.shot(modes.ShotEvent(lx, ly, 1.0, 1), _full22)
    hr2.shot(modes.ShotEvent(rx, ry, 1.1, 2), _full22)
    hr2.restart(_full22)
    check("a restart clears BOTH sides and the host tallies",
          hr2.mode.shots == 0 and hr2.mode2.shots == 0
          and hr2.tallies[0].shots == 0 and hr2.tallies[1].shots == 0
          and hr2.tallies[0].points == 0 and hr2.tallies[1].points == 0)
    check("status names both sides", "L:" in hr2.status() and "R:" in hr2.status())
    hr2.set_mode(_Static(), _full22, players=1)
    check("switching back to one player drops the second instance and the tallies",
          hr2.mode2 is None and hr2.versus is False and hr2.tallies[0].shots == 0)

    # -- plan section 7 open number: do the balls still clear the floor? ------
    eb22 = EscalatingBall()
    eb22.level = eb22.MAX_LEVEL
    sc_r = _versus_host(EscalatingBall)._side_ctx(_full22, 1)
    eb22.start(sc_r)
    fld_r = _Mode.playfield(sc_r)
    floor_ball = _Balls.MIN_RADIUS_FRAC * min(fld_r.width(), fld_r.height())
    print(f"     escalating ball at level 5 in a half field: r = {eb22.r:.1f}px "
          f"(floor {floor_ball:.1f}px; field {fld_r.width():.0f}x{fld_r.height():.0f})")
    check("escalating ball at level 5 in a half field clears the radius floor",
          eb22.r >= floor_ball, f"{eb22.r:.1f}px vs {floor_ball:.1f}px")
    bb22 = _Balls()
    bb22.start(sc_r)
    for b in bb22.balls:
        b.level = bb22.MAX_LEVEL
    rb22 = min(bb22._radius(b, sc_r) for b in bb22.balls)
    print(f"     bouncing balls at level 5 in a half field: r = {rb22:.1f}px "
          f"(floor {floor_ball:.1f}px)")
    check("bouncing balls at level 5 in a half field clear the radius floor",
          rb22 >= floor_ball, f"{rb22:.1f}px vs {floor_ball:.1f}px")

    # ------------------------------------------------------------------
    # 23. versus: lit area and red excess at 4K, every ready drill, every state
    # ------------------------------------------------------------------
    print()
    print("23. versus at 4K: divider + scoreline + two sets of targets against the 2% ceiling")
    from lsa.display.layers import BulletHoles as _BH23
    _LIT23 = 0.02
    _worst23 = (0.0, "")
    for cls23 in (_Static, EscalatingBall, _Balls, ClearTheBoard):
        h23 = _Host()
        t23 = 100.0
        h23.set_mode(cls23(), context(now=t23), players=2)
        c23 = context(now=t23)
        for i in range(10):                      # let balls spawn and settle
            h23.advance(context(now=t23 + i / 60.0))
            c23 = context(now=t23 + i / 60.0)
        # a point on each side's own first target, wherever this drill put it
        pts23 = []
        for m23, w23 in ((h23.mode, 0), (h23.mode2, 1)):
            if hasattr(m23, "targets"):
                pts23.append((m23.targets[0].x, m23.targets[0].y))
            elif hasattr(m23, "balls"):
                pts23.append((m23.balls[0].x, m23.balls[0].y))
            elif hasattr(m23, "aim_point"):
                pts23.append(m23.aim_point(h23._side_ctx(c23, w23)))
            else:
                pts23.append((m23.x, m23.y))
        states23 = []
        r_, lit_, _img = _fh_full(h23.paint, c23)
        states23.append(("idle", r_, lit_))
        h23.shot(modes.ShotEvent(pts23[0][0], pts23[0][1], c23.now, 1), c23)
        h23.shot(modes.ShotEvent(pts23[1][0], pts23[1][1], c23.now, 2), c23)
        c24 = context(now=c23.now + 0.05)
        r_, lit_, _img = _fh_full(h23.paint, c24)
        states23.append(("both sides just hit", r_, lit_))
        holes23 = _BH23()
        for k, (hx, hy) in enumerate(pts23):
            holes23.add(hx, hy, k + 1, 0)

        def _holes_under(p_, c_, h=h23, hl=holes23):
            hl.paint(p_, c_)
            h.paint(p_, c_)

        def _holes_over(p_, c_, h=h23, hl=holes23):
            h.paint(p_, c_)
            hl.paint(p_, c_)

        r_, lit_, _img = _fh_full(_holes_under, c24)
        states23.append(("hit + fresh holes (holes under)", r_, lit_))
        r_, lit_, _img = _fh_full(_holes_over, c24)
        states23.append(("hit + fresh holes (holes over)", r_, lit_))
        for label23, red23, lit23 in states23:
            print(f"         {cls23.title:<16} {label23:<34} lit {lit23 * 100:.3f}%  red excess {red23}")
            check(f"{cls23.title} [{label23}]: red excess exactly 0", red23 == 0, f"worst {red23}")
            check(f"{cls23.title} [{label23}]: inside the 2% lit-area ceiling",
                  lit23 <= _LIT23, f"{lit23 * 100:.3f}%")
            if lit23 > _worst23[0]:
                _worst23 = (lit23, f"{cls23.title} / {label23}")
    print(f"         WORST lit area across all four drills and all states: "
          f"{_worst23[0] * 100:.3f}% ({_worst23[1]})")
    # the chrome alone: what the divider and scoreline cost with an empty drill
    hch = _Host()
    hch.set_mode(_Static(), context(now=1.0), players=2)
    hch.mode.visible = hch.mode2.visible = True

    def _chrome_only(p_, c_, h=hch):
        h._paint_chrome(p_, c_, h._layout(c_))

    r_, lit_, _img = _fh_full(_chrome_only, context(now=1.0))
    print(f"         divider + scoreline alone: lit {lit_ * 100:.4f}%  red excess {r_}")
    check("divider + scoreline alone: red excess 0 and (nearly) no lit area",
          r_ == 0 and lit_ < 0.0005, f"lit {lit_ * 100:.4f}%")

    # -- the host nearest() follows the shot to its own side (log-only) --------
    hn = _versus_host(_Static)
    nr = hn.nearest(modes.ShotEvent(rx, ry, 1.0, 1), _full22)
    nl_ = hn.nearest(modes.ShotEvent(lx, ly, 1.0, 2), _full22)
    check("nearest() for a right-hand shot is measured against the RIGHT side",
          nr is not None and nr[1] < 5.0 and nl_ is not None and nl_[1] < 5.0,
          f"right d={nr[1] if nr else None}, left d={nl_[1] if nl_ else None}")

    # -- 24. the mark set scales by the field's share of the screen width ------
    print()
    print("24. AimField: marks are sized by the field's share of the screen width")
    from lsa.display.layers import AimField as _AF24
    _af24 = _AF24()
    _ar24 = [(3840, 2160), (1920, 1080), (2560, 1080), (1080, 1920), (1600, 1200)]
    # single player is the OLD formulas, bit for bit, on every aspect ratio
    _same24 = True
    for sw24, sh24 in _ar24:
        c24_ = RenderContext(now=0.0, dt=1 / 60.0, rect=QRect(0, 0, sw24, sh24),
                             stats=SessionStats())
        mk24 = _af24.marks(c24_)
        old_c = [sh24 * 0.05, sh24 * 0.13, sh24 * 0.26, sh24 * 0.42]
        old_r = min(sw24, sh24) * 0.055
        _same24 &= (mk24[0][2] == old_c)
        _same24 &= all(m_[2] == [old_r * 0.35, old_r * 0.65, old_r] for m_ in mk24[1:])
    check("whole screen: radii equal the pre-versus formulas exactly, on 5 aspect ratios",
          _same24)
    # a half field is exactly half, for the centre AND the corners (no double count)
    _full24 = RenderContext(now=0.0, dt=1 / 60.0, rect=QRect(0, 0, 3840, 2160),
                            stats=SessionStats())
    _half24 = dataclass_replace(_full24, rect=QRect(0, 0, 1920, 2160), screen_width=3840)
    _fm24, _hm24 = _af24.marks(_full24), _af24.marks(_half24)
    check("a half field's centre radii are exactly 0.5 x the whole screen's",
          _hm24[0][2] == [r_ * 0.5 for r_ in _fm24[0][2]])
    check("a half field's corner radii are exactly 0.5 x the whole screen's (not 0.444)",
          all(h_[2] == [r_ * 0.5 for r_ in f_[2]] for f_, h_ in zip(_fm24[1:], _hm24[1:])),
          f"{_hm24[1][2][-1]:.2f} vs half of {_fm24[1][2][-1]:.2f}")
    _rt24 = dataclass_replace(_half24, rect=QRect(1920, 0, 1920, 2160))
    check("the right half is the same size as the left",
          [m_[2] for m_ in _af24.marks(_rt24)] == [m_[2] for m_ in _hm24])

    # ------------------------------------------------------------------
    # 25. versus rounds: 20 REGISTERED shots per side, the swap, the verdict
    # ------------------------------------------------------------------
    print()
    print("25. versus rounds: a side is done after 20 registered shots; the sides swap; the totals carry")
    from lsa.modes import versus as _V
    from lsa.modes.balls import BouncingBalls as _Balls25
    from lsa.modes.circles import ClearTheBoard as _Ctb25
    from PySide6.QtGui import QImage as _QI25b, QPainter as _QP25b
    N25 = _DEFAULT_MAG
    R = _V.SideResult

    # -- the arithmetic, pure -------------------------------------------------
    # The PROPERTY is that there is one number and the versus module does not
    # keep a second copy of it. The VALUE is his, and pinning a check to it
    # means the build goes red when he exercises a choice that is his to make
    # -- which is what happened when he moved the default from 17 to 19 on
    # 2026-10-20, and a check that fails when nothing is wrong teaches him to
    # ignore checks.
    check("a round is one magazine of registered shots: ONE number, owned by settings, whatever he sets it to",
          N25 == _DEFAULT_MAG and not hasattr(_V, "ROUND_SHOTS"))
    check("two sides with nothing registered: no verdict at all",
          _V.judge(R(), R()).kind == "none")
    check("equal points on equal shots is a tie", _V.judge(R(50, N25), R(50, N25)).kind == "tie")
    check("a margin of ONE point names a winner: there is no refusal gate any more",
          (lambda v_: v_.kind == "win" and v_.winner == 1 and v_.margin == 1)(
              _V.judge(R(49, N25), R(50, N25))))
    check("...and the winner index follows the points, whichever side it is",
          _V.judge(R(60, N25), R(10, N25)).winner == 0 and _V.judge(R(10, N25), R(60, N25)).winner == 1)
    try:
        _V.judge(R(1, 19), R(1, 20))
        unequal_raised = False
    except ValueError:
        unequal_raised = True
    check("unequal registered shots are refused outright: equal denominators are the fairness argument",
          unequal_raised)
    check("the machinery that served the old design is GONE (end_round, best_points, the unseen derivation)",
          not hasattr(_Host, "end_round") and not hasattr(_Mode, "best_points")
          and not hasattr(R, "unseen") and not hasattr(_V, "seen_text")
          and not hasattr(_V.Verdict, "worth") and not hasattr(_V.Verdict, "unseen"))
    allwords = " ".join([
        _V.verdict_sentence(_V.judge(R(50, N25), R(10, N25))),
        _V.verdict_sentence(_V.judge(R(0, N25), R(0, N25))),
        _V.verdict_sentence(_V.judge(R(), R())),
        _V.heard_text(True, 3, 5), _V.heard_text(False, 0), _V.done_text("LEFT", N25, N25)])
    check("no sentence the app can say contains the word 'fair' (RULES: only measured facts)",
          "fair" not in allwords.lower())
    check("a finished side says so in the plan's words: 'LEFT: done, 20 of 20'",
          _V.done_text("LEFT", 20, 20) == "LEFT: done, 20 of 20")
    check("the microphone line NEVER prints a bare count while uncalibrated, whatever the count",
          all("not calibrated" in _V.heard_text(False, n_) and not any(ch.isdigit() for ch in _V.heard_text(False, n_))
              for n_ in (0, 1, 7)))
    # REWORDED 2026-10-19 (his request: "sound more professional"): the LOG line
    # says "trigger pulls ... did not register", and the SCREEN says nothing at
    # all unless there is a count to report.
    check("...and the log says plainly when it is calibrated and heard nothing the camera missed",
          "no trigger pulls heard" in _V.heard_text(True, 0))
    check("...and the SCREEN says nothing when the microphone is uncalibrated or has no count",
          _V.heard_on_screen(False, 5) == "" and _V.heard_on_screen(True, 0) == ""
          and _V.heard_on_screen(True, 0, 0) == "")
    check("...and speaks, briefly and with the measurement, when it has a count",
          _V.heard_on_screen(True, 3).startswith("Microphone: 3 trigger pulls heard")
          and "this match" in _V.heard_on_screen(True, 3, 5)
          and "swap" not in _V.heard_on_screen(True, 3, 5).lower())
    # The points line with its ceiling (2026-10-22): one definition, so the painter and
    # the tests read the same words. The unit follows the HUMAN's number.
    check("`points_of_word`: `65 of 75 points` with a usable ceiling, `1 of 75 points` for one point, "
          "and a zero score still names the ceiling",
          _V.points_of_word(65, 75) == "65 of 75 points" and _V.points_of_word(1, 75) == "1 of 75 points"
          and _V.points_of_word(0, 75) == "0 of 75 points")
    check("...and with NO usable ceiling it is exactly `points_word`: None, zero, negative, NaN, "
          "infinity, a bool, a string",
          all(_V.points_of_word(p_, g_) == _V.points_word(p_)
              for p_ in (0, 1, 65) for g_ in (None, 0, -5, float("nan"), float("inf"), True, "75")))

    # -- the host: the allotment ------------------------------------------------
    def _vh25(cls=_Static, now=1.0):
        h_ = _Host()
        c_ = context(now=now)
        h_.set_mode(cls(), c_, players=2)
        return h_, c_

    def _aim(h_, c_, side_):
        """A point on this side's first target, whatever the drill calls it."""
        m_ = h_.mode if side_ == 0 else h_.mode2
        if hasattr(m_, "aim_point"):
            return m_.aim_point(h_._side_ctx(c_, side_))
        if getattr(m_, "targets", None):
            return (m_.targets[0].x, m_.targets[0].y)
        if getattr(m_, "balls", None):
            return (m_.balls[0].x, m_.balls[0].y)
        return (m_.x, m_.y)

    # THE DEFAULT STEP IS 0.01 s, A BURST, AND A BURST IS A DISCARDED ROUND since
    # 2026-10-22: a full magazine fired 0.01 s apart spans well under a second against a
    # floor of 0.91 x 0.25 s x (N - 1) (3.64 s for 17), so the host judges it a storm and shows no
    # score. That is right for the checks that only need a round to CLOSE (tallies,
    # allotments, layout geometry) and wrong for any check that reads a score, an
    # efficiency or a record off the screen. Those pass `step=PACE`: a human rate, 0.5 s
    # a shot. The two are never mixed by accident -- see sections 25, 49 and 50.
    PACE = 0.5

    def _fire(h_, c_, side_, n_, t0=1.0, on_target=True, step=0.01):
        out_ = None
        for i_ in range(n_):
            if on_target:
                x_, y_ = _aim(h_, c_, side_)
            else:
                x_, y_ = (60.0 if side_ == 0 else 3780.0), 1500.0
            out_ = h_.shot(modes.ShotEvent(x_, y_, t0 + i_ * step, i_ + 1), c_)
        return out_

    h25, hc25 = _vh25()
    _fire(h25, hc25, 0, N25 - 1)
    check("nineteen registered shots in: the left side is not done and says its round",
          not h25._side_done(0) and h25.note_text(0) == "round 1")
    _fire(h25, hc25, 0, 1)
    check("the last registered shot of the magazine uses the allotment: the half says DONE, N of N",
          h25._side_done(0) and h25.note_text(0) == f"LEFT: done, {N25} of {N25}")
    pts25, mshots25, mscore25 = h25.tallies[0].points, h25.mode.shots, h25.mode.score
    out25 = _fire(h25, hc25, 0, 1)
    check("a 21st shot is seen but scores nothing and does not reach the drill",
          out25.hit is False and h25.tallies[0].shots == N25 and h25.tallies[0].after_done == 1
          and h25.tallies[0].points == pts25 and h25.mode.shots == mshots25
          and h25.mode.score == mscore25)
    check("...it is UNSCORED (neither hit nor miss), plays the dry-pull pop INSTEAD of the gunshot, and DRAWS its hole",
          out25.unscored is True and out25.draw_hole and out25.play_sound and out25.dry)
    check("...and an ordinary shot is NOT unscored (the flag marks only the shot the round did not want)",
          _fire(h25, hc25, 1, 1, on_target=False).unscored is False)
    check("...and the other side is untouched: its count, note and drill",
          h25.tallies[1].shots == 1 and h25.mode2.shots == 1 and not h25._side_done(1)
          and h25.note_text(1) == "round 1" and not h25.round_over)
    # a side the app under-detects just shoots more: the round still closes by itself
    _fire(h25, hc25, 1, N25 - 2, t0=5.0, on_target=False)
    check("the other side at 19 registered shots: the round is still open (it waits for the 20th, with no timer)",
          not h25.round_over and h25.tallies[1].shots == N25 - 1)
    _fire(h25, hc25, 1, 1, t0=6.0, on_target=False)
    check("its 20th registered shot closes the round BY ITSELF: both sides banked at exactly 20",
          h25.round_over and h25.verdict is not None and h25.totals[0].shots == N25
          and h25.totals[1].shots == N25)
    check("...banked once: the left player's points are the round's, not doubled",
          h25.totals[0].points == pts25 and h25.totals[1].points == 0)
    tot_before25 = (h25.totals[0].points, h25.totals[1].points)
    _fire(h25, hc25, 0, 3, t0=9.0)
    check("...and shots after the round is over cannot move a total",
          tot_before25 == (h25.totals[0].points, h25.totals[1].points))
    check("a finished round draws no drills and describes no solid shapes", h25.occluders(hc25) == [])
    h25.advance(context(now=19.0))                 # consume the one repaint the verdict itself needs
    check("...and the drills are not ticked behind the verdict (no repaint demanded)",
          [h25.advance(context(now=20.0 + i / 60.0)) for i in range(5)].count(True) == 0)

    # -- the swap carries the totals --------------------------------------------
    h26, hc26 = _vh25()
    _fire(h26, hc26, 0, N25)                       # the left player scores 20 shots
    _fire(h26, hc26, 1, N25, on_target=False)      # the right player: 20 registered, none score
    left_pts = h26.tallies[0].points
    check("round 1 closed by itself with the left player ahead",
          h26.round_over and h26.verdict.kind == "win" and h26.verdict.winner == 0)
    h26.restart(hc26)
    check("N after a round is the NEXT round: round 2, sides swapped, fresh tallies",
          h26.round_no == 2 and h26.seats == (1, 0) and not h26.round_over
          and h26.verdict is None and all(t.shots == 0 for t in h26.tallies))
    check("...the totals followed the PLAYERS: the left-starting player still holds the points",
          h26.totals[0].points == left_pts and h26.totals[1].points == 0)
    # CHANGED 2026-10-19: he does not want to be told about the swap every round.
    # The swap itself is asserted above (seats == (1, 0), the totals follow the
    # PLAYER) and in the log; only the on-screen wording went.
    check("...and the half no longer announces the swap on screen",
          h26.note_text(0) == "round 2" and "swap" not in h26.note_text(0).lower())
    _fire(h26, hc26, 1, N25)       # the player now on the RIGHT is player 0
    _fire(h26, hc26, 0, N25, on_target=False)
    check("round 2: the points land on the RIGHT cell but belong to player 0, who started left",
          h26.round_over and h26.totals[0].points == left_pts + h26.tallies[1].points
          and h26.totals[1].points == 0)
    check("...and the match totals add up across rounds: 40 registered shots each",
          h26.totals[0].shots == 2 * N25 and h26.totals[1].shots == 2 * N25)
    # CHANGED 2026-10-19: the verdict is THIS ROUND's result (judged from the two
    # tallies), so it agrees with the points on the same screen. It used to judge
    # the match totals, which differ from the round from round 2 on. The match is
    # still accumulated per player above, and is in the log. (Section 36 builds a
    # round where the two disagree.)
    check("...the verdict is THIS ROUND's (the right half scored, the left did not): RIGHT wins by that margin",
          h26.verdict.kind == "win" and h26.verdict.winner == 1
          and h26.verdict.margin == h26.tallies[1].points - h26.tallies[0].points,
          f"{h26.verdict}")

    h27, hc27 = _vh25()
    _fire(h27, hc27, 0, N25); _fire(h27, hc27, 1, N25)
    h27.restart(hc27)                              # -> round 2
    _fire(h27, hc27, 0, 4)
    tot27 = list(h27.totals)
    h27.restart(hc27)                              # mid-round: restart THIS round only
    check("N mid-round restarts the round in play and swaps nothing",
          h27.round_no == 2 and h27.seats == (1, 0) and h27.tallies[0].shots == 0
          and h27.totals == tot27)
    h27.set_mode(_Static(), hc27, players=2)
    check("choosing a drill again starts a whole new match",
          h27.round_no == 1 and h27.seats == (0, 1)
          and h27.totals == [_V.SideResult(), _V.SideResult()] and not h27.round_over)

    # -- a click is never a shot ------------------------------------------------
    h28, hc28 = _vh25()
    lx28, ly28 = h28.mode.aim_point(h28._side_ctx(hc28, 0))
    rx28, ry28 = h28.mode2.aim_point(h28._side_ctx(hc28, 1))
    for _k in range(N25 + 4):
        h28.on_click(lx28, ly28, hc28)
        h28.on_click(rx28, ry28, hc28)
    check("forty-eight clicks on the aim marks change no tally, allotment, score or total",
          all((t.shots, t.hits, t.points, t.after_done) == (0, 0, 0, 0) for t in h28.tallies)
          and not h28.round_over and h28.totals == [_V.SideResult(), _V.SideResult()]
          and h28.mode.shots == 0 and h28.mode2.shots == 0)
    _fire(h28, hc28, 0, N25); _fire(h28, hc28, 1, N25)
    snap28 = ([(t.shots, t.after_done) for t in h28.tallies], list(h28.totals))
    for _k in range(6):
        h28.on_click(lx28, ly28, hc28)
        h28.on_click(rx28, ry28, hc28)
    check("...and clicks after the round is over cannot reopen it or move a total",
          snap28 == ([(t.shots, t.after_done) for t in h28.tallies], list(h28.totals))
          and h28.round_over)

    # -- the microphone footnote --------------------------------------------------
    h29, hc29 = _vh25()
    before29 = [(t.shots, t.after_done) for t in h29.tallies]
    h29.note_heard_unseen(); h29.note_heard_unseen(2)
    check("a heard-but-unseen pull is a COUNT: it creates no shot and touches no tally",
          h29.heard_unseen == 3 and before29 == [(t.shots, t.after_done) for t in h29.tallies])
    _fire(h29, hc29, 0, N25); _fire(h29, hc29, 1, N25)
    lines29 = h29.verdict_lines()
    # The microphone line left the verdict block 2026-10-19: small, at the very
    # bottom, and only when it has a count (see section 34). The block holds the
    # score and the sentence; the next-round prompt is the ROUND box.
    check("the verdict block carries no microphone line, and the screen line is empty when uncalibrated",
          not any("icrophone" in l_ for l_ in lines29) and h29.mic_screen_text() == ""
          and "not calibrated" in h29.mic_log_text())
    h29.mic_ready = True
    check("...and with a calibrated microphone the bottom line carries the count heard since the round began",
          "3 trigger pulls heard" in h29.mic_screen_text())
    check("the verdict is three lines (heading, result, and the efficiency sentence, blank when there is none), the "
          "sides' shots, points, seconds and efficiency sit in their own columns, units spelled out, and nothing "
          "says swap or press N",
          lines29[:2] == ["ROUND 1 COMPLETED", "TIED"] and len(lines29) == 3
          and h29.side_block(0)[:3] == ["LEFT", f"{N25} of {N25} shots", _V.points_of_word(
              N25 * 10, h29.cards[0].gs.ghost_points if h29.cards[0] is not None
              and (h29.cards[0].gs.scorable or h29.cards[0].gs.discarded) else None)]
          and h29.side_block(1)[:3] == ["RIGHT", f"{N25} of {N25} shots", _V.points_of_word(
              N25 * 10, h29.cards[1].gs.ghost_points if h29.cards[1] is not None
              and (h29.cards[1].gs.scorable or h29.cards[1].gs.discarded) else None)]
          and len(h29.side_block(0)) == 5 and len(h29.side_block(1)) == 5
          and not any("swap" in l_.lower() or "press n" in l_.lower()
                      for l_ in lines29 + h29.side_block(0) + h29.side_block(1)))
    check("...and it no longer mentions anything 'seen' per side (the counts are equal by construction)",
          not any("seen" in l_ for l_ in lines29))
    h29.restart(hc29)
    check("a new round starts counting the microphone afresh",
          "3 trigger pulls" not in h29.mic_screen_text())
    hsp = _Host()
    hsp.set_mode(_Static(), hc29, players=1)
    hsp.shot(modes.ShotEvent(100.0, 100.0, 1.0, 1), hc29)
    check("single player has no versus machinery: no round over, no verdict, no words, one shot spent",
          hsp.versus is False and hsp.round_over is False and hsp.verdict is None
          and hsp.verdict_lines() == [] and hsp.tallies[0].shots == 1
          and hsp.remaining(0) == N25 - 1)
    hsp1 = _Host()
    hsp1.set_mode(_Static(), hc29, players=1)
    for _k in range(N25 + 5):
        hsp1.shot(modes.ShotEvent(hsp1.mode.aim_point(hc29)[0], hsp1.mode.aim_point(hc29)[1],
                                  1.0 + _k, _k + 1), hc29)
    check("...and single player now has a magazine too: it ends the round at N, and the five after it "
          "never reach the drill",
          hsp1.round_over and hsp1.mode.shots == N25 and hsp1.tallies[0].after_done == 5
          and hsp1.last_outcome.unscored is True)

    # -- the repaint signal (shots OFF target, so no flash can ask for the repaint) --
    h30, hc30 = _vh25()
    for i in range(3):
        h30.advance(context(now=30.0 + i / 60.0))
    check("an idle versus screen asks for no repaints",
          [h30.advance(context(now=31.0 + i / 60.0)) for i in range(30)].count(True) == 0)
    _fire(h30, hc30, 0, N25, on_target=False)
    check("using an allotment asks for a repaint (the 'done' note must appear)",
          h30.advance(context(now=32.0)) is True)
    check("...and then asks for no more", h30.advance(context(now=32.02)) is False)
    _fire(h30, hc30, 0, 1, on_target=False)          # the 21st: only `after_done` moves
    check("a shot after the allotment changes nothing but its own count, and that still repaints once",
          h30.tallies[0].after_done == 1 and h30.advance(context(now=33.0)) is True
          and h30.advance(context(now=33.02)) is False)
    _fire(h30, hc30, 1, N25, on_target=False)
    check("finishing the round asks for a repaint (the verdict must appear)",
          h30.advance(context(now=34.0)) is True)
    check("...and then asks for no more while it sits there",
          [h30.advance(context(now=35.0 + i / 60.0)) for i in range(30)].count(True) == 0)
    h30.note_heard_unseen()
    check("a microphone count arriving while the verdict is up repaints it once",
          h30.advance(context(now=40.0)) is True and h30.advance(context(now=40.02)) is False)
    h31, hc31 = _vh25()
    h31.advance(context(now=50.0)); h31.advance(context(now=50.02))
    h31.mic_ready = True
    check("the microphone becoming calibrated repaints (the verdict wording changes with it)",
          h31.advance(context(now=51.0)) is True)
    painted25 = []
    h33, hc33 = _vh25()
    h33.mode.paint = lambda p_, c_, o_=h33.mode.paint: (painted25.append("L"), o_(p_, c_))
    h33.mode2.paint = lambda p_, c_, o_=h33.mode2.paint: (painted25.append("R"), o_(p_, c_))
    _render25_img = _QI25b(hc33.width, hc33.height, _QI25b.Format_RGB32)
    _pp25 = _QP25b(_render25_img)
    h33.paint(_pp25, hc33)
    live_paints = list(painted25)
    painted25.clear()
    _fire(h33, hc33, 0, N25, on_target=False); _fire(h33, hc33, 1, N25, on_target=False)
    h33.paint(_pp25, hc33)
    _pp25.end()
    check("a live round paints BOTH drills; a finished one paints NEITHER (the verdict stands on a clear screen)",
          sorted(live_paints) == ["L", "R"] and painted25 == [])

    # -- text fits, at 4K and at 1080p, measured from the font -----------------
    from PySide6.QtGui import QFontMetricsF as _FM25, QImage as _QI25, QPainter as _QP25
    from lsa.display.layers import mono as _mono25, px as _px25

    def _states25():
        """(label, host, context) for every state that puts new words on screen."""
        out_ = []
        for (w_, h_) in ((3840, 2160), (1920, 1080)):
            c_ = RenderContext(now=1.0, dt=1 / 60.0, rect=QRect(0, 0, w_, h_), stats=SessionStats())
            hs_ = _Host()
            hs_.set_mode(_Static(), c_, players=2)
            out_.append((f"{w_}x{h_} live round 1", hs_, c_))
            hd_ = _Host(); hd_.set_mode(_Static(), c_, players=2)
            _fire(hd_, c_, 0, N25)
            out_.append((f"{w_}x{h_} left done", hd_, c_))
            for tag_, lon_, ron_ in (("win", True, False), ("tie", False, False)):
                hv_ = _Host(); hv_.set_mode(_Static(), c_, players=2)
                hv_.mic_ready = True; hv_.note_heard_unseen(12)
                _fire(hv_, c_, 0, N25, on_target=lon_, step=PACE)
                _fire(hv_, c_, 1, N25, on_target=ron_, step=PACE)
                out_.append((f"{w_}x{h_} verdict {tag_}", hv_, c_))
                hr_ = _Host(); hr_.set_mode(_Static(), c_, players=2)
                hr_.mic_ready = True; hr_.note_heard_unseen(12)
                for _r in (1, 2):
                    _fire(hr_, c_, 0, N25, on_target=lon_, step=PACE)
                    _fire(hr_, c_, 1, N25, on_target=ron_, step=PACE)
                    if _r == 1:
                        hr_.restart(c_)
                out_.append((f"{w_}x{h_} verdict {tag_}, round 2", hr_, c_))
            # A DISCARDED round (2026-10-22) must fit its boxes and keep the type as large
            # as a scored one, at both sizes and both player counts: it replaces the score
            # block with the same four lines. Single player fires a burst (20 shots in
            # 0.19 s); two players have LEFT at a human pace and RIGHT in a burst, so one
            # screen shows a scored half beside a discarded one. The labels avoid
            # "single round over" (the single-player tally premise counts exactly six of
            # those) and keep "verdict" (so the score-block size bar covers them).
            hx_ = _Host(); hx_.set_mode(_Static(), c_, players=1)
            _fire(hx_, c_, 0, N25, on_target=True)
            out_.append((f"{w_}x{h_} single DISCARDED round", hx_, c_))
            hxv_ = _Host(); hxv_.set_mode(_Static(), c_, players=2)
            _fire(hxv_, c_, 0, N25, on_target=True, step=PACE)
            _fire(hxv_, c_, 1, N25, on_target=True)
            out_.append((f"{w_}x{h_} verdict, right DISCARDED", hxv_, c_))
            # SINGLE PLAYER, round over (2026-10-22). The points line gained its
            # ceiling ("65 of 75 points"), which is LONGER than "65 points" in a
            # block whose font is chosen to fit its widest line, so it is
            # rendered here rather than estimated. Not labelled "verdict": that
            # word selects the two-player type-size bars below. Three states: a
            # card with a ceiling, a round of misses ("0 of N points"), and the
            # fallback with no card at all, which must fit too.
            for tag1_, on1_ in (("all hits", True), ("all misses", False)):
                h1_ = _Host(); h1_.set_mode(_Static(), c_, players=1)
                _fire(h1_, c_, 0, N25, on_target=on1_, step=PACE)
                out_.append((f"{w_}x{h_} single round over, {tag1_}", h1_, c_))
            hn_ = _Host(); hn_.set_mode(_Static(), c_, players=1)
            _fire(hn_, c_, 0, N25, on_target=True, step=PACE)
            hn_.cards = (None, None)
            out_.append((f"{w_}x{h_} single round over, no card", hn_, c_))
        return out_

    def _render25(h_, c_):
        img_ = _QI25(c_.width, c_.height, _QI25.Format_RGB32)
        img_.fill(0xFF0A0A0A)
        p_ = _QP25(img_)
        h_.paint(p_, c_)
        p_.end()

    drawn25 = []
    _orig_draw = _Host._draw_lines

    def _recording_draw(p_, box_, lines_, start_, bold=False, floor=9):
        size_ = _orig_draw(p_, box_, lines_, start_, bold, floor)
        drawn25.append((box_, list(lines_), size_, bold))
        return size_

    _Host._draw_lines = staticmethod(_recording_draw)
    fits25, tiny25, btn25, verdict_sizes = [], [], [], []
    score_sizes = []
    single25 = {}          # label -> (points line as drawn, type size / screen height)
    try:
        for label25, hs25, cs25 in _states25():
            drawn25.clear()
            _render25(hs25, cs25)
            if "single round over" in label25:
                # The tally block, found by its own words so a draw that never
                # happened cannot pass for one that fitted.
                for box_, lines_, size_, _bd in drawn25:
                    if lines_ == hs25.side_block(0):
                        single25[label25] = (lines_[3], size_ / cs25.height)
            if "verdict" in label25 and cs25.width == 3840:
                big_ = max(drawn25, key=lambda d_: len(d_[1]))
                verdict_sizes.append((label25, big_[2] / cs25.height))
            if "verdict" in label25:
                # The SCORE block (four lines, 2026-10-22: the caveat left the
                # screen), found by its own words so a draw that never happened
                # cannot pass for one that was large enough.
                for side_ in (0, 1):
                    sl_ = hs25.score_lines(side_)
                    for box_, lines_, size_, _bd in drawn25:
                        if lines_ == sl_:
                            score_sizes.append((f"{label25} side {side_} score", size_ / cs25.height))
            for box_, lines_, size_, bold_ in drawn25:
                # Each line is measured the way the painter draws it: a line with a
                # bold run is the SUM of its runs under their own fonts, not the
                # joined string under one (2026-10-22). For a plain string this is
                # exactly what `horizontalAdvance` / `height()` gave before.
                wide_ = max(_Host.line_width(t_, size_, bold_) for t_ in lines_)
                high_ = sum(_Host.line_height(t_, size_, bold_) for t_ in lines_)
                if size_ > 9 and (wide_ > box_.width() or high_ > box_.height()):
                    fits25.append(f"{label25}: {lines_[0][:30]!r} {wide_:.0f}x{high_:.0f} in {box_.width()}x{box_.height()}")
                if size_ <= 9:
                    tiny25.append(f"{label25}: {lines_[0][:30]!r} fell to the floor")
            if hs25._button_live:
                btn_ = hs25.menu_button.rect(cs25).toAlignedRect()
                for box_, lines_, size_, _b in drawn25:
                    if box_.intersects(btn_) and not hs25.round_over:
                        btn25.append(f"{label25}: {lines_[0][:30]!r}")
    finally:
        _Host._draw_lines = staticmethod(_orig_draw)
    check("every new string fits its own box at 4K and at 1080p, measured from the font",
          not fits25, "; ".join(fits25[:3]))
    check("...and none was shrunk to the 9px floor", not tiny25, "; ".join(tiny25[:3]))
    check("...and no side's note is drawn under the MENU button",
          not btn25, "; ".join(btn25[:3]))
    print(f"         verdict type as a fraction of screen height: "
          f"{min(s_ for _l, s_ in verdict_sizes):.4f} .. {max(s_ for _l, s_ in verdict_sizes):.4f}")
    check("the verdict type is large enough to read from the line (>= 2.0% of screen height)",
          all(s_ >= 0.020 for _l, s_ in verdict_sizes),
          "; ".join(f"{l_} {s_:.4f}" for l_, s_ in verdict_sizes if s_ < 0.020))
    # The largest block above says nothing about the SCORE block, which is smaller
    # (measured two-player: score 0.0176 of the screen height at BOTH 1080p and 4K,
    # so the fractions are resolution-free). The hard 9 px floor is 0.0083 at 1080p.
    # THE CAVEAT BAR (1.2%) WAS RETIRED WITH THE CAVEAT'S PLACE ON THE SCREEN
    # (2026-10-22, his ruling): nothing is drawn from the note now, so there is no
    # type size to bound; what must hold instead is that it is NOT drawn, below.
    MIN_SCORE_FRAC = 0.015
    check("(premise) the score block was found in the drawn verdicts",
          bool(score_sizes), f"{len(score_sizes)} blocks")
    print(f"         score block {min(s_ for _l, s_ in score_sizes):.4f} of screen height "
          f"(minimum {MIN_SCORE_FRAC})")
    check(f"the SCORE block is at least {MIN_SCORE_FRAC:.1%} of screen height at 4K and 1080p",
          all(s_ >= MIN_SCORE_FRAC for _l, s_ in score_sizes),
          "; ".join(f"{l_} {s_:.4f}" for l_, s_ in score_sizes if s_ < MIN_SCORE_FRAC))
    # The single-player tally, rendered (not estimated): "65 of 75 points" is longer
    # than "65 points" and the block's font is chosen to fit its widest line.
    import re as _re25
    for lbl1_, (pts1_, frac1_) in sorted(single25.items()):
        print(f"         {lbl1_:<44} {pts1_!r:<22} type {frac1_:.4f} of screen height")
    check("(premise) the single-player tally block was drawn in all six states "
          "(4K and 1080p; all hits, all misses, no card)",
          len(single25) == 6, str(sorted(single25)))
    check("...with its CEILING when the round has a card (`N of M points`), and plain `N points` "
          "when it has none",
          all(_re25.fullmatch(r"\d+ of \d+ points", p_) for l_, (p_, _f) in single25.items()
              if "no card" not in l_)
          and all(_re25.fullmatch(r"\d+ points", p_) for l_, (p_, _f) in single25.items()
                  if "no card" in l_), str({k_: v_[0] for k_, v_ in single25.items()}))
    check("...and the type did not collapse to the 9 px floor (0.0083 of 1080p) for the longer line",
          all(f_ > 0.0083 for _p, f_ in single25.values()),
          "; ".join(f"{l_} {f_:.4f}" for l_, (_p, f_) in single25.items() if f_ <= 0.0083))

    # -- EVERY caveat the module can build, rendered, and none of them drawn ------
    # This used to bound each caveat's TYPE SIZE (the first fix was measured on its
    # easiest input and failed on the worst). The caveat left the screen on
    # 2026-10-22, so the same enumeration now proves the opposite: each possible
    # text is put on the card of a finished round and RENDERED, one player and two,
    # 4K and 1080p, and the painter is read back for it.
    import itertools as _it25
    import types as _ty25
    from lsa import scorecard as _SC25
    from lsa import ghost as _G25

    def _every_caveat25():
        # The refusal notes ("NOT A BEST: ...", every combination of four causes) are
        # GONE with the refusal (2026-10-22). What is enumerated now is every note the
        # module can still build: a discarded round's (with a phrase for its reason, and
        # the fallback for a reason that has none), the at-least and estimate captions,
        # and the no-score captions.
        out_ = []
        base_ = dict(scorable=True, discarded=False, lower_bound=False, declared_waits=0)
        for reasons_ in ([_G25.Unfit.TOO_FAST], []):
            out_.append(_SC25._note(_ty25.SimpleNamespace(**dict(
                base_, scorable=False, discarded=True,
                record=_ty25.SimpleNamespace(fit=False, reasons=list(reasons_)))),
                shots=N25, ceiling_known=True))
        for kw_ in (dict(lower_bound=True), dict(declared_waits=1), dict(declared_waits=N25)):
            out_.append(_SC25._note(_ty25.SimpleNamespace(**dict(
                base_, record=_ty25.SimpleNamespace(fit=True, reasons=[]), **kw_)),
                shots=N25, ceiling_known=True))
        for sh_, ck_ in ((1, True), (N25, False), (N25, True)):
            out_.append(_SC25._note(_ty25.SimpleNamespace(**dict(base_, scorable=False)),
                                    shots=sh_, ceiling_known=ck_))
        return sorted({t_ for t_ in out_ if t_}, key=lambda t_: (-len(t_), t_))

    caveats25 = _every_caveat25()

    def _finished25(w_, h_, players_):
        c_ = RenderContext(now=1.0, dt=1 / 60.0, rect=QRect(0, 0, w_, h_), stats=SessionStats())
        hs_ = _Host()
        hs_.set_mode(_Static(), c_, players=players_)
        _fire(hs_, c_, 0, N25, on_target=True, step=PACE)
        if players_ == 2:
            _fire(hs_, c_, 1, N25, on_target=False, step=PACE)
        return hs_, c_

    rows25, missing25, tiny_c25 = [], [], []
    drawn_c25 = []
    _orig_draw_c = _Host._draw_lines

    def _rec_draw_c(p_, box_, lines_, start_, bold=False, floor=9):
        size_ = _orig_draw_c(p_, box_, lines_, start_, bold, floor)
        drawn_c25.append((list(lines_), size_))
        return size_

    _Host._draw_lines = staticmethod(_rec_draw_c)
    try:
        for (w_, h_) in ((3840, 2160), (1920, 1080)):
            for players_ in (1, 2):
                hs_, c_ = _finished25(w_, h_, players_)
                real_cards_ = hs_.cards
                if real_cards_[0] is None:
                    missing25.append(f"{w_}x{h_} {players_}p: no card on a finished round")
                    continue
                for text_ in caveats25:
                    drawn_c25.clear()
                    hs_.cards = tuple(None if k_ is None else dataclass_replace(k_, note=text_)
                                      for k_ in real_cards_)
                    try:
                        _render25(hs_, c_)
                    finally:
                        hs_.cards = real_cards_
                    # THE CAVEAT IS NOT DRAWN (2026-10-22). Rendered with the
                    # card carrying this text, then every drawn line is read
                    # back: a draw that is merely not asked for would pass a
                    # check on the code, so the PAINTER is what is asked. The
                    # round must still have drawn its score block (else a
                    # painter that drew nothing would "not draw" the caveat).
                    rows25.append((f"{w_}x{h_} {players_}p", text_))
                    if any(text_ in ln_ for ls_, _s in drawn_c25 for ln_ in ls_):
                        tiny_c25.append(f"{w_}x{h_} {players_}p {text_!r} was DRAWN")
                    if not any(ls_ == hs_.score_lines(0) or ls_ == hs_.score_lines(0)[1:]
                               for ls_, _s in drawn_c25):
                        missing25.append(f"{w_}x{h_} {players_}p {text_!r}: the score block itself "
                                         f"was not drawn")
    finally:
        _Host._draw_lines = staticmethod(_orig_draw_c)
    print(f"         every caveat the module can build ({len(caveats25)} texts), rendered at 4K and 1080p, "
          f"one player and two")
    check("(premise) every possible caveat was rendered on a finished round, one player and two, at 4K "
          "and at 1080p, with the score block drawn each time",
          not missing25 and len(rows25) == 4 * len(caveats25) and len(caveats25) >= 8,
          f"{len(rows25)} rendered of {4 * len(caveats25)}; missing: " + "; ".join(missing25[:3]))
    check("EVERY possible caveat is absent from the screen, one player and two, 4K and 1080p: it is "
          "built on the card and logged, and never drawn",
          not tiny_c25, "; ".join(tiny_c25[:3]))

    # -- lit area and red excess at 4K, every state, composited with a hole ---
    print("     lit area / red excess of the new furniture at 4K")
    from lsa.display.layers import BulletHoles as _BH25
    worst25, worst25_label = 0.0, ""
    furn25 = 0.0
    furn_verdict25 = 0.0
    for cls25 in (_Static, EscalatingBall, _Balls25, _Ctb25):
        def _mid(h_, c_):
            _fire(h_, c_, 0, 12, t0=c_.now, on_target=True)
            _fire(h_, c_, 1, 7, t0=c_.now, on_target=False)

        def _one_done(h_, c_):
            _fire(h_, c_, 0, N25, t0=c_.now, on_target=True)
            _fire(h_, c_, 1, 5, t0=c_.now, on_target=False)

        def _over_win(h_, c_):
            _fire(h_, c_, 0, N25, t0=c_.now, on_target=True, step=PACE)
            _fire(h_, c_, 1, N25, t0=c_.now, on_target=False, step=PACE)
            h_.mic_ready = True
            h_.note_heard_unseen(4)

        def _over_tie(h_, c_):
            _fire(h_, c_, 0, N25, t0=c_.now, on_target=False, step=PACE)
            _fire(h_, c_, 1, N25, t0=c_.now, on_target=False, step=PACE)

        def _round2(h_, c_):
            _over_win(h_, c_)
            h_.restart(c_)
            _fire(h_, c_, 0, 14, t0=c_.now, on_target=False)
            _fire(h_, c_, 1, 9, t0=c_.now, on_target=True)

        def _round2_over(h_, c_):
            _over_win(h_, c_)
            h_.restart(c_)
            _fire(h_, c_, 0, N25, t0=c_.now, on_target=False, step=PACE)
            _fire(h_, c_, 1, N25, t0=c_.now, on_target=True, step=PACE)

        for slabel, setup25 in (("live, mid-round", _mid), ("one side done", _one_done),
                                ("round over, winner", _over_win),
                                ("round over, tied", _over_tie),
                                ("round 2, sides swapped", _round2),
                                ("round 2 over, match verdict", _round2_over)):
            h_, c_ = _vh25(cls25, now=100.0)
            for i in range(10):
                h_.advance(context(now=100.0 + i / 60.0))
            c_ = context(now=100.2)
            setup25(h_, c_)
            holes25 = _BH25()
            ctr25 = _aim(h_, c_, 0)
            holes25.add(1920.0, 1000.0, 1, 0)          # in the middle of the verdict text
            holes25.add(ctr25[0], ctr25[1], 2, 0)
            holes25.add(900.0, 235.0, 3, 0)            # on the scoreline row

            def _under(p_, cx_, h=h_, hl=holes25):
                hl.paint(p_, cx_); h.paint(p_, cx_)

            def _over(p_, cx_, h=h_, hl=holes25):
                h.paint(p_, cx_); hl.paint(p_, cx_)

            for zlabel, fn_ in (("holes under", _under), ("holes over", _over)):
                r_, lit_, _img = _fh_full(fn_, c_)
                if lit_ > worst25:
                    worst25, worst25_label = lit_, f"{cls25.title} / {slabel} / {zlabel}"
                check(f"{cls25.title} [{slabel}, {zlabel}]: red excess exactly 0 and inside the 2% ceiling",
                      r_ == 0 and lit_ <= 0.02, f"red {r_}, lit {lit_ * 100:.3f}%")

            def _chrome(p_, cx_, h=h_):
                h._paint_chrome(p_, cx_, h._layout(cx_))
            r_, lit_, _img = _fh_full(_chrome, c_)
            # TWO populations (2026-10-22). While shooting is possible the furniture
            # is GREY_MID and spends NOTHING. The round-over screen's NUMBERS are
            # GREY_HI (65, above the 60 lit line) by his ruling, so that screen
            # spends a little, on purpose; it is bounded below instead of being
            # held to zero, and the 2% total in the check above still applies to it.
            if "over" in slabel:
                furn_verdict25 = max(furn_verdict25, lit_)
            else:
                furn25 = max(furn25, lit_)
            check(f"{cls25.title} [{slabel}]: the furniture alone is red excess 0",
                  r_ == 0, f"red {r_}")
    print(f"         WORST total lit across every drill, state and z order: {worst25 * 100:.3f}% ({worst25_label})")
    print(f"         WORST lit by the new furniture alone: {furn25 * 100:.4f}%")
    check("the furniture alone spends less than 0.36% of the screen (the headroom Static Targets leaves)",
          furn25 < 0.0036, f"{furn25 * 100:.4f}%")
    check("...and in fact spends NOTHING: every new glyph stays under luminance 60 at the measured "
          "ceiling (if this fails because the ceiling rose, the text now costs budget: re-measure)",
          furn25 == 0.0, f"{furn25 * 100:.4f}%")
    # The round-over screen's numbers are GREY_HI = 65, five above the lit line, by his
    # ruling of 2026-10-22 ("one step up as GREY_HI"), so that screen is allowed to
    # spend a little and is bounded instead of being held to zero. The bar is HALF the
    # 2% budget: there are no targets on that screen, only the shootable NEXT ROUND box,
    # and the check above still holds its total to the whole 2%.
    print(f"         round-over furniture lit above luminance 60 (the lifted numbers): {furn_verdict25 * 100:.4f}%")
    check("the round-over screen's lifted numbers spend under 1% of the screen above luminance 60 "
          "(half the budget; they are the only thing on it above 60)",
          furn_verdict25 < 0.01, f"{furn_verdict25 * 100:.4f}%")

    # ------------------------------------------------------------------
    # 26. Reaction in versus: ONE shared reveal, mirrored, and no reveal can hang
    # ------------------------------------------------------------------
    print()
    print("26. Reaction in versus: one shared reveal, mirrored; no reveal can hang")
    import contextlib as _cl26
    from lsa.modes.reaction import ReactionDrill as _RD26, SharedReveal as _SR26
    from lsa.audio.mixer import APPEAR_SOUND as _APPEAR26
    from lsa.display.layers import BulletHoles as _BH26
    N26 = _DEFAULT_MAG
    DT26 = 1 / 60.0
    _lay26 = _split.layout(_full22.rect)

    @_cl26.contextmanager
    def _seeded26(seed):
        """Make the host's OWN pairing call hand out a seeded link, so the seam
        under test (`ModeHost._start_all` -> `new_link` -> `pair_with`) is the
        real one and only the random stream is fixed."""
        orig = _RD26.__dict__["new_link"]
        _RD26.new_link = classmethod(lambda cls: _seed_link26(cls, seed))
        try:
            yield
        finally:
            _RD26.new_link = orig

    def _seed_link26(cls, seed):
        link = cls.new_link_for(2)
        link.rng.seed(seed)
        return link

    def _rx26(seed=5, **attrs):
        with _seeded26(seed):
            h = _Host()
            h.set_mode(_RD26(), _full22, players=2)
        return h

    def _until26(h, t, pred, limit):
        """Tick the host in virtual time until `pred()`; returns the time, or
        None when the limit passes (a hang is a failed check, not a loop)."""
        end = t + limit
        while t < end:
            t += DT26
            h.advance(context(now=t))
            if pred():
                return t
        return None

    # -- 26a. the seam: the host pairs the two halves, once per round --------
    h26 = _rx26()
    L26, R26 = h26.mode, h26.mode2
    check("the host hands BOTH halves the SAME reveal object (the shared draw)",
          L26._reveal is R26._reveal and L26._reveal.seats == 2)
    check("...and they own the two seats (0 = left, 1 = right)", (L26.seat, R26.seat) == (0, 1))
    link_a = L26._reveal
    h26.restart(_full22)
    check("N mid-round pairs a FRESH shared reveal (no stale draw carries across)",
          h26.mode._reveal is not link_a and h26.mode._reveal is h26.mode2._reveal)
    link_b = h26.mode._reveal
    h26.round_over = True
    h26.restart(_full22)                 # N after a round: the next round
    check("N after a round pairs a fresh one too",
          h26.mode._reveal is not link_b and h26.mode._reveal is h26.mode2._reveal
          and h26.round_no == 2)
    hs26 = _Host()
    hs26.set_mode(_RD26(), _full22, players=1)
    check("single player holds a ONE-seat reveal with no timeout (same code path, no expiry)",
          hs26.mode._reveal.seats == 1 and hs26.mode._reveal.timeout_s is None
          and hs26.mode2 is None)
    check("a drill that shares nothing hands back no link and pairing is a no-op",
          _Static.new_link() is None and _Static().pair_with(None, 0) is None)

    # -- 26b/c. simultaneity and mirroring over 40 seeded reveals -------------
    cx26 = _lay26.centre_x
    same_tick = origin_equal = times_ok = mirrored = one_cue = drawn_ok = True
    max_attempts = 1
    n_rev = 0
    vetoes26 = []
    _orig_veto = _SR26.veto

    def _spy_veto(self):
        before = self.appear_at
        _orig_veto(self)
        vetoes26.append(self.appear_at == before)

    _SR26.veto = _spy_veto
    try:
        for seed26 in (5, 6):
            # 19 reveals a host: the 20th hit on each side would close the round
            h26 = _rx26(seed26)
            L26, R26 = h26.mode, h26.mode2
            lk26 = L26._reveal
            t26, n_here = 0.0, 0
            while n_here < N26 - 1:
                t26 = _until26(h26, t26, lambda: L26.state == "live" or R26.state == "live", 20.0)
                if t26 is None:
                    break
                n_rev += 1
                n_here += 1
                same_tick &= (L26.state == "live" and R26.state == "live")
                origin_equal &= (L26.shown_at == R26.shown_at == lk26.live_at)
                max_attempts = max(max_attempts, lk26.attempts)
                mirrored &= (abs(L26.x + R26.x - 2.0 * cx26) < 1e-6 and L26.y == R26.y)
                one_cue &= (h26.take_sounds() == [_APPEAR26])
                tl, tr = L26.shown_at + 0.20, L26.shown_at + 0.45
                h26.shot(modes.ShotEvent(L26.x, L26.y, tl, 2 * n_rev), context(now=tl))
                h26.shot(modes.ShotEvent(R26.x, R26.y, tr, 2 * n_rev + 1), context(now=tr))
                times_ok &= bool(L26.times and R26.times) and (abs(L26.times[-1] - 0.20) < 1e-9
                                                               and abs(R26.times[-1] - 0.45) < 1e-9)
                t26 = tr
            drawn_ok &= (lk26.reveal_no == n_here + 1)
    finally:
        _SR26.veto = _orig_veto
    check("38 reveals completed with both halves hitting (nothing hung)", n_rev == 2 * (N26 - 1), f"{n_rev}")
    check("both halves go live in the SAME tick, every reveal", same_tick)
    check("both halves read ONE instant as their clock origin (shown_at == the link's live_at)",
          origin_equal)
    check("so the two measured times share an origin: hits at +0.20 and +0.45 read exactly that",
          times_ok)
    check("ONE draw per reveal for the pair, not one per seat (reveals drawn == reveals shown + 1 pending)",
          drawn_ok, f"{n_rev} shown")
    check("the two targets are MIRROR IMAGES about the divider, every reveal (even after vetoes)",
          mirrored)
    check("(premise) vetoes DID happen in that run, so mirroring survived a redraw",
          max_attempts > 1 and len(vetoes26) > 0, f"max attempts {max_attempts}, {len(vetoes26)} vetoes")
    check("a veto keeps the appearance time (the redraw changes only the position)",
          all(vetoes26) and len(vetoes26) > 0)
    # CHANGED 2026-10-19: both seats queue the cue and the HOST drains both halves
    # and collapses identical cues into one. It used to be queued by the left seat
    # alone, which fell silent for the right half's reveals once the left half
    # had finished and the right was the only one ticking.
    check("exactly one appear cue per reveal, from the host's drain of both halves", one_cue)
    print(f"         {n_rev} reveals: max {max_attempts} position draws for one, {len(vetoes26)} vetoes total")

    # -- 26d. a veto from ONE side's MENU keep-out redraws for BOTH -----------
    # Both corners: the keep-out on the LEFT half is judged by the seat ticked
    # first, so its veto lands before the other seat has looked. The keep-out
    # on the RIGHT half is judged by the seat ticked SECOND, after the first
    # has already accepted -- the order in which a veto that fails to clear the
    # other seat's acceptance leaves the pair on two different positions.
    _saved_m = _Mode.MENU_KEEPOUT_MARGIN_FRAC
    _saved_c = _Mode.MENU_CORNER
    try:
        _Mode.MENU_KEEPOUT_MARGIN_FRAC = 0.30        # a big keep-out in one corner
        for corner26, side26 in (("top left", 0), ("top right", 1)):
            _Mode.MENU_CORNER = corner26
            h26k = _rx26(9)
            Lk, Rk = h26k.mode, h26k.mode2
            own = Lk if side26 == 0 else Rk
            ok_k = True
            vetoed = 0
            tk = 0.0
            for i in range(N26 - 1):
                tk = _until26(h26k, tk, lambda: Lk.state == "live", 20.0)
                if tk is None:
                    ok_k = False
                    break
                vetoed += (Lk._reveal.attempts > 1)
                ok_k &= (_Mode.clear_of_menu(h26k._side_ctx(_full22, side26), own.x, own.y, own.r)
                         and abs(Lk.x + Rk.x - 2.0 * cx26) < 1e-6 and Lk.y == Rk.y)
                tl = Lk.shown_at + 0.2
                h26k.shot(modes.ShotEvent(Lk.x, Lk.y, tl, 2 * i), context(now=tl))
                h26k.shot(modes.ShotEvent(Rk.x, Rk.y, tl, 2 * i + 1), context(now=tl))
                tk = tl
            check(f"[{corner26}] the keep-out on side {side26} vetoes for the pair: no target on the "
                  f"button, both halves still mirrored (19 reveals)",
                  ok_k and vetoed > 0, f"{vetoed} reveals needed a redraw")
    finally:
        _Mode.MENU_KEEPOUT_MARGIN_FRAC = _saved_m
        _Mode.MENU_CORNER = _saved_c

    # -- 26e. exhaustion: bounded, never a hang; the one place mirroring is not promised
    _saved_f = _RD26.MIN_MOVE_FRAC
    try:
        _RD26.MIN_MOVE_FRAC = 0.995         # almost no position is far enough from the last
        h26x = _rx26(3)
        Lx, Rx = h26x.mode, h26x.mode2
        tx = 0.0
        live_both = exhausted = in_halves = 0
        mirrored_x = 0
        for i in range(12):
            tx = _until26(h26x, tx, lambda: Lx.state == "live" or Rx.state == "live", 20.0)
            if tx is None:
                break
            live_both += (Lx.state == "live" and Rx.state == "live")
            lkx = Lx._reveal
            exhausted += lkx.exhausted
            fl, fr = _RD26.playfield(h26x._side_ctx(_full22, 0)), _RD26.playfield(h26x._side_ctx(_full22, 1))
            in_halves += (fl.left() - 1 <= Lx.x <= fl.right() + 1 and fr.left() - 1 <= Rx.x <= fr.right() + 1)
            mirrored_x += (abs(Lx.x + Rx.x - 2.0 * cx26) < 1e-6 and Lx.y == Rx.y)
            tl = Lx.shown_at + 0.2
            h26x.shot(modes.ShotEvent(Lx.x, Lx.y, tl, 2 * i), context(now=tl))
            h26x.shot(modes.ShotEvent(Rx.x, Rx.y, tl, 2 * i + 1), context(now=tl))
            tx = tl
        check("exhaustion (every redraw rejected) never hangs: 12 of 12 reveals went live on both halves",
              live_both == 12, f"{live_both} of 12")
        check("(premise) the fallback path really ran", exhausted >= 6, f"{exhausted} exhausted")
        check("on exhaustion each half takes ITS OWN furthest free corner, inside its own half",
              in_halves == 12, f"{in_halves} of 12")
        print(f"         exhaustion: {exhausted} of 12 reveals fell back; {mirrored_x} of 12 still mirrored "
              f"(not promised: each side picks its own corner)")
    finally:
        _RD26.MIN_MOVE_FRAC = _saved_f

    # -- 26f. single player is EXACTLY what it was ----------------------------
    GOLD = {   # recorded from the pre-change drill, seed 11, 1 seat (tests/_out/react/trace_before.txt)
        (3840, 2160): [(2.240472973072583, 2137.8093794296055, 1725.7895020499543),
                       (2.270995161229479, 770.9085998002932, 1071.8868633475226),
                       (2.648730256498644, 2987.601863925257, 409.28791911041594)],
        (1920, 1080): [(2.240472973072583, 1068.9046897148028, 862.8947510249772),
                       (2.270995161229479, 385.4542999001466, 535.9434316737613),
                       (2.648730256498644, 1493.8009319626285, 204.64395955520797)],
        (2560, 1080): [(2.240472973072583, 1427.1590168063203, 862.8947510249772),
                       (2.270995161229479, 503.63691996726766, 535.9434316737613),
                       (2.648730256498644, 2001.3061303753982, 204.64395955520797)],
    }
    for (gw, gh), want in GOLD.items():
        r26 = _RD26()
        r26._rng.seed(11)
        gctx = lambda n, w=gw, h=gh: RenderContext(now=n, dt=DT26, rect=QRect(0, 0, w, h),
                                                     stats=SessionStats())
        r26.start(gctx(0.0))
        got = []
        t = 0.0
        for _ in range(3):
            wait = r26.next_at - t
            t = r26.next_at + 0.0123
            r26.advance(gctx(t))
            got.append((wait, r26.x, r26.y))
            te = r26.shown_at + 0.2
            r26.shot(modes.ShotEvent(r26.x, r26.y, te, 1), gctx(te))
            t = te
        check(f"single player at {gw}x{gh}: the first three waits and positions are unchanged "
              f"for seed 11 (same random stream, same geometry)",
              all(abs(a - b) < 1e-9 for g, w_ in zip(got, want) for a, b in zip(g, w_)),
              f"{got[0]} vs {want[0]}")
    r26 = _RD26()
    r26.start(context(now=0.0))
    r26.advance(context(now=r26.next_at + 0.01))
    live_pos = (r26.x, r26.y)
    for i in range(400):             # 400 s with the target up and no shot at all
        r26.advance(context(now=r26.shown_at + 1.0 + i))
    check("single player: NO timeout, the target stays up until it is hit (400 s)",
          r26.state == "live" and (r26.x, r26.y) == live_pos and r26.too_slow == 0)
    check("single player: a hit after all that is still a hit and still timed from the reveal",
          r26.shot(modes.ShotEvent(r26.x, r26.y, r26.shown_at + 400.0, 1), context(now=r26.shown_at + 400.0)).hit
          and abs(r26.times[-1] - 400.0) < 1e-9)

    # -- 26g. a reveal can NOT hang: a seat that never hits ------------------
    h26s = _rx26(21)
    A26, B26 = h26s.mode, h26s.mode2
    lk = A26._reveal
    ts, lives, hits_a = 0.0, [], 0
    while ts < 60.0:
        ts += DT26
        h26s.advance(context(now=ts))
        if A26.state == "live" and (not lives or lives[-1] != lk.live_at):
            lives.append(lk.live_at)
            tha = A26.shown_at + 0.25
            out = h26s.shot(modes.ShotEvent(A26.x, A26.y, tha, len(lives)), context(now=max(ts, tha)))
            hits_a += out.hit
    gaps = [b - a for a, b in zip(lives, lives[1:])]
    max_gap = _RD26.MAX_WAIT + _RD26.REVEAL_TIMEOUT_S + 3 * DT26
    check("B never fires: A still gets a fresh target every reveal for 60 s (>= 9 reveals)",
          len(lives) >= 9 and hits_a == len(lives), f"{len(lives)} reveals, {hits_a} A hits")
    check("...no reveal arrived later than MAX_WAIT + the timeout after the last",
          bool(gaps) and max(gaps) <= max_gap, f"worst gap {max(gaps):.2f}s vs {max_gap:.2f}s")
    check("B is charged TOO SLOW for each, never a hit and never a false start",
          B26.too_slow >= len(lives) - 1 and B26.hits == 0 and B26.false_starts == 0
          and B26.shots == 0 and B26.times == [], f"too_slow {B26.too_slow}")
    check("...and B's status says so", "too slow" in B26.status(), B26.status())
    print(f"         60 s with B silent: {len(lives)} reveals, worst gap {max(gaps):.2f}s "
          f"(bound {max_gap:.2f}s), B too slow {B26.too_slow}")

    # the timeout boundary, and what each outcome is
    h26t = _rx26(4)
    A26, B26 = h26t.mode, h26t.mode2
    tt = _until26(h26t, 0.0, lambda: A26.state == "live", 20.0)
    live_at = A26._reveal.live_at
    h26t.advance(context(now=live_at + _RD26.REVEAL_TIMEOUT_S - 0.02))
    check("2.98 s after the reveal both targets are still up", A26.state == "live" and B26.state == "live")
    out_a = h26t.shot(modes.ShotEvent(A26.x, A26.y, live_at + 2.97, 1),
                      context(now=live_at + _RD26.REVEAL_TIMEOUT_S - 0.02))
    check("a hit inside the window is a hit", out_a.hit and abs(A26.times[-1] - 2.97) < 1e-9)
    h26t.advance(context(now=live_at + _RD26.REVEAL_TIMEOUT_S + 1e-6))
    check("at the timeout the unresolved seat is resolved TOO SLOW and its target goes",
          B26.state == "wait" and B26.too_slow == 1 and B26.hits == 0 and B26.false_starts == 0
          and any(f[3] < 0 for f in B26.flashes))
    check("...the flash says so in words (it carries the TOO SLOW sentinel the painter reads)",
          B26.flashes[-1][3] == -1.0)
    check("...and the other half, which hit, is untouched", A26.too_slow == 0 and A26.hits == 1)
    check("a new reveal is armed once both seats have resolved", A26._reveal.phase == "pending"
          and A26._reveal.appear_at > live_at + _RD26.REVEAL_TIMEOUT_S)

    # the timeout is judged at the SHOT'S OWN time (ev.t), not the tick that handles it
    for label26, t_hit, want_hit in (("2.99 s", 2.99, True), ("3.01 s", 3.01, False)):
        hl = _rx26(61)
        La, Lb = hl.mode, hl.mode2
        _until26(hl, 0.0, lambda: La.state == "live", 20.0)
        lat = La._reveal.live_at
        # handled LATE: the tick that sees it is 3.02 s in, no tick in between
        out_l = hl.shot(modes.ShotEvent(La.x, La.y, lat + t_hit, 1), context(now=lat + 3.02))
        if want_hit:
            check(f"a hit fired at {label26} scores even though it is handled at 3.02 s, and is "
                  f"charged neither too slow nor a false start",
                  out_l.hit and La.too_slow == 0 and La.false_starts == 0
                  and abs(La.times[-1] - t_hit) < 1e-9, f"hit={out_l.hit}, slow={La.too_slow}")
        else:
            check(f"a hit fired at {label26} (after the timeout) is NOT scored",
                  not out_l.hit and La.hits == 0 and La.too_slow == 1,
                  f"hit={out_l.hit}, slow={La.too_slow}")
    # A SHOT FIRED JUST LATE IS NOT FIRED EARLY. The first shot after an expiry
    # is forgiven (counted, scores nothing, no false start, "TOO LATE"); a
    # further shot in that wait is a genuine false start.
    hf = _rx26(63)
    Fa, Fb = hf.mode, hf.mode2
    _until26(hf, 0.0, lambda: Fa.state == "live", 20.0)
    lat = Fa._reveal.live_at
    o1 = hf.shot(modes.ShotEvent(Fa.x, Fa.y, lat + 3.01, 1), context(now=lat + 3.02))
    check("a shot at 3.01 s: charged too slow, counted as a shot, scores nothing, and is NOT a false start",
          (not o1.hit) and Fa.too_slow == 1 and Fa.shots == 1 and Fa.hits == 0
          and Fa.false_starts == 0 and o1.target_id == "too-late"
          and hf.tallies[0].shots == 1 and hf.tallies[0].points == 0,
          f"slow {Fa.too_slow}, shots {Fa.shots}, false starts {Fa.false_starts}, {o1.target_id}")
    check("...and the mark says TOO LATE (never TOO EARLY)",
          Fa.flashes[-1][3] == -2.0 and not any(f[3] == 0.0 for f in Fa.flashes))
    o2 = hf.shot(modes.ShotEvent(Fa.x, Fa.y, lat + 3.2, 2), context(now=lat + 3.2))
    check("the SECOND shot in that same wait IS a false start",
          o2.target_id == "false-start" and Fa.false_starts == 1 and Fa.shots == 2)
    # a shot in a wait that followed a HIT is a false start on the FIRST shot
    hh2 = _rx26(64)
    Ha, Hb = hh2.mode, hh2.mode2
    _until26(hh2, 0.0, lambda: Ha.state == "live", 20.0)
    lh = Ha._reveal.live_at
    hh2.shot(modes.ShotEvent(Ha.x, Ha.y, lh + 0.3, 1), context(now=lh + 0.3))
    oh = hh2.shot(modes.ShotEvent(Ha.x, Ha.y, lh + 0.4, 2), context(now=lh + 0.4))
    check("a shot in the wait after a HIT is a false start on the very first shot, as before",
          oh.target_id == "false-start" and Ha.false_starts == 1)
    # the forgiveness does not survive into the next reveal
    hn = _rx26(65)
    Na, Nb = hn.mode, hn.mode2
    _until26(hn, 0.0, lambda: Na.state == "live", 20.0)
    ln = Na._reveal.live_at
    hn.advance(context(now=ln + 3.0 + 1e-6))               # expires: forgiveness now armed
    tn_ = _until26(hn, ln + 3.0 + 1e-6, lambda: Na.state == "live", 20.0)   # the NEXT reveal
    ln2 = Na._reveal.live_at
    hn.shot(modes.ShotEvent(Na.x, Na.y, ln2 + 0.2, 1), context(now=ln2 + 0.2))        # a hit
    on = hn.shot(modes.ShotEvent(Na.x, Na.y, ln2 + 0.3, 2), context(now=ln2 + 0.3))   # early again
    check("the forgiveness does not survive into the next reveal: a shot after the next HIT is a false start",
          tn_ is not None and Na.hits == 1 and on.target_id == "false-start" and Na.false_starts == 1)
    # single player: nothing can expire, so no shot is ever forgiven
    rs = _RD26()
    rs.start(context(now=0.0))
    rs.advance(context(now=rs.next_at + 0.01))
    for i in range(100):
        rs.advance(context(now=rs.shown_at + 5.0 * (i + 1)))
    check("single player: a target never expires, so nothing is ever forgiven or charged too slow",
          rs.state == "live" and rs.too_slow == 0 and not rs._late_grace)
    rs2 = _RD26()
    rs2.start(context(now=0.0))
    check("single player: a shot before the target appears is still a false start",
          rs2.shot(modes.ShotEvent(5.0, 5.0, 0.5, 1), context(now=0.5)).target_id == "false-start")

    # two timeouts in a row must not put the target back where it was
    hm = _rx26(62)
    Lm = hm.mode
    pos_m, tm = [], 0.0
    for _ in range(4):
        tm = _until26(hm, tm, lambda: Lm.state == "live" and (not pos_m or (Lm.x, Lm.y) != pos_m[-1]),
                      20.0)
        pos_m.append((Lm.x, Lm.y))
        tm = Lm._reveal.live_at + _RD26.REVEAL_TIMEOUT_S + 1e-6
        hm.advance(context(now=tm))
    inner_m, far_m, _r = Lm._geometry(hm._side_ctx(_full22, 0))
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pos_m, pos_m[1:])]
    check("a seat with NO hits that times out repeatedly still gets a target far from the last one",
          len(pos_m) == 4 and Lm.hits == 0 and Lm.too_slow >= 3 and all(d >= far_m - 1e-6 for d in steps),
          f"steps {[round(d) for d in steps]} vs {far_m:.0f}")

    # -- 26h. the next reveal waits for BOTH; a late partner is waited for, not skipped
    h26w = _rx26(8)
    A26, B26 = h26w.mode, h26w.mode2
    _until26(h26w, 0.0, lambda: A26.state == "live", 20.0)
    n0 = A26._reveal.reveal_no
    live_at = A26._reveal.live_at
    h26w.shot(modes.ShotEvent(A26.x, A26.y, live_at + 0.2, 1), context(now=live_at + 0.2))
    check("A hit alone does NOT draw the next reveal while B is still unresolved",
          A26._reveal.reveal_no == n0 and A26._reveal.phase == "live")
    h26w.shot(modes.ShotEvent(B26.x, B26.y, live_at + 1.1, 2), context(now=live_at + 1.1))
    check("B's hit resolves the last seat: exactly ONE next reveal is drawn",
          A26._reveal.reveal_no == n0 + 1 and A26._reveal.phase == "pending")
    check("A firing while it waits for B is a false start (a shot at nothing), never scored",
          h26w.shot(modes.ShotEvent(5.0, 5.0, live_at + 1.2, 3), context(now=live_at + 1.2)).target_id
          == "false-start")

    # -- 26i. ... and the ROUND still closes on twenty REGISTERED shots -------
    def _round26(seed, silent_side):
        """A drives hits; `silent_side` never hits (fires wide, one a second).
        Returns (host, virtual seconds to close)."""
        h = _rx26(seed)
        a, b = (h.mode, h.mode2) if silent_side == 1 else (h.mode2, h.mode)
        hit_side, wide_side = (0, 1) if silent_side == 1 else (1, 0)
        tt_, next_wide, n = 0.0, 0.5, 0
        seen_live = None
        while tt_ < 400.0 and not h.round_over:
            tt_ += DT26
            h.advance(context(now=tt_))
            if a.state == "live" and a._reveal.live_at != seen_live:
                seen_live = a._reveal.live_at
                n += 1
                th = a.shown_at + 0.3
                h.shot(modes.ShotEvent(a.x, a.y, th, n), context(now=max(tt_, th)))
            if tt_ >= next_wide:
                next_wide += 1.0
                wx = _lay26.side(wide_side).left() + 30.0
                h.shot(modes.ShotEvent(wx, 400.0, tt_, 1000 + n), context(now=tt_))
        return h, tt_

    for silent in (1, 0):
        hr26, tclose = _round26(33, silent)
        check(f"side {silent} never hits: the round still CLOSES, on twenty registered shots each "
              f"({tclose:.0f} s of play)",
              hr26.round_over and all(t.shots == N26 for t in hr26.tallies) and tclose < 200.0,
              f"over={hr26.round_over}, shots {[t.shots for t in hr26.tallies]}")
        check(f"...with a verdict, from equal denominators",
              hr26.verdict is not None and hr26.verdict.kind in ("win", "tie"))
    # a side that has used up its twenty and will never hit again does not block the other
    hd26 = _rx26(40)
    for i in range(N26):
        hd26.shot(modes.ShotEvent(_lay26.left.left() + 30.0, 400.0, 0.1 + i * 0.01, i + 1),
                  context(now=0.1 + i * 0.01))
    check("(premise) the left side has registered its twenty", hd26.tallies[0].shots == N26
          and hd26._side_done(0))
    B26 = hd26.mode2
    td, nb, seen_live = 0.0, 0, None
    while td < 300.0 and not hd26.round_over:
        td += DT26
        hd26.advance(context(now=td))
        if B26.state == "live" and B26._reveal.live_at != seen_live:
            seen_live = B26._reveal.live_at
            nb += 1
            th = B26.shown_at + 0.3
            hd26.shot(modes.ShotEvent(B26.x, B26.y, th, 100 + nb), context(now=max(td, th)))
    check("a side that is DONE (twenty used) never blocks the other: B keeps getting targets "
          "and closes the round", hd26.round_over and hd26.tallies[1].shots == N26 and td < 200.0,
          f"{nb} reveals, closed at {td:.0f} s")

    # -- 26j. repaint budget in versus ----------------------------------------
    hq = _rx26(12)
    asked = sum(1 for i in range(180) if hq.advance(context(now=i * DT26)))
    check("versus Reaction, three seconds of waiting then both live: almost no repaints requested",
          asked <= 5, f"{asked} repaints in 180 ticks")
    hq = _rx26(13)
    tq, ask_q, tot, unexplained = 0.0, 0, 0, 0
    while tq < 60.0:                      # a whole minute, B silent: timeouts and flashes included
        tq += DT26
        tot += 1
        sig = (hq.mode.state, hq.mode2.state, len(hq.mode.flashes), len(hq.mode2.flashes),
               tuple((t_.shots, t_.points) for t_ in hq.tallies))
        asked_q = bool(hq.advance(context(now=tq)))
        sig2 = (hq.mode.state, hq.mode2.state, len(hq.mode.flashes), len(hq.mode2.flashes),
                tuple((t_.shots, t_.points) for t_ in hq.tallies))
        ask_q += asked_q
        # a repaint is explained by a transition, a tally change, or a flash
        # that is fading; anything else is the running-counter fault again
        if asked_q and sig == sig2 and not (sig[2] or sig[3]):
            unexplained += 1
        if hq.mode.state == "live" and tq - hq.mode._reveal.live_at < DT26 * 1.5:
            hq.shot(modes.ShotEvent(hq.mode.x, hq.mode.y, tq, tot), context(now=tq))
    check("a full minute with a silent partner: every repaint is a transition, a tally change or a "
          "fading flash (none unexplained)", unexplained == 0, f"{unexplained} of {ask_q}")
    check("...and no more than a quarter of the ticks (flashes fade for 0.45 s)",
          ask_q < 0.25 * tot, f"{ask_q} of {tot} ticks")

    # -- 26k. lit area and red excess at 4K, every state ----------------------
    print("     lit area / red excess at 3840x2160 (ceiling 2%):")
    LIT26 = 0.02
    worst26 = (0.0, "")

    def _measure26(label, fn, c_):
        r_, lit_, _img = _fh_full(fn, c_)
        print(f"         {label:<54} lit {lit_ * 100:.3f}%  red excess {r_}")
        check(f"{label}: red excess exactly 0", r_ == 0, f"worst {r_}")
        check(f"{label}: inside the 2% lit-area ceiling", lit_ <= LIT26, f"{lit_ * 100:.3f}%")
        return lit_

    def _state26(seed):
        h = _rx26(seed)
        a, b = h.mode, h.mode2
        _until26(h, 0.0, lambda: a.state == "live", 20.0)
        return h, a, b, a._reveal.live_at

    h_, a_, b_, la_ = _state26(50)
    tn = la_ + 0.1
    lit_states = {}
    lit_states["versus: both live"] = _measure26("versus: both live", h_.paint, context(now=tn))
    hw_ = _rx26(51)
    lit_states["versus: both waiting"] = _measure26("versus: both waiting", hw_.paint, context(now=0.5))
    h_.shot(modes.ShotEvent(a_.x, a_.y, tn, 1), context(now=tn))
    tf = tn + 0.05
    lit_states["versus: one hit (flash), one live"] = _measure26(
        "versus: one flashing, one live", h_.paint, context(now=tf))
    h_.shot(modes.ShotEvent(b_.x, b_.y, tn + 0.02, 2), context(now=tn + 0.02))
    lit_states["versus: both flashing"] = _measure26("versus: both flashing", h_.paint, context(now=tf))
    h2_, a2_, b2_, la2_ = _state26(52)
    h2_.advance(context(now=la2_ + _RD26.REVEAL_TIMEOUT_S + 1e-6))
    lit_states["versus: both TOO SLOW flashing"] = _measure26(
        "versus: both flashing TOO SLOW", h2_.paint, context(now=la2_ + _RD26.REVEAL_TIMEOUT_S + 0.05))

    h2l, a2l, b2l, la2l = _state26(54)
    tl_ = la2l + _RD26.REVEAL_TIMEOUT_S + 1e-6
    h2l.advance(context(now=tl_))
    for m_, sd_ in ((a2l, 0), (b2l, 1)):
        h2l.shot(modes.ShotEvent(m_.x, m_.y, tl_ + 0.01, 40 + sd_), context(now=tl_ + 0.01))
    lit_states["versus: both TOO LATE flashing"] = _measure26(
        "versus: both flashing TOO LATE (after TOO SLOW)", h2l.paint, context(now=tl_ + 0.05))
    # THE WORST CASE THE CODE CAN BUILD: both targets up, and on each half six
    # flashes of all three kinds at full strength, with the opening instruction
    # still on screen. Beyond anything real shooting reaches (about five shots a
    # second against a 0.45 s flash is two or three at once), so it is an upper bound.
    h3_, a3_, b3_, la3_ = _state26(53)
    tw = la3_ + 0.1
    for m_ in (a3_, b3_):
        f_ = _RD26.playfield(h3_._side_ctx(_full22, 0 if m_ is a3_ else 1))
        for k in range(6):
            x_ = f_.left() + f_.width() * (0.15 + 0.14 * k)
            m_.flashes.append((x_, f_.top() + f_.height() * (0.2 + 0.1 * (k % 3)), tw,
                               (0.35, 0.0, -1.0)[k % 3]))
    lit_states["versus: WORST CASE"] = _measure26("versus: WORST CASE (live + 6 flashes a side + help)",
                                                  h3_.paint, context(now=tw))
    holes26 = _BH26()
    holes26.add(a3_.x, a3_.y, 1, 0)
    holes26.add(b3_.x, b3_.y, 2, 0)

    def _under26(p_, c_, h=h3_, hl=holes26):
        hl.paint(p_, c_)
        h.paint(p_, c_)

    def _over26(p_, c_, h=h3_, hl=holes26):
        h.paint(p_, c_)
        hl.paint(p_, c_)

    _measure26("versus: WORST CASE + fresh holes (holes under)", _under26, context(now=tw))
    _measure26("versus: WORST CASE + fresh holes (holes over)", _over26, context(now=tw))
    # single player, the same states, for comparison
    hs1 = _Host()
    hs1.set_mode(_RD26(), context(now=0.0), players=1)
    r1_ = hs1.mode
    r1_._rng.seed(7)
    r1_.start(context(now=0.0))
    r1_.advance(context(now=r1_.next_at + 0.01))
    hs0 = _Host()
    hs0.set_mode(_RD26(), context(now=0.0), players=1)
    _measure26("single player: waiting", hs0.paint, context(now=0.5))
    lit_single_live = _measure26("single player: live", hs1.paint, context(now=r1_.shown_at + 0.1))
    for k in range(6):
        r1_.flashes.append((400.0 + 300.0 * k, 500.0 + 200.0 * (k % 3), r1_.shown_at + 0.1,
                            (0.35, 0.0, -1.0)[k % 3]))
    lit_single_worst = _measure26("single player: WORST CASE (live + 6 flashes + help)", hs1.paint,
                                  context(now=r1_.shown_at + 0.1))
    import numpy as _np26
    from PySide6.QtGui import QImage as _QI26, QPainter as _QP26
    import cv2 as _cv26
    for (aw, ah) in ((1920, 1080), (2560, 1080)):
        mk = lambda n, w=aw, h=ah: RenderContext(now=n, dt=DT26, rect=QRect(0, 0, w, h),
                                                 stats=SessionStats())
        with _seeded26(71):
            ha = _Host()
            ha.set_mode(_RD26(), mk(0.0), players=2)
        ta = _until26(ha, 0.0, lambda: ha.mode.state == "live", 20.0)
        tw_ = ha.mode._reveal.live_at + 0.1
        for m_, sd in ((ha.mode, 0), (ha.mode2, 1)):
            f_ = _RD26.playfield(ha._side_ctx(mk(tw_), sd))
            for k in range(6):
                m_.flashes.append((f_.left() + f_.width() * (0.15 + 0.14 * k),
                                   f_.top() + f_.height() * (0.2 + 0.1 * (k % 3)), tw_,
                                   (0.35, 0.0, -1.0)[k % 3]))
        img = _QI26(aw, ah, _QI26.Format_RGB32)
        img.fill(0xFF0A0A0A)
        pn = _QP26(img)
        ha.paint(pn, mk(tw_))
        pn.end()
        a_ = _np26.frombuffer(img.constBits(), _np26.uint8).reshape(ah, img.bytesPerLine())
        a_ = a_[:, :aw * 4].reshape(ah, aw, 4)
        red_ = int(_cv26.subtract(a_[..., 2], _cv26.max(a_[..., 1], a_[..., 0])).max())
        lit_ = float((_cv26.cvtColor(_np26.ascontiguousarray(a_[..., :3]), _cv26.COLOR_BGR2GRAY) > 60).mean())
        print(f"         versus WORST CASE at {aw}x{ah}: lit {lit_ * 100:.3f}%  red excess {red_}")
        check(f"versus WORST CASE at {aw}x{ah}: red excess 0 and inside the 2% ceiling",
              red_ == 0 and lit_ <= LIT26, f"red {red_}, lit {lit_ * 100:.3f}%")
    print(f"         versus worst {lit_states['versus: WORST CASE'] * 100:.3f}% against single-player "
          f"worst {lit_single_worst * 100:.3f}%; every versus state: "
          + ", ".join(f"{k.split(': ')[1]} {v * 100:.3f}%" for k, v in lit_states.items()))

    # ------------------------------------------------------------------
    # 31. a FINISHED half goes quiet (2026-10-19, found from the firing line)
    # ------------------------------------------------------------------
    print()
    print("31. versus: a half that has used its 20 shots is quiet, says why in the middle, and a partner never waits on it")
    import numpy as _np31
    from lsa.modes.balls import BouncingBalls as _B31
    from lsa.modes.reaction import ReactionDrill as _R31
    N31 = _DEFAULT_MAG

    def _balls31(h_):
        return ([(round(b.x, 3), round(b.y, 3)) for b in h_.mode.balls],
                [(round(b.x, 3), round(b.y, 3)) for b in h_.mode2.balls])

    def _ink31(a_, rect_):
        """Pixels in `rect_` that are not the target background (0x0A)."""
        sub = a_[rect_.top():rect_.bottom() + 1, rect_.left():rect_.right() + 1, :3]
        return (sub != 10).any(axis=2)

    h31, c31 = _vh25(_B31, now=1.0)
    for i31 in range(5):
        h31.advance(context(now=1.0 + i31 / 60.0))
    lay31 = h31._layout(c31)
    _fire(h31, c31, 0, N31, on_target=False)
    check("(setup) the left half has registered its 20th shot and the right has not",
          h31._side_done(0) and not h31._side_done(1) and not h31.round_over)

    # -- not painted: assert on RENDERED PIXELS, not on a flag ------------------
    def _host31(p_, cx_):
        h31.paint(p_, cx_)

    def _chrome31(p_, cx_):
        h31._paint_chrome(p_, cx_, h31._layout(cx_))
        if h31._button_live:
            h31.menu_button.paint(p_, cx_)

    _r, _l, a_host = _fh_full(_host31, c31)
    _r, _l, a_chrome = _fh_full(_chrome31, c31)
    hw31 = lay31.left.width()
    check("the finished LEFT half paints nothing but the host's own furniture (pixel-identical to chrome alone)",
          bool((a_host[:, :hw31] == a_chrome[:, :hw31]).all()))
    check("...while the RIGHT half, still shooting, paints its drill (it differs from chrome alone)",
          not bool((a_host[:, lay31.right.left():] == a_chrome[:, lay31.right.left():]).all()))

    # -- not ticked ---------------------------------------------------------------
    before31 = _balls31(h31)
    for i31 in range(40):
        h31.advance(context(now=2.0 + i31 / 60.0))
    after31 = _balls31(h31)
    check("the finished half is not ticked: its balls do not move",
          before31[0] == after31[0])
    check("...while the other half's balls still move",
          before31[1] != after31[1])
    idle31 = [h31.advance(context(now=4.0 + i31 / 60.0)) for i31 in range(5)]
    check("...and a moving right half still asks for repaints (animation untouched)", any(idle31))

    # -- no solid shapes ----------------------------------------------------------
    h31.mode.occluders = lambda sctx: ["L"]
    h31.mode2.occluders = lambda sctx: ["R"]
    check("a finished half contributes no occluders; the live half still does",
          h31.occluders(c31) == ["R"])
    del h31.mode.occluders, h31.mode2.occluders

    # -- says why, in the MIDDLE of the half --------------------------------------
    ink_l = _ink31(a_chrome, lay31.left)
    row0 = lay31.scoreline.bottom() + 1 - lay31.left.top()
    ys, xs = _np31.nonzero(ink_l[row0:, :])
    ys = ys + row0
    box31 = h31.done_box(0, c31)
    check("the finished half carries the words: ink in the middle band",
          len(xs) > 200, f"{len(xs)} px")
    if len(xs):
        cy31, cx31 = float(ys.mean()) + lay31.left.top(), float(xs.mean()) + lay31.left.left()
        check("...centred in the half horizontally (within 10% of its width)",
              abs(cx31 - (lay31.left.left() + hw31 / 2)) < 0.10 * hw31, f"cx {cx31:.0f}")
        check("...and inside the middle band between the scoreline and the bottom strip",
              box31.top() <= cy31 <= box31.bottom(), f"cy {cy31:.0f} in {box31.top()}..{box31.bottom()}")
    check("...and the words are the host's own: 'LEFT: done, N of N'",
          h31.note_text(0) == f"LEFT: done, {N31} of {N31}")
    _r, _l, a_bare = _fh_full(lambda p_, cx_: h31._paint_chrome(p_, cx_, h31._layout(cx_)), c31)
    check("...no longer in the top note row (it moved to where the targets were)",
          int(_ink31(a_bare, lay31.left_note).sum()) == 0)
    check("...while the right half still shows its round note up there",
          int(_ink31(a_bare, lay31.right_note).sum()) > 0)

    # -- a shot on the quiet half: seen, heard, drawn, counted, scored nowhere ----
    mshots31, pts31 = h31.mode.shots, h31.tallies[0].points
    rshots31 = h31.mode2.shots
    aim31 = _aim(h31, c31, 0)
    out31 = h31.shot(modes.ShotEvent(aim31[0], aim31[1], 9.0, 99), c31)
    check("a shot on the finished half is a DRY pull: the pop instead of the gunshot, and its hole is drawn "
          "(his rule of 2026-10-20, which replaced silence)",
          out31.draw_hole and out31.play_sound and out31.dry)
    check("...is counted in after_done, so the silence stays a number",
          h31.tallies[0].after_done == 1)
    check("...is unscored and reaches neither the drill, the score nor the other half",
          out31.unscored and not out31.hit and h31.mode.shots == mshots31
          and h31.tallies[0].points == pts31 and h31.mode2.shots == rshots31)

    # -- the round still closes, and N still swaps ---------------------------------
    _fire(h31, c31, 1, N31, on_target=False, t0=12.0)
    check("the round still closes when the other side reaches twenty",
          h31.round_over and h31.verdict is not None)
    h31.restart(c31)
    check("N after the round starts the next one with the sides swapped, both halves live again",
          h31.round_no == 2 and h31.seats == (1, 0) and not h31._side_done(0)
          and not h31._side_done(1))
    b31a = _balls31(h31)
    for i31 in range(30):
        h31.advance(context(now=30.0 + i31 / 60.0))
    b31b = _balls31(h31)
    check("...and both halves are ticked again (a retired half is revived by the new round)",
          b31a[0] != b31b[0] and b31a[1] != b31b[1])

    # -- a partner never waits on a quiet half (the shared reveal) -----------------
    for first31 in (0, 1):
        hr31 = _Host()
        hr31.set_mode(_R31(), context(now=1.0), players=2)
        _fire(hr31, context(now=1.0), first31, N31, on_target=False)
        live31 = None
        t31 = 1.0
        while t31 < 12.0 and live31 is None:
            t31 += 1 / 60.0
            hr31.advance(context(now=t31))
            other31 = hr31.mode2 if first31 == 0 else hr31.mode
            if other31.state == "live":
                live31 = t31
        check(f"Reaction: with side {first31} finished, the other side's target STILL appears (no hang on a retired seat)",
              live31 is not None, f"live at {live31}")

    # -- single player is untouched -----------------------------------------------
    hs31 = _Host()
    hs31.set_mode(_B31(), context(now=1.0), players=1)
    for i31 in range(N31 - 1):
        hs31.shot(modes.ShotEvent(60.0, 1500.0, 1.0 + i31 * 0.01, i31 + 1), context(now=1.0))
    p0_31 = [(b.x, b.y) for b in hs31.mode.balls]
    for i31 in range(30):
        hs31.advance(context(now=2.0 + i31 / 60.0))
    check("single player: one round still in the magazine, the drill still ticks, nothing is after_done",
          [(b.x, b.y) for b in hs31.mode.balls] != p0_31 and hs31.tallies[0].after_done == 0
          and hs31.remaining(0) == 1 and not hs31.round_over)
    _r, _l, a_s31 = _fh_full(hs31.paint, context(now=3.0))
    check("single player: the drill still paints across the whole screen",
          bool((a_s31[..., :3] != 10).any()))
    for i31 in range(6):
        hs31.shot(modes.ShotEvent(60.0, 1500.0, 3.0 + i31 * 0.01, 500 + i31), context(now=3.0))
    p1_31 = [(b.x, b.y) for b in hs31.mode.balls]
    for i31 in range(30):
        hs31.advance(context(now=4.0 + i31 / 60.0))
    check("single player: the magazine is empty after N, the round is over and the drill no longer ticks; "
          "the five dry pulls are counted",
          hs31.round_over and [(b.x, b.y) for b in hs31.mode.balls] == p1_31
          and hs31.tallies[0].after_done == 5)

    # -- the colour and lit-area budget, MEASURED with the message up --------------
    worst31, worst31_lbl = 0.0, ""
    for cls31 in (_Static, EscalatingBall, _B31, ClearTheBoard, _R31):
        for lbl31, nleft31, nright31 in (("one side done", N31, 3), ("both sides done", N31, N31)):
            hq, cq = _vh25(cls31, now=1.0)
            _fire(hq, cq, 0, nleft31, on_target=False)
            if nright31:
                _fire(hq, cq, 1, nright31, on_target=False, t0=3.0)
            hq.advance(context(now=5.0))
            r31, l31, _a = _fh_full(hq.paint, context(now=5.0))
            if l31 > worst31:
                worst31, worst31_lbl = l31, f"{cls31.title} / {lbl31}"
            check(f"{cls31.title} [{lbl31}] with the DONE words up: red excess exactly 0, inside the 2% ceiling",
                  r31 == 0 and l31 <= 0.02, f"red {r31}, lit {l31 * 100:.3f}%")
            print(f"         {cls31.title} {lbl31}: lit {l31 * 100:.3f}%  red {r31}")
    print(f"         WORST with a finished half: {worst31 * 100:.3f}% ({worst31_lbl})")

    # ------------------------------------------------------------------
    # 32. THE INVARIANT: a side with shots left always has something to shoot
    #     at; a side that has used its allotment is quiet and says so.
    # ------------------------------------------------------------------
    print()
    print("32. versus: whole rounds through the host, uneven, every drill, asserted on rendered pixels per half")
    from PySide6.QtGui import QImage as _QI32, QPainter as _QP32
    from lsa.modes.image_target import ImageTarget as _I32
    from lsa.modes.advancing import AdvancingTarget as _A32
    from lsa.modes.animated import AnimatedFigure as _AN32
    from lsa.modes.popup import PopUpFigure as _P32
    N32 = _DEFAULT_MAG
    W32, H32 = 1920, 1080

    def _c32(now, playable=None):
        return RenderContext(now=now, dt=1 / 60.0, rect=QRect(0, 0, W32, H32),
                             stats=SessionStats(), playable=playable)

    def _render32(fn, c_):
        img = _QI32(W32, H32, _QI32.Format_RGB32)
        img.fill(0xFF0A0A0A)
        pn = _QP32(img)
        fn(pn, c_)
        pn.end()
        a_ = _np31.frombuffer(img.constBits(), _np31.uint8).reshape(H32, img.bytesPerLine())
        return a_[:, :W32 * 4].reshape(H32, W32, 4)[..., :3].copy()

    def _aim32(h_, side_, now_=0.0):
        """A point that would HIT this side's drill right now, or None."""
        m_ = h_.mode if side_ == 0 else h_.mode2
        c_ = _c32(now_)
        if hasattr(m_, "aim_point"):
            try:
                return m_.aim_point(h_._side_ctx(c_, side_))
            except Exception:
                return None
        ts_ = [t for t in (getattr(m_, "targets", None) or []) if getattr(t, "alive", True)]
        if ts_:
            return (ts_[0].x, ts_[0].y)
        bs_ = getattr(m_, "balls", None)
        if bs_:
            return (bs_[0].x, bs_[0].y)
        if getattr(m_, "state", "") == "live":
            return (m_.x, m_.y)
        return None

    def _play32(cls_, periods, clear_on_last=False, playable=None, rounds=2):
        """Play `rounds` complete rounds. Returns (violations, samples)."""
        bad_, samples_ = [], 0
        h_ = _Host()
        h_.set_mode(cls_(), _c32(1.0, playable), players=2)
        t_ = 1.0
        gap_max = 8.0 if cls_ in (_R31, _P32) else 3.0
        for rnd_ in range(rounds):
            lastseen_ = [t_, t_]
            nxt_ = [t_ + 0.3, t_ + 0.3]
            fired_ = [0, 0]
            alive0_ = [None, None]
            next_sample_ = t_
            t_end_ = t_ + 120.0
            while not h_.round_over and t_ < t_end_:
                t_ += 1 / 60.0
                c_ = _c32(t_, playable)
                h_.advance(c_)
                for s_ in (0, 1):
                    if t_ < nxt_[s_]:
                        continue
                    nxt_[s_] = t_ + periods[(s_ + rnd_) % 2]
                    pt_ = _aim32(h_, s_, t_)
                    if clear_on_last:
                        m_ = h_.mode if s_ == 0 else h_.mode2
                        if alive0_[s_] is None:
                            alive0_[s_] = sum(1 for tt in m_.targets if tt.alive)
                        if fired_[s_] < N32 - alive0_[s_]:
                            pt_ = None
                    if pt_ is None:
                        pt_ = (60.0 if s_ == 0 else W32 - 60.0, 700.0)
                    fired_[s_] += 1
                    h_.shot(modes.ShotEvent(pt_[0], pt_[1], t_, fired_[s_]), c_)
                if t_ >= next_sample_:
                    next_sample_ = t_ + 0.5
                    samples_ += 1
                    lay_ = h_._layout(c_)
                    a_host_ = _render32(h_.paint, c_)
                    a_chr_ = _render32(lambda p_, cx_: (h_._paint_chrome(p_, cx_, h_._layout(cx_)),
                                                         h_.menu_button.paint(p_, cx_) if h_._button_live else None), c_)
                    row0_ = lay_.scoreline.bottom() + 1
                    for s_ in (0, 1):
                        half_ = lay_.side(s_)
                        sl_ = (slice(row0_, H32), slice(half_.left(), half_.right() + 1))
                        diff_ = int((a_host_[sl_] != a_chr_[sl_]).any(axis=2).sum())
                        if h_._side_done(s_):
                            if diff_:
                                bad_.append(f"r{rnd_ + 1} t={t_ - 1.0:.1f}s side {s_} is DONE yet paints {diff_} px of drill")
                            box_ = h_.done_box(s_, c_)
                            msg_ink_ = int((a_host_[box_.top():box_.bottom() + 1,
                                                    box_.left():box_.right() + 1] != 10).any(axis=2).sum())
                            if msg_ink_ < 50:
                                bad_.append(f"r{rnd_ + 1} t={t_ - 1.0:.1f}s side {s_} is DONE and says nothing in the middle")
                        else:
                            if diff_ > 20:
                                lastseen_[s_] = t_
                            elif t_ - lastseen_[s_] > gap_max:
                                bad_.append(f"r{rnd_ + 1} t={t_ - 1.0:.1f}s side {s_} has shots left "
                                            f"({h_.tallies[s_].shots}/{N32}) and nothing to shoot at for "
                                            f"{t_ - lastseen_[s_]:.1f}s")
                                lastseen_[s_] = t_
                # a finished half still gets shot at (he keeps pulling the trigger)
                for s_ in (0, 1):
                    if h_._side_done(s_) and not h_.round_over and int(t_ * 60) % 30 == 0:
                        pt_ = _aim32(h_, s_, t_) or ((300.0 if s_ == 0 else W32 - 300.0), 500.0)
                        before_ = h_.tallies[s_].after_done
                        out_ = h_.shot(modes.ShotEvent(pt_[0], pt_[1], t_, 900), c_)
                        if not (out_.unscored and h_.tallies[s_].after_done == before_ + 1):
                            bad_.append(f"r{rnd_ + 1} a shot on the finished half {s_} was not counted as after_done")
            if not h_.round_over:
                bad_.append(f"round {rnd_ + 1} never closed in 120 s of play")
                break
            c_ = _c32(t_, playable)
            a_ = _render32(h_.paint, c_)
            a_chr_ = _render32(lambda p_, cx_: (h_._paint_chrome(p_, cx_, h_._layout(cx_)),
                                                 h_.menu_button.paint(p_, cx_) if h_._button_live else None), c_)
            if not (a_ == a_chr_).all():
                bad_.append(f"round {rnd_ + 1} is OVER yet the screen shows more than the verdict")
            h_.restart(c_)
            t_ += 0.5
        return bad_, samples_

    pl32 = QRect(300, 150, 1500, 800)          # a realistic camera-coverage rectangle
    for cls32 in (_Static, EscalatingBall, _B31, ClearTheBoard, _R31,
                  _I32, _A32, _AN32, _P32):
        for lbl32, kw32 in (("left fast, right slow", dict(periods=(0.25, 0.6))),
                            ("with a camera-coverage rectangle", dict(periods=(0.25, 0.6), playable=pl32)),
                            ) + ((("a side CLEARS its board on its 20th shot", dict(periods=(0.2, 0.5), clear_on_last=True)),)
                                 if cls32 is ClearTheBoard else ()):
            bad32, n32 = _play32(cls32, **kw32)
            check(f"{cls32.title} [{lbl32}]: 2 whole rounds, {n32} rendered samples, no half ever idle with shots left and no done half ever draws",
                  not bad32, "; ".join(bad32[:3]))

    # ------------------------------------------------------------------
    # 33. a versus round leaves a TRAIL (one log line per lifecycle event)
    # ------------------------------------------------------------------
    print()
    print("33. versus: the round's lifecycle is recorded for the log (the host is pure; the app drains it)")
    h33, c33 = _vh25(_Static, now=1.0)
    ev33 = h33.take_events()
    check("a round starting says which round and which player is on which side, and the magazine",
          len(ev33) == 2 and "round 1 starts" in ev33[0] and "LEFT is player 1" in ev33[0]
          and "RIGHT is player 2" in ev33[0] and f"magazine {N31} each" in ev33[1], str(ev33))
    check("...and draining empties it: nothing is reported twice", h33.take_events() == [])
    _fire(h33, c33, 0, N31, on_target=False)
    ev33 = h33.take_events()
    check("a side using its allotment is recorded once, with its points and shots",
          sum("LEFT has used its allotment" in e and f"{N31} shots" in e and f"magazine {N31}" in e
              for e in ev33) == 1, str(ev33))
    _fire(h33, c33, 0, 3, on_target=False)
    ev33 = h33.take_events()
    check("each shot on the finished half is a line with its running after_done count",
          [("after_done " + str(i)) in ev33[i - 1] for i in (1, 2, 3)] == [True] * 3 and len(ev33) == 3, str(ev33))
    _fire(h33, c33, 1, N31, on_target=False)
    ev33 = h33.take_events()
    check("the round closing records both tallies, the match totals and the verdict",
          any("round 1 closed" in e and "LEFT" in e and "RIGHT" in e and "match totals" in e
              and "3 after done" in e for e in ev33), str(ev33))
    h33.restart(c33)
    ev33 = h33.take_events()
    check("the swap is in the log (it is silent on screen, never silent on disk)",
          any("sides swapped for round 2" in e and "player 1 is now on RIGHT" in e for e in ev33), str(ev33))
    check("...and round 2 records its own start with the seats swapped",
          any("round 2 starts" in e and "LEFT is player 2" in e for e in ev33), str(ev33))
    hs33 = _Host()
    hs33.set_mode(_Static(), context(now=1.0), players=1)
    hs33.shot(modes.ShotEvent(60.0, 1500.0, 1.0, 1), context(now=1.0))
    check("single player records its round's start and magazine, and nothing per shot",
          hs33.take_events() == [f"round 1 starts (Static Targets): magazine {N31}"]
          and hs33.take_events() == [])
    for i33 in range(500):
        h33.shot(modes.ShotEvent(60.0, 1500.0, 50.0 + i33 * 0.01, i33 + 1), c33)
    check("the trail is bounded when nobody drains it", len(h33._events) <= 200)

    # ------------------------------------------------------------------
    # 34. the VERDICT screen: holes cleared, a small microphone line, no swap
    #     wording, and a ROUND box that starts the next round
    # ------------------------------------------------------------------
    print()
    print("34. versus verdict screen: one-shot hole clear, microphone line at the bottom, ROUND box")
    from lsa.display.layers import BulletHoles as _BH34

    def _closed34(cls_=_Static, now=1.0, mic=False, heard=0):
        h_, c_ = _vh25(cls_, now=now)
        h_.mic_ready = mic
        h_.note_heard_unseen(heard)
        _fire(h_, c_, 0, N31)
        _fire(h_, c_, 1, N31, t0=3.0, on_target=False)
        return h_, c_

    hv34, cv34 = _closed34(mic=True, heard=3)
    check("a round closing raises a ONE-SHOT 'ended' fact for the app (it clears the holes)",
          hv34.take_round_ended() is True and hv34.take_round_ended() is False)
    hnone34, _c = _vh25(_Static)
    check("...and it is not raised before a round has closed", hnone34.take_round_ended() is False)
    check("the host never reaches into the scene: it has no reference to a holes layer",
          not any(hasattr(hv34, n_) for n_ in ("holes", "layers", "scene")))

    # -- no swap wording on screen; the swap itself still happens ------------------
    check("the verdict block says nothing of a swap or of pressing N",
          not any("swap" in l_.lower() or "press n" in l_.lower() for l_ in hv34.verdict_lines()))

    # -- painted text, captured from the real painter ------------------------------
    rec34: list = []
    _orig34 = _Host.__dict__["_draw_lines"]

    def _spy34(p_, box_, lines_, start_, bold=False, floor=9):
        sz_ = _orig34.__func__(p_, box_, lines_, start_, bold, floor)
        rec34.append((list(lines_), sz_, box_))
        return sz_

    _Host._draw_lines = staticmethod(_spy34)
    try:
        _fh_full(hv34.paint, context(now=100.0))
    finally:
        _Host._draw_lines = _orig34
    texts34 = [t_ for ls_, _s, _b in rec34 for t_ in ls_]
    check("the box reads NEXT ROUND (renamed 2026-10-20: the round number is in the heading), "
          "and the heading names the round just played, derived",
          "NEXT ROUND" in texts34 and "ROUND 1 COMPLETED" in texts34 and not any("START" in t_ for t_ in texts34),
          str(texts34[-5:]))
    # The how-to ("shoot it, click it or press N") was removed at his request: START is the instruction.
    check("...and carries no how-to line (removed at his request; the box is still shootable, clickable and N-able)",
          not any("shoot it" in t_ or "press N" in t_ for t_ in texts34))
    # THE BOTTOM BOX IS NO LONGER DRAWN (his ruling, 2026-10-22: "no need for this
    # text"). The dry-pull and microphone words are still built (`dry_screen_text`,
    # `mic_screen_text`) and still go to the log; the PAINTER no longer asks for them.
    check("the microphone count is NOT painted on the round screen, with a calibrated microphone and "
          "three heard pulls (nothing below the ROUND box)",
          not any(t_.startswith("Microphone:") or t_.startswith("Dry pulls:")
                  for ls_, _s, _b in rec34 for t_ in ls_)
          and "3 trigger pulls" in hv34.mic_screen_text())
    rbox34 = hv34.round_box(context(now=100.0))
    check("...and no box at all is drawn below the ROUND box",
          all(b_.top() < rbox34.bottom() for ls_, _s, b_ in rec34 if ls_ != ["NEXT ROUND"]),
          str([(ls_[:1], b_.top()) for ls_, _s, b_ in rec34 if b_.top() >= rbox34.bottom()
               and ls_ != ["NEXT ROUND"]]))
    hq34, cq34 = _closed34(mic=False, heard=3)
    rec34.clear()
    _Host._draw_lines = staticmethod(_spy34)
    try:
        _fh_full(hq34.paint, context(now=100.0))
    finally:
        _Host._draw_lines = _orig34
    check("an uncalibrated microphone puts nothing about the microphone or dry pulls on screen either "
          "(the words are still built, and still say it is a lower bound, for the log)",
          not any(t_.startswith("Microphone:") or t_.startswith("Dry pulls:")
                  for ls_, _s, _b in rec34 for t_ in ls_)
          and "or more" in hq34.dry_screen_text())
    ev_q34 = [e_ for e_ in hq34.take_events() if e_.startswith("microphone")]
    ev_v34 = [e_ for e_ in hv34.take_events() if e_.startswith("microphone")]
    check("...but BOTH cases write the microphone line to the log (an unreported check is a failure)",
          len(ev_q34) == 1 and "not calibrated" in ev_q34[0] and len(ev_v34) == 1 and "3 trigger pulls" in ev_v34[0],
          f"{ev_q34} / {ev_v34}")
    hz34, _cz = _closed34(mic=True, heard=0)
    check("a calibrated microphone with nothing to report is also silent on screen and present in the log",
          hz34.mic_screen_text() == "" and any(e_.startswith("microphone: no trigger pulls") for e_ in hz34.take_events()))

    # -- the box: armed 0.8 s after it is first PAINTED ---------------------------
    hb34, cb34 = _closed34()
    rb34 = hb34.round_box(cb34)
    bx34, by34 = rb34.center().x(), rb34.center().y()
    o34 = hb34.shot(modes.ShotEvent(bx34, by34, 50.0, 1), cb34)
    check("a shot at the box BEFORE it has been painted does nothing (a phantom in the same instant as the verdict)",
          not hb34.take_next_round_request() and o34.unscored and not o34.hit)
    _fh_full(hb34.paint, context(now=100.0))
    o34 = hb34.shot(modes.ShotEvent(bx34, by34, 100.5, 2), cb34)
    check("a hit within 0.8 s of its first paint does nothing",
          hb34.take_next_round_request() is False and hb34.round_over)
    o34 = hb34.shot(modes.ShotEvent(bx34, by34, 100.9, 3), cb34)
    check("after that ONE hit shuts it: no second shot needed, counted as neither hit nor miss, and it "
          "BANGS and leaves its hole (the round-over screen is shot like the menu: a control must answer)",
          o34.draw_hole is True and o34.play_sound is True and not o34.hit and o34.unscored)
    check("...and the request is a one-shot for the app to drain",
          hb34.take_next_round_request() is True and hb34.take_next_round_request() is False)
    ht34, ct34 = _closed34()
    _fh_full(ht34.paint, context(now=100.0))
    pre34 = (ht34.tallies[0].after_done, ht34.tallies[1].after_done)
    ht34.shot(modes.ShotEvent(300.0, 1500.0, 101.0, 4), ct34)
    check("a shot ELSEWHERE on the verdict is still seen and counted, and requests nothing",
          ht34.take_next_round_request() is False
          and ht34.tallies[0].after_done == pre34[0] + 1)
    check("a click on the box shuts it at once, by the same request",
          ht34.on_click(ht34.round_box(ct34).center().x(), ht34.round_box(ct34).center().y(), ct34) is True
          and ht34.take_next_round_request() is True)
    check("a click elsewhere on the verdict requests nothing",
          ht34.on_click(300.0, 1500.0, ct34) is False and ht34.take_next_round_request() is False)
    check("the box does nothing while a round is in play",
          not _vh25()[0].round_box(ct34).isNull() and
          (lambda hh_, cc_: (hh_.on_click(hh_.round_box(cc_).center().x(), hh_.round_box(cc_).center().y(), cc_),
                             hh_.take_next_round_request())[1])(*_vh25()) is False)
    # the numbering follows the round: ROUND 3 after round 2
    hn34, cn34 = _closed34()
    hn34.restart(cn34)
    _fire(hn34, cn34, 0, N31)
    _fire(hn34, cn34, 1, N31, t0=3.0, on_target=False)
    rec34.clear()
    _Host._draw_lines = staticmethod(_spy34)
    try:
        _fh_full(hn34.paint, context(now=100.0))
    finally:
        _Host._draw_lines = _orig34
    check("...the heading reads ROUND 2 COMPLETED after round 2 (derived from the round number, never typed)",
          any("ROUND 2 COMPLETED" in t_ for ls_, _s, _b in rec34 for t_ in ls_)
          and any("NEXT ROUND" in t_ for ls_, _s, _b in rec34 for t_ in ls_))
    check("the seats DID swap and the totals follow the PLAYER, whatever the screen says",
          hn34.seats == (1, 0) and hn34.round_no == 2)

    # -- nothing times out ---------------------------------------------------------
    hk34, ck34 = _closed34()
    for i34 in range(0, 3600, 30):
        hk34.advance(context(now=100.0 + i34))
    _r, _l, a_k34 = _fh_full(hk34.paint, context(now=100.0 + 3600))
    check("the verdict stays up for an hour of virtual time: it never times out",
          hk34.round_over and bool((a_k34[..., :3] != 10).any()))

    # -- colour and lit-area budget, MEASURED --------------------------------------
    hm34, cm34 = _closed34(mic=True, heard=3)
    holes34 = _BH34()
    for k34 in range(14):
        holes34.add(500.0 + k34 * 180.0, 800.0 + (k34 % 4) * 150.0, k34 + 1, 0)

    def _noboxpaint(p_, cx_):
        saved = _Host._paint_round_box
        _Host._paint_round_box = lambda self, p2, c2, rb2: None
        try:
            hm34.paint(p_, cx_)
        finally:
            _Host._paint_round_box = saved

    def _with_holes(p_, cx_):
        holes34.paint(p_, cx_)
        hm34.paint(p_, cx_)

    cm34 = context(now=100.0)
    r_a, l_a, _i = _fh_full(lambda p_, cx_: (holes34.paint(p_, cx_), _noboxpaint(p_, cx_)), cm34)
    r_b, l_b, _i = _fh_full(hm34.paint, cm34)
    r_c, l_c, _i = _fh_full(_with_holes, cm34)
    print(f"         verdict screen lit: BEFORE (holes under the text, no box) {l_a * 100:.3f}%   "
          f"AFTER (holes cleared, box + mic line) {l_b * 100:.3f}%   box AND holes {l_c * 100:.3f}%   "
          f"red excess {r_a}/{r_b}/{r_c}")
    check("the verdict screen with the box and the microphone line: red excess exactly 0, inside the 2% ceiling",
          r_b == 0 and l_b <= 0.02, f"red {r_b}, lit {l_b * 100:.3f}%")
    check("...and the holes-under-text state is no better (clearing them cannot raise the lit area)",
          l_b <= l_a + 0.0005 and r_a == 0 and r_c == 0 and l_c <= 0.02, f"{l_a}, {l_b}, {l_c}")

    # ------------------------------------------------------------------
    # 35. drills 6-9 in two players: the same figure/clip on both halves, one
    #     shared reveal for Pop-Up, and the budget MEASURED
    # ------------------------------------------------------------------
    print()
    print("35. versus: Static / Moving / Animated / Pop-Up Target on both halves")
    from lsa.modes.figure import figure_names as _names35
    from lsa.audio.mixer import FIGURE_APPEAR_SOUND as _FA35, FALL_SOUND as _FS35
    import statistics as _st35
    NAMES35 = _names35()
    ctx35 = context(now=1.0)

    def _host35(cls_, forced=None, now=1.0):
        h_ = _Host()
        h_.set_mode(cls_(), context(now=now), players=2)
        if forced:
            h_.mode._forced = h_.mode2._forced = forced
            h_.restart(context(now=now))
        return h_

    # -- A. Static Target: lit area of one figure, two at full size, two as built ----
    lit1, lit2full, lit2, ratio35 = {}, {}, {}, {}
    red35 = 0
    for nm in NAMES35:
        s_ = _I32()
        s_.start(context(now=1.0), figure=nm)
        r1, l1, _a = _fh_full(s_.paint, context(now=1.0))
        h_single = s_._height

        def _two_full(p_, cx_, s=s_):
            lay_ = _split.layout(cx_.rect)
            for which_ in (0, 1):
                c0 = lay_.side(which_).center()
                s._fig.sprite.paint(p_, (c0.x(), s._centre[1]), s._height)
        r2f, l2f, _a = _fh_full(_two_full, context(now=1.0))
        hv = _host35(_I32, nm)
        r2, l2, _a = _fh_full(hv.paint, context(now=1.0))
        red35 = max(red35, r1, r2f, r2)
        lit1[nm], lit2full[nm], lit2[nm] = l1, l2f, l2
        ratio35[nm] = hv.mode._height / h_single
    w1 = max(lit1, key=lit1.get)
    print(f"         STATIC TARGET lit above luminance 60 (worst figure): one figure today {max(lit1.values()) * 100:.3f}% "
          f"({w1}), two at FULL size {max(lit2full.values()) * 100:.3f}%, two as built "
          f"{max(lit2.values()) * 100:.3f}%; shrink factor {min(ratio35.values()):.3f}..{max(ratio35.values()):.3f}; "
          f"red excess {red35}")
    check("Static Target in two players: red excess exactly 0 in every state, for every figure", red35 == 0, str(red35))
    check("...two figures at full size WOULD breach the 2% ceiling (the shrink is needed, not decorative)",
          max(lit2full.values()) > 0.02, f"{max(lit2full.values()) * 100:.3f}%")
    check("...and two as built stay inside it, for every figure", max(lit2.values()) <= 0.02,
          f"{max(lit2.values()) * 100:.3f}%")
    check("...because each figure came out at about 0.70 of its single-player height (his ruling: shrink, "
          "keep the brightness)", 0.62 <= min(ratio35.values()) and max(ratio35.values()) <= 0.78,
          f"{min(ratio35.values()):.3f}..{max(ratio35.values()):.3f}")
    from lsa.artwork import ceiling as _ceil35
    check("...brightness untouched: the same gain read from the artwork ceiling, none applied for the split",
          all(abs(Figure_.gain - _ceil35.gain_for_path(__import__('lsa.artwork.catalog', fromlist=['x']).path_for(Figure_.name), None)) < 1e-9
              for Figure_ in [__import__('lsa.modes.figure', fromlist=['x']).Figure(NAMES35[0], 1920, 2160, 0.015)]))

    # scoring still follows the artwork at the smaller size (ring polygons are normalised)
    from lsa.modes.figure import Figure as _Fig35
    bad35 = []
    for nm in NAMES35:
        big = _Fig35(nm, 3840, 2160, 0.015)
        small = _Fig35(nm, 1920, 2160, 0.015)
        zb = big.innermost((1000.0, 1000.0), 800.0)
        zs = small.innermost((1000.0, 1000.0), 566.0)
        pb, _z = big.score(zb[1], zb[2], (1000.0, 1000.0), 800.0)
        ps, _z = small.score(zs[1], zs[2], (1000.0, 1000.0), 566.0)
        if pb != ps or ps <= 0:
            bad35.append((nm, pb, ps))
    check("the innermost ring still scores the same at the half-width size, for every figure (a figure that "
          "cannot be judged is never drawn, in versus too)", not bad35, str(bad35[:3]))

    # the same figure on both halves, mirrored position, orientation kept
    hs35 = _host35(_I32)
    L35, R35 = hs35.mode, hs35.mode2
    seq35 = ([], [])

    def _watch35(h_):
        for i_, m_ in enumerate((h_.mode, h_.mode2)):
            if m_._fig is not None and (not seq35[i_] or seq35[i_][-1][0] is not m_._fig):
                seq35[i_].append((m_._fig, m_._fig.name, m_._centre))
    _watch35(hs35)
    t35 = 1.0
    for rep35 in range(3):                      # three drops on each half
        for side35 in (0, 1):
            m_ = hs35.mode if side35 == 0 else hs35.mode2
            for guard in range(40):
                if m_._fig is None:
                    break
                ap_ = m_.aim_point(None)
                t35 += 0.05
                hs35.shot(modes.ShotEvent(ap_[0], ap_[1], t35, 1), context(now=t35))
                if m_._marks.total == 0 and m_._fig is None:
                    break
            # let the fall (if any) finish
            for _ in range(80):
                t35 += 1 / 60.0
                hs35.advance(context(now=t35))
                _watch35(hs35)
    n35 = min(len(seq35[0]), len(seq35[1]))
    pl35 = [tuple(p_.names) for p_ in L35._link.plans]    # one (seat0, seat1) pair per shared cycle
    n_a35 = [x[1] for x in seq35[0]]
    n_b35 = [x[1] for x in seq35[1]]

    def _in_order35(seen_, col_):
        """`seen_` is a subsequence of the plans' column `col_`, in order (a figure
        with no fall pose is replaced at the shot, so a half may legitimately skip
        ahead; it may never go back, repeat, or show a figure no plan gave it)."""
        k_ = 0
        for nm_ in seen_:
            while k_ < len(pl35) and pl35[k_][col_] != nm_:
                k_ += 1
            if k_ >= len(pl35):
                return False
            k_ += 1
        return True
    check("Static Target: every shared cycle gives the halves DIFFERENT figures, and neither seat repeats its own "
          "figure from one cycle to the next",
          len(pl35) >= 3 and all(a_ != b_ for a_, b_ in pl35)
          and all(pl35[i_][0] != pl35[i_ + 1][0] and pl35[i_][1] != pl35[i_ + 1][1] for i_ in range(len(pl35) - 1)),
          str(pl35))
    check("...and what each half actually showed is its own seat's column of those plans, in order, starting at cycle 0 "
          "(three drops each)",
          n35 >= 3 and n_a35[0] == pl35[0][0] and n_b35[0] == pl35[0][1]
          and _in_order35(n_a35, 0) and _in_order35(n_b35, 1),
          f"{n_a35} / {n_b35} / {pl35}")
    lay35 = L35._link  # noqa
    f0, f1 = seq35[0][0], seq35[1][0]
    lay_s35 = _split.layout(ctx35.rect)
    rel0 = (f0[2][0] - lay_s35.left.left()) / lay_s35.left.width()
    rel1 = (f1[2][0] - lay_s35.right.left()) / lay_s35.right.width()
    # the plan is shared as fractions of each figure's own travel box, so with two
    # different figures the absolute centres differ by a fraction of the size gap
    check("...in MIRRORED position (to within the size difference of the two figures) and the artwork is not flipped",
          abs(rel0 + rel1 - 1.0) < 0.06
          and abs(f0[2][1] - f1[2][1]) < 0.06 * lay_s35.left.height()
          and f0[0].sprite is not f1[0].sprite,
          f"{rel0:.3f} + {rel1:.3f}")
    hv35 = _host35(_I32, NAMES35[0])
    _r, _l, a35 = _fh_full(hv35.paint, context(now=1.0))
    fl, fr = hv35.mode, hv35.mode2
    bl, br = fl._fig.box(fl._centre, fl._height), fr._fig.box(fr._centre, fr._height)
    cl = a35[int(bl[1]):int(bl[1] + bl[3]), int(bl[0]):int(bl[0] + bl[2]), 1].astype(float)
    cr = a35[int(br[1]):int(br[1] + br[3]), int(br[0]):int(br[0] + br[2]), 1].astype(float)
    hh_, ww_ = min(cl.shape[0], cr.shape[0]), min(cl.shape[1], cr.shape[1])
    cl, cr = cl[:hh_, :ww_], cr[:hh_, :ww_]
    corr = lambda a, b: float(_np31.corrcoef(a.ravel(), b.ravel())[0, 1])
    check("...pixel evidence: the right half's figure matches the left's UNFLIPPED better than flipped",
          corr(cl, cr) > corr(cl, cr[:, ::-1]) + 0.01 and corr(cl, cr) > 0.95,
          f"{corr(cl, cr):.3f} vs flipped {corr(cl, cr[:, ::-1]):.3f}")

    # -- B. Moving Target: closest approach is the binding case ---------------------
    lit_a1, lit_a2 = [], []
    red_a = 0
    for nm in NAMES35:
        sa = _A32()
        sa.start(context(now=1.0), figure=nm)
        ha = _host35(_A32, nm)
        # step both to the near hold
        t_ = 1.0
        for _ in range(60 * 14):
            t_ += 1 / 60.0
            sa.advance(context(now=t_))
            ha.advance(context(now=t_))
            if (all(a.phase == "live" and a.t_a <= t_ - a.t0 < a.t_a + a.t_h for a in sa._actors)
                    and all(a.phase == "live" and a.t_a <= t_ - a.t0 < a.t_a + a.t_h
                            for a in ha.mode._actors + ha.mode2._actors)):
                break
        r1, l1, _a = _fh_full(sa.paint, context(now=t_))
        r2, l2, _a = _fh_full(ha.paint, context(now=t_))
        red_a = max(red_a, r1, r2)
        lit_a1.append(l1)
        lit_a2.append(l2)
    print(f"         MOVING TARGET at closest approach (worst figure): one {max(lit_a1) * 100:.3f}%, "
          f"two as built {max(lit_a2) * 100:.3f}%; red excess {red_a}")
    check("Moving Target in two players: red excess 0 and inside the 2% ceiling at CLOSEST APPROACH, every figure",
          red_a == 0 and max(lit_a2) <= 0.02, f"{max(lit_a2) * 100:.3f}%")
    ha = _host35(_A32)
    ha.mode._forced = ha.mode2._forced = None
    seqa = ([], [])
    t_ = 1.0
    for _ in range(60 * 40):
        t_ += 1 / 60.0
        ha.advance(context(now=t_))
    nsh_a = min(len(ha.mode.shown), len(ha.mode2.shown))
    check("Moving Target: the k-th cycle is the SAME cycle on both halves in timing, hold and drift, with a "
          "DIFFERENT figure on each half and no repeat per half",
          nsh_a >= 2 and all(x_ != y_ for x_, y_ in zip(ha.mode.shown[:nsh_a], ha.mode2.shown[:nsh_a]))
          and all(ha.mode.shown[i_] != ha.mode.shown[i_ + 1] for i_ in range(nsh_a - 1))
          and all(ha.mode2.shown[i_] != ha.mode2.shown[i_ + 1] for i_ in range(nsh_a - 1))
          and ha.mode._link is ha.mode2._link
          and all((getattr(a, "plan", None) is not None) for a in ha.mode._actors + ha.mode2._actors),
          f"{ha.mode.shown} / {ha.mode2.shown}")
    # step to a moment when BOTH halves are walking the same shared cycle (asserted,
    # not skipped: a check guarded by "if it is live" does nothing when it is not)
    pa0, pa1 = ha.mode._actors[0], ha.mode2._actors[0]
    for _ in range(60 * 40):
        if pa0.phase == "live" and pa1.phase == "live" and pa0.plan is pa1.plan:
            break
        t_ += 1 / 60.0
        ha.advance(context(now=t_))
    fa0 = ha.mode.playfield(ha._side_ctx(context(now=t_), 0))
    fa1 = ha.mode2.playfield(ha._side_ctx(context(now=t_), 1))
    check("...with the position mirrored and the drift keeping its direction (premise: both halves live on one plan)",
          pa0.phase == "live" and pa1.phase == "live" and pa0.plan is pa1.plan
          and abs((pa0.hold_xy[0] - fa0.left()) / fa0.width()
                  + (pa1.hold_xy[0] - fa1.left()) / fa1.width() - 1.0) < 0.08
          and abs(pa0.hold_xy[1] - pa1.hold_xy[1]) < 0.08 * fa0.height() and pa0.drift == pa1.drift)

    # -- C. Animated Target: one clip, one pixmap cache --------------------------------
    hn = _host35(_AN32)
    LN, RN = hn.mode, hn.mode2
    check("Animated Target: both halves hold the SAME clip object and the SAME pixmap cache",
          LN.sequence() is RN.sequence() and LN._pix is RN._pix and LN._masks is RN._masks)
    t_ = 1.0
    lit_n1, lit_n2, red_n, boxes_ok = [], [], 0, True
    sn = _AN32()
    sn.start(context(now=1.0))
    lay_n = _split.layout(ctx35.rect)
    for i_ in range(0, 120, 3):
        t_ = 1.0 + (i_ + 0.5) / _AN32.FPS
        sn.advance(context(now=t_))
        hn.advance(context(now=t_))
        r1, l1, _a = _fh_full(sn.paint, context(now=t_))
        r2, l2, _a = _fh_full(hn.paint, context(now=t_))
        red_n = max(red_n, r1, r2)
        lit_n1.append(l1)
        lit_n2.append(l2)
        for which_, m_ in ((0, LN), (1, RN)):
            bx = m_.sequence().box(m_._frame, m_._origin, m_._scale)
            sd = lay_n.side(which_)
            boxes_ok &= (bx[0] >= sd.left() - 1 and bx[0] + bx[2] <= sd.right() + 2)
    print(f"         ANIMATED TARGET lit: one soldier max {max(lit_n1) * 100:.3f}%, two in step max "
          f"{max(lit_n2) * 100:.3f}% (no shrink applied); red excess {red_n}")
    check("Animated Target in two players: red excess 0 and inside the 2% ceiling, so NO shrink is needed",
          red_n == 0 and max(lit_n2) <= 0.02, f"{max(lit_n2) * 100:.3f}%")
    check("...and the soldier stays inside his own half on every sampled frame", boxes_ok)
    # in step: one pixmap per frame; out of step: two
    hn2 = _host35(_AN32)
    hn2.advance(context(now=1.0 + 0.5 / 12))
    _fh_full(hn2.paint, context(now=1.0 + 0.5 / 12))
    n_in = len(hn2.mode._pix)
    hn3 = _host35(_AN32)
    hn3.mode2._go("fall", 1.0)           # one soldier falling while the other walks
    hn3.advance(context(now=1.0 + 0.5 / 12))
    _fh_full(hn3.paint, context(now=1.0 + 0.5 / 12))
    n_out = len(hn3.mode._pix)
    check("in step the two halves build ONE pixmap for a frame; out of step (one falling) they build two",
          n_in == 1 and n_out == 2, f"in step {n_in}, out of step {n_out}")
    check("a fall is per half: one soldier down does not stop the other (each keeps its own count)",
          hn3.mode2._state in ("fall", "rest") and hn3.mode._state == "walk")

    # -- D. Pop-Up: ONE shared reveal --------------------------------------------------
    hp = _host35(_P32)
    LP, RP = hp.mode, hp.mode2
    check("Pop-Up: both halves hold ONE shared reveal object, seats 0 and 1",
          LP._reveal is RP._reveal and (LP.seat, RP.seat) == (0, 1))
    live_same = names_same = origin_same = mirror_ok = True
    cues = []
    t_ = 1.0
    seen = 0
    for _ in range(60 * 60):
        t_ += 1 / 60.0
        hp.advance(context(now=t_))
        snd = hp.take_sounds()
        if snd:
            cues.append(snd)
        if LP.state == "up" or RP.state == "up":
            if (LP.state == "up") != (RP.state == "up"):
                live_same = False
        if LP.state == "up" and RP.state == "up" and len(LP.shown) > seen:
            seen = len(LP.shown)
            names_same &= (LP.shown[-1] != RP.shown[-1])
            origin_same &= (LP.shown_at == RP.shown_at)
            fl_, fr_ = LP.playfield(hp._side_ctx(context(now=t_), 0)), RP.playfield(hp._side_ctx(context(now=t_), 1))
            r0 = (LP._centre[0] - fl_.left()) / fl_.width()
            r1 = (RP._centre[0] - fr_.left()) / fr_.width()
            mirror_ok &= abs(r0 + r1 - 1.0) < 0.12 and abs(LP._centre[1] - RP._centre[1]) < 0.1 * fl_.height()
    check("Pop-Up: both halves go live in the same tick (neither is handed a figure earlier) -- "
          f"{seen} appearances", live_same and seen >= 5, f"{seen}")
    check("...a DIFFERENT figure on each half every time, from the SAME origin, in mirrored position",
          names_same and origin_same and mirror_ok)
    check("...and exactly one appearance cue per appearance (the host collapses the pair)",
          len(cues) == seen and all(c == [_FA35] for c in cues), f"{len(cues)} cues for {seen} appearances")
    # a shot's own sound rides the outcome and is routed by side, never through the drain
    hd = _host35(_P32)
    t_ = 1.0
    for _ in range(60 * 6):
        t_ += 1 / 60.0
        hd.advance(context(now=t_))
        if hd.mode.state == "up" and hd.mode2.state == "up":
            break
    snds = []
    for k_ in range(12):
        ap_ = hd.mode.aim_point(None)
        if ap_ is None:
            break
        t_ += 0.02
        o_ = hd.shot(modes.ShotEvent(ap_[0], ap_[1], t_, k_ + 1), context(now=t_))
        snds.append(o_.sound)
        if o_.sound:
            break
    from lsa.audio.mixer import BANSHEE_SOUND as _BA35, BANSHEE_01_SOUND as _BB35, GRUNT_SOUNDS as _GR35
    check("Pop-Up: the drop sound (per figure since 2026-10-20: a grunt, or a banshee for zombie02 / alien01, whichever figure the "
          "random cast drew) travels on the shot's outcome (routed per side by the host)",
          any(x_ in _GR35 or x_ in (_BA35, _BB35) for x_ in snds), str(snds))

    # a half that is finished never holds up the other (Pop-Up has Reaction's seat problem)
    for first in (0, 1):
        hq = _host35(_P32)
        _fire(hq, context(now=1.0), first, N31, on_target=False)
        other = hq.mode2 if first == 0 else hq.mode
        t_ = 1.0
        live_at = None
        for _ in range(60 * 14):
            t_ += 1 / 60.0
            hq.advance(context(now=t_))
            if other.state == "up":
                live_at = t_
                break
        check(f"Pop-Up: with side {first} finished, the other side's figure STILL appears (the retired seat is not waited on)",
              live_at is not None, f"{live_at}")

    # -- E. the sound drain covers BOTH halves (a cue queued only by the right half is heard)
    hsd = _host35(_P32)
    hsd.mode2._queued_sound = _FA35
    check("a cue queued by the RIGHT half alone is drained and heard (the app used to drain the left half only)",
          hsd.take_sounds() == [_FA35] and hsd.take_sounds() == [])
    hsd.mode._queued_sound = _FA35
    hsd.mode2._queued_sound = _FA35
    hsd.mode2._queued_sound = _FS35 if False else _FA35
    check("...identical cues from the two halves in one tick are ONE sound; different cues are both kept",
          hsd.take_sounds() == [_FA35]
          and (setattr(hsd.mode, "_queued_sound", _FA35), setattr(hsd.mode2, "_queued_sound", _FS35),
               hsd.take_sounds() == [_FA35, _FS35])[-1])

    # ------------------------------------------------------------------
    # 36. the verdict screen: ONE round, the same height every round, agreeing
    #     with its own numbers, and NOTHING on it moves until a player acts
    # ------------------------------------------------------------------
    print()
    print("36. verdict screen: this round only, constant height, internally consistent, and inert until a player acts")
    import re as _re36
    import zlib as _zl36
    N36 = _DEFAULT_MAG
    ALL36 = (_Static, EscalatingBall, _B31, ClearTheBoard, _R31, _I32, _A32, _AN32, _P32)

    def _spy_paint36(h_, c_):
        """Paint the host and capture every (lines, size, box) it drew."""
        rec_: list = []
        orig_ = _Host.__dict__["_draw_lines"]

        def spy_(p_, box_, lines_, start_, bold=False, floor=9):
            sz_ = orig_.__func__(p_, box_, lines_, start_, bold, floor)
            rec_.append((list(lines_), sz_, (box_.left(), box_.top(), box_.width(), box_.height())))
            return sz_
        _Host._draw_lines = staticmethod(spy_)
        try:
            img_ = _render32(h_.paint, c_)
        finally:
            _Host._draw_lines = orig_
        return rec_, img_

    def _round36(h_, c_, t_, left_hits, right_hits):
        """Fire a whole round through the host: `*_hits` on-target shots, the rest off.

        PACED AT A HUMAN RATE (`PACE`, 0.5 s a shot). It was 0.05 s, a burst: sixteen
        gaps made 0.8 s against a discard floor of 3.64 s, so since 2026-10-22 every
        round here was a DISCARDED storm, which by design shows NO efficiency -- and
        the efficiency sentence this section reads came out blank in all four, which
        the premise check below reported. The app changed (deliberately); the fixture
        had not."""
        for k_ in range(N36):
            for s_ in (0, 1):
                want_ = (left_hits, right_hits)[s_]
                pt_ = _aim32(h_, s_, t_) if k_ < want_ else None
                if pt_ is None:
                    pt_ = (60.0 if s_ == 0 else W32 - 60.0, 700.0)
                h_.shot(modes.ShotEvent(pt_[0], pt_[1], t_, k_ + 1), c_)
            t_ += PACE
        return t_

    # -- A. consistency across four rounds, where the ROUND and the MATCH disagree ---
    hr36, cr36 = _vh25(_Static, now=1.0)
    t36 = 1.0
    plan36 = [(N36, 0), (N36 // 2, 0), (0, 0), (0, N36 // 2)]      # (left hits, right hits) per round
    seen36 = []
    seen_eff36 = []
    disc36 = []
    for rnd36, (lh36, rh36) in enumerate(plan36, start=1):
        c36 = _c32(t36)
        t36 = _round36(hr36, c36, t36, lh36, rh36)
        rec36, _img36 = _spy_paint36(hr36, _c32(t36 + 0.01))
        texts36 = [x for ls_, _s, _b in rec36 for x in ls_]
        cols36 = [ls_ for ls_, _s, _b in rec36 if ls_ and ls_[0] in ("LEFT", "RIGHT")]
        pts36 = {ls_[0]: int(_re36.match(r"(\d+)(?: of \d+)? point", ls_[2]).group(1)) for ls_ in cols36}
        res36 = next((x for x in texts36 if "WINS by" in x or x == "TIED"), "")
        m36 = _re36.match(r"(LEFT|RIGHT) WINS by (\d+) points?", res36)
        if m36:
            agree36 = (pts36[m36.group(1)] - pts36["LEFT" if m36.group(1) == "RIGHT" else "RIGHT"] == int(m36.group(2))
                       and pts36[m36.group(1)] > pts36["LEFT" if m36.group(1) == "RIGHT" else "RIGHT"])
        else:
            agree36 = res36 == "TIED" and pts36["LEFT"] == pts36["RIGHT"]
        match36 = hr36.totals[hr36.seats[0]].points - hr36.totals[hr36.seats[1]].points
        seen36.append((rnd36, pts36, res36, match36))
        check(f"round {rnd36}: the rendered result agrees arithmetically with the rendered points "
              f"({pts36} -> {res36!r})", agree36 and f"ROUND {rnd36} COMPLETED" in texts36,
              f"{pts36} {res36}")
        check(f"round {rnd36}: nothing on the screen states the match, a previous round or a swap",
              not any("MATCH" in x.upper() or "swap" in x.lower() or "in the match" in x for x in texts36)
              and sum(1 for x in texts36 if "COMPLETED" in x) == 1)
        # THE RESULT LINE IS BOLD END TO END (his ruling, 2026-10-22: "'Right wins by 11
        # points' should be highlighted in bold"), read from what the PAINTER was handed, not
        # from `result_words`: the seam is `_paint_verdict` -> `emphasised` -> `emphasise`.
        # The efficiency sentence under it is the aside and has only its numbers lifted.
        from lsa.modes import runs as _R36b
        mid36 = next((ls_ for ls_, _s, _b in rec36
                      if ls_ and len(ls_) == 2 and (ls_[0] == "TIED" or "WINS by" in ls_[0])), None)
        res_runs36 = getattr(mid36[0], "runs", ()) if mid36 else ()
        eff_runs36 = getattr(mid36[1], "runs", ()) if mid36 and mid36[1] else ()
        seen_eff36.append(bool(mid36 and mid36[1]))
        disc36.append((hr36.side_discarded(0), hr36.side_discarded(1)))
        check(f"round {rnd36}: the result line reaches the painter as ONE bold coloured run, end to end",
              mid36 is not None and isinstance(mid36[0], _R36b.RunLine)
              and len(res_runs36) == 1 and res_runs36[0].bold is True and res_runs36[0].color is not None
              and res_runs36[0].text == res36, f"{mid36!r} {res_runs36}")
        check(f"round {rnd36}: the efficiency sentence beside it is NOT whole-line bold (numbers only, or blank)",
              mid36 is not None and (not mid36[1] or (
                  isinstance(mid36[1], _R36b.RunLine)
                  and any(not r_.bold for r_ in eff_runs36)
                  and all(r_.text.replace(".", "").replace("%", "").isdigit() for r_ in eff_runs36 if r_.bold))),
              f"{mid36!r}")
        if rnd36 < len(plan36):
            hr36.restart(_c32(t36))
            t36 += 0.5
    check("(premise) NO side of any of those rounds was DISCARDED as a storm: they are paced at a "
          "human rate, so a sound round's efficiency is what is being drawn and read "
          "(a discarded side draws none, which is how the check below once saw only blanks)",
          disc36 and not any(a_ or b_ for a_, b_ in disc36), str(disc36))
    check("(premise) at least one of those rounds drew a NON-BLANK efficiency sentence, so the "
          "'not whole-line bold' check above was exercised on a real sentence and not only on blanks",
          any(seen_eff36), str(seen_eff36))
    # the premise that makes the check sharp: in round 2 the match leader is NOT the round's winner
    check("(premise) in round 2 the running MATCH points the other way from that round's result "
          "(the old code would have printed the match winner over this round's points)",
          seen36[1][2].startswith("LEFT WINS") and seen36[1][3] < 0, str(seen36[1]))
    # R1 left=p1 200; R2 (swapped) left=p2 100; R3 0; R4 (swapped back) right=p1 100
    check("the match is still accumulated per PLAYER across the swaps: player 1 has a full magazine "
          "plus a half one, player 2 the half one",
          (hr36.totals[0].points, hr36.totals[1].points)
          == (N36 * 10 + (N36 // 2) * 10, (N36 // 2) * 10),
          f"{hr36.totals}")
    match_ev36 = [e for e in hr36.take_events() if "closed" in e]
    check("...and EVERY round's closing line carries the full match state in the LOG",
          len(match_ev36) == 4 and all("match totals" in e and "MATCH" in e for e in match_ev36), str(match_ev36[:1]))

    # -- B. the block is the same height at every round -------------------------------
    hh36, ch36 = _vh25(_Static, now=1.0)
    t36 = 1.0
    boxes36, ink36 = [], []
    for rnd36 in range(1, 5):
        t36 = _round36(hh36, _c32(t36), t36, N36 if rnd36 % 2 else 3, 0 if rnd36 % 2 else min(15, N36))
        rec36, img36 = _spy_paint36(hh36, _c32(t36 + 0.01))
        boxes36.append([b_ for ls_, _s, b_ in rec36])
        ys36 = _np31.nonzero((img36 != 10).any(axis=2).any(axis=1))[0]
        ink36.append((int(ys36.min()), int(ys36.max())))
        hh36.restart(_c32(t36))
        t36 += 0.5
    print(f"         verdict ink rows (top, bottom) at rounds 1-4: {ink36}")
    check("the verdict's layout boxes are IDENTICAL at rounds 1, 2, 3 and 4: it cannot grow with the match",
          all(b_ == boxes36[0] for b_ in boxes36), str([len(b_) for b_ in boxes36]))
    check("...and so is the rendered block: the same bottom edge, and a top within a few pixels",
          len({i_[1] for i_ in ink36}) == 1 and max(i_[0] for i_ in ink36) - min(i_[0] for i_ in ink36) <= 4,
          str(ink36))

    # -- C. NOTHING moves until a player acts, with every drill's own timers pending ----
    def _closed36(cls_):
        h_ = _Host()
        c0_ = _c32(1.0)
        h_.set_mode(cls_(), c0_, players=2)
        t_ = 1.0
        for k_ in range(N36):
            for s_ in (0, 1):
                pt_ = _aim32(h_, s_, t_) or ((60.0 if s_ == 0 else W32 - 60.0), 700.0)
                h_.shot(modes.ShotEvent(pt_[0], pt_[1], t_, k_ + 1), _c32(t_))
            t_ += 0.05
            for _ in range(3):
                t_ += 1 / 60.0
                if not h_.round_over:
                    h_.advance(_c32(t_))
        return h_, t_

    idle_total, idle_ticks = 0, 0
    for cls36 in ALL36:
        h_, t_ = _closed36(cls36)
        label_ = cls36.title
        if not h_.round_over:
            check(f"{label_}: (premise) the round closed", False)
            continue
        t_ += 0.1
        h_.advance(_c32(t_))                        # the one repaint the verdict itself asks for
        _rec, img0 = _spy_paint36(h_, _c32(t_))      # first paint: the START box is now armed 0.8 s on
        for _ in range(3):
            h_.advance(_c32(t_ + 0.01))
        snap0 = (h_.round_no, id(h_.mode), id(h_.mode2), type(h_.mode), h_.round_over, h_.verdict,
                 tuple((t.shots, t.hits, t.points, t.after_done) for t in h_.tallies), h_.seats,
                 tuple((r.points, r.shots) for r in h_.totals))
        crc0 = _zl36.crc32(img0.tobytes())
        h_.take_events(); h_.take_round_ended(); h_.take_sounds()
        repaints, extra, t2 = 0, [], t_
        for i36 in range(60 * 150):                   # 150 s of play-time at 60 Hz
            t2 += 1 / 60.0
            if h_.advance(_c32(t2)):
                repaints += 1
            if i36 % 600 == 0:
                if h_.take_next_round_request() or h_.take_menu_request() or h_.take_round_ended():
                    extra.append("a request or end-of-round fact appeared")
                if h_.take_sounds():
                    extra.append("a sound was queued")
        for jump in (400.0, 4000.0, 40000.0):          # and a long jump past every timer there is
            t2 += jump
            if h_.advance(_c32(t2)):
                repaints += 1
        idle_total += repaints
        idle_ticks += 60 * 150 + 3
        snap1 = (h_.round_no, id(h_.mode), id(h_.mode2), type(h_.mode), h_.round_over, h_.verdict,
                 tuple((t.shots, t.hits, t.points, t.after_done) for t in h_.tallies), h_.seats,
                 tuple((r.points, r.shots) for r in h_.totals))
        _rec, img1 = _spy_paint36(h_, _c32(t2))
        same_px = _zl36.crc32(img1.tobytes()) == crc0
        check(f"{label_}: after 150 s plus jumps of 400 s, 4,000 s and 40,000 s the verdict is unchanged "
              f"(same round, same drills, same tallies, same pixels, no request, no sound, "
              f"{repaints} repaints in {60 * 150 + 3} ticks)",
              snap0 == snap1 and same_px and repaints == 0 and not extra and h_.round_over
              and not h_.take_events(),
              f"{'state changed ' if snap0 != snap1 else ''}{'pixels changed ' if not same_px else ''}"
              f"{repaints} repaints {extra}")
    print(f"         repaints requested while a verdict sat idle: {idle_total} in {idle_ticks} ticks across {len(ALL36)} drills")
    hcl, tcl = _closed36(ClearTheBoard)
    check("(premise) the test really does leave Clear The Board's 2.2 s refill timer pending at the round end",
          hcl.mode.cleared_at is not None or hcl.mode2.cleared_at is not None)
    han, tan = _closed36(_AN32)
    check("(premise) ...and the animated soldier mid-fall with his rest timer pending",
          han.mode._state in ("fall", "rest") or han.mode2._state in ("fall", "rest"))

    # -- D. what IS a player action on this screen, and what is not -----------------------
    hv36, _t = _closed36(_Static)
    cv36 = _c32(5.0)
    _spy_paint36(hv36, cv36)
    hv36.advance(cv36)
    rb36 = hv36.round_box(cv36)
    mb36 = hv36.menu_button.rect(cv36).toAlignedRect()
    hv36.shot(modes.ShotEvent(300.0, 900.0, 7.0, 901), cv36)
    check("a shot anywhere else on the verdict starts nothing (seen, drawn, counted, scored nowhere)",
          not hv36.take_next_round_request() and not hv36.take_menu_request() and hv36.round_over)
    check("a click anywhere else on the verdict starts nothing", hv36.on_click(300.0, 900.0, cv36) is False
          and not hv36.take_next_round_request())
    hv36.shot(modes.ShotEvent(mb36.center().x(), mb36.center().y(), 8.0, 902), cv36)
    one_hit36 = hv36.take_menu_request()
    hv36.shot(modes.ShotEvent(mb36.center().x(), mb36.center().y(), 8.1, 903), cv36)
    two_hit36 = hv36.take_menu_request()
    check("the MENU button stays live on the verdict screen: one hit arms it, the second asks for the menu "
          "(his way out is a player action)", one_hit36 is False and two_hit36 is True,
          f"{one_hit36}/{two_hit36}")
    hv36.shot(modes.ShotEvent(rb36.center().x(), rb36.center().y(), 9.0, 904), cv36)
    check("the START ROUND box (by shot) is a player action once armed", hv36.take_next_round_request() is True)

    # ------------------------------------------------------------------
    # 37. THE MAGAZINE (2026-10-20). One number ends the round everywhere and
    # is drawn as bullets: single player AND each half of two players.
    # ------------------------------------------------------------------
    print()
    print("37. the magazine: one configured number is the round's length and the bullets on the screen")
    import hashlib as _hl37
    import io as _io37
    import json as _js37
    import tempfile as _tf37
    import tokenize as _tk37
    import numpy as _np37
    from lsa import settings as _set37
    from lsa.artwork import ceiling as _ceil37, load as _load37, ui as _ui37
    from lsa.modes import magazine as _mag37, split as _split37
    from lsa.modes.base import Mode as _Mode37
    from lsa.display.layers import top_band_height as _tbh37

    N37 = _DEFAULT_MAG
    ctx37 = context(now=1.0)
    SP37 = lambda: (lambda h_: (h_.set_mode(_Static(), ctx37, players=1), h_)[1])(_Host())
    TP37 = lambda: (lambda h_: (h_.set_mode(_Static(), ctx37, players=2), h_)[1])(_Host())

    def _shoot37(h_, n_, x_=100.0, y_=1000.0, t0=1.0, c_=None):
        for i_ in range(n_):
            h_.shot(modes.ShotEvent(x_, y_, t0 + i_ * 0.01, i_ + 1), c_ or ctx37)

    def _bullets_px37(h_, geo_, c_, side_=None):
        """How many slots of the strip have ink in them, from RENDERED pixels."""
        _r, _l, a_ = _fh_full(h_.paint, c_)
        n_ = 0
        for r_ in geo_.slots:
            if int(a_[r_.top():r_.bottom() + 1, r_.left():r_.right() + 1, :3].max()) > 10 + 12:
                n_ += 1
        return n_, a_

    # -- the setting: one owner, a clean default, a clamp, no write on read ----------
    # Again: the owner and the absence of a second copy, not the number. The
    # literal is read FROM settings so the two halves cannot disagree, and the
    # "no stray copy" scan now looks for the live value rather than a list of
    # numbers that goes stale every time he changes his mind.
    check("the default magazine lives in settings, not in the host or the versus module",
          _set37.AppSettings().magazine == _set37.DEFAULT_MAGAZINE
          and _set37.MAGAZINE_MIN <= _set37.DEFAULT_MAGAZINE <= _set37.MAGAZINE_MAX
          and not hasattr(_V, "ROUND_SHOTS")
          and not any(isinstance(v_, int) and not isinstance(v_, bool)
                      and v_ in (_set37.DEFAULT_MAGAZINE, 20)
                      for k_, v_ in vars(_V).items() if not k_.startswith("__")))
    _D37 = _set37.DEFAULT_MAGAZINE          # his number; read it, never retype it
    check("normalise_magazine: whole numbers pass, the range clamps, anything else is the default (never 0)",
          [_set37.normalise_magazine(v_) for v_ in (_D37, 5, 0, -3, 999, 12.0, "9")]
          == [_D37, 5, 1, 1, 50, 12, 9]
          and all(_set37.normalise_magazine(v_) == _D37
                  for v_ in (None, "abc", True, 4.5, float("nan"), [], {})))
    with _tf37.TemporaryDirectory() as td37:
        f37 = Path(td37) / "settings.local.json"
        f37.write_text(_js37.dumps({"players": 2, "volume": 0.3}), encoding="utf-8")
        before37 = f37.read_bytes()
        loaded37 = _set37.AppSettings.load(f37)
        check("his settings file has no 'magazine' key: the default applies and READING it writes nothing",
              loaded37.magazine == _D37 and loaded37.players == 2
              and f37.read_bytes() == before37)
        f37.write_text(_js37.dumps({"magazine": 9}), encoding="utf-8")
        check("a saved magazine is honoured", _set37.AppSettings.load(f37).magazine == 9)
    _src37 = ROOT_SRC / "lsa"
    stray37 = []
    for f_ in list((_src37 / "modes").glob("*.py")) + list((_src37 / "artwork").glob("*.py")) \
            + [_src37 / "tools" / "camera_lab.py", _src37 / "display" / "scene.py"]:
        with open(f_, "rb") as fh_:
            for t_ in _tk37.tokenize(fh_.readline):
                # The literal to hunt for is whatever the magazine IS today, so
                # changing his number cannot quietly disarm this scan -- which
                # is what a hard-coded "17" would have done the moment he said
                # nineteen.
                if t_.type == _tk37.NUMBER and t_.string in (str(_D37), f"{_D37}.0"):
                    stray37.append(f"{f_.name}:{t_.start[0]}")
    check(f"no second copy of the magazine: the number {_D37} appears as a code literal nowhere outside "
          f"settings.py (modes, artwork, the app, the scene)", stray37 == [], str(stray37))
    names37 = []
    for f_ in _src37.rglob("*.py"):
        with open(f_, "rb") as fh_:
            for t_ in _tk37.tokenize(fh_.readline):
                if t_.type == _tk37.NAME and t_.string == "ROUND_SHOTS":
                    names37.append(f"{f_.name}:{t_.start[0]}")
    check("...and ROUND_SHOTS is a name nowhere in src (comments may say what it used to be)",
          names37 == [], str(names37))

    # -- a registered shot spends a round; a click never does -----------------------
    h37 = SP37()
    check("single player starts with the whole magazine", h37.remaining(0) == N37 and not h37.round_over)
    _shoot37(h37, 1)
    check("a registered shot spends one round, hit or miss (this one is a miss off every target)",
          h37.remaining(0) == N37 - 1 and h37.tallies[0].shots == 1)
    tl37 = (h37.tallies[0].shots, h37.tallies[0].hits, h37.tallies[0].points, h37.remaining(0))
    for x_, y_ in ((1920, 1080), (100, 100), (3000, 1800), (1920, 2000)):
        h37.on_click(float(x_), float(y_), ctx37)
    check("a CLICK is never a shot: ten clicks on the targets, the screen and the button spend no round",
          (h37.tallies[0].shots, h37.tallies[0].hits, h37.tallies[0].points, h37.remaining(0)) == tl37
          and h37.on_click(*_aim(h37, ctx37, 0), ctx37) in (True, False)
          and h37.remaining(0) == tl37[3] and h37.tallies[0].shots == tl37[0])

    # -- the round ends on the Nth registered shot ----------------------------------
    h37 = SP37()
    _shoot37(h37, N37 - 1)
    check("single player: one short of the magazine, the round is open", not h37.round_over
          and h37.remaining(0) == 1)
    _shoot37(h37, 1, t0=9.0)
    check("single player: the Nth registered shot empties the magazine and ends the round, once",
          h37.round_over and h37.remaining(0) == 0 and h37.take_round_ended() is True
          and h37.take_round_ended() is False)
    for n_ in (1, 5, 33, 50):
        hn37 = SP37()
        hn37.set_magazine(n_)
        hn37.restart(ctx37)
        _shoot37(hn37, n_ - 1)
        open_ = not hn37.round_over
        _shoot37(hn37, 1, t0=9.0)
        check(f"setting the magazine to {n_} makes the round {n_} shots, in single player",
              open_ and hn37.round_over and hn37.tallies[0].shots == n_ and hn37._round_mag == n_)
    ht37 = TP37()
    ht37.set_magazine(6)
    ht37.restart(ctx37)
    _shoot37(ht37, 5, x_=100.0)
    check("two players, magazine 6: the left side at 5 is not done",
          not ht37._side_done(0) and ht37.remaining(0) == 1 and ht37.remaining(1) == 6)
    _shoot37(ht37, 1, x_=100.0, t0=5.0)
    check("...the 6th empties the LEFT magazine and not the round", ht37._side_done(0) and not ht37.round_over)
    _shoot37(ht37, 6, x_=3700.0, t0=6.0)
    check("...and the RIGHT side's 6th ends the round: each half has its own magazine",
          ht37.round_over and ht37.tallies[1].shots == 6 and ht37.totals[0].shots == 6)
    hm37 = SP37()
    hm37.set_magazine(3)
    _shoot37(hm37, 2)
    hm37.set_magazine(8)
    _shoot37(hm37, 1, t0=4.0)
    check("a setting changed MID-round does not end or stretch that round: it ends at the size it was dealt",
          hm37.round_over and hm37.tallies[0].shots == 3)
    hm37.restart(ctx37)
    check("...and the next round is dealt the new size", hm37._round_mag == 8 and hm37.remaining(0) == 8)

    # -- the menu is a screen, not a drill: shooting a tile spends nothing ---------------
    hmenu37 = _Host()
    hmenu37.set_mode(modes.by_name("menu")(), ctx37, players=1)
    _shoot37(hmenu37, N37 + 3, x_=5.0, y_=5.0)
    check("the MENU spends no rounds and never ends a round",
          hmenu37.tallies[0].shots == 0 and not hmenu37.round_over)

    # -- the bullets, asserted on RENDERED pixels ---------------------------------------
    h37 = SP37()
    geo37 = h37.single_strip_geometry(ctx37)
    check("the single-player strip has exactly one slot per round in the magazine",
          len(geo37.slots) == N37)
    for fired37 in (0, 1, 5, N37 - 1):
        h37 = SP37()
        _shoot37(h37, fired37)
        n_px37, _a37 = _bullets_px37(h37, h37.single_strip_geometry(ctx37), ctx37)
        check(f"after {fired37} shots the render shows {N37 - fired37} bullets (counted from pixels)",
              n_px37 == N37 - fired37, f"{n_px37}")
    h37 = SP37()
    _shoot37(h37, 6)
    g37 = h37.single_strip_geometry(ctx37)
    n_px37, a37 = _bullets_px37(h37, g37, ctx37)
    spent37 = g37.slots[N37 - 6:]
    check("the REMAINING bullets did not move as the magazine emptied: slot 0 is the same place at 17 and at 11",
          g37.slots[0] == SP37().single_strip_geometry(ctx37).slots[0] and n_px37 == N37 - 6)
    check("...and the spent slots are bare ground (the inner end empties first)",
          all(int(a37[r_.top():r_.bottom() + 1, r_.left():r_.right() + 1, :3].max()) <= 10 + 12 for r_ in spent37))
    nb37 = g37.numeral
    check("the remaining count is also a numeral at the fixed outer end, drawn in the same pass",
          int(a37[nb37.top():nb37.bottom() + 1, nb37.left():nb37.right() + 1, :3].max()) > 10 + 12)
    h37 = SP37()
    _shoot37(h37, N37 - 1)
    _shoot37(h37, 1, t0=9.0)
    # at the verdict the strip stands down (the end screen is the readout)
    n_end37, _ = _bullets_px37(h37, h37.single_strip_geometry(ctx37), ctx37)
    check("on the end screen the strip is gone (nothing left to count, and the screen is clear)", n_end37 == 0)

    # two players: one strip per half, the right number each, from pixels
    h37 = TP37()
    _shoot37(h37, 5, x_=100.0)
    lay37 = h37._layout(ctx37)
    gl37, gr37 = h37.half_strip_geometry(ctx37, lay37, 0), h37.half_strip_geometry(ctx37, lay37, 1)
    nl37, a37 = _bullets_px37(h37, gl37, ctx37)
    nr37, _ = _bullets_px37(h37, gr37, ctx37)
    check("two players: the left half shows 12 bullets after 5 shots, the right half still shows 17",
          (nl37, nr37) == (N37 - 5, N37), f"{nl37}/{nr37}")
    _shoot37(h37, N37 - 5, x_=100.0, t0=5.0)
    nl37, _ = _bullets_px37(h37, gl37, ctx37)
    nr37, _ = _bullets_px37(h37, gr37, ctx37)
    check("...a finished half shows none while the other half still shows its own",
          (nl37, nr37) == (0, N37) and h37._side_done(0) and not h37.round_over)

    # -- where they are drawn -----------------------------------------------------------
    corners37 = ("top left", "top right", "bottom left", "bottom right")
    was37 = _Mode37.MENU_CORNER
    try:
        for corner_ in corners37:
            _Mode37.MENU_CORNER = corner_
            h37 = SP37()
            g_ = h37.single_strip_geometry(ctx37)
            menu_ = _Mode37.menu_rect(ctx37)
            keep_ = _Mode37.menu_keepout(ctx37)
            on_left_ = g_.bounds.center().x() < ctx37.width / 2
            check(f"MENU in the {corner_}: the strip is in the LOWER corner on the OPPOSITE side, "
                  f"clear of the button's keep-out (derived, not hard-coded)",
                  g_.bounds.top() > ctx37.height * 0.85 and g_.bounds.bottom() < ctx37.height
                  and on_left_ == (menu_.center().x() > ctx37.width / 2)
                  and not QRect(g_.bounds).intersects(keep_.toAlignedRect()),
                  f"{g_.bounds} vs {keep_}")
            h2_ = TP37()
            lay_ = h2_._layout(ctx37)
            for side_ in (0, 1):
                gg_ = h2_.half_strip_geometry(ctx37, lay_, side_)
                half_ = lay_.side(side_)
                bad_ = []
                if not half_.contains(gg_.bounds):
                    bad_.append("outside its half")
                if gg_.bounds.top() < lay_.scoreline.bottom() + 1:
                    bad_.append("over the scoreline")
                if gg_.bounds.top() < _tbh37(ctx37):
                    bad_.append("inside the top band")
                if QRect(gg_.bounds).intersects(_Mode37.menu_keepout(ctx37).toAlignedRect()):
                    bad_.append("on the MENU keep-out")
                if len(gg_.slots) != N37:
                    bad_.append("wrong slot count")
                check(f"two players, MENU in the {corner_}, half {side_}: under the scoreline row, clear of the top "
                      f"band and the button, inside its half", not bad_, str(bad_))
    finally:
        _Mode37.MENU_CORNER = was37
    h2_ = TP37()
    lay_ = h2_._layout(ctx37)
    gr_ = h2_.half_strip_geometry(ctx37, lay_, 1)
    check("the half without the button has its strip CENTRED in the half (within 2 px)",
          abs(gr_.bounds.center().x() - lay_.right.center().x()) <= 2, f"{gr_.bounds.center().x()} vs {lay_.right.center().x()}")
    check("a drill's ground begins UNDER the bullets: `below_scoreline` honours the magazine band",
          lay_.below_scoreline(QRect(lay_.left)).top() > h2_.half_strip_geometry(ctx37, lay_, 0).bounds.bottom())
    # half-width layout does not rely on px(): the strip is fitted to the field's share of the width
    for nbig in (30, 50):
        hb37 = TP37()
        hb37.set_magazine(nbig)
        hb37.restart(ctx37)
        lb37 = hb37._layout(ctx37)
        okb37 = all(lb37.side(sd_).contains(hb37.half_strip_geometry(ctx37, lb37, sd_).bounds) for sd_ in (0, 1))
        check(f"a magazine of {nbig} in a half still fits inside that half (shrunk to the field's share, B5)", okb37)
        hs_ = SP37()
        hs_.set_magazine(nbig)
        hs_.restart(ctx37)
        gb_ = hs_.single_strip_geometry(ctx37)
        check(f"a magazine of {nbig} in single player stays inside the screen and under half its width",
              QRect(0, 0, SW, SH).contains(gb_.bounds) and gb_.bounds.width() <= SW * 0.47)

    # -- B1 / B1b / B10: red excess 0, nothing above luminance 60, nothing when measuring ----
    worst_red37, worst_lit37 = 0, 0.0
    for lbl_, mk_, fired_ in (("single full", SP37, 0), ("single 1 left", SP37, N37 - 1),
                              ("two full", TP37, 0)):
        h37 = mk_()
        if fired_:
            _shoot37(h37, fired_)
        for fn_ in ((lambda p_, c_: h37._paint_single_strip(p_, c_)) if mk_ is SP37 else
                    (lambda p_, c_: [h37._paint_half_strip(p_, c_, h37._layout(c_), sd_) for sd_ in (0, 1)]),):
            red_, lit_, a_ = _fh_full(fn_, ctx37)
            worst_red37, worst_lit37 = max(worst_red37, red_), max(worst_lit37, lit_)
            mx_ = int(a_[..., :3].max())
            check(f"strip alone, {lbl_}: red excess 0 and no pixel brighter than luminance 60 ({mx_})",
                  red_ == 0 and mx_ <= 60 and lit_ == 0.0)
    print(f"         strip lit area above luminance 60: {worst_lit37 * 100:.4f}% (worst red excess {worst_red37})")
    mctx37 = context(now=1.0)
    mctx37.diagnostics["measuring"] = True
    h37 = SP37()
    _r37, _l37, am37 = _fh_full(lambda p_, c_: h37._paint_single_strip(p_, c_), mctx37)
    check("the strip stands down while the app measures the screen (RULES B2): nothing is drawn",
          int(am37[..., :3].max()) <= 10)
    worst37 = (0.0, "")
    for cls_ in (_Static, EscalatingBall, _B31, ClearTheBoard, _R31, _I32, _A32, _AN32, _P32):
        h37 = _Host()
        h37.set_mode(cls_(), ctx37, players=1)
        r_on, l_on, _a = _fh_full(h37.paint, ctx37)
        r_off, l_off, _a = _fh_full(h37.paint, mctx37)
        if l_on > worst37[0]:
            worst37 = (l_on, cls_.__name__)
        check(f"{cls_.__name__}: the strip adds NO lit area and no red excess (with {l_on * 100:.3f}%, without {l_off * 100:.3f}%)",
              r_on == 0 and abs(l_on - l_off) < 1e-9 and l_on < 0.02, f"{r_on} {l_on} {l_off}")
    print(f"         busiest drill with the full strip: {worst37[0] * 100:.3f}% lit ({worst37[1]}), budget 2%")

    # -- the repaints: only when a shot lands ------------------------------------------
    h37 = SP37()
    for i_ in range(3):
        h37.advance(context(now=2.0 + i_ / 60.0))
    idle37 = sum(1 for i_ in range(180) if h37.advance(context(now=3.0 + i_ / 60.0)))
    print(f"         repaints requested over 180 idle ticks with a full magazine: {idle37}")
    check("a full magazine asks for NO repaints while idle (the strip never animates)", idle37 == 0)
    _shoot37(h37, 1)
    asked37 = [h37.advance(context(now=9.0 + i_ / 60.0)) for i_ in range(10)]
    check("a shot landing asks for exactly one repaint of the strip", asked37.count(True) == 1)

    # -- the end screen: one box, one start path, the same dry gun --------------------------
    hs37 = SP37()
    ht37 = TP37()
    _shoot37(hs37, N37)
    _shoot37(ht37, N37, x_=100.0)
    _shoot37(ht37, N37, x_=3700.0, t0=9.0)
    check("single player's end screen uses the SAME box as two players (one function, one rectangle)",
          hs37.round_box(ctx37) == ht37.round_box(ctx37) and hs37.round_box(ctx37).width() > 0)
    rec37, _img37 = _spy_paint36(hs37, context(now=20.0))
    texts37 = [x for ls_, _s, _b in rec37 for x in ls_]
    # The points line reads "N of M points" since 2026-10-22 (his question: "65 points of
    # how many possible points?"; `versus.points_of_word`). The block is asserted as an
    # ORDERED prefix, line by line, not as a bag of substrings: a reordered or dropped
    # line must fail. M is matched as a positive whole number rather than pinned to 170,
    # because it is the ghost's ceiling for the drill, not a property of this screen.
    import re as _re37
    head37 = texts37[:6]
    m37 = _re37.fullmatch(r"0 of (\d+) points", head37[4]) if len(head37) == 6 else None
    check("the end screen reads (2026-10-20, points line 2026-10-22): ROUND n / shots / hits / misses / "
          "0 of M points / seconds, IN THAT ORDER, then NEXT ROUND last, and no MAGAZINE EMPTY",
          len(head37) == 6 and head37[:4] == ["ROUND 1", f"{N37} shots", "0 hits", f"{N37} misses"]
          and m37 is not None and int(m37.group(1)) > 0
          and _re37.fullmatch(r"[\d.]+ seconds", head37[5]) is not None
          and "NEXT ROUND" in texts37 and texts37.index("NEXT ROUND") > 5
          and texts37[-1] == "NEXT ROUND"
          and "MAGAZINE EMPTY" not in texts37 and not any("START" in t_ for t_ in texts37)
          and hs37.verdict_lines() == ["ROUND 1"], str(texts37))
    check("...the targets are gone from it (the drill is not painted) and the strip is not on it",
          not any("round" in x.lower() and "start" not in x.lower() for x in texts37 if x.startswith("round")))
    seen37 = hs37._box_seen_at
    check("(premise) painting the end screen stamped the moment the box first appeared", seen37 == 20.0)
    ctr37 = hs37.round_box(ctx37).center()
    o_early = hs37.shot(modes.ShotEvent(ctr37.x(), ctr37.y(), seen37 + 0.5, 901), ctx37)
    check("a shot at the START box 0.5 s after it appears does NOTHING (the phantom guard: armed after 0.8 s)",
          not hs37.take_next_round_request() and hs37.round_over)
    check("...but it BANGS and leaves its hole (he is told the app saw it), neither a hit nor a miss",
          o_early.play_sound and o_early.draw_hole and o_early.unscored and not o_early.hit and o_early.dry)
    o_late = hs37.shot(modes.ShotEvent(ctr37.x(), ctr37.y(), seen37 + 0.9, 902), ctx37)
    check("once armed, ONE shot at the box asks for the next round, and bangs and draws its hole",
          hs37.take_next_round_request() is True and o_late.play_sound and o_late.draw_hole)
    o_dry = hs37.shot(modes.ShotEvent(300.0, 900.0, seen37 + 2.0, 903), ctx37)
    check("a shot anywhere else on the end screen is a pull with an empty magazine that still answers: "
          "gunshot and hole as normal, unscored, counted in after_done",
          o_dry.play_sound and o_dry.draw_hole and o_dry.unscored and o_dry.dry and not o_dry.hit
          and hs37.tallies[0].after_done >= 1 and not hs37.take_next_round_request())
    hs37.on_click(float(ctr37.x()), float(ctr37.y()), ctx37)
    check("a CLICK on the box asks for the next round by the same request a shot makes",
          hs37.take_next_round_request() is True)
    t37 = 30.0
    moved37 = False
    for i_ in range(6000):                      # a hundred seconds of an idle end screen
        t37 += 1 / 60.0
        hs37.advance(context(now=t37))
        moved37 = moved37 or hs37.take_next_round_request() or hs37.take_menu_request()
    check("NOTHING auto-starts or returns to the menu from the end screen, however long it is left",
          hs37.round_over and not moved37 and hs37.mode.name == "static")
    hs37.restart(ctx37)
    check("starting the next round (N, the box, a click: one function) deals a fresh full magazine, "
          "keeps the mode, and carries the round number",
          not hs37.round_over and hs37.remaining(0) == N37 and hs37.round_no == 2 and hs37.mode.name == "static"
          and hs37.seats == (0, 1) and hs37.tallies[0].after_done == 0)
    ev37 = hs37.take_events()
    check("the trail records the magazine at round start and each magazine running dry",
          any(f"magazine {N37}" in e and "round 1 starts" in e for e in ev37)
          and any(f"magazine {N37} is empty" in e for e in ev37)
          and any("round 1 closed" in e for e in ev37)
          and any("round-over screen: shot seen at" in e for e in ev37)
          and any(f"round 2 starts" in e and f"magazine {N37}" in e for e in ev37), str(ev37))

    # -- a dry pull: silent, holeless, counted -- before and after ------------------------------
    hd37 = SP37()
    _shoot37(hd37, N37 - 1)
    last37 = hd37.shot(modes.ShotEvent(100.0, 1000.0, 5.0, 77), ctx37)
    check("the LAST round still bangs and still leaves its hole (the dry gun starts AFTER it)",
          last37.play_sound and last37.draw_hole and hd37.round_over)
    dry37 = hd37.shot(modes.ShotEvent(100.0, 1000.0, 6.0, 78), ctx37)
    check("the next pull (the round-over screen is up) answers normally and is still counted: sound, hole, unscored",
          dry37.play_sound is True and dry37.draw_hole is True and dry37.unscored and hd37.tallies[0].after_done == 1)
    check("a fresh round has the whole magazine and no dry pulls; an emptied one has none left",
          hd37.remaining(0) == 0 and SP37().remaining(0) == N37 and TP37().remaining(1) == N37
          and SP37().dry_camera == 0)
    hh37 = TP37()
    _shoot37(hh37, N37, x_=100.0)
    check("...and in two players it is True while only ONE half is empty (a click cannot be given to a side)",
          hh37.remaining(0) == 0 and hh37.remaining(1) == N37 and not hh37.round_over)

    # -- the artwork: true grey, capped, fingerprinted ----------------------------------------------
    bp37 = _ui37.bullet_path()
    raw37 = cv2.imread(str(bp37), cv2.IMREAD_UNCHANGED)
    col37 = cv2.imread(str(bp37), cv2.IMREAD_COLOR).astype(_np37.int16)
    check("images/bullet.png is a one-channel PNG whose colour read has red excess exactly 0",
          bp37.suffix == ".png" and raw37 is not None and raw37.ndim == 2
          and int((col37[..., 2] - _np37.maximum(col37[..., 1], col37[..., 0])).max()) == 0)
    worst_screen37 = 0
    for hp_ in (40, 65, 130, 260):
        b_ = _ui37.bullet_for(hp_)
        v_ = int(b_.max())
        worst_screen37 = max(worst_screen37, int(round(v_ + _load37.BACKGROUND * (1 - v_ / 255.0))))
        check(f"the cartridge drawn {hp_}px tall never exceeds the artwork ceiling or the lit-area cap",
              v_ <= _ceil37.figure_level() and v_ <= _ui37.LIT_CAP_FILE and b_.shape[0] == hp_)
    print(f"         brightest bullet pixel as it lands on the screen: luminance {worst_screen37} (lit threshold 60)")
    check("...which lands at or under luminance 60 on the target background, so it spends none of the 2% budget",
          worst_screen37 <= _load37.LIT_LEVEL)
    real_fl37 = _ceil37.figure_level
    try:
        _ceil37.figure_level = lambda folder=None: 30.0
        check("the cap follows the measured ceiling: a darker figure added later dims the bullets too",
              _ui37.cap_level() == 30 and int(_ui37.bullet_for(130).max()) == 30)
    finally:
        _ceil37.figure_level = real_fl37
    import owned_files as _of37
    check("images/bullet.png is fingerprinted with the rest of what the tests may never touch",
          "resources/images/bullet.png" in _of37.OWNED)

    # ------------------------------------------------------------------
    # 38. DRY PULLS: the UNION of the microphone and the camera, never their
    # sum, logged always, shown small, and scored by nothing (2026-10-20).
    # ------------------------------------------------------------------
    print()
    print("38. dry pulls: one per pull whichever channel saw it, a lower bound without a calibrated microphone, never scored")
    from lsa.audio.listener import Click as _Click38, ShotCorroborator as _Co38

    MAG38 = 3

    def _dry_total38(h_):
        return _V.dry_total(h_.dry_camera, h_.dry_mic_only)

    def _dry_host38(players=1, mic=True):
        h_ = _Host()
        h_.set_magazine(MAG38)
        h_.set_mode(_Static(), ctx37, players=players)
        h_.mic_ready = mic
        return h_

    def _empty38(h_, t0=1.0):
        """Fire the magazine out (both halves in two players). Returns the time
        of the last live shot's final frame."""
        t_ = t0
        for side_x in ((100.0,) if not h_.versus else (100.0, 3700.0)):
            for i_ in range(MAG38):
                t_ = t0 + i_ * 0.1
                h_.shot(modes.ShotEvent(side_x, 1000.0, t_, i_ + 1, t_end=t_ + 0.05), ctx37)
        return t_ + 0.05

    def _pull38(h_, co_, t_, camera=True, click=True):
        """One trigger pull: the camera detection (a dry ShotEvent, announced to
        the corroborator under the SAME time the app uses) and/or its click."""
        if camera:
            h_.shot(modes.ShotEvent(100.0, 1000.0, t_, 500, t_end=t_ + 0.05), ctx37)
            co_.add_shot(t_ + 0.05)
        if click:
            co_.add_clicks([_Click38(t=t_ - 0.004, level=0.3, floor=0.01)])

    def _settle38(h_, co_, now_):
        co_.resolve(now_)
        h_.note_pull_outcomes(co_.take_outcomes())

    # -- one pull on BOTH channels counts ONCE ------------------------------------------
    h38, co38 = _dry_host38(), _Co38()
    t_last38 = _empty38(h38)
    co38.add_shot(t_last38)                          # the last live shot was seen and heard too
    co38.add_clicks([_Click38(t=t_last38 - 0.004, level=0.3, floor=0.01)])
    _settle38(h38, co38, t_last38 + 1.0)
    check("(premise) the magazine is empty and the last live shot's own pairing is NOT a dry pull",
          h38.round_over and h38.dry_camera == 0 and h38.dry_corroborated == 0)
    _pull38(h38, co38, t_last38 + 2.0)
    _settle38(h38, co38, t_last38 + 4.0)
    check("ONE dry pull heard by the microphone AND seen by the camera is counted ONCE (the union, not the sum)",
          _dry_total38(h38) == 1 and h38.dry_camera == 1 and h38.dry_corroborated == 1
          and h38.dry_mic_only == 0,
          f"total {_dry_total38(h38)}, camera {h38.dry_camera}, both {h38.dry_corroborated}, mic-only {h38.dry_mic_only}")
    check("(the trap, stated) adding the channels would have said 2: one click and one detection exist",
          h38.dry_camera + 1 == 2 != _dry_total38(h38))
    _pull38(h38, co38, t_last38 + 5.0)
    _pull38(h38, co38, t_last38 + 6.0)
    _settle38(h38, co38, t_last38 + 8.0)
    check("three pulls on both channels count three, not six", _dry_total38(h38) == 3
          and h38.dry_corroborated == 3)

    # -- camera-only counts -----------------------------------------------------------------
    h38, co38 = _dry_host38(), _Co38()
    t_last38 = _empty38(h38)
    _pull38(h38, co38, t_last38 + 2.0, click=False)
    _settle38(h38, co38, t_last38 + 4.0)
    check("a pull only the CAMERA saw counts: one, uncorroborated",
          _dry_total38(h38) == 1 and h38.dry_camera == 1 and h38.dry_corroborated == 0
          and h38.dry_mic_only == 0)
    check("...and a camera-only pull needs no microphone at all (the host counts the detection itself)",
          (lambda h_: (_empty38(h_), h_.shot(modes.ShotEvent(100.0, 1000.0, 9.0, 600), ctx37),
                       _dry_total38(h_))[-1])(_dry_host38(mic=False)) == 1)

    # -- microphone-only counts when calibrated, and is declared uncountable when not --------------
    h38, co38 = _dry_host38(mic=True), _Co38()
    t_last38 = _empty38(h38)
    _pull38(h38, co38, t_last38 + 2.0, camera=False)
    _settle38(h38, co38, t_last38 + 4.0)
    check("a click the camera never saw counts when the microphone is calibrated: one, heard only",
          _dry_total38(h38) == 1 and h38.dry_camera == 0 and h38.dry_mic_only == 1)
    check("...and the screen line gives the full breakdown",
          h38.dry_screen_text() == "Dry pulls: 1 (0 seen and heard, 0 seen only, 1 heard only)",
          h38.dry_screen_text())
    h38u, co38u = _dry_host38(mic=False), _Co38()
    t_last38 = _empty38(h38u)
    _pull38(h38u, co38u, t_last38 + 2.0, camera=False)
    _settle38(h38u, co38u, t_last38 + 4.0)
    check("an UNCALIBRATED microphone's click is not counted at all (its threshold is a guess)",
          _dry_total38(h38u) == 0 and h38u.dry_mic_only == 0)
    check("...and the figure says it is a LOWER BOUND instead of posing as a total",
          "or more" in h38u.dry_screen_text() and "not calibrated" in h38u.dry_screen_text()
          and "LOWER BOUND" in h38u.dry_log_text())
    h38b, co38b = _dry_host38(mic=True), _Co38()
    t_last38 = _empty38(h38b)
    _pull38(h38b, co38b, t_last38 - 0.5, camera=False)          # heard BEFORE the magazine emptied
    _settle38(h38b, co38b, t_last38 + 1.0)
    check("a click from before the magazine emptied is a live pull, never a dry one",
          _dry_total38(h38b) == 0)

    # -- a MOUSE click never counts -----------------------------------------------------------------
    h38, co38 = _dry_host38(), _Co38()
    _empty38(h38)
    for x_, y_ in ((100, 100), (1920, 1080), (3000, 1800)):
        h38.on_click(float(x_), float(y_), ctx37)
    rb38 = h38.round_box(ctx37).center()
    h38.on_click(float(rb38.x()), float(rb38.y()), ctx37)
    check("a mouse click is not a dry pull: not on the screen, not on the START box, not anywhere",
          _dry_total38(h38) == 0 and h38.dry_camera == 0 and h38.tallies[0].after_done == 0)

    # -- two players: the union again, and half-dry clicks are not attributable -----------------------------
    h38, co38 = _dry_host38(players=2), _Co38()
    t0_38 = 1.0
    for i_ in range(MAG38):
        h38.shot(modes.ShotEvent(100.0, 1000.0, t0_38 + i_ * 0.1, i_ + 1, t_end=t0_38 + i_ * 0.1 + 0.05), ctx37)
    t_half38 = t0_38 + (MAG38 - 1) * 0.1 + 0.05
    _pull38(h38, co38, t_half38 + 1.0, camera=False)             # a click while the RIGHT half still shoots
    _settle38(h38, co38, t_half38 + 3.0)
    check("two players, only the LEFT magazine empty: a heard-only click cannot be given to a side, so it is not counted",
          h38.dry_mic_only == 0 and not h38.round_over)
    h38.shot(modes.ShotEvent(100.0, 1000.0, t_half38 + 4.0, 70, t_end=t_half38 + 4.05), ctx37)
    co38.add_shot(t_half38 + 4.05)
    co38.add_clicks([_Click38(t=t_half38 + 3.996, level=0.3, floor=0.01)])
    _settle38(h38, co38, t_half38 + 6.0)
    check("...but a DETECTION on the empty left half is a dry pull (it has a position), once, corroborated",
          h38.dry_camera == 1 and h38.dry_corroborated == 1 and _dry_total38(h38) == 1)
    t_end38 = t_half38 + 7.0
    for i_ in range(MAG38):
        h38.shot(modes.ShotEvent(3700.0, 1000.0, t_end38 + i_ * 0.1, i_ + 1, t_end=t_end38 + i_ * 0.1 + 0.05), ctx37)
    t_over38 = t_end38 + (MAG38 - 1) * 0.1 + 0.05
    check("(premise) the right magazine emptying ended the round", h38.round_over)
    _pull38(h38, co38, t_over38 + 1.0, camera=False)
    _settle38(h38, co38, t_over38 + 3.0)
    check("two players, round over: a heard-only click now counts (every magazine is empty)",
          h38.dry_mic_only == 1 and _dry_total38(h38) == 2)

    # -- a pull at the START box is a dry pull in BOTH numbers (they agree with their names) ----
    for label38, pl38 in (("single player", 1), ("two players", 2)):
        h38, co38 = _dry_host38(players=pl38), _Co38()
        t_last38 = _empty38(h38)
        bx38 = h38.round_box(ctx37).center()
        h38.shot(modes.ShotEvent(float(bx38.x()), float(bx38.y()), t_last38 + 2.0, 800, t_end=t_last38 + 2.05), ctx37)
        check(f"{label38}: a shot at the START box counts as a dry pull AND in after_done (one side, agreeing with its name)",
              h38.dry_camera == 1 and sum(t_.after_done for t_ in h38.tallies) == 1, f"{h38.dry_camera} / {[t_.after_done for t_ in h38.tallies]}")
        check(f"{label38}: ...and the host recognises that detection as dry for the app's wording, even after the next round starts",
              h38.is_dry_shot(t_last38 + 2.05) and (h38.restart(ctx37), h38.is_dry_shot(t_last38 + 2.05))[1]
              and not h38.is_dry_shot(1234.5))

    # -- NOT SCORED -------------------------------------------------------------------------------------
    h38, co38 = _dry_host38(), _Co38()
    t_last38 = _empty38(h38)
    snap_a = (h38.tallies[0].points, h38.tallies[0].hits, h38.tallies[0].shots,
              h38.verdict, h38.totals[0].points, h38.verdict_lines(), h38.side_block(0))
    for k_ in range(4):
        _pull38(h38, co38, t_last38 + 2.0 + k_)
    _pull38(h38, co38, t_last38 + 9.0, camera=False)
    _settle38(h38, co38, t_last38 + 12.0)
    snap_b = (h38.tallies[0].points, h38.tallies[0].hits, h38.tallies[0].shots,
              h38.verdict, h38.totals[0].points, h38.verdict_lines(), h38.side_block(0))
    check("dry pulls change NO score, hit count, shot count, verdict or end-screen result (NOT scored: his decision, deferred)",
          snap_a == snap_b and _dry_total38(h38) == 5, f"{snap_a} vs {snap_b}")

    # -- on BOTH end screens, small, always, at a constant height --------------------------------------------
    def _bottom38(h_, now_):
        rec_, _img = _spy_paint36(h_, context(now=now_))
        lines_ = [r_ for r_ in rec_ if r_[0] and str(r_[0][0]).startswith("Dry pulls:")]
        return lines_

    for label38, mk38 in (("single player", lambda: _dry_host38(players=1)),
                          ("two players", lambda: _dry_host38(players=2))):
        h38, co38 = mk38(), _Co38()
        t_last38 = _empty38(h38)
        zero38 = _bottom38(h38, 50.0)
        _pull38(h38, co38, t_last38 + 2.0)
        _pull38(h38, co38, t_last38 + 3.0, click=False)
        _settle38(h38, co38, t_last38 + 5.0)
        some38 = _bottom38(h38, 51.0)
        # NOT DRAWN (his ruling, 2026-10-22: "no need for this text"), at zero or at two:
        # the painter no longer asks for it. The words are still BUILT, because the
        # log line below and the introspection view read the same counts.
        check(f"{label38}: the dry-pull line is NOT on the end screen, at zero or after two pulls, "
              f"yet the counts are intact and the words are still built",
              zero38 == [] and some38 == []
              and h38.dry_screen_text().startswith("Dry pulls: 2 ") and _dry_total38(h38) == 2,
              f"{zero38} / {some38}")
        h38.restart(ctx37)
        h38.advance(context(now=1e6))                # the pairing has settled: the line is written now
        ev38 = h38.take_events()
        dry38 = [e_ for e_ in ev38 if e_.startswith("dry pulls after the magazine was empty")]
        check(f"{label38}: the log gets ONE line for the round, with the channel breakdown, when the next round starts",
              len(dry38) == 1 and "2 = 1 corroborated" in dry38[0] and "1 camera-only" in dry38[0]
              and "0 microphone-only" in dry38[0] and "not scored" in dry38[0], str(dry38))
        check(f"{label38}: the tally starts at zero again in the next round",
              h38.dry_camera == h38.dry_corroborated == h38.dry_mic_only == 0 and not h38.round_over)
    hz38 = _dry_host38()
    _empty38(hz38)
    hz38.restart(ctx37)
    hz38.advance(context(now=1e6))
    check("a round that ended with NO dry pulls still writes its line (zero is a measurement)",
          any(e_.startswith("dry pulls after the magazine was empty: 0 = 0 corroborated") for e_ in hz38.take_events()))
    hn38 = _dry_host38()
    hn38.restart(ctx37)
    check("...and a round that never ended writes no dry line (there was no window)",
          not any(e_.startswith("dry pulls") for e_ in hn38.take_events()))

    # ------------------------------------------------------------------
    # 39. THE HOLD: his last hole is seen before the results appear (2026-10-20).
    # "I want to see the hole of my last shot for at least a second. As soon as
    # I fire the number 17, that hole is supposed to be there and you wait at
    # least a second until you display the end-of-round screen."
    # ------------------------------------------------------------------
    print()
    print("39. the hold: the last hole stays on its target for the hold, then the results (single and two players)")
    MAG39 = 3

    def _hh39(players=1):
        h_ = _Host()
        h_.set_magazine(MAG39)
        h_.set_mode(_Static(), context(now=9.0), players=players)
        h_.hold_s = REAL_HOLD_S                 # the product's value; the rest of this file runs with it off
        return h_

    def _fire39(h_, n_, t0, x_=100.0, now=None):
        out_ = None
        for i_ in range(n_):
            t_ = t0 + i_ * 0.1
            out_ = h_.shot(modes.ShotEvent(x_, 1000.0, t_, i_ + 1, t_end=t_ + 0.05),
                           context(now=t_ if now is None else now))
        return out_

    check("the hold is the named constant, 1.2 s: 'at least a second', plus the frame that paints the hole",
          REAL_HOLD_S == 1.2 and _Host.HOLD_S == 0.0 and _Host().hold_s == 0.0)
    h39 = _hh39()
    _fire39(h39, MAG39, 10.0)                       # the last shot is at t = 10.2
    check("the last round is fired: the magazine is EMPTY but the round is NOT over, and nothing was announced yet",
          h39.remaining(0) == 0 and not h39.round_over and h39.take_round_ended() is False)
    check("...and the DRILL IS STILL PAINTED (the hole is seen against the target it struck, not on black)",
          len(h39._live_sides(context(now=10.3))) == 1
          and _fh_full(h39.paint, context(now=10.3))[1] > 0.005)
    h39.advance(context(now=11.0))
    check("one second after the first shot and 0.8 s after the last, the results are NOT up",
          not h39.round_over and len(h39._live_sides(context(now=11.0))) == 1)
    h39.advance(context(now=11.39))
    check("at 1.19 s after the last shot's timestamp: still not up (a flat 1.0 s would have shown it by now)",
          not h39.round_over)
    out39 = _fire39(h39, 1, 10.6, now=10.6)         # a shot DURING the hold
    check("a shot during the hold is DRY: the pop instead of the gunshot, its hole drawn, unscored, counted in "
          "after_done and as a dry pull",
          out39.dry and out39.play_sound and out39.draw_hole and out39.unscored
          and h39.tallies[0].after_done == 1 and h39.dry_camera == 1 and not h39.round_over)
    h39.advance(context(now=11.41))
    check("shortly after the hold the results ARE up, and the arrival is announced ONCE (the app clears the holes then)",
          h39.round_over and h39.take_round_ended() is True and h39.take_round_ended() is False)
    check("...and the drill is no longer painted: the results stand on a clear screen",
          len(h39._live_sides(context(now=11.5))) == 0)
    # timed from the shot's OWN timestamp, never from the tick that handled it
    h39 = _hh39()
    _fire39(h39, MAG39, 10.0, now=11.1)             # every shot HANDLED late, 0.9-1.1 s after it happened
    h39.advance(context(now=11.39))
    early39 = h39.round_over
    h39.advance(context(now=11.5))
    check("a late tick cannot change the hold: handled 0.9 s late, it still expires at the shot's timestamp + 1.2 s "
          "(not 1.2 s after the handling), and not a moment sooner",
          not early39 and h39.round_over)
    h39 = _hh39()
    _fire39(h39, MAG39, 10.0)
    h39.advance(context(now=10.25))
    h39.advance(context(now=10.9))
    check("...and an early tick does not close it (the clock is compared, not the number of ticks)",
          not h39.round_over)
    h39.advance(context(now=40.0))
    check("a very late tick closes it, once", h39.round_over and h39.take_round_ended() is True)

    # two players: a side that empties holds, then goes quiet; the other keeps shooting throughout
    h39 = _hh39(2)
    _fire39(h39, MAG39, 10.0, x_=100.0)             # LEFT empties; its last shot is at 10.2 -> hold to 11.4
    live_now = [i_ for i_, _m, _c in h39._live_sides(context(now=10.5))]
    check("two players: the LEFT side has emptied and is STILL PAINTED during its hold, the right is live",
          live_now == [0, 1] and h39._side_empty(0) and not h39._side_quiet(0, 10.5))
    o_l = _fire39(h39, 1, 10.7, x_=100.0, now=10.7)
    check("a left shot during its hold is dry (the pop, its hole, counted) while the right side is untouched",
          o_l.dry and o_l.play_sound and o_l.draw_hole and h39.tallies[0].after_done == 1
          and h39.tallies[1].shots == 0)
    h39.advance(context(now=11.5))
    live_later = [i_ for i_, _m, _c in h39._live_sides(context(now=11.5))]
    check("...after the hold the LEFT half goes quiet (as it always did) and the right keeps shooting",
          live_later == [1] and h39._side_done(0) and not h39.round_over)
    r_out = _fire39(h39, 2, 12.0, x_=3700.0, now=12.0)
    check("the right side shot throughout: live, scored by its drill, spending its own magazine",
          not r_out.dry and h39.tallies[1].shots == 2 and r_out.play_sound is not False)
    _fire39(h39, 1, 13.0, x_=3700.0, now=13.0)       # RIGHT empties at 13.0 -> hold to 14.2
    h39.advance(context(now=13.5))
    check("both magazines are empty but the RIGHT hold is still running: the round is NOT closed",
          h39.tallies[0].shots == MAG39 and h39.tallies[1].shots == MAG39 and not h39.round_over
          and [i_ for i_, _m, _c in h39._live_sides(context(now=13.5))] == [1])
    h39.advance(context(now=14.19))
    check("...not yet 0.01 s before the right hold expires", not h39.round_over)
    h39.advance(context(now=14.21))
    check("the round closes only once BOTH sides have emptied AND BOTH holds have run",
          h39.round_over and h39.verdict is not None and h39.take_round_ended() is True)
    h39b = _hh39(2)
    _fire39(h39b, MAG39, 10.0, x_=3700.0)
    _fire39(h39b, MAG39, 20.0, x_=100.0)
    h39b.advance(context(now=20.5))
    check("(the other order) the LEFT empties last: nothing closes at the right's expiry, only at the left's",
          not h39b.round_over)
    h39b.advance(context(now=21.5))
    check("...and it closes after the LEFT hold", h39b.round_over)

    # the dry-pull summary waits for the last pairing, and credits it to the round it belongs to
    h39 = _hh39()
    h39.mic_ready = True
    _fire39(h39, MAG39, 10.0)
    h39.advance(context(now=12.0))
    bx39 = h39.round_box(context(now=12.0)).center()
    h39.advance(context(now=12.0))
    _r39, _l39, _a39 = _fh_full(h39.paint, context(now=12.0))      # the box is painted: armed 0.8 s later
    h39.shot(modes.ShotEvent(float(bx39.x()), float(bx39.y()), 13.0, 50, t_end=13.05), context(now=13.0))
    h39.restart(context(now=13.0))                                   # the box shot starts the next round
    check("the line about the dry pulls is NOT written the instant the round resets (its last pairing has not arrived)",
          not any(e_.startswith("dry pulls") for e_ in h39.take_events()))
    h39.note_pull_outcomes([("both", 13.05)])                        # the click is paired a moment later
    h39.advance(context(now=14.5))
    ev39 = [e_ for e_ in h39.take_events() if e_.startswith("dry pulls")]
    check("...it is written once the pairing has settled, and the late pairing is credited to the round it belongs to "
          "(1 corroborated, 0 camera-only), marked settled",
          len(ev39) == 1 and "1 = 1 corroborated" in ev39[0] and "0 camera-only" in ev39[0]
          and ev39[0].endswith("settled"), str(ev39))
    check("...and the new round's own tally was not touched by it",
          h39.dry_camera == 0 and h39.dry_corroborated == 0)

    # session: shots == hits + misses + unplaced, dry pulls apart (the identity the display and /state rely on)
    from lsa.display.session import SessionStats as _SS39
    ss39 = _SS39()
    ss39.register_shot(0, 0, scored=True, score=10)
    ss39.register_shot(0, 0, scored=False)
    ss39.register_shot(0, 0, scored=None)
    ss39.register_dry()
    ss39.register_dry()
    check("SessionStats: shots == hits + misses + unplaced, and a dry pull moves nothing else (not shots, not timing)",
          (ss39.shots, ss39.hits, ss39.misses, ss39.unplaced, ss39.dry_pulls) == (3, 1, 1, 1, 2) and ss39.consistent()
          and len(ss39.history) == 3 and ss39.string_time_s >= 0.0)
    t_before = (ss39.first_shot, ss39.last_shot, len(ss39.splits_ms))
    ss39.register_dry()
    check("...a dry pull leaves the string time's first and last shot and the splits untouched",
          (ss39.first_shot, ss39.last_shot, len(ss39.splits_ms)) == t_before)

    # the single-player results block: his order, derived values, constant height
    h39 = _hh39()
    h39.hold_s = 0.0
    for i_, (hit_, t_) in enumerate(((True, 100.0), (True, 101.5), (False, 124.6))):
        x_, y_ = h39.mode.aim_point(context(now=t_)) if hit_ else (100.0, 1000.0)
        h39.shot(modes.ShotEvent(x_, y_, t_, i_ + 1, t_end=t_ + 0.05), context(now=t_))
    # The points line is "N of M points" (2026-10-22). N is pinned to the figure this round
    # earned (two hits at 10) AND to the host's own tally, so the line is still read from
    # what the host counted; M is matched as a positive whole number (the ghost's ceiling,
    # not a property of this block). The misses are likewise compared with the host's own
    # `round_misses`, so a block that recomputed them from the shot count would disagree
    # the moment the two parted.
    import re as _re39b
    blk39 = h39.side_block(0)
    pm39 = _re39b.fullmatch(r"(\d+) of (\d+) points", blk39[3]) if len(blk39) == 5 else None
    check("the results block reads shots / hits / misses / points / seconds, in his order, from the host's own tallies "
          "(first to last LIVE shot: 24.6 s)",
          h39.round_over and len(blk39) == 5
          and blk39[:3] == ["3 shots", "2 hits", "1 miss"] and blk39[4] == "24.6 seconds"
          and pm39 is not None and int(pm39.group(1)) == 20 and int(pm39.group(2)) > 0
          and int(pm39.group(1)) == h39.tallies[0].points
          and blk39[2] == f"{h39.round_misses(0)} miss"
          and h39.verdict_lines() == ["ROUND 1"], str(blk39))
    heights39 = []
    for rnd_ in range(1, 5):
        h39.restart(context(now=200.0 + rnd_))
        for i_ in range(MAG39):
            tt_ = 300.0 + rnd_ * 10 + i_
            h39.shot(modes.ShotEvent(100.0, 1000.0, tt_, i_ + 1), context(now=tt_))
        rec_, _img_ = _spy_paint36(h39, context(now=500.0 + rnd_))
        heights39.append([(b_[2], b_[1]) for ls_, _s, b_ in rec_ if ls_ and ls_[0] in ("ROUND %d" % h39.round_no,)]
                         + [len(ls_) for ls_, _s, _b in rec_ if len(ls_) == 5])
    check("the results block is the SAME shape at rounds 1, 2, 3 and 4 (a fixed six-line block: five lines + the heading)",
          all(h_ == heights39[0] for h_ in heights39) and heights39[0], str(heights39))
    # ------------------------------------------------------------------
    # 40. THE SELECTION CUE and the DRY-PULL SOUND (2026-10-20)
    # ------------------------------------------------------------------
    print()
    print("40. the selection cue (MENU button, NEXT ROUND box, every menu tile) and the dry-pull pop")
    from lsa.audio import mixer as _mx40
    from lsa.modes.menu import Menu as _Menu40

    check("both samples are registered RESERVED, hidden from the gunshot picker, under their own names",
          _mx40.SELECT_SOUND.startswith("reserved/") and _mx40.DRY_SOUND.startswith("reserved/")
          and _mx40.SELECT_SOUND != _mx40.DRY_SOUND and _mx40.DRY_SOUND != _mx40.VANISH_SOUND
          and "universfield-bubble-pop-06-351337" in _mx40.Mixer.RESERVED_FILES
          and "freesound_community-080997_bullet-39735" in _mx40.Mixer.RESERVED_FILES
          and _mx40.Mixer.RESERVED_FILES["universfield-bubble-pop-06-351337"] == _mx40.DRY_SOUND
          and _mx40.Mixer.RESERVED_FILES["freesound_community-080997_bullet-39735"] == _mx40.SELECT_SOUND)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from silent_audio import silent_mixer as _sm40
    mx40 = _sm40()
    from asset_env import res as _res40            # the asset table, not ROOT / "sounds"
    mx40.load_folder(_res40("sounds"))
    check("...loaded from the real sounds/ folder and absent from names() (the picker): neither can ever be drawn as THE gunshot",
          mx40.has(_mx40.SELECT_SOUND) and mx40.has(_mx40.DRY_SOUND)
          and not any("reserved" in n_ or "bubble pop 06" in n_ or "bullet" == n_.split()[-1] and "080997" in n_
                      for n_ in mx40.names()),
          str(mx40.names()))
    print(f"         dry pop {mx40.duration(_mx40.DRY_SOUND):.3f} s, select cue {mx40.duration(_mx40.SELECT_SOUND):.3f} s, "
          f"ring-target vanish {mx40.duration(_mx40.VANISH_SOUND):.3f} s")

    h40 = SP37()
    h40.hold_s = 0.0
    sc40 = h40.menu_button.rect(ctx37).center()
    o1 = h40.shot(modes.ShotEvent(float(sc40.x()), float(sc40.y()), 5.0, 1), ctx37)
    o2 = h40.shot(modes.ShotEvent(float(sc40.x()), float(sc40.y()), 5.5, 2), ctx37)
    check("a SHOT at the MENU button carries the selection cue on EACH of its two hits (the first arms it and says so)",
          o1.cue == _mx40.SELECT_SOUND and o2.cue == _mx40.SELECT_SOUND and o1.play_sound and o2.play_sound)
    aim40 = h40.mode.aim_point(ctx37)
    o3 = h40.shot(modes.ShotEvent(aim40[0], aim40[1], 6.0, 3), ctx37)
    o4 = h40.shot(modes.ShotEvent(1920.0, 1500.0, 6.5, 4), ctx37)
    check("a shot at a DRILL TARGET (hit or miss) never carries it", o3.cue is None and o4.cue is None and o3.hit)
    h40.take_sounds()
    h40.on_click(float(sc40.x()), float(sc40.y()), ctx37)
    check("a CLICK on the MENU button queues the cue alone (no gunshot exists for a click), once",
          h40.take_sounds() == [_mx40.SELECT_SOUND] and h40.take_sounds() == [])
    tl40 = (h40.tallies[0].shots, h40.tallies[0].hits, h40.tallies[0].points, h40.remaining(0))
    h40.on_click(float(sc40.x()), float(sc40.y()), ctx37)
    check("...and a click moves no score, hit rate or magazine",
          (h40.tallies[0].shots, h40.tallies[0].hits, h40.tallies[0].points, h40.remaining(0)) == tl40)

    # the round-over screen
    h40 = SP37()
    h40.hold_s = 0.0
    _shoot37(h40, N37)
    _fh_full(h40.paint, context(now=50.0))
    bx40 = h40.round_box(ctx37).center()
    ob = h40.shot(modes.ShotEvent(float(bx40.x()), float(bx40.y()), 50.2, 90), ctx37)
    check("a shot at the NEXT ROUND box before it arms: the pop and its hole, and NO cue (it did nothing to confirm)",
          ob.dry and ob.play_sound and ob.draw_hole and ob.cue is None and not h40.take_next_round_request())
    ob = h40.shot(modes.ShotEvent(float(bx40.x()), float(bx40.y()), 51.0, 91), ctx37)
    check("once armed, it starts the next round, with the pop, the hole AND the selection cue after it",
          ob.dry and ob.play_sound and ob.draw_hole and ob.cue == _mx40.SELECT_SOUND
          and h40.take_next_round_request() is True)
    oe = h40.shot(modes.ShotEvent(300.0, 900.0, 52.0, 92), ctx37)
    check("a shot elsewhere on the results screen: the pop and its hole, no cue",
          oe.dry and oe.play_sound and oe.draw_hole and oe.cue is None)
    h40.take_sounds()
    h40.on_click(float(bx40.x()), float(bx40.y()), ctx37)
    check("a CLICK on the NEXT ROUND box plays the cue alone", h40.take_sounds() == [_mx40.SELECT_SOUND])
    h40.restart(ctx37)
    h40.take_sounds()
    h40.mode.shot(modes.ShotEvent(1.0, 1.0, 60.0, 1), ctx37)
    check("(the N key is not one of the cue's places: it queues nothing)", h40.take_sounds() == [])

    # the menu screen
    hm40 = _Host()
    hm40.set_mode(_Menu40(), ctx37, players=1)
    mm40 = hm40.mode
    slots40 = mm40.slots(ctx37)
    kinds40 = [("a drill tile", next(s_ for s_ in slots40 if s_.cls is not None)),
               ("One Player", next(s_ for s_ in slots40 if s_.players == 1)),
               ("Two Players", next(s_ for s_ in slots40 if s_.players == 2)),
               ("EXIT", next(s_ for s_ in slots40 if s_.stem == "exit"))]
    for label_, slot_ in kinds40:
        c_ = slot_.rect.center()
        mm40.pending = None
        mm40.pending_players = None
        out_ = hm40.shot(modes.ShotEvent(float(c_.x()), float(c_.y()), 70.0, 1), ctx37)
        check(f"a shot at {label_} on the menu carries the selection cue (and the gunshot, which it follows)",
              out_.cue == _mx40.SELECT_SOUND and out_.play_sound and not out_.dry, f"{out_}")
    ex40 = kinds40[3][1].rect.center()
    mm40.exit_armed_at = 0.0
    o_a = hm40.shot(modes.ShotEvent(float(ex40.x()), float(ex40.y()), 80.0, 1), ctx37)
    o_b = hm40.shot(modes.ShotEvent(float(ex40.x()), float(ex40.y()), 80.5, 2), ctx37)
    check("EXIT plays the cue on BOTH of its two hits", o_a.cue == _mx40.SELECT_SOUND and o_b.cue == _mx40.SELECT_SOUND
          and mm40.quit_requested)
    hm40.take_sounds()
    hm40.on_click(float(kinds40[0][1].rect.center().x()), float(kinds40[0][1].rect.center().y()), ctx37)
    check("a CLICK on a menu tile plays the cue alone", hm40.take_sounds() == [_mx40.SELECT_SOUND])
    hm40.on_click(1.0, 1.0, ctx37)
    check("a click between the tiles plays nothing", hm40.take_sounds() == [])
    o_miss = hm40.shot(modes.ShotEvent(1.0, 1.0, 90.0, 9), ctx37)
    check("a shot between the tiles carries no cue", o_miss.cue is None)

    # the dry pull: the pop instead of the gunshot, the hole drawn, scoring nothing
    h40 = SP37()
    h40.hold_s = REAL_HOLD_S
    _fire39(h40, N37, 10.0, now=10.0) if False else None
    for i_ in range(N37):
        t_ = 10.0 + i_ * 0.1
        h40.shot(modes.ShotEvent(100.0, 1000.0, t_, i_ + 1, t_end=t_ + 0.05), context(now=t_))
    snap40 = (h40.tallies[0].points, h40.tallies[0].hits, h40.tallies[0].shots, h40.remaining(0))
    od = h40.shot(modes.ShotEvent(100.0, 1000.0, 11.0, 99), context(now=11.0))
    check("a shot on an EMPTY magazine during a drill: dry, plays a sound (the pop, chosen by the app), draws its hole",
          od.dry and od.play_sound and od.draw_hole and od.cue is None and od.sound is None and od.unscored)
    check("...it scores nothing, moves no hit rate, no magazine count, and is counted in after_done and as a dry pull",
          (h40.tallies[0].points, h40.tallies[0].hits, h40.tallies[0].shots, h40.remaining(0)) == snap40
          and h40.tallies[0].after_done == 1 and h40.dry_camera == 1)
    ol = SP37().shot(modes.ShotEvent(100.0, 1000.0, 1.0, 1), ctx37)
    check("...while the same shot with rounds left is NOT dry (the gunshot)", not ol.dry and ol.play_sound and ol.cue is None)
    # ------------------------------------------------------------------
    # 41. THE STATIC TARGET: numbers drawn from the scoring arithmetic, and shaded bands that
    # cost his laser almost nothing (2026-10-20).
    # ------------------------------------------------------------------
    print()
    print("41. the static target: ring numbers derived from the scorer, alternate bands, inside the budget")
    import dataclasses as _dc41
    from PySide6.QtGui import QFontMetricsF as _QFM41
    from lsa.display import layers as _L41
    from lsa.artwork import load as _load41

    def _field_ctx41(half=False):
        c_ = context(now=1.0)
        if half:
            c_ = _dc41.replace(c_, rect=QRect(0, 0, (SW - 5) // 2, SH), screen_width=SW)
        return c_

    for half41 in (False, True):
        c41 = _field_ctx41(half41)
        f41 = _L41.AimField()
        marks41 = f41.marks(c41)
        nums41 = f41.ring_numbers(c41)
        label41 = "a two-player half" if half41 else "the whole screen"
        bad41 = [(t_, x_, y_) for t_, x_, y_, _sz in nums41
                 if _L41.score_against_marks(x_, y_, marks41)[0] != int(t_)]
        check(f"{label41}: EVERY drawn number equals what the REAL scorer awards for a shot at that number "
              f"({len(nums41)} numbers)", nums41 and not bad41, str(bad41[:3]))
        check(f"{label41}: the numbers are only on the CENTRE mark, and only the four directions a real target prints them",
              all(abs(x_ - marks41[0][0]) < 1e-6 or abs(y_ - marks41[0][1]) < 1e-6 for _t, x_, y_, _sz in nums41))
        digs41 = {t_ for t_, *_ in nums41}
        want41 = {str(_L41.ring_points(len(marks41[0][2]), k_)) for k_ in range(len(marks41[0][2]))}
        if half41:
            check(f"{label41}: the 10 is DROPPED here (its digit would be smaller than the legibility floor) and 9, 8, 7 stay",
                  digs41 == {"9", "8", "7"} and "10" in want41, str(sorted(digs41)))
        else:
            check(f"{label41}: all four rings are numbered 10, 9, 8, 7 (derived, not typed)", digs41 == want41 == {"10", "9", "8", "7"},
                  str(sorted(digs41)))
        hs41 = [_QFM41(_L41.mono(sz_, True)).boundingRect("8").height() for _t, _x, _y, sz_ in nums41]
        print(f"         {label41}: digit heights {min(hs41):.0f}-{max(hs41):.0f} px of {SH} ({100 * min(hs41) / SH:.2f}-{100 * max(hs41) / SH:.2f}% of the screen height)")
        check(f"{label41}: no digit is smaller than the legibility floor ({_L41.MIN_DIGIT_H_FRAC * 100:.1f}% of the screen height)",
              min(hs41) >= _L41.MIN_DIGIT_H_FRAC * SH - 0.5)
        cx41, cy41 = marks41[0][0], marks41[0][1]
        check(f"{label41}: the AIM POINT is never inked: no number within the crosshair, and the exact centre pixel "
              f"is bare background",
              all(((x_ - cx41) ** 2 + (y_ - cy41) ** 2) ** 0.5 > SH * 0.016 * c41.field_share + 4 for _t, x_, y_, _sz in nums41))
        if half41:
            check(f"{label41}: every number is inside its own half",
                  all(c41.rect.left() <= x_ <= c41.rect.right() for _t, x_, _y, _sz in nums41))

    # the scoring arithmetic and the drawn numbers cannot disagree: change one and the other follows
    c41 = _field_ctx41()
    f41 = _L41.AimField()
    real_rp41 = _L41.ring_points
    try:
        _L41.ring_points = lambda n_, k_: n_ - k_ + 7
        nums_shift41 = f41.ring_numbers(c41)
        marks_shift41 = f41.marks(c41)
        ok_shift41 = all(_L41.score_against_marks(x_, y_, marks_shift41)[0] == int(t_) for t_, x_, y_, _sz in nums_shift41)
        shifted41 = sorted({t_ for t_, *_ in nums_shift41})
    finally:
        _L41.ring_points = real_rp41
    check("a change to the scoring arithmetic CHANGES the drawn numbers (11, 10, 9, 8), and the scorer still agrees with them",
          shifted41 == ["10", "11", "8", "9"] and ok_shift41, str(shifted41))

    # the shade: a stated maximum, rendered luminance, lit area, red excess
    check("the band shade is at or below its stated MAXIMUM step, and that maximum is the stated 10 units "
          "(a later 'make it clearer' cannot quietly cross it)",
          _L41.BAND_SHADE_STEP <= _L41.BAND_SHADE_MAX_STEP == 10 and _load41.BACKGROUND + _L41.BAND_SHADE_STEP <= 60)
    class _NoBands41:
        pass
    r_on, lit_on, a_on = _fh_full(lambda p_, c_: f41.paint(p_, c_), c41)
    real_rings41 = _L41.SHADED_RINGS
    real_nums41 = _L41.AimField.ring_numbers
    try:
        _L41.SHADED_RINGS = ()
        _L41.AimField.ring_numbers = lambda self, c_: []
        r_off, lit_off, a_off = _fh_full(lambda p_, c_: f41.paint(p_, c_), c41)
        _L41.AimField.ring_numbers = real_nums41
        _L41.SHADED_RINGS = real_rings41
        _L41.AimField.ring_numbers = lambda self, c_: []
        r_b, lit_b, a_b = _fh_full(lambda p_, c_: f41.paint(p_, c_), c41)
    finally:
        _L41.SHADED_RINGS = real_rings41
        _L41.AimField.ring_numbers = real_nums41
    print(f"         field lit above luminance 60: today {lit_off * 100:.4f}%, with bands {lit_b * 100:.4f}%, "
          f"with bands and numbers {lit_on * 100:.4f}% (ceiling 2%)")
    check("the BANDS cost no lit area above luminance 60 (the antialiased edges where a stroke meets a band move it by "
          "less than 0.01 percentage points) and the numbers add none",
          lit_b - lit_off <= 0.0001 and lit_on - lit_b <= 1e-9 and lit_on < 0.02,
          f"{lit_off:.6f} {lit_b:.6f} {lit_on:.6f}")
    check("bands and numbers: red excess exactly 0 on every rendered pixel", r_on == 0 and r_b == 0 and r_off == 0)
    m41 = f41.marks(c41)[0]
    band9 = (m41[0] + (m41[2][0] + m41[2][1]) / 2.0, m41[1] - 0.0)           # middle of the 9 band, on the horizontal axis...
    # ...sampled OFF the axis so no digit is under the sample
    off9 = (int(m41[0] + (m41[2][0] + m41[2][1]) / 2.0 * 0.7071), int(m41[1] + (m41[2][0] + m41[2][1]) / 2.0 * 0.7071))
    off8 = (int(m41[0] + (m41[2][1] + m41[2][2]) / 2.0 * 0.7071), int(m41[1] + (m41[2][1] + m41[2][2]) / 2.0 * 0.7071))
    off7 = (int(m41[0] + (m41[2][2] + m41[2][3]) / 2.0 * 0.7071), int(m41[1] + (m41[2][2] + m41[2][3]) / 2.0 * 0.7071))
    lum41 = lambda a_, pt_: int(a_[pt_[1], pt_[0], :3].max())
    print(f"         rendered luminance: 9 band {lum41(a_on, off9)}, 8 band {lum41(a_on, off8)}, 7 band {lum41(a_on, off7)}, "
          f"background {lum41(a_on, (5, 5))}; a landing dot loses {lum41(a_on, off9) - lum41(a_on, (5, 5))} units on a shaded band")
    check("ALTERNATE rings are shaded: the 9 and the 7 bands at the stated shade, the 10 (the aim disc) and the 8 are bare",
          lum41(a_on, off9) == lum41(a_on, off7) == _load41.BACKGROUND + _L41.BAND_SHADE_STEP
          and lum41(a_on, off8) == lum41(a_on, (5, 5)) == _load41.BACKGROUND
          and a_on[int(m41[1]) + 30, int(m41[0]) + 20, :3].max() == _load41.BACKGROUND)
    corner41 = f41.marks(c41)[1]
    cr41 = corner41[2]
    pts41 = [(int(corner41[0] + (cr41[0] + cr41[1]) / 2.0 * 0.7071), int(corner41[1] + (cr41[0] + cr41[1]) / 2.0 * 0.7071)),
             (int(corner41[0] + (cr41[1] + cr41[2]) / 2.0 * 0.7071), int(corner41[1] + (cr41[1] + cr41[2]) / 2.0 * 0.7071))]
    check("the CORNER marks are NOT shaded and carry no numbers: they are where his far-side shots land (peaks 12-15 against a bar of 9)",
          all(lum41(a_on, q_) == _load41.BACKGROUND for q_ in pts41)
          and all(abs(x_ - corner41[0]) > 1 for _t, x_, _y, _sz in f41.ring_numbers(c41) if abs(_y - corner41[1]) < 1))
    # what a landing dot loses, stated as numbers
    loss_band41 = lum41(a_on, off9) - lum41(a_on, (5, 5))
    print(f"         a dot landing on a shaded band loses {loss_band41} units of red-minus-green/blue score; on bare ground 0; "
          f"his measured far-side shots were 12 and 15 against a bar of 9, so at the corners (unshaded) they lose nothing, "
          f"and a {loss_band41}-unit loss would have taken the 12 to {12 - loss_band41} (under the bar)")
    check("a landing dot loses exactly the shade step, and no more than the stated maximum",
          loss_band41 == _L41.BAND_SHADE_STEP <= _L41.BAND_SHADE_MAX_STEP)
    # both screens: the full host paints the field with the numbers, offscreen, in the SAME pass the modes use
    h41 = _Host()
    h41.set_mode(_Static(), context(now=1.0), players=1)
    r_h, lit_h, _a = _fh_full(h41.paint, context(now=1.0))
    h41b = _Host()
    h41b.set_mode(_Static(), context(now=1.0), players=2)
    r_hb, lit_hb, _ab = _fh_full(h41b.paint, context(now=1.0))
    print(f"         whole Static Targets screen: single {lit_h * 100:.3f}%, two players {lit_hb * 100:.3f}% lit above 60 (ceiling 2%)")
    check("the whole Static Targets screen (single and two players) stays inside the 2% ceiling with red excess 0",
          lit_h < 0.02 and lit_hb < 0.02 and r_h == 0 and r_hb == 0)

    # ------------------------------------------------------------------
    # 42. THE ANIMATED SOLDIER: four hits in the HEAD or CENTRE MASS take him down (2026-10-20)
    # ------------------------------------------------------------------
    print()
    print("42. the soldier: four head or centre-mass hits drop him; limbs score and do not count; zones come from the frame at ev.t")
    from lsa.modes.animated import AnimatedFigure as _AF42
    from lsa.artwork import sequence as _sq42
    import numpy as _np42

    def _sol42():
        m_ = _AF42()
        m_.start(context(now=0.0))
        return m_

    def _t_of42(m_, f_):
        return m_._t0 + (f_ + 0.5) / m_.FPS

    def _pt42(m_, f_, which):
        """A screen point in frame f_'s head / mass / body-not-in-a-zone cell."""
        seq_ = m_.sequence()
        head_, mass_ = seq_.zones(f_)
        sil_ = seq_.silhouette(f_)
        mask_ = {"head": head_, "mass": mass_, "limb": (sil_ > 0) & (head_ == 0) & (mass_ == 0)}[which]
        ys_, xs_ = _np42.nonzero(mask_)
        k_ = len(ys_) // 2
        order_ = _np42.lexsort((xs_, ys_))
        r_, c_ = int(ys_[order_][k_]), int(xs_[order_][k_])
        left_, top_, w_, h_ = seq_.box(f_, m_._origin, m_._scale)
        return (left_ + (c_ + 0.5) / sil_.shape[1] * w_, top_ + (r_ + 0.5) / sil_.shape[0] * h_)

    FR42 = 46                                   # a tall, upright frame
    check("the drop threshold is the named constant, 4 (it was 10), and the counter reads it",
          _AF42.HITS_TO_DROP == 4)
    m42 = _sol42()
    for k_ in range(3):
        o_ = m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "mass"), _t_of42(m42, FR42), k_ + 1), context(now=_t_of42(m42, FR42)))
    check("three centre-mass hits do NOT drop him; each scores and the counter reads 3",
          m42._state == "walk" and m42._hits == 3 and o_.hit and o_.points > 0 and not o_.sound)
    o_ = m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "mass"), _t_of42(m42, FR42), 4), context(now=_t_of42(m42, FR42)))
    check("the FOURTH centre-mass hit drops him, with a grunt (the soldier's fall)",
          m42._state == "fall" and o_.target_id.endswith(":down") and o_.sound is not None)
    m42 = _sol42()
    oh_ = m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "head"), _t_of42(m42, FR42), 1), context(now=_t_of42(m42, FR42)))
    om_ = m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "mass"), _t_of42(m42, FR42), 2), context(now=_t_of42(m42, FR42)))
    check("a HEAD hit advances the counter exactly as a centre-mass hit does (both score, both count one)",
          oh_.hit and om_.hit and oh_.points == om_.points and m42._hits == 2
          and oh_.target_id.endswith(":head") and om_.target_id.endswith(":mass"))
    m42 = _sol42()
    for k_, zn_ in enumerate(("head", "mass", "head", "mass")):
        m42.shot(modes.ShotEvent(*_pt42(m42, FR42, zn_), _t_of42(m42, FR42), k_ + 1), context(now=_t_of42(m42, FR42)))
    check("four hits in any MIXTURE of the two regions drop him", m42._state == "fall")
    m42 = _sol42()
    ol_ = m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "limb"), _t_of42(m42, FR42), 1), context(now=_t_of42(m42, FR42)))
    ol_ = m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "limb"), _t_of42(m42, FR42), 2), context(now=_t_of42(m42, FR42)))
    check("a hit on the silhouette OUTSIDE both regions (a limb) still SCORES and does NOT advance the counter",
          ol_.hit and ol_.points > 0 and m42._hits == 0 and m42._body_hits == 2 and ol_.target_id.endswith(":body")
          and m42._state == "walk")
    for k_ in range(10):
        m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "limb"), _t_of42(m42, FR42), 3 + k_), context(now=_t_of42(m42, FR42)))
    check("...however many limb hits there are, he never goes down", m42._state == "walk" and m42._hits == 0)
    omiss = m42.shot(modes.ShotEvent(5.0, 5.0, _t_of42(m42, FR42), 99), context(now=_t_of42(m42, FR42)))
    check("a clean miss scores nothing", not omiss.hit and omiss.points == 0)

    # the zones: derived, disjoint, per frame, from the silhouette only
    seq42 = m42.sequence()
    overlap42 = [i_ for i_ in range(len(seq42)) if (seq42.zones(i_)[0] & seq42.zones(i_)[1]).any()]
    check("the head and centre-mass regions NEVER overlap, in any frame of the clip: a hit counts once",
          overlap42 == [], str(overlap42[:5]))
    inside42 = all(((seq42.zones(i_)[0] > 0) <= (seq42.silhouette(i_) > 0)).all()
                   and ((seq42.zones(i_)[1] > 0) <= (seq42.silhouette(i_) > 0)).all() for i_ in range(0, len(seq42), 7))
    check("both regions are INSIDE the frame's silhouette (a shot through a gap is not a hit)", inside42)
    hc = {}
    for i_ in (0, 34, 46, 80):
        ys_, xs_ = _np42.nonzero(seq42.zones(i_)[0])
        sil_rows = _np42.nonzero(seq42.silhouette(i_))[0]
        hc[i_] = (float(ys_.mean()), float(xs_.mean()), (ys_.max() - ys_.min() + 1) / float(sil_rows.max() - sil_rows.min() + 1))
    check("the regions are DERIVED PER FRAME from the figure, not a fixed rectangle: the head region sits at a different "
          "place in different frames, and is a steady fraction of the figure's own height",
          len({round(v_[0]) for v_ in hc.values()}) > 1 or len({round(v_[1]) for v_ in hc.values()}) > 1
          and all(0.12 <= v_[2] <= 0.20 for v_ in hc.values()), str(hc))
    synth42 = _np42.zeros((100, 60), _np42.uint8)
    synth42[0:100, 25:35] = 1                       # a tall thin man
    synth42[20:50, 5:55] = 1                        # with wide arms at shoulder height
    h42, m42z = _sq42.zones_from_silhouette(synth42)
    check("zones_from_silhouette on a synthetic figure: head in the top band only, mass in the torso band only, both inside it",
          h42[:17].any() and not h42[17:].any() and m42z[19:52].any() and not m42z[:19].any() and not m42z[52:].any()
          and not (h42 & m42z).any() and ((h42 > 0) <= (synth42 > 0)).all())

    # the frame at ev.t, not the frame up now
    m42 = _sol42()
    fa, fb = 46, 34                                  # in frame 46 a point is in the head; in frame 34 the same point is not
    pa = _pt42(m42, fa, "head")
    in_b = m42.sequence().zone_at(fb, pa[0], pa[1], m42._origin, m42._scale)
    check("(premise) the same screen point is a head hit in one frame and not in another", in_b != "head")
    m42._frame = fb                                  # the frame that is on screen NOW
    o_ = m42.shot(modes.ShotEvent(pa[0], pa[1], _t_of42(m42, fa), 1), context(now=_t_of42(m42, fb)))
    check("centre mass and head are taken from the frame at ev.t (the shot's own timestamp), NOT the frame up when it was handled",
          o_.target_id.endswith(":head") and m42._hits == 1)
    pb = _pt42(m42, fb, "head")
    in_a = m42.sequence().zone_at(fa, pb[0], pb[1], m42._origin, m42._scale)
    m42b = _sol42()
    m42b._frame = fa
    o_ = m42b.shot(modes.ShotEvent(pb[0], pb[1], _t_of42(m42b, fb), 1), context(now=_t_of42(m42b, fa)))
    check("...and the other way round: a late handling cannot turn a hit into a miss or the reverse",
          (in_a != "head") and o_.target_id.endswith(":head"))

    # the counter, and the 'no count' feedback that lives on it
    from PySide6.QtGui import QImage as _QI42, QPainter as _QP42
    class _SpyP42(_QP42):
        texts: list = []

        def drawText(self, *a):
            if len(a) == 3 and isinstance(a[2], str):
                _SpyP42.texts.append(a[2])
            return super().drawText(*a)

    def _label42(m_, now_):
        """The counter's words (`counter_label`, the very string `paint` blits), and the paint itself must not crash."""
        img_ = _QI42(SW, SH, _QI42.Format_RGB32)
        img_.fill(0xFF0A0A0A)
        sp_ = _QP42(img_)
        m_.paint(sp_, context(now=now_))
        sp_.end()
        return m_.counter_label(now_)

    m42 = _sol42()
    t_fr = _t_of42(m42, FR42)
    m42._frame = FR42
    base_lbl = _label42(m42, t_fr)
    m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "mass"), t_fr, 1), context(now=t_fr))
    count_lbl = _label42(m42, t_fr + 0.01)
    m42.shot(modes.ShotEvent(*_pt42(m42, FR42, "limb"), t_fr + 0.2, 2), context(now=t_fr + 0.2))
    nocount_lbl = _label42(m42, t_fr + 0.25)
    later_lbl = _label42(m42, t_fr + 0.2 + _AF42.NOCOUNT_S + 0.5)
    m42.shot(modes.ShotEvent(5.0, 5.0, t_fr + 3.0, 3), context(now=t_fr + 3.0))
    miss_lbl = _label42(m42, t_fr + 3.01)
    check("the counter reads n / THRESHOLD, derived (not a typed 10 or 4)",
          base_lbl == f"0 / {_AF42.HITS_TO_DROP}" and count_lbl == f"1 / {_AF42.HITS_TO_DROP}", f"{base_lbl!r} {count_lbl!r}")
    check("a counting hit advances the digit and says nothing else; a limb hit adds 'HIT - NO COUNT' with the digit unchanged; "
          "a clean miss changes nothing: three different pictures, so a limb hit is not mistaken for a miss",
          count_lbl == "1 / 4" and "NO COUNT" in nocount_lbl and nocount_lbl.startswith("1 / 4")
          and later_lbl == "1 / 4" and miss_lbl == "1 / 4" and len({count_lbl, nocount_lbl}) == 2)
    real_thr = _AF42.HITS_TO_DROP
    try:
        _AF42.HITS_TO_DROP = 6
        mt42 = _sol42()
        mt42._frame = FR42
        lbl6 = _label42(mt42, _t_of42(mt42, FR42))
        for k_ in range(5):
            mt42.shot(modes.ShotEvent(*_pt42(mt42, FR42, "mass"), _t_of42(mt42, FR42), k_ + 1), context(now=_t_of42(mt42, FR42)))
        five_ok = mt42._state == "walk"
        mt42.shot(modes.ShotEvent(*_pt42(mt42, FR42, "mass"), _t_of42(mt42, FR42), 6), context(now=_t_of42(mt42, FR42)))
    finally:
        _AF42.HITS_TO_DROP = real_thr
    check("the counter AND the drop follow the one constant: set to 6, it reads '0 / 6' and drops on the sixth",
          lbl6 == "0 / 6" and five_ok and mt42._state == "fall", lbl6)

    # a 17-round magazine: he goes down four times, with a round to spare
    host42 = _Host()
    host42.set_magazine(17)
    host42.set_mode(_AF42(), context(now=0.0), players=1)
    host42.hold_s = 0.0
    sol = host42.mode
    t42 = sol._t0 + 1.0 / sol.FPS
    drops42 = 0
    shots42 = 0
    for rnd in range(4):
        for k_ in range(4):
            f_ = sol.frame_at(t42)
            xy_ = _pt42(sol, f_, "mass")
            host42.shot(modes.ShotEvent(xy_[0], xy_[1], t42, shots42 + 1), context(now=t42))
            shots42 += 1
            t42 += 0.1
        drops42 += 1 if sol._state == "fall" else 0
        # let the fall clip and the rest play out in virtual time, then he is back
        for _ in range(400):
            sol.advance(context(now=t42))
            t42 += 0.05
            if sol._state == "walk":
                break
    check("with a 17-round magazine he goes down FOUR times (the counter resets after each drop) and one round is left over",
          drops42 == 4 and sol._dropped == 4 and host42.remaining(0) == 1, f"drops {drops42}, dropped {sol._dropped}, left {host42.remaining(0)}")

    # sizes, measured: where the regions are and what they are worth against the position uncertainty
    seq42 = _sol42().sequence()
    mpx = 42.5 * 25.4 / ((3840 ** 2 + 2160 ** 2) ** 0.5)
    for lbl_, f_ in (("nearest/tallest frame", max(range(len(seq42)), key=lambda i_: seq42.frames[i_]["h"])),
                     ("furthest/most crouched frame", min(range(len(seq42)), key=lambda i_: seq42.frames[i_]["h"]))):
        mm_ = _sol42()
        left_, top_, w_, h_ = seq42.box(f_, mm_._origin, mm_._scale)
        hd_, ms_ = seq42.zones(f_)
        ys_, xs_ = _np42.nonzero(hd_)
        hw_, hh_ = (xs_.max() - xs_.min() + 1) * w_ / hd_.shape[1], (ys_.max() - ys_.min() + 1) * h_ / hd_.shape[0]
        print(f"         {lbl_}: head region {hw_:.0f} x {hh_:.0f} px = {100 * hw_ / SW:.2f}% x {100 * hh_ / SH:.2f}% of the screen "
              f"({hw_ * mpx:.0f} x {hh_ * mpx:.0f} mm); the self-test's position error is ~3 mm")
        check(f"{lbl_}: the head region is at least 8x the app's own position uncertainty (3 mm self-test median), "
              f"so a hit there is resolvable by aim, not by luck", min(hw_, hh_) * mpx >= 8 * 3.0)
    # -- 43. different figures per seat (drills 6, 7, 8): the draw, and what it costs in fairness ------
    import random as _rnd43
    from collections import Counter as _C43
    from lsa.artwork import catalog as _cat43
    cast43 = list(NAMES35)
    rng43 = _rnd43.Random(43)
    prev43, a43, b43 = None, [], []
    for _ in range(60000):
        prev43 = _cat43.pick_pair(rng43, prev43, cast43)
        a43.append(prev43[0])
        b43.append(prev43[1])
    n43 = len(cast43)
    ca, cb = _C43(a43), _C43(b43)
    exp43 = len(a43) / n43
    check("pick_pair: the two seats NEVER get the same figure in a cycle, and neither seat repeats its own figure",
          all(x_ != y_ for x_, y_ in zip(a43, b43))
          and all(a43[i_] != a43[i_ + 1] and b43[i_] != b43[i_ + 1] for i_ in range(len(a43) - 1)))
    check(f"...and each seat's draw stays uniform over the {n43} figures (every figure within 8% of its share, 60,000 cycles)",
          len(ca) == n43 and len(cb) == n43
          and max(abs(c_ - exp43) for c_ in list(ca.values()) + list(cb.values())) <= 0.08 * exp43,
          f"seat0 {min(ca.values())}..{max(ca.values())}, seat1 {min(cb.values())}..{max(cb.values())}, expected {exp43:.0f}")
    check("...every ordered pair of different figures occurs",
          len(set(zip(a43, b43))) == n43 * (n43 - 1))
    check("...a forced figure is both seats' figure, and a one-figure cast shares rather than raises",
          _cat43.pick_pair(rng43, None, cast43, "thug01") == ("thug01", "thug01")
          and _cat43.pick_pair(rng43, ("a", "a"), ["a"]) == ("a", "a"))
    from lsa.modes.figure import PlanLink as _PL43
    pl43 = _PL43()
    mk43 = lambda rng, prev, _i: _cat43.pick_pair(rng, prev, cast43)
    got43 = [pl43.get(k_, mk43) for k_ in range(200)]
    check("a shared plan hands both seats the SAME pair object for cycle k (so WHEN and WHERE cannot diverge) "
          "and the pair is different per seat",
          all(pl43.get(k_, mk43) is got43[k_] for k_ in range(200))
          and all(g_[0] != g_[1] for g_ in got43))
    # what the fairness trade costs, measured at the size a half allows (report only; gates nothing)
    from lsa.artwork import zones as _Z43
    from lsa.modes.figure import Figure as _Fig43, SINGLE_BUDGET as _SB43
    sctx43 = hs35._side_ctx(context(now=1.0), 0)
    fld43 = hs35.mode.playfield(sctx43)
    rows43 = []
    for nm_ in cast43:
        f_ = _Fig43(nm_, sctx43.width, sctx43.height, _SB43)
        hpx = f_.fit_height(fld43.height(), sctx43.height, 1.0)
        wpx = hpx * f_.aspect
        top_pts = max(z_["points"] for z_ in f_.zones)
        gx, gy = 90, 160
        cnt_any = cnt_top = 0
        for iy in range(gy):
            for ix in range(gx):
                pts, _z = _Z43.score((ix + 0.5) / gx, (iy + 0.5) / gy, f_.zones, f_.aspect, f_.stroke)
                cnt_any += pts > 0
                cnt_top += pts == top_pts
        cell = wpx * hpx / (gx * gy)
        sil = float(f_._sil.sum()) / f_._sil.size * wpx * hpx
        rows43.append((nm_, cnt_top * cell, cnt_any * cell, sil))
    for col_, lbl_ in ((1, "10-ring"), (2, "whole scored region"), (3, "silhouette")):
        vals_ = sorted((r_[col_], r_[0]) for r_ in rows43)
        print(f"         AREA SPREAD, {lbl_} at two-player half size: largest/smallest = "
              f"{vals_[-1][0] / max(1.0, vals_[0][0]):.2f}x  (largest {vals_[-1][1]} {vals_[-1][0]:.0f} px^2, "
              f"smallest {vals_[0][1]} {vals_[0][0]:.0f} px^2)")
    check("the area spread was measured for every figure (informational: it gates nothing, by his ruling)",
          len(rows43) == n43 and all(r_[1] > 0 and r_[2] > 0 for r_ in rows43))

    # ------------------------------------------------------------------
    # 44. EFFICIENCY: one figure per magazine (points / seconds), its arithmetic
    # printed under the score, an honesty mark naming the app's own faults.
    # ------------------------------------------------------------------
    print()
    print("44. efficiency: points per second per magazine, never averaged, arithmetic printed, honesty mark")
    import re as _re44
    from PySide6.QtGui import QImage as _QI44, QPainter as _QP44
    from lsa.modes import efficiency as _E44

    MAG44 = 4
    ctx44 = context(now=1.0)
    drawn44 = []
    _orig44 = _Host._draw_lines

    def _rec44(p_, box_, lines_, start_, bold=False, floor=9):
        size_ = _orig44(p_, box_, lines_, start_, bold, floor)
        drawn44.append((box_, list(lines_), size_, bold))
        return size_

    def _render44(h_, c_):
        drawn44.clear()
        _Host._draw_lines = staticmethod(_rec44)
        try:
            img_ = _QI44(c_.width, c_.height, _QI44.Format_RGB32)
            img_.fill(0xFF0A0A0A)
            p_ = _QP44(img_)
            h_.paint(p_, c_)
            p_.end()
        finally:
            _Host._draw_lines = staticmethod(_orig44)
        return list(drawn44)

    def _lines44(h_, c_):
        return [ln_ for _b, ls_, _s, _bd in _render44(h_, c_) for ln_ in ls_]

    def _h44(players=1, mag=MAG44):
        h_ = _Host()
        h_.set_magazine(mag)
        h_.set_mode(_Static(), ctx44, players=players)
        return h_

    def _shoot44(h_, side_, times_, hits_, now=None, end_pad=0.05):
        """One shot per entry of `times_` at that ev.t, on target where `hits_` says so."""
        for i_, t_ in enumerate(times_):
            if hits_[i_]:
                x_, y_ = (_aim(h_, ctx44, side_) if h_.versus
                          else h_.mode.aim_point(ctx44))
            else:
                x_, y_ = (60.0 if side_ == 0 else 3780.0), 1500.0
            pad_ = end_pad[i_] if isinstance(end_pad, (list, tuple)) else end_pad
            h_.shot(modes.ShotEvent(x_, y_, t_, i_ + 1, t_end=t_ + pad_),
                    context(now=t_ if now is None else now))

    # -- the arithmetic, pure ------------------------------------------------
    check("his own example: 165 points in 60.0 s is 2.75 and 120 points in 10.0 s is 12.00 (the careful shooter wins "
          "on points, the fast one on efficiency)",
          _E44.efficiency(165, 60.0) == 2.75 and _E44.efficiency(120, 10.0) == 12.0
          and _E44.efficiency_word(_E44.efficiency(120, 10.0)) == "12.00")
    check("the figure is computed from the seconds AS PRINTED: 140 points over 24.63 s prints 24.6 seconds and 5.69, "
          "so the line under it divides to the digit",
          _E44.arithmetic_text(140, 24.63) == "= 140 points / 24.6 seconds"
          and _E44.headline(_E44.efficiency(140, 24.63)) == "EFFICIENCY  5.69"
          and f"{140 / 24.6:.2f}" == "5.69" and f"{140 / 24.63:.2f}" == "5.68")
    check("no time, no figure: a single shot (zero seconds) is '--' and says why, never a huge number or a crash",
          _E44.efficiency(10, 0.0) is None and _E44.headline(None) == "EFFICIENCY  --"
          and "no time to divide by" in _E44.arithmetic_text(10, 0.0))
    check("NO PENALTY ANYWHERE in the arithmetic: efficiency is never negative and zero points is exactly zero",
          all((_E44.efficiency(p_, s_) or 0.0) >= 0.0 for p_ in (0, 1, 10, 170) for s_ in (0.5, 1.0, 24.6, 60.0))
          and _E44.efficiency(0, 10.0) == 0.0)
    clk44 = _E44.Clock()
    for t_ in (12.0, 10.0, 15.0, 11.0):
        clk44.shot(t_)
    check("the clock is order-independent (a shot handled late cannot move it) and belongs to one side of one round",
          clk44.seconds == 5.0)

    # -- single player, on the rendered screen --------------------------------
    h44 = _h44()
    T44 = [10.0, 12.5, 14.0, 15.0]                      # first-to-last = 5.0 s
    _shoot44(h44, 0, T44, [True, True, False, True], now=500.0)       # handled LONG after (a late tick)
    h44.advance(context(now=501.0))
    t44 = h44.tallies[0]
    check("the round is over and the clock is first shot to last, 5.0 s, from each ev.t (the shots were handled at "
          "now=500: a late tick cannot change it)",
          h44.round_over and h44.round_seconds(0) == 5.0 and t44.points > 0)
    shown44 = _lines44(h44, ctx44)
    eff44 = [ln_ for ln_ in shown44 if ln_.startswith("EFFICIENCY")]
    sum44 = [ln_ for ln_ in shown44 if ln_.startswith("= ")]
    # The points line reads "N of M points" since 2026-10-22, so the old filter
    # (`\d+ points?`) matched NOTHING and `pts44` came back empty: the cross-reference
    # between the printed points and the figure divided from them had silently stopped
    # being made, and only the equality against a hand-built string noticed. The line is
    # now parsed for the HUMAN's number N (M, the ghost's ceiling, is not an ingredient of
    # the division), and the check below REQUIRES exactly one such line and one seconds line
    # before it compares anything, so a filter that matches nothing fails by name.
    pts44 = [ln_ for ln_ in shown44 if _re44.fullmatch(r"\d+ (?:of \d+ )?points?", ln_)]
    sec44 = [ln_ for ln_ in shown44 if _re44.fullmatch(r"[\d.]+ seconds", ln_)]
    pm44 = _re44.fullmatch(r"(\d+) (?:of (\d+) )?points?", pts44[0]) if len(pts44) == 1 else None
    printed_n44 = int(pm44.group(1)) if pm44 else None
    printed_s44 = _re44.fullmatch(r"([\d.]+) seconds", sec44[0]).group(1) if len(sec44) == 1 else None
    want44 = (f"{printed_n44 / float(printed_s44):.2f}"
              if printed_n44 is not None and printed_s44 is not None else None)
    unit44 = "point" if printed_n44 == 1 else "points"
    check("the SCREEN shows efficiency = points / seconds, and the points and seconds printed above it are the ones "
          f"it was divided from ({t44.points} points / 5.0 seconds = {want44}): ONE points line and ONE seconds "
          "line are found, the points line carries the host's own tally, and the figure and the sum line are both "
          "built from exactly those two printed numbers",
          len(pts44) == 1 and len(sec44) == 1 and printed_n44 == t44.points and printed_s44 == "5.0"
          and eff44 == [f"EFFICIENCY  {want44}"]
          and sum44 == [f"= {printed_n44} {unit44} / {printed_s44} seconds"],
          str((eff44, sum44, pts44, sec44)))
    m44 = _re44.fullmatch(r"= (\d+) points? / ([\d.]+) seconds", sum44[0]) if sum44 else None
    check("the printed arithmetic divides to the printed figure",
          m44 is not None and abs(int(m44.group(1)) / float(m44.group(2)) - float(eff44[0].split()[1])) < 0.005)
    h44b = _h44()
    # pulses of DIFFERENT lengths (0.9 s ... 0.05 s): a clock built on the last frame would read 4.15 s
    _shoot44(h44b, 0, T44, [True, True, False, True], now=T44[-1], end_pad=[0.9, 0.05, 0.4, 0.05])
    check("the clock reads ev.t, not the pulse's last frame: pulses of different lengths change nothing",
          h44b.round_seconds(0) == 5.0 and h44b.side_efficiency(0) == h44.side_efficiency(0))

    # -- no penalty, no averaging ---------------------------------------------------
    hm44 = _h44()
    _shoot44(hm44, 0, [10.0, 11.0, 12.0, 13.0], [True, False, False, False])
    hz44 = _h44()
    _shoot44(hz44, 0, [10.0, 11.0, 12.0, 13.0], [True, True, True, True])
    p_hit44 = hm44.tallies[0].points
    check("a miss earns nothing and costs nothing else: one hit and three misses is exactly that hit's points / "
          "the first-to-last seconds, with no subtraction (the magazine already charged the rounds)",
          p_hit44 > 0 and hm44.round_misses(0) == 3 and hm44.side_efficiency(0) == p_hit44 / 3.0
          and hz44.tallies[0].points == 4 * p_hit44)
    hall44 = _h44()
    _shoot44(hall44, 0, [10.0, 11.0, 12.0, 13.0], [False] * 4)
    check("every round a miss: 0.00 efficiency, not negative",
          hall44.tallies[0].points == 0 and hall44.side_efficiency(0) == 0.0
          and _E44.headline(hall44.side_efficiency(0)) == "EFFICIENCY  0.00")
    hr44 = _h44()
    _shoot44(hr44, 0, [10.0, 11.0, 12.0, 13.0], [True] * 4)                 # round 1: 3.0 s
    e1_44 = hr44.side_efficiency(0)
    hr44.restart(ctx44)
    _shoot44(hr44, 0, [30.0, 30.5, 31.0, 31.5], [True, True, True, False])  # round 2: 1.5 s
    e2_44 = hr44.side_efficiency(0)
    check("no averaging across rounds: the second round's figure is its own points / its own seconds, not a mean "
          "with the first, and the first's figure was not altered by it",
          hr44.round_seconds(0) == 1.5 and e2_44 == hr44.tallies[0].points / 1.5 and e2_44 != (e1_44 + e2_44) / 2
          and e1_44 == 4 * p_hit44 / 3.0 and not hasattr(hr44, "efficiency_history"))

    # -- two players: both numbers for both sides, the better of each marked, and they can disagree ----
    h2 = _h44(players=2, mag=3)
    _shoot44(h2, 0, [10.0, 40.0, 70.0], [True, True, True])              # LEFT: careful, 60 s
    _shoot44(h2, 1, [10.0, 10.5, 11.0], [True, True, False])             # RIGHT: fast, 1 s, one miss
    h2.advance(context(now=71.0))
    pl2, pr2 = h2.tallies[0].points, h2.tallies[1].points
    el2, er2 = h2.side_efficiency(0), h2.side_efficiency(1)
    vl2 = h2.verdict_lines()
    check("two players: LEFT wins on POINTS and RIGHT is more efficient, and the screen says both",
          pl2 > pr2 and er2 > el2 and vl2[1] == f"LEFT WINS by {pl2 - pr2} points"
          and vl2[2] == f"RIGHT is more efficient, {er2:.2f} to {el2:.2f}", str((pl2, pr2, el2, er2, vl2)))
    b0, b1 = h2.side_block(0), h2.side_block(1)
    check("...both numbers for both sides appear, and the better of each is marked: LEFT's points, RIGHT's efficiency",
          b0[2].endswith("BEST") and not b1[2].endswith("BEST") and b1[4].endswith("BEST") and not b0[4].endswith("BEST")
          and b0[3] == "60.0 seconds" and b1[3] == "1.0 seconds"
          and b0[4].startswith(f"EFFICIENCY  {el2:.2f}") and b1[4].startswith(f"EFFICIENCY  {er2:.2f}"), str((b0, b1)))
    h2b = _h44(players=2, mag=3)
    _shoot44(h2b, 0, [10.0, 11.0, 12.0], [True, True, True])
    _shoot44(h2b, 1, [10.0, 20.0, 30.0], [True, True, False])
    h2b.advance(context(now=40.0))
    vb2 = h2b.verdict_lines()
    check("when one shooter takes both it is ONE line, not two: 'LEFT is also more efficient'",
          vb2[1].startswith("LEFT WINS by") and vb2[2].startswith("LEFT is also more efficient, ") and len(vb2) == 3)
    h2c = _h44(players=2, mag=3)
    _shoot44(h2c, 0, [10.0, 11.0, 12.0], [True, True, True])
    _shoot44(h2c, 1, [10.0, 11.0, 12.0], [True, True, True])
    h2c.advance(context(now=40.0))
    check("equal efficiency says nothing about efficiency (the line stays, blank, so the height does not change)",
          h2c.verdict_lines()[2] == "" and len(h2c.verdict_lines()) == 3)

    # -- the honesty gate: only when a named condition fired, and it names itself -------
    def _round44(feed=None, unseen=0, players=1):
        h_ = _h44(players=players, mag=3)
        if feed:
            feed(h_)
        if unseen:
            h_.note_heard_unseen(unseen)
        for side_ in ((0, 1) if players == 2 else (0,)):
            _shoot44(h_, side_, [10.0, 11.0, 12.0], [True] * 3)
        h_.advance(context(now=13.0))
        return h_

    clean44 = _round44()
    check("a clean round: nothing fired, no star, no footnote, and the two footnote lines are reserved and blank",
          clean44._round_reasons == [] and clean44.mark_text() == "" and clean44.mark_lines() == ["", ""]
          and "*" not in clean44.efficiency_lines(0)[0])

    def _cov44(h_):
        for k_ in range(0, 61):
            h_.note_measurement(1.0 + k_ * 0.1, covered=(0 < k_ <= 60))         # 6.0 s covered
        h_.note_measurement(7.2, covered=False)
    hc44 = _round44(_cov44)
    check("detections suppressed because something covered the screen: named, with the seconds",
          hc44._round_reasons == ["something covered the screen for 6s"]
          and hc44.mark_text() == "* measurement was poor this round: something covered the screen for 6s"
          and hc44.efficiency_lines(0)[0].endswith("*"), str(hc44._round_reasons))

    def _short44(h_):
        h_.note_measurement(0.0, covered=False)
        for k_ in range(1, 11):
            h_.note_measurement(k_ * 0.1, covered=True)                        # 1.0 s only
        h_.note_measurement(3.0, covered=False)
    check("...a covering of a second or so is not the round and does not mark it",
          _round44(_short44)._round_reasons == [])

    def _mv44(h_):
        h_.note_measurement(1.0, moved_count=0)
        h_.note_measurement(2.0, moved_count=1)
    check("the camera reported as having moved during the round: named",
          _round44(_mv44)._round_reasons == ["the camera appears to have moved"])

    def _old_mv44(h_):
        h_.note_measurement(1.0, moved_count=3)
        h_.note_measurement(2.0, moved_count=3)
    check("...a move counted BEFORE the round began is not this round's (the first reading is the baseline)",
          _round44(_old_mv44)._round_reasons == [])

    def _thin44(h_):
        h_.health.margin_floor = 3.0
        h_.note_measurement(1.0, margin=2.1)

    def _fat44(h_):
        h_.health.margin_floor = 3.0
        h_.note_measurement(1.0, margin=9.0)
    check("signal margin under the app's own bar (selfcheck.MARGIN_READY, passed in): named with the number",
          _round44(_thin44)._round_reasons == ["the signal margin was thin (2.1x, under 3.0x)"])
    check("...and a healthy margin does not mark it", _round44(_fat44)._round_reasons == [])
    cl_src44 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools" / "camera_lab.py").read_text(encoding="utf-8")
    from lsa.selfcheck import MARGIN_READY as _MR44
    check("the margin bar the app hands the host is the SELFCHECK constant, imported, not a second copy",
          _MR44 == 3.0 and "health.margin_floor = MARGIN_READY" in cl_src44 and "from lsa.selfcheck import MARGIN_READY" in cl_src44)

    def _bar44(h_):
        h_.note_measurement(1.0, bar=8.0)
        h_.note_measurement(2.0, bar=14.0)

    def _bar_small44(h_):
        h_.note_measurement(1.0, bar=8.0)
        h_.note_measurement(2.0, bar=10.0)
    check("the detection bar moving materially (8 to 14 is 1.75x) during the round: named",
          _round44(_bar44)._round_reasons == ["the detection bar moved from 8 to 14"])
    check("...a small drift (8 to 10) is not material", _round44(_bar_small44)._round_reasons == [])
    check("the microphone hearing pulls the camera never saw: named with the count",
          _round44(unseen=2)._round_reasons == ["the microphone heard 2 trigger pulls the camera never saw"]
          and _round44(unseen=1)._round_reasons == ["the microphone heard 1 trigger pull the camera never saw"])

    def _two44(h_):
        h_.note_measurement(1.0, bar=8.0, moved_count=0)
        h_.note_measurement(2.0, bar=14.0, moved_count=1)
    two44 = _round44(_two44)
    check("two conditions: each reads as its own clause in the one sentence, so neither is generic",
          two44._round_reasons == ["the camera appears to have moved", "the detection bar moved from 8 to 14"]
          and two44.mark_text().count("measurement was poor") == 1 and "; " in two44.mark_text())
    ha44 = _round44()
    ha44.note_measurement(99.0, covered=True)
    ha44.note_measurement(120.0, covered=True)
    check("readings after the round is over are ignored (the reasons are frozen at its end)",
          ha44._round_reasons == [] and ha44.health.covered_s == 0.0)
    nxt44 = _round44(_cov44)
    nxt44.restart(ctx44)
    check("a new round starts clean: nothing carries over from the last one",
          nxt44._round_reasons == [] and nxt44.health.covered_s == 0.0)
    hm2_44 = _round44(_mv44, players=2)
    check("two players: the mark is the round's, drawn once, and both efficiency figures carry the star",
          all(hm2_44.side_block(s_)[4].count("*") == 1 for s_ in (0, 1)) and hm2_44.mark_lines()[0].startswith("* measurement"))

    # -- the screen is the same height with and without the mark, at rounds 1, 2 and 4 -----
    def _geom44(h_):
        return sorted((b_.x(), b_.y(), b_.width(), b_.height(), len(ls_)) for b_, ls_, _s, _bd in _render44(h_, ctx44))

    def _nth44(n_, marked_, players=1):
        h_ = _h44(players=players, mag=3)
        for r_ in range(1, n_ + 1):
            if marked_:
                h_.note_measurement(1.0, bar=8.0)
                h_.note_measurement(2.0, bar=20.0)
            for side_ in ((0, 1) if players == 2 else (0,)):
                _shoot44(h_, side_, [10.0 * r_, 10.0 * r_ + 1, 10.0 * r_ + 2], [True] * 3)
            h_.advance(context(now=10.0 * r_ + 5))
            if r_ < n_:
                h_.restart(ctx44)
        return h_

    for players44 in (1, 2):
        ref44 = _geom44(_nth44(1, False, players44))
        same44 = all(_geom44(_nth44(n_, m_, players44)) == ref44 for n_ in (1, 2, 4) for m_ in (False, True))
        marked44 = _nth44(2, True, players44)
        check(f"{'single' if players44 == 1 else 'two'}-player: the round screen has the same boxes at rounds 1, 2 and 4, "
              f"with and without the mark",
              same44 and bool(marked44._round_reasons) and marked44.mark_lines()[0] != "")

    # -- the words: never called what it is not ------------------------------------------
    src44 = Path(__file__).resolve().parents[1] / "src"
    bad44 = []
    for f_ in src44.rglob("*.py"):
        if _re44.search(r"hit[\s_-]*factor", f_.read_text(encoding="utf-8", errors="replace"), _re44.I):
            bad44.append(f_.name)
    check("the phrase the spec forbids appears nowhere in the app's source: not in code, strings, logs or comments",
          not bad44, str(bad44))
    ev44 = h2.take_events() + h44.take_events() + hc44.take_events()
    allscreen44 = " ".join(_lines44(h2, ctx44) + _lines44(h44, ctx44) + _lines44(hc44, ctx44) + ev44)
    check("...nor in anything the rendered screens or the host's log said; and the log carries each round's figure",
          not _re44.search(r"hit[\s_-]*factor", allscreen44, _re44.I)
          and any("efficiency, magazine" in e_ for e_ in ev44) and any("efficiency, LEFT" in e_ for e_ in ev44)
          and any("efficiency mark: * measurement was poor this round: something covered" in e_ for e_ in ev44),
          str((len(ev44), [e_[:70] for e_ in ev44 if "efficiency" in e_])))

    # ------------------------------------------------------------------
    # 45. THE STATIC SCREEN MUST BE CHEAP TO PAINT (2026-10-20, the camera at 2 fps).
    # The ring numbers' size search constructed ~1000 fonts per paint. Under the
    # offscreen platform that cost 17 ms and no suite noticed; under the Windows
    # font engine it cost 171 ms, the HUD repaints this screen five times a second,
    # and the Qt thread then held the interpreter lock for most of every second
    # while the capture thread's cap.read() took 46 ms instead of 10 (fps 47 -> 2).
    # The platform-independent symptom is the NUMBER OF FONTS BUILT PER PAINT.
    # ------------------------------------------------------------------
    print()
    print("45. Static Targets: no repaints when idle, and a paint that builds almost no fonts")
    import random as _rnd45
    import lsa.display.layers as _L45
    from PySide6.QtGui import QFontMetricsF as _FM45, QImage as _QI45, QPainter as _QP45

    for players45 in (1, 2):
        h45 = _Host()
        h45.set_mode(_Static(), context(now=1.0), players=players45)
        asks45 = sum(1 for k_ in range(180) if h45.advance(context(now=1.0 + k_ / 60.0)))
        check(f"{players45}-player Static Targets with nothing happening asks for NO repaints in 180 idle ticks "
              f"(it asked {asks45})", asks45 == 0)

    built45 = [0]
    _mono45 = _L45.mono

    def _counting45(*a_, **k_):
        built45[0] += 1
        return _mono45(*a_, **k_)
    _L45.mono = _counting45
    try:
        for players45 in (1, 2):
            c45 = context(now=1.0)
            h45 = _Host()
            h45.set_mode(_Static(), c45, players=players45)
            fld45 = h45.mode._field
            sc45 = h45._side_ctx(c45, 0) if players45 == 2 else c45
            img45 = _QI45(c45.width, c45.height, _QI45.Format_RGB32)
            runs45 = []
            _L45.AimField._fit_digits.cache_clear()
            fld45._nums_cache = None
            for _i in range(3):
                built45[0] = 0
                p45 = _QP45(img45)
                fld45.paint(p45, sc45)
                p45.end()
                runs45.append(built45[0])
            check(f"{players45}-player: painting the aim field builds few fonts: cold {runs45[0]} (limit 100), "
                  f"then {runs45[1]} and {runs45[2]} (limit 40; the uncached search built about a thousand every paint)",
                  runs45[0] <= 100 and runs45[1] <= 40 and runs45[2] <= 40, str(runs45))
    finally:
        _L45.mono = _mono45
    n1_45 = fld45.ring_numbers(sc45)
    check("the ring numbers are cached: the same geometry returns the very same list",
          fld45.ring_numbers(sc45) is n1_45 and len(n1_45) > 0)
    small45 = context(now=1.0)
    small45 = type(small45)(now=1.0, dt=1 / 60.0, rect=QRect(0, 0, 1920, 1080), stats=SessionStats())
    n2_45 = fld45.ring_numbers(small45)
    check("...and a different screen size is NOT served from the cache (the numbers are re-sized for it)",
          n2_45 is not n1_45 and n2_45 != n1_45 and fld45.ring_numbers(sc45) == n1_45)

    def _linear45(text_, want_h_, max_w_):
        for size_ in range(200, 7, -1):
            fm_ = _FM45(_L45.mono(size_, True))
            if fm_.boundingRect("8").height() <= want_h_ and fm_.horizontalAdvance(text_) <= max_w_:
                return size_
        return None
    rng45 = _rnd45.Random(45)
    cases45 = [(rng45.choice(["10", "9", "8", "7"]), rng45.uniform(3, 200), rng45.uniform(3, 300)) for _ in range(150)]
    mism45 = [c_ for c_ in cases45 if _L45.AimField._fit_digits(*c_) != _linear45(*c_)]
    check("the size search (a binary search now) returns exactly what the old size-by-size walk returned, over 150 "
          "random fits", not mism45, str(mism45[:2]))

    # ------------------------------------------------------------------
    print("\n46. Keyboard Shortcuts (K): the list is read from the code, every key on it is real, "
          "and the screen costs a shot nothing")
    # The screen exists because a hand-typed key legend went six entries stale and
    # told him the wrong thing for three days with every suite green. So the checks
    # here are about the two ways THIS screen could go stale the same way: its drill
    # rows drifting from the registry (1, 2) and its fixed rows advertising a key
    # nothing handles (2); and about what drawing it costs a shot (4, 5).
    import re as _re46
    import numpy as _np46
    from PySide6.QtCore import QRectF as _R46, Qt as _Qt46
    from PySide6.QtGui import QFontMetricsF as _FM46, QImage as _QI46, QPainter as _QP46
    from lsa.display import layers as _L46
    from lsa.modes import base as _base46, menu as _menu46, shortcuts as _sc46
    from lsa.modes.base import Mode as _Mode46, ModeHost as _MH46

    c4_46 = context(now=1.0)                                            # 3840 x 2160
    c1080_46 = type(c4_46)(now=1.0, dt=1 / 60.0, rect=QRect(0, 0, 1920, 1080), stats=SessionStats())
    _HEADS46 = [_sc46.HEAD_DRILLS, _sc46.HEAD_PLAYERS, _sc46.HEAD_CHECKS,
                _sc46.HEAD_ON_SCREEN, _sc46.HEAD_ALSO]

    class _Spy46(_QP46):
        """Records what is ACTUALLY drawn: the text, the font it was drawn in, the
        colour it was drawn at and the box it was drawn into; and every rounded
        rectangle with its pen width and brush."""

        def __init__(self, dev):
            super().__init__(dev)
            self.texts = []        # (text, point size, bold, pen red, rect or None)
            self.boxes = []        # (rect, pen width, is a bare stroke)

        def drawText(self, *a):
            f = self.font()
            self.texts.append((a[-1], f.pointSize(), f.bold(), self.pen().color().red(),
                               a[0] if isinstance(a[0], _R46) else None))
            return super().drawText(*a)

        def drawRoundedRect(self, *a):
            self.boxes.append((a[0], self.pen().widthF(), self.brush().style() == _Qt46.NoBrush))
            return super().drawRoundedRect(*a)

    def _render46(draw, ctx, spy=False):
        """The paint on the target background (luminance 10) as BGRA, and the painter.
        Filled BEFORE the painter is attached: `fill` does nothing while one is."""
        img = _QI46(ctx.width, ctx.height, _QI46.Format_RGB32)
        img.fill(0xFF0A0A0A)
        p_ = _Spy46(img) if spy else _QP46(img)
        draw(p_, ctx)
        p_.end()
        a_ = _np46.frombuffer(img.constBits(), _np46.uint8).reshape(ctx.height, img.bytesPerLine())
        return a_[:, :ctx.width * 4].reshape(ctx.height, ctx.width, 4).copy(), p_

    def _rows46(texts):
        """{heading: [(key glyph, description), ...]} in the order it was drawn."""
        out, cur = {}, None
        for t_ in texts:
            s_ = t_[0]
            if s_ == _sc46.TITLE:
                continue
            if s_ in _HEADS46:
                cur = s_
                out[cur] = []
                continue
            out[cur].append(s_)
        return {h_: list(zip(v_[0::2], v_[1::2])) for h_, v_ in out.items()}

    m46 = _sc46.Shortcuts()
    m46.start(c4_46)
    check("K is registered to the Keyboard Shortcuts mode, found by key and by name",
          modes.by_key(_Qt46.Key_K) is _sc46.Shortcuts and modes.by_name("shortcuts") is _sc46.Shortcuts
          and _sc46.Shortcuts.title == "Keyboard Shortcuts" and _sc46.Shortcuts.is_screen is True)

    # -- 1. the drill rows are READ from the registry ---------------------------
    _a46, spy46 = _render46(m46.paint, c4_46, spy=True)
    rows46 = _rows46(spy46.texts)
    drawn46 = [r_ for r_ in rows46.get(_sc46.HEAD_DRILLS, []) if r_ != ("N", "New Round")]
    want46 = [(_menu46.glyph_of_key(k_), c_.title) for c_, k_ in _base46.registered() if k_ is not None]
    check("the drill rows DRAWN agree with lsa.modes.registered(), key for key and title for title, "
          "in both directions (and N New Round is there too)",
          len(want46) >= 11 and sorted(drawn46) == sorted(want46)
          and ("N", "New Round") in rows46.get(_sc46.HEAD_DRILLS, []),
          f"drawn {sorted(drawn46)} vs registry {sorted(want46)}")
    _x46 = type("Extra_zz46", (_Mode46,), {
        "name": "zz46", "title": "Zz Test Drill", "summary": "test only",
        "paint": lambda self, p, ctx: None, "start": lambda self, ctx: None})
    _base46.register(key=_Qt46.Key_J)(_x46)
    try:
        with_x46 = _rows46(_render46(m46.paint, c4_46, spy=True)[1].texts).get(_sc46.HEAD_DRILLS, [])
    finally:
        _base46._REGISTRY.pop("zz46", None)
    without_x46 = _rows46(_render46(m46.paint, c4_46, spy=True)[1].texts).get(_sc46.HEAD_DRILLS, [])
    check("...and it is DERIVED, not a snapshot: a mode registered after this one was built gets a row "
          "(same instance, so the layout cache follows the registry), and loses it when removed",
          ("J", "Zz Test Drill") in with_x46 and ("J", "Zz Test Drill") not in without_x46
          and sorted(without_x46) == sorted(rows46[_sc46.HEAD_DRILLS]))
    expect46 = {
        _sc46.HEAD_PLAYERS: [("O", "One Player"), ("T", "Two Players")],
        _sc46.HEAD_CHECKS: [("C", "Self-Calibrate"), ("A", "Self-Check"), ("Q", "Last Verdict"),
                            ("G", "Calibrate by Shooting"),
                            ("U", "Calibrate Brightness by Shooting"),
                            ("X", "Measure the Exposure"),
                            ("F", "Measure the Noise Floor")],
        _sc46.HEAD_ON_SCREEN: [("0", "All Readouts"), ("S", "Stats"), ("B", "Status Band"),
                               ("L", "Signal Meter"), ("P", "Camera Pane"), ("V", "Aim the Camera")],
        _sc46.HEAD_ALSO: [("E", "Erase the Holes"), ("+ -", "Volume Up / Down"),
                          ("Esc", "Exit the Game")]}
    check("the fixed rows are the ones the owner approved in requirements/help_screen_mockup_01.html: "
          "the five headings in order, the title, and every wording",
          list(rows46) == _HEADS46 and all(rows46[h_] == v_ for h_, v_ in expect46.items())
          and any(t_[0] == "KEYBOARD SHORTCUTS" for t_ in spy46.texts), str(list(rows46)))

    # -- 2. every key the screen advertises is actually handled ------------------
    # SOURCE SCAN, from the code that handles keys rather than from a list kept
    # beside it: `camera_lab._on_target_key` (and the key tuples it reads), every
    # Layer.on_key in layers.py and menu.py, the window's own keyPressEvent and
    # TOGGLES, and the mode registry. The menu's O and T are asked at run time,
    # because they are read from its chrome table and not spelled in the handler.
    def _strip46(src):
        src = _re46.sub(r'""".*?"""', "", src, flags=_re46.S)
        return "\n".join(_re46.sub(r"#.*", "", ln_) for ln_ in src.splitlines())

    def _bodies46(src, fname):
        lines_, out_, i_ = src.splitlines(), [], 0
        pat_ = _re46.compile(r"^(\s*)def " + fname + r"\(")
        while i_ < len(lines_):
            m_ = pat_.match(lines_[i_])
            if not m_:
                i_ += 1
                continue
            ind_, body_, j_ = len(m_.group(1)), [lines_[i_]], i_ + 1
            while j_ < len(lines_) and (not lines_[j_].strip()
                                        or len(lines_[j_]) - len(lines_[j_].lstrip()) > ind_):
                body_.append(lines_[j_])
                j_ += 1
            out_.append("\n".join(body_))
            i_ = j_
        return out_

    _raw46 = {n_: (ROOT_SRC / "lsa" / p_).read_text(encoding="utf-8") for n_, p_ in (
        ("lab", "tools/camera_lab.py"), ("layers", "display/layers.py"),
        ("menu", "modes/menu.py"), ("window", "display/window.py"))}
    _target46 = "\n".join(_bodies46(_raw46["lab"], "_on_target_key"))
    for _kn46 in set(_re46.findall(r"self\.(\w+_KEYS)", _target46)):       # VOLUME_UP_KEYS, ...
        _kd46 = _re46.search(_kn46 + r"\s*=\s*\([^)]*\)", _raw46["lab"])
        _target46 += "\n" + (_kd46.group(0) if _kd46 else "")
    _tog46 = _re46.search(r"^\s*TOGGLES\b[^=\n]*=\s*\{[^}]*\}", _raw46["window"], _re46.M)
    srcs46 = {
        "camera_lab._on_target_key": _strip46(_target46),
        "a Layer.on_key": _strip46("\n".join(_bodies46(_raw46["layers"], "on_key")
                                              + _bodies46(_raw46["menu"], "on_key"))),
        "window.keyPressEvent": _strip46("\n".join(_bodies46(_raw46["window"], "keyPressEvent"))),
        "window.TOGGLES": _strip46(_tog46.group(0) if _tog46 else ""),
    }
    _TOK46 = {int(_Qt46.Key_Plus): "Qt.Key_Plus", int(_Qt46.Key_Minus): "Qt.Key_Minus",
              int(_Qt46.Key_Escape): "Qt.Key_Escape"}

    def _handled46(code, srcs):
        """Who handles this key code, or None."""
        if _base46.by_key(code) is not None:
            return "the mode registry"
        tok_ = ("Qt.Key_" + chr(code) if (0x30 <= code <= 0x39 or 0x41 <= code <= 0x5A)
                else _TOK46.get(code))
        if tok_ is not None:
            for name_, text_ in srcs.items():
                if _re46.search(_re46.escape(tok_) + r"\b", text_):
                    return name_
        if _menu46.Menu().on_key(code):
            return "the menu (O, T)"
        return None

    adv46 = _sc46.advertised()
    unhandled46 = [(e_.glyph, c_) for e_ in adv46 for c_ in e_.keys if _handled46(c_, srcs46) is None]
    check(f"EVERY KEY THE SCREEN ADVERTISES IS HANDLED somewhere ({sum(len(e_.keys) for e_ in adv46)} key codes "
          f"on {len(adv46)} rows): _on_target_key, a Layer.on_key, the window, or the mode registry",
          len(adv46) >= 25 and not unhandled46, f"unhandled: {unhandled46}")
    _cut46 = dict(srcs46)
    _cut46["camera_lab._on_target_key"] = srcs46["camera_lab._on_target_key"].replace("Qt.Key_C", "Qt.Key_ZZ")
    check("...and the scan CAN fail: with the Qt.Key_C branch taken out of _on_target_key, C is reported "
          "unhandled, while the real source still finds it",
          _handled46(int(_Qt46.Key_C), _cut46) is None and _handled46(int(_Qt46.Key_C), srcs46) is not None
          and _handled46(0x0100FFFF, srcs46) is None)

    # -- 3. it claims no keys ----------------------------------------------------
    host46 = _MH46()
    host46.set_mode(m46, c4_46)
    keys46 = {int(getattr(_Qt46, "Key_" + ch_)) for ch_ in "1234567890MCAGXFQEKNOTSBLPV"}
    keys46 |= {int(_Qt46.Key_Escape), int(_Qt46.Key_Plus), int(_Qt46.Key_Minus), int(_Qt46.Key_Space)}
    keys46 |= {c_ for e_ in adv46 for c_ in e_.keys}
    swallowed46 = sorted(k_ for k_ in keys46 if m46.on_key(k_) or host46.on_key(k_))
    check(f"it claims NO key: none of {len(keys46)} keys (1-9, M, K, C, A, G, X, F, Q, E, 0 and the rest) is swallowed "
          f"by the mode or by the host, so a press from this screen reaches the app (1 starts drill 1)",
          not swallowed46 and modes.by_key(_Qt46.Key_1) is not None, f"swallowed {swallowed46}")

    # -- 4. red excess exactly 0, and the palette -------------------------------
    # -- 5. the lit-area budget, MEASURED by rendering (through the host, so the MENU button counts)
    sizes46 = (("4K", c4_46), ("1080p", c1080_46))
    for label46, cx46 in sizes46:
        h46 = _MH46()
        h46.set_mode(_sc46.Shortcuts(), cx46)
        a46, _p46 = _render46(h46.paint, cx46)
        lum46 = cv2.cvtColor(_np46.ascontiguousarray(a46[..., :3]), cv2.COLOR_BGR2GRAY)
        red46 = int(cv2.subtract(a46[..., 2], cv2.max(a46[..., 1], a46[..., 0])).max())
        lit46 = float((lum46 > 60).mean())
        ink46 = int((lum46 > 30).sum())
        print(f"     MEASURED {label46}: red excess {red46}, brightest {int(lum46.max())} (ceiling {_L46.CEILING}), "
              f"lit above luminance 60 = {lit46 * 100:.3f}% of the screen (budget 2%), prose luminance "
              f"{_L46.GREY_TEXT.red()}, keys {_L46.GREY_HIT.red()}")
        check(f"{label46}: the whole screen is neutral grey (red excess exactly 0) and something is drawn",
              red46 == 0 and ink46 > 2000, f"red excess {red46}, {ink46} px above 30")
        check(f"{label46}: nothing is brighter than the artwork ceiling ({int(lum46.max())} vs {_L46.CEILING})",
              int(lum46.max()) <= _L46.CEILING)
        check(f"{label46}: lit area (>60) {lit46 * 100:.3f}% is inside the 2% budget, MEASURED by rendering "
              f"(RULES B1b)", 0.0005 < lit46 <= 0.02, f"{lit46 * 100:.3f}%")
        # A MARGIN, not just the bar (2026-10-23): the screen sat at 2.005% at 1080p and was
        # one added row from failing, and a check sitting on its bar is a flake waiting to
        # happen. 1.95% leaves room for the next row at ~0.03 points each.
        check(f"{label46}: lit area (>60) {lit46 * 100:.3f}% keeps a MARGIN under the budget (1.95%), "
              f"so one more row does not break it", lit46 <= 0.0195, f"{lit46 * 100:.3f}%")
    stroke_ok46 = (len(spy46.boxes) == sum(1 for t_ in spy46.texts if t_[2] and t_[3] == _L46.GREY_HIT.red()
                                           and t_[0] in {e_.glyph for e_ in adv46})
                   and all(b_[2] and b_[1] > 0 for b_ in spy46.boxes))
    kinds46 = {}
    for t_ in spy46.texts:
        kinds46.setdefault((t_[2], t_[3]), 0)
        kinds46[(t_[2], t_[3])] += 1
    check("the keys are BOXED with a thin STROKE (never a fill), glyph and box both at the ceiling; prose, headings "
          "and title are the palette's GREY_TEXT, GREY_MID and GREY_HI",
          stroke_ok46 and len(spy46.boxes) == len(adv46)
          and {t_[3] for t_ in spy46.texts if t_[0] == "KEYBOARD SHORTCUTS"} == {_L46.GREY_HI.red()}
          and {t_[3] for t_ in spy46.texts if t_[0] in _HEADS46} == {_L46.GREY_MID.red()}
          and all(t_[3] == _L46.GREY_TEXT.red() for t_ in spy46.texts
                  if not t_[2] and t_[0] not in {e_.glyph for e_ in adv46}),
          f"{len(spy46.boxes)} boxes for {len(adv46)} rows; (bold, grey) -> count {kinds46}")

    # -- 6. no text overruns its box, at both sizes and with the MENU button in any corner -------
    _corner46 = _Mode46.MENU_CORNER
    over46, tight46, scales46 = [], [], []
    try:
        for corner_ in ("top left", "top right", "bottom left", "bottom right"):
            _Mode46.MENU_CORNER = corner_
            for label46, cx46 in sizes46:
                mm46 = _sc46.Shortcuts()
                mm46.start(cx46)
                _a, sp46 = _render46(mm46.paint, cx46, spy=True)
                plan46 = mm46.plan(cx46)
                kb46 = _Mode46.menu_keepout(cx46)
                scales46.append(round(plan46.scale, 2))
                for text_, pt_, bold_, _r, rect_ in sp46.texts:
                    fm_ = _FM46(_L46.mono(pt_, bold_))
                    if (rect_ is None or fm_.horizontalAdvance(text_) > rect_.width() + 0.5
                            or fm_.height() > rect_.height() + 0.5):
                        over46.append(f"{corner_}/{label46}: {text_!r}")
                    elif rect_.intersects(kb46) and not (text_ == "KEYBOARD SHORTCUTS"):
                        tight46.append(f"{corner_}/{label46}: {text_!r} under the MENU button")
                    elif (rect_.left() < -0.5 or rect_.top() < -0.5 or rect_.right() > cx46.width + 0.5
                          or rect_.bottom() > cx46.height + 0.5):
                        tight46.append(f"{corner_}/{label46}: {text_!r} off the screen")
                if (not plan46.fits or plan46.elided or plan46.scale < 0.6
                        or any(it_.rect.intersects(kb46) for it_ in plan46.items)):
                    tight46.append(f"{corner_}/{label46}: fits={plan46.fits} elided={plan46.elided} "
                                   f"scale={plan46.scale:.2f}")
                drawn_t46 = [t_[0] for t_ in sp46.texts]
                if drawn_t46 != [it_.text for it_ in plan46.items]:
                    tight46.append(f"{corner_}/{label46}: what was drawn is not the plan")
                if corner_ == "top left":
                    # The title sits to the RIGHT of the button, level with it.
                    t46 = next(it_ for it_ in plan46.items if it_.kind == "title")
                    if not (t46.rect.left() >= kb46.right()
                            and abs(t46.rect.center().y() - kb46.center().y()) < 1.0):
                        tight46.append(f"{corner_}/{label46}: the title is not beside the MENU button")
    finally:
        _Mode46.MENU_CORNER = _corner46
    check("no text overruns its box, measured with the painter's own font at 3840x2160 and 1920x1080 with the "
          "MENU button in each of the four corners; nothing is elided, nothing is under the button or off the "
          "screen, and the type stays at 0.6 or more of its designed size",
          not over46 and not tight46, f"overruns {over46[:3]}; problems {tight46[:3]}; scales {sorted(set(scales46))}")

    # -- 7. static: no repaints when idle, and the layout is built once per geometry ----------
    ticks46 = sum(1 for i_ in range(180) if m46.advance(context(now=5.0 + i_ / 60.0)))
    hticks46 = sum(1 for i_ in range(180) if host46.advance(context(now=5.0 + i_ / 60.0)))
    check(f"it asks for no repaints when idle: {ticks46} in 180 ticks as a mode, {hticks46} through the host "
          f"(limit 5), and `animated` is False", ticks46 <= 5 and hticks46 <= 5 and m46.animated is False)
    _calls46 = [0]
    _real_bp46 = _sc46.build_plan

    def _counting46(*a, **k):
        _calls46[0] += 1
        return _real_bp46(*a, **k)
    _sc46.build_plan = _counting46
    try:
        mc46 = _sc46.Shortcuts()
        mc46.start(c4_46)
        for _i in range(3):
            _render46(mc46.paint, c4_46)
        built_once46 = _calls46[0]
        same46 = mc46.plan(c4_46) is mc46.plan(c4_46)
        _render46(mc46.paint, c1080_46)
        built_two46 = _calls46[0]
    finally:
        _sc46.build_plan = _real_bp46
    check("the layout is built ONCE per geometry and reused: 3 paints of one screen size build it once, "
          "a second size builds it a second time", built_once46 == 1 and same46 and built_two46 == 2,
          f"{built_once46} then {built_two46}")
    cal46 = type(c4_46)(now=1.0, dt=0.0, rect=QRect(0, 0, SW, SH), stats=SessionStats(), calibrating=True)
    a_cal46, _p = _render46(m46.paint, cal46)
    check("while the grid is being shot it draws NOTHING (one ring to shoot at)",
          int(cv2.cvtColor(_np46.ascontiguousarray(a_cal46[..., :3]), cv2.COLOR_BGR2GRAY).max()) == 10)

    # -- 8. a shot here scores nothing, spends no round, and the MENU button can still claim it ---
    h9 = _MH46()
    h9.set_mode(_sc46.Shortcuts(), c4_46)
    mag46 = h9.magazine
    outs46 = [h9.shot(modes.ShotEvent(SW / 2.0, SH / 2.0, 10.0 + i_ * 0.2, i_ + 1),
                      context(now=10.0 + i_ * 0.2)) for i_ in range(mag46 + 8)]
    check(f"{len(outs46)} shots (more than the {mag46}-round magazine) all come back as unscored misses with a hole, "
          f"and no round is spent: no tally, no end-of-round screen",
          all((not o_.hit) and o_.points == 0 and o_.unscored and o_.draw_hole for o_ in outs46)
          and h9.counts_ammo is False and h9.tallies[0].shots == 0 and not h9.round_over
          and _MH46().counts_ammo is False)
    sc9 = _MH46()
    sc9.set_mode(modes.StaticTargets(), c4_46)
    check("...and the control holds: a real drill on the same host DOES spend rounds",
          sc9.counts_ammo is True)
    bc46 = _Mode46.menu_rect(c4_46).center()
    o1 = h9.shot(modes.ShotEvent(bc46.x(), bc46.y(), 40.0, 100), context(now=40.0))
    armed46 = h9.menu_button.armed_at > 0
    o2 = h9.shot(modes.ShotEvent(bc46.x(), bc46.y(), 41.0, 101), context(now=41.0))
    check("two shots on the MENU button take it (the host offers the screen the shot first and the button "
          "only claims a miss), each with its selection cue",
          armed46 and h9.take_menu_request() and o1.cue is not None and o2.cue is not None)
    h2p46 = _MH46()
    h2p46.set_mode(_sc46.Shortcuts(), c4_46, players=2)
    ctl46 = next((c_ for c_ in modes.available()
                  if not c_.versus_ready and not c_.is_screen and c_.name != "menu"), None)
    h2c46 = _MH46()
    if ctl46 is not None:
        h2c46.set_mode(ctl46(), c4_46, players=2)
    check("with two players saved this screen is not split and does not apologise "
          "(a drill that is not ready for two players DOES, which is the control)",
          not h2p46.versus and h2p46.fallback_note == "" and (ctl46 is None or h2c46.fallback_note != ""))

    # ------------------------------------------------------------------
    # 47. THE GHOST'S INPUTS (2026-10-21, piece 2 of the ghost score).
    # `lsa/ghost.py` scores a round against a perfect shooter and has no table of
    # per-drill values, on purpose: a copy of another module's rule drifts (the
    # exposure probe's own copy of the threshold chose the dimmest dot on the table).
    # So each drill must state the most a shot can be worth FROM ITS OWN SCORER, and
    # record when each of its targets became hittable, on the clock the shots use.
    # Every check below runs the drill and compares; none asserts a drill's value.
    # ------------------------------------------------------------------
    print()
    print("47. the ghost: every drill states its ceiling from its own scorer and records when each target became hittable")
    from lsa import ghost as _g47
    from lsa.modes import figure as _fig47
    from lsa.modes.advancing import AdvancingTarget as _A47
    from lsa.modes.animated import AnimatedFigure as _AN47
    from lsa.modes.balls import BouncingBalls as _BB47
    from lsa.modes.base import Mode as _M47
    from lsa.modes.image_target import ImageTarget as _I47
    from lsa.modes.popup import PopUpFigure as _P47
    from lsa.modes.reaction import ReactionDrill as _RD47, reaction_points as _rp47
    from lsa.modes.static import StaticTargets as _ST47
    from lsa.settings import MAGAZINE_MAX as _MAGMAX47

    TICK47 = 1 / 60.0

    def _ev47(x, y, t, i=1):
        return modes.ShotEvent(x, y, t, i)

    # -- 1. the registry is WALKED, never listed ---------------------------------
    drills47 = [c_ for c_ in modes.available() if not (c_.is_screen or c_.name == "menu")]
    screens47 = [c_ for c_ in modes.available() if c_.is_screen or c_.name == "menu"]
    check("the registry holds drills and screens, so the walk below walks something",
          len(drills47) >= 9 and len(screens47) >= 2,
          f"{len(drills47)} drills, {len(screens47)} screens")
    unanswered47 = []
    for cls_ in drills47:
        m_ = cls_()
        m_.start(context(now=100.0))
        cl_ = m_.ghost_ceiling()
        if cl_ is None or _g47.ghost_points(_DEFAULT_MAG, cl_) is None:
            unanswered47.append(cls_.name)
    check("EVERY registered drill states a ceiling that is a positive whole number for every shot of a "
          "magazine (a drill added next year that forgets is caught here, by name)",
          not unanswered47, f"unanswered: {unanswered47}")
    check("every SCREEN (the menu, the shortcuts list) returns None: it is not a drill, so it is not scorable",
          all(c_().ghost_ceiling() is None for c_ in screens47))
    check("a drill with no ceiling is NOT SCORABLE in the ghost, never a number",
          not _g47.ghost_score(human_points=5, human_elapsed=3.0, shots=4,
                               ceiling=screens47[0]().ghost_ceiling()).scorable)

    # -- 2. each ceiling AGREES with what its own scorer awards a perfect hit -----
    st47 = _ST47()
    sc47 = context(now=1.0)
    check("Static Targets cannot state a ceiling before a round starts (it learns it from its own field)",
          st47.ghost_ceiling() is None)
    st47.start(sc47)
    mk47 = st47._field.marks(sc47)
    pts47 = [st47.shot(_ev47(cx_, cy_, 1.0, i_ + 1), sc47).points for i_, (cx_, cy_, _r) in enumerate(mk47)]
    ce_st47 = st47.ghost_ceiling()
    check("Static Targets: the ceiling is what the scorer awards the dead centre of the best mark, and no "
          "mark's centre scores more", ce_st47(1) == max(pts47) > 0 and ce_st47(17) == ce_st47(1),
          f"centres score {pts47}")

    cb47 = ClearTheBoard(count=12)
    cc47 = context(now=1.0)
    cb47.start(cc47)
    ob47 = cb47.shot(_ev47(cb47.targets[0].x, cb47.targets[0].y, 1.0), cc47)
    check("Clear The Board: the ceiling is what a hit on a circle scored",
          ob47.hit and ob47.points == cb47.ghost_ceiling()(1) > 0, f"{ob47.points}")

    eb47 = EscalatingBall()
    eb47.start(context(now=0.0))
    got47 = [eb47.shot(_ev47(eb47.x, eb47.y, float(k_), k_), context(now=float(k_))).points
             for k_ in range(1, 9)]
    ce47 = eb47.ghost_ceiling()
    check("Escalating Ball: eight perfect hits score exactly the ceiling's k-th value each, it climbs and it caps",
          got47 == [ce47(k_) for k_ in range(1, 9)] and got47[0] < got47[-1] and got47[-1] == got47[-2],
          f"scored {got47}")

    bb47 = _BB47()
    bb47.start(context(now=0.0))
    got47b = []
    for k_ in range(1, 9):
        b0_ = bb47.balls[0]
        got47b.append(bb47.shot(_ev47(b0_.x, b0_.y, float(k_), k_), context(now=float(k_))).points)
    ce47b = bb47.ghost_ceiling()
    check("Bouncing Balls: eight hits on the same ball score the ceiling's k-th value each, it climbs and it caps",
          got47b == [ce47b(k_) for k_ in range(1, 9)] and got47b[0] < got47b[-1] and got47b[-1] == got47b[-2],
          f"scored {got47b}")

    rd47 = _RD47()
    rd47._rng.seed(47)
    t47 = 10.0
    rd47.start(context(now=t47))
    while rd47.state != "live" and t47 < 30.0:
        t47 += TICK47
        rd47.advance(context(now=t47))
    o_rd47 = rd47.shot(_ev47(rd47.x, rd47.y, rd47.shown_at, 1), context(now=t47))
    check("Reaction: the ceiling is what a hit at zero reaction time scored (the scorer's own function)",
          rd47.state == "wait" and o_rd47.points == rd47.ghost_ceiling()(1) == _rp47(0.0) > 0,
          f"{o_rd47.points}")
    check("...and a slower hit scores no more than the ceiling (it is a MAXIMUM)",
          _rp47(0.55) <= rd47.ghost_ceiling()(1) and _rp47(5.0) >= 1)

    for cls_, read_ in ((_P47, lambda m: m.first_points[0]), (_I47, lambda m: m.score_total),
                        (_A47, lambda m: m.points)):
        m_, _th, _pt = _figure_hit_case(cls_)
        check(f"{cls_.title}: the ceiling is what the scorer gave a hit at the best ring's centre",
              read_(m_) == m_.ghost_ceiling()(1) > 0, f"{read_(m_)}")

    an47 = _AN47()
    an47.start(context(now=0.0))
    T47 = _t_of42(an47, FR42)
    oa47 = an47.shot(_ev47(*_pt42(an47, FR42, "mass"), T47), context(now=T47))
    check("Animated Target: the ceiling is what a hit on him scored",
          oa47.hit and oa47.points == an47.ghost_ceiling()(1) > 0, f"{oa47.points}")

    # -- 3. shots-per-target is what it ACTUALLY takes to drop one --------------
    def _shots_to_drop47(drive):
        n_ = 0
        while n_ < 40 and not drive(n_):
            n_ += 1
        return n_ + 1

    mi47 = _I47()
    mi47._rng.seed(5)
    mi47.start(context(now=40.0), figure="thug01")
    pti47 = _zone_point(mi47)
    n_i47 = _shots_to_drop47(lambda n: (mi47.shot(_ev47(*pti47, 50.0 + n * 0.3, n + 1),
                                                  context(now=50.0 + n * 0.3)), mi47.dropped)[1] > 0)
    mp47 = _P47()
    mp47._rng.seed(11)
    mp47.start(context(now=40.0), figure="thug01")
    _drive_to_live(mp47, 40.0)
    ptp47 = _zone_point(mp47)
    n_p47 = _shots_to_drop47(lambda n: (mp47.shot(_ev47(*ptp47, mp47.shown_at + 0.3 + n * 0.05, n + 1),
                                                  context(now=mp47.shown_at + 0.3 + n * 0.05)),
                                        mp47.dropped)[1] > 0)
    ma47 = _A47()
    ma47._rng.seed(11)
    ma47.start(context(now=40.0), figure="thug01")
    _drive_to_live(ma47, 40.0)
    aa47 = ma47._actors[0]
    th47 = aa47.t0 + aa47.t_a + 0.1
    ci47, hi47 = aa47.state(th47)
    ina47 = aa47.fig.innermost(ci47, hi47)
    n_a47 = _shots_to_drop47(lambda n: (ma47.shot(_ev47(ina47[1], ina47[2], th47, n + 1), context(now=th47)),
                                        ma47.dropped)[1] > 0)
    anx47 = _AN47()
    anx47.start(context(now=0.0))
    n_an47 = _shots_to_drop47(lambda n: (anx47.shot(_ev47(*_pt42(anx47, FR42, "mass"), _t_of42(anx47, FR42), n + 1),
                                                    context(now=_t_of42(anx47, FR42))),
                                         anx47._state != "walk")[1])
    check("a figure drops after exactly ghost_shots_per_target() perfect hits: Static Target, Pop-Up, Moving Target, "
          "Animated Target",
          (n_i47, n_p47, n_a47, n_an47) == (mi47.ghost_shots_per_target(), mp47.ghost_shots_per_target(),
                                            ma47.ghost_shots_per_target(), anx47.ghost_shots_per_target()),
          f"{(n_i47, n_p47, n_a47, n_an47)}")
    check("...and the figure value times that many shots reaches the drop total, one fewer does not "
          "(derived, not retyped)",
          all((s_ * c_ >= _fig47.DROP_TOTAL > (s_ - 1) * c_)
              for s_, c_ in ((mi47.ghost_shots_per_target(), mi47.ghost_ceiling()(1)),
                             (mp47.ghost_shots_per_target(), mp47.ghost_ceiling()(1)),
                             (ma47.ghost_shots_per_target(), ma47.ghost_ceiling()(1)))))
    real_top47 = _fig47.cast_top_points()
    _orig_names47, _orig_load47 = _fig47.figure_names, _fig47.Z.load_zones
    try:
        _fig47.figure_names = lambda folder=None: ["a", "b"]
        _fig47.Z.load_zones = lambda n, folder=None: {"zones": [{"points": 10 if n == "a" else 9}]}
        split47 = _fig47.cast_top_points()
        split_ceiling47 = (_I47().ghost_ceiling(), _I47().ghost_shots_per_target())
        _fig47.Z.load_zones = lambda n, folder=None: {"zones": [{"points": 10}, {"points": 7}]}
        agree47 = _fig47.cast_top_points()
        _fig47.figure_names = lambda folder=None: []
        empty47 = _fig47.cast_top_points()
    finally:
        _fig47.figure_names, _fig47.Z.load_zones = _orig_names47, _orig_load47
    check("the figure ceiling is the best zone every figure agrees on: the real cast has one; a cast whose "
          "figures DISAGREE about the top ring, or an empty one, has no single honest ceiling and says None "
          "(so the drill is NOT SCORABLE), and a unanimous cast answers",
          real_top47 is not None and real_top47 == mi47.ghost_ceiling()(1)
          and split47 is None and split_ceiling47 == (None, None)
          and agree47 == 10 and empty47 is None,
          f"real {real_top47}, split {split47}, agree {agree47}, empty {empty47}")

    # -- 4. REACTION: the waits, on the shots' clock, independent of the human ---
    def _play_reaction47(seed, react_s, n_hits, latency=0.07):
        m_ = _RD47()
        m_._rng.seed(seed)
        t_ = 100.0
        m_.start(context(now=t_))
        shown_, hit_ts_, outs_, asks_ = [], [], [], []
        last_ = None
        guard_ = 0
        while len(hit_ts_) < n_hits and guard_ < 100000:
            guard_ += 1
            t_ += TICK47
            asks_.append(bool(m_.advance(context(now=t_))))
            if m_.state == "live":
                if m_.shown_at != last_:
                    last_ = m_.shown_at
                    shown_.append(m_.shown_at)
                if t_ - m_.shown_at >= react_s:
                    # The shot's own time is t_; it is HANDLED `latency` later, as in the app.
                    outs_.append(m_.shot(_ev47(m_.x, m_.y, t_, len(hit_ts_) + 1),
                                         context(now=t_ + latency)))
                    hit_ts_.append(t_)
        return m_, shown_, hit_ts_, outs_, asks_

    LAT47 = 0.07
    fast47, shown_f47, hits_f47, outs_f47, asks_f47 = _play_reaction47(47, 0.4, 6, LAT47)
    check("Reaction records one target per appearance (six), nothing per frame",
          fast47.target_count == 6 == len(outs_f47), f"{fast47.target_count}")
    check("each recorded appearance is the instant the mode itself showed it, and every one lies inside the "
          "virtual clock the shots use (a wall-clock or monotonic read would be billions off)",
          [a_ for a_, _r in fast47._targets] == shown_f47
          and all(100.0 <= a_ <= hits_f47[-1] for a_ in shown_f47))
    check("each wait began at the SHOT'S OWN time (ev.t), not at the later instant it was handled; the first "
          "target waited for nothing",
          [r_ for _a, r_ in fast47._targets] == [None] + hits_f47[:-1])
    w_f47 = fast47.ghost_waits(6)
    want_f47 = [0.0] + [shown_f47[i_] - hits_f47[i_ - 1] for i_ in range(1, 6)]
    check("ghost_waits(6) is exactly appearance minus the previous resolving shot",
          w_f47 is not None and all(abs(a_ - b_) < 1e-9 for a_, b_ in zip(w_f47, want_f47)), str(w_f47))
    check("every wait is inside the drill's own draw (MIN_WAIT..MAX_WAIT) plus the handling latency",
          all(_RD47.MIN_WAIT + LAT47 - 1e-9 <= w_ <= _RD47.MAX_WAIT + LAT47 + 2 * TICK47 for w_ in w_f47[1:]),
          str([round(w_, 2) for w_ in w_f47]))
    slow47, shown_s47, hits_s47, _o, _a = _play_reaction47(47, 1.3, 6, LAT47)
    w_s47 = slow47.ghost_waits(6)
    check("A SLOW HUMAN DOES NOT SLOW THE GHOST: the same seed with a 1.3 s reaction instead of 0.4 s reaches the "
          "last target 4+ s later, and its waits are the same to within a camera tick",
          shown_s47[-1] - shown_f47[-1] > 4.0
          and all(abs(a_ - b_) <= 1.5 * TICK47 for a_, b_ in zip(w_f47, w_s47)),
          f"last target {shown_f47[-1] - 100:.1f}s vs {shown_s47[-1] - 100:.1f}s")
    ce_r47 = fast47.ghost_ceiling()
    av_r47 = fast47.ghost_available_at(6, 60.0)
    hp47 = sum(o_.points for o_ in outs_f47)
    he47 = hits_f47[-1] - hits_f47[0]
    common47 = dict(human_points=hp47, human_elapsed=he47, shots=6, ceiling=ce_r47,
                    has_waits=fast47.ghost_has_waits)
    real47 = _g47.ghost_score(available_at=av_r47, **common47)
    bare47 = _g47.ghost_score(available_at=None, **common47)
    check("Reaction WITH its real waits reads in the hundreds and is NOT a lower bound; WITHOUT them it is the "
          "flagged tens (the 47 he saw)",
          av_r47 is not None and real47.scorable and not real47.lower_bound and 400 <= real47.shown <= 800
          and bare47.lower_bound and bare47.shown < 150 and bare47.shown * 3 < real47.shown,
          f"with {real47.shown}, without {bare47.shown}+")
    exp_r47 = fast47.ghost_expected_wait(6)
    w7_r47 = fast47.ghost_waits(7)
    # CHANGED 2026-10-23 (the ghost was beatable by MISSING; see Mode._ghost_wait_plan). These five
    # checks used to assert that a target the round never drew was priced at the drill's declared
    # average wait and COUNTED as declared. That was the defect: the ghost was charged a wait the round
    # never contained, so a human who missed outscored it. They now assert the opposite, on the same
    # rounds: the unsupplied target costs the ghost NOTHING, no wait is ever declared, and the
    # measured waits are untouched.
    exp_r47 = fast47.ghost_expected_wait(6)
    w7_r47 = fast47.ghost_waits(7)
    check("a round with MORE shots than targets (he missed, so the ghost needs a target that was never drawn): "
          "the ghost is charged NOTHING for it (0.0, never the drill's declared mean), the measured six are "
          "untouched, and NO wait is declared",
          exp_r47 is not None and abs(exp_r47 - (_RD47.MIN_WAIT + _RD47.MAX_WAIT) / 2.0) < 1e-12
          and w7_r47 is not None and len(w7_r47) == 7 and w7_r47[:6] == w_f47 and w7_r47[6] == 0.0
          and w7_r47[6] != exp_r47
          and fast47.ghost_declared_waits(7) == 0 and fast47.ghost_available_at(7, 60.0) is not None,
          f"{w7_r47}")
    check("...and a perfect round (six shots, six targets) declares NOTHING either: every wait is the round's own",
          fast47.ghost_declared_waits(6) == 0 and fast47.ghost_declared_waits(1) == 0)
    miss7_47 = _g47.ghost_score(available_at=fast47.ghost_available_at(7, 60.0),
                                declared_waits=fast47.ghost_declared_waits(7), **dict(common47, shots=7))
    check("...so the score is neither a lower bound nor an estimate: no declared wait on either round",
          miss7_47.scorable and not miss7_47.lower_bound and miss7_47.declared_waits == 0
          and real47.declared_waits == 0, f"{miss7_47.shown}, declared {miss7_47.declared_waits}")
    # The case the brief names: a 17-shot magazine, five misses -> twelve targets drawn, the
    # ghost needs seventeen. The five extra shots are fired at the ghost's cadence, no wait.
    m17_47, shown17_47, hits17_47, outs17_47, _a17 = _play_reaction47(47, 0.4, 12, LAT47)
    he17_47 = (hits17_47[-1] - hits17_47[0]) + 5 * 1.0       # five more shots at about a second each
    common17_47 = dict(human_points=sum(o_.points for o_ in outs17_47), human_elapsed=he17_47, shots=17,
                       ceiling=m17_47.ghost_ceiling(), has_waits=m17_47.ghost_has_waits)
    old17_47 = _g47.ghost_score(available_at=None, **common17_47)
    new17_47 = _g47.ghost_score(available_at=m17_47.ghost_available_at(17, 60.0),
                                declared_waits=m17_47.ghost_declared_waits(17), **common17_47)
    # The ghost's time re-derived by a route that shares nothing with the plan: the twelve recorded
    # waits read straight off `_targets`, then five shots with nothing to wait for.
    hand17_47 = ([0.0] + [0.0 if r_ is None else max(0.0, a_ - r_) for a_, r_ in m17_47._targets[1:]]
                  + [0.0] * 5)
    hand_el17_47 = _g47.ghost_elapsed(17, _g47.available_from_waits(hand17_47, 60.0))[0]
    check("a 17-shot round with five misses (twelve targets drawn): NO wait is declared, the ghost's time is "
          "the twelve measured waits plus five shots at cadence (re-derived from _targets), it is no lower "
          "bound, it cannot exceed the human (speed share <= 1), and it still reads far above the flagged tens",
          m17_47.target_count == 12 and m17_47.ghost_declared_waits(17) == 0
          and new17_47.declared_waits == 0 and not new17_47.lower_bound
          and new17_47.ghost_elapsed is not None and abs(new17_47.ghost_elapsed - hand_el17_47) < 1e-9
          and new17_47.speed_raw is not None and new17_47.speed_raw <= 1.0 + 1e-9
          and new17_47.clamped == ()          # the RAW share: speed_share is clamped at 1.0 and proves nothing
          and new17_47.shown is not None and 0 < new17_47.shown <= 1000
          and old17_47.lower_bound and old17_47.shown < 150 and old17_47.shown * 3 < new17_47.shown,
          f"was {old17_47.shown}+, now {new17_47.shown}, ghost {new17_47.ghost_elapsed}s "
          f"vs by hand {hand_el17_47:.3f}s, human {he17_47:.1f}s")

    class _Mute47(_M47):                       # waits, a ceiling, and NO declared wait
        ghost_has_waits = True

        def ghost_ceiling(self):
            return _g47.constant_ceiling(10)

    mu47_ = _Mute47()
    for k_ in range(4):
        mu47_.note_target(float(k_) * 3.0, None if k_ == 0 else float(k_) * 3.0 - 1.0)
    mute_s47 = _g47.ghost_score(available_at=mu47_.ghost_available_at(5, 60.0), declared_waits=0,
                                human_points=20, human_elapsed=6.0, shots=5, ceiling=mu47_.ghost_ceiling(),
                                has_waits=True)
    check("a drill that CANNOT declare a wait is scored exactly like one that can (2026-10-23: the declaration "
          "is no longer consulted at all): the fifth shot of four targets waits 0.0, the four measured waits "
          "are untouched, nothing is declared, and it is no lower bound",
          mu47_.ghost_expected_wait(4) is None and mu47_.ghost_waits(5) is not None
          and len(mu47_.ghost_waits(5)) == 5 and mu47_.ghost_waits(5)[:4] == mu47_.ghost_waits(4)
          and mu47_.ghost_waits(5)[4] == 0.0
          and mu47_.ghost_available_at(5, 60.0) is not None and mu47_.ghost_declared_waits(5) == 0
          and not mute_s47.lower_bound and mute_s47.declared_waits == 0)
    check("an fps that is not a rate cannot give the one-frame floor, so no times are supplied (lower bound)",
          fast47.ghost_available_at(6, 0) is None and fast47.ghost_available_at(6, None) is None)
    restart_cnt47 = fast47.target_count
    fast47.start(context(now=500.0))
    check("starting the round again forgets the old targets (nothing accumulates across rounds)",
          restart_cnt47 == 6 and fast47.target_count == 0)

    # -- 5. CLEAR THE BOARD: the 2.2 s between boards ----------------------------
    cb2_47 = ClearTheBoard(count=12)
    cb2_47.start(context(now=200.0))
    check("a fresh board records its twelve circles at the dealing instant, none of them waited for",
          cb2_47.target_count == 12 and all(a_ == 200.0 and r_ is None for a_, r_ in cb2_47._targets))
    tl47 = 200.0
    for i_, tg_ in enumerate(list(cb2_47.targets)):
        tl47 = 201.0 + i_ * 0.3
        cb2_47.shot(_ev47(tg_.x, tg_.y, tl47, i_ + 1), context(now=tl47 + 0.05))
    tt47 = tl47
    while cb2_47.target_count == 12 and tt47 < tl47 + 6.0:
        tt47 += TICK47
        cb2_47.advance(context(now=tt47))
    w_cb47 = cb2_47.ghost_waits(24)
    check("the second board is recorded: 24 targets, the first circle of it released by the LAST HIT of the first "
          "(its ev.t), the other eleven waiting for nothing",
          cb2_47.target_count == 24 and cb2_47._targets[12] == (tt47, tl47)
          and all(r_ is None for _a, r_ in cb2_47._targets[13:]))
    check("the wait between boards is NEXT_BOARD_S (to within a tick), once, and every other shot waits 0",
          w_cb47 is not None and cb2_47.NEXT_BOARD_S <= w_cb47[12] <= cb2_47.NEXT_BOARD_S + 1.5 * TICK47
          and all(w_ == 0.0 for i_, w_ in enumerate(w_cb47) if i_ != 12), str(w_cb47[10:14]))
    w25_cb47 = cb2_47.ghost_waits(25)
    check("...and one shot more than the two boards held (he missed): the missing target is the FIRST of a "
          "third board that this round NEVER DEALT, so the ghost is charged nothing for it (not NEXT_BOARD_S: "
          "2026-10-23), the measured 24 are untouched, and nothing is declared",
          w25_cb47 is not None and len(w25_cb47) == 25 and w25_cb47[:24] == w_cb47
          and w25_cb47[24] == 0.0 and w25_cb47[24] != cb2_47.NEXT_BOARD_S
          and cb2_47.ghost_declared_waits(25) == 0 and cb2_47.ghost_declared_waits(24) == 0)
    cb2_47.start(context(now=300.0))
    check("restarting Clear The Board does not stack a third board on the old two", cb2_47.target_count == 12)

    # -- 6. POP-UP: the wait after a drop, and after an unengaged figure ---------
    mq47 = _P47()
    mq47._rng.seed(23)
    mq47.start(context(now=300.0), figure="thug01")
    _drive_to_live(mq47, 300.0)
    spt47 = mq47.ghost_shots_per_target()
    app_q47, rel_q47 = [mq47.shown_at], []
    for _fi in range(2):
        ptq47 = _zone_point(mq47)
        base47 = mq47.shown_at + 0.3
        for i_ in range(spt47):
            ts47 = base47 + i_ * 0.05
            mq47.shot(_ev47(*ptq47, ts47, i_ + 1), context(now=ts47))
        rel_q47.append(ts47)
        _drive_to_live(mq47, ts47 + TICK47)
        app_q47.append(mq47.shown_at)
    check("Pop-Up: three figures recorded, each (after the first) released by the shot that dropped the one before",
          mq47._targets == ((app_q47[0], None), (app_q47[1], rel_q47[0]), (app_q47[2], rel_q47[1]))
          and mq47.dropped == 2, str(mq47._targets))
    wq47 = mq47.ghost_waits(3 * spt47)
    check("the ghost waits before the FIRST shot at each figure and not before the others on it; each wait is the "
          "drill's own draw (MIN_WAIT..MAX_WAIT)",
          wq47 is not None and len(wq47) == 3 * spt47
          and all(w_ == 0.0 for i_, w_ in enumerate(wq47) if i_ % spt47)
          and all(_P47.MIN_WAIT <= wq47[i_] <= _P47.MAX_WAIT + 2 * TICK47 for i_ in (spt47, 2 * spt47)),
          str([round(w_, 2) for w_ in wq47]))
    wq4_47 = mq47.ghost_waits(3 * spt47 + 1)
    check("one shot more than the three figures can take starts a fourth figure that was never drawn: the ghost "
          "is charged NOTHING for it (0.0, not the drill's mean of MIN_WAIT..MAX_WAIT: 2026-10-23), and "
          "nothing is declared",
          wq4_47 is not None and len(wq4_47) == 3 * spt47 + 1 and wq4_47[:3 * spt47] == wq47
          and wq4_47[-1] == 0.0
          and mq47.ghost_declared_waits(3 * spt47 + 1) == 0 and mq47.ghost_declared_waits(3 * spt47) == 0)
    check("...a missing figure that takes SEVERAL shots waits for none of them: every shot on it is 0.0, "
          "and nothing is declared",
          mq47.ghost_declared_waits(4 * spt47) == 0 and mq47.ghost_waits(4 * spt47)[3 * spt47:] == [0.0] * spt47)
    mu47 = _P47()
    mu47._rng.seed(29)
    mu47.start(context(now=400.0), figure="thug01")
    _drive_to_live(mu47, 400.0)
    left47 = mu47.left_at
    tu47, seen_wait47 = mu47.shown_at, False
    while tu47 < mu47.shown_at + 25.0:
        tu47 += TICK47
        mu47.advance(context(now=tu47))
        seen_wait47 = seen_wait47 or mu47.state == "wait"
        if seen_wait47 and mu47.state == "up":
            break
    check("Pop-Up, a figure nobody engaged: the next wait began at the TICK that took it away (its exposure end), "
          "not at a shot",
          mu47.unengaged == 1 and mu47.target_count == 2
          and 0.0 <= mu47._targets[1][1] - left47 <= 1.5 * TICK47
          and _P47.MIN_WAIT <= mu47._targets[1][0] - mu47._targets[1][1] <= _P47.MAX_WAIT + 2 * TICK47,
          str(mu47._targets))

    # -- 7. STATIC TARGET: the fall IS the wait ----------------------------------
    fall_name47 = next((n_ for n_ in _fig47.figure_names() if _fig47.catalog.fall_path(n_) is not None), "thug01")
    mj47 = _I47()
    mj47._rng.seed(37)
    mj47.start(context(now=600.0), figure=fall_name47)
    has_fall47 = mj47._fig.has_fall
    ptj47 = _zone_point(mj47)
    TJ47 = 610.0
    nj47 = 0
    while mj47.dropped == 0 and nj47 < 20:
        mj47.shot(_ev47(*ptj47, TJ47, nj47 + 1), context(now=TJ47))
        nj47 += 1
    tj47 = TJ47
    while mj47.target_count == 1 and tj47 < TJ47 + 5.0:
        tj47 += TICK47
        mj47.advance(context(now=tj47))
    wj47 = mj47.ghost_waits(2 * mj47.ghost_shots_per_target())
    exp_j47 = _fig47.FALL_S if has_fall47 else 0.0
    check("Static Target: the figure that replaces a dropped one was released by the dropping shot's own time, and "
          "waited the fall (FALL_S) -- or nothing, for a figure with no fall pose (the figure under test "
          f"{'has' if has_fall47 else 'has NO'} a pose)",
          mj47.target_count == 2 and mj47._targets[1][1] == TJ47 and wj47 is not None
          and abs(wj47[mj47.ghost_shots_per_target()] - exp_j47) <= 1.5 * TICK47,
          str(mj47._targets))

    # -- 8. MOVING TARGET: gap, concurrent with the fall -------------------------
    mv47 = _A47()
    mv47._rng.seed(31)
    mv47.start(context(now=500.0), figure=fall_name47)
    _drive_to_live(mv47, 500.0)
    av47 = mv47._actors[0]
    app1_47 = av47.t0
    th2_47 = av47.t0 + av47.t_a + 0.1
    cm47, hm47 = av47.state(th2_47)
    inm47 = av47.fig.innermost(cm47, hm47)
    fall_v47 = av47.fig.has_fall
    for i_ in range(mv47.ghost_shots_per_target()):
        mv47.shot(_ev47(inm47[1], inm47[2], th2_47, i_ + 1), context(now=th2_47))
    _drive_to_live(mv47, th2_47 + TICK47)
    app2_47 = mv47._actors[0].t0
    wv47 = app2_47 - th2_47
    check("Moving Target: the second walk was released by the dropping shot and appeared after the gap AND, "
          "when there is a fall pose, no sooner than it",
          mv47.dropped == 1 and mv47._targets == ((app1_47, None), (app2_47, th2_47))
          and wv47 >= _A47.GAP_S[0] - 1e-9 and (not fall_v47 or wv47 >= _fig47.FALL_S - 1e-9)
          and wv47 <= max(_A47.GAP_S[1], _fig47.FALL_S if fall_v47 else 0.0) + 2 * TICK47,
          f"wait {wv47:.2f}s, fall {fall_v47}")
    old_count47 = _A47.COUNT
    _A47.COUNT = 2
    try:
        two47 = _A47()
        check("with COUNT = 2 (OFF today) the chain is not one chain: it says it cannot state a ceiling",
              two47.ghost_ceiling() is None and two47.ghost_shots_per_target() is None)
    finally:
        _A47.COUNT = old_count47

    # -- 9. ANIMATED: the fall clip, then the rest --------------------------------
    am47 = _AN47()
    am47.start(context(now=0.0))
    Tm47 = _t_of42(am47, FR42)
    nm47 = 0
    while am47._state == "walk" and nm47 < 40:
        am47.shot(_ev47(*_pt42(am47, FR42, "mass"), Tm47, nm47 + 1), context(now=Tm47))
        nm47 += 1
    fall47 = am47.fall_sequence()
    nfall47 = len(fall47) if fall47 else 0
    tm47 = Tm47
    while am47._state != "walk" and tm47 < Tm47 + 30.0:
        tm47 += TICK47
        am47.advance(context(now=tm47))
    wm47 = tm47 - Tm47
    check("Animated Target: the next soldier was released by the killing shot's own time and appeared after the "
          "fall clip PLUS the rest, read from the clip and the class constant",
          am47._targets == ((0.0, None), (tm47, Tm47))
          and nfall47 / _AN47.FPS + _AN47.REST_S - 1e-9 <= wm47 <= nfall47 / _AN47.FPS + _AN47.REST_S + 3 * TICK47,
          f"waited {wm47:.2f}s for a {nfall47}-frame fall")

    # -- 10. the drills that make nobody wait say so, and the guard that keeps it true
    check("Static Targets, Escalating Ball and Bouncing Balls declare NO waits and supply no times",
          not _ST47.ghost_has_waits and not EscalatingBall.ghost_has_waits and not _BB47.ghost_has_waits
          and st47.ghost_available_at(17, 60.0) is None and eb47.ghost_available_at(17, 60.0) is None
          and bb47.ghost_available_at(17, 60.0) is None)
    cg47 = context(now=3.0)
    gb47 = EscalatingBall()
    gb47.start(cg47)
    before47 = (gb47.x, gb47.y)
    gb47.shot(_ev47(gb47.x, gb47.y, 3.0), cg47)
    gbb47 = _BB47()
    gbb47.start(cg47)
    hb47 = (gbb47.balls[0].x, gbb47.balls[0].y)
    born_before47 = gbb47.balls[0].born
    gbb47.shot(_ev47(hb47[0], hb47[1], 4.0), context(now=4.0))
    check("...and that is TRUE: the replacement ball is on screen inside the very call that scored the hit "
          "(it moved, and was born at that call's own instant, not the previous one), so nothing waits",
          math.hypot(gb47.x - before47[0], gb47.y - before47[1]) > 1.0
          and born_before47 == 3.0 and gbb47.balls[0].born == 4.0
          and math.hypot(gbb47.balls[0].x - hb47[0], gbb47.balls[0].y - hb47[1]) > 1.0)
    check("every other drill that makes you wait declares it",
          all(c_.ghost_has_waits for c_ in (ClearTheBoard, _RD47, _P47, _I47, _A47, _AN47)))

    # -- 11. the record is bounded, and a restart never stacks ------------------
    fl47 = ClearTheBoard(count=12)
    for i_ in range(_MAGMAX47 + 25):
        fl47.note_target(float(i_), None)
    check("the appearance list is bounded by the largest magazine and keeps the FIRST ones",
          fl47.target_count == _MAGMAX47 and fl47._targets[0] == (0.0, None)
          and fl47._targets[-1] == (float(_MAGMAX47 - 1), None))
    stack47 = []
    for cls_ in drills47:
        m_ = cls_()
        c0_ = context(now=100.0)
        m_.start(c0_)
        fresh_ = m_.target_count
        t_ = 100.0
        for _k in range(540):                    # nine seconds: past the longest first wait
            t_ += TICK47
            m_.advance(context(now=t_))
        grown_ = m_.target_count
        m_.start(context(now=t_ + 1.0))
        if m_.target_count != fresh_:
            stack47.append((cls_.name, fresh_, grown_, m_.target_count))
    check("every drill's start() leaves exactly the targets a fresh start leaves (none stacked from the last "
          "round), after nine seconds of play", not stack47, str(stack47))

    # -- 12. NO EXTRA REPAINTS: the same scripted run, with the recorder switched off -----
    def _asks47(cls_):
        m_ = cls_()
        if cls_ in (_RD47, _P47, _I47, _A47):    # the drills that draw at random: same draws both runs
            m_._rng.seed(3)
        t_ = 100.0
        m_.start(context(now=t_))
        out_ = []
        for _k in range(900):
            t_ += TICK47
            out_.append(bool(m_.advance(context(now=t_))))
        return out_

    real_nt47 = _M47.note_target
    drilled47 = (ClearTheBoard, _RD47, _P47, _I47, _A47, _AN47)
    with_rec47 = {c_.name: _asks47(c_) for c_ in drilled47}
    _M47.note_target = lambda self, appeared_at, released_at=None: None
    try:
        without_rec47 = {c_.name: _asks47(c_) for c_ in drilled47}
    finally:
        _M47.note_target = real_nt47
    check("recording an appearance changes no mode's repaint requests: over 900 ticks (15 s) each of the six "
          "waiting drills asks for exactly the same repaints with the recorder switched off",
          with_rec47 == without_rec47,
          str([n_ for n_ in with_rec47 if with_rec47[n_] != without_rec47[n_]]))
    check("...and the three that sit still while waiting ask for almost none over those 900 ticks (Reaction, "
          "Pop-Up, Static Target): 25 at the very most, against the 5-in-180 idle bar (~25 in 900)",
          all(sum(with_rec47[n_]) <= 5 * 5 for n_ in ("reaction", "popup", "image")),
          str({n_: sum(with_rec47[n_]) for n_ in ("reaction", "popup", "image")}))

    # ------------------------------------------------------------------
    # 48. THE GHOST'S DECLARED WAITS (2026-10-21). A round the human missed shots in has fewer
    # targets than the ghost needs. Each drill that makes the shooter wait states its OWN
    # expected wait for a target nobody drew, derived from the constants its runtime
    # uses. Every expected value below is COMPUTED from those constants here, by a route
    # that shares nothing with the drill's own arithmetic; none is a typed literal.
    # ------------------------------------------------------------------
    print()
    print("48. the ghost's declared waits: each drill states its own, from its own constants")

    def _mean_max48(lo, hi, fl, n=20000):
        return sum(max(lo + (hi - lo) * (k + 0.5) / n, fl) for k in range(n)) / n

    check("Reaction declares the mean of its own draw: (MIN_WAIT + MAX_WAIT) / 2",
          abs(_RD47().ghost_expected_wait(5) - (_RD47.MIN_WAIT + _RD47.MAX_WAIT) / 2.0) < 1e-12)
    check("Pop-Up declares the mean of its own draw: (MIN_WAIT + MAX_WAIT) / 2",
          abs(_P47().ghost_expected_wait(5) - (_P47.MIN_WAIT + _P47.MAX_WAIT) / 2.0) < 1e-12)
    check("...and Pop-Up's runtime really is a bare draw (the fall adds nothing because MIN_WAIT exceeds "
          "it): the declaration would be wrong the day that stops being true",
          _P47.MIN_WAIT >= _fig47.FALL_S)
    falls48 = [_fig47.FALL_S if _fig47.catalog.fall_path(n_) is not None else 0.0
               for n_ in _fig47.figure_names()]
    want_a48 = sum(_mean_max48(_A47.GAP_S[0], _A47.GAP_S[1], f_) for f_ in falls48) / len(falls48)
    got_a48 = _A47().ghost_expected_wait(3)
    check("Moving Target declares E[max(GAP_S draw, fall)] averaged over the cast, checked by integrating the "
          "draw over GAP_S and FALL_S (not the plain GAP_S mean: the next figure never appears before the fall)",
          got_a48 is not None and abs(got_a48 - want_a48) < 1e-3
          and got_a48 >= (_A47.GAP_S[0] + _A47.GAP_S[1]) / 2.0 - 1e-9
          and _A47.GAP_S[0] <= got_a48 <= max(_A47.GAP_S[1], _fig47.FALL_S),
          f"declared {got_a48}, integrated {want_a48:.4f} over {len(falls48)} figures")
    want_i48 = sum(falls48) / len(falls48)
    got_i48 = _I47().ghost_expected_wait(3)
    check("Static Target declares the fall averaged over the cast (FALL_S for a figure with a pose, nothing "
          "for one without), and it lies between nothing and FALL_S",
          got_i48 is not None and abs(got_i48 - want_i48) < 1e-12 and 0.0 <= got_i48 <= _fig47.FALL_S,
          f"declared {got_i48}")
    cb48 = ClearTheBoard(count=5)
    check("Clear The Board declares NEXT_BOARD_S for the first target of each new board and nothing for the "
          "other circles of it, with the board size read from the drill (5 here), NOT a flat 2.2 s per shot",
          cb48.ghost_expected_wait(0) == 0.0 and cb48.ghost_expected_wait(4) == 0.0
          and cb48.ghost_expected_wait(5) == cb48.NEXT_BOARD_S and cb48.ghost_expected_wait(6) == 0.0
          and cb48.ghost_expected_wait(10) == cb48.NEXT_BOARD_S)
    check("Animated Target declares the fall clip's length at FPS plus REST_S, read the way advance() reads "
          "them, and it agrees with the wait that drill actually made in section 9 to within a few ticks",
          abs(am47.ghost_expected_wait(3) - (nfall47 / _AN47.FPS + _AN47.REST_S)) < 1e-12
          and abs(am47.ghost_expected_wait(3) - wm47) <= 3 * TICK47 + 1e-9,
          f"declared {am47.ghost_expected_wait(3):.3f}, measured {wm47:.3f}")
    wa48 = am47.ghost_waits(3 * am47.ghost_shots_per_target())
    # CHANGED 2026-10-23. These two used to assert that a soldier / figure the round never drew was
    # charged the drill's declared fall-and-rest (one declared wait). That is the defect that let a
    # human who MISSED outscore the ghost (a stored 1146 of 1000 on Animated Target): the ghost paid a
    # full fall for a soldier the human never dropped. The unsupplied target is now free.
    check("Animated Target, a third soldier never drawn: the ghost is charged NOTHING for him (0.0, not the "
          "declared fall-and-rest), and nothing is declared",
          wa48 is not None and wa48[2 * am47.ghost_shots_per_target()] == 0.0
          and wa48[2 * am47.ghost_shots_per_target()] != am47.ghost_expected_wait(2)
          and am47.ghost_declared_waits(3 * am47.ghost_shots_per_target()) == 0
          and am47.ghost_declared_waits(2 * am47.ghost_shots_per_target()) == 0)
    wi48 = mj47.ghost_waits(3 * mj47.ghost_shots_per_target())
    check("Static Target, a third figure never drawn: the ghost is charged nothing for it (0.0, not the mean "
          "fall), and nothing is declared",
          wi48 is not None and wi48[2 * mj47.ghost_shots_per_target()] == 0.0
          and (want_i48 == 0.0 or wi48[2 * mj47.ghost_shots_per_target()] != want_i48)
          and mj47.ghost_declared_waits(3 * mj47.ghost_shots_per_target()) == 0
          and mj47.ghost_declared_waits(2 * mj47.ghost_shots_per_target()) == 0)
    check("every drill that makes the shooter wait states an expected wait that is a finite number >= 0 "
          "(the six that wait today: a seventh that forgets would stay a lower bound, and this names it)",
          all((lambda w_: w_ is not None and math.isfinite(w_) and w_ >= 0.0)(c_().ghost_expected_wait(1))
              for c_ in (ClearTheBoard, _RD47, _P47, _I47, _A47, _AN47)))
    check("a drill that makes nobody wait (Static Targets, both ball drills) declares nothing, supplies no "
          "waits and carries no declared count: unaffected",
          all(m_.ghost_expected_wait(3) is None and m_.ghost_waits(17) is None
              and m_.ghost_declared_waits(17) == 0 for m_ in (st47, eb47, bb47)))
    old_count48 = _A47.COUNT
    _A47.COUNT = 2
    try:
        two48 = _A47()
        two48.note_target(0.0, None)
        two48.note_target(5.0, 4.0)
        check("Moving Target with COUNT = 2 still cannot declare: no expected wait, no waits even for a "
              "missing target, no declared count (it stays unscorable, as before)",
              two48.ghost_expected_wait(3) is None and two48.ghost_waits(4) is None
              and two48.ghost_declared_waits(4) == 0 and two48.ghost_ceiling() is None)
    finally:
        _A47.COUNT = old_count48
    # STRUCTURAL, not textual: an earlier form searched for the literal text
    # "GAP_S = ", which `GAP_S: tuple = ...`, `GAP_S=...`, a tuple unpack or
    # `self.GAP_S = ...` all walk straight past. Every binding form is read out of
    # the syntax tree instead (plain, annotated, augmented, tuple/list unpacking,
    # a for target, a walrus, an attribute store, a function parameter default).
    import ast as _ast48
    drill_consts48 = {"MIN_WAIT", "MAX_WAIT", "GAP_S", "NEXT_BOARD_S", "REST_S", "FALL_S"}

    def _bound_names48(tree):
        found = []

        def target(t, line):
            if isinstance(t, _ast48.Name):
                found.append((t.id, line))
            elif isinstance(t, _ast48.Attribute):
                found.append((t.attr, line))
            elif isinstance(t, (_ast48.Tuple, _ast48.List)):
                for e_ in t.elts:
                    target(e_, line)
            elif isinstance(t, _ast48.Starred):
                target(t.value, line)

        for n_ in _ast48.walk(tree):
            if isinstance(n_, _ast48.Assign):
                for t_ in n_.targets:
                    target(t_, n_.lineno)
            elif isinstance(n_, (_ast48.AnnAssign, _ast48.AugAssign)):
                target(n_.target, n_.lineno)
            elif isinstance(n_, (_ast48.For, _ast48.AsyncFor, _ast48.comprehension)):
                target(n_.target, getattr(n_, "lineno", 0))
            elif isinstance(n_, _ast48.NamedExpr):
                target(n_.target, n_.lineno)
            elif isinstance(n_, _ast48.arg):
                found.append((n_.arg, n_.lineno))
            elif isinstance(n_, (_ast48.FunctionDef, _ast48.AsyncFunctionDef, _ast48.ClassDef)):
                found.append((n_.name, n_.lineno))
        return found

    scorer_files48 = (Path(modes.__file__).parent / "base.py",
                      Path(modes.__file__).parent.parent / "ghost.py")
    held48 = [(f_.name, nm_, ln_) for f_ in scorer_files48
              for nm_, ln_ in _bound_names48(_ast48.parse(f_.read_text(encoding="utf-8")))
              if nm_ in drill_consts48]
    check("a drill's expected wait is never retyped in the scorer: ghost.py and base.py bind no drill "
          "constant (MIN_WAIT, MAX_WAIT, GAP_S, NEXT_BOARD_S, REST_S, FALL_S) by ANY assignment form",
          not held48, str(held48))
    probe48 = _bound_names48(_ast48.parse(
        "GAP_S: tuple = (1, 2)\nMIN_WAIT=1\nclass C:\n    REST_S = 3\n    def f(self):\n"
        "        self.FALL_S = 4\n        (a, NEXT_BOARD_S) = 1, 2\n        MAX_WAIT += 1\n"))
    check("... and that walker can fail: it finds the annotated, the spaceless, the class-level, the "
          "attribute, the unpacked and the augmented forms the old text match could not",
          {n_ for n_, _l in probe48} >= drill_consts48, str(sorted({n_ for n_, _l in probe48})))

    # ------------------------------------------------------------------
    # 49. THE GHOST SCORE ON THE ROUND SCREEN, AND THE MACHINE'S BESTS (2026-10-21,
    # piece 3). His instruction: "I only want the score displayed when the round
    # ends. That's it." Everything the arithmetic does is proved in test_ghost
    # section 13 (pure); THIS section is about the seams the pure tests cannot
    # cross: the host filing a round's peaks and pairings, the screen drawing it,
    # the screen staying inert and grey and inside its boxes, and the app writing
    # the file once. Run behind its own crash guard: a raise is a FAILED check, so
    # the mutation runners (which count a suite with no summary as BROKEN) see a
    # catch and not a crash.
    # ------------------------------------------------------------------
    def _section49() -> None:
        print()
        print("49. the ghost score on the round screen: only at the round's end, honest about "
              "what it is, inert and grey, and its bests guarded")
        import copy as _cp49
        import re as _re49
        import zlib as _zl49
        from PySide6.QtGui import QFontMetricsF as _FM49, QPainter as _QP49
        from lsa import bests as _B49, ghost as _G49, scorecard as _SC49
        from lsa.settings import AppSettings as _AS49

        c4k49, c1080_49 = c4_46, c1080_46
        # His real phantom storm's peaks, copied from tests/test_ghost.py section 9
        # (which carries the log line numbers); logs rotate, so this is a COPY.
        STORM49 = [17, 16, 20, 25, 14, 20, 17, 21, 17, 16, 14, 22, 14, 15, 13, 17, 25]

        def _fire49(h_, side_, times_, hit_, peak_):
            m_ = h_.mode if side_ == 0 else h_.mode2
            for i_, t_ in enumerate(times_):
                on_ = hit_[i_] if isinstance(hit_, (list, tuple)) else hit_
                if not on_:
                    x_, y_ = (60.0 if side_ == 0 else 3780.0), 1500.0
                elif h_.versus:
                    x_, y_ = _aim(h_, ctx44, side_)
                elif hasattr(m_, "aim_point"):
                    x_, y_ = m_.aim_point(ctx44)
                else:
                    x_, y_ = m_.x, m_.y
                pk_ = peak_[i_] if isinstance(peak_, (list, tuple)) else peak_
                h_.shot(modes.ShotEvent(x_, y_, t_, i_ + 1, t_end=t_ + 0.05, peak=pk_),
                        context(now=t_))

        def _round49(players=1, mag=4, step=1.0, peak=85.0, hit=True, t0=10.0, store=None,
                     between=None, cls=None, mic=False):
            """A whole round through the host with the REAL hold: fire every side, let the
            caller feed the microphone's verdicts, then let the hold run out."""
            h_ = _Host()
            h_.set_magazine(mag)
            h_.hold_s = REAL_HOLD_S
            h_.mic_ready = mic
            if store is not None:
                h_.bests = _B49.BestBook(store)
            h_.set_mode((cls or _Static)(), ctx44, players=players)
            times_ = [t0 + i_ * step for i_ in range(mag)]
            for s_ in ((0, 1) if players == 2 else (0,)):
                _fire49(h_, s_, times_, hit, peak)
            if between is not None:
                between(h_, times_)
            h_.advance(context(now=times_[-1] + REAL_HOLD_S + 1.0))
            return h_, times_

        # -- 1. the score is on the END screen, and says what the ghost module says ----
        h49, t49 = _round49()
        st49 = h49.tallies[0]
        want49 = _G49.ghost_score(human_points=st49.points, human_elapsed=h49.round_seconds(0),
                                  shots=4, ceiling=h49.mode.ghost_ceiling(), peaks=[85.0] * 4)
        sl49 = h49.score_lines(0)
        check("(premise) a four-round magazine closed, scored points, and the ghost can score it",
              h49.round_over and st49.points > 0 and want49.scorable and want49.shown > 0,
              f"{st49.points} points, shown {want49.shown}")
        check("the end screen's SCORE line is the number the ghost module gives for the same round, "
              "with no `/ 1000` after it, and the block is FOUR lines",
              sl49[0] == f"SCORE  {want49.shown}" and len(sl49) == 4, str(sl49))
        shown49 = _lines44(h49, ctx44)
        check("it is DRAWN: the headline once, and every other line of the block",
              [ln_ for ln_ in shown49 if ln_.startswith("SCORE")] == [sl49[0]]
              and all(ln_ in shown49 for ln_ in sl49), str(sl49))
        check("the two factors are printed beneath it, aim and speed, as the percentages the ghost "
              "module computed",
              sl49[1] == f"AIM {want49.points_share * 100:.0f}%   SPEED {want49.speed_share * 100:.0f}%",
              sl49[1])
        check("the best score and the best efficiency are `... SO FAR  --` when the machine holds no "
              "record (this host has no book), with no GHOST figure and no caveat line",
              sl49[2] == "BEST SCORE SO FAR  --" and sl49[3] == "BEST EFFICIENCY SO FAR  --"
              and not any("GHOST" in ln_ or "NOT A BEST" in ln_ for ln_ in sl49),
              f"{sl49[2]} | {sl49[3]}")
        check("the single-player points line carries its CEILING, the ghost's points for the same round "
              "(the number AIM % is computed from), and the block's other lines are untouched",
              h49.side_block(0) == [f"{st49.shots} shots", f"{st49.hits} hits",
                                    f"{h49.round_misses(0)} " + ("miss" if h49.round_misses(0) == 1 else "misses"),
                                    f"{st49.points} of {want49.ghost_points} points",
                                    h49.side_block(0)[4]]
              and h49.side_block(0)[4].endswith("seconds")
              and f"{st49.points} of {want49.ghost_points} points" in shown49,
              str(h49.side_block(0)))
        real_cards49 = h49.cards
        fallback49 = []
        try:
            for c_ in ((None, None),
                       (dataclass_replace(real_cards49[0],
                                          gs=dataclass_replace(real_cards49[0].gs, scorable=False)), None),
                       (dataclass_replace(real_cards49[0],
                                          gs=dataclass_replace(real_cards49[0].gs, ghost_points=0)), None),
                       (dataclass_replace(real_cards49[0],
                                          gs=dataclass_replace(real_cards49[0].gs, ghost_points=None)), None)):
                h49.cards = c_
                fallback49.append(h49.side_block(0)[3])
        finally:
            h49.cards = real_cards49
        check("...and with NO usable ceiling it falls back to plain `N points`, never raising: no card "
              "(the score failed), an unscorable card, a ghost that took nothing, a ghost with none",
              fallback49 == [f"{st49.points} points"] * 4
              and h49.side_block(0)[3] == f"{st49.points} of {want49.ghost_points} points",
              str(fallback49))
        check("EFFICIENCY is where it was: one figure and its arithmetic, untouched by the score",
              sum(1 for ln_ in shown49 if ln_.startswith("EFFICIENCY")) == 1
              and sum(1 for ln_ in shown49 if ln_.startswith("= ")) == 1)
        check("the screen says nothing about a match, a swap or 'hit factor' because of the score",
              not any(_re49.search(r"match|swap|hit[\s_-]*factor", ln_, _re49.I) for ln_ in shown49))

        # -- 2. ONLY at the round's end ------------------------------------------------
        hm49 = _Host()
        hm49.set_magazine(4)
        hm49.hold_s = REAL_HOLD_S
        hm49.set_mode(_Static(), ctx44, players=1)
        _fire49(hm49, 0, [10.0, 11.0, 12.0], True, 85.0)
        check("mid-round (three of four rounds fired) there is no score: no card, nothing drawn",
              not hm49.round_over and hm49.cards == (None, None)
              and not any(ln_.startswith("SCORE") for ln_ in _lines44(hm49, ctx44)))
        _fire49(hm49, 0, [13.0], True, 85.0)
        check("...nor when the last bullet leaves: the hole is held for a second first",
              not hm49.round_over and hm49.cards == (None, None)
              and not any(ln_.startswith("SCORE") for ln_ in _lines44(hm49, ctx44)))
        hm49.advance(context(now=13.0 + REAL_HOLD_S + 1.0))
        check("...and it appears when the round ENDS",
              hm49.round_over and hm49.cards[0] is not None
              and any(ln_.startswith("SCORE") for ln_ in _lines44(hm49, ctx44)))
        hmenu49 = _Host()
        hmenu49.set_mode(_menu46.Menu(), ctx44, players=1)
        check("a screen that is not a drill (the menu) is never scored",
              hmenu49.counts_ammo is False and hmenu49.cards == (None, None)
              and not any(ln_.startswith("SCORE") for ln_ in _lines44(hmenu49, ctx44)))

        # -- 3. the screen stays INERT with the score on it ----------------------------
        _render46(h49.paint, c4k49)                          # the first paint arms the box
        h49.advance(context(now=t49[-1] + 5.0))
        crc0 = _zl49.crc32(_render46(h49.paint, c4k49)[0].tobytes())
        cards0 = h49.cards
        t_ = t49[-1] + 5.0
        repaints49, ticks49 = 0, 0
        for _i in range(60 * 150):
            t_ += 1 / 60.0
            ticks49 += 1
            if h49.advance(context(now=t_)):
                repaints49 += 1
        for jump_ in (400.0, 4000.0, 40000.0):
            t_ += jump_
            ticks49 += 1
            if h49.advance(context(now=t_)):
                repaints49 += 1
        crc1 = _zl49.crc32(_render46(h49.paint, c4k49)[0].tobytes())
        check(f"with the score on it the round screen still asks for NO repaint: {repaints49} in "
              f"{ticks49} ticks, 150 s at 60 Hz then jumps of 400 s, 4,000 s and 40,000 s; same "
              f"pixels, same cards",
              repaints49 == 0 and crc0 == crc1 and h49.cards is cards0 and h49.round_over,
              f"{repaints49} repaints, pixels {'same' if crc0 == crc1 else 'CHANGED'}")

        # -- 4. the same height at every state; every colour grey, red excess 0 ---------
        def _geom49(h_):
            return _geom44(h_)

        # A DISCARDED round: four shots 0.1 s apart span 0.3 s against a floor of 0.68 s
        # (0.91 x 3 x 0.25), so the host throws it away whole (2026-10-22). `step=0.1`
        # is the only thing that makes it a storm here; a FAINT round at a human pace
        # (peak 15, step 1.0) is sound and scores, and is checked below.
        noted49, _t = _round49(step=0.1)
        check("a DISCARDED round changes NO box (it is the same four lines in the same boxes, so "
              "the screen is the same height as a scored round's), for one player and for two, "
              "and its short note is on the card but nowhere in the drawn lines",
              noted49.cards[0].discarded and noted49.cards[0].note.startswith("DISCARDED")
              and noted49.cards[0].note not in _lines44(noted49, ctx44)
              and _geom49(noted49) == _geom49(h49)
              and _geom49(_round49(players=2, step=0.1)[0]) == _geom49(_round49(players=2)[0]),
              noted49.cards[0].note)
        variants49 = (("one player", h49), ("one player, discarded", noted49),
                      ("two players", _round49(players=2)[0]),
                      ("two players, discarded", _round49(players=2, step=0.1)[0]))
        for lbl_, cx_ in (("4K", c4k49), ("1080p", c1080_49)):
            for name_, h_ in variants49:
                a_, _p = _render46(h_.paint, cx_)
                lum_ = cv2.cvtColor(_np46.ascontiguousarray(a_[..., :3]), cv2.COLOR_BGR2GRAY)
                red_ = int(cv2.subtract(a_[..., 2], cv2.max(a_[..., 1], a_[..., 0])).max())
                lit_ = float((lum_ > 60).mean())
                check(f"{lbl_}, {name_}: red excess exactly 0, nothing above the artwork ceiling "
                      f"({int(lum_.max())} vs {_L46.CEILING}), lit area {lit_ * 100:.3f}% inside the "
                      f"2% budget, and ink on the screen",
                      red_ == 0 and int(lum_.max()) <= _L46.CEILING and lit_ <= 0.02
                      and int((lum_ > 30).sum()) > 2000, f"red {red_}")

        # -- 5. the new text fits its boxes, at both sizes, measured with the painter's font ----
        class _Spy49(_QP49):
            def __init__(self, dev):
                super().__init__(dev)
                self.texts = []

            def drawText(self, *a):
                f_ = self.font()
                if len(a) == 2 and hasattr(a[0], "x") and not hasattr(a[0], "width"):
                    # A RUN, drawn at a point (2026-10-22: the numbers are bold and a
                    # shade brighter, so a line is laid out run by run and Qt's
                    # whole-line AlignCenter is no longer what draws it). The box it
                    # belongs to is the one `_draw_lines` was last handed.
                    self.runs.append((a[1], f_.pointSize(), f_.bold(), float(a[0].x()),
                                      float(a[0].y()), self.cur_box))
                    return super().drawText(*a)
                r_ = a[0] if (len(a) >= 3 and hasattr(a[0], "width") and hasattr(a[0], "height")) else None
                self.texts.append((a[-1], f_.pointSize(), f_.bold(), r_))
                return super().drawText(*a)

        over49, small49, drawn49 = [], [], 0
        _o49 = _Host.__dict__["_draw_lines"]

        def _tag49(p_, box_, lines_, start_, bold=False, floor=9):
            if hasattr(p_, "cur_box"):
                p_.cur_box = box_
            return _o49.__func__(p_, box_, lines_, start_, bold, floor)

        for lbl_, cx_ in (("4K", c4k49), ("1080p", c1080_49)):
            for name_, h_ in variants49:
                mine_ = set()
                for s_ in ((0, 1) if h_.versus else (0,)):
                    mine_.update(h_.score_lines(s_))
                img_ = _QI46(cx_.width, cx_.height, _QI46.Format_RGB32)
                img_.fill(0xFF0A0A0A)
                sp_ = _Spy49(img_)
                sp_.runs = []
                sp_.cur_box = None
                _Host._draw_lines = staticmethod(_tag49)
                try:
                    h_.paint(sp_, cx_)
                finally:
                    _Host._draw_lines = _o49
                sp_.end()
                for text_, pt_, bold_, rect_ in sp_.texts:
                    parts_ = text_.split("\n")
                    if not set(parts_) <= mine_:
                        continue                        # not part of the score block
                    drawn49 += 1
                    fm_ = _FM49(_L46.mono(pt_, bold_))
                    wide_ = max(fm_.horizontalAdvance(p_) for p_ in parts_)
                    high_ = fm_.height() * len(parts_)
                    if rect_ is None or wide_ > rect_.width() + 0.5 or high_ > rect_.height() + 0.5:
                        over49.append(f"{lbl_}/{name_}: {text_[:40]!r} needs {wide_:.0f}x{high_:.0f}")
                    if pt_ < 10:
                        small49.append(f"{lbl_}/{name_}: {text_[:30]!r} at {pt_}pt")
                # The lines drawn run by run: put the runs of one line back together
                # (same box, same baseline), and require the WHOLE line -- measured
                # run by run under each run's own font -- to sit inside its box.
                by_line_ = {}
                for text_, pt_, bold_, x_, y_, box_ in sp_.runs:
                    by_line_.setdefault((id(box_), round(y_, 1)), []).append((text_, pt_, bold_, x_, box_))
                for runs_ in by_line_.values():
                    joined_ = "".join(r_[0] for r_ in runs_)
                    if joined_ not in mine_:
                        continue                        # not part of the score block
                    drawn49 += 1
                    box_ = runs_[0][4]
                    x_end_ = max(r_[3] + _FM49(_L46.mono(r_[1], r_[2])).horizontalAdvance(r_[0]) for r_ in runs_)
                    if box_ is None or runs_[0][3] < box_.left() - 0.5 or x_end_ > box_.right() + 1.5:
                        over49.append(f"{lbl_}/{name_}: {joined_[:40]!r} spans {runs_[0][3]:.0f}..{x_end_:.0f} "
                                      f"in {None if box_ is None else (box_.left(), box_.right())}")
                    if min(r_[1] for r_ in runs_) < 10:
                        small49.append(f"{lbl_}/{name_}: {joined_[:30]!r} at {min(r_[1] for r_ in runs_)}pt")
        check(f"every string of the score block fits the box it is drawn into at 3840x2160 and "
              f"1920x1080, one player and two, with and without a caveat ({drawn49} strings measured, "
              f"line by line)", drawn49 >= 16 and not over49, "; ".join(over49[:3]))
        check("...and none of it is shrunk to the font's floor (a block that fits at 8pt fits "
              "and cannot be read)", not small49, "; ".join(small49[:3]))

        # -- 6. the honest states, through the host --------------------------------------
        withins49: list = []

        def _spied49(avail, declared, fps, store=None):
            h_ = _Host()
            h_.set_magazine(4)
            h_.hold_s = REAL_HOLD_S
            h_.fps = fps
            if store is not None:
                h_.bests = _B49.BestBook(store)
            h_.set_mode(_Static(), ctx44, players=1)
            m_ = h_.mode
            m_.ghost_has_waits = True
            seen_ = []

            # Mirrors `Mode.ghost_available_at(shots, fps, within=None)`: the host now
            # passes the human's clock as `within` (2026-10-23), and a stand-in that
            # takes two arguments dies on the call and is swallowed by `_score_round`'s
            # guard as "score could not be computed" (RULES C12/C17). `within` is
            # recorded so a check can assert the host really supplies it.
            def _avail(n_, fps_, within=None):
                seen_.append((n_, fps_))
                withins49.append(within)
                return avail(n_, fps_)
            m_.ghost_available_at = _avail
            m_.ghost_declared_waits = lambda n_: declared
            _fire49(h_, 0, [10.0, 14.0, 18.0, 22.0], True, 85.0)
            h_.advance(context(now=22.0 + REAL_HOLD_S + 1.0))
            return h_, seen_

        st_lb49: dict = {}
        hlb49, seen_lb49 = _spied49(lambda n_, f_: None, 0, 30.0, st_lb49)
        # INVERTED 2026-10-22 (his ruling: AT LEAST leaves the screen, as ABOUT did). It used to
        # assert the words were DRAWN. It now asserts they are ABSENT from the block and from every
        # line the screen paints, while the SAME check pins the data and the log: basis, the
        # caveat, the `AT LEAST:` sentence. CHANGED AGAIN 2026-10-22 (his second ruling: "if the app
        # shows the score on screen ... the app has to honor that score and record it"): it used to
        # assert that NO record is set; it now asserts the opposite -- the round IS recorded, in the
        # file, marked lower_bound, and the app is told once to write it. Same host, same round.
        check("a drill that makes you wait, whose waits were not supplied, is a LOWER BOUND in the "
              "data and the log, is drawn WITHOUT AT LEAST anywhere, and IS recorded, marked as "
              "a floor",
              hlb49.cards[0].basis == "lower_bound"
              and hlb49.score_lines(0)[0] == f"SCORE  {hlb49.cards[0].gs.shown}"
              and not any("AT LEAST" in ln_ for ln_ in hlb49.score_lines(0))
              and not any("AT LEAST" in ln_ for ln_ in _lines44(hlb49, ctx44))
              and any(ln_ == f"SCORE  {hlb49.cards[0].gs.shown}" for ln_ in _lines44(hlb49, ctx44))
              and hlb49.cards[0].note.startswith("AT LEAST")
              and "AT LEAST: " + hlb49.cards[0].gs.lower_bound_why in _SC49.describe(hlb49.cards[0])
              and len(hlb49.score_lines(0)) == 4
              and hlb49.cards[0].note not in _lines44(hlb49, ctx44)
              and len(st_lb49) == 1
              and [e_.get("score", {}).get("basis") for e_ in st_lb49.values()] == ["lower_bound"]
              and [e_.get("efficiency", {}).get("basis") for e_ in st_lb49.values()] == ["lower_bound"]
              and hlb49.take_bests_changed() is True and hlb49.take_bests_changed() is False,
              hlb49.score_lines(0)[0])
        check("the ghost is handed the CONFIGURED rate the host was given (30.0), not a default and "
              "not a measured one",
              seen_lb49 == [(4, 30.0)], str(seen_lb49))
        check("the ghost is handed the HUMAN'S OWN CLOCK, first shot to last, from the shots' own "
              "timestamps (2026-10-23: it is charged only the waits that lie inside that clock; a "
              "host that stopped passing it would charge waits outside the round)",
              len(withins49) == 1 and withins49[0] == (10.0, 22.0), str(withins49))
        hdc49, seen_dc49 = _spied49(
            lambda n_, f_: _G49.available_from_waits([0.0] + [2.0] * (n_ - 1), f_), 2, 60.0, {})
        check("a round whose ghost waits are partly the drill's own average is DECLARED in the data and "
              "the log (basis, the count, the ESTIMATE sentence), is drawn WITHOUT ABOUT anywhere on the "
              "score block (his ruling, 2026-10-22), and MAY set a record",
              hdc49.cards[0].basis == "declared"
              and hdc49.score_lines(0)[0] == f"SCORE  {hdc49.cards[0].gs.shown}"
              and not any("ABOUT" in ln_ for ln_ in hdc49.score_lines(0))
              and "ESTIMATE: 2 of the ghost's " in _SC49.describe(hdc49.cards[0])
              and "(declared)" in _SC49.describe(hdc49.cards[0])
              and hdc49.cards[0].note == "ESTIMATE: 2 waits assumed"
              and hdc49.cards[0].note not in _lines44(hdc49, ctx44)
              and hdc49.take_bests_changed() is True and seen_dc49 == [(4, 60.0)],
              hdc49.cards[0].note)
        check("the host's default rate is the SETTING's default, not a number of its own",
              _Host().fps == float(_AS49().fps))
        lab_src49 = (ROOT_SRC / "lsa" / "tools" / "camera_lab.py").read_text(encoding="utf-8")
        fps_sets49 = [s_.strip() for s_ in _re49.findall(r'\["modes"\]\.fps\s*=\s*([^\n]+)', lab_src49)]
        check("the app tells the host only the CONFIGURED rate: from the saved setting at start and from "
              "the FPS box when it changes, never from anything measured",
              fps_sets49 == ["float(self.settings.fps)", "float(self.spin_fps.value())"], str(fps_sets49))

        # -- 7. the bests: written once, guarded, never regressed ---------------------------
        key49 = _B49.best_key("static", 4, 1)
        store49: dict = {}
        ha49, _t = _round49(store=store49)
        check("a legitimate round writes BOTH records under drill|magazine|players, once",
              key49 in store49 and set(store49[key49]) == {"score", "efficiency"}
              and ha49.take_bests_changed() is True and ha49.take_bests_changed() is False,
              str(list(store49)))
        rec49 = store49[key49]["score"]
        check("...with the raw ingredients and the basis beside the derived value",
              rec49["points"] == ha49.tallies[0].points and rec49["shots"] == 4
              and abs(rec49["seconds"] - 3.0) < 1e-6 and rec49["basis"] == "measured"
              and rec49["ghost_interval"] == _G49.GHOST_INTERVAL and rec49["ghost_points"] > 0)
        snap49 = _cp49.deepcopy(store49)
        hb49, _t = _round49(store=store49)
        check("the same round again writes NOTHING: the settings file is not touched for a tie",
              hb49.take_bests_changed() is False and store49 == snap49)
        hw49, _t = _round49(store=store49, hit=[True, True, False, False])
        check("a worse round writes nothing either",
              hw49.take_bests_changed() is False and store49 == snap49)
        hf49, _t = _round49(store=store49, step=0.8)
        polls49 = sum(1 for _i in range(600) if hf49.take_bests_changed())
        check("a faster round IS a record: the app is told once (600 polls of the signal, one write), "
              "and the stored score went up",
              polls49 == 1 and store49[key49]["score"]["value"] > snap49[key49]["score"]["value"]
              and abs(store49[key49]["score"]["seconds"] - 2.4) < 1e-6,
              f"{polls49} writes")
        hz49, _t = _round49(store=store49, step=0.8)
        check("...and after it the same pace is a tie again: zero writes over 600 polls",
              sum(1 for _i in range(600) if hz49.take_bests_changed()) == 0)
        check("a round is judged against the saved file as it is NOW: a corrupt entry is replaced "
              "by a valid one and the round does not crash",
              (lambda s_: (_round49(store=s_)[0].take_bests_changed(), isinstance(s_.get(key49), dict))
               )({key49: "junk"}) == (True, True))

        # -- 8. THE STORM through the host, with his real bests in the file ------------------
        real49 = {_B49.best_key("static", 17, 1): {
            "score": {"value": 416.0, "points": 165, "seconds": 9.34, "shots": 17,
                      "basis": "measured", "declared_waits": 0,
                      "ghost_interval": _G49.GHOST_INTERVAL, "when": ""},
            "efficiency": {"value": 17.67, "points": 165, "seconds": 9.34, "shots": 17,
                           "basis": "measured", "when": ""}}}
        snap_r49 = _cp49.deepcopy(real49)
        hs49, _t = _round49(mag=17, step=2.684 / 16, peak=STORM49, hit=[True] * 8 + [False] * 9,
                            store=real49)
        cs49 = hs49.cards[0]
        storm_screen49 = _lines44(hs49, ctx44)
        check("the phantom storm (17 shots in 2.7 s, peaks 13-25, half of them hits) would have "
              "BEATEN his 416 and it is DISCARDED WHOLE: it sets NEITHER record (the file is "
              "byte-for-byte what it was, no write is signalled), and it has no score and no "
              "efficiency on its card",
              cs49.discarded and cs49.would_beat and not cs49.gs.scorable and real49 == snap_r49
              and not hs49.take_bests_changed() and cs49.gs.score is None
              and cs49.efficiency is None and cs49.note.startswith("DISCARDED"), cs49.note)
        check("... and the SCREEN shows none of it: the headline says ROUND DISCARDED and why, in "
              "his terms; no SCORE, AIM, SPEED, BEST or EFFICIENCY figure appears anywhere on it "
              "(EFFICIENCY is `--`), though the tally he can check himself stays",
              "ROUND DISCARDED" in storm_screen49 and "17 shots in 2.7 s is faster" in storm_screen49
              and "than a pistol can fire." in storm_screen49
              and not any(ln_.startswith(("SCORE", "AIM", "BEST", "NEW BEST")) for ln_ in storm_screen49)
              and [ln_ for ln_ in storm_screen49 if ln_.startswith("EFFICIENCY")] == ["EFFICIENCY  --"]
              and not any(_re49.fullmatch(r"= \d+ points? / [\d.]+ seconds", ln_) for ln_ in storm_screen49)
              and any(_re49.fullmatch(r"\d+ of \d+ points", ln_) for ln_ in storm_screen49)
              and any(ln_ == "17 shots" for ln_ in storm_screen49), str(storm_screen49))
        check("... and its log line (the host's own event) says it was discarded, what it would "
              "have scored and that it would have beaten the stored best",
              any(e_.startswith("score: ROUND DISCARDED") and "WOULD HAVE BEATEN" in e_
                  and "it would have scored" in e_ for e_ in hs49.take_events()))
        for label_, kw_, discarded_, cautions_ in (
                ("faint shots at a human pace", dict(step=0.6, peak=STORM49), False,
                 (_G49.Unfit.PEAK_TOO_LOW,)),
                ("laser-bright shots at the storm's pace",
                 dict(step=2.684 / 16, peak=85.0), True, ()),
                ("bright, human pace, a calibrated microphone that confirmed none",
                 dict(step=0.6, peak=85.0, mic=True), False, (_G49.Unfit.MIC_UNCORROBORATED,))):
            sto_ = {}
            hg_, _t = _round49(mag=17, store=sto_, **kw_)
            gg_ = hg_.cards[0].gs
            if discarded_:
                ok_ = (hg_.cards[0].discarded and gg_.record.reasons == (_G49.Unfit.TOO_FAST,)
                       and not hg_.cards[0].gs.scorable and sto_ == {} and not hg_.take_bests_changed()
                       and hg_.score_lines(0)[0] == "ROUND DISCARDED")
            else:
                ok_ = (not hg_.cards[0].discarded and gg_.record.reasons == ()
                       and gg_.record.cautions == cautions_ and hg_.cards[0].gs.scorable
                       and hg_.score_lines(0)[0] == f"SCORE  {gg_.shown}"
                       and set(sto_.get(_B49.best_key("static", 17, 1), {})) == {"score", "efficiency"}
                       and hg_.take_bests_changed() is True)
            check(f"{label_}: " + ("DISCARDED by the span alone, shows no score and writes neither record"
                                   if discarded_ else
                                   "only a CAUTION: scored, SHOWN and RECORDED (a shown score is a "
                                   "recorded score)"),
                  ok_, f"{gg_.record.reasons} / {gg_.record.cautions}")
        sto_c = {}
        hc49, _t = _round49(mag=17, step=0.6, mic=True, store=sto_c,
                            between=lambda h_, ts_: h_.note_pull_outcomes(
                                [("both", t_ + 0.05) for t_ in ts_[:9]]))
        check("the microphone's pairings reach the guard: 9 of 17 shots confirmed is a majority, and "
              "the same round sets the record",
              hc49.cards[0].gs.scorable and hc49.take_bests_changed() is True and sto_c != {})
        sto_d = {}
        hd49, _t = _round49(mag=17, step=0.6, mic=True, store=sto_d,
                            between=lambda h_, ts_: [h_.note_live_unheard(t_ + 0.05, True) for t_ in ts_])
        check("shots fired while the microphone was DEAF leave the denominator: none confirmable, "
              "so the condition is skipped and the round is fit",
              hd49.cards[0].gs.scorable and sto_d != {})
        sto_e = {}
        he49, _t = _round49(mag=17, step=0.6, mic=True, store=sto_e,
                            between=lambda h_, ts_: [h_.note_live_unheard(t_ + 0.05, False) for t_ in ts_])
        check("a microphone that WAS listening and heard nothing is a NOTE and not a verdict "
              "(2026-10-22, his rule that a shown score is a recorded score and only a proven "
              "storm is thrown away): the round at a human pace is scored, recorded, and carries "
              "the caution for the log",
              he49.cards[0].gs.scorable and not he49.cards[0].discarded and sto_e != {}
              and he49.cards[0].gs.record.cautions == (_G49.Unfit.MIC_UNCORROBORATED,)
              and he49.take_bests_changed() is True)
        check("the peaks the guard reads are the shots' own: four 85s, filed per live shot",
              h49.tallies[0].peaks == [85.0] * 4 and len(h49.tallies[0].seen_t) == 4)
        h49.shot(modes.ShotEvent(300.0, 900.0, t49[-1] + 9.0, 99, t_end=t49[-1] + 9.05, peak=1.0),
                 context(now=t49[-1] + 9.0))
        check("a pull after the round is over (a dry pull) adds no peak: only registered shots are filed",
              h49.tallies[0].peaks == [85.0] * 4)

        # -- 9. keys: drill, magazine, players ------------------------------------------------
        sk49: dict = {}
        _round49(store=sk49)
        _round49(mag=5, store=sk49)
        hp49, _t = _round49(players=2, store=sk49)
        hbl49, _t = _round49(cls=EscalatingBall, store=sk49)
        check("bests do not collide: the same drill at magazine 4 and 5, one player and two, and "
              "another drill, are five different keys",
              {_B49.best_key("static", 4, 1), _B49.best_key("static", 5, 1),
               _B49.best_key("static", 4, 2)} <= set(sk49)
              and hbl49.cards[0].key == _B49.best_key("ball", 4, 1)
              and hbl49.cards[0].key != hp49.cards[0].key and len(sk49) >= 3)

        # -- 10. two players: a score for EACH, one write ------------------------------------
        sv49: dict = {}
        hv49, _t = _round49(players=2, store=sv49)
        shown_v49 = _lines44(hv49, ctx44)
        check("two players: each half gets its own score, drawn on the end screen, against the "
              "players-2 key",
              hv49.cards[0] is not None and hv49.cards[1] is not None
              and hv49.cards[0].key == hv49.cards[1].key == _B49.best_key("static", 4, 2)
              and sum(1 for ln_ in shown_v49 if ln_.startswith("SCORE")) == 2
              and hv49.score_lines(0)[0].startswith("SCORE  ")
              and hv49.score_lines(1)[0].startswith("SCORE  "), str(hv49.score_lines(0)))
        check("...and the file is written ONCE for the round, however many halves set a record",
              bool(sv49) and sum(1 for _i in range(600) if hv49.take_bests_changed()) == 1)
        ev49 = hv49.take_events()
        check("the log carries one score line per half, with the factors and the guard's decision",
              sum(1 for e_ in ev49 if e_.startswith("score, ")) == 2
              and all("aim " in e_ and "record" in e_ for e_ in ev49 if e_.startswith("score, ")),
              str([e_[:50] for e_ in ev49]))

        # -- 11. a click is never a shot, and a dry pull is never a record ----------------------
        snap_k49 = _cp49.deepcopy(sk49)
        h49.on_click(300.0, 900.0, ctx44)
        check("a click on the end screen files nothing and writes nothing",
              h49.tallies[0].peaks == [85.0] * 4 and sk49 == snap_k49)

    try:
        _section49()
    except Exception as exc_49:                                     # noqa: BLE001 - the finding
        import traceback as _tb49
        check("section 49 ran to its end without raising", False,
              f"{type(exc_49).__name__}: {exc_49}")
        print("".join(_tb49.format_exc().splitlines(True)[-6:]))

    # ------------------------------------------------------------------
    # 50. THE ROUND SCREEN'S NUMBERS (2026-10-22). His rulings: every number on the
    # screen is bold and one grey step up (GREY_HI), the words around it unchanged;
    # the single-player result is centred in its zone instead of hung from the top;
    # two players carries the ghost's ceiling on its points line like single player.
    # What these checks guard, each of which was a way to get it wrong:
    #   * a bold run is wider than the same characters regular, so a line is MEASURED
    #     run by run (joined, one font, it understates and the box is overrun);
    #   * a line with runs is PLACED run by run, centred by its total width;
    #   * a caller that passes plain strings sees nothing change, to the pixel;
    #   * the splitter is ONE rule, and "--" / "ABOUT" / the arithmetic are left alone;
    #   * the NEXT ROUND box, which he shoots at, has not moved by a pixel.
    # Run behind its own crash guard, like 49: a raise is a FAILED check.
    # ------------------------------------------------------------------
    def _section50() -> None:
        print()
        print("50. the round screen's numbers: bold, one grey step up, measured and placed run by run")
        import re as _re50
        import numpy as _np50
        from PySide6.QtCore import QRect as _QR50, Qt as _Qt50
        from PySide6.QtGui import QFontMetricsF as _FM50, QImage as _QI50, QPainter as _QP50
        from lsa import scorecard as _SC50
        from lsa.display import layers as _L50
        from lsa.modes import runs as _R50, versus as _V50

        # -- 1. the splitter: ONE rule, and what it leaves alone ----------------------
        SENT50 = object()                 # the splitter only CARRIES a colour; any object will do

        def _bolds50(line):
            return [r_.text for r_ in line.runs if r_.bold]

        CASES50 = [
            ("17 shots", ["17"]), ("6 hits", ["6"]), ("11 misses", ["11"]),
            ("20 of 75 points", ["20", "75"]), ("20 points", ["20"]),
            ("1000 of 1500 points  BEST", ["1000", "1500"]), ("1000 points", ["1000"]),
            ("10.7 seconds", ["10.7"]), ("EFFICIENCY  1.87", ["1.87"]),
            ("EFFICIENCY  1.87  *", ["1.87"]), ("EFFICIENCY  --", []),
            ("AIM 27%   SPEED 37%", ["27%", "37%"]), ("AIM --   SPEED --", []),
            ("AIM 82%   SPEED ABOUT 21%", ["82%", "21%"]),
            ("AIM 82%   SPEED AT LEAST 21%", ["82%", "21%"]),
            ("BEST SCORE SO FAR  ABOUT 203", ["203"]), ("BEST SCORE SO FAR  203", ["203"]),
            ("BEST SCORE SO FAR  --", []), ("BEST EFFICIENCY SO FAR  3.82", ["3.82"]),
            ("SCORE  AT LEAST 47", ["47"]), ("LEFT", []), ("RIGHT is more efficient", []), ("", []),
            ("NEW BEST SCORE RECORDED", ["NEW BEST SCORE RECORDED"]),
            ("NEW BEST EFFICIENCY RECORDED", ["NEW BEST EFFICIENCY RECORDED"]),
        ]
        bad50 = []
        for text_, want_ in CASES50:
            rl_ = _R50.emphasise(text_, SENT50)
            if _bolds50(rl_) != want_:
                bad50.append(f"{text_!r}: bold {_bolds50(rl_)} wanted {want_}")
            if "".join(r_.text for r_ in rl_.runs) != text_ or rl_ != text_ or not isinstance(rl_, str):
                bad50.append(f"{text_!r}: the runs do not add back up to the text")
            if any((r_.color is not SENT50) for r_ in rl_.runs if r_.bold):
                bad50.append(f"{text_!r}: a bold run lost its colour")
            if any((r_.bold is not None or r_.color is not None) for r_ in rl_.runs if not r_.bold):
                bad50.append(f"{text_!r}: a plain run was given a weight or a colour")
        check(f"the splitter, {len(CASES50)} lines: integers, decimals, percentages, two numbers in one line, "
              f"1000, a line with no number, and `--` stays plain", not bad50, "; ".join(bad50[:3]))
        about50 = _R50.emphasise("BEST SCORE SO FAR  ABOUT 203", SENT50)
        check("`ABOUT 203`: the qualifier stays regular, the figure goes bold",
              any("ABOUT" in r_.text and not r_.bold for r_ in about50.runs) and _bolds50(about50) == ["203"])
        check("the whole-line prefix is the SCORECARD's, imported not retyped, and both record announcements "
              "start with it",
              _R50.WHOLE_LINE_PREFIX is _SC50.NEW_BEST_PREFIX
              and "NEW BEST SCORE RECORDED".startswith(_SC50.NEW_BEST_PREFIX)
              and "NEW BEST EFFICIENCY RECORDED".startswith(_SC50.NEW_BEST_PREFIX))
        # The two-player RESULT line is the other whole-line case (2026-10-22). Built from the
        # real `result_words` / `verdict_sentence` for every verdict and both wordings, so a
        # reworded result cannot silently fall back to numeric-only emphasis; and the efficiency
        # sentence, which also starts with a side name, must NOT be caught by the same rule.
        res_forms50 = []
        for v_ in (_V50.Verdict("win", 0, 11), _V50.Verdict("win", 1, 1), _V50.Verdict("tie", None, 0),
                   _V50.Verdict("none", None, 0)):
            res_forms50.append(_V50.result_words(v_, _V50.SIDE_NAMES))
            res_forms50.append(_V50.result_words(v_, _V50.SIDE_NAMES, match=True))
            res_forms50.append(_V50.verdict_sentence(v_, _V50.SIDE_NAMES))
        whole50 = [t_ for t_ in res_forms50
                   if (lambda rl_: len(rl_.runs) == 1 and rl_.runs[0].bold is True
                       and rl_.runs[0].color is SENT50 and rl_.runs[0].text == t_)(_R50.emphasise(t_, SENT50))]
        check(f"every result wording the verdict can produce ({len(set(res_forms50))} distinct, e.g. "
              f"{res_forms50[0]!r}) is ONE bold coloured run end to end",
              len(whole50) == len(res_forms50) and "RIGHT WINS by 1 point" in res_forms50
              and "MATCH: LEFT leads by 11 points" in res_forms50, str(set(res_forms50) - set(whole50)))
        eff50 = ["LEFT is more efficient, 9.44 to 9.41", "RIGHT is also more efficient, 3.00 to 2.00", ""]
        check("... but the efficiency sentence beside it keeps numeric-only emphasis (a hierarchy: the "
              "result is the headline, this is the aside)",
              all(not _V50.is_result_line(t_) for t_ in eff50)
              and _bolds50(_R50.emphasise(eff50[0], SENT50)) == ["9.44", "9.41"]
              and _bolds50(_R50.emphasise(eff50[1], SENT50)) == ["3.00", "2.00"]
              and _bolds50(_R50.emphasise("", SENT50)) == [])

        # -- 2. measured run by run, honoured by the fit ---------------------------------
        # The test fonts are monospace, in which a bold advance can equal the regular one,
        # which would make the joined string and the sum agree and the check unable to
        # fail. So bold is made definitively WIDER here (the same family four sizes up).
        orig_mono50 = _L50.mono

        def _wide_bold50(size, bold=False):
            return orig_mono50(size + 4 if bold else size, False)

        _L50.mono = _wide_bold50
        try:
            rl50 = _R50.emphasise("123456 of 789012 points", _L50.GREY_HI)
            per_run50 = sum(_FM50(_wide_bold50(40, bool(r_.bold))).horizontalAdvance(r_.text) for r_ in rl50.runs)
            joined50 = _FM50(_wide_bold50(40, False)).horizontalAdvance(str(rl50))
            measured50 = _Host.line_width(rl50, 40, False)
            check("(premise) a bold run really is wider here, so the joined string understates the line "
                  f"({joined50:.0f} px joined against {per_run50:.0f} px run by run)",
                  per_run50 > joined50 + 2)
            check("a mixed-weight line's measured width is the SUM of its runs, each under its own font "
                  f"({measured50:.1f} against {per_run50:.1f})", abs(measured50 - per_run50) < 1e-6)
            check("...and a plain string measures exactly as it always did (one font, the whole string)",
                  abs(_Host.line_width("20 of 75 points", 40, False)
                      - _FM50(_wide_bold50(40, False)).horizontalAdvance("20 of 75 points")) < 1e-6)
            box50 = _QR50(0, 0, int(joined50 / 0.94) + 2, 400)      # the joined string JUST fits at 40
            img50 = _QI50(box50.width() + 20, 420, _QI50.Format_RGB32)
            img50.fill(0xFF0A0A0A)
            p50 = _QP50(img50)
            size50 = _Host._draw_lines(p50, box50, [rl50], 40)
            p50.end()
            check("the fit search honours it: a line that fits only when measured as one plain string is "
                  f"shrunk (size {size50} of 40), and what it settled on fits run by run while one size "
                  "larger does not",
                  size50 < 40 and _Host.line_width(rl50, size50) <= box50.width() * 0.94
                  and _Host.line_width(rl50, size50 + 1) > box50.width() * 0.94,
                  f"size {size50}")

            # -- 3. placed run by run: centred by the total, each run in its own pen ----
            MID50, HI50 = _L50.GREY_MID.red(), _L50.GREY_HI.red()
            label50 = _R50.Run("WORD  ")
            num50 = _R50.Run("77", True, _L50.GREY_HI)
            pl50 = _R50.RunLine([label50, num50])
            bx50 = _QR50(0, 0, 800, 200)
            im50 = _QI50(800, 200, _QI50.Format_RGB32)
            im50.fill(0xFF0A0A0A)
            q50 = _QP50(im50)
            q50.setPen(_L50.GREY_MID)
            sz50 = _Host._draw_lines(q50, bx50, [pl50], 40)
            pen_after50 = q50.pen().color().red()
            q50.end()
            a50 = _np50.frombuffer(im50.constBits(), _np50.uint8).reshape(200, im50.bytesPerLine())[:, :800 * 4] \
                .reshape(200, 800, 4)[..., 2].astype(int)
            total50 = _Host.line_width(pl50, sz50)
            lw50 = _Host.line_width(_R50.RunLine([label50]), sz50)
            x0_50 = (800 - total50) / 2.0
            cols50 = _np50.nonzero((a50 > 30).any(axis=0))[0]
            lab_max50 = int(a50[:, int(x0_50): int(x0_50 + lw50) - 2].max())
            num_max50 = int(a50[:, int(x0_50 + lw50) + 1: int(x0_50 + total50) + 2].max())
            check(f"a line with runs is centred by its TOTAL width (ink {cols50.min()}..{cols50.max()}, "
                  f"expected about {x0_50:.0f}..{x0_50 + total50:.0f})",
                  len(cols50) > 0 and cols50.min() >= x0_50 - 3 and cols50.max() <= x0_50 + total50 + 3
                  and abs((cols50.min() + cols50.max()) / 2.0 - 400.0) <= 10.0)
            check(f"...the label keeps the call's pen (brightest label pixel {lab_max50}, GREY_MID is {MID50}) "
                  f"and the number is drawn in its own (brightest {num_max50}, GREY_HI is {HI50})",
                  lab_max50 <= MID50 and num_max50 >= HI50 - 2 and HI50 > MID50)
            check("...and the painter's pen is handed back as it was found", pen_after50 == MID50)
        finally:
            _L50.mono = orig_mono50

        # -- 4. plain callers are untouched: same size, same pixels -----------------------
        def _old_fit50(box_, lines_, start_, bold_, floor_=9):
            s_ = max(floor_, int(start_))
            while True:
                fm_ = _FM50(_L50.mono(s_, bold=bold_))
                wide_ = max(fm_.horizontalAdvance(t_) for t_ in lines_)
                high_ = fm_.height() * len(lines_)
                if (wide_ <= box_.width() * 0.94 and high_ <= box_.height() * 0.96) or s_ <= floor_:
                    return s_
                s_ -= 1

        legacy50, legacy_bad50 = 0, []
        for box_, lines_, start_, bold_ in (
                (_QR50(0, 0, 900, 220), ["NEXT ROUND"], 48, True),
                (_QR50(0, 0, 640, 300), ["LEFT", "17 of 17 shots", "26 of 75 points  BEST"], 40, False),
                (_QR50(0, 0, 90, 20), ["a line far too long for this little box"], 48, False),     # bottoms out
                (_QR50(0, 0, 500, 40), ["a", "b", "c", "d", "e"], 30, True)):
            ia_ = _QI50(box_.width(), box_.height(), _QI50.Format_RGB32)
            ib_ = _QI50(box_.width(), box_.height(), _QI50.Format_RGB32)
            ia_.fill(0xFF0A0A0A)
            ib_.fill(0xFF0A0A0A)
            pa_ = _QP50(ia_)
            pa_.setPen(_L50.GREY_MID)
            sa_ = _Host._draw_lines(pa_, box_, lines_, start_, bold=bold_)
            pa_.end()
            pb_ = _QP50(ib_)
            pb_.setPen(_L50.GREY_MID)
            pb_.setFont(_L50.mono(sa_, bold=bold_))
            pb_.drawText(box_, _Qt50.AlignCenter, "\n".join(lines_))
            pb_.end()
            legacy50 += 1
            if sa_ != _old_fit50(box_, lines_, start_, bold_) or bytes(ia_.constBits()) != bytes(ib_.constBits()):
                legacy_bad50.append(f"{lines_[0][:20]!r} size {sa_} vs {_old_fit50(box_, lines_, start_, bold_)}")
        check(f"a caller passing plain strings gets the SAME size and the SAME pixels as before ({legacy50} cases, "
              f"including a box it bottoms out in)", not legacy_bad50, "; ".join(legacy_bad50[:2]))

        # -- 5. the real screen: which lines are lifted, which are not --------------------
        class _Spy50(_QP50):
            def __init__(self, dev):
                super().__init__(dev)
                self.runs = []
                self.cur_box = None

            def drawText(self, *a):
                if len(a) == 2 and hasattr(a[0], "x") and not hasattr(a[0], "width"):
                    f_ = self.font()
                    self.runs.append((a[1], f_.pointSize(), f_.bold(), float(a[0].x()), self.cur_box))
                return super().drawText(*a)

        def _paint50(h_, c_):
            drawn_ = []
            o_ = _Host.__dict__["_draw_lines"]
            img_ = _QI50(c_.width, c_.height, _QI50.Format_RGB32)
            img_.fill(0xFF0A0A0A)
            sp_ = _Spy50(img_)

            def w_(p_, box_, lines_, start_, bold=False, floor=9):
                p_.cur_box = box_
                size_ = o_.__func__(p_, box_, lines_, start_, bold, floor)
                drawn_.append((_QR50(box_), list(lines_), size_, bold))
                return size_
            _Host._draw_lines = staticmethod(w_)
            try:
                h_.paint(sp_, c_)
            finally:
                _Host._draw_lines = o_
            sp_.end()
            return drawn_, sp_.runs

        def _ctx50(w_, h_):
            return RenderContext(now=1.0, dt=1 / 60.0, rect=QRect(0, 0, w_, h_), stats=SessionStats())

        def _single50(w_, h_, on_target=True, step=PACE):
            # A human pace by default: these checks read bold numbers, aim and speed and the
            # record lines off a SCORED round. `step=0.01` makes it a discarded burst.
            c_ = _ctx50(w_, h_)
            hh_ = _Host()
            hh_.set_mode(_Static(), c_, players=1)
            _fire(hh_, c_, 0, N25, on_target=on_target, step=step)
            return hh_, c_

        h50, c50 = _single50(3840, 2160)
        card50 = h50.cards[0]
        check("(premise) the finished single-player round has a scorable card with a ceiling",
              card50 is not None and card50.gs.scorable and card50.gs.ghost_points, str(card50 and card50.gs.ghost_points))
        drawn50, _runs50 = _paint50(h50, c50)

        def _call50(lines_):
            hits_ = [d_ for d_ in drawn50 if d_[1] == list(lines_)]
            return hits_[0] if hits_ else None

        sb50 = h50.side_block(0)
        tally50 = _call50(sb50)
        t50 = h50.tallies[0]
        want_b50 = [[str(t50.shots)], [str(t50.hits)], [str(h50.round_misses(0))],
                    [str(t50.points), str(card50.gs.ghost_points)], [sb50[4].split()[0]]]
        got_b50 = [_bolds50(l_) for l_ in tally50[1]] if tally50 else None
        check("the tally block: shots, hits, misses, points (BOTH numbers of `N of M`) and seconds each have "
              "exactly their numerals bold", got_b50 == want_b50, f"{got_b50} vs {want_b50}")
        check("...in GREY_HI, which is one step up from the label's pen and no brighter than a target outline",
              tally50 is not None
              and all(r_.color == _L50.GREY_HI for l_ in tally50[1] for r_ in l_.runs if r_.bold)
              and _L50.GREY_HI.red() <= _L50.CEILING and _L50.GREY_HI.red() > _L50.GREY_MID.red())
        eff50 = h50.efficiency_lines(0)
        headline50, arith50 = _call50(eff50[:1]), _call50(eff50[1:])
        e_val50 = h50.side_efficiency(0)
        check("EFFICIENCY: the figure is lifted (the label stays bold as it was drawn) ...",
              headline50 is not None and _bolds50(headline50[1][0]) == ([f"{e_val50:.2f}"] if e_val50 is not None else [])
              and headline50[3] is True, str(headline50 and _bolds50(headline50[1][0])))
        check("... and the ARITHMETIC under it (`= 20 points / 10.7 seconds`) is NOT: no run in it, a plain "
              "string, the working and not a result",
              arith50 is not None and arith50[1][0].startswith("= ")
              and not any(isinstance(l_, _R50.RunLine) for l_ in arith50[1]))
        sc50 = h50.score_lines(0)
        head_sc50, rest_sc50 = _call50(sc50[:1]), _call50(sc50[1:])
        check("`SCORE  n` is left alone on one player (already the biggest, bold line of the block) ...",
              head_sc50 is not None and not isinstance(head_sc50[1][0], _R50.RunLine) and head_sc50[3] is True)
        aim50 = _bolds50(rest_sc50[1][0]) if rest_sc50 else None
        check("... and the lines under it are lifted: AIM and SPEED each bold INCLUDING the `%`, and the best "
              "lines their value", aim50 == _re50.findall(r"\d+%", sc50[1]) and len(aim50) == 2
              and _bolds50(rest_sc50[1][1]) == _re50.findall(r"\d+(?:\.\d+)?", sc50[2])
              and _bolds50(rest_sc50[1][2]) == _re50.findall(r"\d+(?:\.\d+)?", sc50[3]), str(aim50))

        # the record announcements, bold end to end, on a card that really holds them
        rec50 = {"value": 12.0, "basis": "measured"}
        new_card50 = dataclass_replace(card50, new_score=True, new_efficiency=True,
                                       best_score=rec50, best_efficiency=rec50)
        new_lines50 = _SC50.lines(new_card50)
        check("(premise) a card that set both records says NEW BEST SCORE RECORDED and NEW BEST EFFICIENCY RECORDED",
              new_lines50[2] == "NEW BEST SCORE RECORDED" and new_lines50[3] == "NEW BEST EFFICIENCY RECORDED",
              str(new_lines50))
        h50.cards = (new_card50, None)
        try:
            drawn_new50, _r = _paint50(h50, c50)
        finally:
            h50.cards = (card50, None)
        new_call50 = [d_ for d_ in drawn_new50 if d_[1] == new_lines50[1:]]
        full50 = (len(new_call50) == 1
                  and all(len(l_.runs) == 1 and l_.runs[0].bold and l_.runs[0].color == _L50.GREY_HI
                          and l_.runs[0].text == str(l_) for l_ in new_call50[0][1][1:]))
        check("the NEW BEST lines are bold and lifted END TO END, as drawn (there is no number in them to pick out)",
              full50, str(new_call50[:1] and [l_.runs for l_ in new_call50[0][1][1:]]))

        # -- 6. every round-over state, rendered, at both sizes -------------------------------
        def _states50():
            out_ = []
            for (w_, h_) in ((3840, 2160), (1920, 1080)):
                for tag_, on_ in (("hits", True), ("misses", False)):
                    hh_, cc_ = _single50(w_, h_, on_)
                    out_.append((f"{w_}x{h_} one player, all {tag_}", hh_, cc_))
                hh_, cc_ = _single50(w_, h_)
                hh_.cards = (None, None)
                out_.append((f"{w_}x{h_} one player, no card", hh_, cc_))
                for basis_ in ("declared", "lower_bound"):
                    hh_, cc_ = _single50(w_, h_)
                    hh_.cards = (dataclass_replace(hh_.cards[0], basis=basis_), None)
                    out_.append((f"{w_}x{h_} one player, {basis_}", hh_, cc_))
                hh_, cc_ = _single50(w_, h_)
                hh_.cards = (dataclass_replace(hh_.cards[0], new_score=True, new_efficiency=True,
                                               best_score=rec50, best_efficiency=rec50), None)
                out_.append((f"{w_}x{h_} one player, both records", hh_, cc_))
                hh_, cc_ = _finished25(w_, h_, 2)
                out_.append((f"{w_}x{h_} two players, left wins", hh_, cc_))
                hh_, cc_ = _finished25(w_, h_, 2)
                hh_.cards = (None, None)
                out_.append((f"{w_}x{h_} two players, no card", hh_, cc_))
                cc_ = _ctx50(w_, h_)
                hh_ = _Host()
                hh_.set_mode(_Static(), cc_, players=2)
                _fire(hh_, cc_, 0, N25, on_target=False, step=PACE)
                _fire(hh_, cc_, 1, N25, on_target=False, step=PACE)
                out_.append((f"{w_}x{h_} two players, tied on nothing", hh_, cc_))
                hh_, cc_ = _finished25(w_, h_, 2)
                hh_.cards = tuple(dataclass_replace(k_, new_score=True, new_efficiency=True,
                                                    best_score=rec50, best_efficiency=rec50)
                                  for k_ in hh_.cards)
                out_.append((f"{w_}x{h_} two players, both records", hh_, cc_))
                # DISCARDED (2026-10-22): the block that replaces the score on a storm must be
                # bold-and-lifted where it has numbers, inside its boxes and off the 9 px
                # floor like every other state, one player and two.
                hh_, cc_ = _single50(w_, h_, step=0.01)
                out_.append((f"{w_}x{h_} one player, DISCARDED", hh_, cc_))
                cc_ = _ctx50(w_, h_)
                hh_ = _Host()
                hh_.set_mode(_Static(), cc_, players=2)
                _fire(hh_, cc_, 0, N25, on_target=True, step=PACE)
                _fire(hh_, cc_, 1, N25, on_target=True, step=0.01)
                out_.append((f"{w_}x{h_} two players, right DISCARDED", hh_, cc_))
            return out_

        fit50, floor50, span50, nbox50, gap50 = [], [], [], [], []
        run_lines50, states_seen50, size_rows50, pts_lines50 = 0, 0, [], []
        for label_, hh_, cc_ in _states50():
            if not hh_.round_over:
                fit50.append(f"{label_}: the round did not close")
                continue
            states_seen50 += 1
            dr_, rr_ = _paint50(hh_, cc_)
            for box_, lines_, size_, bold_ in dr_:
                wide_ = max(_Host.line_width(t_, size_, bold_) for t_ in lines_)
                high_ = sum(_Host.line_height(t_, size_, bold_) for t_ in lines_)
                if wide_ > box_.width() + 0.5 or high_ > box_.height() + 0.5:
                    fit50.append(f"{label_}: {lines_[0][:30]!r} {wide_:.0f}x{high_:.0f} in {box_.width()}x{box_.height()}")
                if size_ <= 9:
                    floor50.append(f"{label_}: {lines_[0][:30]!r} fell to the floor")
                if any(isinstance(t_, _R50.RunLine) for t_ in lines_):
                    run_lines50 += len(lines_)
            # every run, placed: inside the box its line was handed
            lines_of_ = {}
            for text_, pt_, bd_, x_, bx_ in rr_:
                lines_of_.setdefault(id(bx_), []).append((text_, pt_, bd_, x_, bx_))
            for runs_ in lines_of_.values():
                bx_ = runs_[0][4]
                lo_ = min(r_[3] for r_ in runs_)
                hi_ = max(r_[3] + _FM50(_L50.mono(r_[1], r_[2])).horizontalAdvance(r_[0]) for r_ in runs_)
                if bx_ is None or lo_ < bx_.left() - 0.5 or hi_ > bx_.right() + 1.5:
                    span50.append(f"{label_}: {''.join(r_[0] for r_ in runs_)[:30]!r} {lo_:.0f}..{hi_:.0f}")
            # the NEXT ROUND box: where it was and how big it was, derived from the
            # fractions it has always been built from, not from the method under test
            lay_ = hh_._layout(cc_)
            from lsa.display.layers import bottom_strip_for as _bsf50
            bw_, bh_ = round(cc_.rect.width() * 0.26), round(cc_.height * 0.15)
            gap_ = round(cc_.height * 0.02)
            mic_top_ = (cc_.rect.y() + cc_.rect.height() - _bsf50(cc_.height)) - max(8, round(cc_.height * 0.035))
            want_rb_ = _QR50(int(lay_.centre_x - bw_ / 2), mic_top_ - gap_ - bh_, bw_, bh_)
            rb_ = hh_.round_box(cc_)
            drawn_rb_ = [d_[0] for d_ in dr_ if d_[1] == ["NEXT ROUND"]]
            if rb_ != want_rb_ or drawn_rb_ != [want_rb_]:
                nbox50.append(f"{label_}: box {rb_} drawn {drawn_rb_} wanted {want_rb_}")
            # one player: the result is centred in its zone, every box its old size
            if not hh_.versus:
                _l, top_, _b = hh_._verdict_zone(cc_)
                avail_ = rb_.top() - gap_ - top_
                head_ = [d_ for d_ in dr_ if len(d_[1]) == 1 and d_[1][0].startswith("ROUND ")]
                tally_ = [d_ for d_ in dr_ if d_[1] == hh_.side_block(0)]
                rest_ = [d_ for d_ in dr_ if d_[1] == hh_.score_lines(0)[1:]]
                mark_ = [d_ for d_ in dr_ if d_[1] == hh_.mark_lines()]
                if not (head_ and tally_ and rest_ and mark_):
                    gap50.append(f"{label_}: the result's boxes were not all drawn")
                    continue
                above_ = head_[0][0].top() - top_
                below_ = (rb_.top() - gap_) - (rest_[0][0].top() + rest_[0][0].height())
                if abs(above_ - below_) > 0.02 * avail_ or above_ < 0.05 * avail_:
                    gap50.append(f"{label_}: {above_}px above the result, {below_}px below (zone {avail_})")
                if (head_[0][0].height() != round(avail_ * 0.13) or tally_[0][0].height() != round(avail_ * 0.30)
                        or rest_[0][0].height() != round(avail_ * 0.12)
                        or tally_[0][0].width() != round(cc_.rect.width() * 0.45)):
                    gap50.append(f"{label_}: a box changed size")
                if mark_[0][0].top() + mark_[0][0].height() > rb_.top() - gap_ + 1:
                    gap50.append(f"{label_}: the mark box no longer fits above the button")
            else:
                col_ = [d_ for d_ in dr_ if d_[1] == hh_.side_block(0)]
                nocard_ = label_.endswith("no card")
                if col_:
                    size_rows50.append((label_, col_[0][2] / cc_.height, nocard_))
                if "left wins" in label_:
                    pts_lines50.append(hh_.side_block(0)[2])
        check(f"(premise) {states_seen50} round-over states rendered (one player: hits, misses, no card, "
              f"declared, at-least, both records, DISCARDED; two players: left wins, no card, tied, both "
              f"records, right DISCARDED) at 4K and 1080p, {run_lines50} lines drawn run by run",
              states_seen50 == 24 and run_lines50 > 100, f"{states_seen50} states, {run_lines50} run lines")
        check("every drawn line fits its box, measured run by run under each run's own font",
              not fit50, "; ".join(fit50[:3]))
        check("...nothing fell to the 9 px floor", not floor50, "; ".join(floor50[:3]))
        check("...and every run, as PLACED, sits inside the box its line was handed", not span50,
              "; ".join(span50[:3]))
        check("the NEXT ROUND box is where and what size it was before this change, one player and two, "
              "4K and 1080p (re-derived from its own fractions: 0.26 of the width, 0.15 and a 0.02 gap of the "
              "height, above the microphone anchor)", not nbox50, "; ".join(nbox50[:2]))
        check("one player: the result is centred in its zone (the space above it equals the space below, "
              "and is not zero), every box keeps the size it was measured at, and the mark box still fits "
              "above the button", not gap50, "; ".join(gap50[:3]))

        # -- 7. two players, the longer points line --------------------------------------
        with_c50 = {l_: f_ for l_, f_, nc_ in size_rows50 if not nc_}
        without50 = {l_: f_ for l_, f_, nc_ in size_rows50 if nc_}
        for l_, f_ in sorted(with_c50.items()):
            print(f"         {l_:<42} column type {f_:.4f} of screen height")
        for l_, f_ in sorted(without50.items()):
            print(f"         {l_:<42} column type {f_:.4f} of screen height (no ceiling, for comparison)")
        check("(premise) the two-player points line carries the ceiling (`N of M points`)",
              len(pts_lines50) == 2 and all(_re50.fullmatch(r"\d+ of \d+ points(  BEST)?", p_) for p_ in pts_lines50),
              str(pts_lines50))
        check(f"the two-player column, with its longer points line, is not at the 9 px floor and is at least "
              f"{MIN_SCORE_FRAC:.1%} of the screen height at 4K and 1080p (the bar the score block already meets)",
              len(with_c50) >= 6 and all(f_ >= MIN_SCORE_FRAC for f_ in with_c50.values()),
              "; ".join(f"{l_} {f_:.4f}" for l_, f_ in with_c50.items() if f_ < MIN_SCORE_FRAC))

    try:
        _section50()
    except Exception as exc_50:                                     # noqa: BLE001 - the finding
        import traceback as _tb50
        check("section 50 ran to its end without raising", False,
              f"{type(exc_50).__name__}: {exc_50}")
        print("".join(_tb50.format_exc().splitlines(True)[-6:]))

    # ------------------------------------------------------------------
    # 51. THE ROUND SCREEN MUST NOT STARVE THE CAMERA (2026-10-22). In his logs the
    # frame rate read ~19 fps two seconds after every round closed (five of five,
    # three builds) against 38-59 at the verdict. `ModeHost._draw_lines` found its
    # font size by walking down one point at a time, building a font and a metrics
    # object per step: ~90 of each per block at 4K, ~6 blocks, on every repaint,
    # and the HUD tick asks for a repaint five times a second. The identical fault
    # the ring numbers had on 2026-10-20 (one paint 12 ms -> 171 ms, 2.9 fps).
    # NO SUITE SAW EITHER: the repaint-REQUEST check measures how often a repaint is
    # asked for, not what one costs, and the offscreen renderer does this font work
    # far faster than Windows. What these checks guard:
    #   * the bisection returns EXACTLY what the walk returned (500 random cases and
    #     every block of the real screens), through the real font engine;
    #   * an unchanged screen's second paint constructs NO font and runs NO search,
    #     so the cache cannot be removed silently (the check that goes red on revert);
    #   * a first paint does a fraction of the font work the walk did;
    #   * the cache is keyed on everything the answer depends on, and bounded;
    #   * the MENU button's identical loop (painted on every in-play repaint);
    #   * a COST bar, derived below, with the caveat that it is a floor.
    # Run behind its own crash guard, like 49 and 50.
    # ------------------------------------------------------------------
    def _section51() -> None:
        print()
        print("51. the round screen's paint cost: fit by bisection, remembered, and bounded")
        import random as _rnd51
        import statistics as _st51
        import time as _t51
        import PySide6.QtGui as _QG51
        from PySide6.QtCore import QRect as _QR51
        from PySide6.QtGui import QFontMetricsF as _FM51, QImage as _QI51, QPainter as _QP51
        from lsa.display import layers as _L51
        from lsa.modes import base as _B51, runs as _R51

        # -- the reference: the walk this replaced, written independently, real fonts ------
        _refm51: dict = {}

        def _ref_metrics(size_, bold_):
            k_ = (_L51.mono, size_, bold_)              # a substituted `mono` gets its own fonts
            if k_ not in _refm51:
                _refm51[k_] = _FM51(_L51.mono(size_, bold=bold_))
            return _refm51[k_]

        def _runs_of51(t_):
            return t_.runs if isinstance(t_, _R51.RunLine) else (_R51.Run(str(t_)),)

        def _ref_fit51(box_, lines_, start_, bold_, floor_=9):
            s_ = max(floor_, int(start_))
            while True:
                wide_, high_ = 0.0, 0.0
                first_ = True
                for t_ in lines_:
                    w_, hh_ = 0.0, 0.0
                    for r_ in _runs_of51(t_):
                        fm_ = _ref_metrics(s_, bold_ if r_.bold is None else bool(r_.bold))
                        w_ += fm_.horizontalAdvance(r_.text)
                        hh_ = max(hh_, fm_.height())
                    wide_ = w_ if first_ else max(wide_, w_)
                    first_ = False
                    high_ += hh_
                if (wide_ <= box_.width() * 0.94 and high_ <= box_.height() * 0.96) or s_ <= floor_:
                    return s_
                s_ -= 1

        # -- 1. 500 random cases, plain and run-based, against the real font engine ------
        rng51 = _rnd51.Random(20261022)
        WORDS51 = ["SHOTS", "17 of 17 shots", "ROUND 3", "EFFICIENCY  1.87", "AIM 27%   SPEED 37%", "--",
                   "a line far too long for such a box to hold at any size at all", "LEFT", "10.7 seconds",
                   "NEXT ROUND", "1000 of 1500 points  BEST", "NEW BEST SCORE RECORDED", "= 20 points / 10.7 seconds", ""]
        HI51 = _L51.GREY_HI

        def _rand_line51():
            w_ = rng51.choice(WORDS51)
            kind_ = rng51.random()
            if kind_ < 0.45:
                return w_
            if kind_ < 0.85:
                return _R51.emphasise(w_, HI51)
            return _R51.RunLine([_R51.Run("WORD  ", rng51.choice([None, False, True])),
                                 _R51.Run("77", True, HI51), _R51.Run(" tail", rng51.choice([None, False]))])

        mism51, n_run51, n_floor51, n_start51, n_cases51 = [], 0, 0, 0, 500
        sizes51 = set()
        for i_ in range(n_cases51):
            lines_ = [_rand_line51() for _ in range(rng51.randint(1, 6))]
            box_ = _QR51(0, 0, rng51.randint(20, 1500), rng51.randint(10, 600))
            start_ = rng51.choice([rng51.randint(9, 60), rng51.randint(40, 115)])
            bold_ = rng51.random() < 0.5
            floor_ = rng51.choice([9, 9, 9, 10, 12])
            _B51._FITS.clear()                          # a cache hit would compare the answer with itself
            got_ = _B51.ModeHost.fit_size(box_, lines_, start_, bold_, floor_)
            want_ = _ref_fit51(box_, lines_, start_, bold_, floor_)
            n_run51 += any(isinstance(t_, _R51.RunLine) for t_ in lines_)
            n_floor51 += want_ <= floor_
            n_start51 += want_ >= max(floor_, int(start_))
            sizes51.add(want_)
            if got_ != want_:
                mism51.append(f"#{i_} {box_.width()}x{box_.height()} start {start_} bold {bold_} floor {floor_}: "
                              f"{got_} vs walk {want_}")
        check(f"the bisection returns EXACTLY the size the one-point-at-a-time walk returned: {n_cases51} random "
              f"boxes and line sets ({n_run51} with runs), through the real font engine", not mism51,
              "; ".join(mism51[:3]))
        check(f"(premise) those cases reach all three outcomes -- the floor ({n_floor51}), the requested start "
              f"({n_start51}) and sizes in between ({len(sizes51)} distinct sizes)",
              n_floor51 >= 20 and n_start51 >= 20 and len(sizes51) >= 15)

        # -- 2. the real round screens: every block's size, against the walk ------------
        def _ctx51(w_, h_):
            return RenderContext(now=1.0, dt=1 / 60.0, rect=QRect(0, 0, w_, h_), stats=SessionStats())

        def _single51(w_, h_, on_target=True):
            c_ = _ctx51(w_, h_)
            hh_ = _Host()
            hh_.set_mode(_Static(), c_, players=1)
            _fire(hh_, c_, 0, N25, on_target=on_target)
            return hh_, c_

        class _FB51:
            def __init__(self, w_, h_):
                self._w, self._h = w_, h_

            def width(self):
                return self._w

            def height(self):
                return self._h

        def _paint51(h_, c_, img_=None):
            """Paint `h_` and return (every `_draw_lines` call as (box, lines, start, bold, floor, size))."""
            calls_ = []
            o_ = _Host.__dict__["_draw_lines"]
            if img_ is None:
                img_ = _QI51(c_.width, c_.height, _QI51.Format_RGB32)
            img_.fill(0xFF0A0A0A)
            sp_ = _QP51(img_)

            def w_(p_, box_, lines_, start_, bold=False, floor=9):
                size_ = o_.__func__(p_, box_, lines_, start_, bold, floor)
                # the box's own numbers, not a copy rounded to ints: a float width is
                # exactly where a rounding difference between two fits would hide
                calls_.append((_FB51(box_.width(), box_.height()), list(lines_), start_, bold, floor, size_))
                return size_
            _Host._draw_lines = staticmethod(w_)
            try:
                h_.paint(sp_, c_)
            finally:
                _Host._draw_lines = o_
            sp_.end()
            return calls_

        states51 = []
        for (w_, h_) in ((3840, 2160), (1920, 1080)):
            for tag_, on_ in (("hits", True), ("misses", False)):
                hh_, cc_ = _single51(w_, h_, on_)
                states51.append((f"{w_}x{h_} one player, all {tag_}", hh_, cc_))
            hh_, cc_ = _finished25(w_, h_, 2)
            states51.append((f"{w_}x{h_} two players", hh_, cc_))
        screen_bad51, screen_calls51 = [], 0
        for label_, hh_, cc_ in states51:
            if not hh_.round_over:
                screen_bad51.append(f"{label_}: the round did not close")
                continue
            _B51._FITS.clear()
            for box_, lines_, start_, bold_, floor_, size_ in _paint51(hh_, cc_):
                screen_calls51 += 1
                want_ = _ref_fit51(box_, lines_, start_, bold_, floor_)
                if size_ != want_:
                    screen_bad51.append(f"{label_}: {str(lines_[0])[:24]!r} drawn at {size_}, the walk gives {want_}")
        check(f"every block of the real round-over screens is drawn at the size the walk gave ({screen_calls51} "
              f"blocks: one player hits and misses, two players, at 4K and 1080p)",
              not screen_bad51 and screen_calls51 >= 40, "; ".join(screen_bad51[:3]) or f"{screen_calls51} blocks")

        # -- 3. an unchanged screen does NO font work on its second paint ---------------
        # THE CHECK THAT GOES RED WHEN THE CACHE IS REMOVED. Counted at the two places
        # a font can be built (`mono` and the metrics object) and at the search.
        class _CountFM51(_FM51):
            def __init__(self, *a):
                super().__init__(*a)
                cnt51["metrics"] += 1

        cnt51 = {"mono": 0, "metrics": 0, "searches": 0, "probes": 0, "worst": 0}
        orig_mono51, orig_fm51, orig_lf51 = _L51.mono, _QG51.QFontMetricsF, _B51._largest_fitting

        def _counting_mono51(size, bold=False):
            cnt51["mono"] += 1
            return orig_mono51(size, bold)

        def _counting_lf51(fits, start, floor):
            cnt51["searches"] += 1
            n_ = [0]

            def f_(s):
                n_[0] += 1
                cnt51["probes"] += 1
                return fits(s)
            out_ = orig_lf51(f_, start, floor)
            cnt51["worst"] = max(cnt51["worst"], n_[0])
            cnt51.setdefault("ratio_ok", True)
            if n_[0] > (max(1, start - floor).bit_length() + 2):
                cnt51["ratio_ok"] = False
            return out_

        hs51, cs51 = _single51(3840, 2160)
        img51 = _QI51(3840, 2160, _QI51.Format_RGB32)
        _L51.mono, _QG51.QFontMetricsF, _B51._largest_fitting = _counting_mono51, _CountFM51, _counting_lf51
        try:
            _B51._FONTS.clear()
            _B51._FITS.clear()
            first51 = _paint51(hs51, cs51, img51)
            cold51 = dict(cnt51)
            for k_ in ("mono", "metrics", "searches", "probes"):
                cnt51[k_] = 0
            second51 = _paint51(hs51, cs51, img51)
            warm51 = dict(cnt51)
        finally:
            _L51.mono, _QG51.QFontMetricsF, _B51._largest_fitting = orig_mono51, orig_fm51, orig_lf51
        check(f"(premise) the first paint of the round screen did font work ({cold51['mono']} fonts, "
              f"{cold51['metrics']} metrics, {cold51['searches']} searches over {len(first51)} blocks)",
              cold51["mono"] > 0 and cold51["metrics"] > 0 and cold51["searches"] > 0 and len(first51) >= 6)
        check("a SECOND paint of the unchanged round screen constructs NO font, NO metrics object and runs NO size "
              f"search ({warm51['mono']} fonts, {warm51['metrics']} metrics, {warm51['searches']} searches) -- "
              "remove the cache and this is the check that goes red",
              warm51["mono"] == 0 and warm51["metrics"] == 0 and warm51["searches"] == 0)
        check("...and it drew the same sizes both times",
              [c_[5] for c_ in first51] == [c_[5] for c_ in second51])

        # -- 4. a first paint does a fraction of the font work the walk did -------------
        old_fonts51 = 0
        for box_, lines_, start_, bold_, floor_, size_ in first51:
            bolds_ = {bold_ if r_.bold is None else bool(r_.bold) for t_ in lines_ for r_ in _runs_of51(t_)}
            old_fonts51 += (max(floor_, int(start_)) - size_ + 1) * len(bolds_) + 1
        check(f"a COLD paint builds at most half the fonts the walk built for the same blocks "
              f"({cold51['mono']} against {old_fonts51})", cold51["mono"] * 2 <= old_fonts51,
              f"{cold51['mono']} vs {old_fonts51}")
        check(f"every search is logarithmic: the worst took {cold51['worst']} probes, within log2(range)+2 "
              "(the walk took up to ~90)", cold51.get("ratio_ok", False) and cold51["worst"] <= 9,
              f"worst {cold51['worst']}")

        # -- 5. keyed on everything the answer depends on, and bounded ------------------
        orig_mono_k51 = _L51.mono

        def _wide_bold51(size, bold=False):
            return orig_mono_k51(size + 4 if bold else size, False)

        _L51.mono = _wide_bold51
        try:
            rl_ = _R51.emphasise("123456 of 789012 points", HI51)
            plain_ = str(rl_)
            probe_ = _FM51(_wide_bold51(40, False)).horizontalAdvance(plain_)
            bx_ = _QR51(0, 0, int(probe_ / 0.94) + 2, 400)
            _B51._FITS.clear()
            s_plain_ = _B51.ModeHost.fit_size(bx_, [plain_], 40, False)         # cached first ...
            s_runs_ = _B51.ModeHost.fit_size(bx_, [rl_], 40, False)            # ... then same words with a bold run
            s_bold_ = _B51.ModeHost.fit_size(bx_, [plain_], 40, True)
            ok_ = (s_plain_ == _ref_fit51(bx_, [plain_], 40, False) and s_runs_ == _ref_fit51(bx_, [rl_], 40, False)
                   and s_bold_ == _ref_fit51(bx_, [plain_], 40, True))
        finally:
            _L51.mono = orig_mono_k51
        check(f"the cache key holds the runs and the weight: the same words as a plain string ({s_plain_}), with a "
              f"bold run ({s_runs_}) and bold throughout ({s_bold_}) each get their own, correct, size",
              ok_ and s_runs_ < s_plain_ and s_bold_ < s_plain_, f"{s_plain_}/{s_runs_}/{s_bold_}")
        k0_ = len(_B51._FITS)
        for i_ in range(_B51._FITS_MAX * 3):
            _B51.ModeHost.fit_size(_QR51(0, 0, 100 + i_, 80), ["ROUND 3"], 40)
        check(f"the size cache is bounded ({len(_B51._FITS)} entries after {_B51._FITS_MAX * 3} distinct boxes, "
              f"limit {_B51._FITS_MAX}) and the font cache is too ({len(_B51._FONTS)} of {_B51._FONTS_MAX})",
              len(_B51._FITS) <= _B51._FITS_MAX and len(_B51._FONTS) <= _B51._FONTS_MAX, f"was {k0_}")
        last_ = _QR51(0, 0, 100 + _B51._FITS_MAX * 3 - 1, 80)
        n0_ = cnt51["searches"]
        _L51.mono, _QG51.QFontMetricsF, _B51._largest_fitting = orig_mono51, orig_fm51, _counting_lf51
        try:
            _B51.ModeHost.fit_size(last_, ["ROUND 3"], 40)
        finally:
            _B51._largest_fitting = orig_lf51
        check("...and the most recent entry survived the eviction (it is a cache, not a reset)",
              cnt51["searches"] == n0_)

        # -- 6. the MENU button: the same loop, painted on every in-play repaint --------
        from lsa.display.layers import px as _px51

        class _FontSpy51(_QP51):
            def __init__(self, dev):
                super().__init__(dev)
                self.sizes = []

            def setFont(self, f):
                self.sizes.append((f.pointSize(), f.bold()))
                return super().setFont(f)

        btn_bad51, btn_n51 = [], 0
        timg51 = _QI51(64, 64, _QI51.Format_RGB32)
        for h_ in (150, 200, 270, 360, 480, 540, 600, 720, 900, 1080, 1440, 2160):
            for armed_ in (False, True):
                c_ = _ctx51(int(h_ * 16 / 9), h_)
                b_ = _B51.MenuButton()
                if armed_:
                    b_.armed_at = c_.now
                label_ = "MENU - AGAIN" if armed_ else "MENU"
                _B51._FITS.clear()
                sp_ = _FontSpy51(timg51)
                b_.paint(sp_, c_)
                sp_.end()
                width_ = b_.rect(c_).width()
                size_ = _px51(c_, 20)
                f_ = _L51.mono(size_, bold=True)
                while size_ > 9:                                  # the loop that stood here
                    f_ = _L51.mono(size_, bold=True)
                    if _FM51(f_).horizontalAdvance(label_) <= width_ * 0.84:
                        break
                    size_ -= 1
                btn_n51 += 1
                if not sp_.sizes or sp_.sizes[-1] != (f_.pointSize(), True):
                    btn_bad51.append(f"h {h_} {label_!r}: drew {sp_.sizes[-1:] }, the loop gave {f_.pointSize()}")
        check(f"the MENU button is drawn at the size its own walk-down loop gave ({btn_n51} cases, both labels, "
              "1080 and 4K-class and tiny screens)", not btn_bad51, "; ".join(btn_bad51[:3]))

        # -- 7. COST. Measured in this run, against the screen he already shoots on ------
        # BAR, AND WHERE IT COMES FROM. Not a round number: the heaviest screen the app
        # ships is Static Targets (a paint of its own full ring set at 4K); he shot on it
        # at 47.9 fps once its ring numbers were cached (2026-10-20). A round-over screen
        # is text and four outlines, so a WARM paint of it has no business costing more
        # than a WARM paint of that. Both are measured here, together, on this machine,
        # median of 7, so a busy machine moves both and not the ratio.
        # STATED PLAINLY: the offscreen renderer does font work much faster than Windows
        # (the 171 ms ring paint measured ~12 ms offscreen), so this bar is a FLOOR on the
        # real cost, not a guarantee. What goes red when the cache is removed is check 3,
        # which counts work instead of timing it.
        def _median_ms51(h_, c_, reps=7):
            img_ = _QI51(c_.width, c_.height, _QI51.Format_RGB32)
            _paint51(h_, c_, img_)                                # warm: fonts, sizes, pixmaps
            out_ = []
            for _ in range(reps):
                img_.fill(0xFF0A0A0A)
                sp_ = _QP51(img_)
                t0_ = _t51.perf_counter()
                h_.paint(sp_, c_)
                out_.append((_t51.perf_counter() - t0_) * 1000.0)
                sp_.end()
            return _st51.median(out_)

        hr51, cr51 = _single51(3840, 2160)
        sc51 = _ctx51(3840, 2160)
        hst51 = _Host()
        hst51.set_mode(_Static(), sc51, players=1)
        round_ms51 = _median_ms51(hr51, cr51)
        static_ms51 = _median_ms51(hst51, sc51)
        _B51._FONTS.clear()
        _B51._FITS.clear()
        img_c51 = _QI51(3840, 2160, _QI51.Format_RGB32)
        img_c51.fill(0xFF0A0A0A)
        spc51 = _QP51(img_c51)
        t0_ = _t51.perf_counter()
        hr51.paint(spc51, cr51)
        cold_ms51 = (_t51.perf_counter() - t0_) * 1000.0
        spc51.end()
        print(f"         warm round-over paint {round_ms51:.2f} ms, warm Static Targets paint {static_ms51:.2f} ms, "
              f"first (cold) round-over paint {cold_ms51:.2f} ms, at 3840x2160 offscreen")
        check(f"a WARM round-over paint at 4K ({round_ms51:.2f} ms) costs no more than a warm paint of Static "
              f"Targets ({static_ms51:.2f} ms), the heaviest approved screen -- a floor on the real cost, since "
              "the offscreen renderer is faster than Windows at font work",
              round_ms51 <= static_ms51, f"{round_ms51:.2f} vs {static_ms51:.2f}")

    try:
        _section51()
    except Exception as exc_51:                                     # noqa: BLE001 - the finding
        import traceback as _tb51
        check("section 51 ran to its end without raising", False,
              f"{type(exc_51).__name__}: {exc_51}")
        print("".join(_tb51.format_exc().splitlines(True)[-6:]))

    # ------------------------------------------------------------------
    # 52. THE GHOST MUST NOT BE BEATABLE BY MISSING (2026-10-23). `SCORE = 1000 x aim x
    # speed` and the ghost is 1000, so nothing may exceed it. His animated-soldier record of
    # 1146 (17 shots, 14.29 s, 130 points; the ghost took 21.42 s) did, and so did three more of
    # his rounds. The mechanism, measured and not argued: on a drill where DROPPING a target
    # triggers the wait (a figure's fall, the soldier's fall and rest) the ghost never misses, so
    # it drops more targets than the human did and was charged a full fall for each -- including
    # for targets the round never drew, at the drill's average wait. A human who missed dropped
    # fewer, waited less, and was compared with a ghost that waited for a hypothetical better
    # round: on those drills the score paid for missing. The ghost is now charged only the waits
    # the round CONTAINED (`Mode._ghost_wait_plan`).
    #
    # WHAT THIS PROVES, AND HOW IT CAN FAIL. Three families of generated rounds go through the
    # real plan and the real `ghost_score`:
    #   A. a human uniformly `s` times slower than the ghost: his speed share must be EXACTLY
    #      1/s whatever he misses (the old plan returns more than 1/s, rising as he misses more);
    #   B. the same human missing more must never read FASTER (monotone in misses);
    #   C. thousands of random physical humans (cadence >= 0.25 s, reaction >= one frame, waits
    #      from nothing to the soldier's 4.6 s, any hit pattern, a target spawned AFTER the last
    #      shot): the ghost never takes longer than the human and no score exceeds 1000.
    # and a NEGATIVE CONTROL runs the OLD plan (re-written below, never imported) over a world
    # chosen to expose it: if the generator could not see the defect, a green A/B/C would prove
    # nothing. Reverting `_ghost_wait_plan` to price a missing target at `ghost_expected_wait`
    # turns A, B, the fixture rows and the host checks red by name.
    # ------------------------------------------------------------------
    def _section52() -> None:
        print()
        print("52. the ghost is charged only the waits the round contained: missing cannot raise the score")
        import random as _rnd52
        from lsa import ghost as _G52
        from lsa.modes import base as _base52
        from lsa.modes.animated import AnimatedFigure as _AN52
        from lsa.modes.ball import EscalatingBall as _EB52
        from lsa.modes.balls import BouncingBalls as _BB52
        from lsa.modes.base import Mode as _Mode52

        F52 = 1.0 / 60.0                  # one camera frame at the configured 60 fps
        MEAN52 = 2.0                      # what the OLD plan priced a never-drawn target at

        class _Wait52(_Mode52):
            """A drill that makes you wait, with `per` perfect shots to a target, a flat
            ceiling of 10, and a declared mean wait (which the plan must NOT consult)."""
            ghost_has_waits = True

            def __init__(self, per=1, expected=MEAN52):
                super().__init__()
                self.per = per
                self.expected = expected

            def ghost_ceiling(self):
                return _G52.constant_ceiling(10)

            def ghost_shots_per_target(self):
                return self.per

            def ghost_expected_wait(self, target_index):
                return self.expected

        def _old_plan52(mode, shots):
            """THE PLAN THIS REPLACED, re-written here as a negative control (never imported:
            it is the defect). A target the round never drew is priced at the declared mean."""
            out = []
            for i in range(shots):
                if i % mode.per:
                    out.append(0.0)
                    continue
                j = i // mode.per
                if j < len(mode._targets):
                    a_, r_ = mode._targets[j]
                    out.append(0.0 if r_ is None else max(0.0, a_ - r_))
                else:
                    out.append(float(mode.ghost_expected_wait(j)))
            return out

        def _sim52(n, per, hits, waits, gap_wait, gap_plain, t0=100.0):
            """A scripted human on a drill where resolving a target starts a wait for the next.
            Returns (targets as `note_target` would record them, the n shot times). He only
            ever fires at a target that exists. When the LAST shot resolves a target the mode
            still spawns the next one -- after the magazine is empty."""
            targets = [(t0 - 1.0, None)]
            times = [t0]
            on, j, resolved = 0, 0, False
            for i in range(n):
                if i > 0:
                    if resolved:
                        j += 1
                        w_ = waits[j]
                        targets.append((times[-1] + w_, times[-1]))
                        times.append(times[-1] + gap_wait(w_))
                    else:
                        times.append(times[-1] + gap_plain)
                resolved = False
                if hits[i]:
                    on += 1
                    if on == per:
                        on, resolved = 0, True
            if resolved:
                j += 1
                targets.append((times[-1] + waits[j], times[-1]))     # spawned after the last shot
            return targets, times

        def _score52(per, n, hits, targets, times, old=False):
            m_ = _Wait52(per)
            for a_, r_ in targets:
                m_.note_target(a_, r_)
            he_ = times[-1] - times[0]
            if old:
                av_ = _G52.available_from_waits(_old_plan52(m_, n), 60.0)
            else:
                av_ = m_.ghost_available_at(n, 60.0, within=(times[0], times[-1]))
            return _G52.ghost_score(human_points=10 * sum(hits), human_elapsed=he_, shots=n,
                                    ceiling=m_.ghost_ceiling(), has_waits=True, available_at=av_,
                                    declared_waits=m_.ghost_declared_waits(n)), m_

        # -- a. the clock clip, as a pure function -----------------------------------------
        W52 = _base52._wait_inside_clock
        check("a wait is the part of [released, appeared] inside the human's clock: nothing waited for -> 0; "
              "no clock known -> the whole wait; before the first shot -> only the part after it; a target that "
              "appeared AFTER the last shot -> 0 (the mode spawns it whether or not anybody still shoots)",
              W52(5.0, None, (0.0, 9.0)) == 0.0 and W52(5.0, 2.0, None) == 3.0
              and W52(10.0, 4.0, (8.0, 12.0)) == 2.0 and W52(10.0, 4.0, (0.0, 12.0)) == 6.0
              and W52(25.0, 22.0, (10.0, 22.0)) == 0.0 and W52(22.0, 22.0, (10.0, 22.0)) == 0.0
              and W52(5.0, 9.0, (0.0, 20.0)) == 0.0,
              f"{W52(10.0, 4.0, (8.0, 12.0))} {W52(25.0, 22.0, (10.0, 22.0))}")
        bad52 = _Wait52(1)
        bad52.note_target(0.0, None)
        bad52.note_target(5.0, 2.0)
        check("a clock that is not a clock (reversed, NaN, not numbers, wrong length) makes the plan None -- a "
              "lower bound, loudly -- and never a plan built on it; a good clock gives one",
              all(bad52.ghost_waits(2, w_) is None
                  for w_ in ((5.0, 1.0), (float("nan"), 1.0), (0.0, float("inf")), ("a", 1), (1.0,), 7))
              and bad52.ghost_waits(2, (0.0, 9.0)) == [0.0, 3.0] and bad52.ghost_waits(2) == [0.0, 3.0])

        # -- b. an unsupplied target is free; nothing is ever declared -----------------------
        few52 = _Wait52(2)
        for a_, r_ in ((0.0, None), (6.0, 2.0)):
            few52.note_target(a_, r_)
        check("two targets, six shots, two shots to a target: the ghost is charged the ONE wait the round "
              "contained (before shot 3) and nothing for the third target the round never drew -- not "
              "MEAN52, the drill's declared mean",
              few52.ghost_waits(6, (0.0, 20.0)) == [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
              and few52.ghost_expected_wait(2) == MEAN52 and few52.ghost_declared_waits(6) == 0)

        # -- c. A and B: the uniform family -- missing may never read faster -----------------
        rng52 = _rnd52.Random(5252)
        miss_ok52, worst52, cases52 = [], 0.0, 0
        for s52 in (1.0, 1.5, 3.0):
            for per52 in (1, 2, 4):
                for n52 in (5, 9, 17):
                    waits52 = [0.0] + [rng52.choice((0.0, 0.1, 1.0, 2.5, 4.6)) for _ in range(n52 + 3)]
                    gw52 = (lambda w_, s_=s52: s_ * max(0.25, w_ + F52))
                    ss52 = []
                    for k_ in sorted({0, 1, n52 // 3, n52 // 2, n52 - 1, n52}):
                        hits52 = [True] * n52
                        for p_ in rng52.sample(range(n52), k_):
                            hits52[p_] = False
                        tg_, tm_ = _sim52(n52, per52, hits52, waits52, gw52, 0.25 * s52)
                        gs_, _m = _score52(per52, n52, hits52, tg_, tm_)
                        cases52 += 1
                        if not (gs_.scorable and gs_.speed_raw is not None):
                            miss_ok52.append((s52, per52, n52, k_, "not scorable"))
                            continue
                        # THE RAW share, not `speed_share`: the latter is clamped at 1.0 (2026-10-23),
                        # so for s = 1 a plan that charged the ghost too little would read exactly 1/s
                        # here and this check would pass over the very defect it exists to catch.
                        ss52.append(gs_.speed_raw)
                        worst52 = max(worst52, abs(gs_.speed_raw - 1.0 / s52))
                        if abs(gs_.speed_raw - 1.0 / s52) > 1e-9:
                            miss_ok52.append((s52, per52, n52, k_, round(gs_.speed_raw, 4)))
                    if ss52 and max(ss52) > ss52[0] + 1e-9:
                        miss_ok52.append((s52, per52, n52, "monotone", [round(v_, 4) for v_ in ss52]))
        check(f"A. a human uniformly s times slower than the ghost (s = 1, 1.5, 3) has a speed share of "
              f"EXACTLY 1/s whatever he misses, over {cases52} generated rounds -- the old plan reads above "
              f"it, and higher the more he misses (worst deviation {worst52:.2e})",
              cases52 >= 150 and not miss_ok52, str(miss_ok52[:3]))
        check("B. the same human missing more is never FASTER: the speed share of k misses never exceeds "
              "that of none (checked in the same loop, per world)",
              not any(m_[3] == "monotone" for m_ in miss_ok52))

        # the negative control: the same family through the OLD plan must show the defect
        hits_c52 = [True, False] * 8 + [True]
        wc52 = [0.0] + [2.5] * 20
        tg_c, tm_c = _sim52(17, 4, hits_c52, wc52, lambda w_: max(0.25, w_ + F52), 0.25)
        tg_a, tm_a = _sim52(17, 4, [True] * 17, wc52, lambda w_: max(0.25, w_ + F52), 0.25)
        old_c52, _o = _score52(4, 17, hits_c52, tg_c, tm_c, old=True)
        new_c52, _n = _score52(4, 17, hits_c52, tg_c, tm_c)
        old_a52, _o2 = _score52(4, 17, [True] * 17, tg_a, tm_a, old=True)
        check("NEGATIVE CONTROL: the same rounds through the OLD plan show the defect this section exists for -- "
              "a human at the ghost's own pace who MISSES reads a speed share above 1 and above his share when "
              "he misses nothing; through the new plan it is 1 exactly",
              old_c52.speed_raw is not None and old_c52.speed_raw > 1.2
              and old_a52.speed_raw is not None and old_c52.speed_raw > old_a52.speed_raw + 0.1
              and new_c52.speed_raw is not None and abs(new_c52.speed_raw - 1.0) < 1e-9
              and old_c52.clamped and not new_c52.clamped,       # shares are clamped; the RAW one shows it
              f"old {fx52(old_c52.speed_raw)} (hit all: {fx52(old_a52.speed_raw)}), "
              f"new {fx52(new_c52.speed_raw)}")

        # -- d. C: random physical humans -------------------------------------------------
        rng_c52 = _rnd52.Random(2310)
        bad_c52, n_c52, max_score52, max_ratio52, declared_c52 = [], 0, 0.0, 0.0, 0
        for _w in range(1500):
            n_ = rng_c52.randint(2, 30)
            per_ = rng_c52.choice((1, 2, 4))
            p_ = rng_c52.choice((1.0, 0.8, 0.5, 0.2, 0.0))
            hits_ = [rng_c52.random() < p_ for _i in range(n_)]
            waits_ = [0.0] + [rng_c52.choice((0.0, 0.05, 0.3, 1.0, 2.5, 4.6)) * rng_c52.uniform(0.5, 1.5)
                              for _i in range(n_ + 3)]
            c_ = 0.25 + rng_c52.expovariate(2.0)
            r_ = F52 + rng_c52.expovariate(1.0)
            tg_, tm_ = _sim52(n_, per_, hits_, waits_, lambda w_, c=c_, r=r_: max(c, w_ + r), c_)
            gs_, m_ = _score52(per_, n_, hits_, tg_, tm_)
            n_c52 += 1
            declared_c52 += gs_.declared_waits
            if not gs_.scorable:
                bad_c52.append((n_, per_, "not scorable", gs_.not_scorable[:40]))
                continue
            max_score52 = max(max_score52, gs_.score)
            max_ratio52 = max(max_ratio52, gs_.ghost_elapsed / gs_.human_elapsed)
            if (gs_.ghost_elapsed > gs_.human_elapsed + 1e-9 or gs_.score > 1000.0 + 1e-6
                    or gs_.clamped or gs_.lower_bound):          # clamped: the 1000 ceiling must not be what saved it
                bad_c52.append((n_, per_, round(gs_.ghost_elapsed, 3), round(gs_.human_elapsed, 3),
                                round(gs_.score, 1)))
        check(f"C. over {n_c52} random physical humans (cadence >= 0.25 s, reaction >= one frame, waits from "
              f"nothing to 4.6 s, any hit pattern, a target spawned after the last shot) the ghost never takes "
              f"longer than the human and no score exceeds 1000 (highest score {max_score52:.1f}, largest "
              f"ghost/human {max_ratio52:.4f}); none is a lower bound; none declares a wait",
              n_c52 == 1500 and not bad_c52 and declared_c52 == 0, str(bad_c52[:3]))

        # -- e. HIS ROUNDS: the stored 1146 and the three that beat the ghost ---------------
        an52 = _AN52()
        an52.start(context(now=0.0))
        E52 = an52.ghost_expected_wait(1)
        per_an52 = an52.ghost_shots_per_target()
        # The stored record, copied verbatim: settings.local.json, bests "animated|m17|p1",
        # score, `when` 2026-10-22T18:09:16, basis "declared", declared_waits 2. The file is
        # his and is NOT read (a check that reads the world fails when nothing is wrong).
        REC52 = dict(value=1146.4967262729454, points=130, seconds=14.2876, shots=17,
                     ghost_points=170, ghost_seconds=21.4209, declared_waits=2)
        k_all52 = -(-REC52["shots"] // per_an52) - 1               # waits the OLD ghost was charged
        check("(premise) the stored record is impossible on its face: its ghost took longer than the round "
              "(speed share 21.4209 / 14.2876 = 1.499), and its value is the product of the two factors it "
              "stores -- a score that could only be earned by the ghost waiting for soldiers he never shot",
              REC52["ghost_seconds"] > REC52["seconds"]
              and abs(1000.0 * REC52["points"] / REC52["ghost_points"]
                      * REC52["ghost_seconds"] / REC52["seconds"] - REC52["value"]) < 0.5
              and E52 is not None and per_an52 == 4 and k_all52 == 4,
              f"E {E52}, per {per_an52}, K {k_all52}")
        # RECONSTRUCTION: 4 waits at 0.25 s a shot give 21.4209 s only if the two recorded waits
        # and the two declared ones (E each) sum to what is left. Solve for the recorded pair.
        n52 = REC52["shots"]
        sum_all52 = (REC52["ghost_seconds"] - (n52 - 1) * 0.25 - k_all52 * (F52 - 0.25))
        w_rec52 = (sum_all52 - REC52["declared_waits"] * E52) / (k_all52 - REC52["declared_waits"])
        check("(premise) the reconstruction is consistent: the two recorded waits come out as one fall-and-rest "
              "each, within half a second of the declared one, and fit inside his 14.29 s with the cadence",
              0.0 < w_rec52 and abs(w_rec52 - E52) < 0.5
              and (n52 - 1) * 0.25 + 2 * (w_rec52 + F52 - 0.25) <= REC52["seconds"],
              f"recorded wait {w_rec52:.3f}, declared {E52:.3f}")
        fm52 = _Wait52(per_an52, E52)           # the OLD plan prices a missing soldier at THE REAL E
        fm52.note_target(0.0, None)
        fm52.note_target(2.0 + w_rec52, 2.0)
        fm52.note_target(7.0 + w_rec52, 7.0)
        old_w52 = _old_plan52(fm52, n52)
        old_el52 = _G52.ghost_elapsed(n52, _G52.available_from_waits(old_w52, 60.0))[0]
        new_gs52 = _G52.ghost_score(
            human_points=REC52["points"], human_elapsed=REC52["seconds"], shots=n52,
            ceiling=_G52.constant_ceiling(10), has_waits=True,
            available_at=fm52.ghost_available_at(n52, 60.0, within=(0.0, REC52["seconds"])),
            declared_waits=fm52.ghost_declared_waits(n52))
        check("HIS 1146 (animated, 2026-10-22 18:09:16): the old plan reproduces the stored ghost of 21.42 s on "
              "this reconstruction; the plan now charges the two waits his round contained and reads "
              "speed share < 1, a score under 1000, declared 0",
              abs(old_el52 - REC52["ghost_seconds"]) < 1e-3
              and new_gs52.scorable and new_gs52.speed_raw is not None and new_gs52.speed_raw < 1.0
              and not new_gs52.clamped
              and new_gs52.score is not None and new_gs52.score < 1000.0
              and new_gs52.declared_waits == 0 and not new_gs52.lower_bound,
              f"old ghost {old_el52:.3f}s, new ghost {fx52(new_gs52.ghost_elapsed)}s, "
              f"new score {fx52(new_gs52.score)} (stored {REC52['value']:.1f})")
        rows52 = (("speed 2.86: 7.47 s against a ghost of 21.40 s", 7.47, 21.40),
                  ("speed 1.50: 14.29 s against a ghost of 21.42 s (the stored 1146)", 14.2876, 21.4209),
                  ("speed 1.12: 7.53 s against a ghost of 8.42 s", 7.53, 8.42))
        bad_rows52, raw_rows52 = [], []
        for label_, he_, old_g_ in rows52:
            per_wait_ = E52 + F52 - 0.25
            k_old_ = round((old_g_ - (n52 - 1) * 0.25) / per_wait_)            # waits the old ghost paid
            k_fit_ = int((he_ - (n52 - 1) * 0.25) // per_wait_)                # the most his clock can hold
            mm_ = _Wait52(per_an52)
            mm_.note_target(0.0, None)
            at_ = 0.5
            for _k in range(k_fit_):                                           # as many full waits as fit
                mm_.note_target(at_ + E52, at_)
                at_ += E52 + 0.5
            mm_.note_target(he_ + 30.0, he_)                                   # spawned long after his last shot
            gs_ = _G52.ghost_score(
                human_points=130, human_elapsed=he_, shots=n52, ceiling=_G52.constant_ceiling(10),
                has_waits=True, available_at=mm_.ghost_available_at(n52, 60.0, within=(0.0, he_)),
                declared_waits=mm_.ghost_declared_waits(n52))
            if not (k_old_ > k_fit_ and gs_.scorable and gs_.ghost_elapsed <= he_ + 1e-9
                    and gs_.speed_raw is not None and gs_.speed_raw <= 1.0 + 1e-9
                    and gs_.score <= 1000.0):
                bad_rows52.append((label_, k_old_, k_fit_, fx52(gs_.ghost_elapsed), fx52(gs_.score)))
            raw_rows52.append(f"{label_.split(':')[0]}: raw speed {fx52(gs_.speed_raw)}, "
                              f"clamped={bool(gs_.clamped)}")
        check("the three rounds that beat the ghost (speed 2.86, 1.50, 1.12): the old ghost paid for more "
              "falls than his clock could hold (the mechanism), and with as many waits as his clock CAN hold "
              "the ghost no longer takes longer than he did -- every one has a RAW speed share of 1 or less, "
              "so none NEEDS the 1000-ceiling clamp (the wait fix alone was enough for all three; the detail "
              "prints each raw share)",
              not bad_rows52 and len(raw_rows52) == 3, str(bad_rows52) if bad_rows52 else "; ".join(raw_rows52))

        # -- f. the host seam: the clock is passed, the late spawn is not charged, a defect is loud
        def _host52(shots, targets, avail=None, has_waits=True):
            h_ = _Host()
            h_.set_magazine(4)
            h_.hold_s = REAL_HOLD_S
            h_.fps = 60.0
            h_.set_mode(_Static(), ctx44, players=1)
            m_ = h_.mode
            m_.ghost_has_waits = has_waits
            for a_, r_ in targets:
                m_.note_target(a_, r_)
            if avail is not None:
                m_.ghost_available_at = avail
            for i_, (t_, hit_) in enumerate(shots):
                if not hit_:
                    x_, y_ = 60.0, 1500.0
                elif hasattr(m_, "aim_point"):
                    x_, y_ = m_.aim_point(ctx44)
                else:
                    x_, y_ = m_.x, m_.y
                h_.shot(modes.ShotEvent(x_, y_, t_, i_ + 1, t_end=t_ + 0.05, peak=85.0), context(now=t_))
            h_.advance(context(now=shots[-1][0] + REAL_HOLD_S + 1.0))
            return h_

        leak52 = _host52([(10.0, True), (14.0, True), (18.0, False), (22.0, True)],
                         [(5.0, None), (13.0, 10.0), (17.0, 14.0), (25.0, 22.0)])
        gl52 = leak52.cards[0].gs if leak52.cards[0] is not None else None
        want_el52 = 0.25 + 2 * (3.0 + F52)
        check("through the HOST, a round whose last hit made the mode spawn a target 3 s later: the ghost is "
              "charged the two waits the human's clock held and NOT the third, which appeared after his last "
              "shot (6.28 s, not 9.30 s), with nothing declared and no defect logged",
              gl52 is not None and gl52.scorable and abs(gl52.ghost_elapsed - want_el52) < 1e-9
              and gl52.declared_waits == 0 and not any("DEFECT" in ln_ for _k, ln_ in leak52._events),
              f"{None if gl52 is None else gl52.ghost_elapsed} vs {want_el52}")
        loud52 = _host52([(10.0, True), (14.0, True), (18.0, True), (22.0, True)],
                         [(5.0, None), (13.0, 10.0), (17.0, 14.0), (21.0, 18.0)],
                         avail=lambda n_, f_, within=None: _G52.available_from_waits(
                             [0.0] + [10.0] * (n_ - 1), f_))
        gd52 = loud52.cards[0].gs if loud52.cards[0] is not None else None
        defects52 = [ln_ for _k, ln_ in loud52._events if "DEFECT" in ln_]
        check("a round where the ghost STILL ends up slower than the human (here: waits forced to 10 s) is "
              "held at 1000 by the ceiling (speed share exactly 1.0, score at most 1000) BUT NOT QUIETLY: "
              "the log carries one DEFECT line with the RAW share (above 2), the word CLAMPED, the ghost's and "
              "the human's seconds, and the three things that can be wrong",
              gd52 is not None and gd52.speed_share == 1.0 and gd52.score is not None
              and gd52.score <= 1000.0 and gd52.speed_raw is not None and gd52.speed_raw > 2.0
              and gd52.clamped == (_G52.CLAMPED_SPEED,)
              and len(defects52) == 1 and "CLAMPED" in defects52[0]
              and f"raw speed share {gd52.speed_raw:.4f}" in defects52[0]
              and f"ghost {gd52.ghost_elapsed:.3f}s" in defects52[0]
              and f"human {gd52.human_elapsed:.3f}s" in defects52[0]
              and "GHOST_INTERVAL" in defects52[0] and "wait" in defects52[0] and "ceiling" in defects52[0],
              f"{None if gd52 is None else gd52.speed_raw} / {defects52}")
        # The CADENCE case: a drill that makes nobody wait, a human a hair faster than the
        # ghost's 0.25 s (0.24 s a shot, a span of 0.72 s against a discard floor of 0.68 s,
        # so the round is SCORED). This is the clamp firing for the other reason, and the
        # old DEFECT line only existed for drills with waits and said nothing here.
        cad52 = _host52([(10.0 + 0.24 * i_, True) for i_ in range(4)], [], has_waits=False)
        gc52 = cad52.cards[0].gs if cad52.cards[0] is not None else None
        defects_c52 = [ln_ for _k, ln_ in cad52._events if "DEFECT" in ln_]
        check("the DEFECT line is not only for drills that make you wait: a no-wait drill at 0.24 s a shot "
              "(faster than the ghost's 0.25 s cadence but over the discard floor) is scored 1000 or less, "
              "with the clamp recorded and the raw share (0.75 / 0.72 = 1.0417) in exactly one DEFECT line",
              gc52 is not None and gc52.scorable and gc52.speed_share == 1.0
              and gc52.score is not None and gc52.score <= 1000.0
              and gc52.speed_raw is not None and abs(gc52.speed_raw - 0.75 / 0.72) < 1e-6
              and len(defects_c52) == 1 and "raw speed share 1.04" in defects_c52[0],
              f"{None if gc52 is None else gc52.speed_raw} / {defects_c52}")
        # ... and a DISCARDED round is not a DEFECT: a round faster than the ghost is what the
        # discard is for, and the log's describe() line already says so.
        disc52 = _host52([(10.0 + 0.1 * i_, True) for i_ in range(4)], [], has_waits=False)
        gx52 = disc52.cards[0].gs if disc52.cards[0] is not None else None
        check("a DISCARDED round (0.1 s a shot) logs no DEFECT, and its describe() line keeps the raw speed "
              "next to the capped withheld figure",
              gx52 is not None and gx52.discarded
              and not [ln_ for _k, ln_ in disc52._events if "DEFECT" in ln_]
              and any("was clamped to 1.000" in ln_ for _k, ln_ in disc52._events),
              f"{None if gx52 is None else gx52.speed_raw}")

        # -- g. what must NOT have moved ---------------------------------------------------
        check("the drills that make nobody wait are untouched: both ball drills and Static Targets still supply "
              "no waits whatever clock they are handed (the ball rounds, ghost 4.00 s, read as before)",
              all(m_.ghost_waits(17, (0.0, 14.5)) is None and m_.ghost_available_at(17, 60.0, within=(0.0, 14.5)) is None
                  for m_ in (_EB52(), _BB52(), _Static())))
        check("and EVERY registered drill declares 0 waits, for any round length (the declaration is retired, "
              "not hidden behind a flag)",
              all(c_().ghost_declared_waits(n_) == 0
                  for c_ in modes.available() for n_ in (1, 2, 17, 50)))

    def fx52(v):
        return "None" if v is None else format(v, ".3f")

    try:
        _section52()
    except Exception as exc_52:                                     # noqa: BLE001 - the finding
        import traceback as _tb52
        check("section 52 ran to its end without raising", False,
              f"{type(exc_52).__name__}: {exc_52}")
        print("".join(_tb52.format_exc().splitlines(True)[-6:]))

    passed = all(checks)
    print("\n" + (f"ALL {len(checks)} CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
