"""The figure drills (modes 7 pop-up, 8 image target, 9 advancing target), driven by
synthetic shots. No camera, no display.

Split out of `test_modes.py` so that the mutation runner (`tests/mutate_figures.py`)
can run just these checks: every figure-drill fault is shown to turn them red, and
that only stays cheap enough to keep if the suite that has to notice is small.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2                                                         # noqa: E402,F401
import numpy as np                                                 # noqa: E402,F401
from PySide6.QtCore import QRect                                   # noqa: E402
from PySide6.QtGui import QImage as _QI3, QPainter as _QP3         # noqa: E402,F401
from PySide6.QtWidgets import QApplication                         # noqa: E402

import lsa.modes as modes                                          # noqa: E402
from lsa.display.scene import RenderContext                        # noqa: E402
from lsa.display.session import SessionStats                       # noqa: E402

from lsa.display import layers as _LYC                             # noqa: E402  (the brightness ceiling)

from figure_helpers import (SW, SH, context, drive_to_live as _drive_to_live,   # noqa: E402
                            full as _full, zone_point as _zone_point)

from lsa.artwork import catalog as _fcat, load as _fload                # noqa: E402


def _heaviest() -> str:
    """The figure with the most lit ink: the worst case for the paint budget.

    NAMED `soldier01` until 2026-10-18, when he removed it and the whole
    suite died with FileNotFoundError. A fixture chosen for a PROPERTY
    ("the biggest, brightest figure") must be found by that property, or
    it is a filename pinned in six places that breaks when he does some
    artwork -- the same fault as the twelve-figure counts in test_artwork.
    """
    best, worst = None, -1.0
    for n in _fcat.names():
        g = _fload.load_grey(_fcat.path_for(n))
        v = _fload.lit_fraction(g, SW, SH, 1.0)
        if v > worst:
            best, worst = n, v
    return best


HEAVIEST = _heaviest()


def main() -> int:
    QApplication.instance() or QApplication([])
    checks: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    # ----------------------------------------------------------------------
    # Plan 02 steps 17a-20: the figure drills (7 pop-up, 8 image, 9 advancing).
    # ----------------------------------------------------------------------
    print()
    print("18. mode 8, the image target: size, scoring, randomness, repaints")
    from lsa.artwork import catalog as _cat
    from lsa.modes.image_target import ImageTarget
    from lsa.modes.advancing import AdvancingTarget
    from lsa.modes.popup import PopUpFigure
    from lsa.modes import figure as _figmod
    from lsa.modes.figure import Figure as _Fig, COMBINED_BUDGET as _CB
    FIG_NAMES = _cat.names()
    BUDGET = 0.015
    # Sections 18-26 predate the fall pose and shoot strings far past the drop
    # total expecting the next figure at once. They are about other things, so
    # they run with falls OFF (FALL_S = 0 is the documented off switch); section
    # 27 turns them on and tests them.
    _FALL_S_REAL = _figmod.FALL_S
    _figmod.FALL_S = 0.0

    # -- every one of the twelve, at the size the mode actually chooses -------
    lit_by_fig = {}
    worst_red_fig = 0
    for nm in FIG_NAMES:
        m18 = ImageTarget()
        m18._rng.seed(3)
        m18.start(context(now=100.0), figure=nm)
        # the figure alone (the standing hint, drawn for the first 8 s, is measured
        # with it just below: text on top of the figure's own budget)
        hint_on = m18.started_at
        m18.started_at = -1e9
        r18, l18, img18 = _full(m18.paint, context(now=100.0))
        m18.started_at = hint_on
        rh18, lh18, _ih = _full(m18.paint, context(now=100.0))
        check(f"image/{nm}: with the standing hint up {lh18 * 100:.3f}% (<= 2%), red excess {rh18}",
              lh18 <= 0.02 and rh18 == 0)
        lit_by_fig[nm] = (l18, m18._height / SH)
        worst_red_fig = max(worst_red_fig, r18)
        check(f"image/{nm}: drawn at {m18._height / SH:.3f} of the screen height, lights "
              f"{l18 * 100:.3f}% (<= {BUDGET * 100:.1f}%), red excess {r18}",
              l18 <= BUDGET + 1e-9 and r18 == 0)
    worst_fig = max(lit_by_fig, key=lambda k: lit_by_fig[k][0])
    print(f"         worst single figure: {worst_fig} at {lit_by_fig[worst_fig][0] * 100:.3f}% lit; "
          f"worst red excess over all twelve: {worst_red_fig}")

    # -- the figure is never dimmed, and an over-budget size is caught --------
    check("INK_SCALE is 1.0: his artwork is not dimmed by default",
          _figmod.INK_SCALE == 1.0)
    big = ImageTarget()
    big._rng.seed(3)
    big.start(context(now=100.0), figure=HEAVIEST)
    big._height = 0.85 * SH
    big._fig.sprite.set_range(big._height)
    _rb, lit_big, _ib = _full(big.paint, context(now=100.0))
    check(f"negative control: {HEAVIEST} (the heaviest figure) forced to 0.85 of "
          "the height breaches the budget "
          "(so the per-image check can fail)", lit_big > BUDGET, f"{lit_big * 100:.2f}%")

    # -- an image mode that is idle asks for nothing ---------------------------
    im = ImageTarget()
    im._rng.seed(5)
    im.start(context(now=100.0), figure="thug01")
    asked = sum(1 for i in range(180) if im.advance(context(now=100.0 + i / 60.0)))
    check("image idle: at most 5 repaints requested in 180 ticks (zero while holding)",
          asked <= 5 and im.animated is False, f"{asked}")

    # -- scoring ---------------------------------------------------------------
    pt = _zone_point(im)
    out = im.shot(modes.ShotEvent(pt[0], pt[1], 101.0, 1), context(now=101.0))
    check("a shot at the centre of the 10 ring scores 10 and names it",
          out.hit and out.points == 10 and out.target_id == "thug01:10",
          f"{out.points} {out.target_id}")
    near = im.nearest(modes.ShotEvent(pt[0] + 20.0, pt[1], 101.0, 2), context(now=101.0))
    check("nearest() agrees: that ring, 20px from its centre, inside its radius",
          near is not None and near[0] == "thug01:10" and abs(near[1] - 20.0) < 1e-6
          and near[1] < near[2], str(near))
    # -- WHAT SCORES IS WHAT IS DRAWN: the printed digit row, through shot() ----
    def _at(fig_, centre_, h_, ix, iy):
        """Screen point of image pixel (ix, iy) of the 1792x2400 artwork."""
        left_, top_, w_, hh_ = fig_.box(centre_, h_)
        return (left_ + ix / 1791.0 * w_, top_ + iy / 2399.0 * hh_)

    row = []
    for k_, (ix_, want_) in enumerate(((410, 7), (516, 8), (606, 9), (800, 10),
                                       (980, 9), (1076, 8), (1200, 7))):
        x_, y_ = _at(im._fig, im._centre, im._height, ix_, 1200)
        row.append(im.shot(modes.ShotEvent(x_, y_, 101.05 + k_ * 0.01, 10 + k_),
                           context(now=101.05 + k_ * 0.01)).points)
    check("through the mode's real shot(): the printed digits 7 8 9 10 9 8 7 across the "
          "chest score exactly those numbers", row == [7, 8, 9, 10, 9, 8, 7], str(row))
    x_, y_ = _at(im._fig, im._centre, im._height, 800, 255)
    above_ = im.shot(modes.ShotEvent(x_, y_, 101.2, 30), context(now=101.2))
    check("a shot on the printed '7' ABOVE the head dome (no ring is drawn there) scores "
          "nothing, and says so: on the figure, no ring",
          not above_.hit and above_.target_id == "thug01:no-ring", str(above_))
    out = im.shot(modes.ShotEvent(*_at(im._fig, im._centre, im._height, 20, 20), 101.3, 31),
                  context(now=101.3))
    check("a shot on the figure but outside every ring is a MISS, not a hit",
          not out.hit and out.target_id is None)
    out = im.shot(modes.ShotEvent(5.0, 5.0, 101.4, 4), context(now=101.4))
    check("a shot nowhere near is a miss", not out.hit)
    check("the tally counts hits and shots honestly (1 + 7 digit-row hits, 3 misses; "
          "the digit-row 10 is the second 10)",
          im.hits == 8 and im.shots == 11 and im.by_zone == {"10": 2, "9": 2, "8": 2, "7": 2},
          f"{im.hits}/{im.shots} {im.by_zone}")
    check("the status line carries the name, every zone count, score and shots",
          all(s in im.status() for s in ("thug01", "10:2", "9:2", "8:2", "7:2",
                                         "score 68", "11 shots")), im.status())
    flash_on = im.animated is True
    ticks = sum(1 for i in range(60) if im.advance(context(now=101.0 + 0.3 + i / 60.0)))
    check("a hit flashes (animated), and the flash stops by itself",
          flash_on and ticks <= int(im.FLASH_S * 60) and im.animated is False, f"{ticks} ticks")

    # -- a figure that cannot be judged is never drawn --------------------------
    import shutil as _sh
    import tempfile as _tf
    import json as _js
    tmp_cast = Path(_tf.mkdtemp(prefix="lsa_modes_cast_"))
    saved_env = os.environ.get("LSA_TARGETS")
    try:
        from lsa.artwork import load as _ld
        for nm_ in ("thug01", "thug02", "thug03"):
            _sh.copy(_cat.path_for(nm_), tmp_cast / f"{nm_}.jpg")
        _sh.copy(_ld.targets_dir() / "thug01.zones.json", tmp_cast / "thug01.zones.json")
        broken = _js.loads((_ld.targets_dir() / "thug02.zones.json").read_text())
        broken["scored"] = False
        broken["why_not"] = ["C3: test"]
        (tmp_cast / "thug02.zones.json").write_text(_js.dumps(broken))   # thug03: none at all
        os.environ["LSA_TARGETS"] = str(tmp_cast)
        seen = set()
        for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
            m_ = cls_()
            m_._rng.seed(77)
            m_.start(context(now=0.0))
            if cls_ is ImageTarget:
                for _ in range(40):
                    m_.start(context(now=0.0))
                    seen.add(m_._fig.name)
            else:
                t_ = 0.0
                while t_ < 120.0:
                    m_.advance(context(now=t_))
                    t_ += 1 / 20.0
                seen.update(m_.shown)
        check("with one usable figure of three, no mode ever draws the two that cannot be "
              "judged (40 rounds, 120 s of pop-ups and walks)", seen == {"thug01"}, str(seen))
        for nm_ in ("thug01", "thug02", "thug03"):
            (tmp_cast / f"{nm_}.jpg").unlink()
        m_ = ImageTarget()
        m_.start(context(now=0.0))
        asked_ = m_.advance(context(now=0.1))
        check("with NO usable figure the mode says so and does not crash",
              "no artwork" in m_.status() and asked_ is False)
    finally:
        if saved_env is None:
            os.environ.pop("LSA_TARGETS", None)
        else:
            os.environ["LSA_TARGETS"] = saved_env
        _sh.rmtree(tmp_cast, ignore_errors=True)

    # -- randomness: seeded, never the same figure twice, never the same place --
    def _rounds(seed, n=30):
        m_ = ImageTarget()
        m_._rng.seed(seed)
        seq = []
        for _ in range(n):
            m_.start(context(now=100.0))
            w_, h_ = m_._fig.sprite.drawn_size(m_._height)
            fld = m_.playfield(context(now=100.0))
            seq.append((m_._fig.name, m_._centre, m_._height,
                        math.hypot(fld.width() - w_, fld.height() - h_)))
        return seq

    seq_a, seq_b, seq_c = _rounds(21), _rounds(21), _rounds(22)
    check("image: the same seed gives the same sequence of figures and places",
          seq_a == seq_b)
    check("image: a different seed gives a different one",
          [s[0] for s in seq_a] != [s[0] for s in seq_c])
    check("image: over 30 new rounds no figure is shown twice in a row",
          all(a[0] != b[0] for a, b in zip(seq_a, seq_a[1:])))
    check("image: at least 8 of the 12 figures appear in 30 rounds",
          len({s[0] for s in seq_a}) >= 8, f"{len({s[0] for s in seq_a})}")
    far_ok = sum(1 for a, b in zip(seq_a, seq_a[1:])
                 if math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1])
                 >= 0.35 * min(a[3], b[3]) - 1.0)
    check("image: each new round's figure is >= 35% of the diagonal from the last "
          "(or the farthest corner)", far_ok >= 27, f"{far_ok} of 29")

    print()
    print("19. mode 9, the advancing target")

    def _rig(seed=1, figure="thug01", count=1):
        m_ = AdvancingTarget()
        m_.COUNT = count
        m_._rng.seed(seed)
        m_.start(context(now=0.0), figure=figure)
        t_ = _drive_to_live(m_, 0.0)
        return m_, t_

    # repaint rates, per phase
    m9, t_live = _rig()
    a9 = m9._actors[0]
    n_app = sum(1 for i in range(180) if m9.advance(context(now=a9.t0 + 0.05 + i / 60.0)))
    print(f"         repaints requested over 180 ticks (3.0 s) of APPROACH: {n_app}")
    check("approach: 20 Hz while moving (54-66 repaint requests in 180 ticks)",
          54 <= n_app <= 66, f"{n_app}")
    m9b, _ = _rig(seed=2)
    a9b = m9b._actors[0]
    a9b.t_h = 6.0                                  # a long hold so 180 ticks fit inside it
    t_h0 = a9b.t0 + a9b.t_a + 0.2
    m9b.advance(context(now=t_h0 - 1 / 60.0))      # the tick that arrives at the hold
    n_hold = sum(1 for i in range(180) if m9b.advance(context(now=t_h0 + i / 60.0)))
    print(f"         repaints requested over 180 ticks (3.0 s) of HOLDING: {n_hold}")
    check("holding still: at most 5 repaint requests in 180 ticks", n_hold <= 5,
          f"{n_hold}")
    check("holding still: the animated flag says False, as advance() does",
          m9b.animated is False)
    m9c = AdvancingTarget()
    m9c._rng.seed(3)
    m9c.start(context(now=0.0), figure="thug01")
    m9c._actors[0].live_at = 1e9                   # a long blank wait
    n_gap = sum(1 for i in range(180) if m9c.advance(context(now=i / 60.0)))
    check("the blank wait between figures asks for nothing (0 in 180 ticks)", n_gap == 0,
          f"{n_gap}")
    m9d, _ = _rig(seed=2)
    saved_rate = AdvancingTarget.RATE_HZ
    AdvancingTarget.RATE_HZ = 60.0
    n_fast = sum(1 for i in range(180)
                 if m9d.advance(context(now=m9d._actors[0].t0 + 0.05 + i / 60.0)))
    AdvancingTarget.RATE_HZ = saved_rate
    check("negative control: asking for 60 Hz would read ~180 (the count can see it)",
          n_fast > 150, f"{n_fast}")

    # the budget at the NEAREST scale, for every one of the twelve
    worst_near, worst_near_red = (0.0, ""), 0
    for nm in FIG_NAMES:
        mm_, _t = _rig(seed=4, figure=nm)
        mm_.started_at = -1e9                 # the figure alone; the hint is measured in 18
        aa = mm_._actors[0]
        worst_lit_fig, worst_r = 0.0, 0
        # Lit area only grows with size, so the NEAREST hold is where the budget can
        # fail, and the first frame is where something APPEARS: those two for every
        # figure. The whole walk (five phases) for the two fixtures: thug01 (the
        # reference nest) and predator01 (red excess 26 as stored).
        walk = [("first frame", aa.t0), ("near hold", aa.t0 + aa.t_a + 0.1)]
        if nm in ("thug01", "predator01"):
            walk = [("first frame", aa.t0), ("mid approach", aa.t0 + aa.t_a / 2),
                    ("near hold", aa.t0 + aa.t_a + 0.1),
                    ("mid retreat", aa.t0 + aa.t_a + aa.t_h + aa.t_r / 2),
                    ("far hold", aa.t0 + aa.duration - 0.1)]
        for label, tt in walk:
            r_, l_, _i = _full(mm_.paint, context(now=tt))
            worst_lit_fig = max(worst_lit_fig, l_)
            worst_r = max(worst_r, r_)
        worst_near_red = max(worst_near_red, worst_r)
        if worst_lit_fig > worst_near[0]:
            worst_near = (worst_lit_fig, nm)
        check(f"advancing/{nm}: worst lit area over the whole walk {worst_lit_fig * 100:.3f}% "
              f"(<= {BUDGET * 100:.1f}%, checked at the nearest scale and as it appears), "
              f"red excess {worst_r}",
              worst_lit_fig <= BUDGET + 1e-9 and worst_r == 0)
    print(f"         worst single figure over a whole walk: {worst_near[1]} at "
          f"{worst_near[0] * 100:.3f}% lit; worst red excess {worst_near_red}")

    # COUNT = 2: figures ADD, so the pair is checked on the sum
    two = AdvancingTarget()
    two.COUNT = 2
    two._rng.seed(9)
    two.start(context(now=0.0), figure=HEAVIEST)
    t2 = 0.0
    while any(a.phase != "live" for a in two._actors) and t2 < 20.0:
        two.advance(context(now=t2))
        t2 += 1 / 60.0
    a0, a1 = two._actors
    a1.fig = _Fig("predator01", SW, SH, _CB / 2)     # the second-largest lit figure
    a1.fig.sprite.set_range(a1.h_near)
    now2 = t2
    for a in (a0, a1):
        a.t0 = now2 - (a.t_a + 0.2)                  # both in the near hold together
        a.x_lo, a.x_hi = 0.0, float(SW)
        a.drift = 0.0
    a0.hold_xy = (SW * 0.27, a0.hold_xy[1])
    a1.hold_xy = (SW * 0.73, a1.hold_xy[1])
    two.started_at = -1e9                  # the figures alone; the hint is measured in 18
    r2, l2, _i2 = _full(two.paint, context(now=now2))
    print(f"         COUNT=2, {HEAVIEST} + predator01 both at their near hold: "
          f"{l2 * 100:.3f}% lit (each capped at {_CB / 2 * 100:.2f}%), red excess {r2}")
    check("COUNT = 2: the two worst figures together at their nearest stay inside "
          "the combined 1.5%", l2 <= _CB + 1e-9 and r2 == 0, f"{l2 * 100:.3f}%")
    check("COUNT = 2 is OFF by default", AdvancingTarget.COUNT == 1)

    # the shot is tested where the figure WAS, height as well as position
    mr, _tr = _rig(seed=6)
    ar = mr._actors[0]
    ar.drift = 1800.0                                # px/s sideways, so 80 ms is ~144 px
    ar.x_lo, ar.x_hi = 0.0, float(SW)
    ar.hold_xy = (2800.0, ar.hold_xy[1])             # room to move left of it
    t_shot = ar.t0 + ar.t_a - 0.5
    c_then, h_then = ar.state(t_shot)
    now_r = t_shot + 0.08
    c_now, h_now = ar.state(now_r)
    fig_r = ar.fig
    found = None
    inn0 = fig_r.innermost(c_then, h_then)
    for dx in range(0, 160, 4):
        x_try = inn0[1] - dx
        was = fig_r.score(x_try, inn0[2], c_then, h_then)[0]
        now_ = fig_r.score(x_try, inn0[2], c_now, h_now)[0]
        if was == 10 and now_ < 10:
            found = x_try
    check("setup: a point inside the 10 ring 80 ms ago and no longer inside it now exists "
          "(the figure grew and moved)", found is not None and h_now > h_then,
          f"height {h_then:.0f} -> {h_now:.0f}px, moved {c_now[0] - c_then[0]:.0f}px")
    out = mr.shot(modes.ShotEvent(found, inn0[2], t_shot, 1), context(now=now_r))
    check("a shot where the figure WAS 80 ms ago, at the ring it then covered, scores 10 "
          "although the figure has since grown and moved past it",
          out.hit and out.points == 10, str(out))
    out0 = mr.shot(modes.ShotEvent(found, inn0[2], now_r, 2), context(now=now_r))
    check("the same point judged against the figure NOW (age 0) is NOT a 10",
          out0.points < 10, str(out0))
    near_r = mr.nearest(modes.ShotEvent(found, inn0[2], t_shot, 3), context(now=now_r))
    check("nearest() measures at the same rewound state as shot()",
          near_r is not None and near_r[1] <= near_r[2] + 5.0, str(near_r))

    # bands and the time-to-first-hit measurement
    mb, _tb = _rig(seed=7)
    ab = mb._actors[0]
    t_near = ab.t0 + ab.t_a + 0.5
    cb_, hb_ = ab.state(t_near)
    ib_ = ab.fig.innermost(cb_, hb_)
    mb.shot(modes.ShotEvent(ib_[1], ib_[2], t_near, 1), context(now=t_near))
    check("a hit in the near hold lands in the NEAR band and records time since the hold began",
          mb.bands["near"][:2] == [1, 1] and len(mb.hold_times) == 1
          and abs(mb.hold_times[0] - 0.5) < 1e-6, f"{mb.bands} {mb.hold_times}")
    mb.shot(modes.ShotEvent(ib_[1], ib_[2], t_near + 0.1, 2), context(now=t_near + 0.1))
    check("only the FIRST hit after a hold counts for that measurement",
          len(mb.hold_times) == 1)
    t_far = ab.t0 + ab.duration - 0.2
    cf_, hf_ = ab.state(t_far)
    if_ = ab.fig.innermost(cf_, hf_)
    mb.shot(modes.ShotEvent(if_[1], if_[2], t_far, 3), context(now=t_far))
    check("a hit at the far end lands in the FAR band", mb.bands["far"][1] == 1,
          str(mb.bands))
    # sizes are fractions of the screen: half the height, half the figure
    def _near_h(w, h):
        m_ = AdvancingTarget()
        m_._rng.seed(1)
        c_ = RenderContext(now=0.0, dt=0.0, rect=QRect(0, 0, w, h), stats=SessionStats())
        m_.start(c_, figure="thug01")
        for i in range(2000):
            m_.advance(RenderContext(now=i / 60.0, dt=1 / 60.0, rect=QRect(0, 0, w, h),
                                     stats=SessionStats()))
            if m_._actors[0].phase == "live":
                break
        return m_._actors[0].h_near
    n1080, n2160 = _near_h(1920, 1080), _near_h(3840, 2160)
    check("the nearest size scales with the screen, not pixels (2160p / 1080p = 2.0)",
          abs(n2160 / n1080 - 2.0) < 0.1, f"{n1080:.0f}px at 1080p, {n2160:.0f}px at 2160p")

    # randomness over 30 seeded cycles
    def _cycles(seed, n=30):
        m_ = AdvancingTarget()
        m_._rng.seed(seed)
        rec = []
        orig = m_._begin

        def wrapped(a, ctx_):
            orig(a, ctx_)
            w_, h_ = a.fig.sprite.drawn_size(a.h_near)
            fld = m_.playfield(ctx_)
            rec.append(dict(fig=a.fig.name, side=(a.drift > 0) - (a.drift < 0),
                            h_start=a.h_start, h_near=a.h_near, hold=a.hold_xy,
                            diag=math.hypot(fld.width() - w_, fld.height() - h_),
                            durs=(a.t_a, a.t_h, a.t_r, a.t_f), t0=ctx_.now))
        m_._begin = wrapped
        m_.start(context(now=0.0))
        t_ = 0.0
        while len(rec) < n and t_ < 600.0:
            m_.advance(context(now=t_))
            t_ += 1 / 60.0
        return rec

    c1, c2, c3 = _cycles(31), _cycles(31), _cycles(32)
    check("advancing: the same seed gives the same cycles", c1 == c2)
    check("advancing: a different seed gives different ones",
          [c["fig"] for c in c1] != [c["fig"] for c in c3])
    check("over 30 seeded cycles no two consecutive figures are equal",
          len(c1) == 30 and all(a["fig"] != b["fig"] for a, b in zip(c1, c1[1:])))
    check("the drift side is not constant (left, straight and right all occur)",
          {c["side"] for c in c1} == {-1, 0, 1}, str(sorted({c["side"] for c in c1})))
    check("the starting distance is not constant and stays between far and 0.8 of near",
          len({round(c["h_start"]) for c in c1}) >= 20
          and all(0.5 * c["h_near"] - 1 <= c["h_start"] <= 0.8 * c["h_near"] + 1 for c in c1))
    check("the durations vary (approach, hold, retreat, far hold)",
          all(len({round(c["durs"][i], 3) for c in c1}) >= 20 for i in range(4)))
    gaps = [b["t0"] - (a["t0"] + sum(a["durs"])) for a, b in zip(c1, c1[1:])]
    check("the blank wait before each figure varies within 0.5-1.5 s",
          all(0.45 <= g <= 1.55 for g in gaps) and len({round(g, 2) for g in gaps}) >= 15,
          f"{min(gaps):.2f}..{max(gaps):.2f}")
    far_ok = sum(1 for a, b in zip(c1, c1[1:])
                 if math.hypot(a["hold"][0] - b["hold"][0], a["hold"][1] - b["hold"][1])
                 >= 0.35 * min(a["diag"], b["diag"]) - 1.0)
    check("consecutive hold positions are >= 35% of the diagonal apart (or the farthest corner)",
          far_ok >= 27, f"{far_ok} of 29")

    print()
    print("20. mode 7, the pop-up figure")

    def _popups(seed, n=30):
        m_ = PopUpFigure()
        m_._rng.seed(seed)
        rec = []
        orig = m_._appear

        def wrapped(ctx_):
            prev_left = m_.left_at
            orig(ctx_)
            w_, h_ = m_._fig.sprite.drawn_size(m_._height)
            fld = m_.playfield(ctx_)
            rec.append(dict(fig=m_._fig.name, centre=m_._centre, t=ctx_.now,
                            wait=ctx_.now - prev_left, frac=m_._next_frac,
                            h=m_._height / SH, cap=m_._fig.cap,
                            diag=math.hypot(fld.width() - w_, fld.height() - h_)))
        m_._appear = wrapped
        m_.start(context(now=0.0))
        t_ = 0.0
        while len(rec) < n and t_ < 800.0:
            m_.advance(context(now=t_))
            t_ += 1 / 60.0
        return rec

    p1, p2, p3 = _popups(41), _popups(41), _popups(42)
    check("popup: the same seed gives the same figures, places and times", p1 == p2)
    check("popup: a different seed gives different ones",
          [p["fig"] for p in p1] != [p["fig"] for p in p3])
    check("over 30 seeded appearances no two consecutive figures are equal",
          len(p1) == 30 and all(a["fig"] != b["fig"] for a, b in zip(p1, p1[1:])))
    waits = [p["wait"] for p in p1[1:]]
    check("the wait before each figure is random within 1.2-3.5 s (not a fixed delay)",
          all(1.15 <= w <= 3.55 for w in waits) and len({round(w, 2) for w in waits}) >= 20,
          f"{min(waits):.2f}..{max(waits):.2f}")
    far_ok = sum(1 for a, b in zip(p1, p1[1:])
                 if math.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1])
                 >= 0.35 * min(a["diag"], b["diag"]) - 1.0)
    check("consecutive centres are >= 35% of the diagonal apart (or the farthest corner)",
          far_ok >= 27, f"{far_ok} of 29")
    check("the size is random between 0.6 and 1.0 of each figure's own cap",
          all(0.6 - 1e-6 <= p["frac"] <= 1.0 + 1e-6 for p in p1)
          and len({round(p["frac"], 2) for p in p1}) >= 15)

    mp = PopUpFigure()
    mp._rng.seed(12)
    mp.start(context(now=0.0), figure="thug01")
    out = mp.shot(modes.ShotEvent(300.0, 300.0, 0.5, 1), context(now=0.5))
    check("a shot during the wait is a FALSE START, reported and not scored",
          not out.hit and out.target_id == "false-start" and mp.false_starts == 1
          and mp.hits == 0)
    t_up = _drive_to_live(mp, 0.5)
    r1, l1, _i1 = _full(mp.paint, context(now=mp.shown_at))
    check("the appearing figure is red excess 0 on its FIRST frame (anything that appears)",
          r1 == 0 and 0.001 < l1 <= BUDGET, f"red {r1}, lit {l1 * 100:.3f}%")
    ptp = _zone_point(mp)
    out = mp.shot(modes.ShotEvent(ptp[0], ptp[1], mp.shown_at - 0.05, 2),
                  context(now=mp.shown_at + 0.05))
    check("a shot fired BEFORE it appeared but confirmed after is still a false start "
          "(judged by its own time)", out.target_id == "false-start" and mp.false_starts == 2)
    out = mp.shot(modes.ShotEvent(ptp[0], ptp[1], mp.shown_at + 0.30, 3),
                  context(now=mp.shown_at + 0.35))
    check("a hit records the time from appearance to the shot (0.30 s) and its ring",
          out.hit and out.points == 10 and abs(mp.times[0] - 0.30) < 1e-6
          and mp.first_points == [10], f"{mp.times}")
    mp.shot(modes.ShotEvent(ptp[0], ptp[1], mp.shown_at + 0.5, 4),
            context(now=mp.shown_at + 0.5))
    check("only the first hit is the measurement", len(mp.times) == 1 and mp.hits == 2)
    left = mp.left_at
    mp.advance(context(now=left - 0.01))
    still_up = mp.state == "up"
    mp.advance(context(now=left + 0.01))
    check("the figure stays for EXPOSURE_S (2.5 s) and leaves at the deadline even though "
          "it was hit", still_up and mp.state == "wait"
          and abs((left - mp.shown_at) - PopUpFigure.EXPOSURE_S) < 1e-9)
    mq = PopUpFigure()
    mq._rng.seed(13)
    mq.start(context(now=0.0), figure="thug01")
    _drive_to_live(mq, 0.0)
    mq.advance(context(now=mq.left_at + 0.01))
    check("a scored figure that left without a hit is counted as unengaged",
          mq.unengaged == 1)
    mw = PopUpFigure()
    mw._rng.seed(14)
    mw.start(context(now=0.0), figure="thug01")
    mw.next_at = 1e9
    n_wait = sum(1 for i in range(180) if mw.advance(context(now=i / 60.0)))
    check("waiting: 0 repaints in 180 ticks", n_wait == 0, f"{n_wait}")
    mw2 = PopUpFigure()
    mw2._rng.seed(15)
    mw2.start(context(now=0.0), figure="thug01")
    _drive_to_live(mw2, 0.0)
    t_a2 = mw2.shown_at + 0.02
    n_up = sum(1 for i in range(120)
               if mw2.advance(context(now=t_a2 + i / 60.0)))   # 2 s of the 2.5 s up
    check("figure up and still: at most 5 repaints in 120 ticks (the transitions only)",
          n_up <= 5 and mw2.animated is False, f"{n_up}")
    for nm in FIG_NAMES:
        mx = PopUpFigure()
        mx._rng.seed(17)
        mx.start(context(now=0.0), figure=nm)
        mx._next_frac = 1.0
        _drive_to_live(mx, 0.0)
        # the figure alone (the intro hint, drawn for the first 8 s, is measured
        # separately below: it is text, and it sits on top of the figure's budget)
        mx.started_at = -100.0
        rr_, ll_, _ii = _full(mx.paint, context(now=mx.shown_at))
        mx.started_at = mx.shown_at
        rh_, lh_, _ih = _full(mx.paint, context(now=mx.shown_at))
        check(f"popup/{nm}: at its largest the figure lights {ll_ * 100:.3f}% "
              f"(<= {BUDGET * 100:.1f}%), red excess {rr_}; with the intro hint "
              f"{lh_ * 100:.3f}% (<= 2%), red excess {rh_}",
              ll_ <= BUDGET + 1e-9 and rr_ == 0 and lh_ <= 0.02 and rh_ == 0)

    print()
    print("21. one frame of a moving figure, measured")
    import time as _time21
    ma, _ta = _rig(seed=10)
    aa = ma._actors[0]
    img21 = _QI3(SW, SH, _QI3.Format_RGB32)
    img21.fill(0xFF0A0A0A)
    pn = _QP3(img21)
    costs, builds0 = [], aa.fig.sprite.builds
    for i in range(180):
        tt = aa.t0 + 0.05 + i / 60.0
        t0_ = _time21.perf_counter()
        ma.paint(pn, context(now=tt))
        costs.append((_time21.perf_counter() - t0_) * 1000)
    pn.end()
    costs_sorted = sorted(costs)
    print(f"         paint of one frame over 3 s of approach at 3840x2160: median "
          f"{costs_sorted[90]:.2f} ms, p95 {costs_sorted[171]:.2f} ms, worst "
          f"{costs_sorted[-1]:.2f} ms; {aa.fig.sprite.builds - builds0} level builds, "
          f"{aa.fig.sprite.pixmap_builds} pixmap builds; pyramid "
          f"{aa.fig.sprite.pyramid_bytes / 1e6:.1f} MB")
    check("a frame of the approach is a cached blit (median <= 2.5 ms) and levels are "
          "built far less often than frames are painted (2% steps)",
          costs_sorted[90] <= 2.5 and aa.fig.sprite.builds - builds0 < 60,
          f"{costs_sorted[90]:.2f} ms, {aa.fig.sprite.builds - builds0} builds")

    print()
    print("22. a hit is VISIBLE, in every figure mode, always")

    def _pair(cls_, seed):
        """Two identical modes on the same seed; the first is never hit, the second
        is hit at the centre of the innermost ring. Returns (idle, hit, t_hit)."""
        out_ = []
        for _k in range(2):
            m_ = cls_()
            m_._rng.seed(seed)
            m_.start(context(now=0.0), figure="thug01")
            if cls_ is ImageTarget:
                t_h = 50.0
            else:
                _drive_to_live(m_, 0.0)
                if cls_ is PopUpFigure:
                    t_h = m_.shown_at + 0.3
                else:
                    a_ = m_._actors[0]
                    t_h = a_.t0 + a_.t_a + 0.1
            out_.append((m_, t_h))
        return out_[0][0], out_[1][0], out_[0][1]

    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        idle, hit, t_h = _pair(cls_, 5)
        if cls_ is AdvancingTarget:
            a_ = hit._actors[0]
            c_, h_ = a_.state(t_h)
            inn_ = a_.fig.innermost(c_, h_)
            pt_ = (inn_[1], inn_[2])
        else:
            pt_ = _zone_point(hit)
        hit.shot(modes.ShotEvent(pt_[0], pt_[1], t_h, 1), context(now=t_h))
        r0, l0, im0 = _full(idle.paint, context(now=t_h + 0.05))
        r1, l1, im1 = _full(hit.paint, context(now=t_h + 0.05))
        changed = int((im0 != im1).any(axis=2).sum())
        print(f"         {cls_.name:9s} hit at t+0.05 s: {changed} pixels differ from the same "
              f"frame without it; lit {l0 * 100:.3f}% -> {l1 * 100:.3f}%; red excess {r1}")
        check(f"{cls_.name}: a hit changes what is on screen (a ring and its number appear: "
              f">= 600 pixels)", changed >= 600, f"{changed}")
        check(f"{cls_.name}: the feedback is grey only (red excess 0) and the figure plus the "
              f"feedback stays inside the 2% lit-area budget",
              r1 == 0 and l1 <= 0.02, f"red {r1}, lit {l1 * 100:.3f}%")
        hit.advance(context(now=t_h + hit.FLASH_S + 0.2))
        left_ = (hit._actors[0].marks.flashes if cls_ is AdvancingTarget
                 else hit._marks.flashes)
        check(f"{cls_.name}: the flash ends by itself", not left_, str(left_))
        if cls_ is PopUpFigure:
            # the hit stays marked for as long as the figure stands
            t_late = t_h + PopUpFigure.FLASH_S + 0.2
            assert t_late < hit.left_at
            _r, _l, im_late = _full(hit.paint, context(now=t_late))
            _r, _l, im_idle = _full(idle.paint, context(now=t_late))
            kept = int((im_late != im_idle).any(axis=2).sum())
            check("popup: after the flash has faded the hit is STILL marked on the figure "
                  "(a ring is left where it landed until the figure leaves)", kept >= 150,
                  f"{kept} pixels")
            hit.advance(context(now=hit.left_at + 0.01))
            check("popup: the mark goes when the figure goes", len(hit._marks) == 0
                  and not hit._marks.flashes)
    # the time and ring number appear beside the hit in the pop-up
    mt = PopUpFigure()
    mt._rng.seed(5)
    mt.start(context(now=0.0), figure="thug01")
    _drive_to_live(mt, 0.0)
    pz = _zone_point(mt)
    mt.shot(modes.ShotEvent(pz[0], pz[1], mt.shown_at + 0.3, 1), context(now=mt.shown_at + 0.3))
    check("popup: the flash carries the reaction time and the ring (e.g. '300 ms   10')",
          mt._marks.flashes and mt._marks.flashes[-1][3] == "300 ms   10",
          str(mt._marks.flashes))
    mf = PopUpFigure()
    mf._rng.seed(6)
    mf.start(context(now=0.0), figure="thug01")
    mf.shot(modes.ShotEvent(400.0, 400.0, 0.2, 1), context(now=0.2))
    check("popup: a false start is shown too (TOO EARLY, in the dimmer grey)",
          mf.flashes[-1][3] == "TOO EARLY" and mf.flashes[-1][4] is False)

    print()
    print("23. a shot ON the figure but in no ring is visible, and distinct from both a "
          "scoring shot and nothing at all")

    def _geom(m_, t_):
        if m_.name == "advancing":
            a_ = m_._actors[0]
            c_, h_ = a_.state(t_)
            return a_.fig, c_, h_
        return m_._fig, m_._centre, m_._height

    def _screen(fig_, c_, h_, ix, iy):
        l_, t0_, w_, hh_ = fig_.box(c_, h_)
        return (l_ + ix / 1791.0 * w_, t0_ + iy / 2399.0 * hh_)

    def _lum_max(img_a, img_b):
        d_ = (img_a != img_b).any(axis=2)
        if not d_.any():
            return 0, 0
        lum_ = cv2.cvtColor(np.ascontiguousarray(img_b[..., :3]), cv2.COLOR_BGR2GRAY)
        return int(lum_[d_].max()), int(d_.sum())

    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        idle, hit, t_h = _pair(cls_, 5)
        fig_, c_, h_ = _geom(hit, t_h)
        # points the reviewer named: top of the head, a thigh, and well off the figure
        def _find(xs_, ys_):
            """First image pixel in the region that is ON the figure and in no ring."""
            for iy_ in ys_:
                for ix_ in xs_:
                    x_, y_ = _screen(fig_, c_, h_, ix_, iy_)
                    if fig_.on_figure(x_, y_, c_, h_) and fig_.score(x_, y_, c_, h_)[0] == 0:
                        return (ix_, iy_)
            raise AssertionError("no on-figure no-ring point in the region")
        probes = {"top of the head": _find(range(700, 900, 20), range(150, 330, 20)),
                  "thigh": _find(range(200, 700, 20), range(2000, 2300, 20))}
        # the hit, for comparison
        hit_pt = fig_.innermost(c_, h_)
        _r, _l, im_idle = _full(idle.paint, context(now=t_h + 0.05))
        mh = _pair(cls_, 5)[1]
        mh.shot(modes.ShotEvent(hit_pt[1], hit_pt[2], t_h, 1), context(now=t_h))
        _r, _l, im_hit = _full(mh.paint, context(now=t_h + 0.05))
        lum_hit, px_hit = _lum_max(im_idle, im_hit)
        for label_, (ix_, iy_) in probes.items():
            mm_ = _pair(cls_, 5)[1]
            x_, y_ = _screen(fig_, c_, h_, ix_, iy_)
            out_ = mm_.shot(modes.ShotEvent(x_, y_, t_h, 2), context(now=t_h))
            r_, l_, im_m = _full(mm_.paint, context(now=t_h + 0.05))
            lum_m, px_m = _lum_max(im_idle, im_m)
            print(f"         {cls_.name:9s} on-figure miss ({label_}): {px_m} pixels, brightest "
                  f"{lum_m} (a scored hit: {px_hit} px, brightest {lum_hit}); lit "
                  f"{l_ * 100:.3f}%, red excess {r_}")
            check(f"{cls_.name}: a shot on the {label_} (no ring) is reported as on the "
                  f"figure, not a hit, and says 'no ring'",
                  not out_.hit and out_.target_id.endswith(":no-ring"), str(out_))
            check(f"{cls_.name}: ...and DRAWS something ({px_m} pixels), grey only "
                  f"(red excess {r_}), inside the 2% lit-area budget ({l_ * 100:.3f}%)",
                  px_m >= 400 and r_ == 0 and l_ <= 0.02)
            # A MARGIN, NOT `<=`. This check read `0 < lum_m <= lum_hit`, so a
            # no-ring mark drawn at EXACTLY the hit's brightness passed -- which
            # is the whole fault. Mutation 20 of mutate_figures (`c =
            # QColor(GREY_HIT)`) survived every run until 2026-10-19 because of
            # that one character, and its other half compared the two palette
            # CONSTANTS to each other, which no change to the drawing code can
            # move. Measured here: the mark peaks 65-69 against a hit's 78, a
            # worst real ratio of 0.885 -- so 0.93 clears every honest value and
            # still catches equality. RULES C1/C5: the bar is measured, not
            # chosen to be safe.
            check(f"{cls_.name}: ...and it is visibly DIMMER than a scored hit "
                  f"(brightest {lum_m} against {lum_hit} = "
                  f"{lum_m / max(lum_hit, 1):.3f}, bar 0.93), so the two cannot "
                  f"be confused",
                  0 < lum_m <= 0.93 * lum_hit,
                  f"{lum_m} vs {lum_hit} = {lum_m / max(lum_hit, 1):.3f}: a mark "
                  f"as bright as a hit is indistinguishable from one, and on his "
                  f"screen it is lit area that eats the next shot (B10)")
            # And the palette relationship, as its OWN check. Welded to the one
            # above with `and`, it could not be read separately, and it is the
            # half that catches someone brightening GREY_MISS in the palette
            # rather than in the drawing.
            check(f"{cls_.name}: ...and the miss grey is at most 0.6 of the hit "
                  f"grey in the palette itself "
                  f"({_LYC.GREY_MISS.red()} vs {_LYC.GREY_HIT.red()})",
                  _LYC.GREY_MISS.red() <= 0.6 * _LYC.GREY_HIT.red(),
                  "the two greys have drifted together in the palette")
        # nothing at all, far off the figure: the mode draws nothing
        mo = _pair(cls_, 5)[1]
        x_, y_ = _screen(fig_, c_, h_, 5, 5)
        x_ = min(max(x_, 2.0), SW - 2.0)
        far = (fig_.box(c_, h_)[0] - 400.0, 300.0)
        out_ = mo.shot(modes.ShotEvent(far[0], far[1], t_h, 3), context(now=t_h))
        _r, _l, im_o = _full(mo.paint, context(now=t_h + 0.05))
        check(f"{cls_.name}: a shot far OFF the figure is an ordinary miss and the mode draws "
              f"nothing (the hole layer shows it) -- distinct from the dim ring",
              not out_.hit and out_.target_id is None and _lum_max(im_idle, im_o)[1] == 0,
              str(out_))
    # worst case: the most marks the modes allow at once, on the biggest figure
    for cls_ in (ImageTarget, PopUpFigure):
        mw_ = cls_()
        mw_._rng.seed(9)
        mw_.start(context(now=0.0), figure=HEAVIEST)
        if cls_ is PopUpFigure:
            mw_._next_frac = 1.0
            _drive_to_live(mw_, 0.0)
            t0_ = mw_.shown_at + 0.2
        else:
            t0_ = 50.0
        mw_.started_at = -1e9
        fg_, cg_, hg_ = _geom(mw_, t0_)
        zc_ = fg_.innermost(cg_, hg_)
        # Hits: the MOST scoring hits a figure can carry is now four 7s (28 of the 30
        # at which it drops); the old twelve centre hits would drop it after three
        # and leave this measuring an empty figure.
        for k_ in range(4):
            xh_, yh_ = _screen(fg_, cg_, hg_, 410 + 6 * k_, 1200)
            mw_.shot(modes.ShotEvent(xh_, yh_, t0_ + 0.01 * k_, 10 + k_),
                     context(now=t0_ + 0.01 * k_))
        for k_ in range(3):
            xg_, yg_ = _screen(fg_, cg_, hg_, 800 + 40 * k_, 200)
            mw_.shot(modes.ShotEvent(xg_, yg_, t0_ + 0.2 + 0.01 * k_, 40 + k_),
                     context(now=t0_ + 0.2 + 0.01 * k_))
        rw_, lw_, _iw = _full(mw_.paint, context(now=t0_ + 0.25))
        check(f"{cls_.name}: the worst-case setup really carries its 4 hits and 3 no-ring marks "
              f"(not an empty figure)", len(mw_._marks) == 7 and mw_._marks.total == 28,
              f"{len(mw_._marks)} marks, total {mw_._marks.total}")
        print(f"         {cls_.name:9s} worst case (4 hits + 3 no-ring misses, {HEAVIEST} at its "
              f"largest): lit {lw_ * 100:.3f}%, red excess {rw_}, "
              f"{len(mw_._marks.flashes)} flashes up, {len(mw_._marks)} holes carried")
        check(f"{cls_.name}: with the most marks it allows at once, grey only and inside the 2% "
              f"lit-area budget", rw_ == 0 and lw_ <= 0.02, f"{lw_ * 100:.3f}%")
    # mode 9: a shot into the blank between walks
    mg = AdvancingTarget()
    mg._rng.seed(3)
    mg.start(context(now=0.0), figure="thug01")
    out_g = mg.shot(modes.ShotEvent(1500.0, 900.0, 0.2, 1), context(now=0.25))
    _r, _l, im_g0 = _full(AdvancingTarget().paint, context(now=0.3))
    _r, _l, im_g1 = _full(mg.paint, context(now=0.3))
    check("advancing: a shot into the blank between walks is reported as 'no target', is "
          "NOT counted against hits/shots, and is shown (like the pop-up's false start)",
          out_g.target_id == "no-target" and mg.shots == 0 and mg.gap_shots == 1
          and mg.flashes[-1][3] == "NO TARGET" and _lum_max(im_g0, im_g1)[1] >= 300,
          f"{out_g} shots={mg.shots} gap={mg.gap_shots}")
    check("advancing: the status line says how many were fired with nothing up",
          "1 fired with nothing up" in mg.status(), mg.status())
    # the limitation is said where he looks
    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        check(f"{cls_.name}: the head-dome limitation is in the mode's note (logged when "
              f"chosen) and summary",
              "inside the head dome is 8" in cls_.note and "no ring" in cls_.summary,
              cls_.summary)
        mh_ = cls_()
        mh_.start(context(now=0.0), figure="thug01")
        if cls_ is PopUpFigure:
            _drive_to_live(mh_, 0.0)
        img_h = _QI3(SW, SH, _QI3.Format_RGB32)
        img_h.fill(0xFF0A0A0A)
        ph = _QP3(img_h)
        mh_.paint(ph, context(now=0.5 if cls_ is not PopUpFigure else mh_.shown_at + 0.1))
        ph.end()
        a_h = np.frombuffer(img_h.constBits(), np.uint8).reshape(SH, img_h.bytesPerLine())
        a_h = a_h[:, :SW * 4].reshape(SH, SW, 4)
        # the hint sits bottom-left of the playfield: it lights pixels there while up
        band_ = a_h[int(SH * 0.80):int(SH * 0.93), 0:int(SW * 0.5), 1]
        mh_.started_at = -1e9
        img_h2 = _QI3(SW, SH, _QI3.Format_RGB32)
        img_h2.fill(0xFF0A0A0A)
        ph = _QP3(img_h2)
        mh_.paint(ph, context(now=0.5 if cls_ is not PopUpFigure else mh_.shown_at + 0.1))
        ph.end()
        a_h2 = np.frombuffer(img_h2.constBits(), np.uint8).reshape(SH, img_h2.bytesPerLine())
        a_h2 = a_h2[:, :SW * 4].reshape(SH, SW, 4)
        band2_ = a_h2[int(SH * 0.80):int(SH * 0.93), 0:int(SW * 0.5), 1]
        check(f"{cls_.name}: the standing hint (head dome / 'no ring') is drawn on screen "
              f"for the first seconds and gone afterwards",
              int((band_ > _LYC.CEILING // 2).sum()) > int((band2_ > _LYC.CEILING // 2).sum()) + 200,
              f"{int((band_ > _LYC.CEILING // 2).sum())} vs {int((band2_ > _LYC.CEILING // 2).sum())} bright pixels")

    print()
    print("24. a shot ON a figure rides it: the hole belongs to the figure, not the screen")
    from lsa.modes.base import ShotOutcome as _SO
    from lsa.modes.figure import Marks as _Marks, MAX_CARRIED as _MAXC, HOLE_R_FRAC as _HRF

    def _adv(seed):
        m_ = AdvancingTarget()
        m_._rng.seed(seed)
        m_.start(context(now=0.0), figure="thug01")
        _drive_to_live(m_, 0.0)
        return m_

    def _diff(img_a, img_b):
        return (img_a != img_b).any(axis=2)

    def _screen_frac(fig_, c_, h_, u, v):
        l_, t0_, w_, hh_ = fig_.box(c_, h_)
        return (l_ + u * w_, t0_ + v * hh_)

    def _within(mask, pos, rad):
        """How many differing pixels lie within `rad` px of `pos`."""
        x0, x1 = max(0, int(pos[0] - rad)), min(SW, int(pos[0] + rad) + 1)
        y0, y1 = max(0, int(pos[1] - rad)), min(SH, int(pos[1] + rad) + 1)
        sub = mask[y0:y1, x0:x1]
        yy, xx = np.nonzero(sub)
        return int((((xx + x0 - pos[0]) ** 2 + (yy + y0 - pos[1]) ** 2) <= rad * rad).sum())

    # -- 24a: a hit on a WALKING figure; its mark's screen position, frame by frame --
    mA, twinA = _adv(5), _adv(5)
    aA = mA._actors[0]
    t_shot = aA.t0 + 0.3 * aA.t_a
    cA, hA = aA.state(t_shot)
    p7 = _screen(aA.fig, cA, hA, 410, 1200)           # the printed 7, left of the chest
    ph = _screen(aA.fig, cA, hA, 800, 255)            # on the head, above the dome: no ring
    # fired 80 ms ago and reported now, so the rewind is exercised too
    o7 = mA.shot(modes.ShotEvent(p7[0], p7[1], t_shot, 1), context(now=t_shot + 0.08))
    oh = mA.shot(modes.ShotEvent(ph[0], ph[1], t_shot, 2), context(now=t_shot + 0.08))
    check("a scored hit on the walking figure is a hit, and the mode says it ABSORBED the shot",
          o7.hit and o7.points == 7 and o7.absorbed, str(o7))
    check("an on-figure shot in no ring is not a hit, and is ABSORBED too (it is on the figure)",
          (not oh.hit) and oh.target_id.endswith(":no-ring") and oh.absorbed, str(oh))
    check("it keeps coming: a hit does not stop, drop or remove the figure",
          aA.phase == "live" and mA.hits == 1 and aA.state(aA.t0 + aA.t_a + 0.1) is not None)
    check("the hit is carried by the figure in ITS coordinates (fractions of the figure, "
          "between 0 and 1), not in screen pixels",
          len(aA.marks) == 2 and all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
                                     for u, v, _g in aA.marks.holes), str(aA.marks.holes))
    u7, v7, _g7 = aA.marks.holes[0]
    uh, vh, _gh = aA.marks.holes[1]
    stamps = [t_shot + 0.9, t_shot + 1.2, t_shot + 1.5, t_shot + 1.8,
              aA.t0 + aA.t_a + 0.5 * aA.t_h,
              aA.t0 + aA.t_a + aA.t_h + 0.5 * aA.t_r]
    print("         mark of the 7 (fired at the printed 7) against the figure, frame by frame:")
    print(f"         {'t after shot':>12s} {'figure centre':>16s} {'height':>7s} "
          f"{'mark on screen':>18s} {'where a fixed hole would be':>28s} {'mark - true spot':>17s}")
    worst_err, max_travel, wins_ok, clean_ok = 0.0, 0.0, True, True
    for ts in stamps:
        c2, h2 = aA.state(ts)
        mk = aA.marks.screen_position(aA.fig, c2, h2, u7, v7)
        true7 = _screen(aA.fig, c2, h2, 410, 1200)         # the printed 7, NOW
        err = math.hypot(mk[0] - true7[0], mk[1] - true7[1])
        travel = math.hypot(mk[0] - p7[0], mk[1] - p7[1])
        worst_err, max_travel = max(worst_err, err), max(max_travel, travel)
        print(f"         {ts - t_shot:11.2f}s ({c2[0]:6.0f},{c2[1]:6.0f}) {h2:7.0f} "
              f"({mk[0]:7.1f},{mk[1]:7.1f})   ({p7[0]:7.1f},{p7[1]:7.1f}) {err:14.2f}px")
        # pixels: the marked mode against its unmarked twin, flashes long over
        _r, _l, imA = _full(mA.paint, context(now=ts))
        _r, _l, imT = _full(twinA.paint, context(now=ts))
        msk = _diff(imA, imT)
        rA = _HRF * h2 * 1.05
        mh_ = aA.marks.screen_position(aA.fig, c2, h2, uh, vh)
        near = _within(msk, mk, 2.5 * rA) + _within(msk, mh_, 2.5 * rA)
        # the running-total counter beside the figure is the one other thing a hit adds
        _fnt, cx_, cy_, ctw_, cth_ = aA.marks.counter_geometry(context(now=ts), aA.fig, c2, h2)
        cx0_, cx1_ = int(cx_) - 2, int(cx_ + ctw_) + 3
        cy0_, cy1_ = int(cy_ - cth_), int(cy_ + 0.35 * cth_)
        near += int(msk[max(0, cy0_):cy1_, max(0, cx0_):cx1_].sum())
        total = int(msk.sum())
        wins_ok = wins_ok and total > 400 and near >= 0.97 * total
        # and NOTHING is left where the fixed screen hole would have been, once the
        # figure has moved clear of it
        if travel > 6 * rA and math.hypot(p7[0] - mh_[0], p7[1] - mh_[1]) > 6 * rA:
            clean_ok = clean_ok and _within(msk, p7, 2.5 * rA) == 0
    check("the mark's screen position tracks the printed 7 across six frames of the walk "
          f"(worst {worst_err:.2f} px from it)", worst_err < 0.5, f"{worst_err:.3f}px")
    check(f"and it really moved with the figure ({max_travel:.0f} px from where the shot "
          f"landed), so a screen-fixed hole would have been left behind",
          max_travel > 60.0, f"{max_travel:.0f}px")
    check("in the rendered frames, ALL the pixels the hit adds lie on the mark's current spot "
          "(>= 97%)", wins_ok)
    check("and nothing is drawn where the shot first landed once the figure has moved on",
          clean_ok)

    # -- 24b: the seam. Absorbed shots leave no screen hole; a miss still does -------
    check("a shot that MISSES every figure is not absorbed (the host will draw its screen hole)",
          not mA.shot(modes.ShotEvent(40.0, 40.0, stamps[0], 3),
                      context(now=stamps[0])).absorbed)
    check("a plain miss outcome is never absorbed", _SO.miss().absorbed is False)
    for cls_ in (ImageTarget, PopUpFigure):
        idle_, mh2, t_h2 = _pair(cls_, 5)
        f2, c3, h3 = _geom(mh2, t_h2)
        pin = f2.innermost(c3, h3)
        o_in = mh2.shot(modes.ShotEvent(pin[1], pin[2], t_h2, 1), context(now=t_h2))
        o_off = mh2.shot(modes.ShotEvent(f2.box(c3, h3)[0] - 400.0, 300.0, t_h2, 2),
                         context(now=t_h2))
        check(f"{cls_.name}: a hit on the figure is absorbed; a shot far off it is an ordinary "
              f"miss that is NOT (it keeps its screen hole)",
              o_in.hit and o_in.absorbed and not o_off.hit and not o_off.absorbed,
              f"{o_in} / {o_off}")
    mfs = PopUpFigure()
    mfs._rng.seed(6)
    mfs.start(context(now=0.0), figure="thug01")
    check("pop-up: a false start (nothing up yet) is NOT absorbed: it keeps its screen hole",
          not mfs.shot(modes.ShotEvent(400.0, 400.0, 0.2, 1), context(now=0.2)).absorbed)
    mgap = AdvancingTarget()
    mgap._rng.seed(3)
    mgap.start(context(now=0.0), figure="thug01")
    check("advancing: a shot into the blank between walks is NOT absorbed",
          not mgap.shot(modes.ShotEvent(1500.0, 900.0, 0.2, 1), context(now=0.25)).absorbed)

    # -- 24c: the figure leaves and its holes go with it; the screen is clean ---------
    t_gone = aA.t0 + aA.duration + 0.01
    for m_ in (mA, twinA):
        m_.advance(context(now=t_gone))
    check("advancing: when the figure leaves, its marks leave with it (none carried)",
          len(aA.marks) == 0 and not aA.marks.flashes)
    _r, _l, imA = _full(mA.paint, context(now=t_gone + 0.02))
    _r, _l, imT = _full(twinA.paint, context(now=t_gone + 0.02))
    check("advancing: the screen after it left is pixel-identical to a screen where nothing "
          "was ever hit (nothing is left behind)", int(_diff(imA, imT).sum()) == 0,
          f"{int(_diff(imA, imT).sum())} pixels differ")
    drive_t = t_gone
    while not (aA.phase == "live" and aA.fig is not None) and drive_t < t_gone + 8.0:
        drive_t += 1 / 60.0
        mA.advance(context(now=drive_t))
    check("advancing: the NEXT figure starts with no holes (it did not inherit any)",
          aA.phase == "live" and len(aA.marks) == 0)
    idleP, hitP, tP = _pair(PopUpFigure, 5)
    fP, cP, hP = _geom(hitP, tP)
    zP = fP.innermost(cP, hP)
    hitP.shot(modes.ShotEvent(zP[1], zP[2], tP, 1), context(now=tP))
    for m_ in (idleP, hitP):
        m_.advance(context(now=hitP.left_at + 0.01))
    _r, _l, imH = _full(hitP.paint, context(now=hitP.left_at + 0.05))
    _r, _l, imI = _full(idleP.paint, context(now=hitP.left_at + 0.05))
    check("pop-up: after the figure leaves, no hole is left and the screen is pixel-identical "
          "to one never hit", len(hitP._marks) == 0 and int(_diff(imH, imI).sum()) == 0,
          f"{int(_diff(imH, imI).sum())} pixels differ")
    mI = ImageTarget()
    mI._rng.seed(5)
    mI.start(context(now=0.0), figure="thug01")
    zI = _zone_point(mI)
    mI.shot(modes.ShotEvent(zI[0], zI[1], 1.0, 1), context(now=1.0))
    had_hole = len(mI._marks) == 1
    mI.start(context(now=5.0))
    check("image: a new round is a new figure and takes its holes with it",
          had_hole and len(mI._marks) == 0 and not mI._marks.flashes)

    # -- 24d: a figure holding still with holes on it asks for nothing ----------------
    # These two checks are about the repaint behaviour of CARRIED holes, not about the
    # drop: five centre hits total 50 and would drop the figure at the third, so the
    # drop total is lifted for this block only (restored in the finally below).
    _drop_saved = _figmod.DROP_TOTAL
    _figmod.DROP_TOTAL = 10 ** 6
    mH = _adv(5)
    aH = mH._actors[0]
    t_hold = aH.t0 + aH.t_a + 0.1
    cH, hH = aH.state(t_hold)
    iH = aH.fig.innermost(cH, hH)
    for k_ in range(5):
        mH.shot(modes.ShotEvent(iH[1] + 6 * k_, iH[2], t_hold, 1 + k_), context(now=t_hold))
    mH.advance(context(now=t_hold + 0.65))            # the tick that ends the flashes
    asked = sum(1 for i in range(180)
                if mH.advance(context(now=t_hold + 0.7 + i / 180.0 * 0.6)))
    # 180 ticks inside the hold (its shortest length is 1.5 s): no walking, no flashes
    check("advancing: holding still with 5 holes carried, 180 ticks request no repaint "
          f"({asked}) and the mode is not animated",
          asked == 0 and mH.animated is False and len(aH.marks) == 5,
          f"{asked} requests, animated={mH.animated}")
    mHI = ImageTarget()
    mHI._rng.seed(5)
    mHI.start(context(now=0.0), figure="thug01")
    zH = _zone_point(mHI)
    for k_ in range(5):
        mHI.shot(modes.ShotEvent(zH[0] + 6 * k_, zH[1], 1.0, 1 + k_), context(now=1.0))
    mHI.advance(context(now=2.0))                       # the tick that ends the flashes
    askedI = sum(1 for i in range(180) if mHI.advance(context(now=3.0 + i / 60.0)))
    check(f"image: standing with 5 holes carried, 180 ticks request no repaint ({askedI})",
          askedI == 0 and mHI.animated is False)
    _figmod.DROP_TOTAL = _drop_saved

    # -- 24e: the cap ------------------------------------------------------------------
    mk_ = _Marks()
    f_cap = mI._fig
    for k_ in range(_MAXC + 25):
        mk_.add(f_cap, mI._centre[0], mI._centre[1], mI._centre, mI._height, 1.0, "10", True)
    check(f"a figure carries at most {_MAXC} holes (the oldest is dropped past that)",
          len(mk_) == _MAXC, f"{len(mk_)}")

    # -- 24f: the realistic worst, at the NEAREST (largest) scale ----------------------
    # A full approach with every shot landing: ~5 pulls a second per pistol for the
    # ~13 s of a walk is more than the cap, so the cap IS the worst case. The grid is
    # spaced so no two holes overlap (overlap would LOWER the lit area).
    worst_lit_no, worst_lit_with, worst_red = 0.0, 0.0, 0
    for nm_ in (HEAVIEST, "predator01"):
        mW = AdvancingTarget()
        mW._rng.seed(9)
        mW.start(context(now=0.0), figure=nm_)
        _drive_to_live(mW, 0.0)
        aW = mW._actors[0]
        t_n = aW.t0 + aW.t_a + 0.2                           # at the nearest height
        mW.started_at = -1e9
        cW, hW = aW.state(t_n)
        _r, l_no, _i = _full(mW.paint, context(now=t_n))
        for gi in range(8):
            for gj in range(5):
                aW.marks.add(aW.fig, *_screen_frac(aW.fig, cW, hW, 0.20 + gi * 0.085,
                                                    0.30 + gj * 0.08),
                             cW, hW, t_n - 5.0, "10", (gi + gj) % 3 != 0)
        assert len(aW.marks) == _MAXC
        r_w, l_with, _i = _full(mW.paint, context(now=t_n))
        print(f"         {nm_:10s} at its nearest height {hW / SH:.3f} of the screen, "
              f"{len(aW.marks)} holes carried: lit {l_no * 100:.3f}% -> {l_with * 100:.3f}%, "
              f"red excess {r_w}")
        worst_lit_no, worst_lit_with = max(worst_lit_no, l_no), max(worst_lit_with, l_with)
        worst_red = max(worst_red, r_w)
    check(f"{_MAXC} holes on the largest figure at its nearest scale stay inside the 2% "
          f"lit-area ceiling (worst {worst_lit_with * 100:.3f}%, figure alone "
          f"{worst_lit_no * 100:.3f}%) and are grey only (red excess {worst_red})",
          worst_lit_with <= 0.02 and worst_red == 0)


    # ======================================================================
    print()
    print("25. the figure drops at a running total of 30; the flash and the counter show the TOTAL")
    from lsa.modes.figure import DROP_TOTAL as _DROP
    from lsa.display.layers import BulletHoles as _BH
    check("the drop total is 30 (his number, after 21 was shown to measure nothing)",
          _DROP == 30 and _figmod.DROP_TOTAL == 30)

    def _rig(cls_, seed=21):
        m_ = cls_()
        m_._rng.seed(seed)
        m_.start(context(now=0.0), figure="thug01")
        if cls_ is ImageTarget:
            return m_, 10.0
        _drive_to_live(m_, 0.0)
        if cls_ is PopUpFigure:
            return m_, m_.shown_at + 0.2
        a_ = m_._actors[0]
        return m_, a_.t0 + a_.t_a + 0.1                    # the near hold: it stands still

    def _mk(m_):
        return m_._actors[0].marks if m_.name == "advancing" else m_._marks

    def _gone(m_):
        if m_.name == "image":
            # counted AND actually gone: the figure is lying down (no figure to hit)
            # or, with no fall pose, already replaced (nothing falling)
            return m_.dropped >= 1 and (m_._fig is None or m_._fall is None)
        if m_.name == "popup":
            return m_.state == "wait"
        return m_._actors[0].phase == "gap"

    def _fire(m_, t_, ix_, iy_, n_=[0]):
        n_[0] += 1
        fg_, cg_, hg_ = _geom(m_, t_)
        x_, y_ = _screen(fg_, cg_, hg_, ix_, iy_)
        return m_.shot(modes.ShotEvent(x_, y_, t_, 900 + n_[0]), context(now=t_))

    def _flash_word(m_):
        """The text of the newest flash on the figure, last word (the pop-up's first
        hit prefixes its time)."""
        f_ = _mk(m_).flashes
        return f_[-1][3].split()[-1] if f_ else None

    RING = {7: (410, 1200), 8: (516, 1200), 9: (606, 1200), 10: (800, 1200)}
    NORING = (800, 255)                                    # above the head dome: on the figure, no ring
    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        nm_ = cls_.name
        # -- 7 then 9 then 8 flash 7, 16, 24 -----------------------------------
        m_, t_ = _rig(cls_)
        seen_ = []
        for k_, pts_ in enumerate((7, 9, 8)):
            o_ = _fire(m_, t_ + 0.02 * k_, *RING[pts_])
            seen_.append(_flash_word(m_))
            assert o_.hit and o_.points == pts_, (nm_, pts_, o_)
        check(f"{nm_}: a 7, then a 9, then an 8 flash 7, 16, 24 (the running TOTAL, not the ring)",
              seen_ == ["7", "16", "24"], str(seen_))
        check(f"{nm_}: the counter beside the figure reads '24 / 30' and the figure is still up",
              _mk(m_).counter_text() == "24 / 30" and _mk(m_).total == 24 and not _gone(m_))
        # -- three centre hits reach 30, two do not ------------------------------
        m_, t_ = _rig(cls_)
        _fire(m_, t_, *RING[10]); _fire(m_, t_ + 0.02, *RING[10])
        up2_ = not _gone(m_) and _mk(m_).total == 20
        _fire(m_, t_ + 0.04, *RING[10])
        check(f"{nm_}: three centre hits (10 + 10 + 10) reach 30 and the figure is gone at once; "
              f"two (20) leave it up", up2_ and _gone(m_))
        # -- five edge hits reach 30, four (28) do not ---------------------------
        m_, t_ = _rig(cls_)
        for k_ in range(4):
            _fire(m_, t_ + 0.02 * k_, *RING[7])
        up4_ = not _gone(m_) and _mk(m_).total == 28
        _fire(m_, t_ + 0.1, *RING[7])
        check(f"{nm_}: five edge hits (7 x 5) reach 30; four (28) leave the figure up",
              up4_ and _gone(m_))
        # -- one 10 onto an already-23 total: 33, gone at once, flash says 33 ------
        m_, t_ = _rig(cls_)
        for k_, pts_ in enumerate((7, 8, 8)):
            _fire(m_, t_ + 0.02 * k_, *RING[pts_])
        at23_ = _mk(m_).total == 23 and not _gone(m_)
        o_ = _fire(m_, t_ + 0.1, *RING[10])
        scr_ = m_.flashes[-1][3].split()[-1] if m_.flashes else None
        check(f"{nm_}: a single 10 on an already-23 total (7+8+8) drops the figure that instant, "
              f"scores its 10, and the flash left at the spot reads 33 (not 10)",
              at23_ and _gone(m_) and o_.hit and o_.points == 10 and o_.absorbed and scr_ == "33",
              f"flash {scr_}")
        # -- a no-ring shot scores nothing and never advances the total -------------
        m_, t_ = _rig(cls_)
        for k_ in range(6):
            o_ = _fire(m_, t_ + 0.02 * k_, *NORING)
        check(f"{nm_}: six shots on the figure but in no ring add NOTHING (total 0, still up, "
              f"labelled 'no ring')", _mk(m_).total == 0 and not _gone(m_) and not o_.hit
              and _flash_word(m_) == "ring", f"total {_mk(m_).total} {_flash_word(m_)}")
        _fire(m_, t_ + 0.2, *RING[9])
        check(f"{nm_}: ...and a ring hit after them starts the total at its own value (9)",
              _mk(m_).total == 9)
        # -- each figure starts from zero ---------------------------------------------
        m_, t_ = _rig(cls_)
        for k_ in range(3):
            _fire(m_, t_ + 0.02 * k_, *RING[10])
        if nm_ != "image":
            _drive_to_live(m_, t_ + 0.1)
        check(f"{nm_}: the next figure starts from zero (the total belongs to the figure, not "
              f"the round) while the round's tally carries on",
              _mk(m_).total == 0 and len(_mk(m_)) == 0 and m_.hits == 3)
    # -- the pop-up ends its exposure EARLY, and does not call it unengaged -----------
    mp_, tp_ = _rig(PopUpFigure)
    left_before_ = mp_.left_at
    for k_ in range(3):
        _fire(mp_, tp_ + 0.02 * k_, *RING[10])
    mp_.advance(context(now=tp_ + 0.1))
    check("pop-up: reaching 30 ends the exposure at once, long before its timer "
          f"({tp_ + 0.06 - mp_.shown_at:.2f} s of {mp_.EXPOSURE_S} s), as a hit figure not an unengaged one",
          mp_.state == "wait" and mp_.left_at < left_before_ - 1.0 and mp_.unengaged == 0)
    mp_.started_at = -1e9                                   # the standing hint is not the figure
    rb_, lb_, _ib = _full(mp_.paint, context(now=tp_ + 1.5))
    check("pop-up: after the drop the screen shows only the blank wait (no figure)",
          lb_ < 0.002 and rb_ == 0, f"{lb_ * 100:.3f}%")
    late_ = _fire(mp_, tp_ + 0.03, *RING[10])
    check("pop-up: a shot still arriving for the figure that already dropped scores nothing",
          not late_.hit and mp_.hits == 3)
    # -- the counter RIDES the figure and is sized from it ------------------------------
    ma_, ta_ = _rig(AdvancingTarget)
    aa_ = ma_._actors[0]
    _fire(ma_, aa_.t0 + 0.4 * aa_.t_a, *RING[8])
    ctxa_ = context(now=aa_.t0 + 0.4 * aa_.t_a + 0.05)
    offs_, sizes_, hs_ = [], [], []
    for frac_ in (0.4, 0.6, 0.8):
        ts_ = aa_.t0 + frac_ * aa_.t_a
        c_, h_ = aa_.state(ts_)
        fnt_, cx_, cy_, ctw_, cth_ = aa_.marks.counter_geometry(ctxa_, aa_.fig, c_, h_)
        l_, tp0_, w_, hh_ = aa_.fig.box(c_, h_)
        offs_.append((cx_ - (l_ + w_)) / hh_)
        sizes_.append(fnt_.pointSize()); hs_.append(h_)
    check("advancing: the counter keeps the same place relative to the walking figure at three "
          "sizes (it travels and scales with it)", max(offs_) - min(offs_) < 0.002,
          f"{[round(o, 4) for o in offs_]}")
    check("advancing: its text size grows with the figure (a fraction of the figure's height, "
          "not a pixel count)", sizes_[0] < sizes_[2], f"{sizes_} for heights {[round(h) for h in hs_]}")
    far_ = aa_.h_far
    fnt_, _x, _y, ctw_, cth_ = aa_.marks.counter_geometry(ctxa_, aa_.fig, (SW / 2, SH / 2), far_)
    check("the counter stays legible from the firing line even at the figure's farthest "
          f"({cth_}px tall text on a {SH}px screen, >= 1.5%)", cth_ >= 0.015 * SH, f"{cth_}")
    c1080_ = RenderContext(now=0.0, dt=1 / 60.0, rect=QRect(0, 0, 1920, 1080),
                           stats=SessionStats(), playable=None)
    _f2, _x2, _y2, _tw2, th2_ = aa_.marks.counter_geometry(c1080_, aa_.fig, (960, 540), far_ / 2.0)
    _f1, _x1, _y1, _tw1, th1_ = aa_.marks.counter_geometry(ctxa_, aa_.fig, (SW / 2, SH / 2), far_)
    check("nothing is sized in screen pixels: half the screen, half the figure, half the text "
          f"({th2_} vs {th1_})", 0.45 <= th2_ / th1_ <= 0.55, f"{th2_}/{th1_}")

    # -- rendering: red excess EXACTLY 0 in every state, alone AND with the holes layer -----
    def _render_states(cls_):
        out_ = []
        m_, t_ = _rig(cls_)
        holes_ = _BH()
        holes_.add(SW * 0.15, SH * 0.2)
        holes_.add(SW * 0.85, SH * 0.8)
        for h_ in holes_.holes:
            h_.born -= 5.0                                   # settled
        live_ = _BH()
        live_.add(SW * 0.5, SH * 0.1)                          # a fresh one: its impact flash
        _fire(m_, t_, *RING[7])
        out_.append(("7/30, flash live", m_, context(now=t_ + 0.05)))
        _fire(m_, t_ + 0.02, *RING[9]); _fire(m_, t_ + 0.04, *RING[8])
        out_.append(("24/30, flash live", m_, context(now=t_ + 0.07)))
        out_.append(("24/30, flash over", m_, context(now=t_ + (0.7 if cls_ is AdvancingTarget else 1.2))))
        mm_, tt_ = _rig(cls_, 22)
        for k_, pts_ in enumerate((7, 8, 8)):
            _fire(mm_, tt_ + 0.02 * k_, *RING[pts_])
        _fire(mm_, tt_ + 0.1, *RING[10])
        out_.append(("the drop (flash 33 at the spot)", mm_, context(now=tt_ + 0.15)))
        return out_, holes_, live_

    worst_ = 0
    n_states_ = 0
    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        sts_, holes_, live_ = _render_states(cls_)
        for lbl_, m_, c_ in sts_:
            m_.started_at = -1e9
            r_alone_, l_alone_, _i = _full(m_.paint, c_)
            r_both_, l_both_, _i = _full(lambda p_, cc_: (m_.paint(p_, cc_), holes_.paint(p_, cc_),
                                                          live_.paint(p_, cc_)), c_)
            n_states_ += 1
            worst_ = max(worst_, r_alone_, r_both_)
            check(f"{cls_.name} [{lbl_}]: red excess exactly 0 alone ({r_alone_}) and composited "
                  f"with the holes layer ({r_both_}); lit {l_alone_ * 100:.3f}% / {l_both_ * 100:.3f}%",
                  r_alone_ == 0 and r_both_ == 0 and l_both_ <= 0.02)
    print(f"         {n_states_} state renderings, worst red excess {worst_}")
    # the counter is real pixels, inside its own rectangle, and nothing else changes
    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        m_, t_ = _rig(cls_)
        _fire(m_, t_, *RING[8])
        m_.started_at = -1e9
        tn_ = t_ + (0.7 if cls_ is AdvancingTarget else 1.2)
        c_ = context(now=tn_)
        _r, _l, with_ = _full(m_.paint, c_)
        mk_ = _mk(m_)
        saved_ = type(mk_)._paint_counter
        type(mk_)._paint_counter = lambda self, *a, **k: None
        try:
            _r, _l, without_ = _full(m_.paint, c_)
        finally:
            type(mk_)._paint_counter = saved_
        d_ = (with_ != without_).any(axis=2)
        fg_, cg_, hg_ = _geom(m_, tn_)
        fnt_, cx_, cy_, ctw_, cth_ = mk_.counter_geometry(c_, fg_, cg_, hg_)
        x0_, x1_ = int(cx_) - 2, int(cx_ + ctw_) + 3
        y0_, y1_ = int(cy_ - cth_), int(cy_ + 0.35 * cth_)
        inside_ = int(d_[max(0, y0_):y1_, max(0, x0_):x1_].sum())
        check(f"{cls_.name}: the counter is drawn ({int(d_.sum())} px) and ALL of it lies in "
              f"its own rectangle beside the figure", int(d_.sum()) > 300 and inside_ == int(d_.sum()),
              f"{inside_}/{int(d_.sum())}")

    # -- a figure holding still with the counter up asks for no repaint -----------------
    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        m_, t_ = _rig(cls_)
        for k_, pts_ in enumerate((7, 9, 8)):
            _fire(m_, t_ + 0.01 * k_, *RING[pts_])
        base_ = t_ + (0.7 if cls_ is AdvancingTarget else 1.2)
        m_.advance(context(now=base_))                       # the tick that ends the flashes
        span_ = 0.6 if cls_ is AdvancingTarget else 1.0
        asked_ = sum(1 for i in range(180)
                     if m_.advance(context(now=base_ + 0.01 + i / 180.0 * span_)))
        check(f"{cls_.name}: with '24 / 30' showing and the figure holding still, 180 ticks "
              f"request at most 5 repaints ({asked_}) and no flash is left animating",
              _mk(m_).total == 24 and asked_ <= 5 and not _mk(m_).flashes,
              f"{asked_} requests, animated={m_.animated}")

    # -- THE LIT-AREA WORST CASE: the brightest figure, largest, the most it can carry ------
    # Before it drops it carries at most four 7s (28 of 30) and, past those, only no-ring
    # marks (capped at MAX_CARRIED). Two flashes live, the counter at a two-digit total.
    worst_lit = {}
    counter_cost = {}
    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        mw_ = cls_()
        mw_._rng.seed(9)
        mw_.start(context(now=0.0), figure=HEAVIEST)
        if cls_ is PopUpFigure:
            mw_._next_frac = 1.0
            _drive_to_live(mw_, 0.0)
            tw_ = mw_.shown_at + 0.2
        elif cls_ is AdvancingTarget:
            _drive_to_live(mw_, 0.0)
            aw_ = mw_._actors[0]
            tw_ = aw_.t0 + aw_.t_a + 0.2
        else:
            tw_ = 50.0
        fg_, cg_, hg_ = _geom(mw_, tw_)
        for k_ in range(4):
            _fire(mw_, tw_ + 0.001 * k_, 410 + 6 * k_, 1200)
        mk_ = _mk(mw_)
        for gi in range(9):
            for gj in range(4):
                x_, y_ = _screen_frac(fg_, cg_, hg_, 0.12 + gi * 0.095, 0.06 + gj * 0.23)
                mk_.add(fg_, x_, y_, cg_, hg_, tw_ + 0.01, "no ring", False)
        # 4 ring-7 holes + 36 no-ring = the cap of 40; the newest two flashes are live
        assert len(mk_) == _figmod.MAX_CARRIED and mk_.total == 28, (len(mk_), mk_.total)
        c_ = context(now=tw_ + 0.05)
        mw_.started_at = -1e9
        r_a, l_a, _i = _full(mw_.paint, c_)
        mw_.started_at = tw_ - 0.05
        r_h, l_h, _i = _full(mw_.paint, c_)
        # and with 80 settled global holes elsewhere on the screen (misses)
        gh_ = _BH()
        for k_ in range(80):
            gh_.add(SW * (0.05 + 0.9 * (k_ % 16) / 15.0), SH * (0.05 + 0.2 * (k_ // 16)))
        for h_ in gh_.holes:
            h_.born -= 5.0
        mw_.started_at = -1e9
        r_g, l_g, _i = _full(lambda p_, cc_: (mw_.paint(p_, cc_), gh_.paint(p_, cc_)), c_)
        # what the counter ALONE costs in this state
        _mk_t = type(mk_)
        _saved_pc = _mk_t._paint_counter
        _mk_t._paint_counter = lambda self, *a, **k: None
        try:
            _r0, l_nocounter, _i = _full(mw_.paint, c_)
        finally:
            _mk_t._paint_counter = _saved_pc
        counter_cost[cls_.name] = l_a - l_nocounter
        worst_lit[cls_.name] = (l_a, l_h, l_g, max(r_a, r_h, r_g))
        check(f"{cls_.name}: WORST CASE {HEAVIEST} at its largest, 4 hits (28) + 36 no-ring marks "
              f"(40 carried), counter '28 / 30', 2 flashes: lit {l_a * 100:.3f}% "
              f"(counter alone {counter_cost[cls_.name] * 100:.3f}%), "
              f"red excess {worst_lit[cls_.name][3]}",
              l_a <= 0.02 and worst_lit[cls_.name][3] == 0 and counter_cost[cls_.name] > 0.00005)
    # Informational, deliberately NOT asserted: the first 8 s hint on top of 40 carried
    # holes, and 80 settled screen holes on top of that, is a combination no real
    # string reaches (the carried holes alone take ~8 s of two pistols at full rate).
    print("         lit-area worst case by mode (figure only / hint up / + 80 screen holes): "
          + "; ".join(f"{k} {v[0] * 100:.3f}/{v[1] * 100:.3f}/{v[2] * 100:.3f}%"
                      for k, v in worst_lit.items()))

    # ======================================================================
    print()
    print("26. bullet holes are 20% smaller")
    c4k_ = context(now=0.0)
    c2k_ = RenderContext(now=0.0, dt=1 / 60.0, rect=QRect(0, 0, 1920, 1080),
                         stats=SessionStats(), playable=None)
    old4_, old2_ = max(7.0, SH / 110.0), max(7.0, 1080 / 110.0)
    check("the hole radius is 0.8 of what it was, at 4K and at 1080p",
          abs(_BH.hole_radius(c4k_) / old4_ - 0.8) < 1e-9
          and abs(_BH.hole_radius(c2k_) / old2_ - 0.8) < 1e-9,
          f"{_BH.hole_radius(c4k_):.2f} vs {old4_:.2f}")
    check("...and it is still a fraction of the screen (4K hole = 2 x the 1080p hole)",
          abs(_BH.hole_radius(c4k_) / _BH.hole_radius(c2k_) - 2.0) < 1e-9)

    def _hole_extent(scale_):
        saved_ = _BH.SIZE_SCALE
        _BH.SIZE_SCALE = scale_
        try:
            h_ = _BH()
            h_.add(SW / 2, SH / 2)
            for hh_ in h_.holes:
                hh_.born -= 5.0
            r_, _l, img_ = _full(h_.paint, context())
        finally:
            _BH.SIZE_SCALE = saved_
        lum_ = cv2.cvtColor(np.ascontiguousarray(img_[..., :3]), cv2.COLOR_BGR2GRAY)
        ys_, xs_ = np.nonzero(lum_ > 25)
        return r_, int(xs_.max() - xs_.min()), int(ys_.max() - ys_.min()), int((lum_ > 25).sum())
    r_new_, w_new_, h_new_, a_new_ = _hole_extent(0.8)
    r_old_, w_old_, h_old_, a_old_ = _hole_extent(1.0)
    check(f"a rendered hole really is 0.8 of the old width ({w_new_} vs {w_old_} px) and 0.64 of "
          f"its area ({a_new_} vs {a_old_} px), red excess {r_new_}",
          0.75 <= w_new_ / w_old_ <= 0.85 and 0.55 <= a_new_ / a_old_ <= 0.75 and r_new_ == 0)

    # ----------------------------------------------------------------------
    print()
    print("27. a figure is SOLID: bullet holes underneath do not show through it")
    # His words, 2026-10-17: "when they go over the existing shots, they should
    # hide (be in front of) the existing shots. Otherwise they're floating
    # through the existing shots and this is unnatural." The holes layer was
    # already BELOW the modes (z 15 against 40); the sprite is line art with a
    # transparent ground, so they showed through every dark part of it. Layers
    # above announce their solid shapes (Layer.occluders) and BulletHoles leaves
    # them out. Painted through the REAL Scene.paint, which gathers the shapes.
    from lsa.display.scene import Scene as _Sc27
    from lsa.display import layers as _L27

    def _paint27(sc_, t_):
        sc_._context = lambda now, dt, rect, _t=t_: context(now=_t)
        im_ = _QI3(SW, SH, _QI3.Format_RGB32)
        im_.fill(0xFF0A0A0A)
        pp_ = _QP3(im_)
        sc_.paint(pp_, QRect(0, 0, SW, SH))
        pp_.end()
        a_ = np.frombuffer(im_.constBits(), np.uint8).reshape(SH, im_.bytesPerLine() // 4, 4)
        return a_[:, :SW, 1].astype(int)

    for cls27 in (ImageTarget, PopUpFigure, AdvancingTarget):
        m27 = cls27()
        m27._rng.seed(5)
        m27.start(context(now=40.0), figure="thug01")
        if cls27.name == "image":
            t27 = 40.1
        elif cls27.name == "popup":
            t27 = _drive_to_live(m27, 40.0)
        else:
            _drive_to_live(m27, 40.0)
            a27 = m27._actors[0]
            t27 = a27.t0 + a27.t_a + 0.1
        sc27 = _Sc27()
        sc27.add(_L27.SolidBackground())
        holes27 = _L27.BulletHoles()
        sc27.add(holes27)
        host27 = modes.ModeHost()
        sc27.add(host27)
        host27.set_mode(m27, None)
        if cls27.name == "advancing":
            fig27 = a27.fig
            cen27, hh27 = a27.state(t27)
        else:
            fig27, cen27, hh27 = m27._fig, m27._centre, m27._height
        l27, t0_27, w27, h27 = fig27.box(cen27, hh27)
        er27 = cv2.erode(fig27._sil.astype(np.uint8), np.ones((9, 9), np.uint8))
        iy27, ix27 = np.nonzero(er27)
        mh27, mw27 = fig27._sil.shape
        pick27 = np.linspace(0, len(iy27) - 1, 12).astype(int)
        inside27 = [(l27 + (ix27[k] + .5) / mw27 * w27, t0_27 + (iy27[k] + .5) / mh27 * h27)
                    for k in pick27]
        outside27 = [(max(60.0, l27 - 160 - i * 50), t0_27 + 120 + i * 70) for i in range(5)]
        for i27, (x27, y27) in enumerate(inside27 + outside27):
            holes27.add(x27, y27, i27, 0)
        for hh_ in holes27.holes:
            hh_.born -= 5.0                      # settled, not flashing
        holes27.visible = False
        bare27 = _paint27(sc27, t27)
        holes27.visible = True
        solid27 = _paint27(sc27, t27)
        orig27 = modes.ModeHost.occluders
        modes.ModeHost.occluders = lambda self, ctx: []
        try:
            floaty27 = _paint27(sc27, t27)
        finally:
            modes.ModeHost.occluders = orig27

        def _px27(img27, pts27, r27=14):
            d27 = np.abs(img27 - bare27) > 6
            return sum(int(d27[max(0, int(y - r27)):int(y + r27),
                               max(0, int(x - r27)):int(x + r27)].sum()) for x, y in pts27)
        in_solid, in_floaty = _px27(solid27, inside27), _px27(floaty27, inside27)
        out_solid, out_floaty = _px27(solid27, outside27), _px27(floaty27, outside27)
        check(f"{cls27.name}: holes under the figure's silhouette are hidden "
              f"({in_solid} px show against {in_floaty} when it is not solid)",
              in_floaty > 2000 and in_solid <= 0.15 * in_floaty)
        check(f"{cls27.name}: holes OUTSIDE the figure are still drawn "
              f"({out_solid} px, unchanged from {out_floaty})",
              out_floaty > 500 and out_solid >= 0.95 * out_floaty)

    # a layer ABOVE the shape is not cut: only layers beneath obey. (Last mode
    # above, advancing, is still up; lift the holes over the modes.)
    holes27.z = 90
    sc27._layers.sort(key=lambda la: la.z)
    above27 = _paint27(sc27, t27)
    in_above = _px27(above27, inside27)
    check(f"an occluder only hides layers BELOW it: holes at z 90 keep showing "
          f"({in_above} px against {in_floaty} unobstructed)",
          in_above >= 0.9 * in_floaty)
    # THE BALLS ARE SOLID NOW, and this check asserted the opposite. They are
    # DRAWN as outlines -- a fill sits exactly where the dot lands and
    # swallows the shot (B10) -- so "outline" and "not solid" had been one
    # statement. He separated them on 2026-10-18: "the figures (balls,
    # people-like cutouts, etc.) that go over holes should obscure the holes".
    # What a ball looks like and what it OCCUPIES are different questions, and
    # only the second decides whether a hole shows through it.
    from lsa.modes.balls import BouncingBalls as _BB27
    from lsa.modes.ball import EscalatingBall as _EB27
    for cls_o in (_BB27, _EB27):
        mo = cls_o()
        mo.start(context(now=1.0))
        occ_o = mo.occluders(context(now=1.0))
        check(f"{cls_o.name}: every ball on screen announces a solid disc",
              len(occ_o) >= 1 and all(o.rect.width() > 0 for o in occ_o),
              f"{len(occ_o)} occluder(s)")
    _figmod.FALL_S = _FALL_S_REAL
    # ======================================================================
    print()
    print("28. the fall: a dropped figure lies down for one second, then goes")
    import shutil as _sh
    import tempfile as _tf
    from lsa.modes.figure import FALL_S as _FALL, Fall as _FallCls
    from lsa.artwork import load as _load
    _real_targets = _cat.targets_dir()

    # -- the catalogue: everything in targets/ is a figure, falls live apart ----
    root_jpgs = sorted(p.stem for p in _real_targets.glob("*.jpg"))
    bad_root = _cat.excluded()
    check("EVERY image directly in targets/ has verified scoring zones (no exceptions)",
          bad_root == {} and root_jpgs == _cat.all_names(), str(bad_root))
    check("no fall pose sits among the figures (none named *-fall in targets/)",
          not [n for n in root_jpgs if n.endswith("-fall")], str(root_jpgs))
    check("fall poses never enter the random draw",
          not any("fall" in n for n in _cat.names()) and len(_cat.names()) == len(root_jpgs))
    has_fall = [n for n in _cat.names() if _cat.fall_path(n)]
    # REPORT, never REQUIRE (C11, read half): how many of his figures have a pose
    # is his data, and it changes when he finishes his artwork. What stays
    # asserted is the PAIRING, judged against an independent listing of the
    # folder: every pose file reaches the figure of its basename, and every
    # figure that has a file reaches it -- so a mis-paired or missed pose fails.
    _fall_dir = _real_targets / "fall"
    _on_disk = {}
    for _f in (sorted(_fall_dir.glob("*.jpg")) if _fall_dir.is_dir() else []):
        _b = _f.stem[:-5] if _f.stem.endswith("-fall") else _f.stem
        _on_disk.setdefault(_b, _f)
    _paired_ok = all(_cat.fall_path(n) == _on_disk.get(n) for n in _cat.all_names())
    _reachable = all(b in _cat.all_names() for b in _on_disk)
    print(f"         INFO: {len(has_fall)} of {len(_cat.names())} figures in his folder have a fall pose "
          f"(without: {sorted(set(_cat.names()) - set(has_fall)) or 'none'})")
    check("the real folder pairs each fall pose to the figure of its basename, none missed or unreachable "
          "(how many figures have one is reported, not required)",
          _paired_ok and _reachable and _cat.unpaired_falls() == {}
          and sorted(has_fall) == sorted(n for n in _cat.names() if n in _on_disk),
          f"{len(has_fall)} of {len(_cat.names())}")

    # -- the fixture for the optional-pose path: its OWN folder, not his -------
    # A figure that simply disappears when it drops is behaviour of the CODE, so
    # it is exercised on a private folder holding one figure with no pose and
    # one with a pose, built from copies of his art. Reached through LSA_TARGETS,
    # the documented test override of targets_dir().
    NOPOSE, POSED = "nopose01", "posed01"
    _src_names = _cat.names()
    _fix = Path(_tf.mkdtemp())
    for _dst, _src in ((NOPOSE, _src_names[0]), (POSED, _src_names[1])):
        _sh.copy(_real_targets / f"{_src}.jpg", _fix / f"{_dst}.jpg")
        _sh.copy(_real_targets / f"{_src}.zones.json", _fix / f"{_dst}.zones.json")
    (_fix / "fall").mkdir()
    _sh.copy(_real_targets / "fall" / f"{_src_names[0]}.jpg", _fix / "fall" / f"{POSED}.jpg")
    check("fixture: the private folder has a figure with no pose and one with a pose, both drawable",
          _cat.names(_fix) == [NOPOSE, POSED] and _cat.fall_path(NOPOSE, _fix) is None
          and _cat.fall_path(POSED, _fix) is not None, str(_cat.names(_fix)))

    class _OnFixture:
        """Point targets_dir() at the fixture for the duration of a with-block."""
        def __enter__(self):
            self._old = os.environ.get("LSA_TARGETS")
            os.environ["LSA_TARGETS"] = str(_fix)
        def __exit__(self, *_a):
            if self._old is None:
                os.environ.pop("LSA_TARGETS", None)
            else:
                os.environ["LSA_TARGETS"] = self._old

    tmp27 = Path(_tf.mkdtemp())
    try:
        for n in ("thug01", "thug02"):
            _sh.copy(_real_targets / f"{n}.jpg", tmp27 / f"{n}.jpg")
            _sh.copy(_real_targets / f"{n}.zones.json", tmp27 / f"{n}.zones.json")
        (tmp27 / "fall").mkdir()
        _sh.copy(_real_targets / "fall" / "thug01.jpg", tmp27 / "fall" / "thug01-fall.jpg")  # suffix
        _sh.copy(_real_targets / "fall" / "thug01.jpg", tmp27 / "fall" / "ghost.jpg")        # orphan
        check("a file saved as fall/<name>-fall.jpg still pairs with its figure",
              _cat.fall_path("thug01", tmp27) == tmp27 / "fall" / "thug01-fall.jpg")
        check("a fall pose with no figure is reported, by name, as unreachable",
              list(_cat.unpaired_falls(tmp27)) == ["ghost.jpg"], str(_cat.unpaired_falls(tmp27)))
        check("a figure whose pose is missing (thug02) has none, and is still drawn",
              _cat.fall_path("thug02", tmp27) is None and "thug02" in _cat.names(tmp27))
        _sh.copy(_real_targets / "fall" / "thug01.jpg", tmp27 / "fall" / "thug01.jpg")
        check("two files for one figure: the plain name wins, the other is reported",
              _cat.fall_path("thug01", tmp27) == tmp27 / "fall" / "thug01.jpg"
              and "thug01-fall.jpg" in _cat.unpaired_falls(tmp27))
    finally:
        _sh.rmtree(tmp27, ignore_errors=True)

    # -- every pose: grey, and inside its own lit-area budget ----------------
    worst_fr, worst_fl = 0, 0.0
    rows = []
    for nm in has_fall:
        fg = _Fig(nm, SW, SH, BUDGET)
        h_stand = fg.fit_height(SH * 0.98, SH, 1.0)
        fl = _FallCls.start(fg, (SW / 2, SH / 2), h_stand, 10.0, SH)
        fg.prepare_fall(h_stand, SH)
        r_, l_, _im = _full(lambda p_, c_, f_=fl: f_.paint(p_), context(now=10.5))
        rows.append((nm, r_, l_, fl.height / SH, fg.fall_cap, fg.cap))
        worst_fr, worst_fl = max(worst_fr, r_), max(worst_fl, l_)
    for nm, r_, l_, hfrac, fcap, scap in rows:
        print(f"         {nm:10s} fall pose drawn at {hfrac:.3f} of screen height "
              f"(standing cap {scap:.3f}, its own cap {fcap:.3f}): {l_ * 100:.3f}% lit, red excess {r_}")
    check("no fall pose is drawn taller than its OWN lit-area cap (alien01's is lower than its standing one)",
          all(hf <= fcap + 1e-6 for _n, _r, _l, hf, fcap, _s in rows)
          and any(fcap < scap for _n, _r, _l, _h, fcap, scap in rows))
    check(f"every fall pose renders at red excess exactly 0 (worst {worst_fr})", worst_fr == 0)
    check(f"every fall pose at its chosen size lights <= 1.5% of the screen (worst {worst_fl * 100:.3f}%)",
          worst_fl <= 0.015 + 2e-4)
    pred_files = {n: _load.red_excess_of_file(_cat.fall_path(n)) for n in has_fall}
    check(f"no fall file stores more red excess than his standing art does (worst "
          f"{max(pred_files.values(), default=0)}, limit 10); the app draws them grey regardless",
          all(v <= 10 for v in pred_files.values()), str(pred_files))
    if "predator01" in pred_files:
        check(f"predator01's fall file is neutral on disk too (red excess {pred_files['predator01']}; it was 18)",
              pred_files["predator01"] <= 2)
    else:
        print("         INFO: predator01 has no fall pose in his folder; its on-disk neutrality was not checked")

    # -- behaviour, mode by mode ----------------------------------------------
    def _down(cls_, seed=21, name="thug01"):
        """A mode driven until the figure has just dropped. -> (mode, t_drop)."""
        m_ = cls_()
        m_._rng.seed(seed)
        m_.start(context(now=0.0), figure=name)
        if cls_ is ImageTarget:
            t_ = 10.0
            pt_ = _zone_point(m_)
        elif cls_ is PopUpFigure:
            _drive_to_live(m_, 0.0)
            t_ = m_.shown_at + 0.2
            pt_ = _zone_point(m_)
        else:
            _drive_to_live(m_, 0.0)
            a_ = m_._actors[0]
            t_ = a_.t0 + a_.t_a + 0.1
            c_, h_ = a_.state(t_)
            i_ = a_.fig.innermost(c_, h_)
            pt_ = (i_[1], i_[2])
        for k_ in range(3):
            t_ += 0.05 * k_
            out_ = m_.shot(modes.ShotEvent(pt_[0], pt_[1], t_, k_ + 1), context(now=t_))
            assert out_.hit and out_.points == 10, (cls_.name, out_)
        return m_, t_

    def _fall_of(m_):
        return m_._actors[0].fall if m_.name == "advancing" else m_._fall

    def _marks_of(m_):
        return m_._actors[0].marks if m_.name == "advancing" else m_._marks

    for cls_ in (ImageTarget, PopUpFigure, AdvancingTarget):
        m_, t0_ = _down(cls_)
        fl_ = _fall_of(m_)
        check(f"{cls_.name}: reaching 30 shows the fall pose where it stood, for exactly {_FALL} s",
              fl_ is not None and abs((fl_.t1 - fl_.t0) - 1.0) < 1e-9 and fl_.t0 == t0_,
              str(None if fl_ is None else (fl_.t0, fl_.t1)))
        red_, lit_, _img = _full(m_.paint, context(now=t0_ + 0.5))
        m2_, _ = _down(cls_)
        if m2_.name == "advancing":
            m2_._actors[0].fall = None
        else:
            m2_._fall = None
        _r2, l2_, _i2 = _full(m2_.paint, context(now=t0_ + 0.5))
        check(f"{cls_.name}: the pose is really on screen mid-fall (lit {lit_ * 100:.3f}% against "
              f"{l2_ * 100:.3f}% without it), red excess {red_}",
              lit_ > l2_ + 0.002 and red_ == 0)
        m3_, _ = _down(cls_)
        m3_.started_at = t0_ + 0.4                       # the standing hint is up as well
        r3_, l3_, _i3 = _full(m3_.paint, context(now=t0_ + 0.5))
        check(f"{cls_.name}: pose + standing hint + the last flash lit {l3_ * 100:.3f}% (<= 2%), red excess {r3_}",
              l3_ <= 0.02 and r3_ == 0)
        mk_ = _marks_of(m_)
        check(f"{cls_.name}: the figure's marks and counter do not persist onto the pose",
              len(mk_) == 0 and mk_.total == 0)
        # Side A (occlusion) x side B (fall): the pose is a DIFFERENT image from
        # the standing art, so the standing silhouette must never be announced as
        # solid while the pose is what is drawn. Nothing is announced mid-fall.
        check(f"{cls_.name}: mid-fall it announces no solid shape (the standing silhouette is not what is drawn)",
              m_.occluders(context(now=t0_ + 0.5)) == [])
        before = (m_.shots, m_.hits)
        fs0 = m_.fall_shots
        fb = getattr(m_, "false_starts", 0)
        o_ = m_.shot(modes.ShotEvent(1500.0, 900.0, t0_ + 0.4, 9), context(now=t0_ + 0.4))
        check(f"{cls_.name}: a shot during the fall is not a hit, a miss, a false start or a no-target; "
              "it is counted apart and leaves its ordinary hole",
              (not o_.hit) and o_.target_id == "fallen" and not o_.absorbed and o_.draw_hole
              and (m_.shots, m_.hits) == before and m_.fall_shots == fs0 + 1
              and getattr(m_, "false_starts", 0) == fb, str(o_))
        o_ = m_.shot(modes.ShotEvent(1500.0, 900.0, t0_ - 0.02, 10), context(now=t0_ + 0.45))
        check(f"{cls_.name}: a late pulse fired just before the drop is ignored too",
              o_.target_id == "fallen" and (m_.shots, m_.hits) == before)
        mv_ = (m_.moving_at(t0_), m_.moving_at(t0_ + 0.5), m_.moving_at(t0_ + 0.999),
               m_.moving_at(t0_ + 1.001))
        check(f"{cls_.name}: moving_at is true for the whole second after the drop and false at the "
              f"drop shot's own instant and after it: {mv_}",
              mv_ == (False, True, True, False))
        m_.advance(context(now=t0_ + 1.2))
        check(f"{cls_.name}: moving_at stays true for a late shot dated inside the fall, even after it ended",
              m_.moving_at(t0_ + 0.3) and not m_.moving_at(t0_ + 1.5))
        mx_, tx_ = _down(cls_)
        up_ = mx_.animated
        mx_.advance(context(now=tx_ + 1.05))
        check(f"{cls_.name}: it reports animating while the pose is up ({up_}) and the pose is gone after "
              "its second",
              up_ and _fall_of(mx_) is None)

    # -- the next figure: the fall costs one second, no more ------------------
    m_, t0_ = _down(ImageTarget)
    m_.advance(context(now=t0_ + 0.99))
    still = m_._fig is None and m_._fall is not None
    m_.advance(context(now=t0_ + 1.0))
    check("image: the next figure appears when the second is up, not before and not later",
          still and m_._fig is not None and m_._fall is None)
    with _OnFixture():
        m_, t0_ = _down(ImageTarget, name=NOPOSE)
    check("image: a figure with no pose is replaced at once, as it always was (the fall is optional)",
          m_._fall is None and m_._fig is not None and m_.dropped == 1
          and not m_.moving_at(t0_ + 0.5))
    m_, t0_ = _down(PopUpFigure)
    check("pop-up: the next wait runs during the fall, so the next figure comes no sooner than the "
          "fall's end and no later than it would have",
          m_.next_at >= t0_ + _FALL and PopUpFigure.MIN_WAIT > _FALL, f"{m_.next_at - t0_:.2f}s")
    with _OnFixture():
        m_, t0_ = _down(PopUpFigure, name=NOPOSE)
        _late = m_.shot(modes.ShotEvent(100.0, 100.0, t0_ + 0.3, 7), context(now=t0_ + 0.3)).target_id
    check("pop-up: a figure with no pose just leaves (no fall, no span, shots after are false starts)",
          m_._fall is None and m_._fall_span is None and _late == "false-start")
    _gap_saved = AdvancingTarget.GAP_S
    AdvancingTarget.GAP_S = (0.5, 0.5)       # shorter than the fall, so only the hold-back can satisfy this
    try:
        m_, t0_ = _down(AdvancingTarget)
    finally:
        AdvancingTarget.GAP_S = _gap_saved
    a_ = m_._actors[0]
    check("advancing: the next figure is held back until the fall ends (a 0.5 s gap becomes the 1 s fall) "
          "and not otherwise delayed",
          abs(a_.live_at - (t0_ + _FALL)) < 0.05 and a_.phase == "gap", f"{a_.live_at - t0_:.3f}s")
    check("advancing: the pose is frozen where the figure was shot (feet on the same line)",
          a_.fall is not None and abs((a_.fall.centre[1] + a_.fall.height / 2) -
                                      (a_.state(t0_)[0][1] + a_.state(t0_)[1] / 2)) < 1.0)
    with _OnFixture():
        m_, t0_ = _down(AdvancingTarget, name=NOPOSE)
    check("advancing: a figure with no pose leaves at once (no fall, no span)",
          m_._actors[0].fall is None and m_._actors[0].fall_span is None)
    check("the second the fall takes is the second he asked for", _FALL == 1.0)

    _sh.rmtree(_fix, ignore_errors=True)
    passed = all(checks)
    print("\n" + (f"ALL {len(checks)} CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
