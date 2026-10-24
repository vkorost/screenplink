"""Shared by `test_modes.py` and `test_figures.py`: the figure drills' test helpers.

Not a test. Offscreen Qt, no camera, no display.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2                                                         # noqa: E402
import numpy as np                                                 # noqa: E402
from PySide6.QtCore import QRect                                   # noqa: E402
from PySide6.QtGui import QImage as _QI3, QPainter as _QP3         # noqa: E402

import lsa.modes as modes                                          # noqa: E402
from lsa.display.scene import RenderContext                        # noqa: E402
from lsa.display.session import SessionStats                       # noqa: E402

SW, SH = 3840, 2160


def context(now: float = 0.0, dt: float = 1 / 60.0, playable: QRect | None = None):
    return RenderContext(now=now, dt=dt, rect=QRect(0, 0, SW, SH),
                         stats=SessionStats(), playable=playable)


FIGURE_MODES = ("image", "advancing", "popup")


def drive_to_live(m, t_start: float, step: float = 1 / 60.0, limit: float = 12.0):
    """Tick a figure mode in virtual time until its figure is on screen."""
    t = t_start
    while t < t_start + limit:
        m.advance(context(now=t))
        if m.name == "advancing" and m._actors[0].phase == "live":
            return t
        if m.name == "popup" and m.state == "up":
            return t
        t += step
    raise AssertionError(f"{m.name}: no figure appeared within {limit}s")


def zone_point(m):
    """Screen (x, y) of the innermost ring's centre of the figure now up."""
    if m.name == "image":
        fig, centre, h = m._fig, m._centre, m._height
    elif m.name == "popup":
        fig, centre, h = m._fig, m._centre, m._height
    else:
        a = m._actors[0]
        return None
    inn = fig.innermost(centre, h)
    return (inn[1], inn[2])


def figure_hit_case(cls_):
    """(mode, time of the hit, hit point) for a figure mode, on a SCORED figure."""
    m = cls_()
    m._rng.seed(11)
    m.start(context(now=40.0), figure="thug01")
    if cls_.name == "image":
        pt = zone_point(m)
        t_hit = 50.0
    elif cls_.name == "popup":
        t = drive_to_live(m, 40.0)
        pt = zone_point(m)
        t_hit = m.shown_at + 0.3
    else:
        t = drive_to_live(m, 40.0)
        a = m._actors[0]
        t_hit = a.t0 + a.t_a + 0.1                 # in the near hold
        centre, h = a.state(t_hit)
        inn = a.fig.innermost(centre, h)
        pt = (inn[1], inn[2])
    out = m.shot(modes.ShotEvent(pt[0], pt[1], t_hit, 1), context(now=t_hit))
    assert out.hit, f"{cls_.name}: setup shot at the ring centre did not hit"
    return m, t_hit, pt


def figure_states(m, cls_):
    """[(label, ctx)] covering every state a figure mode can be in."""
    states = []
    if cls_.name == "image":
        m.start(context(now=40.0), figure="thug01")
        states.append(("live", context(now=40.1)))
        pt = zone_point(m)
        m.shot(modes.ShotEvent(pt[0], pt[1], 50.0, 1), context(now=50.0))
        states.append(("after hit", context(now=50.05)))
    elif cls_.name == "popup":
        m.start(context(now=40.0), figure="thug01")
        m.shot(modes.ShotEvent(300.0, 300.0, 40.1, 1), context(now=40.1))     # too early
        states.append(("waiting, false start", context(now=40.15)))
        drive_to_live(m, 40.2)
        states.append(("first frame", context(now=m.shown_at)))
        pt = zone_point(m)
        m.shot(modes.ShotEvent(pt[0], pt[1], m.shown_at + 0.3, 2),
               context(now=m.shown_at + 0.3))
        states.append(("after hit", context(now=m.shown_at + 0.31)))
    else:
        m.start(context(now=40.0), figure="thug01")
        drive_to_live(m, 40.0)
        a = m._actors[0]
        states.append(("first frame", context(now=a.t0)))
        states.append(("mid approach", context(now=a.t0 + a.t_a / 2)))
        # a hit WHILE it walks: the mark rides the figure through the later states
        t_walk = a.t0 + 0.3 * a.t_a
        c_w, h_w = a.state(t_walk)
        i_w = a.fig.innermost(c_w, h_w)
        m.shot(modes.ShotEvent(i_w[1], i_w[2], t_walk, 0), context(now=t_walk))
        states.append(("approach, hit riding the figure", context(now=t_walk + 0.3)))
        states.append(("approach, flash over, hole riding", context(now=t_walk + 1.0)))
        t_hold = a.t0 + a.t_a + 0.1
        centre, h = a.state(t_hold)
        inn = a.fig.innermost(centre, h)
        m.shot(modes.ShotEvent(inn[1], inn[2], t_hold, 1), context(now=t_hold))
        states.append(("holding, after hit", context(now=t_hold + 0.05)))
        states.append(("mid retreat", context(now=a.t0 + a.t_a + a.t_h + a.t_r / 2)))
        states.append(("far hold", context(now=a.t0 + a.duration - 0.1)))
    return states


def full(paint_fn, c_):
    """(max red excess, fraction of the screen above luminance 60, pixels)
    over EVERY pixel of a 3840x2160 render on the target background."""
    img = _QI3(SW, SH, _QI3.Format_RGB32)
    img.fill(0xFF0A0A0A)
    pnt = _QP3(img)
    paint_fn(pnt, c_)
    pnt.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(SH, img.bytesPerLine())
    a = a[:, :SW * 4].reshape(SH, SW, 4)
    # uint8 arithmetic in cv2/numpy rather than int32/float64 copies of a
    # 33 MB image: this ran ~200 times and was most of the suite's time.
    # BGR -> grey uses 0.299/0.587/0.114, the same weights as before.
    b_, g_, r_ = a[..., 0], a[..., 1], a[..., 2]
    red = int(cv2.subtract(r_, cv2.max(g_, b_)).max())
    lum = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_BGR2GRAY)
    # a COPY: `a` views the QImage's own buffer, which dies with this
    # function, and a caller that compares two images would read freed memory
    return red, float((lum > 60).mean()), a.copy()
