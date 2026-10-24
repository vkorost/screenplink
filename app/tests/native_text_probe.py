"""Render every piece of target-screen text through the NATIVE Qt platform and
print the worst red excess, as one machine-readable line.

Run as a SUBPROCESS by `test_ui_smoke`, and never with QT_QPA_PLATFORM set:
the point is to see what a real window sees. The unit suites run offscreen,
and the offscreen platform has no sub-pixel text at all -- it measured red
excess 0 on text that, through the native Windows platform, measured 49-86 on
its glyph edges (ClearType writes coloured fringes). Every palette check passed
while the camera read that text as laser hits whenever a layer appeared
(2026-10-17, "when I press zero I immediately hear a few fake shots").

No window is opened and nothing is shown: the platform plugin is loaded and the
text is painted into a QImage.

Output:  NATIVE_TEXT platform=<name> worst=<int> at=<x>,<y> layer=<name>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# The point of the probe: whatever lsa/_env.py chooses by default.
os.environ.pop("QT_QPA_PLATFORM", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lsa  # noqa: E402,F401  (applies the default platform / font engine)
import numpy as np  # noqa: E402
from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from lsa.display import layers as L  # noqa: E402
from lsa.display.scene import Scene  # noqa: E402
from lsa.modes.base import ModeHost as _MH0  # noqa: E402
_MH0.HOLD_S = 0.0     # the probe paints the screens AFTER the close; the hold is not under test


W, H = 1920, 1080


def _redx(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB32)
    b = np.frombuffer(img.constBits(), np.uint8).reshape(
        H, img.bytesPerLine() // 4, 4)[:, :W].astype(int)
    return b[..., 2] - np.maximum(b[..., 1], b[..., 0])


def main() -> int:
    app = QApplication([])
    rect = QRect(0, 0, W, H)
    worst, where, which = 0, (0, 0), "-"

    def measure(name: str, scene: Scene, expect_ink: bool = False) -> None:
        nonlocal worst, where, which
        img = QImage(W, H, QImage.Format_RGB32)
        img.fill(QColor("#0a0a0a"))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        scene.paint(p, rect)
        p.end()
        e = _redx(img)
        if expect_ink:
            # A probe that painted nothing proves nothing (RULES C1/C7): the
            # state must have put visible grey on the screen, or it reports
            # a failure of its own instead of a clean 0.
            g = np.frombuffer(img.convertToFormat(QImage.Format_RGB32).constBits(),
                              np.uint8).reshape(H, -1, 4)[:, :W, 1]
            if int((g > 30).sum()) < 200 and worst < 9999:
                worst, where, which = 9999, (0, 0), f"{name} (painted nothing)"
        if int(e.max()) > worst:
            y, x = np.unravel_index(int(e.argmax()), e.shape)
            worst, where, which = int(e.max()), (int(x), int(y)), name

    def fresh(layer: L.Layer, setup=None) -> Scene:
        s = Scene()
        s.add(L.SolidBackground())
        s.add(layer)
        s.diagnostics["text"] = "detect 7.2ms fps 55\nfloor 4 bar 8"
        s.stats.shots = 12
        s.stats.hits = 7
        if setup:
            setup(layer, s)
        return s

    measure("stats", fresh(L.StatsHud()))
    measure("help", fresh(L.KeyHelp()))
    measure("bglabel", fresh(L.BackgroundLabel(L.SolidBackground())))
    for kind in L.Prompt.KINDS:
        measure(f"prompt[{kind}]", fresh(
            L.Prompt(), lambda l, s, k=kind: l.say("SHOOT THE RING", "detail", k, "3 of 9")))
    for note_kind in ("good", "info", "warn"):
        measure(f"signal[{note_kind}]", fresh(
            L.SignalMeter(), lambda l, s, k=note_kind: l.report(
                120, 14, live=True, seen_ago=0.5, shot_ago=1.0,
                counts=(("rej", 3),), note="seeing the laser", note_kind=k,
                now=time.perf_counter())))
    for kind in L.Alert.KINDS:
        measure(f"alert[{kind}]", fresh(
            L.Alert(), lambda l, s, k=kind: l.show("ALERT", "some detail text", k, dim=True)))

    # The one-second "ONE PLAYER ONLY" note (no detail, no veil): a new string at
    # a new size, drawn while shooting is possible.
    measure("alert[one player only]", fresh(
        L.Alert(), lambda l, s: l.show("ONE PLAYER ONLY", "", "info", dim=False)))

    def hits(l, s):
        for i in range(5):
            l.add(0.2 + i * .1, 0.3 + i * .1, i)

    measure("hits", fresh(L.CameraSpaceHits(visible=True), hits))
    for verdict in L.VerdictBoard.STYLES:
        def board(l, s, v=verdict):
            l.verdict, l.summary = v, "11 of 12 test shots came back"
            l.findings = [("headline text", "do this action now", "measured 3.1x")]
            l.more, l.until, l.visible = 0, 0.0, True
        measure(f"verdict[{verdict}]", fresh(L.VerdictBoard(), board))

    # THE VERSUS SCORELINE: text the host draws from the FULL screen. It is only
    # drawn in two-player mode, so a probe that never builds one would never see
    # its glyph edges (the offscreen suites read 0 on text like this).
    from lsa.modes import ModeHost, StaticTargets

    def versus_board(l, s):
        l.set_mode(StaticTargets(), s._context(0.0, 0.0, rect), players=2)
        l.tallies[0].points, l.tallies[0].shots = 128, 31
        l.tallies[1].points, l.tallies[1].shots = 97, 40

    measure("versus scoreline", fresh(ModeHost(), versus_board))

    # THE MENU (2026-10-19): Single Player / Multiplayer with a BOLD first letter in a
    # slightly brighter grey. A bold glyph edge is where ClearType's fringes were
    # worst, and the menu had never been through this probe at all.
    from lsa.modes.menu import Menu

    def menu_state(players):
        def setup(l, s):
            s.players = players
            l.start(s._context(0.0, 0.0, rect))
        return setup

    for players in (1, 2):
        measure(f"menu players={players}", fresh(Menu(), menu_state(players)), expect_ink=True)

    # KEYBOARD SHORTCUTS (2026-10-21): the largest area of text the app draws, in
    # bold and regular, on the screen the camera watches -- the worst case for
    # ClearType's coloured fringes, and invisible to the offscreen suites.
    from lsa.modes.shortcuts import Shortcuts

    measure("shortcuts", fresh(Shortcuts(), lambda l, s: l.start(s._context(0.0, 0.0, rect))),
            expect_ink=True)

    # SLICE 3: every state that puts a NEW string on the shooting screen -- the
    # two-line scoreline cells, each side's note ("round 2, sides swapped",
    # "LEFT: done, 20 of 20") and the verdict screen in both outcomes plus
    # the match verdict. Built by FIRING shots through the host, as the app does,
    # so what is painted is what a round produces and not a hand-set tally.
    from lsa.modes import ShotEvent
    from lsa.modes import versus as _versus

    def _fire(l, s, side, n, on_target):
        ctx = s._context(0.0, 0.0, rect)
        m = l.mode if side == 0 else l.mode2
        for i in range(n):
            x, y = (m.aim_point(l._side_ctx(ctx, side)) if on_target
                    else ((40.0 if side == 0 else W - 40.0), H * 0.9))
            l.shot(ShotEvent(x, y, 1.0 + i * 0.01, i + 1), ctx)

    def _start(l, s):
        l.set_mode(StaticTargets(), s._context(0.0, 0.0, rect), players=2)
        l.mic_ready = True
        l.note_heard_unseen(3)

    N = __import__('lsa.settings', fromlist=['x']).DEFAULT_MAGAZINE

    def live(l, s):
        _start(l, s)
        _fire(l, s, 0, 5, True)
        _fire(l, s, 1, 3, False)

    def one_done(l, s):
        _start(l, s)
        _fire(l, s, 0, N, True)
        _fire(l, s, 1, 2, False)

    def over_win(l, s):
        _start(l, s)
        _fire(l, s, 0, N, True)
        _fire(l, s, 1, N, False)

    def over_tie(l, s):
        _start(l, s)
        _fire(l, s, 0, N, False)
        _fire(l, s, 1, N, False)

    def round2(l, s):
        over_win(l, s)
        l.restart(s._context(0.0, 0.0, rect))
        _fire(l, s, 0, 4, False)
        _fire(l, s, 1, 6, True)

    def round2_over(l, s):
        over_win(l, s)
        l.restart(s._context(0.0, 0.0, rect))
        _fire(l, s, 0, N, False)
        _fire(l, s, 1, N, True)

    for name, fn in (("versus live", live), ("versus one side done", one_done),
                     ("versus verdict, winner", over_win),
                     ("versus verdict, tied", over_tie),
                     ("versus round 2, swapped", round2),
                     ("versus match verdict", round2_over)):
        measure(name, fresh(ModeHost(), fn))

    # REACTION IN VERSUS (2026-10-19): the first drill whose two halves share a
    # draw. It puts three NEW strings on the shooting screen -- a hit time in
    # ms, "TOO EARLY" and the new "TOO SLOW" -- plus its "wait for it" line and
    # the opening instruction, once per half. Built through the host so what is
    # painted is what a round produces. Clocks are placed relative to the real
    # clock because Scene.paint stamps its own `now` from perf_counter and a
    # flash older than 0.45 s is not drawn at all.
    from lsa.modes.reaction import ReactionDrill

    def _rx_ctx(s, n):
        return s._context(n, 1 / 60.0, rect)

    def _rx_live(l, s, t0):
        l.set_mode(ReactionDrill(), _rx_ctx(s, t0), players=2)
        l.mode._reveal.appear_at = t0 + 0.05
        for i in range(6):
            l.advance(_rx_ctx(s, t0 + 0.05 + i / 60.0))

    def rx_waiting(l, s):
        l.set_mode(ReactionDrill(), _rx_ctx(s, time.perf_counter() - 1.0), players=2)

    def rx_hits(l, s):
        t0 = time.perf_counter() - 1.0
        _rx_live(l, s, t0)
        now = time.perf_counter()
        for m, side in ((l.mode, 0), (l.mode2, 1)):
            c = l._side_ctx(_rx_ctx(s, now), side)
            l.shot(ShotEvent(m.x, m.y, now, side + 1), _rx_ctx(s, now))
            l.shot(ShotEvent(m.x + 5, m.y, now, side + 3), _rx_ctx(s, now))   # false start

    def rx_slow(l, s):
        t0 = time.perf_counter() - 3.12
        _rx_live(l, s, t0)
        l.advance(_rx_ctx(s, t0 + 3.09))                    # both expire: TOO SLOW

    def rx_late(l, s):
        rx_slow(l, s)
        now = time.perf_counter()
        for m, side in ((l.mode, 0), (l.mode2, 1)):
            l.shot(ShotEvent(m.x, m.y, now, side + 1), _rx_ctx(s, now))   # forgiven: TOO LATE

    for name, fn in (("reaction versus waiting", rx_waiting),
                     ("reaction versus hits and false starts", rx_hits),
                     ("reaction versus too slow", rx_slow),
                     ("reaction versus too late", rx_late)):
        measure(name, fresh(ModeHost(), fn), expect_ink=True)

    # THE MAGAZINE (2026-10-20): new strings on the shooting screen -- the
    # remaining-count numeral beside the bullets, the single-player end screen
    # ("17 of 17 shots", "MAGAZINE EMPTY", "START NEXT") and the small dry-pull
    # line at the bottom of both end screens. Single player has not been in this
    # probe before. Painted through the host so what is measured is what a round
    # produces; the versus states above already carry the two strips' numerals.
    def sp_full(l, s):
        l.set_mode(StaticTargets(), s._context(0.0, 0.0, rect), players=1)

    def sp_part(l, s):
        sp_full(l, s)
        ctx = s._context(0.0, 0.0, rect)
        for i in range(6):
            l.shot(ShotEvent(40.0, H * 0.9, 1.0 + i * 0.01, i + 1), ctx)

    def sp_over(l, s):
        sp_full(l, s)
        ctx = s._context(0.0, 0.0, rect)
        for i in range(N):
            l.shot(ShotEvent(40.0, H * 0.9, 1.0 + i * 0.01, i + 1, t_end=1.05 + i * 0.01), ctx)
        l.mic_ready = True
        l.note_heard_unseen(2)
        l.dry_camera, l.dry_corroborated = 3, 2

    def sp_over_uncalibrated(l, s):
        sp_over(l, s)
        l.mic_ready = False

    for name, fn in (("single player, full magazine", sp_full),
                     ("single player, partly spent", sp_part),
                     ("single player end screen, with dry pulls", sp_over),
                     ("single player end screen, microphone uncalibrated", sp_over_uncalibrated)):
        measure(name, fresh(ModeHost(), fn), expect_ink=True)

    # THE MENU (2026-10-20): "One Player" / "Two Players" with a bold O and T, and the
    # label inset moved. Bold glyph edges are where ClearType fringes measured 86 on his
    # real panel, so the menu is painted through the native platform here too.
    from lsa.modes.menu import Menu

    def menu_screen(l, s):
        l.set_mode(Menu(), s._context(0.0, 0.0, rect), players=1)

    measure("menu: One Player / Two Players", fresh(ModeHost(), menu_screen), expect_ink=True)

    print(f"NATIVE_TEXT platform={app.platformName()} worst={worst} "
          f"at={where[0]},{where[1]} layer={which}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
