"""The shoot-to-select menu (mode `menu`, key E) and its tile pictures.

Synthetic shots, offscreen Qt, no camera, no display. Plain asserts with numbered,
named checks and a count at the end, like `test_modes.py` -- these suites do not
use pytest.

What is worth testing about a menu is what would make it feel broken or unsafe
rather than what would make it crash: a mode with no tile (or a tile for a mode
that no longer exists), tiles sized in pixels that are half the size on the 75"
TV, a tile that is shot and starts something from inside the call that is still
executing, text that runs off its own tile, a stroke the camera could mistake for
a laser, a baked brightness that went stale when he added a darker figure, and a
static screen that repaints a 4K surface sixty times a second.

Every check here is meant to be able to fail. The ones I could not prove it for
by construction are named in the report that came with this file.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import ast
import time
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import cv2                                                         # noqa: E402
import numpy as np                                                 # noqa: E402
from PySide6.QtCore import QRect, QRectF, Qt                       # noqa: E402
from PySide6.QtGui import QImage, QPainter, QPixmap, QRegion      # noqa: E402
from PySide6.QtWidgets import QApplication                         # noqa: E402

import lsa.modes as modes                                          # noqa: E402
from lsa.settings import DEFAULT_MAGAZINE as _MAG18                         # noqa: E402
# THE HOLD (2026-10-20): a side that has just fired its last round stays painted for
# `ModeHost.HOLD_S` before the round closes. The checks below are about what happens AT
# the close (verdicts, swaps, quiet halves), not about the hold, so they run with it off;
# section 39 and the hold checks in test_ui_smoke use the REAL value, kept here.
import numpy as _np_menu                                            # noqa: E402
from lsa.modes.base import ModeHost as _MH0                          # noqa: E402
REAL_HOLD_S = _MH0.HOLD_S
_MH0.HOLD_S = 0.0

from lsa.artwork import ceiling, tiles                             # noqa: E402
from lsa.display import layers                                     # noqa: E402
from lsa.display.scene import RenderContext                        # noqa: E402
from lsa.display.session import SessionStats                       # noqa: E402
from lsa.modes import base                                         # noqa: E402
from lsa.modes import versus as versus_mod                          # noqa: E402
from lsa.modes import menu as menu_mod                             # noqa: E402
from lsa.modes.menu import Menu, fit_text, tile_layout             # noqa: E402

from figure_helpers import SW, SH, context                         # noqa: E402


def ctx_for(w: int, h: int, now: float = 0.0, playable=None) -> RenderContext:
    return RenderContext(now=now, dt=1 / 60.0, rect=QRect(0, 0, w, h),
                         stats=SessionStats(), playable=playable)


def render(paint_fn, ctx: RenderContext) -> np.ndarray:
    """The paint on the target background (luminance 10), as BGRA, a COPY.
    The image is filled BEFORE the painter is attached: `fill` does nothing
    while a QPainter is attached to that QImage (the lit-area harness once read
    8.6% on an unchanged clip because of exactly that)."""
    w, h = ctx.width, ctx.height
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(0xFF0A0A0A)
    p = QPainter(img)
    paint_fn(p, ctx)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())
    return a[:, :w * 4].reshape(h, w, 4).copy()


def red_excess(a: np.ndarray) -> int:
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    return int(cv2.subtract(r, cv2.max(g, b)).max())


def luminance(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_BGR2GRAY)


@contextmanager
def extra_mode(name: str, key, title: str):
    """Register a throwaway mode for the length of a block, and ALWAYS remove it,
    so the registry the rest of this run sees is the real one."""
    cls = type("Extra_" + name, (Mode,), {
        "name": name, "title": title, "summary": "test only",
        "paint": lambda self, p, ctx: None,
        "start": lambda self, ctx: None})
    base.register(key=key)(cls)
    try:
        yield cls
    finally:
        base._REGISTRY.pop(name, None)


Mode = base.Mode


def main() -> int:
    app = QApplication.instance() or QApplication([])
    checks: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    c4 = context(now=100.0)                     # 3840 x 2160
    c1080 = ctx_for(1920, 1080, now=100.0)
    # 2026-10-21: a SCREEN is registered like a drill but is not one. The Keyboard
    # Shortcuts page (K) gets no tile -- it has no picture and the menu is the one
    # screen every startup routine photographs, so a twelfth tile costs measurement
    # margin. The complement is asserted below, so this exemption cannot quietly
    # swallow a real drill that stopped getting a tile.
    real_modes = [(c, k) for c, k in base.registered()
                  if c is not Menu and not getattr(c, "is_screen", False)]
    screen_modes = [(c, k) for c, k in base.registered()
                    if c is not Menu and getattr(c, "is_screen", False)]

    # ------------------------------------------------------------------
    print("\n1. the key: M, registered, unique, and not a key anything else answers")
    check("the menu is registered on Qt.Key_M and by_key finds it",
          base.by_key(Qt.Key_M) is Menu)
    check("no other registered mode is on M, and no mode shares a key",
          [c for c, k in base.registered() if k == Qt.Key_M] == [Menu]
          and len([k for _c, k in base.registered() if k is not None])
          == len({k for _c, k in base.registered() if k is not None}))
    # E was the menu's key until he asked for M by name (2026-10-18). The pane's
    # corner cycle moved to J in the same change: a layer claiming a key shadows
    # the app silently, so M must be answered in exactly one place.
    check("E is free again, and no mode answers it",
          base.by_key(Qt.Key_E) is None)
    hits = []
    _n_scanned = 0
    for path in sorted((ROOT / "src").rglob("*.py")):
        _n_scanned += 1
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"Key_M\b", line) and not line.lstrip().startswith("#"):
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{i}")
    # CHANGED 2026-10-20: the player-count keys are O and T, so M is the menu key and
    # NOTHING ELSE again: one registration, in menu.py, and no other file names it.
    check("Qt.Key_M appears in src/ only in menu.py, once: the registration (the menu does not claim it)",
          len(hits) == 1 and all(h_.startswith("src/lsa/modes/menu.py") for h_ in hits),
          f"{hits} (scanned {_n_scanned} files under {ROOT})")

    # ------------------------------------------------------------------
    print("\n2. one tile per registered mode, found from the registry, not listed by name")
    m = Menu()
    m.start(c4)
    slots = m.slots(c4)
    mode_slots = [s for s in slots if s.cls is not None]
    chrome_slots = [s for s in slots if s.cls is None]
    check("every registered mode except the menu itself has exactly one tile",
          sorted(s.cls.name for s in mode_slots) == sorted(c.name for c, _k in real_modes)
          and len(mode_slots) == len(real_modes),
          f"{len(mode_slots)} tiles for {len(real_modes)} modes")
    check("the menu does not offer a tile for itself", all(s.cls is not Menu for s in slots))
    check("a SCREEN gets no tile, and there is at least one to prove the rule is live",
          len(screen_modes) >= 1
          and all(s.cls not in [c for c, _k in screen_modes] for s in mode_slots),
          str([c.name for c, _k in screen_modes]))
    glyphs = [s.glyph for s in mode_slots if s.glyph]   # drills only: chrome glyphs ("1P", "Esc") are not keys in this order
    check("tiles run digits, then letters (glyphs ascending); keyless modes, if any, last",
          glyphs == sorted(glyphs)
          and all(s.glyph == "" for s in mode_slots[len(glyphs):]),
          "".join(glyphs))
    check("each tile's glyph is the key its mode is registered on",
          all(glyphs_ok for glyphs_ok in
              [menu_mod.glyph_of_key(dict((c, k) for c, k in real_modes)[s.cls]) == s.glyph
               for s in mode_slots]))
    # THE ORDER (2026-10-19): Single Player and Multiplayer FIRST, then every drill,
    # then EXIT last.
    check("the chrome tiles are exactly menu_mod.CHROME, the two player-count tiles FIRST, EXIT LAST, "
          "every drill between them, and none has a mode behind it",
          [s.stem for s in chrome_slots] == [c.stem for c in menu_mod.CHROME]
          and slots[:2] == chrome_slots[:2] and slots[2:-1] == mode_slots
          and slots[-1:] == chrome_slots[-1:] and len(chrome_slots) >= 1
          and [s.players for s in slots[:2]] == [1, 2] and slots[-1].stem == "exit"
          and all(s.cls is None for s in chrome_slots)
          # A chrome tile may now carry a key glyph: EXIT shows "Esc",
          # because Escape does the same thing (his instruction, 2026-10-18).
          # What makes it chrome is having no MODE behind it, not having no
          # key -- the old check conflated the two.
          and all(s.glyph == c.glyph
                  for s, c in zip(chrome_slots, menu_mod.CHROME)),
          str([(s.stem, s.title) for s in chrome_slots]))
    check("EXIT is a chrome tile whose picture is found by its stem (menu/exit.png), "
          "with no filename special-cased in the loader",
          any(s.stem == "exit" for s in chrome_slots)
          and all(Menu()._tile_for(s) is not None for s in chrome_slots))
    check("EXIT shows Esc as its key, because Escape closes the app too",
          [c.glyph for c in menu_mod.CHROME if c.stem == "exit"] == ["Esc"])
    n_before = len(slots)
    with extra_mode("zz_nopicture", Qt.Key_J, "No picture drill") as NoPic:
        with_extra = Menu().slots(c4)
        got = [s for s in with_extra if s.cls is NoPic]
        check("a mode registered AFTER the menu was built gets a tile (the list is not cached "
              "or hard-coded)", len(with_extra) == n_before + 1 and len(got) == 1)
        check("a mode with no picture file still has a tile, with its key glyph",
              bool(got) and got[0].glyph == "J" and tiles.for_glyph("J") is None)
        pic = menu_mod.Menu()
        a_np = render(pic.paint, c4)
        r_ = got[0].picture
        region = luminance(a_np)[int(r_.top()):int(r_.bottom()), int(r_.left()):int(r_.right())]
        check("...and that tile is not blank: its glyph is drawn in the picture area",
              int(region.max()) > 30, f"brightest pixel {int(region.max())}")
    with extra_mode("zz_keyless", None, "Keyless drill") as KeyLess:
        kl = [s for s in Menu().slots(c4) if s.cls is KeyLess]
        drills = [s for s in Menu().slots(c4) if s.cls is not None]
        check("a mode with NO key still gets a tile (shootable), drawn after the keyed drills "
              "(and before the chrome), no glyph",
              len(kl) == 1 and kl[0].glyph == "" and drills[-1].cls is KeyLess)
        spy_ok = render(Menu().paint, c4)       # must not raise with a glyph-less tile
        check("...and painting it does not raise", spy_ok is not None)
    check("after the throwaway modes are removed the tiles are the real ones again",
          len(Menu().slots(c4)) == n_before)

    # ------------------------------------------------------------------
    print("\n3. the pictures: measured against the ceiling at load, never trusted")
    want = ceiling.figure_level()
    real_files = tiles.files()
    check("the menu folder holds tile pictures", len(real_files) >= 1, f"{len(real_files)} png")
    bad = []
    over = []
    for f in real_files:
        t = tiles.load_tile(f)
        if t is None:
            bad.append(f.name)
            continue
        if ceiling.lit_p99(t.ink) > want + 0.01 or t.gain > 1.0:
            over.append(f"{f.name} p99 {ceiling.lit_p99(t.ink):.1f}")
    check("every picture loads, is never scaled UP, and its lit p99 is at or under the ceiling",
          not bad and not over, f"unreadable {bad}, over {over}, ceiling {want}")
    check("the one named `menu` (the MENU corner picture) loads by stem",
          tiles.named("menu") is not None)
    check("a corrupt or missing picture is reported, not silently treated as 'no picture'",
          tiles.problems() == {}, str(tiles.problems()))
    tmp = Path(tempfile.mkdtemp())
    try:
        def save(name: str, value: int, size: int = 64) -> Path:
            a = np.zeros((size, size), np.uint8)
            a[8:size - 8, 8:size - 8] = value
            p_ = tmp / name
            cv2.imwrite(str(p_), a)
            return p_
        bright = save("Q_bright.png", 200)
        dim = save("W_dim.png", 35)
        tiles.forget()
        tb, td = tiles.load_tile(bright), tiles.load_tile(dim)
        check("a tile baked BRIGHTER than the ceiling is brought down to it at load",
              tb is not None and tb.baked_p99 > want + 50 and tb.gain < 1.0
              and ceiling.lit_p99(tb.ink) <= want + 0.01,
              f"baked {tb.baked_p99 if tb else None}, gain {tb.gain if tb else None}, "
              f"p99 after {ceiling.lit_p99(tb.ink) if tb else None}, ceiling {want}")
        src_dim = cv2.imread(str(dim), cv2.IMREAD_GRAYSCALE)
        check("a tile already under the ceiling is NOT raised (gain 1.0, pixels identical)",
              td is not None and td.gain == 1.0 and np.array_equal(td.ink, src_dim))
        check("the corrected pixels are read-only (a cached array nobody may edit)",
              tb is not None and not tb.ink.flags.writeable)
        # THE POINT OF RE-CHECKING: the ceiling moved 71 -> 52 -> 71 in one morning.
        real_level = ceiling.figure_level
        ceiling.figure_level = lambda folder=None: 45.0
        try:
            t_low = tiles.load_tile(bright)
        finally:
            ceiling.figure_level = real_level
        t_back = tiles.load_tile(bright)
        check("when the ceiling MOVES the same file is corrected to the new one, not served stale "
              "from the cache",
              t_low is not None and t_low.ceiling == 45.0
              and ceiling.lit_p99(t_low.ink) <= 45.01 and t_low is not tb,
              f"p99 at ceiling 45: {ceiling.lit_p99(t_low.ink) if t_low else None}")
        check("...and moving it back restores the correction against the real one",
              t_back is not None and t_back.ceiling == want
              and abs(ceiling.lit_p99(t_back.ink) - ceiling.lit_p99(tb.ink)) < 0.01)
        (tmp / "X_corrupt.png").write_bytes(b"this is not a png")
        check("a corrupt file gives None and is listed in problems()",
              tiles.load_tile(tmp / "X_corrupt.png") is None
              and str(tmp / "X_corrupt.png") in tiles.problems())
        check("a missing file gives None and is NOT a problem (a drill may simply have no picture)",
              tiles.load_tile(tmp / "nope.png") is None
              and str(tmp / "nope.png") not in tiles.problems())
        check("the glyph is the text before the first underscore; `menu` has none",
              tiles.glyph_of("7_popup.png") == "7" and tiles.glyph_of("A_animated.png") == "A"
              and tiles.glyph_of("menu.png") == "" and tiles.for_glyph("") is None)
        check("a tile is found by its glyph, case-insensitively, in a folder",
              tiles.for_glyph("q", tmp) is not None and tiles.for_glyph("Q", tmp) is not None
              and tiles.for_glyph("Z", tmp) is None)
        tiles.forget()
        old_max = tiles.MAX_TILES
        tiles.MAX_TILES = 2
        try:
            for nm, v in (("a_.png", 60), ("b_.png", 61), ("c_.png", 62)):
                tiles.load_tile(save(nm, v))
            bounded = len(tiles._CACHE) <= 2
        finally:
            tiles.MAX_TILES = old_max
        check("the tile cache is bounded (past MAX_TILES it is dropped and refilled)", bounded)
    finally:
        tiles.forget()
        shutil.rmtree(tmp, ignore_errors=True)
    # Asked of the DRILL tiles. A chrome tile now carries a glyph too (EXIT
    # shows "Esc"), but its picture is found by stem, so judging it by
    # `for_glyph` would be asking the wrong question of the wrong tile.
    pm_ok = all((tiles.for_glyph(s.glyph) is not None)
                == (Menu()._pixmap(s, 200, want) is not None)
                for s in slots if s.glyph and s.cls is not None)
    check("a tile has a pixmap exactly when its drill has a picture file", pm_ok)
    check("...and a chrome tile is found by its stem, not by its key glyph",
          all(Menu()._pixmap(s, 200, want) is not None
              for s in slots if s.cls is None and s.stem))

    # ------------------------------------------------------------------
    print("\n4. geometry is a fraction of the screen, the same at every size")
    def fractions(ctx):
        return [(s.rect.x() / ctx.width, s.rect.y() / ctx.height,
                 s.rect.width() / ctx.width, s.rect.height() / ctx.height)
                for s in Menu().slots(ctx)]
    f4k, f1080 = fractions(c4), fractions(c1080)
    f1440 = fractions(ctx_for(2560, 1440, now=100.0))
    worst = max(abs(a - b) for t4, t1 in zip(f4k, f1080) for a, b in zip(t4, t1))
    worst2 = max(abs(a - b) for t4, t1 in zip(f4k, f1440) for a, b in zip(t4, t1))
    check("the layout at 3840x2160 and at 1920x1080 is the same in fractional terms",
          len(f4k) == len(f1080) > 0 and worst < 1e-3, f"worst difference {worst:.2e}")
    check("...and at 2560x1440 (a size nobody typed in)", worst2 < 1e-3, f"worst {worst2:.2e}")
    check("the default tile is one SEVENTH of the screen width (1.176% lit measured; 1/6 was "
          "1.601%, leaving 0.40% for text)", abs(menu_mod.TILE_WIDTH_FRAC - 1.0 / 7.0) < 1e-12)
    check("a tile is that fraction of the screen width at both sizes, the measured light budget",
          all(abs(s.rect.width() / c.width - menu_mod.TILE_WIDTH_FRAC) < 1e-9
              for c in (c4, c1080) for s in Menu().slots(c)),
          f"{[round(s.rect.width()) for s in slots[:1]]} px at 4K")
    for label, ctx in (("4K", c4), ("1080p", c1080)):
        sl = Menu().slots(ctx)
        field = Mode.playfield(ctx)
        inside = all(field.left() - 0.5 <= s.rect.left() and s.rect.right() <= field.right() + 0.5
                     and field.top() - 0.5 <= s.rect.top() and s.rect.bottom() <= field.bottom() + 0.5
                     for s in sl)
        overlap = [(a.cls.name, b.cls.name) for i, a in enumerate(sl) for b in sl[i + 1:]
                   if a.rect.intersects(b.rect)]
        check(f"{label}: every tile is inside the playfield (clear of the top band and the legend)",
              inside)
        check(f"{label}: no two tiles overlap", not overlap, str(overlap))
    small = ctx_for(SW, SH, now=100.0, playable=QRect(SW // 3, SH // 3, SW // 3, SH // 3))
    sm = Menu().slots(small)
    field = Mode.playfield(small)
    check("a small playable region SHRINKS the tiles to fit it rather than overflowing",
          sm and sm[0].rect.width() < SW * menu_mod.TILE_WIDTH_FRAC - 1
          and all(field.left() - 0.5 <= s.rect.left() and s.rect.right() <= field.right() + 0.5
                  and s.rect.bottom() <= field.bottom() + 0.5 for s in sm),
          f"tile {sm[0].rect.width():.0f}px in a {field.width():.0f}x{field.height():.0f} field")
    check("an empty list of tiles lays out to nothing, not an error",
          tile_layout(QRectF(0, 0, 100, 100), 0, 100) == [])

    # ------------------------------------------------------------------
    print("\n5. drawn at red excess 0, under the ceiling, inside the light budget")
    CEIL = layers.CEILING
    m_idle = Menu()
    m_idle.start(c4)
    m_pend = Menu()
    m_pend.start(c4)
    pc0 = next(s_ for s_ in m_pend.slots(c4) if s_.cls is not None).picture.center()
    m_pend.shot(modes.ShotEvent(pc0.x(), pc0.y(), 100.0, 1), c4)
    states = [("4K idle", m_idle, c4), ("4K a tile chosen", m_pend, c4),
              ("1080p idle", m_idle, c1080)]
    for label, mm, ctx in states:
        a = render(mm.paint, ctx)
        lum = luminance(a)
        lit = float((lum > 60).mean())
        check(f"{label}: every pixel is neutral grey (red excess exactly 0)",
              red_excess(a) == 0, f"worst red excess {red_excess(a)}")
        check(f"{label}: nothing is brighter than the ceiling ({int(lum.max())} vs {CEIL})",
              int(lum.max()) <= CEIL)
        check(f"{label}: lit area (>60) {lit * 100:.3f}% is inside the 2% budget (RULES B1b)",
              0.002 < lit <= 0.02, f"{lit * 100:.3f}%")
    with extra_mode("zz_nopicture", Qt.Key_J, "No picture drill"):
        a = render(Menu().paint, c4)
        check("with a picture-less tile on screen the whole menu is still red excess 0",
              red_excess(a) == 0)
    a_cal = render(Menu().paint, RenderContext(now=100.0, dt=0.0, rect=QRect(0, 0, SW, SH),
                                               stats=SessionStats(), calibrating=True))
    check("while the grid is being shot the menu draws NOTHING (one ring to shoot at)",
          int(luminance(a_cal).max()) == 10)
    a_idle = render(m_idle.paint, c4)
    lum_i = luminance(a_idle)
    dark_tiles = [s.title for s in m_idle.slots(c4)
                  if int(lum_i[int(s.picture.top()):int(s.picture.bottom()),
                               int(s.picture.left()):int(s.picture.right())].max()) <= 30]
    check("every tile's picture area actually has ink in it (a paint that draws nothing fails)",
          not dark_tiles, str(dark_tiles))
    class BorderSpy(QPainter):
        """Records the pen colour of each drawRect: one per tile, in slot order."""
        rec: list = []

        def drawRect(self, *a):
            BorderSpy.rec.append(self.pen().color().red())
            return super().drawRect(*a)

    def borders(mode, ctx):
        img_ = QImage(ctx.width, ctx.height, QImage.Format_RGB32)
        img_.fill(0xFF0A0A0A)
        BorderSpy.rec = []
        sp_ = BorderSpy(img_)
        mode.paint(sp_, ctx)
        sp_.end()
        return list(BorderSpy.rec)

    idle_slots, pend_slots = m_idle.slots(c4), m_pend.slots(c4)
    b_idle, b_pend = borders(m_idle, c4), borders(m_pend, c4)
    check("an idle menu draws EVERY tile's border at the resting level (chrome has cls None and "
          "pending is None: `None is None` must not light EXIT)",
          len(b_idle) == len(idle_slots) and set(b_idle) == {layers.GREY_MISS.red()},
          str(b_idle))
    _chosen_ix = next(i_ for i_, s_ in enumerate(pend_slots) if s_.cls is not None)
    check("a chosen tile's border is the bright one, and the chrome tiles stay at rest",
          len(b_pend) == len(pend_slots) and b_pend[_chosen_ix] == layers.GREY_HI.red()
          and all(b == layers.GREY_MISS.red()
                  for b, s in zip(b_pend, pend_slots) if s.cls is None),
          str(b_pend))
    check("the captions' margin under the lit threshold (60) is stated in code and is not "
          "negative at this ceiling (58 vs 60 today; it moves with the artwork, see the docstring)",
          menu_mod.caption_margin() == menu_mod.LIT_LUMINANCE - layers.GREY_MID.red()
          and menu_mod.caption_margin() >= 0, f"margin {menu_mod.caption_margin()}")

    # ------------------------------------------------------------------
    print("\n6. a static menu asks for (almost) no repaints")
    mr = Menu()
    mr.start(context(now=100.0))
    asked = sum(1 for i in range(180) if mr.advance(context(now=100.0 + i / 60.0)))
    check("idle: at most 5 repaints requested in 180 ticks", asked <= 5,
          f"{asked} repaints in 180 ticks")
    pcr = next(s_ for s_ in mr.slots(c4) if s_.cls is not None).picture.center()
    mr.shot(modes.ShotEvent(pcr.x(), pcr.y(), 101.0, 1), context(now=101.0))
    check("recording a choice asks for a repaint (so the chosen border shows)... ",
          mr.animated is True)
    asked2 = sum(1 for i in range(180) if mr.advance(context(now=101.0 + i / 60.0)))
    check("...exactly one, and then it is idle again", asked2 == 1 and mr.animated is False,
          f"{asked2} repaints in the 180 ticks after")

    # ------------------------------------------------------------------
    print("\n7. a shot inside a tile chooses that mode; a shot outside every tile is a miss")
    ms = Menu()
    ms.start(c4)
    # The DRILL tiles (the player-count tiles lead the grid but are not drills).
    sl = [x for x in ms.slots(c4) if x.cls is not None]
    wrong = []
    for s in [x for x in sl if x.cls is not None]:
        ms.take_pending()
        c = s.picture.center()
        out = ms.shot(modes.ShotEvent(c.x(), c.y(), 100.0, 1), c4)
        if not (out.hit and ms.pending is s.cls and out.target_id == f"tile:{s.cls.name}"):
            wrong.append(s.cls.name)
    check("a shot at the centre of each tile's picture chooses exactly that tile's mode",
          not wrong, str(wrong))
    ms.take_pending()
    cc = sl[0].caption.center()
    out = ms.shot(modes.ShotEvent(cc.x(), cc.y(), 100.0, 2), c4)
    check("a shot on the caption (the name) counts as that tile too",
          out.hit and ms.pending is sl[0].cls)
    ms.take_pending()
    check("a tile hit draws no bullet hole (it would be left on the next screen) but still bangs",
          out.draw_hole is False and out.play_sound is True)
    r0 = sl[0].rect
    out_in = ms.shot(modes.ShotEvent(r0.left() + 0.5, r0.top() + 0.5, 100.0, 3), c4)
    chose_in = ms.take_pending() is sl[0].cls
    out_out = ms.shot(modes.ShotEvent(r0.left() - 0.5, r0.top() + r0.height() / 2, 100.0, 4), c4)
    check("the tile edge is exact: half a pixel inside hits, half a pixel outside misses",
          out_in.hit and chose_in and not out_out.hit and ms.pending is None)
    gap = ms.shot(modes.ShotEvent((sl[0].rect.right() + sl[1].rect.left()) / 2.0,
                                  sl[0].rect.center().y(), 100.0, 5), c4) \
        if abs(sl[0].rect.top() - sl[1].rect.top()) < 1 else None
    check("a shot in the gap between two tiles on a row is a miss",
          gap is not None and not gap.hit and ms.pending is None)
    below = [s for s in sl if s.rect.top() > sl[0].rect.bottom()]
    if below:
        vgap = ms.shot(modes.ShotEvent(sl[0].rect.center().x(),
                                       (sl[0].rect.bottom() + below[0].rect.top()) / 2.0,
                                       100.0, 6), c4)
        check("a shot in the gap between two rows is a miss", not vgap.hit and ms.pending is None)
    far = ms.shot(modes.ShotEvent(5.0, 5.0, 100.0, 7), c4)
    band = ms.shot(modes.ShotEvent(SW / 2.0, 10.0, 100.0, 8), c4)
    check("a shot in no tile (the corner, the top band) is a miss and chooses nothing",
          not far.hit and not band.hit and far.target_id is None and ms.pending is None)
    ms2 = Menu()
    ms2.start(c4)
    sl2 = [x for x in ms2.slots(c4) if x.cls is not None]
    pa, pb = sl2[0].picture.center(), sl2[1].picture.center()
    ms2.shot(modes.ShotEvent(pa.x(), pa.y(), 100.0, 1, pointer=0), c4)
    ms2.shot(modes.ShotEvent(pb.x(), pb.y(), 100.0, 2, pointer=1), c4)
    check("two pistols in one turn: the FIRST choice stands, the second does not overwrite it",
          ms2.pending is sl2[0].cls and ms2.hits == 2 and ms2.shots == 2)
    check("take_pending hands the choice over once",
          ms2.take_pending() is sl2[0].cls and ms2.take_pending() is None)
    # take_pending must be HONEST (RULES C1): a choice and consumption together, or
    # None and consumption of nothing -- never a consumed choice reported as None.
    mh_ = Menu()
    mh_.start(c4)
    none_first = mh_.take_pending()
    mh_._choose(sl2[0].cls)                  # a None-return must not have eaten state
    kept = mh_.pending is sl2[0].cls
    got_ = mh_.take_pending()
    cons_ = mh_.pending is None
    each_ok = []
    for s_ in [x for x in sl2 if x.cls is not None]:
        mh_.pending = s_.cls
        r1 = mh_.take_pending()
        each_ok.append(r1 is s_.cls and mh_.pending is None and mh_.take_pending() is None)
    check("take_pending: None when nothing is pending (and consumes nothing), the class "
          "and consumed when something is, for every drill",
          none_first is None and kept and got_ is sl2[0].cls and cons_ and all(each_ok),
          str(each_ok))
    chrome_ms = Menu()
    chrome_ms.start(c4)
    exit_slot = next(s for s in chrome_ms.slots(c4) if s.cls is None and s.stem == "exit")
    ec = exit_slot.picture.center()
    eo = chrome_ms.shot(modes.ShotEvent(ec.x(), ec.y(), 100.0, 1), c4)
    check("a shot on the EXIT tile is a hit that does NOTHING yet: no choice recorded, no hole "
          "(quitting is a later piece and must take two shots)",
          eo.hit and eo.draw_hole is False and chrome_ms.pending is None
          and chrome_ms.take_pending() is None and eo.target_id == "tile:exit")
    ms2.shot(modes.ShotEvent(pa.x(), pa.y(), 100.0, 3), c4)
    ms2.stop()
    check("stop() clears a recorded choice", ms2.pending is None)
    ms2.start(c4)
    check("start() resets the tally", ms2.shots == 0 and ms2.hits == 0)

    # ------------------------------------------------------------------
    print("\n8. shot() does NOT switch the mode, and neither does the menu: it only records")
    host = base.ModeHost()
    menu_inst = Menu()
    host.set_mode(menu_inst, c4)
    calls = {"set_mode": 0, "stop": 0}
    real_set, real_stop = host.set_mode, menu_inst.stop

    def counting_set(mode, ctx=None):
        calls["set_mode"] += 1
        return real_set(mode, ctx)

    def counting_stop():
        calls["stop"] += 1
        return real_stop()

    host.set_mode = counting_set
    menu_inst.stop = counting_stop
    target = next((s for s in menu_inst.slots(c4) if s.cls is modes.StaticTargets),
                  menu_inst.slots(c4)[0])
    tc = target.picture.center()
    outcome = host.shot(modes.ShotEvent(tc.x(), tc.y(), 100.0, 1), c4)
    check("through the host: the shot is a hit on the right tile",
          outcome.hit and outcome.target_id == f"tile:{target.cls.name}")
    check("IMMEDIATELY after shot() returns the host's mode is unchanged (still the menu)",
          host.mode is menu_inst and calls["set_mode"] == 0)
    check("...the menu was not stopped while it was executing (set_mode would stop it)",
          calls["stop"] == 0)
    check("...and the choice is waiting on a public attribute", menu_inst.pending is target.cls)
    taken = menu_inst.take_pending()
    check("the app's hand-off is take_pending(): the choice comes out once, and taking it "
          "switches NOTHING (the app routes it through camera_lab._select_mode)",
          taken is target.cls and menu_inst.pending is None and host.mode is menu_inst
          and calls["set_mode"] == 0 and calls["stop"] == 0)
    check("the second switch route is GONE: the menu has no launch_pending",
          not hasattr(Menu, "launch_pending") and not hasattr(menu_inst, "launch_pending"))
    other = Menu()
    other.start(c4)
    po = next(s_ for s_ in other.slots(c4) if s_.cls is not None).picture.center()
    other.shot(modes.ShotEvent(po.x(), po.y(), 100.0, 1), c4)
    other.stop()
    check("stop() clears a recorded choice, so a stale one cannot outlive its menu",
          other.pending is None and other.take_pending() is None)
    # The structural half. The behavioural checks above cannot see a switch that
    # is added later by giving the menu a way to reach the host. ANY reference to
    # set_mode or restart in the module's syntax tree counts (a call, a bare
    # attribute, a name, a def); prose in docstrings and comments is not syntax.
    src = (ROOT / "src" / "lsa" / "modes" / "menu.py").read_text(encoding="utf-8")
    offenders = sorted({f"line {node.lineno}: {getattr(node, 'attr', None) or getattr(node, 'id', None) or node.name}"
                        for node in ast.walk(ast.parse(src))
                        if (isinstance(node, ast.Attribute) and node.attr in ("set_mode", "restart", "launch_pending"))
                        or (isinstance(node, ast.Name) and node.id in ("set_mode", "restart", "launch_pending"))
                        or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name in ("set_mode", "restart", "launch_pending"))})
    check("menu.py does not reference set_mode, restart or launch_pending AT ALL "
          "(one route for switching modes: _select_mode)", not offenders, str(offenders))

    print("\n   keys")
    mk = Menu()
    mk.start(c4)
    # THE TRAP (review blocker): while the menu is the active mode its on_key used to
    # return True for every tile key, so TargetWindow.keyPressEvent never emitted
    # key_pressed, camera_lab._on_target_key's by_key path was never reached, and
    # nothing drained `pending` -- the shooter was stranded on the menu with F11
    # (which closes the whole app) the only way out. A mode must not claim a key it
    # cannot act on, so it claims NONE, and the by_key path stays reachable.
    # CHANGED 2026-10-20: the menu claims exactly the keys listed in `MENU_OWN_KEYS`
    # (O and T, the player-count choice) and nothing else; `DELIBERATE_KEY_OVERRIDES`
    # is EMPTY because neither key is used by anything else.
    claimed = [k for k in range(0, 0x1100) if Menu().on_key(k)]
    check("the menu claims EXACTLY its listed keys (O and T) out of 0..0x10FF: nothing else is "
          "swallowed, so _on_target_key's by_key path is reached for every tile key",
          sorted(claimed) == sorted(menu_mod.MENU_OWN_KEYS)
          and sorted(claimed) == sorted([Qt.Key_O, Qt.Key_T]),
          str([hex(k) for k in claimed[:8]]))
    check("...and there are NO overrides left to declare: the list is empty and S, M and T are not in it",
          menu_mod.DELIBERATE_KEY_OVERRIDES == {} and not ({Qt.Key_S, Qt.Key_M} & set(claimed)))
    bad_keys = []
    for s in mk.slots(c4):
        # Drill tiles only: their glyph IS the key. A chrome tile's glyph is
        # a label for a key the menu does not own -- EXIT shows "Esc" and
        # Escape is handled by the window, not by `by_key`.
        if not s.glyph or s.cls is None:
            continue
        host_k = base.ModeHost()
        host_k.set_mode(Menu(), c4)
        # through the HOST, the way the scene reaches it, and the key must fall through
        took = host_k.on_key(ord(s.glyph))
        if took or host_k.mode.pending is not None or base.by_key(ord(s.glyph)) is not s.cls:
            bad_keys.append(s.glyph)
    check("each tile's own key falls through the menu (host.on_key False, nothing recorded) "
          "AND lands on that tile's mode in the app's by_key lookup", not bad_keys,
          str(bad_keys))
    check("a key no mode owns (E, Y) is not consumed and chooses nothing",
          mk.on_key(Qt.Key_E) is False and mk.on_key(Qt.Key_Y) is False and mk.pending is None
          and mk.pending_players is None)
    check("on_key switches nothing and records nothing",
          base.by_key(Qt.Key_5) is not None and mk.on_key(Qt.Key_5) is False
          and mk.pending is None and mk.take_pending() is None)
    with extra_mode("zz_nopicture", Qt.Key_J, "No picture drill") as NoPic:
        check("a mode registered later is likewise left to the by_key path",
              mk.on_key(Qt.Key_J) is False and base.by_key(Qt.Key_J) is NoPic
              and mk.pending is None)

    # ------------------------------------------------------------------
    print("\n9. moving_at: the menu is still, before and after a shot")
    mv = Menu()
    mv.start(c4)
    before = [mv.moving_at(t) for t in (0.0, 50.0, 100.0)]
    pcv = mv.slots(c4)[0].picture.center()
    mv.shot(modes.ShotEvent(pcv.x(), pcv.y(), 100.0, 1), c4)
    after = [mv.moving_at(t) for t in (0.0, 50.0, 100.0, 100.05)]
    check("moving_at is False at any time, and the same answer after a hit (the learner may learn)",
          not any(before) and not any(after))
    check("status() is a short non-empty line naming what the menu is for",
          0 < len(mv.status()) < 100 and "MENU" in mv.status())
    aim = mv.aim_point(c4)
    check("aim_point is inside the first tile (the --modetest walk-through needs one)",
          aim is not None and mv.tile_at(aim[0], aim[1], c4) is not None
          and 0 <= aim[0] <= SW and 0 <= aim[1] <= SH)

    # ------------------------------------------------------------------
    print("\n10. the key glyph and name fit inside their tile, measured with font metrics")

    class Spy(QPainter):
        rec: list = []

        def drawText(self, *a):
            if len(a) >= 3 and isinstance(a[0], (QRectF, QRect)):
                fm = self.fontMetrics()
                self.rec.append((QRectF(a[0]), a[2], fm.horizontalAdvance(a[2]),
                                 fm.height(), self.font().pointSize()))
            return super().drawText(*a)

    def spy_paint(mode, ctx):
        img = QImage(ctx.width, ctx.height, QImage.Format_RGB32)
        img.fill(0xFF0A0A0A)
        Spy.rec = []
        sp = Spy(img)
        mode.paint(sp, ctx)
        sp.end()
        return list(Spy.rec)

    for label, ctx in (("4K", c4), ("1080p", c1080)):
        mm = Menu()
        mm.start(ctx)
        recs = spy_paint(mm, ctx)
        sl = mm.slots(ctx)
        overruns = [f"{t!r} needs {w:.0f}x{h:.0f}, cell {r.width():.0f}x{r.height():.0f}"
                    for r, t, w, h, _ps in recs if w > r.width() + 1 or h > r.height() + 1]
        check(f"{label}: every string drawn fits the cell it is drawn into ({len(recs)} strings)",
              len(recs) >= len(sl) + sum(1 for s in sl if s.glyph) and not overruns,
              "; ".join(overruns[:3]))
        # "MAIN MENU" is the one string that is NOT in a tile, by design: it
        # is the heading above the grid, asked for on 2026-10-18 so the
        # screen says what it is. It gets its own assertions below rather
        # than an exemption that would hide a tile caption escaping.
        stray = [t for r, t, *_ in recs
                 if t != "MAIN MENU"
                 and not any(s.rect.adjusted(-0.5, -0.5, 0.5, 0.5).contains(r)
                             for s in sl)]
        check(f"{label}: every tile string is inside its own tile",
              not stray, str(stray[:3]))
        head = [(r, ps) for r, t, _w, _h, ps in recs if t == "MAIN MENU"]
        check(f"{label}: the screen says MAIN MENU, once", len(head) == 1)
        if head:
            hr, hps = head[0]
            check(f"{label}: ...above the grid, not over a tile",
                  hr.bottom() <= min(s.rect.top() for s in sl) + 1.0
                  and not any(s.rect.intersects(hr) for s in sl),
                  f"heading bottom {hr.bottom():.0f}, "
                  f"tiles start {min(s.rect.top() for s in sl):.0f}")
            check(f"{label}: ...and readable at this size",
                  hps >= menu_mod.NAME_FLOOR_FRAC * layers.px(
                      ctx, menu_mod.HEADING_AT_1080),
                  f"{hps}pt")
        names = {t: ps for _r, t, _w, _h, ps in recs}
        # ONE definition of the floor, read from the module that draws against
        # it. A copy here would let the drawn size and the bar drift apart.
        floor_ps = menu_mod.NAME_FLOOR_FRAC * layers.px(ctx, menu_mod.NAME_AT_1080)
        # A name may be drawn whole, or wrapped onto two lines when one line
        # would be too small to read from the firing line. The PROPERTY is that
        # the whole name reaches his eye at a legible size -- not that it is
        # one drawString, which is a mechanism and would forbid the fix.
        missing, small = [], []
        for s in sl:
            parts = ([s.title] if s.title in names
                     else list(menu_mod.wrap_title(s.title)))
            if s.lead:
                # The player-count tiles draw the FIRST LETTER separately (bold, a
                # little brighter), so the name reaches the painter in two pieces
                # on its first line: the letter, then the rest.
                if s.title[0] in names and s.title[1:] in names:
                    parts = [s.title[0], s.title[1:]]          # drawn on one line
                else:                                          # wrapped onto two
                    first_ = parts[0]
                    parts = [first_[0], first_[1:]] + list(parts[1:])
            if any(t not in names for t in parts):
                missing.append(s.title)
                continue
            worst = min(names[t] for t in parts)
            if worst < floor_ps:
                small.append(f"{s.title} at {worst}pt"
                             + ("" if len(parts) == 1 else " wrapped"))
        check(f"{label}: every mode's name is drawn IN FULL (not elided) at a legible size "
              f"(>= {floor_ps:.0f}pt)", not missing and not small, f"{missing} {small}")
        check(f"{label}: every key glyph is drawn", all(s.glyph in names for s in sl if s.glyph))
    long_title = "W" * 70
    with extra_mode("zz_long", Qt.Key_Z, long_title) as LongCls:
        recs = spy_paint(Menu(), c1080)
        got = [(r, t, w) for r, t, w, _h, _p in recs if t != long_title and t.startswith("W")]
        check("an over-long name is shrunk and elided to its cell, never allowed to overrun it",
              bool(got) and all(w <= r.width() + 1 for r, _t, w in got)
              and not any(t == long_title for _r, t, *_ in recs),
              f"{[(t[:12], round(w)) for _r, t, w in got]}")
    img = QImage(400, 100, QImage.Format_RGB32)
    img.fill(0)
    pnt = QPainter(img)
    size_fit, text_fit = fit_text(pnt, "ok", False, 20, 300, 60)
    size_shr, text_shr = fit_text(pnt, "x" * 60, False, 40, 100, 60)
    wide = pnt.fontMetrics().horizontalAdvance(text_shr)
    pnt.end()
    check("fit_text: text that fits keeps its starting size and is untouched",
          size_fit == 20 and text_fit == "ok")
    check("fit_text: text that cannot fit at any size is elided to the cell",
          text_shr != "x" * 60 and wide <= 100 and size_shr == menu_mod.FONT_FLOOR,
          f"'{text_shr}' {wide}px at {size_shr}pt")

    # ------------------------------------------------------------------
    print("\n11. tiles are SOLID: bullet holes underneath do not show through the art")
    # The pictures are white at coverage v/255 on a TRANSPARENT ground, and holes
    # persist across a mode switch. He photographed exactly this on the animated
    # figure, twice. Tile geometry is the one `slots` computes (no recompute).
    mo = Menu()
    mo.start(c4)
    occ = mo.occluders(c4)
    slots_o = mo.slots(c4)
    check("the menu publishes one occluder per tile, each covering its tile's whole rectangle",
          len(occ) == len(slots_o) > 0
          and all(o.rect.contains(s.rect.toAlignedRect()) for o, s in zip(occ, slots_o)),
          f"{len(occ)} occluders for {len(slots_o)} tiles")
    check("...above the holes layer (an occluder only hides layers BELOW its z)",
          all(o.z > layers.BulletHoles().z for o in occ))
    mk_img = occ[0].mask().toImage()
    check("...with a SOLID mask (alpha > 0 where it is solid, and the mask is not empty)",
          mk_img.width() >= 1 and mk_img.pixelColor(0, 0).alpha() == 255)
    cal_ctx = RenderContext(now=100.0, dt=0.0, rect=QRect(0, 0, SW, SH),
                            stats=SessionStats(), calibrating=True)
    check("while the grid is being shot (paint draws nothing) it publishes nothing",
          mo.occluders(cal_ctx) == [])

    # End to end through the REAL Scene.paint, which gathers the occluders.
    from lsa.display.scene import Scene as _Scene

    def paint_scene(sc_, t_):
        sc_._context = lambda now, dt, rect, _t=t_: context(now=_t)
        im_ = QImage(SW, SH, QImage.Format_RGB32)
        im_.fill(0xFF0A0A0A)
        pp_ = QPainter(im_)
        sc_.paint(pp_, QRect(0, 0, SW, SH))
        pp_.end()
        a_ = np.frombuffer(im_.constBits(), np.uint8).reshape(SH, im_.bytesPerLine() // 4, 4)
        return a_[:, :SW, 1].astype(int)

    sc = _Scene()
    sc.add(layers.SolidBackground())
    holes = layers.BulletHoles()
    sc.add(holes)
    host_o = base.ModeHost()
    sc.add(host_o)
    menu_o = Menu()
    menu_o.start(context(now=100.0))
    host_o.set_mode(menu_o, None)
    tile_rects = [s.rect for s in menu_o.slots(c4)]
    for i, r in enumerate(tile_rects):
        c_ = r.center()
        holes.add(c_.x(), c_.y(), i, 0)
    for h_ in holes.holes:
        h_.born -= 5.0                              # settled, not flashing
    holes.visible = False
    bare = paint_scene(sc, 100.0)
    holes.visible = True
    solid = paint_scene(sc, 100.0)
    real_occ = base.ModeHost.occluders
    base.ModeHost.occluders = lambda self, ctx: []
    try:
        floaty = paint_scene(sc, 100.0)
    finally:
        base.ModeHost.occluders = real_occ

    def shows(img):
        d = np.abs(img - bare) > 6
        return sum(int(d[int(r.top()):int(r.bottom()), int(r.left()):int(r.right())].sum())
                   for r in tile_rects)
    n_solid, n_floaty = shows(solid), shows(floaty)
    check(f"holes at the centre of every tile are hidden by the tiles ({n_solid} px show, "
          f"against {n_floaty} when the menu announces nothing solid)",
          n_floaty > 300 and n_solid <= 0.05 * n_floaty)
    outside = (SW * 0.5, 40.0)
    holes.add(outside[0], outside[1], 99, 0)
    holes.holes[-1].born -= 5.0
    after = paint_scene(sc, 100.0)
    check("a hole in no tile (the top band) is still drawn",
          int((np.abs(after - solid) > 6)[10:80, int(SW * 0.5) - 60:int(SW * 0.5) + 60].sum()) > 50)

    # THE SWEEP. Discovered from BEHAVIOUR, not from a module name: the earlier
    # figure/animated check keyed on which modules hold Sequence or Figure, so a
    # third art loader (tiles.py) was invisible to it. Here every registered mode
    # (and the menu) is driven for ~7 s of virtual time and painted through a
    # painter that records every pixmap or image with an alpha channel it is asked
    # to draw -- the signature of line art on a transparent ground. Wherever one
    # lands, the mode must publish occluders that cover it.
    class ArtSpy(QPainter):
        rec: list = []

        def _note(self, args):
            for i, a in enumerate(args):
                if isinstance(a, (QPixmap, QImage)):
                    if not a.hasAlphaChannel():
                        return
                    before = args[:i]
                    if len(before) == 4 and all(isinstance(v, (int, float)) for v in before):
                        dest = QRectF(*before)
                    elif len(before) == 2 and all(isinstance(v, (int, float)) for v in before):
                        dest = QRectF(before[0], before[1], a.width(), a.height())
                    elif len(before) == 1 and isinstance(before[0], (QRect, QRectF)):
                        dest = QRectF(before[0])
                    elif len(before) == 1:
                        dest = QRectF(before[0].x(), before[0].y(), a.width(), a.height())
                    else:
                        return
                    ArtSpy.rec.append(dest)
                    return

        def drawPixmap(self, *a):
            self._note(a)
            return super().drawPixmap(*a)

        def drawImage(self, *a):
            self._note(a)
            return super().drawImage(*a)

    uncovered: dict = {}
    drew: set = set()
    img_s = QImage(SW, SH, QImage.Format_RGB32)
    for cls_s, _k in base.registered():
        ms_ = cls_s()
        if hasattr(ms_, "_rng"):
            ms_._rng.seed(5)
        t_s = 100.0
        ms_.start(context(now=t_s))
        for k_s in range(1, 420):
            t_s += 1 / 60.0
            ms_.advance(context(now=t_s))
            if k_s % 14:
                continue
            cx_ = context(now=t_s)
            img_s.fill(0xFF0A0A0A)
            ArtSpy.rec = []
            sp_ = ArtSpy(img_s)
            ms_.paint(sp_, cx_)
            sp_.end()
            if not ArtSpy.rec:
                continue
            drew.add(cls_s.name)
            region = QRegion()
            for o_ in ms_.occluders(cx_):
                region = region.united(QRegion(o_.rect))
            for dest in ArtSpy.rec:
                # 2 px of slack per side: an occluder rect is integer-truncated
                # where the picture is drawn at a fractional position.
                inner = dest.adjusted(2, 2, -2, -2)
                if inner.width() <= 0 or inner.height() <= 0:
                    continue
                if not QRegion(inner.toAlignedRect()).subtracted(region).isEmpty():
                    uncovered.setdefault(cls_s.name, []).append(
                        (round(dest.x()), round(dest.y()), round(dest.width()), round(dest.height())))
        ms_.stop()
    check("the sweep SAW art being drawn: the menu and at least one other mode painted a "
          "transparent picture (a sweep that observes nothing passes everything)",
          "menu" in drew and len(drew) >= 2, str(sorted(drew)))
    check("every mode that draws transparent artwork publishes occluders covering it "
          "(on a branch that predates the animated figure's occluders, 'animated' is "
          "the expected name here)", not uncovered,
          "; ".join(f"{n}: {v[:2]}" for n, v in uncovered.items()))


    # ------------------------------------------------------------------
    print("\n12. a shot on a tile STARTS that drill, through the app's own path")
    # THE ONE BEHAVIOUR THE WHOLE FEATURE IS FOR, and it arrived with no test.
    # From the firing line a tile that swallows a shot and does nothing is
    # indistinguishable from a dead detector, which is the complaint this
    # project spent a week chasing. Everything else on this screen degrades
    # gracefully; this does not.
    #
    # Driven through the REAL CameraLab, not a stand-in: the behaviour IS the
    # wiring -- Menu.shot records, the HUD tick drains, `_select_mode`
    # switches, clears the holes and resets the session. A fake host would
    # prove the half that was never in doubt (C12, C14: test across the seam).
    from owned_files import isolate_outputs as _iso12
    _iso12("menu_drain")
    import lsa.settings as _settings12
    from lsa.tools.camera_lab import CameraLab as _Lab12
    _orig12 = _settings12.AppSettings.load

    @staticmethod
    def _no_connect12(path):
        st = _orig12(path)
        st.auto_connect = False
        return st

    _settings12.AppSettings.load = _no_connect12
    try:
        lab12 = _Lab12()
    finally:
        _settings12.AppSettings.load = _orig12
    # A test may never write what he owns, and may never open his camera.
    lab12.settings.auto_connect = True
    lab12._save_settings = lambda *a, **k: None
    lab12._schedule_save = lambda *a, **k: None
    check("the test did not open his camera", lab12.capture is None)

    host12 = lab12.layers.get("modes")
    # MODE 1 WHILE IT MEASURES ITSELF, the menu once it has answered. He
    # revised the order on 2026-10-18 after watching a launch: the screen
    # search, the probe and the self test all photograph this screen, and
    # eleven lit tiles make those measurements worse (B10).
    import lsa.tools.camera_lab as _CL12
    check("the app starts on the drill it measures itself against, not the menu",
          getattr(host12.mode, "name", None) == _CL12.STARTUP_MODE_NAME,
          type(getattr(host12, "mode", None)).__name__)


    lab12._toggle_target_mode()          # opens the fullscreen target screen
    app.processEvents()
    win12 = lab12.target_window
    check("the target screen opened for the drain checks", win12 is not None)

    # THE SCREEN FIRST: `_select_mode` does nothing without a target window,
    # so asking for the menu before one exists silently leaves the startup
    # drill up -- and the check would have been measuring that, not the
    # hand-off.
    lab12._menu_shown_at_startup = False
    lab12._menu_when_setup_done()
    check("...and the menu goes up when setup is done",
          getattr(host12.mode, "name", None) == "menu",
          type(host12.mode).__name__)

    ctx12 = lab12.scene._context(time.perf_counter(), 0.0, win12.rect())
    menu12 = host12.mode
    slots12 = menu12.slots(ctx12)
    # every mode but the menu and the screens (see section 2: a screen has no tile)
    _drills12 = len([c for c, _k in base.registered()
                     if c.name != "menu" and not getattr(c, "is_screen", False)])
    check("the menu lays out one tile per drill, plus the chrome tiles (EXIT)",
          len(slots12) == _drills12 + len(menu_mod.CHROME),
          f"{len(slots12)} tiles = {_drills12} drills + {len(menu_mod.CHROME)} chrome")

    # A hole from the previous string, so the clearing can be observed rather
    # than assumed from `_select_mode`'s source.
    lab12.layers["holes"].add(100.0, 100.0, 1, 0)
    had_hole12 = len(lab12.layers["holes"].holes)

    target12 = next(s for s in slots12 if s.cls is not None and s.cls.name != "menu")
    c12 = target12.picture.center()
    ev12 = modes.ShotEvent(c12.x(), c12.y(), time.perf_counter(), 1)
    out12 = host12.shot(ev12, ctx12)
    check("the shot is reported as a hit on that tile and draws no hole",
          out12.hit and out12.draw_hole is False
          and out12.target_id == f"tile:{target12.cls.name}",
          f"hit={out12.hit} hole={out12.draw_hole} id={out12.target_id}")
    check("...and the mode has NOT switched inside shot() (that would stop a "
          "running object)", host12.mode is menu12)

    started12 = lab12._drain_menu_choice()
    check("the HUD tick starts the drill the shot chose",
          started12 == target12.cls.name
          and isinstance(host12.mode, target12.cls),
          f"drain returned {started12!r}, mode is "
          f"{type(host12.mode).__name__}")
    check("...and the previous string's holes went with the switch (inherited "
          "from _select_mode, not a second mechanism)",
          had_hole12 == 1 and len(lab12.layers["holes"].holes) == 0,
          f"{had_hole12} before, {len(lab12.layers['holes'].holes)} after")
    check("a second drain starts nothing: the choice was consumed once",
          lab12._drain_menu_choice() is None)

    # Back to the menu, and the two refusals.
    lab12._select_mode(Menu())
    menu12b = host12.mode
    check("the menu can be returned to", isinstance(menu12b, Menu))
    menu12b.shot(modes.ShotEvent(c12.x(), c12.y(), time.perf_counter(), 2), ctx12)
    lab12._grid_active = True
    try:
        check("a choice is DROPPED while the calibration grid is running "
              "(a shot at an invisible tile must not fire later)",
              lab12._drain_menu_choice() is None
              and menu12b.pending is None
              and host12.mode is menu12b)
    finally:
        lab12._grid_active = False

    menu12b.shot(modes.ShotEvent(c12.x(), c12.y(), time.perf_counter(), 3), ctx12)
    held12 = menu12b.pending
    _win12, lab12.target_window = lab12.target_window, None
    try:
        check("with no target screen the choice is KEPT, not silently eaten",
              lab12._drain_menu_choice() is None and menu12b.pending is held12)
    finally:
        lab12.target_window = _win12

    # A choice left on a menu that has since been replaced must never start
    # anything: that is a stale click arriving in somebody else's drill.
    stale12 = Menu()
    stale12.start(ctx12)
    stale12.shot(modes.ShotEvent(c12.x(), c12.y(), time.perf_counter(), 4), ctx12)
    lab12._select_mode(target12.cls())
    check("a choice from a replaced menu starts nothing",
          lab12._drain_menu_choice() is None
          and isinstance(host12.mode, target12.cls))

    # The drain runs on the HUD tick itself, not only when called by name --
    # a check that calls the helper directly would pass with nothing wired.
    lab12._select_mode(Menu())
    menu12c = host12.mode
    menu12c.shot(modes.ShotEvent(c12.x(), c12.y(), time.perf_counter(), 5), ctx12)
    lab12._refresh_target_diagnostics()
    check("the HUD tick itself drains the choice (the helper is actually wired)",
          isinstance(host12.mode, target12.cls),
          type(host12.mode).__name__)

    if lab12.target_window is not None:
        lab12.target_window.close()
    app.processEvents()

    # ------------------------------------------------------------------
    print("\n13. the pointer: a click starts a drill, and is never a shot")
    # Asked for on 2026-10-18: "menu should also be triggered by the mouse
    # pointer". The target screen had NO mouse input of any kind before this
    # -- it is a frameless fullscreen surface built for a laser -- so the
    # click path is new from the window down.
    mc = Menu()
    mc.start(c4)
    sl_c = mc.slots(c4)
    tile_c = next(s for s in sl_c if s.cls is not None)
    pc = tile_c.picture.center()
    took = mc.on_click(pc.x(), pc.y(), c4)
    check("a click on a tile chooses that drill",
          took is True and mc.pending is tile_c.cls,
          f"took={took} pending={getattr(mc.pending, 'name', None)}")
    check("...and it is NOT counted as a shot: the gun's tallies are untouched",
          mc.shots == 0 and mc.hits == 0, f"shots={mc.shots} hits={mc.hits}")

    mc2 = Menu()
    mc2.start(c4)
    gap = None
    for yy in range(0, int(SH), 7):
        for xx in range(0, int(SW), 7):
            if mc2.tile_at(float(xx), float(yy), c4) is None:
                gap = (float(xx), float(yy))
                break
        if gap:
            break
    check("a click between the tiles is not consumed (it falls through)",
          gap is not None and mc2.on_click(gap[0], gap[1], c4) is False
          and mc2.pending is None, str(gap))

    chrome_c = next((s for s in sl_c if s.cls is None and s.stem == "exit"), None)
    if chrome_c is not None:
        mc3 = Menu()
        mc3.start(c4)
        cc = chrome_c.picture.center()
        check("a click on EXIT is consumed and starts nothing (it is inert "
              "until it is armed)",
              mc3.on_click(cc.x(), cc.y(), c4) is True and mc3.pending is None)

    # The scene routes it, topmost visible layer first, exactly as it routes a
    # key. A mode that does not want a click returns False and it falls through.
    from lsa.display.scene import Scene as _Sc13
    sc13 = _Sc13()
    host13 = base.ModeHost()
    sc13.add(host13)
    m13 = Menu()
    host13.set_mode(m13, c4)
    check("Scene.on_click reaches the active mode",
          sc13.on_click(pc.x(), pc.y(), c4) is True
          and m13.pending is tile_c.cls)
    host13.set_mode(modes.by_name("static")(), c4)
    check("...and a mode with no use for a click leaves it unconsumed",
          sc13.on_click(pc.x(), pc.y(), c4) is False)

    # END TO END, through the real window and the real app: the event the
    # pointer generates must start the drill. Driven with the SAME helper the
    # introspection server uses, so what is verified here is what is driven
    # when nobody is at the machine.
    from lsa.introspect.gui import post_click as _post_click13
    lab13 = lab12                                   # the lab built in section 12
    # THE SCREEN FIRST, THEN THE MODE. `_select_mode` does nothing at all
    # without a target window (it would have nowhere to build a context), and
    # section 12 closed it -- so selecting first silently left the previous
    # drill up and the check below failed on a lab that was fine.
    if lab13.target_window is None:
        lab13._toggle_target_mode()
        app.processEvents()
    lab13._select_mode(Menu())
    app.processEvents()
    host_l13 = lab13.layers["modes"]
    menu_l13 = host_l13.mode
    check("the app is back on the menu for the pointer check",
          isinstance(menu_l13, Menu), type(menu_l13).__name__)
    win13 = lab13.target_window
    ctx13 = lab13.scene._context(time.perf_counter(), 0.0, win13.rect())
    slot13 = next(s for s in menu_l13.slots(ctx13) if s.cls is not None)
    p13 = slot13.picture.center()
    shots_before13 = lab13.session.shots if hasattr(lab13.session, "shots") else None
    _post_click13(win13, p13.x(), p13.y())
    app.processEvents()
    lab13._refresh_target_diagnostics()
    check("a click posted to the target window starts that drill, end to end",
          isinstance(host_l13.mode, slot13.cls),
          f"{type(host_l13.mode).__name__} after clicking the "
          f"{slot13.cls.name} tile")
    if shots_before13 is not None:
        check("...and the session recorded NO shot for it",
              lab13.session.shots == shots_before13,
              f"{shots_before13} -> {lab13.session.shots}")

    # ------------------------------------------------------------------
    print("\n13b. the target window's pointer: hidden, shown by a move, hidden after 1 s")
    # His request, 2026-10-22: hide the cursor at start, show it on movement,
    # hide it one second after the mouse stops. Windows composites the cursor
    # over the window, so it is absent from every render check in this suite;
    # NO CHECK HERE CAN SEE THE REAL CURSOR (offscreen Qt has no compositor).
    # These bound the app's BEHAVIOUR -- which cursor the window asks for, and
    # that asking costs no repaint -- not the pixels on his panel. Only his
    # bench can confirm the noise floor drops.
    from PySide6.QtCore import QEvent as _QE13, QPointF as _QP13, Qt as _Qt13
    from PySide6.QtGui import QMouseEvent as _QM13
    from PySide6.QtWidgets import QApplication as _QA13
    from lsa.display.window import TargetWindow as _TW13

    def _move13(w, x=40.0, y=40.0):
        _QA13.sendEvent(w, _QM13(_QE13.MouseMove, _QP13(x, y), _Qt13.NoButton,
                                 _Qt13.NoButton, _Qt13.NoModifier))

    fresh13 = _TW13(lab13.scene, target_fps=60)
    check("a NEW target window's cursor is blank from construction, before any "
          "timer has run", fresh13.cursor_hidden() is True
          and fresh13.cursor().shape() == _Qt13.BlankCursor)
    check("...it tracks the mouse without a button held, or no move would arrive",
          fresh13.hasMouseTracking() is True)
    check("the countdown is one second",
          _TW13.CURSOR_HIDE_S == 1.0 and fresh13._cursor_timer.interval() == 1000
          and fresh13._cursor_timer.isSingleShot())
    fresh13.close()
    fresh13.deleteLater()

    check("the live target window starts with a blank cursor",
          win13.cursor_hidden() is True)
    _move13(win13)
    check("a mouse move shows the ordinary cursor at once",
          win13.cursor_hidden() is False and win13._cursor_timer.isActive())

    # No repaint, nothing dirty: measured around SYNCHRONOUS calls, so the
    # scene's own tick cannot be what is being counted.
    upd13, dirty13 = [], []
    real_update13, real_dirty13 = win13.update, lab13.scene.mark_dirty
    win13.update = lambda *a, **k: (upd13.append(1), real_update13(*a, **k))[1]
    lab13.scene.mark_dirty = lambda *a, **k: (dirty13.append(1), real_dirty13(*a, **k))[1]
    try:
        lab13.scene._dirty = False
        win13._hide_cursor()
        _move13(win13, 80.0, 90.0)
        _move13(win13, 120.0, 90.0)
        win13._hide_cursor()
        check("showing and hiding the cursor requests NO repaint and marks "
              "nothing dirty",
              upd13 == [] and dirty13 == [] and lab13.scene._dirty is False,
              f"update calls {len(upd13)}, mark_dirty calls {len(dirty13)}, "
              f"scene dirty {lab13.scene._dirty}")
    finally:
        win13.update, lab13.scene.mark_dirty = real_update13, real_dirty13
        del win13.update, lab13.scene.mark_dirty

    # The one-second timer itself, run for real with the interval shortened
    # (the constant is asserted above, so shortening it here cannot hide a
    # wrong number).
    win13._cursor_timer.setInterval(60)
    _move13(win13)
    shown13 = win13.cursor_hidden() is False
    t13 = time.perf_counter()
    while time.perf_counter() - t13 < 0.5 and not win13.cursor_hidden():
        app.processEvents()
    check("stillness hides it again", shown13 and win13.cursor_hidden() is True)
    _move13(win13)
    check("a further move shows it again", win13.cursor_hidden() is False)
    # Moving keeps restarting the countdown: still moving means still shown.
    t13 = time.perf_counter()
    while time.perf_counter() - t13 < 0.2:
        _move13(win13)
        app.processEvents()
    check("continued movement keeps it visible", win13.cursor_hidden() is False)
    win13._cursor_timer.setInterval(int(_TW13.CURSOR_HIDE_S * 1000))
    win13._hide_cursor()

    # A CLICK STILL WORKS WITH THE POINTER HIDDEN, and is still not a shot.
    lab13._select_mode(Menu())
    app.processEvents()
    host_h13 = lab13.layers["modes"]
    ctx_h13 = lab13.scene._context(time.perf_counter(), 0.0, win13.rect())
    slot_h13 = next(s for s in host_h13.mode.slots(ctx_h13) if s.cls is not None)
    ph13 = slot_h13.picture.center()
    win13._hide_cursor()
    shots_h13 = lab13.session.shots
    _post_click13(win13, ph13.x(), ph13.y())
    app.processEvents()
    # A tile hit only RECORDS the choice (`Menu.pending`); the app starts the
    # drill when the HUD tick drains it in `_refresh_target_diagnostics`. The
    # pre-existing end-to-end check above makes this same call, and this one
    # omitted it, so it asserted a drill had started before the app had been
    # given the turn that starts it. Without the call the check fails on a
    # window that is fine -- it is not a click-delivery regression.
    lab13._refresh_target_diagnostics()
    check("with the cursor hidden and no move first, a click still starts that "
          "tile's drill",
          isinstance(host_h13.mode, slot_h13.cls),
          f"{type(host_h13.mode).__name__} after clicking {slot_h13.cls.name}")
    check("...it did not show the cursor (a click is not a move)",
          win13.cursor_hidden() is True)
    check("...and it was not counted as a shot",
          lab13.session.shots == shots_h13,
          f"{shots_h13} -> {lab13.session.shots}")

    # SCOPE: only the target window. The console keeps its cursor and nothing
    # is installed process-wide.
    _move13(win13)
    win13._hide_cursor()
    check("the operator console's cursor is untouched",
          lab13.cursor().shape() != _Qt13.BlankCursor)
    check("no application-wide override cursor is installed",
          _QA13.overrideCursor() is None)

    # ------------------------------------------------------------------
    print("\n13c. the flash and splash windows: blank cursor for their whole life, "
          "and the ways out are unchanged")
    # His ruling, 2026-10-22: the cursor is hidden on the other two windows the
    # camera photographs. Same honesty as 13b: offscreen Qt has no compositor, so
    # this bounds which cursor each window ASKS for, not the pixels on his panel;
    # only his bench can confirm the noise floor. They need no show-on-move, so
    # the check is simply that it is blank before, during and after being shown.
    from PySide6.QtGui import QKeyEvent as _QK13
    from lsa.calibrate.flash import FlashWindow as _FW13
    from lsa.display.splash import Splash as _SP13

    def _dbl13(w):
        _QA13.sendEvent(w, _QM13(_QE13.MouseButtonDblClick, _QP13(5.0, 5.0),
                                 _Qt13.LeftButton, _Qt13.LeftButton, _Qt13.NoModifier))

    def _esc13(w):
        _QA13.sendEvent(w, _QK13(_QE13.KeyPress, _Qt13.Key_Escape, _Qt13.NoModifier))

    for label13, make13, up13, done13 in (
            ("flash", lambda: _FW13(), lambda w: w.show_on(None, 5.0),
             lambda w: w._done),
            ("splash", lambda: _SP13(image_path=None), lambda w: w.start(None, 5.0),
             lambda w: w._done)):
        for how13 in ("escape", "double click"):
            w13 = make13()
            check(f"a NEW {label13} window's cursor is blank from construction",
                  w13.cursor().shape() == _Qt13.BlankCursor)
            up13(w13)
            app.processEvents()
            blank_up13 = w13.cursor().shape() == _Qt13.BlankCursor
            (_esc13 if how13 == "escape" else _dbl13)(w13)
            app.processEvents()
            check(f"{label13}: blank while up, AND {how13} still stands it down "
                  f"(a hidden pointer must not touch the way out, RULES B6)",
                  blank_up13 and done13(w13) is True,
                  f"blank while up {blank_up13}, stood down {done13(w13)}")
            check(f"{label13}: still blank after standing down ({how13})",
                  w13.cursor().shape() == _Qt13.BlankCursor)
            w13.deleteLater()
    check("...and neither installed an application-wide override cursor",
          _QA13.overrideCursor() is None)

    # ------------------------------------------------------------------
    print("\n14. EXIT: two hits close the app, one hit says so")
    # "I was shooting the exit menu and nothing had happened. Let's put Escape
    # as the exit for the app... and when that menu gets shut, the app closes."
    # It takes TWO because a phantom detection closing the app mid-string is
    # the worst accidental action here -- 92 phantoms arrived in 20 seconds
    # once when the window dropped behind something.
    me = Menu()
    me.start(c4)
    ex = next(s for s in me.slots(c4) if s.cls is None and s.stem == "exit")
    exc = ex.picture.center()
    me.shot(modes.ShotEvent(exc.x(), exc.y(), 100.0, 1), c4)
    check("one hit ARMS exit and does not close anything",
          me.exit_armed_at == 100.0 and me.quit_requested is False)
    me.shot(modes.ShotEvent(exc.x(), exc.y(), 100.9, 2), c4)
    check("the second hit asks the app to close",
          me.quit_requested is True and me.exit_armed_at == 0.0)

    # It must LOOK armed: his complaint was that a hit did nothing visible.
    me2 = Menu()
    me2.start(c4)
    before_px = render(lambda p, c: me2.paint(p, c), c4)
    me2.shot(modes.ShotEvent(exc.x(), exc.y(), 50.0, 1), c4)
    after_px = render(lambda p, c: me2.paint(p, c), c4)
    r = ex.rect
    reg = lambda a: a[int(r.top()):int(r.bottom()), int(r.left()):int(r.right())]
    check("an armed EXIT is visibly different on the tile",
          int(luminance(reg(after_px)).sum()) > int(luminance(reg(before_px)).sum()),
          f"{int(luminance(reg(before_px)).sum())} -> "
          f"{int(luminance(reg(after_px)).sum())}")

    # And it does not stay armed for ever, or an arming shot could combine
    # with an unrelated one minutes later.
    me3 = Menu()
    me3.start(c4)
    me3.shot(modes.ShotEvent(exc.x(), exc.y(), 10.0, 1), c4)
    me3.shot(modes.ShotEvent(exc.x(), exc.y(),
                             10.0 + menu_mod.EXIT_ARM_S + 0.5, 2), c4)
    check("an arming hit expires: a much later hit re-arms instead of closing",
          me3.quit_requested is False and me3.exit_armed_at > 0.0,
          f"window {menu_mod.EXIT_ARM_S}s")

    # Leaving the menu disarms it: coming back must not be one shot from gone.
    me4 = Menu()
    me4.start(c4)
    me4.shot(modes.ShotEvent(exc.x(), exc.y(), 10.0, 1), c4)
    me4.stop()
    check("leaving the menu disarms EXIT",
          me4.exit_armed_at == 0.0 and me4.quit_requested is False)

    # The mouse arms and confirms by the same route.
    me5 = Menu()
    me5.start(c4)
    me5.on_click(exc.x(), exc.y(), c4)
    armed5 = me5.exit_armed_at > 0.0 and not me5.quit_requested
    me5.on_click(exc.x(), exc.y(), ctx_for(SW, SH, now=c4.now + 0.5))
    check("the pointer arms and confirms EXIT the same way",
          armed5 and me5.quit_requested is True)

    # ------------------------------------------------------------------
    print("\n15. EXIT reaches the APP, not just the menu's own flag")
    # HE FOUND THIS IN ONE TRY: "shooting exit twice doesn't work, clicking it
    # twice doesn't work either" -- while six checks in section 14 passed.
    # Every one of them asked the MENU whether it had recorded the request;
    # none asked the APP what it did with it. The drain checked `pending`
    # first and returned early when nothing was chosen, which is exactly the
    # EXIT case, so the quit was never reached. A test of the producer that
    # never runs the consumer (RULES C14): the seam is where the fault was.
    closed15 = {"n": 0}
    lab15 = lab12                                  # the real lab, section 12
    lab15.close = lambda *a, **k: closed15.__setitem__("n", closed15["n"] + 1)
    lab15._select_mode(Menu())
    app.processEvents()
    m15 = lab15.layers["modes"].mode
    ex15 = next(s for s in m15.slots(
        lab15.scene._context(time.perf_counter(), 0.0, lab15.target_window.rect()))
        if s.cls is None and s.stem == "exit")
    p15 = ex15.picture.center()
    c15 = lambda: lab15.scene._context(time.perf_counter(), 0.0,
                                       lab15.target_window.rect())

    m15.shot(modes.ShotEvent(p15.x(), p15.y(), time.perf_counter(), 1), c15())
    lab15._drain_menu_choice()
    check("one hit does NOT close the app", closed15["n"] == 0
          and m15.exit_armed_at > 0.0, f"close() called {closed15['n']}x")
    m15.shot(modes.ShotEvent(p15.x(), p15.y(), m15.exit_armed_at + 0.4, 2), c15())
    lab15._drain_menu_choice()
    check("the second hit closes the app, through the app's own drain",
          closed15["n"] == 1, f"close() called {closed15['n']}x")

    # The pointer, by the same route.
    lab15._select_mode(Menu())
    m15b = lab15.layers["modes"].mode
    m15b.on_click(p15.x(), p15.y(), c15())
    lab15._drain_menu_choice()
    armed15 = closed15["n"] == 1 and m15b.exit_armed_at > 0.0
    m15b.on_click(p15.x(), p15.y(), c15())
    lab15._drain_menu_choice()
    check("two clicks close it too", armed15 and closed15["n"] == 2,
          f"close() called {closed15['n']}x")

    # A drill chosen in the same breath must still start: the quit check sits
    # in front of the `pending` path and must not swallow it.
    lab15._select_mode(Menu())
    m15c = lab15.layers["modes"].mode
    tile15 = next(s for s in m15c.slots(c15()) if s.cls is not None)
    t15 = tile15.picture.center()
    m15c.shot(modes.ShotEvent(t15.x(), t15.y(), time.perf_counter(), 3), c15())
    lab15._drain_menu_choice()
    check("...and a normal tile still starts its drill",
          isinstance(lab15.layers["modes"].mode, tile15.cls)
          and closed15["n"] == 2,
          type(lab15.layers["modes"].mode).__name__)

    # And nothing closes the app while the calibration grid is running.
    lab15._select_mode(Menu())
    m15d = lab15.layers["modes"].mode
    m15d.quit_requested = True
    lab15._grid_active = True
    try:
        lab15._drain_menu_choice()
        check("an EXIT request is DROPPED while the grid is running, never "
              "queued to fire afterwards",
              closed15["n"] == 2 and m15d.quit_requested is False,
              f"close() called {closed15['n']}x")
    finally:
        lab15._grid_active = False

    # ------------------------------------------------------------------
    print("\n16. the player-count selector, MENU side: two boxes, one hit, an underline")
    # The plan's sections 1 and 13. Two chrome tiles that set a SETTING and leave
    # the menu up. Nothing is split yet; these checks cover the boxes and the
    # hand-off, and section 17 covers what the APP does with it.
    import dataclasses
    from lsa.display.scene import Scene

    def sel_slots(mm, ctx):
        return [s for s in mm.slots(ctx) if s.players is not None]

    def shoot16(mm, s, ctx, t=100.0, n=1):
        c = s.picture.center()
        return mm.shot(modes.ShotEvent(c.x(), c.y(), t, n), ctx)

    def underlined16(a, s):
        r = s.rect
        y = int(r.bottom() - r.width() * menu_mod.UNDERLINE_FLOOR_FRAC
                - r.width() * menu_mod.UNDERLINE_THICK_FRAC / 2.0)
        x0 = int(r.left() + r.width() * menu_mod.UNDERLINE_INSET_FRAC) + 3
        x1 = int(r.right() - r.width() * menu_mod.UNDERLINE_INSET_FRAC) - 3
        row = luminance(a)[y, x0:x1]
        return bool((row >= layers.GREY_HI.red() - 2).mean() > 0.95)

    c4p1 = dataclasses.replace(c4, players=1)
    c4p2 = dataclasses.replace(c4, players=2)

    m16 = Menu()
    m16.start(c4)
    all16 = m16.slots(c4)
    sel16 = sel_slots(m16, c4)
    # CHANGED 2026-10-19: "1P"/"2P" are gone. The key is the first letter OF THE WORD
    # (S in Single Player, M in Multiplayer), so the tiles carry no glyph; the second
    # was renamed from "Two Players".
    check("CHROME is single, versus, exit in that order, exit LAST, no glyph on the two count tiles (16a)",
          [c.stem for c in menu_mod.CHROME] == ["single", "versus", "exit"]
          and [c.players for c in menu_mod.CHROME] == [1, 2, None]
          and [c.glyph for c in menu_mod.CHROME[:2]] == ["", ""]
          and [c.key for c in menu_mod.CHROME] == [Qt.Key_O, Qt.Key_T, None]
          and all16[-1].stem == "exit" and [s.stem for s in all16[:2]] == ["single", "versus"])
    check("both boxes are laid out as chrome tiles carrying their count, titled One Player and Two Players (16a)",
          [(s.stem, s.players, s.cls) for s in sel16] == [("single", 1, None), ("versus", 2, None)]
          and [s.title for s in sel16] == ["One Player", "Two Players"])
    field16 = m16.playfield(c4)
    check("...inside the playfield and not overlapping any other tile (16a)",
          all(field16.left() - 0.5 <= s.rect.left() and s.rect.right() <= field16.right() + 0.5
              and s.rect.bottom() <= field16.bottom() + 0.5 for s in sel16)
          and all(not a.rect.intersects(b.rect) for i, a in enumerate(all16) for b in all16[i + 1:]))
    hit16 = [m16.tile_at(s.picture.center().x(), s.picture.center().y(), c4) for s in sel16]
    check("...and each is HIT-TESTED: the centre of its picture finds that very tile (16a)",
          all(h is not None and h.stem == s.stem for h, s in zip(hit16, sel16)))
    check("...and their pictures are the shipped menu/single.png and menu/versus.png (16a)",
          all(Menu()._tile_for(s) is not None for s in sel16))

    # DRAWN WITH THEIR LABELS WHEN NO ARTWORK IS PRESENT. "It worked once the
    # art arrived" is not something anyone tests for afterwards.
    _orig_tf16 = Menu._tile_for
    Menu._tile_for = lambda self, s: None
    try:
        m16n = Menu()
        m16n.start(c4)
        no_pic16 = all(m16n._pixmap(s, 549, 71.0) is None for s in sel_slots(m16n, c4))
        a_noart = render(m16n.paint, c4p1)
    finally:
        Menu._tile_for = _orig_tf16
    lum_n = luminance(a_noart)
    check("no artwork present: the selector tiles really have no picture (precondition) (16a)",
          no_pic16)
    for s in sel16:
        pic, cap = s.picture, s.caption
        pi = lum_n[int(pic.top() + pic.height() * .2):int(pic.bottom() - pic.height() * .2),
                   int(pic.left() + pic.width() * .2):int(pic.right() - pic.width() * .2)]
        # the name sits right of the glyph cell and clear of the underline band
        ni = lum_n[int(cap.top() + cap.height() * .2):int(cap.bottom() - s.rect.width() * .035),
                   int(cap.left() + cap.height() * 1.05):int(cap.right() - cap.height() * .2)]
        if s.glyph:
            check(f"no artwork: {s.title!r} draws its big key glyph in the picture area (16a)",
                  int(pi.max()) > 30, f"max luminance {int(pi.max())} over background 10")
        else:
            check(f"no artwork: {s.title!r} has no glyph (its key is the first letter of the word), "
                  f"so the picture area stays empty (16a)", int(pi.max()) <= 12,
                  f"max luminance {int(pi.max())}")
        check(f"no artwork: {s.title!r} draws its LABEL in the caption (16a)",
              int(ni.max()) > 30, f"max luminance {int(ni.max())} over background 10")
    check("no artwork: the whole menu is still red excess 0 (16a)", red_excess(a_noart) == 0)

    # ONE HIT, NO ARMING.
    for s, want in zip(sel16, (1, 2)):
        mm = Menu()
        mm.start(c4)
        out = shoot16(mm, s, c4)
        check(f"one shot on {s.title!r} records the request {want}, with no arming (16b)",
              mm.pending_players == want and out.hit and out.draw_hole is False
              and out.target_id == f"tile:{s.stem}",
              f"pending_players={mm.pending_players} id={out.target_id}")
        check("...and chooses no drill, arms nothing, asks for no quit (16b)",
              mm.pending is None and mm.exit_armed_at == 0.0 and mm.quit_requested is False)
        mc = Menu()
        mc.start(c4)
        c = s.picture.center()
        took = mc.on_click(c.x(), c.y(), c4p1)
        check(f"one CLICK on {s.title!r} records it too, and is not counted as a shot (16b)",
              took is True and mc.pending_players == want and mc.shots == 0 and mc.hits == 0)
    mf = Menu()
    mf.start(c4)
    shoot16(mf, sel16[1], c4, n=1)
    shoot16(mf, sel16[0], c4, n=2)
    check("the FIRST request in a turn wins; a later one does not overwrite it (16b)",
          mf.pending_players == 2)

    # CONSUME-OR-NOTHING.
    mt = Menu()
    mt.start(c4)
    check("take_pending_players with nothing requested returns None and consumes nothing (16c)",
          mt.take_pending_players() is None and mt.pending_players is None)
    shoot16(mt, sel16[1], c4)
    got1 = mt.take_pending_players()
    check("...with a request it returns it and CONSUMES it; a second call is None (16c)",
          got1 == 2 and mt.pending_players is None and mt.take_pending_players() is None)
    shoot16(mt, sel16[0], c4)
    mt.stop()
    check("stop() clears an unacted request: it must not survive the menu (16c)",
          mt.pending_players is None and mt.take_pending_players() is None)
    shoot16(mt, sel16[0], c4)
    mt.start(c4)
    check("start() clears one too (16c)", mt.pending_players is None)
    check("the keys 1, 2 and P are still not claimed by the menu, and record nothing (16g)",
          all(mt.on_key(int(k)) is False for k in (Qt.Key_1, Qt.Key_2, Qt.Key_P))
          and mt.pending_players is None and mt.pending is None)

    # THE UNDERLINE.
    a_p1 = render(Menu().paint, c4p1)
    a_p2 = render(Menu().paint, c4p2)
    check("ctx.players=1: Single Player is underlined, Two Players is not (16e)",
          underlined16(a_p1, sel16[0]) and not underlined16(a_p1, sel16[1]))
    check("ctx.players=2: Two Players is underlined, Single Player is not (16e)",
          underlined16(a_p2, sel16[1]) and not underlined16(a_p2, sel16[0]))
    n1 = sum(underlined16(a_p1, s) for s in all16)
    n2 = sum(underlined16(a_p2, s) for s in all16)
    check("EXACTLY ONE underline across every tile on the menu, in both states (16e)",
          n1 == 1 and n2 == 1, f"{n1} and {n2}")
    for bad in (0, 7, -1):
        ab = render(Menu().paint, dataclasses.replace(c4, players=bad))
        check(f"a setting of {bad} matching no box still draws exactly one underline (16e)",
              sum(underlined16(ab, s) for s in all16) == 1)
    mh = Menu()
    mh.start(c4)
    shoot16(mh, sel16[1], c4)                       # NO drain has run
    a_hit = render(mh.paint, c4p1)                  # the app's setting is still 1
    check("a hit moves the underline in the SAME repaint, with no drain having run (16e)",
          underlined16(a_hit, sel16[1]) and not underlined16(a_hit, sel16[0])
          and mh.selected_players(c4p1) == 2)
    check("...and it asked for exactly one repaint to do so (16e)",
          mh.animated is True and mh.advance(c4p1) is True and mh.advance(c4p1) is False)

    # THE LIT-AREA BUDGET WITH THE SELECTOR TILES (RULES B1b), MEASURED. If this
    # breaches 2% the answer is to report the number, not to dim anything.
    for label, cx in (("players=1", c4p1), ("players=2", c4p2)):
        a = render(Menu().paint, cx)
        lum = luminance(a)
        lit = float((lum > menu_mod.LIT_LUMINANCE).mean())
        print(f"     MEASURED lit area of the whole menu at 4K, {label}: {lit * 100:.4f}% "
              f"({len(all16)} tiles, {len(all16) - len(sel16) - 1} drills)")
        check(f"{label}: the whole menu keeps red excess 0 (16f)", red_excess(a) == 0,
              f"worst {red_excess(a)}")
        check(f"{label}: nothing brighter than the ceiling ({int(lum.max())} vs {CEIL}) (16f)",
              int(lum.max()) <= CEIL)
        check(f"{label}: lit area (>60) {lit * 100:.3f}% is inside the 2% budget (RULES B1b) (16f)",
              0.002 < lit <= 0.02, f"{lit * 100:.4f}%")

    # The plumbing the app copies the setting through.
    check("RenderContext.players defaults to 1, so every old construction site works (16)",
          RenderContext(now=0.0, dt=0.0, rect=QRect(0, 0, 10, 10),
                        stats=SessionStats()).players == 1)
    sc16 = Scene(SessionStats())
    check("Scene.players defaults to 1 and is copied into every context (16)",
          sc16.players == 1 and sc16._context(0.0, 0.0, QRect(0, 0, 10, 10)).players == 1)
    sc16.players = 2
    check("...and follows the scene when it changes (16)",
          sc16._context(0.0, 0.0, QRect(0, 0, 10, 10)).players == 2)

    # ------------------------------------------------------------------
    print("\n17. the player-count selector reaches the APP, not just the menu")
    # THE MOST IMPORTANT CHECK IN THE SLICE. The EXIT bug got through six
    # checks because each asked the MENU what it had recorded and none asked the
    # APP what it did with it (RULES C14). Everything here asserts the APP's own
    # state, through its real drain, on the real lab built in section 12.
    from lsa.settings import AppSettings as _AS17, normalise_players as _norm17

    lab17 = lab12
    logs17: list[str] = []
    _orig_log17 = lab17._log
    lab17._log = lambda msg, *a, **k: (logs17.append(str(msg)), _orig_log17(msg, *a, **k))[1]
    saves17 = {"n": 0}
    # still a no-op as far as his file is concerned (RULES C11): it only counts.
    lab17._schedule_save = lambda *a, **k: saves17.__setitem__("n", saves17["n"] + 1)
    dirty17 = {"n": 0}
    _orig_md17 = lab17.scene.mark_dirty
    lab17.scene.mark_dirty = lambda *a, **k: (dirty17.__setitem__("n", dirty17["n"] + 1),
                                              _orig_md17(*a, **k))[1]
    c17 = lambda: lab17.scene._context(time.perf_counter(), 0.0, lab17.target_window.rect())

    check("at startup the scene mirrors the saved setting, and it is 1 or 2 (17)",
          lab17.scene.players == lab17.settings.players
          and lab17.settings.players in (1, 2))

    def fresh_menu17(players: int):
        lab17.settings.players = players
        lab17.scene.players = players
        lab17._select_mode(Menu())
        app.processEvents()
        return lab17.layers["modes"].mode

    m17 = fresh_menu17(1)
    versus17 = next(s for s in m17.slots(c17()) if s.players == 2)
    single17 = next(s for s in m17.slots(c17()) if s.players == 1)
    shoot16(m17, versus17, c17())
    check("a hit alone changes NOTHING in the app: the drain is what applies it (17d)",
          lab17.settings.players == 1 and lab17.scene.players == 1)
    saves17["n"], dirty17["n"] = 0, 0
    ret17 = lab17._drain_menu_choice()
    check("the app's real drain applies it: settings.players and scene.players are 2 (17d)",
          ret17 is None and lab17.settings.players == 2 and lab17.scene.players == 2,
          f"settings={lab17.settings.players} scene={lab17.scene.players} ret={ret17!r}")
    check("...the menu is STILL UP (it sets a setting, it starts nothing) (17d)",
          lab17.layers["modes"].mode is m17 and m17.pending_players is None)
    check("...the context every layer paints with now carries 2 (17d)", c17().players == 2)
    check("...it asked for a save and marked the scene dirty so the underline moves (17d)",
          saves17["n"] >= 1 and dirty17["n"] >= 1, f"saves={saves17['n']} dirty={dirty17['n']}")
    check("...and said so in plain words in the log (17d)",
          any("two players selected" in l for l in logs17), str(logs17[-2:]))
    shoot16(m17, single17, c17(), n=2)
    lab17._drain_menu_choice()
    check("a hit on Single Player takes it back to 1, and the log says so (17d)",
          lab17.settings.players == 1 and lab17.scene.players == 1
          and any("one player selected" in l for l in logs17))
    shoot16(m17, single17, c17(), n=3)
    lab17._drain_menu_choice()
    check("a hit on the box already selected is harmless and logged as already (17d)",
          lab17.settings.players == 1 and any("(already)" in l for l in logs17))

    # A shot on 2P and a shot on a drill tile in the SAME 200 ms tick: both land.
    m17b = fresh_menu17(1)
    drill17 = next(s for s in m17b.slots(c17()) if s.cls is not None)
    shoot16(m17b, versus17, c17(), n=4)
    shoot16(m17b, drill17, c17(), n=5)
    ret17b = lab17._drain_menu_choice()
    check("a 2P hit and a drill hit in the same tick BOTH take effect: the request does "
          "not return early (17d)",
          lab17.settings.players == 2 and ret17b == drill17.cls.name
          and isinstance(lab17.layers["modes"].mode, drill17.cls),
          f"players={lab17.settings.players} started={ret17b!r}")

    # The other order of the two hits gives the same result.
    m17c = fresh_menu17(2)
    shoot16(m17c, drill17, c17(), n=6)
    shoot16(m17c, single17, c17(), n=7)
    ret17c = lab17._drain_menu_choice()
    check("...and with the two hits in the opposite order (17d)",
          lab17.settings.players == 1 and ret17c == drill17.cls.name)

    # Dropped while the calibration grid is running.
    m17d = fresh_menu17(1)
    shoot16(m17d, versus17, c17(), n=8)
    lab17._grid_active = True
    try:
        lab17._drain_menu_choice()
        dropped_ok = (lab17.settings.players == 1 and m17d.pending_players is None
                      and any("player-count request" in l and "dropped" in l for l in logs17))
    finally:
        lab17._grid_active = False
    lab17._drain_menu_choice()
    check("a player request is DROPPED while the grid runs, and does not fire when it ends (17)",
          dropped_ok and lab17.settings.players == 1 and lab17.scene.players == 1)

    # Applied even with no target window (it only sets a setting).
    m17e = fresh_menu17(1)
    ctx_e = c17()
    shoot16(m17e, versus17, ctx_e, n=9)
    _w17, lab17.target_window = lab17.target_window, None
    try:
        lab17._drain_menu_choice()
        applied_nowin = lab17.settings.players == 2
    finally:
        lab17.target_window = _w17
    check("with no target screen the request is still applied (it needs no screen) (17d)",
          applied_nowin)

    # A request from a replaced menu changes nothing.
    stale17 = Menu()
    stale17.start(c17())
    shoot16(stale17, versus17, c17(), n=10)
    lab17.settings.players = lab17.scene.players = 1
    lab17._select_mode(drill17.cls())
    app.processEvents()
    lab17._drain_menu_choice()
    check("a player request left on a replaced menu changes nothing (17)",
          lab17.settings.players == 1 and lab17.scene.players == 1)

    # The setting itself: default, round trip, and a corrupt value.
    check("AppSettings.players defaults to 1 (17)", _AS17().players == 1)
    with tempfile.TemporaryDirectory() as td17:
        p17 = Path(td17) / "settings.json"
        s17 = _AS17()
        s17.players = 2
        saved17 = s17.save(p17)
        check("players survives a save and a load (a family evening survives a relaunch) (17)",
              saved17 and _AS17.load(p17).players == 2)
    check("normalise_players turns anything that is not 1 or 2 into 1, so one box is always "
          "selected (17)",
          [_norm17(v) for v in (1, 2, "2", 0, 3, -1, None, "x", True, 2.0)]
          == [1, 2, 2, 1, 1, 1, 1, 1, 1, 2])
    lab17._log = _orig_log17
    lab17.scene.mark_dirty = _orig_md17

    # ------------------------------------------------------------------
    # 18. the saved player count reaches the APP's drill selection
    # ------------------------------------------------------------------
    print("\n18. versus: the app applies the count to the next drill, only after setup")
    import lsa.modes as _m18
    lab18 = lab12
    logs18: list[str] = []
    _orig_log18 = lab18._log
    lab18._log = lambda msg, *a, **k: (logs18.append(str(msg)), _orig_log18(msg, *a, **k))[1]
    host18 = lab18.layers["modes"]
    _flag18 = lab18._menu_shown_at_startup
    _players18 = (lab18.settings.players, lab18.scene.players)
    lab18.settings.players = 2
    lab18.scene.players = 2

    # STARTUP STAYS SINGLE-PLAYER whatever is saved.
    lab18._menu_shown_at_startup = False
    lab18._select_mode(_m18.StaticTargets())
    check("setup not over, 2 saved: the drill runs single-player (startup is mode 1, one screen)",
          host18.versus is False and host18.mode2 is None and host18.fallback_note == "")
    check("...and the log says why, so the setting is not silently ignored",
          any("setup is not over" in l for l in logs18), str(logs18[-2:]))

    # THE FIRST DRILL AFTER SETUP SPLITS, for each of the four ready drills.
    lab18._menu_shown_at_startup = True
    for cls18 in (_m18.by_name("static"), _m18.by_name("ball"),
                  _m18.by_name("balls"), _m18.by_name("circles"),
                  _m18.by_name("reaction")):
        lab18._select_mode(cls18())
        check(f"after setup, 2 saved: {cls18.title} runs one instance per half",
              host18.versus is True and type(host18.mode2) is cls18
              and host18.mode2 is not host18.mode and host18.fallback_note == "")
        if cls18.name == "reaction":
            # ACROSS THE SEAM (RULES C14): chosen the way the app chooses it,
            # the two halves must hold ONE shared reveal, or whoever's wait
            # is shorter wins the round.
            check("...and Reaction, chosen through the app, shares ONE reveal across the halves",
                  host18.mode._reveal is host18.mode2._reveal
                  and host18.mode._reveal.seats == 2
                  and (host18.mode.seat, host18.mode2.seat) == (0, 1))
    check("...and the log says two players is in effect",
          any("runs on each half" in l for l in logs18))

    # A DRILL THAT IS NOT READY RUNS SINGLE-PLAYER AND SAYS SO, on screen and in the log.
    class _NotReady18(base.Mode):
        # Every registered drill is versus-ready now (2026-10-19), so the fallback
        # is exercised with a stand-in that has not been measured.
        name = "notready"
        title = "Not Ready"

        def start(self, ctx):
            pass

        def shot(self, ev, ctx):
            return modes.ShotOutcome.miss()

        def paint(self, p, ctx):
            pass

    for name18 in ("notready",):
        cls18 = _NotReady18
        logs18.clear()
        lab18.layers["alert"].title = ""
        lab18._select_mode(cls18())
        check(f"{cls18.title}: not versus-ready, so single-player across the whole screen",
              host18.versus is False and host18.mode2 is None and host18.fallback_note != "")
        check(f"{cls18.title}: ...and the app says so in the LOG",
              any("not ready for two players" in l for l in logs18), str(logs18[-2:]))
        # The title was "TWO PLAYERS NOT AVAILABLE HERE" for eight seconds, with
        # the whole sentence beneath it and the screen dimmed. He said it
        # "interferes with the play" (2026-10-19), so it is four words, one
        # second, no detail and NO veil; the sentence stays in the log above.
        import lsa.tools.camera_lab as _CL18
        al18 = lab18.layers["alert"]
        check(f"{cls18.title}: ...and ON SCREEN (the alert layer), in four words",
              al18.title == _CL18.ONE_PLAYER_ONLY_TITLE and len(al18.title.split()) <= 4
              and al18.detail == "", al18.title)
        check(f"{cls18.title}: ...for one second at most, and it does NOT dim the screen",
              al18.dim is False and 0.0 < al18.until - time.perf_counter() <= 1.0
              and _CL18.ONE_PLAYER_ONLY_SECONDS <= 1.0,
              f"dim={al18.dim} remaining={al18.until - time.perf_counter():.2f}s")

    # It clears itself when the second is up (advance past its deadline).
    lab18.layers["alert"].advance(lab18.scene._context(lab18.layers["alert"].until + 0.01, 0.0, QRect(0, 0, SW, SH)))
    check("the one-player note clears itself when its second is up",
          not lab18.layers["alert"].visible)
    # LIT AREA, MEASURED: the note with no veil against the old dimmed note.
    # Anything put on this screen while shooting is possible competes with the
    # targets for his attention and for the 2% lit-area budget, and the
    # detector is watching it too.
    def _lit18(title, detail, dim):
        lay_ = layers.Alert()
        lay_.show(title, detail, "info", dim=dim)
        img_ = QImage(SW, SH, QImage.Format_RGB32)
        img_.fill(0xFF0A0A0A)
        pn_ = QPainter(img_)
        lay_.paint(pn_, lab18.scene._context(0.0, 0.0, QRect(0, 0, SW, SH)))
        pn_.end()
        a_ = np.frombuffer(img_.constBits(), np.uint8).reshape(SH, img_.bytesPerLine())[:, :SW * 4].reshape(SH, SW, 4)
        red_ = int(cv2.subtract(a_[..., 2], cv2.max(a_[..., 1], a_[..., 0])).max())
        lum_ = cv2.cvtColor(np.ascontiguousarray(a_[..., :3]), cv2.COLOR_BGR2GRAY)
        return red_, float((lum_ > 60).mean())
    new18 = _lit18(_CL18.ONE_PLAYER_ONLY_TITLE, "", False)
    old18 = _lit18("TWO PLAYERS NOT AVAILABLE HERE", "Animated Target is not ready for two players yet: "
                   "it runs single-player across the whole screen", True)
    print(f"         one-player note: lit {new18[1] * 100:.3f}% red {new18[0]}   (the old dimmed note: lit {old18[1] * 100:.3f}% red {old18[0]})")
    check("the one-player note is red excess 0 and inside the 2% lit-area ceiling, and no worse than the old note",
          new18[0] == 0 and new18[1] <= 0.02 and new18[1] <= old18[1], f"{new18} vs {old18}")

    # The versus trail reaches the LOG through the app's own tick (the seam).
    lab18.settings.players = 2
    lab18.scene.players = 2
    lab18._menu_shown_at_startup = True
    lab18._select_mode(_m18.StaticTargets())
    ctx18 = lab18.scene._context(time.perf_counter(), 0.0, QRect(0, 0, SW, SH))
    for i18 in range(_MAG18):
        host18.shot(modes.ShotEvent(60.0, 1500.0, time.perf_counter(), i18 + 1), ctx18)
    logs18.clear()
    lab18._refresh_target_diagnostics()
    check("the HUD tick writes the round's trail to the log: start was drained, the allotment is there",
          any("versus: LEFT has used its allotment" in l for l in logs18), str(logs18[-3:]))
    logs18.clear()
    lab18._refresh_target_diagnostics()
    check("...and a second tick writes it again for no one: it was drained once",
          not any("versus:" in l for l in logs18))
    lab18.settings.players = 2
    lab18.scene.players = 2

    # THE VERDICT SCREEN, ACROSS THE SEAM (C14): the holes go on the app's tick,
    # and the ROUND box / a click / the N key all reach ONE restart function.
    lab18._select_mode(_m18.StaticTargets())
    ctx18 = lab18.scene._context(time.perf_counter(), 0.0, QRect(0, 0, SW, SH))
    calls18: list = []
    _orig_rm18 = lab18._restart_mode

    def _spy_rm18(via="the N key"):
        calls18.append(via)
        return _orig_rm18(via)
    lab18._restart_mode = _spy_rm18

    def _close18():
        for i_ in range(_MAG18):
            host18.shot(modes.ShotEvent(60.0, 1500.0, time.perf_counter(), i_ + 1), ctx18)
            host18.shot(modes.ShotEvent(SW - 60.0, 1500.0, time.perf_counter(), i_ + 1), ctx18)

    _close18()
    lab18.layers["holes"].add(100.0, 100.0, 1, 0)
    check("(setup) a round is over and a hole is on the screen",
          host18.round_over and len(lab18.layers["holes"].holes) == 1)
    logs18.clear()
    lab18._refresh_target_diagnostics()
    check("the app's tick clears the holes when the verdict goes up, through its ONE clearing site",
          len(lab18.layers["holes"].holes) == 0
          and any("holes cleared: a versus round closed" in l for l in logs18), str(logs18[-4:]))
    lab18.layers["holes"].add(100.0, 100.0, 2, 0)
    lab18._refresh_target_diagnostics()
    check("...and only once: a hole shot onto the verdict is not wiped on every later tick",
          len(lab18.layers["holes"].holes) == 1)

    def _paint_verdict18(now_):
        img_ = QImage(SW, SH, QImage.Format_RGB32)
        img_.fill(0xFF0A0A0A)
        pn_ = QPainter(img_)
        host18.paint(pn_, lab18.scene._context(now_, 0.0, QRect(0, 0, SW, SH)))
        pn_.end()

    t18 = 5000.0
    _paint_verdict18(t18)
    rb18 = host18.round_box(lab18.scene._context(t18, 0.0, QRect(0, 0, SW, SH)))
    host18.shot(modes.ShotEvent(rb18.center().x(), rb18.center().y(), t18 + 0.3, 1), ctx18)
    lab18._refresh_target_diagnostics()
    check("a shot at the ROUND box 0.3 s after it appeared starts nothing", calls18 == [] and host18.round_no == 1)
    host18.shot(modes.ShotEvent(rb18.center().x(), rb18.center().y(), t18 + 1.0, 2), ctx18)
    lab18._refresh_target_diagnostics()
    check("a shot at it after 0.8 s starts round 2, via the one restart function",
          calls18 == ["the ROUND box"] and host18.round_no == 2 and not host18.round_over, str(calls18))
    check("...and round 2 starts on a clean screen, deliberately",
          len(lab18.layers["holes"].holes) == 0)
    _close18()
    lab18._refresh_target_diagnostics()
    _paint_verdict18(t18 + 50.0)
    host18.on_click(rb18.center().x(), rb18.center().y(), ctx18)
    lab18._refresh_target_diagnostics()
    check("a click on it starts round 3 by the SAME function",
          calls18 == ["the ROUND box", "the ROUND box"] and host18.round_no == 3, str(calls18))
    _close18()
    lab18._refresh_target_diagnostics()
    lab18._on_target_key(Qt.Key_N)
    check("the N key starts round 4 by the SAME function",
          calls18 == ["the ROUND box", "the ROUND box", "the N key"] and host18.round_no == 4, str(calls18))
    lab18._refresh_target_diagnostics()
    check("...and the log carries the swap that the screen no longer mentions",
          any("sides swapped for round 4" in l for l in logs18), str(logs18[-3:]))
    lab18._restart_mode = _orig_rm18
    src18 = (Path(__file__).resolve().parents[1] / "src" / "lsa" / "tools" / "camera_lab.py").read_text(encoding="utf-8")
    check("exactly ONE statement in the app clears the holes layer (_clear_holes); every other route calls it",
          src18.count('self.layers["holes"].clear()') == 1)

    # The menu is a screen, not a drill: no split, no notice.
    logs18.clear()
    lab18.layers["alert"].title = ""
    lab18._select_mode(Menu())
    check("the menu itself never splits and never apologises",
          host18.versus is False and host18.fallback_note == ""
          and lab18.layers["alert"].title == ""
          and not any("two players" in l.lower() for l in logs18), str(logs18))

    # One player saved: a ready drill is simply single.
    lab18.settings.players = 1
    lab18.scene.players = 1
    lab18._select_mode(_m18.StaticTargets())
    check("one player saved: a ready drill runs whole-screen with no note",
          host18.versus is False and host18.fallback_note == "")

    # The flag is a declaration on exactly the four measured drills.
    check("versus_ready is True on exactly drills 1-9",
          sorted(c.name for c, _k in _m18.registered() if c.versus_ready)
          == ["advancing", "animated", "ball", "balls", "circles", "image", "popup",
              "reaction", "static"])

    # The sensitivity learner refuses while EITHER side is moving.
    import types as _t18
    lab18._select_mode(_m18.StaticTargets())
    host18.mode2 = _m18.by_name("balls")()           # a moving right side, a still left
    _shot18 = _t18.SimpleNamespace(peak=100.0, first_timestamp=time.perf_counter())
    check("versus: the learner refuses a shot while only the RIGHT side is moving",
          lab18._sens_refusal(_shot18) == "animating")
    host18.mode2 = None
    check("...and does not refuse when nothing is moving",
          lab18._sens_refusal(_shot18) is None)

    lab18._log = _orig_log18
    lab18._menu_shown_at_startup = _flag18
    lab18.settings.players, lab18.scene.players = _players18
    lab18._select_mode(Menu())

    # ------------------------------------------------------------------
    # 19. the menu: Single Player / Multiplayer first, S and M as keys while the
    #     menu is showing, the first letter bold and a little brighter
    # ------------------------------------------------------------------
    print("\n19. menu: the two player-count tiles first; S and M are keys ONLY while the menu is showing")
    from PySide6.QtCore import QEvent as _QE19
    from PySide6.QtGui import QKeyEvent as _QKE19
    mm19 = Menu()
    mm19.start(c4)
    sl19 = mm19.slots(c4)
    check("the menu's first two tiles are One Player then Two Players, EXIT is last, drills 1-9 between",
          [s.title for s in sl19[:2]] == ["One Player", "Two Players"] and sl19[-1].stem == "exit"
          and [s.glyph for s in sl19[2:-1]] == [str(i) for i in range(1, 10)],
          str([s.title for s in sl19]))
    check("the two tiles carry NO glyph prefix, while every numbered drill keeps its digit",
          all(s.glyph == "" for s in sl19[:2]) and all(s.glyph for s in sl19[2:-1]))
    check("only those two tiles lead with a bold first letter (the digit cannot live inside a word)",
          [s.lead for s in sl19] == [True, True] + [False] * (len(sl19) - 2))

    # -- the bold letter: what was painted, with what font and what grey -----------
    class _SpyPainter19(QPainter):
        rec: list = []

        def drawText(self, *a):
            if len(a) == 3 and isinstance(a[2], str):
                _SpyPainter19.rec.append((a[2], self.font().bold(), self.pen().color().red(),
                                          self.pen().color().green(), self.pen().color().blue()))
            return super().drawText(*a)

    img19 = QImage(c4.width, c4.height, QImage.Format_RGB32)
    img19.fill(0xFF0A0A0A)
    _SpyPainter19.rec = []
    sp19 = _SpyPainter19(img19)
    mm19.paint(sp19, c4)
    sp19.end()
    rec19 = list(_SpyPainter19.rec)
    s_first = next((r for r in rec19 if r[0] == "O"), None)
    m_first = next((r for r in rec19 if r[0] == "T"), None)
    s_rest = next((r for r in rec19 if r[0] in ("ne Player", "ne")), None)
    m_rest = next((r for r in rec19 if r[0] in ("wo Players", "wo")), None)
    check("the O of One Player and the T of Two Players are drawn BOLD in the lead grey",
          s_first and m_first and s_first[1] and m_first[1]
          and s_first[2] == m_first[2] == menu_mod.LEAD_GREY.red(), str((s_first, m_first)))
    check("...and the REST of each word is regular weight in the ordinary caption grey",
          s_rest and m_rest and not s_rest[1] and not m_rest[1]
          and s_rest[2] == m_rest[2] == layers.GREY_MID.red(), str((s_rest, m_rest)))
    # DISTINGUISHABLE, not merely different (his review, 2026-10-19: 60 against 58 is
    # two units in sixty, invisible from the firing line). The bar is a number.
    MIN_CONTRAST19 = 1.15       # lead / caption luminance; 58 -> 72 is 1.24, 58 -> 60 is 1.03
    lead19, cap19 = menu_mod.LEAD_GREY.red(), layers.GREY_MID.red()
    check(f"...the lead grey is DISTINGUISHABLE from the caption grey (at least {MIN_CONTRAST19}x, "
          f"not one unit), neutral, and not above the artwork ceiling",
          lead19 / cap19 >= MIN_CONTRAST19 and lead19 <= ceiling.figure_level() + 1
          and menu_mod.LEAD_GREY.red() == menu_mod.LEAD_GREY.green() == menu_mod.LEAD_GREY.blue(),
          f"lead {lead19} vs caption {cap19} = {lead19 / cap19:.2f}x; ceiling {ceiling.figure_level()}")
    # The digits leading the numbered drill tiles are bold in GREY_HI: the menu's two rows must agree.
    digit_rec = next((r for r in rec19 if r[0] == "1"), None)
    check("...and it is the SAME grey as the bold digit leading each numbered drill tile (the rows agree)",
          digit_rec is not None and digit_rec[1] and digit_rec[2] == lead19,
          f"digit {digit_rec} vs lead {lead19}")
    check("...and no other tile name has a bold letter: the drill names are all regular, in the caption grey",
          all((not r[1]) for r in rec19 if r[0] in [s.title for s in sl19[2:-1]]))
    a19 = render(Menu().paint, c4p1) if "c4p1" in dir() else render(mm19.paint, c4)
    lum19 = luminance(a19)
    print(f"         menu at 4K: red excess {red_excess(a19)}, lit above luminance 60: {float((lum19 > 60).mean()) * 100:.3f}%; "
          f"lead letter grey {menu_mod.LEAD_GREY.red()} (bold) vs caption grey {layers.GREY_MID.red()}")
    check("the whole menu with the bold letters: red excess exactly 0 and inside the 2% lit-area ceiling",
          red_excess(a19) == 0 and float((lum19 > 60).mean()) <= 0.02,
          f"red {red_excess(a19)}, lit {float((lum19 > 60).mean()) * 100:.3f}%")

    # -- THE LABEL'S LEFT INSET, asserted on RENDERED PIXELS (2026-10-20) ------------------
    # His screenshot: on the two player-count tiles "the S and the M are touching" the
    # border while the numbered tiles have a clear margin. The cause was the lost 1P/2P
    # glyph, which had supplied the inset. One definition now (`Menu.first_char_x`).
    ink19 = a19[..., :3].max(axis=2)
    firsts19, lasts19, fit_issues19 = {}, {}, []
    for s_ in sl19:
        r_, cap_ = s_.rect, s_.caption
        y0_, y1_ = int(cap_.top() + cap_.height() * 0.1), int(cap_.bottom() - cap_.height() * 0.1)
        bw_ = max(1, int(r_.width() * 0.006)) * 3
        reg_ = ink19[y0_:y1_, int(r_.left()):int(r_.right())]
        cols_ = _np_menu.where((reg_[:, bw_ + 2:-(bw_ + 2)] > 20).any(axis=0))[0] if reg_.size else []
        if len(cols_):
            firsts19[s_.title] = int(cols_.min()) + bw_ + 2
            lasts19[s_.title] = int(cols_.max()) + bw_ + 2
            if lasts19[s_.title] > r_.width() - bw_ - cap_.height() * 0.06:
                fit_issues19.append(f"{s_.title}: ink to {lasts19[s_.title]} of {r_.width():.0f}")
    numbered19 = [v_ for k_, v_ in firsts19.items() if k_ not in ("One Player", "Two Players", "Exit")]
    med19 = sorted(numbered19)[len(numbered19) // 2]
    print(f"         first-ink column from each tile's left edge: numbered median {med19}, "
          f"One Player {firsts19.get('One Player')}, Two Players {firsts19.get('Two Players')}")
    check("every tile's caption starts at the same inset from its own left border, on the rendered pixels: the two "
          "player-count tiles match the numbered tiles (within 3 px at 4K), and none touches the border",
          all(abs(firsts19[t_] - med19) <= 3 for t_ in ("One Player", "Two Players"))
          and all(abs(v_ - med19) <= 3 for v_ in numbered19) and med19 >= 10,
          str(firsts19))
    check("the longest caption still FITS inside its tile after the inset (rendered ink never reaches the right border)",
          not fit_issues19, str(fit_issues19))

    # -- S and M through the REAL window, on and off the menu ------------------------
    lab19 = lab18
    host19 = lab19.layers["modes"]
    win19 = lab19.target_window
    check("(precondition) the target window exists for the key checks", win19 is not None)
    emitted19: list = []
    win19.key_pressed.connect(emitted19.append)

    def _press19(key):
        win19.keyPressEvent(_QKE19(_QE19.KeyPress, key, Qt.NoModifier))

    stats19 = lab19.scene.get("stats")
    lab19.settings.players = 2
    lab19.scene.players = 2
    lab19._menu_shown_at_startup = True
    lab19._select_mode(Menu())
    check("(precondition) the menu is the active mode", isinstance(host19.mode, Menu))
    calls19: list = []
    real_selftest19 = lab19.start_self_test
    lab19.start_self_test = lambda: calls19.append("selftest")
    s_before = stats19.visible
    emitted19.clear()
    _press19(Qt.Key_O)
    check("ON the menu, O selects One Player and reaches neither the stats toggle, the menu key nor the self check",
          host19.mode.pending_players == 1 and stats19.visible == s_before and not emitted19 and not calls19,
          f"pending {host19.mode.pending_players}, stats {s_before}->{stats19.visible}, emitted {emitted19}")
    lab19._drain_menu_choice()
    check("...and the app's drain applies it through the same route a tile hit takes: the setting is 1",
          lab19.settings.players == 1 and host19.mode.pending_players is None)
    _press19(Qt.Key_T)
    check("ON the menu, T selects Two Players and does NOT reach the app (no self check, no menu re-entry)",
          host19.mode.pending_players == 2 and not emitted19 and isinstance(host19.mode, Menu) and not calls19)
    lab19._drain_menu_choice()
    check("...and the drain applies it: the setting is 2", lab19.settings.players == 2)
    # S and M are given back: the menu does not claim them
    emitted19.clear()
    s_before = stats19.visible
    _press19(Qt.Key_S)
    check("ON the menu, S is the stats HUD again and nothing else (it toggles it and selects nothing)",
          stats19.visible != s_before and host19.mode.pending_players is None)
    _press19(Qt.Key_S)
    emitted19.clear()
    _press19(Qt.Key_M)
    check("ON the menu, M reaches the app and restarts the menu, exactly as before this change (not a player-count key)",
          emitted19 == [Qt.Key_M] and host19.mode.pending_players is None, str(emitted19))
    # off the menu
    lab19._select_mode(modes.StaticTargets())
    check("(precondition) a drill is the active mode", not isinstance(host19.mode, Menu))
    s_before = stats19.visible
    emitted19.clear()
    calls19.clear()
    _press19(Qt.Key_S)
    check("OFF the menu, S toggles the stats HUD", stats19.visible != s_before)
    _press19(Qt.Key_S)
    check("...and toggles it back (the key was never swallowed)", stats19.visible == s_before)
    emitted19.clear()
    _press19(Qt.Key_M)
    check("OFF the menu, M reaches the app (key_pressed) ...", emitted19 == [Qt.Key_M], str(emitted19))
    lab19._on_target_key(Qt.Key_M)
    check("...which opens the menu", isinstance(host19.mode, Menu))
    lab19._select_mode(modes.StaticTargets())
    emitted19.clear()
    calls19.clear()
    n_pending19 = host19.mode.pending_players if hasattr(host19.mode, "pending_players") else None
    _press19(Qt.Key_T)
    lab19._on_target_key(Qt.Key_T)
    check("OFF the menu, T does NOTHING (it is no longer the self check, and the menu does not claim it here)",
          not calls19 and n_pending19 is None, str(calls19))
    _press19(Qt.Key_A)
    check("A runs the SELF CHECK (the regression this re-keying could cause: the advice on screen says A)",
          calls19 == ["selftest"], str(calls19))
    holes19 = lab19.layers["holes"]
    holes19.add(100.0, 100.0, 1, 0)
    cam19 = lab19.layers["hits"]
    cam19.visible = False       # it stands down once calibrated: a broadcast key reaches it anyway
    cam19.add(0.5, 0.5, 0)
    calls19.clear()
    _press19(Qt.Key_A)
    check("...and A does NOT clear the holes any more", len(holes19.holes) == 1 and calls19 == ["selftest"])
    calls19.clear()
    _press19(Qt.Key_K)
    lab19._on_target_key(Qt.Key_K)
    # 2026-10-21: K was freed by the rebinding and then CLAIMED by the Keyboard
    # Shortcuts mode. The property worth holding is no longer "K does nothing" but
    # "K opens that screen and starts no measurement" -- pressing it must never
    # reach the self check or the self-calibration, which take the screen for
    # 10-25 seconds and would read from the firing line as the app freezing.
    _shortcuts19 = modes.by_key(Qt.Key_K)
    check("K opens the Keyboard Shortcuts screen and starts no measurement (self-calibrate moved to C, the self check to A)",
          not calls19 and _shortcuts19 is not None
          and _shortcuts19.name == "shortcuts", str(calls19) + " / " + str(_shortcuts19))
    calls19.append("selftest")
    _press19(Qt.Key_E)
    check("E clears the bullet holes AND the hit marks together, both layers, in one press, and runs no self check",
          len(holes19.holes) == 0 and len(cam19.marks) == 0 and calls19 == ["selftest"], f"{len(holes19.holes)} {len(cam19.marks)}")
    lab19.start_self_test = real_selftest19
    win19.key_pressed.disconnect(emitted19.append)
    lab19._select_mode(Menu())

    passed = all(checks)
    print("\n" + (f"ALL {len(checks)} CHECKS PASSED" if passed
                  else f"SOME CHECKS FAILED ({checks.count(False)} of {len(checks)})"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
