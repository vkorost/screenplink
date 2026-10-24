"""The shoot-to-select menu: one tile per drill, shot or keyed to start it.

Asked for as: at startup a grid of squares, one per shooting mode, each with a
picture of what the mode looks like and the key that starts it, shot to choose.
The plan is `requirements/menu_shooting_mode.md` (sections 2, 3 and 6). This module is its
piece 1 only: the tiles, the layout, choosing by shot, and the deferred hand-off.
The MENU corner tile and its two-shot arm, the corner ranking, the mouse, the
app integration (draining `pending`) and what the EXIT tile does are later
pieces and are NOT here.

THE KEY IS M. He asked for it by name on 2026-10-18: "lets wire letter M for
Menu". M was taken -- `CameraPane.on_key` cycled the pane's corner with it --
so that binding moved to J in the same change, because a layer claiming a key
SHADOWS the app silently: layers get first refusal on every key, so a mode
registered on an occupied key would either be shadowed by the layer or shadow
it, with no error either way -- which is how G once spent a session switching
the background to grey instead of calibrating.

The pane went to J, which is unused anywhere in src. Its corner cycle is nearly
dead weight in any case: `CameraPane.wants_paint` is `aiming and visible`, so
since 2026-10-17 it draws nothing on the target screen unless he is aiming. It
was moved rather than deleted because removing a capability is his call.

E, which this mode used until he asked for M, is now free and answered by
nothing.

NOTHING IS HARD-CODED ABOUT WHICH MODES EXIST. The tiles are whatever
`modes.registered()` says, minus this mode, ordered by key glyph (digits, then
letters, then keyless modes). `test_artwork` once pinned twelve figures by name
and blocked the thirteenth; a mode added next year gets a tile with no change
here. A mode with no picture in `menu/` still gets one, drawn as its key glyph
(large, in the picture area) and its name.

THIS MODULE NEVER SWITCHES MODES. `ModeHost.set_mode` calls `stop()` on the
outgoing mode, so switching from inside this mode's own `shot()` would stop the
object that is currently executing and then unwind back into it. Python does not
fail-fast on that the way Qt did for `FlashWindow._give_up` (0xc0000409, found
in minutes); it carries on with a mode that has been stopped, which is worse.
Same remedy as `_give_up`: `shot()` only RECORDS the choice in `pending`; the app
calls `take_pending()` on its next tick, once the stack has unwound, and hands
the class to `camera_lab._select_mode`. THAT is the one route for switching: it
announces the drill in voice, logs the switch and resets session state, and a
second route that skipped all three is how this repo got two control caches.
That is why nothing here may reference `set_mode` or `restart` at all (a check in
`tests/test_menu.py` reads this file's syntax tree and fails if one appears).
`take_pending()` either returns the choice and consumes it, or returns None and
consumes nothing, so "refused" and "nothing was chosen" are never confused
(RULES C1). The first choice in a turn wins (two pistols can land two shots in
one frame); a later one does not overwrite it.

THE CALLER OWNS THE STALENESS TEST. A choice left over from a menu that has since
been replaced must not replace whatever is running now: `stop()` clears it, and
the app must take a choice only while this menu is still `host.mode`.

A KEY IS NOT CLAIMED UNTIL THE APP CAN ACT ON IT. `on_key` returns False for a
tile's key. Layers get first refusal on every key, and `TargetWindow.keyPressEvent`
does not emit `key_pressed` for a key a layer claims, so returning True would
bypass `camera_lab._on_target_key`'s `by_key` path -- the only code that can act
on it -- while nothing drains `pending`: the shooter would be stuck on the menu
with F11 (which closes the whole app) as the only way out. The menu is the one
mode whose entire job is letting him leave. A mode must not claim a key it cannot
act on; do not "tidy" this back to True until the integration that calls
`take_pending()` exists and is tested. Nothing is recorded for a key either: the
`by_key` path switches the mode itself, and a latched `pending` would make the
first-choice-wins rule ignore every later shot.

CHROME TILES. A tile may have no mode behind it (`CHROME`: the two player-count
boxes, then EXIT). It is laid out, drawn and hit-tested like any other, and
`Slot.cls` is None. Its picture is found by its file stem (`Slot.stem`) where a
drill's is found by its key glyph; with no picture it falls through to its glyph
and label, so the selector works with no artwork present.

THE PLAYER-COUNT SELECTOR (`Chrome.players`): a hit on "One Player" or
"Two Players" (or the keys O and T, while the menu is showing) records `pending_players` and leaves the menu up. The app drains it with
`take_pending_players()` on the same tick as `pending`, applies it to its own
setting, and the selected box is UNDERLINED (the pending request wins over the
app's setting for the repaint, so the hit shows at once). Nothing consumes the
count yet. One hit, no arming -- see `_players_hit`.

SIZES. Every tile dimension is a fraction of the screen width (the tile) or of
the tile (gap, caption, border), and every text size goes through
`px(ctx, at_1080)` and is then FITTED to its cell with the painter's own font
metrics, never an assumed advance (a key legend once overran his screen by 848
px). Nothing here is sized in pixels (RULES B5): 42-inch 4K now, a 75-inch TV
and a projector later.

BRIGHTNESS. Everything is neutral grey, red excess 0 by construction: the
pictures are drawn as white at coverage v/255 exactly as the figures are
(`modes/sprite.py`), the border and text come from the palette in
`display/layers.py`. The pictures arrive re-checked against the ceiling
(`artwork/tiles.py`). The resting border is `GREY_MISS`, under luminance 60, so
it does not add to the lit-area budget (RULES B1b, <= 2% above 60); the pictures
alone were measured at 1.176% at one seventh of the width (see TILE_WIDTH_FRAC).

OCCLUDERS. The pictures are white at coverage v/255 on a TRANSPARENT ground, so the
bullet holes (which persist across a mode switch) would show through them: the
fault he photographed twice on the animated figure. `occluders` announces every
tile's whole rectangle as solid, so the holes layer leaves it out.

REPAINTS. Static: it asks for none while idle. A choice being recorded asks for
exactly one so the chosen tile's border brightens until the switch.

SHOTS HERE ARE NOT REWOUND. Other modes rewind a moving target by the shot's age
(`Mode.shot_age`); the tiles never move, so where they were a few frames ago is
where they are, and the shot's own position is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QKeySequence, QPainter, QPen, QPixmap

from ..artwork import ceiling, tiles
from ..audio.mixer import SELECT_SOUND
from ..display.layers import GREY_HI, GREY_MID, GREY_MISS, mono, px
from ..display.scene import Occluder, RenderContext
from . import base
from .base import CHROME_ARM_S, Mode, ShotEvent, ShotOutcome, register

# A tile's width as a fraction of the SCREEN WIDTH. MEASURED, not chosen by eye,
# from the regenerated PNGs (menu/measurements.json): all twelve pictures on
# screen against the ~2% of the screen above luminance 60 that RULES B1b allows
# while shooting is possible --
#     1/6   1.601%   0.40% left for border, key glyph and name
#     1/7   1.176%   0.82% left                                     <- this
# (The menu draws eleven of the twelve today: `menu.png` is a later piece.) At
# 1/7 a tile is ~549 px on a 42-inch 4K panel. It is the DEFAULT: when the tiles
# do not fit the playfield they shrink (`SHRINK`), never grow.
TILE_WIDTH_FRAC = 1.0 / 7.0
MIN_TILE_FRAC = 0.05          # below this a tile is not readable from the firing line
SHRINK = 0.93
GAP_FRAC = 0.10               # between tiles, of the tile width
CAPTION_FRAC = 0.20           # the strip under the picture, of the tile width
BORDER_FRAC = 0.003           # thin: 1.9 px at 640, and it is dim at rest
PENDING_BORDER_X = 3.0        # the chosen tile's border, times the resting width

HEADING_AT_1080 = 34       # "MAIN MENU" above the grid, fitted to the gap
NAME_AT_1080 = 20             # starting sizes, fitted DOWN to the cell
GLYPH_AT_1080 = 26
BIG_GLYPH_AT_1080 = 96        # a tile with no picture: its glyph, large
FONT_FLOOR = 9                # `px`'s own floor

# A caption below this fraction of its starting size is wrapped onto two lines
# instead of being shrunk further. 0.6 is where the suite draws the legibility
# line, and it is ONE definition: the check reads this constant rather than
# carrying its own copy, so the drawn text and the bar it is judged against can
# never drift apart.
NAME_FLOOR_FRAC = 0.6

# The luminance above which a pixel counts toward the 2% lit-area budget (RULES B1b).
LIT_LUMINANCE = 60


def caption_margin() -> int:
    """How far the tile names sit UNDER the lit threshold, at the ceiling this
    process started with. The names are drawn at `GREY_MID`, which the palette
    derives as a fixed fraction of the artwork ceiling (`layers.grey_at(0.75)`),
    so the margin is a function of the artwork and not a constant:

        figure ceiling (file)   on screen   GREY_MID   margin under 60
              71                   78          58            2   <- today
              52                   60          45           15
              40 (MIN_LEVEL)       48          36           24
              no artwork (78)      85          64           -4   <- would be LIT

    The ceiling moved 71 -> 52 -> 71 in one morning when four figures landed. A
    LOWER ceiling only makes the captions dimmer (they stay inside the budget, and
    hide the dot less) and costs legibility from the firing line: at 52 they are
    45 against a background of 10, still readable but the first thing to look at
    if he reports the names are hard to see. Only a ceiling above ~80 -- which
    needs NO artwork on disk at all -- pushes them over 60 and into the budget.
    Computed at import like `layers.CEILING`, so it holds for the whole process.
    """
    return LIT_LUMINANCE - GREY_MID.red()


class Chrome:
    """A tile with no mode behind it: part of the menu, not a drill."""

    def __init__(self, stem: str, title: str, glyph: str = "",
                 players: "int | None" = None, key: "int | None" = None) -> None:
        self.stem = stem          # the picture's file stem in menu/
        self.title = title
        self.glyph = glyph        # what the tile shows as its key
        # The Qt key that chooses this tile WHILE THE MENU IS SHOWING (see
        # `Menu.on_key` and `DELIBERATE_KEY_OVERRIDES`). None: no key.
        self.key = key
        # Set only on the player-count selector tiles: the count a hit on this
        # tile asks for. None for every other tile.
        self.players = players


# Chrome tiles, drawn after the drills in this order.
#
# THE PLAYER-COUNT SELECTOR is the first two entries: a mutually exclusive pair
# of boxes that set a SETTING and leave the menu up (plan
# `requirements/two_player_versus_plan_01.md` sections 1 and 13). Their key
# glyphs are "1P" and "2P", NOT digits: the drills keep 1-9 exactly as they are.
# EXIT STAYS LAST.
#
# EXIT CLOSES THE APP, and it is labelled Esc because Escape does the same.
# His instruction, 2026-10-18, after shooting it and getting nothing: "I was
# shooting the exit menu and nothing had happened. Let's put Escape as the
# exit for the app. So then we will write Esc key for the exit menu; and when
# that menu gets shut, the app closes."
#
# IT TAKES TWO SHOTS, and that is not me hedging his instruction. A phantom
# detection closing the app mid-string is the worst accidental action
# available here, and this project has logged 92 phantoms in twenty seconds
# when the window dropped behind something. The first shot ARMS it and the
# tile says so in words; the second closes. Nothing is silent -- "I shot it
# and nothing happened" is the complaint this is answering, so the first
# shot must visibly do something.
#
# THE TWO PLAYER-COUNT TILES COME FIRST on the screen, before drill 1, and EXIT
# stays last (his instruction, 2026-10-19). They carry NO glyph prefix: the key
# is the first letter OF THE WORD, bold and slightly brighter -- the S in
# "One Player" and the T in "Two Players" (`Slot.lead`, `LEAD_GREY`). A digit
# cannot live inside a word, so the numbered drills keep their "1 Static Targets"
# form. Those letters are real keys, claimed only while the menu is showing.
#
# RENAMED AND RE-KEYED 2026-10-20 (his words: "lets change Single Player and
# Multiplayer to One Player and Two Players; then highlight O and T and use O and
# T as shortcuts to the mode toggles instead of S and M"). It was S and M for a
# day; S is the stats HUD toggle and M opens this menu, so those two needed a
# declared override. O and T are FREE: the self check, which used T, moved to C
# the same day. Every key therefore has exactly one owner and there is no
# exception left to declare.
CHROME = (Chrome("single", "One Player", players=1, key=Qt.Key_O),
          Chrome("versus", "Two Players", players=2, key=Qt.Key_T),
          Chrome("exit", "Exit", glyph="Esc"))

# THE KEYS THIS MENU CLAIMS, as data, ONLY while it is the active mode. Neither is
# used by anything else in the app (see the note above CHROME), so this is NOT an
# override of anything: `DELIBERATE_KEY_OVERRIDES` is empty and stays empty. It
# is kept, empty, because the collision check in `tests/test_ui_smoke.py` reads it
# and fails on any key claimed twice or claimed by the menu without being listed
# here; an empty list is only honest while that check can still fail, and it is
# proven to (section 19 introduces a collision on purpose).
MENU_OWN_KEYS = {c.key: f"{c.title} (player count)" for c in CHROME if c.key is not None}
DELIBERATE_KEY_OVERRIDES: dict = {}

# THE FIRST LETTER OF A PLAYER-COUNT TILE'S NAME: bold, and the SAME grey as the
# bold digit that leads every numbered drill tile (`GREY_HI`, the palette's own
# brightest, at or below the artwork ceiling `ceiling.figure_level()`, RULES B10),
# so the menu's two rows agree. It was first capped at the lit threshold (60) -- two
# units above the caption's 58, invisible from the firing line -- but the budget
# counts AREA above 60, and this is one glyph on each of two tiles: MEASURED at
# about 0.01% of the screen (tests/test_menu.py section 19 prints the menu's lit %
# with it). Neutral, so red excess is exactly 0. The caption stays at `GREY_MID`.
LEAD_GREY = QColor(GREY_HI.red(), GREY_HI.green(), GREY_HI.blue())

# The selector's underline, as fractions of the TILE WIDTH (RULES B5). It is
# drawn INSIDE the tile's own rectangle, in the padding under the caption, so
# the tile's occluder still covers it and no bullet hole shows through. Thin on
# purpose: at 1/7 of a 4K width a tile is ~549 px, so these are ~6.6 px thick
# and ~0.09% of the screen for both boxes -- a rounding error in the lit-area
# budget (RULES B1b). Its grey is the palette's `GREY_HI`, never a literal
# (RULES B10): the brightest thing the menu draws, because it is the one fact
# that must read from the firing line.
UNDERLINE_THICK_FRAC = 0.012
UNDERLINE_INSET_FRAC = 0.05       # in from each side of the tile
UNDERLINE_FLOOR_FRAC = 0.006      # up from the tile's bottom edge

# How long an armed EXIT stays armed. Long enough to fire again deliberately,
# short enough that an arming shot cannot combine with an unrelated one a
# minute later.
#
# ONE DEFINITION, shared with the MENU button that every drill now carries
# (`modes.base.CHROME_ARM_S`). Both are two-hit chrome for the same reason --
# a phantom must not be able to take the shooter out of what they are doing --
# so they must agree, and the way two numbers that must agree stop agreeing is
# being written down twice.
EXIT_ARM_S = CHROME_ARM_S


def glyph_of_key(key: "int | None") -> str:
    """The one-character name of a key as shown on a tile ('' when it has none).

    Printable ASCII is its own glyph (Qt's letter and digit key codes ARE their
    upper-case ASCII). Anything else (F-keys, Home) falls back to Qt's name for
    it, which will not match a one-character tile file and so draws without a
    picture -- visible, not wrong.
    """
    if key is None:
        return ""
    if 0x20 < key < 0x7F:
        return chr(key).upper()
    try:
        return QKeySequence(int(key)).toString()
    except Exception:                              # display only; never fatal
        return ""


def tile_layout(area: QRectF, n: int, screen_w: float,
                frac: float = TILE_WIDTH_FRAC) -> "list[QRectF]":
    """Where `n` tiles go inside `area`: fewest rows, centred, row-major.

    Pure, so the geometry is tested without a mode. A tile is `screen_w * frac`
    wide and that much tall plus a caption strip. When the grid does not fit it
    shrinks by `SHRINK` until it does, down to `MIN_TILE_FRAC`; below that it is
    forced into as many columns as fit and may overflow the bottom (a screen
    with that many modes wants a different menu, and clipping is the honest
    failure, not an overlap).
    """
    if n <= 0:
        return []
    frac = float(frac)
    fit = None
    for _ in range(60):
        s = screen_w * frac
        g = s * GAP_FRAC
        th = s * (1.0 + CAPTION_FRAC)
        max_cols = int((area.width() + g) // (s + g))
        max_rows = int((area.height() + g) // (th + g))
        for rows in range(1, max_rows + 1):
            cols = math.ceil(n / rows)
            if cols <= max_cols:
                fit = (rows, cols)
                break
        if fit is not None:
            break
        if frac <= MIN_TILE_FRAC:
            cols = max(1, max_cols)
            fit = (math.ceil(n / cols), cols)
            break
        frac = max(MIN_TILE_FRAC, frac * SHRINK)
    rows, cols = fit
    block_h = rows * th + (rows - 1) * g
    y0 = area.top() + (area.height() - block_h) / 2.0
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        in_row = min(cols, n - r * cols)
        row_w = in_row * s + (in_row - 1) * g
        x0 = area.left() + (area.width() - row_w) / 2.0
        out.append(QRectF(x0 + c * (s + g), y0 + r * (th + g), s, th))
    return out


@dataclass(frozen=True)
class Slot:
    """One tile on screen: which mode (None for a chrome tile), what it is called,
    and where. `stem` is set only for a chrome tile and names its picture."""

    cls: "type | None"
    glyph: str
    title: str
    rect: QRectF
    stem: str = ""
    # The player count a hit on this tile requests (selector tiles only).
    players: "int | None" = None

    @property
    def lead(self) -> bool:
        """Draw the first letter of the title bold and brighter: the key."""
        return self.players is not None

    @property
    def picture(self) -> QRectF:
        r = self.rect
        return QRectF(r.left(), r.top(), r.width(), r.width())

    @property
    def caption(self) -> QRectF:
        r = self.rect
        return QRectF(r.left(), r.top() + r.width(), r.width(), r.height() - r.width())

    def contains(self, x: float, y: float) -> bool:
        r = self.rect
        return r.left() <= x < r.right() and r.top() <= y < r.bottom()


def wrap_title(title: str) -> "tuple[str, ...]":
    """A drill's name as one line, or two when one line would be too small.

    Split at the space that leaves the two halves most equal, so neither line
    is a lone short word that forces the other to shrink anyway. A name with
    no space stays one line -- there is nowhere to break it.
    """
    words = title.split()
    if len(words) < 2:
        return (title,)
    best, gap = 1, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        d = abs(len(a) - len(b))
        if gap is None or d < gap:
            best, gap = i, d
    return (" ".join(words[:best]), " ".join(words[best:]))

def fit_text(p: QPainter, text: str, bold: bool, start_px: int,
             max_w: float, max_h: float) -> "tuple[int, str]":
    """Pick the largest font size <= start_px at which `text` fits the cell,
    set it on the painter, and return (size, text).

    Measured with the PAINTER's font metrics: the check that caught the 848 px
    legend overrun measures with exactly those, and a guessed advance is what
    put the verdict's action text off the edge of its own box. If even the
    floor does not fit, the text is elided rather than overrun.
    """
    size = max(FONT_FLOOR, int(start_px))
    while True:
        p.setFont(mono(size, bold))
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text), fm.height()
        if (w <= max_w and h <= max_h) or size <= FONT_FLOOR:
            break
        size -= 1
    if w > max_w:
        text = fm.elidedText(text, Qt.ElideRight, max(1, int(max_w)))
    return size, text


@register(key=Qt.Key_M)
class Menu(Mode):
    """A grid of the drills. Shoot one, or press its key, to start it."""

    name = "menu"
    title = "Main Menu"
    summary = "Shoot a square, or press its key, to start that drill."

    CACHE_BOUND = 40          # pixmaps kept; there are ~10 tiles and one size

    @property
    def animated(self) -> bool:
        """True only for the one tick that repaints a recorded choice. Derived
        from the flag `advance` reads, so the two cannot disagree; idle it is
        False and a static menu asks for no repaints at all."""
        return self._dirty

    def __init__(self) -> None:
        self.visible = True
        # The class chosen by a shot or a key, waiting for the app's next tick.
        # PUBLIC on purpose: this is the hand-off (see the module docstring).
        self.pending: "type[Mode] | None" = None
        # EXIT: when it was armed (0.0 = not armed), and whether the second
        # hit has landed. Declared here and assigned unconditionally -- a
        # getattr default has silently disabled a feature three times here.
        self.exit_armed_at = 0.0
        self.quit_requested = False
        # A player-count request (1 or 2) from a selector tile, waiting for the
        # app's next tick. The SETTING lives on the app; this is only the hand-
        # off, and it is also what makes a hit visible in the same repaint
        # (`selected_players`), before the app's 200 ms drain has run. Assigned
        # unconditionally (RULES C8). FIRST REQUEST IN A TURN WINS, as `_choose`
        # does: two pistols can land two shots in one frame, and a later one
        # does not overwrite it.
        self.pending_players: "int | None" = None
        self.shots = 0
        self.hits = 0
        self._dirty = False
        self._pix: "dict[tuple, QPixmap | None]" = {}
        self._solid: "QPixmap | None" = None      # 1x1 opaque, stretched over a tile

    # -- lifecycle ------------------------------------------------------------

    def start(self, ctx: RenderContext) -> None:
        self.pending = None
        self.pending_players = None
        self.exit_armed_at = 0.0
        self.quit_requested = False
        self.shots = self.hits = 0
        self._dirty = False
        self._pix = {}

    def stop(self) -> None:
        self.pending = None
        # An unacted player request must not survive the menu.
        self.pending_players = None
        # AN ARMED EXIT DOES NOT SURVIVE THE MENU. Leaving on a drill and
        # coming back must not find the app one shot from closing.
        self.exit_armed_at = 0.0
        self.quit_requested = False
        self._dirty = False
        self._pix = {}

    def moving_at(self, t: float) -> bool:
        """The menu is still: nothing on it moves, and a recorded choice only
        brightens a border for a moment before the switch. If a hit flash is
        ever added, this must return True for its duration or the sensitivity
        learner will learn from shots fired at a moving thing."""
        return False

    # -- the tiles ------------------------------------------------------------

    def slots(self, ctx: RenderContext) -> "list[Slot]":
        """The tiles for the modes registered RIGHT NOW, laid out for this
        screen. Recomputed on every call: it is cheap, and a cached list is how
        a mode registered after the first call would never get a tile."""
        # A SCREEN GETS NO TILE. The menu lists DRILLS -- things to shoot at --
        # and the Keyboard Shortcuts page (K) is not one: it would arrive as a
        # twelfth, picture-less tile, re-lay the whole grid, and spend lit area
        # on the one screen every startup routine PHOTOGRAPHS. Keeping the menu
        # off the measurements was worth a 1.4x margin on his weak side
        # (2026-10-20). Giving this page a tile is a deliberate choice that
        # needs artwork first; it is reachable with K either way.
        entries = [(cls, key) for cls, key in base.registered()
                   if cls is not Menu and not getattr(cls, "is_screen", False)]
        entries.sort(key=lambda e: (e[1] is None, glyph_of_key(e[1]), e[0].name))
        rects = tile_layout(self.playfield(ctx), len(entries) + len(CHROME), ctx.width)
        # ORDER: the player-count tiles, then every drill, then EXIT.
        first = [c for c in CHROME if c.players is not None]
        last = [c for c in CHROME if c.players is None]
        it = iter(rects)

        def chrome(c: Chrome) -> Slot:
            return Slot(None, c.glyph, c.title, next(it), stem=c.stem, players=c.players)

        out = [chrome(c) for c in first]
        out += [Slot(cls, glyph_of_key(key), cls.title, next(it)) for cls, key in entries]
        out += [chrome(c) for c in last]
        return out

    def tile_at(self, x: float, y: float, ctx: RenderContext) -> "Slot | None":
        for s in self.slots(ctx):
            if s.contains(x, y):
                return s
        return None

    # -- choosing -------------------------------------------------------------

    def _choose(self, cls: type) -> None:
        if self.pending is None:         # the first choice in a turn wins
            self.pending = cls
            self._dirty = True

    def _players_hit(self, s) -> None:
        """A player-count box was shot or clicked: record the request, once.

        ONE HIT, NO ARMING. EXIT is two-hit because a phantom closing the app
        mid-string is the worst available accident; this is reversible, visible,
        and reachable only while the menu is up, so a phantom cannot get at it
        during a drill. Like `_choose`, the first request in a turn wins.

        It records and returns: the SETTING is the app's, applied by its tick
        (`take_pending_players`), and the menu stays up so a drill can then be
        chosen.
        """
        if self.pending_players is None:
            self.pending_players = s.players
        self._dirty = True

    def selected_players(self, ctx: RenderContext) -> "int | None":
        """Which selector box reads as selected: the pending request if there is
        one, else the app's setting (`ctx.players`). The pending overlay exists
        so a hit is visible from the firing line in the SAME repaint, before the
        app's 200 ms drain lands. Falls back to the first offered count when the
        value matches no box, so exactly one is always selected."""
        offered = [c.players for c in CHROME if c.players is not None]
        if not offered:
            return None
        want = self.pending_players if self.pending_players is not None else ctx.players
        return want if want in offered else offered[0]

    def _chrome_hit(self, s, t: float) -> None:
        """EXIT was shot or clicked: arm it, or on the second hit, ask to quit.

        TWO HITS, because a phantom detection closing the app mid-string is
        the worst accidental action in here -- 92 phantoms arrived in twenty
        seconds once, when the window dropped behind something else. The first
        hit arms it and the tile SAYS SO; the second asks the app to close.

        It is a REQUEST, not a close. This runs inside `ModeHost.shot`, and
        tearing down the window from here would destroy objects that are
        still executing. The app drains `quit_requested` on its tick, the
        same way it drains a chosen drill.
        """
        if s.players is not None:
            self._players_hit(s)
            return
        if s.stem != "exit":
            return
        if self.exit_armed_at and (t - self.exit_armed_at) <= EXIT_ARM_S:
            self.exit_armed_at = 0.0
            self.quit_requested = True
        else:
            self.exit_armed_at = t
        self._dirty = True

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        self.shots += 1
        s = self.tile_at(ev.x, ev.y, ctx)
        if s is None:
            return ShotOutcome.miss()
        self.hits += 1
        # EVERY TILE answers with the selection cue (2026-10-20): a drill tile, the
        # One Player / Two Players tiles and EXIT, and on EXIT both of its two hits
        # (the first arms it and says so; a cue confirming that it landed is the
        # point). It FOLLOWS the gunshot, never replaces it.
        if s.cls is None:
            self._chrome_hit(s, ev.t)
            return ShotOutcome(hit=True, target_id=f"tile:{s.stem}",
                               draw_hole=False, cue=SELECT_SOUND)
        # Records the choice and returns. It must NOT switch modes here: this
        # is running inside `ModeHost.shot`, and `set_mode` would stop the
        # object that is executing. See the module docstring and `take_pending`.
        self._choose(s.cls)
        # No bullet hole: the tile starting is the feedback, and a hole would be
        # left behind on whatever screen comes next. The bang still plays.
        return ShotOutcome(hit=True, target_id=f"tile:{s.cls.name}", draw_hole=False,
                           cue=SELECT_SOUND)

    def on_click(self, x: float, y: float, ctx: RenderContext) -> bool:
        """A mouse click on a tile chooses that drill, exactly as a shot does.

        Asked for on 2026-10-18: "menu should also be triggered by the mouse
        pointer". The menu is the one screen in this app that is a set of
        buttons rather than a set of targets, so a pointer belongs on it --
        and when he is at the keyboard rather than at the firing line, it is
        the quickest way in.

        THE SAME DEFERRED ROUTE as a shot and a key: it records the choice and
        the app's tick starts the drill through `_select_mode`. Switching from
        inside an input handler would stop the object still executing.

        IT IS NOT A SHOT AND IS NOT COUNTED AS ONE. `self.shots` and
        `self.hits` are the gun's tallies and are untouched here, so clicking
        through a menu can never flatter a hit rate.
        """
        s = self.tile_at(x, y, ctx)
        if s is None:
            return False                 # between the tiles: not ours
        # A CLICK has no gunshot, so the cue plays alone (the app drains
        # `take_sound` on its tick): the click landing is what it confirms.
        self._queued_sound = SELECT_SOUND
        if s.cls is None:
            self._chrome_hit(s, ctx.now)
            return True
        self._choose(s.cls)
        return True

    def on_key(self, key: int) -> bool:
        """Claims O and T, and ONLY those, and only while the menu is showing.

        O selects One Player and T selects Two Players, through the SAME
        `pending_players` hand-off a tile hit uses (`_players_hit`): one route,
        drained by the app's tick. Neither key is used by anything else (S stays
        the stats HUD, M stays the menu key, T no longer runs the self check:
        that is C), so this shadows nothing; the keys are listed as data in
        `MENU_OWN_KEYS` so the collision check can see them. Every other key
        falls through, as below.

        THE CONTRACT BELOW STILL HOLDS FOR EVERY OTHER KEY, and the integration
        it was waiting for now exists (`take_pending_players` is drained on the
        app's tick and a test drives the real key path):

        Always False for any other key: this mode claims NO other key.

        THE CONTRACT: a mode must not claim a key it cannot act on. Layers get
        first refusal, and a True here stops `TargetWindow.keyPressEvent` emitting
        `key_pressed`, so `camera_lab._on_target_key`'s `by_key` path -- the only
        thing that can switch modes today -- is never reached, and nothing drains
        `pending`: the shooter is stranded on the menu with F11 (closes the whole
        app) as the only way out. Returning False lets a tile's key fall through
        to `by_key`, which starts that drill exactly as it does everywhere else,
        so the menu is useful before the integration lands. Do NOT change this to
        True until the app calls `take_pending()` and a test drives the real key
        path end to end. Nothing is recorded either (see the module docstring).
        """
        for c in CHROME:
            if c.key is not None and key == c.key and c.players is not None:
                self._players_hit(c)
                return True
        return False

    def take_pending(self) -> "type[Mode] | None":
        """The chosen mode class, or None. Either it returns a choice and consumes
        it, or it returns None and consumes nothing -- never a consumed choice
        reported as None. The app hands the result to `_select_mode`, and only
        while this menu is still the host's active mode (see the module
        docstring: the caller owns that test)."""
        if self.pending is None:
            return None
        cls = self.pending
        self.pending = None
        return cls

    def take_pending_players(self) -> "int | None":
        """The requested player count, or None. Either it returns a request and
        consumes it, or it returns None and consumes nothing (RULES C1)."""
        if self.pending_players is None:
            return None
        n = self.pending_players
        self.pending_players = None
        return n

    def aim_point(self, ctx: RenderContext) -> "tuple[float, float] | None":
        """Where the development walk-through (`--modetest`) should aim: the
        centre of the first tile's picture. None when there is no tile."""
        # The first DRILL tile (the player-count tiles lead the grid but are not
        # drills, and the walk-through wants something to start).
        slots = [s for s in self.slots(ctx) if s.cls is not None]
        if not slots:
            return None
        c = slots[0].picture.center()
        return (c.x(), c.y())

    def advance(self, ctx: RenderContext) -> bool:
        """One repaint for a recorded choice, then nothing. A static menu must
        not repaint a 4K surface sixty times a second."""
        if not self._dirty:
            return False
        self._dirty = False
        return True

    def status(self) -> str:
        return "MENU: shoot a square, or press its key, to start that drill"

    # -- drawing --------------------------------------------------------------

    def _solid_mask(self) -> QPixmap:
        """One opaque pixel. `Occluder.mask` is stretched over its rect, so a
        solid rectangle needs no more than this (built once, lazily)."""
        if self._solid is None:
            img = QImage(1, 1, QImage.Format_ARGB32_Premultiplied)
            img.fill(0xFFFFFFFF)
            self._solid = QPixmap.fromImage(img)
        return self._solid

    def occluders(self, ctx: RenderContext) -> "list[Occluder]":
        """Every tile is SOLID: the bullet holes underneath do not show through.

        The pictures are line art on a transparent ground (white at coverage
        v/255), and holes persist across a mode switch, so entering the menu
        after a string would draw his old holes across the tile art -- the fault
        he photographed on the animated figure. The rectangle is the tile's own
        (`slots`, the same one `paint` draws into and `shot` tests against),
        aligned OUTWARD so no edge pixel of it is left uncovered. Nothing while
        the grid is being shot: `paint` draws nothing then.
        """
        if ctx.calibrating:
            return []
        return [Occluder(self.z, s.rect.toAlignedRect(), self._solid_mask)
                for s in self.slots(ctx)]

    def _tile_for(self, s: Slot):
        """The picture for a tile: by key glyph for a drill, by file stem for a
        chrome tile, None for a drill with no picture."""
        # THE STEM WINS WHERE THERE IS ONE. Glyph-first was right while chrome
        # tiles had no key; the moment EXIT started showing "Esc" it looked up
        # a picture named for a key that has no file, and the tile went blank.
        # A chrome tile is identified by what it IS, a drill by the key that
        # starts it.
        if s.stem:
            return tiles.named(s.stem)
        if s.glyph:
            return tiles.for_glyph(s.glyph)
        return None

    def _pixmap(self, s: Slot, side: int, level: float) -> "QPixmap | None":
        """The tile's picture at `side` px, as white at coverage v/255 (so red
        excess is 0 and it composites over any background, as the figures do).
        None for a drill with no picture. Cached by (glyph, side, ceiling): a
        moved ceiling or a new screen size builds a fresh one."""
        key = (s.glyph, s.stem, side, round(level, 2))
        if key in self._pix:
            return self._pix[key]
        tile = self._tile_for(s)
        pm = None
        if tile is not None:
            src = tile.ink
            interp = cv2.INTER_AREA if side < src.shape[0] else cv2.INTER_LINEAR
            v = np.ascontiguousarray(cv2.resize(src, (side, side), interpolation=interp))
            bgra = np.empty((side, side, 4), np.uint8)
            bgra[..., 0] = v
            bgra[..., 1] = v
            bgra[..., 2] = v
            bgra[..., 3] = v                      # premultiplied: colour <= alpha
            img = QImage(bgra.data, side, side, side * 4,
                         QImage.Format_ARGB32_Premultiplied).copy()
            pm = QPixmap.fromImage(img)
        if len(self._pix) >= self.CACHE_BOUND:
            self._pix.clear()
        self._pix[key] = pm
        return pm

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if ctx.calibrating:
            return          # one ring to shoot at while the grid is shot
        level = ceiling.figure_level()
        p.setBrush(Qt.NoBrush)
        slots = self.slots(ctx)
        for s in slots:
            self._paint_tile(p, ctx, s, level)
        self._paint_heading(p, ctx, slots)

    def _paint_heading(self, p: QPainter, ctx: RenderContext, slots) -> None:
        """MAIN MENU, over the grid, so it says what this screen is.

        Asked for on 2026-10-18: "there should be Main Menu written on the
        menu mode so that people know what that is." A screen of pictures
        with keys on them does not announce itself to someone who has not
        been told what it is.

        It sits in the gap ABOVE the tiles, which the layout already leaves
        when the grid is centred, and it is drawn at GREY_MID like the tile
        captions -- not brighter. It is on the screen the camera watches
        while he is shooting at the tiles, so it spends the same light
        budget everything else here does (B1b), and nothing on this screen
        may out-shine a laser dot landing on it (B10).
        """
        if not slots:
            return
        top = min(s.rect.top() for s in slots)
        left = min(s.rect.left() for s in slots)
        right = max(s.rect.right() for s in slots)
        band = QRectF(left, max(0.0, top - px(ctx, HEADING_AT_1080) * 2.2),
                      right - left, px(ctx, HEADING_AT_1080) * 1.8)
        if band.height() <= 4.0:
            return                      # no room above the grid: leave it out
        # The mode's own title, not a second copy of the words: one name for
        # one thing. It is also what the status band says, so the screen and
        # the band cannot drift apart.
        _size, text = fit_text(p, self.title.upper(), True,
                               px(ctx, HEADING_AT_1080),
                               band.width(), band.height())
        p.setPen(QPen(GREY_MID))
        p.drawText(band, Qt.AlignHCenter | Qt.AlignVCenter, text)

    @staticmethod
    def first_char_x(p: QPainter, ctx: RenderContext, cap: QRectF) -> float:
        """Where the first character of a tile's label begins: the left edge of a
        numbered tile's digit, i.e. the caption square's left plus half of what
        the digit leaves over, measured through the font (RULES B5). The player
        tiles start their word here; the numbered tiles' digit is drawn centred
        in the same square and so lands on it. Every tile asks this one function."""
        pad = cap.height() * 0.12
        size, _t = fit_text(p, "1", True, px(ctx, GLYPH_AT_1080),
                            cap.height() - 2 * pad, cap.height() - 2 * pad)
        adv = QFontMetricsF(mono(size, True)).horizontalAdvance("1")
        return cap.left() + max(0.0, (cap.height() - adv) / 2.0)

    @staticmethod
    def _draw_name(p: QPainter, cell: QRectF, text: str, size: int, lead: bool) -> None:
        """One line of a tile's name, left-aligned. With `lead` the first letter
        is drawn BOLD in `LEAD_GREY` and the rest regular in `GREY_MID`; the rest
        starts where the first letter's measured advance ends (monospace, but
        measured, never assumed -- RULES B5)."""
        if not lead or not text:
            p.setFont(mono(size, False))
            p.drawText(cell, Qt.AlignLeft | Qt.AlignVCenter, text)
            return
        w1 = QFontMetricsF(mono(size, True)).horizontalAdvance(text[0])
        p.setFont(mono(size, True))
        p.setPen(QPen(LEAD_GREY))
        p.drawText(QRectF(cell.left(), cell.top(), w1 + 2.0, cell.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, text[0])
        p.setFont(mono(size, False))
        p.setPen(QPen(GREY_MID))
        p.drawText(QRectF(cell.left() + w1, cell.top(), max(1.0, cell.width() - w1), cell.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, text[1:])

    def _paint_tile(self, p: QPainter, ctx: RenderContext, s: Slot, level: float) -> None:
        r = s.rect
        side = max(8, int(round(r.width())))
        pm = self._pixmap(s, side, level)
        if pm is not None:
            p.drawPixmap(int(round(r.left())), int(round(r.top())), pm)
        elif s.glyph:
            pic = s.picture
            _size, text = fit_text(p, s.glyph, True, px(ctx, BIG_GLYPH_AT_1080),
                                   pic.width() * 0.5, pic.height() * 0.5)
            p.setPen(QPen(GREY_HI))
            p.drawText(pic, Qt.AlignCenter, text)

        # The border goes on top of the picture. Brighter and heavier for the
        # tile that has been chosen, so a first shot is visibly registered.
        # `s.cls is not None` matters: a chrome tile has cls None and an idle
        # menu has pending None, and `None is None` would light EXIT at rest.
        chosen = s.cls is not None and self.pending is s.cls
        # AN ARMED EXIT LOOKS ARMED. His complaint was "I was shooting the
        # exit menu and nothing had happened" -- so the hit that arms it must
        # be visible from the firing line, not just recorded.
        armed = (s.cls is None and s.stem == "exit"
                 and self.exit_armed_at > 0.0)
        hot = chosen or armed
        w = max(1.0, r.width() * BORDER_FRAC) * (PENDING_BORDER_X if hot else 1.0)
        p.setPen(QPen(GREY_HI if hot else GREY_MISS, w))
        p.drawRect(r.adjusted(w / 2.0, w / 2.0, -w / 2.0, -w / 2.0))
        if s.players is not None and s.players == self.selected_players(ctx):
            # THE SELECTED BOX IS UNDERLINED. Grey from the palette, a fraction
            # of the tile, red excess 0. Exactly one box gets it.
            ut = max(1.0, r.width() * UNDERLINE_THICK_FRAC)
            inset = r.width() * UNDERLINE_INSET_FRAC
            y = r.bottom() - r.width() * UNDERLINE_FLOOR_FRAC - ut
            # fillRect, not drawRect: it is a fill, and the border-level check
            # spies on drawRect to see one border per tile.
            p.fillRect(QRectF(r.left() + inset, y, r.width() - 2 * inset, ut), GREY_HI)
        if armed:
            pic = s.picture
            _sz, txt = fit_text(p, "AGAIN TO EXIT", True,
                                px(ctx, GLYPH_AT_1080),
                                pic.width() * 0.86, pic.height() * 0.18)
            p.setPen(QPen(GREY_HI))
            p.drawText(QRectF(pic.left(), pic.center().y(),
                              pic.width(), pic.height() * 0.22),
                       Qt.AlignCenter, txt)

        cap = s.caption
        pad = cap.height() * 0.12
        # ONE DEFINITION of where a tile's label starts. A numbered tile's digit is
        # centred in a square as wide as the caption is tall, so its left edge sits
        # a little inside the tile; the two player-count tiles lost their "1P"/"2P"
        # glyph and the word then began hard against the border (his screenshot,
        # 2026-10-20: "the S and the M are touching it"). Every tile now takes its
        # first character's left edge from `first_char_x`, the digit's own, so the
        # two kinds cannot drift apart.
        left = self.first_char_x(p, ctx, cap)
        if s.glyph:
            cell = QRectF(cap.left(), cap.top(), cap.height(), cap.height())
            _size, text = fit_text(p, s.glyph, True, px(ctx, GLYPH_AT_1080),
                                   cell.width() - 2 * pad, cap.height() - 2 * pad)
            p.setPen(QPen(GREY_HI))
            p.drawText(cell, Qt.AlignCenter, text)
            left = cell.right()
        name_cell = QRectF(left, cap.top(), cap.right() - left - pad, cap.height())
        start = px(ctx, NAME_AT_1080)
        size, text = fit_text(p, s.title, s.lead, start,
                              name_cell.width(), cap.height() - 2 * pad)
        lines = [(text, size)]
        if size < NAME_FLOOR_FRAC * start and len(wrap_title(s.title)) > 1:
            # TWO LINES RATHER THAN A SMALLER ONE. At 1/7 of the screen the
            # longer names fitted only at 19-22pt against a floor of 24 on his
            # 4K panel, measured by the suite. Shortening `Mode.title` was the
            # obvious alternative and is the wrong lever: the same strings are
            # what the prompt band across the top of the target screen shows,
            # so a menu-only short name would be a second name for one thing.
            # Wrapping is local to this tile, costs no light budget and leaves
            # the tile size -- and therefore the lit-area headroom -- alone.
            half = (cap.height() - 2 * pad) / 2.0
            parts = wrap_title(s.title)
            fitted = [fit_text(p, t, s.lead, start, name_cell.width(), half)
                      for t in parts]
            # ONE SIZE FOR BOTH LINES: a name whose second line is visibly
            # larger than its first reads as two labels, not one name.
            common = min(sz for sz, _t in fitted)
            lines = [(fit_text(p, t, s.lead, common, name_cell.width(), half)[1],
                      common) for t in parts]
        # GREY_MID, not GREY_TEXT: the names are ~16 characters in a 40-point
        # font on each tile, an estimated 0.45% of the screen, and with the
        # pictures (1.176%) that would leave almost nothing of the 2% budget.
        # GREY_MID is 58 against a lit threshold of 60 TODAY -- two units of
        # margin that exist only because of the artwork ceiling (71 on file).
        # The palette derives GREY_MID from that ceiling (0.75 of its on-screen
        # level), so the margin MOVES with the artwork: at a ceiling of 52 the
        # captions are 45 (15 under, dimmer and still in budget, but harder to
        # read), and they only cross 60 when there is no artwork to measure at
        # all. `caption_margin()` states it in code and the tests pin it. The
        # value is deliberately unchanged; a darker stroke hides the dot less
        # (RULES B10).
        p.setPen(QPen(GREY_MID))
        if len(lines) == 1:
            self._draw_name(p, name_cell, lines[0][0], lines[0][1], s.lead)
        else:
            h = name_cell.height() / len(lines)
            for i, (t, sz) in enumerate(lines):
                self._draw_name(p, QRectF(name_cell.left(), name_cell.top() + i * h,
                                          name_cell.width(), h), t, sz, s.lead and i == 0)
