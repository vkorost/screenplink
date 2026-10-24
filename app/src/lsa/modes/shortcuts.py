"""Keyboard Shortcuts (key K): a full-screen list of every key the app answers to.

Why it exists. The key legend along the bottom of the screen was typed by hand and
listed the pre-2026-10-18 drill numbering for three days with every suite green:
it told him "5 shoot/no-shoot" about a drill that had been removed, and said
nothing about keys that did things. This screen is the answer to "what does this
key do?" and the one thing it has to get right is **not to go stale the same way**.

THE DRILL ROWS ARE READ FROM THE MODE REGISTRY, never typed. `drill_entries()`
asks `lsa.modes.registered()` for every mode that has a key and uses the mode's
own `title`, so a drill added next year appears here with no change to this
file, and a drill renamed or renumbered cannot be mis-described. The two player
rows (O, T) are read from the menu's own `CHROME` table for the same reason. What
CANNOT be derived stays as a table in this module (`CHECKS`, `ON_SCREEN`, `ALSO`,
`NEW_ROUND`): the keys `camera_lab._on_target_key`, the layers and the window own.
`tests/test_modes.py` section 46 checks that every key this screen advertises is
actually handled somewhere, so that table cannot quietly lie either.

The wording, grouping and greys are the approved mockup,
`requirements/help_screen_mockup_01.html`.

IT CLAIMS NO KEYS. `on_key` is False for everything. Layers get first refusal on
every key and a True from here would stop `TargetWindow.keyPressEvent` emitting
`key_pressed`, so `camera_lab._on_target_key` would never see it: the menu once
claimed the drill keys it could not act on and became a trap that only F11
escaped. From this screen, pressing 1 starts drill 1, C calibrates, and so on.

A SHOT HERE SCORES NOTHING. It is reported as a miss that is also `unscored`
(neither a hit nor a miss for the session, so the hit rate is not deflated by
shots at a help screen), which is what lets the MENU button claim it: the host
offers the drill the shot first and the button only takes a miss. The bullet hole
is drawn as for any shot. `is_screen` tells the host this is a screen and not a
drill: it spends no rounds from the magazine and is never split for two players.

STATIC. It asks for no repaints when idle (`animated` is False and `advance` is
the Layer default). The layout is built once per geometry and cached.

WHAT IT DRAWS ON, AND WHY THE GREYS (RULES B1, B1b, B5, B10). The camera watches
this monitor:

  * Red excess exactly 0: neutral greys from the palette in `display/layers.py`,
    no colour of any kind.
  * Nothing brighter than the artwork ceiling. The keys and their boxes are at
    `GREY_HIT` (the ceiling itself), the title at `GREY_HI`, the headings at
    `GREY_MID` and the prose at `GREY_TEXT` -- the last two sit UNDER luminance
    60, so the body of the screen spends nothing of the 2% lit-area budget and
    the whole budget is available to the key glyphs and their boxes.
  * The key boxes are a STROKE, never a fill. A fill is lit area, and a dot
    landing on it loses that grey from its score, unit for unit.
  * Nothing is sized in pixels: sizes are `px(ctx, size_at_1080)` and fractions of
    the screen width, and every string is measured with real font metrics
    (`QFontMetricsF`) and fitted to its box, never a guessed advance.
  * It keeps clear of the MENU button (`Mode.menu_keepout`), wherever the app has
    put it, and draws nothing while the grid is being shot.

The lit-area figure is MEASURED by rendering (tests/test_modes.py section 46),
not argued from arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QFont, QFontMetricsF, QPainter, QPen

from ..display.layers import (BOTTOM_STRIP, GREY_HI, GREY_HIT, GREY_MID, GREY_TEXT,
                              mono, px, top_band_height)
from ..display.scene import RenderContext
from . import base
from .base import Mode, ShotEvent, ShotOutcome, register
from .menu import CHROME, glyph_of_key

TITLE = "KEYBOARD SHORTCUTS"

# Sizes picked at 1080p and scaled by `px`. `mono` takes POINTS, which at the 96
# DPI Windows reports are 3/4 of a CSS pixel; the mockup's 1.47 / 1.38 / 2.0 per
# cent of the screen width are 28 / 26 / 38 px on a 1920-wide screen, so 21 / 20 /
# 29 points. If a screen reports another DPI the glyphs come out proportionally
# bigger or smaller and `build_plan` shrinks them to fit; it cannot grow them.
BODY_AT_1080 = 21
HEAD_AT_1080 = 20
TITLE_AT_1080 = 29
FONT_FLOOR = 9

# Geometry, as fractions of the SCREEN WIDTH (the mockup's container-query units,
# 1cqw = 0.01). Nothing here is in pixels (RULES B5).
SIDE_MARGIN = 0.034      # left and right of the two columns
GUTTER = 0.034           # between the columns
KEY_GAP = 0.008          # between a key's box and its description
ROW_GAP = 0.0024         # between rows
GROUP_GAP = 0.0065       # between groups, and under the header
HEAD_GAP = 0.0035        # between a heading and its first row
BOX_PAD_X = 0.0045       # a key's box, either side of the glyph
BOX_PAD_Y = 0.0006       # ... and above and below it
# The box outline's width. 0.0014 (the mockup's .14cqw) measured 2.005% lit above
# luminance 60 at 1080p against the 2% budget (test_modes section 46, 2026-10-23): the
# budget is spent by each key's box stroke, its glyph and the title, because the prose is
# under 60. This is the smallest lever -- a thinner outline, no row removed, the budget
# untouched -- and it is a FRACTION OF THE SCREEN WIDTH (1.92 px at 1080p, 3.84 at 4K).
BOX_STROKE = 0.0010
BOX_RADIUS = 0.003
TITLE_GAP = 0.012        # between the MENU button and the title

# When the text does not fit it is shrunk uniformly, never below this; below it
# the strings are elided instead of overrunning (and `Plan.elided` says so).
MIN_SCALE = 0.45
SCALE_STEP = 0.96


@dataclass(frozen=True)
class Entry:
    """One row: the glyph in the box, what it does, and the Qt key codes it
    stands for (more than one for a glyph such as "+ -", which is two keys)."""

    glyph: str
    title: str
    keys: tuple


# -- what cannot be derived: the keys the app, the layers and the window own -----

NEW_ROUND = Entry("N", "New Round", (int(Qt.Key_N),))

CHECKS = (
    Entry("C", "Self-Calibrate", (int(Qt.Key_C),)),
    Entry("A", "Self-Check", (int(Qt.Key_A),)),
    Entry("Q", "Last Verdict", (int(Qt.Key_Q),)),
    Entry("G", "Calibrate by Shooting", (int(Qt.Key_G),)),
    Entry("U", "Calibrate Brightness by Shooting", (int(Qt.Key_U),)),
    Entry("X", "Measure the Exposure", (int(Qt.Key_X),)),
    Entry("F", "Measure the Noise Floor", (int(Qt.Key_F),)),
)

ON_SCREEN = (
    Entry("0", "All Readouts", (int(Qt.Key_0),)),
    Entry("S", "Stats", (int(Qt.Key_S),)),
    Entry("B", "Status Band", (int(Qt.Key_B),)),
    Entry("L", "Signal Meter", (int(Qt.Key_L),)),
    Entry("P", "Camera Pane", (int(Qt.Key_P),)),
    Entry("V", "Aim the Camera", (int(Qt.Key_V),)),
)

ALSO = (
    Entry("E", "Erase the Holes", (int(Qt.Key_E),)),
    Entry("+ -", "Volume Up / Down", (int(Qt.Key_Plus), int(Qt.Key_Minus))),
    Entry("Esc", "Exit the Game", (int(Qt.Key_Escape),)),
)

HEAD_DRILLS = "DRILLS - PRESS AGAIN FOR A NEW ROUND"
HEAD_PLAYERS = "PLAYERS - ON THE MENU"
HEAD_CHECKS = "CHECKS"
HEAD_ON_SCREEN = "ON SCREEN - PRESS TO SHOW OR HIDE"
HEAD_ALSO = "ALSO"


def drill_entries() -> "list[Entry]":
    """Every registered mode that has a key, read from the registry NOW.

    Digits first, in numeric order, then the lettered ones in registration order
    (the menu is registered before this screen, so M precedes K, as in the
    mockup). A mode with no key has nothing to advertise and is left out.
    """
    reg = [(cls, key) for cls, key in base.registered() if key is not None]
    digits = sorted((e for e in reg if 0x30 <= int(e[1]) <= 0x39), key=lambda e: int(e[1]))
    rest = [e for e in reg if not (0x30 <= int(e[1]) <= 0x39)]
    return [Entry(glyph_of_key(key), cls.title, (int(key),)) for cls, key in digits + rest]


def player_entries() -> "list[Entry]":
    """One Player / Two Players, read from the menu's own table of chrome tiles."""
    return [Entry(glyph_of_key(c.key), c.title, (int(c.key),))
            for c in CHROME if c.players is not None and c.key is not None]


def groups() -> "tuple[list, list]":
    """(left column, right column), each a list of (heading, [Entry])."""
    left = [(HEAD_DRILLS, drill_entries() + [NEW_ROUND]),
            (HEAD_PLAYERS, player_entries())]
    right = [(HEAD_CHECKS, list(CHECKS)),
             (HEAD_ON_SCREEN, list(ON_SCREEN)),
             (HEAD_ALSO, list(ALSO))]
    return left, right


def advertised() -> "list[Entry]":
    """Every row the screen draws, in drawing order."""
    left, right = groups()
    return [e for col in (left, right) for _h, es in col for e in es]


# -- the layout ------------------------------------------------------------------

@dataclass
class Item:
    """One thing to draw. `rect` is the box the text must fit inside (for a key,
    the outline of its box)."""

    kind: str                  # "title" | "heading" | "key" | "desc"
    text: str
    rect: QRectF
    size: int                  # point size, as `mono` takes it
    bold: bool
    font: QFont
    group: str = ""            # the heading this row sits under ("" for the title)


@dataclass
class Plan:
    items: "list[Item]" = field(default_factory=list)
    scale: float = 1.0
    fits: bool = True
    elided: int = 0            # strings shortened to fit; 0 on any real screen
    stroke: float = 1.0
    radius: float = 1.0
    area: QRectF = field(default_factory=QRectF)
    header: QRectF = field(default_factory=QRectF)
    keepout: QRectF = field(default_factory=QRectF)


def _area(ctx: RenderContext) -> QRectF:
    """Where the two columns may go: the screen, less the instruction band across
    the top and the key legend along the bottom (both are always drawn there)."""
    r = ctx.rect
    top = r.y() + top_band_height(ctx)
    bottom = r.y() + r.height() - max(BOTTOM_STRIP, px(ctx, BOTTOM_STRIP))
    m = r.width() * SIDE_MARGIN
    return QRectF(r.x() + m, top, r.width() - 2.0 * m, max(0.0, bottom - top))


def _try_plan(ctx: RenderContext, left: list, right: list, s: float, elide: bool) -> Plan:
    W = float(ctx.width)
    area = _area(ctx)
    kb = Mode.menu_keepout(ctx)
    top_half = kb.center().y() < ctx.rect.y() + ctx.height / 2.0
    on_left = kb.center().x() < ctx.rect.x() + ctx.width / 2.0

    body = max(FONT_FLOOR, int(px(ctx, BODY_AT_1080) * s))
    head = max(FONT_FLOOR, int(px(ctx, HEAD_AT_1080) * s))
    title = max(FONT_FLOOR, int(px(ctx, TITLE_AT_1080) * s))
    f_body, f_key = mono(body), mono(body, True)
    f_head, f_title = mono(head, True), mono(title, True)
    m_body, m_key = QFontMetricsF(f_body), QFontMetricsF(f_key)
    m_head, m_title = QFontMetricsF(f_head), QFontMetricsF(f_title)

    padx, pady = W * BOX_PAD_X * s, W * BOX_PAD_Y * s
    key_gap, row_gap = W * KEY_GAP * s, W * ROW_GAP * s
    group_gap, head_gap = W * GROUP_GAP * s, W * HEAD_GAP * s
    plan = Plan(scale=s, stroke=max(1.0, W * BOX_STROKE), radius=W * BOX_RADIUS,
                area=area, keepout=kb)

    col_w = (area.width() - W * GUTTER) / 2.0
    box_w = {e.glyph: m_key.horizontalAdvance(e.glyph) + 2.0 * padx
             for col in (left, right) for _h, es in col for e in es}
    key_area_w = max(box_w.values()) if box_w else 0.0
    desc_w = col_w - key_area_w - key_gap
    row_h = m_body.height() + 2.0 * pady
    ok = [True]

    def fitted(text: str, metrics: QFontMetricsF, width: float) -> str:
        if metrics.horizontalAdvance(text) <= width:
            return text
        ok[0] = False
        if elide:
            plan.elided += 1
            return metrics.elidedText(text, Qt.ElideRight, max(1, int(width)))
        return text

    # The header row: the title, beside the MENU button when that is at the top,
    # across the top of the area when the button is at the bottom.
    if top_half:
        gap = W * TITLE_GAP * s
        x0 = kb.right() + gap if on_left else area.left()
        x1 = area.right() if on_left else kb.left() - gap
        header = QRectF(x0, kb.top(), max(0.0, x1 - x0), kb.height())
        cols_top = max(area.top(), kb.bottom()) + group_gap
        cols_bottom = area.bottom()
    else:
        header = QRectF(area.left(), area.top(), area.width(), m_title.height() * 1.3)
        cols_top = header.bottom() + group_gap
        cols_bottom = min(area.bottom(), kb.top())
    if m_title.height() > header.height() + 0.5:
        ok[0] = False
    plan.header = header
    plan.items.append(Item("title", fitted(TITLE, m_title, header.width()), header,
                           title, True, f_title))

    def column(x0: float, col: list) -> float:
        y = cols_top
        for gi, (heading, entries) in enumerate(col):
            if gi:
                y += group_gap
            plan.items.append(Item("heading", fitted(heading, m_head, col_w),
                                   QRectF(x0, y, col_w, m_head.height()), head, True, f_head,
                                   heading))
            y += m_head.height() + head_gap
            for e in entries:
                bw = box_w[e.glyph]
                plan.items.append(Item("key", e.glyph,
                                       QRectF(x0 + key_area_w - bw, y, bw, row_h),
                                       body, True, f_key, heading))
                plan.items.append(Item("desc", fitted(e.title, m_body, desc_w),
                                       QRectF(x0 + key_area_w + key_gap, y, desc_w, row_h),
                                       body, False, f_body, heading))
                y += row_h + row_gap
            y -= row_gap
        return y

    x_left = area.left()
    x_right = area.left() + col_w + W * GUTTER
    ends = (column(x_left, left), column(x_right, right))
    if max(ends) > cols_bottom + 1e-6:
        ok[0] = False
    plan.fits = ok[0]
    return plan


def build_plan(ctx: RenderContext, left: list, right: list) -> Plan:
    """Lay the whole screen out for this geometry: shrink uniformly until it
    fits, and only elide text if it still does not at the smallest scale.

    Measured with the fonts' own metrics, so what is checked is what is drawn.
    """
    s = 1.0
    while True:
        plan = _try_plan(ctx, left, right, s, elide=False)
        if plan.fits:
            return plan
        if s * SCALE_STEP < MIN_SCALE:
            return _try_plan(ctx, left, right, MIN_SCALE, elide=True)
        s *= SCALE_STEP


@register(key=Qt.Key_K)
class Shortcuts(Mode):
    """The key map, as a screen. It is a drill in the registry's sense (it has a
    key and a title and the MENU button leads back from it) and nothing else:
    no targets, no score, no sound of its own."""

    name = "shortcuts"
    title = "Keyboard Shortcuts"
    summary = "Every key the app answers to. Press one; the MENU button goes back."
    animated = False          # static: it asks for no repaints, ever
    # A screen, not a drill: it spends no rounds from the magazine and is never
    # split for two players (`ModeHost.counts_ammo`, `ModeHost.set_mode`).
    is_screen = True

    CACHE_BOUND = 8

    def __init__(self) -> None:
        self.visible = True
        self._plans: "dict[tuple, Plan]" = {}

    # -- the contract -----------------------------------------------------------

    def start(self, ctx: RenderContext) -> None:
        self._plans = {}

    def on_key(self, key: int) -> bool:
        """Claims NO key. See the module docstring: a mode must not claim a key it
        cannot act on, and this one has nothing to act on."""
        return False

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        """A shot here scores nothing. A miss, so the MENU button may claim it, and
        `unscored`, so the session's hit rate does not move."""
        return ShotOutcome(hit=False, unscored=True, target_id="shortcuts")

    def status(self) -> str:
        return "KEYBOARD SHORTCUTS: press any key listed; the MENU button (twice) goes back"

    # -- drawing ----------------------------------------------------------------

    def plan(self, ctx: RenderContext) -> Plan:
        """The layout for this geometry, built once and then reused. Keyed by the
        screen, the MENU button's keep-out (the app moves the button to the
        strongest corner after it has measured one) and the entries themselves, so
        a mode registered later gets a row and a moved button gets a new layout."""
        left, right = groups()
        key = (ctx.rect.getRect(),
               tuple(round(v, 1) for v in Mode.menu_keepout(ctx).getRect()),
               tuple((e.glyph, e.title) for col in (left, right)
                     for _h, es in col for e in es))
        plan = self._plans.get(key)
        if plan is None:
            plan = build_plan(ctx, left, right)
            if len(self._plans) >= self.CACHE_BOUND:
                self._plans.clear()
            self._plans[key] = plan
        return plan

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        if ctx.calibrating:
            return          # one ring to shoot at while the grid is shot
        plan = self.plan(ctx)
        p.save()
        p.setBrush(Qt.NoBrush)
        for it in plan.items:
            p.setFont(it.font)
            if it.kind == "key":
                # A STROKE round the glyph, both at the ceiling. Never a fill.
                p.setRenderHint(QPainter.Antialiasing, True)
                half = plan.stroke / 2.0
                p.setPen(QPen(GREY_HIT, plan.stroke))
                p.drawRoundedRect(it.rect.adjusted(half, half, -half, -half),
                                  plan.radius, plan.radius)
                p.setPen(QPen(GREY_HIT))
                p.drawText(it.rect, Qt.AlignCenter, it.text)
            else:
                colour = {"title": GREY_HI, "heading": GREY_MID}.get(it.kind, GREY_TEXT)
                p.setPen(QPen(colour))
                p.drawText(it.rect, Qt.AlignLeft | Qt.AlignVCenter, it.text)
        p.restore()
