"""The target display's render architecture: a stack of independent layers.

This exists because of where the project is going. Right now the thing on
screen is diagnostic furniture -- some rings to aim at and a block of numbers.
Soon it needs photographic backgrounds, animated targets that walk across the
screen, bullet holes that accumulate, muzzle flashes, hostage targets that must
not be shot, and a scoreboard, all composited together and all animating at 60
frames a second.

Writing that as one ever-growing paintEvent is the obvious move and the wrong
one. So instead: every visual element is a Layer with its own z-order, its own
visibility toggle, its own optional keyboard handling, and an advance(now, dt)
hook that is where animation lives. Adding a new kind of target later means
writing a new Layer, not editing an existing one.

The one firm rule for anything drawn here: NEVER draw saturated red. The camera
is pointed at this screen, and the detector is looking for red. Red graphics are
how you build a target that shoots itself. Layers that must show camera imagery
multiply it by blue first (see CameraPane) for exactly this reason.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from typing import Callable

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QPainter, QPixmap

from .session import SessionStats


class Occluder:
    """An opaque shape a layer is about to draw, announced BEFORE any layer paints.

    Why it exists (2026-10-17): the figure drills draw line art whose dark
    areas are TRANSPARENT (alpha = brightness, so it composites over any
    background). The bullet holes are on a lower layer, so they showed through
    every dark part of a walking figure and it looked like it floated through
    them. Z-order cannot fix that -- the holes were already underneath; the
    figure is simply not opaque.

    So a layer that draws something solid says where, and layers beneath it
    leave that area out (BulletHoles does; furniture may later). `z` is the
    owner's, so a layer only honours occluders that are genuinely above it.
    The mask is built lazily: it costs a resize and a pixmap, and is only
    needed when something underneath actually overlaps.
    """

    def __init__(self, z: int, rect: QRect, make_mask: Callable[[], QPixmap]) -> None:
        self.z = z
        self.rect = rect                      # screen-space bounding box
        self._make = make_mask
        self._mask: QPixmap | None = None

    def mask(self) -> QPixmap:
        """Alpha mask of ANY size: alpha > 0 where the shape is solid. The
        user stretches it over `rect` (smoothly), so a layer may keep one
        small mask for a shape that changes size every frame."""
        if self._mask is None:
            self._mask = self._make()
        return self._mask


@dataclass
class RenderContext:
    """Everything a layer is allowed to know about the world."""

    now: float
    dt: float
    rect: QRect
    stats: SessionStats
    dark_background: bool = True
    diagnostics: dict = field(default_factory=dict)   # live camera/detector numbers
    # The part of the screen the camera can actually see, when the mapping knows.
    # Off-axis, it may not see all of it -- in testing the bottom strip was
    # invisible -- and a target placed where a hit cannot be detected is worse
    # than no target at all: it reads as the app ignoring a good shot.
    playable: QRect | None = None
    # WHERE A SHOT IS RELIABLY DETECTED, as fractions of the screen, measured
    # from real trigger pulls (`lsa/playarea.py`). Distinct from `playable`
    # above, which is where the camera can SEE: the right-hand third of this
    # panel is inside that and outside this one, with geometry good to 0.75mm
    # and signal at 2.4x instead of 11x. None means never measured at this
    # camera position, which is not the same as "all of it is fine".
    play_area: object | None = None
    # The grid is being shot. Furniture that is merely informative gets out of
    # the way: during calibration there is exactly one thing to do, and a ring
    # that has to be picked out from behind a scoreboard is a ring that gets
    # shot at inaccurately.
    calibrating: bool = False
    # A measured camera-to-screen mapping exists, so a shot can be drawn where
    # it actually landed. Layers that only make sense BEFORE that -- the
    # camera-space minimap above all -- stand down. Asked for directly: "there
    # is a camera frame uncalibrated on the right and I don't understand why it
    # is there." It was there because nothing told it the mapping had arrived.
    calibrated: bool = False
    # The app's player-count SETTING (1 or 2), copied in by the Scene exactly as
    # `calibrated` is. A default, so every construction site keeps working.
    # Nothing but the menu's selector reads it yet.
    players: int = 1
    # THE MENU BUTTON'S KEEP-OUT, MEASURED ON THE FULL SCREEN. None everywhere
    # except inside a two-player half-screen context, where the host sets it.
    # `Mode.menu_keepout` derives the rectangle from the playfield and from a
    # fraction of the SMALLER side of the context it is given -- both change
    # when the width halves -- so a drill handed a half context would compute
    # a keep-out in the wrong place while the button is drawn in the right one.
    # The button belongs to the host and is drawn from the whole screen, so the
    # rectangle a drill avoids is carried in here, unchanged, and no drill needs
    # to know a split exists.
    menu_keepout_rect: "QRectF | None" = None
    # THE WIDTH OF THE WHOLE SCREEN, when `rect` is only part of it (a two-player
    # half). None means `rect` IS the screen. `AimField` sizes its marks by the
    # field's share of this width (`field_share`), which is exactly 1.0 when this
    # is None, so a whole-screen context is bit-for-bit what it was before.
    screen_width: "int | None" = None
    # Solid shapes layers are about to draw, gathered by `Scene.paint` before
    # any layer paints (see `Occluder`). Empty outside a paint.
    occluders: list = field(default_factory=list)

    @property
    def field_share(self) -> float:
        """This context's share of the screen's width: exactly 1.0 on a whole
        screen (no division happens), 0.5 for a half."""
        if self.screen_width is None or self.screen_width <= 0:
            return 1.0
        return self.rect.width() / self.screen_width

    @property
    def width(self) -> int:
        return self.rect.width()

    @property
    def height(self) -> int:
        return self.rect.height()


class Layer:
    """One compositable visual element.

    Subclasses override paint(), and advance() if they animate. Returning True
    from advance() marks the scene dirty; a scene with nothing animating stops
    repainting entirely, which keeps a static target screen at near-zero CPU.
    """

    name: str = "layer"
    z: int = 0
    visible: bool = True
    # Set by a layer whose content changes every frame regardless of events.
    animated: bool = False

    # A layer whose content a PERSON reads -- a number, a bar, a caption --
    # should not force a repaint faster than a person can read it. Set this to
    # a rate in Hz and `advance` will report a change at most that often.
    #
    # Measured 2026-10-14: `stats` and `signal` both claimed a change on every
    # tick, so the 3840x2160 target screen repainted 60 times a second
    # permanently, at 8.53 ms a paint -- half a CPU second per wall second,
    # spent on text nobody can read at that rate. The detector needs 6.96 ms
    # per camera frame out of the same budget and was measuring 24 ms live.
    # The signal meter is the sharpest example: its own documentation says it
    # holds a peak for 1.1 s "because a 100 ms pulse against a 10 Hz repaint
    # would otherwise never be seen" -- it was designed for 10 Hz and was
    # demanding 60.
    #
    # Animation is unaffected: a moving target returns a change every tick and
    # the screen keeps up with it. This only throttles furniture.
    read_rate_hz: float = 0.0        # 0 = every tick, as before

    # When this layer last reported a change, on the scene's clock. A CLASS
    # attribute so every layer has one from the moment it exists, rather
    # than being reached for with `getattr(self, "_last_reported", None)`:
    # that construct is what let `stats.last_shot_at` return 0.0 through a
    # default for days while a guard never fired once (RULES C8). None
    # means "never", which is what the throttle needs.
    _last_reported: float | None = None

    def advance(self, ctx: RenderContext) -> bool:
        if not self.animated:
            return False
        if self.read_rate_hz <= 0:
            return True
        period = 1.0 / self.read_rate_hz
        last = self._last_reported
        if last is not None and (ctx.now - last) < period:
            return False
        self._last_reported = ctx.now
        return True

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        raise NotImplementedError

    def occluders(self, ctx: RenderContext) -> "list[Occluder]":
        """Solid shapes `paint` will draw this frame, for layers beneath it to
        leave out. Must describe exactly what `paint` draws for this `ctx`
        (same clock). Default: nothing solid."""
        return []

    def on_key(self, key: int) -> bool:
        """Return True if the key was consumed."""
        return False

    def on_click(self, x: float, y: float, ctx: "RenderContext") -> bool:
        """A mouse click at screen (x, y). Return True if it was consumed.

        THE TARGET SCREEN HAD NO MOUSE INPUT AT ALL until 2026-10-18, when he
        asked for the menu to be usable with the pointer as well as the gun.
        Default False, so a layer that has no use for a click is unaffected
        and the click falls through to whatever is under it.

        A CLICK IS NOT A SHOT, and no layer may treat it as one: it does not
        come from the detector, it has no pulse, no peak and no camera
        position, and it must never reach the score, the hit rate, the
        sensitivity learner or the bullet holes. It is an operator's
        convenience for the screens that are menus. `ShotEvent` stays the
        only thing a drill is scored on.
        """
        return False

    def toggle(self) -> bool:
        self.visible = not self.visible
        return self.visible


class Scene:
    """An ordered stack of layers plus the bookkeeping to paint it efficiently."""


    def __init__(self, stats: SessionStats | None = None) -> None:
        self.stats = stats or SessionStats()
        self._layers: list[Layer] = []
        self._last_advance = time.perf_counter()
        self._dirty = True
        self.diagnostics: dict = {}
        # The part of the screen the camera can actually see. Set once the
        # mapping is measured; None means "assume all of it", which is the right
        # default before anything is known.
        self.playable: QRect | None = None
        # Declared unconditionally so no consumer reads it through getattr
        # with a default (RULES C8).
        self.play_area: object | None = None
        self.calibrating = False
        # A measured mapping exists. Layers that only make sense before one
        # does -- the camera-space minimap -- stand down once it is set.
        self.calibrated = False
        # The app's player-count setting, mirrored here by the app and copied
        # into every RenderContext, like `calibrated`.
        self.players = 1
        self.dark_background = True

    # -- composition ----------------------------------------------------------

    def add(self, layer: Layer) -> Layer:
        self._layers.append(layer)
        self._layers.sort(key=lambda la: la.z)
        self._dirty = True
        return layer

    def remove(self, name: str) -> None:
        self._layers = [la for la in self._layers if la.name != name]
        self._dirty = True

    def get(self, name: str) -> Layer | None:
        for la in self._layers:
            if la.name == name:
                return la
        return None

    def toggle(self, name: str) -> bool | None:
        la = self.get(name)
        if la is None:
            return None
        self._dirty = True
        return la.toggle()

    @property
    def layers(self) -> list[Layer]:
        return list(self._layers)

    # -- frame lifecycle ------------------------------------------------------

    def mark_dirty(self) -> None:
        self._dirty = True

    def advance(self, rect: QRect) -> bool:
        """Tick every layer. Returns True if a repaint is needed."""
        now = time.perf_counter()
        dt = now - self._last_advance
        self._last_advance = now
        ctx = self._context(now, dt, rect)
        dirty = self._dirty
        for la in self._layers:
            if la.visible and la.advance(ctx):
                dirty = True
        self._dirty = False
        return dirty

    def paint(self, p: QPainter, rect: QRect) -> None:
        ctx = self._context(time.perf_counter(), 0.0, rect)
        # Announce what the layers above are about to make solid BEFORE the
        # layers beneath paint, so they can leave that area out.
        for la in self._layers:
            if la.visible:
                ctx.occluders.extend(la.occluders(ctx))
        for la in self._layers:
            if not la.visible:
                continue
            p.save()
            try:
                la.paint(p, ctx)
            finally:
                p.restore()

    # KEYS THAT GO TO EVERY LAYER, not just the topmost that wants them: E (erase)
    # clears the bullet holes AND the hit marks, which live in two layers. The
    # first-refusal rule below would have let only the topmost one hear it (the key
    # was C until 2026-10-20 and the comment said "both" while only one layer did),
    # so a broadcast key is offered to every layer, visible or not -- clearing a
    # hidden layer's history is harmless and leaving it stale is not.
    # (Same class as every key collision here: a key claimed by one layer silently
    # shadows the layers beneath it, raises nothing, and reads as "the key half-works".)
    BROADCAST_KEYS = frozenset({Qt.Key_E})

    def on_key(self, key: int) -> bool:
        if key in self.BROADCAST_KEYS:
            took = False
            for la in self._layers:
                if la.on_key(key):
                    took = True
            if took:
                self._dirty = True
            return took
        # Topmost layer gets first refusal, which matches what a user expects
        # when an overlay is up.
        for la in reversed(self._layers):
            if la.visible and la.on_key(key):
                self._dirty = True
                return True
        return False

    def on_click(self, x: float, y: float, ctx: "RenderContext") -> bool:
        """Deliver a click to the topmost visible layer that wants it.

        Same order as `on_key` and for the same reason: what is on top is
        what the pointer is pointing at.
        """
        for la in reversed(self._layers):
            if la.visible and la.on_click(x, y, ctx):
                self._dirty = True
                return True
        return False

    def _context(self, now: float, dt: float, rect: QRect) -> RenderContext:
        return RenderContext(
            now=now, dt=dt, rect=rect, stats=self.stats,
            dark_background=self.dark_background, diagnostics=self.diagnostics,
            playable=self.playable,
            play_area=self.play_area,
            calibrating=self.calibrating,
            calibrated=self.calibrated,
            players=self.players,
        )
