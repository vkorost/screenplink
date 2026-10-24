"""A figure that MOVES: a filmed soldier, scored on his own silhouette.

Every other figure drill draws a still picture and scores it on the rings
printed in the artwork. This one plays a 10-second clip -- a soldier aiming
two pistols, firing, crouching and rising -- and there are no rings printed
on him, so he is scored the only honest way left: a shot inside the figure
scores, a shot outside does not. WHAT TAKES HIM DOWN is narrower (2026-10-20): four
hits in the HEAD or CENTRE MASS, derived per frame from his own silhouette
(`artwork.sequence.zones_from_silhouette`); a hit elsewhere on him scores but does
not advance the counter.

HIS DECISION, 2026-10-18: "for now you will have to score & mark the hits
inside his frame".

THREE THINGS THIS SHARES WITH THE STILL FIGURES, deliberately:

- **The silhouette is the hit test**, built the same way as
  `figure._silhouette` -- all the ink, thickened, closed and filled. What
  counts as a hit is the shape he can see, not a bounding box.
- **The shot is tested against the frame that was ON SCREEN when it broke**,
  not the frame that happens to be up when the event is handled. The ball
  modes had exactly this bug and he found it from the firing line: "I have
  to shoot in front of the ball and not inside the ball."
- **`moving_at` is True for the whole clip.** The sensitivity learner refuses
  shots fired while a mode's content is moving, because on 2026-10-17 a
  walking figure's edges taught the gain map that the brightest third of the
  screen was its weakest and cost him the right-hand side of the panel.

NOTHING HERE TOUCHES DETECTION, AUDIO OR THE CAMERA. A mode receives a shot
that has already been detected, mapped and sounded; it decides only what that
shot meant. That boundary is his instruction and it is also why this file can
exist at all without risking the baseline.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

from ..artwork.sequence import SRC_H, Sequence
from .. import ghost
from ..audio.fall_sounds import FallSounds
from ..display.layers import GREY_HIT, GREY_TEXT, mono, px
from ..display.scene import Occluder, RenderContext
from .base import Mode, ShotEvent, ShotOutcome, register


class _Shared:
    """What the two halves of a versus round hold IN COMMON: the clip, its fall
    clip and the pixmap caches. "The same animation on both sides" is also the
    performance answer: in step, one frame is built once and blitted twice.
    This mode is the only content in the app with no cache hit rate (a 120-frame
    clip reuses nothing), so a second copy of the cache would double its cost."""

    def __init__(self) -> None:
        self.seq: "Sequence | None" = None
        self.fall: "Sequence | None | bool" = None
        self.pix: "OrderedDict[tuple, QPixmap]" = OrderedDict()
        self.masks: "OrderedDict[tuple, QPixmap]" = OrderedDict()


@register(key=Qt.Key_9)
class AnimatedFigure(Mode):
    """A filmed figure, looping. Hits are scored and marked on his body."""

    name = "animated"
    title = "Animated Target"
    summary = "A moving figure. Head and chest hits take him down."
    animated = True

    SEQUENCE = "soldier"
    FALL_SEQUENCE = "soldier-fall"
    # FOUR (2026-10-20): "make sure that the soldier goes down after 4 shots in the
    # center mass so that on a 17-mag magazine he's supposed to go down 4 times" --
    # and then "let's make it center mass and the head". Only a hit in the head or in
    # centre mass (derived per frame, `artwork.sequence`) advances the counter; the
    # counter above him and the drop both read this one number. It was 10.
    HITS_TO_DROP = 4
    # How long the counter says "no count" after a hit on the figure that was not in
    # the head or centre mass, timed from the shot's own timestamp.
    NOCOUNT_S = 0.8
    REST_S = 2.0               # he asked for the settled pose to HOLD: "let
                               # the last frame stay as is for a second before
                               # restarting". It already held one second from
                               # the moment the clip ends; this is that second
                               # plus the one he asked for on seeing it.
    FPS = 12.0                 # the clip was reduced to 12fps: 5.8MB, not 52MB
    # Of the screen height, for the TALLEST frame of the clip. The still
    # figures land at 0.552-0.706; this sits inside that band so the lit area
    # is comparable and the figure reads at the same distance.
    HEIGHT_FRAC = 0.62
    FEET_FRAC = 0.94           # where his feet sit, as a fraction of height
    POINTS = 10                # a hit is a hit: there are no rings to grade
    MARK_FRAC = 0.012          # mark radius, of screen height
    FLASH_S = 0.35
    PIXMAP_CACHE = 24          # frames of pixmap held at once (~23MB here)

    # A moving figure needs its own frames, not the panel's 60. Capping the
    # REPAINT REQUEST at the clip's own rate is the same fix the HUD needed:
    # a 4K scene paint costs ~8.5ms against a detector budget of ~7ms, and
    # asking for 60 when 12 is all the content has is pure contention.
    read_rate_hz = FPS
    # TWO PLAYERS: the same clip on both halves, sharing one pixmap cache.
    # MEASURED by tests/test_modes.py section 35 (lit area, and paint cost in
    # step and out of step).
    versus_ready = True

    # THE GHOST. Every hit on him scores POINTS (a limb hit too), so the ceiling is
    # that constant. A perfect shooter lands HITS_TO_DROP head / centre-mass hits,
    # he goes down, and the next soldier is shootable only when the fall clip has
    # played AND the REST_S that follows it is over -- so it waits for all of it.
    ghost_has_waits = True

    def ghost_ceiling(self):
        return ghost.constant_ceiling(self.POINTS)

    def ghost_shots_per_target(self):
        return self.HITS_TO_DROP

    def ghost_expected_wait(self, target_index):
        """The wait for a soldier the round never drew: the fall clip plays once
        (`n` frames at FPS), then REST_S, read the same way `advance()` reads them.
        With no fall clip installed he goes straight to the rest."""
        fall = self.fall_sequence()
        n = len(fall) if fall else 0
        return n / self.FPS + self.REST_S

    @classmethod
    def new_link(cls) -> "_Shared":
        return _Shared()

    def pair_with(self, link: "_Shared | None", side: int) -> None:
        self._link = link
        self._pix = link.pix
        self._masks = link.masks

    def retire(self) -> None:
        """Nothing is shared that a finished half could hold up."""

    def __init__(self) -> None:
        super().__init__()
        self._fall_sounds = FallSounds()      # a seedable draw, see audio/fall_sounds.py
        self._link: "_Shared | None" = None
        self._seq: "Sequence | None" = None
        self._fall: "Sequence | None" = None
        self._t0 = 0.0
        self._frame = 0
        self._origin = (0.0, 0.0)
        self._scale = 1.0
        self._hits = 0
        self._shots = 0
        self._dropped = 0
        self._body_hits = 0          # hits on him outside the kill zones: scored, not counted
        self._nocount_at: "float | None" = None
        # walk -> fall -> rest -> walk. His description exactly: he absorbs
        # ten, he goes down, about a second's break, and he is back up
        # shooting.
        self._state = "walk"
        self._state_at = 0.0
        # The killing shot's own time, kept until he is shootable again: the instant
        # the wait for the next soldier began. The ghost's wait record only.
        self._down_at: "float | None" = None
        self._pix: "OrderedDict[tuple, QPixmap]" = OrderedDict()
        self._masks: "OrderedDict[tuple, QPixmap]" = OrderedDict()

    # -- lifecycle -----------------------------------------------------------

    def sequence(self) -> Sequence:
        if self._link is not None:
            if self._link.seq is None:
                self._link.seq = Sequence(self.SEQUENCE)
            return self._link.seq
        if self._seq is None:
            self._seq = Sequence(self.SEQUENCE)
        return self._seq

    def fall_sequence(self) -> "Sequence | None":
        """The fall clip, or None if it is not installed.

        A MISSING FALL MUST NOT COST THE MODE. If the folder is absent he
        still gets a soldier to shoot at who resets at ten; he just does not
        see him go down. The figure drills take the same line with their
        fall poses -- a figure without one simply disappears.
        """
        if self._link is not None:
            if self._link.fall is None:
                try:
                    self._link.fall = Sequence(self.FALL_SEQUENCE)
                except Exception:
                    self._link.fall = False
            return self._link.fall or None
        if self._fall is None:
            try:
                self._fall = Sequence(self.FALL_SEQUENCE)
            except Exception:
                self._fall = False          # looked, not there; do not retry
        return self._fall or None

    def active(self) -> Sequence:
        """Whichever clip is on screen now."""
        # DOWN IS DOWN. "rest" is the second he asked for AFTER the fall --
        # he is lying on the ground through it, so it must keep showing the
        # fall clip's settled pose. Returning the walk clip here snapped him
        # upright for a second and then restarted him, which is not a break,
        # it is a resurrection.
        if self._state in ("fall", "rest"):
            return self.fall_sequence() or self.sequence()
        return self.sequence()

    def start(self, ctx: RenderContext) -> None:
        seq = self.sequence()
        field = self.playfield(ctx)
        tallest = max(m["h"] for m in seq.frames)
        self._scale = (field.height() * self.HEIGHT_FRAC) / float(tallest)
        # Place the SOURCE frame so his feet land near the bottom of the
        # playfield and he is centred on it. The feet line is the anchor
        # because it is stable within 5px across the clip while his height
        # swings 17%.
        feet = float(seq.frames[0]["feet"])
        cx = float(seq.frames[0]["cx"])
        self._origin = (field.center().x() - cx * self._scale,
                        field.top() + field.height() * self.FEET_FRAC
                        - feet * self._scale)
        self._t0 = ctx.now
        self._frame = 0
        self._hits = self._shots = self._dropped = 0
        self._body_hits = 0
        self._nocount_at = None
        self._state = "walk"
        self._state_at = ctx.now
        self.reset_targets()
        self._down_at = None
        self.note_target(ctx.now, None)       # the first soldier: nothing waited for

    def stop(self) -> None:
        self._pix.clear()
        self._masks.clear()

    def moving_at(self, t: float) -> bool:
        """ALWAYS. He never stands still, so nothing fired at him may teach
        the sensitivity map -- see the module docstring."""
        return True

    # -- the clock -----------------------------------------------------------

    def frame_at(self, t: float) -> int:
        """Which frame of the WALK cycle, looping."""
        seq = self.sequence()
        return int((t - self._t0) * self.FPS) % len(seq)

    def _go(self, state: str, now: float, frame: int = 0) -> None:
        """Enter a state, showing `frame` of whatever clip that state uses.

        THE FRAME IS A PARAMETER BECAUSE IT WAS A BUG. The fall branch set
        `_frame` to the clip's last frame -- him on the ground -- and then
        called this, which reset it to 0. So the pose held through the whole
        break was the FIRST frame of the fall, which is him still standing
        and shooting. He watched it and said exactly that: *"I meant the
        frame when the guy is lying down ... not the first frame of the
        shooting sequence"*.

        A helper with a side effect that undid the line above it, and the
        comment beside that line said it was holding the settled pose.
        """
        self._state = state
        self._state_at = now
        self._frame = frame

    def advance(self, ctx: RenderContext) -> bool:
        before = (self._state, self._frame)
        if self._state == "walk":
            self._frame = self.frame_at(ctx.now)

        elif self._state == "fall":
            # PLAYED ONCE, never looped: he goes down and stays down.
            fall = self.fall_sequence()
            n = len(fall) if fall else 0
            f = int((ctx.now - self._state_at) * self.FPS)
            if n and f < n:
                self._frame = f
            else:
                # HOLD THE LAST FRAME OF THE FALL -- him on the ground. The
                # break is timed from the moment the clip ENDS rather than
                # from when it started, so it is a beat of stillness and not
                # a beat the fall has already eaten.
                self._go("rest", ctx.now, frame=max(0, n - 1))

        elif self._state == "rest":
            if ctx.now - self._state_at >= self.REST_S:
                # "and then the soldier appears and he's shooting again"
                self._t0 = ctx.now
                self._hits = 0
                self._nocount_at = None
                self._go("walk", ctx.now)
                self._frame = 0
                self.note_target(ctx.now, self._down_at)    # shootable again
                self._down_at = None

        return (self._state, self._frame) != before

    # -- the shot ------------------------------------------------------------

    def shot(self, ev: ShotEvent, ctx: RenderContext) -> ShotOutcome:
        """Scored against the frame that was on screen when the shot broke."""
        self._shots += 1

        # NOTHING COUNTS WHILE HE IS GOING DOWN OR RESTING. He is already
        # hit; the figure drills refuse shots during their own fall for the
        # same reason, and the sensitivity learner refuses them too.
        if self._state != "walk":
            return ShotOutcome.miss()

        seq = self.sequence()
        # The shot carries its OWN timestamp -- the first frame of its pulse.
        # Testing against `self._frame` would score it against wherever the
        # figure has got to by the time this runs, which is the fault he
        # reported on the balls: "I have to shoot in front of the ball".
        f = self.frame_at(ev.t)
        if not seq.on_figure(f, ev.x, ev.y, self._origin, self._scale):
            return ShotOutcome.miss()

        # WHERE ON HIM, in the frame that was up when the shot broke (`f`, from
        # `ev.t`), never the one up now. The head and centre mass count; the rest
        # of him (a limb, a pistol) still scores but does not advance the counter.
        # From the firing line a limb hit and a clean miss would look identical, so
        # the COUNTER says so for a moment ("no count"): he may not be marked.
        zone = seq.zone_at(f, ev.x, ev.y, self._origin, self._scale)
        if zone is None:
            self._body_hits += 1
            self._nocount_at = ev.t
            return ShotOutcome(hit=True, points=self.POINTS,
                               target_id=f"{self.SEQUENCE}:body")

        self._hits += 1
        if self._hits >= self.HITS_TO_DROP:
            # "when he absorbs 4 ... you should go into the fallen soldier
            # animation". Interrupts the walk where it stands -- waiting for
            # the cycle to finish would put up to a second between the shot
            # that killed him and him falling.
            self._dropped += 1
            self._down_at = ev.t
            self._go("fall", ev.t)
            return ShotOutcome(hit=True, points=self.POINTS,
                               target_id=f"{self.SEQUENCE}:down",
                               sound=self._fall_sounds.pick(self.SEQUENCE))
        return ShotOutcome(hit=True, points=self.POINTS,
                           target_id=f"{self.SEQUENCE}:{zone}")

    def aim_point(self, ctx: RenderContext):
        """Where the walk-through should aim: the middle of his chest.

        The optional hook `_modetest_target` asks for. A mode that cannot
        answer is reported as untestable, and a figure drill cannot be aimed
        at from an attribute -- it is a scaled drawing that moves every frame.
        Taken from the CURRENT frame's crop so it tracks him.
        """
        seq = self.active()
        left, top, w, h = seq.box(self._frame, self._origin, self._scale)
        return (left + w * 0.5, top + h * 0.35)

    def status(self) -> str:
        pct = (100.0 * (self._hits + self._dropped * self.HITS_TO_DROP)
               / self._shots) if self._shots else 0.0
        return (f"absorbed {self._hits}/{self.HITS_TO_DROP}   "
                f"dropped {self._dropped}   "
                f"{self._shots} shots" + (f"   {pct:.0f}% on him" if self._shots else ""))

    # -- drawing -------------------------------------------------------------

    def _pixmap(self, i: int) -> QPixmap:
        """The frame at ITS OWN size, as line art with a transparent ground.

        Alpha tracks brightness, exactly as `Sprite` does it, so the black
        field stays black instead of covering what is beneath it.

        BUILT AT SOURCE SIZE AND SCALED BY THE BLIT, which is the whole
        difference between this mode being usable and not. The first version
        rebuilt the pixmap at the DRAWN size every frame with a smooth
        transform, and the clip cycles all 120 frames so nothing was ever
        reused: 176ms a frame, against a detector budget of ~7ms and a cliff
        at ~14ms where the capture loop drops from 60fps to 30 and halves the
        camera frames inside a 100ms pulse. Scaling at blit time costs a
        fraction of that and looks the same at this magnification.

        The cache is bounded for the reason in `sequence.py`: 120 frames held
        at once would be ~115MB, and CLAUDE.md records a figure cache that
        holds 4.1MiB per image and never evicts.
        """
        key = (self._state, i)
        got = self._pix.get(key)
        if got is not None:
            self._pix.move_to_end(key)
            return got
        g = self.active().grey(i)
        gain = self.active().ink_gain
        if gain != 1.0:
            g = np.clip(g.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        v = np.ascontiguousarray(g)
        gh, gw = v.shape
        bgra = np.empty((gh, gw, 4), np.uint8)
        bgra[..., 0] = v
        bgra[..., 1] = v
        bgra[..., 2] = v
        bgra[..., 3] = v                       # premultiplied: colour <= alpha
        img = QImage(bgra.data, gw, gh, gw * 4,
                     QImage.Format_ARGB32_Premultiplied).copy()
        got = QPixmap.fromImage(img)
        self._pix[key] = got
        while len(self._pix) > self.PIXMAP_CACHE:
            self._pix.popitem(last=False)
        return got

    def _mask(self, i: int) -> QPixmap:
        """This frame's silhouette as an alpha mask, for the holes beneath.

        The SAME silhouette `on_figure` scores against, so what hides a hole
        and what counts as a hit on him are one shape -- `figure.occluder`
        takes the same care for the held figures.

        Built from the 1/8 silhouette (160x90 here, ~58KB) and stretched by
        the painter, never at the drawn size: a 4K-sized mask per frame is
        the 176ms mistake this mode already paid for once, and a walking clip
        reuses nothing. Bounded like `_pix` for the same reason -- 151 frames
        of mask held for ever is the un-evicted figure cache again.
        """
        key = (self._state, i)
        got = self._masks.get(key)
        if got is not None:
            self._masks.move_to_end(key)
            return got
        sil = self.active().silhouette(i)
        sh, sw = sil.shape
        a8 = np.ascontiguousarray(sil.astype(np.uint8) * 255)
        bgra = np.ascontiguousarray(np.repeat(a8[..., None], 4, axis=2))
        img = QImage(bgra.data, sw, sh, sw * 4,
                     QImage.Format_ARGB32_Premultiplied).copy()
        got = QPixmap.fromImage(img)
        self._masks[key] = got
        while len(self._masks) > self.PIXMAP_CACHE:
            self._masks.popitem(last=False)
        return got

    def occluders(self, ctx: RenderContext) -> "list[Occluder]":
        """He walks IN FRONT OF the holes, never behind them.

        His report, 2026-10-18, with two photographs of the screen: *"you are
        drawing the shots on the screen and the soldier moves BEHIND those
        shots -- the soldier needs to move in front of the shots"*.

        Z-order was never the problem and raising it would not have fixed it:
        the holes are already underneath. The clip is line art with a
        TRANSPARENT ground (alpha tracks brightness), so every dark part of
        him -- which is most of him -- let the holes show straight through,
        and he read as a ghost drifting over them.

        The figure drills met this on 2026-10-17 and gained `Occluder` for it;
        this mode simply never published one. Fixing the instance and leaving
        the next content standing is the failure RULES Part E exists for, so
        the right question was not "add a z" but "which modes draw something
        solid and say nothing about it".
        """
        if ctx.calibrating:
            return []
        seq = self.active()
        i = self._frame
        left, top, w, h = seq.box(i, self._origin, self._scale)
        return [Occluder(self.z,
                         QRect(int(left), int(top), int(w), int(h)),
                         lambda i=i: self._mask(i))]

    def paint(self, p: QPainter, ctx: RenderContext) -> None:
        seq = self.active()
        i = self._frame
        left, top, w, h = seq.box(i, self._origin, self._scale)
        pm = self._pixmap(i)
        p.drawPixmap(QRectF(left, top, w, h), pm, QRectF(pm.rect()))

        # WHAT HE ABSORBED, NOT WHERE. His instruction: "You don't paint
        # those shots on the soldier. You just display the number of shots
        # he absorbed."
        #
        # It also keeps the screen the camera watches clean: a mark drawn on
        # a moving figure is new lit area on every frame, and this mode is
        # already the only content here with no pixmap cache hit rate.
        p.setPen(QColor(GREY_TEXT))
        p.setFont(mono(px(ctx, 30)))
        p.drawText(int(left), int(top) - px(ctx, 12), self.counter_label(ctx.now))

    def counter_label(self, now: float) -> str:
        """What the counter above him says now: n / THRESHOLD, or DOWN, plus
        "HIT - NO COUNT" for NOCOUNT_S after a hit on him that was not in the head
        or centre mass (no mark on a moving figure: the feedback lives here).
        A method so the words are tested and the painter reads the same ones."""
        label = (f"{self._hits} / {self.HITS_TO_DROP}" if self._state == "walk"
                 else "DOWN")
        if (self._state == "walk" and self._nocount_at is not None
                and 0.0 <= now - self._nocount_at < self.NOCOUNT_S):
            label += "   HIT - NO COUNT"
        return label

