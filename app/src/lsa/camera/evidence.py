"""What the detector almost saw, kept for a few seconds so a missing shot can be explained.

Plan 02 step 9. The microphone is the one channel that counts trigger pulls
independently of the camera, and the finding it produces most often is "HEARD a
shot with no detection". From the firing line that is three different events
that need three different answers:

  - the dot was there but too faint (a sub-threshold blob: fix is exposure or
    sensitivity),
  - the dot was there but lasted one frame (a too-short rejection),
  - the dot was nowhere the camera looks (off the panel, on the bezel): no
    blob of any brightness, on or off the screen.

The detector already measures all of these every frame; `Stats` keeps only the
last second's maximum, which smears together everything near a click. This
keeps the per-frame values for ~3 s so the window can be exactly "0.25 s
either side of the click".

Pure Python, no Qt, no numpy: it is filled from the capture thread's loop, once
per camera frame, so it is one small tuple and a pop. Reading it
(`summarise`) takes a copy first, so the GUI thread never iterates a deque the
capture thread is appending to.
"""

from __future__ import annotations

from collections import deque


class FrameEvidence:
    """A short ring of per-frame near-miss telemetry."""

    KEEP_S = 3.0

    def __init__(self) -> None:
        # (t, frame_peak, dim_peak, dim_pos, off_screen_peak, rejected_runs_total,
        #  amp_gate, halo, big) where amp_gate is the detector's `last_amp_gate`
        #  or None, halo is its `last_halo` (peak, x, y) or None and big is its
        #  `last_big_rejected` (peak, x, y, area) or None: the brightest blob
        #  `max_area` still rejected, which is NOT a shot the camera missed
        self._ring: deque[tuple] = deque(maxlen=512)

    def push(self, t: float, frame_peak: int, dim_peak: int,
             dim_pos: "tuple[float, float] | None", off_peak: int,
             rejected_runs_total: int,
             amp_gate: "tuple | None" = None,
             halo: "tuple | None" = None,
             big: "tuple | None" = None) -> None:
        ring = self._ring
        ring.append((t, frame_peak, dim_peak, dim_pos, off_peak, rejected_runs_total,
                     amp_gate, halo, big))
        while ring and t - ring[0][0] > self.KEEP_S:
            ring.popleft()

    def clear(self) -> None:
        self._ring.clear()

    def summarise(self, t0: float, t1: float) -> "dict | None":
        """The brightest of each kind of near miss in [t0, t1], or None when no
        frame was recorded in that window (a stalled camera, or a click older
        than the buffer)."""
        rows = list(self._ring)             # a copy: the capture thread appends
        inside = [r for r in rows if t0 <= r[0] <= t1]
        if not inside:
            return None
        before = [r for r in rows if r[0] < t0]
        base = before[-1] if before else inside[0]
        dim = max(inside, key=lambda r: r[2])
        gated = [r[6] for r in inside if r[6] is not None]
        amp = None
        if gated:
            area, need, n, x, y = max(gated, key=lambda g: g[0])
            amp = {"area": area, "need": need, "n": n, "pos": (x, y)}
        haloed = [r[7] for r in inside if r[7] is not None]
        halo = None
        if haloed:
            pk, x, y = max(haloed, key=lambda h: h[0])
            halo = {"peak": pk, "pos": (x, y)}
        bigs = [r[8] for r in inside if r[8] is not None]
        big = None
        if bigs:
            pk, x, y, ar = max(bigs, key=lambda b: b[0])
            big = {"peak": pk, "pos": (x, y), "area": ar}
        return {
            "frames": len(inside),
            "frame_peak": max(r[1] for r in inside),
            "dim_peak": dim[2],
            "dim_pos": dim[3] if dim[2] > 0 else None,
            "off_peak": max(r[4] for r in inside),
            "short_delta": max(0, inside[-1][5] - base[5]),
            "amp": amp,
            "halo": halo,
            "big": big,
        }
