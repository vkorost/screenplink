"""Laser spot detection and multi-shot tracking.

Two separate jobs, deliberately kept apart:

  LaserDetector  per-frame: where are the laser dots in this image?
  ShotTracker    across frames: which runs of detections constitute which shots?

The tracker exists because a LaserLyte trainer emits a ~100 ms pulse, which at
30 fps lands in about three consecutive frames. That is not three hits, it is
one, and the redundancy is useful: a single-frame blip is almost certainly
sensor noise or a stray reflection, while three frames in the same place is a
shot. We also average the centroid over the run, which cuts positional jitter.

Both halves are plural on purpose. Shooting two guns at once is a normal thing
to do, so the detector returns every credible blob in the frame and the tracker
maintains one track per dot, associating detections to tracks by proximity.
A single-blob detector would quietly merge two simultaneous hits into one.

What we cannot do: tell the guns apart. Two 650 nm diodes are optically
identical, so a hit is a hit. Per-gun attribution would need different
wavelengths.

Off-axis note: when the camera views the screen at an angle, a laser dot images
as an ellipse rather than a circle, and gets more elongated the more oblique the
view. Nothing here assumes roundness. The optional elongation filter is off by
default and exists only to reject long specular streaks.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class ScoreMode(str, Enum):
    """How to reduce a colour frame to a single laser-likelihood channel."""

    RED = "red"            # plain red channel; best with heavy underexposure
    CHROMA = "chroma"      # R - max(G, B); rejects anything neutral or bright-white
    GRAY = "gray"          # luminance; the right choice once a 650nm filter is fitted

    @property
    def description(self) -> str:
        return {
            ScoreMode.RED: "Red channel. Simple and fast. Needs the screen underexposed to a ghost.",
            ScoreMode.CHROMA: "R minus max(G,B). Ignores white/grey highlights, keeps saturated red.",
            ScoreMode.GRAY: "Luminance. Use this when a 650nm bandpass filter is on the lens.",
        }[self]


@dataclass
class DetectorParams:
    mode: ScoreMode = ScoreMode.CHROMA
    threshold: int = 60            # on the score image, 0-255
    min_peak: int = 80             # reject dim blobs outright
    blur: int = 3                  # odd kernel; 0 disables. Steadies the centroid.
    min_area: int = 2              # px, at processing scale
    # SIZE EVIDENCE WHERE BRIGHTNESS EVIDENCE HAS BEEN AMPLIFIED.
    #
    # The sensitivity map divides the score by the local relative
    # sensitivity, so where it amplifies by N the peak of a blob no longer
    # means what it did: noise that scored 3 to 5 is reported at 10 to 19,
    # against a bar of 11. The BLOB is untouched by the map. Measured on the
    # session of 2026-10-17 08:43 (microphone as ground truth, far side of
    # the screen, N up to 4): shots a click confirmed covered 130 px at the
    # smallest and 341 px at the median; detections nothing confirmed covered
    # 12 px at the median. So where the map amplifies, the blob must be big
    # enough to be a dot. `amp_area_per_gain` is how much AREA each unit of
    # amplification above 1 demands, in FULL-FRAME pixels (the unit the log
    # and the csv report):
    #
    #     required area = max(min_area, amp_area_per_gain * (N - 1))
    #
    # Zero where N is 1 and up to about 1.1, so a screen with no map, and
    # the part of a screen the map leaves alone, behave exactly as before.
    #
    # DERIVATION OF 16. The smallest strong corroborated shot where N >= 3
    # covered 130 px, at N = 3.73 (SHOT 91). The project asks a shot for a
    # 3x margin over its bar everywhere else, so the bar there is a third of
    # that, 43 px, and 43 / (3.73 - 1) = 15.9, rounded to 16. That is
    # 48 px at the map's floor of N = 4, 32 at N = 3, 16 at N = 2. Set to 0
    # to disable. Read the units in `amplified_min_area`.
    #
    # THE MARGIN THIS LEAVES, MEASURED, AND IT IS THIN. Blob area depends on
    # the threshold, and 16 was derived at this project's lowest-ever bar.
    # From camera_lab.guntest.log (2026-10-14, min_peak 25 -- about four
    # times tighter than the 7-8 of today): the smallest real shot on the
    # far side covered 52 px (SHOT 7, camera x 882, peak 33) against a 48 px
    # ask at N = 4. That is a margin of 1.08x, the one number here that can
    # cost a real shot. Whoever raises a threshold shrinks every blob and
    # moves it silently: re-run tools/replay.py and read the far-side row.
    amp_area_per_gain: float = 16.0
    # Generous on purpose. Blob area depends on the THRESHOLD, not just on the
    # dot: at a high threshold a laser measures 7-60 px, but once auto-calibration
    # drove the noise floor down to 5 and the threshold to 17, the same dots
    # bloomed to 220-522 px. A tight limit tuned at one threshold silently
    # rejects real shots at another, so size is a sanity check here, not a
    # discriminator -- the screen mask, the pulse-duration ceiling and the
    # refractory window are what actually separate shots from everything else.
    max_area: int = 1500
    # A BIG BLOB IS NOT A REJECTION, IT IS A QUESTION (2026-10-17, the near
    # side). `max_area` was written to throw away lamps and reflections, and
    # on the near side it threw away his brightest shots instead: a dot
    # blooms there, the camera being close and looking into a matte surface
    # at a steep angle. Live, in mode 1, 7 of 13 shot-bright blobs (peak
    # 98-109, his median is 85) were discarded as 'too big', 1673 to 9665 px,
    # while specks of peak 9-24 got the holes.
    #
    # The ceiling was never what separated a lamp from a laser. DURATION does
    # that (`ShotTracker.max_duration_ms`), and BRIGHTNESS separates a bloom
    # from a patch of room. Measured over the four sessions of 2026-10-17
    # (35,156 + 17,269 + 6,974 + 3,204 frames, every blob dumped before any
    # filter):
    #   oversize events with peak >= 30   168 (bright)    peak 31-203,
    #       median 104; EVERY ONE lasted 17-239 ms bar one
    #   the one that did not              1371 ms, 20 frames, peak 39,
    #       a steady object -- the duration ceiling kills it at 3.4x margin
    #   oversize events with peak < 30     22 (faint)     peak 8-16, areas
    #       1556-6843, all <= 97 ms: not a pulse of this laser
    # and the 30 is not new: it is the project's measured laser/noise
    # boundary (see `ShotTracker._confirmed`). So a blob over `max_area` is
    # ACCEPTED as a detection when its peak reaches `oversize_min_peak`, and
    # still rejected as 'area' below it. 0 restores the old absolute ceiling.
    oversize_min_peak: int = 30
    # WHERE A BLOOM'S DOT IS. Its intensity-weighted centroid is pulled by
    # the halo (median 9 px, p90 23-28 px, max 58-136 px from the dot's own
    # later detection); the CORE -- the plain mean of the pixels at or above
    # this fraction of the blob's peak -- is tighter (37 events, session
    # 155319, against a later normal-size detection of the same pulse):
    #       weighted   median 8.6 px   p90 23.3   max 58
    #       core 0.7   median 8.9 px   p90 14.5   max 19
    #       core 0.85  median 9.6 px   p90 15.5   max 19
    # 0.7 is the tightest at p90 and max; the choice between 0.6 and 0.85 is
    # inside the noise of 37 samples. Applied ONLY to blobs over `max_area`,
    # so every normal-size detection is positioned exactly as before.
    oversize_core_frac: float = 0.7
    close_iter: int = 1            # morphological close: fills the hole in a saturated donut
    use_background: bool = True    # subtract a running model of the (known, slow) screen content
    bg_alpha: float = 0.05         # how fast that model adapts
    scale: float = 1.0             # process at reduced resolution for speed
    max_blobs: int = 4             # two guns plus headroom; brightest first
    min_separation: float = 10.0   # px; closer than this and we keep only the brighter blob
    # ONE ILLUMINATION EVENT IS ONE SHOT (2026-10-17, the left side of the
    # screen). A pulse blooms: the matte coating scatters it into one large
    # red halo with the dot inside, plus detached lumps of that halo. The
    # halo itself is thrown away by `max_area` BEFORE `_suppress_neighbours`
    # runs, so the lumps survive alone, each clears the bar, and each became
    # a shot, a hole and a gunshot. `min_separation` (10 px) could not
    # catch them: they sit 25-190 px from the bloom.
    #
    # A candidate is a halo lump when a much brighter event is NEAR it, where
    # "event" includes the blob `max_area` rejected, and "near" scales with
    # the event's own size. Both numbers are MEASURED, from the three
    # recordings of 2026-10-17 (video replay, raw pre-filter blob list):
    #   halo_ratio       lump peak / event peak: 0.07-0.27 across ~150 lump
    #                    pairs (p90 0.17), against 0.51 for the one real
    #                    two-gun-looking pair on record. 0.30 sits 11% above
    #                    the worst lump and 41% below that pair. FITTED
    #                    between two measured populations, not derived.
    #   halo_radius_diag lump distance / event bbox diagonal: 0.05-1.88
    #                    (p90 1.40). 3.0 is 1.6x the worst lump. MEASURED
    #                    range, FITTED margin. It is a multiple of the
    #                    event's EXTENT because the bloom is: a 187x225 halo
    #                    throws lumps 190 px out, a 36x32 dot throws them
    #                    40 px out, and no constant covers both.
    # Spatial and within ONE frame on purpose: his real shots are 0.34-0.60 s
    # apart and 100-200 px apart, so any time-based lockout eats fast
    # follow-ups. 0 disables either half.
    #   halo_core_frac   a candidate within this fraction of the event's
    #                    diagonal of its CENTROID is the event's own core, not
    #                    a lump of its halo, and is never folded. Measured
    #                    2026-10-17 session 143752 frames 1612-1614: a 6639 px
    #                    bloom (oversize, so itself rejected) with one 21 px
    #                    speck 7 px from its centre, the ONLY report of a real
    #                    shot there (microphone 0.373 against a bar of 0.19).
    #                    That speck sits at 0.05 of the diagonal; every lump in
    #                    that session sits at 0.34 or more. the first fit, 0.15, sat in
    #                    that gap (first fitted at 0.15 from this ONE event).
    #                    Widened from that single event with the
    #                    placement of the GENUINE dot inside its own bloom:
    #                    across the three sessions, 59 oversize blobs were
    #                    followed within 8 frames by a real (peak >= 30,
    #                    normal-sized) detection, and 56 of those lay at
    #                    0.00-0.24 of the bloom's diagonal from its centroid
    #                    (median ~0.10; the rest were unrelated shots 0.7+
    #                    away). So 0.25 covers where a core really sits, and
    #                    stays under 0.34, the nearest lump in the failing
    #                    session. FITTED: the lower bound is measured
    #                    (n ~ 59) but the upper bound rests on ONE session's
    #                    lumps (session 143752); the other two sessions have
    #                    lumps at 0.06-0.25 which this leaves detected, as
    #                    before. What would settle it: a session recorded
    #                    with the introspection server running, because
    #                    `/score` returns the oversize blobs that
    #                    detections.csv does not.
    halo_ratio: float = 0.30
    halo_radius_diag: float = 3.0
    halo_core_frac: float = 0.25
    max_elongation: float = 0.0    # 0 disables. Rejects long streaks, allows oblique ellipses.
    # Camera-space polygon of the monitor. Anything outside it cannot be a hit,
    # because a hit is by definition a laser dot on the screen. Measured on this
    # setup: with no mask, 1040 of 1356 detections came from a bright window at
    # the edge of frame -- 77% of everything, all of it noise.
    roi_polygon: tuple[tuple[float, float], ...] | None = None
    roi_margin: float = 0.0        # px to shrink the polygon by; trims edge glare

    def sanitized(self) -> "DetectorParams":
        p = DetectorParams(**vars(self))
        if p.blur and p.blur % 2 == 0:
            p.blur += 1
        p.scale = min(1.0, max(0.1, p.scale))
        p.bg_alpha = min(1.0, max(0.001, p.bg_alpha))
        p.max_blobs = max(1, p.max_blobs)
        return p


@dataclass
class Detection:
    """One laser spot in one frame. Coordinates are full-frame pixels, sub-pixel precise."""

    x: float
    y: float
    area: float
    peak: int
    frame_index: int
    timestamp: float
    nx: float = 0.0           # normalized 0..1 across the camera frame
    ny: float = 0.0
    width: float = 0.0        # bounding box, full-frame px
    height: float = 0.0
    elongation: float = 1.0   # long side / short side; >1 means an oblique or smeared dot

    def distance_to(self, other: "Detection") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Shot:
    """A run of detections treated as a single trigger pull."""

    x: float
    y: float
    nx: float
    ny: float
    frames: int
    peak: int
    area: float
    first_timestamp: float
    last_timestamp: float
    duration_ms: float      # estimated laser-on time, including one frame period of slack
    index: int = 0          # 1-based, stable between the provisional and final report
    provisional: bool = False   # position may still be refined when the track closes
    track_id: int = 0       # which track produced it; distinguishes concurrent shots
    elongation: float = 1.0
    members: list[Detection] = field(default_factory=list)


def amplified_min_area(params: "DetectorParams", amplification: float) -> float:
    """The smallest blob that counts, at this local amplification.

    Returned in PROCESSING-scale pixels, the unit `min_area` and the blob
    areas in `_blobs` use. `amp_area_per_gain` is stated in full-frame pixels
    because that is what the log reports, so it is converted here, once:
    areas scale with the square of the linear scale, so a detector run at
    scale 0.5 sees a quarter of the pixels and must ask for a quarter of the
    area. Getting this wrong by the square of the scale would either do
    nothing or reject everything.

    One pure function, so the detector, the replay tool and the tests cannot
    disagree about what the rule is (RULES C14: test across the seam).
    """
    per = float(params.amp_area_per_gain)
    if per <= 0.0 or amplification <= 1.0:
        return float(params.min_area)
    s = params.scale if params.scale < 0.999 else 1.0
    return max(float(params.min_area), per * (float(amplification) - 1.0) * s * s)


def full_frame_to_working_area(params: "DetectorParams", area: float) -> float:
    """An area in FULL-FRAME pixels (`Detection.area`, the log, the csv),
    expressed in the PROCESSING-scale pixels `amplified_min_area` returns.

    Areas scale with the square of the linear scale. The replay tool compares
    a csv area against `amplified_min_area`; without this the two sides are
    different units and agree only at scale 1.0. Same convention as
    `amplified_min_area` (a scale of 0.999 or more is 1.0), and
    `tests/test_detector.py` asserts the two stay consistent (RULES C14).
    """
    s = params.scale if params.scale < 0.999 else 1.0
    return float(area) * s * s


# Rejected-blob recording (see `LaserDetector.record_rejects`). A blob is only
# worth a row if it was close to counting: a known peak under this fraction of
# min_peak is noise and is not recorded. The per-frame cap bounds both the
# file and the work, because the capture loop has ~7 ms a frame and crossing
# ~14 ms halves the frames inside a pulse.
REJECT_PEAK_FRACTION = 0.5
REJECT_ROWS_PER_FRAME = 6


class LaserDetector:
    """Finds every credible laser spot in a frame, brightest first."""

    def __init__(self, params: DetectorParams | None = None) -> None:
        self.params = (params or DetectorParams()).sanitized()
        self._bg: np.ndarray | None = None
        # Kept buffers for the hot path. Declared here, not conjured with
        # getattr on first use: a plausible attribute name read through a
        # default returned 0.0 for days while a guard never fired (C8).
        self._bg8: np.ndarray | None = None      # the model, as uint8
        self._gain: np.ndarray | None = None
        self._gain_scaled: np.ndarray | None = None
        self._recip: np.ndarray | None = None    # 1/gain, at frame size
        self._frame_index = -1
        self.last_score: np.ndarray | None = None   # for the mask preview
        # The brightest blob `max_area` still rejected this frame, inside the
        # screen: (peak, x, y, area) full-frame px, or None. Its own slot so a
        # shot lost to size is reported as such, never as 'saw nothing'.
        self.last_big_rejected: "tuple[int, float, float, float] | None" = None
        self.last_mask: np.ndarray | None = None
        self.last_process_ms = 0.0
        self.last_candidates = 0                    # blobs that passed every filter
        self.last_rejected_roi = 0                  # blobs thrown out for being off-screen
        # --- near-miss telemetry -------------------------------------------
        # The brightest thing in the frame, whether or not it passed anything,
        # and the brightest candidate each filter threw away. From the firing
        # line "a blip just under the threshold" and "the camera saw nothing at
        # all" are the same event -- no hole, no sound -- and they need
        # opposite fixes. Nothing downstream could tell them apart because
        # nothing upstream measured them.
        self.last_frame_peak = 0                    # brightest score anywhere
        self.last_dim_peak = 0                      # brightest blob rejected as too faint
        # Where that blob was (centre of its bounding box, full-frame px), so
        # "heard a shot, saw nothing" can say WHERE the faint thing was.
        # Telemetry only: nothing in the detector reads it.
        self.last_dim_pos: tuple[float, float] | None = None
        self.last_off_screen_peak = 0               # brightest blob rejected as off-screen
        self.last_rejects: dict[str, int] = {}      # reason -> how many
        # The brightest candidate folded into a brighter event's halo this
        # frame: (peak, x, y), or None. Same job as `last_dim_pos`: "heard a
        # shot, nothing announced" must be able to say it was seen and folded.
        self.last_halo: tuple[int, float, float] | None = None
        # The largest blob the size gate (`amplified_min_area`) rejected this
        # frame: (area, area it needed, local amplification, x, y), areas and
        # position in FULL-FRAME units. The gate throws a blob away before its
        # peak is computed, so `last_dim_*` never see it; without this a shot
        # the gate refused was reported as "the camera saw nothing".
        self.last_amp_gate: tuple[float, float, float, float, float] | None = None
        # RECORDING ONLY. When `record_rejects` is on (the capture thread turns
        # it on while a session is being recorded and for no other reason),
        # `_blobs` keeps one row per blob it threw away, so a rejection rule can
        # be A/B-tested later against the detector's OWN rejections instead of
        # a re-encoded video. Rows are (reason, x, y, peak, area, w, h,
        # elongation) in FULL-FRAME units, at most REJECT_ROWS_PER_FRAME a
        # frame. Nothing in the detector reads either field, and with the flag
        # off no extra work is done at all.
        self.record_rejects = False
        self.last_rejected_blobs: "list[tuple]" = []
        self._roi_path: np.ndarray | None = None
        # The same polygon as an image, so whole-frame work can be confined
        # to it. Rebuilt whenever the polygon, the margin or the working
        # frame shape changes.
        self._roi_mask: np.ndarray | None = None
        self._roi_mask_key: tuple | None = None
        self._off_peak_outside = 0
        self._off_count_outside = 0
        self._gain: np.ndarray | None = None      # relative sensitivity, 0..1
        self._gain_scaled: np.ndarray | None = None

    def set_gain_map(self, gain: "np.ndarray | None") -> None:
        """Flatten the detector's sensitivity across the screen.

        A dot on the far side of an obliquely-viewed screen is dimmer and
        spread over fewer pixels, so a single threshold is simultaneously too
        strict there and too loose near the camera. Dividing the score by the
        measured falloff makes one threshold mean the same thing everywhere.

        Pass None to go back to a flat score. Measuring the noise floor AFTER
        setting this is not optional: normalising amplifies noise in the weak
        regions too, so a floor measured on the un-normalised score would now
        be an underestimate.
        """
        self._gain = None if gain is None else np.asarray(gain, dtype=np.float32)
        self._gain_scaled = None
        self._recip = None   # rebuilt at the size the frames actually are
        self._bg = None      # the score image changed meaning

    def set_params(self, params: DetectorParams) -> None:
        new = params.sanitized()
        if new.mode != self.params.mode or new.scale != self.params.scale:
            self._bg = None          # score image changed meaning; old model is garbage
        if new.roi_polygon != self.params.roi_polygon or new.roi_margin != self.params.roi_margin:
            self._roi_path = None    # rebuilt lazily on the next frame
            self._roi_mask = None
            self._roi_mask_key = None
        self.params = new

    def _roi_image(self, shape: tuple[int, int]) -> np.ndarray | None:
        """The screen polygon as a mask over the WORKING frame, cached.

        Returns None when there is no polygon, which means every pixel is
        in play -- the honest state before calibration, and the one the
        detector has to keep working in.
        """
        poly = self.params.roi_polygon
        if not poly or len(poly) < 3:
            return None
        scale = self.params.scale if self.params.scale < 0.999 else 1.0
        key = (shape, poly, self.params.roi_margin, scale)
        if self._roi_mask is not None and self._roi_mask_key == key:
            return self._roi_mask
        m = np.zeros(shape, dtype=np.uint8)
        pts = (np.array(poly, dtype=np.float32) * scale).astype(np.int32)
        cv2.fillConvexPoly(m, pts, 255)
        margin = float(self.params.roi_margin or 0.0)
        if margin > 0:
            # `_inside_roi` requires a blob to sit `margin` px INSIDE the
            # polygon, so the image form has to be eroded by the same amount
            # or the two tests would disagree at the edge.
            r = max(1, int(round(margin * scale)))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            m = cv2.erode(m, k)
        self._roi_mask = m
        self._roi_mask_key = key
        return m

    def _inside_roi(self, x: float, y: float) -> bool:
        """Is this point on the monitor?

        cv2.pointPolygonTest handles the general case, including the trapezoid an
        off-axis camera sees, which a bounding box would not: at an angle, a box
        around the screen can include a large wedge of the room beside it.
        """
        poly = self.params.roi_polygon
        if not poly or len(poly) < 3:
            return True
        if self._roi_path is None:
            self._roi_path = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        dist = cv2.pointPolygonTest(self._roi_path, (float(x), float(y)), True)
        return dist >= self.params.roi_margin

    def reset_background(self) -> None:
        self._bg = None

    # -- score image ----------------------------------------------------------

    def _score(self, frame: np.ndarray) -> np.ndarray:
        if self.params.mode is ScoreMode.GRAY:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(frame)
        if self.params.mode is ScoreMode.RED:
            return r
        # CHROMA: how much more red than the strongest of the other two channels.
        # cv2.subtract saturates at 0, so no clipping by hand is needed.
        return cv2.subtract(r, cv2.max(g, b))

    def _normalise(self, score: np.ndarray) -> np.ndarray:
        """Divide out the measured sensitivity falloff, if one is known.

        MEASURED, 2026-10-16, 1280x720 on this machine:

            astype(float32), divide, clip, astype(uint8)    4.70 ms
            cv2.multiply by a precomputed reciprocal        0.45 ms

        A tenth of the cost, and it matters more than a tenth: his saved
        sensitivity map is applied on every live launch, so the detector
        measures 13.9 ms a frame with it against 8.9 ms without -- and the
        capture loop is `read + detect`, so under about 14 ms it holds 60
        fps and over it falls to 30. That halves the camera frames inside a
        100 ms pulse, which is what decides whether a trigger pull
        registers. It is a cliff, not a slope.
        """
        if self._gain is None:
            return score
        h, w = score.shape[:2]
        if self._recip is None or self._recip.shape[:2] != (h, w):
            g = (self._gain if self._gain.shape[:2] == (h, w)
                 else cv2.resize(self._gain, (w, h),
                                 interpolation=cv2.INTER_LINEAR))
            self._gain_scaled = g
            # Reciprocal ONCE, at the size it will be used at, so the hot
            # path is a single multiply. `cv2.multiply` with dtype CV_8U
            # saturates for us, which is what the clip was doing by hand.
            self._recip = (1.0 / np.maximum(g, 1e-3)).astype(np.float32)
        return cv2.multiply(score, self._recip, dtype=cv2.CV_8U)

    # -- main entry point -----------------------------------------------------

    def process(self, frame: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        """Returns every credible spot in the frame, sorted brightest first."""
        t0 = time.perf_counter()
        p = self.params
        self._frame_index += 1
        ts = time.perf_counter() if timestamp is None else timestamp

        full_h, full_w = frame.shape[:2]
        work = frame
        if p.scale < 0.999:
            work = cv2.resize(frame, None, fx=p.scale, fy=p.scale, interpolation=cv2.INTER_AREA)

        score = self._normalise(self._score(work))
        if p.blur:
            score = cv2.GaussianBlur(score, (p.blur, p.blur), 0)

        detect_on = score
        if p.use_background:
            if self._bg is None or self._bg.shape != score.shape:
                self._bg = score.astype(np.float32)
            # `convertScaleAbs` into a kept buffer rather than `astype`:
            # 0.85 ms -> 0.15 ms, and measured bit-identical on real frames.
            # The background is non-negative, so the abs is a no-op.
            if self._bg8 is None or self._bg8.shape != score.shape:
                self._bg8 = np.empty_like(score)
            cv2.convertScaleAbs(self._bg, dst=self._bg8)
            detect_on = cv2.subtract(score, self._bg8)

        _, mask = cv2.threshold(detect_on, p.threshold, 255, cv2.THRESH_BINARY)
        if p.close_iter:
            # A saturated spot can read as a ring in CHROMA mode: the core blows
            # out all three channels, so it has no chroma at all. Closing fills it.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=p.close_iter)

        # Blank a few pixels at the frame border. This camera puts a fixed
        # artefact at (0,0) that scores 71-76 in chroma -- brighter than most
        # real hits -- and it was the single brightest thing in an entire
        # recording. Anything within a couple of pixels of the edge is a
        # decoder or sensor artefact, never a hit on a monitor that has to be
        # fully inside the view for calibration to pass at all.
        if detect_on.shape[0] > 8 and detect_on.shape[1] > 8:
            b = 3
            detect_on[:b, :] = 0
            detect_on[-b:, :] = 0
            detect_on[:, :b] = 0
            detect_on[:, -b:] = 0
            mask[:b, :] = 0
            mask[-b:, :] = 0
            mask[:, :b] = 0
            mask[:, -b:] = 0

        # Confine the expensive work to the screen, BEFORE doing it.
        #
        # The ROI used to be applied per blob at the very end of `_blobs`,
        # after connected components had labelled the whole frame and the
        # Python loop had run `np.nonzero` over every candidate. On this
        # bench the monitor covers 17% of the camera frame, so 83% of that
        # work could never produce a hit -- and the worst of it is the blown
        # out window beside the monitor, which is exactly where the blobs
        # are. Measured live 2026-10-15: detect 31.76 ms against a camera
        # delivering 60 fps, so the app ran at 21 and a 100 ms pulse spanned
        # two frames against a min_frames of 2. A third of the string was
        # rejected as too short. The camera was never the limit.
        #
        # Nothing is lost: every pixel removed here was going to be thrown
        # away by `_inside_roi` anyway. The off-screen peak is still
        # reported, and measured better than before -- the brightest score
        # anywhere outside the screen, rather than the brightest blob that
        # happened to survive the area and shape filters first.
        roi = self._roi_image(mask.shape[:2])
        self._off_peak_outside = 0
        self._off_count_outside = 0
        if roi is not None:
            outside = cv2.bitwise_not(roi)
            self._off_peak_outside = int(
                cv2.minMaxLoc(detect_on, mask=outside)[1])
            # 'The camera has probably moved' is told from the NUMBER of
            # bright things landing off the screen, not only the brightest,
            # and that diagnosis is the whole reason a shot mapping outside
            # the panel is reported rather than dropped. Removing the
            # pixels before labelling them would have removed the count
            # with them -- the suite caught exactly that.
            #
            # So count them, but only when there is something worth
            # counting: a blob below min_peak was never going to be
            # reported anyway. In a quiet room this branch does not run at
            # all, which is the point -- the common case stays cheap and
            # the diagnostic stays exact for the case it exists to catch.
            if self._off_peak_outside >= p.min_peak:
                off_mask = cv2.bitwise_and(mask, outside)
                n_off, _l, st_off, _c = cv2.connectedComponentsWithStats(
                    off_mask, connectivity=8)
                self._off_count_outside = sum(
                    1 for i in range(1, n_off)
                    if p.min_area <= int(st_off[i, cv2.CC_STAT_AREA])
                    <= p.max_area)
            mask = cv2.bitwise_and(mask, roi)

        self.last_score = detect_on
        self.last_mask = mask
        # Measured on detect_on, not on score, so it is directly comparable
        # with p.threshold -- the number the user is about to be shown it next to.
        self.last_frame_peak = int(detect_on.max()) if detect_on.size else 0

        dets = self._blobs(mask, detect_on, ts, full_w, full_h)

        if p.use_background and self._bg is not None:
            # Update on every frame, including frames holding a detection.
            #
            # Freezing while a laser is visible sounds right -- it stops the
            # model chasing the dot -- but it means a steady bright object in
            # frame is detected forever and the model never learns it. Always
            # updating absorbs a lamp or a reflection within a couple of seconds,
            # and costs almost nothing against a real pulse: at alpha 0.05 three
            # frames move the model by about 14%, against a spot that saturates.
            rate = p.bg_alpha if not dets else p.bg_alpha * 0.5
            # STRAIGHT FROM uint8: accumulateWeighted takes an 8-bit source
            # against a float32 accumulator, so the per-frame conversion was
            # pure waste. 1.28 ms -> 0.09 ms, bit-identical result.
            cv2.accumulateWeighted(score, self._bg, rate)

        self.last_candidates = len(dets)
        self.last_process_ms = (time.perf_counter() - t0) * 1000.0
        return dets

    def _blobs(
        self, mask: np.ndarray, score: np.ndarray, ts: float, full_w: int, full_h: int
    ) -> list[Detection]:
        p = self.params
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        found: list[Detection] = []
        # Blobs `max_area` refused that are still bright: halo-bearing events.
        # (peak, cx, cy, diag), full-frame px.
        oversize: list[tuple[int, float, float, float]] = []
        rejected_roi = 0
        big_rejected: tuple[int, float, float, float] | None = None
        rejects = {"area": 0, "shape": 0, "dim": 0, "roi": 0, "amp_area": 0, "halo": 0}
        rec = self.record_rejects
        rej_rows: list[tuple] = []
        rej_floor = REJECT_PEAK_FRACTION * p.min_peak
        dim_peak = 0
        dim_pos: tuple[float, float] | None = None
        amp_gated: tuple[float, float, float, float, float] | None = None
        off_peak = 0
        inv = 1.0 / p.scale if p.scale < 0.999 else 1.0
        recip = (self._recip if (p.amp_area_per_gain > 0 and self._recip is not None
                                 and self._recip.shape[:2] == score.shape[:2])
                 else None)

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
            w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
            big = area > p.max_area
            if area < p.min_area or big:
                admitted = False
                if big:
                    # Rare (a bloom), so the extra pass over its bbox is cheap.
                    opk = int(score[y0:y0 + h, x0:x0 + w][
                        labels[y0:y0 + h, x0:x0 + w] == i].max())
                    # BRIGHT ENOUGH TO BE A PULSE: not rejected for size (see
                    # `DetectorParams.oversize_min_peak`). It goes on through
                    # the normal path, positioned by its core.
                    admitted = (p.oversize_min_peak > 0
                                and opk >= max(p.min_peak, p.oversize_min_peak))
                    if not admitted and opk >= p.min_peak:
                        ox, oy = float(cents[i][0]) * inv, float(cents[i][1]) * inv
                        if self._inside_roi(ox, oy):
                            if p.halo_ratio > 0:
                                oversize.append((opk, ox, oy,
                                                 float(np.hypot(w, h)) * inv))
                            if big_rejected is None or opk > big_rejected[0]:
                                big_rejected = (opk, ox, oy,
                                                area * inv * inv)
                if not admitted:
                    rejects["area"] += 1
                    # Only an oversize blob that was nearly bright enough: its
                    # peak was computed above anyway. Tiny specks (under
                    # min_area) are never recorded -- they are the numerous,
                    # uninformative kind.
                    if (rec and big and opk >= rej_floor
                            and len(rej_rows) < REJECT_ROWS_PER_FRAME):
                        rej_rows.append((
                            "big", float(cents[i][0]) * inv, float(cents[i][1]) * inv,
                            opk, area * inv * inv, w * inv, h * inv,
                            max(w, h) / max(1.0, float(min(w, h)))))
                    continue

            # Size evidence where the score was amplified. `recip` is the
            # amplification itself (1/gain) at the working frame size, so
            # this reads the LOCAL factor at the blob, not a screen position.
            # Counted under its own reason: it is not "too faint" and must
            # not be reported as such.
            if recip is not None:
                amp = float(recip[min(recip.shape[0] - 1, y0 + h // 2),
                                  min(recip.shape[1] - 1, x0 + w // 2)])
                if area < amplified_min_area(p, amp):
                    rejects["amp_area"] += 1
                    if amp_gated is None or area * inv * inv > amp_gated[0]:
                        amp_gated = (area * inv * inv,
                                     amplified_min_area(p, amp) * inv * inv,
                                     amp, (x0 + w * 0.5) * inv,
                                     (y0 + h * 0.5) * inv)
                    if rec and len(rej_rows) < REJECT_ROWS_PER_FRAME:
                        # The gate throws the blob away before its peak exists;
                        # recording it needs one (cheap, bounded) extra look.
                        pk = int(score[y0:y0 + h, x0:x0 + w][
                            labels[y0:y0 + h, x0:x0 + w] == i].max())
                        rej_rows.append((
                            "amp_area", cents[i][0] * inv, cents[i][1] * inv,
                            pk, area * inv * inv, w * inv, h * inv,
                            max(w, h) / max(1.0, float(min(w, h)))))
                    continue

            elong = max(w, h) / max(1.0, float(min(w, h)))
            if p.max_elongation > 0 and elong > p.max_elongation:
                rejects["shape"] += 1
                if rec and len(rej_rows) < REJECT_ROWS_PER_FRAME:
                    pk = int(score[y0:y0 + h, x0:x0 + w][
                        labels[y0:y0 + h, x0:x0 + w] == i].max())
                    rej_rows.append((
                        "shape", cents[i][0] * inv, cents[i][1] * inv,
                        pk, area * inv * inv, w * inv, h * inv, elong))
                continue

            sel = labels[y0:y0 + h, x0:x0 + w] == i
            patch = score[y0:y0 + h, x0:x0 + w]
            ys, xs = np.nonzero(sel)
            wts = patch[ys, xs].astype(np.float32)
            peak = int(wts.max()) if wts.size else 0
            if peak < p.min_peak:
                rejects["dim"] += 1
                if peak > dim_peak:
                    dim_peak = peak
                    dim_pos = ((x0 + w * 0.5) * inv, (y0 + h * 0.5) * inv)
                if (rec and peak >= rej_floor
                        and len(rej_rows) < REJECT_ROWS_PER_FRAME):
                    rej_rows.append((
                        "dim", (x0 + w * 0.5) * inv, (y0 + h * 0.5) * inv,
                        peak, area * inv * inv, w * inv, h * inv, elong))
                continue
            total = float(wts.sum())
            if total <= 0:
                continue

            # Intensity-weighted centroid. For a saturated flat-topped spot this
            # degenerates to the geometric centre, which is the right answer; for
            # a dim spot it recovers genuine sub-pixel position.
            cx = float((wts * xs).sum() / total) + x0
            cy = float((wts * ys).sum() / total) + y0
            if big and p.oversize_core_frac > 0:
                # A bloom: the halo drags the weighted centroid; the dot is
                # the bright core (see `DetectorParams.oversize_core_frac`).
                core = wts >= p.oversize_core_frac * peak
                cx = float(xs[core].mean()) + x0
                cy = float(ys[core].mean()) + y0
            fx, fy = cx * inv, cy * inv
            if not self._inside_roi(fx, fy):
                rejected_roi += 1
                rejects["roi"] += 1
                off_peak = max(off_peak, peak)
                if rec and len(rej_rows) < REJECT_ROWS_PER_FRAME:
                    rej_rows.append((
                        "roi", fx, fy, peak, area * inv * inv, w * inv, h * inv,
                        elong))
                continue

            found.append(Detection(
                x=fx, y=fy, area=area * (inv * inv), peak=peak,
                frame_index=self._frame_index, timestamp=ts,
                nx=fx / max(1, full_w), ny=fy / max(1, full_h),
                width=w * inv, height=h * inv, elongation=elong,
            ))

        found.sort(key=lambda d: (-d.peak, -d.area))
        kept = self._suppress_neighbours(found, oversize)
        rejects["halo"] = len(found) - len(kept)
        if rec:
            # Folded into a brighter neighbour (min_separation or the halo
            # rule: the detector counts both under "halo"), then anything the
            # max_blobs cut removed. Detections already carry full-frame units.
            kept_ids = {id(d) for d in kept}
            for d in found:
                if id(d) not in kept_ids and len(rej_rows) < REJECT_ROWS_PER_FRAME:
                    rej_rows.append(("halo", d.x, d.y, d.peak, d.area, d.width,
                                     d.height, d.elongation))
            for d in kept[p.max_blobs:]:
                if len(rej_rows) < REJECT_ROWS_PER_FRAME:
                    rej_rows.append(("max_blobs", d.x, d.y, d.peak, d.area,
                                     d.width, d.height, d.elongation))
        self.last_rejected_blobs = rej_rows

        self.last_dim_peak = dim_peak
        self.last_dim_pos = dim_pos
        self.last_big_rejected = big_rejected
        self.last_amp_gate = amp_gated
        # Whichever is brighter, and however many: a blob that reached the
        # ROI test the long way, plus anything outside the screen that the
        # mask removed before it could get there.
        rejected_roi += self._off_count_outside
        rejects["roi"] += self._off_count_outside
        self.last_rejected_roi = rejected_roi
        self.last_rejects = rejects
        self.last_off_screen_peak = max(off_peak, self._off_peak_outside)
        return kept[: p.max_blobs]

    def _suppress_neighbours(
        self, dets: list[Detection],
        oversize: "list[tuple[int, float, float, float]] | None" = None,
    ) -> list[Detection]:
        """One illumination event is one shot: drop blobs that belong to a brighter one.

        `dets` is sorted brightest first. Two rules, both within this frame:

        1. `min_separation`: thresholding a bloomy spot can shatter it into
           fragments that closing does not fully rejoin; closer than this to
           a brighter blob and it is the same spot.
        2. HALO (see `DetectorParams.halo_ratio`): a blob much dimmer than a
           NEARBY brighter event is a lump of that event's scatter halo.
           The events are the surviving detections AND `oversize`, the blobs
           `max_area` already rejected -- the failing case is precisely one
           where the bright event is not in `dets` at all.

        Only an UNSUPPRESSED detection (or an oversize blob) can dominate,
        so a lump never suppresses anything. Sets `last_halo` to the
        brightest thing folded away, for the heard-but-not-seen report.
        There is deliberately NO time-based lockout here.
        """
        p = self.params
        sep = p.min_separation
        self.last_halo = None
        if not dets:
            return dets
        events: list[tuple[int, float, float, float]] = list(oversize or ())
        kept: list[Detection] = []
        for d in dets:                       # already sorted brightest first
            if sep > 0 and not all(d.distance_to(k) > sep for k in kept):
                continue
            if p.halo_ratio > 0 and p.halo_radius_diag > 0:
                folded = False
                for pk, ex, ey, diag in events:
                    dist = float(np.hypot(d.x - ex, d.y - ey))
                    if (d.peak < p.halo_ratio * pk
                            and p.halo_core_frac * diag < dist
                            <= max(sep, p.halo_radius_diag * diag)):
                        folded = True
                        break
                if folded:
                    if self.last_halo is None or d.peak > self.last_halo[0]:
                        self.last_halo = (int(d.peak), float(d.x), float(d.y))
                    continue
            kept.append(d)
            events.append((int(d.peak), float(d.x), float(d.y),
                           float(np.hypot(d.width, d.height))))
        return kept


@dataclass
class _Track:
    track_id: int
    members: list[Detection]
    last_index: int
    # Set once a track has outlasted any plausible pulse. A poisoned track stays
    # alive and keeps claiming its detections -- that is the point. Deleting it
    # instead just lets a fresh track start on the same steady light, which then
    # emits a spurious shot the moment the light finally goes out.
    poisoned: bool = False
    # Index assigned when the track was announced provisionally, if it was.
    announced: int | None = None

    @property
    def last(self) -> Detection:
        return self.members[-1]

    def trim(self, keep: int = 4) -> None:
        """Bound memory on a poisoned track; only recent positions matter."""
        if len(self.members) > keep:
            del self.members[:-keep]


class ShotTracker:
    """Turns per-frame detections into discrete shots, several at a time.

    One track per laser dot. Each frame, detections are matched to open tracks
    by nearest neighbour within max_move_px; anything unmatched starts a new
    track; a track with no detection for max_gap_frames closes and becomes a
    Shot. Two guns fired together therefore produce two tracks and two shots,
    even when their pulses overlap in time.
    """

    def __init__(
        self,
        min_frames: int = 1,
        single_frame_peak: int = 0,
        max_gap_frames: int = 2,
        max_move_px: float = 40.0,
        frame_period_ms: float = 33.3,
        max_tracks: int = 4,
        max_duration_ms: float = 400.0,
        refractory_ms: float = 250.0,
        refractory_px: float = 60.0,
    ) -> None:
        self.min_frames = min_frames
        # A one-frame pulse this bright is taken as real even when
        # min_frames is 2. Zero disables it. See `_confirmed`.
        self.single_frame_peak = int(single_frame_peak)
        # Where single-frame pulses have recently been accepted, so a
        # static object cannot keep producing them. See `_confirmed`.
        self._single_spots: list[tuple[float, float, float]] = []
        self.rejected_single_repeat = 0
        self.max_gap_frames = max_gap_frames
        self.max_move_px = max_move_px
        self.frame_period_ms = frame_period_ms
        self.max_tracks = max_tracks
        # A LaserLyte pulse is ~100 ms. Anything lingering several times longer
        # is a lamp, a reflection or a window -- a steady light source, not a
        # trigger pull. Without this ceiling a single bright object in frame
        # becomes one enormous "shot" the moment it finally disappears, and in
        # the meantime it occupies a track that a real shot might have needed.
        self.max_duration_ms = max_duration_ms
        # A trigger pull that flickers can close one track and immediately open
        # another in the same place. Track-level gap tolerance cannot catch that,
        # because by then the first track is already gone. Measured live: one
        # shot at (275,674) reported three times in 1.5s with pulses of 43, 43
        # and 57 ms -- all far short of the 100 ms a real pull lasts.
        #
        # So: after a shot, ignore further shots at the same spot for a while.
        # 250 ms is comfortably longer than a pulse and far shorter than any
        # plausible split time between two aimed shots.
        self.refractory_ms = refractory_ms
        self.refractory_px = refractory_px
        self._recent: list[tuple[float, float, float]] = []   # x, y, timestamp
        self.merged_fragments = 0
        self._provisional: list[Shot] = []
        self._tracks: list[_Track] = []
        self._count = 0
        self._next_track_id = 0
        self.rejected_runs = 0
        self.rejected_too_long = 0
        self.max_concurrent = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._count = 0
        self._next_track_id = 0
        self.rejected_runs = 0
        self.rejected_too_long = 0
        self.max_concurrent = 0
        self.merged_fragments = 0
        self._recent.clear()
        self._provisional.clear()

    @property
    def count(self) -> int:
        return self._count

    @property
    def open_tracks(self) -> int:
        return len(self._tracks)

    def feed(self, detections: list[Detection], frame_index: int) -> list[Shot]:
        """Call once per frame with every detection in it. Returns shots that just closed."""
        closed: list[Shot] = []

        # 1. Retire tracks that have gone quiet, and abandon ones that have
        #    outstayed any plausible pulse. Done before association so neither a
        #    stale nor a stuck track can steal a detection from a real new shot.
        still_open: list[_Track] = []
        for tr in self._tracks:
            if not tr.poisoned and self._too_long(tr):
                tr.poisoned = True
                self.rejected_too_long += 1
            if tr.poisoned:
                tr.trim()
            if frame_index - tr.last_index > self.max_gap_frames:
                shot = self._close(tr)
                if shot is not None:
                    closed.append(shot)
            else:
                still_open.append(tr)
        self._tracks = still_open

        # 2. Greedy nearest-neighbour association. The numbers here are tiny
        #    (at most a handful of tracks and detections), so an optimal
        #    assignment would be precision without a purpose.
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            for di, det in enumerate(detections):
                dist = det.distance_to(tr.last)
                if dist <= self.max_move_px:
                    pairs.append((dist, ti, di))
        pairs.sort()

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _dist, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            tr = self._tracks[ti]
            tr.members.append(detections[di])
            tr.last_index = frame_index
            used_tracks.add(ti)
            used_dets.add(di)

        # 3. Whatever is left is a new dot.
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            if len(self._tracks) >= self.max_tracks:
                break
            self._tracks.append(_Track(self._next_track_id, [det], frame_index))
            self._next_track_id += 1

        # 4. Announce any track that has just become credible. Waiting for a
        #    track to CLOSE before reporting it costs the pulse length plus the
        #    whole gap tolerance -- about 200 ms at these settings -- and the
        #    delay is plainly visible as late bullet holes. Reporting at
        #    min_frames cuts that to a couple of frames; the final position
        #    arrives later and barely moves.
        for tr in self._tracks:
            if (tr.announced is None and not tr.poisoned
                    and self._confirmed(tr.members)):
                shot = self._make_shot(tr, provisional=True)
                if shot is not None:
                    self._provisional.append(shot)

        self.max_concurrent = max(self.max_concurrent, len(self._tracks))
        return closed

    def take_provisional(self) -> list[Shot]:
        """Shots announced this frame, position not yet final. Drains the queue."""
        out, self._provisional = self._provisional, []
        return out

    def flush(self) -> list[Shot]:
        out = [s for s in (self._close(tr) for tr in self._tracks) if s is not None]
        self._tracks.clear()
        return out

    def _span_ms(self, tr: _Track) -> float:
        if len(tr.members) < 2:
            return self.frame_period_ms
        return (tr.members[-1].timestamp - tr.members[0].timestamp) * 1000.0 + self.frame_period_ms

    def _too_long(self, tr: _Track) -> bool:
        return self.max_duration_ms > 0 and self._span_ms(tr) > self.max_duration_ms

    def _close(self, tr: _Track) -> Shot | None:
        return self._make_shot(tr, provisional=False)

    # How close counts as 'the same place', in camera pixels. The observed
    # static object repeated to 0.07px; a shooter's tightest group is far
    # wider than this once mapped back to the camera, so 6px separates them
    # with room to spare.
    # WHERE THE SINGLE-FRAME BAR COMES FROM.
    #
    # Two independent recorded sessions, hours apart, at different
    # exposures and different noise floors, agree on where the laser
    # population starts and the noise population stops:
    #
    #   session   exposure   noise floor   noise singles   laser singles
    #   13:06        -6           4          10 .. 28        31 .. 255
    #   13:48        -5           7          15 .. 29        31 .. 180
    #
    # The boundary sits at 30 in both, because it is a property of the
    # LASER against this panel and this camera -- not of the room. The
    # noise floor is a property of the room and moved between them.
    #
    # Two anchors were tried and both failed on the data:
    #
    #  - 3x the noise floor derived min_peak. Gave 30 in one session and
    #    39 in the other, and 39 is above where that session s real shots
    #    started: the rule rejected a quarter of the shots it exists to
    #    recover.
    #  - a fraction of the median CONFIRMED shot peak, learned live. It
    #    learns from any pulse that lasted two frames, and at a min_peak
    #    of 13 that includes the phantom population, which dragged the
    #    bar down to 18 and let noise through. The anchor was
    #    contaminated by the very thing it was meant to exclude.
    #
    # So the app sets it to max(30, min_peak + 6): the measured boundary,
    # with a floor that rises if the room is genuinely noisy enough for
    # min_peak to approach it. Measured on both sessions, that keeps
    # 16/17 and 12/12 of the real single-frame shots and admits no noise.
    #
    # It is a number measured on this hardware rather than derived from
    # something else, and it is written down here with its evidence so
    # that a different gun or panel is known to need re-measuring.

    SINGLE_SPOT_PX = 6.0
    # ...and for how long a place stays spent. Long enough that a lamp or a
    # reflection is caught for the whole string, short enough that a target
    # returning to the same spot much later is not punished for it.
    SINGLE_SPOT_WINDOW_S = 120.0

    def _single_bar(self) -> int:
        """How bright a lone frame must be. The configured value, as set."""
        return self.single_frame_peak

    def _confirmed(self, run: list) -> bool:
        """Is this run enough to call a shot?

        Normally it needs `min_frames`. The exception is a single frame so
        bright that nothing but the laser has ever produced it.

        min_frames=2 exists because one frame of noise and one frame of
        laser were indistinguishable -- measured, and true at the time: in
        one session 20% of 'shots' lasted a single frame, fired while
        nobody was shooting, repeating at identical coordinates.

        It is not true any more, and it is expensive. The diode is
        PWM-driven, so a camera sampling it stroboscopically catches a ~100
        ms pulse in a variable number of frames, and sometimes in one.
        Replayed over a recorded session of real shooting (2026-10-15,
        1015 s, 109 laser pulses):

            pulses seen in 1 frame   26   DISCARDED by min_frames=2
            pulses seen in 2 frames  30
            pulses seen in 3 frames  30
            pulses seen in 4+ frames 23

        24% of the shooter's trigger pulls, thrown away -- and not marginal
        ones: the discarded singles measured peak 30..255, median 81, area
        median 560px. That is the 'about one shot in three leaves no hole'
        the shooter reported, and it is not a detection failure at all.

        The two populations no longer overlap, because the exposure and the
        screen mask have both been fixed since min_frames went to 2. Over
        the same recording, grouping EVERY detection into pulses:

            single frames that were noise   77, peak 10..28  (median 13)
            single frames that were laser   17, peak 31..255 (median 107)

        So a bar between them recovers every one and admits nothing:

            peak >= 25 -> 17 real, 4 noise
            peak >= 30 -> 17 real, 0 noise      <- and 30 = 3x min_peak
            peak >= 35 -> 16 real, 0 noise

        Area was tried as a second condition and earns nothing: the noise
        singles reach 1633px, so size does not separate them. Brightness
        does, cleanly, with a gap of three units between the populations.

        The bar is set by the app as a multiple of min_peak, so it rises
        with the rejection threshold instead of being a number tuned to one
        room. Everything else a shot must satisfy still applies: inside the
        screen quad, a plausible size and shape, not a steady light, and
        not inside the refractory window of a shot already reported.
        """
        if len(run) >= self.min_frames:
            return True
        if not self.single_frame_peak or len(run) != 1:
            return False
        d = run[0]
        if int(getattr(d, 'peak', 0)) < self._single_bar():
            return False

        # ...and only ONCE PER PLACE.
        #
        # Brightness alone is not enough, and the first live run with this
        # rule proved it inside a minute:
        #
        #   13:37:55  SHOT 52  cam=(561.12,522.69)  frames=1  peak=45
        #   13:37:58  SHOT 54  cam=(561.13,522.76)  frames=1  peak=48
        #
        # Identical to seven hundredths of a pixel, three seconds apart.
        # That is a static object, and these notes already name the
        # signature: repeated identical coordinates mean a thing in the
        # room, never a trigger pull. Worse, those two spoiled the noise
        # floor five times over and drove it from 4 to 48, which withdrew
        # the sensitivity map and set the rejection bar above every real
        # shot -- the exact 'it worked for a few shots then stopped'
        # failure this project has chased all week.
        #
        # A laser never lands in the same few pixels twice: the shooter
        # cannot hold that still, and the whole point of the drill is that
        # the target moves. Checked against the recorded session, all 17
        # bright single-frame pulses were at positions seen exactly once --
        # so this costs nothing real and removes the whole class.
        #
        # The FIRST appearance is still allowed through: at that moment a
        # static object and a shot are genuinely indistinguishable, and
        # refusing both would throw away a real first shot. One phantom per
        # object per window is the price, against an unbounded stream.
        now = float(getattr(d, 'timestamp', 0.0) or 0.0)
        x = float(getattr(d, 'x', 0.0))
        y = float(getattr(d, 'y', 0.0))
        self._single_spots = [sp for sp in self._single_spots
                              if now - sp[0] < self.SINGLE_SPOT_WINDOW_S]
        for _t, sx, sy in self._single_spots:
            if abs(x - sx) <= self.SINGLE_SPOT_PX and abs(y - sy) <= self.SINGLE_SPOT_PX:
                self.rejected_single_repeat += 1
                return False
        self._single_spots.append((now, x, y))
        return True

    def _make_shot(self, tr: _Track, provisional: bool) -> Shot | None:
        run = tr.members
        if not run:
            return None
        if tr.poisoned:
            return None                   # already counted when it was poisoned
        if not self._confirmed(run):
            if not provisional:
                self.rejected_runs += 1
            return None
        if self._too_long(tr):
            if not provisional:
                self.rejected_too_long += 1
            return None

        # WHERE THE SHOT LANDED IS THE FIRST FRAME, not an average of the pulse.
        #
        # The bullet leaves at trigger break. Everything after that is
        # follow-through: the muzzle drifts while the ~100ms pulse is still
        # lit, and a real round would be long gone. Averaging over the pulse
        # reports a place the shot never was. Measured on a real session: the
        # brightness-weighted average sat a mean of 1.5mm and up to 7.5mm from
        # the first frame, with total muzzle travel reaching 11.5mm -- half an
        # inch of pure artefact, comparable to the whole calibration error.
        #
        # The averaging was there because the first frame might catch the pulse
        # mid-ramp and be dim. Measured, it almost never is: the first frame
        # averages 93% of the pulse's brightest frame, is the brightest outright
        # in half of all shots, and falls below 60% in 1 shot of 30. So take the
        # first frame, and only step forward for that rare dim one.
        best = max(d.peak for d in run)
        lead = run[0]
        if best > 0 and lead.peak < 0.6 * best:
            lead = next((d for d in run if d.peak >= 0.6 * best), run[0])
        x, y = float(lead.x), float(lead.y)
        nx, ny = float(lead.nx), float(lead.ny)

        # Refractory check: is this the tail of a shot already reported? Done
        # after the centroid is known, because "same place" is the whole test.
        now = run[-1].timestamp
        self._recent = [r for r in self._recent
                        if r[2] >= now - self.refractory_ms / 1000.0]
        if tr.announced is None:
            for i, (rx, ry, _rt) in enumerate(self._recent):
                if math.hypot(x - rx, y - ry) <= self.refractory_px:
                    # Refresh the entry rather than just dropping this fragment.
                    # A badly flickering pulse arrives as a chain of them, and if
                    # the window were measured from the original shot it would
                    # expire mid-chain and the tail would register as a second hit.
                    self._recent[i] = (rx, ry, now)
                    # Counted here regardless of which path noticed it: the track
                    # is poisoned on the spot, so this can only happen once for it.
                    self.merged_fragments += 1
                    tr.poisoned = True
                    return None

        span_ms = (run[-1].timestamp - run[0].timestamp) * 1000.0
        if tr.announced is None:
            self._count += 1
            tr.announced = self._count
            self._recent.append((x, y, now))
        index = tr.announced
        return Shot(
            x=x, y=y, nx=nx, ny=ny,
            frames=len(run),
            peak=max(d.peak for d in run),
            area=float(np.mean([d.area for d in run])),
            first_timestamp=run[0].timestamp,
            last_timestamp=run[-1].timestamp,
            # The pulse began somewhere before the first frame that caught it and
            # ended somewhere after the last, so add one frame period of slack.
            duration_ms=span_ms + self.frame_period_ms,
            index=index,
            provisional=provisional,
            track_id=tr.track_id,
            elongation=float(np.mean([d.elongation for d in run])),
            members=list(run),
        )


# Kept so older callers and notebooks do not break; ShotTracker is the real name.
ShotGrouper = ShotTracker
