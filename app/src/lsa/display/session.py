"""Session scoring and timing.

Deliberately separate from both the detector and the renderer, because this is
the part the drills will be built on: a par-time drill, a Bill drill, a
hostage-target scenario and a plain plinking session all want the same numbers
kept the same way, and only differ in what they do with them.

Phase 0 can only fill in part of this. Until the camera is calibrated we know a
shot happened but not what it hit, so `hits` and `misses` stay at zero and
`register_shot` is called with scored=None. Everything time-related -- elapsed,
splits, cadence -- is already meaningful, and split times are the single most
useful number in dry-fire practice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ScoredShot:
    """A shot plus whatever the game layer managed to conclude about it."""

    timestamp: float
    screen_x: float | None = None      # set once calibration exists
    screen_y: float | None = None
    camera_x: float = 0.0
    camera_y: float = 0.0
    target_id: str | None = None
    score: int = 0
    is_hit: bool | None = None         # None = unknown (uncalibrated)
    split_ms: float | None = None      # time since the previous shot
    track_id: int = 0


@dataclass
class SessionStats:
    started: float = field(default_factory=time.perf_counter)
    # WHAT `shots` MEANS (decided 2026-10-20, after the live session read
    # `shots 20, hits 17, misses 0, hit_rate 1.0` for a 17-round magazine plus
    # three dry pulls): ROUNDS FIRED, i.e. live registered shots, and nothing
    # else. The magazine owns that concept, and a rate of 1.0 printed beside a
    # count of 20 invites exactly the wrong reading (that he missed three).
    # So the identity below holds by construction:
    #
    #     shots == hits + misses + unplaced
    #
    # where `unplaced` is a shot the app saw while it had no mapping to place it
    # (outcome unknown, not a miss). A trigger pull with an EMPTY magazine is a
    # dry pull: its own counter, and it touches nothing else here -- not `shots`,
    # not the timing (`first_shot`, `last_shot`, splits), not the history -- so the
    # string time is the time he was actually shooting. The hit rate's
    # denominator is `hits + misses`, which is `shots` when calibrated.
    shots: int = 0
    hits: int = 0
    misses: int = 0
    unplaced: int = 0
    dry_pulls: int = 0
    score: int = 0
    first_shot: float | None = None
    last_shot: float | None = None
    splits_ms: list[float] = field(default_factory=list)
    history: list[ScoredShot] = field(default_factory=list)
    max_history: int = 500

    # Set by whoever owns the session; shown on the HUD when running a drill.
    label: str = ""
    par_time_s: float | None = None

    def reset(self) -> None:
        self.started = time.perf_counter()
        self.shots = 0
        self.hits = 0
        self.misses = 0
        self.unplaced = 0
        self.dry_pulls = 0
        self.score = 0
        self.first_shot = None
        self.last_shot = None
        self.splits_ms.clear()
        self.history.clear()

    def register_dry(self) -> None:
        """A trigger pull with an empty magazine. Counted apart from the shots."""
        self.dry_pulls += 1

    def consistent(self) -> bool:
        """The identity every displayed or served figure must satisfy."""
        return self.shots == self.hits + self.misses + self.unplaced

    def register_shot(
        self,
        camera_x: float,
        camera_y: float,
        screen_x: float | None = None,
        screen_y: float | None = None,
        scored: bool | None = None,
        score: int = 0,
        target_id: str | None = None,
        track_id: int = 0,
        timestamp: float | None = None,
    ) -> ScoredShot:
        now = time.perf_counter() if timestamp is None else timestamp
        split = None
        if self.last_shot is not None:
            split = (now - self.last_shot) * 1000.0
            self.splits_ms.append(split)

        shot = ScoredShot(
            timestamp=now, screen_x=screen_x, screen_y=screen_y,
            camera_x=camera_x, camera_y=camera_y, target_id=target_id,
            score=score, is_hit=scored, split_ms=split, track_id=track_id,
        )

        self.shots += 1
        if scored is True:
            self.hits += 1
            self.score += score
        elif scored is False:
            self.misses += 1
        else:
            self.unplaced += 1
        if self.first_shot is None:
            self.first_shot = now
        self.last_shot = now

        self.history.append(shot)
        if len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]
        return shot

    # -- derived numbers ------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started

    @property
    def string_time_s(self) -> float:
        """First shot to last shot: the number that actually matters in a drill."""
        if self.first_shot is None or self.last_shot is None:
            return 0.0
        return self.last_shot - self.first_shot

    @property
    def scored_shots(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        return (self.hits / self.scored_shots) if self.scored_shots else None

    @property
    def last_split_ms(self) -> float | None:
        return self.splits_ms[-1] if self.splits_ms else None

    @property
    def best_split_ms(self) -> float | None:
        return min(self.splits_ms) if self.splits_ms else None

    @property
    def mean_split_ms(self) -> float | None:
        return sum(self.splits_ms) / len(self.splits_ms) if self.splits_ms else None

    @property
    def shots_per_minute(self) -> float:
        t = self.string_time_s
        return (self.shots - 1) / t * 60.0 if t > 0 and self.shots > 1 else 0.0

    def clock_string(self) -> str:
        t = self.elapsed_s
        return f"{int(t) // 60:02d}:{int(t) % 60:02d}.{int((t % 1) * 10)}"
