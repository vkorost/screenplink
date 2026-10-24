"""Reintroduce faults into COPIES of the tree and watch the figure-drill checks go red.\n\n`tests/mutate_director.py` does this for the Director. This does it for the\nfigure drills (modes 7, 8, 9) and the artwork behind them. A check that has\nonly ever been seen green proves nothing (RULES C1), and a check shown to fail\nONCE, from a script nobody keeps, enforces nothing afterwards -- which is how\nsix rules went missing on this project. So these run with the suite.\n\nEach mutation copies `src`, `tests` and `tools` to `tests/_out/mutate/<n>/`\n(never the real tree, never his files), changes ONE thing, runs the one test\nfile that should notice, and requires at least one `[FAIL]` line. They run in\nparallel, so the whole thing costs about the slowest single suite.\n\n    python tests/mutate_figures.py            all of them\n    python tests/mutate_figures.py 3 7        just these (by number)\n"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_env import copy_scratch_tree, scratch_env      # noqa: E402  (one table of asset overrides)
import mutation_anchors as anchors  # noqa: E402
import mutation_verdict as verdict  # noqa: E402  (the ONE judgement of a result)
import worker_budget as budget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "_out" / "mutate"
PY = sys.executable

# (what is broken, file, old text or None, new text, test file, text a failing line must contain)
M = [
    ("the loader no longer forces true grey (returns 3-channel BGR)",
     "src/lsa/artwork/load.py", "return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)",
     "return bgr", "test_artwork.py", "one uint8 channel"),
    ("the lit-area budget for one figure is 3% instead of 1.5%",
     "src/lsa/modes/figure.py", "SINGLE_BUDGET = 0.015", "SINGLE_BUDGET = 0.03",
     "test_figures.py", "<= 1.5%), red excess"),
    ("a figure's ring polygons are displaced by 0.05 of its width",
     "resources/targets/thug01.zones.json", None, None, "test_artwork.py", "lies on the ink"),
    ("pick_figure always returns the first figure",
     "src/lsa/artwork/catalog.py", "    return pool[rng.randrange(len(pool))]",
     "    return all_names[0]", "test_artwork.py", "pick_figure"),
    ("pick_figure no longer excludes the figure just shown",
     "src/lsa/artwork/catalog.py", "pool = [n for n in all_names if n != previous]",
     "pool = list(all_names)", "test_artwork.py", "never the same figure twice"),
    ("a point-in-polygon test ignores the half-stroke margin",
     "src/lsa/artwork/zones.py", "return d >= -margin", "return d >= 0.0",
     "test_artwork.py", "within half a stroke"),
    ("a figure that cannot be judged is drawn after all (names() ignores exclusions)",
     "src/lsa/artwork/catalog.py", "    return [n for n in all_names(folder) if n not in bad]",
     "    return all_names(folder)", "test_artwork.py", "all-names lists every image"),
    ("an excluded figure can be constructed",
     "src/lsa/modes/figure.py", "        if why:\n            raise ValueError", "        if False:\n            raise ValueError",
     "test_artwork.py", "refused"),
    ("the advancing target asks for a repaint while holding still",
     "src/lsa/modes/advancing.py", "        return changed\n\n    def _live_at",
     "        return changed or any(a.phase == 'live' for a in self._actors)\n\n    def _live_at",
     "test_figures.py", "holding still"),
    ("the advancing target is scored where the figure IS, not where it WAS",
     "src/lsa/modes/advancing.py", "        t = ctx.now - self.shot_age(ev, ctx)\n        if not self._live_at(t) and any(",
     "        t = ctx.now\n        if not self._live_at(t) and any(", "test_figures.py", "WAS 80 ms ago"),
    ("the advancing target always walks in from the same side",
     "src/lsa/modes/advancing.py", "side = self._rng.choice((-1, 0, 1))", "side = 1",
     "test_figures.py", "drift side is not constant"),
    ("the advancing target's budget for two figures is the single-figure one",
     "src/lsa/modes/advancing.py",
     "return SINGLE_BUDGET if self.COUNT <= 1 else COMBINED_BUDGET / self.COUNT",
     "return SINGLE_BUDGET", "test_figures.py", "COUNT = 2"),
    ("the pop-up's wait is a fixed delay",
     "src/lsa/modes/popup.py",
     "self.next_at = ctx.now + self._rng.uniform(self.MIN_WAIT, self.MAX_WAIT)",
     "self.next_at = ctx.now + self.MIN_WAIT", "test_figures.py", "wait before each figure is random"),
    ("the pop-up reuses the previous centre (no 35% move)",
     "src/lsa/modes/popup.py",
     "self._centre = catalog.pick_position(self._rng, field, w / 2.0, h / 2.0,\n"
     "                                             self._prev_centre,\n"
     "                                             avoid=self.menu_keepout(ctx))",
     "self._centre = self._prev_centre or (field.center().x(), field.center().y())",
     "test_figures.py", "35% of the diagonal"),
    ("a hit in the image target draws nothing (no flash, no number, no hole)",
     "src/lsa/modes/image_target.py",
     "        self._marks.paint(p, ctx, fig, self._centre, self._height, self.FLASH_S)",
     "        pass", "test_figures.py", "image: a hit changes what is on screen"),
    ("a figure forgets where it was hit once the flash fades (no hole is kept)",
     "src/lsa/modes/figure.py", "        self.holes = (self.holes + [(u, v, good)])[-self.cap:]",
     "        pass", "test_figures.py", "STILL marked"),
    ("the hit feedback is drawn in the old warm accent",
     "src/lsa/modes/figure.py", "c = QColor(GREY_HIT if good else GREY_MISS)",
     "c = QColor('#d8c070')", "test_modes.py", "grey"),
    ("a key-legend entry is made long enough to run off the 4K screen",
     "src/lsa/display/layers.py",
     '        ((" ", "(pressing a mode\'s key again = a new round of that mode)"),),',
     '        ((" ", "(pressing a mode\'s key again starts a new round of that mode, and every key the app uses while shooting is listed below, including calibration and measurement)"),),',
     "test_ui_smoke.py", "every string any layer or mode draws"),
    ("a shot on the figure but in no ring draws nothing (the old silence)",
     "src/lsa/modes/image_target.py",
     "                self._marks.add(fig, ev.x, ev.y, self._centre, self._height,\n"
     "                                ev.t, NO_RING, False)\n",
     "                pass\n", "test_figures.py", "...and DRAWS something"),
    ("a shot into the blank between walks is counted as a miss at a figure",
     "src/lsa/modes/advancing.py", "            self.gap_shots += 1\n",
     "            self.gap_shots += 1\n            self.shots += 1\n", "test_figures.py", "blank between walks"),
    ("the on-figure miss is drawn as bright as a scored hit",
     "src/lsa/modes/figure.py", "c = QColor(GREY_HIT if good else GREY_MISS)",
     "c = QColor(GREY_HIT)", "test_figures.py", "DIMMER"),
    ("a mark does not ride the figure: it is placed as a fraction of the SCREEN, not of the figure",
     "src/lsa/modes/figure.py",
     "        left, top, w, h = fig.box(centre, height_px)\n        r = max(ctx.height * 0.002",
     "        left, top, w, h = (0.0, 0.0, 3840.0, 2160.0)\n        r = max(ctx.height * 0.002",
     "test_figures.py", "ALL the pixels the hit adds"),
    ("a hit on the image figure ALSO leaves a screen hole (it is not marked as absorbed)",
     "src/lsa/modes/image_target.py",
     "        return ShotOutcome(hit=True, points=points, absorbed=True,",
     "        return ShotOutcome(hit=True, points=points, absorbed=False,",
     "test_figures.py", "image: a hit on the figure is absorbed"),
    ("a hit on the advancing figure ALSO leaves a screen hole (it is not marked as absorbed)",
     "src/lsa/modes/advancing.py",
     "            return ShotOutcome(hit=True, points=points, absorbed=True,",
     "            return ShotOutcome(hit=True, points=points, absorbed=False,",
     "test_figures.py", "scored hit on the walking figure"),
    ("a hit on the pop-up figure ALSO leaves a screen hole (it is not marked as absorbed)",
     "src/lsa/modes/popup.py",
     "        return ShotOutcome(hit=True, points=points, absorbed=True,",
     "        return ShotOutcome(hit=True, points=points, absorbed=False,",
     "test_figures.py", "popup: a hit on the figure is absorbed"),
    ("the app draws a screen hole for an absorbed shot (it ignores the mode's 'absorbed')",
     "src/lsa/tools/camera_lab.py",
     "                if not absorbed:\n                    _holes_before = holes.added\n                    holes.add(sx, sy, shot.index, points)",
     "                if True:\n                    _holes_before = holes.added\n                    holes.add(sx, sy, shot.index, points)",
     "test_ui_smoke.py", "ABSORBS"),
    ("a MISS stops leaving its screen hole while a mode is playing",
     "src/lsa/tools/camera_lab.py",
     "                if not absorbed:\n                    _holes_before = holes.added\n                    holes.add(sx, sy, shot.index, points)",
     "                if not absorbed and scored is None:\n                    _holes_before = holes.added\n                    holes.add(sx, sy, shot.index, points)",
     "test_ui_smoke.py", "still leaves its ordinary hole"),
    ("a shot that misses the image figure is marked absorbed (so its screen hole is withheld)",
     "src/lsa/modes/image_target.py",
     "            return ShotOutcome.miss()\n        self.hits += 1",
     "            return ShotOutcome(hit=False, absorbed=True)\n        self.hits += 1",
     "test_figures.py", "image: a hit on the figure is absorbed"),
    ("the advancing figure keeps its holes when it leaves (they outlive it)",
     "src/lsa/modes/advancing.py",
     "        a.marks.clear()                    # its holes leave with it",
     "        pass", "test_figures.py", "advancing: when the figure leaves"),
    ("the pop-up figure keeps its holes when it leaves",
     "src/lsa/modes/popup.py",
     "            self._marks.clear()            # the holes go with the figure",
     "            pass", "test_figures.py", "pop-up: after the figure leaves"),
    ("a new image round keeps the previous figure's holes",
     "src/lsa/modes/image_target.py",
     "        self._marks = Marks()\n        if not self._names:\n            self._fig = None",
     "        if not self._names:\n            self._fig = None", "test_figures.py", "image: a new round"),
    ("a figure holding still with holes on it asks for a repaint every tick",
     "src/lsa/modes/figure.py", "        return had\n", "        return True\n",
     "test_figures.py", "holding still with 5 holes"),
    ("a figure carries every hole ever fired at it (no cap)",
     "src/lsa/modes/figure.py", "        self.holes = (self.holes + [(u, v, good)])[-self.cap:]",
     "        self.holes = self.holes + [(u, v, good)]",
     "test_figures.py", "a figure carries at most"),
    ("the pop-up cannot be aimed at by the development walk-through",
     "src/lsa/modes/popup.py", '        if self.state != "up" or self._fig is None:\n            return None\n        inner = self._fig.innermost(self._centre, self._height)\n        return (inner[1], inner[2])\n\n    def nearest',
     '        return None\n\n    def nearest', "test_ui_smoke.py", "walk-through finds something"),
    ("the camera pane draws its live image whatever the state (the 08:50 feedback loop)",
     "src/lsa/display/layers.py",
     "        if not self.draws_camera_image():\n            return                  # see the class comment: never while shooting",
     "        if False:\n            return",
     "test_ui_smoke.py", "draws nothing at all"),
    ("a held key (auto-repeat) is treated as a series of presses",
     "src/lsa/display/window.py", "if ev.isAutoRepeat() and key not in self.REPEATABLE_KEYS:",
     "if False:", "test_ui_smoke.py", "HELD key"),
    # -- the running total of 30, the flash and counter that show it, the 0.8 holes ----
    ("the drop total is 21 (the number that was shown to measure nothing)",
     "src/lsa/modes/figure.py", "DROP_TOTAL = 30", "DROP_TOTAL = 21",
     "test_figures.py", "the drop total is 30"),
    ("the figure drops only ABOVE 30, so three 10s (exactly 30) leave it standing",
     "src/lsa/modes/figure.py", "return self.total >= DROP_TOTAL", "return self.total > DROP_TOTAL",
     "test_figures.py", "three centre hits"),
    ("the image target never drops its figure at the total",
     "src/lsa/modes/image_target.py",
     "        if self._marks.done:\n            # The figure drops NOW.",
     "        if False and self._marks.done:\n            # The figure drops NOW.", "test_figures.py", "image: three centre hits"),
    ("the pop-up keeps the figure up after 30 (no early end)",
     "src/lsa/modes/popup.py",
     "        if self._marks.done:\n            # Reached the drop total: the exposure ends NOW.",
     "        if False and self._marks.done:\n            # Reached the drop total: the exposure ends NOW.", "test_figures.py", "reaching 30 ends the exposure"),
    ("the advancing figure never leaves at the total",
     "src/lsa/modes/advancing.py", "                fig = a.fig\n                record_mark()   # the flash that says why the figure dropped IS a mark\n                a.released_at = ev.t\n                self._arm(a, ctx)\n                a.fall = Fall.start",
     "                fig = a.fig\n                record_mark()   # the flash that says why the figure dropped IS a mark\n                a.released_at = ev.t\n                a.fall = Fall.start", "test_figures.py", "advancing: three centre hits"),
    ("a shot in no ring advances the image figure's total",
     "src/lsa/modes/image_target.py", "                self.no_ring += 1\n",
     "                self.no_ring += 1\n                self._marks.tally(7)\n",
     "test_figures.py", "image: six shots on the figure but in no ring"),
    ("a shot in no ring advances the pop-up figure's total",
     "src/lsa/modes/popup.py", "                self.no_ring += 1\n",
     "                self.no_ring += 1\n                self._marks.tally(7)\n",
     "test_figures.py", "popup: six shots on the figure but in no ring"),
    ("a shot in no ring advances the walking figure's total",
     "src/lsa/modes/advancing.py", "                self.no_ring += 1\n",
     "                self.no_ring += 1\n                a.marks.tally(7)\n",
     "test_figures.py", "advancing: six shots on the figure but in no ring"),
    ("the image flash shows the hit's ring value, not the running total",
     "src/lsa/modes/image_target.py",
     "            self._marks.add(fig, ev.x, ev.y, self._centre, self._height,\n                            ev.t, str(total), True)",
     "            self._marks.add(fig, ev.x, ev.y, self._centre, self._height,\n                            ev.t, str(points), True)",
     "test_figures.py", "image: a 7, then a 9, then an 8 flash 7, 16, 24"),
    ("the pop-up flash shows the hit's ring value, not the running total",
     "src/lsa/modes/popup.py", "        label = str(total)\n", "        label = str(points)\n",
     "test_figures.py", "popup: a 7, then a 9, then an 8 flash 7, 16, 24"),
    ("the walking figure's flash shows the hit's ring value, not the running total",
     "src/lsa/modes/advancing.py",
     "a.marks.add(a.fig, ev.x, ev.y, centre, h, ev.t, str(total), True)",
     "a.marks.add(a.fig, ev.x, ev.y, centre, h, ev.t, str(points), True)",
     "test_figures.py", "advancing: a 7, then a 9, then an 8 flash 7, 16, 24"),
    ("the dropping hit leaves no flash at the spot (the figure vanishes unexplained)",
     "src/lsa/modes/image_target.py",
     "            self.flashes = (self.flashes + [(ev.x, ev.y, ev.t, str(total), True)])[-MAX_MARKS:]\n            record_mark()   # the flash that says why the figure dropped IS a mark\n            self._fall = Fall.start(",
     "            record_mark()   # the flash that says why the figure dropped IS a mark\n            self._fall = Fall.start(", "test_figures.py", "image: a single 10 on an already-23"),
    ("the walking figure's total survives into its next figure (clear() forgets to reset it)",
     "src/lsa/modes/figure.py", "        self.flashes = []\n        self.total = 0\n\n    def add",
     "        self.flashes = []\n\n    def add", "test_figures.py", "advancing: the next figure starts from zero"),
    ("the counter is never drawn",
     "src/lsa/modes/figure.py", "            self._paint_counter(p, ctx, fig, centre, height_px)",
     "            pass", "test_figures.py", "the counter is drawn"),
    ("the counter is sized in screen pixels, not from the figure",
     "src/lsa/modes/figure.py",
     "size = max(px(ctx, COUNTER_MIN_1080), round(COUNTER_H_FRAC * h))", "size = 40",
     "test_figures.py", "nothing is sized in screen pixels"),
    ("the counter does not ride the figure: it is fixed on the screen",
     "src/lsa/modes/figure.py", "        x = left + w + gap\n", "        x = ctx.width * 0.8\n",
     "test_figures.py", "keeps the same place relative"),
    ("the counter is drawn in the old warm accent (red excess 24)",
     "src/lsa/modes/figure.py",
     "        p.setPen(QPen(QColor(GREY_HIT)))\n        p.drawText(int(x), int(y), self.counter_text())",
     "        p.setPen(QPen(QColor('#d8c070')))\n        p.drawText(int(x), int(y), self.counter_text())",
     "test_figures.py", "red excess exactly 0"),
    ("a figure with the counter up asks for a repaint on every tick",
     "src/lsa/modes/figure.py", "        return had\n", "        return had or self.total > 0\n",
     "test_figures.py", "with '24 / 30' showing"),
    ("a shot still arriving for a pop-up figure that already dropped scores on it",
     "src/lsa/modes/popup.py", "        if self._spent:\n", "        if False:\n",
     "test_figures.py", "still arriving"),
    ("the bullet hole is back at its old size",
     "src/lsa/display/layers.py", "    SIZE_SCALE = 0.8\n", "    SIZE_SCALE = 1.0\n",
     "test_figures.py", "hole radius is 0.8"),
]


SUITE_TIMEOUT_S = 900


def judge(rc: int, text: str, frag: str) -> tuple[str, str]:
    """('caught' | 'MISSED' | 'BROKEN', why). tests/mutation_verdict.py is the ONE
    judgement: caught only when the entry's OWN named check fails. The `(named or
    fails)` fallback this runner carried counted a crashed section as a catch."""
    return verdict.classify(rc, text, frag)


def _scratch_tree(d: Path) -> None:
    # The ONE copy (tests/asset_env.py). targets/ is always copied now: test_artwork
    # and test_introspect fingerprint it relative to their own location, so a scratch
    # tree without it fails them as NOT FOUND, and the data mutation (a displaced
    # ring polygon) lands on this copy and never on his artwork.
    copy_scratch_tree(ROOT, d)


def _run_suite(d: Path, test: str) -> tuple[int, str]:
    env = scratch_env(d)        # every override points at the scratch COPY, not at his folders
    try:
        r = subprocess.run([PY, str(d / "tests" / test)], cwd=str(d), env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=SUITE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return -9, f"TIMED OUT after {SUITE_TIMEOUT_S}s"
    return r.returncode, r.stdout + "\n" + r.stderr


def _baseline(pick: list) -> list[str]:
    """Every suite ANY picked entry names, unmutated (2026-10-23: none was run)."""
    d = OUT / "baseline"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _scratch_tree(d)
    try:
        return verdict.baseline({M[i][4] for i in pick}, lambda s: _run_suite(d, s))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run(i: int, m: tuple) -> tuple:
    """(name, label, why), label one of caught / MISSED / BROKEN / UNPLACED."""
    name, rel, old, new, test, frag = m
    d = OUT / str(i)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _scratch_tree(d)
    p = d / rel
    raw = p.read_bytes()
    if old is None:
        j = json.loads(raw)
        for z in j["zones"]:
            z["pts"] = [[x + 0.05, y] for x, y in z["pts"]]
        out = json.dumps(j).encode()
    else:
        text = raw.decode()
        mutated = anchors.apply(text, old, new)
        if mutated is None:
            shutil.rmtree(d, ignore_errors=True)
            return name, "UNPLACED", (f"MUTATION DID NOT APPLY: {old[:50]!r}: "
                                      + anchors.not_found_reason(text, old))
        out = mutated.encode()
    p.write_bytes(out)
    rc, text = _run_suite(d, test)
    shutil.rmtree(d, ignore_errors=True)
    label, why = judge(rc, text, frag)
    return name, label, why


def _main() -> int:
    pick = [int(a) for a in sys.argv[1:]] or list(range(len(M)))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    red = _baseline(pick)
    if red:
        print("BASELINE IS RED: these suites fail with NO fault applied, so "
              "'caught' would mean nothing. Fix them first.")
        for ln in red:
            print("  " + ln)
        shutil.rmtree(OUT, ignore_errors=True)
        return 2
    workers = budget.workers(18)
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda i: (i, run(i, M[i])), pick))
    bad = 0
    for i, (name, label, why) in res:
        bad += label != "caught"
        print(f"  {label:<8} {i:2d} {name}")
        print(f"             -> {why[:220]}")
    shutil.rmtree(OUT, ignore_errors=True)
    broken = sum(1 for _i, r in res if r[1] == "BROKEN")
    print(f"{len(res) - bad} of {len(res)} figure-drill faults caught "
          f"in {time.perf_counter() - t0:.0f}s"
          + (f" ({broken} BROKEN: the suite crashed rather than failing the named check)"
             if broken else "")
          + ("" if not bad else " -- A CHECK THAT CANNOT FAIL IS NOT A CHECK"))
    return 1 if bad else 0



def main() -> int:
    """One mutation run at a time, and it REFUSES rather than queueing.

    See `tests/worker_budget.py` for the two incidents. The slot is shared
    by every `mutate_*` script, because the 2026-10-17 one was two
    DIFFERENT scripts, each politely capped on its own.
    """
    try:
        with budget.exclusive():
            return _main()
    except budget.Busy as why:
        print(f"REFUSED: {why}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
