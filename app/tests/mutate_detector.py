"""Reintroduce faults into COPIES of the tree and watch the area-gate checks go red.

`tests/mutate_director.py` does this for the Director and `tests/mutate_figures.py`
for the figure drills. This does it for the one piece of the DETECTOR that
asks for size evidence where the sensitivity map has amplified the score
(`amp_area_per_gain`, `amplified_min_area`). The detector is the most dangerous
area on this project -- one changed constant once cost half a string of live
shooting -- so a check on it that has only ever been seen green is worth
nothing (RULES C1, E3).

Each mutation copies `src`, `tests` and `tools` to `tests/_out/mutate_det/<n>/`
(never the real tree, never his files), changes ONE thing, runs the test file
that should notice, and requires a `[FAIL]` line containing the expected text
(tests/mutation_verdict.py: a crash is BROKEN, never caught; a red baseline over
every suite named refuses the run).

    python tests/mutate_detector.py            all of them
    python tests/mutate_detector.py 1 3        just these (by number)
"""

from __future__ import annotations

import concurrent.futures as cf
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
OUT = ROOT / "tests" / "_out" / "mutate_det"
PY = sys.executable
LASER = "src/lsa/detect/laser.py"
CAMLAB = "src/lsa/tools/camera_lab.py"

# (what is broken, file, old text, new text, test file, text a failing line must contain)
M = [
    ("the gate is never applied (every small amplified blob passes, as before)",
     LASER, "                if area < amplified_min_area(p, amp):",
     "                if False:",
     "test_detector.py", "NOT a dot"),
    ("the gate is read from a fixed pixel, not from where the blob is (position, not amplification)",
     LASER,
     "                amp = float(recip[min(recip.shape[0] - 1, y0 + h // 2),\n"
     "                                  min(recip.shape[1] - 1, x0 + w // 2)])",
     "                amp = float(recip[0, 0])",
     "test_detector.py", "follows the map"),
    ("the scale conversion is dropped (full-frame px compared with processing-scale px)",
     LASER, "per * (float(amplification) - 1.0) * s * s)",
     "per * (float(amplification) - 1.0))",
     "test_detector.py", "UNITS"),
    ("the gate asks for area at N = 1 too (behaviour changes where nothing is amplified)",
     LASER, "per * (float(amplification) - 1.0) * s * s)",
     "per * float(amplification) * s * s)",
     "test_detector.py", "N = 1"),
    ("a gated blob is reported as 'too faint' (the diagnostic would blame the wrong thing)",
     LASER, '                    rejects["amp_area"] += 1',
     '                    rejects["dim"] += 1',
     "test_detector.py", "not as 'dim'"),
    ("the ask no longer grows with amplification (a flat 16 px wherever the map acts)",
     LASER, "per * (float(amplification) - 1.0) * s * s)",
     "per * s * s)",
     "test_detector.py", "N = 2 asks 16"),
    # -- the exposure probe's near-side constraint (2026-10-17) ----------------
    ("the near side is never judged saturated (the old weak-end-only choice)",
     "src/lsa/exposure.py",
     "    return m is not None and m > STRONG_MEDIAN_LIMIT",
     "    return False",
     "test_exposure.py", "-5 is chosen"),
    ("the limit is loosened to the clip rate's 56% bin, not its knee",
     "src/lsa/exposure.py", "SAT_CHROMA = 90.0", "SAT_CHROMA = 130.0",
     "test_exposure.py", "derived"),
    ("the probe never draws the dot at the strong end",
     "src/lsa/routines.py",
     "            if strong_spot is not None else None)\n    synth.show_at((rect.width() * spot[0]",
     "            if False else None)\n    synth.show_at((rect.width() * spot[0]",
     "test_routines.py", "BOTH ends"),
    ("the spread warning is never shown on the target screen",
     "src/lsa/routines.py", '        lab._alert("ONE EXPOSURE CANNOT COVER THIS PANEL", text,',
     '        (lambda *a, **k: None)("ONE EXPOSURE CANNOT COVER THIS PANEL", text,',
     "test_routines.py", "target screen is told"),
    ("the warning is shown BEFORE the noise floor (a panel that becomes the floor)",
     "src/lsa/routines.py", "        spread_warning = choice.warning",
     "        spread_warning = choice.warning\n        if spread_warning:\n            yield Call(lambda: _say_spread(lab, spread_warning))",
     "test_routines.py", "AFTER the noise floor"),
    # -- the size gate must say WHY it refused a blob (2026-10-17) -------------
    ("a gated blob is not remembered (the report says 'saw nothing' again)",
     LASER, "                    if amp_gated is None or area * inv * inv > amp_gated[0]:",
     "                    if False:",
     "test_detector.py", "gated blob is kept"),
    ("the evidence window drops the gated blob",
     "src/lsa/camera/evidence.py", '            "amp": amp,', '            "amp": None,',
     "test_feedback.py", "evidence window carries it"),
    ("the report has no branch for a gated blob (it falls through to 'nothing')",
     "src/lsa/shotlog.py", '    if summary.get("amp"):\n        return MISSED_AMP_AREA',
     '    if False:\n        return MISSED_AMP_AREA',
     "test_feedback.py", "own branch"),
    ("Stats never counts a gated blob",
     "src/lsa/camera/capture.py", '            s.rejected_amp_area += rj.get("amp_area", 0)',
     "            pass",
     "test_feedback.py", "counts a gated blob"),
    # -- the sensitivity learner must not learn from what is not a laser ---------
    ("the learner has no laser/noise boundary (the walking figure's edges teach the map)",
     "src/lsa/tools/camera_lab.py",
     '        if shot.peak < self.SINGLE_FRAME_PEAK:\n            if moving:\n                return "faint"',
     '        if False:\n            if moving:\n                return "faint"',
     "test_feedback.py", "ignores the 9"),
    ("the learner learns while a mode is animating",
     "src/lsa/tools/camera_lab.py",
     "        moving = mode is not None and mode.moving_at(shot.first_timestamp)",
     "        moving = False",
     "test_feedback.py", "NOTHING is learned"),
    # -- a faint shot may teach ONLY on a trigger click's word (2026-10-23) ------
    ("corroboration needs no CALIBRATED microphone (a guessed threshold vouches for shots)",
     "src/lsa/tools/camera_lab.py", "lis.clock_is_perf_counter and lis.calibrated",
     "lis.clock_is_perf_counter and True",
     "test_feedback.py", "UNCALIBRATED"),
    ("corroboration ignores which CLOCK the click was heard on (RULES B4)",
     "src/lsa/tools/camera_lab.py", "(lis.running and lis.clock_is_perf_counter and",
     "(lis.running and True and",
     "test_feedback.py", "DIFFERENT clock"),
    ("the corroboration window is symmetric (forward = back), which admits the app's own bang",
     "src/lsa/tools/camera_lab.py", "                        self.SENS_CLICK_FORWARD) is not None):",
     "                        self.corroborator.window) is not None):",
     "test_feedback.py", "AFTER the pulse"),
    ("a click is not marked claimed, so one click corroborates every detection near it",
     "src/lsa/audio/listener.py", "            if best is not None:\n                best.matched = True\n",
     "            if best is not None:\n                pass\n",
     "test_feedback.py", "exactly ONE of two"),
    ("corroboration clears 'animating' (a click near a walking figure's edge teaches the map)",
     "src/lsa/tools/camera_lab.py",
     '            if moving:\n                return "faint"            # no click is claimed',
     '            if False:\n                return "faint"            # no click is claimed',
     "test_feedback.py", "MOVING mode is refused"),
    ("the learner never consults the refusal",
     "src/lsa/tools/camera_lab.py", "        why = self._sens_refusal(shot)\n",
     "        why = None\n",
     "test_feedback.py", "ignores the 9"),
    ("the learner ignores silently (no log line saying what it ignored)",
     "src/lsa/tools/camera_lab.py",
     '            if k == 1 or k % 10 == 0:\n                _h = self.layers.get("modes")',
     '            if False:\n                _h = self.layers.get("modes")',
     "test_feedback.py", "LOG SAYS WHAT IT IGNORED"),
    # -- the shot grid's gain map takes the learner's three steps (2026-10-23) ---
    ("every gain map is built over the whole camera frame again (it amplifies the room)",
     "src/lsa/calibrate/mapping.py",
     "    return sensitivity_map(samples, frame_size, region=region)",
     "    return sensitivity_map(samples, frame_size, region=None)",
     "test_feedback.py", "OUTSIDE the screen is not amplified"),
    ("the grid's measured falloff is not persisted (it dies with the session)",
     "src/lsa/tools/camera_lab.py",
     "        self._save_sensitivity(usable)\n",
     "        pass\n",
     "test_feedback.py", "reaches the file"),
    ("the grid's map is applied but never judged (it stays on however much worse the bar gets)",
     "src/lsa/tools/camera_lab.py",
     # RE-QUOTED 2026-10-23: `_apply_sensitivity_map` takes a `why` now (the
     # brightness grid names itself), so the line reads `why or "..."`.
     "        self._judge_map_against_a_fresh_floor(\n"
     "            gain, why or \"this map was measured by the shot grid\")",
     "        self._set_gain_map(gain, \"the sensitivity map\")\n"
     "        self._calibrating = True\n"
     "        QTimer.singleShot(400, self._start_noise_floor)",
     "test_feedback.py", "handed to the judgement"),
    # -- tools/replay.py cannot silently prove nothing ---------------------------
    # build_gain has TWO raises (too few rows; sensitivity_map said no), and
    # either alone is enough for the 3-row case, so the mutation replaces the
    # whole span with the old behaviour (return whatever came back, even None).
    ("build_gain returns None again instead of raising when there is no map",
     "tools/replay.py",
     "    if len(rows) < 4:\n"
     "        raise NoGainMap(\n"
     "            f\"{calibration} holds {len(rows)} sensitivity rows (4 are needed), so \"\n"
     "            f\"there is no gain map and the size gate has nothing to act on.\")\n"
     "    samples = [Sample(screen=(0.0, 0.0),\n"
     "                      camera=(float(r[\"camera\"][0]), float(r[\"camera\"][1])),\n"
     "                      peak=float(r[\"peak\"])) for r in rows]\n"
     # RE-QUOTED 2026-10-23: replay now goes through `confined_sensitivity_map`
     # like every live path, so it can no longer build a whole-frame map the
     # detector would refuse. The anchor has to follow the line it quotes.
     "    corners = mapping.screen_rect_in_camera()\n"
     "    gain = confined_sensitivity_map(samples, frame_size, corners)\n"
     "    if gain is None:\n"
     "        raise NoGainMap(f\"{calibration}: no gain map could be built from \"\n"
     "                        f\"{len(rows)} rows (no screen outline, or the peaks \"\n"
     "                        f\"give no usable map), so the size gate has nothing \"\n"
     "                        f\"to act on.\")\n"
     "    return gain",
     "    samples = [Sample(screen=(0.0, 0.0),\n"
     "                      camera=(float(r[\"camera\"][0]), float(r[\"camera\"][1])),\n"
     "                      peak=float(r[\"peak\"])) for r in rows]\n"
     "    corners = mapping.screen_rect_in_camera()\n"
     "    return confined_sensitivity_map(samples, frame_size, corners)",
     "test_replay.py", "raises NoGainMap"),
    ("the csv area is compared in the wrong unit (full-frame px against processing px)",
     "tools/replay.py",
     "    return (full_frame_to_working_area(params, area_full_frame)\n            >= amplified_min_area(params, amplification))",
     "    return area_full_frame >= amplified_min_area(params, amplification)",
     "test_replay.py", "SCALE 0.5"),
    ("the log is never checked against the session",
     "tools/replay.py", "    check_log_matches_session(live, session, log)\n", "",
     "test_replay.py", "refuses a log from another session"),
    ("the report prints no gain map line",
     "tools/replay.py", '    print("gain map used: " + (', '    print("" + (',
     "test_replay.py", "prints the map"),
    ("the report does not say which question the source answers",
     "tools/replay.py", '    print(QUESTION.get(res.get("source", "video"), ""))', "    pass",
     "test_replay.py", "states which question"),
    ("the saved max_area is allowed to exceed the code's limit again",
     "src/lsa/tools/camera_lab.py",
     "self.sp_maxarea = spin(10, self.MAX_AREA_CEILING, self.MAX_AREA_CEILING, 50)",
     "self.sp_maxarea = spin(10, 100000, 2000, 50)",
     "test_ui_smoke.py", "max_area"),
    # -- the learner judges the screen when the shot was FIRED (2026-10-17) -----
    ("the learner reads `animated` after the hit again (it refuses every scoring hit)",
     "src/lsa/tools/camera_lab.py",
     # RE-QUOTED 2026-10-23: the line this anchored on was restructured when
     # corroboration was added (the moving test moved ahead of the peak test,
     # so a refused shot cannot spend a click), and `test_rules` section M
     # caught the anchor matching 0 times -- i.e. this fault had no live proof.
     # That is the C17 class exactly: the change that edits a line owes the
     # anchor quoting it.
     "        moving = mode is not None and mode.moving_at(shot.first_timestamp)",
     "        moving = mode is not None and mode.animated",
     "test_feedback.py", "is ACCEPTED"),
    ("a walking figure no longer counts as moving content",
     "src/lsa/modes/advancing.py",
     '        return (any(a.phase == "live" and a.moving(t) for a in self._actors)\n'
     '                or any(a.fall_span is not None and a.fall_span[0] < t < a.fall_span[1]',
     '        return (False\n'
     '                or any(a.fall_span is not None and a.fall_span[0] < t < a.fall_span[1]',
     "test_feedback.py", "MOVING is refused"),
    # -- the probe may not report a number its own dot contradicts ---------------
    ("the dot no longer gates the prediction (the anchor reads 100% again)",
     "src/lsa/exposure.py",
     "    return dot_clears_bar(probe, bar) and dot_clears_bar(anchor, anchor_bar)",
     "    return True",
     "test_exposure.py", "no longer '100%'"),
    ("a stand-in anchor inherits the -5 strong-end allowance again",
     "src/lsa/exposure.py",
     "    if abs(anchor.exposure - GUN_ANCHOR_EXPOSURE) > 1e-6:\n        return None\n",
     "    if False:\n        return None\n",
     "test_exposure.py", "stand-in anchor"),
    ("the probe ranges its dot at the brightest candidate again, not the anchor",
     "src/lsa/routines.py",
     "    queue.sort(key=lambda v: 0 if abs(v - expo.GUN_ANCHOR_EXPOSURE) < 1e-6 else 1)",
     "    pass",
     "test_routines.py", "first exposure written"),
    ("the auto-range crawls up the ladder one rung at a time again",
     "src/lsa/routines.py",
     "                    while len(ladder) > 1 and ladder[0] < _need:",
     "                    while False:",
     "test_routines.py", "straight to 0.26"),
    ("the strong end is never drawn in the shared photograph",
     "src/lsa/routines.py",
     "                  brightness=brightness, also=also)",
     "                  brightness=brightness, also=None)",
     "test_routines.py", "both ends were still drawn"),
    ("each spot is read over the whole frame, so the brighter dot is returned twice",
     "src/lsa/tools/camera_lab.py",
     "        R = self.PROBE_SPOT_HALF",
     "        R = 100000",
     "test_feedback.py", "reads its OWN dot"),
    ("an unmeasurable probe tells the log but not the target screen",
     "src/lsa/routines.py",
     "            yield Call(lambda: _say_unmeasurable(lab, choice.warning))",
     "            pass",
     "test_routines.py", "TARGET SCREEN is told"),
    # -- one illumination event is one shot: the halo rule (2026-10-17) --------
    ("the halo rule is never applied (the lumps of a bloom are shots again)",
     LASER, "            if p.halo_ratio > 0 and p.halo_radius_diag > 0:",
     "            if False:",
     "test_detector.py", "AFTER: the three lumps"),
    ("the area-rejected bloom is not an event (the failing frames: the bright thing is not in dets)",
     LASER, "        events: list[tuple[int, float, float, float]] = list(oversize or ())",
     "        events: list[tuple[int, float, float, float]] = []",
     "test_detector.py", "AFTER: the three lumps"),
    ("the radius is a constant that cannot cover a bloom (no multiple of the event's extent)",
     LASER, "                            <= max(sep, p.halo_radius_diag * diag)):",
     "                            <= max(sep, 30.0)):",
     "test_detector.py", "AFTER: the three lumps"),
    ("the radius is unbounded (every dim blob anywhere is folded, so a far second gun is lost)",
     LASER, "                            <= max(sep, p.halo_radius_diag * diag)):",
     "                            <= max(sep, 1e9)):",
     "test_detector.py", "FAR outside"),
    ("the event's own core is folded too (the only report of a bloom-only shot is lost)",
     LASER, "                            and p.halo_core_frac * diag < dist",
     "                            and -1.0 < dist",
     "test_detector.py", "own core"),
    ("the ratio is loose (a comparable second pistol is folded into the first)",
     LASER, "                    if (d.peak < p.halo_ratio * pk",
     "                    if (d.peak < 0.95 * pk",
     "test_detector.py", "TWO GUNS: a comparable"),
    ("a folded lump is not counted (it vanishes without a trace)",
     LASER, '        rejects["halo"] = len(found) - len(kept)',
     '        rejects["halo"] = 0',
     "test_detector.py", "own reason"),
    ("a folded lump is not remembered for the report",
     LASER, "                    if self.last_halo is None or d.peak > self.last_halo[0]:",
     "                    if False:",
     "test_detector.py", "remembered for the heard-not-seen"),
    ("Stats never counts a folded lump",
     "src/lsa/camera/capture.py", '            s.rejected_halo += rj.get("halo", 0)',
     "            pass",
     "test_feedback.py", "rejected_halo"),
    ("the capture loop never hands the folded blob to the evidence ring",
     "src/lsa/camera/capture.py", "                                halo=det.last_halo,",
     "                                halo=None,",
     "test_feedback.py", "evidence window carries the folded blob"),
    ("the evidence window drops the folded blob",
     "src/lsa/camera/evidence.py", '            "halo": halo,', '            "halo": None,',
     "test_detector.py", "carries the halo-folded blob"),
    ("the report has no branch for a folded blob (it falls through to 'nothing')",
     "src/lsa/shotlog.py", '    if summary.get("halo"):\n        return MISSED_HALO',
     '    if False:\n        return MISSED_HALO',
     "test_detector.py", "branch is HALO"),
    # -- a blob over max_area is a question, not a rejection (2026-10-17) -------
    ("the size ceiling is absolute again (every bright near-side bloom is discarded)",
     LASER, "                    admitted = (p.oversize_min_peak > 0\n"
     "                                and opk >= max(p.min_peak, p.oversize_min_peak))",
     "                    admitted = False",
     "test_detector.py", "IS a DETECTION"),
    ("any big blob is admitted whatever its brightness (a faint patch becomes a shot)",
     LASER, "                    admitted = (p.oversize_min_peak > 0\n"
     "                                and opk >= max(p.min_peak, p.oversize_min_peak))",
     "                    admitted = True",
     "test_detector.py", "FAINT (peak 20)"),
    ("the brightness bar for a big blob is the bare min_peak (the faint patch of room qualifies)",
     LASER, "                                and opk >= max(p.min_peak, p.oversize_min_peak))",
     "                                and opk >= p.min_peak)",
     "test_detector.py", "FAINT (peak 20)"),
    ("a bloom is positioned by its halo-dragged centroid again, not its core",
     LASER, "            if big and p.oversize_core_frac > 0:",
     "            if False:",
     "test_detector.py", "bloom position = its core"),
    ("the core rule is applied to EVERY blob (normal detections move)",
     LASER, "            if big and p.oversize_core_frac > 0:",
     "            if p.oversize_core_frac > 0:",
     "test_detector.py", "NORMAL-size blob is positioned exactly"),
    ("a rejected faint patch is not counted under 'area'",
     LASER, '                if not admitted:\n                    rejects["area"] += 1',
     '                if not admitted:\n                    rejects["area"] += 0',
     "test_detector.py", "counted 'area'"),
    ("a rejected faint patch is not remembered for the report",
     LASER, "        self.last_big_rejected = big_rejected",
     "        self.last_big_rejected = None",
     "test_detector.py", "remembered for the heard-but-not-seen"),
    ("the capture loop never hands the oversize patch to the evidence ring",
     "src/lsa/camera/capture.py", "                                big=det.last_big_rejected)",
     "                                big=None)",
     "test_feedback.py", "carried as evidence"),
    ("the evidence window drops the oversize patch",
     "src/lsa/camera/evidence.py", '            "big": big,', '            "big": None,',
     "test_detector.py", "carries the oversize blob"),
    ("the report has no branch for an oversize patch (it falls through to 'nothing')",
     "src/lsa/shotlog.py", '    if summary.get("big"):\n        return MISSED_TOO_BIG',
     '    if False:\n        return MISSED_TOO_BIG',
     "test_detector.py", "branch is TOO_BIG"),
    ("the duration ceiling is gone (a bright lamp-sized blob is a shot for as long as it burns)",
     LASER, "        return self.max_duration_ms > 0 and self._span_ms(tr) > self.max_duration_ms",
     "        return False",
     "test_detector.py", "duration ceiling poisons it"),
    ("the default brightness bar for a big blob drifts off the measured boundary",
     LASER, "    oversize_min_peak: int = 30\n", "    oversize_min_peak: int = 8\n",
     "test_detector.py", "defaults: oversize_min_peak 30"),
    # -- the bullet hole is exactly grey, alone and composited ------------------
    ("the hole's halo goes back to the warm off-white (red excess 5)",
     "src/lsa/display/layers.py", "        halo = grey_at(0.88) if dark else QColor(24, 24, 24)",
     "        halo = (QColor(grey_at(0.88).red() + 5, grey_at(0.88).green(), grey_at(0.88).blue() - 5)\n"
     "                if dark else QColor(24, 24, 24))",
     "test_modes.py", "hole ALONE"),
    ("the hole's rim goes back to the warm off-white (red excess 5)",
     "src/lsa/display/layers.py", "        rim = grey_at(1.0) if dark else QColor(28, 28, 28)",
     "        rim = (QColor(grey_at(1.0).red() + 5, grey_at(1.0).green(), grey_at(1.0).blue() - 5)\n"
     "               if dark else QColor(28, 28, 28))",
     "test_modes.py", "hole ALONE"),
    ("the impact flash goes back to amber (red excess 8)",
     "src/lsa/display/layers.py", "        flash = QColor(GREY_HI)",
     "        flash = QColor(GREY_HI.red() + 8, GREY_HI.green(), GREY_HI.blue())",
     "test_modes.py", "hole ALONE"),
    ("the light-background hole gains a warm tint (red excess 66, which survives the 16% alpha)",
     "src/lsa/display/layers.py", "        halo = grey_at(0.88) if dark else QColor(24, 24, 24)",
     "        halo = grey_at(0.88) if dark else QColor(90, 24, 24)",
     "test_modes.py", "LIGHT background"),
    # -- rejected-blob recording (rejected.csv) ---------------------------------
    # `record_rejects` was green in test_detector section 33 and test_replay with
    # nothing ever shown to fail it. Each fault below is one property of the
    # recording. NOT anchorable because NO check exists (see the report that
    # accompanied this block): the capture-side writer in camera/capture.py (row
    # values written to the csv, REJECT_ROW_CAP, the flag set at open and cleared
    # at close), the `roi` reason (recorded only for a blob straddling the
    # screen edge, because the mask is applied before labelling), and the
    # per-frame cap at the big / shape / roi / halo sites.
    ("the flag is ignored and rejects are ALWAYS recorded (the off-by-default promise is void)",
     LASER, "        rec = self.record_rejects\n", "        rec = True\n",
     "test_detector.py", "flag off (the default)"),
    ("an oversize blob is no longer recorded (reason 'big' vanishes)",
     LASER, "                    if (rec and big and opk >= rej_floor",
     "                    if (False and big and opk >= rej_floor",
     "test_detector.py", "recorded as big"),
    ("a size-gated blob is no longer recorded (reason 'amp_area' vanishes)",
     LASER, "                    if rec and len(rej_rows) < REJECT_ROWS_PER_FRAME:\n"
     "                        # The gate throws the blob away before its peak exists;",
     "                    if False and len(rej_rows) < REJECT_ROWS_PER_FRAME:\n"
     "                        # The gate throws the blob away before its peak exists;",
     "test_detector.py", "recorded as amp_area"),
    ("a streak rejected for its shape is no longer recorded (reason 'shape' vanishes)",
     LASER, "                if rec and len(rej_rows) < REJECT_ROWS_PER_FRAME:\n"
     "                    pk = int(",
     "                if False and len(rej_rows) < REJECT_ROWS_PER_FRAME:\n"
     "                    pk = int(",
     "test_detector.py", "recorded as shape"),
    ("a blob under min_peak is no longer recorded (reason 'dim' vanishes)",
     LASER, "                if (rec and peak >= rej_floor", "                if (False and peak >= rej_floor",
     "test_detector.py", "recorded as dim"),
    ("a lump folded into a brighter event is no longer recorded (reason 'halo' vanishes)",
     LASER, "                if id(d) not in kept_ids and len(rej_rows)",
     "                if False and id(d) not in kept_ids and len(rej_rows)",
     "test_detector.py", "recorded as halo"),
    ("a blob cut by max_blobs is no longer recorded (reason 'max_blobs' vanishes)",
     LASER, "            for d in kept[p.max_blobs:]:\n", "            for d in []:\n",
     "test_detector.py", "recorded as max_blobs"),
    ("an amp_area row is written with x and y swapped (wrong position)",
     LASER, '                            "amp_area", cents[i][0] * inv, cents[i][1] * inv,',
     '                            "amp_area", cents[i][1] * inv, cents[i][0] * inv,',
     "test_detector.py", "its position is the blob's"),
    ("an amp_area row is written with the wrong area (doubled)",
     LASER, '                            "amp_area", cents[i][0] * inv, cents[i][1] * inv,\n'
     "                            pk, area * inv * inv,",
     '                            "amp_area", cents[i][0] * inv, cents[i][1] * inv,\n'
     "                            pk, area * inv * inv * 2.0,",
     "test_detector.py", "recorded as amp_area"),
    ("a big row is written with x and y swapped (wrong position)",
     LASER, '                            "big", float(cents[i][0]) * inv, float(cents[i][1]) * inv,',
     '                            "big", float(cents[i][1]) * inv, float(cents[i][0]) * inv,',
     "test_detector.py", "the big row carries"),
    ("a big row is written with the wrong area (halved)",
     LASER, "                            opk, area * inv * inv, w * inv, h * inv,",
     "                            opk, area * inv * inv * 0.5, w * inv, h * inv,",
     "test_detector.py", "the big row carries"),
    ("a dim row is written with x and y swapped (wrong position)",
     LASER, '                        "dim", (x0 + w * 0.5) * inv, (y0 + h * 0.5) * inv,',
     '                        "dim", (y0 + h * 0.5) * inv, (x0 + w * 0.5) * inv,',
     "test_detector.py", "recorded as dim"),
    ("a max_blobs row is written with the wrong area (zero)",
     LASER, '                    rej_rows.append(("max_blobs", d.x, d.y, d.peak, d.area,',
     '                    rej_rows.append(("max_blobs", d.x, d.y, d.peak, 0.0,',
     "test_detector.py", "recorded as max_blobs"),
    ("the per-frame cap is not enforced where the amp_area rows are written",
     LASER, "                    if rec and len(rej_rows) < REJECT_ROWS_PER_FRAME:\n"
     "                        # The gate throws the blob away before its peak exists;",
     "                    if rec:\n"
     "                        # The gate throws the blob away before its peak exists;",
     "test_detector.py", "at most"),
    ("the per-frame cap is not enforced where the dim rows are written",
     LASER, "                if (rec and peak >= rej_floor\n"
     "                        and len(rej_rows) < REJECT_ROWS_PER_FRAME):",
     "                if (rec and peak >= rej_floor):",
     "test_detector.py", "the cap holds for dim"),
    ("the per-frame cap is not enforced where the max_blobs rows are written",
     LASER, "            for d in kept[p.max_blobs:]:\n                if len(rej_rows) < REJECT_ROWS_PER_FRAME:",
     "            for d in kept[p.max_blobs:]:\n                if True:",
     "test_detector.py", "the cap holds for max_blobs"),
    ("read_rejected() returns an EMPTY result for a session with no rejected.csv ('none recorded' reads as 'none rejected')",
     "tools/replay.py", "    if not path.exists():\n        return None\n    rows, by_reason, by_frame",
     "    if not path.exists():\n        return dict(rows=[], by_reason={}, by_frame={})\n    rows, by_reason, by_frame",
     "test_replay.py", "NONE RECORDED"),
    ("replay() turns a missing rejected.csv into an empty count instead of None",
     "tools/replay.py", "        rejected=None if rej is None else rej[\"by_reason\"],",
     "        rejected=rej[\"by_reason\"] if rej else {},",
     "test_replay.py", "NONE RECORDED"),
    ("the report prints a missing rejected.csv as 'none rejected'",
     "tools/replay.py", '        rj = res.get("rejected")\n',
     '        rj = res.get("rejected") or {}\n',
     "test_replay.py", "in the report too"),
    # -- /state on his real calibration ----------------------------------------
    ("/state treats held_out as a bare number again (crashes on the real dict)",
     "src/lsa/introspect/view.py", "        if isinstance(raw, dict):",
     "        if False:",
     "test_introspect.py", "held-out"),
    # -- the brightness grid (key U), 2026-10-23 -------------------------------
    # Every anchor below is quoted as the code reads AFTER the change that added
    # the routine; each is caught by a check in test_ui_smoke.py section "the
    # brightness grid (U)" whose label contains the text in the last column.
    ("the grid does not relax the detector's thresholds (a faint spot can never register)",
     CAMLAB, "            self._relax_for_calibration()\n"
             "            if self._sgrid_prior_gain is not None:\n",
     "            if self._sgrid_prior_gain is not None:\n",
     "test_ui_smoke.py", "faint shots can register"),
    ("one shot is enough to record a spot (a single stray teaches the map)",
     CAMLAB, "        recent = self._sgrid_shots[-self._grid_window:]\n"
             "        group = self._agreeing_group(recent, want)\n",
     "        recent = self._sgrid_shots[-self._grid_window:]\n"
             "        group = self._agreeing_group(recent, 1)\n",
     "test_ui_smoke.py", "two agreeing shots do not record a spot"),
    ("the grid's own ORDER is never consulted (an object that appears mid-run is a spot)",
     CAMLAB, "                 or self._out_of_grid_order(\n"
             "                     i, mx, my, points=self._sgrid_points,",
     "                 or \"\" and self._out_of_grid_order(\n"
             "                     i, mx, my, points=self._sgrid_points,",
     "test_ui_smoke.py", "with the aim check off, the grid's own ORDER"),
    ("the saved mapping's prediction is never consulted (an object present from spot 1 is a spot)",
     CAMLAB, "        wrong = (self._sgrid_off_aim(i, mx, my)\n",
     "        wrong = (\"\" and self._sgrid_off_aim(i, mx, my)\n",
     "test_ui_smoke.py", "a steady object at the wrong place is REJECTED"),
    ("finishing clears the mapping (the grid's whole premise is that it never touches it)",
     CAMLAB, "        prior = self._sgrid_prior_gain\n"
             "        self._sgrid_end(\"finished\", restore_prior_gain=False)\n",
     "        prior = self._sgrid_prior_gain\n"
             "        self.mapping = None\n"
             "        self._sgrid_end(\"finished\", restore_prior_gain=False)\n",
     "test_ui_smoke.py", "it never touched the mapping"),
    ("the ONE exit forgets to restore the thresholds (every way out leaves the bar relaxed)",
     CAMLAB, "        # The consequential step first, before anything below can raise.\n"
             "        self._restore_levels()\n",
     "        # The consequential step first, before anything below can raise.\n"
             "        pass\n",
     "test_ui_smoke.py", "thresholds are back"),
    ("closing the target screen does not end the grid (the flag and the bar outlive the window)",
     CAMLAB, "        self._sgrid_end(\"the target screen closed\")\n",
     "        pass\n",
     "test_ui_smoke.py", "closing the target screen ends it"),
    ("closing the app does not end the grid (the relaxed bar is saved into settings.local.json)",
     CAMLAB, "        self._sgrid_end(\"the app is closing\")\n"
             "        self._restore_levels()\n",
     "        pass\n",
     "test_ui_smoke.py", "settings.local.json at close: the detector bar WRITTEN there"),
    ("closing the app leaves the SHOT grid's (G) relaxed bar standing (the bare restore after _sgrid_end is gone; "
     "it is saved into settings.local.json)",
     CAMLAB, "        self._sgrid_end(\"the app is closing\")\n"
             "        self._restore_levels()\n",
     "        self._sgrid_end(\"the app is closing\")\n"
             "        pass\n",
     "test_ui_smoke.py", "shot grid (G): closing the app in the middle of it"),
    ("the idle deadline is gone (a forgotten grid holds the screen for ever)",
     CAMLAB, "            if quiet >= self.SGRID_IDLE_LIMIT_S:\n",
     "            if False:\n",
     "test_ui_smoke.py", "the idle limit gives the screen back"),
    ("a camera that went away does not end the grid",
     CAMLAB, "            if self.capture is None:\n"
             "                self._sgrid_end(\"the camera was disconnected\")\n"
             "                return\n",
     "            if False:\n"
             "                self._sgrid_end(\"the camera was disconnected\")\n"
             "                return\n",
     "test_ui_smoke.py", "the camera going away ends it"),
    ("the four-spot minimum is gone (three spots build a map, or fail without saying why)",
     CAMLAB, "        if len(samples) < self.LEARN_MIN_BINS or not spans:\n",
     "        if len(samples) < 1:\n",
     "test_ui_smoke.py", "it needs FOUR"),
    ("a second start re-enters the run (it loses the map that was set aside)",
     CAMLAB, "        if self._sgrid_active:\n"
             "            self._log(\"brightness grid is already running; ignoring the second request\")\n"
             "            return\n",
     "        if False:\n"
             "            self._log(\"brightness grid is already running; ignoring the second request\")\n"
             "            return\n",
     "test_ui_smoke.py", "a second start changes nothing"),
    ("the grid is not in the central exclusion list (a Director routine may start over it)",
     CAMLAB, "        (\"_sgrid_active\", \"the brightness grid\"),\n", "",
     "test_ui_smoke.py", "central exclusion list"),
    ("its shots are not routed away from the mode (they score, spend rounds and draw holes)",
     CAMLAB, "        if self._sgrid_active and self._camera_sees_our_screen():\n",
     "        if False and self._camera_sees_our_screen():\n",
     "test_ui_smoke.py", "shots score NOTHING"),
    ("the map is built from the mapping's own markers, not from the shots",
     CAMLAB, "                self.mapping, samples=samples,\n",
     "                self.mapping,\n",
     "test_ui_smoke.py", "built from the SHOTS"),
    ("the measured falloff is not persisted (via the brightness grid)",
     CAMLAB, "        self._save_sensitivity(usable)\n", "        pass\n",
     "test_ui_smoke.py", "reached the file"),
    ("a map already in force is not set aside (the grid measures through it)",
     CAMLAB, "            if self._sgrid_prior_gain is not None:\n"
             "                self._set_gain_map(\n"
             "                    None, ",
     "            if False:\n"
             "                self._set_gain_map(\n"
             "                    None, ",
     "test_ui_smoke.py", "SET ASIDE"),
    ("the set-aside map is not remembered at the one funnel (so nothing can put it back)",
     CAMLAB, "        self._gain_in_force = gain\n", "        pass\n",
     "test_ui_smoke.py", "SET ASIDE"),
    ("the set-aside map is not put back on an exit (detection is silently uniform afterwards)",
     CAMLAB, "        if restore_prior_gain and prior is not None:\n",
     "        if False and prior is not None:\n",
     "test_ui_smoke.py", "map set aside"),
    ("the screen mask is dropped during the grid, as the shot grid does (the window beside the monitor supplies spots)",
     CAMLAB, "        if self._grid_active:\n"
             "            return None\n"
             "        # `_sgrid_active` (the brightness grid) is DELIBERATELY not tested\n",
     "        if self._grid_active or self._sgrid_active:\n"
             "            return None\n"
             "        # `_sgrid_active` (the brightness grid) is DELIBERATELY not tested\n",
     "test_ui_smoke.py", "screen MASK stays on"),
    ("the U key does not start it",
     CAMLAB, "            self.start_sensitivity_grid()\n"
             "            return\n"
             "        if key == Qt.Key_N:\n",
     "            pass\n"
             "            return\n"
             "        if key == Qt.Key_N:\n",
     "test_ui_smoke.py", "U STARTS it"),
    ("the key legend does not mention U",
     "src/lsa/display/layers.py",
     "         (\"G\", \"calibrate by shooting\"),\n"
     "         (\"U\", \"calibrate brightness by shooting\")),",
     "         (\"G\", \"calibrate by shooting\")),",
     "test_ui_smoke.py", "the key legend says U beside G"),
    ("the Keyboard Shortcuts screen does not list U (pinned word for word by test_modes section 46)",
     "src/lsa/modes/shortcuts.py",
     "    Entry(\"U\", \"Calibrate Brightness by Shooting\", (int(Qt.Key_U),)),\n", "",
     "test_modes.py", "the fixed rows are the ones the owner approved"),
    ("the shortcuts screen words U differently from the legend (two wordings for one key)",
     "src/lsa/modes/shortcuts.py", "\"Calibrate Brightness by Shooting\"", "\"Measure Brightness\"",
     "test_ui_smoke.py", "names U in the legend's own words"),
]


SUITE_TIMEOUT_S = 900


def judge(rc: int, text: str, frag: str) -> tuple[str, str]:
    """('caught' | 'MISSED' | 'BROKEN', why). tests/mutation_verdict.py is the
    ONE judgement: caught only when the entry's OWN named check fails. The
    fallback this used to carry -- `(named or fails)`, i.e. "no check names the
    fault, so accept any failure" -- is what counted 22 crashed checks as caught
    on 2026-10-23."""
    return verdict.classify(rc, text, frag)


def _scratch_tree(d: Path) -> None:
    # The ONE copy (tests/asset_env.py): code, every asset folder, and COPIES of
    # the files he owns. Four hand-written lists of folders here and in the other
    # runners are why `config/`, `sounds/` and `.venv/pyvenv.cfg` were missing.
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


def run(i: int, m: tuple) -> tuple:
    """(name, label, why) with label one of caught / MISSED / BROKEN / UNPLACED."""
    name, rel, old, new, test, frag = m
    d = OUT / str(i)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _scratch_tree(d)
    p = d / rel
    raw = p.read_bytes().decode()
    mutated = anchors.apply(raw, old, new)
    if mutated is None:
        shutil.rmtree(d, ignore_errors=True)
        return name, "UNPLACED", (f"MUTATION DID NOT APPLY: {old[:60]!r}: "
                                  + anchors.not_found_reason(raw, old))
    p.write_bytes(mutated.encode())
    rc, text = _run_suite(d, test)
    shutil.rmtree(d, ignore_errors=True)
    label, why = judge(rc, text, frag)
    return name, label, why


def _baseline(pick: list) -> list[str]:
    """Every suite ANY picked entry names, unmutated, in a scratch tree built the
    way the mutants' are. 2026-10-23: this runner had no baseline at all and
    reported 112 of 112 over a red test_ui_smoke."""
    d = OUT / "baseline"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _scratch_tree(d)
    try:
        return verdict.baseline({M[i][4] for i in pick}, lambda s: _run_suite(d, s))
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
    workers = budget.workers(8)
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda i: (i, run(i, M[i])), pick))
    bad = 0
    for i, (name, label, why) in res:
        bad += label != "caught"
        print(f"  {label:<8} {i:2d} {name}")
        print(f"             -> {why[:220]}")
    shutil.rmtree(OUT, ignore_errors=True)
    broken = sum(1 for _i, r in res if r[1] == "BROKEN")
    print(f"{len(res) - bad} of {len(res)} area-gate faults caught "
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
