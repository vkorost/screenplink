"""Everything the introspection server knows, collected in one place.

THE GOVERNING PRINCIPLE, agreed with the owner: **expose measurements, not
conclusions.** On 2026-10-17 the app printed "100% of his shots clear the bar,
margin 6.1x" while the measurement in the same line showed its own test dot
below its own threshold. An endpoint returning the app's opinion would have
made me wrong faster.

So the numbers here are the ones the detector and the camera actually run on:
the effective `DetectorParams`, the real gain map, the score map it thresholds,
the raw blob list, the settings as loaded beside the settings as saved. The
app's own verdicts are exposed too -- under `app_says`, labelled as the app's
claim rather than as evidence -- because cross-checking a claim against the
measurement beside it is the whole point.

This is also the ONLY module that touches the `CameraLab`, and it reaches the
camera exclusively through `readonly.capture_reader`, so a handler cannot write
a control by forgetting not to (RULES A4, design constraint 1).
"""

from __future__ import annotations

import math
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import cv2
import numpy as np

from lsa.artwork import assets
from . import readonly
from .readonly import WriteAttempted          # noqa: F401  (re-exported for handlers)


def _num(v):
    """JSON cannot encode a numpy scalar, and `Mapping.to_dict` once returned
    False from `save()` for exactly that reason. Everything leaving here is a
    plain Python number."""
    if v is None:
        return None
    if isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    return v


def _params_dict(p) -> dict:
    out = {}
    for f in fields(p):
        v = getattr(p, f.name)
        if f.name == "roi_polygon":
            out[f.name] = ([[_num(x), _num(y)] for x, y in v] if v else None)
        elif f.name == "mode":
            out[f.name] = str(getattr(v, "value", v))
        else:
            out[f.name] = _num(v)
    return out


class LabView:
    """Read-only access to a running `CameraLab` (or a stand-in with the same
    attributes, which is what the tests drive)."""

    def __init__(self, lab, bridge=None, tap=None, log=None) -> None:
        self._lab = lab
        self._bridge = bridge
        self._tap = tap
        # Anything this server DOES is logged, so a mode change nobody at the
        # machine caused is attributable. Only `press_key` uses it: every
        # other endpoint reads.
        self._log = log
        self.server = None
        self.started = time.perf_counter()
        self.started_wall = time.time()

    # -- the camera, read-only -------------------------------------------

    def _cap(self):
        return readonly.capture_reader(self._lab.capture)

    def _det(self, cap=None):
        return readonly.detector_reader(self._cap() if cap is None else cap)

    # -- /health ---------------------------------------------------------

    def health(self, server_stats: dict) -> dict:
        lab = self._lab
        cap = self._cap()
        return {
            "ok": True,
            "app": "lsa camera lab",
            "version": _version(),
            "pid": _pid(),
            "role": _call(lab, "_process_role"),
            "uptime_s": _num(time.perf_counter() - self.started),
            "started_wall": _num(self.started_wall),
            "now_wall": _num(time.time()),
            # "open" means the app HOLDS a capture thread; "streaming" means
            # frames have actually arrived. Two different facts, and this
            # project has a device class that opens perfectly and delivers a
            # picture of a disconnected-cable icon (RULES A11), so one answer
            # would be the wrong one.
            "camera_open": bool(cap is not None),
            "camera_streaming": bool(cap is not None and cap._stats.frames > 0),
            "frames_ever": int(lab._frames_ever),
            "target_window": bool(lab.target_window is not None),
            # THE COST THIS SERVER IMPOSES, as a number rather than a claim.
            "server": server_stats,
        }

    # -- /state ----------------------------------------------------------

    def state(self) -> dict:
        lab = self._lab
        cap = self._cap()
        det = self._det(cap)
        return {
            "at_wall": _num(time.time()),
            "at_perf": _num(time.perf_counter()),
            "mode": self._mode(),
            "layers": self._layers(),
            "detector": self._detector_state(cap, det),
            "tracker": self._tracker_state(cap),
            "floor": self._floor(),
            "exposure": self._exposure(cap),
            "gain_map": self._gain_map(det),
            "mapping": self._mapping(),
            "screen": self._screen(),
            "camera": self._camera(cap, det),
            "director": self._director(),
            "session": self._session(),
            "audio": self._audio(),
            # LABELLED AS A CLAIM, NOT AS EVIDENCE. Everything above is
            # measured; this is what the app concluded from it, kept here so
            # the two can be compared in one request instead of being
            # confused for each other.
            "app_says": self._app_says(),
        }

    # -- pieces of /state ------------------------------------------------

    def _mode(self) -> dict:
        host = self._lab.layers.get("modes")
        mode = host.mode if host is not None else None
        if mode is None:
            return {"active": None}
        key = None
        try:
            from lsa.modes.base import _REGISTRY
            for entry in _REGISTRY.values():
                if isinstance(mode, entry.cls) and entry.key is not None:
                    key = _key_name(entry.key)
                    break
        except Exception:                                # pragma: no cover
            key = None
        outcome = host.last_outcome
        return {
            "active": mode.name,
            "class": type(mode).__name__,
            "title": mode.title,
            "key": key,
            "status": _call(mode, "status"),
            "visible": bool(mode.visible),
            "animated": bool(mode.animated),
            "last_outcome": (None if outcome is None else {
                "hit": bool(outcome.hit),
                "points": _num(outcome.points),
                "target_id": outcome.target_id,
            }),
        }

    def _layers(self) -> list:
        out = []
        for la in self._lab.scene.layers:
            out.append({
                "name": la.name,
                "class": type(la).__name__,
                "z": _num(la.z),
                "visible": bool(la.visible),
                "animated": bool(la.animated),
                "read_rate_hz": _num(la.read_rate_hz),
            })
        return out

    def _detector_state(self, cap, det) -> dict:
        if det is None:
            return {"running": False,
                    "note": "the camera is not open, so no detector is running"}
        p = det.params
        out = {"running": True, "effective": _params_dict(p)}
        out["last_frame"] = {
            "index": _num(det._frame_index),
            "process_ms": _num(det.last_process_ms),
            "candidates": _num(det.last_candidates),
            "frame_peak": _num(det.last_frame_peak),
            "dim_peak": _num(det.last_dim_peak),
            "dim_pos": ([_num(det.last_dim_pos[0]), _num(det.last_dim_pos[1])]
                        if det.last_dim_pos else None),
            "off_screen_peak": _num(det.last_off_screen_peak),
            "rejects": {k: int(v) for k, v in (det.last_rejects or {}).items()},
            "amp_gate": (None if det.last_amp_gate is None else {
                "area": _num(det.last_amp_gate[0]),
                "needed": _num(det.last_amp_gate[1]),
                "amplification": _num(det.last_amp_gate[2]),
                "pos": [_num(det.last_amp_gate[3]), _num(det.last_amp_gate[4])],
            }),
        }
        return out

    def _tracker_state(self, cap) -> dict:
        tr = readonly.tracker_reader(cap)
        if cap is None:
            return {"running": False}
        cfg = cap._grouper_config
        out = {"running": True, "effective": _dc(cfg)}
        if tr is not None:
            out["live"] = {
                "shots": _num(tr.count),
                "open_tracks": _num(tr.open_tracks),
                "max_concurrent": _num(tr.max_concurrent),
                "rejected_too_short": _num(tr.rejected_runs),
                "rejected_too_long": _num(tr.rejected_too_long),
            }
        return out

    def _floor(self) -> dict:
        lab = self._lab
        cap = self._cap()
        det = self._det(cap)
        bar = None if det is None else _num(det.params.min_peak)
        thr = None if det is None else _num(det.params.threshold)
        return {
            # The floor is what was MEASURED against this room; the bar is what
            # the detector is actually running on. Reporting only one of them is
            # how "min_peak = floor + 15" sat above his real far-side shots for
            # a session with nothing to compare it against.
            "measured_against_this_room": bool(lab._floor_measured),
            "noise_peak": _num(lab._noise_peak),
            "bar_min_peak": bar,
            "threshold": thr,
            "measurement_in_flight": bool(lab._floor_in_flight),
            "retries_after_a_shot_spoiled_it": int(lab._noise_retries),
            "deferred_to_the_probe": bool(lab._floor_deferred),
            "floor_before_gain_map": _num(lab._floor_before_gain),
            "single_frame_peak": _num(lab.SINGLE_FRAME_PEAK),
        }

    def _exposure(self, cap) -> dict:
        lab = self._lab
        written = {}
        gain_open = None
        if cap is not None:
            written = {k: _num(v) for k, v in dict(cap._control_values).items()}
            gain_open = _num(cap.gain_at_open())
        return {
            # Only what THIS APP wrote. The camera's readback lies -- it reports
            # exposure -6.0 and auto_exposure 0.0 while being neither -- so a
            # driver read here would be worse than no number at all, and taking
            # one costs a UVC transaction (RULES A4).
            "written_by_this_app": written,
            # None is a REAL value for these (never measured on this camera;
            # "auto" in the candidate list), read off attributes the lab
            # always declares -- not a default standing in for a missing name.
            "shooting": _num(lab._shooting_exposure),
            "at_boot_from_disk": _num(lab._boot_exposure),
            "calibration_candidates": [
                _num(v) for v in lab._calib_exposures],
            "gain_as_the_camera_reported_at_open": gain_open,
            "note": "readbacks are not reported: this camera lies about them, "
                    "and asking costs a control transaction.",
        }

    def _gain_map(self, det) -> dict:
        if det is None:
            return {"present": False, "note": "the camera is not open"}
        gain = det._gain
        lab = self._lab
        mapping = lab.mapping
        regions = len(mapping.sensitivity) if mapping is not None else 0
        if gain is None:
            return {
                "present": False,
                "regions_known_to_the_mapping": regions,
                "note": "no sensitivity normalisation is being applied, so a "
                        "peak is the raw score",
            }
        g = np.asarray(gain, dtype=np.float32)
        floor = float(g.min()) if g.size else 0.0
        return {
            "present": True,
            "shape": [int(g.shape[0]), int(g.shape[1])],
            "min": _num(floor),
            "max": _num(float(g.max()) if g.size else 0.0),
            "mean": _num(float(g.mean()) if g.size else 0.0),
            # What the detector actually multiplies a score by, which is the
            # number that matters: it is 1/gain, floored, so the amplification
            # at the weakest place is the one that can turn noise into signal.
            "max_amplification": _num(1.0 / max(floor, 1e-3)),
            "regions_known_to_the_mapping": regions,
            "shots_learned_from": len(lab._sens_shots),
            "shots_at_last_fit": int(lab._sens_applied),
            "refused_to_learn_from": dict(lab._sens_ignored),
            # Faint detections taught on a trigger click's word: ever, and
            # among the samples now held. A map built mostly on these is a
            # different claim from one built on bright shots.
            "admitted_on_the_microphones_word": int(lab._sens_admitted),
            "held_samples_resting_on_a_click": int(lab._sens_by_click),
        }

    def _mapping(self) -> dict:
        lab = self._lab
        m = lab.mapping
        if m is None:
            return {
                "present": False,
                "note": "there is nowhere to draw a hit. A detection is "
                        "reported and not placed.",
                "measured_this_run": False,
            }
        mm = _call(lab, "_mm_per_px")
        # `held_out` is a DICT on a real calibration ({median_px, worst_px,
        # median_mm, worst_mm, points}) and was a bare number on the stand-in
        # and on older files; `held * mm` crashed /state on his real file.
        # The measured millimetres are preferred: they were computed against
        # the real screen, where `mm` here is a current estimate.
        # THE ONE getattr-WITH-A-DEFAULT LEFT IN THIS FILE THAT READS THE APP'S
        # OWN OBJECT, and it is deliberate: `held_out` is not a declared field
        # of `Mapping`. The app attaches it after the held-out fit and
        # `Mapping.load` attaches it from the file, so a mapping that has not
        # been judged yet genuinely lacks the attribute, and "not judged" is the
        # true answer. `test_introspect` section 12b allowlists exactly this line
        # by name; a second one fails the build.
        raw = getattr(m, "held_out", None)
        held_worst = held_mm = held_worst_mm = held_n = None
        if isinstance(raw, dict):
            held = _num(raw.get("median_px"))
            held_worst = _num(raw.get("worst_px"))
            held_mm = _num(raw.get("median_mm"))
            held_worst_mm = _num(raw.get("worst_mm"))
            pts = raw.get("points")
            held_n = len(pts) if isinstance(pts, (list, tuple)) else None
        else:
            held = _num(raw)
        if held_mm is None and held is not None and mm:
            held_mm = _num(held * mm)
        return {
            "present": True,
            # HELD-OUT, FIRST. RMS over the fitted points flatters any model
            # with enough freedom; one fit had an RMS of 1287px on a 3840px
            # screen and was saved and announced as a success (RULES C9).
            "held_out_px": held,
            "held_out_mm": held_mm,
            "held_out_worst_px": held_worst,
            "held_out_worst_mm": held_worst_mm,
            "held_out_points": held_n,
            "rms_after_px": _num(m.rms_after),
            "rms_before_px": _num(m.rms_before),
            "max_error_px": _num(m.max_error),
            "mm_per_screen_px": _num(mm),
            "screen_size": [int(x) for x in m.screen_size],
            "frame_size": [int(x) for x in m.frame_size],
            "samples": len(m.samples),
            "dot_radius": _num(m.dot_radius),
            "sensitivity_regions": len(m.sensitivity),
            "measured_this_run": bool(lab._mapping_measured_now),
            "confirmed_by_the_two_marker_check": bool(lab._check_confirmed),
            "two_marker_check_attempted": bool(lab._check_tried),
        }

    def _screen(self) -> dict:
        lab = self._lab
        win = lab.target_window
        geo = None
        in_front = None
        why = ""
        if win is not None:
            try:
                r = win.geometry()
                geo = [int(r.x()), int(r.y()), int(r.width()), int(r.height())]
            except Exception:                            # pragma: no cover
                geo = None
            # `in_front_of_camera()` only READS window stacking. The app's own
            # `_screen_is_ours()` raises the window as a side effect, which is
            # right for a measurement about to be taken and wrong for a
            # question being asked from outside -- an instrument must not move
            # what it measures.
            try:
                in_front = bool(win.in_front_of_camera())
            except Exception:                            # pragma: no cover
                in_front = None
            # Read OUTSIDE the try: `in_front_of_camera()` is what sets it, and
            # a wrong name here must raise rather than be swallowed into
            # "in_front is None" (RULES C8).
            why = str(win.cover_reason or "")
        quad = lab.screen_quad
        return {
            "target_window_open": bool(win is not None),
            "target_window_ever_opened": bool(lab._target_ever_opened),
            "in_front_of_the_camera": in_front,
            "cover_reason": why,
            "geometry": geo,
            # The one predicate every consumer of a detection consults
            # (RULES B9). Reported as the app last computed it, with the
            # reasons, because a silent suppression is indistinguishable from
            # a detector that has stopped working.
            "camera_sees_our_background": bool(lab._cover_ok),
            "detections_suppressed_because_it_did_not": int(lab._cover_suppressed),
            "suppression_causes": sorted(lab._cover_causes),
            "flash_quad_confidence": (None if quad is None
                                      else _num(quad.confidence)),
            "calibrating_flag": bool(lab._calibrating),
        }

    def _camera(self, cap, det) -> dict:
        lab = self._lab
        if cap is None:
            return {
                "open": False,
                "frames_ever": int(lab._frames_ever),
                "devices_measured_blind": sorted(
                    int(i) for i in lab._blind_devices),
            }
        s = cap._stats
        cfg = cap.config
        info = dict(cap._open_info or {})
        return {
            "open": True,
            "running": bool(cap._running),
            "requested": {"index": _num(cfg.index), "backend": cfg.backend,
                          "width": _num(cfg.width), "height": _num(cfg.height),
                          "fourcc": cfg.fourcc, "fps": _num(cfg.fps)},
            "negotiated": {k: _num(v) for k, v in info.items()
                           if not isinstance(v, (list, dict))},
            "fps": _num(s.fps),
            "frames": _num(s.frames),
            "frames_ever": int(lab._frames_ever),
            "read_ms": _num(s.read_ms),
            "detect_ms": _num(s.process_ms),
            # Why this matters more than it looks: the capture loop is
            # read + detect, so under about 14 ms it holds 60 fps and over it
            # falls to 30 -- which halves the camera frames inside a 100 ms
            # pulse. It is a cliff, not a slope.
            "loop_ms": _num((s.read_ms or 0.0) + (s.process_ms or 0.0)),
            "dropped_reads": _num(s.dropped),
            # THE ONE OPERATION MEASURED TO HANG THIS CAMERA (RULES A4).
            "control_writes": _num(s.control_writes),
            "recording": bool(cap._writer is not None),
            "recorded_frames": _num(s.recorded_frames),
            "dropped_record": _num(s.dropped_record),
            "rolling_1s": {"frame_peak": _num(s.frame_peak),
                           "dim_peak": _num(s.dim_peak),
                           "off_screen_peak": _num(s.off_screen_peak)},
            "cumulative_rejects": {"dim": _num(s.rejected_dim),
                                   "off_screen": _num(s.rejected_off_screen),
                                   "amp_area": _num(s.rejected_amp_area),
                                   "halo": _num(s.rejected_halo),
                                   "too_short": _num(s.rejected_runs),
                                   "too_long": _num(s.rejected_too_long)},
            "shots": _num(s.shots),
            "last_detection_s_ago": _num(
                (time.perf_counter() - s.last_detection_at)
                if s.last_detection_at else None),
            "last_shot_s_ago": _num(
                (time.perf_counter() - s.last_shot_at) if s.last_shot_at else None),
            "elongation": _num(s.elongation),
            "devices_measured_blind": sorted(
                int(i) for i in lab._blind_devices),
        }

    def _director(self) -> dict:
        d = self._lab.director
        return {
            "present": True,
            "running": _call(d, "busy_with"),
            "label": _call(d, "busy_label"),
            "waiting_for": _call(d, "waiting_for"),
            "is_drawing": bool(_call(d, "is_drawing")),
            "owns_detections": bool(_call(d, "owns_detections")),
            "measuring_screen": bool(_call(d, "measuring_screen")),
            "world_busy": _call(self._lab, "_screen_busy"),
            "queue": [j.name for j in list(d._queue)],
            "control_writes": _num(d.control_writes),
            "control_writes_skipped": _num(d.control_writes_skipped),
            "history": [{"name": n, "outcome": o, "ms": _num(ms)}
                        for n, o, ms in list(d.history)[-20:]],
            "refusals": [{"wanted": a, "busy_with": b}
                         for a, b in list(d.refusals)[-20:]],
            # MUST STAY EMPTY. Two routines interleaving is the class of fault
            # the one run slot exists to make impossible.
            "overlaps": list(d.overlaps),
        }

    def _session(self) -> dict:
        s = self._lab.session
        return {
            "shots": _num(s.shots),
            "hits": _num(s.hits),
            "misses": _num(s.misses),
            # shots == hits + misses + unplaced, always; dry pulls are their own count
            "unplaced": _num(s.unplaced),
            "dry_pulls": _num(s.dry_pulls),
            "score": _num(s.score),
            "hit_rate": _num(_call(s, "hit_rate")),
            "string_time_s": _num(_call(s, "string_time_s")),
            "last_split_ms": _num(_call(s, "last_split_ms")),
            "best_split_ms": _num(_call(s, "best_split_ms")),
            "label": s.label,
        }

    def _audio(self) -> dict:
        """The sound output, the microphone, and what each says about the other.

        EVERY NAME HERE IS READ DIRECTLY, with no default. Five of them were
        wrong for weeks (`device_name`, `threshold`, `clicks`, `heard_only`,
        `seen_only` -- the real ones are `device`, `min_level`, `heard`,
        `heard_not_seen`, `seen_not_heard`) and every one reported null
        through a `getattr` default. A null here looks like an answer: it hid
        the heard-versus-seen counts the gun test is built on. A wrong name
        must raise in the first second.
        """
        lab = self._lab
        mixer = lab.mixer
        listener = lab.listener
        corr = lab.corroborator
        # One locked read of the three fast-moving numbers, so they describe
        # the same instant.
        level, floor, heard = listener.snapshot()
        return {
            "output_available": bool(mixer.available),
            "output_name": str(mixer.output_name),
            "output_latency_ms": _num(mixer.latency_ms),
            "output_error": str(mixer.error),
            "listening": bool(listener.running),
            # An input device INDEX, or None for the system default. The
            # listener holds no name; the saved name is in /settings.
            "mic_device": _num(listener.device),
            # The three numbers a click must clear, each measured: the
            # absolute level, the adaptive room floor, and how many times
            # above that floor a block must sit. The bar in force is
            # max(min_level, floor * margin) -- not computed here, because a
            # copy of that rule in a second place is how a bar drifts.
            "mic_min_level": _num(listener.min_level),
            "mic_floor": _num(floor),
            "mic_margin": _num(listener.margin),
            "mic_level": _num(level),
            "mic_error": str(listener.error),
            "clicks_heard": _num(heard),
            "heard_and_seen": _num(corr.both),
            "heard_not_seen": _num(corr.heard_not_seen),
            "seen_not_heard": _num(corr.seen_not_heard),
            # CAN THE MICROPHONE HEAR THE SPEAKERS, as a measurement: the
            # coupling and the clean readings it rests on, what was refused
            # and why, and whether it is stale or distrusted. This decides
            # whether the microphone may listen through the app's own bang
            # or must go deaf for the flat 0.8 s.
            "coupling": _jsonable(listener.coupling_report()),
        }

    # -- /settings -------------------------------------------------------

    def settings(self) -> dict:
        """EFFECTIVE beside SAVED beside the CODE DEFAULT, and which differ.

        This is the endpoint that would have shown `max_area = 2000` in his
        settings against a code default of 1500 in one request. The three
        columns are needed, not two: a saved value silently beating a code
        default is a fault this project has shipped twice -- `min_frames` sat
        at 1 for days because `settings.local.json` held the old value, and
        `holes: false` meant every shot he fired was detected, mapped and
        drawn into a layer that was switched off.
        """
        import json
        lab = self._lab
        cap = self._cap()
        det = self._det(cap)

        disk_path = _settings_path()
        disk: dict = {}
        disk_error = ""
        if disk_path is not None and disk_path.exists():
            try:
                disk = json.loads(disk_path.read_text(encoding="utf-8"))
            except Exception as exc:                     # noqa: BLE001
                disk_error = repr(exc)

        from lsa.camera.capture import CaptureConfig, GrouperConfig
        from lsa.detect.laser import DetectorParams
        from lsa.settings import AppSettings

        rows: list[dict] = []

        def add(group: str, key: str, effective, saved, default) -> None:
            rows.append({
                "group": group, "key": key,
                "effective": _num(effective),
                "saved_on_disk": _num(saved),
                "code_default": _num(default),
                "differs_from_saved": not _same(effective, saved),
                "differs_from_default": not _same(effective, default),
            })

        # the detector, as the capture thread is actually running it
        dflt_det = DetectorParams()
        saved_det = dict(disk.get("detector") or {})
        eff_det = _params_dict(det.params) if det is not None else {}
        for f in fields(dflt_det):
            if f.name == "roi_polygon":
                continue                   # measured, never saved; see /state
            add("detector", f.name,
                eff_det.get(f.name, None) if det is not None else None,
                saved_det.get(f.name, None),
                _num(getattr(dflt_det, f.name)) if f.name != "mode"
                else str(dflt_det.mode.value))

        dflt_tr = GrouperConfig()
        saved_tr = dict(disk.get("tracker") or {})
        eff_tr = _dc(cap._grouper_config) if cap is not None else {}
        for f in fields(dflt_tr):
            add("tracker", f.name, eff_tr.get(f.name), saved_tr.get(f.name),
                getattr(dflt_tr, f.name))

        # the app's own settings, as loaded into memory
        live = lab.settings
        dflt_app = AppSettings()
        for f in fields(dflt_app):
            if f.name in ("detector", "tracker", "layers"):
                continue                   # expanded into their own groups
            # `f` comes from AppSettings itself, so this name exists by
            # construction. `live` is None only in a stand-in; a real lab
            # always has one.
            add("app", f.name,
                getattr(live, f.name) if live is not None else None,
                disk.get(f.name, None), getattr(dflt_app, f.name))

        # the camera format actually negotiated, against what was asked for
        dflt_cfg = CaptureConfig()
        if cap is not None:
            info = dict(cap._open_info or {})
            for f in fields(dflt_cfg):
                asked = getattr(cap.config, f.name)
                got = info.get(f.name, asked)
                rows.append({
                    "group": "camera", "key": f.name,
                    "effective": _num(got), "requested": _num(asked),
                    "saved_on_disk": _num(disk.get(
                        {"index": "device_index"}.get(f.name, f.name))),
                    "code_default": _num(getattr(dflt_cfg, f.name)),
                    "differs_from_saved": False,
                    "differs_from_default": not _same(got, getattr(dflt_cfg, f.name)),
                    "honoured": _same(got, asked),
                })

        # which layers are on, against what the file remembers
        saved_layers = dict(disk.get("layers") or {})
        for la in lab.scene.layers:
            add("layers", la.name, bool(la.visible),
                saved_layers.get(la.name, None), None)

        return {
            "saved_file": str(disk_path) if disk_path else None,
            "saved_file_read_error": disk_error,
            "note": "'effective' is what the capture thread, the tracker and "
                    "the scene are running on right now. 'saved_on_disk' is "
                    "settings.local.json as it stands on disk, which the app "
                    "rewrites on a 1.2s timer, so it can lag. A saved value "
                    "silently beating a code default has cost this project "
                    "two sessions.",
            "differs_from_saved": sorted(
                f"{r['group']}.{r['key']}" for r in rows if r["differs_from_saved"]),
            "differs_from_default": sorted(
                f"{r['group']}.{r['key']}" for r in rows if r["differs_from_default"]),
            "rows": rows,
        }

    # -- /frame ----------------------------------------------------------

    def frame(self, query) -> tuple:
        """The latest camera frame, losslessly.

        The reason this endpoint exists at all: to see what the camera saw I
        pulled frames out of an 815 MB AVI, and then found that the AVI's
        re-encode destroys exactly the near-threshold noise I was chasing --
        so the question could not be answered from the recording at all. PNG
        is lossless and `.npy` is the array itself.
        """
        cap = self._cap()
        det = self._det(cap)
        if cap is None:
            return {"error": "the camera is not open"}, None
        frame = cap._last_frame
        if frame is None:
            return {"error": "no frame has been delivered yet"}, None
        idx = None if det is None else int(det._frame_index)
        img = np.array(frame, copy=True)
        want = (query.get("format") or "png").lower()
        if want == "npy":
            return self._frame_meta(cap, img, idx), _npy(img)
        b, g, r = cv2.split(img)
        chroma = cv2.subtract(r, cv2.max(g, b))
        meta = self._frame_meta(cap, img, idx)
        meta["raw_chroma"] = {
            "max": _num(int(chroma.max())),
            "p99": _num(float(np.percentile(chroma, 99))),
            "p999": _num(float(np.percentile(chroma, 99.9))),
            "mean": _num(float(chroma.mean())),
            "note": "red - max(green, blue) on the RAW frame: the quantity "
                    "the detector starts from, before the gain map, the blur "
                    "and the background subtraction. See /score for all four "
                    "terms as numbers.",
        }
        if want == "json":
            return meta, None
        png = _png(img)
        meta["png_bytes"] = len(png) if png else 0
        return meta, png

    def _frame_meta(self, cap, img, idx) -> dict:
        s = cap._stats
        b, g, r = cv2.split(img)
        return {
            "frame_index": idx,
            "shape": [int(img.shape[0]), int(img.shape[1]),
                      int(img.shape[2]) if img.ndim > 2 else 1],
            "dtype": str(img.dtype),
            "channel_order": "BGR",
            "at_wall": _num(time.time()),
            "fps": _num(s.fps),
            "frames": _num(s.frames),
            "mean": _num(float(img.mean())),
            "max": _num(int(img.max())),
            "channel_means": {"b": _num(float(b.mean())),
                              "g": _num(float(g.mean())),
                              "r": _num(float(r.mean()))},
            "lossless": True,
            "formats": {"png": "/frame", "npy": "/frame?format=npy",
                        "json": "/frame.json"},
        }

    # -- /score ----------------------------------------------------------

    # Bounded on purpose: a neighbourhood is a debugging answer, not a data
    # dump, and eight arrays of 129x129 is already about 1.3 MB of JSON.
    MAX_BOX = 129
    DEFAULT_BOX = 33

    def score(self, query) -> tuple:
        """THE DETECTOR'S ARITHMETIC, AS NUMBERS.

        This is the endpoint the whole server exists for, and a PNG alone
        cannot do its job. The question that was unanswerable on 2026-10-17
        was of the form *"the detector reported peak 9 at this pixel while the
        recorded frame shows raw chroma 1 -- where did 9 come from?"*, and an
        8-bit image of the final score answers none of it: it clips at 255, it
        rounds, and it carries no information about what the value MEANS.

        So the four terms the score is built from are returned as plain
        numbers over a neighbourhood:

            raw         red - max(green, blue) on this frame, before anything
            amplification  1/gain, what the sensitivity map multiplies raw by
            normalised  raw * amplification, saturated at 255 by cv2
            blurred     normalised after the Gaussian, which is what the
                        background model tracks
            background  the running model, float32
            score       blurred - background: THE ARRAY THE DETECTOR
                        THRESHOLDS, read straight off `last_score`

        Subtract to see each term's contribution instead of inferring it. The
        PNG and the `.npy` are both offered; the numbers are the deliverable.
        """
        cap = self._cap()
        det = self._det(cap)
        if cap is None or det is None:
            return {"error": "the camera is not open"}, None

        # ONE CONSISTENT READ. `_last_frame`, `last_score`, `_bg` and `_recip`
        # are all set inside one pass of the capture loop, so they correspond
        # only if the loop did not advance between the reads. Checked, not
        # assumed, and reported either way (RULES C3).
        snap = _snapshot(cap, det)
        if snap is None:
            return {"error": "no frame has been processed yet"}, None
        frame, final, bg, recip, idx, consistent = snap

        p = det.params
        scale = p.scale if p.scale < 0.999 else 1.0
        work = frame
        if scale < 1.0:
            work = cv2.resize(frame, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        raw = det._score(work)                      # pure; allocates fresh
        amp = (np.ones(raw.shape[:2], dtype=np.float32) if recip is None
               else _fit(recip, raw.shape[:2]))
        normalised = cv2.multiply(raw, amp, dtype=cv2.CV_8U)
        blurred = (cv2.GaussianBlur(normalised, (p.blur, p.blur), 0)
                   if p.blur else normalised)
        # THE BACKGROUND THE DETECTOR SUBTRACTED IS NOT THE ONE ON THE OBJECT.
        #
        # `process()` subtracts the model, then updates it with
        # `accumulateWeighted` at the end of the same pass -- so `_bg` read
        # from here is ONE UPDATE AHEAD of the term that produced
        # `last_score`. Measured on a synthetic dot: raw 54 x 3.33 = 180, the
        # detector reported 180, and recomputing it against the model as read
        # gave 176. A four-point mystery in the one endpoint whose whole job
        # is that the arithmetic adds up.
        #
        # The update is an exponential moving average, so it inverts exactly:
        #
        #     bg_after = (1 - rate) * bg_before + rate * score
        #     bg_before = (bg_after - rate * score) / (1 - rate)
        #
        # and `rate` is halved on a frame that held a detection, which is why
        # `last_candidates` is read rather than assumed. Both are reported:
        # `background` is what the model holds now, `background_before_update`
        # is the term the score was actually made with, and the recomputation
        # uses the latter so it agrees to the pixel.
        bg_before = None
        if p.use_background and bg is not None:
            rate = float(p.bg_alpha) * (0.5 if det.last_candidates else 1.0)
            if 0.0 < rate < 1.0:
                bg_before = (bg - rate * blurred.astype(np.float32)) / (1.0 - rate)
            else:                                        # pragma: no cover
                bg_before = bg
        recomputed = (cv2.subtract(blurred, cv2.convertScaleAbs(bg_before))
                      if bg_before is not None else blurred)

        want = (query.get("format") or "json").lower()
        if want == "npy":
            return {"frame_index": idx, "shape": list(final.shape),
                    "dtype": str(final.dtype)}, _npy(final)

        meta = {
            "frame_index": idx,
            "frame_and_score_are_the_same_frame": consistent,
            "at_wall": _num(time.time()),
            "shape": [int(final.shape[0]), int(final.shape[1])],
            "dtype": str(final.dtype),
            "processing_scale": _num(scale),
            "full_frame_shape": [int(frame.shape[0]), int(frame.shape[1])],
            "saturates_at": 255,
            "bars": {
                "threshold": _num(p.threshold),
                "min_peak": _num(p.min_peak),
                "min_area": _num(p.min_area),
                "max_area": _num(p.max_area),
                "amp_area_per_gain": _num(p.amp_area_per_gain),
                "max_elongation": _num(p.max_elongation),
                "note": "threshold makes the mask; min_peak is the bar a blob's "
                        "peak must clear; both are in the units of `score`.",
            },
            "whole_frame": _spread(final, p.threshold, p.min_peak),
            "raw_chroma_whole_frame": _spread(raw, p.threshold, p.min_peak),
            "background": ({"present": False} if bg is None else {
                "present": True, "dtype": str(bg.dtype),
                "mean": _num(float(bg.mean())), "max": _num(float(bg.max())),
                "alpha": _num(p.bg_alpha),
                "rate_this_frame": _num(
                    float(p.bg_alpha) * (0.5 if det.last_candidates else 1.0)),
                "note": "updated on EVERY frame including frames holding a "
                        "detection, so a steady bright object is absorbed in "
                        "seconds (the rate is halved on a frame that held "
                        "one). `background` is the model AFTER this frame's "
                        "update; `background_before_update` in `at.terms` is "
                        "the term the score was actually made with, recovered "
                        "by inverting the moving average. May be mid-update "
                        "when copied.",
            }),
            "gain_map": ({"present": False} if recip is None else {
                "present": True,
                "max_amplification": _num(float(amp.max())),
                "min_amplification": _num(float(amp.min())),
                "note": "amplification is 1/gain, floored by the detector. "
                        "The BLOB is untouched by it, which is why "
                        "amp_area_per_gain asks for area where it amplifies.",
            }),
            "roi": self._roi_measure(final, p),
            "regions": _regions(final, 3, 3),
            "blobs": self._blobs(final, raw, amp, p, scale, frame.shape[:2],
                                 limit=_int(query.get("blobs"), None)),
            "formats": {"json": "/score", "png": "/score.png",
                        "npy": "/score?format=npy",
                        "numbers": "/score?x=&y=&box=33"},
        }
        meta["at"] = self._neighbourhood(
            query, final, raw, amp, normalised, blurred, bg, recomputed, scale,
            frame, bg_before)
        if want == "json":
            return meta, None
        png = _png(final)
        meta["png_bytes"] = len(png) if png else 0
        return meta, png

    def _neighbourhood(self, query, final, raw, amp, normalised, blurred, bg,
                       recomputed, scale, frame, bg_before=None) -> dict:
        """Every term of the arithmetic, as numbers, around one point."""
        box = max(3, min(self.MAX_BOX, _int(query.get("box"), self.DEFAULT_BOX)))
        fx = _float(query.get("x"), None)
        fy = _float(query.get("y"), None)
        if fx is None or fy is None:
            # The brightest pixel of the score is almost always the question.
            _mn, _mx, _ml, loc = cv2.minMaxLoc(final)
            wx, wy = int(loc[0]), int(loc[1])
            chosen = "the brightest pixel of the score map"
        else:
            wx, wy = int(round(fx * scale)), int(round(fy * scale))
            chosen = "the point asked for"
        h, w = final.shape[:2]
        half = box // 2
        x0, y0 = max(0, wx - half), max(0, wy - half)
        x1, y1 = min(w, wx + half + 1), min(h, wy + half + 1)

        def cut(a, as_int=True):
            # `tolist()` rather than a Python loop: the conversion holds
            # the GIL, and every millisecond it holds is a millisecond the
            # detector may wait for -- Python's switch interval is 5 ms and
            # the capture loop has 14 ms of detect time before it drops from
            # 60 fps to 30. MEASURED on this machine, the seven 33x33 grids
            # one request returns: 0.75 ms by Python loop, 0.04 ms by
            # tolist.
            if a is None:
                return None
            sub = a[y0:y1, x0:x1]
            if as_int:
                return np.asarray(sub, dtype=np.int32).tolist()
            return np.round(np.asarray(sub, dtype=np.float32), 3).tolist()

        return {
            "centre_working_px": [wx, wy],
            "centre_full_frame_px": [_num(wx / scale), _num(wy / scale)],
            "chosen": chosen,
            "box": box,
            "window_working_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "row_order": "top to bottom, left to right, from window_working_px",
            "terms": {
                "raw_chroma": cut(raw),
                "amplification": cut(amp, as_int=False),
                "normalised": cut(normalised),
                "blurred": cut(blurred),
                "background_before_update": cut(bg_before, as_int=False),
                "background_after_update": cut(bg, as_int=False),
                "score_from_the_detector": cut(final),
                "score_recomputed_here": cut(recomputed),
            },
            "at_the_centre": {
                "raw_chroma": _num(int(raw[wy, wx])),
                "amplification": _num(float(amp[wy, wx])),
                "normalised": _num(int(normalised[wy, wx])),
                "blurred": _num(int(blurred[wy, wx])),
                "background_before_update": (
                    None if bg_before is None else _num(float(bg_before[wy, wx]))),
                "background_after_update": (
                    None if bg is None else _num(float(bg[wy, wx]))),
                "score_from_the_detector": _num(int(final[wy, wx])),
                "score_recomputed_here": _num(int(recomputed[wy, wx])),
                "frame_bgr": [int(v) for v in
                              np.asarray(frame)[min(int(wy / scale),
                                                    frame.shape[0] - 1),
                                                min(int(wx / scale),
                                                    frame.shape[1] - 1)]],
            },
            "how_to_read_it": "score = blur(raw * amplification) - "
                              "background_before_update, each step saturating "
                              "at 255. The detector subtracts the model and "
                              "THEN updates it, so background_after_update -- "
                              "what the object holds now -- is one step ahead "
                              "and is not the term that made this score. If "
                              "score_recomputed_here still disagrees with "
                              "score_from_the_detector, the frame advanced "
                              "between the two reads: check "
                              "frame_and_score_are_the_same_frame.",
        }

    def _roi_measure(self, final, p) -> dict:
        poly = p.roi_polygon
        if not poly or len(poly) < 3:
            return {"polygon": None,
                    "note": "no screen mask: every pixel is in play. That is "
                            "the honest state before calibration."}
        scale = p.scale if p.scale < 0.999 else 1.0
        pts = (np.array(poly, dtype=np.float32) * scale).astype(np.int32)
        m = np.zeros(final.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(m, pts, 255)
        inside = int((m > 0).sum())
        outside = cv2.bitwise_not(m)
        return {
            "polygon": [[_num(x), _num(y)] for x, y in poly],
            "margin_px": _num(p.roi_margin),
            "coverage_pct": _num(100.0 * inside / float(m.size)),
            "max_inside": _num(int(cv2.minMaxLoc(final, mask=m)[1])),
            "max_outside": _num(int(cv2.minMaxLoc(final, mask=outside)[1])),
            "note": "without a mask, 1040 of 1356 detections once came from a "
                    "bright window at the edge of frame -- 77% of everything.",
        }

    # Twelve is enough to see the dot, the noise around it and anything the
    # size gate refused. More is available with `?blobs=N`, because a frame
    # with forty candidates is itself the finding -- but it is not what a
    # routine query should pay for while he is shooting.
    DEFAULT_BLOBS = 12
    MAX_BLOBS_REPORTED = 60

    def _blobs(self, final, raw, amp, p, scale, full_shape, limit=None) -> dict:
        """EVERY blob above the threshold, before the detector's filters, with
        the reason each one would be kept or dropped.

        Raw, because the detector only ever reports what survived. "The camera
        saw nothing" and "the camera saw it and threw it away" are the same
        event from the firing line and need opposite fixes."""
        from lsa.detect.laser import amplified_min_area
        _t, mask = cv2.threshold(final, p.threshold, 255, cv2.THRESH_BINARY)
        if p.close_iter:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k,
                                    iterations=p.close_iter)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        poly = p.roi_polygon
        path = (np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
                if poly and len(poly) >= 3 else None)
        rows = []
        order = sorted(range(1, n), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
        want = self.DEFAULT_BLOBS if limit is None else max(
            1, min(self.MAX_BLOBS_REPORTED, int(limit)))
        for i in order[:want]:
            area_w = int(stats[i, cv2.CC_STAT_AREA])
            x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH]); bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            sub = final[y:y + bh, x:x + bw]
            sublab = labels[y:y + bh, x:x + bw] == i
            peak = int(sub[sublab].max()) if sublab.any() else 0
            cx, cy = float(cents[i][0]), float(cents[i][1])
            fcx, fcy = cx / scale, cy / scale
            n_local = float(amp[min(int(cy), amp.shape[0] - 1),
                                min(int(cx), amp.shape[1] - 1)])
            need = amplified_min_area(p, n_local)
            on_screen = (True if path is None else
                         bool(cv2.pointPolygonTest(path, (fcx, fcy), True)
                              >= p.roi_margin))
            long_side, short_side = max(bw, bh), max(1, min(bw, bh))
            elong = long_side / float(short_side)
            drops = []
            if peak < p.min_peak:
                drops.append(f"too faint: peak {peak} < min_peak {p.min_peak}")
            if area_w < p.min_area:
                drops.append(f"too small: {area_w} < min_area {p.min_area}")
            if area_w > p.max_area and not (
                    p.oversize_min_peak > 0
                    and peak >= max(p.min_peak, p.oversize_min_peak)):
                # Over the ceiling is accepted when bright enough to be a
                # pulse (DetectorParams.oversize_min_peak): only a faint one
                # is dropped for size.
                drops.append(f"too big and too faint: {area_w} > max_area "
                             f"{p.max_area}, peak {peak} < {p.oversize_min_peak}")
            if area_w < need:
                drops.append(f"size gate: {area_w} < {need:.0f} required at "
                             f"amplification {n_local:.2f}")
            if not on_screen:
                drops.append("off the screen mask")
            if p.max_elongation and elong > p.max_elongation:
                drops.append(f"too elongated: {elong:.2f} > {p.max_elongation}")
            rows.append({
                "centroid_working_px": [_num(cx), _num(cy)],
                "centroid_full_frame_px": [_num(fcx), _num(fcy)],
                "bbox_working_px": [x, y, bw, bh],
                "area_working_px": area_w,
                "area_full_frame_px": _num(area_w / (scale * scale)),
                "peak": peak,
                "raw_chroma_peak": int(raw[y:y + bh, x:x + bw][sublab].max())
                if sublab.any() else 0,
                "local_amplification": _num(n_local),
                "area_required_by_the_size_gate": _num(need),
                "elongation": _num(elong),
                "inside_screen_mask": on_screen,
                "would_be_dropped_because": drops,
                "would_be_reported": not drops,
            })
        return {
            "count_above_threshold": max(0, n - 1),
            "reported": len(rows),
            "max_blobs_the_detector_keeps": _num(p.max_blobs),
            "min_separation_px": _num(p.min_separation),
            "note": "this is the list BEFORE the detector's filters, with the "
                    "reason each blob would be dropped. The detector reports "
                    "only survivors, so this is the only place a near miss is "
                    "visible as a measurement rather than as a counter.",
            "rows": rows,
        }

    # -- /shots ----------------------------------------------------------

    def shots(self, n: int) -> dict:
        tap = self._tap
        if tap is None:
            return {"error": "no shot tap is installed"}
        cap = self._cap()
        det = self._det(cap)
        bar = float(det.params.min_peak) if det is not None else 0.0
        mm_cam = _call(self._lab, "_mm_per_camera_px")
        mm_cam = mm_cam if isinstance(mm_cam, (int, float)) else None
        mapping = self._lab.mapping
        rows = []
        now = time.perf_counter()
        for at, shot in tap.recent(max(1, min(200, n))):
            screen = None
            if mapping is not None:
                try:
                    sx, sy = mapping.to_screen(shot.x, shot.y)
                    screen = [_num(sx), _num(sy)]
                except Exception:                        # pragma: no cover
                    screen = None
            members = [{
                "frame_index": _num(d.frame_index),
                "t_offset_ms": _num((d.timestamp - shot.first_timestamp) * 1000.0),
                "x": _num(d.x), "y": _num(d.y),
                "peak": _num(d.peak), "area": _num(d.area),
                "width": _num(d.width), "height": _num(d.height),
                "elongation": _num(d.elongation),
            } for d in (shot.members or [])]
            evidence = None
            if cap is not None:
                try:
                    evidence = cap.evidence_between(shot.first_timestamp - 0.25,
                                                    shot.last_timestamp + 0.25)
                except Exception:                        # pragma: no cover
                    evidence = None
            row = {
                "index": _num(shot.index),
                "track_id": _num(shot.track_id),
                "age_s": _num(now - at),
                "camera": [_num(shot.x), _num(shot.y)],
                "camera_normalised": [_num(shot.nx), _num(shot.ny)],
                "screen": screen,
                "frames": _num(shot.frames),
                "peak": _num(shot.peak),
                "area": _num(shot.area),
                "elongation": _num(shot.elongation),
                "duration_ms": _num(shot.duration_ms),
                "provisional": bool(shot.provisional),
                "peak_over_bar": _num(shot.peak / bar) if bar > 0 else None,
                # PER-FRAME, which is the whole point: the reported position is
                # the FIRST frame, not the brightest and not the average, and
                # a hole that disagrees with the dot is diagnosed from the
                # drift between them.
                "members": members,
                "near_miss_evidence_around_it": _jsonable(evidence),
            }
            try:
                from lsa import shotlog
                row["evidence_line"] = shotlog.shot_evidence(shot, bar, mm_cam)
            except Exception:                            # pragma: no cover
                pass
            rows.append(row)
        return {
            "shots_seen_this_session": int(tap.seen),
            "kept": len(rows),
            "ring_capacity": int(type(tap).KEEP),
            "bar_min_peak": _num(bar),
            "mm_per_camera_px": _num(mm_cam),
            "note": "newest last. The reported position is the FIRST frame of "
                    "the pulse: a round leaves at trigger break while the "
                    "laser stays lit for ~100ms of follow-through, so the "
                    "pulse average reports a place the bullet was never at.",
            "rows": rows,
        }

    # -- /drawn ----------------------------------------------------------

    def drawn(self, query) -> tuple:
        """WHAT THE APP HAS PAINTED, AND WHERE, MEASURED.

        Not "what the layers say they draw" -- what is actually lit, grabbed
        from the target window and measured per region: its red excess against
        the limits in RULES B1 (30 static, 8 for anything that moves or
        appears), its luminance against the area budget in B1b (no more than
        2% of the screen above luminance 60 while shooting is possible), and
        every region's box mapped into CAMERA pixels through the current
        mapping.

        This is the measurement that identified the phantom detections at
        camera (558,525) as the app's OWN signal meter, after I had diagnosed
        them as a static object in his room and was about to build a
        position-identity rule that would have rejected his real groups. The
        app knows exactly what it paints; that is the question to ask of a
        repeating detection.
        """
        lab = self._lab
        win = lab.target_window
        if win is None:
            return False, {
                "error": "there is no target window",
                "detail": "nothing is being painted on the shooter's screen, "
                          "so there is nothing to measure. Before one has "
                          "ever opened this is the operator-console flow; "
                          "after one has, it means the screen was CLOSED and "
                          "the camera is watching the desktop.",
                "target_window_ever_opened": bool(lab._target_ever_opened),
                "layers": self._layers(),
            }
        if self._bridge is None:
            return False, {"error": "no Qt bridge: cannot grab the screen"}
        width = max(120, min(960, _int(query.get("width"), 480)))
        try:
            grabbed = self._bridge.run(lambda: _grab(win, width))
        except Exception as exc:                         # noqa: BLE001
            return False, {
                "error": "could not grab the target screen",
                "detail": repr(exc),
                "note": "the Qt thread did not take the work inside its "
                        "deadline, or the window had no pixels. Nothing was "
                        "changed and no key was sent.",
            }
        if grabbed is None:
            return False, {"error": "the target window had nothing to grab"}
        bgr, full_w, full_h = grabbed
        out = self._measure_ink(bgr, full_w, full_h, query)
        out["clickable"] = self._clickable()
        return True, out

    def _clickable(self) -> dict:
        """WHERE THE THINGS YOU CAN CLICK OR SHOOT ACTUALLY ARE, in screen px.

        Added 2026-10-18, when he asked for the menu to be driven and verified
        with nobody at the machine. Without this an agent has to RECOMPUTE the
        tile layout to know where to click -- which tests its own arithmetic
        against itself and would pass while the app drew the grid somewhere
        else entirely. The app knows where it put them; it should say so.

        Reported for any mode that lays out `slots`, not for the menu by name,
        so a future screen of buttons is covered without anyone remembering
        this. Measurements, not conclusions: the rectangle as drawn, and the
        point at the middle of its picture, which is what a shot or a click
        should be aimed at.
        """
        host = self._lab.layers.get("modes") if self._lab.layers else None
        mode = host.mode if host is not None else None
        win = self._lab.target_window
        # `hasattr(mode, "slots")` is a CAPABILITY probe, deliberate: only a
        # screen of tiles lays out clickable regions, and which modes do is
        # open-ended.
        if mode is None or win is None or not hasattr(mode, "slots"):
            return {"available": False,
                    "why": "the active mode lays out no clickable regions"}
        try:
            ctx = self._bridge.run(
                lambda: self._lab.scene._context(time.perf_counter(), 0.0,
                                                 win.rect()))
            slots = self._bridge.run(lambda: list(mode.slots(ctx)))
        except Exception as exc:                         # noqa: BLE001
            return {"available": False, "why": repr(exc)}
        out = []
        for s in slots:
            r, pic = s.rect, s.picture
            cls = s.cls
            out.append({
                "name": (cls.name if cls is not None else None) or s.stem,
                "title": s.title,
                "key": s.glyph or None,
                "starts_a_drill": cls is not None,
                "rect": [round(r.left(), 1), round(r.top(), 1),
                         round(r.width(), 1), round(r.height(), 1)],
                "aim_at": [round(pic.center().x(), 1),
                           round(pic.center().y(), 1)],
            })
        return {"available": True, "mode": mode.name, "regions": out,
                "note": "aim_at is the middle of the tile's picture. POST "
                        "/click?x=&y= there, or put a shot there."}

    # Lit at all. Matches `_report_screen_ink`, deliberately: two thresholds
    # for one question is how a check comes to mean nothing.
    INK_LUMA = 12
    # The area budget's threshold (RULES B1b).
    BRIGHT_LUMA = 60

    def _measure_ink(self, bgr, full_w, full_h, query) -> dict:
        h, w = bgr.shape[:2]
        sx, sy = full_w / float(w), full_h / float(h)
        b, g, r = cv2.split(bgr)
        luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_excess = cv2.subtract(r, cv2.max(g, b))
        lit = (luma > self.INK_LUMA).astype(np.uint8) * 255
        bright = (luma > self.BRIGHT_LUMA)
        total = float(luma.size)
        mapping = self._lab.mapping

        n, labels, stats, cents = cv2.connectedComponentsWithStats(lit, 8)
        regions = []
        order = sorted(range(1, n), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
        for i in order[:30]:
            x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH]); bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            m = (labels[y:y + bh, x:x + bw] == i)
            re_sub = red_excess[y:y + bh, x:x + bw][m]
            lu_sub = luma[y:y + bh, x:x + bw][m]
            box_screen = [_num(x * sx), _num(y * sy), _num(bw * sx), _num(bh * sy)]
            regions.append({
                "screen_box": box_screen,
                "screen_centre": [_num((x + bw / 2) * sx), _num((y + bh / 2) * sy)],
                "camera_box_corners": _corners_in_camera(mapping, x * sx, y * sy,
                                                         bw * sx, bh * sy),
                "lit_px_sampled": int(m.sum()),
                "max_luminance": _num(int(lu_sub.max())) if lu_sub.size else 0,
                "mean_luminance": _num(float(lu_sub.mean())) if lu_sub.size else 0,
                "max_red_excess": _num(int(re_sub.max())) if re_sub.size else 0,
                "mean_red_excess": _num(float(re_sub.mean())) if re_sub.size else 0,
            })

        worst = int(red_excess.max()) if red_excess.size else 0
        wy, wx = (np.unravel_index(int(red_excess.argmax()), red_excess.shape)
                  if red_excess.size else (0, 0))
        return {
            "grab": {"sampled": [int(w), int(h)],
                     "screen": [int(full_w), int(full_h)],
                     "at_wall": _num(time.time())},
            "layers": self._layers(),
            "mode": self._mode(),
            "lit_pct": _num(100.0 * float((luma > self.INK_LUMA).sum()) / total),
            "lit_pct_above_luminance_60": _num(100.0 * float(bright.sum()) / total),
            "area_budget": {
                "limit_pct": 2.0, "above_luminance": self.BRIGHT_LUMA,
                "within": bool(100.0 * float(bright.sum()) / total <= 2.0),
                "note": "RULES B1b. The exposure strategy is 'the screen "
                        "fades to a ghost, the 650nm dot still saturates'. A "
                        "lit panel is not a ghost, and a shot has to "
                        "out-shine whatever is drawn.",
            },
            "red_excess": {
                "max": worst,
                "max_at_screen": [_num(float(wx) * sx), _num(float(wy) * sy)],
                "max_at_camera": _point_in_camera(mapping, float(wx) * sx,
                                                  float(wy) * sy),
                "limit_static": 30,
                "limit_moving_or_appearing": 8,
                "note": "the detector scores red - max(green, blue). 8 is the "
                        "limit for anything that moves or appears, and even 8 "
                        "has fired: his saved sensitivity map amplifies the "
                        "weakest region 4x, so 8 arrives at the detector as "
                        "16-18 against a bar of 14. Draw at 0.",
            },
            "regions": regions,
            "mapping_present": bool(mapping is not None),
            "note": "boxes are in SCREEN pixels, scaled up from the sampled "
                    "grab, and each one is also given as four CAMERA-pixel "
                    "corners through the current mapping -- so a repeating "
                    "detection can be matched against what the app painted "
                    "there.",
        }

    # -- POST /key -------------------------------------------------------

    def press_key(self, name: str) -> tuple:
        """One key to the target window, WITHOUT taking the keyboard off anyone.

        Pressing a key used to mean activating the window and sending OS-level
        input. Windows routinely refuses foreground activation to a process
        that is not already in front, so the activation half failed silently,
        the keystroke went wherever the focus actually was, and on one
        occasion it knocked the target window behind Settings -- which then
        tripped the screen-ownership guard and cost a run.

        This posts the event to the window object, so focus is irrelevant and
        nothing moves on screen. It then travels `TargetWindow.keyPressEvent`,
        which is the real path a human keypress takes: layers get first
        refusal, F11 and Escape keep their meanings, and anything the app
        binds reaches `_on_target_key`. Nothing here reaches around that, so a
        key that starts a routine starts it through the one run slot
        (constraint 2) and is refused by the Director if something else holds
        it -- exactly as it would be for him.
        """
        from .gui import post_key, resolve_key
        lab = self._lab
        key = resolve_key(name or "")
        if key is None:
            return False, {
                "error": "unknown key",
                "asked_for": name,
                "accepted": "a single character (a-z, 0-9, [ ] + - = , . /) "
                            "or a Qt name: Escape, Space, Backspace, Return, "
                            "Tab, Delete, Left, Right, Up, Down, F1..F12",
            }
        win = lab.target_window
        if win is None:
            return False, {
                "error": "there is no target window to send a key to",
                "detail": "the app's keys belong to the shooter's screen. "
                          "Open it with F11 on the console, or launch with "
                          "--target.",
            }
        if self._bridge is None:
            return False, {"error": "no Qt bridge: cannot post an event"}
        before = {"mode": self._mode().get("active"),
                  "visible_layers": [la["name"] for la in self._layers()
                                     if la["visible"]],
                  "director_running": self._director().get("running")}
        try:
            self._bridge.run(lambda: post_key(win, key, _text_for(key)))
        except Exception as exc:                         # noqa: BLE001
            return False, {"error": "could not post the key",
                           "detail": repr(exc),
                           "note": "the Qt thread did not take it inside its "
                                   "deadline. Nothing was sent."}
        # LOGGED, because a key that arrives from outside must be
        # attributable. This project has lost time to a log line that named
        # the wrong cause; an unexplained mode change would be worse.
        self._say(f"introspection: key {name!r} (Qt {key}) posted to the "
                  f"target screen, no focus taken")
        return True, {
            "sent": name, "qt_key": int(key),
            "delivered_to": "the target window's keyPressEvent, the same path "
                            "a human keypress takes",
            "focus_taken": False,
            "before": before,
            "note": "the effect is ASYNCHRONOUS: the event is queued on the "
                    "Qt thread. Re-read /state to see what it did. A key that "
                    "starts a routine is refused by the Director if something "
                    "else holds the run slot, which /state reports under "
                    "director.refusals.",
        }

    def _say(self, msg: str) -> None:
        """Log from a SERVER thread without touching Qt off the GUI thread.

        THIS KILLED THE APP, and the fault dump named it in one line:

            camera_lab.py:10063 in _log      <- appendPlainText, a QWidget
            view.py:1429 in click
            server.py:219 in _serve          <- the HTTP thread

        `CameraLab._log` writes to the log pane, so calling it from a request
        thread touches a widget from the wrong thread: undefined behaviour
        that is usually fine and occasionally an access violation. It took
        one click in a run of forty to fall over -- which is the worst shape
        a fault can have, because the obvious conclusion is "the menu is
        flaky" rather than "the instrument is unsafe".

        The bridge already exists to run work on the GUI thread, and every
        other GUI touch here goes through it. The logging did not, in BOTH
        actions, from the day `POST /key` was written.
        """
        if self._log is None:
            return
        if self._bridge is None:                         # pragma: no cover
            self._log(msg)
            return
        try:
            self._bridge.run(lambda: self._log(msg))
        except Exception:                                # noqa: BLE001
            # A dropped log line is a cost; a crashed app is a session. The
            # action itself has already happened either way.
            pass

    def click(self, x, y) -> "tuple[bool, dict]":
        """Deliver one left click to the target screen at (x, y). No focus taken.

        The pointer half of `key`, added 2026-10-18 when he asked for the menu
        to work with the mouse as well as the gun and then left the machine --
        so the whole menu had to be driven and verified with nobody at the PC.

        It takes the same path a human click takes (`mousePressEvent` ->
        `Scene.on_click`, topmost visible layer first) and it is NEVER turned
        into a shot: no pulse, no peak, no camera position, nothing in the
        score or the hit rate. A tool that could manufacture hits would make
        every rate this app reports worthless.
        """
        from .gui import post_click
        lab = self._lab
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            return False, {"error": "x and y must be numbers",
                           "asked_for": {"x": x, "y": y}}
        win = lab.target_window
        if win is None:
            return False, {
                "error": "there is no target window to click on",
                "detail": "the shooter's screen is where the menu is. Open it "
                          "with F11 on the console, or launch with --target.",
            }
        rect = win.rect()
        if not (0 <= fx < rect.width() and 0 <= fy < rect.height()):
            return False, {
                "error": "that point is outside the target screen",
                "asked_for": {"x": fx, "y": fy},
                "screen": {"width": rect.width(), "height": rect.height()},
            }
        if self._bridge is None:
            return False, {"error": "no Qt bridge: cannot post an event"}
        before = {"mode": self._mode().get("active")}
        try:
            self._bridge.run(lambda: post_click(win, fx, fy))
        except Exception as exc:                         # noqa: BLE001
            return False, {"error": "could not post the click",
                           "detail": repr(exc),
                           "note": "the Qt thread did not take it inside its "
                                   "deadline. Nothing was sent."}
        self._say(f"introspection: click at ({fx:.0f},{fy:.0f}) posted to "
                  f"the target screen, no focus taken")
        return True, {
            "clicked": {"x": fx, "y": fy},
            "delivered_to": "the target window's mousePressEvent, the same "
                            "path a human click takes",
            "focus_taken": False,
            "is_a_shot": False,
            "before": before,
            "note": "the effect is ASYNCHRONOUS: the event is queued on the "
                    "Qt thread. Re-read /state to see what it did.",
        }

    def _app_says(self) -> dict:
        """The app's own conclusions. A datum to cross-check, never the truth."""
        lab = self._lab
        report = lab._last_report
        out = {
            "note": "these are the APP'S conclusions, exposed only so they "
                    "can be cross-checked against the measurements above, and "
                    "never as the source of truth. On 2026-10-17 the "
                    "app reported '100% clear the bar, margin 6.1x' while its "
                    "own test dot measured below its own threshold.",
            "ready_for_real_shots": bool(_call(lab, "_ready_for_real_shots")),
            "app_is_drawing_now": bool(_call(lab, "_app_is_drawing_now")),
            "prompt": list(_call(lab, "_prompt_state") or ()),
        }
        if report is not None:
            out["verdict"] = str(report.verdict.name).lower()
            out["summary"] = str(report.summary or "")
            out["findings"] = [
                {"verdict": str(f.verdict.name).lower(),
                 "headline": str(f.headline or ""),
                 "detail": str(f.detail or ""),
                 "action": str(f.action or "")}
                for f in report.findings]
        return out


# --- measurement helpers ---------------------------------------------------


def _same(a, b) -> bool:
    """Equal for the purposes of "does the effective value differ"."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return str(a) == str(b)


def _int(raw, default):
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float(raw, default):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _settings_path():
    """Where `settings.local.json` is, honouring the test isolation env var.

    Read from the environment rather than imported from `camera_lab`, which
    would be a circular import and would also drag Qt into a module that does
    not need it.
    """
    import os
    raw = os.environ.get("LSA_SETTINGS")
    if raw:
        return Path(raw)
    return assets.user_dir("config") / "settings.local.json"


def _png(arr) -> "bytes | None":
    """Lossless. PNG, never JPEG: the whole reason /frame exists is that the
    recording's re-encode destroyed the near-threshold noise being chased."""
    try:
        ok, buf = cv2.imencode(".png", arr)
        return bytes(buf) if ok else None
    except Exception:                                    # pragma: no cover
        return None


def _npy(arr) -> "bytes | None":
    """The array itself, exact dtype and shape. No interpretation at all."""
    import io as _io
    try:
        bio = _io.BytesIO()
        np.save(bio, np.ascontiguousarray(arr), allow_pickle=False)
        return bio.getvalue()
    except Exception:                                    # pragma: no cover
        return None


def _snapshot(cap, det):
    """One consistent read of the frame and everything derived from it.

    `_last_frame`, `last_score`, `_bg` and `_recip` are assigned in one pass of
    the capture loop. Reading them from another thread can straddle two passes,
    which would make the arithmetic in /score not add up for a reason that has
    nothing to do with the detector. So the frame index is read either side and
    the answer SAYS whether they correspond, rather than hoping.
    """
    for _attempt in (0, 1):
        idx0 = int(det._frame_index)
        frame = cap._last_frame
        final = det.last_score
        bg = det._bg
        recip = det._recip
        if frame is None or final is None:
            return None
        frame_c = np.array(frame, copy=True)
        final_c = np.array(final, copy=True)
        bg_c = None if bg is None else np.array(bg, copy=True)
        recip_c = None if recip is None else np.array(recip, copy=True)
        idx1 = int(det._frame_index)
        if idx0 == idx1:
            return frame_c, final_c, bg_c, recip_c, idx0, True
    return frame_c, final_c, bg_c, recip_c, idx1, False


def _fit(arr, shape):
    """Resize a float map to the working shape, the way the detector does."""
    if arr.shape[:2] == tuple(shape):
        return arr
    return cv2.resize(arr, (int(shape[1]), int(shape[0])),
                      interpolation=cv2.INTER_LINEAR)


def _hist_stats(hist, total: int) -> dict:
    """max, mean and two percentiles of a uint8 image, all from its histogram.

    MEASURED, and the reason this exists rather than `a.max()`, `a.mean()` and
    `np.percentile`: those are 56 separate reductions over a 1280x720 array
    per /score request, and `np.percentile` SORTS. One request measured 73 ms
    that way, all of it holding the GIL -- and Python's switch interval is
    5 ms while the capture loop has 14 ms of detect time before it drops from
    60 fps to 30 and halves the camera frames inside a 100 ms pulse. A slow
    endpoint is not merely slow.

    `np.bincount` is a single O(n) C pass and every statistic above is then
    EXACT for 8-bit data, not an approximation: the histogram holds the whole
    distribution.
    """
    if total <= 0:
        return {"max": 0.0, "mean": 0.0, "p99": 0.0, "p999": 0.0}
    levels = np.arange(hist.size, dtype=np.float64)
    cum = np.cumsum(hist)
    nz = np.nonzero(hist)[0]
    return {
        "max": float(nz[-1]) if nz.size else 0.0,
        "mean": float((levels * hist).sum() / total),
        "p99": float(np.searchsorted(cum, 0.99 * total, side="left")),
        "p999": float(np.searchsorted(cum, 0.999 * total, side="left")),
        "cum": cum,
    }


def _spread(arr, threshold, min_peak) -> dict:
    """What the distribution of a score image looks like, against its bars."""
    a = np.asarray(arr)
    if a.size == 0:
        return {}
    total = int(a.size)
    if a.dtype == np.uint8:
        hist = np.bincount(np.ascontiguousarray(a).reshape(-1), minlength=256)
        st = _hist_stats(hist, total)
        cum = st["cum"]
        t = max(0, min(255, int(threshold)))
        m = max(0, min(255, int(min_peak)))
        above_t = total if t <= 0 else total - int(cum[t - 1])
        above_p = total if m <= 0 else total - int(cum[m - 1])
        mx, mean, p99, p999 = st["max"], st["mean"], st["p99"], st["p999"]
    else:
        hist = None
        flat = a.reshape(-1)
        mx, mean = float(a.max()), float(a.mean())
        p99 = float(np.percentile(flat, 99))
        p999 = float(np.percentile(flat, 99.9))
        above_t = int((a >= threshold).sum())
        above_p = int((a >= min_peak).sum())
    out = {
        "max": _num(mx),
        "mean": _num(mean),
        "p99": _num(p99),
        "p999": _num(p999),
        "px_at_or_above_threshold": int(above_t),
        "px_at_or_above_min_peak": int(above_p),
        "px_total": total,
    }
    if hist is not None:
        # Coarse, 16 buckets of 16: enough to see a noise floor sitting at 1-3
        # with a dot out at 180, which is the shape of a healthy frame.
        out["histogram_16"] = [int(hist[i * 16:(i + 1) * 16].sum())
                               for i in range(16)]
        out["histogram_note"] = "16 buckets of 16 score levels, 0-15 first"
    return out


def _regions(arr, cols: int, rows: int) -> dict:
    """A grid of maxima. The one number this project keeps needing is "is the
    far side of the screen dimmer", and a whole-frame maximum cannot say."""
    a = np.asarray(arr)
    h, w = a.shape[:2]
    cells = []
    for ry in range(rows):
        for rx in range(cols):
            y0, y1 = h * ry // rows, h * (ry + 1) // rows
            x0, x1 = w * rx // cols, w * (rx + 1) // cols
            sub = a[y0:y1, x0:x1]
            if not sub.size:
                cells.append({"col": rx, "row": ry, "max": None})
                continue
            if sub.dtype == np.uint8:
                st = _hist_stats(
                    np.bincount(np.ascontiguousarray(sub).reshape(-1),
                                minlength=256), int(sub.size))
                mx, mean, p99 = st["max"], st["mean"], st["p99"]
            else:
                mx, mean = float(sub.max()), float(sub.mean())
                p99 = float(np.percentile(sub, 99))
            cells.append({
                "col": rx, "row": ry,
                "box_working_px": [x0, y0, x1 - x0, y1 - y0],
                "max": _num(mx),
                "p99": _num(p99),
                "mean": _num(mean),
            })
    return {"cols": cols, "rows": rows, "cells": cells}


def _grab(win, width: int):
    """Grab the target window as BGR. MUST run on the Qt thread.

    Taken from the window rather than re-rendered, so it is literally the
    image the camera is looking at: a re-render would differ wherever a layer
    sizes in absolute pixels, and those differences would read as a fault.
    """
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import QImage as _QImage
    pm = win.grab()
    if pm.isNull():
        return None
    full_w, full_h = pm.width(), pm.height()
    img = (pm.toImage().scaledToWidth(int(width), _Qt.SmoothTransformation)
           .convertToFormat(_QImage.Format_RGBA8888))
    h, w = img.height(), img.width()
    if h <= 0 or w <= 0:
        return None
    arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=h * w * 4)
    rgba = arr.reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR).copy(), full_w, full_h


def _point_in_camera(mapping, sx: float, sy: float):
    if mapping is None:
        return None
    try:
        pt = mapping.to_camera(float(sx), float(sy))
    except Exception:                                    # pragma: no cover
        return None
    if pt is None:
        return None
    return [_num(pt[0]), _num(pt[1])]


def _corners_in_camera(mapping, x: float, y: float, w: float, h: float):
    if mapping is None:
        return None
    out = []
    for cx, cy in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
        pt = _point_in_camera(mapping, cx, cy)
        if pt is None:
            return None
        out.append(pt)
    return out


def _jsonable(obj):
    """Numpy scalars and tuples out of the evidence ring, into plain JSON."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return _num(obj)


def _text_for(key: int) -> str:
    """The `text` a real QKeyEvent would carry, so a layer reading `ev.text()`
    sees what it would from a keyboard."""
    if 0x20 <= key <= 0x7E:
        return chr(key).lower()
    return ""

# --- small helpers ---------------------------------------------------------


def _dc(obj) -> dict:
    if obj is None or not is_dataclass(obj):
        return {}
    return {f.name: _num(getattr(obj, f.name)) for f in fields(obj)}


def _call(obj, name: str):
    """Call a zero-argument method (or read a property), and never let the
    server die of what the METHOD raises. A name that does not exist RAISES
    AttributeError: this used to take a `default` and return it for a missing
    name, under a docstring that claimed the opposite, which is how
    `stats.last_shot_at` returned 0.0 for days while a guard never fired
    (RULES C8)."""
    fn = getattr(obj, name)
    try:
        return fn() if callable(fn) else fn
    except Exception as exc:                             # noqa: BLE001
        return f"<{name} raised {exc!r}>"


def _key_name(key: int) -> str:
    if 0x30 <= key <= 0x39 or 0x41 <= key <= 0x5A:
        return chr(key)
    return str(key)


_VERSION_CACHE: "list[str]" = []


def _version() -> str:
    """The git commit this is running, read once. A measurement of WHICH code
    answered, so a report can never be attributed to the wrong build."""
    if _VERSION_CACHE:
        return _VERSION_CACHE[0]
    root = assets.bundle_root()      # a frozen bundle has no .git: "unknown", honestly
    out = "unknown"
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = (root / ".git" / head[5:]).read_text(encoding="utf-8").strip()
            out = ref[:12]
        else:
            out = head[:12]
    except Exception:                                    # pragma: no cover
        out = "unknown"
    _VERSION_CACHE.append(out)
    return out


def _pid() -> int:
    import os
    return os.getpid()
