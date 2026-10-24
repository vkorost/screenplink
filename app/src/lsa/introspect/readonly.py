"""A read-only window onto the capture thread, enforced by the object, not by care.

RULES A4, and the first constraint of the agreed design: **the introspection
server may read camera state and may never write one.** A single
`CAP_PROP_EXPOSURE` write has hung this camera for 272 seconds, while five
minutes of pure streaming returned 17,945 frames with zero failures. Streaming
is safe; control traffic is not.

"The server does not write controls" is a convention, and this project has
measured what conventions are worth: `SCREEN_OWNERS` was added to end the
"each routine guards against the ones its author thought of" bug and ended up
consulted by one of the six routines that needed it (RULES E2). So the server
is not handed the `CaptureThread`. It is handed a `Reader`, which does not
HAVE `set_control`, `apply_preset`, `start_exposure_search`, `request_probe`,
`request_driver_dialog` or `_post`, and raises `WriteAttempted` if anyone asks
for one. A handler cannot reach a write by remembering not to.

The same wrapper is used for the detector and the tracker. Those writes are not
camera hazards, but `set_params`, `set_gain_map` and `reset` would change what
the detector does while an agent is looking at it -- an instrument that alters
what it measures (RULES, the exposure probe that wedged the camera it was
measuring).

Two checks, because a structural guard alone would pass a handler that got hold
of the real object some other way:

  * `tests/test_introspect.py` section 2 asserts every write name on the real
    `CaptureThread` is refused by the `Reader` built from it.
  * the same file drives every endpoint against a capture stand-in that RAISES
    on any write-shaped access, and asserts `Stats.control_writes` did not move.
"""

from __future__ import annotations


class WriteAttempted(RuntimeError):
    """Something in the introspection server asked for a write. It is a bug."""


class Reader:
    """Exposes exactly the named attributes of `obj`, and nothing else.

    Not a dataclass and not `__getattr__`-with-a-blacklist: a blacklist has to
    be kept in step with the thing it wraps, and the one method added later is
    the one that gets through.
    """

    __slots__ = ("_obj", "_ok", "_what")

    def __init__(self, obj: object, ok: "frozenset[str]", what: str) -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_ok", ok)
        object.__setattr__(self, "_what", what)

    # -- the whole point -------------------------------------------------

    def __getattr__(self, name: str):
        ok = object.__getattribute__(self, "_ok")
        what = object.__getattribute__(self, "_what")
        if name not in ok:
            raise WriteAttempted(
                f"the introspection server asked for {what}.{name}, which is "
                f"not on its read-only list. If it is a read, add it to the "
                f"list in lsa/introspect/readonly.py and say so; if it acts on "
                f"the camera, the detector or the screen, it must go through "
                f"the Director instead (RULES A4, and the design's constraints "
                f"1 and 2).")
        return getattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name: str, value) -> None:
        raise WriteAttempted(
            f"the introspection server tried to assign "
            f"{object.__getattribute__(self, '_what')}.{name}. It reads.")

    def __delattr__(self, name: str) -> None:
        raise WriteAttempted("the introspection server tried to delete an attribute.")

    def __repr__(self) -> str:                           # pragma: no cover
        return f"<Reader {object.__getattribute__(self, '_what')}>"

    # `is None` on the wrapper must not lie about the thing inside it.
    def __bool__(self) -> bool:
        return bool(object.__getattribute__(self, "_obj") is not None)

    def unwrap_for_tests(self) -> object:                # pragma: no cover
        """Only the tests, and only to prove the wrapper is around the real thing."""
        return object.__getattribute__(self, "_obj")


# --- the lists -------------------------------------------------------------
#
# Every name here is a READ. Each one is used by `view.py`; a name nobody uses
# is removed rather than kept "in case", because an allowlist nobody prunes
# becomes the blacklist this class exists to avoid.

CAPTURE_READ = frozenset({
    "config",             # CaptureConfig: index, backend, format, requested fps
    "preview_width",
    "isRunning",          # QThread, read-only
    "written_value",      # what THIS APP wrote to a control; never a driver read
    "gain_at_open",       # the gain the camera reported before we wrote anything
    "evidence_between",   # the per-frame near-miss ring, already a copy
    "_detector",          # wrapped again below
    "_grouper",           # wrapped again below
    "_grouper_config",
    "_stats",
    "_last_frame",        # the latest BGR frame; a fresh array per read() call
    "_open_info",         # the format negotiated at open, cached there
    "_control_values",    # the cache of what this app has written
    "_running",
    "_show_mask",
    "_draw_overlay",
    "_writer",            # only ever tested for None, to report "recording"
    "_session_dir",
    "_noise_until",
    "_room_until",
})

# NOT on the list, deliberately, with the reason:
#   set_control, apply_preset, start_exposure_search, request_probe,
#   request_driver_dialog   -- UVC control writes (RULES A4)
#   _post, _cap, _commands, _mutex  -- the route to cap.set(), which is the
#                                      same hazard one indirection away
#   request_still, start_noise_calibration, start_room_measurement,
#   set_detector_params, set_grouper_config, set_gain_map, reset_background,
#   reset_stats, start_recording, stop_recording, save_snapshot, stop,
#   set_aim_help, set_show_mask, set_draw_overlay
#                           -- all ACT. Anything that acts takes the one run
#                              slot like everything else (constraint 2).

DETECTOR_READ = frozenset({
    "params",
    "last_score", "last_mask", "last_process_ms", "last_candidates",
    "last_rejected_roi", "last_frame_peak", "last_dim_peak", "last_dim_pos",
    "last_off_screen_peak", "last_rejects", "last_amp_gate",
    "_gain", "_gain_scaled", "_frame_index", "_bg",
    # `_recip` is 1/gain at frame size -- the number a raw chroma value is
    # literally multiplied by. Reported rather than recomputed, so what
    # `/score` shows as the amplification is the array the hot path uses.
    "_recip",
    # `_score` is PURE: it splits the frame and subtracts, allocating fresh
    # arrays and touching no state on the detector. Calling the detector's own
    # one rather than reimplementing the same two lines here is RULES C14 --
    # a seam tested on neither side is how `Probe.bar()` drifted from
    # `_on_noise_measured` and let the dimmest dot on the table win.
    "_score",
})
# NOT `_roi_image` or `_inside_roi`, although both are reads: each fills a
# lazy cache on the detector on first use, and an instrument that writes to
# the object it is measuring is how the exposure probe came to wedge the
# camera it was probing. `view.py` runs the polygon test itself with cv2.

TRACKER_READ = frozenset({
    "count", "open_tracks", "max_concurrent",
    "rejected_runs", "rejected_too_long", "cfg", "config",
})


def capture_reader(capture: object) -> "Reader | None":
    """Wrap the live capture thread, or None when the camera is not open."""
    if capture is None:
        return None
    return Reader(capture, CAPTURE_READ, "capture")


def detector_reader(capture_rd: "Reader | None") -> "Reader | None":
    if capture_rd is None:
        return None
    det = capture_rd._detector
    return None if det is None else Reader(det, DETECTOR_READ, "detector")


def tracker_reader(capture_rd: "Reader | None") -> "Reader | None":
    if capture_rd is None:
        return None
    tr = capture_rd._grouper
    return None if tr is None else Reader(tr, TRACKER_READ, "tracker")
