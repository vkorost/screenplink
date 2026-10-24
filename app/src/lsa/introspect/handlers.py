"""One function per endpoint. Each returns `(status, content_type, payload)`.

Handlers hold no state and make no decisions: they call `LabView`, which is the
only thing that touches the app. That split exists so the whole surface can be
exercised headless against a stand-in lab, which is how `tests/test_introspect.py`
drives every endpoint with no camera and no screen.
"""

from __future__ import annotations

JSON = "application/json"
PNG = "image/png"
OCTET = "application/octet-stream"


def routes_for(view) -> dict:
    """The route table. `(method, path) -> fn(query, body)`."""

    def health(query, body):
        stats = {}
        server = view.server          # declared by LabView.__init__; None until installed
        if server is not None:
            stats.update(server.stats())
        if view._bridge is not None:
            stats.update(view._bridge.stats())
        return 200, JSON, view.health(stats)

    def state(query, body):
        return 200, JSON, view.state()

    def settings(query, body):
        return 200, JSON, view.settings()

    def frame(query, body):
        return _encode(view.frame, query, default="png")

    def frame_png(query, body):
        return _encode(view.frame, query, default="png", force="png")

    def frame_json(query, body):
        return _encode(view.frame, query, default="json", force="json")

    def score(query, body):
        return _encode(view.score, query, default="json")

    def score_png(query, body):
        return _encode(view.score, query, default="png", force="png")

    def score_npy(query, body):
        return _encode(view.score, query, default="npy", force="npy")

    def shots(query, body):
        return 200, JSON, view.shots(_int(query.get("n"), 10))

    def drawn(query, body):
        ok, payload = view.drawn(query)
        return (200 if ok else 503), JSON, payload

    def key(query, body):
        name = query.get("key") or query.get("name") or ""
        if not name and body:
            import json
            try:
                name = str(json.loads(body.decode("utf-8")).get("key", ""))
            except Exception:                            # noqa: BLE001
                name = body.decode("utf-8", "replace").strip()
        ok, payload = view.press_key(name)
        return (200 if ok else 409), JSON, payload

    def click(query, body):
        x, y = query.get("x"), query.get("y")
        if (x is None or y is None) and body:
            import json
            try:
                data = json.loads(body.decode("utf-8"))
                x, y = data.get("x", x), data.get("y", y)
            except Exception:                            # noqa: BLE001
                pass
        ok, payload = view.click(x, y)
        return (200 if ok else 409), JSON, payload

    def index(query, body):
        return 200, JSON, {
            "app": "lsa camera lab introspection",
            "principle": "measurements, not conclusions. The app's own "
                         "verdicts appear under 'app_says' as one more datum "
                         "to cross-check, never as the source of truth.",
            "endpoints": {
                "GET /health": "alive, uptime, version, whether a camera is "
                               "open, and what this server has cost",
                "GET /state": "mode, layers, floor, bar, exposure, gain map, "
                              "mapping error, screen ownership, fps, control "
                              "writes, what the Director is running",
                "GET /settings": "effective beside saved beside code default",
                "GET /frame": "the latest camera frame, PNG (lossless)",
                "GET /frame.json": "its measurements without the image",
                "GET /frame?format=npy": "the frame as a .npy array, exact",
                "GET /score": "the normalised score map the detector "
                              "thresholds, as MEASUREMENTS: the raw chroma, "
                              "the local amplification, the background model "
                              "and the final score as numbers over a "
                              "neighbourhood, plus every blob above the "
                              "threshold with the reasons it would be dropped",
                "GET /score?x=&y=&box=33": "that neighbourhood around a point "
                                           "of your choosing (default: the "
                                           "brightest pixel of the score map)",
                "GET /score.png": "that map as a lossless PNG -- but 8-bit, "
                                  "so it clips and rounds. For a quick look, "
                                  "never for an answer",
                "GET /score.npy": "the score array itself, exact dtype and "
                                  "shape, no interpretation at all",
                "GET /shots?n=10": "the last N detections with per-frame evidence",
                "GET /drawn": "what is painted, where, in screen AND camera "
                              "coordinates, with its red excess",
                "POST /key?key=g": "one key to the target window, no focus stolen",
                "POST /click?x=960&y=540": "one left click at those screen "
                                            "pixels, no focus stolen. A CLICK, "
                                            "never a shot: nothing reaches the "
                                            "score or the hit rate",
            },
            "no_writes": "no camera control is ever written from here; a "
                         "single exposure write has hung this camera for 272s",
        }

    return {
        ("GET", "/"): index,
        ("GET", "/health"): health,
        ("GET", "/state"): state,
        ("GET", "/settings"): settings,
        ("GET", "/frame"): frame,
        ("GET", "/frame.png"): frame_png,
        ("GET", "/frame.json"): frame_json,
        ("GET", "/score"): score,
        ("GET", "/score.png"): score_png,
        ("GET", "/score.npy"): score_npy,
        ("GET", "/shots"): shots,
        ("GET", "/drawn"): drawn,
        ("POST", "/key"): key,
        ("POST", "/click"): click,
    }


def _int(raw, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _encode(view_fn, query, default: str, force: "str | None" = None):
    """Three encodings of one measurement, and the view is told which is wanted
    so it does not pay to produce the other two.

    `json` is always available and is the only form that carries the
    arithmetic. `png` is lossless but 8-bit, so it clips and rounds and says
    nothing about what a value MEANS -- it is for a quick look, never the
    answer. `npy` is the array itself, exact dtype and shape, no
    interpretation at all.
    """
    want = (force or query.get("format") or default).lower()
    query["format"] = want
    meta, payload = view_fn(query)
    if want == "png":
        return (200, PNG, payload) if payload else (503, JSON, meta)
    if want == "npy":
        return ((200, OCTET, payload) if payload else (503, JSON, meta))
    return 200, JSON, meta
