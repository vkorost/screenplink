"""The in-process introspection server: plain HTTP, JSON, localhost, read-only
except for one action.

Why it exists, in the owner's words: the recurring failure on this project is
**the app knew and nobody could see it.** In one morning: the exposure probe
measured the centre of the screen instead of the weak region and never logged
which spot; `max_area` was 2000 in his settings against a code default of 1500,
invisible until someone read a JSON file; the sensitivity map was poisoned by a
walking figure, discoverable only by reading `calibration.json` after shutdown;
pressing a key meant stealing window focus and knocking the target window
behind another; and seeing what the camera saw meant pulling frames out of an
815 MB AVI -- whose re-encode destroys exactly the near-threshold noise being
chased, so the question was unanswerable. Every one of those is a two-second
query against a live endpoint.

What it is not: `tools/replay.py` is the regression instrument -- it proved the
size gate against 26,614 recorded frames. This is the live-debug instrument.

THE FIVE RULES THIS FILE ENFORCES, each from a scar:

1. **No camera control writes, ever.** The camera is reached only through
   `readonly.capture_reader`, which does not have a write on it (RULES A4).
2. **Nothing bypasses the Director.** The one action, `POST /key`, posts a real
   key event to the target window, so it travels the same path a human keypress
   does and any routine it starts takes the one run slot (constraint 2).
3. **Off the hot path.** Its own thread; reads are plain attribute reads; the
   two pieces that need Qt go through `GuiBridge` with a deadline; and at most
   `MAX_INFLIGHT` requests are served at once so a hammering client cannot
   compete with the detector for memory bandwidth. The cost is measured and
   reported by `/health` (RULES C3).
4. **Off by default**, and identical behaviour when off (`--introspect`, or
   `LSA_INTROSPECT`).
5. **Localhost only.** `bind()` refuses any other host, by exception, before a
   socket exists.

And it writes nothing the user owns: there is no code path in this package that
opens a file for writing (RULES C11, enforced in `tests/test_introspect.py`).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

LOCAL_ONLY = ("127.0.0.1", "localhost", "::1")

DEFAULT_PORT = 8777


class NotLocal(ValueError):
    """Someone asked for a non-loopback bind. There is no reason to want one."""


class Busy(RuntimeError):
    pass


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=1, sort_keys=False,
                      default=str).encode("utf-8") + b"\n"


class IntrospectServer:
    """Owns the socket, the thread and the route table."""

    # Two at once is plenty for an agent with curl, and it is a ceiling on
    # how much this server can ever cost the capture loop: a /score request
    # copies and PNG-encodes a frame-sized array, and twenty of those at once
    # would be a measurable amount of memory bandwidth taken from a detector
    # that has 14 ms to spend.
    MAX_INFLIGHT = 2

    # AND ONE HEAVY REQUEST AT A TIME. `/frame`, `/score` and `/drawn` each
    # copy and measure a frame-sized array; a /score at 1280x720 measures
    # about 16 ms of work on this thread. The budget it must not disturb is
    # the 14 ms cliff: the capture loop is read + detect, and under about
    # 14 ms of detect it holds 60 fps while over it it falls to 30 -- which
    # halves the camera frames inside a 100 ms pulse and is what decides
    # whether a trigger pull registers.
    #
    # Measured in `tests/test_introspect.py` section 11, at his real
    # 1280x720: the detector alone runs 5.7 ms median / 6.7 ms p95, and with
    # two clients requesting /score with no pause between requests it runs
    # 5.7 / 8.0 -- so the cost of being watched is about +1.3 ms at the 95th
    # percentile, against 6.7 ms of headroom.
    HEAVY = frozenset({"/frame", "/frame.png", "/frame.json",
                       "/score", "/score.png", "/score.npy", "/drawn"})

    def __init__(self, routes: dict, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, log=None) -> None:
        if host not in LOCAL_ONLY:
            raise NotLocal(
                f"the introspection server refuses to bind to {host!r}. It "
                f"exposes the camera's view of the room, the screen's content "
                f"and a key-press endpoint, and it is a debugging aid on one "
                f"machine. Loopback only: {LOCAL_ONLY}.")
        self.host = host
        self.port = int(port)
        self.routes = routes
        self._log = log or (lambda _s: None)
        self._gate = threading.Semaphore(self.MAX_INFLIGHT)
        self._heavy_gate = threading.Semaphore(1)
        self._httpd: "ThreadingHTTPServer | None" = None
        self._thread: "threading.Thread | None" = None
        self.requests = 0
        self.errors = 0
        self.busy_refusals = 0
        self.served_ms_total = 0.0
        self.served_ms_max = 0.0
        self.last_path = ""
        self.started_at = 0.0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> int:
        """Bind, serve on a daemon thread, and return the port actually used."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "lsa-introspect/1"

            def log_message(self, fmt, *args) -> None:
                # NOT to stderr. The app is usually launched with pythonw, and
                # a per-request line in the live log would bury the one
                # account of an unattended launch -- `poke()` once produced
                # about four hundred identical refusal lines there.
                pass

            def do_GET(self) -> None:                   # noqa: N802
                server._serve(self, "GET")

            def do_POST(self) -> None:                  # noqa: N802
                server._serve(self, "POST")

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self.started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="lsa-introspect",
            kwargs={"poll_interval": 0.2}, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:                            # pragma: no cover
                pass
            try:
                self._httpd.server_close()
            except Exception:                            # pragma: no cover
                pass
            self._httpd = None

    # -- serving ----------------------------------------------------------

    def stats(self) -> dict:
        return {
            "port": self.port,
            "requests": self.requests,
            "errors": self.errors,
            "busy_refusals": self.busy_refusals,
            "max_inflight": self.MAX_INFLIGHT,
            "max_heavy_inflight": 1,
            "served_ms_max": round(self.served_ms_max, 2),
            "served_ms_mean": round(self.served_ms_total / self.requests, 2)
            if self.requests else 0.0,
            "last_path": self.last_path,
            "uptime_s": round(time.perf_counter() - self.started_at, 1)
            if self.started_at else 0.0,
        }

    def _serve(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        t0 = time.perf_counter()
        parsed = urlparse(handler.path)
        path = parsed.path.rstrip("/") or "/"
        query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
        self.last_path = f"{method} {path}"

        if not self._gate.acquire(blocking=False):
            self.busy_refusals += 1
            self._reply(handler, 503, "application/json", _json_bytes({
                "error": "busy",
                "detail": f"the introspection server serves at most "
                          f"{self.MAX_INFLIGHT} requests at once, so that it "
                          f"can never compete with the detector for the CPU. "
                          f"Retry.",
            }))
            return
        heavy = path in self.HEAVY
        if heavy and not self._heavy_gate.acquire(timeout=5.0):
            self._gate.release()
            self.busy_refusals += 1
            self._reply(handler, 503, "application/json", _json_bytes({
                "error": "busy",
                "detail": "one frame-sized request is served at a time, so "
                          "this server can never take the CPU the detector "
                          "needs. Retry.",
            }))
            return
        try:
            body = b""
            length = int(handler.headers.get("Content-Length") or 0)
            if length > 0:
                body = handler.rfile.read(min(length, 1 << 16))
            route = self.routes.get((method, path))
            if route is None:
                self._reply(handler, 404, "application/json", _json_bytes({
                    "error": "no such endpoint",
                    "available": sorted(f"{m} {p}" for m, p in self.routes),
                }))
                return
            status, ctype, payload = route(query, body)
            if isinstance(payload, (dict, list)):
                payload = _json_bytes(payload)
                ctype = "application/json"
            self.requests += 1
            self._reply(handler, status, ctype, payload)
        except Exception as exc:                         # noqa: BLE001
            self.errors += 1
            import traceback
            self._reply(handler, 500, "application/json", _json_bytes({
                "error": repr(exc),
                "traceback": traceback.format_exc().splitlines()[-12:],
            }))
        finally:
            if heavy:
                self._heavy_gate.release()
            self._gate.release()
            ms = (time.perf_counter() - t0) * 1000.0
            self.served_ms_total += ms
            self.served_ms_max = max(self.served_ms_max, ms)

    @staticmethod
    def _reply(handler: BaseHTTPRequestHandler, status: int, ctype: str,
               payload: bytes) -> None:
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", ctype)
            handler.send_header("Content-Length", str(len(payload)))
            # The browser is not the client here, but a stray page must not be
            # able to read the camera's view of the room.
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            handler.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
            pass
