"""ScreenPlink's MCP face: `screenplink.exe --mcp`, or `python -m lsa --mcp`.

A Model Context Protocol server speaking JSON-RPC 2.0 over stdin/stdout, one
JSON object per line. It is a THIN HTTP CLIENT against the introspection server
that lives inside the running app (127.0.0.1:8777 by default), and nothing else.

STDLIB ONLY, deliberately and permanently. No cv2, no Qt, no numpy, not even
indirectly -- and so no `lsa.introspect` import either, because that package
pulls in the view layer. Two reasons, both load-bearing:

  * `camera_lab.main()` builds a QApplication on its second line, so importing
    anything near it costs seconds a call;
  * Claude will often call a tool BEFORE the app is running. This process must
    start and answer instantly in that case, and say how to start the app.

NOTHING but protocol may ever be written to stdout: a stray print corrupts the
stream. Logging goes to stderr.

Read-only for the camera BY CONSTRUCTION. There is exactly one tool per
introspection route and no more; the two that act (`press_key`, `click`) post a
key or a click to the TARGET WINDOW, travelling the same path a human's does.
No camera control is reachable from here: a single exposure write has hung this
camera family for 272 seconds (RULES A4).

THE DESCRIPTIONS BELOW ARE LIFTED VERBATIM from the `endpoints` dict in
`introspect/handlers.py`. They are what steers the model using them, and a
paraphrase is how a producer/consumer seam drifts. `tests/test_mcp.py` compares
every one with the live dict, and checks every route has a tool (or a stated
reason it has none) and every tool names a real route.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import sys
import urllib.parse

SERVER_NAME = "screenplink"
# A LITERAL, and it must equal `lsa.__version__`. It cannot be imported from
# there: this module may import nothing but the standard library, so that
# `--mcp` starts without Qt, OpenCV or the app package (test_mcp proves the
# heavy modules stay unimported). A seam between two copies of one fact is
# exactly where this project has been bitten before, so test_mcp checks that
# the two agree rather than trusting the comment.
SERVER_VERSION = "1.0.0"
DEFAULT_PORT = 8777                 # = lsa.introspect.server.DEFAULT_PORT; a test compares
HOST = "127.0.0.1"
# "Not running" must be answered inside ~2 s. Measured on this Windows box: a
# connect to a CLOSED loopback port does not fail fast, it burns the whole
# timeout (2.02 s at 2.0). A listening app completes the handshake in the kernel
# in well under a millisecond, so one second is generous.
CONNECT_TIMEOUT = 1.0
READ_TIMEOUT = 20.0                 # the app's own bridge deadlines are shorter

# `handlers.py` index(): "principle". Verbatim.
PRINCIPLE = ("measurements, not conclusions. The app's own verdicts appear "
             "under 'app_says' as one more datum to cross-check, never as "
             "the source of truth.")

INSTRUCTIONS = (
    PRINCIPLE + " These tools read the running ScreenPlink app through its "
    "introspection server. If a tool says the app is not listening, start it "
    "with `screenplink.exe --target --introspect` (introspection is OFF by "
    "default)."
)

NOT_RUNNING = (
    "ScreenPlink is not listening on http://{host}:{port}.\n"
    "Start it with:  screenplink.exe --target --introspect\n"
    "(`--introspect` is OFF by default; without it the app opens no port at "
    "all. LSA_INTROSPECT=<port> does the same.) Then call this tool again."
    "{detail}"
)

# --- the tool table ----------------------------------------------------------
# name -> route, description (verbatim from handlers.py `endpoints`), params,
# result kind. "params" maps a query-parameter name to (json type, description).
# A param description is also lifted verbatim where the dict has an entry for it.

_SCORE_QUERY = ("that neighbourhood around a point "
                "of your choosing (default: the "
                "brightest pixel of the score map)")

TOOLS: dict = {
    "health": {
        "route": ("GET", "/health"),
        "description": "alive, uptime, version, whether a camera is "
                       "open, and what this server has cost",
        "params": {}, "kind": "json",
    },
    "state": {
        "route": ("GET", "/state"),
        "description": "mode, layers, floor, bar, exposure, gain map, "
                       "mapping error, screen ownership, fps, control "
                       "writes, what the Director is running",
        "params": {}, "kind": "json",
    },
    "settings": {
        "route": ("GET", "/settings"),
        "description": "effective beside saved beside code default",
        "params": {}, "kind": "json",
    },
    "frame": {
        "route": ("GET", "/frame"),
        "description": "the latest camera frame, PNG (lossless)",
        "params": {}, "kind": "image",
    },
    "frame_measurements": {
        "route": ("GET", "/frame.json"),
        "description": "its measurements without the image",
        "params": {}, "kind": "json",
    },
    "score": {
        "route": ("GET", "/score"),
        "description": "the normalised score map the detector "
                       "thresholds, as MEASUREMENTS: the raw chroma, "
                       "the local amplification, the background model "
                       "and the final score as numbers over a "
                       "neighbourhood, plus every blob above the "
                       "threshold with the reasons it would be dropped",
        "params": {
            "x": ("number", _SCORE_QUERY),
            "y": ("number", _SCORE_QUERY),
            "box": ("integer", _SCORE_QUERY),
        },
        "kind": "json",
    },
    "score_png": {
        "route": ("GET", "/score.png"),
        "description": "that map as a lossless PNG -- but 8-bit, "
                       "so it clips and rounds. For a quick look, "
                       "never for an answer",
        "params": {}, "kind": "image",
    },
    "shots": {
        "route": ("GET", "/shots"),
        "description": "the last N detections with per-frame evidence",
        "params": {"n": ("integer", "the last N detections with per-frame evidence")},
        "kind": "json",
    },
    "drawn": {
        "route": ("GET", "/drawn"),
        "description": "what is painted, where, in screen AND camera "
                       "coordinates, with its red excess",
        "params": {}, "kind": "json",
    },
    "press_key": {
        "route": ("POST", "/key"),
        "description": "one key to the target window, no focus stolen",
        "params": {"key": ("string", "one key to the target window, no focus stolen")},
        "kind": "json",
    },
    "click": {
        "route": ("POST", "/click"),
        "description": "one left click at those screen "
                       "pixels, no focus stolen. A CLICK, "
                       "never a shot: nothing reaches the "
                       "score or the hit rate",
        "params": {
            "x": ("number", "one left click at those screen "
                            "pixels, no focus stolen. A CLICK, "
                            "never a shot: nothing reaches the "
                            "score or the hit rate"),
            "y": ("number", "one left click at those screen "
                            "pixels, no focus stolen. A CLICK, "
                            "never a shot: nothing reaches the "
                            "score or the hit rate"),
        },
        "kind": "json",
    },
}

# Routes the app serves that have NO tool, each with the reason. The test fails
# on a route that is in neither place, so a route added later cannot be missed.
NO_TOOL = {
    ("GET", "/"): "the index itself: its endpoint descriptions and principle "
                  "are what this table and INSTRUCTIONS are made from",
    ("GET", "/frame.png"): "the same bytes as /frame (an alias); two tools "
                           "for one image would only invite the wrong one",
    ("GET", "/score.npy"): "a raw .npy array: no MCP content type carries it "
                           "so that a model can use it, and /score already "
                           "gives the exact numbers",
}

# Query parameters sent for "GET /frame" etc. never include `format`: the PNG
# tools are the PNG routes, and a client cannot ask for npy through them.

READ_ONLY_ROUTES = {"GET"}


# --- plumbing ---------------------------------------------------------------

def log(msg: str) -> None:
    # Windowed exe: a parent that supplies no stderr leaves sys.stderr as None,
    # and print(file=None) falls back to sys.stdout -- which is the PROTOCOL
    # stream. Say nothing rather than corrupt it.
    if sys.stderr is None:
        return
    print(f"[screenplink-mcp] {msg}", file=sys.stderr, flush=True)


def port_from(args=None, env=None) -> int:
    """The port the app listens on. Mirrors `lsa.introspect.port_from` without
    importing it (that package drags in the view layer); a test compares them.

    `--introspect=PORT` / `--port=PORT` on our own command line, else
    LSA_INTROSPECT as a number, else the default. Anything unparseable falls
    back to the default rather than killing the server.
    """
    env = os.environ if env is None else env
    for a in (args or ()):
        for prefix in ("--introspect=", "--port="):
            if a.startswith(prefix):
                raw = a.split("=", 1)[1].strip()
                if raw.isdigit() and int(raw) > 0:
                    return int(raw)
    raw = str(env.get("LSA_INTROSPECT", "") or "").strip()
    if raw.isdigit() and int(raw) > 1:
        return int(raw)
    return DEFAULT_PORT


class Backend:
    """HTTP GET/POST against the introspection server. Never raises to the caller."""

    def __init__(self, port: int, host: str = HOST) -> None:
        self.port = port
        self.host = host

    def request(self, method: str, path: str, query: dict):
        """-> (status, content_type, body_bytes), or (None, "", b"") plus a
        reason string as a 4th element when the app cannot be reached."""
        qs = urllib.parse.urlencode(query)
        target = path + (("?" + qs) if qs else "")
        conn = None
        try:
            conn = http.client.HTTPConnection(self.host, self.port,
                                              timeout=CONNECT_TIMEOUT)
            conn.connect()
            conn.sock.settimeout(READ_TIMEOUT)
            conn.request(method, target, body=b"" if method == "POST" else None,
                         headers={"Content-Length": "0"} if method == "POST" else {})
            resp = conn.getresponse()
            body = resp.read()
            return resp.status, resp.getheader("Content-Type", ""), body, ""
        except (ConnectionError, TimeoutError, OSError) as e:
            return None, "", b"", f"{type(e).__name__}: {e}"
        except http.client.HTTPException as e:
            return None, "", b"", f"{type(e).__name__}: {e}"
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:                        # noqa: BLE001
                    pass


def _coerce(name: str, typ: str, value):
    """A tool argument -> its query-string text, or raise ValueError."""
    if typ == "integer":
        if isinstance(value, bool):
            raise ValueError(f"'{name}' must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"'{name}' must be an integer")
        return str(int(value))
    if typ == "number":
        if isinstance(value, bool):
            raise ValueError(f"'{name}' must be a number")
        f = float(value)
        return str(int(f)) if f.is_integer() and abs(f) < 1e15 else repr(f)
    return str(value)


def call_tool(backend: Backend, name: str, arguments: dict | None) -> dict:
    """-> an MCP tools/call result."""
    spec = TOOLS.get(name)
    if spec is None:
        return _text(f"unknown tool '{name}'", error=True)
    arguments = arguments or {}
    if not isinstance(arguments, dict):
        return _text("arguments must be an object", error=True)
    query: dict = {}
    for k, v in arguments.items():
        if k not in spec["params"]:
            return _text(f"tool '{name}' takes no argument '{k}'"
                         f" (it takes: {', '.join(spec['params']) or 'none'})",
                         error=True)
        if v is None:
            continue
        try:
            query[k] = _coerce(k, spec["params"][k][0], v)
        except (ValueError, TypeError) as e:
            return _text(str(e), error=True)
    if name == "press_key" and "key" not in query:
        return _text("press_key needs 'key'", error=True)
    if name == "click" and not ("x" in query and "y" in query):
        return _text("click needs 'x' and 'y' (screen pixels)", error=True)

    method, path = spec["route"]
    status, ctype, body, why = backend.request(method, path, query)
    if status is None:
        return _text(NOT_RUNNING.format(
            host=backend.host, port=backend.port,
            detail=f"\n(connection: {why})" if why else ""), error=True)

    if spec["kind"] == "image" and status == 200 and ctype.startswith("image/png"):
        return {"content": [{"type": "image",
                             "data": base64.b64encode(body).decode("ascii"),
                             "mimeType": "image/png"}]}
    text = body.decode("utf-8", "replace")
    if status >= 400:
        return _text(f"HTTP {status} from {method} {path}:\n{text}", error=True)
    return _text(text)


def _text(s: str, error: bool = False) -> dict:
    out = {"content": [{"type": "text", "text": s}]}
    if error:
        out["isError"] = True
    return out


def tool_list() -> list:
    out = []
    for name, spec in TOOLS.items():
        props = {k: {"type": t, "description": d}
                 for k, (t, d) in spec["params"].items()}
        method, _ = spec["route"]
        out.append({
            "name": name,
            "description": spec["description"],
            "inputSchema": {"type": "object", "properties": props,
                            "additionalProperties": False},
            "annotations": {"readOnlyHint": method in READ_ONLY_ROUTES,
                            "openWorldHint": False},
        })
    return out


# --- JSON-RPC ----------------------------------------------------------------

SUPPORTED_PROTOCOL = "2025-06-18"


def _err(rid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def handle(msg, backend: Backend):
    """One decoded JSON-RPC message -> a response dict, or None (notification)."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _err(msg.get("id") if isinstance(msg, dict) else None,
                    -32600, "invalid request")
    method = msg.get("method")
    has_id = "id" in msg
    rid = msg.get("id")
    params = msg.get("params") or {}

    if not isinstance(method, str):
        # A response from the client to something we never sent: ignore.
        return None if "result" in msg or "error" in msg else _err(rid, -32600, "no method")

    if method == "initialize":
        want = params.get("protocolVersion") if isinstance(params, dict) else None
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": want if isinstance(want, str) and want else SUPPORTED_PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS,
        }}
    if method.startswith("notifications/"):
        return None
    if not has_id:
        return None                       # any other notification: no reply
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": tool_list()}}
    if method == "tools/call":
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            return _err(rid, -32602, "tools/call needs params.name")
        if params["name"] not in TOOLS:
            return _err(rid, -32602, f"unknown tool: {params['name']}")
        return {"jsonrpc": "2.0", "id": rid,
                "result": call_tool(backend, params["name"], params.get("arguments"))}
    return _err(rid, -32601, f"method not found: {method}")


def serve(stdin, stdout, backend: Backend) -> None:
    """Read lines until EOF. `stdout` is a binary stream; only protocol goes to it."""
    def send(obj) -> None:
        stdout.write((json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
                      + "\n").encode("ascii"))
        stdout.flush()

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
        except ValueError:
            send(_err(None, -32700, "parse error"))
            continue
        try:
            replies = [handle(m, backend) for m in msg] if isinstance(msg, list) \
                else [handle(msg, backend)]
        except Exception as e:                           # noqa: BLE001
            log(f"internal error: {type(e).__name__}: {e}")
            rid = msg.get("id") if isinstance(msg, dict) else None
            replies = [_err(rid, -32603, "internal error")] if rid is not None else []
        replies = [r for r in replies if r is not None]
        if isinstance(msg, list):
            if replies:
                send(replies)
        else:
            for r in replies:
                send(r)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    port = port_from(argv)
    log(f"MCP server on stdio; app expected at http://{HOST}:{port}")
    serve(sys.stdin.buffer, sys.stdout.buffer, Backend(port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
