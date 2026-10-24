"""The MCP face of the app: `python -m lsa --mcp`, driven as a real child process.

The server under test is spawned exactly as Claude Code would spawn it and
spoken to over its stdin/stdout. A stub HTTP server stands in for the running
app (RULES C12: test the thing, not its stand-in). Nothing here opens a camera,
a window or a sound, and nothing touches a file the user owns.

What is guarded, each from a scar in this project:

  * the producer/consumer SEAM with `introspect/handlers.py` (C14): every
    route has a tool or a stated reason it has none, every tool names a real
    route, and every description is the dict's text verbatim;
  * STDLIB ONLY, proven by loading it and looking at `sys.modules`, because
    `camera_lab.main()` builds a QApplication on its second line;
  * "the app is not running" is the normal first case: answered inside about
    two seconds, with how to start it, and never a socket traceback;
  * stdout carries nothing but protocol, byte for byte.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import base64
import importlib.util
import json
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYEXE = str(PY if PY.exists() else sys.executable)

CHECKS = 0
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILED.append(what + (f"  --  {detail}" if detail else ""))
        print(f"  FAIL  {what}")
        if detail:
            print(f"        {detail}")


def section(n: str) -> None:
    print(f"\n=== {n} ===")


# ---------------------------------------------------------------------------
section("1. the seam with introspect/handlers.py")
# Loaded BY PATH, not as a package: `lsa.introspect` imports the view layer.
_spec = importlib.util.spec_from_file_location(
    "handlers_under_test", SRC / "lsa" / "introspect" / "handlers.py")
handlers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handlers)
ROUTES = handlers.routes_for(None)
INDEX = ROUTES[("GET", "/")]({}, b"")[2]
ENDPOINTS: dict = INDEX["endpoints"]

sys.path.insert(0, str(SRC))
import lsa.mcp as mcp                                         # noqa: E402

check(len(ROUTES) == 14, "the app serves 14 routes (this suite was written "
      "against that)", f"it now serves {len(ROUTES)}: read the new ones")


def dict_entries_for(method: str, path: str) -> list[tuple[str, str]]:
    """Endpoint-dict entries whose key, query stripped, is this route."""
    out = []
    for k, v in ENDPOINTS.items():
        m, _, rest = k.partition(" ")
        if m == method and rest.split("?", 1)[0] == path:
            out.append((k, v))
    return out


tool_routes = {spec["route"] for spec in mcp.TOOLS.values()}
for route in ROUTES:
    has_tool = route in tool_routes
    excused = route in mcp.NO_TOOL
    check(has_tool != excused,
          f"route {route[0]} {route[1]} has a tool, or a stated reason it has "
          "none (and not both)",
          "add a tool to mcp.TOOLS, or the route to mcp.NO_TOOL with a reason")
for route, why in mcp.NO_TOOL.items():
    check(route in ROUTES and bool(why.strip()),
          f"NO_TOOL entry {route} names a real route and gives a reason")
for name, spec in mcp.TOOLS.items():
    check(spec["route"] in ROUTES,
          f"tool '{name}' names a real route ({spec['route'][1]})")

for name, spec in mcp.TOOLS.items():
    entries = dict_entries_for(*spec["route"])
    check(bool(entries), f"tool '{name}' has an endpoints-dict entry to lift from")
    texts = {v for _, v in entries}
    plain = [v for k, v in entries if "?" not in k]
    primary = plain[0] if plain else entries[0][1]
    check(spec["description"] == primary,
          f"tool '{name}' description is the dict's text verbatim",
          f"tool: {spec['description']!r}\n        dict: {primary!r}")
    for p, (_t, pdesc) in spec["params"].items():
        check(pdesc in texts,
              f"tool '{name}' parameter '{p}' description is the dict's text "
              "verbatim")

# Every dict entry is accounted for: a tool, or an excused route.
for k in ENDPOINTS:
    m, _, rest = k.partition(" ")
    route = (m, rest.split("?", 1)[0])
    check(route in tool_routes or route in mcp.NO_TOOL,
          f"endpoints entry '{k}' is covered by a tool or an excuse")

check(mcp.PRINCIPLE == INDEX["principle"],
      "the principle is the app's own, verbatim")
check(INDEX["principle"] in mcp.INSTRUCTIONS,
      "the principle is carried in the server's instructions")

# Writes: the only tools that act are the two that post to the TARGET WINDOW.
posting = sorted(s["route"][1] for s in mcp.TOOLS.values()
                 if s["route"][0] != "GET")
check(posting == ["/click", "/key"],
      "only /key and /click are non-GET tools; nothing can reach a camera "
      "control", f"got {posting}")
check(not any("exposure" in n or "control" in n or "camera" in n
              for n in mcp.TOOLS), "no tool is named for a camera control")

# port parsing agrees with the app's own
from lsa.introspect import port_from as app_port_from        # noqa: E402
from lsa.introspect.server import DEFAULT_PORT as APP_DEFAULT  # noqa: E402

check(mcp.DEFAULT_PORT == APP_DEFAULT,
      "the default port is the app's default port")
for env, args in [({"LSA_INTROSPECT": "8899"}, []), ({"LSA_INTROSPECT": "1"}, []),
                  ({}, ["--introspect=9123"]), ({"LSA_INTROSPECT": "true"}, [])]:
    want = app_port_from(args, env)
    check(mcp.port_from(args, env) == want,
          f"port_from agrees with the app for env={env} args={args}",
          f"app {want}, mcp {mcp.port_from(args, env)}")
check(mcp.port_from([], {}) == APP_DEFAULT,
      "with nothing asked for, the MCP server still looks at the default port "
      "(the app may have been started with --introspect)")
check(mcp.port_from(["--port=9001"], {}) == 9001, "--port=N is honoured")
check(mcp.port_from([], {"LSA_INTROSPECT": "garbage"}) == APP_DEFAULT,
      "an unparseable port falls back to the default instead of dying")

# ---------------------------------------------------------------------------
section("2. stdlib only, proven by loading it")
import ast                                                    # noqa: E402

tree = ast.parse((SRC / "lsa" / "mcp.py").read_text(encoding="utf-8"))
mods = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        mods |= {a.name.split(".")[0] for a in node.names}
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            mods.add("." * node.level + (node.module or ""))
        else:
            mods.add((node.module or "").split(".")[0])
non_std = sorted(m for m in mods if m not in sys.stdlib_module_names)
check(not non_std, "mcp.py imports only the standard library", f"{non_std}")

# mcp.py may not IMPORT the version (stdlib only, above), so it carries a
# literal. Two copies of one fact is a seam, and this project loses days in
# seams, so the copies are checked against each other here.
_pkg_init = (SRC / "lsa" / "__init__.py").read_text(encoding="utf-8")
_declared = ""
for _line in _pkg_init.splitlines():
    if _line.startswith("__version__"):
        _declared = _line.split("=", 1)[1].split("#")[0].strip().strip(chr(34)).strip(chr(39))
        break
check(bool(_declared), "lsa/__init__.py declares __version__", _declared)
check(_declared == mcp.SERVER_VERSION,
      "the MCP server reports the product version, not one of its own",
      f"lsa.__version__={_declared!r} mcp.SERVER_VERSION={mcp.SERVER_VERSION!r}")

_ENV = dict(os.environ)
_ENV["PYTHONPATH"] = str(SRC)
_ENV.pop("LSA_INTROSPECT", None)
_ENV["QT_QPA_PLATFORM"] = "offscreen"
_probe = (
    "import sys, io\n"
    "sys.stdin = io.TextIOWrapper(io.BytesIO(b''))\n"
    "from lsa.__main__ import main\n"
    "rc = main(['--mcp'])\n"
    "heavy = [m for m in ('cv2','numpy','PySide6','sounddevice','lsa.tools',"
    "'lsa.tools.camera_lab','lsa.introspect') if m in sys.modules]\n"
    "print('RC', rc, 'HEAVY', heavy)\n")
_t0 = time.perf_counter()
_r = subprocess.run([PYEXE, "-c", _probe], env=_ENV, capture_output=True,
                    text=True, timeout=60)
_dt = time.perf_counter() - _t0
check(_r.returncode == 0 and "RC 0 HEAVY []" in _r.stdout,
      "`python -m lsa --mcp` dispatch loads no cv2, numpy, Qt, camera_lab or "
      "introspect package", _r.stdout.strip() + _r.stderr.strip()[-300:])
check(_dt < 3.0, f"the MCP face starts and exits in under 3 s ({_dt:.2f}s)")

# ---------------------------------------------------------------------------
section("3. a stub app, and a real child process speaking to it")


def make_png(w: int = 4, h: int = 3) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes((x * 60 + y * 20) % 256 for x in range(w))
                   for y in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PNG = make_png()
SEEN: list[tuple[str, str, dict]] = []


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):                               # silence
        pass

    def _send(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve(self, method):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        SEEN.append((method, u.path, q))
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
        j = lambda o, s=200: self._send(s, "application/json",       # noqa: E731
                                        json.dumps(o).encode())
        if (method, u.path) == ("GET", "/health"):
            j({"alive": True, "uptime_s": 12.5, "camera_open": False})
        elif (method, u.path) == ("GET", "/state"):
            j({"floor": 2, "bar": 8, "app_says": {"verdict": "ready"}})
        elif (method, u.path) == ("GET", "/settings"):
            j({"effective": {"max_area": 1500}})
        elif (method, u.path) in (("GET", "/frame"), ("GET", "/score.png")):
            self._send(200, "image/png", PNG)
        elif (method, u.path) == ("GET", "/frame.json"):
            j({"shape": [720, 1280], "mean": 12.0})
        elif (method, u.path) == ("GET", "/score"):
            j({"center": [q.get("x"), q.get("y")], "box": q.get("box"),
               "blobs": []})
        elif (method, u.path) == ("GET", "/shots"):
            j({"n": q.get("n"), "shots": []})
        elif (method, u.path) == ("GET", "/drawn"):
            j({"error": "no target window"}, 503)
        elif (method, u.path) in (("POST", "/key"), ("POST", "/click")):
            j({"ok": True, "got": q})
        else:
            j({"error": "no such route"}, 404)

    def do_GET(self):
        self._serve("GET")

    def do_POST(self):
        self._serve("POST")


httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
STUB_PORT = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()


def free_closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Child:
    """`python -m lsa --mcp` as Claude Code would run it."""

    def __init__(self, port: int, extra_args=(), env_port=True):
        env = dict(_ENV)
        if env_port:
            env["LSA_INTROSPECT"] = str(port)
        self.p = subprocess.Popen(
            [PYEXE, "-m", "lsa", "--mcp", *extra_args], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=str(ROOT))
        self.lines: "queue.Queue[bytes]" = queue.Queue()
        self.all_stdout: list[bytes] = []
        self._id = 0
        threading.Thread(target=self._pump, daemon=True).start()
        self.err = b""
        threading.Thread(target=self._pump_err, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.all_stdout.append(line)
            self.lines.put(line)
        self.lines.put(b"")                                   # EOF marker

    def _pump_err(self):
        self.err = self.p.stderr.read()

    def send_raw(self, text: str):
        self.p.stdin.write(text.encode("utf-8") + b"\n")
        self.p.stdin.flush()

    def send(self, obj):
        self.send_raw(json.dumps(obj))

    def recv(self, timeout=15.0):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        return json.loads(line) if line else None

    def rpc(self, method, params=None, timeout=15.0):
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        r = self.recv(timeout)
        return r

    def call(self, name, args=None, timeout=15.0):
        return self.rpc("tools/call", {"name": name, "arguments": args or {}},
                        timeout)

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:                                     # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        return self.p.returncode


TRANSCRIPT: list[str] = []


def note(direction: str, obj) -> None:
    s = json.dumps(obj)
    if len(s) > 330:
        s = s[:330] + " ...(truncated for the transcript)"
    TRANSCRIPT.append(f"{direction} {s}")


c = Child(STUB_PORT)

# -- handshake
init_req = {"protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}
note(">>", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": init_req})
r = c.rpc("initialize", init_req)
note("<<", r)
check(r is not None and r.get("id") == 1 and "result" in r, "initialize answers")
res = (r or {}).get("result", {})
check(res.get("protocolVersion") == "2024-11-05",
      "initialize echoes the client's requested protocolVersion")
check("tools" in res.get("capabilities", {}), "initialize advertises tools")
check(res.get("serverInfo", {}).get("name") == "screenplink",
      "initialize names the server")
check(INDEX["principle"] in res.get("instructions", ""),
      "the instructions carry the principle verbatim, over the wire")

note(">>", {"jsonrpc": "2.0", "method": "notifications/initialized"})
c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
r = c.rpc("ping")
note("<<", r)
check(r is not None and r.get("id") == 2 and r.get("result") == {},
      "ping is answered, and the reply to it is the NEXT line (so the "
      "initialized notification got no reply)")

# -- tools/list
note(">>", {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
r = c.rpc("tools/list")
note("<<", r)
tools = {t["name"]: t for t in (r or {}).get("result", {}).get("tools", [])}
check(set(tools) == set(mcp.TOOLS) and len(tools) == 11,
      "tools/list names every tool and no other", f"{sorted(tools)}")
for name, spec in mcp.TOOLS.items():
    t = tools.get(name, {})
    check(t.get("description") == spec["description"],
          f"over the wire: '{name}' description is verbatim")
    sch = t.get("inputSchema", {})
    check(sch.get("type") == "object" and "properties" in sch,
          f"over the wire: '{name}' has an object inputSchema")
check(tools["press_key"]["annotations"]["readOnlyHint"] is False
      and tools["state"]["annotations"]["readOnlyHint"] is True,
      "the acting tools are not marked read-only, the looking ones are")

# -- JSON tool calls
note(">>", {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "state", "arguments": {}}})
r = c.call("state")
note("<<", r)
body = (r or {}).get("result", {})
check(not body.get("isError") and body["content"][0]["type"] == "text"
      and json.loads(body["content"][0]["text"])["app_says"]["verdict"] == "ready",
      "a JSON tool returns the app's JSON verbatim as text content")

r = c.call("score", {"x": 900, "y": 360.0, "box": 9})
q = SEEN[-1]
check(q[1] == "/score" and q[2] == {"x": "900", "y": "360", "box": "9"},
      "score forwards x, y and box as the query", f"{q}")

r = c.call("shots", {"n": 5})
check(SEEN[-1][1] == "/shots" and SEEN[-1][2] == {"n": "5"},
      "shots forwards n")

r = c.call("press_key", {"key": "g"})
check(SEEN[-1][0] == "POST" and SEEN[-1][1] == "/key"
      and SEEN[-1][2] == {"key": "g"}, "press_key POSTs /key?key=g",
      f"{SEEN[-1]}")
r = c.call("click", {"x": 960, "y": 540})
check(SEEN[-1][0] == "POST" and SEEN[-1][1] == "/click"
      and SEEN[-1][2] == {"x": "960", "y": "540"}, "click POSTs /click?x=&y=")

for nm, route in (("health", "/health"), ("settings", "/settings"),
                  ("frame_measurements", "/frame.json")):
    r = c.call(nm)
    check(SEEN[-1][1] == route and not (r["result"].get("isError")),
          f"'{nm}' reaches {route}")

# -- image tools
for nm in ("frame", "score_png"):
    if nm == "frame":
        note(">>", {"jsonrpc": "2.0", "id": c._id + 1, "method": "tools/call",
                    "params": {"name": nm, "arguments": {}}})
    r = c.call(nm)
    if nm == "frame":
        note("<<", r)
    blk = (r or {}).get("result", {}).get("content", [{}])[0]
    check(blk.get("type") == "image" and blk.get("mimeType") == "image/png",
          f"'{nm}' returns MCP image content with mimeType image/png")
    raw = base64.b64decode(blk.get("data", ""), validate=True) if blk.get("data") else b""
    check(raw == PNG and raw[:8] == b"\x89PNG\r\n\x1a\n",
          f"'{nm}' image data is the real PNG bytes, base64, not a path or a "
          "hexdump")
    check(len(raw) > 8 and struct.unpack(">II", raw[16:24]) == (4, 3),
          f"'{nm}' PNG decodes to the stub's 4x3 IHDR")

# -- errors from the app, and from the caller
r = c.call("drawn")
note("<<", r)
out = r["result"]
check(out.get("isError") is True and "503" in out["content"][0]["text"]
      and "no target window" in out["content"][0]["text"],
      "an HTTP 503 from the app comes back as an error WITH the app's body")
r = c.call("state", {"bogus": 1})
check(r["result"].get("isError") is True, "an unknown argument is refused")
r = c.call("press_key", {})
check(r["result"].get("isError") is True, "press_key without a key is refused")
r = c.call("click", {"x": 1})
check(r["result"].get("isError") is True, "click without y is refused")
r = c.call("shots", {"n": "five"})
check(r["result"].get("isError") is True, "a non-numeric n is refused")
before = len(SEEN)
r = c.call("score", {"box": 3.5})
check(r["result"].get("isError") is True and len(SEEN) == before,
      "a refused call never reaches the app")

# -- protocol errors
r = c.rpc("no/such/method")
note("<<", r)
check(r is not None and r.get("error", {}).get("code") == -32601,
      "an unknown method is a -32601 error, not a crash")
r = c.rpc("tools/call", {"name": "nope"})
check(r is not None and r.get("error", {}).get("code") == -32602,
      "an unknown tool is a -32602 error")
c.send_raw("this is not json")
r = c.recv()
check(r is not None and r.get("error", {}).get("code") == -32700
      and r.get("id") is None, "garbage on stdin is a -32700 parse error")
r = c.rpc("ping")
check(r is not None and r.get("result") == {},
      "the server is still alive after garbage")
c.send({"jsonrpc": "2.0", "method": "something/unknown"})
r = c.rpc("ping")
check(r is not None and r.get("id") == c._id,
      "an unknown notification gets no reply")

# -- stdout is protocol and only protocol
rc = c.close()
time.sleep(0.2)
bad = []
for ln in c.all_stdout:
    try:
        o = json.loads(ln)
        if not (isinstance(o, dict) and o.get("jsonrpc") == "2.0"):
            bad.append(ln[:80])
    except ValueError:
        bad.append(ln[:80])
    if not ln.endswith(b"\n") or ln.count(b"\n") != 1:
        bad.append(b"not one line: " + ln[:60])
check(len(c.all_stdout) > 20 and not bad,
      "every byte on stdout is one JSON-RPC object per line, nothing else",
      f"{bad[:3]}")
check(rc == 0, "closing stdin ends the server cleanly (exit 0)", f"rc={rc}")
check(b"MCP server on stdio" in c.err,
      "the server's own log line went to stderr", f"{c.err[:120]!r}")

# ---------------------------------------------------------------------------
section("4. the app is not running -- the normal first case")
dead = free_closed_port()
d = Child(dead)
t0 = time.perf_counter()
r = d.rpc("initialize", init_req)
first = time.perf_counter() - t0
check(r is not None and first < 3.0,
      f"the server answers initialize with no app running ({first:.2f}s)")
worst = 0.0
for name in mcp.TOOLS:
    args = {}
    if name == "press_key":
        args = {"key": "g"}
    elif name == "click":
        args = {"x": 1, "y": 1}
    t0 = time.perf_counter()
    r = d.call(name, args)
    dt = time.perf_counter() - t0
    worst = max(worst, dt)
    out = (r or {}).get("result", {})
    txt = out.get("content", [{}])[0].get("text", "")
    check(out.get("isError") is True and "not listening" in txt
          and "screenplink.exe --target --introspect" in txt
          and "OFF by default" in txt and "Traceback" not in txt,
          f"'{name}' says the app is not listening and how to start it")
    check(dt < 2.0, f"'{name}' answers 'not running' in under 2 s ({dt:.2f}s)")
check(str(dead) in txt, "the message names the port it tried")
d.close()

# --port=N on the command line, with the environment clear
e = Child(STUB_PORT, extra_args=(f"--port={STUB_PORT}",), env_port=False)
e.rpc("initialize", init_req)
r = e.call("health")
check(not r["result"].get("isError"),
      "--port=N on the command line points the server at the app")
e.close()

# ---------------------------------------------------------------------------
section("5. the dispatcher")
dtree = ast.parse((SRC / "lsa" / "__main__.py").read_text(encoding="utf-8"))
top = [n for n in dtree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
check(all((isinstance(n, ast.ImportFrom) and n.module == "__future__")
          or (isinstance(n, ast.Import) and [a.name for a in n.names] == ["sys"])
          for n in top), "__main__ imports nothing but sys at module level")
_main = next(n for n in dtree.body
             if isinstance(n, ast.FunctionDef) and n.name == "main")
_order = []
for n in _main.body:
    if isinstance(n, ast.If) and "--mcp" in ast.dump(n.test):
        _order.append("mcp-branch")
        check(any(isinstance(x, ast.Return) for x in n.body),
              "the --mcp branch returns (never falls through to the app)")
        check("camera_lab" not in ast.dump(n),
              "the --mcp branch never mentions camera_lab")
    elif isinstance(n, ast.ImportFrom) and "camera_lab" in (n.module or ""):
        _order.append("camera_lab-import")
check(_order == ["mcp-branch", "camera_lab-import"],
      "--mcp is decided BEFORE camera_lab is imported", f"{_order}")

# A windowed exe whose parent supplies no stderr has sys.stderr = None, and
# print(file=None) falls back to sys.stdout -- the protocol stream.
import contextlib, io                                         # noqa: E402
_real_err = sys.stderr
_out = io.StringIO()
try:
    sys.stderr = None
    with contextlib.redirect_stdout(_out):
        mcp.log("this must not reach the protocol stream")
finally:
    sys.stderr = _real_err
check(_out.getvalue() == "", "log() with no stderr writes NOTHING to stdout",
      repr(_out.getvalue()))
_err = io.StringIO()
with contextlib.redirect_stderr(_err):
    mcp.log("hello")
check("hello" in _err.getvalue(), "log() with a stderr still reaches it")

httpd.shutdown()

print("\n--- transcript of a real session against the stub "
      "(long lines truncated) ---")
for ln in TRANSCRIPT:
    print(ln)
print("--- end transcript ---")

print()
if FAILED:
    print(f"{len(FAILED)} OF {CHECKS} CHECKS FAILED:")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print(f"ALL {CHECKS} MCP CHECKS PASSED")
