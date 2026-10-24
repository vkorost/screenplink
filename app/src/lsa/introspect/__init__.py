"""In-process introspection for a running Camera Lab. OFF unless asked for.

    .venv\\Scripts\\pythonw.exe -m lsa.tools.camera_lab --target --introspect
    set LSA_INTROSPECT=1          (or LSA_INTROSPECT=8899 for a chosen port)

Then, from any shell on this machine:

    curl -s http://127.0.0.1:8777/health
    curl -s http://127.0.0.1:8777/state
    curl -s http://127.0.0.1:8777/settings
    curl -s -o frame.png http://127.0.0.1:8777/frame
    curl -s "http://127.0.0.1:8777/score?x=900&y=360&box=9"
    curl -s -o score.npy http://127.0.0.1:8777/score.npy
    curl -s -o score.png http://127.0.0.1:8777/score.png
    curl -s "http://127.0.0.1:8777/shots?n=5"
    curl -s http://127.0.0.1:8777/drawn
    curl -s -X POST "http://127.0.0.1:8777/key?key=g"

OFF BY DEFAULT, and the app must behave identically when it is off. That is a
testable claim, not a promise: `port_from` returns 0 when neither the flag nor
the environment variable is present, `install` is the only thing that creates a
thread or a socket, and `tests/test_introspect.py` section 1 asserts a
`CameraLab` built with neither set has `lab.introspect is None` and no
`lsa-introspect` thread in the process.

Phase 2 (NOT built, and it needs the user to register it in his client config)
is a thin MCP wrapper: one MCP tool per endpoint, each one an HTTP GET against
this server, returning the JSON verbatim as the tool result and the two PNG
endpoints as image content. Nothing in the app changes for it, which is why the
endpoint shapes here are flat JSON objects with no cursors and no state --
every call is independently meaningful.
"""

from __future__ import annotations

import os

from .readonly import WriteAttempted                     # noqa: F401
from .server import DEFAULT_PORT, IntrospectServer, NotLocal   # noqa: F401
from .view import LabView


def port_from(args, env=None) -> int:
    """0 means "not asked for". Nothing is started in that case.

    Accepts `--introspect`, `--introspect=PORT`, and `LSA_INTROSPECT` set to
    `1`/`true`/`on`/`yes` or to a port number. An unparseable value is NOT
    silently ignored: it raises, because a flag that quietly does nothing is
    the shape of four checks on this project that could not fail (RULES C1).
    """
    env = os.environ if env is None else env
    for a in (args or ()):
        if a == "--introspect":
            return DEFAULT_PORT
        if a.startswith("--introspect="):
            raw = a.split("=", 1)[1].strip()
            return DEFAULT_PORT if raw in ("", "1", "on", "yes", "true") else int(raw)
    raw = str(env.get("LSA_INTROSPECT", "") or "").strip().lower()
    if not raw or raw in ("0", "off", "no", "false"):
        return 0
    if raw in ("1", "on", "yes", "true"):
        return DEFAULT_PORT
    return int(raw)


class Introspection:
    """What `install` hands back: the server, and the one wiring point."""

    def __init__(self, server: IntrospectServer, view: LabView, bridge, tap) -> None:
        self.server = server
        self.view = view
        self.bridge = bridge
        self.tap = tap

    @property
    def port(self) -> int:
        return self.server.port

    def attach_capture(self, capture) -> None:
        """Start filling the shot ring from this capture thread.

        The app calls this when it opens a camera. The connection is made
        here, inside the package, rather than by the app -- so the one
        property that matters (`record` runs on the GUI thread, not in the
        capture loop) is testable and can be broken by a mutation. It was not,
        when the app made the connection, and the mutation run said so.
        """
        if self.tap is not None:
            self.tap.attach(capture.shot_detected)

    def stop(self) -> None:
        self.server.stop()


def install(lab, port: int = DEFAULT_PORT, host: str = "127.0.0.1",
            bridge=None, tap=None, log=None) -> Introspection:
    """Start the server against `lab`. Never called unless `port_from` said so."""
    if bridge is None or tap is None:
        # Imported here so the package can be used (and tested) without Qt.
        from .gui import GuiBridge, ShotTap
        parent = lab if hasattr(lab, "metaObject") else None
        bridge = GuiBridge(parent) if bridge is None else bridge
        tap = ShotTap(parent) if tap is None else tap
    view = LabView(lab, bridge=bridge, tap=tap, log=log)
    from .handlers import routes_for
    server = IntrospectServer(routes_for(view), host=host, port=port, log=log)
    actual = server.start()
    view.server = server
    if log is not None:
        log(f"introspection server on http://{host}:{actual}  "
            f"(GET /health /state /settings /frame /score /shots /drawn, "
            f"POST /key). Read-only except /key; no camera control writes.")
    return Introspection(server, view, bridge, tap)
