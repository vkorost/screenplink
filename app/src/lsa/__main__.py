"""`python -m lsa [--mcp | app arguments]` -- and the frozen entry point.

`--mcp` is decided HERE, before anything can import `camera_lab`: that module
builds a QApplication on its second line, so reaching it drags Qt and OpenCV in
and costs seconds a call. The MCP server must answer instantly, including when
the app is not running at all.
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--mcp" in argv:
        from .mcp import main as mcp_main
        return mcp_main([a for a in argv if a != "--mcp"])
    from .tools.camera_lab import main as app_main      # heavy: Qt + OpenCV
    sys.argv = [sys.argv[0]] + argv
    result = app_main()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
