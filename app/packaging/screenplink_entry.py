"""The frozen entry point. ONE executable, modes chosen by ARGUMENT.

    screenplink.exe --target        the app (Play.cmd runs this)
    screenplink.exe --mcp           the MCP server (.mcp.json runs this)
    screenplink.exe --doctor --all  the camera doctor (Doctor.cmd runs this,
                                    with its output redirected to a file)

There is deliberately NO dispatch on the executable's own name. An earlier
version looked at `sys.executable` for "screenplink-mcp" / "screenplink-doctor",
because three separate exes were launched with no arguments. With one binary
that is dead code, and worse, a trap: a path that happens to contain those words
would silently change how the app starts, which is invisible until it is on
someone's screen. Whoever launches a mode says so in the arguments, and a bare
`screenplink.exe` is always the app.

ORDER MATTERS. `--doctor` and `--mcp` are decided BEFORE anything can import
`lsa.tools.camera_lab`: that module builds a QApplication on its second line, so
reaching it drags Qt and OpenCV in and costs seconds a call. The MCP server in
particular must answer instantly, including when the app is not running at all.
"""

from __future__ import annotations

import sys


def main() -> int:
    argv = list(sys.argv[1:])
    if "--doctor" in argv:
        # No Qt and no camera_lab: the doctor imports Qt only for the flash
        # check, and only when it runs one.
        from lsa.doctor import main as doctor_main
        return doctor_main([a for a in argv if a != "--doctor"])
    if "--mcp" in argv:
        # Stdlib only, and deliberately imported before anything else can pull
        # in Qt or OpenCV. A tool call must answer in milliseconds, including
        # when the app is not running -- which is the normal first case.
        from lsa.mcp import main as mcp_main
        return mcp_main([a for a in argv if a != "--mcp"])

    from lsa.__main__ import main as app_main
    return app_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
