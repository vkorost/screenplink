"""One command that answers "is the camera actually fit to shoot with?"

THIS IS NOW A FRONT DOOR. The checks live in `src/lsa/doctor.py`, which is also
what ships as `screenplink.exe --doctor`; this script only chooses the camera the
way the development app does (the one remembered in config/settings.local.json)
and hands over. One implementation, two doors, so the tool and the release
cannot drift apart.

The order of the checks is the lesson, and the reasons are in the doctor's
docstring: open, frame rate with the exposure FIXED, does exposure change the
image, noise floor, can the flash be seen, and finally save a frame and LOOK.

    .venv\\Scripts\\python.exe tools\\camera_health.py              the remembered camera
    .venv\\Scripts\\python.exe tools\\camera_health.py --index 1 --no-flash
    .venv\\Scripts\\python.exe tools\\camera_health.py --all        every real camera, ranked

Pictures land in recordings\\ as doctor_cam<N>_view.png and doctor_cam<N>_dark.png
(this tool used to write a single recordings\\camera_health.png).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa import doctor                                   # noqa: E402
from lsa.camera.devices import list_cameras              # noqa: E402
from lsa.settings import AppSettings                     # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a in ("--all", "--index", "--frame") for a in argv):
        # No camera chosen: use the one the app remembers, as this tool always did.
        path = Path(os.environ.get("LSA_SETTINGS") or ROOT / "config" / "settings.local.json")
        settings = AppSettings.load(path)
        resolved = settings.resolve_device(list_cameras())
        index = resolved if isinstance(resolved, int) else getattr(resolved, "index", 0)
        print(f"(no --index: using the camera remembered in {path.name}: index {index})")
        argv += ["--index", str(index)]
    return doctor.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
