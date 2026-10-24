"""Camera discovery.

OpenCV can open a camera by index but cannot tell you its name, which makes for
a miserable UI when three of the four devices on this machine are virtual Canon
bridges. pygrabber walks the DirectShow graph and gives us real names in the
same order DirectShow hands them to OpenCV, so CAP_DSHOW index N is the Nth name.

MSMF enumerates separately and is NOT guaranteed to match that ordering, so
names for MSMF are flagged as a guess in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2

# Devices that are software bridges rather than real capture hardware. Opening
# them usually succeeds and then delivers nothing, which looks like a bug.
VIRTUAL_NAME_PATTERNS = (
    r"EOS Webcam",
    r"OBS Virtual",
    r"NVIDIA Broadcast",
    r"Streamlabs",
    r"XSplit",
    r"ManyCam",
    r"Snap Camera",
    r"Virtual Camera",
)


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    is_virtual: bool

    @property
    def label(self) -> str:
        tag = "  [virtual]" if self.is_virtual else ""
        return f"{self.index}: {self.name}{tag}"


def _is_virtual(name: str) -> bool:
    return any(re.search(p, name, re.IGNORECASE) for p in VIRTUAL_NAME_PATTERNS)


def list_cameras() -> list[CameraInfo]:
    """Enumerate DirectShow capture devices in OpenCV index order."""
    try:
        from pygrabber.dshow_graph import FilterGraph

        names = FilterGraph().get_input_devices()
    except Exception:
        # No pygrabber / COM trouble: fall back to opaque indices.
        names = [f"Camera {i}" for i in range(_probe_index_count())]

    return [CameraInfo(i, n, _is_virtual(n)) for i, n in enumerate(names)]


def _probe_index_count(limit: int = 8) -> int:
    """Last-resort probe: how many indices can actually be opened."""
    count = 0
    for i in range(limit):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        cap.release()
        if ok:
            count = i + 1
    return count


def available_backends() -> dict[str, int]:
    """Capture backends this OpenCV build supports, name -> CAP_* constant."""
    out: dict[str, int] = {}
    for b in cv2.videoio_registry.getCameraBackends():
        out[cv2.videoio_registry.getBackendName(b)] = int(b)
    # MSMF first, deliberately. Measured on a Logitech C920 (see CLAUDE.md):
    # DirectShow ignores a FOURCC request, forces YUY2 and delivers 10 fps at
    # 720p, which is below the floor this project needs -- at 10 fps a 100 ms
    # laser pulse occupies a single frame. MSMF honours MJPG and delivers 32.
    ordered = {k: out[k] for k in ("MSMF", "DSHOW") if k in out}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    return ordered


def fourcc_to_str(value: float) -> str:
    """Decode a FOURCC, or say plainly that the backend did not report one.

    MSMF hands back small integers like 22 rather than packed ASCII, which decode
    to control characters and then end up in logs and JSON looking like data
    corruption. Better to admit the backend did not tell us.
    """
    v = int(value)
    if v <= 0:
        return "?"
    chars = [chr((v >> (8 * i)) & 0xFF) for i in range(4)]
    if all(32 <= ord(c) < 127 for c in chars):
        return "".join(chars).strip()
    return f"(not reported: {v})"


def str_to_fourcc(code: str) -> int:
    return int(cv2.VideoWriter_fourcc(*code.ljust(4)[:4]))
