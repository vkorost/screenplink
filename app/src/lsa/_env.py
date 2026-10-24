"""Environment tweaks that must be applied before OpenCV initialises.

Import this before cv2. `lsa/__init__.py` does, so `import lsa.anything` is
enough; modules that import cv2 at the top of the file do it explicitly.

OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
    Without it, opening a Logitech C910 through Media Foundation takes
    **73.8 seconds**, consistently, every time. With it, 0.43 seconds. Same
    camera, same format, same 30 fps once running -- the entire cost is MSMF
    negotiating hardware transforms during initialisation. Measured on this
    machine 2026-10-13; the C920 benefits too (0.31 s).

    Without this the only alternative was DirectShow, which on both cameras
    ignores the MJPG request, forces YUY2 and delivers 10 fps at 720p -- too
    slow for the multi-frame shot confirmation the detector depends on.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

# Keep OpenCV's own thread pool modest: the capture thread is the only heavy
# consumer and the UI needs the rest of the machine to stay responsive.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "1000")

# QT TEXT MUST NOT BE SUB-PIXEL ANTIALIASED (measured 2026-10-17).
#
# Qt on Windows draws text through DirectWrite with ClearType, which writes
# COLOURED fringes on glyph edges. Pure grey text (red excess 0 by palette)
# measured red excess 49-86 on those edges when rendered through the native
# platform -- offscreen rendering has no sub-pixel text and showed 0, which is
# why every palette check passed while the camera read the text as laser hits
# whenever a layer appeared (the fake shots when 0 is pressed).
#
# The DirectWrite engine ignores QFont.NoSubpixelAntialias; the GDI engine
# honours it (and display/layers.mono() asks for it). The engine can only be
# chosen before QApplication exists, which is what this module is for. Never
# override a platform someone chose on purpose (tests set "offscreen").
if os.name == "nt":
    os.environ.setdefault("QT_QPA_PLATFORM", "windows:fontengine=gdi")
