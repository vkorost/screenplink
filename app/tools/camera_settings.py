"""Open the camera driver's OWN property sheet.

Some of the settings that matter most on this hardware are not reachable
through any `CAP_PROP_*` name OpenCV exposes:

- **Field of view / zoom** on the Razer Kiyo Pro. `CAP_PROP_ZOOM` is accepted
  and ignored -- the readback sits at 100.0 whatever you write -- because the
  FOV (103 / 90 / 80 degrees) is a vendor setting, not a UVC one. If the camera
  cannot see the whole monitor, widening the FOV here is the fix, and it beats
  moving the camera.
- **Low Light Compensation**, which brightens the image automatically and
  fights a manual exposure. It defaults ON and a USB replug resets it.

Both live on the DirectShow property page, which only DirectShow can open.

    .venv\\Scripts\\python.exe tools\\camera_settings.py [index]

The dialog is modal to this process: it blocks until you close it, and the
camera is released afterwards. Nothing else may hold the camera while this
runs.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402

index = int(sys.argv[1]) if len(sys.argv) > 1 else 2

print(f"opening camera {index} on DirectShow (property pages are a DirectShow "
      f"feature; MSMF has no equivalent)...")
cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("could not open the camera. Is another program holding it?")
    raise SystemExit(1)

# A frame first: the property page is a property of a running graph, and on a
# graph that has never delivered anything it can come up empty or not at all.
for _ in range(10):
    cap.read()

print()
print("The driver's own settings window is opening now.")
print("  - Camera Control tab: field of view / zoom, and Low Light Compensation")
print("  - Video Proc Amp tab: brightness, contrast, saturation, white balance")
print()
print("For this project: widen the field of view as far as it goes, and make")
print("sure Low Light Compensation is UNTICKED -- it fights manual exposure.")
print()
print("This window will wait until you close that dialog.")

cap.set(cv2.CAP_PROP_SETTINGS, 1)

print("dialog closed; releasing the camera")
cap.release()
