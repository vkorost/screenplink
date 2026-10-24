"""Is the camera actually delivering frames? One question, measured, no writes.

Exists for the repeatability gate, which has to decide between launches
whether the next one is worth running. It was deciding that by grepping the
app's log for stall messages, which is a guess about an instrument rather
than a measurement of it -- and it guessed wrong in both directions on
2026-10-15: it aborted a healthy sweep on a shutdown warning that turned out
to be survivable, and earlier it carried on through four launches against a
camera that was already dead.

A wedged Kiyo Pro needs a physical replug. Every launch after it stalls
measures the stall and not the app, so the gate must stop rather than spend
his monitor producing four copies of one failure.

Deliberately writes NO controls. Control traffic is the operation measured to
hang this camera; streaming is not. Opening and reading is safe, and it is
exactly the thing in question.

    python tools\\camera_alive.py [index]     exit 0 if alive, 1 if not
"""

from __future__ import annotations

import os
import sys
import time

# Must be set before cv2 is imported: 73.8s -> 0.43s to open. See lsa/_env.py.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402

# A healthy open on this camera measures 0.4s; a wedged one has measured 5s,
# 7s and 40s. A healthy read returns 50+ frames in three seconds even on auto
# exposure in a dark room; a wedged one returns none at all.
OPEN_SECONDS_MAX = 20.0
FRAMES_MIN = 20
SECONDS = 3.0


def main() -> int:
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t = time.time()
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    opened = time.time() - t
    ok = bad = 0
    t = time.time()
    while time.time() - t < SECONDS:
        got, _frame = cap.read()
        if got:
            ok += 1
        else:
            bad += 1
    cap.release()

    alive = ok >= FRAMES_MIN and opened <= OPEN_SECONDS_MAX
    print(f"camera {index}: open {opened:.2f}s  frames {ok}  failures {bad}  "
          f"{ok / SECONDS:.1f} fps  -> {'ALIVE' if alive else 'WEDGED'}")
    if not alive:
        print("  it needs a physical replug; nothing in software recovers it.")
    return 0 if alive else 1


if __name__ == "__main__":
    raise SystemExit(main())
