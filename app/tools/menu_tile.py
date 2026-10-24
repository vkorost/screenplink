#!/usr/bin/env python3
"""Bake a staged drawing into a menu tile: 512x512, single-channel grey, peak
at the measured artwork ceiling.

    python tools/menu_tile.py image-staging/exit_menu.jpg menu/exit.png
    python tools/menu_tile.py ... --dry-run        # measure, write nothing

WHY A SCRIPT AND NOT A ONE-OFF EDIT
-----------------------------------
The peak is NOT 71 by definition. It is `artwork.ceiling.figure_level()`, the
smallest p99-of-lit-pixels over everything in `targets/`, and it MOVES when he
adds a darker figure. A tile baked by hand to today's number is wrong the day
the ceiling drops, and nothing would say so. Re-running this rebakes from the
staged original at whatever the ceiling is then.

THE TWO LEVELS THAT MATTER, AND THEY ARE DIFFERENT
--------------------------------------------------
* ceiling (71 today) -- the PEAK any artwork may reach, so a laser dot landing
  on a stroke still scores. See artwork/ceiling.py for the derivation.
* 60 -- the luminance above which a pixel spends the RULES B1b light budget
  (~2% of screen while shooting is possible). `ink` counts pixels above 60,
  which is why `ink` is far smaller than "pixels that are not black".

So the stretch has to put the stroke BODY above 60 while the peak stays at the
ceiling: a narrow window. Mapping the body below 60 gives a tile that is legal,
invisible from the firing line, and reports a flattering ink figure.

A black point is taken from the image rather than assumed, so a faint watermark
or JPEG haze does not survive as ink it never earned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lsa.artwork import ceiling  # noqa: E402

TILE = 512
INK_LEVEL = 60                 # RULES B1b budget threshold
SIXTH_PX = 640                 # a tile at 1/6 of a 3840-wide panel
SCREEN_PX = 3840 * 2160
MARGIN = 0.055                 # of the content box, each side


def content_box(grey: np.ndarray, floor: int) -> tuple[int, int, int, int]:
    """Square box around every pixel above `floor`, padded, clipped to frame."""
    ys, xs = np.where(grey > floor)
    if not len(xs):
        raise SystemExit("nothing above luminance %d in the source" % floor)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    side = max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = grey.shape
    side = min(side, w, h)
    left = int(round(min(max(cx - side / 2, 0), w - side)))
    top = int(round(min(max(cy - side / 2, 0), h - side)))
    return left, top, int(round(side)), int(round(side))


def bake(src: Path, black: int | None, white_pct: float, peak: float):
    grey = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise SystemExit("cannot read %s" % src)

    # Black point: the gap between the drawing and whatever haze sits under it.
    # Otsu over the non-black pixels finds it without being told where it is.
    if black is None:
        sample = grey[grey > 8]
        black = int(cv2.threshold(sample.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

    left, top, bw, bh = content_box(grey, black)
    crop = grey[top:top + bh, left:left + bw]
    small = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA)

    # White point from the stroke body, measured AFTER the downscale, because
    # resampling is what actually thins the strokes.
    strokes = small[small > black]
    if not strokes.size:
        raise SystemExit("the crop is empty above the black point")
    white = float(np.percentile(strokes, white_pct))
    if white <= black:
        raise SystemExit("white point %.0f is not above black point %d" % (white, black))

    out = (small.astype(np.float32) - black) / (white - black)
    out = np.clip(out, 0.0, 1.0) * peak
    out = np.rint(out).astype(np.uint8)
    out[out > peak] = int(peak)
    return out, dict(black=black, white=round(white, 1),
                     crop=[left, top, bw, bh], src_shape=list(grey.shape))


def measure(tile: np.ndarray, path: Path) -> dict:
    ink = int((tile > INK_LEVEL).sum())
    at_sixth = ink * (SIXTH_PX / TILE) ** 2
    return {
        "ink": ink,
        "peak": int(tile.max()),
        "red_excess": 0,                       # single-channel grey, by construction
        "lit_pct_at_sixth": round(100 * at_sixth / SCREEN_PX, 4),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("dest")
    ap.add_argument("--black", type=int, default=None,
                    help="black point; default is Otsu over the non-black pixels")
    ap.add_argument("--white-pct", type=float, default=55.0,
                    help="percentile of the lit pixels that saturates at the ceiling")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    peak = ceiling.figure_level()
    tile, info = bake(Path(a.source), a.black, a.white_pct, peak)
    dest = Path(a.dest)

    print("ceiling (figure_level): %.0f" % peak)
    print("black point:            %d%s"
          % (info["black"], "" if a.black is not None else "  (Otsu)"))
    print("white point:            %.1f  (p%.0f of lit)" % (info["white"], a.white_pct))
    print("source %dx%d -> crop %s -> %dx%d"
          % (info["src_shape"][1], info["src_shape"][0], info["crop"], TILE, TILE))

    if a.dry_run:
        m = measure(tile, dest)
        print("would write %s: ink %d, peak %d, lit %.4f%% at 1/6"
              % (dest, m["ink"], m["peak"], m["lit_pct_at_sixth"]))
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), tile):
        raise SystemExit("failed to write %s" % dest)
    m = measure(tile, dest)

    book = dest.parent / "measurements.json"
    data = json.loads(book.read_text(encoding="utf-8")) if book.exists() else {}
    data[dest.stem] = m
    book.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")

    print("wrote %s" % dest)
    for k, v in m.items():
        print("   %-18s %s" % (k, v))
    print("recorded in %s" % book)
    return 0


if __name__ == "__main__":
    sys.exit(main())
