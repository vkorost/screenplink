"""Scoring zones for a figure, as DATA in normalised image coordinates.

The artwork already carries its scoring rings, drawn as closed grey curves over
head and chest. They are bottle-shaped, so no rectangle or ellipse follows them
(measured: best single-shape fits 0.48-0.86). So the zones here are the rings
THEMSELVES: each one a polygon extracted from the printed ink by
`lsa.artwork.rings` and stored in the per-image sidecar (`<name>.zones.json`).
What scores is what is drawn.

PURE: numpy and cv2 only, no Qt.

Coordinates are normalised: x in 0..1 of the image WIDTH, y in 0..1 of the
image HEIGHT, so the same data scores a figure drawn at any size. Hit testing
runs in a space where one unit is one image height (`x` is multiplied by the
image aspect first), so a distance means the same in both directions.

A zone is {"points": int, "name": str, "pts": [[x, y], ...]}. Zones may overlap
(the head dome is inside the 8 curve and outside the 7 curve), and a shot scores
the HIGHEST points among all zones that contain it; outside every zone it is a
miss. A shot within half a stroke of a curve counts as on it, because the ring
is drawn as a line of that width and the visible edge is what the shooter sees.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def compile_zones(zones: list, aspect: float) -> list:
    """Copies of `zones` carrying `contour`: the polygon as a float32 (N,1,2)
    array in unit-height space, ready for `cv2.pointPolygonTest`."""
    out = []
    for z in zones:
        p = np.asarray(z["pts"], np.float32)
        c = np.empty((len(p), 1, 2), np.float32)
        c[:, 0, 0] = p[:, 0] * aspect
        c[:, 0, 1] = p[:, 1]
        out.append(dict(z, contour=c))
    return out


def inside(zone: dict, nx: float, ny: float, aspect: float, margin: float) -> bool:
    """Is the normalised point inside the zone's polygon, or within `margin`
    (a fraction of the image height) outside its edge?"""
    d = cv2.pointPolygonTest(zone["contour"], (float(nx * aspect), float(ny)), True)
    return d >= -margin


def score(nx: float, ny: float, zones: list, aspect: float,
          stroke: float = 0.0) -> tuple[int, "str | None"]:
    """(points, zone name) for the highest-scoring compiled zone containing the
    point; (0, None) outside every zone. `zones` come from `compile_zones`."""
    best: tuple[int, "str | None"] = (0, None)
    for z in zones:
        if z["points"] > best[0] and inside(z, nx, ny, aspect, stroke / 2.0):
            best = (z["points"], z["name"])
    return best


def centre_and_radius(zone: dict, aspect: float) -> tuple[float, float, float]:
    """(cx, cy, r) of the zone in normalised coordinates, r a fraction of the
    image HEIGHT: the polygon's centroid and the distance from it to its
    nearest edge. For `Mode.nearest`, which only logs."""
    c = zone["contour"]
    m = cv2.moments(c)
    if m["m00"] == 0:
        p = c[:, 0, :].mean(axis=0)
        cx, cy = float(p[0]), float(p[1])
    else:
        cx, cy = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    r = abs(cv2.pointPolygonTest(c, (cx, cy), True))
    return cx / aspect, cy, r


def zones_path(name: str, folder: "Path | None" = None) -> Path:
    from .load import targets_dir
    return (folder if folder is not None else targets_dir()) / f"{name}.zones.json"


def load_zones(name: str, folder: "Path | None" = None) -> dict:
    """The sidecar for `name`."""
    return json.loads(zones_path(name, folder).read_text(encoding="utf-8"))
