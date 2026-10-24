"""Extract the printed scoring rings from every figure and write the sidecars.

    python tools/zones_author.py --author     write targets/<name>.zones.json
    python tools/zones_author.py --preview    overlays to tests/_out/zones/
    python tools/zones_author.py --report     print the verification table

Writes only `targets/*.zones.json` and `tests/_out/zones/`; the artwork is never
touched. Plan 02 step 16, revised 2026-10-17: the first version fitted
rectangles and ellipses and could verify only 6 of 12 figures on one ring,
because the printed rings are bottle-shaped. This reads the rings themselves.

HOW
  1. TEMPLATE. thug01, thug02 and thug03 are the same printed target under
     three different drawings, so the ink they share (their AND) is the printed
     target and nothing else: the four ring curves and the digits. Rough
     polylines (hand-read control points, below) are snapped onto that clean
     ink (`rings.snap`): C1 the outer silhouette (7 | outside), C2 the bottle
     (head dome + chest, 8), C3 the oval (9) and C4 the inner rounded
     rectangle (10).
  2. PER IMAGE. The template is laid over the image with a similarity
     transform found by search (`rings.register`) -- it lands at scale 1.000 and
     offset 0-2 px on all twelve -- and each polygon is snapped again onto THAT
     image's own ink. THE INK LEVEL IS PER IMAGE (0.55 x its 99th-percentile
     luminance, 36-64 across the twelve), and it is a NORMALISATION, not a
     loosening: the printed rings are drawn at different brightness in different
     files. Measured (share of a ring's outline within 3 px of ink, mean over the
     four rings): at a fixed cut of 70 it collapses to 4.6%, 30.4%, 59.4%, 76.2%,
     82.5%, 84.7% and 88.1% for thug06, thug05, axman01, predator01, zombie02,
     alien01 and zombie01 (SEVEN of the twelve; per ring the reviewer measured as
     low as 0.0%), while each image's own level gives 98.0-99.1% on all twelve.
     Not just thug06: a fixed level would have scored seven figures against ink it
     could not see.
  3. VERIFY, against the ink the polygon came from. Per polygon: distance from
     its vertices to the nearest ink pixel (median, p95, max, in pixels of a
     2400-pixel-high image) and how far snapping had to move it from the
     registered template (mean, max). A figure is scored only if all four
     polygons pass; otherwise it is written with `scored: false` and the
     figure is excluded from every random draw (`lsa.artwork.catalog`).

The chord of C1 across the neck is not printed (the shoulder lines stop at the
neck); it is kept straight and excluded from the ink check. IT DOES DECIDE SHOTS.
An earlier version of this comment said it lies inside C2 and so never matters;
that was measured false: the 7-region's top edge sits at y 0.322-0.325 of the image
height and the chord is at 0.323, with 161-184 grid samples scoring 7 within 1% of
it, so across the neck the chord IS the boundary (7 below it, nothing above). The
placement is deliberate -- it follows the line the shoulders stop on, i.e. the
collar -- but it is an inferred edge, not a printed one.

SIZE. Each ring is stored after simplification (`SIMPLIFY_EPS_PX`) as 29 to 79
vertices, not hundreds: the dense polygon is what was verified against the ink, and
simplification moves no vertex's curve by more than that tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2                                            # noqa: E402
import numpy as np                                    # noqa: E402

from lsa.artwork import catalog, load, rings as R     # noqa: E402

TEMPLATE_SOURCES = ("thug01", "thug02", "thug03")     # share one printed target
LEVEL_OF_P99 = 0.55
SNAP_REACH = 14
STEP_PX = 6.0
SIMPLIFY_EPS_PX = 1.0
ANCHOR = 0.25            # pull toward the registered template, per unit of reach

# Verification tolerances, in pixels of the 2400-px-high image.
MAX_P95_PX = 3.0
MAX_MAX_PX = 12.0
MAX_DEV_MEAN_PX = 3.0
MAX_DEV_MAX_PX = SNAP_REACH - 0.5      # never pinned at the search limit

H0 = 2400

# The printed digit row across the chest, "7 8 9 10 9 8 7", read off thug01 at
# half scale (x 205 258 303 400 490 538 600, y 600) and doubled. Each digit sits
# in the band it names, so the extracted rings must score 7 8 9 10 9 8 7 at
# those points: the printed numbers are an independent check on the polygons,
# which were fitted to line positions and know nothing of digits.
DIGIT_ROW = ((410, 1200, 7), (516, 1200, 8), (606, 1200, 9), (800, 1200, 10),
             (980, 1200, 9), (1076, 1200, 8), (1200, 1200, 7))
POINTS = {"C1": 7, "C2": 8, "C3": 9, "C4": 10}


def _h(*p):                      # control points read at half scale
    return np.array([(x * 2.0, y * 2.0) for x, y in p], float)


# Rough prior polylines for thug01's AND image, hand-read from an overlay.
C1_PRIOR = _h((190, 392), (300, 388), (405, 388), (505, 388), (610, 390), (625, 410),
              (627, 880), (615, 922), (600, 940), (430, 1078), (415, 1080),
              (400, 1078), (205, 935), (188, 885), (185, 420))
C1_FROZEN = np.zeros(len(C1_PRIOR), bool)
C1_FROZEN[1:4] = True            # the chord across the neck
C2_PRIOR = _h((400, 152), (470, 165), (500, 200), (515, 250), (510, 310), (495, 360),
              (500, 385), (520, 410), (545, 440), (558, 480), (560, 560), (558, 800),
              (545, 850), (515, 885), (470, 910), (415, 922), (360, 912), (305, 885),
              (260, 855), (238, 820), (233, 750), (232, 560), (238, 480), (255, 430),
              (290, 402), (305, 390), (300, 375), (295, 330), (295, 260), (305, 210),
              (330, 175), (365, 158))
C3_PRIOR = np.array([(570, 1060), (600, 985), (660, 915), (750, 885), (840, 878),
                     (930, 900), (990, 950), (1025, 1030), (1032, 1100), (1032, 1330),
                     (1020, 1400), (985, 1465), (930, 1510), (860, 1538), (830, 1543),
                     (760, 1535), (690, 1505), (620, 1455), (582, 1400), (570, 1340)],
                    float)


def _rounded_rect(x0, y0, x1, y1, r, n=120):
    import math
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0),
                       (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(n // 4 + 1):
            a = math.radians(a0 + 90 * k / (n // 4))
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


C4_PRIOR = _rounded_rect(658, 996, 908, 1328, 90)
PRIORS = (("C1", C1_PRIOR, C1_FROZEN), ("C2", C2_PRIOR, None),
          ("C3", C3_PRIOR, None), ("C4", C4_PRIOR, None))


def ink_level(grey: np.ndarray) -> int:
    return max(20, int(LEVEL_OF_P99 * load.ink_level(grey)))


def build_template() -> dict:
    """Snap the priors onto the ink shared by the template sources."""
    m = None
    for n in TEMPLATE_SOURCES:
        g = load.load_grey(catalog.path_for(n))
        ink = (g > ink_level(g)).astype(np.uint8)
        if m is None:
            m = ink > 0
        else:
            m &= cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
    score = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.6)
    out = {}
    for name, prior, fro in PRIORS:
        reach = 40 if name == "C3" else 30
        poly, f, _ = R.snap(prior, score, reach=reach, step=STEP_PX, frozen=fro)
        out[name] = (poly, f)
    return out


def _simplify(poly: np.ndarray, frozen: np.ndarray):
    c = poly.astype(np.float32).reshape(-1, 1, 2)
    s = cv2.approxPolyDP(c, SIMPLIFY_EPS_PX, True).reshape(-1, 2).astype(np.float64)
    d = np.hypot(s[:, None, 0] - poly[None, :, 0], s[:, None, 1] - poly[None, :, 1])
    return s, frozen[np.argmin(d, axis=1)]


def extract(name: str, template: dict) -> dict:
    """The verified polygons of one image, or the reason it failed."""
    g = load.load_grey(catalog.path_for(name))
    h, w = g.shape
    lvl = ink_level(g)
    dist = R.ink_distance(g, lvl)
    ink = (g > lvl).astype(np.uint8)
    thick = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    pts = [template[k][0][~template[k][1]] for k in POINTS]
    d2 = cv2.resize(dist, None, fx=0.5, fy=0.5)
    s, tx, ty, _ = R.register([p * 0.5 for p in pts], d2, cap=10.0)
    s, tx, ty, reg_cost = R.register(pts, dist, cap=14.0, scales=[s])
    score = R.stroke_score(np.clip(g.astype(np.float32) * (120.0 / (lvl / LEVEL_OF_P99)),
                                   0, 255).astype(np.uint8))
    result = {"image": f"{name}.jpg", "ink_level": lvl,
              "registration": {"scale": round(s, 4), "tx_px": round(tx, 1),
                               "ty_px": round(ty, 1), "mean_dist_px": round(reg_cost, 3)},
              "zones": [], "verify": {}, "scored": True, "why_not": []}
    for k, pts_ in POINTS.items():
        reg_poly = R.transform(template[k][0], s, tx, ty, w / 2.0, h / 2.0)
        poly, fro, off = R.snap(reg_poly, score, reach=SNAP_REACH, step=STEP_PX,
                                frozen=template[k][1], anchor=ANCHOR)
        med, p95, mx = R.measure(poly, dist, skip=fro)
        dev = np.abs(off[~fro])
        stroke = 2.0 * float(np.median(R._sample(thick, poly[~fro])))
        simp, sfro = _simplify(poly, fro)
        v = {"median_px": round(med, 2), "p95_px": round(p95, 2), "max_px": round(mx, 2),
             "template_dev_mean_px": round(float(dev.mean()), 2),
             "template_dev_max_px": round(float(dev.max()), 2),
             "stroke_px": round(stroke, 1), "vertices": int(len(simp)),
             "synthetic_vertices": int(sfro.sum())}
        bad = []
        if p95 > MAX_P95_PX:
            bad.append(f"p95 {p95:.1f}px > {MAX_P95_PX}")
        if mx > MAX_MAX_PX:
            bad.append(f"max {mx:.1f}px > {MAX_MAX_PX}")
        if dev.mean() > MAX_DEV_MEAN_PX:
            bad.append(f"template deviation mean {dev.mean():.1f}px > {MAX_DEV_MEAN_PX}")
        if dev.max() > MAX_DEV_MAX_PX:
            bad.append(f"template deviation max {dev.max():.1f}px at the search limit")
        v["ok"] = not bad
        result["verify"][k] = v
        if bad:
            result["scored"] = False
            result["why_not"].append(f"{k}: " + "; ".join(bad))
        result["zones"].append({"points": pts_, "name": str(pts_),
                                "pts": [[round(float(x) / (w - 1), 5),
                                         round(float(y) / (h - 1), 5)] for x, y in simp]})
    result["zones"].sort(key=lambda z: -z["points"])
    # The printed digits, read back through the extracted rings.
    from lsa.artwork import zones as Z0
    asp0 = w / h
    cz0 = Z0.compile_zones(result["zones"], asp0)
    stroke0 = float(np.median([v["stroke_px"] for v in result["verify"].values()])) / (h - 1)
    digits = []
    for dx, dy, want in DIGIT_ROW:
        qx, qy = R.transform(np.array([[dx, dy]], float), s, tx, ty, w / 2.0, h / 2.0)[0]
        got = Z0.score(qx / (w - 1), qy / (h - 1), cz0, asp0, stroke0)[0]
        digits.append([want, got])
    result["digit_row"] = digits
    if any(want != got for want, got in digits):
        result["scored"] = False
        result["why_not"].append("the printed digit row 7 8 9 10 9 8 7 reads back as "
                                 + " ".join(str(g) for _, g in digits))
    # Nesting, on the data: 10 in 9 in 8, and the chest part of 8 inside 7.
    if result["scored"]:
        from lsa.artwork import zones as Z
        asp = w / h
        cz = {z["name"]: z for z in Z.compile_zones(result["zones"], asp)}
        stroke_n = np.median([v["stroke_px"] for v in result["verify"].values()]) / (h - 1)
        pairs = [("10", "9"), ("9", "8")]
        for inner, outer in pairs:
            for x, y in np.asarray(next(z for z in result["zones"] if z["name"] == inner)["pts"]):
                if not Z.inside(cz[outer], x, y, asp, stroke_n):
                    result["scored"] = False
                    result["why_not"].append(f"{inner} is not inside {outer}")
                    break
        chord_y = C1_PRIOR[1, 1] / (H0 - 1)
        for x, y in np.asarray(next(z for z in result["zones"] if z["name"] == "8")["pts"]):
            if y > chord_y + 0.01 and not Z.inside(cz["7"], x, y, asp, stroke_n):
                result["scored"] = False
                result["why_not"].append("the chest part of 8 is not inside 7")
                break
    result["stroke"] = round(float(np.median([v["stroke_px"] for v in
                                              result["verify"].values()])) / (h - 1), 6)
    result["aspect"] = w / h
    result["note"] = ("Zones are the printed ring curves extracted from the ink. "
                      "10 = inside C4, 9 = inside C3, 8 = inside C2 (head dome and "
                      "chest), 7 = inside C1 (chest). The head carries a printed '9' "
                      "and a printed '7' but no printed ring for either: inside the "
                      "dome scores 8 and outside it scores nothing, because only "
                      "drawn curves score. The top edge of 7 across the neck is an "
                      "inferred chord (not printed) and does decide shots there.")
    return result


def author_all() -> dict:
    tdir = load.targets_dir()
    template = build_template()
    table = {}
    for name in catalog.all_names():
        res = extract(name, template)
        (tdir / f"{name}.zones.json").write_text(json.dumps(res, indent=1),
                                                 encoding="utf-8")
        table[name] = res
    stale = tdir / "zones.template.json"
    if stale.exists():
        stale.unlink()         # the analytic-shape template of the first version
    return table


def report(table: dict) -> None:
    worst = {"p95": 0.0, "max": 0.0, "dev_mean": 0.0, "dev_max": 0.0}
    for name, r in table.items():
        parts = []
        for k, v in r["verify"].items():
            parts.append(f"{k} p95 {v['p95_px']:.1f} max {v['max_px']:.1f} "
                         f"dev {v['template_dev_mean_px']:.1f}/{v['template_dev_max_px']:.1f}")
            worst["p95"] = max(worst["p95"], v["p95_px"])
            worst["max"] = max(worst["max"], v["max_px"])
            worst["dev_mean"] = max(worst["dev_mean"], v["template_dev_mean_px"])
            worst["dev_max"] = max(worst["dev_max"], v["template_dev_max_px"])
        reg = r["registration"]
        print(f"{name:11s} lvl {r['ink_level']:2d} reg s={reg['scale']:.3f} "
              f"t=({reg['tx_px']:.0f},{reg['ty_px']:.0f}) | " + " | ".join(parts)
              + f" -> {'SCORED' if r['scored'] else 'EXCLUDED ' + str(r['why_not'])}")
    print(f"worst over all figures and rings: p95 {worst['p95']:.2f}px, max "
          f"{worst['max']:.2f}px, template deviation mean {worst['dev_mean']:.2f}px "
          f"max {worst['dev_max']:.2f}px (of a 2400px-high image)")


def preview() -> None:
    out_dir = ROOT / "tests" / "_out" / "zones"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = {"7": (0, 0, 255), "8": (255, 255, 0), "9": (0, 255, 255), "10": (0, 255, 0)}
    for name in catalog.all_names():
        g = load.load_grey(catalog.path_for(name))
        info = json.loads((load.targets_dir() / f"{name}.zones.json").read_text())
        h, w = g.shape
        img = cv2.cvtColor(np.clip(g.astype(int) * 2, 0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
        for z in info["zones"]:
            p = R.to_px(np.asarray(z["pts"]), w, h)
            cv2.polylines(img, [np.round(p).astype(np.int32)], True, cols[z["name"]], 4)
        cv2.imwrite(str(out_dir / f"{name}.png"),
                    cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_AREA))
    print(f"previews in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    tbl = None
    if a.author:
        tbl = author_all()
    if a.report or a.author:
        if tbl is None:
            tbl = {n: json.loads((load.targets_dir() / f"{n}.zones.json").read_text())
                   for n in catalog.all_names()}
        report(tbl)
    if a.preview:
        preview()
