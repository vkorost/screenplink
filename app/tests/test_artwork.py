"""The figure artwork: loader, scoring zones, the randomness rules, the sprite.

Plan 02 steps 15, 16, 17 and 17a (zones revised 2026-10-17: polygons from the ink). No camera, no display (offscreen Qt only for
the sprite section). Reads `targets/` and WRITES NOTHING to it; the whole run
is fingerprinted (`tests/owned_files.py` fingerprints `targets/` too).
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np                                          # noqa: E402

from owned_files import OwnedFiles                          # noqa: E402

owned = OwnedFiles()

from lsa.artwork import catalog, load, zones as Z          # noqa: E402

checks: list[bool] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    checks.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


NAMES = catalog.names()
SW, SH = 3840, 2160

# The coordinator's measurement, as data (pixels above luminance 60 as a percent
# of a 3840x2160 screen, one figure at 40/60/80/100% of the screen height). A
# changed file or a changed resampler moves these.
LIT_TABLE = {
    "alien01": (0.50, 1.30, 2.42, 3.87), "axman01": (0.35, 0.96, 1.83, 2.95),
    "predator01": (0.53, 1.39, 2.63, 4.21), "soldier01": (0.67, 1.60, 2.90, 4.57),
    "thug01": (0.55, 1.28, 2.29, 3.59), "thug02": (0.52, 1.21, 2.19, 3.43),
    "thug03": (0.50, 1.19, 2.15, 3.37), "thug04": (0.42, 1.04, 1.92, 3.04),
    "thug05": (0.31, 0.85, 1.64, 2.66), "thug06": (0.16, 0.45, 0.90, 1.48),
    "zombie01": (0.44, 1.10, 2.04, 3.25), "zombie02": (0.40, 1.06, 2.01, 3.22),
}
RED_AS_STORED = {"thug02": 6, "thug01": 8, "thug04": 8, "thug05": 8, "alien01": 8,
                 "soldier01": 8, "zombie01": 8, "zombie02": 8, "thug03": 10,
                 "thug06": 10, "axman01": 12, "predator01": 26}


def main() -> int:
    print("\n1. the artwork is all there, and the loader imports no Qt")
    # NOT A COUNT. This read "len(NAMES) == 12" and would have failed the
    # whole suite the moment he added a figure -- which he does often, and
    # did on 2026-10-18 (soldier01 out, four new characters in). A check
    # that breaks because he did some work teaches him to ignore checks.
    # The PROPERTY is that the catalogue is non-empty, sorted, and that
    # every figure in it is usable; the number is not a property.
    check(f"{len(NAMES)} figures on disk, sorted",
          len(NAMES) >= 6 and NAMES == sorted(NAMES), str(NAMES))
    out = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, r'%s'); import lsa.artwork.load, "
         "lsa.artwork.zones, lsa.artwork.catalog; "
         "print('PySide6' in sys.modules, 'PySide6.QtGui' in sys.modules)"
         % (ROOT / "src")], capture_output=True, text=True)
    check("lsa.artwork imports no Qt (the CV core stays portable)",
          out.stdout.split() == ["False", "False"], out.stdout + out.stderr)

    print("\n2. every file is forced to true grey, and the conversion is load-bearing")
    worst_after, worst_stored = 0, 0
    for n in NAMES:
        path = catalog.path_for(n)
        stored = load.red_excess_of_file(path)
        g = load.load_grey(path)
        # what the app draws from it: one byte copied into R, G and B
        rgb = np.stack([g, g, g], axis=-1).astype(np.int16)
        after = int((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])).max())
        worst_after = max(worst_after, after)
        worst_stored = max(worst_stored, stored)
        # THE PROPERTY IS "0 AFTER LOAD", not the stored value. This read
        # `stored == RED_AS_STORED[n]` against a table keyed by filename,
        # so adding a figure raised KeyError and took the whole suite with
        # it. The stored numbers are kept as a REGRESSION ANCHOR for the
        # files they were measured on -- predator01 at 26 is the fixture
        # that proves the loader neutralises badly coloured art -- and a
        # figure he adds later is simply measured, not rejected.
        expect = RED_AS_STORED.get(n)
        check(f"{n}: one uint8 channel, red excess 0 after load "
              f"(as stored {stored}"
              + (f", expected {expect})" if expect is not None
                 else ", not previously measured)"),
              g.ndim == 2 and g.dtype == np.uint8 and after == 0
              and (expect is None or stored == expect))
    check("predator01 is the fixture: as stored it is over three times the limit of 8",
          RED_AS_STORED["predator01"] == 26 and worst_stored == 26 and worst_after == 0)

    print("\n3. lit area matches the measured table (+-0.10 percentage points)")
    worst_dev = 0.0
    for n in NAMES:
        g = load.load_grey(catalog.path_for(n))
        got = [load.lit_fraction(g, SW, SH, f) * 100 for f in (0.4, 0.6, 0.8, 1.0)]
        # A REGRESSION ANCHOR for the figures it was measured on, and a
        # MEASUREMENT for one he adds later. Keyed by filename, this raised
        # KeyError on a new figure and took the suite with it -- the same
        # fault as RED_AS_STORED above. The budget itself is enforced in
        # section 4, which derives each figure's size cap from its own lit
        # fraction and needs no table.
        want = LIT_TABLE.get(n)
        if want is None:
            check(f"{n}: " + " ".join(f"{v:.2f}" for v in got) + "  (new, measured)",
                  all(0.0 < v < 3.0 for v in got),
                  "add this row to LIT_TABLE to pin it as a regression anchor")
            continue
        dev = max(abs(a - b) for a, b in zip(got, want))
        worst_dev = max(worst_dev, dev)
        check(f"{n}: " + " ".join(f"{v:.2f}" for v in got), dev <= 0.10,
              f"worst deviation {dev:.3f} pp")
    print(f"         worst deviation over the {len(LIT_TABLE)} pinned "
          f"figures: {worst_dev:.3f} pp")

    print("\n4. the size cap is derived per image, from its own lit fraction")
    caps = {}
    for n in NAMES:
        g = load.load_grey(catalog.path_for(n))
        h = load.max_height_for_budget(g, SW, SH, 0.015)
        lit = load.lit_fraction(g, SW, SH, h, background=load.BACKGROUND)
        caps[n] = h
        check(f"{n}: cap {h:.3f} of screen height lights {lit * 100:.3f}% over the "
              f"background (<= 1.5%), ink p99 {load.ink_level(g)}",
              lit <= 0.015 + 1e-9 and load.MIN_HEIGHT_FRAC <= h <= load.MAX_HEIGHT_FRAC)
    check("the caps differ per image (3.2x in lit area at one size)",
          max(caps.values()) - min(caps.values()) > 0.12,
          f"{min(caps.values()):.3f} .. {max(caps.values()):.3f}")
    check("a looser budget gives a bigger figure, a tighter one a smaller",
          load.max_height_for_budget(load.load_grey(catalog.path_for("thug01")), SW, SH, 0.01)
          < caps["thug01"]
          < load.max_height_for_budget(load.load_grey(catalog.path_for("thug01")), SW, SH, 0.02))

    print("\n5. scoring zones: polygons, point-in-polygon, the margin")
    aspect = 0.75
    sq_out = dict(points=8, name="8", pts=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
    sq_in = dict(points=10, name="10", pts=[[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
    cz = Z.compile_zones([sq_in, sq_out], aspect)
    check("inside the inner polygon scores its points, between the two the outer's, "
          "outside scores nothing",
          Z.score(0.5, 0.5, cz, aspect) == (10, "10")
          and Z.score(0.3, 0.5, cz, aspect) == (8, "8")
          and Z.score(0.1, 0.5, cz, aspect) == (0, None))
    stroke = 0.01
    check("a point within half a stroke OUTSIDE an edge counts as on it; beyond that, no",
          Z.score(0.2 - 0.004 / aspect, 0.5, cz, aspect, stroke)[0] == 8
          and Z.score(0.2 - 0.02 / aspect, 0.5, cz, aspect, stroke)[0] == 0)
    check("a concave (bottle) polygon is respected: a point in the notch is outside",
          Z.score(0.55, 0.3, Z.compile_zones([dict(points=7, name="7", pts=[
              [0.2, 0.2], [0.5, 0.2], [0.5, 0.5], [0.6, 0.5], [0.6, 0.2], [0.8, 0.2],
              [0.8, 0.8], [0.2, 0.8]])], aspect), aspect)[0] == 0)
    cx_, cy_, r_ = Z.centre_and_radius(cz[0], aspect)
    check("centre and radius of the inner square: its centre, and the distance to its nearest edge (0.2 x 0.75 / 2 in unit-height space)",
          abs(cx_ - 0.5) < 1e-3 and abs(cy_ - 0.5) < 1e-3 and abs(r_ - 0.075) < 1e-3,
          f"{cx_:.3f} {cy_:.3f} {r_:.3f}")

    print("\n6. the zone data is the printed rings, verified against the ink")
    info_all = {n: Z.load_zones(n) for n in NAMES}
    check(f"every image has a zones file and every one is scored "
          f"(all {len(NAMES)} verified)",
          len(info_all) == len(NAMES) and all(i["scored"] for i in info_all.values()),
          str({n: i["why_not"] for n, i in info_all.items() if not i["scored"]}))
    import importlib.util
    spec = importlib.util.spec_from_file_location("zones_author", ROOT / "tools" / "zones_author.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    from lsa.artwork import rings as RG
    worst = {"7": 1.0, "8": 1.0, "9": 1.0, "10": 1.0}
    worst_p95 = 0.0
    shifted_best = 0.0
    for n in NAMES:
        g = load.load_grey(catalog.path_for(n))
        h, w = g.shape
        dist = RG.ink_distance(g, A.ink_level(g))
        info = info_all[n]
        names_ = [z["name"] for z in info["zones"]]
        check(f"{n}: four rings 10, 9, 8, 7, each a polygon of at least 20 vertices",
              names_ == ["10", "9", "8", "7"] and all(len(z["pts"]) >= 20 for z in info["zones"]))
        for z in info["zones"]:
            p = RG.resample_closed(RG.to_px(np.array(z["pts"]), w, h), 4.0)
            d = RG._sample(dist, p)
            cov = float((d <= 3.0).mean())
            worst[z["name"]] = min(worst[z["name"]], cov)
            worst_p95 = max(worst_p95, float(np.percentile(d, 95)))
            need = 0.92 if z["name"] == "7" else 0.96      # 7 carries the unprinted neck chord
            check(f"{n}: ring {z['name']} lies on the ink: {cov * 100:.1f}% of its outline within "
                  f"3 px of stroke (need {need * 100:.0f}%)", cov >= need)
            sh = p.copy()
            sh[:, 0] += 0.05 * w
            shifted_best = max(shifted_best, float((RG._sample(dist, sh) <= 3.0).mean()))
        # the printed digits, read back through the extracted rings
        cz_ = Z.compile_zones(info["zones"], info["aspect"])
        got = []
        for dx, dy, want in A.DIGIT_ROW:
            got.append((want, Z.score(dx / (w - 1), dy / (h - 1), cz_, info["aspect"],
                                      info["stroke"])[0]))
        check(f"{n}: the printed digit row 7 8 9 10 9 8 7 reads back as "
              f"{' '.join(str(g_) for _, g_ in got)}", all(a == b for a, b in got))
        # nesting, on the data
        zz = {z["name"]: z for z in cz_}
        nest_ok = True
        for inner, outer in (("10", "9"), ("9", "8")):
            for x, y in np.asarray(next(z for z in info["zones"] if z["name"] == inner)["pts"]):
                nest_ok &= Z.inside(zz[outer], x, y, info["aspect"], info["stroke"])
        check(f"{n}: 10 lies inside 9 and 9 inside 8", nest_ok)
    print(f"         worst ink coverage per ring over all twelve: "
          f"{ {k: round(v, 3) for k, v in worst.items()} }; worst p95 distance to ink "
          f"{worst_p95:.2f} px; best shifted-by-0.05 coverage {shifted_best:.3f}")
    check("RED PROOF: every ring shifted by dx=0.05 falls below 80% coverage on every image "
          "(so the coverage check can fail)", shifted_best < 0.80, f"{shifted_best:.3f}")
    # a ring displaced in the data is caught by the digit row too
    g1 = info_all["thug01"]
    moved = [dict(z, pts=[[x + 0.05, y] for x, y in z["pts"]]) for z in g1["zones"]]
    cz_m = Z.compile_zones(moved, g1["aspect"])
    got_m = [Z.score(dx / 1791, dy / 2399, cz_m, g1["aspect"], g1["stroke"])[0]
             for dx, dy, _ in A.DIGIT_ROW]
    check("RED PROOF: the digit-row check fails for displaced rings", got_m != [7, 8, 9, 10, 9, 8, 7],
          str(got_m))
    # scoring the head, stated plainly
    cz1 = Z.compile_zones(g1["zones"], g1["aspect"])
    head8 = Z.score(800 / 1791, 350 / 2399, cz1, g1["aspect"], g1["stroke"])[0]
    head9 = Z.score(820 / 1791, 530 / 2399, cz1, g1["aspect"], g1["stroke"])[0]
    above = Z.score(800 / 1791, 255 / 2399, cz1, g1["aspect"], g1["stroke"])[0]
    check("the printed '8' in the head dome scores 8", head8 == 8, str(head8))
    check("KNOWN LIMITATION, recorded not hidden: the head carries a printed '9' and '7' "
          "but no printed ring for either, so the face scores 8 and above the dome nothing",
          head9 == 8 and above == 0, f"face {head9}, above the dome {above}")
    check("the author tool's gates are stated, and the data satisfies them",
          A.MAX_P95_PX == 3.0 and all(
              v["ok"] for i in info_all.values() for v in i["verify"].values()))

    print("\n6b. a figure that cannot be judged is never drawn")
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="lsa_cast_"))
    try:
        for n in ("thug01", "thug02", "thug03"):
            shutil.copy(catalog.path_for(n), tmp / f"{n}.jpg")
        shutil.copy(load.targets_dir() / "thug01.zones.json", tmp / "thug01.zones.json")
        bad = json.loads((load.targets_dir() / "thug02.zones.json").read_text())
        bad["scored"] = False
        bad["why_not"] = ["C3: test"]
        (tmp / "thug02.zones.json").write_text(json.dumps(bad))
        # thug03 has no zones file at all: "a thirteenth image he drops in tomorrow"
        check("all-names lists every image, names() only the verified ones",
              catalog.all_names(tmp) == ["thug01", "thug02", "thug03"]
              and catalog.names(tmp) == ["thug01"], str(catalog.names(tmp)))
        ex = catalog.excluded(tmp)
        check("the excluded figures are named with their reasons",
              set(ex) == {"thug02", "thug03"} and "did not verify" in ex["thug02"]
              and "no zones file" in ex["thug03"], str(ex))
        rng_ = random.Random(3)
        drawn, prev = set(), None
        for _ in range(500):
            prev = catalog.pick_figure(rng_, prev, catalog.names(tmp))
            drawn.add(prev)
        check("a 500-draw random sequence over the cast never includes an excluded figure "
              "(a one-figure cast does not even exclude its own repeat)",
              drawn == {"thug01"})
        from PySide6.QtWidgets import QApplication as _QA
        _QA.instance() or _QA([])
        from lsa.modes.figure import Figure
        refused = []
        for n in ("thug02", "thug03"):
            try:
                Figure(n, 3840, 2160, 0.015, tmp)
            except (ValueError, OSError) as exc:      # OSError: no zones file at all
                refused.append(str(exc))
        check("constructing an excluded figure is refused, not silently drawn",
              len(refused) == 2 and "never drawn" in refused[0] and "never drawn" in refused[1], str(refused))
        check("the real folder excludes nothing today", catalog.excluded() == {},
              str(catalog.excluded()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n7. the randomness rules (seeded, so deterministic by construction)")
    rng = random.Random(1)
    prev, seq, counts = None, [], {n: 0 for n in NAMES}
    repeats = 0
    for _ in range(12000):
        n = catalog.pick_figure(rng, prev, NAMES)
        if n == prev:
            repeats += 1
        counts[n] += 1
        seq.append(n)
        prev = n
    lo, hi = min(counts.values()) / 12000, max(counts.values()) / 12000
    check("pick_figure: never the same figure twice in a row over 12,000 draws",
          repeats == 0, f"{repeats} repeats")
    _uniform = 1.0 / len(NAMES)
    _lo, _hi = _uniform * 0.75, _uniform * 1.25
    check(f"pick_figure: every figure drawn within 25% of uniform "
          f"({_lo * 100:.1f}%-{_hi * 100:.1f}%, uniform is {_uniform * 100:.1f}% "
          f"over {len(NAMES)} figures)",
          _lo <= lo and hi <= _hi, f"{lo * 100:.2f}% .. {hi * 100:.2f}%")
    rng2 = random.Random(1)
    seq2, prev = [], None
    for _ in range(12000):
        prev = catalog.pick_figure(rng2, prev, NAMES)
        seq2.append(prev)
    check("the same seed gives the same sequence", seq == seq2)
    rng3 = random.Random(2)
    seq3, prev = [], None
    for _ in range(200):
        prev = catalog.pick_figure(rng3, prev, NAMES)
        seq3.append(prev)
    check("a different seed gives a different sequence", seq3 != seq[:200])
    # the check can see a repeat: drop the exclusion and one appears at once
    rng4 = random.Random(1)
    prev, saw = None, 0
    for _ in range(1000):
        n = NAMES[rng4.randrange(len(NAMES))]
        saw += n == prev
        prev = n
    check("the no-repeat check is able to fail: without the exclusion repeats appear",
          saw > 20, f"{saw} repeats in 1000")
    # not a bag: after eleven draws a figure can still be drawn again before all twelve
    seen_gap = max(len(set(seq[i:i + 11])) for i in range(0, 600))
    check("no shuffled bag: eleven consecutive draws need not be eleven different figures",
          any(len(set(seq[i:i + 11])) < 11 for i in range(0, 600)), f"{seen_gap}")

    class _Field:
        def __init__(self, l, t, r, b):
            self._l, self._t, self._r, self._b = l, t, r, b

        def left(self): return self._l
        def right(self): return self._r
        def top(self): return self._t
        def bottom(self): return self._b

        def center(self):
            class P:
                pass
            p = P()
            p.x = lambda s=self: (s._l + s._r) / 2
            p.y = lambda s=self: (s._t + s._b) / 2
            return p

    fld = _Field(0, 160, 3840, 1980)
    rngp = random.Random(1)
    prev_c, bad_move, bad_in = None, 0, 0
    hw, hh = 500.0, 700.0
    inner_diag = math.hypot(3840 - 2 * hw, 1820 - 2 * hh)
    corners = [(hw, 160 + hh), (3840 - hw, 160 + hh), (3840 - hw, 1980 - hh),
               (hw, 1980 - hh)]
    for _ in range(1000):
        c = catalog.pick_position(rngp, fld, hw, hh, prev_c)
        if not (hw <= c[0] <= 3840 - hw and 160 + hh <= c[1] <= 1980 - hh):
            bad_in += 1
        if prev_c is not None and math.hypot(c[0] - prev_c[0], c[1] - prev_c[1]) \
                < 0.35 * inner_diag - 1e-6 and c not in corners:
            bad_move += 1
        prev_c = c
    check("pick_position: 1,000 draws all inside the inset field", bad_in == 0)
    check("pick_position: every consecutive pair >= 35% of the diagonal apart "
          "(or the farthest corner)", bad_move == 0, f"{bad_move} too close")
    rngp = random.Random(7)
    a = [catalog.pick_position(rngp, fld, hw, hh, None) for _ in range(5)]
    rngp = random.Random(7)
    b = [catalog.pick_position(rngp, fld, hw, hh, None) for _ in range(5)]
    check("pick_position is deterministic for a seed", a == b)

    print("\n8. no mode or artwork module uses the module-level `random`")
    offenders = []
    pat = re.compile(r"\brandom\.(random|uniform|choice|choices|randrange|randint|"
                     r"shuffle|sample|gauss|seed)\s*\(")
    for f in list((ROOT / "src" / "lsa" / "artwork").glob("*.py")) + \
            [ROOT / "src" / "lsa" / "modes" / f
             for f in ("image_target.py", "advancing.py", "popup.py", "figure.py",
                       "sprite.py")]:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{f.name}:{i}")
    check("every draw goes through a seeded generator", not offenders, str(offenders))

    print("\n9. the sprite: pre-scaled, cached, grey by construction")
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from lsa.modes import sprite as S

    g = load.load_grey(catalog.path_for("predator01"))
    spr = S.Sprite(g)
    spr.set_range(0.85 * SH)
    img = QImage(SW, SH, QImage.Format_RGB32)
    img.fill(0xFF0A0A0A)
    p = QPainter(img)
    spr.paint(p, (SW / 2, SH / 2), 0.85 * SH)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(SH, img.bytesPerLine())
    a = a[:, :SW * 4].reshape(SH, SW, 4).astype(np.int16)
    excess = int((a[..., 2] - np.maximum(a[..., 1], a[..., 0])).max())
    check("predator01 (red excess 26 as stored) renders at red excess exactly 0 "
          "per pixel at max height", excess == 0, f"{excess}")
    # the check can fail: the same file built from BGR instead of the grey byte
    import cv2
    bgr = cv2.imread(str(catalog.path_for("predator01")))
    bgr_excess = int((bgr[..., 2].astype(np.int16)
                      - np.maximum(bgr[..., 1], bgr[..., 0])).max())
    check("negative control: building the figure from BGR would read red excess 26",
          bgr_excess == 26, f"{bgr_excess}")
    # alpha works: a pixel that is black in the art equals the scene background
    w_, h_ = spr.drawn_size(0.85 * SH)
    left, top = S.Sprite.top_left((SW / 2, SH / 2), (w_, h_))
    lvl = spr._level(spr.level_for(0.85 * SH))
    ys, xs = np.where(lvl[:200, :200] == 0)
    if len(ys):
        px_ = a[top + ys[0], left + xs[0]]
        check("a pixel that is black in the art is the scene background (no black "
              "rectangle)", tuple(int(v) for v in px_[:3]) == (10, 10, 10), str(px_))
    else:                                            # pragma: no cover
        check("a black pixel exists to test alpha", False)
    bg_corner = a[5, 5]
    check("outside the figure's box the background is untouched",
          tuple(int(v) for v in bg_corner[:3]) == (10, 10, 10))
    # timing: a paint is one blit
    times = []
    img2 = QImage(SW, SH, QImage.Format_RGB32)
    img2.fill(0xFF0A0A0A)
    pp = QPainter(img2)
    for _ in range(30):
        t0 = time.perf_counter()
        spr.paint(pp, (SW / 2, SH / 2), 0.85 * SH)
        times.append((time.perf_counter() - t0) * 1000)
    pp.end()
    paint_ms = sorted(times)[len(times) // 2]
    print(f"         sprite paint at 3840x2160, 0.85 height: median {paint_ms:.2f} ms "
          f"(worst {max(times):.2f})")
    check("a sprite paint is <= 2.5 ms (median of 30)", paint_ms <= 2.5, f"{paint_ms:.2f}ms")
    # a level change. A TIMING check: the machine may be busy (the suite runs beside
    # other work), so it takes the best of three medians -- a load spike raises one
    # run, it cannot lower all three below what the code costs.
    best_step = 1e9
    for _attempt in range(3):
        spr2 = S.Sprite(g)
        spr2.set_range(0.85 * SH)
        steps = []
        for k in range(10):
            hpx = 0.85 * SH / (S.STEP ** (k * 3))
            t0 = time.perf_counter()
            spr2.warm(hpx)
            steps.append((time.perf_counter() - t0) * 1000)
        best_step = min(best_step, sorted(steps)[len(steps) // 2])
    step_ms = best_step
    print(f"         level change (resize + pixmap): best-of-3 median {step_ms:.2f} ms, "
          f"worst of last {max(steps):.2f} ms")
    check("a level change is <= 8 ms (best of three medians of 10)", step_ms <= 8.0,
          f"{step_ms:.2f}ms")
    # memory: walk the whole range
    spr3 = S.Sprite(g)
    spr3.set_range(0.85 * SH)
    peak = 0
    for k in range(0, 80):
        spr3._level(k)
        peak = max(peak, spr3.pyramid_bytes)
    print(f"         pyramid after building 80 levels: peak {peak / 1e6:.1f} MB "
          f"(limit {S.MAX_PYRAMID_BYTES / 1e6:.0f} MB), levels held {len(spr3._levels)}")
    check("the pyramid is bounded at 40 MB however many levels are visited",
          peak <= S.MAX_PYRAMID_BYTES, f"{peak / 1e6:.1f} MB")
    nlevels = int(math.log(0.85 / 0.30) / math.log(S.STEP))
    full = sum(int(round(0.85 * SH / S.STEP ** k)) * int(round(0.85 * SH / S.STEP ** k
              * spr.aspect)) for k in range(nlevels + 1))
    print(f"         the FULL series ({nlevels + 1} levels, 0.30..0.85 of 2160) would be "
          f"{full / 1e6:.1f} MB; the LRU keeps it under 40")
    # the drawn size is what scoring sees
    d = spr.drawn_size(0.5 * SH)
    check("drawn_size reports the level that is actually drawn (within 2% of asked)",
          abs(d[1] - 0.5 * SH) / (0.5 * SH) <= 0.011, str(d))

    # OwnedFiles.check calls check(cond, label, detail); this file's check is
    # (label, cond, detail). Passing it straight through swapped them and made
    # the check unable to fail (RULES C1), so adapt the order.
    owned.check(lambda cond, label, detail="": check(label, cond, detail))
    n = len(checks)
    if all(checks):
        print(f"\nALL {n} CHECKS PASSED")
        return 0
    print(f"\nSOME CHECKS FAILED ({n - sum(checks)} of {n})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
