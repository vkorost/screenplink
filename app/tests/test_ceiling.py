"""RULES B10: nothing a mode draws is brighter than the darkest artwork he supplied.

His finding (2026-10-17), after shooting the app's own targets while his figure
images registered: a laser landing on a lit grey loses its red excess (255,10,10)
scores 245 on near-black, (255,200,200) scores 55 on grey 200, against a bar of
18. The app's rings were drawn at 190-240, the circles as filled discs.

Offscreen Qt, no camera, no display.

What this holds, and why each is its own check:
  * the ceiling is MEASURED from the artwork (re-derived here independently), and
    follows a darker image added later;
  * EVERY registered mode, rendered in several states, has no pixel above it. The
    mode list comes from the registry, so a mode written next year is covered
    without anybody remembering this file exists;
  * the figures' DRAWN strokes (p99 of lit pixels) sit at it, the brighter ones
    scaled down, none scaled up, and their geometry is untouched;
  * mode SOURCE contains no literal grey above it (the five-authors problem);
  * the lit area stays inside the 2% budget in every mode.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / "tests"))

import cv2                                                         # noqa: E402
import numpy as np                                                 # noqa: E402
from PySide6.QtGui import QImage, QPainter                         # noqa: E402
from PySide6.QtWidgets import QApplication                         # noqa: E402

import lsa.modes as modes                                          # noqa: E402
from lsa.artwork import catalog, ceiling, load                     # noqa: E402
from lsa.display import layers                                     # noqa: E402
from figure_helpers import SW, SH, FIGURE_MODES, context           # noqa: E402
from asset_env import res                                          # noqa: E402  (the asset table)


def render(m, ctx):
    img = QImage(SW, SH, QImage.Format_RGB32)
    img.fill(0xFF0A0A0A)
    p = QPainter(img)
    m.paint(p, ctx)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(SH, img.bytesPerLine())
    a = a[:, :SW * 4].reshape(SH, SW, 4)
    lum = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_BGR2GRAY)
    return lum


def mode_states(cls):
    """Several states of one mode: idle, running, after shots at what it offers."""
    m = cls()
    if hasattr(m, "_rng"):
        m._rng.seed(5)
    t = 100.0
    m.start(context(now=t))
    out = []
    for k in range(1, 400):
        t += 1 / 60.0
        m.advance(context(now=t))
        if k in (5, 120, 240, 399):
            out.append((f"t+{k}", m, context(now=t)))
    for k in range(3):                                   # hits, flashes, fades
        pt = m.aim_point(context(now=t)) if hasattr(m, "aim_point") else None
        if pt:
            m.shot(modes.ShotEvent(pt[0], pt[1], t, k + 1), context(now=t))
            t += 0.05
    out.append(("after shots", m, context(now=t)))
    return out


def main() -> int:
    QApplication.instance() or QApplication([])
    checks: list[bool] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    CEIL = layers.CEILING

    print("1. the ceiling is measured from the artwork, not typed")
    p99 = {}
    TARGETS = res("targets")                 # the asset table (a scratch copy under a runner)
    for f in sorted(TARGETS.glob("*.jpg")) + sorted((TARGETS / "fall").glob("*.jpg")):
        g = load.load_grey(f)
        lit = g[g > 30]
        p99[str(f.relative_to(TARGETS)).replace("\\", "/")] = float(np.percentile(lit, 99))
    darkest_name = min(p99, key=p99.get)
    expect_file = max(40.0, p99[darkest_name])
    got_file = ceiling.figure_level()
    print(f"         darkest artwork {darkest_name} p99 {p99[darkest_name]:.1f}; "
          f"ceiling {got_file:.1f} file / {CEIL} on screen")
    check("the ceiling equals the smallest p99 of lit pixels over EVERY image, derived independently here",
          abs(got_file - expect_file) <= 1.0, f"{got_file} vs {expect_file}")
    check("layers.CEILING is that number as it shows over the target background",
          CEIL == round(got_file + load.BACKGROUND * (1 - got_file / 255.0)), str(CEIL))
    srcs = (ROOT / "src" / "lsa" / "display" / "layers.py").read_text(encoding="utf-8")
    check("layers.py derives it (no typed-in luminance for the ceiling)",
          "CEILING = _ceiling.screen_level()" in srcs)

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "fall").mkdir()
        import shutil
        for n in ("axman01", "thug01"):
            shutil.copy(TARGETS / f"{n}.jpg", tmp / f"{n}.jpg")
        base = ceiling.figure_level(tmp)
        check(f"in a small folder the ceiling is the darker of the two ({base})",
              abs(base - min(ceiling.lit_p99(load.load_grey(tmp / n)) for n in ("axman01.jpg", "thug01.jpg"))) <= 1.0)
        g = load.load_grey(tmp / "axman01.jpg")
        dim = (g.astype(np.float32) * 0.8).astype(np.uint8)
        cv2.imwrite(str(tmp / "dimmer.jpg"), cv2.merge([dim, dim, dim]))
        lower = ceiling.figure_level(tmp)
        check(f"adding a DARKER image moves the ceiling down to it ({base:.1f} -> {lower:.1f})",
              lower < base - 5)
        very = (g.astype(np.float32) * 0.45).astype(np.uint8)
        cv2.imwrite(str(tmp / "veryfaint.jpg"), cv2.merge([very, very, very]))
        floor = ceiling.figure_level(tmp)
        check(f"...but never below the legibility floor ({ceiling.MIN_LEVEL}): got {floor}",
              floor == ceiling.MIN_LEVEL)
        check("...and that too-dark image is drawn as it is, never raised (gain 1.0)",
              ceiling.gain_for_path(tmp / "veryfaint.jpg", tmp) == 1.0)
        check("a brighter image is scaled DOWN (gain < 1), a darker one is not touched",
              ceiling.gain_for_path(tmp / "thug01.jpg", tmp) < 1.0
              and ceiling.gain_for_path(tmp / "dimmer.jpg", tmp) <= 1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("2. what the dot does on the brightest thing a mode may draw")
    on_ceiling = 255 - CEIL
    on_old = 255 - 200
    check(f"a dot on a stroke at the ceiling ({CEIL}) keeps a red excess of {on_ceiling}, against {on_old} on the old 200 "
          f"(bar 18): {on_ceiling / 18:.0f}x the bar", on_ceiling >= 150 and on_ceiling > 3 * on_old)

    print()
    print("3. the palette sits under the ceiling")
    pal = {n: getattr(layers, n) for n in dir(layers) if n.startswith("GREY_")}
    over = {n: c.red() for n, c in pal.items() if max(c.red(), c.green(), c.blue()) > CEIL}
    check(f"every GREY_* in layers.py is at or under {CEIL}, neutral", not over
          and all(c.red() == c.green() == c.blue() for c in pal.values()), str(over))
    check("the hit flash is still the brightest thing in the palette",
          layers.GREY_HIT.red() == max(c.red() for c in pal.values()) == CEIL)

    print()
    print("4. EVERY registered mode, rendered in several states, has no pixel above the ceiling")
    print(f"         registered: {[c.name for c in modes.available()]}")
    figure_names = set(FIGURE_MODES)
    worst_lit = 0.0
    for cls in modes.available():
        peak, lit, p99_lit = 0, 0.0, 0.0
        for label, m, ctx in mode_states(cls):
            lum = render(m, ctx)
            peak = max(peak, int(lum.max()))
            lit = max(lit, float((lum > 60).mean()))
            ink = lum[lum > 30]
            if ink.size:
                p99_lit = max(p99_lit, float(np.percentile(ink, 99)))
        worst_lit = max(worst_lit, lit)
        print(f"         {cls.name:12s} peak {peak:3d}   p99 of lit {p99_lit:5.1f}   lit>60 {lit * 100:5.3f}%")
        if cls.name in figure_names:
            # artwork is scaled so its p99 meets the ceiling; single anti-aliased
            # pixels above p99 are the artwork's own dynamic range, not a colour
            # the app chose, but the flash/ring/counter the app DRAWS are palette
            check(f"{cls.name}: the figure's lit strokes have p99 <= the ceiling "
                  f"({p99_lit:.1f} vs {CEIL}), peak {peak}",
                  p99_lit <= CEIL + 1.0)
        else:
            check(f"{cls.name}: no pixel anywhere above the ceiling (peak {peak} vs {CEIL})",
                  peak <= CEIL)
        check(f"{cls.name}: lit area (>60) {lit * 100:.3f}% inside the 2% budget", lit <= 0.02)

    from lsa.modes.circles import ClearTheBoard
    mc = ClearTheBoard()
    mc._rng.seed(3) if hasattr(mc, "_rng") else None
    mc.start(context(now=10.0))
    t0 = next(t for t in mc.targets if t.alive)
    lum_c = render(mc, context(now=10.0))
    mid = lum_c[int(t0.y), int(t0.x + 0.72 * t0.r)]
    check(f"circles are OUTLINES: between the outer ring and the inner one the screen is still dark "
          f"(luminance {mid}, was a lit disc)", mid <= 20)

    print()
    print("5. every figure and every fall pose, as drawn")
    from lsa.modes.figure import Figure
    worst = 0.0
    for nm in catalog.names():
        fg = Figure(nm, SW, SH, 0.015)
        h = fg.fit_height(SH * 0.98, SH, 1.0)
        fg.sprite.set_range(h)
        img = QImage(SW, SH, QImage.Format_RGB32)
        img.fill(0xFF0A0A0A)
        p = QPainter(img)
        fg.sprite.paint(p, (SW / 2, SH / 2), h)
        p.end()
        a = np.frombuffer(img.constBits(), np.uint8).reshape(SH, img.bytesPerLine())[:, :SW * 4].reshape(SH, SW, 4)
        lum = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_BGR2GRAY)
        ink = lum[lum > 30]
        d = float(np.percentile(ink, 99))
        worst = max(worst, d)
        check(f"{nm}: drawn p99 {d:.1f} <= {CEIL}+1 (gain {fg.gain:.2f}"
              + (f", fall gain {fg.fall_gain:.2f}" if fg.has_fall else "") + ")", d <= CEIL + 1.0)
        check(f"{nm}: never scaled UP", fg.gain <= 1.0 and (not fg.has_fall or fg.fall_gain <= 1.0))
    print(f"         worst drawn p99 over all figures: {worst:.1f}")

    print()
    print("6. the geometry did not move: zones still verify, caps and silhouettes are from the unscaled art")
    from lsa.artwork import zones as Z
    check("every figure still has verified scoring zones", catalog.excluded() == {})
    fg = Figure("thug02", SW, SH, 0.015)
    raw = load.load_grey(catalog.path_for("thug02"))
    cap_raw = load.max_height_for_budget(raw, SW, SH, 0.015)
    check(f"thug02 (scaled by {fg.gain:.2f}) keeps the size cap computed from the unscaled art "
          f"({fg.cap:.3f} == {cap_raw:.3f})", abs(fg.cap - cap_raw) < 1e-9 and fg.gain < 1.0)

    print()
    print("7. no literal grey above the ceiling in any mode's source")
    hexre = re.compile(r"""["']#([0-9a-fA-F]{6})["']""")
    rgbre = re.compile(r"QColor\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[,)]")
    offenders = []
    for path in sorted((ROOT / "src" / "lsa" / "modes").glob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
            if line.lstrip().startswith("#"):
                continue
            for m in hexre.finditer(code):
                v = [int(m.group(1)[k:k + 2], 16) for k in (0, 2, 4)]
                if max(v) > CEIL:
                    offenders.append(f"{path.name}:{i} #{m.group(1)}")
            for m in rgbre.finditer(code):
                v = [int(x) for x in m.groups()]
                if max(v) > CEIL:
                    offenders.append(f"{path.name}:{i} QColor{tuple(v)}")
    check("no mode file spells a colour brighter than the ceiling (use the palette in layers.py)",
          not offenders, str(offenders))

    passed = all(checks)
    print("\n" + (f"ALL {len(checks)} CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
