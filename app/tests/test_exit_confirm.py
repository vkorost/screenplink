"""Escape asks before it closes the app, and nothing else about exiting moves.

Asked for 2026-11-09: "pressing the Esc button brings up a dialog with YES/NO
buttons (yes being the default) asking if the user would like to exit and second
press of Esc removes that dialog; user would have to press either Enter or Space
bar key on YES to exit; or the user would select NO and in this case he is back
to the game; that mechanism doesn't change the exit from Menu behavior from the
menu page the user exists with two shots into the exit tile".

WHAT THIS FILE IS CAREFUL ABOUT, because an exit path is the worst thing in an
app to get wrong in either direction:

  * **Section B drives the REAL `TargetWindow.keyPressEvent`** with real
    `QKeyEvent`s and counts real `quit_requested` emissions. The last key fault
    that escaped this project did so because a test called the app's handler
    directly and bypassed the window, where layers get the key first (RULES C8:
    have the test CALL the path).
  * **Section B also proves the app can still be closed.** A confirmation
    dialog is a trap if it can swallow the way out, and the menu once claimed
    keys it could not act on and became exactly that -- escapable only by F11.
    So: F11 exits at once WITH the dialog up, Escape always takes it down, and
    while a measurement owns the screen Escape keeps its old immediate exit,
    because a dialog that may not be drawn (RULES B2) is a dialog nobody can
    answer.
  * **Section C measures what it draws**, on the screen the camera watches: red
    excess exactly 0 (B1), nothing above the artwork ceiling (B10), the lit area
    (B1b, and the round underneath is still running so shooting is possible),
    and that it draws nothing at all while the app is measuring.
  * **Section E asserts the thing he said must NOT change**: the menu's Exit
    tile still takes two shots and reaches the app by its own route, with this
    dialog not involved.

Each section says what goes red when the behaviour it guards is reverted.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from owned_files import OwnedFiles, isolate_outputs      # noqa: E402

owned = OwnedFiles()
isolate_outputs("exit_confirm")

import cv2                                               # noqa: E402
import numpy as np                                       # noqa: E402
from PySide6.QtCore import QEvent, QRect, Qt             # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent, QPainter    # noqa: E402
from PySide6.QtWidgets import QApplication               # noqa: E402

from lsa import modes                                    # noqa: E402
from lsa.display import layers as L                      # noqa: E402
from lsa.display.layers import ExitConfirm               # noqa: E402
from lsa.display.scene import RenderContext, Scene       # noqa: E402
from lsa.display.session import SessionStats             # noqa: E402
from lsa.display.window import TargetWindow              # noqa: E402
from lsa.modes.base import ModeHost                      # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)

CHECKS = 0
FAILED: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  [PASS] {what}")
    else:
        FAILED.append(what)
        print(f"  [FAIL] {what}" + (f" -- {detail}" if detail else ""))


SW, SH = 3840, 2160


def ctx(w: int = SW, h: int = SH, now: float = 1.0, **kw) -> RenderContext:
    return RenderContext(now=now, dt=1 / 60.0, rect=QRect(0, 0, w, h),
                         stats=SessionStats(), **kw)


# =============================================================================
print("\n=== A. the layer's answer: YES by default, and one way to take it ===")
print("    reverting `ExitConfirm.on_key` is what these measure")

d = ExitConfirm()
check(not d.visible and d.name == "exit",
      "it starts hidden, under the name the window looks it up by")
check(d.on_key(Qt.Key_Escape) is False and not d.visible,
      "a hidden dialog claims nothing, so Escape reaches the window that owns it")

quits: list[int] = []
d.on_exit = lambda: quits.append(1)
d.ask()
check(d.visible and d.yes is True,
      "raised, it points at YES -- his instruction, 'yes being the default'")
check(d.on_key(Qt.Key_Right) is True and d.yes is False,
      "Right points at NO (the right-hand answer), and the dialog claims the key")
check(d.on_key(Qt.Key_Left) is True and d.yes is True, "Left points back at YES")
check(d.on_key(Qt.Key_N) is True and d.yes is False, "N points at NO")
check(d.on_key(Qt.Key_Y) is True and d.yes is True, "Y points at YES")
check(not quits,
      "...and NONE of those four exited the app: Y and N choose, they do not "
      "answer", "'the user would HAVE TO press either Enter or Space to exit'")

d.yes = False
check(d.on_key(Qt.Key_Return) is True and not d.visible and not quits
      and d.answered == "stay",
      "Enter on NO puts the dialog away and goes back to the game")
d.ask()
check(d.on_key(Qt.Key_Escape) is True and not d.visible and not quits,
      "Escape on the dialog puts it away too: his 'second press of Esc removes "
      "that dialog'")
d.ask()
check(d.yes is True,
      "...and the next raise points at YES again, not at the answer that was "
      "left behind")
check(d.on_key(Qt.Key_Return) is True and quits == [1] and not d.visible
      and d.answered == "exit",
      "Enter on YES exits, exactly once")
d.ask()
check(d.on_key(Qt.Key_Space) is True and quits == [1, 1],
      "Space on YES exits too, as he asked")

for key, name in ((Qt.Key_F11, "F11"), (Qt.Key_1, "1"), (Qt.Key_M, "M"),
                  (Qt.Key_Up, "Up"), (Qt.Key_C, "C")):
    dd = ExitConfirm()
    dd.on_exit = lambda: quits.append(9)
    dd.ask()
    check(dd.on_key(key) is False and dd.visible,
          f"it does NOT claim {name}: a modal dialog that swallowed everything "
          f"would swallow the way out as well")

dnone = ExitConfirm()
dnone.ask()
check(dnone.on_key(Qt.Key_Return) is False and dnone.visible,
      "with no route out handed to it, YES claims nothing rather than appearing "
      "to work (RULES C1)",
      "the one place in this app where a silent nothing reads as 'it will not close'")


# =============================================================================
print("\n=== B. through the REAL window: it asks, and it can still be closed ===")
print("    reverting the Escape branch in `window.keyPressEvent` is what these "
      "measure")

scene = Scene(SessionStats())
host = scene.add(ModeHost())
host.set_mode(modes.by_key(Qt.Key_1)(), ctx(640, 360))
board = scene.add(ExitConfirm())
win = TargetWindow(scene)
win.resize(640, 360)
asked: list[int] = []
win.quit_requested.connect(lambda: asked.append(1))


def press(key, repeat: bool = False) -> None:
    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, "", repeat, 1))
    app.processEvents()          # `request_quit` emits on the next loop turn


check(hasattr(win, "request_quit"),
      "the window has ONE route out, which F11, Escape and the dialog all use",
      "three copies of hide-then-emit is the half-fix this project keeps paying for")

press(Qt.Key_Escape)
check(board.visible and board.yes is True and not asked,
      "Escape raises the dialog and does NOT close the app", f"asked {len(asked)}")
check(board.on_exit is not None,
      "...and the window handed it the route out as it raised it, so YES can act")

press(Qt.Key_Escape)
check(not board.visible and not asked,
      "a second Escape takes it down and still does not close the app")

press(Qt.Key_Escape)
press(Qt.Key_Right)
press(Qt.Key_Return)
check(not board.visible and not asked,
      "Escape, Right, Enter: answered NO, back to the game, nothing closed")

press(Qt.Key_Escape)
press(Qt.Key_F11)
check(len(asked) == 1,
      "F11 CLOSES THE APP WITH THE DIALOG UP: the way out cannot be swallowed",
      f"asked {len(asked)}")

board.dismiss()
asked.clear()
press(Qt.Key_Escape)
press(Qt.Key_Return)
check(len(asked) == 1 and not board.visible,
      "Escape then Enter closes the app, through the window's own signal",
      f"asked {len(asked)}")

asked.clear()
press(Qt.Key_Escape)
press(Qt.Key_Space)
check(len(asked) == 1, "Escape then Space closes it too")

# while the app is measuring its own screen ------------------------------------
asked.clear()
board.dismiss()
scene.diagnostics["measuring"] = True
press(Qt.Key_Escape)
check(len(asked) == 1 and not board.visible,
      "WHILE A MEASUREMENT OWNS THE SCREEN Escape keeps its old immediate exit: "
      "nothing may cover a screen being measured (B2), and a dialog that cannot "
      "be drawn cannot be answered", f"asked {len(asked)}, up={board.visible}")
scene.diagnostics["measuring"] = False
asked.clear()
scene.calibrating = True
press(Qt.Key_Escape)
check(len(asked) == 1 and not board.visible,
      "...and the same during a calibration")
scene.calibrating = False

# a window with no dialog at all must still exit on Escape ---------------------
bare = Scene(SessionStats())
bare_win = TargetWindow(bare)
bare_asked: list[int] = []
bare_win.quit_requested.connect(lambda: bare_asked.append(1))
bare_win.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier, "", False, 1))
app.processEvents()
check(len(bare_asked) == 1,
      "a scene with NO exit layer still closes on Escape: the dialog is an "
      "addition to the exit path, never a precondition for it")

# a held Escape is one press ---------------------------------------------------
asked.clear()
board.dismiss()
press(Qt.Key_Escape)
press(Qt.Key_Escape, repeat=True)
press(Qt.Key_Escape, repeat=True)
check(board.visible and not asked,
      "a HELD Escape does not flicker the dialog open and shut: auto-repeat is "
      "dropped by the window, as it is for every discrete key")
board.dismiss()


# =============================================================================
print("\n=== C. what it draws on the screen the camera watches ===")
print("    reverting any colour here to a hue, or to a filled panel, is what "
      "these measure")


def render(layer, c: RenderContext):
    img = QImage(c.width, c.height, QImage.Format_RGB32)
    img.fill(0xFF0A0A0A)                     # the target background
    p = QPainter(img)
    layer.paint(p, c)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(
        c.height, img.bytesPerLine())[:, :c.width * 4].reshape(c.height, c.width, 4)
    b_, g_, r_ = a[..., 0], a[..., 1], a[..., 2]
    red = int(cv2.subtract(r_, cv2.max(g_, b_)).max())
    lum = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_BGR2GRAY)
    return red, int(lum.max()), float((lum > 60).mean()), int((lum > 30).sum())


shown = ExitConfirm()
shown.ask()
for label, (w, h) in (("4K", (SW, SH)), ("1080p", (1920, 1080))):
    c = ctx(w, h)
    red, brightest, lit, ink = render(shown, c)
    print(f"     MEASURED {label}: red excess {red}, brightest {brightest} "
          f"(ceiling {L.CEILING}), lit above luminance 60 = {lit * 100:.3f}% "
          f"of the screen (budget 2%), ink {ink} px")
    check(red == 0 and ink > 500,
          f"{label}: neutral grey, red excess exactly 0, and something IS drawn",
          f"red {red}, {ink} px")
    check(brightest <= L.CEILING,
          f"{label}: nothing brighter than the artwork ceiling "
          f"({brightest} vs {L.CEILING})")
    # 0.6% is DERIVED, not picked: the shipping shooting state measures about
    # 0.9% (RULES B1b), so a dialog under 0.6 leaves the pair under 1.5 against
    # the 2% budget. 1080p is the binding size, as it is for the Keys screen,
    # because a small bold glyph's stems snap to whole pixels.
    check(lit <= 0.006,
          f"{label}: lit area {lit * 100:.3f}% leaves the 2% budget room for the "
          f"drill underneath -- the round is still running, so shooting is still "
          f"possible (B1b)", f"{lit * 100:.3f}%")

# THE WHOLE PROPERTY, not the dialog on its own. The budget is on the SCREEN,
# and while this dialog is up a drill is still drawing underneath it, so the
# only honest measurement is the pair together (RULES E4: a test that asserts
# "some guard exists" is not enough; assert the thing that must be true).
#
# OVER EVERY REGISTERED DRILL, not one of them. Picking Static Targets by hand
# would be measuring the drill that was easy to reach rather than the one that
# binds (RULES C6), and a drill added next year is covered here without anybody
# remembering -- the same reason `test_ceiling` renders them all.
#
# AND THROUGH A REAL `Scene`, not a context built by hand: `ctx.asking` is
# DERIVED from the dialog's own visibility inside `Scene._context`, so a
# hand-made context would measure a screen that never stands down and would
# pass while the app broke the rule (RULES C12 -- a stand-in must mirror the
# real thing, not paraphrase it).
def scene_lit(cls, w: int, h: int, up: bool, closed: bool = False):
    sc = Scene(SessionStats())
    sc.dark_background = True
    h_ = sc.add(ModeHost())
    d_ = sc.add(ExitConfirm())
    if closed:
        h_.set_magazine(3)
    h_.set_mode(cls(), ctx(w, h))
    if closed:
        # Shoot the magazine dry and let the hold expire, so what is on screen
        # is the END-OF-ROUND SCREEN: a results panel, which is the brightest
        # thing a drill ever draws and the state a stray Escape is most likely
        # to meet. Measuring only freshly-started drills would have missed it.
        for i in range(3):
            h_.shot(modes.ShotEvent(100.0, 1000.0, 1.0 + i * 0.01, i + 1), ctx(w, h))
        h_.advance(ctx(w, h, now=20.0))
    if up:
        d_.ask()
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(0xFF0A0A0A)
    p = QPainter(img)
    sc.paint(p, QRect(0, 0, w, h))
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(
        h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4)
    b_, g_, r_ = a[..., 0], a[..., 1], a[..., 2]
    red_ = int(cv2.subtract(r_, cv2.max(g_, b_)).max())
    lum_ = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_BGR2GRAY)
    return red_, float((lum_ > 60).mean())


for label, (w, h) in (("4K", (SW, SH)), ("1080p", (1920, 1080))):
    rows = []
    for cls, _key in modes.registered():
        try:
            red_b, lit_b = scene_lit(cls, w, h, up=True)
        except Exception as exc:                            # noqa: BLE001
            check(False, f"{label}: {cls.__name__} could not be started for the "
                         f"brightness measurement", str(exc))
            continue
        rows.append((lit_b, cls.__name__, red_b))
    rows.sort(reverse=True)
    lit_b, name, _red = rows[0]
    print(f"     MEASURED {label} WITH THE DIALOG UP OVER EACH SCREEN: worst of "
          f"{len(rows)} is {name} at {lit_b * 100:.3f}% lit (budget 2%, "
          f"headroom {(0.02 - lit_b) * 100:.3f})")
    check(all(r[2] == 0 for r in rows) and lit_b <= 0.02,
          f"{label}: the dialog over EVERY screen ({len(rows)} of them) stays "
          f"inside the 2% lit-area budget and at red excess 0; worst is {name}",
          f"{name} at {lit_b * 100:.3f}%")
    # The ceiling is `test_ceiling`'s property over the same modes, and two
    # definitions of one fact is how this project has been bitten before; it is
    # deliberately not re-asserted here with a second harness.

    # ...and over the END-OF-ROUND SCREEN of every drill that has one.
    ends = []
    for cls, _key in modes.registered():
        probe = ModeHost()
        probe.set_mode(cls(), ctx(w, h))
        if not probe.counts_ammo:
            continue                     # the menu and the screens end no round
        red_e, lit_e = scene_lit(cls, w, h, up=True, closed=True)
        ends.append((lit_e, cls.__name__, red_e))
    ends.sort(reverse=True)
    lit_e, name_e, _red_e = ends[0]
    print(f"     MEASURED {label} WITH THE DIALOG UP OVER A CLOSED ROUND: worst "
          f"of {len(ends)} is {name_e} at {lit_e * 100:.3f}% lit (budget 2%, "
          f"headroom {(0.02 - lit_e) * 100:.3f})")
    check(all(r[2] == 0 for r in ends) and lit_e <= 0.02,
          f"{label}: the dialog over an END-OF-ROUND screen is inside the 2% "
          f"budget too -- a results panel is the brightest thing a drill draws, "
          f"and a stray Escape is likeliest right there; worst is {name_e}",
          f"{name_e} at {lit_e * 100:.3f}%")

# ...and the reason that passes: the brightest screen stands down -------------
sc_name = "Shortcuts"
_r_up, lit_up = scene_lit(modes.by_key(Qt.Key_K), 1920, 1080, up=True)
_r_down, lit_down = scene_lit(modes.by_key(Qt.Key_K), 1920, 1080, up=False)
print(f"     MEASURED 1080p: the Keyboard Shortcuts screen is {lit_down * 100:.3f}% "
      f"lit alone and {lit_up * 100:.3f}% with the dialog asking in front of it")
check(lit_up < lit_down and lit_up <= 0.01,
      "the Keyboard Shortcuts screen STANDS DOWN while a question is in front of "
      "it: it is the brightest screen in the app and the pair would otherwise "
      "break the 2% rule at 2.379%",
      f"{lit_down * 100:.3f}% alone -> {lit_up * 100:.3f}% asking")

derived = Scene(SessionStats())
d_der = derived.add(ExitConfirm())
check(derived._context(1.0, 0.0, QRect(0, 0, SW, SH)).asking is False,
      "`ctx.asking` is False while nothing is asking")
d_der.ask()
check(derived._context(1.0, 0.0, QRect(0, 0, SW, SH)).asking is True,
      "...True while the dialog is up, DERIVED from the layer every frame")
d_der.on_key(Qt.Key_Escape)
check(derived._context(1.0, 0.0, QRect(0, 0, SW, SH)).asking is False,
      "...and False again the instant the dialog closes ITSELF, which a stored "
      "flag would have got wrong the first time he pressed Escape twice")
check(Scene(SessionStats()).add(ModeHost()).asks_a_question is False,
      "a layer that asks nothing says so by default, declared on `Layer` rather "
      "than reached for with a getattr default (RULES C8)")

hidden = ExitConfirm()
_r, _b, _l, ink_hidden = render(hidden, ctx())
check(ink_hidden == 0, "hidden, it draws nothing at all")
_r, _b, _l, ink_meas = render(shown, ctx(diagnostics={"measuring": True}))
check(ink_meas == 0,
      "and it stands down while the app measures its own screen (B2): a dialog "
      "on the screen would be part of the measurement")
_r, _b, _l, ink_cal = render(shown, ctx(calibrating=True))
check(ink_cal == 0, "...and during a calibration")

# the layout: on screen, and the selection moves -------------------------------
for w, h in ((SW, SH), (1920, 1080), (1280, 720)):
    c = ctx(w, h)
    box = shown.layout(c)
    inside = all(QRect(0, 0, w, h).contains(r.toAlignedRect())
                 for r in box.values())
    check(inside, f"{w}x{h}: the title, both answers, the hint and the outline "
                  f"are all on the screen", str({k: v.toAlignedRect() for k, v in box.items()}))
yes_box = ExitConfirm()
yes_box.ask()
a_yes = yes_box.layout(ctx())["box"]
yes_box.yes = False
a_no = yes_box.layout(ctx())["box"]
check(a_yes.center().x() < a_no.center().x() and a_yes != a_no,
      "the outline is around the CHOSEN answer and moves with it, YES on the "
      "left", f"{a_yes.center().x():.0f} vs {a_no.center().x():.0f}")
check(shown.layout(ctx())["yes"].center().x()
      < shown.layout(ctx())["no"].center().x(),
      "...and YES really is the left-hand answer, which is what Left means")


# =============================================================================
print("\n=== D. it says what to press, in both places that list keys ===")
print("    reverting the Keys screen row or the bottom legend leaves a shooter "
      "pressing Escape twice and wondering")

from lsa.introspect.gui import resolve_key                # noqa: E402
from lsa.modes import shortcuts                          # noqa: E402

hint = ExitConfirm.HINT.upper()
check("ENTER" in hint and "SPACE" in hint and "ESC" in hint,
      "the dialog itself names Enter, Space and Escape, so it is answerable "
      "without the manual", ExitConfirm.HINT)
check("?" in ExitConfirm.TITLE and "EXIT" in ExitConfirm.TITLE.upper(),
      "...and it asks a question rather than announcing something",
      ExitConfirm.TITLE)

esc_row = next((e for e in shortcuts.ALSO if int(Qt.Key_Escape) in e.keys), None)
check(esc_row is not None and "ask" in esc_row.title.lower(),
      "the K (Keyboard Shortcuts) screen says Escape ASKS first",
      getattr(esc_row, "title", "MISSING"))

legend = " ".join(L.KeyHelp().lines).lower()
check("esc" in legend and "ask" in legend,
      "the bottom legend says so too", "a key legend that lies is how this "
      "project lost three days once")
check("f11" in legend and "at once" in legend,
      "...and that F11 is the one that does not ask, which is the way out if "
      "the dialog is ever in the way")

for name, key in (("Return", Qt.Key_Return), ("Space", Qt.Key_Space),
                  ("Left", Qt.Key_Left), ("Right", Qt.Key_Right)):
    check(resolve_key(name) == int(key),
          f"the introspection server can press {name}, so the dialog can be "
          f"driven and read back without his hands")
check(resolve_key("y") == int(Qt.Key_Y) and resolve_key("n") == int(Qt.Key_N),
      "...and Y and N")


# =============================================================================
print("\n=== E. the MENU's Exit tile is untouched: two shots, its own route ===")
print("    this is the thing he said must NOT change")

from lsa.modes.menu import EXIT_ARM_S, Menu              # noqa: E402

mctx = ctx()
menu = Menu()
menu.start(mctx)
slot = next(s for s in menu.slots(mctx) if s.cls is None and s.stem == "exit")
cx, cy = slot.rect.center().x(), slot.rect.center().y()
guard = ExitConfirm()

def shoot_tile(m, when: float) -> None:
    """A real shot at the Exit tile, through `Menu.shot`, which is the path the
    pistol takes (RULES C8: call the path, do not reach past it)."""
    c = ctx(now=when)
    m.shot(modes.ShotEvent(float(cx), float(cy), when, 1), c)


shoot_tile(menu, 100.0)
check(menu.quit_requested is False and menu.exit_armed_at > 0.0,
      "one shot on the Exit tile ARMS it and does not exit")
check(not guard.visible,
      "...and does not raise the Escape dialog: the tile has its own route and "
      "its own confirmation")
shoot_tile(menu, 100.0 + EXIT_ARM_S / 2.0)
check(menu.quit_requested is True,
      "the second shot inside the arming window exits, through the menu's own "
      "`quit_requested`, which the app drains on its tick")

menu2 = Menu()
menu2.start(mctx)
shoot_tile(menu2, 100.0)
shoot_tile(menu2, 100.0 + EXIT_ARM_S + 1.0)
check(menu2.quit_requested is False,
      "...and a second shot too late re-arms instead of exiting, exactly as "
      "before")
check(int(Qt.Key_Escape) not in {int(k) for k in (ExitConfirm.ACTIVATE_KEYS)},
      "Escape is not an activate key: it answers neither YES nor NO, it just "
      "takes the question away")
check(float(cx) > 0 and float(cy) > 0,
      "the Exit tile is still a tile with a position on screen, shootable as it "
      "always was", f"{cx:.0f},{cy:.0f}")


# =============================================================================
owned.check(lambda ok, what, detail="": check(ok, what, detail))
print("\n" + (f"ALL {CHECKS} CHECKS PASSED" if not FAILED else
              f"{len(FAILED)} OF {CHECKS} CHECKS FAILED"))
raise SystemExit(0 if not FAILED else 1)
