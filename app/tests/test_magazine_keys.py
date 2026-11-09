"""The magazine from the firing line: Up / Down set the size, Right adds ten.

Asked for 2026-11-09: "up/down arrows to increase/decrease the size of the mag
(the min would be 6 the max would be 30 and that setting would be remembered);
and I want the + button to add bullets to the current mag to the current session
only, 10 at a time; if there are too many to fit into the display you will
display only the bullets that fit the screen".

`+` could not have it -- it is the volume, and the manual says so -- so the top
up is the RIGHT ARROW and the four arrows are now the magazine cluster. All four
were unbound; nothing in the app had ever answered one.

WHAT THIS FILE MEASURES RATHER THAN READS.

  * **Section B asserts the ROUND, not the number.** `_round_mag` changing is a
    claim; the thing that matters is that the string actually gets longer, so
    the check shoots the magazine dry, tops up, and keeps shooting until the
    round closes at the NEW size. A field that moved without the round
    following it is RULES C1 -- a key that appears to work and silently does
    nothing -- and C6, measuring the easy thing instead of the thing.
  * **Section C derives `MIN_SHRINK`.** The floor in `modes/magazine.py` claims
    that every size the keys can reach still shows every bullet. That is a
    measurement over all 25 sizes, in single player and in both halves of a
    two-player screen, at two screen sizes -- and the binding case is a HALF,
    not the single strip, which is how 0.75 nearly shipped hiding one bullet of
    thirty from each player.
  * **Section D proves the numeral cannot lie.** A truncated strip under-DRAWS;
    the one thing it may never do is under-COUNT, because the whole purpose of
    the strip is to say how many rounds are left. The recorder reads the text
    the real `paint` hands to the real text fitter.

Each section says what goes red when the behaviour it guards is reverted.
"""

from __future__ import annotations

import safe_stdio  # noqa: F401  -- a suite's own printer must not crash it

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from owned_files import OwnedFiles, isolate_outputs      # noqa: E402

owned = OwnedFiles()
isolate_outputs("magazine_keys")     # BEFORE camera_lab is imported

import lsa.audio.mixer as mixer_mod                       # noqa: E402
import lsa.audio.listener as listener_mod                 # noqa: E402

_opened = []


class _Tripwire:
    """No suite may open a real audio device (and none needs to here)."""

    def __init__(self, *a, **k):
        _opened.append(1)
        raise RuntimeError("TRIPWIRE: a test tried to open a real audio stream")


mixer_mod.sd.OutputStream = _Tripwire
listener_mod.sd.InputStream = _Tripwire

from PySide6.QtCore import QRect, Qt                      # noqa: E402
from PySide6.QtWidgets import QApplication                # noqa: E402

import figure_helpers as fh                               # noqa: E402
from lsa import modes                                     # noqa: E402
from lsa.modes import magazine as mag                     # noqa: E402
from lsa.modes.base import ModeHost                       # noqa: E402
from lsa.settings import (DEFAULT_MAGAZINE, MAGAZINE_KEY_MAX,   # noqa: E402
                          MAGAZINE_KEY_MIN, MAGAZINE_KEY_STEP,
                          MAGAZINE_MAX, MAGAZINE_MIN, MAGAZINE_TOPUP,
                          normalise_magazine)

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


def ctx(w: int = 3840, h: int = 2160, now: float = 1.0):
    from lsa.display.scene import RenderContext
    from lsa.display.session import SessionStats
    return RenderContext(now=now, dt=1 / 60.0, rect=QRect(0, 0, w, h),
                         stats=SessionStats())


_Static = modes.by_key(Qt.Key_1)
C4 = ctx()

# THE HOLD IS OFF FOR THIS FILE, as it is for test_modes, and for the same
# reason: a round does not close on the shot that empties the magazine, it
# closes when the last hole has been on its target for `hold_s` as well. With
# the product's 1.2 s in place, every "the round closed" check would depend on
# which clock the check happened to pass in, and three of them did on the first
# run of this file. The ONE check that is about the hold puts the real value
# back and names it.
REAL_HOLD_S = ModeHost.HOLD_S
ModeHost.HOLD_S = 0.0


def host(players: int = 1, size=None, c=None):
    h = ModeHost()
    if size is not None:
        h.set_magazine(size)
    h.set_mode(_Static(), c or C4, players=players)
    return h


def shoot(h, n, x=100.0, y=1000.0, t0=1.0, c=None):
    for i in range(n):
        h.shot(modes.ShotEvent(x, y, t0 + i * 0.01, i + 1), c or C4)


# =============================================================================
print("\n=== A. the two ranges, and the one press that is not a ladder ===")
print("    reverting `next_magazine` lets a key leave his 6-30 band, or step by "
      "a rung and skip the exact size he asked for")

from lsa.tools.camera_lab import CameraLab                # noqa: E402

check((MAGAZINE_KEY_MIN, MAGAZINE_KEY_MAX) == (6, 30),
      "the keys reach 6 to 30, his numbers",
      f"{MAGAZINE_KEY_MIN}-{MAGAZINE_KEY_MAX}")
check(MAGAZINE_MIN <= MAGAZINE_KEY_MIN and MAGAZINE_KEY_MAX <= MAGAZINE_MAX,
      "...inside what a magazine may BE, which stays wider (1-50) so a top-up "
      "can pass 30 and a hand-edited file is clamped, never refused",
      f"keys {MAGAZINE_KEY_MIN}-{MAGAZINE_KEY_MAX} in {MAGAZINE_MIN}-{MAGAZINE_MAX}")
check(MAGAZINE_KEY_MIN <= DEFAULT_MAGAZINE <= MAGAZINE_KEY_MAX,
      f"...and his default of {DEFAULT_MAGAZINE} is reachable from the keys, so "
      f"a fresh install can be adjusted in both directions")

walk, n = [], MAGAZINE_KEY_MIN
for _ in range(40):
    n = CameraLab.next_magazine(n, +1)
    walk.append(n)
check(walk[-1] == MAGAZINE_KEY_MAX and walk.count(MAGAZINE_KEY_MAX) > 1,
      "Up climbs to the top and STOPS there (it does not wrap to the bottom)",
      f"{walk[:6]} ... {walk[-3:]}")
check(sorted(set(walk)) == list(range(MAGAZINE_KEY_MIN + 1, MAGAZINE_KEY_MAX + 1)),
      f"...reaching EVERY size between {MAGAZINE_KEY_MIN} and {MAGAZINE_KEY_MAX}: "
      f"one round a press, not a ladder",
      "he asked for the exact mag size; a rung ladder cannot name 23")
check(MAGAZINE_KEY_STEP == 1, "...and the step is that one round", str(MAGAZINE_KEY_STEP))

down, n = [], MAGAZINE_KEY_MAX
for _ in range(40):
    n = CameraLab.next_magazine(n, -1)
    down.append(n)
check(down[-1] == MAGAZINE_KEY_MIN and min(down) == MAGAZINE_KEY_MIN,
      "Down walks to 6 and stops: a magazine of 0 would end every round before "
      "it began", f"{down[-3:]}")

check(CameraLab.next_magazine(45, +1) == MAGAZINE_KEY_MAX
      and CameraLab.next_magazine(45, -1) < MAGAZINE_KEY_MAX,
      "a file hand-edited to 45 is pulled INTO the band by the first press, "
      "rather than stepping to 46 or appearing inert",
      f"up {CameraLab.next_magazine(45, +1)}, down {CameraLab.next_magazine(45, -1)}")
check(CameraLab.next_magazine(2, -1) == MAGAZINE_KEY_MIN
      and CameraLab.next_magazine(2, +1) == MAGAZINE_KEY_MIN + MAGAZINE_KEY_STEP,
      "...and so is a file hand-edited to 2")
check(CameraLab.next_magazine("junk", +1) == normalise_magazine("junk") + MAGAZINE_KEY_STEP
      and MAGAZINE_KEY_MIN <= CameraLab.next_magazine(None, -1) <= MAGAZINE_KEY_MAX,
      "junk in the setting still gives a usable size, never a crash under his "
      "finger", f"{CameraLab.next_magazine('junk', +1)}")
check(MAGAZINE_TOPUP == 10, "the top-up deals ten rounds, his number", str(MAGAZINE_TOPUP))


# =============================================================================
print("\n=== B. the ROUND gets longer, and the saved size does not ===")
print("    reverting `ModeHost.add_rounds` to touch `magazine`, or dropping the "
      "`round_over` guard, is what these measure")

h = host(size=8)
check(h.round_magazine == 8 and h.remaining(0) == 8,
      "a round is dealt the size that was set before it started")
shoot(h, 3)
check(h.round_magazine == 8 and h.remaining(0) == 5,
      "three shots leave five, and the dealt size does not move")

got = h.add_rounds(MAGAZINE_TOPUP)
check(got == 18 and h.remaining(0) == 15,
      "the top-up deals ten more into THIS round: 8 -> 18, and the five left "
      "became fifteen", f"got {got}, remaining {h.remaining(0)}")
check(h.magazine == 8,
      "...and the SAVED size is untouched, so the next round is dealt 8 again",
      f"magazine {h.magazine}")

# the property that matters: the string really is longer -------------------------
shoot(h, 14, t0=5.0)
check(not h.round_over and h.remaining(0) == 1,
      "the round is STILL OPEN at seventeen shots, where an 8-round magazine "
      "would have closed nine shots ago",
      f"over={h.round_over} remaining={h.remaining(0)}")
shoot(h, 1, t0=9.0)
check(h.round_over and h.tallies[0].shots == 18,
      "...and it closes on the eighteenth: the top-up lengthened the round "
      "itself, not just a number",
      f"over={h.round_over} shots={h.tallies[0].shots}")
check(h.add_rounds(MAGAZINE_TOPUP) is None and h.round_magazine == 18,
      "a CLOSED round refuses the top-up: its score is banked and on screen, "
      "and re-opening it would change a result he is reading")
h.restart(C4)
check(h.round_magazine == 8 and h.remaining(0) == 8,
      "the next round is dealt the saved 8: the top-up was for that round only",
      f"{h.round_magazine}")

# the hold after the last shot ---------------------------------------------------
check(REAL_HOLD_S == 1.2 and ModeHost.HOLD_S == 0.0,
      f"the product's hold is {REAL_HOLD_S} s and this file otherwise runs with "
      f"it off, so a 'the round closed' check cannot pass by which clock it was "
      f"handed", f"{REAL_HOLD_S} / {ModeHost.HOLD_S}")
h2 = host(size=6)
h2.hold_s = REAL_HOLD_S                         # the product's value, for this one
shoot(h2, 6)
h2.advance(ctx(now=1.10))                       # inside the hold, before it closes
rescued = h2.add_rounds(MAGAZINE_TOPUP)
check(not h2.round_over and rescued == 16 and h2.remaining(0) == 10,
      "a top-up DURING the hold after the last shot rescues the round",
      f"over={h2.round_over} got={rescued}")

# the screens that have no magazine ---------------------------------------------
hm = ModeHost()
hm.set_mode(modes.by_name("menu")(), C4, players=1)
check(hm.add_rounds(MAGAZINE_TOPUP) is None and not hm.counts_ammo,
      "the MENU has no magazine to top up, and says so by refusing")
hk = ModeHost()
hk.set_mode(modes.by_key(Qt.Key_K)(), C4, players=1)
check(hk.add_rounds(MAGAZINE_TOPUP) is None and not hk.counts_ammo,
      "...nor does the Keyboard Shortcuts screen (it spends no rounds either)")

# the ceiling --------------------------------------------------------------------
hc = host(size=MAGAZINE_KEY_MAX)
seen = [hc.add_rounds(MAGAZINE_TOPUP) for _ in range(6)]
check(max(x for x in seen if x is not None) == MAGAZINE_MAX
      and hc.round_magazine == MAGAZINE_MAX,
      f"the top-up is clamped at {MAGAZINE_MAX}, however many times it is pressed",
      f"{seen}")
check(hc.add_rounds(MAGAZINE_TOPUP) == MAGAZINE_MAX,
      "...and at the ceiling it returns the size unchanged rather than None, so "
      "the app can tell 'full' from 'no round to extend'")

# two players --------------------------------------------------------------------
h2p = host(players=2, size=6)
shoot(h2p, 6, x=100.0)
check(h2p._side_done(0) and not h2p.round_over and h2p.remaining(1) == 6,
      "two players: the left half is out, the right half is not")
check(h2p.add_rounds(MAGAZINE_TOPUP) == 16
      and h2p.remaining(0) == 10 and h2p.remaining(1) == 16,
      "...and the top-up deals ten to BOTH halves: one magazine size, as the "
      "round has always had", f"{h2p.remaining(0)} / {h2p.remaining(1)}")


# =============================================================================
print("\n=== C. MIN_SHRINK is derived: every size he can SET shows every bullet ===")
print("    raising `MIN_SHRINK` past 0.742 hides a bullet of thirty in a "
      "two-player half, which is the case that nearly shipped")

worst = []
for w, hgt in ((3840, 2160), (1920, 1080)):
    c = ctx(w, hgt)
    for n in range(MAGAZINE_KEY_MIN, MAGAZINE_KEY_MAX + 1):
        hs = host(size=n, c=c)
        g = hs.single_strip_geometry(c)
        if g.truncated or len(g.slots) != n or not QRect(0, 0, w, hgt).contains(g.bounds):
            worst.append(f"{w}x{hgt} single {n}: {len(g.slots)} slots")
        ht = host(players=2, size=n, c=c)
        lay = ht._layout(c)
        for side in (0, 1):
            gg = ht.half_strip_geometry(c, lay, side)
            if gg.truncated or len(gg.slots) != n or not lay.side(side).contains(gg.bounds):
                worst.append(f"{w}x{hgt} half {side} {n}: {len(gg.slots)} slots")
check(not worst,
      f"every magazine from {MAGAZINE_KEY_MIN} to {MAGAZINE_KEY_MAX}, single and "
      f"both halves, at 4K and 1080p, draws ALL its bullets inside its field",
      str(worst[:4]))

# the ratio the constant is set from, stated as a measurement
c = ctx()
hbig = host(players=2, size=MAGAZINE_KEY_MAX, c=c)
lay = hbig._layout(c)
gh = hbig.half_strip_geometry(c, lay, 0)
nominal = max(8, round(2160 * mag.BULLET_H_FRAC))
check(mag.MIN_SHRINK <= gh.bullet_h / nominal + 1e-9,
      f"the floor ({mag.MIN_SHRINK}) sits under what thirty rounds in a half "
      f"actually need ({gh.bullet_h / nominal:.3f} of nominal), with margin",
      f"bullet {gh.bullet_h}px of a nominal {nominal}px")
check(0.5 < mag.MIN_SHRINK < 1.0,
      "...and it is a real floor: at 1.0 nothing could ever shrink, at 0 nothing "
      "would ever truncate", str(mag.MIN_SHRINK))

# past what the keys can set, the strip truncates instead of shrinking away ------
for n in (40, MAGAZINE_MAX):
    hs = host(size=n)
    g = hs.single_strip_geometry(C4)
    check(g.capacity == n and g.truncated and len(g.slots) >= 20
          and QRect(0, 0, 3840, 2160).contains(g.bounds)
          and g.bullet_h >= int(nominal * mag.MIN_SHRINK) - 1,
          f"a topped-up magazine of {n} shows {len(g.slots)} countable bullets "
          f"rather than {n} tiny ones, still inside the screen",
          f"{len(g.slots)} slots, bullet {g.bullet_h}px, bounds {g.bounds}")
    hs2 = host(players=2, size=n)
    lay2 = hs2._layout(C4)
    for side in (0, 1):
        gg = hs2.half_strip_geometry(C4, lay2, side)
        check(gg.capacity == n and len(gg.slots) < n and lay2.side(side).contains(gg.bounds),
              f"...and in half {side} of a two-player screen, still inside the half",
              f"{len(gg.slots)} of {n}, bounds {gg.bounds}")


# =============================================================================
print("\n=== D. a truncated strip under-DRAWS and never under-COUNTS ===")
print("    clamping the numeral to the slots instead of the capacity is the "
      "fault here: the strip would say 32 with 50 rounds in the gun")


class _Recorder:
    """Stands in for the painter and the text fitter, so the check reads the
    text the REAL `magazine.paint` produces (RULES C12: a stand-in mirrors the
    real thing -- it takes the same calls and records them, it does not
    paraphrase the drawing)."""

    def __init__(self) -> None:
        self.bullets = 0
        self.texts: list[str] = []

    def drawPixmap(self, *a) -> None:          # noqa: N802  (Qt's name)
        self.bullets += 1

    def setPen(self, *a) -> None:              # noqa: N802
        pass

    def draw_lines(self, _p, _rect, lines, _size, bold=False) -> None:
        self.texts.extend(str(x) for x in lines)


def drawn(n_capacity: int, remaining: int):
    h_ = host(size=n_capacity)
    g = h_.single_strip_geometry(C4)
    rec = _Recorder()
    mag.paint(rec, g, remaining, C4, rec.draw_lines)
    return g, rec


g50, rec50 = drawn(MAGAZINE_MAX, MAGAZINE_MAX)
check(rec50.texts == [str(MAGAZINE_MAX)],
      f"a full magazine of {MAGAZINE_MAX} on a truncated strip still COUNTS "
      f"{MAGAZINE_MAX}", f"the numeral read {rec50.texts}")
check(rec50.bullets == len(g50.slots) < MAGAZINE_MAX,
      f"...while drawing the {len(g50.slots)} bullets that fit, and no more",
      f"{rec50.bullets} bullets in {len(g50.slots)} slots")

for left in (MAGAZINE_MAX - 1, 40, len(g50.slots) + 1):
    _g, rec = drawn(MAGAZINE_MAX, left)
    check(rec.texts == [str(left)] and rec.bullets == len(g50.slots),
          f"with {left} left the numeral says {left} and the row stays full",
          f"{rec.texts} / {rec.bullets}")

for left in (len(g50.slots), 7, 1, 0):
    _g, rec = drawn(MAGAZINE_MAX, left)
    check(rec.texts == [str(left)] and rec.bullets == left,
          f"once {left} are left the row empties bullet by bullet, as it always "
          f"has", f"{rec.texts} / {rec.bullets}")

_g, rec_over = drawn(17, 99)
check(rec_over.texts == ["17"] and rec_over.bullets == 17,
      "a count larger than the magazine is still clamped to the magazine (the "
      "strip may not invent rounds)", f"{rec_over.texts}")
_g, rec_neg = drawn(17, -3)
check(rec_neg.texts == ["0"] and rec_neg.bullets == 0,
      "...and a negative count reads 0, not a crash under his finger")

# and the same thing through the real painter, from rendered pixels --------------
hpx = host(size=MAGAZINE_MAX)
gpx = hpx.single_strip_geometry(C4)
_red, _lit, img = fh.full(hpx.paint, C4)
inked = sum(1 for r in gpx.slots
            if int(img[r.top():r.bottom() + 1, r.left():r.right() + 1, :3].max()) > 22)
nb = gpx.numeral
check(inked == len(gpx.slots)
      and int(img[nb.top():nb.bottom() + 1, nb.left():nb.right() + 1, :3].max()) > 22,
      f"RENDERED: all {len(gpx.slots)} slots of the truncated strip have a "
      f"cartridge in them, and the numeral is drawn beside them",
      f"{inked} of {len(gpx.slots)} inked")


# =============================================================================
print("\n=== E. the keys are bound, discoverable, and collide with nothing ===")
print("    reverting a row of the Keys screen or the bottom legend leaves a key "
      "nobody can find")

from lsa.display.layers import KeyHelp                    # noqa: E402
from lsa.introspect.gui import resolve_key                # noqa: E402
from lsa.modes import shortcuts                           # noqa: E402

check(Qt.Key_Up in CameraLab.MAG_UP_KEYS and Qt.Key_Down in CameraLab.MAG_DOWN_KEYS,
      "the app binds Up to a bigger magazine and Down to a smaller one")
check(CameraLab.MAG_TOPUP_KEYS == (Qt.Key_Right,),
      "...and the RIGHT arrow to ten more rounds in this round",
      "`+` is the volume and the manual says so; `R` is the camera pane's red")

magkeys = (set(CameraLab.MAG_UP_KEYS) | set(CameraLab.MAG_DOWN_KEYS)
           | set(CameraLab.MAG_TOPUP_KEYS))
others = (set(CameraLab.VOLUME_UP_KEYS) | set(CameraLab.VOLUME_DOWN_KEYS)
          | set(CameraLab.SPEED_UP_KEYS) | set(CameraLab.SPEED_DOWN_KEYS)
          | {int(Qt.Key_BracketLeft), int(Qt.Key_BracketRight), int(Qt.Key_R),
             int(Qt.Key_D), int(Qt.Key_W), int(Qt.Key_Y), int(Qt.Key_J),
             int(Qt.Key_Escape), int(Qt.Key_Space), int(Qt.Key_Backspace)})
check(not (magkeys & others),
      "the magazine keys take nothing that was already in use: not the volume, "
      "the drill speed, the camera pane's keys, or Escape",
      str(sorted(magkeys & others)))
check(all(modes.by_key(k) is None for k in magkeys),
      "...and no drill answers to an arrow either")

from lsa.display.window import TargetWindow                # noqa: E402

check(not (magkeys & set(int(k) for k in TargetWindow.REPEATABLE_KEYS)),
      "an arrow is NOT repeatable: a held Right would have put the magazine at "
      "the ceiling from one lean on the keyboard")
check(not any(k in TargetWindow.TOGGLES for k in magkeys),
      "...and the window claims no arrow before the app sees it")

row = next((e for e in shortcuts.MAGAZINE if int(Qt.Key_Up) in e.keys), None)
check(row is not None and int(Qt.Key_Down) in row.keys,
      "the K (Keyboard Shortcuts) screen carries Up and Down on one row, with "
      "the real Qt key codes and not just a glyph")
top = next((e for e in shortcuts.MAGAZINE if int(Qt.Key_Right) in e.keys), None)
check(top is not None and "10" in top.title,
      "...and the Right arrow's row says how many rounds it adds",
      getattr(top, "title", "MISSING"))
check(shortcuts.HEAD_MAGAZINE in [h for h, _e in shortcuts.groups()[0]],
      "...and the group is on the screen, in the column with room for it")
check(all(e in shortcuts.advertised() for e in shortcuts.MAGAZINE),
      "...so `advertised()` carries them, which is what test_modes section 46 "
      "cross-checks against the app's own key handler")

legend = " ".join(KeyHelp().lines).lower()
check("up dn" in legend and "right" in legend and "magazine" in legend,
      "the bottom legend names them too, for a shooter who never presses K")
check("this round only" in legend,
      "...and says that the ten rounds are for this round only",
      "the one thing about the key that is not guessable")

check(resolve_key("Up") == int(Qt.Key_Up) and resolve_key("Down") == int(Qt.Key_Down)
      and resolve_key("Right") == int(Qt.Key_Right),
      "the introspection server can press all three without his hands")


# =============================================================================
print("\n=== F. in the app: the key moves the setting, the host and the file ===")
print("    this is the C8 half -- the path is CALLED, not scanned for a "
      "plausible method name")

import lsa.settings as _settings                           # noqa: E402
import lsa.tools.camera_lab as cl                          # noqa: E402

_orig_load = _settings.AppSettings.load


@staticmethod
def _no_connect(path):
    st = _orig_load(path)
    st.auto_connect = False
    return st


_settings.AppSettings.load = _no_connect
cl.CameraLab.hard_exit_on_close = False
try:
    lab = cl.CameraLab()
finally:
    _settings.AppSettings.load = _orig_load

saves: list[int] = []
real_save = lab._save_settings
lab._save_settings = lambda *a, **k: None
lab._schedule_save = lambda *a, **k: saves.append(1)
check(not _opened, "building the lab opened no audio device")
check(os.environ.get("LSA_SETTINGS") == str(cl.SETTINGS_PATH)
      and Path(cl.SETTINGS_PATH).resolve() != (HERE.parent / "settings.local.json").resolve(),
      "the settings file under test is the isolated copy, not his",
      str(cl.SETTINGS_PATH))

lab_host = lab.layers.get("modes")
check(lab_host is not None and lab_host.magazine == normalise_magazine(lab.settings.magazine),
      "the app hands the saved magazine to the host at startup")

lab.settings.magazine = 10
lab_host.set_magazine(10)
lab._on_target_key(Qt.Key_Up)
check(lab.settings.magazine == 11 and lab_host.magazine == 11,
      "Up moves the setting AND the host together, from one keystroke",
      f"settings {lab.settings.magazine}, host {lab_host.magazine}")
lab._on_target_key(Qt.Key_Down)
lab._on_target_key(Qt.Key_Down)
check(lab.settings.magazine == 9 and lab_host.magazine == 9, "Down lowers both")
check(saves, "...and a change schedules the settings save")

for _ in range(40):
    lab._on_target_key(Qt.Key_Down)
check(lab.settings.magazine == MAGAZINE_KEY_MIN,
      f"holding Down lands on {MAGAZINE_KEY_MIN} and stays there",
      str(lab.settings.magazine))
for _ in range(40):
    lab._on_target_key(Qt.Key_Up)
check(lab.settings.magazine == MAGAZINE_KEY_MAX,
      f"...and Up lands on {MAGAZINE_KEY_MAX}", str(lab.settings.magazine))

lab.settings.magazine = 12
lab_host.set_magazine(12)
real_save()
saved = json.loads(Path(cl.SETTINGS_PATH).read_text(encoding="utf-8"))
check(saved.get("magazine") == 12,
      "the saved file holds the magazine, so it survives a relaunch",
      f"magazine={saved.get('magazine')}")

# the top-up, through the key, with a drill up ----------------------------------
lab_host.set_mode(_Static(), C4, players=1)
lab_host.set_magazine(12)
lab_host.restart(C4)
shoot(lab_host, 4)
before = lab_host.round_magazine
lab._on_target_key(Qt.Key_Right)
check(lab_host.round_magazine == before + MAGAZINE_TOPUP
      and lab.settings.magazine == 12,
      "the Right arrow tops up the ROUND and leaves the saved size alone",
      f"round {lab_host.round_magazine}, saved {lab.settings.magazine}")
n_saves = len(saves)
lab._on_target_key(Qt.Key_Right)
check(len(saves) == n_saves,
      "...and schedules no save at all: it is for this session only",
      f"{len(saves) - n_saves} saves")

# the refusals SAY something (C1: a key that quietly does nothing is the fault)
lines: list[str] = []
lab._log = lambda m: lines.append(m)
lab_host.set_mode(modes.by_name("menu")(), C4, players=1)
lab._on_target_key(Qt.Key_Right)
check(lines and "no magazine" in lines[-1].lower(),
      "on the MENU the Right arrow REPORTS that there is no magazine to top up",
      lines[-1] if lines else "nothing logged")
lab_host.set_mode(_Static(), C4, players=1)
lab_host.set_magazine(6)
lab_host.restart(C4)
shoot(lab_host, 6)
lab_host.advance(ctx(now=9.0))                  # let the hold expire and close it
lines.clear()
lab._on_target_key(Qt.Key_Right)
check(lines and "closed" in lines[-1].lower() and "press n" in lines[-1].lower(),
      "...and on a closed round it says so and names the key that starts a new "
      "one", lines[-1] if lines else "nothing logged")
lines.clear()
lab._on_target_key(Qt.Key_Up)
check(lines and "magazine" in lines[-1].lower()
      and ("next round" in lines[-1].lower() or "ends at" in lines[-1].lower()
           or "changed with it" in lines[-1].lower()),
      "a size change REPORTS the number and which round it applies to (C3)",
      lines[-1] if lines else "nothing logged")


# =============================================================================
owned.check(lambda ok, what, detail="": check(ok, what, detail))
print("\n" + (f"ALL {CHECKS} CHECKS PASSED" if not FAILED else
              f"{len(FAILED)} OF {CHECKS} CHECKS FAILED"))
raise SystemExit(0 if not FAILED else 1)
