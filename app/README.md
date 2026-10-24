# Laser Shooting App

Dry-fire laser training on a PC monitor. Targets appear on screen, a USB camera
watches the screen, and when you fire a laser trainer pistol at it the app works
out which pixel you hit, marks it and makes a noise.

Built for Windows, for LaserLyte LT-TTL trainers and a Logitech C920 — though
nothing is hard-coded to those.

**Status: the game exists.** Nine drills, a seventeen-round magazine, scoring, a
shoot-at-it menu and a two-player split screen, packaged as a single Windows
executable. Everything below this line is still the Camera Lab instrument the app
was built on top of, and is out of date as a description of the product.

---

## The screen it needs

**A matte (anti-glare) monitor.** This is the one hardware requirement nothing
else can make up for, so it is worth settling before anything else.

**Nothing goes on top of the screen.** Some camera-based systems ask you to tape
a film over a flat screen before the camera can see the laser at all. The largest
system in the category states it plainly in its own TV setup guide: *"Use with TV
or monitor will require our filter material that is taped to the screen in order
to track the laser"*, and explains why — *"Some TVs will only reflect the laser at
an angle incident to the angle the laser enters the screen. Therefore the camera
will not be able to see the laser and track it."* A matte panel is the case where
that is not true, which is why it is required here and why nothing goes on top of
it. Checked against vendor documentation on 23 October 2026.

The app finds your shot by watching the screen for the laser spot, and it can only
see that spot if the screen scatters some of the beam back toward the camera. A
matte screen is micro-textured and throws light in every direction at once, so the
spot is visible from almost any angle — which is the whole reason an ordinary
webcam parked off to one side works at all. A glossy screen behaves like a mirror:
shoot it at an angle and most of the beam leaves at the matching angle on the far
side, away from the camera, while an anti-reflection coating weakens what little
comes back and the panel's own layers absorb much of the rest.

So a glossy screen often registers hits fired square on and misses most of those
fired from an angle, and **moving the camera does not fix it**: the mirror
direction depends on where the shot landed, so it changes with every shot and no
fixed camera position can follow it.

| Display | Status |
|---|---|
| Matte / anti-glare computer monitor | Supported |
| Glossy or semi-gloss monitor | Not supported |
| Glossy TV — most LCD, QLED and OLED sets | Not supported |
| Projector and screen | Not supported in v1 |

The projector is the one entry there that is not about gloss. A projection screen
is diffuse and scatters the dot perfectly well; the obstacle is the palette. The
app draws light grey on black, and a projector makes black by projecting nothing,
so the dark parts of the picture are whatever the room does to a white screen.
Making it work means inverting the whole scheme — dark targets on a lit field —
and with it every brightness rule derived from the present one. That is a
different app rather than a setting, and it is not in version 1.

| Tested | Finish | Result |
|---|---|---|
| Dell UltraSharp U4320Q, 43" 4K | Matte | Hits visible at every angle tried |
| Samsung 75" QLED, 2019 Q7 series | Glossy, anti-reflective | Sometimes visible head-on, mostly invisible at an angle |

**To check yours:** turn it off and stand with a lamp or window behind you. A
sharp, mirror-like reflection means glossy; a blurred, diffuse glow means matte.
In the specifications, "anti-glare", "matte" or "3H anti-glare coating" is what
you want, and "glossy", "glare type" or "anti-reflection glass" is what you do
not.

The app's self check (**C**) draws a dozen red dots at known places, photographs
them and reports how many it found and how far out it placed them. Read it
carefully: the **position error** is real, but the **detection rate is not your
pistol's** — a screen emits its own test dot while a laser reflects off the
coating, and the two fall off across a screen quite differently. The honest test
of a doubtful screen is a magazine of real shots from where you intend to stand.

---

## Setup

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then:

```bat
run_camera_lab.bat
```

## First run, in order

Work down the numbered panels on the right. They are in this order for a reason.

**1 — Which camera?**
Click *Which camera?*. It grabs one still from each camera so you can see which
one is pointed at the monitor. Note its index, pick it in *Device*, press
*Connect*. Leave the defaults: **MSMF**, 1280×720, MJPG, 30 fps.

> **Do not switch the backend to DSHOW on a C920.** Measured: DirectShow ignores
> the MJPG request, forces YUY2 and delivers 10 fps. That is below the floor the
> detector needs — a 100 ms pulse would land in a single frame. The app warns you
> if the delivered rate drops under 20.

If the resolution or codec you asked for is not what you got, the log says so.
*Scan formats…* opens each candidate format in turn and **times real frame
delivery** — the MEASURED column is the only one worth believing, because a
webcam that reports 30 fps and delivers 5 is a common and silent failure.

**2 — Exposure.**
Click *Probe controls* first. It nudges every control, reads it back, and greys
out the ones the driver ignores. Then click **Laser hunt (dark)**. The preview
should go almost black — that is correct and is the entire trick: with the
screen underexposed to a ghost, the laser is the only thing bright enough to
register. If exposure refuses to stick, use *Driver dialog…* to open Logitech's
own property sheet, which can set things OpenCV's pass-through sometimes cannot.

**3 — Detector.**
Click **Measure noise floor (3 s)** first, and do not fire while it runs. It
finds the brightest thing the detector sees with no laser present and sets the
threshold above it. Guessing that number is what produces shots you never fired.

Keep **Laser-safe preview (blue)** ticked. The preview shows the camera feed on
the same monitor the camera is watching, which is an infinite mirror — the laser
dot recurses inside it and every reflection looks like a real hit. Rendering the
preview multiplied by blue zeroes the red channel so it cannot register.

Then leave the rest alone at first. Fire at the screen and watch the preview. A green
crosshair should snap onto the dot. Tick *Show mask* to see exactly what the
detector sees. If nothing is found, lower *Threshold* and *Min peak*. If
everything is found, raise them.

**4 — Shot tracking.**
`Min frames per shot = 2` is the useful default: a 100 ms LaserLyte pulse spans
about three frames at 30 fps, so demanding two throws away single-frame sensor
noise for free.

**5 — Target screen (F11).**
Fullscreen aiming field with a scoreboard, for actually standing back and
shooting. Keys on that screen:

| Key | Does |
|---|---|
| `P` | camera pane on/off |
| `S` | stats panel on/off |
| `A` | aim marks on/off |
| `I` | detector diagnostics on/off |
| `H` | key help on/off |
| `D` `W` `R` `G` | background: dark / white / red / grey |
| `C` | clear hit markers |
| `M` `[` `]` | move / resize the camera pane |
| `Esc` | back to the lab |

`W` and `R` are stress tests, not decoration. White is the hardest background
for any detector; red is the hardest specifically for the chroma detector. If
detection survives both, your room and your target art will not trouble it.

**6 — Hit-rate test.** The number that decides the project. Press *Reset
counters*, go to the target screen, fire a known number of shots, come back and
type how many you fired. Anything under 95% means more tuning.

**7 — Recording.** Saves video, a per-frame detection CSV and every setting, so
detection can be retuned offline against your real room instead of you standing
there shooting while someone guesses.

---

## Camera placement

The camera does not need to be square-on to the screen; an angle is expected and
fine. Because the laser lands on the actual screen pixel there is no parallax
error, and a homography describes an oblique view of a flat screen exactly.

Two things to keep in mind. Past roughly 45° the far side of the screen gets few
enough camera pixels that accuracy suffers. And the camera must see all four
corners of the monitor — calibration needs them.

## Tests

```bat
run_tests.bat
```

Neither suite needs a camera or a display. `test_detector.py` plays synthetic
100 ms pulses through the real pipeline — including two guns fired at once, two
guns offset in time, rapid fire at one spot, and an off-axis elliptical dot.
`test_ui_smoke.py` builds the entire UI headless and renders the target screen
to `target_screen.png`, checking among other things that nothing drawn on it is
red enough for the camera to mistake for a laser.

## Safety

The LaserLyte trainers are 5 mW Class IIIa. They will not harm an LCD. Do not
look into the muzzle.
