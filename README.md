# ScreenPlink

**Dry-fire laser training where the monitor itself is the target.** The app draws
targets on your screen, an ordinary USB webcam watches that screen from the side,
and when you fire a laser trainer pistol at it the app works out which pixel you
hit, marks it, and makes a noise.

No paper targets. No projector. Nothing laid over the screen — no diffusion
film, no cling film, no fabric. No proprietary camera, no infrared, and nothing
attached to the gun beyond the laser cartridge you already own. Free, offline,
no account.

<!-- DELIBERATE DIFFERENCE, DO NOT "FIX" IT (agreed with the trailer session,
     2026-10-24). The trailer narration says "No projector. No special screen.
     No paper targets." This file deliberately does NOT say "no special screen",
     and states the matte requirement instead.

     They are not in conflict. In the trailer, "no special screen" means nothing
     you have to buy, fit or lay over the panel -- true, and the contrast that
     lands in eight seconds of voiceover. The README and the video description
     are where there is room to say which panels actually work, which is why the
     caveat was assigned here rather than to the script.

     The narration is already rendered to audio in two voices, so changing that
     line costs a re-render, not an edit. If it ever does change, the trailer
     session sends the exact replacement and THIS file follows it: the README is
     the canonical phrasing and the video description is lifted from the two
     sentences above. Do not align them the other way round, and do not quietly
     reword either one to match the other. -->

**The one thing it does require is a matte, anti-glare monitor** — not glossy,
not a television. That is physics, not a preference, and it is the first thing
to check before anything else here.

> **Read the [User Guide](manual/README.md) first if you just want to shoot.**
> Thirteen chapters, plus PDF and EPUB, indexed there. Start with
> [what the screen has to be](manual/chapters/01-the-screen-it-needs.md) — it
> decides whether anything else works. This file is for people who want to know
> how it works, or to change it.
>
> Something not working? **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — start
> there, and start at the top: the most common fault is not in this app at all.
>
> Finding your way around the source? **[PROJECT-LAYOUT.md](PROJECT-LAYOUT.md)**.

---

## What it is, precisely

> ScreenPlink turns a compatible matte monitor into a two-player laser dry-fire
> target, using an ordinary USB webcam positioned off-axis — with measured
> camera-to-screen calibration, sensitivity learned from real shots, and a
> camera-verified simulated-shot self-test. It runs free and offline, without an
> account or installer.

### What is deliberately NOT claimed

Two adversarial reviews were commissioned on 2026-10-23 and told to *disprove* a
list of capability claims rather than confirm them. **They found predecessors, so
no "first" and no "only" is claimed anywhere in this project.**

Three claims are settled against us by both reviews independently, and are never
asserted here in any form: the display being the target surface, an off-axis
camera with a software-measured mapping, and two shooters on one shared screen
have all been done before. Prior work worth knowing about:

- **Electron** (electrontir.ru) — a camera-based interactive range that
  specifies a matte LED monitor and advertises simultaneous independent lanes.
- **Laserboard** ([Ben-Kaniobi/Laserboard](https://github.com/Ben-Kaniobi/Laserboard))
  — an external camera detecting a laser on a normal monitor, with automatic
  perspective calibration.
- **opendryfire** — an OpenCV homography for deliberately oblique camera
  placement, onto printed targets.

Several other claims came back *supported* in one review and *insufficient* in
the other. The brief's own tie-break resolves a disagreement to the negative
until the counterexample has been opened and read by hand, and nobody has done
that yet; one review also states that it is interim and that two named patents
were never examined. So the position is stated in the reviewers' words rather
than ours:

> "This review did not verify another implementation combining all of these
> features, but it did identify predecessors for several individual
> capabilities and did not establish a historical first."

### What the nearest products require, that this does not

This comparison holds up without any claim about novelty. Provenance, stated
rather than implied: the first two rows are quoted from the vendors' own setup
documentation in the reviews; the rest are published vendor statements as
recorded in those reviews, not read by us from the primary page.

| Product | What it needs for a flat screen |
|---|---|
| **Laser Ammo Smokeless Range** | A proprietary tracking camera, and for a TV, a sheet of GAMFusion 10‑55 diffusion film laid across the panel. |
| **iFightBack** | Reflective cling film applied to the screen. |
| **ShootOFF** (dormant since ~2021) | Sheer fabric or projection cloth stretched over the glass. |
| **Tiradot** | A projector; TV support listed as "in development". |
| **LASR / LASR‑X, Strikeman, Mantis** | A paper target, a wall, or a phone screen. |

Here: an unmodified matte panel, nothing laid over it, an ordinary webcam off to
one side.

### The finding that shaped the engineering

Three competitive-landscape reports from 2026-10-18 agree on one thing that
matters more than any feature: the dominant reason people abandon this category
is **not** thin content — it is setup friction and unreliable detection. Users
keep the laser cartridge and drop the app. That is why most of the work here went
into a launch that ends in READY in about thirty seconds, or tells you exactly
what to fix.

---

## How it works

### The dot

A LaserLyte LT‑TTL trainer emits roughly **100 ms of 650 nm red** per trigger
pull, about 5 mW, Class IIIa. It will not harm an LCD.

### Why the panel has to be matte

A **matte** anti-glare coating scatters the dot diffusely, so it can be seen from
well off to one side. A **glossy** panel reflects it specularly, away from the
camera, in a direction that changes with every shot position — so no camera
placement can follow it.

**A matte, anti-glare monitor — not glossy, not a television.** The measurements
behind that, including the glossy panel this was tried on and failed, are
[chapter 1 of the guide](manual/chapters/01-the-screen-it-needs.md).

### Why 60 frames per second

100 ms at 60 fps puts the pulse in **about six camera frames**. That is what makes
the whole thing feasible: a shot is confirmed from several frames rather than one,
so a single bright frame of noise is not a hit, the position can be averaged, and
two shots can be told apart. At 30 fps the same pulse lands in three frames and
the margin disappears.

### What "seeing the dot" actually means

The detector does not look for *bright*. It looks for **red in excess of the other
channels** — `red − max(green, blue)` — which a 650 nm dot produces overwhelmingly
and a grey target does not produce at all.

Two consequences run through the entire app, and explain things that otherwise
look arbitrary:

- **Nothing drawn on the target screen may be red**, because the camera cannot
  tell the app's own artwork from a laser. Everything is neutral grey, and the
  tests measure red excess per pixel on rendered frames rather than trusting the
  palette.
- **Nothing drawn may be too bright either.** A dot landing on a lit grey target
  scores *lower*, because red saturates while green and blue are already high.
  Measured: the same dot scores 245 on near-black and 55 on grey at luminance 200.
  So every target is capped at the brightness of the darkest supplied artwork.
  That is why the whole game is grey and dim; [chapter 0](manual/chapters/00-what-it-is.md)
  puts it in the shooter's terms.

### Where the shot landed

The camera sees the screen as a trapezoid, so the app fits a **homography** —
exact for a flat screen viewed from any angle — plus **one radial lens term**,
because a wide-angle webcam is not a pinhole. The fit is measured, never assumed:
the app photographs its own markers at known screen positions, lit and unlit, and
subtracts. Twelve markers fit the mapping and six more, at positions in between,
are held back to measure it. **Held-out error is the number that is reported**,
typically well under a millimetre, because an error measured against the points it
was fitted to flatters any model.

### Why the far side of the screen is harder

Viewed from an angle, the far side of the panel is further away *and* at a steeper
angle, so the dot is both dimmer and spread over roughly six times fewer camera
pixels. Measured on one session: mean peak 62 in the centre against 34 at the far
edge. A single threshold cannot serve both ends, so the app learns a **per-region
sensitivity map from real shots** and normalises the score, which is also why the
app gets better at the far corner the more it is used.

### Testing itself before you fire

The screen can test the camera, which is how the app knows it is working before
a trigger is pulled: it draws its own red dots at known positions, at the
brightness of real recorded hits, photographs them through the same camera and
the same mapping your shots use, and reports READY or the specific reason not.

**Read that number with one caveat, or it misleads.** A screen *emits* its test
dot while a laser *reflects* off the panel's coating, and the two fall off across
a screen quite differently — measured here, the synthetic dot held roughly steady
from side to side while a real laser dropped by half. So the **position error is
real**, because it is measured through the same mapping your shots are; the
**detection rate is not your pistol's**. It proves the app can see dots on your
screen and place them accurately. It cannot tell you what fraction of your
trigger pulls will register.

### Hearing the gun

The camera's microphone is an independent channel. It answers a question the
camera cannot: *did a trigger pull happen at all?* Heard and seen means working;
heard but not seen means a shot was missed; seen but not heard is a phantom.
**Audio never creates a hit** — it has no position, and somewhere is the one thing
it cannot know.

---

## What you need

| | |
|---|---|
| **Display** | A **matte** monitor. Glossy panels and projectors are not supported. |
| **Camera** | Any USB webcam that does **60 fps** over MSMF. Developed against a Razer Kiyo Pro; a C920 works at 30 fps with less margin. |
| **Laser** | A visible red (650 nm) laser trainer cartridge or trainer pistol. Two are supported at once. |
| **Room** | Whatever you have. The app measures the room it is in and sets its own exposure from that measurement; it does not ask you to change the lighting. |
| **OS** | Windows. Python 3.13 if you run from source. |

The camera goes **beside, above or below the monitor**, angled at it — not in
front of it, and not downrange. It must see all four corners of the screen.

## Running it

From a release (no Python needed): unzip and run **`Play.cmd`**. `Doctor.cmd`
checks the camera and writes a report you can read or send on.

From source:

```bat
python -m venv app\.venv
app\.venv\Scripts\python.exe -m pip install -r app\requirements.txt
app\scripts\Play.cmd
```

Close it with **Escape** or the Exit tile — never End Task. A force-killed app can
leave the camera needing a physical replug, and `app\scripts\stop_lab.ps1` exists
to close it the way the app would.

## Tests

```bat
app\scripts\run_tests.bat
```

Twenty-six suites, no camera and no display needed — they run offscreen with the
camera disabled. Alongside them are six **mutation runners**, which put known
faults back into the code on purpose and fail the build unless the check written
for each one goes red. A check that cannot fail is not a check, and this project
has shipped several; that is what those runners are for.

`app\tests\test_coldstart.py` drives the whole startup chain against a simulated
camera, seven launches end to end. It is the one that finds the faults that matter.

## Safety

The LaserLyte trainers are 5 mW Class IIIa. They will not harm an LCD. **Do not
look into the muzzle.** Clear the firearm; this is dry-fire practice.
