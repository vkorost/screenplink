# ScreenPlink

A laser dry-fire range on a monitor you already own. The screen shows targets, a
webcam watches the screen, and when you fire a laser trainer pistol at it the app
works out which pixel you hit, draws a bullet hole there and plays a shot.

**Nothing goes on your screen.** Some camera-based systems ask you to tape a film
over a flat screen before they can see the laser — the largest one in the
category supplies a vinyl sheet for it. The requirement here is instead that the
screen be **matte** to begin with, which is the same fact seen from the other
side, and the section below explains why.

## Before you download it: three pieces of hardware

Everything here is about kit you own. None of it is a setting, and none of it can
be worked around in software — so it is worth checking first rather than after a
161 MB download.

**1. A matte, anti-glare monitor.**
Not a glossy screen, and not a television. A glossy panel reflects the laser away
from the camera the moment you shoot it from an angle, and no camera position can
recover it. *To check in ten seconds: switch the screen off and stand with a lamp
or a window behind you. A sharp mirror image is glossy. A blurred, diffuse glow is
matte.* The next section explains why it matters that much.

**2. A webcam that delivers 60 frames per second.**
A laser pulse lasts about a tenth of a second. At 60 fps it lands in roughly six
camera frames, and the app needs two of them to agree before it will call a shot.
At 30 fps it lands in three, which leaves almost no margin. A camera that quietly
drops to 10 fps — which several of the webcams tested here do, without saying so —
puts the whole pulse in one frame, and then nothing works at all.

Check the *measured* rate rather than the box: `Doctor.cmd` times real frame
delivery, with the exposure pinned first so a dim room cannot flatter or spoil the
number, and reports how many frames a pulse would land in. The camera also has to
see the whole screen with a margin around it.

**3. A laser trainer pistol, or a laser cartridge for the pistol you own.**
Anything that flashes a red dot when the trigger breaks. Built and measured
against a LaserLyte LT-TTL.

Plus Windows 10 or 11. Nothing is installed and nothing touches the registry —
unzip it and run it. No projector, no special screen, no sensor bar, and **the
camera does not have to sit in front of the monitor**: off to one side at an angle
is normal and fully supported.

## The screen has to be matte

ScreenPlink finds your shot by watching the screen for the laser spot, and it can
only see that spot if the screen scatters some of the beam back toward the camera.

A **matte** screen is micro-textured and throws light in every direction at once,
so the spot is visible from almost anywhere — which is the whole reason a webcam
off to one side works at all. A **glossy** screen behaves like a mirror: shoot it
at an angle and most of the beam leaves at the matching angle on the far side,
away from the camera, and an anti-reflection coating weakens what little returns.
Such a screen often registers hits fired square on and misses most of the ones
fired from an angle, and **moving the camera cannot fix it** — the mirror
direction changes with every shot position, so no fixed camera can follow it.

| Display | Works |
|---|---|
| Matte / anti-glare computer monitor | Yes |
| Glossy or semi-gloss monitor | No |
| Glossy TV — most LCD, QLED and OLED sets | No |
| Projector and screen | Not in version 1 |

The projector is the one entry there that is not about gloss. A projection screen
is diffuse and would scatter the dot perfectly well; the obstacle is the palette.
This app draws light grey on black, and a projector makes black by projecting
nothing — so making it work means inverting the whole scheme, dark targets on a
lit field, along with every brightness rule derived from the present one. That is
a different app rather than a setting.

Besides the reflection test above, the specification will tell you: "anti-glare",
"matte" or "3H anti-glare coating" is what you want, and "glossy", "glare type" or
"anti-reflection glass" is what you do not.

Tested: a Dell UltraSharp U4320Q (43", matte) — hits visible at every angle tried,
and the screen everything in this app was measured on. A Samsung 75" QLED (2019
Q7 series, glossy anti-reflective) — sometimes visible head-on, mostly invisible
at an angle.

**Keep the folder together.** `screenplink.exe` is only about 6 MB because it is
a launcher — everything it needs (Python, Qt, OpenCV, the artwork, the sounds)
lives in the `_internal` folder beside it, and `Play.cmd`, `Doctor.cmd` and
`.mcp.json` expect to sit next to the .exe. Copying the .exe somewhere on its
own will not work. Move or copy the whole folder.

## Getting it working

The honest answer is that aiming a camera at a screen and detecting a 100 ms
flash of red is fiddly, and it is fiddly in ways that look like the app being
broken. So this ships with an assistant.

**Open this folder in Claude Code and ask it to set the app up.** It will find
your cameras, tell you which one is actually usable, look at what the camera sees
and tell you how to move it, start the app, read what the app measured, and tell
you in plain terms whether you are ready to shoot or what physically has to
change. `CLAUDE.md` in this folder is written for it, and `.mcp.json` connects it
to the running app so it is reading real measurements rather than guessing.

If you would rather do it yourself:

```
Doctor.cmd          pick a camera, and see what it sees
Play.cmd            run the app
```

`Doctor.cmd` writes its report to `doctor-report.txt` in this folder and then
shows it; that file is where the report lives, and it is what Claude Code reads.
Everything is one program, `screenplink.exe`, started with different arguments:
`Play.cmd` and `Doctor.cmd` are just short ways of doing that.

The app measures its own setup every time it starts — the exposure, the noise
floor, the detection threshold, and the mapping from camera to screen — and tells
you on screen when it is ready. The first launch on a new machine takes longer
than later ones, because it has nothing remembered to check against.

## While it is measuring, do not fire

It flashes the screen, draws small markers and fires test shots at itself. It
says `HOLD FIRE` while that is happening. A real shot during the noise
measurement makes *that shot* the noise floor, and then the bar sits above every
real hit for the rest of the session — which feels exactly like "it worked for
the first few shots and then stopped".

## Closing it

**Escape**, or the **Exit** tile on the menu, or `stop_screenplink.ps1`.

Do not End Task it and do not force-kill it. Cutting a streaming USB camera off
mid-transaction can hang the camera's own firmware, and then the only fix is to
unplug it and plug it back in.

## The drills

```
1 Static Targets    2 Escalating Ball   3 Bouncing Balls
4 Reaction          5 Clear The Board   6 Pop-Up Target
7 Static Target     8 Moving Target     9 Animated Target
```

**M** opens the menu, which you can shoot or click. **O** and **T** switch
between one player and two — in two-player mode the screen splits and two people
shoot at once, one per side.

You get a magazine of 17 rounds. Every shot spends one, hit or miss, and the
round ends when it is empty: this is training, so the point is to know what you
did with the rounds you had. A pull on an empty magazine is a dry pull — it draws
its hole and pops instead of banging, and scores nothing.

After each round you get shots, hits, misses, points, seconds, and two figures:

- **EFFICIENCY**, points earned per second, with the arithmetic printed underneath
  it so there is nothing to take on trust.
- **SCORE**, you against a perfect shooter on that same round: `1000 × aim ×
  speed`, where aim is your points over the most that round could have given and
  speed is how long that shooter would have taken over how long you took. Both
  factors are shown — `AIM 93%  SPEED 27%` says speed up and leave your aim
  alone, which one number could not.

The best of each is remembered per drill, per magazine size and per player count,
and shown beside the round. A round the app cannot believe — shots arriving faster
than any trigger finger can produce them, which is what a burst of phantom
detections looks like — is discarded whole rather than scored, because a score
like that in the record book could never be legitimately beaten.

Other keys worth knowing: **+** and **−** for volume, **E** to erase the holes,
**G** to re-measure the mapping if you have moved the camera, **X** to re-measure
the exposure if the light has changed, **0** to hide the information panels.

## Where your things are kept

Everything is beside the executable, so you can copy the whole folder to another
machine:

```
config\settings.local.json    what has been measured on this machine
config\calibration.json       the camera-to-screen mapping
config\settings.default.json  what the app assumes before it measures anything
logs\                         what happened on each launch
```

To start completely fresh, delete `config\settings.local.json` and
`config\calibration.json`. The app will measure everything again.

## If it is not working

Ask Claude Code — it can read the app's live measurements and the log, and the
instructions it has include the specific failures this project has already met.
Short of that, `logs\` holds an account of every launch, and
`Doctor.cmd` (report in `doctor-report.txt`) will tell you whether the camera itself is fit to
shoot with.

Three things that are almost never what they look like. If the app seems to fire
shots you did not fire, something coloured is on the monitor the camera watches
— move other windows off it. If the signal is weak, **do not darken the room**: a
dark room is the best condition this app can have, and the usual cure is a longer
exposure or getting the camera closer to the screen. And if hits register when you
stand square to the screen but stop when you move to one side, that is the screen
itself — see the matte section above.

The mouse pointer vanishing while you shoot is deliberate, not a fault: a large
bright cursor sitting on the panel is a bright object on the screen the camera is
reading. It comes back as soon as you move the mouse.
