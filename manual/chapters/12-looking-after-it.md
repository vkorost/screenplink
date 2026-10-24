# Looking After It

## Closing it properly

**Escape**, or the **Exit** tile shot twice, or `stop_screenplink.ps1`.

Do not End Task it and do not force-kill it. Cutting a streaming USB camera off
mid-transaction can hang the camera's own firmware, and then the only cure is to
unplug it and plug it back in. On one occasion it took down the whole USB hub and
both cameras with it. Every graceful exit route in the app exists because of that.

Running two copies at once does the same thing: the older one loses the camera
mid-stream. The launcher refuses a second instance for that reason.

## When something changes

| What changed | What to do |
|----------------------------------|--------------------------------------|
| You moved or knocked the camera | **C**, or **G** if the markers will not confirm |
| The light in the room changed | **X** |
| Something bright was on screen during setup | **F** |
| The screen is cluttered | **0**, then **E** to wipe the holes |
| You are not sure it is seeing your shots | **A**, then **Q** to re-read the verdict |

A mapping measured in one camera position is valid only in that position, and
nothing can detect after the fact that a camera has been nudged. If holes start
landing away from the shots, that is the first thing to suspect.

## Where your things are kept

Everything lives beside the executable, so the whole folder can be copied to
another machine:

```
config\settings.local.json    what has been measured on this machine,
                              and the best round it has seen
config\calibration.json       the camera-to-screen mapping, and the
                              sensitivity it has learned from real shots
config\settings.default.json  what the app assumes before measuring anything
logs\                         an account of every launch
```

To start completely fresh, delete `settings.local.json` and `calibration.json`.
The app measures everything again on the next launch.

To clear the **records** only, delete the `bests` section inside
`settings.local.json` and leave the rest of it alone. Every best score and best
efficiency goes; nothing the app measured about your camera or your room does.

**One thing worth knowing before you delete them.** The mapping rebuilds itself in
about ninety seconds, but the sensitivity map -- which evens out the fact that the
far side of an angled screen is dimmer to the camera than the near side -- is
learned only from real shots over time. A folder that has been shot in for a week
is genuinely better than a fresh one on the same hardware in the same room. Back
those two files up rather than throwing them away.

## If it is not working

`Doctor.cmd` tests every camera on the machine and writes `doctor-report.txt`.
Run it with the app closed: two programs on one camera can leave the driver in a
bad state.

The folder also contains a `CLAUDE.md` written for Claude Code, and a `.mcp.json`
that lets it query the running app directly for what the camera is seeing and
what the detector measured. Opening the folder in Claude Code and asking it to
set the app up is the fastest route through a difficult camera.

Four diagnoses that are almost never what they look like:

- **Shots register when you stand square to the screen and stop registering when
  you move to one side.** That is the screen, not the app. A glossy or
  anti-reflective panel sends the beam away from the camera when you shoot at an
  angle, and nothing can be adjusted to recover it. See *The Screen It Needs*.
- **Phantom shots** are something coloured on the monitor the camera watches,
  not a fault in the detector. Move other windows off that screen.
- **A weak signal is not a reason to darken the room.** A dark room is the best
  condition this app can have. The cures are a longer exposure (**X**), a camera
  closer to the screen, or a shorter range.
- **No sound at all**, while the log says the gunshot played. Open the Windows
  volume mixer **with the app running** -- it lists a program only while that
  program is open, which is why this one is so easy to miss -- and look at
  ScreenPlink's own slider. Windows keeps a separate volume for every
  application on every output device, so a fresh install arrives with a fresh
  slider, and one sitting at the bottom is indistinguishable from an app whose
  sound is broken. Three days went into finding that the first time.
