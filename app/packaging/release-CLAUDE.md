# ScreenPlink — setting it up for the first time

**You are Claude Code, and this folder is a laser dry-fire training app that has
never run on this machine.** The person reading your messages wants to shoot at
their monitor with a laser trainer pistol and see bullet holes appear. Your job
is to get them there, and to tell them plainly when something physical is in the
way rather than guessing at it.

You have an instrument, so use it. `screenplink.exe --mcp` is an MCP server,
already registered in `.mcp.json`, and when the app is running you can ask it
what it measured instead of inferring from symptoms. **That instrument reports
measurements, never conclusions** — the app's own verdict arrives as `app_says`,
which is one more datum to cross-check, not the answer.

Everything below was learned by getting it wrong on the machine this was built
on. The order of the steps is the lesson, not the steps.

---

## THE ONE RULE THAT MATTERS MOST

**Never force-kill this app.** No `Stop-Process -Force`, no End Task, no closing
the terminal while it runs. Cutting a streaming USB camera off mid-transaction
hangs it inside its own firmware: the next launch takes 40 seconds, reads zero
frames and logs ninety thousand read failures, and the only cure is physically
unplugging the camera. On the development machine this happened repeatedly and
was blamed on "a flaky camera" for two days. It was always the shutdown.

Close it with **Escape** or the **Exit** tile, or run `stop_screenplink.ps1`,
which posts a close message and waits. If you have already force-killed it, say
so and ask the user to unplug and replug the camera — do not spend an hour
diagnosing a wedge you caused.

**And never run two things on one camera.** The app and the camera doctor must
not be open at the same time. On Windows, two processes on one camera can leave
the driver in a bad state.

---

## STEP 1 — WHICH CAMERA, AND IS IT FIT TO SHOOT WITH?

Do this with the app **closed**.

```
Doctor.cmd
```

`Doctor.cmd` runs `screenplink.exe --doctor --all`. ScreenPlink is ONE
executable that takes its mode as an argument, and it has no console window of
its own (a console would sit lit on the monitor the camera watches), so
`Doctor.cmd` redirects the doctor's output into **`doctor-report.txt` beside it**
and then prints that file. **Read `doctor-report.txt`** — it is the report, and
it stays there after the window is gone.

It finds every camera on the machine and tests each one in a deliberate order.
Read the table it prints, and then **look at the PNG it saves for each camera.**
That last step is not a formality: it answers "is the camera even pointed at the
monitor" in one second, where every indirect measurement took an hour and gave
the wrong answer.

The order of its checks, and why:

| # | question | why it comes here |
|---|---|---|
| 1 | does it open, does it deliver frames | nothing else means anything otherwise |
| 2 | what frame rate, with the exposure FIXED | see the warning below |
| 3 | **does changing the exposure actually change the picture?** | everything downstream depends on exposure, so this is asked before any of it |
| 4 | chroma noise floor | this is the bar a laser hit has to beat |
| 5 | can it see the monitor flash | geometry: is the screen in frame |
| 6 | **save a frame and look at it** | the step that has settled six faults here |

### What the answers mean

**A camera that opens and delivers a picture of nothing.** Virtual cameras — OBS,
"EOS Webcam Utility", Snap Camera, manufacturer bridges — open perfectly, report
a sensible resolution, deliver frames at a sensible rate, and every frame is a
logo or a disconnected-cable icon. One of them shows a **saturated red X**, which
is precisely what this app's detector hunts. The doctor flags devices it believes
are virtual. Trust the saved frame over the device's name.

**Do not reason from camera names or indices.** On the development machine the
same physical camera appeared at index 1, then index 2, named "USB Video Device",
then "HD Pro Webcam C920", then "USB Video Device" again — within one morning.
Two different libraries disagree about which name belongs to which index, so a
name printed beside an index can belong to another device entirely. **Identify a
camera by what it does:** save a frame, look at it, measure its frame rate.

**Frame rate is not cosmetic, and it is the one number that might sink this.** A
laser trainer pulse lasts about 100 ms. The app confirms a shot by seeing it in
more than one camera frame, so at 60 fps a pulse lands in about six frames and at
30 fps in about three. Everything here was measured at 60. If this machine's
camera gives 30 fps, tell the user early — it should still work, with less
margin, and it is the first thing to suspect if shots go missing.

Also: **measure frame rate with the exposure fixed.** On auto-exposure in a dim
room the camera lengthens its integration until the frame rate falls to meet it,
so the number tracks the room rather than the device. This tool reported 28.3,
28.7 and 51.2 fps for the same camera within minutes before that was understood.

**If exposure does not change the picture** — check 3 reading under about 25%,
where a healthy camera moves 80% or more — the camera is accepting exposure
writes and ignoring them. On the Razer Kiyo Pro this was a "Low Light
Compensation" checkbox, not reachable through any normal camera property,
defaulting on, and silently returning after every replug. Other cameras have
their own equivalent. Look in the manufacturer's control panel and in Windows
camera settings for anything automatic: compensation, HDR, auto-exposure
priority, face tracking.

**If it cannot see the monitor flash, there are THREE causes and only one of
them is the camera.** All three give a flash difference near zero and they need
opposite responses, so do not stop at the first plausible one.

1. **The display is asleep.** A sleeping monitor and a blind camera read
   identically. The doctor tells them apart by asking whether the "white" frame
   ever actually got bright, and says which case it is. On the development
   machine three launches died this way while the owner was out of the room,
   and the app concluded the camera was broken.
2. **The camera is watching a DIFFERENT MONITOR from the one being flashed.**
   The flash goes to the *primary* display unless told otherwise. On a
   multi-monitor machine — which is most machines — that is simply the wrong
   screen, and nothing about the camera is wrong at all. Pass `--screen N` to
   flash the monitor the camera is actually pointed at, and make this your
   first suspicion if the user has more than one display.
3. **The camera genuinely cannot see the screen** — lens cap, a blind, pointed
   at a wall, or a virtual device.

Ask the user how many monitors they have and which one the camera faces,
BEFORE concluding anything about their hardware. If the app later reports that
it cannot find the screen, the same three causes apply in the same order.

**If a USB device is present but does nothing, read its problem code before
theorising about drivers.** `Get-PnpDeviceProperty -InstanceId <id> -KeyName
DEVPKEY_Device_ProblemCode`. Code 14 means it is waiting on a reboot to finish a
previous operation, and no amount of driver work will help. Code 43 on a root hub
means that whole branch has failed and every device behind it is gone.

---

## STEP 2 — WHERE TO PUT THE CAMERA

The camera does **not** need to be in front of the screen, or square to it.
Off-axis at an angle is normal and fully supported: the app photographs the
monitor to work out the mapping, so a screen that images as a trapezoid is fine.

What it does need:

- **The whole screen in frame, with margin.** If the panel is clipped at the edge
  of the camera image, the corners cannot be measured, and a mapping fitted
  without them is wrong in exactly the region the user will complain about.
- **Nothing bright in frame, if it can be avoided.** On the development machine a
  blown-out window beside the monitor supplied 1040 of 1356 detections — 77% of
  everything — before the screen region was masked out. A lamp, a window, a lit
  doorway: move the camera or close the blind, because no threshold fixes it.
- **No reflection of a bright thing on the panel itself.** A specular highlight on
  the screen defeats the flash-based screen search outright: the subtraction it
  depends on cannot see through it.

**Use the frame to aim it, not the user's description of where it points.** Ask
the camera for a frame, save it, and look:

```
Doctor.cmd --index N --frame aim.png
```

(Any arguments you give `Doctor.cmd` replace the default `--all`; the output
goes to `doctor-report.txt` again.)

Then tell the user what to change in physical terms — "tilt it down about ten
degrees, the bottom third of the monitor is out of shot" — and look again. That
is the loop. It takes about three iterations and it is the highest-value thing
you can do for them.

---

## STEP 3 — LAUNCH IT, AND LET IT MEASURE

```
screenplink.exe --target --introspect
```

(`Play.cmd` starts the app too, but with `--target` only. You want
`--introspect`, so run the line above.)

`--introspect` is what turns on the server your MCP tools talk to. **It is off by
default and the app behaves identically without it** — if you forget it, your
tools will report that nothing is listening.

The app measures its own setup on every launch, in about thirty seconds on known
hardware. Expect longer the first time here: nothing has been measured on this
machine, so it will run a full exposure sweep instead of checking a remembered
value.

**Leave it alone while it does this.** It flashes the monitor white and black to
find the screen, draws small markers to measure the camera-to-screen mapping, and
fires emulated shots at itself to test the whole chain. Tell the user not to fire
during it. The app says `HOLD FIRE` on screen while a measurement runs, and a
real shot during a noise measurement makes that shot the noise floor — which puts
the bar above every real hit for the rest of the session. The user's own words for
this, before it was understood: *"It worked for the first few shots and then it
stopped working."* Those first few shots were why it stopped.

---

## STEP 4 — READ WHAT IT MEASURED

Ask for the state. These are the numbers that decide whether this will work:

| what to read | healthy | what it means otherwise |
|---|---|---|
| frame rate | 45–60 | see the frame-rate warning above |
| noise floor | 0–5 | something bright is in frame, or on the screen |
| the detection bar | 8–15 | derived from the floor, so a high floor is the cause |
| mapping error, **held-out** | under 2 mm | the mapping is wrong; re-measure with `G` |
| self test | 12 of 12 | see below; a middling rate is often not a fault |
| signal margin | 3x or more | under 3x, the far side of the screen loses shots |

**Read held-out error, never fit error.** A fit flatters itself on the points it
was fitted to. One calibration on the development machine reported a small error
while being wrong by 1287 pixels on a 3840-pixel screen, and it was saved and
announced as a success. Held-out error is what the shooter actually feels.

**A middling self-test rate is not necessarily a problem.** The app tests itself
by drawing its own red dots and trying to detect them. A screen *emits* its dot;
a real laser *reflects* off the panel's matte coating, and those are different
physics. Five dots drawn at identical brightness measured peaks of 53, 17, 27,
137 and 13 — a ninefold spread — while real laser shots on the same setup read 86
to 103. So a low rate means a plumbing fault and should stop you; a middling rate
should defer to the other measurements.

### When the margin is thin, DO NOT TELL THEM TO DIM THE ROOM

This is the most-repeated mistake in this project's history. It was told four
times to a man sitting in a room already darkened as far as he could take it and
still see his gun sights. He said so twice.

**A dark room is the best condition this app can have.** Darkness is what permits
a longer exposure, and the laser's brightness scales with exposure time. Read the
measurement before giving any advice:

- **Noise floor high, over about 25:** something bright that is not the laser is in
  frame. *That* is when the room, a blind, or the desktop behind the app is worth
  changing.
- **Noise floor low and the signal still weak:** the dot itself is faint. The cures
  are a **longer exposure** (`X` re-measures it), or **more dot pixels** — move the
  camera closer, zoom in, or shorten the shooting distance. Not the room.

---

## STEP 5 — THEIR FIRST SHOTS

Have them fire a few deliberate shots at the middle of the screen, then a few at
the far side. Ask for the last few detections and read what became of each one.
Every trigger pull is accounted for, and the app distinguishes "nothing was seen
at any level" from "something was seen and was too faint" from "it was seen off
the screen" — three different cures that used to share one sentence.

The app also listens on a microphone, purely as an independent count of trigger
pulls. It never creates a shot: it has no position, and somewhere is the one
thing it cannot know. But "heard 15, saw 12" is the real detection rate, and the
only honest way to measure one. It stays silent until it has calibrated itself
from three shots the camera also saw, so the first few pulls do not count.

**If a hole appears where the shot did not land**, that is the mapping, not the
detector. Check the held-out error and re-measure with `G` if it is poor.

**If holes appear in the right places but some shots leave none**, that is
signal. Read the floor, the bar, and what the rejected detections measured.

**If the app seems to fire shots nobody fired**, it is almost always something on
the screen the camera watches. The detector looks for red — specifically red in
excess of both green and blue — so anything coloured that appears or moves on that
monitor can read as a hit. On the development machine the culprits were, in order:
the app's own operator console, its own signal meter, its own camera preview pane,
and a dictation app's always-on-top status panel. Ask what is painted and what red
excess it carries, and ask the user to move other windows off that monitor.

---

## WHAT YOU MUST NOT DO

- **Never write a camera control through the instrument.** The tools are read-only
  for the camera by construction, deliberately: a single exposure write has hung
  this camera family for 272 seconds. If you want a different exposure, press the
  key that makes the app measure one.
- **Never guess a threshold.** Every threshold here is measured. A guessed one is
  how this project got phantom shots — a guess of 60 against a real noise floor of
  110. If you want to move a bar, measure the floor first.
- **Never change a detector constant to make a symptom go away.** Raising one bar
  by 22 units, on sound reasoning, moved what the exposure search chose and cost
  the owner half a shooting string. Constants here are load-bearing in more places
  than they look.
- **Never conclude from one good run.** The chain is thirteen stages in series. At
  95% each that is a 51% chance of an end-to-end success, which is why "it worked
  once" has been wrong here more often than right.
- **Do not read a measurement taken after the symptom passed as exoneration.** Ask
  when it was sampled relative to the fault. A reading from the recovered state
  looks exactly like a clean bill of health, and was once used to overrule a
  correct hypothesis from the person with their hands on the hardware.

---

## THE KEYS, FOR WHEN YOU NEED TO DRIVE IT

You can send a key without stealing focus, which matters because the app's window
has to stay in front of the monitor the camera watches.

```
1 Static Targets    2 Escalating Ball   3 Bouncing Balls
4 Reaction          5 Clear The Board   6 Pop-Up Target
7 Static Target     8 Moving Target     9 Animated Target
M menu       O one player     T two players     C self check
E erase holes        G re-measure the mapping   X re-measure the exposure
S stats      L signal meter   0 show/hide the information layers
Esc exit     F11 close the target window
```

**Static Targets (1) is the best screen to measure against** — thin rings on
black, with almost nothing lit. The menu is the worst. Measured side by side on
the development machine: the menu gave 8 of 12 on the self test at a noise floor
of 4, and Static Targets gave 12 of 12 at a floor of 1.

---

## IF IT WILL NOT WORK AT ALL

Say so plainly, name the measurement that says so, and name the physical thing
that has to change. An app that blocks a working setup wastes exactly as much of
the user's time as one that passes a broken one — and the honest failure here is
always a number, not a feeling. The log is in `logs\`, written afresh on every
launch, and it is the account of what actually happened.
