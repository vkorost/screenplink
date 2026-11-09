# When it does not work

The technical end: log lines, the camera doctor, the registry, a wedged camera,
and how to ask the running app what it is measuring.

**For the everyday cases (closing it properly, which key to press when something
changes, where your configuration lives and what deleting each file costs) read
[chapter 12 of the guide](manual/chapters/12-looking-after-it.md).** That chapter
also carries three diagnoses that are almost never what they look like, and it is
the only place they are written down on purpose, so that a second weaker copy
cannot drift away from it.

---

## Before anything else: read what the app already told you

The app writes `app\logs\camera_lab.log`, with a dated header per launch, and it
is the account of what it measured rather than what it guessed. Every launch ends
with one machine-readable line:

```
GATE role=live screen=3840x2160 mapping=confirmed verdict=ready detected=12/12
     held_out_mm=0.94 error_mm=4.59 fps=50.1 margin=4.56 floor=3 bar=9
     exposure=-5 rejected_short=0 writes=0 coupling=none findings=0
```

| Field | What it means, and what good looks like |
|---|---|
| `role` | `live` or `harness`. A harness line is a simulated camera. Never read one as a measurement of your bench |
| `mapping` | `confirmed` (the saved calibration was checked with two markers and kept), `measured` (it was remeasured) or absent |
| `verdict` | `ready`, `degraded` or `blocked` |
| `detected` | of the app's own test dots, how many it saw. 12/12 is healthy. **This is not your pistol's hit rate**. See the self-test caveat in the README |
| `held_out_mm` | calibration error measured on points it was *not* fitted to. Under about 1.5 mm is healthy; `-1.00` means no figure |
| `fps` | camera frames per second in the app. Under ~40 means a 100 ms pulse is landing in too few frames |
| `margin` | how far the app's test dots cleared the detection bar. Under 3 is thin |
| `floor` / `bar` | the measured noise floor, and the detection threshold derived from it |
| `writes` | camera control writes this launch. Should be single digits; this camera has hung for minutes on one |
| `coupling` | whether the microphone has learned to tell the app's own gunshot from yours. `none` usually means the app's sound is not actually audible. See below |
| `findings` | how many problems the self-check raised |

Every shot also writes a line carrying the evidence behind it: which frame of the
pulse the position came from, how bright that frame was against the brightest,
how far the dot drifted during the pulse, and the margin over the bar
(`best/bar 89/8 = 11.1x`). `app\tools\read_string.py` tabulates a whole string.

---

## No sound

**Check the Windows volume mixer first, and check it against the app with the app
running.** This is almost never the app.

Windows keeps a volume **per application, per output device**, identifies the
application **by its executable path**, and will happily set it to zero and
remember that forever. A virtual environment you build later is a *different
path*, so it gets its own fresh entry. Raising the slider for one Python does
not cover the next one you create. The mixer only lists a program while it is
running, so start the app first.

This cost three days here. The app opened the right device, loaded the right
sample, and logged `gunshot ... played on DELL U4320Q` while the slider sat at 0%.

To see it without the app running, read what Windows has stored:

```bat
app\.venv\Scripts\python.exe app\tools\audio_meter.py
```

It prints the live sessions on every output, then decodes the remembered ones
from the undocumented store Windows keeps them in:

```
HKCU\Software\Microsoft\Internet Explorer\LowRegistry\Audio\PolicyConfig\PropertyStore
```

Under each entry, `{219ED5A0-...}` holds `3` = master volume as a float, `4` =
per-channel volumes, `5` = mute. The decode is inferred, not documented, so the
tool prints the raw bytes beside every value. **It reads and never writes**;
changing the volume is yours to do in the mixer.

**Knock-on effect while the output is silent:** the microphone can never learn
what the app's own gunshot sounds like, so it falls back to going deaf for a
fixed 0.8 s after every shot instead of the measured 0.07–0.22 s. You will see
`coupling=none` on the GATE line and `DEAF` on shot lines. Fixing the slider
fixes both.

---

## The camera

```bat
app\scripts\Doctor.cmd
```

Runs with the app **closed**, because two processes on one camera is its own fault, and
writes `doctor-report.txt` beside itself, because a console window sitting lit on
the monitor the camera watches is not a neutral observer. It tests every camera
it finds, skips ones it recognises as virtual, and ranks them.

The six checks are in that order on purpose: each one is worthless if the one
before it failed.

| Check | Healthy | What a failure means |
|---|---|---|
| **1. opens and delivers** | opens in well under a second | nothing below means anything |
| **2. frame rate** | 60 fps (≈6 frames per pulse) | 30 fps halves your frames per shot. Measured with the exposure *fixed*, or the number tracks the room rather than the camera |
| **3. exposure actually changes the image** | ~100% spread | **the most important check.** A camera that accepts exposure writes and ignores them makes every number after it lie. Look for anything automatic in the vendor's control panel and Windows camera settings: low-light compensation, HDR, auto-exposure priority |
| **4. chroma noise floor** | p99 of 0–4 | this is the bar a hit must beat; a real hit measures 150–230 |
| **5. sees the monitor flash** | difference 40+ | under 18 means it is not seeing *this* screen change: aimed elsewhere, or at a different monitor. **A sleeping display and a blind camera read identically here and need opposite responses**, so the doctor checks whether the white frame ever got bright |
| **6. look at it** | two PNGs saved | open them. One glance answers "is it pointed at the monitor", which no number can |

A healthy report ends like this:

```
rank index  result       fps   exp%  floor   flash  camera
1    2      HEALTHY     60.4    100      0    44.0  USB Video Device
2    1      POOR        30.0      0      0     0.0  HD Pro Webcam C920
```

**Virtual cameras are skipped by name.** Software devices open fine and deliver
frames of nothing, sometimes containing a saturated red X, which is precisely
what the detector hunts.

### The names move, and the app now works around it

Windows does not guarantee that a camera keeps its name or its position across a
reboot. On the bench this app was built on, a restart moved the generic name
"USB Video Device" from one camera onto another, and the app, which remembers
its camera by name, opened the wrong one.

The app remembers the name because that normally survives a reshuffle better
than an index does. What matters is that a wrong guess recovers by itself:

- When the camera it opened cannot see the monitor flash, the app asks **Windows**
  whether the display is actually powered, which costs no camera traffic.
- If the screen is on, the black frame belongs to the camera, so the app moves to
  the next real camera and measures again. Each one is tried at most once per
  launch, so it cannot cycle.
- If the screen is off, dimmed, or Windows will not say, the app reports a
  sleeping display and leaves the camera alone. Cycling cameras cannot light a
  screen, and letting go of a working one to prove that has cost whole sessions.

So if the log says the display looks asleep while you are looking at a lit
screen, the thing to check is whether Windows has gone quiet about the power
state, not the camera. `Doctor.cmd` settles it in one run: it ranks every camera
and saves a picture from each, and the picture says which one is pointed at the
monitor.

**The names in that report are Windows' names, not the cameras'.** Trust the
measurements and the pictures over the label beside them.

### A wedged camera

Symptoms: the camera takes tens of seconds to open, or opens and delivers no
frames at all, with **nothing in the Windows event log**. That silence is the
diagnosis: Windows logs a device being removed, so if nothing was logged,
nothing was removed, and the hang is inside the camera's own firmware.

Almost always this follows a **force-kill**. Terminating the app runs no cleanup,
so the camera is cut off mid-stream with its driver mid-transaction. Use
`app\scripts\stop_lab.ps1`, or Escape in the app. Never End Task.

Recovery, in order of how little it disturbs:

```bat
app\.venv\Scripts\python.exe app\tools\camera_alive.py     :: read-only: is it alive?
```

1. Elevated shell: `pnputil /restart-device "<instance id>"`. Needs admin; it
   fails with "Access is denied" otherwise.
2. Device Manager → the camera → Disable device, then Enable device.
3. Unplug and replug, **preferably into a port on a different USB controller.**

### When Windows itself is unhappy

Read the device's **problem code before theorising about drivers**:

```powershell
Get-PnpDevice -Class Camera | Select-Object Status, FriendlyName, InstanceId
Get-PnpDeviceProperty -InstanceId '<id>' -KeyName DEVPKEY_Device_ProblemCode
```

| Code | Means | Do |
|---|---|---|
| **14** `CM_PROB_NEED_RESTART` | a previous operation is pending | **reboot.** `pnputil /restart-device` refuses in as many words |
| **43** `CM_PROB_FAILED_POST_START` | the device, or the hub it is on, has failed | if two cameras vanished together, suspect the **USB root hub** they share, not the cameras |
| **Status: Unknown** on several at once | present in the registry, absent from the bus | that is a bus-level fault, not a software one |

`Restart-Service FrameServer -Force` needs an elevated shell and fails with
"Cannot open FrameServer service" otherwise.

---

## Asking the running app what it sees

The app can answer for itself, live, instead of being guessed at from the
wreckage afterwards. It is **off by default**:

```bat
app\scripts\Play.cmd --introspect          :: or set LSA_INTROSPECT=1
curl -s http://127.0.0.1:8777/health
```

| Route | Answers |
|---|---|
| `/state` | effective parameters, the floor beside the bar it produced, the gain map, mapping error, what owns the screen, fps, control writes |
| `/settings` | effective / saved-on-disk / code default, side by side, which is how a setting silently beating a code default gets caught |
| `/score` | the detector's arithmetic as numbers, plus **every blob above threshold before any filter**, each with the reasons it would be dropped |
| `/shots` | recent detections with their per-frame evidence |
| `/drawn` | what is painted, in screen and camera coordinates |
| `/frame` | the latest camera frame and the detector's score map |

It exposes **measurements, never conclusions**. The app has printed "100% of his
shots would clear the bar" on the same line as a measurement showing its own test
dot *below* its own threshold, and an endpoint returning the app's opinion would
only have been wrong faster. It binds to loopback only and cannot write a camera
control.

---

## Shots are not leaving holes

Chapter 12 of the guide covers this from the firing line. From a terminal, in
this order:

1. **Read the GATE line.** `verdict`, `margin`, `floor`/`bar` and `detected`
   together say whether the app could see anything at all.
2. **Diff the configuration, both files.** `config\settings.local.json` *and*
   `config\calibration.json`. The sensitivity map lives in the second and is
   invisible in the first; a working setup and a broken one have differed by
   exactly that, with nothing in the settings to show for it.
3. **Read the per-shot lines.** A detection that was *seen and rejected* says so
   and says why; `best/bar` gives the margin it actually had.
4. **Look at what the camera saw.** The app saves `verdict_camera.png` and
   `verdict_screen.png` at every verdict, and `probe_lit.png` / `probe_dark.png`
   when the exposure probe gives up. If the dot is visible in `probe_lit.png`,
   the camera missed a frame and any advice about unplugging it is wrong.

---

## Shots on a MOVING target score, but only on its leading edge

The symptom is specific and worth recognising: clean hits on the side the
target is moving *away from* do not register, while hits in front of it do, and
it gets worse the faster the target goes.

That is a clock error, not a geometry one. A shot's timestamp is taken when the
camera frame reaches the program, and the display pipeline and the camera's own
pipeline both happened before that, to a target the shooter had already aimed
at. The hit test rewinds a moving target to where it was when the shot was
taken, and unless it also allows for those two delays the tested centre sits
ahead of the one on the glass.

- **`pipeline_lag_s` in `config\settings.local.json`** is that allowance, in
  seconds. It is measured on this machine from ordinary shooting and saved; it
  is not a preference to tune by hand, though setting it is harmless because
  it is clamped and re-learned.
- **The log reports every change**, with the sample it rested on:
  `shot timing: over N shots at moving targets the median landed +Xms behind
  the centre the hit test used, so the pipeline lag moves AAms -> BBms`.
- **The self test reports the independent figure** it measures with no shooter
  involved: `paint-to-detection latency over N dots: min ...`. Read the
  **minimum**; waiting for the next camera frame can only add to it.
- If the two disagree wildly, trust the shooting figure. The dot probe measures
  a different path.

A sanity check that costs nothing: a mapping error moves shots sideways as well
as along the direction of travel. If the cross-track error is centred on zero
and only the along-track error is biased, the mapping is fine and the clock is
not.

---

## It says the camera is not delivering its own test dot

The app paints a full-red dot and measures what the camera makes of it. On a
healthy launch that reads 25 to 97. Below the line it says so -- and since
2026-11-08 what it says depends on whether it has seen it **once or twice**.

- **Once** is DEGRADED: *"the app's own test dot read faint. Measure it
  again."* It names the evidence from the same launch that disagrees (frame
  rate, and the mapping error if there is a mapping), and asks for **X** to
  re-measure the exposure and **C** to re-run the check.
- **Twice in one session** is BLOCKED, with the original advice: stop, close
  the app by its window, unplug the camera and plug it back in.

The split exists because a single faint reading proved transient: a dot read 10
and the launch was blocked, and four minutes later the identical exposure read
111 with nothing changed but time. Note also that **this dot is not your
laser**: a screen emits it, while a laser reflects off the panel far more
brightly, so shots can register perfectly well while this dot reads faint. That
is the usual explanation when the app calls the camera dead and the holes keep
appearing.

Ambient light is the other common cause. Light falling on a matte panel lifts
all three colour channels, and the detector scores red minus the larger of
green and blue -- so a lit room collapses the dot's score without raising the
noise floor. Check `verdict_camera.png`: if the room is visible, that is the
answer.

---

## The app will not start

- **A second instance.** Two processes on one camera leaves the driver in a bad
  state. Close the first properly.
- **No virtual environment**, running from source. `Play.cmd` says so and prints
  the two commands that build one.
- **It starts and nothing appears.** The target window may have lost the
  foreground: Windows refuses to bring a window forward for a process that is
  not already in front. `app\tools\window_stack.py` names whatever is covering
  it. An always-on-top panel from another application is the usual culprit, and
  the app logs `the app's own window ... is in front of the target screen` when
  the window in the way is one of its own.

## The tests fail

```bat
app\scripts\run_tests.bat          :: everything, no camera needed
app\scripts\run_affected.ps1       :: only what your change touches
```

Read the log rather than the exit code, suite by suite. Two things that look like
failures and are not: a **mutation runner** reporting a fault as caught is working
correctly: it deliberately breaks the code and expects a specific check to go
red, and a **timing check** failing while something else heavy runs on the
machine is contention, not a regression. Re-run it quiet rather than moving the
bar.
