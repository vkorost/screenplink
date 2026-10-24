# Lessons learned

Building ScreenPlink took about two weeks and produced a long list of mistakes
worth recording. Every item below was paid for: something broke, it was measured,
and the finding changed the code. Most of them are not about lasers or cameras at
all, which is why they might be useful to someone building something else.

The internal rule registry that enforces these is not published, because it is a
working document full of references only this project understands. This file is
the part that generalises.

---

## 1. Measure the quantity you actually consume, not one that correlates with it

This project made the same mistake five separate times, in five different
disguises, and it was expensive every time.

The exposure was once chosen by **mean frame brightness**, as a proxy for "is the
background dark enough". That holds in a lit room. Darken the room and every
candidate's mean collapses toward zero, the rule has nothing left to discriminate
on, and it settles on the darkest setting available, which is where the sensor
stops behaving monotonically. **So the best possible condition, a dark room,
is what broke detection.** The fix was to measure the chroma score the detector
actually thresholds, at each candidate, and choose on that.

Three more things were tried as a stand-in for "how brightly does a laser image
here", and all three measure something adjacent:

| Proxy | What it reads | Truth |
|---|---|---|
| White calibration markers | 227-245 across the panel, an **8%** spread | a saturated square measures panel illumination, not point spread |
| A synthetic red dot drawn on screen | essentially flat | a screen **emits**; a laser **reflects** off matte coating |
| The mapping's local Jacobian | predicts the far side within 0.14 | **backwards**: it models geometry, and the falloff is radiometry |

Ground truth from real shots: peak 76 near, 47 far, and blob area 396 px against
149 px. A two-fold variation that none of the three proxies could see.

**A proxy that works in the conditions you tested it in is worse than no rule,
because it fails silently and confidently.**

## 2. A check that cannot fail is not a check

The whole-launch test harness **exited after its first launch and returned
success, for as long as it had existed.** The application ends with `os._exit`
deliberately, because a thread blocked in a camera driver cannot be joined, and
the harness built the application inside its own process. So launch one killed
the test runner. With status 0.

A claim of "8 of 8 cold starts ready" was quoted from it. That claim had never
been measured.

The cure is mutation testing: reintroduce each fault on purpose and require the
check written for it to go red. The first run of that exercise reintroduced seven
faults and **three survived**. One of them revealed that the property being
tested was not the property that mattered.

Checks on the checks followed, and each caught something real:

- every suite on disk must be invoked by the runner, **and every invocation must
  be able to fail the run** (two runner paths had a literal tab where a backslash
  belonged, and had been decoration since the day they were added);
- every mutation anchor must still match its target exactly once (one was pinned
  to code text that a line-ending change silently disarmed);
- the installed git hook must equal the one in the repository.

**Silence is not success.** A step that quietly does nothing is worse than one
that fails, because it looks like coverage.

## 3. Repeatability is the property. One good run proves nothing

An end-to-end startup here is thirteen stages in series. At 95% each that is a
51% chance of success. Every "it works now" claim made from a single launch was
wrong within the hour.

One calibration measured **0.4 mm** of held-out error, and the next, with no code
change between them, measured **273 mm**. The first was luck. Only after running
the same thing five times in a row did the real faults name themselves, and each
one was visible in the numbers rather than in the code.

**Do not use the word "ready" from one good run.** This project adopted a gate of
five consecutive clean cold starts before the word was allowed.

## 4. Report held-out error, never fit error

A model with enough freedom flatters itself on the points it was fitted to. One
calibration here reported a small error while being wrong by **1287 pixels on a
3840 pixel screen**, and it was saved and announced as a success.

Leave some measurements out of the fit and report the error on those. It is the
only number that reflects what the user will feel.

## 5. Never kill a process that holds a device

Force-killing the application left the camera wedged inside its own firmware:

```
after a force-kill       opened in 40.3 s,  0 frames, 90,816 read failures
after a clean shutdown   opened in  0.44 s, 429 frames, 0 failures
```

Windows logged nothing at all during those stalls, which is itself the
diagnostic: the operating system logs a device being removed, so if nothing was
logged, nothing was removed.

Worse, it was self-inflicted and self-sustaining. The camera stalls, the capture
thread blocks in a driver read, that live thread keeps the process alive, the
window closes but nothing exits, the stop script forces a kill, the camera is
wedged, and the next launch stalls. **Every link in that loop had already been
fixed once in isolation.**

Two days were spent blaming "a flaky camera". It was always the shutdown.

## 6. Identify hardware by what it does, never by its name or index

In a single morning one physical camera appeared at index 1, then index 2, named
"USB Video Device", then "HD Pro Webcam C920", then "USB Video Device" again. Two
enumeration libraries disagreed about which name belonged to which index, so a
name printed beside an index could belong to a different device entirely.

Virtual cameras make it worse: they open successfully, report a sensible
resolution, deliver frames at a sensible rate, and every frame is a logo. One of
them contains a saturated red X, which is precisely what this application's
detector hunts for.

**Open it, save a frame, look at it, and measure what it can do.** That one step
settled six faults here, and it was repeatedly the last thing anyone tried.

## 7. Before theorising about drivers, read what the operating system already recorded

A device that was present and did nothing turned out to be sitting at problem
code 14, `CM_PROB_NEED_RESTART`, left over from a replug. No amount of driver
work would have helped; it needed a reboot, and the tooling said so in as many
words when asked.

Separately, two cameras vanishing simultaneously was one fault and not two: the
USB root hub they shared had failed with code 43.

## 8. A dead output channel looks exactly like success

Three separate feedback channels here failed silently, each for a different
reason, and each looked fine from the inside:

- The audio library kept accepting writes to a stream whose endpoint had gone
  away. Nothing raised; the sound simply went nowhere.
- A microphone threshold guessed rather than measured produced six "shot heard"
  reports in thirty seconds with nobody shooting.
- Gunshots played correctly, at the right device, **into a per-application volume
  slider set to zero**, for three days. The operating system stores that value
  per executable path and remembers it while the program is not running, so a
  virtual environment rebuilt later gets its own fresh entry.

The last one is the general lesson: **before theorising about an application's
audio, read what the operating system already stores about that application.**

And whenever a device is chosen on the user's behalf, **name it in the log**. A
silently wrong choice is indistinguishable from success.

## 9. Do not let a program draw anything that looks like the signal it detects

The detector scores red in excess of both green and blue. Anything red on the
monitor it watches can therefore read as a hit. In order, the culprits were: the
application's own operator console, its own signal meter, its own camera preview
pane, and a dictation tool's always-on-top status panel.

Two refinements, both learned the hard way:

- **A limit for static text is not a limit for anything that moves or appears.**
  A thin line of text and a large panel at the same colour are not the same risk,
  especially when the measurement is a maximum over a region.
- **Sub-pixel text antialiasing writes coloured fringes on glyph edges.** Grey
  text was arriving at the camera at a red excess of 86, above the median of a
  real laser hit, while every source-level colour check read zero. Rendering
  tests only catch what they render, and a palette scan cannot see what a
  rasteriser adds.

There is a second, less obvious version of this. The detector reads red minus the
strongest of green and blue, so **a bright grey target drowns the dot that hits
it**: red saturates while green and blue stay high. Making targets brighter made
them harder to hit. Everything the application draws is now capped at the
brightness of the darkest supplied artwork.

## 10. A saved setting silently beats a code default

A constant was changed in code to prevent a known failure, the tests passed, and
the application kept running with the old value, because a saved settings file
held it and the file wins. The log even reported the old value and it was nearly
missed.

**When changing a default that exists to prevent a failure, change the saved
state too and put the bad value out of reach.**

The inverse also happened: a program wrote its own measured value back to its
settings on exit, then read it at the next launch, drifting a little further each
time. **A cache with no invalidation is a constant.**

## 11. Asynchronous routines need a scheduler, not flags

About fifteen linear hardware procedures (set a control, draw, wait, photograph,
restore) were hand-converted into timer callback chains, with each procedure's
local state hoisted onto one large object as flags. The result, measured:

| | |
|---|---|
| timer hand-offs in one file | 74 |
| re-entries into the event loop mid-step | 15 |
| reads of an object's own attribute through a default | 35 |
| one flag that meant four different things | written in 12 places, read in 24 |
| **the central mutual-exclusion list** | **consulted by 1 of the 6 routines that needed it** |

That last line is the whole argument. The exclusion list had been added *to end*
a class of bug, and was then read by one routine. Two procedures ran at once,
each photograph was taken for a step other than the one on screen, and the
resulting message was "the camera cannot see the screen" twelve times in a row.
It cost two sessions, and the guard against it is three lines.

Replacing the flags with **one run slot**, where a routine is a generator that
yields what it is waiting for, made the properties hold by construction rather
than by everyone remembering: two routines cannot interleave, cleanup cannot be
skipped, shutdown stands everything down, and no wait is unbounded.

## 12. A cleanup that works because of what happens to run next is not a cleanup

A routine left a full-screen black cover in place. It never showed, because the
step that always followed happened to tear the cover down as a side effect. The
day that step started being skipped, the screen went black for the whole session
and the application measured its own cover and reported a 145 mm error, four
lines after measuring 2 mm.

Put cleanup in a `finally`, in the routine that owns it, and never rely on a
successor.

## 13. Fix the rule, not the instance

Nearly every expensive bug here was one mistake made in several places. A
calibration profile was documented as deliberately leaving a camera control
alone; that was correct while its only caller ran before anything had changed the
control, and quietly wrong for the caller added later. The note existed. The code
did it anyway.

**When a fault is found, fix the rule and then go and find every other violation
of it.** The failure is not ignorance of the rule, it is fixing the instance in
front of you and leaving the same mistake standing in four other places.

## 14. Expose measurements, never conclusions

An in-process server that answers "what are you measuring right now" changed how
long diagnosis took here, from hours to about ninety seconds.

The design rule that makes it useful: **it returns numbers, not verdicts.** The
application once printed *"100% of measured shots clear the bar, margin 6.1x"* on
the same line as a measurement showing its own test dot below its own threshold.
An endpoint that returned the application's opinion would only have been wrong
faster.

The most valuable thing it exposes is **what was rejected**: logs record only the
detections that survived, and a recording's re-encode destroys near-threshold
detail, so a rejected candidate exists nowhere except in the running process.

## 15. Read the log before naming a cause, and read stderr too

A change was blamed for a failure that had begun thirty-nine seconds earlier. The
correct hypothesis came from whoever had their hands on the hardware and was overruled
by a measurement sampled **after the fault had passed**, which looks exactly like
a clean bill of health.

And on the last day of this list: an application-level log cannot describe a
crash in the application's own paint path. One missing artwork file raised an
exception inside the paint handler, so the signal that says "this frame is on the
glass" never fired. Every wait-for-paint timed out, a self test drew no dots, a
probe photographed a blank screen, and the application advised unplugging a
camera that had measured a sub-millimetre mapping four seconds earlier. The log
looked merely disappointing. **Standard error carried the traceback 1,722 times
in a single launch.**

## 16. An impossible measurement is a clue, not noise

The same fault produced this table, three exposure candidates measured within one
second:

```
-4: dot 109      -5: dot 5      -6: dot 108
```

No response curve has that shape. A dot cannot be invisible at one setting and
bright at the settings either side. That is not a dim dot, it is a photograph of
something that had not been drawn yet, and it pointed straight at the cause.

## 17. A check pinned to the state of the world fails when nothing is wrong

Adding artwork to this project once broke seven checks at once: a count of files,
a per-filename table, a statistical band around one divided by the number of
items, a hard-coded list of names. None of them was testing behaviour; all of
them were testing what happened to be on disk that week.

A check that goes red when nothing is wrong teaches people to ignore checks.
Derive from the catalogue, measure the property, and pin nothing to a filename.

## 18. Test across the seam, not up to it

A producer wrote a field and its test confirmed the field was written. A consumer
never read it. Both sides were green and the feature did not work.

Worse: a field was added to a machine-readable status line, and the parser that
consumed it in another language pinned the old prefix. The next automated run
would have scored a perfectly healthy application as a total failure. The
producer had a test. The consumer had a test. **Nothing ran one against the
other.**

## 19. A folder-level check cannot see a missing file

A packaging rule shipped only one named file out of an asset folder, on the
reasoning that the others were staging material referenced nowhere. That was true
when written. A file the application draws on **every repaint of every screen**
moved into that folder afterwards, and the rule silently left it out.

The release gate checked that each asset folder was present, and reported success
while the folder was missing the one file that mattered. It now compares every
file in the repository against every file in the bundle.

**Check the thing, not the container that holds it.**

## 20. A note that was true when written and is false now is worse than no note

Two notes in this project's own documentation stated that the repository was
local-only and that nothing should ever be pushed. A remote had existed for four
days. They were caught only because another reader went through the file.

On the last day of this list, a note reading *"never measured: two lasers at
once"* was taken as current and used to weaken a documentation claim. The
measurement had been made five days earlier, and the evidence was sitting in the
logs: fourteen pairs of shots landing in the same camera frame, both well above
threshold, on opposite halves of the screen.

Date every claim about state, and strike the line out when it stops being true.
Prose cannot be grepped, and nobody reliably re-reads their own prose.

---

## The shortest version

1. Measure what you consume, not something that correlates with it.
2. Prove every check can fail, then prove it again after you change the code.
3. One good run is not a result. Repeat it.
4. Report error on data the model never saw.
5. Never force-kill something holding a device.
6. Identify hardware by behaviour, and look at a frame with your own eyes.
7. Name the device, the file, and the setting you chose on the user's behalf.
8. Anything that draws on a screen you are measuring is a suspect.
9. Put cleanup in a `finally`, never in whatever runs next.
10. Expose measurements. A conclusion is only ever as good as the number under it.
