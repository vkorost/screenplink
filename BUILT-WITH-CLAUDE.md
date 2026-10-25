# Built with Claude Code

ScreenPlink was written by [Claude Code](https://claude.com/claude-code) over
about two weeks. I directed it, shot at it, and decided what was acceptable. I
did not write the implementation.

This page is the index to how that worked, and to the documents worth reading if
you are doing something similar.

| | |
|---|---|
| 🧭 **[Lessons learned](LESSONS-LEARNED.md)** | Twenty findings, each with the measurement that produced it. The most useful file here, and mostly not about lasers |
| 🤖 **[Asking the app what it sees](MCP.md)** | The in-process MCP server: eleven tools, what they return, and why they return measurements rather than the app's opinion |
| 🔧 **[Troubleshooting](TROUBLESHOOTING.md)** | What the app measures about itself, field by field. This is also the file an agent reads when something is wrong |
| 🗺️ **[Project layout](PROJECT-LAYOUT.md)** | Every folder, and which test suites cover it |
| 📖 **[User guide](manual/README.md)** | Thirteen chapters, written for the person holding the pistol rather than the one reading the code |

---

## The actual problem

The game logic was never the hard part. The hard part was getting an agent to
**develop and debug a program that measures the physical world**, when the agent
cannot see the monitor, cannot pull the trigger, and cannot tell whether the dot
it is reasoning about was ever on the screen.

Three things made that tractable. None of them is about prompting.

## 1. Give the agent an instrument, not a log

The app runs an in-process server that answers what it is measuring **right
now**: the latest camera frame, the detector's arithmetic for every candidate
blob including the ones it rejected and the reason each was dropped, what is
painted in both screen and camera coordinates, and key and click injection that
does not steal focus from a fullscreen window.

**It returns measurements, never conclusions.** The app's own verdict is
available but labelled as one more datum to cross-check, because the app's
conclusions were repeatedly wrong: it once printed *"100% of measured shots clear
the bar, margin 6.1x"* on the same line as a measurement showing its own test dot
below its own threshold. An endpoint that returned the opinion would have been
wrong faster.

The single most valuable thing it exposes is **what was rejected**. Logs record
only the detections that survived, and a recording's re-encode destroys
near-threshold detail, so a rejected candidate exists nowhere except inside the
running process.

This took diagnosis from hours to minutes. Details and the full tool list are in
**[MCP.md](MCP.md)**.

### A worked example

A packaged build told its owner **"NOT READY, the camera is not delivering the
app's own test dot, unplug the camera"**. The camera was fine.

What the instrument gave, in order:

1. A mapping measured **that same launch** at 0.91 mm held-out error, with a sane
   noise floor and `ready_for_real_shots: true`. A mapping is produced by
   photographing eighteen markers on that screen, so the camera could not be
   blind four seconds later. Two statements from one program, one of them false.
2. A self check re-run through the instrument, without stealing focus from the
   fullscreen window: **12 of 12**, minutes later, same camera.
3. An exposure table that could not exist: three candidates measured within one
   second reading `-4: dot 109`, `-5: dot 5`, `-6: dot 108`. No response curve
   has that shape. That is not a dim dot, it is a photograph of something that
   had not been drawn yet.

The cause was **one artwork file missing from the release bundle**. The resulting
exception was raised inside the paint handler, so the signal that says "this
frame is on the glass" never fired, every wait-for-paint timed out, the self test
drew no dots, and a probe photographed a blank screen and blamed the hardware.
**1,722 tracebacks in a single launch, and none of it in the application log**,
because a log cannot describe a crash in its own paint path.

It had been read as an intermittent camera fault for days.

## 2. Make the tests able to fail, and prove it

The whole-launch test harness **exited after its first launch and reported
success, for as long as it had existed.** A figure quoted for days, "8 of 8 cold
starts ready", had never been measured at all.

The fix is mutation testing: reintroduce each known fault on purpose and require
the check written for it to go red. The first run put seven faults back and
**three survived**, and one of those revealed that the property being tested was
not the property that mattered.

The project now does this **326 times across six runners, and catches all 326**.
There are 30 suites, none of which needs a camera or a screen, and they ship with
both the repository and the release.

Checks on the checks followed, each of which caught something real: every suite
on disk must be invoked by the runner **and every invocation must be able to fail
the run**; every mutation anchor must still match its target exactly once; the
installed git hook must equal the one in the repository.

**An agent will happily report that a green suite means the code works. The only
defence is a suite that has been seen to fail for the right reason.**

## 3. Gate the word "ready"

An end-to-end startup here is thirteen stages in series. At 95% each that is a
51% chance of success, which is why every "it works now" claimed from a single
launch was wrong within the hour.

One calibration measured **0.4 mm** of held-out error and the next measured
**273 mm**, with no code change between them. Nothing counts as working until
five consecutive clean cold starts.

A related habit, for the same reason: **report held-out error, never fit error.**
One calibration here reported a small error while being wrong by 1287 pixels on a
3840 pixel screen, and it was saved and announced as a success.

## 4. Write the rules down where they execute

There is a rule registry in the working tree, 54 rules, each learned by breaking
it. **37 of them are enforced by a check that fails the build**, and the registry
itself is checked, so a rule cannot quietly disappear in a rewrite. That matters
because it happened: a rewrite once dropped six rules, three of which described
damage done the same day.

The registry is not published, because it is a working document full of
references only this project understands. The part that generalises is
**[LESSONS-LEARNED.md](LESSONS-LEARNED.md)**.

Prose alone does not survive. A note that was true when written and is false now
is worse than no note, and this project proved it twice: two notes claimed the
repository was local-only four days after a remote existed, and a third said two
simultaneous lasers had never been measured five days after they had.

## What it actually produced

Verified on hardware on an install that had never run, configured without a human
touching it:

```
camera chosen by measurement   60 fps, exposure spread 100%, flash difference 43
first launch                   READY, 12 of 12, held-out 0.92 mm,
                               3.9 mm position error, 52.8 fps, 0 control writes
then, with the gun             85 real shots at 4.2x to 12.1x margin
```

An independent microphone channel counts trigger pulls separately from the
camera, which is the only honest way to measure a detection rate here: **84 to
89% of heard trigger pulls become a detected shot.**

## Honest limits

- It was built and measured against **one** monitor and **one** webcam. Nobody
  knows what the second rig does, and that is the real limit of this project.
- Several faults were invisible to every test and only appeared on real hardware:
  sub-pixel text antialiasing writing coloured fringes the detector could see,
  and a target bright enough to drown the laser that hit it. An offscreen
  renderer does neither.
- The agent produced confident wrong diagnoses regularly. What caught them was
  never a better prompt. It was a measurement that contradicted the claim.

---

## Elsewhere

- Discussion of how this was built:
  [r/ClaudeAI](https://www.reddit.com/r/ClaudeAI/comments/1xeqtp1/i_built_a_laser_dryfire_shooting_game_with_claude/)
- 🎬 [Trailer, what it is](https://youtu.be/tg2Bt8K5IE0)
- 🔬 [How it actually works](https://youtu.be/pIdJhRYK-Bw)
