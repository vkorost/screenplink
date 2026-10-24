# What the App Does Before It Lets You Shoot

The half minute between launching and the menu is the app measuring itself. It
matters because the chain from a trigger pull to a bullet hole is a dozen stages
in series, and a weak link anywhere in it looks identical to every other weak
link from the firing line.

In order, it:

- **finds the screen** by flashing the monitor white and then black and
  subtracting one from the other;
- **measures the exposure** by drawing its own red dot and photographing it at
  several camera settings, then picking the one where the dot stands furthest
  clear of the background;
- **measures the noise floor** -- the brightest red thing in view when nothing is
  being fired -- and sets the detection bar from it, rather than from a guess;
- **confirms the camera-to-screen mapping**, by drawing two markers at known
  places and finding them. If they do not confirm, it measures the whole mapping
  again from eighteen markers;
- **fires a dozen test dots at itself** at the brightness of real shots, and
  counts how many came back.

Then it says what it found. A green verdict means it detected its own shots and
is ready. A problem names itself and names the one thing to do about it.

**This is the part to leave alone.** Every step photographs the screen, so a shot
fired into it is measured as part of the room.

Afterwards you can re-run any of it by hand, and it is worth knowing which key
does what:

| Key | What it does | When you want it |
|----|------------------------|--------------------------------------------|
| **C** | Self-calibrate | After moving the camera |
| **A** | Self-check | "Is it seeing my shots?" |
| **Q** | Show the last verdict again | It cleared before you read it |
| **G** | Calibrate by shooting a grid | The mapping is wrong and the markers will not fix it |
| **X** | Measure the exposure | The light in the room has changed |
| **F** | Measure the noise floor | Something bright was on screen during setup |

The first launch on a new machine takes longer than every later one, because it
has nothing remembered to check against.
