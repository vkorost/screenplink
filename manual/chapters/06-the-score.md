# The Score, and the Record Book

At the end of every magazine the app shows two figures: a **score**, which is you
measured against a perfect shooter on that same round, and an **efficiency**,
which is points per second. The whole screen, single player, is this:

```
ROUND 2

  17 shots
  16 hits
  1 miss
  70 of 75 points
  14.6 seconds

  EFFICIENCY  4.79
  = 70 points / 14.6 seconds

  SCORE  256
  AIM 93%   SPEED 27%
  BEST SCORE SO FAR  212
  BEST EFFICIENCY SO FAR  3.99

        [ NEXT ROUND ]
```

The tally at the top is measurement you can check yourself. `70 of 75 points` is
your points beside the most a perfect shooter could have taken from the targets
you were shown.

**The screen never advances by itself.** It waits for a shot at the **NEXT ROUND**
box, a click on it, or **N**. The box is armed 0.8 seconds after it appears, so a
shot still in flight when the magazine emptied cannot press it.

## The score: you against a ghost

A *ghost* shoots the same round you just shot. It hits every target in the best
zone available, and it fires as fast as the round allows -- no faster than the
app's own floor on shot cadence, and charged exactly the same unavoidable waiting
the drill imposed on you, because waiting for a target to appear is not something
either of you can shoot your way out of.

Both of you are then scored the way the round is already scored -- points over the
clock -- and the comparison is the score:

```
SCORE = 1000  x  aim  x  speed

aim    = your points  /  the ghost's points        how well you shot
speed  = the ghost's seconds  /  your seconds      how fast you shot
```

**Both factors are shown, because the single number cannot say which you are.**
Across the rounds this was measured on, the median aim figure was 82 per cent and
the median speed figure 21: careful shooting, taken slowly. One number would have
reported a middling score and left you guessing which half to work on.
`AIM 93%   SPEED 27%` tells you to speed up and leave your aim alone.

A score of about **1000 means you matched a perfect shooter at this machine's own
pace**. That is the intent of the scale, and it is reachable: on a drill with no
waiting, a shooter firing at the app's cadence floor and hitting every best zone
scores exactly 1000.

One rough edge, stated rather than left for you to find: on the drills where
*dropping* a target starts a wait -- a figure's fall, the soldier going down --
the ghost drops more targets than you do and therefore waits more than you did,
which can push a score past 1000. The highest seen here is 1146. It is a known
defect in the comparison, not a reward for anything you did, and it is being
fixed. Treat 1000 as "as good as this machine can be shot", not as a ceiling.

### The ghost's pace, and why it is frozen

The ghost fires one shot every **0.25 seconds**. That is not a guess at how fast a
trigger finger moves; it is measured, and it is a property of *this app*. Across
three thousand pairs of consecutive shots in real logs, the shortest gap the
detector has ever accepted between two shots at the same place on the screen is
0.254 seconds -- the length of the tracker's own refractory window.

That number is deliberately frozen. The score is directly proportional to it, so
if it were ever quietly re-tuned, every record ever set would change meaning
without anybody touching the record. Each stored score therefore carries the
cadence it was computed against, and a record set against a different one is
ignored rather than compared as though it still meant the same thing.

## The efficiency: points per second

**Points earned divided by seconds taken**, and the sum is printed underneath so
there is nothing to take on trust. The seconds are rounded to one decimal first
and the division is done on the rounded figure, so the line below always equals
the figure above it to the digit.

The clock runs from the instant the first round breaks to the instant the last one
does, from the shots' own timestamps. Time spent settling before the first shot
costs nothing; a slow string costs everything.

Nothing is averaged across rounds and no running mean is kept. One magazine, one
figure, on its own screen.

### Why there is no penalty for a miss

> *"When you pull up your gun and you have your mag and you need to do what you
> can with that mag, therefore you should be judged based on how you spend those
> 17 bullets, nothing else."*

The magazine is the penalty. A round that misses is spent and cannot be recovered,
so it has already cost its full value by earning nothing. Subtracting a further
ten points would charge twice, and the size of the second charge would be
invented. A shot that lands on a figure but outside every scoring ring is the same
case: it earns nothing, it spends its round, and that is the whole of its cost.

## What earns points

| Drill | Points for a hit |
|------------------------------------------|---------------------------------------------|
| Static Targets, and the figure drills | The ring it landed in: 10, 9, 8 or 7 |
| Escalating Ball, Bouncing Balls | The level the ball had reached |
| Clear The Board | 1 per circle |
| Reaction | 10 for a quarter of a second, falling to 1 past a second and a half |
| Animated Target | 10 for any hit; there are no rings printed on him to grade |

On the figures, the rings that score are the rings printed in the drawing, and a
hit inside the figure but outside all of them **scores nothing**. That is not an
edge case: between 42 and 51 per cent of the ink in each figure -- heads above the
top ring, limbs, the weapon -- lies outside every scoring ring. Such a shot leaves
a dim grey mark so that it is visibly different both from a miss and from the app
not having seen it. Note also that the artwork prints a 9 and a 7 on the head with
no ring behind either: a hit there is on the figure, not in a scoring zone.

## The record book

`BEST SCORE SO FAR` and `BEST EFFICIENCY SO FAR` are what this machine has seen.
When a round beats one, that line becomes `NEW BEST SCORE RECORDED` or `NEW BEST
EFFICIENCY RECORDED` in its place.

Records are kept **per drill, per magazine size and per player count** -- the
Moving Target with a magazine of seventeen in two-player mode is its own entry.
Drills hand the ghost wildly different amounts of dead time, so a single global
best would make most of them unbeatable for ever; and a best set on a magazine of
five cannot be beaten on a magazine of seventeen. Each entry stores the
ingredients as well as the result -- the points, the seconds, the shot count, the
time it was set and whether the comparison was fully measured or partly assumed --
so an old record can be re-derived rather than orphaned if the model changes.

Three things to know about it:

**It says "best so far", not "personal best", and the wording is exact.** There is
no shooter identity anywhere in the app -- nothing to log in to, nothing to
select, nothing stored about who fired. The player count in the key is a *count*,
not a name. Two people shooting the same machine share one record book.

**There are no lifetime totals and no round history.** Only the best per entry is
kept. Every round's full detail -- every shot, every reject, every figure --
goes to the log in `logs\` and nowhere else.

**It lives in `config\settings.local.json`, under `bests`.** Deleting that one
section resets every record and leaves your calibration alone, which is what you
want: the calibration is the part that took real shots to learn.

## A round is sound, or it is discarded

Sometimes the app sees shots nobody fired -- something coloured appearing on the
monitor it is watching will do it. A burst like that can fill a magazine in two
seconds and score higher than any real round, and once that is in the record book
it can never be legitimately beaten.

So a round the app cannot believe is thrown away whole, and says so:

```
ROUND 2

  17 shots
  16 hits
  1 miss
  70 of 75 points
  2.7 seconds

  EFFICIENCY  --

  ROUND DISCARDED
  17 shots in 2.7 s is faster
  than a pistol can fire.
  No score, nothing saved.
```

The tally stays, because that is measurement you can verify for yourself. The
judgement disappears: no score, no aim, no speed, no efficiency, nothing stored.
There used to be a middle state -- a score shown with a note saying it would not
be kept -- and it was rightly thrown out:

> *"How does it make sense to you to display the score and at the same time
> declare it invalid?"*

**The app never shows a figure it will not honour.** The converse holds too: a
score that appears on the screen is recorded if it beats the stored best. There is
no third answer.

**Only the rate of fire can discard a round** -- shots arriving faster than the
tracker is able to accept them, which no trigger finger can do. Dim shots and a
quiet microphone are measured and written to the log, and they never throw a round
away: a dim median is the normal state of a real session, and discarding a good
round for it would be worse than the fault it guards against.

## When the app marks its own figure

If the app's own measurement was poor while you were shooting, the efficiency is
starred and the reason is named:

```
  EFFICIENCY  3.10  *

  * measurement was poor this round: something
    covered the screen for 6s
```

The conditions are read from measurements the app already holds: something
covering the screen, the camera appearing to have moved, a thin signal margin, the
detection bar shifting mid-round, or the microphone hearing trigger pulls the
camera never saw.

This exists because of a rule the app follows everywhere: **it never coaches you on
its own blindness.** A figure depressed because the app could not see half your
shots is "you are pulling left" with a number attached, and you would train away a
flinch you do not have.

## What this is not

This is not USPSA or IPSC scoring and it is not hit factor. The efficiency figure
is the same shape of arithmetic answering a different question, by choice. Hit
factor subtracts penalties before dividing and assumes a power factor and a
rulebook edition; none of that is here, and nothing in this app should ever be
described as if it were.
