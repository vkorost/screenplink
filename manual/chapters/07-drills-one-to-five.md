# The Drills: One to Five

## 1 - Static Targets

![A big target in the middle, four small ones in the corners, and the rings that score.](media/mode1.jpg){width=6.9in}

A large ringed target in the middle and four small aim marks in the corners.
Nothing moves, nothing escalates and nothing disappears: you shoot, holes appear,
that is the drill. Each ring is worth one more than the ring outside it: the
big target scores 7 to 10, and the smaller corner marks 7 to 9.

This is what the app lands on while it is measuring, and what key 1 returns to.
It replaced a blank "free aim" screen, which was removed for the best of reasons:
*"Is free aim just a blank screen without anything? It is useless to me."*

A hit flashes a brief ring on the mark it struck, and each mark keeps a small
tally of the hits it has taken, so the shape of a string can be read off the
screen without turning on any readouts.

## 2 - Escalating Ball

![One ball, and the level it has reached. Each side escalates on its own.](media/mode2.jpg){width=6.9in}

One ball bounces around. Every time you hit it, it comes back five per cent
faster, up to level five and no further. Hitting a level-four ball is worth four
points, so the score is a record of how far you got.

Three decisions worth knowing:

- **The cap is real.** It was uncapped at ten per cent a hit, and the ball
  outran the detection hardware: past about two screen widths a second a ball
  moves further than its own diameter during the hundred milliseconds the laser
  is lit, and where it was when the trigger broke stops being a meaningful
  question.
- **The ball never shrinks** in this drill. A smaller ball was reported
  unhittable, and a target smaller than the laser's own dot is a test of the
  detector rather than of marksmanship.
- **A new ball never appears on top of the muzzle.** It comes back at the far
  side of the playfield from your last hit, so nobody wins a level by standing
  still.

Missing costs nothing but time. Punishing a miss teaches flinching, and the clock
punishes it already.

## 3 - Bouncing Balls

![Several balls, each carrying its own level.](media/mode3.jpg){width=6.9in}

Several balls at once, and **each one carries its own level**, printed on it.
Hitting the easy one does not punish you on the hard one, so after a few minutes
the screen holds a spread of difficulties at the same time: the ball you keep
hitting becomes the one you can barely hit, while the others wait. That is what
makes it a drill rather than a single escalating target.

Five levels again, each five per cent faster and five per cent smaller than the
last. The shrink is half what it was originally, because the complaint was
precisely about small targets; and there is a floor on the radius, set from the
measured size of the laser dot on real hardware.

Both ball drills draw a small target face on the ball rather than leaving it a
plain disc. The face is drawn at a little over half the brightness ceiling and in
thin strokes, because every unit of brightness under the dot comes off that
shot's score: an earlier solid fill cost about fifty-five units and took a
typical shot from 85 down to 30. The target was hiding the shots that hit it.

## 4 - Reaction

![One target, revealed without warning. In two-player mode both sides see it at the same instant.](media/mode4.jpg){width=6.9in}

A blank screen, a wait of unpredictable length, and then one target. The clock
starts the instant it is drawn and stops when you hit it.

- **The delay is random**, between 1.2 and 3.5 seconds. A fixed delay is trained
  against within a few repetitions, and then it measures your counting rather
  than your reaction.
- **Firing before the target appears is a false start.** It is reported and never
  scored, even if the target arrives a moment later. A shot that left early was
  a shot at nothing, and counting it would reward the one habit the drill exists
  to break.
- **The target moves** at least a third of the screen between repetitions, so it
  cannot be pre-aimed during the wait.

It reports your best and your **median** time. The median is the honest one: a
single lucky repetition is not a skill.

## 5 - Clear The Board

![Twelve circles, one clock.](media/mode5.jpg){width=6.9in}

Twelve targets, one clock, hit them all. It reports the time and the number of
rounds it took.

Targets never overlap and never touch an edge, because an ambiguous hit between
two circles or a target half off the screen makes the result arguable in exactly
the wrong direction. A hit is scored against the circle it landed on and not the
nearest circle: shooting between two targets is a miss, and generosity there
would quietly inflate every hit rate the app reports.

When the board is cleared it pauses for a couple of seconds on your time and then
deals a fresh one, so the screen is never just empty while you wonder whether
something broke.
