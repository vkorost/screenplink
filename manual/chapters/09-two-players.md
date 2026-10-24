# Two Players

Press **T**, or shoot the Two Players tile. The screen splits down the middle and
two people shoot at once, one per side, at the same drill. **O** puts it back to
one. Every drill in the app runs two-player.

![Two players, one drill, one magazine each.](media/mode3.jpg){width=6.9in}

## Why the split is left and right

Type sizes, the reserved band at the top and the strip at the bottom are all
worked out from the height of the screen. Splitting left and right keeps every
one of them exactly as it is in single player. A top-and-bottom split would have
halved the height and shrunk every piece of text on a screen that is read from
several metres away.

Each half gets exactly half of what is left after the divider, so the two
playfields are equal by construction. A shot to the left of the divider's centre
line belongs to the left player and anything from the centre line rightwards
belongs to the right: one comparison, no dead zone, no argument.

## A round is one magazine each

Each side has its own strip of seventeen and its own score. A side is finished
when the app has seen and scored seventeen of its shots.

That is a deliberate choice about fairness. If the app under-detects one
shooter's pistol, that shooter simply pulls the trigger a few more times to reach
seventeen registered shots. Both sides therefore always reach exactly the same
denominator, the round always closes by itself, and no guess about shots the
camera never saw ever enters the result.

> *"If the shots are missed, they are missed randomly from both sides. Just count
> the shots that had been registered."*

## One reveal, shared

In the Reaction drill both halves are shown their target at the same instant. Two
independent copies would each draw their own wait, one player would see his
target up to two seconds before the other, and the drill would measure luck
rather than reaction.

A reveal that one side never resolves does not hang the other: after a timeout
that side is marked **TOO SLOW** -- counted, never a hit and never a false start
-- and the next reveal goes up for both.

## Different figures, same instant

In the figure drills each side gets a *different* figure, appearing at the same
moment, mirrored so the pair is symmetric about the divider. The cost of that is
known and was accepted: the figures are not all the same size, and the ten-ring
area varies by about 1.6 times across the cast. It is a game between two people
in a room, and watching your opponent shoot a different villain is worth more
than the last few per cent of fairness.

## Who wins

**Points win the round.** Everything else is shown beside the result and may
disagree with it:

```
 LEFT                            RIGHT
 17 of 17 shots                  17 of 17 shots
 165 of 170 points  BEST         120 of 170 points
 60.0 seconds                    10.0 seconds
 EFFICIENCY  2.75                EFFICIENCY  12.00  BEST

 SCORE  65                       SCORE  282
 AIM 97%   SPEED 7%              AIM 71%   SPEED 40%
 BEST SCORE SO FAR  212          BEST SCORE SO FAR  212
 BEST EFFICIENCY SO FAR  3.99    BEST EFFICIENCY SO FAR  12.00

            ROUND 2 COMPLETED
            LEFT WINS by 45 points
            RIGHT is more efficient, 12.00 to 2.75

                [ NEXT ROUND ]
```

That example is the specification, not an accident. A shooter who takes a minute
to place careful hits wins on points, while a shooter who empties the magazine in
ten seconds wins on efficiency -- and, here, on the score as well, because the
score counts speed and the round does not. Both sides' figures are shown and the
better of each is marked, so the trade is visible rather than inferred, and the
argument afterwards is about which number you care about rather than about what
happened.

Each side's record book is the same one described in *The Score*: a two-player
half is stored separately from a solo round, because it is a different test, but
there is no shooter identity in it. The two of you share one book.

**The sides swap every round** and the totals follow the player rather than the
half of the screen, so a better view, a better angle or a brighter corner cannot
decide a match. The swap is not announced on screen; it is in the log.
