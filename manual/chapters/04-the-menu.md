# The Menu, and Moving Between Drills

![The Main Menu. Two player-count tiles, nine drills, and Exit.](media/main_menu.jpg){width=6.9in}

Press **M** for the menu at any time. Every tile can be shot, clicked with the
mouse, or started by pressing the key printed in its corner.

## How a choice is made

A shot at a tile does not switch the drill on the spot. It records the choice,
and the app acts on it a fraction of a second later, on its next tick. There is
one route for switching drills, and it is the route that announces the new drill
aloud, writes it to the log and resets the score. A second route that skipped
those three is the kind of thing that quietly rots.

**The first choice in a turn wins.** Two pistols can land two shots in the same
camera frame, and a later one does not overwrite an earlier one.

**Nothing about the tiles is typed into the menu.** They are whatever drills the
app has registered, in key order, with their own names. A drill added in a later
version appears here by itself, correctly labelled, with no risk of the menu
describing a drill that no longer exists -- which the old hand-typed key legend
did for three days.

## Starting a new round

Press a drill's own key again to restart it: a fresh magazine, a cleared score,
and the holes wiped. **N** does the same thing for whatever is running. When a
round has finished, the **NEXT ROUND** box on the round screen starts the next
one; it can be shot, clicked or started with N, and it is armed 0.8 seconds after
it appears so that a shot still in flight cannot press it.

## One player, two players

**O** is one player and **T** is two, and the first two tiles on the menu do the
same thing. The choice applies from the next drill you start.

## Exit

The **Exit** tile closes the app, and it is labelled Esc because Escape does the
same thing. **It takes two hits.** The first arms it and the tile says so; the
second closes the app, and the arming lapses after four seconds. A phantom
detection closing the app in the middle of a string is the worst accident
available here, which is why this one tile is the only thing in the app that asks
twice.

## Why the menu is not up while the app measures

Eleven lit tiles are a lot of light on a screen the camera is trying to read. The
app therefore spends its measuring time on Static Targets -- thin rings on black
-- and only brings the menu up when it has finished. Measured on the same machine
minutes apart: with the menu up the noise floor read 3 and the test dots on the
far side of the screen measured 13; on Static Targets the floor read 1 and the
same dots measured 32.
