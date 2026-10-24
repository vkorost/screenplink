# The Magazine: Seventeen Rounds

![A full magazine on both sides, drawn one bullet per round. The numeral sits at the fixed end.](media/mode4.jpg){width=6.9in}

A round of ScreenPlink is one magazine. The default is **seventeen**, because
that is what a CZ Shadow holds, and the point of the whole design is to train the
awareness of the magazine you are actually carrying.

> *"This is not just the game. This is training, and people should understand the
> exact size of their mag and the exact number of bullets they have left in that
> mag."*

## The rules, and there are only four

**Every shot the app registers spends one round, hit or miss.** Nothing gives a
round back.

**The round ends when the magazine is empty.** Not on a timer, not on a target
count. One number decides both how many bullets are drawn and how long the round
lasts, so the strip can never mean anything other than what is left.

**The strip empties from the far end.** The bullets nearest the anchor never
move, so a glance finds the same bullets in the same place every time. The
numeral at the fixed end is there because reading "12" from several metres away
is faster than counting twelve outlines.

**A pull on an empty magazine is a dry pull.** It draws its hole and plays a
bubble pop instead of the gunshot -- the gunshot means a live round went off, and
that was not one. It scores nothing and moves no hit rate. The strip tells you
the gun is dry before the pull; the pop tells you after it.

## The last hole

When the last round goes, the app waits about 1.2 seconds before putting the
round screen up.

> *"I want to see the hole of my last shot for at least a second. As soon as I
> fire the number 17, that hole is supposed to be there, and you wait at least a
> second until you display the end-of-round screen."*

So the last hole is seen where it landed, against the target it struck, rather
than being swept away by a results screen the instant it arrives.

## Changing the size

The magazine is stored in `config\settings.local.json` as `magazine`, and may be
anything from 1 to 50 -- beyond that the strip stops being readable at a glance.
There is no on-screen control for it yet. A change takes effect at the start of
the next round rather than ending the one you are shooting.
