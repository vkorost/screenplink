# The Screen It Needs

**ScreenPlink needs a matte, anti-glare computer monitor.** This is the one
hardware requirement in the app that cannot be worked around by moving anything,
changing a setting or buying a better camera, so it is worth settling before you
set anything up.

Glossy, semi-gloss and "glass-look" anti-reflective screens are not supported.
That includes most modern televisions. On those surfaces a laser hit fired from
an angle is frequently undetectable.

| Display | Works |
|------------------------------------------------|---------------------------|
| Matte or anti-glare computer monitor | Yes |
| Glossy or semi-gloss monitor | No |
| Glossy television -- most LCD, QLED and OLED sets | No |
| Projector and screen | Not in version 1 |

A projector is the one entry on that list that is not about gloss. A projection
screen is diffuse and would scatter the dot perfectly well. The obstacle is the
palette: this app draws light grey on black, and a projector makes black by
projecting nothing, so the dark parts of the picture are simply whatever the room
does to a white screen. Making it work means inverting the whole scheme -- dark
targets on a lit field -- together with every brightness rule derived from the
present one. That is a different app rather than a setting, and it is not in this
version.

## Why the surface decides it

The app finds your shot by watching the screen and looking for the laser spot. It
can only see that spot if the screen scatters some of the beam back toward the
camera.

A **matte** screen has a micro-textured coating that throws incoming light in
every direction at once. The spot is visible from almost anywhere, which is the
whole reason an ordinary webcam parked off to one side works at all.

A **glossy** screen behaves like a mirror. Shoot it at an angle and most of the
beam leaves at the matching angle on the other side, away from the camera. An
anti-reflection coating weakens what little comes back, and the panel's own layers
absorb a good part of the remainder.

So a glossy screen often registers hits fired square on and misses most of the
ones fired from an angle. **Moving the camera does not fix it.** The mirror
direction depends on where on the screen the shot landed, so it changes with every
shot, and no fixed camera position can follow it.

## What has actually been tried

| Display | Finish | Result |
|---------------------------------|----------------------|---------------------------------|
| Dell UltraSharp U4320Q, 43-inch 4K | Matte | Hits visible at every angle tried |
| Samsung 75-inch QLED, 2019 Q7 series | Glossy, anti-reflective | Sometimes visible head-on, mostly invisible at an angle |

The Dell is the screen this app was built and measured on. Everything in this
guide -- every threshold, every timing, every number -- came off it.

## Why no other app asks this of you

The requirement reads like a limitation until you look at what everything else in
this category asks for instead. Checked on 22 October 2026, against the vendors'
own published documentation.

Laser dry-fire products divide into two kinds, and **neither of them shoots at a
screen**. Most watch a **printed paper target**: the camera is pointed at the
target on the wall, and a television, if one is involved at all, is only there to
show you your hits from across the room. The rest **project** onto a screen or a
white wall, and the surface they specify is a matte one -- one system calls for
"a white projection screen (1.0 gain max, no silver screen) or flat, non-glossy,
white wall", which is the same physics as this chapter arrived at from the other
direction.

Two products address using a flat screen as the target. **Both of them tell you
to cover it first.**

One recommends against the idea altogether -- *"we recommend using a projector
instead of a flat screen (TV, monitor, LED, LCD, etc)"* -- and its description of
what goes wrong is this app's symptom, written by someone who has never seen it:

> "you see a bright laser (red or green) dot on the screen, but from the camera
> the same laser dot is not even visible"

Its remedy is to stick a transparent film made for laser pointers onto the
screen: one *"transparent enough to see through but frosted enough to reflect the
laser"*. That is, to stop it being glossy.

The other is the largest system in the category, and its own setup guide for
televisions is blunter still:

> "Use with TV or monitor will require our filter material that is taped to the
> screen in order to track the laser and help control reflections."

> "Some TVs will only reflect the laser at an angle incident to the angle the
> laser enters the screen. Therefore the camera will not be able to see the laser
> and track it."

> "Some TVs have a protective layer that is not directly on the screen surface...
> There is no way to fix this unfortunately and this means it will not work."

The material they supply for the job is a vinyl tablecloth. The middle quotation
is this chapter's physics, in the words of the people who sell the market-leading
system -- and their answer to it is to tape something over the panel.

Nothing goes on your screen here. The requirement is instead that the panel be
matte to begin with, which is the same fact approached from the other side: a
frosted film and a matte coating do the same job, and one of them is already on
the monitor.

### And the disagreement, which is the interesting part

That same manual blames **anti-glare coatings** for losing the dot. This app
measures the opposite: anti-glare is the finish that works. Both can be true, and
what reconciles them is where the camera stands. That manual also advises placing
the camera **directly in front of the screen**.

Square on, a glossy panel's mirror reflection comes straight back toward the
camera, so gloss helps and a matte coating spreads the same light away. Off to one
side, the mirror reflection leaves on the far side of the screen and only the
diffuse scatter ever reaches the lens -- so matte becomes the only thing that
works and gloss fails. The glossy television in the table above behaved exactly
that way: sometimes visible head-on, mostly invisible at an angle.

**So the matte requirement is the price of the camera being off to one side**, and
that is the trade this app makes on purpose. It is what lets you shoot at a
monitor you already own, with nothing standing between you and the screen, no
projector to cast your own shadow across, and no paper to print and replace.

## How to tell what you have

**Look at a reflection.** Switch the display off and stand so that a lamp or a
window is behind you.

- A sharp, mirror-like reflection means glossy, and it will not work properly.
- A blurred, diffuse glow means matte, and it will.

**Read the specification.** "Anti-glare", "matte" or "3H anti-glare hard coating"
is what you want. "Glossy", "glare type" or "anti-reflection glass" is what you do
not.

**Then let the app look at it.** Press **C** and it runs its self check: it draws
a dozen red dots at known places on your screen at the brightness of real shots,
photographs each one, and reports how many it found and how far out it placed
them.

Be careful how you read that number, because it answers a narrower question than
it appears to. A screen *emits* its own test dot, while your laser *reflects* off
the panel's coating, and the two fall off across a screen quite differently --
measured here, the app's own dot held roughly steady from one side of the screen
to the other while a real laser dropped by half. So:

- the **position error** it reports is real, because it is measured through the
  same camera-to-screen mapping your shots are;
- the **detection rate** it reports is not your pistol's. It proves the app can
  see dots on your screen and place them accurately. It cannot tell you what
  fraction of your trigger pulls will register.

The honest test of a glossy screen is still a magazine of real shots fired from
where you intend to stand.

## Size, and getting the dot big enough

Any size of matte screen is fine, and nothing in the app is laid out in pixels
(see the previous chapter), so a drill plays the same on a 24-inch monitor as on a
43-inch one.

What changes with size is how many camera pixels the laser spot lands on. The
bigger the screen, the smaller a share of the camera's view each shot occupies. On
a large display, either use a camera with more resolution or move it closer, so
that the screen fills the frame with only a modest margin around it. A spot
spread across four camera pixels is near the limit of what can be told from
sensor noise; the same spot across twenty is unmistakable.

Two settings matter for the same reason, and the app handles both by itself: the
camera's exposure and gain are driven down and locked, because automatic exposure
brightens the whole picture and washes the spot out, and the targets are drawn
dark, because any lit grey underneath the dot is subtracted from it, one unit for
one unit.
