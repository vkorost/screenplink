# Quick Start

## What you need

- **Windows 10 or 11**, and a **matte, anti-glare monitor**. Not a glossy screen
  and not a television -- see the previous chapter, which is about nothing else.
- **A USB webcam that delivers sixty frames a second.** A laser pulse lasts about
  a tenth of a second, so at sixty frames it lands in about six of them and the
  app needs two to agree before it will call a shot. At thirty it lands in three,
  which leaves almost no margin, and a camera that quietly drops to ten -- which
  several do, without saying so -- puts the whole pulse in a single frame, where
  nothing works. Check the measured rate rather than the box: **Doctor.cmd** times
  real frame delivery and says how many frames a pulse would land in.
- **A laser trainer cartridge or pistol** that flashes a red dot when the trigger
  breaks.

Unzip the folder anywhere and keep it together. Nothing is installed and nothing
is written to the registry.

## Five steps

**1. Aim the camera at the screen.** It needs the whole display in view with a
margin around it. It may sit to the left or the right, above or below, at an
angle. It must not be able to see out of a window if you can help it.

**2. Start it.** Double-click **Play.cmd**. The app opens full screen on the
monitor the camera is watching.

**3. Do not fire while it measures.** For the first half minute it flashes the
screen, draws small markers on it and fires test dots at itself. The screen says
**HOLD FIRE** while that is happening. A real shot fired during the noise
measurement makes *that shot* the noise floor, and then the detection bar sits
above every genuine hit for the rest of the session, which feels exactly like
the app working for a few shots and then stopping.

**4. Shoot a tile on the menu.** When the measurements are done the Main Menu
appears. Shoot the square you want, click it, or press the key printed on it.

**5. Shoot.** Every shot that registers draws a hole and plays a bang. You have
seventeen rounds. When the magazine is empty the round ends and the screen shows
what you did with it.

To leave, press **Escape**, or shoot the **Exit** tile twice.

## Three things that look like faults and are not

**The app seems to fire shots you did not fire.** Something coloured is on the
monitor the camera watches: another window, a notification, a status panel from
some other program. Move it off that screen.

**The mouse pointer disappears.** It is hidden on purpose while you shoot, and
comes back the moment you move the mouse. A large bright cursor sitting on the
panel is a bright object on the screen the camera is reading: measured here, it
took the noise floor from 3 to 17 on its own and blocked a launch.

**The signal is weak, so you reach for the light switch.** Do not. A dark room is
the best condition this app can have, because darkness is what lets it use a
longer exposure, and the laser's brightness scales with exposure time. If shots
are being missed, press **X** to measure the exposure against the room as it is
now, or move the camera closer to the screen.

## If it will not behave

Close the app and run **Doctor.cmd**. It tests every camera on the machine,
writes its findings to `doctor-report.txt` and shows them. It will tell you
whether the camera itself is fit to shoot with, which is the first question worth
answering.

## One safety line

This is a dry-fire trainer. It assumes a laser cartridge or a dedicated laser
pistol and no ammunition anywhere in the room. The app has no way of knowing what
is in your hand, so that part is yours.
