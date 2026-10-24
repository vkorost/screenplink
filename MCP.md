# Asking the app what it sees

ScreenPlink can answer questions about itself **while it is running**, over an
in-process server. Claude Code talks to it either through the MCP server the
release registers in `.mcp.json`, or directly over HTTP on loopback. Same
routes, same answers.

This exists because the recurring failure in this project was never that the
app did not know. It was that the app knew and nobody could see it.

---

## Turning it on

It is **off by default**, and the app behaves identically without it.

```bat
screenplink.exe --target --introspect     :: the game, with the server
curl -s http://127.0.0.1:8777/health
```

The release ships `.mcp.json` beside the executable, so a Claude Code session
started in that folder finds the tools by itself:

```json
{ "mcpServers": { "screenplink": { "command": "./screenplink.exe",
                                   "args": ["--mcp"] } } }
```

---

## What it can be asked

| Tool | Route | Answers |
|---|---|---|
| `health` | `GET /health` | alive, uptime, whether a camera is open, what this server has cost |
| `state` | `GET /state` | mode, layers, floor, bar, exposure, gain map, mapping error, screen ownership, fps, control writes, what the Director is running |
| `settings` | `GET /settings` | effective beside saved beside code default |
| `frame` | `GET /frame` | the latest camera frame, lossless PNG |
| `frame_measurements` | `GET /frame.json` | its measurements without the image |
| `score` | `GET /score` | the detector's arithmetic as numbers: raw chroma, local amplification, background model, final score, **plus every blob above threshold with the reasons it would be dropped** |
| `score_png` | `GET /score.png` | that map as a picture. 8 bit, so it clips and rounds: for a quick look, never for an answer |
| `shots` | `GET /shots` | the last N detections with per-frame evidence |
| `drawn` | `GET /drawn` | what is painted, where, in screen AND camera coordinates, with its red excess |
| `press_key` | `POST /key` | one key to the target window, no focus stolen |
| `click` | `POST /click` | one left click at those screen pixels, no focus stolen |

---

## The one rule it is built around

**It returns measurements, never conclusions.**

The app's own verdict is available, but it arrives as `app_says` and is labelled
as one more datum to cross-check rather than the answer. That is not caution for
its own sake. On 2026-10-17 the app printed *"100% of his measured shots clear
the bar, margin 6.1x"* on the same line as a measurement showing its own test
dot **below** its own threshold. An endpoint that returned the app's opinion
would only have been wrong faster.

The most useful single route is `score`, because the log records only the
detections that **survived**. A blob that was rejected has never existed anywhere
except inside the running process, and a recording's re-encode destroys
near-threshold detail. The live process is the only place that information is.

---

## What it will not do

- **It cannot write a camera control.** Read-only for the camera by construction,
  and the test suite booby-traps the capture thread's write methods to prove no
  route reaches them. A single exposure write has hung this camera family for
  272 seconds.
- **It binds to loopback only.**
- **A click is never a shot.** It has no pulse, no peak and no camera position,
  and nothing it does reaches the score, the hit rate or the sensitivity map.
- **It stays off the hot path.** The detector has about 14 ms per frame before
  60 fps becomes 30. Measured at 1280x720: detector alone 5.06 ms median and
  7.11 ms p95; with an agent querying every route, +0.04 and -0.21 ms.

---

## A worked example: the day it paid for itself

On 2026-10-24 a packaged build told its owner **NOT READY, the camera is not
delivering the app's own test dot, unplug the camera**. The camera was fine.

What the instrument gave, in order:

1. `state` reported a mapping measured **this launch** at 0.91 mm held out,
   fps 48, noise floor 2, bar 8, and `ready_for_real_shots: true`. A mapping is
   measured by photographing eighteen markers on that screen, so the camera
   could not be blind four seconds later. Two app statements, one of them false.
2. `press_key` re-ran the self check without stealing focus from the fullscreen
   window: **12 of 12 at margin 14**, on the same camera, minutes later.
3. The exposure probe's own table, same second, three candidates:
   `-4: dot 109`, `-5: dot 5`, `-6: dot 108`. No exposure curve has that shape.
   One photograph had been taken before its dot was painted.
4. `drawn` and a window capture showed the MENU button and the magazine missing
   from a drill, which pointed at the paint path rather than the camera.

The cause was a single artwork file left out of the bundle. The exception was
raised inside `paintEvent`, so the signal that says "this frame is on the glass"
never fired, every wait-for-paint timed out, the self test drew no dots and the
probe photographed a blank screen. One absent file, 1,722 tracebacks in one
launch, and several days of it being read as an intermittent camera fault.

**Every wrong diagnosis that week came from reasoning about symptoms. The right
one came from asking the running app.**

---

## Driving it without a person at the keyboard

`press_key` and `click` exist so a session can exercise the app while the window
stays in front of the monitor the camera watches. That matters more than it
sounds: Windows refuses to bring a window forward for a process that is not
already in front, so anything that steals focus leaves the camera photographing
the desktop, and every measurement taken after that is a measurement of the
wrong screen.

The keys worth knowing:

```
1-9  the drills     M menu    C self check    E erase holes
G    re-measure the mapping   X re-measure the exposure
S    stats          L signal meter            0 show/hide the info layers
Esc  exit           F11 close the target window
```

**`O` (one player) and `T` (two players) are accepted only on the menu.** Sent
inside a drill they are delivered and ignored, with nothing said, which reads
exactly like the app not listening.

---

## If nothing is listening

The server is off unless `--introspect` was passed, or `LSA_INTROSPECT=1` is set.
`Play.cmd` does not pass it. There is no error when it is absent: the tools
simply report that nothing is there.
