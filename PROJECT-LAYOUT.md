# What is in this repository, folder by folder

A map for anyone changing the code. For shooting the app, read the
[User Guide](manual/README.md) instead; for things going wrong, read
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

```
screenplink/
├── README.md            what it is, and how it works
├── PROJECT-LAYOUT.md    this file
├── TROUBLESHOOTING.md   when it does not work
├── app/                 the program: code, tests, tools, launchers
├── resources/           everything the app draws or plays
└── manual/              the user guide (GENERATED — see below)
```

Three folders, and the split is deliberate: **`app/` is what the program *is* and
what it *writes*; `resources/` is what it *reads*; `manual/` is for the person
holding the pistol.** The code sits one level down in `app/` so that a release can
be assembled by taking `app/` plus `resources/` and nothing else.

---

## `app/` — the program

| | |
|---|---|
| `requirements.txt` | direct dependencies, pinned |
| `requirements.lock.txt` | the whole resolved tree, exact versions |
| `requirements-build.txt` | what the PyInstaller build needs on top |

### `app/src/lsa/` — the package

`lsa` is the whole application. The CV core deliberately imports no Qt, so it
stays usable if the shell is ever replaced; `artwork` is checked for that by a
test that imports it in a subprocess and fails if Qt appears.

| Package | What it owns |
|---|---|
| **`detect/`** | `laser.py` — the detector. Scores every frame as red minus the strongest of green and blue, finds blobs, tracks them across frames, and decides what is a shot. One file, because it is the hot path: it has about 14 ms per frame before 60 fps becomes 30. |
| **`camera/`** | Opening the device and keeping it alive. `capture.py` is the capture thread, `controls.py` the exposure/gain writes (rationed — this camera has hung for minutes on one write), `devices.py` enumerates and flags virtual cameras, `evidence.py` the per-frame ring buffer that makes a shot explainable afterwards. |
| **`calibrate/`** | Working out where the screen is and which pixel a dot maps to. `pattern.py` photographs markers the app draws, `mapping.py` fits the homography plus radial term and holds the learned sensitivity map, `screen.py` the flash-based screen search, `align.py` the drift correction, `flash.py` the full-screen flash window. |
| **`display/`** | Everything on the target screen. `window.py` is the fullscreen window, `scene.py` composites layers, `layers.py` holds the HUD, prompts and verdict board, `splash.py` the startup art, `foreground.py` the Windows dance that makes the window actually take the screen, `winowner.py` names whatever is covering it. |
| **`modes/`** | The drills, one module each, plus `base.py` for what they share, `menu.py` for the shoot-to-select menu, `magazine.py`, `efficiency.py` for the score, `versus.py` and `split.py` for two players, `sprite.py` for the figure artwork. A drill is a `Layer` with one extra hook, so nothing else in the app knows modes exist. |
| **`audio/`** | `mixer.py` plays, `listener.py` hears the gun through the camera's microphone, `voice.py` speaks, `bang.py` and `fall_sounds.py` choose which sample. |
| **`artwork/`** | Loading pictures and making them safe to draw: `load.py` forces true grey, `ceiling.py` derives the brightness cap from the darkest supplied figure, `zones.py` and `rings.py` extract scoring rings from the ink, `catalog.py` is the cast, `assets.py` is **the one place that says where every asset folder is**. |
| **`introspect/`** | An in-process HTTP server (off by default) that answers what the app is measuring right now. Read-only by construction: the capture thread's write methods are booby-trapped in tests to prove no route can reach them. |
| **`tools/`** | `camera_lab.py` — the application object itself, the operator console, and the startup chain. `align_check.py` checks camera placement. |
| **`director.py`**, **`routines.py`** | One run slot and the measurement routines that take it. Every hardware procedure — set an exposure, draw, wait, photograph, put it back — is a generator that yields what it is waiting for, so two of them can never interleave. |
| **`selfcheck.py`** | Measurements in, verdict and actions out. Pure, so it is tested exhaustively. |

### `app/tests/` — 52 files

Twenty-six suites plus six **mutation runners** and the helpers they share. No
suite needs a camera or a display. The mutation runners put known faults back
into the code and fail the build unless the check written for each one goes red —
a check that cannot fail is not a check, and this project has shipped several.
`test_coldstart.py` drives seven whole launches against a simulated camera.

### `app/tools/` — 20 files

Instruments, not application code. `camera_health.py` is the doctor,
`camera_alive.py` opens the camera read-only, `audio_meter.py` reads what Windows
stores about our volume, `status.py` prints the project's state from the logs and
tags on disk, `window_stack.py` names what is covering the target screen,
`replay.py` replays a recording through the real detector, `read_string.py`
tabulates a shooting string, `suites_for.py` and `suite_plan.py` say which suites
a change needs.

### `app/scripts/` — the launchers

| | |
|---|---|
| `Play.cmd` | the game. Runs the exe if it is beside it, else from source |
| `Doctor.cmd` | the camera doctor; writes `doctor-report.txt` |
| `run_tests.bat` | every suite |
| `run_affected.ps1` | only the suites a change needs |
| `run_live.ps1` | a live run, refusing to start against a wedged camera |
| `repeat_gate.ps1` | N cold launches, judged on a fixed checklist |
| `stop_lab.ps1` | closes the app the way the app would. **Never End Task** |
| `run_camera_lab.bat` | the operator console, for development |

### `app/packaging/`

`assemble.py` builds the one-folder Windows release and **refuses** a bundle whose
assets are not where the asset table says. `screenplink_entry.py` is the single
entry point that decides between the game, the doctor and the MCP face before
anything heavy is imported.

---

## `resources/` — everything the app reads

Nothing here is code, and every folder is reached through `asset_dir()` in
`artwork/assets.py` rather than by counting parent directories, so the same names
work from source and inside a frozen bundle.

| Folder | Holds |
|---|---|
| `targets/` | 15 figures plus a scoring-zone sidecar each; `targets/fall/` holds one dropped pose per figure |
| `animations/soldier/` | 121 frames of the walking soldier; `soldier-fall/` 32 more for his fall |
| `menu/` | 14 tiles for the shoot-to-select menu |
| `images/` | the startup art and the magazine bullet |
| `sounds/` | 23 samples: 7 gunshots the app may pick between, and 16 reserved effects it must never offer as a gunshot |
| `voice/` | 22 spoken lines in three voices — `announcer`, `mobster`, `soldier` |

---

## `manual/` — the user guide

Thirteen chapters, a PDF and an EPUB, indexed by [manual/README.md](manual/README.md).

> **Do not hand-edit anything under `manual/`.** It is generated: a deploy script
> regenerates all of it from the authoring source, so an edit made here is
> destroyed on the next build without warning. Send changes to whoever owns the
> book instead. The PDF and EPUB are rewritten by every build and will show as
> modified in git constantly — that is a rebuild, not a change.

---

## What is not in the repository

**Created when you run the app**, under `app/`, and all gitignored:

| | |
|---|---|
| `config/` | `settings.local.json` (your camera, exposure and thresholds) and `calibration.json` (the mapping measured for your camera position, and the sensitivity learned from your own shots) |
| `logs/` | `camera_lab.log`, one dated header per launch — the app's account of itself |
| `cache/` | derived and disposable; deleting it costs a recompute |
| `recordings/` | frames and videos the app saves when something is worth looking at |

**A shot-in `config/` is genuinely better than a fresh one**, because the
sensitivity map can only be learned from real shots. It is the one thing worth
backing up.

**Deliberately absent**, because they would not help anyone cloning this: the
author's working notes, the project's internal rules, the research reports, and
the calibration backups from one particular room.
