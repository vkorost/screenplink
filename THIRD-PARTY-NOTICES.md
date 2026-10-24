# Third-party notices

ScreenPlink's own code is [Apache-2.0](LICENSE). The Windows release bundles the
libraries below, which keep their own licences. The asset credits (sounds,
images, speech) are in the [README](README.md#credits-and-what-is-actually-mine).

## Libraries under the LGPL

These two are the reason this file exists. The LGPL permits shipping them beside
differently licensed code, which is why ScreenPlink's own Apache-2.0 terms are
unaffected, and in return it asks that you be told what they are, where their
source is, and that you be able to replace them with your own build.

| Component | Licence | Source |
|---|---|---|
| **Qt 6** and **PySide6** / **shiboken6** | LGPLv3 | <https://download.qt.io/official_releases/QtForPython/> and <https://download.qt.io/official_releases/qt/> |
| **libsndfile** (inside `soundfile`) | LGPL-2.1 | <https://github.com/libsndfile/libsndfile> |

**You can replace them.** The release is a one-folder build on purpose, so the Qt
libraries are ordinary files you can swap:

```
ScreenPlink/_internal/PySide6/Qt6Core.dll, Qt6Gui.dll, Qt6Widgets.dll, ...
ScreenPlink/_internal/_soundfile_data/
```

Drop in your own build of the same version and the app will use it. (One-folder
was chosen because a one-file build unpacks a few hundred megabytes into a
temporary directory on every launch, and startup time is what three days of work
went into cutting. It makes this straightforward as well.)

## Libraries under permissive licences

| Component | Licence |
|---|---|
| NumPy | BSD-3-Clause |
| OpenCV (`opencv-python`) | Apache-2.0 |
| sounddevice, and PortAudio inside it | MIT |
| soundfile (the Python wrapper) | BSD-3-Clause |
| pycaw | MIT |
| pygrabber | MIT |
| comtypes, cffi | MIT |
| CPython and its standard library | PSF License |

## Building it yourself instead

Nothing here is required if you run from source: `app/requirements.txt` fetches
each library from PyPI under its own terms, and no binaries are redistributed by
this repository. The notices above apply to the packaged Windows release, which
is the only thing that redistributes them.
