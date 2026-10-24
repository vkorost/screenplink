# PyInstaller spec for ScreenPlink -- ONE FOLDER, not one file.
#
# Build from the app\ folder (the one holding src\, packaging\ and .venv\):
#     .venv\Scripts\pyinstaller.exe packaging\screenplink.spec --noconfirm ^
#         --distpath build --workpath build\_work
#
# WHY ONE FOLDER AND NOT ONE FILE
# -------------------------------
# A one-file build unpacks the whole bundle into a temp directory on EVERY
# launch. This bundle is a couple of hundred megabytes, and startup time is the
# thing three days of work went into cutting (149s -> 30s on the owner's bench).
# It would also put the app's own data in a path that changes every run, which
# is precisely the confusion `assets.user_root()` exists to end. One folder,
# zipped for the download.
#
# WHY THE EXCLUSIONS ARE SAFE, AND WHAT THEY ARE WORTH
# ----------------------------------------------------
# Measured in this venv, 2026-10-20:
#
#     PySide6 installed                656 MB
#       of which Qt6WebEngineCore.dll  195 MB   <- excluded
#     Qt6Core + Qt6Gui + Qt6Widgets     26 MB   <- all the app actually uses
#
# The whole of src/ imports exactly three Qt modules -- QtCore (37 imports),
# QtGui (30) and QtWidgets (7). Nothing imports QML, Quick, WebEngine, 3D,
# Charts, Multimedia, Designer, Pdf, Sql, Bluetooth or Network. So they are
# excluded by name rather than left to the hook's judgement.
#
# DO NOT exclude the Qt PLUGINS. `platforms/qwindows.dll` is how the window
# opens at all, and `imageformats/qjpeg.dll` is how the figure artwork and the
# splash load -- the targets are .jpg. Dropping those gives an app that starts
# and draws nothing, which reads as a broken install.
#
# WHAT IS DELIBERATELY NOT IN THE BUNDLE
# --------------------------------------
#   scripts/   gen_voice_lines.py carries an ElevenLabs API key in PLAINTEXT.
#              It is gitignored, and PyInstaller does not read gitignore, so it
#              is excluded here explicitly. Nothing from scripts/ ships.
#   tests/     1,900-odd checks, the fake camera, the mutation runners. Not a
#              user's business and it is gitignored too.
#   images/fall-originals/   19 MB of staging material, referenced nowhere in
#              src/. The fall POSES the app actually draws live in targets/fall.
#   images/*   except the one splash the app names. The other three files in
#              there, including a 7680x4320 PNG, are not referenced in src/.
#   config/    HIS measured state. A shipped settings.local.json would be a
#              saved setting silently beating a code default -- a scar on this
#              project -- and his exposure, camera name and ROI would poison a
#              different camera's first run. The app's own defaults are already
#              correct for an unmeasured machine: no device name, index -1, and
#              exposure None, which is what forces a full sweep rather than
#              trusting a number measured in another room.
#
# WHAT GOES BESIDE THE EXE RATHER THAN INSIDE THE BUNDLE
# ------------------------------------------------------
# .mcp.json, Play.cmd, Doctor.cmd, CLAUDE.md, README.md and the generated
# settings.default.json are copied next to screenplink.exe by the build script, NOT listed as datas here.
# They have to be where the user and Claude Code can see and read them;
# `datas` would bury them in _internal/ where nothing looks.

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH).parent          # noqa: F821  (SPECPATH is PyInstaller's)

# ---------------------------------------------------------------------------
# The asset folders, read from the app's own table so this file cannot drift
# from it. assets.ASSET_DIRS is the single place that says which folders exist;
# it is stdlib-only and loadable by path on purpose, which is what lets this
# spec import it without dragging Qt or OpenCV into the build process.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_assets", ROOT / "src" / "lsa" / "artwork" / "assets.py")
_assets = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_assets)

# Frozen, `asset_dir(name)` resolves to `bundle_root()/resources/<name>`, and
# bundle_root() is sys._MEIPASS -- which in a one-folder build is the _internal
# directory. So each folder is placed at `resources/<name>` (ASSET_PARENT, read from
# the same module) and the app finds it by the same rule it uses from source. The
# SOURCE of each folder is <project>/resources/<name>; ROOT here is app/, so the
# project root is ROOT.parent.
RESOURCES = ROOT.parent / _assets.ASSET_PARENT
datas = []
for _name in _assets.ASSET_DIRS:
    src = RESOURCES / _name
    if not src.is_dir():
        raise SystemExit(f"asset folder missing, refusing to build a bundle "
                         f"that cannot draw: {src}")
    if _name == "images":
        # SHIP EXACTLY THE FILES THE CODE NAMES, and refuse to build if that
        # list does not account for everything in the folder.
        #
        # The old rule shipped the splash alone, because the other files here
        # were staging material referenced nowhere. That stopped being true:
        # bullet.png moved in, the magazine draws it on EVERY repaint of every
        # drill, and the bundle shipped one of the two files the app needs.
        # The miss was invisible until it reached hardware, where the
        # FileNotFoundError was raised INSIDE paintEvent -- so `painted` never
        # emitted, every `Paint()` timed out, the self test drew no dots, the
        # exposure probe photographed a blank screen, and the app told its
        # owner to unplug a camera that had just measured a 0.91mm mapping.
        # One absent file, 1,722 tracebacks in one launch, and days of it
        # being read as an intermittent camera fault.
        #
        # The names are READ OUT OF THE SOURCE, never retyped: ui.py already
        # carries the comment "read out of this source by the packaging spec",
        # which was true of the intent and false of the code.
        _named = {}
        for _mod, _const in (("display/splash.py", "SPLASH_NAME"),
                             ("artwork/ui.py", "BULLET_NAME")):
            _text = (ROOT / "src" / "lsa" / _mod).read_text(
                encoding="utf-8", errors="replace")
            _hit = None
            for line in _text.splitlines():
                if line.startswith(_const):
                    _hit = line.split("=", 1)[1].split("#")[0].strip()
                    _hit = _hit.strip(chr(34)).strip(chr(39))
                    break
            if not _hit:
                raise SystemExit(f"could not read {_const} from {_mod} -- refusing "
                                 f"to guess which images the app needs")
            if not (src / _hit).is_file():
                raise SystemExit(f"{_const} names {_hit}, which is not in {src}")
            _named[_hit] = _const
        # Anything here the code does not name is a decision, not an oversight:
        # name it in src/ or take it out of the folder.
        _stray = sorted(f.name for f in src.iterdir() if f.name not in _named)
        if _stray:
            raise SystemExit(
                f"{src} holds files the code does not name: {_stray}. Every file "
                f"in images/ must be named by a constant in src/ (see SPLASH_NAME "
                f"and BULLET_NAME) so that a file the app draws can never be left "
                f"out of the bundle again. Name it, or move it out of the folder.")
        for _file in sorted(_named):
            datas.append((str(src / _file), f"{_assets.ASSET_PARENT}/{_name}"))
    else:
        datas.append((str(src), f"{_assets.ASSET_PARENT}/{_name}"))

# PortAudio and libsndfile ride along as data in these wheels, and the app is
# silent without them -- which on this project looks exactly like success.
binaries = collect_dynamic_libs("sounddevice") + collect_dynamic_libs("soundfile")

# ---------------------------------------------------------------------------
# Qt modules this app never imports. Worth about 300 MB of download.
# ---------------------------------------------------------------------------
EXCLUDE_QT = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtWebView",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
    "PySide6.QtQuickControls2", "PySide6.QtQuickTest", "PySide6.QtQmlFeatures",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtUiTools", "PySide6.QtTest",
    "PySide6.QtSql", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtPrintSupport",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtLocation", "PySide6.QtSerialPort", "PySide6.QtSerialBus",
    "PySide6.QtHttpServer", "PySide6.QtNetworkAuth", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtSensors", "PySide6.QtSpatialAudio",
    "PySide6.QtStateMachine", "PySide6.QtSvg", "PySide6.QtSvgWidgets",
    "PySide6.QtTextToSpeech", "PySide6.QtAxContainer", "PySide6.QtDBus",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtConcurrent",
    "PySide6.QtCanvasPainter", "PySide6.QtLabsStyleKit",
]

# Things pulled in by habit that this app does not use. matplotlib and tkinter
# in particular arrive through numpy/other packages and are tens of megabytes.
EXCLUDE_OTHER = [
    "tkinter", "matplotlib", "PIL", "pandas", "scipy", "IPython",
    "pytest", "setuptools", "pip", "wheel",
    "PyQt5", "PyQt6", "PySide2",          # one Qt binding, never four
]

a = Analysis(                              # noqa: F821
    [str(ROOT / "packaging" / "screenplink_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    # pygrabber enumerates DirectShow devices through comtypes, and comtypes
    # generates wrapper modules at run time -- which is a classic way for a
    # frozen build to work on the build machine and fail on a clean one. These
    # are named so the hook cannot miss them; whether it is sufficient is a
    # question for the clean-machine test, not for this comment.
    hiddenimports=[
        "comtypes", "comtypes.client", "comtypes.stream",
        "pygrabber", "pygrabber.dshow_graph",
        "sounddevice", "soundfile", "pycaw", "pycaw.pycaw",
        "lsa.mcp",                         # reached only via --mcp, not by import
        "lsa.doctor",                      # reached only by dispatch, not by import
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDE_QT + EXCLUDE_OTHER,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)                          # noqa: F821

# ---------------------------------------------------------------------------
# ONE EXECUTABLE. MODES ARE CHOSEN BY ARGUMENT, NOT BY BUILDING MORE BINARIES.
# ---------------------------------------------------------------------------
# An earlier version of this file built three exes (windowed app, console MCP,
# console doctor) from one Analysis, on the reasoning that a windowed Windows
# process has no console to print to. That multiplied artefacts to dodge a
# dispatch problem; the owner rejected it, and he was right. There is one
# binary and it takes arguments:
#
#     screenplink.exe --target        the app            (Play.cmd)
#     screenplink.exe --mcp           the MCP server     (.mcp.json)
#     screenplink.exe --doctor --all  the camera doctor  (Doctor.cmd)
#
# It stays WINDOWED (console=False). A console-subsystem exe pops a console
# window onto the very monitor the camera watches, and a lit window in frame is
# this project's oldest source of phantom detections -- the same reason
# `startScreenPlink.cmd` only holds its window open on failure.
#
# The cost of windowed is that a windowed process started from a terminal does
# not print to that terminal. It is paid by REDIRECTION in the wrappers rather
# than by Windows API tricks (no AttachConsole, no AllocConsole):
#
#   * --mcp needs nothing. Claude Code supplies pipes for stdin, stdout and
#     stderr, and a windowed binary speaks the protocol correctly over them
#     (measured 2026-10-20: protocol-only stdout).
#   * --doctor is run by Doctor.cmd as `screenplink.exe --doctor --all
#     > doctor-report.txt 2>&1`, then the file is shown. That guarantees
#     visible output AND leaves a report Claude Code can read directly, which
#     is better than console text for the real first-run use.
#
# Dispatch is by ARGUMENT ONLY (see screenplink_entry.py). There is no dispatch
# on the executable's own name, because there is only one name.
#
# What has NOT been proven without a build: the frozen `--doctor` path and the
# two .cmd wrappers. Those wait for the clean-machine test.
app = EXE(                                 # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="screenplink",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                             # UPX-packed DLLs trip antivirus, and
                                           # this is a .exe a stranger downloads
    console=False,                         # see the note above
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(                            # noqa: F821
    app,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ScreenPlink",
)
