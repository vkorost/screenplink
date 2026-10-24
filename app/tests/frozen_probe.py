"""Child process for test_rules section R: resolve every path the app uses, with or
without a simulated PyInstaller layout, and print them as JSON.

argv: <src dir> <bundle dir> <app dir (holds the .exe)> <frozen: 1|0>
Runs with every LSA_* override removed, so the DEFAULTS are what is measured.
Each probe is caught separately: against code that predates the fix, a probe that
cannot be answered is reported as an ERROR string, not a crash, so the parent can
fail on it by name.
"""
import json, os, sys
src, bundle, app, frozen = sys.argv[1:5]
for k in [k for k in os.environ if k.startswith("LSA_")]:
    del os.environ[k]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, src)
if frozen == "1":
    sys.frozen = True
    sys._MEIPASS = bundle
    sys.executable = os.path.join(app, "ScreenPlink.exe")
out = {}


def probe(name, fn):
    try:
        out[name] = str(fn())
    except Exception as e:                           # noqa: BLE001
        out[name] = f"ERROR: {type(e).__name__}: {e}"


from lsa.artwork import load, tiles, sequence, ui, ceiling
probe("targets", load.targets_dir)
probe("menu", tiles.menu_dir)
probe("animations", sequence.animations_dir)
probe("bullet", ui.bullet_path)          # must land in the same images folder as the splash
probe("cache_file", lambda: ceiling._CACHE_FILE)
from lsa.display import splash
probe("images", lambda: splash.splash_path().parent)
from lsa.tools import camera_lab as cl
probe("voice", lambda: cl.CameraLab.voice_dir)
probe("sounds", lambda: __import__("lsa.artwork.assets", fromlist=["x"]).asset_dir("sounds"))
probe("recordings", lambda: cl.RECORDINGS)
probe("calibration", lambda: cl.CALIBRATION_PATH)
probe("settings", lambda: cl.SETTINGS_PATH)
probe("log", lambda: cl.LOG_PATH)
from lsa.introspect import view
probe("introspect_settings", view._settings_path)
print("PROBE_JSON " + json.dumps(out))
