"""Laser Shooting App."""

# Must happen before anything imports cv2. See _env for the measurements.
from . import _env  # noqa: F401

# THE PRODUCT VERSION, and the only one. 1.0.0 because this repository has
# published nothing before: the 1.1.0 that was hardcoded in mcp.py was
# inherited from a different repository and meant nothing here. Everything
# that reports a version reads THIS -- the MCP server identifies itself with
# it, and /health reports it beside the commit when there is a commit to
# read, so a bug report names a build instead of saying "unknown".
__version__ = "1.0.0"
