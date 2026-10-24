"""Arcade modes: the pluggable things to shoot at.

Importing this package registers the built-in modes, so anything that wants a
list of what is available only has to import `lsa.modes`.
"""

from .base import (Mode, ModeHost, ShotEvent, ShotOutcome, available, by_key,
                   by_name, register, registered)
from . import static as _static            # noqa: F401  (registers on import)
from .static import StaticTargets
from . import circles as _circles          # noqa: F401
from . import ball as _ball                # noqa: F401
from . import balls as _balls              # noqa: F401
from . import reaction as _reaction        # noqa: F401
# discriminate (shoot/no-shoot) was REMOVED on 2026-10-18 at his request --
# "I don't really understand what that is" -- and everything after it moved
# up a number. The module is gone from the package; its file is kept out of
# the import so nothing registers it.

from . import popup as _popup              # noqa: F401  (key 7)
from . import image_target as _image       # noqa: F401  (key 8)
from . import advancing as _advancing      # noqa: F401  (key 9)
from . import animated as _animated        # noqa: F401  (key A)
from . import menu as _menu                # noqa: F401  (key E; imported last, it lists the others)
from . import shortcuts as _shortcuts      # noqa: F401  (key K; after the menu, so M precedes K on its screen)

__all__ = ["Mode", "ModeHost", "ShotEvent", "ShotOutcome",
           "available", "by_key", "by_name", "register", "registered"]
