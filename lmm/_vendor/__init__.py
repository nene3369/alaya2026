"""Pure-Python fallback shims for numpy, scipy, and pytest.

These minimal implementations support running LMM without compiled
C-extension dependencies.  When the *real* packages are installed they
take precedence because site-packages is searched first; the shims are
only activated when this directory is explicitly prepended to sys.path
(see ``inject()`` below).
"""

from __future__ import annotations

import os
import sys

_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))


def inject() -> None:
    """Add the _vendor directory to *sys.path* so that ``import numpy``
    etc. resolve to the shim packages when the real ones are unavailable.

    This is a no-op when *functional* numpy and scipy are already
    importable (i.e. they expose ``ndarray`` and ``sparse``).
    """
    try:
        import numpy as _np

        _np.ndarray  # verify it is functional, not a namespace stub

        import scipy.sparse  # noqa: F401 — also verify scipy

        return  # real packages are available – nothing to do
    except (ImportError, AttributeError):
        pass

    # Save references to stale namespace modules before deletion.
    _stale_numpy = sys.modules.get("numpy")
    _stale_scipy = sys.modules.get("scipy")

    # Invalidate any cached namespace stubs so the shim can replace them.
    for mod in list(sys.modules):
        if mod == "numpy" or mod.startswith("numpy."):
            del sys.modules[mod]
        if mod == "scipy" or mod.startswith("scipy."):
            del sys.modules[mod]

    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)

    # Re-import shims and patch stale module objects in-place so that
    # any existing references (e.g. ``import numpy as np`` captured
    # before inject) also gain the shim attributes.
    import numpy as _np_shim  # noqa: E402
    import scipy as _scipy_shim  # noqa: E402

    for _stale, _shim in [(_stale_numpy, _np_shim), (_stale_scipy, _scipy_shim)]:
        if _stale is not None and _stale is not _shim:
            for attr in dir(_shim):
                if not attr.startswith("__"):
                    try:
                        setattr(_stale, attr, getattr(_shim, attr))
                    except (AttributeError, TypeError):
                        pass
