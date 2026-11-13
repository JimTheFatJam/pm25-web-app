"""
backend/_ee_windows_compat.py
============================================================================
Windows compatibility shim for `import ee`.

The problem, confirmed directly: earthengine-api's `ee/__init__.py`
unconditionally imports its CLI tool (`from .main import main`), which
imports `blessings` for terminal color/formatting, which imports two
Unix-only stdlib modules: `fcntl` (for `ioctl`) and `termios` (for
`TIOCGWINSZ`). Neither exists on Windows -- `pip install fcntl` fails
because there is no such package; it's a compiled-in part of Unix
Python builds, not something publishable to PyPI. `windows-curses`
fixes the `_curses` import one step earlier in the chain, but does not
provide `fcntl`, which is a separate, unrelated module.

None of this is actually needed to USE the Earth Engine Python API
(ee.Initialize(), ee.Image(), reduceRegions(), etc.) -- it is only used
by blessings to query terminal window size for CLI color output, a
feature this backend (a headless Flask server, never printing colored
CLI output) never exercises.

Confirmed directly (by reading blessings' own source) that its one
`ioctl()` call is already wrapped in `try: ... except IOError: pass`,
with a graceful fallback to environment variables or `(None, None)` --
exactly the code path already written to handle a non-tty environment.
`IOError` is an alias for `OSError` in Python 3, so a stub `ioctl` that
simply raises `OSError` triggers that existing, intended fallback --
this is not a hack around blessings' logic, it's giving it the exact
signal it already knows how to handle.

This module does NOTHING on Linux/macOS (where the real fcntl/termios
already exist) -- it only installs the stub if the real import fails,
so it's always safe to import first, unconditionally, before `import ee`.
"""

import sys
import types


def _install_stub_if_missing(module_name, attrs):
    try:
        __import__(module_name)
        return  # real module already available (Linux/macOS, or a Windows
                 # environment where this was somehow already resolved) --
                 # do not shadow it.
    except ImportError:
        pass

    stub = types.ModuleType(module_name)
    for name, value in attrs.items():
        setattr(stub, name, value)
    sys.modules[module_name] = stub


def _stub_ioctl(*args, **kwargs):
    # Mirrors exactly what a real ioctl() raises when the target isn't a
    # tty (e.g. output piped to a file) -- the case blessings' own
    # except IOError branch already exists to handle gracefully.
    raise OSError("fcntl.ioctl is not available on this platform (Windows) -- "
                  "stubbed by backend/_ee_windows_compat.py")


_install_stub_if_missing("fcntl", {"ioctl": _stub_ioctl})
_install_stub_if_missing("termios", {"TIOCGWINSZ": 0})
