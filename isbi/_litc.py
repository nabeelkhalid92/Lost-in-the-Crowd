"""Puts the repository folders on sys.path and exposes the file locations of litc_paths.py."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ("isbi", "paper", "inference", "eval", ""):
    _p = _os.path.join(_ROOT, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
from litc_paths import *  # noqa: F401,F403,E402
