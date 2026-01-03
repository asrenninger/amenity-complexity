"""
Amenity Complexity.

Toolkit to:
- load/normalize POI data (Foursquare OSS Places, Overture Places)
- map POIs to spatial units (H3 / polygons / FUAs)
- compute neighborhood & amenity complexity metrics (ECI/PCI style)

This package uses lazy imports so importing `amenity_complexity` is fast and
does not immediately import heavy dependencies until needed.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "__version__",
    "core",
    "geo",
    "io",
    "poi",
]

# Version (works when installed; falls back when running from a source checkout)
try:
    __version__ = version("amenity-complexity")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

_LAZY_SUBMODULES = {"core", "geo", "io", "poi"}


def __getattr__(name: str):
    """Lazy-load selected submodules on first attribute access."""
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache for next time
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | _LAZY_SUBMODULES)
