"""Public package facade for the Wildlife Soundscape research system."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wildlife-soundscape")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
