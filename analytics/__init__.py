"""
Research analytics package.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

This package provides higher-level research analytics over persisted
acoustic events, classifications, environmental measurements,
localization results, and continuous soundscape metrics.
"""

from __future__ import annotations

from .indices import (
    SoundscapeIndicesConfig,
    SoundscapeIndicesResult,
    calculate_aci,
    calculate_acoustic_entropy,
    calculate_bioacoustic_index,
    calculate_ndsi,
    calculate_soundscape_indices,
)
from .soundscape_service import SoundscapeService

__all__ = (
    "SoundscapeIndicesConfig",
    "SoundscapeIndicesResult",
    "SoundscapeService",
    "calculate_aci",
    "calculate_acoustic_entropy",
    "calculate_bioacoustic_index",
    "calculate_ndsi",
    "calculate_soundscape_indices",
)