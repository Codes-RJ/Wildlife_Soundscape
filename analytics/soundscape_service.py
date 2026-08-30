"""
Continuous soundscape monitoring service.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Manage laptop-side rolling audio buffers per node, periodically compute
continuous ecoacoustic soundscape indices (ACI, NDSI, Acoustic Entropy,
Bioacoustic Index), and persist the results for long-term soundscape analysis.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from .indices import (
    SoundscapeIndicesConfig,
    SoundscapeIndicesResult,
    calculate_soundscape_indices,
)

logger = logging.getLogger(__name__)


class SoundscapeService:
    """
    Stateful service buffering streaming audio and computing continuous indices.
    """

    def __init__(
        self,
        *,
        config: SoundscapeIndicesConfig | None = None,
        window_duration_seconds: float = 60.0,
        database: Any | None = None,
    ) -> None:
        """
        Initialize the soundscape service.

        Parameters
        ----------
        config:
            Ecoacoustic index configuration.
        window_duration_seconds:
            Duration of rolling analysis window (default: 60.0 s).
        database:
            Optional EventDatabase instance to persist index records.
        """
        self.config = config if config is not None else SoundscapeIndicesConfig()
        if window_duration_seconds <= 0:
            raise ValueError("window_duration_seconds must be positive.")
        self.window_duration_seconds = window_duration_seconds
        self.window_samples = int(round(self.window_duration_seconds * self.config.sample_rate))
        self.database = database

        # Buffer per node: node_id -> list of audio segments
        self._buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        self._buffered_samples: dict[int, int] = defaultdict(int)

    def append_audio(
        self,
        node_id: int,
        audio_chunk: np.ndarray,
        *,
        session_id: int,
        start_sample: int,
    ) -> SoundscapeIndicesResult | None:
        """
        Append audio from a node. If a complete window is accumulated,
        compute indices, optionally persist them, and slide/reset the buffer.
        """
        if audio_chunk.ndim != 1 or audio_chunk.size == 0:
            return None

        self._buffers[node_id].append(audio_chunk)
        self._buffered_samples[node_id] += audio_chunk.size

        if self._buffered_samples[node_id] >= self.window_samples:
            # Concatenate accumulated audio
            full_audio = np.concatenate(self._buffers[node_id])
            window_audio = full_audio[:self.window_samples]

            # Retain remainder
            remainder = full_audio[self.window_samples:]
            self._buffers[node_id] = [remainder] if remainder.size > 0 else []
            self._buffered_samples[node_id] = remainder.size

            # Compute indices
            result = calculate_soundscape_indices(
                window_audio,
                sample_rate=self.config.sample_rate,
                config=self.config,
            )

            # Persist if database is configured
            if self.database is not None and hasattr(self.database, "add_soundscape_indices"):
                try:
                    end_sample = start_sample + self.window_samples
                    self.database.add_soundscape_indices(
                        session_id=session_id,
                        node_id=node_id,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        result=result,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to persist soundscape indices for node %s: %s",
                        node_id,
                        exc,
                    )

            return result

        return None

    def reset_buffers(self) -> None:
        """
        Clear all rolling audio buffers.
        """
        self._buffers.clear()
        self._buffered_samples.clear()
