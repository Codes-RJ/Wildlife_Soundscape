"""Continuous, session-aware ecoacoustic soundscape analysis.

The service consumes sample-indexed mono audio chunks, forms contiguous
per-node windows, calculates ecoacoustic indices, and optionally persists the
result. Missing samples are never hidden by joining data across a gap, and
audio from different acquisition sessions is never combined.
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
    """Buffer contiguous streaming audio and persist complete index windows."""

    def __init__(
        self,
        *,
        config: SoundscapeIndicesConfig | None = None,
        window_duration_seconds: float = 60.0,
        database: Any | None = None,
    ) -> None:
        self.config = config if config is not None else SoundscapeIndicesConfig()

        try:
            duration = float(window_duration_seconds)
        except (TypeError, ValueError) as exc:
            raise TypeError("window_duration_seconds must be numeric.") from exc

        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("window_duration_seconds must be finite and positive.")

        self.window_duration_seconds = duration
        self.window_samples = int(round(duration * self.config.sample_rate))
        if self.window_samples <= 0:
            raise ValueError("window_duration_seconds produces an empty window.")

        self.database = database
        self.active_session_id: int | None = None

        self._buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        self._buffered_samples: dict[int, int] = defaultdict(int)
        self._buffer_start_samples: dict[int, int] = {}
        self._next_samples: dict[int, int] = {}

    @staticmethod
    def _positive_integer(value: Any, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        return value

    @staticmethod
    def _sample_index(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("start_sample must be an integer.")
        if value < 0:
            raise ValueError("start_sample cannot be negative.")
        return value

    def start_session(self, session_id: int) -> None:
        """Begin a new acquisition session with empty per-node buffers."""
        session_id = self._positive_integer(session_id, name="session_id")
        self.reset_buffers()
        self.active_session_id = session_id

    def stop_session(self, session_id: int | None = None) -> None:
        """Discard incomplete windows and end the active session."""
        if session_id is not None:
            session_id = self._positive_integer(session_id, name="session_id")
            if self.active_session_id not in (None, session_id):
                raise ValueError("session_id does not match the active soundscape session.")

        self.reset_buffers()
        self.active_session_id = None

    def _reset_node(self, node_id: int, *, start_sample: int) -> None:
        self._buffers[node_id] = []
        self._buffered_samples[node_id] = 0
        self._buffer_start_samples[node_id] = start_sample
        self._next_samples[node_id] = start_sample

    def append_audio(
        self,
        node_id: int,
        audio_chunk: np.ndarray,
        *,
        session_id: int,
        start_sample: int,
    ) -> SoundscapeIndicesResult | None:
        """Append one sample-indexed chunk and process one complete window.

        A gap resets only the affected node because inventing silence would
        change the indices. An overlapping or duplicate prefix is trimmed so
        retransmitted samples are not counted twice.
        """
        node_id = self._positive_integer(node_id, name="node_id")
        session_id = self._positive_integer(session_id, name="session_id")
        start_sample = self._sample_index(start_sample)

        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError("audio_chunk must be a numpy.ndarray.")
        if audio_chunk.ndim != 1:
            raise ValueError("audio_chunk must be one-dimensional.")
        if audio_chunk.size == 0:
            return None

        # Standalone callers may omit start_session(). A session change still
        # clears every node so samples can never cross that boundary.
        if self.active_session_id != session_id:
            self.start_session(session_id)

        chunk = np.ascontiguousarray(audio_chunk).copy()

        if node_id not in self._next_samples:
            self._reset_node(node_id, start_sample=start_sample)

        expected_sample = self._next_samples[node_id]

        if start_sample > expected_sample:
            logger.warning(
                "Soundscape gap; resetting node buffer | node=%d expected=%d received=%d",
                node_id,
                expected_sample,
                start_sample,
            )
            self._reset_node(node_id, start_sample=start_sample)
            expected_sample = start_sample

        elif start_sample < expected_sample:
            overlap = expected_sample - start_sample
            if overlap >= chunk.size:
                return None
            chunk = chunk[overlap:].copy()
            start_sample = expected_sample

        self._buffers[node_id].append(chunk)
        self._buffered_samples[node_id] += int(chunk.size)
        self._next_samples[node_id] = start_sample + int(chunk.size)

        if self._buffered_samples[node_id] < self.window_samples:
            return None

        full_audio = np.concatenate(self._buffers[node_id])
        window_audio = full_audio[: self.window_samples]
        remainder = full_audio[self.window_samples :]

        window_start = self._buffer_start_samples[node_id]
        window_end = window_start + self.window_samples

        self._buffers[node_id] = [remainder] if remainder.size else []
        self._buffered_samples[node_id] = int(remainder.size)
        self._buffer_start_samples[node_id] = window_end

        result = calculate_soundscape_indices(
            window_audio,
            sample_rate=self.config.sample_rate,
            config=self.config,
        )

        if self.database is not None and hasattr(self.database, "add_soundscape_indices"):
            try:
                self.database.add_soundscape_indices(
                    session_id=session_id,
                    node_id=node_id,
                    start_sample=window_start,
                    end_sample=window_end,
                    result=result,
                )
            except Exception:
                logger.exception(
                    "Failed to persist soundscape indices | node=%d session=0x%08X",
                    node_id,
                    session_id,
                )

        return result

    def reset_buffers(self) -> None:
        """Clear all per-node rolling state without changing configuration."""
        self._buffers.clear()
        self._buffered_samples.clear()
        self._buffer_start_samples.clear()
        self._next_samples.clear()
