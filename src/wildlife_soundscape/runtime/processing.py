"""Bounded, ordered processing with explicit overload accounting."""

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    accepted: int = 0
    completed: int = 0
    failed: int = 0
    dropped: int = 0
    dropped_samples: int = 0
    high_watermark: int = 0
    max_latency_seconds: float = 0.0
    max_processing_seconds: float = 0.0


class OrderedProcessor:
    """One worker owns stream writes and DSP; queues contain owned PCM copies.

    Drop newest on overload. Sample indices remain unchanged, so downstream
    gap detection cannot mistake missing audio for a continuous recording.
    Drain must finish before session buffers, recorder, or detector are reset.
    """

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.queue: asyncio.Queue[tuple[Callable[[], None], float]] = asyncio.Queue(
            capacity
        )
        self.metrics = ProcessingMetrics()
        self._task: asyncio.Task[None] | None = None
        self._accepting = True

    def submit(self, operation: Callable[[], None], *, samples: int = 0) -> bool:
        if not self._accepting or self.queue.full():
            self.metrics.dropped += 1
            self.metrics.dropped_samples += samples
            logger.warning("Processing overload: dropped item (%d samples)", samples)
            return False
        self.queue.put_nowait((operation, time.monotonic()))
        self.metrics.accepted += 1
        self.metrics.high_watermark = max(
            self.metrics.high_watermark, self.queue.qsize()
        )
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="wildlife-processing")
        return True

    async def _run(self) -> None:
        # Exit when idle: no background task remains attached to a closed loop.
        while not self.queue.empty():
            operation, queued_at = self.queue.get_nowait()
            started = time.monotonic()
            try:
                await asyncio.to_thread(operation)
                self.metrics.completed += 1
            except Exception:
                self.metrics.failed += 1
                logger.exception("Acquisition processing failed")
            finally:
                finished = time.monotonic()
                self.metrics.max_processing_seconds = max(
                    self.metrics.max_processing_seconds, finished - started
                )
                self.metrics.max_latency_seconds = max(
                    self.metrics.max_latency_seconds, finished - queued_at
                )
                self.queue.task_done()

    async def drain(self) -> None:
        if self._task is not None:
            # Cancellation must not leave a thread mutating reset session state.
            task = self._task
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    async def close(self) -> None:
        self._accepting = False
        await self.drain()

    def snapshot(self) -> dict[str, Any]:
        return {**asdict(self.metrics), "queue_depth": self.queue.qsize()}
