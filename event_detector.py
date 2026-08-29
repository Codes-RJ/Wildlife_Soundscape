from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque

import numpy as np

from config import AudioConfig, EventDetectionConfig
from models import AudioBlock


@dataclass(frozen=True, slots=True)
class BlockDetection:
    node_id: int
    sample_index: int
    end_sample: int
    rms_dbfs: float
    noise_floor_dbfs: float
    threshold_dbfs: float
    spectral_flux: float
    active: bool


@dataclass(frozen=True, slots=True)
class AcousticEvent:
    event_id: int
    session_id: int
    start_sample: int
    end_sample: int
    trigger_nodes: tuple[int, ...]
    peak_rms_dbfs: float

    @property
    def sample_count(self) -> int:
        return max(0, self.end_sample - self.start_sample)


class NodeEventDetector:
    """Adaptive block-level detector for one node.

    It combines RMS energy above a rolling low-quantile noise floor with
    normalized spectral flux. A very strong energy excursion can bypass the
    flux gate so short impulses are not missed.
    """

    def __init__(self, node_id: int, config: EventDetectionConfig) -> None:
        self.node_id = node_id
        self.config = config
        self._noise: Deque[float] = deque(maxlen=config.noise_history_blocks)
        self._previous_spectrum: np.ndarray | None = None
        self._active = False
        self._attack_count = 0
        self._release_count = 0

    @staticmethod
    def _rms_dbfs(samples: np.ndarray) -> float:
        x = np.asarray(samples, dtype=np.float64)
        if x.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(x * x)))
        if rms <= 1e-12:
            return -120.0
        return 20.0 * math.log10(rms / 32768.0)

    def _spectral_flux(self, samples: np.ndarray) -> float:
        x = np.asarray(samples, dtype=np.float64)
        if x.size < 16:
            return 0.0
        x = x - np.mean(x)
        window = np.hanning(x.size)
        spectrum = np.abs(np.fft.rfft(x * window))
        norm = float(np.linalg.norm(spectrum))
        if norm > 1e-12:
            spectrum /= norm
        if self._previous_spectrum is None or self._previous_spectrum.size != spectrum.size:
            flux = 0.0
        else:
            diff = spectrum - self._previous_spectrum
            flux = float(np.sqrt(np.mean(np.square(np.maximum(diff, 0.0)))))
        self._previous_spectrum = spectrum
        return flux

    def process(self, block: AudioBlock) -> BlockDetection:
        rms_dbfs = self._rms_dbfs(block.samples)
        flux = self._spectral_flux(block.samples)

        if self._noise:
            noise_floor = float(np.quantile(np.asarray(self._noise), self.config.noise_quantile))
        else:
            noise_floor = self.config.initial_noise_dbfs

        trigger = noise_floor + self.config.trigger_margin_db
        release = noise_floor + self.config.release_margin_db
        strong_trigger = noise_floor + self.config.strong_energy_margin_db

        candidate = (
            (rms_dbfs >= trigger and flux >= self.config.min_spectral_flux)
            or rms_dbfs >= strong_trigger
        )

        if self._active:
            if rms_dbfs < release:
                self._release_count += 1
            else:
                self._release_count = 0
            if self._release_count >= self.config.release_blocks:
                self._active = False
                self._release_count = 0
                self._attack_count = 0
        else:
            if candidate:
                self._attack_count += 1
            else:
                self._attack_count = 0
            if self._attack_count >= self.config.attack_blocks:
                self._active = True
                self._attack_count = 0

        # Only update the background estimator when the node is not currently
        # active; otherwise the event itself would raise the estimated floor.
        if not self._active:
            self._noise.append(rms_dbfs)

        return BlockDetection(
            node_id=block.node_id,
            sample_index=block.sample_index,
            end_sample=block.end_sample,
            rms_dbfs=rms_dbfs,
            noise_floor_dbfs=noise_floor,
            threshold_dbfs=trigger,
            spectral_flux=flux,
            active=self._active,
        )


class MultiNodeEventDetector:
    """Associates node-level decisions on the common sample timeline."""

    def __init__(
        self,
        audio: AudioConfig,
        config: EventDetectionConfig,
        node_ids: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        self.audio = audio
        self.config = config
        self.node_ids = tuple(node_ids)
        self.detectors = {n: NodeEventDetector(n, config) for n in self.node_ids}
        self._pending: dict[int, dict[int, BlockDetection]] = {}
        self._active_event: dict[str, object] | None = None
        self._next_event_id = 1

    @property
    def pre_pad_samples(self) -> int:
        return int(round(self.config.pre_pad_s * self.audio.sample_rate))

    @property
    def post_pad_samples(self) -> int:
        return int(round(self.config.post_pad_s * self.audio.sample_rate))

    @property
    def min_event_samples(self) -> int:
        return int(round(self.config.min_event_ms / 1000.0 * self.audio.sample_rate))

    @property
    def max_event_samples(self) -> int:
        return int(round(self.config.max_event_s * self.audio.sample_rate))

    def reset(self) -> None:
        self._pending.clear()
        self._active_event = None
        self.detectors = {n: NodeEventDetector(n, self.config) for n in self.node_ids}

    def process(self, block: AudioBlock) -> list[AcousticEvent]:
        if not self.config.enabled or block.node_id not in self.detectors:
            return []

        detection = self.detectors[block.node_id].process(block)
        bucket = self._pending.setdefault(block.sample_index, {})
        bucket[block.node_id] = detection

        # Wait for all nodes when possible. This keeps decisions deterministic
        # and is suitable for our synchronized 3-node lab array.
        if not all(node_id in bucket for node_id in self.node_ids):
            self._prune(block.sample_index)
            return []

        decisions = self._pending.pop(block.sample_index)
        return self._evaluate(block.session_id, decisions)

    def _evaluate(
        self,
        session_id: int,
        decisions: dict[int, BlockDetection],
    ) -> list[AcousticEvent]:
        active_nodes = tuple(sorted(n for n, d in decisions.items() if d.active))
        block_start = min(d.sample_index for d in decisions.values())
        block_end = max(d.end_sample for d in decisions.values())
        peak = max(d.rms_dbfs for d in decisions.values())

        if self._active_event is None:
            if len(active_nodes) >= self.config.min_nodes:
                self._active_event = {
                    "session_id": session_id,
                    "start_sample": max(0, block_start - self.pre_pad_samples),
                    "active_start_sample": block_start,
                    "last_active_end": block_end,
                    "trigger_nodes": set(active_nodes),
                    "peak_rms_dbfs": peak,
                }
            return []

        event = self._active_event
        if int(event["session_id"]) != session_id:
            self._active_event = None
            return []

        event["peak_rms_dbfs"] = max(float(event["peak_rms_dbfs"]), peak)
        cast_nodes = event["trigger_nodes"]
        assert isinstance(cast_nodes, set)
        cast_nodes.update(active_nodes)

        if len(active_nodes) >= self.config.min_nodes:
            event["last_active_end"] = block_end

        start = int(event["start_sample"])
        last_active_end = int(event["last_active_end"])
        max_reached = block_end - start >= self.max_event_samples
        released = (
            len(active_nodes) < self.config.min_nodes
            and block_end >= last_active_end + self.post_pad_samples
        )

        if not (max_reached or released):
            return []

        end = min(
            block_end if max_reached else last_active_end + self.post_pad_samples,
            start + self.max_event_samples,
        )
        self._active_event = None

        # Duration criterion applies to the detected active region, excluding
        # pre/post padding.
        active_start = int(event["active_start_sample"])
        active_duration = max(0, last_active_end - active_start)
        if active_duration < self.min_event_samples:
            return []

        result = AcousticEvent(
            event_id=self._next_event_id,
            session_id=session_id,
            start_sample=start,
            end_sample=end,
            trigger_nodes=tuple(sorted(cast_nodes)),
            peak_rms_dbfs=float(event["peak_rms_dbfs"]),
        )
        self._next_event_id += 1
        return [result]

    def _prune(self, latest_sample: int) -> None:
        # Prevent unbounded growth if one node temporarily drops a packet.
        horizon = self.audio.frames_per_block * 8
        stale = [idx for idx in self._pending if latest_sample - idx > horizon]
        for idx in stale:
            self._pending.pop(idx, None)
