from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from protocol import EnvironmentPayload, HeartbeatPayload, HelloPayload, SyncPayload


@dataclass(frozen=True, slots=True)
class AudioBlock:
    node_id: int
    sequence: int
    session_id: int
    sample_index: int
    local_micros: int
    i2s_error_count: int
    flags: int
    samples: np.ndarray

    @property
    def end_sample(self) -> int:
        return self.sample_index + int(self.samples.size)


@dataclass(frozen=True, slots=True)
class EnvironmentSample:
    node_id: int
    session_id: int
    sample_index: int
    value: EnvironmentPayload


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    node_id: int
    connected: bool
    peer: str | None
    hello: HelloPayload | None
    session_id: int | None
    last_sequence: int | None
    last_sample_index: int | None
    packets_received: int
    audio_packets_received: int
    crc_errors: int
    protocol_errors: int
    sequence_gaps: int
    sequence_resets: int
    sample_gaps: int
    duplicate_or_old_packets: int
    clipped_packets: int
    clock_fault_packets: int
    congested_packets: int
    latest_environment: EnvironmentPayload | None
    latest_heartbeat: HeartbeatPayload | None
    latest_sync: SyncPayload | None
