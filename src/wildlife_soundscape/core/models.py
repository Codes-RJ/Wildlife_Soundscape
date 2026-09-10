from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wildlife_soundscape.core.protocol import (
    EnvironmentPayload,
    HeartbeatPayload,
    HelloPayload,
    SyncPayload,
)


# ======================================================================
# INTEGER LIMITS
# ======================================================================


UINT8_MAX = 0xFF
UINT32_MAX = 0xFFFFFFFF
UINT64_MAX = 0xFFFFFFFFFFFFFFFF


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _require_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a genuine integer value.

    bool is intentionally rejected because bool is a subclass of int in
    Python and should not silently become an identifier/counter.
    """

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        raise TypeError(f"{name} must be an integer")

    if not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer")

    return int(value)


def _require_uint8(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require an unsigned 8-bit integer.
    """

    value = _require_int(
        value,
        name=name,
    )

    if not (0 <= value <= UINT8_MAX):
        raise ValueError(f"{name} must fit uint8")

    return value


def _require_uint32(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require an unsigned 32-bit integer.
    """

    value = _require_int(
        value,
        name=name,
    )

    if not (0 <= value <= UINT32_MAX):
        raise ValueError(f"{name} must fit uint32")

    return value


def _require_uint64(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require an unsigned 64-bit integer.
    """

    value = _require_int(
        value,
        name=name,
    )

    if not (0 <= value <= UINT64_MAX):
        raise ValueError(f"{name} must fit uint64")

    return value


def _require_node_id(
    value: int,
) -> int:
    """
    Validate the uint8 node identifier used by Protocol v4.

    Node zero is reserved/invalid for acoustic nodes.
    """

    value = _require_uint8(
        value,
        name="node_id",
    )

    if value == 0:
        raise ValueError("node_id cannot be zero")

    return value


def _require_non_negative_counter(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a generic non-negative diagnostic counter.
    """

    value = _require_int(
        value,
        name=name,
    )

    if value < 0:
        raise ValueError(f"{name} cannot be negative")

    return value


# ======================================================================
# AUDIO BLOCK
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class AudioBlock:
    """
    One decoded mono PCM16 AUDIO packet.

    Timing semantics
    ----------------
    sample_index
        Shared-clock sample index corresponding to samples[0].

        This is the authoritative coarse audio timeline used across
        nodes.

    local_micros
        ESP32-local diagnostic timestamp only.

        It must not be used for cross-node TDOA synchronization.

    sequence
        Packet/network diagnostic sequence number.

        It is independent from sample_index.
    """

    node_id: int

    sequence: int

    session_id: int

    sample_index: int

    local_micros: int

    i2s_error_count: int

    flags: int

    samples: np.ndarray

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate the decoded audio block without performing unnecessary
        PCM copies.
        """

        _require_node_id(self.node_id)

        _require_uint32(
            self.sequence,
            name="sequence",
        )

        _require_uint32(
            self.session_id,
            name="session_id",
        )

        _require_uint64(
            self.sample_index,
            name="sample_index",
        )

        _require_uint32(
            self.local_micros,
            name="local_micros",
        )

        _require_uint32(
            self.i2s_error_count,
            name="i2s_error_count",
        )

        _require_uint8(
            self.flags,
            name="flags",
        )

        # --------------------------------------------------------------
        # PCM ARRAY
        # --------------------------------------------------------------

        if not isinstance(
            self.samples,
            np.ndarray,
        ):
            raise TypeError("samples must be a NumPy ndarray")

        if self.samples.ndim != 1:
            raise ValueError(("AudioBlock samples must be a mono 1-D array"))

        if self.samples.size == 0:
            raise ValueError("AudioBlock samples cannot be empty")

        if self.samples.dtype.kind != "i":
            raise TypeError(
                ("AudioBlock samples must contain signed integer PCM values")
            )

        if self.samples.dtype.itemsize != 2:
            raise TypeError(("AudioBlock samples must be 16-bit PCM"))

    # ==================================================================
    # SAMPLE COUNT
    # ==================================================================

    @property
    def sample_count(
        self,
    ) -> int:
        """
        Number of PCM samples in this packet.
        """

        return int(self.samples.size)

    # ==================================================================
    # END SAMPLE
    # ==================================================================

    @property
    def end_sample(
        self,
    ) -> int:
        """
        Exclusive sample index immediately after this audio block.

        Example
        -------
        sample_index = 2048
        samples.size = 1024

        end_sample = 3072
        """

        return int(self.sample_index) + self.sample_count


# ======================================================================
# ENVIRONMENT SAMPLE
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class EnvironmentSample:
    """
    Environmental telemetry associated with the acoustic sample
    timeline.

    session_id is essential because sample_index restarts from zero on
    every acquisition session.
    """

    node_id: int

    session_id: int

    sample_index: int

    value: EnvironmentPayload

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:

        _require_node_id(self.node_id)

        _require_uint32(
            self.session_id,
            name="session_id",
        )

        _require_uint64(
            self.sample_index,
            name="sample_index",
        )

        if not isinstance(
            self.value,
            EnvironmentPayload,
        ):
            raise TypeError(("value must be an EnvironmentPayload"))


# ======================================================================
# NODE SNAPSHOT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class NodeSnapshot:
    """
    Read-only status snapshot of one ESP32 node.

    This structure is intended for:

        CLI status
        future dashboard
        diagnostics
        logging
        monitoring

    It contains state copied from NodeState at a particular moment and
    does not itself manage node lifecycle.
    """

    # ------------------------------------------------------------------
    # IDENTITY / CONNECTION
    # ------------------------------------------------------------------

    node_id: int

    connected: bool

    peer: str | None

    hello: HelloPayload | None

    # ------------------------------------------------------------------
    # STREAM STATE
    # ------------------------------------------------------------------

    session_id: int | None

    last_sequence: int | None

    last_sample_index: int | None

    # ------------------------------------------------------------------
    # PACKET DIAGNOSTICS
    # ------------------------------------------------------------------

    packets_received: int

    audio_packets_received: int

    crc_errors: int

    protocol_errors: int

    sequence_gaps: int

    sequence_resets: int

    sample_gaps: int

    duplicate_or_old_packets: int

    # ------------------------------------------------------------------
    # HEALTH FLAGS
    # ------------------------------------------------------------------

    clipped_packets: int

    clock_fault_packets: int

    congested_packets: int

    # ------------------------------------------------------------------
    # LATEST TELEMETRY
    # ------------------------------------------------------------------

    latest_environment: EnvironmentPayload | None

    latest_heartbeat: HeartbeatPayload | None

    latest_sync: SyncPayload | None

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:

        _require_node_id(self.node_id)

        if not isinstance(
            self.connected,
            bool,
        ):
            raise TypeError("connected must be bool")

        if self.peer is not None and not isinstance(
            self.peer,
            str,
        ):
            raise TypeError("peer must be str or None")

        if self.hello is not None and not isinstance(
            self.hello,
            HelloPayload,
        ):
            raise TypeError("hello must be HelloPayload or None")

        # --------------------------------------------------------------
        # OPTIONAL PROTOCOL COUNTERS
        # --------------------------------------------------------------

        if self.session_id is not None:
            _require_uint32(
                self.session_id,
                name="session_id",
            )

        if self.last_sequence is not None:
            _require_uint32(
                self.last_sequence,
                name="last_sequence",
            )

        if self.last_sample_index is not None:
            _require_uint64(
                self.last_sample_index,
                name="last_sample_index",
            )

        # --------------------------------------------------------------
        # DIAGNOSTIC COUNTERS
        # --------------------------------------------------------------

        counter_fields = (
            (
                "packets_received",
                self.packets_received,
            ),
            (
                "audio_packets_received",
                self.audio_packets_received,
            ),
            (
                "crc_errors",
                self.crc_errors,
            ),
            (
                "protocol_errors",
                self.protocol_errors,
            ),
            (
                "sequence_gaps",
                self.sequence_gaps,
            ),
            (
                "sequence_resets",
                self.sequence_resets,
            ),
            (
                "sample_gaps",
                self.sample_gaps,
            ),
            (
                "duplicate_or_old_packets",
                self.duplicate_or_old_packets,
            ),
            (
                "clipped_packets",
                self.clipped_packets,
            ),
            (
                "clock_fault_packets",
                self.clock_fault_packets,
            ),
            (
                "congested_packets",
                self.congested_packets,
            ),
        )

        for (
            field_name,
            field_value,
        ) in counter_fields:
            _require_non_negative_counter(
                field_value,
                name=field_name,
            )

        # --------------------------------------------------------------
        # TELEMETRY TYPES
        # --------------------------------------------------------------

        if self.latest_environment is not None and not isinstance(
            self.latest_environment,
            EnvironmentPayload,
        ):
            raise TypeError(("latest_environment must be EnvironmentPayload or None"))

        if self.latest_heartbeat is not None and not isinstance(
            self.latest_heartbeat,
            HeartbeatPayload,
        ):
            raise TypeError(("latest_heartbeat must be HeartbeatPayload or None"))

        if self.latest_sync is not None and not isinstance(
            self.latest_sync,
            SyncPayload,
        ):
            raise TypeError(("latest_sync must be SyncPayload or None"))

    # ==================================================================
    # HEALTH SUMMARY
    # ==================================================================

    @property
    def has_clock_faults(
        self,
    ) -> bool:
        """
        Whether at least one CLOCK_FAULT packet has been observed.
        """

        return self.clock_fault_packets > 0

    @property
    def has_congestion(
        self,
    ) -> bool:
        """
        Whether at least one QUEUE_CONGESTED packet has been observed.
        """

        return self.congested_packets > 0

    @property
    def has_clipping(
        self,
    ) -> bool:
        """
        Whether at least one clipped packet has been observed.
        """

        return self.clipped_packets > 0
