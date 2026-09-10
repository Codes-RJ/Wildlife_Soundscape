from __future__ import annotations
import math
from dataclasses import dataclass
from .validation import (
    _require_finite,
    _require_positive_int,
    UINT16_MAX,
    UINT32_MAX,
    INT16_MAX,
)


@dataclass(
    frozen=True,
    slots=True,
)
class NetworkConfig:
    processing_queue_blocks: int = 512

    """
    TCP receiver/network settings used by the laptop server.
    """

    host: str = "0.0.0.0"

    port: int = 5001

    read_timeout_s: float = 10.0

    hello_timeout_s: float = 10.0

    max_payload_bytes: int = 64 * 1024

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate laptop TCP receiver settings.
        """

        _require_positive_int(
            self.processing_queue_blocks, name="processing_queue_blocks"
        )
        if not isinstance(
            self.host,
            str,
        ):
            raise TypeError("Network host must be a string.")

        if not (self.host.strip()):
            raise ValueError("Network host cannot be empty.")

        if isinstance(
            self.port,
            bool,
        ) or not isinstance(
            self.port,
            int,
        ):
            raise TypeError("Network port must be an integer.")

        if not (1 <= self.port <= 65_535):
            raise ValueError(("Network port must be between 1 and 65535."))

        read_timeout = _require_finite(
            self.read_timeout_s,
            name="Network read_timeout_s",
        )

        if read_timeout <= 0:
            raise ValueError(("Network read_timeout_s must be greater than 0."))

        hello_timeout = _require_finite(
            self.hello_timeout_s,
            name="Network hello_timeout_s",
        )

        if hello_timeout <= 0:
            raise ValueError(("Network hello_timeout_s must be greater than 0."))

        _require_positive_int(
            self.max_payload_bytes,
            name="Network max_payload_bytes",
        )

        if self.max_payload_bytes > UINT32_MAX:
            raise ValueError(
                (
                    "Network max_payload_bytes "
                    "cannot exceed Protocol-v4 "
                    "uint32 payloadLength capacity."
                )
            )


@dataclass(
    frozen=True,
    slots=True,
)
class AudioConfig:
    """
    Core acoustic acquisition configuration.

    These values form part of the ESP32/laptop binary contract and must
    remain consistent with all three firmware nodes.
    """

    # ------------------------------------------------------------------
    # ACQUISITION
    # ------------------------------------------------------------------

    sample_rate: int = 48_000

    frames_per_block: int = 1024

    channels: int = 1

    sample_width_bytes: int = 2

    # ------------------------------------------------------------------
    # LAPTOP STREAM BUFFER
    # ------------------------------------------------------------------

    buffer_seconds: float = 20.0

    # Maximum expected coarse sampleIndex difference during diagnostic
    # stream alignment.
    sync_tolerance_samples: int = 10

    # ------------------------------------------------------------------
    # CONTINUOUS SESSION RECORDING
    # ------------------------------------------------------------------

    record_wav: bool = True

    # ==================================================================
    # DERIVED AUDIO VALUES
    # ==================================================================

    @property
    def block_duration_s(
        self,
    ) -> float:
        """
        Duration represented by one AUDIO packet.
        """

        return self.frames_per_block / self.sample_rate

    @property
    def bytes_per_block(
        self,
    ) -> int:
        """
        AUDIO payload size for one Protocol-v4 PCM block.
        """

        return self.frames_per_block * self.channels * self.sample_width_bytes

    @property
    def buffer_samples(
        self,
    ) -> int:
        """
        Approximate number of samples represented by the configured
        laptop stream retention period.
        """

        return int(math.ceil(self.buffer_seconds * self.sample_rate))

    @property
    def blocks_in_buffer(
        self,
    ) -> int:
        """
        Number of complete AUDIO blocks retained per node.

        Two additional blocks provide a small operational margin around
        the requested time duration.
        """

        required = int(
            math.ceil(self.buffer_seconds * self.sample_rate / self.frames_per_block)
        )

        return max(
            8,
            required + 2,
        )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate acquisition values against Protocol v4.
        """

        _require_positive_int(
            self.sample_rate,
            name="Audio sample_rate",
        )

        if self.sample_rate > UINT32_MAX:
            raise ValueError(("Audio sample_rate exceeds HELLO uint32 capacity."))

        _require_positive_int(
            self.frames_per_block,
            name="Audio frames_per_block",
        )

        if self.frames_per_block > UINT16_MAX:
            raise ValueError(("Audio frames_per_block exceeds HELLO uint16 capacity."))

        if self.channels != 1:
            raise ValueError(
                (
                    "Current Wildlife Soundscape "
                    "Protocol-v4 transport requires "
                    "mono audio from each ESP32 node."
                )
            )

        if self.sample_width_bytes != 2:
            raise ValueError(
                (
                    "Current Protocol-v4 AUDIO "
                    "transport requires PCM16 "
                    "(2 bytes per sample)."
                )
            )

        buffer_seconds = _require_finite(
            self.buffer_seconds,
            name="Audio buffer_seconds",
        )

        if buffer_seconds <= 0:
            raise ValueError(("Audio buffer_seconds must be greater than 0."))

        if isinstance(
            self.sync_tolerance_samples,
            bool,
        ) or not isinstance(
            self.sync_tolerance_samples,
            int,
        ):
            raise TypeError(("Audio sync_tolerance_samples must be an integer."))

        if self.sync_tolerance_samples < 0:
            raise ValueError(("Audio sync_tolerance_samples cannot be negative."))

        if self.sync_tolerance_samples > INT16_MAX:
            raise ValueError(
                ("Audio sync_tolerance_samples exceeds HELLO int16 capacity.")
            )

        if not isinstance(
            self.record_wav,
            bool,
        ):
            raise TypeError("Audio record_wav must be bool.")
