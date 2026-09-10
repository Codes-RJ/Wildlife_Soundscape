"""Public configuration facade; subsystem implementations live in settings/."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from .settings.validation import _require_finite, UINT8_MAX
from .settings.acquisition import NetworkConfig as NetworkConfig
from .settings.acquisition import AudioConfig as AudioConfig
from .settings.localization import TDOACalibrationConfig as TDOACalibrationConfig
from .settings.localization import LocalizationConfig as LocalizationConfig
from .settings.detection import EventDetectionConfig as EventDetectionConfig
from .settings.detection import DSPConfig as DSPConfig
from .settings.classification import ClassificationConfig as ClassificationConfig
from .settings.research import PersistenceConfig as PersistenceConfig
from .settings.research import AnalyticsConfig as AnalyticsConfig
from .settings.research import DashboardConfig as DashboardConfig


@dataclass(
    frozen=True,
    slots=True,
)
class AppConfig:
    """
    Root configuration object for the complete laptop-side system.

    AppConfig performs validation that spans multiple subsystems.
    """

    network: NetworkConfig = field(default_factory=NetworkConfig)

    audio: AudioConfig = field(default_factory=AudioConfig)

    localization: LocalizationConfig = field(default_factory=LocalizationConfig)

    detection: EventDetectionConfig = field(default_factory=EventDetectionConfig)

    dsp: DSPConfig = field(default_factory=DSPConfig)

    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)

    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # ------------------------------------------------------------------
    # EXPECTED NODES
    # ------------------------------------------------------------------

    expected_nodes: frozenset[int] = frozenset(
        {
            1,
            2,
            3,
        }
    )

    # ------------------------------------------------------------------
    # CONTINUOUS SESSION RECORDINGS
    # ------------------------------------------------------------------

    recordings_dir: Path = Path("data/recordings")

    # ------------------------------------------------------------------
    # CLI STATUS
    # ------------------------------------------------------------------

    print_status_every_s: float = 10.0

    # ==================================================================
    # CROSS-CONFIG VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate the complete system configuration.
        """

        # ==============================================================
        # INDIVIDUAL SECTIONS
        # ==============================================================

        self.network.validate()

        self.audio.validate()

        self.dsp.validate(self.audio.sample_rate)

        self.classification.validate()

        self.persistence.validate()

        self.analytics.validate()

        self.dashboard.validate()

        # ==============================================================
        # EXPECTED NODE SET
        # ==============================================================

        if not isinstance(
            self.expected_nodes,
            frozenset,
        ):
            raise TypeError(("expected_nodes must be a frozenset of node IDs."))

        if len(self.expected_nodes) < 3:
            raise ValueError(
                (
                    "At least three acoustic nodes "
                    "are required for the current "
                    "2-D TDOA localization design."
                )
            )

        for node_id in self.expected_nodes:
            if isinstance(
                node_id,
                bool,
            ):
                raise TypeError(("Expected node IDs must be integers."))

            if not isinstance(
                node_id,
                int,
            ):
                raise TypeError(("Expected node IDs must be integers."))

            if not (1 <= node_id <= UINT8_MAX):
                raise ValueError(
                    (
                        "Expected node IDs must "
                        "fit Protocol-v4 uint8 "
                        "and lie between 1 and 255."
                    )
                )

        # ==============================================================
        # LOCALIZATION + CALIBRATION
        # ==============================================================

        self.localization.validate(
            sample_rate=self.audio.sample_rate,
            expected_nodes=self.expected_nodes,
        )

        # ==============================================================
        # EVENT DETECTOR
        # ==============================================================

        self.detection.validate(expected_node_count=len(self.expected_nodes))

        # ==============================================================
        # ENVIRONMENTAL ANALYTICS NODE
        # ==============================================================

        if (
            self.analytics.environmental_node_id is not None
            and self.analytics.environmental_node_id not in self.expected_nodes
        ):
            raise ValueError(
                ("Analytics environmental_node_id must be one of expected_nodes.")
            )

        # ==============================================================
        # AUDIO PAYLOAD vs NETWORK PAYLOAD LIMIT
        # ==============================================================

        if self.audio.bytes_per_block > self.network.max_payload_bytes:
            raise ValueError(
                (
                    "Network max_payload_bytes "
                    "is too small for one AUDIO "
                    "packet payload. "
                    f"Audio block requires "
                    f"{self.audio.bytes_per_block} bytes, "
                    "but network allows only "
                    f"{self.network.max_payload_bytes}."
                )
            )

        # ==============================================================
        # COARSE ALIGNMENT CONTRACT
        # ==============================================================

        if (
            self.localization.max_alignment_search_samples
            > self.audio.sync_tolerance_samples
        ):
            raise ValueError(
                (
                    "Localization "
                    "max_alignment_search_samples "
                    "cannot exceed Audio "
                    "sync_tolerance_samples."
                )
            )

        # ==============================================================
        # EVENT BUFFER RETENTION
        # ==============================================================

        detector_release_s = self.detection.release_blocks * self.audio.block_duration_s

        required_event_retention_s = (
            self.detection.pre_pad_s
            + self.detection.max_event_s
            + self.detection.post_pad_s
            + detector_release_s
            + (2.0 * self.audio.block_duration_s)
        )

        if self.audio.buffer_seconds < required_event_retention_s:
            raise ValueError(
                (
                    "Audio buffer_seconds is too short "
                    "for the configured maximum event "
                    "retention window. "
                    f"Configured={self.audio.buffer_seconds:.3f}s, "
                    f"required>={required_event_retention_s:.3f}s."
                )
            )

        # ==============================================================
        # LOCALIZATION WINDOW vs STREAM BUFFER
        # ==============================================================

        localization_window_s = (
            self.localization.window_samples / self.audio.sample_rate
        )

        if localization_window_s > self.audio.buffer_seconds:
            raise ValueError(
                ("Localization window is longer than the retained audio buffer.")
            )

        # ==============================================================
        # DSP EVENT MINIMUM vs DETECTOR MINIMUM
        # ==============================================================

        minimum_detected_samples = int(
            math.ceil(self.detection.min_event_ms / 1000.0 * self.audio.sample_rate)
        )

        if self.dsp.minimum_event_samples > minimum_detected_samples:
            raise ValueError(
                (
                    "DSP minimum_event_samples exceeds "
                    "the minimum event duration accepted "
                    "by the detector. This could cause "
                    "valid detected events to be rejected "
                    "by DSP."
                )
            )

        # ==============================================================
        # RECORDING PATH
        # ==============================================================

        if not isinstance(
            self.recordings_dir,
            Path,
        ):
            raise TypeError(("recordings_dir must be pathlib.Path."))

        # ==============================================================
        # STATUS INTERVAL
        # ==============================================================

        status_interval = _require_finite(
            self.print_status_every_s,
            name="print_status_every_s",
        )

        if status_interval <= 0:
            raise ValueError(("print_status_every_s must be greater than 0."))


# ======================================================================
# GLOBAL APPLICATION CONFIGURATION
# ======================================================================


def _load_config() -> AppConfig:
    """Load safe defaults, then optional local deployment overrides."""

    from .deployment import load_app_config

    return load_app_config(AppConfig())


CONFIG = _load_config()
