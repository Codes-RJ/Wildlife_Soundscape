from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .validation import _require_finite, _require_positive_int, UINT8_MAX


@dataclass(
    frozen=True,
    slots=True,
)
class PersistenceConfig:
    """
    Database and event-file persistence settings.
    """

    database_path: Path = Path("data/database/events.db")

    events_dir: Path = Path("data/events")

    save_event_wav: bool = True

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate persistence paths.
        """

        if not isinstance(
            self.database_path,
            Path,
        ):
            raise TypeError(("Persistence database_path must be pathlib.Path."))

        if not isinstance(
            self.events_dir,
            Path,
        ):
            raise TypeError(("Persistence events_dir must be pathlib.Path."))

        if not self.database_path.name:
            raise ValueError(
                ("Persistence database_path must include a database filename.")
            )

        if not isinstance(
            self.save_event_wav,
            bool,
        ):
            raise TypeError(("Persistence save_event_wav must be bool."))


@dataclass(
    frozen=True,
    slots=True,
)
class AnalyticsConfig:
    """
    Research-analytics configuration.

    This section controls higher-level analysis over persisted events.

    It does not affect:

        ESP32 acquisition
        protocol timing
        event detection
        DSP
        classification
        TDOA localization

    Therefore analytics parameters can be adjusted for exploratory
    research without changing the underlying recorded observations.
    """

    enabled: bool = True

    # ------------------------------------------------------------------
    # CONTINUOUS SOUNDSCAPE INDICES
    # ------------------------------------------------------------------

    soundscape_enabled: bool = True

    soundscape_window_seconds: float = 60.0

    bucket_seconds: int = 3600

    environmental_node_id: int | None = 1

    environmental_alpha: float = 0.05

    environmental_min_samples: int = 5

    neutral_threshold: float = 0.05

    cell_size_m: float = 0.25

    max_grid_cells: int = 10_000

    max_transition_gap_s: float | None = 300.0

    same_class_transitions_only: bool = False

    min_activity_events: int = 5

    min_localized_events: int = 5

    min_transitions: int = 3

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate research-analytics parameters.
        """

        if not isinstance(
            self.enabled,
            bool,
        ):
            raise TypeError(("Analytics enabled must be bool."))

        if not isinstance(
            self.soundscape_enabled,
            bool,
        ):
            raise TypeError(("Analytics soundscape_enabled must be bool."))

        soundscape_window = _require_finite(
            self.soundscape_window_seconds,
            name=("Analytics soundscape_window_seconds"),
        )

        if soundscape_window <= 0.0:
            raise ValueError(
                ("Analytics soundscape_window_seconds must be greater than 0.")
            )

        _require_positive_int(
            self.bucket_seconds,
            name="Analytics bucket_seconds",
        )

        if self.environmental_node_id is not None:
            _require_positive_int(
                self.environmental_node_id,
                name=("Analytics environmental_node_id"),
            )

            if self.environmental_node_id > UINT8_MAX:
                raise ValueError(
                    (
                        "Analytics "
                        "environmental_node_id "
                        "must fit Protocol-v4 "
                        "uint8 nodeId."
                    )
                )

        alpha = _require_finite(
            self.environmental_alpha,
            name=("Analytics environmental_alpha"),
        )

        if not (0.0 < alpha < 1.0):
            raise ValueError(("Analytics environmental_alpha must be in (0, 1)."))

        _require_positive_int(
            self.environmental_min_samples,
            name=("Analytics environmental_min_samples"),
        )

        if self.environmental_min_samples < 3:
            raise ValueError(
                ("Analytics environmental_min_samples must be at least 3.")
            )

        neutral = _require_finite(
            self.neutral_threshold,
            name=("Analytics neutral_threshold"),
        )

        if not (0.0 <= neutral <= 1.0):
            raise ValueError(("Analytics neutral_threshold must be in [0, 1]."))

        cell_size = _require_finite(
            self.cell_size_m,
            name="Analytics cell_size_m",
        )

        if cell_size <= 0.0:
            raise ValueError(("Analytics cell_size_m must be greater than 0."))

        _require_positive_int(
            self.max_grid_cells,
            name="Analytics max_grid_cells",
        )

        if self.max_transition_gap_s is not None:
            transition_gap = _require_finite(
                self.max_transition_gap_s,
                name=("Analytics max_transition_gap_s"),
            )

            if transition_gap <= 0.0:
                raise ValueError(
                    ("Analytics max_transition_gap_s must be greater than 0 or None.")
                )

        if not isinstance(
            self.same_class_transitions_only,
            bool,
        ):
            raise TypeError(("Analytics same_class_transitions_only must be bool."))

        _require_positive_int(
            self.min_activity_events,
            name=("Analytics min_activity_events"),
        )

        _require_positive_int(
            self.min_localized_events,
            name=("Analytics min_localized_events"),
        )

        _require_positive_int(
            self.min_transitions,
            name=("Analytics min_transitions"),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class DashboardConfig:
    """
    Local research/dashboard presentation configuration.

    These settings affect only visualization and UI data volume.

    They do not alter scientific source data or persisted event records.
    """

    enabled: bool = True

    page_title: str = "Wildlife Soundscape Monitor"

    auto_refresh: bool = True

    refresh_interval_s: float = 2.0

    recent_events_limit: int = 50

    session_list_limit: int = 100

    max_plot_points: int = 5000

    max_audio_preview_s: float = 30.0

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate dashboard/UI parameters.
        """

        if not isinstance(
            self.enabled,
            bool,
        ):
            raise TypeError(("Dashboard enabled must be bool."))

        if not isinstance(
            self.page_title,
            str,
        ):
            raise TypeError(("Dashboard page_title must be a string."))

        if not (self.page_title.strip()):
            raise ValueError(("Dashboard page_title cannot be empty."))

        if not isinstance(
            self.auto_refresh,
            bool,
        ):
            raise TypeError(("Dashboard auto_refresh must be bool."))

        refresh_interval = _require_finite(
            self.refresh_interval_s,
            name=("Dashboard refresh_interval_s"),
        )

        if refresh_interval <= 0.0:
            raise ValueError(("Dashboard refresh_interval_s must be greater than 0."))

        _require_positive_int(
            self.recent_events_limit,
            name=("Dashboard recent_events_limit"),
        )

        _require_positive_int(
            self.session_list_limit,
            name=("Dashboard session_list_limit"),
        )

        _require_positive_int(
            self.max_plot_points,
            name=("Dashboard max_plot_points"),
        )

        max_audio_preview = _require_finite(
            self.max_audio_preview_s,
            name=("Dashboard max_audio_preview_s"),
        )

        if max_audio_preview <= 0.0:
            raise ValueError(("Dashboard max_audio_preview_s must be greater than 0."))
