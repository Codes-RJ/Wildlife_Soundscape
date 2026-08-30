from __future__ import annotations

import itertools
import math

from dataclasses import dataclass

import numpy as np

from config import (
    LocalizationConfig,
)

from environment import (
    calculate_speed_of_sound_mps,
)

from stream_manager import (
    StreamManager,
)

from .filtering import (
    bandpass_filter,
)

from .gcc_phat import (
    gcc_phat,
)

from .solver import (
    PositionResult,
    solve_position,
)

from .tdoa import (
    TDOAMeasurement,
    physical_max_delay,
)


# ======================================================================
# CONSTANTS
# ======================================================================


MIN_LOCALIZATION_WINDOW_SAMPLES = (
    64
)


# ======================================================================
# LOCALIZATION RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LocalizationResult:
    """
    Result of one complete multi-node TDOA localization operation.

    position
        Nonlinear position-solver result.

    measurements
        Pairwise TDOA measurements used or rejected by the solver.

    window_start_sample
        Absolute shared-clock sampleIndex at which the localization
        window begins.

    window_samples
        Number of samples in each node waveform.

    speed_of_sound_mps
        Propagation speed actually used by GCC physical constraints and
        the position solver.

    node_rms
        Raw PCM RMS for each participating microphone.

    environment_used
        Environmental tuple:

            (temperature_c, humidity_percent, pressure_hpa)

        when environmental sound-speed correction was successfully used.

        None means the configured/furnished fallback propagation speed
        was used.
    """

    position: PositionResult

    measurements: tuple[
        TDOAMeasurement,
        ...,
    ]

    window_start_sample: int

    window_samples: int

    speed_of_sound_mps: float

    node_rms: dict[
        int,
        float,
    ]

    environment_used: (
        tuple[
            float,
            float,
            float,
        ]
        | None
    ) = None

    # ==================================================================
    # SUCCESS
    # ==================================================================

    @property
    def success(
        self,
    ) -> bool:
        """
        Convenience view of the nonlinear solver success state.
        """

        return bool(
            self.position.success
        )


# ======================================================================
# LOCALIZATION ENGINE
# ======================================================================


class LocalizationEngine:
    """
    Laptop-side acoustic source-localization orchestrator.

    Processing chain
    ----------------

        StreamManager absolute sample windows
                    ↓
            session validation
                    ↓
             per-node raw RMS
                    ↓
        optional zero-phase band-pass
                    ↓
              microphone pairs
                    ↓
          physical delay constraints
                    ↓
                GCC-PHAT
                    ↓
           TDOA measurements
                    ↓
          nonlinear position solve


    Timing authority
    ----------------
    Coarse synchronization comes from the shared-clock sampleIndex.

    GCC-PHAT estimates the remaining waveform/acoustic propagation delay.

    TCP arrival time and ESP32 localMicros are never used for TDOA.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        streams: StreamManager,
        config: LocalizationConfig,
    ) -> None:

        if not isinstance(
            streams,
            StreamManager,
        ):

            raise TypeError(
                (
                    "streams must be a "
                    "StreamManager instance."
                )
            )

        if not isinstance(
            config,
            LocalizationConfig,
        ):

            raise TypeError(
                (
                    "config must be a "
                    "LocalizationConfig instance."
                )
            )

        self.streams = (
            streams
        )

        self.config = (
            config
        )

    # ==================================================================
    # WINDOW LENGTH
    # ==================================================================

    def _resolve_window_length(
        self,
        length: int | None,
    ) -> int:
        """
        Resolve and validate one localization-window length.
        """

        if (
            length
            is None
        ):

            result = int(
                self.config.window_samples
            )

        else:

            if (
                isinstance(
                    length,
                    bool,
                )
                or not isinstance(
                    length,
                    int,
                )
            ):

                raise TypeError(
                    (
                        "localization window length "
                        "must be an integer."
                    )
                )

            result = int(
                length
            )

        if (
            result
            < MIN_LOCALIZATION_WINDOW_SAMPLES
        ):

            raise ValueError(
                (
                    "localization window must "
                    f"contain at least "
                    f"{MIN_LOCALIZATION_WINDOW_SAMPLES} "
                    "samples."
                )
            )

        return result

    # ==================================================================
    # START SAMPLE
    # ==================================================================

    @staticmethod
    def _validate_start_sample(
        start_sample: int,
    ) -> int:
        """
        Validate one non-negative absolute sampleIndex.
        """

        if (
            isinstance(
                start_sample,
                bool,
            )
            or not isinstance(
                start_sample,
                int,
            )
        ):

            raise TypeError(
                (
                    "start_sample must be "
                    "an integer."
                )
            )

        start_sample = int(
            start_sample
        )

        if (
            start_sample
            < 0
        ):

            raise ValueError(
                (
                    "start_sample cannot "
                    "be negative."
                )
            )

        return start_sample

    # ==================================================================
    # ARRAY BOUNDS
    # ==================================================================

    def _bounds(
        self,
    ) -> (
        tuple[
            tuple[
                float,
                float,
            ],
            tuple[
                float,
                float,
            ],
        ]
        | None
    ):
        """
        Build optional position-solver bounds from microphone geometry.
        """

        if not (
            self.config.constrain_to_array_bounds
        ):

            return None

        xs = [
            float(
                xy[0]
            )
            for xy
            in self.config.node_positions.values()
        ]

        ys = [
            float(
                xy[1]
            )
            for xy
            in self.config.node_positions.values()
        ]

        margin = float(
            self.config.bounds_margin_m
        )

        return (
            (
                min(
                    xs
                )
                - margin,

                min(
                    ys
                )
                - margin,
            ),
            (
                max(
                    xs
                )
                + margin,

                max(
                    ys
                )
                + margin,
            ),
        )

    # ==================================================================
    # SESSION CONSISTENCY
    # ==================================================================

    def _common_stream_session(
        self,
    ) -> int | None:
        """
        Return the common non-zero acquisition session represented by all
        configured localization nodes.

        None is returned when:

            a node is not registered
            a node has no active session
            node sessions disagree

        sampleIndex values from different sessions must never be compared
        merely because their numerical values happen to overlap.
        """

        session_ids: set[
            int
        ] = set()

        for node_id in (
            self.config.node_positions
        ):

            state = (
                self.streams.nodes.get(
                    node_id
                )
            )

            if state is None:

                return None

            session_id = (
                state.session_id
            )

            if (
                session_id
                is None
                or session_id
                == 0
            ):

                return None

            session_ids.add(
                int(
                    session_id
                )
            )

        if (
            len(
                session_ids
            )
            != 1
        ):

            return None

        return next(
            iter(
                session_ids
            )
        )

    # ==================================================================
    # SPEED OF SOUND VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_speed_of_sound(
        value: float,
        *,
        name: str,
    ) -> float:
        """
        Require a finite positive propagation speed.
        """

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                (
                    f"{name} must be "
                    "a numeric value."
                )
            ) from exc

        if (
            not math.isfinite(
                value
            )
            or value
            <= 0.0
        ):

            raise ValueError(
                (
                    f"{name} must be finite "
                    "and greater than 0."
                )
            )

        return value

    # ==================================================================
    # SPEED OF SOUND RESOLUTION
    # ==================================================================

    def _resolve_speed(
        self,
        *,
        sample_index: int,
        speed_of_sound_mps: float | None,
    ) -> tuple[
        float,
        tuple[
            float,
            float,
            float,
        ]
        | None,
    ]:
        """
        Resolve the propagation speed used for one localization window.

        Priority
        --------
        1. Explicit caller-provided speed.
        2. Valid environmental estimate when enabled.
        3. Configured fallback speed.

        Invalid environmental telemetry must not make an otherwise valid
        acoustic localization operation fail.
        """

        # ==============================================================
        # EXPLICIT OVERRIDE
        # ==============================================================

        if (
            speed_of_sound_mps
            is not None
        ):

            speed = (
                self._validate_speed_of_sound(
                    speed_of_sound_mps,
                    name=
                        "speed_of_sound_mps",
                )
            )

            return (
                speed,
                None,
            )

        # ==============================================================
        # ENVIRONMENTAL CORRECTION
        # ==============================================================

        if (
            self.config.use_environmental_speed
        ):

            environment = (
                self.streams
                .get_environment_near(
                    sample_index
                )
            )

            if (
                environment
                is not None
            ):

                try:

                    speed = (
                        calculate_speed_of_sound_mps(
                            environment.temperature_c,
                            environment.humidity_percent,
                            environment.pressure_hpa,
                        )
                    )

                    speed = (
                        self._validate_speed_of_sound(
                            speed,
                            name=
                                (
                                    "environmental "
                                    "speed_of_sound_mps"
                                ),
                        )
                    )

                except (
                    ArithmeticError,
                    TypeError,
                    ValueError,
                ):

                    # --------------------------------------------------
                    # Invalid environmental telemetry must not destroy
                    # localization. Fall through to configured speed.
                    # --------------------------------------------------

                    pass

                else:

                    environment_used = (
                        float(
                            environment.temperature_c
                        ),
                        float(
                            environment.humidity_percent
                        ),
                        float(
                            environment.pressure_hpa
                        ),
                    )

                    return (
                        speed,
                        environment_used,
                    )

        # ==============================================================
        # CONFIGURED FALLBACK
        # ==============================================================

        fallback = (
            self._validate_speed_of_sound(
                self.config.speed_of_sound_mps,
                name=
                    (
                        "configured "
                        "speed_of_sound_mps"
                    ),
            )
        )

        return (
            fallback,
            None,
        )

    # ==================================================================
    # LOCALIZE EXPLICIT WINDOW
    # ==================================================================

    def locate_window(
        self,
        *,
        start_sample: int,
        length: int | None = None,
        speed_of_sound_mps: float | None = None,
    ) -> LocalizationResult:
        """
        Localize an acoustic source from one common absolute sample window.
        """

        # ==============================================================
        # WINDOW
        # ==============================================================

        start_sample = (
            self._validate_start_sample(
                start_sample
            )
        )

        window_samples = (
            self._resolve_window_length(
                length
            )
        )

        # ==============================================================
        # SESSION CONSISTENCY
        # ==============================================================

        common_session = (
            self._common_stream_session()
        )

        if (
            common_session
            is None
        ):

            raise RuntimeError(
                (
                    "Localization requires all "
                    "configured nodes to belong "
                    "to the same active non-zero "
                    "acquisition session."
                )
            )

        # ==============================================================
        # ENVIRONMENT LOOKUP POSITION
        # ==============================================================
        #
        # Use the temporal center of the localization window rather than
        # its leading edge when selecting nearby telemetry.
        # ==============================================================

        environment_lookup_sample = (
            start_sample
            + window_samples
            // 2
        )

        # ==============================================================
        # SPEED OF SOUND
        # ==============================================================

        (
            speed_of_sound,
            environment_used,
        ) = (
            self._resolve_speed(
                sample_index=
                    environment_lookup_sample,

                speed_of_sound_mps=
                    speed_of_sound_mps,
            )
        )

        # ==============================================================
        # SAMPLE RATE
        # ==============================================================

        sample_rate = float(
            self.streams
            .audio_config
            .sample_rate
        )

        if (
            not math.isfinite(
                sample_rate
            )
            or sample_rate
            <= 0.0
        ):

            raise ValueError(
                (
                    "StreamManager sample rate "
                    "must be finite and positive."
                )
            )

        # ==============================================================
        # EXTRACT NODE WINDOWS
        # ==============================================================

        windows: dict[
            int,
            np.ndarray,
        ] = {}

        node_rms: dict[
            int,
            float,
        ] = {}

        for node_id in sorted(
            self.config.node_positions
        ):

            # ----------------------------------------------------------
            # RECHECK SESSION IMMEDIATELY BEFORE EXTRACTION
            # ----------------------------------------------------------

            state = (
                self.streams.nodes.get(
                    node_id
                )
            )

            if (
                state is None
                or state.session_id
                != common_session
            ):

                raise RuntimeError(
                    (
                        "Node session changed while "
                        "localization window was "
                        "being assembled."
                    )
                )

            # ----------------------------------------------------------
            # ABSOLUTE PCM WINDOW
            # ----------------------------------------------------------

            raw = (
                self.streams
                .get_window(
                    node_id,
                    start_sample,
                    window_samples,
                )
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            if (
                raw.ndim
                != 1
                or raw.size
                != window_samples
            ):

                raise RuntimeError(
                    (
                        f"node {node_id} returned "
                        "an invalid localization "
                        "window shape."
                    )
                )

            if not np.all(
                np.isfinite(
                    raw
                )
            ):

                raise ValueError(
                    (
                        f"node {node_id} localization "
                        "window contains non-finite "
                        "samples."
                    )
                )

            # ----------------------------------------------------------
            # RAW RMS
            # ----------------------------------------------------------

            if (
                raw.size
                > 0
            ):

                raw_rms = float(
                    np.sqrt(
                        np.mean(
                            raw
                            * raw,
                            dtype=np.float64,
                        )
                    )
                )

            else:

                raw_rms = (
                    0.0
                )

            if not math.isfinite(
                raw_rms
            ):

                raw_rms = (
                    0.0
                )

            node_rms[
                node_id
            ] = (
                raw_rms
            )

            # ----------------------------------------------------------
            # LOCALIZATION CONDITIONING
            # ----------------------------------------------------------

            if (
                self.config.bandpass_enabled
            ):

                conditioned = (
                    bandpass_filter(
                        raw,

                        sample_rate=
                            sample_rate,

                        low_hz=
                            self.config
                            .bandpass_low_hz,

                        high_hz=
                            self.config
                            .bandpass_high_hz,

                        order=
                            self.config
                            .bandpass_order,
                    )
                )

            else:

                conditioned = np.ascontiguousarray(
                    raw,
                    dtype=np.float64,
                )

            if (
                conditioned.ndim
                != 1
                or conditioned.size
                != window_samples
            ):

                raise RuntimeError(
                    (
                        f"node {node_id} conditioned "
                        "localization waveform has "
                        "an invalid shape."
                    )
                )

            windows[
                node_id
            ] = (
                conditioned
            )

        # ==============================================================
        # VERIFY SESSION DID NOT CHANGE DURING EXTRACTION
        # ==============================================================

        if (
            self._common_stream_session()
            != common_session
        ):

            raise RuntimeError(
                (
                    "Acquisition session changed "
                    "during localization."
                )
            )

        # ==============================================================
        # PAIRWISE TDOA
        # ==============================================================

        measurements: list[
            TDOAMeasurement
        ] = []

        for (
            node_a,
            node_b,
        ) in itertools.combinations(
            sorted(
                windows
            ),
            2,
        ):

            # ----------------------------------------------------------
            # PHYSICAL PAIR LIMIT
            # ----------------------------------------------------------

            max_delay_seconds = (
                physical_max_delay(
                    self.config
                    .node_positions[
                        node_a
                    ],

                    self.config
                    .node_positions[
                        node_b
                    ],

                    speed_of_sound_mps=
                        speed_of_sound,
                )
            )

            # ----------------------------------------------------------
            # SIGNAL-ENERGY GATE
            # ----------------------------------------------------------

            if (
                min(
                    node_rms[
                        node_a
                    ],
                    node_rms[
                        node_b
                    ],
                )
                < self.config.min_rms
            ):

                measurements.append(
                    TDOAMeasurement(
                        node_a=
                            node_a,

                        node_b=
                            node_b,

                        delay_seconds=
                            0.0,

                        delay_samples=
                            0.0,

                        peak_ratio=
                            0.0,

                        max_delay_seconds=
                            max_delay_seconds,

                        valid=
                            False,

                        reason=
                            (
                                "insufficient "
                                "signal energy"
                            ),
                    )
                )

                continue

            # ----------------------------------------------------------
            # GCC-PHAT
            # ----------------------------------------------------------
            #
            # signal = B
            # reference = A
            #
            # Therefore:
            #
            #     positive delay
            #         B arrives later than A
            #
            # which exactly matches:
            #
            #     TDOAMeasurement(node_a=A, node_b=B)
            # ----------------------------------------------------------

            gcc_result = (
                gcc_phat(
                    windows[
                        node_b
                    ],

                    windows[
                        node_a
                    ],

                    sample_rate=
                        sample_rate,

                    max_delay_seconds=
                        max_delay_seconds,

                    interpolation=
                        self.config.interpolation,

                    min_peak_ratio=
                        self.config.min_peak_ratio,
                )
            )

            # ----------------------------------------------------------
            # DEFENSIVE PHYSICAL CHECK
            # ----------------------------------------------------------
            #
            # GCC-PHAT already searches only the physically feasible
            # interval. This second check protects the interface if the
            # lower-level implementation changes later.
            # ----------------------------------------------------------

            physical_tolerance = max(
                1e-12,
                max_delay_seconds
                * 1e-9,
            )

            physically_valid = (
                abs(
                    gcc_result.delay_seconds
                )
                <= (
                    max_delay_seconds
                    + physical_tolerance
                )
            )

            valid = bool(
                gcc_result.valid
                and physically_valid
            )

            if not physically_valid:

                reason = (
                    "delay outside physical "
                    "pair limit"
                )

            else:

                reason = (
                    gcc_result.reason
                )

            # ----------------------------------------------------------
            # PEAK RATIO
            # ----------------------------------------------------------

            peak_ratio = float(
                gcc_result.peak_ratio
            )

            if not math.isfinite(
                peak_ratio
            ):

                # TDOAMeasurement deliberately stores finite diagnostic
                # values. A saturated uniqueness ratio is represented by
                # a large finite value rather than infinity.
                peak_ratio = (
                    float(
                        np.finfo(
                            np.float64
                        ).max
                    )
                )

            # ----------------------------------------------------------
            # MEASUREMENT
            # ----------------------------------------------------------

            measurements.append(
                TDOAMeasurement(
                    node_a=
                        node_a,

                    node_b=
                        node_b,

                    delay_seconds=
                        float(
                            gcc_result
                            .delay_seconds
                        ),

                    delay_samples=
                        float(
                            gcc_result
                            .delay_samples
                        ),

                    peak_ratio=
                        peak_ratio,

                    max_delay_seconds=
                        float(
                            max_delay_seconds
                        ),

                    valid=
                        valid,

                    reason=
                        reason,
                )
            )

        # ==============================================================
        # NONLINEAR POSITION SOLVE
        # ==============================================================

        position = (
            solve_position(
                self.config.node_positions,
                measurements,

                speed_of_sound_mps=
                    speed_of_sound,

                bounds=
                    self._bounds(),
            )
        )

        # ==============================================================
        # RESULT
        # ==============================================================

        return LocalizationResult(
            position=
                position,

            measurements=
                tuple(
                    measurements
                ),

            window_start_sample=
                start_sample,

            window_samples=
                window_samples,

            speed_of_sound_mps=
                speed_of_sound,

            node_rms=
                node_rms,

            environment_used=
                environment_used,
        )

    # ==================================================================
    # LOCALIZE NEWEST COMMON WINDOW
    # ==================================================================

    def locate_latest(
        self,
        *,
        length: int | None = None,
    ) -> LocalizationResult | None:
        """
        Localize the newest complete common sample window available from
        all configured nodes.

        None is returned when the required synchronized session/window is
        not yet available.
        """

        window_samples = (
            self._resolve_window_length(
                length
            )
        )

        # ==============================================================
        # COMMON SESSION
        # ==============================================================

        common_session = (
            self._common_stream_session()
        )

        if (
            common_session
            is None
        ):

            return None

        # ==============================================================
        # NEWEST AVAILABLE END PER NODE
        # ==============================================================

        latest_ends: list[
            int
        ] = []

        for node_id in (
            self.config.node_positions
        ):

            state = (
                self.streams.nodes.get(
                    node_id
                )
            )

            if (
                state is None
                or state.session_id
                != common_session
                or not state.audio_blocks
            ):

                return None

            latest_block = (
                state.audio_blocks[
                    -1
                ]
            )

            if (
                latest_block.session_id
                != common_session
            ):

                return None

            latest_ends.append(
                int(
                    latest_block.end_sample
                )
            )

        # ==============================================================
        # COMMON END
        # ==============================================================

        common_end = min(
            latest_ends
        )

        if (
            common_end
            < window_samples
        ):

            return None

        start_sample = (
            common_end
            - window_samples
        )

        # ==============================================================
        # SESSION RECHECK
        # ==============================================================

        if (
            self._common_stream_session()
            != common_session
        ):

            return None

        return (
            self.locate_window(
                start_sample=
                    start_sample,

                length=
                    window_samples,
            )
        )