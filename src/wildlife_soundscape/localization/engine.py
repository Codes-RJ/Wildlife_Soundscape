from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import itertools
import math

from dataclasses import (
    dataclass,
)


# ======================================================================
# THIRD PARTY
# ======================================================================


import numpy as np


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================


from wildlife_soundscape.core.config import (
    LocalizationConfig,
)


# ======================================================================
# ENVIRONMENT
# ======================================================================


from wildlife_soundscape.core.environment import (
    calculate_speed_of_sound_mps,
)


# ======================================================================
# STREAMING
# ======================================================================


from wildlife_soundscape.acquisition.stream_manager import (
    StreamManager,
)


# ======================================================================
# LOCALIZATION COMPONENTS
# ======================================================================


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
        Pairwise TDOA measurements supplied to the position solver.

        IMPORTANT:

        When TDOA calibration is enabled, delay_seconds and
        delay_samples contain the CALIBRATED delays.

        When calibration is disabled, they contain the raw GCC-PHAT
        delays.

    window_start_sample
        Absolute shared-clock sampleIndex at which the localization
        window begins.

    window_samples
        Number of samples in each node waveform.

    speed_of_sound_mps
        Propagation speed actually used by:

            physical delay constraints
            GCC-PHAT search windows
            position solver

    node_rms
        Raw PCM RMS for each participating microphone.

    environment_used
        Environmental tuple:

            (
                temperature_c,
                humidity_percent,
                pressure_hpa,
            )

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
        Convenience view of nonlinear solver success.
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
       geometric physical-delay limit
                    ↓
       optional calibration allowance
                    ↓
                GCC-PHAT
                    ↓
             raw pair TDOA
                    ↓
        optional timing calibration
                    ↓
          corrected pair TDOA
                    ↓
      corrected physical-limit check
                    ↓
           TDOA measurements
                    ↓
          nonlinear position solve


    Timing authority
    ----------------
    Coarse synchronization comes from the shared-clock sampleIndex.

    GPIO27 is a session/start marker only.

    GCC-PHAT estimates waveform/acoustic propagation delay.

    TCP arrival time and ESP32 localMicros are never used for TDOA.


    Calibration model
    -----------------
    When enabled:

        pair_offset(A, B)
            =
        bias_B - bias_A

    and:

        corrected_tdoa
            =
        raw_gcc_tdoa - pair_offset

    Calibration therefore compensates repeatable channel-specific
    timing bias before the geometric solver is invoked.
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

        return (
            result
        )

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

        return (
            start_sample
        )

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

            return (
                None
            )

        xs = [
            float(
                xy[
                    0
                ]
            )
            for xy
            in self.config.node_positions.values()
        ]

        ys = [
            float(
                xy[
                    1
                ]
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

            if (
                state
                is None
            ):

                return (
                    None
                )

            session_id = (
                state.session_id
            )

            if (
                session_id
                is None
                or session_id
                == 0
            ):

                return (
                    None
                )

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

            return (
                None
            )

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

        return (
            value
        )

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
        Resolve propagation speed for one localization window.

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
                    # localization. Fall through to configured fallback.
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
    # CALIBRATION OFFSET
    # ==================================================================

    def _pair_calibration_offset_s(
        self,
        node_a: int,
        node_b: int,
    ) -> float:
        """
        Resolve configured systematic TDOA offset for one pair.

        Convention
        ----------
        Returned value corresponds to:

            arrival_B - arrival_A

        With calibration disabled this method returns exactly zero.
        """

        offset = (
            self.config
            .tdoa_calibration
            .pair_offset_s(
                node_a,
                node_b,
            )
        )

        try:

            offset = float(
                offset
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                (
                    "Configured TDOA calibration "
                    f"offset for pair "
                    f"({node_a}, {node_b}) "
                    "must be numeric."
                )
            ) from exc

        if not math.isfinite(
            offset
        ):

            raise ValueError(
                (
                    "Configured TDOA calibration "
                    f"offset for pair "
                    f"({node_a}, {node_b}) "
                    "must be finite."
                )
            )

        return (
            offset
        )

    # ==================================================================
    # CALIBRATION CORRECTION
    # ==================================================================

    def _correct_pair_delay_s(
        self,
        node_a: int,
        node_b: int,
        measured_delay_s: float,
    ) -> float:
        """
        Apply configured timing calibration to a raw GCC-PHAT delay.

        corrected
            =
        measured - pair_offset
        """

        try:

            measured = float(
                measured_delay_s
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                (
                    "GCC-PHAT delay must "
                    "be numeric."
                )
            ) from exc

        if not math.isfinite(
            measured
        ):

            raise ValueError(
                (
                    "GCC-PHAT delay must "
                    "be finite."
                )
            )

        corrected = (
            self.config
            .tdoa_calibration
            .correct_tdoa_s(
                node_a,
                node_b,
                measured,
            )
        )

        corrected = float(
            corrected
        )

        if not math.isfinite(
            corrected
        ):

            raise ValueError(
                (
                    "Calibrated TDOA must "
                    "be finite."
                )
            )

        return (
            corrected
        )

    # ==================================================================
    # GCC SEARCH LIMIT
    # ==================================================================

    def _gcc_search_limit_s(
        self,
        *,
        physical_max_delay_s: float,
        calibration_offset_s: float,
    ) -> float:
        """
        Build GCC-PHAT search interval.

        Why this is larger than the geometric limit
        --------------------------------------------
        The physical propagation delay is bounded by microphone spacing.

        However, before calibration is applied, the raw measured delay
        can additionally contain systematic channel timing bias:

            raw_delay
                ≈
            physical_delay + calibration_offset

        Therefore GCC-PHAT must be allowed to observe:

            physical_max
            +
            abs(calibration_offset)

        After measurement, calibration is removed and the corrected
        delay is checked against the original physical limit.
        """

        physical_limit = float(
            physical_max_delay_s
        )

        offset = float(
            calibration_offset_s
        )

        if (
            not math.isfinite(
                physical_limit
            )
            or physical_limit
            < 0.0
        ):

            raise ValueError(
                (
                    "physical_max_delay_s must "
                    "be finite and non-negative."
                )
            )

        if not math.isfinite(
            offset
        ):

            raise ValueError(
                (
                    "calibration_offset_s must "
                    "be finite."
                )
            )

        return (
            physical_limit
            + abs(
                offset
            )
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
        Localize an acoustic source from one common absolute sample
        window.
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
        # Use temporal center of localization window instead of leading
        # edge when selecting nearby environmental telemetry.
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
                state
                is None
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

                conditioned = (
                    np.ascontiguousarray(
                        raw,
                        dtype=np.float64,
                    )
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

            if not np.all(
                np.isfinite(
                    conditioned
                )
            ):

                raise ValueError(
                    (
                        f"node {node_id} conditioned "
                        "localization waveform contains "
                        "non-finite samples."
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
            # GEOMETRIC PHYSICAL PAIR LIMIT
            # ----------------------------------------------------------
            #
            # This is the maximum physically possible acoustic
            # propagation delay between the microphones.
            # ----------------------------------------------------------

            physical_max_delay_seconds = (
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

            physical_max_delay_seconds = float(
                physical_max_delay_seconds
            )

            if (
                not math.isfinite(
                    physical_max_delay_seconds
                )
                or physical_max_delay_seconds
                < 0.0
            ):

                raise ValueError(
                    (
                        "physical_max_delay returned "
                        "an invalid pair limit for "
                        f"nodes ({node_a}, {node_b})."
                    )
                )

            # ----------------------------------------------------------
            # CONFIGURED PAIR CALIBRATION OFFSET
            # ----------------------------------------------------------

            calibration_offset_s = (
                self._pair_calibration_offset_s(
                    node_a,
                    node_b,
                )
            )

            # ----------------------------------------------------------
            # RAW GCC SEARCH LIMIT
            # ----------------------------------------------------------
            #
            # GCC must search far enough to observe the physical delay
            # PLUS any known systematic hardware/channel offset.
            #
            # With calibration disabled:
            #
            #     calibration_offset = 0
            #
            # and this becomes exactly the original physical limit.
            # ----------------------------------------------------------

            gcc_search_limit_s = (
                self._gcc_search_limit_s(
                    physical_max_delay_s=
                        physical_max_delay_seconds,

                    calibration_offset_s=
                        calibration_offset_s,
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
                            physical_max_delay_seconds,

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
            #         =
            #     B arrives later than A
            #
            # matching:
            #
            #     TDOAMeasurement(
            #         node_a=A,
            #         node_b=B,
            #     )
            #
            # GCC returns RAW measured channel delay.
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
                        gcc_search_limit_s,

                    interpolation=
                        self.config.interpolation,

                    min_peak_ratio=
                        self.config.min_peak_ratio,

                    beta=
                        self.config.gcc_beta,

                    frequency_band_hz=
                        self.config.gcc_frequency_band_hz,
                )
            )

            # ----------------------------------------------------------
            # RAW DELAY
            # ----------------------------------------------------------

            raw_delay_seconds = float(
                gcc_result.delay_seconds
            )

            if not math.isfinite(
                raw_delay_seconds
            ):

                raise ValueError(
                    (
                        "GCC-PHAT produced a "
                        "non-finite delay."
                    )
                )

            # ----------------------------------------------------------
            # TIMING CALIBRATION
            # ----------------------------------------------------------
            #
            # corrected:
            #
            #     raw - (bias_B - bias_A)
            #
            # With calibration disabled this is exactly:
            #
            #     corrected = raw
            # ----------------------------------------------------------

            corrected_delay_seconds = (
                self._correct_pair_delay_s(
                    node_a,
                    node_b,
                    raw_delay_seconds,
                )
            )

            corrected_delay_samples = (
                corrected_delay_seconds
                * sample_rate
            )

            # ----------------------------------------------------------
            # DEFENSIVE PHYSICAL CHECK
            # ----------------------------------------------------------
            #
            # IMPORTANT:
            #
            # Physical validity is checked AFTER calibration.
            #
            # Raw GCC delay may legitimately exceed the geometric limit
            # by a known systematic timing offset.
            # ----------------------------------------------------------

            physical_tolerance = max(
                1e-12,

                physical_max_delay_seconds
                * 1e-9,
            )

            physically_valid = (
                abs(
                    corrected_delay_seconds
                )
                <= (
                    physical_max_delay_seconds
                    + physical_tolerance
                )
            )

            valid = bool(
                gcc_result.valid
                and physically_valid
            )

            # ----------------------------------------------------------
            # REASON
            # ----------------------------------------------------------

            if not (
                physically_valid
            ):

                reason = (
                    "delay outside physical pair limit"
                    if calibration_offset_s == 0.0
                    else "calibrated delay outside physical pair limit"
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
                # values.
                #
                # A saturated uniqueness ratio is represented by a large
                # finite number instead of infinity.
                peak_ratio = (
                    float(
                        np.finfo(
                            np.float64
                        ).max
                    )
                )

            # ----------------------------------------------------------
            # CALIBRATED MEASUREMENT
            # ----------------------------------------------------------
            #
            # The solver receives corrected_delay_seconds.
            #
            # The original public TDOAMeasurement contract therefore does
            # not need to change.
            # ----------------------------------------------------------

            measurements.append(
                TDOAMeasurement(
                    node_a=
                        node_a,

                    node_b=
                        node_b,

                    delay_seconds=
                        corrected_delay_seconds,

                    delay_samples=
                        corrected_delay_samples,

                    peak_ratio=
                        peak_ratio,

                    max_delay_seconds=
                        physical_max_delay_seconds,

                    valid=
                        valid,

                    reason=
                        reason,
                )
            )

        # ==============================================================
        # FINAL SESSION CONSISTENCY CHECK
        # ==============================================================
        #
        # Pairwise correlation can take non-trivial CPU time.
        #
        # Do not solve a position if a new acquisition session replaced
        # the one from which the waveforms were extracted.
        # ==============================================================

        if (
            self._common_stream_session()
            != common_session
        ):

            raise RuntimeError(
                (
                    "Acquisition session changed "
                    "during TDOA estimation."
                )
            )

        # ==============================================================
        # NONLINEAR POSITION SOLVE
        # ==============================================================
        #
        # measurements now contain calibrated TDOAs when calibration is
        # enabled.
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

            return (
                None
            )

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
                state
                is None
                or state.session_id
                != common_session
                or not state.audio_blocks
            ):

                return (
                    None
                )

            latest_block = (
                state.audio_blocks[
                    -1
                ]
            )

            if (
                latest_block.session_id
                != common_session
            ):

                return (
                    None
                )

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

            return (
                None
            )

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

            return (
                None
            )

        return (
            self.locate_window(
                start_sample=
                    start_sample,

                length=
                    window_samples,
            )
        )
