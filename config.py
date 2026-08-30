from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from dataclasses import (
    dataclass,
    field,
)

from pathlib import (
    Path,
)


# ======================================================================
# PROTOCOL LIMITS
# ======================================================================
#
# These values mirror fields frozen in Protocol v4.
#
# Keeping protocol limits in configuration validation prevents a valid
# Python configuration from producing values that cannot be represented
# by the ESP32 wire protocol.
# ======================================================================


UINT8_MAX = (
    0xFF
)


UINT16_MAX = (
    0xFFFF
)


UINT32_MAX = (
    0xFFFFFFFF
)


INT16_MAX = (
    0x7FFF
)


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _require_finite(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require a finite numeric configuration value.
    """

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(
        result
    ):

        raise ValueError(
            f"{name} must be finite."
        )

    return (
        result
    )


def _require_positive_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a positive non-boolean integer.
    """

    if isinstance(
        value,
        bool,
    ):

        raise TypeError(
            f"{name} must be an integer."
        )

    if not isinstance(
        value,
        int,
    ):

        raise TypeError(
            f"{name} must be an integer."
        )

    if (
        value
        <= 0
    ):

        raise ValueError(
            f"{name} must be greater than 0."
        )

    return (
        value
    )


# ======================================================================
# NETWORK CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class NetworkConfig:
    """
    TCP receiver/network settings used by the laptop server.
    """

    host: str = (
        "0.0.0.0"
    )

    port: int = (
        5001
    )

    read_timeout_s: float = (
        10.0
    )

    hello_timeout_s: float = (
        10.0
    )

    max_payload_bytes: int = (
        64
        * 1024
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate laptop TCP receiver settings.
        """

        if not isinstance(
            self.host,
            str,
        ):

            raise TypeError(
                "Network host must be a string."
            )

        if not (
            self.host.strip()
        ):

            raise ValueError(
                "Network host cannot be empty."
            )

        if (
            isinstance(
                self.port,
                bool,
            )
            or not isinstance(
                self.port,
                int,
            )
        ):

            raise TypeError(
                "Network port must be an integer."
            )

        if not (
            1
            <= self.port
            <= 65_535
        ):

            raise ValueError(
                (
                    "Network port must be "
                    "between 1 and 65535."
                )
            )

        read_timeout = (
            _require_finite(
                self.read_timeout_s,
                name=
                    "Network read_timeout_s",
            )
        )

        if (
            read_timeout
            <= 0
        ):

            raise ValueError(
                (
                    "Network read_timeout_s "
                    "must be greater than 0."
                )
            )

        hello_timeout = (
            _require_finite(
                self.hello_timeout_s,
                name=
                    "Network hello_timeout_s",
            )
        )

        if (
            hello_timeout
            <= 0
        ):

            raise ValueError(
                (
                    "Network hello_timeout_s "
                    "must be greater than 0."
                )
            )

        _require_positive_int(
            self.max_payload_bytes,
            name=
                "Network max_payload_bytes",
        )

        if (
            self.max_payload_bytes
            > UINT32_MAX
        ):

            raise ValueError(
                (
                    "Network max_payload_bytes "
                    "cannot exceed Protocol-v4 "
                    "uint32 payloadLength capacity."
                )
            )


# ======================================================================
# AUDIO CONFIGURATION
# ======================================================================


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

    sample_rate: int = (
        48_000
    )

    frames_per_block: int = (
        1024
    )

    channels: int = (
        1
    )

    sample_width_bytes: int = (
        2
    )

    # ------------------------------------------------------------------
    # LAPTOP STREAM BUFFER
    # ------------------------------------------------------------------

    buffer_seconds: float = (
        20.0
    )

    # Maximum expected coarse sampleIndex difference during diagnostic
    # stream alignment.
    sync_tolerance_samples: int = (
        10
    )

    # ------------------------------------------------------------------
    # CONTINUOUS SESSION RECORDING
    # ------------------------------------------------------------------

    record_wav: bool = (
        True
    )

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

        return (
            self.frames_per_block
            / self.sample_rate
        )

    @property
    def bytes_per_block(
        self,
    ) -> int:
        """
        AUDIO payload size for one Protocol-v4 PCM block.
        """

        return (
            self.frames_per_block
            * self.channels
            * self.sample_width_bytes
        )

    @property
    def buffer_samples(
        self,
    ) -> int:
        """
        Approximate number of samples represented by the configured
        laptop stream retention period.
        """

        return int(
            math.ceil(
                self.buffer_seconds
                * self.sample_rate
            )
        )

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
            math.ceil(
                self.buffer_seconds
                * self.sample_rate
                / self.frames_per_block
            )
        )

        return max(
            8,
            required
            + 2,
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
            name=
                "Audio sample_rate",
        )

        if (
            self.sample_rate
            > UINT32_MAX
        ):

            raise ValueError(
                (
                    "Audio sample_rate exceeds "
                    "HELLO uint32 capacity."
                )
            )

        _require_positive_int(
            self.frames_per_block,
            name=
                "Audio frames_per_block",
        )

        if (
            self.frames_per_block
            > UINT16_MAX
        ):

            raise ValueError(
                (
                    "Audio frames_per_block exceeds "
                    "HELLO uint16 capacity."
                )
            )

        if (
            self.channels
            != 1
        ):

            raise ValueError(
                (
                    "Current Wildlife Soundscape "
                    "Protocol-v4 transport requires "
                    "mono audio from each ESP32 node."
                )
            )

        if (
            self.sample_width_bytes
            != 2
        ):

            raise ValueError(
                (
                    "Current Protocol-v4 AUDIO "
                    "transport requires PCM16 "
                    "(2 bytes per sample)."
                )
            )

        buffer_seconds = (
            _require_finite(
                self.buffer_seconds,
                name=
                    "Audio buffer_seconds",
            )
        )

        if (
            buffer_seconds
            <= 0
        ):

            raise ValueError(
                (
                    "Audio buffer_seconds must "
                    "be greater than 0."
                )
            )

        if (
            isinstance(
                self.sync_tolerance_samples,
                bool,
            )
            or not isinstance(
                self.sync_tolerance_samples,
                int,
            )
        ):

            raise TypeError(
                (
                    "Audio sync_tolerance_samples "
                    "must be an integer."
                )
            )

        if (
            self.sync_tolerance_samples
            < 0
        ):

            raise ValueError(
                (
                    "Audio sync_tolerance_samples "
                    "cannot be negative."
                )
            )

        if (
            self.sync_tolerance_samples
            > INT16_MAX
        ):

            raise ValueError(
                (
                    "Audio sync_tolerance_samples "
                    "exceeds HELLO int16 capacity."
                )
            )

        if not isinstance(
            self.record_wav,
            bool,
        ):

            raise TypeError(
                "Audio record_wav must be bool."
            )


# ======================================================================
# TDOA CALIBRATION CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TDOACalibrationConfig:
    """
    Runtime TDOA timing-calibration configuration.

    Calibration model
    -----------------
    Each microphone/node has one estimated timing bias relative to an
    arbitrary calibration reference node.

    For nodes A and B:

        pair_offset(A, B)
            =
        bias_B - bias_A

    The runtime corrected TDOA is therefore:

        corrected_tdoa
            =
        measured_tdoa - pair_offset


    Why node biases are stored
    --------------------------
    Storing node biases instead of three unrelated pair offsets
    guarantees internally consistent calibration:

        offset(1, 2)
        +
        offset(2, 3)
        =
        offset(1, 3)


    Safety
    ------
    Calibration is disabled by default.

    Real node biases must be measured using controlled calibration data
    before `enabled` is changed to True.

    Zero offsets must not be presented as experimentally calibrated
    values.
    """

    # ------------------------------------------------------------------
    # MASTER SWITCH
    # ------------------------------------------------------------------

    enabled: bool = (
        False
    )

    # ------------------------------------------------------------------
    # CALIBRATION REFERENCE
    # ------------------------------------------------------------------
    #
    # Bias for this node is fixed at exactly zero.
    #
    # Pairwise bias differences are independent of which node is chosen
    # as this reference.
    # ------------------------------------------------------------------

    reference_node: int = (
        1
    )

    # ------------------------------------------------------------------
    # ESTIMATED NODE BIASES
    # ------------------------------------------------------------------
    #
    # Units:
    #     seconds
    #
    # Example only after actual calibration:
    #
    # {
    #     1: 0.0,
    #     2: 0.000020,
    #     3: -0.000010,
    # }
    #
    # The default is intentionally empty because calibration has not yet
    # been physically measured.
    # ------------------------------------------------------------------

    node_biases_s: dict[
        int,
        float,
    ] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # CALIBRATION PROVENANCE
    # ------------------------------------------------------------------

    result_path: Path = (
        Path(
            "data/calibration/tdoa_calibration.json"
        )
    )

    generated_at: str | None = (
        None
    )

    # ------------------------------------------------------------------
    # FIT QUALITY
    # ------------------------------------------------------------------
    #
    # RMS disagreement between directly observed pair residuals and the
    # coherent per-node timing-bias model.
    #
    # This is populated from TDOACalibrationResult when calibration is
    # accepted for runtime use.
    # ------------------------------------------------------------------

    rms_pair_consistency_error_s: float | None = (
        None
    )

    # Maximum accepted consistency error expressed in AUDIO samples.
    #
    # At 48 kHz:
    #
    #     1 sample ≈ 20.83 microseconds
    # ------------------------------------------------------------------

    max_consistency_error_samples: float = (
        1.0
    )

    # ==================================================================
    # DERIVED CALIBRATION VALUES
    # ==================================================================

    def pair_offset_s(
        self,
        node_a: int,
        node_b: int,
    ) -> float:
        """
        Return calibrated timing offset for requested pair orientation.

        Convention
        ----------
        Offset corresponds to:

            arrival_B - arrival_A

        If calibration is disabled, zero is returned.

        Raises
        ------
        KeyError
            If calibration is enabled but one of the required node
            biases is unavailable.
        """

        if (
            node_a
            == node_b
        ):

            raise ValueError(
                (
                    "TDOA calibration requires "
                    "two different node IDs."
                )
            )

        if not (
            self.enabled
        ):

            return (
                0.0
            )

        try:

            bias_a = (
                self.node_biases_s[
                    node_a
                ]
            )

            bias_b = (
                self.node_biases_s[
                    node_b
                ]
            )

        except KeyError as exc:

            raise KeyError(
                (
                    "Missing TDOA calibration bias "
                    f"for node {exc.args[0]}."
                )
            ) from exc

        return (
            bias_b
            - bias_a
        )

    def pair_offset_samples(
        self,
        node_a: int,
        node_b: int,
        *,
        sample_rate: int,
    ) -> float:
        """
        Return calibrated pair timing offset in samples.
        """

        _require_positive_int(
            sample_rate,
            name=
                (
                    "TDOA calibration "
                    "sample_rate"
                ),
        )

        return (
            self.pair_offset_s(
                node_a,
                node_b,
            )
            * sample_rate
        )

    def correct_tdoa_s(
        self,
        node_a: int,
        node_b: int,
        measured_tdoa_s: float,
    ) -> float:
        """
        Apply configured timing calibration to one TDOA measurement.
        """

        measured = (
            _require_finite(
                measured_tdoa_s,
                name=
                    (
                        "Measured TDOA"
                    ),
            )
        )

        return (
            measured
            - self.pair_offset_s(
                node_a,
                node_b,
            )
        )

    def maximum_absolute_pair_offset_s(
        self,
    ) -> float:
        """
        Return largest configured absolute pair timing offset.

        This value is useful when constructing a GCC-PHAT search window,
        because systematic channel delay can slightly extend the
        observed delay beyond the purely geometric propagation limit.
        """

        if (
            not self.enabled
            or len(
                self.node_biases_s
            )
            < 2
        ):

            return (
                0.0
            )

        biases = tuple(
            float(
                value
            )
            for value
            in self.node_biases_s.values()
        )

        return (
            max(
                biases
            )
            - min(
                biases
            )
        )

    def maximum_absolute_pair_offset_samples(
        self,
        *,
        sample_rate: int,
    ) -> float:
        """
        Return maximum configured absolute pair offset in samples.
        """

        _require_positive_int(
            sample_rate,
            name=
                (
                    "TDOA calibration "
                    "sample_rate"
                ),
        )

        return (
            self.maximum_absolute_pair_offset_s()
            * sample_rate
        )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
        *,
        sample_rate: int,
        expected_nodes: frozenset[int],
    ) -> None:
        """
        Validate runtime calibration configuration.

        Calibration remains intentionally strict when enabled.

        A calibration cannot be enabled unless:

            every expected node has a bias
            reference-node bias is zero
            calibration fit quality is available
            fit quality satisfies configured acceptance threshold
        """

        if not isinstance(
            self.enabled,
            bool,
        ):

            raise TypeError(
                (
                    "TDOA calibration enabled "
                    "must be bool."
                )
            )

        _require_positive_int(
            sample_rate,
            name=
                (
                    "TDOA calibration "
                    "sample_rate"
                ),
        )

        _require_positive_int(
            self.reference_node,
            name=
                (
                    "TDOA calibration "
                    "reference_node"
                ),
        )

        if (
            self.reference_node
            not in expected_nodes
        ):

            raise ValueError(
                (
                    "TDOA calibration reference_node "
                    "must be one of expected_nodes."
                )
            )

        # ==============================================================
        # RESULT PATH
        # ==============================================================

        if not isinstance(
            self.result_path,
            Path,
        ):

            raise TypeError(
                (
                    "TDOA calibration result_path "
                    "must be pathlib.Path."
                )
            )

        if not (
            self.result_path.name
        ):

            raise ValueError(
                (
                    "TDOA calibration result_path "
                    "must include a filename."
                )
            )

        # ==============================================================
        # TIMESTAMP / PROVENANCE
        # ==============================================================

        if (
            self.generated_at
            is not None
        ):

            if not isinstance(
                self.generated_at,
                str,
            ):

                raise TypeError(
                    (
                        "TDOA calibration generated_at "
                        "must be a string or None."
                    )
                )

            if not (
                self.generated_at.strip()
            ):

                raise ValueError(
                    (
                        "TDOA calibration generated_at "
                        "cannot be empty when provided."
                    )
                )

        # ==============================================================
        # NODE BIASES
        # ==============================================================

        if not isinstance(
            self.node_biases_s,
            dict,
        ):

            raise TypeError(
                (
                    "TDOA calibration node_biases_s "
                    "must be a dictionary."
                )
            )

        normalized_biases: dict[
            int,
            float,
        ] = {}

        for node_id, raw_bias in (
            self.node_biases_s.items()
        ):

            if (
                isinstance(
                    node_id,
                    bool,
                )
                or not isinstance(
                    node_id,
                    int,
                )
            ):

                raise TypeError(
                    (
                        "TDOA calibration node-bias "
                        "keys must be integer node IDs."
                    )
                )

            if (
                node_id
                not in expected_nodes
            ):

                raise ValueError(
                    (
                        "TDOA calibration contains "
                        f"unexpected node {node_id}."
                    )
                )

            bias = (
                _require_finite(
                    raw_bias,
                    name=
                        (
                            "TDOA calibration bias "
                            f"for node {node_id}"
                        ),
                )
            )

            normalized_biases[
                node_id
            ] = (
                bias
            )

        # ==============================================================
        # CONSISTENCY THRESHOLD
        # ==============================================================

        max_consistency_samples = (
            _require_finite(
                self.max_consistency_error_samples,
                name=
                    (
                        "TDOA calibration "
                        "max_consistency_error_samples"
                    ),
            )
        )

        if (
            max_consistency_samples
            <= 0.0
        ):

            raise ValueError(
                (
                    "TDOA calibration "
                    "max_consistency_error_samples "
                    "must be greater than 0."
                )
            )

        # ==============================================================
        # FIT QUALITY
        # ==============================================================

        consistency_error_s: float | None = (
            None
        )

        if (
            self.rms_pair_consistency_error_s
            is not None
        ):

            consistency_error_s = (
                _require_finite(
                    self.rms_pair_consistency_error_s,
                    name=
                        (
                            "TDOA calibration "
                            "rms_pair_consistency_error_s"
                        ),
                )
            )

            if (
                consistency_error_s
                < 0.0
            ):

                raise ValueError(
                    (
                        "TDOA calibration "
                        "rms_pair_consistency_error_s "
                        "cannot be negative."
                    )
                )

        # ==============================================================
        # STRICT REQUIREMENTS ONLY WHEN CALIBRATION IS ACTIVE
        # ==============================================================

        if not (
            self.enabled
        ):

            return

        missing_biases = (
            expected_nodes
            - set(
                normalized_biases
            )
        )

        if (
            missing_biases
        ):

            raise ValueError(
                (
                    "Enabled TDOA calibration is "
                    "missing timing biases for "
                    f"node(s): {sorted(missing_biases)}."
                )
            )

        reference_bias = (
            normalized_biases[
                self.reference_node
            ]
        )

        if (
            abs(
                reference_bias
            )
            > 1e-12
        ):

            raise ValueError(
                (
                    "TDOA calibration reference-node "
                    "bias must be zero."
                )
            )

        if (
            consistency_error_s
            is None
        ):

            raise ValueError(
                (
                    "Enabled TDOA calibration requires "
                    "rms_pair_consistency_error_s from "
                    "the calibration fit."
                )
            )

        consistency_error_samples = (
            consistency_error_s
            * sample_rate
        )

        if (
            consistency_error_samples
            > max_consistency_samples
        ):

            raise ValueError(
                (
                    "TDOA calibration fit is outside "
                    "the configured quality threshold. "
                    f"RMS consistency error="
                    f"{consistency_error_samples:.3f} samples, "
                    f"maximum allowed="
                    f"{max_consistency_samples:.3f} samples."
                )
            )


# ======================================================================
# LOCALIZATION CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LocalizationConfig:
    """
    GCC-PHAT / TDOA acoustic localization configuration.
    """

    # ------------------------------------------------------------------
    # MICROPHONE GEOMETRY
    # ------------------------------------------------------------------
    #
    # Initial laboratory geometry:
    #
    #             Node 2
    #              /  \
    #             /    \
    #        Node 1----Node 3
    #
    # 1 m equilateral triangle.
    #
    # Replace these coordinates with physically measured microphone
    # coordinates during calibration.
    # ------------------------------------------------------------------

    node_positions: dict[
        int,
        tuple[
            float,
            float,
        ],
    ] = field(
        default_factory=lambda: {
            1: (
                0.0,
                0.0,
            ),
            2: (
                0.5,
                0.8660254037844386,
            ),
            3: (
                1.0,
                0.0,
            ),
        }
    )

    reference_node: int = (
        1
    )

    # ------------------------------------------------------------------
    # TDOA WINDOW
    # ------------------------------------------------------------------

    window_samples: int = (
        8192
    )

    # Maximum coarse sampleIndex correction considered before waveform
    # GCC-PHAT estimation.
    max_alignment_search_samples: int = (
        10
    )

    # Fractional-delay interpolation factor used by GCC-PHAT.
    interpolation: int = (
        8
    )

    # ------------------------------------------------------------------
    # QUALITY THRESHOLDS
    # ------------------------------------------------------------------

    min_peak_ratio: float = (
        1.10
    )

    min_rms: float = (
        50.0
    )

    # ------------------------------------------------------------------
    # SPEED OF SOUND
    # ------------------------------------------------------------------

    speed_of_sound_mps: float = (
        343.0
    )

    use_environmental_speed: bool = (
        True
    )

    # ------------------------------------------------------------------
    # LOCALIZATION SIGNAL CONDITIONING
    # ------------------------------------------------------------------

    bandpass_enabled: bool = (
        True
    )

    bandpass_low_hz: float = (
        200.0
    )

    bandpass_high_hz: float = (
        12_000.0
    )

    bandpass_order: int = (
        4
    )

    # ------------------------------------------------------------------
    # TDOA TIMING CALIBRATION
    # ------------------------------------------------------------------
    #
    # Disabled until controlled physical calibration has been performed.
    #
    # Calibration values are applied AFTER raw GCC-PHAT TDOA
    # measurement and BEFORE the geometric localization solver.
    # ------------------------------------------------------------------

    tdoa_calibration: TDOACalibrationConfig = field(
        default_factory=
            TDOACalibrationConfig
    )

    # ------------------------------------------------------------------
    # SOLVER
    # ------------------------------------------------------------------

    constrain_to_array_bounds: bool = (
        False
    )

    bounds_margin_m: float = (
        0.5
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
        *,
        sample_rate: int,
        expected_nodes: frozenset[int],
    ) -> None:
        """
        Validate localization geometry and numerical settings.
        """

        if (
            self.reference_node
            not in expected_nodes
        ):

            raise ValueError(
                (
                    "Localization reference_node "
                    "must be one of expected_nodes."
                )
            )

        # ==============================================================
        # NODE COORDINATES
        # ==============================================================

        missing_positions = (
            expected_nodes
            - set(
                self.node_positions.keys()
            )
        )

        if (
            missing_positions
        ):

            raise ValueError(
                (
                    "Missing localization coordinates "
                    "for node(s): "
                    f"{sorted(missing_positions)}"
                )
            )

        coordinates: list[
            tuple[
                float,
                float,
            ]
        ] = []

        for node_id in sorted(
            expected_nodes
        ):

            position = (
                self.node_positions[
                    node_id
                ]
            )

            try:

                x_raw, y_raw = (
                    position
                )

            except Exception as exc:

                raise ValueError(
                    (
                        "Localization position for "
                        f"node {node_id} must contain "
                        "exactly two coordinates."
                    )
                ) from exc

            x = (
                _require_finite(
                    x_raw,
                    name=
                        (
                            "Localization node "
                            f"{node_id} x"
                        ),
                )
            )

            y = (
                _require_finite(
                    y_raw,
                    name=
                        (
                            "Localization node "
                            f"{node_id} y"
                        ),
                )
            )

            coordinates.append(
                (
                    x,
                    y,
                )
            )

        # --------------------------------------------------------------
        # UNIQUE MICROPHONE POSITIONS
        # --------------------------------------------------------------

        if (
            len(
                set(
                    coordinates
                )
            )
            != len(
                coordinates
            )
        ):

            raise ValueError(
                (
                    "Localization microphones "
                    "must have unique coordinates."
                )
            )

        # --------------------------------------------------------------
        # NON-COLLINEAR ARRAY
        # --------------------------------------------------------------

        non_collinear = (
            False
        )

        coordinate_count = (
            len(
                coordinates
            )
        )

        for i in range(
            coordinate_count
            - 2
        ):

            x1, y1 = (
                coordinates[
                    i
                ]
            )

            for j in range(
                i
                + 1,
                coordinate_count
                - 1,
            ):

                x2, y2 = (
                    coordinates[
                        j
                    ]
                )

                for k in range(
                    j
                    + 1,
                    coordinate_count,
                ):

                    x3, y3 = (
                        coordinates[
                            k
                        ]
                    )

                    twice_area = abs(
                        (
                            x2
                            - x1
                        )
                        * (
                            y3
                            - y1
                        )
                        - (
                            y2
                            - y1
                        )
                        * (
                            x3
                            - x1
                        )
                    )

                    if (
                        twice_area
                        > 1e-12
                    ):

                        non_collinear = (
                            True
                        )

                        break

                if (
                    non_collinear
                ):

                    break

            if (
                non_collinear
            ):

                break

        if not (
            non_collinear
        ):

            raise ValueError(
                (
                    "Localization microphone geometry "
                    "is collinear. 2-D TDOA requires "
                    "at least three non-collinear "
                    "microphone positions."
                )
            )

        # ==============================================================
        # WINDOW / ALIGNMENT
        # ==============================================================

        _require_positive_int(
            self.window_samples,
            name=
                "Localization window_samples",
        )

        if (
            isinstance(
                self.max_alignment_search_samples,
                bool,
            )
            or not isinstance(
                self.max_alignment_search_samples,
                int,
            )
        ):

            raise TypeError(
                (
                    "Localization "
                    "max_alignment_search_samples "
                    "must be an integer."
                )
            )

        if (
            self.max_alignment_search_samples
            < 0
        ):

            raise ValueError(
                (
                    "Localization "
                    "max_alignment_search_samples "
                    "cannot be negative."
                )
            )

        _require_positive_int(
            self.interpolation,
            name=
                "Localization interpolation",
        )

        # ==============================================================
        # QUALITY
        # ==============================================================

        peak_ratio = (
            _require_finite(
                self.min_peak_ratio,
                name=
                    "Localization min_peak_ratio",
            )
        )

        if (
            peak_ratio
            < 1.0
        ):

            raise ValueError(
                (
                    "Localization min_peak_ratio "
                    "must be at least 1.0."
                )
            )

        minimum_rms = (
            _require_finite(
                self.min_rms,
                name=
                    "Localization min_rms",
            )
        )

        if (
            minimum_rms
            < 0
        ):

            raise ValueError(
                (
                    "Localization min_rms "
                    "cannot be negative."
                )
            )

        # ==============================================================
        # SPEED OF SOUND
        # ==============================================================

        sound_speed = (
            _require_finite(
                self.speed_of_sound_mps,
                name=
                    "Localization speed_of_sound_mps",
            )
        )

        if (
            sound_speed
            <= 0
        ):

            raise ValueError(
                (
                    "Localization fallback "
                    "speed_of_sound_mps must "
                    "be greater than 0."
                )
            )

        if not isinstance(
            self.use_environmental_speed,
            bool,
        ):

            raise TypeError(
                (
                    "Localization "
                    "use_environmental_speed "
                    "must be bool."
                )
            )

        # ==============================================================
        # BANDPASS
        # ==============================================================

        nyquist = (
            sample_rate
            / 2.0
        )

        if (
            self.bandpass_enabled
        ):

            low_hz = (
                _require_finite(
                    self.bandpass_low_hz,
                    name=
                        (
                            "Localization "
                            "bandpass_low_hz"
                        ),
                )
            )

            high_hz = (
                _require_finite(
                    self.bandpass_high_hz,
                    name=
                        (
                            "Localization "
                            "bandpass_high_hz"
                        ),
                )
            )

            if not (
                0.0
                < low_hz
                < high_hz
                < nyquist
            ):

                raise ValueError(
                    (
                        "Localization bandpass "
                        "frequencies must satisfy "
                        "0 < low < high < Nyquist."
                    )
                )

        _require_positive_int(
            self.bandpass_order,
            name=
                "Localization bandpass_order",
        )

        # ==============================================================
        # TDOA CALIBRATION
        # ==============================================================

        self.tdoa_calibration.validate(
            sample_rate=
                sample_rate,

            expected_nodes=
                expected_nodes,
        )

        # ==============================================================
        # SOLVER
        # ==============================================================

        if not isinstance(
            self.constrain_to_array_bounds,
            bool,
        ):

            raise TypeError(
                (
                    "Localization "
                    "constrain_to_array_bounds "
                    "must be bool."
                )
            )

        margin = (
            _require_finite(
                self.bounds_margin_m,
                name=
                    "Localization bounds_margin_m",
            )
        )

        if (
            margin
            < 0
        ):

            raise ValueError(
                (
                    "Localization bounds_margin_m "
                    "cannot be negative."
                )
            )


# ======================================================================
# EVENT DETECTION CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class EventDetectionConfig:
    """
    Adaptive multi-node acoustic event detector configuration.
    """

    enabled: bool = (
        True
    )

    # ------------------------------------------------------------------
    # BACKGROUND NOISE MODEL
    # ------------------------------------------------------------------

    noise_history_blocks: int = (
        96
    )

    noise_quantile: float = (
        0.30
    )

    initial_noise_dbfs: float = (
        -52.0
    )

    trigger_margin_db: float = (
        8.0
    )

    release_margin_db: float = (
        4.0
    )

    # ------------------------------------------------------------------
    # SPECTRAL ACTIVITY
    # ------------------------------------------------------------------

    min_spectral_flux: float = (
        0.06
    )

    strong_energy_margin_db: float = (
        15.0
    )

    # ------------------------------------------------------------------
    # TEMPORAL / MULTI-NODE AGREEMENT
    # ------------------------------------------------------------------

    attack_blocks: int = (
        1
    )

    release_blocks: int = (
        2
    )

    min_nodes: int = (
        2
    )

    # ------------------------------------------------------------------
    # EVENT LENGTH
    # ------------------------------------------------------------------

    min_event_ms: float = (
        30.0
    )

    max_event_s: float = (
        12.0
    )

    pre_pad_s: float = (
        0.50
    )

    post_pad_s: float = (
        0.75
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
        *,
        expected_node_count: int,
    ) -> None:
        """
        Validate detector thresholds and temporal settings.
        """

        if not isinstance(
            self.enabled,
            bool,
        ):

            raise TypeError(
                "Event detection enabled must be bool."
            )

        _require_positive_int(
            self.noise_history_blocks,
            name=
                (
                    "Event detection "
                    "noise_history_blocks"
                ),
        )

        noise_quantile = (
            _require_finite(
                self.noise_quantile,
                name=
                    (
                        "Event detection "
                        "noise_quantile"
                    ),
            )
        )

        if not (
            0.0
            <= noise_quantile
            <= 1.0
        ):

            raise ValueError(
                (
                    "Event detection "
                    "noise_quantile must be "
                    "between 0 and 1."
                )
            )

        initial_noise = (
            _require_finite(
                self.initial_noise_dbfs,
                name=
                    (
                        "Event detection "
                        "initial_noise_dbfs"
                    ),
            )
        )

        if (
            initial_noise
            > 0
        ):

            raise ValueError(
                (
                    "Event detection "
                    "initial_noise_dbfs cannot "
                    "exceed 0 dBFS."
                )
            )

        trigger_margin = (
            _require_finite(
                self.trigger_margin_db,
                name=
                    (
                        "Event detection "
                        "trigger_margin_db"
                    ),
            )
        )

        release_margin = (
            _require_finite(
                self.release_margin_db,
                name=
                    (
                        "Event detection "
                        "release_margin_db"
                    ),
            )
        )

        if (
            trigger_margin
            <= 0
        ):

            raise ValueError(
                (
                    "Event detection "
                    "trigger_margin_db must "
                    "be greater than 0."
                )
            )

        if (
            release_margin
            < 0
        ):

            raise ValueError(
                (
                    "Event detection "
                    "release_margin_db cannot "
                    "be negative."
                )
            )

        if (
            release_margin
            >= trigger_margin
        ):

            raise ValueError(
                (
                    "Event detection release_margin_db "
                    "must remain below "
                    "trigger_margin_db to provide "
                    "detector hysteresis."
                )
            )

        spectral_flux = (
            _require_finite(
                self.min_spectral_flux,
                name=
                    (
                        "Event detection "
                        "min_spectral_flux"
                    ),
            )
        )

        if (
            spectral_flux
            < 0
        ):

            raise ValueError(
                (
                    "Event detection "
                    "min_spectral_flux "
                    "cannot be negative."
                )
            )

        strong_margin = (
            _require_finite(
                self.strong_energy_margin_db,
                name=
                    (
                        "Event detection "
                        "strong_energy_margin_db"
                    ),
            )
        )

        if (
            strong_margin
            < trigger_margin
        ):

            raise ValueError(
                (
                    "Event detection "
                    "strong_energy_margin_db "
                    "should be at least as large "
                    "as trigger_margin_db."
                )
            )

        _require_positive_int(
            self.attack_blocks,
            name=
                "Event detection attack_blocks",
        )

        _require_positive_int(
            self.release_blocks,
            name=
                "Event detection release_blocks",
        )

        _require_positive_int(
            self.min_nodes,
            name=
                "Event detection min_nodes",
        )

        if (
            self.min_nodes
            < 2
        ):

            raise ValueError(
                (
                    "Current multi-node detector "
                    "requires min_nodes >= 2."
                )
            )

        if (
            self.min_nodes
            > expected_node_count
        ):

            raise ValueError(
                (
                    "Event detection min_nodes "
                    "cannot exceed the number "
                    "of expected nodes."
                )
            )

        minimum_ms = (
            _require_finite(
                self.min_event_ms,
                name=
                    (
                        "Event detection "
                        "min_event_ms"
                    ),
            )
        )

        maximum_s = (
            _require_finite(
                self.max_event_s,
                name=
                    (
                        "Event detection "
                        "max_event_s"
                    ),
            )
        )

        pre_pad = (
            _require_finite(
                self.pre_pad_s,
                name=
                    (
                        "Event detection "
                        "pre_pad_s"
                    ),
            )
        )

        post_pad = (
            _require_finite(
                self.post_pad_s,
                name=
                    (
                        "Event detection "
                        "post_pad_s"
                    ),
            )
        )

        if (
            minimum_ms
            <= 0
        ):

            raise ValueError(
                (
                    "Event detection min_event_ms "
                    "must be greater than 0."
                )
            )

        if (
            maximum_s
            <= 0
        ):

            raise ValueError(
                (
                    "Event detection max_event_s "
                    "must be greater than 0."
                )
            )

        if (
            minimum_ms
            / 1000.0
            > maximum_s
        ):

            raise ValueError(
                (
                    "Event detection min_event_ms "
                    "cannot exceed max_event_s."
                )
            )

        if (
            pre_pad
            < 0
        ):

            raise ValueError(
                (
                    "Event detection pre_pad_s "
                    "cannot be negative."
                )
            )

        if (
            post_pad
            < 0
        ):

            raise ValueError(
                (
                    "Event detection post_pad_s "
                    "cannot be negative."
                )
            )


# ======================================================================
# DSP CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class DSPConfig:
    """
    Laptop-side acoustic preprocessing and feature-extraction settings.

    Controls:

        preprocessing
        FFT/STFT
        MFCC
        spectral descriptors
        SNR estimation
        best-node selection
        normalized model-waveform generation

    Raw PCM recordings remain untouched.
    """

    remove_dc: bool = (
        True
    )

    bandpass_enabled: bool = (
        True
    )

    low_cutoff_hz: float = (
        100.0
    )

    high_cutoff_hz: float = (
        16_000.0
    )

    filter_order: int = (
        4
    )

    normalize_for_model: bool = (
        True
    )

    model_target_peak: float = (
        0.98
    )

    n_fft: int = (
        2048
    )

    hop_length: int = (
        512
    )

    n_mfcc: int = (
        13
    )

    n_mels: int = (
        64
    )

    mfcc_fmin_hz: float = (
        50.0
    )

    mfcc_fmax_hz: float = (
        16_000.0
    )

    roll_percent: float = (
        0.85
    )

    snr_frame_length: int = (
        512
    )

    noise_quantile: float = (
        0.30
    )

    signal_quantile: float = (
        0.90
    )

    noise_prepad_fraction: float = (
        0.80
    )

    minimum_event_samples: int = (
        64
    )

    digital_noise_floor: float = (
        1.0
        / 32768.0
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
        sample_rate: int,
    ) -> None:
        """
        Validate DSP settings against acquisition sample rate.
        """

        _require_positive_int(
            sample_rate,
            name=
                "DSP sample_rate",
        )

        nyquist = (
            sample_rate
            / 2.0
        )

        if not isinstance(
            self.remove_dc,
            bool,
        ):

            raise TypeError(
                "DSP remove_dc must be bool."
            )

        if not isinstance(
            self.bandpass_enabled,
            bool,
        ):

            raise TypeError(
                "DSP bandpass_enabled must be bool."
            )

        if (
            self.bandpass_enabled
        ):

            low_hz = (
                _require_finite(
                    self.low_cutoff_hz,
                    name=
                        "DSP low_cutoff_hz",
                )
            )

            high_hz = (
                _require_finite(
                    self.high_cutoff_hz,
                    name=
                        "DSP high_cutoff_hz",
                )
            )

            if (
                low_hz
                <= 0
            ):

                raise ValueError(
                    (
                        "DSP low_cutoff_hz "
                        "must be greater than 0."
                    )
                )

            if (
                high_hz
                <= low_hz
            ):

                raise ValueError(
                    (
                        "DSP high_cutoff_hz "
                        "must be greater than "
                        "low_cutoff_hz."
                    )
                )

            if (
                high_hz
                >= nyquist
            ):

                raise ValueError(
                    (
                        "DSP high_cutoff_hz must "
                        "remain below Nyquist "
                        f"({nyquist:.1f} Hz)."
                    )
                )

        _require_positive_int(
            self.filter_order,
            name=
                "DSP filter_order",
        )

        if not isinstance(
            self.normalize_for_model,
            bool,
        ):

            raise TypeError(
                (
                    "DSP normalize_for_model "
                    "must be bool."
                )
            )

        model_peak = (
            _require_finite(
                self.model_target_peak,
                name=
                    "DSP model_target_peak",
            )
        )

        if not (
            0.0
            < model_peak
            <= 1.0
        ):

            raise ValueError(
                (
                    "DSP model_target_peak "
                    "must be in (0, 1]."
                )
            )

        _require_positive_int(
            self.n_fft,
            name=
                "DSP n_fft",
        )

        _require_positive_int(
            self.hop_length,
            name=
                "DSP hop_length",
        )

        if (
            self.hop_length
            > self.n_fft
        ):

            raise ValueError(
                (
                    "DSP hop_length cannot "
                    "exceed n_fft."
                )
            )

        _require_positive_int(
            self.n_mfcc,
            name=
                "DSP n_mfcc",
        )

        _require_positive_int(
            self.n_mels,
            name=
                "DSP n_mels",
        )

        if (
            self.n_mels
            < self.n_mfcc
        ):

            raise ValueError(
                (
                    "DSP n_mels must be "
                    "greater than or equal "
                    "to n_mfcc."
                )
            )

        fmin = (
            _require_finite(
                self.mfcc_fmin_hz,
                name=
                    "DSP mfcc_fmin_hz",
            )
        )

        fmax = (
            _require_finite(
                self.mfcc_fmax_hz,
                name=
                    "DSP mfcc_fmax_hz",
            )
        )

        if (
            fmin
            < 0
        ):

            raise ValueError(
                (
                    "DSP mfcc_fmin_hz "
                    "cannot be negative."
                )
            )

        if (
            fmax
            <= fmin
        ):

            raise ValueError(
                (
                    "DSP mfcc_fmax_hz must "
                    "exceed mfcc_fmin_hz."
                )
            )

        if (
            fmax
            > nyquist
        ):

            raise ValueError(
                (
                    "DSP mfcc_fmax_hz cannot "
                    "exceed Nyquist "
                    f"({nyquist:.1f} Hz)."
                )
            )

        roll_percent = (
            _require_finite(
                self.roll_percent,
                name=
                    "DSP roll_percent",
            )
        )

        if not (
            0.0
            < roll_percent
            < 1.0
        ):

            raise ValueError(
                (
                    "DSP roll_percent must "
                    "be between 0 and 1."
                )
            )

        _require_positive_int(
            self.snr_frame_length,
            name=
                "DSP snr_frame_length",
        )

        noise_quantile = (
            _require_finite(
                self.noise_quantile,
                name=
                    "DSP noise_quantile",
            )
        )

        signal_quantile = (
            _require_finite(
                self.signal_quantile,
                name=
                    "DSP signal_quantile",
            )
        )

        if not (
            0.0
            <= noise_quantile
            <= 1.0
        ):

            raise ValueError(
                (
                    "DSP noise_quantile must "
                    "be between 0 and 1."
                )
            )

        if not (
            0.0
            <= signal_quantile
            <= 1.0
        ):

            raise ValueError(
                (
                    "DSP signal_quantile must "
                    "be between 0 and 1."
                )
            )

        if (
            signal_quantile
            <= noise_quantile
        ):

            raise ValueError(
                (
                    "DSP signal_quantile must "
                    "exceed noise_quantile."
                )
            )

        noise_fraction = (
            _require_finite(
                self.noise_prepad_fraction,
                name=
                    (
                        "DSP "
                        "noise_prepad_fraction"
                    ),
            )
        )

        if not (
            0.0
            < noise_fraction
            <= 1.0
        ):

            raise ValueError(
                (
                    "DSP noise_prepad_fraction "
                    "must be in (0, 1]."
                )
            )

        _require_positive_int(
            self.minimum_event_samples,
            name=
                "DSP minimum_event_samples",
        )

        digital_floor = (
            _require_finite(
                self.digital_noise_floor,
                name=
                    "DSP digital_noise_floor",
            )
        )

        if (
            digital_floor
            <= 0
        ):

            raise ValueError(
                (
                    "DSP digital_noise_floor "
                    "must be greater than 0."
                )
            )


# ======================================================================
# CLASSIFICATION CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ClassificationConfig:
    """
    Acoustic-classification subsystem configuration.

    Current operational backend:

        heuristic

    Reserved future backends:

        pretrained
        birdnet
        ensemble
    """

    enabled: bool = (
        True
    )

    backend: str = (
        "heuristic"
    )

    fallback_to_heuristic: bool = (
        True
    )

    provide_model_audio: bool = (
        True
    )

    model_path: Path | None = (
        None
    )

    labels_path: Path | None = (
        None
    )

    model_sample_rate: int | None = (
        None
    )

    top_k: int = (
        5
    )

    model_min_confidence: float = (
        0.20
    )

    latitude: float | None = (
        None
    )

    longitude: float | None = (
        None
    )

    week: int | None = (
        None
    )

    use_geo_filter: bool = (
        False
    )

    geo_min_confidence: float = (
        0.03
    )

    taxonomy_path: Path | None = (
        None
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate classification configuration.

        Model files are intentionally not required to exist while the
        heuristic backend is active.
        """

        if not isinstance(
            self.enabled,
            bool,
        ):

            raise TypeError(
                (
                    "Classification enabled "
                    "must be bool."
                )
            )

        if not isinstance(
            self.backend,
            str,
        ):

            raise TypeError(
                (
                    "Classification backend "
                    "must be a string."
                )
            )

        backend = (
            self.backend
            .strip()
            .lower()
        )

        if not (
            backend
        ):

            raise ValueError(
                (
                    "Classification backend "
                    "cannot be empty."
                )
            )

        supported_backends = {
            "heuristic",
            "pretrained",
            "birdnet",
            "ensemble",
        }

        if (
            backend
            not in supported_backends
        ):

            raise ValueError(
                (
                    "Unsupported classification "
                    f"backend '{self.backend}'. "
                    "Supported values: "
                    f"{sorted(supported_backends)}"
                )
            )

        if not isinstance(
            self.fallback_to_heuristic,
            bool,
        ):

            raise TypeError(
                (
                    "Classification "
                    "fallback_to_heuristic "
                    "must be bool."
                )
            )

        if not isinstance(
            self.provide_model_audio,
            bool,
        ):

            raise TypeError(
                (
                    "Classification "
                    "provide_model_audio "
                    "must be bool."
                )
            )

        if (
            self.model_path
            is not None
            and not isinstance(
                self.model_path,
                Path,
            )
        ):

            raise TypeError(
                (
                    "Classification model_path "
                    "must be Path or None."
                )
            )

        if (
            self.labels_path
            is not None
            and not isinstance(
                self.labels_path,
                Path,
            )
        ):

            raise TypeError(
                (
                    "Classification labels_path "
                    "must be Path or None."
                )
            )

        if (
            self.model_sample_rate
            is not None
        ):

            _require_positive_int(
                self.model_sample_rate,
                name=
                    (
                        "Classification "
                        "model_sample_rate"
                    ),
            )

        _require_positive_int(
            self.top_k,
            name=
                "Classification top_k",
        )

        minimum_confidence = (
            _require_finite(
                self.model_min_confidence,
                name=
                    (
                        "Classification "
                        "model_min_confidence"
                    ),
            )
        )

        if not (
            0.0
            <= minimum_confidence
            <= 1.0
        ):

            raise ValueError(
                (
                    "Classification "
                    "model_min_confidence "
                    "must be between 0 and 1."
                )
            )

        if not isinstance(
            self.use_geo_filter,
            bool,
        ):

            raise TypeError(
                (
                    "Classification "
                    "use_geo_filter "
                    "must be bool."
                )
            )

        if self.latitude is not None:
            lat = _require_finite(
                self.latitude,
                name="Classification latitude",
            )
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Classification latitude must be in [-90, 90], got {lat}."
                )

        if self.longitude is not None:
            lon = _require_finite(
                self.longitude,
                name="Classification longitude",
            )
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Classification longitude must be in [-180, 180], got {lon}."
                )

        if self.week is not None:
            _require_positive_int(
                self.week,
                name="Classification week",
            )
            if not (1 <= self.week <= 48):
                raise ValueError(
                    f"Classification week must be between 1 and 48, got {self.week}."
                )

        geo_min_conf = _require_finite(
            self.geo_min_confidence,
            name="Classification geo_min_confidence",
        )
        if not (0.0 <= geo_min_conf <= 1.0):
            raise ValueError(
                f"Classification geo_min_confidence must be in [0, 1], got {geo_min_conf}."
            )

        if (
            self.taxonomy_path is not None
            and not isinstance(self.taxonomy_path, Path)
        ):
            raise TypeError("Classification taxonomy_path must be Path or None.")


# ======================================================================
# PERSISTENCE CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PersistenceConfig:
    """
    Database and event-file persistence settings.
    """

    database_path: Path = (
        Path(
            "data/database/events.db"
        )
    )

    events_dir: Path = (
        Path(
            "data/events"
        )
    )

    save_event_wav: bool = (
        True
    )

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

            raise TypeError(
                (
                    "Persistence database_path "
                    "must be pathlib.Path."
                )
            )

        if not isinstance(
            self.events_dir,
            Path,
        ):

            raise TypeError(
                (
                    "Persistence events_dir "
                    "must be pathlib.Path."
                )
            )

        if (
            not self.database_path.name
        ):

            raise ValueError(
                (
                    "Persistence database_path "
                    "must include a database filename."
                )
            )

        if not isinstance(
            self.save_event_wav,
            bool,
        ):

            raise TypeError(
                (
                    "Persistence save_event_wav "
                    "must be bool."
                )
            )


# ======================================================================
# RESEARCH ANALYTICS CONFIGURATION
# ======================================================================


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

    enabled: bool = (
        True
    )

    bucket_seconds: int = (
        3600
    )

    environmental_node_id: int | None = (
        1
    )

    environmental_alpha: float = (
        0.05
    )

    environmental_min_samples: int = (
        5
    )

    neutral_threshold: float = (
        0.05
    )

    cell_size_m: float = (
        0.25
    )

    max_grid_cells: int = (
        10_000
    )

    max_transition_gap_s: float | None = (
        300.0
    )

    same_class_transitions_only: bool = (
        False
    )

    min_activity_events: int = (
        5
    )

    min_localized_events: int = (
        5
    )

    min_transitions: int = (
        3
    )

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

            raise TypeError(
                (
                    "Analytics enabled "
                    "must be bool."
                )
            )

        _require_positive_int(
            self.bucket_seconds,
            name=
                "Analytics bucket_seconds",
        )

        if (
            self.environmental_node_id
            is not None
        ):

            _require_positive_int(
                self.environmental_node_id,
                name=
                    (
                        "Analytics "
                        "environmental_node_id"
                    ),
            )

            if (
                self.environmental_node_id
                > UINT8_MAX
            ):

                raise ValueError(
                    (
                        "Analytics "
                        "environmental_node_id "
                        "must fit Protocol-v4 "
                        "uint8 nodeId."
                    )
                )

        alpha = (
            _require_finite(
                self.environmental_alpha,
                name=
                    (
                        "Analytics "
                        "environmental_alpha"
                    ),
            )
        )

        if not (
            0.0
            < alpha
            < 1.0
        ):

            raise ValueError(
                (
                    "Analytics "
                    "environmental_alpha "
                    "must be in (0, 1)."
                )
            )

        _require_positive_int(
            self.environmental_min_samples,
            name=
                (
                    "Analytics "
                    "environmental_min_samples"
                ),
        )

        if (
            self.environmental_min_samples
            < 3
        ):

            raise ValueError(
                (
                    "Analytics "
                    "environmental_min_samples "
                    "must be at least 3."
                )
            )

        neutral = (
            _require_finite(
                self.neutral_threshold,
                name=
                    (
                        "Analytics "
                        "neutral_threshold"
                    ),
            )
        )

        if not (
            0.0
            <= neutral
            <= 1.0
        ):

            raise ValueError(
                (
                    "Analytics "
                    "neutral_threshold "
                    "must be in [0, 1]."
                )
            )

        cell_size = (
            _require_finite(
                self.cell_size_m,
                name=
                    "Analytics cell_size_m",
            )
        )

        if (
            cell_size
            <= 0.0
        ):

            raise ValueError(
                (
                    "Analytics cell_size_m "
                    "must be greater than 0."
                )
            )

        _require_positive_int(
            self.max_grid_cells,
            name=
                "Analytics max_grid_cells",
        )

        if (
            self.max_transition_gap_s
            is not None
        ):

            transition_gap = (
                _require_finite(
                    self.max_transition_gap_s,
                    name=
                        (
                            "Analytics "
                            "max_transition_gap_s"
                        ),
                )
            )

            if (
                transition_gap
                <= 0.0
            ):

                raise ValueError(
                    (
                        "Analytics "
                        "max_transition_gap_s "
                        "must be greater than 0 "
                        "or None."
                    )
                )

        if not isinstance(
            self.same_class_transitions_only,
            bool,
        ):

            raise TypeError(
                (
                    "Analytics "
                    "same_class_transitions_only "
                    "must be bool."
                )
            )

        _require_positive_int(
            self.min_activity_events,
            name=
                (
                    "Analytics "
                    "min_activity_events"
                ),
        )

        _require_positive_int(
            self.min_localized_events,
            name=
                (
                    "Analytics "
                    "min_localized_events"
                ),
        )

        _require_positive_int(
            self.min_transitions,
            name=
                (
                    "Analytics "
                    "min_transitions"
                ),
        )


# ======================================================================
# DASHBOARD CONFIGURATION
# ======================================================================


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

    enabled: bool = (
        True
    )

    page_title: str = (
        "Wildlife Soundscape Monitor"
    )

    auto_refresh: bool = (
        True
    )

    refresh_interval_s: float = (
        2.0
    )

    recent_events_limit: int = (
        50
    )

    session_list_limit: int = (
        100
    )

    max_plot_points: int = (
        5000
    )

    max_audio_preview_s: float = (
        30.0
    )

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

            raise TypeError(
                (
                    "Dashboard enabled "
                    "must be bool."
                )
            )

        if not isinstance(
            self.page_title,
            str,
        ):

            raise TypeError(
                (
                    "Dashboard page_title "
                    "must be a string."
                )
            )

        if not (
            self.page_title.strip()
        ):

            raise ValueError(
                (
                    "Dashboard page_title "
                    "cannot be empty."
                )
            )

        if not isinstance(
            self.auto_refresh,
            bool,
        ):

            raise TypeError(
                (
                    "Dashboard auto_refresh "
                    "must be bool."
                )
            )

        refresh_interval = (
            _require_finite(
                self.refresh_interval_s,
                name=
                    (
                        "Dashboard "
                        "refresh_interval_s"
                    ),
            )
        )

        if (
            refresh_interval
            <= 0.0
        ):

            raise ValueError(
                (
                    "Dashboard "
                    "refresh_interval_s "
                    "must be greater than 0."
                )
            )

        _require_positive_int(
            self.recent_events_limit,
            name=
                (
                    "Dashboard "
                    "recent_events_limit"
                ),
        )

        _require_positive_int(
            self.session_list_limit,
            name=
                (
                    "Dashboard "
                    "session_list_limit"
                ),
        )

        _require_positive_int(
            self.max_plot_points,
            name=
                (
                    "Dashboard "
                    "max_plot_points"
                ),
        )

        max_audio_preview = (
            _require_finite(
                self.max_audio_preview_s,
                name=
                    (
                        "Dashboard "
                        "max_audio_preview_s"
                    ),
            )
        )

        if (
            max_audio_preview
            <= 0.0
        ):

            raise ValueError(
                (
                    "Dashboard "
                    "max_audio_preview_s "
                    "must be greater than 0."
                )
            )


# ======================================================================
# APPLICATION CONFIGURATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class AppConfig:
    """
    Root configuration object for the complete laptop-side system.

    AppConfig performs validation that spans multiple subsystems.
    """

    network: NetworkConfig = field(
        default_factory=
            NetworkConfig
    )

    audio: AudioConfig = field(
        default_factory=
            AudioConfig
    )

    localization: LocalizationConfig = field(
        default_factory=
            LocalizationConfig
    )

    detection: EventDetectionConfig = field(
        default_factory=
            EventDetectionConfig
    )

    dsp: DSPConfig = field(
        default_factory=
            DSPConfig
    )

    classification: ClassificationConfig = field(
        default_factory=
            ClassificationConfig
    )

    persistence: PersistenceConfig = field(
        default_factory=
            PersistenceConfig
    )

    analytics: AnalyticsConfig = field(
        default_factory=
            AnalyticsConfig
    )

    dashboard: DashboardConfig = field(
        default_factory=
            DashboardConfig
    )

    # ------------------------------------------------------------------
    # EXPECTED NODES
    # ------------------------------------------------------------------

    expected_nodes: frozenset[
        int
    ] = frozenset(
        {
            1,
            2,
            3,
        }
    )

    # ------------------------------------------------------------------
    # CONTINUOUS SESSION RECORDINGS
    # ------------------------------------------------------------------

    recordings_dir: Path = (
        Path(
            "data/recordings"
        )
    )

    # ------------------------------------------------------------------
    # CLI STATUS
    # ------------------------------------------------------------------

    print_status_every_s: float = (
        10.0
    )

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

        self.dsp.validate(
            self.audio.sample_rate
        )

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

            raise TypeError(
                (
                    "expected_nodes must be "
                    "a frozenset of node IDs."
                )
            )

        if (
            len(
                self.expected_nodes
            )
            < 3
        ):

            raise ValueError(
                (
                    "At least three acoustic nodes "
                    "are required for the current "
                    "2-D TDOA localization design."
                )
            )

        for node_id in (
            self.expected_nodes
        ):

            if isinstance(
                node_id,
                bool,
            ):

                raise TypeError(
                    (
                        "Expected node IDs "
                        "must be integers."
                    )
                )

            if not isinstance(
                node_id,
                int,
            ):

                raise TypeError(
                    (
                        "Expected node IDs "
                        "must be integers."
                    )
                )

            if not (
                1
                <= node_id
                <= UINT8_MAX
            ):

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
            sample_rate=
                self.audio.sample_rate,

            expected_nodes=
                self.expected_nodes,
        )

        # ==============================================================
        # EVENT DETECTOR
        # ==============================================================

        self.detection.validate(
            expected_node_count=
                len(
                    self.expected_nodes
                )
        )

        # ==============================================================
        # ENVIRONMENTAL ANALYTICS NODE
        # ==============================================================

        if (
            self.analytics.environmental_node_id
            is not None
            and self.analytics.environmental_node_id
            not in self.expected_nodes
        ):

            raise ValueError(
                (
                    "Analytics environmental_node_id "
                    "must be one of expected_nodes."
                )
            )

        # ==============================================================
        # AUDIO PAYLOAD vs NETWORK PAYLOAD LIMIT
        # ==============================================================

        if (
            self.audio.bytes_per_block
            > self.network.max_payload_bytes
        ):

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
            self.localization
            .max_alignment_search_samples
            > self.audio
            .sync_tolerance_samples
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

        detector_release_s = (
            self.detection.release_blocks
            * self.audio.block_duration_s
        )

        required_event_retention_s = (
            self.detection.pre_pad_s
            + self.detection.max_event_s
            + self.detection.post_pad_s
            + detector_release_s
            + (
                2.0
                * self.audio.block_duration_s
            )
        )

        if (
            self.audio.buffer_seconds
            < required_event_retention_s
        ):

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
            self.localization.window_samples
            / self.audio.sample_rate
        )

        if (
            localization_window_s
            > self.audio.buffer_seconds
        ):

            raise ValueError(
                (
                    "Localization window is longer "
                    "than the retained audio buffer."
                )
            )

        # ==============================================================
        # DSP EVENT MINIMUM vs DETECTOR MINIMUM
        # ==============================================================

        minimum_detected_samples = int(
            math.ceil(
                self.detection.min_event_ms
                / 1000.0
                * self.audio.sample_rate
            )
        )

        if (
            self.dsp.minimum_event_samples
            > minimum_detected_samples
        ):

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

            raise TypeError(
                (
                    "recordings_dir must be "
                    "pathlib.Path."
                )
            )

        # ==============================================================
        # STATUS INTERVAL
        # ==============================================================

        status_interval = (
            _require_finite(
                self.print_status_every_s,
                name=
                    "print_status_every_s",
            )
        )

        if (
            status_interval
            <= 0
        ):

            raise ValueError(
                (
                    "print_status_every_s must "
                    "be greater than 0."
                )
            )


# ======================================================================
# GLOBAL APPLICATION CONFIGURATION
# ======================================================================


CONFIG = (
    AppConfig()
)