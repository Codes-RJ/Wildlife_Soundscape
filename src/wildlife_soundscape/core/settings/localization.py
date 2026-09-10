from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from .validation import _require_finite, _require_positive_int


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

    enabled: bool = False

    # ------------------------------------------------------------------
    # CALIBRATION REFERENCE
    # ------------------------------------------------------------------
    #
    # Bias for this node is fixed at exactly zero.
    #
    # Pairwise bias differences are independent of which node is chosen
    # as this reference.
    # ------------------------------------------------------------------

    reference_node: int = 1

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
    ] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # CALIBRATION PROVENANCE
    # ------------------------------------------------------------------

    result_path: Path = Path("data/calibration/tdoa_calibration.json")

    generated_at: str | None = None

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

    rms_pair_consistency_error_s: float | None = None

    # Maximum accepted consistency error expressed in AUDIO samples.
    #
    # At 48 kHz:
    #
    #     1 sample ≈ 20.83 microseconds
    # ------------------------------------------------------------------

    max_consistency_error_samples: float = 1.0

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

        if node_a == node_b:
            raise ValueError(("TDOA calibration requires two different node IDs."))

        if not (self.enabled):
            return 0.0

        try:
            bias_a = self.node_biases_s[node_a]

            bias_b = self.node_biases_s[node_b]

        except KeyError as exc:
            raise KeyError(
                (f"Missing TDOA calibration bias for node {exc.args[0]}.")
            ) from exc

        return bias_b - bias_a

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
            name=("TDOA calibration sample_rate"),
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

        measured = _require_finite(
            measured_tdoa_s,
            name=("Measured TDOA"),
        )

        return measured - self.pair_offset_s(
            node_a,
            node_b,
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

        if not self.enabled or len(self.node_biases_s) < 2:
            return 0.0

        biases = tuple(float(value) for value in self.node_biases_s.values())

        return max(biases) - min(biases)

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
            name=("TDOA calibration sample_rate"),
        )

        return self.maximum_absolute_pair_offset_s() * sample_rate

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
            raise TypeError(("TDOA calibration enabled must be bool."))

        _require_positive_int(
            sample_rate,
            name=("TDOA calibration sample_rate"),
        )

        _require_positive_int(
            self.reference_node,
            name=("TDOA calibration reference_node"),
        )

        if self.reference_node not in expected_nodes:
            raise ValueError(
                ("TDOA calibration reference_node must be one of expected_nodes.")
            )

        # ==============================================================
        # RESULT PATH
        # ==============================================================

        if not isinstance(
            self.result_path,
            Path,
        ):
            raise TypeError(("TDOA calibration result_path must be pathlib.Path."))

        if not (self.result_path.name):
            raise ValueError(("TDOA calibration result_path must include a filename."))

        # ==============================================================
        # TIMESTAMP / PROVENANCE
        # ==============================================================

        if self.generated_at is not None:
            if not isinstance(
                self.generated_at,
                str,
            ):
                raise TypeError(
                    ("TDOA calibration generated_at must be a string or None.")
                )

            if not (self.generated_at.strip()):
                raise ValueError(
                    ("TDOA calibration generated_at cannot be empty when provided.")
                )

        # ==============================================================
        # NODE BIASES
        # ==============================================================

        if not isinstance(
            self.node_biases_s,
            dict,
        ):
            raise TypeError(("TDOA calibration node_biases_s must be a dictionary."))

        normalized_biases: dict[
            int,
            float,
        ] = {}

        for node_id, raw_bias in self.node_biases_s.items():
            if isinstance(
                node_id,
                bool,
            ) or not isinstance(
                node_id,
                int,
            ):
                raise TypeError(
                    ("TDOA calibration node-bias keys must be integer node IDs.")
                )

            if node_id not in expected_nodes:
                raise ValueError(
                    (f"TDOA calibration contains unexpected node {node_id}.")
                )

            bias = _require_finite(
                raw_bias,
                name=(f"TDOA calibration bias for node {node_id}"),
            )

            normalized_biases[node_id] = bias

        # ==============================================================
        # CONSISTENCY THRESHOLD
        # ==============================================================

        max_consistency_samples = _require_finite(
            self.max_consistency_error_samples,
            name=("TDOA calibration max_consistency_error_samples"),
        )

        if max_consistency_samples <= 0.0:
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

        consistency_error_s: float | None = None

        if self.rms_pair_consistency_error_s is not None:
            consistency_error_s = _require_finite(
                self.rms_pair_consistency_error_s,
                name=("TDOA calibration rms_pair_consistency_error_s"),
            )

            if consistency_error_s < 0.0:
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

        if not (self.enabled):
            return

        missing_biases = expected_nodes - set(normalized_biases)

        if missing_biases:
            raise ValueError(
                (
                    "Enabled TDOA calibration is "
                    "missing timing biases for "
                    f"node(s): {sorted(missing_biases)}."
                )
            )

        reference_bias = normalized_biases[self.reference_node]

        if abs(reference_bias) > 1e-12:
            raise ValueError(("TDOA calibration reference-node bias must be zero."))

        if consistency_error_s is None:
            raise ValueError(
                (
                    "Enabled TDOA calibration requires "
                    "rms_pair_consistency_error_s from "
                    "the calibration fit."
                )
            )

        consistency_error_samples = consistency_error_s * sample_rate

        if consistency_error_samples > max_consistency_samples:
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

    reference_node: int = 1

    # ------------------------------------------------------------------
    # TDOA WINDOW
    # ------------------------------------------------------------------

    window_samples: int = 8192

    # Maximum coarse sampleIndex correction considered before waveform
    # GCC-PHAT estimation.
    max_alignment_search_samples: int = 10

    # Fractional-delay interpolation factor used by GCC-PHAT.
    interpolation: int = 8

    # Frequency-domain GCC weighting. A value of 1.0 is the established
    # PHAT baseline; lower values retain progressively more magnitude
    # information for controlled research comparisons.
    gcc_beta: float = 1.0

    # Optional FFT-bin mask applied inside GCC after the localization
    # pre-filter. None preserves the established full-spectrum behavior.
    gcc_frequency_band_hz: (
        tuple[
            float,
            float,
        ]
        | None
    ) = None

    # ------------------------------------------------------------------
    # QUALITY THRESHOLDS
    # ------------------------------------------------------------------

    min_peak_ratio: float = 1.10

    min_rms: float = 50.0

    # ------------------------------------------------------------------
    # SPEED OF SOUND
    # ------------------------------------------------------------------

    speed_of_sound_mps: float = 343.0

    use_environmental_speed: bool = True

    # ------------------------------------------------------------------
    # LOCALIZATION SIGNAL CONDITIONING
    # ------------------------------------------------------------------

    bandpass_enabled: bool = True

    bandpass_low_hz: float = 200.0

    bandpass_high_hz: float = 12_000.0

    bandpass_order: int = 4

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
        default_factory=TDOACalibrationConfig
    )

    # ------------------------------------------------------------------
    # SOLVER
    # ------------------------------------------------------------------

    constrain_to_array_bounds: bool = False

    bounds_margin_m: float = 0.5

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

        if self.reference_node not in expected_nodes:
            raise ValueError(
                ("Localization reference_node must be one of expected_nodes.")
            )

        # ==============================================================
        # NODE COORDINATES
        # ==============================================================

        missing_positions = expected_nodes - set(self.node_positions.keys())

        if missing_positions:
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

        for node_id in sorted(expected_nodes):
            position = self.node_positions[node_id]

            try:
                x_raw, y_raw = position

            except Exception as exc:
                raise ValueError(
                    (
                        "Localization position for "
                        f"node {node_id} must contain "
                        "exactly two coordinates."
                    )
                ) from exc

            x = _require_finite(
                x_raw,
                name=(f"Localization node {node_id} x"),
            )

            y = _require_finite(
                y_raw,
                name=(f"Localization node {node_id} y"),
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

        if len(set(coordinates)) != len(coordinates):
            raise ValueError(("Localization microphones must have unique coordinates."))

        # --------------------------------------------------------------
        # NON-COLLINEAR ARRAY
        # --------------------------------------------------------------

        non_collinear = False

        coordinate_count = len(coordinates)

        for i in range(coordinate_count - 2):
            x1, y1 = coordinates[i]

            for j in range(
                i + 1,
                coordinate_count - 1,
            ):
                x2, y2 = coordinates[j]

                for k in range(
                    j + 1,
                    coordinate_count,
                ):
                    x3, y3 = coordinates[k]

                    twice_area = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))

                    if twice_area > 1e-12:
                        non_collinear = True

                        break

                if non_collinear:
                    break

            if non_collinear:
                break

        if not (non_collinear):
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
            name="Localization window_samples",
        )

        if isinstance(
            self.max_alignment_search_samples,
            bool,
        ) or not isinstance(
            self.max_alignment_search_samples,
            int,
        ):
            raise TypeError(
                ("Localization max_alignment_search_samples must be an integer.")
            )

        if self.max_alignment_search_samples < 0:
            raise ValueError(
                ("Localization max_alignment_search_samples cannot be negative.")
            )

        _require_positive_int(
            self.interpolation,
            name="Localization interpolation",
        )

        gcc_beta = _require_finite(
            self.gcc_beta,
            name="Localization gcc_beta",
        )

        if not (0.0 <= gcc_beta <= 1.0):
            raise ValueError(("Localization gcc_beta must be in [0, 1]."))

        if self.gcc_frequency_band_hz is not None:
            if (
                not isinstance(
                    self.gcc_frequency_band_hz,
                    tuple,
                )
                or len(self.gcc_frequency_band_hz) != 2
            ):
                raise TypeError(
                    (
                        "Localization "
                        "gcc_frequency_band_hz must "
                        "be a (low_hz, high_hz) tuple "
                        "or None."
                    )
                )

            gcc_low_hz = _require_finite(
                self.gcc_frequency_band_hz[0],
                name=("Localization GCC frequency-band lower bound"),
            )

            gcc_high_hz = _require_finite(
                self.gcc_frequency_band_hz[1],
                name=("Localization GCC frequency-band upper bound"),
            )

            if not (0.0 <= gcc_low_hz < gcc_high_hz <= sample_rate / 2.0):
                raise ValueError(
                    (
                        "Localization GCC frequency "
                        "band must satisfy "
                        "0 <= low < high <= Nyquist."
                    )
                )

        # ==============================================================
        # QUALITY
        # ==============================================================

        peak_ratio = _require_finite(
            self.min_peak_ratio,
            name="Localization min_peak_ratio",
        )

        if peak_ratio < 1.0:
            raise ValueError(("Localization min_peak_ratio must be at least 1.0."))

        minimum_rms = _require_finite(
            self.min_rms,
            name="Localization min_rms",
        )

        if minimum_rms < 0:
            raise ValueError(("Localization min_rms cannot be negative."))

        # ==============================================================
        # SPEED OF SOUND
        # ==============================================================

        sound_speed = _require_finite(
            self.speed_of_sound_mps,
            name="Localization speed_of_sound_mps",
        )

        if sound_speed <= 0:
            raise ValueError(
                ("Localization fallback speed_of_sound_mps must be greater than 0.")
            )

        if not isinstance(
            self.use_environmental_speed,
            bool,
        ):
            raise TypeError(("Localization use_environmental_speed must be bool."))

        # ==============================================================
        # BANDPASS
        # ==============================================================

        nyquist = sample_rate / 2.0

        if self.bandpass_enabled:
            low_hz = _require_finite(
                self.bandpass_low_hz,
                name=("Localization bandpass_low_hz"),
            )

            high_hz = _require_finite(
                self.bandpass_high_hz,
                name=("Localization bandpass_high_hz"),
            )

            if not (0.0 < low_hz < high_hz < nyquist):
                raise ValueError(
                    (
                        "Localization bandpass "
                        "frequencies must satisfy "
                        "0 < low < high < Nyquist."
                    )
                )

        _require_positive_int(
            self.bandpass_order,
            name="Localization bandpass_order",
        )

        # ==============================================================
        # TDOA CALIBRATION
        # ==============================================================

        self.tdoa_calibration.validate(
            sample_rate=sample_rate,
            expected_nodes=expected_nodes,
        )

        # ==============================================================
        # SOLVER
        # ==============================================================

        if not isinstance(
            self.constrain_to_array_bounds,
            bool,
        ):
            raise TypeError(("Localization constrain_to_array_bounds must be bool."))

        margin = _require_finite(
            self.bounds_margin_m,
            name="Localization bounds_margin_m",
        )

        if margin < 0:
            raise ValueError(("Localization bounds_margin_m cannot be negative."))
