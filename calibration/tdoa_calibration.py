"""
TDOA timing calibration.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Estimate systematic timing bias between synchronized acoustic nodes
using controlled recordings from known source positions.

The localization system uses the TDOA convention:

    TDOA(A, B)
        =
    arrival_time_B - arrival_time_A

Therefore:

    positive TDOA
        ->
    the signal arrived later at Node B

and the theoretical propagation delay is:

    expected_tdoa
        =
    (distance_B - distance_A) / speed_of_sound


Calibration model
-----------------
For a controlled calibration observation:

    residual
        =
    measured_tdoa - expected_tdoa

A persistent residual can indicate channel-specific timing bias.

For nodes A and B:

    residual_AB
        ≈
    bias_B - bias_A

Using observations from several microphone pairs, this module estimates
a coherent node-delay model using weighted least squares.

One reference node is fixed to:

    bias_reference = 0

For this project's three-node configuration, Node 1 will normally be
used as the reference node.


Why node-bias fitting matters
-----------------------------
Calibrating every microphone pair independently can create an
internally inconsistent result.

For three nodes, physically consistent timing offsets should satisfy:

    offset_12 + offset_23 ≈ offset_13

Representing calibration as per-node timing biases automatically
enforces this relationship.


Recommended calibration procedure
---------------------------------
Use several known source positions distributed across the test area.

At every reference position:

    1. Produce a short impulsive or broadband calibration sound.
    2. Capture synchronized audio from all three nodes.
    3. Estimate pairwise TDOA using the same GCC-PHAT pipeline used
       during normal localization.
    4. Record:
           known source position
           measured pair TDOA
           speed of sound
    5. Repeat multiple times.

Using several positions helps distinguish a genuine fixed channel delay
from errors caused by:

    inaccurate microphone coordinates
    source-position error
    acoustic reflections
    environmental variation
    GCC-PHAT estimation noise


Important
---------
Calibration should correct repeatable timing bias.

It should NOT be used to hide:

    incorrect microphone coordinates
    multipath/reflection problems
    bad synchronization
    dropped samples
    incorrect speed-of-sound assumptions
    poor source-position measurements

Those problems should be corrected independently.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


from collections import defaultdict

from dataclasses import (
    dataclass,
    replace,
)

from datetime import (
    datetime,
    timezone,
)

from math import (
    isfinite,
    sqrt,
)

from numbers import (
    Integral,
    Real,
)

from typing import (
    Iterable,
    Mapping,
    Sequence,
)


# ======================================================================
# THIRD PARTY
# ======================================================================


import numpy as np


# ======================================================================
# TYPE ALIASES
# ======================================================================


NodeId = int

NodePair = tuple[
    NodeId,
    NodeId,
]

Position = tuple[
    float,
    ...,
]


# ======================================================================
# DEFAULTS
# ======================================================================


DEFAULT_SAMPLE_RATE = (
    48_000
)


DEFAULT_SPEED_OF_SOUND_MPS = (
    343.0
)


DEFAULT_REFERENCE_NODE = (
    1
)


DEFAULT_MIN_OBSERVATIONS_PER_PAIR = (
    3
)


DEFAULT_MODIFIED_Z_THRESHOLD = (
    3.5
)


MAD_SCALE_FACTOR = (
    0.6744897501960817
)


FLOAT_EPSILON = (
    1e-15
)


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _positive_int(
    value: object,
    *,
    name: str,
) -> int:
    """
    Validate a strictly positive integer.
    """

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Integral,
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    result = int(
        value
    )

    if result <= 0:
        raise ValueError(
            f"{name} must be > 0."
        )

    return result


def _node_id(
    value: object,
    *,
    name: str,
) -> int:
    """
    Validate an acoustic node identifier.
    """

    return _positive_int(
        value,
        name=name,
    )


def _finite_float(
    value: object,
    *,
    name: str,
) -> float:
    """
    Validate and normalize a finite real number.
    """

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Real,
        )
    ):
        raise TypeError(
            f"{name} must be a real number."
        )

    result = float(
        value
    )

    if not isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _positive_float(
    value: object,
    *,
    name: str,
) -> float:
    """
    Validate a finite strictly positive real number.
    """

    result = _finite_float(
        value,
        name=name,
    )

    if result <= 0.0:
        raise ValueError(
            f"{name} must be > 0."
        )

    return result


def _normalize_position(
    value: Sequence[Real],
    *,
    name: str,
) -> Position:
    """
    Normalize a 2-D or 3-D Cartesian position.
    """

    if isinstance(
        value,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            f"{name} must be a numeric position."
        )

    try:
        raw_values = tuple(
            value
        )

    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable numeric position."
        ) from exc

    if len(
        raw_values
    ) not in (
        2,
        3,
    ):
        raise ValueError(
            f"{name} must contain 2 or 3 coordinates."
        )

    normalized = tuple(
        _finite_float(
            coordinate,
            name=f"{name}[{index}]",
        )
        for index, coordinate
        in enumerate(
            raw_values
        )
    )

    return normalized


# ======================================================================
# NODE PAIR NORMALIZATION
# ======================================================================


def canonical_pair(
    node_a: int,
    node_b: int,
) -> NodePair:
    """
    Return a deterministic node-pair representation.

    Canonical pairs always satisfy:

        node_a < node_b
    """

    a = _node_id(
        node_a,
        name="node_a",
    )

    b = _node_id(
        node_b,
        name="node_b",
    )

    if a == b:
        raise ValueError(
            "A TDOA pair must contain two different nodes."
        )

    if a < b:
        return (
            a,
            b,
        )

    return (
        b,
        a,
    )


def orient_pair_value(
    node_a: int,
    node_b: int,
    value: float,
) -> tuple[
    NodePair,
    float,
]:
    """
    Convert a pair-oriented value into canonical pair orientation.

    Example
    -------
    Given:

        TDOA(2, 1) = -0.0004

    canonical orientation is:

        TDOA(1, 2) = +0.0004
    """

    a = _node_id(
        node_a,
        name="node_a",
    )

    b = _node_id(
        node_b,
        name="node_b",
    )

    numeric_value = _finite_float(
        value,
        name="value",
    )

    pair = canonical_pair(
        a,
        b,
    )

    if (
        a,
        b,
    ) == pair:
        return (
            pair,
            numeric_value,
        )

    return (
        pair,
        -numeric_value,
    )


# ======================================================================
# DISTANCE
# ======================================================================


def euclidean_distance_m(
    first: Sequence[Real],
    second: Sequence[Real],
) -> float:
    """
    Calculate Euclidean distance between two Cartesian positions.
    """

    a = _normalize_position(
        first,
        name="first",
    )

    b = _normalize_position(
        second,
        name="second",
    )

    if len(
        a
    ) != len(
        b
    ):
        raise ValueError(
            "Position dimensions must match."
        )

    return sqrt(
        sum(
            (
                coordinate_a
                - coordinate_b
            )
            ** 2
            for coordinate_a, coordinate_b
            in zip(
                a,
                b,
            )
        )
    )


# ======================================================================
# EXPECTED TDOA
# ======================================================================


def expected_tdoa_seconds(
    source_position_m: Sequence[Real],
    node_a_position_m: Sequence[Real],
    node_b_position_m: Sequence[Real],
    *,
    speed_of_sound_mps: float = DEFAULT_SPEED_OF_SOUND_MPS,
) -> float:
    """
    Calculate ideal propagation TDOA for a known source position.

    Convention
    ----------
    Returned value is:

        arrival_B - arrival_A

    which equals:

        (distance_B - distance_A) / c
    """

    speed = _positive_float(
        speed_of_sound_mps,
        name="speed_of_sound_mps",
    )

    source = _normalize_position(
        source_position_m,
        name="source_position_m",
    )

    node_a = _normalize_position(
        node_a_position_m,
        name="node_a_position_m",
    )

    node_b = _normalize_position(
        node_b_position_m,
        name="node_b_position_m",
    )

    if not (
        len(
            source
        )
        == len(
            node_a
        )
        == len(
            node_b
        )
    ):
        raise ValueError(
            "Source and node positions must have matching dimensions."
        )

    distance_a = euclidean_distance_m(
        source,
        node_a,
    )

    distance_b = euclidean_distance_m(
        source,
        node_b,
    )

    return (
        distance_b
        - distance_a
    ) / speed


# ======================================================================
# KNOWN-SOURCE MEASUREMENT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class KnownSourceMeasurement:
    """
    One controlled TDOA measurement.

    `measured_tdoa_s` uses:

        arrival_B - arrival_A

    for `node_a`, `node_b`.

    Pair orientation is normalized automatically.
    """

    node_a: int

    node_b: int

    source_position_m: Position

    measured_tdoa_s: float

    speed_of_sound_mps: float = (
        DEFAULT_SPEED_OF_SOUND_MPS
    )

    quality_weight: float = (
        1.0
    )

    def __post_init__(
        self,
    ) -> None:

        original_a = _node_id(
            self.node_a,
            name="node_a",
        )

        original_b = _node_id(
            self.node_b,
            name="node_b",
        )

        pair = canonical_pair(
            original_a,
            original_b,
        )

        measured = _finite_float(
            self.measured_tdoa_s,
            name="measured_tdoa_s",
        )

        if (
            original_a,
            original_b,
        ) != pair:
            measured = (
                -measured
            )

        source = _normalize_position(
            self.source_position_m,
            name="source_position_m",
        )

        speed = _positive_float(
            self.speed_of_sound_mps,
            name="speed_of_sound_mps",
        )

        weight = _positive_float(
            self.quality_weight,
            name="quality_weight",
        )

        object.__setattr__(
            self,
            "node_a",
            pair[0],
        )

        object.__setattr__(
            self,
            "node_b",
            pair[1],
        )

        object.__setattr__(
            self,
            "source_position_m",
            source,
        )

        object.__setattr__(
            self,
            "measured_tdoa_s",
            measured,
        )

        object.__setattr__(
            self,
            "speed_of_sound_mps",
            speed,
        )

        object.__setattr__(
            self,
            "quality_weight",
            weight,
        )


# ======================================================================
# NORMALIZED CALIBRATION OBSERVATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TDOACalibrationObservation:
    """
    Known-source measurement after theoretical TDOA calculation.
    """

    node_a: int

    node_b: int

    source_position_m: Position

    measured_tdoa_s: float

    expected_tdoa_s: float

    residual_s: float

    speed_of_sound_mps: float

    quality_weight: float

    def __post_init__(
        self,
    ) -> None:

        pair = canonical_pair(
            self.node_a,
            self.node_b,
        )

        if pair != (
            self.node_a,
            self.node_b,
        ):
            raise ValueError(
                "TDOACalibrationObservation must use canonical pair order."
            )

        source = _normalize_position(
            self.source_position_m,
            name="source_position_m",
        )

        measured = _finite_float(
            self.measured_tdoa_s,
            name="measured_tdoa_s",
        )

        expected = _finite_float(
            self.expected_tdoa_s,
            name="expected_tdoa_s",
        )

        residual = _finite_float(
            self.residual_s,
            name="residual_s",
        )

        speed = _positive_float(
            self.speed_of_sound_mps,
            name="speed_of_sound_mps",
        )

        weight = _positive_float(
            self.quality_weight,
            name="quality_weight",
        )

        object.__setattr__(
            self,
            "source_position_m",
            source,
        )

        object.__setattr__(
            self,
            "measured_tdoa_s",
            measured,
        )

        object.__setattr__(
            self,
            "expected_tdoa_s",
            expected,
        )

        object.__setattr__(
            self,
            "residual_s",
            residual,
        )

        object.__setattr__(
            self,
            "speed_of_sound_mps",
            speed,
        )

        object.__setattr__(
            self,
            "quality_weight",
            weight,
        )

    def to_dict(
        self,
    ) -> dict[
        str,
        object,
    ]:
        """
        Serialize the observation.
        """

        return {
            "node_a":
                self.node_a,

            "node_b":
                self.node_b,

            "source_position_m":
                list(
                    self.source_position_m
                ),

            "measured_tdoa_s":
                self.measured_tdoa_s,

            "expected_tdoa_s":
                self.expected_tdoa_s,

            "residual_s":
                self.residual_s,

            "speed_of_sound_mps":
                self.speed_of_sound_mps,

            "quality_weight":
                self.quality_weight,
        }


# ======================================================================
# PAIR CALIBRATION
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PairTimingCalibration:
    """
    Timing-bias estimate for one canonical microphone pair.

    observed_offset_s
        Robust residual estimated directly from measurements.

    modeled_offset_s
        Offset predicted by the coherent per-node timing-bias model.

    consistency_error_s
        observed_offset_s - modeled_offset_s
    """

    node_a: int

    node_b: int

    observation_count: int

    inlier_count: int

    rejected_count: int

    observed_offset_s: float

    modeled_offset_s: float

    consistency_error_s: float

    residual_std_s: float

    residual_mad_s: float

    median_absolute_residual_s: float

    def __post_init__(
        self,
    ) -> None:

        pair = canonical_pair(
            self.node_a,
            self.node_b,
        )

        if pair != (
            self.node_a,
            self.node_b,
        ):
            raise ValueError(
                "PairTimingCalibration must use canonical pair order."
            )

        observation_count = _positive_int(
            self.observation_count,
            name="observation_count",
        )

        inlier_count = _positive_int(
            self.inlier_count,
            name="inlier_count",
        )

        if (
            isinstance(
                self.rejected_count,
                bool,
            )
            or not isinstance(
                self.rejected_count,
                Integral,
            )
        ):
            raise TypeError(
                "rejected_count must be an integer."
            )

        rejected_count = int(
            self.rejected_count
        )

        if rejected_count < 0:
            raise ValueError(
                "rejected_count must be >= 0."
            )

        if (
            inlier_count
            + rejected_count
            != observation_count
        ):
            raise ValueError(
                "inlier_count + rejected_count must equal observation_count."
            )

        object.__setattr__(
            self,
            "observation_count",
            observation_count,
        )

        object.__setattr__(
            self,
            "inlier_count",
            inlier_count,
        )

        object.__setattr__(
            self,
            "rejected_count",
            rejected_count,
        )

        for field_name in (
            "observed_offset_s",
            "modeled_offset_s",
            "consistency_error_s",
            "residual_std_s",
            "residual_mad_s",
            "median_absolute_residual_s",
        ):

            value = _finite_float(
                getattr(
                    self,
                    field_name,
                ),
                name=field_name,
            )

            if (
                field_name
                in (
                    "residual_std_s",
                    "residual_mad_s",
                    "median_absolute_residual_s",
                )
                and value < 0.0
            ):
                raise ValueError(
                    f"{field_name} must be >= 0."
                )

            object.__setattr__(
                self,
                field_name,
                value,
            )

    @property
    def pair(
        self,
    ) -> NodePair:
        """
        Return canonical node pair.
        """

        return (
            self.node_a,
            self.node_b,
        )

    def observed_offset_samples(
        self,
        sample_rate: int,
    ) -> float:
        """
        Convert directly observed offset to samples.
        """

        rate = _positive_int(
            sample_rate,
            name="sample_rate",
        )

        return (
            self.observed_offset_s
            * rate
        )

    def modeled_offset_samples(
        self,
        sample_rate: int,
    ) -> float:
        """
        Convert coherent modeled offset to samples.
        """

        rate = _positive_int(
            sample_rate,
            name="sample_rate",
        )

        return (
            self.modeled_offset_s
            * rate
        )

    def to_dict(
        self,
        *,
        sample_rate: int | None = None,
    ) -> dict[
        str,
        object,
    ]:
        """
        Serialize pair calibration.
        """

        data: dict[
            str,
            object,
        ] = {
            "node_a":
                self.node_a,

            "node_b":
                self.node_b,

            "observation_count":
                self.observation_count,

            "inlier_count":
                self.inlier_count,

            "rejected_count":
                self.rejected_count,

            "observed_offset_s":
                self.observed_offset_s,

            "modeled_offset_s":
                self.modeled_offset_s,

            "consistency_error_s":
                self.consistency_error_s,

            "residual_std_s":
                self.residual_std_s,

            "residual_mad_s":
                self.residual_mad_s,

            "median_absolute_residual_s":
                self.median_absolute_residual_s,
        }

        if sample_rate is not None:

            data[
                "observed_offset_samples"
            ] = (
                self.observed_offset_samples(
                    sample_rate
                )
            )

            data[
                "modeled_offset_samples"
            ] = (
                self.modeled_offset_samples(
                    sample_rate
                )
            )

        return data


# ======================================================================
# NODE TIMING BIAS
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class NodeTimingBias:
    """
    Estimated timing bias for one node relative to the reference node.
    """

    node_id: int

    bias_s: float

    is_reference: bool = (
        False
    )

    def __post_init__(
        self,
    ) -> None:

        node_id = _node_id(
            self.node_id,
            name="node_id",
        )

        bias = _finite_float(
            self.bias_s,
            name="bias_s",
        )

        if not isinstance(
            self.is_reference,
            bool,
        ):
            raise TypeError(
                "is_reference must be a bool."
            )

        object.__setattr__(
            self,
            "node_id",
            node_id,
        )

        object.__setattr__(
            self,
            "bias_s",
            bias,
        )

    def bias_samples(
        self,
        sample_rate: int,
    ) -> float:
        """
        Convert timing bias into samples.
        """

        rate = _positive_int(
            sample_rate,
            name="sample_rate",
        )

        return (
            self.bias_s
            * rate
        )

    def to_dict(
        self,
        *,
        sample_rate: int | None = None,
    ) -> dict[
        str,
        object,
    ]:
        """
        Serialize node timing bias.
        """

        data: dict[
            str,
            object,
        ] = {
            "node_id":
                self.node_id,

            "bias_s":
                self.bias_s,

            "is_reference":
                self.is_reference,
        }

        if sample_rate is not None:

            data[
                "bias_samples"
            ] = (
                self.bias_samples(
                    sample_rate
                )
            )

        return data


# ======================================================================
# COMPLETE CALIBRATION RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TDOACalibrationResult:
    """
    Complete coherent TDOA calibration result.
    """

    generated_at: str

    sample_rate: int

    reference_node: int

    pair_calibrations: tuple[
        PairTimingCalibration,
        ...,
    ]

    node_biases: tuple[
        NodeTimingBias,
        ...,
    ]

    observation_count: int

    accepted_observation_count: int

    rejected_observation_count: int

    rms_pair_consistency_error_s: float

    warnings: tuple[
        str,
        ...,
    ] = ()

    def __post_init__(
        self,
    ) -> None:

        sample_rate = _positive_int(
            self.sample_rate,
            name="sample_rate",
        )

        reference_node = _node_id(
            self.reference_node,
            name="reference_node",
        )

        if (
            isinstance(
                self.observation_count,
                bool,
            )
            or not isinstance(
                self.observation_count,
                Integral,
            )
        ):
            raise TypeError(
                "observation_count must be an integer."
            )

        if (
            isinstance(
                self.accepted_observation_count,
                bool,
            )
            or not isinstance(
                self.accepted_observation_count,
                Integral,
            )
        ):
            raise TypeError(
                "accepted_observation_count must be an integer."
            )

        if (
            isinstance(
                self.rejected_observation_count,
                bool,
            )
            or not isinstance(
                self.rejected_observation_count,
                Integral,
            )
        ):
            raise TypeError(
                "rejected_observation_count must be an integer."
            )

        total = int(
            self.observation_count
        )

        accepted = int(
            self.accepted_observation_count
        )

        rejected = int(
            self.rejected_observation_count
        )

        if (
            total < 0
            or accepted < 0
            or rejected < 0
        ):
            raise ValueError(
                "Observation counts must be >= 0."
            )

        if (
            accepted
            + rejected
            > total
        ):
            raise ValueError(
                "Accepted and rejected observations exceed total observations."
            )

        consistency = _finite_float(
            self.rms_pair_consistency_error_s,
            name="rms_pair_consistency_error_s",
        )

        if consistency < 0.0:
            raise ValueError(
                "rms_pair_consistency_error_s must be >= 0."
            )

        if not isinstance(
            self.generated_at,
            str,
        ):
            raise TypeError(
                "generated_at must be a string."
            )

        if not self.generated_at.strip():
            raise ValueError(
                "generated_at must not be empty."
            )

        pair_calibrations = tuple(
            self.pair_calibrations
        )

        node_biases = tuple(
            self.node_biases
        )

        warnings = tuple(
            str(
                warning
            )
            for warning
            in self.warnings
        )

        object.__setattr__(
            self,
            "sample_rate",
            sample_rate,
        )

        object.__setattr__(
            self,
            "reference_node",
            reference_node,
        )

        object.__setattr__(
            self,
            "pair_calibrations",
            pair_calibrations,
        )

        object.__setattr__(
            self,
            "node_biases",
            node_biases,
        )

        object.__setattr__(
            self,
            "observation_count",
            total,
        )

        object.__setattr__(
            self,
            "accepted_observation_count",
            accepted,
        )

        object.__setattr__(
            self,
            "rejected_observation_count",
            rejected,
        )

        object.__setattr__(
            self,
            "rms_pair_consistency_error_s",
            consistency,
        )

        object.__setattr__(
            self,
            "warnings",
            warnings,
        )

    # ==================================================================
    # PAIR LOOKUP
    # ==================================================================

    def calibration_for_pair(
        self,
        node_a: int,
        node_b: int,
    ) -> PairTimingCalibration | None:
        """
        Return pair calibration if available.
        """

        pair = canonical_pair(
            node_a,
            node_b,
        )

        for calibration in self.pair_calibrations:

            if calibration.pair == pair:

                return calibration

        return None

    # ==================================================================
    # OFFSET LOOKUP
    # ==================================================================

    def pair_offset_seconds(
        self,
        node_a: int,
        node_b: int,
    ) -> float:
        """
        Return modeled offset in requested pair orientation.

        Raises
        ------
        KeyError
            If the pair was not calibrated.
        """

        original_a = _node_id(
            node_a,
            name="node_a",
        )

        original_b = _node_id(
            node_b,
            name="node_b",
        )

        pair = canonical_pair(
            original_a,
            original_b,
        )

        calibration = (
            self.calibration_for_pair(
                pair[0],
                pair[1],
            )
        )

        if calibration is None:
            raise KeyError(
                f"No TDOA calibration exists for pair {pair}."
            )

        offset = (
            calibration.modeled_offset_s
        )

        if (
            original_a,
            original_b,
        ) == pair:

            return offset

        return (
            -offset
        )

    # ==================================================================
    # OFFSET IN SAMPLES
    # ==================================================================

    def pair_offset_samples(
        self,
        node_a: int,
        node_b: int,
    ) -> float:
        """
        Return modeled pair offset expressed in samples.
        """

        return (
            self.pair_offset_seconds(
                node_a,
                node_b,
            )
            * self.sample_rate
        )

    # ==================================================================
    # CORRECT TDOA
    # ==================================================================

    def correct_tdoa_seconds(
        self,
        node_a: int,
        node_b: int,
        measured_tdoa_s: float,
    ) -> float:
        """
        Apply timing calibration.

        corrected
            =
        measured - calibration_offset
        """

        measured = _finite_float(
            measured_tdoa_s,
            name="measured_tdoa_s",
        )

        offset = (
            self.pair_offset_seconds(
                node_a,
                node_b,
            )
        )

        return (
            measured
            - offset
        )

    # ==================================================================
    # SERIALIZATION
    # ==================================================================

    def to_dict(
        self,
    ) -> dict[
        str,
        object,
    ]:
        """
        Serialize complete calibration result.
        """

        return {
            "generated_at":
                self.generated_at,

            "sample_rate":
                self.sample_rate,

            "reference_node":
                self.reference_node,

            "observation_count":
                self.observation_count,

            "accepted_observation_count":
                self.accepted_observation_count,

            "rejected_observation_count":
                self.rejected_observation_count,

            "rms_pair_consistency_error_s":
                self.rms_pair_consistency_error_s,

            "rms_pair_consistency_error_samples":
                (
                    self.rms_pair_consistency_error_s
                    * self.sample_rate
                ),

            "node_biases": [
                bias.to_dict(
                    sample_rate=self.sample_rate
                )
                for bias
                in self.node_biases
            ],

            "pair_calibrations": [
                calibration.to_dict(
                    sample_rate=self.sample_rate
                )
                for calibration
                in self.pair_calibrations
            ],

            "warnings":
                list(
                    self.warnings
                ),
        }


# ======================================================================
# BUILD ONE CALIBRATION OBSERVATION
# ======================================================================


def build_calibration_observation(
    measurement: KnownSourceMeasurement,
    node_positions_m: Mapping[
        int,
        Sequence[Real],
    ],
) -> TDOACalibrationObservation:
    """
    Convert one known-source measurement into a calibration residual.
    """

    if not isinstance(
        measurement,
        KnownSourceMeasurement,
    ):
        raise TypeError(
            "measurement must be a KnownSourceMeasurement."
        )

    if measurement.node_a not in node_positions_m:
        raise KeyError(
            f"Missing position for Node {measurement.node_a}."
        )

    if measurement.node_b not in node_positions_m:
        raise KeyError(
            f"Missing position for Node {measurement.node_b}."
        )

    node_a_position = _normalize_position(
        node_positions_m[
            measurement.node_a
        ],
        name=f"node_positions_m[{measurement.node_a}]",
    )

    node_b_position = _normalize_position(
        node_positions_m[
            measurement.node_b
        ],
        name=f"node_positions_m[{measurement.node_b}]",
    )

    expected = expected_tdoa_seconds(
        measurement.source_position_m,
        node_a_position,
        node_b_position,
        speed_of_sound_mps=
            measurement.speed_of_sound_mps,
    )

    residual = (
        measurement.measured_tdoa_s
        - expected
    )

    return TDOACalibrationObservation(
        node_a=
            measurement.node_a,

        node_b=
            measurement.node_b,

        source_position_m=
            measurement.source_position_m,

        measured_tdoa_s=
            measurement.measured_tdoa_s,

        expected_tdoa_s=
            expected,

        residual_s=
            residual,

        speed_of_sound_mps=
            measurement.speed_of_sound_mps,

        quality_weight=
            measurement.quality_weight,
    )


# ======================================================================
# BUILD OBSERVATIONS
# ======================================================================


def build_calibration_observations(
    measurements: Iterable[
        KnownSourceMeasurement
    ],
    node_positions_m: Mapping[
        int,
        Sequence[Real],
    ],
) -> tuple[
    TDOACalibrationObservation,
    ...,
]:
    """
    Normalize a collection of controlled measurements.
    """

    observations: list[
        TDOACalibrationObservation
    ] = []

    for measurement in measurements:

        observations.append(
            build_calibration_observation(
                measurement,
                node_positions_m,
            )
        )

    return tuple(
        observations
    )


# ======================================================================
# MEDIAN ABSOLUTE DEVIATION
# ======================================================================


def _median_absolute_deviation(
    values: np.ndarray,
) -> float:
    """
    Calculate unscaled median absolute deviation.
    """

    if values.size == 0:
        return 0.0

    center = float(
        np.median(
            values
        )
    )

    return float(
        np.median(
            np.abs(
                values
                - center
            )
        )
    )


# ======================================================================
# ROBUST INLIER MASK
# ======================================================================


def _robust_inlier_mask(
    values: np.ndarray,
    *,
    modified_z_threshold: float,
) -> np.ndarray:
    """
    Detect strong outliers using median/MAD modified Z-score.

    For very small samples or zero MAD, all observations are retained.
    """

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if values.size < 3:

        return np.ones(
            values.shape,
            dtype=bool,
        )

    median_value = float(
        np.median(
            values
        )
    )

    mad = _median_absolute_deviation(
        values
    )

    if mad <= FLOAT_EPSILON:

        return np.ones(
            values.shape,
            dtype=bool,
        )

    modified_z = (
        MAD_SCALE_FACTOR
        * (
            values
            - median_value
        )
        / mad
    )

    return (
        np.abs(
            modified_z
        )
        <= modified_z_threshold
    )


# ======================================================================
# WEIGHTED MEAN
# ======================================================================


def _weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Calculate positive-weight weighted mean.
    """

    weight_sum = float(
        np.sum(
            weights
        )
    )

    if (
        not isfinite(
            weight_sum
        )
        or weight_sum <= 0.0
    ):
        raise ValueError(
            "Calibration weights must have a positive finite sum."
        )

    result = float(
        np.sum(
            values
            * weights
        )
        / weight_sum
    )

    if not isfinite(
        result
    ):
        raise ValueError(
            "Weighted calibration mean is not finite."
        )

    return result


# ======================================================================
# WEIGHTED STANDARD DEVIATION
# ======================================================================


def _weighted_std(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    center: float,
) -> float:
    """
    Calculate weighted RMS deviation around a supplied center.
    """

    if values.size == 0:
        return 0.0

    weight_sum = float(
        np.sum(
            weights
        )
    )

    if weight_sum <= 0.0:
        return 0.0

    variance = float(
        np.sum(
            weights
            * (
                values
                - center
            )
            ** 2
        )
        / weight_sum
    )

    return sqrt(
        max(
            0.0,
            variance,
        )
    )


# ======================================================================
# ESTIMATE DIRECT PAIR CALIBRATIONS
# ======================================================================


def _estimate_pair_calibrations(
    observations: Sequence[
        TDOACalibrationObservation
    ],
    *,
    min_observations_per_pair: int,
    modified_z_threshold: float,
) -> tuple[
    tuple[
        PairTimingCalibration,
        ...,
    ],
    tuple[
        str,
        ...,
    ],
]:
    """
    Estimate robust direct residual for every sufficiently observed pair.
    """

    grouped: dict[
        NodePair,
        list[
            TDOACalibrationObservation
        ],
    ] = defaultdict(
        list
    )

    for observation in observations:

        grouped[
            (
                observation.node_a,
                observation.node_b,
            )
        ].append(
            observation
        )

    calibrations: list[
        PairTimingCalibration
    ] = []

    warnings: list[
        str
    ] = []

    for pair in sorted(
        grouped
    ):

        pair_observations = (
            grouped[
                pair
            ]
        )

        count = len(
            pair_observations
        )

        if (
            count
            < min_observations_per_pair
        ):

            warnings.append(
                (
                    f"Pair {pair} has only {count} calibration "
                    f"observations; at least "
                    f"{min_observations_per_pair} are required."
                )
            )

            continue

        residuals = np.asarray(
            [
                observation.residual_s
                for observation
                in pair_observations
            ],
            dtype=np.float64,
        )

        weights = np.asarray(
            [
                observation.quality_weight
                for observation
                in pair_observations
            ],
            dtype=np.float64,
        )

        inlier_mask = (
            _robust_inlier_mask(
                residuals,
                modified_z_threshold=
                    modified_z_threshold,
            )
        )

        inlier_residuals = (
            residuals[
                inlier_mask
            ]
        )

        inlier_weights = (
            weights[
                inlier_mask
            ]
        )

        if (
            inlier_residuals.size == 0
        ):

            warnings.append(
                f"Pair {pair} had no valid calibration inliers."
            )

            continue

        observed_offset = (
            _weighted_mean(
                inlier_residuals,
                inlier_weights,
            )
        )

        residual_std = (
            _weighted_std(
                inlier_residuals,
                inlier_weights,
                center=observed_offset,
            )
        )

        residual_mad = (
            _median_absolute_deviation(
                inlier_residuals
            )
        )

        median_absolute_residual = float(
            np.median(
                np.abs(
                    inlier_residuals
                    - observed_offset
                )
            )
        )

        inlier_count = int(
            np.count_nonzero(
                inlier_mask
            )
        )

        rejected_count = (
            count
            - inlier_count
        )

        calibrations.append(
            PairTimingCalibration(
                node_a=
                    pair[0],

                node_b=
                    pair[1],

                observation_count=
                    count,

                inlier_count=
                    inlier_count,

                rejected_count=
                    rejected_count,

                observed_offset_s=
                    observed_offset,

                # Replaced later by coherent node-bias model.
                modeled_offset_s=
                    observed_offset,

                consistency_error_s=
                    0.0,

                residual_std_s=
                    residual_std,

                residual_mad_s=
                    residual_mad,

                median_absolute_residual_s=
                    median_absolute_residual,
            )
        )

    return (
        tuple(
            calibrations
        ),
        tuple(
            warnings
        ),
    )


# ======================================================================
# GRAPH CONNECTIVITY
# ======================================================================


def _connected_nodes(
    pairs: Sequence[
        PairTimingCalibration
    ],
    *,
    reference_node: int,
) -> set[
    int
]:
    """
    Find nodes connected to the calibration reference node.
    """

    adjacency: dict[
        int,
        set[
            int
        ],
    ] = defaultdict(
        set
    )

    for calibration in pairs:

        adjacency[
            calibration.node_a
        ].add(
            calibration.node_b
        )

        adjacency[
            calibration.node_b
        ].add(
            calibration.node_a
        )

    connected: set[
        int
    ] = {
        reference_node
    }

    stack = [
        reference_node
    ]

    while stack:

        node = stack.pop()

        for neighbour in adjacency.get(
            node,
            (),
        ):

            if neighbour in connected:
                continue

            connected.add(
                neighbour
            )

            stack.append(
                neighbour
            )

    return connected


# ======================================================================
# PAIR FIT WEIGHT
# ======================================================================


def _pair_fit_weight(
    calibration: PairTimingCalibration,
    *,
    sample_rate: int,
) -> float:
    """
    Calculate stable least-squares pair weight.

    Precision is limited to at least approximately one sample so that
    an unrealistically tiny measured standard deviation does not produce
    a numerically dominant calibration pair.
    """

    sample_period_s = (
        1.0
        / sample_rate
    )

    uncertainty = max(
        calibration.residual_std_s,
        sample_period_s,
    )

    return (
        calibration.inlier_count
        / (
            uncertainty
            ** 2
        )
    )


# ======================================================================
# FIT COHERENT NODE TIMING BIASES
# ======================================================================


def _fit_node_biases(
    pair_calibrations: Sequence[
        PairTimingCalibration
    ],
    *,
    reference_node: int,
    sample_rate: int,
) -> tuple[
    dict[
        int,
        float,
    ],
    set[
        int
    ],
    tuple[
        str,
        ...,
    ],
]:
    """
    Fit:

        bias_B - bias_A = observed_pair_offset

    using weighted least squares.

    Reference node bias is fixed at zero.
    """

    warnings: list[
        str
    ] = []

    if not pair_calibrations:

        return (
            {
                reference_node:
                    0.0
            },
            {
                reference_node
            },
            (
                "No pair calibrations were available for node-bias fitting.",
            ),
        )

    all_nodes: set[
        int
    ] = {
        reference_node
    }

    for calibration in pair_calibrations:

        all_nodes.add(
            calibration.node_a
        )

        all_nodes.add(
            calibration.node_b
        )

    connected = _connected_nodes(
        pair_calibrations,
        reference_node=
            reference_node,
    )

    disconnected = (
        all_nodes
        - connected
    )

    if disconnected:

        warnings.append(
            (
                "Calibration graph contains nodes not connected to "
                f"reference Node {reference_node}: "
                f"{sorted(disconnected)}."
            )
        )

    fitted_nodes = sorted(
        node
        for node
        in connected
        if node
        != reference_node
    )

    if not fitted_nodes:

        return (
            {
                reference_node:
                    0.0
            },
            connected,
            tuple(
                warnings
            ),
        )

    column_index = {
        node_id:
            index
        for index, node_id
        in enumerate(
            fitted_nodes
        )
    }

    design_rows: list[
        list[
            float
        ]
    ] = []

    targets: list[
        float
    ] = []

    row_weights: list[
        float
    ] = []

    for calibration in pair_calibrations:

        if (
            calibration.node_a
            not in connected
            or calibration.node_b
            not in connected
        ):
            continue

        row = [
            0.0
        ] * len(
            fitted_nodes
        )

        if (
            calibration.node_a
            != reference_node
        ):

            row[
                column_index[
                    calibration.node_a
                ]
            ] -= (
                1.0
            )

        if (
            calibration.node_b
            != reference_node
        ):

            row[
                column_index[
                    calibration.node_b
                ]
            ] += (
                1.0
            )

        design_rows.append(
            row
        )

        targets.append(
            calibration.observed_offset_s
        )

        row_weights.append(
            _pair_fit_weight(
                calibration,
                sample_rate=
                    sample_rate,
            )
        )

    if not design_rows:

        warnings.append(
            "No connected calibration equations were available."
        )

        return (
            {
                reference_node:
                    0.0
            },
            connected,
            tuple(
                warnings
            ),
        )

    design = np.asarray(
        design_rows,
        dtype=np.float64,
    )

    target = np.asarray(
        targets,
        dtype=np.float64,
    )

    weights = np.asarray(
        row_weights,
        dtype=np.float64,
    )

    sqrt_weights = np.sqrt(
        weights
    )

    weighted_design = (
        design
        * sqrt_weights[
            :,
            np.newaxis,
        ]
    )

    weighted_target = (
        target
        * sqrt_weights
    )

    solution, _, rank, _ = (
        np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )
    )

    if rank < len(
        fitted_nodes
    ):

        warnings.append(
            (
                "Node timing-bias system is rank deficient; "
                "collect additional pairwise calibration measurements."
            )
        )

    node_biases: dict[
        int,
        float,
    ] = {
        reference_node:
            0.0
    }

    for node_id, index in column_index.items():

        node_biases[
            node_id
        ] = float(
            solution[
                index
            ]
        )

    return (
        node_biases,
        connected,
        tuple(
            warnings
        ),
    )


# ======================================================================
# APPLY COHERENT NODE MODEL TO PAIRS
# ======================================================================


def _apply_node_bias_model(
    pair_calibrations: Sequence[
        PairTimingCalibration
    ],
    *,
    node_biases: Mapping[
        int,
        float,
    ],
    connected_nodes: set[
        int
    ],
) -> tuple[
    PairTimingCalibration,
    ...,
]:
    """
    Replace direct pair offsets with coherent modeled offsets where
    reference-connected node biases are available.
    """

    updated: list[
        PairTimingCalibration
    ] = []

    for calibration in pair_calibrations:

        if (
            calibration.node_a
            in connected_nodes
            and calibration.node_b
            in connected_nodes
            and calibration.node_a
            in node_biases
            and calibration.node_b
            in node_biases
        ):

            modeled_offset = (
                node_biases[
                    calibration.node_b
                ]
                - node_biases[
                    calibration.node_a
                ]
            )

        else:

            # Disconnected pair:
            #
            # retain direct robust estimate rather than inventing
            # a reference-relative node delay.
            modeled_offset = (
                calibration.observed_offset_s
            )

        consistency_error = (
            calibration.observed_offset_s
            - modeled_offset
        )

        updated.append(
            replace(
                calibration,

                modeled_offset_s=
                    modeled_offset,

                consistency_error_s=
                    consistency_error,
            )
        )

    return tuple(
        updated
    )


# ======================================================================
# RMS PAIR CONSISTENCY ERROR
# ======================================================================


def _rms_pair_consistency_error(
    calibrations: Sequence[
        PairTimingCalibration
    ],
) -> float:
    """
    Quantify disagreement between independently measured pair offsets
    and the coherent node-bias timing model.
    """

    if not calibrations:
        return 0.0

    errors = np.asarray(
        [
            calibration.consistency_error_s
            for calibration
            in calibrations
        ],
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                errors
                ** 2
            )
        )
    )


# ======================================================================
# MAIN CALIBRATION API
# ======================================================================


def calibrate_tdoa(
    measurements: Iterable[
        KnownSourceMeasurement
    ],
    node_positions_m: Mapping[
        int,
        Sequence[Real],
    ],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    reference_node: int = DEFAULT_REFERENCE_NODE,
    min_observations_per_pair: int = DEFAULT_MIN_OBSERVATIONS_PER_PAIR,
    modified_z_threshold: float = DEFAULT_MODIFIED_Z_THRESHOLD,
    generated_at: datetime | None = None,
) -> TDOACalibrationResult:
    """
    Estimate TDOA timing calibration from known-source measurements.

    Parameters
    ----------
    measurements
        Controlled pairwise TDOA observations.

    node_positions_m
        Known node coordinates.

        Example:

            {
                1: (0.0, 0.0),
                2: (1.0, 0.0),
                3: (0.0, 1.0),
            }

    sample_rate
        Acoustic sampling frequency.

    reference_node
        Node whose timing bias is fixed to zero.

        Node 1 is recommended for the current system.

    min_observations_per_pair
        Minimum observations required before a pair contributes to
        calibration.

    modified_z_threshold
        Robust MAD-based outlier threshold.

    generated_at
        Optional explicit calibration timestamp.

    Returns
    -------
    TDOACalibrationResult
        Coherent timing calibration.
    """

    rate = _positive_int(
        sample_rate,
        name="sample_rate",
    )

    reference = _node_id(
        reference_node,
        name="reference_node",
    )

    minimum_pair_observations = (
        _positive_int(
            min_observations_per_pair,
            name="min_observations_per_pair",
        )
    )

    z_threshold = _positive_float(
        modified_z_threshold,
        name="modified_z_threshold",
    )

    normalized_positions: dict[
        int,
        Position,
    ] = {}

    for raw_node_id, raw_position in node_positions_m.items():

        node_id = _node_id(
            raw_node_id,
            name="node_positions_m node id",
        )

        normalized_positions[
            node_id
        ] = (
            _normalize_position(
                raw_position,
                name=f"node_positions_m[{node_id}]",
            )
        )

    if reference not in normalized_positions:
        raise ValueError(
            (
                f"reference_node {reference} does not have "
                "a configured position."
            )
        )

    measurement_tuple = tuple(
        measurements
    )

    observations = (
        build_calibration_observations(
            measurement_tuple,
            normalized_positions,
        )
    )

    if not observations:
        raise ValueError(
            "At least one calibration measurement is required."
        )

    direct_pairs, pair_warnings = (
        _estimate_pair_calibrations(
            observations,
            min_observations_per_pair=
                minimum_pair_observations,
            modified_z_threshold=
                z_threshold,
        )
    )

    if not direct_pairs:
        raise ValueError(
            (
                "No microphone pair has enough usable observations "
                "to estimate TDOA calibration."
            )
        )

    node_bias_map, connected_nodes, fit_warnings = (
        _fit_node_biases(
            direct_pairs,
            reference_node=
                reference,
            sample_rate=
                rate,
        )
    )

    pair_calibrations = (
        _apply_node_bias_model(
            direct_pairs,
            node_biases=
                node_bias_map,
            connected_nodes=
                connected_nodes,
        )
    )

    node_biases = tuple(
        NodeTimingBias(
            node_id=
                node_id,

            bias_s=
                bias,

            is_reference=
                (
                    node_id
                    == reference
                ),
        )
        for node_id, bias
        in sorted(
            node_bias_map.items()
        )
    )

    accepted_count = sum(
        calibration.inlier_count
        for calibration
        in pair_calibrations
    )

    rejected_count = sum(
        calibration.rejected_count
        for calibration
        in pair_calibrations
    )

    warnings: list[
        str
    ] = []

    warnings.extend(
        pair_warnings
    )

    warnings.extend(
        fit_warnings
    )

    expected_nodes = set(
        normalized_positions
    )

    calibrated_nodes = {
        bias.node_id
        for bias
        in node_biases
    }

    missing_nodes = (
        expected_nodes
        - calibrated_nodes
    )

    if missing_nodes:

        warnings.append(
            (
                "No reference-connected timing bias could be estimated "
                f"for nodes: {sorted(missing_nodes)}."
            )
        )

    consistency_error = (
        _rms_pair_consistency_error(
            pair_calibrations
        )
    )

    sample_period_s = (
        1.0
        / rate
    )

    if (
        consistency_error
        > sample_period_s
    ):

        warnings.append(
            (
                "Pair calibration disagreement exceeds one sample. "
                "Review microphone coordinates, source positions, "
                "reflections and synchronization quality."
            )
        )

    if generated_at is None:

        timestamp = (
            datetime.now(
                timezone.utc
            )
        )

    else:

        if not isinstance(
            generated_at,
            datetime,
        ):
            raise TypeError(
                "generated_at must be a datetime or None."
            )

        timestamp = generated_at

        if timestamp.tzinfo is None:

            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        else:

            timestamp = timestamp.astimezone(
                timezone.utc
            )

    return TDOACalibrationResult(
        generated_at=
            timestamp.isoformat(),

        sample_rate=
            rate,

        reference_node=
            reference,

        pair_calibrations=
            pair_calibrations,

        node_biases=
            node_biases,

        observation_count=
            len(
                observations
            ),

        accepted_observation_count=
            accepted_count,

        rejected_observation_count=
            rejected_count,

        rms_pair_consistency_error_s=
            consistency_error,

        warnings=
            tuple(
                warnings
            ),
    )


# ======================================================================
# OFFSET MAP
# ======================================================================


def calibration_offsets_by_pair(
    calibration: TDOACalibrationResult,
) -> dict[
    NodePair,
    float,
]:
    """
    Return coherent canonical pair offsets in seconds.

    Example
    -------
    {
        (1, 2): 0.000012,
        (1, 3): -0.000008,
        (2, 3): -0.000020,
    }
    """

    if not isinstance(
        calibration,
        TDOACalibrationResult,
    ):
        raise TypeError(
            "calibration must be a TDOACalibrationResult."
        )

    return {
        pair_calibration.pair:
            pair_calibration.modeled_offset_s
        for pair_calibration
        in calibration.pair_calibrations
    }


# ======================================================================
# NODE BIAS MAP
# ======================================================================


def calibration_node_biases(
    calibration: TDOACalibrationResult,
) -> dict[
    int,
    float,
]:
    """
    Return reference-relative node timing biases in seconds.
    """

    if not isinstance(
        calibration,
        TDOACalibrationResult,
    ):
        raise TypeError(
            "calibration must be a TDOACalibrationResult."
        )

    return {
        node_bias.node_id:
            node_bias.bias_s
        for node_bias
        in calibration.node_biases
    }