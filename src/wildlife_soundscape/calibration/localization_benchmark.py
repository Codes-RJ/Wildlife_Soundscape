"""
Localization accuracy benchmarking.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Evaluate acoustic source-localization accuracy against controlled
ground-truth source positions.

The benchmark layer does not perform localization itself.

Instead, it evaluates completed localization observations containing:

    known source position
    estimated source position
    localization success/failure
    optional metadata


Primary metrics
---------------
For successful localizations this module calculates:

    Euclidean position error
    mean error
    RMSE
    median error
    standard deviation
    P90 error
    P95 error
    maximum error
    X-axis bias
    Y-axis bias
    repeatability

It also reports:

    attempted localizations
    successful localizations
    failed localizations
    success rate


Calibration evaluation
----------------------
The same controlled source positions should ideally be evaluated both:

    before TDOA calibration

and:

    after TDOA calibration

This allows calibration effectiveness to be reported objectively rather
than inferred from individual examples.


Scientific interpretation
-------------------------
Localization accuracy should be measured against positions that were
independently established.

The benchmark should not use estimated source positions as its own
ground truth.

For a credible experiment:

    microphone coordinates should be measured
    reference source coordinates should be measured
    multiple trials should be collected
    source positions should cover the usable test region
    unsuccessful localization attempts should be retained

Removing failed trials would artificially inflate reported performance.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


from dataclasses import (
    dataclass,
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


Position2D = tuple[
    float,
    float,
]


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _finite_float(
    value: object,
    *,
    name: str,
) -> float:
    """
    Require a finite real value.
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

    return (
        result
    )


def _non_negative_float(
    value: object,
    *,
    name: str,
) -> float:
    """
    Require a finite non-negative real value.
    """

    result = (
        _finite_float(
            value,
            name=name,
        )
    )

    if (
        result
        < 0.0
    ):

        raise ValueError(
            f"{name} must be >= 0."
        )

    return (
        result
    )


def _non_negative_int(
    value: object,
    *,
    name: str,
) -> int:
    """
    Require a non-negative integer.
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

    if (
        result
        < 0
    ):

        raise ValueError(
            f"{name} must be >= 0."
        )

    return (
        result
    )


def _normalize_position_2d(
    value: Sequence[float],
    *,
    name: str,
) -> Position2D:
    """
    Normalize one 2-D Cartesian position.
    """

    try:

        coordinates = tuple(
            value
        )

    except TypeError as exc:

        raise TypeError(
            f"{name} must be iterable."
        ) from exc

    if (
        len(
            coordinates
        )
        != 2
    ):

        raise ValueError(
            f"{name} must contain exactly two coordinates."
        )

    return (
        _finite_float(
            coordinates[
                0
            ],
            name=f"{name}[0]",
        ),
        _finite_float(
            coordinates[
                1
            ],
            name=f"{name}[1]",
        ),
    )


# ======================================================================
# DISTANCE
# ======================================================================


def position_error_m(
    reference_position_m: Sequence[float],
    estimated_position_m: Sequence[float],
) -> float:
    """
    Calculate Euclidean localization error in metres.
    """

    reference = (
        _normalize_position_2d(
            reference_position_m,
            name="reference_position_m",
        )
    )

    estimated = (
        _normalize_position_2d(
            estimated_position_m,
            name="estimated_position_m",
        )
    )

    dx = (
        estimated[
            0
        ]
        - reference[
            0
        ]
    )

    dy = (
        estimated[
            1
        ]
        - reference[
            1
        ]
    )

    return sqrt(
        (
            dx
            * dx
        )
        +
        (
            dy
            * dy
        )
    )


# ======================================================================
# BENCHMARK SAMPLE
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LocalizationBenchmarkSample:
    """
    One controlled localization trial.

    reference_position_m
        Independently known source coordinates.

    estimated_position_m
        Localization estimate.

        Must be None when localization failed.

    success
        True only when the localization pipeline produced a usable
        position estimate.

    label
        Optional experimental source-position identifier.

        Example:

            "P01"
            "center"
            "north-east"

    trial_index
        Optional repetition number.

    metadata
        Optional immutable-style metadata payload supplied by the
        experiment orchestration layer.
    """

    reference_position_m: Position2D

    estimated_position_m: Position2D | None

    success: bool

    label: str | None = (
        None
    )

    trial_index: int | None = (
        None
    )

    metadata: Mapping[
        str,
        object,
    ] | None = (
        None
    )

    def __post_init__(
        self,
    ) -> None:

        reference = (
            _normalize_position_2d(
                self.reference_position_m,
                name="reference_position_m",
            )
        )

        object.__setattr__(
            self,
            "reference_position_m",
            reference,
        )

        if not isinstance(
            self.success,
            bool,
        ):

            raise TypeError(
                "success must be bool."
            )

        # ==============================================================
        # SUCCESSFUL SAMPLE
        # ==============================================================

        if (
            self.success
        ):

            if (
                self.estimated_position_m
                is None
            ):

                raise ValueError(
                    (
                        "Successful benchmark sample "
                        "requires estimated_position_m."
                    )
                )

            estimated = (
                _normalize_position_2d(
                    self.estimated_position_m,
                    name="estimated_position_m",
                )
            )

            object.__setattr__(
                self,
                "estimated_position_m",
                estimated,
            )

        # ==============================================================
        # FAILED SAMPLE
        # ==============================================================

        else:

            if (
                self.estimated_position_m
                is not None
            ):

                raise ValueError(
                    (
                        "Failed benchmark sample must "
                        "use estimated_position_m=None."
                    )
                )

        # ==============================================================
        # LABEL
        # ==============================================================

        if (
            self.label
            is not None
        ):

            if not isinstance(
                self.label,
                str,
            ):

                raise TypeError(
                    "label must be a string or None."
                )

            if not (
                self.label.strip()
            ):

                raise ValueError(
                    "label cannot be empty when supplied."
                )

        # ==============================================================
        # TRIAL INDEX
        # ==============================================================

        if (
            self.trial_index
            is not None
        ):

            trial_index = (
                _non_negative_int(
                    self.trial_index,
                    name="trial_index",
                )
            )

            object.__setattr__(
                self,
                "trial_index",
                trial_index,
            )

        # ==============================================================
        # METADATA
        # ==============================================================

        if (
            self.metadata
            is not None
            and not isinstance(
                self.metadata,
                Mapping,
            )
        ):

            raise TypeError(
                "metadata must be a mapping or None."
            )

    # ==================================================================
    # ERROR
    # ==================================================================

    @property
    def error_m(
        self,
    ) -> float | None:
        """
        Euclidean position error.

        Failed localizations return None.
        """

        if (
            not self.success
            or self.estimated_position_m
            is None
        ):

            return (
                None
            )

        return position_error_m(
            self.reference_position_m,
            self.estimated_position_m,
        )

    # ==================================================================
    # AXIS ERROR
    # ==================================================================

    @property
    def x_error_m(
        self,
    ) -> float | None:
        """
        Signed X error:

            estimated_x - reference_x
        """

        if (
            not self.success
            or self.estimated_position_m
            is None
        ):

            return (
                None
            )

        return (
            self.estimated_position_m[
                0
            ]
            - self.reference_position_m[
                0
            ]
        )

    @property
    def y_error_m(
        self,
    ) -> float | None:
        """
        Signed Y error:

            estimated_y - reference_y
        """

        if (
            not self.success
            or self.estimated_position_m
            is None
        ):

            return (
                None
            )

        return (
            self.estimated_position_m[
                1
            ]
            - self.reference_position_m[
                1
            ]
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
        Serialize benchmark sample.
        """

        estimated: list[
            float
        ] | None

        if (
            self.estimated_position_m
            is None
        ):

            estimated = (
                None
            )

        else:

            estimated = [
                self.estimated_position_m[
                    0
                ],
                self.estimated_position_m[
                    1
                ],
            ]

        return {
            "reference_position_m": [
                self.reference_position_m[
                    0
                ],
                self.reference_position_m[
                    1
                ],
            ],

            "estimated_position_m":
                estimated,

            "success":
                self.success,

            "label":
                self.label,

            "trial_index":
                self.trial_index,

            "error_m":
                self.error_m,

            "x_error_m":
                self.x_error_m,

            "y_error_m":
                self.y_error_m,

            "metadata":
                (
                    dict(
                        self.metadata
                    )
                    if self.metadata is not None
                    else None
                ),
        }


# ======================================================================
# PER-POSITION SUMMARY
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PositionBenchmarkSummary:
    """
    Benchmark statistics for one ground-truth source position.
    """

    label: str

    reference_position_m: Position2D

    attempt_count: int

    success_count: int

    failure_count: int

    success_rate: float

    mean_error_m: float | None

    rmse_m: float | None

    median_error_m: float | None

    std_error_m: float | None

    p90_error_m: float | None

    maximum_error_m: float | None

    mean_estimated_position_m: Position2D | None

    repeatability_std_m: float | None

    def __post_init__(
        self,
    ) -> None:

        reference = (
            _normalize_position_2d(
                self.reference_position_m,
                name="reference_position_m",
            )
        )

        object.__setattr__(
            self,
            "reference_position_m",
            reference,
        )

        attempt_count = (
            _non_negative_int(
                self.attempt_count,
                name="attempt_count",
            )
        )

        success_count = (
            _non_negative_int(
                self.success_count,
                name="success_count",
            )
        )

        failure_count = (
            _non_negative_int(
                self.failure_count,
                name="failure_count",
            )
        )

        if (
            success_count
            + failure_count
            != attempt_count
        ):

            raise ValueError(
                (
                    "success_count + failure_count "
                    "must equal attempt_count."
                )
            )

        success_rate = (
            _finite_float(
                self.success_rate,
                name="success_rate",
            )
        )

        if not (
            0.0
            <= success_rate
            <= 1.0
        ):

            raise ValueError(
                "success_rate must lie in [0, 1]."
            )

        object.__setattr__(
            self,
            "attempt_count",
            attempt_count,
        )

        object.__setattr__(
            self,
            "success_count",
            success_count,
        )

        object.__setattr__(
            self,
            "failure_count",
            failure_count,
        )

        object.__setattr__(
            self,
            "success_rate",
            success_rate,
        )

    def to_dict(
        self,
    ) -> dict[
        str,
        object,
    ]:
        """
        Serialize per-position benchmark summary.
        """

        mean_position = (
            list(
                self.mean_estimated_position_m
            )
            if self.mean_estimated_position_m
            is not None
            else None
        )

        return {
            "label":
                self.label,

            "reference_position_m":
                list(
                    self.reference_position_m
                ),

            "attempt_count":
                self.attempt_count,

            "success_count":
                self.success_count,

            "failure_count":
                self.failure_count,

            "success_rate":
                self.success_rate,

            "mean_error_m":
                self.mean_error_m,

            "rmse_m":
                self.rmse_m,

            "median_error_m":
                self.median_error_m,

            "std_error_m":
                self.std_error_m,

            "p90_error_m":
                self.p90_error_m,

            "maximum_error_m":
                self.maximum_error_m,

            "mean_estimated_position_m":
                mean_position,

            "repeatability_std_m":
                self.repeatability_std_m,
        }


# ======================================================================
# COMPLETE BENCHMARK RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class LocalizationBenchmarkResult:
    """
    Aggregate localization benchmark.
    """

    attempt_count: int

    success_count: int

    failure_count: int

    success_rate: float

    mean_error_m: float | None

    rmse_m: float | None

    median_error_m: float | None

    std_error_m: float | None

    p90_error_m: float | None

    p95_error_m: float | None

    maximum_error_m: float | None

    x_bias_m: float | None

    y_bias_m: float | None

    radial_bias_m: float | None

    per_position: tuple[
        PositionBenchmarkSummary,
        ...,
    ]

    warnings: tuple[
        str,
        ...,
    ] = ()

    def __post_init__(
        self,
    ) -> None:

        attempt_count = (
            _non_negative_int(
                self.attempt_count,
                name="attempt_count",
            )
        )

        success_count = (
            _non_negative_int(
                self.success_count,
                name="success_count",
            )
        )

        failure_count = (
            _non_negative_int(
                self.failure_count,
                name="failure_count",
            )
        )

        if (
            success_count
            + failure_count
            != attempt_count
        ):

            raise ValueError(
                (
                    "success_count + failure_count "
                    "must equal attempt_count."
                )
            )

        success_rate = (
            _finite_float(
                self.success_rate,
                name="success_rate",
            )
        )

        if not (
            0.0
            <= success_rate
            <= 1.0
        ):

            raise ValueError(
                "success_rate must lie in [0, 1]."
            )

        object.__setattr__(
            self,
            "attempt_count",
            attempt_count,
        )

        object.__setattr__(
            self,
            "success_count",
            success_count,
        )

        object.__setattr__(
            self,
            "failure_count",
            failure_count,
        )

        object.__setattr__(
            self,
            "success_rate",
            success_rate,
        )

        object.__setattr__(
            self,
            "per_position",
            tuple(
                self.per_position
            ),
        )

        object.__setattr__(
            self,
            "warnings",
            tuple(
                str(
                    warning
                )
                for warning
                in self.warnings
            ),
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
        Serialize aggregate benchmark result.
        """

        return {
            "attempt_count":
                self.attempt_count,

            "success_count":
                self.success_count,

            "failure_count":
                self.failure_count,

            "success_rate":
                self.success_rate,

            "mean_error_m":
                self.mean_error_m,

            "rmse_m":
                self.rmse_m,

            "median_error_m":
                self.median_error_m,

            "std_error_m":
                self.std_error_m,

            "p90_error_m":
                self.p90_error_m,

            "p95_error_m":
                self.p95_error_m,

            "maximum_error_m":
                self.maximum_error_m,

            "x_bias_m":
                self.x_bias_m,

            "y_bias_m":
                self.y_bias_m,

            "radial_bias_m":
                self.radial_bias_m,

            "per_position": [
                summary.to_dict()
                for summary
                in self.per_position
            ],

            "warnings":
                list(
                    self.warnings
                ),
        }


# ======================================================================
# EMPTY METRICS
# ======================================================================


def _empty_error_metrics() -> dict[
    str,
    float | None,
]:
    """
    Return standard empty metric set.
    """

    return {
        "mean_error_m":
            None,

        "rmse_m":
            None,

        "median_error_m":
            None,

        "std_error_m":
            None,

        "p90_error_m":
            None,

        "p95_error_m":
            None,

        "maximum_error_m":
            None,
    }


# ======================================================================
# ERROR STATISTICS
# ======================================================================


def _error_statistics(
    errors: Sequence[float],
) -> dict[
    str,
    float | None,
]:
    """
    Calculate aggregate radial error statistics.
    """

    if not errors:

        return (
            _empty_error_metrics()
        )

    values = np.asarray(
        errors,
        dtype=np.float64,
    )

    if (
        values.ndim
        != 1
    ):

        raise ValueError(
            "errors must form a one-dimensional sequence."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):

        raise ValueError(
            "errors contain non-finite values."
        )

    if np.any(
        values
        < 0.0
    ):

        raise ValueError(
            "position errors cannot be negative."
        )

    return {
        "mean_error_m":
            float(
                np.mean(
                    values
                )
            ),

        "rmse_m":
            float(
                np.sqrt(
                    np.mean(
                        values
                        ** 2
                    )
                )
            ),

        "median_error_m":
            float(
                np.median(
                    values
                )
            ),

        "std_error_m":
            float(
                np.std(
                    values,
                    ddof=0,
                )
            ),

        "p90_error_m":
            float(
                np.percentile(
                    values,
                    90.0,
                )
            ),

        "p95_error_m":
            float(
                np.percentile(
                    values,
                    95.0,
                )
            ),

        "maximum_error_m":
            float(
                np.max(
                    values
                )
            ),
    }


# ======================================================================
# POSITION GROUP KEY
# ======================================================================


def _sample_group_key(
    sample: LocalizationBenchmarkSample,
) -> tuple[
    str,
    Position2D,
]:
    """
    Build deterministic grouping key.

    Explicit labels are preferred.

    Unlabelled samples are grouped by exact normalized ground-truth
    coordinates.
    """

    if (
        sample.label
        is not None
    ):

        label = (
            sample.label
        )

    else:

        x, y = (
            sample.reference_position_m
        )

        label = (
            f"({x:.6f}, {y:.6f})"
        )

    return (
        label,
        sample.reference_position_m,
    )


# ======================================================================
# REPEATABILITY
# ======================================================================


def _repeatability_std_m(
    estimated_positions: Sequence[
        Position2D
    ],
) -> float | None:
    """
    Estimate localization repeatability.

    Definition
    ----------
    1. Calculate the centroid of repeated estimates.
    2. Calculate radial distance of every estimate from that centroid.
    3. Return RMS radial spread.

    This describes estimate dispersion independently from ground-truth
    accuracy.

    A system can therefore be:

        repeatable but inaccurate

    if it repeatedly estimates the same biased location.
    """

    if (
        len(
            estimated_positions
        )
        < 2
    ):

        return (
            None
        )

    positions = np.asarray(
        estimated_positions,
        dtype=np.float64,
    )

    centroid = np.mean(
        positions,
        axis=0,
    )

    deltas = (
        positions
        - centroid
    )

    radial_squared = np.sum(
        deltas
        ** 2,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(
                radial_squared
            )
        )
    )


# ======================================================================
# PER-POSITION SUMMARY
# ======================================================================


def _summarize_position_group(
    label: str,
    reference_position_m: Position2D,
    samples: Sequence[
        LocalizationBenchmarkSample
    ],
) -> PositionBenchmarkSummary:
    """
    Summarize repeated trials at one known source position.
    """

    attempt_count = len(
        samples
    )

    successful = [
        sample
        for sample
        in samples
        if sample.success
    ]

    success_count = len(
        successful
    )

    failure_count = (
        attempt_count
        - success_count
    )

    success_rate = (
        success_count
        / attempt_count
        if attempt_count
        else 0.0
    )

    errors = [
        float(
            sample.error_m
        )
        for sample
        in successful
        if sample.error_m
        is not None
    ]

    metrics = (
        _error_statistics(
            errors
        )
    )

    estimated_positions = [
        sample.estimated_position_m
        for sample
        in successful
        if sample.estimated_position_m
        is not None
    ]

    if (
        estimated_positions
    ):

        estimated_array = np.asarray(
            estimated_positions,
            dtype=np.float64,
        )

        mean_estimated = (
            float(
                np.mean(
                    estimated_array[
                        :,
                        0
                    ]
                )
            ),
            float(
                np.mean(
                    estimated_array[
                        :,
                        1
                    ]
                )
            ),
        )

    else:

        mean_estimated = (
            None
        )

    repeatability = (
        _repeatability_std_m(
            estimated_positions
        )
    )

    return PositionBenchmarkSummary(
        label=
            label,

        reference_position_m=
            reference_position_m,

        attempt_count=
            attempt_count,

        success_count=
            success_count,

        failure_count=
            failure_count,

        success_rate=
            success_rate,

        mean_error_m=
            metrics[
                "mean_error_m"
            ],

        rmse_m=
            metrics[
                "rmse_m"
            ],

        median_error_m=
            metrics[
                "median_error_m"
            ],

        std_error_m=
            metrics[
                "std_error_m"
            ],

        p90_error_m=
            metrics[
                "p90_error_m"
            ],

        maximum_error_m=
            metrics[
                "maximum_error_m"
            ],

        mean_estimated_position_m=
            mean_estimated,

        repeatability_std_m=
            repeatability,
    )


# ======================================================================
# MAIN BENCHMARK
# ======================================================================


def benchmark_localization(
    samples: Iterable[
        LocalizationBenchmarkSample
    ],
) -> LocalizationBenchmarkResult:
    """
    Evaluate localization performance.

    Failed localizations remain part of success-rate statistics but do
    not receive an artificial position error.

    This distinction prevents two misleading approaches:

        assigning an arbitrary huge error to failure

    or:

        silently removing failed trials from the experiment
    """

    normalized_samples = tuple(
        samples
    )

    if not (
        normalized_samples
    ):

        raise ValueError(
            "At least one localization benchmark sample is required."
        )

    for sample in (
        normalized_samples
    ):

        if not isinstance(
            sample,
            LocalizationBenchmarkSample,
        ):

            raise TypeError(
                (
                    "All benchmark samples must be "
                    "LocalizationBenchmarkSample instances."
                )
            )

    # ==================================================================
    # SUCCESS / FAILURE
    # ==================================================================

    attempt_count = len(
        normalized_samples
    )

    successful = [
        sample
        for sample
        in normalized_samples
        if sample.success
    ]

    success_count = len(
        successful
    )

    failure_count = (
        attempt_count
        - success_count
    )

    success_rate = (
        success_count
        / attempt_count
    )

    # ==================================================================
    # RADIAL ERROR
    # ==================================================================

    errors = [
        float(
            sample.error_m
        )
        for sample
        in successful
        if sample.error_m
        is not None
    ]

    metrics = (
        _error_statistics(
            errors
        )
    )

    # ==================================================================
    # SIGNED X/Y ERROR
    # ==================================================================

    x_errors = [
        float(
            sample.x_error_m
        )
        for sample
        in successful
        if sample.x_error_m
        is not None
    ]

    y_errors = [
        float(
            sample.y_error_m
        )
        for sample
        in successful
        if sample.y_error_m
        is not None
    ]

    if (
        x_errors
        and y_errors
    ):

        x_bias = float(
            np.mean(
                np.asarray(
                    x_errors,
                    dtype=np.float64,
                )
            )
        )

        y_bias = float(
            np.mean(
                np.asarray(
                    y_errors,
                    dtype=np.float64,
                )
            )
        )

        radial_bias = sqrt(
            (
                x_bias
                * x_bias
            )
            +
            (
                y_bias
                * y_bias
            )
        )

    else:

        x_bias = (
            None
        )

        y_bias = (
            None
        )

        radial_bias = (
            None
        )

    # ==================================================================
    # GROUP BY REFERENCE POSITION
    # ==================================================================

    grouped: dict[
        tuple[
            str,
            Position2D,
        ],
        list[
            LocalizationBenchmarkSample
        ],
    ] = {}

    for sample in (
        normalized_samples
    ):

        key = (
            _sample_group_key(
                sample
            )
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            sample
        )

    per_position = tuple(
        _summarize_position_group(
            label,
            reference_position,
            grouped[
                (
                    label,
                    reference_position,
                )
            ],
        )
        for (
            label,
            reference_position,
        )
        in sorted(
            grouped,
            key=lambda item: (
                item[
                    0
                ],
                item[
                    1
                ][
                    0
                ],
                item[
                    1
                ][
                    1
                ],
            ),
        )
    )

    # ==================================================================
    # WARNINGS
    # ==================================================================

    warnings: list[
        str
    ] = []

    if (
        success_count
        == 0
    ):

        warnings.append(
            (
                "No localization attempt succeeded; "
                "position-error metrics are unavailable."
            )
        )

    elif (
        success_count
        < 3
    ):

        warnings.append(
            (
                "Fewer than three successful localizations "
                "are available; statistical metrics are not "
                "yet representative."
            )
        )

    if (
        success_rate
        < 0.80
    ):

        warnings.append(
            (
                "Localization success rate is below 80%. "
                "Review synchronization, signal quality, geometry "
                "and GCC-PHAT acceptance thresholds."
            )
        )

    if (
        len(
            per_position
        )
        < 3
    ):

        warnings.append(
            (
                "Benchmark uses fewer than three distinct reference "
                "positions. Spatial coverage of the evaluation is limited."
            )
        )

    return LocalizationBenchmarkResult(
        attempt_count=
            attempt_count,

        success_count=
            success_count,

        failure_count=
            failure_count,

        success_rate=
            success_rate,

        mean_error_m=
            metrics[
                "mean_error_m"
            ],

        rmse_m=
            metrics[
                "rmse_m"
            ],

        median_error_m=
            metrics[
                "median_error_m"
            ],

        std_error_m=
            metrics[
                "std_error_m"
            ],

        p90_error_m=
            metrics[
                "p90_error_m"
            ],

        p95_error_m=
            metrics[
                "p95_error_m"
            ],

        maximum_error_m=
            metrics[
                "maximum_error_m"
            ],

        x_bias_m=
            x_bias,

        y_bias_m=
            y_bias,

        radial_bias_m=
            radial_bias,

        per_position=
            per_position,

        warnings=
            tuple(
                warnings
            ),
    )


# ======================================================================
# CALIBRATION COMPARISON
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class CalibrationBenchmarkComparison:
    """
    Compare localization performance before and after calibration.
    """

    before: LocalizationBenchmarkResult

    after: LocalizationBenchmarkResult

    mean_error_improvement_m: float | None

    mean_error_improvement_percent: float | None

    rmse_improvement_m: float | None

    rmse_improvement_percent: float | None

    success_rate_change: float

    calibration_improved_mean_error: bool | None

    calibration_improved_rmse: bool | None

    def to_dict(
        self,
    ) -> dict[
        str,
        object,
    ]:
        """
        Serialize comparison.
        """

        return {
            "before":
                self.before.to_dict(),

            "after":
                self.after.to_dict(),

            "mean_error_improvement_m":
                self.mean_error_improvement_m,

            "mean_error_improvement_percent":
                self.mean_error_improvement_percent,

            "rmse_improvement_m":
                self.rmse_improvement_m,

            "rmse_improvement_percent":
                self.rmse_improvement_percent,

            "success_rate_change":
                self.success_rate_change,

            "calibration_improved_mean_error":
                self.calibration_improved_mean_error,

            "calibration_improved_rmse":
                self.calibration_improved_rmse,
        }


# ======================================================================
# IMPROVEMENT
# ======================================================================


def _metric_improvement(
    before: float | None,
    after: float | None,
) -> tuple[
    float | None,
    float | None,
    bool | None,
]:
    """
    Compare an error metric.

    Positive improvement means the calibrated result has LOWER error.
    """

    if (
        before
        is None
        or after
        is None
    ):

        return (
            None,
            None,
            None,
        )

    before_value = (
        _non_negative_float(
            before,
            name="before",
        )
    )

    after_value = (
        _non_negative_float(
            after,
            name="after",
        )
    )

    improvement = (
        before_value
        - after_value
    )

    percentage: float | None

    if (
        before_value
        > 0.0
    ):

        percentage = (
            improvement
            / before_value
            * 100.0
        )

    else:

        percentage = (
            0.0
            if after_value == 0.0
            else None
        )

    improved = (
        after_value
        < before_value
    )

    return (
        improvement,
        percentage,
        improved,
    )


# ======================================================================
# COMPARE BEFORE / AFTER CALIBRATION
# ======================================================================


def compare_calibration_benchmarks(
    before: LocalizationBenchmarkResult,
    after: LocalizationBenchmarkResult,
) -> CalibrationBenchmarkComparison:
    """
    Compare uncalibrated and calibrated benchmark summaries.

    For strongest scientific validity, both benchmark sets should use:

        identical reference positions
        identical trial structure
        identical source signals
        identical array geometry
        equivalent environmental conditions

    Ideally, the same recorded synchronized waveforms should be processed
    once without calibration and once with calibration. That isolates the
    effect of timing calibration from trial-to-trial environmental noise.
    """

    if not isinstance(
        before,
        LocalizationBenchmarkResult,
    ):

        raise TypeError(
            (
                "before must be a "
                "LocalizationBenchmarkResult."
            )
        )

    if not isinstance(
        after,
        LocalizationBenchmarkResult,
    ):

        raise TypeError(
            (
                "after must be a "
                "LocalizationBenchmarkResult."
            )
        )

    (
        mean_improvement,
        mean_improvement_percent,
        mean_improved,
    ) = (
        _metric_improvement(
            before.mean_error_m,
            after.mean_error_m,
        )
    )

    (
        rmse_improvement,
        rmse_improvement_percent,
        rmse_improved,
    ) = (
        _metric_improvement(
            before.rmse_m,
            after.rmse_m,
        )
    )

    return CalibrationBenchmarkComparison(
        before=
            before,

        after=
            after,

        mean_error_improvement_m=
            mean_improvement,

        mean_error_improvement_percent=
            mean_improvement_percent,

        rmse_improvement_m=
            rmse_improvement,

        rmse_improvement_percent=
            rmse_improvement_percent,

        success_rate_change=
            (
                after.success_rate
                - before.success_rate
            ),

        calibration_improved_mean_error=
            mean_improved,

        calibration_improved_rmse=
            rmse_improved,
    )


# ======================================================================
# FLAT SAMPLE ROWS
# ======================================================================


def benchmark_sample_rows(
    samples: Iterable[
        LocalizationBenchmarkSample
    ],
) -> list[
    dict[
        str,
        object,
    ]
]:
    """
    Convert benchmark samples into flat rows suitable for CSV export.
    """

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for sample in (
        samples
    ):

        if not isinstance(
            sample,
            LocalizationBenchmarkSample,
        ):

            raise TypeError(
                (
                    "All samples must be "
                    "LocalizationBenchmarkSample instances."
                )
            )

        reference_x, reference_y = (
            sample.reference_position_m
        )

        if (
            sample.estimated_position_m
            is None
        ):

            estimated_x = (
                None
            )

            estimated_y = (
                None
            )

        else:

            estimated_x, estimated_y = (
                sample.estimated_position_m
            )

        rows.append(
            {
                "label":
                    sample.label,

                "trial_index":
                    sample.trial_index,

                "reference_x_m":
                    reference_x,

                "reference_y_m":
                    reference_y,

                "estimated_x_m":
                    estimated_x,

                "estimated_y_m":
                    estimated_y,

                "success":
                    sample.success,

                "error_m":
                    sample.error_m,

                "x_error_m":
                    sample.x_error_m,

                "y_error_m":
                    sample.y_error_m,
            }
        )

    return (
        rows
    )
