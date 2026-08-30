"""
Tests for localization.solver.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. PositionResult contract
    2. exact 2D localization from synthetic TDOA
    3. TDOA sign convention
    4. different source positions
    5. measurement-order independence
    6. pair-order consistency
    7. use of only valid measurements
    8. insufficient-measurement failure
    9. insufficient-node failure
    10. optional solver bounds
    11. residual interpretation
    12. imperfect/noisy TDOA handling
    13. deterministic behaviour
    14. physically meaningful output fields

Sign convention
---------------
For a pair:

    node_a = A
    node_b = B

the TDOA measurement is:

    delay_seconds
        = arrival_time_B - arrival_time_A

The position solver therefore predicts:

    (distance_B - distance_A) / c

and compares that prediction with the measured TDOA.

This convention must remain consistent across:

    GCC-PHAT
        ↓
    TDOAMeasurement
        ↓
    solve_position()

Current solver API
------------------

    solve_position(
        node_positions,
        measurements,
        speed_of_sound_mps=...,
        bounds=...,
    )
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from itertools import (
    combinations,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from localization.solver import (
    PositionResult,
    solve_position,
)

from localization.tdoa import (
    TDOAMeasurement,
    physical_max_delay,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SPEED_OF_SOUND_MPS = (
    343.0
)


SAMPLE_RATE = (
    48_000.0
)


# ======================================================================
# TEST GEOMETRY
# ======================================================================


NODE_POSITIONS = {
    1: (
        0.0,
        0.0,
    ),

    2: (
        1.0,
        0.0,
    ),

    3: (
        0.5,
        0.8660254037844386,
    ),
}


# ======================================================================
# TEST HELPERS
# ======================================================================


def distance(
    point_a: tuple[
        float,
        float,
    ],
    point_b: tuple[
        float,
        float,
    ],
) -> float:
    """
    Euclidean distance in meters.
    """

    dx = (
        float(
            point_a[
                0
            ]
        )
        - float(
            point_b[
                0
            ]
        )
    )

    dy = (
        float(
            point_a[
                1
            ]
        )
        - float(
            point_b[
                1
            ]
        )
    )

    return math.hypot(
        dx,
        dy,
    )


def exact_tdoa(
    source_xy: tuple[
        float,
        float,
    ],
    node_a: int,
    node_b: int,
    *,
    speed_of_sound_mps: float = SPEED_OF_SOUND_MPS,
) -> float:
    """
    Compute mathematically exact TDOA for a known source.

    Convention:

        TDOA_AB
            = arrival_B - arrival_A

            = (distance_B - distance_A) / c
    """

    distance_a = distance(
        source_xy,
        NODE_POSITIONS[
            node_a
        ],
    )

    distance_b = distance(
        source_xy,
        NODE_POSITIONS[
            node_b
        ],
    )

    return (
        distance_b
        - distance_a
    ) / float(
        speed_of_sound_mps
    )


def make_measurement(
    source_xy: tuple[
        float,
        float,
    ],
    node_a: int,
    node_b: int,
    *,
    speed_of_sound_mps: float = SPEED_OF_SOUND_MPS,
    delay_offset_seconds: float = 0.0,
    valid: bool = True,
    peak_ratio: float = 5.0,
    reason: str = "",
) -> TDOAMeasurement:
    """
    Construct one synthetic physically valid TDOA measurement.
    """

    delay_seconds = (
        exact_tdoa(
            source_xy,
            node_a,
            node_b,
            speed_of_sound_mps=
                speed_of_sound_mps,
        )
        + float(
            delay_offset_seconds
        )
    )

    delay_samples = (
        delay_seconds
        * SAMPLE_RATE
    )

    maximum_delay = physical_max_delay(
        NODE_POSITIONS[
            node_a
        ],
        NODE_POSITIONS[
            node_b
        ],
        speed_of_sound_mps=
            speed_of_sound_mps,
    )

    return TDOAMeasurement(
        node_a=
            node_a,

        node_b=
            node_b,

        delay_seconds=
            delay_seconds,

        delay_samples=
            delay_samples,

        peak_ratio=
            peak_ratio,

        max_delay_seconds=
            maximum_delay,

        valid=
            valid,

        reason=
            reason,
    )


def make_all_pair_measurements(
    source_xy: tuple[
        float,
        float,
    ],
    *,
    speed_of_sound_mps: float = SPEED_OF_SOUND_MPS,
) -> tuple[
    TDOAMeasurement,
    ...,
]:
    """
    Generate all three pairwise TDOA measurements for Nodes 1, 2 and 3.
    """

    return tuple(
        make_measurement(
            source_xy,
            node_a,
            node_b,
            speed_of_sound_mps=
                speed_of_sound_mps,
        )

        for (
            node_a,
            node_b,
        )
        in combinations(
            sorted(
                NODE_POSITIONS
            ),
            2,
        )
    )


# ======================================================================
# RESULT CONTRACT
# ======================================================================


def test_solver_returns_position_result() -> None:

    source = (
        0.45,
        0.35,
    )

    measurements = (
        make_all_pair_measurements(
            source
        )
    )

    result = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert isinstance(
        result,
        PositionResult,
    )


def test_successful_result_contains_finite_diagnostics() -> None:

    source = (
        0.45,
        0.35,
    )

    result = solve_position(
        NODE_POSITIONS,
        make_all_pair_measurements(
            source
        ),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        result.success
    )

    assert math.isfinite(
        result.x
    )

    assert math.isfinite(
        result.y
    )

    assert math.isfinite(
        result.residual_rms_seconds
    )

    assert math.isfinite(
        result.residual_rms_meters
    )

    assert math.isfinite(
        result.cost
    )

    assert (
        result.nfev
        > 0
    )

    assert isinstance(
        result.message,
        str,
    )


# ======================================================================
# EXACT LOCALIZATION
# ======================================================================


@pytest.mark.parametrize(
    "source_xy",
    [
        (
            0.45,
            0.35,
        ),

        (
            0.25,
            0.20,
        ),

        (
            0.70,
            0.22,
        ),

        (
            0.50,
            0.55,
        ),
    ],
)
def test_solver_recovers_known_source_from_exact_tdoa(
    source_xy: tuple[
        float,
        float,
    ],
) -> None:

    measurements = (
        make_all_pair_measurements(
            source_xy
        )
    )

    result = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        result.success
    )

    assert (
        result.x
        == pytest.approx(
            source_xy[
                0
            ],
            abs=
                1e-4,
        )
    )

    assert (
        result.y
        == pytest.approx(
            source_xy[
                1
            ],
            abs=
                1e-4,
        )
    )

    assert (
        result.residual_rms_seconds
        < 1e-7
    )

    assert (
        result.residual_rms_meters
        < 1e-4
    )


# ======================================================================
# SIGN CONVENTION
# ======================================================================


def test_exact_tdoa_sign_matches_distance_difference() -> None:

    source = (
        0.15,
        0.15,
    )

    delay_12 = exact_tdoa(
        source,
        1,
        2,
    )

    distance_1 = distance(
        source,
        NODE_POSITIONS[
            1
        ],
    )

    distance_2 = distance(
        source,
        NODE_POSITIONS[
            2
        ],
    )

    # Source is closer to Node 1 than Node 2.
    assert (
        distance_2
        > distance_1
    )

    # Therefore Node 2 receives the waveform later.
    assert (
        delay_12
        > 0.0
    )


def test_reversing_pair_reverses_tdoa_sign() -> None:

    source = (
        0.25,
        0.30,
    )

    delay_ab = exact_tdoa(
        source,
        1,
        2,
    )

    delay_ba = exact_tdoa(
        source,
        2,
        1,
    )

    assert (
        delay_ab
        == pytest.approx(
            -delay_ba,
            rel=
                1e-12,
            abs=
                1e-15,
        )
    )


# ======================================================================
# PAIR ORIENTATION
# ======================================================================


def test_solver_accepts_reversed_pair_when_delay_sign_is_also_reversed() -> None:

    source = (
        0.40,
        0.30,
    )

    measurements = (
        make_measurement(
            source,
            2,
            1,
        ),

        make_measurement(
            source,
            3,
            1,
        ),

        make_measurement(
            source,
            3,
            2,
        ),
    )

    result = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        result.success
    )

    assert (
        result.x
        == pytest.approx(
            source[
                0
            ],
            abs=
                1e-4,
        )
    )

    assert (
        result.y
        == pytest.approx(
            source[
                1
            ],
            abs=
                1e-4,
        )
    )


# ======================================================================
# MEASUREMENT ORDER
# ======================================================================


def test_measurement_order_does_not_change_solution() -> None:

    source = (
        0.38,
        0.42,
    )

    measurements = list(
        make_all_pair_measurements(
            source
        )
    )

    forward = solve_position(
        NODE_POSITIONS,
        tuple(
            measurements
        ),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    reverse = solve_position(
        NODE_POSITIONS,
        tuple(
            reversed(
                measurements
            )
        ),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        forward.success
    )

    assert (
        reverse.success
    )

    assert (
        forward.x
        == pytest.approx(
            reverse.x,
            abs=
                1e-8,
        )
    )

    assert (
        forward.y
        == pytest.approx(
            reverse.y,
            abs=
                1e-8,
        )
    )


# ======================================================================
# VALID / INVALID MEASUREMENT FILTERING
# ======================================================================


def test_solver_ignores_invalid_measurement() -> None:

    source = (
        0.45,
        0.35,
    )

    valid_measurements = list(
        make_all_pair_measurements(
            source
        )
    )

    invalid_extra = TDOAMeasurement(
        node_a=
            1,

        node_b=
            2,

        delay_seconds=
            0.002,

        delay_samples=
            96.0,

        peak_ratio=
            0.5,

        max_delay_seconds=
            physical_max_delay(
                NODE_POSITIONS[
                    1
                ],
                NODE_POSITIONS[
                    2
                ],
                speed_of_sound_mps=
                    SPEED_OF_SOUND_MPS,
            ),

        valid=
            False,

        reason=
            "synthetic rejected correlation",
    )

    measurements = tuple(
        valid_measurements
        + [
            invalid_extra
        ]
    )

    result = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        result.success
    )

    assert (
        result.x
        == pytest.approx(
            source[
                0
            ],
            abs=
                1e-4,
        )
    )

    assert (
        result.y
        == pytest.approx(
            source[
                1
            ],
            abs=
                1e-4,
        )
    )


# ======================================================================
# INSUFFICIENT MEASUREMENTS
# ======================================================================


def test_solver_fails_with_no_measurements() -> None:

    result = solve_position(
        NODE_POSITIONS,
        (),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert not (
        result.success
    )

    assert isinstance(
        result,
        PositionResult,
    )


def test_solver_fails_with_only_one_valid_measurement() -> None:

    source = (
        0.4,
        0.3,
    )

    measurements = (
        make_measurement(
            source,
            1,
            2,
        ),
    )

    result = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert not (
        result.success
    )


def test_solver_requires_at_least_three_unique_nodes() -> None:
    """
    Multiple measurements involving only two microphones still cannot
    determine a unique 2D source position.
    """

    source = (
        0.4,
        0.3,
    )

    first = make_measurement(
        source,
        1,
        2,
    )

    second = TDOAMeasurement(
        node_a=
            2,

        node_b=
            1,

        delay_seconds=
            -first.delay_seconds,

        delay_samples=
            -first.delay_samples,

        peak_ratio=
            first.peak_ratio,

        max_delay_seconds=
            first.max_delay_seconds,

        valid=
            True,

        reason=
            "",
    )

    result = solve_position(
        NODE_POSITIONS,
        (
            first,
            second,
        ),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert not (
        result.success
    )


# ======================================================================
# FAILED RESULT CONTRACT
# ======================================================================


def test_failed_solver_still_returns_position_result() -> None:

    result = solve_position(
        NODE_POSITIONS,
        (),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert isinstance(
        result,
        PositionResult,
    )

    assert not (
        result.success
    )

    assert isinstance(
        result.message,
        str,
    )

    assert (
        len(
            result.message
        )
        > 0
    )


# ======================================================================
# RESIDUAL INTERPRETATION
# ======================================================================


def test_residual_meter_conversion_matches_speed_of_sound() -> None:

    source = (
        0.40,
        0.28,
    )

    measurements = list(
        make_all_pair_measurements(
            source
        )
    )

    # --------------------------------------------------------------
    # DELIBERATELY ADD SMALL TIMING ERROR
    # --------------------------------------------------------------

    original = (
        measurements[
            0
        ]
    )

    timing_error = (
        5e-6
    )

    measurements[
        0
    ] = TDOAMeasurement(
        node_a=
            original.node_a,

        node_b=
            original.node_b,

        delay_seconds=
            original.delay_seconds
            + timing_error,

        delay_samples=
            (
                original.delay_seconds
                + timing_error
            )
            * SAMPLE_RATE,

        peak_ratio=
            original.peak_ratio,

        max_delay_seconds=
            original.max_delay_seconds,

        valid=
            True,

        reason=
            "",
    )

    result = solve_position(
        NODE_POSITIONS,
        tuple(
            measurements
        ),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        result.success
    )

    assert (
        result.residual_rms_meters
        == pytest.approx(
            result.residual_rms_seconds
            * SPEED_OF_SOUND_MPS,
            rel=
                1e-9,
            abs=
                1e-12,
        )
    )


# ======================================================================
# NOISY TDOA
# ======================================================================


def test_small_tdoa_error_produces_reasonable_position() -> None:
    """
    A few microseconds of timing error should perturb the estimate but
    should not completely destabilize the solver for a well-conditioned
    triangular array.
    """

    source = (
        0.42,
        0.32,
    )

    pair_offsets = {
        (
            1,
            2,
        ):
            3e-6,

        (
            1,
            3,
        ):
            -2e-6,

        (
            2,
            3,
        ):
            1e-6,
    }

    measurements = tuple(
        make_measurement(
            source,
            node_a,
            node_b,
            delay_offset_seconds=
                pair_offsets[
                    (
                        node_a,
                        node_b,
                    )
                ],
        )

        for (
            node_a,
            node_b,
        )
        in combinations(
            sorted(
                NODE_POSITIONS
            ),
            2,
        )
    )

    result = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        result.success
    )

    position_error_m = distance(
        (
            result.x,
            result.y,
        ),
        source,
    )

    assert (
        position_error_m
        < 0.05
    )

    assert (
        result.residual_rms_seconds
        >= 0.0
    )

    assert (
        result.residual_rms_meters
        >= 0.0
    )


# ======================================================================
# OPTIONAL BOUNDS
# ======================================================================


def test_solver_accepts_bounds_containing_source() -> None:

    source = (
        0.45,
        0.35,
    )

    measurements = (
        make_all_pair_measurements(
            source
        )
    )

    bounds = (
        (
            0.0,
            0.0,
        ),
        (
            1.0,
            1.0,
        ),
    )

    result = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
        bounds=
            bounds,
    )

    assert (
        result.success
    )

    assert (
        0.0
        <= result.x
        <= 1.0
    )

    assert (
        0.0
        <= result.y
        <= 1.0
    )

    assert (
        result.x
        == pytest.approx(
            source[
                0
            ],
            abs=
                1e-4,
        )
    )

    assert (
        result.y
        == pytest.approx(
            source[
                1
            ],
            abs=
                1e-4,
        )
    )


def test_solver_result_respects_configured_bounds() -> None:

    source = (
        0.45,
        0.35,
    )

    bounds = (
        (
            0.0,
            0.0,
        ),
        (
            0.40,
            0.30,
        ),
    )

    result = solve_position(
        NODE_POSITIONS,
        make_all_pair_measurements(
            source
        ),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
        bounds=
            bounds,
    )

    # True source lies outside this artificial box. The optimizer may
    # still succeed numerically, but its estimate must remain inside the
    # requested search bounds.

    assert (
        0.0
        <= result.x
        <= 0.40
    )

    assert (
        0.0
        <= result.y
        <= 0.30
    )


# ======================================================================
# SHARED FIXTURE GEOMETRY
# ======================================================================


def test_solver_recovers_shared_known_source_fixture(
    node_positions,
    known_source_xy,
    speed_of_sound_mps,
) -> None:
    """
    Uses the common conftest localization geometry rather than this
    module's local geometry.
    """

    measurements = []

    for (
        node_a,
        node_b,
    ) in combinations(
        sorted(
            node_positions
        ),
        2,
    ):

        distance_a = distance(
            known_source_xy,
            node_positions[
                node_a
            ],
        )

        distance_b = distance(
            known_source_xy,
            node_positions[
                node_b
            ],
        )

        delay_seconds = (
            distance_b
            - distance_a
        ) / speed_of_sound_mps

        max_delay_seconds = physical_max_delay(
            node_positions[
                node_a
            ],
            node_positions[
                node_b
            ],
            speed_of_sound_mps=
                speed_of_sound_mps,
        )

        measurements.append(
            TDOAMeasurement(
                node_a=
                    node_a,

                node_b=
                    node_b,

                delay_seconds=
                    delay_seconds,

                delay_samples=
                    delay_seconds
                    * SAMPLE_RATE,

                peak_ratio=
                    5.0,

                max_delay_seconds=
                    max_delay_seconds,

                valid=
                    True,

                reason=
                    "",
            )
        )

    result = solve_position(
        node_positions,
        tuple(
            measurements
        ),
        speed_of_sound_mps=
            speed_of_sound_mps,
    )

    assert (
        result.success
    )

    assert (
        result.x
        == pytest.approx(
            known_source_xy[
                0
            ],
            abs=
                1e-4,
        )
    )

    assert (
        result.y
        == pytest.approx(
            known_source_xy[
                1
            ],
            abs=
                1e-4,
        )
    )


# ======================================================================
# DETERMINISM
# ======================================================================


def test_solver_is_deterministic_for_same_input() -> None:

    source = (
        0.37,
        0.29,
    )

    measurements = (
        make_all_pair_measurements(
            source
        )
    )

    first = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    second = solve_position(
        NODE_POSITIONS,
        measurements,
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        first.success
        == second.success
    )

    assert (
        first.x
        == pytest.approx(
            second.x,
            abs=
                1e-12,
        )
    )

    assert (
        first.y
        == pytest.approx(
            second.y,
            abs=
                1e-12,
        )
    )

    assert (
        first.residual_rms_seconds
        == pytest.approx(
            second.residual_rms_seconds,
            abs=
                1e-15,
        )
    )


# ======================================================================
# PERFECT DATA RESIDUAL
# ======================================================================


def test_exact_measurements_have_near_zero_residual() -> None:

    source = (
        0.5,
        0.30,
    )

    result = solve_position(
        NODE_POSITIONS,
        make_all_pair_measurements(
            source
        ),
        speed_of_sound_mps=
            SPEED_OF_SOUND_MPS,
    )

    assert (
        result.success
    )

    assert (
        result.residual_rms_seconds
        < 1e-8
    )

    assert (
        result.residual_rms_meters
        < 1e-5
    )