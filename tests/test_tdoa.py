"""
Tests for localization.tdoa.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. TDOAMeasurement construction
    2. positive and negative delay sign preservation
    3. node-pair validation
    4. finite delay validation
    5. peak-ratio validation
    6. maximum-delay validation
    7. physical-bound enforcement
    8. invalid/rejected measurement representation
    9. physical_max_delay geometry
    10. speed-of-sound validation
    11. zero-distance node geometry
    12. distance symmetry

Sign convention
---------------
For:

    node_a = A
    node_b = B

the measurement is:

    delay_seconds
        = arrival_time_B - arrival_time_A

Therefore:

    positive delay
        B received the acoustic waveform later than A

    negative delay
        B received the acoustic waveform earlier than A

This convention must remain consistent with:

    GCC-PHAT
        ↓
    TDOAMeasurement
        ↓
    solve_position()
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math


# ======================================================================
# THIRD-PARTY
# ======================================================================


import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.localization.tdoa import (
    TDOAMeasurement,
    physical_max_delay,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SAMPLE_RATE = (
    48_000.0
)


SPEED_OF_SOUND_MPS = (
    343.0
)


# ======================================================================
# TEST HELPERS
# ======================================================================


def make_measurement(
    *,
    node_a: int = 1,
    node_b: int = 2,
    delay_samples: float = 4.0,
    peak_ratio: float = 2.0,
    max_delay_seconds: float = 0.001,
    valid: bool = True,
    reason: str = "",
) -> TDOAMeasurement:
    """
    Construct one deterministic TDOA measurement.

    delay_seconds is derived from delay_samples so tests keep the two
    representations numerically consistent.
    """

    delay_seconds = (
        float(
            delay_samples
        )
        / SAMPLE_RATE
    )

    return TDOAMeasurement(
        node_a=
            node_a,

        node_b=
            node_b,

        delay_seconds=
            delay_seconds,

        delay_samples=
            float(
                delay_samples
            ),

        peak_ratio=
            peak_ratio,

        max_delay_seconds=
            max_delay_seconds,

        valid=
            valid,

        reason=
            reason,
    )


# ======================================================================
# BASIC MEASUREMENT CONTRACT
# ======================================================================


def test_tdoa_measurement_accepts_valid_values() -> None:

    measurement = make_measurement(
        node_a=
            1,

        node_b=
            2,

        delay_samples=
            5.5,

        peak_ratio=
            2.25,

        max_delay_seconds=
            0.002,
    )

    assert (
        measurement.node_a
        == 1
    )

    assert (
        measurement.node_b
        == 2
    )

    assert (
        measurement.delay_samples
        == pytest.approx(
            5.5
        )
    )

    assert (
        measurement.delay_seconds
        == pytest.approx(
            5.5
            / SAMPLE_RATE
        )
    )

    assert (
        measurement.peak_ratio
        == pytest.approx(
            2.25
        )
    )

    assert (
        measurement.max_delay_seconds
        == pytest.approx(
            0.002
        )
    )

    assert (
        measurement.valid
        is True
    )

    assert (
        measurement.reason
        == ""
    )


# ======================================================================
# SIGN CONVENTION
# ======================================================================


def test_positive_delay_is_preserved() -> None:
    """
    Positive:

        arrival_B > arrival_A
    """

    measurement = make_measurement(
        delay_samples=
            12.0
    )

    assert (
        measurement.delay_samples
        > 0.0
    )

    assert (
        measurement.delay_seconds
        > 0.0
    )


def test_negative_delay_is_preserved() -> None:
    """
    Negative:

        arrival_B < arrival_A
    """

    measurement = make_measurement(
        delay_samples=
            -12.0
    )

    assert (
        measurement.delay_samples
        < 0.0
    )

    assert (
        measurement.delay_seconds
        < 0.0
    )


def test_zero_delay_is_valid() -> None:

    measurement = make_measurement(
        delay_samples=
            0.0
    )

    assert (
        measurement.delay_samples
        == 0.0
    )

    assert (
        measurement.delay_seconds
        == 0.0
    )

    assert (
        measurement.valid
    )


# ======================================================================
# NODE ID VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "node_id",
    [
        1,
        2,
        3,
        255,
    ],
)
def test_valid_node_ids_are_accepted(
    node_id: int,
) -> None:

    other_node = (
        1
        if node_id
        != 1
        else 2
    )

    measurement = make_measurement(
        node_a=
            node_id,

        node_b=
            other_node,
    )

    assert (
        measurement.node_a
        == node_id
    )


@pytest.mark.parametrize(
    "node_id",
    [
        0,
        -1,
        256,
        1000,
    ],
)
def test_rejects_invalid_node_a(
    node_id: int,
) -> None:

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            node_a=
                node_id,

            node_b=
                2,
        )


@pytest.mark.parametrize(
    "node_id",
    [
        0,
        -1,
        256,
        1000,
    ],
)
def test_rejects_invalid_node_b(
    node_id: int,
) -> None:

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            node_a=
                1,

            node_b=
                node_id,
        )


@pytest.mark.parametrize(
    "node_id",
    [
        True,
        False,
        1.0,
        2.5,
        "1",
    ],
)
def test_rejects_non_integer_node_id(
    node_id,
) -> None:

    with pytest.raises(
        TypeError
    ):

        make_measurement(
            node_a=
                node_id,

            node_b=
                2,
        )


def test_rejects_same_node_pair() -> None:
    """
    TDOA requires two different sensors.
    """

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            node_a=
                2,

            node_b=
                2,
        )


# ======================================================================
# DELAY VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "bad_delay_seconds",
    [
        float(
            "nan"
        ),
        float(
            "inf"
        ),
        float(
            "-inf"
        ),
    ],
)
def test_rejects_non_finite_delay_seconds(
    bad_delay_seconds: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        TDOAMeasurement(
            node_a=
                1,

            node_b=
                2,

            delay_seconds=
                bad_delay_seconds,

            delay_samples=
                1.0,

            peak_ratio=
                2.0,

            max_delay_seconds=
                0.001,

            valid=
                True,

            reason=
                "",
        )


@pytest.mark.parametrize(
    "bad_delay_samples",
    [
        float(
            "nan"
        ),
        float(
            "inf"
        ),
        float(
            "-inf"
        ),
    ],
)
def test_rejects_non_finite_delay_samples(
    bad_delay_samples: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        TDOAMeasurement(
            node_a=
                1,

            node_b=
                2,

            delay_seconds=
                0.0001,

            delay_samples=
                bad_delay_samples,

            peak_ratio=
                2.0,

            max_delay_seconds=
                0.001,

            valid=
                True,

            reason=
                "",
        )


# ======================================================================
# PEAK-RATIO VALIDATION
# ======================================================================


def test_zero_peak_ratio_can_be_stored_for_rejected_measurement() -> None:

    measurement = make_measurement(
        peak_ratio=
            0.0,

        valid=
            False,

        reason=
            "weak correlation",
    )

    assert (
        measurement.peak_ratio
        == 0.0
    )

    assert not (
        measurement.valid
    )


@pytest.mark.parametrize(
    "peak_ratio",
    [
        -0.01,
        -1.0,
    ],
)
def test_rejects_negative_peak_ratio(
    peak_ratio: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            peak_ratio=
                peak_ratio,

            valid=
                False,
        )


@pytest.mark.parametrize(
    "peak_ratio",
    [
        float(
            "nan"
        ),
        float(
            "inf"
        ),
        float(
            "-inf"
        ),
    ],
)
def test_rejects_non_finite_peak_ratio(
    peak_ratio: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            peak_ratio=
                peak_ratio,

            valid=
                False,
        )


# ======================================================================
# MAXIMUM DELAY VALIDATION
# ======================================================================


def test_zero_max_delay_accepts_zero_delay() -> None:

    measurement = make_measurement(
        delay_samples=
            0.0,

        max_delay_seconds=
            0.0,
    )

    assert (
        measurement.max_delay_seconds
        == 0.0
    )

    assert (
        measurement.valid
    )


def test_rejects_negative_max_delay() -> None:

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            max_delay_seconds=
                -0.001,

            valid=
                False,
        )


@pytest.mark.parametrize(
    "max_delay",
    [
        float(
            "nan"
        ),
        float(
            "inf"
        ),
        float(
            "-inf"
        ),
    ],
)
def test_rejects_non_finite_max_delay(
    max_delay: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            max_delay_seconds=
                max_delay,

            valid=
                False,
        )


# ======================================================================
# PHYSICAL BOUND ENFORCEMENT
# ======================================================================


def test_valid_measurement_inside_physical_bound() -> None:

    maximum = (
        10.0
        / SAMPLE_RATE
    )

    measurement = make_measurement(
        delay_samples=
            8.0,

        max_delay_seconds=
            maximum,

        valid=
            True,
    )

    assert (
        abs(
            measurement.delay_seconds
        )
        < measurement.max_delay_seconds
    )


def test_valid_measurement_exactly_at_positive_physical_bound() -> None:

    maximum_samples = (
        10.0
    )

    measurement = make_measurement(
        delay_samples=
            maximum_samples,

        max_delay_seconds=
            maximum_samples
            / SAMPLE_RATE,

        valid=
            True,
    )

    assert (
        measurement.delay_seconds
        == pytest.approx(
            measurement.max_delay_seconds
        )
    )


def test_valid_measurement_exactly_at_negative_physical_bound() -> None:

    maximum_samples = (
        10.0
    )

    measurement = make_measurement(
        delay_samples=
            -maximum_samples,

        max_delay_seconds=
            maximum_samples
            / SAMPLE_RATE,

        valid=
            True,
    )

    assert (
        abs(
            measurement.delay_seconds
        )
        == pytest.approx(
            measurement.max_delay_seconds
        )
    )


def test_rejects_valid_measurement_outside_positive_physical_bound() -> None:

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            delay_samples=
                20.0,

            max_delay_seconds=
                10.0
                / SAMPLE_RATE,

            valid=
                True,
        )


def test_rejects_valid_measurement_outside_negative_physical_bound() -> None:

    with pytest.raises(
        ValueError
    ):

        make_measurement(
            delay_samples=
                -20.0,

            max_delay_seconds=
                10.0
                / SAMPLE_RATE,

            valid=
                True,
        )


def test_rejected_measurement_may_store_out_of_bounds_delay() -> None:
    """
    A rejected GCC result may still be retained diagnostically even when
    its estimated delay lies outside the physical pair limit.

    The important rule is that such a measurement cannot be marked
    valid.
    """

    measurement = make_measurement(
        delay_samples=
            20.0,

        max_delay_seconds=
            10.0
            / SAMPLE_RATE,

        valid=
            False,

        reason=
            "outside physical delay bound",
    )

    assert not (
        measurement.valid
    )

    assert (
        abs(
            measurement.delay_seconds
        )
        > measurement.max_delay_seconds
    )

    assert (
        measurement.reason
        == "outside physical delay bound"
    )


# ======================================================================
# REJECTED MEASUREMENT METADATA
# ======================================================================


def test_rejected_measurement_preserves_reason() -> None:

    reason = (
        "correlation peak ratio below threshold"
    )

    measurement = make_measurement(
        valid=
            False,

        reason=
            reason,
    )

    assert not (
        measurement.valid
    )

    assert (
        measurement.reason
        == reason
    )


# ======================================================================
# PHYSICAL MAXIMUM DELAY
# ======================================================================


def test_physical_max_delay_for_one_meter_pair() -> None:
    """
    For nodes separated by one meter:

        tau_max = 1 / 343 seconds
    """

    result = physical_max_delay(
        (
            0.0,
            0.0,
        ),
        (
            1.0,
            0.0,
        ),
        SPEED_OF_SOUND_MPS,
    )

    expected = (
        1.0
        / SPEED_OF_SOUND_MPS
    )

    assert (
        result
        == pytest.approx(
            expected,
            rel=
                1e-12,
        )
    )


def test_physical_max_delay_uses_euclidean_distance() -> None:
    """
    Node separation:

        dx = 3 m
        dy = 4 m

    Euclidean distance = 5 m.
    """

    result = physical_max_delay(
        (
            0.0,
            0.0,
        ),
        (
            3.0,
            4.0,
        ),
        SPEED_OF_SOUND_MPS,
    )

    expected = (
        5.0
        / SPEED_OF_SOUND_MPS
    )

    assert (
        result
        == pytest.approx(
            expected,
            rel=
                1e-12,
        )
    )


def test_physical_max_delay_is_symmetric() -> None:

    node_a = (
        -1.5,
        2.25,
    )

    node_b = (
        4.0,
        -3.0,
    )

    delay_ab = physical_max_delay(
        node_a,
        node_b,
        SPEED_OF_SOUND_MPS,
    )

    delay_ba = physical_max_delay(
        node_b,
        node_a,
        SPEED_OF_SOUND_MPS,
    )

    assert (
        delay_ab
        == pytest.approx(
            delay_ba,
            rel=
                1e-15,
        )
    )


def test_physical_max_delay_for_same_position_is_zero() -> None:

    node = (
        0.5,
        0.8660254037844386,
    )

    result = physical_max_delay(
        node,
        node,
        SPEED_OF_SOUND_MPS,
    )

    assert (
        result
        == 0.0
    )


# ======================================================================
# GEOMETRY VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "position",
    [
        (
            0.0,
        ),

        (
            0.0,
            1.0,
            2.0,
        ),

        (),

        "0,0",
    ],
)
def test_physical_max_delay_rejects_non_2d_position(
    position,
) -> None:

    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):

        physical_max_delay(
            position,
            (
                1.0,
                0.0,
            ),
            SPEED_OF_SOUND_MPS,
        )


@pytest.mark.parametrize(
    "coordinate",
    [
        float(
            "nan"
        ),
        float(
            "inf"
        ),
        float(
            "-inf"
        ),
    ],
)
def test_physical_max_delay_rejects_non_finite_coordinate(
    coordinate: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        physical_max_delay(
            (
                0.0,
                coordinate,
            ),
            (
                1.0,
                0.0,
            ),
            SPEED_OF_SOUND_MPS,
        )


# ======================================================================
# SPEED OF SOUND VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "speed",
    [
        0.0,
        -1.0,
        -343.0,
    ],
)
def test_physical_max_delay_rejects_nonpositive_speed_of_sound(
    speed: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        physical_max_delay(
            (
                0.0,
                0.0,
            ),
            (
                1.0,
                0.0,
            ),
            speed,
        )


@pytest.mark.parametrize(
    "speed",
    [
        float(
            "nan"
        ),
        float(
            "inf"
        ),
        float(
            "-inf"
        ),
    ],
)
def test_physical_max_delay_rejects_non_finite_speed_of_sound(
    speed: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        physical_max_delay(
            (
                0.0,
                0.0,
            ),
            (
                1.0,
                0.0,
            ),
            speed,
        )


# ======================================================================
# DIFFERENT SPEEDS OF SOUND
# ======================================================================


def test_higher_speed_of_sound_reduces_maximum_delay() -> None:

    node_a = (
        0.0,
        0.0,
    )

    node_b = (
        1.0,
        0.0,
    )

    slower = physical_max_delay(
        node_a,
        node_b,
        330.0,
    )

    faster = physical_max_delay(
        node_a,
        node_b,
        350.0,
    )

    assert (
        faster
        < slower
    )


# ======================================================================
# PROJECT ARRAY GEOMETRY
# ======================================================================


def test_one_meter_equilateral_array_has_equal_pair_limits(
    node_positions,
    speed_of_sound_mps,
) -> None:
    """
    The shared test geometry is an equilateral triangle with one-meter
    sides.

    Therefore every microphone pair should have the same physical
    propagation-delay limit.
    """

    delay_12 = physical_max_delay(
        node_positions[
            1
        ],
        node_positions[
            2
        ],
        speed_of_sound_mps,
    )

    delay_13 = physical_max_delay(
        node_positions[
            1
        ],
        node_positions[
            3
        ],
        speed_of_sound_mps,
    )

    delay_23 = physical_max_delay(
        node_positions[
            2
        ],
        node_positions[
            3
        ],
        speed_of_sound_mps,
    )

    expected = (
        1.0
        / speed_of_sound_mps
    )

    assert (
        delay_12
        == pytest.approx(
            expected,
            rel=
                1e-12,
        )
    )

    assert (
        delay_13
        == pytest.approx(
            expected,
            rel=
                1e-12,
        )
    )

    assert (
        delay_23
        == pytest.approx(
            expected,
            rel=
                1e-9,
        )
    )


# ======================================================================
# SAMPLE-DOMAIN INTERPRETATION
# ======================================================================


def test_one_meter_pair_physical_limit_in_samples() -> None:
    """
    At 48 kHz and c = 343 m/s:

        maximum delay
            ≈ 0.002915 s

        maximum samples
            ≈ 139.94 samples

    This illustrates why the GCC-PHAT physical search region depends on
    actual microphone spacing.
    """

    maximum_seconds = physical_max_delay(
        (
            0.0,
            0.0,
        ),
        (
            1.0,
            0.0,
        ),
        SPEED_OF_SOUND_MPS,
    )

    maximum_samples = (
        maximum_seconds
        * SAMPLE_RATE
    )

    expected_samples = (
        SAMPLE_RATE
        / SPEED_OF_SOUND_MPS
    )

    assert (
        maximum_samples
        == pytest.approx(
            expected_samples,
            rel=
                1e-12,
        )
    )

    assert (
        maximum_samples
        == pytest.approx(
            139.94,
            abs=
                0.02,
        )
    )


# ======================================================================
# FINITE RESULT
# ======================================================================


def test_physical_max_delay_is_finite_for_valid_geometry() -> None:

    result = physical_max_delay(
        (
            -10.0,
            5.0,
        ),
        (
            20.0,
            -12.0,
        ),
        SPEED_OF_SOUND_MPS,
    )

    assert (
        math.isfinite(
            result
        )
    )

    assert (
        result
        >= 0.0
    )