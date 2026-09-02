"""
Tests for environment.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. temperature effect on speed of sound
    2. humidity effect
    3. relative-humidity clamping
    4. positive-pressure validation
    5. non-finite input rejection
    6. deterministic behaviour
    7. finite/positive output
    8. representative environmental conditions

Scientific scope
----------------
The environmental model is an engineering approximation used to improve
the localization sound-speed estimate.

These tests verify software behaviour and physical plausibility.

They do not claim laboratory-grade atmospheric acoustics accuracy.
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


from wildlife_soundscape.core.environment import (
    calculate_speed_of_sound_mps,
)


# ======================================================================
# CONSTANTS
# ======================================================================


STANDARD_PRESSURE_HPA = (
    1013.25
)


# ======================================================================
# TEMPERATURE EFFECT
# ======================================================================


def test_speed_of_sound_increases_with_temperature() -> None:

    cold = calculate_speed_of_sound_mps(
        10.0,
        50.0,
        STANDARD_PRESSURE_HPA,
    )

    warm = calculate_speed_of_sound_mps(
        30.0,
        50.0,
        STANDARD_PRESSURE_HPA,
    )

    assert (
        warm
        > cold
    )

    assert (
        330.0
        < cold
        < 350.0
    )

    assert (
        345.0
        < warm
        < 360.0
    )


def test_temperature_change_has_meaningful_effect() -> None:
    """
    Temperature should affect localization sound speed by substantially
    more than numerical floating-point noise.
    """

    low = calculate_speed_of_sound_mps(
        5.0,
        50.0,
        STANDARD_PRESSURE_HPA,
    )

    high = calculate_speed_of_sound_mps(
        35.0,
        50.0,
        STANDARD_PRESSURE_HPA,
    )

    assert (
        high
        - low
        > 10.0
    )


# ======================================================================
# HUMIDITY EFFECT
# ======================================================================


def test_humidity_has_small_positive_effect() -> None:

    dry = calculate_speed_of_sound_mps(
        25.0,
        10.0,
        STANDARD_PRESSURE_HPA,
    )

    humid = calculate_speed_of_sound_mps(
        25.0,
        90.0,
        STANDARD_PRESSURE_HPA,
    )

    assert (
        humid
        > dry
    )

    assert (
        humid
        - dry
        < 3.0
    )


def test_humidity_effect_is_smaller_than_large_temperature_change() -> None:

    humidity_effect = (
        calculate_speed_of_sound_mps(
            25.0,
            90.0,
            STANDARD_PRESSURE_HPA,
        )
        - calculate_speed_of_sound_mps(
            25.0,
            10.0,
            STANDARD_PRESSURE_HPA,
        )
    )

    temperature_effect = (
        calculate_speed_of_sound_mps(
            35.0,
            50.0,
            STANDARD_PRESSURE_HPA,
        )
        - calculate_speed_of_sound_mps(
            15.0,
            50.0,
            STANDARD_PRESSURE_HPA,
        )
    )

    assert (
        temperature_effect
        > humidity_effect
    )


# ======================================================================
# HUMIDITY CLAMPING
# ======================================================================


def test_humidity_below_zero_is_clamped_to_zero() -> None:
    """
    Relative humidity below 0% is not physically meaningful.

    The finalized environmental model clamps RH into [0, 100].
    """

    below_zero = calculate_speed_of_sound_mps(
        25.0,
        -50.0,
        STANDARD_PRESSURE_HPA,
    )

    zero = calculate_speed_of_sound_mps(
        25.0,
        0.0,
        STANDARD_PRESSURE_HPA,
    )

    assert (
        below_zero
        == pytest.approx(
            zero,
            rel=
                1e-12,
            abs=
                1e-12,
        )
    )


def test_humidity_above_one_hundred_is_clamped_to_one_hundred() -> None:

    above = calculate_speed_of_sound_mps(
        25.0,
        150.0,
        STANDARD_PRESSURE_HPA,
    )

    hundred = calculate_speed_of_sound_mps(
        25.0,
        100.0,
        STANDARD_PRESSURE_HPA,
    )

    assert (
        above
        == pytest.approx(
            hundred,
            rel=
                1e-12,
            abs=
                1e-12,
        )
    )


# ======================================================================
# REPRESENTATIVE CONDITIONS
# ======================================================================


@pytest.mark.parametrize(
    (
        "temperature_c",
        "humidity_percent",
        "pressure_hpa",
    ),
    [
        (
            0.0,
            30.0,
            1013.25,
        ),
        (
            15.0,
            50.0,
            1013.25,
        ),
        (
            25.0,
            70.0,
            1000.0,
        ),
        (
            35.0,
            90.0,
            995.0,
        ),
        (
            45.0,
            20.0,
            1020.0,
        ),
    ],
)
def test_representative_environment_returns_finite_positive_speed(
    temperature_c: float,
    humidity_percent: float,
    pressure_hpa: float,
) -> None:

    result = calculate_speed_of_sound_mps(
        temperature_c,
        humidity_percent,
        pressure_hpa,
    )

    assert (
        math.isfinite(
            result
        )
    )

    assert (
        result
        > 0.0
    )


def test_typical_room_condition_is_physically_reasonable() -> None:

    result = calculate_speed_of_sound_mps(
        20.0,
        50.0,
        STANDARD_PRESSURE_HPA,
    )

    # Broad engineering sanity range rather than pinning the test to one
    # exact atmospheric formula.
    assert (
        335.0
        < result
        < 350.0
    )


# ======================================================================
# PRESSURE VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "pressure_hpa",
    [
        0.0,
        -1.0,
        -1013.25,
    ],
)
def test_rejects_nonpositive_pressure(
    pressure_hpa: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        calculate_speed_of_sound_mps(
            25.0,
            50.0,
            pressure_hpa,
        )


def test_different_positive_pressures_produce_valid_results() -> None:
    """
    Avoid asserting an overly specific pressure trend here.

    The environmental module uses an engineering approximation and the
    important interface contract is that valid positive atmospheric
    pressures yield finite sound-speed estimates.
    """

    low_pressure = calculate_speed_of_sound_mps(
        25.0,
        50.0,
        900.0,
    )

    standard_pressure = calculate_speed_of_sound_mps(
        25.0,
        50.0,
        STANDARD_PRESSURE_HPA,
    )

    high_pressure = calculate_speed_of_sound_mps(
        25.0,
        50.0,
        1100.0,
    )

    for value in (
        low_pressure,
        standard_pressure,
        high_pressure,
    ):

        assert (
            math.isfinite(
                value
            )
        )

        assert (
            value
            > 0.0
        )


# ======================================================================
# NON-FINITE TEMPERATURE
# ======================================================================


@pytest.mark.parametrize(
    "temperature_c",
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
def test_rejects_non_finite_temperature(
    temperature_c: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        calculate_speed_of_sound_mps(
            temperature_c,
            50.0,
            STANDARD_PRESSURE_HPA,
        )


# ======================================================================
# NON-FINITE HUMIDITY
# ======================================================================


@pytest.mark.parametrize(
    "humidity_percent",
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
def test_rejects_non_finite_humidity(
    humidity_percent: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        calculate_speed_of_sound_mps(
            25.0,
            humidity_percent,
            STANDARD_PRESSURE_HPA,
        )


# ======================================================================
# NON-FINITE PRESSURE
# ======================================================================


@pytest.mark.parametrize(
    "pressure_hpa",
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
def test_rejects_non_finite_pressure(
    pressure_hpa: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        calculate_speed_of_sound_mps(
            25.0,
            50.0,
            pressure_hpa,
        )


# ======================================================================
# DETERMINISM
# ======================================================================


def test_speed_of_sound_calculation_is_deterministic() -> None:

    first = calculate_speed_of_sound_mps(
        27.5,
        63.0,
        1007.5,
    )

    second = calculate_speed_of_sound_mps(
        27.5,
        63.0,
        1007.5,
    )

    assert (
        first
        == second
    )


# ======================================================================
# LOCALIZATION-RELEVANT SANITY
# ======================================================================


def test_environmental_speed_is_close_to_nominal_localization_value() -> None:
    """
    The fixed localization fallback is approximately 343 m/s.

    Normal room conditions should therefore produce a value in the same
    general range rather than an unrealistic atmospheric estimate.
    """

    result = calculate_speed_of_sound_mps(
        20.0,
        50.0,
        STANDARD_PRESSURE_HPA,
    )

    assert (
        abs(
            result
            - 343.0
        )
        < 10.0
    )