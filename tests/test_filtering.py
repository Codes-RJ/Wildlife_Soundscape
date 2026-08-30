"""
Tests for localization.filtering.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. pass-band frequency preservation
    2. low-frequency attenuation
    3. high-frequency attenuation
    4. mixed-signal filtering
    5. zero-signal behaviour
    6. output shape preservation
    7. finite floating-point output
    8. input-array immutability
    9. deterministic behaviour
    10. sample-rate validation
    11. cutoff-frequency validation
    12. Nyquist-frequency validation
    13. filter-order validation
    14. input dimensionality validation
    15. empty-input validation
    16. non-finite-input validation

Role in localization
--------------------
The localization chain is:

    synchronized PCM windows
            ↓
    band-pass filtering
            ↓
    GCC-PHAT
            ↓
    TDOA validation
            ↓
    nonlinear position solver

The filter is intentionally tested independently from GCC-PHAT so a
correlation failure can be distinguished from a preprocessing failure.
"""


from __future__ import annotations


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np
import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from localization.filtering import (
    bandpass_filter,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SAMPLE_RATE = (
    48_000
)


LOW_CUTOFF_HZ = (
    200.0
)


HIGH_CUTOFF_HZ = (
    12_000.0
)


FILTER_ORDER = (
    4
)


SIGNAL_DURATION_S = (
    0.5
)


# ======================================================================
# TEST HELPERS
# ======================================================================


def make_tone(
    frequency_hz: float,
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_s: float = SIGNAL_DURATION_S,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """
    Generate a deterministic real-valued sinusoid.
    """

    sample_count = int(
        round(
            float(
                sample_rate
            )
            * float(
                duration_s
            )
        )
    )

    sample_index = np.arange(
        sample_count,
        dtype=np.float64,
    )

    time_s = (
        sample_index
        / float(
            sample_rate
        )
    )

    signal = (
        float(
            amplitude
        )
        * np.sin(
            (
                2.0
                * np.pi
                * float(
                    frequency_hz
                )
                * time_s
            )
            + float(
                phase_rad
            )
        )
    )

    return np.ascontiguousarray(
        signal,
        dtype=np.float64,
    )


def calculate_rms(
    signal: np.ndarray,
) -> float:
    """
    RMS using float64 accumulation.
    """

    values = np.asarray(
        signal,
        dtype=np.float64,
    )

    if (
        values.size
        == 0
    ):

        return (
            0.0
        )

    return float(
        np.sqrt(
            np.mean(
                values
                * values,
                dtype=np.float64,
            )
        )
    )


def run_standard_filter(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Apply the standard localization test band-pass.
    """

    return bandpass_filter(
        signal,
        sample_rate=
            SAMPLE_RATE,

        low_hz=
            LOW_CUTOFF_HZ,

        high_hz=
            HIGH_CUTOFF_HZ,

        order=
            FILTER_ORDER,
    )


# ======================================================================
# ORIGINAL REGRESSION TEST
# ======================================================================


def test_bandpass_attenuates_out_of_band_tone() -> None:
    """
    Preserve the intent of the original test.

    A 2 kHz signal lies well inside the 200 Hz - 12 kHz pass band.

    A 50 Hz signal lies below it.
    """

    inside = make_tone(
        2000.0
    )

    outside = make_tone(
        50.0
    )

    y_inside = run_standard_filter(
        inside
    )

    y_outside = run_standard_filter(
        outside
    )

    assert (
        calculate_rms(
            y_inside
        )
        > (
            5.0
            * calculate_rms(
                y_outside
            )
        )
    )


# ======================================================================
# PASS-BAND PRESERVATION
# ======================================================================


@pytest.mark.parametrize(
    "frequency_hz",
    [
        500.0,
        1000.0,
        2000.0,
        5000.0,
        8000.0,
    ],
)
def test_passband_tones_retain_significant_energy(
    frequency_hz: float,
) -> None:
    """
    Frequencies comfortably inside the pass band should not be heavily
    suppressed.
    """

    signal = make_tone(
        frequency_hz
    )

    filtered = run_standard_filter(
        signal
    )

    input_rms = calculate_rms(
        signal
    )

    output_rms = calculate_rms(
        filtered
    )

    assert (
        output_rms
        > (
            input_rms
            * 0.50
        )
    )


# ======================================================================
# LOW-FREQUENCY ATTENUATION
# ======================================================================


@pytest.mark.parametrize(
    "frequency_hz",
    [
        20.0,
        50.0,
        80.0,
    ],
)
def test_low_frequency_tones_are_attenuated(
    frequency_hz: float,
) -> None:

    signal = make_tone(
        frequency_hz
    )

    filtered = run_standard_filter(
        signal
    )

    assert (
        calculate_rms(
            filtered
        )
        < (
            calculate_rms(
                signal
            )
            * 0.25
        )
    )


# ======================================================================
# HIGH-FREQUENCY ATTENUATION
# ======================================================================


@pytest.mark.parametrize(
    "frequency_hz",
    [
        15_000.0,
        18_000.0,
        20_000.0,
    ],
)
def test_high_frequency_tones_are_attenuated(
    frequency_hz: float,
) -> None:

    signal = make_tone(
        frequency_hz
    )

    filtered = run_standard_filter(
        signal
    )

    assert (
        calculate_rms(
            filtered
        )
        < (
            calculate_rms(
                signal
            )
            * 0.30
        )
    )


# ======================================================================
# MIXED SIGNAL
# ======================================================================


def test_bandpass_retains_inband_component_from_mixed_signal() -> None:
    """
    Construct:

        50 Hz
        + 2 kHz
        + 18 kHz

    The filtered result should correlate strongly with the desired
    2 kHz component.
    """

    low = make_tone(
        50.0,
        amplitude=
            1.0,
    )

    desired = make_tone(
        2000.0,
        amplitude=
            1.0,
    )

    high = make_tone(
        18_000.0,
        amplitude=
            1.0,
    )

    mixed = (
        low
        + desired
        + high
    )

    filtered = run_standard_filter(
        mixed
    )

    # --------------------------------------------------------------
    # NORMALIZED CORRELATION
    # --------------------------------------------------------------

    numerator = float(
        np.dot(
            filtered,
            desired,
        )
    )

    denominator = (
        float(
            np.linalg.norm(
                filtered
            )
        )
        * float(
            np.linalg.norm(
                desired
            )
        )
    )

    assert (
        denominator
        > 0.0
    )

    correlation = (
        numerator
        / denominator
    )

    assert (
        correlation
        > 0.90
    )


# ======================================================================
# ZERO SIGNAL
# ======================================================================


def test_zero_signal_remains_zero() -> None:

    signal = np.zeros(
        4096,
        dtype=np.float64,
    )

    filtered = run_standard_filter(
        signal
    )

    np.testing.assert_allclose(
        filtered,
        0.0,
        atol=
            1e-12,
        rtol=
            0.0,
    )


# ======================================================================
# OUTPUT CONTRACT
# ======================================================================


def test_filter_preserves_signal_shape() -> None:

    signal = make_tone(
        2000.0
    )

    filtered = run_standard_filter(
        signal
    )

    assert (
        filtered.shape
        == signal.shape
    )


def test_filter_returns_one_dimensional_signal() -> None:

    signal = make_tone(
        2000.0
    )

    filtered = run_standard_filter(
        signal
    )

    assert (
        filtered.ndim
        == 1
    )


def test_filter_output_is_real_floating_point() -> None:

    signal = make_tone(
        2000.0
    )

    filtered = run_standard_filter(
        signal
    )

    assert (
        np.issubdtype(
            filtered.dtype,
            np.floating,
        )
    )

    assert not (
        np.iscomplexobj(
            filtered
        )
    )


def test_filter_output_is_finite() -> None:

    signal = make_tone(
        3000.0
    )

    filtered = run_standard_filter(
        signal
    )

    assert np.all(
        np.isfinite(
            filtered
        )
    )


# ======================================================================
# INPUT IMMUTABILITY
# ======================================================================


def test_filter_does_not_modify_input_array() -> None:

    signal = make_tone(
        2500.0
    )

    original = (
        signal.copy()
    )

    run_standard_filter(
        signal
    )

    np.testing.assert_array_equal(
        signal,
        original,
    )


# ======================================================================
# DETERMINISM
# ======================================================================


def test_filter_is_deterministic() -> None:

    signal = make_tone(
        3500.0
    )

    first = run_standard_filter(
        signal
    )

    second = run_standard_filter(
        signal
    )

    np.testing.assert_allclose(
        first,
        second,
        rtol=
            0.0,
        atol=
            0.0,
    )


# ======================================================================
# SAMPLE-RATE VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "sample_rate",
    [
        0,
        -1,
        -48_000,
    ],
)
def test_rejects_nonpositive_sample_rate(
    sample_rate: int,
) -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                sample_rate,

            low_hz=
                LOW_CUTOFF_HZ,

            high_hz=
                HIGH_CUTOFF_HZ,

            order=
                FILTER_ORDER,
        )


@pytest.mark.parametrize(
    "sample_rate",
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
def test_rejects_nonfinite_sample_rate(
    sample_rate: float,
) -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                sample_rate,

            low_hz=
                LOW_CUTOFF_HZ,

            high_hz=
                HIGH_CUTOFF_HZ,

            order=
                FILTER_ORDER,
        )


# ======================================================================
# LOW CUTOFF VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "low_hz",
    [
        0.0,
        -1.0,
        -200.0,
    ],
)
def test_rejects_nonpositive_low_cutoff(
    low_hz: float,
) -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                SAMPLE_RATE,

            low_hz=
                low_hz,

            high_hz=
                HIGH_CUTOFF_HZ,

            order=
                FILTER_ORDER,
        )


@pytest.mark.parametrize(
    "low_hz",
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
def test_rejects_nonfinite_low_cutoff(
    low_hz: float,
) -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                SAMPLE_RATE,

            low_hz=
                low_hz,

            high_hz=
                HIGH_CUTOFF_HZ,

            order=
                FILTER_ORDER,
        )


# ======================================================================
# HIGH CUTOFF VALIDATION
# ======================================================================


def test_rejects_high_cutoff_equal_to_low_cutoff() -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                SAMPLE_RATE,

            low_hz=
                2000.0,

            high_hz=
                2000.0,

            order=
                FILTER_ORDER,
        )


def test_rejects_high_cutoff_below_low_cutoff() -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                SAMPLE_RATE,

            low_hz=
                5000.0,

            high_hz=
                1000.0,

            order=
                FILTER_ORDER,
        )


@pytest.mark.parametrize(
    "high_hz",
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
def test_rejects_nonfinite_high_cutoff(
    high_hz: float,
) -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                SAMPLE_RATE,

            low_hz=
                LOW_CUTOFF_HZ,

            high_hz=
                high_hz,

            order=
                FILTER_ORDER,
        )


# ======================================================================
# NYQUIST VALIDATION
# ======================================================================


def test_rejects_high_cutoff_equal_to_nyquist() -> None:

    signal = make_tone(
        2000.0
    )

    nyquist = (
        SAMPLE_RATE
        / 2.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                SAMPLE_RATE,

            low_hz=
                LOW_CUTOFF_HZ,

            high_hz=
                nyquist,

            order=
                FILTER_ORDER,
        )


def test_rejects_high_cutoff_above_nyquist() -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                SAMPLE_RATE,

            low_hz=
                LOW_CUTOFF_HZ,

            high_hz=
                25_000.0,

            order=
                FILTER_ORDER,
        )


# ======================================================================
# FILTER ORDER VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "order",
    [
        0,
        -1,
        -4,
    ],
)
def test_rejects_nonpositive_filter_order(
    order: int,
) -> None:

    signal = make_tone(
        2000.0
    )

    with pytest.raises(
        ValueError
    ):

        bandpass_filter(
            signal,
            sample_rate=
                SAMPLE_RATE,

            low_hz=
                LOW_CUTOFF_HZ,

            high_hz=
                HIGH_CUTOFF_HZ,

            order=
                order,
        )


# ======================================================================
# SIGNAL SHAPE VALIDATION
# ======================================================================


def test_rejects_two_dimensional_signal() -> None:

    signal = np.zeros(
        (
            1024,
            2,
        ),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError
    ):

        run_standard_filter(
            signal
        )


def test_rejects_empty_signal() -> None:

    signal = np.array(
        [],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError
    ):

        run_standard_filter(
            signal
        )


# ======================================================================
# NON-FINITE AUDIO VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "bad_value",
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
def test_rejects_nonfinite_audio(
    bad_value: float,
) -> None:

    signal = make_tone(
        2000.0
    )

    signal[
        100
    ] = (
        bad_value
    )

    with pytest.raises(
        ValueError
    ):

        run_standard_filter(
            signal
        )