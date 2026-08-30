"""
Tests for localization.gcc_phat.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. GCCPHATResult return contract
    2. zero-delay estimation
    3. positive integer delay estimation
    4. negative integer delay estimation
    5. fractional / sub-sample delay estimation
    6. delay sign convention
    7. delay-seconds / delay-samples consistency
    8. interpolation consistency
    9. physically constrained lag search
    10. strong broadband correlation quality
    11. silence / zero-energy rejection
    12. unequal-length rejection
    13. multidimensional-input rejection
    14. non-finite-input rejection
    15. sample-rate validation
    16. interpolation validation
    17. peak-ratio threshold validation
    18. epsilon validation
    19. maximum-delay validation
    20. input arrays are not modified
    21. deterministic behaviour

Sign convention
---------------
The finalized localization contract defines:

    positive delay
        signal arrived LATER than reference

    negative delay
        signal arrived EARLIER than reference

Therefore:

    gcc_phat(
        signal=node_b_audio,
        reference=node_a_audio,
    )

returns:

    arrival_time_B - arrival_time_A

which is directly compatible with:

    TDOAMeasurement(
        node_a=A,
        node_b=B,
        ...
    )
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


from localization.gcc_phat import (
    GCCPHATResult,
    gcc_phat,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SAMPLE_RATE = (
    48_000.0
)


WINDOW_SAMPLES = (
    8192
)


DEFAULT_INTERPOLATION = (
    8
)


# ======================================================================
# TEST SIGNAL HELPERS
# ======================================================================


def make_broadband_reference(
    *,
    length: int = WINDOW_SAMPLES,
    seed: int = 20260830,
) -> np.ndarray:
    """
    Create deterministic broadband localization audio.

    Broadband noise is preferable to a pure sinusoid for GCC-PHAT unit
    tests because a sinusoid produces many periodic correlation peaks.

    The signal is tapered near both edges to reduce edge discontinuities.
    """

    rng = np.random.default_rng(
        seed
    )

    signal = rng.normal(
        loc=
            0.0,

        scale=
            1.0,

        size=
            length,
    )

    # --------------------------------------------------------------
    # EDGE TAPER
    # --------------------------------------------------------------

    window = np.hanning(
        length
    )

    signal = (
        signal
        * window
    )

    return np.ascontiguousarray(
        signal,
        dtype=np.float64,
    )


def shift_with_zeros(
    signal: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    """
    Shift a signal without circular wrapping.

    Positive delay
    --------------
    Output signal occurs later than input/reference.

        reference:
            [A B C D E ...]

        signal delayed +2:
            [0 0 A B C ...]

    Negative delay
    --------------
    Output occurs earlier than reference.

        delay = -2:

            [C D E ... 0 0]
    """

    if (
        signal.ndim
        != 1
    ):

        raise ValueError(
            "test helper requires mono 1-D signal"
        )

    output = np.zeros_like(
        signal
    )

    delay_samples = int(
        delay_samples
    )

    # ==============================================================
    # ZERO DELAY
    # ==============================================================

    if (
        delay_samples
        == 0
    ):

        output[
            :
        ] = (
            signal
        )

        return (
            output
        )

    # ==============================================================
    # POSITIVE DELAY
    # ==============================================================

    if (
        delay_samples
        > 0
    ):

        if (
            delay_samples
            >= signal.size
        ):

            return (
                output
            )

        output[
            delay_samples:
        ] = (
            signal[
                :-delay_samples
            ]
        )

        return (
            output
        )

    # ==============================================================
    # NEGATIVE DELAY
    # ==============================================================

    advance = (
        -delay_samples
    )

    if (
        advance
        >= signal.size
    ):

        return (
            output
        )

    output[
        :-advance
    ] = (
        signal[
            advance:
        ]
    )

    return (
        output
    )


def fractional_delay(
    signal: np.ndarray,
    delay_samples: float,
) -> np.ndarray:
    """
    Apply a synthetic fractional-sample delay using the Fourier
    shift theorem.

    This helper is used only for GCC-PHAT verification.

    Positive delay
    --------------
    The returned waveform arrives later than the reference.

    Implementation detail
    ---------------------
    The source signal is embedded inside a much larger zero-padded
    buffer before the phase shift is applied.

    This substantially reduces circular-wrap contamination compared
    with applying the fractional delay directly to an N-sample FFT.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if (
        signal.ndim
        != 1
    ):

        raise ValueError(
            "fractional_delay requires mono 1-D input"
        )

    if (
        signal.size
        == 0
    ):

        raise ValueError(
            "fractional_delay requires non-empty input"
        )

    if not np.all(
        np.isfinite(
            signal
        )
    ):

        raise ValueError(
            "fractional_delay requires finite input"
        )

    delay_samples = float(
        delay_samples
    )

    if not np.isfinite(
        delay_samples
    ):

        raise ValueError(
            "delay_samples must be finite"
        )

    # ==============================================================
    # LARGE ZERO-PADDED WORK BUFFER
    # ==============================================================

    input_length = int(
        signal.size
    )

    padded_length = int(
        input_length
        * 4
    )

    insertion_start = (
        input_length
    )

    padded = np.zeros(
        padded_length,
        dtype=np.float64,
    )

    padded[
        insertion_start:
        insertion_start
        + input_length
    ] = (
        signal
    )

    # ==============================================================
    # FOURIER SHIFT
    # ==============================================================

    spectrum = np.fft.rfft(
        padded
    )

    frequencies = np.fft.rfftfreq(
        padded_length,
        d=
            1.0,
    )

    phase = np.exp(
        (
            -2j
            * np.pi
            * frequencies
            * delay_samples
        )
    )

    shifted = np.fft.irfft(
        spectrum
        * phase,
        n=
            padded_length,
    )

    # ==============================================================
    # EXTRACT ORIGINAL-LENGTH REGION
    # ==============================================================

    result = shifted[
        insertion_start:
        insertion_start
        + input_length
    ]

    return np.ascontiguousarray(
        result,
        dtype=np.float64,
    )


# ======================================================================
# RESULT CONTRACT
# ======================================================================


def test_gcc_phat_returns_result_object() -> None:

    reference = (
        make_broadband_reference()
    )

    result = gcc_phat(
        signal=
            reference.copy(),

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,
    )

    assert isinstance(
        result,
        GCCPHATResult,
    )


# ======================================================================
# ZERO DELAY
# ======================================================================


def test_identical_signals_have_zero_delay() -> None:

    reference = (
        make_broadband_reference()
    )

    result = gcc_phat(
        signal=
            reference.copy(),

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.10,
    )

    assert (
        result.valid
    )

    assert (
        result.delay_samples
        == pytest.approx(
            0.0,
            abs=
                1.0
                / DEFAULT_INTERPOLATION,
        )
    )

    assert (
        result.delay_seconds
        == pytest.approx(
            0.0,
            abs=
                (
                    1.0
                    / DEFAULT_INTERPOLATION
                    / SAMPLE_RATE
                ),
        )
    )


# ======================================================================
# POSITIVE INTEGER DELAY
# ======================================================================


@pytest.mark.parametrize(
    "expected_delay_samples",
    [
        1,
        3,
        7,
        12,
        25,
    ],
)
def test_positive_integer_delay(
    expected_delay_samples: int,
) -> None:
    """
    Positive delay means SIGNAL is later than REFERENCE.
    """

    reference = (
        make_broadband_reference(
            seed=
                1000
                + expected_delay_samples
        )
    )

    signal = shift_with_zeros(
        reference,
        expected_delay_samples,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    assert (
        result.valid
    )

    assert (
        result.delay_samples
        == pytest.approx(
            expected_delay_samples,
            abs=
                0.5,
        )
    )


# ======================================================================
# NEGATIVE INTEGER DELAY
# ======================================================================


@pytest.mark.parametrize(
    "expected_delay_samples",
    [
        -1,
        -4,
        -9,
        -17,
        -28,
    ],
)
def test_negative_integer_delay(
    expected_delay_samples: int,
) -> None:
    """
    Negative delay means SIGNAL is earlier than REFERENCE.
    """

    reference = (
        make_broadband_reference(
            seed=
                2000
                + abs(
                    expected_delay_samples
                )
        )
    )

    signal = shift_with_zeros(
        reference,
        expected_delay_samples,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    assert (
        result.valid
    )

    assert (
        result.delay_samples
        == pytest.approx(
            expected_delay_samples,
            abs=
                0.5,
        )
    )


# ======================================================================
# FRACTIONAL / SUB-SAMPLE DELAY
# ======================================================================


def test_gcc_phat_recovers_known_fractional_delay() -> None:
    """
    Regression preserved from the original composite localization test.

    This test is specifically important because GCC-PHAT interpolation
    exists to estimate delays between integer sample positions.

    Ground truth:

        +7.25 samples

    Expected:

        approximately +7.25 samples

    Positive sign means the signal waveform arrives later than the
    reference waveform.
    """

    expected_delay_samples = (
        7.25
    )

    reference = (
        make_broadband_reference(
            length=
                4096,

            seed=
                42,
        )
    )

    signal = fractional_delay(
        reference,
        expected_delay_samples,
    )

    interpolation = (
        16
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        max_delay_seconds=
            0.001,

        interpolation=
            interpolation,

        min_peak_ratio=
            1.05,
    )

    assert (
        result.valid
    )

    assert (
        result.delay_samples
        == pytest.approx(
            expected_delay_samples,
            abs=
                0.20,
        )
    )

    assert (
        result.delay_seconds
        == pytest.approx(
            (
                expected_delay_samples
                / SAMPLE_RATE
            ),
            abs=
                (
                    0.20
                    / SAMPLE_RATE
                ),
        )
    )


# ======================================================================
# FRACTIONAL SIGN CONVENTION
# ======================================================================


def test_fractional_delay_sign_reverses_when_inputs_are_swapped() -> None:

    expected_delay_samples = (
        5.50
    )

    reference = (
        make_broadband_reference(
            length=
                4096,

            seed=
                43,
        )
    )

    delayed = fractional_delay(
        reference,
        expected_delay_samples,
    )

    forward = gcc_phat(
        signal=
            delayed,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        max_delay_seconds=
            0.001,

        interpolation=
            16,

        min_peak_ratio=
            1.05,
    )

    reverse = gcc_phat(
        signal=
            reference,

        reference=
            delayed,

        sample_rate=
            SAMPLE_RATE,

        max_delay_seconds=
            0.001,

        interpolation=
            16,

        min_peak_ratio=
            1.05,
    )

    assert (
        forward.valid
    )

    assert (
        reverse.valid
    )

    assert (
        forward.delay_samples
        == pytest.approx(
            expected_delay_samples,
            abs=
                0.20,
        )
    )

    assert (
        reverse.delay_samples
        == pytest.approx(
            -expected_delay_samples,
            abs=
                0.20,
        )
    )


# ======================================================================
# SIGN CONVENTION
# ======================================================================


def test_swapping_signal_and_reference_reverses_delay_sign() -> None:

    reference = (
        make_broadband_reference(
            seed=
                3001
        )
    )

    signal = shift_with_zeros(
        reference,
        14,
    )

    forward = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    reverse = gcc_phat(
        signal=
            reference,

        reference=
            signal,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    assert (
        forward.valid
    )

    assert (
        reverse.valid
    )

    assert (
        forward.delay_samples
        == pytest.approx(
            14.0,
            abs=
                0.5,
        )
    )

    assert (
        reverse.delay_samples
        == pytest.approx(
            -14.0,
            abs=
                0.5,
        )
    )

    assert (
        forward.delay_samples
        == pytest.approx(
            -reverse.delay_samples,
            abs=
                0.5,
        )
    )


# ======================================================================
# SAMPLE / SECOND CONSISTENCY
# ======================================================================


def test_delay_seconds_matches_delay_samples() -> None:

    reference = (
        make_broadband_reference(
            seed=
                4001
        )
    )

    signal = shift_with_zeros(
        reference,
        11,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    assert (
        result.delay_seconds
        == pytest.approx(
            result.delay_samples
            / SAMPLE_RATE,
            rel=
                1e-12,
            abs=
                1e-15,
        )
    )


# ======================================================================
# INTERPOLATION
# ======================================================================


@pytest.mark.parametrize(
    "interpolation",
    [
        1,
        2,
        4,
        8,
        16,
    ],
)
def test_integer_delay_is_stable_across_interpolation_factors(
    interpolation: int,
) -> None:

    expected_delay = (
        6
    )

    reference = (
        make_broadband_reference(
            seed=
                5001
        )
    )

    signal = shift_with_zeros(
        reference,
        expected_delay,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            interpolation,

        min_peak_ratio=
            1.05,
    )

    assert (
        result.valid
    )

    assert (
        result.delay_samples
        == pytest.approx(
            expected_delay,
            abs=
                max(
                    0.5,
                    1.0
                    / interpolation,
                ),
        )
    )


# ======================================================================
# PHYSICAL SEARCH LIMIT
# ======================================================================


def test_max_delay_limits_search_region() -> None:
    """
    Search must never return a delay outside the physical pair limit.

    This is important because GCC-PHAT can contain multipath or
    secondary correlation peaks outside the microphone-pair propagation
    interval.
    """

    reference = (
        make_broadband_reference(
            seed=
                6001
        )
    )

    signal = shift_with_zeros(
        reference,
        30,
    )

    maximum_samples = (
        10
    )

    maximum_seconds = (
        maximum_samples
        / SAMPLE_RATE
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        max_delay_seconds=
            maximum_seconds,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    assert (
        abs(
            result.delay_seconds
        )
        <= maximum_seconds
        + 1e-12
    )

    assert (
        abs(
            result.delay_samples
        )
        <= maximum_samples
        + (
            1.0
            / DEFAULT_INTERPOLATION
        )
    )


def test_delay_inside_physical_limit_is_recovered() -> None:

    reference = (
        make_broadband_reference(
            seed=
                6002
        )
    )

    expected_delay = (
        8
    )

    signal = shift_with_zeros(
        reference,
        expected_delay,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        max_delay_seconds=
            12.0
            / SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    assert (
        result.valid
    )

    assert (
        result.delay_samples
        == pytest.approx(
            expected_delay,
            abs=
                0.5,
        )
    )


# ======================================================================
# CORRELATION QUALITY
# ======================================================================


def test_clear_broadband_delay_has_good_peak_ratio() -> None:

    reference = (
        make_broadband_reference(
            seed=
                7001
        )
    )

    signal = shift_with_zeros(
        reference,
        5,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    assert (
        result.valid
    )

    assert (
        result.peak_ratio
        >= 1.05
    )


def test_peak_ratio_is_nonnegative() -> None:

    reference = (
        make_broadband_reference(
            seed=
                7002
        )
    )

    signal = shift_with_zeros(
        reference,
        -7,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.0,
    )

    assert (
        result.peak_ratio
        >= 0.0
    )


# ======================================================================
# SILENCE / ZERO ENERGY
# ======================================================================


def test_silence_is_not_valid_localization_measurement() -> None:
    """
    Zero-energy signals contain no arrival-time information.

    The estimator should return a structured invalid result rather than
    generating a false delay.
    """

    signal = np.zeros(
        WINDOW_SAMPLES,
        dtype=np.float64,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            signal.copy(),

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.10,
    )

    assert isinstance(
        result,
        GCCPHATResult,
    )

    assert not (
        result.valid
    )


# ======================================================================
# UNEQUAL LENGTHS
# ======================================================================


def test_rejects_unequal_signal_lengths() -> None:
    """
    LocalizationEngine supplies synchronized windows of equal length.

    GCC-PHAT should therefore reject mismatched windows instead of
    silently truncating or padding one side.
    """

    signal = np.zeros(
        4096,
        dtype=np.float64,
    )

    reference = np.zeros(
        4095,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                signal,

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,
        )


# ======================================================================
# DIMENSION VALIDATION
# ======================================================================


def test_rejects_multidimensional_signal() -> None:

    signal = np.zeros(
        (
            1024,
            2,
        ),
        dtype=np.float64,
    )

    reference = np.zeros(
        2048,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                signal,

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,
        )


def test_rejects_multidimensional_reference() -> None:

    signal = np.zeros(
        2048,
        dtype=np.float64,
    )

    reference = np.zeros(
        (
            1024,
            2,
        ),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                signal,

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,
        )


# ======================================================================
# NON-FINITE INPUT
# ======================================================================


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_non_finite_signal(
    bad_value: float,
) -> None:

    signal = np.zeros(
        2048,
        dtype=np.float64,
    )

    reference = np.zeros(
        2048,
        dtype=np.float64,
    )

    signal[
        100
    ] = (
        bad_value
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                signal,

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_non_finite_reference(
    bad_value: float,
) -> None:

    signal = np.zeros(
        2048,
        dtype=np.float64,
    )

    reference = np.zeros(
        2048,
        dtype=np.float64,
    )

    reference[
        100
    ] = (
        bad_value
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                signal,

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,
        )


# ======================================================================
# SAMPLE RATE VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "sample_rate",
    [
        0.0,
        -48_000.0,
        np.nan,
        np.inf,
    ],
)
def test_rejects_invalid_sample_rate(
    sample_rate: float,
) -> None:

    reference = (
        make_broadband_reference(
            length=
                2048
        )
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                reference.copy(),

            reference=
                reference,

            sample_rate=
                sample_rate,
        )


# ======================================================================
# INTERPOLATION VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "interpolation",
    [
        0,
        -1,
        -8,
    ],
)
def test_rejects_nonpositive_interpolation(
    interpolation: int,
) -> None:

    reference = (
        make_broadband_reference(
            length=
                2048
        )
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                reference.copy(),

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,

            interpolation=
                interpolation,
        )


# ======================================================================
# PEAK-RATIO THRESHOLD VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "minimum_peak_ratio",
    [
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_rejects_invalid_minimum_peak_ratio(
    minimum_peak_ratio: float,
) -> None:

    reference = (
        make_broadband_reference(
            length=
                2048
        )
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                reference.copy(),

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,

            min_peak_ratio=
                minimum_peak_ratio,
        )


# ======================================================================
# EPSILON VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "epsilon",
    [
        0.0,
        -1e-12,
        np.nan,
        np.inf,
    ],
)
def test_rejects_invalid_epsilon(
    epsilon: float,
) -> None:

    reference = (
        make_broadband_reference(
            length=
                2048
        )
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                reference.copy(),

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,

            epsilon=
                epsilon,
        )


# ======================================================================
# MAXIMUM DELAY VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "max_delay_seconds",
    [
        -0.001,
        np.nan,
        np.inf,
    ],
)
def test_rejects_invalid_maximum_delay(
    max_delay_seconds: float,
) -> None:

    reference = (
        make_broadband_reference(
            length=
                2048
        )
    )

    with pytest.raises(
        ValueError
    ):

        gcc_phat(
            signal=
                reference.copy(),

            reference=
                reference,

            sample_rate=
                SAMPLE_RATE,

            max_delay_seconds=
                max_delay_seconds,
        )


def test_zero_maximum_delay_allows_only_zero_lag() -> None:

    reference = (
        make_broadband_reference(
            length=
                2048,

            seed=
                8001
        )
    )

    signal = shift_with_zeros(
        reference,
        10,
    )

    result = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        max_delay_seconds=
            0.0,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.0,
    )

    assert (
        result.delay_samples
        == pytest.approx(
            0.0,
            abs=
                1e-12,
        )
    )

    assert (
        result.delay_seconds
        == pytest.approx(
            0.0,
            abs=
                1e-15,
        )
    )


# ======================================================================
# INPUT IMMUTABILITY
# ======================================================================


def test_gcc_phat_does_not_modify_input_arrays() -> None:

    reference = (
        make_broadband_reference(
            seed=
                9001
        )
    )

    signal = shift_with_zeros(
        reference,
        8,
    )

    original_reference = (
        reference.copy()
    )

    original_signal = (
        signal.copy()
    )

    gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    np.testing.assert_array_equal(
        reference,
        original_reference,
    )

    np.testing.assert_array_equal(
        signal,
        original_signal,
    )


# ======================================================================
# REPEATABILITY
# ======================================================================


def test_gcc_phat_is_deterministic() -> None:

    reference = (
        make_broadband_reference(
            seed=
                10001
        )
    )

    signal = shift_with_zeros(
        reference,
        -13,
    )

    first = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    second = gcc_phat(
        signal=
            signal,

        reference=
            reference,

        sample_rate=
            SAMPLE_RATE,

        interpolation=
            DEFAULT_INTERPOLATION,

        min_peak_ratio=
            1.05,
    )

    assert (
        first
        == second
    )