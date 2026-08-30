from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

from scipy.fft import (
    irfft,
    next_fast_len,
    rfft,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_EPSILON = (
    1e-12
)


MINIMUM_WINDOW_SAMPLES = (
    8
)


# ======================================================================
# RESULT MODEL
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class GCCPHATResult:
    """
    Result of one GCC-PHAT delay estimate.

    Sign convention
    ---------------
    Positive delay means:

        signal arrives later than reference.

    Negative delay means:

        signal arrives earlier than reference.
    """

    delay_seconds: float

    delay_samples: float

    peak_value: float

    peak_ratio: float

    max_delay_samples: float

    valid: bool

    reason: str = ""


# ======================================================================
# INVALID RESULT HELPER
# ======================================================================


def _invalid_result(
    reason: str,
    *,
    max_delay_samples: float = 0.0,
    peak_value: float = 0.0,
    peak_ratio: float = 0.0,
) -> GCCPHATResult:
    """
    Construct one standardized invalid GCC-PHAT result.
    """

    return GCCPHATResult(
        delay_seconds=
            0.0,

        delay_samples=
            0.0,

        peak_value=
            float(
                peak_value
            ),

        peak_ratio=
            float(
                peak_ratio
            ),

        max_delay_samples=
            float(
                max_delay_samples
            ),

        valid=
            False,

        reason=
            str(
                reason
            ),
    )


# ======================================================================
# SIGNAL PREPARATION
# ======================================================================


def _prepare(
    signal: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Prepare one mono waveform for GCC-PHAT.

    The signal is:

        converted to float64
        validated as one-dimensional
        checked for finite values
        DC-corrected
        made contiguous

    Multidimensional input is deliberately rejected rather than
    flattened.

    Each GCC-PHAT input must correspond to exactly one physical
    microphone waveform.
    """

    x = np.asarray(
        signal,
        dtype=np.float64,
    )

    # ==================================================================
    # DIMENSIONALITY
    # ==================================================================

    if (
        x.ndim
        != 1
    ):

        raise ValueError(
            (
                f"{name} must be mono 1-D audio, "
                f"got shape {x.shape}."
            )
        )

    # ==================================================================
    # EMPTY
    # ==================================================================

    if (
        x.size
        == 0
    ):

        return np.ascontiguousarray(
            x,
            dtype=np.float64,
        )

    # ==================================================================
    # FINITE VALUES
    # ==================================================================

    if not np.all(
        np.isfinite(
            x
        )
    ):

        raise ValueError(
            (
                f"{name} contains NaN "
                "or infinite samples."
            )
        )

    # ==================================================================
    # REMOVE DC
    # ==================================================================

    mean_value = float(
        np.mean(
            x,
            dtype=np.float64,
        )
    )

    x = (
        x
        - mean_value
    )

    return np.ascontiguousarray(
        x,
        dtype=np.float64,
    )


# ======================================================================
# GCC-PHAT
# ======================================================================


def gcc_phat(
    signal: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate: float,
    max_delay_seconds: float | None = None,
    interpolation: int = 8,
    epsilon: float = DEFAULT_EPSILON,
    min_peak_ratio: float = 1.0,
    beta: float = 1.0,
    frequency_band_hz: tuple[float, float] | None = None,
) -> GCCPHATResult:
    """
    Estimate the arrival-time difference between two microphone signals
    using Generalized Cross-Correlation with PHAT weighting.

    Parameters
    ----------
    signal
        Target microphone waveform.

    reference
        Reference microphone waveform.

    sample_rate
        Sampling frequency in Hz.

    max_delay_seconds
        Optional physically possible absolute delay limit.

        When supplied, correlation peaks outside:

            ± max_delay_seconds

        are ignored.

    interpolation
        Correlation interpolation factor.

        Example:

            interpolation = 8

        provides delay estimates on a 1/8-sample grid.

    epsilon
        Numerical floor used during PHAT normalization.

    min_peak_ratio
        Minimum ratio between the selected correlation peak and the best
        competing peak outside approximately ±1 native sample.

    Returns
    -------
    GCCPHATResult

    Sign convention
    ---------------
    Positive:

        signal arrives later than reference.

    Negative:

        signal arrives earlier than reference.


    Notes
    -----
    GCC-PHAT estimates residual acoustic delay from waveform content.

    It does not replace the shared-clock/sampleIndex synchronization
    layer.
    """

    # ==================================================================
    # SAMPLE RATE
    # ==================================================================

    try:

        sample_rate = float(
            sample_rate
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            (
                "sample_rate must be "
                "a numeric value."
            )
        ) from exc

    if (
        not math.isfinite(
            sample_rate
        )
        or sample_rate
        <= 0.0
    ):

        raise ValueError(
            (
                "sample_rate must be finite "
                "and greater than 0."
            )
        )

    # ==================================================================
    # INTERPOLATION
    # ==================================================================

    if (
        isinstance(
            interpolation,
            bool,
        )
        or not isinstance(
            interpolation,
            int,
        )
    ):

        raise TypeError(
            (
                "interpolation must be "
                "an integer."
            )
        )

    if (
        interpolation
        <= 0
    ):

        raise ValueError(
            (
                "interpolation must be "
                "greater than 0."
            )
        )

    # ==================================================================
    # EPSILON
    # ==================================================================

    try:

        epsilon = float(
            epsilon
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            (
                "epsilon must be "
                "a numeric value."
            )
        ) from exc

    if (
        not math.isfinite(
            epsilon
        )
        or epsilon
        <= 0.0
    ):

        raise ValueError(
            (
                "epsilon must be finite "
                "and greater than 0."
            )
        )

    # ==================================================================
    # PEAK-RATIO THRESHOLD
    # ==================================================================

    try:

        min_peak_ratio = float(
            min_peak_ratio
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            (
                "min_peak_ratio must be "
                "a numeric value."
            )
        ) from exc

    if (
        not math.isfinite(
            min_peak_ratio
        )
        or min_peak_ratio
        < 1.0
    ):

        raise ValueError(
            (
                "min_peak_ratio must be finite "
                "and at least 1.0."
            )
        )

    # ==================================================================
    # FRACTIONAL PHAT WEIGHTING (BETA)
    # ==================================================================

    try:

        beta = float(
            beta
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            (
                "beta must be a numeric value."
            )
        ) from exc

    if (
        not math.isfinite(
            beta
        )
        or not (
            0.0
            <= beta
            <= 1.0
        )
    ):

        raise ValueError(
            (
                "beta must be finite and lie between 0 and 1."
            )
        )

    # ==================================================================
    # FREQUENCY BAND-LIMITING
    # ==================================================================

    if (
        frequency_band_hz
        is not None
    ):

        if not isinstance(
            frequency_band_hz,
            (tuple, list),
        ) or len(frequency_band_hz) != 2:

            raise TypeError(
                (
                    "frequency_band_hz must be a tuple of (low_hz, high_hz) or None."
                )
            )

        try:

            low_hz = float(
                frequency_band_hz[0]
            )

            high_hz = float(
                frequency_band_hz[1]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                (
                    "frequency_band_hz elements must be numeric."
                )
            ) from exc

        if not (
            math.isfinite(low_hz)
            and math.isfinite(high_hz)
        ):

            raise ValueError(
                (
                    "frequency_band_hz elements must be finite."
                )
            )

        if low_hz < 0.0:

            raise ValueError(
                (
                    "frequency_band_hz lower bound must be non-negative."
                )
            )

        if high_hz <= low_hz:

            raise ValueError(
                (
                    "frequency_band_hz upper bound must be strictly greater than lower bound."
                )
            )

        nyquist = (
            sample_rate
            / 2.0
        )

        if high_hz > nyquist:

            raise ValueError(
                (
                    f"frequency_band_hz upper bound ({high_hz} Hz) exceeds Nyquist ({nyquist} Hz)."
                )
            )

    # ==================================================================
    # PHYSICAL DELAY LIMIT
    # ==================================================================

    if (
        max_delay_seconds
        is not None
    ):

        try:

            max_delay_seconds = float(
                max_delay_seconds
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                (
                    "max_delay_seconds must be "
                    "numeric or None."
                )
            ) from exc

        if (
            not math.isfinite(
                max_delay_seconds
            )
            or max_delay_seconds
            < 0.0
        ):

            raise ValueError(
                (
                    "max_delay_seconds must be "
                    "finite and non-negative."
                )
            )

    # ==================================================================
    # PREPARE WAVEFORMS
    # ==================================================================

    x = (
        _prepare(
            signal,
            name=
                "signal",
        )
    )

    y = (
        _prepare(
            reference,
            name=
                "reference",
        )
    )

    # ==================================================================
    # WINDOW LENGTH
    # ==================================================================

    if (
        x.size
        < MINIMUM_WINDOW_SAMPLES
        or y.size
        < MINIMUM_WINDOW_SAMPLES
    ):

        return _invalid_result(
            "window too short"
        )

    # --------------------------------------------------------------
    # Our localization system extracts equal absolute sample windows
    # from each node.
    #
    # Unequal lengths therefore indicate an upstream timeline/window
    # construction error rather than a condition GCC-PHAT should hide.
    # --------------------------------------------------------------

    if (
        x.size
        != y.size
    ):

        raise ValueError(
            (
                "signal/reference window "
                "length mismatch"
            )
        )

    # ==================================================================
    # ENERGY CHECK
    # ==================================================================

    x_energy = float(
        np.dot(
            x,
            x,
        )
    )

    y_energy = float(
        np.dot(
            y,
            y,
        )
    )

    if (
        not math.isfinite(
            x_energy
        )
        or not math.isfinite(
            y_energy
        )
        or x_energy
        <= epsilon
        or y_energy
        <= epsilon
    ):

        return _invalid_result(
            "zero-energy window"
        )

    # ==================================================================
    # LINEAR-CORRELATION FFT SIZE
    # ==================================================================

    n_linear = (
        int(
            x.size
        )
        + int(
            y.size
        )
        - 1
    )

    n_fft = int(
        next_fast_len(
            n_linear
        )
    )

    # ==================================================================
    # FFT
    # ==================================================================

    spectrum_x = (
        rfft(
            x,
            n=
                n_fft,
        )
    )

    spectrum_y = (
        rfft(
            y,
            n=
                n_fft,
        )
    )

    # ==================================================================
    # CROSS POWER SPECTRUM
    # ==================================================================

    cross_spectrum = (
        spectrum_x
        * np.conj(
            spectrum_y
        )
    )

    magnitude = np.abs(
        cross_spectrum
    )

    # ==================================================================
    # PHAT WEIGHTING
    # ==================================================================
    #
    # Bins without meaningful cross-spectrum magnitude are explicitly
    # kept at zero rather than dividing tiny values by epsilon and
    # treating numerical noise as informative phase.
    # ==================================================================

    informative_bins = (
        magnitude
        > epsilon
    )

    if not np.any(
        informative_bins
    ):

        return _invalid_result(
            (
                "no informative "
                "cross-spectrum bins"
            )
        )

    phat = np.zeros_like(
        cross_spectrum
    )

    if beta == 1.0:

        phat[
            informative_bins
        ] = (
            cross_spectrum[
                informative_bins
            ]
            / magnitude[
                informative_bins
            ]
        )

    elif beta == 0.0:

        phat[
            informative_bins
        ] = (
            cross_spectrum[
                informative_bins
            ]
        )

    else:

        phat[
            informative_bins
        ] = (
            cross_spectrum[
                informative_bins
            ]
            / (
                magnitude[
                    informative_bins
                ]
                ** beta
            )
        )

    # ==================================================================
    # BAND LIMITING
    # ==================================================================

    if (
        frequency_band_hz
        is not None
    ):

        freq_grid = np.linspace(
            0.0,
            sample_rate / 2.0,
            len(cross_spectrum),
        )

        band_mask = (
            (freq_grid >= frequency_band_hz[0])
            & (freq_grid <= frequency_band_hz[1])
        )

        phat = (
            phat
            * band_mask
        )

    # ==================================================================
    # INTERPOLATED CORRELATION
    # ==================================================================

    correlation_length = (
        n_fft
        * interpolation
    )

    correlation_circular = (
        irfft(
            phat,
            n=
                correlation_length,
        )
    )

    if not np.all(
        np.isfinite(
            correlation_circular
        )
    ):

        return _invalid_result(
            (
                "non-finite GCC "
                "correlation"
            )
        )

    # ==================================================================
    # CENTER ZERO LAG
    # ==================================================================
    #
    # For equal-length localization windows, the meaningful linear lag
    # range is:
    #
    #     -(N - 1) ... +(N - 1)
    #
    # represented on the interpolated grid.
    # ==================================================================

    max_linear_lag_samples = (
        int(
            x.size
        )
        - 1
    )

    max_linear_lag_i = (
        max_linear_lag_samples
        * interpolation
    )

    correlation = np.concatenate(
        (
            correlation_circular[
                -max_linear_lag_i:
            ],

            correlation_circular[
                :
                max_linear_lag_i
                + 1
            ],
        )
    )

    interpolated_lags = np.arange(
        -max_linear_lag_i,
        max_linear_lag_i
        + 1,
        dtype=np.int64,
    )

    # ==================================================================
    # PHYSICALLY POSSIBLE DELAY
    # ==================================================================

    if (
        max_delay_seconds
        is None
    ):

        max_delay_samples = float(
            max_linear_lag_samples
        )

    else:

        requested_max_delay_samples = (
            max_delay_seconds
            * sample_rate
        )

        max_delay_samples = min(
            float(
                max_linear_lag_samples
            ),
            float(
                requested_max_delay_samples
            ),
        )

    max_delay_i = int(
        math.floor(
            max_delay_samples
            * interpolation
        )
    )

    feasible = (
        np.abs(
            interpolated_lags
        )
        <= max_delay_i
    )

    if not np.any(
        feasible
    ):

        return _invalid_result(
            "empty physical lag window",
            max_delay_samples=
                max_delay_samples,
        )

    # ==================================================================
    # PEAK SEARCH
    # ==================================================================

    feasible_correlation = np.abs(
        correlation[
            feasible
        ]
    )

    feasible_lags = (
        interpolated_lags[
            feasible
        ]
    )

    if (
        feasible_correlation.size
        == 0
    ):

        return _invalid_result(
            "empty GCC correlation window",
            max_delay_samples=
                max_delay_samples,
        )

    peak_local_index = int(
        np.argmax(
            feasible_correlation
        )
    )

    peak_value = float(
        feasible_correlation[
            peak_local_index
        ]
    )

    winning_lag_i = int(
        feasible_lags[
            peak_local_index
        ]
    )

    if (
        not math.isfinite(
            peak_value
        )
        or peak_value
        <= epsilon
    ):

        return _invalid_result(
            "no meaningful GCC peak",
            max_delay_samples=
                max_delay_samples,
            peak_value=
                peak_value,
        )

    # ==================================================================
    # PEAK UNIQUENESS
    # ==================================================================
    #
    # Exclude approximately ±1 native sample around the winning
    # interpolated peak.
    #
    # Comparing only immediately adjacent oversampled points would make
    # one broad physical correlation peak look artificially ambiguous.
    # ==================================================================

    exclusion_radius = (
        interpolation
    )

    competing_mask = np.ones(
        feasible_correlation.size,
        dtype=bool,
    )

    exclusion_start = max(
        0,
        peak_local_index
        - exclusion_radius,
    )

    exclusion_end = min(
        feasible_correlation.size,
        peak_local_index
        + exclusion_radius
        + 1,
    )

    competing_mask[
        exclusion_start:
        exclusion_end
    ] = (
        False
    )

    if np.any(
        competing_mask
    ):

        second_peak = float(
            np.max(
                feasible_correlation[
                    competing_mask
                ]
            )
        )

    else:

        second_peak = (
            0.0
        )

    peak_ratio = (
        peak_value
        / max(
            second_peak,
            epsilon,
        )
    )

    if not math.isfinite(
        peak_ratio
    ):

        peak_ratio = float(
            "inf"
        )

    # ==================================================================
    # DELAY
    # ==================================================================

    delay_samples = (
        winning_lag_i
        / interpolation
    )

    delay_seconds = (
        delay_samples
        / sample_rate
    )

    # ==================================================================
    # QUALITY DECISION
    # ==================================================================

    valid = bool(
        peak_ratio
        >= min_peak_ratio
    )

    if valid:

        reason = (
            ""
        )

    else:

        reason = (
            "ambiguous GCC peak ratio "
            f"{peak_ratio:.3f}"
        )

    # ==================================================================
    # RESULT
    # ==================================================================

    return GCCPHATResult(
        delay_seconds=
            float(
                delay_seconds
            ),

        delay_samples=
            float(
                delay_samples
            ),

        peak_value=
            float(
                peak_value
            ),

        peak_ratio=
            float(
                peak_ratio
            ),

        max_delay_samples=
            float(
                max_delay_samples
            ),

        valid=
            valid,

        reason=
            reason,
    )