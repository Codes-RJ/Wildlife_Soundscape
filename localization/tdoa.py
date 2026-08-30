from __future__ import annotations

from dataclasses import dataclass

import math


# ======================================================================
# CONSTANTS
# ======================================================================


UINT8_MAX = (
    0xFF
)


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _validate_node_id(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate one Protocol-v4 acoustic node identifier.
    """

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):

        raise TypeError(
            f"{name} must be an integer."
        )

    value = int(
        value
    )

    if not (
        1
        <= value
        <= UINT8_MAX
    ):

        raise ValueError(
            (
                f"{name} must lie between "
                "1 and 255."
            )
        )

    return value


def _validate_finite_float(
    value: float,
    *,
    name: str,
) -> float:
    """
    Convert one numeric value to a finite float.
    """

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            (
                f"{name} must be "
                "a numeric value."
            )
        ) from exc

    if not math.isfinite(
        value
    ):

        raise ValueError(
            (
                f"{name} must be "
                "finite."
            )
        )

    return value


def _validate_point(
    point: tuple[
        float,
        float,
    ],
    *,
    name: str,
) -> tuple[
    float,
    float,
]:
    """
    Validate one two-dimensional microphone position.
    """

    if not isinstance(
        point,
        (
            tuple,
            list,
        ),
    ):

        raise TypeError(
            (
                f"{name} must be a "
                "2-D coordinate."
            )
        )

    if (
        len(
            point
        )
        != 2
    ):

        raise ValueError(
            (
                f"{name} must contain "
                "exactly two coordinates."
            )
        )

    x = (
        _validate_finite_float(
            point[0],
            name=
                f"{name}[0]",
        )
    )

    y = (
        _validate_finite_float(
            point[1],
            name=
                f"{name}[1]",
        )
    )

    return (
        x,
        y,
    )


# ======================================================================
# TDOA MEASUREMENT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class TDOAMeasurement:
    """
    One pairwise Time Difference of Arrival measurement.

    Sign convention
    ---------------
    delay_seconds is:

        arrival at node_b
        minus
        arrival at node_a

    Therefore:

        positive delay
            node_b received the acoustic wave later than node_a

        negative delay
            node_b received the acoustic wave earlier than node_a
    """

    node_a: int

    node_b: int

    delay_seconds: float

    delay_samples: float

    peak_ratio: float

    max_delay_seconds: float

    valid: bool

    reason: str = ""

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate one immutable TDOA measurement.
        """

        # ==============================================================
        # NODE IDS
        # ==============================================================

        node_a = (
            _validate_node_id(
                self.node_a,
                name=
                    "node_a",
            )
        )

        node_b = (
            _validate_node_id(
                self.node_b,
                name=
                    "node_b",
            )
        )

        if (
            node_a
            == node_b
        ):

            raise ValueError(
                (
                    "TDOA node_a and node_b "
                    "must refer to different nodes."
                )
            )

        # ==============================================================
        # NUMERIC VALUES
        # ==============================================================

        delay_seconds = (
            _validate_finite_float(
                self.delay_seconds,
                name=
                    "delay_seconds",
            )
        )

        delay_samples = (
            _validate_finite_float(
                self.delay_samples,
                name=
                    "delay_samples",
            )
        )

        peak_ratio = (
            _validate_finite_float(
                self.peak_ratio,
                name=
                    "peak_ratio",
            )
        )

        max_delay_seconds = (
            _validate_finite_float(
                self.max_delay_seconds,
                name=
                    "max_delay_seconds",
            )
        )

        if (
            peak_ratio
            < 0.0
        ):

            raise ValueError(
                (
                    "peak_ratio cannot "
                    "be negative."
                )
            )

        if (
            max_delay_seconds
            < 0.0
        ):

            raise ValueError(
                (
                    "max_delay_seconds cannot "
                    "be negative."
                )
            )

        # ==============================================================
        # PHYSICAL DELAY CONSISTENCY
        # ==============================================================
        #
        # A tiny numerical allowance prevents floating-point rounding
        # around the physical boundary from causing unnecessary failure.
        # ==============================================================

        numerical_tolerance = max(
            1e-12,
            max_delay_seconds
            * 1e-9,
        )

        if (
            self.valid
            and abs(
                delay_seconds
            )
            > (
                max_delay_seconds
                + numerical_tolerance
            )
        ):

            raise ValueError(
                (
                    "Valid TDOA delay exceeds "
                    "the configured physical "
                    "maximum delay."
                )
            )

        # ==============================================================
        # NORMALIZE STORED VALUES
        # ==============================================================

        object.__setattr__(
            self,
            "node_a",
            node_a,
        )

        object.__setattr__(
            self,
            "node_b",
            node_b,
        )

        object.__setattr__(
            self,
            "delay_seconds",
            delay_seconds,
        )

        object.__setattr__(
            self,
            "delay_samples",
            delay_samples,
        )

        object.__setattr__(
            self,
            "peak_ratio",
            peak_ratio,
        )

        object.__setattr__(
            self,
            "max_delay_seconds",
            max_delay_seconds,
        )

        object.__setattr__(
            self,
            "valid",
            bool(
                self.valid
            ),
        )

        object.__setattr__(
            self,
            "reason",
            str(
                self.reason
            ),
        )


# ======================================================================
# MICROPHONE-PAIR DISTANCE
# ======================================================================


def pair_distance(
    a: tuple[
        float,
        float,
    ],
    b: tuple[
        float,
        float,
    ],
) -> float:
    """
    Calculate Euclidean distance between two microphone positions.

    Parameters
    ----------
    a
        First microphone coordinate:

            (x, y)

    b
        Second microphone coordinate:

            (x, y)

    Returns
    -------
    float
        Pair separation in meters.
    """

    a_x, a_y = (
        _validate_point(
            a,
            name=
                "a",
        )
    )

    b_x, b_y = (
        _validate_point(
            b,
            name=
                "b",
        )
    )

    distance = math.hypot(
        a_x
        - b_x,

        a_y
        - b_y,
    )

    if not math.isfinite(
        distance
    ):

        raise ValueError(
            (
                "Microphone-pair distance "
                "is not finite."
            )
        )

    return float(
        distance
    )


# ======================================================================
# PHYSICAL MAXIMUM TDOA
# ======================================================================


def physical_max_delay(
    a: tuple[
        float,
        float,
    ],
    b: tuple[
        float,
        float,
    ],
    speed_of_sound_mps: float,
) -> float:
    """
    Calculate the maximum physically possible absolute TDOA between two
    microphones.

    Formula
    -------
        tau_max = d_ab / c

    where:

        d_ab
            Euclidean microphone separation in meters.

        c
            Speed of sound in meters per second.

    Interpretation
    --------------
    No source in the far or near field can produce a pairwise arrival
    time difference larger in magnitude than the microphone separation
    divided by propagation speed.

    This bound is therefore used to restrict GCC-PHAT peak search.
    """

    speed_of_sound_mps = (
        _validate_finite_float(
            speed_of_sound_mps,
            name=
                "speed_of_sound_mps",
        )
    )

    if (
        speed_of_sound_mps
        <= 0.0
    ):

        raise ValueError(
            (
                "speed_of_sound_mps must "
                "be greater than 0."
            )
        )

    distance_m = (
        pair_distance(
            a,
            b,
        )
    )

    maximum_delay = (
        distance_m
        / speed_of_sound_mps
    )

    if not math.isfinite(
        maximum_delay
    ):

        raise ValueError(
            (
                "Calculated physical maximum "
                "delay is not finite."
            )
        )

    return float(
        maximum_delay
    )