from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math


# ======================================================================
# PROTOCOL LIMITS
# ======================================================================
#
# These values mirror fields frozen in Protocol v4.
#
# Keeping protocol limits in configuration validation prevents a valid
# Python configuration from producing values that cannot be represented
# by the ESP32 wire protocol.
# ======================================================================


UINT8_MAX = 0xFF


UINT16_MAX = 0xFFFF


UINT32_MAX = 0xFFFFFFFF


INT16_MAX = 0x7FFF


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _require_finite(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require a finite numeric configuration value.
    """

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(f"{name} must be numeric.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def _require_positive_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a positive non-boolean integer.
    """

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(f"{name} must be an integer.")

    if not isinstance(
        value,
        int,
    ):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")

    return value


# ======================================================================
# NETWORK CONFIGURATION
# ======================================================================
