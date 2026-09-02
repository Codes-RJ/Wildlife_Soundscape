from __future__ import annotations

import math


def calculate_speed_of_sound_mps(
    temperature_c: float,
    humidity_percent: float = 50.0,
    pressure_hpa: float = 1013.25,
) -> float:
    """Engineering approximation for sound speed in humid air.

    This is deliberately *not* labelled as the full Cramer equation. It uses
    temperature as the dominant term and a vapor-pressure correction for
    humidity. Pressure is used in the humidity correction. It is sufficiently
    accurate for this lab prototype and can later be replaced by a rigorously
    validated Cramer implementation without changing the localization API.
    """
    if not math.isfinite(temperature_c):
        raise ValueError("temperature_c must be finite")
    if not math.isfinite(humidity_percent):
        raise ValueError("humidity_percent must be finite")
    if not math.isfinite(pressure_hpa) or pressure_hpa <= 0:
        raise ValueError("pressure_hpa must be finite and positive")

    rh = min(100.0, max(0.0, float(humidity_percent)))
    t = float(temperature_c)
    p = float(pressure_hpa)

    # Buck saturation-vapor-pressure approximation, hPa.
    p_sat = 6.1121 * math.exp((18.678 - t / 234.5) * (t / (257.14 + t)))
    vapor_pressure = (rh / 100.0) * p_sat

    # Dry-air temperature term with a modest humid-air correction.
    c_dry = 331.3 * math.sqrt(1.0 + t / 273.15)
    c = c_dry * (1.0 + 0.16 * (vapor_pressure / p))

    # Sanity bound for ordinary terrestrial laboratory conditions.
    return float(min(380.0, max(300.0, c)))
