"""
Environmental association analytics.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module evaluates statistical associations between environmental
conditions and acoustic-activity measurements.

Typical questions
-----------------
Examples include:

    Does detected acoustic activity increase with temperature?

    Is activity duration associated with humidity?

    Does atmospheric pressure show a monotonic association with
    event rate?

Default environmental variables
-------------------------------
    temperature_c
    humidity_percent
    pressure_hpa

Typical response variables
--------------------------
    event_count
    active_duration_s
    event_rate_per_hour

Statistical method
------------------
The default method is Spearman rank correlation.

Spearman correlation measures monotonic association and does not require
the variables themselves to follow a normal distribution.

Scientific interpretation
-------------------------
These results represent ASSOCIATIONS.

They must not be presented as evidence that:

    temperature causes wildlife activity
    humidity causes vocalization
    pressure changes cause animal movement

Potential confounders include:

    time of day
    season
    species composition
    microphone sensitivity
    background noise
    habitat
    rainfall
    wind
    human activity
    detector performance

Observation-unit requirement
----------------------------
For activity-vs-environment analysis, rows should represent regular time
bins such as:

    5 minutes
    15 minutes
    30 minutes
    1 hour

and MUST ideally include bins where:

    event_count == 0

Using only detected-event rows creates event-conditioned sampling and
can bias activity-environment relationships.

Example normalized row
----------------------
{
    "temperature_c": 26.4,
    "humidity_percent": 72.0,
    "pressure_hpa": 1007.8,
    "event_count": 4,
    "active_duration_s": 8.2,
    "event_rate_per_hour": 8.0,
}

Missing/non-finite pairs are excluded pairwise.

No database access is performed in this module.
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)

from typing import (
    Any,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np

from scipy.stats import (
    spearmanr,
)


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from .models import (
    AssociationDirection,
    AssociationStrength,
    EnvironmentalAssociation,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_ALPHA = 0.05


DEFAULT_MIN_SAMPLES = 5


DEFAULT_NEUTRAL_THRESHOLD = 0.05


DEFAULT_ENVIRONMENTAL_FIELDS = {
    "temperature_c": "temperature_c",
    "humidity_percent": "humidity_percent",
    "pressure_hpa": "pressure_hpa",
}


DEFAULT_RESPONSE_FIELDS = {
    "event_count": "event_count",
    "active_duration_s": "active_duration_s",
    "event_rate_per_hour": "event_rate_per_hour",
}


# ----------------------------------------------------------------------
# Descriptive effect-size thresholds.
#
# These are intentionally treated as project-level descriptive
# categories rather than biological laws.
#
# |rho| < 0.10     negligible
#       < 0.30     weak
#       < 0.50     moderate
#       < 0.70     strong
#       >= 0.70    very strong
# ----------------------------------------------------------------------


NEGLIGIBLE_LIMIT = 0.10


WEAK_LIMIT = 0.30


MODERATE_LIMIT = 0.50


STRONG_LIMIT = 0.70


# ======================================================================
# ROW ACCESS
# ======================================================================


def _row_value(
    row: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a value from dictionary-like or sqlite3.Row-like input.
    """

    # ==============================================================
    # DICTIONARY / MAPPING-LIKE GET
    # ==============================================================

    getter = getattr(
        row,
        "get",
        None,
    )

    if callable(getter):
        try:
            return getter(
                key,
                default,
            )

        except (
            KeyError,
            TypeError,
        ):
            pass

    # ==============================================================
    # SQLITE ROW / INDEXED MAPPING
    # ==============================================================

    try:
        return row[key]

    except (
        KeyError,
        IndexError,
        TypeError,
    ):
        return default


# ======================================================================
# ALPHA VALIDATION
# ======================================================================


def _validate_alpha(
    alpha: float,
) -> float:
    """
    Validate statistical significance threshold.
    """

    try:
        alpha = float(alpha)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError("alpha must be numeric.") from exc

    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be finite and in (0, 1).")

    return alpha


# ======================================================================
# MINIMUM SAMPLE COUNT
# ======================================================================


def _validate_min_samples(
    min_samples: int,
) -> int:
    """
    Require at least three paired observations.

    The project defaults to five as an intentionally modest minimum for
    exploratory analysis.

    Larger datasets are preferable for research conclusions.
    """

    if isinstance(
        min_samples,
        bool,
    ) or not isinstance(
        min_samples,
        int,
    ):
        raise TypeError("min_samples must be an integer.")

    if min_samples < 3:
        raise ValueError("min_samples must be at least 3.")

    return int(min_samples)


# ======================================================================
# NEUTRAL THRESHOLD
# ======================================================================


def _validate_neutral_threshold(
    value: float,
) -> float:
    """
    Validate coefficient threshold treated as effectively neutral.
    """

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(("neutral_threshold must be numeric.")) from exc

    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(("neutral_threshold must be finite and in [0, 1]."))

    return value


# ======================================================================
# FIELD-MAPPING VALIDATION
# ======================================================================


def _validate_field_mapping(
    fields: Mapping[
        str,
        str,
    ],
    *,
    name: str,
) -> dict[
    str,
    str,
]:
    """
    Validate logical-variable -> row-field mappings.

    Example
    -------
    {
        "temperature_c": "temperature_c",
        "humidity_percent": "humidity_percent",
    }
    """

    if not isinstance(
        fields,
        Mapping,
    ):
        raise TypeError(f"{name} must be a mapping.")

    if not (fields):
        raise ValueError(f"{name} cannot be empty.")

    result: dict[
        str,
        str,
    ] = {}

    for logical_name, row_field in fields.items():
        if not isinstance(
            logical_name,
            str,
        ):
            raise TypeError((f"{name} logical names must be strings."))

        if not isinstance(
            row_field,
            str,
        ):
            raise TypeError((f"{name} row-field names must be strings."))

        logical_name = logical_name.strip()

        row_field = row_field.strip()

        if not (logical_name):
            raise ValueError((f"{name} contains an empty logical name."))

        if not (row_field):
            raise ValueError((f"{name} contains an empty row-field name."))

        result[logical_name] = row_field

    return result


# ======================================================================
# NUMERIC NORMALIZATION
# ======================================================================


def _finite_float_or_none(
    value: Any,
) -> float | None:
    """
    Convert a possible value into a finite float.

    Missing, malformed, NaN and infinite values become None.
    """

    if value is None:
        return None

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(result):
        return None

    return result


# ======================================================================
# PAIRED FINITE OBSERVATIONS
# ======================================================================


def finite_pairs(
    x_values: Sequence[Any] | np.ndarray,
    y_values: Sequence[Any] | np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Return pairwise finite observations.

    A pair is retained only when BOTH x and y are finite numeric values.

    Returns
    -------
    x
        Contiguous float64 vector.

    y
        Contiguous float64 vector.
    """

    try:
        x_length = len(x_values)

        y_length = len(y_values)

    except TypeError as exc:
        raise TypeError(
            ("x_values and y_values must be sized one-dimensional sequences.")
        ) from exc

    if x_length != y_length:
        raise ValueError(("x_values and y_values must have equal length."))

    x_clean: list[float] = []

    y_clean: list[float] = []

    for x_value, y_value in zip(
        x_values,
        y_values,
        strict=False,
    ):
        x_numeric = _finite_float_or_none(x_value)

        y_numeric = _finite_float_or_none(y_value)

        if x_numeric is None or y_numeric is None:
            continue

        x_clean.append(x_numeric)

        y_clean.append(y_numeric)

    return (
        np.ascontiguousarray(
            x_clean,
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            y_clean,
            dtype=np.float64,
        ),
    )


# ======================================================================
# ASSOCIATION DIRECTION
# ======================================================================


def classify_association_direction(
    coefficient: float | None,
    *,
    neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,
) -> AssociationDirection:
    """
    Convert a correlation coefficient into a descriptive direction.

    Values very close to zero are treated as neutral to avoid assigning
    strong semantic meaning to trivial numerical signs.
    """

    neutral_threshold = _validate_neutral_threshold(neutral_threshold)

    if coefficient is None:
        return AssociationDirection.NEUTRAL

    try:
        coefficient = float(coefficient)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError("coefficient must be numeric or None.") from exc

    if not math.isfinite(coefficient):
        raise ValueError("coefficient must be finite.")

    if not (-1.0 <= coefficient <= 1.0):
        raise ValueError(("coefficient must be in [-1, 1]."))

    if abs(coefficient) <= neutral_threshold:
        return AssociationDirection.NEUTRAL

    if coefficient > 0.0:
        return AssociationDirection.POSITIVE

    return AssociationDirection.NEGATIVE


# ======================================================================
# ASSOCIATION STRENGTH
# ======================================================================


def classify_association_strength(
    coefficient: float | None,
) -> AssociationStrength:
    """
    Convert |Spearman rho| into an internal descriptive strength class.

    These thresholds are descriptive conventions for this project, not
    universal biological significance thresholds.
    """

    if coefficient is None:
        return AssociationStrength.NEGLIGIBLE

    try:
        coefficient = float(coefficient)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError("coefficient must be numeric or None.") from exc

    if not math.isfinite(coefficient):
        raise ValueError("coefficient must be finite.")

    if not (-1.0 <= coefficient <= 1.0):
        raise ValueError(("coefficient must be in [-1, 1]."))

    magnitude = abs(coefficient)

    if magnitude < NEGLIGIBLE_LIMIT:
        return AssociationStrength.NEGLIGIBLE

    if magnitude < WEAK_LIMIT:
        return AssociationStrength.WEAK

    if magnitude < MODERATE_LIMIT:
        return AssociationStrength.MODERATE

    if magnitude < STRONG_LIMIT:
        return AssociationStrength.STRONG

    return AssociationStrength.VERY_STRONG


# ======================================================================
# CONSTANT-SERIES DETECTION
# ======================================================================


def _is_constant(
    values: np.ndarray,
) -> bool:
    """
    Return True when all values in a vector are identical.

    Spearman correlation is undefined when either variable is constant.
    """

    if values.size == 0:
        return True

    return bool(np.all(values == values[0]))


# ======================================================================
# SPEARMAN ASSOCIATION
# ======================================================================


def calculate_spearman_association(
    x_values: Sequence[Any] | np.ndarray,
    y_values: Sequence[Any] | np.ndarray,
    *,
    environmental_variable: str,
    response_variable: str,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,
) -> EnvironmentalAssociation:
    """
    Calculate one pairwise Spearman association.

    Invalid or missing observations are removed pairwise.

    Insufficient / undefined data
    -----------------------------
    When:

        paired sample count < min_samples

    or:

        either variable is constant

    the result is represented as:

        coefficient = None
        p_value = None
        direction = NEUTRAL
        strength = NEGLIGIBLE
        statistically_significant = False

    This is preferable to inventing a zero coefficient for an undefined
    correlation.
    """

    # ==============================================================
    # NAMES
    # ==============================================================

    if not isinstance(
        environmental_variable,
        str,
    ):
        raise TypeError(("environmental_variable must be a string."))

    if not isinstance(
        response_variable,
        str,
    ):
        raise TypeError(("response_variable must be a string."))

    environmental_variable = environmental_variable.strip()

    response_variable = response_variable.strip()

    if not (environmental_variable):
        raise ValueError(("environmental_variable cannot be empty."))

    if not (response_variable):
        raise ValueError(("response_variable cannot be empty."))

    # ==============================================================
    # PARAMETERS
    # ==============================================================

    alpha = _validate_alpha(alpha)

    min_samples = _validate_min_samples(min_samples)

    neutral_threshold = _validate_neutral_threshold(neutral_threshold)

    # ==============================================================
    # FINITE PAIRED DATA
    # ==============================================================

    (
        x,
        y,
    ) = finite_pairs(
        x_values,
        y_values,
    )

    sample_count = int(x.size)

    # ==============================================================
    # INSUFFICIENT DATA
    # ==============================================================

    if sample_count < min_samples:
        return EnvironmentalAssociation(
            environmental_variable=environmental_variable,
            response_variable=response_variable,
            sample_count=sample_count,
            coefficient=None,
            p_value=None,
            direction=AssociationDirection.NEUTRAL,
            strength=AssociationStrength.NEGLIGIBLE,
            statistically_significant=False,
            method="spearman",
        )

    # ==============================================================
    # CONSTANT INPUT
    # ==============================================================

    if _is_constant(x) or _is_constant(y):
        return EnvironmentalAssociation(
            environmental_variable=environmental_variable,
            response_variable=response_variable,
            sample_count=sample_count,
            coefficient=None,
            p_value=None,
            direction=AssociationDirection.NEUTRAL,
            strength=AssociationStrength.NEGLIGIBLE,
            statistically_significant=False,
            method="spearman",
        )

    # ==============================================================
    # SPEARMAN RANK CORRELATION
    # ==============================================================

    result = spearmanr(
        x,
        y,
        nan_policy="omit",
    )

    coefficient = float(result.statistic)

    p_value = float(result.pvalue)

    # ==============================================================
    # DEFENSIVE NUMERICAL CHECK
    # ==============================================================

    if not math.isfinite(coefficient) or not math.isfinite(p_value):
        return EnvironmentalAssociation(
            environmental_variable=environmental_variable,
            response_variable=response_variable,
            sample_count=sample_count,
            coefficient=None,
            p_value=None,
            direction=AssociationDirection.NEUTRAL,
            strength=AssociationStrength.NEGLIGIBLE,
            statistically_significant=False,
            method="spearman",
        )

    # --------------------------------------------------------------
    # Protect against extremely small floating-point excursions.
    # --------------------------------------------------------------

    coefficient = min(
        1.0,
        max(
            -1.0,
            coefficient,
        ),
    )

    p_value = min(
        1.0,
        max(
            0.0,
            p_value,
        ),
    )

    # ==============================================================
    # INTERPRETATION
    # ==============================================================

    direction = classify_association_direction(
        coefficient,
        neutral_threshold=neutral_threshold,
    )

    strength = classify_association_strength(coefficient)

    statistically_significant = bool(p_value < alpha)

    # ==============================================================
    # RESULT
    # ==============================================================

    return EnvironmentalAssociation(
        environmental_variable=environmental_variable,
        response_variable=response_variable,
        sample_count=sample_count,
        coefficient=coefficient,
        p_value=p_value,
        direction=direction,
        strength=strength,
        statistically_significant=statistically_significant,
        method="spearman",
    )


# ======================================================================
# EXTRACT SERIES FROM ROWS
# ======================================================================


def extract_numeric_series(
    rows: Iterable[Any],
    field_name: str,
) -> tuple[
    float | None,
    ...,
]:
    """
    Extract one optional numeric field from rows.

    Invalid/non-finite values become None so pairing logic can remove
    them appropriately.
    """

    if not isinstance(
        field_name,
        str,
    ):
        raise TypeError("field_name must be a string.")

    field_name = field_name.strip()

    if not (field_name):
        raise ValueError("field_name cannot be empty.")

    values: list[float | None] = []

    for row in rows:
        values.append(
            _finite_float_or_none(
                _row_value(
                    row,
                    field_name,
                )
            )
        )

    return tuple(values)


# ======================================================================
# BUILD ENVIRONMENTAL ASSOCIATIONS
# ======================================================================


def build_environmental_associations(
    rows: Iterable[Any],
    *,
    environmental_fields: Mapping[
        str,
        str,
    ]
    | None = None,
    response_fields: Mapping[
        str,
        str,
    ]
    | None = None,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,
) -> tuple[
    EnvironmentalAssociation,
    ...,
]:
    """
    Calculate the full environmental-response association set.

    Parameters
    ----------
    rows
        Normalized temporal observation rows.

        For activity analysis these should normally be regular temporal
        bins, including zero-event bins.

    environmental_fields
        Mapping:

            output variable name -> row field name

        Defaults to:

            temperature_c
            humidity_percent
            pressure_hpa

    response_fields
        Mapping:

            output response name -> row field name

        Defaults to:

            event_count
            active_duration_s
            event_rate_per_hour

    alpha
        Significance threshold.

    min_samples
        Minimum number of finite paired observations.

    Returns
    -------
    tuple[EnvironmentalAssociation, ...]

        One result for every:

            environmental variable
                ×
            response variable

        pair.
    """

    alpha = _validate_alpha(alpha)

    min_samples = _validate_min_samples(min_samples)

    neutral_threshold = _validate_neutral_threshold(neutral_threshold)

    # ==============================================================
    # FIELD MAPS
    # ==============================================================

    environmental_mapping = _validate_field_mapping(
        (
            DEFAULT_ENVIRONMENTAL_FIELDS
            if environmental_fields is None
            else environmental_fields
        ),
        name="environmental_fields",
    )

    response_mapping = _validate_field_mapping(
        (DEFAULT_RESPONSE_FIELDS if response_fields is None else response_fields),
        name="response_fields",
    )

    # ==============================================================
    # MATERIALIZE ONCE
    # ==============================================================

    materialized_rows = tuple(rows)

    # ==============================================================
    # PRE-EXTRACT SERIES
    # ==============================================================

    environmental_series = {
        logical_name: extract_numeric_series(
            materialized_rows,
            row_field,
        )
        for logical_name, row_field in environmental_mapping.items()
    }

    response_series = {
        logical_name: extract_numeric_series(
            materialized_rows,
            row_field,
        )
        for logical_name, row_field in response_mapping.items()
    }

    # ==============================================================
    # CROSS PRODUCT
    # ==============================================================

    associations: list[EnvironmentalAssociation] = []

    for environmental_name in environmental_mapping:
        x_values = environmental_series[environmental_name]

        for response_name in response_mapping:
            y_values = response_series[response_name]

            association = calculate_spearman_association(
                x_values,
                y_values,
                environmental_variable=environmental_name,
                response_variable=response_name,
                alpha=alpha,
                min_samples=min_samples,
                neutral_threshold=neutral_threshold,
            )

            associations.append(association)

    return tuple(associations)


# ======================================================================
# FILTER VALID ASSOCIATIONS
# ======================================================================


def valid_environmental_associations(
    associations: Iterable[EnvironmentalAssociation],
) -> tuple[
    EnvironmentalAssociation,
    ...,
]:
    """
    Return only associations with an actual computed coefficient.
    """

    result: list[EnvironmentalAssociation] = []

    for association in associations:
        if not isinstance(
            association,
            EnvironmentalAssociation,
        ):
            raise TypeError(
                ("associations must contain EnvironmentalAssociation objects.")
            )

        if association.coefficient is not None:
            result.append(association)

    return tuple(result)


# ======================================================================
# FILTER SIGNIFICANT ASSOCIATIONS
# ======================================================================


def significant_environmental_associations(
    associations: Iterable[EnvironmentalAssociation],
) -> tuple[
    EnvironmentalAssociation,
    ...,
]:
    """
    Return only associations passing their configured significance test.

    Important
    ---------
    Statistical significance does not imply:

        biological importance
        practical importance
        causation
    """

    result: list[EnvironmentalAssociation] = []

    for association in associations:
        if not isinstance(
            association,
            EnvironmentalAssociation,
        ):
            raise TypeError(
                ("associations must contain EnvironmentalAssociation objects.")
            )

        if association.statistically_significant:
            result.append(association)

    return tuple(result)


# ======================================================================
# SORT BY ABSOLUTE ASSOCIATION
# ======================================================================


def rank_environmental_associations(
    associations: Iterable[EnvironmentalAssociation],
) -> tuple[
    EnvironmentalAssociation,
    ...,
]:
    """
    Sort computed associations from strongest to weakest |rho|.

    Undefined associations are placed last.

    Deterministic alphabetical tie-breaking is used.
    """

    materialized = tuple(associations)

    for association in materialized:
        if not isinstance(
            association,
            EnvironmentalAssociation,
        ):
            raise TypeError(
                ("associations must contain EnvironmentalAssociation objects.")
            )

    def sort_key(
        association: EnvironmentalAssociation,
    ):

        if association.coefficient is None:
            return (
                1,
                0.0,
                association.environmental_variable,
                association.response_variable,
            )

        return (
            0,
            -abs(association.coefficient),
            association.environmental_variable,
            association.response_variable,
        )

    return tuple(
        sorted(
            materialized,
            key=sort_key,
        )
    )


# ======================================================================
# P-VALUE ADJUSTMENT — BENJAMINI-HOCHBERG
# ======================================================================


def benjamini_hochberg_adjust(
    p_values: Sequence[float | None],
) -> tuple[
    float | None,
    ...,
]:
    """
    Apply Benjamini-Hochberg false-discovery-rate adjustment.

    None values are preserved and excluded from the correction family.

    This helper is provided for later research-report use when many
    environmental hypotheses are evaluated simultaneously.

    The basic EnvironmentalAssociation model currently stores one
    p-value, so this adjustment is not silently applied by
    build_environmental_associations().
    """

    # ==============================================================
    # COLLECT VALID P VALUES
    # ==============================================================

    indexed_values: list[
        tuple[
            int,
            float,
        ]
    ] = []

    output: list[float | None] = [None for _ in p_values]

    for index, value in enumerate(p_values):
        if value is None:
            continue

        try:
            numeric = float(value)

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(("p-values must be numeric or None.")) from exc

        if not math.isfinite(numeric) or not (0.0 <= numeric <= 1.0):
            raise ValueError(("p-values must be finite and in [0, 1]."))

        indexed_values.append(
            (
                index,
                numeric,
            )
        )

    hypothesis_count = len(indexed_values)

    if hypothesis_count == 0:
        return tuple(output)

    # ==============================================================
    # ASCENDING RAW P-VALUE ORDER
    # ==============================================================

    ordered = sorted(
        indexed_values,
        key=lambda item: item[1],
    )

    # ==============================================================
    # RAW BH VALUES
    # ==============================================================

    adjusted_sorted = [0.0 for _ in range(hypothesis_count)]

    for rank, (
        _original_index,
        p_value,
    ) in enumerate(
        ordered,
        start=1,
    ):
        adjusted_sorted[rank - 1] = min(
            1.0,
            (p_value * hypothesis_count / rank),
        )

    # ==============================================================
    # MONOTONICITY CORRECTION
    # ==============================================================

    for index in range(
        hypothesis_count - 2,
        -1,
        -1,
    ):
        adjusted_sorted[index] = min(
            adjusted_sorted[index],
            adjusted_sorted[index + 1],
        )

    # ==============================================================
    # RESTORE ORIGINAL ORDER
    # ==============================================================

    for (
        (
            original_index,
            _raw_p_value,
        ),
        adjusted,
    ) in zip(
        ordered,
        adjusted_sorted,
        strict=False,
    ):
        output[original_index] = float(adjusted)

    return tuple(output)
