"""
Tests for localization.engine.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. LocalizationResult success delegation
    2. LocalizationEngine constructor validation
    3. localization-window length validation
    4. start-sample validation
    5. optional array solver bounds
    6. common acquisition-session detection
    7. explicit sound-speed override
    8. environmental sound-speed correction
    9. environmental lookup at window center
    10. invalid-environment fallback
    11. configured speed fallback
    12. raw per-node RMS calculation
    13. localization band-pass invocation
    14. GCC-PHAT pair construction
    15. GCC-PHAT signal/reference orientation
    16. physical pair-delay limits
    17. low-energy TDOA rejection
    18. GCC validity propagation
    19. defensive physical-delay rejection
    20. infinite peak-ratio normalization
    21. solver argument forwarding
    22. session-change protection
    23. invalid stream-window protection
    24. locate_latest common-window selection
    25. locate_latest unavailable-window behaviour

Architecture
------------
The lower-level mathematical components are tested separately:

    test_filtering.py
        localization conditioning

    test_gcc_phat.py
        waveform delay estimation

    test_tdoa.py
        physical pair measurement

    test_solver.py
        nonlinear source-position reconstruction

This module tests the orchestration layer that combines those pieces:

    StreamManager
        ↓
    LocalizationEngine
        ├── session validation
        ├── absolute sample windows
        ├── environmental sound speed
        ├── raw RMS
        ├── band-pass conditioning
        ├── GCC-PHAT
        ├── TDOA construction
        └── solve_position()
                ↓
        LocalizationResult
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import math

from dataclasses import (
    replace,
)

from types import (
    SimpleNamespace,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np
import pytest


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


import localization.engine as engine_module


from config import (
    CONFIG,
    LocalizationConfig,
)

from localization.engine import (
    MIN_LOCALIZATION_WINDOW_SAMPLES,
    LocalizationEngine,
    LocalizationResult,
)

from protocol import (
    EnvironmentPayload,
)

from stream_manager import (
    StreamManager,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SESSION_ID = (
    0x12345678
)


SECOND_SESSION_ID = (
    0x87654321
)


SAMPLE_RATE = (
    48_000
)


TEST_NODE_POSITIONS = {
    1: (
        0.0,
        0.0,
    ),

    2: (
        0.5,
        0.8660254037844386,
    ),

    3: (
        1.0,
        0.0,
    ),
}


# ======================================================================
# TEST HELPERS
# ======================================================================


def make_localization_config(
    **overrides,
) -> LocalizationConfig:
    """
    Derive a deterministic localization configuration from the real
    project configuration.

    Band-pass filtering is disabled by default in orchestration tests so
    filtering behaviour can be isolated to test_filtering.py.

    Individual tests explicitly enable it when required.
    """

    defaults = {
        "node_positions":
            dict(
                TEST_NODE_POSITIONS
            ),

        "window_samples":
            1024,

        "bandpass_enabled":
            False,

        "speed_of_sound_mps":
            343.0,

        "use_environmental_speed":
            False,

        "min_rms":
            1.0,

        "interpolation":
            8,

        "min_peak_ratio":
            1.10,

        "constrain_to_array_bounds":
            False,
    }

    defaults.update(
        overrides
    )

    return replace(
        CONFIG.localization,
        **defaults,
    )


def activate_common_session(
    streams: StreamManager,
    *,
    session_id: int = SESSION_ID,
) -> None:
    """
    Put all registered node states into one common acquisition session.
    """

    for state in (
        streams.nodes.values()
    ):

        state.reset_stream_tracking(
            session_id=
                session_id
        )


def install_windows(
    monkeypatch,
    streams: StreamManager,
    windows: dict[
        int,
        np.ndarray,
    ],
):
    """
    Replace StreamManager.get_window() with deterministic arrays.

    A call log is returned so tests can verify absolute window
    extraction.
    """

    calls = []

    def fake_get_window(
        node_id: int,
        start_sample: int,
        length: int,
        *,
        fill_value: int = 0,
    ) -> np.ndarray:

        calls.append(
            (
                node_id,
                start_sample,
                length,
                fill_value,
            )
        )

        signal = np.asarray(
            windows[
                node_id
            ]
        )

        return signal[
            :length
        ].copy()

    monkeypatch.setattr(
        streams,
        "get_window",
        fake_get_window,
    )

    return (
        calls
    )


def install_no_environment(
    monkeypatch,
    streams: StreamManager,
) -> None:
    """
    Make environmental lookup deterministically return None.
    """

    monkeypatch.setattr(
        streams,
        "get_environment_near",
        lambda sample_index:
            None,
    )


def make_solver_result(
    *,
    success: bool = True,
    x: float = 0.45,
    y: float = 0.35,
):
    """
    Lightweight position-solver result test double.

    LocalizationResult only requires the position object to expose the
    solver fields it consumes.
    """

    return SimpleNamespace(
        success=
            success,

        x=
            x,

        y=
            y,

        residual_rms_seconds=
            0.0,

        residual_rms_meters=
            0.0,

        cost=
            0.0,

        nfev=
            1,

        message=
            "synthetic solver result",
    )


def make_gcc_result(
    *,
    delay_samples: float = 0.0,
    delay_seconds: float | None = None,
    peak_ratio: float = 2.0,
    valid: bool = True,
    reason: str = "",
):
    """
    Lightweight GCC-PHAT result object exposing the exact attributes
    consumed by LocalizationEngine.
    """

    if (
        delay_seconds
        is None
    ):

        delay_seconds = (
            float(
                delay_samples
            )
            / SAMPLE_RATE
        )

    return SimpleNamespace(
        delay_samples=
            float(
                delay_samples
            ),

        delay_seconds=
            float(
                delay_seconds
            ),

        peak_ratio=
            float(
                peak_ratio
            ),

        valid=
            bool(
                valid
            ),

        reason=
            reason,
    )


def install_successful_gcc(
    monkeypatch,
):
    """
    Install deterministic zero-delay GCC-PHAT.

    Returns a call log.
    """

    calls = []

    def fake_gcc_phat(
        signal,
        reference,
        *,
        sample_rate,
        max_delay_seconds,
        interpolation,
        min_peak_ratio,
    ):

        calls.append(
            {
                "signal":
                    np.asarray(
                        signal
                    ).copy(),

                "reference":
                    np.asarray(
                        reference
                    ).copy(),

                "sample_rate":
                    sample_rate,

                "max_delay_seconds":
                    max_delay_seconds,

                "interpolation":
                    interpolation,

                "min_peak_ratio":
                    min_peak_ratio,
            }
        )

        return make_gcc_result()

    monkeypatch.setattr(
        engine_module,
        "gcc_phat",
        fake_gcc_phat,
    )

    return (
        calls
    )


def install_solver(
    monkeypatch,
    *,
    result=None,
):
    """
    Install a deterministic nonlinear-solver test double.

    Returns:

        (solver_calls, solver_result)
    """

    if (
        result
        is None
    ):

        result = (
            make_solver_result()
        )

    calls = []

    def fake_solve_position(
        node_positions,
        measurements,
        *,
        speed_of_sound_mps,
        bounds=None,
    ):

        calls.append(
            {
                "node_positions":
                    node_positions,

                "measurements":
                    tuple(
                        measurements
                    ),

                "speed_of_sound_mps":
                    speed_of_sound_mps,

                "bounds":
                    bounds,
            }
        )

        return (
            result
        )

    monkeypatch.setattr(
        engine_module,
        "solve_position",
        fake_solve_position,
    )

    return (
        calls,
        result,
    )


def make_standard_windows(
    *,
    length: int = 1024,
) -> dict[
    int,
    np.ndarray,
]:
    """
    Three distinguishable deterministic localization waveforms.

    Different amplitudes make RMS calculations and GCC signal/reference
    orientation easy to verify.
    """

    return {
        1:
            np.full(
                length,
                100.0,
                dtype=np.float64,
            ),

        2:
            np.full(
                length,
                200.0,
                dtype=np.float64,
            ),

        3:
            np.full(
                length,
                300.0,
                dtype=np.float64,
            ),
    }


# ======================================================================
# LOCALIZATION RESULT
# ======================================================================


def test_localization_result_success_reflects_position_success() -> None:

    result = LocalizationResult(
        position=
            make_solver_result(
                success=
                    True
            ),

        measurements=
            (),

        window_start_sample=
            100,

        window_samples=
            1024,

        speed_of_sound_mps=
            343.0,

        node_rms={
            1:
                100.0,

            2:
                100.0,

            3:
                100.0,
        },

        environment_used=
            None,
    )

    assert (
        result.success
        is True
    )


def test_localization_result_failure_reflects_position_failure() -> None:

    result = LocalizationResult(
        position=
            make_solver_result(
                success=
                    False
            ),

        measurements=
            (),

        window_start_sample=
            100,

        window_samples=
            1024,

        speed_of_sound_mps=
            343.0,

        node_rms={},

        environment_used=
            None,
    )

    assert (
        result.success
        is False
    )


# ======================================================================
# CONSTRUCTOR
# ======================================================================


def test_engine_accepts_stream_manager_and_localization_config(
    stream_manager,
) -> None:

    config = (
        make_localization_config()
    )

    engine = LocalizationEngine(
        stream_manager,
        config,
    )

    assert (
        engine.streams
        is stream_manager
    )

    assert (
        engine.config
        is config
    )


def test_engine_rejects_non_stream_manager() -> None:

    with pytest.raises(
        TypeError
    ):

        LocalizationEngine(
            object(),
            make_localization_config(),
        )


def test_engine_rejects_non_localization_config(
    stream_manager,
) -> None:

    with pytest.raises(
        TypeError
    ):

        LocalizationEngine(
            stream_manager,
            object(),
        )


# ======================================================================
# WINDOW LENGTH
# ======================================================================


def test_resolve_window_length_uses_configuration_default(
    stream_manager,
) -> None:

    config = make_localization_config(
        window_samples=
            2048
    )

    engine = LocalizationEngine(
        stream_manager,
        config,
    )

    assert (
        engine._resolve_window_length(
            None
        )
        == 2048
    )


def test_resolve_window_length_accepts_valid_override(
    stream_manager,
) -> None:

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    assert (
        engine._resolve_window_length(
            512
        )
        == 512
    )


@pytest.mark.parametrize(
    "length",
    [
        True,
        False,
        64.5,
        "1024",
    ],
)
def test_resolve_window_length_rejects_non_integer_override(
    stream_manager,
    length,
) -> None:

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    with pytest.raises(
        TypeError
    ):

        engine._resolve_window_length(
            length
        )


@pytest.mark.parametrize(
    "length",
    [
        0,
        1,
        32,
        MIN_LOCALIZATION_WINDOW_SAMPLES
        - 1,
    ],
)
def test_resolve_window_length_rejects_too_short_window(
    stream_manager,
    length: int,
) -> None:

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    with pytest.raises(
        ValueError
    ):

        engine._resolve_window_length(
            length
        )


def test_minimum_localization_window_length_is_accepted(
    stream_manager,
) -> None:

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    assert (
        engine._resolve_window_length(
            MIN_LOCALIZATION_WINDOW_SAMPLES
        )
        == MIN_LOCALIZATION_WINDOW_SAMPLES
    )


# ======================================================================
# START SAMPLE
# ======================================================================


@pytest.mark.parametrize(
    "start_sample",
    [
        0,
        1,
        10_000,
    ],
)
def test_validate_start_sample_accepts_nonnegative_integer(
    start_sample: int,
) -> None:

    assert (
        LocalizationEngine
        ._validate_start_sample(
            start_sample
        )
        == start_sample
    )


@pytest.mark.parametrize(
    "start_sample",
    [
        True,
        False,
        10.5,
        "100",
    ],
)
def test_validate_start_sample_rejects_non_integer(
    start_sample,
) -> None:

    with pytest.raises(
        TypeError
    ):

        LocalizationEngine._validate_start_sample(
            start_sample
        )


def test_validate_start_sample_rejects_negative_value() -> None:

    with pytest.raises(
        ValueError
    ):

        LocalizationEngine._validate_start_sample(
            -1
        )


# ======================================================================
# SOLVER BOUNDS
# ======================================================================


def test_bounds_returns_none_when_constraints_disabled(
    stream_manager,
) -> None:

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            constrain_to_array_bounds=
                False
        ),
    )

    assert (
        engine._bounds()
        is None
    )


def test_bounds_are_derived_from_microphone_geometry(
    stream_manager,
) -> None:

    config = make_localization_config(
        constrain_to_array_bounds=
            True,

        bounds_margin_m=
            0.25,
    )

    engine = LocalizationEngine(
        stream_manager,
        config,
    )

    lower, upper = (
        engine._bounds()
    )

    assert lower[
        0
    ] == pytest.approx(
        -0.25
    )

    assert lower[
        1
    ] == pytest.approx(
        -0.25
    )

    assert upper[
        0
    ] == pytest.approx(
        1.25
    )

    assert upper[
        1
    ] == pytest.approx(
        0.8660254037844386
        + 0.25
    )


# ======================================================================
# COMMON SESSION
# ======================================================================


def test_common_stream_session_returns_common_active_session(
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    assert (
        engine._common_stream_session()
        == SESSION_ID
    )


def test_common_stream_session_returns_none_without_active_session(
    stream_manager,
) -> None:

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    assert (
        engine._common_stream_session()
        is None
    )


def test_common_stream_session_returns_none_for_mixed_sessions(
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    stream_manager.nodes[
        3
    ].reset_stream_tracking(
        session_id=
            SECOND_SESSION_ID
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    assert (
        engine._common_stream_session()
        is None
    )


def test_common_stream_session_returns_none_when_configured_node_missing(
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    config = make_localization_config(
        node_positions={
            **TEST_NODE_POSITIONS,

            4: (
                2.0,
                2.0,
            ),
        }
    )

    engine = LocalizationEngine(
        stream_manager,
        config,
    )

    assert (
        engine._common_stream_session()
        is None
    )


# ======================================================================
# SPEED VALIDATION
# ======================================================================


@pytest.mark.parametrize(
    "speed",
    [
        300.0,
        343.0,
        360.0,
    ],
)
def test_validate_speed_of_sound_accepts_positive_finite_values(
    speed: float,
) -> None:

    result = LocalizationEngine._validate_speed_of_sound(
        speed,
        name=
            "speed",
    )

    assert (
        result
        == pytest.approx(
            speed
        )
    )


@pytest.mark.parametrize(
    "speed",
    [
        0.0,
        -1.0,
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
def test_validate_speed_of_sound_rejects_invalid_value(
    speed: float,
) -> None:

    with pytest.raises(
        ValueError
    ):

        LocalizationEngine._validate_speed_of_sound(
            speed,
            name=
                "speed",
        )


def test_validate_speed_of_sound_rejects_non_numeric_value() -> None:

    with pytest.raises(
        TypeError
    ):

        LocalizationEngine._validate_speed_of_sound(
            object(),
            name=
                "speed",
        )


# ======================================================================
# SPEED RESOLUTION — EXPLICIT OVERRIDE
# ======================================================================


def test_explicit_speed_override_has_highest_priority(
    monkeypatch,
    stream_manager,
) -> None:

    environment_calls = []

    monkeypatch.setattr(
        stream_manager,
        "get_environment_near",
        lambda sample_index:
            environment_calls.append(
                sample_index
            ),
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            use_environmental_speed=
                True,

            speed_of_sound_mps=
                340.0,
        ),
    )

    speed, environment_used = engine._resolve_speed(
        sample_index=
            5000,

        speed_of_sound_mps=
            350.0,
    )

    assert (
        speed
        == pytest.approx(
            350.0
        )
    )

    assert (
        environment_used
        is None
    )

    # Explicit override bypasses telemetry entirely.
    assert (
        environment_calls
        == []
    )


# ======================================================================
# SPEED RESOLUTION — ENVIRONMENT
# ======================================================================


def test_environmental_speed_is_used_when_enabled(
    monkeypatch,
    stream_manager,
) -> None:

    environment = EnvironmentPayload(
        25.0,
        60.0,
        1008.0,
    )

    monkeypatch.setattr(
        stream_manager,
        "get_environment_near",
        lambda sample_index:
            environment,
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            use_environmental_speed=
                True,

            speed_of_sound_mps=
                330.0,
        ),
    )

    speed, environment_used = engine._resolve_speed(
        sample_index=
            12345,

        speed_of_sound_mps=
            None,
    )

    expected = (
        engine_module
        .calculate_speed_of_sound_mps(
            25.0,
            60.0,
            1008.0,
        )
    )

    assert (
        speed
        == pytest.approx(
            expected
        )
    )

    assert (
        environment_used
        == (
            25.0,
            60.0,
            1008.0,
        )
    )


def test_missing_environment_uses_configured_fallback(
    monkeypatch,
    stream_manager,
) -> None:

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            use_environmental_speed=
                True,

            speed_of_sound_mps=
                341.5,
        ),
    )

    speed, environment_used = engine._resolve_speed(
        sample_index=
            1000,

        speed_of_sound_mps=
            None,
    )

    assert (
        speed
        == pytest.approx(
            341.5
        )
    )

    assert (
        environment_used
        is None
    )


def test_invalid_environment_uses_configured_fallback(
    monkeypatch,
    stream_manager,
) -> None:
    """
    Invalid environmental telemetry must not destroy localization.
    """

    invalid_environment = SimpleNamespace(
        temperature_c=
            25.0,

        humidity_percent=
            50.0,

        pressure_hpa=
            0.0,
    )

    monkeypatch.setattr(
        stream_manager,
        "get_environment_near",
        lambda sample_index:
            invalid_environment,
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            use_environmental_speed=
                True,

            speed_of_sound_mps=
                342.0,
        ),
    )

    speed, environment_used = engine._resolve_speed(
        sample_index=
            1000,

        speed_of_sound_mps=
            None,
    )

    assert (
        speed
        == pytest.approx(
            342.0
        )
    )

    assert (
        environment_used
        is None
    )


# ======================================================================
# LOCATE WINDOW — SESSION REQUIREMENT
# ======================================================================


def test_locate_window_requires_common_active_session(
    stream_manager,
) -> None:

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    with pytest.raises(
        RuntimeError
    ):

        engine.locate_window(
            start_sample=
                0,

            length=
                1024,
        )


# ======================================================================
# LOCATE WINDOW — ENVIRONMENT LOOKUP CENTER
# ======================================================================


def test_locate_window_looks_up_environment_at_window_center(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    windows = make_standard_windows()

    install_windows(
        monkeypatch,
        stream_manager,
        windows,
    )

    lookup_samples = []

    monkeypatch.setattr(
        stream_manager,
        "get_environment_near",
        lambda sample_index:
            (
                lookup_samples.append(
                    sample_index
                )
                or None
            ),
    )

    install_successful_gcc(
        monkeypatch
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            use_environmental_speed=
                True
        ),
    )

    engine.locate_window(
        start_sample=
            10_000,

        length=
            1024,
    )

    assert (
        lookup_samples
        == [
            10_000
            + 1024
            // 2
        ]
    )


# ======================================================================
# LOCATE WINDOW — ABSOLUTE WINDOW EXTRACTION
# ======================================================================


def test_locate_window_requests_same_absolute_window_from_every_node(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    calls = install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    install_successful_gcc(
        monkeypatch
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    engine.locate_window(
        start_sample=
            5000,

        length=
            1024,
    )

    assert [
        (
            node_id,
            start,
            length,
        )

        for (
            node_id,
            start,
            length,
            _fill,
        )
        in calls
    ] == [
        (
            1,
            5000,
            1024,
        ),
        (
            2,
            5000,
            1024,
        ),
        (
            3,
            5000,
            1024,
        ),
    ]


# ======================================================================
# LOCATE WINDOW — NODE RMS
# ======================================================================


def test_locate_window_calculates_raw_pcm_rms_per_node(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    install_successful_gcc(
        monkeypatch
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    assert (
        result.node_rms[
            1
        ]
        == pytest.approx(
            100.0
        )
    )

    assert (
        result.node_rms[
            2
        ]
        == pytest.approx(
            200.0
        )
    )

    assert (
        result.node_rms[
            3
        ]
        == pytest.approx(
            300.0
        )
    )


# ======================================================================
# LOCATE WINDOW — BAND-PASS
# ======================================================================


def test_bandpass_is_applied_to_every_node_when_enabled(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    filter_calls = []

    def fake_bandpass(
        signal,
        *,
        sample_rate,
        low_hz,
        high_hz,
        order,
    ):

        filter_calls.append(
            {
                "signal":
                    np.asarray(
                        signal
                    ).copy(),

                "sample_rate":
                    sample_rate,

                "low_hz":
                    low_hz,

                "high_hz":
                    high_hz,

                "order":
                    order,
            }
        )

        return np.ascontiguousarray(
            signal,
            dtype=np.float64,
        )

    monkeypatch.setattr(
        engine_module,
        "bandpass_filter",
        fake_bandpass,
    )

    install_successful_gcc(
        monkeypatch
    )

    install_solver(
        monkeypatch
    )

    config = make_localization_config(
        bandpass_enabled=
            True,

        bandpass_low_hz=
            250.0,

        bandpass_high_hz=
            10_000.0,

        bandpass_order=
            6,
    )

    engine = LocalizationEngine(
        stream_manager,
        config,
    )

    engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    assert (
        len(
            filter_calls
        )
        == 3
    )

    for call in (
        filter_calls
    ):

        assert (
            call[
                "sample_rate"
            ]
            == pytest.approx(
                SAMPLE_RATE
            )
        )

        assert (
            call[
                "low_hz"
            ]
            == pytest.approx(
                250.0
            )
        )

        assert (
            call[
                "high_hz"
            ]
            == pytest.approx(
                10_000.0
            )
        )

        assert (
            call[
                "order"
            ]
            == 6
        )


def test_bandpass_is_not_called_when_disabled(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    def unexpected_bandpass(
        *args,
        **kwargs,
    ):

        raise AssertionError(
            "bandpass_filter should not have been called"
        )

    monkeypatch.setattr(
        engine_module,
        "bandpass_filter",
        unexpected_bandpass,
    )

    install_successful_gcc(
        monkeypatch
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            bandpass_enabled=
                False
        ),
    )

    engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )


# ======================================================================
# GCC PAIRS
# ======================================================================


def test_three_nodes_generate_three_pairwise_gcc_calls(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    gcc_calls = (
        install_successful_gcc(
            monkeypatch
        )
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    assert (
        len(
            gcc_calls
        )
        == 3
    )

    assert (
        len(
            result.measurements
        )
        == 3
    )

    assert [
        (
            measurement.node_a,
            measurement.node_b,
        )
        for measurement
        in result.measurements
    ] == [
        (
            1,
            2,
        ),
        (
            1,
            3,
        ),
        (
            2,
            3,
        ),
    ]


# ======================================================================
# GCC SIGN / ARGUMENT ORIENTATION
# ======================================================================


def test_gcc_receives_node_b_as_signal_and_node_a_as_reference(
    monkeypatch,
    stream_manager,
) -> None:
    """
    Critical sign-convention regression.

    For pair (A, B):

        gcc_phat(
            signal=B,
            reference=A,
        )

    therefore positive GCC delay means:

        arrival_B > arrival_A

    exactly matching TDOAMeasurement(A, B).
    """

    activate_common_session(
        stream_manager
    )

    windows = make_standard_windows()

    install_windows(
        monkeypatch,
        stream_manager,
        windows,
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    gcc_calls = (
        install_successful_gcc(
            monkeypatch
        )
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    # First pair is (1, 2).
    first_call = (
        gcc_calls[
            0
        ]
    )

    np.testing.assert_array_equal(
        first_call[
            "signal"
        ],
        windows[
            2
        ],
    )

    np.testing.assert_array_equal(
        first_call[
            "reference"
        ],
        windows[
            1
        ],
    )


# ======================================================================
# GCC CONFIGURATION
# ======================================================================


def test_gcc_receives_localization_configuration(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    gcc_calls = (
        install_successful_gcc(
            monkeypatch
        )
    )

    install_solver(
        monkeypatch
    )

    config = make_localization_config(
        interpolation=
            16,

        min_peak_ratio=
            1.35,
    )

    engine = LocalizationEngine(
        stream_manager,
        config,
    )

    engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    for call in gcc_calls:

        assert (
            call[
                "sample_rate"
            ]
            == pytest.approx(
                SAMPLE_RATE
            )
        )

        assert (
            call[
                "interpolation"
            ]
            == 16
        )

        assert (
            call[
                "min_peak_ratio"
            ]
            == pytest.approx(
                1.35
            )
        )

        assert (
            call[
                "max_delay_seconds"
            ]
            > 0.0
        )


# ======================================================================
# LOW ENERGY GATE
# ======================================================================


def test_low_energy_pair_is_marked_invalid_without_calling_gcc(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    windows = {
        1:
            np.full(
                1024,
                0.1,
                dtype=np.float64,
            ),

        2:
            np.full(
                1024,
                100.0,
                dtype=np.float64,
            ),

        3:
            np.full(
                1024,
                100.0,
                dtype=np.float64,
            ),
    }

    install_windows(
        monkeypatch,
        stream_manager,
        windows,
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    gcc_calls = (
        install_successful_gcc(
            monkeypatch
        )
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            min_rms=
                10.0
        ),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    measurements = {
        (
            measurement.node_a,
            measurement.node_b,
        ):
            measurement

        for measurement
        in result.measurements
    }

    # Node 1 participates in two pairs and is below min_rms.
    assert not (
        measurements[
            (
                1,
                2,
            )
        ].valid
    )

    assert (
        measurements[
            (
                1,
                2,
            )
        ].reason
        == "insufficient signal energy"
    )

    assert not (
        measurements[
            (
                1,
                3,
            )
        ].valid
    )

    # Only pair 2-3 should reach GCC.
    assert (
        len(
            gcc_calls
        )
        == 1
    )


# ======================================================================
# GCC VALIDITY
# ======================================================================


def test_invalid_gcc_result_produces_invalid_tdoa_measurement(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    def fake_gcc(
        *args,
        **kwargs,
    ):

        return make_gcc_result(
            delay_samples=
                0.0,

            peak_ratio=
                1.01,

            valid=
                False,

            reason=
                "peak ratio below threshold",
        )

    monkeypatch.setattr(
        engine_module,
        "gcc_phat",
        fake_gcc,
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    assert all(
        not measurement.valid

        for measurement
        in result.measurements
    )

    assert all(
        (
            measurement.reason
            == "peak ratio below threshold"
        )

        for measurement
        in result.measurements
    )


# ======================================================================
# DEFENSIVE PHYSICAL DELAY CHECK
# ======================================================================


def test_delay_outside_physical_pair_limit_is_rejected(
    monkeypatch,
    stream_manager,
) -> None:
    """
    GCC already receives the physical search limit, but the engine
    performs an additional defensive interface check.
    """

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    def fake_gcc(
        signal,
        reference,
        *,
        sample_rate,
        max_delay_seconds,
        interpolation,
        min_peak_ratio,
    ):

        return make_gcc_result(
            delay_seconds=
                max_delay_seconds
                * 2.0,

            delay_samples=
                (
                    max_delay_seconds
                    * 2.0
                    * sample_rate
                ),

            peak_ratio=
                5.0,

            valid=
                True,

            reason=
                "",
        )

    monkeypatch.setattr(
        engine_module,
        "gcc_phat",
        fake_gcc,
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    assert all(
        not measurement.valid

        for measurement
        in result.measurements
    )

    assert all(
        (
            measurement.reason
            == "delay outside physical pair limit"
        )

        for measurement
        in result.measurements
    )


# ======================================================================
# INFINITE PEAK RATIO
# ======================================================================


def test_infinite_gcc_peak_ratio_is_stored_as_large_finite_value(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    def fake_gcc(
        *args,
        **kwargs,
    ):

        return make_gcc_result(
            peak_ratio=
                float(
                    "inf"
                )
        )

    monkeypatch.setattr(
        engine_module,
        "gcc_phat",
        fake_gcc,
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    for measurement in (
        result.measurements
    ):

        assert math.isfinite(
            measurement.peak_ratio
        )

        assert (
            measurement.peak_ratio
            == np.finfo(
                np.float64
            ).max
        )


# ======================================================================
# SOLVER INVOCATION
# ======================================================================


def test_solver_receives_geometry_measurements_speed_and_bounds(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    install_successful_gcc(
        monkeypatch
    )

    solver_calls, _ = install_solver(
        monkeypatch
    )

    config = make_localization_config(
        speed_of_sound_mps=
            345.0,

        constrain_to_array_bounds=
            True,

        bounds_margin_m=
            0.2,
    )

    engine = LocalizationEngine(
        stream_manager,
        config,
    )

    result = engine.locate_window(
        start_sample=
            2000,

        length=
            1024,

        speed_of_sound_mps=
            345.0,
    )

    assert (
        len(
            solver_calls
        )
        == 1
    )

    call = (
        solver_calls[
            0
        ]
    )

    assert (
        call[
            "node_positions"
        ]
        == config.node_positions
    )

    assert (
        len(
            call[
                "measurements"
            ]
        )
        == 3
    )

    assert (
        call[
            "speed_of_sound_mps"
        ]
        == pytest.approx(
            345.0
        )
    )

    assert (
        call[
            "bounds"
        ]
        == engine._bounds()
    )

    assert (
        result.speed_of_sound_mps
        == pytest.approx(
            345.0
        )
    )


# ======================================================================
# RESULT METADATA
# ======================================================================


def test_localization_result_records_window_metadata(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(
            length=
                512
        ),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    install_successful_gcc(
        monkeypatch
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    result = engine.locate_window(
        start_sample=
            7777,

        length=
            512,
    )

    assert (
        result.window_start_sample
        == 7777
    )

    assert (
        result.window_samples
        == 512
    )


# ======================================================================
# ENVIRONMENT RESULT METADATA
# ======================================================================


def test_result_records_environment_used(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    environment = EnvironmentPayload(
        26.0,
        70.0,
        1005.0,
    )

    monkeypatch.setattr(
        stream_manager,
        "get_environment_near",
        lambda sample_index:
            environment,
    )

    install_successful_gcc(
        monkeypatch
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            use_environmental_speed=
                True
        ),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    assert (
        result.environment_used
        == (
            26.0,
            70.0,
            1005.0,
        )
    )


def test_explicit_speed_override_records_no_environment(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_successful_gcc(
        monkeypatch
    )

    install_solver(
        monkeypatch
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            use_environmental_speed=
                True
        ),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,

        speed_of_sound_mps=
            350.0,
    )

    assert (
        result.speed_of_sound_mps
        == pytest.approx(
            350.0
        )
    )

    assert (
        result.environment_used
        is None
    )


# ======================================================================
# INVALID RAW WINDOW SHAPE
# ======================================================================


def test_locate_window_rejects_short_node_window(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    windows = make_standard_windows()

    windows[
        2
    ] = np.zeros(
        1023,
        dtype=np.float64,
    )

    install_windows(
        monkeypatch,
        stream_manager,
        windows,
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    with pytest.raises(
        RuntimeError
    ):

        engine.locate_window(
            start_sample=
                0,

            length=
                1024,
        )


def test_locate_window_rejects_multidimensional_node_window(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    windows = make_standard_windows()

    windows[
        2
    ] = np.zeros(
        (
            1024,
            1,
        ),
        dtype=np.float64,
    )

    install_windows(
        monkeypatch,
        stream_manager,
        windows,
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    with pytest.raises(
        RuntimeError
    ):

        engine.locate_window(
            start_sample=
                0,

            length=
                1024,
        )


# ======================================================================
# NON-FINITE RAW WINDOW
# ======================================================================


def test_locate_window_rejects_nonfinite_samples(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    windows = make_standard_windows()

    windows[
        2
    ][
        100
    ] = (
        np.nan
    )

    install_windows(
        monkeypatch,
        stream_manager,
        windows,
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    with pytest.raises(
        ValueError
    ):

        engine.locate_window(
            start_sample=
                0,

            length=
                1024,
        )


# ======================================================================
# INVALID CONDITIONED WINDOW
# ======================================================================


def test_locate_window_rejects_invalid_bandpass_output_shape(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    monkeypatch.setattr(
        engine_module,
        "bandpass_filter",
        lambda signal, **kwargs:
            np.asarray(
                signal[
                    :-1
                ],
                dtype=np.float64,
            ),
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            bandpass_enabled=
                True
        ),
    )

    with pytest.raises(
        RuntimeError
    ):

        engine.locate_window(
            start_sample=
                0,

            length=
                1024,
        )


# ======================================================================
# SESSION CHANGE DURING EXTRACTION
# ======================================================================


def test_session_change_during_window_assembly_is_rejected(
    monkeypatch,
    stream_manager,
) -> None:
    """
    sampleIndex values from two acquisition sessions must never be mixed.
    """

    activate_common_session(
        stream_manager
    )

    windows = (
        make_standard_windows()
    )

    def fake_get_window(
        node_id: int,
        start_sample: int,
        length: int,
        *,
        fill_value: int = 0,
    ):

        result = (
            windows[
                node_id
            ][
                :length
            ].copy()
        )

        # Node 3 passed the pre-extraction session check, but the
        # acquisition changes immediately afterward. The final global
        # session recheck must detect this.
        if (
            node_id
            == 3
        ):

            stream_manager.nodes[
                3
            ].reset_stream_tracking(
                session_id=
                    SECOND_SESSION_ID
            )

        return (
            result
        )

    monkeypatch.setattr(
        stream_manager,
        "get_window",
        fake_get_window,
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    with pytest.raises(
        RuntimeError
    ):

        engine.locate_window(
            start_sample=
                0,

            length=
                1024,
        )


# ======================================================================
# LOCATE LATEST — NO SESSION
# ======================================================================


def test_locate_latest_returns_none_without_common_session(
    stream_manager,
) -> None:

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    assert (
        engine.locate_latest()
        is None
    )


# ======================================================================
# LOCATE LATEST — MISSING AUDIO
# ======================================================================


def test_locate_latest_returns_none_when_node_has_no_audio(
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    assert (
        engine.locate_latest()
        is None
    )


# ======================================================================
# LOCATE LATEST — COMMON END
# ======================================================================


def test_locate_latest_uses_least_advanced_node_end(
    monkeypatch,
    stream_manager,
    make_audio_block,
) -> None:
    """
    The newest common localization window must end at the least advanced
    node, not the most advanced node.

    Example:

        Node 1 latest end = 1256
        Node 2 latest end = 1356
        Node 3 latest end = 1306

        common_end = 1256

    For length 128:

        start = 1128
    """

    activate_common_session(
        stream_manager
    )

    block_length = (
        256
    )

    starts = {
        1:
            1000,

        2:
            1100,

        3:
            1050,
    }

    for node_id, start in (
        starts.items()
    ):

        block = make_audio_block(
            node_id=
                node_id,

            session_id=
                SESSION_ID,

            sample_index=
                start,

            samples=
                np.zeros(
                    block_length,
                    dtype=np.int16,
                ),
        )

        stream_manager.nodes[
            node_id
        ].audio_blocks.append(
            block
        )

    config = make_localization_config(
        window_samples=
            128
    )

    engine = LocalizationEngine(
        stream_manager,
        config,
    )

    calls = []

    sentinel = object()

    def fake_locate_window(
        *,
        start_sample: int,
        length: int | None = None,
        speed_of_sound_mps: float | None = None,
    ):

        calls.append(
            (
                start_sample,
                length,
                speed_of_sound_mps,
            )
        )

        return (
            sentinel
        )

    monkeypatch.setattr(
        engine,
        "locate_window",
        fake_locate_window,
    )

    result = engine.locate_latest(
        length=
            128
    )

    assert (
        result
        is sentinel
    )

    assert (
        calls
        == [
            (
                1128,
                128,
                None,
            )
        ]
    )


# ======================================================================
# LOCATE LATEST — INSUFFICIENT HISTORY
# ======================================================================


def test_locate_latest_returns_none_when_common_end_is_too_small(
    stream_manager,
    make_audio_block,
) -> None:

    activate_common_session(
        stream_manager
    )

    for node_id in (
        1,
        2,
        3,
    ):

        stream_manager.nodes[
            node_id
        ].audio_blocks.append(
            make_audio_block(
                node_id=
                    node_id,

                session_id=
                    SESSION_ID,

                sample_index=
                    0,

                samples=
                    np.zeros(
                        64,
                        dtype=np.int16,
                    ),
            )
        )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(
            window_samples=
                128
        ),
    )

    assert (
        engine.locate_latest(
            length=
                128
        )
        is None
    )


# ======================================================================
# LOCATE LATEST — STALE BLOCK SESSION
# ======================================================================


def test_locate_latest_rejects_latest_block_from_wrong_session(
    stream_manager,
    make_audio_block,
) -> None:

    activate_common_session(
        stream_manager
    )

    for node_id in (
        1,
        2,
    ):

        stream_manager.nodes[
            node_id
        ].audio_blocks.append(
            make_audio_block(
                node_id=
                    node_id,

                session_id=
                    SESSION_ID,

                sample_index=
                    1000,
            )
        )

    stream_manager.nodes[
        3
    ].audio_blocks.append(
        make_audio_block(
            node_id=
                3,

            session_id=
                SECOND_SESSION_ID,

            sample_index=
                1000,
        )
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    assert (
        engine.locate_latest()
        is None
    )


# ======================================================================
# COMPLETE RESULT USES SOLVER RESULT
# ======================================================================


def test_localization_result_wraps_exact_solver_result_object(
    monkeypatch,
    stream_manager,
) -> None:

    activate_common_session(
        stream_manager
    )

    install_windows(
        monkeypatch,
        stream_manager,
        make_standard_windows(),
    )

    install_no_environment(
        monkeypatch,
        stream_manager,
    )

    install_successful_gcc(
        monkeypatch
    )

    solver_result = make_solver_result(
        success=
            True,

        x=
            0.4,

        y=
            0.3,
    )

    install_solver(
        monkeypatch,
        result=
            solver_result,
    )

    engine = LocalizationEngine(
        stream_manager,
        make_localization_config(),
    )

    result = engine.locate_window(
        start_sample=
            0,

        length=
            1024,
    )

    assert (
        result.position
        is solver_result
    )

    assert (
        result.success
        is True
    )