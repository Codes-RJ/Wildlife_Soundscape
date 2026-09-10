"""
Shared pytest fixtures.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Provide deterministic, reusable test data for the backend test suite.

This file intentionally contains only broadly reusable fixtures.

Subsystem-specific fixtures should remain inside their corresponding
test modules when they are not useful elsewhere.

Examples
--------
Shared here:

    AudioConfig
    NodeState factories
    StreamManager
    PCM16 waveform generation
    AudioBlock generation
    known microphone geometry
    known source position
    deterministic session IDs
    deterministic random generator

Kept inside individual test modules:

    classifier-specific feature vectors
    database-specific rows
    event-pipeline mocks
    protocol corruption cases
    GCC-PHAT edge cases
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import sys

from collections.abc import (
    Callable,
)

from pathlib import (
    Path,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np
import pytest


# ======================================================================
# PROJECT ROOT
# ======================================================================
#
# This allows pytest to locate project modules whether tests are launched
# from:
#
#     project root
#
# or:
#
#     tests/
#
# without requiring the project to already be installed as a Python
# package.
# ======================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.core.config import (
    AudioConfig,
)

from wildlife_soundscape.core.models import (
    AudioBlock,
)

from wildlife_soundscape.acquisition.node import (
    NodeState,
)

from wildlife_soundscape.acquisition.stream_manager import (
    StreamManager,
)


# ======================================================================
# TEST CONSTANTS
# ======================================================================


TEST_SAMPLE_RATE = 48_000


TEST_FRAMES_PER_BLOCK = 1024


TEST_SESSION_ID = 0x12345678


TEST_SECOND_SESSION_ID = 0x87654321


TEST_NODE_IDS = (
    1,
    2,
    3,
)


# ======================================================================
# AUDIO CONFIGURATION
# ======================================================================


@pytest.fixture
def audio_config() -> AudioConfig:
    """
    Standard project-compatible audio configuration used by backend
    tests.

    WAV recording is disabled because unit tests should not create
    persistent recording files unless a test explicitly targets
    WavRecorder.
    """

    return AudioConfig(
        sample_rate=TEST_SAMPLE_RATE,
        frames_per_block=TEST_FRAMES_PER_BLOCK,
        channels=1,
        sample_width_bytes=2,
        buffer_seconds=10,
        sync_tolerance_samples=10,
        record_wav=False,
    )


# ======================================================================
# SESSION IDS
# ======================================================================


@pytest.fixture
def session_id() -> int:
    """
    Stable non-zero Protocol-v4 acquisition session ID.
    """

    return TEST_SESSION_ID


@pytest.fixture
def second_session_id() -> int:
    """
    Different valid acquisition session ID.

    Used for:

        session-transition tests
        stale-packet tests
        cross-session isolation tests
    """

    return TEST_SECOND_SESSION_ID


# ======================================================================
# DETERMINISTIC RANDOM GENERATOR
# ======================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """
    Deterministic NumPy random generator.

    Tests using this shared RNG remain reproducible across executions.
    """

    return np.random.default_rng(20260830)


# ======================================================================
# PCM16 FACTORY
# ======================================================================


@pytest.fixture
def make_pcm16() -> Callable[..., np.ndarray]:
    """
    Factory producing deterministic mono PCM16 waveforms.

    Parameters accepted by the returned factory
    -------------------------------------------
    length
        Number of output samples.

    frequency_hz
        Tone frequency in Hz.

    amplitude
        Sinusoidal amplitude expressed directly in PCM16 sample units.

        Example:

            amplitude = 6000

    sample_rate
        Sampling frequency in Hz.

    phase_rad
        Initial sinusoidal phase in radians.

    noise_std
        Optional Gaussian-noise standard deviation in PCM16 units.

    seed
        Local deterministic noise-generator seed.

    Returns
    -------
    numpy.ndarray
        One-dimensional, C-contiguous signed int16 waveform.
    """

    def _make_pcm16(
        *,
        length: int = TEST_FRAMES_PER_BLOCK,
        frequency_hz: float = 2000.0,
        amplitude: float = 6000.0,
        sample_rate: int = TEST_SAMPLE_RATE,
        phase_rad: float = 0.0,
        noise_std: float = 0.0,
        seed: int = 12345,
    ) -> np.ndarray:

        # ==============================================================
        # LENGTH
        # ==============================================================

        if isinstance(
            length,
            bool,
        ) or not isinstance(
            length,
            int,
        ):
            raise TypeError(("length must be an integer"))

        if length <= 0:
            raise ValueError(("length must be greater than zero"))

        # ==============================================================
        # SAMPLE RATE
        # ==============================================================

        if isinstance(
            sample_rate,
            bool,
        ) or not isinstance(
            sample_rate,
            int,
        ):
            raise TypeError(("sample_rate must be an integer"))

        if sample_rate <= 0:
            raise ValueError(("sample_rate must be greater than zero"))

        # ==============================================================
        # TIME AXIS
        # ==============================================================

        sample_indices = np.arange(
            length,
            dtype=np.float64,
        )

        time_s = sample_indices / float(sample_rate)

        # ==============================================================
        # SINUSOID
        # ==============================================================

        waveform = float(amplitude) * np.sin(
            (2.0 * np.pi * float(frequency_hz) * time_s) + float(phase_rad)
        )

        # ==============================================================
        # OPTIONAL NOISE
        # ==============================================================

        if noise_std > 0.0:
            local_rng = np.random.default_rng(seed)

            waveform = waveform + local_rng.normal(
                loc=0.0,
                scale=float(noise_std),
                size=length,
            )

        # ==============================================================
        # PCM16 RANGE
        # ==============================================================

        waveform = np.clip(
            waveform,
            -32768.0,
            32767.0,
        )

        # ==============================================================
        # OUTPUT
        # ==============================================================

        return np.ascontiguousarray(waveform.astype(np.int16))

    return _make_pcm16


# ======================================================================
# AUDIO BLOCK FACTORY
# ======================================================================


@pytest.fixture
def make_audio_block(
    make_pcm16: Callable[..., np.ndarray],
) -> Callable[..., AudioBlock]:
    """
    Factory producing structurally valid AudioBlock objects.

    Defaults correspond to one normal 1024-frame PCM16 AUDIO packet
    from Node 1.

    Important
    ---------
    This fixture is intended to construct VALID blocks.

    Tests that deliberately need malformed:

        dtype
        dimensions
        identifiers
        ranges

    should construct AudioBlock directly so this helper does not
    accidentally normalize the malformed input.
    """

    def _make_audio_block(
        *,
        node_id: int = 1,
        sequence: int = 0,
        session_id: int = TEST_SESSION_ID,
        sample_index: int = 0,
        local_micros: int = 0,
        i2s_error_count: int = 0,
        flags: int = 0,
        samples: np.ndarray | None = None,
    ) -> AudioBlock:

        # ==============================================================
        # DEFAULT PCM
        # ==============================================================

        if samples is None:
            pcm = make_pcm16()

        else:
            # ----------------------------------------------------------
            # Shared fixture output must always represent a valid mono
            # PCM16 waveform.
            # ----------------------------------------------------------

            pcm = np.asarray(
                samples,
                dtype=np.int16,
            )

            if pcm.ndim != 1:
                raise ValueError(("make_audio_block samples must be mono 1-D audio"))

            if pcm.size == 0:
                raise ValueError(("make_audio_block samples cannot be empty"))

            pcm = np.ascontiguousarray(
                pcm,
                dtype=np.int16,
            )

        # ==============================================================
        # AUDIO BLOCK
        # ==============================================================

        return AudioBlock(
            node_id=node_id,
            sequence=sequence,
            session_id=session_id,
            sample_index=sample_index,
            local_micros=local_micros,
            i2s_error_count=i2s_error_count,
            flags=flags,
            samples=pcm,
        )

    return _make_audio_block


# ======================================================================
# NODE STATE FACTORY
# ======================================================================


@pytest.fixture
def make_node_state(
    audio_config: AudioConfig,
) -> Callable[..., NodeState]:
    """
    Factory producing clean NodeState instances.

    No session is established automatically.

    Individual tests remain responsible for deciding whether they need:

        disconnected/idle state
        active acquisition state
        explicit session transition
    """

    def _make_node_state(
        *,
        node_id: int = 1,
    ) -> NodeState:

        return NodeState(
            node_id=node_id,
            audio_config=audio_config,
        )

    return _make_node_state


# ======================================================================
# THREE CLEAN NODE STATES
# ======================================================================


@pytest.fixture
def node_states(
    make_node_state: Callable[..., NodeState],
) -> dict[
    int,
    NodeState,
]:
    """
    Standard clean three-node runtime state.

    Session identity is deliberately NOT initialized here.

    This avoids hiding session-management behaviour from tests.
    """

    return {node_id: make_node_state(node_id=node_id) for node_id in TEST_NODE_IDS}


# ======================================================================
# STREAM MANAGER
# ======================================================================


@pytest.fixture
def stream_manager(
    audio_config: AudioConfig,
    node_states: dict[
        int,
        NodeState,
    ],
) -> StreamManager:
    """
    StreamManager with Nodes 1, 2 and 3 already registered.

    No session or audio is inserted automatically.

    This gives tests a neutral baseline and prevents implicit fixture
    setup from masking session-transition bugs.
    """

    manager = StreamManager(audio_config)

    for state in node_states.values():
        manager.register_state(state)

    return manager


# ======================================================================
# ACTIVE NODE STATES
# ======================================================================


@pytest.fixture
def active_node_states(
    node_states: dict[
        int,
        NodeState,
    ],
    session_id: int,
) -> dict[
    int,
    NodeState,
]:
    """
    Three node states initialized into the same acquisition session.

    Useful for tests that specifically require a synchronized active
    session without testing session establishment itself.
    """

    for state in node_states.values():
        state.reset_stream_tracking(session_id=session_id)

    return node_states


# ======================================================================
# ACTIVE STREAM MANAGER
# ======================================================================


@pytest.fixture
def active_stream_manager(
    audio_config: AudioConfig,
    active_node_states: dict[
        int,
        NodeState,
    ],
) -> StreamManager:
    """
    StreamManager whose three registered nodes already belong to one
    common active acquisition session.

    Intended primarily for:

        stream reconstruction tests
        localization tests
        event-pipeline tests

    when session establishment itself is not under test.
    """

    manager = StreamManager(audio_config)

    for state in active_node_states.values():
        manager.register_state(state)

    return manager


# ======================================================================
# MICROPHONE GEOMETRY
# ======================================================================


@pytest.fixture
def node_positions() -> dict[
    int,
    tuple[
        float,
        float,
    ],
]:
    """
    Standard one-meter equilateral-triangle microphone geometry.

    Coordinates are expressed in meters.

                  Node 2
                    /\\
                   /  \\
                1m/    \\1m
                 /      \\
                /________\\
             Node 1  1m  Node 3
    """

    return {
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
# KNOWN SOURCE POSITION
# ======================================================================


@pytest.fixture
def known_source_xy() -> tuple[
    float,
    float,
]:
    """
    Known synthetic acoustic-source position used by localization tests.

    The position is deliberately inside the triangular microphone array.
    """

    return (
        0.45,
        0.35,
    )


# ======================================================================
# SPEED OF SOUND
# ======================================================================


@pytest.fixture
def speed_of_sound_mps() -> float:
    """
    Standard fallback sound speed for deterministic localization tests.

    Units:
        meters / second
    """

    return 343.0
