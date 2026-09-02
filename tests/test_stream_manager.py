"""
Tests for stream_manager.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. Node registration
    2. Audio ingestion
    3. Coarse sampleIndex alignment
    4. Alignment tolerance
    5. Buffered nearest-block selection
    6. Session consistency
    7. Absolute sample-index window reconstruction
    8. Preservation of missing-sample gaps
    9. Custom fill values
    10. Stream-buffer reset behavior
    11. Continuous WAV recording
    12. WAV gap preservation
    13. WAV overlap handling

Important
---------
StreamManager performs only coarse digital-stream alignment.

It must never remove the physical acoustic propagation delay.

Fine inter-node delay remains available to GCC-PHAT/TDOA.
"""


from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import wave

from pathlib import (
    Path,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np
import pytest


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

from wildlife_soundscape.core.protocol import (
    SyncPayload,
)

from wildlife_soundscape.acquisition.stream_manager import (
    StreamManager,
    WavRecorder,
)


# ======================================================================
# CONSTANTS
# ======================================================================


TEST_SESSION_ID = (
    0x12345678
)


SECOND_SESSION_ID = (
    0x87654321
)


# ======================================================================
# TEST HELPERS
# ======================================================================


def make_config() -> AudioConfig:
    """
    Small-block configuration for StreamManager unit tests.
    """

    return AudioConfig(
        sample_rate=
            48_000,

        frames_per_block=
            16,

        channels=
            1,

        sample_width_bytes=
            2,

        buffer_seconds=
            1,

        sync_tolerance_samples=
            10,

        record_wav=
            False,
    )


def make_samples(
    *,
    length: int = 16,
    start_value: int = 0,
) -> np.ndarray:
    """
    Produce deterministic PCM16 samples.
    """

    return np.ascontiguousarray(
        np.arange(
            start_value,
            start_value
            + length,
            dtype=np.int16,
        )
    )


def make_block(
    node_id: int,
    start: int,
    *,
    sequence: int = 1,
    session_id: int = TEST_SESSION_ID,
    samples: np.ndarray | None = None,
) -> AudioBlock:
    """
    Build one valid mono PCM16 AudioBlock.
    """

    if (
        samples
        is None
    ):

        samples = (
            make_samples()
        )

    return AudioBlock(
        node_id=
            node_id,

        sequence=
            sequence,

        session_id=
            session_id,

        sample_index=
            start,

        local_micros=
            0,

        i2s_error_count=
            0,

        flags=
            0,

        samples=
            np.ascontiguousarray(
                samples,
                dtype=np.int16,
            ),
    )


def make_manager() -> tuple[
    StreamManager,
    dict[
        int,
        NodeState,
    ],
]:
    """
    Build a StreamManager with Nodes 1, 2 and 3 registered.
    """

    config = (
        make_config()
    )

    manager = (
        StreamManager(
            config
        )
    )

    states: dict[
        int,
        NodeState,
    ] = {}

    for node_id in (
        1,
        2,
        3,
    ):

        state = NodeState(
            node_id=
                node_id,

            audio_config=
                config,
        )

        state.reset_stream_tracking(
            session_id=
                TEST_SESSION_ID
        )

        manager.register_state(
            state
        )

        states[
            node_id
        ] = (
            state
        )

    return (
        manager,
        states,
    )


# ======================================================================
# NODE REGISTRATION
# ======================================================================


def test_register_state() -> None:

    config = (
        make_config()
    )

    manager = (
        StreamManager(
            config
        )
    )

    state = NodeState(
        node_id=
            1,

        audio_config=
            config,
    )

    manager.register_state(
        state
    )

    assert (
        manager.nodes[
            1
        ]
        is state
    )


def test_register_state_replaces_existing_state() -> None:

    config = (
        make_config()
    )

    manager = (
        StreamManager(
            config
        )
    )

    first = NodeState(
        node_id=
            1,

        audio_config=
            config,
    )

    second = NodeState(
        node_id=
            1,

        audio_config=
            config,
    )

    manager.register_state(
        first
    )

    manager.register_state(
        second
    )

    assert (
        manager.nodes[
            1
        ]
        is second
    )


# ======================================================================
# AUDIO INGESTION
# ======================================================================


def test_add_audio_routes_block_to_correct_node() -> None:

    (
        manager,
        states,
    ) = (
        make_manager()
    )

    block = make_block(
        2,
        0,
    )

    manager.add_audio(
        block
    )

    assert (
        len(
            states[
                1
            ].audio_blocks
        )
        == 0
    )

    assert (
        len(
            states[
                2
            ].audio_blocks
        )
        == 1
    )

    assert (
        len(
            states[
                3
            ].audio_blocks
        )
        == 0
    )

    assert (
        states[
            2
        ].audio_blocks[
            0
        ]
        is block
    )


def test_add_audio_rejects_unregistered_node() -> None:

    config = (
        make_config()
    )

    manager = (
        StreamManager(
            config
        )
    )

    block = make_block(
        1,
        0,
    )

    with pytest.raises(
        KeyError
    ):

        manager.add_audio(
            block
        )


# ======================================================================
# COARSE ALIGNMENT
# ======================================================================


def test_alignment_within_tolerance() -> None:
    """
    The least-advanced newest block defines the common coarse target.

        Node 1 = 100
        Node 2 = 103
        Node 3 = 98

    Therefore:

        target = 98

        offsets:
            N1 = +2
            N2 = +5
            N3 =  0
    """

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    manager.add_audio(
        make_block(
            1,
            100,
        )
    )

    manager.add_audio(
        make_block(
            2,
            103,
        )
    )

    manager.add_audio(
        make_block(
            3,
            98,
        )
    )

    aligned = (
        manager.latest_aligned_blocks()
    )

    assert (
        aligned
        is not None
    )

    assert (
        aligned.target_sample_index
        == 98
    )

    assert (
        aligned.offsets
        == {
            1: 2,
            2: 5,
            3: 0,
        }
    )


def test_alignment_at_exact_tolerance_is_valid() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    manager.add_audio(
        make_block(
            1,
            100,
        )
    )

    manager.add_audio(
        make_block(
            2,
            110,
        )
    )

    manager.add_audio(
        make_block(
            3,
            100,
        )
    )

    aligned = (
        manager.latest_aligned_blocks(
            tolerance_samples=
                10
        )
    )

    assert (
        aligned
        is not None
    )

    assert (
        aligned.target_sample_index
        == 100
    )

    assert (
        aligned.offsets[
            2
        ]
        == 10
    )


def test_alignment_outside_tolerance_returns_none() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    manager.add_audio(
        make_block(
            1,
            100,
        )
    )

    manager.add_audio(
        make_block(
            2,
            111,
        )
    )

    manager.add_audio(
        make_block(
            3,
            100,
        )
    )

    aligned = (
        manager.latest_aligned_blocks(
            tolerance_samples=
                10
        )
    )

    assert (
        aligned
        is None
    )


def test_alignment_returns_none_when_requested_node_has_no_audio() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    manager.add_audio(
        make_block(
            1,
            100,
        )
    )

    manager.add_audio(
        make_block(
            2,
            100,
        )
    )

    # Node 3 deliberately has no block.

    assert (
        manager.latest_aligned_blocks()
        is None
    )


def test_alignment_selects_older_buffered_block_nearest_common_target() -> None:
    """
    Important asynchronous-network regression test.

    Node 1 has already received its next TCP packet while Nodes 2 and 3
    are still one block behind.

    The manager should select Node 1's older buffered block rather than
    declaring the streams unsynchronized.
    """

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    # --------------------------------------------------------------
    # NODE 1 IS ONE PACKET AHEAD
    # --------------------------------------------------------------

    manager.add_audio(
        make_block(
            1,
            100,
            sequence=
                1,
        )
    )

    manager.add_audio(
        make_block(
            1,
            116,
            sequence=
                2,
        )
    )

    # --------------------------------------------------------------
    # NODES 2 AND 3
    # --------------------------------------------------------------

    manager.add_audio(
        make_block(
            2,
            100,
            sequence=
                1,
        )
    )

    manager.add_audio(
        make_block(
            3,
            100,
            sequence=
                1,
        )
    )

    aligned = (
        manager.latest_aligned_blocks()
    )

    assert (
        aligned
        is not None
    )

    assert (
        aligned.target_sample_index
        == 100
    )

    assert (
        aligned.blocks[
            1
        ].sample_index
        == 100
    )

    assert (
        aligned.offsets
        == {
            1: 0,
            2: 0,
            3: 0,
        }
    )


def test_alignment_rejects_mixed_sessions() -> None:
    """
    Numerically equal sampleIndex values from different START sessions
    do not represent the same physical time.
    """

    config = (
        make_config()
    )

    manager = (
        StreamManager(
            config
        )
    )

    for (
        node_id,
        session_id,
    ) in (
        (
            1,
            TEST_SESSION_ID,
        ),
        (
            2,
            TEST_SESSION_ID,
        ),
        (
            3,
            SECOND_SESSION_ID,
        ),
    ):

        state = NodeState(
            node_id=
                node_id,

            audio_config=
                config,
        )

        state.reset_stream_tracking(
            session_id=
                session_id
        )

        manager.register_state(
            state
        )

        manager.add_audio(
            make_block(
                node_id,
                100,
                session_id=
                    session_id,
            )
        )

    assert (
        manager.latest_aligned_blocks()
        is None
    )


def test_alignment_rejects_negative_tolerance() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    for node_id in (
        1,
        2,
        3,
    ):

        manager.add_audio(
            make_block(
                node_id,
                0,
            )
        )

    with pytest.raises(
        ValueError
    ):

        manager.latest_aligned_blocks(
            tolerance_samples=
                -1
        )


def test_alignment_rejects_duplicate_requested_nodes() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    with pytest.raises(
        ValueError
    ):

        manager.latest_aligned_blocks(
            node_ids=(
                1,
                1,
                2,
            )
        )


# ======================================================================
# SAMPLE-INDEXED WINDOW RECONSTRUCTION
# ======================================================================


def test_get_window_exact_block() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    samples = make_samples(
        length=
            16,
        start_value=
            100,
    )

    manager.add_audio(
        make_block(
            1,
            1000,
            samples=
                samples,
        )
    )

    result = manager.get_window(
        1,
        1000,
        16,
    )

    np.testing.assert_array_equal(
        result,
        samples,
    )

    assert (
        result.dtype
        == np.int16
    )


def test_get_window_extracts_partial_block() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    samples = make_samples(
        length=
            16,
        start_value=
            10,
    )

    manager.add_audio(
        make_block(
            1,
            100,
            samples=
                samples,
        )
    )

    result = manager.get_window(
        1,
        104,
        8,
    )

    np.testing.assert_array_equal(
        result,
        samples[
            4:12
        ],
    )


def test_get_window_preserves_sample_gap_as_silence() -> None:
    """
    Missing PCM must remain a hole in the sample timeline.

    Timeline:

        0 .. 15      real data
        16 .. 31     missing
        32 .. 47     real data
    """

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    first = np.full(
        16,
        100,
        dtype=np.int16,
    )

    second = np.full(
        16,
        200,
        dtype=np.int16,
    )

    manager.add_audio(
        make_block(
            1,
            0,
            sequence=
                1,
            samples=
                first,
        )
    )

    manager.add_audio(
        make_block(
            1,
            32,
            sequence=
                2,
            samples=
                second,
        )
    )

    result = manager.get_window(
        1,
        0,
        48,
    )

    np.testing.assert_array_equal(
        result[
            0:16
        ],
        first,
    )

    np.testing.assert_array_equal(
        result[
            16:32
        ],
        np.zeros(
            16,
            dtype=np.int16,
        ),
    )

    np.testing.assert_array_equal(
        result[
            32:48
        ],
        second,
    )


def test_get_window_uses_custom_fill_value() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    result = manager.get_window(
        1,
        100,
        8,
        fill_value=
            -1234,
    )

    np.testing.assert_array_equal(
        result,
        np.full(
            8,
            -1234,
            dtype=np.int16,
        ),
    )


def test_get_window_before_first_block_contains_fill_then_pcm() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    samples = np.array(
        [
            10,
            20,
            30,
            40,
        ],
        dtype=np.int16,
    )

    manager.add_audio(
        make_block(
            1,
            104,
            samples=
                samples,
        )
    )

    result = manager.get_window(
        1,
        100,
        8,
    )

    expected = np.array(
        [
            0,
            0,
            0,
            0,
            10,
            20,
            30,
            40,
        ],
        dtype=np.int16,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_get_window_zero_length_returns_empty_pcm16() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    result = manager.get_window(
        1,
        0,
        0,
    )

    assert (
        result.size
        == 0
    )

    assert (
        result.dtype
        == np.int16
    )


def test_get_window_rejects_negative_start_sample() -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    with pytest.raises(
        ValueError
    ):

        manager.get_window(
            1,
            -1,
            16,
        )


@pytest.mark.parametrize(
    "fill_value",
    [
        -32769,
        32768,
    ],
)
def test_get_window_rejects_fill_value_outside_pcm16(
    fill_value: int,
) -> None:

    (
        manager,
        _,
    ) = (
        make_manager()
    )

    with pytest.raises(
        ValueError
    ):

        manager.get_window(
            1,
            0,
            16,
            fill_value=
                fill_value,
        )


def test_get_window_rejects_unregistered_node() -> None:

    config = (
        make_config()
    )

    manager = (
        StreamManager(
            config
        )
    )

    with pytest.raises(
        KeyError
    ):

        manager.get_window(
            1,
            0,
            16,
        )


# ======================================================================
# AUDIO BUFFER RESET
# ======================================================================


def test_reset_audio_buffers_clears_sample_timeline_state() -> None:
    """
    Regression test for session restart handling.

    sampleIndex restarts at zero on every acquisition START.

    Clearing only audio_blocks is not enough; old sample-timeline state
    must not make new sampleIndex=0 data look stale.
    """

    (
        manager,
        states,
    ) = (
        make_manager()
    )

    state = (
        states[
            1
        ]
    )

    manager.add_audio(
        make_block(
            1,
            100,
        )
    )

    state.last_sequence = (
        77
    )

    state.last_sample_index = (
        100
    )

    state.latest_sync = SyncPayload(
        session_id=
            TEST_SESSION_ID,

        sync_id=
            1,

        sample_index=
            100,

        local_micros=
            12345,
    )

    assert (
        state.expected_next_audio_sample
        is not None
    )

    manager.reset_audio_buffers()

    assert (
        len(
            state.audio_blocks
        )
        == 0
    )

    assert (
        state.expected_next_audio_sample
        is None
    )

    assert (
        state.last_sample_index
        is None
    )

    assert (
        state.latest_sync
        is None
    )

    # --------------------------------------------------------------
    # Session identity and network sequence diagnostics are deliberately
    # retained until the actual session-transition packet is observed.
    # --------------------------------------------------------------

    assert (
        state.session_id
        == TEST_SESSION_ID
    )

    assert (
        state.last_sequence
        == 77
    )


def test_reset_audio_buffers_affects_all_registered_nodes() -> None:

    (
        manager,
        states,
    ) = (
        make_manager()
    )

    for node_id in (
        1,
        2,
        3,
    ):

        manager.add_audio(
            make_block(
                node_id,
                0,
            )
        )

    manager.reset_audio_buffers()

    for state in (
        states.values()
    ):

        assert (
            len(
                state.audio_blocks
            )
            == 0
        )

        assert (
            state.expected_next_audio_sample
            is None
        )


# ======================================================================
# WAV RECORDER HELPERS
# ======================================================================


def read_pcm16_wav(
    path: Path,
) -> tuple[
    int,
    int,
    int,
    np.ndarray,
]:
    """
    Read one test WAV and return metadata + PCM16 samples.
    """

    with wave.open(
        str(
            path
        ),
        "rb",
    ) as wav:

        channels = (
            wav.getnchannels()
        )

        sample_width = (
            wav.getsampwidth()
        )

        sample_rate = (
            wav.getframerate()
        )

        frame_count = (
            wav.getnframes()
        )

        raw = wav.readframes(
            frame_count
        )

    samples = np.frombuffer(
        raw,
        dtype="<i2",
    ).copy()

    return (
        channels,
        sample_width,
        sample_rate,
        samples,
    )


# ======================================================================
# WAV RECORDER
# ======================================================================


def test_wav_recorder_start_and_stop(
    tmp_path: Path,
) -> None:

    config = (
        make_config()
    )

    recorder = WavRecorder(
        tmp_path,
        config,
    )

    assert not (
        recorder.is_recording
    )

    recorder.start(
        "session_test",
        (
            1,
            2,
            3,
        ),
    )

    assert (
        recorder.is_recording
    )

    for node_id in (
        1,
        2,
        3,
    ):

        assert (
            tmp_path
            / "session_test"
            / f"node_{node_id}.wav"
        ).exists()

    recorder.stop()

    assert not (
        recorder.is_recording
    )


def test_wav_recorder_metadata(
    tmp_path: Path,
) -> None:

    config = (
        make_config()
    )

    recorder = WavRecorder(
        tmp_path,
        config,
    )

    recorder.start(
        "metadata",
        (
            1,
        ),
    )

    recorder.write(
        make_block(
            1,
            0,
            samples=
                np.array(
                    [
                        1,
                        2,
                        3,
                        4,
                    ],
                    dtype=np.int16,
                ),
        )
    )

    recorder.stop()

    (
        channels,
        sample_width,
        sample_rate,
        samples,
    ) = read_pcm16_wav(
        tmp_path
        / "metadata"
        / "node_1.wav"
    )

    assert (
        channels
        == 1
    )

    assert (
        sample_width
        == 2
    )

    assert (
        sample_rate
        == 48_000
    )

    np.testing.assert_array_equal(
        samples,
        np.array(
            [
                1,
                2,
                3,
                4,
            ],
            dtype=np.int16,
        ),
    )


def test_wav_recorder_preserves_leading_gap_as_silence(
    tmp_path: Path,
) -> None:

    config = (
        make_config()
    )

    recorder = WavRecorder(
        tmp_path,
        config,
    )

    recorder.start(
        "gap_test",
        (
            1,
        ),
    )

    block_samples = np.array(
        [
            10,
            20,
            30,
            40,
        ],
        dtype=np.int16,
    )

    recorder.write(
        make_block(
            1,
            4,
            samples=
                block_samples,
        )
    )

    recorder.stop()

    (
        _,
        _,
        _,
        samples,
    ) = read_pcm16_wav(
        tmp_path
        / "gap_test"
        / "node_1.wav"
    )

    expected = np.array(
        [
            0,
            0,
            0,
            0,
            10,
            20,
            30,
            40,
        ],
        dtype=np.int16,
    )

    np.testing.assert_array_equal(
        samples,
        expected,
    )


def test_wav_recorder_skips_overlapping_prefix(
    tmp_path: Path,
) -> None:

    config = (
        make_config()
    )

    recorder = WavRecorder(
        tmp_path,
        config,
    )

    recorder.start(
        "overlap_test",
        (
            1,
        ),
    )

    first = np.array(
        [
            1,
            2,
            3,
            4,
        ],
        dtype=np.int16,
    )

    overlapping = np.array(
        [
            30,
            40,
            50,
            60,
        ],
        dtype=np.int16,
    )

    recorder.write(
        make_block(
            1,
            0,
            sequence=
                1,
            samples=
                first,
        )
    )

    # --------------------------------------------------------------
    # Starts at sample 2.
    #
    # Samples corresponding to timeline positions 2 and 3 have already
    # been written. Only 50 and 60 should extend the file.
    # --------------------------------------------------------------

    recorder.write(
        make_block(
            1,
            2,
            sequence=
                2,
            samples=
                overlapping,
        )
    )

    recorder.stop()

    (
        _,
        _,
        _,
        samples,
    ) = read_pcm16_wav(
        tmp_path
        / "overlap_test"
        / "node_1.wav"
    )

    expected = np.array(
        [
            1,
            2,
            3,
            4,
            50,
            60,
        ],
        dtype=np.int16,
    )

    np.testing.assert_array_equal(
        samples,
        expected,
    )


def test_wav_recorder_ignores_fully_duplicate_block(
    tmp_path: Path,
) -> None:

    config = (
        make_config()
    )

    recorder = WavRecorder(
        tmp_path,
        config,
    )

    recorder.start(
        "duplicate_test",
        (
            1,
        ),
    )

    original = np.array(
        [
            1,
            2,
            3,
            4,
        ],
        dtype=np.int16,
    )

    duplicate = np.array(
        [
            100,
            200,
            300,
            400,
        ],
        dtype=np.int16,
    )

    recorder.write(
        make_block(
            1,
            0,
            sequence=
                1,
            samples=
                original,
        )
    )

    recorder.write(
        make_block(
            1,
            0,
            sequence=
                2,
            samples=
                duplicate,
        )
    )

    recorder.stop()

    (
        _,
        _,
        _,
        samples,
    ) = read_pcm16_wav(
        tmp_path
        / "duplicate_test"
        / "node_1.wav"
    )

    np.testing.assert_array_equal(
        samples,
        original,
    )


@pytest.mark.parametrize(
    "session_label",
    [
        "",
        "   ",
        ".",
        "..",
        "../outside",
        "folder/session",
    ],
)
def test_wav_recorder_rejects_invalid_session_label(
    tmp_path: Path,
    session_label: str,
) -> None:

    recorder = WavRecorder(
        tmp_path,
        make_config(),
    )

    with pytest.raises(
        ValueError
    ):

        recorder.start(
            session_label,
            (
                1,
            ),
        )


def test_wav_recorder_rejects_duplicate_node_ids(
    tmp_path: Path,
) -> None:

    recorder = WavRecorder(
        tmp_path,
        make_config(),
    )

    with pytest.raises(
        ValueError
    ):

        recorder.start(
            "duplicate_nodes",
            (
                1,
                1,
            ),
        )