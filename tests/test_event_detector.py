"""
Tests for event_detector.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Coverage
--------
These tests verify:

    1. adaptive background establishment
    2. strong multi-node acoustic-event detection
    3. configurable N-of-M node agreement
    4. rejection of single-node activity
    5. sampleIndex association within synchronization tolerance
    6. rejection of groups outside synchronization tolerance
    7. attack hysteresis
    8. release hysteresis
    9. session isolation
    10. duplicate / overlapping block rejection
    11. detector reset behaviour
    12. monotonically increasing event identifiers
    13. AcousticEvent metadata
    14. disabled-detector behaviour
    15. constructor consistency checks

Important
---------
The detector performs event detection and multi-node association.

It does NOT estimate acoustic propagation delay.

Small start-sample differences within sync_tolerance_samples are only
used to associate blocks representing the same coarse shared-clock
interval.

Fine acoustic arrival-time differences remain in the PCM waveform for
later GCC-PHAT/TDOA localization.
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


from wildlife_soundscape.core.config import (
    AudioConfig,
    EventDetectionConfig,
)

from wildlife_soundscape.pipeline.event_detector import (
    AcousticEvent,
    MultiNodeEventDetector,
)

from wildlife_soundscape.core.models import (
    AudioBlock,
)


# ======================================================================
# CONSTANTS
# ======================================================================


SAMPLE_RATE = 48_000


FRAMES_PER_BLOCK = 256


SESSION_ID = 7


SECOND_SESSION_ID = 8


NODE_IDS = (
    1,
    2,
    3,
)


# ======================================================================
# TEST HELPERS
# ======================================================================


def make_audio_config(
    *,
    sync_tolerance_samples: int = 10,
) -> AudioConfig:
    """
    Small deterministic audio configuration for event-detector tests.
    """

    return AudioConfig(
        sample_rate=SAMPLE_RATE,
        frames_per_block=FRAMES_PER_BLOCK,
        buffer_seconds=5,
        sync_tolerance_samples=sync_tolerance_samples,
    )


def make_detection_config(
    *,
    enabled: bool = True,
    attack_blocks: int = 1,
    release_blocks: int = 1,
    min_nodes: int = 2,
    min_event_ms: float = 1.0,
    pre_pad_s: float = 0.01,
    post_pad_s: float = 0.01,
) -> EventDetectionConfig:
    """
    Deterministic detector configuration.

    Spectral flux is set to zero so these unit tests primarily exercise
    the detector state machine and adaptive energy logic rather than
    depending on a particular spectral pattern.
    """

    return EventDetectionConfig(
        enabled=enabled,
        noise_history_blocks=8,
        noise_quantile=0.30,
        initial_noise_dbfs=-52.0,
        trigger_margin_db=6.0,
        release_margin_db=3.0,
        min_spectral_flux=0.0,
        strong_energy_margin_db=10.0,
        attack_blocks=attack_blocks,
        release_blocks=release_blocks,
        min_nodes=min_nodes,
        min_event_ms=min_event_ms,
        max_event_s=12.0,
        pre_pad_s=pre_pad_s,
        post_pad_s=post_pad_s,
    )


def make_detector(
    *,
    sync_tolerance_samples: int = 10,
    enabled: bool = True,
    attack_blocks: int = 1,
    release_blocks: int = 1,
    min_nodes: int = 2,
    min_event_ms: float = 1.0,
    pre_pad_s: float = 0.01,
    post_pad_s: float = 0.01,
) -> MultiNodeEventDetector:
    """
    Build the standard three-node detector.
    """

    audio = make_audio_config(sync_tolerance_samples=sync_tolerance_samples)

    detection = make_detection_config(
        enabled=enabled,
        attack_blocks=attack_blocks,
        release_blocks=release_blocks,
        min_nodes=min_nodes,
        min_event_ms=min_event_ms,
        pre_pad_s=pre_pad_s,
        post_pad_s=post_pad_s,
    )

    return MultiNodeEventDetector(
        audio,
        detection,
        NODE_IDS,
    )


def make_block(
    node_id: int,
    sample_index: int,
    samples: np.ndarray,
    *,
    session_id: int = SESSION_ID,
    sequence: int | None = None,
) -> AudioBlock:
    """
    Construct one detector input block.
    """

    pcm = np.ascontiguousarray(
        np.asarray(
            samples,
            dtype=np.int16,
        )
    )

    if pcm.ndim != 1:
        raise ValueError("test block samples must be 1-D")

    if sequence is None:
        sequence = sample_index // max(
            1,
            pcm.size,
        )

    return AudioBlock(
        node_id=node_id,
        sequence=int(sequence),
        session_id=session_id,
        sample_index=sample_index,
        local_micros=0,
        i2s_error_count=0,
        flags=0,
        samples=pcm,
    )


def quiet_block(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Low-level background noise.
    """

    values = rng.normal(
        0.0,
        30.0,
        FRAMES_PER_BLOCK,
    )

    return np.ascontiguousarray(
        np.clip(
            values,
            -32768.0,
            32767.0,
        ).astype(np.int16)
    )


def strong_block(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    High-energy broadband activity.
    """

    values = rng.normal(
        0.0,
        5000.0,
        FRAMES_PER_BLOCK,
    )

    return np.ascontiguousarray(
        np.clip(
            values,
            -32768.0,
            32767.0,
        ).astype(np.int16)
    )


def feed_group(
    detector: MultiNodeEventDetector,
    *,
    base_index: int,
    samples_by_node: dict[
        int,
        np.ndarray,
    ],
    offsets: dict[
        int,
        int,
    ]
    | None = None,
    session_id: int = SESSION_ID,
) -> list[AcousticEvent]:
    """
    Feed one logical three-node block group.

    offsets allows each node to have a slightly different sampleIndex,
    exercising the detector's sync-tolerance grouping.
    """

    if offsets is None:
        offsets = {node_id: 0 for node_id in NODE_IDS}

    completed: list[AcousticEvent] = []

    for node_id in NODE_IDS:
        completed.extend(
            detector.process(
                make_block(
                    node_id,
                    base_index + offsets[node_id],
                    samples_by_node[node_id],
                    session_id=session_id,
                )
            )
        )

    return completed


def feed_background(
    detector: MultiNodeEventDetector,
    rng: np.random.Generator,
    *,
    start_index: int = 0,
    groups: int = 10,
    offsets: dict[
        int,
        int,
    ]
    | None = None,
    session_id: int = SESSION_ID,
) -> int:
    """
    Establish the adaptive background model.

    Returns the next base sampleIndex.
    """

    base_index = start_index

    for _ in range(groups):
        samples = {node_id: quiet_block(rng) for node_id in NODE_IDS}

        result = feed_group(
            detector,
            base_index=base_index,
            samples_by_node=samples,
            offsets=offsets,
            session_id=session_id,
        )

        assert result == []

        base_index += FRAMES_PER_BLOCK

    return base_index


def release_event(
    detector: MultiNodeEventDetector,
    rng: np.random.Generator,
    *,
    start_index: int,
    groups: int = 5,
    offsets: dict[
        int,
        int,
    ]
    | None = None,
    session_id: int = SESSION_ID,
) -> tuple[
    int,
    list[AcousticEvent],
]:
    """
    Feed enough quiet groups to release and post-pad an active event.
    """

    base_index = start_index

    completed: list[AcousticEvent] = []

    for _ in range(groups):
        samples = {node_id: quiet_block(rng) for node_id in NODE_IDS}

        completed.extend(
            feed_group(
                detector,
                base_index=base_index,
                samples_by_node=samples,
                offsets=offsets,
                session_id=session_id,
            )
        )

        base_index += FRAMES_PER_BLOCK

    return (
        base_index,
        completed,
    )


# ======================================================================
# BASIC MULTI-NODE EVENT
# ======================================================================


def test_multinode_event_detects_strong_burst() -> None:

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(1)

    index = feed_background(
        detector,
        rng,
    )

    # ==============================================================
    # STRONG EVENT ACROSS ALL THREE NODES
    # ==============================================================

    event_signal = strong_block(rng)

    completed = feed_group(
        detector,
        base_index=index,
        samples_by_node={
            1: event_signal,
            2: event_signal.copy(),
            3: event_signal.copy(),
        },
    )

    # Event should have started but not yet completed.
    assert completed == []

    event_index = index

    index += FRAMES_PER_BLOCK

    # ==============================================================
    # RELEASE + POST-PAD
    # ==============================================================

    (
        _,
        completed,
    ) = release_event(
        detector,
        rng,
        start_index=index,
    )

    assert len(completed) == 1

    event = completed[0]

    assert event.trigger_nodes == (
        1,
        2,
        3,
    )

    assert event.session_id == SESSION_ID

    assert event.start_sample <= event_index

    assert event.end_sample > event.start_sample

    assert event.sample_count == (event.end_sample - event.start_sample)

    assert np.isfinite(event.peak_rms_dbfs)


# ======================================================================
# N-OF-M AGREEMENT
# ======================================================================


def test_two_of_three_nodes_can_trigger_when_min_nodes_is_two() -> None:

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(2)

    index = feed_background(
        detector,
        rng,
    )

    strong = strong_block(rng)

    completed = feed_group(
        detector,
        base_index=index,
        samples_by_node={
            1: strong,
            2: strong.copy(),
            # Third microphone observes only background.
            3: quiet_block(rng),
        },
    )

    assert completed == []

    index += FRAMES_PER_BLOCK

    (
        _,
        completed,
    ) = release_event(
        detector,
        rng,
        start_index=index,
    )

    assert len(completed) == 1

    assert completed[0].trigger_nodes == (
        1,
        2,
    )


def test_single_node_activity_does_not_trigger_two_node_detector() -> None:

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(3)

    index = feed_background(
        detector,
        rng,
    )

    completed: list[AcousticEvent] = []

    # Several strong blocks on Node 1 alone should still fail the
    # multi-node agreement requirement.
    for _ in range(3):
        completed.extend(
            feed_group(
                detector,
                base_index=index,
                samples_by_node={
                    1: strong_block(rng),
                    2: quiet_block(rng),
                    3: quiet_block(rng),
                },
            )
        )

        index += FRAMES_PER_BLOCK

    (
        _,
        released,
    ) = release_event(
        detector,
        rng,
        start_index=index,
    )

    completed.extend(released)

    assert completed == []


def test_three_node_requirement_rejects_two_node_burst() -> None:

    detector = make_detector(min_nodes=3)

    rng = np.random.default_rng(4)

    index = feed_background(
        detector,
        rng,
    )

    strong = strong_block(rng)

    completed = feed_group(
        detector,
        base_index=index,
        samples_by_node={
            1: strong,
            2: strong.copy(),
            3: quiet_block(rng),
        },
    )

    index += FRAMES_PER_BLOCK

    (
        _,
        released,
    ) = release_event(
        detector,
        rng,
        start_index=index,
    )

    completed.extend(released)

    assert completed == []


# ======================================================================
# SAMPLE-INDEX ASSOCIATION TOLERANCE
# ======================================================================


def test_multinode_blocks_within_sync_tolerance_form_one_event() -> None:
    """
    Regression test for tolerant coarse block association.

    Node block starts differ by:

        Node 1: +0
        Node 2: +3
        Node 3: -2

    Total spread = 5 samples.

    With sync_tolerance_samples=10 these belong to one coarse logical
    block group.
    """

    detector = make_detector(
        sync_tolerance_samples=10,
        min_nodes=2,
    )

    rng = np.random.default_rng(5)

    offsets = {
        1: 0,
        2: 3,
        3: -2,
    }

    # Start sufficiently above zero so the negative Node-3 offset remains
    # a valid non-negative absolute sampleIndex.
    index = feed_background(
        detector,
        rng,
        start_index=10_000,
        offsets=offsets,
    )

    strong = strong_block(rng)

    completed = feed_group(
        detector,
        base_index=index,
        offsets=offsets,
        samples_by_node={
            1: strong,
            2: strong.copy(),
            3: strong.copy(),
        },
    )

    assert completed == []

    index += FRAMES_PER_BLOCK

    (
        _,
        completed,
    ) = release_event(
        detector,
        rng,
        start_index=index,
        offsets=offsets,
    )

    assert len(completed) == 1

    assert completed[0].trigger_nodes == (
        1,
        2,
        3,
    )


def test_blocks_outside_sync_tolerance_do_not_form_one_group() -> None:
    """
    A start-index spread larger than the configured tolerance must not
    be treated as one synchronized detector decision.
    """

    detector = make_detector(
        sync_tolerance_samples=10,
        min_nodes=3,
    )

    rng = np.random.default_rng(6)

    offsets = {
        1: 0,
        2: 11,
        3: 0,
    }

    index = feed_background(
        detector,
        rng,
        start_index=10_000,
        offsets=offsets,
    )

    strong = strong_block(rng)

    completed = feed_group(
        detector,
        base_index=index,
        offsets=offsets,
        samples_by_node={
            1: strong,
            2: strong.copy(),
            3: strong.copy(),
        },
    )

    index += FRAMES_PER_BLOCK

    (
        _,
        released,
    ) = release_event(
        detector,
        rng,
        start_index=index,
        offsets=offsets,
    )

    completed.extend(released)

    assert completed == []


# ======================================================================
# ATTACK HYSTERESIS
# ======================================================================


def test_attack_requires_requested_number_of_active_blocks() -> None:
    """
    With attack_blocks=2, one isolated strong block must not establish
    a complete event.

    Two consecutive active groups should.
    """

    detector = make_detector(
        attack_blocks=2,
        release_blocks=1,
        min_nodes=2,
    )

    rng = np.random.default_rng(7)

    index = feed_background(
        detector,
        rng,
    )

    # ==============================================================
    # ONE ISOLATED STRONG GROUP
    # ==============================================================

    strong = strong_block(rng)

    completed = feed_group(
        detector,
        base_index=index,
        samples_by_node={
            1: strong,
            2: strong.copy(),
            3: strong.copy(),
        },
    )

    assert completed == []

    index += FRAMES_PER_BLOCK

    # Break the attack sequence.
    completed.extend(
        feed_group(
            detector,
            base_index=index,
            samples_by_node={node_id: quiet_block(rng) for node_id in NODE_IDS},
        )
    )

    assert completed == []

    index += FRAMES_PER_BLOCK

    # ==============================================================
    # TWO CONSECUTIVE STRONG GROUPS
    # ==============================================================

    for _ in range(2):
        strong = strong_block(rng)

        completed.extend(
            feed_group(
                detector,
                base_index=index,
                samples_by_node={
                    1: strong,
                    2: strong.copy(),
                    3: strong.copy(),
                },
            )
        )

        index += FRAMES_PER_BLOCK

    assert completed == []

    (
        _,
        released,
    ) = release_event(
        detector,
        rng,
        start_index=index,
    )

    completed.extend(released)

    assert len(completed) == 1


# ======================================================================
# RELEASE HYSTERESIS
# ======================================================================


def test_release_requires_configured_number_of_quiet_blocks() -> None:

    detector = make_detector(
        attack_blocks=1,
        release_blocks=2,
        min_nodes=2,
        # Remove post-padding from this state-machine-focused test.
        post_pad_s=0.0,
    )

    rng = np.random.default_rng(8)

    index = feed_background(
        detector,
        rng,
    )

    strong = strong_block(rng)

    assert (
        feed_group(
            detector,
            base_index=index,
            samples_by_node={
                1: strong,
                2: strong.copy(),
                3: strong.copy(),
            },
        )
        == []
    )

    index += FRAMES_PER_BLOCK

    # First quiet group: event must remain open.
    first_quiet = feed_group(
        detector,
        base_index=index,
        samples_by_node={node_id: quiet_block(rng) for node_id in NODE_IDS},
    )

    assert first_quiet == []

    index += FRAMES_PER_BLOCK

    # Second quiet group satisfies release_blocks=2.
    second_quiet = feed_group(
        detector,
        base_index=index,
        samples_by_node={node_id: quiet_block(rng) for node_id in NODE_IDS},
    )

    assert len(second_quiet) == 1


# ======================================================================
# SESSION ISOLATION
# ======================================================================


def test_session_change_discards_incomplete_previous_event() -> None:
    """
    An event started in one acquisition session must never be completed
    using blocks from another session.

    sampleIndex values restart from zero on each START.
    """

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(9)

    index = feed_background(
        detector,
        rng,
        session_id=SESSION_ID,
    )

    strong = strong_block(rng)

    # Start an event in session 7.
    assert (
        feed_group(
            detector,
            base_index=index,
            samples_by_node={
                1: strong,
                2: strong.copy(),
                3: strong.copy(),
            },
            session_id=SESSION_ID,
        )
        == []
    )

    # ==============================================================
    # NEW ACQUISITION SESSION
    # ==============================================================

    new_index = feed_background(
        detector,
        rng,
        start_index=0,
        groups=10,
        session_id=SECOND_SESSION_ID,
    )

    # Nothing from session 7 should have been completed by session-8
    # background blocks.

    strong = strong_block(rng)

    assert (
        feed_group(
            detector,
            base_index=new_index,
            samples_by_node={
                1: strong,
                2: strong.copy(),
                3: strong.copy(),
            },
            session_id=SECOND_SESSION_ID,
        )
        == []
    )

    new_index += FRAMES_PER_BLOCK

    (
        _,
        completed,
    ) = release_event(
        detector,
        rng,
        start_index=new_index,
        session_id=SECOND_SESSION_ID,
    )

    assert len(completed) == 1

    assert completed[0].session_id == SECOND_SESSION_ID


# ======================================================================
# DUPLICATE / OVERLAPPING INPUT
# ======================================================================


def test_duplicate_node_block_does_not_create_duplicate_event() -> None:

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(10)

    index = feed_background(
        detector,
        rng,
    )

    strong = strong_block(rng)

    node_1_block = make_block(
        1,
        index,
        strong,
    )

    # First copy is accepted.
    assert detector.process(node_1_block) == []

    # Exact same timeline block is stale/duplicate and must not become a
    # second detector vote.
    assert detector.process(node_1_block) == []

    # Complete the real three-node group.
    assert (
        detector.process(
            make_block(
                2,
                index,
                strong.copy(),
            )
        )
        == []
    )

    assert (
        detector.process(
            make_block(
                3,
                index,
                strong.copy(),
            )
        )
        == []
    )

    index += FRAMES_PER_BLOCK

    (
        _,
        completed,
    ) = release_event(
        detector,
        rng,
        start_index=index,
    )

    assert len(completed) == 1


def test_overlapping_old_block_is_ignored_without_breaking_future_detection() -> None:

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(11)

    index = feed_background(
        detector,
        rng,
    )

    # Feed one normal quiet group.
    quiet = {node_id: quiet_block(rng) for node_id in NODE_IDS}

    assert (
        feed_group(
            detector,
            base_index=index,
            samples_by_node=quiet,
        )
        == []
    )

    # Node 1 now receives an overlapping block beginning halfway inside
    # its already-consumed previous block.
    overlapping_start = index + FRAMES_PER_BLOCK // 2

    assert (
        detector.process(
            make_block(
                1,
                overlapping_start,
                strong_block(rng),
            )
        )
        == []
    )

    # Continue at the correct next absolute block boundary.
    index += FRAMES_PER_BLOCK

    strong = strong_block(rng)

    assert (
        feed_group(
            detector,
            base_index=index,
            samples_by_node={
                1: strong,
                2: strong.copy(),
                3: strong.copy(),
            },
        )
        == []
    )

    index += FRAMES_PER_BLOCK

    (
        _,
        completed,
    ) = release_event(
        detector,
        rng,
        start_index=index,
    )

    assert len(completed) == 1


# ======================================================================
# RESET
# ======================================================================


def test_reset_discards_active_detector_state() -> None:

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(12)

    index = feed_background(
        detector,
        rng,
    )

    strong = strong_block(rng)

    assert (
        feed_group(
            detector,
            base_index=index,
            samples_by_node={
                1: strong,
                2: strong.copy(),
                3: strong.copy(),
            },
        )
        == []
    )

    detector.reset()

    # Quiet blocks after reset must not finish the event that was active
    # before reset.
    completed: list[AcousticEvent] = []

    index = 0

    for _ in range(5):
        completed.extend(
            feed_group(
                detector,
                base_index=index,
                samples_by_node={node_id: quiet_block(rng) for node_id in NODE_IDS},
                session_id=SECOND_SESSION_ID,
            )
        )

        index += FRAMES_PER_BLOCK

    assert completed == []


def test_event_ids_continue_increasing_after_reset() -> None:
    """
    Detector reset clears session/state but should not reuse event IDs
    inside the same process.
    """

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(13)

    # ==============================================================
    # FIRST EVENT
    # ==============================================================

    index = feed_background(
        detector,
        rng,
        session_id=SESSION_ID,
    )

    strong = strong_block(rng)

    feed_group(
        detector,
        base_index=index,
        samples_by_node={
            1: strong,
            2: strong.copy(),
            3: strong.copy(),
        },
        session_id=SESSION_ID,
    )

    index += FRAMES_PER_BLOCK

    (
        _,
        first_completed,
    ) = release_event(
        detector,
        rng,
        start_index=index,
        session_id=SESSION_ID,
    )

    assert len(first_completed) == 1

    first_id = first_completed[0].event_id

    # ==============================================================
    # RESET
    # ==============================================================

    detector.reset()

    # ==============================================================
    # SECOND EVENT
    # ==============================================================

    index = feed_background(
        detector,
        rng,
        start_index=0,
        session_id=SECOND_SESSION_ID,
    )

    strong = strong_block(rng)

    feed_group(
        detector,
        base_index=index,
        samples_by_node={
            1: strong,
            2: strong.copy(),
            3: strong.copy(),
        },
        session_id=SECOND_SESSION_ID,
    )

    index += FRAMES_PER_BLOCK

    (
        _,
        second_completed,
    ) = release_event(
        detector,
        rng,
        start_index=index,
        session_id=SECOND_SESSION_ID,
    )

    assert len(second_completed) == 1

    second_id = second_completed[0].event_id

    assert second_id > first_id


# ======================================================================
# DETECTION DISABLED
# ======================================================================


def test_disabled_detector_never_emits_event() -> None:

    detector = make_detector(
        enabled=False,
        min_nodes=2,
    )

    rng = np.random.default_rng(14)

    completed: list[AcousticEvent] = []

    index = 0

    # Deliberately feed sustained high-energy audio.
    for _ in range(10):
        strong = strong_block(rng)

        completed.extend(
            feed_group(
                detector,
                base_index=index,
                samples_by_node={
                    1: strong,
                    2: strong.copy(),
                    3: strong.copy(),
                },
            )
        )

        index += FRAMES_PER_BLOCK

    assert completed == []


# ======================================================================
# EVENT METADATA
# ======================================================================


def test_completed_event_contains_consistent_metadata() -> None:

    detector = make_detector(min_nodes=2)

    rng = np.random.default_rng(15)

    index = feed_background(
        detector,
        rng,
    )

    trigger_index = index

    strong = strong_block(rng)

    feed_group(
        detector,
        base_index=index,
        samples_by_node={
            1: strong,
            2: strong.copy(),
            3: quiet_block(rng),
        },
    )

    index += FRAMES_PER_BLOCK

    (
        _,
        completed,
    ) = release_event(
        detector,
        rng,
        start_index=index,
    )

    assert len(completed) == 1

    event = completed[0]

    assert event.event_id > 0

    assert event.session_id == SESSION_ID

    assert event.trigger_nodes == (
        1,
        2,
    )

    assert event.start_sample >= 0

    assert event.start_sample <= trigger_index

    assert event.end_sample > trigger_index

    assert event.sample_count > 0

    assert event.sample_count == (event.end_sample - event.start_sample)

    assert np.isfinite(event.peak_rms_dbfs)


# ======================================================================
# CONSTRUCTION VALIDATION
# ======================================================================


def test_detector_rejects_sync_tolerance_equal_to_block_size() -> None:
    """
    Coarse association tolerance must remain smaller than a complete
    audio block; otherwise adjacent blocks could become ambiguous.
    """

    audio = AudioConfig(
        sample_rate=SAMPLE_RATE,
        frames_per_block=16,
        buffer_seconds=5,
        sync_tolerance_samples=16,
    )

    config = make_detection_config()

    with pytest.raises(ValueError):
        MultiNodeEventDetector(
            audio,
            config,
            NODE_IDS,
        )


def test_detector_rejects_min_nodes_larger_than_configured_node_count() -> None:

    audio = make_audio_config()

    config = make_detection_config(min_nodes=4)

    with pytest.raises(ValueError):
        MultiNodeEventDetector(
            audio,
            config,
            NODE_IDS,
        )
