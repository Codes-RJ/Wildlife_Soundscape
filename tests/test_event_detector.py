import numpy as np

from config import AudioConfig, EventDetectionConfig
from event_detector import MultiNodeEventDetector
from models import AudioBlock


def block(node, index, samples, session=7):
    return AudioBlock(
        node_id=node,
        sequence=index // len(samples),
        session_id=session,
        sample_index=index,
        local_micros=0,
        i2s_error_count=0,
        flags=0,
        samples=np.asarray(samples, dtype=np.int16),
    )


def test_multinode_event_detects_strong_burst():
    audio = AudioConfig(frames_per_block=256, buffer_seconds=5, sample_rate=48_000)
    cfg = EventDetectionConfig(
        noise_history_blocks=8,
        min_spectral_flux=0.0,
        trigger_margin_db=6.0,
        release_margin_db=3.0,
        strong_energy_margin_db=10.0,
        pre_pad_s=0.01,
        post_pad_s=0.01,
        min_event_ms=1.0,
        release_blocks=1,
    )
    det = MultiNodeEventDetector(audio, cfg)
    rng = np.random.default_rng(1)
    idx = 0
    completed = []

    # establish background
    for _ in range(10):
        for node in (1, 2, 3):
            completed += det.process(block(node, idx, rng.normal(0, 30, 256)))
        idx += 256

    # strong event across all nodes
    event = rng.normal(0, 5000, 256)
    for node in (1, 2, 3):
        completed += det.process(block(node, idx, event))
    idx += 256

    # enough quiet blocks to release + post-pad
    for _ in range(4):
        for node in (1, 2, 3):
            completed += det.process(block(node, idx, rng.normal(0, 30, 256)))
        idx += 256

    assert len(completed) == 1
    assert completed[0].trigger_nodes == (1, 2, 3)
    assert completed[0].end_sample > completed[0].start_sample
