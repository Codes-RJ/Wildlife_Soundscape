import numpy as np

from config import AudioConfig
from models import AudioBlock
from node import NodeState
from stream_manager import StreamManager


def make_block(node_id: int, start: int) -> AudioBlock:
    return AudioBlock(
        node_id=node_id,
        sequence=1,
        session_id=1,
        sample_index=start,
        local_micros=0,
        i2s_error_count=0,
        flags=0,
        samples=np.arange(16, dtype=np.int16),
    )


def test_alignment_within_tolerance():
    cfg = AudioConfig(frames_per_block=16, sync_tolerance_samples=10)
    manager = StreamManager(cfg)
    for node_id in (1, 2, 3):
        state = NodeState(node_id, cfg)
        manager.register_state(state)
    manager.add_audio(make_block(1, 100))
    manager.add_audio(make_block(2, 103))
    manager.add_audio(make_block(3, 98))
    aligned = manager.latest_aligned_blocks()
    assert aligned is not None
    assert aligned.offsets == {1: -3, 2: 0, 3: -5}
