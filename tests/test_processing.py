import asyncio
import threading

import pytest

from wildlife_soundscape.runtime.processing import OrderedProcessor


@pytest.mark.asyncio
async def test_slow_processing_does_not_block_loop_and_overload_is_visible():
    processor = OrderedProcessor(2)
    entered = threading.Event()
    release = threading.Event()
    result = []

    def slow():
        entered.set()
        assert release.wait(3)
        result.append(1)

    processor.submit(slow)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert processor.submit(lambda: result.append(2))
        assert processor.submit(lambda: result.append(3))
        assert not processor.submit(lambda: result.append(4), samples=1024)
        # This callback must execute while the DSP worker remains blocked.
        await asyncio.wait_for(asyncio.sleep(0), 0.5)
    finally:
        release.set()
        await processor.close()
    assert result == [1, 2, 3]
    assert processor.metrics.dropped_samples == 1024
    assert processor.metrics.completed == 3
    assert processor.snapshot()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_failed_item_does_not_prevent_drain_or_next_item():
    processor = OrderedProcessor()
    result = []

    def fail():
        raise OSError("storage offline")

    processor.submit(fail)
    processor.submit(lambda: result.append("saved"))
    await processor.drain()
    assert result == ["saved"]
    assert processor.metrics.failed == 1
    assert processor.metrics.completed == 1


@pytest.mark.asyncio
async def test_cancelled_drain_finishes_inflight_work_before_returning():
    processor = OrderedProcessor()
    release = threading.Event()
    entered = threading.Event()

    def work():
        entered.set()
        assert release.wait(3)

    processor.submit(work)
    assert await asyncio.to_thread(entered.wait, 3)
    draining = asyncio.create_task(processor.drain())
    await asyncio.sleep(0)
    draining.cancel()
    await asyncio.sleep(0)
    assert not draining.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await draining
    assert processor.metrics.completed == 1
