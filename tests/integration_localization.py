from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import asyncio
import contextlib
import math

from config import CONFIG
from localization import LocalizationEngine
from server import ReceiverServer
from simulator import FakeNode, SharedSimulation


async def run() -> None:
    server = ReceiverServer(CONFIG)
    await server.start()
    shared = SharedSimulation()
    nodes = [FakeNode(i, "127.0.0.1", CONFIG.network.port, shared) for i in (1, 2, 3)]
    node_tasks = [asyncio.create_task(n.run()) for n in nodes]
    try:
        await server.wait_for_nodes(timeout=5)
        await server.start_acquisition("integration_localization")
        engine = LocalizationEngine(server.streams, CONFIG.localization)
        best = None
        for _ in range(40):
            await asyncio.sleep(0.10)
            result = engine.locate_latest()
            if result and result.success:
                err = math.hypot(result.position.x - shared.source_xy[0], result.position.y - shared.source_xy[1])
                if best is None or err < best[0]:
                    best = (err, result)
        assert best is not None, "no successful localization"
        err, result = best
        print(f"truth={shared.source_xy}, estimated=({result.position.x:.4f},{result.position.y:.4f}), error={err:.4f} m")
        for m in result.measurements:
            print(m)
        assert err < 0.15, f"localization error too high: {err}"
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.stop_acquisition(), timeout=2)
        for t in node_tasks:
            t.cancel()
        await asyncio.gather(*node_tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.close(), timeout=2)


if __name__ == "__main__":
    asyncio.run(run())
