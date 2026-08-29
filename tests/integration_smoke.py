from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import tempfile
from dataclasses import replace

from config import CONFIG, AppConfig
from server import ReceiverServer
from simulator import FakeNode, SharedSimulation


async def main():
    network = replace(CONFIG.network, host='127.0.0.1', port=55001)
    audio = replace(CONFIG.audio, record_wav=False)
    cfg = AppConfig(network=network, audio=audio, expected_nodes=CONFIG.expected_nodes, recordings_dir=tempfile.mkdtemp(), print_status_every_s=5)
    server = ReceiverServer(cfg)
    await server.start()

    shared = SharedSimulation()
    nodes = [FakeNode(i, '127.0.0.1', 55001, shared) for i in (1,2,3)]
    tasks = [asyncio.create_task(n.run()) for n in nodes]
    try:
        await server.wait_for_nodes(timeout=3)
        await server.start_acquisition('smoke')
        await asyncio.sleep(0.25)
        assert all(server.connections[i].state.audio_packets_received > 0 for i in (1,2,3))
        aligned = None
        for _ in range(30):
            aligned = server.streams.latest_aligned_blocks()
            if aligned is not None:
                break
            await asyncio.sleep(0.02)
        assert aligned is not None, [server.connections[i].state.last_sample_index for i in (1,2,3)]
        print('SMOKE OK', {i: server.connections[i].state.audio_packets_received for i in (1,2,3)}, aligned.offsets)
        await server.stop_acquisition()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())
