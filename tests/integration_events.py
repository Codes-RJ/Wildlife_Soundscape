from __future__ import annotations

import asyncio
import contextlib
import tempfile
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG, AppConfig, PersistenceConfig
from server import ReceiverServer
from simulator import FakeNode, SharedSimulation


async def run() -> None:
    root = Path(tempfile.mkdtemp())
    network = replace(CONFIG.network, host="127.0.0.1", port=55003)
    audio = replace(CONFIG.audio, record_wav=False)
    persistence = PersistenceConfig(
        database_path=root / "events.db",
        events_dir=root / "events",
        save_event_wav=True,
    )
    cfg = AppConfig(
        network=network,
        audio=audio,
        localization=CONFIG.localization,
        detection=CONFIG.detection,
        persistence=persistence,
        expected_nodes=CONFIG.expected_nodes,
        recordings_dir=root / "recordings",
        print_status_every_s=10,
    )
    server = ReceiverServer(cfg)
    await server.start()

    shared = SharedSimulation()
    node_tasks = [
        asyncio.create_task(FakeNode(i, "127.0.0.1", 55003, shared).run())
        for i in (1, 2, 3)
    ]
    try:
        await server.wait_for_nodes(timeout=5)
        await server.start_acquisition("integration_events")
        await asyncio.sleep(3.4)
        rows = server.events.database.recent_events(10)
        assert rows, "no events were detected/persisted"
        latest = rows[0]
        print(
            "EVENT OK",
            {
                "id": latest["id"],
                "start": latest["start_sample"],
                "end": latest["end_sample"],
                "x": latest["x_m"],
                "y": latest["y_m"],
                "temperature_c": latest["temperature_c"],
            },
        )
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.stop_acquisition(), timeout=2)
        for task in node_tasks:
            task.cancel()
        await asyncio.gather(*node_tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.close(), timeout=2)


if __name__ == "__main__":
    asyncio.run(run())
