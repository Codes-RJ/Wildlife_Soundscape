"""Read experiment provenance without migrating or modifying source databases."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Any


def read_manifests(
    path: Path, session_id: int | None = None
) -> dict[int, dict[str, Any]]:
    with closing(
        sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='session_manifests'"
        ).fetchone():
            return {}
        query = "SELECT session_id, manifest_json FROM session_manifests"
        args: tuple[int, ...] = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            args = (session_id,)
        return {row[0]: json.loads(row[1]) for row in conn.execute(query, args)}


def write_manifest_sidecar(output: Path, manifests: dict[int, dict[str, Any]]) -> Path:
    path = output.with_suffix(output.suffix + ".manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "sessions": manifests}, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    return path
