"""Immutable, JSON-compatible descriptions of acquisition settings."""

from dataclasses import asdict, fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
from typing import Any, cast

from .config import AppConfig


def config_from_manifest(manifest: dict[str, Any]) -> AppConfig:
    """Restore the recorded settings, rejecting unknown manifest versions."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported experiment manifest version")

    def restore(template: Any, value: Any, name: str = "") -> Any:
        if is_dataclass(template) and not isinstance(template, type):
            dataclass_type = cast(Any, type(template))
            return dataclass_type(
                **{
                    field.name: restore(
                        getattr(template, field.name), value[field.name], field.name
                    )
                    for field in fields(template)
                }
            )
        if isinstance(template, Path):
            return Path(value)
        if isinstance(template, frozenset):
            return frozenset(value)
        if isinstance(template, tuple):
            return tuple(tuple(v) if isinstance(v, list) else v for v in value)
        if isinstance(template, dict):
            if name in ("node_positions", "node_positions_m", "node_biases_s") or any(
                isinstance(k, int) for k in template
            ):
                return {
                    int(k): tuple(v) if isinstance(v, list) else v
                    for k, v in value.items()
                }
        return value

    return restore(AppConfig(), manifest["config"])


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [
            _json_value(v)
            for v in (sorted(value) if isinstance(value, (set, frozenset)) else value)
        ]
    return value


def experiment_manifest(config: AppConfig) -> dict[str, Any]:
    """Capture effective settings; unknown deployed firmware is explicitly unknown."""
    settings = _json_value(asdict(config))
    packages: dict[str, str | None] = {}
    for name in ("wildlife-soundscape", "numpy", "scipy", "librosa", "birdnet"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    commit = os.environ.get("WILDLIFE_SOFTWARE_COMMIT")
    dirty: bool | None = None
    root = Path(__file__).resolve().parents[3]
    if (root / ".git").exists():
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=root,
                    text=True,
                    timeout=2,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
    encoded = json.dumps(settings, sort_keys=True, allow_nan=False).encode()
    return {
        "schema_version": 1,
        "protocol_version": 4,
        "software_commit": commit,
        "working_tree_dirty": dirty,
        "python_version": platform.python_version(),
        "packages": packages,
        "firmware_build": os.environ.get("WILDLIFE_FIRMWARE_BUILD"),
        "model_identity": os.environ.get("WILDLIFE_MODEL_ID"),
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "config": settings,
    }
