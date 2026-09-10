"""Local, deployment-specific configuration overrides.

The optional ``config.local.json`` file is intentionally ignored by Git so a
deployment can hold its LAN address, measured geometry, and tuning values
without exposing them in source control.
"""

from __future__ import annotations

import json
import os

from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from .config import AppConfig


DEFAULT_CONFIG_PATH = Path("config.local.json")


def deployment_config_path() -> Path:
    """Return the local override path, optionally selected by an environment variable."""

    configured_path = os.environ.get("WILDLIFE_CONFIG_PATH")
    return Path(configured_path) if configured_path else DEFAULT_CONFIG_PATH


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _only_known_keys(values: dict[str, Any], *, name: str, allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported {name} setting(s): {', '.join(unknown)}.")


def _node_positions(value: Any) -> dict[int, tuple[float, float]]:
    positions = _mapping(value, name="localization.node_positions")
    result: dict[int, tuple[float, float]] = {}
    for node_id, coordinates in positions.items():
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError("Each localization.node_positions value must be [x, y].")
        result[int(node_id)] = (float(coordinates[0]), float(coordinates[1]))
    return result


def load_app_config(default: AppConfig) -> AppConfig:
    """Apply supported local JSON overrides to a validated default configuration."""

    path = deployment_config_path()
    if not path.is_file():
        return default

    try:
        values = _mapping(json.loads(path.read_text(encoding="utf-8")), name="root")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid deployment configuration {path}: {exc.msg}.") from exc

    _only_known_keys(
        values,
        name="root",
        allowed={"network", "localization", "detection", "classification", "persistence"},
    )
    config = default

    if "network" in values:
        network = _mapping(values["network"], name="network")
        _only_known_keys(network, name="network", allowed={"host", "port"})
        config = replace(config, network=replace(config.network, **network))

    if "localization" in values:
        localization = _mapping(values["localization"], name="localization")
        _only_known_keys(
            localization,
            name="localization",
            allowed={
                "node_positions",
                "constrain_to_array_bounds",
                "bounds_margin_m",
                "use_environmental_speed",
                "min_peak_ratio",
            },
        )
        if "node_positions" in localization:
            localization["node_positions"] = _node_positions(localization["node_positions"])
        config = replace(config, localization=replace(config.localization, **localization))

    if "detection" in values:
        detection = _mapping(values["detection"], name="detection")
        _only_known_keys(
            detection,
            name="detection",
            allowed={"trigger_margin_db", "release_margin_db", "min_spectral_flux", "min_nodes"},
        )
        config = replace(config, detection=replace(config.detection, **detection))

    if "classification" in values:
        classification = _mapping(values["classification"], name="classification")
        _only_known_keys(
            classification,
            name="classification",
            allowed={"backend", "latitude", "longitude", "week", "use_geo_filter"},
        )
        config = replace(
            config,
            classification=replace(config.classification, **classification),
        )

    if "persistence" in values:
        persistence = _mapping(values["persistence"], name="persistence")
        _only_known_keys(
            persistence,
            name="persistence",
            allowed={"database_path", "events_dir"},
        )
        for key in ("database_path", "events_dir"):
            if key in persistence:
                persistence[key] = Path(persistence[key])
        config = replace(config, persistence=replace(config.persistence, **persistence))

    config.__post_init__()
    return config
