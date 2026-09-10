import json

import pytest


def test_load_app_config_applies_local_overrides(tmp_path, monkeypatch):
    from wildlife_soundscape.core.config import AppConfig
    from wildlife_soundscape.core.deployment import load_app_config

    path = tmp_path / "deployment.json"
    path.write_text(
        json.dumps(
            {
                "network": {"host": "127.0.0.1", "port": 5002},
                "localization": {
                    "node_positions": {
                        "1": [0.0, 0.0],
                        "2": [0.0, 1.0],
                        "3": [1.0, 0.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WILDLIFE_CONFIG_PATH", str(path))

    config = load_app_config(AppConfig())

    assert config.network.host == "127.0.0.1"
    assert config.network.port == 5002
    assert config.localization.node_positions[2] == (0.0, 1.0)


def test_load_app_config_rejects_unknown_settings(tmp_path, monkeypatch):
    from wildlife_soundscape.core.config import AppConfig
    from wildlife_soundscape.core.deployment import load_app_config

    path = tmp_path / "deployment.json"
    path.write_text('{"network": {"unknown": true}}', encoding="utf-8")
    monkeypatch.setenv("WILDLIFE_CONFIG_PATH", str(path))

    with pytest.raises(ValueError, match="Unsupported network setting"):
        load_app_config(AppConfig())
