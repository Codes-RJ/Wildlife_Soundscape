from dataclasses import asdict, replace
import json

from wildlife_soundscape.core.config import AppConfig
from wildlife_soundscape.core.provenance import (
    experiment_manifest,
    config_from_manifest,
)
from wildlife_soundscape.storage.database import EventDatabase
from wildlife_soundscape.storage.provenance import (
    read_manifests,
    write_manifest_sidecar,
)


def test_manifest_restores_settings_and_exports_without_mutation(tmp_path):
    config = AppConfig()
    config = replace(
        config,
        localization=replace(
            config.localization,
            tdoa_calibration=replace(
                config.localization.tdoa_calibration, node_biases_s={1: 0.0, 2: 0.00001}
            ),
        ),
    )
    manifest = experiment_manifest(config)
    restored = config_from_manifest(json.loads(json.dumps(manifest)))
    assert asdict(restored) == asdict(config)
    database = EventDatabase(tmp_path / "events.db")
    database.start_session(42, "test", manifest=manifest)
    manifests = read_manifests(database.path)
    assert manifests[42] == manifest
    output = write_manifest_sidecar(tmp_path / "events.csv", manifests)
    assert json.loads(output.read_text())["sessions"]["42"] == manifest
