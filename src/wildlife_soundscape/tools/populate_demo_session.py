"""Create an additive, explicitly synthetic dashboard dataset with array coverage.

Run: python -m wildlife_soundscape.tools.populate_demo_session --events 144
Every run creates a new session; existing sessions and recordings are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
import secrets
import wave
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wildlife_soundscape.analytics.indices import SoundscapeIndicesConfig, SoundscapeIndicesResult
from wildlife_soundscape.classification.classifier import AcousticClass, ClassificationResult
from wildlife_soundscape.core.config import CONFIG, AppConfig
from wildlife_soundscape.core.protocol import EnvironmentPayload
from wildlife_soundscape.core.provenance import experiment_manifest
from wildlife_soundscape.datasets.demo import demo_points
from wildlife_soundscape.dsp.features import extract_acoustic_features
from wildlife_soundscape.dsp.preprocessing import PreprocessingConfig, preprocess_event_audio
from wildlife_soundscape.localization.solver import PositionResult
from wildlife_soundscape.pipeline.event_detector import AcousticEvent
from wildlife_soundscape.storage.database import EventDatabase


def generate_demo_dataset(
    config: AppConfig = CONFIG, *, event_count: int = 144, seed: int = 20260909,
) -> dict:
    points = demo_points(config.localization.node_positions, count=event_count, seed=seed)
    db_path = Path(config.persistence.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = EventDatabase(db_path)
    # New IDs and directories prevent a second run from duplicating/replacing a session.
    session_id = secrets.randbelow(0xFFFFFFFF) + 1
    with db._connect() as conn:
        while conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone():
            session_id = secrets.randbelow(0xFFFFFFFF) + 1
    manifest = experiment_manifest(config)
    manifest.update(synthetic=True, dataset_generator="populate_demo_session",
                    generator_version="2.0", seed=seed, event_count=event_count,
                    label_source="illustrative broad-class fixtures, not model predictions",
                    localization_source="synthetic coordinates, not solver estimates")
    session_label = f"Synthetic soundscape · {event_count // 2} inside / {event_count // 2} outside"
    db.start_session(session_id, session_label, manifest=manifest)
    storage = Path(config.persistence.events_dir) / f"session_{session_id:08X}"
    storage.mkdir(parents=True, exist_ok=False)
    rate = config.audio.sample_rate
    rng = np.random.default_rng(seed)
    classes = list(AcousticClass)
    frequencies = (3200.0, 7800.0, 1400.0, 450.0, 6000.0, 2200.0)
    records = []
    for index, point in enumerate(points):
        label = classes[index % len(classes)]
        duration = 0.5 + 0.15 * (index % 4)
        count = int(duration * rate)
        time = np.arange(count) / rate
        frequency = min(frequencies[index % len(classes)], rate * 0.2)
        carrier = np.sin(2 * np.pi * (frequency * time + 150 * time**2))
        envelope = np.sin(np.linspace(0, np.pi, count)) ** 2
        source = 0.45 * carrier * envelope
        if label == AcousticClass.NOISE:
            source = 0.18 * rng.standard_normal(count) * envelope
        elif label == AcousticClass.UNKNOWN:
            source = 0.06 * carrier * envelope + 0.04 * rng.standard_normal(count)
        event_dir = storage / f"event_{index + 1:04d}"
        event_dir.mkdir()
        node_audio = {}
        for node_id, xy in sorted(config.localization.node_positions.items()):
            distance = math.hypot(point.x - xy[0], point.y - xy[1])
            delay = int(round(distance / 343.0 * rate))
            samples = np.pad(source, (delay, 0))[:count] / (1 + distance)
            samples += rng.normal(0, 0.001, count)
            pcm = np.round(np.clip(samples, -1.0, 1.0 - 1 / 32768) * 32768).astype("<i2")
            node_audio[node_id] = pcm
            with wave.open(str(event_dir / f"node_{node_id}.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm.tobytes())
        best_node = max(node_audio, key=lambda node: float(np.mean(node_audio[node].astype(float)**2)))
        features = extract_acoustic_features(preprocess_event_audio(
            node_audio[best_node], PreprocessingConfig(sample_rate=rate),
        ), noise_rms=0.001)
        start = int(index * 5 * rate)
        event = AcousticEvent(
            event_id=index + 1, session_id=session_id, start_sample=start,
            end_sample=start + count, trigger_nodes=tuple(sorted(node_audio)),
            peak_rms_dbfs=20 * math.log10(max(features.rms, 1e-12)),
        )
        position = PositionResult(
            x=point.x, y=point.y, success=True, residual_rms_seconds=0.0,
            residual_rms_meters=0.0, cost=0.0, nfev=0,
            message="Synthetic fixture location; no localization solver was run",
        )
        event_id = db.add_event(
            event=event,
            environment=EnvironmentPayload(
                temperature_c=21.5 + 2 * math.sin(index / 12),
                humidity_percent=62 + 5 * math.cos(index / 10), pressure_hpa=1013.2,
            ),
            localization=SimpleNamespace(
                position=position, tdoa_measurements=[],
                node_positions=dict(config.localization.node_positions), speed_of_sound_mps=343.0,
            ),
            event_directory=str(event_dir), best_node_id=best_node,
        )
        db.add_event_features(event_id=event_id, source_node_id=best_node, features=features)
        scores = {category.value: 0.02 for category in classes}
        confidence = 0.8 + (index % 5) * 0.03
        scores[label.value] = confidence
        second = AcousticClass.BIRD if label != AcousticClass.BIRD else AcousticClass.INSECT
        scores[second.value] = 0.12
        db.add_classification(event_id=event_id, result=ClassificationResult(
            label=label, confidence=confidence, second_label=second,
            second_confidence=0.12, margin=confidence - 0.12, scores=scores,
            reasons=("Synthetic illustrative class; not a model prediction or species observation.",
                     f"Synthetic source is {point.region} the configured microphone triangle."),
            classifier_name="SyntheticDemoGenerator", classifier_version="2.0",
        ))
        records.append({"event_id": event_id, "class": label.value, **asdict(point)})

    windows = max(1, math.ceil(event_count * 5 / 60))
    cfg = SoundscapeIndicesConfig()
    for window in range(windows):
        for node_id in sorted(config.localization.node_positions):
            phase = math.pi * window / max(windows, 1)
            temporal = 0.82 + 0.04 * math.sin(phase)
            spectral = 0.76 + 0.05 * math.cos(phase)
            bio = 0.045 + 0.03 * math.sin(phase)
            anthro = 0.015 + 0.002 * node_id
            result = SoundscapeIndicesResult(
                aci=14 + 8 * math.sin(phase) + node_id * 0.5,
                ndsi=(bio - anthro) / (bio + anthro),
                acoustic_entropy=temporal * spectral,
                temporal_entropy=temporal, spectral_entropy=spectral,
                bioacoustic_index=8 + 5 * math.sin(phase),
                anthrophony_power=anthro, biophony_power=bio, parameters=cfg,
            )
            db.add_soundscape_indices(
                session_id=session_id, node_id=node_id,
                start_sample=window * rate * 60, end_sample=(window + 1) * rate * 60,
                result=result,
            )
    db.stop_session(session_id)
    # The fixture represents acquisition time, not the few seconds needed to
    # insert it. Keep all event/metric windows inside a completed past session.
    with db._connect() as conn:
        conn.execute(
            "UPDATE sessions SET started_at=datetime(stopped_at, ?) WHERE session_id=?",
            (f"-{windows * 60} seconds", session_id),
        )
    report = {"session_id": session_id, "session_hex": f"{session_id:08X}",
              "synthetic": True, "seed": seed, "event_count": event_count,
              "regions": dict(Counter(point.region for point in points)),
              "classes": dict(Counter(record["class"] for record in records)),
              "node_positions": dict(config.localization.node_positions),
              "soundscape_windows": windows * len(config.localization.node_positions),
              "points": records}
    # Generated data artifact: kept beside the synthetic WAV evidence.
    (storage / "synthetic_points.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "points"}, indent=2))
    print(f"Point manifest: {storage / 'synthetic_points.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    generate_demo_dataset(event_count=args.events, seed=args.seed)


if __name__ == "__main__":
    main()
