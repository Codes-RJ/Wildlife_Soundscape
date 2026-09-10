"""Recompute old heuristic Unknown events from preserved WAV evidence.

Dry run by default. --apply first creates a SQLite backup, then updates derived
features/classifications only. Model predictions and raw recordings are retained.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import wave
from collections import Counter
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wildlife_soundscape.classification.classifier import HeuristicClassifier
from wildlife_soundscape.core.config import CONFIG, AppConfig
from wildlife_soundscape.dsp.features import FeatureConfig, extract_acoustic_features
from wildlife_soundscape.dsp.preprocessing import PreprocessingConfig, preprocess_event_audio
from wildlife_soundscape.storage.database import EventDatabase


def reprocess_unknown_events(
    config: AppConfig = CONFIG, *, apply: bool = False, limit: int | None = None,
) -> dict:
    path = Path(config.persistence.database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as source:
        source.row_factory = sqlite3.Row
        rows = source.execute("""
            SELECT e.id, e.event_directory, f.source_node_id, f.rms, f.snr_db,
                   m.manifest_json
            FROM events e JOIN classifications c ON c.event_id=e.id
            JOIN event_features f ON f.event_id=e.id
            LEFT JOIN session_manifests m ON m.session_id=e.session_id
            WHERE c.label='unknown' AND c.classifier_name='heuristic_acoustic_classifier'
            AND NOT EXISTS (SELECT 1 FROM event_processing p WHERE p.event_id=e.id
                            AND p.stage='feature_reprocess_v2' AND p.status='complete')
            ORDER BY e.id
            LIMIT ?
        """, (-1 if limit is None else limit,)).fetchall()
        backup = None
        if apply and rows:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = path.with_name(f"{path.stem}.before-feature-v2.{stamp}.db")
            with closing(sqlite3.connect(backup)) as target:
                source.backup(target)
            print(f"Database backup: {backup}", flush=True)

    database = EventDatabase(path) if apply and rows else None
    counts: Counter[str] = Counter()
    failures = []
    for index, row in enumerate(rows):
        try:
            audio_path = Path(row["event_directory"]) / f"node_{row['source_node_id']}.wav"
            with wave.open(str(audio_path), "rb") as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError("Expected mono PCM16 evidence")
                rate = wav.getframerate()
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            # Replay the session's DSP settings when they were recorded.
            manifest = json.loads(row["manifest_json"] or "{}")
            dsp = asdict(config.dsp)
            dsp.update(manifest.get("config", {}).get("dsp", {}))
            pre_keys = ("remove_dc", "bandpass_enabled", "low_cutoff_hz",
                        "high_cutoff_hz", "filter_order", "normalize_for_model", "model_target_peak")
            feature_keys = ("n_fft", "hop_length", "n_mfcc", "n_mels", "roll_percent",
                            "mfcc_fmin_hz", "mfcc_fmax_hz")
            audio = preprocess_event_audio(pcm, PreprocessingConfig(
                sample_rate=rate, **{key: dsp[key] for key in pre_keys},
            ))
            # Retain the independently estimated pre-trigger noise floor.
            noise_rms = None if row["snr_db"] is None else row["rms"] / 10 ** (row["snr_db"] / 20)
            features = extract_acoustic_features(audio, noise_rms=noise_rms, config=FeatureConfig(
                sample_rate=rate, **{key: dsp[key] for key in feature_keys},
            ))
            result = HeuristicClassifier().classify(features)
            result = replace(result, reasons=result.reasons + (
                "Reprocessed preserved PCM with energy-weighted spectral features v2.",
            ))
            if database is not None:
                database.add_event_features(event_id=row["id"], source_node_id=row["source_node_id"], features=features)
                database.add_classification(event_id=row["id"], result=result)
                database.set_event_processing_status(row["id"], "feature_reprocess_v2", "complete", str(backup))
            counts[result.label.value] += 1
        except (OSError, ValueError, TypeError, KeyError, wave.Error) as exc:
            failures.append({"event_id": row["id"], "error": str(exc)})
        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(rows)}: {dict(counts)}", flush=True)
    report = {"apply": apply, "selected": len(rows), "labels": dict(counts),
              "failures": failures, "backup": str(backup) if backup else None}
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up the database and update derived results")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = reprocess_unknown_events(apply=args.apply, limit=args.limit)
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
