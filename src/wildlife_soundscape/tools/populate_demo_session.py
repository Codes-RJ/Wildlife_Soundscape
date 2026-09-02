"""
Populate a rich, scientifically plausible demonstration acquisition session.

This allows full interactive exploration of the Streamlit dashboard:
- Live Monitor
- Research Analysis (Temporal Activity, Environmental Correlations, Spatial Occupancy, Behavior Indicators)
- Continuous Ecoacoustic Soundscape Metrics (ACI, NDSI, Acoustic Entropy H, Bioacoustic Index)
- Audio Inspector (Waveform & Spectrogram preview)
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from wildlife_soundscape.analytics.indices import SoundscapeIndicesConfig, SoundscapeIndicesResult
from wildlife_soundscape.core.config import CONFIG
from wildlife_soundscape.storage.database import EventDatabase
from wildlife_soundscape.pipeline.event_detector import AcousticEvent
from wildlife_soundscape.localization.solver import PositionResult
from wildlife_soundscape.core.protocol import EnvironmentPayload


def generate_demo_dataset() -> None:
    db_path = Path(CONFIG.persistence.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = EventDatabase(db_path)

    session_id = 0x2026A1
    session_label = "Temperate Forest Canopy Bioacoustic Survey"
    db.start_session(session_id, session_label)

    sample_rate = CONFIG.audio.sample_rate  # 48000
    event_storage_dir = Path(CONFIG.persistence.events_dir) / f"session_{session_id:08X}"
    event_storage_dir.mkdir(parents=True, exist_ok=True)

    print(f"Populating demo session {session_id:08X} in {db_path}...")

    # Node positions (1-meter equilateral triangle)
    node_positions = {
        1: (-0.5, -0.288675),
        2: (0.5, -0.288675),
        3: (0.0, 0.577350),
    }

    # Species and soundscape events
    species_events = [
        ("BIRD", "Melospiza melodia (Song Sparrow)", 0.94, "Passerella iliaca", 0.12, 3200.0, 18.5, 4.2, 3.8),
        ("BIRD", "Hylocichla mustelina (Wood Thrush)", 0.89, "Catharus guttatus", 0.15, 2800.0, 15.2, -2.1, 5.4),
        ("AMPHIBIAN", "Pseudacris crucifer (Spring Peeper)", 0.92, "Dryophytes versicolor", 0.08, 2900.0, 22.0, 1.5, -3.2),
        ("INSECT", "Neotibicen linnei (Cicada)", 0.87, "Tibicen canicularis", 0.21, 6500.0, 12.4, -4.5, -1.8),
        ("BIRD", "Poecile atricapillus (Black-capped Chickadee)", 0.96, "Baeolophus bicolor", 0.09, 3800.0, 19.8, 0.8, 2.1),
        ("MAMMAL", "Sciurus carolinensis (Eastern Gray Squirrel)", 0.82, "Tamias striatus", 0.18, 1800.0, 11.5, 3.2, -2.0),
        ("BIRD", "Setophaga ruticilla (American Redstart)", 0.91, "Setophaga coronata", 0.14, 5200.0, 16.7, -1.8, 4.0),
        ("BIRD", "Zonotrichia albicollis (White-throated Sparrow)", 0.93, "Spizella passerina", 0.11, 4100.0, 20.1, 2.4, 1.9),
        ("AMPHIBIAN", "Lithobates clamitans (Green Frog)", 0.85, "Lithobates catesbeianus", 0.19, 1400.0, 14.0, -3.0, -4.5),
        ("BIRD", "Dryocopus pileatus (Pileated Woodpecker)", 0.95, "Colaptes auratus", 0.08, 1200.0, 25.3, 5.0, 6.2),
        ("BIRD", "Cardinalis cardinalis (Northern Cardinal)", 0.97, "Pipilo erythrophthalmus", 0.07, 2400.0, 23.1, -1.2, 1.4),
        ("INSECT", "Gryllus pennsylvanicus (Field Cricket)", 0.88, "Oecanthus fultoni", 0.16, 4800.0, 13.6, 2.8, -3.5),
    ]

    base_sample = 0

    for i, (tax_class, species_name, conf, _sec_name, sec_conf, dom_freq, snr, x_pos, y_pos) in enumerate(species_events):
        event_id = i + 1
        event_dur_s = 0.65 + 0.3 * (i % 3)
        dur_samples = int(event_dur_s * sample_rate)
        start_sample = base_sample + int(i * 12.5 * sample_rate)
        end_sample = start_sample + dur_samples

        # Create mock audio file
        event_dir = event_storage_dir / f"event_{event_id:04d}"
        event_dir.mkdir(parents=True, exist_ok=True)

        t = np.linspace(0, event_dur_s, dur_samples, endpoint=False)
        carrier = np.sin(2 * np.pi * dom_freq * t) * np.exp(-3 * t / event_dur_s)
        harmonics = 0.4 * np.sin(2 * np.pi * dom_freq * 1.5 * t)
        noise = 0.05 * np.random.randn(dur_samples)
        audio_float = 0.7 * (carrier + harmonics) + noise
        audio_int16 = (audio_float * 32767).astype(np.int16)

        for node_id in (1, 2, 3):
            wav_path = event_dir / f"node_{node_id}.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

        # AcousticEvent
        event = AcousticEvent(
            event_id=event_id,
            session_id=session_id,
            start_sample=start_sample,
            end_sample=end_sample,
            trigger_nodes=(1, 2, 3),
            peak_rms_dbfs=-18.0 + snr * 0.5,
        )

        env_payload = EnvironmentPayload(
            temperature_c=21.5 + 2.0 * math.sin(i / 3.0),
            humidity_percent=62.0 + 5.0 * math.cos(i / 2.5),
            pressure_hpa=1013.2 + 0.5 * math.sin(i),
        )

        from types import SimpleNamespace

        mock_loc = SimpleNamespace(
            position=PositionResult(
                x=x_pos,
                y=y_pos,
                success=True,
                residual_rms_seconds=0.0008,
                residual_rms_meters=0.27,
                cost=0.04,
                nfev=6,
                message="optimal",
            ),
            tdoa_measurements=[],
            node_positions=node_positions,
            speed_of_sound_mps=343.8,
        )

        db_event_id = db.add_event(
            event=event,
            environment=env_payload,
            localization=mock_loc,
            event_directory=str(event_dir),
            best_node_id=1,
        )

        # DSP Features
        from wildlife_soundscape.dsp.features import AcousticFeatures
        from wildlife_soundscape.classification.classifier import AcousticClass, ClassificationResult

        mfcc_mean_vals = tuple(float(f) for f in (np.linspace(-15.0, 5.0, 13) + np.random.randn(13) * 0.5))
        mfcc_std_vals = tuple(float(f) for f in (np.ones(13) * 1.8 + np.random.rand(13) * 0.4))

        acoustic_feats = AcousticFeatures(
            duration_s=event_dur_s,
            rms=0.12 + 0.05 * (i % 2),
            peak_amplitude=0.65,
            crest_factor=3.8,
            zero_crossing_rate=0.15 + (dom_freq / 20000.0),
            dominant_frequency_hz=dom_freq,
            spectral_centroid_hz=dom_freq + 450.0,
            spectral_bandwidth_hz=1200.0,
            spectral_rolloff_hz=dom_freq + 1600.0,
            spectral_flatness=0.04,
            spectral_flux=0.28,
            snr_db=snr,
            mfcc_mean=mfcc_mean_vals,
            mfcc_std=mfcc_std_vals,
        )

        db.add_event_features(
            event_id=db_event_id,
            source_node_id=1,
            features=acoustic_feats,
        )

        # Classification
        primary_cls = AcousticClass[tax_class]
        if primary_cls == AcousticClass.BIRD:
            second_cls = AcousticClass.INSECT
        elif primary_cls == AcousticClass.INSECT:
            second_cls = AcousticClass.AMPHIBIAN
        elif primary_cls == AcousticClass.AMPHIBIAN:
            second_cls = AcousticClass.BIRD
        else:
            second_cls = AcousticClass.NOISE

        clf_result = ClassificationResult(
            label=primary_cls,
            confidence=conf,
            second_label=second_cls,
            second_confidence=sec_conf,
            margin=conf - sec_conf,
            scores={primary_cls: conf, second_cls: sec_conf, AcousticClass.NOISE: 0.05, AcousticClass.UNKNOWN: 0.02},
            reasons=(f"Species: {species_name}", f"Class: {tax_class}"),
            classifier_name="BirdNET-V3+TaxonomyEnsemble",
            classifier_version="3.0.0",
        )

        db.add_classification(
            event_id=db_event_id,
            result=clf_result,
        )

    # Add Continuous Ecoacoustic Soundscape Indices (e.g. 10 time windows across the hour)
    cfg = SoundscapeIndicesConfig()
    for window_idx in range(12):
        window_start = window_idx * (sample_rate * 60)
        window_end = (window_idx + 1) * (sample_rate * 60)
        t_factor = window_idx / 12.0

        for node_id in (1, 2, 3):
            # Simulate dawn chorus increase then leveling off
            aci_val = 14.0 + 8.0 * math.sin(math.pi * t_factor) + (node_id * 0.5)
            ndsi_val = 0.35 + 0.4 * math.sin(math.pi * t_factor) - 0.05 * node_id
            entropy_val = 0.65 + 0.2 * math.cos(math.pi * t_factor)
            bi_val = 8.5 + 5.0 * math.sin(math.pi * t_factor)

            res = SoundscapeIndicesResult(
                aci=round(aci_val, 2),
                ndsi=round(ndsi_val, 3),
                acoustic_entropy=round(entropy_val, 3),
                temporal_entropy=round(min(1.0, entropy_val * 1.1), 3),
                spectral_entropy=round(min(1.0, entropy_val * 0.95), 3),
                bioacoustic_index=round(bi_val, 2),
                anthrophony_power=round(0.015 + 0.005 * (window_idx % 3), 4),
                biophony_power=round(0.045 + 0.03 * math.sin(math.pi * t_factor), 4),
                parameters=cfg,
            )
            db.add_soundscape_indices(
                session_id=session_id,
                node_id=node_id,
                start_sample=window_start,
                end_sample=window_end,
                result=res,
            )

    print(f"Successfully generated {len(species_events)} classified & localized acoustic events and 36 continuous soundscape index windows.")


if __name__ == "__main__":
    generate_demo_dataset()
