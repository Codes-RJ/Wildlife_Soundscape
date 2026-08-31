"""
Tests for continuous ecoacoustic soundscape indices and soundscape service.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analytics.indices import (
    SoundscapeIndicesConfig,
    SoundscapeIndicesResult,
    calculate_aci,
    calculate_acoustic_entropy,
    calculate_bioacoustic_index,
    calculate_ndsi,
    calculate_soundscape_indices,
)
from analytics.soundscape_service import SoundscapeService
from config import AnalyticsConfig
from database import EventDatabase


# ======================================================================
# CONFIGURATION TESTS
# ======================================================================


def test_soundscape_indices_config_defaults_and_validation() -> None:
    cfg = SoundscapeIndicesConfig()
    assert cfg.sample_rate == 48000
    assert cfg.n_fft == 1024
    assert cfg.hop_length == 512
    assert cfg.ndsi_anthrophony_min_hz == 1000.0
    assert cfg.ndsi_anthrophony_max_hz == 2000.0
    assert cfg.ndsi_biophony_min_hz == 2000.0
    assert cfg.ndsi_biophony_max_hz == 8000.0

    # Invalid sample rate
    with pytest.raises(ValueError):
        SoundscapeIndicesConfig(sample_rate=0)

    # Invalid n_fft (non power of 2)
    with pytest.raises(ValueError):
        SoundscapeIndicesConfig(n_fft=1000)

    # Inverted band
    with pytest.raises(ValueError):
        SoundscapeIndicesConfig(ndsi_anthrophony_min_hz=3000.0, ndsi_anthrophony_max_hz=2000.0)

    # Exceeding Nyquist
    with pytest.raises(ValueError):
        SoundscapeIndicesConfig(sample_rate=16000, ndsi_biophony_max_hz=10000.0)


def test_application_soundscape_config_validation() -> None:
    config = AnalyticsConfig()
    assert config.soundscape_enabled
    assert config.soundscape_window_seconds == 60.0

    with pytest.raises(TypeError):
        AnalyticsConfig(soundscape_enabled=1).validate()

    with pytest.raises(ValueError):
        AnalyticsConfig(soundscape_window_seconds=0.0).validate()

    with pytest.raises(ValueError):
        AnalyticsConfig(soundscape_window_seconds=float("nan")).validate()


# ======================================================================
# ACI TESTS
# ======================================================================


def test_calculate_aci_behavior() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)

    # Silence
    silence = np.zeros(sr, dtype=np.float32)
    assert calculate_aci(silence, sr) == 0.0

    # Constant sine wave (constant amplitude -> low ACI)
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    aci_tone = calculate_aci(tone, sr)

    # Modulated chirp / complex pulsed signal -> higher ACI
    mod = (np.sin(2 * np.pi * 10 * t) + 1.0) * 0.5
    chirp = (np.sin(2 * np.pi * 3000 * t) * mod).astype(np.float32)
    aci_chirp = calculate_aci(chirp, sr)

    assert aci_chirp > aci_tone
    assert aci_chirp >= 0.0


# ======================================================================
# NDSI TESTS
# ======================================================================


def test_calculate_ndsi_frequency_selectivity() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)

    # Anthrophony pure tone (1500 Hz inside [1000, 2000] Hz)
    anthro_tone = np.sin(2 * np.pi * 1500 * t).astype(np.float32)
    ndsi_anthro, p_anthro, p_bio = calculate_ndsi(anthro_tone, sr)
    assert ndsi_anthro < -0.8
    assert p_anthro > p_bio

    # Biophony pure tone (4000 Hz inside [2000, 8000] Hz)
    bio_tone = np.sin(2 * np.pi * 4000 * t).astype(np.float32)
    ndsi_bio, p_anthro2, p_bio2 = calculate_ndsi(bio_tone, sr)
    assert ndsi_bio > 0.8
    assert p_bio2 > p_anthro2

    # Silence
    silence = np.zeros(sr, dtype=np.float32)
    ndsi_silence, _, _ = calculate_ndsi(silence, sr)
    assert ndsi_silence == 0.0


# ======================================================================
# ACOUSTIC ENTROPY TESTS
# ======================================================================


def test_calculate_acoustic_entropy() -> None:
    sr = 48000
    rng = np.random.default_rng(42)

    # White noise has broad spectral distribution -> high spectral entropy
    noise = rng.standard_normal(sr).astype(np.float32)
    h_noise, ht_noise, hf_noise = calculate_acoustic_entropy(noise, sr)
    assert 0.0 <= h_noise <= 1.0
    assert hf_noise > 0.8  # high spectral entropy

    # Pure tone has peaked spectral distribution -> lower spectral entropy
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = np.sin(2 * np.pi * 2000 * t).astype(np.float32)
    h_tone, ht_tone, hf_tone = calculate_acoustic_entropy(tone, sr)
    assert hf_tone < hf_noise

    # Silence
    silence = np.zeros(sr, dtype=np.float32)
    h_silence, _, _ = calculate_acoustic_entropy(silence, sr)
    assert h_silence == 0.0


# ======================================================================
# BIOACOUSTIC INDEX TESTS
# ======================================================================


def test_calculate_bioacoustic_index() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)

    # In-band signal (3500 Hz)
    in_band = np.sin(2 * np.pi * 3500 * t).astype(np.float32)
    bi_in = calculate_bioacoustic_index(in_band, sr)

    # Out-of-band signal (500 Hz)
    out_band = np.sin(2 * np.pi * 500 * t).astype(np.float32)
    bi_out = calculate_bioacoustic_index(out_band, sr)

    assert bi_in > bi_out
    assert bi_in >= 0.0


# ======================================================================
# UNIFIED CALCULATION AND DATABASE PERSISTENCE
# ======================================================================


def test_unified_indices_and_database_persistence(tmp_path: Path) -> None:
    sr = 48000
    db_path = tmp_path / "events.db"
    db = EventDatabase(db_path)
    session_id = 42
    db.start_session(session_id, "Test Soundscape Session")

    # Generate synthetic soundscape audio
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = (
        0.5 * np.sin(2 * np.pi * 3000 * t)
        + 0.2 * np.sin(2 * np.pi * 1200 * t)
    ).astype(np.float32)

    result = calculate_soundscape_indices(audio, sr)
    assert isinstance(result, SoundscapeIndicesResult)
    assert -1.0 <= result.ndsi <= 1.0
    assert result.aci >= 0.0
    assert 0.0 <= result.acoustic_entropy <= 1.0
    assert result.bioacoustic_index >= 0.0

    # Persist to database
    row_id = db.add_soundscape_indices(
        session_id=session_id,
        node_id=1,
        start_sample=0,
        end_sample=sr,
        result=result,
    )
    assert row_id > 0

    # Query back
    records = db.get_soundscape_indices(session_id=session_id, node_id=1)
    assert len(records) == 1
    rec = records[0]
    assert rec["id"] == row_id
    assert rec["session_id"] == session_id
    assert rec["node_id"] == 1
    assert rec["start_sample"] == 0
    assert rec["end_sample"] == sr
    assert pytest.approx(rec["aci"], rel=1e-4) == result.aci
    assert pytest.approx(rec["ndsi"], rel=1e-4) == result.ndsi
    assert "n_fft" in rec["parameters"]


def test_soundscape_service_streaming_buffer(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    db = EventDatabase(db_path)
    session_id = 99
    db.start_session(session_id, "Streaming Test")

    # 1.0 second window for fast testing
    service = SoundscapeService(
        window_duration_seconds=1.0,
        database=db,
    )
    sr = service.config.sample_rate

    # Push two half-second chunks
    chunk1 = np.ones(sr // 2, dtype=np.float32) * 0.1
    chunk2 = np.ones(sr // 2, dtype=np.float32) * 0.2

    res1 = service.append_audio(node_id=1, audio_chunk=chunk1, session_id=session_id, start_sample=0)
    assert res1 is None  # not enough audio yet

    res2 = service.append_audio(node_id=1, audio_chunk=chunk2, session_id=session_id, start_sample=sr // 2)
    assert res2 is not None  # window completed
    assert isinstance(res2, SoundscapeIndicesResult)

    # Verify DB persistence
    records = db.get_soundscape_indices(session_id=session_id)
    assert len(records) == 1
    assert records[0]["start_sample"] == 0
    assert records[0]["end_sample"] == sr


def test_soundscape_service_does_not_join_across_gap(tmp_path: Path) -> None:
    db = EventDatabase(tmp_path / "gap.db")
    session_id = 100
    db.start_session(session_id, "Gap Test")
    service = SoundscapeService(window_duration_seconds=0.02, database=db)
    window_samples = service.window_samples

    first = np.ones(window_samples // 2, dtype=np.float32)
    second = np.ones(window_samples // 2, dtype=np.float32)

    assert service.append_audio(
        1,
        first,
        session_id=session_id,
        start_sample=0,
    ) is None

    # The missing region resets the partial window instead of compressing it.
    assert service.append_audio(
        1,
        second,
        session_id=session_id,
        start_sample=window_samples,
    ) is None
    assert db.get_soundscape_indices(session_id=session_id) == []

    result = service.append_audio(
        1,
        second,
        session_id=session_id,
        start_sample=window_samples + second.size,
    )
    assert result is not None

    records = db.get_soundscape_indices(session_id=session_id)
    assert len(records) == 1
    assert records[0]["start_sample"] == window_samples
    assert records[0]["end_sample"] == 2 * window_samples


def test_soundscape_service_isolates_sessions_and_trims_duplicates(
    tmp_path: Path,
) -> None:
    db = EventDatabase(tmp_path / "sessions.db")
    first_session = 101
    second_session = 102
    db.start_session(first_session, "First")
    db.start_session(second_session, "Second")
    service = SoundscapeService(window_duration_seconds=0.02, database=db)
    half_window = service.window_samples // 2
    chunk = np.ones(half_window, dtype=np.float32)

    assert service.append_audio(
        1,
        chunk,
        session_id=first_session,
        start_sample=0,
    ) is None

    # A session change clears the half-window from the previous session.
    assert service.append_audio(
        1,
        chunk,
        session_id=second_session,
        start_sample=0,
    ) is None

    # A fully duplicated chunk is ignored rather than double-counted.
    assert service.append_audio(
        1,
        chunk,
        session_id=second_session,
        start_sample=0,
    ) is None

    result = service.append_audio(
        1,
        chunk,
        session_id=second_session,
        start_sample=half_window,
    )
    assert result is not None
    assert db.get_soundscape_indices(session_id=first_session) == []

    records = db.get_soundscape_indices(session_id=second_session)
    assert len(records) == 1
    assert records[0]["start_sample"] == 0
    assert records[0]["end_sample"] == service.window_samples
