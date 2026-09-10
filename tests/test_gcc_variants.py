"""
Tests for GCC-PHAT-beta fractional weighting and frequency band-limiting.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import replace

import numpy as np
import pytest

from wildlife_soundscape.core.config import CONFIG
from wildlife_soundscape.localization.gcc_phat import gcc_phat
from wildlife_soundscape.tools.benchmark_gcc_variants import run_benchmark


def test_localization_config_validates_gcc_research_options() -> None:
    valid = replace(
        CONFIG.localization,
        gcc_beta=0.75,
        gcc_frequency_band_hz=(1000.0, 8000.0),
    )
    valid.validate(
        sample_rate=CONFIG.audio.sample_rate,
        expected_nodes=CONFIG.expected_nodes,
    )

    for invalid_beta in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            replace(
                CONFIG.localization,
                gcc_beta=invalid_beta,
            ).validate(
                sample_rate=CONFIG.audio.sample_rate,
                expected_nodes=CONFIG.expected_nodes,
            )

    with pytest.raises(TypeError):
        replace(
            CONFIG.localization,
            gcc_frequency_band_hz=[1000.0, 8000.0],
        ).validate(
            sample_rate=CONFIG.audio.sample_rate,
            expected_nodes=CONFIG.expected_nodes,
        )

    for invalid_band in (
        (8000.0, 1000.0),
        (-1.0, 8000.0),
        (1000.0, 30000.0),
    ):
        with pytest.raises(ValueError):
            replace(
                CONFIG.localization,
                gcc_frequency_band_hz=invalid_band,
            ).validate(
                sample_rate=CONFIG.audio.sample_rate,
                expected_nodes=CONFIG.expected_nodes,
            )


def test_gcc_phat_beta_validation() -> None:
    sig = np.sin(np.linspace(0, 10, 500, dtype=np.float64))
    # Valid betas
    res1 = gcc_phat(sig, sig, sample_rate=48000, beta=1.0)
    assert res1.valid
    res0 = gcc_phat(sig, sig, sample_rate=48000, beta=0.0)
    assert res0.valid
    res_half = gcc_phat(sig, sig, sample_rate=48000, beta=0.5)
    assert res_half.valid

    # Invalid betas
    with pytest.raises(ValueError):
        gcc_phat(sig, sig, sample_rate=48000, beta=-0.1)
    with pytest.raises(ValueError):
        gcc_phat(sig, sig, sample_rate=48000, beta=1.5)
    with pytest.raises(TypeError):
        gcc_phat(sig, sig, sample_rate=48000, beta="invalid")


def test_gcc_phat_frequency_band_validation() -> None:
    sig = np.sin(np.linspace(0, 10, 500, dtype=np.float64))
    # Valid band
    res = gcc_phat(sig, sig, sample_rate=48000, frequency_band_hz=(1000.0, 5000.0))
    assert res.valid

    # Invalid bands
    with pytest.raises(ValueError):
        gcc_phat(sig, sig, sample_rate=48000, frequency_band_hz=(-100.0, 5000.0))
    with pytest.raises(ValueError):
        gcc_phat(sig, sig, sample_rate=48000, frequency_band_hz=(5000.0, 1000.0))
    with pytest.raises(ValueError):
        gcc_phat(
            sig, sig, sample_rate=48000, frequency_band_hz=(1000.0, 30000.0)
        )  # exceeds 24k Nyquist
    with pytest.raises(TypeError):
        gcc_phat(sig, sig, sample_rate=48000, frequency_band_hz="1000-5000")


def test_gcc_phat_band_limiting_selectivity() -> None:
    sr = 48000
    duration_s = 0.1
    t = np.linspace(
        0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float64
    )

    # 4-6 kHz in-band chirp delayed by 10 samples (positive delay: signal arrives later)
    delay_samples = 10
    chirp_f = 4000.0 + 2000.0 * (t / duration_s)
    sig_ref = np.sin(2.0 * np.pi * chirp_f * t)
    sig_delayed = np.roll(sig_ref, delay_samples)

    # Out-of-band interference (100 Hz hum with huge amplitude)
    hum = 10.0 * np.sin(2.0 * np.pi * 100.0 * t)

    # Bandpass filter around in-band signal [3000, 7000] Hz
    res_banded = gcc_phat(
        sig_delayed + hum,
        sig_ref + hum,
        sample_rate=sr,
        frequency_band_hz=(3000.0, 7000.0),
        beta=1.0,
    )
    assert res_banded.valid
    assert pytest.approx(res_banded.delay_samples, abs=0.5) == delay_samples


def test_benchmark_gcc_variants_runner() -> None:
    results = run_benchmark(
        betas=[1.0, 0.5],
        bands=[(1000.0, 8000.0)],
        snr_levels_db=[20.0],
    )
    assert len(results) == 2
    for r in results:
        assert r.scenario == "synthetic_chirp_noise"
        assert r.pos_error_m is not None
        assert r.pos_error_m < 1.0  # sub-meter accuracy under 20 dB SNR
        assert type(r.success) is bool
        assert np.isfinite(r.true_tdoa_21_s)
        assert np.isfinite(r.measured_tdoa_21_s)
        assert np.isfinite(r.true_tdoa_31_s)
        assert np.isfinite(r.measured_tdoa_31_s)
        assert np.isfinite(r.true_tdoa_32_s)
        assert np.isfinite(r.measured_tdoa_32_s)

    # Every benchmark row must remain directly exportable by the CLI.
    json.dumps([asdict(result) for result in results])
