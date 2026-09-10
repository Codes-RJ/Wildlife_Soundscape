"""
Benchmark tool for evaluating GCC-PHAT-beta and frequency band-limiting variants.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
Systematically quantify the TDOA estimation and spatial localization accuracy
of standard GCC-PHAT (beta=1.0) versus fractional whitening (0 <= beta < 1)
and frequency band-limiting across controlled synthetic acoustic scenarios
(SNR variations, ambient noise, out-of-band interference).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wildlife_soundscape.localization.gcc_phat import gcc_phat  # noqa: E402
from wildlife_soundscape.localization.solver import solve_position  # noqa: E402
from wildlife_soundscape.localization.tdoa import TDOAMeasurement  # noqa: E402


@dataclass(frozen=True, slots=True)
class BenchmarkResultRow:
    scenario: str
    snr_db: float
    beta: float
    band_low_hz: float
    band_high_hz: float
    true_x_m: float
    true_y_m: float
    estimated_x_m: float | None
    estimated_y_m: float | None
    pos_error_m: float | None
    true_tdoa_21_s: float
    measured_tdoa_21_s: float
    true_tdoa_31_s: float
    measured_tdoa_31_s: float
    true_tdoa_32_s: float
    measured_tdoa_32_s: float
    tdoa_rms_error_s: float
    success: bool
    reason: str


def run_benchmark(
    *,
    betas: list[float] | None = None,
    bands: list[tuple[float, float]] | None = None,
    snr_levels_db: list[float] | None = None,
    sample_rate: int = 48000,
    speed_of_sound_mps: float = 343.0,
) -> list[BenchmarkResultRow]:
    """
    Execute systematic GCC variant benchmarks.
    """
    if betas is None:
        betas = [1.0, 0.8, 0.5, 0.2, 0.0]
    if bands is None:
        bands = [(0.0, float(sample_rate // 2)), (1000.0, 8000.0)]
    if snr_levels_db is None:
        snr_levels_db = [20.0, 10.0, 0.0, -5.0]

    # Sensor geometry: 3-node triangular array
    node_positions = {
        1: (0.0, 0.0),
        2: (2.0, 0.0),
        3: (1.0, 1.732),
    }

    # True source location
    true_x, true_y = 5.0, 4.0
    d1 = math.hypot(true_x - node_positions[1][0], true_y - node_positions[1][1])
    d2 = math.hypot(true_x - node_positions[2][0], true_y - node_positions[2][1])
    d3 = math.hypot(true_x - node_positions[3][0], true_y - node_positions[3][1])

    true_tau_21 = (d2 - d1) / speed_of_sound_mps
    true_tau_31 = (d3 - d1) / speed_of_sound_mps
    true_tau_32 = (d3 - d2) / speed_of_sound_mps

    pairs = [(1, 2, true_tau_21), (1, 3, true_tau_31), (2, 3, true_tau_32)]

    results: list[BenchmarkResultRow] = []
    duration_s = 0.5
    t = np.linspace(
        0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float64
    )

    # Base source signal: modulated bioacoustic chirp (2.5 kHz to 4.5 kHz)
    chirp_freq = 2500.0 + 2000.0 * (t / duration_s)
    clean_signal = np.sin(2.0 * np.pi * chirp_freq * t)

    # Out-of-band low-frequency noise (e.g. 100 Hz hum/wind)
    noise_hum = 0.5 * np.sin(2.0 * np.pi * 100.0 * t)

    rng = np.random.default_rng(12345)

    for snr in snr_levels_db:
        # Scale clean signal vs white noise
        sig_power = float(np.mean(clean_signal**2))
        noise_power = sig_power / (10.0 ** (snr / 10.0))
        noise_sigma = math.sqrt(noise_power)

        # Generate received signals per node with fractional delay via FFT
        node_audio: dict[int, np.ndarray] = {}
        for nid, (nx, ny) in node_positions.items():
            dist = math.hypot(true_x - nx, true_y - ny)
            delay_s = dist / speed_of_sound_mps

            # FFT delay (phase shift by delay_s seconds)
            spec = np.fft.rfft(clean_signal)
            freqs = np.fft.rfftfreq(len(clean_signal), 1.0 / sample_rate)
            phase_shift = np.exp(-2j * np.pi * freqs * delay_s)
            delayed_sig = np.fft.irfft(spec * phase_shift, n=len(clean_signal))

            # Add white noise and low-freq hum
            w_noise = rng.standard_normal(len(clean_signal)) * noise_sigma
            node_audio[nid] = delayed_sig + w_noise + noise_hum

        for beta in betas:
            for band in bands:
                tdoa_measurements: list[TDOAMeasurement] = []
                tdoa_errors: list[float] = []
                measured_tdoas: dict[tuple[int, int], float] = {}

                for na, nb, true_delay in pairs:
                    max_pair_delay = (
                        math.hypot(
                            node_positions[na][0] - node_positions[nb][0],
                            node_positions[na][1] - node_positions[nb][1],
                        )
                        / speed_of_sound_mps
                    )

                    res = gcc_phat(
                        signal=node_audio[nb],
                        reference=node_audio[na],
                        sample_rate=sample_rate,
                        max_delay_seconds=max_pair_delay * 1.2,
                        beta=beta,
                        frequency_band_hz=band,
                    )

                    tdoa_err = abs(res.delay_seconds - true_delay)
                    tdoa_errors.append(tdoa_err)
                    measured_tdoas[(na, nb)] = float(res.delay_seconds)

                    tdoa_measurements.append(
                        TDOAMeasurement(
                            node_a=na,
                            node_b=nb,
                            delay_seconds=res.delay_seconds,
                            delay_samples=res.delay_samples,
                            peak_ratio=res.peak_ratio,
                            max_delay_seconds=max_pair_delay * 1.2,
                            valid=res.valid,
                            reason=res.reason,
                        )
                    )

                tdoa_rms = float(np.sqrt(np.mean(np.array(tdoa_errors) ** 2)))

                # Solve 2-D position
                pos_res = solve_position(
                    node_positions,
                    tdoa_measurements,
                    speed_of_sound_mps=speed_of_sound_mps,
                )

                pos_err = None
                if pos_res.success:
                    pos_err = float(math.hypot(pos_res.x - true_x, pos_res.y - true_y))

                results.append(
                    BenchmarkResultRow(
                        scenario="synthetic_chirp_noise",
                        snr_db=snr,
                        beta=beta,
                        band_low_hz=band[0],
                        band_high_hz=band[1],
                        true_x_m=true_x,
                        true_y_m=true_y,
                        estimated_x_m=float(pos_res.x) if pos_res.success else None,
                        estimated_y_m=float(pos_res.y) if pos_res.success else None,
                        pos_error_m=pos_err,
                        true_tdoa_21_s=float(true_tau_21),
                        measured_tdoa_21_s=measured_tdoas[(1, 2)],
                        true_tdoa_31_s=float(true_tau_31),
                        measured_tdoa_31_s=measured_tdoas[(1, 3)],
                        true_tdoa_32_s=float(true_tau_32),
                        measured_tdoa_32_s=measured_tdoas[(2, 3)],
                        tdoa_rms_error_s=tdoa_rms,
                        success=bool(pos_res.success),
                        reason=str(pos_res.message),
                    )
                )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GCC-PHAT variants.")
    parser.add_argument(
        "--output-csv", type=Path, default=Path("benchmark_gcc_results.csv")
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    print("Running GCC-PHAT variant benchmarks...")
    results = run_benchmark()

    # Export CSV
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[k for k in BenchmarkResultRow.__annotations__]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"Benchmark completed: {len(results)} configurations evaluated.")
    print(f"CSV saved to {args.output_csv}")

    if args.output_json:
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"JSON saved to {args.output_json}")


if __name__ == "__main__":
    main()
