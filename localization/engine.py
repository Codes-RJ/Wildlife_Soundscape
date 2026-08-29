from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from config import LocalizationConfig
from environment import calculate_speed_of_sound_mps
from stream_manager import StreamManager
from .filtering import bandpass_filter
from .gcc_phat import gcc_phat
from .solver import PositionResult, solve_position
from .tdoa import TDOAMeasurement, physical_max_delay


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    position: PositionResult
    measurements: tuple[TDOAMeasurement, ...]
    window_start_sample: int
    window_samples: int
    speed_of_sound_mps: float
    node_rms: dict[int, float]
    environment_used: tuple[float, float, float] | None = None

    @property
    def success(self) -> bool:
        return self.position.success


class LocalizationEngine:
    def __init__(self, streams: StreamManager, config: LocalizationConfig) -> None:
        self.streams = streams
        self.config = config

    def _bounds(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if not self.config.constrain_to_array_bounds:
            return None
        xs = [xy[0] for xy in self.config.node_positions.values()]
        ys = [xy[1] for xy in self.config.node_positions.values()]
        m = self.config.bounds_margin_m
        return ((min(xs) - m, min(ys) - m), (max(xs) + m, max(ys) + m))

    def _resolve_speed(
        self,
        *,
        start_sample: int,
        speed_of_sound_mps: float | None,
    ) -> tuple[float, tuple[float, float, float] | None]:
        if speed_of_sound_mps is not None:
            return float(speed_of_sound_mps), None

        if self.config.use_environmental_speed:
            env = self.streams.get_environment_near(start_sample)
            if env is not None:
                c = calculate_speed_of_sound_mps(
                    env.temperature_c,
                    env.humidity_percent,
                    env.pressure_hpa,
                )
                return c, (env.temperature_c, env.humidity_percent, env.pressure_hpa)

        return float(self.config.speed_of_sound_mps), None

    def locate_window(
        self,
        *,
        start_sample: int,
        length: int | None = None,
        speed_of_sound_mps: float | None = None,
    ) -> LocalizationResult:
        n = int(length or self.config.window_samples)
        if n < 64:
            raise ValueError("localization window must be >= 64 samples")

        c, env_used = self._resolve_speed(
            start_sample=start_sample,
            speed_of_sound_mps=speed_of_sound_mps,
        )
        if c <= 0:
            raise ValueError("speed of sound must be positive")

        sample_rate = self.streams.audio_config.sample_rate
        windows: dict[int, np.ndarray] = {}
        rms: dict[int, float] = {}
        for node_id in sorted(self.config.node_positions):
            raw = self.streams.get_window(node_id, start_sample, n).astype(np.float64)
            rms[node_id] = float(np.sqrt(np.mean(raw * raw))) if raw.size else 0.0
            if self.config.bandpass_enabled:
                conditioned = bandpass_filter(
                    raw,
                    sample_rate=sample_rate,
                    low_hz=self.config.bandpass_low_hz,
                    high_hz=self.config.bandpass_high_hz,
                    order=self.config.bandpass_order,
                )
            else:
                conditioned = raw
            windows[node_id] = conditioned

        measurements: list[TDOAMeasurement] = []
        for a, b in itertools.combinations(sorted(windows), 2):
            max_tau = physical_max_delay(
                self.config.node_positions[a],
                self.config.node_positions[b],
                speed_of_sound_mps=c,
            )

            if min(rms[a], rms[b]) < self.config.min_rms:
                measurements.append(
                    TDOAMeasurement(
                        node_a=a,
                        node_b=b,
                        delay_seconds=0.0,
                        delay_samples=0.0,
                        peak_ratio=0.0,
                        max_delay_seconds=max_tau,
                        valid=False,
                        reason="insufficient signal energy",
                    )
                )
                continue

            # gcc_phat(signal=B, reference=A) => arrival(B) - arrival(A)
            g = gcc_phat(
                windows[b],
                windows[a],
                sample_rate=sample_rate,
                max_delay_seconds=max_tau,
                interpolation=self.config.interpolation,
                min_peak_ratio=self.config.min_peak_ratio,
            )
            physically_valid = abs(g.delay_seconds) <= max_tau + (1 / sample_rate)
            valid = g.valid and physically_valid
            reason = g.reason
            if not physically_valid:
                reason = "delay outside physical pair limit"

            measurements.append(
                TDOAMeasurement(
                    node_a=a,
                    node_b=b,
                    delay_seconds=g.delay_seconds,
                    delay_samples=g.delay_samples,
                    peak_ratio=g.peak_ratio,
                    max_delay_seconds=max_tau,
                    valid=valid,
                    reason=reason,
                )
            )

        pos = solve_position(
            self.config.node_positions,
            measurements,
            speed_of_sound_mps=c,
            bounds=self._bounds(),
        )
        return LocalizationResult(
            position=pos,
            measurements=tuple(measurements),
            window_start_sample=start_sample,
            window_samples=n,
            speed_of_sound_mps=c,
            node_rms=rms,
            environment_used=env_used,
        )

    def locate_latest(self, *, length: int | None = None) -> LocalizationResult | None:
        n = int(length or self.config.window_samples)
        ends = []
        for node_id in self.config.node_positions:
            state = self.streams.nodes.get(node_id)
            if state is None or not state.audio_blocks:
                return None
            ends.append(state.audio_blocks[-1].end_sample)
        common_end = min(ends)
        if common_end < n:
            return None
        return self.locate_window(start_sample=common_end - n, length=n)
