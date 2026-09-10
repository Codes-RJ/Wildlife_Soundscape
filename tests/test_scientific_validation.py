import json
import math

import pytest

from wildlife_soundscape.calibration.tdoa_calibration import (
    KnownSourceMeasurement,
    calibrate_tdoa,
)
from wildlife_soundscape.calibration.localization_benchmark import (
    LocalizationBenchmarkSample,
    benchmark_localization,
)
from wildlife_soundscape.analytics.activity import build_activity_bins


def test_calibration_recovers_known_channel_biases_with_reversed_pairs():
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0)}
    biases = {1: 0.0, 2: 2 / 48000, 3: -1 / 48000}
    observations = []
    for source in ((0.2, 0.3), (0.6, 0.4), (1.2, 0.8), (-0.2, 0.7)):
        for a, b in ((1, 2), (3, 1), (2, 3)):
            # Independent geometric ground truth, not the implementation helper.
            delay = (
                math.dist(source, positions[b]) - math.dist(source, positions[a])
            ) / 343.0
            observations.append(
                KnownSourceMeasurement(a, b, source, delay + biases[b] - biases[a])
            )
    result = calibrate_tdoa(observations, positions, min_observations_per_pair=3)
    assert result.accepted_observation_count == len(observations)
    assert result.pair_offset_seconds(1, 2) == pytest.approx(2 / 48000)
    assert result.pair_offset_seconds(3, 1) == pytest.approx(1 / 48000)
    assert result.rms_pair_consistency_error_s < 1e-12
    assert json.loads(json.dumps(result.to_dict()))["sample_rate"] == 48000


def test_benchmark_keeps_failed_trials_in_success_denominator():
    result = benchmark_localization(
        [
            LocalizationBenchmarkSample((0.0, 0.0), (0.3, 0.4), True),
            LocalizationBenchmarkSample((0.0, 0.0), (0.0, 0.0), True),
            LocalizationBenchmarkSample((0.0, 0.0), None, False),
        ]
    )
    assert result.attempt_count == 3
    assert result.failure_count == 1
    assert result.success_rate == pytest.approx(2 / 3)
    assert result.mean_error_m == pytest.approx(0.25)
    assert result.rmse_m == pytest.approx(math.sqrt(0.125))


def test_activity_event_crosses_bin_without_double_counting():
    bins = build_activity_bins(
        [{"id": 1, "event_time": "2026-01-01T00:00:50Z", "duration_s": 20}],
        bucket_seconds=60,
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:03:00Z",
    )
    assert [b.event_count for b in bins] == [1, 0, 0]
    assert [b.active_duration_s for b in bins] == pytest.approx([10, 10, 0])


def test_environmental_missing_pairs_and_constant_values_are_not_fabricated_correlations():
    from wildlife_soundscape.analytics.environmental import (
        calculate_spearman_association,
    )

    kwargs = dict(
        environmental_variable="temperature",
        response_variable="event_rate",
        min_samples=3,
    )
    constant = calculate_spearman_association([20, 20, 20, 20], [0, 1, 2, 3], **kwargs)
    assert constant.coefficient is None
    assert not constant.statistically_significant
    missing = calculate_spearman_association(
        [1, 2, None, 3, 4], [1, 2, 100, 3, 4], **kwargs
    )
    assert missing.sample_count == 4
    assert missing.coefficient == pytest.approx(1)
