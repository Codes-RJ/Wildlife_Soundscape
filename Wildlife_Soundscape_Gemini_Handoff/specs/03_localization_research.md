# Spec — GCC-PHAT Research Variants

## Preserve existing contract

- current beta=1 behavior is the baseline;
- delay sign is unchanged;
- solver equation is unchanged;
- shared sampleIndex coarse alignment is unchanged.

## New parameters

Suggested GCC function additions:
- `beta: float = 1.0`
- `frequency_band_hz: tuple[float, float] | None = None`

Validation:
- beta finite and in [0,1];
- band lower >= 0;
- band upper > lower;
- upper <= Nyquist.

## Frequency-domain weighting

Conceptually:
`weighted_cross_spectrum = G / (abs(G) ** beta + eps)`

Use an implementation that is numerically stable at near-zero magnitude.

## Band limiting

Apply an FFT-bin mask before inverse transform.

Do not derive a band from spectral centroid/bandwidth unless that method is
explicitly evaluated. Initial benchmark should support fixed bands.

## Benchmark

CSV/JSON output should include:
- beta
- low_hz/high_hz
- scenario
- true source x/y
- estimated source x/y
- localization error m
- pairwise true/measured TDOA
- success/failure reason
