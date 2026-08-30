# Spec — Ecoacoustic Indices

## Functions

Implement clean pure functions first, then service integration.

Suggested API:
- `calculate_aci(...)`
- `calculate_ndsi(...)`
- `calculate_acoustic_entropy(...)`
- `calculate_bioacoustic_index(...)`
- `calculate_soundscape_indices(...)`

Use NumPy/SciPy/Librosa only if already justified by project dependencies.

## Numerical requirements

- mono 1-D input;
- finite samples;
- positive sample rate;
- stable behavior for silence;
- configurable FFT/hop;
- configurable frequency bands;
- explicit epsilon/zero-denominator handling.

## NDSI

Persist the exact bands used.

Suggested defaults:
- anthropogenic: 1000–2000 Hz
- biological: 2000–8000 Hz

Do not call these universal.

## ACI

Use a published ACI-style spectrogram amplitude-difference formulation.

The chosen FFT, hop, segmentation and aggregation strategy must be documented.

## Entropy

If implementing combined acoustic entropy:
- calculate normalized temporal entropy;
- calculate normalized spectral entropy;
- define combined H explicitly, e.g. product if matching the chosen literature
  definition.

## BI

Document:
- frequency band;
- spectrum representation;
- baseline/reference handling;
- integration method.

## Storage

Indices are window metrics, not event metrics.
Keep them out of the event table.
