# Validation and Acceptance Criteria

## A. Classification

Pass conditions:
- `second_label=None` always pairs with `second_confidence=None`.
- BirdNET absence does not break heuristic classification.
- BirdNET non-bird taxa are never silently labeled BIRD.
- UNKNOWN/abstention remains zero-evidence in ensemble voting.
- taxonomy metadata lookup failure is explicit and safe.
- geographic prior remains optional.

## B. Ecoacoustic indices

Synthetic tests should include:
- silence;
- white noise;
- pure tone;
- amplitude-modulated tone;
- two-band synthetic signal for NDSI;
- very short signal;
- non-finite input rejection.

Assertions should emphasize:
- finite output;
- defined ranges where applicable;
- deterministic behavior;
- documented handling of zero denominator;
- parameter reproducibility.

Do not create tests that assume an ecological interpretation from synthetic data.

## C. GCC research mode

Required:
- beta outside [0,1] rejected;
- invalid frequency band rejected;
- beta=1 approximately equals original PHAT implementation;
- known synthetic delay recovers correct sign;
- existing B-A TDOA tests remain unchanged.

## D. GIS

Required:
- local (0,0) -> exact configured origin;
- east/north orientation test;
- azimuth rotation test;
- invalid latitude/longitude rejected;
- no silent export without georeference.

## E. Database

Required:
- clean DB creates new table;
- existing DB upgrades without losing events;
- indices insert/query round-trip;
- parameter metadata preserved.

## F. Dashboard

Required:
- works with zero index rows;
- works without BirdNET;
- works without georeference;
- no exception when localization is missing;
- plots use DB/service interfaces, not direct SQL scattered through UI.

## G. Final command sequence

Run from project root:

1. syntax/import checks
2. core pytest suite
3. BirdNET-specific pytest selection if installed
4. research benchmark smoke test
5. firmware compile separately

Do not conflate a syntax check with a runtime pass.
Do not conflate simulator success with field localization accuracy.
