# Research Notes for Implementation Decisions

## BirdNET V3

Current BirdNET V3 documentation describes:
- a global acoustic taxonomy containing birds and non-birds;
- 32 kHz internal model sampling with automatic resampling;
- 3-second segments;
- 1280-dimensional final embeddings;
- `predict(...)` and `encode(...)`;
- a separate V3 GeoModel for location/week occurrence prediction.

Therefore:
- the acquisition system can remain 48 kHz;
- the adapter must not treat all model classes as birds;
- geographic occurrence and acoustic confidence should remain traceable as
  different evidence sources;
- 1280, not 1024, is the relevant V3 embedding size.

## Ecoacoustic indices

ACI, NDSI, acoustic entropy and BI measure different properties of a soundscape.
They should be interpreted jointly rather than used as direct species-count
estimators.

NDSI band definitions vary in literature. Keep frequency bands configurable and
record them with results.

## GCC-PHAT-beta

Fractional whitening is a research parameter, not a guaranteed improvement.

The project should preserve beta=1 as the baseline and quantify alternatives.

## 3D localization

Do not add a generic 3D solver to the current 3-node planar array and present it
as resolved 3D positioning. A future 3D array should add a non-coplanar fourth
sensor/node and be validated separately.

## Publication framing

Good measurable contributions:
- baseline vs fractional/band-limited GCC localization accuracy;
- continuous ecoacoustic-index trends alongside detected events;
- effect of geographic/taxonomic filtering on BirdNET false positives;
- GIS-ready localization export;
- optional embedding clustering as exploratory analysis.

Avoid claiming:
- acoustic indices directly equal biodiversity;
- automated BirdNET output is ground truth;
- an unknown embedding cluster is a new species;
- simulator accuracy equals field accuracy.
