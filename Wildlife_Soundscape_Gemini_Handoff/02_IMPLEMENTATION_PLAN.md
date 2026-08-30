# Ordered Implementation Plan

## Phase 0 — Baseline protection

Goal: establish a clean known state before feature work.

Actions:
- create branch `research-upgrades`;
- ensure corrected BirdNET and Ensemble files are present;
- run syntax/import checks;
- run current non-BirdNET unit tests;
- record baseline failures before making research changes.

Exit gate:
- baseline failures documented;
- no unrelated refactors.

---

## Phase 1 — BirdNET scientific correctness

### Problem

BirdNET V3 is not a bird-only taxonomy. Its current taxonomy includes birds and
non-birds. Therefore the adapter must not blindly transform every accepted
BirdNET prediction into the broad class `BIRD`.

### Required architecture

Add an explicit taxonomy/context layer.

Preferred design:

- create `classification/birdnet_context.py`;
- represent a taxon-group mapping independently from model inference;
- accept a trusted taxonomy metadata source;
- map:
  - Aves -> BIRD
  - Insecta -> INSECT
  - Amphibia -> AMPHIBIAN
  - Mammalia -> MAMMAL
  - configured anthropogenic/geophony sound classes -> NOISE
  - unsupported/unknown groups -> UNKNOWN
- never infer group by parsing the common/scientific name.

BirdNET species scores remain namespaced evidence.

### Geo prior

Add optional geographic/temporal context, but do not make it mandatory.

Configuration should support:
- latitude
- longitude
- week-of-year or date-derived week
- geo minimum confidence

Use BirdNET's GeoModel as a post-filter/prior source. Keep acoustic confidence
and geographic occurrence probability separately traceable.

Do not multiply probabilities blindly unless the resulting score is explicitly
documented as a project-specific fusion score.

### Safe fallback

If taxonomy is not available:
- preserve species evidence;
- do not assert a broad biological class that cannot be established reliably;
- return UNKNOWN/abstention or use an explicitly configured bird-only allowlist.

Exit gate:
- non-bird BirdNET predictions cannot become BIRD by accident;
- existing heuristic backend still works without BirdNET installed;
- ensemble graceful degradation remains intact.

---

## Phase 2 — Continuous ecoacoustic indices

Goal: complement discrete event analysis with continuous soundscape metrics.

Create:
- `analytics/indices.py`
- `analytics/soundscape_service.py` if a service/rolling-window layer is needed.

Minimum indices:
- ACI — Acoustic Complexity Index
- NDSI — Normalized Difference Soundscape Index
- Acoustic Entropy H
- Bioacoustic Index BI

### Windowing

Use a configurable continuous analysis window.

Recommended starting configuration:
- 60 s windows per node;
- dashboard can aggregate to 5-minute/hourly summaries.

Do not require saving all raw continuous audio. A rolling laptop-side buffer is
acceptable.

### Parameter reproducibility

All parameters must be configurable and persisted with the result or recorded in
application configuration:
- sample rate
- FFT size
- hop length
- frequency bands
- window duration
- NDSI anthrophony band
- NDSI biophony band
- BI band

Default NDSI bands may be configured as:
- anthrophony: 1–2 kHz
- biophony: 2–8 kHz

But the implementation must not claim these bands are universal.

### Persistence

Add a new table, preferably:

`soundscape_indices`

Suggested columns:
- id
- session_id
- node_id
- window_start_sample
- window_end_sample
- window_start_time
- window_duration_s
- aci
- ndsi
- acoustic_entropy
- bioacoustic_index
- parameter_version / configuration JSON

Do not overload `acoustic_events`.

Exit gate:
- deterministic tests on synthetic signals;
- no event-pipeline regression.

---

## Phase 3 — Localization research mode

Goal: preserve standard GCC-PHAT while making weighting/band-limiting
experimentally selectable.

Modify the existing localization implementation minimally.

Required options:
- `beta = 1.0` => current PHAT baseline
- `0 <= beta <= 1`
- optional frequency band `(low_hz, high_hz)`

Conceptual weighting:

`G / (abs(G) ** beta + epsilon)`

Do not change the existing delay sign convention.

### Benchmark sweep

Add a research tool, e.g.:

`tools/benchmark_gcc_variants.py`

It should compare:
- beta 0.0
- beta 0.5
- beta 0.65
- beta 0.75
- beta 0.85
- beta 1.0
- optional band-limited variants

Report:
- TDOA error
- localization Euclidean error
- median error
- mean error
- p95 error
- failure rate
- condition/scenario label

Do not declare beta 0.75 superior without measured evidence.

Exit gate:
- beta=1 reproduces baseline behavior within numerical tolerance;
- solver sign tests remain unchanged.

---

## Phase 4 — GeoJSON spatial export

Goal: export localized events to GIS safely.

Do not directly add local x/y values to latitude/longitude.

Add configuration for:
- array origin latitude
- array origin longitude
- array azimuth relative to true north

Transform:

local array x/y
-> rotate to East/North
-> local geographic projection
-> WGS84 longitude/latitude

Preferred implementation:
- use `pyproj` as an optional GIS dependency;
- use a local azimuthal-equidistant or topocentric transformation centered on
  the array origin.

Add to `tools/export_events.py`:
- `--format geojson`

GeoJSON properties should include, when available:
- event id
- timestamp
- broad class
- classification confidence
- species prediction
- localization quality/error metric
- session id
- source node information

Exit gate:
- a zero local coordinate maps to the configured array origin;
- known 10 m East/North test cases transform correctly;
- no coordinates are exported if array georeference is absent.

---

## Phase 5 — Database, analytics and dashboard integration

Only after Phases 1–4 work independently:

- add database migration/table creation for soundscape indices;
- expose index queries in dashboard data access;
- add soundscape time-series plots;
- optionally expose localization-method metadata;
- show GIS export availability;
- show acoustic vs geographic BirdNET evidence distinctly.

Do not clutter live view with research-only diagnostics unless useful.

Exit gate:
- dashboard still works when no indices exist;
- dashboard still works when BirdNET is unavailable;
- old database can be opened/migrated safely.

---

## Phase 6 — Tests, research exports and documentation

Add/extend tests for:
- BirdNET taxonomy mapping;
- BirdNET no-taxonomy abstention;
- geo prior optionality;
- ecoacoustic index edge cases;
- GCC beta validation;
- beta=1 baseline equivalence;
- band-limit validation;
- local XY -> geographic transform;
- database round-trip;
- dashboard empty-data behavior.

Add research export fields where useful.

Then run:
1. core unit tests;
2. integration tests;
3. BirdNET-specific tests if dependency is installed;
4. firmware compile separately;
5. hardware/calibration only after software validation.

---

## Phase 7 — Optional later research extension

Do NOT block the current upgrade on embeddings.

After BirdNET V3 runtime is working, optionally add:
- `model.encode(...)`;
- store 1280-dimensional V3 embeddings for selected uncertain events;
- offline UMAP + HDBSCAN clustering.

Treat clusters as recurring acoustic types, not verified species.
