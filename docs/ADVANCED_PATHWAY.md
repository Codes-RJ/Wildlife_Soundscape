# Advanced Development Pathway

## Purpose

This document defines the path from the current presentation-ready research
prototype to a measured, field-tested wildlife bioacoustics system. It covers:

- a better live and historical audio experience;
- licensed real-data acquisition and governance;
- BirdNET baseline integration;
- training a custom non-bird classifier;
- hybrid inference, uncertainty, and human review;
- localization calibration and field acceptance;
- operational hardening and final release evidence.

The goal is not to claim that an AI model is always exact. The goal is to make
every result traceable, quantify performance on held-out data, and abstain as
`unknown` when the available evidence is insufficient.

## Current baseline

The repository already provides:

- synchronized three-node PCM acquisition and continuous WAV recording;
- adaptive event detection and short synchronized evidence clips;
- DSP features and a broad heuristic classifier;
- optional BirdNET and ensemble adapters;
- TDOA/GCC-PHAT localization;
- SQLite persistence, experiment manifests, processing status, and exports;
- soundscape and environmental analytics;
- separate human-review records;
- Monitor, Sessions, and Setup dashboard areas;
- additive six-class synthetic fixtures;
- automated linting, type checking, tests, coverage, package builds, and
  dependency auditing.

Synthetic sessions demonstrate software behavior. They are not training data,
field validation, or proof of species accuracy.

## Target architecture

```text
Synchronized microphone audio
        |
        +--> continuous session WAVs
        |
        +--> event detector --> event + surrounding context
                                  |
                                  +--> BirdNET bird-species predictions
                                  |
                                  +--> custom broad/non-bird predictions
                                  |
                                  +--> calibrated hybrid decision
                                             |
                          +------------------+------------------+
                          |                  |                  |
                     accepted label      unknown          human review
                          |                  |                  |
                          +------------------+------------------+
                                             |
                              versioned evidence and evaluation
```

BirdNET should be treated as a bird specialist. A separate model should handle
amphibian, insect, mammal, noise, and other target categories. The hybrid layer
must preserve each member's raw output rather than reducing all evidence to one
opaque score.

## Decisions required before model work

Record these decisions in a study specification before downloading or labeling
data:

| Decision | Required definition |
|---|---|
| Deployment region | Country, state, reserve/site, and approximate coordinates |
| Target outputs | Species-level birds, named non-bird taxa, broad groups, or a combination |
| Target species | Explicit inclusion list and scientific-name authority |
| Non-target policy | Human activity, machines, rain, wind, silence, domestic animals, and unfamiliar wildlife |
| Operating conditions | Day/night, seasons, habitat, recorder placement, and expected distance |
| Primary metric | Metric used to select the final model before viewing the test result |
| Error priorities | Whether false alarms or missed detections are more costly for each target |
| Review authority | Who can create and approve ground-truth annotations |
| Distribution policy | Which recordings, labels, models, and derived features may be redistributed |

Do not train a nominal "all wildlife" model without a bounded taxonomy and
deployment region. An undefined negative class makes both training and accuracy
claims unreliable.

## Phase 0 — freeze the study contract

### Actions

1. Create a versioned study specification containing all decisions above.
2. Assign an immutable taxonomy version and stable class identifiers.
3. Define `unknown`, `background`, `noise`, and `unreviewed` separately.
4. Define multi-label behavior for overlapping calls.
5. Select metrics and acceptance thresholds before final testing.
6. Define which site/session/recordist groups are reserved for the final test.

### Deliverables

- `docs/studies/<study-id>.md`
- versioned taxonomy CSV/JSON;
- data-license policy;
- predeclared evaluation protocol.

### Gate

No dataset download or model tuning begins until every required decision has an
owner and a recorded answer.

## Phase 1 — complete the recording experience

### Actions

1. Add a rolling 30–60 second audio player for an active session.
2. Keep event clips as compact evidence, not as the only recording view.
3. Show adjustable event context and the complete finalized WAV for stopped
   sessions.
4. Add event markers to a complete-session timeline.
5. Provide synchronized node switching without changing the selected event or
   time interval.
6. Show explicit states for active, finalizing, finalized, missing, truncated,
   and gap-filled recordings.
7. Avoid repeatedly loading a growing full WAV during dashboard auto-refresh.

### Deliverables

- rolling-audio service/API;
- session timeline component;
- finalized-recording player;
- audio-state and boundary tests.

### Gate

- Active monitoring remains responsive during a long recording.
- Stopped sessions play complete WAVs from every available node.
- Event and context boundaries agree with `start_sample`, `end_sample`, and the
  session sample rate.
- Missing or truncated evidence is displayed, never silently substituted.

## Phase 2 — establish the real-data contract

### Actions

1. Extend the dataset manifest to require:
   - source and stable recording identifier;
   - license and attribution;
   - checksum and byte size;
   - original and converted audio properties;
   - location precision and privacy policy;
   - date/season and recording group;
   - taxonomy version;
   - annotation state and reviewer;
   - permitted training and redistribution uses.
2. Preserve downloaded originals as immutable source evidence.
3. Convert audio into a derived workspace; never replace originals.
4. Detect duplicates by content checksum and source identity.
5. Reject corrupt, clipped, empty, unsupported, or incorrectly labeled files.
6. Record every conversion and resampling operation.

### Deliverables

- versioned manifest schema;
- dataset importer and validator;
- checksum inventory;
- license/attribution report;
- reproducible conversion command.

### Gate

- 100% of accepted recordings have source, license, checksum, taxonomy, and
  group identity.
- No checksum appears in more than one data split.
- Validation reports zero unresolved corrupt or unsupported files.

## Phase 3 — acquire representative data

### Actions

1. Identify public datasets only after the deployment scope is fixed.
2. Audit every source's current license and redistribution conditions.
3. Collect local recordings with the actual microphone hardware.
4. Include target positives, difficult negatives, overlapping calls, and
   realistic background conditions.
5. Sample across sites, devices, seasons, times, weather, distance, and SNR.
6. Retain recordings where no target is present.
7. Keep public-source data and local field data distinguishable in manifests.

BirdNET's official tooling supports location/week context and custom
classifiers. Its training guidance recommends substantial labeled coverage and
explicit background/non-event examples. See the
[BirdNET library](https://github.com/birdnet-team/birdnet) and
[BirdNET Analyzer training documentation](https://github.com/birdnet-team/BirdNET-Analyzer).

### Deliverables

- licensed source registry;
- raw-data inventory;
- class/site/device coverage report;
- documented gaps requiring more collection.

### Gate

Data coverage is reviewed by class and deployment condition. A large clip count
from one recording, location, or individual does not count as diverse coverage.

## Phase 4 — annotate and adjudicate ground truth

### Actions

1. Build an annotation queue from complete recordings and candidate events.
2. Allow event-boundary correction and multi-label annotations.
3. Keep model suggestions visually subordinate to human ground truth.
4. Blind a representative subset so reviewers do not see model predictions.
5. Require two reviewers for the locked evaluation subset where practical.
6. Record disagreement, adjudication, reviewer identity, and taxonomy version.
7. Export annotations without modifying original predictions.

### Deliverables

- annotation UI and portable annotation export;
- reviewer handbook;
- disagreement/adjudication report;
- frozen ground-truth snapshot.

### Gate

- The locked test set has approved labels.
- Ground truth and model predictions remain separate database concepts.
- Inter-reviewer agreement and unresolved disagreement counts are reported.

## Phase 5 — create leakage-safe splits

### Actions

1. Assign entire recording groups to train, validation, or test.
2. Group by the strongest leakage boundary available: source recording,
   session, site/day, individual, and recordist/device.
3. Deduplicate before splitting.
4. Freeze the test split and its checksums.
5. Permit tuning only on training and validation groups.
6. Add an out-of-domain test set from unseen sites or devices.

### Deliverables

- deterministic split manifest;
- leakage audit;
- class and condition balance report;
- locked test checksum.

### Gate

There is zero recording-group overlap across splits. The final test set cannot
be regenerated differently by changing a random seed.

## Phase 6 — benchmark existing classifiers

### Actions

1. Run the heuristic classifier unchanged as the transparent baseline.
2. Run BirdNET with and without geographic/seasonal context.
3. Evaluate confidence thresholds using validation data only.
4. Measure:
   - precision, recall, and F1 per class/species;
   - macro-F1 and balanced accuracy;
   - false positives per recording hour;
   - area under the precision-recall curve;
   - unknown/rejection coverage and error;
   - confidence calibration;
   - latency, memory, and throughput;
   - performance by site, device, SNR, weather, and distance.
5. Preserve predictions, configuration, model identity, and runtime version.

### Deliverables

- baseline evaluation command;
- machine-readable predictions;
- confusion matrices and calibration plots;
- signed baseline report.

### Gate

The baseline report is reproducible from a clean environment. Test-set results
are generated once after threshold/model selection is frozen.

## Phase 7 — train the custom non-bird model

### Recommended first approach

Start with embeddings from a suitable pretrained audio encoder and train a
small calibrated classifier. Compare it against a compact convolutional model
before considering a larger architecture. This reduces data and deployment
risk while establishing a measurable baseline.

### Actions

1. Define fixed-duration training windows and overlap behavior.
2. Use training-only augmentation for gain, background mixing, time shift, and
   limited spectral variation.
3. Include explicit hard negatives and mixed-source examples.
4. Handle class imbalance without duplicating recording groups into other
   splits.
5. Tune only against validation groups.
6. Calibrate probabilities after model selection.
7. Export the candidate model and label map to a portable format such as ONNX
   or TFLite.
8. Record training code version, data manifest hash, hyperparameters, seed,
   metrics, and model checksum.

### Deliverables

- reproducible training command/configuration;
- model artifact and label map;
- training history and provenance card;
- validation comparison against existing baselines.

### Gate

The candidate must improve the predeclared primary metric without creating an
unacceptable regression in critical-class false alarms, rejection behavior, or
runtime performance.

## Phase 8 — build and calibrate hybrid inference

### Actions

1. Preserve BirdNET species scores and custom-model scores independently.
2. Map species predictions to the versioned project taxonomy.
3. Define deterministic fusion rules using validation data.
4. Add an abstention gate for low confidence, low margin, poor SNR, model
   disagreement, or out-of-distribution evidence.
5. Support multiple simultaneous accepted labels when the study contract
   permits it.
6. Avoid interpreting BirdNET's lack of a bird prediction as evidence for a
   non-bird class.
7. Benchmark single-node, best-node, and multi-node score fusion.

### Deliverables

- versioned fusion policy;
- hybrid backend;
- threshold and calibration report;
- failure and fallback tests.

### Gate

Every final label can be traced to its member predictions, thresholds, model
versions, and decision reasons. Runtime failure of an optional model is clearly
recorded as fallback rather than reported as successful model inference.

## Phase 9 — integrate research-grade model output

### Actions

1. Extend persistence without deleting legacy broad classifications.
2. Store species taxonomy ID, common/scientific names, probability, rank,
   member backend, model checksum, and decision policy version.
3. Show prediction, human review, and ground truth as distinct fields.
4. Add species/class filters, confidence filters, and reviewed/unreviewed
   queues.
5. Add confusion matrices, threshold curves, calibration plots, and error
   browsing to Sessions.
6. Export raw member predictions and final hybrid decisions.
7. Add model/data cards to Setup.

### Deliverables

- compatible database migration;
- model inference service;
- evaluation and error-analysis dashboard;
- portable research exports;
- migration and rollback tests.

### Gate

Legacy sessions still open correctly. Every new prediction is reproducible from
preserved audio, configuration, model artifact, and taxonomy.

## Phase 10 — calibrate and validate localization

### Actions

1. Measure microphone coordinates rather than relying on nominal geometry.
2. Run timing-bias calibration with a known source.
3. Use held-out test positions inside and outside the array.
4. Repeat trials across source direction, distance, SNR, and environmental
   conditions.
5. Preserve rejected and failed trials in the denominator.
6. Report success rate, median error, RMSE, 95th-percentile error, signed bias,
   and error versus SNR/location.
7. Add uncertainty estimates and quality filtering to displayed positions.
8. Evaluate whether a fourth node or larger geometry is required.

### Deliverables

- calibration artifact tied to node identities and geometry;
- held-out localization benchmark;
- uncertainty visualization;
- calibration-expiry and mismatch warnings.

### Gate

Deployment-specific localization thresholds are met on held-out physical
trials. Synthetic localization performance is never substituted for this gate.

## Phase 11 — run a controlled field pilot

### Actions

1. Deploy at one bounded site before scaling.
2. Test reconnects, packet loss, missing clock, processing overload, disk
   pressure, power interruption, and graceful recovery.
3. Monitor recording completeness and storage growth.
4. Sample predictions for blinded expert review.
5. Compare public-data performance with local field performance.
6. Document enclosure, power, networking, security, and privacy controls.

### Deliverables

- field-pilot protocol and daily checklist;
- incident and recovery log;
- reviewed false-positive/false-negative sample;
- field performance and storage report.

### Gate

The system completes the planned observation period without silent data loss,
and measured local performance satisfies the predeclared acceptance criteria.

## Phase 12 — release and continuous improvement

### Actions

1. Freeze code, model, taxonomy, configuration, and dataset-manifest versions.
2. Run the complete release gate.
3. Produce a model card, dataset card, calibration report, known-limitations
   statement, and operating guide.
4. Use reviewed production errors to build the next training set.
5. Never train on the locked test set; create a new future test version when
   scope changes.
6. Monitor drift by season, site, device, and background condition.

### Release command

```powershell
scripts\validate_windows.bat
```

### Gate

The release is complete only when software, model, dataset, calibration, and
field-pilot evidence all refer to immutable versions and can be independently
reproduced.

## Recommended implementation order

The first development cycle should execute in this exact order:

1. Complete and approve the Phase 0 study specification.
2. Implement rolling live audio and complete-session event markers.
3. Finalize the dataset manifest and license checks.
4. Build importers for the approved public and local data sources.
5. Extend annotation to boundaries, multi-label review, and adjudication.
6. Freeze leakage-safe train/validation/test groups.
7. Benchmark the unchanged heuristic classifier.
8. Enable and benchmark BirdNET using the selected site context.
9. Train the first custom non-bird baseline.
10. Compare and calibrate the hybrid decision policy.
11. Integrate versioned species/model output into storage and the dashboard.
12. Complete physical localization trials and the controlled field pilot.

## Definition of done

The advanced system is considered complete when:

- the target geography and taxonomy are explicit;
- every training/evaluation recording has valid provenance and licensing;
- train, validation, and test recording groups do not overlap;
- human ground truth is independent from predictions;
- model performance and confidence calibration are reported on locked data;
- unfamiliar or weak evidence can be rejected as `unknown`;
- complete recordings and event context remain reviewable;
- model, taxonomy, dataset, code, and configuration versions are persisted;
- localization passes held-out physical trials;
- the field pilot demonstrates recovery without silent evidence loss;
- the complete software release gate passes.

Until those conditions are met, describe the system as a research prototype and
report model predictions as predictions rather than confirmed wildlife
observations.
