# Wildlife Soundscape — Current Codebase Report

**Audit date:** 2 September 2026  
**Repository version inspected:** `6ec70a8` — *Minor Updates* (31 August 2026)  
**Scope:** the current working tree, including its existing local changes. This report describes what is implemented in code and separately identifies what is simulated, optional, configuration-dependent, or still requires physical validation.

## 1. Executive summary

This repository is a laptop-centred, three-node wildlife-acoustic monitoring prototype. Its main purpose is to receive synchronised microphone audio from three ESP32 nodes, detect multi-node acoustic events, estimate an event position with time-difference-of-arrival (TDOA), derive audio features, classify the sound, persist the evidence, and present live and research-oriented views in a local Streamlit dashboard.

The software path is substantial and currently coherent:

- Three expected nodes (`1`, `2`, `3`) stream 48 kHz mono PCM16 blocks to a TCP receiver.
- The receiver validates Protocol v4 packets, tracks node health and sessions, buffers audio on a common sample timeline, and sends START/STOP/PING controls.
- Multi-node event detection uses adaptive noise thresholds, RMS level, spectral flux, timing agreement, pre/post padding, and a two-node minimum by default.
- Completed events can be localized through band-pass filtering, GCC-PHAT pair delays, calibration offsets, physical-delay checks, and a nonlinear 2-D position solver.
- The selected best microphone channel is preprocessed, feature-extracted, classified, written as per-node event WAV files, and persisted with provenance in SQLite.
- The dashboard reads the persisted SQLite database and WAV files rather than receiving audio directly over a browser socket.
- Analytics produce temporal activity, environmental association, soundscape-index, spatial occupancy/transition, and conservative behaviour-indicator summaries.

The current default classifier is **not species-level**. It is the deterministic heuristic classifier, which produces only `bird`, `insect`, `amphibian`, `mammal`, `noise`, or `unknown`. BirdNET support exists as an optional backend, but its package/model files and the required taxonomy CSV are not bundled or enabled by the default configuration. The repository contains no packaged wildlife recordings, trained weights, database, calibration data, or taxonomy file; its `data/` runtime folders are empty placeholders at audit time.

Software validation completed successfully on this workstation:

| Check | Result |
|---|---|
| Python bytecode compilation | Passed |
| `ruff check .` | Passed — “All checks passed!” |
| `pytest` | **756 passed** in 16.75 s |

This validates the Python implementation and synthetic/integration test coverage. It does **not** validate ESP32 compilation/flashing, Wi-Fi credentials, microphone wiring, shared-clock signal integrity, BME280 readings, acoustic localisation accuracy in a real environment, or biological classification accuracy in the field.

## 2. System status at a glance

| Area | Current status | Important boundary |
|---|---|---|
| Python receiver and pipeline | Implemented and test-covered | Requires the configured dependencies and free local port 5001 |
| Three-node simulation | Implemented and test-covered | Generates synthetic acoustics, not genuine wildlife behaviour or recording conditions |
| ESP32 firmware | Implemented for one master and two slaves | Needs credentials, compilation, flashing, wiring, and bench testing |
| Event detection | Implemented and enabled by default | Thresholds must be tuned against deployment noise |
| TDOA localisation | Implemented | Default triangle is a 1 m lab geometry; physical calibration is disabled |
| DSP/features | Implemented | Feature values are acoustic descriptors, not species proof |
| Default classification | Implemented | Broad acoustic class only; no exact species output |
| BirdNET integration | Implemented as optional | Needs optional installation, model assets, taxonomy mapping, and deployment validation |
| SQLite/WAV persistence | Implemented | Runtime corpus is currently empty |
| Dashboard/graphs | Implemented | Local dashboard; no remote auth, user accounts, or real-time browser transport |
| Research analytics | Implemented | Reports acoustic-event patterns, not abundance, identity, trajectories, or causation |
| External data connectors | Not implemented | No automatic download/API ingestion is currently in the runtime path |

## 3. Repository layout and responsibilities

The repository contains 103 tracked source, firmware, documentation, and launch-script files matching the primary code extensions at audit time, plus the Python virtual environment and generated/cache areas. The principal layout is:

| Path | Responsibility |
|---|---|
| `main.py`, `server.py` | Receiver CLI, asyncio TCP service, node/session control and packet dispatch |
| `protocol.py`, `models.py`, `node.py` | Binary protocol, immutable transport models, per-node state and diagnostics |
| `stream_manager.py` | Per-node sample buffers, aligned windows, environmental lookup, continuous WAV recording helper |
| `event_detector.py`, `event_pipeline.py` | Multi-node acoustic event lifecycle and completed-event processing/persistence |
| `config.py` | Typed, cross-validated application configuration and safe default values |
| `database.py` | SQLite schema, migrations, writes, queries, analytics rows and soundscape-index persistence |
| `dsp/` | PCM conversion, DC removal, filtering, normalisation, STFT/MFCC and feature extraction |
| `localization/` | Band-pass conditioning, GCC-PHAT, TDOA validation and nonlinear location solving |
| `calibration/` | Known-source TDOA observations, robust pair offsets, node-bias fitting and benchmarks |
| `classification/` | Common classifier contract, heuristic backend, optional BirdNET backend/context and ensemble adapter |
| `analytics/` | Activity, environmental, spatial, soundscape-index and conservative behaviour analysis |
| `dashboard/` | Streamlit app, live/research pages, data access, Plotly figures and event-audio rendering |
| `firmware/` | Arduino/ESP32 firmware: `Node_1_Master`, `Node_2_Slave`, `Node_3_Slave` |
| `simulator.py` | Protocol-compatible three-node synthetic acquisition simulator |
| `tools/` | Event and research export, demo-session population, GCC variant benchmark |
| `scripts/` | Windows setup, validation and combined website/receiver/simulator/dashboard launchers |
| `tests/` | Unit, integration, simulator, protocol, dashboard, DSP, localisation and persistence tests |
| `docs/` | Protocol reference, release notes and repository roadmap |
| `data/` | Runtime output location: database, recordings, event WAVs and exports; empty placeholders now |

The package metadata in `pyproject.toml` declares project version `0.1.0`, requires Python 3.13 or newer, and provides installed commands for the receiver, simulator, event export, research export, and GCC benchmark.

## 4. End-to-end design and data flow

```text
ESP32 Node 1 (master) ─┐
ESP32 Node 2 (slave) ──┼─ TCP / Protocol v4 ─> ReceiverServer
ESP32 Node 3 (slave) ─┘                              │
                                                     ▼
                              NodeState + StreamManager common timeline
                                                     │
                         ┌──────────── Event detector ────────────┐
                         │                                         │
                         ▼                                         ▼
                  completed multi-node event              environmental telemetry
                         │                                         │
                         ├─ localisation: filtered GCC-PHAT → TDOA → 2-D solver
                         ├─ DSP: selected channel → features/model waveform
                         ├─ classification: heuristic or optional model
                         ├─ WAV evidence: one synchronised WAV per node
                         └─ SQLite: session, event, telemetry, features, result, indices
                                                                    │
                                                                    ▼
                                        Streamlit dashboard / Plotly / exports
```

The authoritative shared time coordinate is `sampleIndex`, not any individual ESP32 wall clock. The receiver keeps missing regions as gaps rather than compressing the timeline. This is important: preserving discontinuities prevents a network loss from being turned into a false TDOA shift.

## 5. Hardware architecture and acoustic acquisition

### 5.1 Node roles

The physical target is a three-microphone array.

| Node | Role | Main hardware responsibility |
|---|---|---|
| Node 1 | I2S master | Drives shared BCLK/WS, captures its INMP441 microphone, emits the session SYNC pulse and reads BME280 environmental telemetry |
| Node 2 | I2S slave | Captures its local INMP441 using Node 1’s shared I2S clock lines |
| Node 3 | I2S slave | Captures its local INMP441 using Node 1’s shared I2S clock lines |

The firmware uses INMP441 digital I2S MEMS microphones. Node 1 uses BCLK GPIO26, WS GPIO25 and microphone data GPIO33; its BME280 I²C pins are GPIO21/22. Nodes 2 and 3 consume the shared BCLK GPIO26 and WS GPIO25, use a local microphone-data GPIO33, and receive the Node 1 GPIO27 session marker.

GPIO27 is intentionally **not** the TDOA clock. It indicates a controlled session start; the common acoustic sample clock comes from Node 1’s shared BCLK and WS. The firmware arms slaves before the master starts its I2S acquisition and includes I2S/clock/stuck-SYNC health diagnostics.

### 5.2 Audio contract

Default laptop and firmware contract:

| Setting | Default |
|---|---:|
| Sample rate | 48,000 Hz |
| Channels | 1 mono per node |
| Sample representation | signed PCM16 little-endian |
| Frames per network audio block | 1,024 |
| Nominal block duration | 21.33 ms |
| Raw audio block payload | 2,048 bytes |
| Laptop retention per node | 20 s, with a derived block buffer margin |
| Expected node IDs | 1, 2, 3 |

The default localisation geometry is a 1 m equilateral triangle: Node 1 `(0, 0)`, Node 2 `(0.5, 0.8660254)`, Node 3 `(1, 0)`. It is explicitly a lab starting point and must be replaced with measured microphone coordinates before field conclusions are drawn.

### 5.3 Firmware operational boundaries

The three sketches implement reconnect handling, bounded queues, TCP write-stall handling, CRC packet output, heartbeat/health data, and START/STOP/PING controls. Wi-Fi credentials are still template values such as `YOUR_WIFI_PASSWORD`; the firmware cannot join a real network until deployment values are supplied. The audit did not compile or flash these sketches, so their toolchain compatibility and hardware timing must be verified separately.

## 6. Communication, session control and health signals

### 6.1 Protocol v4

`docs/protocol.md` is the canonical interface between firmware v1.4.0 and the Python receiver. ESP32-to-laptop packets use a fixed 40-byte little-endian header:

| Header field | Meaning |
|---|---|
| `magic` | `0x574C5343` (`WLSC`) |
| `protocolVersion` | `4` |
| `nodeId`, `packetType`, `flags` | Source, semantic type and health flags |
| `sequence` | Network/ordering diagnostics |
| `sessionId` | Laptop-owned acquisition session |
| `sampleIndex` | Common coarse PCM timeline |
| `localMicros` | Node-local diagnostic clock only; never cross-node subtracted |
| `i2sErrorCount` | Cumulative capture-error diagnostic |
| `payloadLength`, `payloadCRC32` | Bounded body framing and integrity checking |

Packet types are `HELLO`, `AUDIO`, `ENVIRONMENT`, `HEARTBEAT`, and `SYNC`. Packet-health bits identify clipping, slave clock faults, and queue congestion. Payload CRC32, exact payload parsing, header validation, session acceptance checks, and per-node sequence tracking are implemented on the laptop side.

Laptop-to-node controls use a compact 8-byte control frame (`magic 0xC0DE`, control version 1) with `START`, `STOP`, and `PING`. The receiver generates a session ID, checks that the expected nodes are connected, sends controls, and rolls back a failed start. It rejects stale/wrong-session data rather than mixing sessions.

### 6.2 Local networking

The receiver defaults to `0.0.0.0:5001`, a 10 s read timeout, a 10 s HELLO timeout, and a 64 KiB maximum payload. Connections run through asyncio. The default local dashboard is Streamlit on port `8501`.

There is no TLS, device authentication, signed firmware, encrypted transport, role-based dashboard access, cloud relay, or multi-user tenancy in the current implementation. It is therefore suitable for a controlled lab/LAN prototype, not an exposed production service.

### 6.3 What “live” means here

There are three distinct live-like modes, which should not be conflated:

1. **Physical live acquisition:** flashed nodes connect to the local TCP receiver and stream real microphone/telemetry data. This is implemented in software but not hardware-validated by this audit.
2. **Synthetic live demo:** `simulator.py` acts as three protocol-compatible nodes, giving the receiver/dashboard a changing stream without hardware.
3. **Dashboard refresh:** Streamlit polls persisted local data on a default two-second refresh interval. It is not a direct browser audio stream and does not subscribe to a websocket.

## 7. Runtime pipeline

### 7.1 Buffering and alignment

`NodeState` records stream/session state, sequences, health counters, packet flags, environmental samples and connection diagnostics. `StreamManager` retains PCM blocks per node, can return aligned blocks and exact time windows, locates nearby environmental data, and preserves sample-index gaps. `WavRecorder` provides a continuous per-node recording facility in addition to the event snippets written by the pipeline.

### 7.2 Event detection

`MultiNodeEventDetector` combines per-node block detections into a single acoustic event. The defaults are deliberately dynamic rather than a fixed amplitude threshold:

| Mechanism | Default |
|---|---:|
| Noise-history length | 96 blocks |
| Background estimate | 30th percentile |
| Initial noise floor | -52 dBFS |
| Trigger / release margin | +8 / +4 dB |
| Minimum spectral flux | 0.06 |
| Strong-energy override | +15 dB |
| Attack / release blocks | 1 / 2 |
| Required agreeing nodes | 2 |
| Event duration range | 30 ms to 12 s |
| Context saved around event | 0.50 s pre-pad, 0.75 s post-pad |

The detector state changes as new blocks update the background model and as nodes agree or release. This is a useful adaptive mechanism, but its operating point needs false-positive/false-negative measurement for each habitat, microphone gain, season and weather condition.

### 7.3 Localisation

For an event, the pipeline obtains the aligned multi-node window, optionally uses the environmental reading nearest to the midpoint to estimate speed of sound, conditions audio with a 200–12,000 Hz fourth-order bandpass, then estimates pairwise delays through GCC-PHAT.

Default localisation parameters are:

| Setting | Default |
|---|---:|
| Window length | 8,192 samples |
| GCC interpolation | 8× |
| GCC beta | 1.0 (PHAT) |
| Coarse alignment search | ±10 samples |
| Minimum peak ratio | 1.10 |
| Minimum RMS | 50 |
| Nominal speed of sound | 343 m/s |
| Environmental compensation | enabled |
| Calibration offsets | disabled until measured |
| Position bounds | unconstrained by default |

The engine rejects invalid/physically impossible delays, applies optional pair timing and node-bias calibration, and uses a nonlinear solver to estimate an `(x_m, y_m)` point plus residual/quality information. A location is an acoustic source estimate, not a confirmed animal coordinate. Reverberation, multiple simultaneous calls, poor geometry, microphone mismatch and unmet common-clock assumptions can make estimates wrong or unavailable.

### 7.4 DSP and feature extraction

Raw PCM evidence is retained. The selected best microphone channel is independently processed for analysis/model input using DC removal, a 100–16,000 Hz fourth-order band-pass filter, peak normalisation for model input, and an STFT (`n_fft=2048`, hop `512`). The system obtains duration, RMS, peak amplitude, crest factor, zero-crossing rate, SNR, dominant frequency, spectral centroid/bandwidth/rolloff/flatness/flux, and 13-MFCC mean and standard-deviation vectors. These features support broad classification, visualisation, quality review and later research exports.

### 7.5 Classification and current animal markings

The default `heuristic` backend is active and produces exactly these project-level labels:

| Current label | Meaning |
|---|---|
| `bird` | Broad bird-like acoustic pattern |
| `insect` | Broad insect-like acoustic pattern |
| `amphibian` | Broad amphibian-like acoustic pattern |
| `mammal` | Broad mammal-like acoustic pattern |
| `noise` | Non-biological/ambiguous environmental or anthropogenic sound |
| `unknown` | Insufficient/ambiguous evidence or safe abstention |

Therefore **lion, tiger, bear, leopard, jaguar, and exact bird species are not currently marked as individual classes in normal operation**. There is no individual-animal recognition, sex/age/call-type annotation, abundance estimator, or cross-event identity tracking.

The classifier framework does include optional `birdnet` and `ensemble` paths. BirdNET receives a normalised event waveform, retains top species scores in the result metadata, and can use geographic/seasonal priors. A structured taxonomy CSV maps exact model labels to the project’s broad groups (`Aves`, `Insecta`, `Amphibia`, `Mammalia`, `Noise`). Unknown or missing taxonomy must safely abstain rather than silently claim a bird. Defaults are intentionally conservative: BirdNET is not installed by the base dependency set, `backend="heuristic"`, no model/taxonomy path is configured, geo filtering is disabled, and the BirdNET confidence threshold would be 0.20 if enabled.

### 7.6 Event transaction and evidence trail

The completed-event pipeline is intentionally failure-tolerant:

1. confirm that the event belongs to the active session;
2. look up midpoint environmental telemetry;
3. attempt localisation;
4. choose the best channel and derive DSP/model data;
5. attempt classification;
6. write one synchronised `node_1.wav`, `node_2.wav`, `node_3.wav` event file where available;
7. persist the core event; then independently persist features and classification;
8. update runtime status and logs.

If localisation, feature extraction, classification, or an individual WAV write fails, the core event can still remain in the database. This is the correct resilience policy for monitoring, provided failed stages are surfaced in operations/quality reports.

## 8. Persistence, recordings and export

SQLite at `data/database/events.db` is created at runtime. The configured event directory is `data/events`; continuous recordings use `data/recordings`; exports use `data/exports`. At audit time the supplied `data/` folders contained only zero-byte `.gitkeep` placeholders, so no wildlife corpus, field database, event WAV, calibration result or pre-existing observation record is included.

### 8.1 Database schema

| Table | Stored information |
|---|---|
| `sessions` | Laptop-owned ID, label, start and stop timestamps |
| `telemetry` | Node ID, sample index, temperature, humidity, pressure and insertion time |
| `events` | Detector ID, session/sample range, trigger nodes, peak level, selected node, environment, speed of sound, optional position/residual/success flag, event directory and creation time |
| `event_features` | One selected-channel feature row per event, including time/spectral descriptors, SNR and MFCC JSON |
| `classifications` | Label, primary/secondary confidence, margin, score/reason JSON, classifier name/version |
| `soundscape_indices` | Per-node time window, ACI, NDSI, entropy values, bioacoustic/anthrophony/biophony powers and parameter JSON |

The database initialiser includes indexes for time/session/node/event query paths and conservative schema migration logic. Query methods support recent events, event details, sessions, telemetry, analytics rows/bins and soundscape-index retrieval.

### 8.2 Timestamp policy

Database creation timestamps say when a row was written. Scientific/acoustic timing is anchored to `sessionId + sampleIndex + sampleRate`, allowing an event’s position in the shared capture timeline to be reconstructed even when asynchronous packet arrival differs. This distinction is essential for reproducible TDOA and activity analysis.

### 8.3 Export and reproducibility support

Packaged CLI tools include:

- `wildlife-export-events` for event data;
- `wildlife-export-research` for derived research metrics;
- `wildlife-benchmark-gcc` for GCC variant evaluation;
- `tools/populate_demo_session.py` for dashboard/demo data;
- calibration modules for known-source timing fits and localisation benchmarks.

Raw event WAVs, detector event IDs, session/sample ranges, feature settings, classifier name/version, score/reason JSON, geometry and calibration configuration should travel together in any serious research export. The present code stores much of that provenance, but a formal experiment manifest and data-version registry are still recommended.

## 9. Analytics, graphs, spatial “tracks” and dynamic views

### 9.1 Temporal and class activity

Persisted event rows are bucketed by time (default one-hour buckets) to produce event counts, active duration, mean confidence, dominant broad class, class totals and peak-activity hour. This describes detected acoustic activity; it is not direct abundance or population density.

### 9.2 Environmental association

Regular environment/activity bins, including zero-event intervals, are analysed with Spearman rank associations. Defaults use node 1 telemetry, alpha `0.05`, a minimum five samples, and a neutral coefficient threshold of `0.05`. The implementation and dashboard warn that association is not causation and that event-only rows are inadequate for this analysis because silent periods would be missing.

### 9.3 Continuous ecoacoustic soundscape indices

The `SoundscapeService` can aggregate streaming node audio into 60-second windows (enabled by default) and persist:

- **ACI** — Acoustic Complexity Index;
- **NDSI** — Normalized Difference Soundscape Index, comparing configured anthrophony (1–2 kHz) and biophony (2–8 kHz) bands;
- acoustic, temporal and spectral entropy;
- bioacoustic index;
- anthrophony and biophony power.

These are useful soundscape descriptors and trend indicators, but do not directly identify species or prove biological health without a study design and ground truth.

### 9.4 Spatial occupancy, hotspot and transitions

Successful position estimates can be binned into a 0.25 m occupancy grid (up to 10,000 cells). Analytics report localisation coverage, occupied cells, a hotspot, spatial-distribution entropy, and ordered transitions between consecutive event cells. Transitions can be limited to a 300 s gap and optionally constrained to same-class events.

These are **not animal tracks**. A transition represents a change between consecutive localised acoustic events; it must not be presented as an individual animal trajectory, home range, or confirmed movement path. There is no data-association layer that says two calls came from the same animal, and no multi-target tracker.

### 9.5 Conservative behaviour indicators

The behaviour subsystem derives status/score indicators only after minimum data requirements (default five activity events, five localised events, three transitions). Its model classes and dashboard deliberately label results as conservative acoustic indicators rather than confirmed ethological behaviour. This is an appropriate design choice and should be retained when adding species-level models.

## 10. Dashboard, visual design and interaction

`dashboard/app.py` starts the local Streamlit application titled **Wildlife Soundscape Monitor**. Presentation configuration is local-only and defaults to automatic refresh every 2 seconds, 50 recent events, 100 listed sessions, 5,000 maximum plot points and 30-second audio previews.

### 10.1 Live Monitor

The live view surfaces receiver/node status, session and database state, connection/stream health, I2S/clock/congestion/clipping diagnostics, recent event rows, event/localisation summaries and live Plotly views. The sidebar includes system/configuration/session information and navigation between live and research modes. It is a read/presentation layer over runtime state and persisted records; it does not control a remote authenticated fleet.

### 10.2 Research Analysis

The research view selects a persisted session and renders structured warnings alongside:

- event-count, active-duration, mean-duration, class-count, peak-hour and localisation-coverage metrics;
- temporal activity timeline and broad-class distribution;
- environmental time series and environmental-association matrix/table;
- continuous soundscape-index timeline and index summaries;
- localisation scatter and occupancy heatmap;
- spatial transition/behaviour-indicator views;
- event evidence chooser, audio player, selected-node/event summary, waveform and spectrogram;
- downloadable session/research output.

The audio view discovers saved event WAVs, extracts mono PCM16 metadata/samples, decimates waveform points where needed, creates a preview WAV in memory and uses Plotly for waveform/spectrogram images. Visualisations make persistent evidence inspectable rather than merely displaying model labels.

### 10.3 Design constraints

The user interface favours local research inspection over a polished public product: no central design-token system, no remote deployment configuration, no user/role model, no multilingual UI, no adaptive mobile workflow, and no accessibility audit artefacts were found. Streamlit and Plotly provide a functional, data-rich interface; a production field portal would need a separate UX/accessibility/security pass.

## 11. Simulation and line-wise Windows launch workflow

### 11.1 Simulator model

`simulator.py` implements three fake nodes that speak the same protocol and participate in the same session-control path as hardware. It has shared geometry, a synthetic source, distance-based delay/gain, a shared simulated sample timeline, master/slave arming and clock gates, HELLO/AUDIO/telemetry/SYNC/heartbeat packets, generated waveform events and commands. It gives the receiver, database, pipeline and dashboard changing test data without ESP32 devices.

The simulator is valuable for software integration and repeatable localisation tests, but is not a substitute for field validation. It cannot reproduce microphone frequency response, wind/rain, wildlife call diversity, vegetation propagation, reverberation, RF packet loss patterns, sensor drift, animal movement/overlap, true temperature gradients, or labelling uncertainty.

### 11.2 One-folder launch files

All Windows operational wrappers are in `scripts/`:

| File | Purpose |
|---|---|
| `setup_windows.bat` | Create/install the local environment |
| `validate_windows.bat` | Compile/lint/test validation wrapper |
| `run_receiver.bat` | Start receiver only |
| `run_simulator.bat` | Start simulator only |
| `run_dashboard.bat` | Start Streamlit only |
| `run_live.bat` | Physical-node workflow |
| `run_demo.bat` | Three-node simulation workflow |
| `run_website.bat` | Batch wrapper for combined website startup |
| `run_website.ps1` | Combined orchestrator with port/process startup checks |
| `_common.bat` | Shared script environment helpers |

`run_website.ps1` is the most complete orchestrator. In `demo` mode it starts the receiver with automatic acquisition, waits for TCP port 5001, starts the simulator, starts Streamlit, waits for dashboard port 8501, and cleans up processes if startup fails. In `live` mode it starts the receiver/dashboard and waits for physically connected nodes. The script verifies that ports can be opened, but it does not prove audio integrity, node count, model inference, dashboard content, or physical localisation accuracy. Once child windows are started successfully, they run independently; operators must still monitor receiver logs and stop the children safely.

Suggested operator order is therefore:

1. run `scripts\setup_windows.bat` once after a clean checkout;
2. run `scripts\validate_windows.bat` after code/dependency changes;
3. run `scripts\run_website.bat demo` for a no-hardware end-to-end smoke workflow;
4. inspect receiver logs, SQLite events and dashboard content;
5. only then use `scripts\run_website.bat live` after physical setup, measured geometry and calibration.

## 12. Dataset and model inventory

### 12.1 What the repository currently includes

It includes code, tests, simulator-generated waveforms and empty runtime directories. It does **not** include:

- a labelled animal-audio training dataset;
- real field recordings;
- an SQLite event database;
- BirdNET/Perch/custom trained model weights;
- a BirdNET labels/taxonomy CSV;
- a curated lion/tiger/bear/bird species library;
- ground-truth localisation/calibration measurements;
- an external live-data API/client.

As a result, this codebase is ready to receive/generate and store data, but it is not yet provisioned as a species library or trained field-recognition product.

### 12.2 Free or openly accessible sources worth evaluating

The following sources are candidates for a legal, versioned data/model acquisition plan. “Free” does not mean unrestricted; each source’s licence, attribution, redistribution, non-commercial and API terms must be honoured.

| Source | Value for this project | Important licence/operational note |
|---|---|---|
| [BirdNET V3](https://zenodo.org/records/20703646) | Optional ONNX-capable recognition model with more than 11,000 biological/acoustic classes; useful first model for bird-heavy deployments and some non-bird labels | Model releases/labels change; model is CC BY-SA 4.0 and project code is MIT; its own release notes say taxonomy cleanup is ongoing and it lacks a dedicated noise fallback |
| [BirdNET taxonomy](https://birdnet.cornell.edu/taxonomy/download) | Versioned scientific/common-name metadata, external IDs and multilingual names for a durable species registry | Use as a local versioned import, not an unverified name list |
| [Google Perch V2](https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2) | Broad 15k-class bioacoustic model with 1,536-d embeddings and non-avian taxa; particularly attractive for a custom classifier library | Apache-2.0 model; logits can be uncalibrated for rare/unbalanced taxa—calibration and local validation remain necessary |
| [iNatSounds](https://cvl-umass.github.io/iNatSounds/) | 230k recordings across 5.5k species and several taxa; includes birds, insects, amphibians, mammals and reptiles | Research/education, non-commercial use; do not redistribute audio; respect per-record licence metadata |
| [Xeno-canto birds through GBIF](https://www.gbif.org/dataset/b1047888-ae52-4179-9dd5-5448ea342a24) | Continuously maintained global bird sound observations | GBIF subset is CC BY-NC 4.0; preserve creator/recording attribution and source licence |
| [Xeno-canto land mammals through GBIF](https://www.gbif.org/dataset/80f94317-a61f-477f-af06-037632ad6e3b) and [frogs](https://www.gbif.org/dataset/bcf8d1fc-6bf0-4f57-9076-7d9ae2828ec2) | Useful extensions beyond birds | Non-commercial terms and source attribution apply; normalise taxonomy/quality before training |
| [BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet) | Large benchmark/evaluation collection (6,800+ h, nearly 10k bird classes) | CC BY-NC 4.0; particularly useful for evaluation rather than a single operational library |
| [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026) | Recent multi-taxon soundscape benchmark for five-second probability windows | Access and use are governed by the competition/data terms; do not make it an undisclosed production corpus |
| [BEANS](https://github.com/earthspecies/beans) | Unified benchmark suite spanning birds, terrestrial/marine mammals, frogs and insects | Around 300 GB full scale; individual source licences still control the data |
| [Animal Sound Archive](https://www.museumfuernaturkunde.berlin/en/research/collection/animal-sound-archive/) | Collection of roughly 120k recordings across birds, mammals and other taxa | Check per-record availability/licensing; many recordings are zoo/experimental rather than wild habitat |
| [Macaulay Library](https://www.macaulaylibrary.org/) | Very large living archive of birds, mammals and amphibians, excellent reference material | Listening is broadly available; downloads/research use have request/licence restrictions—see [media policy](https://support.ebird.org/en/support/solutions/articles/48001064551-using-and-requesting-media/) |
| [NatureLM-audio datasets](https://projects.earthspecies.org/naturelm-audio/datasets.html) and [alp-data](https://github.com/earthspecies/alp-data) | Large multimodal corpus references and a licence-aware dataset interface | Diverse sources and licences; select subsets deliberately rather than bulk ingesting |
| [AudioSet](https://research.google.com/audioset/) | Human-labelled environmental/noise/hard-negative clips | Useful to improve `noise` and false-positive handling, not a replacement for exact wildlife taxonomy |
| [iNaturalist sound observation search](https://www.inaturalist.org/pages/search%2Burls) | Queryable occurrence/sound metadata and links for selected taxa/locations | The API has rate/volume limits; use authorised exports/caches, not bulk scraping |

There is no universally free, continuously streaming, globally complete wildlife-audio feed that should be treated as real-time ground truth. The most “live” practical sources are maintained catalogues and observation APIs. For this system, external data should first become a controlled local reference/training/evaluation library with licences, source IDs, labels, dates, geographies and splits recorded; it should not be blindly mixed into field events.

### 12.3 Requested species examples and realistic coverage path

| Requested group | Current default system | Recommended model/library path |
|---|---|---|
| Exact bird species | Not represented; only `bird` | BirdNET V3 plus a versioned BirdNET taxonomy, and local validation/evaluation with BirdSet/Xeno-canto/BirdCLEF subsets relevant to the deployment region |
| Lion — *Panthera leo* | Not represented; `mammal` at most | Perch V2 label inventory includes *Panthera leo*; BirdNET V3 label inventory includes it in current releases; validate against licensed lion vocalisations before marking detections as confirmed |
| Tiger — *Panthera tigris* | Not represented | Perch V2 label inventory includes *Panthera tigris*; create a target-species calibration/evaluation set rather than assuming a broad mammal detector can identify it |
| Leopard — *Panthera pardus* | Not represented | Perch V2 includes it; use target recordings and region-specific hard negatives |
| Jaguar — *Panthera onca* | Not represented | Perch V2 and BirdNET V3 labels include it; deployment geography is essential |
| Snow leopard — *Panthera uncia* | Not represented | Perch V2 includes it; likely sparse data, so use an abstention threshold and human review |
| Bears | Not represented | Perch V2 lists giant panda, sun bear, sloth bear, spectacled bear, American black bear, brown bear, polar bear and Asian black bear; current BirdNET V3 labels include at least giant panda and American black bear. Treat label availability as a starting point, not accuracy evidence |

## 13. Limitations and risk register

1. **No field validation yet:** passing software tests cannot establish detection recall/precision, TDOA error, model accuracy or environmental robustness.
2. **Default model is broad-class heuristic:** exact animal/species claims are unavailable by design in the current configuration.
3. **No local training/evaluation corpus:** the empty `data/` directories make results non-reproducible until a governed corpus and field dataset are collected.
4. **Three-node 2-D geometry:** height is not estimated; small arrays are geometry-sensitive and reverberation can dominate delay estimates.
5. **No individual identity or tracking:** “tracks” are aggregate transitions between localised events only.
6. **Single-source assumptions:** overlapping calls and multi-animal scenes can confuse event segmentation, correlation and classification.
7. **Wi-Fi/TCP dependency:** congestion, disconnects and packet loss can make evidence unavailable; the code diagnoses these but cannot remove RF risk.
8. **Hardware deployment gaps:** credentials, power, enclosure/weatherproofing, clock wiring, microphone placement/gain consistency and BME280 performance still need engineering validation.
9. **Security/privacy gaps:** local TCP and dashboard lack encryption/authentication; audio may capture people or sensitive locations.
10. **Scientific interpretation boundaries:** model confidence is not biological truth; associations are not causation; acoustic activity is not abundance.

## 14. Recommendations for upgrades and updates

### Priority 0 — make the existing system field-verifiable

1. **Perform and archive a hardware acceptance test.** Compile/flash all three sketches, measure BCLK/WS/SYNC with a logic analyser, verify shared sample alignment, test START/STOP/reconnect/queue congestion, and record microphone gain/noise-floor checks. Add a signed checklist and test artefacts under `data/calibration/` or a dedicated experiment folder.
2. **Measure real geometry and calibrate TDOA.** Replace the default triangle coordinates, run known-source tests across the expected monitoring area, fit/store calibration offsets/node biases, and report median/95th-percentile localisation error by distance and SNR. Do not enable or advertise spatial analytics as animal locations before this work.
3. **Run a controlled end-to-end demo each release.** Use `run_website.bat demo`, confirm receiver → simulated nodes → event DB/WAV → dashboard manually, and make that flow a documented release check in addition to the existing 756 automated tests.
4. **Move Wi-Fi secrets out of firmware source.** Use ignored local configuration/provisioning, document credential rotation, and never commit live SSIDs/passwords. Add a firmware build configuration profile for the laptop IP and deployment site.

### Priority 1 — turn broad classes into defensible species results

5. **Build a versioned species registry.** Add first-class `taxa` and `event_taxon_predictions` tables rather than storing species only inside a JSON score map. Record scientific name, accepted taxon ID, common names, rank, source taxonomy version, source model/version, confidence, threshold, geographic prior and review state.
6. **Add a pluggable embedding/model layer, starting with Perch V2 and BirdNET V3.** Keep BirdNET for broad deployment support; evaluate Perch V2 for the requested big-cat/bear coverage and for custom classifiers on embeddings. Pin model/label hashes and licenses in a model registry. Do not assume label presence means sufficient precision/recall.
7. **Create a target-species library with hard negatives.** For each deployment region, collect/licence positive calls and confusable background/non-target calls. Split by recorder/site/time to prevent leakage. Include lion, tiger, leopard, jaguar and bear examples only where ecologically relevant; use `unknown` and human review for low-confidence output.
8. **Add prediction calibration and human review.** Calibrate probabilities per model/species, set abstention thresholds, surface top-k species/audio/spectrogram/provenance for review, and store reviewer decision/notes. Report precision-recall, false-alarm rate and per-species coverage—not accuracy alone.
9. **Use occurrence metadata as a prior, not a label.** Import a frozen eBird/iNaturalist/GBIF occurrence snapshot for the site/season and preserve its version. Keep the acoustic score and geographic prior separately visible to avoid circular inference.

### Priority 2 — improve live operation and data quality

10. **Add operational observability.** Persist receiver/node disconnects, packet loss/sequence gaps, clock faults, queue congestion, CPU/memory, process health and dashboard staleness. Expose a clear health page and alerting threshold rather than relying on terminal windows.
11. **Harden sessions and recovery.** Add automatic child-process supervision/restart, graceful shutdown, daily/session rotation, disk-space checks, database backups, WAL checkpoint/retention policies and recovery testing for abrupt power/network loss.
12. **Strengthen event detection.** Add site profiles, scheduled adaptive threshold retuning, labelled false-positive review, source-separation/multi-event handling experiments, and a configurable detector evaluation report. Avoid silent parameter changes by saving the detector configuration with each session.
13. **Improve localisation quality control.** Store each pair’s raw/corrected delay, peak ratio, retained/rejected reasons, geometry condition metric and uncertainty ellipse. Consider four or more nodes and 3-D localisation only after the three-node calibrated baseline is measured.
14. **Expand environmental sensing carefully.** Add calibrated battery/power status, microphone/self-noise diagnostics and optional wind/rain/light sensors. Use environmental readings to contextualise quality and speed of sound, not to make causal biological claims.

### Priority 3 — data product, dashboard and research quality

15. **Add a governed data-ingestion pipeline.** Create a manifest-driven importer for selected external datasets with source URL/ID, licence, attribution, checksum, taxonomy mapping, sample rate, annotation confidence, region/date, train/validation/test split and deletion/retention policy. Never bulk scrape external sources or redistribute audio where prohibited.
16. **Make “tracks” scientifically honest in the UI.** Rename them to “acoustic-location transitions” by default, show their limits beside every plot, and add actual multi-target tracking only if identity/data-association assumptions can be validated.
17. **Upgrade dashboard UX.** Add time/site/species/confidence/SNR/localisation-quality filters, a map/image overlay with measured microphone positions, calibration state badges, data-quality tooltips, CSV/GeoJSON/WAV evidence bundles, keyboard/accessibility checks and responsive layouts.
18. **Add test/CI artefacts.** Run the existing compile/lint/pytest suite on every change; add firmware compile checks, simulator-to-dashboard smoke tests, database migration tests against historical files, package-install tests and a coverage trend. Preserve benchmark datasets/results as release artefacts.
19. **Secure deployment before leaving a controlled LAN.** Add device authentication, TLS or a protected relay/VPN, dashboard authentication/authorization, audit logs, encryption-at-rest policy and a privacy/location data-retention assessment.

## 15. Recommended implementation sequence

The most effective next sequence is:

1. validate the physical three-node array and archive calibration/bench results;
2. collect a small, licensed, site-relevant field dataset with ground truth;
3. add the taxa/prediction schema and versioned taxonomy/model registry;
4. benchmark BirdNET V3 and Perch V2 against the curated field dataset, including `unknown`/noise performance;
5. deploy only calibrated species outputs with review and abstention;
6. then invest in resilience, external-data ingestion, dashboard polish, security and larger field deployments.

That order keeps the existing system’s strongest property intact: it preserves real acoustic evidence and explicit uncertainty instead of converting incomplete data into confident wildlife claims.
