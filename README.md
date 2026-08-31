# Wildlife Soundscape Mapping & Behavior Analysis System

### A Synchronized Multi-Node IoT Acoustic Monitoring Platform for Wildlife Detection, Classification, Localization, Environmental Association, and Spatial Behavior Analysis

---

## Project Status

> **Research Prototype / Lab-Scale System**  
> **Firmware baseline:** v1.4.0  
> **Binary protocol:** Protocol v4  
> **Acquisition:** 48 kHz, mono PCM16 per node  
> **Reference deployment:** 3 synchronized ESP32 acoustic nodes  
> **Processing architecture:** Edge acquisition + laptop-side DSP/AI/localization/analytics  
> **Current validation state:** Final cross-module software, firmware, hardware, and experimental validation is still pending.

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Why This System Is Needed](#3-why-this-system-is-needed)
4. [Project Objectives](#4-project-objectives)
5. [Key Engineering Contributions](#5-key-engineering-contributions)
6. [Why Passive Acoustic Monitoring](#6-why-passive-acoustic-monitoring)
7. [Comparison With Other Wildlife Monitoring Methods](#7-comparison-with-other-wildlife-monitoring-methods)
8. [System Architecture](#8-system-architecture)
9. [Reference Hardware Architecture](#9-reference-hardware-architecture)
10. [Acoustic Node Roles](#10-acoustic-node-roles)
11. [Pin Mapping](#11-pin-mapping)
12. [I2S Synchronization Architecture](#12-i2s-synchronization-architecture)
13. [GPIO27 SYNC Signal](#13-gpio27-sync-signal)
14. [Session and Sample Timeline](#14-session-and-sample-timeline)
15. [Firmware Architecture](#15-firmware-architecture)
16. [Sensor-Abstraction Strategy](#16-sensor-abstraction-strategy)
17. [Protocol v4](#17-protocol-v4)
18. [Network and Bandwidth Model](#18-network-and-bandwidth-model)
19. [Laptop Software Architecture](#19-laptop-software-architecture)
20. [Acoustic Event Detection](#20-acoustic-event-detection)
21. [DSP Pipeline](#21-dsp-pipeline)
22. [Acoustic Feature Extraction](#22-acoustic-feature-extraction)
23. [Classification Architecture](#23-classification-architecture)
24. [Heuristic Classifier](#24-heuristic-classifier)
25. [BirdNET Integration](#25-birdnet-integration)
26. [Ensemble Classification](#26-ensemble-classification)
27. [TDOA Localization](#27-tdoa-localization)
28. [GCC-PHAT](#28-gcc-phat)
29. [Physical Delay Constraints](#29-physical-delay-constraints)
30. [Environmental Speed-of-Sound Compensation](#30-environmental-speed-of-sound-compensation)
31. [Nonlinear Position Solver](#31-nonlinear-position-solver)
32. [TDOA Calibration](#32-tdoa-calibration)
33. [Localization Benchmarking](#33-localization-benchmarking)
34. [Environmental Monitoring](#34-environmental-monitoring)
35. [Persistence and Database Architecture](#35-persistence-and-database-architecture)
36. [Scientific Timestamp Policy](#36-scientific-timestamp-policy)
37. [Research Analytics](#37-research-analytics)
38. [Temporal Activity Analysis](#38-temporal-activity-analysis)
39. [Environmental Association Analysis](#39-environmental-association-analysis)
40. [Spatial Analysis](#40-spatial-analysis)
41. [Behavior Indicators](#41-behavior-indicators)
42. [Dashboard](#42-dashboard)
43. [Research Export Tools](#43-research-export-tools)
44. [Repository Structure](#44-repository-structure)
45. [Software Requirements](#45-software-requirements)
46. [Python Installation](#46-python-installation)
47. [Optional BirdNET Installation](#47-optional-birdnet-installation)
48. [ESP32 Firmware Requirements](#48-esp32-firmware-requirements)
49. [Physical Wiring Guidelines](#49-physical-wiring-guidelines)
50. [Firmware Configuration](#50-firmware-configuration)
51. [Running Without Hardware](#51-running-without-hardware)
52. [Running With Hardware](#52-running-with-hardware)
53. [Dashboard Launch](#53-dashboard-launch)
54. [Calibration Procedure](#54-calibration-procedure)
55. [Localization Benchmark Procedure](#55-localization-benchmark-procedure)
56. [Recommended Experimental Methodology](#56-recommended-experimental-methodology)
57. [Evaluation Metrics](#57-evaluation-metrics)
58. [Testing Strategy](#58-testing-strategy)
59. [Reliability and Fault Handling](#59-reliability-and-fault-handling)
60. [Security Considerations](#60-security-considerations)
61. [Current Limitations](#61-current-limitations)
62. [Real-World Deployment Considerations](#62-real-world-deployment-considerations)
63. [Research-Paper Opportunities](#63-research-paper-opportunities)
64. [Potential Product and Patent Directions](#64-potential-product-and-patent-directions)
65. [Future Development](#65-future-development)
66. [Troubleshooting](#66-troubleshooting)
67. [Reproducibility Principles](#67-reproducibility-principles)
68. [Scientific Interpretation Limits](#68-scientific-interpretation-limits)
69. [Current Development Status](#69-current-development-status)
70. [Citation and Licensing](#70-citation-and-licensing)

---

# 1. Project Overview

The **Wildlife Soundscape Mapping & Behavior Analysis System** is a multi-node Internet of Things and bioacoustic research platform designed to:

- continuously acquire wildlife soundscapes,
- detect acoustically significant events,
- extract signal-processing features,
- classify broad wildlife sound categories,
- optionally perform species-oriented inference using BirdNET,
- estimate the spatial origin of sounds using TDOA,
- associate detected activity with environmental conditions,
- map acoustic activity spatially,
- generate conservative behavior-related indicators,
- store observations in a structured database,
- visualize the system through a local research dashboard,
- export datasets and research metrics for external analysis.

The reference implementation uses **three synchronized ESP32 acoustic nodes** connected to a laptop over Wi-Fi.

Unlike a conventional wireless microphone network, the three microphones do not operate from independent sampling clocks.

Node 1 distributes the I2S:

- Bit Clock (`BCLK`)
- Word Select / Left-Right Clock (`WS/LRCLK`)

to Nodes 2 and 3.

This allows all microphones to operate from the same sampling-clock domain, which is fundamental for meaningful **Time Difference of Arrival (TDOA)** estimation.

The ESP32 nodes primarily perform:

- acquisition,
- buffering,
- health monitoring,
- environmental sensing,
- TCP transport.

The computationally expensive tasks remain on the laptop:

- event detection,
- DSP,
- classification,
- GCC-PHAT,
- TDOA estimation,
- localization,
- calibration,
- database persistence,
- analytics,
- dashboard visualization.

This keeps the embedded layer deterministic and lightweight while allowing the research algorithms to evolve independently.

---

# 2. Problem Statement

Wildlife observation traditionally relies on methods such as:

- direct human surveys,
- camera traps,
- GPS collars,
- radio telemetry,
- thermal cameras,
- visual drones,
- manual acoustic surveys,
- standalone passive acoustic recorders.

Each technique has advantages, but no single method simultaneously provides:

- non-contact observation,
- continuous temporal monitoring,
- classification,
- environmental context,
- low-cost multi-point sensing,
- approximate spatial localization,
- automated research analytics.

A standard passive acoustic recorder can answer:

> “What sound occurred?”

but generally cannot independently answer:

> “Where approximately did it originate?”

Likewise, an independent network of low-cost microphones may record the same signal, but if each node uses its own unsynchronized oscillator, sample-level timing differences may be dominated by clock error rather than acoustic propagation.

The project therefore investigates whether a low-cost synchronized IoT array can combine:

**Passive Acoustic Monitoring + Digital Signal Processing + TDOA Localization + AI Classification + Environmental Context + Research Analytics**

inside one integrated platform.

---

# 3. Why This System Is Needed

Wildlife activity is often:

- nocturnal,
- visually obscured,
- distributed over large areas,
- intermittent,
- species-specific,
- sensitive to human presence.

Acoustic monitoring can operate when visual monitoring becomes difficult.

Examples include:

- dense vegetation,
- night-time activity,
- low-light environments,
- canopy environments,
- hidden nesting sites,
- amphibian monitoring near wetlands,
- insect activity,
- bird vocalization surveys,
- mammalian calls that occur outside camera field of view.

A synchronized acoustic network adds spatial information that an ordinary recorder lacks.

The long-term objective is therefore not merely:

> “Record wildlife audio.”

It is closer to:

> **Automatically observe when, where, and under what environmental conditions biologically relevant acoustic activity occurs.**

---

# 4. Project Objectives

The project is designed around the following engineering and research objectives.

### Primary objectives

1. Build a three-node synchronized acoustic acquisition system.
2. Maintain a common sample timeline across all microphones.
3. Detect acoustic events automatically.
4. Preserve synchronized multi-node event recordings.
5. Extract meaningful acoustic DSP features.
6. Classify detected sounds into broad wildlife-related categories.
7. Support optional pretrained species-oriented classification.
8. Estimate source position using TDOA.
9. Compensate localization for environmental speed-of-sound variation.
10. Calibrate persistent node timing bias.
11. Quantitatively benchmark localization accuracy.
12. Persist structured observations in SQLite.
13. Perform temporal, environmental, and spatial research analytics.
14. Present results through a local dashboard.
15. Export reproducible datasets for external research.

---

# 5. Key Engineering Contributions

The system combines several components that are often implemented independently.

### Shared-clock microphone network

All three microphones operate under one I2S timing domain.

### Sample-index synchronization

Acoustic timing is represented using a shared sample index rather than TCP packet arrival time.

### Multi-node event detection

Events can require agreement from multiple acoustic nodes rather than relying on a single microphone.

### Environmental acoustic compensation

Temperature, humidity, and pressure measurements can modify the assumed speed of sound.

### TDOA calibration layer

Persistent timing bias is modeled as coherent **per-node timing offsets** instead of unrelated pair corrections.

### Modular classifier architecture

Classification supports interchangeable backends:

- heuristic,
- BirdNET,
- ensemble,
- future pretrained models.

### Research analytics layer

Persisted acoustic observations can be transformed into:

- temporal activity profiles,
- environmental associations,
- spatial occupancy maps,
- hotspot summaries,
- transition indicators,
- conservative behavior-related metrics.

### Hardware abstraction

The embedded firmware is being structured so that specific sensors can later be replaced while maintaining standardized outputs whenever technically possible.

---

# 6. Why Passive Acoustic Monitoring

Passive Acoustic Monitoring, or PAM, records biological and environmental sounds without requiring constant human observation.

It is particularly useful for vocal species.

Potential advantages include:

- continuous operation,
- low disturbance,
- operation at night,
- monitoring beyond camera field of view,
- retrospective analysis,
- automatic event detection,
- scalable long-duration datasets.

However, acoustic monitoring also has limitations:

- reverberation,
- wind,
- anthropogenic noise,
- overlapping species,
- source-level variation,
- microphone frequency-response differences,
- difficulty identifying silent animals.

The present system does not attempt to replace every wildlife-monitoring technology.

Instead, it targets the subset of wildlife activity that produces measurable acoustic evidence.

---

# 7. Comparison With Other Wildlife Monitoring Methods

| Method | Main Strength | Major Limitation | Comparison With This System |
|---|---|---|---|
| Human field survey | Expert contextual interpretation | Labor-intensive and intermittent | Acoustic system can operate continuously |
| Camera trap | Strong visual evidence | Limited field of view and lighting dependence | Acoustic sensing can detect activity outside camera view |
| Thermal camera | Works in low light | Cost and limited species identity | Acoustic nodes are cheaper but provide weaker physical identity |
| GPS collar | Excellent individual trajectory data | Requires capture/tagging | Acoustic system is contactless but cannot prove individual identity |
| Radio telemetry | Reliable tagged-animal tracking | Requires transmitter on animal | Acoustic localization is non-invasive but less precise |
| LiDAR | Excellent geometry/environment mapping | Does not inherently classify sound-producing animals | Complementary rather than competing technology |
| Radar | Useful for movement and flying targets | Higher complexity/cost | Acoustic sensing better captures vocalization |
| Single acoustic recorder | Simple and proven | No intrinsic source localization | Multi-node synchronization adds spatial inference |
| Independent wireless microphones | Flexible deployment | Unsynchronized clocks damage TDOA accuracy | Shared I2S timing is the main architectural advantage |
| Microphone array | High-quality localization possible | Often expensive and centralized | This project explores a distributed low-cost IoT implementation |

A GPS collar strongly surpasses the present system when the research objective is:

> tracking one known animal continuously.

A high-quality calibrated commercial microphone array may surpass it in:

- TDOA precision,
- microphone matching,
- synchronization quality,
- localization accuracy.

The proposed architecture is more appropriate when the goal is:

> low-cost, non-contact, distributed monitoring of acoustically active wildlife.

---

# 8. System Architecture

```text
                         ┌─────────────────────┐
                         │       Node 1        │
                         │ ESP32 + INMP441     │
                         │ BME280              │
                         │ I2S CLOCK MASTER    │
                         └─────────┬───────────┘
                                   │
                           BCLK + WS + SYNC
                                   │
                   ┌───────────────┴───────────────┐
                   │                               │
          ┌────────▼────────┐             ┌────────▼────────┐
          │     Node 2      │             │     Node 3      │
          │ ESP32 + INMP441 │             │ ESP32 + INMP441 │
          │ I2S SLAVE       │             │ I2S SLAVE       │
          └────────┬────────┘             └────────┬────────┘
                   │                               │
                   └──────────── Wi-Fi ────────────┘
                                   │
                            TCP / Protocol v4
                                   │
                         ┌─────────▼─────────┐
                         │      Laptop       │
                         │                   │
                         │ TCP Receiver      │
                         │ Stream Manager    │
                         │ Event Detector    │
                         │ DSP               │
                         │ Classification    │
                         │ GCC-PHAT / TDOA   │
                         │ Localization      │
                         │ Calibration       │
                         │ SQLite            │
                         │ Analytics         │
                         │ Streamlit UI      │
                         └───────────────────┘
```

---

# 9. Reference Hardware Architecture

The current laboratory reference platform uses:

| Component | Quantity | Purpose |
|---|---:|---|
| ESP32 DevKit V1 / ESP32-WROOM-32 | 3 | Acquisition and network nodes |
| INMP441 I2S MEMS microphone | 3 | Digital acoustic sensing |
| BME280 | 1 | Environmental sensing |
| Laptop/PC | 1 | Central processing |
| USB power | 3 | Node power |
| CAT5e or similar twisted-pair cable | As required | Shared-clock wiring |

The initial array geometry is an approximately equilateral triangle:

```text
                 Node 2
              (0.5, 0.866)
                 /   \
                /     \
               /       \
              /         \
       Node 1 -------- Node 3
       (0, 0)          (1, 0)
```

Nominal side length:

```text
1.0 metre
```

These coordinates are software defaults only.

For real localization experiments, actual microphone coordinates must be physically measured and entered into configuration.

---

# 10. Acoustic Node Roles

## Node 1 — Master

Node 1 performs:

- local INMP441 acquisition,
- I2S master clock generation,
- shared BCLK distribution,
- shared WS distribution,
- GPIO27 synchronization signalling,
- BME280 acquisition,
- Wi-Fi communication,
- TCP communication,
- Protocol v4 packet generation,
- health reporting.

## Node 2 — Slave

Node 2 performs:

- local microphone acquisition,
- receives BCLK from Node 1,
- receives WS from Node 1,
- receives SYNC from Node 1,
- Wi-Fi communication,
- TCP communication,
- health monitoring.

## Node 3 — Slave

Node 3 performs the same synchronization role as Node 2.

---

# 11. Pin Mapping

| Signal | Node 1 Master | Node 2 Slave | Node 3 Slave |
|---|---|---|---|
| BCLK | GPIO26 OUT | GPIO26 IN | GPIO26 IN |
| WS / LRCLK | GPIO25 OUT | GPIO25 IN | GPIO25 IN |
| Local microphone SD | GPIO33 IN | GPIO33 IN | GPIO33 IN |
| SYNC | GPIO27 OUT | GPIO27 IN | GPIO27 IN |
| Ground | GND | GND | GND |
| BME280 SDA | GPIO21 | — | — |
| BME280 SCL | GPIO22 | — | — |

All INMP441 modules currently use the same selected I2S channel.

The current reference wiring ties the microphone `L/R` selection for the required slot and firmware extracts that configured slot consistently.

**Microphone data outputs must never be electrically tied together.**

Each microphone has its own local:

```text
INMP441 SD → GPIO33
```

on its own ESP32.

Only the clocks are shared.

---

# 12. I2S Synchronization Architecture

This is one of the most important design decisions in the system.

If each ESP32 generated its own nominal 48 kHz sampling clock, then even small oscillator differences would cause the streams to drift over time.

For TDOA localization, the relevant acoustic delays are extremely small.

For microphones separated by 1 metre:

\[
\tau_{max} \approx \frac{1}{343}
\]

\[
\tau_{max} \approx 2.915 \text{ ms}
\]

At 48 kHz:

\[
2.915 \text{ ms} \times 48000
\approx 140 \text{ samples}
\]

A small unsynchronized clock drift can therefore become significant relative to the actual acoustic delay.

The system avoids independent sample-rate domains by making Node 1 the I2S clock master.

```text
Node 1
 BCLK ─────────────┬──────────> Node 2
                   └──────────> Node 3

 WS   ─────────────┬──────────> Node 2
                   └──────────> Node 3
```

All microphones are therefore sampled from the same clock source.

---

# 13. GPIO27 SYNC Signal

GPIO27 is **not the TDOA clock**.

Its purpose is:

- session-start marking,
- stream-start coordination,
- synchronization diagnostics.

The actual sample-rate synchronization comes from:

```text
BCLK + WS
```

The distinction is important.

```text
BCLK / WS
    → continuous sample timing

GPIO27 SYNC
    → start/session marker
```

Periodic GPIO27 pulses are deliberately not used to artificially reset or “repair” sample counters during an active acquisition session.

---

# 14. Session and Sample Timeline

Every acquisition run is associated with a shared:

```text
sessionId
```

generated by the laptop.

The intended START ordering is:

```text
Node 2
  ↓
Node 3
  ↓
Node 1
```

Node 1 then starts the shared I2S clock.

Every audio packet contains a 64-bit:

```text
sampleIndex
```

representing its position on the shared acquisition timeline.

This is the authoritative timing coordinate for cross-node waveform assembly.

The system does **not** use:

- TCP arrival time,
- Python receive timestamps,
- ESP32 `localMicros`

as the primary TDOA timing source.

`localMicros` remains diagnostic only.

---

# 15. Firmware Architecture

The firmware uses FreeRTOS task separation.

Reference task allocation:

## Node 1

```text
Core 0
└── Audio capture task
    Priority 4
    Sole I2S owner

Core 1
├── Network task
│   Priority 3
│
└── Environment task
    Priority 1
```

## Nodes 2 and 3

```text
Core 0
└── Audio capture task
    Priority 4

Core 1
├── Network task
│   Priority 3
│
└── Synchronization health task
    Priority 2
```

The audio task is deliberately the sole owner of:

- I2S start,
- I2S reads,
- I2S stop.

This avoids unsafe cross-core teardown of a blocking peripheral operation.

Communication between tasks occurs through:

- FreeRTOS queues,
- event groups,
- protected shared state.

The 64-bit sample counter is protected using an ESP32 critical section.

---

# 16. Sensor-Abstraction Strategy

The project is designed so that the current sensors are **reference implementations**, not permanently hard-coded architectural dependencies.

Current acoustic adapter:

```text
INMP441
      ↓
48 kHz
mono
PCM16
      ↓
standardized acoustic output
```

A future microphone can replace the INMP441 provided the adapter can produce an equivalent waveform contract.

Current environmental adapter:

```text
BME280
   ↓
temperature °C
humidity %
pressure hPa
```

A future environmental sensor may replace the BME280 if equivalent measurements are available.

The software must **never fabricate an unavailable physical measurement** merely to preserve an interface.

For example, if a replacement sensor measures temperature and humidity but not pressure, the correct engineering response is to:

- use an additional pressure sensor,
- make pressure optional,
- or revise the environmental model.

---

# 17. Protocol v4

The ESP32 nodes communicate with the laptop using a custom binary TCP protocol.

## Packet header

Protocol v4 uses a 40-byte header.

Conceptual layout:

| Field | Type |
|---|---|
| magic | uint32 |
| protocolVersion | uint8 |
| nodeId | uint8 |
| packetType | uint8 |
| flags | uint8 |
| sequence | uint32 |
| sessionId | uint32 |
| sampleIndex | uint64 |
| localMicros | uint32 |
| i2sErrorCount | uint32 |
| payloadLength | uint32 |
| payloadCRC32 | uint32 |

Magic:

```text
0x574C5343
```

Protocol:

```text
4
```

## Packet types

The protocol currently supports logical packet categories including:

```text
HELLO
AUDIO
ENVIRONMENT
HEARTBEAT
SYNC
```

## Control commands

Laptop control commands include:

```text
START = 0xA1
STOP  = 0xA2
PING  = 0xA3
```

---

# 18. Network and Bandwidth Model

Acquisition configuration:

```text
48,000 samples/second
×
2 bytes/sample
×
1 mono channel
```

Per node:

\[
48000 \times 2
=
96000\text{ bytes/s}
\]

approximately:

```text
96 kB/s
0.768 Mbit/s
```

For three nodes:

\[
3 \times 96000
=
288000\text{ bytes/s}
\]

approximately:

```text
288 kB/s
2.304 Mbit/s
```

This value excludes:

- TCP overhead,
- IP overhead,
- Wi-Fi MAC overhead,
- protocol headers,
- environment packets,
- heartbeat packets.

The bandwidth is practical for a local Wi-Fi network but should not be interpreted as negligible for battery-powered field deployment.

---

# 19. Laptop Software Architecture

The laptop performs the computational pipeline.

```text
TCP Receiver
    ↓
Protocol Parser
    ↓
Node State
    ↓
Stream Manager
    ↓
Event Detector
    ↓
Event Pipeline
    ├── DSP
    ├── Classification
    ├── TDOA Localization
    └── Persistence
          ↓
       SQLite/WAV
          ↓
       Analytics
          ↓
       Dashboard
          ↓
       Research Exports
```

Major modules include:

```text
protocol.py
node.py
server.py
stream_manager.py
event_detector.py
event_pipeline.py
environment.py
database.py
```

---

# 20. Acoustic Event Detection

The system uses an acoustic event detector rather than a voice-specific VAD.

It combines:

- adaptive background energy estimation,
- RMS energy,
- spectral flux,
- hysteresis,
- attack logic,
- release logic,
- multi-node agreement.

Reference defaults include:

```text
noise history          = 96 blocks
noise quantile         = 0.30

trigger margin         = 8 dB
release margin         = 4 dB

minimum spectral flux  = 0.06
strong energy margin   = 15 dB

minimum trigger nodes  = 2

attack                 = 1 block
release                = 2 blocks

minimum event duration = 30 ms
maximum event duration = 12 s

pre-event padding      = 0.50 s
post-event padding     = 0.75 s
```

At 1024 samples per block:

\[
\frac{1024}{48000}
=
0.02133\text{ s}
\]

or approximately:

```text
21.33 ms per audio packet
```

Candidate attack blocks are not allowed to contaminate the background-noise model.

This helps avoid a newly arriving sound being immediately learned as “normal background.”

---

# 21. DSP Pipeline

Detected events are passed through the laptop-side signal-processing pipeline.

Typical stages are:

```text
PCM16
  ↓
floating-point conversion
  ↓
DC removal
  ↓
optional band-pass filtering
  ↓
feature extraction
  ↓
optional model normalization
```

Raw recordings are preserved independently from normalized model waveforms.

This distinction is important because research traceability requires retaining the original measured signal.

---

# 22. Acoustic Feature Extraction

The DSP layer extracts features including:

- RMS amplitude,
- peak amplitude,
- crest factor,
- zero-crossing rate,
- dominant frequency,
- spectral centroid,
- spectral bandwidth,
- spectral rolloff,
- spectral flatness,
- spectral flux,
- signal-to-noise estimate,
- MFCC statistics.

Reference STFT settings:

```text
FFT length    = 2048
hop length    = 512
```

Reference MFCC settings:

```text
MFCC count    = 13
Mel filters   = 64
fmin          = 50 Hz
fmax          = 16 kHz
```

These features serve several purposes:

- heuristic classification,
- research analysis,
- model diagnostics,
- best-channel selection,
- dataset export.

---

# 23. Classification Architecture

Classification is implemented using a backend abstraction.

```text
ClassificationInput
        ↓
ClassifierBackend
        ↓
ClassificationResult
```

This prevents `event_pipeline.py` from depending directly on one ML framework.

Supported architecture:

```text
classification/
├── base.py
├── classifier.py
├── factory.py
├── heuristic_backend.py
├── birdnet_backend.py
└── ensemble_backend.py
```

Broad system ontology:

```text
bird
insect
amphibian
mammal
noise
unknown
```

The broad ontology is intentionally more stable than a species taxonomy.

---

# 24. Heuristic Classifier

The heuristic classifier is the lightweight baseline backend.

Advantages:

- no trained model required,
- low dependency cost,
- explainable,
- fast,
- useful for simulation,
- useful as a fallback.

It uses DSP characteristics to estimate broad acoustic classes.

It should not be interpreted as a species-identification model.

The heuristic backend is particularly valuable as:

- a baseline,
- a fallback,
- a debugging classifier,
- one ensemble member.

---

# 25. BirdNET Integration

An optional BirdNET backend is available for species-oriented acoustic inference.

Architecture:

```text
48 kHz event waveform
        ↓
BirdNET adapter
        ↓
model-specific preprocessing/resampling
        ↓
species predictions
        ↓
project adapter
        ↓
broad class = BIRD
        +
species evidence preserved
```

The common `ClassificationResult` remains broad-category oriented.

BirdNET species evidence is therefore preserved through:

- classifier score metadata,
- explanation strings,
- classifier identity/version.

Conceptually:

```text
bird → 0.91

species:Species_A → 0.91
species:Species_B → 0.34
```

A BirdNET prediction is still an **automated inference**, not automatically a verified biological observation.

Independent validation remains necessary for research claims.

---

# 26. Ensemble Classification

The ensemble backend combines:

```text
Heuristic
    +
BirdNET
    ↓
weighted broad-class fusion
```

Initial weights:

```text
Heuristic = 1.0
BirdNET   = 1.0
```

Equal weights are intentional.

No classifier should receive greater authority before controlled labeled evaluation demonstrates that it deserves greater weight.

Each member's broad scores are normalized before weighted combination.

The final result preserves both:

- ensemble scores,
- member-specific evidence.

The ensemble can also record partial failures.

Example:

```text
Heuristic succeeds
BirdNET fails
       ↓
partial result may remain available
       +
BirdNET failure retained in reasons
```

when heuristic fallback is enabled.

---

# 27. TDOA Localization

The system estimates source position using **Time Difference of Arrival**.

For microphone pair A and B:

\[
TDOA(A,B)=t_B-t_A
\]

The sign convention is:

```text
positive TDOA
=
signal reached B later than A
```

For a hypothetical source position \(p\):

\[
\tau_{AB}
=
\frac{
\|p-m_B\|-\|p-m_A\|
}{
c
}
\]

where:

- \(m_A\) = position of microphone A,
- \(m_B\) = position of microphone B,
- \(c\) = speed of sound.

The solver finds a position whose predicted pairwise delays best match measured delays.

---

# 28. GCC-PHAT

Waveform delays are estimated using **Generalized Cross-Correlation with Phase Transform**.

Conceptually:

\[
R_{PHAT}(\tau)
=
\mathcal{F}^{-1}
\left[
\frac{
X_1(f)X_2^*(f)
}{
|X_1(f)X_2^*(f)|+\epsilon
}
\right]
\]

The delay is estimated from the correlation peak.

Advantages include relative robustness against amplitude differences.

However, GCC-PHAT is still affected by:

- reverberation,
- echoes,
- multiple simultaneous sources,
- narrow-band calls,
- low SNR,
- microphone mismatch.

---

# 29. Physical Delay Constraints

For microphones separated by distance \(d\):

\[
|\tau| \leq \frac{d}{c}
\]

Therefore a measured delay outside this interval is physically impossible for direct free-field propagation.

The localization pipeline uses microphone geometry to constrain GCC-PHAT delay search.

When calibration is active, the raw GCC search interval can be extended by the known channel timing offset.

After calibration is removed, the corrected delay must still satisfy the true geometric limit.

---

# 30. Environmental Speed-of-Sound Compensation

The speed of sound is not perfectly constant.

Temperature and atmospheric conditions modify acoustic propagation.

The system therefore supports:

```text
BME280 telemetry
      ↓
temperature
humidity
pressure
      ↓
speed-of-sound estimate
      ↓
TDOA localization
```

If valid environmental data is unavailable:

```text
343 m/s
```

is used as the configured fallback.

Environmental telemetry near the **temporal center** of the localization window is preferred rather than blindly using the newest available reading.

---

# 31. Nonlinear Position Solver

Given multiple TDOA measurements, the position solver minimizes disagreement between:

- measured pair delays,
- geometry-predicted pair delays.

Conceptually:

\[
r_{AB}(x,y)
=
\tau_{AB}^{measured}
-
\frac{
d_B(x,y)-d_A(x,y)
}{
c
}
\]

The position is obtained by minimizing the residual vector.

The current solver uses nonlinear least-squares optimization with robust-loss behavior.

Optional spatial bounds can restrict the solution to a region surrounding the microphone array.

---

# 32. TDOA Calibration

Shared I2S clocks significantly improve synchronization, but systematic channel timing bias may still remain.

Possible sources include:

- microphone module differences,
- digital path latency,
- I2S slot behavior,
- hardware interface delay,
- implementation offsets.

The calibration system models a timing bias for each node.

Let:

```text
bias_A
bias_B
```

represent channel delays.

Then:

\[
offset_{AB}=bias_B-bias_A
\]

and:

\[
TDOA_{corrected}
=
TDOA_{raw}-offset_{AB}
\]

This is superior to storing three unrelated pair corrections because the node-bias model automatically enforces cycle consistency:

\[
offset_{12}+offset_{23}=offset_{13}
\]

The calibration algorithm supports:

- known source positions,
- expected theoretical TDOA,
- measured TDOA,
- residual calculation,
- robust outlier handling,
- weighted pair estimates,
- least-squares node bias estimation,
- pair consistency analysis.

**Calibration is disabled by default.**

It must remain disabled until real physical calibration measurements exist.

Do not insert fabricated timing corrections.

---

# 33. Localization Benchmarking

The project includes a dedicated localization benchmark layer.

For known source position:

\[
(x_{true},y_{true})
\]

and estimated position:

\[
(x_{est},y_{est})
\]

the radial error is:

\[
e
=
\sqrt{
(x_{est}-x_{true})^2
+
(y_{est}-y_{true})^2
}
\]

Benchmark metrics include:

- attempted localizations,
- successful localizations,
- failure count,
- success rate,
- mean localization error,
- RMSE,
- median error,
- error standard deviation,
- P90 error,
- P95 error,
- maximum error,
- X-axis bias,
- Y-axis bias,
- radial systematic bias,
- per-position repeatability.

The benchmark also supports comparison between:

```text
uncalibrated
     vs
calibrated
```

results.

---

## Accuracy vs Repeatability

These must not be confused.

### Accuracy

How close estimates are to the true source position.

### Repeatability

How tightly repeated estimates cluster.

A system could repeatedly estimate approximately:

```text
true:
(0.50, 0.50)

estimates:
(0.70, 0.50)
(0.69, 0.51)
(0.71, 0.49)
```

This would have:

- good repeatability,
- poor accuracy.

A persistent timing bias could produce exactly this type of behavior.

---

# 34. Environmental Monitoring

Node 1 contains the environmental sensor in the current reference design.

Measured quantities:

```text
temperature        °C
relative humidity  %
pressure           hPa
```

Environmental telemetry serves two separate roles.

### Localization

Adjust propagation-speed estimation.

### Research analytics

Investigate associations between acoustic activity and environmental conditions.

These two uses should not be conflated.

---

# 35. Persistence and Database Architecture

SQLite is used for structured local persistence.

Default database:

```text
data/database/events.db
```

The database stores information such as:

- acquisition sessions,
- telemetry,
- event metadata,
- DSP features,
- classification outputs,
- localization outputs,
- filesystem references.

Event waveforms are stored as WAV files rather than SQLite audio BLOBs.

Conceptually:

```text
data/events/
└── <session>/
    └── event_000001/
        ├── node_1.wav
        ├── node_2.wav
        └── node_3.wav
```

This keeps the relational database focused on searchable metadata while making waveforms easy to inspect using external audio tools.

---

# 36. Scientific Timestamp Policy

Two timestamps must be distinguished.

## Database insertion time

```text
created_at
```

means approximately:

> when SQLite inserted the record.

It does **not necessarily represent when the acoustic event occurred**.

## Acoustic observation time

The scientifically meaningful event time is reconstructed from the acquisition timeline:

\[
t_{event}
=
t_{session-start}
+
\frac{
startSample
}{
sampleRate
}
\]

For a 48 kHz session:

```text
start_sample = 480000
```

means:

\[
480000/48000
=
10\text{ seconds}
\]

after session start.

Analytics and research export tools therefore use the sample-derived timestamp rather than database insertion latency.

---

# 37. Research Analytics

The `analytics/` package converts persisted observations into higher-level research summaries.

Architecture:

```text
event rows
environmental bins
localization results
classification results
        ↓
analytics/
        ├── activity.py
        ├── environmental.py
        ├── spatial.py
        ├── behavior.py
        └── service.py
        ↓
ResearchAnalyticsReport
```

The analytics layer does not control:

- ESP32 acquisition,
- networking,
- event detection,
- localization,
- database writes.

It is a read-side research layer.

---

# 38. Temporal Activity Analysis

Temporal analysis can calculate:

- event counts,
- event rate,
- time-bin activity,
- hourly activity profile,
- peak activity period,
- class-specific activity distributions.

For example:

```text
00:00–01:00   2 events
01:00–02:00   5 events
02:00–03:00   9 events
...
```

Such summaries may support investigation of:

- nocturnal activity,
- dawn chorus,
- recurring calling periods,
- temporal niche separation.

These interpretations require sufficient data and ecological context.

---

# 39. Environmental Association Analysis

Environmental analytics can evaluate relationships between variables such as:

```text
temperature
humidity
pressure
```

and response variables such as:

```text
event count
class-specific activity
```

The implementation supports rank-based correlation analysis.

Association direction may be categorized as:

```text
positive
negative
neutral
```

and strength as:

```text
negligible
weak
moderate
strong
very strong
```

Multiple-comparison adjustment support is also available.

### Critical interpretation rule

Correlation is not causation.

For example:

> “Bird acoustic activity increased with temperature”

is an observational association.

It does **not automatically prove** that temperature caused the increased activity.

---

# 40. Spatial Analysis

Localized acoustic events can be grouped into spatial cells.

The analysis layer can calculate:

- localization coverage,
- occupied cells,
- spatial occupancy,
- hotspot cell,
- occupancy entropy,
- event transitions between cells.

A heatmap can therefore represent:

```text
where detected acoustic activity was concentrated
```

rather than only when it occurred.

---

## Spatial transitions

Suppose consecutive localized events occur in:

```text
Cell A
  ↓
Cell B
  ↓
Cell C
```

The analytics system may record transitions:

```text
A → B
B → C
```

However, this does **not prove that the same individual animal physically moved from A to B to C**.

Without identity tracking, these remain transitions between consecutive observations.

---

# 41. Behavior Indicators

The behavior module deliberately produces **indicators**, not definitive ethological labels.

Possible indicators include:

- temporal concentration,
- class concentration,
- spatial concentration,
- spatial redistribution,
- repeated-area activity.

Example interpretation:

```text
Many events repeatedly localized near one area
```

may indicate concentrated acoustic activity.

It does not automatically prove:

```text
nesting
territorial behavior
feeding
mating
migration
```

Those claims require independent ecological evidence.

---

# 42. Dashboard

The local dashboard is implemented using Streamlit.

Architecture:

```text
dashboard/
├── __init__.py
├── app.py
├── data_access.py
├── plots.py
├── audio_view.py
├── live_view.py
└── analysis_view.py
```

The dashboard is a **presentation layer**.

It should not perform:

- event detection,
- signal acquisition,
- classifier training,
- TDOA solving,
- database mutation.

Major views include:

### Live Monitor

Designed to display recent system and event information.

### Research Analysis

Designed for session-oriented exploration of:

- activity,
- classes,
- environment,
- spatial distributions,
- research indicators.

### Audio visualization

Supports event waveform and spectrogram inspection.

---

# 43. Research Export Tools

Two export tools are planned/implemented for different purposes.

---

## Event exporter

```text
tools/export_events.py
```

Exports individual acoustic observations.

Supported representations include:

```text
CSV
JSON
JSONL
```

Example:

```powershell
python -m tools.export_events --format csv
```

Session-specific export:

```powershell
python -m tools.export_events --session-id 12345 --format json
```

Conceptual exported fields include:

- event identity,
- session identity,
- reconstructed event time,
- sample positions,
- detector metadata,
- environmental data,
- localization,
- DSP features,
- MFCC,
- classification,
- traceability fields.

---

## Research-metrics exporter

```text
tools/export_research_metrics.py
```

Exports higher-level analytics.

Example:

```powershell
python -m tools.export_research_metrics --session-id 12345
```

Conceptual output:

```text
data/exports/
├── research_metrics_session_12345.json
├── research_metrics_session_12345_summary.csv
└── research_metrics_session_12345_tables/
    ├── activity_bins.csv
    ├── class_activity.csv
    ├── environmental_associations.csv
    ├── spatial_cells.csv
    ├── spatial_transitions.csv
    └── behavior_indicators.csv
```

Exact table names depend on the final analytics-report structure.

---

# 44. Repository Structure

Representative architecture:

```text
Wildlife_Soundscape/
│
├── firmware/
│   ├── Node_1_Master/Node_1_Master.ino
│   ├── Node_2_Slave/Node_2_Slave.ino
│   ├── Node_3_Slave/Node_3_Slave.ino
│   └── README.md
│
├── config.py
├── database.py
├── environment.py
├── event_detector.py
├── event_pipeline.py
├── main.py
├── models.py
├── node.py
├── protocol.py
├── server.py
├── simulator.py
├── stream_manager.py
│
├── src/
│   └── wildlife_soundscape/
│       ├── __init__.py
│       ├── __main__.py
│       └── cli.py
│
├── analytics/
│   ├── __init__.py
│   ├── models.py
│   ├── activity.py
│   ├── environmental.py
│   ├── spatial.py
│   ├── behavior.py
│   └── service.py
│
├── calibration/
│   ├── __init__.py
│   ├── tdoa_calibration.py
│   └── localization_benchmark.py
│
├── classification/
│   ├── __init__.py
│   ├── base.py
│   ├── classifier.py
│   ├── factory.py
│   ├── heuristic_backend.py
│   ├── birdnet_backend.py
│   └── ensemble_backend.py
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   ├── data_access.py
│   ├── plots.py
│   ├── audio_view.py
│   ├── live_view.py
│   └── analysis_view.py
│
├── dsp/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── features.py
│
├── localization/
│   ├── __init__.py
│   ├── filtering.py
│   ├── gcc_phat.py
│   ├── tdoa.py
│   ├── solver.py
│   └── engine.py
│
├── tools/
│   ├── export_events.py
│   └── export_research_metrics.py
│
├── tests/
│   └── unit/integration test modules
│
├── data/
│   ├── database/
│   ├── events/
│   ├── recordings/
│   ├── exports/
│   └── calibration/
│
├── docs/
│   ├── README.md
│   ├── protocol.md
│   ├── release-notes.md
│   └── roadmap.md
├── DEVELOPMENT.md
├── requirements.txt
├── README.md
└── .gitignore
```

The Python application currently retains root-level compatibility modules.
The planned installable `src/wildlife_soundscape` migration is tracked in
`docs/roadmap.md` and will be performed separately from functional changes.

---

# 45. Software Requirements

Current core Python dependencies include:

```text
numpy
scipy
librosa
streamlit
plotly
```

The authoritative dependency constraints and optional dependency groups are in:

```text
pyproject.toml
```

`requirements.txt` and `requirements-dev.txt` remain compatibility inputs.

The optional BirdNET backend uses additional dependencies and should remain separate from the minimum core installation.

Development and validation dependencies, including pytest, Ruff, mypy,
and coverage tooling, are defined separately in:

```text
requirements-dev.txt
```

---

# 46. Python Installation

Recommended current development target:

```text
Python 3.13
```

Windows / VS Code:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
python -m pip install -e .
```

For development, testing, and static analysis, install the development
environment instead. It includes the runtime requirements transitively:

```powershell
python -m pip install -e ".[dev]"
```

The requirement files remain available as compatibility inputs for deployment
environments that do not yet install from `pyproject.toml`.

---

## Verify interpreter

```powershell
python --version
```

Verify environment:

```powershell
python -c "import numpy, scipy, librosa; print('Core dependencies OK')"
```

---

# 47. Optional BirdNET Installation

BirdNET is not required for the core heuristic classifier.

The current optional BirdNET adapter targets the BirdNET acoustic model through an ONNX runtime.

Typical optional installation:

```powershell
pip install "birdnet[onnx]"
```

BirdNET should remain outside the mandatory dependency set unless it becomes a required deployment component.

After installation, the classification backend can eventually be selected using:

```python
ClassificationConfig(
    backend="birdnet",
)
```

or:

```python
ClassificationConfig(
    backend="ensemble",
)
```

depending on the final configuration workflow.

---

# 48. ESP32 Firmware Requirements

Reference embedded target:

```text
ESP32 DevKit V1
ESP32-WROOM-32
Arduino-ESP32 3.x
```

Current firmware relies on:

- ESP32 I2S support,
- Wi-Fi,
- TCP client networking,
- FreeRTOS,
- Wire/I2C,
- Adafruit BME280 library on Node 1.

The firmware must be compile-audited against the exact installed Arduino-ESP32 core before hardware deployment.

---

# 49. Physical Wiring Guidelines

Because BCLK and WS are high-frequency digital timing signals, wiring quality affects system reliability.

Initial laboratory recommendation:

```text
0.5–1.0 m node spacing
short clock wiring
shared ground
twisted signal/ground pairs
```

CAT5e can be used conceptually as:

```text
BCLK + GND
WS   + GND
SYNC + GND
```

Optional source termination:

```text
33–47 Ω
```

near the master output may be evaluated if ringing or signal-integrity problems appear.

Avoid unnecessarily long breadboard jumper wiring for shared clocks.

---

# 50. Firmware Configuration

Before flashing the ESP32 nodes, configure Wi-Fi and laptop IP settings in all firmware files.

Conceptually:

```cpp
const char* WIFI_SSID =
    "YOUR_WIFI_NAME";

const char* WIFI_PASSWORD =
    "YOUR_WIFI_PASSWORD";

IPAddress LAPTOP_IP(
    192,
    168,
    1,
    100
);
```

Default laptop TCP server port:

```text
5001
```

The laptop should preferably have a stable local IP during experiments.

---

# 51. Running Without Hardware

The simulator allows development of the laptop processing stack without physical ESP32 nodes.

Terminal 1:

```powershell
wildlife-receiver
```

Terminal 2:

```powershell
wildlife-simulator
```

The legacy `python main.py` and `python simulator.py` commands remain supported
during the package migration.

Available CLI operations may include:

```text
start
status
locate
events
stop
quit
```

The simulator generates multiple delayed acoustic streams from a known virtual source and uses the same protocol concepts as hardware nodes.

Reference simulation geometry:

```text
Node 1 = (0.0, 0.0)
Node 2 = (0.5, 0.8660254)
Node 3 = (1.0, 0.0)
```

Example simulated source:

```text
(0.45, 0.35)
```

Simulation validates software behavior.

It does **not validate**:

- microphone tolerances,
- real Wi-Fi behavior,
- I2S electrical integrity,
- reverberation,
- acoustic reflections,
- physical synchronization,
- real localization accuracy.

---

# 52. Running With Hardware

Recommended bring-up order:

```text
Stage 1
Node 1 only

Stage 2
Node 1 + Node 2

Stage 3
Node 1 + Node 2 + Node 3

Stage 4
Verify TCP streaming

Stage 5
Verify common sessionId

Stage 6
Verify sampleIndex behavior

Stage 7
Verify synchronized audio

Stage 8
Verify environmental telemetry

Stage 9
Verify event detection

Stage 10
Verify DSP

Stage 11
Verify classification

Stage 12
Verify raw TDOA

Stage 13
Perform physical calibration

Stage 14
Enable calibrated localization

Stage 15
Benchmark localization

Stage 16
Run dashboard and analytics
```

Do not begin by testing the complete pipeline simultaneously.

Progressive integration makes faults substantially easier to isolate.

---

# 53. Dashboard Launch

From the repository root:

```powershell
streamlit run dashboard/app.py
```

The dashboard should be treated as a local research interface.

The database and acquisition process may run independently.

---

# 54. Calibration Procedure

Calibration must be performed after the physical array is assembled.

## Step 1 — Measure microphone coordinates

Do not rely on nominal geometry.

Measure the actual microphone acoustic-center coordinates.

Update:

```python
LocalizationConfig.node_positions
```

accordingly.

---

## Step 2 — Choose known source locations

Use several accurately measured positions.

Avoid performing calibration using only one source point.

A useful calibration design should include:

- central source locations,
- off-axis locations,
- locations near different array regions.

---

## Step 3 — Generate repeatable calibration sound

Suitable signals may include:

- broadband impulses,
- sharp claps,
- controlled speaker pulses,
- reproducible broadband test signals.

A broadband signal usually provides better correlation information than a nearly pure tone.

---

## Step 4 — Measure pair TDOAs

For each known source position:

```text
Node 1 ↔ Node 2
Node 1 ↔ Node 3
Node 2 ↔ Node 3
```

measure raw TDOAs.

---

## Step 5 — Calculate expected TDOA

For known source \(s\):

\[
\tau_{AB}^{expected}
=
\frac{
d(s,B)-d(s,A)
}{
c
}
\]

---

## Step 6 — Calculate residual

\[
r_{AB}
=
\tau_{AB}^{measured}
-
\tau_{AB}^{expected}
\]

Persistent residuals suggest systematic timing bias.

---

## Step 7 — Fit node biases

The calibration system estimates coherent node biases relative to one reference node.

Reference:

```text
Node 1 bias = 0
```

Then offsets become:

\[
offset_{AB}=bias_B-bias_A
\]

---

## Step 8 — Evaluate calibration quality

Review:

- observation count,
- rejected outliers,
- residual spread,
- pair consistency,
- RMS consistency error.

Calibration should not be enabled merely because the fitting algorithm returned numbers.

---

## Step 9 — Enable calibration

Only after the calibration result is accepted should:

```python
enabled=True
```

be used.

---

# 55. Localization Benchmark Procedure

Calibration data and benchmark data should ideally be separated.

Using exactly the same observations for:

- parameter fitting,
- final performance claims

can produce optimistic results.

Recommended approach:

```text
Calibration positions
       ↓
estimate timing bias

Different benchmark positions
       ↓
measure localization performance
```

At each benchmark position:

1. Measure true source coordinates.
2. Record several repeated sound events.
3. Run localization.
4. Preserve failures.
5. Calculate error.
6. Compare calibrated and uncalibrated results.

Failed localization attempts must not simply be deleted from the reported experiment.

---

# 56. Recommended Experimental Methodology

A strong research evaluation should control:

- array geometry,
- source geometry,
- source sound,
- source height,
- microphone height,
- environmental conditions,
- room/field characteristics.

Potential experimental phases:

## Experiment A — Ideal software simulation

Purpose:

- validate mathematics,
- validate sign conventions,
- validate solver.

## Experiment B — Quiet indoor controlled test

Purpose:

- validate hardware synchronization,
- calibrate channel bias,
- establish best-case hardware performance.

## Experiment C — Reverberant indoor test

Purpose:

- evaluate multipath sensitivity.

## Experiment D — Outdoor short-range test

Purpose:

- evaluate practical wildlife-like acoustic localization.

## Experiment E — Environmental association collection

Purpose:

- collect sufficiently long datasets for temporal/environmental analytics.

## Experiment F — Classification evaluation

Purpose:

- compare heuristic,
- BirdNET,
- ensemble.

---

# 57. Evaluation Metrics

## Event detection

Recommended metrics:

```text
precision
recall
F1-score
false-positive rate
false-trigger count/hour
missed-event rate
```

---

## Classification

Recommended metrics:

```text
accuracy
macro F1
per-class precision
per-class recall
confusion matrix
bird recall
false-bird rate
```

If species predictions are evaluated:

```text
top-1 accuracy
top-k accuracy
species precision
species recall
```

---

## Localization

Recommended metrics:

```text
success rate
mean error
RMSE
median error
P90
P95
maximum error
X bias
Y bias
repeatability
```

---

## Synchronization/network

Recommended metrics:

```text
sequence gaps
packet loss
sample timeline discontinuities
I2S errors
clock fault count
queue congestion
reconnect count
```

---

## Performance

Potential measurements:

```text
CPU utilization
RAM usage
network throughput
dashboard latency
event-processing latency
classification latency
localization latency
```

---

# 58. Testing Strategy

The project contains unit and integration tests for major subsystems including:

- protocol parsing,
- node state,
- stream management,
- environment calculations,
- event detection,
- database,
- DSP,
- classification,
- GCC-PHAT,
- TDOA,
- solver,
- localization,
- simulator,
- event pipeline.

Newer modules should also receive tests for:

```text
TDOA calibration
localization benchmark
analytics
BirdNET adapter
ensemble fusion
export tools
dashboard data contracts
```

A full regression test run is required after final compatibility cleanup.

Typical command:

```powershell
pytest -q
```

Current README intentionally does **not** state a final passing test count because the expanded architecture has not yet completed its final regression cycle.

---

# 59. Reliability and Fault Handling

The architecture includes several defensive mechanisms.

## Protocol

- CRC32 payload validation,
- protocol version checks,
- packet length checks,
- sequence diagnostics,
- session validation.

## Wi-Fi/TCP

Reconnect logic uses exponential delay behavior.

Reference progression:

```text
1 s
2 s
4 s
8 s
...
up to approximately 30 s
```

## ESP32 audio

Tracked diagnostics include:

- I2S error count,
- clipping,
- queue congestion,
- slave clock health.

## BME280

Initialization attempts multiple supported I2C addresses and validates sensor identity.

## Session isolation

Audio from different sessions must never be combined merely because their sample indices overlap numerically.

## Event persistence

Core event metadata should remain persistable even if optional downstream analysis fails.

---

# 60. Security Considerations

The current architecture is a laboratory research system.

It assumes:

```text
trusted local Wi-Fi
```

The custom TCP transport currently should not be treated as a secure Internet protocol.

Potential production security requirements include:

- encrypted transport,
- device authentication,
- secure provisioning,
- protected credentials,
- firmware signing,
- configuration integrity,
- access control,
- audit logging.

Wi-Fi credentials embedded directly in firmware are acceptable for a controlled prototype but are not recommended for a commercial deployment.

---

# 61. Current Limitations

The system currently has several important limitations.

### Three-node 2-D localization

The reference system estimates planar position.

It does not yet provide robust full 3-D localization.

### Reverberation

Reflections can create false correlation peaks.

### Multiple simultaneous sources

The current localization pipeline primarily assumes one dominant acoustic source in a localization window.

### Geometry sensitivity

Incorrect microphone coordinates directly affect localization accuracy.

### Microphone mismatch

Low-cost microphone modules may exhibit unit-to-unit variation.

### Species inference limitations

Broad classification and BirdNET inference do not automatically establish biological ground truth.

### No individual identity tracking

The system cannot prove that two calls belong to the same animal.

### Wi-Fi dependency

Reference nodes currently depend on local Wi-Fi.

### USB-powered laboratory design

The current reference implementation is not yet an autonomous solar/battery field station.

### No weatherproof enclosure

Environmental ruggedization is outside the current lab prototype.

### No production security layer

The transport is designed for a trusted research LAN.

---

# 62. Real-World Deployment Considerations

A field-ready implementation would need significant additional engineering.

Potential requirements include:

```text
weatherproof enclosure
acoustic membrane
condensation management
battery power
solar charging
power monitoring
watchdog recovery
local storage
mesh/LoRa/cellular backhaul
remote OTA updates
tamper resistance
GPS/georeferencing
time synchronization across separated arrays
remote health monitoring
```

The current shared-wired I2S architecture is most appropriate for a **local array**, not microphones separated by hundreds of metres.

Large-area monitoring would likely use multiple local synchronized arrays connected through a higher-level network.

---

# 63. Research-Paper Opportunities

The platform supports several possible research questions.

## 1. Low-cost synchronized wildlife acoustic localization

Research question:

> How accurately can shared-clock ESP32 microphone nodes localize biologically relevant acoustic events?

---

## 2. Environmental compensation

Compare:

```text
constant 343 m/s
        vs
environment-derived speed of sound
```

and evaluate localization improvement.

---

## 3. Timing calibration

Compare:

```text
uncalibrated TDOA
        vs
calibrated TDOA
```

using controlled benchmark positions.

---

## 4. Classifier comparison

Compare:

```text
heuristic
BirdNET
ensemble
```

using labeled acoustic events.

---

## 5. Event detector evaluation

Investigate adaptive multi-node detection versus:

- static energy threshold,
- single-node detection.

---

## 6. Environmental acoustic ecology

Evaluate relationships among:

```text
temperature
humidity
pressure
time of day
acoustic activity
```

with appropriate statistical caution.

---

## 7. Spatial soundscape analysis

Investigate:

- hotspot persistence,
- spatial concentration,
- acoustic occupancy changes.

---

## 8. Ablation studies

A strong paper should consider disabling one component at a time.

Examples:

```text
without calibration
without environmental correction
without band-pass filtering
without multi-node event agreement
heuristic only
BirdNET only
ensemble
```

This helps demonstrate which architectural components actually contribute measurable value.

---

# 64. Potential Product and Patent Directions

The present system should not be assumed patentable merely because it combines several technologies.

Patentability requires:

- novelty,
- inventive step/non-obviousness,
- industrial applicability,
- prior-art analysis,
- jurisdiction-specific legal assessment.

However, several directions may provide stronger technical differentiation if genuinely developed beyond known prior art.

---

## A. Self-calibrating distributed acoustic array

Potential concept:

```text
known/controlled acoustic stimulus
        ↓
automatic pair-delay residual estimation
        ↓
coherent node timing-bias model
        ↓
automatic calibration acceptance
        ↓
runtime localization correction
```

A stronger product could continuously assess calibration health rather than relying on one manual calibration.

---

## B. Sensor-independent acoustic node architecture

A hardware abstraction capable of accepting different microphone and environmental sensors while preserving standardized outputs could improve product maintainability.

---

## C. Confidence-aware multimodel classification

Potential extension:

```text
DSP heuristic
+
bioacoustic pretrained model
+
environment
+
localization confidence
        ↓
confidence-aware decision fusion
```

This could be stronger than simple weighted voting if experimentally validated.

---

## D. Adaptive localization quality control

The system could dynamically determine whether localization is trustworthy using:

- GCC peak quality,
- pair consistency,
- SNR,
- reverberation indicators,
- geometry,
- environmental conditions.

---

## E. Acoustic behavior mapping

A future research/product layer could combine:

```text
class
location
time
environment
recurrence
```

to identify statistically significant spatiotemporal activity patterns.

These should remain **behavior indicators** unless biological ground truth supports stronger conclusions.

---

## F. Autonomous distributed field arrays

Future architecture could combine multiple local synchronized arrays with:

- low-power networking,
- edge classification,
- event-only transmission,
- cloud or research-server aggregation.

---

# 65. Future Development

Potential future work includes:

### Localization

- uncertainty estimation,
- confidence ellipse,
- GDOP analysis,
- alternative GCC weighting,
- multi-source localization,
- 3-D arrays,
- improved initialization.

### Classification

- additional pretrained models,
- project-specific trained models,
- dynamic ensemble weighting,
- calibration of classifier confidence,
- species-specific validation.

### Embedded systems

- lower-power acquisition,
- battery operation,
- solar charging,
- onboard storage,
- ESP32-S3 or specialized edge accelerator variants where justified.

### Networking

- mDNS/discovery,
- automatic node registration,
- encrypted transport,
- mesh topology,
- store-and-forward operation.

### Analytics

- longer-term occupancy models,
- statistically validated temporal trends,
- richer environmental models,
- multi-day/multi-season comparisons.

### Dashboard

- calibration view,
- localization-quality diagnostics,
- classification comparison,
- experiment-management interface,
- export controls.

---

# 66. Troubleshooting

## Node does not connect

Check:

```text
SSID
password
laptop IP
TCP port
firewall
same Wi-Fi network
```

---

## No audio

Check:

```text
INMP441 power
GND
GPIO33
L/R selection
BCLK
WS
firmware slot selection
```

---

## Slaves show clock faults

Check:

```text
Node 1 I2S master running
BCLK continuity
WS continuity
shared ground
cable length
signal integrity
```

---

## Localization is unstable

Check:

```text
actual microphone coordinates
source SNR
room reflections
microphone spacing
GCC peak ratio
band-pass settings
calibration state
speed-of-sound configuration
```

---

## Localization is consistently shifted

Possible causes:

```text
systematic timing bias
incorrect geometry
wrong TDOA sign convention
microphone channel mismatch
```

Perform controlled calibration before modifying solver mathematics.

---

## Many false acoustic events

Review:

```text
noise floor
trigger margin
release margin
spectral flux
minimum node agreement
environmental noise
```

---

## BirdNET backend falls back to heuristic

Check:

```text
BirdNET installation
ONNX runtime
model initialization
provide_model_audio
classification configuration
```

---

## Dashboard cannot find records

Check:

```text
configured database path
session availability
database schema
current working directory
```

---

# 67. Reproducibility Principles

Research reproducibility is a design goal.

Experiments should record:

- software commit,
- firmware version,
- protocol version,
- microphone coordinates,
- sample rate,
- array spacing,
- calibration parameters,
- classifier backend,
- classifier/model version,
- environmental settings,
- source coordinates,
- source type,
- trial number.

Do not manually modify experimental results to make plots appear cleaner.

Failed events and failed localization attempts are scientifically meaningful data.

---

# 68. Scientific Interpretation Limits

This project distinguishes measurement from interpretation.

## Direct measurements

Examples:

```text
PCM waveform
temperature
humidity
pressure
sampleIndex
```

## Derived measurements

Examples:

```text
RMS
MFCC
TDOA
estimated x/y position
```

## Model inference

Examples:

```text
bird
insect
species prediction
```

## Higher-level research indicators

Examples:

```text
hotspot
temporal concentration
spatial redistribution
environmental association
```

Each step introduces additional assumptions.

Therefore:

```text
detected sound
≠
verified species

species prediction
≠
confirmed individual

consecutive localized events
≠
confirmed animal trajectory

environmental correlation
≠
causation

repeated spatial activity
≠
proven nesting/territorial behavior
```

These distinctions should remain explicit in publications and demonstrations.

---

# 69. Current Development Status

The present codebase contains the major architectural components required for the research prototype.

| Subsystem | Implementation Status | Final Validation |
|---|---|---|
| Protocol v4 | Implemented | Pending final regression |
| Node state management | Implemented | Pending final regression |
| Stream manager | Implemented | Pending final regression |
| ESP32 Node 1 firmware | Implemented | Compile/hardware audit pending |
| ESP32 Node 2 firmware | Implemented | Compile/hardware audit pending |
| ESP32 Node 3 firmware | Implemented | Compile/hardware audit pending |
| Environment handling | Implemented | Pending final regression |
| Event detector | Implemented | Pending final regression |
| DSP preprocessing | Implemented | Pending final regression |
| Feature extraction | Implemented | Pending final regression |
| Heuristic classifier | Implemented | Pending final regression |
| BirdNET adapter | Implemented | Optional dependency/runtime validation pending |
| Ensemble classifier | Implemented | Runtime validation pending |
| GCC-PHAT | Implemented | Pending final regression |
| TDOA representation | Implemented | Pending final regression |
| Position solver | Implemented | Pending final regression |
| Localization engine | Implemented | Pending final regression |
| Calibration algorithm | Implemented | Physical calibration not yet performed |
| Localization benchmark | Implemented | Real benchmark data not yet collected |
| SQLite persistence | Implemented | Final schema compatibility audit pending |
| Research analytics | Implemented | Final compatibility audit pending |
| Streamlit dashboard | Implemented | Final interface audit pending |
| Event exporter | Implemented | Database-contract audit pending |
| Research exporter | Implemented | Analytics-contract audit pending |
| Full test suite | Existing + requires updates | Pending |
| Real hardware experiment | Not yet completed | Pending |
| Research dataset | Not yet collected | Pending |

The project should therefore currently be described as:

> **A substantially implemented research prototype undergoing final integration and experimental validation.**

It should **not yet** be described as a field-validated wildlife tracking product.

---

# 70. Citation and Licensing

## Project citation

A formal project citation can be added after:

- authorship is finalized,
- institution information is finalized,
- repository release version is selected,
- publication status is known.

Suggested placeholder:

```bibtex
@software{wildlife_soundscape_system,
  title  = {Wildlife Soundscape Mapping and Behavior Analysis System},
  year   = {2026},
  note   = {Research prototype}
}
```

---

## BirdNET

If BirdNET-derived results are used in research, the corresponding BirdNET publication and model/software documentation should be cited appropriately.

A commonly cited BirdNET research publication is:

```text
Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021).
BirdNET: A deep learning solution for avian diversity monitoring.
Ecological Informatics, 61, 101236.
```

Third-party software and machine-learning models may use licenses different from this repository.

Review all applicable software and model licenses before:

- redistribution,
- publication,
- commercial deployment,
- incorporation into a product.

---

# Final System Concept

The complete research architecture can be summarized as:

```text
Wildlife acoustic activity
          ↓
3 synchronized microphones
          ↓
shared I2S sampling clock
          ↓
ESP32 acquisition
          ↓
Protocol v4 / TCP
          ↓
sample-indexed stream reconstruction
          ↓
adaptive multi-node event detection
          ↓
synchronized acoustic event
          ↓
DSP feature extraction
          ↓
classification
   ├── heuristic
   ├── BirdNET
   └── ensemble
          ↓
GCC-PHAT
          ↓
raw TDOA
          ↓
timing calibration
          ↓
environment-aware localization
          ↓
(x, y)
          ↓
SQLite + synchronized WAV files
          ↓
research analytics
   ├── temporal
   ├── environmental
   ├── spatial
   └── behavioral indicators
          ↓
Streamlit dashboard
          ↓
CSV / JSON / JSONL research exports
          ↓
experimental evaluation
          ↓
research conclusions
```

---

## Core Design Principle

The system is intentionally divided into layers:

```text
ESP32
=
measure and transport reliably

Laptop DSP
=
extract acoustic information

Localization
=
estimate spatial origin

Classification
=
estimate acoustic identity

Analytics
=
summarize observations

Research methodology
=
determine what conclusions are scientifically justified
```

Keeping those responsibilities separate makes the project easier to:

- debug,
- validate,
- extend,
- benchmark,
- publish,
- reproduce.

---

**Wildlife Soundscape Mapping & Behavior Analysis System**  
**Protocol v4 · ESP32 synchronized acoustic array · Python research stack**
