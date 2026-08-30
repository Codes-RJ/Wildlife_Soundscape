"""
Calibration package.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This package contains calibration and validation utilities for the
multi-node acoustic localization subsystem.

The calibration layer exists to quantify and compensate for systematic
timing and localization errors introduced by:

    microphone placement uncertainty
    node-position measurement error
    acoustic propagation conditions
    hardware timing offsets
    channel-specific delays
    installation geometry
    repeatable measurement bias


Current calibration scope
-------------------------
The initial calibration workflow focuses on:

    TDOA offset estimation
    pairwise microphone timing bias
    localization accuracy evaluation
    localization residual analysis
    calibration repeatability


Modules
-------
tdoa_calibration
    Estimate pairwise timing offsets from controlled calibration
    recordings and produce calibration parameters suitable for the
    localization pipeline.

localization_benchmark
    Evaluate localization performance against known reference-source
    positions and report accuracy metrics.


System relationship
-------------------
The calibration layer sits between controlled measurements and the
runtime localization system.

Typical workflow:

    known source position
            ↓
    synchronized 3-node recording
            ↓
    GCC-PHAT / TDOA measurements
            ↓
    calibration analysis
            ↓
    timing-offset parameters
            ↓
    localization configuration
            ↓
    runtime localization engine


Design principle
----------------
Calibration code must not control:

    ESP32 acquisition
    Wi-Fi/TCP transport
    event detection
    acoustic classification
    database persistence
    dashboard rendering

Those responsibilities remain in their respective system layers.


Scientific interpretation
-------------------------
Calibration parameters should be derived from controlled measurements.

They should not be guessed or introduced merely to improve apparent
localization results.

Calibration should use:

    known microphone/node coordinates
    known or carefully measured source coordinates
    repeatable acoustic impulses or test signals
    multiple measurements when practical

Reported calibration performance should distinguish:

    systematic error
    random error
    localization residual
    absolute position error
    measurement repeatability


Import policy
-------------
The package initializer intentionally performs no eager imports.

Consumers should import required calibration functionality directly,
for example:

    from calibration.tdoa_calibration import ...
    from calibration.localization_benchmark import ...

This keeps package initialization lightweight and avoids unnecessary
coupling with DSP and localization modules.
"""