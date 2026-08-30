"""
Local dashboard package.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This package provides the local web interface for:

    live system monitoring
    recent acoustic-event inspection
    environmental telemetry
    localization visualization
    temporal activity analysis
    spatial activity analysis
    research analytics
    event-audio inspection

Planned modules
---------------
data_access
    Read-only bridge between EventDatabase, analytics and dashboard views.

plots
    Reusable Plotly visualization builders.

audio_view
    Event WAV discovery, validation and visualization helpers.

live_view
    Live/recent system-monitoring interface.

analysis_view
    Historical and research-analytics interface.

app
    Main Streamlit application entry point.

Design principle
----------------
Dashboard code is presentation-layer code.

It must not contain:

    acoustic acquisition logic
    event-detection logic
    DSP processing
    classification algorithms
    localization algorithms
    database write logic

Those responsibilities remain in the core system.
"""