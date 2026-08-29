"""
Digital Signal Processing package.

Wildlife Soundscape Mapping & Behavior Analysis System
"""

# ======================================================================
# PREPROCESSING
# ======================================================================

from .preprocessing import (
    PreprocessedAudio,
    PreprocessingConfig,
    apply_bandpass_filter,
    pcm16_to_float32,
    peak_normalize,
    preprocess_event_audio,
    remove_dc_offset,
)

# ======================================================================
# FEATURES
# ======================================================================

from .features import (
    AcousticFeatures,
    FeatureConfig,
    calculate_crest_factor,
    calculate_dominant_frequency,
    calculate_log_spectrogram,
    calculate_peak_amplitude,
    calculate_rms,
    calculate_snr_db,
    calculate_spectral_flux,
    extract_acoustic_features,
)


__all__ = [

    # Preprocessing
    "PreprocessedAudio",
    "PreprocessingConfig",
    "pcm16_to_float32",
    "remove_dc_offset",
    "apply_bandpass_filter",
    "peak_normalize",
    "preprocess_event_audio",

    # Features
    "AcousticFeatures",
    "FeatureConfig",
    "calculate_rms",
    "calculate_peak_amplitude",
    "calculate_crest_factor",
    "calculate_snr_db",
    "calculate_dominant_frequency",
    "calculate_spectral_flux",
    "calculate_log_spectrogram",
    "extract_acoustic_features",
]