from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ======================================================================
# NETWORK CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """
    TCP receiver/network settings used by the laptop server.
    """

    host: str = "0.0.0.0"

    port: int = 5001

    read_timeout_s: float = 10.0

    hello_timeout_s: float = 10.0

    max_payload_bytes: int = (
        64 * 1024
    )


# ======================================================================
# AUDIO CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """
    Core audio acquisition settings.

    These values must remain compatible with the ESP32 firmware.
    """

    # --------------------------------------------------------------
    # ACQUISITION
    # --------------------------------------------------------------

    sample_rate: int = 48_000

    frames_per_block: int = 1024

    channels: int = 1

    sample_width_bytes: int = 2

    # --------------------------------------------------------------
    # LAPTOP-SIDE STREAM BUFFERING
    # --------------------------------------------------------------

    buffer_seconds: float = 20.0

    # Maximum expected coarse alignment difference between nodes.
    sync_tolerance_samples: int = 10

    # --------------------------------------------------------------
    # SESSION RECORDING
    # --------------------------------------------------------------

    record_wav: bool = True

    @property
    def blocks_in_buffer(
        self,
    ) -> int:
        """
        Number of audio blocks retained in the stream buffer.
        """

        return max(
            8,
            int(
                self.buffer_seconds
                * self.sample_rate
                / self.frames_per_block
            )
            + 2,
        )


# ======================================================================
# LOCALIZATION CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class LocalizationConfig:
    """
    TDOA / GCC-PHAT acoustic source-localization settings.
    """

    # --------------------------------------------------------------
    # MICROPHONE GEOMETRY
    # --------------------------------------------------------------
    #
    # Initial simulator / laboratory geometry:
    # 1 m equilateral triangle.
    #
    # Replace these coordinates with measured real microphone
    # coordinates during physical calibration.
    # --------------------------------------------------------------

    node_positions: dict[
        int,
        tuple[float, float],
    ] = field(
        default_factory=lambda: {
            1: (
                0.0,
                0.0,
            ),
            2: (
                0.5,
                0.8660254037844386,
            ),
            3: (
                1.0,
                0.0,
            ),
        }
    )

    reference_node: int = 1

    # Number of synchronized samples used for each localization
    # calculation.
    window_samples: int = 8192

    # Small coarse-alignment correction allowed before GCC-PHAT.
    max_alignment_search_samples: int = 10

    # GCC-PHAT fractional-sample interpolation factor.
    interpolation: int = 8

    # Correlation quality threshold.
    min_peak_ratio: float = 1.10

    # Minimum PCM RMS required before attempting localization.
    min_rms: float = 50.0

    # --------------------------------------------------------------
    # SPEED OF SOUND
    # --------------------------------------------------------------

    # Fallback when BME280 telemetry is unavailable.
    speed_of_sound_mps: float = 343.0

    # Use live temperature/humidity/pressure when available.
    use_environmental_speed: bool = True

    # --------------------------------------------------------------
    # LOCALIZATION-SPECIFIC SIGNAL CONDITIONING
    # --------------------------------------------------------------

    bandpass_enabled: bool = True

    bandpass_low_hz: float = 200.0

    bandpass_high_hz: float = 12_000.0

    bandpass_order: int = 4

    # --------------------------------------------------------------
    # SOLVER
    # --------------------------------------------------------------

    constrain_to_array_bounds: bool = False

    bounds_margin_m: float = 0.5


# ======================================================================
# EVENT DETECTION CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class EventDetectionConfig:
    """
    Adaptive multi-node acoustic-event detection configuration.
    """

    enabled: bool = True

    # --------------------------------------------------------------
    # BACKGROUND NOISE MODEL
    # --------------------------------------------------------------

    noise_history_blocks: int = 96

    noise_quantile: float = 0.30

    initial_noise_dbfs: float = -52.0

    trigger_margin_db: float = 8.0

    release_margin_db: float = 4.0

    # --------------------------------------------------------------
    # SPECTRAL ACTIVITY
    # --------------------------------------------------------------

    min_spectral_flux: float = 0.06

    strong_energy_margin_db: float = 15.0

    # --------------------------------------------------------------
    # TEMPORAL / MULTI-NODE AGREEMENT
    # --------------------------------------------------------------

    attack_blocks: int = 1

    release_blocks: int = 2

    min_nodes: int = 2

    # --------------------------------------------------------------
    # EVENT LENGTH
    # --------------------------------------------------------------

    min_event_ms: float = 30.0

    max_event_s: float = 12.0

    pre_pad_s: float = 0.50

    post_pad_s: float = 0.75


# ======================================================================
# DSP CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class DSPConfig:
    """
    Laptop-side acoustic preprocessing and feature-extraction settings.

    Controls:
        - preprocessing
        - FFT / STFT
        - MFCC
        - spectral descriptors
        - SNR estimation
        - best-node selection
        - normalized model waveform generation

    Raw PCM recordings are never modified by these settings.
    """

    # --------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------

    remove_dc: bool = True

    bandpass_enabled: bool = True

    # Broad wildlife-analysis range.
    low_cutoff_hz: float = 100.0

    high_cutoff_hz: float = 16_000.0

    filter_order: int = 4

    # --------------------------------------------------------------
    # MODEL INPUT NORMALIZATION
    # --------------------------------------------------------------

    normalize_for_model: bool = True

    model_target_peak: float = 0.98

    # --------------------------------------------------------------
    # FFT / STFT
    # --------------------------------------------------------------

    n_fft: int = 2048

    hop_length: int = 512

    # --------------------------------------------------------------
    # MFCC
    # --------------------------------------------------------------

    n_mfcc: int = 13

    n_mels: int = 64

    mfcc_fmin_hz: float = 50.0

    mfcc_fmax_hz: float = 16_000.0

    # --------------------------------------------------------------
    # SPECTRAL FEATURES
    # --------------------------------------------------------------

    roll_percent: float = 0.85

    # --------------------------------------------------------------
    # SNR / BEST-NODE ESTIMATION
    # --------------------------------------------------------------

    snr_frame_length: int = 512

    noise_quantile: float = 0.30

    signal_quantile: float = 0.90

    noise_prepad_fraction: float = 0.80

    minimum_event_samples: int = 64

    # PCM16 one-count equivalent after float conversion.
    digital_noise_floor: float = (
        1.0 / 32768.0
    )

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    def validate(
        self,
        sample_rate: int,
    ) -> None:
        """
        Validate DSP configuration against acquisition sample rate.
        """

        if sample_rate <= 0:

            raise ValueError(
                "sample_rate must be greater than 0."
            )

        nyquist = (
            sample_rate
            / 2.0
        )

        # ==========================================================
        # PREPROCESSING BANDPASS
        # ==========================================================

        if self.bandpass_enabled:

            if self.low_cutoff_hz <= 0:

                raise ValueError(
                    "DSP low_cutoff_hz must be greater than 0."
                )

            if (
                self.high_cutoff_hz
                <= self.low_cutoff_hz
            ):

                raise ValueError(
                    "DSP high_cutoff_hz must be greater "
                    "than low_cutoff_hz."
                )

            if (
                self.high_cutoff_hz
                >= nyquist
            ):

                raise ValueError(
                    "DSP high_cutoff_hz must remain below "
                    f"Nyquist ({nyquist:.1f} Hz)."
                )

        if self.filter_order <= 0:

            raise ValueError(
                "DSP filter_order must be greater than 0."
            )

        # ==========================================================
        # NORMALIZATION
        # ==========================================================

        if not (
            0.0
            < self.model_target_peak
            <= 1.0
        ):

            raise ValueError(
                "DSP model_target_peak must be in (0, 1]."
            )

        # ==========================================================
        # STFT
        # ==========================================================

        if self.n_fft <= 0:

            raise ValueError(
                "DSP n_fft must be greater than 0."
            )

        if self.hop_length <= 0:

            raise ValueError(
                "DSP hop_length must be greater than 0."
            )

        if (
            self.hop_length
            > self.n_fft
        ):

            raise ValueError(
                "DSP hop_length cannot exceed n_fft."
            )

        # ==========================================================
        # MFCC
        # ==========================================================

        if self.n_mfcc <= 0:

            raise ValueError(
                "DSP n_mfcc must be greater than 0."
            )

        if self.n_mels <= 0:

            raise ValueError(
                "DSP n_mels must be greater than 0."
            )

        if (
            self.n_mels
            < self.n_mfcc
        ):

            raise ValueError(
                "DSP n_mels must be greater than or equal "
                "to n_mfcc."
            )

        if self.mfcc_fmin_hz < 0:

            raise ValueError(
                "DSP mfcc_fmin_hz cannot be negative."
            )

        if (
            self.mfcc_fmax_hz
            <= self.mfcc_fmin_hz
        ):

            raise ValueError(
                "DSP mfcc_fmax_hz must exceed mfcc_fmin_hz."
            )

        if (
            self.mfcc_fmax_hz
            > nyquist
        ):

            raise ValueError(
                "DSP mfcc_fmax_hz cannot exceed "
                f"Nyquist ({nyquist:.1f} Hz)."
            )

        # ==========================================================
        # SPECTRAL FEATURES
        # ==========================================================

        if not (
            0.0
            < self.roll_percent
            < 1.0
        ):

            raise ValueError(
                "DSP roll_percent must be between 0 and 1."
            )

        # ==========================================================
        # CHANNEL QUALITY / SNR
        # ==========================================================

        if self.snr_frame_length <= 0:

            raise ValueError(
                "DSP snr_frame_length must be greater than 0."
            )

        if not (
            0.0
            <= self.noise_quantile
            <= 1.0
        ):

            raise ValueError(
                "DSP noise_quantile must be between 0 and 1."
            )

        if not (
            0.0
            <= self.signal_quantile
            <= 1.0
        ):

            raise ValueError(
                "DSP signal_quantile must be between 0 and 1."
            )

        if (
            self.signal_quantile
            <= self.noise_quantile
        ):

            raise ValueError(
                "DSP signal_quantile must exceed noise_quantile."
            )

        if not (
            0.0
            < self.noise_prepad_fraction
            <= 1.0
        ):

            raise ValueError(
                "DSP noise_prepad_fraction must be in (0, 1]."
            )

        if (
            self.minimum_event_samples
            <= 0
        ):

            raise ValueError(
                "DSP minimum_event_samples must be greater than 0."
            )

        if (
            self.digital_noise_floor
            <= 0
        ):

            raise ValueError(
                "DSP digital_noise_floor must be greater than 0."
            )


# ======================================================================
# CLASSIFICATION CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    """
    Acoustic-classification subsystem configuration.

    Current backend
    ---------------
    heuristic

    Future backends may include:
        - pretrained
        - birdnet
        - ensemble

    The pipeline should obtain its classifier through a backend factory
    rather than instantiating a specific classifier directly.
    """

    # --------------------------------------------------------------
    # MASTER SWITCH
    # --------------------------------------------------------------

    enabled: bool = True

    # --------------------------------------------------------------
    # BACKEND SELECTION
    # --------------------------------------------------------------

    # Current supported implementation.
    backend: str = "heuristic"

    # If a future external/pretrained backend cannot initialize,
    # whether the system may fall back to the heuristic baseline.
    fallback_to_heuristic: bool = True

    # --------------------------------------------------------------
    # MODEL AUDIO
    # --------------------------------------------------------------

    # Keep normalized waveform available for classifiers that require
    # waveform input.
    provide_model_audio: bool = True

    # --------------------------------------------------------------
    # FUTURE PRETRAINED-MODEL SETTINGS
    # --------------------------------------------------------------
    #
    # These fields do not force the current heuristic classifier to
    # load a model. They reserve configuration at the correct layer so
    # future model adapters do not require another AppConfig redesign.
    # --------------------------------------------------------------

    model_path: Path | None = None

    labels_path: Path | None = None

    # None means the backend uses the acquisition sample rate or its
    # own documented native rate.
    model_sample_rate: int | None = None

    # Number of top predictions a future model backend may retain.
    top_k: int = 5

    # Generic minimum score for future learned-model predictions.
    #
    # This is NOT the internal threshold used by the current heuristic
    # classifier.
    model_min_confidence: float = 0.20

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    def validate(
        self,
    ) -> None:
        """
        Validate classification configuration.

        Paths are deliberately not required to exist while using the
        heuristic backend.
        """

        backend = (
            self.backend
            .strip()
            .lower()
        )

        if not backend:

            raise ValueError(
                "Classification backend cannot be empty."
            )

        supported_backends = {
            "heuristic",
            "pretrained",
            "birdnet",
            "ensemble",
        }

        if (
            backend
            not in supported_backends
        ):

            raise ValueError(
                (
                    "Unsupported classification backend "
                    f"'{self.backend}'. "
                    f"Supported values: "
                    f"{sorted(supported_backends)}"
                )
            )

        if (
            self.model_sample_rate
            is not None
            and self.model_sample_rate <= 0
        ):

            raise ValueError(
                "Classification model_sample_rate must be "
                "greater than 0 when provided."
            )

        if self.top_k <= 0:

            raise ValueError(
                "Classification top_k must be greater than 0."
            )

        if not (
            0.0
            <= self.model_min_confidence
            <= 1.0
        ):

            raise ValueError(
                "Classification model_min_confidence must "
                "be between 0 and 1."
            )


# ======================================================================
# PERSISTENCE CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class PersistenceConfig:
    """
    Database and event-file persistence settings.
    """

    database_path: Path = Path(
        "data/database/events.db"
    )

    events_dir: Path = Path(
        "data/events"
    )

    save_event_wav: bool = True


# ======================================================================
# APPLICATION CONFIGURATION
# ======================================================================


@dataclass(frozen=True, slots=True)
class AppConfig:
    """
    Root configuration object for the complete laptop-side system.
    """

    network: NetworkConfig = field(
        default_factory=NetworkConfig
    )

    audio: AudioConfig = field(
        default_factory=AudioConfig
    )

    localization: LocalizationConfig = field(
        default_factory=LocalizationConfig
    )

    detection: EventDetectionConfig = field(
        default_factory=EventDetectionConfig
    )

    dsp: DSPConfig = field(
        default_factory=DSPConfig
    )

    classification: ClassificationConfig = field(
        default_factory=ClassificationConfig
    )

    persistence: PersistenceConfig = field(
        default_factory=PersistenceConfig
    )

    # Expected ESP32 acoustic nodes.
    expected_nodes: frozenset[int] = (
        frozenset(
            {
                1,
                2,
                3,
            }
        )
    )

    # Continuous/session WAV recordings.
    recordings_dir: Path = Path(
        "data/recordings"
    )

    print_status_every_s: float = 10.0

    # ==================================================================
    # CROSS-CONFIG VALIDATION
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate configuration values that depend on other sections.
        """

        # ==========================================================
        # NETWORK
        # ==========================================================

        if not (
            1
            <= self.network.port
            <= 65_535
        ):

            raise ValueError(
                "Network port must be between 1 and 65535."
            )

        if (
            self.network.read_timeout_s
            <= 0
        ):

            raise ValueError(
                "Network read_timeout_s must be greater than 0."
            )

        if (
            self.network.hello_timeout_s
            <= 0
        ):

            raise ValueError(
                "Network hello_timeout_s must be greater than 0."
            )

        if (
            self.network.max_payload_bytes
            <= 0
        ):

            raise ValueError(
                "Network max_payload_bytes must be greater than 0."
            )

        # ==========================================================
        # AUDIO
        # ==========================================================

        if self.audio.sample_rate <= 0:

            raise ValueError(
                "Audio sample_rate must be greater than 0."
            )

        if (
            self.audio.frames_per_block
            <= 0
        ):

            raise ValueError(
                "Audio frames_per_block must be greater than 0."
            )

        if (
            self.audio.channels
            != 1
        ):

            raise ValueError(
                "Current Wildlife Soundscape protocol expects "
                "mono audio per ESP32 node."
            )

        if (
            self.audio.sample_width_bytes
            != 2
        ):

            raise ValueError(
                "Current transport expects PCM16 "
                "(2 bytes per sample)."
            )

        if (
            self.audio.buffer_seconds
            <= 0
        ):

            raise ValueError(
                "Audio buffer_seconds must be greater than 0."
            )

        if (
            self.audio.sync_tolerance_samples
            < 0
        ):

            raise ValueError(
                "Audio sync_tolerance_samples cannot be negative."
            )

        # ==========================================================
        # EXPECTED NODES
        # ==========================================================

        if len(
            self.expected_nodes
        ) < 3:

            raise ValueError(
                "At least three acoustic nodes are required "
                "for the current 2-D TDOA localization design."
            )

        if any(
            node_id <= 0
            for node_id
            in self.expected_nodes
        ):

            raise ValueError(
                "Expected node IDs must be positive integers."
            )

        if (
            self.localization.reference_node
            not in self.expected_nodes
        ):

            raise ValueError(
                "Localization reference_node must be one "
                "of expected_nodes."
            )

        missing_positions = (
            self.expected_nodes
            - set(
                self.localization.node_positions.keys()
            )
        )

        if missing_positions:

            raise ValueError(
                "Missing localization coordinates for node(s): "
                f"{sorted(missing_positions)}"
            )

        # ==========================================================
        # LOCALIZATION
        # ==========================================================

        if (
            self.localization.window_samples
            <= 0
        ):

            raise ValueError(
                "Localization window_samples must be greater than 0."
            )

        if (
            self.localization.max_alignment_search_samples
            < 0
        ):

            raise ValueError(
                "Localization max_alignment_search_samples "
                "cannot be negative."
            )

        if (
            self.localization.interpolation
            <= 0
        ):

            raise ValueError(
                "Localization interpolation must be greater than 0."
            )

        if (
            self.localization.min_peak_ratio
            <= 0
        ):

            raise ValueError(
                "Localization min_peak_ratio must be greater than 0."
            )

        if (
            self.localization.min_rms
            < 0
        ):

            raise ValueError(
                "Localization min_rms cannot be negative."
            )

        if (
            self.localization.speed_of_sound_mps
            <= 0
        ):

            raise ValueError(
                "Fallback speed_of_sound_mps must be greater than 0."
            )

        # ----------------------------------------------------------
        # LOCALIZATION FREQUENCY RANGE
        # ----------------------------------------------------------

        nyquist = (
            self.audio.sample_rate
            / 2.0
        )

        if (
            self.localization.bandpass_enabled
        ):

            if not (
                0.0
                < self.localization.bandpass_low_hz
                < self.localization.bandpass_high_hz
                < nyquist
            ):

                raise ValueError(
                    "Localization bandpass frequencies must satisfy "
                    "0 < low < high < Nyquist."
                )

        if (
            self.localization.bandpass_order
            <= 0
        ):

            raise ValueError(
                "Localization bandpass_order must be greater than 0."
            )

        if (
            self.localization.bounds_margin_m
            < 0
        ):

            raise ValueError(
                "Localization bounds_margin_m cannot be negative."
            )

        # ==========================================================
        # EVENT DETECTOR
        # ==========================================================

        if (
            self.detection.min_nodes
            > len(
                self.expected_nodes
            )
        ):

            raise ValueError(
                "Event detection min_nodes cannot exceed "
                "the number of expected nodes."
            )

        if (
            self.detection.min_nodes
            <= 0
        ):

            raise ValueError(
                "Event detection min_nodes must be greater than 0."
            )

        if (
            self.detection.noise_history_blocks
            <= 0
        ):

            raise ValueError(
                "Event detection noise_history_blocks must "
                "be greater than 0."
            )

        if not (
            0.0
            <= self.detection.noise_quantile
            <= 1.0
        ):

            raise ValueError(
                "Event detection noise_quantile must be "
                "between 0 and 1."
            )

        if (
            self.detection.attack_blocks
            <= 0
        ):

            raise ValueError(
                "Event detection attack_blocks must be greater than 0."
            )

        if (
            self.detection.release_blocks
            <= 0
        ):

            raise ValueError(
                "Event detection release_blocks must be greater than 0."
            )

        if (
            self.detection.min_event_ms
            <= 0
        ):

            raise ValueError(
                "Event detection min_event_ms must be greater than 0."
            )

        if (
            self.detection.pre_pad_s
            < 0
        ):

            raise ValueError(
                "Event detection pre_pad_s cannot be negative."
            )

        if (
            self.detection.post_pad_s
            < 0
        ):

            raise ValueError(
                "Event detection post_pad_s cannot be negative."
            )

        if (
            self.detection.max_event_s
            <= 0
        ):

            raise ValueError(
                "Event detection max_event_s must be greater than 0."
            )

        # ==========================================================
        # DSP
        # ==========================================================

        self.dsp.validate(
            self.audio.sample_rate
        )

        # ==========================================================
        # CLASSIFICATION
        # ==========================================================

        self.classification.validate()

        # A model-specific sample rate does not need to equal the
        # acquisition rate because future backends may resample.
        #
        # We therefore validate positivity only and intentionally do
        # NOT enforce:
        #
        # model_sample_rate == audio.sample_rate

        # ==========================================================
        # STATUS INTERVAL
        # ==========================================================

        if (
            self.print_status_every_s
            <= 0
        ):

            raise ValueError(
                "print_status_every_s must be greater than 0."
            )


# ======================================================================
# GLOBAL APPLICATION CONFIGURATION
# ======================================================================


CONFIG = AppConfig()