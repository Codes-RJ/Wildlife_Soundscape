from __future__ import annotations
from dataclasses import dataclass
from .validation import _require_finite, _require_positive_int


@dataclass(
    frozen=True,
    slots=True,
)
class EventDetectionConfig:
    """
    Adaptive multi-node acoustic event detector configuration.
    """

    enabled: bool = True

    # ------------------------------------------------------------------
    # BACKGROUND NOISE MODEL
    # ------------------------------------------------------------------

    noise_history_blocks: int = 96

    noise_quantile: float = 0.30

    initial_noise_dbfs: float = -52.0

    trigger_margin_db: float = 8.0

    release_margin_db: float = 4.0

    # ------------------------------------------------------------------
    # SPECTRAL ACTIVITY
    # ------------------------------------------------------------------

    min_spectral_flux: float = 0.06

    strong_energy_margin_db: float = 15.0

    # ------------------------------------------------------------------
    # TEMPORAL / MULTI-NODE AGREEMENT
    # ------------------------------------------------------------------

    attack_blocks: int = 1

    release_blocks: int = 2

    min_nodes: int = 2

    # ------------------------------------------------------------------
    # EVENT LENGTH
    # ------------------------------------------------------------------

    min_event_ms: float = 30.0

    max_event_s: float = 12.0

    pre_pad_s: float = 0.50

    post_pad_s: float = 0.75

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
        *,
        expected_node_count: int,
    ) -> None:
        """
        Validate detector thresholds and temporal settings.
        """

        if not isinstance(
            self.enabled,
            bool,
        ):
            raise TypeError("Event detection enabled must be bool.")

        _require_positive_int(
            self.noise_history_blocks,
            name=("Event detection noise_history_blocks"),
        )

        noise_quantile = _require_finite(
            self.noise_quantile,
            name=("Event detection noise_quantile"),
        )

        if not (0.0 <= noise_quantile <= 1.0):
            raise ValueError(
                ("Event detection noise_quantile must be between 0 and 1.")
            )

        initial_noise = _require_finite(
            self.initial_noise_dbfs,
            name=("Event detection initial_noise_dbfs"),
        )

        if initial_noise > 0:
            raise ValueError(
                ("Event detection initial_noise_dbfs cannot exceed 0 dBFS.")
            )

        trigger_margin = _require_finite(
            self.trigger_margin_db,
            name=("Event detection trigger_margin_db"),
        )

        release_margin = _require_finite(
            self.release_margin_db,
            name=("Event detection release_margin_db"),
        )

        if trigger_margin <= 0:
            raise ValueError(
                ("Event detection trigger_margin_db must be greater than 0.")
            )

        if release_margin < 0:
            raise ValueError(("Event detection release_margin_db cannot be negative."))

        if release_margin >= trigger_margin:
            raise ValueError(
                (
                    "Event detection release_margin_db "
                    "must remain below "
                    "trigger_margin_db to provide "
                    "detector hysteresis."
                )
            )

        spectral_flux = _require_finite(
            self.min_spectral_flux,
            name=("Event detection min_spectral_flux"),
        )

        if spectral_flux < 0:
            raise ValueError(("Event detection min_spectral_flux cannot be negative."))

        strong_margin = _require_finite(
            self.strong_energy_margin_db,
            name=("Event detection strong_energy_margin_db"),
        )

        if strong_margin < trigger_margin:
            raise ValueError(
                (
                    "Event detection "
                    "strong_energy_margin_db "
                    "should be at least as large "
                    "as trigger_margin_db."
                )
            )

        _require_positive_int(
            self.attack_blocks,
            name="Event detection attack_blocks",
        )

        _require_positive_int(
            self.release_blocks,
            name="Event detection release_blocks",
        )

        _require_positive_int(
            self.min_nodes,
            name="Event detection min_nodes",
        )

        if self.min_nodes < 2:
            raise ValueError(("Current multi-node detector requires min_nodes >= 2."))

        if self.min_nodes > expected_node_count:
            raise ValueError(
                (
                    "Event detection min_nodes "
                    "cannot exceed the number "
                    "of expected nodes."
                )
            )

        minimum_ms = _require_finite(
            self.min_event_ms,
            name=("Event detection min_event_ms"),
        )

        maximum_s = _require_finite(
            self.max_event_s,
            name=("Event detection max_event_s"),
        )

        pre_pad = _require_finite(
            self.pre_pad_s,
            name=("Event detection pre_pad_s"),
        )

        post_pad = _require_finite(
            self.post_pad_s,
            name=("Event detection post_pad_s"),
        )

        if minimum_ms <= 0:
            raise ValueError(("Event detection min_event_ms must be greater than 0."))

        if maximum_s <= 0:
            raise ValueError(("Event detection max_event_s must be greater than 0."))

        if minimum_ms / 1000.0 > maximum_s:
            raise ValueError(
                ("Event detection min_event_ms cannot exceed max_event_s.")
            )

        if pre_pad < 0:
            raise ValueError(("Event detection pre_pad_s cannot be negative."))

        if post_pad < 0:
            raise ValueError(("Event detection post_pad_s cannot be negative."))


@dataclass(
    frozen=True,
    slots=True,
)
class DSPConfig:
    """
    Laptop-side acoustic preprocessing and feature-extraction settings.

    Controls:

        preprocessing
        FFT/STFT
        MFCC
        spectral descriptors
        SNR estimation
        best-node selection
        normalized model-waveform generation

    Raw PCM recordings remain untouched.
    """

    remove_dc: bool = True

    bandpass_enabled: bool = True

    low_cutoff_hz: float = 100.0

    high_cutoff_hz: float = 16_000.0

    filter_order: int = 4

    normalize_for_model: bool = True

    model_target_peak: float = 0.98

    n_fft: int = 2048

    hop_length: int = 512

    n_mfcc: int = 13

    n_mels: int = 64

    mfcc_fmin_hz: float = 50.0

    mfcc_fmax_hz: float = 16_000.0

    roll_percent: float = 0.85

    snr_frame_length: int = 512

    noise_quantile: float = 0.30

    signal_quantile: float = 0.90

    noise_prepad_fraction: float = 0.80

    minimum_event_samples: int = 64

    digital_noise_floor: float = 1.0 / 32768.0

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def validate(
        self,
        sample_rate: int,
    ) -> None:
        """
        Validate DSP settings against acquisition sample rate.
        """

        _require_positive_int(
            sample_rate,
            name="DSP sample_rate",
        )

        nyquist = sample_rate / 2.0

        if not isinstance(
            self.remove_dc,
            bool,
        ):
            raise TypeError("DSP remove_dc must be bool.")

        if not isinstance(
            self.bandpass_enabled,
            bool,
        ):
            raise TypeError("DSP bandpass_enabled must be bool.")

        if self.bandpass_enabled:
            low_hz = _require_finite(
                self.low_cutoff_hz,
                name="DSP low_cutoff_hz",
            )

            high_hz = _require_finite(
                self.high_cutoff_hz,
                name="DSP high_cutoff_hz",
            )

            if low_hz <= 0:
                raise ValueError(("DSP low_cutoff_hz must be greater than 0."))

            if high_hz <= low_hz:
                raise ValueError(
                    ("DSP high_cutoff_hz must be greater than low_cutoff_hz.")
                )

            if high_hz >= nyquist:
                raise ValueError(
                    (
                        "DSP high_cutoff_hz must "
                        "remain below Nyquist "
                        f"({nyquist:.1f} Hz)."
                    )
                )

        _require_positive_int(
            self.filter_order,
            name="DSP filter_order",
        )

        if not isinstance(
            self.normalize_for_model,
            bool,
        ):
            raise TypeError(("DSP normalize_for_model must be bool."))

        model_peak = _require_finite(
            self.model_target_peak,
            name="DSP model_target_peak",
        )

        if not (0.0 < model_peak <= 1.0):
            raise ValueError(("DSP model_target_peak must be in (0, 1]."))

        _require_positive_int(
            self.n_fft,
            name="DSP n_fft",
        )

        _require_positive_int(
            self.hop_length,
            name="DSP hop_length",
        )

        if self.hop_length > self.n_fft:
            raise ValueError(("DSP hop_length cannot exceed n_fft."))

        _require_positive_int(
            self.n_mfcc,
            name="DSP n_mfcc",
        )

        _require_positive_int(
            self.n_mels,
            name="DSP n_mels",
        )

        if self.n_mels < self.n_mfcc:
            raise ValueError(("DSP n_mels must be greater than or equal to n_mfcc."))

        fmin = _require_finite(
            self.mfcc_fmin_hz,
            name="DSP mfcc_fmin_hz",
        )

        fmax = _require_finite(
            self.mfcc_fmax_hz,
            name="DSP mfcc_fmax_hz",
        )

        if fmin < 0:
            raise ValueError(("DSP mfcc_fmin_hz cannot be negative."))

        if fmax <= fmin:
            raise ValueError(("DSP mfcc_fmax_hz must exceed mfcc_fmin_hz."))

        if fmax > nyquist:
            raise ValueError(
                (f"DSP mfcc_fmax_hz cannot exceed Nyquist ({nyquist:.1f} Hz).")
            )

        roll_percent = _require_finite(
            self.roll_percent,
            name="DSP roll_percent",
        )

        if not (0.0 < roll_percent < 1.0):
            raise ValueError(("DSP roll_percent must be between 0 and 1."))

        _require_positive_int(
            self.snr_frame_length,
            name="DSP snr_frame_length",
        )

        noise_quantile = _require_finite(
            self.noise_quantile,
            name="DSP noise_quantile",
        )

        signal_quantile = _require_finite(
            self.signal_quantile,
            name="DSP signal_quantile",
        )

        if not (0.0 <= noise_quantile <= 1.0):
            raise ValueError(("DSP noise_quantile must be between 0 and 1."))

        if not (0.0 <= signal_quantile <= 1.0):
            raise ValueError(("DSP signal_quantile must be between 0 and 1."))

        if signal_quantile <= noise_quantile:
            raise ValueError(("DSP signal_quantile must exceed noise_quantile."))

        noise_fraction = _require_finite(
            self.noise_prepad_fraction,
            name=("DSP noise_prepad_fraction"),
        )

        if not (0.0 < noise_fraction <= 1.0):
            raise ValueError(("DSP noise_prepad_fraction must be in (0, 1]."))

        _require_positive_int(
            self.minimum_event_samples,
            name="DSP minimum_event_samples",
        )

        digital_floor = _require_finite(
            self.digital_noise_floor,
            name="DSP digital_noise_floor",
        )

        if digital_floor <= 0:
            raise ValueError(("DSP digital_noise_floor must be greater than 0."))
