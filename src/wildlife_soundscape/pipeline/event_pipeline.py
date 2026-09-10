from __future__ import annotations

from wildlife_soundscape.core.provenance import experiment_manifest


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import logging
import math
import wave

from pathlib import (
    Path,
)


# ======================================================================
# THIRD PARTY
# ======================================================================


import numpy as np


# ======================================================================
# CLASSIFICATION
# ======================================================================


from wildlife_soundscape.classification import (
    ClassificationInput,
    ClassificationResult,
    ClassifierBackend,
)


from wildlife_soundscape.classification.factory import (
    create_classifier_backend,
)


# ======================================================================
# CONFIGURATION
# ======================================================================


from wildlife_soundscape.core.config import (
    AppConfig,
)


# ======================================================================
# DATABASE
# ======================================================================


from wildlife_soundscape.storage.database import (
    EventDatabase,
)


# ======================================================================
# DSP
# ======================================================================


from wildlife_soundscape.dsp.features import (
    AcousticFeatures,
    FeatureConfig,
    extract_acoustic_features,
)


from wildlife_soundscape.dsp.preprocessing import (
    PreprocessedAudio,
    PreprocessingConfig,
    preprocess_event_audio,
)


# ======================================================================
# EVENT DETECTION
# ======================================================================


from wildlife_soundscape.pipeline.event_detector import (
    AcousticEvent,
    MultiNodeEventDetector,
)


# ======================================================================
# LOCALIZATION
# ======================================================================


from wildlife_soundscape.localization import (
    LocalizationEngine,
    LocalizationResult,
)


# ======================================================================
# CORE MODELS / PROTOCOL / STREAMS
# ======================================================================


from wildlife_soundscape.core.models import (
    AudioBlock,
)


from wildlife_soundscape.core.protocol import (
    EnvironmentPayload,
)


from wildlife_soundscape.acquisition.stream_manager import (
    StreamManager,
)


# ======================================================================
# LOGGER
# ======================================================================


logger = logging.getLogger(__name__)


# ======================================================================
# PROTOCOL LIMITS
# ======================================================================


UINT32_MAX = 0xFFFFFFFF


# ======================================================================
# EVENT PIPELINE
# ======================================================================


class EventPipeline:
    """
    Complete automatic acoustic-event processing pipeline.

    Flow
    ----
    AudioBlock
        ↓
    Multi-node event detection
        ↓
    Synchronized event extraction
        ↓
    Environmental association
        ↓
    3-node TDOA localization
        ↓
    Per-node preprocessing
        ↓
    Best-SNR microphone selection
        ↓
             ┌─────────────────────┐
             │                     │
             ▼                     ▼
    DSP feature extraction     Model waveform
             │                     │
             └──────────┬──────────┘
                        ▼
              ClassificationInput
                        ↓
              ClassifierBackend
                        ↓
             ClassificationResult
                        ↓
                 WAV persistence
                        ↓
                 SQLite database


    Classification architecture
    ---------------------------
    ClassificationInput intentionally permits:

        features = AcousticFeatures | None
        model_audio = ndarray | None

    Each ClassifierBackend owns the validation of the inputs it actually
    requires.

    Examples:

        heuristic
            features required
            waveform unnecessary

        BirdNET
            waveform required
            handcrafted features unnecessary

        ensemble
            members validate independently

    This distinction is essential for ensemble graceful degradation.

    For example, when an optional waveform model fails or its model input
    is unavailable, a feature-based ensemble member can still classify
    the event when the ensemble failure policy permits it.


    Scientific data separation
    --------------------------
    Raw synchronized event WAV data remains independent from:

        normalized model waveform
        DSP features
        classifier inference

    Model preprocessing therefore does not modify the persisted source
    recordings.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        streams: StreamManager,
        config: AppConfig,
        classifier_backend: ClassifierBackend | None = None,
    ) -> None:

        # ==============================================================
        # INPUT VALIDATION
        # ==============================================================

        if not isinstance(
            streams,
            StreamManager,
        ):
            raise TypeError(("streams must be a StreamManager instance."))

        if not isinstance(
            config,
            AppConfig,
        ):
            raise TypeError(("config must be an AppConfig instance."))

        if classifier_backend is not None and not isinstance(
            classifier_backend,
            ClassifierBackend,
        ):
            raise TypeError(("classifier_backend must be ClassifierBackend or None."))

        self.streams = streams

        self.config = config

        # ==============================================================
        # EVENT DETECTOR
        # ==============================================================

        self.detector = MultiNodeEventDetector(
            config.audio,
            config.detection,
            tuple(sorted(config.expected_nodes)),
        )

        # ==============================================================
        # LOCALIZATION
        # ==============================================================

        self.localizer = LocalizationEngine(
            streams,
            config.localization,
        )

        # ==============================================================
        # DATABASE
        # ==============================================================

        self.database = EventDatabase(config.persistence.database_path)

        # ==============================================================
        # DSP PREPROCESSING
        # ==============================================================

        self.preprocessing_config = PreprocessingConfig(
            sample_rate=config.audio.sample_rate,
            remove_dc=config.dsp.remove_dc,
            bandpass_enabled=config.dsp.bandpass_enabled,
            low_cutoff_hz=config.dsp.low_cutoff_hz,
            high_cutoff_hz=config.dsp.high_cutoff_hz,
            filter_order=config.dsp.filter_order,
            normalize_for_model=config.dsp.normalize_for_model,
            model_target_peak=config.dsp.model_target_peak,
        )

        # ==============================================================
        # DSP FEATURE EXTRACTION
        # ==============================================================

        self.feature_config = FeatureConfig(
            sample_rate=config.audio.sample_rate,
            n_fft=config.dsp.n_fft,
            hop_length=config.dsp.hop_length,
            n_mfcc=config.dsp.n_mfcc,
            n_mels=config.dsp.n_mels,
            roll_percent=config.dsp.roll_percent,
            mfcc_fmin_hz=config.dsp.mfcc_fmin_hz,
            mfcc_fmax_hz=config.dsp.mfcc_fmax_hz,
        )

        # ==============================================================
        # CLASSIFICATION BACKEND
        # ==============================================================

        if classifier_backend is not None:
            self.classifier_backend: ClassifierBackend | None = classifier_backend

        else:
            self.classifier_backend = create_classifier_backend(config.classification)

        # ==============================================================
        # SESSION STATE
        # ==============================================================

        self.active_session_id: int | None = None

        self.active_session_label: str | None = None

        # ==============================================================
        # RUNTIME EVENT STATE
        # ==============================================================

        self.completed_events = 0

        self.last_event_db_id: int | None = None

        self.last_best_node_id: int | None = None

        self.last_features: AcousticFeatures | None = None

        self.last_classification: ClassificationResult | None = None

    # ==================================================================
    # CLASSIFICATION STATE
    # ==================================================================

    @property
    def classification_enabled(
        self,
    ) -> bool:
        """
        Whether classification is enabled and an operational backend is
        available.
        """

        return (
            self.config.classification.enabled and self.classifier_backend is not None
        )

    # ==================================================================
    # SESSION CONTROL
    # ==================================================================

    def start_session(
        self,
        session_id: int,
        label: str,
    ) -> None:
        """
        Start a new event-processing session.

        Session identifier
        ------------------
        Protocol v4 represents ``sessionId`` using uint32.

        Therefore this method accepts:

            1 .. 0xFFFFFFFF

        only.


        Persistence ordering
        --------------------
        Database session creation occurs before in-memory pipeline state
        becomes active.

        A persistence failure therefore cannot leave this EventPipeline
        incorrectly marked as active.
        """

        # ==============================================================
        # SESSION ID
        # ==============================================================

        if isinstance(
            session_id,
            bool,
        ) or not isinstance(
            session_id,
            int,
        ):
            raise TypeError(("session_id must be an integer."))

        if not (1 <= session_id <= UINT32_MAX):
            raise ValueError(("session_id must lie between 1 and 0xFFFFFFFF."))

        # ==============================================================
        # LABEL
        # ==============================================================

        if not isinstance(
            label,
            str,
        ):
            raise TypeError(("session label must be a string."))

        label = label.strip()

        if not (label):
            raise ValueError(("session label cannot be empty."))

        # ==============================================================
        # EXISTING SESSION
        # ==============================================================

        if self.active_session_id is not None:
            raise RuntimeError(
                (
                    "EventPipeline already has "
                    "an active session "
                    f"0x{self.active_session_id:08X}"
                )
            )

        # ==============================================================
        # PERSIST SESSION FIRST
        # ==============================================================

        self.database.start_session(
            session_id,
            label,
            manifest=experiment_manifest(self.config),
        )

        # ==============================================================
        # ACTIVATE PIPELINE STATE
        # ==============================================================

        self.active_session_id = session_id

        self.active_session_label = label

        self.completed_events = 0

        self.last_event_db_id = None

        self.last_best_node_id = None

        self.last_features = None

        self.last_classification = None

        self.detector.reset()

        # ==============================================================
        # LOG
        # ==============================================================

        if self.classification_enabled and self.classifier_backend is not None:
            logger.info(
                (
                    "Event pipeline session started "
                    "| session=0x%08X "
                    "| classifier=%s "
                    "| backend_version=%s"
                ),
                session_id,
                self.classifier_backend.name,
                self.classifier_backend.version,
            )

        else:
            logger.info(
                (
                    "Event pipeline session started "
                    "| session=0x%08X "
                    "| classification=disabled"
                ),
                session_id,
            )

    # ==================================================================
    # STOP SESSION
    # ==================================================================

    def stop_session(
        self,
    ) -> None:
        """
        Stop the current event-processing session.

        Calling this method while no session is active is safe.

        Runtime state is cleared even if persistence shutdown raises.
        """

        session_id = self.active_session_id

        # ==============================================================
        # ALREADY INACTIVE
        # ==============================================================

        if session_id is None:
            self.active_session_label = None

            self.detector.reset()

            return

        # ==============================================================
        # DATABASE + RUNTIME TEARDOWN
        # ==============================================================

        try:
            self.database.stop_session(session_id)

        finally:
            self.active_session_id = None

            self.active_session_label = None

            self.detector.reset()

    # ==================================================================
    # AUDIO ENTRY POINT
    # ==================================================================

    def on_audio(
        self,
        block: AudioBlock,
    ) -> list[AcousticEvent]:
        """
        Feed one audio block into the event detector.

        Audio outside the active acquisition session is ignored.
        """

        if not isinstance(
            block,
            AudioBlock,
        ):
            raise TypeError("block must be an AudioBlock.")

        # ==============================================================
        # NO ACTIVE SESSION
        # ==============================================================

        if self.active_session_id is None:
            return []

        # ==============================================================
        # STALE SESSION
        # ==============================================================

        if block.session_id != self.active_session_id:
            logger.debug(
                (
                    "Ignoring stale audio block "
                    "node=%s "
                    "block_session=0x%08X "
                    "active_session=0x%08X"
                ),
                block.node_id,
                block.session_id,
                self.active_session_id,
            )

            return []

        # ==============================================================
        # DETECTOR
        # ==============================================================

        events = self.detector.process(block)

        # ==============================================================
        # COMPLETE EVENTS
        # ==============================================================

        for event in events:
            try:
                self._persist_event(event)

            except Exception:
                logger.exception(
                    ("Failed to process event %s"),
                    event.event_id,
                )

        return events

    # ==================================================================
    # LOCALIZATION WINDOW SELECTION
    # ==================================================================

    def _localization_start(
        self,
        event: AcousticEvent,
    ) -> tuple[
        int,
        int,
    ]:
        """
        Select a high-energy event region for TDOA localization.

        Uses an O(n) cumulative-energy search rather than repeatedly
        convolving the event with a rectangular kernel.
        """

        window_samples = int(self.config.localization.window_samples)

        event_length = int(event.sample_count)

        # ==============================================================
        # INVALID / EMPTY EVENT
        # ==============================================================

        if event_length <= 0:
            return (
                int(event.start_sample),
                0,
            )

        if window_samples <= 0:
            return (
                int(event.start_sample),
                0,
            )

        # ==============================================================
        # EVENT SHORTER THAN LOCALIZATION WINDOW
        # ==============================================================

        if event_length <= window_samples:
            return (
                max(
                    0,
                    int(event.start_sample),
                ),
                event_length,
            )

        # ==============================================================
        # REFERENCE MICROPHONE WINDOW
        # ==============================================================

        reference_node = self.config.localization.reference_node

        samples = self.streams.get_window(
            reference_node,
            event.start_sample,
            event_length,
        )

        x = np.asarray(
            samples,
            dtype=np.float64,
        )

        if x.size == 0:
            return (
                int(event.start_sample),
                0,
            )

        if x.size <= window_samples:
            return (
                int(event.start_sample),
                int(x.size),
            )

        # ==============================================================
        # O(n) ROLLING ENERGY
        # ==============================================================

        energy = x * x

        cumulative = np.empty(
            energy.size + 1,
            dtype=np.float64,
        )

        cumulative[0] = 0.0

        np.cumsum(
            energy,
            dtype=np.float64,
            out=cumulative[1:],
        )

        rolling_energy = cumulative[window_samples:] - cumulative[:-window_samples]

        if rolling_energy.size == 0:
            relative_start = 0

        else:
            relative_start = int(np.argmax(rolling_energy))

        return (
            int(event.start_sample + relative_start),
            window_samples,
        )

    # ==================================================================
    # SHORT-TIME RMS
    # ==================================================================

    def _frame_rms_values(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate non-overlapping short-time RMS values.
        """

        frame_length = int(self.config.dsp.snr_frame_length)

        if frame_length <= 0:
            return np.array(
                [],
                dtype=np.float64,
            )

        x = np.asarray(
            signal,
            dtype=np.float64,
        )

        if x.size == 0:
            return np.array(
                [],
                dtype=np.float64,
            )

        # ==============================================================
        # VERY SHORT SIGNAL
        # ==============================================================

        if x.size < frame_length:
            value = float(np.sqrt(np.mean(x * x)))

            return np.array(
                [value],
                dtype=np.float64,
            )

        # ==============================================================
        # COMPLETE NON-OVERLAPPING FRAMES
        # ==============================================================

        frame_count = x.size // frame_length

        usable_samples = frame_count * frame_length

        frames = x[:usable_samples].reshape(
            frame_count,
            frame_length,
        )

        rms_values = np.sqrt(
            np.mean(
                frames * frames,
                axis=1,
            )
        )

        return np.asarray(
            rms_values,
            dtype=np.float64,
        )

    # ==================================================================
    # NOISE ESTIMATION
    # ==================================================================

    def _estimate_noise_rms(
        self,
        signal: np.ndarray,
    ) -> float | None:
        """
        Estimate background RMS from the event pre-trigger region.
        """

        x = np.asarray(
            signal,
            dtype=np.float32,
        )

        if x.size == 0:
            return None

        pre_pad_samples = int(
            round(self.config.detection.pre_pad_s * self.config.audio.sample_rate)
        )

        preferred_noise_samples = int(
            pre_pad_samples * self.config.dsp.noise_prepad_fraction
        )

        preferred_noise_samples = max(
            int(self.config.dsp.snr_frame_length),
            preferred_noise_samples,
        )

        preferred_noise_samples = min(
            preferred_noise_samples,
            x.size,
        )

        noise_region = x[:preferred_noise_samples]

        rms_values = self._frame_rms_values(noise_region)

        if rms_values.size == 0:
            return None

        finite_values = rms_values[np.isfinite(rms_values)]

        if finite_values.size == 0:
            return None

        noise_rms = float(
            np.quantile(
                finite_values,
                self.config.dsp.noise_quantile,
            )
        )

        if not math.isfinite(noise_rms):
            return None

        return max(
            0.0,
            noise_rms,
        )

    # ==================================================================
    # CHANNEL QUALITY
    # ==================================================================

    def _channel_quality_score(
        self,
        signal: np.ndarray,
        noise_rms: float | None,
    ) -> float:
        """
        Calculate robust SNR-like channel-selection score.

        Best-node selection affects:

            feature extraction
            classification

        Localization continues to use all synchronized microphones.
        """

        rms_values = self._frame_rms_values(signal)

        if rms_values.size == 0:
            return float("-inf")

        finite_values = rms_values[np.isfinite(rms_values)]

        if finite_values.size == 0:
            return float("-inf")

        signal_level = float(
            np.quantile(
                finite_values,
                self.config.dsp.signal_quantile,
            )
        )

        if not math.isfinite(signal_level) or signal_level <= 0.0:
            return float("-inf")

        digital_floor = float(self.config.dsp.digital_noise_floor)

        if noise_rms is None:
            effective_noise = digital_floor

        else:
            effective_noise = max(
                float(noise_rms),
                digital_floor,
            )

        score = 20.0 * math.log10(signal_level / effective_noise)

        if not math.isfinite(score):
            return float("-inf")

        return float(score)

    # ==================================================================
    # MODEL AUDIO EXTRACTION
    # ==================================================================

    def _prepare_model_audio(
        self,
        processed: PreprocessedAudio,
        event: AcousticEvent,
    ) -> np.ndarray | None:
        """
        Prepare normalized waveform input for classifiers that need audio.

        No waveform copy is created when the active backend does not use
        audio.

        Feature extraction success remains independent from this path.
        """

        # ==============================================================
        # CLASSIFICATION DISABLED
        # ==============================================================

        if not (self.classification_enabled):
            return None

        # ==============================================================
        # CONFIGURATION POLICY
        # ==============================================================

        if not (self.config.classification.provide_model_audio):
            return None

        # ==============================================================
        # MODEL SIGNAL
        # ==============================================================

        try:
            model_audio = np.asarray(
                processed.model_signal,
                dtype=np.float32,
            )

        except Exception as exc:
            logger.warning(
                ("Event %s: unable to construct model waveform: %s"),
                event.event_id,
                exc,
            )

            return None

        # ==============================================================
        # SHAPE
        # ==============================================================

        if model_audio.ndim != 1:
            logger.warning(
                ("Event %s: model waveform is not mono/1-D"),
                event.event_id,
            )

            return None

        # ==============================================================
        # LENGTH
        # ==============================================================

        if model_audio.size == 0:
            return None

        # ==============================================================
        # FINITE VALUES
        # ==============================================================

        if not np.all(np.isfinite(model_audio)):
            logger.warning(
                ("Event %s: model waveform contains non-finite values"),
                event.event_id,
            )

            return None

        # ==============================================================
        # CONTIGUOUS PRIVATE COPY
        # ==============================================================

        return np.ascontiguousarray(
            model_audio,
            dtype=np.float32,
        ).copy()

    # ==================================================================
    # BEST NODE + DSP EXTRACTION
    # ==================================================================

    def _extract_best_channel_data(
        self,
        event: AcousticEvent,
    ) -> tuple[
        int | None,
        AcousticFeatures | None,
        np.ndarray | None,
    ]:
        """
        Select the best available microphone for DSP/classification.

        Model waveform and handcrafted features are generated
        independently after best-channel selection.

        Therefore:

            feature extraction failure
                does not destroy model_audio

            model-audio preparation failure
                does not destroy extracted features
        """

        best_node_id: int | None = None

        best_score = float("-inf")

        best_audio: PreprocessedAudio | None = None

        best_noise_rms: float | None = None

        # ==============================================================
        # EVALUATE ALL MICROPHONES
        # ==============================================================

        for node_id in sorted(self.config.expected_nodes):
            # ----------------------------------------------------------
            # RAW WINDOW
            # ----------------------------------------------------------

            try:
                pcm_samples = self.streams.get_window(
                    node_id,
                    event.start_sample,
                    event.sample_count,
                )

            except Exception as exc:
                logger.warning(
                    ("Event %s: unable to read Node %s audio: %s"),
                    event.event_id,
                    node_id,
                    exc,
                )

                continue

            pcm_samples = np.asarray(
                pcm_samples,
                dtype=np.int16,
            )

            if (
                pcm_samples.ndim != 1
                or pcm_samples.size < self.config.dsp.minimum_event_samples
            ):
                continue

            # ----------------------------------------------------------
            # PREPROCESS
            # ----------------------------------------------------------

            try:
                processed = preprocess_event_audio(
                    pcm_samples,
                    self.preprocessing_config,
                )

            except Exception as exc:
                logger.warning(
                    ("Event %s: preprocessing failed for Node %s: %s"),
                    event.event_id,
                    node_id,
                    exc,
                )

                continue

            # ----------------------------------------------------------
            # NOISE ESTIMATION
            # ----------------------------------------------------------

            noise_rms = self._estimate_noise_rms(processed.amplitude_signal)

            # ----------------------------------------------------------
            # CHANNEL QUALITY
            # ----------------------------------------------------------

            score = self._channel_quality_score(
                processed.amplitude_signal,
                noise_rms,
            )

            logger.debug(
                ("Event %s Node %s quality score = %.2f dB"),
                event.event_id,
                node_id,
                score,
            )

            # ----------------------------------------------------------
            # BEST CHANNEL
            # ----------------------------------------------------------

            if score > best_score:
                best_score = score

                best_node_id = int(node_id)

                best_audio = processed

                best_noise_rms = noise_rms

        # ==============================================================
        # NO VALID MICROPHONE
        # ==============================================================

        if best_node_id is None or best_audio is None:
            return (
                None,
                None,
                None,
            )

        # ==============================================================
        # MODEL AUDIO
        # ==============================================================

        model_audio = self._prepare_model_audio(
            best_audio,
            event,
        )

        # ==============================================================
        # FEATURE EXTRACTION
        # ==============================================================

        features: AcousticFeatures | None = None

        try:
            features = extract_acoustic_features(
                best_audio,
                noise_rms=best_noise_rms,
                config=self.feature_config,
            )

        except Exception as exc:
            logger.warning(
                ("Event %s: feature extraction failed on Node %s: %s"),
                event.event_id,
                best_node_id,
                exc,
            )

            # ----------------------------------------------------------
            # Do not return early.
            #
            # Waveform-based classifiers may still classify this event.
            # ----------------------------------------------------------

        return (
            best_node_id,
            features,
            model_audio,
        )

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    def _classify_event(
        self,
        *,
        event: AcousticEvent,
        best_node_id: int | None,
        features: AcousticFeatures | None,
        model_audio: np.ndarray | None,
    ) -> ClassificationResult | None:
        """
        Execute the configured classification backend.

        Input ownership
        ---------------
        EventPipeline creates the backend-independent ClassificationInput.

        The classifier backend owns requirement validation.

        EventPipeline intentionally does NOT call:

            backend.validate_input(...)

        before:

            backend.classify(...)

        because composite backends such as EnsembleClassifierBackend may
        allow one member to fail while another remains usable.
        """

        # ==============================================================
        # CLASSIFICATION DISABLED
        # ==============================================================

        if not (self.classification_enabled):
            return None

        backend = self.classifier_backend

        if backend is None:
            return None

        # ==============================================================
        # BACKEND-INDEPENDENT INPUT
        # ==============================================================

        try:
            classification_input = ClassificationInput(
                features=features,
                sample_rate=self.config.audio.sample_rate,
                model_audio=model_audio,
                source_node_id=best_node_id,
                detector_event_id=event.event_id,
                session_id=event.session_id,
            )

        except Exception as exc:
            logger.warning(
                ("Event %s classification input creation failed: %s"),
                event.event_id,
                exc,
            )

            return None

        # ==============================================================
        # BACKEND EXECUTION
        # ==============================================================
        #
        # Each backend is responsible for validating its own required
        # data.
        #
        # Examples:
        #
        # heuristic:
        #     validates features
        #
        # BirdNET:
        #     validates waveform
        #
        # ensemble:
        #     members validate independently
        # ==============================================================

        try:
            result = backend.classify(classification_input)

        except Exception as exc:
            logger.warning(
                ("Event %s classification failed using backend '%s': %s"),
                event.event_id,
                backend.name,
                exc,
            )

            return None

        # ==============================================================
        # RESULT CONTRACT
        # ==============================================================

        if not isinstance(
            result,
            ClassificationResult,
        ):
            logger.warning(
                (
                    "Event %s classifier backend '%s' "
                    "returned %s instead of "
                    "ClassificationResult"
                ),
                event.event_id,
                backend.name,
                type(result).__name__,
            )

            return None

        return result

    # ==================================================================
    # EVENT PROCESSING
    # ==================================================================

    def _processing_status(
        self, event_id: int, stage: str, status: str, detail: str | None = None
    ) -> None:
        try:
            self.database.set_event_processing_status(event_id, stage, status, detail)
        except Exception:
            logger.exception("Unable to store processing status for event %s", event_id)

    def _persist_event(
        self,
        event: AcousticEvent,
    ) -> None:
        """
        Execute the complete processing chain for one completed event.

        Persistence policy
        ------------------
        The core acoustic event remains persistable even when optional
        downstream stages fail.

        Optional stages include:

            localization
            DSP feature extraction
            classification
            individual WAV writes
        """

        # ==============================================================
        # ACTIVE SESSION REQUIRED
        # ==============================================================

        if self.active_session_id is None:
            logger.debug(
                ("Ignoring event %s because no pipeline session is active"),
                event.event_id,
            )

            return

        # ==============================================================
        # SESSION VALIDATION
        # ==============================================================

        if event.session_id != self.active_session_id:
            logger.warning(
                ("Ignoring event from stale session 0x%08X (active=0x%08X)"),
                event.session_id,
                self.active_session_id,
            )

            return

        # ==============================================================
        # EVENT MIDPOINT
        # ==============================================================

        event_dir: Path | None = None
        if self.config.persistence.save_event_wav:
            event_dir = (
                self.config.persistence.events_dir
                / f"session_{event.session_id:08X}"
                / f"event_{event.event_id:06d}"
            )
        database_event_id = self.database.add_event(
            event,
            environment=None,
            localization=None,
            event_directory=None if event_dir is None else str(event_dir),
        )
        self._processing_status(database_event_id, "processing", "pending")
        try:
            self._analyze_event(event, database_event_id, event_dir)
        except Exception as exc:
            self._processing_status(database_event_id, "processing", "failed", str(exc))
            raise

    def _analyze_event(
        self, event: AcousticEvent, database_event_id: int, event_dir: Path | None
    ) -> None:
        """Enrich a durable event; unexpected errors are recorded by the caller."""
        midpoint = (event.start_sample + event.end_sample) // 2

        # ==============================================================
        # ENVIRONMENT
        # ==============================================================

        environment = self.streams.get_environment_near(midpoint)

        # ==============================================================
        # LOCALIZATION
        # ==============================================================

        localization: LocalizationResult | None = None

        try:
            (
                localization_start,
                localization_length,
            ) = self._localization_start(event)

            if localization_length >= self.config.dsp.minimum_event_samples:
                localization = self.localizer.locate_window(
                    start_sample=localization_start,
                    length=localization_length,
                )

        except Exception as exc:
            logger.warning(
                ("Event %s localization unavailable: %s"),
                event.event_id,
                exc,
            )

        # ==============================================================
        # BEST MICROPHONE + DSP
        # ==============================================================

        best_node_id: int | None = None

        features: AcousticFeatures | None = None

        model_audio: np.ndarray | None = None

        try:
            (
                best_node_id,
                features,
                model_audio,
            ) = self._extract_best_channel_data(event)

        except Exception as exc:
            logger.warning(
                ("Event %s DSP unavailable: %s"),
                event.event_id,
                exc,
            )

        # ==============================================================
        # CLASSIFICATION
        # ==============================================================

        classification = self._classify_event(
            event=event,
            best_node_id=best_node_id,
            features=features,
            model_audio=model_audio,
        )

        try:
            self.database.update_event_analysis(
                database_event_id,
                environment=environment,
                localization=localization,
                best_node_id=best_node_id,
            )
            self._processing_status(database_event_id, "analysis_storage", "complete")
        except Exception as exc:
            self._processing_status(
                database_event_id, "analysis_storage", "failed", str(exc)
            )
            logger.exception("Event %s analysis storage failed", event.event_id)
        self._processing_status(
            database_event_id,
            "localization",
            "complete"
            if localization is not None and localization.position.success
            else "unavailable",
        )
        self._processing_status(
            database_event_id,
            "classification",
            "complete" if classification is not None else "unavailable",
        )
        if event_dir is not None:
            self._processing_status(database_event_id, "audio", "pending")
            try:
                event_dir.mkdir(parents=True, exist_ok=True)
                failures = []
                for node_id in sorted(self.config.expected_nodes):
                    try:
                        samples = self.streams.get_window(
                            node_id, event.start_sample, event.sample_count
                        )
                        self._write_wav(event_dir / f"node_{node_id}.wav", samples)
                    except Exception as exc:
                        failures.append(f"node {node_id}: {exc}")
                self._processing_status(
                    database_event_id,
                    "audio",
                    "partial" if failures else "complete",
                    "; ".join(failures) or None,
                )
            except Exception as exc:
                logger.exception("Event %s audio storage failed", event.event_id)
                self._processing_status(database_event_id, "audio", "failed", str(exc))

        # ==============================================================
        # STORE DSP FEATURES
        # ==============================================================

        if features is not None and best_node_id is not None:
            try:
                self.database.add_event_features(
                    event_id=database_event_id,
                    source_node_id=best_node_id,
                    features=features,
                )
                self._processing_status(
                    database_event_id, "feature_storage", "complete"
                )

            except Exception as exc:
                self._processing_status(
                    database_event_id, "feature_storage", "failed", str(exc)
                )
                logger.warning(
                    ("Event %s stored, but DSP feature persistence failed: %s"),
                    event.event_id,
                    exc,
                )

        # ==============================================================
        # STORE CLASSIFICATION
        # ==============================================================

        if classification is not None:
            try:
                self.database.add_classification(
                    event_id=database_event_id,
                    result=classification,
                )
                self._processing_status(
                    database_event_id, "classification_storage", "complete"
                )

            except Exception as exc:
                self._processing_status(
                    database_event_id, "classification_storage", "failed", str(exc)
                )
                logger.warning(
                    ("Event %s stored, but classification persistence failed: %s"),
                    event.event_id,
                    exc,
                )

        # ==============================================================
        # UPDATE RUNTIME STATE
        # ==============================================================

        self._processing_status(database_event_id, "processing", "complete")
        self.completed_events += 1

        self.last_event_db_id = database_event_id

        self.last_best_node_id = best_node_id

        self.last_features = features

        self.last_classification = classification

        # ==============================================================
        # EVENT LOG
        # ==============================================================

        if localization is not None and localization.position.success:
            logger.info(
                (
                    "EVENT #%d DB#%d "
                    "nodes=%s "
                    "samples=%d..%d "
                    "best_node=%s "
                    "position=(%.3f, %.3f)m "
                    "c=%.2fm/s"
                ),
                event.event_id,
                database_event_id,
                event.trigger_nodes,
                event.start_sample,
                event.end_sample,
                (str(best_node_id) if best_node_id is not None else "N/A"),
                localization.position.x,
                localization.position.y,
                localization.speed_of_sound_mps,
            )

        else:
            logger.info(
                (
                    "EVENT #%d DB#%d "
                    "nodes=%s "
                    "samples=%d..%d "
                    "best_node=%s "
                    "localization=unavailable"
                ),
                event.event_id,
                database_event_id,
                event.trigger_nodes,
                event.start_sample,
                event.end_sample,
                (str(best_node_id) if best_node_id is not None else "N/A"),
            )

        # ==============================================================
        # DSP LOG
        # ==============================================================

        if features is not None:
            logger.info(
                (
                    "DSP event=%d "
                    "node=%s "
                    "duration=%.3fs "
                    "SNR=%s "
                    "dominant=%.1fHz "
                    "centroid=%.1fHz "
                    "bandwidth=%.1fHz "
                    "rolloff=%.1fHz"
                ),
                event.event_id,
                best_node_id,
                features.duration_s,
                (f"{features.snr_db:.2f}dB" if features.snr_db is not None else "N/A"),
                features.dominant_frequency_hz,
                features.spectral_centroid_hz,
                features.spectral_bandwidth_hz,
                features.spectral_rolloff_hz,
            )

        # ==============================================================
        # CLASSIFICATION LOG
        # ==============================================================

        if classification is not None:
            second_label = (
                classification.second_label.value
                if classification.second_label is not None
                else "N/A"
            )

            second_confidence = classification.second_confidence

            logger.info(
                (
                    "CLASSIFICATION event=%d "
                    "backend=%s "
                    "version=%s "
                    "label=%s "
                    "confidence=%.3f "
                    "second=%s "
                    "second_confidence=%s "
                    "margin=%.3f"
                ),
                event.event_id,
                # ------------------------------------------------------
                # Log the backend recorded by the actual result rather
                # than merely the configured wrapper/backend object.
                # ------------------------------------------------------
                classification.classifier_name,
                classification.classifier_version,
                classification.label.value,
                classification.confidence,
                second_label,
                (
                    f"{second_confidence:.3f}"
                    if second_confidence is not None
                    else "N/A"
                ),
                classification.margin,
            )

    # ==================================================================
    # ENVIRONMENT STORAGE
    # ==================================================================

    def add_environment(
        self,
        *,
        node_id: int,
        session_id: int,
        sample_index: int,
        environment: EnvironmentPayload,
    ) -> None:
        """
        Persist one environmental telemetry packet.

        Session ownership is normally enforced by the receiver/server
        before this method is reached.
        """

        self.database.add_environment(
            session_id=session_id,
            node_id=node_id,
            sample_index=sample_index,
            environment=environment,
        )

    # ==================================================================
    # WAV WRITER
    # ==================================================================

    def _write_wav(
        self,
        path: Path,
        samples: np.ndarray,
    ) -> None:
        """
        Write one synchronized mono PCM16 event recording.

        The source sample array is not modified.
        """

        if not isinstance(
            path,
            Path,
        ):
            raise TypeError("path must be pathlib.Path.")

        pcm = np.asarray(
            samples,
            dtype="<i2",
        )

        if pcm.ndim != 1:
            raise ValueError(("event WAV samples must be mono/1-D"))

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with wave.open(
            str(path),
            "wb",
        ) as wav:
            wav.setnchannels(self.config.audio.channels)

            wav.setsampwidth(self.config.audio.sample_width_bytes)

            wav.setframerate(self.config.audio.sample_rate)

            wav.writeframes(pcm.tobytes())
