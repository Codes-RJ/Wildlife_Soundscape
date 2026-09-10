from __future__ import annotations

from collections import deque

from dataclasses import dataclass

import math

from typing import Deque

import numpy as np

from wildlife_soundscape.core.config import (
    AudioConfig,
    EventDetectionConfig,
)

from wildlife_soundscape.core.models import (
    AudioBlock,
)


# ======================================================================
# CONSTANTS
# ======================================================================


PCM16_FULL_SCALE = 32768.0


MINIMUM_DBFS = -120.0


# ======================================================================
# BLOCK-LEVEL DETECTION RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class BlockDetection:
    """
    Acoustic activity decision for one PCM block from one node.
    """

    node_id: int

    sample_index: int

    end_sample: int

    rms_dbfs: float

    noise_floor_dbfs: float

    threshold_dbfs: float

    spectral_flux: float

    active: bool


# ======================================================================
# COMPLETE ACOUSTIC EVENT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class AcousticEvent:
    """
    Multi-node acoustic event represented on the common sampleIndex
    timeline.
    """

    event_id: int

    session_id: int

    start_sample: int

    end_sample: int

    trigger_nodes: tuple[
        int,
        ...,
    ]

    peak_rms_dbfs: float

    # ==================================================================
    # SAMPLE COUNT
    # ==================================================================

    @property
    def sample_count(
        self,
    ) -> int:
        """
        Total extracted event-window length including configured padding.
        """

        return max(
            0,
            self.end_sample - self.start_sample,
        )


# ======================================================================
# INTERNAL ACTIVE EVENT STATE
# ======================================================================


@dataclass(
    slots=True,
)
class _ActiveEventState:
    """
    Mutable internal state while a multi-node event is active.
    """

    session_id: int

    start_sample: int

    active_start_sample: int

    last_active_end: int

    trigger_nodes: set[int]

    peak_rms_dbfs: float


# ======================================================================
# NODE EVENT DETECTOR
# ======================================================================


class NodeEventDetector:
    """
    Adaptive block-level activity detector for one acoustic node.

    Detection combines:

        RMS energy above adaptive background
        +
        normalized spectral flux

    A sufficiently strong energy excursion may bypass the spectral-flux
    condition so short impulses are not missed.


    Noise-floor model
    -----------------
    A rolling low quantile of background RMS values is used.

    Candidate attack blocks are deliberately excluded from background
    learning so an event onset cannot raise its own detection threshold.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        node_id: int,
        config: EventDetectionConfig,
    ) -> None:

        if not isinstance(
            node_id,
            int,
        ) or isinstance(
            node_id,
            bool,
        ):
            raise TypeError("node_id must be an integer")

        if node_id <= 0:
            raise ValueError(("node_id must be greater than 0"))

        if not isinstance(
            config,
            EventDetectionConfig,
        ):
            raise TypeError(("config must be an EventDetectionConfig instance"))

        self.node_id = node_id

        self.config = config

        # ==============================================================
        # ADAPTIVE NOISE HISTORY
        # ==============================================================

        self._noise: Deque[float] = deque(maxlen=config.noise_history_blocks)

        # ==============================================================
        # SPECTRAL HISTORY
        # ==============================================================

        self._previous_spectrum: np.ndarray | None = None

        self._previous_end_sample: int | None = None

        self._session_id: int | None = None

        # ==============================================================
        # HYSTERESIS STATE
        # ==============================================================

        self._active = False

        self._attack_count = 0

        self._release_count = 0

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(
        self,
        *,
        session_id: int | None = None,
    ) -> None:
        """
        Reset adaptive and temporal detector state.

        This is required at acquisition-session boundaries because:

            sampleIndex restarts at zero
            spectral history is session-specific
            adaptive noise history may belong to different conditions
        """

        self._noise.clear()

        self._previous_spectrum = None

        self._previous_end_sample = None

        self._session_id = session_id

        self._active = False

        self._attack_count = 0

        self._release_count = 0

    # ==================================================================
    # RMS
    # ==================================================================

    @staticmethod
    def _rms_dbfs(
        samples: np.ndarray,
    ) -> float:
        """
        Calculate PCM16 RMS level in dBFS.
        """

        x = np.asarray(
            samples,
            dtype=np.float64,
        )

        if x.size == 0:
            return MINIMUM_DBFS

        rms = float(np.sqrt(np.mean(x * x)))

        if not math.isfinite(rms) or rms <= 1e-12:
            return MINIMUM_DBFS

        dbfs = 20.0 * math.log10(rms / PCM16_FULL_SCALE)

        return float(
            max(
                MINIMUM_DBFS,
                dbfs,
            )
        )

    # ==================================================================
    # SPECTRAL FLUX
    # ==================================================================

    def _spectral_flux(
        self,
        samples: np.ndarray,
    ) -> float:
        """
        Calculate normalized positive spectral flux.

        The previous spectrum is maintained only across contiguous blocks.
        """

        x = np.asarray(
            samples,
            dtype=np.float64,
        )

        if x.size < 16:
            self._previous_spectrum = None

            return 0.0

        # ==============================================================
        # REMOVE DC
        # ==============================================================

        x = x - np.mean(x)

        # ==============================================================
        # WINDOW + FFT
        # ==============================================================

        window = np.hanning(x.size)

        spectrum = np.abs(np.fft.rfft(x * window))

        # ==============================================================
        # NORMALIZE MAGNITUDE VECTOR
        # ==============================================================

        norm = float(np.linalg.norm(spectrum))

        if norm > 1e-12:
            spectrum = spectrum / norm

        # ==============================================================
        # FIRST / INCOMPATIBLE SPECTRUM
        # ==============================================================

        if (
            self._previous_spectrum is None
            or self._previous_spectrum.size != spectrum.size
        ):
            flux = 0.0

        else:
            difference = spectrum - self._previous_spectrum

            positive_difference = np.maximum(
                difference,
                0.0,
            )

            flux = float(np.sqrt(np.mean(np.square(positive_difference))))

        self._previous_spectrum = spectrum

        if not math.isfinite(flux):
            return 0.0

        return max(
            0.0,
            float(flux),
        )

    # ==================================================================
    # PROCESS BLOCK
    # ==================================================================

    def process(
        self,
        block: AudioBlock,
    ) -> BlockDetection:
        """
        Process one PCM block from this detector's node.
        """

        # ==============================================================
        # TYPE / NODE
        # ==============================================================

        if not isinstance(
            block,
            AudioBlock,
        ):
            raise TypeError(("block must be an AudioBlock instance"))

        if block.node_id != self.node_id:
            raise ValueError(
                (
                    "AudioBlock node_id does not "
                    "match NodeEventDetector node_id "
                    f"({block.node_id} != "
                    f"{self.node_id})"
                )
            )

        # ==============================================================
        # SESSION TRANSITION
        # ==============================================================

        if self._session_id != block.session_id:
            self.reset(session_id=block.session_id)

        # ==============================================================
        # DISCONTINUOUS SAMPLE TIMELINE
        # ==============================================================
        #
        # Spectral flux across a packet/sample gap is not physically
        # meaningful.
        #
        # Retain noise/hysteresis state, but restart spectral history.
        # ==============================================================

        if (
            self._previous_end_sample is not None
            and block.sample_index != self._previous_end_sample
        ):
            self._previous_spectrum = None

        # ==============================================================
        # SIGNAL FEATURES
        # ==============================================================

        rms_dbfs = self._rms_dbfs(block.samples)

        spectral_flux = self._spectral_flux(block.samples)

        self._previous_end_sample = block.end_sample

        # ==============================================================
        # ADAPTIVE NOISE FLOOR
        # ==============================================================

        if self._noise:
            noise_floor_dbfs = float(
                np.quantile(
                    np.asarray(
                        self._noise,
                        dtype=np.float64,
                    ),
                    self.config.noise_quantile,
                )
            )

        else:
            noise_floor_dbfs = float(self.config.initial_noise_dbfs)

        # ==============================================================
        # THRESHOLDS
        # ==============================================================

        trigger_threshold = noise_floor_dbfs + self.config.trigger_margin_db

        release_threshold = noise_floor_dbfs + self.config.release_margin_db

        strong_threshold = noise_floor_dbfs + self.config.strong_energy_margin_db

        # ==============================================================
        # EVENT CANDIDATE
        # ==============================================================

        candidate = bool(
            (
                rms_dbfs >= trigger_threshold
                and spectral_flux >= self.config.min_spectral_flux
            )
            or (rms_dbfs >= strong_threshold)
        )

        # ==============================================================
        # ACTIVE → RELEASE
        # ==============================================================

        if self._active:
            if rms_dbfs < release_threshold:
                self._release_count += 1

            else:
                self._release_count = 0

            if self._release_count >= self.config.release_blocks:
                self._active = False

                self._release_count = 0

                self._attack_count = 0

        # ==============================================================
        # INACTIVE → ATTACK
        # ==============================================================

        else:
            if candidate:
                self._attack_count += 1

            else:
                self._attack_count = 0

            if self._attack_count >= self.config.attack_blocks:
                self._active = True

                self._attack_count = 0

        # ==============================================================
        # BACKGROUND MODEL UPDATE
        # ==============================================================
        #
        # Do NOT learn:
        #
        #   active event blocks
        #   candidate attack blocks
        #
        # Otherwise event onset can artificially raise the adaptive
        # background estimate before attack_blocks is satisfied.
        # ==============================================================

        if not self._active and not candidate:
            self._noise.append(rms_dbfs)

        # ==============================================================
        # RESULT
        # ==============================================================

        return BlockDetection(
            node_id=block.node_id,
            sample_index=block.sample_index,
            end_sample=block.end_sample,
            rms_dbfs=rms_dbfs,
            noise_floor_dbfs=noise_floor_dbfs,
            threshold_dbfs=trigger_threshold,
            spectral_flux=spectral_flux,
            active=self._active,
        )


# ======================================================================
# MULTI-NODE EVENT DETECTOR
# ======================================================================


class MultiNodeEventDetector:
    """
    Associate node-level acoustic decisions on the common sampleIndex
    timeline.

    Coarse synchronization
    ----------------------
    Node block starts are NOT required to be numerically identical.

    Blocks are associated when their complete start-index spread remains
    within AudioConfig.sync_tolerance_samples.

    Example
    -------

        Node 1 = 10240
        Node 2 = 10245
        Node 3 = 10238

    With tolerance = 10:

        max - min = 7

    so the three detections belong to one synchronized decision group.


    Fine acoustic propagation delay remains untouched and is handled
    later by GCC-PHAT/TDOA.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        audio: AudioConfig,
        config: EventDetectionConfig,
        node_ids: tuple[
            int,
            ...,
        ] = (
            1,
            2,
            3,
        ),
    ) -> None:

        if not isinstance(
            audio,
            AudioConfig,
        ):
            raise TypeError(("audio must be an AudioConfig instance"))

        if not isinstance(
            config,
            EventDetectionConfig,
        ):
            raise TypeError(("config must be an EventDetectionConfig instance"))

        normalized_nodes = tuple(int(node_id) for node_id in node_ids)

        if not normalized_nodes:
            raise ValueError(("At least one detector node is required"))

        if any(node_id <= 0 for node_id in normalized_nodes):
            raise ValueError(("Detector node IDs must be greater than 0"))

        if len(set(normalized_nodes)) != len(normalized_nodes):
            raise ValueError(("Detector node IDs cannot contain duplicates"))

        if config.min_nodes > len(normalized_nodes):
            raise ValueError(
                (
                    "Event detector min_nodes "
                    "cannot exceed configured "
                    "detector node count"
                )
            )

        if audio.sync_tolerance_samples >= audio.frames_per_block:
            raise ValueError(
                ("sync_tolerance_samples must remain smaller than one audio block")
            )

        self.audio = audio

        self.config = config

        self.node_ids = normalized_nodes

        # ==============================================================
        # PER-NODE DETECTORS
        # ==============================================================

        self.detectors = {
            node_id: NodeEventDetector(
                node_id,
                config,
            )
            for node_id in self.node_ids
        }

        # ==============================================================
        # PENDING SYNCHRONIZED DECISIONS
        # ==============================================================
        #
        # Key:
        #
        #     (session_id, anchor_sample_index)
        #
        # anchor_sample_index is simply the first block start assigned to
        # that pending group.
        # ==============================================================

        self._pending: dict[
            tuple[
                int,
                int,
            ],
            dict[
                int,
                BlockDetection,
            ],
        ] = {}

        # ==============================================================
        # ACTIVE EVENT
        # ==============================================================

        self._active_event: _ActiveEventState | None = None

        # ==============================================================
        # SESSION TRACKING
        # ==============================================================

        self._session_id: int | None = None

        # ==============================================================
        # PER-NODE PROCESSED TIMELINE
        # ==============================================================

        self._last_processed_end: dict[
            int,
            int,
        ] = {}

        # ==============================================================
        # EVENT IDENTIFIER
        # ==============================================================

        self._next_event_id = 1

    # ==================================================================
    # PADDING / DURATION PROPERTIES
    # ==================================================================

    @property
    def pre_pad_samples(
        self,
    ) -> int:
        """
        Number of samples retained before detected activity.
        """

        return max(
            0,
            int(round(self.config.pre_pad_s * self.audio.sample_rate)),
        )

    @property
    def post_pad_samples(
        self,
    ) -> int:
        """
        Number of samples retained after detected activity.
        """

        return max(
            0,
            int(round(self.config.post_pad_s * self.audio.sample_rate)),
        )

    @property
    def min_event_samples(
        self,
    ) -> int:
        """
        Minimum confirmed active duration in samples.
        """

        return max(
            1,
            int(
                math.ceil((self.config.min_event_ms / 1000.0) * self.audio.sample_rate)
            ),
        )

    @property
    def max_event_samples(
        self,
    ) -> int:
        """
        Maximum persisted event-window duration in samples.
        """

        return max(
            1,
            int(round(self.config.max_event_s * self.audio.sample_rate)),
        )

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(
        self,
    ) -> None:
        """
        Reset detector runtime state.

        _next_event_id deliberately remains monotonic for the lifetime of
        this detector instance.
        """

        self._pending.clear()

        self._active_event = None

        self._session_id = None

        self._last_processed_end.clear()

        self.detectors = {
            node_id: NodeEventDetector(
                node_id,
                self.config,
            )
            for node_id in self.node_ids
        }

    # ==================================================================
    # SESSION TRANSITION
    # ==================================================================

    def _begin_session(
        self,
        session_id: int,
    ) -> None:
        """
        Reset session-specific detector state while retaining the global
        monotonic event counter.
        """

        self._pending.clear()

        self._active_event = None

        self._last_processed_end.clear()

        self._session_id = session_id

        for detector in self.detectors.values():
            detector.reset(session_id=session_id)

    # ==================================================================
    # FIND SYNCHRONIZED PENDING GROUP
    # ==================================================================

    def _find_pending_key(
        self,
        *,
        session_id: int,
        detection: BlockDetection,
    ) -> (
        tuple[
            int,
            int,
        ]
        | None
    ):
        """
        Find the best pending multi-node group for one block detection.

        A candidate group is valid only when the complete sample-start
        spread after insertion remains inside synchronization tolerance.
        """

        tolerance = self.audio.sync_tolerance_samples

        candidates: list[
            tuple[
                int,
                int,
                tuple[
                    int,
                    int,
                ],
            ]
        ] = []

        for (
            key,
            bucket,
        ) in self._pending.items():
            bucket_session_id, anchor = key

            # ----------------------------------------------------------
            # SESSION
            # ----------------------------------------------------------

            if bucket_session_id != session_id:
                continue

            # ----------------------------------------------------------
            # ONE DECISION PER NODE PER GROUP
            # ----------------------------------------------------------

            if detection.node_id in bucket:
                continue

            starts = [item.sample_index for item in bucket.values()]

            starts.append(detection.sample_index)

            spread = max(starts) - min(starts)

            if spread > tolerance:
                continue

            anchor_distance = abs(detection.sample_index - anchor)

            candidates.append(
                (
                    spread,
                    anchor_distance,
                    key,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2][1],
            )
        )

        return candidates[0][2]

    # ==================================================================
    # PROCESS AUDIO BLOCK
    # ==================================================================

    def process(
        self,
        block: AudioBlock,
    ) -> list[AcousticEvent]:
        """
        Process one node AUDIO block.

        Returns zero or one newly completed AcousticEvent.
        """

        # ==============================================================
        # MASTER SWITCH / NODE FILTER
        # ==============================================================

        if not (self.config.enabled):
            return []

        if block.node_id not in self.detectors:
            return []

        # AUDIO acquisition packets should carry a real non-zero session.
        if block.session_id == 0:
            return []

        # ==============================================================
        # SESSION
        # ==============================================================

        if self._session_id != block.session_id:
            self._begin_session(block.session_id)

        # ==============================================================
        # DUPLICATE / OLD / OVERLAPPING BLOCK
        # ==============================================================
        #
        # NodeState normally rejects these before buffering, but
        # EventPipeline receives the original block independently.
        #
        # Therefore the detector protects its own timeline as well.
        # ==============================================================

        previous_end = self._last_processed_end.get(block.node_id)

        if previous_end is not None and block.sample_index < previous_end:
            return []

        self._last_processed_end[block.node_id] = block.end_sample

        # ==============================================================
        # NODE-LEVEL DECISION
        # ==============================================================

        detection = self.detectors[block.node_id].process(block)

        # ==============================================================
        # ASSOCIATE WITH SYNCHRONIZED GROUP
        # ==============================================================

        key = self._find_pending_key(
            session_id=block.session_id,
            detection=detection,
        )

        if key is None:
            key = (
                block.session_id,
                block.sample_index,
            )

            self._pending[key] = {}

        bucket = self._pending[key]

        bucket[block.node_id] = detection

        # ==============================================================
        # WAIT FOR COMPLETE LAB-ARRAY DECISION
        # ==============================================================
        #
        # We intentionally evaluate once all configured node decisions
        # are available for the synchronized block group.
        #
        # This makes event decisions deterministic and independent of TCP
        # arrival ordering.
        #
        # min_nodes controls how many of those node decisions must report
        # ACTIVE.
        # ==============================================================

        if not all(node_id in bucket for node_id in self.node_ids):
            self._prune(
                latest_sample=block.sample_index,
                session_id=block.session_id,
            )

            return []

        decisions = self._pending.pop(key)

        return self._evaluate(
            block.session_id,
            decisions,
        )

    # ==================================================================
    # EVALUATE MULTI-NODE DECISIONS
    # ==================================================================

    def _evaluate(
        self,
        session_id: int,
        decisions: dict[
            int,
            BlockDetection,
        ],
    ) -> list[AcousticEvent]:
        """
        Update/finalize the current multi-node acoustic event.
        """

        if not decisions:
            return []

        # ==============================================================
        # ACTIVE NODE SET
        # ==============================================================

        active_nodes = tuple(
            sorted(
                node_id
                for (
                    node_id,
                    detection,
                ) in decisions.items()
                if detection.active
            )
        )

        # ==============================================================
        # COMMON BLOCK REGION
        # ==============================================================

        block_start = min(detection.sample_index for detection in decisions.values())

        block_end = max(detection.end_sample for detection in decisions.values())

        peak_rms = max(detection.rms_dbfs for detection in decisions.values())

        # ==============================================================
        # NO ACTIVE EVENT
        # ==============================================================

        if self._active_event is None:
            if len(active_nodes) >= self.config.min_nodes:
                self._active_event = _ActiveEventState(
                    session_id=session_id,
                    start_sample=max(
                        0,
                        block_start - self.pre_pad_samples,
                    ),
                    active_start_sample=block_start,
                    last_active_end=block_end,
                    trigger_nodes=set(active_nodes),
                    peak_rms_dbfs=peak_rms,
                )

            return []

        # ==============================================================
        # SESSION SAFETY
        # ==============================================================

        event = self._active_event

        if event.session_id != session_id:
            self._active_event = None

            # Do not discard the first block of the replacement session.
            if len(active_nodes) >= self.config.min_nodes:
                self._active_event = _ActiveEventState(
                    session_id=session_id,
                    start_sample=max(
                        0,
                        block_start - self.pre_pad_samples,
                    ),
                    active_start_sample=block_start,
                    last_active_end=block_end,
                    trigger_nodes=set(active_nodes),
                    peak_rms_dbfs=peak_rms,
                )

            return []

        # ==============================================================
        # UPDATE PEAK / PARTICIPATING NODES
        # ==============================================================

        event.peak_rms_dbfs = max(
            event.peak_rms_dbfs,
            peak_rms,
        )

        event.trigger_nodes.update(active_nodes)

        # ==============================================================
        # LAST CONFIRMED MULTI-NODE ACTIVITY
        # ==============================================================

        if len(active_nodes) >= self.config.min_nodes:
            event.last_active_end = block_end

        # ==============================================================
        # COMPLETION CONDITIONS
        # ==============================================================

        max_reached = block_end - event.start_sample >= self.max_event_samples

        released = len(active_nodes) < self.config.min_nodes and block_end >= (
            event.last_active_end + self.post_pad_samples
        )

        if not (max_reached or released):
            return []

        # ==============================================================
        # FINAL END SAMPLE
        # ==============================================================

        if max_reached:
            requested_end = block_end

        else:
            requested_end = event.last_active_end + self.post_pad_samples

        end_sample = min(
            requested_end,
            event.start_sample + self.max_event_samples,
        )

        # ==============================================================
        # CLEAR ACTIVE EVENT
        # ==============================================================

        self._active_event = None

        # ==============================================================
        # MINIMUM ACTIVE DURATION
        # ==============================================================

        active_duration = max(
            0,
            event.last_active_end - event.active_start_sample,
        )

        if active_duration < self.min_event_samples:
            return []

        # ==============================================================
        # COMPLETED EVENT
        # ==============================================================

        result = AcousticEvent(
            event_id=self._next_event_id,
            session_id=session_id,
            start_sample=event.start_sample,
            end_sample=end_sample,
            trigger_nodes=tuple(sorted(event.trigger_nodes)),
            peak_rms_dbfs=float(event.peak_rms_dbfs),
        )

        self._next_event_id += 1

        return [result]

    # ==================================================================
    # PRUNE INCOMPLETE ASSOCIATION GROUPS
    # ==================================================================

    def _prune(
        self,
        *,
        latest_sample: int,
        session_id: int,
    ) -> None:
        """
        Prevent unbounded growth if one node temporarily drops packets.

        Eight block periods are retained, which is substantially larger
        than the normal synchronization tolerance while remaining small
        enough to bound memory use.
        """

        horizon = self.audio.frames_per_block * 8

        stale_keys: list[
            tuple[
                int,
                int,
            ]
        ] = []

        for (
            key_session,
            anchor_sample,
        ) in self._pending:
            # Old session pending decisions are never useful.
            if key_session != session_id:
                stale_keys.append(
                    (
                        key_session,
                        anchor_sample,
                    )
                )

                continue

            if latest_sample - anchor_sample > horizon:
                stale_keys.append(
                    (
                        key_session,
                        anchor_sample,
                    )
                )

        for key in stale_keys:
            self._pending.pop(
                key,
                None,
            )
