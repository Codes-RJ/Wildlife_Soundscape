from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from config import AudioConfig
from models import AudioBlock
from node import NodeState


# ======================================================================
# CONSTANTS
# ======================================================================


PCM16_MIN = -32768
PCM16_MAX = 32767

# Silence is written in bounded chunks so a large sample-index gap does
# not require allocating one enormous NumPy array.
WAV_SILENCE_CHUNK_SAMPLES = 65_536


# ======================================================================
# ALIGNED BLOCK RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class AlignedBlocks:
    """
    Approximately sample-aligned audio blocks from requested nodes.

    target_sample_index
        Common coarse sample-index target.

    blocks
        Selected block for each node.

    offsets
        Difference between each selected block's starting sampleIndex
        and target_sample_index.

    Important
    ---------
    This represents coarse digital-stream alignment only.

    Acoustic propagation delay is NOT removed here.

    Residual waveform delay remains available for GCC-PHAT/TDOA.
    """

    target_sample_index: int

    blocks: dict[
        int,
        AudioBlock,
    ]

    offsets: dict[
        int,
        int,
    ]


# ======================================================================
# STREAM MANAGER
# ======================================================================


class StreamManager:
    """
    Maintain sample-indexed audio and environmental streams.

    Synchronization model
    ---------------------
    Shared ESP32/I2S clock
            ↓
       sampleIndex
            ↓
      StreamManager
            ↓
      coarse alignment
            ↓
       PCM waveforms
            ↓
       GCC-PHAT/TDOA

    This class NEVER uses the following for cross-node alignment:

        TCP packet arrival time
        laptop wall-clock time
        ESP32 localMicros

    `localMicros` remains diagnostic information only.

    Missing PCM samples are preserved as timeline gaps rather than
    compressing the waveform.
    """

    def __init__(
        self,
        audio_config: AudioConfig,
    ) -> None:

        self.audio_config = (
            audio_config
        )

        self.nodes: dict[
            int,
            NodeState,
        ] = {}

    # ==================================================================
    # NODE REGISTRATION
    # ==================================================================

    def register_state(
        self,
        state: NodeState,
    ) -> None:
        """
        Register or replace runtime state for one ESP32 node.

        Re-registration normally occurs after a TCP reconnect.
        """

        node_id = int(
            state.node_id
        )

        if (
            node_id
            <= 0
        ):

            raise ValueError(
                "node_id must be greater than 0"
            )

        self.nodes[
            node_id
        ] = state

    # ==================================================================
    # SESSION RESET
    # ==================================================================

    def reset_audio_buffers(
        self,
    ) -> None:
        """
        Remove buffered PCM from all currently registered nodes.

        This must occur before every new acquisition session because
        each ESP32 restarts its sampleIndex timeline at zero.

        Without this reset:

            Session A sampleIndex = 0...
            Session B sampleIndex = 0...

        could coexist inside the same laptop-side buffer and produce
        invalid extraction/localization.
        """

        for state in (
            self.nodes.values()
        ):

            audio_blocks = getattr(
                state,
                "audio_blocks",
                None,
            )

            if audio_blocks is None:

                continue

            audio_blocks.clear()

    # ==================================================================
    # AUDIO INGESTION
    # ==================================================================

    def add_audio(
        self,
        block: AudioBlock,
    ) -> None:
        """
        Add one PCM block to its registered node state.
        """

        state = (
            self.nodes.get(
                block.node_id
            )
        )

        if state is None:

            raise KeyError(
                (
                    f"node {block.node_id} "
                    "is not registered"
                )
            )

        state.add_audio(
            block
        )

    # ==================================================================
    # NODE-ID NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize_node_ids(
        node_ids: Iterable[int],
    ) -> tuple[int, ...]:
        """
        Normalize and validate a requested node collection.
        """

        normalized = tuple(
            int(
                node_id
            )
            for node_id
            in node_ids
        )

        if not normalized:

            raise ValueError(
                "At least one node_id is required"
            )

        if any(
            node_id <= 0
            for node_id
            in normalized
        ):

            raise ValueError(
                "node IDs must be greater than 0"
            )

        if (
            len(
                set(
                    normalized
                )
            )
            != len(
                normalized
            )
        ):

            raise ValueError(
                "node_ids cannot contain duplicates"
            )

        return normalized

    # ==================================================================
    # LATEST COARSE ALIGNMENT
    # ==================================================================

    def latest_aligned_blocks(
        self,
        node_ids: Iterable[int] = (
            1,
            2,
            3,
        ),
        *,
        tolerance_samples: int | None = None,
    ) -> AlignedBlocks | None:
        """
        Find the newest approximately aligned block available from every
        requested node.

        Why this does NOT simply compare each node's newest block
        ----------------------------------------------------------
        TCP packet arrival is asynchronous.

        Example:

            Node 1 newest sampleIndex = 10240
            Node 2 newest sampleIndex = 9216
            Node 3 newest sampleIndex = 10240

        Node 1 and Node 3 may still contain their 9216 blocks.

        Comparing only the newest packet would incorrectly report that
        the streams are not aligned.

        Instead:

            target = least-advanced node's newest sampleIndex

        and each other node's buffer is searched for the block nearest
        that common sample position.

        This remains a diagnostic/coarse-alignment operation.

        Fine acoustic delays are intentionally left to GCC-PHAT.
        """

        requested_nodes = (
            self._normalize_node_ids(
                node_ids
            )
        )

        tolerance = (
            self.audio_config
            .sync_tolerance_samples

            if tolerance_samples
            is None

            else int(
                tolerance_samples
            )
        )

        if (
            tolerance
            < 0
        ):

            raise ValueError(
                (
                    "tolerance_samples "
                    "cannot be negative"
                )
            )

        # ==============================================================
        # SNAPSHOT AVAILABLE BLOCKS
        # ==============================================================

        available: dict[
            int,
            tuple[
                AudioBlock,
                ...,
            ],
        ] = {}

        for node_id in (
            requested_nodes
        ):

            state = (
                self.nodes.get(
                    node_id
                )
            )

            if state is None:

                return None

            blocks = tuple(
                state.audio_blocks
            )

            if not blocks:

                return None

            available[
                node_id
            ] = blocks

        # ==============================================================
        # COMMON TARGET
        # ==============================================================
        #
        # The least-advanced stream determines the newest timeline
        # position that all nodes could currently share.
        # ==============================================================

        target = min(
            int(
                blocks[
                    -1
                ].sample_index
            )
            for blocks
            in available.values()
        )

        # ==============================================================
        # SELECT NEAREST BLOCK PER NODE
        # ==============================================================

        selected: dict[
            int,
            AudioBlock,
        ] = {}

        offsets: dict[
            int,
            int,
        ] = {}

        for (
            node_id,
            blocks,
        ) in available.items():

            nearest = min(
                blocks,
                key=lambda block:
                    abs(
                        int(
                            block.sample_index
                        )
                        - target
                    ),
            )

            offset = (
                int(
                    nearest.sample_index
                )
                - target
            )

            if (
                abs(
                    offset
                )
                > tolerance
            ):

                return None

            selected[
                node_id
            ] = nearest

            offsets[
                node_id
            ] = int(
                offset
            )

        # ==============================================================
        # SESSION CONSISTENCY
        # ==============================================================

        session_ids = {
            int(
                block.session_id
            )
            for block
            in selected.values()
        }

        if (
            len(
                session_ids
            )
            != 1
        ):

            return None

        return AlignedBlocks(
            target_sample_index=
                int(
                    target
                ),

            blocks=
                selected,

            offsets=
                offsets,
        )

    # ==================================================================
    # ENVIRONMENT LOOKUP
    # ==================================================================

    def get_environment_near(
        self,
        sample_index: int,
        *,
        preferred_node: int = 1,
    ):
        """
        Return the environmental sample nearest an audio sample index.

        Node 1 is preferred because the current hardware design places
        the BME280 on Node 1.
        """

        sample_index = int(
            sample_index
        )

        preferred_node = int(
            preferred_node
        )

        if (
            sample_index
            < 0
        ):

            return None

        state = (
            self.nodes.get(
                preferred_node
            )
        )

        if state is None:

            return None

        return state.environment_near(
            sample_index
        )

    # ==================================================================
    # SAMPLE-INDEXED AUDIO WINDOW
    # ==================================================================

    def get_window(
        self,
        node_id: int,
        start_sample: int,
        length: int,
        *,
        fill_value: int = 0,
    ) -> np.ndarray:
        """
        Reconstruct a mono PCM16 window using absolute sample indices.

        Missing samples are filled rather than removed.

        Example
        -------
        Received:

            samples 0..1023
            samples 2048..3071

        Requested:

            0..3071

        Result:

            real PCM
            1024 samples of fill_value
            real PCM

        This behavior is essential because compressing gaps would shift
        later waveform samples and destroy the physical timing needed
        for synchronized analysis and TDOA.
        """

        node_id = int(
            node_id
        )

        start_sample = int(
            start_sample
        )

        length = int(
            length
        )

        fill_value = int(
            fill_value
        )

        # ==============================================================
        # VALIDATION
        # ==============================================================

        if (
            node_id
            <= 0
        ):

            raise ValueError(
                "node_id must be greater than 0"
            )

        if (
            start_sample
            < 0
        ):

            raise ValueError(
                "start_sample cannot be negative"
            )

        if (
            length
            <= 0
        ):

            return np.empty(
                0,
                dtype=np.int16,
            )

        if not (
            PCM16_MIN
            <= fill_value
            <= PCM16_MAX
        ):

            raise ValueError(
                (
                    "fill_value must fit "
                    "signed PCM16"
                )
            )

        state = (
            self.nodes.get(
                node_id
            )
        )

        if state is None:

            raise KeyError(
                (
                    f"node {node_id} "
                    "is not registered"
                )
            )

        end_sample = (
            start_sample
            + length
        )

        # ==============================================================
        # OUTPUT TIMELINE
        # ==============================================================

        output = np.full(
            length,
            fill_value,
            dtype=np.int16,
        )

        # Snapshot prevents unexpected container mutation while the
        # reconstruction loop is running.
        blocks = tuple(
            state.audio_blocks
        )

        # ==============================================================
        # COPY OVERLAPPING PCM
        # ==============================================================

        for block in blocks:

            block_start = int(
                block.sample_index
            )

            block_end = int(
                block.end_sample
            )

            # Completely before requested window.
            if (
                block_end
                <= start_sample
            ):

                continue

            # Completely after requested window.
            if (
                block_start
                >= end_sample
            ):

                continue

            overlap_start = max(
                start_sample,
                block_start,
            )

            overlap_end = min(
                end_sample,
                block_end,
            )

            if (
                overlap_start
                >= overlap_end
            ):

                continue

            src_start = (
                overlap_start
                - block_start
            )

            dst_start = (
                overlap_start
                - start_sample
            )

            count = (
                overlap_end
                - overlap_start
            )

            source = (
                block.samples[
                    src_start:
                    src_start + count
                ]
            )

            # Defensive protection against malformed AudioBlock metadata.
            actual_count = min(
                int(
                    source.size
                ),
                int(
                    count
                ),
            )

            if (
                actual_count
                <= 0
            ):

                continue

            output[
                dst_start:
                dst_start + actual_count
            ] = source[
                :actual_count
            ]

        return output


# ======================================================================
# CONTINUOUS WAV RECORDER
# ======================================================================


class WavRecorder:
    """
    Continuous per-node PCM16 WAV recorder.

    Timing preservation
    -------------------
    sampleIndex gaps are represented as digital silence.

    Missing samples are NEVER removed from the output timeline.

    This makes continuous recordings useful for:

        debugging
        synchronization inspection
        offline DSP
        research data review
    """

    def __init__(
        self,
        root_dir: (
            str
            | os.PathLike[str]
        ),
        audio_config: AudioConfig,
    ) -> None:

        self.root = Path(
            root_dir
        )

        self.audio_config = (
            audio_config
        )

        self._files: dict[
            int,
            wave.Wave_write,
        ] = {}

        self._expected_sample: dict[
            int,
            int,
        ] = {}

        self._session_label: (
            str
            | None
        ) = None

    # ==================================================================
    # RECORDING STATE
    # ==================================================================

    @property
    def is_recording(
        self,
    ) -> bool:
        """
        Whether at least one WAV file is currently open.
        """

        return bool(
            self._files
        )

    # ==================================================================
    # SESSION LABEL VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_session_label(
        session_label: str,
    ) -> str:
        """
        Validate a directory-safe session label.
        """

        label = str(
            session_label
        ).strip()

        if not label:

            raise ValueError(
                "session_label cannot be empty"
            )

        path = Path(
            label
        )

        if (
            path.is_absolute()
            or len(
                path.parts
            )
            != 1
            or label
            in {
                ".",
                "..",
            }
        ):

            raise ValueError(
                (
                    "session_label must be "
                    "a single directory name"
                )
            )

        return label

    # ==================================================================
    # START RECORDING
    # ==================================================================

    def start(
        self,
        session_label: str,
        node_ids: Iterable[int],
    ) -> None:
        """
        Start one WAV file for every requested acoustic node.

        File creation is transactional: if any WAV cannot be opened,
        all files already opened for this attempted session are closed.
        """

        label = (
            self._validate_session_label(
                session_label
            )
        )

        nodes = tuple(
            int(
                node_id
            )
            for node_id
            in node_ids
        )

        if not nodes:

            raise ValueError(
                "At least one node_id is required"
            )

        if any(
            node_id <= 0
            for node_id
            in nodes
        ):

            raise ValueError(
                "node IDs must be greater than 0"
            )

        if (
            len(
                set(
                    nodes
                )
            )
            != len(
                nodes
            )
        ):

            raise ValueError(
                "node_ids cannot contain duplicates"
            )

        # --------------------------------------------------------------
        # Close any previous recording first.
        # --------------------------------------------------------------

        self.stop()

        session_dir = (
            self.root
            / label
        )

        session_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            for node_id in (
                nodes
            ):

                path = (
                    session_dir
                    / f"node_{node_id}.wav"
                )

                wav = wave.open(
                    str(
                        path
                    ),
                    "wb",
                )

                wav.setnchannels(
                    self.audio_config.channels
                )

                wav.setsampwidth(
                    self.audio_config.sample_width_bytes
                )

                wav.setframerate(
                    self.audio_config.sample_rate
                )

                self._files[
                    node_id
                ] = wav

                self._expected_sample[
                    node_id
                ] = 0

        except Exception:

            self.stop()

            raise

        self._session_label = (
            label
        )

    # ==================================================================
    # SILENCE WRITER
    # ==================================================================

    @staticmethod
    def _write_silence(
        wav: wave.Wave_write,
        sample_count: int,
    ) -> None:
        """
        Write PCM16 digital silence using bounded chunks.
        """

        remaining = int(
            sample_count
        )

        if (
            remaining
            <= 0
        ):

            return

        chunk = np.zeros(
            min(
                remaining,
                WAV_SILENCE_CHUNK_SAMPLES,
            ),
            dtype="<i2",
        )

        while (
            remaining
            > 0
        ):

            count = min(
                remaining,
                int(
                    chunk.size
                ),
            )

            wav.writeframesraw(
                chunk[
                    :count
                ].tobytes()
            )

            remaining -= (
                count
            )

    # ==================================================================
    # WRITE BLOCK
    # ==================================================================

    def write(
        self,
        block: AudioBlock,
    ) -> None:
        """
        Write one PCM block while preserving its sample-index timeline.

        Three cases are handled:

        1. block.sample_index > expected
               missing samples → write digital silence

        2. block.sample_index < expected
               repeated/overlapping samples → skip duplicate prefix

        3. block.sample_index == expected
               write normally
        """

        wav = (
            self._files.get(
                block.node_id
            )
        )

        if wav is None:

            return

        expected = (
            self._expected_sample.get(
                block.node_id
            )
        )

        if expected is None:

            return

        block_start = int(
            block.sample_index
        )

        if (
            block_start
            < 0
        ):

            raise ValueError(
                (
                    "AudioBlock sample_index "
                    "cannot be negative"
                )
            )

        samples = np.asarray(
            block.samples,
            dtype="<i2",
        )

        if (
            samples.ndim
            != 1
        ):

            raise ValueError(
                (
                    "WAV recorder expects "
                    "mono 1-D PCM"
                )
            )

        if (
            samples.size
            == 0
        ):

            return

        # ==============================================================
        # GAP
        # ==============================================================

        if (
            block_start
            > expected
        ):

            gap = (
                block_start
                - expected
            )

            self._write_silence(
                wav,
                gap,
            )

            expected = (
                block_start
            )

        # ==============================================================
        # OVERLAP / DUPLICATE DATA
        # ==============================================================

        elif (
            block_start
            < expected
        ):

            overlap = (
                expected
                - block_start
            )

            # Entire packet has already been represented in the file.
            if (
                overlap
                >= samples.size
            ):

                return

            samples = (
                samples[
                    overlap:
                ]
            )

            block_start = (
                expected
            )

        # ==============================================================
        # WRITE REMAINING PCM
        # ==============================================================

        wav.writeframesraw(
            samples.tobytes()
        )

        self._expected_sample[
            block.node_id
        ] = (
            block_start
            + int(
                samples.size
            )
        )

    # ==================================================================
    # STOP
    # ==================================================================

    def stop(
        self,
    ) -> None:
        """
        Finalize and close all open WAV files.
        """

        files = list(
            self._files.values()
        )

        # Clear state first so a close failure cannot leave the recorder
        # logically marked as active.
        self._files.clear()

        self._expected_sample.clear()

        self._session_label = None

        for wav in files:

            try:

                wav.close()

            except Exception:

                # Recorder shutdown should not prevent the rest of the
                # receiver from closing.
                pass