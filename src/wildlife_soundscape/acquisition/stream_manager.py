from __future__ import annotations

import os
import wave

from dataclasses import dataclass

from pathlib import Path

from typing import Iterable

import numpy as np

from wildlife_soundscape.core.config import (
    AudioConfig,
)

from wildlife_soundscape.core.models import (
    AudioBlock,
)

from wildlife_soundscape.acquisition.node import (
    NodeState,
)


# ======================================================================
# CONSTANTS
# ======================================================================


PCM16_MIN = -32768

PCM16_MAX = 32767


UINT8_MAX = 0xFF

UINT64_MAX = 0xFFFFFFFFFFFFFFFF


# Silence is emitted in bounded chunks so a large sampleIndex gap does
# not require one correspondingly huge temporary NumPy allocation.
WAV_SILENCE_CHUNK_SAMPLES = 65_536


# ======================================================================
# INTEGER VALIDATION
# ======================================================================


def _require_integer(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a genuine integer value.

    bool is rejected because Python treats bool as a subclass of int.
    """

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(f"{name} must be an integer")

    if not isinstance(
        value,
        int,
    ):
        raise TypeError(f"{name} must be an integer")

    return int(value)


def _require_node_id(
    value: int,
) -> int:
    """
    Validate one Protocol-v4 node identifier.
    """

    value = _require_integer(
        value,
        name="node_id",
    )

    if not (1 <= value <= UINT8_MAX):
        raise ValueError(("node_id must lie between 1 and 255"))

    return value


def _require_uint64(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate one unsigned 64-bit sample-index value.
    """

    value = _require_integer(
        value,
        name=name,
    )

    if not (0 <= value <= UINT64_MAX):
        raise ValueError((f"{name} must lie in the uint64 range"))

    return value


# ======================================================================
# ALIGNED BLOCK RESULT
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class AlignedBlocks:
    """
    Approximately sample-aligned AUDIO blocks from requested nodes.

    target_sample_index
        Common coarse sampleIndex target.

    blocks
        Selected AUDIO block for each node.

    offsets
        Difference between each selected block's starting sampleIndex
        and target_sample_index.

    Important
    ---------
    This represents only digital/shared-clock coarse alignment.

    Acoustic propagation delay is deliberately NOT removed here.

    Sub-block waveform delay remains available for GCC-PHAT/TDOA.
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

        Shared ESP32 I2S clock
                ↓
           sampleIndex
                ↓
          StreamManager
                ↓
        coarse alignment
                ↓
          PCM waveforms
                ↓
         GCC-PHAT / TDOA


    The following are NEVER used for cross-node alignment:

        TCP packet arrival time
        laptop wall-clock time
        ESP32 localMicros


    Missing PCM remains represented as missing positions on the absolute
    sample timeline.

    get_window() reconstructs those positions using fill values rather
    than compressing the waveform.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        audio_config: AudioConfig,
    ) -> None:

        if not isinstance(
            audio_config,
            AudioConfig,
        ):
            raise TypeError(("audio_config must be an AudioConfig instance"))

        self.audio_config = audio_config

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

        Replacement normally occurs following a TCP reconnect.
        """

        if not isinstance(
            state,
            NodeState,
        ):
            raise TypeError(("state must be a NodeState instance"))

        node_id = _require_node_id(state.node_id)

        # --------------------------------------------------------------
        # AUDIO CONTRACT
        # --------------------------------------------------------------
        #
        # NodeState should be using the same acquisition settings as the
        # StreamManager that owns it.
        # --------------------------------------------------------------

        if state.audio_config.sample_rate != self.audio_config.sample_rate:
            raise ValueError(
                (f"node {node_id} AudioConfig sample_rate does not match StreamManager")
            )

        if state.audio_config.frames_per_block != self.audio_config.frames_per_block:
            raise ValueError(
                (
                    f"node {node_id} AudioConfig "
                    "frames_per_block does not match "
                    "StreamManager"
                )
            )

        if state.audio_config.channels != self.audio_config.channels:
            raise ValueError(
                (f"node {node_id} AudioConfig channels do not match StreamManager")
            )

        if (
            state.audio_config.sample_width_bytes
            != self.audio_config.sample_width_bytes
        ):
            raise ValueError(
                (
                    f"node {node_id} AudioConfig "
                    "sample width does not match "
                    "StreamManager"
                )
            )

        self.nodes[node_id] = state

    # ==================================================================
    # AUDIO-TIMELINE RESET
    # ==================================================================

    def reset_audio_buffers(
        self,
    ) -> None:
        """
        Reset sample-dependent PCM state before a new acquisition START.

        Why this is necessary
        ---------------------
        Every ESP32 restarts sampleIndex from zero on START.

        Therefore the following values from the previous session cannot
        remain authoritative:

            buffered PCM
            expected_next_audio_sample
            last_sample_index
            latest session-specific SYNC marker

        Network sequence tracking is deliberately preserved here.

        The first packet belonging to the new non-zero session will let
        NodeState.observe_header() establish/reset the complete session
        state.

        Environment history is likewise left to NodeState's explicit
        session transition because it carries its own session IDs.
        """

        for state in self.nodes.values():
            state.audio_blocks.clear()

            state.expected_next_audio_sample = None

            state.last_sample_index = None

            state.latest_sync = None

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

        if not isinstance(
            block,
            AudioBlock,
        ):
            raise TypeError(("block must be an AudioBlock instance"))

        state = self.nodes.get(block.node_id)

        if state is None:
            raise KeyError((f"node {block.node_id} is not registered"))

        state.add_audio(block)

    # ==================================================================
    # NODE-ID NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize_node_ids(
        node_ids: Iterable[int],
    ) -> tuple[int, ...]:
        """
        Normalize and validate a requested collection of node IDs.
        """

        try:
            raw_nodes = tuple(node_ids)

        except TypeError as exc:
            raise TypeError(("node_ids must be an iterable of integers")) from exc

        if not raw_nodes:
            raise ValueError(("At least one node_id is required"))

        normalized = tuple(_require_node_id(node_id) for node_id in raw_nodes)

        if len(set(normalized)) != len(normalized):
            raise ValueError(("node_ids cannot contain duplicates"))

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
        Find the newest approximately aligned AUDIO block available from
        every requested node.

        Why newest-only comparison is insufficient
        -------------------------------------------
        TCP delivery between nodes is asynchronous.

        Example:

            Node 1 newest = 10240
            Node 2 newest = 9216
            Node 3 newest = 10240

        Nodes 1 and 3 may still retain their blocks beginning at 9216.

        Therefore:

            target =
                newest start position of the least-advanced stream

        Each other node's buffered history is then searched for the
        closest block start.

        The selected blocks must:

            belong to one common session
            fall within coarse synchronization tolerance

        This operation never compensates acoustic propagation delay.
        """

        requested_nodes = self._normalize_node_ids(node_ids)

        # ==============================================================
        # TOLERANCE
        # ==============================================================

        if tolerance_samples is None:
            tolerance = self.audio_config.sync_tolerance_samples

        else:
            tolerance = _require_integer(
                tolerance_samples,
                name="tolerance_samples",
            )

        if tolerance < 0:
            raise ValueError(("tolerance_samples cannot be negative"))

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

        for node_id in requested_nodes:
            state = self.nodes.get(node_id)

            if state is None:
                return None

            blocks = tuple(state.audio_blocks)

            if not blocks:
                return None

            available[node_id] = blocks

        # ==============================================================
        # COMMON TARGET
        # ==============================================================
        #
        # The least-advanced node's newest block determines the newest
        # block-start region all requested nodes may currently share.
        # ==============================================================

        target = min(int(blocks[-1].sample_index) for blocks in available.values())

        # ==============================================================
        # SELECT NEAREST BLOCK
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
                key=lambda block: abs(int(block.sample_index) - target),
            )

            offset = int(nearest.sample_index) - target

            if abs(offset) > tolerance:
                return None

            selected[node_id] = nearest

            offsets[node_id] = int(offset)

        # ==============================================================
        # SESSION CONSISTENCY
        # ==============================================================

        session_ids = {int(block.session_id) for block in selected.values()}

        if len(session_ids) != 1:
            return None

        session_id = next(iter(session_ids))

        # AUDIO packets from acquisition should always use a non-zero
        # session ID.
        if session_id == 0:
            return None

        return AlignedBlocks(
            target_sample_index=int(target),
            blocks=selected,
            offsets=offsets,
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
        Return environmental telemetry nearest an audio sample position.

        Node 1 is preferred because the current hardware architecture
        places the BME280 on the master node.
        """

        try:
            sample_index = _require_uint64(
                sample_index,
                name="sample_index",
            )

            preferred_node = _require_node_id(preferred_node)

        except (
            TypeError,
            ValueError,
        ):
            return None

        state = self.nodes.get(preferred_node)

        if state is None:
            return None

        return state.environment_near(sample_index)

    # ==================================================================
    # SAMPLE-INDEXED PCM WINDOW
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
        Reconstruct a mono PCM16 window using absolute sample positions.

        Missing samples are filled rather than removed.

        Example
        -------
        Received:

            samples 0..1023
            samples 2048..3071

        Requested:

            samples 0..3071

        Result:

            real PCM
            1024 fill samples
            real PCM


        This is essential for localization.

        Removing the missing region would shift all subsequent waveform
        positions and corrupt TDOA timing.
        """

        # ==============================================================
        # VALIDATION
        # ==============================================================

        node_id = _require_node_id(node_id)

        start_sample = _require_uint64(
            start_sample,
            name="start_sample",
        )

        length = _require_integer(
            length,
            name="length",
        )

        fill_value = _require_integer(
            fill_value,
            name="fill_value",
        )

        if length <= 0:
            return np.empty(
                0,
                dtype=np.int16,
            )

        if not (PCM16_MIN <= fill_value <= PCM16_MAX):
            raise ValueError(("fill_value must fit signed PCM16"))

        if start_sample + length > UINT64_MAX + 1:
            raise ValueError(
                ("requested audio window exceeds uint64 sampleIndex space")
            )

        # ==============================================================
        # NODE
        # ==============================================================

        state = self.nodes.get(node_id)

        if state is None:
            raise KeyError((f"node {node_id} is not registered"))

        end_sample = start_sample + length

        # ==============================================================
        # OUTPUT TIMELINE
        # ==============================================================

        output = np.full(
            length,
            fill_value,
            dtype=np.int16,
        )

        # Snapshot the deque before reconstruction.
        blocks = tuple(state.audio_blocks)

        if not blocks:
            return output

        # ==============================================================
        # SESSION CONSISTENCY
        # ==============================================================
        #
        # NodeState normally guarantees one session per audio deque.
        #
        # This additional guard prevents a malformed/test-injected mixed
        # timeline from being silently reconstructed.
        # ==============================================================

        session_ids = {int(block.session_id) for block in blocks}

        if len(session_ids) > 1:
            raise RuntimeError(
                (f"node {node_id} audio buffer contains multiple session IDs")
            )

        # ==============================================================
        # COPY OVERLAPPING PCM
        # ==============================================================

        for block in blocks:
            block_start = int(block.sample_index)

            block_end = int(block.end_sample)

            # ----------------------------------------------------------
            # BLOCK COMPLETELY BEFORE WINDOW
            # ----------------------------------------------------------

            if block_end <= start_sample:
                continue

            # ----------------------------------------------------------
            # BLOCK COMPLETELY AFTER WINDOW
            # ----------------------------------------------------------

            if block_start >= end_sample:
                continue

            # ----------------------------------------------------------
            # INTERSECTION
            # ----------------------------------------------------------

            overlap_start = max(
                start_sample,
                block_start,
            )

            overlap_end = min(
                end_sample,
                block_end,
            )

            if overlap_start >= overlap_end:
                continue

            src_start = overlap_start - block_start

            dst_start = overlap_start - start_sample

            count = overlap_end - overlap_start

            source = block.samples[src_start : src_start + count]

            # ----------------------------------------------------------
            # DEFENSIVE SIZE LIMIT
            # ----------------------------------------------------------

            actual_count = min(
                int(source.size),
                int(count),
            )

            if actual_count <= 0:
                continue

            output[dst_start : dst_start + actual_count] = source[:actual_count]

        return output


# ======================================================================
# CONTINUOUS WAV RECORDER
# ======================================================================


class WavRecorder:
    """
    Continuous per-node PCM16 WAV recorder.

    Timeline preservation
    ---------------------
    sampleIndex gaps are represented as digital silence.

    Missing PCM is never removed from the output timeline.

    Continuous recordings therefore remain useful for:

        synchronization inspection
        debugging
        offline DSP
        replay
        research-data review
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        root_dir: (str | os.PathLike[str]),
        audio_config: AudioConfig,
    ) -> None:

        if not isinstance(
            audio_config,
            AudioConfig,
        ):
            raise TypeError(("audio_config must be an AudioConfig instance"))

        self.root = Path(root_dir)

        self.audio_config = audio_config

        self._files: dict[
            int,
            wave.Wave_write,
        ] = {}

        self._expected_sample: dict[
            int,
            int,
        ] = {}

        self._session_label: str | None = None

    # ==================================================================
    # RECORDING STATE
    # ==================================================================

    @property
    def is_recording(
        self,
    ) -> bool:
        """
        Whether at least one per-node WAV file is currently open.
        """

        return bool(self._files)

    # ==================================================================
    # SESSION LABEL
    # ==================================================================

    @staticmethod
    def _validate_session_label(
        session_label: str,
    ) -> str:
        """
        Validate a single-component recording directory name.

        ReceiverServer performs stricter platform-safe validation before
        calling this API. This check remains for standalone use.
        """

        if not isinstance(
            session_label,
            str,
        ):
            raise TypeError(("session_label must be a string"))

        label = session_label.strip()

        if not label:
            raise ValueError(("session_label cannot be empty"))

        path = Path(label)

        if (
            path.is_absolute()
            or len(path.parts) != 1
            or label
            in {
                ".",
                "..",
            }
        ):
            raise ValueError(("session_label must be a single directory name"))

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
        Start one PCM16 WAV file for every requested acoustic node.

        File creation is transactional.

        If opening/configuring any WAV fails, all files already opened
        for this attempted recording session are closed.
        """

        label = self._validate_session_label(session_label)

        nodes = StreamManager._normalize_node_ids(node_ids)

        # ==============================================================
        # CLOSE PREVIOUS RECORDING
        # ==============================================================

        self.stop()

        # ==============================================================
        # SESSION DIRECTORY
        # ==============================================================

        session_dir = self.root / label

        session_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==============================================================
        # OPEN NODE WAV FILES
        # ==============================================================

        try:
            for node_id in nodes:
                path = session_dir / f"node_{node_id}.wav"

                wav = wave.open(
                    str(path),
                    "wb",
                )

                wav.setnchannels(self.audio_config.channels)

                wav.setsampwidth(self.audio_config.sample_width_bytes)

                wav.setframerate(self.audio_config.sample_rate)

                self._files[node_id] = wav

                # ------------------------------------------------------
                # Every START begins sampleIndex at zero.
                # ------------------------------------------------------

                self._expected_sample[node_id] = 0

        except Exception:
            self.stop()

            raise

        self._session_label = label

    # ==================================================================
    # DIGITAL-SILENCE WRITER
    # ==================================================================

    @staticmethod
    def _write_silence(
        wav: wave.Wave_write,
        sample_count: int,
    ) -> None:
        """
        Write PCM16 digital silence using bounded memory.
        """

        remaining = int(sample_count)

        if remaining <= 0:
            return

        chunk_size = min(
            remaining,
            WAV_SILENCE_CHUNK_SAMPLES,
        )

        chunk = np.zeros(
            chunk_size,
            dtype="<i2",
        )

        while remaining > 0:
            count = min(
                remaining,
                int(chunk.size),
            )

            wav.writeframesraw(chunk[:count].tobytes())

            remaining -= count

    # ==================================================================
    # WRITE AUDIO BLOCK
    # ==================================================================

    def write(
        self,
        block: AudioBlock,
    ) -> None:
        """
        Write one AUDIO block while preserving the sampleIndex timeline.

        Cases
        -----
        block.sample_index > expected

            A gap exists.
            Digital silence is inserted.


        block.sample_index < expected

            Data overlaps already-written PCM.
            The duplicate prefix is skipped.


        block.sample_index == expected

            PCM is written directly.
        """

        if not isinstance(
            block,
            AudioBlock,
        ):
            raise TypeError(("block must be an AudioBlock instance"))

        wav = self._files.get(block.node_id)

        if wav is None:
            return

        expected = self._expected_sample.get(block.node_id)

        if expected is None:
            return

        block_start = _require_uint64(
            block.sample_index,
            name="AudioBlock sample_index",
        )

        samples = np.asarray(
            block.samples,
            dtype="<i2",
        )

        if samples.ndim != 1:
            raise ValueError(("WAV recorder expects mono 1-D PCM"))

        if samples.size == 0:
            return

        # ==============================================================
        # GAP
        # ==============================================================

        if block_start > expected:
            gap = block_start - expected

            self._write_silence(
                wav,
                gap,
            )

            expected = block_start

        # ==============================================================
        # OVERLAP / DUPLICATE
        # ==============================================================

        elif block_start < expected:
            overlap = expected - block_start

            # Entire AUDIO packet is already represented in the file.
            if overlap >= samples.size:
                return

            samples = samples[overlap:]

            block_start = expected

        # ==============================================================
        # WRITE PCM
        # ==============================================================

        wav.writeframesraw(samples.tobytes())

        self._expected_sample[block.node_id] = block_start + int(samples.size)

    # ==================================================================
    # STOP RECORDING
    # ==================================================================

    def stop(
        self,
    ) -> None:
        """
        Finalize and close all open WAV files.

        Recorder state is cleared before close operations so one failed
        filesystem close cannot leave the object logically recording.
        """

        files = list(self._files.values())

        # ==============================================================
        # CLEAR LOGICAL STATE FIRST
        # ==============================================================

        self._files.clear()

        self._expected_sample.clear()

        self._session_label = None

        # ==============================================================
        # FINALIZE FILES
        # ==============================================================

        for wav in files:
            try:
                wav.close()

            except Exception:
                # Recorder shutdown must not prevent the rest of the
                # acquisition stack from closing.
                pass
