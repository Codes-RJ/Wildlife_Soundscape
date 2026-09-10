"""
Event-audio discovery and visualization helpers.

Wildlife Soundscape Mapping & Behavior Analysis System
------------------------------------------------------

Purpose
-------
This module provides the dashboard-side interface to persisted event WAV
files.

EventPipeline stores synchronized event audio on the filesystem rather
than as SQLite BLOB data.

Typical event directory
-----------------------

    data/events/event_000001/

        node_1.wav
        node_2.wav
        node_3.wav


Responsibilities
----------------
This module handles:

    event-directory resolution
    node WAV discovery
    WAV metadata validation
    PCM16 mono decoding
    bounded audio preview generation
    waveform visualization
    spectrogram visualization


This module does NOT:

    access SQLite directly
    modify event recordings
    perform classification
    perform localization
    perform event detection
    alter research analytics


Scientific note
---------------
Dashboard waveform and spectrogram operations are for inspection and
visualization.

They do not modify the raw persisted event recordings.
"""

from __future__ import annotations


# ======================================================================
# STANDARD LIBRARY
# ======================================================================


import io
import math
import re
import wave

from dataclasses import (
    dataclass,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
    Iterable,
)


# ======================================================================
# THIRD-PARTY
# ======================================================================


import numpy as np

import plotly.graph_objects as go

from wildlife_soundscape.dashboard.palette import apply_chart_theme


# ======================================================================
# PROJECT IMPORTS
# ======================================================================


from wildlife_soundscape.dsp.features import (
    calculate_log_spectrogram,
)


# ======================================================================
# CONSTANTS
# ======================================================================


DEFAULT_MAX_AUDIO_PREVIEW_S = 30.0


DEFAULT_MAX_WAVEFORM_POINTS = 5000


DEFAULT_SPECTROGRAM_N_FFT = 2048


DEFAULT_SPECTROGRAM_HOP_LENGTH = 512


EXPECTED_SAMPLE_WIDTH_BYTES = 2


EXPECTED_CHANNEL_COUNT = 1


NODE_FILENAME_PATTERN = re.compile(
    r"^node_(\d+)\.wav$",
    flags=re.IGNORECASE,
)


# ======================================================================
# AUDIO FILE MODEL
# ======================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class EventAudioFile:
    """
    Metadata describing one persisted node WAV file.
    """

    node_id: int

    path: Path

    sample_rate: int

    channels: int

    sample_width_bytes: int

    frame_count: int

    duration_s: float

    file_size_bytes: int

    # ==================================================================
    # SERIALIZATION
    # ==================================================================

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "node_id": self.node_id,
            "path": str(self.path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "file_size_bytes": self.file_size_bytes,
        }


# ======================================================================
# ROW ACCESS
# ======================================================================


def _row_value(
    row: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a value from dict-like or sqlite3.Row-like input.
    """

    getter = getattr(
        row,
        "get",
        None,
    )

    if callable(getter):
        try:
            return getter(
                key,
                default,
            )

        except (
            KeyError,
            TypeError,
        ):
            pass

    try:
        return row[key]

    except (
        KeyError,
        IndexError,
        TypeError,
    ):
        return default


# ======================================================================
# POSITIVE INTEGER VALIDATION
# ======================================================================


def _positive_int(
    value: int,
    *,
    name: str,
) -> int:
    """
    Require a positive Python integer.
    """

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        int,
    ):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")

    return int(value)


# ======================================================================
# POSITIVE FLOAT VALIDATION
# ======================================================================


def _positive_finite(
    value: float,
    *,
    name: str,
) -> float:
    """
    Require a finite numeric value > 0.
    """

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(f"{name} must be numeric.") from exc

    if not math.isfinite(result) or result <= 0.0:
        raise ValueError((f"{name} must be finite and greater than 0."))

    return result


# ======================================================================
# EVENT DIRECTORY RESOLUTION
# ======================================================================


def resolve_event_directory(
    event_row: Any,
    *,
    project_root: str | Path = Path("."),
) -> Path | None:
    """
    Resolve ``event_directory`` from one persisted event row.

    Relative paths are interpreted relative to project_root.

    Returns None when the event contains no directory path.
    """

    raw_directory = _row_value(
        event_row,
        "event_directory",
    )

    if raw_directory is None:
        return None

    directory_text = str(raw_directory).strip()

    if not (directory_text):
        return None

    root = Path(project_root)

    path = Path(directory_text)

    if not (path.is_absolute()):
        path = root / path

    return path.expanduser().resolve(strict=False)


# ======================================================================
# NODE ID FROM FILENAME
# ======================================================================


def node_id_from_wav_filename(
    path: str | Path,
) -> int | None:
    """
    Extract node identifier from:

        node_<id>.wav

    Examples
    --------
    node_1.wav -> 1
    node_3.wav -> 3
    example.wav -> None
    """

    path = Path(path)

    match = NODE_FILENAME_PATTERN.match(path.name)

    if match is None:
        return None

    node_id = int(match.group(1))

    if node_id <= 0:
        return None

    return node_id


# ======================================================================
# WAV METADATA
# ======================================================================


def read_wav_metadata(
    path: str | Path,
    *,
    node_id: int | None = None,
    require_pcm16_mono: bool = True,
) -> EventAudioFile:
    """
    Read and validate metadata from one WAV file.

    Parameters
    ----------
    path
        WAV file path.

    node_id
        Optional explicit node identifier.

        When omitted, the node ID is derived from the filename.

    require_pcm16_mono
        Enforce the event-recording contract used by this project:

            channels = 1
            sample width = 2 bytes
    """

    path = Path(path).expanduser().resolve(strict=False)

    if not (path.exists()):
        raise FileNotFoundError(f"WAV file does not exist: {path}")

    if not (path.is_file()):
        raise ValueError(f"WAV path is not a file: {path}")

    if path.suffix.lower() != ".wav":
        raise ValueError(f"Audio file must use .wav extension: {path}")

    if node_id is None:
        node_id = node_id_from_wav_filename(path)

    if node_id is None:
        raise ValueError(
            (f"Unable to determine node_id from WAV filename: {path.name}")
        )

    node_id = _positive_int(
        node_id,
        name="node_id",
    )

    try:
        with wave.open(
            str(path),
            "rb",
        ) as wav_file:
            channels = int(wav_file.getnchannels())

            sample_width_bytes = int(wav_file.getsampwidth())

            sample_rate = int(wav_file.getframerate())

            frame_count = int(wav_file.getnframes())

            compression_type = wav_file.getcomptype()

    except (
        wave.Error,
        EOFError,
    ) as exc:
        raise ValueError((f"Invalid or unsupported WAV file: {path}")) from exc

    # ==============================================================
    # BASIC WAV VALIDATION
    # ==============================================================

    if channels <= 0:
        raise ValueError("WAV channel count must be positive.")

    if sample_width_bytes <= 0:
        raise ValueError("WAV sample width must be positive.")

    if sample_rate <= 0:
        raise ValueError("WAV sample rate must be positive.")

    if frame_count < 0:
        raise ValueError("WAV frame count cannot be negative.")

    if compression_type != "NONE":
        raise ValueError(("Dashboard currently supports uncompressed PCM WAV only."))

    # ==============================================================
    # PROJECT AUDIO CONTRACT
    # ==============================================================

    if require_pcm16_mono:
        if channels != EXPECTED_CHANNEL_COUNT:
            raise ValueError((f"Event WAV must be mono. Found {channels} channels."))

        if sample_width_bytes != EXPECTED_SAMPLE_WIDTH_BYTES:
            raise ValueError(
                (
                    "Event WAV must contain PCM16 "
                    "(2-byte samples). "
                    f"Found sample width "
                    f"{sample_width_bytes} bytes."
                )
            )

    duration_s = frame_count / float(sample_rate)

    return EventAudioFile(
        node_id=node_id,
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
        frame_count=frame_count,
        duration_s=float(duration_s),
        file_size_bytes=int(path.stat().st_size),
    )


# ======================================================================
# DISCOVER EVENT WAV FILES
# ======================================================================


def discover_event_audio_files(
    event_row: Any,
    *,
    project_root: str | Path = Path("."),
    expected_nodes: Iterable[int] | None = None,
    require_pcm16_mono: bool = True,
) -> tuple[
    EventAudioFile,
    ...,
]:
    """
    Discover all valid node WAV files stored for one event.

    Files are returned in ascending node-ID order.

    Missing individual node WAVs are tolerated so the dashboard can
    still inspect partially persisted events.
    """

    event_directory = resolve_event_directory(
        event_row,
        project_root=project_root,
    )

    if event_directory is None:
        return ()

    if not (event_directory.exists()):
        return ()

    if not (event_directory.is_dir()):
        return ()

    expected: set[int] | None = None

    if expected_nodes is not None:
        expected = set()

        for node_id in expected_nodes:
            expected.add(
                _positive_int(
                    node_id,
                    name="expected node_id",
                )
            )

    discovered: list[EventAudioFile] = []

    for path in sorted(event_directory.glob("*.wav")):
        discovered_node_id = node_id_from_wav_filename(path)

        if discovered_node_id is None:
            continue

        if expected is not None and discovered_node_id not in expected:
            continue

        try:
            metadata = read_wav_metadata(
                path,
                require_pcm16_mono=require_pcm16_mono,
            )

        except (
            FileNotFoundError,
            ValueError,
        ):
            # ----------------------------------------------------------
            # One malformed/missing node recording should not prevent
            # discovery of other valid recordings for the event.
            # ----------------------------------------------------------

            continue

        discovered.append(metadata)

    discovered.sort(key=lambda item: item.node_id)

    return tuple(discovered)


# ======================================================================
# RAW PCM16 LOADING
# ======================================================================


def load_pcm16_mono(
    audio_file: EventAudioFile | str | Path,
    *,
    max_duration_s: float | None = None,
) -> tuple[
    np.ndarray,
    int,
]:
    """
    Decode one event WAV into mono PCM16 samples.

    Returns
    -------
    pcm
        Contiguous numpy.int16 array.

    sample_rate
        WAV sample rate.


    max_duration_s
    ----------------
    When provided, only the first N seconds are decoded.

    This limit affects dashboard preview only and does not modify the
    source WAV.
    """

    if isinstance(
        audio_file,
        EventAudioFile,
    ):
        metadata = audio_file

    else:
        metadata = read_wav_metadata(audio_file)

    if metadata.channels != EXPECTED_CHANNEL_COUNT:
        raise ValueError(("PCM decoder currently requires mono event WAV audio."))

    if metadata.sample_width_bytes != EXPECTED_SAMPLE_WIDTH_BYTES:
        raise ValueError(("PCM decoder currently requires 16-bit event WAV audio."))

    frame_limit = metadata.frame_count

    if max_duration_s is not None:
        max_duration_s = _positive_finite(
            max_duration_s,
            name="max_duration_s",
        )

        frame_limit = min(
            frame_limit,
            int(math.ceil(max_duration_s * metadata.sample_rate)),
        )

    try:
        with wave.open(
            str(metadata.path),
            "rb",
        ) as wav_file:
            raw_bytes = wav_file.readframes(frame_limit)

    except (
        wave.Error,
        EOFError,
    ) as exc:
        raise ValueError((f"Unable to decode event WAV: {metadata.path}")) from exc

    pcm = np.frombuffer(
        raw_bytes,
        dtype="<i2",
    )

    pcm = np.ascontiguousarray(
        pcm,
        dtype=np.int16,
    )

    return (
        pcm,
        metadata.sample_rate,
    )


# ======================================================================
# PCM16 -> FLOAT
# ======================================================================


def pcm16_to_float(
    pcm: np.ndarray,
) -> np.ndarray:
    """
    Convert PCM16 samples to float32 approximately in [-1, 1].
    """

    if not isinstance(
        pcm,
        np.ndarray,
    ):
        raise TypeError("pcm must be a numpy array.")

    if pcm.ndim != 1:
        raise ValueError("pcm must be one-dimensional.")

    if pcm.dtype != np.int16:
        try:
            pcm = np.asarray(
                pcm,
                dtype=np.int16,
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(("pcm must contain int16-compatible samples.")) from exc

    signal = pcm.astype(np.float32) / 32768.0

    return np.ascontiguousarray(
        signal,
        dtype=np.float32,
    )


# ======================================================================
# WAV PREVIEW ENCODING
# ======================================================================


def build_preview_wav_bytes(
    audio_file: EventAudioFile | str | Path,
    *,
    max_duration_s: float = DEFAULT_MAX_AUDIO_PREVIEW_S,
) -> bytes:
    """
    Create browser-playable WAV bytes for a bounded dashboard preview.

    The persisted source recording is never modified.

    If the original event WAV is shorter than max_duration_s, the full
    event is included.
    """

    max_duration_s = _positive_finite(
        max_duration_s,
        name="max_duration_s",
    )

    pcm, sample_rate = load_pcm16_mono(
        audio_file,
        max_duration_s=max_duration_s,
    )

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav_file:
        wav_file.setnchannels(EXPECTED_CHANNEL_COUNT)

        wav_file.setsampwidth(EXPECTED_SAMPLE_WIDTH_BYTES)

        wav_file.setframerate(sample_rate)

        wav_file.writeframes(
            pcm.astype(
                "<i2",
                copy=False,
            ).tobytes()
        )

    return buffer.getvalue()


# ======================================================================
# WAVEFORM DECIMATION
# ======================================================================


def _waveform_decimation_indices(
    sample_count: int,
    *,
    max_points: int,
) -> np.ndarray:
    """
    Return evenly distributed visualization indices.

    Scientific processing is never performed on this decimated signal.
    """

    sample_count = _positive_int(
        sample_count,
        name="sample_count",
    )

    max_points = _positive_int(
        max_points,
        name="max_points",
    )

    if sample_count <= max_points:
        return np.arange(
            sample_count,
            dtype=np.int64,
        )

    return np.linspace(
        0,
        sample_count - 1,
        num=max_points,
        dtype=np.int64,
    )


# ======================================================================
# EMPTY AUDIO FIGURE
# ======================================================================


def _empty_audio_figure(
    *,
    title: str,
    message: str,
) -> go.Figure:
    """
    Produce a simple Plotly empty-state figure.
    """

    figure = go.Figure()

    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
    )

    figure.update_xaxes(visible=False)

    figure.update_yaxes(visible=False)

    apply_chart_theme(figure)
    figure.update_layout(
        title=title,
        height=300,
        margin=dict(
            l=40,
            r=30,
            t=60,
            b=40,
        ),
    )

    return figure


# ======================================================================
# WAVEFORM FIGURE
# ======================================================================


def build_waveform_figure(
    audio_file: EventAudioFile | str | Path,
    *,
    max_duration_s: float = DEFAULT_MAX_AUDIO_PREVIEW_S,
    max_points: int = DEFAULT_MAX_WAVEFORM_POINTS,
) -> go.Figure:
    """
    Build a Plotly waveform for one event-node recording.

    The signal may be decimated for browser rendering, but the source
    recording remains untouched.
    """

    max_duration_s = _positive_finite(
        max_duration_s,
        name="max_duration_s",
    )

    max_points = _positive_int(
        max_points,
        name="max_points",
    )

    if isinstance(
        audio_file,
        EventAudioFile,
    ):
        metadata = audio_file

    else:
        metadata = read_wav_metadata(audio_file)

    pcm, sample_rate = load_pcm16_mono(
        metadata,
        max_duration_s=max_duration_s,
    )

    if pcm.size == 0:
        return _empty_audio_figure(
            title=(f"Node {metadata.node_id} Waveform"),
            message="Audio file contains no samples.",
        )

    signal = pcm16_to_float(pcm)

    indices = _waveform_decimation_indices(
        int(signal.size),
        max_points=max_points,
    )

    times = indices.astype(np.float64) / float(sample_rate)

    values = signal[indices]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines",
            name=f"Node {metadata.node_id}",
            hovertemplate=("Time: %{x:.4f} s<br>Amplitude: %{y:.5f}<extra></extra>"),
        )
    )

    apply_chart_theme(figure)
    figure.update_layout(
        title=(f"Node {metadata.node_id} Waveform"),
        xaxis_title="Time (s)",
        yaxis_title="Normalized Amplitude",
        height=330,
        margin=dict(
            l=50,
            r=30,
            t=60,
            b=45,
        ),
        hovermode="x",
    )

    figure.update_yaxes(
        range=[
            -1.0,
            1.0,
        ]
    )

    return figure


# ======================================================================
# SPECTROGRAM FIGURE
# ======================================================================


def build_spectrogram_figure(
    audio_file: EventAudioFile | str | Path,
    *,
    max_duration_s: float = DEFAULT_MAX_AUDIO_PREVIEW_S,
    n_fft: int = DEFAULT_SPECTROGRAM_N_FFT,
    hop_length: int = DEFAULT_SPECTROGRAM_HOP_LENGTH,
) -> go.Figure:
    """
    Build a log-power spectrogram for dashboard inspection.

    DSP feature extraction is not repeated or modified.

    This uses the project's existing log-spectrogram helper purely for
    visualization.
    """

    max_duration_s = _positive_finite(
        max_duration_s,
        name="max_duration_s",
    )

    n_fft = _positive_int(
        n_fft,
        name="n_fft",
    )

    hop_length = _positive_int(
        hop_length,
        name="hop_length",
    )

    if hop_length > n_fft:
        raise ValueError("hop_length cannot exceed n_fft.")

    if isinstance(
        audio_file,
        EventAudioFile,
    ):
        metadata = audio_file

    else:
        metadata = read_wav_metadata(audio_file)

    pcm, sample_rate = load_pcm16_mono(
        metadata,
        max_duration_s=max_duration_s,
    )

    if pcm.size == 0:
        return _empty_audio_figure(
            title=(f"Node {metadata.node_id} Spectrogram"),
            message="Audio file contains no samples.",
        )

    signal = pcm16_to_float(pcm)

    (
        log_spectrogram,
        frequencies,
        times,
    ) = calculate_log_spectrogram(
        signal,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if log_spectrogram.size == 0 or frequencies.size == 0 or times.size == 0:
        return _empty_audio_figure(
            title=(f"Node {metadata.node_id} Spectrogram"),
            message="Unable to construct spectrogram.",
        )

    figure = go.Figure(
        data=go.Heatmap(
            x=times,
            y=frequencies,
            z=log_spectrogram,
            colorbar=dict(title="dB"),
            hovertemplate=(
                "Time: %{x:.4f} s<br>"
                "Frequency: %{y:.1f} Hz<br>"
                "Level: %{z:.2f} dB"
                "<extra></extra>"
            ),
        )
    )

    apply_chart_theme(figure)
    figure.update_layout(
        title=(f"Node {metadata.node_id} Spectrogram"),
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=420,
        margin=dict(
            l=55,
            r=30,
            t=60,
            b=45,
        ),
    )

    return figure


# ======================================================================
# EVENT AUDIO SUMMARY
# ======================================================================


def event_audio_summary(
    event_row: Any,
    *,
    project_root: str | Path = Path("."),
    expected_nodes: Iterable[int] | None = None,
) -> dict[
    str,
    Any,
]:
    """
    Build a dashboard-friendly summary of available event recordings.
    """

    event_directory = resolve_event_directory(
        event_row,
        project_root=project_root,
    )

    files = discover_event_audio_files(
        event_row,
        project_root=project_root,
        expected_nodes=expected_nodes,
    )

    return {
        "event_directory": (
            str(event_directory) if event_directory is not None else None
        ),
        "available": bool(files),
        "node_count": len(files),
        "nodes": tuple(audio_file.node_id for audio_file in files),
        "files": tuple(audio_file.to_dict() for audio_file in files),
    }
