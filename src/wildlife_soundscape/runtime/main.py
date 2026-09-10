from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging

from datetime import datetime
from typing import Any

from wildlife_soundscape.core.config import CONFIG

from wildlife_soundscape.runtime.server import (
    ReceiverServer,
)


# ======================================================================
# MODULE LOGGER
# ======================================================================


logger = logging.getLogger(__name__)


# ======================================================================
# LOGGING
# ======================================================================


def configure_logging() -> None:
    """
    Configure application-wide console logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
    )


# ======================================================================
# DISPLAY HELPERS
# ======================================================================


def _format_session_id(
    session_id: int | None,
) -> str:
    """
    Format a session identifier consistently for CLI output.
    """

    if session_id is None:
        return "NONE"

    return f"0x{session_id:08X}"


def _load_json(
    value: Any,
) -> Any | None:
    """
    Safely decode a JSON value read from SQLite.

    Returns None when the stored value is unavailable or malformed.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
            int,
            float,
            bool,
        ),
    ):
        return value

    try:
        return json.loads(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _format_optional_float(
    value: Any,
    *,
    precision: int = 3,
    suffix: str = "",
) -> str:
    """
    Safely format an optional numeric database/runtime value.
    """

    if value is None:
        return "N/A"

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return str(value)

    return f"{number:.{precision}f}{suffix}"


# ======================================================================
# CLASSIFICATION STATUS
# ======================================================================


def print_classification_status(
    server: ReceiverServer,
    *,
    indent: str = "",
) -> None:
    """
    Display configured and actually active classifier state.

    This intentionally distinguishes:

        classification disabled
        configured backend
        active backend
        heuristic fallback
    """

    classification_config = server.config.classification

    # ==================================================================
    # DISABLED
    # ==================================================================

    if not (classification_config.enabled):
        print((f"{indent}Classification: DISABLED"))

        return

    # ==================================================================
    # CONFIGURED
    # ==================================================================

    configured_name = classification_config.backend.strip().lower()

    print((f"{indent}Classification: ENABLED | configured={configured_name}"))

    # ==================================================================
    # ACTIVE
    # ==================================================================

    backend = server.events.classifier_backend

    if backend is None:
        print((f"{indent}                active=NONE"))

        return

    print((f"{indent}                active={backend.name} v{backend.version}"))

    # ==================================================================
    # FALLBACK
    # ==================================================================

    if configured_name != "heuristic" and backend.name == "heuristic_backend":
        print(
            (
                f"{indent}"
                "                "
                "fallback=heuristic "
                f"(requested={configured_name})"
            )
        )


# ======================================================================
# STATUS DISPLAY
# ======================================================================


def print_status(
    server: ReceiverServer,
) -> None:
    """
    Print current receiver, node, synchronization, DSP and classifier
    state.
    """

    config = server.config

    print("\n" + "=" * 96)

    print("WILDLIFE SOUNDSCAPE RECEIVER STATUS")

    print((f"Active session: {_format_session_id(server.active_session_id)}"))

    print((f"Detected events this session: {server.events.completed_events}"))

    processing = server.processing.snapshot()
    print(
        "Processing: "
        f"queued={processing['queue_depth']} "
        f"accepted={processing['accepted']} "
        f"dropped={processing['dropped']} "
        f"max_latency={processing['max_latency_seconds']:.3f}s"
    )

    print_classification_status(server)

    print("=" * 96)

    # ==================================================================
    # NODE STATUS
    # ==================================================================

    for node_id in sorted(config.expected_nodes):
        connection = server.connections.get(node_id)

        if connection is None:
            print(f"Node {node_id}: OFFLINE")

            continue

        snapshot = connection.state.snapshot()

        heartbeat = snapshot.latest_heartbeat

        hello = snapshot.hello

        role = "MASTER" if (hello is not None and hello.master_node) else "SLAVE"

        print(
            (
                f"Node {node_id}: "
                f"{'ONLINE' if snapshot.connected else 'DISCONNECTED'} "
                f"{role} "
                f"| session={_format_session_id(snapshot.session_id)}"
            )
        )

        print(
            (
                "        "
                f"packets={snapshot.packets_received} "
                f"| audio={snapshot.audio_packets_received} "
                f"| crc={snapshot.crc_errors} "
                f"| protocol={snapshot.protocol_errors} "
                f"| seq_gaps={snapshot.sequence_gaps} "
                f"| seq_resets={snapshot.sequence_resets}"
            )
        )

        print(
            (
                "        "
                f"sample_gaps={snapshot.sample_gaps} "
                f"| old/dup={snapshot.duplicate_or_old_packets} "
                f"| clip={snapshot.clipped_packets} "
                f"| clock_fault={snapshot.clock_fault_packets} "
                f"| congest={snapshot.congested_packets}"
            )
        )

        if hello is not None:
            print(
                (
                    "        "
                    f"fw={hello.firmware} "
                    f"| {hello.sample_rate} Hz "
                    f"| {hello.frames_per_packet} samples/packet "
                    f"| sync_tol={hello.sync_tolerance_samples}"
                )
            )

        if heartbeat is not None:
            print(
                (
                    "        "
                    f"RSSI={heartbeat.wifi_rssi} dBm "
                    f"| heap={heartbeat.free_heap} "
                    f"| I2Serr={heartbeat.i2s_errors} "
                    f"| dropped={heartbeat.dropped_audio_blocks} "
                    f"| tx={heartbeat.transmitted_audio_blocks} "
                    f"| queue={heartbeat.audio_queue_depth} "
                    f"| streaming={heartbeat.streaming}"
                )
            )

            if heartbeat.bme_available is not None:
                print((f"        BME280 available={heartbeat.bme_available}"))

            if heartbeat.sync_received is not None:
                print(
                    (
                        "        "
                        f"SYNC received="
                        f"{heartbeat.sync_received} "
                        f"| clock healthy="
                        f"{heartbeat.clock_healthy}"
                    )
                )

        environment = snapshot.latest_environment

        if environment is not None:
            print(
                (
                    "        "
                    f"ENV="
                    f"{environment.temperature_c:.1f} C, "
                    f"{environment.humidity_percent:.1f}% RH, "
                    f"{environment.pressure_hpa:.1f} hPa"
                )
            )

        sync = snapshot.latest_sync

        if sync is not None:
            print(
                (
                    "        "
                    f"SYNC id={sync.sync_id} "
                    f"| sample={sync.sample_index} "
                    f"| session="
                    f"{_format_session_id(sync.session_id)}"
                )
            )

    # ==================================================================
    # STREAM ALIGNMENT
    # ==================================================================

    try:
        aligned = server.streams.latest_aligned_blocks()

    except Exception as exc:
        aligned = None

        print((f"Aligned latest block: error ({exc})"))

    else:
        if aligned is not None:
            print(
                (
                    "Aligned latest block "
                    f"@ sample "
                    f"{aligned.target_sample_index}: "
                    f"offsets={aligned.offsets}"
                )
            )

        else:
            print(("Aligned latest block: not available"))

    # ==================================================================
    # LATEST DSP
    # ==================================================================

    features = server.events.last_features

    if features is not None:
        print("-" * 96)

        snr_text = f"{features.snr_db:.2f} dB" if features.snr_db is not None else "N/A"

        print(
            (
                "LATEST DSP "
                f"| DB#{server.events.last_event_db_id} "
                f"| node={server.events.last_best_node_id} "
                f"| duration={features.duration_s:.3f}s "
                f"| SNR={snr_text}"
            )
        )

        print(
            (
                "            "
                f"dominant="
                f"{features.dominant_frequency_hz:.1f} Hz "
                f"| centroid="
                f"{features.spectral_centroid_hz:.1f} Hz "
                f"| bandwidth="
                f"{features.spectral_bandwidth_hz:.1f} Hz "
                f"| rolloff="
                f"{features.spectral_rolloff_hz:.1f} Hz"
            )
        )

        print(
            (
                "            "
                f"ZCR="
                f"{features.zero_crossing_rate:.5f} "
                f"| flatness="
                f"{features.spectral_flatness:.5f} "
                f"| flux="
                f"{features.spectral_flux:.5f}"
            )
        )

    # ==================================================================
    # LATEST CLASSIFICATION
    # ==================================================================

    classification = server.events.last_classification

    if classification is not None:
        print("-" * 96)

        print(
            (
                "LATEST CLASSIFICATION "
                f"| {classification.label.value.upper()} "
                f"| confidence="
                f"{classification.confidence:.3f} "
                f"| margin="
                f"{classification.margin:.3f}"
            )
        )

        if classification.second_label is not None:
            second_confidence = _format_optional_float(
                classification.second_confidence,
                precision=3,
            )

            print(
                (
                    "            "
                    f"second="
                    f"{classification.second_label.value.upper()} "
                    f"({second_confidence})"
                )
            )

        print(
            (
                "            "
                f"classifier="
                f"{classification.classifier_name} "
                f"v{classification.classifier_version}"
            )
        )

        if classification.reasons:
            print("            reasons:")

            for reason in classification.reasons:
                print((f"              - {reason}"))

    elif not (server.events.classification_enabled):
        print("-" * 96)

        print(("LATEST CLASSIFICATION | disabled"))

    print("=" * 96)


# ======================================================================
# DATABASE EVENT SUMMARY
# ======================================================================


def print_event_row(
    row,
) -> None:
    """
    Print one compact SQLite event summary.
    """

    # ==================================================================
    # POSITION
    # ==================================================================

    if row["x_m"] is None or row["y_m"] is None:
        position = "n/a"

    else:
        position = f"({row['x_m']:.3f}, {row['y_m']:.3f}) m"

    # ==================================================================
    # BEST NODE
    # ==================================================================

    best_node = "n/a" if row["best_node_id"] is None else str(row["best_node_id"])

    # ==================================================================
    # SNR
    # ==================================================================

    snr = "n/a" if row["snr_db"] is None else (f"{row['snr_db']:.2f} dB")

    # ==================================================================
    # DOMINANT FREQUENCY
    # ==================================================================

    dominant = (
        "n/a"
        if row["dominant_frequency_hz"] is None
        else (f"{row['dominant_frequency_hz']:.1f} Hz")
    )

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    if row["classification_label"] is None:
        classification_text = "unclassified"

    else:
        confidence = row["classification_confidence"]

        classification_text = str(row["classification_label"]).upper()

        if confidence is not None:
            classification_text += f" ({float(confidence):.3f})"

    # ==================================================================
    # OUTPUT
    # ==================================================================

    print(
        (
            f"DB#{row['id']} "
            f"| session="
            f"{_format_session_id(row['session_id'])} "
            f"| samples="
            f"{row['start_sample']}.."
            f"{row['end_sample']} "
            f"| nodes={row['trigger_nodes']} "
            f"| best_node={best_node} "
            f"| SNR={snr} "
            f"| dominant={dominant} "
            f"| class={classification_text} "
            f"| pos={position}"
        )
    )


# ======================================================================
# PERIODIC STATUS
# ======================================================================


async def status_loop(
    server: ReceiverServer,
) -> None:
    """
    Periodically print receiver status.

    A display error is logged rather than permanently terminating the
    background status task.
    """

    while True:
        await asyncio.sleep(server.config.print_status_every_s)

        try:
            print_status(server)

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(("Periodic status display failed"))


# ======================================================================
# LOCALIZATION DISPLAY
# ======================================================================


def print_localization_result(
    result,
) -> None:
    """
    Print one manual GCC-PHAT/TDOA localization result.
    """

    position = result.position

    print(
        (
            "Estimated source: "
            f"x={position.x:.4f} m, "
            f"y={position.y:.4f} m "
            f"| success={position.success}"
        )
    )

    print((f"Residual RMS: {position.residual_rms_meters:.4f} m-equivalent"))

    print((f"Speed of sound: {result.speed_of_sound_mps:.2f} m/s"))

    print("Pairwise TDOA measurements:")

    for measurement in result.measurements:
        state = "OK" if measurement.valid else (f"REJECT ({measurement.reason})")

        print(
            (
                "  "
                f"{measurement.node_a}"
                f"->{measurement.node_b}: "
                f"{measurement.delay_samples:+.3f} samples "
                "("
                f"{measurement.delay_seconds * 1e6:+.2f} us"
                "), "
                f"peak_ratio="
                f"{measurement.peak_ratio:.2f} "
                f"{state}"
            )
        )


# ======================================================================
# DETAILED EVENT DISPLAY
# ======================================================================


def print_event_details(
    row,
) -> None:
    """
    Print complete persisted information for one acoustic event.
    """

    print("\n" + "-" * 96)

    print("EVENT SUMMARY")

    print("-" * 96)

    print_event_row(row)

    # ==================================================================
    # CORE EVENT
    # ==================================================================

    print("\nCore Event:")

    print((f"  Detector event ID: {row['detector_event_id']}"))

    print((f"  Session ID: {_format_session_id(row['session_id'])}"))

    print((f"  Start sample: {row['start_sample']}"))

    print((f"  End sample: {row['end_sample']}"))

    print((f"  Trigger nodes: {row['trigger_nodes']}"))

    if row["peak_rms_dbfs"] is not None:
        print((f"  Peak RMS: {row['peak_rms_dbfs']:.2f} dBFS"))

    print((f"  Best DSP node: {row['best_node_id']}"))

    # ==================================================================
    # ENVIRONMENT
    # ==================================================================

    if row["temperature_c"] is not None:
        print("\nEnvironment:")

        print((f"  Temperature: {row['temperature_c']:.2f} C"))

        print((f"  Humidity: {row['humidity_percent']:.2f}%"))

        print((f"  Pressure: {row['pressure_hpa']:.2f} hPa"))

    # ==================================================================
    # LOCALIZATION
    # ==================================================================

    print("\nLocalization:")

    if row["x_m"] is None or row["y_m"] is None:
        print("  Localization unavailable")

    else:
        print((f"  Position: x={row['x_m']:.4f} m, y={row['y_m']:.4f} m"))

        print((f"  Success: {bool(row['localization_success'])}"))

        if row["localization_residual_m"] is not None:
            print((f"  Residual RMS: {row['localization_residual_m']:.5f} m"))

        if row["speed_of_sound_mps"] is not None:
            print((f"  Speed of sound: {row['speed_of_sound_mps']:.3f} m/s"))

    # ==================================================================
    # DSP FEATURES
    # ==================================================================

    if row["duration_s"] is not None:
        print("\nDSP Features:")

        print((f"  Source node: {row['feature_source_node_id']}"))

        print((f"  Duration: {row['duration_s']:.3f} s"))

        print((f"  RMS: {row['rms']:.6f}"))

        print((f"  Peak amplitude: {row['peak_amplitude']:.6f}"))

        print((f"  Crest factor: {row['crest_factor']:.3f}"))

        print((f"  Zero-crossing rate: {row['zero_crossing_rate']:.6f}"))

        print((f"  Dominant frequency: {row['dominant_frequency_hz']:.2f} Hz"))

        print((f"  Spectral centroid: {row['spectral_centroid_hz']:.2f} Hz"))

        print((f"  Spectral bandwidth: {row['spectral_bandwidth_hz']:.2f} Hz"))

        print((f"  Spectral rolloff: {row['spectral_rolloff_hz']:.2f} Hz"))

        print((f"  Spectral flatness: {row['spectral_flatness']:.6f}"))

        print((f"  Spectral flux: {row['spectral_flux']:.6f}"))

        print(
            (
                "  SNR: "
                + (f"{row['snr_db']:.2f} dB" if row["snr_db"] is not None else "N/A")
            )
        )

        # --------------------------------------------------------------
        # MFCC
        # --------------------------------------------------------------

        mfcc_mean = _load_json(row["mfcc_mean_json"])

        mfcc_std = _load_json(row["mfcc_std_json"])

        if mfcc_mean is not None:
            print((f"  MFCC mean: {mfcc_mean}"))

        if mfcc_std is not None:
            print((f"  MFCC std: {mfcc_std}"))

    else:
        print("\nDSP Features: unavailable")

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    if row["classification_label"] is not None:
        print("\nClassification:")

        print((f"  Label: {str(row['classification_label']).upper()}"))

        if row["classification_confidence"] is not None:
            print((f"  Confidence: {row['classification_confidence']:.3f}"))

        # --------------------------------------------------------------
        # SECOND CANDIDATE
        # --------------------------------------------------------------

        if row["classification_second_label"] is not None:
            second_confidence = row["classification_second_confidence"]

            second_text = str(row["classification_second_label"]).upper()

            if second_confidence is not None:
                second_text += f" ({float(second_confidence):.3f})"

            print((f"  Second candidate: {second_text}"))

        # --------------------------------------------------------------
        # DECISION MARGIN
        # --------------------------------------------------------------

        if row["classification_margin"] is not None:
            print((f"  Decision margin: {row['classification_margin']:.3f}"))

        # --------------------------------------------------------------
        # CLASSIFIER IDENTITY
        # --------------------------------------------------------------

        classifier_name = row["classifier_name"]

        classifier_version = row["classifier_version"]

        if classifier_name is not None:
            classifier_text = str(classifier_name)

            if classifier_version is not None:
                classifier_text += f" v{classifier_version}"

            print((f"  Classifier: {classifier_text}"))

        # --------------------------------------------------------------
        # CLASS SCORES
        # --------------------------------------------------------------

        scores = _load_json(row["classification_scores_json"])

        if isinstance(
            scores,
            dict,
        ):
            print("  Class scores:")

            try:
                ranked_scores = sorted(
                    scores.items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )

            except (
                TypeError,
                ValueError,
            ):
                ranked_scores = list(scores.items())

            for (
                label,
                score,
            ) in ranked_scores:
                score_text = _format_optional_float(
                    score,
                    precision=3,
                )

                print((f"    {str(label).upper():<14}{score_text}"))

        # --------------------------------------------------------------
        # CLASSIFICATION REASONS
        # --------------------------------------------------------------

        reasons = _load_json(row["classification_reasons_json"])

        if isinstance(
            reasons,
            (
                list,
                tuple,
            ),
        ):
            print("  Reasons:")

            for reason in reasons:
                print((f"    - {reason}"))

    else:
        print("\nClassification: unavailable")

    # ==================================================================
    # EVENT FILES
    # ==================================================================

    if row["event_directory"] is not None:
        print((f"\nEvent directory: {row['event_directory']}"))

    print("-" * 96)


# ======================================================================
# COMMAND HELP
# ======================================================================


def print_commands() -> None:
    """
    Print supported interactive CLI commands.
    """

    print("\nCommands:")

    print("  start")

    print("  stop")

    print("  ping")

    print("  status")

    print("  locate")

    print("  events")

    print("  event <db_id>")

    print("  help")

    print("  quit")


# ======================================================================
# COMMAND LOOP
# ======================================================================


async def command_loop(
    server: ReceiverServer,
) -> None:
    """
    Interactive receiver command loop.
    """

    print_commands()

    while True:
        # ==============================================================
        # READ COMMAND
        # ==============================================================

        try:
            raw_command = await asyncio.to_thread(
                input,
                "> ",
            )

        except EOFError:
            print("\nInput stream closed.")

            return

        raw_command = raw_command.strip()

        if not raw_command:
            continue

        parts = raw_command.split()

        command = parts[0].lower()

        # ==============================================================
        # START
        # ==============================================================

        if command == "start":
            if server.active_session_id is not None:
                print(
                    (
                        "A session is already active: "
                        f"{_format_session_id(server.active_session_id)}"
                    )
                )

                continue

            label = datetime.now().strftime("session_%Y%m%d_%H%M%S")

            try:
                session_id = await server.start_acquisition(label)

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print((f"START failed: {exc}"))

                continue

            print((f"Started session {_format_session_id(session_id)}"))

            continue

        # ==============================================================
        # STOP
        # ==============================================================

        if command == "stop":
            if server.active_session_id is None:
                print(("No acquisition session is active"))

                continue

            try:
                await server.stop_acquisition()

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print((f"STOP completed with an error: {exc}"))

                continue

            print("Acquisition stopped")

            continue

        # ==============================================================
        # PING
        # ==============================================================

        if command == "ping":
            try:
                await server.ping_all()

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print((f"PING failed: {exc}"))

            else:
                print("PING requested")

            continue

        # ==============================================================
        # STATUS
        # ==============================================================

        if command == "status":
            try:
                print_status(server)

            except Exception as exc:
                print((f"Status display failed: {exc}"))

            continue

        # ==============================================================
        # MANUAL LOCALIZATION
        # ==============================================================

        if command == "locate":
            try:
                # Reuse the exact LocalizationEngine already owned by
                # EventPipeline instead of constructing a second engine.
                result = server.events.localizer.locate_latest()

            except Exception as exc:
                print((f"Localization failed: {exc}"))

                continue

            if result is None:
                print(("Localization unavailable: not enough common audio yet"))

                continue

            print_localization_result(result)

            continue

        # ==============================================================
        # RECENT EVENTS
        # ==============================================================

        if command == "events":
            try:
                rows = server.events.database.recent_events(10)

            except Exception as exc:
                print((f"Unable to read events: {exc}"))

                continue

            if not rows:
                print(("No detected events stored yet"))

                continue

            for row in rows:
                try:
                    print_event_row(row)

                except Exception as exc:
                    print((f"Unable to display event row: {exc}"))

            continue

        # ==============================================================
        # ONE DATABASE EVENT
        # ==============================================================

        if command == "event":
            if len(parts) != 2:
                print(("Usage: event <database_id>"))

                continue

            try:
                database_id = int(parts[1])

            except ValueError:
                print(("Usage: event <database_id>"))

                continue

            if database_id <= 0:
                print(("Database event ID must be greater than 0"))

                continue

            try:
                row = server.events.database.get_event(database_id)

            except Exception as exc:
                print((f"Database lookup failed: {exc}"))

                continue

            if row is None:
                print((f"No event with database ID {database_id}"))

                continue

            try:
                print_event_details(row)

            except Exception as exc:
                print((f"Unable to display event: {exc}"))

            continue

        # ==============================================================
        # HELP
        # ==============================================================

        if command in {
            "help",
            "?",
        }:
            print_commands()

            continue

        # ==============================================================
        # QUIT
        # ==============================================================

        if command in {
            "quit",
            "exit",
            "q",
        }:
            return

        # ==============================================================
        # UNKNOWN COMMAND
        # ==============================================================

        print(("Unknown command. Type 'help' for commands."))


# ======================================================================
# STARTUP INFORMATION
# ======================================================================


def print_startup_info(
    server: ReceiverServer,
) -> None:
    """
    Display the configured receiver architecture at startup.
    """

    config = server.config

    print("\nWildlife Soundscape Receiver")

    print((f"Listening on {config.network.host}:{config.network.port}"))

    print((f"Expected nodes: {sorted(config.expected_nodes)}"))

    print(
        (
            "Audio: "
            f"{config.audio.sample_rate} Hz "
            f"| {config.audio.frames_per_block} samples/block "
            f"| {config.audio.block_duration_s * 1000.0:.2f} ms/block "
            "| PCM16 mono"
        )
    )

    print(
        (
            "Per-node stream buffer: "
            f"{config.audio.buffer_seconds:.1f} s "
            f"| {config.audio.blocks_in_buffer} blocks"
        )
    )

    print(
        (
            "Event detection: "
            + ("ENABLED" if (config.detection.enabled) else "DISABLED")
        )
    )

    print(
        (
            "Localization: "
            "GCC-PHAT/TDOA "
            f"| reference node="
            f"{config.localization.reference_node} "
            f"| window="
            f"{config.localization.window_samples} samples "
            f"| interpolation="
            f"{config.localization.interpolation}x"
        )
    )

    print(
        (
            "Environmental sound-speed correction: "
            + (
                "ENABLED"
                if (config.localization.use_environmental_speed)
                else "DISABLED"
            )
        )
    )

    print_classification_status(server)

    if config.classification.enabled:
        print(
            (
                "Model audio delivery: "
                + (
                    "ENABLED"
                    if (config.classification.provide_model_audio)
                    else "DISABLED"
                )
            )
        )

    print(
        (
            "Continuous WAV recording: "
            + ("ENABLED" if (config.audio.record_wav) else "DISABLED")
        )
    )

    print(
        (
            "Event WAV storage: "
            + ("ENABLED" if (config.persistence.save_event_wav) else "DISABLED")
        )
    )

    print((f"Database: {config.persistence.database_path}"))


# ======================================================================
# MAIN ASYNC
# ======================================================================


async def start_acquisition_when_ready(
    server: ReceiverServer,
    session_label: str,
    *,
    poll_interval_seconds: float = 0.25,
) -> int:
    """Wait for every configured node, then start one acquisition session."""

    previous_missing: tuple[int, ...] | None = None

    while True:
        missing = tuple(
            node_id
            for node_id in server.config.expected_nodes
            if (
                (connection := server.connections.get(node_id)) is None
                or not connection.state.connected
                or connection.writer.is_closing()
            )
        )

        if not missing:
            break

        if missing != previous_missing:
            print(f"Automatic acquisition is waiting for nodes: {list(missing)}")
            previous_missing = missing

        await asyncio.sleep(poll_interval_seconds)

    session_id = await server.start_acquisition(session_label)

    print(
        "Automatic acquisition started: "
        f"{_format_session_id(session_id)} "
        f"label={session_label!r}"
    )

    return session_id


async def main_async(
    *,
    auto_start: bool = False,
    session_label: str = "launcher",
) -> None:
    """
    Initialize and run the laptop receiver application.
    """

    configure_logging()

    server = ReceiverServer(CONFIG)

    status_task: asyncio.Task[None] | None = None

    try:
        # ==============================================================
        # TCP SERVER
        # ==============================================================

        await server.start()

        # ==============================================================
        # STARTUP INFORMATION
        # ==============================================================

        print_startup_info(server)

        # ==============================================================
        # BACKGROUND STATUS
        # ==============================================================

        status_task = asyncio.create_task(
            status_loop(server),
            name="receiver-status-loop",
        )

        # ==============================================================
        # OPTIONAL AUTOMATIC ACQUISITION
        # ==============================================================

        if auto_start:
            await start_acquisition_when_ready(
                server,
                session_label,
            )

        # ==============================================================
        # INTERACTIVE CLI
        # ==============================================================

        await command_loop(server)

    finally:
        # ==============================================================
        # STATUS TASK
        # ==============================================================

        if status_task is not None:
            status_task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await status_task

        # ==============================================================
        # SERVER / ACQUISITION SHUTDOWN
        # ==============================================================
        #
        # ReceiverServer.close() already owns acquisition shutdown.
        #
        # Calling stop_acquisition() separately here would duplicate that
        # lifecycle operation.
        # ==============================================================

        try:
            await server.close()

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(("Receiver shutdown encountered an error"))


# ======================================================================
# APPLICATION ENTRY POINT
# ======================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the receiver command-line parser."""

    parser = argparse.ArgumentParser(
        description="Wildlife Soundscape receiver",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help=("wait for all configured nodes and automatically start acquisition"),
    )
    parser.add_argument(
        "--session-label",
        default="launcher",
        help="label used by --auto-start (default: launcher)",
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> None:
    """
    Application entry point.
    """

    args = build_argument_parser().parse_args(argv)

    try:
        asyncio.run(
            main_async(
                auto_start=args.auto_start,
                session_label=args.session_label,
            )
        )

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
