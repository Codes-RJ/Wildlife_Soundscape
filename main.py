from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from config import CONFIG

from localization import (
    LocalizationEngine,
)

from server import (
    ReceiverServer,
)


# ======================================================================
# LOGGING
# ======================================================================


def configure_logging() -> None:
    """
    Configure application-wide console logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


# ======================================================================
# DISPLAY HELPERS
# ======================================================================


def _format_session_id(
    session_id: int | None,
) -> str:
    """
    Format a session ID consistently for CLI output.
    """

    if session_id is None:
        return "NONE"

    return f"0x{session_id:08X}"


def _load_json(
    value: Any,
) -> Any | None:
    """
    Safely decode JSON stored in SQLite.

    Returns None when the value is unavailable or malformed.
    """

    if value is None:
        return None

    try:
        return json.loads(
            value
        )

    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


# ======================================================================
# CLASSIFICATION STATUS
# ======================================================================


def print_classification_status(
    server: ReceiverServer,
    *,
    indent: str = "",
) -> None:
    """
    Display configured and active classification state.

    This distinguishes:

        configured backend
        active backend
        disabled classification
        heuristic fallback
    """

    classification_config = (
        CONFIG.classification
    )

    if not classification_config.enabled:

        print(
            f"{indent}Classification: DISABLED"
        )

        return

    print(
        (
            f"{indent}Classification: ENABLED "
            f"| configured={classification_config.backend}"
        )
    )

    backend = (
        server.events.classifier_backend
    )

    if backend is None:

        print(
            (
                f"{indent}                "
                "active backend=NONE"
            )
        )

        return

    print(
        (
            f"{indent}                "
            f"active={backend.name} "
            f"v{backend.version}"
        )
    )

    configured_name = (
        classification_config.backend
        .strip()
        .lower()
    )

    # Current operational backend is called heuristic_backend.
    # If some other backend was requested but the factory returned the
    # heuristic implementation, this indicates fallback.
    if (
        configured_name
        != "heuristic"
        and backend.name
        == "heuristic_backend"
    ):

        print(
            (
                f"{indent}                "
                f"fallback=heuristic "
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
    Print current receiver, node, DSP and classification state.
    """

    print(
        "\n"
        + "=" * 88
    )

    print(
        "WILDLIFE SOUNDSCAPE RECEIVER STATUS"
    )

    print(
        (
            "Active session: "
            f"{_format_session_id(server.active_session_id)}"
        )
    )

    print(
        (
            "Detected events this session: "
            f"{server.events.completed_events}"
        )
    )

    print_classification_status(
        server
    )

    print(
        "=" * 88
    )

    # ==================================================================
    # NODE STATUS
    # ==================================================================

    for node_id in sorted(
        CONFIG.expected_nodes
    ):

        connection = (
            server.connections.get(
                node_id
            )
        )

        if connection is None:

            print(
                f"Node {node_id}: OFFLINE"
            )

            continue

        snapshot = (
            connection.state.snapshot()
        )

        heartbeat = (
            snapshot.latest_heartbeat
        )

        role = (
            "MASTER"
            if (
                snapshot.hello
                and snapshot.hello.master_node
            )
            else "SLAVE"
        )

        print(
            (
                f"Node {node_id}: ONLINE {role} "
                f"| packets={snapshot.packets_received} "
                f"audio={snapshot.audio_packets_received} "
                f"crc={snapshot.crc_errors} "
                f"seq_gaps={snapshot.sequence_gaps} "
                f"sample_gaps={snapshot.sample_gaps} "
                f"clip={snapshot.clipped_packets} "
                f"clock_fault={snapshot.clock_fault_packets} "
                f"congest={snapshot.congested_packets}"
            )
        )

        if heartbeat is not None:

            print(
                (
                    "        "
                    f"RSSI={heartbeat.wifi_rssi} dBm "
                    f"heap={heartbeat.free_heap} "
                    f"I2Serr={heartbeat.i2s_errors} "
                    f"queue={heartbeat.audio_queue_depth} "
                    f"streaming={heartbeat.streaming}"
                )
            )

        environment = (
            snapshot.latest_environment
        )

        if environment is not None:

            print(
                (
                    "        "
                    f"ENV={environment.temperature_c:.1f} C, "
                    f"{environment.humidity_percent:.1f}% RH, "
                    f"{environment.pressure_hpa:.1f} hPa"
                )
            )

    # ==================================================================
    # ALIGNMENT
    # ==================================================================

    aligned = (
        server.streams.latest_aligned_blocks()
    )

    if aligned is not None:

        print(
            (
                "Aligned latest block @ sample "
                f"{aligned.target_sample_index}: "
                f"offsets={aligned.offsets}"
            )
        )

    else:

        print(
            "Aligned latest block: not available"
        )

    # ==================================================================
    # LATEST DSP
    # ==================================================================

    features = (
        server.events.last_features
    )

    if features is not None:

        print(
            "-" * 88
        )

        snr_text = (
            f"{features.snr_db:.2f} dB"
            if features.snr_db is not None
            else "N/A"
        )

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
                f"dominant={features.dominant_frequency_hz:.1f} Hz "
                f"| centroid={features.spectral_centroid_hz:.1f} Hz "
                f"| bandwidth={features.spectral_bandwidth_hz:.1f} Hz "
                f"| rolloff={features.spectral_rolloff_hz:.1f} Hz"
            )
        )

    # ==================================================================
    # LATEST CLASSIFICATION
    # ==================================================================

    classification = (
        server.events.last_classification
    )

    if classification is not None:

        print(
            "-" * 88
        )

        print(
            (
                "LATEST CLASSIFICATION "
                f"| {classification.label.value.upper()} "
                f"| confidence={classification.confidence:.3f} "
                f"| margin={classification.margin:.3f}"
            )
        )

        if (
            classification.second_label
            is not None
        ):

            print(
                (
                    "            "
                    "second="
                    f"{classification.second_label.value.upper()} "
                    f"({classification.second_confidence:.3f})"
                )
            )

        print(
            (
                "            "
                f"classifier={classification.classifier_name} "
                f"v{classification.classifier_version}"
            )
        )

        if classification.reasons:

            print(
                "            reasons:"
            )

            for reason in (
                classification.reasons
            ):

                print(
                    f"              - {reason}"
                )

    elif not server.events.classification_enabled:

        print(
            "-" * 88
        )

        print(
            "LATEST CLASSIFICATION | disabled"
        )

    print(
        "=" * 88
    )


# ======================================================================
# DATABASE EVENT SUMMARY
# ======================================================================


def print_event_row(
    row,
) -> None:
    """
    Print one compact database event summary.
    """

    position = (
        "n/a"
        if (
            row["x_m"] is None
            or row["y_m"] is None
        )
        else (
            f"({row['x_m']:.3f}, "
            f"{row['y_m']:.3f}) m"
        )
    )

    best_node = (
        "n/a"
        if row["best_node_id"] is None
        else str(
            row["best_node_id"]
        )
    )

    snr = (
        "n/a"
        if row["snr_db"] is None
        else (
            f"{row['snr_db']:.2f} dB"
        )
    )

    dominant = (
        "n/a"
        if row[
            "dominant_frequency_hz"
        ] is None
        else (
            f"{row['dominant_frequency_hz']:.1f} Hz"
        )
    )

    if (
        row["classification_label"]
        is None
    ):

        classification_text = (
            "unclassified"
        )

    else:

        confidence = (
            row["classification_confidence"]
        )

        classification_text = (
            row["classification_label"]
            .upper()
        )

        if confidence is not None:

            classification_text += (
                f" ({confidence:.3f})"
            )

    print(
        (
            f"DB#{row['id']} "
            f"| session={_format_session_id(row['session_id'])} "
            f"| samples={row['start_sample']}..{row['end_sample']} "
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
    """

    while True:

        await asyncio.sleep(
            CONFIG.print_status_every_s
        )

        print_status(
            server
        )


# ======================================================================
# MANUAL LOCALIZATION
# ======================================================================


def print_localization_result(
    result,
) -> None:
    """
    Print one manual localization result.
    """

    position = (
        result.position
    )

    print(
        (
            "Estimated source: "
            f"x={position.x:.4f} m, "
            f"y={position.y:.4f} m "
            f"| success={position.success}"
        )
    )

    print(
        (
            "Residual RMS: "
            f"{position.residual_rms_meters:.4f} "
            "m-equivalent"
        )
    )

    print(
        (
            "Speed of sound: "
            f"{result.speed_of_sound_mps:.2f} m/s"
        )
    )

    for measurement in (
        result.measurements
    ):

        state = (
            "OK"
            if measurement.valid
            else (
                "REJECT "
                f"({measurement.reason})"
            )
        )

        print(
            (
                f"  "
                f"{measurement.node_a}"
                f"->{measurement.node_b}: "
                f"{measurement.delay_samples:+.3f} samples "
                f"("
                f"{measurement.delay_seconds * 1e6:+.2f} us"
                f"), "
                f"peak_ratio={measurement.peak_ratio:.2f} "
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
    Print complete database information for one acoustic event.
    """

    print(
        "\n"
        + "-" * 88
    )

    print(
        "EVENT SUMMARY"
    )

    print(
        "-" * 88
    )

    print_event_row(
        row
    )

    # ==================================================================
    # CORE EVENT
    # ==================================================================

    print(
        "\nCore Event:"
    )

    print(
        (
            "  Detector event ID: "
            f"{row['detector_event_id']}"
        )
    )

    print(
        (
            "  Session ID: "
            f"{_format_session_id(row['session_id'])}"
        )
    )

    print(
        (
            "  Start sample: "
            f"{row['start_sample']}"
        )
    )

    print(
        (
            "  End sample: "
            f"{row['end_sample']}"
        )
    )

    print(
        (
            "  Trigger nodes: "
            f"{row['trigger_nodes']}"
        )
    )

    if row["peak_rms_dbfs"] is not None:

        print(
            (
                "  Peak RMS: "
                f"{row['peak_rms_dbfs']:.2f} dBFS"
            )
        )

    print(
        (
            "  Best DSP node: "
            f"{row['best_node_id']}"
        )
    )

    # ==================================================================
    # ENVIRONMENT
    # ==================================================================

    if (
        row["temperature_c"]
        is not None
    ):

        print(
            "\nEnvironment:"
        )

        print(
            (
                "  Temperature: "
                f"{row['temperature_c']:.2f} C"
            )
        )

        print(
            (
                "  Humidity: "
                f"{row['humidity_percent']:.2f}%"
            )
        )

        print(
            (
                "  Pressure: "
                f"{row['pressure_hpa']:.2f} hPa"
            )
        )

    # ==================================================================
    # LOCALIZATION
    # ==================================================================

    print(
        "\nLocalization:"
    )

    if (
        row["x_m"] is None
        or row["y_m"] is None
    ):

        print(
            "  Localization unavailable"
        )

    else:

        print(
            (
                "  Position: "
                f"x={row['x_m']:.4f} m, "
                f"y={row['y_m']:.4f} m"
            )
        )

        print(
            (
                "  Success: "
                f"{bool(row['localization_success'])}"
            )
        )

        if (
            row[
                "localization_residual_m"
            ]
            is not None
        ):

            print(
                (
                    "  Residual RMS: "
                    f"{row['localization_residual_m']:.5f} m"
                )
            )

        if (
            row["speed_of_sound_mps"]
            is not None
        ):

            print(
                (
                    "  Speed of sound: "
                    f"{row['speed_of_sound_mps']:.3f} m/s"
                )
            )

    # ==================================================================
    # DSP FEATURES
    # ==================================================================

    if (
        row["duration_s"]
        is not None
    ):

        print(
            "\nDSP Features:"
        )

        print(
            (
                "  Source node: "
                f"{row['feature_source_node_id']}"
            )
        )

        print(
            (
                "  Duration: "
                f"{row['duration_s']:.3f} s"
            )
        )

        print(
            (
                "  RMS: "
                f"{row['rms']:.6f}"
            )
        )

        print(
            (
                "  Peak amplitude: "
                f"{row['peak_amplitude']:.6f}"
            )
        )

        print(
            (
                "  Crest factor: "
                f"{row['crest_factor']:.3f}"
            )
        )

        print(
            (
                "  Zero-crossing rate: "
                f"{row['zero_crossing_rate']:.6f}"
            )
        )

        print(
            (
                "  Dominant frequency: "
                f"{row['dominant_frequency_hz']:.2f} Hz"
            )
        )

        print(
            (
                "  Spectral centroid: "
                f"{row['spectral_centroid_hz']:.2f} Hz"
            )
        )

        print(
            (
                "  Spectral bandwidth: "
                f"{row['spectral_bandwidth_hz']:.2f} Hz"
            )
        )

        print(
            (
                "  Spectral rolloff: "
                f"{row['spectral_rolloff_hz']:.2f} Hz"
            )
        )

        print(
            (
                "  Spectral flatness: "
                f"{row['spectral_flatness']:.6f}"
            )
        )

        print(
            (
                "  Spectral flux: "
                f"{row['spectral_flux']:.6f}"
            )
        )

        print(
            (
                "  SNR: "
                + (
                    f"{row['snr_db']:.2f} dB"
                    if row["snr_db"] is not None
                    else "N/A"
                )
            )
        )

        mfcc_mean = (
            _load_json(
                row["mfcc_mean_json"]
            )
        )

        mfcc_std = (
            _load_json(
                row["mfcc_std_json"]
            )
        )

        if mfcc_mean is not None:

            print(
                (
                    "  MFCC mean: "
                    f"{mfcc_mean}"
                )
            )

        if mfcc_std is not None:

            print(
                (
                    "  MFCC std: "
                    f"{mfcc_std}"
                )
            )

    else:

        print(
            "\nDSP Features: unavailable"
        )

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    if (
        row["classification_label"]
        is not None
    ):

        print(
            "\nClassification:"
        )

        print(
            (
                "  Label: "
                f"{row['classification_label'].upper()}"
            )
        )

        if (
            row[
                "classification_confidence"
            ]
            is not None
        ):

            print(
                (
                    "  Confidence: "
                    f"{row['classification_confidence']:.3f}"
                )
            )

        if (
            row[
                "classification_second_label"
            ]
            is not None
        ):

            second_confidence = (
                row[
                    "classification_second_confidence"
                ]
            )

            second_text = (
                row[
                    "classification_second_label"
                ]
                .upper()
            )

            if (
                second_confidence
                is not None
            ):

                second_text += (
                    f" ({second_confidence:.3f})"
                )

            print(
                (
                    "  Second candidate: "
                    f"{second_text}"
                )
            )

        if (
            row["classification_margin"]
            is not None
        ):

            print(
                (
                    "  Decision margin: "
                    f"{row['classification_margin']:.3f}"
                )
            )

        classifier_name = (
            row["classifier_name"]
        )

        classifier_version = (
            row["classifier_version"]
        )

        if classifier_name is not None:

            classifier_text = (
                classifier_name
            )

            if (
                classifier_version
                is not None
            ):

                classifier_text += (
                    f" v{classifier_version}"
                )

            print(
                (
                    "  Classifier: "
                    f"{classifier_text}"
                )
            )

        # --------------------------------------------------------------
        # CLASS SCORES
        # --------------------------------------------------------------

        scores = (
            _load_json(
                row[
                    "classification_scores_json"
                ]
            )
        )

        if isinstance(
            scores,
            dict,
        ):

            print(
                "  Class scores:"
            )

            try:

                ranked_scores = sorted(
                    scores.items(),
                    key=lambda item:
                        float(
                            item[1]
                        ),
                    reverse=True,
                )

            except (
                TypeError,
                ValueError,
            ):

                ranked_scores = (
                    list(
                        scores.items()
                    )
                )

            for (
                label,
                score,
            ) in ranked_scores:

                try:

                    score_text = (
                        f"{float(score):.3f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    score_text = str(
                        score
                    )

                print(
                    (
                        f"    "
                        f"{str(label).upper():<14}"
                        f"{score_text}"
                    )
                )

        # --------------------------------------------------------------
        # CLASSIFICATION REASONS
        # --------------------------------------------------------------

        reasons = (
            _load_json(
                row[
                    "classification_reasons_json"
                ]
            )
        )

        if isinstance(
            reasons,
            list,
        ):

            print(
                "  Reasons:"
            )

            for reason in reasons:

                print(
                    f"    - {reason}"
                )

    else:

        print(
            "\nClassification: unavailable"
        )

    # ==================================================================
    # EVENT FILES
    # ==================================================================

    if (
        row["event_directory"]
        is not None
    ):

        print(
            (
                "\nEvent directory: "
                f"{row['event_directory']}"
            )
        )

    print(
        "-" * 88
    )


# ======================================================================
# COMMAND HELP
# ======================================================================


def print_commands() -> None:
    """
    Print supported interactive CLI commands.
    """

    print(
        "\nCommands:"
    )

    print(
        "  start"
    )

    print(
        "  stop"
    )

    print(
        "  ping"
    )

    print(
        "  status"
    )

    print(
        "  locate"
    )

    print(
        "  events"
    )

    print(
        "  event <db_id>"
    )

    print(
        "  quit"
    )


# ======================================================================
# COMMAND LOOP
# ======================================================================


async def command_loop(
    server: ReceiverServer,
) -> None:
    """
    Interactive receiver command loop.
    """

    localizer = LocalizationEngine(
        server.streams,
        CONFIG.localization,
    )

    print_commands()

    while True:

        raw_command = (
            await asyncio.to_thread(
                input,
                "> ",
            )
        ).strip()

        if not raw_command:

            continue

        parts = (
            raw_command.split()
        )

        command = (
            parts[0]
            .lower()
        )

        # ==============================================================
        # START
        # ==============================================================

        if command == "start":

            if (
                server.active_session_id
                is not None
            ):

                print(
                    (
                        "A session is already active: "
                        f"{_format_session_id(server.active_session_id)}"
                    )
                )

                continue

            label = (
                datetime.now()
                .strftime(
                    "session_%Y%m%d_%H%M%S"
                )
            )

            try:

                session_id = (
                    await server.start_acquisition(
                        label
                    )
                )

                print(
                    (
                        "Started session "
                        f"{_format_session_id(session_id)}"
                    )
                )

            except RuntimeError as exc:

                print(
                    f"START failed: {exc}"
                )

        # ==============================================================
        # STOP
        # ==============================================================

        elif command == "stop":

            if (
                server.active_session_id
                is None
            ):

                print(
                    "No acquisition session is active"
                )

                continue

            await server.stop_acquisition()

            print(
                "Acquisition stopped"
            )

        # ==============================================================
        # PING
        # ==============================================================

        elif command == "ping":

            await server.ping_all()

        # ==============================================================
        # STATUS
        # ==============================================================

        elif command == "status":

            print_status(
                server
            )

        # ==============================================================
        # MANUAL LOCALIZATION
        # ==============================================================

        elif command == "locate":

            try:

                result = (
                    localizer.locate_latest()
                )

            except Exception as exc:

                print(
                    (
                        "Localization failed: "
                        f"{exc}"
                    )
                )

                continue

            if result is None:

                print(
                    (
                        "Localization unavailable: "
                        "not enough common audio yet"
                    )
                )

                continue

            print_localization_result(
                result
            )

        # ==============================================================
        # RECENT EVENTS
        # ==============================================================

        elif command == "events":

            rows = (
                server.events.database
                .recent_events(
                    10
                )
            )

            if not rows:

                print(
                    "No detected events stored yet"
                )

                continue

            for row in rows:

                print_event_row(
                    row
                )

        # ==============================================================
        # ONE EVENT
        # ==============================================================

        elif command == "event":

            if len(parts) != 2:

                print(
                    "Usage: event <database_id>"
                )

                continue

            try:

                database_id = int(
                    parts[1]
                )

            except ValueError:

                print(
                    "Usage: event <database_id>"
                )

                continue

            if database_id <= 0:

                print(
                    "Database event ID must be greater than 0"
                )

                continue

            row = (
                server.events.database
                .get_event(
                    database_id
                )
            )

            if row is None:

                print(
                    (
                        "No event with database ID "
                        f"{database_id}"
                    )
                )

                continue

            print_event_details(
                row
            )

        # ==============================================================
        # QUIT
        # ==============================================================

        elif command in {
            "quit",
            "exit",
            "q",
        }:

            return

        # ==============================================================
        # UNKNOWN COMMAND
        # ==============================================================

        else:

            print(
                (
                    "Unknown command. Use: "
                    "start | stop | ping | status | locate | "
                    "events | event <db_id> | quit"
                )
            )


# ======================================================================
# STARTUP INFORMATION
# ======================================================================


def print_startup_info(
    server: ReceiverServer,
) -> None:
    """
    Display configured receiver architecture at startup.
    """

    print(
        "\nWildlife Soundscape Receiver"
    )

    print(
        (
            "Listening on "
            f"{CONFIG.network.host}:"
            f"{CONFIG.network.port}"
        )
    )

    print(
        (
            "Expected nodes: "
            f"{sorted(CONFIG.expected_nodes)}"
        )
    )

    print(
        (
            "Audio: "
            f"{CONFIG.audio.sample_rate} Hz "
            f"| {CONFIG.audio.frames_per_block} samples/block "
            "| PCM16 mono"
        )
    )

    print(
        (
            "Event detection: "
            + (
                "ENABLED"
                if CONFIG.detection.enabled
                else "DISABLED"
            )
        )
    )

    print(
        (
            "Localization: "
            f"GCC-PHAT/TDOA "
            f"| reference node="
            f"{CONFIG.localization.reference_node}"
        )
    )

    print_classification_status(
        server
    )

    if (
        CONFIG.classification.enabled
    ):

        print(
            (
                "Model audio delivery: "
                + (
                    "ENABLED"
                    if (
                        CONFIG.classification
                        .provide_model_audio
                    )
                    else "DISABLED"
                )
            )
        )


# ======================================================================
# MAIN ASYNC
# ======================================================================


async def main_async() -> None:
    """
    Initialize and run the laptop-side receiver application.
    """

    configure_logging()

    server = ReceiverServer(
        CONFIG
    )

    await server.start()

    print_startup_info(
        server
    )

    status_task = (
        asyncio.create_task(
            status_loop(
                server
            ),
            name="receiver-status-loop",
        )
    )

    try:

        await command_loop(
            server
        )

    finally:

        status_task.cancel()

        try:

            await status_task

        except asyncio.CancelledError:

            pass

        # --------------------------------------------------------------
        # Stop acquisition only when a session remains active.
        # --------------------------------------------------------------

        if (
            server.active_session_id
            is not None
        ):

            try:

                await server.stop_acquisition()

            except Exception:

                logging.getLogger(
                    __name__
                ).exception(
                    "Failed to stop acquisition during shutdown"
                )

        await server.close()


# ======================================================================
# ENTRY POINT
# ======================================================================


def main() -> None:
    """
    Application entry point.
    """

    try:

        asyncio.run(
            main_async()
        )

    except KeyboardInterrupt:

        pass


if __name__ == "__main__":

    main()