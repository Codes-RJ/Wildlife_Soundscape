from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import time

from dataclasses import (
    dataclass,
    field,
)

import numpy as np

from wildlife_soundscape.core.config import (
    CONFIG,
)

from wildlife_soundscape.core.protocol import (
    CONTROL_SIZE,
    ControlCommand,
    EnvironmentPayload,
    HeartbeatPayload,
    HelloPayload,
    PacketFlags,
    PacketType,
    SyncPayload,
    build_packet,
    pack_environment,
    pack_hello,
    pack_master_heartbeat,
    pack_slave_heartbeat,
    pack_sync,
    unpack_control,
)


# ======================================================================
# PROJECT AUDIO SETTINGS
# ======================================================================


SAMPLE_RATE = (
    CONFIG.audio.sample_rate
)

FRAMES_PER_BLOCK = (
    CONFIG.audio.frames_per_block
)

BLOCK_PERIOD_S = (
    FRAMES_PER_BLOCK
    / SAMPLE_RATE
)


# ======================================================================
# SIMULATED ENVIRONMENT
# ======================================================================


SIM_TEMPERATURE_C = (
    27.5
)

SIM_HUMIDITY_PERCENT = (
    58.0
)

SIM_PRESSURE_HPA = (
    1007.2
)


# ======================================================================
# SIMULATION CONSTANTS
# ======================================================================


ENVIRONMENT_PERIOD_SAMPLES = (
    SAMPLE_RATE * 2
)

EVENT_FIRST_SAMPLE = (
    SAMPLE_RATE * 2
)

EVENT_PERIOD_SAMPLES = (
    SAMPLE_RATE * 5
)

UINT32_MASK = (
    0xFFFFFFFF
)


# ======================================================================
# SPEED OF SOUND
# ======================================================================


def calculate_simulated_speed_of_sound() -> float:
    """
    Estimate humid-air speed of sound from the same environmental
    values transmitted by the simulated Node 1 BME280.

    The simulator deliberately uses environmental conditions that are
    also visible to the laptop pipeline so localization can exercise
    dynamic sound-speed correction.
    """

    temperature_c = (
        SIM_TEMPERATURE_C
    )

    humidity_percent = (
        SIM_HUMIDITY_PERCENT
    )

    pressure_hpa = (
        SIM_PRESSURE_HPA
    )

    # --------------------------------------------------------------
    # BUCK SATURATION VAPOUR PRESSURE APPROXIMATION
    # --------------------------------------------------------------

    saturation_pressure_hpa = (
        6.1121
        * math.exp(
            (
                18.678
                - temperature_c
                / 234.5
            )
            * (
                temperature_c
                / (
                    257.14
                    + temperature_c
                )
            )
        )
    )

    vapour_pressure_hpa = (
        humidity_percent
        / 100.0
        * saturation_pressure_hpa
    )

    # --------------------------------------------------------------
    # HUMID-AIR APPROXIMATION
    # --------------------------------------------------------------

    speed = (
        331.3
        * math.sqrt(
            1.0
            + temperature_c
            / 273.15
        )
        * (
            1.0
            + 0.16
            * (
                vapour_pressure_hpa
                / pressure_hpa
            )
        )
    )

    return float(
        speed
    )


# ======================================================================
# SHARED PHYSICAL SIMULATION
# ======================================================================


@dataclass(
    slots=True,
)
class SharedSimulation:
    """
    Physical state shared by all three virtual acoustic nodes.

    This represents the common world in which the microphones exist.

    It does NOT represent TCP synchronization.

    All microphones observe the same synthetic source but with:

        different propagation delay
        different distance attenuation
        independent microphone noise
    """

    # ------------------------------------------------------------------
    # KNOWN TEST SOURCE POSITION
    # ------------------------------------------------------------------

    source_xy: tuple[
        float,
        float,
    ] = (
        0.45,
        0.35,
    )

    # ------------------------------------------------------------------
    # MICROPHONE POSITIONS
    # ------------------------------------------------------------------

    node_xy: dict[
        int,
        tuple[
            float,
            float,
        ],
    ] = field(
        default_factory=lambda:
            dict(
                CONFIG.localization.node_positions
            )
    )

    # ------------------------------------------------------------------
    # ENVIRONMENT-DEPENDENT SPEED OF SOUND
    # ------------------------------------------------------------------

    speed_of_sound: float = field(
        default_factory=
            calculate_simulated_speed_of_sound
    )

    # ------------------------------------------------------------------
    # NODE ARMING STATE
    # ------------------------------------------------------------------

    armed_session: dict[
        int,
        int,
    ] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # MASTER CLOCK STATE
    # ------------------------------------------------------------------

    clock_session_id: (
        int
        | None
    ) = None

    clock_event: asyncio.Event = field(
        default_factory=
            asyncio.Event
    )

    # ==================================================================
    # GEOMETRY
    # ==================================================================

    def distance_to_source(
        self,
        node_id: int,
    ) -> float:
        """
        Euclidean distance from source to one microphone.
        """

        if (
            node_id
            not in self.node_xy
        ):

            raise KeyError(
                (
                    "No simulated position for "
                    f"node {node_id}"
                )
            )

        node_x, node_y = (
            self.node_xy[
                node_id
            ]
        )

        source_x, source_y = (
            self.source_xy
        )

        return float(
            math.hypot(
                source_x
                - node_x,

                source_y
                - node_y,
            )
        )

    # ==================================================================
    # PROPAGATION DELAY
    # ==================================================================

    def delay_samples(
        self,
        node_id: int,
    ) -> int:
        """
        Acoustic propagation delay from source to microphone.

        Integer-sample propagation is sufficient for the current
        simulator. GCC-PHAT may still estimate fractional delay from
        correlation interpolation.
        """

        distance = (
            self.distance_to_source(
                node_id
            )
        )

        delay_seconds = (
            distance
            / self.speed_of_sound
        )

        return int(
            round(
                delay_seconds
                * SAMPLE_RATE
            )
        )

    # ==================================================================
    # DISTANCE ATTENUATION
    # ==================================================================

    def amplitude_gain(
        self,
        node_id: int,
    ) -> float:
        """
        Produce simple distance-dependent attenuation.

        This is intentionally not claimed as a complete free-field SPL
        propagation model.

        Its purpose is to make node SNR differ so the best-microphone
        selection logic is exercised.
        """

        distance = (
            self.distance_to_source(
                node_id
            )
        )

        return float(
            1.0
            / (
                0.15
                + distance
            )
        )

    # ==================================================================
    # ARM NODE
    # ==================================================================

    def arm_node(
        self,
        node_id: int,
        session_id: int,
    ) -> None:
        """
        Mark one virtual ESP32 as armed for an acquisition session.
        """

        self.armed_session[
            int(
                node_id
            )
        ] = int(
            session_id
        )

    # ==================================================================
    # DISARM NODE
    # ==================================================================

    def disarm_node(
        self,
        node_id: int,
        session_id: int,
    ) -> None:
        """
        Remove one node from the armed set only when the session matches.
        """

        current = (
            self.armed_session.get(
                int(
                    node_id
                )
            )
        )

        if (
            current
            == int(
                session_id
            )
        ):

            self.armed_session.pop(
                int(
                    node_id
                ),
                None,
            )

    # ==================================================================
    # ACTIVATE MASTER CLOCK
    # ==================================================================

    def activate_clock(
        self,
        session_id: int,
    ) -> bool:
        """
        Activate the simulated shared BCLK/WS timeline.

        Returns
        -------
        bool
            True when both slave nodes were already armed for the same
            session.

        The simulator still activates the clock on an ordering violation
        so such violations can be observed rather than deadlocking the
        simulator.
        """

        session_id = int(
            session_id
        )

        slaves_ready = all(
            (
                self.armed_session.get(
                    node_id
                )
                == session_id
            )
            for node_id
            in (
                2,
                3,
            )
        )

        self.clock_session_id = (
            session_id
        )

        self.clock_event.set()

        return slaves_ready

    # ==================================================================
    # DEACTIVATE MASTER CLOCK
    # ==================================================================

    def deactivate_clock(
        self,
        session_id: int,
    ) -> None:
        """
        Stop the shared simulated audio clock for the matching session.
        """

        if (
            self.clock_session_id
            != int(
                session_id
            )
        ):

            return

        self.clock_session_id = (
            None
        )

        self.clock_event.clear()

    # ==================================================================
    # NODE STREAMING ELIGIBILITY
    # ==================================================================

    def node_can_stream(
        self,
        node_id: int,
        session_id: int,
    ) -> bool:
        """
        Whether one virtual ESP32 currently has valid shared clock.

        A node streams only when:

            node is armed for session
            AND
            Node 1 clock is active for the same session
        """

        session_id = int(
            session_id
        )

        return (
            self.armed_session.get(
                int(
                    node_id
                )
            )
            == session_id

            and self.clock_session_id
            == session_id
        )


# ======================================================================
# FAKE ESP32 NODE
# ======================================================================


class FakeNode:
    """
    Protocol-compatible software representation of one ESP32 node.

    Node 1
        simulated I2S master
        simulated BME280 owner

    Node 2 / Node 3
        simulated I2S slaves

    The implementation intentionally mirrors the actual system
    architecture rather than merely flooding the receiver with PCM.
    """

    def __init__(
        self,
        node_id: int,
        host: str,
        port: int,
        shared: SharedSimulation,
    ) -> None:

        self.node_id = int(
            node_id
        )

        if (
            self.node_id
            not in {
                1,
                2,
                3,
            }
        ):

            raise ValueError(
                (
                    "Simulator node_id must "
                    "be 1, 2 or 3"
                )
            )

        self.host = str(
            host
        )

        self.port = int(
            port
        )

        self.shared = (
            shared
        )

        # --------------------------------------------------------------
        # PROTOCOL STATE
        # --------------------------------------------------------------

        self.sequence = (
            0
        )

        self.session_id = (
            0
        )

        self.sample_index = (
            0
        )

        # --------------------------------------------------------------
        # ACQUISITION STATE
        # --------------------------------------------------------------

        self.armed = (
            False
        )

        self._sync_sent_session: (
            int
            | None
        ) = None

        self._next_block_deadline: (
            float
            | None
        ) = None

        # --------------------------------------------------------------
        # TELEMETRY STATE
        # --------------------------------------------------------------

        self.next_environment_sample = (
            0
        )

        self.start_time = (
            time.monotonic()
        )

        # --------------------------------------------------------------
        # TCP WRITE SERIALIZATION
        # --------------------------------------------------------------
        #
        # AUDIO, heartbeat, environment and SYNC originate from separate
        # asyncio tasks. A common lock guarantees packet sequence order
        # also matches byte-stream write order.
        # --------------------------------------------------------------

        self._write_lock = (
            asyncio.Lock()
        )

        # --------------------------------------------------------------
        # INDEPENDENT NODE NOISE
        # --------------------------------------------------------------

        self.rng = (
            np.random.default_rng(
                1000
                + self.node_id
            )
        )

        # --------------------------------------------------------------
        # REUSABLE SHARED-TYPE ACOUSTIC CALL
        # --------------------------------------------------------------

        self.event_waveform = (
            self._build_event_waveform()
        )

    # ==================================================================
    # ROLE
    # ==================================================================

    @property
    def master_node(
        self,
    ) -> bool:

        return (
            self.node_id
            == 1
        )

    # ==================================================================
    # STREAMING STATE
    # ==================================================================

    @property
    def streaming(
        self,
    ) -> bool:
        """
        Whether this node currently has a valid active shared clock.
        """

        if not self.armed:

            return False

        if (
            self.session_id
            == 0
        ):

            return False

        return (
            self.shared.node_can_stream(
                self.node_id,
                self.session_id,
            )
        )

    # ==================================================================
    # SYNTHETIC WILDLIFE-LIKE CALL
    # ==================================================================

    def _build_event_waveform(
        self,
    ) -> np.ndarray:
        """
        Build a synthetic tonal/harmonic chirping event.

        Duration
            approximately 0.55 s

        Fundamental
            approximately 1.2 kHz -> 4.2 kHz

        Components
            chirp
            second harmonic
            amplitude modulation
            smooth onset/offset envelope

        This waveform is NOT intended to model or identify any specific
        wildlife species.

        It exists only to exercise:

            adaptive event detection
            DSP feature extraction
            broad heuristic classification
            microphone quality selection
            GCC-PHAT
            TDOA localization
        """

        duration_s = (
            0.55
        )

        sample_count = int(
            round(
                duration_s
                * SAMPLE_RATE
            )
        )

        t = (
            np.arange(
                sample_count,
                dtype=np.float64,
            )
            / SAMPLE_RATE
        )

        # --------------------------------------------------------------
        # LINEAR CHIRP
        # --------------------------------------------------------------

        frequency_start_hz = (
            1200.0
        )

        frequency_end_hz = (
            4200.0
        )

        chirp_rate = (
            (
                frequency_end_hz
                - frequency_start_hz
            )
            / duration_s
        )

        phase = (
            2.0
            * np.pi
            * (
                frequency_start_hz
                * t
                + 0.5
                * chirp_rate
                * t
                * t
            )
        )

        # --------------------------------------------------------------
        # FUNDAMENTAL
        # --------------------------------------------------------------

        fundamental = np.sin(
            phase
        )

        # --------------------------------------------------------------
        # SECOND HARMONIC
        # --------------------------------------------------------------

        harmonic = (
            0.35
            * np.sin(
                2.0
                * phase
                + 0.4
            )
        )

        # --------------------------------------------------------------
        # AMPLITUDE MODULATION
        # --------------------------------------------------------------

        modulation = (
            0.78
            + 0.22
            * np.sin(
                2.0
                * np.pi
                * 7.0
                * t
            )
        )

        # --------------------------------------------------------------
        # SMOOTH EVENT ENVELOPE
        # --------------------------------------------------------------

        envelope_phase = (
            np.arange(
                sample_count,
                dtype=np.float64,
            )
            / max(
                1,
                sample_count - 1,
            )
        )

        envelope = (
            np.sin(
                np.pi
                * envelope_phase
            )
            ** 2
        )

        waveform = (
            (
                fundamental
                + harmonic
            )
            * modulation
            * envelope
        )

        # --------------------------------------------------------------
        # BASE DIGITAL AMPLITUDE
        # --------------------------------------------------------------

        waveform *= (
            7000.0
        )

        return np.asarray(
            waveform,
            dtype=np.float64,
        )

    # ==================================================================
    # PACKET SEQUENCE
    # ==================================================================

    def next_sequence(
        self,
    ) -> int:
        """
        Return current uint32 sequence and advance with wraparound.
        """

        value = int(
            self.sequence
        )

        self.sequence = (
            self.sequence
            + 1
        ) & UINT32_MASK

        return value

    # ==================================================================
    # LOCAL MICROSECOND COUNTER
    # ==================================================================

    def local_micros(
        self,
    ) -> int:
        """
        Produce ESP32-like wrapping local microsecond diagnostics.

        This value is intentionally unrelated to cross-node TDOA.
        """

        return (
            int(
                (
                    time.monotonic()
                    - self.start_time
                )
                * 1_000_000
            )
            & UINT32_MASK
        )

    # ==================================================================
    # PACKET BUILDING
    # ==================================================================

    def packet(
        self,
        packet_type: PacketType,
        payload: bytes = b"",
        *,
        sample_index: int | None = None,
        flags: int = 0,
    ) -> bytes:
        """
        Build one Protocol-v4 packet.
        """

        effective_sample_index = (
            self.sample_index
            if sample_index is None
            else int(
                sample_index
            )
        )

        return build_packet(
            node_id=
                self.node_id,

            packet_type=
                packet_type,

            sequence=
                self.next_sequence(),

            session_id=
                self.session_id,

            sample_index=
                effective_sample_index,

            local_micros=
                self.local_micros(),

            i2s_error_count=
                0,

            flags=
                int(
                    flags
                ),

            payload=
                payload,
        )

    # ==================================================================
    # SERIALIZED PACKET SEND
    # ==================================================================

    async def _send_packet(
        self,
        writer: asyncio.StreamWriter,
        packet_type: PacketType,
        payload: bytes = b"",
        *,
        sample_index: int | None = None,
        flags: int = 0,
    ) -> None:
        """
        Serialize creation + transmission of one complete packet.

        Keeping packet creation inside the lock guarantees sequence
        numbers follow the same order as TCP writes.
        """

        async with (
            self._write_lock
        ):

            if (
                writer.is_closing()
            ):

                raise ConnectionError(
                    (
                        f"simulated node {self.node_id} "
                        "writer is closing"
                    )
                )

            frame = (
                self.packet(
                    packet_type,
                    payload,
                    sample_index=
                        sample_index,
                    flags=
                        flags,
                )
            )

            writer.write(
                frame
            )

            await writer.drain()

    # ==================================================================
    # HELLO PAYLOAD
    # ==================================================================

    def hello_payload(
        self,
    ) -> bytes:
        """
        Build Protocol-v4 node HELLO.
        """

        return pack_hello(
            HelloPayload(
                sample_rate=
                    SAMPLE_RATE,

                frames_per_packet=
                    FRAMES_PER_BLOCK,

                bits_per_sample=
                    16,

                channels=
                    1,

                master_node=
                    self.master_node,

                sync_tolerance_samples=
                    CONFIG.audio.sync_tolerance_samples,

                firmware=
                    "sim-proto-v4",
            )
        )

    # ==================================================================
    # AUDIO GENERATION
    # ==================================================================

    def generate_audio(
        self,
        start_sample: int,
    ) -> tuple[
        np.ndarray,
        bool,
    ]:
        """
        Generate one PCM16 block.

        Returns
        -------
        samples
            Mono PCM16 audio.

        clipped
            Whether the un-clipped floating waveform exceeded the PCM16
            numeric range.
        """

        start_sample = int(
            start_sample
        )

        sample_count = (
            FRAMES_PER_BLOCK
        )

        # ==============================================================
        # INDEPENDENT MICROPHONE NOISE
        # ==============================================================

        output = (
            self.rng.normal(
                0.0,
                75.0,
                sample_count,
            )
            .astype(
                np.float64
            )
        )

        # ==============================================================
        # PHYSICAL PROPAGATION
        # ==============================================================

        propagation_delay = (
            self.shared.delay_samples(
                self.node_id
            )
        )

        gain = (
            self.shared.amplitude_gain(
                self.node_id
            )
        )

        event_length = int(
            self.event_waveform.size
        )

        # ==============================================================
        # MICROPHONE TIME -> SOURCE EMISSION TIME
        # ==============================================================

        first_emission_timeline_sample = (
            start_sample
            - propagation_delay
        )

        last_emission_timeline_sample = (
            start_sample
            + sample_count
            - 1
            - propagation_delay
        )

        # ==============================================================
        # FIND POSSIBLE PERIODIC EVENTS OVERLAPPING THIS BLOCK
        # ==============================================================

        occurrence_start = (
            math.floor(
                (
                    first_emission_timeline_sample
                    - EVENT_FIRST_SAMPLE
                )
                / EVENT_PERIOD_SAMPLES
            )
            - 1
        )

        occurrence_end = (
            math.floor(
                (
                    last_emission_timeline_sample
                    - EVENT_FIRST_SAMPLE
                )
                / EVENT_PERIOD_SAMPLES
            )
            + 1
        )

        # ==============================================================
        # ADD EVENT PORTIONS
        # ==============================================================

        for occurrence in range(
            occurrence_start,
            occurrence_end + 1,
        ):

            emission_sample = (
                EVENT_FIRST_SAMPLE
                + occurrence
                * EVENT_PERIOD_SAMPLES
            )

            # No source event before acquisition sample zero.
            if (
                emission_sample
                < 0
            ):

                continue

            arrival_sample = (
                emission_sample
                + propagation_delay
            )

            overlap_start = max(
                start_sample,
                arrival_sample,
            )

            overlap_end = min(
                start_sample
                + sample_count,

                arrival_sample
                + event_length,
            )

            if (
                overlap_start
                >= overlap_end
            ):

                continue

            destination_start = (
                overlap_start
                - start_sample
            )

            source_start = (
                overlap_start
                - arrival_sample
            )

            count = (
                overlap_end
                - overlap_start
            )

            output[
                destination_start:
                destination_start + count
            ] += (
                self.event_waveform[
                    source_start:
                    source_start + count
                ]
                * gain
            )

        # ==============================================================
        # CLIPPING FLAG — DETECT BEFORE NUMERIC CLIP
        # ==============================================================

        clipped = bool(
            np.any(
                output
                > 32767.0
            )
            or np.any(
                output
                < -32768.0
            )
        )

        # ==============================================================
        # PCM16
        # ==============================================================

        pcm = (
            np.clip(
                output,
                -32768.0,
                32767.0,
            )
            .astype(
                "<i2"
            )
        )

        return (
            pcm,
            clipped,
        )

    # ==================================================================
    # CONNECTION
    # ==================================================================

    async def run(
        self,
    ) -> None:
        """
        Connect one fake ESP32 to the laptop receiver.
        """

        reader, writer = (
            await asyncio.open_connection(
                self.host,
                self.port,
            )
        )

        print(
            (
                f"[SIM] Node {self.node_id} connected "
                f"to {self.host}:{self.port}"
            )
        )

        # ==============================================================
        # FIRST PACKET MUST BE HELLO
        # ==============================================================

        await self._send_packet(
            writer,
            PacketType.HELLO,
            self.hello_payload(),
            sample_index=0,
        )

        # ==============================================================
        # PARALLEL NODE TASKS
        # ==============================================================

        tasks = (
            asyncio.create_task(
                self._command_loop(
                    reader,
                    writer,
                ),
                name=
                    f"sim-node-{self.node_id}-commands",
            ),

            asyncio.create_task(
                self._stream_loop(
                    writer
                ),
                name=
                    f"sim-node-{self.node_id}-audio",
            ),

            asyncio.create_task(
                self._heartbeat_loop(
                    writer
                ),
                name=
                    f"sim-node-{self.node_id}-heartbeat",
            ),
        )

        try:

            done, _pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:

                with contextlib.suppress(
                    asyncio.IncompleteReadError,
                    ConnectionError,
                    OSError,
                ):

                    task.result()

        finally:

            for task in (
                tasks
            ):

                task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            writer.close()

            with contextlib.suppress(
                ConnectionError,
                OSError,
            ):

                await asyncio.wait_for(
                    writer.wait_closed(),
                    timeout=2.0,
                )

            print(
                (
                    f"[SIM] Node {self.node_id} disconnected"
                )
            )

    # ==================================================================
    # START SESSION
    # ==================================================================

    async def _arm_session(
        self,
        session_id: int,
    ) -> None:
        """
        Reset simulated ESP32 acquisition state for one START command.

        State mutation is serialized with packet writes so an idle
        heartbeat cannot be half-transmitted while the protocol
        sequence/session counters are being reset.
        """

        session_id = int(
            session_id
        )

        if (
            session_id
            <= 0
        ):

            raise ValueError(
                (
                    "START session ID must "
                    "be non-zero"
                )
            )

        async with (
            self._write_lock
        ):

            self.session_id = (
                session_id
            )

            self.sequence = (
                0
            )

            self.sample_index = (
                0
            )

            self.next_environment_sample = (
                0
            )

            self.armed = (
                True
            )

            self._sync_sent_session = (
                None
            )

            self._next_block_deadline = (
                None
            )

            self.shared.arm_node(
                self.node_id,
                session_id,
            )

    # ==================================================================
    # STOP SESSION
    # ==================================================================

    def _disarm_session(
        self,
        session_id: int,
    ) -> None:
        """
        Stop one simulated acquisition session.
        """

        session_id = int(
            session_id
        )

        # Ignore stale STOP belonging to another session.
        if (
            session_id
            != self.session_id
        ):

            return

        self.armed = (
            False
        )

        self.shared.disarm_node(
            self.node_id,
            session_id,
        )

        self._next_block_deadline = (
            None
        )

        # Node 1 owns the shared clock and therefore removes it last.
        if self.master_node:

            self.shared.deactivate_clock(
                session_id
            )

    # ==================================================================
    # COMMAND LOOP
    # ==================================================================

    async def _command_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Process fixed 8-byte laptop -> ESP32 control frames.

        Important
        ---------
        Slave START no longer blocks this command loop while waiting for
        Node 1.

        This means STOP rollback remains processable even when master
        startup fails.
        """

        while True:

            raw = (
                await reader.readexactly(
                    CONTROL_SIZE
                )
            )

            frame = (
                unpack_control(
                    raw
                )
            )

            # ==========================================================
            # START
            # ==============================================================

            if (
                frame.command
                == ControlCommand.START
            ):

                if (
                    frame.session_id
                    == 0
                ):

                    print(
                        (
                            f"[SIM] Node {self.node_id}: "
                            "ignored START with session_id=0"
                        )
                    )

                    continue

                await self._arm_session(
                    frame.session_id
                )

                print(
                    (
                        f"[SIM] Node {self.node_id} armed "
                        f"for session "
                        f"0x{frame.session_id:08X}"
                    )
                )

                # ------------------------------------------------------
                # MASTER ACTIVATES SHARED BCLK/WS
                # ------------------------------------------------------

                if self.master_node:

                    slaves_ready = (
                        self.shared.activate_clock(
                            frame.session_id
                        )
                    )

                    if not slaves_ready:

                        print(
                            (
                                "[SIM] WARNING: master START "
                                f"session 0x{frame.session_id:08X} "
                                "before both slaves were armed"
                            )
                        )

                    else:

                        print(
                            (
                                "[SIM] Shared audio clock active "
                                f"for session "
                                f"0x{frame.session_id:08X}"
                            )
                        )

                continue

            # ==========================================================
            # STOP
            # ==============================================================

            if (
                frame.command
                == ControlCommand.STOP
            ):

                self._disarm_session(
                    frame.session_id
                )

                print(
                    (
                        f"[SIM] Node {self.node_id} stopped "
                        f"session "
                        f"0x{frame.session_id:08X}"
                    )
                )

                continue

            # ==========================================================
            # PING
            # ==============================================================

            if (
                frame.command
                == ControlCommand.PING
            ):

                await self._send_heartbeat(
                    writer
                )

                continue

    # ==================================================================
    # SESSION SYNC
    # ==================================================================

    async def _send_session_sync(
        self,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Send one session boundary SYNC marker.

        This marker is diagnostic/session coordination only.

        Shared sampleIndex remains the coarse timing authority.
        """

        session_id = (
            self.session_id
        )

        if (
            session_id
            <= 0
        ):

            return

        sync = SyncPayload(
            session_id=
                session_id,

            sync_id=
                1,

            sample_index=
                0,

            local_micros=
                self.local_micros(),
        )

        await self._send_packet(
            writer,
            PacketType.SYNC,
            pack_sync(
                sync
            ),
            sample_index=0,
        )

        self._sync_sent_session = (
            session_id
        )

    # ==================================================================
    # STREAM LOOP
    # ==================================================================

    async def _stream_loop(
        self,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Generate approximately real-time PCM blocks.

        SampleIndex advancement is based only on emitted sample count,
        not wall-clock timestamps.
        """

        while True:

            # ==========================================================
            # WAIT FOR VALID SHARED CLOCK
            # ==============================================================

            if not self.streaming:

                self._next_block_deadline = (
                    None
                )

                await asyncio.sleep(
                    0.005
                )

                continue

            # ==========================================================
            # FIRST PACKET IN SESSION = SYNC MARKER
            # ==============================================================

            if (
                self._sync_sent_session
                != self.session_id
            ):

                await self._send_session_sync(
                    writer
                )

            # ==========================================================
            # INITIALIZE PACING
            # ==============================================================

            if (
                self._next_block_deadline
                is None
            ):

                self._next_block_deadline = (
                    time.monotonic()
                )

            block_start = (
                self.sample_index
            )

            # ==========================================================
            # GENERATE AUDIO
            # ==============================================================

            (
                samples,
                clipped,
            ) = self.generate_audio(
                block_start
            )

            # ==========================================================
            # HEALTH FLAGS
            # ==============================================================

            flags = (
                PacketFlags.NONE
            )

            if clipped:

                flags |= (
                    PacketFlags.CLIPPED
                )

            # ==========================================================
            # AUDIO PACKET
            # ==============================================================

            await self._send_packet(
                writer,
                PacketType.AUDIO,
                samples.tobytes(),
                sample_index=
                    block_start,
                flags=
                    int(
                        flags
                    ),
            )

            self.sample_index += int(
                samples.size
            )

            # ==========================================================
            # NODE 1 BME280
            # ==============================================================

            if (
                self.master_node
                and self.streaming
                and self.sample_index
                >= self.next_environment_sample
            ):

                environment = (
                    EnvironmentPayload(
                        temperature_c=
                            SIM_TEMPERATURE_C,

                        humidity_percent=
                            SIM_HUMIDITY_PERCENT,

                        pressure_hpa=
                            SIM_PRESSURE_HPA,
                    )
                )

                await self._send_packet(
                    writer,
                    PacketType.ENVIRONMENT,
                    pack_environment(
                        environment
                    ),
                    sample_index=
                        self.sample_index,
                )

                self.next_environment_sample = (
                    self.sample_index
                    + ENVIRONMENT_PERIOD_SAMPLES
                )

            # ==========================================================
            # REAL-TIME PACING
            # ==============================================================

            assert (
                self._next_block_deadline
                is not None
            )

            self._next_block_deadline += (
                BLOCK_PERIOD_S
            )

            remaining = (
                self._next_block_deadline
                - time.monotonic()
            )

            if (
                remaining
                > 0.0
            ):

                await asyncio.sleep(
                    remaining
                )

            else:

                # If the simulator falls behind, do not insert or delete
                # samples. Reset only the wall-clock pacing reference.
                #
                # sampleIndex remains continuous.
                self._next_block_deadline = (
                    time.monotonic()
                )

    # ==================================================================
    # HEARTBEAT
    # ==================================================================

    async def _send_heartbeat(
        self,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Send one role-specific simulated heartbeat.
        """

        currently_streaming = (
            self.streaming
        )

        heartbeat = HeartbeatPayload(
            uptime_seconds=
                int(
                    time.monotonic()
                    - self.start_time
                ),

            wifi_rssi=
                -45
                - self.node_id,

            free_heap=
                220_000,

            dropped_audio_blocks=
                0,

            transmitted_audio_blocks=
                int(
                    self.sample_index
                    // FRAMES_PER_BLOCK
                ),

            i2s_errors=
                0,

            audio_queue_depth=
                0,

            streaming=
                currently_streaming,

            bme_available=(
                True
                if self.master_node
                else None
            ),

            sync_received=(
                (
                    self._sync_sent_session
                    == self.session_id
                )
                if not self.master_node
                else None
            ),

            clock_healthy=(
                currently_streaming
                if not self.master_node
                else None
            ),
        )

        payload = (
            pack_master_heartbeat(
                heartbeat
            )
            if self.master_node
            else pack_slave_heartbeat(
                heartbeat
            )
        )

        await self._send_packet(
            writer,
            PacketType.HEARTBEAT,
            payload,
        )

    # ==================================================================
    # PERIODIC HEARTBEAT LOOP
    # ==================================================================

    async def _heartbeat_loop(
        self,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Emit node health every five seconds.
        """

        while True:

            await asyncio.sleep(
                5.0
            )

            await self._send_heartbeat(
                writer
            )


# ======================================================================
# SIMULATOR MAIN
# ======================================================================


async def main_async(
    host: str,
    port: int,
) -> None:
    """
    Launch all three fake ESP32 nodes.
    """

    shared = (
        SharedSimulation()
    )

    print(
        (
            "[SIM] Protocol v4 "
            "| 3-node Wildlife Soundscape simulator"
        )
    )

    print(
        (
            "[SIM] Source position: "
            f"x={shared.source_xy[0]:.3f} m, "
            f"y={shared.source_xy[1]:.3f} m"
        )
    )

    print(
        (
            "[SIM] Speed of sound: "
            f"{shared.speed_of_sound:.2f} m/s"
        )
    )

    print(
        (
            "[SIM] Audio: "
            f"{SAMPLE_RATE} Hz "
            f"| {FRAMES_PER_BLOCK} samples/block "
            "| PCM16 mono"
        )
    )

    print(
        (
            "[SIM] Synthetic acoustic event: "
            "0.55 s harmonic chirp "
            "| first emission=2 s "
            "| period=5 s"
        )
    )

    for node_id in (
        1,
        2,
        3,
    ):

        print(
            (
                f"[SIM] Node {node_id}: "
                f"position={shared.node_xy[node_id]} "
                f"| distance="
                f"{shared.distance_to_source(node_id):.3f} m "
                f"| propagation_delay="
                f"{shared.delay_samples(node_id)} samples"
            )
        )

    nodes = [
        FakeNode(
            node_id=
                node_id,

            host=
                host,

            port=
                port,

            shared=
                shared,
        )
        for node_id
        in (
            1,
            2,
            3,
        )
    ]

    await asyncio.gather(
        *(
            node.run()
            for node
            in nodes
        )
    )


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    """
    Simulator command-line entry point.
    """

    parser = (
        argparse.ArgumentParser(
            description=(
                "Three-node Wildlife Soundscape "
                "Protocol-v4 acoustic simulator"
            )
        )
    )

    parser.add_argument(
        "--host",
        default=
            "127.0.0.1",
        help=
            "Laptop receiver host",
    )

    parser.add_argument(
        "--port",
        type=
            int,
        default=
            CONFIG.network.port,
        help=
            "Laptop receiver TCP port",
    )

    args = (
        parser.parse_args()
    )

    if not (
        1
        <= args.port
        <= 65535
    ):

        parser.error(
            "port must be between 1 and 65535"
        )

    try:

        asyncio.run(
            main_async(
                args.host,
                args.port,
            )
        )

    except KeyboardInterrupt:

        pass


# ======================================================================
# ENTRY POINT
# ======================================================================


if __name__ == "__main__":

    main()
