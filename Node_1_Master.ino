/*
  Wildlife Soundscape Mapping & Behavior Analysis System
  Node 1 MASTER firmware v1.4.0 / wire protocol v4

  Target:
      ESP32 DevKit V1
      ESP32-WROOM-32
      Arduino-ESP32 3.x

  CURRENT SENSORS
  ---------------
  Acoustic:
      INMP441 digital I2S MEMS microphone

      BCLK = GPIO26
      WS   = GPIO25
      SD   = GPIO33
      L/R  = GND
      VDD  = 3V3

  Environment:
      BME280

      SDA = GPIO21
      SCL = GPIO22

  Synchronization:
      GPIO27 -> Node 2 GPIO27
      GPIO27 -> Node 3 GPIO27

  IMPORTANT
  ---------
  Node 1 is the I2S MASTER.

  Node 1 generates the shared:
      BCLK
      WS / LRCLK

  Nodes 2 and 3 consume those clocks in I2S SLAVE mode.

  Therefore:

      shared BCLK + WS
          =
      acoustic sample-rate synchronization

  GPIO27 SYNC is only:
      session marker
      start marker
      diagnostic reference

  GPIO27 is NOT the TDOA sample clock.

  TDOA is reconstructed on the laptop using:
      shared-clock sampleIndex
      +
      GCC-PHAT waveform delay estimation


  SENSOR-REPLACEMENT DESIGN
  -------------------------
  Sensor-specific code is isolated inside:

      AUDIO SENSOR ADAPTER
      ENVIRONMENT SENSOR ADAPTER

  The rest of the firmware operates on standardized outputs:

      Audio:
          48 kHz
          mono
          signed PCM16

      Environment:
          temperatureC
          humidityPercent
          pressureHpa

  Therefore a future sensor replacement should primarily require
  modification of the relevant adapter rather than the network,
  protocol, session, queue or analytics architecture.
*/


// ======================================================================
// INCLUDES
// ======================================================================


#include <Arduino.h>

#include <WiFi.h>

#include <Wire.h>

#include <ESP_I2S.h>

#include <Adafruit_Sensor.h>

#include <Adafruit_BME280.h>

#include <math.h>

#include <string.h>


#include "freertos/FreeRTOS.h"

#include "freertos/task.h"

#include "freertos/queue.h"

#include "freertos/event_groups.h"


// ======================================================================
// NODE IDENTITY
// ======================================================================


#define NODE_ID 1

#define FIRMWARE_VERSION "1.4.0"


// ======================================================================
// USER NETWORK SETTINGS
// ======================================================================


const char* WIFI_SSID =
    "YOUR_WIFI_NAME";


const char* WIFI_PASSWORD =
    "YOUR_WIFI_PASSWORD";


IPAddress LAPTOP_IP(
    192,
    168,
    1,
    100
);


constexpr uint16_t LAPTOP_PORT =
    5001;


// ======================================================================
// HARDWARE PINS
// ======================================================================


// Shared master-generated I2S clocks.
constexpr int PIN_BCLK =
    26;


constexpr int PIN_WS =
    25;


// Local microphone data only.
constexpr int PIN_MIC_DATA =
    33;


// Session/start marker.
constexpr int PIN_SYNC =
    27;


// Environmental sensor bus.
constexpr int PIN_BME_SDA =
    21;


constexpr int PIN_BME_SCL =
    22;


// ======================================================================
// STANDARDIZED AUDIO OUTPUT
// ======================================================================


constexpr uint32_t SAMPLE_RATE =
    48000;


constexpr uint16_t FRAMES_PER_BLOCK =
    1024;


// INMP441 is read using stereo I2S slots.
//
// One I2S frame:
//
//     LEFT  int32
//     RIGHT int32
//
// L/R pin of INMP441 is grounded, therefore its data appears in LEFT.
constexpr uint16_t RAW_VALUES_PER_BLOCK =
    FRAMES_PER_BLOCK * 2;


constexpr uint8_t CURRENT_MIC_SLOT_INDEX =
    0;


// Laptop alignment diagnostic tolerance.
//
// At 48 kHz:
//
//     10 samples ≈ 0.208 ms
constexpr int16_t SYNC_ALIGN_TOLERANCE_SAMPLES =
    10;


// ======================================================================
// QUEUE CONFIGURATION
// ======================================================================


// 6 × 1024 samples.
//
// Approximately:
//
//     6 × 21.33 ms
//     ≈ 128 ms
//
// of short-term audio buffering.
constexpr uint8_t AUDIO_QUEUE_DEPTH =
    6;


constexpr uint8_t TELEMETRY_QUEUE_DEPTH =
    8;


constexpr uint8_t AUDIO_CONTROL_QUEUE_DEPTH =
    4;


// ======================================================================
// PERIODIC INTERVALS
// ======================================================================


constexpr uint32_t ENV_INTERVAL_MS =
    2000;


constexpr uint32_t ENV_RETRY_INTERVAL_MS =
    30000;


constexpr uint32_t HEARTBEAT_INTERVAL_MS =
    5000;


// ======================================================================
// SYNCHRONIZATION PULSE
// ======================================================================


constexpr uint32_t SYNC_RESET_LOW_MS =
    5;


// Give already-armed slaves time to observe GPIO27 before BCLK/WS start.
constexpr uint32_t SYNC_SLAVE_ARM_MS =
    2;


constexpr uint32_t SYNC_MIN_HIGH_MS =
    10;


// ======================================================================
// CONNECTION MANAGEMENT
// ======================================================================


constexpr uint32_t RECONNECT_MIN_MS =
    1000;


constexpr uint32_t RECONNECT_MAX_MS =
    30000;


constexpr uint32_t TCP_WRITE_STALL_TIMEOUT_MS =
    2000;


// ======================================================================
// AUDIO CONTROL TIMEOUT
// ======================================================================


constexpr uint32_t AUDIO_CONTROL_TIMEOUT_MS =
    1500;


// ======================================================================
// PROTOCOL CONSTANTS
// ======================================================================


constexpr uint32_t PACKET_MAGIC =
    0x574C5343UL;


// ASCII:
// WLSC


constexpr uint8_t PROTOCOL_VERSION =
    4;


constexpr uint16_t CONTROL_MAGIC =
    0xC0DE;


constexpr uint8_t CONTROL_VERSION =
    1;


// ======================================================================
// PACKET FLAGS
// ======================================================================


enum PacketFlags : uint8_t {

  FLAG_NONE =
      0x00,

  FLAG_CLIPPED =
      0x01,

  FLAG_CLOCK_FAULT =
      0x02,

  FLAG_QUEUE_CONGESTED =
      0x04
};


// ======================================================================
// PACKET TYPES
// ======================================================================


enum PacketType : uint8_t {

  PKT_HELLO =
      1,

  PKT_AUDIO =
      2,

  PKT_ENVIRONMENT =
      3,

  PKT_HEARTBEAT =
      4,

  PKT_SYNC =
      5
};


// ======================================================================
// LAPTOP CONTROL COMMANDS
// ======================================================================


enum ControlCommand : uint8_t {

  CMD_START =
      0xA1,

  CMD_STOP =
      0xA2,

  CMD_PING =
      0xA3
};


// ======================================================================
// INTERNAL AUDIO CONTROL
// ======================================================================


enum AudioControlAction : uint8_t {

  AUDIO_CONTROL_START =
      1,

  AUDIO_CONTROL_STOP =
      2
};


// ======================================================================
// CONTROL FRAME
// ======================================================================


struct __attribute__((packed)) ControlFrame {

  uint16_t magic;

  uint8_t version;

  uint8_t command;

  uint32_t sessionId;
};


// ======================================================================
// PACKET HEADER
// ======================================================================


struct __attribute__((packed)) PacketHeader {

  uint32_t magic;

  uint8_t protocolVersion;

  uint8_t nodeId;

  uint8_t packetType;

  uint8_t flags;

  uint32_t sequence;

  uint32_t sessionId;

  uint64_t sampleIndex;

  uint32_t localMicros;

  uint32_t i2sErrorCount;

  uint32_t payloadLength;

  uint32_t payloadCRC32;
};


// ======================================================================
// HELLO PAYLOAD
// ======================================================================


struct __attribute__((packed)) HelloPayload {

  uint32_t sampleRate;

  uint16_t framesPerPacket;

  uint8_t bitsPerSample;

  uint8_t channels;

  uint8_t masterNode;

  int16_t syncToleranceSamples;

  char firmware[16];
};


// ======================================================================
// ENVIRONMENT PAYLOAD
// ======================================================================


struct __attribute__((packed)) EnvironmentPayload {

  float temperatureC;

  float humidityPercent;

  float pressureHpa;
};


// ======================================================================
// SYNC PAYLOAD
// ======================================================================


struct __attribute__((packed)) SyncPayload {

  uint32_t sessionId;

  uint32_t syncId;

  uint64_t sampleIndex;

  uint32_t localMicros;
};


// ======================================================================
// MASTER HEARTBEAT
// ======================================================================


struct __attribute__((packed)) HeartbeatPayload {

  uint32_t uptimeSeconds;

  int32_t wifiRSSI;

  uint32_t freeHeap;

  uint32_t droppedAudioBlocks;

  uint32_t transmittedAudioBlocks;

  uint32_t i2sErrors;

  uint16_t audioQueueDepth;

  uint8_t streaming;

  uint8_t bmeAvailable;
};


// ======================================================================
// INTERNAL STANDARD ENVIRONMENT READING
// ======================================================================
//
// Future environmental sensors should populate THIS structure.
//
// Everything after the adapter remains unchanged.
// ======================================================================


struct EnvironmentReading {

  float temperatureC;

  float humidityPercent;

  float pressureHpa;
};


// ======================================================================
// AUDIO QUEUE BLOCK
// ======================================================================


struct AudioBlock {

  uint32_t sessionId;

  uint64_t sampleIndex;

  uint32_t localMicros;

  uint32_t i2sErrorCountAtCapture;

  uint16_t frameCount;

  uint8_t flags;

  int16_t pcm[FRAMES_PER_BLOCK];
};


// ======================================================================
// TELEMETRY QUEUE MESSAGE
// ======================================================================


struct TelemetryMessage {

  PacketType type;

  uint32_t sessionId;

  uint64_t sampleIndex;

  uint32_t localMicros;

  uint32_t i2sErrorCountAtCapture;

  uint32_t payloadLength;

  uint8_t flags;

  uint8_t payload[64];
};


// ======================================================================
// INTERNAL AUDIO COMMAND
// ======================================================================


struct AudioControlMessage {

  AudioControlAction action;

  uint32_t sessionId;
};


// ======================================================================
// RUNTIME SNAPSHOT
// ======================================================================


struct RuntimeSnapshot {

  bool streaming;

  uint32_t sessionId;

  uint64_t sampleIndex;
};


// ======================================================================
// STATISTICS SNAPSHOT
// ======================================================================


struct StatisticsSnapshot {

  uint32_t droppedAudioBlocks;

  uint32_t transmittedAudioBlocks;

  uint32_t i2sErrors;

  bool clockFault;
};


// ======================================================================
// PROTOCOL SIZE CHECKS
// ======================================================================


static_assert(
    sizeof(ControlFrame) == 8,
    "ControlFrame size mismatch"
);


static_assert(
    sizeof(PacketHeader) == 40,
    "PacketHeader size mismatch"
);


static_assert(
    sizeof(HelloPayload) == 27,
    "HelloPayload size mismatch"
);


static_assert(
    sizeof(EnvironmentPayload) == 12,
    "EnvironmentPayload size mismatch"
);


static_assert(
    sizeof(SyncPayload) == 20,
    "SyncPayload size mismatch"
);


static_assert(
    sizeof(HeartbeatPayload) == 28,
    "Master heartbeat size mismatch"
);


static_assert(
    sizeof(EnvironmentPayload) <= 64,
    "Telemetry buffer too small"
);


static_assert(
    sizeof(HeartbeatPayload) <= 64,
    "Telemetry buffer too small"
);


// ======================================================================
// HARDWARE OBJECTS
// ======================================================================


I2SClass I2S;


Adafruit_BME280 bme;


WiFiClient tcpClient;


// ======================================================================
// FREERTOS OBJECTS
// ======================================================================


QueueHandle_t audioQueue =
    nullptr;


QueueHandle_t telemetryQueue =
    nullptr;


QueueHandle_t audioControlQueue =
    nullptr;


EventGroupHandle_t audioEvents =
    nullptr;


// ======================================================================
// AUDIO EVENT BITS
// ======================================================================


constexpr EventBits_t AUDIO_EVENT_STARTED =
    (1U << 0);


constexpr EventBits_t AUDIO_EVENT_STOPPED =
    (1U << 1);


constexpr EventBits_t AUDIO_EVENT_FAILED =
    (1U << 2);


// ======================================================================
// SHARED STATE LOCK
// ======================================================================
//
// ESP32 is a 32-bit MCU.
//
// uint64_t sampleIndex is therefore protected against torn cross-core
// reads.
// ======================================================================


portMUX_TYPE stateMux =
    portMUX_INITIALIZER_UNLOCKED;


// ======================================================================
// SHARED RUNTIME STATE
// ======================================================================


bool runtimeStreaming =
    false;


uint32_t currentSessionId =
    0;


uint64_t totalSampleCount =
    0;


// ======================================================================
// STATISTICS
// ======================================================================


uint32_t droppedAudioBlocks =
    0;


uint32_t transmittedAudioBlocks =
    0;


uint32_t i2sErrors =
    0;


bool clockFaultActive =
    false;


// ======================================================================
// NETWORK-TASK STATE
// ======================================================================
//
// Only networkTask serializes packets.
// ======================================================================


uint32_t packetSequence =
    0;


uint32_t syncId =
    0;


// ======================================================================
// SENSOR STATE
// ======================================================================


volatile bool environmentSensorAvailable =
    false;


// I2S ownership remains entirely inside audioCaptureTask.
bool audioDriverStarted =
    false;


// ======================================================================
// SENSOR IDENTIFICATION
// ======================================================================


const char* currentAudioSensorName() {

  return "INMP441";
}


const char* currentEnvironmentSensorName() {

  return "BME280";
}


// ======================================================================
// RUNTIME STATE
// ======================================================================


RuntimeSnapshot snapshotRuntimeState() {

  RuntimeSnapshot snapshot{};

  portENTER_CRITICAL(
      &stateMux
  );

  snapshot.streaming =
      runtimeStreaming;

  snapshot.sessionId =
      currentSessionId;

  snapshot.sampleIndex =
      totalSampleCount;

  portEXIT_CRITICAL(
      &stateMux
  );

  return snapshot;
}


// ======================================================================
// SET RUNTIME STATE
// ======================================================================


void setRuntimeState(
    bool streaming,
    uint32_t sessionId,
    uint64_t sampleIndex
) {

  portENTER_CRITICAL(
      &stateMux
  );

  runtimeStreaming =
      streaming;

  currentSessionId =
      sessionId;

  totalSampleCount =
      sampleIndex;

  portEXIT_CRITICAL(
      &stateMux
  );
}


// ======================================================================
// RESERVE SAMPLE RANGE
// ======================================================================
//
// Only accepted while the requested session is still active.
//
// This operation atomically:
//
//     reads current sampleIndex
//     increments sampleIndex
//
// ensuring Core 1 can never observe a torn uint64_t value.
// ======================================================================


bool reserveSampleRange(
    uint32_t sessionId,
    uint16_t frameCount,
    uint64_t& startSample
) {

  bool accepted =
      false;

  portENTER_CRITICAL(
      &stateMux
  );

  if (
      runtimeStreaming &&
      currentSessionId == sessionId
  ) {

    startSample =
        totalSampleCount;

    totalSampleCount +=
        frameCount;

    accepted =
        true;
  }

  portEXIT_CRITICAL(
      &stateMux
  );

  return accepted;
}


// ======================================================================
// STATISTICS
// ======================================================================


uint32_t incrementI2SErrors() {

  uint32_t value;

  portENTER_CRITICAL(
      &stateMux
  );

  ++i2sErrors;

  value =
      i2sErrors;

  portEXIT_CRITICAL(
      &stateMux
  );

  return value;
}


void incrementDroppedAudioBlocks() {

  portENTER_CRITICAL(
      &stateMux
  );

  ++droppedAudioBlocks;

  portEXIT_CRITICAL(
      &stateMux
  );
}


void incrementTransmittedAudioBlocks() {

  portENTER_CRITICAL(
      &stateMux
  );

  ++transmittedAudioBlocks;

  portEXIT_CRITICAL(
      &stateMux
  );
}


void setClockFault(
    bool fault
) {

  portENTER_CRITICAL(
      &stateMux
  );

  clockFaultActive =
      fault;

  portEXIT_CRITICAL(
      &stateMux
  );
}


StatisticsSnapshot snapshotStatistics() {

  StatisticsSnapshot snapshot{};

  portENTER_CRITICAL(
      &stateMux
  );

  snapshot.droppedAudioBlocks =
      droppedAudioBlocks;

  snapshot.transmittedAudioBlocks =
      transmittedAudioBlocks;

  snapshot.i2sErrors =
      i2sErrors;

  snapshot.clockFault =
      clockFaultActive;

  portEXIT_CRITICAL(
      &stateMux
  );

  return snapshot;
}


// ======================================================================
// CRC32
// ======================================================================


uint32_t calculateCRC32(
    const uint8_t* data,
    size_t length
) {

  uint32_t crc =
      0xFFFFFFFFUL;

  for (
      size_t i = 0;
      i < length;
      ++i
  ) {

    crc ^=
        data[i];

    for (
        uint8_t bit = 0;
        bit < 8;
        ++bit
    ) {

      if (
          crc & 1U
      ) {

        crc =
            (crc >> 1)
            ^ 0xEDB88320UL;

      } else {

        crc >>=
            1;
      }
    }
  }

  return crc
      ^ 0xFFFFFFFFUL;
}


// ======================================================================
// TCP WRITE
// ======================================================================


bool sendAll(
    const uint8_t* data,
    size_t length
) {

  size_t sent =
      0;

  uint32_t lastProgress =
      millis();

  while (
      sent < length
  ) {

    if (
        !tcpClient.connected()
    ) {

      return false;
    }

    size_t written =
        tcpClient.write(
            data + sent,
            length - sent
        );

    if (
        written > 0
    ) {

      sent +=
          written;

      lastProgress =
          millis();

      continue;
    }

    if (
        millis() - lastProgress
        >= TCP_WRITE_STALL_TIMEOUT_MS
    ) {

      Serial.println(
          "[TCP] stalled write; closing connection"
      );

      tcpClient.stop();

      return false;
    }

    vTaskDelay(
        pdMS_TO_TICKS(1)
    );
  }

  return true;
}


// ======================================================================
// SEND PROTOCOL PACKET
// ======================================================================


bool sendPacket(
    PacketType type,
    uint32_t sessionId,
    uint64_t sampleIndex,
    uint32_t captureMicros,
    uint32_t captureErrors,
    const void* payload,
    uint32_t payloadLength,
    uint8_t flags = FLAG_NONE
) {

  PacketHeader header{};

  header.magic =
      PACKET_MAGIC;

  header.protocolVersion =
      PROTOCOL_VERSION;

  header.nodeId =
      NODE_ID;

  header.packetType =
      static_cast<uint8_t>(
          type
      );

  header.flags =
      flags;

  header.sequence =
      packetSequence++;

  header.sessionId =
      sessionId;

  header.sampleIndex =
      sampleIndex;

  header.localMicros =
      captureMicros;

  header.i2sErrorCount =
      captureErrors;

  header.payloadLength =
      payloadLength;

  if (
      payload != nullptr &&
      payloadLength > 0
  ) {

    header.payloadCRC32 =
        calculateCRC32(
            reinterpret_cast<const uint8_t*>(
                payload
            ),
            payloadLength
        );

  } else {

    header.payloadCRC32 =
        0;
  }

  if (
      !sendAll(
          reinterpret_cast<const uint8_t*>(
              &header
          ),
          sizeof(header)
      )
  ) {

    return false;
  }

  if (
      payload != nullptr &&
      payloadLength > 0
  ) {

    return sendAll(
        reinterpret_cast<const uint8_t*>(
            payload
        ),
        payloadLength
    );
  }

  return true;
}


// ======================================================================
// HEALTH FLAGS
// ======================================================================


uint8_t currentHealthFlags() {

  uint8_t flags =
      FLAG_NONE;

  if (
      audioQueue != nullptr
  ) {

    const UBaseType_t depth =
        uxQueueMessagesWaiting(
            audioQueue
        );

    // Queue congestion threshold ≈ 80%.
    if (
        depth * 5U
        >= AUDIO_QUEUE_DEPTH * 4U
    ) {

      flags |=
          FLAG_QUEUE_CONGESTED;
    }
  }

  StatisticsSnapshot stats =
      snapshotStatistics();

  if (
      stats.clockFault
  ) {

    flags |=
        FLAG_CLOCK_FAULT;
  }

  return flags;
}


// ======================================================================
// WI-FI SETUP
// ======================================================================
//
// Connection itself is handled by networkTask.
//
// setup() therefore does not need to block waiting for Wi-Fi.
// ======================================================================


void setupWiFi() {

  WiFi.mode(
      WIFI_STA
  );

  WiFi.setSleep(
      false
  );

  WiFi.setAutoReconnect(
      true
  );

  WiFi.begin(
      WIFI_SSID,
      WIFI_PASSWORD
  );

  Serial.println(
      "[WiFi] connection initiated"
  );
}


// ======================================================================
// ENSURE WI-FI
// ======================================================================


bool ensureWiFi() {

  if (
      WiFi.status() == WL_CONNECTED
  ) {

    return true;
  }

  uint32_t retry =
      RECONNECT_MIN_MS;

  const uint32_t disconnectedAt =
      millis();

  while (
      WiFi.status() != WL_CONNECTED
  ) {

    Serial.printf(
        "[WiFi] reconnect; wait=%lu ms\n",
        retry
    );

    WiFi.reconnect();

    const uint32_t attemptStarted =
        millis();

    while (
        millis() - attemptStarted
        < retry
    ) {

      if (
          WiFi.status() == WL_CONNECTED
      ) {

        Serial.printf(
            "[WiFi] connected after %lu ms | IP=",
            millis() - disconnectedAt
        );

        Serial.println(
            WiFi.localIP()
        );

        return true;
      }

      vTaskDelay(
          pdMS_TO_TICKS(100)
      );
    }

    if (
        retry >= RECONNECT_MAX_MS / 2U
    ) {

      retry =
          RECONNECT_MAX_MS;

    } else {

      retry *=
          2U;
    }
  }

  return true;
}


// ======================================================================
// ENSURE LAPTOP TCP CONNECTION
// ======================================================================


bool ensureLaptopConnection() {

  if (
      tcpClient.connected()
  ) {

    return true;
  }

  tcpClient.stop();

  uint32_t retry =
      RECONNECT_MIN_MS;

  while (
      !tcpClient.connected()
  ) {

    ensureWiFi();

    Serial.printf(
        "[TCP] connecting %s:%u | retry=%lu ms\n",
        LAPTOP_IP.toString().c_str(),
        LAPTOP_PORT,
        retry
    );

    if (
        tcpClient.connect(
            LAPTOP_IP,
            LAPTOP_PORT
        )
    ) {

      tcpClient.setNoDelay(
          true
      );

      tcpClient.setTimeout(
          50
      );

      Serial.println(
          "[TCP] connected"
      );

      return true;
    }

    vTaskDelay(
        pdMS_TO_TICKS(
            retry
        )
    );

    if (
        retry >= RECONNECT_MAX_MS / 2U
    ) {

      retry =
          RECONNECT_MAX_MS;

    } else {

      retry *=
          2U;
    }
  }

  return true;
}


// ======================================================================
// QUEUE FLUSH
// ======================================================================


void flushAudioQueue() {

  AudioBlock block{};

  while (
      audioQueue != nullptr &&
      xQueueReceive(
          audioQueue,
          &block,
          0
      ) == pdTRUE
  ) {
  }
}


void flushTelemetryQueue() {

  TelemetryMessage message{};

  while (
      telemetryQueue != nullptr &&
      xQueueReceive(
          telemetryQueue,
          &message,
          0
      ) == pdTRUE
  ) {
  }
}


// ======================================================================
// ======================================================================
// AUDIO SENSOR ADAPTER
// ======================================================================
// ======================================================================
//
// CURRENT BACKEND:
//     INMP441 digital I2S microphone
//
// Standardized output:
//     signed PCM16
//     mono
//     48 kHz
//
// FUTURE SENSOR CHANGE
// --------------------
// If another compatible I2S microphone is used, usually only:
//
//     pins
//     I2S slot
//     word alignment / conversion
//
// need modification.
//
// An analog microphone would require replacing these acquisition
// functions with an ADC/I2S-ADC implementation while keeping the
// AudioBlock / protocol output unchanged.
// ======================================================================


// ======================================================================
// BEGIN CURRENT AUDIO SENSOR
// ======================================================================


bool audioSensorBeginMaster() {

  if (
      audioDriverStarted
  ) {

    I2S.end();

    audioDriverStarted =
        false;
  }

  I2S.setPins(
      PIN_BCLK,
      PIN_WS,
      -1,
      PIN_MIC_DATA
  );

  bool ok =
      I2S.begin(
          I2S_MODE_STD,
          SAMPLE_RATE,
          I2S_DATA_BIT_WIDTH_32BIT,
          I2S_SLOT_MODE_STEREO,
          -1,
          I2S_ROLE_MASTER
      );

  if (
      !ok
  ) {

    const uint32_t errorCount =
        incrementI2SErrors();

    setClockFault(
        true
    );

    Serial.printf(
        "[I2S] start failed | errors=%lu | code=%d\n",
        errorCount,
        I2S.lastError()
    );

    I2S.end();

    audioDriverStarted =
        false;

    return false;
  }

  audioDriverStarted =
      true;

  setClockFault(
      false
  );

  Serial.printf(
      "[I2S] MASTER active | sensor=%s | %lu Hz\n",
      currentAudioSensorName(),
      SAMPLE_RATE
  );

  return true;
}


// ======================================================================
// STOP CURRENT AUDIO SENSOR
// ======================================================================


void audioSensorEnd() {

  if (
      !audioDriverStarted
  ) {

    return;
  }

  I2S.end();

  audioDriverStarted =
      false;

  Serial.println(
      "[I2S] stopped"
  );
}


// ======================================================================
// READ CURRENT AUDIO SENSOR
// ======================================================================


size_t audioSensorReadRaw(
    int32_t* destination,
    size_t rawValueCount
) {

  return I2S.readBytes(
      reinterpret_cast<char*>(
          destination
      ),
      rawValueCount
          * sizeof(int32_t)
  );
}


// ======================================================================
// CURRENT SENSOR WORD -> PCM16
// ======================================================================


inline int16_t audioSensorWordToPCM16(
    int32_t rawWord
) {

  // INMP441 provides a 24-bit signed sample inside the 32-bit I2S word.
  //
  // The upper 16 significant bits are retained for the standardized
  // PCM16 network representation.
  return static_cast<int16_t>(
      rawWord >> 16
  );
}


// ======================================================================
// EXTRACT MICROPHONE WORD FROM I2S FRAME
// ======================================================================


inline int32_t audioSensorFrameWord(
    const int32_t* raw,
    size_t frameIndex
) {

  return raw[
      frameIndex * 2
      + CURRENT_MIC_SLOT_INDEX
  ];
}


// ======================================================================
// ======================================================================
// ENVIRONMENT SENSOR ADAPTER
// ======================================================================
// ======================================================================
//
// CURRENT BACKEND:
//     BME280
//
// Standardized output:
//     temperatureC
//     humidityPercent
//     pressureHpa
//
// FUTURE SENSOR CHANGE
// --------------------
// Replace environmentSensorBegin() and environmentSensorRead().
//
// Everything after this adapter continues consuming EnvironmentReading.
//
// If a future single sensor does not provide all three physical
// quantities, use an appropriate sensor combination.
//
// Missing physical measurements must NOT be fabricated.
// ======================================================================


// ======================================================================
// TRY BME280 ADDRESS
// ======================================================================


bool tryBME280Address(
    uint8_t address
) {

  for (
      int attempt = 1;
      attempt <= 3;
      ++attempt
  ) {

    Serial.printf(
        "[ENV] BME280 0x%02X attempt %d\n",
        address,
        attempt
    );

    if (
        bme.begin(
            address,
            &Wire
        )
    ) {

      if (
          bme.sensorID() == 0x60
      ) {

        return true;
      }

      Serial.printf(
          "[ENV] unexpected chip ID: 0x%02lX\n",
          static_cast<unsigned long>(
              bme.sensorID()
          )
      );

      return false;
    }

    delay(
        100
    );
  }

  return false;
}


// ======================================================================
// BEGIN ENVIRONMENT SENSOR
// ======================================================================


bool environmentSensorBegin() {

  bool available =
      tryBME280Address(
          0x76
      );

  if (
      !available
  ) {

    available =
        tryBME280Address(
            0x77
        );
  }

  environmentSensorAvailable =
      available;

  if (
      available
  ) {

    Serial.printf(
        "[ENV] %s ready\n",
        currentEnvironmentSensorName()
    );

  } else {

    Serial.printf(
        "[ENV] %s unavailable\n",
        currentEnvironmentSensorName()
    );
  }

  return available;
}


// ======================================================================
// READ STANDARDIZED ENVIRONMENT VALUES
// ======================================================================


bool environmentSensorRead(
    EnvironmentReading& reading
) {

  if (
      !environmentSensorAvailable
  ) {

    return false;
  }

  reading.temperatureC =
      bme.readTemperature();

  reading.humidityPercent =
      bme.readHumidity();

  reading.pressureHpa =
      bme.readPressure()
      / 100.0F;

  if (
      !isfinite(reading.temperatureC) ||
      !isfinite(reading.humidityPercent) ||
      !isfinite(reading.pressureHpa)
  ) {

    return false;
  }

  return true;
}


// ======================================================================
// HELLO PACKET
// ======================================================================


bool sendHello() {

  RuntimeSnapshot runtime =
      snapshotRuntimeState();

  StatisticsSnapshot stats =
      snapshotStatistics();

  HelloPayload payload{};

  payload.sampleRate =
      SAMPLE_RATE;

  payload.framesPerPacket =
      FRAMES_PER_BLOCK;

  payload.bitsPerSample =
      16;

  payload.channels =
      1;

  payload.masterNode =
      1;

  payload.syncToleranceSamples =
      SYNC_ALIGN_TOLERANCE_SAMPLES;

  strncpy(
      payload.firmware,
      FIRMWARE_VERSION,
      sizeof(payload.firmware) - 1
  );

  return sendPacket(
      PKT_HELLO,
      runtime.sessionId,
      runtime.sampleIndex,
      micros(),
      stats.i2sErrors,
      &payload,
      sizeof(payload),
      currentHealthFlags()
  );
}


// ======================================================================
// AUDIO TASK -> STOPPED STATE
// ======================================================================


void audioTaskStopHardware() {

  audioSensorEnd();

  setRuntimeState(
      false,
      0,
      0
  );

  setClockFault(
      false
  );

  xEventGroupClearBits(
      audioEvents,
      AUDIO_EVENT_STARTED
  );

  xEventGroupSetBits(
      audioEvents,
      AUDIO_EVENT_STOPPED
  );
}


// ======================================================================
// REQUEST AUDIO START
// ======================================================================


bool requestAudioStart(
    uint32_t sessionId
) {

  AudioControlMessage command{};

  command.action =
      AUDIO_CONTROL_START;

  command.sessionId =
      sessionId;

  xEventGroupClearBits(
      audioEvents,
      AUDIO_EVENT_STARTED
      | AUDIO_EVENT_STOPPED
      | AUDIO_EVENT_FAILED
  );

  if (
      xQueueSend(
          audioControlQueue,
          &command,
          pdMS_TO_TICKS(100)
      ) != pdTRUE
  ) {

    Serial.println(
        "[AUDIO] unable to queue START"
    );

    return false;
  }

  EventBits_t result =
      xEventGroupWaitBits(
          audioEvents,
          AUDIO_EVENT_STARTED
              | AUDIO_EVENT_FAILED,
          pdFALSE,
          pdFALSE,
          pdMS_TO_TICKS(
              AUDIO_CONTROL_TIMEOUT_MS
          )
      );

  return (
      result
      & AUDIO_EVENT_STARTED
  );
}


// ======================================================================
// REQUEST AUDIO STOP
// ======================================================================


bool requestAudioStop() {

  EventBits_t currentBits =
      xEventGroupGetBits(
          audioEvents
      );

  RuntimeSnapshot runtime =
      snapshotRuntimeState();

  if (
      !runtime.streaming &&
      (currentBits & AUDIO_EVENT_STOPPED)
  ) {

    return true;
  }

  AudioControlMessage command{};

  command.action =
      AUDIO_CONTROL_STOP;

  command.sessionId =
      runtime.sessionId;

  xEventGroupClearBits(
      audioEvents,
      AUDIO_EVENT_STOPPED
  );

  if (
      xQueueSend(
          audioControlQueue,
          &command,
          pdMS_TO_TICKS(100)
      ) != pdTRUE
  ) {

    Serial.println(
        "[AUDIO] unable to queue STOP"
    );

    return false;
  }

  EventBits_t result =
      xEventGroupWaitBits(
          audioEvents,
          AUDIO_EVENT_STOPPED,
          pdFALSE,
          pdFALSE,
          pdMS_TO_TICKS(
              AUDIO_CONTROL_TIMEOUT_MS
          )
      );

  return (
      result
      & AUDIO_EVENT_STOPPED
  );
}


// ======================================================================
// STOP ACQUISITION SESSION
// ======================================================================


bool stopStreaming() {

  digitalWrite(
      PIN_SYNC,
      LOW
  );

  bool stopped =
      requestAudioStop();

  if (
      !stopped
  ) {

    Serial.println(
        "[SYSTEM] audio STOP timed out"
    );
  }

  flushAudioQueue();

  flushTelemetryQueue();

  setRuntimeState(
      false,
      0,
      0
  );

  Serial.println(
      "[SYSTEM] acquisition stopped"
  );

  return stopped;
}


// ======================================================================
// START ACQUISITION SESSION
// ======================================================================


bool startStreaming(
    uint32_t sessionId
) {

  if (
      sessionId == 0
  ) {

    Serial.println(
        "[CONTROL] START rejected: sessionId=0"
    );

    return false;
  }

  RuntimeSnapshot existing =
      snapshotRuntimeState();

  // Duplicate START for same running session is harmless.
  if (
      existing.streaming &&
      existing.sessionId == sessionId
  ) {

    Serial.printf(
        "[CONTROL] duplicate START ignored | session=0x%08lX\n",
        static_cast<unsigned long>(
            sessionId
        )
    );

    return true;
  }

  // Do not silently replace an active session.
  if (
      existing.streaming &&
      existing.sessionId != sessionId
  ) {

    Serial.printf(
        "[CONTROL] START rejected | active=0x%08lX requested=0x%08lX\n",
        static_cast<unsigned long>(
            existing.sessionId
        ),
        static_cast<unsigned long>(
            sessionId
        )
    );

    return false;
  }

  // Ensure clean boundary from any previous session.
  if (
      !stopStreaming()
  ) {

    return false;
  }

  packetSequence =
      0;

  flushAudioQueue();

  flushTelemetryQueue();

  setRuntimeState(
      false,
      sessionId,
      0
  );

  // ================================================================
  // SESSION MARKER
  // ================================================================

  digitalWrite(
      PIN_SYNC,
      LOW
  );

  delay(
      SYNC_RESET_LOW_MS
  );

  digitalWrite(
      PIN_SYNC,
      HIGH
  );

  const uint32_t syncEdgeMicros =
      micros();

  // Slaves are already armed by the laptop before Node 1 receives START.
  //
  // This brief delay gives them time to observe GPIO27 before the
  // master begins BCLK/WS generation.
  delay(
      SYNC_SLAVE_ARM_MS
  );

  // ================================================================
  // START I2S ON DEDICATED AUDIO TASK
  // ================================================================

  if (
      !requestAudioStart(
          sessionId
      )
  ) {

    digitalWrite(
        PIN_SYNC,
        LOW
    );

    setRuntimeState(
        false,
        0,
        0
    );

    Serial.println(
        "[SYSTEM] failed to start audio acquisition"
    );

    return false;
  }

  // ================================================================
  // GUARANTEE MINIMUM SYNC PULSE WIDTH
  // ================================================================

  const uint32_t minimumPulseUs =
      SYNC_MIN_HIGH_MS * 1000UL;

  const uint32_t elapsedPulseUs =
      micros() - syncEdgeMicros;

  if (
      elapsedPulseUs < minimumPulseUs
  ) {

    delayMicroseconds(
        minimumPulseUs
        - elapsedPulseUs
    );
  }

  digitalWrite(
      PIN_SYNC,
      LOW
  );

  // ================================================================
  // SEND SYNC METADATA
  // ================================================================
  //
  // networkTask is blocked inside startStreaming() right now.
  //
  // Therefore captured AUDIO may already be queued, but the SYNC packet
  // is transmitted before networkTask begins draining audioQueue.
  // ================================================================

  ++syncId;

  StatisticsSnapshot stats =
      snapshotStatistics();

  SyncPayload payload{};

  payload.sessionId =
      sessionId;

  payload.syncId =
      syncId;

  payload.sampleIndex =
      0;

  payload.localMicros =
      syncEdgeMicros;

  if (
      !sendPacket(
          PKT_SYNC,
          sessionId,
          0,
          syncEdgeMicros,
          stats.i2sErrors,
          &payload,
          sizeof(payload),
          currentHealthFlags()
      )
  ) {

    Serial.println(
        "[SYNC] transmission failed; stopping acquisition"
    );

    stopStreaming();

    return false;
  }

  Serial.printf(
      "[SYNC] session=0x%08lX | sync=%lu | sampleIndex=0\n",
      static_cast<unsigned long>(
          sessionId
      ),
      static_cast<unsigned long>(
          syncId
      )
  );

  return true;
}


// ======================================================================
// AUDIO CAPTURE TASK
// ======================================================================
//
// Core 0
// Priority 4
//
// This task exclusively owns:
//
//     I2S.begin()
//     I2S.readBytes()
//     I2S.end()
//
// Therefore no second core can terminate I2S while a DMA read is
// executing.
// ======================================================================


void audioCaptureTask(
    void*
) {

  static int32_t raw[
      RAW_VALUES_PER_BLOCK
  ];

  bool locallyRunning =
      false;

  uint32_t localSessionId =
      0;

  AudioControlMessage control{};

  AudioBlock block{};

  while (
      true
  ) {

    // ================================================================
    // IDLE: WAIT FOR START
    // ================================================================

    if (
        !locallyRunning
    ) {

      if (
          xQueueReceive(
              audioControlQueue,
              &control,
              portMAX_DELAY
          ) != pdTRUE
      ) {

        continue;
      }

      if (
          control.action == AUDIO_CONTROL_STOP
      ) {

        audioTaskStopHardware();

        continue;
      }

      if (
          control.action != AUDIO_CONTROL_START ||
          control.sessionId == 0
      ) {

        continue;
      }

      xEventGroupClearBits(
          audioEvents,
          AUDIO_EVENT_FAILED
              | AUDIO_EVENT_STOPPED
      );

      setRuntimeState(
          false,
          control.sessionId,
          0
      );

      if (
          !audioSensorBeginMaster()
      ) {

        setRuntimeState(
            false,
            0,
            0
        );

        xEventGroupSetBits(
            audioEvents,
            AUDIO_EVENT_FAILED
                | AUDIO_EVENT_STOPPED
        );

        continue;
      }

      localSessionId =
          control.sessionId;

      locallyRunning =
          true;

      setRuntimeState(
          true,
          localSessionId,
          0
      );

      xEventGroupSetBits(
          audioEvents,
          AUDIO_EVENT_STARTED
      );

      continue;
    }

    // ================================================================
    // CHECK CONTROL BEFORE NEXT DMA BLOCK
    // ================================================================

    if (
        xQueueReceive(
            audioControlQueue,
            &control,
            0
        ) == pdTRUE
    ) {

      if (
          control.action == AUDIO_CONTROL_STOP
      ) {

        audioTaskStopHardware();

        locallyRunning =
            false;

        localSessionId =
            0;

        continue;
      }

      // Defensive duplicate START handling.
      if (
          control.action == AUDIO_CONTROL_START &&
          control.sessionId == localSessionId
      ) {

        xEventGroupSetBits(
            audioEvents,
            AUDIO_EVENT_STARTED
        );
      }
    }

    // ================================================================
    // BLOCKING I2S CAPTURE
    // ================================================================

    size_t bytesRead =
        audioSensorReadRaw(
            raw,
            RAW_VALUES_PER_BLOCK
        );

    const int i2sLastError =
        I2S.lastError();

    // ================================================================
    // STOP MAY HAVE ARRIVED DURING READ
    // ================================================================

    if (
        xQueueReceive(
            audioControlQueue,
            &control,
            0
        ) == pdTRUE
    ) {

      if (
          control.action == AUDIO_CONTROL_STOP
      ) {

        // Discard this last block at the requested session boundary.
        audioTaskStopHardware();

        locallyRunning =
            false;

        localSessionId =
            0;

        continue;
      }
    }

    RuntimeSnapshot runtime =
        snapshotRuntimeState();

    if (
        !runtime.streaming ||
        runtime.sessionId != localSessionId
    ) {

      continue;
    }

    // ================================================================
    // I2S ERROR
    // ================================================================

    if (
        bytesRead == 0
    ) {

      const uint32_t errorCount =
          incrementI2SErrors();

      setClockFault(
          true
      );

      Serial.printf(
          "[I2S] read failure #%lu | code=%d\n",
          static_cast<unsigned long>(
              errorCount
          ),
          i2sLastError
      );

      vTaskDelay(
          pdMS_TO_TICKS(1)
      );

      continue;
    }

    // ================================================================
    // FRAME VALIDATION
    // ================================================================

    constexpr size_t BYTES_PER_I2S_FRAME =
        sizeof(int32_t) * 2;

    uint8_t blockFlags =
        FLAG_NONE;

    if (
        bytesRead % BYTES_PER_I2S_FRAME != 0
    ) {

      incrementI2SErrors();

      setClockFault(
          true
      );

      blockFlags |=
          FLAG_CLOCK_FAULT;
    }

    size_t frames =
        bytesRead
        / BYTES_PER_I2S_FRAME;

    if (
        frames > FRAMES_PER_BLOCK
    ) {

      frames =
          FRAMES_PER_BLOCK;
    }

    if (
        frames == 0
    ) {

      incrementI2SErrors();

      setClockFault(
          true
      );

      continue;
    }

    if (
        frames != FRAMES_PER_BLOCK
    ) {

      incrementI2SErrors();

      setClockFault(
          true
      );

      blockFlags |=
          FLAG_CLOCK_FAULT;

    } else {

      setClockFault(
          false
      );
    }

    // ================================================================
    // ATOMIC SAMPLE TIMELINE
    // ================================================================

    uint64_t blockSampleIndex =
        0;

    if (
        !reserveSampleRange(
            localSessionId,
            static_cast<uint16_t>(
                frames
            ),
            blockSampleIndex
        )
    ) {

      continue;
    }

    StatisticsSnapshot stats =
        snapshotStatistics();

    block.sessionId =
        localSessionId;

    block.sampleIndex =
        blockSampleIndex;

    block.localMicros =
        micros();

    block.i2sErrorCountAtCapture =
        stats.i2sErrors;

    block.frameCount =
        static_cast<uint16_t>(
            frames
        );

    block.flags =
        blockFlags;

    // ================================================================
    // SENSOR-SPECIFIC RAW DATA -> STANDARD PCM16
    // ================================================================

    for (
        size_t frame = 0;
        frame < frames;
        ++frame
    ) {

      const int32_t sensorWord =
          audioSensorFrameWord(
              raw,
              frame
          );

      const int16_t pcm =
          audioSensorWordToPCM16(
              sensorWord
          );

      block.pcm[frame] =
          pcm;

      if (
          pcm >= 32760 ||
          pcm <= -32760
      ) {

        block.flags |=
            FLAG_CLIPPED;
      }
    }

    // ================================================================
    // NON-BLOCKING HANDOFF TO NETWORK CORE
    // ================================================================

    if (
        xQueueSend(
            audioQueue,
            &block,
            0
        ) != pdTRUE
    ) {

      // sampleIndex is intentionally NOT rewound.
      //
      // The resulting gap is visible to the laptop and therefore does
      // not masquerade as continuous audio.
      incrementDroppedAudioBlocks();
    }
  }
}


// ======================================================================
// ENVIRONMENT TASK
// ======================================================================
//
// Core 1
// Priority 1
//
// Runs independently from audio capture.
// ======================================================================


void environmentTask(
    void*
) {

  uint32_t lastRetry =
      0;

  while (
      true
  ) {

    // ================================================================
    // SENSOR RECONNECT / RECOVERY
    // ================================================================

    if (
        !environmentSensorAvailable
    ) {

      uint32_t now =
          millis();

      if (
          now - lastRetry
          >= ENV_RETRY_INTERVAL_MS
      ) {

        lastRetry =
            now;

        environmentSensorBegin();
      }

      vTaskDelay(
          pdMS_TO_TICKS(
              ENV_INTERVAL_MS
          )
      );

      continue;
    }

    // ================================================================
    // ONLY OBSERVE ENVIRONMENT DURING ACQUISITION
    // ================================================================

    RuntimeSnapshot before =
        snapshotRuntimeState();

    if (
        !before.streaming ||
        before.sessionId == 0
    ) {

      vTaskDelay(
          pdMS_TO_TICKS(
              ENV_INTERVAL_MS
          )
      );

      continue;
    }

    EnvironmentReading reading{};

    if (
        !environmentSensorRead(
            reading
        )
    ) {

      Serial.println(
          "[ENV] invalid environmental reading"
      );

      vTaskDelay(
          pdMS_TO_TICKS(
              ENV_INTERVAL_MS
          )
      );

      continue;
    }

    RuntimeSnapshot after =
        snapshotRuntimeState();

    // Do not attach a reading taken across a session transition.
    if (
        !after.streaming ||
        after.sessionId != before.sessionId
    ) {

      vTaskDelay(
          pdMS_TO_TICKS(
              ENV_INTERVAL_MS
          )
      );

      continue;
    }

    EnvironmentPayload payload{};

    payload.temperatureC =
        reading.temperatureC;

    payload.humidityPercent =
        reading.humidityPercent;

    payload.pressureHpa =
        reading.pressureHpa;

    StatisticsSnapshot stats =
        snapshotStatistics();

    TelemetryMessage message{};

    message.type =
        PKT_ENVIRONMENT;

    message.sessionId =
        after.sessionId;

    message.sampleIndex =
        after.sampleIndex;

    message.localMicros =
        micros();

    message.i2sErrorCountAtCapture =
        stats.i2sErrors;

    message.payloadLength =
        sizeof(payload);

    message.flags =
        currentHealthFlags();

    memcpy(
        message.payload,
        &payload,
        sizeof(payload)
    );

    xQueueSend(
        telemetryQueue,
        &message,
        0
    );

    vTaskDelay(
        pdMS_TO_TICKS(
            ENV_INTERVAL_MS
        )
    );
  }
}


// ======================================================================
// QUEUE HEARTBEAT
// ======================================================================


void queueHeartbeat() {

  RuntimeSnapshot runtime =
      snapshotRuntimeState();

  StatisticsSnapshot stats =
      snapshotStatistics();

  HeartbeatPayload heartbeat{};

  heartbeat.uptimeSeconds =
      millis() / 1000UL;

  if (
      WiFi.status() == WL_CONNECTED
  ) {

    heartbeat.wifiRSSI =
        WiFi.RSSI();

  } else {

    heartbeat.wifiRSSI =
        0;
  }

  heartbeat.freeHeap =
      ESP.getFreeHeap();

  heartbeat.droppedAudioBlocks =
      stats.droppedAudioBlocks;

  heartbeat.transmittedAudioBlocks =
      stats.transmittedAudioBlocks;

  heartbeat.i2sErrors =
      stats.i2sErrors;

  if (
      audioQueue != nullptr
  ) {

    heartbeat.audioQueueDepth =
        static_cast<uint16_t>(
            uxQueueMessagesWaiting(
                audioQueue
            )
        );

  } else {

    heartbeat.audioQueueDepth =
        0;
  }

  heartbeat.streaming =
      runtime.streaming
          ? 1
          : 0;

  heartbeat.bmeAvailable =
      environmentSensorAvailable
          ? 1
          : 0;

  TelemetryMessage message{};

  message.type =
      PKT_HEARTBEAT;

  message.sessionId =
      runtime.sessionId;

  message.sampleIndex =
      runtime.sampleIndex;

  message.localMicros =
      micros();

  message.i2sErrorCountAtCapture =
      stats.i2sErrors;

  message.payloadLength =
      sizeof(heartbeat);

  message.flags =
      currentHealthFlags();

  memcpy(
      message.payload,
      &heartbeat,
      sizeof(heartbeat)
  );

  xQueueSend(
      telemetryQueue,
      &message,
      0
  );
}


// ======================================================================
// PROCESS LAPTOP COMMANDS
// ======================================================================


void processCommands() {

  while (
      tcpClient.connected() &&
      tcpClient.available()
          >= static_cast<int>(
              sizeof(ControlFrame)
          )
  ) {

    ControlFrame frame{};

    size_t received =
        tcpClient.readBytes(
            reinterpret_cast<char*>(
                &frame
            ),
            sizeof(frame)
        );

    if (
        received != sizeof(frame)
    ) {

      Serial.println(
          "[CONTROL] incomplete frame"
      );

      return;
    }

    if (
        frame.magic != CONTROL_MAGIC ||
        frame.version != CONTROL_VERSION
    ) {

      Serial.println(
          "[CONTROL] invalid frame"
      );

      continue;
    }

    switch (
        frame.command
    ) {

      // ==============================================================
      // START
      // ==============================================================

      case CMD_START:

        startStreaming(
            frame.sessionId
        );

        break;


      // ==============================================================
      // STOP
      // ==============================================================

      case CMD_STOP: {

        RuntimeSnapshot runtime =
            snapshotRuntimeState();

        // Ignore stale STOP for a different running session.
        if (
            runtime.streaming &&
            frame.sessionId != 0 &&
            frame.sessionId != runtime.sessionId
        ) {

          Serial.printf(
              "[CONTROL] stale STOP ignored | active=0x%08lX requested=0x%08lX\n",
              static_cast<unsigned long>(
                  runtime.sessionId
              ),
              static_cast<unsigned long>(
                  frame.sessionId
              )
          );

          break;
        }

        stopStreaming();

        break;
      }


      // ==============================================================
      // PING
      // ==============================================================

      case CMD_PING:

        queueHeartbeat();

        break;


      // ==============================================================
      // UNKNOWN
      // ==============================================================

      default:

        Serial.printf(
            "[CONTROL] unknown command 0x%02X\n",
            frame.command
        );

        break;
    }
  }
}


// ======================================================================
// NETWORK TASK
// ======================================================================
//
// Core 1
// Priority 3
//
// Responsible for:
//
//     Wi-Fi recovery
//     TCP recovery
//     protocol serialization
//     commands
//     audio transmission
//     telemetry transmission
//
// Audio acquisition continues independently on Core 0 even if Wi-Fi
// temporarily reconnects.
// ======================================================================


void networkTask(
    void*
) {

  AudioBlock audio{};

  TelemetryMessage telemetry{};

  bool helloSent =
      false;

  uint32_t lastHeartbeat =
      0;

  while (
      true
  ) {

    // ================================================================
    // NETWORK RECOVERY
    // ================================================================

    if (
        WiFi.status() != WL_CONNECTED
    ) {

      helloSent =
          false;

      ensureWiFi();
    }

    if (
        !tcpClient.connected()
    ) {

      helloSent =
          false;

      ensureLaptopConnection();
    }

    // ================================================================
    // HELLO IS FIRST PACKET AFTER TCP CONNECTION
    // ================================================================

    if (
        !helloSent
    ) {

      if (
          sendHello()
      ) {

        helloSent =
            true;

      } else {

        tcpClient.stop();

        continue;
      }
    }

    // ================================================================
    // CONTROL
    // ================================================================

    processCommands();

    // ================================================================
    // AUDIO
    // ================================================================

    if (
        xQueueReceive(
            audioQueue,
            &audio,
            0
        ) == pdTRUE
    ) {

      RuntimeSnapshot runtime =
          snapshotRuntimeState();

      // Never relabel an old queued block as belonging to another
      // session.
      if (
          runtime.streaming &&
          runtime.sessionId == audio.sessionId
      ) {

        bool sent =
            sendPacket(
                PKT_AUDIO,
                audio.sessionId,
                audio.sampleIndex,
                audio.localMicros,
                audio.i2sErrorCountAtCapture,
                audio.pcm,
                static_cast<uint32_t>(
                    audio.frameCount
                    * sizeof(int16_t)
                ),
                static_cast<uint8_t>(
                    audio.flags
                    | currentHealthFlags()
                )
            );

        if (
            sent
        ) {

          incrementTransmittedAudioBlocks();

        } else {

          incrementDroppedAudioBlocks();

          helloSent =
              false;

          tcpClient.stop();
        }
      }
    }

    // ================================================================
    // TELEMETRY
    // ================================================================

    if (
        xQueueReceive(
            telemetryQueue,
            &telemetry,
            0
        ) == pdTRUE
    ) {

      RuntimeSnapshot runtime =
          snapshotRuntimeState();

      bool staleEnvironment =
          telemetry.type == PKT_ENVIRONMENT &&
          (
              !runtime.streaming ||
              telemetry.sessionId != runtime.sessionId
          );

      if (
          !staleEnvironment
      ) {

        bool sent =
            sendPacket(
                telemetry.type,
                telemetry.sessionId,
                telemetry.sampleIndex,
                telemetry.localMicros,
                telemetry.i2sErrorCountAtCapture,
                telemetry.payload,
                telemetry.payloadLength,
                static_cast<uint8_t>(
                    telemetry.flags
                    | currentHealthFlags()
                )
            );

        if (
            !sent
        ) {

          helloSent =
              false;

          tcpClient.stop();
        }
      }
    }

    // ================================================================
    // PERIODIC HEARTBEAT
    // ================================================================

    uint32_t now =
        millis();

    if (
        now - lastHeartbeat
        >= HEARTBEAT_INTERVAL_MS
    ) {

      lastHeartbeat =
          now;

      queueHeartbeat();
    }

    vTaskDelay(
        pdMS_TO_TICKS(1)
    );
  }
}


// ======================================================================
// SETUP
// ======================================================================


void setup() {

  Serial.begin(
      115200
  );

  delay(
      1000
  );

  Serial.println();

  Serial.println(
      "========================================================"
  );

  Serial.println(
      "Wildlife Soundscape MASTER v1.4.0 / protocol v4"
  );

  Serial.println(
      "Target: ESP32 DevKit V1 / ESP32-WROOM-32"
  );

  Serial.printf(
      "Audio sensor: %s\n",
      currentAudioSensorName()
  );

  Serial.printf(
      "Environment sensor: %s\n",
      currentEnvironmentSensorName()
  );

  Serial.println(
      "========================================================"
  );


  // ==================================================================
  // SYNC
  // ==================================================================

  pinMode(
      PIN_SYNC,
      OUTPUT
  );

  digitalWrite(
      PIN_SYNC,
      LOW
  );


  // ==================================================================
  // FREERTOS RESOURCES
  // ==================================================================

  audioQueue =
      xQueueCreate(
          AUDIO_QUEUE_DEPTH,
          sizeof(AudioBlock)
      );


  telemetryQueue =
      xQueueCreate(
          TELEMETRY_QUEUE_DEPTH,
          sizeof(TelemetryMessage)
      );


  audioControlQueue =
      xQueueCreate(
          AUDIO_CONTROL_QUEUE_DEPTH,
          sizeof(AudioControlMessage)
      );


  audioEvents =
      xEventGroupCreate();


  if (
      audioQueue == nullptr ||
      telemetryQueue == nullptr ||
      audioControlQueue == nullptr ||
      audioEvents == nullptr
  ) {

    while (
        true
    ) {

      Serial.println(
          "[FATAL] FreeRTOS allocation failed"
      );

      delay(
          1000
      );
    }
  }


  // Audio hardware initially stopped.
  xEventGroupSetBits(
      audioEvents,
      AUDIO_EVENT_STOPPED
  );


  // ==================================================================
  // ENVIRONMENT SENSOR
  // ==================================================================

  Wire.begin(
      PIN_BME_SDA,
      PIN_BME_SCL,
      100000
  );


  environmentSensorBegin();


  // ==================================================================
  // WI-FI
  // ==================================================================
  //
  // Non-blocking startup:
  //
  // networkTask handles actual connection/reconnection.
  // ==================================================================

  setupWiFi();


  // ==================================================================
  // TASKS
  // ==================================================================

  BaseType_t audioTaskCreated =
      xTaskCreatePinnedToCore(
          audioCaptureTask,
          "AudioCapture",
          4096,
          nullptr,
          4,
          nullptr,
          0
      );


  BaseType_t networkTaskCreated =
      xTaskCreatePinnedToCore(
          networkTask,
          "Network",
          6144,
          nullptr,
          3,
          nullptr,
          1
      );


  BaseType_t environmentTaskCreated =
      xTaskCreatePinnedToCore(
          environmentTask,
          "Environment",
          3072,
          nullptr,
          1,
          nullptr,
          1
      );


  if (
      audioTaskCreated != pdPASS ||
      networkTaskCreated != pdPASS ||
      environmentTaskCreated != pdPASS
  ) {

    while (
        true
    ) {

      Serial.println(
          "[FATAL] task creation failed"
      );

      delay(
          1000
      );
    }
  }


  Serial.println(
      "[SYSTEM] MASTER READY"
  );
}


// ======================================================================
// ARDUINO LOOP
// ======================================================================


void loop() {

  // Application logic is fully task-driven.
  vTaskDelay(
      pdMS_TO_TICKS(
          1000
      )
  );
}