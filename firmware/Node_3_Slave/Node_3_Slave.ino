/*
  Wildlife Soundscape Mapping & Behavior Analysis System
  Node 3 SLAVE firmware v1.4.0 / wire protocol v4

  Target
  ------
  ESP32 DevKit V1
  ESP32-WROOM-32
  Arduino-ESP32 3.x


  CURRENT AUDIO SENSOR
  --------------------
  INMP441 digital I2S MEMS microphone

      BCLK = GPIO26  <- shared from Node 1
      WS   = GPIO25  <- shared from Node 1
      SD   = GPIO33  <- local microphone
      L/R  = GND
      VDD  = 3V3


  SYNCHRONIZATION
  ---------------
  GPIO27 <- SYNC from Node 1


  IMPORTANT
  ---------
  Node 3 is an I2S SLAVE.

  Node 3 MUST NOT generate its own:

      BCLK
      WS / LRCLK

  Node 1 generates the physical audio clocks.

  Therefore:

      Node 1 BCLK/WS
              ↓
      Node 1 + Node 2 + Node 3
              ↓
      common sample-rate clock

  GPIO27 SYNC is only used to mark the start of a laptop-controlled
  acquisition session.

  GPIO27 is NOT used as the TDOA sample clock.


  LAPTOP LOCALIZATION
  -------------------
  TDOA uses:

      shared-clock sampleIndex
      +
      GCC-PHAT waveform delay estimation


  SENSOR REPLACEMENT
  ------------------
  Sensor-specific audio logic is isolated inside the AUDIO SENSOR
  ADAPTER section.

  The remainder of Node 3 expects standardized audio:

      mono
      PCM16
      48 kHz

  Therefore another compatible I2S microphone can later replace the
  INMP441 with limited changes to the adapter.
*/


// ======================================================================
// INCLUDES
// ======================================================================


#include <Arduino.h>

#include <WiFi.h>

#include "driver/i2s_std.h"

#include <math.h>

#include <string.h>


#include "freertos/FreeRTOS.h"

#include "freertos/task.h"

#include "freertos/queue.h"

#include "freertos/event_groups.h"


// ======================================================================
// NODE IDENTITY
// ======================================================================


#define NODE_ID 3

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


// External clocks from Node 1.
constexpr int PIN_BCLK =
    26;


constexpr int PIN_WS =
    25;


// Local microphone SD line.
constexpr int PIN_MIC_DATA =
    33;


// Session/start marker from Node 1.
constexpr int PIN_SYNC =
    27;


// ======================================================================
// STANDARD AUDIO FORMAT
// ======================================================================


constexpr uint32_t SAMPLE_RATE =
    48000;


constexpr uint16_t FRAMES_PER_BLOCK =
    1024;


// INMP441 is received using stereo I2S frames.
//
// One frame:
//
//     LEFT  int32
//     RIGHT int32
//
// INMP441 L/R is grounded, therefore its sample is in LEFT.
constexpr uint16_t RAW_VALUES_PER_BLOCK =
    FRAMES_PER_BLOCK * 2;


constexpr uint8_t CURRENT_MIC_SLOT_INDEX =
    0;


constexpr int16_t SYNC_ALIGN_TOLERANCE_SAMPLES =
    10;


// ======================================================================
// QUEUES
// ======================================================================


constexpr uint8_t AUDIO_QUEUE_DEPTH =
    6;


constexpr uint8_t TELEMETRY_QUEUE_DEPTH =
    6;


constexpr uint8_t AUDIO_CONTROL_QUEUE_DEPTH =
    4;


// ======================================================================
// INTERVALS
// ======================================================================


constexpr uint32_t HEARTBEAT_INTERVAL_MS =
    5000;


constexpr uint32_t SYNC_HEALTH_INTERVAL_MS =
    100;


constexpr uint32_t CLOCK_STALL_MS =
    1000;


// Master normally returns SYNC LOW after ~10 ms.
//
// A continuously HIGH line for 1 second is considered abnormal.
constexpr uint32_t SYNC_STUCK_HIGH_US =
    1000000UL;


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
// I2S READ CONTROL
// ======================================================================
//
// Slave I2S depends on the master clocks.
//
// If Node 1 disappears, a permanently blocking slave read would make
// clean session shutdown difficult.
//
// I2SClass inherits Stream::setTimeout(), and ESP32's own ESP_SR code
// uses setTimeout() before readBytes().
//
// This gives the audio task opportunities to:
//     process STOP
//     detect clock loss
//     recover cleanly
// ======================================================================


constexpr uint32_t I2S_READ_TIMEOUT_MS =
    250;


constexpr uint32_t AUDIO_CONTROL_TIMEOUT_MS =
    1500;


// ======================================================================
// PROTOCOL CONSTANTS
// ======================================================================


constexpr uint32_t PACKET_MAGIC =
    0x574C5343UL;


// WLSC


constexpr uint8_t PROTOCOL_VERSION =
    4;


constexpr uint16_t CONTROL_MAGIC =
    0xC0DE;


constexpr uint8_t CONTROL_VERSION =
    1;


// ======================================================================
// FLAGS
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
// CONTROL COMMANDS
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

  AUDIO_CONTROL_ARM =
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
// HELLO
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
// SYNC
// ======================================================================


struct __attribute__((packed)) SyncPayload {

  uint32_t sessionId;

  uint32_t syncId;

  uint64_t sampleIndex;

  uint32_t localMicros;
};


// ======================================================================
// SLAVE HEARTBEAT
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

  uint8_t syncReceived;

  uint8_t clockHealthy;
};


// ======================================================================
// AUDIO BLOCK
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
// TELEMETRY MESSAGE
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

  bool armed;

  bool streaming;

  bool syncReceived;

  bool clockHealthy;

  bool clockFaultLatched;

  uint32_t sessionId;

  uint64_t sampleIndex;

  uint32_t syncId;

  uint32_t lastSyncMicros;
};


// ======================================================================
// STATISTICS SNAPSHOT
// ======================================================================


struct StatisticsSnapshot {

  uint32_t droppedAudioBlocks;

  uint32_t transmittedAudioBlocks;

  uint32_t i2sErrors;
};


// ======================================================================
// PENDING SYNC REPORT
// ======================================================================


struct PendingSyncReport {

  uint32_t sessionId;

  uint32_t syncId;

  uint32_t localMicros;
};


// ======================================================================
// PROTOCOL SIZE VALIDATION
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
    sizeof(SyncPayload) == 20,
    "SyncPayload size mismatch"
);


static_assert(
    sizeof(HeartbeatPayload) == 29,
    "Slave heartbeat size mismatch"
);


static_assert(
    sizeof(SyncPayload) <= 64,
    "Telemetry payload buffer too small"
);


static_assert(
    sizeof(HeartbeatPayload) <= 64,
    "Telemetry payload buffer too small"
);


// ======================================================================
// HARDWARE OBJECTS
// ======================================================================


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


constexpr EventBits_t AUDIO_EVENT_ARMED =
    (1U << 0);


constexpr EventBits_t AUDIO_EVENT_STOPPED =
    (1U << 1);


constexpr EventBits_t AUDIO_EVENT_FAILED =
    (1U << 2);


// ======================================================================
// SHARED STATE LOCK
// ======================================================================
//
// ESP32-WROOM-32 is a 32-bit MCU.
//
// sampleIndex is uint64_t.
//
// State therefore needs explicit protection when accessed from:
//
//     Core 0
//     Core 1
//     GPIO ISR
// ======================================================================


portMUX_TYPE stateMux =
    portMUX_INITIALIZER_UNLOCKED;


// ======================================================================
// SHARED RUNTIME STATE
// ======================================================================


bool runtimeArmed =
    false;


bool runtimeStreaming =
    false;


bool runtimeSyncReceived =
    false;


bool runtimeClockHealthy =
    false;


bool runtimeClockFaultLatched =
    false;


uint32_t currentSessionId =
    0;


uint64_t totalSampleCount =
    0;


uint32_t currentSyncId =
    0;


uint32_t lastSyncLocalMicros =
    0;


bool syncReportPending =
    false;


// ======================================================================
// STATISTICS
// ======================================================================


uint32_t droppedAudioBlocks =
    0;


uint32_t transmittedAudioBlocks =
    0;


uint32_t i2sErrors =
    0;


// ======================================================================
// NETWORK-ONLY STATE
// ======================================================================


uint32_t packetSequence =
    0;


// ======================================================================
// AUDIO DRIVER STATE
// ======================================================================
//
// Only audioCaptureTask accesses this value and owns the I2S peripheral.
// ======================================================================


bool audioDriverStarted =
    false;


i2s_chan_handle_t audioRxChannel =
    nullptr;


esp_err_t audioLastError =
    ESP_OK;


// ======================================================================
// CURRENT SENSOR NAME
// ======================================================================


const char* currentAudioSensorName() {

  return "INMP441";
}


// ======================================================================
// RUNTIME SNAPSHOT
// ======================================================================


RuntimeSnapshot snapshotRuntimeState() {

  RuntimeSnapshot snapshot{};

  portENTER_CRITICAL(
      &stateMux
  );

  snapshot.armed =
      runtimeArmed;

  snapshot.streaming =
      runtimeStreaming;

  snapshot.syncReceived =
      runtimeSyncReceived;

  snapshot.clockHealthy =
      runtimeClockHealthy;

  snapshot.clockFaultLatched =
      runtimeClockFaultLatched;

  snapshot.sessionId =
      currentSessionId;

  snapshot.sampleIndex =
      totalSampleCount;

  snapshot.syncId =
      currentSyncId;

  snapshot.lastSyncMicros =
      lastSyncLocalMicros;

  portEXIT_CRITICAL(
      &stateMux
  );

  return snapshot;
}


// ======================================================================
// SET IDLE
// ======================================================================


void setRuntimeIdle(
    bool preserveClockFault = false
) {

  portENTER_CRITICAL(
      &stateMux
  );

  runtimeArmed =
      false;

  runtimeStreaming =
      false;

  runtimeSyncReceived =
      false;

  runtimeClockHealthy =
      false;

  if (
      !preserveClockFault
  ) {

    runtimeClockFaultLatched =
        false;
  }

  currentSessionId =
      0;

  totalSampleCount =
      0;

  syncReportPending =
      false;

  portEXIT_CRITICAL(
      &stateMux
  );
}


// ======================================================================
// MARK SLAVE ARMED
// ======================================================================


void setRuntimeArmed(
    uint32_t sessionId
) {

  portENTER_CRITICAL(
      &stateMux
  );

  runtimeArmed =
      true;

  runtimeStreaming =
      false;

  runtimeSyncReceived =
      false;

  runtimeClockHealthy =
      false;

  runtimeClockFaultLatched =
      false;

  currentSessionId =
      sessionId;

  totalSampleCount =
      0;

  syncReportPending =
      false;

  portEXIT_CRITICAL(
      &stateMux
  );
}


// ======================================================================
// CLOCK HEALTH
// ======================================================================


void setClockHealthy(
    bool healthy
) {

  portENTER_CRITICAL(
      &stateMux
  );

  runtimeClockHealthy =
      healthy;

  portEXIT_CRITICAL(
      &stateMux
  );
}


void latchClockFault() {

  portENTER_CRITICAL(
      &stateMux
  );

  runtimeClockHealthy =
      false;

  runtimeClockFaultLatched =
      true;

  portEXIT_CRITICAL(
      &stateMux
  );
}


// ======================================================================
// SAMPLE RANGE
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
      runtimeArmed &&
      runtimeStreaming &&
      runtimeSyncReceived &&
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

  portEXIT_CRITICAL(
      &stateMux
  );

  return snapshot;
}


// ======================================================================
// CONSUME PENDING SYNC REPORT
// ======================================================================


bool consumePendingSyncReport(
    PendingSyncReport& report
) {

  bool available =
      false;

  portENTER_CRITICAL(
      &stateMux
  );

  if (
      syncReportPending
  ) {

    report.sessionId =
        currentSessionId;

    report.syncId =
        currentSyncId;

    report.localMicros =
        lastSyncLocalMicros;

    syncReportPending =
        false;

    available =
        true;
  }

  portEXIT_CRITICAL(
      &stateMux
  );

  return available;
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
// TCP SEND
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
// SEND PACKET
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

  RuntimeSnapshot runtime =
      snapshotRuntimeState();

  if (
      runtime.clockFaultLatched
  ) {

    flags |=
        FLAG_CLOCK_FAULT;
  }

  if (
      audioQueue != nullptr
  ) {

    const UBaseType_t depth =
        uxQueueMessagesWaiting(
            audioQueue
        );

    if (
        depth * 5U
        >= AUDIO_QUEUE_DEPTH * 4U
    ) {

      flags |=
          FLAG_QUEUE_CONGESTED;
    }
  }

  return flags;
}


// ======================================================================
// WI-FI
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
// LAPTOP CONNECTION
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
// CURRENT:
//     INMP441
//
// INPUT:
//     external BCLK/WS generated by Node 1
//
// STANDARD OUTPUT:
//     48 kHz
//     mono
//     PCM16
//
// A future I2S microphone should require changes mainly here.
// ======================================================================


// ======================================================================
// START SLAVE AUDIO SENSOR
// ======================================================================


bool audioSensorBeginSlave() {

  if (
      audioDriverStarted
  ) {

    i2s_channel_disable(
        audioRxChannel
    );

    i2s_del_channel(
        audioRxChannel
    );

    audioRxChannel =
        nullptr;

    audioDriverStarted =
        false;
  }

  i2s_chan_config_t channelConfig =
      I2S_CHANNEL_DEFAULT_CONFIG(
          I2S_NUM_AUTO,
          I2S_ROLE_SLAVE
      );

  audioLastError =
      i2s_new_channel(
          &channelConfig,
          nullptr,
          &audioRxChannel
      );

  i2s_std_config_t standardConfig = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(
          SAMPLE_RATE
      ),
      .slot_cfg = I2S_STD_PHILIP_SLOT_DEFAULT_CONFIG(
          I2S_DATA_BIT_WIDTH_32BIT,
          I2S_SLOT_MODE_STEREO
      ),
      .gpio_cfg = {
          .mclk = I2S_GPIO_UNUSED,
          .bclk = static_cast<gpio_num_t>(
              PIN_BCLK
          ),
          .ws = static_cast<gpio_num_t>(
              PIN_WS
          ),
          .dout = I2S_GPIO_UNUSED,
          .din = static_cast<gpio_num_t>(
              PIN_MIC_DATA
          ),
          .invert_flags = {
              .mclk_inv = false,
              .bclk_inv = false,
              .ws_inv = false,
          },
      },
  };

  if (
      audioLastError == ESP_OK
  ) {

    audioLastError =
        i2s_channel_init_std_mode(
            audioRxChannel,
            &standardConfig
        );
  }

  if (
      audioLastError == ESP_OK
  ) {

    audioLastError =
        i2s_channel_enable(
            audioRxChannel
        );
  }

  const bool ok =
      audioLastError == ESP_OK;

  if (
      !ok
  ) {

    const uint32_t errorCount =
        incrementI2SErrors();

    latchClockFault();

    Serial.printf(
        "[I2S] SLAVE start failed | errors=%lu | code=%d\n",
        static_cast<unsigned long>(
            errorCount
        ),
        static_cast<int>(
            audioLastError
        )
    );

    if (
        audioRxChannel != nullptr
    ) {

      i2s_del_channel(
          audioRxChannel
      );

      audioRxChannel =
          nullptr;
    }

    return false;
  }

  audioDriverStarted =
      true;

  Serial.printf(
      "[I2S] SLAVE armed | sensor=%s | waiting for Node 1 BCLK/WS\n",
      currentAudioSensorName()
  );

  return true;
}


// ======================================================================
// STOP AUDIO SENSOR
// ======================================================================


void audioSensorEnd() {

  if (
      !audioDriverStarted
  ) {

    return;
  }

  audioLastError =
      i2s_channel_disable(
          audioRxChannel
      );

  const esp_err_t deleteError =
      i2s_del_channel(
          audioRxChannel
      );

  if (
      audioLastError == ESP_OK
  ) {

    audioLastError =
        deleteError;
  }

  audioRxChannel =
      nullptr;

  audioDriverStarted =
      false;

  Serial.println(
      "[I2S] SLAVE stopped"
  );
}


// ======================================================================
// READ RAW AUDIO
// ======================================================================


size_t audioSensorReadRaw(
    int32_t* destination,
    size_t rawValueCount
) {

  size_t bytesRead =
      0;

  audioLastError =
      i2s_channel_read(
          audioRxChannel,
          destination,
          rawValueCount
              * sizeof(int32_t),
          &bytesRead,
          I2S_READ_TIMEOUT_MS
      );

  return bytesRead;
}


int audioSensorLastError() {

  return static_cast<int>(
      audioLastError
  );
}


// ======================================================================
// SENSOR WORD -> PCM16
// ======================================================================


inline int16_t audioSensorWordToPCM16(
    int32_t rawWord
) {

  return static_cast<int16_t>(
      rawWord >> 16
  );
}


// ======================================================================
// EXTRACT LEFT I2S SLOT
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
// SYNC ISR
// ======================================================================
//
// Extremely small.
//
// No networking.
// No queue allocations.
// No Serial.
//
// It simply anchors this slave's session sample timeline to zero when
// the master's start pulse arrives.
// ======================================================================


void IRAM_ATTR syncISR() {

  const uint32_t localMicros =
      micros();

  portENTER_CRITICAL_ISR(
      &stateMux
  );

  if (
      runtimeArmed &&
      !runtimeSyncReceived &&
      currentSessionId != 0
  ) {

    totalSampleCount =
        0;

    runtimeSyncReceived =
        true;

    runtimeStreaming =
        true;

    // SYNC proves that the marker arrived.
    //
    // Clock health becomes true only after actual BCLK/WS-driven audio
    // frames are received.
    runtimeClockHealthy =
        false;

    ++currentSyncId;

    lastSyncLocalMicros =
        localMicros;

    syncReportPending =
        true;
  }

  portEXIT_CRITICAL_ISR(
      &stateMux
  );
}


// ======================================================================
// SYNC GPIO SETUP
// ======================================================================


void setupSync() {

  pinMode(
      PIN_SYNC,
      INPUT_PULLDOWN
  );

  attachInterrupt(
      digitalPinToInterrupt(
          PIN_SYNC
      ),
      syncISR,
      RISING
  );
}


// ======================================================================
// HELLO
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

  // Node 3 is not the clock master.
  payload.masterNode =
      0;

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
// AUDIO TASK STOP
// ======================================================================


void audioTaskStopHardware() {

  audioSensorEnd();

  setRuntimeIdle(
      false
  );

  xEventGroupClearBits(
      audioEvents,
      AUDIO_EVENT_ARMED
  );

  xEventGroupSetBits(
      audioEvents,
      AUDIO_EVENT_STOPPED
  );
}


// ======================================================================
// REQUEST SLAVE ARM
// ======================================================================


bool requestAudioArm(
    uint32_t sessionId
) {

  AudioControlMessage command{};

  command.action =
      AUDIO_CONTROL_ARM;

  command.sessionId =
      sessionId;

  xEventGroupClearBits(
      audioEvents,
      AUDIO_EVENT_ARMED
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
        "[AUDIO] unable to queue ARM"
    );

    return false;
  }

  EventBits_t result =
      xEventGroupWaitBits(
          audioEvents,
          AUDIO_EVENT_ARMED
              | AUDIO_EVENT_FAILED,
          pdFALSE,
          pdFALSE,
          pdMS_TO_TICKS(
              AUDIO_CONTROL_TIMEOUT_MS
          )
      );

  return (
      result
      & AUDIO_EVENT_ARMED
  );
}


// ======================================================================
// REQUEST AUDIO STOP
// ======================================================================


bool requestAudioStop() {

  RuntimeSnapshot runtime =
      snapshotRuntimeState();

  if (
      !runtime.armed
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
// STOP SESSION
// ======================================================================


bool stopStreaming() {

  bool stopped =
      requestAudioStop();

  flushAudioQueue();

  flushTelemetryQueue();

  setRuntimeIdle(
      false
  );

  if (
      stopped
  ) {

    Serial.println(
        "[SYSTEM] SLAVE stopped"
    );

  } else {

    Serial.println(
        "[SYSTEM] SLAVE STOP timed out"
    );
  }

  return stopped;
}


// ======================================================================
// ARM SESSION
// ======================================================================
//
// Laptop sequence:
//
//     START Node 3
//     START Node 3
//     START Node 1
//
// Node 3 begins its I2S peripheral in SLAVE mode BEFORE the master
// starts BCLK/WS.
// ======================================================================


bool armStreaming(
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

  if (
      existing.armed &&
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

  if (
      existing.armed &&
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

  if (
      !stopStreaming()
  ) {

    return false;
  }

  packetSequence =
      0;

  flushAudioQueue();

  flushTelemetryQueue();

  if (
      !requestAudioArm(
          sessionId
      )
  ) {

    setRuntimeIdle(
        false
    );

    Serial.println(
        "[SYSTEM] unable to arm I2S slave"
    );

    return false;
  }

  Serial.printf(
      "[SYSTEM] ARMED session=0x%08lX | waiting for GPIO27 SYNC\n",
      static_cast<unsigned long>(
          sessionId
      )
  );

  return true;
}


// ======================================================================
// AUDIO CAPTURE TASK
// ======================================================================
//
// Core 0 / highest application priority.
//
// This task exclusively owns:
//
//     I2S.begin()
//     I2S.readBytes()
//     I2S.end()
//
// No Core 1 task tears down the I2S driver while DMA capture is active.
// ======================================================================


void audioCaptureTask(
    void*
) {

  static int32_t raw[
      RAW_VALUES_PER_BLOCK
  ];

  bool locallyArmed =
      false;

  uint32_t localSessionId =
      0;

  AudioControlMessage control{};

  AudioBlock block{};

  while (
      true
  ) {

    // ================================================================
    // IDLE
    // ================================================================

    if (
        !locallyArmed
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
          control.action != AUDIO_CONTROL_ARM ||
          control.sessionId == 0
      ) {

        continue;
      }

      xEventGroupClearBits(
          audioEvents,
          AUDIO_EVENT_FAILED
              | AUDIO_EVENT_STOPPED
      );

      if (
          !audioSensorBeginSlave()
      ) {

        setRuntimeIdle(
            true
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

      locallyArmed =
          true;

      // ISR is permitted to accept SYNC only after the I2S peripheral
      // has successfully entered SLAVE mode.
      setRuntimeArmed(
          localSessionId
      );

      xEventGroupSetBits(
          audioEvents,
          AUDIO_EVENT_ARMED
      );

      continue;
    }

    // ================================================================
    // WAITING FOR MASTER SYNC
    // ================================================================

    RuntimeSnapshot runtime =
        snapshotRuntimeState();

    if (
        !runtime.syncReceived
    ) {

      if (
          xQueueReceive(
              audioControlQueue,
              &control,
              pdMS_TO_TICKS(1)
          ) == pdTRUE
      ) {

        if (
            control.action == AUDIO_CONTROL_STOP
        ) {

          audioTaskStopHardware();

          locallyArmed =
              false;

          localSessionId =
              0;

          continue;
        }
      }

      continue;
    }

    // ================================================================
    // CONTROL BEFORE READ
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

        locallyArmed =
            false;

        localSessionId =
            0;

        continue;
      }
    }

    // ================================================================
    // EXTERNAL-CLOCK I2S READ
    // ================================================================

    size_t bytesRead =
        audioSensorReadRaw(
            raw,
            RAW_VALUES_PER_BLOCK
        );

    const int i2sLastError =
        audioSensorLastError();

    // ================================================================
    // PROCESS STOP IMMEDIATELY AFTER READ/TIMEOUT
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

        locallyArmed =
            false;

        localSessionId =
            0;

        continue;
      }
    }

    runtime =
        snapshotRuntimeState();

    if (
        !runtime.armed ||
        !runtime.syncReceived ||
        runtime.sessionId != localSessionId
    ) {

      continue;
    }

    // ================================================================
    // NO CLOCK / READ FAILURE
    // ================================================================

    if (
        bytesRead == 0
    ) {

      const uint32_t errorCount =
          incrementI2SErrors();

      latchClockFault();

      Serial.printf(
          "[I2S] no data / clock timeout #%lu | code=%d\n",
          static_cast<unsigned long>(
              errorCount
          ),
          i2sLastError
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

      latchClockFault();

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

      latchClockFault();

      continue;
    }

    if (
        frames == FRAMES_PER_BLOCK
    ) {

      setClockHealthy(
          true
      );

    } else {

      incrementI2SErrors();

      latchClockFault();

      blockFlags |=
          FLAG_CLOCK_FAULT;
    }

    // ================================================================
    // SAMPLE INDEX
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
    // INMP441 -> PCM16
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
    // CORE 0 -> CORE 1
    // ================================================================

    if (
        xQueueSend(
            audioQueue,
            &block,
            0
        ) != pdTRUE
    ) {

      // Do not rewind the clock-derived sample timeline.
      //
      // Laptop sees the missing sampleIndex region explicitly.
      incrementDroppedAudioBlocks();
    }
  }
}


// ======================================================================
// QUEUE SYNC REPORT
// ======================================================================


void queueSyncReport(
    const PendingSyncReport& sync
) {

  StatisticsSnapshot stats =
      snapshotStatistics();

  SyncPayload payload{};

  payload.sessionId =
      sync.sessionId;

  payload.syncId =
      sync.syncId;

  payload.sampleIndex =
      0;

  // Diagnostic local ESP32 timestamp only.
  payload.localMicros =
      sync.localMicros;

  TelemetryMessage message{};

  message.type =
      PKT_SYNC;

  message.sessionId =
      sync.sessionId;

  message.sampleIndex =
      0;

  message.localMicros =
      sync.localMicros;

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
}


// ======================================================================
// HEARTBEAT
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

  heartbeat.syncReceived =
      runtime.syncReceived
          ? 1
          : 0;

  heartbeat.clockHealthy =
      runtime.clockHealthy
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
// SYNC / CLOCK HEALTH TASK
// ======================================================================
//
// GPIO27 health and sample-clock progress diagnostics.
//
// It does not create synchronization.
//
// BCLK/WS remain the actual shared sample clock.
// ======================================================================


void syncHealthTask(
    void*
) {

  uint64_t previousSampleIndex =
      0;

  uint32_t lastProgressMs =
      millis();

  bool clockStallReported =
      false;

  bool syncHighReported =
      false;

  while (
      true
  ) {

    RuntimeSnapshot runtime =
        snapshotRuntimeState();

    // ================================================================
    // SESSION ACTIVE
    // ================================================================

    if (
        runtime.streaming &&
        runtime.syncReceived
    ) {

      // --------------------------------------------------------------
      // SYNC LINE HEALTH
      // --------------------------------------------------------------

      if (
          digitalRead(PIN_SYNC) == HIGH
      ) {

        const uint32_t highDurationUs =
            static_cast<uint32_t>(
                micros()
                - runtime.lastSyncMicros
            );

        if (
            highDurationUs > SYNC_STUCK_HIGH_US
        ) {

          latchClockFault();

          if (
              !syncHighReported
          ) {

            Serial.println(
                "[SYNC] fault: GPIO27 stuck HIGH"
            );

            syncHighReported =
                true;
          }
        }

      } else {

        syncHighReported =
            false;
      }

      // --------------------------------------------------------------
      // SHARED CLOCK PROGRESS
      // --------------------------------------------------------------

      if (
          runtime.sampleIndex != previousSampleIndex
      ) {

        previousSampleIndex =
            runtime.sampleIndex;

        lastProgressMs =
            millis();

        clockStallReported =
            false;

      } else if (
          millis() - lastProgressMs
          > CLOCK_STALL_MS
      ) {

        latchClockFault();

        if (
            !clockStallReported
        ) {

          Serial.println(
              "[CLOCK] fault: shared sample clock stalled"
          );

          clockStallReported =
              true;
        }
      }

    } else {

      previousSampleIndex =
          runtime.sampleIndex;

      lastProgressMs =
          millis();

      clockStallReported =
          false;

      syncHighReported =
          false;
    }

    vTaskDelay(
        pdMS_TO_TICKS(
            SYNC_HEALTH_INTERVAL_MS
        )
    );
  }
}


// ======================================================================
// CONTROL COMMANDS
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

    const size_t received =
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

        armStreaming(
            frame.sessionId
        );

        break;


      // ==============================================================
      // STOP
      // ==============================================================

      case CMD_STOP: {

        RuntimeSnapshot runtime =
            snapshotRuntimeState();

        if (
            runtime.armed &&
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
// Core 1.
//
// Audio acquisition continues independently on Core 0.
//
// Temporary Wi-Fi failure can therefore create dropped queued blocks,
// but it does not redefine the sample clock.
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
    // HELLO
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
    // COMMANDS
    // ================================================================

    processCommands();

    // ================================================================
    // SYNC REPORT
    // ================================================================

    PendingSyncReport pendingSync{};

    if (
        consumePendingSyncReport(
            pendingSync
        )
    ) {

      queueSyncReport(
          pendingSync
      );
    }

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

      bool validTelemetry =
          (
              telemetry.type == PKT_HEARTBEAT
          )
          ||
          (
              telemetry.sessionId != 0 &&
              telemetry.sessionId == runtime.sessionId
          );

      if (
          validTelemetry
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

    const uint32_t now =
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
      "Wildlife Soundscape NODE 3 SLAVE v1.4.0 / protocol v4"
  );

  Serial.println(
      "Target: ESP32 DevKit V1 / ESP32-WROOM-32"
  );

  Serial.printf(
      "Audio sensor: %s\n",
      currentAudioSensorName()
  );

  Serial.println(
      "Clock source: Node 1 shared BCLK/WS"
  );

  Serial.println(
      "========================================================"
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


  xEventGroupSetBits(
      audioEvents,
      AUDIO_EVENT_STOPPED
  );


  // ==================================================================
  // SYNC
  // ==================================================================

  setupSync();


  // ==================================================================
  // WI-FI
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


  BaseType_t syncTaskCreated =
      xTaskCreatePinnedToCore(
          syncHealthTask,
          "SyncHealth",
          3072,
          nullptr,
          2,
          nullptr,
          1
      );


  if (
      audioTaskCreated != pdPASS ||
      networkTaskCreated != pdPASS ||
      syncTaskCreated != pdPASS
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
      "[SYSTEM] NODE 3 SLAVE READY"
  );
}


// ======================================================================
// LOOP
// ======================================================================


void loop() {

  // Runtime operation is FreeRTOS-task driven.
  vTaskDelay(
      pdMS_TO_TICKS(
          1000
      )
  );
}
