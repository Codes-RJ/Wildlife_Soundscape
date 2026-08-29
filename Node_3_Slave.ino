/*
  Wildlife Soundscape Mapping & Behavior Analysis System
  Slave Node 3 firmware v1.4.0 / wire protocol v4

  ESP32 DevKit V1 / ESP32-WROOM-32
  INMP441: BCLK=GPIO26 (shared input), WS=GPIO25 (shared input),
            SD=GPIO33, L/R=GND, VDD=3V3
  SYNC   : GPIO27 input from master
*/

#include <Arduino.h>
#include <WiFi.h>
#include <ESP_I2S.h>

#define NODE_ID 3
#define FIRMWARE_VERSION "1.4.0"

const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
IPAddress LAPTOP_IP(192, 168, 1, 100);
constexpr uint16_t LAPTOP_PORT = 5001;

constexpr int PIN_BCLK = 26;
constexpr int PIN_WS = 25;
constexpr int PIN_MIC_DATA = 33;
constexpr int PIN_SYNC = 27;

constexpr uint32_t SAMPLE_RATE = 48000;
constexpr uint16_t FRAMES_PER_BLOCK = 1024;
constexpr uint16_t RAW_VALUES_PER_BLOCK = FRAMES_PER_BLOCK * 2;
constexpr int16_t SYNC_ALIGN_TOLERANCE_SAMPLES = 10;
constexpr uint8_t AUDIO_QUEUE_DEPTH = 6;
constexpr uint8_t TELEMETRY_QUEUE_DEPTH = 6;
constexpr uint32_t HEARTBEAT_INTERVAL_MS = 5000;
constexpr uint32_t RECONNECT_MIN_MS = 1000;
constexpr uint32_t RECONNECT_MAX_MS = 30000;
constexpr uint32_t SYNC_STUCK_HIGH_US = 1000000;
constexpr uint32_t CLOCK_STALL_MS = 1000;

constexpr uint32_t PACKET_MAGIC = 0x574C5343UL;
constexpr uint8_t PROTOCOL_VERSION = 4;
constexpr uint16_t CONTROL_MAGIC = 0xC0DE;
constexpr uint8_t CONTROL_VERSION = 1;

enum PacketFlags : uint8_t {
  FLAG_NONE = 0x00, FLAG_CLIPPED = 0x01, FLAG_CLOCK_FAULT = 0x02,
  FLAG_QUEUE_CONGESTED = 0x04
};

enum PacketType : uint8_t {
  PKT_HELLO = 1, PKT_AUDIO = 2, PKT_ENVIRONMENT = 3,
  PKT_HEARTBEAT = 4, PKT_SYNC = 5
};

enum ControlCommand : uint8_t { CMD_START = 0xA1, CMD_STOP = 0xA2, CMD_PING = 0xA3 };

struct __attribute__((packed)) ControlFrame {
  uint16_t magic; uint8_t version; uint8_t command; uint32_t sessionId;
};

struct __attribute__((packed)) PacketHeader {
  uint32_t magic; uint8_t protocolVersion; uint8_t nodeId; uint8_t packetType; uint8_t flags;
  uint32_t sequence; uint32_t sessionId; uint64_t sampleIndex;
  uint32_t localMicros; uint32_t i2sErrorCount; uint32_t payloadLength; uint32_t payloadCRC32;
};

struct __attribute__((packed)) HelloPayload {
  uint32_t sampleRate; uint16_t framesPerPacket; uint8_t bitsPerSample; uint8_t channels;
  uint8_t masterNode; int16_t syncToleranceSamples; char firmware[16];
};

struct __attribute__((packed)) SyncPayload {
  uint32_t sessionId; uint32_t syncId; uint64_t sampleIndex; uint32_t localMicros;
};

struct __attribute__((packed)) HeartbeatPayload {
  uint32_t uptimeSeconds; int32_t wifiRSSI; uint32_t freeHeap;
  uint32_t droppedAudioBlocks; uint32_t transmittedAudioBlocks; uint32_t i2sErrors;
  uint16_t audioQueueDepth; uint8_t streaming; uint8_t syncReceived; uint8_t clockHealthy;
};

struct AudioBlock {
  uint64_t sampleIndex; uint32_t localMicros; uint32_t i2sErrorCountAtCapture;
  uint16_t frameCount; uint8_t flags; int16_t pcm[FRAMES_PER_BLOCK];
};

struct TelemetryMessage {
  PacketType type; uint64_t sampleIndex; uint32_t localMicros;
  uint32_t payloadLength; uint8_t flags; uint8_t payload[64];
};

static_assert(sizeof(ControlFrame) == 8, "ControlFrame size mismatch");
static_assert(sizeof(PacketHeader) == 40, "PacketHeader size mismatch");
static_assert(sizeof(HelloPayload) == 27, "HelloPayload size mismatch");
static_assert(sizeof(SyncPayload) == 20, "SyncPayload size mismatch");
static_assert(sizeof(HeartbeatPayload) == 29, "Slave Heartbeat size mismatch");

I2SClass I2S;
WiFiClient tcpClient;
QueueHandle_t audioQueue = nullptr;
QueueHandle_t telemetryQueue = nullptr;

volatile bool streaming = false;
volatile bool syncReceived = false;
volatile bool waitingForSync = false;
volatile bool i2sStarted = false;
volatile bool clockHealthy = false;
volatile bool syncReportPending = false;
volatile uint64_t totalSampleCount = 0;
volatile uint32_t droppedAudioBlocks = 0;
volatile uint32_t transmittedAudioBlocks = 0;
volatile uint32_t i2sErrors = 0;
volatile uint32_t syncRiseMicros = 0;
volatile uint32_t lastSyncLocalMicros = 0;
volatile bool clockFaultLatched = false;
uint32_t packetSequence = 0;
uint32_t currentSessionId = 0;
uint32_t syncId = 0;

uint32_t calculateCRC32(const uint8_t* data, size_t length) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit)
      crc = (crc & 1U) ? ((crc >> 1) ^ 0xEDB88320UL) : (crc >> 1);
  }
  return crc ^ 0xFFFFFFFFUL;
}

bool sendAll(const uint8_t* data, size_t length) {
  size_t sent = 0;
  while (sent < length) {
    if (!tcpClient.connected()) return false;
    size_t written = tcpClient.write(data + sent, length - sent);
    if (!written) { vTaskDelay(pdMS_TO_TICKS(1)); continue; }
    sent += written;
  }
  return true;
}

bool sendPacket(PacketType type, uint64_t sampleIndex, uint32_t captureMicros,
                uint32_t captureErrors, const void* payload, uint32_t payloadLength,
                uint8_t flags = FLAG_NONE) {
  PacketHeader h{};
  h.magic = PACKET_MAGIC; h.protocolVersion = PROTOCOL_VERSION; h.nodeId = NODE_ID;
  h.packetType = static_cast<uint8_t>(type); h.flags = flags; h.sequence = packetSequence++;
  h.sessionId = currentSessionId; h.sampleIndex = sampleIndex; h.localMicros = captureMicros;
  h.i2sErrorCount = captureErrors; h.payloadLength = payloadLength;
  h.payloadCRC32 = (payload && payloadLength)
      ? calculateCRC32(reinterpret_cast<const uint8_t*>(payload), payloadLength) : 0;
  if (!sendAll(reinterpret_cast<const uint8_t*>(&h), sizeof(h))) return false;
  return (!payload || !payloadLength)
      ? true : sendAll(reinterpret_cast<const uint8_t*>(payload), payloadLength);
}

uint8_t currentHealthFlags() {
  uint8_t flags = FLAG_NONE;
  if (clockFaultLatched) flags |= FLAG_CLOCK_FAULT;
  if (audioQueue) {
    const UBaseType_t depth = uxQueueMessagesWaiting(audioQueue);
    if (depth * 5U >= AUDIO_QUEUE_DEPTH * 4U) flags |= FLAG_QUEUE_CONGESTED;
  }
  return flags;
}

bool ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;
  uint32_t retry = RECONNECT_MIN_MS;
  uint32_t disconnectedAt = millis();
  while (WiFi.status() != WL_CONNECTED) {
    Serial.printf("[WiFi] reconnect; wait=%lu ms\n", retry);
    WiFi.reconnect();
    uint32_t started = millis();
    while (millis() - started < retry) {
      if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[WiFi] restored after %lu ms | IP=", millis() - disconnectedAt);
        Serial.println(WiFi.localIP());
        return true;
      }
      vTaskDelay(pdMS_TO_TICKS(100));
    }
    retry = min(retry * 2UL, RECONNECT_MAX_MS);
  }
  return true;
}

void setupWiFi() {
  WiFi.mode(WIFI_STA); WiFi.setSleep(false); WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD); ensureWiFi();
}

bool ensureLaptopConnection() {
  if (tcpClient.connected()) return true;
  tcpClient.stop();
  uint32_t retry = RECONNECT_MIN_MS;
  while (!tcpClient.connected()) {
    ensureWiFi();
    if (tcpClient.connect(LAPTOP_IP, LAPTOP_PORT)) {
      tcpClient.setNoDelay(true); tcpClient.setTimeout(50);
      Serial.println("[TCP] connected"); return true;
    }
    Serial.printf("[TCP] retry=%lu ms\n", retry);
    vTaskDelay(pdMS_TO_TICKS(retry));
    retry = min(retry * 2UL, RECONNECT_MAX_MS);
  }
  return true;
}

void flushQueues() {
  AudioBlock a; while (audioQueue && xQueueReceive(audioQueue, &a, 0) == pdTRUE) {}
  TelemetryMessage t; while (telemetryQueue && xQueueReceive(telemetryQueue, &t, 0) == pdTRUE) {}
}

void sendHello() {
  HelloPayload p{};
  p.sampleRate = SAMPLE_RATE; p.framesPerPacket = FRAMES_PER_BLOCK;
  p.bitsPerSample = 16; p.channels = 1; p.masterNode = 0;
  p.syncToleranceSamples = SYNC_ALIGN_TOLERANCE_SAMPLES;
  strncpy(p.firmware, FIRMWARE_VERSION, sizeof(p.firmware)-1);
  sendPacket(PKT_HELLO, totalSampleCount, micros(), i2sErrors, &p, sizeof(p));
}

void stopI2S() {
  if (!i2sStarted) return;
  I2S.end(); i2sStarted = false;
}

bool startI2S() {
  stopI2S();
  I2S.setPins(PIN_BCLK, PIN_WS, -1, PIN_MIC_DATA);
  bool ok = I2S.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT,
                      I2S_SLOT_MODE_STEREO, -1, I2S_ROLE_SLAVE);
  if (!ok) {
    ++i2sErrors; Serial.printf("[I2S] slave begin error=%d\n", I2S.lastError()); return false;
  }
  i2sStarted = true;
  Serial.println("[I2S] SLAVE armed for external BCLK/WS");
  return true;
}

void IRAM_ATTR syncISR() {
  if (!waitingForSync) return;
  totalSampleCount = 0;
  ++syncId;
  syncRiseMicros = micros();
  lastSyncLocalMicros = syncRiseMicros;
  syncReceived = true;
  syncReportPending = true;
  waitingForSync = false;
  clockHealthy = true;
}

void setupSync() {
  pinMode(PIN_SYNC, INPUT_PULLDOWN);
  attachInterrupt(digitalPinToInterrupt(PIN_SYNC), syncISR, RISING);
}

void stopStreaming() {
  streaming = false; waitingForSync = false; syncReceived = false; clockHealthy = false;
  clockFaultLatched = false;
  stopI2S(); flushQueues(); totalSampleCount = 0;
  Serial.println("[SYSTEM] stopped; queues flushed");
}

void armStreaming(uint32_t sessionId) {
  stopStreaming();
  currentSessionId = sessionId ? sessionId : 1;
  packetSequence = 0; totalSampleCount = 0;
  if (!startI2S()) return;
  syncReceived = false; waitingForSync = true; streaming = true; clockFaultLatched = false;
  Serial.printf("[SYSTEM] ARMED session=0x%08lX, waiting for SYNC\n", currentSessionId);
}

inline int16_t convertToPCM16(int32_t value) { return static_cast<int16_t>(value >> 16); }

void audioCaptureTask(void*) {
  static int32_t raw[RAW_VALUES_PER_BLOCK];
  AudioBlock block{};
  while (true) {
    if (!streaming || !syncReceived || !i2sStarted) {
      vTaskDelay(pdMS_TO_TICKS(2)); continue;
    }
    size_t bytesRead = I2S.readBytes(reinterpret_cast<char*>(raw), sizeof(raw));
    if (bytesRead == 0) {
      ++i2sErrors; Serial.printf("[I2S] read error #%lu code=%d\n", i2sErrors, I2S.lastError());
      vTaskDelay(pdMS_TO_TICKS(1)); continue;
    }
    if (!streaming || !syncReceived) continue;
    if (bytesRead % (sizeof(int32_t)*2) != 0) ++i2sErrors;
    size_t frames = min(bytesRead/(sizeof(int32_t)*2), static_cast<size_t>(FRAMES_PER_BLOCK));
    block.sampleIndex = totalSampleCount; block.localMicros = micros();
    block.i2sErrorCountAtCapture = i2sErrors; block.frameCount = static_cast<uint16_t>(frames);
    block.flags = FLAG_NONE;
    for (size_t i=0; i<frames; ++i) {
      block.pcm[i] = convertToPCM16(raw[i*2]);
      if (block.pcm[i] >= 32760 || block.pcm[i] <= -32760) block.flags |= FLAG_CLIPPED;
    }
    totalSampleCount += frames;
    if (xQueueSend(audioQueue, &block, 0) != pdTRUE) ++droppedAudioBlocks;
  }
}

void queueSyncReport() {
  SyncPayload p{};
  p.sessionId = currentSessionId; p.syncId = syncId; p.sampleIndex = 0;
  p.localMicros = lastSyncLocalMicros;
  TelemetryMessage msg{};
  msg.type = PKT_SYNC; msg.sampleIndex = 0; msg.localMicros = lastSyncLocalMicros;
  msg.payloadLength = sizeof(p); msg.flags = currentHealthFlags(); memcpy(msg.payload, &p, sizeof(p));
  xQueueSend(telemetryQueue, &msg, 0);
}

void queueHeartbeat() {
  HeartbeatPayload hb{};
  hb.uptimeSeconds = millis()/1000; hb.wifiRSSI = WiFi.RSSI(); hb.freeHeap = ESP.getFreeHeap();
  hb.droppedAudioBlocks = droppedAudioBlocks; hb.transmittedAudioBlocks = transmittedAudioBlocks;
  hb.i2sErrors = i2sErrors; hb.audioQueueDepth = uxQueueMessagesWaiting(audioQueue);
  hb.streaming = streaming ? 1 : 0; hb.syncReceived = syncReceived ? 1 : 0;
  hb.clockHealthy = clockHealthy ? 1 : 0;
  TelemetryMessage msg{};
  msg.type = PKT_HEARTBEAT; msg.sampleIndex = totalSampleCount; msg.localMicros = micros();
  msg.payloadLength = sizeof(hb); msg.flags = currentHealthFlags(); memcpy(msg.payload, &hb, sizeof(hb));
  xQueueSend(telemetryQueue, &msg, 0);
}

void syncHealthTask(void*) {
  uint64_t previousSamples = 0;
  uint32_t lastProgress = millis();
  while (true) {
    if (streaming && syncReceived) {
      if (digitalRead(PIN_SYNC) == HIGH) {
        uint32_t highUs = static_cast<uint32_t>(micros() - syncRiseMicros);
        if (highUs > SYNC_STUCK_HIGH_US) {
          Serial.println("[SYNC] fault: line stuck HIGH");
          clockHealthy = false; clockFaultLatched = true; syncReceived = false; streaming = false;
        }
      }

      uint64_t current = totalSampleCount;
      if (current != previousSamples) {
        previousSamples = current; lastProgress = millis(); clockHealthy = true;
      } else if (millis() - lastProgress > CLOCK_STALL_MS) {
        Serial.println("[CLOCK] fault: sample counter stalled");
        clockHealthy = false; clockFaultLatched = true; syncReceived = false; streaming = false;
      }
    } else {
      previousSamples = totalSampleCount;
      lastProgress = millis();
    }
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

void processCommands() {
  while (tcpClient.connected() && tcpClient.available() >= static_cast<int>(sizeof(ControlFrame))) {
    ControlFrame frame{};
    size_t n = tcpClient.readBytes(reinterpret_cast<char*>(&frame), sizeof(frame));
    if (n != sizeof(frame)) return;
    if (frame.magic != CONTROL_MAGIC || frame.version != CONTROL_VERSION) {
      Serial.println("[CONTROL] invalid frame"); continue;
    }
    switch(frame.command) {
      case CMD_START: armStreaming(frame.sessionId); break;
      case CMD_STOP: stopStreaming(); break;
      case CMD_PING: queueHeartbeat(); break;
      default: Serial.printf("[CONTROL] unknown 0x%02X\n", frame.command); break;
    }
  }
}

void networkTask(void*) {
  AudioBlock audio{}; TelemetryMessage telemetry{};
  bool helloSent = false; uint32_t lastHeartbeat = 0;
  while (true) {
    if (WiFi.status() != WL_CONNECTED) { helloSent=false; ensureWiFi(); }
    if (!tcpClient.connected()) { helloSent=false; ensureLaptopConnection(); }
    if (!helloSent) { sendHello(); helloSent=true; }
    processCommands();
    if (syncReportPending) { syncReportPending=false; queueSyncReport(); }
    if (xQueueReceive(audioQueue, &audio, 0) == pdTRUE) {
      if (sendPacket(PKT_AUDIO, audio.sampleIndex, audio.localMicros,
                     audio.i2sErrorCountAtCapture, audio.pcm,
                     audio.frameCount*sizeof(int16_t),
                     static_cast<uint8_t>(audio.flags | currentHealthFlags()))) ++transmittedAudioBlocks;
    }
    if (xQueueReceive(telemetryQueue, &telemetry, 0) == pdTRUE) {
      sendPacket(telemetry.type, telemetry.sampleIndex, telemetry.localMicros,
                 i2sErrors, telemetry.payload, telemetry.payloadLength,
                 static_cast<uint8_t>(telemetry.flags | currentHealthFlags()));
    }
    uint32_t now=millis();
    if (now-lastHeartbeat >= HEARTBEAT_INTERVAL_MS) { lastHeartbeat=now; queueHeartbeat(); }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

void setup() {
  Serial.begin(115200); delay(1000);
  Serial.printf("\nWildlife Soundscape SLAVE %d v1.4.0 / protocol v4\n", NODE_ID);
  setupSync();
  audioQueue = xQueueCreate(AUDIO_QUEUE_DEPTH, sizeof(AudioBlock));
  telemetryQueue = xQueueCreate(TELEMETRY_QUEUE_DEPTH, sizeof(TelemetryMessage));
  if (!audioQueue || !telemetryQueue) {
    while(true) { Serial.println("[FATAL] queue allocation"); delay(1000); }
  }
  setupWiFi(); ensureLaptopConnection();
  xTaskCreatePinnedToCore(audioCaptureTask, "AudioCapture", 4096, nullptr, 4, nullptr, 0);
  xTaskCreatePinnedToCore(networkTask, "Network", 6144, nullptr, 3, nullptr, 1);
  xTaskCreatePinnedToCore(syncHealthTask, "SyncHealth", 3072, nullptr, 2, nullptr, 1);
  Serial.println("[SYSTEM] SLAVE READY");
}

void loop() { vTaskDelay(pdMS_TO_TICKS(1000)); }
