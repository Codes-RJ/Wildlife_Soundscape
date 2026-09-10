# ESP32 Firmware

Each node is stored in its own Arduino-compatible sketch directory:

```text
firmware/
├── Node_1_Master/Node_1_Master.ino
├── Node_2_Slave/Node_2_Slave.ino
└── Node_3_Slave/Node_3_Slave.ino
```

Before compiling or flashing, configure the Wi-Fi credentials and receiver IP
for the deployment environment. Do not commit real credentials.

The reproducible compile baseline in `.github/workflows/firmware.yml` pins
Arduino CLI 1.3.1, Arduino-ESP32 3.3.0, Adafruit BME280 2.3.0, Adafruit Unified
Sensor 1.1.15, and Adafruit BusIO 1.17.2. CI compiles every sketch for
`esp32:esp32:esp32`.

Compilation does not replace validation on the physical DevKit, INMP441, and
BME280 hardware. Follow [`docs/OPERATIONS.md`](../docs/OPERATIONS.md) and the
full [`docs/TECHNICAL_REFERENCE.md`](../docs/TECHNICAL_REFERENCE.md).
