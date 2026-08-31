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

The sketches target the Arduino-ESP32 3.x API and still require compilation
against the exact selected board core plus validation on the physical DevKit,
INMP441, and BME280 hardware.

Protocol behavior is documented in [`docs/protocol.md`](../docs/protocol.md).
