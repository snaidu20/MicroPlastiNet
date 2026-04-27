# M1 — IoT Edge Node Simulator

Simulates a deployed **MPN-Edge** node — a $30 weather-proof unit hosting an
ESP32-CAM, a SEN0189 turbidity sensor, a TDS probe, and a 6-channel NIR
spectrometer. The unit ingests a trickle of river water through a transparent
flow-through chamber, runs a tiny on-device detector, and (only when the
detector flags) publishes a signed payload over MQTT.

## Why simulated
Building real hardware would block the project on shipping + assembly. The
simulator emits the exact same `SensorPayload` schema the real device will
emit — every downstream module (M2a, M2b, M3, M4) is unchanged when you
swap in real hardware.

## Files
- `sensors.py` — physical sensor models with datasheet-accurate noise
- `edge_detector.py` — TFLite-Micro-style on-device first-pass detector
- `edge_simulator.py` — generates signed payload streams (JSONL or MQTT)
- `cloud_listener.py` — verifies + routes payloads on the cloud side

## Run

```bash
# 1. Generate a day of data from 7 stations across coastal Georgia rivers.
python -m src.m1_iot_edge.edge_simulator \
    --mode file --steps 240 \
    --out data/processed/edge_stream.jsonl

# 2. Verify on the cloud side.
python -m src.m1_iot_edge.cloud_listener \
    --in data/processed/edge_stream.jsonl \
    --out-dir data/processed
```

## Geographic grounding
Stations are placed at real coordinates along the **Ogeechee, Savannah, and
Altamaha rivers** in coastal Georgia — these watersheds drain into the
Atlantic and provide a realistic deployment context for an actual
freshwater microplastics monitoring network.

## Cybersecurity (M6)
Every payload is signed with HMAC-SHA256 using a per-station secret from
the `KeyStore`. The cloud listener verifies the signature, freshness, and
uniqueness (replay protection) before passing the payload downstream.

## Honesty
Sensor noise models are *physically plausible* but not calibrated against
real ESP32 hardware. The `edge_detector` is a 3-feature logistic gate, not
a TFLite model — but it implements the same `(is_suspicious, confidence)`
contract, so the rest of the pipeline is unchanged when a real model lands.
