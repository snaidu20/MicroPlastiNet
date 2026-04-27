"""Edge-node simulator orchestrator.

Generates a continuous stream of signed SensorPayload messages for one or
more virtual MPN-Edge stations along coastal Georgia rivers.

Two output modes:
  - `--mode mqtt`   publish to a real MQTT broker (paho-mqtt)
  - `--mode file`   append JSON-lines to data/processed/edge_stream.jsonl

By default we run `file` mode so the demo works offline.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import time
from pathlib import Path
from typing import Iterable, Optional

from src.common.schemas import SensorPayload
from src.m1_iot_edge.sensors import TrueState, sample_all
from src.m1_iot_edge.edge_detector import edge_predict
from src.m6_security import KeyStore, sign_payload


# --- Coastal Georgia stations (real coordinates, picked to align with
# the Savannah / Ogeechee / Altamaha watersheds in coastal Georgia).
DEFAULT_STATIONS = [
    {"id": "OGE-01", "name": "Ogeechee @ Eden",        "lat": 32.1773, "lon": -81.4082},
    {"id": "OGE-02", "name": "Ogeechee @ Statesboro",  "lat": 32.4488, "lon": -81.7832},
    {"id": "OGE-03", "name": "Ogeechee @ Midville",    "lat": 32.8221, "lon": -82.2367},
    {"id": "SAV-01", "name": "Savannah @ Coastal",     "lat": 32.1495, "lon": -81.1637},
    {"id": "SAV-02", "name": "Savannah @ Augusta",     "lat": 33.4735, "lon": -82.0105},
    {"id": "ALT-01", "name": "Altamaha @ Darien",      "lat": 31.3697, "lon": -81.4334},
    {"id": "ALT-02", "name": "Altamaha @ Jesup",       "lat": 31.6078, "lon": -81.8857},
]


POLYMERS = ["PE", "PET", "PP", "PS", "PVC", "Other"]


def _draw_true_state(rng: random.Random,
                     contamination_prob: float = 0.20) -> TrueState:
    """Draw a plausible true environmental state for one timestep.

    With probability `contamination_prob` we draw a contamination event
    (>= 5 particles, polymer skewed by source mix).
    """
    if rng.random() < contamination_prob:
        # Contaminated event
        particle_count = rng.randint(5, 60)
        polymer = rng.choices(POLYMERS,
                              weights=[0.30, 0.25, 0.18, 0.12, 0.10, 0.05])[0]
    else:
        particle_count = rng.randint(0, 2)
        polymer = "Other"
    return TrueState(
        particle_count=particle_count,
        dominant_polymer=polymer,
        base_turbidity_ntu=rng.uniform(1.5, 4.5),
        water_temp_c=rng.uniform(14.0, 28.0),
        flow_rate_lps=rng.uniform(0.05, 0.30),
        base_tds_ppm=rng.uniform(150.0, 320.0),
    )


def _fake_image_b64(particle_count: int) -> str:
    """Return a tiny fake JPEG-ish blob proportional to particle count.

    The real edge node would attach a downsampled JPEG. We just pack the
    particle count into a recognizable placeholder so downstream logging
    can confirm payload size scales with event severity.
    """
    blob = bytes([particle_count % 256] * 64)
    return base64.b64encode(blob).decode("ascii")


def stream_payloads(stations: Iterable[dict],
                    keystore: KeyStore,
                    n_steps: int = 240,
                    seed: int = 42,
                    sleep_seconds: float = 0.0) -> Iterable[SensorPayload]:
    """Yield signed SensorPayload objects for `n_steps` × #stations."""
    rng = random.Random(seed)
    for step in range(n_steps):
        for st in stations:
            state = _draw_true_state(rng)
            reading = sample_all(state)
            edge_flag, edge_conf = edge_predict(reading)
            payload = SensorPayload.new(
                station_id=st["id"],
                lat=st["lat"], lon=st["lon"],
                readings=reading,
                image_b64=_fake_image_b64(state.particle_count) if edge_flag else None,
                edge_flag=edge_flag,
                edge_confidence=edge_conf,
            )
            secret = keystore.get_or_create_secret(st["id"])
            sign_payload(payload, secret)
            yield payload
            if sleep_seconds:
                time.sleep(sleep_seconds)


def _publish_mqtt(payloads: Iterable[SensorPayload], host: str, port: int,
                  topic: str, tls: bool) -> None:
    import paho.mqtt.client as mqtt  # type: ignore
    client = mqtt.Client(client_id="mpn-edge-sim")
    if tls:
        from src.m6_security import build_tls_context
        client.tls_set_context(build_tls_context(verify_peer=False))
    client.connect(host, port, keepalive=60)
    client.loop_start()
    try:
        for p in payloads:
            client.publish(topic, p.to_json(), qos=1)
    finally:
        client.loop_stop()
        client.disconnect()


def _publish_file(payloads: Iterable[SensorPayload], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for p in payloads:
            f.write(p.to_json() + "\n")
            n += 1
    return n


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["file", "mqtt"], default="file")
    ap.add_argument("--steps", type=int, default=240,
                    help="time steps per station (default 240 = 1 day @ 6 min)")
    ap.add_argument("--out", default="data/processed/edge_stream.jsonl")
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=8883)
    ap.add_argument("--mqtt-topic", default="microplastinet/data")
    ap.add_argument("--no-tls", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keystore", default="data/processed/keystore.json")
    args = ap.parse_args(argv)

    ks = KeyStore(Path(args.keystore))
    payloads = stream_payloads(DEFAULT_STATIONS, ks,
                               n_steps=args.steps, seed=args.seed)

    if args.mode == "file":
        n = _publish_file(payloads, Path(args.out))
        print(f"[edge_simulator] wrote {n} signed payloads to {args.out}")
    else:
        _publish_mqtt(payloads, args.mqtt_host, args.mqtt_port,
                      args.mqtt_topic, tls=not args.no_tls)
        print(f"[edge_simulator] streamed payloads to "
              f"mqtt{'s' if not args.no_tls else ''}://{args.mqtt_host}:{args.mqtt_port}/{args.mqtt_topic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
