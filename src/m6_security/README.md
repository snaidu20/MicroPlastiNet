# M6 — Cybersecurity Layer

> *MicroPlastiNet treats cybersecurity as a first-class concern alongside ML, Deep Learning, GenAI, and Data Science — because environmental monitoring data is only as trustworthy as the channel it travels over.*

## Why this exists
Pollution-monitoring IoT systems are a tampering-magnet. An industrial actor with skin in the game has direct financial incentive to spoof or replay readings — yet **almost no published microplastic IoT paper addresses message integrity**. M6 closes that gap.

## What it does
| Threat | Mitigation |
|---|---|
| Payload tampering on the wire | HMAC-SHA256 over canonical JSON |
| Replay of an old payload | Per-payload nonce + bounded LRU nonce cache |
| Stale messages | 5-minute timestamp freshness window |
| Long-lived key compromise | Per-station key rotation with 30-minute grace |
| Eavesdropping | TLS 1.3 transport (MQTT over TLS) |

## Files
- `signing.py` — `sign_payload()` / `verify_payload()` / `NonceCache`
- `keystore.py` — `KeyStore` with rotate + grace window
- `tls.py` — `build_tls_context()` for paho-mqtt

## Usage
```python
from src.m6_security import sign_payload, verify_payload, NonceCache, KeyStore

ks = KeyStore(Path("data/keys.json"))
secret = ks.get_or_create_secret("station-ogeechee-03")

# Edge side:
payload = SensorPayload.new(...)
sign_payload(payload, secret)
mqtt_client.publish("microplastinet/data", payload.to_json())

# Broker side:
nonce_cache = NonceCache()
ok, reason = verify_payload(payload, secret, nonce_cache)
if not ok:
    log.warning("rejected: %s", reason)
```

## References
- RFC 2104 — HMAC: Keyed-Hashing for Message Authentication
- NIST SP 800-107 r1 — Recommendation for Applications Using Approved Hash Algorithms
- OWASP IoT Top 10 (2018) — I3 Insecure Communications
