"""M6 — Cybersecurity Layer for MicroPlastiNet.

Implements:
- HMAC-SHA256 payload signing
- Replay-attack prevention via nonces + timestamp window
- Lightweight key rotation (per-station shared secrets)
- TLS context helpers for paho-mqtt clients
"""
from .signing import sign_payload, verify_payload, NonceCache
from .keystore import KeyStore
from .tls import build_tls_context

__all__ = [
    "sign_payload",
    "verify_payload",
    "NonceCache",
    "KeyStore",
    "build_tls_context",
]
