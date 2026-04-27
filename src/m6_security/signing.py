"""HMAC-SHA256 payload signing with replay protection.

Why this matters:
  An IoT pollution-monitoring system is a high-value tampering target
  (industrial actors have direct financial incentives to spoof readings).
  This module guarantees integrity + freshness of every sensor message
  using lightweight cryptography that runs on an ESP32.

References:
  - RFC 2104  (HMAC)
  - NIST SP 800-107 (HMAC-SHA256 strength)
  - OWASP IoT Top 10 (insecure communications)
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import OrderedDict
from typing import Optional

from src.common.schemas import SensorPayload


# Maximum allowed clock skew between edge and broker, in seconds.
# Anything older than this is considered a replay or stale message.
MAX_TIMESTAMP_AGE_SECONDS = 300  # 5 minutes


def _hmac_sha256(secret: bytes, message: str) -> str:
    return hmac.new(secret, message.encode("utf-8"), hashlib.sha256).hexdigest()


def sign_payload(payload: SensorPayload, secret: bytes) -> SensorPayload:
    """Attach a fresh nonce and HMAC-SHA256 signature to the payload.

    Returns the same payload object (mutated) for convenience.
    """
    payload.nonce = secrets.token_hex(16)  # 128-bit nonce
    canonical = payload.to_canonical_json()
    payload.signature = _hmac_sha256(secret, canonical)
    return payload


class NonceCache:
    """Bounded LRU cache of recently-seen nonces to block replays.

    On the broker side, we keep the last `maxsize` nonces. Any new payload
    whose nonce is already in the cache is rejected as a replay.
    """

    def __init__(self, maxsize: int = 10_000):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize

    def seen(self, nonce: str) -> bool:
        if nonce in self._cache:
            return True
        self._cache[nonce] = time.time()
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return False


def verify_payload(
    payload: SensorPayload,
    secret: bytes,
    nonce_cache: Optional[NonceCache] = None,
    max_age_seconds: int = MAX_TIMESTAMP_AGE_SECONDS,
) -> tuple[bool, str]:
    """Verify a payload's HMAC, freshness, and uniqueness.

    Returns (ok, reason).
    """
    if not payload.signature:
        return False, "missing_signature"
    if not payload.nonce:
        return False, "missing_nonce"

    # Recompute HMAC over canonical (sig-stripped) payload.
    canonical = payload.to_canonical_json()
    expected = _hmac_sha256(secret, canonical)
    if not hmac.compare_digest(expected, payload.signature):
        return False, "bad_signature"

    # Freshness check using payload timestamp.
    try:
        from datetime import datetime
        ts = datetime.fromisoformat(payload.timestamp_utc.replace("Z", "+00:00"))
        age = time.time() - ts.timestamp()
    except Exception:
        return False, "bad_timestamp"

    if age > max_age_seconds:
        return False, f"stale (age={age:.0f}s)"
    if age < -max_age_seconds:
        return False, f"clock_skew_future (age={age:.0f}s)"

    # Replay check.
    if nonce_cache is not None and nonce_cache.seen(payload.nonce):
        return False, "replay_detected"

    return True, "ok"
