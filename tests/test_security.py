"""Adversarial tests for M6 (cybersecurity) — proves the layer catches:
  - tampering with payload fields
  - replay of an old signed payload
  - signing with the wrong key
  - stale timestamps
"""
import copy
import json
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.schemas import SensorPayload, SensorReading
from src.m6_security import (
    KeyStore, NonceCache,
    sign_payload, verify_payload,
)


def _make_payload(station_id: str = "OGE-01") -> SensorPayload:
    return SensorPayload.new(
        station_id=station_id,
        lat=32.18, lon=-81.41,
        readings=SensorReading(
            turbidity_ntu=3.5, tds_ppm=200.0,
            nir_absorbance=[0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
            water_temp_c=20.0, flow_rate_lps=0.1,
        ),
    )


def test_happy_path(tmp_path: Path):
    ks = KeyStore(tmp_path / "ks.json")
    secret = ks.get_or_create_secret("OGE-01")
    p = _make_payload()
    sign_payload(p, secret)
    ok, reason = verify_payload(p, secret, NonceCache())
    assert ok, reason


def test_tampering_caught(tmp_path: Path):
    """Flip the turbidity reading after signing — must be rejected."""
    ks = KeyStore(tmp_path / "ks.json")
    secret = ks.get_or_create_secret("OGE-01")
    p = _make_payload()
    sign_payload(p, secret)

    # Attacker mutates the reading in transit.
    p.readings.turbidity_ntu = 0.1  # hide a contamination event

    ok, reason = verify_payload(p, secret, NonceCache())
    assert not ok and reason == "bad_signature"


def test_replay_caught(tmp_path: Path):
    """Same payload received twice must be rejected the second time."""
    ks = KeyStore(tmp_path / "ks.json")
    secret = ks.get_or_create_secret("OGE-01")
    p = _make_payload()
    sign_payload(p, secret)

    cache = NonceCache()
    ok1, _ = verify_payload(p, secret, cache)
    ok2, reason2 = verify_payload(p, secret, cache)
    assert ok1
    assert not ok2 and reason2 == "replay_detected"


def test_wrong_key_caught(tmp_path: Path):
    ks = KeyStore(tmp_path / "ks.json")
    real_secret = ks.get_or_create_secret("OGE-01")
    fake_secret = secrets.token_bytes(32)

    p = _make_payload()
    sign_payload(p, fake_secret)
    ok, reason = verify_payload(p, real_secret, NonceCache())
    assert not ok and reason == "bad_signature"


def test_stale_timestamp_caught(tmp_path: Path):
    ks = KeyStore(tmp_path / "ks.json")
    secret = ks.get_or_create_secret("OGE-01")
    p = _make_payload()
    p.timestamp_utc = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    sign_payload(p, secret)

    ok, reason = verify_payload(p, secret, NonceCache())
    assert not ok and reason.startswith("stale")


def test_key_rotation_grace_window(tmp_path: Path):
    """After rotation, payloads signed with the *old* key still verify
    during the grace window."""
    ks = KeyStore(tmp_path / "ks.json")
    old_secret = ks.get_or_create_secret("OGE-01")
    p = _make_payload()
    sign_payload(p, old_secret)

    new_secret = ks.rotate("OGE-01")
    assert new_secret != old_secret

    candidates = ks.candidate_secrets("OGE-01", grace_minutes=30)
    assert len(candidates) == 2  # current + previous

    cache = NonceCache()
    ok = False
    for cand in candidates:
        v, _ = verify_payload(p, cand, cache)
        ok = ok or v
    assert ok


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for fn in [test_happy_path, test_tampering_caught, test_replay_caught,
                   test_wrong_key_caught, test_stale_timestamp_caught,
                   test_key_rotation_grace_window]:
            sub = tmp / fn.__name__
            sub.mkdir()
            fn(sub)
            print(f"  PASS  {fn.__name__}")
    print("\nAll security tests passed.")
