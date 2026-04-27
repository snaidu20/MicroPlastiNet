"""Per-station shared-secret keystore with rotation support.

In production this would be backed by AWS Secrets Manager / Azure Key Vault /
HashiCorp Vault. For the reference implementation we use a JSON file with
file-system permissions, which is a reasonable on-prem fallback.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


class KeyStore:
    """Holds the current and previous secret per station to allow rotation
    without dropping in-flight messages.

    Each station has:
      - current_secret : hex string (32 bytes)
      - previous_secret: hex string or None
      - rotated_at_utc : ISO timestamp
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"stations": {}}, indent=2))

    def _load(self) -> dict:
        return json.loads(self.path.read_text())

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path)

    def get_or_create_secret(self, station_id: str) -> bytes:
        data = self._load()
        st = data["stations"].get(station_id)
        if st is None:
            secret = secrets.token_hex(32)
            data["stations"][station_id] = {
                "current_secret": secret,
                "previous_secret": None,
                "rotated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._save(data)
            return bytes.fromhex(secret)
        return bytes.fromhex(st["current_secret"])

    def rotate(self, station_id: str) -> bytes:
        """Generate a fresh secret for a station while keeping the old one
        accepted for a transition period.
        """
        data = self._load()
        st = data["stations"].setdefault(station_id, {})
        st["previous_secret"] = st.get("current_secret")
        st["current_secret"] = secrets.token_hex(32)
        st["rotated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._save(data)
        return bytes.fromhex(st["current_secret"])

    def candidate_secrets(self, station_id: str,
                          grace_minutes: int = 30) -> list[bytes]:
        """Return [current, previous] if previous is still within the grace
        window; otherwise [current] only.
        """
        data = self._load()
        st = data["stations"].get(station_id)
        if st is None:
            return []
        out = [bytes.fromhex(st["current_secret"])]
        prev = st.get("previous_secret")
        if prev:
            try:
                rotated = datetime.fromisoformat(st["rotated_at_utc"])
                if datetime.now(timezone.utc) - rotated < timedelta(minutes=grace_minutes):
                    out.append(bytes.fromhex(prev))
            except Exception:
                pass
        return out
