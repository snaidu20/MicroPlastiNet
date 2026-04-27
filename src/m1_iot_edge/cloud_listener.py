"""Cloud-side listener that consumes signed payloads, verifies them, and
hands them off to the M2/M3 pipeline.

In `file` mode it reads a JSONL stream produced by the simulator. In `mqtt`
mode it subscribes to the broker.

Verified payloads are appended to `data/processed/verified_stream.jsonl`,
rejected ones to `data/processed/rejected_stream.jsonl` for audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from src.common.schemas import SensorPayload
from src.m6_security import KeyStore, NonceCache, verify_payload


def _verify_one(payload: SensorPayload, keystore: KeyStore,
                nonce_cache: NonceCache) -> tuple[bool, str]:
    secrets = keystore.candidate_secrets(payload.station_id)
    if not secrets:
        return False, "unknown_station"
    last_reason = "unknown"
    for s in secrets:
        ok, reason = verify_payload(payload, s, nonce_cache)
        if ok:
            return True, "ok"
        last_reason = reason
    return False, last_reason


def consume_file(in_path: Path, out_dir: Path, keystore: KeyStore) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    verified_path = out_dir / "verified_stream.jsonl"
    rejected_path = out_dir / "rejected_stream.jsonl"
    nonce_cache = NonceCache()
    counts = {"verified": 0, "rejected": 0, "by_reason": {}}
    with in_path.open() as f, \
         verified_path.open("w") as vout, \
         rejected_path.open("w") as rout:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = SensorPayload.from_json(line)
            ok, reason = _verify_one(payload, keystore, nonce_cache)
            if ok:
                vout.write(line + "\n")
                counts["verified"] += 1
            else:
                rec = {"reason": reason, "payload": json.loads(line)}
                rout.write(json.dumps(rec) + "\n")
                counts["rejected"] += 1
                counts["by_reason"][reason] = counts["by_reason"].get(reason, 0) + 1
    return counts


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/edge_stream.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--keystore", default="data/processed/keystore.json")
    args = ap.parse_args(argv if argv is not None else None)

    counts = consume_file(Path(args.inp), Path(args.out_dir),
                          KeyStore(Path(args.keystore)))
    print(f"[cloud_listener] verified={counts['verified']} "
          f"rejected={counts['rejected']} reasons={counts['by_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
