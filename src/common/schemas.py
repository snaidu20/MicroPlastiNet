"""
Shared data schemas used across all MicroPlastiNet modules.

The IoT edge node (M1) emits SensorPayload messages over MQTT.
Cloud modules (M2a/M2b/M3/M4) consume and augment them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional
import json
import uuid


@dataclass
class SensorReading:
    """Raw sensor readings from a single time step on an MPN-Edge node."""
    turbidity_ntu: float          # Nephelometric Turbidity Units
    tds_ppm: float                # Total Dissolved Solids (parts per million)
    nir_absorbance: list[float]   # 6-channel NIR absorbance (e.g., 940/1050/1200/1450/1550/1650 nm)
    water_temp_c: float
    flow_rate_lps: float          # Litres per second through chamber


@dataclass
class SensorPayload:
    """Full payload published by an MPN-Edge node over MQTT.

    `image_b64` is the base64-encoded JPEG (downsampled by edge ML).
    `edge_flag` is the on-device first-pass detector verdict.
    `signature` and `nonce` are filled in by the security layer (M6).
    """
    payload_id: str
    station_id: str
    lat: float
    lon: float
    timestamp_utc: str
    readings: SensorReading
    image_b64: Optional[str] = None
    edge_flag: bool = False           # True if on-device model flagged "suspicious"
    edge_confidence: float = 0.0
    firmware_version: str = "1.0.0"
    nonce: Optional[str] = None
    signature: Optional[str] = None   # HMAC-SHA256 of canonical payload

    @staticmethod
    def new(station_id: str, lat: float, lon: float, readings: SensorReading,
            image_b64: Optional[str] = None, edge_flag: bool = False,
            edge_confidence: float = 0.0) -> "SensorPayload":
        return SensorPayload(
            payload_id=str(uuid.uuid4()),
            station_id=station_id,
            lat=lat,
            lon=lon,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            readings=readings,
            image_b64=image_b64,
            edge_flag=edge_flag,
            edge_confidence=edge_confidence,
        )

    def to_canonical_json(self) -> str:
        """Canonical JSON representation for HMAC signing.

        Excludes the signature itself; sorts keys deterministically.
        """
        d = asdict(self)
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "SensorPayload":
        d = json.loads(raw)
        d["readings"] = SensorReading(**d["readings"])
        return cls(**d)


@dataclass
class DetectionResult:
    """Output of M2a vision module per particle."""
    bbox: list[float]              # [x1, y1, x2, y2] in image pixels
    size_mm: float
    shape_class: str               # fragment | fiber | film | bead | foam
    shape_confidence: float


@dataclass
class PolymerResult:
    """Output of M2b spectral module."""
    polymer: str                   # PE | PET | PP | PS | PVC | Other
    probabilities: dict[str, float]
    confidence: float


@dataclass
class StationVerdict:
    """Full pipeline output per station per timestep."""
    payload_id: str
    station_id: str
    timestamp_utc: str
    particle_count: int
    detections: list[DetectionResult] = field(default_factory=list)
    polymer: Optional[PolymerResult] = None
    estimated_concentration_per_m3: float = 0.0
    contamination_level: str = "low"   # low | moderate | high | severe
    notes: str = ""
