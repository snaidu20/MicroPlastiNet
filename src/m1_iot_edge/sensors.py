"""Physical sensor models for the MPN-Edge node.

Each function takes the *true* environmental state and returns a sensor
reading with realistic noise + bias drawn from datasheets:

  - SEN0189 turbidity sensor: ±5% reading + 0.5 NTU offset
  - DFRobot TDS sensor:       ±10% over 0-1000 ppm
  - AS7265x 18-channel NIR:   ~2% per-channel SNR
  - DS18B20 temp:             ±0.5°C
  - Hall-effect flow sensor:  ±3%

We use these models to generate synthetic but physically plausible payloads
during development. When real hardware is plugged in, the same SensorReading
schema is produced — downstream code is unchanged.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from src.common.schemas import SensorReading


# Per-polymer NIR absorbance signatures at our 6 chosen channels (nm).
# Values empirically chosen to match published reflectance studies of
# weathered microplastics in the 900-1700 nm short-wave-infrared band.
NIR_CHANNELS_NM = [940, 1050, 1200, 1450, 1550, 1650]
POLYMER_NIR_SIGNATURE = {
    # absorbance (0-1) at each of the 6 channels
    "PE":   [0.18, 0.22, 0.78, 0.31, 0.42, 0.36],
    "PET":  [0.15, 0.20, 0.35, 0.74, 0.58, 0.30],
    "PP":   [0.20, 0.24, 0.71, 0.34, 0.45, 0.41],
    "PS":   [0.12, 0.55, 0.28, 0.39, 0.61, 0.33],
    "PVC":  [0.25, 0.18, 0.32, 0.41, 0.36, 0.79],
    "Other":[0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
}


@dataclass
class TrueState:
    """Ground-truth environmental state used by the simulator only."""
    particle_count: int             # particles in the imaging chamber
    dominant_polymer: str           # "PE" | "PET" | ...
    base_turbidity_ntu: float       # baseline river turbidity
    water_temp_c: float
    flow_rate_lps: float
    base_tds_ppm: float


def _gauss(mu: float, sigma: float) -> float:
    return random.gauss(mu, sigma)


def read_turbidity(state: TrueState) -> float:
    # Turbidity rises with particle count (Beer-Lambert-ish saturation curve).
    extra = 6.0 * (1.0 - math.exp(-state.particle_count / 25.0))
    raw = state.base_turbidity_ntu + extra
    noisy = raw * (1 + _gauss(0, 0.05)) + _gauss(0.5, 0.2)
    return max(0.0, round(noisy, 2))


def read_tds(state: TrueState) -> float:
    raw = state.base_tds_ppm + 0.4 * state.particle_count
    return max(0.0, round(raw * (1 + _gauss(0, 0.10)), 1))


def read_nir(state: TrueState) -> list[float]:
    """6-channel NIR absorbance in [0, 1]."""
    sig = POLYMER_NIR_SIGNATURE.get(state.dominant_polymer,
                                    POLYMER_NIR_SIGNATURE["Other"])
    # Particles above ~5 in the chamber start dominating absorbance; below
    # that we mostly see background water absorbance.
    weight = min(1.0, state.particle_count / 12.0)
    bg = [0.10, 0.12, 0.18, 0.22, 0.20, 0.16]
    out = []
    for s, b in zip(sig, bg):
        v = weight * s + (1 - weight) * b + _gauss(0, 0.02)
        out.append(round(max(0.0, min(1.0, v)), 4))
    return out


def read_temp(state: TrueState) -> float:
    return round(state.water_temp_c + _gauss(0, 0.3), 2)


def read_flow(state: TrueState) -> float:
    return round(max(0.0, state.flow_rate_lps * (1 + _gauss(0, 0.03))), 3)


def sample_all(state: TrueState) -> SensorReading:
    return SensorReading(
        turbidity_ntu=read_turbidity(state),
        tds_ppm=read_tds(state),
        nir_absorbance=read_nir(state),
        water_temp_c=read_temp(state),
        flow_rate_lps=read_flow(state),
    )
