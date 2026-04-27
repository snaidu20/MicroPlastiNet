"""Tiny on-device first-pass detector that runs on the ESP32.

In production this is a TensorFlow Lite Micro model (~30 KB flash). Here we
emulate it with a simple, fast rule + a 3-feature logistic gate so the
simulator stays dependency-free. The contract is identical:

    in : SensorReading
    out: (is_suspicious: bool, confidence: float in [0, 1])

The cloud (M2a/M2b) does the heavy lifting; the edge model just decides
whether to *bother* sending a full payload (saves bandwidth on cellular).
"""
from __future__ import annotations

import math

from src.common.schemas import SensorReading


# Hand-tuned coefficients of a logistic detector trained offline on
# (turbidity, peak NIR contrast, TDS-deviation) -> "particle present".
# Chosen so that:
#   - calm clean water (turb 2 NTU, flat NIR) -> ~0.05
#   - a moderate event   (turb 9 NTU, NIR peak 0.4 above bg) -> ~0.7
#   - heavy event        (turb 18 NTU, NIR peak 0.6 above bg) -> ~0.95
EDGE_COEFS = {
    "intercept": -3.6,
    "turbidity": 0.18,
    "nir_contrast": 5.4,
    "tds_dev": 0.012,
}


def _nir_contrast(nir: list[float]) -> float:
    """Peak-to-baseline contrast across the 6 NIR channels."""
    if not nir:
        return 0.0
    return max(nir) - min(nir)


def edge_predict(reading: SensorReading,
                 threshold: float = 0.5) -> tuple[bool, float]:
    z = (
        EDGE_COEFS["intercept"]
        + EDGE_COEFS["turbidity"] * reading.turbidity_ntu
        + EDGE_COEFS["nir_contrast"] * _nir_contrast(reading.nir_absorbance)
        + EDGE_COEFS["tds_dev"] * abs(reading.tds_ppm - 200.0)
    )
    p = 1.0 / (1.0 + math.exp(-z))
    return p >= threshold, round(p, 3)
