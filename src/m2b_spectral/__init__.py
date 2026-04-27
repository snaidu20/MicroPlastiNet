"""
M2b — Spectral Deep Learning Classifier for MicroPlastiNet.
Classifies polymer type (PE, PET, PP, PS, PVC, Other) from FTIR/Raman spectra.
"""

from .infer import load_model, PolymerClassifier
from .synthetic_spectra import (
    generate_spectrum, generate_dataset,
    POLYMER_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    WAVENUMBERS, N_POINTS,
)
from .model import build_model, SpectralCNN, SpectralMLP

__all__ = [
    "load_model",
    "PolymerClassifier",
    "generate_spectrum",
    "generate_dataset",
    "build_model",
    "SpectralCNN",
    "SpectralMLP",
    "POLYMER_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "WAVENUMBERS",
    "N_POINTS",
]
