"""
synthetic_spectra.py — Realistic synthetic FTIR/Raman spectra generator for 6 polymer classes.

NOTE: This module generates SYNTHETIC spectra because the Rochman Lab datasets
(SLoPP/SLoPP-E and FLOPP/FLOPP-e) require institutional access and cannot be
downloaded in the sandbox. All spectral parameters are derived from published
literature characteristic peaks. Replace synthetic data with real Rochman CSVs
when available (see dataset.py for the real-data loading path).

Characteristic peaks sourced from:
- Rochman et al. (2022) FLOPP/FLOPP-e FTIR library
- Munno et al. (2020) SLoPP-E Raman library
- Käppler et al. (2016) "Analysis of environmental microplastics by vibrational
  microspectroscopy: FTIR, Raman or both?"

Spectral range: 400–4000 cm⁻¹ at 4 cm⁻¹ resolution → 901 data points
Polymers: PE, PET, PP, PS, PVC, Other (6 classes)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# ── Global spectral axis ─────────────────────────────────────────────────────
WAVENUMBER_START = 400    # cm⁻¹
WAVENUMBER_END   = 4000   # cm⁻¹
WAVENUMBER_STEP  = 4      # cm⁻¹ resolution (FTIR standard)
WAVENUMBERS      = np.arange(WAVENUMBER_START, WAVENUMBER_END + WAVENUMBER_STEP,
                              WAVENUMBER_STEP)  # 901 points
N_POINTS         = len(WAVENUMBERS)  # 901

# ── Polymer class labels ─────────────────────────────────────────────────────
POLYMER_CLASSES = ["PE", "PET", "PP", "PS", "PVC", "Other"]
CLASS_TO_IDX    = {p: i for i, p in enumerate(POLYMER_CLASSES)}
IDX_TO_CLASS    = {i: p for p, i in CLASS_TO_IDX.items()}

# ── Characteristic peak definitions ──────────────────────────────────────────
# Each peak: (center_cm1, relative_intensity, width_cm1)
POLYMER_PEAKS: Dict[str, List[Tuple[float, float, float]]] = {
    "PE": [
        # Strong CH₂ stretching asymmetric + symmetric
        (2916, 1.00, 12),   # ν_as(CH₂) strongest
        (2848, 0.90, 10),   # ν_s(CH₂)
        # CH₂ scissoring + rocking
        (1462, 0.55, 8),    # δ(CH₂) scissoring
        (1375, 0.15, 6),    # δ(CH₃) — weak, HDPE
        (720,  0.60, 8),    # ρ(CH₂) rocking — hallmark of long CH₂ chains
        (730,  0.25, 6),    # HDPE crystal split band
    ],
    "PET": [
        # Ester C=O stretch — dominant
        (1715, 1.00, 14),   # ν(C=O) strongest
        (1240, 0.80, 10),   # ν(C-O) asymmetric ester
        (1090, 0.65, 10),   # ν(C-O) symmetric + ring
        (725,  0.50, 8),    # benzene ring out-of-plane
        (1410, 0.30, 8),    # in-plane ring vibration
        (2960, 0.15, 8),    # CH stretch (weak in PET)
        (3060, 0.20, 10),   # aromatic C-H stretch
    ],
    "PP": [
        # Methyl + methylene stretches
        (2950, 1.00, 10),   # ν_as(CH₃)
        (2920, 0.70, 10),   # ν_as(CH₂)
        (2870, 0.55, 8),    # ν_s(CH₃)
        (2840, 0.35, 8),    # ν_s(CH₂)
        # Deformation bands
        (1455, 0.60, 8),    # δ_as(CH₃) + δ(CH₂)
        (1375, 0.55, 6),    # δ_s(CH₃) — doublet hallmark of iPP
        (1360, 0.30, 6),    # δ_s(CH₃) doublet partner
        (998,  0.70, 8),    # ν(C-C) helix regularity band — iPP marker
        (840,  0.50, 8),    # rocking CH₃ — isotactic PP
        (808,  0.20, 6),    # ρ(CH₂)
    ],
    "PS": [
        # Aromatic C-H stretch
        (3082, 0.40, 8),    # aromatic ν(C-H) overtone
        (3060, 0.50, 8),    # aromatic ν(C-H)
        (3025, 1.00, 10),   # monosubstituted aromatic C-H — PS hallmark
        # Ring stretching
        (1601, 0.65, 8),    # ring ν(C=C) band 1
        (1492, 0.55, 7),    # ring ν(C=C) band 2
        # CH₂ + CH deformation
        (1452, 0.45, 7),    # δ(CH₂)
        # Out-of-plane deformation — monosubstituted benzene
        (756,  0.50, 8),    # γ(C-H) out-of-plane 5H
        (700,  0.80, 9),    # ring puckering — very diagnostic
        (538,  0.35, 7),    # ring deformation
    ],
    "PVC": [
        # C-H stretches
        (2960, 0.40, 10),   # ν_as(CH₂)
        (2912, 1.00, 10),   # ν(C-H) PVC specific
        (2851, 0.30, 8),    # ν_s(CH₂)
        # C-Cl stretches — hallmark of PVC
        (615,  0.85, 10),   # ν(C-Cl) main band
        (635,  0.40, 8),    # ν(C-Cl) shoulder
        (690,  0.30, 7),    # ν(C-Cl) atactic
        # Deformation
        (1426, 0.70, 8),    # δ(CH₂)
        (1330, 0.60, 8),    # wagging/twisting CH
        (1255, 0.30, 8),    # twisting
        (960,  0.25, 7),    # skeletal
    ],
    # "Other" = broad noisy background with no dominant polymer peaks
    "Other": [],
}


def _gaussian(x: np.ndarray, center: float, intensity: float, width: float) -> np.ndarray:
    """Gaussian peak function. Width = half-width at 1/e (≈ HWHM/0.83)."""
    return intensity * np.exp(-((x - center) ** 2) / (2 * (width / 2.355) ** 2))


def _generate_baseline(n: int, rng: np.random.Generator,
                        complexity: str = "medium") -> np.ndarray:
    """
    Realistic baseline drift via low-frequency Fourier components + slope.
    complexity: 'low' | 'medium' | 'high'
    """
    x = np.linspace(0, 1, n)
    n_terms = {"low": 2, "medium": 4, "high": 7}[complexity]
    baseline = rng.uniform(0.02, 0.15) * x  # linear slope
    for k in range(1, n_terms + 1):
        amp   = rng.uniform(0, 0.08) / k
        phase = rng.uniform(0, 2 * np.pi)
        baseline += amp * np.sin(2 * np.pi * k * x + phase)
    return baseline


def _add_noise(spectrum: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Add shot noise (Poisson) + detector noise (Gaussian) at specified SNR."""
    signal_level = np.max(spectrum)
    if signal_level < 1e-8:
        signal_level = 1e-4
    sigma = signal_level / snr
    noise = rng.normal(0, sigma, size=spectrum.shape)
    return spectrum + noise


def generate_spectrum(polymer: str, rng: np.random.Generator,
                      noise_snr: float = 25.0,
                      peak_jitter_std: float = 5.0,
                      intensity_scale_range: Tuple[float, float] = (0.5, 1.5),
                      baseline_complexity: str = "medium") -> np.ndarray:
    """
    Generate one synthetic spectrum for a given polymer class.

    Parameters
    ----------
    polymer : str
        One of POLYMER_CLASSES.
    rng : np.random.Generator
        Seeded random generator for reproducibility.
    noise_snr : float
        Signal-to-noise ratio (signal peak / noise std). Default 25.
    peak_jitter_std : float
        Std dev (cm⁻¹) for random peak position jitter. Default 5.
    intensity_scale_range : tuple
        (min, max) multiplier on the whole spectrum intensity. Default (0.5, 1.5).
    baseline_complexity : str
        'low' | 'medium' | 'high' baseline drift. Default 'medium'.

    Returns
    -------
    np.ndarray of shape (901,), absorbance units [0, 1] normalized.
    """
    spectrum = np.zeros(N_POINTS, dtype=np.float32)

    if polymer == "Other":
        # Broad, featureless noisy spectrum with very weak random bumps
        n_random_peaks = rng.integers(1, 6)
        for _ in range(n_random_peaks):
            center = rng.uniform(500, 3800)
            intens = rng.uniform(0.05, 0.25)
            width  = rng.uniform(20, 80)
            spectrum += _gaussian(WAVENUMBERS, center, intens, width).astype(np.float32)
    else:
        peaks = POLYMER_PEAKS[polymer]
        for (center, rel_intensity, width) in peaks:
            # Position jitter: simulate sample-to-sample peak shifts
            jittered_center = center + rng.normal(0, peak_jitter_std)
            # Width jitter: ±20% of nominal
            jittered_width = width * rng.uniform(0.8, 1.2)
            spectrum += _gaussian(WAVENUMBERS, jittered_center,
                                  rel_intensity, jittered_width).astype(np.float32)

    # Global intensity scaling (simulates concentration / path-length variation)
    scale = rng.uniform(*intensity_scale_range)
    spectrum *= scale

    # Baseline drift
    baseline = _generate_baseline(N_POINTS, rng, baseline_complexity).astype(np.float32)
    spectrum += baseline

    # Noise
    spectrum = _add_noise(spectrum, noise_snr, rng).astype(np.float32)

    # Clip negatives (physical constraint: absorbance ≥ 0)
    spectrum = np.clip(spectrum, 0, None)

    # Min-max normalize to [0, 1]
    s_max = spectrum.max()
    if s_max > 1e-8:
        spectrum = spectrum / s_max

    return spectrum


def generate_dataset(n_per_class: int = 500,
                     seed: int = 42,
                     noise_snr_range: Tuple[float, float] = (15.0, 40.0),
                     peak_jitter_std: float = 6.0,
                     save_path: Optional[str] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced synthetic dataset for all 6 polymer classes.

    Parameters
    ----------
    n_per_class : int
        Number of spectra per polymer class. Default 500.
    seed : int
        Master random seed. Default 42.
    noise_snr_range : tuple
        Range of SNR values randomly sampled per spectrum.
    peak_jitter_std : float
        Peak position jitter in cm⁻¹.
    save_path : str, optional
        If provided, save (X, y) as .npz to this path.

    Returns
    -------
    X : np.ndarray, shape (N, 901)  — float32 spectra
    y : np.ndarray, shape (N,)      — int64 class labels
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for cls_idx, polymer in enumerate(POLYMER_CLASSES):
        for i in range(n_per_class):
            snr = rng.uniform(*noise_snr_range)
            # Vary baseline complexity across samples
            bc = rng.choice(["low", "medium", "high"], p=[0.2, 0.6, 0.2])
            spectrum = generate_spectrum(
                polymer,
                rng,
                noise_snr=snr,
                peak_jitter_std=peak_jitter_std,
                baseline_complexity=bc
            )
            X_list.append(spectrum)
            y_list.append(cls_idx)

    X = np.stack(X_list, axis=0).astype(np.float32)   # (3000, 901)
    y = np.array(y_list, dtype=np.int64)               # (3000,)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, X=X, y=y,
                            wavenumbers=WAVENUMBERS,
                            polymer_classes=POLYMER_CLASSES)
        print(f"Saved synthetic dataset → {save_path}")
        print(f"  Shape: X={X.shape}, y={y.shape}")
        print(f"  Classes: {POLYMER_CLASSES}")
        print(f"  Spectra range: {WAVENUMBERS[0]}–{WAVENUMBERS[-1]} cm⁻¹ ({N_POINTS} pts)")

    return X, y


# ── CLI entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic FTIR/Raman spectra")
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="/home/user/workspace/MicroPlastiNet/data/processed/m2b/synthetic_spectra.npz")
    args = parser.parse_args()

    X, y = generate_dataset(
        n_per_class=args.n_per_class,
        seed=args.seed,
        save_path=args.output
    )
    print(f"✓ Generated {len(X)} spectra ({args.n_per_class} per class × {len(POLYMER_CLASSES)} classes)")
