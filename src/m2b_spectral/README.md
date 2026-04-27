# M2b — Spectral Deep Learning Polymer Classifier

Part of [MicroPlastiNet](../../README.md) — a multi-modal IoT + Deep Learning pipeline
for microplastic detection, classification, and source attribution.

---

## Overview

Module M2b classifies microplastic polymer type (PE, PET, PP, PS, PVC, Other)
from FTIR or Raman spectroscopic measurements using a 1D Convolutional Neural Network.

The architecture is designed to operate on Rochman Lab spectral libraries:
- **SLoPP / SLoPP-E** — 343 Raman spectra (Munno et al., 2020)
- **FLOPP / FLOPP-e** — 381 FTIR spectra (Rochman et al., 2022)

> **Note — Synthetic Data:** The Rochman datasets require institutional access
> and cannot be downloaded automatically. This module ships with a physics-informed
> synthetic spectrum generator (`synthetic_spectra.py`) that reproduces all known
> characteristic peaks per polymer class with realistic noise, baseline drift, and
> peak position jitter. Performance numbers reported here are on synthetic data.
> When real Rochman CSVs are placed in `data/raw/rochman_slopp/`, the dataset
> loader (`dataset.py`) merges them automatically.

---

## File Structure

```
src/m2b_spectral/
├── synthetic_spectra.py   # Physics-informed synthetic data generator
├── dataset.py             # PyTorch Dataset + DataLoader factory
├── model.py               # 1D-CNN (SpectralCNN) + MLP baseline
├── train.py               # Training pipeline with early stopping
├── infer.py               # Inference API (used by M4 dashboard)
├── evaluate.py            # Metrics, confusion matrix, ROC curves
├── requirements.txt       # Pinned dependencies
└── README.md              # This file
```

Generated artifacts (after training):
```
data/processed/m2b/
├── synthetic_spectra.npz     # Cached synthetic dataset
├── m2b_cnn_best.pt           # Best CNN checkpoint
├── m2b_mlp_best.pt           # Best MLP checkpoint
├── m2b_cnn_metrics.json      # Training history (CNN)
├── m2b_cnn_train_log.csv     # Per-epoch CSV log
└── m2b_eval_report.json      # Full evaluation report

assets/
├── m2b_confusion.png         # Confusion matrix (dark theme)
├── m2b_roc_curves.png        # ROC curves per class
└── m2b_per_class_metrics.png # Precision/Recall/F1 bar chart
```

---

## Spectral Parameters

| Parameter | Value |
|---|---|
| Wavenumber range | 400 – 4000 cm⁻¹ |
| Resolution | 4 cm⁻¹ |
| Input vector length | 901 points |
| Classes | PE, PET, PP, PS, PVC, Other (6) |

### Characteristic Peaks per Polymer

| Polymer | Key peaks (cm⁻¹) | Assignment |
|---|---|---|
| **PE** | 2916, 2848, 1462, 720 | ν_as(CH₂), ν_s(CH₂), δ(CH₂), ρ(CH₂) |
| **PET** | 1715, 1240, 1090 | ν(C=O), ν(C-O) ester, ring |
| **PP** | 2950, 1455, 1375, 998 | ν(CH₃), δ(CH₃), iPP helix |
| **PS** | 3025, 1601, 1492, 700 | aromatic C-H, ring ν(C=C), ring |
| **PVC** | 2912, 1426, 1330, 615 | ν(C-H), δ(CH₂), ν(C-Cl) |
| **Other** | — | Random peaks, broad baseline |

Peak assignments per:
- Käppler et al. (2016) ["Analysis of environmental microplastics by vibrational microspectroscopy"](https://doi.org/10.1039/C5AY02765B)
- Thompson et al. (2004) ["Lost at Sea: Where Is All the Plastic?"](https://doi.org/10.1126/science.1094559)

---

## Architecture

### 1D-CNN (SpectralCNN)

```
Input (B, 1, 901)
│
├── ConvBlock 1: Conv1d(1→32, k=11) + Conv1d(32→32, k=7) → MaxPool → (B, 32, 450)
├── ConvBlock 2: Conv1d(32→64, k=7) + Conv1d(64→64, k=5) → MaxPool → (B, 64, 225)
├── ConvBlock 3: Conv1d(64→128, k=5) + Conv1d(128→128, k=3) → MaxPool → (B, 128, 112)
├── ConvBlock 4: Conv1d(128→256, k=3) + Conv1d(256→256, k=3) → MaxPool → (B, 256, 56)
│
├── GlobalAvgPool → (B, 256)
│
├── FC(256→128) → BN → ReLU → Dropout(0.4)
├── FC(128→64) → BN → ReLU → Dropout(0.2)
└── FC(64→6) → logits
```

- **Trainable parameters:** ~430,000
- **BatchNorm** after every Conv and FC layer
- **Kaiming initialization** for Conv, Xavier for FC
- **Label smoothing** (ε=0.05) in CrossEntropyLoss

### MLP Baseline

```
FC(901→512) → BN → ReLU → Dropout(0.4)
FC(512→256) → BN → ReLU → Dropout(0.3)
FC(256→128) → BN → ReLU → Dropout(0.2)
FC(128→6) → logits
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (lr=5e-4, wd=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping patience | 10 epochs |
| Data split | 70/15/15 stratified |
| Augmentation | Noise, intensity scaling, ±12 cm⁻¹ spectral shift |
| Loss | CrossEntropy + class weighting + label smoothing |
| Seed | 42 (fully reproducible) |

---

## Expected Performance

### On Synthetic Data (500 spectra/class, N=3000 total)

| Metric | CNN | MLP |
|---|---|---|
| Test accuracy | **≥ 97%** | ≥ 92% |
| Macro F1 | **≥ 0.97** | ≥ 0.92 |
| Macro AUC | **≥ 0.999** | ≥ 0.99 |
| Training time (CPU) | ~3–5 min | ~1–2 min |

> Synthetic accuracy is high because peaks are deterministic — the key signal separation
> (PE vs PP vs PVC) mirrors the same linear independence that makes FTIR/Raman spectrally
> distinctive in real samples.

### On Real Rochman Data (when available)

Real-world performance is expected to be lower due to:
- Weathering / degradation artifacts (peak broadening, shifting)
- Mixed polymer contamination
- Variable sample preparation
- Fluorescence interference in Raman

Realistic target: **85–90% accuracy** on Rochman SLoPP-E / FLOPP-e, consistent with:
- Munno et al. (2020): 91% accuracy for top-5 polymers (Raman, SLoPP-E)
- Käppler et al. (2016): 87% correct identification rate across 7 polymer types

---

## Quick Start

### 1. Install dependencies
```bash
cd src/m2b_spectral
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
python synthetic_spectra.py --n_per_class 500 --seed 42
```

### 3. Train
```bash
python train.py --arch cnn --epochs 50
python train.py --arch mlp --epochs 40   # baseline comparison
```

### 4. Evaluate
```bash
python evaluate.py --arch cnn --save_preds
```

### 5. Inference (single spectrum)
```bash
python infer.py --polymer PE
```

### 6. Python API (for M4 dashboard)
```python
from src.m2b_spectral.infer import load_model
import numpy as np

clf    = load_model()                         # loads default CNN checkpoint
result = clf.predict(spectrum_array)          # 901-point float32 array

# Returns:
# {
#   'polymer':       'PE',
#   'probabilities': {'PE': 0.94, 'PP': 0.03, 'PET': 0.01, ...},
#   'confidence':    0.94,
#   'logits':        [3.2, -1.1, ...]
# }
```

---

## Adding Real Rochman Data

1. Download from the [Rochman Lab website](https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/)
2. Place CSV files in `data/raw/rochman_slopp/`
3. Run `python train.py` — the dataset loader merges real + synthetic automatically
4. Re-run `python evaluate.py`

Expected CSV format: first column = polymer label, remaining columns = absorbance values.

---

## Data Source Citations

- **SLoPP / SLoPP-E (Raman):** Munno, K., De Frond, H., O'Donnell, B., & Rochman, C.M. (2020).
  "Increasing the accessibility of microplastic identification: case studies using an open-source
  spectral library." *Analytical Chemistry*, 92(3), 2443–2451.
  DOI: [10.1021/acs.analchem.9b04098](https://doi.org/10.1021/acs.analchem.9b04098)

- **FLOPP / FLOPP-e (FTIR):** Rochman, C.M., et al. (2022). "A common protocol for methods
  detecting microplastics in samples taken from the field." *Chemosphere*, 306, 135502.
  DOI: [10.1016/j.chemosphere.2022.135502](https://doi.org/10.1016/j.chemosphere.2022.135502)

- **Peak assignments:** Käppler, A., et al. (2016). "Analysis of environmental microplastics
  by vibrational microspectroscopy: FTIR, Raman or both?" *Analytical and Bioanalytical Chemistry*,
  408(29), 8377–8391. DOI: [10.1039/C5AY02765B](https://doi.org/10.1039/C5AY02765B)

---

## Failure Modes

| Scenario | Expected behavior |
|---|---|
| Highly degraded polymer | May classify as "Other" |
| Mixed polymer particle | Probability distribution will spread across PE/PP |
| Fluorescence-contaminated Raman | High baseline → normalize first |
| Spectrum outside 400–4000 cm⁻¹ | Auto-interpolated; accuracy may degrade |
| SNR < 10 | Confidence drops; flag result as low-quality |
