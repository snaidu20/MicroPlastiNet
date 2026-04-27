"""
infer.py — Inference API for M2b Spectral Polymer Classifier.

Clean API designed for consumption by M4 dashboard and other modules.

Public interface:
    load_model(ckpt_path=None, arch='cnn') → PolymerClassifier
    classifier.predict(spectrum) → {polymer, probabilities, confidence}

Supports:
  - numpy array input (901 wavenumber points, or any length → resampled)
  - CSV row input (single row or dict with wavenumber keys)
  - Batch inference for multiple spectra

Usage:
    from infer import load_model
    clf = load_model()
    result = clf.predict(spectrum_array)
    # → {'polymer': 'PE', 'probabilities': {'PE': 0.94, ...}, 'confidence': 0.94}
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model, SpectralCNN, SpectralMLP
from synthetic_spectra import POLYMER_CLASSES, N_POINTS, WAVENUMBERS

# ── Default paths ─────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CNN_CKPT = os.path.join(_BASE, "data", "processed", "m2b", "m2b_cnn_best.pt")
DEFAULT_MLP_CKPT = os.path.join(_BASE, "data", "processed", "m2b", "m2b_mlp_best.pt")


class PolymerClassifier:
    """
    Polymer type classifier from FTIR/Raman spectra.

    Parameters
    ----------
    model : nn.Module
        Loaded PyTorch model (SpectralCNN or SpectralMLP).
    device : torch.device
    class_names : list[str]
        Ordered polymer class names matching model output indices.
    """

    def __init__(self, model, device: torch.device, class_names: list):
        self.model       = model
        self.device      = device
        self.class_names = class_names
        self.model.eval()

    def _preprocess(self, spectrum: np.ndarray) -> torch.Tensor:
        """
        Preprocess raw spectrum numpy array:
          1. Resample to 901 points if needed (linear interpolation)
          2. Clip negative values
          3. Min-max normalize to [0, 1]
          4. Add batch + channel dims → (1, 1, 901)
        """
        spectrum = np.asarray(spectrum, dtype=np.float32).ravel()

        # Resample if not 901 points
        if len(spectrum) != N_POINTS:
            src_wn   = np.linspace(WAVENUMBERS[0], WAVENUMBERS[-1], len(spectrum))
            spectrum = np.interp(WAVENUMBERS, src_wn, spectrum).astype(np.float32)

        spectrum = np.clip(spectrum, 0, None)
        mx = spectrum.max()
        if mx > 1e-8:
            spectrum = spectrum / mx

        tensor = torch.tensor(spectrum, dtype=torch.float32)
        return tensor.unsqueeze(0).unsqueeze(0)   # (1, 1, 901)

    def predict(self, spectrum: np.ndarray) -> dict:
        """
        Classify a single spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            1-D array of absorbance/intensity values (any length; resampled to 901).

        Returns
        -------
        dict with keys:
            polymer       (str)   — top predicted class, e.g. "PE"
            probabilities (dict)  — {class: float} for all 6 classes, sums to 1
            confidence    (float) — probability of top class ∈ [0, 1]
            logits        (list)  — raw model logits for debugging
        """
        x = self._preprocess(spectrum).to(self.device)

        with torch.no_grad():
            logits = self.model(x)           # (1, 6)
            probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        top_idx    = int(np.argmax(probs))
        polymer    = self.class_names[top_idx]
        confidence = float(probs[top_idx])

        return {
            "polymer":       polymer,
            "probabilities": {cls: float(p)
                              for cls, p in zip(self.class_names, probs)},
            "confidence":    confidence,
            "logits":        logits.squeeze(0).cpu().numpy().tolist(),
        }

    def predict_batch(self, spectra: np.ndarray) -> list:
        """
        Classify a batch of spectra.

        Parameters
        ----------
        spectra : np.ndarray, shape (N, L)
            N spectra of length L.

        Returns
        -------
        list of N prediction dicts (same format as predict()).
        """
        results = []
        # Process individually to handle variable lengths gracefully
        for i in range(len(spectra)):
            results.append(self.predict(spectra[i]))
        return results

    def predict_from_csv_row(self, row: dict) -> dict:
        """
        Classify from a CSV row dict.

        Expected: keys are wavenumber strings (e.g. "2916"), values are floats.
        Or keys 'wavenumber' mapped to list and 'intensity' mapped to list.
        """
        if "wavenumber" in row and "intensity" in row:
            wn  = np.asarray(row["wavenumber"], dtype=np.float32)
            val = np.asarray(row["intensity"],  dtype=np.float32)
            spectrum = np.interp(WAVENUMBERS, wn, val).astype(np.float32)
        else:
            # Assume dict keys are wavenumber numbers
            wns = sorted(float(k) for k in row.keys())
            val = np.array([float(row[str(int(w)) if str(int(w)) in row else w])
                            for w in wns], dtype=np.float32)
            spectrum = np.interp(WAVENUMBERS, np.array(wns), val).astype(np.float32)
        return self.predict(spectrum)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(ckpt_path: str = None,
               arch: str = "cnn",
               device_str: str = "auto") -> PolymerClassifier:
    """
    Load a trained PolymerClassifier from checkpoint.

    Parameters
    ----------
    ckpt_path : str, optional
        Path to .pt checkpoint. If None, uses default CNN checkpoint.
    arch : str
        'cnn' or 'mlp'. Ignored if checkpoint contains arch info.
    device_str : str
        'auto', 'cpu', or 'cuda'.

    Returns
    -------
    PolymerClassifier instance ready for inference.
    """
    if ckpt_path is None:
        ckpt_path = DEFAULT_CNN_CKPT if arch == "cnn" else DEFAULT_MLP_CKPT

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run train.py first to generate the model checkpoint."
        )

    device = (torch.device("cuda") if device_str == "auto" and torch.cuda.is_available()
              else torch.device("cpu"))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    arch_from_ckpt   = ckpt.get("arch", arch)
    class_names      = ckpt.get("class_names", POLYMER_CLASSES)
    n_classes        = ckpt.get("n_classes", len(class_names))
    input_dim        = ckpt.get("input_dim", N_POINTS)

    model = build_model(arch_from_ckpt, n_classes=n_classes, input_len=input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print(f"[INFO] Loaded {arch_from_ckpt.upper()} from {ckpt_path}")
    print(f"       Val acc at checkpoint: {ckpt.get('val_acc', 'N/A')}")
    print(f"       Classes: {class_names}")

    return PolymerClassifier(model, device, class_names)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from synthetic_spectra import generate_spectrum
    import random

    parser = argparse.ArgumentParser(description="Run inference on a synthetic test spectrum")
    parser.add_argument("--polymer", type=str, default=None,
                        help="Polymer to test (default: random)")
    parser.add_argument("--arch",    type=str, default="cnn")
    parser.add_argument("--seed",    type=int, default=99)
    args = parser.parse_args()

    rng    = np.random.default_rng(args.seed)
    polymer = args.polymer or random.choice(POLYMER_CLASSES)
    print(f"\n[Demo] Generating test spectrum for: {polymer}")

    spectrum = generate_spectrum(polymer, rng)

    clf    = load_model(arch=args.arch)
    result = clf.predict(spectrum)

    print(f"\n{'─'*45}")
    print(f"  True polymer:     {polymer}")
    print(f"  Predicted:        {result['polymer']}")
    print(f"  Confidence:       {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
    print(f"\n  Probabilities:")
    for cls, prob in sorted(result["probabilities"].items(),
                             key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:>6}: {prob:.4f}  {bar}")
    print(f"{'─'*45}")
    print(f"\n  Correct: {'✓' if result['polymer'] == polymer else '✗'}")
