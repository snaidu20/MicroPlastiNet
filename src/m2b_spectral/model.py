"""
model.py — 1D-CNN and MLP models for polymer spectral classification.

Architecture (1D-CNN):
    Input: (B, 1, 901) — batch × channel × wavenumber
    Block 1: Conv1d(1→32, k=7)   → BN → ReLU → MaxPool(2)  → (B, 32, 450)
    Block 2: Conv1d(32→64, k=5)  → BN → ReLU → MaxPool(2)  → (B, 64, 225)
    Block 3: Conv1d(64→128, k=5) → BN → ReLU → MaxPool(2)  → (B, 128, 112)
    Block 4: Conv1d(128→256, k=3)→ BN → ReLU → MaxPool(2)  → (B, 256, 56)
    GlobalAvgPool → (B, 256)
    FC(256→128) → ReLU → Dropout(0.4)
    FC(128→64)  → ReLU → Dropout(0.2)
    FC(64→6)    → (logits, no softmax — use CrossEntropyLoss)

MLP Baseline:
    FC(901→512) → BN → ReLU → Dropout(0.4)
    FC(512→256) → BN → ReLU → Dropout(0.3)
    FC(256→128) → BN → ReLU → Dropout(0.2)
    FC(128→6)

Both models expose a `predict_proba` method for the inference API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


# ── 1D-CNN Block helper ───────────────────────────────────────────────────────

class ConvBlock1D(nn.Module):
    """Conv1d → BatchNorm → ReLU → optional MaxPool."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, pool: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2  # "same" padding
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── 1D-CNN ────────────────────────────────────────────────────────────────────

class SpectralCNN(nn.Module):
    """
    1D-CNN for polymer spectral classification.

    Input shape:  (B, 1, 901)
    Output shape: (B, n_classes)  — raw logits

    Reference architecture inspired by:
    - Zhang et al. (2018) "Deep learning for spectroscopy"
    - Weid et al. (2022) "Machine learning for FTIR microplastics"
    """

    def __init__(self, n_classes: int = 6, input_len: int = 901,
                 dropout_fc: float = 0.4):
        super().__init__()
        self.n_classes = n_classes

        # ── Convolutional backbone ──────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1: broad receptive field to capture wide peaks
            ConvBlock1D(1,   32,  kernel_size=11, pool=True),   # → (B, 32, 450)
            ConvBlock1D(32,  32,  kernel_size=7,  pool=False),  # residual width

            # Block 2
            ConvBlock1D(32,  64,  kernel_size=7,  pool=True),   # → (B, 64, 225)
            ConvBlock1D(64,  64,  kernel_size=5,  pool=False),

            # Block 3
            ConvBlock1D(64,  128, kernel_size=5,  pool=True),   # → (B, 128, 112)
            ConvBlock1D(128, 128, kernel_size=3,  pool=False),

            # Block 4: fine-grained peak discrimination
            ConvBlock1D(128, 256, kernel_size=3,  pool=True),   # → (B, 256, 56)
            ConvBlock1D(256, 256, kernel_size=3,  pool=False),
        )

        # ── Global average pooling → flatten ───────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, 256, 1) → (B, 256)

        # ── Classifier head ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc / 2),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 901) → logits: (B, n_classes)"""
        x = self.features(x)          # (B, 256, L)
        x = self.global_pool(x)       # (B, 256, 1)
        x = x.squeeze(-1)             # (B, 256)
        return self.classifier(x)     # (B, n_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. x: (B, 1, 901)"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── MLP Baseline ──────────────────────────────────────────────────────────────

class SpectralMLP(nn.Module):
    """
    Fully-connected baseline for ablation comparison.

    Input shape:  (B, 1, 901)  — same as CNN (channel dim squeezed internally)
    Output shape: (B, n_classes)  — raw logits
    """

    def __init__(self, n_classes: int = 6, input_len: int = 901,
                 dropout: float = 0.4):
        super().__init__()
        self.n_classes = n_classes

        self.net = nn.Sequential(
            nn.Flatten(),              # (B, 901)
            nn.Linear(input_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 901) → logits: (B, n_classes)"""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Residual 1D-CNN (optional deeper variant) ────────────────────────────────

class ResidualBlock1D(nn.Module):
    """1D Residual block with skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(arch: str = "cnn", n_classes: int = 6,
                input_len: int = 901, **kwargs) -> nn.Module:
    """
    Factory function.

    arch: 'cnn' → SpectralCNN  |  'mlp' → SpectralMLP
    """
    if arch == "cnn":
        return SpectralCNN(n_classes=n_classes, input_len=input_len, **kwargs)
    elif arch == "mlp":
        return SpectralMLP(n_classes=n_classes, input_len=input_len, **kwargs)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose 'cnn' or 'mlp'.")


# ── CLI summary ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for arch in ["cnn", "mlp"]:
        model = build_model(arch)
        dummy = torch.randn(4, 1, 901)
        out   = model(dummy)
        print(f"{arch.upper():>4}  |  params={model.n_parameters():,}  "
              f"|  output shape={tuple(out.shape)}")
