"""
train.py — Training pipeline for M2b Spectral Polymer Classifier.

Features:
  - Stratified train/val/test split (70/15/15)
  - Class-weighted cross-entropy loss (handles imbalance)
  - Cosine annealing LR schedule with warm restarts
  - Early stopping on validation loss (patience=10)
  - Best-model checkpointing (val accuracy)
  - Per-epoch CSV logging
  - Reproducible via seeding

Usage:
    python train.py                          # default settings
    python train.py --arch mlp --epochs 40   # MLP baseline
    python train.py --epochs 50 --lr 3e-4    # custom
"""

import os
import sys
import json
import time
import argparse
import csv
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── Add module root to path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders, POLYMER_CLASSES
from model import build_model

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR    = os.path.join(_BASE, "data", "processed", "m2b")
ASSETS_DIR  = os.path.join(_BASE, "assets")
os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ── Seeding ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module,
             loader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "min"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.counter   = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Main training function ────────────────────────────────────────────────────

def train(
    arch:           str   = "cnn",
    n_per_class:    int   = 500,
    epochs:         int   = 50,
    batch_size:     int   = 64,
    lr:             float = 5e-4,
    weight_decay:   float = 1e-4,
    patience:       int   = 10,
    seed:           int   = 42,
    device_str:     str   = "auto",
) -> Dict:
    """
    Full training run. Returns metrics dict.
    """
    set_seed(seed)

    device = (torch.device("cuda") if device_str == "auto" and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[INFO] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        n_per_class=n_per_class,
        seed=seed,
        batch_size=batch_size,
        augment_train=True,
    )
    class_weights = meta["class_weights"].to(device)
    print(f"[INFO] Classes: {meta['class_names']}")
    print(f"[INFO] Class weights: {class_weights.cpu().numpy().round(3)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(arch, n_classes=meta["n_classes"],
                        input_len=meta["input_dim"]).to(device)
    print(f"[INFO] Model: {arch.upper()}  |  params={model.n_parameters():,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr / 20
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_epoch     = 0
    ckpt_path      = os.path.join(CKPT_DIR, f"m2b_{arch}_best.pt")
    log_path       = os.path.join(CKPT_DIR, f"m2b_{arch}_train_log.csv")
    early_stop     = EarlyStopping(patience=patience, mode="min")
    history        = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr"])

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | "
          f"{'Val Loss':>10} | {'Val Acc':>9} | {'LR':>9}")
    print("─" * 74)

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        current_lr            = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr
        })
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}",
                                     f"{val_loss:.5f}", f"{val_acc:.5f}", f"{current_lr:.6f}"])

        print(f"{epoch:>6} | {train_loss:>11.5f} | {train_acc:>9.4f}% | "
              f"{val_loss:>10.5f} | {val_acc:>8.4f}% | {current_lr:>9.6f}")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":       epoch,
                "arch":        arch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "train_acc":   train_acc,
                "class_names": meta["class_names"],
                "input_dim":   meta["input_dim"],
                "n_classes":   meta["n_classes"],
            }, ckpt_path)

        # Early stopping
        if early_stop(val_loss):
            print(f"\n[INFO] Early stopping at epoch {epoch} "
                  f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch})")
            break

    elapsed = time.time() - t_start
    print(f"\n[INFO] Training complete in {elapsed:.1f}s")
    print(f"[INFO] Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"[INFO] Checkpoint: {ckpt_path}")

    # ── Final test evaluation with best model ─────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[INFO] Test accuracy: {test_acc:.4f}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "arch":           arch,
        "seed":           seed,
        "best_epoch":     best_epoch,
        "best_val_acc":   best_val_acc,
        "test_acc":       test_acc,
        "test_loss":      test_loss,
        "n_params":       model.n_parameters(),
        "n_train":        meta["n_train"],
        "n_val":          meta["n_val"],
        "n_test":         meta["n_test"],
        "class_names":    meta["class_names"],
        "data_source":    meta["source"],
        "training_time_s": elapsed,
        "history":        history,
    }
    metrics_path = os.path.join(CKPT_DIR, f"m2b_{arch}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved → {metrics_path}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M2b Spectral Classifier")
    parser.add_argument("--arch",         type=str,   default="cnn",
                        choices=["cnn", "mlp"])
    parser.add_argument("--n_per_class",  type=int,   default=500)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       type=str,   default="auto")
    args = parser.parse_args()

    metrics = train(
        arch=args.arch,
        n_per_class=args.n_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        device_str=args.device,
    )
    print(f"\n{'='*50}")
    print(f"  FINAL TEST ACCURACY: {metrics['test_acc']:.4%}")
    print(f"  BEST VAL  ACCURACY:  {metrics['best_val_acc']:.4%}")
    print(f"{'='*50}")
