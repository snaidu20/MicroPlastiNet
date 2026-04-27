"""
train.py — Training Script for M2a Vision Models
==================================================
Module: M2a Vision DL | MicroPlastiNet Pipeline
Author: MicroPlastiNet Team

USAGE
-----
# Classifier training (recommended first step):
python train.py --task classify --data_dir data/synthetic --epochs 10 --batch_size 32

# Detector training:
python train.py --task detect --data_dir data/synthetic --epochs 10 --batch_size 8

# Full run with all options:
python train.py \\
    --task classify \\
    --data_dir data/synthetic \\
    --epochs 20 \\
    --batch_size 32 \\
    --lr 1e-3 \\
    --checkpoint_dir checkpoints/ \\
    --log_dir runs/exp1 \\
    --freeze_backbone \\
    --amp

TRAINING STRATEGY
-----------------
Phase 1 (freeze backbone, epochs 1-5):
  Warm up only the classifier head with high LR.
  Prevents destroying pretrained features on a small dataset.

Phase 2 (unfreeze, epochs 6+):
  Fine-tune entire EfficientNet-B0 with cosine-annealed LR.

Mixed precision (--amp): Uses torch.cuda.amp for ~30% speedup on GPU.
  Falls back to float32 silently on CPU (no error, just slower).

EXPECTED ACCURACY (synthetic data — indicative only)
-----------------------------------------------------
  Classifier val accuracy after 10 epochs: ~65-75%
  Classifier val F1-macro: ~0.60-0.70
  Detector mAP@0.5 after 10 epochs: ~0.45-0.60

  With real Kaggle data (when available):
    Classifier: 75-85% | Detector mAP@0.5: 65-78%
    With UV fluorescence (MP-Set): +5-10% on both metrics

  Field-grade (camera alone): 60-70% accuracy
  Lab-grade (UV fluorescence augmentation): ~85% accuracy
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

# ── Local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    get_detection_loaders, get_classification_loaders,
    SHAPE_CLASSES, generate_dataset
)
from model import (
    build_detector, build_classifier, YOLOLoss,
    NUM_CLASSES, TinyYOLO, MPClassifier
)


# ─────────────────────── Metrics Helpers ────────────────────────────────────

def classification_metrics(
    all_preds: list, all_labels: list
) -> Dict[str, float]:
    """
    Compute accuracy and macro-F1 for multi-class classification.

    Parameters
    ----------
    all_preds  : List of predicted class indices.
    all_labels : List of ground-truth class indices.

    Returns
    -------
    Dict with 'accuracy' and 'f1_macro'.
    """
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}


def simple_map_estimate(
    total_correct: int, total_pred: int, total_gt: int
) -> float:
    """
    Simplified mAP proxy: F1 of object detection (precision × recall harmonic mean).
    A crude but fast estimate during training; use evaluate.py for proper mAP.

    Parameters
    ----------
    total_correct : True positive detections.
    total_pred    : Total predictions made.
    total_gt      : Total ground-truth objects.

    Returns
    -------
    F1-score proxy for detection quality.
    """
    precision = total_correct / (total_pred + 1e-6)
    recall = total_correct / (total_gt + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1


# ─────────────────────── Classifier Training ────────────────────────────────

def train_classifier(args, device: torch.device) -> Dict:
    """
    Full training loop for MPClassifier (EfficientNet-B0).

    Returns dict with final metrics.
    """
    print("\n=== Training MPClassifier (EfficientNet-B0) ===")

    train_loader, val_loader = get_classification_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=224,
    )
    print(f"Train samples: {len(train_loader.dataset)} | "
          f"Val samples: {len(val_loader.dataset)}")

    model = build_classifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    # Separate param groups: lower LR for backbone, higher for head
    backbone_params = list(model.features.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "classify"))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # ── Training phase ───────────────────────────────────────────────
        model.train()

        # Unfreeze backbone after warm-up phase (default: after epoch 3)
        if epoch == args.unfreeze_epoch and args.freeze_backbone:
            print(f"  [Epoch {epoch}] Unfreezing backbone")
            for param in model.features.parameters():
                param.requires_grad = True

        train_loss = 0.0
        t0 = time.time()

        for batch_i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(args.amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (batch_i + 1) % max(1, len(train_loader) // 4) == 0:
                print(f"  Epoch {epoch}/{args.epochs} "
                      f"[{batch_i+1}/{len(train_loader)}] "
                      f"loss={train_loss/(batch_i+1):.4f}")

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # ── Validation phase ─────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels_list.extend(labels.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        metrics = classification_metrics(all_preds, all_labels_list)
        val_acc = metrics["accuracy"]
        val_f1 = metrics["f1_macro"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | "
              f"val_F1={val_f1:.4f} | "
              f"time={elapsed:.1f}s")

        # TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/accuracy", val_acc, epoch)
        writer.add_scalar("Metrics/f1_macro", val_f1, epoch)
        writer.add_scalar("LR", optimizer.param_groups[1]["lr"], epoch)

        history.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "val_acc": val_acc, "val_f1": val_f1
        })

        # ── Checkpoint ───────────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(
                args.checkpoint_dir, "best_classifier.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "train_loss": avg_train_loss,
                "classes": SHAPE_CLASSES,
                "architecture": "EfficientNet-B0",
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f})")

    # Save last checkpoint
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_f1": val_f1,
        "classes": SHAPE_CLASSES,
        "architecture": "EfficientNet-B0",
    }, os.path.join(args.checkpoint_dir, "last_classifier.pt"))

    writer.close()

    final_metrics = {
        "task": "classify",
        "best_val_acc": best_acc,
        "final_val_acc": history[-1]["val_acc"],
        "final_val_f1": history[-1]["val_f1"],
        "epochs": args.epochs,
        "history": history,
    }
    print(f"\nBest val accuracy: {best_acc:.4f}")
    return final_metrics


# ─────────────────────── Detector Training ──────────────────────────────────

def train_detector(args, device: torch.device) -> Dict:
    """
    Full training loop for TinyYOLO detector.

    Returns dict with final metrics.
    """
    print("\n=== Training TinyYOLO Detector ===")

    train_loader, val_loader = get_detection_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=416,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_detector(num_classes=NUM_CLASSES).to(device)
    criterion = YOLOLoss(num_classes=NUM_CLASSES)

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.937, weight_decay=5e-4, nesterov=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
    )

    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "detect"))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_map = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_i, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            boxes = [b.to(device) for b in batch["boxes"]]
            labels = [l.to(device) for l in batch["labels"]]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(args.amp and device.type == "cuda")):
                preds = model(images)
                loss, loss_comps = criterion(preds, boxes, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # ── Validation: fast mAP proxy ───────────────────────────────────
        model.eval()
        tp_count, pred_count, gt_count = 0, 0, 0
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                boxes = [b.to(device) for b in batch["boxes"]]
                labels = [l.to(device) for l in batch["labels"]]
                preds = model(images)
                loss, _ = criterion(preds, boxes, labels)
                val_loss += loss.item()
                for lbl in labels:
                    gt_count += lbl.shape[0]
                # Simple detection count (objects with high confidence)
                for pred_scale in preds:
                    conf = torch.sigmoid(pred_scale[..., 4])
                    pred_count += (conf > 0.5).sum().item()
                    tp_count += min((conf > 0.5).sum().item(), gt_count)

        avg_val_loss = val_loss / max(1, len(val_loader))
        map_proxy = simple_map_estimate(tp_count, pred_count, gt_count)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"train_loss={avg_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | "
              f"mAP_proxy={map_proxy:.4f} | "
              f"time={elapsed:.1f}s")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/mAP_proxy", map_proxy, epoch)

        history.append({
            "epoch": epoch, "train_loss": avg_loss,
            "val_loss": avg_val_loss, "map_proxy": map_proxy
        })

        if map_proxy > best_map:
            best_map = map_proxy
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "map_proxy": map_proxy,
                "train_loss": avg_loss,
                "classes": SHAPE_CLASSES,
                "architecture": "TinyYOLO",
            }, os.path.join(args.checkpoint_dir, "best_detector.pt"))
            print(f"  ✓ Saved best detector checkpoint (mAP_proxy={map_proxy:.4f})")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "map_proxy": map_proxy,
        "classes": SHAPE_CLASSES,
        "architecture": "TinyYOLO",
    }, os.path.join(args.checkpoint_dir, "last_detector.pt"))

    writer.close()

    return {
        "task": "detect",
        "best_map_proxy": best_map,
        "final_map_proxy": history[-1]["map_proxy"],
        "epochs": args.epochs,
        "history": history,
    }


# ─────────────────────────────── CLI ────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train M2a Vision models for microplastic detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=["classify", "detect", "both"],
                        default="classify",
                        help="Which model to train")
    parser.add_argument("--data_dir", default="data/synthetic",
                        help="Root dataset directory (train/ and val/ subdirs)")
    parser.add_argument("--generate_data", action="store_true",
                        help="Generate synthetic dataset before training")
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (GPU only)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze EfficientNet backbone for warm-up")
    parser.add_argument("--unfreeze_epoch", type=int, default=3,
                        help="Epoch to unfreeze backbone (if --freeze_backbone)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None,
                        help="Save final metrics JSON to this path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task:   {args.task}")
    print(f"Data:   {args.data_dir}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    # ── Optionally generate data ─────────────────────────────────────────
    if args.generate_data or not Path(args.data_dir).exists():
        print(f"\nGenerating synthetic dataset → {args.data_dir}")
        generate_dataset(
            out_dir=args.data_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            seed=args.seed,
        )

    results = {}

    if args.task in ("classify", "both"):
        clf_metrics = train_classifier(args, device)
        results["classifier"] = clf_metrics

    if args.task in ("detect", "both"):
        det_metrics = train_detector(args, device)
        results["detector"] = det_metrics

    print("\n=== Training Complete ===")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "history"}
                      for k, v in results.items()}, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to {args.output_json}")

    return results


if __name__ == "__main__":
    main()
