"""
finalize.py — Finalize training (save metrics JSON), run evaluate.py, 
run sample inference. Called after training completes or is interrupted
with a valid checkpoint.
"""
import os
import sys
import json
import csv
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders, POLYMER_CLASSES
from model import build_model
from infer import load_model
from evaluate import evaluate

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(_BASE, "data", "processed", "m2b")
ASSETS_DIR = os.path.join(_BASE, "assets")

def finalize_training(arch="cnn", seed=42):
    ckpt_path = os.path.join(PROC_DIR, f"m2b_{arch}_best.pt")
    log_path  = os.path.join(PROC_DIR, f"m2b_{arch}_train_log.csv")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    # Read history from CSV
    history = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append({k: float(v) for k, v in row.items()})

    # Load checkpoint for meta
    device = torch.device("cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Get data for test eval
    _, _, test_loader, meta = get_dataloaders(seed=seed, augment_train=False)
    
    model = build_model(arch, n_classes=ckpt.get("n_classes", 6),
                        input_len=ckpt.get("input_dim", 901))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Compute test accuracy
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_true, all_pred, all_proba = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            probs  = torch.softmax(logits, dim=-1).numpy()
            preds  = probs.argmax(axis=1)
            total_loss += loss.item() * len(y_batch)
            correct    += (preds == y_batch.numpy()).sum()
            total      += len(y_batch)
            all_true.extend(y_batch.numpy())
            all_pred.extend(preds)
            all_proba.extend(probs)

    test_acc  = correct / total
    test_loss = total_loss / total

    best_val_acc = ckpt.get("val_acc", max((h.get("val_acc", 0) for h in history), default=0))
    best_epoch   = ckpt.get("epoch", len(history))

    print(f"[INFO] Test accuracy: {test_acc:.4%} (best val: {best_val_acc:.4%} @ epoch {best_epoch})")

    # Save metrics JSON
    metrics = {
        "arch":           arch,
        "seed":           seed,
        "best_epoch":     int(best_epoch),
        "best_val_acc":   float(best_val_acc),
        "test_acc":       float(test_acc),
        "test_loss":      float(test_loss),
        "n_params":       sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_train":        meta["n_train"],
        "n_val":          meta["n_val"],
        "n_test":         meta["n_test"],
        "class_names":    meta["class_names"],
        "data_source":    meta["source"],
        "history":        history,
    }
    metrics_path = os.path.join(PROC_DIR, f"m2b_{arch}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved → {metrics_path}")

    return metrics, np.array(all_true), np.array(all_pred), np.array(all_proba)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="cnn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = finalize_training(args.arch, args.seed)
    if result is None:
        sys.exit(1)
    
    metrics, y_true, y_pred, y_proba = result

    print("\n[Step 2] Running full evaluation + generating plots...")
    eval_report = evaluate(arch=args.arch, seed=args.seed, save_preds=True)
    
    print(f"\n[Step 3] Sample inference demo...")
    from synthetic_spectra import generate_spectrum, POLYMER_CLASSES
    
    clf = load_model(arch=args.arch)
    rng = np.random.default_rng(777)
    
    sample_results = []
    print(f"\n{'Polymer':>8} | {'Predicted':>9} | {'Confidence':>11} | Correct")
    print("─" * 55)
    for polymer in POLYMER_CLASSES:
        spectrum = generate_spectrum(polymer, rng)
        result   = clf.predict(spectrum)
        correct  = "✓" if result["polymer"] == polymer else "✗"
        sample_results.append({
            "true": polymer,
            "pred": result["polymer"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "correct": result["polymer"] == polymer,
        })
        print(f"{polymer:>8} | {result['polymer']:>9} | {result['confidence']:>10.4f} | {correct}")
    
    # Save sample inference output
    samples_path = os.path.join(PROC_DIR, "m2b_sample_inference.json")
    with open(samples_path, "w") as f:
        json.dump(sample_results, f, indent=2)
    print(f"\n[INFO] Sample inference → {samples_path}")
    
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Architecture:      {args.arch.upper()}")
    print(f"  Best Val Accuracy: {metrics['best_val_acc']:.4%}")
    print(f"  Test Accuracy:     {metrics['test_acc']:.4%}")
    print(f"  Macro AUC:         {eval_report['macro_auc']:.4f}")
    print(f"  Epochs trained:    {metrics['best_epoch']}")
    print(f"  Parameters:        {metrics['n_params']:,}")
    print(f"{'='*60}")
    print(f"\n  Artifacts:")
    print(f"    Checkpoint:     {os.path.join(PROC_DIR, f'm2b_{args.arch}_best.pt')}")
    print(f"    Metrics JSON:   {os.path.join(PROC_DIR, f'm2b_{args.arch}_metrics.json')}")
    print(f"    Confusion PNG:  {os.path.join(PROC_DIR.replace('processed/m2b', 'assets'), 'm2b_confusion.png').replace('data/', '')}")
    print(f"    Eval Report:    {os.path.join(PROC_DIR, 'm2b_eval_report.json')}")
