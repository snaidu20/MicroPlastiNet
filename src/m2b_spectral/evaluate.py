"""
evaluate.py — Full evaluation suite for M2b Spectral Classifier.

Generates:
  - Confusion matrix PNG  → assets/m2b_confusion.png
  - ROC curves PNG        → assets/m2b_roc_curves.png
  - Per-class metrics     → assets/m2b_per_class_metrics.png
  - JSON report           → data/processed/m2b/m2b_eval_report.json

Usage:
    python evaluate.py                      # use default CNN checkpoint
    python evaluate.py --arch mlp           # evaluate MLP
    python evaluate.py --save_preds         # also save all predictions CSV
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders, POLYMER_CLASSES
from infer import load_model

# Matplotlib: non-interactive backend for headless rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(_BASE, "assets")
PROC_DIR   = os.path.join(_BASE, "data", "processed", "m2b")
os.makedirs(ASSETS_DIR, exist_ok=True)


# ── Inference on test set ─────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(clf, test_loader, device):
    """Run full test set through classifier. Returns (y_true, y_pred, y_proba)."""
    clf.model.eval()
    all_true, all_pred, all_proba = [], [], []

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits  = clf.model(X_batch)
        probs   = torch.softmax(logits, dim=-1).cpu().numpy()
        preds   = probs.argmax(axis=1)

        all_true.extend(y_batch.numpy())
        all_pred.extend(preds)
        all_proba.extend(probs)

    return (np.array(all_true), np.array(all_pred),
            np.array(all_proba))


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0D1117")

    cmap = LinearSegmentedColormap.from_list(
        "mp_cmap", ["#0D1117", "#1a3a5c", "#0EA5E9", "#38BDF8", "#BAE6FD"])

    for ax, data, fmt, title in [
        (axes[0], cm,     "d",    "Absolute Counts"),
        (axes[1], cm_pct, ".1f",  "Row-normalized (%)"),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right",
                           fontsize=12, color="white")
        ax.set_yticklabels(class_names, fontsize=12, color="white")
        ax.set_xlabel("Predicted", fontsize=13, color="#94A3B8", labelpad=10)
        ax.set_ylabel("True", fontsize=13, color="#94A3B8", labelpad=10)
        ax.set_title(title, fontsize=14, color="white", pad=14)
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

        # Cell annotations
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val     = data[i, j]
                txt     = f"{val:{fmt}}"
                color   = "white" if val < thresh else "#0D1117"
                weight  = "bold" if i == j else "normal"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=11, color=color, fontweight=weight)

    # Overall accuracy annotation
    acc = (y_true == y_pred).mean()
    fig.suptitle(
        f"MicroPlastiNet M2b — Polymer Spectral Classifier\n"
        f"Confusion Matrix  |  Overall Accuracy: {acc:.2%}",
        fontsize=15, color="white", y=1.01, fontweight="bold"
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Confusion matrix → {save_path}")


# ── Plot: ROC Curves ──────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_proba, class_names, save_path):
    y_bin    = label_binarize(y_true, classes=list(range(len(class_names))))
    colors   = ["#0EA5E9", "#38BDF8", "#F97316", "#A78BFA", "#34D399", "#F87171"]

    fig, ax  = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    auc_scores = {}
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls}  (AUC = {roc_auc:.3f})")

    # Macro-average
    fpr_grid  = np.linspace(0, 1, 200)
    tpr_macro = np.zeros_like(fpr_grid)
    for i in range(len(class_names)):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        tpr_macro += np.interp(fpr_grid, fpr_i, tpr_i)
    tpr_macro /= len(class_names)
    macro_auc = auc(fpr_grid, tpr_macro)
    ax.plot(fpr_grid, tpr_macro, "w--", lw=2.5,
            label=f"Macro avg  (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#4B5563", lw=1, linestyle=":")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13, color="#94A3B8")
    ax.set_ylabel("True Positive Rate",  fontsize=13, color="#94A3B8")
    ax.set_title("ROC Curves — One-vs-Rest per Polymer Class",
                 fontsize=14, color="white", pad=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    legend = ax.legend(loc="lower right", fontsize=11,
                        facecolor="#161B22", edgecolor="#30363D",
                        labelcolor="white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] ROC curves → {save_path}")
    return auc_scores, macro_auc


# ── Plot: Per-class bar chart ─────────────────────────────────────────────────

def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    x    = np.arange(len(class_names))
    w    = 0.26
    cols = {"Precision": "#0EA5E9", "Recall": "#34D399", "F1": "#F97316"}

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    for idx, (label, vals, col) in enumerate(
        zip(cols.keys(), [prec, rec, f1], cols.values())
    ):
        bars = ax.bar(x + idx * w - w, vals, w, label=label, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12, color="white")
    ax.set_ylabel("Score", fontsize=13, color="#94A3B8")
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-class Precision / Recall / F1",
                 fontsize=14, color="white", pad=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    legend = ax.legend(fontsize=11, facecolor="#161B22",
                        edgecolor="#30363D", labelcolor="white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Per-class metrics → {save_path}")

    return {
        cls: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for cls, p, r, f, s in zip(class_names, prec, rec, f1, support)
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(arch: str = "cnn", seed: int = 42, save_preds: bool = False):
    print(f"[INFO] Evaluating {arch.upper()} model...")

    # Load data (test split only needed)
    _, _, test_loader, meta = get_dataloaders(seed=seed, augment_train=False)

    # Load model
    clf    = load_model(arch=arch)
    device = clf.device

    # Run inference
    y_true, y_pred, y_proba = get_predictions(clf, test_loader, device)
    class_names = meta["class_names"]

    # Text report
    report = classification_report(y_true, y_pred,
                                    target_names=class_names,
                                    digits=4)
    print("\n" + report)

    # Overall accuracy
    acc = (y_true == y_pred).mean()
    print(f"Overall Test Accuracy: {acc:.4%}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm_path  = os.path.join(ASSETS_DIR, "m2b_confusion.png")
    roc_path = os.path.join(ASSETS_DIR, "m2b_roc_curves.png")
    bar_path = os.path.join(ASSETS_DIR, "m2b_per_class_metrics.png")

    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    auc_scores, macro_auc = plot_roc_curves(y_true, y_proba, class_names, roc_path)
    per_class = plot_per_class_metrics(y_true, y_pred, class_names, bar_path)

    # ── JSON report ───────────────────────────────────────────────────────────
    eval_report = {
        "arch":          arch,
        "seed":          seed,
        "test_accuracy": float(acc),
        "macro_auc":     float(macro_auc),
        "auc_per_class": {k: float(v) for k, v in auc_scores.items()},
        "per_class":     per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names":   class_names,
        "n_test":        len(y_true),
        "artifacts": {
            "confusion_matrix": cm_path,
            "roc_curves":       roc_path,
            "per_class_chart":  bar_path,
        }
    }

    report_path = os.path.join(PROC_DIR, "m2b_eval_report.json")
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"[INFO] Eval report → {report_path}")

    # ── Optional: save predictions CSV ────────────────────────────────────────
    if save_preds:
        import csv
        preds_path = os.path.join(PROC_DIR, "m2b_test_predictions.csv")
        with open(preds_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["sample_idx", "true_label", "pred_label", "correct"] + \
                     [f"prob_{c}" for c in class_names]
            writer.writerow(header)
            for i, (yt, yp, ypr) in enumerate(zip(y_true, y_pred, y_proba)):
                writer.writerow([
                    i,
                    class_names[yt],
                    class_names[yp],
                    int(yt == yp)
                ] + [f"{p:.6f}" for p in ypr])
        print(f"[INFO] Predictions CSV → {preds_path}")

    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate M2b Spectral Classifier")
    parser.add_argument("--arch",       type=str,              default="cnn")
    parser.add_argument("--seed",       type=int,              default=42)
    parser.add_argument("--save_preds", action="store_true")
    args = parser.parse_args()

    report = evaluate(arch=args.arch, seed=args.seed, save_preds=args.save_preds)
    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY:  {report['test_accuracy']:.4%}")
    print(f"  MACRO AUC:      {report['macro_auc']:.4f}")
    print(f"{'='*50}")
