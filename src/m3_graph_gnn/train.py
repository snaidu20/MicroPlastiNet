"""
train.py — Training Pipeline for M3 Graph GNN
==============================================
Trains GraphSAGE and GAT on node-level log-concentration regression
using a TEMPORAL SPLIT:
  - Train: 2015–2020 (roughly 75% of records)
  - Val:   2021–2022 (~15%)
  - Test:  2023      (~10%)

This respects the arrow of time — the model is trained on historical data
and evaluated on future observations, the only meaningful evaluation for
environmental monitoring systems.

Saves:
  - checkpoints/graphsage_best.pt
  - checkpoints/gat_best.pt
  - training_history.json
  - classical_baseline.pkl

Usage:
    python train.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

# Local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import GraphSAGERegressor, GATRegressor, ClassicalBaseline, build_node_regression_targets

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {DEVICE}")

DATA_DIR = Path("/home/user/workspace/MicroPlastiNet/data/processed/m3")
CKPT_DIR = Path("/home/user/workspace/MicroPlastiNet/src/m3_graph_gnn/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading graph and concentration data...")
    data = torch.load(DATA_DIR / "flow_graph.pt", weights_only=False)
    data = data.to(DEVICE)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    station_ids = data.station_ids

    # Per-station mean log-concentration for each split
    y_train, mask_train = build_node_regression_targets(train_df, data, station_ids)
    y_val, mask_val = build_node_regression_targets(val_df, data, station_ids)
    y_test, mask_test = build_node_regression_targets(test_df, data, station_ids)

    y_train = y_train.to(DEVICE)
    y_val = y_val.to(DEVICE)
    y_test = y_test.to(DEVICE)
    mask_train = mask_train.to(DEVICE)
    mask_val = mask_val.to(DEVICE)
    mask_test = mask_test.to(DEVICE)

    print(f"  Train stations with data: {mask_train.sum().item()}")
    print(f"  Val   stations with data: {mask_val.sum().item()}")
    print(f"  Test  stations with data: {mask_test.sum().item()}")

    return data, y_train, y_val, y_test, mask_train, mask_val, mask_test


# ──────────────────────────────────────────────────────────────────────────────
# Training loop (shared)
# ──────────────────────────────────────────────────────────────────────────────

def train_gnn(
    model_name: str,
    model: nn.Module,
    data,
    y_train, y_val,
    mask_train, mask_val,
    n_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 40,
) -> dict:
    """Generic training loop for any GNN regressor."""

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )
    criterion = nn.MSELoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_r2": [], "val_r2": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training {model_name} for {n_epochs} epochs")
    print(f"{'='*60}")

    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(pred[mask_train], y_train[mask_train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(data.x, data.edge_index, data.edge_attr)
            val_loss = criterion(val_pred[mask_val], y_val[mask_val])

        # R² scores
        train_r2 = r2_score(
            y_train[mask_train].cpu().numpy(),
            pred[mask_train].detach().cpu().numpy()
        )
        val_r2 = r2_score(
            y_val[mask_val].cpu().numpy(),
            val_pred[mask_val].cpu().numpy()
        )

        scheduler.step(val_loss)

        history["train_loss"].append(float(loss))
        history["val_loss"].append(float(val_loss))
        history["train_r2"].append(float(train_r2))
        history["val_r2"].append(float(val_r2))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), CKPT_DIR / f"{model_name}_best.pt")
        else:
            patience_counter += 1

        if epoch % 50 == 0 or patience_counter >= patience:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:4d} | "
                f"train_loss={loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_R²={train_r2:.4f} | val_R²={val_r2:.4f} | "
                f"t={elapsed:.1f}s"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Test evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(model_name, model, data, y_test, mask_test) -> dict:
    """Load best checkpoint and evaluate on test set."""
    ckpt = CKPT_DIR / f"{model_name}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr)

    y_np = y_test[mask_test].cpu().numpy()
    p_np = pred[mask_test].cpu().numpy()

    test_r2 = r2_score(y_np, p_np)
    test_mse = float(np.mean((y_np - p_np) ** 2))
    test_mae = float(np.mean(np.abs(y_np - p_np)))

    print(f"\n[{model_name}] TEST RESULTS:")
    print(f"  R²  = {test_r2:.4f}")
    print(f"  MSE = {test_mse:.4f}")
    print(f"  MAE = {test_mae:.4f}")

    return {"r2": test_r2, "mse": test_mse, "mae": test_mae}


# ──────────────────────────────────────────────────────────────────────────────
# Classical baseline
# ──────────────────────────────────────────────────────────────────────────────

def train_classical_baseline(data, y_train, y_val, y_test, mask_train, mask_val, mask_test):
    """Compute centrality features and train Ridge regression baseline."""
    import pickle
    import networkx as nx

    print("\nTraining classical baseline (centrality + Ridge regression)...")

    # Reconstruct NetworkX graph
    ei = data.edge_index.cpu().numpy()
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(range(200))
    for i in range(ei.shape[1]):
        G_nx.add_edge(int(ei[0, i]), int(ei[1, i]))

    baseline = ClassicalBaseline(alpha=1.0)
    centrality = baseline.compute_centrality_features(G_nx, num_nodes=200)

    x_np = data.x.cpu().numpy()
    y_train_np = y_train.cpu().numpy().ravel()
    y_val_np = y_val.cpu().numpy().ravel()
    y_test_np = y_test.cpu().numpy().ravel()
    mask_train_np = mask_train.cpu().numpy()
    mask_val_np = mask_val.cpu().numpy()
    mask_test_np = mask_test.cpu().numpy()

    # Fit on train
    baseline.fit(x_np, centrality, y_train_np, mask_train_np)

    # Evaluate
    val_preds = baseline.predict(x_np, centrality, mask_val_np)
    val_r2 = r2_score(y_val_np[mask_val_np], val_preds)

    test_preds = baseline.predict(x_np, centrality, mask_test_np)
    test_r2 = r2_score(y_test_np[mask_test_np], test_preds)
    test_mse = float(np.mean((y_test_np[mask_test_np] - test_preds) ** 2))
    test_mae = float(np.mean(np.abs(y_test_np[mask_test_np] - test_preds)))

    print(f"\n[Classical Baseline] RESULTS:")
    print(f"  Val R²  = {val_r2:.4f}")
    print(f"  Test R² = {test_r2:.4f}")
    print(f"  Test MSE = {test_mse:.4f}")
    print(f"  Test MAE = {test_mae:.4f}")

    # Save baseline model
    with open(CKPT_DIR / "classical_baseline.pkl", "wb") as f:
        pickle.dump({"model": baseline, "centrality": centrality}, f)

    return {
        "val_r2": val_r2,
        "test_r2": test_r2,
        "test_mse": test_mse,
        "test_mae": test_mae,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Load
    data, y_train, y_val, y_test, mask_train, mask_val, mask_test = load_data()

    # ── GraphSAGE ──────────────────────────────────────────────────────────
    sage_model = GraphSAGERegressor(
        in_channels=9, hidden_channels=128, num_layers=3, dropout=0.3
    ).to(DEVICE)

    sage_history = train_gnn(
        "graphsage", sage_model, data,
        y_train, y_val, mask_train, mask_val,
        n_epochs=300, lr=1e-3, patience=40
    )
    sage_test = evaluate_on_test("graphsage", sage_model, data, y_test, mask_test)

    # ── GAT ───────────────────────────────────────────────────────────────
    gat_model = GATRegressor(
        in_channels=9, hidden_channels=64, heads=8, dropout=0.3
    ).to(DEVICE)

    gat_history = train_gnn(
        "gat", gat_model, data,
        y_train, y_val, mask_train, mask_val,
        n_epochs=300, lr=5e-4, patience=40
    )
    gat_test = evaluate_on_test("gat", gat_model, data, y_test, mask_test)

    # ── Classical Baseline ─────────────────────────────────────────────────
    classical_results = train_classical_baseline(
        data, y_train, y_val, y_test, mask_train, mask_val, mask_test
    )

    # ── Save combined results ──────────────────────────────────────────────
    results = {
        "graphsage": {
            "history": sage_history,
            "test": sage_test,
        },
        "gat": {
            "history": gat_history,
            "test": gat_test,
        },
        "classical_baseline": classical_results,
    }

    out_path = CKPT_DIR / "training_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved → {out_path}")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  GraphSAGE Test R²:       {sage_test['r2']:.4f}")
    print(f"  GAT       Test R²:       {gat_test['r2']:.4f}")
    print(f"  Classical Baseline R²:   {classical_results['test_r2']:.4f}")
    print(f"  GNN vs Classical gain:   {((sage_test['r2'] - classical_results['test_r2']) / max(classical_results['test_r2'], 1e-6) * 100):.1f}%")

    return results


if __name__ == "__main__":
    main()
