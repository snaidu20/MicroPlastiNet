"""
evaluate.py — Comprehensive Evaluation for M3 Graph GNN
========================================================
Produces:
  1. R² comparison table (GraphSAGE vs GAT vs Classical Baseline)
  2. Attribution accuracy vs ground truth source emissions
  3. Interactive graph visualization → assets/m3_graph.html
  4. Summary results plot → assets/m3_results.png
  5. Sample attribution example (printed + saved)

Usage:
    python evaluate.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from model import GATRegressor, GraphSAGERegressor, build_node_regression_targets
from attribution import SourceAttributor

DATA_DIR = Path("/home/user/workspace/MicroPlastiNet/data/processed/m3")
CKPT_DIR = Path("/home/user/workspace/MicroPlastiNet/src/m3_graph_gnn/checkpoints")
ASSETS_DIR = Path("/home/user/workspace/MicroPlastiNet/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load everything
# ──────────────────────────────────────────────────────────────────────────────

def load_all():
    import pandas as pd

    data = torch.load(DATA_DIR / "flow_graph.pt", weights_only=False)
    data = data.to(DEVICE)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    with open(DATA_DIR / "source_emissions_ground_truth.json") as f:
        ground_truth = {int(k): float(v) for k, v in json.load(f).items()}

    with open(CKPT_DIR / "training_results.json") as f:
        train_results = json.load(f)

    return data, train_df, val_df, test_df, ground_truth, train_results


# ──────────────────────────────────────────────────────────────────────────────
# 2. Load trained models
# ──────────────────────────────────────────────────────────────────────────────

def load_models():
    sage = GraphSAGERegressor(in_channels=9, hidden_channels=128, num_layers=3)
    sage.load_state_dict(torch.load(CKPT_DIR / "graphsage_best.pt", map_location=DEVICE, weights_only=True))
    sage = sage.to(DEVICE).eval()

    gat = GATRegressor(in_channels=9, hidden_channels=64, heads=8)
    gat.load_state_dict(torch.load(CKPT_DIR / "gat_best.pt", map_location=DEVICE, weights_only=True))
    gat = gat.to(DEVICE).eval()

    return sage, gat


# ──────────────────────────────────────────────────────────────────────────────
# 3. Interactive graph visualization (PyVis)
# ──────────────────────────────────────────────────────────────────────────────

def build_graph_visualization(data, node_meta_list, G_nx=None):
    """Build an interactive HTML graph using PyVis."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("PyVis not available — skipping HTML visualization")
        return None

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#0f172a",
        directed=True,
        notebook=False,
    )
    net.set_options("""
    var options = {
        "nodes": {
            "borderWidth": 2,
            "shadow": {"enabled": true, "color": "rgba(0,0,0,0.5)"}
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
            "smooth": {"type": "dynamic"},
            "color": {"inherit": false, "opacity": 0.5}
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -4000,
                "centralGravity": 0.3,
                "springLength": 80,
                "damping": 0.5
            }
        }
    }
    """)

    # Color scheme
    type_colors = {
        "station":             "#0284c7",   # sky blue
        "factory":             "#dc2626",   # red
        "urban_runoff":        "#ea580c",   # orange
        "agricultural_runoff": "#16a34a",   # green
        "junction":            "#94a3b8",   # slate
    }
    type_sizes = {
        "station": 18,
        "factory": 16,
        "urban_runoff": 14,
        "agricultural_runoff": 12,
        "junction": 8,
    }

    # Add nodes
    for nid in range(data.num_nodes):
        ntype = data.node_types[nid]
        label = data.node_labels[nid]
        lat = data.node_lats[nid]
        lon = data.node_lons[nid]
        color = type_colors.get(ntype, "#ffffff")
        size = type_sizes.get(ntype, 10)

        title = (
            f"<b>{label}</b><br>"
            f"Type: {ntype}<br>"
            f"Lat: {lat:.4f}, Lon: {lon:.4f}<br>"
            f"Node ID: {nid}"
        )

        net.add_node(
            int(nid),
            label=label if ntype != "junction" else "",
            title=title,
            color=color,
            size=float(size),
        )

    # Add edges (subsample for readability — show all edges ≤500)
    ei = data.edge_index.cpu().numpy()
    ea = data.edge_attr.cpu().numpy()
    n_edges = ei.shape[1]

    for i in range(n_edges):
        u, v = int(ei[0, i]), int(ei[1, i])
        flow_rate = ea[i, 0]
        dist = ea[i, 1]
        # Color edges by flow rate
        r = int(255 * (1 - flow_rate))
        b = int(255 * flow_rate)
        color_hex = f"#{r:02x}00{b:02x}"
        title = f"Flow: {flow_rate:.2f} | Dist: {dist:.2f}"
        net.add_edge(u, v, color=color_hex, title=title, width=float(max(0.5, flow_rate * 2)))

    # Add legend using a separate approach
    legend_html = """
    <div style="position:absolute;top:10px;right:10px;background:#ffffff;border:1px solid #e2e8f0;padding:12px;border-radius:8px;color:#0f172a;font-family:Inter,Arial,sans-serif;font-size:13px;z-index:999;box-shadow:0 1px 3px rgba(15,23,42,0.08);">
        <b>Legend</b><br><br>
        <span style="color:#0284c7">&#9679;</span> Sampling Station (50)<br>
        <span style="color:#dc2626">&#9679;</span> Factory Source (30)<br>
        <span style="color:#ea580c">&#9679;</span> Urban Runoff (35)<br>
        <span style="color:#16a34a">&#9679;</span> Agricultural Runoff (35)<br>
        <span style="color:#94a3b8">&#9679;</span> River Junction (50)<br><br>
        <i style="color:#64748b">Coastal Georgia Rivers<br>Ogeechee · Savannah · Altamaha</i>
    </div>
    """

    out_path = ASSETS_DIR / "m3_graph.html"
    # Generate HTML
    net.generate_html()
    html_content = net.html

    # Inject legend
    html_content = html_content.replace("</body>", legend_html + "\n</body>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Saved interactive graph → {out_path}")
    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Summary results plot
# ──────────────────────────────────────────────────────────────────────────────

def build_results_plot(train_results, data, gat_model, sage_model, attribution_example):
    """Create a 4-panel summary figure."""
    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    fig.suptitle(
        "MicroPlastiNet — M3 Graph GNN Results\nCoastal Georgia Hydrological Source Attribution",
        fontsize=16, color="white", fontweight="bold", y=0.98
    )

    axes_color = "#1c1f26"
    text_color = "white"
    grid_color = "#2a2d36"

    # ── Panel 1: Training loss curves ────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1, facecolor=axes_color)
    sage_hist = train_results["graphsage"]["history"]
    gat_hist = train_results["gat"]["history"]

    epochs_sage = range(1, len(sage_hist["train_loss"]) + 1)
    epochs_gat = range(1, len(gat_hist["train_loss"]) + 1)

    ax1.plot(epochs_sage, sage_hist["val_loss"], color="#00d4ff", linewidth=2, label="GraphSAGE Val")
    ax1.plot(epochs_sage, sage_hist["train_loss"], color="#00d4ff", linewidth=1, alpha=0.4, linestyle="--", label="GraphSAGE Train")
    ax1.plot(epochs_gat, gat_hist["val_loss"], color="#ff9500", linewidth=2, label="GAT Val")
    ax1.plot(epochs_gat, gat_hist["train_loss"], color="#ff9500", linewidth=1, alpha=0.4, linestyle="--", label="GAT Train")

    ax1.set_xlabel("Epoch", color=text_color, fontsize=10)
    ax1.set_ylabel("MSE Loss", color=text_color, fontsize=10)
    ax1.set_title("Training Loss Curves", color=text_color, fontsize=12)
    ax1.legend(fontsize=8, facecolor="#2a2d36", labelcolor=text_color)
    ax1.tick_params(colors=text_color)
    ax1.set_facecolor(axes_color)
    for spine in ax1.spines.values():
        spine.set_edgecolor(grid_color)
    ax1.grid(alpha=0.2, color=grid_color)

    # ── Panel 2: R² comparison bar chart ─────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2, facecolor=axes_color)
    models = ["GraphSAGE", "GAT", "Classical\nBaseline"]
    test_r2 = [
        train_results["graphsage"]["test"]["r2"],
        train_results["gat"]["test"]["r2"],
        train_results["classical_baseline"]["test_r2"],
    ]
    colors = ["#00d4ff", "#ff9500", "#888888"]
    bars = ax2.bar(models, test_r2, color=colors, alpha=0.85, edgecolor="white", linewidth=1)

    for bar, r2 in zip(bars, test_r2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"R²={r2:.3f}",
            ha="center", va="bottom", color=text_color, fontsize=11, fontweight="bold"
        )

    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Test R²", color=text_color, fontsize=10)
    ax2.set_title("Model Performance Comparison\n(Test Set — Temporal Split 2023)", color=text_color, fontsize=12)
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_edgecolor(grid_color)
    ax2.grid(axis="y", alpha=0.2, color=grid_color)
    ax2.set_facecolor(axes_color)

    # ── Panel 3: R² training curves ──────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3, facecolor=axes_color)
    ax3.plot(epochs_sage, sage_hist["val_r2"], color="#00d4ff", linewidth=2, label="GraphSAGE Val R²")
    ax3.plot(epochs_gat, gat_hist["val_r2"], color="#ff9500", linewidth=2, label="GAT Val R²")
    ax3.axhline(
        train_results["classical_baseline"]["val_r2"],
        color="#888888", linewidth=1.5, linestyle=":", label="Classical Baseline"
    )
    ax3.set_xlabel("Epoch", color=text_color, fontsize=10)
    ax3.set_ylabel("R²", color=text_color, fontsize=10)
    ax3.set_title("R² Convergence", color=text_color, fontsize=12)
    ax3.legend(fontsize=8, facecolor="#2a2d36", labelcolor=text_color)
    ax3.tick_params(colors=text_color)
    ax3.set_facecolor(axes_color)
    for spine in ax3.spines.values():
        spine.set_edgecolor(grid_color)
    ax3.grid(alpha=0.2, color=grid_color)

    # ── Panel 4: Attribution bar chart ───────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4, facecolor=axes_color)
    if attribution_example and "sources" in attribution_example:
        sources = attribution_example["sources"]
        labels = [f"Node {s['node_id']}\n({s.get('node_type','?')[:8]})" for s in sources]
        probs = [s["attribution_probability"] for s in sources]
        type_color_map = {
            "factory": "#ff4444",
            "urban_runoff": "#ff9500",
            "agricultural_runoff": "#7cfc00",
        }
        bar_colors = [type_color_map.get(s.get("node_type", ""), "#888") for s in sources]
        bars4 = ax4.barh(labels[::-1], probs[::-1], color=bar_colors[::-1], alpha=0.85,
                         edgecolor="white", linewidth=0.8)

        for bar, prob in zip(bars4, probs[::-1]):
            ax4.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", color=text_color, fontsize=9
            )

        station_label = attribution_example["station"]["label"]
        ax4.set_xlabel("Attribution Probability", color=text_color, fontsize=10)
        ax4.set_title(
            f"Source Attribution\n{station_label}",
            color=text_color, fontsize=12
        )
        ax4.tick_params(colors=text_color)
        ax4.set_facecolor(axes_color)
        for spine in ax4.spines.values():
            spine.set_edgecolor(grid_color)
        ax4.grid(axis="x", alpha=0.2, color=grid_color)

    # ── Panel 5: Node-type distribution ──────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5, facecolor=axes_color)
    type_counts = {}
    for ntype in data.node_types:
        type_counts[ntype] = type_counts.get(ntype, 0) + 1

    labels5 = list(type_counts.keys())
    values5 = list(type_counts.values())
    colors5 = ["#00d4ff", "#ff4444", "#ff9500", "#7cfc00", "#888888"]

    wedges, texts, autotexts = ax5.pie(
        values5, labels=None, autopct="%1.0f%%",
        colors=colors5[:len(labels5)], startangle=90,
        pctdistance=0.7, textprops={"color": "white", "fontsize": 9}
    )
    ax5.legend(
        wedges, [f"{l} ({v})" for l, v in zip(labels5, values5)],
        loc="lower center", bbox_to_anchor=(0.5, -0.15),
        fontsize=8, facecolor="#2a2d36", labelcolor=text_color, ncol=2
    )
    ax5.set_title("Graph Node Composition\n(200 nodes total)", color=text_color, fontsize=12)

    # ── Panel 6: Map-style scatter of node positions ──────────────────────────
    ax6 = fig.add_subplot(2, 3, 6, facecolor="#0a0e1a")
    type_plot_colors = {
        "station": ("#00d4ff", 60, "Sampling Station"),
        "factory": ("#ff4444", 50, "Factory Source"),
        "urban_runoff": ("#ff9500", 40, "Urban Runoff"),
        "agricultural_runoff": ("#7cfc00", 30, "Agricultural Runoff"),
        "junction": ("#555555", 15, "River Junction"),
    }

    for ntype, (col, sz, lbl) in type_plot_colors.items():
        lons = [data.node_lons[i] for i in range(200) if data.node_types[i] == ntype]
        lats = [data.node_lats[i] for i in range(200) if data.node_types[i] == ntype]
        ax6.scatter(lons, lats, c=col, s=sz, label=lbl, alpha=0.75, edgecolors="none", zorder=3)

    # Draw edges
    ei = data.edge_index.cpu().numpy()
    for i in range(0, min(ei.shape[1], 400), 1):  # plot up to 400 edges
        u, v = int(ei[0, i]), int(ei[1, i])
        ax6.plot(
            [data.node_lons[u], data.node_lons[v]],
            [data.node_lats[u], data.node_lats[v]],
            color="#2a4a7a", linewidth=0.4, alpha=0.4, zorder=1
        )

    ax6.set_xlabel("Longitude", color=text_color, fontsize=10)
    ax6.set_ylabel("Latitude", color=text_color, fontsize=10)
    ax6.set_title("Coastal Georgia Hydrological Graph\nOgeechee · Savannah · Altamaha", color=text_color, fontsize=12)
    ax6.legend(fontsize=7, facecolor="#1a1a2e", labelcolor=text_color, loc="upper right")
    ax6.tick_params(colors=text_color)
    for spine in ax6.spines.values():
        spine.set_edgecolor(grid_color)
    ax6.set_facecolor("#0a0e1a")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = ASSETS_DIR / "m3_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved results plot → {out_path}")
    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Attribution accuracy evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_attribution(gat_model, data, ground_truth):
    """Evaluate attribution accuracy on synthetic ground truth."""
    print("\nEvaluating attribution accuracy...")
    attributor = SourceAttributor(gat_model, data)

    acc_results = attributor.attribution_accuracy(
        ground_truth_emissions=ground_truth,
        station_sample=20,
        top_k=5,
        method="integrated_gradients",
    )

    print(f"  Mean Spearman R: {acc_results['mean_spearman_r']:.4f}")
    print(f"  Top-1 Accuracy:  {acc_results['top1_accuracy']:.4f}")
    print(f"  Stations eval'd: {acc_results['n_stations_evaluated']}")
    return acc_results


# ──────────────────────────────────────────────────────────────────────────────
# 6. Sample attribution example
# ──────────────────────────────────────────────────────────────────────────────

def get_attribution_example(gat_model, data, node_meta_csv_path=None):
    """Get a sample attribution result for display."""
    import csv
    import json as _json

    # Load node metadata
    node_meta = {}
    if node_meta_csv_path and Path(node_meta_csv_path).exists():
        with open(node_meta_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = int(row["node_id"])
                node_meta[nid] = row

    attributor = SourceAttributor(gat_model, data)

    # Use station 5 (typically mid-Savannah area)
    station_id = data.station_ids[5]
    ranking = attributor.attribute(station_id, method="integrated_gradients", top_k=5)

    sources_list = []
    for rank, (src_id, prob) in enumerate(ranking.items(), start=1):
        meta = node_meta.get(src_id, {})
        sources_list.append({
            "rank": rank,
            "node_id": src_id,
            "node_type": meta.get("node_type", "unknown"),
            "label": meta.get("label", str(src_id)),
            "lat": float(meta.get("lat", 0)),
            "lon": float(meta.get("lon", 0)),
            "attribution_probability": prob,
        })

    station_meta = node_meta.get(station_id, {})
    example = {
        "station": {
            "id": station_id,
            "label": station_meta.get("label", str(station_id)),
            "lat": float(station_meta.get("lat", 0)),
            "lon": float(station_meta.get("lon", 0)),
            "river": station_meta.get("river", "?"),
        },
        "timestamp": "2023-07-15",
        "method": "integrated_gradients",
        "sources": sources_list,
    }

    out_path = CKPT_DIR / "sample_attribution.json"
    with open(out_path, "w") as f:
        _json.dump(example, f, indent=2)
    print(f"Saved sample attribution → {out_path}")
    return example


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MicroPlastiNet M3 — Evaluation Pipeline")
    print("=" * 70)

    data, train_df, val_df, test_df, ground_truth, train_results = load_all()
    sage_model, gat_model = load_models()

    # Print results table
    print("\n" + "="*70)
    print("CONCENTRATION PREDICTION (Test Set — Temporal Split 2023)")
    print("="*70)
    print(f"{'Model':<25} {'R²':>8} {'MSE':>10} {'MAE':>10}")
    print("-"*55)
    for model_name, key in [("GraphSAGE", "graphsage"), ("GAT", "gat")]:
        t = train_results[key]["test"]
        print(f"{model_name:<25} {t['r2']:>8.4f} {t['mse']:>10.4f} {t['mae']:>10.4f}")
    cb = train_results["classical_baseline"]
    print(f"{'Classical Baseline':<25} {cb['test_r2']:>8.4f} {cb['test_mse']:>10.4f} {cb['test_mae']:>10.4f}")
    print("-"*55)

    gnn_r2 = max(train_results["graphsage"]["test"]["r2"], train_results["gat"]["test"]["r2"])
    cb_r2 = cb["test_r2"]
    if cb_r2 > 0:
        gain = (gnn_r2 - cb_r2) / cb_r2 * 100
        print(f"Best GNN vs Classical:  +{gain:.1f}% relative R² improvement")

    # Attribution accuracy
    acc = evaluate_attribution(gat_model, data, ground_truth)

    # Sample attribution
    example = get_attribution_example(
        gat_model, data,
        node_meta_csv_path=str(DATA_DIR / "node_metadata.csv")
    )

    print("\nSample Attribution Result:")
    print(f"  Station: {example['station']['label']} (ID {example['station']['id']})")
    print(f"  River:   {example['station']['river']}")
    print(f"  Top-5 Source Attribution (Integrated Gradients):")
    for s in example["sources"]:
        print(f"    [{s['rank']}] {s['label']} ({s['node_type']}): {s['attribution_probability']:.4f}")

    # Visualization
    print("\nBuilding visualizations...")
    build_graph_visualization(data, [])
    build_results_plot(train_results, data, gat_model, sage_model, example)

    # Save comprehensive eval report
    eval_report = {
        "concentration_prediction": {
            "graphsage": train_results["graphsage"]["test"],
            "gat": train_results["gat"]["test"],
            "classical_baseline": cb,
        },
        "attribution_accuracy": acc,
        "sample_attribution": example,
        "dataset_stats": {
            "train_records": len(train_df),
            "val_records": len(val_df),
            "test_records": len(test_df),
            "n_stations": len(data.station_ids),
            "n_sources": len(data.source_ids),
            "n_junctions": len(data.junction_ids),
            "n_edges": data.edge_index.shape[1],
        }
    }

    report_path = ASSETS_DIR / "m3_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nSaved evaluation report → {report_path}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"  Graph HTML:    {ASSETS_DIR / 'm3_graph.html'}")
    print(f"  Results plot:  {ASSETS_DIR / 'm3_results.png'}")
    print(f"  Eval report:   {report_path}")


if __name__ == "__main__":
    main()
