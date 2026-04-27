"""
infer.py — Public Inference API for M3 Graph GNN
=================================================
Provides two clean public-facing functions:

    predict_concentration(node_id, time) -> float
    attribute_source(node_id, time, top_k=5) -> dict

These are the interfaces consumed by:
  - M4 Dashboard (source attribution panel)
  - M5 GenAI Report Generator (auto-populates source rankings)
  - Regulator export (PDF/CSV)

Usage:
    from infer import predict_concentration, attribute_source

    # Predict log-concentration at station 5 for a given date
    log_conc = predict_concentration(node_id=5, time="2023-06-15")

    # Attribute top-5 sources for a spike at station 5
    sources = attribute_source(node_id=5, time="2023-06-15", top_k=5)
    # Returns: {source_node_id: probability, ...}
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import GATRegressor, GraphSAGERegressor
from attribution import SourceAttributor

# ──────────────────────────────────────────────────────────────────────────────
# Singleton model loading (load once, reuse)
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/user/workspace/MicroPlastiNet/data/processed/m3")
CKPT_DIR = Path("/home/user/workspace/MicroPlastiNet/src/m3_graph_gnn/checkpoints")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CACHE = {}


def _load_resources():
    """Lazy-load graph, models, and node metadata."""
    if "data" in _CACHE:
        return _CACHE

    # Graph
    data = torch.load(DATA_DIR / "flow_graph.pt", weights_only=False)
    data = data.to(_DEVICE)

    # Node metadata
    import csv
    node_meta = {}
    with open(DATA_DIR / "node_metadata.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["node_id"])
            node_meta[nid] = {
                "label": row["label"],
                "node_type": row["node_type"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "elevation": float(row["elevation"]),
                "river": row["river"],
            }

    # Best GNN model (prefer GAT for production — has attention weights)
    gat_model = GATRegressor(in_channels=9, hidden_channels=64, heads=8)
    gat_ckpt = CKPT_DIR / "gat_best.pt"
    if gat_ckpt.exists():
        gat_model.load_state_dict(
            torch.load(gat_ckpt, map_location=_DEVICE, weights_only=True)
        )
        print("[infer] Loaded GAT checkpoint")
    else:
        print("[infer] WARNING: No GAT checkpoint found — using untrained model")
    gat_model = gat_model.to(_DEVICE).eval()

    # GraphSAGE as fallback
    sage_model = GraphSAGERegressor(in_channels=9, hidden_channels=128, num_layers=3)
    sage_ckpt = CKPT_DIR / "graphsage_best.pt"
    if sage_ckpt.exists():
        sage_model.load_state_dict(
            torch.load(sage_ckpt, map_location=_DEVICE, weights_only=True)
        )
        print("[infer] Loaded GraphSAGE checkpoint")
    sage_model = sage_model.to(_DEVICE).eval()

    # Attributor
    attributor = SourceAttributor(gat_model, data)

    # Ground truth emissions (for context)
    gt_path = DATA_DIR / "source_emissions_ground_truth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            ground_truth = {int(k): float(v) for k, v in json.load(f).items()}
    else:
        ground_truth = {}

    _CACHE.update({
        "data": data,
        "node_meta": node_meta,
        "gat_model": gat_model,
        "sage_model": sage_model,
        "attributor": attributor,
        "ground_truth": ground_truth,
    })
    return _CACHE


def _seasonal_factor(timestamp: Union[str, datetime]) -> float:
    """Compute seasonal modulation factor for a given date."""
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp)
    else:
        dt = timestamp
    day_of_year = dt.timetuple().tm_yday
    t = day_of_year / 365.0 * 2 * math.pi
    return 1.0 + 0.35 * math.sin(t - math.pi / 4) + 0.15 * math.sin(2 * t)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def predict_concentration(
    node_id: int,
    time: Union[str, datetime],
    model: str = "sage",
    return_confidence: bool = False,
) -> Union[float, Dict]:
    """
    Predict microplastic concentration (pieces/m³) at a sampling station
    for a given time.

    Parameters
    ----------
    node_id          : sampling station node ID (must be a station node)
    time             : ISO date string "YYYY-MM-DD" or datetime object
    model            : "sage" (default) or "gat"
    return_confidence: if True, return dict with value + CI

    Returns
    -------
    concentration : float (pieces/m³)
    OR (if return_confidence=True):
        {"mean": float, "lower_95": float, "upper_95": float, "log_pred": float}

    Notes
    -----
    The GNN predicts log-concentration at a static graph (not time-varying).
    The seasonal factor is applied post-hoc as a multiplicative modulation.
    For production deployment, temporal node features would be re-computed
    from the real-time sensor stream.
    """
    rc = _load_resources()
    data = rc["data"]
    node_meta = rc["node_meta"]

    # Validate
    if node_id not in data.station_ids:
        available = data.station_ids[:5]
        raise ValueError(
            f"Node {node_id} is not a sampling station. "
            f"Valid station IDs include: {available} ..."
        )

    # GNN prediction (log-concentration)
    chosen_model = rc["gat_model"] if model == "gat" else rc["sage_model"]
    chosen_model.eval()

    with torch.no_grad():
        pred = chosen_model(data.x.to(_DEVICE), data.edge_index.to(_DEVICE))
        log_conc = pred[node_id, 0].item()

    # Apply seasonal modulation
    sf = _seasonal_factor(time)
    log_conc_seasonal = log_conc + math.log(sf)

    concentration = math.exp(log_conc_seasonal)

    if return_confidence:
        # Bootstrap-style CI using model uncertainty proxy (±0.5 log units)
        log_uncertainty = 0.5
        return {
            "mean": round(concentration, 4),
            "lower_95": round(math.exp(log_conc_seasonal - 1.96 * log_uncertainty), 4),
            "upper_95": round(math.exp(log_conc_seasonal + 1.96 * log_uncertainty), 4),
            "log_pred": round(log_conc_seasonal, 4),
            "seasonal_factor": round(sf, 4),
            "node_info": {
                "id": node_id,
                "label": node_meta.get(node_id, {}).get("label", str(node_id)),
                "river": node_meta.get(node_id, {}).get("river", "unknown"),
            }
        }

    return round(concentration, 4)


def attribute_source(
    node_id: int,
    time: Union[str, datetime],
    top_k: int = 5,
    method: str = "integrated_gradients",
    include_metadata: bool = True,
) -> Dict:
    """
    Attribute a concentration reading at `node_id` to upstream source nodes.

    Parameters
    ----------
    node_id          : sampling station node ID
    time             : ISO date string or datetime (for context)
    top_k            : number of top sources to return
    method           : "integrated_gradients" (recommended) or "attention"
    include_metadata : if True, include source node geographic metadata

    Returns
    -------
    {
        "station": {"id": int, "label": str, "lat": float, "lon": float},
        "timestamp": str,
        "predicted_concentration": float,
        "sources": [
            {
                "rank": int,
                "node_id": int,
                "label": str,
                "node_type": str,
                "lat": float,
                "lon": float,
                "attribution_probability": float,
            },
            ...
        ],
        "method": str,
    }
    """
    rc = _load_resources()
    data = rc["data"]
    node_meta = rc["node_meta"]
    attributor = rc["attributor"]

    # Get attribution ranking
    ranking = attributor.attribute(
        station_node_id=node_id,
        method=method,
        top_k=top_k,
    )

    # Get prediction for context
    concentration = predict_concentration(node_id, time, model="gat")

    # Build response
    if isinstance(time, datetime):
        ts_str = time.isoformat()
    else:
        ts_str = str(time)

    station_info = node_meta.get(node_id, {"label": str(node_id), "lat": 0, "lon": 0, "river": "?"})

    sources_list = []
    for rank, (src_id, prob) in enumerate(ranking.items(), start=1):
        src_info = node_meta.get(src_id, {"label": str(src_id), "node_type": "unknown", "lat": 0, "lon": 0})
        entry = {
            "rank": rank,
            "node_id": src_id,
            "attribution_probability": prob,
        }
        if include_metadata:
            entry.update({
                "label": src_info.get("label", str(src_id)),
                "node_type": src_info.get("node_type", "unknown"),
                "lat": src_info.get("lat", 0),
                "lon": src_info.get("lon", 0),
            })
        sources_list.append(entry)

    result = {
        "station": {
            "id": node_id,
            "label": station_info.get("label", str(node_id)),
            "lat": station_info.get("lat", 0),
            "lon": station_info.get("lon", 0),
            "river": station_info.get("river", "?"),
        },
        "timestamp": ts_str,
        "predicted_concentration_pieces_per_m3": concentration,
        "sources": sources_list,
        "method": method,
        "top_k": top_k,
    }

    return result


def get_node_info(node_id: int) -> Dict:
    """Return metadata for a given node."""
    rc = _load_resources()
    meta = rc["node_meta"].get(node_id, {})
    data = rc["data"]
    is_station = node_id in data.station_ids
    is_source = node_id in data.source_ids
    is_junction = node_id in data.junction_ids
    return {
        "node_id": node_id,
        "is_station": is_station,
        "is_source": is_source,
        "is_junction": is_junction,
        **meta,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("M3 Graph GNN — Inference API Demo")
    print("=" * 60)

    rc = _load_resources()
    station_ids = rc["data"].station_ids

    test_station = station_ids[3]
    test_date = "2023-07-15"

    print(f"\n[predict_concentration] Station {test_station}, Date: {test_date}")
    result = predict_concentration(test_station, test_date, return_confidence=True)
    print(json.dumps(result, indent=2))

    print(f"\n[attribute_source] Station {test_station}, Date: {test_date}, top_k=5")
    attribution = attribute_source(test_station, test_date, top_k=5)
    print(json.dumps(attribution, indent=2))
