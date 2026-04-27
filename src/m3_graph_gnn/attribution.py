"""
attribution.py — Gradient-Based Source Attribution via Integrated Gradients
============================================================================
Given an observed concentration spike at a sampling station node, this module
ranks upstream SOURCE nodes by their causal contribution using two methods:

1. Integrated Gradients (Sundararajan et al., 2017) — the primary method.
   Computes the integral of gradients along a straight path from a baseline
   (zero graph features) to the observed graph. Each source node's feature
   gradient integral is its attribution score.

   Reference: Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic
   Attribution for Deep Networks. ICML 2017.
   https://arxiv.org/abs/1703.01365

2. GAT Attention-based attribution — secondary method.
   Uses the final-layer attention weights α_ij to trace source → station
   influence along flow paths.

METHODOLOGICAL ANALOGY:
   Integrated Gradients on a GNN is structurally analogous to computing
   effective connectivity in network science. Both ask: "given observed
   activity at node B, how much of it is causally attributable to node A?"
   Information-theoretic measures (transfer entropy, Granger causality)
   answer this in classical network analysis; here, we answer it via
   gradient attribution — same philosophical question, modern deep
   learning solution.

Usage:
    from attribution import SourceAttributor
    attr = SourceAttributor(gat_model, data)
    ranking = attr.attribute(station_node_id=5, top_k=5)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import GATRegressor, GraphSAGERegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Integrated Gradients Implementation
# ──────────────────────────────────────────────────────────────────────────────

def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,           # [N, F] node features
    edge_index: torch.Tensor,  # [2, E]
    edge_attr: torch.Tensor,   # [E, 3]
    target_node: int,          # which node's output to attribute
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Compute Integrated Gradients attribution for all nodes w.r.t.
    the output at `target_node`.

    Parameters
    ----------
    model       : trained GNN model
    x           : node feature matrix [N, F]
    edge_index  : edge connectivity [2, E]
    edge_attr   : edge features [E, 3]
    target_node : index of the node whose prediction we attribute
    baseline    : baseline input (default: zero tensor)
    n_steps     : number of interpolation steps (higher = more accurate)

    Returns
    -------
    attributions : torch.Tensor [N, F]
        Per-node, per-feature attribution scores.
        Sum across features gives per-node total attribution.
    """
    model.eval()

    if baseline is None:
        # Baseline = zero features (represents "no pollution signal")
        baseline = torch.zeros_like(x)

    # Interpolation path: baseline + α*(x - baseline) for α in [0,1]
    alphas = torch.linspace(0, 1, n_steps, device=x.device)
    grad_sum = torch.zeros_like(x)

    for alpha in alphas:
        x_interp = baseline + alpha * (x - baseline)
        x_interp = x_interp.detach().requires_grad_(True)

        # Forward pass
        pred = model(x_interp, edge_index, edge_attr)

        # Scalar: prediction at target node
        score = pred[target_node, 0]

        # Backward
        model.zero_grad()
        score.backward()

        if x_interp.grad is not None:
            grad_sum += x_interp.grad.detach()

    # IG formula: (x - baseline) * (1/n_steps) * Σ gradients
    attributions = (x - baseline) * (grad_sum / n_steps)
    return attributions.detach()   # [N, F]


# ──────────────────────────────────────────────────────────────────────────────
# Source Attributor Class
# ──────────────────────────────────────────────────────────────────────────────

class SourceAttributor:
    """
    Unified attribution interface supporting both Integrated Gradients (IG)
    on GraphSAGE/GAT and attention-weight attribution on GAT.

    Example
    -------
    >>> attr = SourceAttributor(model, data)
    >>> ranking = attr.attribute(station_node_id=5, method="integrated_gradients")
    >>> print(ranking)
    # {node_id: probability, ...} for top-k upstream sources
    """

    def __init__(
        self,
        model: nn.Module,
        data,                  # PyG Data from graph_builder
        source_ids: Optional[List[int]] = None,
    ):
        self.model = model.to(DEVICE)
        self.data = data
        self.source_ids = source_ids if source_ids is not None else data.source_ids
        self.station_ids = data.station_ids

        self.x = data.x.to(DEVICE)
        self.edge_index = data.edge_index.to(DEVICE)
        self.edge_attr = data.edge_attr.to(DEVICE)

    def attribute(
        self,
        station_node_id: int,
        method: str = "integrated_gradients",
        top_k: int = 5,
        n_steps: int = 50,
    ) -> Dict[int, float]:
        """
        Attribute a concentration prediction at `station_node_id` to
        upstream source nodes.

        Parameters
        ----------
        station_node_id : node ID of the sampling station
        method          : "integrated_gradients" or "attention"
        top_k           : return top-k sources
        n_steps         : IG interpolation steps (for IG method)

        Returns
        -------
        ranking : dict {source_node_id: probability}
            Probabilities sum to 1 over returned sources.
            Ordered by contribution (descending).
        """
        assert station_node_id in self.station_ids, \
            f"Node {station_node_id} is not a sampling station."

        if method == "integrated_gradients":
            return self._ig_attribution(station_node_id, top_k, n_steps)
        elif method == "attention":
            return self._attention_attribution(station_node_id, top_k)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'integrated_gradients' or 'attention'.")

    def _ig_attribution(
        self, station_node_id: int, top_k: int, n_steps: int
    ) -> Dict[int, float]:
        """Integrated Gradients attribution."""
        node_attr = integrated_gradients(
            model=self.model,
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            target_node=station_node_id,
            baseline=None,
            n_steps=n_steps,
        )   # [N, F]

        # Per-node total attribution: sum of absolute feature attributions
        node_scores = node_attr.abs().sum(dim=1).cpu().numpy()   # [N]

        # Filter to source nodes only
        source_scores = {
            src_id: float(node_scores[src_id])
            for src_id in self.source_ids
        }

        # Sort and take top-k
        sorted_sources = sorted(source_scores.items(), key=lambda x: -x[1])[:top_k]

        # Convert to probabilities
        total = sum(v for _, v in sorted_sources)
        if total < 1e-10:
            # Uniform fallback
            n = len(sorted_sources)
            return {k: 1.0 / n for k, _ in sorted_sources}

        ranking = {k: round(v / total, 6) for k, v in sorted_sources}
        return ranking

    def _attention_attribution(
        self, station_node_id: int, top_k: int
    ) -> Dict[int, float]:
        """
        GAT attention-weight attribution.
        Traces attention flow from sources to the target station along the
        directed flow graph.
        """
        if not hasattr(self.model, 'get_attention_weights'):
            raise ValueError("Model does not support attention attribution. Use GAT.")

        self.model.eval()
        with torch.no_grad():
            _ = self.model(self.x, self.edge_index, self.edge_attr)

        attn_data = self.model.get_attention_weights()
        if attn_data is None:
            raise ValueError("No attention weights available. Run a forward pass first.")

        edge_idx = attn_data["edge_index"].cpu().numpy()   # [2, E]
        alpha = attn_data["alpha"].cpu().numpy()            # [E, 1] or [E, heads]

        # Collapse heads by mean
        if alpha.ndim > 1:
            alpha = alpha.mean(axis=1)   # [E]

        # Build edge → attention dict
        edge_alpha = {}
        for i in range(edge_idx.shape[1]):
            u, v = int(edge_idx[0, i]), int(edge_idx[1, i])
            edge_alpha[(u, v)] = float(alpha[i])

        # For each source, compute attention path to station
        # Simple 1-hop + 2-hop aggregation
        source_scores = {}
        for src_id in self.source_ids:
            # Direct edge
            direct = edge_alpha.get((src_id, station_node_id), 0.0)
            # Through intermediate nodes (junctions/stations)
            indirect = 0.0
            for mid_id in self.data.junction_ids + self.data.station_ids:
                if mid_id == station_node_id:
                    continue
                a1 = edge_alpha.get((src_id, mid_id), 0.0)
                a2 = edge_alpha.get((mid_id, station_node_id), 0.0)
                indirect += a1 * a2

            source_scores[src_id] = direct + 0.5 * indirect

        sorted_sources = sorted(source_scores.items(), key=lambda x: -x[1])[:top_k]
        total = sum(v for _, v in sorted_sources)
        if total < 1e-10:
            n = len(sorted_sources)
            return {k: 1.0 / n for k, _ in sorted_sources}

        ranking = {k: round(v / total, 6) for k, v in sorted_sources}
        return ranking

    def attribute_batch(
        self,
        station_node_ids: List[int],
        method: str = "integrated_gradients",
        top_k: int = 5,
    ) -> Dict[int, Dict[int, float]]:
        """
        Run attribution for multiple stations.

        Returns
        -------
        {station_id: {source_id: probability, ...}, ...}
        """
        results = {}
        for s_id in station_node_ids:
            try:
                results[s_id] = self.attribute(s_id, method=method, top_k=top_k)
            except Exception as e:
                results[s_id] = {"error": str(e)}
        return results

    def attribution_accuracy(
        self,
        ground_truth_emissions: Dict[int, float],
        station_sample: int = 10,
        top_k: int = 5,
        method: str = "integrated_gradients",
    ) -> Dict[str, float]:
        """
        Evaluate attribution accuracy against ground-truth source emissions.

        Metric: for each sampled station, compute the rank correlation between
        IG-attributed scores and true emission rates for the top-k sources.

        Also computes "top-1 accuracy": whether the highest-attributed source
        is truly the highest emitter visible to that station.
        """
        from scipy.stats import spearmanr

        # True top emitters
        true_top = sorted(ground_truth_emissions.items(), key=lambda x: -x[1])
        true_rank = {src_id: rank for rank, (src_id, _) in enumerate(true_top)}

        sample_stations = self.station_ids[:station_sample]

        spearman_rs = []
        top1_hits = []

        for s_id in sample_stations:
            try:
                ranking = self.attribute(s_id, method=method, top_k=top_k)
            except Exception:
                continue

            # Get attributed scores in consistent order
            attributed_ids = list(ranking.keys())
            attr_scores = [ranking[k] for k in attributed_ids]
            true_scores = [ground_truth_emissions.get(k, 0.0) for k in attributed_ids]

            if len(attr_scores) >= 2 and sum(true_scores) > 0:
                rs, _ = spearmanr(attr_scores, true_scores)
                spearman_rs.append(float(rs) if not np.isnan(rs) else 0.0)

            # Top-1 accuracy
            if attributed_ids:
                top1_attributed = attributed_ids[0]
                true_top1 = true_top[0][0]
                top1_hits.append(1 if top1_attributed == true_top1 else 0)

        return {
            "mean_spearman_r": float(np.mean(spearman_rs)) if spearman_rs else 0.0,
            "top1_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
            "n_stations_evaluated": len(spearman_rs),
        }


if __name__ == "__main__":
    # Quick test
    print("Testing attribution module...")

    DATA_DIR = Path("/home/user/workspace/MicroPlastiNet/data/processed/m3")
    CKPT_DIR = Path("/home/user/workspace/MicroPlastiNet/src/m3_graph_gnn/checkpoints")

    data = torch.load(DATA_DIR / "flow_graph.pt", weights_only=False)

    # Load best GAT model
    gat_model = GATRegressor(in_channels=9, hidden_channels=64, heads=8)
    ckpt = CKPT_DIR / "gat_best.pt"
    if ckpt.exists():
        gat_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        print("Loaded GAT checkpoint")
    else:
        print("No GAT checkpoint found — using random weights for test")

    attr = SourceAttributor(gat_model, data)

    # Test on first station
    station_id = data.station_ids[0]
    print(f"\nAttributing station {station_id} using Integrated Gradients...")
    ranking = attr.attribute(station_node_id=station_id, method="integrated_gradients", top_k=5)
    print(f"Top-5 source attribution:")
    for src_id, prob in ranking.items():
        print(f"  Node {src_id}: {prob:.4f}")
