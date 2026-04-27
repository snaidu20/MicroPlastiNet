"""
model.py — GNN Architectures for Microplastic Source Attribution
================================================================
Implements three models for node-level concentration regression:

1. GraphSAGE — inductive, scalable neighbourhood aggregation
   Hamilton, W. et al. (2017). Inductive Representation Learning on Large
   Graphs. NeurIPS 2017. https://arxiv.org/abs/1706.02216

2. GAT — Graph Attention Network with interpretable attention weights
   Veličković, P. et al. (2018). Graph Attention Networks. ICLR 2018.
   https://arxiv.org/abs/1710.10903

3. Classical baseline — graph centrality features + linear regression
   (Compares traditional graph mining with modern deep GNNs.)

Architecture note: attention weights in the GAT head serve as proxy
source-contribution scores, analogous to transfer-entropy edge weights
used in network connectivity analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    SAGEConv,
    GATConv,
    global_mean_pool,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. GraphSAGE Concentration Regressor
# ──────────────────────────────────────────────────────────────────────────────

class GraphSAGERegressor(nn.Module):
    """
    GraphSAGE for node-level concentration prediction.

    Architecture:
        Input → SAGEConv(128) → BN → ReLU → Dropout
              → SAGEConv(64)  → BN → ReLU → Dropout
              → SAGEConv(32)  → BN → ReLU
              → Linear(1)     → scalar log-concentration

    The model is trained to predict log(concentration) to handle the
    log-normal distribution of microplastic counts.
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [32]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.head = nn.Linear(32, out_channels)

    def forward(self, x, edge_index, edge_attr=None, return_embeddings=False):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        embeddings = x.clone()
        out = self.head(x)

        if return_embeddings:
            return out, embeddings
        return out   # [N, 1] log-concentration predictions


# ──────────────────────────────────────────────────────────────────────────────
# 2. GAT Concentration Regressor
# ──────────────────────────────────────────────────────────────────────────────

class GATRegressor(nn.Module):
    """
    Graph Attention Network for node-level concentration prediction.

    The multi-head attention weights α_ij serve as interpretability signals:
    higher α_ij between source i and station j → source i contributes more
    to the concentration at j.

    This mirrors the transfer entropy / effective connectivity framework
    used in network science: both methods ask
    "how much does node A's state influence node B?"

    Architecture:
        Input → GAT(heads=8, 64-per-head) → ELU → Dropout
              → GAT(heads=4, 32-per-head) → ELU → Dropout
              → GAT(heads=1, 32)          → ELU
              → Linear(1)
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 64,
        out_channels: int = 1,
        heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads,
            dropout=dropout, concat=True
        )
        self.conv2 = GATConv(
            hidden_channels * heads, 32, heads=4,
            dropout=dropout, concat=True
        )
        self.conv3 = GATConv(
            32 * 4, 32, heads=1,
            dropout=dropout, concat=False
        )

        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(32 * 4)
        self.bn3 = nn.BatchNorm1d(32)

        self.head = nn.Linear(32, out_channels)

        # Store last attention weights for attribution
        self._last_attention = None

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        # Layer 1
        x, (edge_idx1, alpha1) = self.conv1(
            x, edge_index, return_attention_weights=True
        )
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x, (edge_idx2, alpha2) = self.conv2(
            x, edge_index, return_attention_weights=True
        )
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x, (edge_idx3, alpha3) = self.conv3(
            x, edge_index, return_attention_weights=True
        )
        x = self.bn3(x)
        x = F.elu(x)

        # Store attention weights for attribution (use last layer)
        self._last_attention = {
            "edge_index": edge_idx3.detach(),
            "alpha": alpha3.detach(),
        }

        embeddings = x.clone()
        out = self.head(x)

        if return_attention:
            return out, (edge_idx3, alpha3), embeddings
        return out   # [N, 1]

    def get_attention_weights(self):
        """Return the last forward pass attention weights."""
        return self._last_attention


# ──────────────────────────────────────────────────────────────────────────────
# 3. Classical Baseline — Graph Centrality + Linear Regression
# ──────────────────────────────────────────────────────────────────────────────

class ClassicalBaseline:
    """
    Classical graph-mining baseline:
    1. Compute graph centrality metrics (in-degree, betweenness, PageRank,
       closeness) for each node.
    2. Concatenate with raw node features.
    3. Fit Ridge regression to predict log-concentration.

    This baseline lets us quantify the value added by GNN message-passing
    over traditional centrality-based features — a direct comparison that
    validates the GNN approach (analogous to comparing transfer entropy
    baselines with GNN-based brain connectivity analysis).
    """

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        self.centrality_features = None
        self.is_fitted = False

    def compute_centrality_features(self, G_nx, num_nodes: int) -> np.ndarray:
        """Compute centrality vectors for all nodes."""
        import networkx as nx

        in_deg = dict(G_nx.in_degree())
        out_deg = dict(G_nx.out_degree())

        # PageRank
        try:
            pr = nx.pagerank(G_nx, alpha=0.85, max_iter=200)
        except Exception:
            pr = {n: 1.0 / num_nodes for n in G_nx.nodes()}

        # Betweenness (sample-based for speed)
        try:
            bc = nx.betweenness_centrality(G_nx, k=min(50, num_nodes), normalized=True)
        except Exception:
            bc = {n: 0.0 for n in G_nx.nodes()}

        # Closeness on undirected version
        try:
            cl = nx.closeness_centrality(G_nx.to_undirected())
        except Exception:
            cl = {n: 0.0 for n in G_nx.nodes()}

        feats = np.zeros((num_nodes, 5))
        for n in range(num_nodes):
            feats[n, 0] = in_deg.get(n, 0)
            feats[n, 1] = out_deg.get(n, 0)
            feats[n, 2] = pr.get(n, 0)
            feats[n, 3] = bc.get(n, 0)
            feats[n, 4] = cl.get(n, 0)

        self.centrality_features = feats
        return feats

    def fit(
        self,
        x: np.ndarray,              # [N, node_feat_dim] raw node features
        centrality: np.ndarray,     # [N, 5] centrality features
        y: np.ndarray,              # [N] log-concentration targets
        mask: np.ndarray,           # boolean mask — which nodes have labels
    ):
        combined = np.concatenate([x[mask], centrality[mask]], axis=1)
        self.model.fit(combined, y[mask])
        self.is_fitted = True

    def predict(
        self,
        x: np.ndarray,
        centrality: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        combined = np.concatenate([x[mask], centrality[mask]], axis=1)
        return self.model.predict(combined)

    def score(
        self,
        x: np.ndarray,
        centrality: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        preds = self.predict(x, centrality, mask)
        return float(np.corrcoef(preds, y[mask])[0, 1] ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: build graph-level dataset for node regression
# ──────────────────────────────────────────────────────────────────────────────

def build_node_regression_targets(df_split, data, station_ids):
    """
    For a given time-period split DataFrame, compute per-station mean
    log-concentration and return as a tensor aligned with node indices.

    Returns
    -------
    y : torch.Tensor [N, 1]  — log-concentration for station nodes, 0 elsewhere
    mask : torch.BoolTensor [N] — True for station nodes that have data
    """
    N = data.num_nodes
    y = torch.zeros(N, 1, dtype=torch.float)
    mask = torch.zeros(N, dtype=torch.bool)

    for s_id in station_ids:
        rows = df_split[df_split["station_id"] == s_id]
        if len(rows) > 0:
            mean_log_conc = rows["log_concentration"].mean()
            y[s_id, 0] = mean_log_conc
            mask[s_id] = True

    return y, mask


if __name__ == "__main__":
    # Quick smoke test
    import torch
    x = torch.randn(200, 9)
    edge_index = torch.randint(0, 200, (2, 500))

    sage = GraphSAGERegressor(in_channels=9)
    out_sage = sage(x, edge_index)
    print(f"GraphSAGE output shape: {out_sage.shape}")

    gat = GATRegressor(in_channels=9)
    out_gat = gat(x, edge_index)
    print(f"GAT output shape: {out_gat.shape}")
    print("Model smoke test passed.")
