# M3 — Graph Neural Network for Microplastic Source Attribution

**MicroPlastiNet · Module 3 of 6**

> *"Where is the pollution coming from?"* — M3 answers this question by inverting a GNN trained on hydrological flow graphs to attribute observed microplastic concentrations at sampling stations to upstream candidate sources.

---

## Overview

Module 3 builds a **directed hydrological flow graph** spanning the Ogeechee, Savannah, and Altamaha river systems in coastal Georgia and trains two Graph Neural Network architectures to:

1. **Predict** microplastic concentrations (pieces/m³) at unmeasured nodes given upstream conditions
2. **Invert** those predictions via gradient-based attribution to rank upstream *sources* (factories, urban runoff, agricultural runoff) by their causal contribution to observed concentration spikes

---

## Geographic Grounding

The graph uses **real-world coordinates** from coastal Georgia's major river systems:

| River | Length in Study Area | Key Features |
|---|---|---|
| **Savannah River** | Augusta → Port of Savannah | Major industrial corridor; port, paper mills |
| **Ogeechee River** | Sandersville → Ossabaw Sound | Agricultural runoff; Statesboro urban area |
| **Altamaha River** | Douglas → Wolf Island | Ocmulgee/Oconee confluences; timber industry |

Node coordinates are sourced from USGS StreamStats / NHD Plus hydrology and grounded against known industrial, urban, and agricultural sites in the region.

Data topology follows **HydroSHEDS** drainage network conventions (Lehner et al., 2008):
- Directed edges follow downstream flow (higher elevation → lower elevation)
- Edge features: flow rate (m³/s), distance (km), wind correlation (ERA5 prevailing SW→NE coastal Georgia pattern)

---

## Graph Architecture

```
200 nodes total:
  50 × Sampling Stations     (measurement sites along all three rivers)
 100 × Candidate Sources     (30 factories + 35 urban runoff + 35 agricultural)
  50 × River Junctions       (confluences, routing nodes)

Node features [9-dim]:
  [norm_lat, norm_lon, norm_elevation, norm_pop_density, one_hot_type×5]

Edge features [3-dim]:
  [flow_rate_norm, distance_norm, wind_correlation]
```

---

## Data Simulation

**`synthetic_concentrations.py`** generates ~22,000 records following NOAA NCEI Marine Microplastics Database statistical patterns:

- **Distribution:** Log-normal (μ_ln = 1.5, σ_ln = 1.2) — calibrated to observed freshwater microplastic concentrations of 0.4–7,000 pieces/m³ (Koelmans et al., 2019)
- **Seasonal modulation:** Spring (day 60–180) and late-summer storm runoff peaks
- **Storm events:** 5% probability of 3–10× concentration spikes
- **Causal signal:** Concentrations at sampling stations are a *flow-weighted, distance-decayed function of upstream source emissions* — this is the signal the GNN learns to invert

**Temporal split (respects arrow of time):**
| Split | Years | Records | Purpose |
|---|---|---|---|
| Train | 2015–2020 | ~16,500 | Historical learning |
| Val   | 2021–2022 | ~3,300  | Hyperparameter tuning |
| Test  | 2023      | ~2,200  | Final evaluation |

---

## Model Architectures

### 1. GraphSAGE — Concentration Regression

```python
GraphSAGERegressor(
    in_channels=9,
    hidden_channels=128,
    num_layers=3,        # 3 message-passing rounds = 3-hop neighbourhood
    dropout=0.3,
)
```

**Reference:** Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS 2017. https://arxiv.org/abs/1706.02216

GraphSAGE uses *neighbourhood sampling + mean aggregation* to produce node embeddings. Its inductive design means it generalises to new nodes added as the sensor network expands — important for real-world deployment.

**Training:**
- Loss: MSE on log-concentration (log-normal targets)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- LR schedule: ReduceLROnPlateau (factor=0.5, patience=15)
- Early stopping: patience=40 epochs

---

### 2. GAT — Graph Attention Network (Interpretable)

```python
GATRegressor(
    in_channels=9,
    hidden_channels=64,
    heads=8,             # Layer 1: 8 attention heads
    dropout=0.3,
)
# Layer 2: 4 heads; Layer 3: 1 head
```

**Reference:** Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks.* ICLR 2018. https://arxiv.org/abs/1710.10903

The key advantage of GAT for source attribution: **attention weights α_ij directly encode how much node i's features influence node j's prediction.** Higher attention from source i to station j means i contributes more to j's concentration — analogous to edge weights in a causal connectivity graph.

---

### 3. Classical Baseline — Graph Centrality + Ridge Regression

Features: in-degree, out-degree, PageRank, betweenness centrality, closeness centrality (all computed on the directed flow graph) + raw node features → Ridge regression on log-concentration.

**Purpose:** Quantifies the value added by GNN message-passing over traditional graph-mining features.

---

## Attribution Method: Integrated Gradients

The **key research novelty** of M3 is the inversion of the trained GNN to identify pollution sources.

### Algorithm

Given an observed concentration spike at sampling station *s*, we compute:

$$\text{Attr}(i) = (x_i - \bar{x}_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F(s)}{\partial x_i}\bigg|_{x = \bar{x} + \frac{k}{m}(x - \bar{x})}$$

where:
- $x_i$ = features of source node *i*
- $\bar{x}_i$ = baseline (zero features = "no pollution")
- $F(s)$ = GNN's predicted concentration at station *s*
- $m$ = number of interpolation steps (default 50)

This satisfies the **Completeness axiom**: $\sum_i \text{Attr}(i) = F(x) - F(\bar{x})$, so attributions sum to the full prediction gap.

**Reference:** Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* ICML 2017. https://arxiv.org/abs/1703.01365

---

## Methodological Foundations

The attribution pipeline in M3 sits at the intersection of three established lines of work:

1. **Graph Neural Networks** for spatial reasoning over irregular topologies (GraphSAGE, GAT).
2. **Information-theoretic causality** on networks (transfer entropy, Granger causality), which formalizes "how much does node A's state influence node B?"
3. **Axiomatic attribution** for deep networks (Integrated Gradients), which provides gradient-based answers to the same causal-influence question.

M3 unifies these threads: a GNN learns the hydrological dynamics, then Integrated Gradients inverts the trained model to assign upstream sources causal credit for downstream concentration spikes — a gradient-theoretic analogue of effective connectivity analysis.

---

## File Structure

```
m3_graph_gnn/
├── graph_builder.py           — 200-node hydrological flow graph (real GA coords)
├── synthetic_concentrations.py — 22k NOAA-pattern microplastic records
├── model.py                   — GraphSAGE, GAT, ClassicalBaseline
├── train.py                   — Temporal-split training pipeline
├── attribution.py             — Integrated Gradients + GAT attention attribution
├── infer.py                   — Public API: predict_concentration(), attribute_source()
├── evaluate.py                — R² metrics, viz, attribution accuracy
├── README.md                  — This file
├── requirements.txt           — Dependencies
└── checkpoints/
    ├── graphsage_best.pt
    ├── gat_best.pt
    ├── classical_baseline.pkl
    └── training_results.json
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Build the graph
python graph_builder.py

# 2. Generate synthetic concentrations
python synthetic_concentrations.py

# 3. Train all models
python train.py

# 4. Run evaluation + visualizations
python evaluate.py

# 5. Use the inference API
python infer.py
```

---

## API Reference

### `infer.predict_concentration(node_id, time, return_confidence=False)`

```python
from infer import predict_concentration

# Point estimate
conc = predict_concentration(node_id=5, time="2023-07-15")
# → 4.82 pieces/m³

# With confidence interval
result = predict_concentration(node_id=5, time="2023-07-15", return_confidence=True)
# → {"mean": 4.82, "lower_95": 1.23, "upper_95": 18.9, ...}
```

### `infer.attribute_source(node_id, time, top_k=5)`

```python
from infer import attribute_source

result = attribute_source(node_id=5, time="2023-07-15", top_k=5)
# → {
#     "station": {"id": 5, "label": "station_005", "river": "savannah"},
#     "predicted_concentration_pieces_per_m3": 4.82,
#     "sources": [
#       {"rank": 1, "node_id": 52, "node_type": "factory", "attribution_probability": 0.38},
#       {"rank": 2, "node_id": 71, "node_type": "urban_runoff", "attribution_probability": 0.24},
#       ...
#     ],
#     "method": "integrated_gradients"
#   }
```

---

## Key References

1. **HydroSHEDS:** Lehner, B., Verdin, K., & Jarvis, A. (2008). New global hydrography derived from spaceborne elevation data. *Geophysical Research Letters, 35*(10). https://www.hydrosheds.org/

2. **NOAA NCEI Microplastics:** NOAA National Centers for Environmental Information. *Marine Microplastics Database.* https://www.ncei.noaa.gov/products/microplastics

3. **GraphSAGE:** Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017.* https://arxiv.org/abs/1706.02216

4. **GAT:** Veličković, P. et al. (2018). Graph attention networks. *ICLR 2018.* https://arxiv.org/abs/1710.10903

5. **PyTorch Geometric:** Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds.* https://arxiv.org/abs/1903.02428

6. **Integrated Gradients:** Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML 2017.* https://arxiv.org/abs/1703.01365

7. **Microplastic distributions:** Koelmans, A. A. et al. (2019). Microplastics in freshwaters and drinking water: Critical review and assessment of data quality. *Water Research, 155,* 410–422. https://doi.org/10.1016/j.watres.2019.02.054

8. **ERA5 wind data:** Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society, 146*(730), 1999–2049. https://www.ecmwf.int/

---

## Limitations and Failure Modes

- **Synthetic data:** All concentration records are simulated; real-world deployment requires integration with NOAA NCEI records and real-time sensor feeds from M1.
- **Static graph:** The current graph does not update topology in response to seasonal flooding, dam operations, or new construction. Dynamic graph variants (DCRNN, EvolveGCN) are natural extensions.
- **Attribution uncertainty:** IG attributions have no built-in confidence intervals. Monte Carlo dropout or deep ensembles should be added for production use.
- **Graph connectivity gaps:** Sources with no flow path to any station receive zero attribution — a correct but potentially misleading result for regulators who expect all sources to appear.
- **Resolution:** At 200 nodes, the graph is a coarse representation of the ~300 km coastline. Full HydroSHEDS integration would increase to ~10,000+ nodes.

---

*Module 3 of MicroPlastiNet — Multi-modal Microplastic Detection and Source Attribution Pipeline*
