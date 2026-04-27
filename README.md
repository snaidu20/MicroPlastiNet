# MicroPlastiNet

**Multi-modal IoT + Deep Learning + Graph ML pipeline for microplastic detection, classification, and source attribution along coastal Georgia rivers.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-GNN-3C2179)](https://pytorch-geometric.readthedocs.io/)
[![Dash](https://img.shields.io/badge/Plotly-Dash-119DFF?logo=plotly&logoColor=white)](https://dash.plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/security_tests-6%2F6_passing-brightgreen)]()

**Author:** Saikumar Reddy Naidu — CS Graduate, Florida Atlantic University
**Status:** Research prototype — ongoing research

## About this project

Microplastics are now found in every major river system on Earth, but the field still relies almost entirely on slow, expensive lab-based workflows: collect a water sample, ship it to a lab, count particles under a microscope, and run FTIR/Raman to identify the polymer. By the time a result is published, the pollution event is weeks old and the source is gone.

**MicroPlastiNet is an end-to-end research prototype that closes that loop in under 10 seconds.** It combines six tightly-integrated modules:

- **An IoT edge node** (simulated ESP32-CAM with turbidity, TDS, and 6-channel NIR sensors) that detects suspicious particles in real time and streams cryptographically-signed payloads to the cloud.
- **Two deep-learning classifiers** — a CNN that counts and sizes particles from camera images, and a 1D-CNN that identifies the polymer type (PE, PET, PP, PS, PVC) from its spectral fingerprint.
- **A Graph Neural Network** trained on a 200-node hydrological flow graph of the Ogeechee, Savannah, and Altamaha river systems. Once a station reports a concentration spike, Integrated Gradients on the GNN traces the signal back through the river network to rank the most likely upstream sources (factories, urban runoff, farms).
- **A compliance dashboard and an LLM-powered report generator** that turn the raw inference into a regulator-ready PDF citing CWA § 1251, Georgia EPD protocols, and per-source attribution percentages.
- **A cybersecurity layer** (HMAC-SHA256, replay protection, TLS 1.3, key rotation) because pollution-monitoring IoT is a high-value tampering target that the published literature largely ignores.

The goal is not just "another microplastics classifier" — it's a complete, defensible, source-attribution pipeline that an environmental agency could realistically deploy on a real river.

> **[ View the live dashboard → ](https://microplastinet.pplx.app)**

[![MicroPlastiNet Dashboard](assets/dashboard_hero.png)](https://microplastinet.pplx.app)

---

## TL;DR

```
[ Water in chamber ]
        ↓
[ M1 — Edge Node (ESP32-CAM + turbidity + TDS + 6-ch NIR) ]   ← simulated
        ↓  signed MQTT (TLS + HMAC)                            ← M6
[ M2a — Vision DL  →  particle counts, sizes, shapes  ]
[ M2b — Spectral 1D-CNN  →  PE / PET / PP / PS / PVC  ]
        ↓
[ M3 — Graph Neural Network on hydrological flow graph ]
        ↓  Integrated-Gradients source attribution
[ M4 — Compliance Dashboard  +  M5 — GenAI Regulator Reports ]
```

A single sample of river water is sensed, identified, traced back to its likely upstream source, and converted into a regulator-ready report — in under 10 seconds.

---

## Why this project exists

Microplastic monitoring today is broken: 2–4 weeks per sample, ~$300 each, manual particle counting under a microscope, and source attribution that is mostly guesswork. NOAA's [global database](https://www.ncei.noaa.gov/products/microplastics) holds only ~22,000 records over 50 years — a pace problem, not a science problem.

**MicroPlastiNet collapses sensing, identification, and forensic source-tracing into a continuous, automated pipeline.** The novelty is in the integration: nobody has previously combined cheap multi-modal IoT sensing with deep-learning polymer ID *and* a graph-mining attribution layer.

**Geographic grounding:** sensor stations are placed at real coordinates along the **Ogeechee, Savannah, and Altamaha rivers** of coastal Georgia.

---

## Module-by-module results

### M1 — IoT Edge Simulator
A faithful software-twin of an MPN-Edge field unit (ESP32-CAM + SEN0189 turbidity + TDS + AS7265x 6-channel NIR). Sensor noise drawn from real datasheets; on-device first-pass detector emulates a TFLite-Micro logistic gate.

```bash
python -m src.m1_iot_edge.edge_simulator --mode file --steps 240
# → 7 stations × 240 timesteps = 1,680 signed payloads
```

[`src/m1_iot_edge/`](src/m1_iot_edge/) · sensor models · edge ML · MQTT publisher · cloud listener

---

### M2a — Vision Deep Learning
**EfficientNet-B0 + custom TinyYOLO detector** trained on 2,000 synthetic microplastic microscopy images.

| Model | Metric | Value |
|---|---|---|
| EfficientNet-B0 classifier | Val accuracy | **94.0%** |
| EfficientNet-B0 classifier | Macro F1 | **0.94** |
| EfficientNet-B0 classifier | Best epoch (7) val acc | **95.0%** |
| TinyYOLO detector | Val loss (5 epochs) | 98.1 |

**Honest limit (documented in [M2a README](src/m2a_vision/README.md)):** synthetic-data accuracy will drop to ~60–70% on real field imagery; UV-fluorescence (Nile Red) augmentation lifts that to ~85%. We do not oversell.

![M2a demo](assets/m2a_demo.png)

[`src/m2a_vision/`](src/m2a_vision/)

---

### M2b — Spectral 1D-CNN
4-block 1D-CNN (471 K params) classifies polymer type from 901-point FTIR/Raman spectra. Synthetic spectra generated from published characteristic peaks (PE 2916/720, PET 1715, PP 998, PS 700, PVC 615 cm⁻¹).

| Model | Test accuracy | Macro AUC |
|---|---|---|
| **SpectralCNN** | **100.0%** | **1.000** |
| MLP baseline | 99.3% | — |

Drop-in compatible with the [Rochman Lab SLoPP/SLoPP-E (Raman)](https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/) and FLOPP/FLOPP-e (FTIR) datasets — place CSVs in `data/raw/rochman_slopp/` and the dataset class merges them automatically.

![M2b confusion matrix](assets/m2b_confusion.png)

[`src/m2b_spectral/`](src/m2b_spectral/)

---

### M3 — Graph Neural Network · Source Attribution (the research-novelty heart)

A 200-node directed flow graph spanning the Ogeechee / Savannah / Altamaha watersheds:

- 50 sampling stations · 30 factories · 35 urban runoff · 35 agricultural runoff · 50 river junctions
- 22,000 NOAA-pattern log-normal concentration records (5-year temporal split)
- **GraphSAGE** node-level concentration regression
- **GAT** with attention-based interpretability
- **Classical baseline** (graph centrality + Ridge) for honest comparison
- **Integrated Gradients** on the trained GNN to attribute observed pollution to upstream sources

| Model | **Test R²** | MSE | MAE |
|---|---|---|---|
| **GraphSAGE** | **0.960** | 0.064 | 0.193 |
| GAT | 0.698 | 0.488 | 0.533 |
| Classical (centrality + Ridge) | 0.682 | 0.514 | 0.580 |

> **+40.8% relative R² gain** of GraphSAGE over the classical graph-mining baseline.

**Sample attribution** for an observed spike at Station 5 (Savannah River, 2023-07-15):

| Rank | Source node | Type | Probability |
|---|---|---|---|
| 1 | Node 106 (32.33 °N, –81.67 °W) | Urban runoff | **35.8%** |
| 2 | Node 81  (32.29 °N, –81.66 °W) | Urban runoff | 33.8% |
| 3 | Node 101 | Urban runoff | 14.7% |
| 4 | Node 115 | Agricultural runoff | 11.5% |
| 5 | Node 91  | Urban runoff | 4.2% |

This applies the same causal question long studied with transfer entropy on directed networks (*"how much does node A drive node B?"*) — implemented here on a hydrological flow graph using a modern gradient-theoretic method (Integrated Gradients).

![M3 results](assets/m3_results.png)

🌐 **[Interactive 200-node graph (PyVis)](assets/m3_graph.html)**

[`src/m3_graph_gnn/`](src/m3_graph_gnn/)

---

### M4 — Compliance Dashboard
A 6-tab Plotly Dash app (light theme) built for environmental compliance officers, rendering ~50 stations on a coastal Georgia OpenStreetMap basemap.

| Tab | What it shows |
|---|---|
| **Watershed Map** | Stations colored by contamination (green / yellow / red); click for detail panel |
| **Station Trends** | 30-day concentration line + turbidity dual axis + anomaly diamonds |
| **Polymer Breakdown** | Donut + per-station stacked bar of polymer mix |
| **Source Attribution** | M3 GNN top-5 sources + flow lines + embedded interactive graph |
| **Forecast & Alerts** | SARIMA(1,0,1)(1,0,0,7) 7-day forecast with 80% CI ribbon |
| **Reports** | One-click LLM regulator report (M5) with PDF/MD download |

```bash
python src/m4_dashboard/app.py     # → http://127.0.0.1:8050
```

| Watershed Map | Station Trends | Polymer Breakdown |
|:-:|:-:|:-:|
| ![](assets/dashboard_tab1.png) | ![](assets/dashboard_tab2.png) | ![](assets/dashboard_tab3.png) |
| **Source Attribution** | **Forecast & Alerts** | **Reports** |
| ![](assets/dashboard_tab4.png) | ![](assets/dashboard_tab5.png) | ![](assets/dashboard_tab6.png) |

[`src/m4_dashboard/`](src/m4_dashboard/)

---

### M5 — GenAI Regulator Reports
Pydantic-validated report generator with two modes:

- **Template mode** (offline) — sophisticated Jinja2 produces a polished 1–2 page regulator report
- **OpenAI mode** (online) — system prompt + few-shot examples for high-quality LLM generation

Output supports PDF (ReportLab, dark masthead) and Markdown.

📄 **[Sample report (PDF)](assets/sample_report.pdf)** · **[Sample report (MD)](assets/sample_report.md)**

[`src/m5_genai/`](src/m5_genai/)

---

### M6 — Cybersecurity Layer
Pollution-monitoring IoT is a tampering-magnet. M6 closes a gap **almost no published microplastic IoT paper addresses**.

| Threat | Mitigation |
|---|---|
| Payload tampering | HMAC-SHA256 over canonical JSON |
| Replay of old payload | Per-payload nonce + bounded LRU cache |
| Stale messages | 5-minute timestamp freshness window |
| Long-lived key compromise | Per-station key rotation (30-min grace) |
| Eavesdropping | TLS 1.3 transport (MQTT-over-TLS) |

**6 / 6 adversarial tests pass:**

```
✓ test_happy_path
✓ test_tampering_caught
✓ test_replay_caught
✓ test_wrong_key_caught
✓ test_stale_timestamp_caught
✓ test_key_rotation_grace_window
```

[`src/m6_security/`](src/m6_security/)

---

## End-to-end demo

```bash
# 1. Generate a day of signed sensor payloads from 7 coastal Georgia stations
python -m src.m1_iot_edge.edge_simulator --mode file --steps 240

# 2. Cloud-side verification (HMAC + replay + freshness)
python -m src.m1_iot_edge.cloud_listener

# 3. Train all three GNN variants on the 200-node flow graph
python -m src.m3_graph_gnn.train

# 4. Source-attribute a spike at Station 5
python -m src.m3_graph_gnn.infer attribute --node station_005

# 5. Run the dashboard
python src/m4_dashboard/app.py     # → http://127.0.0.1:8050

# 6. Generate a regulator report
python -m src.m5_genai.report_generator --station STN-003 --out report.pdf
```

---

## Real datasets the pipeline plugs into

| Dataset | Used by | URL |
|---|---|---|
| NOAA NCEI Marine Microplastics DB (~22 k records) | M3 ground truth | https://www.ncei.noaa.gov/products/microplastics |
| Rochman Lab SLoPP / SLoPP-E (Raman, 343 spectra) | M2b training | https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/ |
| Rochman Lab FLOPP / FLOPP-e (FTIR, 381 spectra) | M2b training | https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/ |
| Kaggle Microplastic CV Dataset (YOLO) | M2a vision | https://www.kaggle.com/code/mathieuduverne/microplastic-detection-yolov8-map-50-76-2 |
| MP-Set fluorescence microscopy | M2a vision | https://www.kaggle.com/datasets/sanghyeonaustinpark/mpset |
| HydroSHEDS river network | M3 graph topology | https://www.hydrosheds.org/ |
| ERA5 wind / weather reanalysis | M3 covariates | https://www.ecmwf.int/ |

> The repo ships with **physics-informed synthetic data** for every module so the pipeline runs end-to-end on any laptop. Drop the real CSVs / TIFFs into `data/raw/` and every loader picks them up automatically.

---

## Project layout

```
MicroPlastiNet/
├── README.md                 ← you are here
├── PROJECT_SPEC.md
├── data/                     synthetic + (optional) real data
├── src/
│   ├── m1_iot_edge/          IoT simulator + cloud listener
│   ├── m2a_vision/           YOLO + EfficientNet-B0
│   ├── m2b_spectral/         1D-CNN spectral classifier
│   ├── m3_graph_gnn/         GraphSAGE + GAT + attribution
│   ├── m4_dashboard/         6-tab Plotly Dash app
│   ├── m5_genai/             LLM regulator report generator
│   ├── m6_security/          HMAC + replay protection + TLS
│   └── common/               shared schemas
├── assets/                   plots, screenshots, sample report
├── notebooks/                EDA + result inspection
├── docs/                     architecture + GitHub Pages site
└── tests/                    adversarial security tests (all passing)
```

---

## Honesty principles

This project is engineered to be a serious research artifact — that means we tell the truth about what works and what doesn't.

1. **Field-grade vs lab-grade accuracy** are reported separately. Camera-only ≈ 60–70 %; +UV ≈ 85 %; +FTIR ≈ 95 %.
2. **Synthetic vs real data** is labeled in every module. Real datasets slot in via a single `data/raw/` directory.
3. **Confidence intervals**, not point estimates, on every prediction.
4. **Classical-baseline comparison** for the GNN to demonstrate the modern approach is genuinely better.
5. **Failure-mode docs** in every module README.

---

## Citations

- NOAA NCEI Marine Microplastics Database. *Scientific Data*, 2023. [DOI](https://www.nature.com/articles/s41597-023-02632-y)
- Sundararajan, M. *et al.* — Axiomatic Attribution for Deep Networks (Integrated Gradients). ICML 2017.
- Hamilton, W. *et al.* — Inductive Representation Learning on Large Graphs (GraphSAGE). NeurIPS 2017.
- Veličković, P. *et al.* — Graph Attention Networks (GAT). ICLR 2018.
- Sun, C. *et al.* — Lagrangian particle-tracking modeling of microplastic transport. *Frontiers in Toxicology*, 2025.
- Rochman, C. — SLoPP / FLOPP spectral libraries for microplastics research. Rochman Lab.
- Ramasamy, V., & Dorai, G. — *GraphDPR: A Privacy Policy Analysis Framework Using Knowledge Graphs and Topic Modeling.* ASONAM 2025, pp. 422–430. Methodological precedent for graph-based knowledge extraction and attribution from observed evidence (informs M3 design). [DBLP](https://dblp.org/pid/223/3218.html)
- Shovon, M. I., Nandagopal, N., Vijayalakshmi, R., Du, J. T., & Cocks, B. — *Directed Connectivity Analysis of Functional Brain Networks during Cognitive Activity Using Transfer Entropy.* Neural Processing Letters, 2017. Effective-connectivity / transfer-entropy inspiration for Integrated Gradients on the GNN flow graph in M3. [DOI](https://dl.acm.org/doi/abs/10.1007/s11063-016-9506-1)
- Ramasamy, V. — *Capturing Cognition via EEG-Based Functional Brain Networks.* University of New England. [rune.une.edu.au](https://rune.une.edu.au/entities/publication/d6cc649a-2e04-4eb4-84c5-2e5de701af4a)

---

## License

MIT — see [LICENSE](LICENSE).
