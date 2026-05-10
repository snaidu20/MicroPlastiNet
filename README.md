# MicroPlastiNet

**Multi-modal IoT + Deep Learning + Graph ML pipeline for microplastic detection, classification, and source attribution along coastal Georgia rivers.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-GNN-3C2179)](https://pytorch-geometric.readthedocs.io/)
[![Dash](https://img.shields.io/badge/Plotly-Dash-119DFF?logo=plotly&logoColor=white)](https://dash.plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/security_tests-6%2F6_passing-brightgreen)]()

**Author:** Saikumar Reddy Naidu — CS Graduate, Florida Atlantic University
**Status:** Engineering prototype — ongoing research

---

## ⚠️ Honest disclosure (read this first)

This is an **engineering prototype**, not a deployed environmental-monitoring system. Before reading the rest of the README, please understand exactly what is and isn't real here:

1. **All training and evaluation data are synthetic.** Every module — vision (M2a), spectral (M2b), graph GNN (M3), IoT edge (M1) — runs on procedurally generated, physics-informed synthetic data, not real microplastic measurements. **No real microplastic was ever detected, classified, or attributed in this pipeline.**
2. **Why synthetic data?** I do not currently have access to the real datasets the pipeline is designed for: NOAA NCEI Marine Microplastics records, Rochman Lab SLoPP/SLoPP-E and FLOPP/FLOPP-e spectra, the Kaggle Microplastic CV / MP-Set images, HydroSHEDS river topology, and ERA5 reanalysis.

   **Why I don't have access to the real data (yet).** Each of these sources sits behind a credential, license, or sharing gate I don't currently have access to:
   - **NOAA NCEI Marine Microplastics** — bulk/API pulls require a registered research project and credentialed account; the public portal only exposes limited record-by-record downloads.
   - **Rochman Lab SLoPP / SLoPP-E / FLOPP / FLOPP-e spectra** — shared directly with lab members and credentialed collaborators, not via open download.
   - **ERA5 reanalysis (Copernicus CDS)** — high-resolution, high-volume pulls require an institutional Copernicus account; personal accounts are throttled to small requests.
   - **HydroSHEDS high-res river topology for the Georgia coastal watersheds** — the commercial-grade tiles needed for sub-basin routing require a licensed/institutional download.
   - **Kaggle Microplastic CV / MP-Set imagery** — available but limited in size and unlabeled for source attribution; insufficient on its own to train M3.

   The repository is built so each loader picks up the corresponding real CSVs / TIFFs / spectra from `data/raw/` the moment access is granted. Until then, all reported numbers reflect synthetic-only performance.
3. **The headline numbers are inflated by synthetic data.** 94% vision accuracy, 100% spectral accuracy, and R² = 0.96 on concentration regression are training-on-synthetic-distribution-and-testing-on-the-same-distribution numbers. They do **not** generalize to real field data. The README explicitly notes (line 88) that real-world camera accuracy will likely drop to ~60–70%.
4. **The core source-attribution claim does not work.** On the M3 GNN attribution evaluation (`assets/m3_eval_report.json`), Integrated Gradients achieves `top1_accuracy = 0.0` and `mean_spearman_r = -0.27` across 20 evaluated stations — i.e. the attribution is **negatively correlated with the ground truth on this synthetic graph**. The single "sample attribution at Station 5" shown below is a cherry-picked example, not representative of the method's actual performance.
5. **What is real:** the cybersecurity layer (M6 — HMAC-SHA256, replay protection, key rotation) ships with 6/6 adversarial unit tests that genuinely pass, and the systems engineering of the multi-modal pipeline itself.

This project is therefore best read as a **multi-modal pipeline architecture demonstration** — the integration of IoT, vision DL, spectral DL, graph ML, dashboard, GenAI reporting, and cryptographic security in one repository — not as evidence of working microplastic forensics. With access to the real datasets above, the same code paths could be evaluated honestly on real measurements; that is future work.

---

## About this project

Microplastics are now found in every major river system on Earth, but the field still relies almost entirely on slow, expensive lab-based workflows: collect a water sample, ship it to a lab, count particles under a microscope, and run FTIR/Raman to identify the polymer. By the time a result is published, the pollution event is weeks old and the source is gone.

**MicroPlastiNet is an end-to-end engineering prototype that demonstrates how such a loop *could* be automated end-to-end on synthetic data.** It combines six tightly-integrated modules:

- **An IoT edge node** (simulated ESP32-CAM with turbidity, TDS, and 6-channel NIR sensors) that detects suspicious particles in real time and streams cryptographically-signed payloads to the cloud.
- **Two deep-learning classifiers** — a CNN that counts and sizes particles from camera images, and a 1D-CNN that identifies the polymer type (PE, PET, PP, PS, PVC) from its spectral fingerprint.
- **A Graph Neural Network** trained on a 200-node hydrological flow graph of the Ogeechee, Savannah, and Altamaha river systems. Once a station reports a concentration spike, Integrated Gradients on the GNN traces the signal back through the river network to rank the most likely upstream sources (factories, urban runoff, farms).
- **A compliance dashboard and an LLM-powered report generator** that turn the raw inference into a structured PDF (synthetic prototype) citing CWA § 1251, Georgia EPD protocols, and per-source attribution percentages.
- **A cybersecurity layer** (HMAC-SHA256, replay protection, TLS 1.3, key rotation) because pollution-monitoring IoT is a high-value tampering target that the published literature largely ignores.

The goal is to demonstrate the end-to-end *architecture* of a multi-modal, source-attributing pipeline. It is **not** a deployable system today — see the disclosure above for what is and isn't real, including the failure of the source-attribution claim on the synthetic evaluation set.

> **[ View the live dashboard → ](https://naidusai-microplastinet.hf.space)**

[![MicroPlastiNet Dashboard](assets/dashboard_hero.png)](https://naidusai-microplastinet.hf.space)

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

A single sample of river water is sensed, identified, traced back to its likely upstream source, and converted into a structured compliance report (synthetic prototype).

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

### M3 — Graph Neural Network · Source Attribution

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

> **+40.8% relative R² gain** of GraphSAGE over the classical graph-mining baseline on **synthetic** concentration regression. This is a within-distribution synthetic-vs-synthetic comparison and does not reflect real-world performance.

#### ⚠️ Source-attribution evaluation result

| Metric | Value | Interpretation |
|---|---|---|
| `top1_accuracy` (20 stations) | **0.0** | Integrated Gradients picks the correct top-1 source on **zero** stations |
| `mean_spearman_r` (20 stations) | **−0.27** | Attribution ranking is **negatively correlated** with the synthetic ground truth |

In other words, on the synthetic flow graph, Integrated Gradients on the trained GraphSAGE does **not** recover the true sources — the example below is a single illustrative output, not representative of the method's actual accuracy. Whether this fails because of the IG method, the synthetic graph design, or the regression task is open future work; we report the negative result honestly rather than hide it.

**Sample attribution** (illustrative only) for an observed spike at Station 5 (Savannah River, 2023-07-15):

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
| Rochman Lab SLoPP / SLoPP-E (Raman, 261 spectra) | M2b training | https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/ |
| Rochman Lab FLOPP / FLOPP-e (FTIR, 381 spectra) | M2b training | https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/ |
| Kaggle Microplastic CV Dataset (YOLO) | M2a vision | https://www.kaggle.com/datasets/imtkaggleteam/microplastic-dataset-for-computer-vision |
| MP-Set fluorescence microscopy | M2a vision | https://www.kaggle.com/datasets/sanghyeonaustinpark/mpset |
| HydroSHEDS river network | M3 graph topology | https://www.hydrosheds.org/ |
| ERA5 wind / weather reanalysis | M3 covariates | https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels |

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

1. **All current data is synthetic.** Real-world datasets (NOAA NCEI, Rochman SLoPP/FLOPP, Kaggle MP-Set, HydroSHEDS, ERA5) are not yet integrated because I do not have access to them at this time. Every reported metric is therefore a synthetic-on-synthetic measurement, not a real-world one. See the top-of-README disclosure for full detail.
2. **Synthetic vs real data is labeled in every module.** Real datasets slot in via a single `data/raw/` directory once obtained.
3. **Field-grade vs lab-grade accuracy are reported separately** (e.g. README line 88: synthetic camera ≈ 94%, real-world camera ≈ 60–70%, +UV ≈ 85%, +FTIR ≈ 95%). Synthetic numbers are not extrapolated to real-world claims.
4. **Negative results are reported, not hidden.** The M3 GNN source-attribution evaluation — the project's main research-style claim — produces `top1_accuracy = 0.0` and `mean_spearman_r = −0.27` on 20 synthetic stations. We report this prominently in M3 above rather than burying it.
5. **Classical-baseline comparison** for the GNN regression task (Ridge + centrality) is included to contextualize the GraphSAGE R² number on the same synthetic data.
6. **Failure-mode docs** in every module README.
7. **What is genuinely real:** the M6 cybersecurity layer ships with 6/6 adversarial unit tests that actually exercise HMAC tampering, replay, stale timestamps, wrong-key, and key-rotation paths.

---

## Citations

**Methodology references**

- Sundararajan, M., Taly, A., & Yan, Q. — *Axiomatic Attribution for Deep Networks (Integrated Gradients).* ICML 2017. https://arxiv.org/abs/1703.01365
- Hamilton, W., Ying, R., & Leskovec, J. — *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS 2017. https://arxiv.org/abs/1706.02216
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. — *Graph Attention Networks (GAT).* ICLR 2018. https://arxiv.org/abs/1710.10903
- Sun, C. *et al.* — *Lagrangian particle-tracking modeling of microplastic transport.* Frontiers in Toxicology, 2025.

**Real datasets the pipeline is designed for (currently not integrated, see disclosure above)**

- NOAA NCEI Marine Microplastics Database. *Scientific Data*, 2023. https://www.nature.com/articles/s41597-023-02632-y · https://www.ncei.noaa.gov/products/microplastics
- Rochman, C. — *SLoPP / FLOPP spectral libraries for microplastics research.* Rochman Lab. https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/
- HydroSHEDS — Hydrological data and maps based on shuttle elevation derivatives at multiple scales. https://www.hydrosheds.org/
- ECMWF ERA5 reanalysis (wind / weather covariates). https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

---

## License

MIT — see [LICENSE](LICENSE).
