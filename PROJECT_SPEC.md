# MicroPlastiNet — Master Project Specification

## Purpose
Multi-modal IoT + Deep Learning + Graph ML pipeline for microplastic detection, classification, and source attribution.

## Architecture (4 Core Modules + 2 Add-ons)

### M1 — IoT Edge (Simulated)
- ESP32-CAM + turbidity (SEN0189) + TDS + NIR sensor fusion (simulated in software)
- On-device TFLite-Micro first-pass detector
- Encrypted MQTT publishing with TLS + HMAC integrity hash
- Output: JSON payloads to cloud broker

### M2a — Vision DL Classifier
- YOLOv8 fine-tuned for microplastic detection/segmentation
- EfficientNet-B0 backbone for shape classification (fragment/fiber/film/bead/foam)
- Trained on: Kaggle Microplastic CV dataset, MP-Set fluorescence dataset
- Output: particle counts, sizes, shape categories, confidence

### M2b — Spectral DL Classifier
- 1D-CNN on FTIR + Raman spectra
- Trained on: Rochman Lab SLoPP/SLoPP-E (Raman 343 spectra), FLOPP/FLOPP-e (FTIR 381 spectra)
- 6-class polymer classification: PE, PET, PP, PS, PVC, Other
- Output: polymer probabilities + confidence

### M3 — Graph Neural Network (Source Attribution)
- GraphSAGE / GAT on hydrological flow graph
- Nodes: sampling sites + candidate sources (factories, urban runoff, river junctions)
- Edges: hydrological flow (HydroSHEDS) + wind transport (ERA5)
- Trained on: NOAA NCEI Marine Microplastics Database (~22k records)
- Inversion via gradient-based attribution → source ranking

### M4 — Compliance Dashboard
- Plotly Dash / Streamlit
- Watershed map with sensor stations
- Station trends, polymer breakdown, source attribution panel
- Forecast & alerts (concentration spikes 24-48 hr forecast)
- PDF/CSV export for regulators

### M5 — GenAI Report Generator (Add-on)
- LLM (OpenAI / local Llama) prompt-engineered pipeline
- Auto-generates plain-language regulator reports from sensor + graph outputs
- Integrates remediation suggestions

### M6 — Cybersecurity Layer (Add-on)
- TLS 1.3 for MQTT
- HMAC-SHA256 integrity hashes per payload
- Lightweight on-device key rotation
- Replay-attack prevention via nonces + timestamps

## Real Datasets Used (All Open-Access)
| Dataset | Use | URL |
|---|---|---|
| NOAA NCEI Marine Microplastics DB | M3 ground truth (~22k records) | https://www.ncei.noaa.gov/products/microplastics |
| Rochman SLoPP/SLoPP-E (Raman) | M2b training | https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/ |
| Rochman FLOPP/FLOPP-e (FTIR) | M2b training | https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/ |
| Kaggle Microplastic CV | M2a vision | https://www.kaggle.com/code/mathieuduverne/microplastic-detection-yolov8-map-50-76-2 |
| MP-Set fluorescence | M2a vision | https://www.kaggle.com/datasets/sanghyeonaustinpark/mpset |
| HydroSHEDS river network | M3 graph topology | https://www.hydrosheds.org/ |
| ERA5 wind/weather | M3 covariates | https://www.ecmwf.int/ |

## Tech Stack
- Python 3.11
- PyTorch, PyTorch Geometric (GNNs)
- Ultralytics YOLOv8
- TensorFlow Lite Micro (M1 simulation)
- Plotly Dash for M4
- Paho-MQTT, cryptography for M6
- OpenAI / Hugging Face for M5

## Repository Layout
```
MicroPlastiNet/
├── README.md
├── PROJECT_SPEC.md            (this file)
├── data/
│   ├── raw/                   (downloaded datasets)
│   └── processed/             (cleaned, split)
├── src/
│   ├── m1_iot_edge/
│   ├── m2a_vision/
│   ├── m2b_spectral/
│   ├── m3_graph_gnn/
│   ├── m4_dashboard/
│   ├── m5_genai/
│   ├── m6_security/
│   └── common/
├── notebooks/                 (EDA, results)
├── docs/                      (architecture, methodology, GitHub Pages site)
├── assets/                    (screenshots, diagrams)
├── configs/                   (YAML configs per module)
├── scripts/                   (data download, training, eval)
└── tests/
```

## Honesty Principles
- Report confidence intervals, not point estimates
- Distinguish field-grade vs lab-grade accuracy
- Document failure modes and limits
- Use real data; mark all simulated portions explicitly
