# MicroPlastiNet — Architecture

## Pipeline Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                       FIELD SITE — coastal Georgia                  │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │  M1 — MPN-Edge node ($30, weatherproof)                 │      │
│   │  ┌──────────┐  ┌────────┐  ┌──────┐  ┌─────────────┐   │      │
│   │  │ESP32-CAM │  │Turbidity│  │ TDS  │  │AS7265x NIR  │   │      │
│   │  │          │  │SEN0189  │  │probe │  │(6 channels) │   │      │
│   │  └────┬─────┘  └────┬───┘  └──┬───┘  └──────┬──────┘   │      │
│   │       └────────┬────┴─────────┴─────────────┘          │      │
│   │                ▼                                        │      │
│   │       ┌────────────────────┐                           │      │
│   │       │ TFLite-Micro detector (logistic gate emulation)│      │
│   │       └─────────┬──────────┘                           │      │
│   │                 ▼                                       │      │
│   │       ┌─────────────────────┐                          │      │
│   │       │ M6 sign payload     │  HMAC-SHA256 + nonce     │      │
│   │       │ (per-station secret)│                          │      │
│   │       └─────────┬───────────┘                          │      │
│   └─────────────────│──────────────────────────────────────┘      │
└─────────────────────│──────────────────────────────────────────────┘
                      │   MQTT/TLS 1.3
                      ▼
┌────────────────────────────────────────────────────────────────────┐
│                              CLOUD                                  │
│                                                                     │
│   ┌────────────────────────────────────────────────────────┐      │
│   │  M1 cloud_listener  →  verify HMAC + freshness + nonce │      │
│   └─────────────────────┬──────────────────────────────────┘      │
│                         ▼                                          │
│   ┌──────────────────────────────────┐ ┌──────────────────────┐  │
│   │  M2a — Vision DL                  │ │  M2b — Spectral CNN  │  │
│   │  • EfficientNet-B0 classifier     │ │  • 1D-CNN, 471k pp   │  │
│   │  • TinyYOLO detector              │ │  • 6-class polymer   │  │
│   │  → counts, sizes, shapes          │ │  → PE/PET/PP/PS/PVC  │  │
│   └────────────────┬──────────────────┘ └─────────┬────────────┘  │
│                    └────────────┬─────────────────┘                │
│                                 ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  M3 — GNN on hydrological flow graph (200 nodes)         │    │
│   │  • GraphSAGE concentration regression  (R² = 0.96)       │    │
│   │  • GAT  (interpretable attention)                        │    │
│   │  • Classical centrality+Ridge baseline                   │    │
│   │  • Integrated Gradients → upstream source ranking        │    │
│   └────────────────┬────────────────────────────────────────┘    │
│                    ▼                                               │
│   ┌──────────────────────────────────┐ ┌──────────────────────┐  │
│   │  M4 — Dashboard (Plotly Dash)    │ │  M5 — GenAI reports  │  │
│   │  6 tabs · watershed map · forecasts │ │ Template + OpenAI │  │
│   │  source-attribution panel        │ │  PDF + Markdown      │  │
│   └──────────────────────────────────┘ └──────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## Data flow per timestep (1 station, 1 reading)

```
t = 0.000s   River water enters the chamber
t = 0.001s   Camera + 4 sensors fire, raw values acquired
t = 0.002s   Edge detector decides (suspicious, 0.78)
t = 0.003s   Payload assembled, HMAC signed, nonce stamped
t = 0.5s     Payload arrives at broker (TLS handshake amortized)
t = 0.6s     Cloud listener verifies, queues to M2a + M2b
t = 1.5s     M2a returns: 11 particles, mean size 0.7 mm
t = 1.9s     M2b returns: PET 0.81 / PE 0.12 / other 0.07
t = 2.4s     M3 attribution: top source urban_runoff_026 @ 0.68
t = 2.5s     M4 dashboard updates Station 5 to RED
t = 2.6s     M5 starts drafting regulator report (async)
```

## Why each design choice

| Decision | Reason |
|---|---|
| Synthetic data first | Lets every module run end-to-end on a laptop while leaving real-data slots open in `data/raw/` |
| Multi-modal sensor fusion | A camera alone cannot ID polymer type; NIR adds the chemistry |
| Per-station HMAC keys | Compromise of one station does not compromise the network |
| GraphSAGE *and* GAT | SAGE wins R²; GAT wins interpretability — we ship both |
| Integrated Gradients | Axiomatic attribution method; gradient-theoretic analogue of transfer-entropy / effective-connectivity analysis on graphs |
| Coastal Georgia geography | Real coordinates anchor the project in a realistic deployment region with mixed industrial, urban, and agricultural pollution sources |
