# M4 — MicroPlastiNet Compliance Dashboard

[![Dashboard Screenshot](../../assets/dashboard_hero.png)](http://localhost:8050)

A Plotly Dash multi-tab monitoring dashboard for the MicroPlastiNet microplastic
detection network across coastal Georgia (Ogeechee, Savannah, and Altamaha rivers).

---

## Quick Start

```bash
# From project root
pip install -r src/m4_dashboard/requirements.txt

# Run in mock-data mode (default)
python src/m4_dashboard/app.py

# Run with real M2a/M2b/M3 outputs
MOCK_DATA=false python src/m4_dashboard/app.py

# Custom port
python src/m4_dashboard/app.py --port 8051

# Debug mode
python src/m4_dashboard/app.py --debug
```

Open http://127.0.0.1:8050 in your browser.

---

## Tabs

### Tab 1 — Watershed Map
- Mapbox/Plotly scatter map of 50 stations across 3 Georgia river systems
- Green (LOW) / Yellow (MEDIUM) / Red (HIGH) contamination dots
- Click any station → right side panel with full sensor readings
- KPI header bar: station counts, average concentration

### Tab 2 — Station Trends
- Select any station from dropdown
- 30-day MP concentration trend with turbidity overlay
- Automated anomaly spikes highlighted with red diamonds
- Anomaly log table below chart

### Tab 3 — Polymer Breakdown
- Donut pie chart: polymer composition at selected station
- Horizontal bar chart: classifier confidence per polymer class
- Stacked bar chart: polymer distribution across all 50 stations

### Tab 4 — Source Attribution *(headline tab)*
- Top 5 GNN-attributed upstream sources with probability bars
- Hydrological flow map showing source-to-station routing
- Embeds `assets/m3_graph.html` once M3 GNN module generates it

### Tab 5 — Forecast & Alerts
- SARIMA(1,0,1)(1,0,0,7) 7-day forecast per station
- Historical + forecast + 80% confidence interval
- Alert threshold line at 65 p/L
- High-alert station badges at top

### Tab 6 — Reports
- Select station + report mode (Template offline / OpenAI GPT-4o)
- Generate inline HTML-rendered regulator report
- Download as PDF or Markdown
- Calls M5 GenAI module (src/m5_genai/report_generator.py)

---

## File Structure

```
src/m4_dashboard/
├── app.py           ← Dash app entry point
├── data_loader.py   ← Data loading + mock generation
├── layout.py        ← All UI components, light theme
├── callbacks.py     ← All interactivity (Dash callbacks)
├── requirements.txt
└── README.md
```

---

## Data Mode

| Mode | Description |
|---|---|
| `MOCK_DATA=true` (default) | Generates reproducible synthetic data with realistic AR(1) time series, injected anomalies, and dirichlet polymer distributions |
| `MOCK_DATA=false` | Loads from M2a/M2b/M3 output files in `data/processed/` |

---

## Integration Points

| Module | Source | Data |
|---|---|---|
| M2a Vision | `data/processed/detections/` | Particle counts, shapes, sizes |
| M2b Spectral | `data/processed/polymer/` | Polymer fractions + confidence |
| M3 GNN | `src/m3_graph_gnn/outputs/attribution_results.json` | Source attribution probs |
| M5 GenAI | `src/m5_genai/report_generator.py` | Auto-generated report text |

---

## Design

- **Theme:** Agency-grade light (white surface, slate text, deep teal accent `#0f766e`)
- **Audience:** Environmental compliance officers and watershed regulators
- **Framework:** Plotly Dash + dash-bootstrap-components (FLATLY theme)
- **Map:** Plotly Scattermap with `carto-positron` tile layer
- **Font:** Inter (Google Fonts) + JetBrains Mono for KPI numbers
