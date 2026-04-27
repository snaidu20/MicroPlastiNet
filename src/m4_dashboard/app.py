"""
app.py — MicroPlastiNet M4 Compliance Dashboard

Plotly Dash dashboard for environmental compliance teams.
Visualizes microplastic contamination across coastal Georgia river systems.

Usage:
    python app.py [--port 8050] [--debug]
    MOCK_DATA=false python app.py   # use real M2a/M2b/M3 outputs

Tabs:
    1. Watershed Map      — 50 sensor stations, color-coded contamination
    2. Station Trends     — 30-day concentrations with anomaly highlights
    3. Polymer Breakdown  — pie + confidence + stacked bar
    4. Source Attribution — GNN-derived upstream sources with probability bars
    5. Forecast & Alerts  — 7-day SARIMA forecast per station
    6. Reports            — GenAI regulator report generation (M5 integration)
"""

import sys
import os

# Make local imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import dash
import dash_bootstrap_components as dbc

from data_loader import load_station_metadata, MOCK_DATA
from layout import make_layout
from callbacks import register_callbacks

# ─── App Initialization ────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    title="MicroPlastiNet Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
    assets_folder=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets"
    ),
)
server = app.server

# ─── Load stations for dropdown population ─────────────────────────────────────
df_stations = load_station_metadata()
station_options = [
    {"label": f"{r['station_id']} — {r['name']}", "value": r["station_id"]}
    for _, r in df_stations.iterrows()
]

# ─── Layout ────────────────────────────────────────────────────────────────────
app.layout = make_layout(station_options)

# ─── Register all callbacks ────────────────────────────────────────────────────
register_callbacks(app)

# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroPlastiNet Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    mode_label = "DEMO" if MOCK_DATA else "REAL"
    print(f"\n{'='*60}")
    print(f"  MicroPlastiNet Dashboard — {mode_label} DATA")
    print(f"  http://127.0.0.1:{args.port}")
    print(f"  Stations loaded: {len(df_stations)}")
    print(f"{'='*60}\n")

    app.run(
        debug=args.debug,
        host="0.0.0.0",
        port=args.port,
        use_reloader=False,
    )
