"""
data_loader.py — MicroPlastiNet M4 Dashboard Data Loader

MOCK_DATA=True  → generates realistic synthetic data (default, no upstream deps)
MOCK_DATA=False → loads from M2a/M2b/M3 outputs
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
MOCK_DATA = os.environ.get("MOCK_DATA", "true").lower() != "false"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = BASE_DIR / "assets"
M3_OUTPUT = BASE_DIR / "src" / "m3_graph_gnn" / "outputs" / "attribution_results.json"

# ─── Station Metadata ──────────────────────────────────────────────────────────
# 50 stations across Ogeechee, Savannah, Altamaha river corridors in coastal Georgia

RIVER_SYSTEMS = {
    "Ogeechee": {
        "color": "#0284c7",
        "lat_range": (31.9, 32.6),
        "lon_range": (-81.8, -81.0),
        "n_stations": 17,
    },
    "Savannah": {
        "color": "#ea580c",
        "lat_range": (32.0, 32.8),
        "lon_range": (-81.2, -80.9),
        "n_stations": 16,
    },
    "Altamaha": {
        "color": "#16a34a",
        "lat_range": (31.3, 31.9),
        "lon_range": (-81.7, -81.1),
        "n_stations": 17,
    },
}

POLYMER_TYPES = ["PE", "PET", "PP", "PS", "PVC", "Other"]
# Color palette reference (also used by callbacks)
COLORS = {
    "bg_deep":     "#f5f7fa",
    "bg_panel":    "#ffffff",
    "bg_card":     "#ffffff",
    "accent_cyan": "#0284c7",
    "accent_teal": "#0d9488",
    "accent_amber":"#d97706",
    "accent_red":  "#dc2626",
    "accent_green":"#16a34a",
    "text_primary":"#0f172a",
    "text_muted":  "#64748b",
    "border":      "#e2e8f0",
    "high":        "#dc2626",
    "medium":      "#d97706",
    "low":         "#16a34a",
}

POLYMER_COLORS = {
    "PE":    "#0284c7",  # blue
    "PET":   "#ea580c",  # orange
    "PP":    "#d97706",  # amber
    "PS":    "#7c3aed",  # violet
    "PVC":   "#dc2626",  # red
    "Other": "#0d9488",  # teal
}

SOURCE_TYPES = [
    "Upstream Wastewater Outfall",
    "Urban Stormwater Runoff",
    "Agricultural Drainage",
    "Industrial Discharge",
    "Marine Vessel Traffic",
    "Atmospheric Deposition",
    "Coastal Erosion",
]


def _seed_rng(seed=42):
    return np.random.default_rng(seed)


def load_station_metadata() -> pd.DataFrame:
    """Return DataFrame of 50 sensor stations with lat/lon, river, status."""
    if not MOCK_DATA:
        meta_path = BASE_DIR / "data" / "processed" / "station_metadata.csv"
        if meta_path.exists():
            return pd.read_csv(meta_path)

    rng = _seed_rng(42)
    records = []
    station_id = 1

    for river, cfg in RIVER_SYSTEMS.items():
        n = cfg["n_stations"]
        lats = rng.uniform(*cfg["lat_range"], n)
        lons = rng.uniform(*cfg["lon_range"], n)
        for i in range(n):
            # Assign contamination level (determines dot color on map)
            base_level = rng.uniform(0, 100)
            status = "HIGH" if base_level > 66 else ("MEDIUM" if base_level > 33 else "LOW")
            records.append({
                "station_id":   f"STN-{station_id:03d}",
                "name":         f"{river} Stn {i+1}",
                "river":        river,
                "lat":          round(float(lats[i]), 5),
                "lon":          round(float(lons[i]), 5),
                "status":       status,
                "mp_conc":      round(float(base_level), 2),   # particles/L
                "temp_c":       round(float(rng.uniform(18, 28)), 1),
                "turbidity_ntu":round(float(rng.uniform(1, 45)), 1),
                "ph":           round(float(rng.uniform(6.5, 8.2)), 2),
                "depth_m":      round(float(rng.uniform(0.3, 4.5)), 1),
                "install_date": f"202{rng.integers(1, 4)}-{rng.integers(1,12):02d}-{rng.integers(1,28):02d}",
                "color":        cfg["color"],
            })
            station_id += 1

    return pd.DataFrame(records)


def load_time_series(station_id: str, days: int = 30) -> pd.DataFrame:
    """Return daily MP concentration time series for a station."""
    if not MOCK_DATA:
        ts_path = BASE_DIR / "data" / "processed" / "timeseries" / f"{station_id}.csv"
        if ts_path.exists():
            return pd.read_csv(ts_path, parse_dates=["date"])

    rng = _seed_rng(sum(ord(c) for c in station_id))
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days, -1, -1)]

    # Generate AR(1) process with seasonal component
    base = float(rng.uniform(10, 60))
    values = [base]
    for d in dates[1:]:
        seasonal = 8 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365)
        noise = float(rng.normal(0, 3))
        new_val = max(0.1, 0.88 * values[-1] + 0.12 * base + seasonal + noise)
        values.append(new_val)

    # Inject 1-2 anomaly spikes
    n_anomalies = rng.integers(1, 3)
    anomaly_idx = rng.choice(range(5, len(values) - 2), n_anomalies, replace=False)
    anomaly_flags = [False] * len(values)
    for idx in anomaly_idx:
        values[idx] += float(rng.uniform(30, 70))
        anomaly_flags[idx] = True

    return pd.DataFrame({
        "date":      pd.to_datetime(dates),
        "mp_conc":   [round(v, 2) for v in values],
        "turbidity": [round(v + float(rng.normal(0, 2)), 2) for v in values],
        "anomaly":   anomaly_flags,
    })


def load_polymer_breakdown(station_id: str) -> dict:
    """Return polymer type distribution for a station."""
    if not MOCK_DATA:
        poly_path = BASE_DIR / "data" / "processed" / "polymer" / f"{station_id}.json"
        if poly_path.exists():
            with open(poly_path) as f:
                return json.load(f)

    rng = _seed_rng(sum(ord(c) for c in station_id) + 1000)
    raw = rng.dirichlet(alpha=[3, 2, 2, 1, 1, 1])
    return {
        "station_id": station_id,
        "polymers": {p: round(float(v), 4) for p, v in zip(POLYMER_TYPES, raw)},
        "confidence": {p: round(float(rng.uniform(0.72, 0.98)), 3) for p in POLYMER_TYPES},
        "total_particles": int(rng.integers(120, 2400)),
    }


def load_source_attribution(station_id: str, event_id: str = None) -> dict:
    """Return top-5 source attribution for a contamination event."""
    if not MOCK_DATA and M3_OUTPUT.exists():
        with open(M3_OUTPUT) as f:
            data = json.load(f)
        if station_id in data:
            return data[station_id]

    rng = _seed_rng(sum(ord(c) for c in station_id) + 9999)
    n_sources = 5
    chosen = rng.choice(SOURCE_TYPES, n_sources, replace=False)
    probs_raw = rng.dirichlet(alpha=[4, 2.5, 2, 1.5, 1])
    probs = sorted(zip(probs_raw, chosen), reverse=True)

    # Realistic upstream source locations
    source_lats = rng.uniform(31.8, 33.2, n_sources)
    source_lons = rng.uniform(-82.5, -81.0, n_sources)

    return {
        "station_id":   station_id,
        "event_id":     event_id or f"EVT-{rng.integers(1000, 9999)}",
        "event_date":   (datetime.now() - timedelta(days=int(rng.integers(1, 10)))).strftime("%Y-%m-%d"),
        "sources": [
            {
                "rank":        i + 1,
                "name":        name,
                "probability": round(float(prob), 4),
                "confidence":  round(float(rng.uniform(0.7, 0.97)), 3),
                "distance_km": round(float(rng.uniform(2, 45)), 1),
                "lat":         round(float(source_lats[i]), 5),
                "lon":         round(float(source_lons[i]), 5),
            }
            for i, (prob, name) in enumerate(probs)
        ],
    }


def load_all_polymer_breakdown() -> pd.DataFrame:
    """Return polymer breakdown for ALL stations (for stacked bar chart)."""
    stations = load_station_metadata()
    records = []
    for sid in stations["station_id"]:
        pb = load_polymer_breakdown(sid)
        row = {"station_id": sid}
        row.update(pb["polymers"])
        records.append(row)
    return pd.DataFrame(records)


def load_forecast(station_id: str, days_ahead: int = 7) -> pd.DataFrame:
    """
    Generate 7-day forecast using statsmodels SARIMA (or simple AR fallback).
    Returns DataFrame with date, predicted_conc, lower_ci, upper_ci, alert.
    """
    ts = load_time_series(station_id, days=60)

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                ts["mp_conc"],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 0, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=50)
            forecast = result.get_forecast(steps=days_ahead)
            pred = forecast.predicted_mean.values
            ci = forecast.conf_int(alpha=0.2)
            lower = ci.iloc[:, 0].values
            upper = ci.iloc[:, 1].values
    except Exception:
        # Fallback: simple AR(1)-like extrapolation
        last_val = float(ts["mp_conc"].iloc[-1])
        rng = _seed_rng(hash(station_id) % 10000)
        pred = [max(0, last_val + float(rng.normal(0, 5))) for _ in range(days_ahead)]
        lower = [max(0, v - 15) for v in pred]
        upper = [v + 20 for v in pred]

    end_date = datetime.now()
    future_dates = [end_date + timedelta(days=i + 1) for i in range(days_ahead)]

    # Threshold: HIGH alert if predicted > 65 particles/L
    alert_threshold = 65.0
    return pd.DataFrame({
        "date":       pd.to_datetime(future_dates),
        "predicted":  [round(float(v), 2) for v in pred],
        "lower_ci":   [round(float(v), 2) for v in lower],
        "upper_ci":   [round(float(v), 2) for v in upper],
        "alert":      [float(v) > alert_threshold for v in pred],
    })


def get_m3_graph_html() -> str | None:
    """Return path to M3 interactive graph HTML if available."""
    graph_html = ASSETS_DIR / "m3_graph.html"
    if graph_html.exists():
        return str(graph_html)
    return None


def get_map_token() -> str:
    """Return Mapbox token or empty string for open-street-map fallback."""
    return os.environ.get("MAPBOX_TOKEN", "")
