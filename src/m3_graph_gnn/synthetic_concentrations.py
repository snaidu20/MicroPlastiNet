"""
synthetic_concentrations.py — Synthetic Microplastic Concentration Data Generator
===================================================================================
Simulates ~22,000 microplastic concentration records following NOAA NCEI Marine
Microplastics Database statistical patterns.

Distribution:
  - Log-normal (μ_ln ≈ 1.5, σ_ln ≈ 1.2), units: pieces/m³
  - Seasonal modulation (higher in spring/summer runoff events)
  - Concentrations at SAMPLING STATIONS are a noisy function of upstream
    SOURCE nodes, weighted by flow rate and inverse distance — this gives
    the GNN a genuine signal to learn.

Reference:
  - NOAA NCEI Marine Microplastics Database:
    https://www.ncei.noaa.gov/products/microplastics
  - Koelmans, A.A. et al. (2019). Microplastics in freshwaters and drinking
    water: Critical review and assessment of data quality. Water Research, 155,
    410-422. https://doi.org/10.1016/j.watres.2019.02.054
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Log-normal parameters calibrated to NOAA NCEI microplastic observations
# Typical freshwater: 0.4 – 7,000 pieces/m³ (median ~4 pieces/m³)
LN_MU = 1.5       # ln-space mean → exp(1.5) ≈ 4.5 pieces/m³ median
LN_SIGMA = 1.2    # ln-space std → covers ~3 orders of magnitude


def _seasonal_factor(day_of_year: int) -> float:
    """
    Seasonal modulation: spring (day 60-180) and late-summer storm runoff
    (day 200-260) drive higher concentrations.
    """
    t = day_of_year / 365.0 * 2 * math.pi
    # Primary peak: spring; secondary peak: late summer
    return 1.0 + 0.35 * math.sin(t - math.pi / 4) + 0.15 * math.sin(2 * t)


def _upstream_influence(
    station_id: int,
    source_ids: list,
    source_emission: np.ndarray,
    node_meta: dict,
    G,
    edge_data: dict,
    decay_km: float = 80.0,
) -> float:
    """
    Compute the concentration contribution at a sampling station from all
    upstream source nodes.

    Uses a flow-weighted inverse-distance propagation:
        conc = Σ_s  emission_s × flow_weight_s × exp(-dist_s / decay_km)

    where dist_s is the graph shortest-path distance (sum of edge distances).

    Parameters
    ----------
    station_id  : target sampling station node
    source_ids  : list of source node IDs
    source_emission : baseline emission rate per source node
    node_meta   : dict of node metadata
    G           : NetworkX DiGraph
    edge_data   : dict (u, v) → edge attributes
    decay_km    : exponential decay constant (km)
    """
    total = 0.0
    for i, src_id in enumerate(source_ids):
        try:
            # Use shortest path length (sum of distances) in the flow graph
            path_length = nx_shortest_path_distance(G, src_id, station_id)
            if path_length == float("inf"):
                continue
            # Flow weight: average flow rate along shortest path
            path = _get_shortest_path(G, src_id, station_id)
            if path is None:
                continue
            flow_weights = []
            for j in range(len(path) - 1):
                edata = G.edges[path[j], path[j + 1]]
                flow_weights.append(edata.get("flow_rate", 1.0))
            avg_flow = np.mean(flow_weights) if flow_weights else 1.0
            # Combine
            contribution = source_emission[i] * avg_flow * math.exp(-path_length / decay_km)
            total += contribution
        except Exception:
            continue
    return total


def nx_shortest_path_distance(G, src, dst) -> float:
    """Return sum-of-distances shortest path, or inf if unreachable."""
    try:
        import networkx as nx
        path = nx.shortest_path(G, src, dst, weight="distance")
        dist = sum(G.edges[path[i], path[i + 1]]["distance"] for i in range(len(path) - 1))
        return dist
    except Exception:
        return float("inf")


def _get_shortest_path(G, src, dst):
    """Return shortest path node list or None."""
    try:
        import networkx as nx
        return nx.shortest_path(G, src, dst, weight="distance")
    except Exception:
        return None


def generate_concentrations(
    data,           # PyG Data object from graph_builder
    node_meta: dict,
    G,
    n_records: int = 22000,
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    save_dir: str = None,
) -> pd.DataFrame:
    """
    Generate synthetic microplastic concentration time series.

    Returns
    -------
    df : pd.DataFrame
        Columns: timestamp, station_id, concentration_pieces_per_m3,
                 log_concentration, day_of_year, season_factor,
                 upstream_signal (ground truth), noise_level
    """
    import networkx as nx

    station_ids = data.station_ids   # list of 50 station node IDs
    source_ids = data.source_ids     # list of 100 source node IDs

    # ── Assign source baseline emissions (ground truth) ────────────────────
    # Each source has a "true" emission rate (pieces/m³ equivalent at 1 km)
    # Factories emit more; urban runoff moderate; agricultural low-moderate
    source_emission = np.zeros(len(source_ids))
    for i, src_id in enumerate(source_ids):
        ntype = node_meta[src_id]["node_type"]
        if ntype == "factory":
            source_emission[i] = np.random.lognormal(mean=2.5, sigma=0.8)  # ~12 peak
        elif ntype == "urban_runoff":
            source_emission[i] = np.random.lognormal(mean=1.5, sigma=0.6)  # ~4.5 peak
        else:  # agricultural
            source_emission[i] = np.random.lognormal(mean=0.8, sigma=0.5)  # ~2.2 peak

    # Save ground-truth emissions
    source_emission_dict = {
        int(source_ids[i]): float(source_emission[i])
        for i in range(len(source_ids))
    }

    # ── Pre-compute upstream signals for each station ──────────────────────
    # (expensive — do once and cache)
    print("Pre-computing upstream influence signals...")
    station_upstream_signal = {}
    for s_id in station_ids:
        sig = _upstream_influence(
            station_id=s_id,
            source_ids=source_ids,
            source_emission=source_emission,
            node_meta=node_meta,
            G=G,
            edge_data={},
            decay_km=80.0,
        )
        station_upstream_signal[s_id] = max(sig, 0.01)

    print(f"Upstream signals computed. Range: "
          f"{min(station_upstream_signal.values()):.3f} – "
          f"{max(station_upstream_signal.values()):.3f}")

    # ── Generate time series ───────────────────────────────────────────────
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    records = []
    n_per_station = n_records // len(station_ids)
    extra = n_records - n_per_station * len(station_ids)

    for s_idx, s_id in enumerate(station_ids):
        n_this = n_per_station + (1 if s_idx < extra else 0)
        # Sample random dates for this station
        chosen_dates = np.random.choice(date_range, size=n_this, replace=True)
        chosen_dates = sorted(pd.DatetimeIndex(chosen_dates))

        base_signal = station_upstream_signal[s_id]

        for dt in chosen_dates:
            day_of_year = dt.dayofyear
            sf = _seasonal_factor(day_of_year)
            # Log-normal base concentration
            base_conc = np.random.lognormal(mean=LN_MU, sigma=LN_SIGMA)
            # Add upstream signal (scaled so signal matters but noise exists)
            signal_contribution = base_signal * sf * 0.3
            # Storm event: occasional 3-10× spike (5% probability)
            storm_multiplier = np.random.choice(
                [1.0, np.random.uniform(3, 10)],
                p=[0.95, 0.05]
            )
            # Final concentration
            conc = (base_conc + signal_contribution) * storm_multiplier
            conc = max(conc, 0.01)  # physical floor

            records.append({
                "timestamp": dt,
                "station_id": s_id,
                "concentration_pieces_per_m3": round(conc, 4),
                "log_concentration": round(math.log(conc), 4),
                "day_of_year": day_of_year,
                "season_factor": round(sf, 4),
                "upstream_signal": round(base_signal, 4),
                "storm_event": storm_multiplier > 1.0,
                "noise_level": round(base_conc / (base_signal + 1e-6), 4),
            })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    print(f"\nConcentration dataset: {len(df)} records")
    print(f"  Mean:   {df['concentration_pieces_per_m3'].mean():.3f} pieces/m³")
    print(f"  Median: {df['concentration_pieces_per_m3'].median():.3f} pieces/m³")
    print(f"  Max:    {df['concentration_pieces_per_m3'].max():.3f} pieces/m³")
    print(f"  Storm events: {df['storm_event'].sum()} ({df['storm_event'].mean()*100:.1f}%)")

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save main dataframe
        df.to_csv(save_path / "concentrations.csv", index=False)
        print(f"Saved concentration data → {save_path / 'concentrations.csv'}")

        # Save ground-truth emissions
        with open(save_path / "source_emissions_ground_truth.json", "w") as f:
            json.dump(source_emission_dict, f, indent=2)
        print(f"Saved ground-truth emissions → {save_path / 'source_emissions_ground_truth.json'}")

        # Train/val/test temporal split
        # Train: 2015-2020 (75%), Val: 2021-2022 (15%), Test: 2023 (10%)
        train_df = df[df["timestamp"].dt.year <= 2020]
        val_df = df[(df["timestamp"].dt.year >= 2021) & (df["timestamp"].dt.year <= 2022)]
        test_df = df[df["timestamp"].dt.year >= 2023]

        train_df.to_csv(save_path / "train.csv", index=False)
        val_df.to_csv(save_path / "val.csv", index=False)
        test_df.to_csv(save_path / "test.csv", index=False)

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return df, source_emission_dict


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from graph_builder import build_graph

    data, node_meta, G = build_graph(
        save_dir="/home/user/workspace/MicroPlastiNet/data/processed/m3"
    )

    df, emissions = generate_concentrations(
        data=data,
        node_meta=node_meta,
        G=G,
        n_records=22000,
        save_dir="/home/user/workspace/MicroPlastiNet/data/processed/m3",
    )

    print("\nSample records:")
    print(df.head())
    print("\nSource emission sample (first 5 sources):")
    for k, v in list(emissions.items())[:5]:
        print(f"  Node {k}: {v:.3f} emission units")
